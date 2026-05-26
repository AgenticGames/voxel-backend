//! Scene type — semantically-clustered POI with sub-voxel centroid, AABB,
//! confidence, age, event history ring buffer, and tags.

use glam::Vec3;
use serde::{Deserialize, Serialize};

/// Stable Scene identifier. Monotonically allocated by `WorldMemory`;
/// persisted across save/load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SceneId(pub u64);

/// Scene kind. Numeric values are stable across the FFI boundary —
/// 0..=3 match the existing `PoiKind` values in voxel-ffi/src/poi_scanner.rs
/// (Lava, Water, Stress, Bridge); 4..=6 match the topology kinds documented
/// in voxel-ffi/src/types.rs near FfiPoi (CeilingDome, Chokepoint, WallNiche).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SceneKind {
    Lava = 0,
    Water = 1,
    Stress = 2,
    Bridge = 3,
    CeilingDome = 4,
    Chokepoint = 5,
    WallNiche = 6,
}

impl SceneKind {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Lava),
            1 => Some(Self::Water),
            2 => Some(Self::Stress),
            3 => Some(Self::Bridge),
            4 => Some(Self::CeilingDome),
            5 => Some(Self::Chokepoint),
            6 => Some(Self::WallNiche),
            _ => None,
        }
    }

    /// Returns true for the four topology kinds gated by
    /// `enable_topology_scenes` in Block 1 (UE doesn't know these yet).
    pub fn is_topology(self) -> bool {
        matches!(
            self,
            Self::CeilingDome | Self::Chokepoint | Self::WallNiche
        )
    }
}

/// Axis-aligned bounding box in **Rust voxel coords** (Y-up). FFI layer
/// converts to UE world units before exposing.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Aabb {
    pub fn empty() -> Self {
        Self {
            min: [f32::INFINITY; 3],
            max: [f32::NEG_INFINITY; 3],
        }
    }

    pub fn point(p: Vec3) -> Self {
        Self {
            min: [p.x, p.y, p.z],
            max: [p.x, p.y, p.z],
        }
    }

    pub fn extend_point(&mut self, p: Vec3) {
        self.min[0] = self.min[0].min(p.x);
        self.min[1] = self.min[1].min(p.y);
        self.min[2] = self.min[2].min(p.z);
        self.max[0] = self.max[0].max(p.x);
        self.max[1] = self.max[1].max(p.y);
        self.max[2] = self.max[2].max(p.z);
    }

    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }

    pub fn center(&self) -> Vec3 {
        Vec3::new(
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
            0.5 * (self.min[2] + self.max[2]),
        )
    }

    /// Half-extent diagonal length, in voxels. Used as the "extent radius"
    /// in the legacy adapter — strictly better than the per-chunk
    /// `cs_f * 0.5` fallback in voxel-ffi/src/poi_scanner.rs:256.
    pub fn extent_radius(&self) -> f32 {
        let hx = 0.5 * (self.max[0] - self.min[0]);
        let hy = 0.5 * (self.max[1] - self.min[1]);
        let hz = 0.5 * (self.max[2] - self.min[2]);
        (hx * hx + hy * hy + hz * hz).sqrt()
    }
}

/// A single event recorded in a Scene's history ring buffer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SceneHistoryEvent {
    /// Tag id (interpretation depends on Scene kind). Common: 0=created,
    /// 1=refreshed-via-scan, 2=event-promoted, 3=cluster-merged.
    pub tag: u8,
    /// Seconds since `WorldMemory.start_instant`.
    pub at_secs: u32,
}

/// Maximum history events per Scene (ring buffer cap).
pub const SCENE_HISTORY_CAP: usize = 16;

/// Tag bitmask — semantic flags attached to a Scene. Bits are free-form;
/// callers agree on conventions. Examples in Block 1:
///   - 0x01 = "fresh" (created within last 60 s)
///   - 0x02 = "player-placed" (bridge anchor / brush originated)
///   - 0x04 = "natural" (worldgen/drift originated)
///   - 0x08 = "sleep-evolved" (modified during sleep cycle)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SceneTags(pub u64);

impl SceneTags {
    pub const FRESH: SceneTags = SceneTags(0x01);
    pub const PLAYER_PLACED: SceneTags = SceneTags(0x02);
    pub const NATURAL: SceneTags = SceneTags(0x04);
    pub const SLEEP_EVOLVED: SceneTags = SceneTags(0x08);

    pub fn set(&mut self, tag: SceneTags) {
        self.0 |= tag.0;
    }
    pub fn clear(&mut self, tag: SceneTags) {
        self.0 &= !tag.0;
    }
    pub fn has(self, tag: SceneTags) -> bool {
        (self.0 & tag.0) != 0
    }
}

/// A scene — the unit of "interesting thing in the world" surfaced to the
/// sleep montage and other consumers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    pub id: SceneId,
    pub kind: SceneKind,
    /// Score-weighted sub-voxel centroid (Rust coords).
    pub centroid: [f32; 3],
    pub aabb: Aabb,
    pub score: f32,
    /// Confidence ∈ [0,1] — how many independent signals concur this Scene
    /// is real. A single-chunk Scene with one detection is ~0.3; multi-chunk
    /// drift-confirmed-plus-event-confirmed is ~1.0.
    pub confidence: f32,
    /// Age in seconds since first observation.
    pub age_secs: u32,
    /// Seconds since last score-refresh (drift tick or push event).
    pub last_seen_secs: u32,
    /// Chunk coords belonging to this Scene (in Rust chunk coord space).
    pub chunks: Vec<(i32, i32, i32)>,
    /// Ring buffer of recent history events. Newest at the end.
    pub history: Vec<SceneHistoryEvent>,
    pub tags: SceneTags,
}

impl Scene {
    pub fn new(id: SceneId, kind: SceneKind, centroid: Vec3) -> Self {
        Self {
            id,
            kind,
            centroid: [centroid.x, centroid.y, centroid.z],
            aabb: Aabb::point(centroid),
            score: 0.0,
            confidence: 0.0,
            age_secs: 0,
            last_seen_secs: 0,
            chunks: Vec::new(),
            history: Vec::new(),
            tags: SceneTags::default(),
        }
    }

    pub fn centroid_vec(&self) -> Vec3 {
        Vec3::new(self.centroid[0], self.centroid[1], self.centroid[2])
    }

    /// Append to the history ring buffer, capped at SCENE_HISTORY_CAP.
    pub fn record_history(&mut self, tag: u8, at_secs: u32) {
        if self.history.len() >= SCENE_HISTORY_CAP {
            self.history.remove(0);
        }
        self.history.push(SceneHistoryEvent { tag, at_secs });
    }
}

/// Filter for `WorldMemory::scenes` queries.
#[derive(Debug, Clone, Copy)]
pub struct SceneFilter {
    /// Bitmask of allowed kinds. Bit N set ⇒ kind with discriminant N
    /// passes. `0xFF` = all kinds.
    pub kind_mask: u32,
    /// Minimum score required.
    pub min_score: f32,
    /// Minimum confidence required.
    pub min_confidence: f32,
    /// Include topology kinds (CeilingDome/Chokepoint/WallNiche)? Block 1
    /// default is false — UE doesn't handle them yet.
    pub include_topology: bool,
}

impl SceneFilter {
    pub fn all() -> Self {
        Self {
            kind_mask: 0xFFFFFFFF,
            min_score: 0.0,
            min_confidence: 0.0,
            include_topology: false,
        }
    }

    pub fn matches(&self, scene: &Scene) -> bool {
        let kind_bit = 1u32 << (scene.kind as u8 as u32);
        if (self.kind_mask & kind_bit) == 0 {
            return false;
        }
        if scene.score < self.min_score {
            return false;
        }
        if scene.confidence < self.min_confidence {
            return false;
        }
        if !self.include_topology && scene.kind.is_topology() {
            return false;
        }
        true
    }
}

impl Default for SceneFilter {
    fn default() -> Self {
        Self::all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aabb_extents() {
        let mut bb = Aabb::point(Vec3::new(0.0, 0.0, 0.0));
        bb.extend_point(Vec3::new(10.0, 5.0, 2.0));
        assert_eq!(bb.center(), Vec3::new(5.0, 2.5, 1.0));
        // diagonal 5,2.5,1 → sqrt(25+6.25+1) = sqrt(32.25) ≈ 5.679
        assert!((bb.extent_radius() - 5.6789).abs() < 0.01);
    }

    #[test]
    fn scene_kind_topology_flag() {
        assert!(SceneKind::CeilingDome.is_topology());
        assert!(SceneKind::Chokepoint.is_topology());
        assert!(SceneKind::WallNiche.is_topology());
        assert!(!SceneKind::Lava.is_topology());
        assert!(!SceneKind::Bridge.is_topology());
    }

    #[test]
    fn scene_filter_excludes_topology_by_default() {
        let mut s_lava = Scene::new(SceneId(1), SceneKind::Lava, Vec3::ZERO);
        s_lava.score = 100.0;
        s_lava.confidence = 0.8;
        let mut s_dome = Scene::new(SceneId(2), SceneKind::CeilingDome, Vec3::ZERO);
        s_dome.score = 100.0;
        s_dome.confidence = 0.8;

        let mut f = SceneFilter::all();
        assert!(f.matches(&s_lava));
        assert!(!f.matches(&s_dome));
        f.include_topology = true;
        assert!(f.matches(&s_dome));
    }

    #[test]
    fn history_ring_buffer_caps() {
        let mut s = Scene::new(SceneId(1), SceneKind::Lava, Vec3::ZERO);
        for i in 0..(SCENE_HISTORY_CAP + 5) {
            s.record_history(0, i as u32);
        }
        assert_eq!(s.history.len(), SCENE_HISTORY_CAP);
        // Oldest 5 dropped — first remaining entry's at_secs is 5.
        assert_eq!(s.history[0].at_secs, 5);
        assert_eq!(s.history.last().unwrap().at_secs, (SCENE_HISTORY_CAP + 4) as u32);
    }

    #[test]
    fn scene_tags_set_and_check() {
        let mut tags = SceneTags::default();
        assert!(!tags.has(SceneTags::FRESH));
        tags.set(SceneTags::FRESH);
        tags.set(SceneTags::NATURAL);
        assert!(tags.has(SceneTags::FRESH));
        assert!(tags.has(SceneTags::NATURAL));
        assert!(!tags.has(SceneTags::PLAYER_PLACED));
        tags.clear(SceneTags::FRESH);
        assert!(!tags.has(SceneTags::FRESH));
        assert!(tags.has(SceneTags::NATURAL));
    }
}
