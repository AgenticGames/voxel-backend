//! Sleep-time POI scanner: scores loaded chunks by geothermal/structural
//! "interestingness" and stashes top-K candidates for the sleep montage to
//! orbit. Runs synchronously inside the Sleep handler after geological-time
//! advancement, so the snapshot reflects the *post-sleep* world.
//!
//! **Unified candidate pool** — bridges, lava, water, and stress all
//! compete for the same 3 montage slots. Scoring (per-chunk where relevant):
//!   - lava_voxels    × 10.0    (most striking on stream)
//!   - water_voxels   × 6.0
//!   - stress_voxels  × 8.0     (high-stress connected mass)
//!   - bridge baseline 80.0     (per pair — moderate; outscored by big
//!                               geological events but solid enough that
//!                               a quiet sleep cycle still showcases them)
//!
//! Ore formation is covered transitively: a freshly enriched ore vein is
//! always co-located with the geological event that produced it (hydrothermal
//! water, lava aureole, or stress fracture), so the corresponding water/lava/
//! stress POI already orbits the right spot.
//!
//! Connected-mass detection is approximated by raw voxel count above a
//! threshold per chunk — a real flood-fill across chunks would be nicer
//! but isn't needed for the top-3 cap.
//!
//! UE pulls the cached top-K via `voxel_request_list_top_pois` and pins
//! the chunks during the montage POI rotation.

use std::sync::{Arc, Mutex};

use voxel_fluid::FluidSnapshot;

use crate::crystal_anchors::CrystalAnchorManager;
use crate::store::ChunkStore;

/// Stress threshold above which a voxel counts toward the "high stress
/// connected" score. Tuned conservatively — most terrain sits well below.
pub const STRESS_HIGH_THRESHOLD: f32 = 0.6;

/// Minimum fluid level per cell to count toward water/lava score.
pub const FLUID_MIN_LEVEL: f32 = 0.10;

/// Per-chunk vote thresholds — chunks need at least this many qualifying
/// voxels of a kind to register as a POI candidate. Filters noise from
/// any single isolated voxel.
pub const MIN_LAVA_VOTES: usize = 16;
pub const MIN_WATER_VOTES: usize = 24;
pub const MIN_STRESS_VOTES: usize = 32;

/// Bridge baseline score — "moderate" per user direction. Outscored by big
/// geological events but solid enough that a quiet sleep still showcases
/// the bridge the player built.
pub const BRIDGE_BASELINE_SCORE: f32 = 80.0;
/// Small per-distance bonus so a wider bridge (more visual content) edges
/// out a short one.
pub const BRIDGE_LENGTH_BONUS_PER_VOXEL: f32 = 1.2;

/// Per-kind score weights (centralised so tracker + scanner stay in sync).
pub const SCORE_PER_LAVA_VOXEL: f32 = 10.0;
pub const SCORE_PER_WATER_VOXEL: f32 = 6.0;
pub const SCORE_PER_STRESS_VOXEL: f32 = 8.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PoiKind {
    Bridge = 0,
    Lava = 1,
    Water = 2,
    Stress = 3,
}

#[derive(Debug, Clone, Copy)]
pub struct Poi {
    pub kind: PoiKind,
    pub score: f32,
    /// World-voxel chunk coordinate in Rust space.
    pub chunk_coord: (i32, i32, i32),
    /// World-voxel center position of the chunk (for camera focus).
    pub center_world_rust: glam::Vec3,
    /// Approximate "radius of interest" in world voxels. Used by the
    /// montage camera to size its orbit: a bridge needs a wider orbit
    /// than a single-chunk lava pool.
    pub extent_radius_voxels: f32,
}

/// Cached top-K result. Held on the engine; the sleep handler refills it.
#[derive(Default)]
pub struct PoiCache {
    pub pois: Vec<Poi>,
    /// Bumped every time the cache is refilled — UE can use this to skip
    /// stale results.
    pub generation: u64,
}

pub type SharedPoiCache = Arc<Mutex<PoiCache>>;

pub fn new_cache() -> SharedPoiCache {
    Arc::new(Mutex::new(PoiCache::default()))
}

/// Score and rank candidates from all sources (fluid, stress, ore, bridges)
/// into a single pool; keep top-K. Caller invokes once per sleep cycle.
///
/// Pass `None` for `anchors` if bridge candidates shouldn't be considered
/// (e.g. early-game saves with no anchor manager).
pub fn scan_top_pois(
    store: &ChunkStore,
    fluid_snap: &FluidSnapshot,
    anchors: Option<&CrystalAnchorManager>,
    chunk_size: usize,
    k: usize,
) -> Vec<Poi> {
    if k == 0 {
        return Vec::new();
    }
    let cs_f = chunk_size as f32;

    let mut candidates: Vec<Poi> = Vec::new();

    // ─── Fluid passes (water / lava) ─────────────────────────────────
    for (chunk_coord, cells) in &fluid_snap.chunks {
        let mut lava = 0usize;
        let mut water = 0usize;
        for cell in cells {
            if cell.level < FLUID_MIN_LEVEL {
                continue;
            }
            if cell.fluid_type.is_lava() {
                lava += 1;
            } else if cell.fluid_type.is_water() {
                water += 1;
            }
        }
        let center = chunk_center_world(*chunk_coord, cs_f);
        // Per-chunk POIs sit within a single chunk → orbit radius = half-chunk
        let chunk_half = cs_f * 0.5;
        if lava >= MIN_LAVA_VOTES {
            candidates.push(Poi {
                kind: PoiKind::Lava,
                score: lava as f32 * SCORE_PER_LAVA_VOXEL,
                chunk_coord: *chunk_coord,
                center_world_rust: center,
                extent_radius_voxels: chunk_half,
            });
        }
        if water >= MIN_WATER_VOTES {
            candidates.push(Poi {
                kind: PoiKind::Water,
                score: water as f32 * SCORE_PER_WATER_VOXEL,
                chunk_coord: *chunk_coord,
                center_world_rust: center,
                extent_radius_voxels: chunk_half,
            });
        }
    }

    // ─── Stress pass ─────────────────────────────────────────────────
    for (chunk_coord, sf) in &store.stress_fields {
        let mut high = 0usize;
        for &s in &sf.stress {
            if s > STRESS_HIGH_THRESHOLD {
                high += 1;
            }
        }
        if high >= MIN_STRESS_VOTES {
            let center = chunk_center_world(*chunk_coord, cs_f);
            candidates.push(Poi {
                kind: PoiKind::Stress,
                score: high as f32 * SCORE_PER_STRESS_VOXEL,
                chunk_coord: *chunk_coord,
                center_world_rust: center,
                extent_radius_voxels: cs_f * 0.5,
            });
        }
    }

    // ─── Bridge pass ─────────────────────────────────────────────────
    // Each grown crystal bridge becomes one candidate at the arch midpoint.
    // Baseline-moderate score modulated slightly by bridge length so larger
    // bridges edge out short ones in close ties.
    if let Some(mgr) = anchors {
        for pair in mgr.list_grown_pairs() {
            let dist = (pair.anchor_b_pos_rust - pair.anchor_a_pos_rust).length();
            let score = BRIDGE_BASELINE_SCORE + dist * BRIDGE_LENGTH_BONUS_PER_VOXEL;
            // chunk_coord derived from midpoint so 3x3x3 pin covers the arch
            let cx = (pair.midpoint_rust.x / cs_f).floor() as i32;
            let cy = (pair.midpoint_rust.y / cs_f).floor() as i32;
            let cz = (pair.midpoint_rust.z / cs_f).floor() as i32;
            candidates.push(Poi {
                kind: PoiKind::Bridge,
                score,
                chunk_coord: (cx, cy, cz),
                center_world_rust: pair.midpoint_rust,
                // Bridge spans from anchor A to anchor B — half-length is the
                // "radius of interest" so the camera frames the whole arch.
                extent_radius_voxels: dist * 0.5,
            });
        }
    }

    // Sort descending by score, keep top-K. Sort is by total score across all
    // kinds — a quiet sleep cycle keeps bridges; a violent one bumps them
    // out for lava/stress.
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(k);
    candidates
}

/// Center world-voxel position of a chunk (Rust coords).
fn chunk_center_world(chunk_coord: (i32, i32, i32), chunk_size_f: f32) -> glam::Vec3 {
    glam::Vec3::new(
        chunk_coord.0 as f32 * chunk_size_f + chunk_size_f * 0.5,
        chunk_coord.1 as f32 * chunk_size_f + chunk_size_f * 0.5,
        chunk_coord.2 as f32 * chunk_size_f + chunk_size_f * 0.5,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::density::DensityField;
    use voxel_core::stress::StressField;

    #[test]
    fn empty_store_returns_no_pois() {
        let store = ChunkStore::new(8);
        let snap = FluidSnapshot::default();
        let pois = scan_top_pois(&store, &snap, None, 8, 5);
        assert!(pois.is_empty());
    }

    #[test]
    fn stress_voxels_above_threshold_register() {
        let mut store = ChunkStore::new(8);
        // Add a chunk with one stress field full of high values
        let size = 9; // chunk_size + 1
        let mut sf = StressField::new(size);
        for s in sf.stress.iter_mut() {
            *s = 1.0; // well above threshold
        }
        store.stress_fields.insert((0, 0, 0), sf);
        // Need a density field to pair (even if empty for our scan)
        store
            .density_fields
            .insert((0, 0, 0), DensityField::new(size));

        let snap = FluidSnapshot::default();
        let pois = scan_top_pois(&store, &snap, None, 8, 5);
        assert_eq!(pois.len(), 1);
        assert_eq!(pois[0].kind, PoiKind::Stress);
        assert_eq!(pois[0].chunk_coord, (0, 0, 0));
    }

    #[test]
    fn bridges_compete_in_unified_pool() {
        let mut mgr = CrystalAnchorManager::default();
        // Place a bridge pair and mark grown
        let _r1 = mgr.place_anchor(glam::Vec3::new(0.0, 0.0, 0.0), glam::Vec3::Y);
        mgr.place_anchor(glam::Vec3::new(30.0, 0.0, 0.0), glam::Vec3::Y);
        let _ = mgr.mark_pending_pairs_grown();
        assert_eq!(mgr.list_grown_pairs().len(), 1);

        // Empty store/snapshot but pass the anchor mgr
        let store = ChunkStore::new(8);
        let snap = FluidSnapshot::default();
        let pois = scan_top_pois(&store, &snap, Some(&mgr), 8, 5);
        assert_eq!(pois.len(), 1);
        assert_eq!(pois[0].kind, PoiKind::Bridge);
        // Score includes baseline + length bonus
        let expected = BRIDGE_BASELINE_SCORE + 30.0 * BRIDGE_LENGTH_BONUS_PER_VOXEL;
        assert!((pois[0].score - expected).abs() < 1e-3);
    }

    #[test]
    fn big_lava_outscores_bridge() {
        // 200 lava voxels in one chunk should beat a single short bridge.
        let store = ChunkStore::new(8);
        let mut snap = FluidSnapshot::default();
        let mut cells = Vec::new();
        for _ in 0..200 {
            cells.push(voxel_fluid::cell::FluidCell {
                level: 1.0,
                fluid_type: voxel_fluid::cell::FluidType::Lava,
                ..Default::default()
            });
        }
        snap.chunks.insert((0, 0, 0), cells);

        let mut mgr = CrystalAnchorManager::default();
        mgr.place_anchor(glam::Vec3::new(100.0, 0.0, 0.0), glam::Vec3::Y);
        mgr.place_anchor(glam::Vec3::new(110.0, 0.0, 0.0), glam::Vec3::Y);
        let _ = mgr.mark_pending_pairs_grown();

        let pois = scan_top_pois(&store, &snap, Some(&mgr), 8, 3);
        // Top should be lava, not bridge
        assert_eq!(pois[0].kind, PoiKind::Lava);
        assert!(pois[0].score > pois[1].score);
    }

    #[test]
    fn quiet_sleep_features_bridges() {
        // Empty world, only bridges → bridges fill the slots.
        let store = ChunkStore::new(8);
        let snap = FluidSnapshot::default();
        let mut mgr = CrystalAnchorManager::default();
        // Two bridges
        mgr.place_anchor(glam::Vec3::new(0.0, 0.0, 0.0), glam::Vec3::Y);
        mgr.place_anchor(glam::Vec3::new(30.0, 0.0, 0.0), glam::Vec3::Y);
        mgr.place_anchor(glam::Vec3::new(0.0, 0.0, 100.0), glam::Vec3::Y);
        mgr.place_anchor(glam::Vec3::new(30.0, 0.0, 100.0), glam::Vec3::Y);
        let _ = mgr.mark_pending_pairs_grown();

        let pois = scan_top_pois(&store, &snap, Some(&mgr), 8, 3);
        assert_eq!(pois.len(), 2);
        assert!(pois.iter().all(|p| p.kind == PoiKind::Bridge));
    }

    #[test]
    fn top_k_truncates() {
        let mut store = ChunkStore::new(8);
        let size = 9;
        // Insert 5 chunks all with high stress, descending fill ratios
        for i in 0..5i32 {
            let mut sf = StressField::new(size);
            // Higher i → more high-stress voxels → higher score
            let n_high = ((i + 1) * 50) as usize;
            for (idx, s) in sf.stress.iter_mut().enumerate() {
                if idx < n_high {
                    *s = 1.0;
                }
            }
            store.stress_fields.insert((i, 0, 0), sf);
            store.density_fields.insert((i, 0, 0), DensityField::new(size));
        }
        let snap = FluidSnapshot::default();
        let pois = scan_top_pois(&store, &snap, None, 8, 3);
        assert_eq!(pois.len(), 3);
        // Top-K should be sorted descending by score (chunk 4 has the most)
        assert_eq!(pois[0].chunk_coord, (4, 0, 0));
        assert_eq!(pois[1].chunk_coord, (3, 0, 0));
        assert_eq!(pois[2].chunk_coord, (2, 0, 0));
    }
}
