//! Crystal Growth Bridge anchor manager.
//!
//! Holds the in-flight set of player-thrown anchors. Two anchors form a pair;
//! at the next deep sleep, each `PairedReadyForGrowth` pair triggers an
//! organic-branching crystal bridge to grow between the two anchor positions.
//!
//! All positions stored here are **Rust voxel-space** (Y-up, right-hand).
//! The FFI layer converts to/from UE world space via `crate::convert`.
//!
//! State machine:
//!     (start) ──► place #1 ──► SingleAnchor
//!                                  │
//!                                  ▼ place #2 (close enough)
//!                       PairedReadyForGrowth ◄─ both anchors
//!                                  │
//!                                  ▼ next deep sleep
//!                              Grown ◄─ both anchors
//!
//! Caps: at most 4 `PairedReadyForGrowth` pairs queued at once (keeps the
//! sleep-montage POI rotation bounded). Re-paired anchors aren't supported —
//! a third thrown anchor always opens a new pair, never re-pairs with an
//! already-paired one.

use glam::Vec3;
use std::collections::HashMap;

/// Max paired-pending pairs queued before further pair-completions are rejected.
/// The 4×2.5s POI rotation in the sleep montage keeps the cinematic bounded.
pub const MAX_PAIRED_PENDING: usize = 4;

/// Max distance between the two anchors of a pair, in voxels.
pub const MAX_PAIR_DISTANCE_VOXELS: f32 = 60.0;

/// Reject second-anchor placement if it lands within this many voxels of any
/// existing anchor — prevents pair-self situations and overlapping markers.
pub const MIN_ANCHOR_SEPARATION_VOXELS: f32 = 2.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnchorState {
    SingleAnchor { pair_token: u64 },
    PairedReadyForGrowth { partner_id: u64, pair_token: u64 },
    Grown { partner_id: u64, pair_token: u64 },
}

#[derive(Debug, Clone)]
pub struct AnchorMarker {
    pub id: u64,
    /// World position in Rust voxel space.
    pub world_pos_rust: Vec3,
    pub surface_normal: Vec3,
    pub state: AnchorState,
}

/// FFI-friendly error codes returned by `place_anchor`.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaceAnchorError {
    Ok = 0,
    TooFarFromPartner = 1,
    CapReached = 2,
    NoSolidUnder = 3,
    DuplicateTooClose = 4,
}

#[derive(Debug, Clone)]
pub struct PlaceAnchorResult {
    pub error: PlaceAnchorError,
    pub anchor_id: u64,
    pub pair_completed: bool,
    pub partner_id: u64,
    pub pair_token: u64,
}

impl PlaceAnchorResult {
    fn error_only(e: PlaceAnchorError) -> Self {
        Self {
            error: e,
            anchor_id: 0,
            pair_completed: false,
            partner_id: 0,
            pair_token: 0,
        }
    }
}

/// Summary of a single paired pair — used both for "pending" listings and
/// post-growth "grown" listings (same fields on the FFI side).
#[derive(Debug, Clone)]
pub struct PairInfo {
    pub pair_token: u64,
    pub anchor_a_id: u64,
    pub anchor_b_id: u64,
    pub anchor_a_pos_rust: Vec3,
    pub anchor_b_pos_rust: Vec3,
    /// Lifted arch midpoint (per `voxel-ffi::crystal_anchors::arch_midpoint`).
    pub midpoint_rust: Vec3,
}

pub struct CrystalAnchorManager {
    anchors: HashMap<u64, AnchorMarker>,
    next_id: u64,
    next_pair_token: u64,
    /// First-anchor-of-a-pair: thrown but partner not yet placed.
    pending_solo_id: Option<u64>,
}

impl Default for CrystalAnchorManager {
    fn default() -> Self {
        Self {
            anchors: HashMap::new(),
            next_id: 1,
            next_pair_token: 1,
            pending_solo_id: None,
        }
    }
}

impl CrystalAnchorManager {
    /// Place a new anchor in Rust voxel space. Implements the state machine
    /// from the module docs.
    pub fn place_anchor(&mut self, pos_rust: Vec3, normal: Vec3) -> PlaceAnchorResult {
        // Duplicate-too-close: reject if any existing anchor is within MIN_ANCHOR_SEPARATION
        for existing in self.anchors.values() {
            if (existing.world_pos_rust - pos_rust).length() < MIN_ANCHOR_SEPARATION_VOXELS {
                return PlaceAnchorResult::error_only(PlaceAnchorError::DuplicateTooClose);
            }
        }

        if let Some(solo_id) = self.pending_solo_id {
            // Second anchor of an existing pair — validate distance to partner.
            let partner_pos = match self.anchors.get(&solo_id) {
                Some(a) => a.world_pos_rust,
                // Defensive: solo_id stale (anchor was cancelled). Treat this throw
                // as a fresh first anchor.
                None => {
                    self.pending_solo_id = None;
                    return self.place_first_anchor(pos_rust, normal);
                }
            };

            let dist = (partner_pos - pos_rust).length();
            if dist > MAX_PAIR_DISTANCE_VOXELS {
                return PlaceAnchorResult::error_only(PlaceAnchorError::TooFarFromPartner);
            }

            // Check cap: how many PairedReadyForGrowth pairs do we already have?
            // Each pair contributes 2 anchors in that state, so divide by 2.
            let paired_count = self
                .anchors
                .values()
                .filter(|a| matches!(a.state, AnchorState::PairedReadyForGrowth { .. }))
                .count()
                / 2;
            if paired_count >= MAX_PAIRED_PENDING {
                return PlaceAnchorResult::error_only(PlaceAnchorError::CapReached);
            }

            // Allocate the second anchor and flip both to PairedReadyForGrowth.
            let new_id = self.alloc_id();
            let pair_token = match self.anchors.get(&solo_id) {
                Some(a) => match a.state {
                    AnchorState::SingleAnchor { pair_token } => pair_token,
                    _ => self.alloc_pair_token(),
                },
                None => self.alloc_pair_token(),
            };

            self.anchors.insert(
                new_id,
                AnchorMarker {
                    id: new_id,
                    world_pos_rust: pos_rust,
                    surface_normal: normal,
                    state: AnchorState::PairedReadyForGrowth {
                        partner_id: solo_id,
                        pair_token,
                    },
                },
            );

            // Flip the first anchor's state too
            if let Some(a) = self.anchors.get_mut(&solo_id) {
                a.state = AnchorState::PairedReadyForGrowth {
                    partner_id: new_id,
                    pair_token,
                };
            }

            self.pending_solo_id = None;
            PlaceAnchorResult {
                error: PlaceAnchorError::Ok,
                anchor_id: new_id,
                pair_completed: true,
                partner_id: solo_id,
                pair_token,
            }
        } else {
            self.place_first_anchor(pos_rust, normal)
        }
    }

    fn place_first_anchor(&mut self, pos_rust: Vec3, normal: Vec3) -> PlaceAnchorResult {
        // Cap is enforced at *pair completion* time (place #2), not at place #1
        // — a solo unpaired anchor doesn't count toward the limit. Throw #1
        // always succeeds here.
        let new_id = self.alloc_id();
        let pair_token = self.alloc_pair_token();
        self.anchors.insert(
            new_id,
            AnchorMarker {
                id: new_id,
                world_pos_rust: pos_rust,
                surface_normal: normal,
                state: AnchorState::SingleAnchor { pair_token },
            },
        );
        self.pending_solo_id = Some(new_id);
        PlaceAnchorResult {
            error: PlaceAnchorError::Ok,
            anchor_id: new_id,
            pair_completed: false,
            partner_id: 0,
            pair_token,
        }
    }

    /// Cancel an anchor by id. Returns true if anchor was found and removed.
    /// If the anchor was paired, the partner is also removed (you can't have
    /// half-pairs). If the anchor was the pending_solo, clears that slot.
    pub fn cancel_anchor(&mut self, anchor_id: u64) -> bool {
        let removed = match self.anchors.remove(&anchor_id) {
            Some(a) => a,
            None => return false,
        };

        match removed.state {
            AnchorState::SingleAnchor { .. } => {
                if self.pending_solo_id == Some(anchor_id) {
                    self.pending_solo_id = None;
                }
            }
            AnchorState::PairedReadyForGrowth { partner_id, .. }
            | AnchorState::Grown { partner_id, .. } => {
                // Remove partner too — bridge can't exist with one anchor
                self.anchors.remove(&partner_id);
            }
        }
        true
    }

    /// Find the nearest unpaired anchor to a world position. Returns None if
    /// no unpaired anchors exist within `max_dist_voxels`.
    pub fn nearest_unpaired(&self, pos_rust: Vec3, max_dist_voxels: f32) -> Option<u64> {
        self.anchors
            .values()
            .filter(|a| matches!(a.state, AnchorState::SingleAnchor { .. }))
            .map(|a| (a.id, (a.world_pos_rust - pos_rust).length()))
            .filter(|(_, d)| *d <= max_dist_voxels)
            .min_by(|(_, da), (_, db)| da.partial_cmp(db).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id)
    }

    /// Return one summary per PairedReadyForGrowth pair (each pair listed once).
    pub fn list_pending_pairs(&self) -> Vec<PairInfo> {
        self.list_pairs_in_state(|s| matches!(s, AnchorState::PairedReadyForGrowth { .. }))
    }

    /// Return one summary per Grown pair (each pair listed once).
    pub fn list_grown_pairs(&self) -> Vec<PairInfo> {
        self.list_pairs_in_state(|s| matches!(s, AnchorState::Grown { .. }))
    }

    fn list_pairs_in_state<F: Fn(&AnchorState) -> bool>(&self, filter: F) -> Vec<PairInfo> {
        let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut out = Vec::new();
        for anchor in self.anchors.values() {
            if !filter(&anchor.state) {
                continue;
            }
            let (partner_id, pair_token) = match anchor.state {
                AnchorState::PairedReadyForGrowth {
                    partner_id,
                    pair_token,
                }
                | AnchorState::Grown {
                    partner_id,
                    pair_token,
                } => (partner_id, pair_token),
                _ => continue,
            };
            if seen.contains(&pair_token) {
                continue;
            }
            seen.insert(pair_token);
            let partner = match self.anchors.get(&partner_id) {
                Some(p) => p,
                None => continue, // half-orphan; skip
            };
            out.push(PairInfo {
                pair_token,
                anchor_a_id: anchor.id,
                anchor_b_id: partner.id,
                anchor_a_pos_rust: anchor.world_pos_rust,
                anchor_b_pos_rust: partner.world_pos_rust,
                midpoint_rust: arch_midpoint(anchor.world_pos_rust, partner.world_pos_rust),
            });
        }
        out
    }

    /// Flip all PairedReadyForGrowth pairs to Grown. Returns the list of
    /// pairs that just transitioned, in deterministic order (sorted by
    /// pair_token). Phase 3 actually generates the voxels using this list.
    pub fn mark_pending_pairs_grown(&mut self) -> Vec<PairInfo> {
        let pending = self.list_pending_pairs();
        let mut sorted = pending.clone();
        sorted.sort_by_key(|p| p.pair_token);
        for pair in &sorted {
            if let Some(a) = self.anchors.get_mut(&pair.anchor_a_id) {
                if let AnchorState::PairedReadyForGrowth {
                    partner_id,
                    pair_token,
                } = a.state
                {
                    a.state = AnchorState::Grown {
                        partner_id,
                        pair_token,
                    };
                }
            }
            if let Some(b) = self.anchors.get_mut(&pair.anchor_b_id) {
                if let AnchorState::PairedReadyForGrowth {
                    partner_id,
                    pair_token,
                } = b.state
                {
                    b.state = AnchorState::Grown {
                        partner_id,
                        pair_token,
                    };
                }
            }
        }
        sorted
    }

    fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn alloc_pair_token(&mut self) -> u64 {
        let t = self.next_pair_token;
        self.next_pair_token += 1;
        t
    }

    pub fn anchor_count(&self) -> usize {
        self.anchors.len()
    }

    pub fn paired_pair_count(&self) -> usize {
        self.anchors
            .values()
            .filter(|a| matches!(a.state, AnchorState::PairedReadyForGrowth { .. }))
            .count()
            / 2
    }
}

impl CrystalAnchorManager {
    /// Serialize the manager to a JSON string. Pairs the `world_placed_*` save
    /// pattern — packs into the world delta blob so save/load stays atomic.
    /// Format is hand-rolled to avoid adding a serde-derive dep.
    pub fn to_json_string(&self) -> String {
        use serde_json::{json, Value};

        let anchors: Vec<Value> = self.anchors
            .iter()
            .map(|(_id, a)| {
                let (state_tag, partner_id, pair_token) = match a.state {
                    AnchorState::SingleAnchor { pair_token } => ("Single", 0u64, pair_token),
                    AnchorState::PairedReadyForGrowth { partner_id, pair_token } => ("Paired", partner_id, pair_token),
                    AnchorState::Grown { partner_id, pair_token } => ("Grown", partner_id, pair_token),
                };
                json!({
                    "id": a.id,
                    "pos": [a.world_pos_rust.x, a.world_pos_rust.y, a.world_pos_rust.z],
                    "normal": [a.surface_normal.x, a.surface_normal.y, a.surface_normal.z],
                    "state": state_tag,
                    "partner_id": partner_id,
                    "pair_token": pair_token,
                })
            })
            .collect();

        json!({
            "version": 1,
            "next_id": self.next_id,
            "next_pair_token": self.next_pair_token,
            "pending_solo_id": self.pending_solo_id.unwrap_or(0),
            "has_pending_solo": self.pending_solo_id.is_some(),
            "anchors": anchors,
        }).to_string()
    }

    /// Deserialize from the JSON produced by `to_json_string`. On any parse
    /// error returns a default (empty) manager — saves predating the feature
    /// store an empty string here, and we don't want to fail loads for it.
    pub fn from_json_string(s: &str) -> Self {
        use serde_json::Value;
        if s.is_empty() {
            return Self::default();
        }
        let root: Value = match serde_json::from_str(s) {
            Ok(v) => v,
            Err(_) => return Self::default(),
        };
        let next_id = root.get("next_id").and_then(|v| v.as_u64()).unwrap_or(1);
        let next_pair_token = root.get("next_pair_token").and_then(|v| v.as_u64()).unwrap_or(1);
        let pending_solo_id = if root.get("has_pending_solo").and_then(|v| v.as_bool()).unwrap_or(false) {
            root.get("pending_solo_id").and_then(|v| v.as_u64())
        } else {
            None
        };

        let mut anchors = HashMap::new();
        if let Some(arr) = root.get("anchors").and_then(|v| v.as_array()) {
            for a in arr {
                let id = match a.get("id").and_then(|v| v.as_u64()) {
                    Some(v) => v,
                    None => continue,
                };
                let pos_arr = match a.get("pos").and_then(|v| v.as_array()) {
                    Some(v) if v.len() == 3 => v,
                    _ => continue,
                };
                let nrm_arr = match a.get("normal").and_then(|v| v.as_array()) {
                    Some(v) if v.len() == 3 => v,
                    _ => continue,
                };
                let pos = Vec3::new(
                    pos_arr[0].as_f64().unwrap_or(0.0) as f32,
                    pos_arr[1].as_f64().unwrap_or(0.0) as f32,
                    pos_arr[2].as_f64().unwrap_or(0.0) as f32,
                );
                let normal = Vec3::new(
                    nrm_arr[0].as_f64().unwrap_or(0.0) as f32,
                    nrm_arr[1].as_f64().unwrap_or(1.0) as f32,
                    nrm_arr[2].as_f64().unwrap_or(0.0) as f32,
                );
                let partner_id = a.get("partner_id").and_then(|v| v.as_u64()).unwrap_or(0);
                let pair_token = a.get("pair_token").and_then(|v| v.as_u64()).unwrap_or(0);
                let state = match a.get("state").and_then(|v| v.as_str()).unwrap_or("Single") {
                    "Paired" => AnchorState::PairedReadyForGrowth { partner_id, pair_token },
                    "Grown" => AnchorState::Grown { partner_id, pair_token },
                    _ => AnchorState::SingleAnchor { pair_token },
                };
                anchors.insert(id, AnchorMarker {
                    id,
                    world_pos_rust: pos,
                    surface_normal: normal,
                    state,
                });
            }
        }

        Self {
            anchors,
            next_id,
            next_pair_token,
            pending_solo_id,
        }
    }
}

/// Arch midpoint for a bridge — lifts mid-chord by `min(6, 0.10 * dist)`
/// Rust voxels, producing a gentle arched silhouette.
pub fn arch_midpoint(a: Vec3, b: Vec3) -> Vec3 {
    let mid = (a + b) * 0.5;
    let dist = (b - a).length();
    let lift = (0.10 * dist).min(6.0);
    // Y-up in Rust → lift along +Y
    mid + Vec3::new(0.0, lift, 0.0)
}

// ─── Bridge growth (Phase 3) ────────────────────────────────────────────────

/// Inline deterministic xorshift PRNG so we don't pull in rand_chacha as a dep
/// just for branch selection.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    if x == 0 {
        x = 0x9E3779B97F4A7C15;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn rng_range_f32(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let r = (xorshift64(state) as f64) / (u64::MAX as f64);
    lo + (hi - lo) * r as f32
}

fn rng_range_usize(state: &mut u64, lo: usize, hi_exclusive: usize) -> usize {
    if hi_exclusive <= lo {
        return lo;
    }
    let r = xorshift64(state) as usize;
    lo + r % (hi_exclusive - lo)
}

/// Walking-spine segments along A → arch → B at constant radius `spine_radius`.
/// Stepping is `step` voxels along the polyline.
fn build_spine(a: Vec3, arch: Vec3, b: Vec3, step: f32, spine_radius: f32) -> Vec<voxel_gen::worm::path::WormSegment> {
    let mut out = Vec::new();
    // A→arch leg
    let leg1_len = (arch - a).length();
    let n1 = ((leg1_len / step).ceil() as i32).max(1);
    for i in 0..=n1 {
        let t = i as f32 / n1 as f32;
        out.push(voxel_gen::worm::path::WormSegment {
            position: a.lerp(arch, t),
            radius: spine_radius,
        });
    }
    // arch→B leg
    let leg2_len = (b - arch).length();
    let n2 = ((leg2_len / step).ceil() as i32).max(1);
    for i in 1..=n2 {
        let t = i as f32 / n2 as f32;
        out.push(voxel_gen::worm::path::WormSegment {
            position: arch.lerp(b, t),
            radius: spine_radius,
        });
    }
    out
}

/// Perturb a unit tangent direction by up to `max_angle_deg` degrees, using
/// the supplied PRNG. Returns a normalized vector.
fn perturb_dir(tangent: Vec3, state: &mut u64, max_angle_deg: f32) -> Vec3 {
    let t = tangent.normalize_or_zero();
    if t.length_squared() < 1e-6 {
        return Vec3::new(1.0, 0.0, 0.0);
    }
    // Pick an arbitrary perpendicular axis
    let axis_seed = if t.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let perp1 = axis_seed.cross(t).normalize_or_zero();
    let perp2 = t.cross(perp1).normalize_or_zero();

    // Random tilt within cone
    let max_rad = max_angle_deg.to_radians();
    let tilt = rng_range_f32(state, 0.0, max_rad);
    let twist = rng_range_f32(state, 0.0, std::f32::consts::TAU);

    let along = t * tilt.cos();
    let side = (perp1 * twist.cos() + perp2 * twist.sin()) * tilt.sin();
    (along + side).normalize_or_zero()
}

/// Generate 1-3 side branches off a main path. Each branch starts at a random
/// middle-segment of the parent and grows 4-10 voxels in a perturbed direction.
fn roll_branches(path: &[voxel_gen::worm::path::WormSegment], state: &mut u64) -> Vec<voxel_gen::worm::path::WormSegment> {
    if path.len() < 4 {
        return Vec::new();
    }
    let n_branches = rng_range_usize(state, 1, 4); // 1..=3
    let mut out = Vec::new();
    for _ in 0..n_branches {
        let lo = path.len() / 5;
        let hi = (path.len() * 4 / 5).max(lo + 1);
        let idx = rng_range_usize(state, lo, hi);
        let parent = &path[idx];
        let parent_next = if idx + 1 < path.len() {
            path[idx + 1].position
        } else {
            // Approximate from previous
            parent.position + (parent.position - path[idx.saturating_sub(1)].position)
        };
        let tangent = (parent_next - parent.position).normalize_or_zero();
        let dir = perturb_dir(tangent, state, 25.0);
        let length = rng_range_f32(state, 4.0, 10.0);
        let branch_end = parent.position + dir * length;
        let branch_steps = (length / 1.0).ceil() as u32;
        let branch_seed = xorshift64(state);
        let branch = voxel_gen::worm::path::generate_worm_path(
            branch_seed,
            parent.position,
            branch_end,
            1.0,
            branch_steps,
            0.8,
            1.6,
        );
        out.extend(branch);
    }
    out
}

/// Merge overlapping per-chunk dirty-range entries — same chunk key entries
/// get unioned into one bounding box. Brings remesh/sync_boundary cost from
/// O(segments) down to O(chunks).
fn merge_dirty_chunks(
    entries: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)>,
) -> Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> {
    use std::collections::HashMap;
    let mut map: HashMap<(i32, i32, i32), (usize, usize, usize, usize, usize, usize)> = HashMap::new();
    for (key, lx, ly, lz, hx, hy, hz) in entries {
        match map.get_mut(&key) {
            Some(b) => {
                b.0 = b.0.min(lx);
                b.1 = b.1.min(ly);
                b.2 = b.2.min(lz);
                b.3 = b.3.max(hx);
                b.4 = b.4.max(hy);
                b.5 = b.5.max(hz);
            }
            None => {
                map.insert(key, (lx, ly, lz, hx, hy, hz));
            }
        }
    }
    map.into_iter()
        .map(|(k, b)| (k, b.0, b.1, b.2, b.3, b.4, b.5))
        .collect()
}

/// Grow all PairedReadyForGrowth pairs into crystal-voxel bridges. Returns
/// the list of pairs that just transitioned to Grown (deterministic order).
/// Called from the sleep handler before geological-time advancement.
pub fn grow_pending_bridges(
    mgr: &mut CrystalAnchorManager,
    store: &mut crate::store::ChunkStore,
    config: &voxel_gen::config::GenerationConfig,
    world_scale: f32,
) {
    use voxel_core::material::Material;

    let pairs = mgr.list_pending_pairs();
    if pairs.is_empty() {
        return;
    }

    // Compute overall chunk-coord bounds across all pairs for a single undo
    // snapshot. This is conservative but cheap.
    let eb = config.effective_bounds();
    let mut overall_lo = (i32::MAX, i32::MAX, i32::MAX);
    let mut overall_hi = (i32::MIN, i32::MIN, i32::MIN);

    for pair in &pairs {
        let a = pair.anchor_a_pos_rust;
        let b = pair.anchor_b_pos_rust;
        let arch = arch_midpoint(a, b);
        // Bound for undo: bbox of {a, b, arch} expanded by max_radius (3.0).
        for p in [a, b, arch] {
            let r = 3.5; // worst-case segment radius
            let (lo, hi) = crate::brushes::chunk_range_for_sphere_pub(p, r, eb);
            overall_lo.0 = overall_lo.0.min(lo.0);
            overall_lo.1 = overall_lo.1.min(lo.1);
            overall_lo.2 = overall_lo.2.min(lo.2);
            overall_hi.0 = overall_hi.0.max(hi.0);
            overall_hi.1 = overall_hi.1.max(hi.1);
            overall_hi.2 = overall_hi.2.max(hi.2);
        }
    }
    if overall_lo.0 <= overall_hi.0 {
        crate::brushes::capture_undo_for_range_pub(store, overall_lo, overall_hi);
    }

    let mut all_dirty: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> =
        Vec::new();

    for pair in &pairs {
        let a = pair.anchor_a_pos_rust;
        let b = pair.anchor_b_pos_rust;
        let arch = arch_midpoint(a, b);
        let dist = (b - a).length();
        if dist < 1.0 {
            continue; // skip degenerate pairs
        }
        let steps_half = (((dist * 0.7).ceil()) as u32).max(8);
        let seed_ab = pair.pair_token.wrapping_mul(0x9E3779B97F4A7C15);
        let seed_ba = seed_ab ^ 0xA5A5_A5A5_A5A5_A5A5;

        let path_ab = voxel_gen::worm::path::generate_worm_path(seed_ab, a, arch, 1.0, steps_half, 1.5, 3.0);
        let path_ba = voxel_gen::worm::path::generate_worm_path(seed_ba, b, arch, 1.0, steps_half, 1.5, 3.0);

        // Walking spine guarantees continuous walkable surface even if the
        // noise-perturbed worm paths wobble.
        let spine = build_spine(a, arch, b, 0.6, 1.2);

        // Side branches — purely decorative, give the bridge the "organic
        // branching" silhouette.
        let mut rng_state = seed_ab.wrapping_add(0xBEEF_BEEF_BEEF_BEEF);
        let mut branches = roll_branches(&path_ab, &mut rng_state);
        branches.extend(roll_branches(&path_ba, &mut rng_state));

        for seg in path_ab
            .iter()
            .chain(path_ba.iter())
            .chain(spine.iter())
            .chain(branches.iter())
        {
            crate::brushes::fill_sphere_into(
                store,
                seg.position,
                seg.radius,
                Material::Crystal,
                config,
                &mut all_dirty,
            );
        }
    }

    // Dedupe & union per-chunk bounds, then a single finalize.
    let merged = merge_dirty_chunks(all_dirty);
    if !merged.is_empty() {
        let _outcome = crate::brushes::finalize_brush_batch(store, merged, config, world_scale);
    }

    // Flip pairs to Grown — they're now real voxels in the world.
    let _ = mgr.mark_pending_pairs_grown();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z)
    }

    #[test]
    fn first_throw_creates_solo() {
        let mut mgr = CrystalAnchorManager::default();
        let r = mgr.place_anchor(v(0.0, 0.0, 0.0), v(0.0, 1.0, 0.0));
        assert_eq!(r.error, PlaceAnchorError::Ok);
        assert!(!r.pair_completed);
        assert_eq!(mgr.anchor_count(), 1);
        assert_eq!(mgr.paired_pair_count(), 0);
    }

    #[test]
    fn second_throw_within_range_pairs() {
        let mut mgr = CrystalAnchorManager::default();
        let r1 = mgr.place_anchor(v(0.0, 0.0, 0.0), v(0.0, 1.0, 0.0));
        assert!(!r1.pair_completed);
        let r2 = mgr.place_anchor(v(30.0, 0.0, 0.0), v(0.0, 1.0, 0.0));
        assert_eq!(r2.error, PlaceAnchorError::Ok);
        assert!(r2.pair_completed);
        assert_eq!(r2.partner_id, r1.anchor_id);
        assert_eq!(mgr.paired_pair_count(), 1);
        assert_eq!(mgr.list_pending_pairs().len(), 1);
    }

    #[test]
    fn second_throw_too_far_rejects() {
        let mut mgr = CrystalAnchorManager::default();
        mgr.place_anchor(v(0.0, 0.0, 0.0), v(0.0, 1.0, 0.0));
        let r = mgr.place_anchor(v(100.0, 0.0, 0.0), v(0.0, 1.0, 0.0));
        assert_eq!(r.error, PlaceAnchorError::TooFarFromPartner);
        assert_eq!(mgr.anchor_count(), 1); // unchanged
        assert_eq!(mgr.paired_pair_count(), 0);
    }

    #[test]
    fn third_throw_starts_new_pair() {
        let mut mgr = CrystalAnchorManager::default();
        mgr.place_anchor(v(0.0, 0.0, 0.0), v(0.0, 1.0, 0.0));
        mgr.place_anchor(v(30.0, 0.0, 0.0), v(0.0, 1.0, 0.0));
        let r = mgr.place_anchor(v(0.0, 0.0, 50.0), v(0.0, 1.0, 0.0));
        assert!(!r.pair_completed); // it's the start of a new pair
        assert_eq!(mgr.anchor_count(), 3);
        assert_eq!(mgr.paired_pair_count(), 1); // only the first pair is paired
    }

    #[test]
    fn cap_reached_at_4_pairs() {
        let mut mgr = CrystalAnchorManager::default();
        // Build 4 pairs (8 throws)
        for pair_i in 0..4 {
            let zoff = pair_i as f32 * 100.0;
            mgr.place_anchor(v(0.0, 0.0, zoff), v(0.0, 1.0, 0.0));
            let r = mgr.place_anchor(v(30.0, 0.0, zoff), v(0.0, 1.0, 0.0));
            assert!(r.pair_completed);
        }
        // 9th and 10th throws — 9 opens a new solo, 10 tries to complete
        mgr.place_anchor(v(0.0, 0.0, 1000.0), v(0.0, 1.0, 0.0));
        let r10 = mgr.place_anchor(v(30.0, 0.0, 1000.0), v(0.0, 1.0, 0.0));
        assert_eq!(r10.error, PlaceAnchorError::CapReached);
        assert_eq!(mgr.paired_pair_count(), 4);
    }

    #[test]
    fn duplicate_too_close_rejects() {
        let mut mgr = CrystalAnchorManager::default();
        mgr.place_anchor(v(0.0, 0.0, 0.0), v(0.0, 1.0, 0.0));
        let r = mgr.place_anchor(v(0.5, 0.0, 0.0), v(0.0, 1.0, 0.0));
        assert_eq!(r.error, PlaceAnchorError::DuplicateTooClose);
    }

    #[test]
    fn cancel_solo_clears_pending() {
        let mut mgr = CrystalAnchorManager::default();
        let r = mgr.place_anchor(v(0.0, 0.0, 0.0), v(0.0, 1.0, 0.0));
        assert!(mgr.cancel_anchor(r.anchor_id));
        assert_eq!(mgr.anchor_count(), 0);
        // After cancel, next throw should be a fresh solo
        let r2 = mgr.place_anchor(v(5.0, 0.0, 0.0), v(0.0, 1.0, 0.0));
        assert!(!r2.pair_completed);
    }

    #[test]
    fn cancel_paired_removes_partner() {
        let mut mgr = CrystalAnchorManager::default();
        let r1 = mgr.place_anchor(v(0.0, 0.0, 0.0), v(0.0, 1.0, 0.0));
        let r2 = mgr.place_anchor(v(30.0, 0.0, 0.0), v(0.0, 1.0, 0.0));
        assert!(r2.pair_completed);
        assert!(mgr.cancel_anchor(r1.anchor_id));
        // Both gone
        assert_eq!(mgr.anchor_count(), 0);
    }

    #[test]
    fn mark_pending_pairs_grown_flips_states() {
        let mut mgr = CrystalAnchorManager::default();
        mgr.place_anchor(v(0.0, 0.0, 0.0), v(0.0, 1.0, 0.0));
        mgr.place_anchor(v(30.0, 0.0, 0.0), v(0.0, 1.0, 0.0));
        assert_eq!(mgr.list_pending_pairs().len(), 1);
        assert_eq!(mgr.list_grown_pairs().len(), 0);
        let grown = mgr.mark_pending_pairs_grown();
        assert_eq!(grown.len(), 1);
        assert_eq!(mgr.list_pending_pairs().len(), 0);
        assert_eq!(mgr.list_grown_pairs().len(), 1);
        // Idempotency: running again does nothing
        let grown2 = mgr.mark_pending_pairs_grown();
        assert_eq!(grown2.len(), 0);
    }

    #[test]
    fn nearest_unpaired_finds_closest() {
        let mut mgr = CrystalAnchorManager::default();
        let r1 = mgr.place_anchor(v(0.0, 0.0, 0.0), v(0.0, 1.0, 0.0));
        // Far throw, will be solo
        mgr.place_anchor(v(0.0, 0.0, 200.0), v(0.0, 1.0, 0.0)); // wait — would be a second of a pair
        // Actually that 2nd throw will try to pair with r1 and fail (too far),
        // leaving only r1 as the solo. So pending_solo is still r1.
        let near = mgr.nearest_unpaired(v(0.0, 0.0, 5.0), 30.0);
        assert_eq!(near, Some(r1.anchor_id));
        let far = mgr.nearest_unpaired(v(1000.0, 0.0, 1000.0), 30.0);
        assert!(far.is_none());
    }

    #[test]
    fn arch_midpoint_lifts_upward() {
        let m = arch_midpoint(v(0.0, 0.0, 0.0), v(60.0, 0.0, 0.0));
        // 60 voxels → 6 voxel lift (capped)
        assert!((m.x - 30.0).abs() < 1e-4);
        assert!((m.y - 6.0).abs() < 1e-4);
        assert!((m.z - 0.0).abs() < 1e-4);
    }

    // ─── Integration: bridge growth writes crystal voxels ───────────────────
    use voxel_core::material::Material;
    use voxel_gen::config::GenerationConfig;
    use crate::store::ChunkStore;
    use voxel_core::density::DensityField;

    /// Build a 3x1x3 grid of solid chunks (Limestone). Bridge between (4,4,4)
    /// and (12,4,4) in voxel space will cross multiple chunks.
    fn make_test_store_for_bridge() -> (ChunkStore, GenerationConfig) {
        let mut config = GenerationConfig::default();
        config.chunk_size = 8;
        let size = config.chunk_size + 1;
        let mut store = ChunkStore::new(8);
        for cx in 0..3 {
            for cz in 0..3 {
                let mut field = DensityField::new(size);
                for z in 0..size {
                    for y in 0..size {
                        for x in 0..size {
                            let s = field.get_mut(x, y, z);
                            s.density = 1.0;
                            s.material = Material::Limestone;
                        }
                    }
                }
                store.density_fields.insert((cx, 0, cz), field);
            }
        }
        (store, config)
    }

    #[test]
    fn grow_pending_bridges_writes_crystal_voxels() {
        let mut mgr = CrystalAnchorManager::default();
        let (mut store, config) = make_test_store_for_bridge();

        // Two anchors 8 voxels apart along x
        let r1 = mgr.place_anchor(v(4.0, 4.0, 4.0), v(0.0, 1.0, 0.0));
        let r2 = mgr.place_anchor(v(12.0, 4.0, 4.0), v(0.0, 1.0, 0.0));
        assert_eq!(r1.error, PlaceAnchorError::Ok);
        assert_eq!(r2.error, PlaceAnchorError::Ok);
        assert!(r2.pair_completed);
        assert_eq!(mgr.list_pending_pairs().len(), 1);

        // Grow at sleep
        grow_pending_bridges(&mut mgr, &mut store, &config, 1.0);

        // State flipped to Grown
        assert_eq!(mgr.list_pending_pairs().len(), 0);
        assert_eq!(mgr.list_grown_pairs().len(), 1);

        // At least one voxel along the (4..12, 4, 4) line should now be Crystal
        let mut found_crystal = false;
        for cx in 0..3 {
            if let Some(f) = store.density_fields.get(&(cx, 0, 0)) {
                for x in 0..f.size {
                    for y in 0..f.size {
                        for z in 0..f.size {
                            if f.get(x, y, z).material == Material::Crystal {
                                found_crystal = true;
                            }
                        }
                    }
                }
            }
        }
        assert!(found_crystal, "no Crystal voxels found after grow_pending_bridges");
    }

    #[test]
    fn json_roundtrip_preserves_state() {
        let mut mgr = CrystalAnchorManager::default();
        let r1 = mgr.place_anchor(v(0.0, 0.0, 0.0), v(0.0, 1.0, 0.0));
        let r2 = mgr.place_anchor(v(30.0, 0.0, 0.0), v(0.0, 1.0, 0.0));
        assert!(r2.pair_completed);
        // Add a solo too
        let r3 = mgr.place_anchor(v(0.0, 0.0, 100.0), v(0.0, 1.0, 0.0));
        assert!(!r3.pair_completed);

        let json = mgr.to_json_string();
        let restored = CrystalAnchorManager::from_json_string(&json);

        // Counts match
        assert_eq!(restored.anchor_count(), 3);
        assert_eq!(restored.paired_pair_count(), 1);
        assert_eq!(restored.list_pending_pairs().len(), 1);

        // pending_solo carried over
        assert!(restored.pending_solo_id.is_some());

        // IDs persist (placing a 4th anchor wouldn't collide with restored ids)
        assert!(restored.next_id > r3.anchor_id);
    }

    #[test]
    fn json_empty_roundtrip() {
        let mgr = CrystalAnchorManager::default();
        let json = mgr.to_json_string();
        let restored = CrystalAnchorManager::from_json_string(&json);
        assert_eq!(restored.anchor_count(), 0);
        assert_eq!(restored.next_id, 1);
    }

    #[test]
    fn json_empty_string_loads_default() {
        let restored = CrystalAnchorManager::from_json_string("");
        assert_eq!(restored.anchor_count(), 0);
    }

    #[test]
    fn grow_pending_bridges_is_idempotent() {
        let mut mgr = CrystalAnchorManager::default();
        let (mut store, config) = make_test_store_for_bridge();

        mgr.place_anchor(v(4.0, 4.0, 4.0), v(0.0, 1.0, 0.0));
        mgr.place_anchor(v(12.0, 4.0, 4.0), v(0.0, 1.0, 0.0));

        grow_pending_bridges(&mut mgr, &mut store, &config, 1.0);
        let grown_after_first = mgr.list_grown_pairs().len();
        // Second call should do nothing — no pairs remain pending
        grow_pending_bridges(&mut mgr, &mut store, &config, 1.0);
        let grown_after_second = mgr.list_grown_pairs().len();
        assert_eq!(grown_after_first, grown_after_second);
    }
}
