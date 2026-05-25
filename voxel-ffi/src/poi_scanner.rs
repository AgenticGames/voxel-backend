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

/// Per-kind score weighting. Reused by both the synchronous scanner and the
/// background tracker — single source of truth, no drift if tuned.
#[derive(Debug, Clone, Copy)]
pub struct ChunkScoreBreakdown {
    pub lava: f32,
    pub water: f32,
    pub stress: f32,
}

impl ChunkScoreBreakdown {
    pub fn best(self) -> (PoiKind, f32) {
        if self.lava >= self.water && self.lava >= self.stress {
            (PoiKind::Lava, self.lava)
        } else if self.water >= self.stress {
            (PoiKind::Water, self.water)
        } else {
            (PoiKind::Stress, self.stress)
        }
    }
}

/// Score from already-counted vote totals. Single source of truth for the
/// "votes → kind scores" mapping (vote thresholds + weight multipliers).
/// Both [`score_chunk`] (sync scanner) and the background tracker call this
/// so the formula can't drift.
pub fn score_from_votes(
    lava: usize,
    water: usize,
    stress: usize,
    // STUB (2026-05-25): topology votes are accepted but not yet scored. See
    // PoiKind comment for context.
    _dome_count: usize,
    _choke_count: usize,
    _niche_count: usize,
) -> ChunkScoreBreakdown {
    let lava_score = if lava >= MIN_LAVA_VOTES {
        lava as f32 * SCORE_PER_LAVA_VOXEL
    } else {
        0.0
    };
    let water_score = if water >= MIN_WATER_VOTES {
        water as f32 * SCORE_PER_WATER_VOXEL
    } else {
        0.0
    };
    let stress_score = if stress >= MIN_STRESS_VOTES {
        stress as f32 * SCORE_PER_STRESS_VOXEL
    } else {
        0.0
    };
    ChunkScoreBreakdown {
        lava: lava_score,
        water: water_score,
        stress: stress_score,
    }
}

/// Count high-stress voxels in a StressField. Extracted so the background
/// tracker can call this under a batched read lock without re-importing
/// the threshold constant.
pub fn count_high_stress_voxels(sf: &voxel_core::stress::StressField) -> usize {
    sf.stress
        .iter()
        .filter(|&&v| v > STRESS_HIGH_THRESHOLD)
        .count()
}

/// Count lava + water voxels in a chunk's fluid cells. Extracted for the
/// same reason as [`count_high_stress_voxels`].
pub fn count_fluid_voxels(cells: &[voxel_fluid::cell::FluidCell]) -> (usize, usize) {
    let (mut lava, mut water) = (0usize, 0usize);
    for c in cells {
        if c.level < FLUID_MIN_LEVEL {
            continue;
        }
        if c.fluid_type.is_lava() {
            lava += 1;
        } else if c.fluid_type.is_water() {
            water += 1;
        }
    }
    (lava, water)
}

/// Score a single chunk given its raw fluid cells (Some if loaded in the
/// snapshot, None if not present) and its stress field (Some if loaded,
/// None if unloaded mid-scan).
pub fn score_chunk(
    fluid_cells: Option<&[voxel_fluid::cell::FluidCell]>,
    stress_field: Option<&voxel_core::stress::StressField>,
) -> ChunkScoreBreakdown {
    let (lava, water) = fluid_cells.map(count_fluid_voxels).unwrap_or((0, 0));
    let stress = stress_field.map(count_high_stress_voxels).unwrap_or(0);
    score_from_votes(lava, water, stress, 0, 0, 0)
}

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
    // STUB (2026-05-25): topology kinds are referenced by poi_tracker.rs but
    // the full topology-vote pipeline (in-flight as of 2026-05-25 review per
    // PERF_REVIEW_2026-05-25.md) hasn't landed yet. These variants exist so
    // the crate compiles; scoring never picks them with the current
    // ChunkScoreBreakdown.best(), so they're inert at runtime. Replace with
    // real selection logic once the topology probe ships.
    CeilingDome = 4,
    Chokepoint = 5,
    WallNiche = 6,
}

/// STUB (2026-05-25): topology-vote totals — see PoiKind comment. Real impl
/// will replace this with per-kind voxel counts + centroid tracking.
#[derive(Debug, Clone, Copy, Default)]
pub struct TopologyVotes {
    pub dome_count: usize,
    pub choke_count: usize,
    pub niche_count: usize,
    pub centroid: glam::IVec3,
}

impl TopologyVotes {
    /// Centroid offset (in chunk-local voxel coords) for a given topology
    /// kind. Stub returns the cached centroid regardless of kind.
    pub fn centroid_for(&self, _kind: PoiKind, _chunk_size: usize) -> glam::IVec3 {
        self.centroid
    }
}

/// STUB (2026-05-25): cross-chunk topology counter. Real impl scans
/// neighboring chunks to identify domes / chokepoints / niches. Returns
/// zeros so the topology kinds never win the scoring race.
pub fn count_topology_votes_cross_chunk(
    _store: &ChunkStore,
    _coord: (i32, i32, i32),
    _chunk_size: usize,
) -> TopologyVotes {
    TopologyVotes::default()
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

    // ─── Per-chunk pass via the shared scorer ────────────────────────
    // Union of fluid + stress chunk coords so we score every chunk that has
    // *any* signal (chunk without stress field can still have fluid cells).
    let mut all_coords: std::collections::HashSet<(i32, i32, i32)> =
        std::collections::HashSet::new();
    all_coords.extend(fluid_snap.chunks.keys().copied());
    all_coords.extend(store.stress_fields.keys().copied());

    for chunk_coord in all_coords {
        let fluid_cells = fluid_snap.chunks.get(&chunk_coord).map(|v| v.as_slice());
        let stress_field = store.stress_fields.get(&chunk_coord);
        let breakdown = score_chunk(fluid_cells, stress_field);
        let (best_kind, best_score) = breakdown.best();
        if best_score <= 0.0 {
            continue;
        }
        candidates.push(Poi {
            kind: best_kind,
            score: best_score,
            chunk_coord,
            center_world_rust: chunk_center_world(chunk_coord, cs_f),
            extent_radius_voxels: cs_f * 0.5,
        });
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
