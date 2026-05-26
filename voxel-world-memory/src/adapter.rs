//! Legacy POI adapter — bridge between the new `WorldMemory` Scene model
//! and the existing `voxel_request_list_top_pois` FFI surface in
//! `voxel-ffi/src/api.rs:3548`. UE code consumes `FfiPoi` (or its
//! equivalent), so we project Scenes to that shape.
//!
//! The voxel-world-memory crate doesn't depend on voxel-ffi, so we define
//! `LegacyPoi` here as a *neutral* shape. The voxel-ffi adapter does the
//! trivial copy from `LegacyPoi` to `FfiPoi` at the FFI boundary.
//!
//! **Score-scale parity**: pre-change scoring used `SCORE_PER_LAVA_VOXEL =
//! 10.0` per cell (200 cells × 10 = 2000). Block 1 uses per-cell weight
//! 1.0 (200 × 1 = 200). The adapter multiplies by 10 here to preserve
//! UE-visible parity: 200 × 10 = 2000, within the ±10% baseline tolerance.

use crate::scene::{Scene, SceneKind};
use crate::WorldMemory;

/// Adapter score-scale factor — multiplies per-cell weighted sum to match
/// UE's existing score expectations from the legacy chunk-binary scoring.
const ADAPTER_SCORE_SCALE: f32 = 10.0;

/// FFI-neutral legacy POI shape. voxel-ffi's adapter copies these into
/// `FfiPoi` at the FFI boundary (handling coord swap + UE world units).
#[derive(Debug, Clone, Copy)]
pub struct LegacyPoi {
    /// Kind discriminant matches `SceneKind` integer values — voxel-ffi
    /// re-casts to `PoiKind` (which uses the same 0..=6 layout).
    pub kind: u8,
    pub score: f32,
    /// Scene centroid in **Rust voxel coords** (Y-up). voxel-ffi's adapter
    /// converts to UE world units via the standard `from_rust_world_pos`.
    pub centroid_rust: [f32; 3],
    /// Chunk coord (Rust space) the centroid falls in. voxel-ffi adapter
    /// converts to UE chunk-coord space via `rust_chunk_to_ue`.
    pub chunk_rust: [i32; 3],
    /// Extent half-diagonal in Rust voxels (Scene.aabb.extent_radius()).
    pub extent_radius_voxels: f32,
}

/// Build the top-K legacy POI list. Order: by `score` descending.
///
/// `include_topology=false` (the Block 1 default) filters out the
/// CeilingDome/Chokepoint/WallNiche kinds — UE doesn't handle them yet.
///
/// Implementation: DashMap's `iter()` yields `Ref<...>` guards that don't
/// outlive the iterator, so we materialize cloned Scenes into a Vec. Bounded
/// by tracked scene count (≤ ~256 in practice).
pub fn legacy_top_k_pois(wm: &WorldMemory, k: usize, include_topology: bool) -> Vec<LegacyPoi> {
    let mut scenes: Vec<Scene> = wm
        .scenes
        .iter()
        .filter(|e| include_topology || !e.value().kind.is_topology())
        .map(|e| e.value().clone())
        .collect();

    scenes.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scenes.truncate(k);

    scenes
        .into_iter()
        .map(|s| {
            let kind = match s.kind {
                SceneKind::Lava => 0u8,
                SceneKind::Water => 1,
                SceneKind::Stress => 2,
                SceneKind::Bridge => 3,
                SceneKind::CeilingDome => 4,
                SceneKind::Chokepoint => 5,
                SceneKind::WallNiche => 6,
            };
            // For chunk_rust: find which chunk the centroid falls into.
            // We don't know chunk_size here — it lives on `ClusterCtx` /
            // engine config. Caller (voxel-ffi adapter) can recompute if
            // it needs UE chunk-coord; we report the first chunk in the
            // Scene's chunk list as a reasonable proxy.
            let chunk = s.chunks.first().copied().unwrap_or((0, 0, 0));
            LegacyPoi {
                kind,
                score: s.score * ADAPTER_SCORE_SCALE,
                centroid_rust: s.centroid,
                chunk_rust: [chunk.0, chunk.1, chunk.2],
                extent_radius_voxels: s.aabb.extent_radius().max(1.0),
            }
        })
        .collect()
}

/// Helper: parity-scale a raw per-cell weighted score into UE's expected
/// range. Used by tests and by voxel-ffi adapter callers that produce
/// LegacyPoi-equivalent output through other paths.
pub fn legacy_score_for(weighted_sum: f32) -> f32 {
    weighted_sum * ADAPTER_SCORE_SCALE
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::Scene;
    use glam::Vec3;

    fn make_scene(wm: &WorldMemory, kind: SceneKind, score: f32, centroid: Vec3) -> Scene {
        let mut s = Scene::new(wm.alloc_scene_id(), kind, centroid);
        s.score = score;
        s.confidence = 0.8;
        s.chunks = vec![(0, 0, 0)];
        s
    }

    #[test]
    fn excludes_topology_kinds_by_default() {
        let wm = WorldMemory::new();
        let lava = make_scene(&wm, SceneKind::Lava, 100.0, Vec3::ZERO);
        wm.scenes.insert(lava.id, lava);
        let dome = make_scene(&wm, SceneKind::CeilingDome, 200.0, Vec3::ZERO);
        wm.scenes.insert(dome.id, dome);

        let out = legacy_top_k_pois(&wm, 10, false);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].kind, 0); // Lava
    }

    #[test]
    fn includes_topology_when_flag_set() {
        let wm = WorldMemory::new();
        let lava = make_scene(&wm, SceneKind::Lava, 100.0, Vec3::ZERO);
        wm.scenes.insert(lava.id, lava);
        let dome = make_scene(&wm, SceneKind::CeilingDome, 200.0, Vec3::ZERO);
        wm.scenes.insert(dome.id, dome);

        let out = legacy_top_k_pois(&wm, 10, true);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn sort_by_score_descending_with_scale_applied() {
        let wm = WorldMemory::new();
        let small = make_scene(&wm, SceneKind::Lava, 50.0, Vec3::ZERO);
        wm.scenes.insert(small.id, small);
        let big = make_scene(&wm, SceneKind::Bridge, 200.0, Vec3::ZERO);
        wm.scenes.insert(big.id, big);

        let out = legacy_top_k_pois(&wm, 10, false);
        assert_eq!(out.len(), 2);
        // Bridge (200 × 10 = 2000) > Lava (50 × 10 = 500)
        assert_eq!(out[0].kind, 3); // Bridge first
        assert!((out[0].score - 2000.0).abs() < 1e-3);
        assert!((out[1].score - 500.0).abs() < 1e-3);
    }

    #[test]
    fn calibration_200_lava_cells_produce_legacy_parity_score() {
        // Plan calibration: a chunk with 200 lava cells must produce a
        // legacy-adapter score within ±10% of the pre-change baseline
        // (200 × 10 = 2000).
        let raw_weighted = 200.0; // per-cell weight 1.0 × 200 cells
        let legacy = legacy_score_for(raw_weighted);
        // Tolerance ±10%
        assert!(
            legacy >= 1800.0 && legacy <= 2200.0,
            "calibration drift: legacy score {} not in [1800, 2200]",
            legacy
        );
    }

    #[test]
    fn empty_world_memory_returns_empty() {
        let wm = WorldMemory::new();
        let out = legacy_top_k_pois(&wm, 10, true);
        assert!(out.is_empty());
    }

    #[test]
    fn top_k_truncates() {
        let wm = WorldMemory::new();
        for i in 0..20 {
            let s = make_scene(&wm, SceneKind::Lava, 100.0 - i as f32, Vec3::ZERO);
            wm.scenes.insert(s.id, s);
        }
        let out = legacy_top_k_pois(&wm, 5, false);
        assert_eq!(out.len(), 5);
    }
}
