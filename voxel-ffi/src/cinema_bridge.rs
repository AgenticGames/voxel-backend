//! voxel-ffi ↔ voxel-cinema integration.
//!
//! voxel-cinema is a pure library that doesn't depend on voxel-ffi. This
//! module bridges the gap: it wraps the engine's `ChunkStoreGrid` and
//! `surface_probe` into the shapes voxel-cinema expects, then dispatches
//! to `voxel_cinema::compose`.
//!
//! The FFI surface `voxel_request_shot_candidates` (in api.rs, added in
//! C13) routes through this helper.

use glam::Vec3;
use voxel_cinema::{compose, IntentMask, ShotCandidate};
use voxel_world_memory::scene::{Scene, SceneId};

use crate::engine::VoxelEngine;
use crate::pathing::ChunkStoreGrid;
use crate::surface_probe::{probe_surface, SurfaceKind as FfiSurfaceKind};

/// Look up a Scene by id from the engine's WorldMemory and compose shot
/// candidates against the live cave geometry. Returns at most `count`
/// candidates, sorted by composition score.
pub fn compose_for_engine(
    engine: &VoxelEngine,
    scene_id: SceneId,
    intent_mask: IntentMask,
    count: usize,
) -> Vec<ShotCandidate> {
    let scene = match engine.world_memory.scenes.get(&scene_id) {
        Some(s) => s.value().clone(),
        None => return Vec::new(),
    };
    compose_for_scene(engine, &scene, intent_mask, count)
}

/// Compose against a Scene already in hand (test path + CLI). Allows
/// callers that already have the Scene to avoid the DashMap lookup.
pub fn compose_for_scene(
    engine: &VoxelEngine,
    scene: &Scene,
    intent_mask: IntentMask,
    count: usize,
) -> Vec<ShotCandidate> {
    let chunk_size = engine.chunk_size();
    // Acquire the read lock for the full compose duration. Compose touches
    // density + stress via the grid wrapper + the probe closure. Block 1
    // target: <50 ms p99 — well under any writer-starvation concern.
    let store_arc = engine.store_arc();
    let store = match store_arc.read() {
        Ok(g) => g,
        Err(_) => return Vec::new(),
    };

    // ChunkStoreGrid wraps the chunk store + chunk_size for path-planner
    // queries. cell_factor=2 matches the planner's standard cell resolution.
    let grid = ChunkStoreGrid {
        store: &store,
        chunk_size,
        cell_factor: 2,
        occupied_cells: None,
        requester_cell: None,
    };

    // Surface probe closure — wraps `surface_probe::probe_surface` and
    // converts ProbeResult into voxel-cinema's neutral ProbeData shape.
    let probe = |rust_pos: Vec3, hint: Vec3| -> Option<voxel_cinema::ProbeData> {
        let result = probe_surface(&store, rust_pos, chunk_size, hint);
        // Convert SurfaceKind enums (both are u8-compatible).
        let kind = match result.kind {
            FfiSurfaceKind::Solid => voxel_cinema::probe::SurfaceKind::Solid,
            FfiSurfaceKind::AirOpen => voxel_cinema::probe::SurfaceKind::AirOpen,
            FfiSurfaceKind::Floor => voxel_cinema::probe::SurfaceKind::Floor,
            FfiSurfaceKind::Wall => voxel_cinema::probe::SurfaceKind::Wall,
            FfiSurfaceKind::Ceiling => voxel_cinema::probe::SurfaceKind::Ceiling,
            FfiSurfaceKind::Overhang => voxel_cinema::probe::SurfaceKind::Overhang,
        };
        Some(voxel_cinema::ProbeData {
            kind,
            normal: [result.normal.x, result.normal.y, result.normal.z],
            cavity_radius: result.cavity_radius,
            clearance_rust: result.clearance_rust,
        })
    };

    compose(scene, intent_mask, count, &grid, &probe)
}

#[cfg(test)]
mod tests {
    // Engine-level tests for compose_for_engine live in api.rs's test mod
    // alongside the FFI integration tests (C13). This module is just the
    // bridge; per-intent compose logic is exercised in voxel-cinema's own
    // tests.
}
