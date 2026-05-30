//! Shared brush types and private helpers used across the brush submodules.
//!
//! The five helpers below were file-private in the original `brushes.rs`; they
//! are promoted to `pub(crate)` so the split submodules can call them. Behavior
//! is unchanged — pure relocation.

use glam::Vec3;
use voxel_gen::config::GenerationConfig;

use crate::delta::ChunkSnapshot;
use crate::store::{sync_boundary_density, ChunkStore};
use crate::types::ConvertedMesh;

/// One stroke of brush history. Stores pre-state of every chunk in the brush AABB
/// so the operation can be reversed exactly via `apply_to`.
pub struct UndoStroke {
    pub snapshots: Vec<((i32, i32, i32), ChunkSnapshot)>,
    /// Pre-state mushroom placements per chunk (for mushroom brushes that
    /// don't modify density). Empty when the stroke only touched density.
    /// On undo, each chunk's `mushroom_placements` entry is replaced wholesale.
    pub mushroom_snapshots: Vec<((i32, i32, i32), Vec<voxel_gen::MushroomPlacement>)>,
}

/// Capture pre-state snapshots for any density-loaded chunks in `[lo..=hi]`,
/// push as a single undo stroke. Bounded by `store.undo_max_depth` — oldest
/// strokes are dropped when full.
///
/// Captures BOTH density+material and (if present) the painted-stress overlay
/// so PaintStress-brush undo round-trips correctly. Chunks with no painted
/// layer still cost ~0 extra bytes (Option<Vec<u8>> stays `None`).
pub(crate) fn capture_undo_for_range(
    store: &mut ChunkStore,
    lo: (i32, i32, i32),
    hi: (i32, i32, i32),
) {
    let mut snapshots = Vec::new();
    for cz in lo.2..=hi.2 {
        for cy in lo.1..=hi.1 {
            for cx in lo.0..=hi.0 {
                let key = (cx, cy, cz);
                if let Some(density) = store.density_fields.get(&key) {
                    let sf = store.stress_fields.get(&key);
                    let supf = store.support_fields.get(&key);
                    snapshots.push((key, ChunkSnapshot::from_chunk(density, sf, supf)));
                }
            }
        }
    }
    if snapshots.is_empty() {
        return;
    }
    store.undo_stack.push_back(UndoStroke { snapshots, mushroom_snapshots: Vec::new() });
    while store.undo_stack.len() > store.undo_max_depth {
        store.undo_stack.pop_front();
    }
}

/// Capture pre-state mushroom placements for the given chunks and push a
/// mushroom-only undo stroke. Used by the mushroom paint/erase brushes which
/// don't modify density.
pub(crate) fn capture_mushroom_undo(store: &mut ChunkStore, keys: &[(i32, i32, i32)]) {
    if keys.is_empty() { return; }
    let mut snaps = Vec::with_capacity(keys.len());
    for &k in keys {
        let prior = store.mushroom_placements.get(&k).cloned().unwrap_or_default();
        snaps.push((k, prior));
    }
    store.undo_stack.push_back(UndoStroke {
        snapshots: Vec::new(),
        mushroom_snapshots: snaps,
    });
    while store.undo_stack.len() > store.undo_max_depth {
        store.undo_stack.pop_front();
    }
}

pub struct BrushOutcome {
    pub meshes: Vec<((i32, i32, i32), ConvertedMesh)>,
    /// Chunks where material/solidity actually flipped (for crystal recompute).
    pub flipped_chunks: Vec<(i32, i32, i32)>,
}

/// Compute the inclusive chunk-coord range overlapping a sphere.
pub(crate) fn chunk_range_for_sphere(center: Vec3, radius: f32, eb: f32) -> ((i32, i32, i32), (i32, i32, i32)) {
    let lo = (
        ((center.x - radius) / eb).floor() as i32,
        ((center.y - radius) / eb).floor() as i32,
        ((center.z - radius) / eb).floor() as i32,
    );
    let hi = (
        ((center.x + radius) / eb).floor() as i32,
        ((center.y + radius) / eb).floor() as i32,
        ((center.z + radius) / eb).floor() as i32,
    );
    (lo, hi)
}

/// Standard "iterate one chunk's voxels overlapping a sphere" loop body.
/// Returns local grid bounds for the dirty rect plus a `changed` flag.
pub(crate) fn local_sphere_bounds(
    center: Vec3,
    radius: f32,
    origin: Vec3,
    vs: f32,
    grid_size: usize,
) -> (Vec3, f32, usize, usize, usize, usize, usize, usize) {
    let grid_center = (center - origin) / vs;
    let grid_radius = radius / vs;
    let lo_x = ((grid_center.x - grid_radius).floor() as i32).max(0) as usize;
    let hi_x = ((grid_center.x + grid_radius).ceil() as usize + 1).min(grid_size);
    let lo_y = ((grid_center.y - grid_radius).floor() as i32).max(0) as usize;
    let hi_y = ((grid_center.y + grid_radius).ceil() as usize + 1).min(grid_size);
    let lo_z = ((grid_center.z - grid_radius).floor() as i32).max(0) as usize;
    let hi_z = ((grid_center.z + grid_radius).ceil() as usize + 1).min(grid_size);
    (grid_center, grid_radius, lo_x, lo_y, lo_z, hi_x, hi_y, hi_z)
}

pub(crate) fn finalize_brush(
    store: &mut ChunkStore,
    mut dirty_chunks: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)>,
    config: &GenerationConfig,
    world_scale: f32,
) -> BrushOutcome {
    let flipped_chunks: Vec<(i32, i32, i32)> = dirty_chunks.iter().map(|&(k, ..)| k).collect();

    let extra_dirty = sync_boundary_density(
        &mut store.density_fields,
        &dirty_chunks,
        config.chunk_size,
    );
    dirty_chunks.extend(extra_dirty);

    let dirty_keys: Vec<_> = dirty_chunks.iter().map(|&(k, ..)| k).collect();
    store.modification_tracker.mark_dirty_many(&dirty_keys);

    let meshes = store.remesh_dirty(&dirty_chunks, config, world_scale);
    BrushOutcome { meshes, flipped_chunks }
}
