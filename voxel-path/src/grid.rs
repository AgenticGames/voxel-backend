//! `CellGrid` trait — the world view that A* sees.
//!
//! Implemented by `voxel-ffi/src/pathing.rs::ChunkStoreGrid` against the live
//! ChunkStore. Stub grids in `tests.rs` use simple in-memory `HashMap`s for
//! standalone testing.

use glam::{IVec3, Vec3};

pub trait CellGrid {
    /// Side length of one pathing cell in voxel units.
    /// e.g. cell_factor = 2 → cell_size = 2.0 voxels = 80 UE units (world_scale = 40).
    fn cell_size(&self) -> f32;

    /// Is this cell occupied by solid voxels? Unloaded chunks should return
    /// `true` so the planner treats them as impassable barriers.
    fn is_solid(&self, cell: IVec3) -> bool;

    /// Is the chunk covering this cell currently loaded?
    /// When false, `is_solid` returns true; the planner uses this signal to
    /// flag `PathStatus::PartiallyUnloaded` so the AI can re-plan once chunks
    /// stream in. Stub grids may always return true.
    fn is_loaded(&self, cell: IVec3) -> bool {
        let _ = cell;
        true
    }

    /// Dominant outward surface normal at `cell` — `Vec3::ZERO` if the cell is
    /// not adjacent to any solid neighbor. Used by `MovementMode::Surface`.
    ///
    /// Implementation guidance: sum `(cell - solid_neighbor) * step` over the
    /// six face neighbors and normalize. For corner cells (multiple equal-area
    /// surfaces) the dominant face wins; the consumer can interpolate between
    /// consecutive path nodes' normals.
    fn surface_normal_at(&self, cell: IVec3) -> Vec3;

    /// Count of solid face-neighbors at `cell` (0..=6). Used by Surface mode
    /// traversability check (`>= 1`) and tie-breaking for path scoring.
    fn solid_face_neighbor_count(&self, cell: IVec3) -> u32 {
        let mut count = 0u32;
        for (dx, dy, dz) in [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ] {
            if self.is_solid(IVec3::new(cell.x + dx, cell.y + dy, cell.z + dz)) {
                count += 1;
            }
        }
        count
    }
}
