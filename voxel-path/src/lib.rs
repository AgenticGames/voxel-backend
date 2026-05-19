//! Voxel-aware 3D pathfinding for the Mithril project.
//!
//! Pure A* over a `CellGrid` trait abstraction. Three movement modes:
//!  - `Flying`: any air cell is traversable (wasps, bats)
//!  - `Walking`: air cells with solid below (hermit crabs, toads, turtles, crocs)
//!  - `Surface`: air cells adjacent to any solid (spiders on floors/walls/ceilings)
//!
//! Surface mode supports full 2D-manifold traversal — a single A* result can
//! span floor→wall→ceiling transitions, with each path node carrying the
//! dominant surface normal for the AI consumer to orient against.
//!
//! Cells are 3D integer coordinates (IVec3). The grid implementation is
//! responsible for mapping cell→world space; the planner is unit-agnostic.
//!
//! Live world integration lives in `voxel-ffi/src/pathing.rs` (`ChunkStoreGrid`).

pub mod grid;
pub mod movement;
pub mod astar;
pub mod smoothing;

#[cfg(test)]
mod tests;

pub use grid::CellGrid;
pub use movement::{MovementMode, can_traverse};
pub use astar::{compute_path, PathRequest, PathOutcome, PathStatus, PathNode};
