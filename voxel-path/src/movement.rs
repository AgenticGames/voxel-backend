//! Movement modes — drives the per-mode traversability predicate that A*
//! applies to candidate cells.

use crate::grid::CellGrid;
use glam::IVec3;

#[derive(Debug, Clone, Copy)]
pub enum MovementMode {
    /// Any air cell is traversable. For wasps, bats, future flying enemies.
    /// `agent_radius_cells` reserved for future clearance checks; current
    /// implementation treats it as informational.
    Flying { agent_radius_cells: f32 },

    /// Air cells with solid directly below. Models grounded creatures that
    /// fall under gravity (hermit crabs, toads, turtles, crocs).
    Walking { agent_radius_cells: f32 },

    /// Air cells with at least one solid face-neighbor. Spiders walking on
    /// floors, walls, ceilings, and across normal transitions at corners.
    Surface { agent_radius_cells: f32 },
}

/// Can an agent in this mode occupy `cell`?
///
/// Wraps the grid's traversability predicate. Used by A* both to expand
/// neighbors and to validate the start/goal cells.
pub fn can_traverse<G: CellGrid>(grid: &G, cell: IVec3, mode: MovementMode) -> bool {
    if grid.is_solid(cell) {
        return false;
    }
    match mode {
        MovementMode::Flying { .. } => true,
        MovementMode::Walking { .. } => {
            // Require solid floor below — grounded movement.
            grid.is_solid(IVec3::new(cell.x, cell.y, cell.z - 1))
        }
        MovementMode::Surface { .. } => {
            // Any solid face-neighbor counts — spider can adhere.
            grid.solid_face_neighbor_count(cell) >= 1
        }
    }
}

impl MovementMode {
    /// Extract the agent radius (in pathing-cell units) regardless of variant.
    pub fn agent_radius(self) -> f32 {
        match self {
            MovementMode::Flying { agent_radius_cells }
            | MovementMode::Walking { agent_radius_cells }
            | MovementMode::Surface { agent_radius_cells } => agent_radius_cells,
        }
    }

    pub fn is_surface(self) -> bool {
        matches!(self, MovementMode::Surface { .. })
    }
}
