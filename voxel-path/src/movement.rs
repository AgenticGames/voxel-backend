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
    // Clearance shell (2026-08-01): `agent_radius_cells` was "informational"
    // for years — so shortest-path routes hugged every wall and skimmed every
    // lip, exactly where the render mesh bulges past the voxel boundary (48
    // of 90 fine-cell sense-trail corridor points sat inside UE collision).
    // With a radius of a full cell or more, the cell alone cannot contain the
    // agent: require the face-neighbor shell open too, giving routes a
    // one-cell standoff from geometry. Radius-driven, so coarse-grid AI
    // (radius < 1 cell there) keeps its old behavior. Surface mode is exempt
    // — wall-hugging is its job — and Walking exempts the floor it stands on.
    let shell = mode.agent_radius() >= 1.0;
    match mode {
        MovementMode::Flying { .. } => {
            !shell || face_shell_open(grid, cell, /*skip_below=*/false)
        }
        MovementMode::Walking { .. } => {
            // Require solid floor below — grounded movement. ⚠️ UP IS +Y:
            // the live grid (voxel-ffi ChunkStoreGrid) feeds A* Rust voxel
            // coords, where UE +Z (up) maps to Rust +Y (see voxel-ffi
            // convert.rs / nodes_to_ue). This read `cell.z - 1` until
            // 2026-08-01 — a HORIZONTAL neighbor — which rejected every open
            // floor cell in the game and silently disabled Walking mode
            // world-wide (sense trail #206, creature AI never pathed).
            if !grid.is_solid(IVec3::new(cell.x, cell.y - 1, cell.z)) {
                return false;
            }
            !shell || face_shell_open(grid, cell, /*skip_below=*/true)
        }
        MovementMode::Surface { .. } => {
            // Any solid face-neighbor counts — spider can adhere.
            grid.solid_face_neighbor_count(cell) >= 1
        }
    }
}

/// All face-neighbors of `cell` open (optionally ignoring the one below,
/// which Walking REQUIRES to be solid). +Y is up.
fn face_shell_open<G: CellGrid>(grid: &G, cell: IVec3, skip_below: bool) -> bool {
    const FACES: [(i32, i32, i32); 6] = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ];
    for (dx, dy, dz) in FACES {
        if skip_below && dy == -1 {
            continue;
        }
        if grid.is_solid(IVec3::new(cell.x + dx, cell.y + dy, cell.z + dz)) {
            return false;
        }
    }
    true
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
