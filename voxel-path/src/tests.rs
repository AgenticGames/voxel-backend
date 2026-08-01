//! Tests against synthesized in-memory grids. No dependency on voxel-ffi or
//! the live ChunkStore — pure algorithm verification.

use crate::astar::{compute_path, PathRequest, PathStatus};
use crate::grid::CellGrid;
use crate::movement::MovementMode;
use glam::{IVec3, Vec3};
use std::collections::HashMap;

/// Stub grid: explicitly-marked solid cells. Anything else is air. All cells
/// are "loaded."
#[derive(Default)]
struct StubGrid {
    solids: HashMap<(i32, i32, i32), ()>,
}

impl StubGrid {
    fn set_solid(&mut self, c: IVec3) {
        self.solids.insert((c.x, c.y, c.z), ());
    }

    fn fill_box(&mut self, lo: IVec3, hi: IVec3) {
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    self.set_solid(IVec3::new(x, y, z));
                }
            }
        }
    }
}

impl CellGrid for StubGrid {
    fn cell_size(&self) -> f32 { 1.0 }
    fn is_solid(&self, cell: IVec3) -> bool {
        self.solids.contains_key(&(cell.x, cell.y, cell.z))
    }
    fn surface_normal_at(&self, cell: IVec3) -> Vec3 {
        let mut n = Vec3::ZERO;
        for (dx, dy, dz) in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)] {
            if self.is_solid(IVec3::new(cell.x+dx, cell.y+dy, cell.z+dz)) {
                n -= Vec3::new(dx as f32, dy as f32, dz as f32);
            }
        }
        n.normalize_or_zero()
    }
}

// ─── Flying ─────────────────────────────────────────────────────

#[test]
fn flying_straight_line_open_space() {
    let grid = StubGrid::default();
    let req = PathRequest {
        from: IVec3::new(0, 0, 0),
        to: IVec3::new(5, 0, 0),
        mode: MovementMode::Flying { agent_radius_cells: 0.5 },
        smooth: false,
        ..Default::default()
    };
    let outcome = compute_path(&grid, req);
    assert_eq!(outcome.status, PathStatus::Success);
    assert!(outcome.nodes.len() >= 6);
    assert_eq!(outcome.nodes.first().unwrap().cell, IVec3::new(0, 0, 0));
    assert_eq!(outcome.nodes.last().unwrap().cell, IVec3::new(5, 0, 0));
}

#[test]
fn flying_around_pillar() {
    let mut grid = StubGrid::default();
    // Solid pillar from (2,-10,0) to (2,10,5) blocks direct line.
    grid.fill_box(IVec3::new(2, -10, 0), IVec3::new(2, 10, 5));
    let req = PathRequest {
        from: IVec3::new(0, 0, 2),
        to: IVec3::new(5, 0, 2),
        mode: MovementMode::Flying { agent_radius_cells: 0.5 },
        smooth: false,
        ..Default::default()
    };
    let outcome = compute_path(&grid, req);
    assert_eq!(outcome.status, PathStatus::Success);
    // Path must avoid x=2 cells along y=0
    for n in &outcome.nodes {
        if n.cell.x == 2 && n.cell.y == 0 && n.cell.z >= 0 && n.cell.z <= 5 {
            panic!("path passes through pillar at {:?}", n.cell);
        }
    }
}

#[test]
fn flying_no_path_through_sealed_box() {
    // Start inside a hollow 1-cell at (0,0,0); seal it with 6 walls
    let mut grid = StubGrid::default();
    grid.set_solid(IVec3::new(1, 0, 0));
    grid.set_solid(IVec3::new(-1, 0, 0));
    grid.set_solid(IVec3::new(0, 1, 0));
    grid.set_solid(IVec3::new(0, -1, 0));
    grid.set_solid(IVec3::new(0, 0, 1));
    grid.set_solid(IVec3::new(0, 0, -1));

    let req = PathRequest {
        from: IVec3::new(0, 0, 0),
        to: IVec3::new(5, 0, 0),
        mode: MovementMode::Flying { agent_radius_cells: 0.5 },
        ..Default::default()
    };
    let outcome = compute_path(&grid, req);
    assert_eq!(outcome.status, PathStatus::NoPath);
}

#[test]
fn max_nodes_early_termination() {
    let grid = StubGrid::default();
    // Far-apart points + tiny budget → MaxNodesReached
    let req = PathRequest {
        from: IVec3::new(0, 0, 0),
        to: IVec3::new(1000, 0, 0),
        mode: MovementMode::Flying { agent_radius_cells: 0.5 },
        max_nodes: 10,
        ..Default::default()
    };
    let outcome = compute_path(&grid, req);
    assert_eq!(outcome.status, PathStatus::MaxNodesReached);
}

// ─── Walking ────────────────────────────────────────────────────

// ⚠️ UP IS +Y in these worlds — matching the live ChunkStoreGrid convention
// (UE +Z maps to Rust +Y at the FFI boundary). These tests were originally
// written z-up, which let the z-up bug in the Walking predicate pass its own
// suite while rejecting every real cell in the game (#206).

#[test]
fn walking_requires_floor_below() {
    let mut grid = StubGrid::default();
    // Floor at y=0
    for x in 0..=5 {
        for z in -1..=1 {
            grid.set_solid(IVec3::new(x, 0, z));
        }
    }
    let req = PathRequest {
        from: IVec3::new(0, 1, 0),
        to: IVec3::new(5, 1, 0),
        mode: MovementMode::Walking { agent_radius_cells: 0.5 },
        smooth: false,
        ..Default::default()
    };
    let outcome = compute_path(&grid, req);
    assert_eq!(outcome.status, PathStatus::Success);
    for n in &outcome.nodes {
        // Every node must have solid directly below (y-1) — that's the
        // Walking traversability predicate.
        assert!(
            grid.is_solid(IVec3::new(n.cell.x, n.cell.y - 1, n.cell.z)),
            "node {:?} has no floor below",
            n.cell
        );
    }
}

#[test]
fn walking_no_path_over_pit() {
    let mut grid = StubGrid::default();
    // Floor at y=0 with a pit at x=3,4 (no floor there)
    for x in 0..=10 {
        if x == 3 || x == 4 { continue; }
        for z in -1..=1 {
            grid.set_solid(IVec3::new(x, 0, z));
        }
    }
    let req = PathRequest {
        from: IVec3::new(0, 1, 0),
        to: IVec3::new(10, 1, 0),
        mode: MovementMode::Walking { agent_radius_cells: 0.5 },
        smooth: false,
        ..Default::default()
    };
    let outcome = compute_path(&grid, req);
    // No floor means no walking path
    assert_eq!(outcome.status, PathStatus::NoPath);
}

#[test]
fn flying_radius_one_requires_face_shell() {
    // Open 5³ air box with a single solid voxel at the origin. A wide agent
    // (radius ≥ 1 cell) may not occupy the solid's face-neighbors; a thin
    // agent may. Diagonal neighbors stay legal for both.
    let mut grid = StubGrid::default();
    grid.set_solid(IVec3::new(0, 0, 0));
    let wide = MovementMode::Flying { agent_radius_cells: 1.5 };
    let thin = MovementMode::Flying { agent_radius_cells: 0.5 };
    assert!(!crate::movement::can_traverse(&grid, IVec3::new(1, 0, 0), wide));
    assert!(crate::movement::can_traverse(&grid, IVec3::new(1, 0, 0), thin));
    assert!(crate::movement::can_traverse(&grid, IVec3::new(1, 1, 0), wide));
}

#[test]
fn walking_shell_ignores_the_floor_it_stands_on() {
    // Flat open floor at y=0: a wide Walking agent must still be able to
    // stand one above it (the shell skips the below-neighbor Walking
    // REQUIRES solid), but a solid wall voxel beside it blocks the wide
    // agent and not the thin one.
    let mut grid = StubGrid::default();
    for x in -3..=3 {
        for z in -3..=3 {
            grid.set_solid(IVec3::new(x, 0, z));
        }
    }
    let wide = MovementMode::Walking { agent_radius_cells: 1.5 };
    let thin = MovementMode::Walking { agent_radius_cells: 0.5 };
    assert!(crate::movement::can_traverse(&grid, IVec3::new(0, 1, 0), wide));
    grid.set_solid(IVec3::new(1, 1, 0)); // wall voxel beside the agent
    assert!(!crate::movement::can_traverse(&grid, IVec3::new(0, 1, 0), wide));
    assert!(crate::movement::can_traverse(&grid, IVec3::new(0, 1, 0), thin));
}

#[test]
fn walking_rejects_open_floor_when_up_axis_misread() {
    // Regression guard for #206: an open chamber — flat floor at y=0, air
    // everywhere above, nothing along z. A cell one above the floor MUST be
    // traversable. The old z-up predicate looked sideways for its "floor",
    // found air, and rejected every standable cell in the world.
    let mut grid = StubGrid::default();
    for x in -3..=3 {
        for z in -3..=3 {
            grid.set_solid(IVec3::new(x, 0, z));
        }
    }
    let mode = MovementMode::Walking { agent_radius_cells: 0.5 };
    assert!(
        crate::movement::can_traverse(&grid, IVec3::new(0, 1, 0), mode),
        "cell directly above an open floor must be walkable"
    );
    assert!(
        !crate::movement::can_traverse(&grid, IVec3::new(0, 2, 0), mode),
        "cell floating two above the floor must not be walkable"
    );
}

// ─── Surface ────────────────────────────────────────────────────

#[test]
fn surface_l_shape_floor_to_wall() {
    let mut grid = StubGrid::default();
    // Floor along z=0, y in [-1,1], x in [0..5]
    for x in 0..=5 {
        for y in -1..=1 {
            grid.set_solid(IVec3::new(x, y, 0));
        }
    }
    // Wall rising at x=5, z in [1..5], y in [-1,1]
    for z in 1..=5 {
        for y in -1..=1 {
            grid.set_solid(IVec3::new(5, y, z));
        }
    }
    // Start: on floor near origin. Goal: on wall at top.
    let req = PathRequest {
        from: IVec3::new(0, 0, 1),
        to: IVec3::new(4, 0, 5),
        mode: MovementMode::Surface { agent_radius_cells: 0.5 },
        smooth: false,
        ..Default::default()
    };
    let outcome = compute_path(&grid, req);
    assert_eq!(outcome.status, PathStatus::Success);
    assert!(outcome.nodes.len() >= 4);
    // First node on floor normal ≈ +Z; last on wall, normal ≈ -X
    let first_n = outcome.nodes.first().unwrap().surface_normal;
    let last_n = outcome.nodes.last().unwrap().surface_normal;
    assert!(first_n.z > 0.5, "first node should face up, got {:?}", first_n);
    assert!(last_n.x < -0.5, "last node should face -X, got {:?}", last_n);
}

#[test]
fn surface_picks_solid_adjacent_only() {
    // Floor at z=0; isolated air cell at z=5 with no solid neighbor
    let mut grid = StubGrid::default();
    for x in 0..=5 {
        for y in -1..=1 {
            grid.set_solid(IVec3::new(x, y, 0));
        }
    }
    let req = PathRequest {
        from: IVec3::new(0, 0, 1),
        to: IVec3::new(5, 0, 1),
        mode: MovementMode::Surface { agent_radius_cells: 0.5 },
        smooth: false,
        ..Default::default()
    };
    let outcome = compute_path(&grid, req);
    assert_eq!(outcome.status, PathStatus::Success);
    for n in &outcome.nodes {
        // Every node must be surface-adjacent (solid_face_neighbor_count >= 1)
        let count = grid.solid_face_neighbor_count(n.cell);
        assert!(count >= 1, "node {:?} has no solid neighbor", n.cell);
    }
}

// ─── Smoothing ──────────────────────────────────────────────────

#[test]
fn theta_smoothing_collinear_collapse() {
    let grid = StubGrid::default();
    let req = PathRequest {
        from: IVec3::new(0, 0, 0),
        to: IVec3::new(10, 0, 0),
        mode: MovementMode::Flying { agent_radius_cells: 0.5 },
        smooth: true,
        ..Default::default()
    };
    let raw_req = PathRequest { smooth: false, ..req.clone() };
    let raw = compute_path(&grid, raw_req);
    let smoothed = compute_path(&grid, req);
    assert_eq!(smoothed.status, PathStatus::Success);
    assert!(smoothed.nodes.len() < raw.nodes.len(),
        "smoothed should be shorter — raw {} vs smoothed {}",
        raw.nodes.len(), smoothed.nodes.len());
    // Should always retain endpoints.
    assert_eq!(smoothed.nodes.first().unwrap().cell, IVec3::new(0, 0, 0));
    assert_eq!(smoothed.nodes.last().unwrap().cell, IVec3::new(10, 0, 0));
}

#[test]
fn same_cell_trivial_path() {
    let grid = StubGrid::default();
    let req = PathRequest {
        from: IVec3::new(3, 4, 5),
        to: IVec3::new(3, 4, 5),
        mode: MovementMode::Flying { agent_radius_cells: 0.5 },
        ..Default::default()
    };
    let outcome = compute_path(&grid, req);
    assert_eq!(outcome.status, PathStatus::Success);
    assert_eq!(outcome.nodes.len(), 1);
}

#[test]
fn solid_start_invalid_endpoint() {
    let mut grid = StubGrid::default();
    grid.set_solid(IVec3::new(0, 0, 0));
    let req = PathRequest {
        from: IVec3::new(0, 0, 0),
        to: IVec3::new(5, 0, 0),
        mode: MovementMode::Flying { agent_radius_cells: 0.5 },
        ..Default::default()
    };
    let outcome = compute_path(&grid, req);
    assert_eq!(outcome.status, PathStatus::InvalidEndpoint);
}
