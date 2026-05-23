//! Integration tests for A* correctness around the g-relaxation path.
//!
//! The 2026-05-24 perf pass refactored `compute_path` to:
//!  - Carry `g_score` on each open-heap entry (lazy closed-set via stale pop).
//!  - Skip `can_traverse` for already-known neighbors (their entry in
//!    `g_score` proves traversability).
//!  - Use a single `g_score.get(&neighbor).copied()` probe for both the
//!    "seen before" check and the relaxation comparison.
//!
//! These tests construct grids where the FIRST g-score recorded for a cell
//! is suboptimal — the optimal path is discovered later through a different
//! ancestor. They verify that:
//!   * the final path is optimal (g-relaxation actually runs),
//!   * the stale heap entry is correctly skipped on pop,
//!   * paths around obstacles still find the shortest route.

use glam::{IVec3, Vec3};
use std::collections::HashSet;
use voxel_path::astar::{compute_path, PathRequest, PathStatus};
use voxel_path::grid::CellGrid;
use voxel_path::movement::MovementMode;

/// Stub air grid used everywhere — explicit solid set, everything else air.
#[derive(Default)]
struct StubGrid {
    solids: HashSet<(i32, i32, i32)>,
}
impl StubGrid {
    fn set(&mut self, c: IVec3) { self.solids.insert((c.x, c.y, c.z)); }
    fn fill_box(&mut self, lo: IVec3, hi: IVec3) {
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    self.set(IVec3::new(x, y, z));
                }
            }
        }
    }
}
impl CellGrid for StubGrid {
    fn cell_size(&self) -> f32 { 1.0 }
    fn is_solid(&self, c: IVec3) -> bool {
        self.solids.contains(&(c.x, c.y, c.z))
    }
    fn surface_normal_at(&self, _c: IVec3) -> Vec3 { Vec3::ZERO }
}

/// Compute the euclidean length of a sequence of cells. Smoothing-off so the
/// raw A* trail's cumulative length is what we measure.
fn path_length(nodes: &[voxel_path::astar::PathNode]) -> f32 {
    let mut sum = 0.0f32;
    for w in nodes.windows(2) {
        let a = w[0].cell;
        let b = w[1].cell;
        let dx = (a.x - b.x) as f32;
        let dy = (a.y - b.y) as f32;
        let dz = (a.z - b.z) as f32;
        sum += (dx * dx + dy * dy + dz * dz).sqrt();
    }
    sum
}

#[test]
fn diagonal_beats_manhattan_route_to_corner() {
    // Open 3D space. Going from (0,0,0) to (3,3,0) — the optimal A* path uses
    // 3 diagonal (1,1,0)-type steps (length 3*√2 ≈ 4.243), NOT 6 face-axis
    // steps (length 6.0). This exercises g-relaxation: face neighbors get
    // pushed first with smaller f-bias, but the diagonal entries supersede
    // them on relaxation.
    let grid = StubGrid::default();
    let outcome = compute_path(&grid, PathRequest {
        from: IVec3::new(0, 0, 0),
        to: IVec3::new(3, 3, 0),
        mode: MovementMode::Flying { agent_radius_cells: 0.5 },
        smooth: false,
        ..Default::default()
    });
    assert_eq!(outcome.status, PathStatus::Success);
    let len = path_length(&outcome.nodes);
    let optimal = 3.0_f32 * 2.0_f32.sqrt();
    assert!(
        (len - optimal).abs() < 0.01,
        "expected optimal diagonal path of length {} but got {} (nodes: {:?})",
        optimal, len, outcome.nodes.iter().map(|n| n.cell).collect::<Vec<_>>()
    );
}

#[test]
fn optimal_path_around_block_via_relaxation() {
    // A 1-cell wall at x=1, y in [-1,1], z=0 forces a detour. The optimal path
    // is over the corner via diagonal moves, not straight around the long way.
    // Multiple ancestors discover the cells past the wall — g-relaxation must
    // pick the shorter.
    let mut grid = StubGrid::default();
    grid.fill_box(IVec3::new(1, -1, 0), IVec3::new(1, 1, 0));
    let outcome = compute_path(&grid, PathRequest {
        from: IVec3::new(0, 0, 0),
        to: IVec3::new(3, 0, 0),
        mode: MovementMode::Flying { agent_radius_cells: 0.5 },
        smooth: false,
        ..Default::default()
    });
    assert_eq!(outcome.status, PathStatus::Success);
    // Must not pass through any of the wall cells.
    for n in &outcome.nodes {
        assert!(
            !grid.is_solid(n.cell),
            "path goes through wall cell {:?}",
            n.cell,
        );
    }
    // Optimal path: out-of-plane diagonal (e.g. via z=±1 or y=±2). The
    // straight Manhattan detour would be (0,0,0)→(0,2,0)→(2,2,0)→(2,0,0)→(3,0,0)
    // = 7 face steps. Diagonal route is ~4.5 units. Assert ≤ 5.0 to allow
    // either of several equivalent diagonal-corner routes.
    let len = path_length(&outcome.nodes);
    assert!(
        len < 5.0,
        "expected diagonal detour (len < 5.0), got {} (nodes: {:?})",
        len, outcome.nodes.iter().map(|n| n.cell).collect::<Vec<_>>()
    );
}

#[test]
fn stale_pop_does_not_re_expand_better_predecessor() {
    // A worst-case-ish layout for the lazy-closed-set: a "broad waist" of open
    // space where many cells get pushed multiple times with improving
    // g-scores before any one gets its final expansion. The output must still
    // be optimal AND no cell should appear twice in the path.
    let grid = StubGrid::default();
    let outcome = compute_path(&grid, PathRequest {
        from: IVec3::new(0, 0, 0),
        to: IVec3::new(5, 5, 5),
        mode: MovementMode::Flying { agent_radius_cells: 0.5 },
        smooth: false,
        ..Default::default()
    });
    assert_eq!(outcome.status, PathStatus::Success);
    let cells: Vec<_> = outcome.nodes.iter().map(|n| n.cell).collect();
    let unique: HashSet<_> = cells.iter().copied().collect();
    assert_eq!(unique.len(), cells.len(), "path visits some cell twice: {:?}", cells);
    // Optimal length: 5 corner-diagonals of length √3 = 5√3 ≈ 8.66.
    let len = path_length(&outcome.nodes);
    let optimal = 5.0_f32 * 3.0_f32.sqrt();
    assert!(
        (len - optimal).abs() < 0.01,
        "expected optimal corner-diagonal length {} but got {}",
        optimal, len
    );
}
