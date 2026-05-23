//! Micro-benchmark for the A* refactor. Not a real `#[bench]` (avoid nightly);
//! run via:
//!   cargo test -p voxel-path --release --test astar_bench -- --ignored --nocapture
//! to see the wall-time print.
//!
//! Builds a 60x60x60 grid with scattered pillars and runs N=200 paths of
//! varying length, summing wall-time. Marked `#[ignore]` so it doesn't run on
//! every `cargo test` invocation — opt-in only because release builds are slow
//! and the numbers are noisy on shared dev machines.

use glam::{IVec3, Vec3};
use std::collections::HashSet;
use std::time::Instant;
use voxel_path::astar::{compute_path, PathRequest, PathStatus};
use voxel_path::grid::CellGrid;
use voxel_path::movement::MovementMode;

#[derive(Default)]
struct PillaredGrid {
    solids: HashSet<(i32, i32, i32)>,
}
impl PillaredGrid {
    fn build() -> Self {
        let mut g = Self::default();
        // ~50 random pillars scattered through a 60^3 volume. Deterministic
        // PRNG (LCG) so benchmark is reproducible.
        let mut state: u32 = 0xC0FFEE;
        for _ in 0..50 {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let px = ((state >> 8) % 56) as i32 + 2;
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let py = ((state >> 8) % 56) as i32 + 2;
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let h = ((state >> 8) % 10) as i32 + 4;
            for dz in 0..h {
                g.solids.insert((px, py, dz));
            }
        }
        g
    }
}
impl CellGrid for PillaredGrid {
    fn cell_size(&self) -> f32 { 1.0 }
    fn is_solid(&self, c: IVec3) -> bool {
        self.solids.contains(&(c.x, c.y, c.z))
    }
    fn surface_normal_at(&self, _c: IVec3) -> Vec3 { Vec3::ZERO }
}

#[test]
#[ignore]
fn astar_bench_pillar_field() {
    let grid = PillaredGrid::build();
    let n_paths = 200usize;
    let mut pairs: Vec<(IVec3, IVec3)> = Vec::with_capacity(n_paths);
    let mut state: u32 = 0xDEADBEEF;
    let mut next = || {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        ((state >> 8) % 56) as i32 + 2
    };
    for _ in 0..n_paths {
        let from = IVec3::new(next(), next(), 12);
        let to   = IVec3::new(next(), next(), 12);
        if grid.is_solid(from) || grid.is_solid(to) { continue; }
        pairs.push((from, to));
    }

    let start = Instant::now();
    let mut succ = 0usize;
    let mut nopath = 0usize;
    for (from, to) in &pairs {
        let outcome = compute_path(&grid, PathRequest {
            from: *from,
            to: *to,
            mode: MovementMode::Flying { agent_radius_cells: 0.5 },
            smooth: false,
            max_nodes: 20_000,
            ..Default::default()
        });
        match outcome.status {
            PathStatus::Success => succ += 1,
            PathStatus::NoPath | PathStatus::MaxNodesReached => nopath += 1,
            _ => {}
        }
    }
    let elapsed = start.elapsed();
    eprintln!(
        "astar_bench_pillar_field: {} paths in {:?} ({} ok, {} no-path/max), avg {:?}/path",
        pairs.len(), elapsed, succ, nopath,
        elapsed / pairs.len() as u32,
    );
}
