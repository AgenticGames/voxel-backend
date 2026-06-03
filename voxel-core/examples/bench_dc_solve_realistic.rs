//! Microbench for `dual_contouring::solve::solve_dc_vertices` with REALISTIC
//! (non-axis-aligned) hermite normals.
//!
//! `bench_dc_solve` builds its hermite with pure axis-aligned normals
//! (`Vec3::X/Y/Z`). That makes every cell's `ATA` matrix diagonal, so the
//! Jacobi eigensolver inside `QefData::solve` exits on its first convergence
//! check — i.e. it does ZERO rotations and the SVD is effectively free. Real
//! worldgen feeds `solve_dc_vertices` normals taken from the *gradient* of a
//! smooth density field (see `voxel-gen` hermite extraction), which point in
//! arbitrary directions and produce dense `ATA` matrices that take several
//! Jacobi sweeps to diagonalise. This bench reproduces that by deriving each
//! edge normal from the analytic gradient of a smooth terrain SDF, so the
//! eigensolver cost — the real hot spot — actually shows up.
//!
//! Run: `cargo run --release -p voxel-core --example bench_dc_solve_realistic`

use glam::Vec3;
use voxel_core::hermite::{EdgeIntersection, EdgeKey, HermiteData};
use voxel_core::material::Material;
use voxel_core::dual_contouring::solve::solve_dc_vertices;
use std::time::Instant;

/// Smooth terrain SDF (same shape family as `bench_dc_solve`).
fn terrain_sdf(x: f32, y: f32, z: f32, gs: f32) -> f32 {
    let h = gs * 0.5
        + 4.0 * (x * 0.45).sin()
        + 3.0 * (z * 0.37).cos()
        + 2.0 * ((x + z) * 0.21).sin();
    y - h
}

/// Analytic-ish gradient of the SDF via central differences, normalised.
/// This is what real hermite extraction stores as the edge normal — a unit
/// vector pointing along the surface gradient, almost never axis-aligned.
fn terrain_normal(x: f32, y: f32, z: f32, gs: f32) -> Vec3 {
    let h = 0.5_f32;
    let dx = terrain_sdf(x + h, y, z, gs) - terrain_sdf(x - h, y, z, gs);
    let dy = terrain_sdf(x, y + h, z, gs) - terrain_sdf(x, y - h, z, gs);
    let dz = terrain_sdf(x, y, z + h, gs) - terrain_sdf(x, y, z - h, gs);
    Vec3::new(dx, dy, dz).normalize_or_zero()
}

fn material_at(x: usize, y: usize, z: usize) -> Material {
    match (x + y * 2 + z) % 4 {
        0 => Material::Limestone,
        1 => Material::Granite,
        2 => Material::Sandstone,
        _ => Material::Basalt,
    }
}

fn build_hermite(gs: usize) -> HermiteData {
    let gsf = gs as f32;
    let mut hermite = HermiteData::default();
    let s = |x: usize, y: usize, z: usize| terrain_sdf(x as f32, y as f32, z as f32, gsf);

    for z in 0..=gs { for y in 0..=gs { for x in 0..gs {
        let (a, b) = (s(x, y, z), s(x + 1, y, z));
        if (a > 0.0) != (b > 0.0) {
            let t = a / (a - b);
            // Normal sampled at the actual crossing point — gradient direction.
            let n = terrain_normal(x as f32 + t, y as f32, z as f32, gsf);
            hermite.edges.insert(EdgeKey::new(x as u8, y as u8, z as u8, 0),
                EdgeIntersection { t, normal: n, material: material_at(x, y, z) });
        }
    }}}
    for z in 0..=gs { for y in 0..gs { for x in 0..=gs {
        let (a, b) = (s(x, y, z), s(x, y + 1, z));
        if (a > 0.0) != (b > 0.0) {
            let t = a / (a - b);
            let n = terrain_normal(x as f32, y as f32 + t, z as f32, gsf);
            hermite.edges.insert(EdgeKey::new(x as u8, y as u8, z as u8, 1),
                EdgeIntersection { t, normal: n, material: material_at(x, y, z) });
        }
    }}}
    for z in 0..gs { for y in 0..=gs { for x in 0..=gs {
        let (a, b) = (s(x, y, z), s(x, y, z + 1));
        if (a > 0.0) != (b > 0.0) {
            let t = a / (a - b);
            let n = terrain_normal(x as f32, y as f32, z as f32 + t, gsf);
            hermite.edges.insert(EdgeKey::new(x as u8, y as u8, z as u8, 2),
                EdgeIntersection { t, normal: n, material: material_at(x, y, z) });
        }
    }}}
    hermite
}

fn main() {
    let gs: usize = 30; // live cs=30 override (grid is gs cells per axis)
    let hermite = build_hermite(gs);

    let dc = solve_dc_vertices(&hermite, gs);
    let real_verts = dc.iter().filter(|v| !v.x.is_nan()).count();
    println!("grid_size={gs}  edges={}  dc_cells={}  surface_cells={real_verts}  (gradient normals)",
        hermite.edge_count(), dc.len());

    let calls = 2000usize;
    let rounds = 6usize;
    let mut best = f64::INFINITY;
    let mut sink = 0u64;
    for r in 0..rounds {
        let t = Instant::now();
        for _ in 0..calls {
            let v = solve_dc_vertices(&hermite, gs);
            sink = sink.wrapping_add(v.iter().filter(|p| !p.x.is_nan()).count() as u64);
        }
        let us_per_call = t.elapsed().as_secs_f64() * 1e6 / calls as f64;
        println!("round {r}: {us_per_call:.2} us/call");
        best = best.min(us_per_call);
    }
    println!("BEST: {best:.2} us/call  (sink={sink})");
}
