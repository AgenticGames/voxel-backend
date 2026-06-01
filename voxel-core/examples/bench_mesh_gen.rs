//! Microbench for `dual_contouring::mesh_gen::generate_mesh`.
//!
//! Builds a realistic chunk-like hermite (a sinusoidal terrain sheet through a cs=30 grid,
//! the live UE override), solves DC vertices once, then times ONLY `generate_mesh` over many
//! pre-built (hermite, dc_vertices) inputs. Used for the A/B that backs the
//! flat-vertex-map perf change (git-stash the diff to get the baseline side).
//!
//! Run: `cargo run --release -p voxel-core --example bench_mesh_gen`

use glam::Vec3;
use voxel_core::hermite::{EdgeIntersection, EdgeKey, HermiteData};
use voxel_core::material::Material;
use voxel_core::dual_contouring::solve::solve_dc_vertices;
use voxel_core::dual_contouring::mesh_gen::generate_mesh;
use std::time::Instant;

/// Sinusoidal terrain SDF: negative (solid) below the height surface.
fn terrain_sdf(x: f32, y: f32, z: f32, gs: f32) -> f32 {
    let h = gs * 0.5
        + 4.0 * (x * 0.45).sin()
        + 3.0 * (z * 0.37).cos()
        + 2.0 * ((x + z) * 0.21).sin();
    y - h
}

fn material_at(x: usize, y: usize, z: usize) -> Material {
    // A few materials interleaved so the mesh is multi-material like a real chunk.
    match (x + y * 2 + z) % 4 {
        0 => Material::Limestone,
        1 => Material::Granite,
        2 => Material::Sandstone,
        _ => Material::Basalt,
    }
}

/// Walk all three edge axes of a (gs+1)^3 sample grid, recording sign-changing edges.
fn build_hermite(gs: usize) -> HermiteData {
    let gsf = gs as f32;
    let mut hermite = HermiteData::default();
    let s = |x: usize, y: usize, z: usize| terrain_sdf(x as f32, y as f32, z as f32, gsf);

    // X edges
    for z in 0..=gs { for y in 0..=gs { for x in 0..gs {
        let (a, b) = (s(x, y, z), s(x + 1, y, z));
        if (a > 0.0) != (b > 0.0) {
            let t = a / (a - b);
            hermite.edges.insert(EdgeKey::new(x as u8, y as u8, z as u8, 0),
                EdgeIntersection { t, normal: Vec3::X * if a < b { 1.0 } else { -1.0 }, material: material_at(x, y, z) });
        }
    }}}
    // Y edges
    for z in 0..=gs { for y in 0..gs { for x in 0..=gs {
        let (a, b) = (s(x, y, z), s(x, y + 1, z));
        if (a > 0.0) != (b > 0.0) {
            let t = a / (a - b);
            hermite.edges.insert(EdgeKey::new(x as u8, y as u8, z as u8, 1),
                EdgeIntersection { t, normal: Vec3::Y * if a < b { 1.0 } else { -1.0 }, material: material_at(x, y, z) });
        }
    }}}
    // Z edges
    for z in 0..gs { for y in 0..=gs { for x in 0..=gs {
        let (a, b) = (s(x, y, z), s(x, y, z + 1));
        if (a > 0.0) != (b > 0.0) {
            let t = a / (a - b);
            hermite.edges.insert(EdgeKey::new(x as u8, y as u8, z as u8, 2),
                EdgeIntersection { t, normal: Vec3::Z * if a < b { 1.0 } else { -1.0 }, material: material_at(x, y, z) });
        }
    }}}
    hermite
}

fn main() {
    let gs: usize = 30; // live cs=30 override (grid is gs cells per axis)
    let hermite = build_hermite(gs);
    let dc = solve_dc_vertices(&hermite, gs);

    let real_verts = dc.iter().filter(|v| !v.x.is_nan()).count();
    println!("grid_size={gs}  edges={}  dc_cells={}  surface_cells={real_verts}",
        hermite.edge_count(), dc.len());

    // Sanity: report output size once.
    let sample = generate_mesh(&hermite, &dc, gs);
    println!("mesh: {} verts / {} tris", sample.vertex_count(), sample.triangle_count());

    let calls = 2000usize;
    let rounds = 6usize;
    let mut best = f64::INFINITY;
    let mut sink = 0u64;
    for r in 0..rounds {
        let t = Instant::now();
        for _ in 0..calls {
            let m = generate_mesh(&hermite, &dc, gs);
            sink = sink.wrapping_add(m.vertices.len() as u64 + m.triangles.len() as u64);
        }
        let us_per_call = t.elapsed().as_secs_f64() * 1e6 / calls as f64;
        println!("round {r}: {us_per_call:.2} us/call");
        best = best.min(us_per_call);
    }
    println!("BEST: {best:.2} us/call  (sink={sink})");
}
