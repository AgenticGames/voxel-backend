//! Microbench for `region_gen::generate_chunk_seam_quads` (the LIVE cross-chunk
//! seam path — worker seam pass, brush re-mesh, sleep-morph all call it; the
//! test-only `dual_contouring::seam::stitch_seam` is NOT this path).
//!
//! Builds a 2^3 block of adjacent chunks from a sinusoidal terrain sheet (cs=30,
//! the live UE override), extracts each chunk's boundary edges + DC vertices into
//! a ChunkSeamData map, then times ONLY `generate_chunk_seam_quads` for the
//! origin chunk over many calls. The origin's seam edges reference its +X/+Y/+Z
//! face/edge/corner neighbors, so every per-cell neighbor lookup hits a present
//! chunk — the realistic case the hoist targets. git-stash the diff to get the
//! baseline side of the A/B.
//!
//! Run: `cargo run --release -p voxel-gen --example bench_seam_quads`

use glam::Vec3;
use std::collections::HashMap;
use std::time::Instant;
use voxel_core::hermite::{EdgeIntersection, EdgeKey, HermiteData};
use voxel_core::material::Material;
use voxel_core::dual_contouring::solve::solve_dc_vertices;
use voxel_gen::region_gen::{
    extract_boundary_edges, generate_chunk_seam_quads, ChunkSeamData,
};

/// Sinusoidal terrain SDF in WORLD space; negative (solid) below the surface.
fn terrain_sdf(x: f32, y: f32, z: f32) -> f32 {
    let h = 15.0
        + 4.0 * (x * 0.45).sin()
        + 3.0 * (z * 0.37).cos()
        + 2.0 * ((x + z) * 0.21).sin();
    y - h
}

fn material_at(x: i32, y: i32, z: i32) -> Material {
    match (x + y * 2 + z).rem_euclid(4) {
        0 => Material::Limestone,
        1 => Material::Granite,
        2 => Material::Sandstone,
        _ => Material::Basalt,
    }
}

/// Build the seam data for one chunk at `(cx,cy,cz)` (chunk coords) given a
/// world-space terrain. Samples the (gs+1)^3 corner grid local to the chunk,
/// records sign-changing edges, solves DC vertices, and extracts boundary edges.
fn build_chunk(cx: i32, cy: i32, cz: i32, gs: usize) -> ChunkSeamData {
    let ox = cx * gs as i32;
    let oy = cy * gs as i32;
    let oz = cz * gs as i32;
    let s = |lx: usize, ly: usize, lz: usize| {
        terrain_sdf(
            (ox + lx as i32) as f32,
            (oy + ly as i32) as f32,
            (oz + lz as i32) as f32,
        )
    };

    let mut hermite = HermiteData::default();
    // X edges
    for z in 0..=gs { for y in 0..=gs { for x in 0..gs {
        let (a, b) = (s(x, y, z), s(x + 1, y, z));
        if (a > 0.0) != (b > 0.0) {
            let t = a / (a - b);
            hermite.edges.insert(EdgeKey::new(x as u8, y as u8, z as u8, 0),
                EdgeIntersection { t, normal: Vec3::X * if a < b { 1.0 } else { -1.0 }, material: material_at(ox + x as i32, oy + y as i32, oz + z as i32) });
        }
    }}}
    // Y edges
    for z in 0..=gs { for y in 0..gs { for x in 0..=gs {
        let (a, b) = (s(x, y, z), s(x, y + 1, z));
        if (a > 0.0) != (b > 0.0) {
            let t = a / (a - b);
            hermite.edges.insert(EdgeKey::new(x as u8, y as u8, z as u8, 1),
                EdgeIntersection { t, normal: Vec3::Y * if a < b { 1.0 } else { -1.0 }, material: material_at(ox + x as i32, oy + y as i32, oz + z as i32) });
        }
    }}}
    // Z edges
    for z in 0..gs { for y in 0..=gs { for x in 0..=gs {
        let (a, b) = (s(x, y, z), s(x, y, z + 1));
        if (a > 0.0) != (b > 0.0) {
            let t = a / (a - b);
            hermite.edges.insert(EdgeKey::new(x as u8, y as u8, z as u8, 2),
                EdgeIntersection { t, normal: Vec3::Z * if a < b { 1.0 } else { -1.0 }, material: material_at(ox + x as i32, oy + y as i32, oz + z as i32) });
        }
    }}}

    let dc_vertices = solve_dc_vertices(&hermite, gs);
    let boundary_edges = extract_boundary_edges(&hermite, gs);
    ChunkSeamData {
        dc_vertices,
        world_origin: Vec3::new(ox as f32, oy as f32, oz as f32),
        boundary_edges,
    }
}

fn main() {
    let gs: usize = 30; // live cs=30 override

    // 2^3 block so the origin chunk's +face/+edge/+corner neighbors are all present.
    let mut all: HashMap<(i32, i32, i32), ChunkSeamData> = HashMap::new();
    for cz in 0..2 { for cy in 0..2 { for cx in 0..2 {
        all.insert((cx, cy, cz), build_chunk(cx, cy, cz, gs));
    }}}

    let origin = (0, 0, 0);
    let edge_count = all[&origin].boundary_edges.len();

    let sample = generate_chunk_seam_quads(origin, &all, gs);
    println!(
        "grid_size={gs}  boundary_edges={edge_count}  seam mesh: {} verts / {} tris",
        sample.vertex_count(), sample.triangle_count()
    );

    let calls = 5000usize;
    let rounds = 6usize;
    let mut best = f64::INFINITY;
    let mut sink = 0u64;
    for r in 0..rounds {
        let t = Instant::now();
        for _ in 0..calls {
            let m = generate_chunk_seam_quads(origin, &all, gs);
            sink = sink.wrapping_add(m.vertices.len() as u64 + m.triangles.len() as u64);
        }
        let us_per_call = t.elapsed().as_secs_f64() * 1e6 / calls as f64;
        println!("round {r}: {us_per_call:.3} us/call");
        best = best.min(us_per_call);
    }
    println!("BEST: {best:.3} us/call  (sink={sink})");
}
