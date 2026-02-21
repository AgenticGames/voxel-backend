use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::time::Instant;

use glam::Vec3;
use voxel_core::chunk::ChunkCoord;
use voxel_core::dual_contouring::mesh_gen::generate_mesh;
use voxel_core::dual_contouring::solve::solve_dc_vertices;
use voxel_core::export::write_obj;
use voxel_core::hermite::{EdgeIntersection, EdgeKey};
use voxel_core::mesh::{Mesh, Vertex, Triangle};
use voxel_gen::config::GenerationConfig;
use voxel_gen::density::DensityField;
use voxel_gen::hermite_extract::extract_hermite_data;
use voxel_gen::worm::carve::carve_worm_into_density;
use voxel_gen::worm::connect::{find_cavern_centers, plan_worm_connections};
use voxel_gen::worm::path::generate_worm_path;

use crate::validation::validate_mesh;

/// Per-chunk data kept for seam stitching after per-chunk meshing.
struct ChunkSeamData {
    dc_vertices: Vec<Vec3>,
    world_origin: Vec3,
    /// Boundary edges: edges on this chunk's positive faces whose quads span
    /// into a neighbor chunk. Only these need cross-chunk stitching.
    boundary_edges: Vec<(EdgeKey, EdgeIntersection)>,
}

/// Run the generate command.
///
/// Usage: voxel-cli generate [--seed N] [--output path.obj] [--chunk-range x0,y0,z0,x1,y1,z1]
pub fn run(args: &[String]) {
    let mut seed = 42u64;
    let mut output = "output.obj".to_string();
    let mut range_min = (0i32, 0i32, 0i32);
    let mut range_max = (1i32, 1i32, 1i32);
    let mut closed = false;

    // Optional config overrides (None = use default)
    let mut cavern_freq: Option<f64> = None;
    let mut cavern_threshold: Option<f64> = None;
    let mut detail_octaves: Option<u32> = None;
    let mut detail_persistence: Option<f64> = None;
    let mut warp_amplitude: Option<f64> = None;
    let mut worms_per_region: Option<f32> = None;
    let mut worm_radius_min: Option<f32> = None;
    let mut worm_radius_max: Option<f32> = None;
    let mut worm_step_length: Option<f32> = None;
    let mut worm_max_steps: Option<u32> = None;
    let mut worm_falloff_power: Option<f32> = None;

    // Parse arguments
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                if i < args.len() {
                    seed = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("Invalid seed value: {}", args[i]);
                        std::process::exit(1);
                    });
                }
            }
            "--output" => {
                i += 1;
                if i < args.len() {
                    output = args[i].clone();
                }
            }
            "--chunk-range" => {
                i += 1;
                if i < args.len() {
                    let parts: Vec<i32> = args[i]
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    if parts.len() == 6 {
                        range_min = (parts[0], parts[1], parts[2]);
                        range_max = (parts[3], parts[4], parts[5]);
                    } else if parts.len() == 3 {
                        // Single chunk at specified position
                        range_min = (parts[0], parts[1], parts[2]);
                        range_max = (parts[0] + 1, parts[1] + 1, parts[2] + 1);
                    } else {
                        eprintln!("Invalid chunk-range format. Use: x0,y0,z0,x1,y1,z1 or x,y,z");
                        std::process::exit(1);
                    }
                }
            }
            "--closed" => { closed = true; }
            "--cavern-freq" => { i += 1; if i < args.len() { cavern_freq = args[i].parse().ok(); } }
            "--cavern-threshold" => { i += 1; if i < args.len() { cavern_threshold = args[i].parse().ok(); } }
            "--detail-octaves" => { i += 1; if i < args.len() { detail_octaves = args[i].parse().ok(); } }
            "--detail-persistence" => { i += 1; if i < args.len() { detail_persistence = args[i].parse().ok(); } }
            "--warp-amplitude" => { i += 1; if i < args.len() { warp_amplitude = args[i].parse().ok(); } }
            "--worms-per-region" => { i += 1; if i < args.len() { worms_per_region = args[i].parse().ok(); } }
            "--worm-radius-min" => { i += 1; if i < args.len() { worm_radius_min = args[i].parse().ok(); } }
            "--worm-radius-max" => { i += 1; if i < args.len() { worm_radius_max = args[i].parse().ok(); } }
            "--worm-step-length" => { i += 1; if i < args.len() { worm_step_length = args[i].parse().ok(); } }
            "--worm-max-steps" => { i += 1; if i < args.len() { worm_max_steps = args[i].parse().ok(); } }
            "--worm-falloff-power" => { i += 1; if i < args.len() { worm_falloff_power = args[i].parse().ok(); } }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
            }
        }
        i += 1;
    }

    let mut config = GenerationConfig {
        seed,
        ..Default::default()
    };
    if let Some(v) = cavern_freq { config.noise.cavern_frequency = v; }
    if let Some(v) = cavern_threshold { config.noise.cavern_threshold = v; }
    if let Some(v) = detail_octaves { config.noise.detail_octaves = v; }
    if let Some(v) = detail_persistence { config.noise.detail_persistence = v; }
    if let Some(v) = warp_amplitude { config.noise.warp_amplitude = v; }
    if let Some(v) = worms_per_region { config.worm.worms_per_region = v; }
    if let Some(v) = worm_radius_min { config.worm.radius_min = v; }
    if let Some(v) = worm_radius_max { config.worm.radius_max = v; }
    if let Some(v) = worm_step_length { config.worm.step_length = v; }
    if let Some(v) = worm_max_steps { config.worm.max_steps = v; }
    if let Some(v) = worm_falloff_power { config.worm.falloff_power = v; }

    let num_chunks_x = range_max.0 - range_min.0;
    let num_chunks_y = range_max.1 - range_min.1;
    let num_chunks_z = range_max.2 - range_min.2;
    let total_chunks = num_chunks_x * num_chunks_y * num_chunks_z;
    let gs = config.chunk_size; // grid_size = number of cells per axis = 16

    println!("Generating {}x{}x{} chunks ({} total) with seed {}...",
        num_chunks_x, num_chunks_y, num_chunks_z, total_chunks, seed);
    let start = Instant::now();

    // ── Phase 1: Generate base density fields (noise only, no worms) ──
    let mut density_fields: HashMap<(i32, i32, i32), DensityField> = HashMap::new();
    for cz in range_min.2..range_max.2 {
        for cy in range_min.1..range_max.1 {
            for cx in range_min.0..range_max.0 {
                let coord = ChunkCoord::new(cx, cy, cz);
                let density = voxel_gen::density::generate_density_field(&config, coord.world_origin_sized(gs));
                density_fields.insert((cx, cy, cz), density);
            }
        }
    }

    // ── Phase 2: Collect cavern centers from ALL chunks ──
    let mut all_cavern_centers: Vec<Vec3> = Vec::new();
    for cz in range_min.2..range_max.2 {
        for cy in range_min.1..range_max.1 {
            for cx in range_min.0..range_max.0 {
                let density = &density_fields[&(cx, cy, cz)];
                let coord = ChunkCoord::new(cx, cy, cz);
                let flat = density.densities();
                let centers = find_cavern_centers(&flat, density.size, coord.world_origin_sized(gs));
                all_cavern_centers.extend(centers);
            }
        }
    }

    // ── Phase 3: Plan global worm connections ──
    let global_worm_seed = config.seed.wrapping_add(0xF1F0_CAFE);
    let total_worms = (config.worm.worms_per_region * total_chunks as f32).ceil() as u32;
    let connections = plan_worm_connections(global_worm_seed, &all_cavern_centers, total_worms);

    // ── Phase 4: Generate worm paths and carve into all overlapping chunks ──
    let chunk_size_f = config.chunk_size as f32;
    for (i, (worm_start, _end)) in connections.iter().enumerate() {
        let worm_seed = config.seed.wrapping_add(0x1000).wrapping_add(i as u64 * 1000);
        let segments = generate_worm_path(
            worm_seed,
            *worm_start,
            config.worm.step_length,
            config.worm.max_steps,
            config.worm.radius_min,
            config.worm.radius_max,
        );
        if segments.is_empty() {
            continue;
        }

        // Bounding box of entire worm path to find affected chunks
        let max_r = segments.iter().map(|s| s.radius).fold(0.0f32, f32::max);
        let mut path_min = segments[0].position;
        let mut path_max = segments[0].position;
        for seg in &segments {
            path_min = path_min.min(seg.position);
            path_max = path_max.max(seg.position);
        }
        path_min -= Vec3::splat(max_r);
        path_max += Vec3::splat(max_r);

        let min_cx = (path_min.x / chunk_size_f).floor() as i32;
        let max_cx = (path_max.x / chunk_size_f).floor() as i32;
        let min_cy = (path_min.y / chunk_size_f).floor() as i32;
        let max_cy = (path_max.y / chunk_size_f).floor() as i32;
        let min_cz = (path_min.z / chunk_size_f).floor() as i32;
        let max_cz = (path_max.z / chunk_size_f).floor() as i32;

        for cz in min_cz..=max_cz {
            for cy in min_cy..=max_cy {
                for cx in min_cx..=max_cx {
                    if let Some(density) = density_fields.get_mut(&(cx, cy, cz)) {
                        let coord = ChunkCoord::new(cx, cy, cz);
                        carve_worm_into_density(
                            density,
                            &segments,
                            coord.world_origin_sized(gs),
                            config.worm.falloff_power,
                        );
                    }
                }
            }
        }
    }

    // ── Phase 5: Seal boundary faces AFTER worms (so clamping re-seals carved edges) ──
    if closed {
        for (&(cx, cy, cz), density) in &mut density_fields {
            density.clamp_boundary_faces(
                cx == range_min.0, cx == range_max.0 - 1,
                cy == range_min.1, cy == range_max.1 - 1,
                cz == range_min.2, cz == range_max.2 - 1,
            );
        }
    }

    // ── Phase 6: Extract hermite, mesh, seam stitch ──
    let mut meshes = Vec::new();
    let mut chunk_seam: HashMap<(i32, i32, i32), ChunkSeamData> = HashMap::new();
    let mut total_vertices = 0usize;
    let mut total_triangles = 0usize;
    let mut chunks_valid = 0u32;
    let mut chunks_invalid = 0u32;

    for cz in range_min.2..range_max.2 {
        for cy in range_min.1..range_max.1 {
            for cx in range_min.0..range_max.0 {
                let density = density_fields.remove(&(cx, cy, cz)).unwrap();
                let coord = ChunkCoord::new(cx, cy, cz);

                let hermite = extract_hermite_data(&density);
                let cell_size = density.size - 1;
                let dc_vertices = solve_dc_vertices(&hermite, cell_size);

                // Generate mesh via dual contouring
                let mut mesh = generate_mesh(&hermite, &dc_vertices, cell_size, config.max_edge_length, 0.0);

                // Validate before transforming (validation thresholds are in grid space)
                let valid = validate_mesh(&mesh);
                if valid {
                    chunks_valid += 1;
                } else {
                    chunks_invalid += 1;
                    eprintln!("  WARNING: chunk ({},{},{}) failed validation", cx, cy, cz);
                }

                // Extract boundary edges for seam stitching before dropping hermite data
                let boundary_edges = extract_boundary_edges(&hermite, gs);

                // Transform mesh vertices from local grid space to world space
                let world_origin = coord.world_origin_sized(gs);
                for v in &mut mesh.vertices {
                    v.position += world_origin;
                }

                total_vertices += mesh.vertex_count();
                total_triangles += mesh.triangle_count();
                meshes.push(mesh);

                chunk_seam.insert((cx, cy, cz), ChunkSeamData {
                    dc_vertices,
                    world_origin,
                    boundary_edges,
                });
            }
        }
    }

    // Generate seam mesh to fill gaps between adjacent chunks
    let seam_mesh = generate_seam_mesh(&chunk_seam, gs);
    let seam_verts = seam_mesh.vertex_count();
    let seam_tris = seam_mesh.triangle_count();
    total_vertices += seam_verts;
    total_triangles += seam_tris;

    // Merge all per-chunk meshes + seam mesh into one
    let mut combined = Mesh::new();
    for m in &meshes {
        combined.merge(m);
    }
    combined.merge(&seam_mesh);

    let elapsed = start.elapsed();

    // Write OBJ file
    match File::create(&output) {
        Ok(file) => {
            let mut writer = BufWriter::new(file);
            if let Err(e) = write_obj(&combined, &mut writer) {
                eprintln!("Failed to write OBJ: {}", e);
                std::process::exit(1);
            }
            println!("Wrote {}", output);
        }
        Err(e) => {
            eprintln!("Failed to create output file: {}", e);
            std::process::exit(1);
        }
    }

    println!("Stats:");
    println!("  Chunks: {} ({} valid, {} invalid)", total_chunks, chunks_valid, chunks_invalid);
    println!("  Vertices: {} (seam: {})", total_vertices, seam_verts);
    println!("  Triangles: {} (seam: {})", total_triangles, seam_tris);
    println!("  Time: {:.2?}", elapsed);

    if chunks_invalid > 0 {
        println!("  QUALITY: {}/{} chunks passed validation", chunks_valid, total_chunks);
    } else {
        println!("  QUALITY: All chunks passed validation");
    }
}

// ── Seam stitching ─────────────────────────────────────────────────────────

/// Extract boundary edges from hermite data that need cross-chunk seam quads.
///
/// An edge is on the positive boundary if its quad would reference cells at
/// coord >= gs (the grid_size). These quads are skipped by per-chunk mesh_gen
/// but can be emitted by looking up DC vertices from the neighbor chunk.
///
/// We only collect positive-boundary edges. Negative-boundary edges (at coord 0)
/// are the same physical edges as the neighbor chunk's positive-boundary edges,
/// so they get processed from that side, preventing duplicates.
fn extract_boundary_edges(
    hermite: &voxel_core::hermite::HermiteData,
    gs: usize,
) -> Vec<(EdgeKey, EdgeIntersection)> {
    let mut edges = Vec::new();
    for (key, intersection) in hermite.edges.iter() {
        let x = key.x() as usize;
        let y = key.y() as usize;
        let z = key.z() as usize;
        let axis = key.axis();

        // Boundary edge: its quad cells include at least one cell at coord >= gs.
        // For each axis, determine which coordinates can go out of range:
        //   X-edges: cells use (x, y-1..y, z-1..z) → boundary if y==gs or z==gs
        //   Y-edges: cells use (x-1..x, y, z-1..z) → boundary if x==gs or z==gs
        //   Z-edges: cells use (x-1..x, y-1..y, z) → boundary if x==gs or y==gs
        let is_boundary = match axis {
            0 => y == gs || z == gs,
            1 => x == gs || z == gs,
            2 => x == gs || y == gs,
            _ => false,
        };

        if is_boundary {
            edges.push((key, intersection.clone()));
        }
    }
    edges
}

/// Generate seam mesh by emitting quads for boundary edges using DC vertices
/// from adjacent chunks.
///
/// For each boundary edge on a chunk's positive face:
/// 1. Compute the 4 cell positions that share the edge
/// 2. Skip if any cell is on the negative side (handled by another chunk)
/// 3. For cells at coord >= gs, look up DC vertex from the neighbor chunk
/// 4. Emit a quad connecting the 4 DC vertices
fn generate_seam_mesh(
    chunks: &HashMap<(i32, i32, i32), ChunkSeamData>,
    gs: usize,
) -> Mesh {
    let mut mesh = Mesh::new();
    let gs_i = gs as i32;

    for (&(cx, cy, cz), chunk) in chunks {
        for (edge_key, intersection) in &chunk.boundary_edges {
            let ex = edge_key.x() as i32;
            let ey = edge_key.y() as i32;
            let ez = edge_key.z() as i32;
            let axis = edge_key.axis() as usize;

            // Compute the 4 cell positions for this edge's quad
            let cells = match axis {
                0 => [(ex, ey - 1, ez - 1), (ex, ey, ez - 1), (ex, ey, ez), (ex, ey - 1, ez)],
                1 => [(ex - 1, ey, ez - 1), (ex, ey, ez - 1), (ex, ey, ez), (ex - 1, ey, ez)],
                2 => [(ex - 1, ey - 1, ez), (ex, ey - 1, ez), (ex, ey, ez), (ex - 1, ey, ez)],
                _ => continue,
            };

            // Skip if any cell has negative coordinates — that edge's seam quad
            // is handled by the neighboring chunk's positive boundary instead.
            if cells.iter().any(|&(x, y, z)| x < 0 || y < 0 || z < 0) {
                continue;
            }

            // Look up DC vertex for each cell from the appropriate chunk
            let mut positions = [Vec3::ZERO; 4];
            let mut valid = true;

            for (i, &(cell_x, cell_y, cell_z)) in cells.iter().enumerate() {
                // Determine which chunk this cell belongs to
                let chunk_dx = if cell_x >= gs_i { 1 } else { 0 };
                let chunk_dy = if cell_y >= gs_i { 1 } else { 0 };
                let chunk_dz = if cell_z >= gs_i { 1 } else { 0 };

                let neighbor_key = (cx + chunk_dx, cy + chunk_dy, cz + chunk_dz);

                // Local cell coords within the neighbor chunk
                let lx = (cell_x - chunk_dx * gs_i) as usize;
                let ly = (cell_y - chunk_dy * gs_i) as usize;
                let lz = (cell_z - chunk_dz * gs_i) as usize;

                if let Some(neighbor) = chunks.get(&neighbor_key) {
                    let cell_idx = lz * gs * gs + ly * gs + lx;
                    if cell_idx >= neighbor.dc_vertices.len() {
                        valid = false;
                        break;
                    }
                    let pos = neighbor.dc_vertices[cell_idx];
                    if pos.x.is_nan() {
                        valid = false;
                        break;
                    }
                    // Transform from local grid space to world space
                    positions[i] = pos + neighbor.world_origin;
                } else {
                    // Neighbor chunk doesn't exist (edge of generated region)
                    valid = false;
                    break;
                }
            }

            if !valid {
                continue;
            }

            // Emit the quad as 2 triangles
            let base = mesh.vertices.len() as u32;
            for pos in &positions {
                mesh.vertices.push(Vertex {
                    position: *pos,
                    normal: intersection.normal,
                    material: intersection.material,
                });
            }

            // Winding order based on surface normal direction
            let axis_dir = match axis {
                0 => Vec3::X,
                1 => Vec3::Y,
                _ => Vec3::Z,
            };
            let normal_dot = intersection.normal.dot(axis_dir);

            let (tri_a, tri_b) = if normal_dot > 0.0 {
                ([base, base + 1, base + 2], [base, base + 2, base + 3])
            } else {
                ([base + 2, base + 1, base], [base + 3, base + 2, base])
            };

            // Skip degenerate triangles (near-zero area)
            if !is_degenerate(&mesh.vertices, tri_a) {
                mesh.triangles.push(Triangle { indices: tri_a });
            }
            if !is_degenerate(&mesh.vertices, tri_b) {
                mesh.triangles.push(Triangle { indices: tri_b });
            }
        }
    }

    mesh
}

fn is_degenerate(vertices: &[Vertex], indices: [u32; 3]) -> bool {
    let v0 = vertices[indices[0] as usize].position;
    let v1 = vertices[indices[1] as usize].position;
    let v2 = vertices[indices[2] as usize].position;
    (v1 - v0).cross(v2 - v0).length_squared() < 1e-10
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_generate_runs() {
        let output = std::env::temp_dir()
            .join("voxel_test_generate.obj")
            .to_string_lossy()
            .to_string();
        super::run(&[
            "--seed".to_string(),
            "42".to_string(),
            "--output".to_string(),
            output,
        ]);
    }

    #[test]
    fn test_generate_multi_chunk() {
        let output = std::env::temp_dir()
            .join("voxel_test_multi.obj")
            .to_string_lossy()
            .to_string();
        super::run(&[
            "--seed".to_string(),
            "1".to_string(),
            "--chunk-range".to_string(),
            "0,0,0,2,2,1".to_string(),
            "--output".to_string(),
            output,
        ]);
    }

    #[test]
    fn test_seam_mesh_produced() {
        // 2x2x1 should produce seam geometry between the 4 chunks
        let output = std::env::temp_dir()
            .join("voxel_test_seam.obj")
            .to_string_lossy()
            .to_string();
        super::run(&[
            "--seed".to_string(),
            "1".to_string(),
            "--chunk-range".to_string(),
            "0,0,0,2,2,1".to_string(),
            "--output".to_string(),
            output,
        ]);
        // Test passes if it completes without panic.
        // The seam stats are printed to stdout.
    }
}
