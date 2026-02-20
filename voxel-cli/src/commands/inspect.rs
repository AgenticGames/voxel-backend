use std::collections::HashMap;
use std::time::Instant;

use voxel_core::chunk::ChunkCoord;
use voxel_core::dual_contouring::mesh_gen::generate_mesh;
use voxel_core::dual_contouring::solve::solve_dc_vertices;
use voxel_core::material::Material;
use voxel_gen::config::GenerationConfig;
use voxel_gen::hermite_extract::extract_hermite_data;

/// Run the inspect command.
///
/// Usage: voxel-cli inspect [--seed N] [--chunk x,y,z]
pub fn run(args: &[String]) {
    let mut seed = 42u64;
    let mut chunk_pos = (0i32, 0i32, 0i32);

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
            "--chunk" => {
                i += 1;
                if i < args.len() {
                    let parts: Vec<i32> = args[i]
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    if parts.len() == 3 {
                        chunk_pos = (parts[0], parts[1], parts[2]);
                    } else {
                        eprintln!("Invalid chunk format. Use: x,y,z");
                        std::process::exit(1);
                    }
                }
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
            }
        }
        i += 1;
    }

    let config = GenerationConfig {
        seed,
        ..Default::default()
    };

    let coord = ChunkCoord::new(chunk_pos.0, chunk_pos.1, chunk_pos.2);
    let world_origin = coord.world_origin();

    println!("Inspecting chunk ({},{},{}) with seed {}...", chunk_pos.0, chunk_pos.1, chunk_pos.2, seed);
    let start = Instant::now();

    // Generate density field (includes worm carving)
    let (density, _pools) = voxel_gen::generate_density(coord, &config);
    let density_time = start.elapsed();

    // Extract hermite data
    let hermite = extract_hermite_data(&density);
    let hermite_time = start.elapsed() - density_time;

    // Solve DC vertices and generate mesh
    let cell_size = density.size - 1;
    let dc_vertices = solve_dc_vertices(&hermite, cell_size);
    let mesh = generate_mesh(&hermite, &dc_vertices, cell_size, config.max_edge_length, 0.0);
    let mesh_time = start.elapsed() - density_time - hermite_time;

    let total_time = start.elapsed();

    // Compute density stats
    let mut air_count = 0usize;
    let mut solid_count = 0usize;
    let mut min_density = f32::MAX;
    let mut max_density = f32::MIN;
    let mut sum_density = 0.0f64;
    let mut material_counts: HashMap<Material, usize> = HashMap::new();

    for sample in &density.samples {
        if sample.density <= 0.0 {
            air_count += 1;
        } else {
            solid_count += 1;
        }
        min_density = min_density.min(sample.density);
        max_density = max_density.max(sample.density);
        sum_density += sample.density as f64;
        *material_counts.entry(sample.material).or_insert(0) += 1;
    }

    let avg_density = sum_density / density.samples.len() as f64;

    println!();
    println!("Chunk Info:");
    println!("  Coord: ({},{},{})", coord.x, coord.y, coord.z);
    println!("  World origin: ({},{},{})", world_origin.x, world_origin.y, world_origin.z);
    println!("  Grid size: {}^3 = {} samples", density.size, density.samples.len());
    println!();
    println!("Density Stats:");
    println!("  Air voxels: {} ({:.1}%)", air_count, air_count as f64 / density.samples.len() as f64 * 100.0);
    println!("  Solid voxels: {} ({:.1}%)", solid_count, solid_count as f64 / density.samples.len() as f64 * 100.0);
    println!("  Min density: {:.4}", min_density);
    println!("  Max density: {:.4}", max_density);
    println!("  Avg density: {:.4}", avg_density);
    println!();
    println!("Material Distribution:");
    let mut sorted_materials: Vec<_> = material_counts.into_iter().collect();
    sorted_materials.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    for (material, count) in &sorted_materials {
        let pct = *count as f64 / density.samples.len() as f64 * 100.0;
        println!("  {:?}: {} ({:.1}%)", material, count, pct);
    }
    println!();
    println!("Hermite Data:");
    println!("  Edge intersections: {}", hermite.edges.len());
    println!();
    println!("Mesh Stats:");
    println!("  Vertices: {}", mesh.vertex_count());
    println!("  Triangles: {}", mesh.triangle_count());
    println!();
    println!("Timing:");
    println!("  Density generation: {:.2?}", density_time);
    println!("  Hermite extraction: {:.2?}", hermite_time);
    println!("  Mesh generation: {:.2?}", mesh_time);
    println!("  Total: {:.2?}", total_time);
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_inspect_runs() {
        super::run(&["--seed".to_string(), "42".to_string()]);
    }
}
