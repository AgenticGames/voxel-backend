use std::collections::HashMap;
use std::time::Instant;

use voxel_core::density::DensityField;
use voxel_core::material::Material;
use voxel_core::stress::{StressField, SupportField};
use voxel_gen::config::GenerationConfig;
use voxel_gen::region_gen;

/// Run the sleep command.
///
/// Usage: voxel-cli sleep [--seed N] [--chunks N] [--sleeps N] [--verbose]
pub fn run(args: &[String]) {
    let mut seed = 42u64;
    let mut chunks_per_axis = 3u32;
    let mut num_sleeps = 1u32;
    let mut verbose = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => { i += 1; if i < args.len() { seed = args[i].parse().unwrap_or(42); } }
            "--chunks" => { i += 1; if i < args.len() { chunks_per_axis = args[i].parse().unwrap_or(3); } }
            "--sleeps" => { i += 1; if i < args.len() { num_sleeps = args[i].parse().unwrap_or(1); } }
            "--verbose" => { verbose = true; }
            _ => { eprintln!("Unknown option: {}", args[i]); }
        }
        i += 1;
    }

    let chunks_per_axis = chunks_per_axis.min(5); // Cap at 5x5x1 for sanity

    println!("Sleep test: seed={}, {}x{}x1 region, {} sleep cycles",
        seed, chunks_per_axis, chunks_per_axis, num_sleeps);

    let config = GenerationConfig {
        seed,
        ..Default::default()
    };

    // Generate region density fields
    let start = Instant::now();
    let coords: Vec<(i32, i32, i32)> = (0..chunks_per_axis as i32)
        .flat_map(|cy| (0..chunks_per_axis as i32).map(move |cx| (cx, cy, 0)))
        .collect();

    let (mut density_fields, _pools) = region_gen::generate_region_densities(&coords, &config);
    let gen_elapsed = start.elapsed();
    println!("  Region generated in {:.2?} ({} chunks)", gen_elapsed, coords.len());

    // Count initial materials
    let initial_counts = count_materials(&density_fields, config.chunk_size);
    if verbose {
        println!("  Initial materials:");
        print_material_counts(&initial_counts);
    }

    // Initialize stress and support fields
    let grid_size = config.chunk_size + 1;
    let mut stress_fields: HashMap<(i32, i32, i32), StressField> = HashMap::new();
    let mut support_fields: HashMap<(i32, i32, i32), SupportField> = HashMap::new();
    for &key in density_fields.keys() {
        stress_fields.insert(key, StressField::new(grid_size));
        support_fields.insert(key, SupportField::new(grid_size));
    }

    // Run sleep cycles
    let sleep_config = voxel_sleep::SleepConfig::default();
    for cycle in 1..=num_sleeps {
        let cycle_start = Instant::now();

        let result = voxel_sleep::execute_sleep(
            &sleep_config,
            &mut density_fields,
            &mut stress_fields,
            &mut support_fields,
            (0, 0, 0), // player chunk
            cycle,
            None,
        );

        let cycle_elapsed = cycle_start.elapsed();
        println!("  Sleep cycle {} in {:.2?}: {} chunks changed, {} metamorphosed, {} minerals grown, {} supports degraded, {} collapses",
            cycle, cycle_elapsed,
            result.chunks_changed,
            result.voxels_metamorphosed,
            result.minerals_grown,
            result.supports_degraded,
            result.collapses_triggered,
        );

        if verbose && !result.transform_log.is_empty() {
            println!("    Transforms:");
            for entry in &result.transform_log {
                println!("      {} x{}", entry.description, entry.count);
            }
        }
    }

    // Count final materials
    let final_counts = count_materials(&density_fields, config.chunk_size);
    if verbose {
        println!("  Final materials:");
        print_material_counts(&final_counts);
    }

    // Print material diff
    println!("  Material changes:");
    let all_materials: Vec<Material> = Material::all_solid().to_vec();
    let mut any_change = false;
    for &mat in &all_materials {
        let before = *initial_counts.get(&mat).unwrap_or(&0);
        let after = *final_counts.get(&mat).unwrap_or(&0);
        let diff = after as i64 - before as i64;
        if diff != 0 {
            println!("    {}: {} -> {} ({:+})", mat.display_name(), before, after, diff);
            any_change = true;
        }
    }
    if !any_change {
        println!("    (no changes)");
    }

    println!("Done.");
}

fn count_materials(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
) -> HashMap<Material, u32> {
    let mut counts: HashMap<Material, u32> = HashMap::new();
    for df in density_fields.values() {
        for z in 0..chunk_size {
            for y in 0..chunk_size {
                for x in 0..chunk_size {
                    let mat = df.get(x, y, z).material;
                    if mat.is_solid() {
                        *counts.entry(mat).or_insert(0) += 1;
                    }
                }
            }
        }
    }
    counts
}

fn print_material_counts(counts: &HashMap<Material, u32>) {
    let mut sorted: Vec<_> = counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    for (mat, count) in sorted {
        println!("      {}: {}", mat.display_name(), count);
    }
}
