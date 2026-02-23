use std::fs::{self, File};
use std::io::BufWriter;
use std::time::Instant;

use rayon::prelude::*;
use voxel_core::chunk::ChunkCoord;
use voxel_core::dual_contouring::mesh_gen::generate_mesh;
use voxel_core::dual_contouring::solve::solve_dc_vertices;
use voxel_core::export::write_obj;
use voxel_gen::config::GenerationConfig;
use voxel_gen::hermite_extract::extract_hermite_data;

use crate::report::{FailureDetail, TestReport};
use crate::validation::{validate_mesh, check_navigability, check_watertight};

/// Run the batch-test command.
///
/// Usage: voxel-cli batch-test [--seeds N] [--output-dir path] [--threads N]
pub fn run(args: &[String]) {
    let mut num_seeds = 10u64;
    let mut output_dir = "batch_output".to_string();
    let mut threads = 0usize; // 0 = use rayon default

    // Parse arguments
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--seeds" => {
                i += 1;
                if i < args.len() {
                    num_seeds = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("Invalid seeds value: {}", args[i]);
                        std::process::exit(1);
                    });
                }
            }
            "--output-dir" => {
                i += 1;
                if i < args.len() {
                    output_dir = args[i].clone();
                }
            }
            "--threads" => {
                i += 1;
                if i < args.len() {
                    threads = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("Invalid threads value: {}", args[i]);
                        std::process::exit(1);
                    });
                }
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
            }
        }
        i += 1;
    }

    // Configure rayon thread pool if specified
    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .ok(); // Ignore error if already initialized
    }

    // Create output directory
    fs::create_dir_all(&output_dir).unwrap_or_else(|e| {
        eprintln!("Failed to create output directory: {}", e);
        std::process::exit(1);
    });

    println!("Running batch test with {} seeds...", num_seeds);
    let start = Instant::now();

    let seeds: Vec<u64> = (0..num_seeds).collect();

    // Test each seed in parallel
    let results: Vec<FailureDetail> = seeds
        .par_iter()
        .map(|&seed| {
            let config = GenerationConfig {
                seed,
                ..Default::default()
            };

            let coord = ChunkCoord::new(0, 0, 0);

            // Generate density field (includes worm carving)
            let (density, _pools) = voxel_gen::generate_density(coord, &config);

            // Check navigability
            let densities = density.densities();
            let navigable = check_navigability(&densities, density.size);

            // Extract hermite, solve DC vertices, and generate mesh
            let hermite = extract_hermite_data(&density);
            let cell_size = density.size - 1;
            let dc_vertices = solve_dc_vertices(&hermite, cell_size);
            let mesh = generate_mesh(&hermite, &dc_vertices, cell_size);

            // Validate mesh
            let mesh_valid = validate_mesh(&mesh);
            let watertight = check_watertight(&mesh);

            // Pass/fail based on mesh validity and navigability.
            // Watertight is reported as info only — single-chunk meshes have
            // open boundary edges by design; watertight requires seam stitching.
            let passed = mesh_valid && navigable;

            let mut reasons = Vec::new();
            if !mesh_valid {
                reasons.push("mesh validation failed");
            }
            if !navigable {
                reasons.push("not navigable");
            }
            if !watertight {
                reasons.push("not watertight");
            }

            // Export OBJ for ALL seeds so the viewer can display any of them
            let obj_path = format!("{}/seed_{}.obj", output_dir, seed);
            if let Ok(file) = File::create(&obj_path) {
                let mut writer = BufWriter::new(file);
                let _ = write_obj(&mesh, &mut writer);
            }

            FailureDetail {
                seed,
                passed,
                reason: if passed { String::new() } else { reasons.join(", ") },
                obj_path: Some(obj_path),
            }
        })
        .collect();

    let elapsed = start.elapsed();

    // Build report
    let failed = results.iter().filter(|r| !r.passed).count() as u32;
    let report = TestReport {
        total_seeds: num_seeds as u32,
        passed: num_seeds as u32 - failed,
        failed,
        results,
        failures: Vec::new(),
    };

    // Write JSON report
    let report_path = format!("{}/report.json", output_dir);
    if let Err(e) = report.write_to_file(&report_path) {
        eprintln!("Failed to write report: {}", e);
    }

    println!("Batch test complete in {:.2?}", elapsed);
    println!("  Total: {}", report.total_seeds);
    println!("  Passed: {}", report.passed);
    println!("  Failed: {}", report.failed);
    println!("  Pass rate: {:.1}%", report.pass_rate() * 100.0);
    println!("  Report: {}", report_path);
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_batch_test_minimal() {
        // Test with 1 seed to verify the pipeline works
        super::run(&[
            "--seeds".to_string(),
            "1".to_string(),
            "--output-dir".to_string(),
            std::env::temp_dir()
                .join("voxel_batch_test")
                .to_string_lossy()
                .to_string(),
        ]);
    }
}
