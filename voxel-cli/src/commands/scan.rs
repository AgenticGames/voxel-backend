use std::collections::HashMap;
use std::time::Instant;

use voxel_core::dual_contouring::mesh_gen::generate_mesh;
use voxel_core::dual_contouring::solve::solve_dc_vertices;
use voxel_core::hermite::HermiteData;
use voxel_core::mesh::Mesh;
use voxel_core::world_scan::{self, ScanConfig, ScanSeamData, ScanWormSegment};
use voxel_gen::config::{GenerationConfig, WormConfig};
use voxel_gen::hermite_extract::extract_hermite_data;
use voxel_gen::region_gen;

/// Run the scan command.
///
/// Usage: voxel-cli scan [--seed N] [--chunks N] [--output PATH] [--no-PASS] [--self-intersection] [--rays N] [--subsamples N]
pub fn run(args: &[String]) {
    let mut seed = 42u64;
    let mut chunks_per_axis = 7i32; // load_radius=3 → -3..+3 = 7 per axis
    let mut output = "scan_report.json".to_string();
    let mut scan_config = ScanConfig::default();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                if i < args.len() { seed = args[i].parse().unwrap_or(42); }
            }
            "--chunks" => {
                i += 1;
                if i < args.len() { chunks_per_axis = args[i].parse().unwrap_or(3); }
            }
            "--output" | "-o" => {
                i += 1;
                if i < args.len() { output = args[i].clone(); }
            }
            // Disable flags
            "--no-density-seam" => { scan_config.enable_density_seam = false; }
            "--no-mesh-topology" => { scan_config.enable_mesh_topology = false; }
            "--no-seam-completeness" => { scan_config.enable_seam_completeness = false; }
            "--no-navigability" => { scan_config.enable_navigability = false; }
            "--no-worm-truncation" => { scan_config.enable_worm_truncation = false; }
            "--no-thin-walls" => { scan_config.enable_thin_walls = false; }
            "--no-winding" => { scan_config.enable_winding_consistency = false; }
            "--no-degenerate-tri" => { scan_config.enable_degenerate_triangles = false; }
            "--no-worm-carve" => { scan_config.enable_worm_carve_verify = false; }
            "--self-intersection" => { scan_config.enable_self_intersection = true; }
            "--no-seam-quality" => { scan_config.enable_seam_mesh_quality = false; }
            // Accuracy params
            "--rays" => {
                i += 1;
                if i < args.len() { scan_config.raymarch_rays_per_chunk = args[i].parse().unwrap_or(0); }
            }
            "--subsamples" => {
                i += 1;
                if i < args.len() { scan_config.density_subsample_count = args[i].parse().unwrap_or(0); }
            }
            _ => { eprintln!("Unknown option: {}", args[i]); }
        }
        i += 1;
    }

    // Cap at 9 for sanity (load_radius=4 → 9x9x9 = 729 chunks max)
    chunks_per_axis = chunks_per_axis.min(9).max(1);

    let config = GenerationConfig {
        seed,
        chunk_size: 24,
        region_size: 3,
        worm: WormConfig {
            radius_min: 6.0,
            radius_max: 8.0,
            step_length: 0.8,
            max_steps: 800,
            ..Default::default()
        },
        ..Default::default()
    };
    let chunk_size = config.chunk_size;

    println!(
        "World scan: seed={}, {}x{}x{} chunks ({} total)",
        seed,
        chunks_per_axis,
        chunks_per_axis,
        chunks_per_axis,
        chunks_per_axis * chunks_per_axis * chunks_per_axis
    );

    let start = Instant::now();

    // Generate NxNxN chunk coordinates centered at origin
    let half = chunks_per_axis / 2;
    let coords: Vec<(i32, i32, i32)> = {
        let mut v = Vec::new();
        for cz in -half..chunks_per_axis - half {
            for cy in -half..chunks_per_axis - half {
                for cx in -half..chunks_per_axis - half {
                    v.push((cx, cy, cz));
                }
            }
        }
        v
    };

    // Phase 1: Generate region densities (includes worm carving + formations)
    let gen_start = Instant::now();
    let (density_fields, _pools, worm_paths, _timings) =
        region_gen::generate_region_densities(&coords, &config);
    let gen_elapsed = gen_start.elapsed();
    println!("  Region generated in {:.2?} ({} chunks)", gen_elapsed, coords.len());

    // Phase 2: Mesh all chunks + extract seam data + collect hermite data
    let mesh_start = Instant::now();
    let mut base_meshes: HashMap<(i32, i32, i32), Mesh> = HashMap::new();
    let mut seam_data: HashMap<(i32, i32, i32), ScanSeamData> = HashMap::new();
    let mut hermite_map: HashMap<(i32, i32, i32), HermiteData> = HashMap::new();

    for &coord in &coords {
        let density = match density_fields.get(&coord) {
            Some(d) => d,
            None => continue,
        };

        let hermite = extract_hermite_data(density);
        let cell_size = density.size - 1;
        let dc_vertices = solve_dc_vertices(&hermite, cell_size);
        let mesh = generate_mesh(&hermite, &dc_vertices, cell_size);

        // Extract boundary edges for seam completeness check
        let boundary_edges = region_gen::extract_boundary_edges(&hermite, chunk_size);

        base_meshes.insert(coord, mesh);
        seam_data.insert(coord, ScanSeamData { boundary_edges });
        hermite_map.insert(coord, hermite);
    }

    let mesh_elapsed = mesh_start.elapsed();
    println!("  Meshed {} chunks in {:.2?}", base_meshes.len(), mesh_elapsed);

    // Phase 3: Convert worm paths to scan format
    let scan_worm_paths: Vec<Vec<ScanWormSegment>> = worm_paths
        .iter()
        .map(|path| {
            path.iter()
                .map(|seg| ScanWormSegment {
                    position: [seg.position.x, seg.position.y, seg.position.z],
                    radius: seg.radius,
                })
                .collect()
        })
        .collect();

    // Phase 4: Run world scan
    let scan_start = Instant::now();
    let mut result = world_scan::scan_world_with_config(
        &density_fields,
        &base_meshes,
        Some(&hermite_map),
        &seam_data,
        &scan_worm_paths,
        chunk_size,
        &scan_config,
    );
    result.seed = Some(seed);
    let scan_elapsed = scan_start.elapsed();
    println!("  Scan completed in {:.2?}", scan_elapsed);

    // Phase 5: Write JSON report
    let json = result.to_json_string();
    if let Err(e) = std::fs::write(&output, &json) {
        eprintln!("Failed to write report: {}", e);
        std::process::exit(1);
    }

    let total_elapsed = start.elapsed();

    // Print summary
    println!(
        "Scan complete: {} chunks, {} issues ({} errors, {} warnings, {} info), {} volumes",
        result.chunks_scanned,
        result.issues.len(),
        result.summary.total_errors,
        result.summary.total_warnings,
        result.summary.total_info,
        result.volumes.len()
    );
    println!("Total time: {:.2?}", total_elapsed);
    println!("Report written to: {}", output);
}
