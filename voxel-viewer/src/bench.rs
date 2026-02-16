mod region;

use std::time::Instant;
use glam::Vec3;
use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use voxel_gen::config::GenerationConfig;
use region::GeneratedRegion;

fn main() {
    let chunk_size: usize = 38;
    let range = (3, 3, 3);
    let seeds = [1u64, 2, 7, 42, 100];
    let mine_count = 10;
    let mine_radius = 5.0f32;

    println!("=== Voxel Backend Benchmark ===");
    println!("Config: chunk_size={}, range={}x{}x{}", chunk_size, range.0, range.1, range.2);
    println!("Seeds: {:?}", seeds);
    println!("Mine tests: {} sphere mines per region (radius={})", mine_count, mine_radius);
    println!();

    let mut gen_times = Vec::new();
    let mut mine_times = Vec::new();
    let mut mine_full_times = Vec::new(); // mine + json
    let mut json_times = Vec::new();
    let mut total_verts = Vec::new();

    for &seed in &seeds {
        let mut config = GenerationConfig::default();
        config.seed = seed;
        config.chunk_size = chunk_size;

        // Benchmark generation
        let gen_start = Instant::now();
        let mut region = GeneratedRegion::generate(
            config,
            (0, 0, 0),
            (range.0, range.1, range.2),
            false,
        );
        let gen_elapsed = gen_start.elapsed();
        gen_times.push(gen_elapsed.as_secs_f64());

        // Benchmark JSON serialization
        let json_start = Instant::now();
        let json = region.to_json_mesh();
        let json_elapsed = json_start.elapsed();
        json_times.push(json_elapsed.as_secs_f64());

        let verts = json.positions.len() / 3;
        let tris = json.indices.len() / 3;
        total_verts.push(verts);

        println!("Seed {}: gen={:.0}ms, json={:.0}ms, {} verts, {} tris",
            seed, gen_elapsed.as_secs_f64() * 1000.0, json_elapsed.as_secs_f64() * 1000.0, verts, tris);

        // Benchmark mining at random locations within the region bounds
        let region_size = chunk_size as f32 * range.0 as f32;
        let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(999));

        for i in 0..mine_count {
            // Random point within the region
            let center = Vec3::new(
                rng.gen_range(mine_radius..region_size - mine_radius),
                rng.gen_range(mine_radius..region_size - mine_radius),
                rng.gen_range(mine_radius..region_size - mine_radius),
            );

            // Mine only (carve + remesh + rebuild combined)
            let mine_start = Instant::now();
            let result = region.mine_sphere(center, mine_radius);
            let mine_elapsed = mine_start.elapsed();

            // Full roundtrip: mine + JSON conversion (what the user feels)
            let json_start2 = Instant::now();
            let _json2 = region.to_json_mesh();
            let json_elapsed2 = json_start2.elapsed();

            mine_times.push(mine_elapsed.as_secs_f64());
            mine_full_times.push(mine_elapsed.as_secs_f64() + json_elapsed2.as_secs_f64());

            let total_mined: u32 = result.mined_materials.values().sum();
            if i < 3 {
                println!("  Mine[{}] at ({:.0},{:.0},{:.0}): mine={:.0}ms + json={:.0}ms = {:.0}ms total, {} voxels",
                    i, center.x, center.y, center.z,
                    mine_elapsed.as_secs_f64() * 1000.0,
                    json_elapsed2.as_secs_f64() * 1000.0,
                    (mine_elapsed + json_elapsed2).as_secs_f64() * 1000.0,
                    total_mined);
            }
        }
    }

    println!();
    println!("=== RESULTS ===");

    let gen_avg = gen_times.iter().sum::<f64>() / gen_times.len() as f64;
    let gen_min = gen_times.iter().cloned().fold(f64::MAX, f64::min);
    let gen_max = gen_times.iter().cloned().fold(0.0f64, f64::max);
    println!("Generation:    avg={:.0}ms  min={:.0}ms  max={:.0}ms  (n={})",
        gen_avg * 1000.0, gen_min * 1000.0, gen_max * 1000.0, gen_times.len());

    let mine_avg = mine_times.iter().sum::<f64>() / mine_times.len() as f64;
    let mine_min = mine_times.iter().cloned().fold(f64::MAX, f64::min);
    let mine_max = mine_times.iter().cloned().fold(0.0f64, f64::max);
    println!("Mine (carve):  avg={:.0}ms  min={:.0}ms  max={:.0}ms  (n={})",
        mine_avg * 1000.0, mine_min * 1000.0, mine_max * 1000.0, mine_times.len());

    let full_avg = mine_full_times.iter().sum::<f64>() / mine_full_times.len() as f64;
    let full_min = mine_full_times.iter().cloned().fold(f64::MAX, f64::min);
    let full_max = mine_full_times.iter().cloned().fold(0.0f64, f64::max);
    println!("Mine (total):  avg={:.0}ms  min={:.0}ms  max={:.0}ms  (n={})",
        full_avg * 1000.0, full_min * 1000.0, full_max * 1000.0, mine_full_times.len());

    let json_avg = json_times.iter().sum::<f64>() / json_times.len() as f64;
    println!("JSON conv:     avg={:.0}ms  (n={})", json_avg * 1000.0, json_times.len());

    let vert_avg = total_verts.iter().sum::<usize>() / total_verts.len();
    println!("Avg verts:     {}", vert_avg);
}
