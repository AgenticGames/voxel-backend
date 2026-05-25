use std::time::Instant;
use voxel_gen::config::GenerationConfig;
use voxel_gen::density::generate_density_field;

fn main() {
    let config = GenerationConfig::default();
    let origin = glam::Vec3::new(0.0, -150.0, 0.0);

    // Warm up
    for _ in 0..2 {
        let _ = generate_density_field(&config, origin);
    }

    const RUNS: usize = 5;
    let mut total = std::time::Duration::ZERO;
    for r in 0..RUNS {
        let start = Instant::now();
        let f = generate_density_field(&config, origin);
        let el = start.elapsed();
        total += el;
        // Don't optimize away
        std::hint::black_box(&f);
        println!("Run {}: {:?}", r, el);
    }
    println!(
        "Avg generate_density_field: {:?} (over {} runs)",
        total / RUNS as u32,
        RUNS
    );
}
