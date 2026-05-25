use std::time::Instant;
use voxel_noise::simplex::Simplex3D;
use voxel_noise::NoiseSource;

fn main() {
    let noise = Simplex3D::new(424242);
    const N: usize = 4_000_000;

    // Warm up
    let mut acc = 0.0f64;
    for i in 0..200_000 {
        let x = (i as f64) * 0.013 - 1234.0;
        let y = (i as f64) * 0.019 + 567.0;
        let z = (i as f64) * 0.029 - 89.0;
        acc += noise.sample(x, y, z);
    }
    std::hint::black_box(acc);

    let start = Instant::now();
    let mut acc = 0.0f64;
    for i in 0..N {
        let x = (i as f64) * 0.013 - 1234.0;
        let y = (i as f64) * 0.019 + 567.0;
        let z = (i as f64) * 0.029 - 89.0;
        acc += noise.sample(x, y, z);
    }
    std::hint::black_box(acc);
    let elapsed = start.elapsed();

    let ns_per = elapsed.as_nanos() as f64 / N as f64;
    println!(
        "Simplex3D: {} samples in {:?} -> {:.2} ns/sample (acc={:.3})",
        N, elapsed, ns_per, acc
    );
}
