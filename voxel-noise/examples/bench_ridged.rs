use std::time::Instant;
use voxel_noise::ridged::RidgedMulti;
use voxel_noise::simplex::Simplex3D;
use voxel_noise::NoiseSource;

fn main() {
    // Match in-tree usage: copper_ridged = 5 oct, reef_noise = 4 oct
    let ridged5 = RidgedMulti::new(Simplex3D::new(424242), 5, 2.5, 2.0);
    let ridged4 = RidgedMulti::new(Simplex3D::new(424242), 4, 2.0, 2.0);
    const N: usize = 1_000_000;

    // Warm up
    let mut acc = 0.0f64;
    for i in 0..100_000 {
        let x = (i as f64) * 0.013 - 1234.0;
        let y = (i as f64) * 0.019 + 567.0;
        let z = (i as f64) * 0.029 - 89.0;
        acc += ridged5.sample(x, y, z);
        acc += ridged4.sample(x, y, z);
    }
    std::hint::black_box(acc);

    let start = Instant::now();
    let mut acc = 0.0f64;
    for i in 0..N {
        let x = (i as f64) * 0.013 - 1234.0;
        let y = (i as f64) * 0.019 + 567.0;
        let z = (i as f64) * 0.029 - 89.0;
        acc += ridged5.sample(x, y, z);
    }
    std::hint::black_box(acc);
    let elapsed = start.elapsed();
    let ns_per = elapsed.as_nanos() as f64 / N as f64;
    println!(
        "Ridged(5 oct): {} samples in {:?} -> {:.2} ns/sample (acc={:.3})",
        N, elapsed, ns_per, acc
    );

    let start = Instant::now();
    let mut acc = 0.0f64;
    for i in 0..N {
        let x = (i as f64) * 0.013 - 1234.0;
        let y = (i as f64) * 0.019 + 567.0;
        let z = (i as f64) * 0.029 - 89.0;
        acc += ridged4.sample(x, y, z);
    }
    std::hint::black_box(acc);
    let elapsed = start.elapsed();
    let ns_per = elapsed.as_nanos() as f64 / N as f64;
    println!(
        "Ridged(4 oct): {} samples in {:?} -> {:.2} ns/sample (acc={:.3})",
        N, elapsed, ns_per, acc
    );
}
