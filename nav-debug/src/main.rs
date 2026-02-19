use voxel_gen::config::GenerationConfig;
use voxel_core::chunk::ChunkCoord;
use std::collections::VecDeque;

fn count_components(density: &[f32], size: usize) -> (usize, Vec<usize>) {
    let total = size * size * size;
    let mut visited = vec![false; total];
    let mut components = Vec::new();
    let mut air_count = 0usize;

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = z * size * size + y * size + x;
                if density[idx] < 0.0 {
                    air_count += 1;
                    if !visited[idx] {
                        let mut count = 0usize;
                        let mut queue = VecDeque::new();
                        visited[idx] = true;
                        queue.push_back((x, y, z));
                        count += 1;

                        while let Some((cx, cy, cz)) = queue.pop_front() {
                            let neighbors = [
                                (cx.wrapping_sub(1), cy, cz),
                                (cx + 1, cy, cz),
                                (cx, cy.wrapping_sub(1), cz),
                                (cx, cy + 1, cz),
                                (cx, cy, cz.wrapping_sub(1)),
                                (cx, cy, cz + 1),
                            ];
                            for (nx, ny, nz) in neighbors {
                                if nx < size && ny < size && nz < size {
                                    let n_idx = nz * size * size + ny * size + nx;
                                    if !visited[n_idx] && density[n_idx] < 0.0 {
                                        visited[n_idx] = true;
                                        count += 1;
                                        queue.push_back((nx, ny, nz));
                                    }
                                }
                            }
                        }
                        components.push(count);
                    }
                }
            }
        }
    }
    (air_count, components)
}

fn main() {
    for seed in [35u64, 42] {
        let config = GenerationConfig {
            seed,
            ..Default::default()
        };
        let coord = ChunkCoord::new(0, 0, 0);
        let world_origin = coord.world_origin();
        let density = voxel_gen::density::generate_density_field(&config, world_origin);
        let densities = density.densities();
        let (air, mut components) = count_components(&densities, density.size);
        components.sort_by(|a, b| b.cmp(a));
        let largest = components.first().copied().unwrap_or(0);
        let pct = if air > 0 { largest as f64 / air as f64 * 100.0 } else { 100.0 };
        let num_comp = components.len();
        let small: Vec<_> = components.iter().take(5).collect();
        println!("Seed {}: air={} ({:.1}%), components={}, largest={} ({:.1}%), top5={:?}",
            seed, air, air as f64 / densities.len() as f64 * 100.0,
            num_comp, largest, pct, small);
    }
}
