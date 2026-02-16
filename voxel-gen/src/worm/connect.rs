use glam::Vec3;
use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

/// Plan worm connections between cavern centers in a region.
///
/// Uses a region-deterministic RNG to select which centers to connect.
/// Each connection is a (start, end) pair for worm path generation.
pub fn plan_worm_connections(
    seed: u64,
    cavern_centers: &[Vec3],
    worms_per_region: u32,
) -> Vec<(Vec3, Vec3)> {
    if cavern_centers.len() < 2 {
        return Vec::new();
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut connections = Vec::new();
    let n = cavern_centers.len();

    // Generate up to worms_per_region connections
    for _ in 0..worms_per_region {
        let a = rng.gen_range(0..n);
        let mut b = rng.gen_range(0..n);
        // Ensure we don't connect a center to itself
        if n > 1 {
            while b == a {
                b = rng.gen_range(0..n);
            }
        } else {
            continue;
        }

        connections.push((cavern_centers[a], cavern_centers[b]));
    }

    connections
}

/// Find cavern centers from a density field by identifying regions of low density.
///
/// Scans the density field at a coarse resolution and identifies local minima
/// (most air-like) as cavern centers, in world-space coordinates.
pub fn find_cavern_centers(
    densities: &[f32],
    size: usize,
    world_origin: Vec3,
) -> Vec<Vec3> {
    let mut centers = Vec::new();
    let step = (size / 4).max(1);

    for z in (step..size - step).step_by(step) {
        for y in (step..size - step).step_by(step) {
            for x in (step..size - step).step_by(step) {
                let idx = z * size * size + y * size + x;
                let density = densities[idx];

                // Consider this a cavern center if density is negative (air)
                if density < -0.05 {
                    // Check if this is a local minimum in the neighborhood
                    let is_local_min = is_local_minimum(densities, size, x, y, z, step);
                    if is_local_min {
                        centers.push(Vec3::new(
                            world_origin.x + x as f32,
                            world_origin.y + y as f32,
                            world_origin.z + z as f32,
                        ));
                    }
                }
            }
        }
    }

    centers
}

/// Check if position (x,y,z) is a local minimum in the density field
fn is_local_minimum(densities: &[f32], size: usize, x: usize, y: usize, z: usize, step: usize) -> bool {
    let center_idx = z * size * size + y * size + x;
    let center_val = densities[center_idx];

    // Check 6 cardinal neighbors at step distance
    let neighbors = [
        (x.wrapping_sub(step), y, z),
        (x + step, y, z),
        (x, y.wrapping_sub(step), z),
        (x, y + step, z),
        (x, y, z.wrapping_sub(step)),
        (x, y, z + step),
    ];

    for (nx, ny, nz) in neighbors {
        if nx < size && ny < size && nz < size {
            let n_idx = nz * size * size + ny * size + nx;
            if densities[n_idx] < center_val {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_connections_empty() {
        let connections = plan_worm_connections(42, &[], 3);
        assert!(connections.is_empty());
    }

    #[test]
    fn test_plan_connections_single_center() {
        let centers = vec![Vec3::new(5.0, 5.0, 5.0)];
        let connections = plan_worm_connections(42, &centers, 3);
        assert!(connections.is_empty());
    }

    #[test]
    fn test_plan_connections_two_centers() {
        let centers = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 10.0, 10.0),
        ];
        let connections = plan_worm_connections(42, &centers, 3);
        assert_eq!(connections.len(), 3);
        for (a, b) in &connections {
            assert_ne!(*a, *b);
        }
    }

    #[test]
    fn test_plan_connections_deterministic() {
        let centers = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 10.0, 10.0),
            Vec3::new(5.0, 5.0, 5.0),
        ];
        let c1 = plan_worm_connections(42, &centers, 5);
        let c2 = plan_worm_connections(42, &centers, 5);
        assert_eq!(c1.len(), c2.len());
        for (a, b) in c1.iter().zip(c2.iter()) {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1, b.1);
        }
    }

    #[test]
    fn test_find_cavern_centers_all_solid() {
        // All densities 1.0 → no cavern centers
        let size = 16;
        let densities = vec![1.0f32; size * size * size];
        let centers = find_cavern_centers(&densities, size, Vec3::ZERO);
        assert!(centers.is_empty());
    }
}
