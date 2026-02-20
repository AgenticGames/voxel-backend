//! Cave pool placement: water and lava lakes on cave floors.
//!
//! Runs per-chunk after worm carving (Step 3), before formations (Step 4).
//! Detects flat floor areas, clusters them, filters by noise + RNG,
//! carves basins, and emits pool descriptors for the viewer.

use std::collections::VecDeque;

use glam::Vec3;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use voxel_core::material::Material;
use voxel_noise::simplex::Simplex3D;
use voxel_noise::NoiseSource;

use crate::config::PoolConfig;
use crate::density::DensityField;

/// Fluid type for a pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoolFluid {
    Water,
    Lava,
}

/// Descriptor for a placed pool, returned to the viewer for rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolDescriptor {
    /// World-space X of pool center
    pub world_x: f32,
    /// World-space Y of pool center
    pub world_y: f32,
    /// World-space Z of pool center
    pub world_z: f32,
    /// Pool radius in voxels
    pub radius: f32,
    /// World-space Y of the pool surface (water level)
    pub surface_y: f32,
    /// World-space Y of the basin floor
    pub floor_y: f32,
    /// Water or Lava
    pub fluid_type: PoolFluid,
}

/// A cluster of adjacent floor cells at similar Y levels.
struct FloorCluster {
    cells: Vec<(usize, usize, usize)>, // (x, y, z) in grid coords
    centroid_x: f32,
    centroid_y: f32,
    centroid_z: f32,
    min_y: usize,
}

/// Place cave pools in a density field. Returns descriptors for placed pools.
pub fn place_pools(
    density: &mut DensityField,
    config: &PoolConfig,
    world_origin: Vec3,
    global_seed: u64,
    chunk_seed: u64,
) -> Vec<PoolDescriptor> {
    if !config.enabled {
        return Vec::new();
    }

    let size = density.size;
    if size < 4 {
        return Vec::new();
    }

    // Step 1: Detect floor surfaces — solid voxels with air above
    let mut floor_cells: Vec<(usize, usize, usize)> = Vec::new();
    for z in 1..size - 1 {
        for y in 1..size - 2 {
            for x in 1..size - 1 {
                let sample = density.get(x, y, z);
                let above = density.get(x, y + 1, z);
                if sample.material.is_solid() && !above.material.is_solid() {
                    floor_cells.push((x, y, z));
                }
            }
        }
    }

    if floor_cells.is_empty() {
        return Vec::new();
    }

    // Step 2: Cluster adjacent floors via BFS flood-fill on XZ at similar Y
    let clusters = cluster_floors(&floor_cells, size, config.min_area);

    if clusters.is_empty() {
        return Vec::new();
    }

    // Noise and RNG for filtering
    let noise = Simplex3D::new(global_seed);
    let mut rng = ChaCha8Rng::seed_from_u64(chunk_seed.wrapping_add(0xB001_CAFE));

    let mut descriptors = Vec::new();

    for cluster in &clusters {
        // Step 3: Noise filter at cluster centroid (world-space)
        let world_cx = world_origin.x + cluster.centroid_x;
        let world_cy = world_origin.y + cluster.centroid_y;
        let world_cz = world_origin.z + cluster.centroid_z;

        let noise_val = noise.sample(
            world_cx as f64 * config.placement_frequency,
            world_cy as f64 * config.placement_frequency,
            world_cz as f64 * config.placement_frequency,
        );
        if noise_val < config.placement_threshold {
            continue;
        }

        // Step 4: RNG filter
        if rng.gen::<f32>() >= config.pool_chance {
            continue;
        }

        // Determine fluid type based on depth and lava_fraction
        let fluid_type = if world_cy < config.lava_depth_max as f32
            && rng.gen::<f32>() < config.lava_fraction
        {
            PoolFluid::Lava
        } else {
            PoolFluid::Water
        };

        // Step 5: Carve basin
        // Compute cluster extent on XZ to determine effective radius
        let mut min_x = usize::MAX;
        let mut max_x = 0usize;
        let mut min_z = usize::MAX;
        let mut max_z = 0usize;
        for &(cx, _cy, cz) in &cluster.cells {
            min_x = min_x.min(cx);
            max_x = max_x.max(cx);
            min_z = min_z.min(cz);
            max_z = max_z.max(cz);
        }
        let extent_x = (max_x - min_x + 1) as f32;
        let extent_z = (max_z - min_z + 1) as f32;
        let half_extent = (extent_x.min(extent_z) / 2.0).floor() as usize;
        let effective_radius = half_extent.min(config.max_radius).max(1);

        let center_x = (min_x + max_x) / 2;
        let center_z = (min_z + max_z) / 2;
        let floor_y = cluster.min_y;
        let surface_y = floor_y + 1; // pool surface is at the air layer above floor

        // Verify headroom
        let mut has_headroom = true;
        for dy in 1..=config.min_air_above {
            let check_y = surface_y + dy;
            if check_y >= size {
                break;
            }
            let sample = density.get(center_x, check_y, center_z);
            if sample.material.is_solid() {
                has_headroom = false;
                break;
            }
        }
        if !has_headroom {
            continue;
        }

        let r2 = (effective_radius * effective_radius) as i32;

        // Carve basin: set voxels below floor to air within radius circle
        for dz in -(effective_radius as i32)..=(effective_radius as i32) {
            for dx in -(effective_radius as i32)..=(effective_radius as i32) {
                if dx * dx + dz * dz > r2 {
                    continue;
                }
                let gx = center_x as i32 + dx;
                let gz = center_z as i32 + dz;
                if gx < 0 || gx >= size as i32 || gz < 0 || gz >= size as i32 {
                    continue;
                }
                let gx = gx as usize;
                let gz = gz as usize;

                // Carve basin_depth voxels below the floor
                for d in 0..config.basin_depth {
                    let gy = floor_y.wrapping_sub(d);
                    if gy >= size || gy == 0 {
                        break;
                    }
                    let sample = density.get_mut(gx, gy, gz);
                    if sample.material.is_solid() {
                        sample.density = -1.0;
                        sample.material = Material::Air;
                    }
                }

                // Also ensure the floor surface itself is air (the pool water level)
                if surface_y < size {
                    let sample = density.get_mut(gx, surface_y, gz);
                    if sample.material.is_solid() {
                        sample.density = -1.0;
                        sample.material = Material::Air;
                    }
                }
            }
        }

        // Reinforce rim: solid ring at floor level just outside the pool radius
        let rim_r = effective_radius + 1;
        let rim_r2 = (rim_r * rim_r) as i32;
        for dz in -(rim_r as i32)..=(rim_r as i32) {
            for dx in -(rim_r as i32)..=(rim_r as i32) {
                let dist2 = dx * dx + dz * dz;
                // Only the ring between pool radius and rim radius
                if dist2 <= r2 || dist2 > rim_r2 {
                    continue;
                }
                let gx = center_x as i32 + dx;
                let gz = center_z as i32 + dz;
                if gx < 1 || gx >= size as i32 - 1 || gz < 1 || gz >= size as i32 - 1 {
                    continue;
                }
                let gx = gx as usize;
                let gz = gz as usize;

                for h in 0..config.rim_height {
                    let gy = floor_y + h;
                    if gy >= size {
                        break;
                    }
                    if !density.get(gx, gy, gz).material.is_solid() {
                        // Get the host rock material from nearby solid before mutating
                        let host_mat = find_nearby_solid(density, gx, gy, gz, size);
                        let sample = density.get_mut(gx, gy, gz);
                        sample.density = 1.0;
                        sample.material = host_mat;
                    }
                }
            }
        }

        // Step 6: Emit descriptor
        descriptors.push(PoolDescriptor {
            world_x: world_origin.x + center_x as f32,
            world_y: world_origin.y + surface_y as f32,
            world_z: world_origin.z + center_z as f32,
            radius: effective_radius as f32,
            surface_y: world_origin.y + surface_y as f32,
            floor_y: world_origin.y + (floor_y as f32 - config.basin_depth as f32),
            fluid_type,
        });
    }

    descriptors
}

/// Check if a pool's rim is still intact (for drainable pool support).
/// Returns true if the pool is still contained (all rim voxels solid).
pub fn is_pool_contained(
    density: &DensityField,
    descriptor: &PoolDescriptor,
    world_origin: Vec3,
) -> bool {
    let size = density.size;
    let effective_radius = descriptor.radius as usize;
    let r2 = (effective_radius * effective_radius) as i32;
    let rim_r = effective_radius + 1;
    let rim_r2 = (rim_r * rim_r) as i32;

    // Convert world coords back to local grid coords
    let center_x = (descriptor.world_x - world_origin.x) as i32;
    let center_z = (descriptor.world_z - world_origin.z) as i32;
    let surface_y = (descriptor.surface_y - world_origin.y) as usize;

    for dz in -(rim_r as i32)..=(rim_r as i32) {
        for dx in -(rim_r as i32)..=(rim_r as i32) {
            let dist2 = dx * dx + dz * dz;
            if dist2 <= r2 || dist2 > rim_r2 {
                continue;
            }
            let gx = center_x + dx;
            let gz = center_z + dz;
            if gx < 0 || gx >= size as i32 || gz < 0 || gz >= size as i32 {
                continue;
            }
            // Check at floor level (surface_y - 1) and surface_y
            for check_y in [surface_y.saturating_sub(1), surface_y] {
                if check_y >= size {
                    continue;
                }
                let sample = density.get(gx as usize, check_y, gz as usize);
                if !sample.material.is_solid() {
                    return false; // Rim breached
                }
            }
        }
    }

    true
}

/// BFS flood-fill to cluster adjacent floor cells on XZ at similar Y levels.
fn cluster_floors(
    floor_cells: &[(usize, usize, usize)],
    grid_size: usize,
    min_area: usize,
) -> Vec<FloorCluster> {
    // Build a lookup set for O(1) membership checks
    // Key: (x, z) -> y values at that position
    let mut floor_map: std::collections::HashMap<(usize, usize), Vec<usize>> =
        std::collections::HashMap::new();
    for &(x, y, z) in floor_cells {
        floor_map.entry((x, z)).or_default().push(y);
    }

    let mut visited: std::collections::HashSet<(usize, usize, usize)> =
        std::collections::HashSet::new();
    let mut clusters = Vec::new();

    // Sort floor cells for deterministic iteration
    let mut sorted_cells = floor_cells.to_vec();
    sorted_cells.sort();

    for &(x, y, z) in &sorted_cells {
        if visited.contains(&(x, y, z)) {
            continue;
        }

        // BFS from this cell
        let mut queue = VecDeque::new();
        let mut cells = Vec::new();
        queue.push_back((x, y, z));
        visited.insert((x, y, z));

        while let Some((cx, cy, cz)) = queue.pop_front() {
            cells.push((cx, cy, cz));

            // Check 4-connected neighbors on XZ plane
            let neighbors: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
            for (dx, dz) in neighbors {
                let nx = cx as i32 + dx;
                let nz = cz as i32 + dz;
                if nx < 0 || nx >= grid_size as i32 || nz < 0 || nz >= grid_size as i32 {
                    continue;
                }
                let nx = nx as usize;
                let nz = nz as usize;

                if let Some(ys) = floor_map.get(&(nx, nz)) {
                    for &ny in ys {
                        // Similar Y level: within 2 voxels
                        if !visited.contains(&(nx, ny, nz))
                            && (ny as i32 - cy as i32).unsigned_abs() <= 2
                        {
                            visited.insert((nx, ny, nz));
                            queue.push_back((nx, ny, nz));
                        }
                    }
                }
            }
        }

        if cells.len() < min_area {
            continue;
        }

        // Compute centroid and min_y
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_z = 0.0f32;
        let mut min_y = usize::MAX;
        for &(cx, cy, cz) in &cells {
            sum_x += cx as f32;
            sum_y += cy as f32;
            sum_z += cz as f32;
            min_y = min_y.min(cy);
        }
        let n = cells.len() as f32;

        clusters.push(FloorCluster {
            cells,
            centroid_x: sum_x / n,
            centroid_y: sum_y / n,
            centroid_z: sum_z / n,
            min_y,
        });
    }

    clusters
}

/// Find the material of a nearby solid voxel (for rim reinforcement).
fn find_nearby_solid(density: &DensityField, x: usize, y: usize, z: usize, size: usize) -> Material {
    let offsets: [(i32, i32, i32); 6] = [
        (0, -1, 0),
        (0, 1, 0),
        (-1, 0, 0),
        (1, 0, 0),
        (0, 0, -1),
        (0, 0, 1),
    ];
    for (dx, dy, dz) in offsets {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        let nz = z as i32 + dz;
        if nx >= 0 && nx < size as i32 && ny >= 0 && ny < size as i32 && nz >= 0 && nz < size as i32 {
            let sample = density.get(nx as usize, ny as usize, nz as usize);
            if sample.material.is_solid() {
                return sample.material;
            }
        }
    }
    // Fallback: use Limestone as default host rock
    Material::Limestone
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PoolConfig;

    #[test]
    fn test_disabled_returns_empty() {
        let config = PoolConfig {
            enabled: false,
            ..Default::default()
        };
        let mut density = DensityField::new(17);
        let result = place_pools(
            &mut density,
            &config,
            Vec3::ZERO,
            42,
            42,
        );
        assert!(result.is_empty());
    }

    #[test]
    fn test_place_pools_deterministic() {
        let config = PoolConfig {
            pool_chance: 1.0,
            placement_threshold: -1.0, // accept everything
            min_area: 2,
            ..Default::default()
        };

        // Create a density field with a flat floor
        let size = 17;
        let mut density1 = DensityField::new(size);
        let mut density2 = DensityField::new(size);

        // Fill bottom half with solid, top half with air
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let sample1 = density1.get_mut(x, y, z);
                    let sample2 = density2.get_mut(x, y, z);
                    if y < size / 2 {
                        sample1.density = 1.0;
                        sample1.material = Material::Limestone;
                        sample2.density = 1.0;
                        sample2.material = Material::Limestone;
                    } else {
                        sample1.density = -1.0;
                        sample1.material = Material::Air;
                        sample2.density = -1.0;
                        sample2.material = Material::Air;
                    }
                }
            }
        }

        let r1 = place_pools(&mut density1, &config, Vec3::ZERO, 42, 100);
        let r2 = place_pools(&mut density2, &config, Vec3::ZERO, 42, 100);

        assert_eq!(r1.len(), r2.len(), "Pool count should be deterministic");
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.world_x, b.world_x);
            assert_eq!(a.world_y, b.world_y);
            assert_eq!(a.world_z, b.world_z);
            assert_eq!(a.fluid_type, b.fluid_type);
        }
    }

    #[test]
    fn test_cluster_floors_basic() {
        // 3 adjacent floor cells should form one cluster
        let cells = vec![(5, 8, 5), (6, 8, 5), (7, 8, 5), (5, 8, 6), (6, 8, 6), (7, 8, 6)];
        let clusters = cluster_floors(&cells, 17, 3);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].cells.len(), 6);
    }

    #[test]
    fn test_cluster_floors_min_area_filter() {
        // 2 cells, min_area=3 → no clusters
        let cells = vec![(5, 8, 5), (6, 8, 5)];
        let clusters = cluster_floors(&cells, 17, 3);
        assert!(clusters.is_empty());
    }
}
