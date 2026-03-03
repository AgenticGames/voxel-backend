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

/// A fluid source cell to be injected into the fluid simulation.
#[derive(Debug, Clone)]
pub struct FluidSeed {
    /// Chunk coordinate
    pub chunk: (i32, i32, i32),
    /// Local X within chunk
    pub lx: u8,
    /// Local Y within chunk
    pub ly: u8,
    /// Local Z within chunk
    pub lz: u8,
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

/// Place cave pools in a density field. Returns descriptors and fluid seeds for placed pools.
pub fn place_pools(
    density: &mut DensityField,
    config: &PoolConfig,
    world_origin: Vec3,
    global_seed: u64,
    chunk_seed: u64,
) -> (Vec<PoolDescriptor>, Vec<FluidSeed>) {
    if !config.enabled {
        eprintln!("[POOL] disabled, skipping");
        return (Vec::new(), Vec::new());
    }

    let size = density.size;
    if size < 4 {
        return (Vec::new(), Vec::new());
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
        eprintln!("[POOL] chunk ({},{},{}) no floor cells found (size={})",
            world_origin.x, world_origin.y, world_origin.z, size);
        return (Vec::new(), Vec::new());
    }

    // Step 2: Cluster adjacent floors via BFS flood-fill on XZ at similar Y
    let clusters = cluster_floors(&floor_cells, size, config.min_area);
    eprintln!("[POOL] chunk ({},{},{}) floors={} clusters={} (min_area={}, thresh={:.2}, chance={:.2}, basin_depth={}, max_radius={})",
        world_origin.x, world_origin.y, world_origin.z,
        floor_cells.len(), clusters.len(),
        config.min_area, config.placement_threshold, config.pool_chance,
        config.basin_depth, config.max_radius);

    if clusters.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Noise and RNG for filtering
    let noise = Simplex3D::new(global_seed);
    let mut rng = ChaCha8Rng::seed_from_u64(chunk_seed.wrapping_add(0xB001_CAFE));

    let mut descriptors = Vec::new();
    let mut fluid_seeds = Vec::new();

    // Compute chunk coordinate from world origin and chunk size
    let chunk_size = density.size - 1; // density grid is chunk_size + 1
    let chunk_cx = (world_origin.x / chunk_size as f32).floor() as i32;
    let chunk_cy = (world_origin.y / chunk_size as f32).floor() as i32;
    let chunk_cz = (world_origin.z / chunk_size as f32).floor() as i32;

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
            eprintln!("[POOL]   cluster cells={} noise={:.3} < thresh={:.2} → skip",
                cluster.cells.len(), noise_val, config.placement_threshold);
            continue;
        }

        // Step 4: RNG filter
        let rng_roll = rng.gen::<f32>();
        if rng_roll >= config.pool_chance {
            eprintln!("[POOL]   cluster cells={} rng={:.3} >= chance={:.2} → skip",
                cluster.cells.len(), rng_roll, config.pool_chance);
            continue;
        }

        // Three-way probability: empty / lava / water
        let sum = config.water_pct + config.lava_pct + config.empty_pct;
        if sum <= 0.0 {
            continue;
        }
        let roll = rng.gen::<f32>() * sum;
        if roll < config.empty_pct {
            continue; // skip this site entirely
        }
        let fluid_type = if roll < config.empty_pct + config.lava_pct {
            PoolFluid::Lava
        } else {
            PoolFluid::Water
        };

        // Step 5: Carve basin
        // Compute cluster extent on XZ to determine effective radius.
        // Use area-based radius (sqrt(cells/PI)) instead of min-extent,
        // because worm tunnels produce elongated clusters where min-extent
        // would give radius=1 even for large clusters.
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
        let area_radius = ((cluster.cells.len() as f32) / std::f32::consts::PI).sqrt() as usize;
        let effective_radius = area_radius.min(config.max_radius).max(1);

        let center_x = (min_x + max_x) / 2;
        let center_z = (min_z + max_z) / 2;
        // Use the actual floor Y at the center position, not the cluster minimum.
        // This prevents pools from being carved at the wrong Y level when a
        // sloped tunnel produces a cluster spanning several Y levels.
        let floor_y = find_floor_y_at_center(&cluster.cells, center_x, center_z)
            .unwrap_or(cluster.min_y);
        let surface_y = floor_y + 1; // pool surface is at the air layer above floor

        // Verify headroom: sample multiple points around the center and accept
        // if ANY point has sufficient air above. In curved worm tunnels, the
        // geometric center may have low clearance while nearby points are fine.
        let headroom_offsets: [(i32, i32); 5] = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)];
        let mut has_headroom = false;
        for &(hdx, hdz) in &headroom_offsets {
            let hx = center_x as i32 + hdx;
            let hz = center_z as i32 + hdz;
            if hx < 0 || hx >= size as i32 || hz < 0 || hz >= size as i32 {
                continue;
            }
            let mut point_ok = true;
            for dy in 1..=config.min_air_above {
                let check_y = surface_y + dy;
                if check_y >= size {
                    break;
                }
                let sample = density.get(hx as usize, check_y, hz as usize);
                if sample.material.is_solid() {
                    point_ok = false;
                    break;
                }
            }
            if point_ok {
                has_headroom = true;
                break;
            }
        }
        if !has_headroom {
            eprintln!("[POOL]   cluster cells={} no headroom at center ({},{},{}) → skip",
                cluster.cells.len(), center_x, surface_y, center_z);
            continue;
        }

        // Validate: floor_y should be solid and surface_y should be air at center.
        // This catches cases where the chosen Y doesn't have a valid floor/air interface.
        if floor_y < size && surface_y < size {
            let floor_sample = density.get(center_x, floor_y, center_z);
            let surface_sample = density.get(center_x, surface_y, center_z);
            if !floor_sample.material.is_solid() || surface_sample.material.is_solid() {
                eprintln!("[POOL]   cluster cells={} invalid floor/air at center ({},{},{}) floor_solid={} surface_air={} → skip",
                    cluster.cells.len(), center_x, floor_y, center_z,
                    floor_sample.material.is_solid(), !surface_sample.material.is_solid());
                continue;
            }
        }

        eprintln!("[POOL]   CARVING pool: center=({},{},{}) radius={} basin_depth={} fluid={:?}",
            center_x, floor_y, center_z, effective_radius, config.basin_depth, fluid_type);

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

        // Step 7: Collect fluid seeds at the pool surface level within the basin
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
                if surface_y >= size {
                    continue;
                }
                fluid_seeds.push(FluidSeed {
                    chunk: (chunk_cx, chunk_cy, chunk_cz),
                    lx: gx as u8,
                    ly: surface_y as u8,
                    lz: gz as u8,
                    fluid_type,
                });
            }
        }
    }

    (descriptors, fluid_seeds)
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
                        // Similar Y level: within 1 voxel (tighter to avoid
                        // merging floor cells at very different heights)
                        if !visited.contains(&(nx, ny, nz))
                            && (ny as i32 - cy as i32).unsigned_abs() <= 1
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

/// Find the floor Y at the pool center position, or nearest floor cell.
/// Falls back to modal Y (most common Y level) if no nearby cell exists.
fn find_floor_y_at_center(
    cells: &[(usize, usize, usize)],
    cx: usize,
    cz: usize,
) -> Option<usize> {
    // 1. Exact match at (cx, cz)
    for &(x, y, z) in cells {
        if x == cx && z == cz {
            return Some(y);
        }
    }

    // 2. Nearest cell within radius 2 on XZ plane
    let mut best_dist = u32::MAX;
    let mut best_y = None;
    for &(x, y, z) in cells {
        let dx = (x as i32 - cx as i32).unsigned_abs();
        let dz = (z as i32 - cz as i32).unsigned_abs();
        if dx <= 2 && dz <= 2 {
            let dist = dx * dx + dz * dz;
            if dist < best_dist {
                best_dist = dist;
                best_y = Some(y);
            }
        }
    }
    if best_y.is_some() {
        return best_y;
    }

    // 3. Modal Y (most common Y level in the cluster)
    let mut y_counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for &(_, y, _) in cells {
        *y_counts.entry(y).or_insert(0) += 1;
    }
    y_counts.into_iter().max_by_key(|&(_, count)| count).map(|(y, _)| y)
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
        let (descriptors, seeds) = place_pools(
            &mut density,
            &config,
            Vec3::ZERO,
            42,
            42,
        );
        assert!(descriptors.is_empty());
        assert!(seeds.is_empty());
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

        let (r1, s1) = place_pools(&mut density1, &config, Vec3::ZERO, 42, 100);
        let (r2, s2) = place_pools(&mut density2, &config, Vec3::ZERO, 42, 100);

        assert_eq!(r1.len(), r2.len(), "Pool count should be deterministic");
        assert_eq!(s1.len(), s2.len(), "Seed count should be deterministic");
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.world_x, b.world_x);
            assert_eq!(a.world_y, b.world_y);
            assert_eq!(a.world_z, b.world_z);
            assert_eq!(a.fluid_type, b.fluid_type);
        }
    }

    #[test]
    fn test_empty_pct_produces_no_pools() {
        let config = PoolConfig {
            pool_chance: 1.0,
            placement_threshold: -1.0,
            min_area: 2,
            water_pct: 0.0,
            lava_pct: 0.0,
            empty_pct: 1.0,
            ..Default::default()
        };

        let size = 17;
        let mut density = DensityField::new(size);
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let sample = density.get_mut(x, y, z);
                    if y < size / 2 {
                        sample.density = 1.0;
                        sample.material = Material::Limestone;
                    } else {
                        sample.density = -1.0;
                        sample.material = Material::Air;
                    }
                }
            }
        }

        let (descriptors, seeds) = place_pools(&mut density, &config, Vec3::ZERO, 42, 100);
        assert!(descriptors.is_empty(), "empty_pct=1.0 should produce no pools");
        assert!(seeds.is_empty(), "empty_pct=1.0 should produce no seeds");
    }

    #[test]
    fn test_water_pct_produces_water_seeds() {
        let config = PoolConfig {
            pool_chance: 1.0,
            placement_threshold: -1.0,
            min_area: 2,
            water_pct: 1.0,
            lava_pct: 0.0,
            empty_pct: 0.0,
            ..Default::default()
        };

        let size = 17;
        let mut density = DensityField::new(size);
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let sample = density.get_mut(x, y, z);
                    if y < size / 2 {
                        sample.density = 1.0;
                        sample.material = Material::Limestone;
                    } else {
                        sample.density = -1.0;
                        sample.material = Material::Air;
                    }
                }
            }
        }

        let (descriptors, seeds) = place_pools(&mut density, &config, Vec3::ZERO, 42, 100);
        for d in &descriptors {
            assert_eq!(d.fluid_type, PoolFluid::Water, "water_pct=1.0 should produce only water pools");
        }
        for s in &seeds {
            assert_eq!(s.fluid_type, PoolFluid::Water, "water_pct=1.0 should produce only water seeds");
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

    /// Diagnostic: Generate real cave density and check if pools are placed.
    /// This uses the full generation pipeline (noise + worms) to create
    /// realistic terrain and then tests pool placement with extreme settings.
    #[test]
    fn test_pools_in_real_cave_terrain() {
        use crate::config::GenerationConfig;
        use voxel_core::chunk::ChunkCoord;

        let mut config = GenerationConfig::default();
        // Extreme pool settings: every floor cluster should get a pool
        config.pools = PoolConfig {
            enabled: true,
            placement_frequency: 0.08,
            placement_threshold: -1.0, // accept everything (noise >= -1 always true)
            pool_chance: 1.0,          // every candidate becomes a pool
            min_area: 2,               // tiny clusters qualify
            max_radius: 4,
            basin_depth: 3,
            rim_height: 1,
            water_pct: 1.0,
            lava_pct: 0.0,
            empty_pct: 0.0,
            min_air_above: 1, // reduced headroom requirement
        };
        // Disable formations so they don't interfere
        config.formations.enabled = false;

        let mut total_pools = 0;
        let mut total_floors = 0;
        let mut total_clusters = 0;
        let mut chunks_with_caves = 0;

        // Try a 4x4x4 grid of chunks to find ones with caves
        for cx in -2..2 {
            for cy in -2..2 {
                for cz in -2..2 {
                    let coord = ChunkCoord::new(cx, cy, cz);
                    let (density, pool_descs, fluid_seeds) =
                        crate::generate_density(coord, &config);

                    // Count floor cells manually to diagnose
                    let size = density.size;
                    let mut floor_count = 0;
                    for z in 1..size - 1 {
                        for y in 1..size - 2 {
                            for x in 1..size - 1 {
                                let sample = density.get(x, y, z);
                                let above = density.get(x, y + 1, z);
                                if sample.material.is_solid() && !above.material.is_solid() {
                                    floor_count += 1;
                                }
                            }
                        }
                    }

                    if floor_count > 0 {
                        chunks_with_caves += 1;
                    }
                    total_floors += floor_count;
                    total_pools += pool_descs.len();

                    if !pool_descs.is_empty() {
                        eprintln!(
                            "[DIAG] chunk ({},{},{}) floors={} pools={} seeds={}",
                            cx, cy, cz, floor_count, pool_descs.len(), fluid_seeds.len()
                        );
                        for d in &pool_descs {
                            eprintln!(
                                "  pool at ({:.0},{:.0},{:.0}) r={:.0} fluid={:?}",
                                d.world_x, d.world_y, d.world_z, d.radius, d.fluid_type
                            );
                        }
                    }
                }
            }
        }

        eprintln!(
            "\n[DIAG SUMMARY] chunks_with_caves={} total_floors={} total_pools={}",
            chunks_with_caves, total_floors, total_pools
        );

        // We expect SOME pools to be placed across 64 chunks
        assert!(
            chunks_with_caves > 0,
            "No chunks with caves found in 4x4x4 grid — terrain gen may not produce caves at these coords"
        );
        assert!(
            total_pools > 0,
            "No pools placed despite {} floor cells across {} chunks with caves. \
             Floor detection or clustering may not work with real cave geometry.",
            total_floors,
            chunks_with_caves
        );
    }

    #[test]
    fn test_pool_floor_y_matches_center() {
        // Create a density field with a sloped floor: Y=5 at x<8, Y=10 at x>=8.
        // The cluster center will be at x=8 where floor is Y=10.
        // The bug was using min_y=5 which would place the pool inside the wall.
        let config = PoolConfig {
            pool_chance: 1.0,
            placement_threshold: -1.0,
            min_area: 2,
            water_pct: 1.0,
            lava_pct: 0.0,
            empty_pct: 0.0,
            basin_depth: 2,
            max_radius: 3,
            min_air_above: 1,
            ..Default::default()
        };

        let size = 17;
        let mut density = DensityField::new(size);

        // Fill everything solid first
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let sample = density.get_mut(x, y, z);
                    sample.density = 1.0;
                    sample.material = Material::Limestone;
                }
            }
        }

        // Carve a tunnel with a sloped floor:
        // At x < 8, floor is at Y=5 (air from Y=6..12)
        // At x >= 8, floor is at Y=10 (air from Y=11..12)
        // The slope is within ±1 Y per step so BFS will still connect them
        let floor_heights: Vec<(usize, usize)> = (3usize..14).map(|x| {
            let floor_y = if x < 8 { 5 + (x.saturating_sub(3).min(5)) } else { 10 };
            (x, floor_y)
        }).collect();

        for &(x, floor_y) in &floor_heights {
            for z in 5..12 {
                for y in (floor_y + 1)..13 {
                    let sample = density.get_mut(x, y, z);
                    sample.density = -1.0;
                    sample.material = Material::Air;
                }
            }
        }

        let (descriptors, _seeds) = place_pools(&mut density, &config, Vec3::ZERO, 42, 100);

        // If any pool is placed, its floor_y (in world space) should NOT be at Y=5
        // (the old min_y). It should be at the actual floor level near the center.
        for d in &descriptors {
            let pool_floor_y = d.surface_y - 1.0; // surface_y = floor_y + 1
            let pool_center_x = d.world_x as usize;

            // Look up what the actual floor Y is at the pool center
            let expected_floor_y = floor_heights.iter()
                .find(|&&(x, _)| x == pool_center_x)
                .map(|&(_, fy)| fy as f32);

            if let Some(expected) = expected_floor_y {
                let diff = (pool_floor_y - expected).abs();
                assert!(
                    diff <= 1.0,
                    "Pool at x={} has floor_y={} but actual floor is at y={}. \
                     Pool placed at wrong Y level (off by {} voxels).",
                    pool_center_x, pool_floor_y, expected, diff
                );
            }
        }
    }

    #[test]
    fn test_find_floor_y_at_center_exact() {
        let cells = vec![(5, 10, 5), (6, 10, 5), (7, 8, 5)];
        // Exact match at (5, 5)
        assert_eq!(find_floor_y_at_center(&cells, 5, 5), Some(10));
        // Exact match at (7, 5) — different Y
        assert_eq!(find_floor_y_at_center(&cells, 7, 5), Some(8));
    }

    #[test]
    fn test_find_floor_y_at_center_nearest() {
        let cells = vec![(5, 10, 5), (6, 10, 5)];
        // No exact match at (5, 6), but (5, 5) is within radius 2
        assert_eq!(find_floor_y_at_center(&cells, 5, 6), Some(10));
    }

    #[test]
    fn test_find_floor_y_at_center_modal() {
        // No cells near (0, 0), should return modal Y (10 appears twice)
        let cells = vec![(10, 10, 10), (11, 10, 10), (12, 8, 10)];
        assert_eq!(find_floor_y_at_center(&cells, 0, 0), Some(10));
    }
}
