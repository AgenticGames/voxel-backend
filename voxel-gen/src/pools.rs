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
    min_y: usize,
    centroid_x: f32,
    centroid_y: f32,
    centroid_z: f32,
}

/// Place cave pools in a density field. Returns descriptors and fluid seeds for placed pools.
pub fn place_pools(
    density: &mut DensityField,
    config: &PoolConfig,
    world_origin: Vec3,
    global_seed: u64,
    chunk_seed: u64,
    chunk_coord: (i32, i32, i32),
) -> (Vec<PoolDescriptor>, Vec<FluidSeed>) {
    if !config.enabled {
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
        return (Vec::new(), Vec::new());
    }

    // Step 2: Cluster adjacent floors via BFS flood-fill on XZ at similar Y
    let clusters = cluster_floors(&floor_cells, size, config.min_area);

    if clusters.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Noise and RNG for filtering
    let noise = Simplex3D::new(global_seed);
    let mut rng = ChaCha8Rng::seed_from_u64(chunk_seed.wrapping_add(0xB001_CAFE));

    let mut descriptors = Vec::new();
    let mut fluid_seeds = Vec::new();

    let (chunk_cx, chunk_cy, chunk_cz) = chunk_coord;

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
        let rng_roll = rng.gen::<f32>();
        if rng_roll >= config.pool_chance {
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
        // Compute effective radius from cluster XZ extent
        let mut min_x = usize::MAX;
        let mut max_x = 0usize;
        let mut min_z = usize::MAX;
        let mut max_z = 0usize;
        for &(cx, _, cz) in &cluster.cells {
            if cx < min_x { min_x = cx; }
            if cx > max_x { max_x = cx; }
            if cz < min_z { min_z = cz; }
            if cz > max_z { max_z = cz; }
        }
        let extent_x = (max_x - min_x + 1) as f32;
        let extent_z = (max_z - min_z + 1) as f32;
        let half_extent = (extent_x.min(extent_z) / 2.0).floor() as usize;
        let effective_radius = half_extent.min(config.max_radius).max(1);

        let center_x = (min_x + max_x) / 2;
        let center_z = (min_z + max_z) / 2;
        let floor_y = cluster.min_y;
        let surface_y = floor_y + 1;

        let r2 = (effective_radius * effective_radius) as i32;

        // Headroom check: single center-point
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
                    let host_mat = find_nearby_solid(density, gx, gy, gz, size);
                    let sample = density.get_mut(gx, gy, gz);
                    sample.density = 1.0;
                    sample.material = host_mat;
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

        // Step 7: Pre-fill entire basin volume with fluid seeds
        let basin_bottom_y = floor_y.saturating_sub(config.basin_depth.saturating_sub(1));
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
                // Fill every Y level from basin bottom to surface
                for gy in basin_bottom_y..=surface_y {
                    if gy >= size {
                        continue;
                    }
                    fluid_seeds.push(FluidSeed {
                        chunk: (chunk_cx, chunk_cy, chunk_cz),
                        lx: gx as u8,
                        ly: gy as u8,
                        lz: gz as u8,
                        fluid_type,
                    });
                }
            }
        }
    }

    (descriptors, fluid_seeds)
}

/// Diagnostic result for each placement gate, used by the force-spawn debug tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolDiagnostics {
    /// Target voxel coordinates (local to chunk)
    pub target_x: usize,
    pub target_y: usize,
    pub target_z: usize,
    /// Chunk coordinate
    pub chunk_cx: i32,
    pub chunk_cy: i32,
    pub chunk_cz: i32,
    /// Gate 1: floor cell valid (solid with air above, no horizontal air neighbors)
    pub floor_cell_valid: bool,
    pub floor_has_horizontal_air: bool,
    pub floor_density_positive: bool,
    pub floor_air_above: bool,
    /// Gate 2: ground depth
    pub ground_depth_measured: usize,
    pub ground_depth_required: usize,
    pub ground_depth_pass: bool,
    /// Gate 3: cluster size (BFS floor count)
    pub cluster_size_measured: usize,
    pub cluster_size_required: usize,
    pub cluster_size_pass: bool,
    /// Gate 4: noise value
    pub noise_value: f64,
    pub noise_threshold: f64,
    pub noise_pass: bool,
    /// Gate 5: air above
    pub air_above_measured: usize,
    pub air_above_required: usize,
    pub air_above_pass: bool,
    /// Gate 6: cave height (distance to ceiling)
    pub cave_height_measured: usize,
    pub cave_height_max: usize,
    pub cave_height_pass: bool,
    /// Gate 7: floor thickness below basin bottom
    pub floor_thickness_measured: usize,
    pub floor_thickness_required: usize,
    pub floor_thickness_pass: bool,
    /// Summary
    pub checks_passed: usize,
    pub checks_total: usize,
    pub would_spawn_naturally: bool,
}

/// Force-spawn a pool at a specific location for debugging. Runs each placement gate
/// as a read-only diagnostic, then unconditionally carves a basin and emits fluid seeds.
///
/// If the target voxel is air, scans downward up to 8 voxels to find a floor surface.
///
/// Returns diagnostics and fluid seeds for the carved pool.
pub fn force_spawn_pool(
    density: &mut DensityField,
    config: &PoolConfig,
    world_origin: Vec3,
    global_seed: u64,
    target_x: usize,
    target_y: usize,
    target_z: usize,
    fluid_type: PoolFluid,
    chunk_coord: (i32, i32, i32),
) -> (PoolDiagnostics, Vec<FluidSeed>) {
    let size = density.size;

    // If target is air, scan downward to find nearest floor surface
    let mut floor_y = target_y;
    if target_y < size {
        let sample = density.get(target_x, target_y, target_z);
        if sample.density <= 0.0 {
            // Air — scan down up to 8 voxels
            for d in 1..=8usize {
                let check_y = target_y.wrapping_sub(d);
                if check_y == 0 || check_y >= size {
                    break;
                }
                let s = density.get(target_x, check_y, target_z);
                if s.density > 0.0 {
                    floor_y = check_y;
                    break;
                }
            }
        }
    }

    // Clamp to valid range
    let floor_y = floor_y.min(size.saturating_sub(3));
    let surface_y = floor_y + 1;

    // === Run diagnostic gates (read-only) ===

    // Gate 1: Floor cell valid
    let floor_density_positive = floor_y < size && density.get(target_x, floor_y, target_z).density > 0.0;
    let floor_air_above = surface_y < size && density.get(target_x, surface_y, target_z).density <= 0.0;
    let has_air_xn = target_x > 0 && density.get(target_x - 1, floor_y, target_z).density <= 0.0;
    let has_air_xp = target_x + 1 < size && density.get(target_x + 1, floor_y, target_z).density <= 0.0;
    let has_air_zn = target_z > 0 && density.get(target_x, floor_y, target_z - 1).density <= 0.0;
    let has_air_zp = target_z + 1 < size && density.get(target_x, floor_y, target_z + 1).density <= 0.0;
    let floor_has_horizontal_air = has_air_xn || has_air_xp || has_air_zn || has_air_zp;
    let floor_cell_valid = floor_density_positive && floor_air_above && !floor_has_horizontal_air;

    // Gate 2: Ground depth
    let ground_depth_required = config.min_ground_depth;
    let mut ground_depth_measured = 0usize;
    for d in 1..=ground_depth_required.max(8) {
        let check_y = floor_y.wrapping_sub(d);
        if check_y == 0 || check_y >= size {
            ground_depth_measured = ground_depth_required.max(8);
            break;
        }
        if density.get(target_x, check_y, target_z).density > 0.0 {
            ground_depth_measured += 1;
        } else {
            break;
        }
    }
    let ground_depth_pass = ground_depth_required == 0 || ground_depth_measured >= ground_depth_required;

    // Gate 3: Cluster size — BFS floor count around target
    let cluster_size_required = config.min_area;
    let mut cluster_size_measured = 0usize;
    {
        let mut visited = std::collections::HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back((target_x, floor_y, target_z));
        visited.insert((target_x, target_z));
        while let Some((cx, cy, cz)) = queue.pop_front() {
            // Check if this is a valid floor cell
            if cx >= size || cy >= size || cy + 1 >= size {
                continue;
            }
            let s = density.get(cx, cy, cz);
            let a = density.get(cx, cy + 1, cz);
            if s.density <= 0.0 || a.density > 0.0 {
                continue;
            }
            cluster_size_measured += 1;
            // BFS expand on XZ
            for &(dx, dz) in &[(!0isize, 0isize), (1, 0), (0, !0), (0, 1)] {
                let nx = cx.wrapping_add(dx as usize);
                let nz = cz.wrapping_add(dz as usize);
                if nx < size && nz < size && visited.insert((nx, nz)) {
                    // Check Y tolerance
                    for dy_off in 0..=config.max_y_step {
                        for &sign in &[0isize, -1, 1] {
                            let ny = if sign == 0 { cy } else { cy.wrapping_add((dy_off as isize * sign) as usize) };
                            if ny < size && ny + 1 < size {
                                let ns = density.get(nx, ny, nz);
                                let na = density.get(nx, ny + 1, nz);
                                if ns.density > 0.0 && na.density <= 0.0 {
                                    queue.push_back((nx, ny, nz));
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let cluster_size_pass = cluster_size_measured >= cluster_size_required;

    // Gate 4: Noise value
    let noise = Simplex3D::new(global_seed);
    let world_x = world_origin.x + target_x as f32;
    let world_y = world_origin.y + floor_y as f32;
    let world_z = world_origin.z + target_z as f32;
    let noise_value = noise.sample(
        world_x as f64 * config.placement_frequency,
        world_y as f64 * config.placement_frequency,
        world_z as f64 * config.placement_frequency,
    );
    let noise_pass = noise_value >= config.placement_threshold;

    // Gate 5: Air above
    let air_above_required = config.min_air_above;
    let mut air_above_measured = 0usize;
    for dy in 1..size {
        let check_y = surface_y + dy;
        if check_y >= size {
            break;
        }
        if density.get(target_x, check_y, target_z).density <= 0.0 {
            air_above_measured += 1;
        } else {
            break;
        }
    }
    let air_above_pass = air_above_measured >= air_above_required;

    // Gate 6: Cave height (distance to ceiling from surface)
    let cave_height_max = config.max_cave_height;
    let mut cave_height_measured = 0usize;
    {
        let start_y = surface_y + 1;
        let mut found_ceiling = false;
        for dy in 0..size {
            let check_y = start_y + dy;
            if check_y >= size {
                break;
            }
            if density.get(target_x, check_y, target_z).material.is_solid() {
                cave_height_measured = dy;
                found_ceiling = true;
                break;
            }
        }
        if !found_ceiling {
            cave_height_measured = size; // no ceiling found
        }
    }
    let cave_height_pass = cave_height_max == 0 || cave_height_measured <= cave_height_max;

    // Gate 7: Floor thickness below basin bottom
    let floor_thickness_required = config.min_floor_thickness;
    let mut floor_thickness_measured = 0usize;
    if config.basin_depth > 0 {
        let basin_bottom = floor_y.saturating_sub(config.basin_depth);
        for d in 1..=floor_thickness_required.max(8) {
            let check_y = basin_bottom.wrapping_sub(d);
            if check_y >= size || check_y == 0 {
                floor_thickness_measured = floor_thickness_required.max(8);
                break;
            }
            if density.get(target_x, check_y, target_z).material.is_solid() {
                floor_thickness_measured += 1;
            } else {
                break;
            }
        }
    } else {
        floor_thickness_measured = floor_thickness_required; // no basin, trivially passes
    }
    let floor_thickness_pass = floor_thickness_required == 0 || floor_thickness_measured >= floor_thickness_required;

    // Summary
    let checks = [
        floor_cell_valid,
        ground_depth_pass,
        cluster_size_pass,
        noise_pass,
        air_above_pass,
        cave_height_pass,
        floor_thickness_pass,
    ];
    let checks_passed = checks.iter().filter(|&&c| c).count();
    let checks_total = checks.len();
    let would_spawn_naturally = checks_passed == checks_total;

    let diagnostics = PoolDiagnostics {
        target_x,
        target_y: floor_y,
        target_z,
        chunk_cx: chunk_coord.0,
        chunk_cy: chunk_coord.1,
        chunk_cz: chunk_coord.2,
        floor_cell_valid,
        floor_has_horizontal_air,
        floor_density_positive,
        floor_air_above,
        ground_depth_measured,
        ground_depth_required,
        ground_depth_pass,
        cluster_size_measured,
        cluster_size_required,
        cluster_size_pass,
        noise_value,
        noise_threshold: config.placement_threshold,
        noise_pass,
        air_above_measured,
        air_above_required,
        air_above_pass,
        cave_height_measured,
        cave_height_max,
        cave_height_pass,
        floor_thickness_measured,
        floor_thickness_required,
        floor_thickness_pass,
        checks_passed,
        checks_total,
        would_spawn_naturally,
    };

    // === Force-carve basin regardless of diagnostics ===
    let effective_radius = config.max_radius.max(1);
    let r2 = (effective_radius * effective_radius) as i32;

    // Carve basin
    for dz in -(effective_radius as i32)..=(effective_radius as i32) {
        for dx in -(effective_radius as i32)..=(effective_radius as i32) {
            if dx * dx + dz * dz > r2 {
                continue;
            }
            let gx = target_x as i32 + dx;
            let gz = target_z as i32 + dz;
            if gx < 0 || gx >= size as i32 || gz < 0 || gz >= size as i32 {
                continue;
            }
            let gx = gx as usize;
            let gz = gz as usize;

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

            // Ensure surface is air
            if surface_y < size {
                let sample = density.get_mut(gx, surface_y, gz);
                if sample.material.is_solid() {
                    sample.density = -1.0;
                    sample.material = Material::Air;
                }
            }
        }
    }

    // Emit fluid seeds
    let mut fluid_seeds = Vec::new();
    let basin_bottom_y = floor_y.saturating_sub(config.basin_depth.saturating_sub(1));
    for dz in -(effective_radius as i32)..=(effective_radius as i32) {
        for dx in -(effective_radius as i32)..=(effective_radius as i32) {
            if dx * dx + dz * dz > r2 {
                continue;
            }
            let gx = target_x as i32 + dx;
            let gz = target_z as i32 + dz;
            if gx < 0 || gx >= size as i32 || gz < 0 || gz >= size as i32 {
                continue;
            }
            for gy in basin_bottom_y..=surface_y {
                if gy >= size {
                    continue;
                }
                fluid_seeds.push(FluidSeed {
                    chunk: chunk_coord,
                    lx: gx as u8,
                    ly: gy as u8,
                    lz: gz as u8,
                    fluid_type,
                });
            }
        }
    }

    (diagnostics, fluid_seeds)
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

        let min_y = cells.iter().map(|&(_, y, _)| y).min().unwrap_or(0);
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_z = 0.0f32;
        for &(cx, cy, cz) in &cells {
            sum_x += cx as f32;
            sum_y += cy as f32;
            sum_z += cz as f32;
        }
        let n = cells.len() as f32;

        clusters.push(FloorCluster {
            cells,
            min_y,
            centroid_x: sum_x / n,
            centroid_y: sum_y / n,
            centroid_z: sum_z / n,
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
        let (descriptors, seeds) = place_pools(
            &mut density,
            &config,
            Vec3::ZERO,
            42,
            42,
            (0, 0, 0),
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
            max_cave_height: 0,       // disable ceiling check (flat terrain has no ceiling)
            min_floor_thickness: 0,   // disable floor thickness check
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

        let (r1, s1) = place_pools(&mut density1, &config, Vec3::ZERO, 42, 100, (0, 0, 0));
        let (r2, s2) = place_pools(&mut density2, &config, Vec3::ZERO, 42, 100, (0, 0, 0));

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
            max_cave_height: 0,
            min_floor_thickness: 0,
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

        let (descriptors, seeds) = place_pools(&mut density, &config, Vec3::ZERO, 42, 100, (0, 0, 0));
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
            max_cave_height: 0,
            min_floor_thickness: 0,
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

        let (descriptors, seeds) = place_pools(&mut density, &config, Vec3::ZERO, 42, 100, (0, 0, 0));
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

    #[test]
    fn test_cluster_floors_y_tolerance() {
        // Floor cells at Y=8, 9, 10, 9, 8 — all within hardcoded Y tolerance of 2
        let cells = vec![
            (3, 8, 5), (4, 9, 5), (5, 10, 5), (6, 9, 5), (7, 8, 5),
        ];
        let clusters = cluster_floors(&cells, 17, 2);
        assert_eq!(clusters.len(), 1, "cells within Y tolerance of 2 should form one cluster");
        assert_eq!(clusters[0].cells.len(), 5);
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
            ..Default::default()
        };
        // Disable formations so they don't interfere
        config.formations.enabled = false;

        let mut total_pools = 0;
        let mut total_floors = 0;
        let mut chunks_with_caves = 0;

        // Try a 4x4x4 grid of chunks to find ones with caves
        for cx in -2..2 {
            for cy in -2..2 {
                for cz in -2..2 {
                    let coord = ChunkCoord::new(cx, cy, cz);
                    let (density, pool_descs, fluid_seeds, _river_springs) =
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
    fn test_pool_prefills_basin_volume() {
        // Verify that seeds span multiple Y levels (not just surface_y)
        let config = PoolConfig {
            pool_chance: 1.0,
            placement_threshold: -1.0,
            min_area: 2,
            water_pct: 1.0,
            lava_pct: 0.0,
            empty_pct: 0.0,
            basin_depth: 3,
            max_cave_height: 0,
            min_floor_thickness: 0,
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

        let (descriptors, seeds) = place_pools(&mut density, &config, Vec3::ZERO, 42, 100, (0, 0, 0));
        if !descriptors.is_empty() {
            // Seeds should span multiple Y levels due to basin pre-fill
            let y_values: std::collections::HashSet<u8> = seeds.iter().map(|s| s.ly).collect();
            assert!(
                y_values.len() > 1,
                "Pool seeds should span multiple Y levels (basin_depth=3), got {} unique Y values: {:?}",
                y_values.len(), y_values
            );
        }
    }

}
