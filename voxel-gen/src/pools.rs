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

    // Step 1: Detect floor surfaces — solid voxels with air above,
    // filtered by ground depth to reject pillar tops and formation surfaces.
    let mut floor_cells: Vec<(usize, usize, usize)> = Vec::new();
    for z in 1..size - 1 {
        for y in 1..size - 2 {
            for x in 1..size - 1 {
                let sample = density.get(x, y, z);
                let above = density.get(x, y + 1, z);
                // Use density values (not material enum) to match formations logic.
                // Floor = solid voxel (density > 0) with air above (density <= 0).
                if sample.density > 0.0 && above.density <= 0.0 {
                    // Wall exclusion: reject voxels that are also wall surfaces.
                    // A floor-only cell has solid neighbors horizontally; a wall cell
                    // has air to the side. This matches formations.rs detect_surfaces()
                    // which classifies Floor vs Wall mutually exclusively (lines 821-825).
                    let has_air_xn = x > 0 && density.get(x - 1, y, z).density <= 0.0;
                    let has_air_xp = x + 1 < size && density.get(x + 1, y, z).density <= 0.0;
                    let has_air_zn = z > 0 && density.get(x, y, z - 1).density <= 0.0;
                    let has_air_zp = z + 1 < size && density.get(x, y, z + 1).density <= 0.0;
                    if has_air_xn || has_air_xp || has_air_zn || has_air_zp {
                        continue; // wall surface or wall-floor edge — skip
                    }

                    // Ground depth check: require min_ground_depth contiguous solid below
                    if config.min_ground_depth > 0 {
                        let mut solid_below = 0usize;
                        for d in 1..=config.min_ground_depth {
                            let check_y = y.wrapping_sub(d);
                            if check_y == 0 || check_y >= size {
                                // Hit grid bottom — counts as passing
                                solid_below = config.min_ground_depth;
                                break;
                            }
                            if density.get(x, check_y, z).density > 0.0 {
                                solid_below += 1;
                            } else {
                                break; // air gap — not deep ground
                            }
                        }
                        if solid_below < config.min_ground_depth {
                            continue; // pillar top or formation surface
                        }
                    }

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
    let clusters = cluster_floors(&floor_cells, size, config.min_area, config.max_y_step);
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
        // Compute effective radius from cluster area.
        let area_radius = ((cluster.cells.len() as f32) / std::f32::consts::PI).sqrt() as usize;
        let effective_radius = area_radius.min(config.max_radius).max(1);

        // Pick center as the cluster cell closest to the XZ centroid.
        // Use that cell's actual Y — it's a real floor cell at the cave bottom.
        // (Median Y is wrong for bowl-shaped caverns where wall-climbing cells
        // inflate the median to mid-air. The adaptive footprint validation
        // handles undulating terrain around this center Y.)
        let (center_x, floor_y, center_z) = find_nearest_to_centroid(
            &cluster.cells, cluster.centroid_x, cluster.centroid_z,
        );
        let surface_y = floor_y + 1; // pool surface is at the air layer above floor

        let r2 = (effective_radius * effective_radius) as i32;

        // Pool footprint validation: check that enough of the disc has floor near center Y.
        // Scans within ±footprint_y_tolerance to accept undulating terrain.
        {
            let mut floor_count = 0u32;
            let mut total_count = 0u32;
            let tol = config.footprint_y_tolerance;
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
                    total_count += 1;
                    // Check if ANY Y in [median_y - tol, median_y + tol] has solid-below-air
                    let y_lo = floor_y.saturating_sub(tol);
                    let y_hi = (floor_y + tol).min(size.saturating_sub(2));
                    let mut cell_ok = false;
                    for check_y in y_lo..=y_hi {
                        if check_y + 1 < size {
                            let f = density.get(gx as usize, check_y, gz as usize);
                            let a = density.get(gx as usize, check_y + 1, gz as usize);
                            if f.material.is_solid() && !a.material.is_solid() {
                                cell_ok = true;
                                break;
                            }
                        }
                    }
                    if cell_ok {
                        floor_count += 1;
                    }
                }
            }
            if total_count > 0 && (floor_count * 2) < total_count {
                eprintln!("[POOL]   insufficient footprint floor={}/{} at center ({},{},{}) → skip",
                    floor_count, total_count, center_x, floor_y, center_z);
                continue;
            }
        }

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

        // Cave ceiling check: scan upward to find a solid ceiling within max_cave_height.
        // If no ceiling is found, this is open terrain (not inside a cave) → skip.
        if config.max_cave_height > 0 {
            let mut has_ceiling = false;
            let ceiling_offsets: [(i32, i32); 5] = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)];
            for &(cdx, cdz) in &ceiling_offsets {
                let cx_check = center_x as i32 + cdx;
                let cz_check = center_z as i32 + cdz;
                if cx_check < 0 || cx_check >= size as i32 || cz_check < 0 || cz_check >= size as i32 {
                    continue;
                }
                let start_y = surface_y + config.min_air_above + 1;
                for dy in 0..config.max_cave_height {
                    let check_y = start_y + dy;
                    if check_y >= size {
                        break;
                    }
                    if density.get(cx_check as usize, check_y, cz_check as usize).material.is_solid() {
                        has_ceiling = true;
                        break;
                    }
                }
                if has_ceiling {
                    break;
                }
            }
            if !has_ceiling {
                eprintln!("[POOL]   cluster cells={} no ceiling within {} at center ({},{},{}) → skip",
                    cluster.cells.len(), config.max_cave_height, center_x, surface_y, center_z);
                continue;
            }
        }

        // Validate: floor/air interface near center, scanning within ±footprint_y_tolerance
        // to handle undulating terrain.
        {
            let tol = config.footprint_y_tolerance;
            let y_lo = floor_y.saturating_sub(tol);
            let y_hi = (floor_y + tol).min(size.saturating_sub(2));
            let mut found_interface = false;
            for check_y in y_lo..=y_hi {
                if check_y + 1 < size {
                    let f = density.get(center_x, check_y, center_z);
                    let a = density.get(center_x, check_y + 1, center_z);
                    if f.material.is_solid() && !a.material.is_solid() {
                        found_interface = true;
                        break;
                    }
                }
            }
            if !found_interface {
                eprintln!("[POOL]   cluster cells={} no floor/air interface within ±{} of floor_y={} at center ({},{}) → skip",
                    cluster.cells.len(), tol, floor_y, center_x, center_z);
                continue;
            }
        }

        // Floor thickness check: ensure enough solid below basin bottom to support the pool.
        if config.min_floor_thickness > 0 && config.basin_depth > 0 {
            let thickness_offsets: [(i32, i32); 5] = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)];
            let mut thin_count = 0u32;
            let mut checked = 0u32;
            for &(tdx, tdz) in &thickness_offsets {
                let tx = center_x as i32 + tdx;
                let tz = center_z as i32 + tdz;
                if tx < 0 || tx >= size as i32 || tz < 0 || tz >= size as i32 {
                    continue;
                }
                checked += 1;
                let basin_bottom = floor_y.saturating_sub(config.basin_depth);
                let mut solid_below = 0usize;
                for d in 1..=config.min_floor_thickness {
                    let check_y = basin_bottom.wrapping_sub(d);
                    if check_y >= size || check_y == 0 {
                        break;
                    }
                    if density.get(tx as usize, check_y, tz as usize).material.is_solid() {
                        solid_below += 1;
                    }
                }
                if solid_below < config.min_floor_thickness {
                    thin_count += 1;
                }
            }
            // Skip if majority of sample points have insufficient floor thickness
            if checked > 0 && thin_count * 2 > checked {
                eprintln!("[POOL]   cluster cells={} thin floor at center ({},{},{}) thin={}/{} → skip",
                    cluster.cells.len(), center_x, floor_y, center_z, thin_count, checked);
                continue;
            }
        }

        // Log cluster Y range and chosen center for debugging
        let cluster_min_y = cluster.cells.iter().map(|&(_, y, _)| y).min().unwrap_or(0);
        let cluster_max_y = cluster.cells.iter().map(|&(_, y, _)| y).max().unwrap_or(0);
        eprintln!("[POOL]   CARVING pool: center=({},{},{}) radius={} basin_depth={} fluid={:?} cluster_y_range={}..{} cells={}",
            center_x, floor_y, center_z, effective_radius, config.basin_depth, fluid_type,
            cluster_min_y, cluster_max_y, cluster.cells.len());

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

                // DEBUG: Basin floor — set voxel just below carved depth to Limestone
                let basin_floor_y = floor_y.wrapping_sub(config.basin_depth);
                if basin_floor_y > 0 && basin_floor_y < size {
                    let sample = density.get_mut(gx, basin_floor_y, gz);
                    sample.density = 1.0;
                    sample.material = Material::Limestone;
                }

                // DEBUG: Below-basin foundation — one more layer below basin floor
                let foundation_y = basin_floor_y.wrapping_sub(1);
                if foundation_y > 0 && foundation_y < size {
                    let sample = density.get_mut(gx, foundation_y, gz);
                    if sample.material.is_solid() {
                        sample.material = Material::Limestone;
                    }
                }
            }
        }

        // DEBUG: Basin walls — set solid voxels adjacent to carved area to Limestone
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

                // Check each carved cell's 6 neighbors for solid — set to Limestone
                for d in 0..config.basin_depth {
                    let gy = floor_y.wrapping_sub(d);
                    if gy >= size || gy == 0 {
                        break;
                    }
                    let wall_offsets: [(i32, i32, i32); 6] = [
                        (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1),
                    ];
                    for (wdx, wdy, wdz) in wall_offsets {
                        let wx = gx as i32 + wdx;
                        let wy = gy as i32 + wdy;
                        let wz = gz as i32 + wdz;
                        if wx >= 0 && wx < size as i32 && wy >= 0 && wy < size as i32 && wz >= 0 && wz < size as i32 {
                            let ws = density.get_mut(wx as usize, wy as usize, wz as usize);
                            if ws.material.is_solid() {
                                ws.material = Material::Limestone;
                            }
                        }
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
                    // DEBUG: Use Limestone directly for rim visibility
                    let sample = density.get_mut(gx, gy, gz);
                    if !sample.material.is_solid() {
                        sample.density = 1.0;
                    }
                    sample.material = Material::Limestone;
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
/// `max_y_step` controls how many Y-levels apart adjacent cells can be and still merge.
fn cluster_floors(
    floor_cells: &[(usize, usize, usize)],
    grid_size: usize,
    min_area: usize,
    max_y_step: usize,
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
                        // Similar Y level: within max_y_step voxels
                        if !visited.contains(&(nx, ny, nz))
                            && (ny as i32 - cy as i32).unsigned_abs() <= max_y_step as u32
                        {
                            visited.insert((nx, ny, nz));
                            queue.push_back((nx, ny, nz));
                        }
                    }
                }
            }
        }

        // Y-range trimming: discard cells that climbed up cave walls.
        // Keep only cells within 2*max_y_step of the cluster's minimum Y.
        // This caps vertical span so wall-climbing BFS cells are removed.
        let min_y = cells.iter().map(|&(_, y, _)| y).min().unwrap_or(0);
        let max_allowed_y = min_y + 2 * max_y_step;
        let pre_trim = cells.len();
        cells.retain(|&(_, y, _)| y <= max_allowed_y);
        if cells.len() < pre_trim {
            eprintln!("[POOL]   cluster Y-trim: min_y={} max_allowed_y={} trimmed {} → {} cells",
                min_y, max_allowed_y, pre_trim, cells.len());
        }

        if cells.len() < min_area {
            continue;
        }

        // Compute centroid from retained (trimmed) cells only
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
            centroid_x: sum_x / n,
            centroid_y: sum_y / n,
            centroid_z: sum_z / n,
        });
    }

    clusters
}

/// Find the cluster cell nearest to the XZ centroid, preferring lowest Y.
/// Two-pass: (1) find min XZ dist², (2) among cells within threshold of that min, pick lowest Y.
/// This ensures we select the actual cave floor, not a wall/ledge cell at the same XZ.
fn find_nearest_to_centroid(
    cells: &[(usize, usize, usize)],
    centroid_x: f32,
    centroid_z: f32,
) -> (usize, usize, usize) {
    // Pass 1: find minimum XZ distance²
    let mut min_dist2 = f32::MAX;
    for &(x, _y, z) in cells {
        let dx = x as f32 - centroid_x;
        let dz = z as f32 - centroid_z;
        let dist2 = dx * dx + dz * dz;
        if dist2 < min_dist2 {
            min_dist2 = dist2;
        }
    }

    // Pass 2: among cells within threshold of min_dist², pick lowest Y
    let threshold = min_dist2 + 2.0;
    let mut best = cells[0];
    let mut best_y = usize::MAX;
    for &(x, y, z) in cells {
        let dx = x as f32 - centroid_x;
        let dz = z as f32 - centroid_z;
        let dist2 = dx * dx + dz * dz;
        if dist2 <= threshold && y < best_y {
            best_y = y;
            best = (x, y, z);
        }
    }
    best
}

// DEBUG: find_nearby_solid temporarily unused — rim uses Limestone directly for visibility.
// Restore when removing debug Limestone overrides.
#[allow(dead_code)]
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
        let clusters = cluster_floors(&cells, 17, 3, 1);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].cells.len(), 6);
    }

    #[test]
    fn test_cluster_floors_min_area_filter() {
        // 2 cells, min_area=3 → no clusters
        let cells = vec![(5, 8, 5), (6, 8, 5)];
        let clusters = cluster_floors(&cells, 17, 3, 1);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_cluster_floors_y_step_connects_undulating() {
        // Floor cells at Y=8, 9, 10, 9, 8 across X — undulating by ±1 per step.
        // With max_y_step=1, they should form one cluster.
        // With max_y_step=2, cells 2 steps apart should also connect.
        let cells = vec![
            (3, 8, 5), (4, 9, 5), (5, 10, 5), (6, 9, 5), (7, 8, 5),
        ];
        // max_y_step=1: all connect because each neighbor differs by exactly 1
        let clusters_1 = cluster_floors(&cells, 17, 2, 1);
        assert_eq!(clusters_1.len(), 1, "max_y_step=1 should connect ±1 undulating floor");
        assert_eq!(clusters_1[0].cells.len(), 5);

        // Cells with a 2-step Y gap between adjacent XZ neighbors.
        // Use enough cells at each Y level so they can form clusters individually.
        let cells_gap = vec![
            (3, 8, 5), (3, 8, 6),   // cluster A at Y=8
            (4, 10, 5), (4, 10, 6), // cluster B at Y=10
        ];
        // max_y_step=1: gaps of 2 → two separate clusters
        let clusters_tight = cluster_floors(&cells_gap, 17, 2, 1);
        assert_eq!(clusters_tight.len(), 2, "max_y_step=1 should NOT connect 2-step gaps");

        // max_y_step=2: gaps of 2 → single cluster
        let clusters_wide = cluster_floors(&cells_gap, 17, 2, 2);
        assert_eq!(clusters_wide.len(), 1, "max_y_step=2 should connect 2-step gaps");
        assert_eq!(clusters_wide[0].cells.len(), 4);
    }

    #[test]
    fn test_footprint_y_tolerance_accepts_undulating() {
        // Build a density field with an undulating floor (Y varies between 7-9)
        // and verify pools can place on it with footprint_y_tolerance=2.
        let config = PoolConfig {
            pool_chance: 1.0,
            placement_threshold: -1.0,
            min_area: 3,
            water_pct: 1.0,
            lava_pct: 0.0,
            empty_pct: 0.0,
            max_cave_height: 0,
            min_floor_thickness: 0,
            min_ground_depth: 0,
            max_y_step: 2,
            footprint_y_tolerance: 2,
            ..Default::default()
        };

        let size = 17;
        let mut density = DensityField::new(size);

        // Fill everything solid
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let sample = density.get_mut(x, y, z);
                    sample.density = 1.0;
                    sample.material = Material::Limestone;
                }
            }
        }

        // Carve air above an undulating floor (Y=7 at edges, Y=9 at center)
        for z in 3..14 {
            for x in 3..14 {
                let dx = (x as f32 - 8.0).abs();
                let dz = (z as f32 - 8.0).abs();
                let dist = dx.max(dz);
                let floor_y = if dist <= 2.0 { 9 } else if dist <= 4.0 { 8 } else { 7 };
                for y in (floor_y + 1)..15 {
                    let sample = density.get_mut(x, y, z);
                    sample.density = -1.0;
                    sample.material = Material::Air;
                }
            }
        }

        let (descriptors, _seeds) = place_pools(&mut density, &config, Vec3::ZERO, 42, 100);
        assert!(
            !descriptors.is_empty(),
            "Pools should place on undulating terrain with footprint_y_tolerance=2"
        );
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
    fn test_find_nearest_to_centroid_prefers_lowest_y() {
        // Two-pass: among cells near centroid, lowest Y wins.
        // (6,10,5) is at dist²=0, (7,8,5) is at dist²=1. Both within threshold=2.0.
        // Lowest Y=8 → picks (7,8,5).
        let cells = vec![(5, 10, 5), (6, 10, 5), (7, 8, 5)];
        let result = find_nearest_to_centroid(&cells, 6.0, 5.0);
        assert_eq!(result, (7, 8, 5));
    }

    #[test]
    fn test_find_nearest_to_centroid_offset() {
        // (10,3,10) is closest in XZ at dist²=0.5, and also has lowest Y=3
        let cells = vec![(2, 5, 2), (8, 7, 8), (10, 3, 10)];
        let result = find_nearest_to_centroid(&cells, 9.5, 9.5);
        assert_eq!(result, (10, 3, 10));
    }

    #[test]
    fn test_find_nearest_to_centroid_single() {
        let cells = vec![(4, 6, 4)];
        let result = find_nearest_to_centroid(&cells, 0.0, 0.0);
        assert_eq!(result, (4, 6, 4));
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

        let (descriptors, seeds) = place_pools(&mut density, &config, Vec3::ZERO, 42, 100);
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

    #[test]
    fn test_ground_depth_rejects_pillar_tops() {
        // Create a pillar 3 voxels tall with air below — should be rejected
        // when min_ground_depth=4
        let config = PoolConfig {
            pool_chance: 1.0,
            placement_threshold: -1.0,
            min_area: 2,
            water_pct: 1.0,
            lava_pct: 0.0,
            empty_pct: 0.0,
            max_cave_height: 0,
            min_floor_thickness: 0,
            min_ground_depth: 4, // require 4 solid below
            ..Default::default()
        };

        let size = 17;
        let mut density = DensityField::new(size);

        // Fill everything as air
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let sample = density.get_mut(x, y, z);
                    sample.density = -1.0;
                    sample.material = Material::Air;
                }
            }
        }

        // Create a thin pillar: solid at y=5,6,7 (3 voxels tall), air below at y=4
        // Top of pillar at y=7, air at y=8 — this is a "floor cell" at y=7
        // But only 2 solid below (y=6, y=5) before hitting air at y=4
        for z in 5..12 {
            for x in 5..12 {
                for y in 5..8 {
                    let sample = density.get_mut(x, y, z);
                    sample.density = 1.0;
                    sample.material = Material::Limestone;
                }
            }
        }

        let (descriptors, _seeds) = place_pools(&mut density, &config, Vec3::ZERO, 42, 100);
        assert!(
            descriptors.is_empty(),
            "Pools should NOT spawn on pillar tops with min_ground_depth=4 \
             (pillar only 3 voxels tall). Got {} pools.",
            descriptors.len()
        );
    }
}
