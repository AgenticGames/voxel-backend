//! Geological spring detection — spring lines at contacts, vadose drips, aquifer breaches,
//! kimberlite pipe lava, hydrothermal springs, and artesian springs.

use voxel_core::density::DensityField;
use voxel_core::material::Material;
use voxel_noise::simplex::Simplex3D;
use voxel_noise::NoiseSource;

use crate::config::{
    ArtesianConfig, HydrothermalConfig, PipeLavaConfig, WaterTableConfig,
};
use crate::density::water_table_y_at;

/// Describes a water source to be placed by the fluid system.
#[derive(Debug, Clone)]
pub struct SpringDescriptor {
    pub lx: u8,
    pub ly: u8,
    pub lz: u8,
    pub level: f32,
    pub source_type: SpringType,
}

/// Classification of spring origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpringType {
    /// Geological contact between permeable and impermeable rock.
    SpringLine,
    /// Seepage through permeable ceiling below water table.
    VadoseDrip,
    /// Water released by mining into a confined aquifer zone.
    AquiferBreach,
    /// Source from an underground river passage.
    RiverSource,
    /// Artesian spring from a confined aquifer at depth.
    Artesian,
}

/// Describes a lava source from kimberlite pipes or lava tubes.
#[derive(Debug, Clone)]
pub struct LavaDescriptor {
    pub lx: u8,
    pub ly: u8,
    pub lz: u8,
    pub level: f32,
}

// ── Helper: 6 face-adjacent offsets ──

const FACE_OFFSETS: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

/// Returns true if the air cell at (x,y,z) has at least one solid face-neighbor.
/// Springs must touch a rock surface — no floating mid-cavern.
fn has_solid_face_neighbor(
    density: &DensityField,
    grid_size: usize,
    x: usize,
    y: usize,
    z: usize,
) -> bool {
    for (dx, dy, dz) in FACE_OFFSETS {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        let nz = z as i32 + dz;
        if nx < 0
            || ny < 0
            || nz < 0
            || nx >= grid_size as i32
            || ny >= grid_size as i32
            || nz >= grid_size as i32
        {
            continue;
        }
        if density.get(nx as usize, ny as usize, nz as usize).density > 0.0 {
            return true;
        }
    }
    false
}

/// Detect geological contact springs: air voxels below the water table where
/// permeable rock meets impermeable rock.
pub fn detect_spring_lines(
    density: &DensityField,
    chunk: (i32, i32, i32),
    chunk_size: usize,
    config: &WaterTableConfig,
    host_rock_config: &crate::config::HostRockConfig,
    seed: u64,
) -> Vec<SpringDescriptor> {
    if !config.enabled {
        return Vec::new();
    }

    let _ = host_rock_config; // available for future depth queries
    let grid_size = chunk_size + 1; // density grid is chunk_size+1
    let mut results = Vec::new();

    for z in 0..chunk_size {
        for y in 0..chunk_size {
            for x in 0..chunk_size {
                if results.len() >= config.max_springs_per_chunk as usize {
                    return results;
                }

                let sample = density.get(x, y, z);

                // Must be air
                if sample.density > 0.0 || sample.material != Material::Air {
                    continue;
                }

                // Must touch rock surface (no floating mid-cavern)
                if !has_solid_face_neighbor(density, grid_size, x, y, z) {
                    continue;
                }

                let wx = chunk.0 as f64 * chunk_size as f64 + x as f64;
                let wy = chunk.1 as f64 * chunk_size as f64 + y as f64;
                let wz = chunk.2 as f64 * chunk_size as f64 + z as f64;

                // Must be below water table
                let wt_y = water_table_y_at(wx, wz, config, seed);
                if wy > wt_y {
                    continue;
                }

                // Check 6 face-adjacent voxels for geological contact
                let mut has_permeable = false;
                let mut has_impermeable = false;
                let mut best_porosity: f32 = 0.0;

                for (dx, dy, dz) in FACE_OFFSETS {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;

                    if nx < 0 || ny < 0 || nz < 0
                        || nx >= grid_size as i32
                        || ny >= grid_size as i32
                        || nz >= grid_size as i32
                    {
                        continue;
                    }

                    let ns = density.get(nx as usize, ny as usize, nz as usize);

                    if ns.density <= 0.0 {
                        continue; // neighbor is air
                    }

                    if ns.material.is_permeable() {
                        has_permeable = true;
                        best_porosity = best_porosity.max(ns.material.porosity());
                    }
                    if ns.material.is_impermeable() {
                        has_impermeable = true;
                    }
                }

                // Geological contact: permeable meets impermeable
                if has_permeable && has_impermeable && best_porosity >= config.min_porosity_for_spring
                {
                    results.push(SpringDescriptor {
                        lx: x as u8,
                        ly: y as u8,
                        lz: z as u8,
                        level: config.spring_flow_rate,
                        source_type: SpringType::SpringLine,
                    });
                }
            }
        }
    }

    results
}

/// Detect vadose drips: air voxels below the water table with permeable rock ceiling
/// and solid rock mass above.
pub fn detect_vadose_drips(
    density: &DensityField,
    chunk: (i32, i32, i32),
    chunk_size: usize,
    config: &WaterTableConfig,
    seed: u64,
) -> Vec<SpringDescriptor> {
    if !config.enabled {
        return Vec::new();
    }

    let grid_size = chunk_size + 1;
    let drip_noise = Simplex3D::new(seed.wrapping_add(710));
    let mut results = Vec::new();

    for z in 0..chunk_size {
        for y in 0..chunk_size {
            for x in 0..chunk_size {
                if results.len() >= config.max_drips_per_chunk as usize {
                    return results;
                }

                let sample = density.get(x, y, z);

                // Must be air
                if sample.density > 0.0 || sample.material != Material::Air {
                    continue;
                }

                // Must touch rock surface (no floating mid-cavern)
                if !has_solid_face_neighbor(density, grid_size, x, y, z) {
                    continue;
                }

                let wx = chunk.0 as f64 * chunk_size as f64 + x as f64;
                let wy = chunk.1 as f64 * chunk_size as f64 + y as f64;
                let wz = chunk.2 as f64 * chunk_size as f64 + z as f64;

                // Must be below water table
                let wt_y = water_table_y_at(wx, wz, config, seed);
                if wy > wt_y {
                    continue;
                }

                // Check voxel directly above — must be solid AND permeable
                if y + 1 >= grid_size {
                    continue;
                }
                let above = density.get(x, y + 1, z);

                if above.density <= 0.0 || !above.material.is_permeable() {
                    continue;
                }

                // Check 2+ voxels above the ceiling — must also be solid (real rock mass)
                let mut solid_above = 0;
                for dy in 2..=3u32 {
                    if y + dy as usize >= grid_size {
                        break;
                    }
                    if density.get(x, y + dy as usize, z).density > 0.0 {
                        solid_above += 1;
                    }
                }
                if solid_above < 2 {
                    continue;
                }

                // Drip noise threshold
                let freq = config.drip_noise_frequency;
                let val = drip_noise.sample(wx * freq, wy * freq, wz * freq);
                let norm = val * 0.5 + 0.5;
                if norm < config.drip_noise_threshold {
                    continue;
                }

                results.push(SpringDescriptor {
                    lx: x as u8,
                    ly: y as u8,
                    lz: z as u8,
                    level: config.drip_level,
                    source_type: SpringType::VadoseDrip,
                });
            }
        }
    }

    results
}

/// Detect aquifer breaches after mining — springs near recently mined cells.
pub fn detect_aquifer_breaches(
    density: &DensityField,
    chunk: (i32, i32, i32),
    chunk_size: usize,
    config: &WaterTableConfig,
    host_rock_config: &crate::config::HostRockConfig,
    seed: u64,
    mined_cells: &[(usize, usize, usize)],
) -> Vec<SpringDescriptor> {
    if !config.enabled || mined_cells.is_empty() {
        return Vec::new();
    }

    let _ = host_rock_config;
    let grid_size = chunk_size + 1;
    let mut results = Vec::new();

    // For each air voxel within Manhattan distance 2 of a mined cell,
    // check for geological contact (same as spring lines).
    for z in 0..chunk_size {
        for y in 0..chunk_size {
            for x in 0..chunk_size {
                let sample = density.get(x, y, z);

                if sample.density > 0.0 || sample.material != Material::Air {
                    continue;
                }

                // Check Manhattan distance to any mined cell
                let near_mined = mined_cells.iter().any(|&(mx, my, mz)| {
                    let dist = (x as i32 - mx as i32).unsigned_abs()
                        + (y as i32 - my as i32).unsigned_abs()
                        + (z as i32 - mz as i32).unsigned_abs();
                    dist <= 2
                });
                if !near_mined {
                    continue;
                }

                let wx = chunk.0 as f64 * chunk_size as f64 + x as f64;
                let wy = chunk.1 as f64 * chunk_size as f64 + y as f64;
                let wz = chunk.2 as f64 * chunk_size as f64 + z as f64;

                let wt_y = water_table_y_at(wx, wz, config, seed);
                if wy > wt_y {
                    continue;
                }

                // Check for permeable + impermeable contact
                let mut has_permeable = false;
                let mut has_impermeable = false;
                let mut best_porosity: f32 = 0.0;

                for (dx, dy, dz) in FACE_OFFSETS {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;
                    if nx < 0 || ny < 0 || nz < 0
                        || nx >= grid_size as i32
                        || ny >= grid_size as i32
                        || nz >= grid_size as i32
                    {
                        continue;
                    }
                    let ns = density.get(nx as usize, ny as usize, nz as usize);
                    if ns.density <= 0.0 {
                        continue;
                    }
                    if ns.material.is_permeable() {
                        has_permeable = true;
                        best_porosity = best_porosity.max(ns.material.porosity());
                    }
                    if ns.material.is_impermeable() {
                        has_impermeable = true;
                    }
                }

                if has_permeable && has_impermeable && best_porosity >= config.min_porosity_for_spring
                {
                    results.push(SpringDescriptor {
                        lx: x as u8,
                        ly: y as u8,
                        lz: z as u8,
                        level: config.spring_flow_rate,
                        source_type: SpringType::AquiferBreach,
                    });
                }
            }
        }
    }

    results
}

/// Detect lava sources adjacent to kimberlite pipe voxels at depth.
pub fn detect_pipe_lava(
    density: &DensityField,
    chunk: (i32, i32, i32),
    chunk_size: usize,
    config: &PipeLavaConfig,
    seed: u64,
) -> Vec<LavaDescriptor> {
    if !config.enabled {
        return Vec::new();
    }

    let grid_size = chunk_size + 1;
    let lava_noise = Simplex3D::new(seed.wrapping_add(720));
    let mut results = Vec::new();

    // Kimberlite pipe bottom is around -200, activation_depth default -80
    let pipe_bottom = -200.0_f64;

    for z in 0..chunk_size {
        for y in 0..chunk_size {
            for x in 0..chunk_size {
                if results.len() >= config.max_lava_per_chunk as usize {
                    return results;
                }

                let sample = density.get(x, y, z);

                // Must be air
                if sample.density > 0.0 || sample.material != Material::Air {
                    continue;
                }

                let wy = chunk.1 as f64 * chunk_size as f64 + y as f64;

                // Must be below activation depth (too shallow = no magma)
                if wy > config.activation_depth {
                    continue;
                }

                // Check if any face-adjacent voxel is kimberlite
                let mut has_kimberlite = false;
                for (dx, dy, dz) in FACE_OFFSETS {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;
                    if nx < 0 || ny < 0 || nz < 0
                        || nx >= grid_size as i32
                        || ny >= grid_size as i32
                        || nz >= grid_size as i32
                    {
                        continue;
                    }
                    if density.get(nx as usize, ny as usize, nz as usize).material == Material::Kimberlite {
                        has_kimberlite = true;
                        break;
                    }
                }

                if !has_kimberlite {
                    continue;
                }

                // Depth-based probability: deeper = more lava
                let depth_factor = ((config.activation_depth - wy)
                    / (config.activation_depth - pipe_bottom))
                    .clamp(0.0, 1.0);
                let prob = depth_factor * config.depth_scaling;

                let wx = chunk.0 as f64 * chunk_size as f64 + x as f64;
                let wz = chunk.2 as f64 * chunk_size as f64 + z as f64;
                let noise_val = lava_noise.sample(wx * 0.1, wy * 0.1, wz * 0.1) * 0.5 + 0.5;

                if noise_val < (1.0 - prob) {
                    continue;
                }

                results.push(LavaDescriptor {
                    lx: x as u8,
                    ly: y as u8,
                    lz: z as u8,
                    level: 1.0,
                });
            }
        }
    }

    results
}

/// Detect hydrothermal springs near kimberlite pipes and basalt (lava tube walls).
pub fn detect_hydrothermal_springs(
    density: &DensityField,
    chunk: (i32, i32, i32),
    chunk_size: usize,
    water_config: &WaterTableConfig,
    hydro_config: &HydrothermalConfig,
    seed: u64,
) -> Vec<SpringDescriptor> {
    if !hydro_config.enabled || !water_config.enabled {
        return Vec::new();
    }

    let grid_size = chunk_size + 1;
    let search_r = hydro_config.radius as i32;
    let mut results = Vec::new();

    // First pass: find heat source positions (kimberlite or basalt) in this chunk
    let mut heat_sources: Vec<(usize, usize, usize)> = Vec::new();
    for z in 0..chunk_size {
        for y in 0..chunk_size {
            for x in 0..chunk_size {
                let mat = density.get(x, y, z).material;
                if mat == Material::Kimberlite || mat == Material::Basalt {
                    heat_sources.push((x, y, z));
                }
            }
        }
    }

    if heat_sources.is_empty() {
        return results;
    }

    // Second pass: find air voxels below water table near heat sources
    for z in 0..chunk_size {
        for y in 0..chunk_size {
            for x in 0..chunk_size {
                if results.len() >= hydro_config.max_per_chunk as usize {
                    return results;
                }

                let s = density.get(x, y, z);

                if s.density > 0.0 || s.material != Material::Air {
                    continue;
                }

                // Must touch rock surface (no floating mid-cavern)
                if !has_solid_face_neighbor(density, grid_size, x, y, z) {
                    continue;
                }

                let wx = chunk.0 as f64 * chunk_size as f64 + x as f64;
                let wy = chunk.1 as f64 * chunk_size as f64 + y as f64;
                let wz = chunk.2 as f64 * chunk_size as f64 + z as f64;

                // Must be below water table
                let wt_y = water_table_y_at(wx, wz, water_config, seed);
                if wy > wt_y {
                    continue;
                }

                // Check proximity to any heat source
                let near_heat = heat_sources.iter().any(|&(hx, hy, hz)| {
                    let dist = (x as i32 - hx as i32).abs()
                        + (y as i32 - hy as i32).abs()
                        + (z as i32 - hz as i32).abs();
                    dist <= search_r
                });
                if !near_heat {
                    continue;
                }

                // Must have at least one permeable neighbor (water pathway)
                let mut has_permeable = false;
                for (dx, dy, dz) in FACE_OFFSETS {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;
                    if nx < 0 || ny < 0 || nz < 0
                        || nx >= grid_size as i32
                        || ny >= grid_size as i32
                        || nz >= grid_size as i32
                    {
                        continue;
                    }
                    let ns = density.get(nx as usize, ny as usize, nz as usize);
                    if ns.density > 0.0 && ns.material.is_permeable() {
                        has_permeable = true;
                        break;
                    }
                }

                if !has_permeable {
                    continue;
                }

                results.push(SpringDescriptor {
                    lx: x as u8,
                    ly: y as u8,
                    lz: z as u8,
                    level: water_config.spring_flow_rate,
                    source_type: SpringType::SpringLine,
                });
            }
        }
    }

    results
}

/// Detect artesian springs from a confined aquifer lens at depth.
pub fn detect_artesian_springs(
    density: &DensityField,
    chunk: (i32, i32, i32),
    chunk_size: usize,
    config: &ArtesianConfig,
    seed: u64,
) -> Vec<SpringDescriptor> {
    if !config.enabled {
        return Vec::new();
    }

    let grid_size = chunk_size + 1;
    let pressure_noise = Simplex3D::new(seed.wrapping_add(730));
    let mut results = Vec::new();
    let mut site_count = 0u32;

    for z in 0..chunk_size {
        for y in 0..chunk_size {
            for x in 0..chunk_size {
                if site_count >= config.max_per_chunk as u32 {
                    return results;
                }

                let sample = density.get(x, y, z);

                // Must be air
                if sample.density > 0.0 || sample.material != Material::Air {
                    continue;
                }

                // Must touch rock surface (no floating mid-cavern)
                if !has_solid_face_neighbor(density, grid_size, x, y, z) {
                    continue;
                }

                let wy = chunk.1 as f64 * chunk_size as f64 + y as f64;

                // Must be at depth (granite/slate region)
                if wy > 20.0 {
                    continue;
                }

                // Check downward 1-5 voxels for sandstone (aquifer lens)
                let mut found_aquifer = false;
                for dy in 1..=5i32 {
                    if y as i32 - dy < 0 {
                        break;
                    }
                    let check_y = (y as i32 - dy) as usize;
                    let check_s = density.get(x, check_y, z);
                    if check_s.density > 0.0
                        && check_s.material == Material::Sandstone
                    {
                        // Verify impermeable rock above and below the sandstone
                        let mut has_cap = false;
                        let mut has_base = false;

                        // Check above aquifer (between air and sandstone — should be impermeable)
                        for dy2 in 1..dy {
                            let mid_y = (y as i32 - dy2) as usize;
                            let mid_s = density.get(x, mid_y, z);
                            if mid_s.density > 0.0
                                && mid_s.material.is_impermeable()
                            {
                                has_cap = true;
                                break;
                            }
                        }

                        // Check below aquifer
                        if check_y >= 1 {
                            let below_s = density.get(x, check_y - 1, z);
                            if below_s.density > 0.0
                                && below_s.material.is_impermeable()
                            {
                                has_base = true;
                            }
                        }

                        if has_cap && has_base {
                            found_aquifer = true;
                            break;
                        }
                    }
                }

                if !found_aquifer {
                    continue;
                }

                // Pressure noise determines spring intensity
                let wx = chunk.0 as f64 * chunk_size as f64 + x as f64;
                let wz = chunk.2 as f64 * chunk_size as f64 + z as f64;
                let pressure =
                    pressure_noise.sample(wx * config.pressure_noise_freq, 0.0, wz * config.pressure_noise_freq)
                        * 0.5
                        + 0.5;

                if pressure < 0.4 {
                    continue;
                }

                // Emit a 3x3x2 pocket of finite fluid around this site
                site_count += 1;
                for dz in -1..=1i32 {
                    for dy in 0..=1i32 {
                        for dx in -1..=1i32 {
                            let px = x as i32 + dx;
                            let py = y as i32 + dy;
                            let pz = z as i32 + dz;
                            if px < 0 || px >= chunk_size as i32
                                || py < 0 || py >= chunk_size as i32
                                || pz < 0 || pz >= chunk_size as i32
                            {
                                continue;
                            }
                            results.push(SpringDescriptor {
                                lx: px as u8,
                                ly: py as u8,
                                lz: pz as u8,
                                level: 0.9,
                                source_type: SpringType::Artesian,
                            });
                        }
                    }
                }
            }
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::density::DensityField;

    fn make_test_density(chunk_size: usize) -> DensityField {
        let grid_size = chunk_size + 1;
        // All solid granite by default
        let mut density = DensityField::new(grid_size);
        for s in &mut density.samples {
            s.density = 1.0;
            s.material = Material::Granite;
        }
        density
    }

    fn set_voxel(
        density: &mut DensityField,
        _grid_size: usize,
        x: usize,
        y: usize,
        z: usize,
        d: f32,
        mat: Material,
    ) {
        let s = density.get_mut(x, y, z);
        s.density = d;
        s.material = mat;
    }

    #[test]
    fn no_springs_above_water_table() {
        let chunk_size = 16;
        let grid_size = chunk_size + 1;
        let mut density = make_test_density(chunk_size);
        // Place air voxel at y=15 with limestone neighbor and granite neighbor
        set_voxel(&mut density, grid_size, 8, 15, 8, -1.0, Material::Air);
        set_voxel(&mut density, grid_size, 7, 15, 8, 1.0, Material::Limestone);
        set_voxel(&mut density, grid_size, 9, 15, 8, 1.0, Material::Granite);

        // Chunk at Y=100 → world y = 100*16 + 15 = 1615 → far above water table (170)
        let config = WaterTableConfig { enabled: true, ..WaterTableConfig::default() };
        let host = crate::config::HostRockConfig::default();
        let result = detect_spring_lines(&density, (0, 100, 0), chunk_size, &config, &host, 42);
        assert!(result.is_empty(), "No springs above water table");
    }

    #[test]
    fn spring_at_geological_contact() {
        let chunk_size = 16;
        let grid_size = chunk_size + 1;
        let mut density = make_test_density(chunk_size);
        // Air voxel at position (8, 8, 8)
        set_voxel(&mut density, grid_size, 8, 8, 8, -1.0, Material::Air);
        // Limestone neighbor (permeable)
        set_voxel(&mut density, grid_size, 7, 8, 8, 1.0, Material::Limestone);
        // Granite neighbor (impermeable)
        set_voxel(&mut density, grid_size, 9, 8, 8, 1.0, Material::Granite);

        // Chunk at Y=0 → world y = 8 → below water table (170 default)
        let config = WaterTableConfig { enabled: true, ..WaterTableConfig::default() };
        let host = crate::config::HostRockConfig::default();
        let result = detect_spring_lines(&density, (0, 0, 0), chunk_size, &config, &host, 42);
        assert!(!result.is_empty(), "Should detect spring at contact");
        assert_eq!(result[0].source_type, SpringType::SpringLine);
    }

    #[test]
    fn no_spring_without_impermeable() {
        let chunk_size = 16;
        let grid_size = chunk_size + 1;
        let mut density = make_test_density(chunk_size);
        set_voxel(&mut density, grid_size, 8, 8, 8, -1.0, Material::Air);
        // Only limestone neighbors (permeable), no impermeable
        set_voxel(&mut density, grid_size, 7, 8, 8, 1.0, Material::Limestone);
        set_voxel(&mut density, grid_size, 9, 8, 8, 1.0, Material::Limestone);
        // Make the granite neighbors into limestone
        set_voxel(&mut density, grid_size, 8, 7, 8, 1.0, Material::Limestone);
        set_voxel(&mut density, grid_size, 8, 9, 8, 1.0, Material::Limestone);
        set_voxel(&mut density, grid_size, 8, 8, 7, 1.0, Material::Limestone);
        set_voxel(&mut density, grid_size, 8, 8, 9, 1.0, Material::Limestone);

        let config = WaterTableConfig { enabled: true, ..WaterTableConfig::default() };
        let host = crate::config::HostRockConfig::default();
        let result = detect_spring_lines(&density, (0, 0, 0), chunk_size, &config, &host, 42);
        assert!(result.is_empty(), "No spring without impermeable neighbor");
    }

    #[test]
    fn drip_below_permeable_ceiling() {
        let chunk_size = 16;
        let grid_size = chunk_size + 1;
        let mut density = make_test_density(chunk_size);
        // Air voxel
        set_voxel(&mut density, grid_size, 8, 5, 8, -1.0, Material::Air);
        // Permeable ceiling (sandstone)
        set_voxel(&mut density, grid_size, 8, 6, 8, 1.0, Material::Sandstone);
        // Solid rock mass above (granite at y=7 and y=8)
        set_voxel(&mut density, grid_size, 8, 7, 8, 1.0, Material::Granite);
        set_voxel(&mut density, grid_size, 8, 8, 8, 1.0, Material::Granite);

        // Use very low drip threshold to guarantee drips
        let config = WaterTableConfig {
            enabled: true,
            drip_noise_threshold: 0.0, // accept all
            ..WaterTableConfig::default()
        };
        let result = detect_vadose_drips(&density, (0, 0, 0), chunk_size, &config, 42);
        assert!(!result.is_empty(), "Should detect drip below permeable ceiling");
        assert_eq!(result[0].source_type, SpringType::VadoseDrip);
    }

    #[test]
    fn no_spring_floating_in_air() {
        // Air cell at depth with sandstone aquifer below but NO solid face-neighbors
        // should NOT produce an artesian spring (floating mid-cavern).
        let chunk_size = 16;
        let grid_size = chunk_size + 1;
        let mut density = make_test_density(chunk_size);

        // Carve a large air pocket (8,4,8) through (8,8,8) — all air, no solid neighbors for (8,6,8)
        for y in 4..=8 {
            set_voxel(&mut density, grid_size, 8, y, 8, -1.0, Material::Air);
        }
        // Also clear horizontal neighbors of (8,6,8) so it has 0 solid face-neighbors
        set_voxel(&mut density, grid_size, 7, 6, 8, -1.0, Material::Air);
        set_voxel(&mut density, grid_size, 9, 6, 8, -1.0, Material::Air);
        set_voxel(&mut density, grid_size, 8, 6, 7, -1.0, Material::Air);
        set_voxel(&mut density, grid_size, 8, 6, 9, -1.0, Material::Air);
        // Also clear horizontal neighbors of (8,5,8) so its 3x3x2 pocket can't reach y=6
        set_voxel(&mut density, grid_size, 7, 5, 8, -1.0, Material::Air);
        set_voxel(&mut density, grid_size, 9, 5, 8, -1.0, Material::Air);
        set_voxel(&mut density, grid_size, 8, 5, 7, -1.0, Material::Air);
        set_voxel(&mut density, grid_size, 8, 5, 9, -1.0, Material::Air);

        // Place sandstone aquifer below with impermeable cap and base
        // y=3 is granite (impermeable cap), y=2 is sandstone (aquifer), y=1 is granite (base)
        set_voxel(&mut density, grid_size, 8, 3, 8, 1.0, Material::Granite);
        set_voxel(&mut density, grid_size, 8, 2, 8, 1.0, Material::Sandstone);
        set_voxel(&mut density, grid_size, 8, 1, 8, 1.0, Material::Granite);

        // Chunk at Y=-2 → world y = -2*16 + 6 = -26 → below depth threshold (20.0)
        let config = crate::config::ArtesianConfig {
            enabled: true,
            max_per_chunk: 10,
            pressure_noise_freq: 0.0, // constant noise → will pass threshold
            ..crate::config::ArtesianConfig::default()
        };
        let result = detect_artesian_springs(&density, (0, -2, 0), chunk_size, &config, 42);
        // (8,6,8) has no solid face-neighbors → should be rejected
        let has_floating = result.iter().any(|s| s.ly == 6);
        assert!(!has_floating, "No artesian spring should float mid-air without solid neighbor");
    }

    #[test]
    fn budget_cap_enforced() {
        let chunk_size = 16;
        let grid_size = chunk_size + 1;
        let mut density = make_test_density(chunk_size);
        // Create many valid spring sites
        for x in 0..16 {
            set_voxel(&mut density, grid_size, x, 8, 8, -1.0, Material::Air);
            if x > 0 {
                set_voxel(&mut density, grid_size, x - 1, 8, 8, 1.0, Material::Limestone);
            }
        }
        // Make sure there's impermeable rock too
        for x in 0..16 {
            set_voxel(&mut density, grid_size, x, 7, 8, 1.0, Material::Granite);
        }

        let config = WaterTableConfig {
            enabled: true,
            max_springs_per_chunk: 3,
            ..WaterTableConfig::default()
        };
        let host = crate::config::HostRockConfig::default();
        let result = detect_spring_lines(&density, (0, 0, 0), chunk_size, &config, &host, 42);
        assert!(result.len() <= 3, "Budget cap should limit springs to 3");
    }

    #[test]
    fn artesian_pocket_is_finite_cluster() {
        // Artesian springs should emit >1 descriptor per site (3x3x2 pocket)
        // and all descriptors should have level < 1.0 (finite, not source)
        let chunk_size = 16;
        let grid_size = chunk_size + 1;
        let mut density = make_test_density(chunk_size);

        // Air voxel at (8, 5, 8) with solid face-neighbor
        set_voxel(&mut density, grid_size, 8, 5, 8, -1.0, Material::Air);
        // Solid neighbor (so it's not floating)
        set_voxel(&mut density, grid_size, 7, 5, 8, 1.0, Material::Granite);

        // Aquifer structure: granite cap at y=3, sandstone at y=2, granite base at y=1
        set_voxel(&mut density, grid_size, 8, 3, 8, 1.0, Material::Granite);
        set_voxel(&mut density, grid_size, 8, 2, 8, 1.0, Material::Sandstone);
        set_voxel(&mut density, grid_size, 8, 1, 8, 1.0, Material::Granite);

        // Chunk at Y=-2 → world y = -2*16 + 5 = -27 → below depth threshold (20.0)
        let config = crate::config::ArtesianConfig {
            enabled: true,
            max_per_chunk: 10,
            pressure_noise_freq: 0.0, // constant noise → will pass threshold
            ..crate::config::ArtesianConfig::default()
        };
        let result = detect_artesian_springs(&density, (0, -2, 0), chunk_size, &config, 42);
        // Should have >1 descriptor (3x3x2 pocket = up to 18 descriptors)
        assert!(result.len() > 1,
            "Artesian pocket should emit >1 descriptor, got {}", result.len());
        // All should be finite (level < 1.0)
        for s in &result {
            assert!(s.level < 1.0,
                "Artesian descriptor level={} should be < 1.0 (finite)", s.level);
        }
    }
}
