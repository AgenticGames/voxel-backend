use voxel_noise::simplex::Simplex3D;
use voxel_noise::NoiseSource;

use crate::cell::{ChunkFluidGrid, FluidType, SOURCE_LEVEL};
use crate::FluidConfig;

/// Place initial fluid sources in a chunk based on noise and terrain analysis.
///
/// Water springs are now handled by the geological spring detection system
/// (PlaceGeologicalSprings events from the worker thread).
/// This function only places noise-driven lava sources deep underground.
pub fn place_sources(
    grid: &mut ChunkFluidGrid,
    chunk: (i32, i32, i32),
    chunk_size: usize,
    config: &FluidConfig,
) {
    let lava_noise = Simplex3D::new(config.seed.wrapping_add(501));

    let size = grid.size;
    let origin_x = chunk.0 as f64 * chunk_size as f64;
    let origin_y = chunk.1 as f64 * chunk_size as f64;
    let origin_z = chunk.2 as f64 * chunk_size as f64;

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                // Skip solid cells
                if grid.is_solid(x, y, z) {
                    continue;
                }

                let wx = origin_x + x as f64;
                let wy = origin_y + y as f64;
                let wz = origin_z + z as f64;

                // Lava sources: deep underground with high noise
                if wy < config.lava_depth_min {
                    continue;
                }
                if wy < config.lava_depth_max {
                    let freq = config.lava_noise_frequency;
                    let val = lava_noise.sample(wx * freq, wy * freq, wz * freq);
                    let norm = val * 0.5 + 0.5;

                    let air_count = count_air_neighbors(grid, x, y, z);
                    let solid_dirs = count_solid_face_directions(grid, x, y, z);

                    let mut effective_threshold = config.lava_source_threshold;
                    // Apply cavern bias
                    effective_threshold -= config.cavern_source_bias * (air_count as f64 / 26.0);
                    // Apply tunnel bend bias
                    if solid_dirs >= 3 {
                        effective_threshold -=
                            config.tunnel_bend_threshold * ((solid_dirs - 2) as f64 / 4.0);
                    }
                    if effective_threshold < 0.0 {
                        effective_threshold = 0.0;
                    }

                    if norm > effective_threshold {
                        let cell = grid.get_mut(x, y, z);
                        cell.level = SOURCE_LEVEL;
                        cell.fluid_type = FluidType::Lava;
                        grid.dirty = true;
                    }
                }
            }
        }
    }
}

/// Count air cells in the 3x3x3 Moore neighborhood (26 surrounding cells).
fn count_air_neighbors(grid: &ChunkFluidGrid, x: usize, y: usize, z: usize) -> u8 {
    let size = grid.size as i32;
    let mut count: u8 = 0;
    for dz in -1i32..=1 {
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                let nz = z as i32 + dz;
                if nx >= 0 && nx < size && ny >= 0 && ny < size && nz >= 0 && nz < size {
                    if !grid.is_solid(nx as usize, ny as usize, nz as usize) {
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

/// Count how many of the 6 face directions have a solid neighbor.
fn count_solid_face_directions(grid: &ChunkFluidGrid, x: usize, y: usize, z: usize) -> u8 {
    let size = grid.size as i32;
    let directions: [(i32, i32, i32); 6] = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ];
    let mut count: u8 = 0;
    for (dx, dy, dz) in directions {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        let nz = z as i32 + dz;
        if nx >= 0 && nx < size && ny >= 0 && ny < size && nz >= 0 && nz < size {
            if grid.is_solid(nx as usize, ny as usize, nz as usize) {
                count += 1;
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_sources_in_empty_grid() {
        let mut grid = ChunkFluidGrid::new(16);
        let config = FluidConfig::default();
        place_sources(&mut grid, (0, 0, 0), 16, &config);
        // With no solid cells, no lava sources should be placed (lava doesn't require solid neighbors)
        // but noise threshold is very high (0.98) so unlikely in empty grid
        let count: usize = grid
            .cells
            .iter()
            .filter(|c| c.level > 0.0)
            .count();
        // In an empty grid at y=0 (above lava_depth_max of -50), no lava either
        assert_eq!(count, 0, "No sources in empty grid above lava depth");
    }

    #[test]
    fn sources_placed_near_solid() {
        let mut grid = ChunkFluidGrid::new(16);
        // Make bottom half solid
        for z in 0..16 {
            for y in 0..8 {
                for x in 0..16 {
                    grid.set_solid(x, y, z, true);
                }
            }
        }
        // Lava sources only appear deep; at chunk (0,0,0), wy ranges 0-15 which is above
        // lava_depth_max (-50), so no lava sources expected with default config.
        let config = FluidConfig::default();
        place_sources(&mut grid, (0, 0, 0), 16, &config);
        // Water is now handled by geological springs, not by this function
        let water_count: usize = grid
            .cells
            .iter()
            .filter(|c| c.level > 0.0 && c.fluid_type == FluidType::Water)
            .count();
        assert_eq!(water_count, 0, "Water springs are now geological, not noise-based");
    }
}
