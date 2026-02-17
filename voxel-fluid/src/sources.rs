use voxel_noise::simplex::Simplex3D;
use voxel_noise::NoiseSource;

use crate::cell::{ChunkFluidGrid, FluidType, SOURCE_LEVEL};
use crate::FluidConfig;

/// Place initial fluid sources in a chunk based on noise and terrain analysis.
///
/// Water springs: placed where air meets solid with high noise value.
/// Lava sources: placed deep underground with high noise value.
/// Finite water pockets: placed in enclosed air spaces.
pub fn place_sources(
    grid: &mut ChunkFluidGrid,
    chunk: (i32, i32, i32),
    chunk_size: usize,
    config: &FluidConfig,
) {
    let water_noise = Simplex3D::new(config.seed.wrapping_add(500));
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

                // Water springs: air cell adjacent to solid with high noise
                if has_solid_neighbor(grid, x, y, z) {
                    let freq = 0.05;
                    let val = water_noise.sample(wx * freq, wy * freq, wz * freq);
                    let norm = val * 0.5 + 0.5;
                    if norm > config.water_spring_threshold {
                        let cell = grid.get_mut(x, y, z);
                        cell.level = SOURCE_LEVEL;
                        cell.fluid_type = FluidType::Water;
                        grid.dirty = true;
                        continue;
                    }
                }

                // Lava sources: deep underground with high noise
                if wy < config.lava_depth_max {
                    let freq = 0.03;
                    let val = lava_noise.sample(wx * freq, wy * freq, wz * freq);
                    let norm = val * 0.5 + 0.5;
                    if norm > config.lava_source_threshold {
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

/// Check if a cell has any solid face-adjacent neighbor.
fn has_solid_neighbor(grid: &ChunkFluidGrid, x: usize, y: usize, z: usize) -> bool {
    let size = grid.size;
    let neighbors: [(i32, i32, i32); 6] = [
        (x as i32 + 1, y as i32, z as i32),
        (x as i32 - 1, y as i32, z as i32),
        (x as i32, y as i32 + 1, z as i32),
        (x as i32, y as i32 - 1, z as i32),
        (x as i32, y as i32, z as i32 + 1),
        (x as i32, y as i32, z as i32 - 1),
    ];

    for (nx, ny, nz) in neighbors {
        if nx >= 0 && nx < size as i32 && ny >= 0 && ny < size as i32 && nz >= 0 && nz < size as i32 {
            if grid.is_solid(nx as usize, ny as usize, nz as usize) {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_sources_in_empty_grid() {
        let mut grid = ChunkFluidGrid::new(16);
        let config = FluidConfig::default();
        place_sources(&mut grid, (0, 0, 0), 16, &config);
        // With no solid cells, no water springs should be placed
        let water_count: usize = grid.cells.iter().filter(|c| c.level > 0.0 && c.fluid_type == FluidType::Water).count();
        assert_eq!(water_count, 0, "No water springs without solid neighbors");
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
        // Use very low threshold so sources are abundant
        let config = FluidConfig {
            water_spring_threshold: 0.01,
            ..FluidConfig::default()
        };
        place_sources(&mut grid, (0, 0, 0), 16, &config);
        let water_count: usize = grid.cells.iter().filter(|c| c.level > 0.0).count();
        assert!(water_count > 0, "Should place some water sources near solid surface");
    }
}
