use std::collections::HashMap;

use crate::cell::{ChunkFluidGrid, FluidType, MAX_LEVEL, MIN_LEVEL, SOURCE_LEVEL};
use crate::FluidConfig;

/// Simulate one tick of fluid for all loaded chunks.
///
/// Uses double-buffering: reads from current state, writes to a new buffer,
/// then swaps. Gravity flows downward first, then horizontal spread.
///
/// Returns the set of chunk keys that had any fluid changes (dirty).
pub fn tick_fluid(
    chunks: &mut HashMap<(i32, i32, i32), ChunkFluidGrid>,
    chunk_size: usize,
    is_lava_tick: bool,
    config: &FluidConfig,
) -> Vec<(i32, i32, i32)> {
    let mut dirty = Vec::new();

    // Collect keys to iterate
    let keys: Vec<(i32, i32, i32)> = chunks.keys().copied().collect();

    for key in keys {
        let changed = tick_chunk(chunks, key, chunk_size, is_lava_tick, config);
        if changed {
            dirty.push(key);
        }
    }

    dirty
}

/// Count how many of the 6 face neighbors are solid (or out of bounds).
fn count_solid_face_neighbors(solid_mask: &[u64], size: usize, x: usize, y: usize, z: usize) -> u8 {
    let mut count: u8 = 0;
    let deltas: [(i32, i32, i32); 6] = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ];
    for (dx, dy, dz) in deltas {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        let nz = z as i32 + dz;
        if nx < 0 || nx >= size as i32 || ny < 0 || ny >= size as i32 || nz < 0 || nz >= size as i32 {
            count += 1; // out of bounds = solid
        } else {
            let ni = nz as usize * size * size + ny as usize * size + nx as usize;
            if (solid_mask[ni / 64] >> (ni % 64)) & 1 == 1 {
                count += 1;
            }
        }
    }
    count
}

/// Simulate one tick for a single chunk. Returns true if any cell changed.
fn tick_chunk(
    chunks: &mut HashMap<(i32, i32, i32), ChunkFluidGrid>,
    key: (i32, i32, i32),
    _chunk_size: usize,
    is_lava_tick: bool,
    config: &FluidConfig,
) -> bool {
    let grid = match chunks.get(&key) {
        Some(g) => g,
        None => return false,
    };

    let size = grid.size;

    // Create new buffer
    let mut new_cells = grid.cells.clone();
    let solid_mask = grid.solid_mask.clone();
    let mut changed = false;

    // Process each cell
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = z * size * size + y * size + x;

                // Skip solid cells
                let word = idx / 64;
                let bit = idx % 64;
                if (solid_mask[word] >> bit) & 1 == 1 {
                    new_cells[idx].level = 0.0;
                    continue;
                }

                let cell = &grid.cells[idx];
                if cell.level < MIN_LEVEL {
                    continue;
                }

                // Check fluid type vs tick type
                let is_lava = cell.fluid_type == FluidType::Lava;
                if is_lava && !is_lava_tick {
                    continue;
                }
                if !is_lava && is_lava_tick {
                    continue;
                }

                // Skip flow for sources trapped in solid rock pockets
                let solid_neighbors = count_solid_face_neighbors(&solid_mask, size, x, y, z);
                if solid_neighbors >= 5 {
                    continue;
                }

                let is_source = cell.is_source();

                let flow_rate = if is_lava { config.lava_flow_rate } else { config.water_flow_rate };
                let horizontal_spread = if is_lava { config.lava_spread_rate } else { config.water_spread_rate };

                // Gravity: try to flow down
                if y > 0 {
                    let below_idx = z * size * size + (y - 1) * size + x;
                    let below_solid = (solid_mask[below_idx / 64] >> (below_idx % 64)) & 1 == 1;
                    if !below_solid {
                        let below_space = MAX_LEVEL - new_cells[below_idx].level;
                        if below_space > MIN_LEVEL {
                            let transfer = cell.level.min(below_space).min(flow_rate * 2.0);
                            if transfer > MIN_LEVEL {
                                // Sources emit fluid without losing any
                                if !is_source {
                                    new_cells[idx].level -= transfer;
                                }
                                new_cells[below_idx].level += transfer;
                                new_cells[below_idx].fluid_type = cell.fluid_type;
                                changed = true;
                            }
                        }
                    }
                }
                // Cross-chunk downward flow: y==0 means neighbor chunk below
                else {
                    let below_key = (key.0, key.1 - 1, key.2);
                    if let Some(below_grid) = chunks.get(&below_key) {
                        let by = size - 1;
                        let below_idx = z * size * size + by * size + x;
                        let below_solid = {
                            let w = below_idx / 64;
                            let b = below_idx % 64;
                            (below_grid.solid_mask[w] >> b) & 1 == 1
                        };
                        if !below_solid {
                            let below_space = MAX_LEVEL - below_grid.cells[below_idx].level;
                            if below_space > MIN_LEVEL {
                                let transfer = new_cells[idx].level.min(below_space).min(flow_rate * 2.0);
                                if transfer > MIN_LEVEL {
                                    if !is_source {
                                        new_cells[idx].level -= transfer;
                                    }
                                    changed = true;
                                    // Note: we don't modify the neighbor chunk here to avoid
                                    // borrow issues. Cross-chunk flow is approximate.
                                }
                            }
                        }
                    }
                }

                // Horizontal spread (4 neighbors: +x, -x, +z, -z)
                if new_cells[idx].level > MIN_LEVEL {
                    let remaining = new_cells[idx].level;
                    let neighbors: [(i32, i32, i32); 4] = [
                        (x as i32 + 1, y as i32, z as i32),
                        (x as i32 - 1, y as i32, z as i32),
                        (x as i32, y as i32, z as i32 + 1),
                        (x as i32, y as i32, z as i32 - 1),
                    ];

                    for (nx, ny, nz) in neighbors {
                        if nx < 0 || nx >= size as i32 || ny < 0 || ny >= size as i32 || nz < 0 || nz >= size as i32 {
                            continue;
                        }
                        let ni = nz as usize * size * size + ny as usize * size + nx as usize;
                        let n_solid = (solid_mask[ni / 64] >> (ni % 64)) & 1 == 1;
                        if n_solid {
                            continue;
                        }
                        let n_level = new_cells[ni].level;
                        if remaining > n_level + MIN_LEVEL {
                            let diff = remaining - n_level;
                            let transfer = (diff * horizontal_spread).min(flow_rate);
                            if transfer > MIN_LEVEL {
                                // Sources emit fluid without losing any
                                if !is_source {
                                    new_cells[idx].level -= transfer;
                                }
                                new_cells[ni].level += transfer;
                                new_cells[ni].fluid_type = cell.fluid_type;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
    }

    // Clean up tiny residual amounts
    for cell in &mut new_cells {
        if cell.level < MIN_LEVEL && cell.level > 0.0 {
            cell.level = 0.0;
        }
    }

    // Swap buffer
    if changed {
        if let Some(grid) = chunks.get_mut(&key) {
            grid.cells = new_cells;
            grid.dirty = true;
        }
    }

    changed
}

/// Detect water-lava contact and return cells to solidify (become basalt).
///
/// A water cell adjacent to a lava source becomes basalt (obsidian-like).
/// Returns list of (chunk_key, x, y, z) positions to solidify.
pub fn detect_solidification(
    chunks: &HashMap<(i32, i32, i32), ChunkFluidGrid>,
) -> Vec<((i32, i32, i32), usize, usize, usize)> {
    let mut solidify = Vec::new();

    for (&key, grid) in chunks {
        let size = grid.size;
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let cell = grid.get(x, y, z);
                    if cell.level < MIN_LEVEL {
                        continue;
                    }

                    // Check if this is a lava cell adjacent to water (or vice versa)
                    let is_lava = cell.fluid_type == FluidType::Lava;
                    let neighbors: [(i32, i32, i32); 6] = [
                        (x as i32 + 1, y as i32, z as i32),
                        (x as i32 - 1, y as i32, z as i32),
                        (x as i32, y as i32 + 1, z as i32),
                        (x as i32, y as i32 - 1, z as i32),
                        (x as i32, y as i32, z as i32 + 1),
                        (x as i32, y as i32, z as i32 - 1),
                    ];

                    for (nx, ny, nz) in neighbors {
                        if nx < 0 || nx >= size as i32 || ny < 0 || ny >= size as i32 || nz < 0 || nz >= size as i32 {
                            continue;
                        }
                        let n = grid.get(nx as usize, ny as usize, nz as usize);
                        if n.level < MIN_LEVEL {
                            continue;
                        }
                        let n_is_lava = n.fluid_type == FluidType::Lava;
                        if is_lava && !n_is_lava {
                            // Lava touched water: lava solidifies
                            solidify.push((key, x, y, z));
                            break;
                        }
                    }
                }
            }
        }
    }

    solidify
}

/// Regenerate source blocks: source cells always maintain SOURCE_LEVEL.
pub fn regen_sources(chunks: &mut HashMap<(i32, i32, i32), ChunkFluidGrid>) {
    for grid in chunks.values_mut() {
        for cell in &mut grid.cells {
            if cell.is_source() {
                cell.level = SOURCE_LEVEL;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(size: usize) -> ChunkFluidGrid {
        ChunkFluidGrid::new(size)
    }

    #[test]
    fn gravity_flows_down() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);
        // Place water at y=8 (non-source level so it can flow)
        grid.get_mut(8, 8, 8).level = 0.8;
        grid.get_mut(8, 8, 8).fluid_type = FluidType::Water;
        chunks.insert(key, grid);

        let config = crate::FluidConfig::default();
        // Single tick should move water one step down
        tick_fluid(&mut chunks, 16, false, &config);

        // Check that water exists anywhere below y=8
        let grid = &chunks[&key];
        let mut found_below = false;
        for y in 0..8 {
            if grid.get(8, y, 8).level > 0.0 {
                found_below = true;
                break;
            }
        }
        assert!(found_below, "Water should flow down from y=8 after one tick");
    }

    #[test]
    fn source_regenerates() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);
        grid.get_mut(8, 8, 8).level = SOURCE_LEVEL;
        grid.get_mut(8, 8, 8).fluid_type = FluidType::Water;
        chunks.insert(key, grid);

        let config = crate::FluidConfig::default();
        // Tick
        tick_fluid(&mut chunks, 16, false, &config);

        // Source should still be at SOURCE_LEVEL
        regen_sources(&mut chunks);
        let grid = &chunks[&key];
        assert_eq!(grid.get(8, 8, 8).level, SOURCE_LEVEL);
    }

    #[test]
    fn solid_blocks_flow() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);
        // Place water at y=8
        grid.get_mut(8, 8, 8).level = 0.5;
        grid.get_mut(8, 8, 8).fluid_type = FluidType::Water;
        // Make y=7 solid
        grid.set_solid(8, 7, 8, true);
        chunks.insert(key, grid);

        let config = crate::FluidConfig::default();
        tick_fluid(&mut chunks, 16, false, &config);

        let grid = &chunks[&key];
        assert_eq!(grid.get(8, 7, 8).level, 0.0, "Water should not enter solid cell");
    }

    #[test]
    fn lava_water_solidification() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);

        // Lava at (8,8,8)
        grid.get_mut(8, 8, 8).level = 0.5;
        grid.get_mut(8, 8, 8).fluid_type = FluidType::Lava;

        // Water at (9,8,8)
        grid.get_mut(9, 8, 8).level = 0.5;
        grid.get_mut(9, 8, 8).fluid_type = FluidType::Water;

        chunks.insert(key, grid);

        let solidify = detect_solidification(&chunks);
        assert!(!solidify.is_empty(), "Should detect solidification");
    }

    #[test]
    fn contained_source_doesnt_flow() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);
        // Place water source at (8,8,8)
        grid.get_mut(8, 8, 8).level = SOURCE_LEVEL;
        grid.get_mut(8, 8, 8).fluid_type = FluidType::Water;
        // Surround with solid on all 6 faces
        grid.set_solid(7, 8, 8, true);
        grid.set_solid(9, 8, 8, true);
        grid.set_solid(8, 7, 8, true);
        grid.set_solid(8, 9, 8, true);
        grid.set_solid(8, 8, 7, true);
        grid.set_solid(8, 8, 9, true);
        chunks.insert(key, grid);

        let config = crate::FluidConfig::default();
        // Tick several times
        for _ in 0..10 {
            tick_fluid(&mut chunks, 16, false, &config);
        }

        // No fluid should exist outside the source cell
        let grid = &chunks[&key];
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    if (x, y, z) == (8, 8, 8) { continue; }
                    assert!(grid.get(x, y, z).level < 0.001,
                        "Fluid leaked to ({},{},{}) with level {}", x, y, z, grid.get(x, y, z).level);
                }
            }
        }
    }
}
