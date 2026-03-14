use std::collections::{HashMap, HashSet};

use crate::cell::{ChunkFluidGrid, MIN_LEVEL, SOURCE_LEVEL};

/// After a density update, squeeze excess fluid from cells whose capacity decreased.
/// Excess is pushed to non-solid neighbors; any remainder is evaporated.
pub fn squeeze_excess_fluid(grid: &mut ChunkFluidGrid) {
    let size = grid.size;
    let mut any_change = false;

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = z * size * size + y * size + x;
                let capacity = grid.cell_cap[idx];
                let level = grid.cells[idx].level;

                if level <= capacity {
                    continue;
                }

                let excess = level - capacity;
                grid.cells[idx].level = capacity;
                any_change = true;

                // Try to push excess to neighbors
                let mut remaining = excess;
                let fluid_type = grid.cells[idx].fluid_type;
                let deltas: [(i32, i32, i32); 6] = [
                    (0, 1, 0), (0, -1, 0), // up/down first
                    (1, 0, 0), (-1, 0, 0),
                    (0, 0, 1), (0, 0, -1),
                ];
                for (dx, dy, dz) in deltas {
                    if remaining < MIN_LEVEL {
                        break;
                    }
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;
                    if nx < 0 || nx >= size as i32 || ny < 0 || ny >= size as i32
                        || nz < 0 || nz >= size as i32
                    {
                        continue;
                    }
                    let ni = nz as usize * size * size + ny as usize * size + nx as usize;
                    let n_capacity = grid.cell_cap[ni];
                    let n_space = (n_capacity - grid.cells[ni].level).max(0.0);
                    if n_space > MIN_LEVEL {
                        let push = remaining.min(n_space);
                        grid.cells[ni].level += push;
                        grid.cells[ni].fluid_type = fluid_type;
                        remaining -= push;
                    }
                }
                // Any remaining excess evaporates (no neighbor space available)
            }
        }
    }

    if any_change {
        grid.dirty = true;
    }
}

/// Equalize water levels across connected horizontal regions at each Y level.
///
/// For each Y layer, flood-fills connected water cells across chunk boundaries
/// and averages their levels. This provides instant long-range communication —
/// the "suction" effect that pulls water toward lower elevation openings.
///
/// Returns the set of dirty chunk keys.
pub fn equalize_horizontal(
    chunks: &mut HashMap<(i32, i32, i32), ChunkFluidGrid>,
    chunk_size: usize,
    is_lava: bool,
) -> HashSet<(i32, i32, i32)> {
    let mut dirty = HashSet::new();

    // Build global index: collect all water cells with level > MIN_LEVEL
    // Key: (world_x, world_y, world_z) → (chunk_key, local_x, local_y, local_z, level, capacity)
    let mut water_cells: HashMap<(i32, i32, i32), ((i32, i32, i32), usize, usize, usize, f32, f32)> = HashMap::new();

    // Determine the Y range across all chunks
    let mut min_world_y = i32::MAX;
    let mut max_world_y = i32::MIN;

    for (&chunk_key, grid) in chunks.iter() {
        if !grid.has_fluid {
            continue;
        }
        let size = grid.size;
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let cell = grid.get(x, y, z);
                    if cell.level < MIN_LEVEL {
                        continue;
                    }
                    if cell.fluid_type.is_lava() != is_lava {
                        continue;
                    }
                    // Skip source cells — they maintain their own level
                    if cell.is_source() {
                        continue;
                    }
                    let cap = grid.cell_capacity(x, y, z);
                    if cap < MIN_LEVEL {
                        continue;
                    }
                    let wx = chunk_key.0 * chunk_size as i32 + x as i32;
                    let wy = chunk_key.1 * chunk_size as i32 + y as i32;
                    let wz = chunk_key.2 * chunk_size as i32 + z as i32;
                    water_cells.insert((wx, wy, wz), (chunk_key, x, y, z, cell.level, cap));
                    min_world_y = min_world_y.min(wy);
                    max_world_y = max_world_y.max(wy);
                }
            }
        }
    }

    if water_cells.is_empty() {
        return dirty;
    }

    // For each Y level, flood-fill connected regions on XZ plane and average levels
    let mut visited: HashSet<(i32, i32, i32)> = HashSet::new();

    for wy in min_world_y..=max_world_y {
        // Find all unvisited water cells at this Y
        let cells_at_y: Vec<(i32, i32, i32)> = water_cells.keys()
            .filter(|&&(_, y, _)| y == wy)
            .copied()
            .collect();

        for start in cells_at_y {
            if visited.contains(&start) {
                continue;
            }

            // BFS flood-fill on XZ plane at this Y level
            let mut region: Vec<(i32, i32, i32)> = Vec::new();
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(start);
            visited.insert(start);

            while let Some(pos) = queue.pop_front() {
                region.push(pos);
                // 4-connected on XZ plane
                for &(dx, dz) in &[(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                    let neighbor = (pos.0 + dx, pos.1, pos.2 + dz);
                    if !visited.contains(&neighbor) && water_cells.contains_key(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }

            if region.len() < 2 {
                continue; // single cell, nothing to equalize
            }

            // Compute total water and cell count, then average
            let mut total_water = 0.0f32;
            let mut total_cap = 0.0f32;
            for &pos in &region {
                let (_, _, _, _, level, cap) = water_cells[&pos];
                total_water += level;
                total_cap += cap;
            }

            // Don't equalize if total capacity is near zero
            if total_cap < MIN_LEVEL {
                continue;
            }

            // Damped equalization: blend toward the average rather than snapping.
            // This preserves flow gradients toward drains while still leveling pools.
            let avg_fill = total_water / total_cap;
            const EQ_DAMPING: f32 = 0.3; // blend 30% toward average each tick

            for &pos in &region {
                let (chunk_key, lx, ly, lz, old_level, cap) = water_cells[&pos];
                let target = (avg_fill * cap).min(cap);
                let new_level = old_level + EQ_DAMPING * (target - old_level);
                if (new_level - old_level).abs() > MIN_LEVEL {
                    if let Some(grid) = chunks.get_mut(&chunk_key) {
                        let cell = grid.get_mut(lx, ly, lz);
                        cell.level = new_level;
                        grid.dirty = true;
                    }
                    dirty.insert(chunk_key);
                }
            }
        }
    }

    dirty
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
        if !grid.has_fluid {
            continue;
        }
        let size = grid.size;
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let cell = grid.get(x, y, z);
                    if cell.level < MIN_LEVEL {
                        continue;
                    }

                    // Check if this is a lava cell adjacent to water (or vice versa)
                    let is_lava = cell.fluid_type.is_lava();
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
                        let n_is_lava = n.fluid_type.is_lava();
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
    use std::collections::HashMap;
    use crate::cell::{ChunkDensityCache, ChunkFluidGrid, FluidType, SOURCE_LEVEL};

    fn make_chunk(size: usize) -> ChunkFluidGrid {
        ChunkFluidGrid::new(size)
    }

    fn empty_density_cache() -> HashMap<(i32, i32, i32), ChunkDensityCache> {
        HashMap::new()
    }

    #[test]
    fn squeeze_excess_works() {
        let mut grid = make_chunk(16);
        grid.get_mut(8, 8, 8).level = 0.8;
        grid.get_mut(8, 8, 8).fluid_type = FluidType::Water;
        grid.set_density(8, 8, 8, 0.5);
        squeeze_excess_fluid(&mut grid);

        assert!(grid.get(8, 8, 8).level <= 0.001, "Level should be squeezed to capacity, got {}", grid.get(8, 8, 8).level);
        let deltas: [(i32, i32, i32); 6] = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)];
        let mut total_neighbors = 0.0f32;
        for (dx, dy, dz) in deltas {
            let nx = (8i32 + dx) as usize;
            let ny = (8i32 + dy) as usize;
            let nz = (8i32 + dz) as usize;
            total_neighbors += grid.get(nx, ny, nz).level;
        }
        assert!(total_neighbors > 0.7, "Excess fluid should have been pushed to neighbors, got {}", total_neighbors);
    }

    #[test]
    fn lava_water_solidification() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);
        grid.get_mut(8, 8, 8).level = 0.5;
        grid.get_mut(8, 8, 8).fluid_type = FluidType::Lava;
        grid.get_mut(9, 8, 8).level = 0.5;
        grid.get_mut(9, 8, 8).fluid_type = FluidType::Water;
        grid.has_fluid = true;
        chunks.insert(key, grid);

        let solidify = detect_solidification(&chunks);
        assert!(!solidify.is_empty(), "Should detect solidification");
    }

    #[test]
    fn water_subtype_solidifies_lava() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);
        grid.get_mut(8, 8, 8).level = 0.5;
        grid.get_mut(8, 8, 8).fluid_type = FluidType::Lava;
        grid.get_mut(9, 8, 8).level = 0.5;
        grid.get_mut(9, 8, 8).fluid_type = FluidType::WaterSpringLine;
        grid.has_fluid = true;
        chunks.insert(key, grid);

        let solidify = detect_solidification(&chunks);
        assert!(!solidify.is_empty(), "Water subtype should solidify lava");
    }

    #[test]
    fn source_regenerates() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);
        grid.get_mut(8, 8, 8).level = SOURCE_LEVEL;
        grid.get_mut(8, 8, 8).fluid_type = FluidType::Water;
        grid.get_mut(8, 8, 8).is_source = true;
        grid.has_fluid = true;
        chunks.insert(key, grid);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        super::super::tick_fluid(&mut chunks, &density_cache, 16, false, &config, true);

        regen_sources(&mut chunks);
        let grid = &chunks[&key];
        assert_eq!(grid.get(8, 8, 8).level, SOURCE_LEVEL);
    }
}
