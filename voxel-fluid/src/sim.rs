use std::collections::{HashMap, HashSet};

use crate::cell::{ChunkDensityCache, ChunkFluidGrid, FluidType, MIN_LEVEL, SOURCE_LEVEL};
use crate::FluidConfig;

/// A pending fluid transfer across a chunk boundary.
struct CrossChunkTransfer {
    dest_key: (i32, i32, i32),
    dest_x: usize,
    dest_y: usize,
    dest_z: usize,
    amount: f32,
    fluid_type: FluidType,
}

/// Simulate one tick of fluid for all loaded chunks.
///
/// Uses double-buffering: reads from current state, writes to a new buffer,
/// then swaps. Gravity flows downward first, then horizontal spread uses
/// fill-fraction equalization for correct behavior with partial-volume cells.
///
/// Returns the set of chunk keys that had any fluid changes (dirty).
pub fn tick_fluid(
    chunks: &mut HashMap<(i32, i32, i32), ChunkFluidGrid>,
    chunk_densities: &HashMap<(i32, i32, i32), ChunkDensityCache>,
    chunk_size: usize,
    is_lava_tick: bool,
    config: &FluidConfig,
) -> HashSet<(i32, i32, i32)> {
    let mut dirty: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut all_transfers: Vec<CrossChunkTransfer> = Vec::new();

    // Collect keys to iterate
    let keys: Vec<(i32, i32, i32)> = chunks.keys().copied().collect();

    // Pre-promote adjacent chunk grids from density cache so cross-chunk flow
    // can detect and flow into neighbors that have density data but no grid yet.
    // Without this, tick_chunk skips neighbors with no grid (chicken-and-egg).
    let fluid_keys: Vec<(i32, i32, i32)> = keys.iter()
        .filter(|k| chunks.get(k).map_or(false, |g| g.has_fluid))
        .copied()
        .collect();
    let offsets: [(i32, i32, i32); 6] = [
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    ];
    for fk in &fluid_keys {
        for &(dx, dy, dz) in &offsets {
            let adj = (fk.0 + dx, fk.1 + dy, fk.2 + dz);
            if !chunks.contains_key(&adj) {
                if let Some(cache) = chunk_densities.get(&adj) {
                    let grid = ChunkFluidGrid::from_density_cache(cache);
                    chunks.insert(adj, grid);
                }
            }
        }
    }

    // Re-collect keys after promotion (new grids may have been added)
    let keys: Vec<(i32, i32, i32)> = chunks.keys().copied().collect();

    for key in keys {
        // Skip chunks with no fluid and not dirty (nothing to simulate)
        {
            let grid = match chunks.get(&key) {
                Some(g) => g,
                None => continue,
            };
            if !grid.has_fluid && !grid.dirty {
                continue;
            }
        }

        let (changed, transfers) = tick_chunk(chunks, key, chunk_size, is_lava_tick, config);
        if changed {
            dirty.insert(key);
        }
        all_transfers.extend(transfers);
    }

    // Apply cross-chunk transfers (second pass — no borrow conflicts)
    for xfer in &all_transfers {
        // If target chunk has no grid but density exists, create grid on demand
        if !chunks.contains_key(&xfer.dest_key) {
            if let Some(cache) = chunk_densities.get(&xfer.dest_key) {
                let grid = ChunkFluidGrid::from_density_cache(cache);
                chunks.insert(xfer.dest_key, grid);
            } else {
                continue; // no density data, can't create grid
            }
        }

        if let Some(grid) = chunks.get_mut(&xfer.dest_key) {
            let capacity = grid.cell_capacity(xfer.dest_x, xfer.dest_y, xfer.dest_z);
            if capacity < MIN_LEVEL {
                continue; // solid cell
            }
            let cell = grid.get_mut(xfer.dest_x, xfer.dest_y, xfer.dest_z);
            let space = capacity - cell.level;
            let actual = xfer.amount.min(space).max(0.0);
            if actual > MIN_LEVEL {
                cell.level += actual;
                cell.fluid_type = xfer.fluid_type;
                grid.dirty = true;
                grid.has_fluid = true;
                dirty.insert(xfer.dest_key);
            }
        }
    }

    dirty
}

/// Count how many of the 6 face neighbors are solid (or out of bounds).
fn count_solid_face_neighbors(grid: &ChunkFluidGrid, x: usize, y: usize, z: usize) -> u8 {
    let size = grid.size;
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
            if grid.is_solid(nx as usize, ny as usize, nz as usize) {
                count += 1;
            }
        }
    }
    count
}

/// Map out-of-bounds coords to (neighbor_chunk_key, local_x, local_y, local_z).
/// Returns None for multi-axis overflow (diagonal chunks) or if all coords are in bounds.
fn resolve_neighbor(
    key: (i32, i32, i32),
    nx: i32,
    ny: i32,
    nz: i32,
    size: usize,
) -> Option<((i32, i32, i32), usize, usize, usize)> {
    let s = size as i32;
    let mut chunk_key = key;
    let mut lx = nx;
    let mut ly = ny;
    let mut lz = nz;
    let mut crosses = 0u8;

    if lx < 0 {
        chunk_key.0 -= 1;
        lx = s - 1;
        crosses += 1;
    } else if lx >= s {
        chunk_key.0 += 1;
        lx = 0;
        crosses += 1;
    }

    if ly < 0 {
        chunk_key.1 -= 1;
        ly = s - 1;
        crosses += 1;
    } else if ly >= s {
        chunk_key.1 += 1;
        ly = 0;
        crosses += 1;
    }

    if lz < 0 {
        chunk_key.2 -= 1;
        lz = s - 1;
        crosses += 1;
    } else if lz >= s {
        chunk_key.2 += 1;
        lz = 0;
        crosses += 1;
    }

    if crosses != 1 {
        return None; // Multi-axis or same chunk
    }

    Some((chunk_key, lx as usize, ly as usize, lz as usize))
}

/// Simulate one tick for a single chunk. Returns (changed, cross_chunk_transfers).
fn tick_chunk(
    chunks: &mut HashMap<(i32, i32, i32), ChunkFluidGrid>,
    key: (i32, i32, i32),
    _chunk_size: usize,
    is_lava_tick: bool,
    config: &FluidConfig,
) -> (bool, Vec<CrossChunkTransfer>) {
    let grid = match chunks.get(&key) {
        Some(g) => g,
        None => return (false, Vec::new()),
    };

    let size = grid.size;

    // Early return: if no cell has fluid, nothing to simulate
    if !grid.has_fluid {
        return (false, Vec::new());
    }

    // Create new buffer + snapshot density/solid
    let mut new_cells = grid.cells.clone();
    let cell_solid = &grid.cell_solid;
    let mut changed = false;
    let mut cross_transfers: Vec<CrossChunkTransfer> = Vec::new();

    // Process each cell
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = z * size * size + y * size + x;

                // Skip solid cells (all 8 corners positive)
                if cell_solid[idx] {
                    new_cells[idx].level = 0.0;
                    continue;
                }

                let cell = &grid.cells[idx];
                if cell.level < MIN_LEVEL {
                    continue;
                }

                // Check fluid type vs tick type
                let is_lava = cell.fluid_type.is_lava();
                if is_lava && !is_lava_tick {
                    continue;
                }
                if !is_lava && is_lava_tick {
                    continue;
                }

                // Skip flow for sources trapped in solid rock pockets
                let solid_neighbors = count_solid_face_neighbors(
                    chunks.get(&key).unwrap(), x, y, z,
                );
                if solid_neighbors >= 5 {
                    continue;
                }

                let is_source = cell.is_source();
                let src_capacity = if cell_solid[idx] { 0.0 } else { 1.0 };

                let flow_rate = if is_lava { config.lava_flow_rate } else { config.water_flow_rate };
                let horizontal_spread = if is_lava { config.lava_spread_rate } else { config.water_spread_rate };

                // Gravity: try to flow down (4x flow rate for fast pooling)
                if y > 0 {
                    let below_idx = z * size * size + (y - 1) * size + x;
                    if !cell_solid[below_idx] {
                        let below_capacity = if cell_solid[below_idx] { 0.0 } else { 1.0 };
                        let below_space = (below_capacity - new_cells[below_idx].level).max(0.0);
                        if below_space > MIN_LEVEL {
                            let transfer = cell.level.min(below_space).min(flow_rate * 4.0);
                            if transfer > MIN_LEVEL {
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
                        let below_capacity = below_grid.cell_capacity(x, by, z);
                        if below_capacity > MIN_LEVEL {
                            let below_space = (below_capacity - below_grid.cells[below_idx].level).max(0.0);
                            if below_space > MIN_LEVEL {
                                let transfer = new_cells[idx].level.min(below_space).min(flow_rate * 4.0);
                                if transfer > MIN_LEVEL {
                                    if !is_source {
                                        new_cells[idx].level -= transfer;
                                    }
                                    cross_transfers.push(CrossChunkTransfer {
                                        dest_key: below_key,
                                        dest_x: x,
                                        dest_y: by,
                                        dest_z: z,
                                        amount: transfer,
                                        fluid_type: cell.fluid_type,
                                    });
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                // Slope flow: when gravity is blocked by solid, flow diagonally down.
                // Check 4 neighbors at y-1: (x±1, y-1, z) and (x, y-1, z±1).
                if new_cells[idx].level > MIN_LEVEL {
                    let slope_below_solid = if y > 0 {
                        let below_idx = z * size * size + (y - 1) * size + x;
                        cell_solid[below_idx]
                    } else {
                        // y==0: check chunk below
                        let below_key = (key.0, key.1 - 1, key.2);
                        if let Some(below_grid) = chunks.get(&below_key) {
                            below_grid.cell_capacity(x, size - 1, z) < MIN_LEVEL
                        } else {
                            true // no chunk below = treat as solid
                        }
                    };

                    if slope_below_solid {
                        // Collect slope candidates and sort by available space (prefer emptier)
                        let slope_offsets: [(i32, i32, i32); 4] = [
                            (1, -1, 0),
                            (-1, -1, 0),
                            (0, -1, 1),
                            (0, -1, -1),
                        ];

                        // Gather candidates: (available_space, target_index_or_cross_chunk_info)
                        let mut candidates: Vec<(f32, usize, bool, (i32, i32, i32), usize, usize, usize)> = Vec::new();

                        for (dx, dy, dz) in slope_offsets {
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;
                            let nz = z as i32 + dz;

                            if nx < 0 || nx >= size as i32 || nz < 0 || nz >= size as i32 {
                                // Cross-chunk slope flow for X/Z boundary
                                if let Some((dest_key, tx, ty, tz)) = resolve_neighbor(key, nx, ny, nz, size) {
                                    if let Some(nbr_grid) = chunks.get(&dest_key) {
                                        let cap = nbr_grid.cell_capacity(tx, ty, tz);
                                        if cap >= MIN_LEVEL {
                                            let bi = tz * size * size + ty * size + tx;
                                            let dst_space = (cap - nbr_grid.cells[bi].level).max(0.0);
                                            if dst_space > MIN_LEVEL {
                                                candidates.push((dst_space, 0, true, dest_key, tx, ty, tz));
                                            }
                                        }
                                    }
                                }
                                continue;
                            }

                            if ny >= 0 && ny < size as i32 {
                                // Within same chunk
                                let ni = nz as usize * size * size + ny as usize * size + nx as usize;
                                if cell_solid[ni] {
                                    continue;
                                }
                                let dst_capacity = if cell_solid[ni] { 0.0 } else { 1.0 };
                                if dst_capacity < MIN_LEVEL {
                                    continue;
                                }
                                let dst_space = (dst_capacity - new_cells[ni].level).max(0.0);
                                if dst_space > MIN_LEVEL {
                                    candidates.push((dst_space, ni, false, key, 0, 0, 0));
                                }
                            } else if ny < 0 {
                                // Cross-chunk: target is in chunk below at y=size-1
                                let below_key = (key.0, key.1 - 1, key.2);
                                if let Some(below_grid) = chunks.get(&below_key) {
                                    let tx = nx as usize;
                                    let ty = size - 1;
                                    let tz = nz as usize;
                                    let cap = below_grid.cell_capacity(tx, ty, tz);
                                    if cap < MIN_LEVEL {
                                        continue;
                                    }
                                    let bi = tz * size * size + ty * size + tx;
                                    let dst_space = (cap - below_grid.cells[bi].level).max(0.0);
                                    if dst_space > MIN_LEVEL {
                                        candidates.push((dst_space, 0, true, below_key, tx, ty, tz));
                                    }
                                }
                            }
                        }

                        // Sort by available space descending (prefer emptier targets)
                        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

                        for (dst_space, ni, is_cross, dest_key, dest_x, dest_y, dest_z) in candidates {
                            if new_cells[idx].level < MIN_LEVEL && !is_source {
                                break;
                            }
                            let transfer = new_cells[idx].level.min(dst_space).min(flow_rate * 2.0);
                            if transfer > MIN_LEVEL {
                                if !is_source {
                                    new_cells[idx].level -= transfer;
                                }
                                if is_cross {
                                    cross_transfers.push(CrossChunkTransfer {
                                        dest_key,
                                        dest_x,
                                        dest_y,
                                        dest_z,
                                        amount: transfer,
                                        fluid_type: cell.fluid_type,
                                    });
                                } else {
                                    new_cells[ni].level += transfer;
                                    new_cells[ni].fluid_type = cell.fluid_type;
                                }
                                changed = true;
                            }
                        }
                    }
                }

                // Horizontal spread using fill-fraction equalization
                if new_cells[idx].level > MIN_LEVEL {
                    let remaining = new_cells[idx].level;
                    let src_fill = if src_capacity > MIN_LEVEL {
                        remaining / src_capacity
                    } else {
                        1.0
                    };
                    let neighbors: [(i32, i32, i32); 4] = [
                        (x as i32 + 1, y as i32, z as i32),
                        (x as i32 - 1, y as i32, z as i32),
                        (x as i32, y as i32, z as i32 + 1),
                        (x as i32, y as i32, z as i32 - 1),
                    ];

                    for (nx, ny, nz) in neighbors {
                        if nx < 0 || nx >= size as i32 || ny < 0 || ny >= size as i32 || nz < 0 || nz >= size as i32 {
                            // Cross-chunk horizontal flow
                            if let Some((dest_key, tx, ty, tz)) = resolve_neighbor(key, nx, ny, nz, size) {
                                if let Some(nbr_grid) = chunks.get(&dest_key) {
                                    let cap = nbr_grid.cell_capacity(tx, ty, tz);
                                    if cap >= MIN_LEVEL {
                                        let bi = tz * size * size + ty * size + tx;
                                        let dst_fill = nbr_grid.cells[bi].level / cap;
                                        let diff = src_fill - dst_fill;
                                        if diff > MIN_LEVEL {
                                            let transfer = (diff * horizontal_spread * src_capacity).min(flow_rate);
                                            if transfer > MIN_LEVEL {
                                                if !is_source {
                                                    new_cells[idx].level -= transfer;
                                                }
                                                cross_transfers.push(CrossChunkTransfer {
                                                    dest_key,
                                                    dest_x: tx,
                                                    dest_y: ty,
                                                    dest_z: tz,
                                                    amount: transfer,
                                                    fluid_type: cell.fluid_type,
                                                });
                                                changed = true;
                                            }
                                        }
                                    }
                                }
                            }
                            continue;
                        }
                        let ni = nz as usize * size * size + ny as usize * size + nx as usize;
                        if cell_solid[ni] {
                            continue;
                        }
                        let dst_capacity = if cell_solid[ni] { 0.0 } else { 1.0 };
                        if dst_capacity < MIN_LEVEL {
                            continue;
                        }
                        let dst_fill = new_cells[ni].level / dst_capacity;
                        let diff = src_fill - dst_fill;
                        if diff > MIN_LEVEL {
                            let transfer = (diff * horizontal_spread * src_capacity).min(flow_rate);
                            if transfer > MIN_LEVEL {
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

    // Clean up tiny residual amounts and track has_fluid
    let mut any_fluid = false;
    for cell in &mut new_cells {
        if cell.level < MIN_LEVEL && cell.level > 0.0 {
            cell.level = 0.0;
        }
        if cell.level >= MIN_LEVEL {
            any_fluid = true;
        }
    }

    // Swap buffer
    if changed {
        if let Some(grid) = chunks.get_mut(&key) {
            grid.cells = new_cells;
            grid.dirty = true;
            grid.has_fluid = any_fluid;
        }
    } else if let Some(grid) = chunks.get_mut(&key) {
        // Even without changes, update has_fluid status
        grid.has_fluid = any_fluid;
    }

    (changed, cross_transfers)
}

/// After a density update, squeeze excess fluid from cells whose capacity decreased.
/// Excess is pushed to non-solid neighbors; any remainder is evaporated.
pub fn squeeze_excess_fluid(grid: &mut ChunkFluidGrid) {
    let size = grid.size;
    let mut any_change = false;

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = z * size * size + y * size + x;
                let capacity = if grid.cell_solid[idx] { 0.0 } else { 1.0 };
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
                    let n_capacity = if grid.cell_solid[ni] { 0.0 } else { 1.0 };
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

    fn make_chunk(size: usize) -> ChunkFluidGrid {
        ChunkFluidGrid::new(size)
    }

    fn empty_density_cache() -> HashMap<(i32, i32, i32), ChunkDensityCache> {
        HashMap::new()
    }

    #[test]
    fn gravity_flows_down() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);
        // Place water at y=8 (non-source level so it can flow)
        grid.get_mut(8, 8, 8).level = 0.8;
        grid.get_mut(8, 8, 8).fluid_type = FluidType::Water;
        grid.has_fluid = true;
        chunks.insert(key, grid);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        // Single tick should move water one step down
        tick_fluid(&mut chunks, &density_cache, 16, false, &config);

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
        grid.has_fluid = true;
        chunks.insert(key, grid);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        // Tick
        tick_fluid(&mut chunks, &density_cache, 16, false, &config);

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
        grid.has_fluid = true;
        // Make y=7 solid
        grid.set_density(8, 7, 8, 1.0);
        chunks.insert(key, grid);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        tick_fluid(&mut chunks, &density_cache, 16, false, &config);

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

        // Lava at (8,8,8)
        grid.get_mut(8, 8, 8).level = 0.5;
        grid.get_mut(8, 8, 8).fluid_type = FluidType::Lava;

        // WaterSpringLine at (9,8,8) — should solidify lava just like Water
        grid.get_mut(9, 8, 8).level = 0.5;
        grid.get_mut(9, 8, 8).fluid_type = FluidType::WaterSpringLine;

        grid.has_fluid = true;
        chunks.insert(key, grid);

        let solidify = detect_solidification(&chunks);
        assert!(!solidify.is_empty(), "Water subtype should solidify lava");
    }

    #[test]
    fn cross_chunk_downward_flow() {
        // Water at y=0 of upper chunk should flow to y=15 of lower chunk
        let mut chunks = HashMap::new();
        let upper_key = (0, 1, 0);
        let lower_key = (0, 0, 0);

        let mut upper_grid = make_chunk(16);
        // Place water at y=0 (chunk boundary)
        upper_grid.get_mut(8, 0, 8).level = 0.8;
        upper_grid.get_mut(8, 0, 8).fluid_type = FluidType::Water;
        upper_grid.has_fluid = true;
        chunks.insert(upper_key, upper_grid);

        let lower_grid = make_chunk(16);
        chunks.insert(lower_key, lower_grid);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        tick_fluid(&mut chunks, &density_cache, 16, false, &config);

        // Fluid should appear at y=15 of lower chunk
        let lower = &chunks[&lower_key];
        assert!(
            lower.get(8, 15, 8).level > 0.0,
            "Water should flow across chunk boundary to y=15 below"
        );

        // Upper chunk should have lost some fluid
        let upper = &chunks[&upper_key];
        assert!(
            upper.get(8, 0, 8).level < 0.8,
            "Upper chunk should have transferred fluid downward"
        );
    }

    #[test]
    fn contained_source_doesnt_flow() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);
        // Place water source at (8,8,8)
        grid.get_mut(8, 8, 8).level = SOURCE_LEVEL;
        grid.get_mut(8, 8, 8).fluid_type = FluidType::Water;
        grid.has_fluid = true;
        // Surround with solid on all 6 faces
        grid.set_density(7, 8, 8, 1.0);
        grid.set_density(9, 8, 8, 1.0);
        grid.set_density(8, 7, 8, 1.0);
        grid.set_density(8, 9, 8, 1.0);
        grid.set_density(8, 8, 7, 1.0);
        grid.set_density(8, 8, 9, 1.0);
        chunks.insert(key, grid);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        // Tick several times
        for _ in 0..10 {
            tick_fluid(&mut chunks, &density_cache, 16, false, &config);
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

    #[test]
    fn partial_capacity_limits_fill() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);
        // Cell with density=-0.5 (air side of SDF → binary capacity 1.0)
        grid.set_density(8, 7, 8, -0.5);
        // Place water above
        grid.get_mut(8, 8, 8).level = 0.8;
        grid.get_mut(8, 8, 8).fluid_type = FluidType::Water;
        grid.has_fluid = true;
        chunks.insert(key, grid);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        // Multiple ticks to let fluid settle
        for _ in 0..20 {
            tick_fluid(&mut chunks, &density_cache, 16, false, &config);
        }

        // With binary capacity, density=-0.5 → capacity 1.0
        // Fluid should flow down and not exceed capacity
        let grid = &chunks[&key];
        assert!(
            grid.get(8, 7, 8).level <= 1.001,
            "Fluid should not exceed cell capacity, got {}",
            grid.get(8, 7, 8).level
        );
    }

    #[test]
    fn slope_flow_down() {
        // Water on a solid floor with adjacent air below should flow diagonally down
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);

        // Place water at (8, 5, 8) — sitting on solid
        grid.get_mut(8, 5, 8).level = 0.8;
        grid.get_mut(8, 5, 8).fluid_type = FluidType::Water;
        grid.has_fluid = true;

        // Make floor solid at y=4, x=8, z=8 (directly below)
        grid.set_density(8, 4, 8, 1.0);

        // But leave (9, 4, 8) as air — slope target
        // Default density is -1.0 (air), so no change needed

        chunks.insert(key, grid);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        tick_fluid(&mut chunks, &density_cache, 16, false, &config);

        let grid = &chunks[&key];
        assert!(
            grid.get(9, 4, 8).level > 0.0,
            "Water should slope-flow diagonally down to (9,4,8), got {}",
            grid.get(9, 4, 8).level
        );
    }

    #[test]
    fn slope_flow_cascades() {
        // Staircase terrain with a basin at the bottom to catch water.
        // Water starts at (8,5,8), cascades down steps, collects in basin at y=1.
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);

        // Water source at top of staircase
        grid.get_mut(8, 5, 8).level = 0.8;
        grid.get_mut(8, 5, 8).fluid_type = FluidType::Water;
        grid.has_fluid = true;

        // Staircase floors (each step one lower in Y, one offset in X)
        grid.set_density(8, 4, 8, 1.0);   // floor under start
        grid.set_density(9, 3, 8, 1.0);   // floor under step 1
        grid.set_density(10, 2, 8, 1.0);  // floor under step 2

        // Basin floor at y=0 across the area to catch cascaded water
        for x in 8..14 {
            grid.set_density(x, 0, 8, 1.0);
        }

        chunks.insert(key, grid);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();

        // Multiple ticks to let water cascade all the way down
        for _ in 0..15 {
            tick_fluid(&mut chunks, &density_cache, 16, false, &config);
        }

        // Check that water reached below starting Y level (anywhere y < 4)
        let grid = &chunks[&key];
        let mut found_below = false;
        for y in 0..4 {
            for x in 7..14 {
                if grid.get(x, y, 8).level > MIN_LEVEL {
                    found_below = true;
                    break;
                }
            }
            if found_below { break; }
        }
        assert!(found_below, "Water should cascade down the staircase to lower Y levels");
    }

    #[test]
    fn slope_flow_blocked_by_solid() {
        // If all slope targets are solid, water should NOT slope-flow
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);

        grid.get_mut(8, 5, 8).level = 0.8;
        grid.get_mut(8, 5, 8).fluid_type = FluidType::Water;
        grid.has_fluid = true;

        // Make floor solid
        grid.set_density(8, 4, 8, 1.0);
        // Make ALL slope targets solid too
        grid.set_density(9, 4, 8, 1.0);
        grid.set_density(7, 4, 8, 1.0);
        grid.set_density(8, 4, 9, 1.0);
        grid.set_density(8, 4, 7, 1.0);

        chunks.insert(key, grid);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        tick_fluid(&mut chunks, &density_cache, 16, false, &config);

        // Water should stay at y=5 (only horizontal spread possible)
        let grid = &chunks[&key];
        for dy in [4i32] {
            for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let nx = (8i32 + dx) as usize;
                let ny = (5i32 + dy) as usize;
                let nz = (8i32 + dz) as usize;
                assert!(
                    grid.get(nx, ny, nz).level < MIN_LEVEL,
                    "Water should not enter solid slope target ({},{},{}), got {}",
                    nx, ny, nz, grid.get(nx, ny, nz).level
                );
            }
        }
    }

    #[test]
    fn increased_spread_rate() {
        // Verify default spread rate is now 0.6 and water reaches adjacent cells quickly
        let config = crate::FluidConfig::default();
        assert!(
            (config.water_spread_rate - 0.6).abs() < 0.01,
            "Default water_spread_rate should be 0.6, got {}",
            config.water_spread_rate
        );

        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);

        // Place water on solid floor with horizontal room
        grid.get_mut(8, 5, 8).level = 0.8;
        grid.get_mut(8, 5, 8).fluid_type = FluidType::Water;
        grid.has_fluid = true;

        // Solid floor at y=4 across the area (blocks gravity + slope flow)
        for x in 0..16 {
            for z in 0..16 {
                grid.set_density(x, 4, z, 1.0);
            }
        }

        chunks.insert(key, grid);

        let density_cache = empty_density_cache();
        for _ in 0..3 {
            tick_fluid(&mut chunks, &density_cache, 16, false, &config);
        }

        // Check that water reached immediate neighbors after just 3 ticks
        let grid = &chunks[&key];
        let neighbor = grid.get(9, 5, 8).level;
        assert!(
            neighbor > MIN_LEVEL,
            "Water should spread to adjacent cell (9,5,8) with rate 0.6 after 3 ticks, got {}",
            neighbor
        );
    }

    #[test]
    fn cross_chunk_slope_flow() {
        // Water at y=0 with solid below should slope-flow to adjacent cell in chunk below
        let mut chunks = HashMap::new();
        let upper_key = (0, 1, 0);
        let lower_key = (0, 0, 0);

        let mut upper_grid = make_chunk(16);
        upper_grid.get_mut(8, 0, 8).level = 0.8;
        upper_grid.get_mut(8, 0, 8).fluid_type = FluidType::Water;
        upper_grid.has_fluid = true;

        let mut lower_grid = make_chunk(16);
        // Make (8, 15, 8) in lower chunk solid (directly below upper's y=0)
        lower_grid.set_density(8, 15, 8, 1.0);
        // Leave (9, 15, 8) as air — slope target

        chunks.insert(upper_key, upper_grid);
        chunks.insert(lower_key, lower_grid);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        tick_fluid(&mut chunks, &density_cache, 16, false, &config);

        let lower = &chunks[&lower_key];
        assert!(
            lower.get(9, 15, 8).level > 0.0,
            "Water should cross-chunk slope-flow to (9,15,8), got {}",
            lower.get(9, 15, 8).level
        );
    }

    #[test]
    fn cross_chunk_horizontal_flow() {
        // Water at x=15 with solid floor should spread to x=0 of +X neighbor chunk
        let mut chunks = HashMap::new();
        let key_a = (0, 0, 0);
        let key_b = (1, 0, 0);

        let mut grid_a = make_chunk(16);
        // Solid floor at y=0
        for x in 0..16 {
            for z in 0..16 {
                grid_a.set_density(x, 0, z, 1.0);
            }
        }
        // Place water at x=15, y=1 (on solid floor, near +X boundary)
        grid_a.get_mut(15, 1, 8).level = 0.8;
        grid_a.get_mut(15, 1, 8).fluid_type = FluidType::Water;
        grid_a.has_fluid = true;
        chunks.insert(key_a, grid_a);

        let mut grid_b = make_chunk(16);
        // Solid floor in neighbor too
        for x in 0..16 {
            for z in 0..16 {
                grid_b.set_density(x, 0, z, 1.0);
            }
        }
        chunks.insert(key_b, grid_b);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        // Several ticks to let water spread horizontally
        for _ in 0..10 {
            tick_fluid(&mut chunks, &density_cache, 16, false, &config);
        }

        // Water should have spread to x=0 of the +X neighbor chunk
        let nbr = &chunks[&key_b];
        assert!(
            nbr.get(0, 1, 8).level > 0.0,
            "Water should flow across chunk boundary to x=0 of +X neighbor, got {}",
            nbr.get(0, 1, 8).level
        );
    }

    #[test]
    fn cross_chunk_slope_flow_xz() {
        // Water at x=15, y=1 with solid directly below should slope-flow to +X neighbor
        let mut chunks = HashMap::new();
        let key_a = (0, 0, 0);
        let key_b = (1, 0, 0);

        let mut grid_a = make_chunk(16);
        // Solid floor at y=0 everywhere
        for x in 0..16 {
            for z in 0..16 {
                grid_a.set_density(x, 0, z, 1.0);
            }
        }
        // Place water at x=15, y=1
        grid_a.get_mut(15, 1, 8).level = 0.8;
        grid_a.get_mut(15, 1, 8).fluid_type = FluidType::Water;
        grid_a.has_fluid = true;
        chunks.insert(key_a, grid_a);

        let grid_b = make_chunk(16);
        // No floor in neighbor at y=0 — so slope target (x=0, y=0, z=8) is open
        chunks.insert(key_b, grid_b);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        for _ in 0..5 {
            tick_fluid(&mut chunks, &density_cache, 16, false, &config);
        }

        // Water should have slope-flowed to x=0, y=0 in the +X neighbor
        let nbr = &chunks[&key_b];
        assert!(
            nbr.get(0, 0, 8).level > 0.0,
            "Water should cross-chunk slope-flow to (0,0,8) in +X neighbor, got {}",
            nbr.get(0, 0, 8).level
        );
    }

    #[test]
    fn squeeze_excess_works() {
        let mut grid = make_chunk(16);
        // Fill cell with water
        grid.get_mut(8, 8, 8).level = 0.8;
        grid.get_mut(8, 8, 8).fluid_type = FluidType::Water;

        // Make the cell solid (capacity drops to 0.0 with binary capacity)
        grid.set_density(8, 8, 8, 0.5);

        squeeze_excess_fluid(&mut grid);

        // Level should be squeezed to capacity (0.0 for solid)
        assert!(
            grid.get(8, 8, 8).level <= 0.001,
            "Level should be squeezed to capacity, got {}",
            grid.get(8, 8, 8).level
        );
        // All excess should have been pushed to neighbors
        let mut total_neighbors = 0.0f32;
        let deltas: [(i32, i32, i32); 6] = [
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
        ];
        for (dx, dy, dz) in deltas {
            let nx = (8i32 + dx) as usize;
            let ny = (8i32 + dy) as usize;
            let nz = (8i32 + dz) as usize;
            total_neighbors += grid.get(nx, ny, nz).level;
        }
        assert!(
            total_neighbors > 0.7,
            "Excess fluid should have been pushed to neighbors, got {}",
            total_neighbors
        );
    }
}
