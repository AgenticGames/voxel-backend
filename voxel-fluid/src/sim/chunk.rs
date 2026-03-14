use std::collections::HashMap;

use crate::cell::{ChunkFluidGrid, FluidType, MIN_LEVEL, ORPHAN_EVAP_TICKS, ORPHAN_THRESHOLD};
use crate::FluidConfig;

/// A pending fluid transfer across a chunk boundary.
pub(super) struct CrossChunkTransfer {
    pub dest_key: (i32, i32, i32),
    pub dest_x: usize,
    pub dest_y: usize,
    pub dest_z: usize,
    pub amount: f32,
    pub fluid_type: FluidType,
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
pub(super) fn resolve_neighbor(
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
pub(super) fn tick_chunk(
    chunks: &mut HashMap<(i32, i32, i32), ChunkFluidGrid>,
    key: (i32, i32, i32),
    _chunk_size: usize,
    is_lava_tick: bool,
    config: &FluidConfig,
    decrement_grace: bool,
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
    let cell_cap = &grid.cell_cap;
    let mut changed = false;
    let mut cross_transfers: Vec<CrossChunkTransfer> = Vec::new();

    // Pre-compute column fluid weight for pressure equalization.
    // fluid_weight[idx] = total fluid in this cell plus all cells above in the same column.
    // A taller column has higher weight at its base, driving upward pressure in shorter neighbors.
    let total = size * size * size;
    let mut fluid_weight = vec![0.0f32; total];
    for z in 0..size {
        for x in 0..size {
            let mut cumulative = 0.0f32;
            for y in (0..size).rev() {
                let idx = z * size * size + y * size + x;
                cumulative += grid.cells[idx].level;
                fluid_weight[idx] = cumulative;
            }
        }
    }

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
                let has_grace = cell.grace_ticks > 0;
                let src_capacity = cell_cap[idx];

                let flow_rate = if is_lava { config.lava_flow_rate } else { config.water_flow_rate };
                let horizontal_spread = if is_lava { config.lava_spread_rate } else { config.water_spread_rate };
                let pressure_rate = if is_lava { config.lava_pressure_rate } else { config.water_pressure_rate };

                // Gravity: try to flow down (8x flow rate for fast pooling)
                if y > 0 {
                    let below_idx = z * size * size + (y - 1) * size + x;
                    if cell_cap[below_idx] > MIN_LEVEL {
                        let below_capacity = cell_cap[below_idx];
                        let below_space = (below_capacity - new_cells[below_idx].level).max(0.0);
                        if below_space > MIN_LEVEL {
                            let transfer = cell.level.min(below_space).min(flow_rate * 8.0);
                            if transfer > MIN_LEVEL {
                                if !is_source && !has_grace {
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
                                let transfer = new_cells[idx].level.min(below_space).min(flow_rate * 8.0);
                                if transfer > MIN_LEVEL {
                                    if !is_source && !has_grace {
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
                        cell_cap[below_idx] < MIN_LEVEL
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

                        // Gather candidates: (channel_score, available_space, target_index_or_cross_chunk_info)
                        // Channel score: prefer cells that already have water (self-reinforcing streams)
                        let mut candidates: Vec<(f32, f32, usize, bool, (i32, i32, i32), usize, usize, usize)> = Vec::new();

                        for (dx, dy, dz) in slope_offsets {
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;
                            let nz = z as i32 + dz;

                            // Bug #3 fix: check horizontal neighbor (dx,0,dz) is passable
                            // to prevent water teleporting diagonally through solid walls.
                            {
                                let hx = x as i32 + dx;
                                let hz = z as i32 + dz;
                                if hx >= 0 && hx < size as i32 && hz >= 0 && hz < size as i32 {
                                    let horiz_idx = if dx != 0 {
                                        z * size * size + y * size + hx as usize
                                    } else {
                                        hz as usize * size * size + y * size + x
                                    };
                                    if cell_cap[horiz_idx] < MIN_LEVEL {
                                        continue; // wall blocks diagonal path
                                    }
                                }
                            }

                            if nx < 0 || nx >= size as i32 || nz < 0 || nz >= size as i32 {
                                // Cross-chunk slope flow for X/Z boundary
                                if let Some((dest_key, tx, ty, tz)) = resolve_neighbor(key, nx, ny, nz, size) {
                                    if let Some(nbr_grid) = chunks.get(&dest_key) {
                                        let cap = nbr_grid.cell_capacity(tx, ty, tz);
                                        if cap >= MIN_LEVEL {
                                            let bi = tz * size * size + ty * size + tx;
                                            let existing = nbr_grid.cells[bi].level;
                                            let dst_space = (cap - existing).max(0.0);
                                            if dst_space > MIN_LEVEL {
                                                let score = existing * 10.0 + dst_space;
                                                candidates.push((score, dst_space, 0, true, dest_key, tx, ty, tz));
                                            }
                                        }
                                    }
                                }
                                continue;
                            }

                            if ny >= 0 && ny < size as i32 {
                                // Within same chunk
                                let ni = nz as usize * size * size + ny as usize * size + nx as usize;
                                if cell_cap[ni] < MIN_LEVEL {
                                    continue;
                                }
                                let dst_capacity = cell_cap[ni];
                                if dst_capacity < MIN_LEVEL {
                                    continue;
                                }
                                let dst_space = (dst_capacity - new_cells[ni].level).max(0.0);
                                if dst_space > MIN_LEVEL {
                                    // Use old state for channel score (not biased by iteration order)
                                    let existing = grid.cells[ni].level;
                                    let score = existing * 10.0 + dst_space;
                                    candidates.push((score, dst_space, ni, false, key, 0, 0, 0));
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
                                    let existing = below_grid.cells[bi].level;
                                    let dst_space = (cap - existing).max(0.0);
                                    if dst_space > MIN_LEVEL {
                                        let score = existing * 10.0 + dst_space;
                                        candidates.push((score, dst_space, 0, true, below_key, tx, ty, tz));
                                    }
                                }
                            }
                        }

                        // Sort by channel score descending: prefer cells with existing water
                        // (self-reinforcing streams), then available space as tiebreaker
                        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

                        // Orphan puddles get boosted slope flow (8x vs 4x)
                        let slope_mult = if cell.level < ORPHAN_THRESHOLD && cell.stagnant_ticks > 0 { 8.0 } else { 4.0 };
                        for (_score, dst_space, ni, is_cross, dest_key, dest_x, dest_y, dest_z) in candidates {
                            if new_cells[idx].level < MIN_LEVEL && !is_source && !has_grace {
                                break;
                            }
                            let transfer = new_cells[idx].level.min(dst_space).min(flow_rate * slope_mult);
                            if transfer > MIN_LEVEL {
                                if !is_source && !has_grace {
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
                // Skip for orphan puddles — force them downhill only
                let is_orphan = cell.level < ORPHAN_THRESHOLD && cell.stagnant_ticks > 0;
                if new_cells[idx].level > MIN_LEVEL && !is_orphan {
                    let neighbors: [(i32, i32, i32); 4] = [
                        (x as i32 + 1, y as i32, z as i32),
                        (x as i32 - 1, y as i32, z as i32),
                        (x as i32, y as i32, z as i32 + 1),
                        (x as i32, y as i32, z as i32 - 1),
                    ];

                    for (nx, ny, nz) in neighbors {
                        // Recompute src_fill from current level each iteration
                        // to prevent over-deduction when multiple neighbors drain us
                        if new_cells[idx].level < MIN_LEVEL && !is_source && !has_grace {
                            break;
                        }
                        let src_fill = if src_capacity > MIN_LEVEL {
                            new_cells[idx].level / src_capacity
                        } else {
                            1.0
                        };

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
                                            let transfer = (diff * horizontal_spread * src_capacity)
                                                .min(flow_rate)
                                                .min(new_cells[idx].level); // prevent overdrain
                                            if transfer > MIN_LEVEL {
                                                if !is_source && !has_grace {
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
                        if cell_cap[ni] < MIN_LEVEL {
                            continue;
                        }
                        let dst_capacity = cell_cap[ni];
                        if dst_capacity < MIN_LEVEL {
                            continue;
                        }
                        let dst_fill = new_cells[ni].level / dst_capacity;
                        let diff = src_fill - dst_fill;
                        if diff > MIN_LEVEL {
                            let dst_space = (dst_capacity - new_cells[ni].level).max(0.0);
                            let transfer = (diff * horizontal_spread * src_capacity)
                                .min(flow_rate)
                                .min(new_cells[idx].level) // prevent overdrain
                                .min(dst_space); // prevent overfill
                            if transfer > MIN_LEVEL {
                                if !is_source && !has_grace {
                                    new_cells[idx].level -= transfer;
                                }
                                new_cells[ni].level += transfer;
                                new_cells[ni].fluid_type = cell.fluid_type;
                                changed = true;
                            }
                        }
                    }
                }

                // Phase 4: Upward pressure equalization
                // Water pushes up when pressurized from below and a neighboring column
                // has more total fluid weight (indicating a higher water surface).
                // This implements hydrostatic pressure: taller columns push shorter
                // neighbors upward through connected fluid.
                if new_cells[idx].level > MIN_LEVEL && y + 1 < size {
                    let below_pressurized = if y > 0 {
                        let bi = z * size * size + (y - 1) * size + x;
                        cell_cap[bi] < MIN_LEVEL || new_cells[bi].level >= 0.95
                    } else {
                        true // chunk floor acts as pressure boundary
                    };

                    if below_pressurized {
                        let ai = z * size * size + (y + 1) * size + x;
                        if cell_cap[ai] > MIN_LEVEL {
                            // Compare column weight with horizontal neighbors
                            let our_weight = fluid_weight[idx];
                            let mut max_neighbor_weight = 0.0f32;
                            for &(dx, dz) in &[(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                                let nx = x as i32 + dx;
                                let nz = z as i32 + dz;
                                if nx >= 0 && nx < size as i32 && nz >= 0 && nz < size as i32 {
                                    let ni = nz as usize * size * size + y * size + nx as usize;
                                    if cell_cap[ni] > MIN_LEVEL {
                                        max_neighbor_weight = max_neighbor_weight.max(fluid_weight[ni]);
                                    }
                                }
                            }

                            let weight_diff = max_neighbor_weight - our_weight;
                            if weight_diff > 0.5 {
                                let above_space = (cell_cap[ai] - new_cells[ai].level).max(0.0);
                                let push = (weight_diff * pressure_rate * 0.3)
                                    .min(above_space)
                                    .min(flow_rate)
                                    .min(new_cells[idx].level);
                                if push > MIN_LEVEL && !is_source && !has_grace {
                                    new_cells[idx].level -= push;
                                    new_cells[ai].level += push;
                                    new_cells[ai].fluid_type = cell.fluid_type;
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Consolidate thin films: push sub-MIN_LEVEL water to a neighbor before zeroing.
    // This prevents silent water loss on slopes where thin films drop below MIN_LEVEL.
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = z * size * size + y * size + x;
                let level = new_cells[idx].level;
                if level <= 0.0 || level >= MIN_LEVEL {
                    continue; // skip empty or substantial cells
                }
                // Try to push tiny amount to a neighbor that has water
                let fluid_type = new_cells[idx].fluid_type;
                let mut pushed = false;
                // Prefer downward, then horizontal, then up
                let consolidate_offsets: [(i32, i32, i32); 6] = [
                    (0, -1, 0),
                    (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1),
                    (0, 1, 0),
                ];
                for &(dx, dy, dz) in &consolidate_offsets {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;
                    if nx < 0 || nx >= size as i32 || ny < 0 || ny >= size as i32
                        || nz < 0 || nz >= size as i32 { continue; }
                    let ni = nz as usize * size * size + ny as usize * size + nx as usize;
                    if new_cells[ni].level >= MIN_LEVEL && cell_cap[ni] > MIN_LEVEL {
                        let space = cell_cap[ni] - new_cells[ni].level;
                        if space > 0.0 {
                            let push = level.min(space);
                            new_cells[ni].level += push;
                            new_cells[ni].fluid_type = fluid_type;
                            new_cells[idx].level = 0.0;
                            pushed = true;
                            break;
                        }
                    }
                }
                if !pushed {
                    new_cells[idx].level = 0.0; // no neighbor to absorb, evaporate
                }
            }
        }
    }

    // Clean up negative from overdrain and track has_fluid
    let mut any_fluid = false;
    for cell in &mut new_cells {
        if cell.level < MIN_LEVEL {
            cell.level = 0.0;
        }
        if cell.level >= MIN_LEVEL {
            any_fluid = true;
        }
    }

    // Bug #2 fix: redistribute excess fluid to neighbors instead of silently clamping.
    // Skip cells with grace ticks (they act as sources, overflow is expected).
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = z * size * size + y * size + x;
                if new_cells[idx].grace_ticks > 0 { continue; }
                let cap = cell_cap[idx];
                if new_cells[idx].level <= cap { continue; }
                let excess = new_cells[idx].level - cap;
                new_cells[idx].level = cap;
                let mut remaining = excess;
                let fluid_type = new_cells[idx].fluid_type;
                for &(dx, dy, dz) in &[(0i32,1i32,0i32),(0,-1,0),(1,0,0),(-1,0,0),(0,0,1),(0,0,-1)] {
                    if remaining < MIN_LEVEL { break; }
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;
                    if nx < 0 || nx >= size as i32 || ny < 0 || ny >= size as i32
                        || nz < 0 || nz >= size as i32 { continue; }
                    let ni = nz as usize * size * size + ny as usize * size + nx as usize;
                    let n_space = (cell_cap[ni] - new_cells[ni].level).max(0.0);
                    if n_space > MIN_LEVEL {
                        let push = remaining.min(n_space);
                        new_cells[ni].level += push;
                        new_cells[ni].fluid_type = fluid_type;
                        remaining -= push;
                    }
                }
                // Any remaining excess that couldn't be redistributed evaporates (rare)
            }
        }
    }

    // Decrement grace ticks (only on last substep to avoid N-times-faster expiry)
    if decrement_grace {
        for cell in &mut new_cells {
            if cell.grace_ticks > 0 {
                cell.grace_ticks -= 1;
            }
        }

        // Orphan puddle tracking + evaporation (only on last substep, like grace)
        let old_cells = &chunks.get(&key).unwrap().cells;
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let idx = z * size * size + y * size + x;
                    let old_level = old_cells[idx].level;
                    let new_level = new_cells[idx].level;
                    let cell_changed = (new_level - old_level).abs() > MIN_LEVEL;

                    // Check if any neighbor has substantial water (pool edge, not orphan)
                    let mut has_pool_neighbor = false;
                    if new_level > MIN_LEVEL && new_level < ORPHAN_THRESHOLD && !cell_changed {
                        let offsets: [(i32,i32,i32); 6] = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)];
                        for &(dx,dy,dz) in &offsets {
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;
                            let nz = z as i32 + dz;
                            if nx >= 0 && nx < size as i32 && ny >= 0 && ny < size as i32
                                && nz >= 0 && nz < size as i32
                            {
                                let ni = nz as usize * size * size + ny as usize * size + nx as usize;
                                if new_cells[ni].level >= ORPHAN_THRESHOLD {
                                    has_pool_neighbor = true;
                                    break;
                                }
                            }
                        }
                    }

                    if new_level > MIN_LEVEL && new_level < ORPHAN_THRESHOLD
                        && !cell_changed && !has_pool_neighbor
                    {
                        new_cells[idx].stagnant_ticks = new_cells[idx].stagnant_ticks.saturating_add(1);
                    } else {
                        new_cells[idx].stagnant_ticks = 0;
                    }

                    // Evaporate truly stuck puddles
                    if new_cells[idx].stagnant_ticks >= ORPHAN_EVAP_TICKS
                        && !new_cells[idx].is_source
                        && new_cells[idx].grace_ticks == 0
                    {
                        new_cells[idx].level *= 0.85; // 15% decay per tick
                        if new_cells[idx].level < MIN_LEVEL {
                            new_cells[idx].level = 0.0;
                            new_cells[idx].stagnant_ticks = 0;
                        }
                        changed = true;
                    }
                }
            }
        }

        // --- Flow entrainment pass ---
        // Fast-moving water drags adjacent stagnant water via viscous coupling.
        // Drain delta (how much a cell lost this tick) is the flow signal.
        let flow_rate = if is_lava_tick { config.lava_flow_rate } else { config.water_flow_rate };
        let entrain_threshold = flow_rate * 0.5;
        let entrain_rate = flow_rate * 2.0;

        // Pre-compute drain deltas (positive = cell lost water this tick)
        let old_cells = &chunks.get(&key).unwrap().cells;
        let mut drain_delta = vec![0.0f32; total];
        for idx in 0..total {
            drain_delta[idx] = (old_cells[idx].level - new_cells[idx].level).max(0.0);
        }

        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let idx = z * size * size + y * size + x;
                    let level = new_cells[idx].level;

                    // Entrain low-to-moderate water toward stronger nearby flow
                    // Cap at 0.5 to protect deep pools from sideways disruption
                    // Require stagnant_ticks > 0 so fresh arrivals cascade normally
                    if level < MIN_LEVEL || level >= 0.5 { continue; }
                    if new_cells[idx].is_source || new_cells[idx].grace_ticks > 0 { continue; }
                    if new_cells[idx].stagnant_ticks == 0 { continue; }

                    // Find horizontal neighbor with largest drain delta
                    // (vertical entrainment skipped — gravity/slope flow handles that)
                    let offsets: [(i32,i32,i32); 4] = [
                        (1,0,0),(-1,0,0),(0,0,1),(0,0,-1)
                    ];
                    let mut best_ni = 0usize;
                    let mut best_drain = 0.0f32;
                    for &(dx, dy, dz) in &offsets {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        let nz = z as i32 + dz;
                        if nx < 0 || nx >= size as i32 || ny < 0 || ny >= size as i32
                            || nz < 0 || nz >= size as i32 { continue; }
                        let ni = nz as usize * size * size + ny as usize * size + nx as usize;
                        if cell_cap[ni] < MIN_LEVEL { continue; } // solid
                        if drain_delta[ni] > best_drain {
                            best_drain = drain_delta[ni];
                            best_ni = ni;
                        }
                    }

                    // Skip if already flowing at >= 30% of the best neighbor's rate
                    if best_drain < entrain_threshold { continue; }
                    if drain_delta[idx] > best_drain * 0.3 { continue; }

                    {
                        let space = (cell_cap[best_ni] - new_cells[best_ni].level).max(0.0);
                        let transfer = level.min(space).min(entrain_rate);
                        if transfer > MIN_LEVEL {
                            new_cells[idx].level -= transfer;
                            new_cells[best_ni].level += transfer;
                            new_cells[best_ni].fluid_type = new_cells[idx].fluid_type;
                            changed = true;
                        }
                    }
                }
            }
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
