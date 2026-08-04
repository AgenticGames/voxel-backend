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
    /// Bounded-flow propagation: hops the dest cell will have after this transfer.
    /// 0 means "no source recorded / unbounded" (for legacy compatibility).
    pub dest_hops: u8,
    /// Bounded-flow propagation: dest cell inherits this max from the source it
    /// propagated from. 0 = unlimited.
    pub dest_max_flow: u8,
}

/// Number of hops at the tail of a bounded source where flow tapers from 1.0 → 0.
/// Cells closer to the source than `max - TAPER_HOPS` get level=1.0 (full).
/// Cells in the last `TAPER_HOPS` ramp linearly down toward 0 at the limit.
pub const TAPER_HOPS: u8 = 4;

/// Per-cell level cap for a child of a bounded source.
/// `new_hops`: hops the destination cell will have after this transfer (1 = direct neighbor).
/// `max_flow`: source's max_flow_dist (0 = unlimited → returns 1.0).
/// Returns 0.0 if the hop would exceed the limit (caller should skip transfer).
#[inline]
pub fn bounded_level_cap(new_hops: u8, max_flow: u8) -> f32 {
    if max_flow == 0 { return 1.0; }
    if new_hops > max_flow { return 0.0; }
    let head_zone = max_flow.saturating_sub(TAPER_HOPS);
    if new_hops <= head_zone {
        1.0
    } else {
        // Linear ramp 1.0 → 0.0 across the last TAPER_HOPS cells.
        let into_taper = (new_hops - head_zone) as f32;
        let taper_span = TAPER_HOPS.min(max_flow) as f32;
        (1.0 - into_taper / taper_span).max(0.0)
    }
}

/// Returns true if a transfer from `src` should be skipped because it would
/// exceed the source's bounded reach. Used to short-circuit before doing the
/// usual transfer math.
#[inline]
pub fn bounded_blocks_transfer(src_hops: u8, src_max_flow: u8) -> bool {
    if src_max_flow == 0 { return false; }
    src_hops >= src_max_flow
}

/// Count how many of the 6 face neighbors are solid (or out of bounds).
///
/// Operates on the precomputed `cell_solid` bitfield directly so callers in
/// the inner xyz loop don't have to do a `chunks.get(&key)` HashMap probe per
/// voxel just to reach the same data they already hold above the loop.
#[inline]
fn count_solid_face_neighbors(cell_solid: &[bool], size: usize, x: usize, y: usize, z: usize) -> u8 {
    let s = size as i32;
    let mut count: u8 = 0;
    let stride_y = size;
    let stride_z = size * size;
    let deltas: [(i32, i32, i32); 6] = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ];
    for (dx, dy, dz) in deltas {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        let nz = z as i32 + dz;
        if nx < 0 || nx >= s || ny < 0 || ny >= s || nz < 0 || nz >= s {
            count += 1; // out of bounds = solid
        } else {
            let ni = (nz as usize) * stride_z + (ny as usize) * stride_y + (nx as usize);
            if cell_solid[ni] {
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
    let (size, total, has_fluid) = match chunks.get(&key) {
        Some(g) => (g.size, g.size * g.size * g.size, g.has_fluid),
        None => return (false, Vec::new()),
    };

    // Early return: if no cell has fluid, nothing to simulate
    if !has_fluid {
        return (false, Vec::new());
    }

    // Take ALL scratch buffers off the grid up front, before we hold any
    // immutable borrow of `cell_cap`/`cell_solid` etc. Restored at the end.
    // Reusing these allocations is the single biggest win for tick perf —
    // at chunk_size=30 each scratch is ≈540KB / ≈108KB / ≈108KB, allocated
    // 6× per water tick × N chunks/sec on the old code path.
    let (mut new_cells, mut fluid_weight, mut drain_scratch) = {
        let g = chunks.get_mut(&key).unwrap();
        let mut nc = std::mem::take(&mut g.scratch_cells);
        let mut fw = std::mem::take(&mut g.scratch_weights);
        let dd = std::mem::take(&mut g.scratch_drain);
        if nc.len() == total {
            nc.copy_from_slice(&g.cells);
        } else {
            nc.clear();
            nc.extend_from_slice(&g.cells);
        }
        fw.clear();
        fw.resize(total, 0.0);
        (nc, fw, dd)
    };

    let grid = chunks.get(&key).unwrap();
    let cell_solid = &grid.cell_solid;
    let cell_cap = &grid.cell_cap;
    // Face gating (2026-08-04, bug #215): the rendered surface must be a
    // transit barrier. Checked from the SOURCE cell's own corners, so
    // cross-chunk directions need no neighbor data.
    let corners = &grid.cell_corners;
    let face_open = |idx: usize, dx: i32, dy: i32, dz: i32| -> bool {
        !crate::cell::face_blocked(corners, idx, dx, dy, dz)
    };
    let mut changed = false;
    let mut cross_transfers: Vec<CrossChunkTransfer> = Vec::new();
    // Reused across all cells in this chunk-tick. Slope flow gathers ≤4
    // candidate neighbors per cell, sorts them, then drains. Hoisting the
    // Vec (and `.clear()`-ing between cells) drops one heap alloc per
    // slope-active cell — same style as the cells/weights/drain scratch
    // reuse already done on `ChunkFluidGrid`.
    let mut slope_candidates:
        Vec<(f32, f32, usize, bool, (i32, i32, i32), usize, usize, usize)> =
        Vec::with_capacity(4);

    // ---- Neighbor-chunk references, hoisted out of the per-voxel loop ----
    // Cross-chunk flow only ever READS the 5 neighbour chunks (the one below +
    // the 4 lateral). Every cross-chunk *write* is deferred into
    // `cross_transfers`, and every within-chunk write targets the owned
    // `new_cells` scratch — so `chunks` is never mutated inside the loop. The
    // neighbour keys are invariant for the whole chunk-tick, yet the old code
    // re-probed `chunks.get(&neighbour_key)` (std HashMap = SipHash on a 12-byte
    // key) for *every boundary fluid voxel*. Probe each once here; the per-voxel
    // sites below then pick the right cached reference with a few cheap tuple
    // comparisons instead. Same hoist the stress passes applied to per-cell
    // HashMap probes (see PERF_REVIEW history).
    let key_below = (key.0, key.1 - 1, key.2);
    let key_xp = (key.0 + 1, key.1, key.2);
    let key_xn = (key.0 - 1, key.1, key.2);
    let key_zp = (key.0, key.1, key.2 + 1);
    let key_zn = (key.0, key.1, key.2 - 1);
    let nbr_below = chunks.get(&key_below);
    let nbr_xp = chunks.get(&key_xp);
    let nbr_xn = chunks.get(&key_xn);
    let nbr_zp = chunks.get(&key_zp);
    let nbr_zn = chunks.get(&key_zn);

    // Pre-compute column fluid weight for pressure equalization (Phase 4).
    // fluid_weight[idx] = total fluid in this cell plus all cells above in the same column.
    // A taller column has higher weight at its base, driving upward pressure in shorter neighbors.
    //
    // Optimization: fluid_weight starts zeroed (`fw.resize(total, 0.0)` above).
    // For each column, walk down from the top and skip writes until the first
    // non-empty cell — empty cells above stay 0 (the correct cumulative). Then
    // accumulate normally from there. At chunk_size=30 most caves have ~5–20%
    // column-height coverage by fluid; this turns the dense 27 000-cell scan
    // into the column-height work only. Strides are also better: the original
    // (z, x, y_rev) loop hit a fresh cache line per step (Y stride ≈ 360 B on
    // FluidCell at size=30); we now read only as many y-rows as actually hold
    // fluid plus the cells beneath them.
    let stride_y = size;
    let stride_z = size * size;
    for z in 0..size {
        for x in 0..size {
            let base = z * stride_z + x;
            let mut y = size;
            let mut cumulative = 0.0f32;
            // Skip the all-empty cap of this column — leaves fluid_weight = 0.
            while y > 0 {
                y -= 1;
                let idx = base + y * stride_y;
                let level = grid.cells[idx].level;
                if level > 0.0 {
                    cumulative = level;
                    fluid_weight[idx] = cumulative;
                    break;
                }
            }
            // From the topmost fluid cell down to y=0, accumulate normally:
            // empty cells beneath stacked fluid still need their (non-zero)
            // cumulative recorded so Phase-4 neighbor lookups are correct.
            while y > 0 {
                y -= 1;
                let idx = base + y * stride_y;
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

                // Skip flow for sources trapped in solid rock pockets.
                // Reads the borrowed cell_solid slice directly — avoids the
                // per-voxel HashMap.get(&key) probe that the previous version
                // performed in the inner xyz loop.
                let solid_neighbors = count_solid_face_neighbors(cell_solid, size, x, y, z);
                if solid_neighbors >= 5 {
                    continue;
                }

                let is_source = cell.is_source();
                let has_grace = cell.grace_ticks > 0;
                let src_capacity = cell_cap[idx];

                let flow_rate = if is_lava { config.lava_flow_rate } else { config.water_flow_rate };
                let horizontal_spread = if is_lava { config.lava_spread_rate } else { config.water_spread_rate };
                let pressure_rate = if is_lava { config.lava_pressure_rate } else { config.water_pressure_rate };
                // Lava falls in HALF-gulps (2026-08-04): the water 8x gravity
                // multiplier moved near-full cells per tick, so a cascade's
                // cells flip full->empty->full and the mesh flashes wildly.
                // At 4x, transit cells hold partial levels across ticks and
                // a fall reads as a continuous ribbon. Water keeps 8x.
                let gravity_mult = if is_lava { 4.0 } else { 8.0 };

                // Gravity: try to flow down (8x flow rate for fast pooling)
                if y > 0 {
                    let below_idx = z * size * size + (y - 1) * size + x;
                    if cell_cap[below_idx] > MIN_LEVEL
                        && face_open(idx, 0, -1, 0)
                        && !bounded_blocks_transfer(cell.hops_from_source, cell.max_flow_dist)
                    {
                        let below_capacity = cell_cap[below_idx];
                        let below_space = (below_capacity - new_cells[below_idx].level).max(0.0);
                        if below_space > MIN_LEVEL {
                            // Bounded-flow cap: child receives at most `level_cap`.
                            let new_hops = cell.hops_from_source.saturating_add(1);
                            let cap = bounded_level_cap(new_hops, cell.max_flow_dist);
                            let bounded_space = (cap - new_cells[below_idx].level).max(0.0).min(below_space);
                            let transfer = cell.level.min(bounded_space).min(flow_rate * gravity_mult);
                            if transfer > MIN_LEVEL {
                                if !is_source && !has_grace {
                                    new_cells[idx].level -= transfer;
                                }
                                new_cells[below_idx].level += transfer;
                                new_cells[below_idx].fluid_type = cell.fluid_type;
                                new_cells[below_idx].hops_from_source = new_hops;
                                new_cells[below_idx].max_flow_dist = cell.max_flow_dist;
                                changed = true;
                            }
                        }
                    }
                }
                // Cross-chunk downward flow: y==0 means neighbor chunk below
                else {
                    // Void-cull backstop (2026-08-04, user directive): no
                    // loaded chunk below AND our own bottom face is open =
                    // this fluid is resting on the EDGE OF LOADED SPACE, not
                    // on rock. The old "treat as solid" default made the
                    // bottom of the world a raft — escaped fluid rivers
                    // travelled along the void-side underside forever. Let
                    // it fall out of the world instead (sources included:
                    // their emissions vanish rather than accumulate, and the
                    // self-extinguish pass then retires them).
                    if nbr_below.is_none()
                        && face_open(idx, 0, -1, 0)
                        && new_cells[idx].level > MIN_LEVEL
                    {
                        new_cells[idx].level = 0.0;
                        changed = true;
                    }
                    let below_key = key_below;
                    if let Some(below_grid) = nbr_below {
                        if face_open(idx, 0, -1, 0)
                            && !bounded_blocks_transfer(cell.hops_from_source, cell.max_flow_dist)
                        {
                            let by = size - 1;
                            let below_idx = z * size * size + by * size + x;
                            let below_capacity = below_grid.cell_capacity(x, by, z);
                            if below_capacity > MIN_LEVEL {
                                let below_space = (below_capacity - below_grid.cells[below_idx].level).max(0.0);
                                if below_space > MIN_LEVEL {
                                    let new_hops = cell.hops_from_source.saturating_add(1);
                                    let cap = bounded_level_cap(new_hops, cell.max_flow_dist);
                                    let bounded_space = (cap - below_grid.cells[below_idx].level).max(0.0).min(below_space);
                                    let transfer = new_cells[idx].level.min(bounded_space).min(flow_rate * gravity_mult);
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
                                            dest_hops: new_hops,
                                            dest_max_flow: cell.max_flow_dist,
                                        });
                                        changed = true;
                                    }
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
                        if let Some(below_grid) = nbr_below {
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
                        // Channel score: prefer cells that already have water (self-reinforcing streams).
                        // Reuses `slope_candidates` (hoisted to the fn top) so we don't re-allocate per cell.
                        slope_candidates.clear();
                        let candidates = &mut slope_candidates;

                        for (dx, dy, dz) in slope_offsets {
                            // Face gating: the sideways leg of the diagonal
                            // must not cross a rendered surface (the dest
                            // cell's own down-face gate covers the drop leg
                            // next tick).
                            if !face_open(idx, dx, 0, dz) {
                                continue;
                            }
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
                                    // Single-axis ±X/±Z crossing here → use the hoisted
                                    // neighbour ref instead of re-probing the HashMap.
                                    let nbr = if dest_key == key_xp { nbr_xp }
                                        else if dest_key == key_xn { nbr_xn }
                                        else if dest_key == key_zp { nbr_zp }
                                        else { debug_assert_eq!(dest_key, key_zn); nbr_zn };
                                    if let Some(nbr_grid) = nbr {
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
                                let below_key = key_below;
                                if let Some(below_grid) = nbr_below {
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
                        // Bounded-flow gate (apply once per source per tick).
                        let bounded_blocked = bounded_blocks_transfer(cell.hops_from_source, cell.max_flow_dist);
                        let new_hops = cell.hops_from_source.saturating_add(1);
                        let level_cap = bounded_level_cap(new_hops, cell.max_flow_dist);
                        for &(_score, dst_space, ni, is_cross, dest_key, dest_x, dest_y, dest_z) in candidates.iter() {
                            if bounded_blocked { break; }
                            if new_cells[idx].level < MIN_LEVEL && !is_source && !has_grace {
                                break;
                            }
                            // Cap dst's allowed receive level by bounded-flow rule.
                            let dst_existing = if is_cross { 0.0 } else { new_cells[ni].level };
                            let bounded_space = (level_cap - dst_existing).max(0.0).min(dst_space);
                            let transfer = new_cells[idx].level.min(bounded_space).min(flow_rate * slope_mult);
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
                                        dest_hops: new_hops,
                                        dest_max_flow: cell.max_flow_dist,
                                    });
                                } else {
                                    new_cells[ni].level += transfer;
                                    new_cells[ni].fluid_type = cell.fluid_type;
                                    new_cells[ni].hops_from_source = new_hops;
                                    new_cells[ni].max_flow_dist = cell.max_flow_dist;
                                }
                                changed = true;
                            }
                        }
                    }
                }

                // Horizontal spread using fill-fraction equalization
                // Skip for orphan puddles — force them downhill only
                let is_orphan = cell.level < ORPHAN_THRESHOLD && cell.stagnant_ticks > 0;
                if new_cells[idx].level > MIN_LEVEL && !is_orphan
                    && !bounded_blocks_transfer(cell.hops_from_source, cell.max_flow_dist)
                {
                    let neighbors: [(i32, i32, i32); 4] = [
                        (x as i32 + 1, y as i32, z as i32),
                        (x as i32 - 1, y as i32, z as i32),
                        (x as i32, y as i32, z as i32 + 1),
                        (x as i32, y as i32, z as i32 - 1),
                    ];
                    let new_hops_h = cell.hops_from_source.saturating_add(1);
                    let level_cap_h = bounded_level_cap(new_hops_h, cell.max_flow_dist);

                    for (nx, ny, nz) in neighbors {
                        // Face gating: no spreading through a rendered surface.
                        if !face_open(idx, nx - x as i32, 0, nz - z as i32) {
                            continue;
                        }
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
                                // Horizontal spread is same-Y → single-axis ±X/±Z crossing;
                                // use the hoisted neighbour ref, no HashMap probe.
                                let nbr = if dest_key == key_xp { nbr_xp }
                                    else if dest_key == key_xn { nbr_xn }
                                    else if dest_key == key_zp { nbr_zp }
                                    else { debug_assert_eq!(dest_key, key_zn); nbr_zn };
                                if let Some(nbr_grid) = nbr {
                                    let cap = nbr_grid.cell_capacity(tx, ty, tz);
                                    if cap >= MIN_LEVEL {
                                        let bi = tz * size * size + ty * size + tx;
                                        let dst_fill = nbr_grid.cells[bi].level / cap;
                                        let diff = src_fill - dst_fill;
                                        if diff > MIN_LEVEL {
                                            // Bounded-flow level cap on cross-chunk dst.
                                            let dst_existing = nbr_grid.cells[bi].level;
                                            let bounded_room = (level_cap_h - dst_existing).max(0.0);
                                            let transfer = (diff * horizontal_spread * src_capacity)
                                                .min(flow_rate)
                                                .min(new_cells[idx].level)
                                                .min(bounded_room);
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
                                                    dest_hops: new_hops_h,
                                                    dest_max_flow: cell.max_flow_dist,
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
                            // Bounded-flow level cap.
                            let bounded_space = (level_cap_h - new_cells[ni].level).max(0.0).min(dst_space);
                            let transfer = (diff * horizontal_spread * src_capacity)
                                .min(flow_rate)
                                .min(new_cells[idx].level) // prevent overdrain
                                .min(bounded_space);       // prevent overfill + bounded cap
                            if transfer > MIN_LEVEL {
                                if !is_source && !has_grace {
                                    new_cells[idx].level -= transfer;
                                }
                                new_cells[ni].level += transfer;
                                new_cells[ni].fluid_type = cell.fluid_type;
                                new_cells[ni].hops_from_source = new_hops_h;
                                new_cells[ni].max_flow_dist = cell.max_flow_dist;
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
                        if cell_cap[ai] > MIN_LEVEL && face_open(idx, 0, 1, 0) {
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
                            if weight_diff > 0.5
                                && !bounded_blocks_transfer(cell.hops_from_source, cell.max_flow_dist)
                            {
                                let above_space = (cell_cap[ai] - new_cells[ai].level).max(0.0);
                                let new_hops_u = cell.hops_from_source.saturating_add(1);
                                let cap_u = bounded_level_cap(new_hops_u, cell.max_flow_dist);
                                let bounded_space = (cap_u - new_cells[ai].level).max(0.0).min(above_space);
                                let push = (weight_diff * pressure_rate * 0.3)
                                    .min(bounded_space)
                                    .min(flow_rate)
                                    .min(new_cells[idx].level);
                                if push > MIN_LEVEL && !is_source && !has_grace {
                                    new_cells[idx].level -= push;
                                    new_cells[ai].level += push;
                                    new_cells[ai].fluid_type = cell.fluid_type;
                                    new_cells[ai].hops_from_source = new_hops_u;
                                    new_cells[ai].max_flow_dist = cell.max_flow_dist;
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
                // Lava is STICKY (2026-08-04, user request): stagnant lava
                // dribbles up to the orphan threshold get vacuumed into a
                // fuller neighbor instead of sitting apart and evaporating —
                // breakout blobs pull themselves together. Water keeps the
                // original sub-MIN-films-only rule; the stagnation gate stops
                // this from disturbing actively flowing lava.
                let is_sticky_lava = is_lava_tick
                    && new_cells[idx].fluid_type.is_lava()
                    && new_cells[idx].stagnant_ticks > 2
                    && !new_cells[idx].is_source
                    && level < ORPHAN_THRESHOLD;
                if level <= 0.0 || (level >= MIN_LEVEL && !is_sticky_lava) {
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
                    if new_cells[ni].level >= MIN_LEVEL && cell_cap[ni] > MIN_LEVEL
                        && face_open(idx, dx, dy, dz)
                    {
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
                if !pushed && is_sticky_lava {
                    // Range-2 crawl (2026-08-04): isolated specs have no wet
                    // neighbor at all — scan two steps out along open faces
                    // and MOVE one step toward the nearest wet lava cell.
                    // Specs converge into blobs over a few ticks instead of
                    // sitting apart as scattered dots until evaporation.
                    'crawl: for &(dx, dy, dz) in &consolidate_offsets {
                        let (sx, sy, sz) = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
                        let (fx, fy, fz) = (x as i32 + 2 * dx, y as i32 + 2 * dy, z as i32 + 2 * dz);
                        if sx < 0 || sx >= size as i32 || sy < 0 || sy >= size as i32
                            || sz < 0 || sz >= size as i32
                            || fx < 0 || fx >= size as i32 || fy < 0 || fy >= size as i32
                            || fz < 0 || fz >= size as i32 { continue; }
                        let si = sz as usize * size * size + sy as usize * size + sx as usize;
                        let fi = fz as usize * size * size + fy as usize * size + fx as usize;
                        if new_cells[fi].level >= MIN_LEVEL
                            && new_cells[fi].fluid_type.is_lava()
                            && cell_cap[si] > MIN_LEVEL
                            && face_open(idx, dx, dy, dz)
                            && face_open(si, dx, dy, dz)
                        {
                            let space = cell_cap[si] - new_cells[si].level;
                            if space > 0.0 {
                                let step = level.min(space);
                                new_cells[si].level += step;
                                new_cells[si].fluid_type = fluid_type;
                                new_cells[idx].level = 0.0;
                                pushed = true;
                                break 'crawl;
                            }
                        }
                    }
                }
                if !pushed && !is_sticky_lava {
                    new_cells[idx].level = 0.0; // no neighbor to absorb, evaporate
                }
            }
        }
    }

    // Clean up negative from overdrain and track has_fluid + has_lava +
    // has_sources. The latter two are recomputed here so per-tick passes
    // (`detect_lava_water_quench`, `regen_sources`) can skip whole chunks
    // that have no lava / no sources — paid for by a single fused pass
    // instead of a full N³ probe per chunk.
    let mut any_fluid = false;
    let mut any_lava = false;
    let mut any_source = false;
    for cell in &mut new_cells {
        if cell.level < MIN_LEVEL {
            cell.level = 0.0;
        }
        if cell.level >= MIN_LEVEL {
            any_fluid = true;
            if cell.fluid_type.is_lava() { any_lava = true; }
            if cell.is_source { any_source = true; }
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

                    // Source self-extinguish (2026-08-04): source cells never
                    // deduct on outflow (their own level is useless as a
                    // signal) — instead watch whether the fluid they emit
                    // ever ACCUMULATES. A source at steady state (sealed
                    // pool, filled basin, spring-fed river under its hop
                    // bound) has at least one passable neighbor holding a
                    // solid level. A source pumping into a sink (void below
                    // the world, pinhole into nowhere) has neighbors that
                    // never hold anything — everything falls away — and an
                    // eternal pump is exactly the bug-#215 world-flooder.
                    // Gate on the CURRENT tick's fluid type so the divisor'd
                    // lava cadence can't falsely grow/reset the streak on
                    // water ticks.
                    if new_cells[idx].is_source
                        && new_cells[idx].fluid_type.is_lava() == is_lava_tick
                    {
                        let mut max_neighbor_level = 0.0f32;
                        let mut has_passable_neighbor = false;
                        let offsets: [(i32,i32,i32); 6] =
                            [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)];
                        for &(dx,dy,dz) in &offsets {
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;
                            let nz = z as i32 + dz;
                            if nx < 0 || nx >= size as i32 || ny < 0 || ny >= size as i32
                                || nz < 0 || nz >= size as i32
                            {
                                continue; // cross-chunk unknown — ignore
                            }
                            let ni = nz as usize * size * size + ny as usize * size + nx as usize;
                            if cell_cap[ni] < MIN_LEVEL {
                                continue; // solid
                            }
                            has_passable_neighbor = true;
                            max_neighbor_level = max_neighbor_level.max(new_cells[ni].level);
                        }

                        if has_passable_neighbor
                            && max_neighbor_level < crate::cell::SOURCE_DRAIN_LEVEL
                        {
                            new_cells[idx].drain_ticks =
                                new_cells[idx].drain_ticks.saturating_add(1);
                            if new_cells[idx].drain_ticks
                                >= crate::cell::SOURCE_DRAIN_DEMOTE_TICKS
                            {
                                new_cells[idx].is_source = false;
                                new_cells[idx].drain_ticks = 0;
                                changed = true;
                            }
                        } else {
                            // Fully-encased sources idle harmlessly; a held
                            // neighbor means steady state (or honest filling).
                            new_cells[idx].drain_ticks = 0;
                        }
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

        // Pre-compute drain deltas (positive = cell lost water this tick).
        // Reuses the per-grid scratch buffer (taken at the top of the fn).
        drain_scratch.clear();
        drain_scratch.resize(total, 0.0);
        let old_cells = &chunks.get(&key).unwrap().cells;
        for idx in 0..total {
            drain_scratch[idx] = (old_cells[idx].level - new_cells[idx].level).max(0.0);
        }
        let drain_delta = &mut drain_scratch;

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

    // Swap buffer + return scratches to the grid for the next tick.
    // We swap rather than overwrite so the OLD `cells` allocation becomes
    // the next tick's scratch — zero new heap traffic in steady state.
    if let Some(grid) = chunks.get_mut(&key) {
        if changed {
            std::mem::swap(&mut grid.cells, &mut new_cells);
            grid.dirty = true;
        }
        grid.has_fluid = any_fluid;
        grid.has_lava = any_lava;
        grid.has_sources = any_source;
        grid.scratch_cells = new_cells;
        grid.scratch_weights = fluid_weight;
        grid.scratch_drain = drain_scratch;
    }

    (changed, cross_transfers)
}
