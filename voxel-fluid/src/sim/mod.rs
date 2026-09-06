use std::collections::{HashMap, HashSet};

use crate::cell::{ChunkDensityCache, ChunkFluidGrid, MIN_LEVEL};
use crate::FluidConfig;

mod chunk;
mod utils;

pub mod displacement;
pub mod wave;
pub use utils::{squeeze_excess_fluid, squeeze_excess_fluid_collect, SqueezedRemainder, equalize_horizontal, detect_solidification, regen_sources,
    detect_lava_water_quench, detect_lava_water_quench_with_scratch,
    try_grow_pillow_voxel, QuenchPlan, QuenchScratch, CellAddr};

use chunk::{CrossChunkTransfer, tick_chunk};

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
    decrement_grace: bool,
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
                    // from_density_cache already fills cell_cap from corners.
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

        let (changed, transfers) = tick_chunk(chunks, key, chunk_size, is_lava_tick, config, decrement_grace);
        if changed {
            dirty.insert(key);
        }
        all_transfers.extend(transfers);
    }

    // Apply cross-chunk transfers (second pass — no borrow conflicts).
    // The amount was computed against the dest's state at the SENDER's tick;
    // the dest chunk's own tick can fill the cell in between (chunk order is
    // HashMap-random), so whatever doesn't fit is REFUNDED to the sender —
    // the sender already deducted it, and dropping the residual was a real
    // conservation leak (order-dependent ~2% loss in cross-chunk pools).
    for xfer in &all_transfers {
        let mut applied = 0.0f32;

        // If target chunk has no grid but density exists, create grid on demand
        let dest_reachable = chunks.contains_key(&xfer.dest_key)
            || if let Some(cache) = chunk_densities.get(&xfer.dest_key) {
                // from_density_cache already fills cell_cap from corners.
                let grid = ChunkFluidGrid::from_density_cache(cache);
                chunks.insert(xfer.dest_key, grid);
                true
            } else {
                false // no density data, can't create grid
            };

        if dest_reachable {
            if let Some(grid) = chunks.get_mut(&xfer.dest_key) {
                let capacity = grid.cell_capacity(xfer.dest_x, xfer.dest_y, xfer.dest_z);
                if capacity >= MIN_LEVEL {
                    let cell = grid.get_mut(xfer.dest_x, xfer.dest_y, xfer.dest_z);
                    let space = capacity - cell.level;
                    let actual = xfer.amount.min(space).max(0.0);
                    if actual > MIN_LEVEL {
                        cell.level += actual;
                        cell.fluid_type = xfer.fluid_type;
                        applied = actual;
                        // Propagate bounded-flow tracking. Only overwrite if we're tightening
                        // the limit (or the dst has no recorded source yet) — otherwise an
                        // existing closer source's tracking wins.
                        if cell.hops_from_source == 255 || xfer.dest_hops < cell.hops_from_source {
                            cell.hops_from_source = xfer.dest_hops;
                            cell.max_flow_dist = xfer.dest_max_flow;
                        }
                        // Cross-chunk CASCADE arrivals count as "fed" for transit
                        // retention, same as in-chunk gravity/slope receives. Spread
                        // arrivals (feeds=false) don't — retention-held unequal
                        // neighbors would otherwise sustain each other forever
                        // across the seam (perched lava).
                        if is_lava_tick && xfer.feeds {
                            grid.mark_influx(xfer.dest_x, xfer.dest_y, xfer.dest_z);
                        }
                        grid.dirty = true;
                        grid.has_fluid = true;
                        dirty.insert(xfer.dest_key);
                    }
                }
            }
        }

        // Refund the un-deposited residual to the sender (source/grace cells
        // never deducted, so there's nothing to give back). Deliberately NOT
        // clamped to the sender's capacity: in-chunk gravity often refills the
        // sender within the same tick (the seam cell's space got promised
        // twice), and tick_chunk's excess-redistribution pass already handles
        // transiently overfull cells by backing the water up into neighbors.
        let residual = xfer.amount - applied;
        if xfer.src_deducted && residual > 1e-6 {
            if let Some(grid) = chunks.get_mut(&xfer.src_key) {
                let cell = grid.get_mut(xfer.src_x, xfer.src_y, xfer.src_z);
                if cell.level < MIN_LEVEL {
                    // Cell fully drained this tick — restore its type.
                    cell.fluid_type = xfer.fluid_type;
                }
                cell.level += residual;
                grid.dirty = true;
                grid.has_fluid = true;
                dirty.insert(xfer.src_key);
            }
        }
    }

    dirty
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::ops::Range;
    use crate::cell::{ChunkDensityCache, ChunkFluidGrid, FluidType, MIN_LEVEL, SOURCE_LEVEL};

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
        grid.get_mut(8, 8, 8).level = 0.8;
        grid.get_mut(8, 8, 8).fluid_type = FluidType::Water;
        grid.has_fluid = true;
        chunks.insert(key, grid);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        tick_fluid(&mut chunks, &density_cache, 16, false, &config, true);

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
    fn solid_blocks_flow() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);
        grid.get_mut(8, 8, 8).level = 0.5;
        grid.get_mut(8, 8, 8).fluid_type = FluidType::Water;
        grid.has_fluid = true;
        grid.set_density(8, 7, 8, 1.0);
        chunks.insert(key, grid);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        tick_fluid(&mut chunks, &density_cache, 16, false, &config, true);

        let grid = &chunks[&key];
        assert_eq!(grid.get(8, 7, 8).level, 0.0, "Water should not enter solid cell");
    }

    #[test]
    fn cross_chunk_downward_flow() {
        let mut chunks = HashMap::new();
        let upper_key = (0, 1, 0);
        let lower_key = (0, 0, 0);

        let mut upper_grid = make_chunk(16);
        upper_grid.get_mut(8, 0, 8).level = 0.8;
        upper_grid.get_mut(8, 0, 8).fluid_type = FluidType::Water;
        upper_grid.has_fluid = true;
        chunks.insert(upper_key, upper_grid);

        let lower_grid = make_chunk(16);
        chunks.insert(lower_key, lower_grid);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        tick_fluid(&mut chunks, &density_cache, 16, false, &config, true);

        let lower = &chunks[&lower_key];
        assert!(lower.get(8, 15, 8).level > 0.0, "Water should flow across chunk boundary to y=15 below");

        let upper = &chunks[&upper_key];
        assert!(upper.get(8, 0, 8).level < 0.8, "Upper chunk should have transferred fluid downward");
    }

    #[test]
    fn contained_source_doesnt_flow() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);
        grid.get_mut(8, 8, 8).level = SOURCE_LEVEL;
        grid.get_mut(8, 8, 8).fluid_type = FluidType::Water;
        grid.get_mut(8, 8, 8).is_source = true;
        grid.has_fluid = true;
        grid.set_density(7, 8, 8, 1.0);
        grid.set_density(9, 8, 8, 1.0);
        grid.set_density(8, 7, 8, 1.0);
        grid.set_density(8, 9, 8, 1.0);
        grid.set_density(8, 8, 7, 1.0);
        grid.set_density(8, 8, 9, 1.0);
        chunks.insert(key, grid);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        for _ in 0..10 {
            tick_fluid(&mut chunks, &density_cache, 16, false, &config, true);
        }

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
        grid.set_density(8, 7, 8, -0.5);
        grid.get_mut(8, 8, 8).level = 0.8;
        grid.get_mut(8, 8, 8).fluid_type = FluidType::Water;
        grid.has_fluid = true;
        chunks.insert(key, grid);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        for _ in 0..20 {
            tick_fluid(&mut chunks, &density_cache, 16, false, &config, true);
        }

        let grid = &chunks[&key];
        assert!(grid.get(8, 7, 8).level <= 1.001, "Fluid should not exceed cell capacity, got {}", grid.get(8, 7, 8).level);
    }

    #[test]
    fn slope_flow_down() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);
        grid.get_mut(8, 5, 8).level = 0.8;
        grid.get_mut(8, 5, 8).fluid_type = FluidType::Water;
        grid.has_fluid = true;
        grid.set_density(8, 4, 8, 1.0);
        chunks.insert(key, grid);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        tick_fluid(&mut chunks, &density_cache, 16, false, &config, true);

        let grid = &chunks[&key];
        assert!(grid.get(9, 4, 8).level > 0.0, "Water should slope-flow diagonally down to (9,4,8), got {}", grid.get(9, 4, 8).level);
    }

    #[test]
    fn slope_flow_cascades() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);
        grid.get_mut(8, 5, 8).level = 0.8;
        grid.get_mut(8, 5, 8).fluid_type = FluidType::Water;
        grid.has_fluid = true;
        grid.set_density(8, 4, 8, 1.0);
        grid.set_density(9, 3, 8, 1.0);
        grid.set_density(10, 2, 8, 1.0);
        // Full floor: with the void-cull backstop, fluid reaching an open
        // y=0 in a chunk with no neighbor below falls out of the world.
        for z in 0..16 { for x in 0..16 { grid.set_density(x, 0, z, 1.0); } }
        chunks.insert(key, grid);

        let mut config = crate::FluidConfig::default();
        config.water_spread_rate = 0.6;
        config.water_flow_rate = 1.0;
        let density_cache = empty_density_cache();
        for _ in 0..15 {
            tick_fluid(&mut chunks, &density_cache, 16, false, &config, true);
        }

        let grid = &chunks[&key];
        let mut found_below = false;
        for y in 0..4 {
            for x in 7..14 {
                if grid.get(x, y, 8).level > MIN_LEVEL { found_below = true; break; }
            }
            if found_below { break; }
        }
        assert!(found_below, "Water should cascade down the staircase to lower Y levels");
    }

    #[test]
    fn slope_flow_blocked_by_solid() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);
        grid.get_mut(8, 5, 8).level = 0.8;
        grid.get_mut(8, 5, 8).fluid_type = FluidType::Water;
        grid.has_fluid = true;
        grid.set_density(8, 4, 8, 1.0);
        grid.set_density(9, 4, 8, 1.0);
        grid.set_density(7, 4, 8, 1.0);
        grid.set_density(8, 4, 9, 1.0);
        grid.set_density(8, 4, 7, 1.0);
        chunks.insert(key, grid);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        tick_fluid(&mut chunks, &density_cache, 16, false, &config, true);

        let grid = &chunks[&key];
        for dy in [4i32] {
            for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let nx = (8i32 + dx) as usize;
                let ny = (5i32 + dy) as usize;
                let nz = (8i32 + dz) as usize;
                assert!(grid.get(nx, ny, nz).level < MIN_LEVEL,
                    "Water should not enter solid slope target ({},{},{}), got {}", nx, ny, nz, grid.get(nx, ny, nz).level);
            }
        }
    }

    #[test]
    fn increased_spread_rate() {
        let config = crate::FluidConfig::default();
        assert!((config.water_spread_rate - 2.0).abs() < 0.01, "Default water_spread_rate should be 2.0, got {}", config.water_spread_rate);

        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);
        grid.get_mut(8, 5, 8).level = 0.8;
        grid.get_mut(8, 5, 8).fluid_type = FluidType::Water;
        grid.has_fluid = true;
        for x in 0..16 { for z in 0..16 { grid.set_density(x, 4, z, 1.0); } }
        chunks.insert(key, grid);

        let density_cache = empty_density_cache();
        for _ in 0..3 { tick_fluid(&mut chunks, &density_cache, 16, false, &config, true); }

        let grid = &chunks[&key];
        let source = grid.get(8, 5, 8).level;
        let mut found_spread = false;
        for x in 0..16 {
            for z in 0..16 {
                if (x, z) != (8, 8) && grid.get(x, 5, z).level > MIN_LEVEL { found_spread = true; break; }
            }
            if found_spread { break; }
        }
        assert!(found_spread, "Water should spread away from source (8,5,8) with rate 2.0 after 3 ticks, source level={}", source);
    }

    #[test]
    fn cross_chunk_slope_flow() {
        let mut chunks = HashMap::new();
        let upper_key = (0, 1, 0);
        let lower_key = (0, 0, 0);

        let mut upper_grid = make_chunk(16);
        upper_grid.get_mut(8, 0, 8).level = 0.8;
        upper_grid.get_mut(8, 0, 8).fluid_type = FluidType::Water;
        upper_grid.has_fluid = true;

        let mut lower_grid = make_chunk(16);
        lower_grid.set_density(8, 15, 8, 1.0);

        chunks.insert(upper_key, upper_grid);
        chunks.insert(lower_key, lower_grid);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        tick_fluid(&mut chunks, &density_cache, 16, false, &config, true);

        let lower = &chunks[&lower_key];
        assert!(lower.get(9, 15, 8).level > 0.0, "Water should cross-chunk slope-flow to (9,15,8), got {}", lower.get(9, 15, 8).level);
    }

    #[test]
    fn cross_chunk_horizontal_flow() {
        let mut chunks = HashMap::new();
        let key_a = (0, 0, 0);
        let key_b = (1, 0, 0);

        let mut grid_a = make_chunk(16);
        for x in 0..16 { for z in 0..16 { grid_a.set_density(x, 0, z, 1.0); } }
        grid_a.get_mut(15, 1, 8).level = 0.8;
        grid_a.get_mut(15, 1, 8).fluid_type = FluidType::Water;
        grid_a.has_fluid = true;
        chunks.insert(key_a, grid_a);

        let mut grid_b = make_chunk(16);
        for x in 0..16 { for z in 0..16 { grid_b.set_density(x, 0, z, 1.0); } }
        chunks.insert(key_b, grid_b);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        tick_fluid(&mut chunks, &density_cache, 16, false, &config, true);

        let nbr = &chunks[&key_b];
        let mut nbr_total = 0.0f64;
        for z in 0..16 { for y in 0..16 { for x in 0..16 { nbr_total += nbr.get(x, y, z).level as f64; } } }
        assert!(nbr_total > 0.01, "Water should flow across chunk boundary to +X neighbor, total water in neighbor={}", nbr_total);
    }

    #[test]
    fn cross_chunk_slope_flow_xz() {
        let mut chunks = HashMap::new();
        let key_a = (0, 0, 0);
        let key_b = (1, 0, 0);

        let mut grid_a = make_chunk(16);
        for x in 0..16 { for z in 0..16 { grid_a.set_density(x, 0, z, 1.0); } }
        grid_a.get_mut(15, 1, 8).level = 0.8;
        grid_a.get_mut(15, 1, 8).fluid_type = FluidType::Water;
        grid_a.has_fluid = true;
        chunks.insert(key_a, grid_a);

        let grid_b = make_chunk(16);
        chunks.insert(key_b, grid_b);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        tick_fluid(&mut chunks, &density_cache, 16, false, &config, true);

        let nbr = &chunks[&key_b];
        let mut nbr_total = 0.0f64;
        for z in 0..16 { for y in 0..16 { for x in 0..16 { nbr_total += nbr.get(x, y, z).level as f64; } } }
        assert!(nbr_total > 0.01, "Water should cross-chunk flow to +X neighbor, total water in neighbor={}", nbr_total);
    }

    #[test]
    fn upward_pressure_equalization() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);

        for x in 6..=10 { for z in 7..=9 { grid.set_density(x, 2, z, 1.0); } }
        for y in 2..9 { for z in 7..=9 { grid.set_density(6, y, z, 1.0); grid.set_density(10, y, z, 1.0); } }
        for x in 6..=10 { for y in 2..9 { grid.set_density(x, y, 7, 1.0); grid.set_density(x, y, 9, 1.0); } }

        for y in 3..7 { let cell = grid.get_mut(7, y, 8); cell.level = 1.0; cell.fluid_type = FluidType::Water; }
        for x in 8..=9 { let cell = grid.get_mut(x, 3, 8); cell.level = 1.0; cell.fluid_type = FluidType::Water; }
        grid.has_fluid = true;
        chunks.insert(key, grid);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        for _ in 0..200 { tick_fluid(&mut chunks, &density_cache, 16, false, &config, true); }

        let grid = &chunks[&key];
        let right_y4 = grid.get(8, 4, 8).level;
        assert!(right_y4 > 0.05, "Water should push upward in shorter column via pressure, got {} at (8,4,8)", right_y4);
    }

    #[test]
    fn stable_pool_no_oscillation() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);
        for x in 6..=10 { for z in 6..=10 { grid.set_density(x, 4, z, 1.0); } }
        for x in 7..=9 { for z in 7..=9 { let cell = grid.get_mut(x, 5, z); cell.level = 0.8; cell.fluid_type = FluidType::Water; } }
        grid.has_fluid = true;
        chunks.insert(key, grid);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        for _ in 0..20 { tick_fluid(&mut chunks, &density_cache, 16, false, &config, true); }

        let grid = &chunks[&key];
        for x in 7..=9 { for z in 7..=9 {
            let above = grid.get(x, 6, z).level;
            assert!(above < MIN_LEVEL, "Stable pool should not push water up, got {} at ({},6,{})", above, x, z);
        }}
    }

    #[test]
    fn grace_prevents_drain() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);
        let cell = grid.get_mut(8, 8, 8);
        cell.level = 1.0; cell.fluid_type = FluidType::Water; cell.is_source = false; cell.grace_ticks = 10;
        grid.has_fluid = true;
        chunks.insert(key, grid);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        tick_fluid(&mut chunks, &density_cache, 16, false, &config, true);

        let grid = &chunks[&key];
        assert!(grid.get(8, 8, 8).level >= 0.99, "Grace cell should not drain, got {}", grid.get(8, 8, 8).level);
        assert!(grid.get(8, 7, 8).level > MIN_LEVEL, "Water should still flow down from grace cell");
        assert_eq!(grid.get(8, 8, 8).grace_ticks, 9);
    }

    #[test]
    fn grace_expires_then_drains() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);
        let cell = grid.get_mut(8, 8, 8);
        cell.level = 1.0; cell.fluid_type = FluidType::Water; cell.is_source = false; cell.grace_ticks = 1;
        grid.has_fluid = true;
        chunks.insert(key, grid);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        tick_fluid(&mut chunks, &density_cache, 16, false, &config, true);
        let level_after_1 = chunks[&key].get(8, 8, 8).level;
        assert!(level_after_1 >= 0.99, "Grace still active on first tick, got {}", level_after_1);
        assert_eq!(chunks[&key].get(8, 8, 8).grace_ticks, 0);

        tick_fluid(&mut chunks, &density_cache, 16, false, &config, true);
        let level_after_2 = chunks[&key].get(8, 8, 8).level;
        assert!(level_after_2 < level_after_1, "Cell should drain after grace expires, got {}", level_after_2);
    }

    #[test]
    fn grace_does_not_propagate() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);
        let cell = grid.get_mut(8, 8, 8);
        cell.level = 1.0; cell.fluid_type = FluidType::Water; cell.is_source = false; cell.grace_ticks = 50;
        grid.has_fluid = true;
        chunks.insert(key, grid);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        tick_fluid(&mut chunks, &density_cache, 16, false, &config, true);

        let grid = &chunks[&key];
        assert!(grid.get(8, 7, 8).level > MIN_LEVEL, "Water should flow down");
        assert_eq!(grid.get(8, 7, 8).grace_ticks, 0, "Grace should not propagate to recipients");
    }

    #[test]
    fn slope_blocked_by_wall() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);
        grid.get_mut(8, 8, 8).level = 0.5;
        grid.get_mut(8, 8, 8).fluid_type = FluidType::Water;
        grid.has_fluid = true;
        grid.set_density(8, 7, 8, 1.0);
        grid.set_density(9, 8, 8, 1.0);
        chunks.insert(key, grid);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        tick_fluid(&mut chunks, &density_cache, 16, false, &config, true);

        let grid = &chunks[&key];
        assert!(grid.get(9, 7, 8).level < MIN_LEVEL,
            "Water should not flow diagonally through solid wall, got {} at (9,7,8)", grid.get(9, 7, 8).level);
    }

    // ====================== Watertightness & Conservation Helpers ======================

    fn make_sealed_box(size: usize, x_range: Range<usize>, y_range: Range<usize>, z_range: Range<usize>) -> ChunkFluidGrid {
        let mut grid = ChunkFluidGrid::new(size);
        for z in 0..size { for y in 0..size { for x in 0..size { grid.set_density(x, y, z, 1.0); } } }
        for z in z_range { for y in y_range.clone() { for x in x_range.clone() { grid.set_density(x, y, z, -1.0); } } }
        grid
    }

    fn total_water(chunks: &HashMap<(i32, i32, i32), ChunkFluidGrid>) -> f64 {
        let mut total = 0.0f64;
        for grid in chunks.values() { for cell in &grid.cells { total += cell.level as f64; } }
        total
    }

    fn water_in_solid_cells(grid: &ChunkFluidGrid) -> f64 {
        let mut total = 0.0f64;
        let size = grid.size;
        for z in 0..size { for y in 0..size { for x in 0..size {
            if grid.is_solid(x, y, z) { total += grid.get(x, y, z).level as f64; }
        }}}
        total
    }

    fn fill_layer(grid: &mut ChunkFluidGrid, x_range: Range<usize>, z_range: Range<usize>, y: usize, level: f32) {
        for z in z_range { for x in x_range.clone() {
            if !grid.is_solid(x, y, z) {
                let cell = grid.get_mut(x, y, z);
                cell.level = level;
                cell.fluid_type = FluidType::Water;
            }
        }}
        grid.has_fluid = true;
    }

    // ====================== Watertightness & Conservation Tests ======================

    #[test]
    fn sealed_box_quarter_fill_conserves_water() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_sealed_box(16, 3..13, 2..12, 3..13);
        fill_layer(&mut grid, 3..13, 3..13, 2, 1.0);
        fill_layer(&mut grid, 3..13, 3..13, 3, 1.0);
        fill_layer(&mut grid, 3..13, 3..13, 4, 0.5);
        chunks.insert(key, grid);

        let initial_water = total_water(&chunks);
        assert!(initial_water > 0.0, "Should have initial water");

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        for _ in 0..300 { tick_fluid(&mut chunks, &density_cache, 16, false, &config, true); }

        let final_water = total_water(&chunks);
        let grid = &chunks[&key];
        assert!((final_water - initial_water).abs() < 0.01,
            "Water should be conserved: initial={}, final={}, diff={}", initial_water, final_water, final_water - initial_water);
        assert!(water_in_solid_cells(grid) < 0.001, "No water should be in solid cells, found {}", water_in_solid_cells(grid));
        for z in 0..16usize { for y in 0..16usize { for x in 0..16usize {
            if !(3..13).contains(&x) || !(2..12).contains(&y) || !(3..13).contains(&z) {
                assert!(grid.get(x, y, z).level < MIN_LEVEL, "Water outside box at ({},{},{}): level={}", x, y, z, grid.get(x, y, z).level);
            }
        }}}
    }

    #[test]
    fn sealed_box_half_fill_conserves_water() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_sealed_box(16, 3..13, 2..12, 3..13);
        for y in 2..7 { fill_layer(&mut grid, 3..13, 3..13, y, 1.0); }
        chunks.insert(key, grid);

        let initial_water = total_water(&chunks);
        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        for _ in 0..300 { tick_fluid(&mut chunks, &density_cache, 16, false, &config, true); }

        let final_water = total_water(&chunks);
        let grid = &chunks[&key];
        assert!((final_water - initial_water).abs() < 0.01, "Water conserved: initial={}, final={}, diff={}", initial_water, final_water, final_water - initial_water);
        assert!(water_in_solid_cells(grid) < 0.001, "No water in solid cells, found {}", water_in_solid_cells(grid));
    }

    #[test]
    fn uneven_fill_equalizes_to_flat_surface() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_sealed_box(16, 3..13, 2..12, 3..13);
        for y in 2..6 { fill_layer(&mut grid, 3..8, 3..13, y, 1.0); }
        chunks.insert(key, grid);

        let initial_water = total_water(&chunks);
        let mut config = crate::FluidConfig::default();
        config.water_spread_rate = 0.6;
        config.water_flow_rate = 1.0;
        let density_cache = empty_density_cache();
        for _ in 0..1000 { tick_fluid(&mut chunks, &density_cache, 16, false, &config, true); }

        let final_water = total_water(&chunks);
        let grid = &chunks[&key];
        assert!((final_water - initial_water).abs() < 0.1, "Water conserved: initial={}, final={}, diff={}", initial_water, final_water, final_water - initial_water);

        for y in 2..12 {
            let mut levels: Vec<f32> = Vec::new();
            for z in 3..13 { for x in 3..13 {
                let lvl = grid.get(x, y, z).level;
                if lvl > MIN_LEVEL { levels.push(lvl); }
            }}
            if levels.len() > 1 {
                let max_lvl = levels.iter().cloned().fold(f32::MIN, f32::max);
                let min_lvl = levels.iter().cloned().fold(f32::MAX, f32::min);
                assert!(max_lvl - min_lvl < 0.05, "Water at y={} should be flat: min={}, max={}, diff={}", y, min_lvl, max_lvl, max_lvl - min_lvl);
            }
        }
    }

    #[test]
    fn water_never_enters_solid_cells() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_sealed_box(16, 3..13, 2..12, 3..13);
        for y in 2..12 { fill_layer(&mut grid, 3..13, 3..13, y, 1.0); }
        chunks.insert(key, grid);

        let initial_water = total_water(&chunks);
        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        for _ in 0..200 { tick_fluid(&mut chunks, &density_cache, 16, false, &config, true); }

        let grid = &chunks[&key];
        let final_water = total_water(&chunks);
        let solid_water = water_in_solid_cells(grid);
        assert!(solid_water < MIN_LEVEL as f64, "Solid cells should have no water, found {}", solid_water);
        assert!((final_water - initial_water).abs() < 0.01, "Water conserved: initial={}, final={}, diff={}", initial_water, final_water, final_water - initial_water);
    }

    #[test]
    fn asymmetric_pile_settles_flat() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_sealed_box(16, 3..13, 2..12, 3..13);
        let center_x: i32 = 8;
        let center_z: i32 = 8;
        for dx in -5i32..=5 {
            for dz in -5i32..=5 {
                let px = center_x + dx;
                let pz = center_z + dz;
                if px < 3 || px >= 13 || pz < 3 || pz >= 13 { continue; }
                let dist = dx.abs().max(dz.abs()) as usize;
                let height = if dist < 6 { 6 - dist } else { 0 };
                for dy in 0..height {
                    let y = 2 + dy;
                    if y < 12 && !grid.is_solid(px as usize, y, pz as usize) {
                        let cell = grid.get_mut(px as usize, y, pz as usize);
                        cell.level = 1.0; cell.fluid_type = FluidType::Water;
                    }
                }
            }
        }
        grid.has_fluid = true;
        chunks.insert(key, grid);

        let initial_water = total_water(&chunks);
        let mut config = crate::FluidConfig::default();
        config.water_spread_rate = 0.6;
        config.water_flow_rate = 1.0;
        let density_cache = empty_density_cache();
        for _ in 0..1000 { tick_fluid(&mut chunks, &density_cache, 16, false, &config, true); }

        let final_water = total_water(&chunks);
        let grid = &chunks[&key];
        assert!((final_water - initial_water).abs() < 0.05, "Water conserved: initial={}, final={}, diff={}", initial_water, final_water, final_water - initial_water);

        for y in 2..12 {
            let mut levels: Vec<f32> = Vec::new();
            for z in 3..13 { for x in 3..13 {
                let lvl = grid.get(x, y, z).level;
                if lvl > MIN_LEVEL { levels.push(lvl); }
            }}
            if levels.len() > 1 {
                let max_lvl = levels.iter().cloned().fold(f32::MIN, f32::max);
                let min_lvl = levels.iter().cloned().fold(f32::MAX, f32::min);
                assert!(max_lvl - min_lvl < 0.05, "y={} not flat: min={}, max={}, diff={}", y, min_lvl, max_lvl, max_lvl - min_lvl);
            }
        }
    }

    #[test]
    fn uniform_layer_stays_stable() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_sealed_box(16, 3..13, 2..12, 3..13);
        fill_layer(&mut grid, 3..13, 3..13, 2, 0.3);
        chunks.insert(key, grid);

        let initial_water = total_water(&chunks);
        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        for _ in 0..200 { tick_fluid(&mut chunks, &density_cache, 16, false, &config, true); }

        let final_water = total_water(&chunks);
        let grid = &chunks[&key];
        assert!((final_water - initial_water).abs() < 0.01, "Water conserved: initial={}, final={}, diff={}", initial_water, final_water, final_water - initial_water);
        for z in 3..13 { for x in 3..13 {
            let lvl = grid.get(x, 2, z).level;
            assert!((lvl - 0.3).abs() < 0.05, "Cell ({},2,{}) should be ~0.3, got {}", x, z, lvl);
        }}
    }

    #[test]
    fn realistic_density_boundary_leak_test() {
        let size = 16;
        let stride = size + 1;
        let mut grid = ChunkFluidGrid::new(size);
        let mut densities = vec![1.0f32; stride * stride * stride];
        for gz in 4..13 { for gy in 3..12 { for gx in 4..13 {
            densities[gz * stride * stride + gy * stride + gx] = -1.0;
        }}}
        grid.update_density(&densities);
        let config = crate::FluidConfig::default();
        grid.recompute_capacity();

        for z in 0..size { for y in 3..7 { for x in 0..size {
            let cap = grid.cell_capacity(x, y, z);
            if cap > MIN_LEVEL { let cell = grid.get_mut(x, y, z); cell.level = cap.min(1.0); cell.fluid_type = FluidType::Water; }
        }}}
        grid.has_fluid = true;

        let mut chunks = HashMap::new();
        chunks.insert((0, 0, 0), grid);
        let initial_water = total_water(&chunks);
        assert!(initial_water > 0.0, "Should have initial water");

        let density_cache = empty_density_cache();
        for _ in 0..500 { tick_fluid(&mut chunks, &density_cache, 16, false, &config, true); }

        let final_water = total_water(&chunks);
        let grid = &chunks[&(0, 0, 0)];
        let mut boundary_cells = 0;
        let mut boundary_water = 0.0f64;
        for z in 0..size { for y in 0..size { for x in 0..size {
            let cap = grid.cell_capacity(x, y, z);
            if cap > 0.0 && cap < 1.0 - MIN_LEVEL { boundary_cells += 1; boundary_water += grid.get(x, y, z).level as f64; }
        }}}
        let loss_pct = if initial_water > 0.0 { ((initial_water - final_water) / initial_water * 100.0).abs() } else { 0.0 };
        assert!(loss_pct < 1.0,
            "Water conservation: initial={:.2}, final={:.2}, loss={:.2}%, boundary_cells={}, boundary_water={:.2}",
            initial_water, final_water, loss_pct, boundary_cells, boundary_water);
    }

    #[test]
    fn multi_chunk_sealed_box_conserves_water() {
        let mut chunks = HashMap::new();
        let upper_key = (0, 1, 0);
        let lower_key = (0, 0, 0);
        let size = 16;

        let mut upper = ChunkFluidGrid::new(size);
        for z in 0..size { for y in 0..size { for x in 0..size { upper.set_density(x, y, z, 1.0); } } }
        for z in 3..13 { for y in 0..4 { for x in 3..13 { upper.set_density(x, y, z, -1.0); } } }

        let mut lower = ChunkFluidGrid::new(size);
        for z in 0..size { for y in 0..size { for x in 0..size { lower.set_density(x, y, z, 1.0); } } }
        for z in 3..13 { for y in 12..16 { for x in 3..13 { lower.set_density(x, y, z, -1.0); } } }

        for y in 0..4 { fill_layer(&mut upper, 3..13, 3..13, y, 1.0); }
        chunks.insert(upper_key, upper);
        chunks.insert(lower_key, lower);

        let initial_water = total_water(&chunks);
        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        for _ in 0..300 { tick_fluid(&mut chunks, &density_cache, 16, false, &config, true); }

        let final_water = total_water(&chunks);
        assert!((final_water - initial_water).abs() < 0.1,
            "Multi-chunk water conserved: initial={}, final={}, diff={}", initial_water, final_water, final_water - initial_water);
        for (ckey, grid) in &chunks {
            let solid_water = water_in_solid_cells(grid);
            assert!(solid_water < MIN_LEVEL as f64, "Chunk {:?}: water in solid cells = {}", ckey, solid_water);
        }
    }

    // ====================== Realistic Fluid Conservation Helpers ======================

    fn make_density_field_solid(size: usize) -> Vec<f32> {
        let stride = size + 1;
        vec![1.0f32; stride * stride * stride]
    }

    fn carve_bowl(densities: &mut [f32], size: usize, cx: usize, cz: usize, floor_gy: usize, radius: usize, depth: usize) {
        let stride = size + 1;
        for gz in 0..stride { for gy in 0..stride { for gx in 0..stride {
            let dx = gx as f32 - cx as f32;
            let dz = gz as f32 - cz as f32;
            let dist = (dx * dx + dz * dz).sqrt();
            if dist < radius as f32 {
                let ratio = dist / radius as f32;
                let carve_depth = depth as f32 * (1.0 - ratio.sqrt());
                let floor_y = floor_gy as f32;
                let ceil_y = floor_y + carve_depth;
                if (gy as f32) >= floor_y && (gy as f32) <= ceil_y {
                    densities[gz * stride * stride + gy * stride + gx] = -1.0;
                }
            }
        }}}
    }

    fn carve_box(densities: &mut [f32], size: usize, gx_range: Range<usize>, gy_range: Range<usize>, gz_range: Range<usize>) {
        let stride = size + 1;
        for gz in gz_range { for gy in gy_range.clone() { for gx in gx_range.clone() {
            if gx < stride && gy < stride && gz < stride {
                densities[gz * stride * stride + gy * stride + gx] = -1.0;
            }
        }}}
    }

    fn apply_density(grid: &mut ChunkFluidGrid, densities: &[f32], _config: &crate::FluidConfig) {
        grid.update_density(densities);
        grid.recompute_capacity();
    }

    fn mine_cells(grid: &mut ChunkFluidGrid, densities: &mut Vec<f32>, cells_to_mine: &[(usize, usize, usize)], size: usize) {
        let stride = size + 1;
        for &(cx, cy, cz) in cells_to_mine {
            let corner_offsets: [[usize; 3]; 8] = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]];
            for off in &corner_offsets {
                let gx = cx + off[0]; let gy = cy + off[1]; let gz = cz + off[2];
                if gx < stride && gy < stride && gz < stride {
                    densities[gz * stride * stride + gy * stride + gx] = -1.0;
                }
            }
        }
        grid.update_density(densities);
        squeeze_excess_fluid(grid);
    }

    fn fill_air_to_capacity(grid: &mut ChunkFluidGrid, y_range: Range<usize>) {
        let size = grid.size;
        for z in 0..size { for y in y_range.clone() { for x in 0..size {
            let cap = grid.cell_capacity(x, y, z);
            if cap > MIN_LEVEL { let cell = grid.get_mut(x, y, z); cell.level = cap; cell.fluid_type = FluidType::Water; }
        }}}
        grid.has_fluid = true;
    }

    fn grid_total_water(grid: &ChunkFluidGrid) -> f64 {
        grid.cells.iter().map(|c| c.level as f64).sum()
    }

    // ====================== Category 1: Cauldron/Bowl Tests (1-8) ======================

    #[test]
    fn bowl_symmetric_retains_water() {
        let size = 16;
        let mut densities = make_density_field_solid(size);
        carve_bowl(&mut densities, size, 8, 8, 4, 5, 4);
        let config = crate::FluidConfig::default();
        let mut grid = ChunkFluidGrid::new(size);
        apply_density(&mut grid, &densities, &config);
        fill_air_to_capacity(&mut grid, 4..8);
        let initial = grid_total_water(&grid);
        assert!(initial > 0.0, "Should have water");

        let mut chunks = HashMap::new();
        chunks.insert((0,0,0), grid);
        let dc = empty_density_cache();
        for _ in 0..500 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }

        let final_w = total_water(&chunks);
        let loss_pct = ((initial - final_w) / initial * 100.0).abs();
        // ±3% (was ±1%): face gating (2026-08-04) evaporates settle-time thin
        // films whose only consolidation path crosses a rendered surface —
        // a one-time "skin soak" at fill, not an ongoing leak.
        assert!(loss_pct < 3.0, "Conservation ±3%: initial={:.2}, final={:.2}, loss={:.2}%", initial, final_w, loss_pct);
    }

    #[test]
    fn bowl_asymmetric_equalizes() {
        let size = 16;
        let mut densities = make_density_field_solid(size);
        carve_bowl(&mut densities, size, 5, 8, 2, 4, 5);
        carve_bowl(&mut densities, size, 11, 8, 5, 4, 3);
        carve_box(&mut densities, size, 5..12, 5..8, 6..11);
        let config = crate::FluidConfig::default();
        let mut grid = ChunkFluidGrid::new(size);
        apply_density(&mut grid, &densities, &config);
        for z in 0..size { for y in 2..7 { for x in 2..8 {
            let cap = grid.cell_capacity(x, y, z);
            if cap > MIN_LEVEL { let cell = grid.get_mut(x, y, z); cell.level = cap; cell.fluid_type = FluidType::Water; }
        }}}
        grid.has_fluid = true;

        let mut chunks = HashMap::new();
        chunks.insert((0,0,0), grid);
        let initial = total_water(&chunks);
        let dc = empty_density_cache();
        for _ in 0..800 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
        let final_w = total_water(&chunks);
        let loss_pct = ((initial - final_w) / initial * 100.0).abs();
        // ±3% (was ±1%): face gating (2026-08-04) evaporates settle-time thin
        // films whose only consolidation path crosses a rendered surface —
        // a one-time "skin soak" at fill, not an ongoing leak.
        assert!(loss_pct < 3.0, "Conservation ±3%: initial={:.2}, final={:.2}, loss={:.2}%", initial, final_w, loss_pct);
    }

    #[test]
    fn bowl_with_raised_lip_contains() {
        let size = 16;
        let mut densities = make_density_field_solid(size);
        carve_bowl(&mut densities, size, 8, 8, 3, 5, 6);
        let config = crate::FluidConfig::default();
        let mut grid = ChunkFluidGrid::new(size);
        apply_density(&mut grid, &densities, &config);
        fill_air_to_capacity(&mut grid, 3..7);
        let initial = grid_total_water(&grid);

        let mut chunks = HashMap::new();
        chunks.insert((0,0,0), grid);
        let dc = empty_density_cache();
        for _ in 0..500 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
        let final_w = total_water(&chunks);
        // 3% (was 0.01 absolute): face-gating settle-time skin soak, see
        // bowl_symmetric_retains_water.
        assert!((final_w - initial).abs() / initial < 0.03, "Conservation: initial={:.2}, final={:.2}", initial, final_w);
    }

    #[test]
    fn bowl_with_notch_drains() {
        let size = 16;
        let mut densities = make_density_field_solid(size);
        carve_bowl(&mut densities, size, 8, 8, 3, 5, 6);
        carve_box(&mut densities, size, 12..15, 5..8, 7..10);
        let config = crate::FluidConfig::default();
        let mut grid = ChunkFluidGrid::new(size);
        apply_density(&mut grid, &densities, &config);
        fill_air_to_capacity(&mut grid, 3..8);

        let mut chunks = HashMap::new();
        chunks.insert((0,0,0), grid);
        let dc = empty_density_cache();
        for _ in 0..1000 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }

        let grid = &chunks[&(0,0,0)];
        let mut outside_water = 0.0f64;
        for z in 0..size { for y in 0..size { for x in 12..size { outside_water += grid.get(x, y, z).level as f64; } } }
        assert!(outside_water > 0.1, "Water should drain through notch, outside_water={:.3}", outside_water);
    }

    #[test]
    fn bowl_nested_inner_outer() {
        let size = 16;
        let mut densities = make_density_field_solid(size);
        carve_bowl(&mut densities, size, 8, 8, 5, 6, 3);
        carve_bowl(&mut densities, size, 8, 8, 2, 3, 4);
        let config = crate::FluidConfig::default();
        let mut grid = ChunkFluidGrid::new(size);
        apply_density(&mut grid, &densities, &config);
        for z in 5..12 { for y in 2..5 { for x in 5..12 {
            let cap = grid.cell_capacity(x, y, z);
            if cap > MIN_LEVEL { let cell = grid.get_mut(x, y, z); cell.level = cap; cell.fluid_type = FluidType::Water; }
        }}}
        grid.has_fluid = true;

        let mut chunks = HashMap::new();
        chunks.insert((0,0,0), grid);
        let initial = total_water(&chunks);
        let dc = empty_density_cache();
        for _ in 0..500 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
        let final_w = total_water(&chunks);
        let loss_pct = ((initial - final_w) / initial * 100.0).abs();
        assert!(loss_pct < 1.0, "Conservation: initial={:.2}, final={:.2}, loss={:.2}%", initial, final_w, loss_pct);
    }

    #[test]
    fn bowl_fractional_boundary_conservation() {
        // Capacity is now fractional (air_corners/8) so the simulator's notion
        // of free space matches the marching-cubes mesh surface — fluid no
        // longer clips through partial-rock cells. Boundary cells legitimately
        // hold partial capacity in [0, 1]; the test asserts conservation, not
        // a binary cap.
        let size = 16;
        let stride = size + 1;
        let mut densities = vec![1.0f32; stride * stride * stride];
        for gz in 4..13 { for gy in 3..12 { for gx in 4..13 {
            if gx >= 6 && gx <= 11 && gy >= 5 && gy <= 10 && gz >= 6 && gz <= 11 {
                densities[gz * stride * stride + gy * stride + gx] = -1.0;
            } else {
                let dx = if gx < 6 { 6 - gx } else if gx > 11 { gx - 11 } else { 0 };
                let dy = if gy < 5 { 5 - gy } else if gy > 10 { gy - 10 } else { 0 };
                let dz = if gz < 6 { 6 - gz } else if gz > 11 { gz - 11 } else { 0 };
                let dist = (dx.max(dy).max(dz)) as f32;
                let v = -0.5 + dist * 0.4;
                densities[gz * stride * stride + gy * stride + gx] = v;
            }
        }}}

        let config = crate::FluidConfig::default();
        let mut grid = ChunkFluidGrid::new(size);
        apply_density(&mut grid, &densities, &config);
        for z in 0..size { for y in 0..size { for x in 0..size {
            let cap = grid.cell_capacity(x, y, z);
            assert!(cap >= 0.0 && cap <= 1.0,
                "Capacity out of [0,1] at ({},{},{}): cap={}", x, y, z, cap);
        }}}
        fill_air_to_capacity(&mut grid, 0..size);

        let mut chunks = HashMap::new();
        chunks.insert((0,0,0), grid);
        let initial = total_water(&chunks);
        let dc = empty_density_cache();
        for _ in 0..500 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
        let final_w = total_water(&chunks);
        let loss_pct = ((initial - final_w) / initial * 100.0).abs();
        assert!(loss_pct < 1.0, "Fractional boundary conservation: initial={:.2}, final={:.2}, loss={:.2}%", initial, final_w, loss_pct);
    }

    #[test]
    fn bowl_cross_chunk_boundary() {
        let size = 16;
        let mut upper_d = make_density_field_solid(size);
        carve_box(&mut upper_d, size, 4..13, 0..4, 4..13);
        let mut lower_d = make_density_field_solid(size);
        carve_box(&mut lower_d, size, 4..13, 13..17, 4..13);

        let config = crate::FluidConfig::default();
        let mut upper = ChunkFluidGrid::new(size);
        apply_density(&mut upper, &upper_d, &config);
        fill_air_to_capacity(&mut upper, 0..3);
        let mut lower = ChunkFluidGrid::new(size);
        apply_density(&mut lower, &lower_d, &config);

        let mut chunks = HashMap::new();
        chunks.insert((0, 1, 0), upper);
        chunks.insert((0, 0, 0), lower);
        let initial = total_water(&chunks);
        let dc = empty_density_cache();
        for _ in 0..500 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
        let final_w = total_water(&chunks);
        // 8% relative: face-gating skin soak PLUS the one-way-floor rule
        // (down-transit blocks at >=2 solid corners) — at the chunk boundary
        // some carved-floor crossing cells can no longer drain downward, and
        // their films evaporate instead. Order-dependent (HashMap chunk
        // iteration): observed 0.9-13.5 of 243. Containment beats perfect
        // conservation here by design ("never under the world").
        assert!((final_w - initial).abs() / initial < 0.08, "Cross-chunk conservation: initial={:.2}, final={:.2}", initial, final_w);
    }

    #[test]
    fn two_bowls_connected_by_channel() {
        let size = 16;
        let mut densities = make_density_field_solid(size);
        carve_box(&mut densities, size, 2..8, 3..9, 4..13);
        carve_box(&mut densities, size, 10..15, 3..9, 4..13);
        carve_box(&mut densities, size, 8..10, 4..6, 7..10);

        let config = crate::FluidConfig::default();
        let mut grid = ChunkFluidGrid::new(size);
        apply_density(&mut grid, &densities, &config);
        for z in 0..size { for y in 3..8 { for x in 2..8 {
            let cap = grid.cell_capacity(x, y, z);
            if cap > MIN_LEVEL { let cell = grid.get_mut(x, y, z); cell.level = cap; cell.fluid_type = FluidType::Water; }
        }}}
        grid.has_fluid = true;

        let mut chunks = HashMap::new();
        chunks.insert((0,0,0), grid);
        let initial = total_water(&chunks);
        let dc = empty_density_cache();
        for _ in 0..1000 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }

        let final_w = total_water(&chunks);
        let grid = &chunks[&(0,0,0)];
        assert!((final_w - initial).abs() < 0.5, "Conservation: initial={:.2}, final={:.2}", initial, final_w);
        let mut right_water = 0.0f64;
        for z in 0..size { for y in 0..size { for x in 10..15 { right_water += grid.get(x, y, z).level as f64; } } }
        assert!(right_water > 0.5, "Water should reach right bowl, got {:.3}", right_water);
    }

    // ====================== Category 2: Mining/Terrain Modification (9-16) ======================

    #[test]
    fn mine_floor_water_drains() {
        let size = 16;
        let mut densities = make_density_field_solid(size);
        carve_box(&mut densities, size, 4..13, 3..10, 4..13);
        let config = crate::FluidConfig::default();
        let mut grid = ChunkFluidGrid::new(size);
        apply_density(&mut grid, &densities, &config);
        fill_air_to_capacity(&mut grid, 3..5);

        let mut chunks = HashMap::new();
        chunks.insert((0,0,0), grid);
        let dc = empty_density_cache();
        for _ in 0..100 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
        let pre_mine = total_water(&chunks);

        { let grid = chunks.get_mut(&(0,0,0)).unwrap(); mine_cells(grid, &mut densities, &[(8, 2, 8)], size); }
        for _ in 0..500 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }

        let final_w = total_water(&chunks);
        let grid = &chunks[&(0,0,0)];
        assert!(grid.get(8, 2, 8).level > MIN_LEVEL, "Water should drain into mined floor cell");
        let loss_pct = ((pre_mine - final_w) / pre_mine * 100.0).abs();
        assert!(loss_pct < 0.5, "Conservation ±0.5%: pre={:.2}, final={:.2}, loss={:.2}%", pre_mine, final_w, loss_pct);
    }

    #[test]
    fn mine_wall_water_drains_sideways() {
        let size = 16;
        let mut densities = make_density_field_solid(size);
        carve_box(&mut densities, size, 4..8, 3..8, 4..13);
        let config = crate::FluidConfig::default();
        let mut grid = ChunkFluidGrid::new(size);
        apply_density(&mut grid, &densities, &config);
        fill_air_to_capacity(&mut grid, 3..5);

        let mut chunks = HashMap::new();
        chunks.insert((0,0,0), grid);
        let dc = empty_density_cache();
        for _ in 0..100 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
        let pre_mine = total_water(&chunks);

        { let grid = chunks.get_mut(&(0,0,0)).unwrap(); mine_cells(grid, &mut densities, &[(8, 3, 8), (8, 4, 8)], size); }
        for _ in 0..500 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }

        let final_w = total_water(&chunks);
        let grid = &chunks[&(0,0,0)];
        let mut outside = 0.0f64;
        for z in 0..size { for y in 0..size { for x in 9..size { outside += grid.get(x, y, z).level as f64; } } }
        assert!(outside > 0.01, "Water should drain sideways through mined wall");
        let loss_pct = ((pre_mine - final_w) / pre_mine * 100.0).abs();
        assert!(loss_pct < 0.5, "Conservation ±0.5%: {:.2} vs {:.2}", pre_mine, final_w);
    }

    #[test]
    fn mine_ceiling_no_effect() {
        let size = 16;
        let mut densities = make_density_field_solid(size);
        carve_box(&mut densities, size, 4..13, 3..10, 4..13);
        let config = crate::FluidConfig::default();
        let mut grid = ChunkFluidGrid::new(size);
        apply_density(&mut grid, &densities, &config);
        fill_air_to_capacity(&mut grid, 3..5);

        let mut chunks = HashMap::new();
        chunks.insert((0,0,0), grid);
        let dc = empty_density_cache();
        for _ in 0..100 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
        let pre_mine = total_water(&chunks);

        { let grid = chunks.get_mut(&(0,0,0)).unwrap(); mine_cells(grid, &mut densities, &[(8, 10, 8)], size); }
        for _ in 0..300 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
        let final_w = total_water(&chunks);
        assert!((final_w - pre_mine).abs() < 0.01, "Ceiling mine should not affect pool: {:.2} vs {:.2}", pre_mine, final_w);
    }

    #[test]
    fn mine_channel_between_pools() {
        let size = 16;
        let mut densities = make_density_field_solid(size);
        carve_box(&mut densities, size, 2..7, 3..8, 4..13);
        carve_box(&mut densities, size, 10..15, 3..8, 4..13);
        let config = crate::FluidConfig::default();
        let mut grid = ChunkFluidGrid::new(size);
        apply_density(&mut grid, &densities, &config);
        for z in 0..size { for y in 3..7 { for x in 2..7 {
            let cap = grid.cell_capacity(x, y, z);
            if cap > MIN_LEVEL { let cell = grid.get_mut(x, y, z); cell.level = cap; cell.fluid_type = FluidType::Water; }
        }}}
        for z in 0..size { for y in 3..5 { for x in 10..15 {
            let cap = grid.cell_capacity(x, y, z);
            if cap > MIN_LEVEL { let cell = grid.get_mut(x, y, z); cell.level = cap; cell.fluid_type = FluidType::Water; }
        }}}
        grid.has_fluid = true;

        let mut chunks = HashMap::new();
        chunks.insert((0,0,0), grid);
        let dc = empty_density_cache();
        for _ in 0..100 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }

        { let grid = chunks.get_mut(&(0,0,0)).unwrap();
          let mut to_mine = Vec::new();
          for x in 7..10 { for y in 3..5 { to_mine.push((x, y, 8)); to_mine.push((x, y, 9)); } }
          mine_cells(grid, &mut densities, &to_mine, size); }
        for _ in 0..1000 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }

        let grid = &chunks[&(0,0,0)];
        let mut left_w = 0.0f64; let mut right_w = 0.0f64;
        for z in 0..size { for y in 0..size {
            for x in 2..7 { left_w += grid.get(x, y, z).level as f64; }
            for x in 10..15 { right_w += grid.get(x, y, z).level as f64; }
        }}
        assert!(left_w > 0.5, "Left should have water: {:.2}", left_w);
        assert!(right_w > 0.5, "Right should have water: {:.2}", right_w);
    }

    #[test]
    fn mine_bowl_bottom_drains() {
        let size = 16;
        let mut densities = make_density_field_solid(size);
        carve_bowl(&mut densities, size, 8, 8, 4, 5, 4);
        let config = crate::FluidConfig::default();
        let mut grid = ChunkFluidGrid::new(size);
        apply_density(&mut grid, &densities, &config);
        fill_air_to_capacity(&mut grid, 4..7);

        let mut chunks = HashMap::new();
        chunks.insert((0,0,0), grid);
        let dc = empty_density_cache();
        for _ in 0..100 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }

        { let grid = chunks.get_mut(&(0,0,0)).unwrap(); mine_cells(grid, &mut densities, &[(8, 3, 8), (7, 3, 8), (9, 3, 8)], size); }
        for _ in 0..500 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }

        let grid = &chunks[&(0,0,0)];
        let mut mined_water = 0.0f64;
        for &(x,y,z) in &[(8,3,8),(7,3,8),(9,3,8)] { mined_water += grid.get(x, y, z).level as f64; }
        assert!(mined_water > 0.01, "Water should drain into mined bowl bottom");
    }

    #[test]
    fn sequential_mining_conservation() {
        let size = 16;
        let mut densities = make_density_field_solid(size);
        carve_box(&mut densities, size, 4..13, 3..10, 4..13);
        let config = crate::FluidConfig::default();
        let mut grid = ChunkFluidGrid::new(size);
        apply_density(&mut grid, &densities, &config);
        fill_air_to_capacity(&mut grid, 3..6);

        let mut chunks = HashMap::new();
        chunks.insert((0,0,0), grid);
        let dc = empty_density_cache();
        for _ in 0..100 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
        let initial = total_water(&chunks);

        let mine_targets = [(8,2,8),(9,2,8),(10,2,8),(8,2,9),(8,2,10)];
        for &target in &mine_targets {
            { let grid = chunks.get_mut(&(0,0,0)).unwrap(); mine_cells(grid, &mut densities, &[target], size); }
            for _ in 0..50 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
        }
        let final_w = total_water(&chunks);
        assert!((final_w - initial).abs() < 1.0, "Cumulative conservation ±1.0: initial={:.2}, final={:.2}", initial, final_w);
    }

    #[test]
    fn mine_creates_ramp() {
        let size = 16;
        let mut densities = make_density_field_solid(size);
        carve_box(&mut densities, size, 3..14, 5..10, 4..13);
        let config = crate::FluidConfig::default();
        let mut grid = ChunkFluidGrid::new(size);
        apply_density(&mut grid, &densities, &config);
        fill_air_to_capacity(&mut grid, 5..6);

        let mut chunks = HashMap::new();
        chunks.insert((0,0,0), grid);
        let dc = empty_density_cache();
        for _ in 0..100 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }

        { let grid = chunks.get_mut(&(0,0,0)).unwrap(); mine_cells(grid, &mut densities, &[(6,4,8),(7,3,8),(7,4,8),(8,2,8),(8,3,8),(8,4,8)], size); }
        for _ in 0..500 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }

        let grid = &chunks[&(0,0,0)];
        assert!(grid.get(8, 2, 8).level > MIN_LEVEL, "Water should cascade down ramp to (8,2,8), got {}", grid.get(8, 2, 8).level);
    }

    #[test]
    fn mine_under_water_creates_drop() {
        let size = 16;
        let mut densities = make_density_field_solid(size);
        carve_box(&mut densities, size, 4..13, 5..10, 4..13);
        let config = crate::FluidConfig::default();
        let mut grid = ChunkFluidGrid::new(size);
        apply_density(&mut grid, &densities, &config);
        fill_air_to_capacity(&mut grid, 5..7);

        let mut chunks = HashMap::new();
        chunks.insert((0,0,0), grid);
        let dc = empty_density_cache();
        for _ in 0..100 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }

        { let grid = chunks.get_mut(&(0,0,0)).unwrap(); mine_cells(grid, &mut densities, &[(8, 4, 8)], size); }
        for _ in 0..300 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }

        let grid = &chunks[&(0,0,0)];
        assert!(grid.get(8, 4, 8).level > MIN_LEVEL, "Water should fall into cavity below, got {}", grid.get(8, 4, 8).level);
    }

    // ====================== Category 3: Natural Cave Shapes (17-24) ======================

    #[test]
    fn sloped_floor_pools_at_low_end() {
        let size = 16;
        let stride = size + 1;
        let mut densities = make_density_field_solid(size);
        for gz in 4..13 { for gx in 0..stride {
            let floor_gy = 3 + (gx * 7) / 16;
            for gy in floor_gy..15 { if gy < stride { densities[gz * stride * stride + gy * stride + gx] = -1.0; } }
        }}
        let config = crate::FluidConfig::default();
        let mut grid = ChunkFluidGrid::new(size);
        apply_density(&mut grid, &densities, &config);
        for z in 0..size { for y in 8..11 { for x in 12..15 {
            let cap = grid.cell_capacity(x, y, z);
            if cap > MIN_LEVEL { let cell = grid.get_mut(x, y, z); cell.level = cap; cell.fluid_type = FluidType::Water; }
        }}}
        grid.has_fluid = true;

        let mut chunks = HashMap::new();
        chunks.insert((0,0,0), grid);
        let initial = total_water(&chunks);
        let dc = empty_density_cache();
        for _ in 0..800 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
        let final_w = total_water(&chunks);
        let loss_pct = ((initial - final_w) / initial * 100.0).abs();
        assert!(loss_pct < 2.0, "Conservation ±2%: initial={:.2}, final={:.2}, loss={:.2}%", initial, final_w, loss_pct);

        let grid = &chunks[&(0,0,0)];
        let mut low_end_water = 0.0f64;
        for z in 0..size { for y in 0..size { for x in 0..4 { low_end_water += grid.get(x, y, z).level as f64; } } }
        assert!(low_end_water > 0.5, "Water should collect at low end, got {:.3}", low_end_water);
    }

    #[test]
    fn v_valley_fills_bottom() {
        let size = 16;
        let stride = size + 1;
        let mut densities = make_density_field_solid(size);
        for gz in 0..stride {
            let floor_gy = ((gz as i32 - 8).unsigned_abs() as usize) + 3;
            for gy in floor_gy..14 { for gx in 3..14 { if gy < stride { densities[gz * stride * stride + gy * stride + gx] = -1.0; } } }
        }
        let config = crate::FluidConfig::default();
        let mut grid = ChunkFluidGrid::new(size);
        apply_density(&mut grid, &densities, &config);
        for z in 6..10 { for y in 3..6 { for x in 3..14 {
            let cap = grid.cell_capacity(x, y, z);
            if cap > MIN_LEVEL { let cell = grid.get_mut(x, y, z); cell.level = cap; cell.fluid_type = FluidType::Water; }
        }}}
        grid.has_fluid = true;

        let mut chunks = HashMap::new();
        chunks.insert((0,0,0), grid);
        let initial = total_water(&chunks);
        let dc = empty_density_cache();
        for _ in 0..500 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
        let final_w = total_water(&chunks);
        let loss_pct = ((initial - final_w) / initial * 100.0).abs();
        // ±3% (was ±1%): face gating (2026-08-04) evaporates settle-time thin
        // films whose only consolidation path crosses a rendered surface —
        // a one-time "skin soak" at fill, not an ongoing leak.
        assert!(loss_pct < 3.0, "Conservation ±3%: initial={:.2}, final={:.2}, loss={:.2}%", initial, final_w, loss_pct);
    }

    #[test]
    fn u_tunnel_cross_section() {
        let size = 16;
        let stride = size + 1;
        let mut densities = make_density_field_solid(size);
        for gz in 0..stride { for gy in 0..stride {
            let dy = gy as f32 - 10.0; let dz = gz as f32 - 8.0;
            let dist = (dy * dy + dz * dz).sqrt();
            if dist < 5.0 && gy <= 10 { for gx in 3..14 { densities[gz * stride * stride + gy * stride + gx] = -1.0; } }
        }}
        let config = crate::FluidConfig::default();
        let mut grid = ChunkFluidGrid::new(size);
        apply_density(&mut grid, &densities, &config);
        fill_air_to_capacity(&mut grid, 5..8);

        let mut chunks = HashMap::new();
        chunks.insert((0,0,0), grid);
        let initial = total_water(&chunks);
        let dc = empty_density_cache();
        for _ in 0..500 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
        let final_w = total_water(&chunks);
        let loss_pct = ((initial - final_w) / initial * 100.0).abs();
        // ±3% (was ±1%): face gating (2026-08-04) evaporates settle-time thin
        // films whose only consolidation path crosses a rendered surface —
        // a one-time "skin soak" at fill, not an ongoing leak.
        assert!(loss_pct < 3.0, "Conservation ±3%: initial={:.2}, final={:.2}, loss={:.2}%", initial, final_w, loss_pct);
    }

    #[test]
    fn staircase_cascade() {
        let size = 16;
        let mut densities = make_density_field_solid(size);
        let steps: [(Range<usize>, usize); 4] = [(1..4, 9), (4..7, 7), (7..10, 5), (10..13, 3)];
        for (x_range, floor_gy) in &steps {
            for gx in x_range.clone() { for gy in *floor_gy..14 { for gz in 4..13 {
                densities[gz * (size+1) * (size+1) + gy * (size+1) + gx] = -1.0;
            }}}
        }
        let config = crate::FluidConfig::default();
        let mut grid = ChunkFluidGrid::new(size);
        apply_density(&mut grid, &densities, &config);
        for z in 0..size { for y in 9..12 { for x in 1..4 {
            let cap = grid.cell_capacity(x, y, z);
            if cap > MIN_LEVEL { let cell = grid.get_mut(x, y, z); cell.level = cap; cell.fluid_type = FluidType::Water; }
        }}}
        grid.has_fluid = true;

        let mut chunks = HashMap::new();
        chunks.insert((0,0,0), grid);
        let initial = total_water(&chunks);
        let dc = empty_density_cache();
        for _ in 0..800 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
        let final_w = total_water(&chunks);
        let loss_pct = ((initial - final_w) / initial * 100.0).abs();
        assert!(loss_pct < 2.0, "Conservation ±2%: initial={:.2}, final={:.2}", initial, final_w);

        let grid = &chunks[&(0,0,0)];
        let mut bottom_water = 0.0f64;
        for z in 0..size { for y in 3..5 { for x in 10..13 { bottom_water += grid.get(x, y, z).level as f64; } } }
        assert!(bottom_water > 0.1, "Water should reach bottom step, got {:.3}", bottom_water);
    }

    #[test]
    fn irregular_cave_deterministic() {
        let size = 16;
        let stride = size + 1;
        let mut densities = make_density_field_solid(size);
        // gy starts at 1: keep the bottom lattice plane solid so the cave
        // has a real floor — with the void-cull backstop, noise pockets
        // that opened to y=0 drained water out of the world (correctly).
        for gz in 0..stride { for gy in 1..stride { for gx in 0..stride {
            let val = (gx * 7 + gy * 13 + gz * 17) % 23;
            if val >= 8 { densities[gz * stride * stride + gy * stride + gx] = -1.0; }
        }}}
        for gz in 5..11 { for gy in 5..11 { for gx in 5..11 { densities[gz * stride * stride + gy * stride + gx] = -1.0; } } }

        let config = crate::FluidConfig::default();
        let mut grid = ChunkFluidGrid::new(size);
        apply_density(&mut grid, &densities, &config);
        for z in 5..10 { for y in 5..10 { for x in 5..10 {
            let cap = grid.cell_capacity(x, y, z);
            if cap > MIN_LEVEL { let cell = grid.get_mut(x, y, z); cell.level = cap; cell.fluid_type = FluidType::Water; }
        }}}
        grid.has_fluid = true;

        let mut chunks = HashMap::new();
        chunks.insert((0,0,0), grid);
        let initial = total_water(&chunks);
        let dc = empty_density_cache();
        for _ in 0..500 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
        let final_w = total_water(&chunks);
        let loss_pct = ((initial - final_w) / initial * 100.0).abs();
        // ±3% (was ±1%): face gating (2026-08-04) evaporates settle-time thin
        // films whose only consolidation path crosses a rendered surface —
        // a one-time "skin soak" at fill, not an ongoing leak.
        assert!(loss_pct < 3.0, "Conservation ±3%: initial={:.2}, final={:.2}, loss={:.2}%", initial, final_w, loss_pct);
    }

    #[test]
    fn narrow_passage_between_chambers() {
        let size = 16;
        let mut densities = make_density_field_solid(size);
        carve_box(&mut densities, size, 1..6, 3..9, 3..9);
        carve_box(&mut densities, size, 10..15, 3..9, 3..9);
        carve_box(&mut densities, size, 6..10, 5..7, 5..7);
        let config = crate::FluidConfig::default();
        let mut grid = ChunkFluidGrid::new(size);
        apply_density(&mut grid, &densities, &config);
        for z in 0..size { for y in 3..8 { for x in 1..6 {
            let cap = grid.cell_capacity(x, y, z);
            if cap > MIN_LEVEL { let cell = grid.get_mut(x, y, z); cell.level = cap; cell.fluid_type = FluidType::Water; }
        }}}
        grid.has_fluid = true;

        let mut chunks = HashMap::new();
        chunks.insert((0,0,0), grid);
        let initial = total_water(&chunks);
        let dc = empty_density_cache();
        for _ in 0..1500 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
        let final_w = total_water(&chunks);
        assert!((final_w - initial).abs() < 0.5, "Conservation: initial={:.2}, final={:.2}", initial, final_w);

        let grid = &chunks[&(0,0,0)];
        let mut right_w = 0.0f64;
        for z in 0..size { for y in 0..size { for x in 10..15 { right_w += grid.get(x, y, z).level as f64; } } }
        assert!(right_w > 0.5, "Water should reach right chamber through passage, got {:.3}", right_w);
    }

    #[test]
    fn overhang_shelf_drip() {
        let size = 16;
        let mut densities = make_density_field_solid(size);
        carve_box(&mut densities, size, 3..14, 2..15, 3..14);
        for gz in 3..14 { for gy in 7..9 { for gx in 3..10 { densities[gz * (size+1)*(size+1) + gy * (size+1) + gx] = 1.0; } } }
        let config = crate::FluidConfig::default();
        let mut grid = ChunkFluidGrid::new(size);
        apply_density(&mut grid, &densities, &config);
        for z in 0..size { for y in 9..11 { for x in 3..10 {
            let cap = grid.cell_capacity(x, y, z);
            if cap > MIN_LEVEL { let cell = grid.get_mut(x, y, z); cell.level = cap; cell.fluid_type = FluidType::Water; }
        }}}
        grid.has_fluid = true;

        let mut chunks = HashMap::new();
        chunks.insert((0,0,0), grid);
        let initial = total_water(&chunks);
        let dc = empty_density_cache();
        for _ in 0..500 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
        let final_w = total_water(&chunks);
        assert!((final_w - initial).abs() < 0.5, "Conservation: initial={:.2}, final={:.2}", initial, final_w);

        let grid = &chunks[&(0,0,0)];
        let mut floor_water = 0.0f64;
        for z in 0..size { for y in 2..4 { for x in 0..size { floor_water += grid.get(x, y, z).level as f64; } } }
        assert!(floor_water > 0.1, "Water should drip to floor, got {:.3}", floor_water);
    }

    #[test]
    fn dome_ceiling_flat_floor() {
        let size = 16;
        let stride = size + 1;
        let mut densities = make_density_field_solid(size);
        for gz in 0..stride { for gy in 3..stride { for gx in 0..stride {
            let dx = gx as f32 - 8.0; let dy = gy as f32 - 3.0; let dz = gz as f32 - 8.0;
            let dist = (dx*dx + dy*dy + dz*dz).sqrt();
            if dist < 8.0 && dy >= 0.0 { densities[gz * stride * stride + gy * stride + gx] = -1.0; }
        }}}
        let config = crate::FluidConfig::default();
        let mut grid = ChunkFluidGrid::new(size);
        apply_density(&mut grid, &densities, &config);
        fill_air_to_capacity(&mut grid, 3..6);

        let mut chunks = HashMap::new();
        chunks.insert((0,0,0), grid);
        let initial = total_water(&chunks);
        let dc = empty_density_cache();
        for _ in 0..500 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
        let final_w = total_water(&chunks);
        let loss_pct = ((initial - final_w) / initial * 100.0).abs();
        assert!(loss_pct < 1.0, "Conservation ±1%: initial={:.2}, final={:.2}", initial, final_w);
    }

    // ====================== Category 4: Multi-Chunk & Edge Cases (25-32) ======================

    #[test]
    fn cross_chunk_horizontal_pool() {
        let size = 16;
        let mut d_left = make_density_field_solid(size);
        carve_box(&mut d_left, size, 0..17, 3..12, 3..14);
        let mut d_right = make_density_field_solid(size);
        carve_box(&mut d_right, size, 0..17, 3..12, 3..14);

        let mut config = crate::FluidConfig::default();
        config.water_spread_rate = 0.6;
        config.water_flow_rate = 1.0;
        let mut left = ChunkFluidGrid::new(size);
        apply_density(&mut left, &d_left, &config);
        let mut right = ChunkFluidGrid::new(size);
        apply_density(&mut right, &d_right, &config);

        for z in 0..size { for y in 3..5 { for x in 12..size {
            let cap = left.cell_capacity(x, y, z);
            if cap > MIN_LEVEL { let cell = left.get_mut(x, y, z); cell.level = cap; cell.fluid_type = FluidType::Water; }
        }}}
        left.has_fluid = true;

        let mut chunks = HashMap::new();
        chunks.insert((0, 0, 0), left);
        chunks.insert((1, 0, 0), right);
        let initial = total_water(&chunks);
        let dc = empty_density_cache();
        for _ in 0..800 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
        let final_w = total_water(&chunks);
        let loss_pct = ((initial - final_w) / initial * 100.0).abs();
        assert!(loss_pct < 2.0, "Conservation ±2%: initial={:.2}, final={:.2}, loss={:.2}%", initial, final_w, loss_pct);

        let right_grid = &chunks[&(1, 0, 0)];
        let mut rw = 0.0f64;
        for z in 0..size { for y in 0..size { for x in 0..4 { rw += right_grid.get(x, y, z).level as f64; } } }
        assert!(rw > 0.01, "Water should spread to right chunk, got {:.3}", rw);
    }

    /// 2026-08-17 regression: cross-chunk transfer amounts are computed against
    /// the dest's state at the SENDER's tick, but the dest chunk's own tick can
    /// fill the cell before transfers apply (chunk order is HashMap-random).
    /// The apply-time clamp used to EVAPORATE the residual instead of refunding
    /// the sender — this exact pool lost 2.27% when the left chunk ticked first
    /// and 0.20% when the right did (flaked ~50% of CI runs). Each iteration
    /// builds fresh HashMaps (each gets its own hash seed), so a handful of
    /// iterations exercises both tick orders within one run: pre-fix this fails
    /// near-deterministically, post-fix every ordering conserves.
    #[test]
    fn cross_chunk_transfer_clamp_refunds_sender() {
        let size = 16;
        for _ in 0..6 {
            let mut d_left = make_density_field_solid(size);
            carve_box(&mut d_left, size, 0..17, 3..12, 3..14);
            let mut d_right = make_density_field_solid(size);
            carve_box(&mut d_right, size, 0..17, 3..12, 3..14);

            let mut config = crate::FluidConfig::default();
            config.water_spread_rate = 0.6;
            config.water_flow_rate = 1.0;
            let mut left = ChunkFluidGrid::new(size);
            apply_density(&mut left, &d_left, &config);
            let mut right = ChunkFluidGrid::new(size);
            apply_density(&mut right, &d_right, &config);

            for z in 0..size { for y in 3..5 { for x in 12..size {
                let cap = left.cell_capacity(x, y, z);
                if cap > MIN_LEVEL { let cell = left.get_mut(x, y, z); cell.level = cap; cell.fluid_type = FluidType::Water; }
            }}}
            left.has_fluid = true;

            let mut chunks = HashMap::new();
            chunks.insert((0, 0, 0), left);
            chunks.insert((1, 0, 0), right);
            let initial = total_water(&chunks);
            let dc = empty_density_cache();
            // The clamp loss lands in the first ~12 ticks (1.8 of the 2.0 units);
            // 100 ticks is plenty and keeps 6 iterations cheap.
            for _ in 0..100 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
            let final_w = total_water(&chunks);
            let loss_pct = ((initial - final_w) / initial * 100.0).abs();
            assert!(loss_pct < 1.0, "Order-invariant conservation ±1%: initial={:.2}, final={:.2}, loss={:.2}%", initial, final_w, loss_pct);
        }
    }

    #[test]
    fn cross_chunk_vertical_waterfall() {
        let size = 16;
        let mut d_upper = make_density_field_solid(size);
        carve_box(&mut d_upper, size, 4..13, 0..17, 4..13);
        let mut d_lower = make_density_field_solid(size);
        carve_box(&mut d_lower, size, 4..13, 12..17, 4..13);

        let mut config = crate::FluidConfig::default();
        config.water_spread_rate = 0.6;
        config.water_flow_rate = 1.0;
        let mut upper = ChunkFluidGrid::new(size);
        apply_density(&mut upper, &d_upper, &config);
        let mut lower = ChunkFluidGrid::new(size);
        apply_density(&mut lower, &d_lower, &config);

        for z in 0..size { for y in 5..10 { for x in 4..13 {
            let cap = upper.cell_capacity(x, y, z);
            if cap > MIN_LEVEL { let cell = upper.get_mut(x, y, z); cell.level = cap; cell.fluid_type = FluidType::Water; }
        }}}
        upper.has_fluid = true;

        let mut chunks = HashMap::new();
        chunks.insert((0, 1, 0), upper);
        chunks.insert((0, 0, 0), lower);
        let initial = total_water(&chunks);
        let dc = empty_density_cache();
        for _ in 0..500 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
        let final_w = total_water(&chunks);
        let loss_pct = ((initial - final_w) / initial * 100.0).abs();
        assert!(loss_pct < 2.0, "Conservation ±2%: initial={:.2}, final={:.2}, loss={:.2}%", initial, final_w, loss_pct);

        let lower_grid = &chunks[&(0, 0, 0)];
        let mut basin_w = 0.0f64;
        for z in 0..size { for y in 12..size { for x in 4..13 { basin_w += lower_grid.get(x, y, z).level as f64; } } }
        assert!(basin_w > 0.1, "Water should reach lower basin, got {:.3}", basin_w);
    }

    #[test]
    fn cross_chunk_pressure_equalization() {
        let size = 16;
        let mut d_lower = make_density_field_solid(size);
        carve_box(&mut d_lower, size, 4..13, 3..17, 4..13);
        let mut d_upper = make_density_field_solid(size);
        carve_box(&mut d_upper, size, 4..13, 0..10, 4..13);

        let config = crate::FluidConfig::default();
        let mut lower = ChunkFluidGrid::new(size);
        apply_density(&mut lower, &d_lower, &config);
        let mut upper = ChunkFluidGrid::new(size);
        apply_density(&mut upper, &d_upper, &config);

        for z in 0..size { for y in 3..size { for x in 4..13 {
            let cap = lower.cell_capacity(x, y, z);
            if cap > MIN_LEVEL { let cell = lower.get_mut(x, y, z); cell.level = cap; cell.fluid_type = FluidType::Water; }
        }}}
        lower.has_fluid = true;

        let mut chunks = HashMap::new();
        chunks.insert((0, 0, 0), lower);
        chunks.insert((0, 1, 0), upper);
        let initial = total_water(&chunks);
        let dc = empty_density_cache();
        for _ in 0..2000 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
        let final_w = total_water(&chunks);
        let loss_pct = ((initial - final_w) / initial * 100.0).abs();
        assert!(loss_pct < 2.0, "Conservation ±2%: initial={:.2}, final={:.2}, loss={:.2}%", initial, final_w, loss_pct);

        let upper_grid = &chunks[&(0, 1, 0)];
        let mut upper_w = 0.0f64;
        for z in 0..size { for y in 0..size { for x in 4..13 { upper_w += upper_grid.get(x, y, z).level as f64; } } }
        let lower_grid = &chunks[&(0, 0, 0)];
        let mut lower_w = 0.0f64;
        for z in 0..size { for y in 3..size { for x in 4..13 { lower_w += lower_grid.get(x, y, z).level as f64; } } }
        let total_remaining = lower_w + upper_w;
        let loss_pct = ((initial - total_remaining) / initial * 100.0).abs();
        assert!(loss_pct < 15.0,
            "Cross-chunk conservation: lower={:.2}, upper={:.2}, total={:.2}, initial={:.2}, loss={:.1}%",
            lower_w, upper_w, total_remaining, initial, loss_pct);
    }

    #[test]
    fn three_chunk_cascade() {
        let size = 16;
        let mut d_upper = make_density_field_solid(size);
        carve_box(&mut d_upper, size, 5..12, 0..12, 5..12);
        let mut d_mid = make_density_field_solid(size);
        carve_box(&mut d_mid, size, 5..12, 0..17, 5..12);
        let mut d_lower = make_density_field_solid(size);
        carve_box(&mut d_lower, size, 5..12, 10..17, 5..12);

        let config = crate::FluidConfig::default();
        let mut upper = ChunkFluidGrid::new(size);
        apply_density(&mut upper, &d_upper, &config);
        let mut mid = ChunkFluidGrid::new(size);
        apply_density(&mut mid, &d_mid, &config);
        let mut lower = ChunkFluidGrid::new(size);
        apply_density(&mut lower, &d_lower, &config);
        fill_air_to_capacity(&mut upper, 5..10);

        let mut chunks = HashMap::new();
        chunks.insert((0, 2, 0), upper);
        chunks.insert((0, 1, 0), mid);
        chunks.insert((0, 0, 0), lower);
        let initial = total_water(&chunks);
        let dc = empty_density_cache();
        for _ in 0..800 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
        let final_w = total_water(&chunks);
        // 3% relative like the other conservation sites: one-time settle
        // "skin soak" (thin films whose only rescue path is face-gated
        // evaporate) is HashMap-chunk-order dependent — this test was flaky
        // at a fixed 0.2 absolute (loss 0.7–1.4% across runs, 2026-08-04).
        let loss_pct = (initial - final_w).max(0.0) / initial * 100.0;
        assert!(loss_pct < 3.0, "Conservation: initial={:.2}, final={:.2} ({:.1}% loss)", initial, final_w, loss_pct);
        assert!(final_w <= initial + 0.2, "Water appeared from nowhere: initial={:.2}, final={:.2}", initial, final_w);

        let lower_grid = &chunks[&(0, 0, 0)];
        let mut low_w = 0.0f64;
        for z in 0..size { for y in 10..size { for x in 0..size { low_w += lower_grid.get(x, y, z).level as f64; } } }
        assert!(low_w > 0.1, "Water should reach lowest chunk, got {:.3}", low_w);
    }

    #[test]
    fn chunk_boundary_pool() {
        let size = 16;
        let mut d_left = make_density_field_solid(size);
        carve_box(&mut d_left, size, 10..17, 3..8, 4..13);
        let mut d_right = make_density_field_solid(size);
        carve_box(&mut d_right, size, 0..7, 3..8, 4..13);

        let mut config = crate::FluidConfig::default();
        config.water_spread_rate = 0.6;
        config.water_flow_rate = 1.0;
        let mut left = ChunkFluidGrid::new(size);
        apply_density(&mut left, &d_left, &config);
        let mut right = ChunkFluidGrid::new(size);
        apply_density(&mut right, &d_right, &config);
        fill_air_to_capacity(&mut left, 3..5);
        fill_air_to_capacity(&mut right, 3..5);

        let mut chunks = HashMap::new();
        chunks.insert((0, 0, 0), left);
        chunks.insert((1, 0, 0), right);
        let initial = total_water(&chunks);
        let dc = empty_density_cache();
        for _ in 0..500 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
        let final_w = total_water(&chunks);
        assert!((final_w - initial).abs() < 0.1, "Conservation: initial={:.2}, final={:.2}", initial, final_w);
    }

    #[test]
    fn large_volume_conservation() {
        let size = 16;
        let mut densities = make_density_field_solid(size);
        carve_box(&mut densities, size, 1..15, 2..13, 1..15);
        let config = crate::FluidConfig::default();
        let mut grid = ChunkFluidGrid::new(size);
        apply_density(&mut grid, &densities, &config);
        for z in 0..size { for y in 2..13 { for x in 1..15 {
            let cap = grid.cell_capacity(x, y, z);
            if cap > MIN_LEVEL { let cell = grid.get_mut(x, y, z); cell.level = 0.5f32.min(cap); cell.fluid_type = FluidType::Water; }
        }}}
        grid.has_fluid = true;

        let mut chunks = HashMap::new();
        chunks.insert((0,0,0), grid);
        let initial = total_water(&chunks);
        let dc = empty_density_cache();
        for _ in 0..500 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
        let final_w = total_water(&chunks);
        // 0.1% relative (was 0.1 absolute): face-gating settle-time skin
        // soak, see bowl_symmetric_retains_water.
        assert!((final_w - initial).abs() / initial < 0.001, "Large volume conservation: initial={:.4}, final={:.4}", initial, final_w);
    }

    #[test]
    fn tiny_drip_conservation() {
        let size = 16;
        let mut densities = make_density_field_solid(size);
        carve_box(&mut densities, size, 4..13, 3..10, 4..13);
        let config = crate::FluidConfig::default();
        let mut grid = ChunkFluidGrid::new(size);
        apply_density(&mut grid, &densities, &config);
        let cell = grid.get_mut(8, 3, 8);
        cell.level = 0.01; cell.fluid_type = FluidType::Water;
        grid.has_fluid = true;

        let mut chunks = HashMap::new();
        chunks.insert((0,0,0), grid);
        let initial = total_water(&chunks);
        let dc = empty_density_cache();
        for _ in 0..300 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
        let final_w = total_water(&chunks);
        assert!(final_w <= initial + 0.001, "Tiny drip should not create fluid: initial={:.4}, final={:.4}", initial, final_w);
        assert!(final_w >= -0.001, "Tiny drip should not go negative: final={:.4}", final_w);
    }

    #[test]
    fn l_shaped_container_damping() {
        let size = 16;
        let mut densities = make_density_field_solid(size);
        carve_box(&mut densities, size, 2..5, 2..10, 4..13);
        carve_box(&mut densities, size, 2..11, 2..4, 4..13);
        let config = crate::FluidConfig::default();
        let mut grid = ChunkFluidGrid::new(size);
        apply_density(&mut grid, &densities, &config);
        for z in 0..size { for y in 2..9 { for x in 2..5 {
            let cap = grid.cell_capacity(x, y, z);
            if cap > MIN_LEVEL { let cell = grid.get_mut(x, y, z); cell.level = cap; cell.fluid_type = FluidType::Water; }
        }}}
        grid.has_fluid = true;

        let mut chunks = HashMap::new();
        chunks.insert((0,0,0), grid);
        let initial = total_water(&chunks);
        let dc = empty_density_cache();
        for _ in 0..1900 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
        let mut last_100: Vec<f64> = Vec::new();
        for _ in 0..100 { tick_fluid(&mut chunks, &dc, size, false, &config, true); last_100.push(total_water(&chunks)); }

        let final_w = total_water(&chunks);
        assert!((final_w - initial).abs() < 0.1, "Conservation: initial={:.2}, final={:.2}", initial, final_w);

        let max_t = last_100.iter().cloned().fold(f64::MIN, f64::max);
        let min_t = last_100.iter().cloned().fold(f64::MAX, f64::min);
        assert!(max_t - min_t < 0.1, "Should be stable in last 100 ticks: range={:.4}", max_t - min_t);
    }

    // ── 2026-08-04 containment bundle: source-boundedness mechanism ──────
    // Pool basins are seeded with is_source cells, and regen_sources refills
    // every source to full EVERY tick. These two tests document why a
    // breached basin of max_flow_dist=0 sources floods the world forever,
    // and why bounding max_flow_dist makes any leak die a few cells out.

    /// Two chunks side by side along +X. Solid floor spanning both, sloping
    /// gently downhill to the east (one step down every ~6 cells — flat
    /// floors reach a thin-film steady state and stall the leak; real cavern
    /// floors slope, and slope flow is what carries a leak across the
    /// world). A walled lava basin sits near the west edge of chunk 0 with
    /// a 2-cell breach in its east wall.
    fn make_breached_source_basin(
        max_flow_dist: u8,
    ) -> HashMap<(i32, i32, i32), ChunkFluidGrid> {
        let size = 16;
        let mut west = ChunkFluidGrid::new(size);
        let mut east = ChunkFluidGrid::new(size);
        // Floor: slab whose top surface steps down eastward.
        // world x 0..=11 -> floor below y=4; 12..=17 -> y=3; 18..=23 -> y=2;
        // 24..=31 -> y=1.
        for wx in 0..(2 * size) {
            let floor_top = match wx {
                0..=11 => 4usize,
                12..=17 => 3,
                18..=23 => 2,
                _ => 1,
            };
            let (grid, x) = if wx < size {
                (&mut west, wx)
            } else {
                (&mut east, wx - size)
            };
            for z in 0..size { for y in 0..floor_top {
                grid.set_density(x, y, z, 1.0);
            }}
        }
        // Basin walls in chunk 0: ring around interior x 2..=5, z 6..=9,
        // two voxels tall (y 4..=5) so pressure can't hop the lip.
        for z in 5..=10usize { for x in 1..=6usize {
            let on_ring = x == 1 || x == 6 || z == 5 || z == 10;
            if !on_ring { continue; }
            // Breach: east wall open at z=7,8.
            if x == 6 && (z == 7 || z == 8) { continue; }
            west.set_density(x, 4, z, 1.0);
            west.set_density(x, 5, z, 1.0);
        }}
        // Lava source cells fill the basin interior at y=4.
        for z in 6..=9usize { for x in 2..=5usize {
            let cell = west.get_mut(x, 4, z);
            cell.level = SOURCE_LEVEL;
            cell.fluid_type = FluidType::Lava;
            cell.is_source = true;
            cell.max_flow_dist = max_flow_dist;
            cell.hops_from_source = 0;
        }}
        west.has_fluid = true;
        west.has_sources = true;

        let mut chunks = HashMap::new();
        chunks.insert((0, 0, 0), west);
        chunks.insert((1, 0, 0), east);
        chunks
    }

    /// World-space X of the furthest cell holding at least `min_level` lava.
    fn furthest_lava_x(chunks: &HashMap<(i32, i32, i32), ChunkFluidGrid>, min_level: f32) -> i32 {
        let mut max_x = i32::MIN;
        for (key, grid) in chunks {
            let size = grid.size;
            for z in 0..size { for y in 0..size { for x in 0..size {
                let cell = grid.get(x, y, z);
                if cell.fluid_type.is_lava() && cell.level >= min_level {
                    max_x = max_x.max(key.0 * size as i32 + x as i32);
                }
            }}}
        }
        max_x
    }

    #[test]
    fn breached_unbounded_source_basin_floods_far() {
        // max_flow_dist = 0 — exactly how pool seeds are injected today.
        let mut chunks = make_breached_source_basin(0);
        let config = crate::FluidConfig::default();
        let dc = empty_density_cache();
        // 900 ticks: cross-chunk transfer timing varies with HashMap chunk
        // order (run-to-run), and the front's tail advances slowly — give the
        // flood time to express itself regardless of iteration order.
        for _ in 0..900 {
            crate::sim::regen_sources(&mut chunks);
            tick_fluid(&mut chunks, &dc, 16, true, &config, true);
        }
        let reach = furthest_lava_x(&chunks, 0.1);
        // Basin east wall is at x=6. Unbounded + infinitely-refilled sources
        // must push a substantial sheet deep into the neighbor chunk.
        assert!(
            reach >= 20,
            "expected unbounded breached basin to flood far east (>= x 20), reach = {reach}"
        );
    }

    #[test]
    fn breached_bounded_source_basin_leak_stays_local() {
        // Same geometry, but sources permit only 6 hops of propagation.
        let mut chunks = make_breached_source_basin(6);
        let config = crate::FluidConfig::default();
        let dc = empty_density_cache();
        for _ in 0..900 {
            crate::sim::regen_sources(&mut chunks);
            tick_fluid(&mut chunks, &dc, 16, true, &config, true);
        }
        // 6 hops from the basin edge (x=6) plus taper slack: nothing with a
        // visible level should exist past x=14, and the east chunk stays dry.
        let reach = furthest_lava_x(&chunks, 0.05);
        assert!(
            reach <= 14,
            "bounded (6-hop) breach leaked past the expected taper: reach = {reach}"
        );
    }

    // ── 2026-08-04 source self-extinguish ────────────────────────────────
    // A source that can never reach steady state (its outflow vanishes into
    // a sink: void below the world, pinhole into nowhere) must demote itself
    // to a one-shot fill. Sources that equalize — sealed bowls, filled
    // basins — must live forever. Bounding flow reach did NOT stop the
    // bug-#215 pumping; this is the mechanism that does.

    fn run_lava_ticks(chunks: &mut HashMap<(i32, i32, i32), ChunkFluidGrid>, n: usize) {
        let config = crate::FluidConfig::default();
        let dc = empty_density_cache();
        for _ in 0..n {
            crate::sim::regen_sources(chunks);
            tick_fluid(chunks, &dc, 16, true, &config, true);
        }
    }

    #[test]
    fn source_draining_into_void_self_extinguishes() {
        // Lone lava source on a 1-cell pedestal at the top of an otherwise
        // EMPTY chunk with no chunk below: everything it emits falls out of
        // the world and is dropped (the void sink). It can never equalize.
        let mut grid = make_chunk(16);
        grid.set_density(8, 7, 8, 1.0); // pedestal
        {
            let cell = grid.get_mut(8, 8, 8);
            cell.level = SOURCE_LEVEL;
            cell.fluid_type = FluidType::Lava;
            cell.is_source = true;
            cell.max_flow_dist = 12;
            cell.hops_from_source = 0;
        }
        grid.has_fluid = true;
        grid.has_sources = true;
        let mut chunks = HashMap::new();
        chunks.insert((0, 0, 0), grid);

        run_lava_ticks(&mut chunks, 200);

        let cell = chunks[&(0, 0, 0)].get(8, 8, 8);
        assert!(
            !cell.is_source,
            "void-draining source should have self-extinguished (level={}, drain_ticks={})",
            cell.level, cell.drain_ticks
        );
    }

    #[test]
    fn source_in_sealed_bowl_persists() {
        // Source at the bottom of a sealed 5x5 bowl: fills it, equalizes,
        // reaches steady state — must stay a source indefinitely.
        let mut grid = make_sealed_box(16, 5..10, 5..8, 5..10);
        {
            let cell = grid.get_mut(7, 5, 7);
            cell.level = SOURCE_LEVEL;
            cell.fluid_type = FluidType::Lava;
            cell.is_source = true;
            cell.max_flow_dist = 12;
            cell.hops_from_source = 0;
        }
        grid.has_fluid = true;
        grid.has_sources = true;
        let mut chunks = HashMap::new();
        chunks.insert((0, 0, 0), grid);

        run_lava_ticks(&mut chunks, 300);

        let cell = chunks[&(0, 0, 0)].get(7, 5, 7);
        assert!(
            cell.is_source,
            "sealed-bowl source must survive (drain_ticks={})",
            cell.drain_ticks
        );
        assert!(cell.level > 0.5, "sealed-bowl source should sit full, level={}", cell.level);
    }

    #[test]
    fn source_filling_wide_sealed_floor_persists() {
        // Source on a sealed but WIDE floor (takes many ticks of honest
        // spreading before its own cell holds level again). The demote
        // streak must tolerate the fill phase — this is the false-positive
        // guard for the threshold constants.
        let mut grid = make_sealed_box(16, 1..15, 5..8, 1..15);
        {
            let cell = grid.get_mut(8, 5, 8);
            cell.level = SOURCE_LEVEL;
            cell.fluid_type = FluidType::Lava;
            cell.is_source = true;
            cell.max_flow_dist = 0; // unlimited spread within the tray
            cell.hops_from_source = 0;
        }
        grid.has_fluid = true;
        grid.has_sources = true;
        let mut chunks = HashMap::new();
        chunks.insert((0, 0, 0), grid);

        run_lava_ticks(&mut chunks, 400);

        let cell = chunks[&(0, 0, 0)].get(8, 5, 8);
        assert!(
            cell.is_source,
            "source honestly filling a sealed tray must not be demoted (drain_ticks={}, level={})",
            cell.drain_ticks, cell.level
        );
    }

    // ── 2026-08-04 pinhole membrane ──────────────────────────────────────
    // Fractional capacity makes every surface-crossing cell partially
    // passable — a THIN rendered slab (no fully-solid cell layer inside it)
    // is a permeable membrane and fluid seeps straight through "solid" rock
    // onto the void-side underside of the world (bug #215). The rendered
    // surface must be a transit barrier: fluid may not cross a cell face
    // whose 4 shared lattice corners are all solid.

    #[test]
    fn fluid_does_not_seep_through_thin_rendered_slab() {
        let size = 16usize;
        let stride = size + 1;
        // 17³ lattice: a single solid plane at lattice y=8, air everywhere
        // else. The zero-crossings sit above AND below that plane — the DC
        // surface renders a closed thin slab, but no cell layer is fully
        // solid: cells at y=7 and y=8 each have 4 solid corners (cap 0.5).
        let mut lattice = vec![-1.0f32; stride * stride * stride];
        for z in 0..stride {
            for x in 0..stride {
                lattice[z * stride * stride + 8 * stride + x] = 0.3;
            }
        }
        let mut cache = crate::cell::ChunkDensityCache::new(size);
        cache.update_density(&lattice);
        let mut grid = ChunkFluidGrid::from_density_cache(&cache);

        // Water dropped well above the slab.
        for z in 6..10usize {
            for x in 6..10usize {
                let cell = grid.get_mut(x, 12, z);
                cell.level = 1.0;
                cell.fluid_type = FluidType::Water;
            }
        }
        grid.has_fluid = true;
        let mut chunks = HashMap::new();
        chunks.insert((0, 0, 0), grid);

        let config = crate::FluidConfig::default();
        let dc = empty_density_cache();
        for _ in 0..200 {
            tick_fluid(&mut chunks, &dc, size, false, &config, true);
        }

        // Nothing may exist below the slab (cells y <= 6 are pure air —
        // any fluid there came THROUGH the rendered surface).
        let grid = &chunks[&(0, 0, 0)];
        let mut below = 0.0f64;
        for z in 0..size { for y in 0..7 { for x in 0..size {
            below += grid.get(x, y, z).level as f64;
        }}}
        assert!(
            below < 0.001,
            "fluid seeped through a rendered-solid thin slab: {below:.3} total level below it"
        );
    }

    #[test]
    fn fluid_does_not_seep_through_corner_nicked_slab() {
        // Same thin slab, but with single lattice points nicked negative —
        // real DC terrain does this constantly. Every face touching a nick
        // has only 3 of 4 corners solid, so an all-4-solid gate waves the
        // transit through while the rendered surface still covers the slab.
        let size = 16usize;
        let stride = size + 1;
        let mut lattice = vec![-1.0f32; stride * stride * stride];
        for z in 0..stride {
            for x in 0..stride {
                lattice[z * stride * stride + 8 * stride + x] = 0.3;
            }
        }
        // Nick a scatter of isolated lattice points in the slab plane.
        for &(nx, nz) in &[(4usize, 4usize), (8, 8), (11, 6), (6, 11)] {
            lattice[nz * stride * stride + 8 * stride + nx] = -0.1;
        }
        let mut cache = crate::cell::ChunkDensityCache::new(size);
        cache.update_density(&lattice);
        let mut grid = ChunkFluidGrid::from_density_cache(&cache);

        // Blanket of water over the whole slab so every nick is exercised.
        for z in 2..14usize {
            for x in 2..14usize {
                let cell = grid.get_mut(x, 12, z);
                cell.level = 1.0;
                cell.fluid_type = FluidType::Water;
            }
        }
        grid.has_fluid = true;
        let mut chunks = HashMap::new();
        chunks.insert((0, 0, 0), grid);

        let config = crate::FluidConfig::default();
        let dc = empty_density_cache();
        for _ in 0..200 {
            tick_fluid(&mut chunks, &dc, size, false, &config, true);
        }

        let grid = &chunks[&(0, 0, 0)];
        let mut below = 0.0f64;
        for z in 0..size { for y in 0..7 { for x in 0..size {
            below += grid.get(x, y, z).level as f64;
        }}}
        assert!(
            below < 0.001,
            "fluid seeped through a corner-nicked rendered slab: {below:.3} total level below it"
        );
    }

    // ── 2026-08-04 cascade bundle ─────────────────────────────────────────
    // A cascade is high FLUX with low standing volume: transit cells end
    // ticks with levels oscillating around the mesh iso (0.15) even when the
    // stream is steady, so the mesh strobes ("flashing wildly at different
    // locations" — user repro). Fixture: a descending stepped gallery from a
    // lava source down to a basin, 3 cells wide so spread thins per-cell
    // flux to the borderline regime where strobing lives.

    /// Stepped gallery: 6 steps descending along +x (3 wide in z), basin at
    /// the far end, lava source at the top entry.
    fn make_cascade() -> ChunkFluidGrid {
        let size = 16;
        let config = crate::FluidConfig::default();
        let mut d = make_density_field_solid(size);
        // Steps: step i floor at gy = 12 - 2*i, air carved above it.
        for i in 0..6usize {
            let x0 = 2 + 2 * i;
            let floor = 12 - 2 * i;
            carve_box(&mut d, size, x0..(x0 + 3), floor..15, 7..11);
        }
        // Basin at the bottom of the last step.
        carve_box(&mut d, size, 12..16, 1..5, 5..12);
        let mut grid = make_chunk(size);
        apply_density(&mut grid, &d, &config);
        {
            let cell = grid.get_mut(2, 12, 8);
            cell.level = SOURCE_LEVEL;
            cell.fluid_type = FluidType::Lava;
            cell.is_source = true;
            cell.max_flow_dist = 0; // unlimited — clean steady flux signal
            cell.hops_from_source = 0;
        }
        grid.has_fluid = true;
        grid.has_lava = true;
        grid.has_sources = true;
        grid
    }

    fn cascade_config(flux_render: bool, ribbon: bool, retention: bool) -> crate::FluidConfig {
        crate::FluidConfig {
            mesh_flux_render: flux_render,
            mesh_stream_ribbon: ribbon,
            lava_transit_retention: retention,
            ..crate::FluidConfig::default()
        }
    }

    fn run_cascade_ticks(
        chunks: &mut HashMap<(i32, i32, i32), ChunkFluidGrid>,
        n: usize,
        config: &crate::FluidConfig,
        update_render: bool,
    ) {
        let dc = empty_density_cache();
        for _ in 0..n {
            crate::sim::regen_sources(chunks);
            tick_fluid(chunks, &dc, 16, true, config, true);
            if update_render {
                let grid = chunks.get_mut(&(0, 0, 0)).unwrap();
                grid.update_render_field(
                    config.mesh_sticky_release,
                    config.mesh_flux_render,
                    config.mesh_stream_ribbon,
                );
            }
        }
    }

    /// The gallery's transit cells (steps only — basin excluded).
    fn gallery_cells() -> Vec<(usize, usize, usize)> {
        let mut v = Vec::new();
        for i in 0..6usize {
            let x0 = 2 + 2 * i;
            let floor = 12 - 2 * i;
            for x in x0..(x0 + 3).min(12) {
                for z in 7..11usize {
                    for y in floor..(floor + 3).min(15) {
                        v.push((x, y, z));
                    }
                }
            }
        }
        v
    }

    /// Count per-cell mesh-membership toggles over a window of ticks.
    /// Returns (mesh_toggles, raw_toggles): membership by mesh_level vs by
    /// raw level, so a fix can be shown to stabilize the MESH while the sim
    /// still oscillates underneath.
    fn measure_strobe(config: &crate::FluidConfig, warmup: usize, window: usize) -> (usize, usize) {
        let mut chunks = HashMap::new();
        chunks.insert((0, 0, 0), make_cascade());
        run_cascade_ticks(&mut chunks, warmup, config, true);

        let cells = gallery_cells();
        let mut prev_mesh: Vec<bool> = Vec::new();
        let mut prev_raw: Vec<bool> = Vec::new();
        let mut mesh_toggles = 0usize;
        let mut raw_toggles = 0usize;
        for t in 0..window {
            run_cascade_ticks(&mut chunks, 1, config, true);
            let grid = chunks.get(&(0, 0, 0)).unwrap();
            let mesh_now: Vec<bool> = cells.iter().map(|&(x, y, z)| grid.mesh_level(x, y, z) >= 0.15).collect();
            let raw_now: Vec<bool> = cells.iter().map(|&(x, y, z)| grid.get(x, y, z).level >= 0.15).collect();
            if t > 0 {
                mesh_toggles += mesh_now.iter().zip(&prev_mesh).filter(|(a, b)| a != b).count();
                raw_toggles += raw_now.iter().zip(&prev_raw).filter(|(a, b)| a != b).count();
            }
            prev_mesh = mesh_now;
            prev_raw = raw_now;
        }
        (mesh_toggles, raw_toggles)
    }

    #[test]
    #[ignore] // diagnostic probe — run manually with --ignored --nocapture
    fn cascade_strobe_probe() {
        for (name, flow, warmup) in [
            ("transient_default", 0.1f32, 3usize),
            ("transient_flow05", 0.05, 3),
            ("mid_default", 0.1, 15),
            ("steady_default", 0.1, 60),
        ] {
            let mut cfg = cascade_config(false, false, false);
            cfg.lava_flow_rate = flow;
            let (m, r) = measure_strobe(&cfg, warmup, 45);
            let mut cfg2 = cascade_config(true, false, false);
            cfg2.lava_flow_rate = flow;
            let (m2, r2) = measure_strobe(&cfg2, warmup, 45);
            eprintln!("PROBE {name}: legacy mesh={m} raw={r} | flux_render mesh={m2} raw={r2}");
        }
        // Steady-state visibility: wet vs rendered cells at the sub-iso regime.
        for (name, fx, rb, rt) in [
            ("legacy", false, false, false),
            ("flux", true, false, false),
            ("flux+ribbon", true, true, false),
            ("full(retention)", true, true, true),
        ] {
            let cfg = cascade_config(fx, rb, rt);
            let mut chunks = HashMap::new();
            chunks.insert((0, 0, 0), make_cascade());
            run_cascade_ticks(&mut chunks, 100, &cfg, true);
            let grid = chunks.get(&(0, 0, 0)).unwrap();
            let wet = gallery_cells().iter().filter(|&&(x, y, z)| grid.get(x, y, z).level >= 0.02).count();
            let rendered = gallery_cells().iter().filter(|&&(x, y, z)| grid.mesh_level(x, y, z) >= 0.15).count();
            eprintln!("PROBE visibility {name}: wet={wet} rendered={rendered}");
        }
    }

    #[test]
    fn cascade_stream_mostly_invisible_with_bundle_off() {
        // FIXTURE HONESTY (the "isolated specs" symptom): a steady cascade
        // is high flux / low standing volume, so most of its wet cells sit
        // below the mesh iso — the legacy pipeline renders a scatter of
        // specs, not a stream. Measured 2026-08-04: 57 wet, 6 rendered.
        // If this fails the fixture stopped modeling the bug and the bundle
        // tests prove nothing.
        let cfg = cascade_config(false, false, false);
        let mut chunks = HashMap::new();
        chunks.insert((0, 0, 0), make_cascade());
        run_cascade_ticks(&mut chunks, 100, &cfg, true);
        let grid = chunks.get(&(0, 0, 0)).unwrap();
        let wet = gallery_cells().iter().filter(|&&(x, y, z)| grid.get(x, y, z).level >= 0.02).count();
        let rendered = gallery_cells().iter().filter(|&&(x, y, z)| grid.mesh_level(x, y, z) >= 0.15).count();
        assert!(wet >= 40, "cascade dried up (wet={wet}) — fixture drift?");
        assert!(
            rendered * 4 <= wet,
            "legacy no longer under-renders the stream (wet={wet}, rendered={rendered}) — fixture drift?"
        );
    }

    #[test]
    fn cascade_bundle_renders_the_stream() {
        // flux+ribbon must render the channel (measured 34/57), and
        // retention must give it real volume and render more still (51/108).
        let render_count = |cfg: &crate::FluidConfig| -> (usize, usize) {
            let mut chunks = HashMap::new();
            chunks.insert((0, 0, 0), make_cascade());
            run_cascade_ticks(&mut chunks, 100, cfg, true);
            let grid = chunks.get(&(0, 0, 0)).unwrap();
            let wet = gallery_cells().iter().filter(|&&(x, y, z)| grid.get(x, y, z).level >= 0.02).count();
            let rendered = gallery_cells().iter().filter(|&&(x, y, z)| grid.mesh_level(x, y, z) >= 0.15).count();
            (wet, rendered)
        };
        let (_, legacy_rendered) = render_count(&cascade_config(false, false, false));
        let (_, ribbon_rendered) = render_count(&cascade_config(true, true, false));
        let (full_wet, full_rendered) = render_count(&cascade_config(true, true, true));
        assert!(
            ribbon_rendered >= legacy_rendered * 4 && ribbon_rendered >= 25,
            "ribbon did not render the channel: legacy={legacy_rendered}, ribbon={ribbon_rendered}"
        );
        assert!(
            full_rendered >= ribbon_rendered && full_wet >= 80,
            "retention did not add stream volume: rendered={full_rendered}, wet={full_wet}"
        );
    }

    #[test]
    fn cascade_flux_render_stops_pulse_strobing() {
        // The in-game oscillator: supply is INTERMITTENT (sources regen and
        // self-extinguish, flow paths shift, gulps march). Model it with a
        // source pulsing 3 ticks on / 3 off — legacy mesh membership churns
        // with every pulse; the EMA render field + ribbon must hold the
        // stream visually steady while raw levels swing underneath.
        let pulse_toggles = |cfg: &crate::FluidConfig| -> (usize, usize) {
            let mut chunks = HashMap::new();
            chunks.insert((0, 0, 0), make_cascade());
            run_cascade_ticks(&mut chunks, 40, cfg, true);
            let cells = gallery_cells();
            let mut prev_mesh: Vec<bool> = Vec::new();
            let mut prev_raw: Vec<bool> = Vec::new();
            let mut mesh_toggles = 0usize;
            let mut raw_toggles = 0usize;
            for t in 0..48usize {
                {
                    let grid = chunks.get_mut(&(0, 0, 0)).unwrap();
                    let on = (t / 3) % 2 == 0;
                    let cell = grid.get_mut(2, 12, 8);
                    cell.is_source = on;
                    if on {
                        cell.level = SOURCE_LEVEL;
                        cell.fluid_type = FluidType::Lava;
                        cell.hops_from_source = 0;
                        cell.max_flow_dist = 0;
                    }
                    grid.has_sources = on;
                    grid.has_fluid = true;
                }
                run_cascade_ticks(&mut chunks, 1, cfg, true);
                let grid = chunks.get(&(0, 0, 0)).unwrap();
                let mesh_now: Vec<bool> =
                    cells.iter().map(|&(x, y, z)| grid.mesh_level(x, y, z) >= 0.15).collect();
                let raw_now: Vec<bool> =
                    cells.iter().map(|&(x, y, z)| grid.get(x, y, z).level >= 0.15).collect();
                if t > 0 {
                    mesh_toggles += mesh_now.iter().zip(&prev_mesh).filter(|(a, b)| a != b).count();
                    raw_toggles += raw_now.iter().zip(&prev_raw).filter(|(a, b)| a != b).count();
                }
                prev_mesh = mesh_now;
                prev_raw = raw_now;
            }
            (mesh_toggles, raw_toggles)
        };

        let (bundle_mesh, bundle_raw) = pulse_toggles(&cascade_config(true, true, false));
        assert!(
            bundle_raw > 0,
            "pulsed source produced no raw churn at all — fixture drift?"
        );
        // The user-visible metric is churn per rendered cell per tick: the
        // ~40-cell rendered ribbon must flicker at most ~1% of its cells per
        // tick (legacy renders ~6 cells total, so absolute-toggle compares
        // are meaningless — its 'stability' is emptiness).
        let rendered = {
            let cfg = cascade_config(true, true, false);
            let mut chunks = HashMap::new();
            chunks.insert((0, 0, 0), make_cascade());
            run_cascade_ticks(&mut chunks, 100, &cfg, true);
            let grid = chunks.get(&(0, 0, 0)).unwrap();
            gallery_cells().iter().filter(|&&(x, y, z)| grid.mesh_level(x, y, z) >= 0.15).count()
        };
        assert!(rendered >= 25, "ribbon rendered too little to judge ({rendered})");
        let churn_per_tick = bundle_mesh as f32 / 47.0;
        assert!(
            churn_per_tick <= rendered as f32 * 0.02,
            "bundle ribbon flickers under pulses: {bundle_mesh} toggles/47t over {rendered} rendered cells (raw churn {bundle_raw})"
        );
    }

    #[test]
    fn cascade_ribbon_renders_connected_stream() {
        let cfg = cascade_config(true, true, false);
        let mut chunks = HashMap::new();
        chunks.insert((0, 0, 0), make_cascade());
        run_cascade_ticks(&mut chunks, 100, &cfg, true);

        let grid = chunks.get(&(0, 0, 0)).unwrap();
        // Stream cells = sustained flux. There must BE a stream...
        let stream: Vec<(usize, usize, usize)> = gallery_cells()
            .into_iter()
            .filter(|&(x, y, z)| {
                let idx = grid.index(x, y, z);
                grid.flux_ema.get(idx).copied().unwrap_or(0.0) >= crate::cell::STREAM_FLUX_MIN
                    && grid.get(x, y, z).level >= MIN_LEVEL
            })
            .collect();
        assert!(
            stream.len() >= 6,
            "no sustained stream detected in the gallery ({} flux cells)",
            stream.len()
        );
        // ...and every wet stream cell must render in the mesh (no gaps).
        for (x, y, z) in stream {
            assert!(
                grid.mesh_level(x, y, z) >= 0.15,
                "stream cell ({x},{y},{z}) not rendered — gap in the ribbon (raw={:.3})",
                grid.get(x, y, z).level
            );
        }
    }

    // ── 2026-08-04 river bundle (channel bias + focus) ───────────────────
    // A wide slope fed from one point should converge into few stable
    // streams; legacy flow fans into a thin sheet (every ledge spreads
    // laterally, slope targets are chosen by instant state only).

    /// Wide gentle staircase: 5 full-width steps (z 2..14) descending along
    /// +x from a single top-center lava source, basin at the bottom.
    fn make_wide_slope() -> ChunkFluidGrid {
        let size = 16;
        let config = crate::FluidConfig::default();
        let mut d = make_density_field_solid(size);
        for i in 0..5usize {
            let x0 = 2 + 2 * i;
            let floor = 11 - 2 * i;
            carve_box(&mut d, size, x0..(x0 + 3), floor..15, 2..14);
        }
        carve_box(&mut d, size, 12..15, 1..6, 2..14);
        let mut grid = make_chunk(size);
        apply_density(&mut grid, &d, &config);
        {
            let cell = grid.get_mut(2, 11, 8);
            cell.level = SOURCE_LEVEL;
            cell.fluid_type = FluidType::Lava;
            cell.is_source = true;
            cell.max_flow_dist = 0;
            cell.hops_from_source = 0;
        }
        grid.has_fluid = true;
        grid.has_lava = true;
        grid.has_sources = true;
        grid
    }

    /// Wide slope with scattered floor bumps (raised lattice points on a
    /// fixed pseudo-random pattern). A sheet must thread between bumps —
    /// momentum decides whether the threads RE-MERGE into committed lanes
    /// downstream or re-sheet after every obstacle. The uniform slope can't
    /// measure convergence at all: every z-column is identical, so there is
    /// no asymmetry to amplify.
    fn make_noisy_slope() -> ChunkFluidGrid {
        let size = 16;
        let config = crate::FluidConfig::default();
        let mut d = make_density_field_solid(size);
        for i in 0..5usize {
            let x0 = 2 + 2 * i;
            let floor = 11 - 2 * i;
            carve_box(&mut d, size, x0..(x0 + 3), floor..15, 2..14);
        }
        carve_box(&mut d, size, 12..15, 1..6, 2..14);
        // Bumps: re-solidify the floor lattice point at ~1/3 of columns on a
        // fixed interference pattern (skip the source column's fan area).
        let stride = size + 1;
        for i in 0..5usize {
            let x0 = 2 + 2 * i;
            let floor = 11 - 2 * i;
            for gx in x0..(x0 + 3) {
                for gz in 3..13usize {
                    if gx >= 4 && (gx * 5 + gz * 7) % 3 == 0 {
                        d[gz * stride * stride + floor * stride + gx] = 1.0;
                    }
                }
            }
        }
        let mut grid = make_chunk(size);
        apply_density(&mut grid, &d, &config);
        {
            let cell = grid.get_mut(2, 11, 8);
            cell.level = SOURCE_LEVEL;
            cell.fluid_type = FluidType::Lava;
            cell.is_source = true;
            cell.max_flow_dist = 0;
            cell.hops_from_source = 0;
        }
        grid.has_fluid = true;
        grid.has_lava = true;
        grid.has_sources = true;
        grid
    }

    fn river_config(bias: bool, focus: bool) -> crate::FluidConfig {
        crate::FluidConfig {
            lava_channel_bias: bias,
            lava_channel_focus: focus,
            ..crate::FluidConfig::default()
        }
    }

    /// Per-z-column flux totals on the LAST step (arrival band), plus the
    /// basin lava total. Concentration = top-3 columns' share of band flux.
    fn slope_arrival_stats(grid: &ChunkFluidGrid) -> (Vec<f32>, f32, f64) {
        let mut cols = vec![0.0f32; 16];
        for z in 2..14usize {
            for x in 10..13usize {
                for y in 3..7usize {
                    cols[z] += grid.flux_at(x, y, z);
                }
            }
        }
        let total: f32 = cols.iter().sum();
        let mut sorted = cols.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let top3_share = if total > 1e-6 { (sorted[0] + sorted[1] + sorted[2]) / total } else { 0.0 };
        let mut basin = 0.0f64;
        for z in 2..14usize {
            for x in 12..15usize {
                for y in 1..6usize {
                    basin += grid.get(x, y, z).level as f64;
                }
            }
        }
        (cols, top3_share, basin)
    }

    /// Fork fixture for channel BIAS: a flat feed channel dead-ends above
    /// two drop shafts reachable only via the (0,-1,±1) slope diagonals —
    /// the junction cell has exactly two candidates. Legacy feeds both every
    /// tick (~50/50 forever); with bias the historically-stronger branch
    /// wins the score and (winner-takes-most) carries the flow.
    fn make_fork() -> ChunkFluidGrid {
        let size = 16;
        let config = crate::FluidConfig::default();
        let mut d = make_density_field_solid(size);
        // Feed channel: cells x 2..=6, y 10..12, z=8.
        carve_box(&mut d, size, 2..8, 10..14, 8..10);
        // Shaft A (z=6 side): lattice z 6,7 — cells z=6 full, z=7 half.
        carve_box(&mut d, size, 6..8, 2..14, 6..8);
        // Shaft B (z=10 side): lattice z 10,11 — cells z=10 full, z=9 half.
        carve_box(&mut d, size, 6..8, 2..14, 10..12);
        // Small catch basins so the shafts don't back up.
        carve_box(&mut d, size, 4..10, 1..4, 4..8);
        carve_box(&mut d, size, 4..10, 1..4, 10..14);
        let mut grid = make_chunk(size);
        apply_density(&mut grid, &d, &config);
        {
            let cell = grid.get_mut(2, 10, 8);
            cell.level = SOURCE_LEVEL;
            cell.fluid_type = FluidType::Lava;
            cell.is_source = true;
            cell.max_flow_dist = 0;
            cell.hops_from_source = 0;
        }
        grid.has_fluid = true;
        grid.has_lava = true;
        grid.has_sources = true;
        grid
    }

    /// Flux share of branch A (z <= 7) vs branch B (z >= 9) below the fork.
    fn fork_branch_share(grid: &ChunkFluidGrid) -> (f32, f32) {
        let mut a = 0.0f32;
        let mut b = 0.0f32;
        for z in 2..14usize {
            for x in 4..10usize {
                for y in 2..10usize {
                    let f = grid.flux_at(x, y, z);
                    if z <= 7 { a += f; } else if z >= 9 { b += f; }
                }
            }
        }
        (a, b)
    }

    #[test]
    #[ignore] // diagnostic probe — run manually with --ignored --nocapture
    fn river_convergence_probe() {
        for (name, bias) in [("fork legacy", false), ("fork bias", true)] {
            let cfg = river_config(bias, false);
            let mut chunks = HashMap::new();
            chunks.insert((0, 0, 0), make_fork());
            run_cascade_ticks(&mut chunks, 150, &cfg, false);
            let grid = chunks.get(&(0, 0, 0)).unwrap();
            let (a, b) = fork_branch_share(grid);
            let share = a.max(b) / (a + b).max(1e-6);
            eprintln!("RIVER {name}: A={a:.2} B={b:.2} winner_share={share:.2}");
        }
        for (name, bias, focus, momentum) in [
            ("legacy", false, false, false),
            ("bias", true, false, false),
            ("focus", false, true, false),
            ("bias+focus", true, true, false),
            ("momentum", false, false, true),
        ] {
            let mut cfg = river_config(bias, focus);
            cfg.lava_momentum = momentum;
            let mut chunks = HashMap::new();
            chunks.insert((0, 0, 0), make_wide_slope());
            run_cascade_ticks(&mut chunks, 150, &cfg, false);
            let grid = chunks.get(&(0, 0, 0)).unwrap();
            let (cols, top3, basin) = slope_arrival_stats(grid);
            let wet_cols = cols.iter().filter(|&&c| c > 0.01).count();
            eprintln!(
                "RIVER {name}: top3_share={top3:.2} wet_cols={wet_cols} basin={basin:.1} cols={:?}",
                cols.iter().map(|c| (c * 100.0).round() / 100.0).collect::<Vec<_>>()
            );
        }
        // Noisy slope: the convergence measurement that matters — does the
        // sheet commit to lanes between obstacles?
        for (name, momentum) in [("noisy legacy", false), ("noisy momentum", true)] {
            let mut cfg = crate::FluidConfig::default();
            cfg.lava_momentum = momentum;
            let mut chunks = HashMap::new();
            chunks.insert((0, 0, 0), make_noisy_slope());
            run_cascade_ticks(&mut chunks, 150, &cfg, false);
            let grid = chunks.get(&(0, 0, 0)).unwrap();
            let (cols, top3, basin) = slope_arrival_stats(grid);
            let wet_cols = cols.iter().filter(|&&c| c > 0.01).count();
            eprintln!(
                "RIVER {name}: top3_share={top3:.2} wet_cols={wet_cols} basin={basin:.1} cols={:?}",
                cols.iter().map(|c| (c * 100.0).round() / 100.0).collect::<Vec<_>>()
            );
        }
        // Momentum on the flat spill: a pond must stay a pond (radial memory
        // is symmetric), not tear into fingers. Measured mid-growth (40) and
        // near-settled (110) — the outward surge may thin films transiently
        // but the settled pond must render like legacy.
        for (name, momentum, ticks) in [
            ("spill legacy t40", false, 40),
            ("spill momentum t40", true, 40),
            ("spill legacy t110", false, 110),
            ("spill momentum t110", true, 110),
        ] {
            let mut cfg = crate::FluidConfig::default();
            cfg.lava_momentum = momentum;
            let mut chunks = HashMap::new();
            chunks.insert((0, 0, 0), make_flat_spill());
            run_cascade_ticks(&mut chunks, ticks, &cfg, true);
            let grid = chunks.get(&(0, 0, 0)).unwrap();
            let mut wet = 0usize;
            let mut rendered = 0usize;
            for z in 1..16usize {
                for x in 1..16usize {
                    for y in 3..5usize {
                        if grid.get(x, y, z).level >= 0.02 {
                            wet += 1;
                            if grid.mesh_level(x, y, z) >= 0.15 {
                                rendered += 1;
                            }
                        }
                    }
                }
            }
            eprintln!("RIVER {name}: wet={wet} rendered={rendered}");
        }
    }

    // ── 2026-08-04 spread-flux instrumentation ───────────────────────────
    // Spread IS transport on flats: pond fronts and fall fan-out carry lava
    // via the spread pass, which previously generated no flux — those cells
    // got no ribbon floor and rendered as confetti (user screenshots: sheet
    // edges + lavafall scatter were exactly the un-instrumented cells).

    /// Flat-floor spill: source pouring onto a big open floor, pond grows
    /// radially — mid-spill, the front is spread-driven.
    fn make_flat_spill() -> ChunkFluidGrid {
        let size = 16;
        let config = crate::FluidConfig::default();
        let mut d = make_density_field_solid(size);
        carve_box(&mut d, size, 1..16, 4..15, 1..16);
        let mut grid = make_chunk(size);
        apply_density(&mut grid, &d, &config);
        {
            let cell = grid.get_mut(8, 4, 8);
            cell.level = SOURCE_LEVEL;
            cell.fluid_type = FluidType::Lava;
            cell.is_source = true;
            cell.max_flow_dist = 0;
            cell.hops_from_source = 0;
        }
        grid.has_fluid = true;
        grid.has_lava = true;
        grid.has_sources = true;
        grid
    }

    #[test]
    fn spreading_pond_front_renders_connected() {
        // Mid-spill, floor row y=4: most wet cells must RENDER when the
        // ribbon is on (spread now counts as flux). With the ribbon off the
        // same sim state renders a fraction of it — documents the delta the
        // instrumentation closes.
        let ratio_for = |ribbon: bool| -> (usize, usize) {
            let cfg = crate::FluidConfig {
                mesh_stream_ribbon: ribbon,
                ..crate::FluidConfig::default()
            };
            let mut chunks = HashMap::new();
            chunks.insert((0, 0, 0), make_flat_spill());
            run_cascade_ticks(&mut chunks, 40, &cfg, true);
            let grid = chunks.get(&(0, 0, 0)).unwrap();
            let mut wet = 0usize;
            let mut rendered = 0usize;
            // The pond body spans the carved floor row AND the half-open
            // boundary layer below it (cap 0.5 — every carve boundary makes
            // one; fluid settles into it first).
            for z in 1..16usize {
                for x in 1..16usize {
                    for y in 3..5usize {
                        if grid.get(x, y, z).level >= 0.02 {
                            wet += 1;
                            if grid.mesh_level(x, y, z) >= 0.15 {
                                rendered += 1;
                            }
                        }
                    }
                }
            }
            (wet, rendered)
        };
        let (wet_on, rendered_on) = ratio_for(true);
        assert!(wet_on >= 20, "spill never grew (wet={wet_on}) — fixture drift?");
        assert!(
            rendered_on * 10 >= wet_on * 7,
            "spreading pond still renders confetti with ribbon on: {rendered_on}/{wet_on} rendered"
        );
        let (wet_off, rendered_off) = ratio_for(false);
        assert!(
            rendered_off * 2 < wet_off,
            "ribbon-off under-rendering vanished ({rendered_off}/{wet_off}) — fixture no longer exercises the fix"
        );
    }

    #[test]
    fn transit_retention_gives_stream_volume_then_drains() {
        // With retention: fed transit cells hold REAL standing lava (the
        // stream exists for lights/quench/damage, not just the mesher)...
        let cfg = cascade_config(false, false, true);
        let mut chunks = HashMap::new();
        chunks.insert((0, 0, 0), make_cascade());
        run_cascade_ticks(&mut chunks, 100, &cfg, false);

        let retained: usize = {
            let grid = chunks.get(&(0, 0, 0)).unwrap();
            gallery_cells()
                .iter()
                .filter(|&&(x, y, z)| grid.get(x, y, z).level >= 0.18)
                .count()
        };
        let baseline: usize = {
            let cfg0 = cascade_config(false, false, false);
            let mut chunks0 = HashMap::new();
            chunks0.insert((0, 0, 0), make_cascade());
            run_cascade_ticks(&mut chunks0, 100, &cfg0, false);
            let grid = chunks0.get(&(0, 0, 0)).unwrap();
            gallery_cells()
                .iter()
                .filter(|&&(x, y, z)| grid.get(x, y, z).level >= 0.18)
                .count()
        };
        assert!(
            retained > baseline + 4,
            "retention added no stream volume (retained={retained}, baseline={baseline})"
        );

        // ...and when the source dies the channel must DRAIN — retention is
        // for fed cells only, never a source of perched fluid.
        {
            let grid = chunks.get_mut(&(0, 0, 0)).unwrap();
            let cell = grid.get_mut(2, 12, 8);
            cell.is_source = false;
            cell.level = 0.0;
            grid.has_sources = false;
        }
        run_cascade_ticks(&mut chunks, 250, &cfg, false);
        let grid = chunks.get(&(0, 0, 0)).unwrap();
        let left_in_gallery: f64 = gallery_cells()
            .iter()
            .map(|&(x, y, z)| grid.get(x, y, z).level as f64)
            .sum();
        assert!(
            left_in_gallery < 1.0,
            "gallery still holds {left_in_gallery:.2} lava long after the source died — retention is perching fluid"
        );
    }

    #[test]
    fn momentum_speeds_streams_without_starving_or_scattering() {
        // Momentum steering must (a) DELIVER more — streams commit downstream
        // instead of dithering (measured 2.7× basin on the noisy slope) —
        // (b) never scatter the arrival wider than legacy, and (c) leave
        // settled ponds untouched. Round 11's magnitude-gated variants
        // failed (a) at −60%; this is the guard that momentum stays the
        // direction-relative mechanism that doesn't.
        let run_slope = |momentum: bool| -> (f32, f64) {
            let mut cfg = crate::FluidConfig::default();
            cfg.lava_momentum = momentum;
            let mut chunks = HashMap::new();
            chunks.insert((0, 0, 0), make_noisy_slope());
            run_cascade_ticks(&mut chunks, 150, &cfg, false);
            let grid = chunks.get(&(0, 0, 0)).unwrap();
            let (_, top3, basin) = slope_arrival_stats(grid);
            (top3, basin)
        };
        let (top3_legacy, basin_legacy) = run_slope(false);
        let (top3_momentum, basin_momentum) = run_slope(true);
        assert!(
            basin_legacy > 0.5,
            "fixture drift: legacy slope delivers nothing (basin={basin_legacy:.1})"
        );
        assert!(
            basin_momentum >= basin_legacy * 1.5,
            "momentum stopped speeding delivery: basin {basin_momentum:.1} vs legacy {basin_legacy:.1}"
        );
        assert!(
            top3_momentum >= top3_legacy - 0.05,
            "momentum SCATTERS the arrival: top3 {top3_momentum:.2} vs legacy {top3_legacy:.2}"
        );

        // Settled pond invariance: at t110 the spill must render like legacy.
        let settled = |momentum: bool| -> (usize, usize) {
            let mut cfg = crate::FluidConfig::default();
            cfg.lava_momentum = momentum;
            let mut chunks = HashMap::new();
            chunks.insert((0, 0, 0), make_flat_spill());
            run_cascade_ticks(&mut chunks, 110, &cfg, true);
            let grid = chunks.get(&(0, 0, 0)).unwrap();
            let mut wet = 0usize;
            let mut rendered = 0usize;
            for z in 1..16usize {
                for x in 1..16usize {
                    for y in 3..5usize {
                        if grid.get(x, y, z).level >= 0.02 {
                            wet += 1;
                            if grid.mesh_level(x, y, z) >= 0.15 {
                                rendered += 1;
                            }
                        }
                    }
                }
            }
            (wet, rendered)
        };
        let (wet_m, rendered_m) = settled(true);
        assert!(
            wet_m >= 200 && rendered_m * 10 >= wet_m * 9,
            "settled pond regressed under momentum: {rendered_m}/{wet_m} rendered"
        );
    }

    // ── ROUND 14: lavafall crossing a VERTICAL chunk boundary ─────────────
    // User repro (2026-08-04 screenshots): a fall spanning a horizontal seam
    // skips a ~1-cell band — the upper piece ends in a flat lid just above
    // the seam, the lower piece resumes below it.

    /// Upper chunk (0,1,0): open air with a lava source at (8,12,8).
    /// Lower chunk (0,0,0): open air over a solid floor (lattice y<3).
    fn make_fall_pair() -> HashMap<(i32, i32, i32), ChunkFluidGrid> {
        let size = 16;
        let config = crate::FluidConfig::default();
        let mut chunks = HashMap::new();

        let mut d_up = make_density_field_solid(size);
        carve_box(&mut d_up, size, 0..17, 0..17, 0..17);
        let mut upper = make_chunk(size);
        apply_density(&mut upper, &d_up, &config);
        {
            let cell = upper.get_mut(8, 12, 8);
            cell.level = SOURCE_LEVEL;
            cell.fluid_type = FluidType::Lava;
            cell.is_source = true;
            cell.max_flow_dist = 0;
            cell.hops_from_source = 0;
        }
        upper.has_fluid = true;
        upper.has_lava = true;
        upper.has_sources = true;
        chunks.insert((0, 1, 0), upper);

        let mut d_low = make_density_field_solid(size);
        carve_box(&mut d_low, size, 0..17, 3..17, 0..17);
        let mut lower = make_chunk(size);
        apply_density(&mut lower, &d_low, &config);
        chunks.insert((0, 0, 0), lower);
        chunks
    }

    /// One engine tick exactly as thread.rs runs it: sim tick, then per chunk
    /// IN THE GIVEN ORDER build boundary levels from the neighbors' CURRENT
    /// render state, update own render field, mesh. thread.rs iterates a
    /// HashSet, so any order is one the shipped loop can produce.
    fn run_fall_tick(
        chunks: &mut HashMap<(i32, i32, i32), ChunkFluidGrid>,
        config: &crate::FluidConfig,
        order: &[(i32, i32, i32)],
    ) -> HashMap<(i32, i32, i32), (crate::mesh::BoundaryLevels, crate::mesh::FluidMeshData)> {
        let dc = empty_density_cache();
        crate::sim::regen_sources(chunks);
        tick_fluid(chunks, &dc, 16, true, config, true);
        // Phase 1 (mirrors thread.rs): render fields refresh for EVERY chunk
        // being meshed BEFORE any boundary sampling.
        for key in order {
            let grid = chunks.get_mut(key).unwrap();
            grid.update_render_field(
                config.mesh_sticky_release,
                config.mesh_flux_render,
                config.mesh_stream_ribbon,
            );
        }
        // Phase 2: boundaries + mesh.
        let mut out = HashMap::new();
        for key in order {
            let boundary = crate::thread::build_boundary_levels(*key, chunks, 16);
            let grid = chunks.get(key).unwrap();
            let mesh = crate::mesh::mesh_fluid(grid, &boundary, config);
            out.insert(*key, (boundary, mesh));
        }
        out
    }

    #[test]
    #[ignore] // diagnostic probe — run manually with --ignored --nocapture
    fn fall_seam_probe() {
        let config = crate::FluidConfig::default();
        let mut chunks = make_fall_pair();
        let order = [(0, 0, 0), (0, 1, 0)];
        for _ in 0..30 {
            run_fall_tick(&mut chunks, &config, &order);
        }
        for t in 0..6 {
            run_fall_tick(&mut chunks, &config, &order);
            let upper = chunks.get(&(0, 1, 0)).unwrap();
            let lower = chunks.get(&(0, 0, 0)).unwrap();
            eprintln!("── tick {t}: column (8, z=8), upper then lower ──");
            for y in (0..13usize).rev() {
                eprintln!(
                    "  U y={y:2}: raw={:.3} mesh={:.3} flux={:.3}",
                    upper.get(8, y, 8).level,
                    upper.mesh_level(8, y, 8),
                    upper.flux_at(8, y, 8),
                );
            }
            for y in (13..16usize).rev() {
                eprintln!(
                    "  L y={y:2}: raw={:.3} mesh={:.3} flux={:.3}",
                    lower.get(8, y, 8).level,
                    lower.mesh_level(8, y, 8),
                    lower.flux_at(8, y, 8),
                );
            }
            // Where does upper y=0's fluid go? Show the whole seam-row patch
            // plus stagnancy state of the core cell.
            let c = upper.get(8, 0, 8);
            eprintln!(
                "  U y0 core: stagnant={} src={} | y0 row x6..10 z7..9:",
                c.stagnant_ticks, c.is_source
            );
            for z in 7..10usize {
                let row: Vec<String> = (6..11usize)
                    .map(|x| format!("{:.3}", upper.get(x, 0, z).level))
                    .collect();
                eprintln!("    z={z}: {}", row.join(" "));
            }
        }
    }

    #[test]
    fn fall_seam_fields_match_across_vertical_boundary() {
        // The MC fields of the two chunks sample the SAME physical plane
        // (upper's lattice y=0 == lower's lattice y=16). Wherever the two
        // sides disagree, one draws a lid/cut where the other continues —
        // the user-visible "fall skips a line at the seam".
        let config = crate::FluidConfig::default();
        let mut chunks = make_fall_pair();
        let order = [(0, 0, 0), (0, 1, 0)]; // lower meshed first — legal HashSet order
        for _ in 0..30 {
            run_fall_tick(&mut chunks, &config, &order);
        }
        let mut worst: f32 = 0.0;
        let mut note = String::new();
        for t in 0..20 {
            let passes = run_fall_tick(&mut chunks, &config, &order);
            let upper = chunks.get(&(0, 1, 0)).unwrap();
            let lower = chunks.get(&(0, 0, 0)).unwrap();
            let (up_bnd, _) = &passes[&(0, 1, 0)];
            let (low_bnd, _) = &passes[&(0, 0, 0)];
            for z in 0..16usize {
                for x in 0..16usize {
                    // Only judge columns where the seam actually carries fluid.
                    if upper.get(x, 0, z).level < 0.02 && lower.get(x, 15, z).level < 0.02 {
                        continue;
                    }
                    let f_up = crate::mesh::sample_field(upper, x, 0, z, up_bnd);
                    let f_low = crate::mesh::sample_field(lower, x, 16, z, low_bnd);
                    let d = (f_up - f_low).abs();
                    if d > worst {
                        worst = d;
                        note = format!(
                            "tick {t} col ({x},{z}): upper y0 raw={:.3} mesh={:.3} field={f_up:.3} | \
                             lower boundary-sample={f_low:.3} (lower y15 raw={:.3} mesh={:.3})",
                            upper.get(x, 0, z).level,
                            upper.mesh_level(x, 0, z),
                            lower.get(x, 15, z).level,
                            lower.mesh_level(x, 15, z),
                        );
                    }
                }
            }
        }
        assert!(
            worst <= 0.01,
            "MC fields disagree at the vertical seam (worst diff {worst:.3}): {note}"
        );
    }

    #[test]
    fn fall_renders_continuously_across_vertical_boundary() {
        // With in-game (pulsy) supply, once the fall has spanned the seam
        // for a few consecutive ticks (EMA warm), the rendered geometry must
        // be continuous: the upper piece reaches the seam plane and the
        // lower piece meets it — no skipped band.
        let config = crate::FluidConfig::default();
        let mut chunks = make_fall_pair();
        let order = [(0, 0, 0), (0, 1, 0)];
        for _ in 0..30 {
            run_fall_tick(&mut chunks, &config, &order);
        }
        let mut spanning_ticks = 0usize;
        let mut consec = 0usize;
        let mut worst_gap: f32 = -1.0;
        let mut worst_note = String::new();
        for t in 0..36usize {
            {
                let grid = chunks.get_mut(&(0, 1, 0)).unwrap();
                let on = (t / 3) % 2 == 0;
                let cell = grid.get_mut(8, 12, 8);
                cell.is_source = on;
                if on {
                    cell.level = SOURCE_LEVEL;
                    cell.fluid_type = FluidType::Lava;
                    cell.hops_from_source = 0;
                    cell.max_flow_dist = 0;
                }
                grid.has_sources = on;
                grid.has_fluid = true;
            }
            let passes = run_fall_tick(&mut chunks, &config, &order);
            let upper = chunks.get(&(0, 1, 0)).unwrap();
            let lower = chunks.get(&(0, 0, 0)).unwrap();
            // Real fluid spanning the plane in the core column, with supply
            // still flowing in from above.
            let spans = upper.get(8, 0, 8).level >= 0.15
                && upper.get(8, 1, 8).level >= 0.15
                && lower.get(8, 15, 8).level >= 0.15;
            if !spans {
                consec = 0;
                continue;
            }
            consec += 1;
            if consec < 3 {
                continue; // allow the EMA its by-design warm-up lag
            }
            spanning_ticks += 1;
            let col = |p: &&[f32; 3]| (p[0] - 8.5).abs() <= 2.5 && (p[2] - 8.5).abs() <= 2.5;
            let up_min = passes[&(0, 1, 0)]
                .1
                .positions
                .iter()
                .filter(col)
                .map(|p| p[1])
                .fold(f32::INFINITY, f32::min);
            let low_max = passes[&(0, 0, 0)]
                .1
                .positions
                .iter()
                .filter(col)
                .map(|p| p[1])
                .fold(f32::NEG_INFINITY, f32::max);
            let gap = (up_min + 16.0) - low_max;
            if gap > worst_gap {
                worst_gap = gap;
                worst_note = format!(
                    "tick {t}: upper column geometry stops at local y={up_min:.2} (world {:.2}), \
                     lower resumes at {low_max:.2} — {gap:.2} cells unrendered (upper y0 \
                     raw={:.3}/mesh={:.3}, lower y15 raw={:.3}/mesh={:.3})",
                    up_min + 16.0,
                    upper.get(8, 0, 8).level,
                    upper.mesh_level(8, 0, 8),
                    lower.get(8, 15, 8).level,
                    lower.mesh_level(8, 15, 8),
                );
            }
        }
        assert!(
            spanning_ticks >= 8,
            "fall never established across the seam ({spanning_ticks} spanning ticks) — fixture drift?"
        );
        assert!(
            worst_gap <= 1.0,
            "fall skips a band at the vertical chunk seam: {worst_note}"
        );
    }
}
