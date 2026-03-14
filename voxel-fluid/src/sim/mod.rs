use std::collections::{HashMap, HashSet};

use crate::cell::{ChunkDensityCache, ChunkFluidGrid, MIN_LEVEL};
use crate::FluidConfig;

mod chunk;
mod utils;

pub use utils::{squeeze_excess_fluid, equalize_horizontal, detect_solidification, regen_sources};

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
                    let mut grid = ChunkFluidGrid::from_density_cache(cache);
                    grid.recompute_capacity();
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

    // Apply cross-chunk transfers (second pass — no borrow conflicts)
    for xfer in &all_transfers {
        // If target chunk has no grid but density exists, create grid on demand
        if !chunks.contains_key(&xfer.dest_key) {
            if let Some(cache) = chunk_densities.get(&xfer.dest_key) {
                let mut grid = ChunkFluidGrid::from_density_cache(cache);
                grid.recompute_capacity();
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
        for x in 8..14 { grid.set_density(x, 0, 8, 1.0); }
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
        assert!(loss_pct < 1.0, "Conservation ±1%: initial={:.2}, final={:.2}, loss={:.2}%", initial, final_w, loss_pct);
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
        assert!(loss_pct < 1.0, "Conservation ±1%: initial={:.2}, final={:.2}, loss={:.2}%", initial, final_w, loss_pct);
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
        assert!((final_w - initial).abs() < 0.01, "Conservation: initial={:.2}, final={:.2}", initial, final_w);
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
    fn bowl_binary_boundary_conservation() {
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
            assert!(cap == 0.0 || cap == 1.0, "Binary capacity violated at ({},{},{}): cap={}", x, y, z, cap);
        }}}
        fill_air_to_capacity(&mut grid, 0..size);

        let mut chunks = HashMap::new();
        chunks.insert((0,0,0), grid);
        let initial = total_water(&chunks);
        let dc = empty_density_cache();
        for _ in 0..500 { tick_fluid(&mut chunks, &dc, size, false, &config, true); }
        let final_w = total_water(&chunks);
        let loss_pct = ((initial - final_w) / initial * 100.0).abs();
        assert!(loss_pct < 1.0, "Binary boundary conservation: initial={:.2}, final={:.2}, loss={:.2}%", initial, final_w, loss_pct);
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
        assert!((final_w - initial).abs() < 0.1, "Cross-chunk conservation: initial={:.2}, final={:.2}", initial, final_w);
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
        assert!(loss_pct < 1.0, "Conservation ±1%: initial={:.2}, final={:.2}, loss={:.2}%", initial, final_w, loss_pct);
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
        assert!(loss_pct < 1.0, "Conservation ±1%: initial={:.2}, final={:.2}, loss={:.2}%", initial, final_w, loss_pct);
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
        for gz in 0..stride { for gy in 0..stride { for gx in 0..stride {
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
        assert!(loss_pct < 1.0, "Conservation ±1%: initial={:.2}, final={:.2}, loss={:.2}%", initial, final_w, loss_pct);
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
        assert!((final_w - initial).abs() < 0.2, "Conservation: initial={:.2}, final={:.2}", initial, final_w);

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
        assert!((final_w - initial).abs() < 0.1, "Large volume conservation: initial={:.4}, final={:.4}", initial, final_w);
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
}
