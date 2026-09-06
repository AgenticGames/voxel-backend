//! Collapse displacement (2026-09-06, "option 1").
//!
//! When rock lands in fluid, `squeeze_excess_fluid` shrinks the affected
//! cells to their new capacity and pushes the excess into the six
//! neighbours. Whatever did not fit used to evaporate - a slab landing in a
//! pool simply ATE the water. This module conserves it instead: the lost
//! volume is queued as a `Displacement` centred on the squeezed cells, and
//! each sim tick re-injects a slice of it into the pool at an expanding
//! Chebyshev ring around the centre, at the fluid surface. The normal
//! spreading rules then flatten the raised ring outward, which reads as a
//! swell radiating from the impact. Ring 1 first, advancing one ring every
//! `DISPLACE_RING_TICKS`, so the injection front itself travels outward.
//!
//! Only columns that already hold fluid within a few cells of the centre's
//! height accept displaced volume (the ring spreads THROUGH the pool, never
//! onto dry floor or into walls). A displacement that cannot place anything
//! for `DISPLACE_MAX_RING` rings or `DISPLACE_MAX_AGE` ticks is dropped -
//! the pool was too small to hold what fell into it, which is the correct
//! physical answer (it overflowed).
//!
//! State lives in a module-level mutex rather than in the sim loop's locals
//! so `handle_event` (many parameters, several call sites) did not need a
//! signature change; the fluid thread is the only writer.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use crate::cell::{ChunkFluidGrid, FluidType, MIN_LEVEL};

use super::utils::SqueezedRemainder;

/// Ticks the injection front dwells on one ring before advancing.
pub const DISPLACE_RING_TICKS: u32 = 2;
/// Outer limit of the ring walk (cells).
pub const DISPLACE_MAX_RING: i32 = 12;
/// Hard lifetime of one displacement (ticks; 90 = 3 s at 30 Hz).
pub const DISPLACE_MAX_AGE: u32 = 90;
/// Fraction of the remaining volume offered to the current ring per tick.
pub const DISPLACE_TICK_FRACTION: f32 = 0.5;
/// How far above / below the displacement centre a ring column is searched
/// for the fluid surface.
pub const DISPLACE_SEARCH_UP: i32 = 2;
pub const DISPLACE_SEARCH_DOWN: i32 = 4;
/// Two displacements of the same fluid closer than this merge (one collapse
/// hits a pool across several TerrainModified chunk events).
pub const DISPLACE_MERGE_DIST: i32 = 4;

#[derive(Debug, Clone)]
pub struct Displacement {
    pub fluid_type: FluidType,
    pub remaining: f32,
    /// World cell of the volume-weighted centroid of the squeezed cells.
    pub center: (i32, i32, i32),
    pub ring: i32,
    pub ticks_in_ring: u32,
    pub age: u32,
    /// Volume actually re-injected so far (diagnostics).
    pub placed: f32,
}

static PENDING: Mutex<Vec<Displacement>> = Mutex::new(Vec::new());

/// Diagnostics counters (volume dropped because no ring could take it).
static DROPPED_TOTAL: Mutex<f32> = Mutex::new(0.0);

/// Queue the remainder a squeeze could not place. `chunk` + `cs` convert
/// the remainders' chunk-local cells to world cells.
pub fn queue_displacement(chunk: (i32, i32, i32), cs: usize, lost: &[SqueezedRemainder]) {
    if lost.is_empty() {
        return;
    }
    let cs_i = cs as i32;
    // Group by fluid type (a quench boundary could squeeze both at once).
    let mut by_type: HashMap<u8, (f32, f64, f64, f64)> = HashMap::new();
    for r in lost {
        if r.lost < MIN_LEVEL {
            continue;
        }
        let e = by_type.entry(r.fluid_type as u8).or_insert((0.0, 0.0, 0.0, 0.0));
        let wx = chunk.0 * cs_i + r.lx as i32;
        let wy = chunk.1 * cs_i + r.ly as i32;
        let wz = chunk.2 * cs_i + r.lz as i32;
        e.0 += r.lost;
        e.1 += wx as f64 * r.lost as f64;
        e.2 += wy as f64 * r.lost as f64;
        e.3 += wz as f64 * r.lost as f64;
    }
    let mut pending = PENDING.lock().unwrap();
    for (ft, (total, sx, sy, sz)) in by_type {
        if total < MIN_LEVEL {
            continue;
        }
        let center = (
            (sx / total as f64).round() as i32,
            (sy / total as f64).round() as i32,
            (sz / total as f64).round() as i32,
        );
        let fluid_type = FluidType::from_u8(ft);
        // Merge with a nearby pending displacement of the same fluid.
        if let Some(d) = pending.iter_mut().find(|d| {
            d.fluid_type as u8 == ft
                && (d.center.0 - center.0).abs() <= DISPLACE_MERGE_DIST
                && (d.center.1 - center.1).abs() <= DISPLACE_MERGE_DIST
                && (d.center.2 - center.2).abs() <= DISPLACE_MERGE_DIST
        }) {
            let w_old = d.remaining.max(0.0);
            let w_new = total;
            let w = (w_old + w_new).max(MIN_LEVEL);
            d.center = (
                ((d.center.0 as f32 * w_old + center.0 as f32 * w_new) / w).round() as i32,
                ((d.center.1 as f32 * w_old + center.1 as f32 * w_new) / w).round() as i32,
                ((d.center.2 as f32 * w_old + center.2 as f32 * w_new) / w).round() as i32,
            );
            d.remaining += total;
            continue;
        }
        pending.push(Displacement {
            fluid_type,
            remaining: total,
            center,
            ring: 1,
            ticks_in_ring: 0,
            age: 0,
            placed: 0.0,
        });
    }
}

/// Number of displacements still spilling (diagnostics / tests).
pub fn pending_count() -> usize {
    PENDING.lock().unwrap().len()
}

/// Volume dropped so far because no ring could take it (diagnostics / tests).
pub fn dropped_total() -> f32 {
    *DROPPED_TOTAL.lock().unwrap()
}

/// Clear all state (tests).
pub fn reset() {
    PENDING.lock().unwrap().clear();
    *DROPPED_TOTAL.lock().unwrap() = 0.0;
}

#[inline]
fn cell_at<'a>(
    chunks: &'a mut HashMap<(i32, i32, i32), ChunkFluidGrid>,
    cs: i32,
    wx: i32, wy: i32, wz: i32,
) -> Option<(&'a mut ChunkFluidGrid, usize, (i32, i32, i32))> {
    let key = (wx.div_euclid(cs), wy.div_euclid(cs), wz.div_euclid(cs));
    let grid = chunks.get_mut(&key)?;
    let idx = grid.index(
        wx.rem_euclid(cs) as usize,
        wy.rem_euclid(cs) as usize,
        wz.rem_euclid(cs) as usize,
    );
    Some((grid, idx, key))
}

/// Find the fluid surface in column (wx, wz) near `cy`: the topmost cell in
/// the search window that holds fluid. Returns that cell's y.
fn surface_y(
    chunks: &mut HashMap<(i32, i32, i32), ChunkFluidGrid>,
    cs: i32,
    wx: i32, cy: i32, wz: i32,
) -> Option<i32> {
    for wy in ((cy - DISPLACE_SEARCH_DOWN)..=(cy + DISPLACE_SEARCH_UP)).rev() {
        if let Some((grid, idx, _)) = cell_at(chunks, cs, wx, wy, wz) {
            if grid.cells[idx].level > MIN_LEVEL && grid.cell_cap[idx] > MIN_LEVEL {
                return Some(wy);
            }
        }
    }
    None
}

/// Add up to `amount` to the surface cell of a column, overflowing into the
/// cell above it if that one has capacity. Returns what was placed.
fn add_to_column(
    chunks: &mut HashMap<(i32, i32, i32), ChunkFluidGrid>,
    cs: i32,
    wx: i32, sy: i32, wz: i32,
    amount: f32,
    fluid_type: FluidType,
    dirty: &mut HashSet<(i32, i32, i32)>,
) -> f32 {
    let mut placed = 0.0f32;
    for wy in [sy, sy + 1] {
        if amount - placed < MIN_LEVEL {
            break;
        }
        if let Some((grid, idx, key)) = cell_at(chunks, cs, wx, wy, wz) {
            let cap = grid.cell_cap[idx];
            if cap <= MIN_LEVEL {
                continue;
            }
            let cur = grid.cells[idx].level;
            // Never mix fluids: an occupied cell of another type is a wall.
            if cur > MIN_LEVEL && grid.cells[idx].fluid_type as u8 != fluid_type as u8 {
                continue;
            }
            let space = (cap - cur).max(0.0);
            if space <= MIN_LEVEL {
                continue;
            }
            let push = (amount - placed).min(space);
            grid.cells[idx].level = cur + push;
            grid.cells[idx].fluid_type = fluid_type;
            grid.dirty = true;
            dirty.insert(key);
            placed += push;
        }
    }
    placed
}

/// One sim tick of spilling. Call once per tick before the flow step so the
/// injected ring is smoothed by the same tick. Returns the chunks touched.
pub fn spill_displacements(
    chunks: &mut HashMap<(i32, i32, i32), ChunkFluidGrid>,
    cs: usize,
) -> HashSet<(i32, i32, i32)> {
    let mut dirty: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut pending = PENDING.lock().unwrap();
    if pending.is_empty() {
        return dirty;
    }
    let cs_i = cs as i32;

    for d in pending.iter_mut() {
        d.age += 1;
        if d.remaining < MIN_LEVEL {
            continue;
        }
        let (cx, cy, cz) = d.center;
        let r = d.ring;

        // Ring cells at Chebyshev distance r, paired with their surface y.
        let mut columns: Vec<(i32, i32, i32)> = Vec::new();
        for dx in -r..=r {
            for dz in -r..=r {
                if dx.abs() != r && dz.abs() != r {
                    continue;
                }
                let wx = cx + dx;
                let wz = cz + dz;
                if let Some(sy) = surface_y(chunks, cs_i, wx, cy, wz) {
                    columns.push((wx, sy, wz));
                }
            }
        }

        if !columns.is_empty() {
            // Halve the remainder each tick while it is worth a wave; once
            // under half a cell, place the rest at once so the displacement
            // finishes instead of decaying towards MIN_LEVEL forever.
            let budget = if d.remaining < 0.5 { d.remaining } else { d.remaining * DISPLACE_TICK_FRACTION };
            let share = budget / columns.len() as f32;
            let mut placed_now = 0.0f32;
            for &(wx, sy, wz) in &columns {
                if d.remaining - placed_now < MIN_LEVEL {
                    break;
                }
                let want = share.min(d.remaining - placed_now);
                placed_now += add_to_column(chunks, cs_i, wx, sy, wz, want, d.fluid_type, &mut dirty);
            }
            d.remaining -= placed_now;
            d.placed += placed_now;
        }

        d.ticks_in_ring += 1;
        if columns.is_empty() || d.ticks_in_ring >= DISPLACE_RING_TICKS {
            d.ring += 1;
            d.ticks_in_ring = 0;
        }
    }

    // Retire finished / hopeless displacements.
    let mut dropped = 0.0f32;
    pending.retain(|d| {
        if d.remaining < MIN_LEVEL {
            return false;
        }
        if d.ring > DISPLACE_MAX_RING || d.age > DISPLACE_MAX_AGE {
            dropped += d.remaining;
            return false;
        }
        true
    });
    if dropped > 0.0 {
        *DROPPED_TOTAL.lock().unwrap() += dropped;
    }
    dirty
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::utils::squeeze_excess_fluid_collect;

    /// Both tests drive the module-level PENDING / DROPPED statics.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn total_level(grid: &ChunkFluidGrid) -> f32 {
        grid.cells.iter().map(|c| c.level).sum()
    }

    /// A slab landing in the middle of a full pool must not lose volume:
    /// after the squeeze + a few spill ticks, the pool holds what it held
    /// (minus nothing), and the ring cells around the landing are fuller
    /// than the pool's far corners.
    #[test]
    fn collapse_into_pool_conserves_volume_and_raises_a_ring() {
        let _serial = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        let size =16usize;
        let cs = size;
        let mut grid = ChunkFluidGrid::new(size);
        // Solid floor at y=3, air above; pool 12x12 at y=4 and y=5, 90% full.
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let d = if y <= 3 { 1.0 } else { -1.0 };
                    grid.set_density(x, y, z, d);
                }
            }
        }
        for z in 2..14 {
            for x in 2..14 {
                for y in 4..=5 {
                    let i = grid.index(x, y, z);
                    grid.cells[i].level = 0.9;
                    grid.cells[i].fluid_type = FluidType::WaterPool;
                }
            }
        }
        let before = total_level(&grid);

        // Rock lands: a 3x3 block at y=4..=5 becomes solid.
        for z in 7..10 {
            for x in 7..10 {
                for y in 4..=5 {
                    grid.set_density(x, y, z, 1.0);
                }
            }
        }
        let lost = squeeze_excess_fluid_collect(&mut grid);
        let lost_total: f32 = lost.iter().map(|r| r.lost).sum();
        assert!(lost_total > 0.5, "squeeze should have had leftovers to displace, got {lost_total}");
        let after_squeeze = total_level(&grid);
        assert!(
            (before - after_squeeze - lost_total).abs() < 1e-3,
            "squeeze accounting: before {before} after {after_squeeze} lost {lost_total}"
        );

        queue_displacement((0, 0, 0), cs, &lost);
        assert_eq!(pending_count(), 1);

        let mut chunks: HashMap<(i32, i32, i32), ChunkFluidGrid> = HashMap::new();
        chunks.insert((0, 0, 0), grid);
        for _ in 0..12 {
            spill_displacements(&mut chunks, cs);
        }
        let grid = chunks.get(&(0, 0, 0)).unwrap();
        let after = total_level(grid);
        // Sub-MIN_LEVEL crumbs are discarded per squeezed cell by design;
        // anything beyond 0.02% of the pool would be a real leak.
        assert!(
            (after - before).abs() < before * 2e-4,
            "volume not conserved: before {before} after {after} dropped {}",
            dropped_total()
        );
        assert_eq!(pending_count(), 0, "displacement should have fully spilled");

        // Ring 1 around the block (x 6 / 10, z 6 / 10) at the surface is
        // fuller than a far corner of the pool.
        let ring_i = grid.index(6, 5, 8);
        let corner_i = grid.index(2, 5, 2);
        assert!(
            grid.cells[ring_i].level > grid.cells[corner_i].level + 0.01,
            "ring {} should exceed corner {}",
            grid.cells[ring_i].level, grid.cells[corner_i].level
        );
        reset();
    }

    /// A pool too small for the displaced volume overflows: the remainder is
    /// dropped and counted, never left pending forever.
    #[test]
    fn hopeless_displacement_is_dropped_and_counted() {
        let _serial = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        let size =8usize;
        let mut grid = ChunkFluidGrid::new(size);
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    grid.set_density(x, y, z, if y <= 2 { 1.0 } else { -1.0 });
                }
            }
        }
        // One full cell of water, everything around it solid → nowhere to go.
        for z in 0..size {
            for x in 0..size {
                for y in 3..size {
                    if !(x == 4 && z == 4 && y == 3) {
                        grid.set_density(x, y, z, 1.0);
                    }
                }
            }
        }
        let i = grid.index(4, 3, 4);
        grid.cells[i].level = 1.0;
        grid.cells[i].fluid_type = FluidType::WaterPool;
        // Now the cell itself turns solid.
        grid.set_density(4, 3, 4, 1.0);
        let lost = squeeze_excess_fluid_collect(&mut grid);
        assert!((lost.iter().map(|r| r.lost).sum::<f32>() - 1.0).abs() < 1e-3);
        queue_displacement((0, 0, 0), size, &lost);
        let mut chunks: HashMap<(i32, i32, i32), ChunkFluidGrid> = HashMap::new();
        chunks.insert((0, 0, 0), grid);
        for _ in 0..(DISPLACE_MAX_AGE + 5) {
            spill_displacements(&mut chunks, size);
        }
        assert_eq!(pending_count(), 0);
        assert!((dropped_total() - 1.0).abs() < 1e-3, "dropped {}", dropped_total());
        reset();
    }
}
