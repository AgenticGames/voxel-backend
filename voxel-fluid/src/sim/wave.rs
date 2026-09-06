//! Collapse impact waves (2026-09-06, "option 2").
//!
//! The fluid automaton moves volume by LEVEL DIFFERENCE only (transfer ∝
//! diff), and `equalize_horizontal` flattens every connected pool layer
//! each tick. Neither has inertia, so a slab landing in a pool can only
//! produce a bulge that diffuses away. Waves need momentum.
//!
//! This module runs a small shallow-water step (the "virtual pipe" model:
//! per-face flux with inertia, gravity from surface-height differences,
//! damping, outflow limiting) over the water surface in a bounded region
//! around each impact, for `WAVE_LIFE_TICKS`, and then hands the region
//! back to the automaton. While a region is alive its columns are masked
//! out of `equalize_horizontal` (see `masked_columns`), otherwise the
//! equalizer would erase the crest every tick. The automaton's own spread
//! still runs inside the region and acts as extra damping.
//!
//! The height field is read from the automaton's cells each tick (top water
//! cell of each column, `y + level/cap`) and written back as level changes
//! to that top cell, overflowing into the cell above or drawing from the
//! cell below, so mass is conserved and the mesh sees ordinary level data.
//! Faces respect the rendered-surface barrier (`face_blocked`, bug #215):
//! a thin wall is a wall to waves too. Columns without water are walls,
//! which is what gives reflection off the pool edge.
//!
//! An impact seeds the region with an outward radial flux proportional to
//! the displaced volume; the displacement ring injection
//! (`sim::displacement`) supplies the mass bump at the same time. Water
//! only - lava is viscous and runs on its own divided tick.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use crate::cell::{face_blocked, ChunkFluidGrid, MIN_LEVEL};

// ── Tunables ──────────────────────────────────────────────────────────
/// Wave regions step and remesh at THIS rate, independent of the pool
/// automaton's `tick_rate` (live config runs the pool at 3 Hz, which would
/// make a crest crawl at ~1 cell/s and update its mesh three times a
/// second). The region is a few hundred columns, so stepping it at 30 Hz
/// is cheap; the pool itself stays on its own cadence.
pub const WAVE_TICK_HZ: f32 = 30.0;
/// Half-width of the wave region in columns (region is (2R+1)²).
pub const WAVE_REGION_RADIUS: i32 = 24;
/// Ticks a region stays active after its last impulse (150 = 5 s @ 30 Hz).
pub const WAVE_LIFE_TICKS: u32 = 300;
/// Gravity term in cell units per tick²: flux gains G·Δh each tick.
/// Wave speed ≈ sqrt(G·depth) cells/tick → ~0.35-0.5 cells/tick for a
/// 1-2 cell deep pool (10-15 cells/s), which reads as a brisk ripple.
pub const WAVE_G: f32 = 0.05;
/// Per-tick flux retention (inertia damping). 0.985 ≈ energy halves in ~1.5 s.
pub const WAVE_DAMP: f32 = 0.994;
/// A column may not lose more than this fraction of its available depth
/// in one tick (keeps the scheme stable and levels non-negative).
pub const WAVE_MAX_DRAIN: f32 = 0.60;
/// Radius of the initial outward flux disc.
pub const WAVE_IMPULSE_RADIUS: i32 = 5;
/// Initial face flux at the disc edge per unit of displaced volume, and its clamp.
pub const WAVE_IMPULSE_PER_VOLUME: f32 = 0.03;
pub const WAVE_IMPULSE_MAX: f32 = 0.35;
/// Cap on the surface-height change of one column in one tick (cells).
pub const WAVE_MAX_DH: f32 = 0.50;
/// How far above / below the region centre a column is searched for its
/// water surface.
pub const WAVE_SEARCH_UP: i32 = 3;
pub const WAVE_SEARCH_DOWN: i32 = 3;
/// Two impacts closer than this share one region.
pub const WAVE_MERGE_DIST: i32 = 8;
/// Below this crest amplitude (cells) a region is considered calm and
/// released early so equalize can take over.
pub const WAVE_CALM_AMPLITUDE: f32 = 0.004;
/// Grace before the calm check may release a region (ticks).
pub const WAVE_CALM_MIN_AGE: u32 = 30;

#[derive(Debug, Clone)]
pub struct WaveRegion {
    /// World cell of the impact; `.1` is the water-surface layer.
    pub center: (i32, i32, i32),
    pub radius: i32,
    pub age: u32,
    /// Face flux between column c and c+1 (x) / c and c+W (z).
    pub flux_x: Vec<f32>,
    pub flux_z: Vec<f32>,
    /// Diagnostics: last tick's peak |h - mean| over the region.
    pub last_amplitude: f32,
}

impl WaveRegion {
    #[inline]
    fn width(&self) -> usize {
        (self.radius * 2 + 1) as usize
    }
}

static REGIONS: Mutex<Vec<WaveRegion>> = Mutex::new(Vec::new());
/// Columns (wx, wz) currently owned by a wave region - `equalize_horizontal`
/// leaves these alone. Rebuilt every step.
static MASK: Mutex<Vec<(i32, i32)>> = Mutex::new(Vec::new());

/// Columns the equalizer must skip this tick (None when no region is alive,
/// so the common path costs one mutex probe).
pub fn masked_columns() -> Option<HashSet<(i32, i32)>> {
    let m = MASK.lock().unwrap();
    if m.is_empty() { None } else { Some(m.iter().copied().collect()) }
}

pub fn region_count() -> usize {
    REGIONS.lock().unwrap().len()
}

pub fn reset() {
    REGIONS.lock().unwrap().clear();
    MASK.lock().unwrap().clear();
}

/// Seed (or re-energise) a wave region for an impact at `center` that
/// displaced `volume` cells of water.
pub fn spawn_impact(center: (i32, i32, i32), volume: f32) {
    if volume < MIN_LEVEL {
        return;
    }
    let f0 = (volume * WAVE_IMPULSE_PER_VOLUME).clamp(0.01, WAVE_IMPULSE_MAX);
    let mut regions = REGIONS.lock().unwrap();
    let existing = regions.iter_mut().find(|r| {
        (r.center.0 - center.0).abs() <= WAVE_MERGE_DIST
            && (r.center.1 - center.1).abs() <= 2
            && (r.center.2 - center.2).abs() <= WAVE_MERGE_DIST
    });
    let region: &mut WaveRegion = match existing {
        Some(r) => {
            r.age = 0;
            r
        }
        None => {
            let radius = WAVE_REGION_RADIUS;
            let w = (radius * 2 + 1) as usize;
            regions.push(WaveRegion {
                center,
                radius,
                age: 0,
                flux_x: vec![0.0; w * w],
                flux_z: vec![0.0; w * w],
                last_amplitude: 0.0,
            });
            regions.last_mut().unwrap()
        }
    };
    // Outward radial flux on faces within the impulse disc, strongest at
    // the disc edge (the crest starts there), zero at the very centre.
    let w = region.width();
    let r = region.radius;
    let ox = center.0 - region.center.0; // impact offset inside the region
    let oz = center.2 - region.center.2;
    for iz in 0..w {
        for ix in 0..w {
            let c = iz * w + ix;
            // face +x sits at (ix + 0.5, iz); face +z at (ix, iz + 0.5)
            let fx_dx = (ix as i32 - r - ox) as f32 + 0.5;
            let fx_dz = (iz as i32 - r - oz) as f32;
            let d = (fx_dx * fx_dx + fx_dz * fx_dz).sqrt();
            if d > 0.1 && d <= WAVE_IMPULSE_RADIUS as f32 + 0.5 {
                let fall = (d / WAVE_IMPULSE_RADIUS as f32).min(1.0);
                region.flux_x[c] += f0 * fall * (fx_dx / d);
            }
            let fz_dx = (ix as i32 - r - ox) as f32;
            let fz_dz = (iz as i32 - r - oz) as f32 + 0.5;
            let d = (fz_dx * fz_dx + fz_dz * fz_dz).sqrt();
            if d > 0.1 && d <= WAVE_IMPULSE_RADIUS as f32 + 0.5 {
                let fall = (d / WAVE_IMPULSE_RADIUS as f32).min(1.0);
                region.flux_z[c] += f0 * fall * (fz_dz / d);
            }
        }
    }
}

#[inline]
fn cell_key(cs: i32, wx: i32, wy: i32, wz: i32) -> ((i32, i32, i32), usize, usize, usize) {
    (
        (wx.div_euclid(cs), wy.div_euclid(cs), wz.div_euclid(cs)),
        wx.rem_euclid(cs) as usize,
        wy.rem_euclid(cs) as usize,
        wz.rem_euclid(cs) as usize,
    )
}

/// One column of the wave grid for this tick.
#[derive(Clone, Copy)]
struct Col {
    has: bool,
    key: (i32, i32, i32),
    idx: usize,
    top_y: i32,
    cap: f32,
    level: f32,
    /// Surface height in cells: top_y + level / cap.
    h: f32,
    /// Available depth this column can give up (its own fraction + one
    /// cell if the cell below also holds water).
    avail: f32,
    open_px: bool,
    open_pz: bool,
}

impl Col {
    const NONE: Col = Col {
        has: false, key: (0, 0, 0), idx: 0, top_y: 0, cap: 0.0, level: 0.0, h: 0.0,
        avail: 0.0, open_px: false, open_pz: false,
    };
}

/// Locate the water surface in column (wx, wz) near `cy`.
fn sample_column(
    chunks: &HashMap<(i32, i32, i32), ChunkFluidGrid>,
    cs: i32,
    wx: i32, cy: i32, wz: i32,
) -> Col {
    for wy in ((cy - WAVE_SEARCH_DOWN)..=(cy + WAVE_SEARCH_UP)).rev() {
        let (key, lx, ly, lz) = cell_key(cs, wx, wy, wz);
        let Some(grid) = chunks.get(&key) else { continue };
        let idx = grid.index(lx, ly, lz);
        let cell = &grid.cells[idx];
        let cap = grid.cell_cap[idx];
        if cell.level <= MIN_LEVEL || cap <= MIN_LEVEL || !cell.fluid_type.is_water() || cell.is_source() {
            continue;
        }
        // Top water cell found. Depth below: one more cell if it is water too.
        let below_has = {
            let (bk, bx, by, bz) = cell_key(cs, wx, wy - 1, wz);
            chunks.get(&bk).map(|g| {
                let bi = g.index(bx, by, bz);
                g.cells[bi].level > MIN_LEVEL && g.cell_cap[bi] > MIN_LEVEL
            }).unwrap_or(false)
        };
        let frac = (cell.level / cap).min(1.0);
        return Col {
            has: true,
            key,
            idx,
            top_y: wy,
            cap,
            level: cell.level,
            h: wy as f32 + frac,
            avail: frac + if below_has { 1.0 } else { 0.0 },
            open_px: !face_blocked(&grid.cell_corners, idx, 1, 0, 0),
            open_pz: !face_blocked(&grid.cell_corners, idx, 0, 0, 1),
        };
    }
    // No water in the window. A DRY column that still has an open cell
    // sitting on solid within the window is a floor the wave can flood
    // (inflow only: its surface is its floor, avail = 0). Without this an
    // impact that drains a column completely left a permanent hole - a
    // dry column used to be a wall, and nothing could flow back into it.
    let mut solid_y: Option<i32> = None;
    for wy in ((cy - WAVE_SEARCH_DOWN)..=(cy + WAVE_SEARCH_UP)).rev() {
        let (key, lx, ly, lz) = cell_key(cs, wx, wy, wz);
        let Some(grid) = chunks.get(&key) else { continue };
        let idx = grid.index(lx, ly, lz);
        if grid.cell_cap[idx] <= MIN_LEVEL {
            solid_y = Some(wy);
            break;
        }
    }
    if let Some(ys) = solid_y {
        let wy = ys + 1;
        if wy <= cy + WAVE_SEARCH_UP {
            let (key, lx, ly, lz) = cell_key(cs, wx, wy, wz);
            if let Some(grid) = chunks.get(&key) {
                let idx = grid.index(lx, ly, lz);
                let cap = grid.cell_cap[idx];
                let cell = &grid.cells[idx];
                if cap > MIN_LEVEL && (cell.level <= MIN_LEVEL || cell.fluid_type.is_water()) && !cell.is_source() {
                    return Col {
                        has: true,
                        key,
                        idx,
                        top_y: wy,
                        cap,
                        level: 0.0,
                        h: wy as f32,
                        avail: 0.0,
                        open_px: !face_blocked(&grid.cell_corners, idx, 1, 0, 0),
                        open_pz: !face_blocked(&grid.cell_corners, idx, 0, 0, 1),
                    };
                }
            }
        }
    }
    Col::NONE
}

/// Apply a surface-height change to a column: adjust the top cell, overflow
/// into the cell above, draw from the cell below.
fn apply_dh(
    chunks: &mut HashMap<(i32, i32, i32), ChunkFluidGrid>,
    cs: i32,
    wx: i32, wz: i32,
    col: &Col,
    dh: f32,
    dirty: &mut HashSet<(i32, i32, i32)>,
) {
    if dh.abs() < 1e-5 {
        return;
    }
    let mut delta = dh * col.cap; // in level units of the top cell
    let mut ft = crate::cell::FluidType::WaterPool;
    // Top cell.
    if let Some(grid) = chunks.get_mut(&col.key) {
        let cell = &mut grid.cells[col.idx];
        if cell.level <= MIN_LEVEL && !cell.fluid_type.is_water() {
            cell.fluid_type = crate::cell::FluidType::WaterPool; // flooding a dry floor cell
        }
        ft = cell.fluid_type;
        let new_level = cell.level + delta;
        let clamped = new_level.clamp(0.0, col.cap);
        delta = new_level - clamped; // leftover: >0 overflow, <0 deficit
        cell.level = clamped;
        grid.dirty = true;
        dirty.insert(col.key);
    }
    if delta > MIN_LEVEL {
        // Overflow into the cell above if it can hold water.
        let (k, x, y, z) = cell_key(cs, wx, col.top_y + 1, wz);
        if let Some(grid) = chunks.get_mut(&k) {
            let i = grid.index(x, y, z);
            let cap = grid.cell_cap[i];
            if cap > MIN_LEVEL {
                let cell = &mut grid.cells[i];
                if cell.level <= MIN_LEVEL || cell.fluid_type.is_water() {
                    if cell.level <= MIN_LEVEL { cell.fluid_type = ft; }
                    cell.level = (cell.level + delta).min(cap);
                    grid.dirty = true;
                    dirty.insert(k);
                }
            }
        }
        // Whatever still does not fit stays with the top cell (already clamped): lost.
    } else if delta < -MIN_LEVEL {
        // Draw the deficit from the cell below.
        let (k, x, y, z) = cell_key(cs, wx, col.top_y - 1, wz);
        if let Some(grid) = chunks.get_mut(&k) {
            let i = grid.index(x, y, z);
            let cell = &mut grid.cells[i];
            if cell.level > MIN_LEVEL && cell.fluid_type.is_water() {
                cell.level = (cell.level + delta).max(0.0);
                grid.dirty = true;
                dirty.insert(k);
            }
        }
    }
}

/// One sim tick for every live region. Call after displacement spill and
/// before `equalize_horizontal`. Returns the chunks touched.
pub fn step_waves(
    chunks: &mut HashMap<(i32, i32, i32), ChunkFluidGrid>,
    cs: usize,
) -> HashSet<(i32, i32, i32)> {
    let mut dirty: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut regions = REGIONS.lock().unwrap();
    let mut mask = MASK.lock().unwrap();
    mask.clear();
    if regions.is_empty() {
        return dirty;
    }
    let cs_i = cs as i32;

    for region in regions.iter_mut() {
        region.age += 1;
        let w = region.width();
        let r = region.radius;
        let (cx, cy, cz) = region.center;

        // 1. Sample the surface.
        let mut cols: Vec<Col> = vec![Col::NONE; w * w];
        let mut sum_h = 0.0f32;
        let mut n_h = 0usize;
        for iz in 0..w {
            for ix in 0..w {
                let wx = cx - r + ix as i32;
                let wz = cz - r + iz as i32;
                let c = sample_column(chunks, cs_i, wx, cy, wz);
                if c.has {
                    mask.push((wx, wz));
                    sum_h += c.h;
                    n_h += 1;
                }
                cols[iz * w + ix] = c;
            }
        }
        if n_h == 0 {
            region.age = WAVE_LIFE_TICKS + 1; // nothing to wave on
            continue;
        }
        let mean_h = sum_h / n_h as f32;

        // 2. Flux update with inertia. A face is open when both columns hold
        //    water at (nearly) the same layer and the rendered surface does
        //    not block it.
        // A crest taller than a cell stacks into the layer above, and a trough
        // can empty its top cell, so two neighbouring surfaces of one pool can
        // sit two layers apart. Anything further is a real step (a fall).
        let face_open = |a: &Col, b: &Col, open_bit: bool| -> bool {
            a.has && b.has && open_bit && (a.top_y - b.top_y).abs() <= 2
        };
        for iz in 0..w {
            for ix in 0..w {
                let c = iz * w + ix;
                if ix + 1 < w {
                    let a = &cols[c];
                    let b = &cols[c + 1];
                    region.flux_x[c] = if face_open(a, b, a.open_px) {
                        region.flux_x[c] * WAVE_DAMP + WAVE_G * (a.h - b.h)
                    } else { 0.0 };
                } else {
                    region.flux_x[c] = 0.0;
                }
                if iz + 1 < w {
                    let a = &cols[c];
                    let b = &cols[c + w];
                    region.flux_z[c] = if face_open(a, b, a.open_pz) {
                        region.flux_z[c] * WAVE_DAMP + WAVE_G * (a.h - b.h)
                    } else { 0.0 };
                } else {
                    region.flux_z[c] = 0.0;
                }
            }
        }

        // 3. Outflow limiting per column.
        let mut scale: Vec<f32> = vec![1.0; w * w];
        for iz in 0..w {
            for ix in 0..w {
                let c = iz * w + ix;
                if !cols[c].has { continue; }
                let mut out = 0.0f32;
                if ix + 1 < w { out += region.flux_x[c].max(0.0); }
                if ix > 0 { out += (-region.flux_x[c - 1]).max(0.0); }
                if iz + 1 < w { out += region.flux_z[c].max(0.0); }
                if iz > 0 { out += (-region.flux_z[c - w]).max(0.0); }
                let limit = cols[c].avail * WAVE_MAX_DRAIN;
                if out > limit && out > 1e-6 {
                    scale[c] = limit / out;
                }
            }
        }
        for iz in 0..w {
            for ix in 0..w {
                let c = iz * w + ix;
                if ix + 1 < w {
                    let f = region.flux_x[c];
                    let src = if f > 0.0 { c } else { c + 1 };
                    region.flux_x[c] = f * scale[src];
                }
                if iz + 1 < w {
                    let f = region.flux_z[c];
                    let src = if f > 0.0 { c } else { c + w };
                    region.flux_z[c] = f * scale[src];
                }
            }
        }

        // 4. Height change = inflow - outflow; apply.
        let mut amplitude = 0.0f32;
        for iz in 0..w {
            for ix in 0..w {
                let c = iz * w + ix;
                if !cols[c].has { continue; }
                let mut dh = 0.0f32;
                if ix + 1 < w { dh -= region.flux_x[c]; }
                if ix > 0 { dh += region.flux_x[c - 1]; }
                if iz + 1 < w { dh -= region.flux_z[c]; }
                if iz > 0 { dh += region.flux_z[c - w]; }
                let dh = dh.clamp(-WAVE_MAX_DH, WAVE_MAX_DH);
                amplitude = amplitude.max((cols[c].h + dh - mean_h).abs());
                let wx = cx - r + ix as i32;
                let wz = cz - r + iz as i32;
                apply_dh(chunks, cs_i, wx, wz, &cols[c], dh, &mut dirty);
            }
        }
        region.last_amplitude = amplitude;
        if region.age > WAVE_CALM_MIN_AGE && amplitude < WAVE_CALM_AMPLITUDE {
            region.age = WAVE_LIFE_TICKS + 1; // calm: release early
        }
    }

    regions.retain(|r| r.age <= WAVE_LIFE_TICKS);
    if regions.is_empty() {
        mask.clear();
    }
    dirty
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::FluidType;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn pool_grid(size: usize, floor_y: usize, lo: usize, hi: usize, level: f32) -> ChunkFluidGrid {
        let mut g = ChunkFluidGrid::new(size);
        for z in 0..size { for y in 0..size { for x in 0..size {
            g.set_density(x, y, z, if y <= floor_y { 1.0 } else { -1.0 });
        }}}
        for z in lo..hi { for x in lo..hi {
            for y in (floor_y + 1)..=(floor_y + 2) {
                let i = g.index(x, y, z);
                // Physical pool: full lower layer, partial top layer.
                g.cells[i].level = if y == floor_y + 1 { 1.0 } else { level };
                g.cells[i].fluid_type = FluidType::WaterPool;
            }
        }}
        // Rim: everything outside the pool footprint is solid up to three
        // cells above the floor, like a real basin. Without it the pool sits
        // on a flat floor and (correctly) floods outward, which is not what
        // these tests measure.
        for z in 0..size { for x in 0..size {
            if x >= lo && x < hi && z >= lo && z < hi { continue; }
            for y in (floor_y + 1)..=(floor_y + 3) {
                g.set_density(x, y, z, 1.0);
            }
        }}
        g.has_fluid = true;
        g
    }

    fn total(g: &ChunkFluidGrid) -> f32 { g.cells.iter().map(|c| c.level).sum() }

    /// Surface height profile along x through the centre row at the top layer.
    fn profile(g: &ChunkFluidGrid, y: usize, z: usize, lo: usize, hi: usize) -> Vec<f32> {
        (lo..hi).map(|x| { let i = g.index(x, y, z); g.cells[i].level }).collect()
    }

    #[test]
    fn impulse_makes_a_travelling_crest_that_reflects_and_conserves_volume() {
        let _s = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        let size = 30usize;
        let floor = 3usize;
        let (lo, hi) = (2usize, 28usize);
        let mut chunks: HashMap<(i32, i32, i32), ChunkFluidGrid> = HashMap::new();
        chunks.insert((0, 0, 0), pool_grid(size, floor, lo, hi, 0.5));
        let before = total(&chunks[&(0, 0, 0)]);

        spawn_impact((15, (floor + 2) as i32, 15), 12.0);
        assert_eq!(region_count(), 1);

        // Track where the crest (max level on the centre row, right of centre) sits.
        let mut crest_x: Vec<usize> = Vec::new();
        let mut center_series: Vec<f32> = Vec::new();
        for t in 0..140 {
            step_waves(&mut chunks, size);
            let g = &chunks[&(0, 0, 0)];
            // Outer band only (r >= 4) so the centre rebound does not count as the crest.
            let row = profile(g, floor + 2, 15, 19, hi);
            let (mut bx, mut bv) = (0usize, -1.0f32);
            for (k, v) in row.iter().enumerate() { if *v > bv { bv = *v; bx = 19 + k; } }
            if t % 5 == 0 { crest_x.push(bx); }
            if std::env::var("WAVE_DEBUG").is_ok() && t % 4 == 0 {
                let full = profile(g, floor + 2, 15, lo, hi);
                let s: Vec<String> = full.iter().map(|v| format!("{:4.2}", v)).collect();
                println!("t{:3} {}", t, s.join(" "));
            }
            center_series.push(g.cells[g.index(15, floor + 2, 15)].level);
        }
        let g = &chunks[&(0, 0, 0)];
        let after = total(g);
        assert!((after - before).abs() < before * 1e-3, "volume drifted: {before} -> {after}");

        // The crest moves outward over the first samples (monotone non-decreasing
        // x for the first ~25 ticks = 5 samples) and reaches the far side.
        // Sampled every 5 ticks: by t=20 the outer crest must sit at least two
        // columns further out than where the impulse put it.
        assert!(crest_x[4] >= crest_x[0] + 2,
            "crest did not travel outward: {:?}", crest_x);
        assert!(crest_x.iter().any(|&x| x >= 23), "crest never reached the wall: {:?}", crest_x);

        // Reflection: after the initial trough at the centre, the centre level
        // rises again later (a returning crest) before settling.
        let min_t = (0..60).min_by(|&a, &b| center_series[a].partial_cmp(&center_series[b]).unwrap()).unwrap();
        let later_max = center_series[min_t..].iter().cloned().fold(f32::MIN, f32::max);
        assert!(later_max > center_series[min_t] + 0.01,
            "no returning crest at the centre: min {} at t{}, later max {}", center_series[min_t], min_t, later_max);

        // Settling: the region calms and releases within its lifetime.
        for _ in 0..WAVE_LIFE_TICKS { step_waves(&mut chunks, size); }
        assert_eq!(region_count(), 0, "region did not release");
        assert!(masked_columns().is_none());
        reset();
    }

    #[test]
    fn waves_do_not_cross_a_solid_wall() {
        let _s = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        let size = 30usize;
        let floor = 3usize;
        let mut g = pool_grid(size, floor, 2, 28, 0.5);
        // Solid wall at x = 15 across the whole pool, two cells thick.
        for z in 0..size { for y in 0..size { for x in 14..=15 {
            g.set_density(x, y, z, 1.0);
            let i = g.index(x, y, z);
            g.cells[i].level = 0.0;
        }}}
        let initial: Vec<f32> = g.cells.iter().map(|c| c.level).collect();
        let mut chunks: HashMap<(i32, i32, i32), ChunkFluidGrid> = HashMap::new();
        chunks.insert((0, 0, 0), g);
        spawn_impact((8, (floor + 2) as i32, 15), 12.0);
        for _ in 0..120 { step_waves(&mut chunks, size); }
        let g = &chunks[&(0, 0, 0)];
        // Far side (x >= 16) untouched: every level exactly as it started.
        for z in 2..28 { for x in 16..28 { for y in (floor + 1)..=(floor + 2) {
            let i = g.index(x, y, z);
            let l = g.cells[i].level;
            assert!((l - initial[i]).abs() < 1e-6, "leak through wall at ({x},{y},{z}): {l} vs {}", initial[i]);
        }}}
        reset();
    }
}
