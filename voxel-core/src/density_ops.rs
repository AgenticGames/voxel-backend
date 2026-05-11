//! Shared density-field operations used by both `voxel-ffi`'s building flatten
//! and `voxel-core`'s collapse rubble placement.
//!
//! Everything here works on a raw `&mut HashMap<(i32, i32, i32), DensityField>`
//! so it can be called from any crate that holds chunk densities.
//!
//! Key facilities:
//!   - `cell_locations` — fan-out a world cell to ALL chunks that share it
//!     (1, 2, 4, or 8 chunks for interior/face/edge/corner cells). Required
//!     for seam-aware writes — chunks share their +cs boundary face, and
//!     `sync_boundary_density` later runs `min()` between neighbors which
//!     would pull our writes down without this fan-out.
//!   - `write_force/raise/lower/all_locations` — single-cell density edits
//!     that handle the seam fan-out and track every cell touched.
//!   - `subvoxel_boundary_densities` — given a target Y as a float, return
//!     the integer cell + densities to write so the DC iso-surface lands at
//!     that exact sub-voxel position.
//!   - `natural_floor_y_iso` — find the existing iso-surface position in a
//!     column by linear interpolation. Works for both legacy integer-Y
//!     placements and sub-voxel placements.
//!   - `count_air_face_neighbors` — used to classify cells as thin features.
//!   - `formation_removal_pass` — iterative erosion of stalactite/stalagmite/
//!     column geometry inside an axis-aligned cylinder.

use std::collections::{HashMap, HashSet};

use crate::density::DensityField;
use crate::material::Material;

// ── Types ─────────────────────────────────────────────────────────────────

/// Tracks every cell touched by a write so we can:
///   1. restore intended values after `sync_boundary_density` runs `min()`
///      at chunk seams (uses `new_density`/`new_material`); and
///   2. roll back to the pre-write state for cinematic preview-then-commit
///      flows (uses `orig_density`/`orig_material`).
#[derive(Debug, Clone, Copy)]
pub struct WrittenCell {
    pub key: (i32, i32, i32),
    pub lx: usize,
    pub ly: usize,
    pub lz: usize,
    pub new_density: f32,
    pub new_material: Material,
    pub orig_density: f32,
    pub orig_material: Material,
}

// ── Multi-chunk seam-aware fan-out ────────────────────────────────────────

/// A single world voxel cell can be stored in up to 8 chunk DensityFields
/// because each chunk includes its +cs boundary face (size = chunk_size + 1).
/// World cell (wx, wy, wz) lives in:
///   - Primary chunk (wx/cs, wy/cs, wz/cs) at local (wx%cs, wy%cs, wz%cs).
///   - For each axis where `local==0` (the cell is on the chunk's lower face),
///     ALSO in the previous chunk at local index = cs.
///
/// Returns up to 8 (chunk_key, lx, ly, lz) locations.
pub fn cell_locations(
    cs: i32, wx: i32, wy: i32, wz: i32,
) -> [Option<((i32, i32, i32), usize, usize, usize)>; 8] {
    let cx = wx.div_euclid(cs);
    let cy = wy.div_euclid(cs);
    let cz = wz.div_euclid(cs);
    let lx = wx.rem_euclid(cs) as usize;
    let ly = wy.rem_euclid(cs) as usize;
    let lz = wz.rem_euclid(cs) as usize;
    let mx = lx == 0;
    let my = ly == 0;
    let mz = lz == 0;

    let mut out: [Option<((i32, i32, i32), usize, usize, usize)>; 8] = [None; 8];
    let mut idx = 0usize;
    for fx in [false, true] {
        if fx && !mx { continue; }
        for fy in [false, true] {
            if fy && !my { continue; }
            for fz in [false, true] {
                if fz && !mz { continue; }
                let cx2 = if fx { cx - 1 } else { cx };
                let cy2 = if fy { cy - 1 } else { cy };
                let cz2 = if fz { cz - 1 } else { cz };
                let lx2 = if fx { cs as usize } else { lx };
                let ly2 = if fy { cs as usize } else { ly };
                let lz2 = if fz { cs as usize } else { lz };
                out[idx] = Some(((cx2, cy2, cz2), lx2, ly2, lz2));
                idx += 1;
            }
        }
    }
    out
}

// ── Sub-voxel surface placement (3A) ─────────────────────────────────────

/// For a target iso surface at `target_y_float`, return the integer floor
/// cell index and the densities to write at (floor_cell, floor_cell+1) so
/// the DC iso-surface lands EXACTLY at `target_y_float`.
///
/// Convention: density crosses 0 at the surface. Linear interpolation
/// between adjacent cells means iso position = y + d_lo / (d_lo - d_hi).
/// We solve for the (d_lo, d_hi) pair given the desired iso fraction.
///
///   frac ≤ 0.5: d_solid = frac/(1-frac) ∈ [0, 1], d_air = -1.
///   frac > 0.5: d_solid = 1, d_air = (frac-1)/frac ∈ [-1, 0].
#[inline]
pub fn subvoxel_boundary_densities(target_y_float: f32) -> (i32, f32, f32) {
    let target_y = target_y_float.floor() as i32;
    let frac = (target_y_float - target_y as f32).clamp(0.0, 1.0);
    let (d_solid, d_air) = if frac <= 0.5 {
        let denom = (1.0 - frac).max(1e-3);
        ((frac / denom).clamp(0.0, 1.0), -1.0)
    } else {
        let denom = frac.max(1e-3);
        (1.0, ((frac - 1.0) / denom).clamp(-1.0, 0.0))
    };
    (target_y, d_solid, d_air)
}

// ── Sampling ──────────────────────────────────────────────────────────────

/// Read the density at world cell (wx, wy, wz). Returns 1.0 (deep solid) for
/// cells in unloaded chunks — conservative default so we never carve into
/// unknown territory.
#[inline]
pub fn read_density(
    fields: &HashMap<(i32, i32, i32), DensityField>,
    cs: i32,
    wx: i32, wy: i32, wz: i32,
) -> f32 {
    let cx = wx.div_euclid(cs);
    let cy = wy.div_euclid(cs);
    let cz = wz.div_euclid(cs);
    let lx = wx.rem_euclid(cs) as usize;
    let ly = wy.rem_euclid(cs) as usize;
    let lz = wz.rem_euclid(cs) as usize;
    fields.get(&(cx, cy, cz))
        .map(|df| df.get(lx, ly, lz).density).unwrap_or(1.0)
}

/// Count how many of the 6 face-adjacent neighbors are air (density ≤ 0).
/// Used for the "thin feature" classification in the formation removal pass.
///
/// Optimized: most face-adjacent reads (typically 5 of 6, often all 6) sit in
/// the same chunk as the center cell. We cache the primary chunk lookup and
/// fall back to the generic `read_density` only when a neighbor crosses a
/// chunk boundary. Inside `formation_removal_pass` this runs for thousands
/// of cells × 3 erosion iterations, so eliminating the 5 redundant
/// `HashMap.get` calls (and the div_euclid/rem_euclid math feeding them) is
/// directly on the hot path.
#[inline]
pub fn count_air_face_neighbors(
    fields: &HashMap<(i32, i32, i32), DensityField>,
    cs: i32,
    wx: i32, wy: i32, wz: i32,
) -> u8 {
    let cx = wx.div_euclid(cs);
    let cy = wy.div_euclid(cs);
    let cz = wz.div_euclid(cs);
    let lx = wx.rem_euclid(cs);
    let ly = wy.rem_euclid(cs);
    let lz = wz.rem_euclid(cs);
    let cs_u = cs as usize;

    // Primary chunk cached once for all 6 neighbor checks.
    let primary = fields.get(&(cx, cy, cz));

    // Helper: returns true if neighbor at (dx, dy, dz) is air.
    // Stays inside the primary chunk when the local coord is in [0, cs-1]
    // after the offset; otherwise crosses a face and falls back to the
    // slow path with its own chunk lookup.
    let mut count = 0u8;

    // -X / +X
    if lx > 0 {
        if let Some(df) = primary {
            if df.get((lx - 1) as usize, ly as usize, lz as usize).density <= 0.0 { count += 1; }
        } else if read_density(fields, cs, wx - 1, wy, wz) <= 0.0 { count += 1; }
    } else if read_density(fields, cs, wx - 1, wy, wz) <= 0.0 { count += 1; }

    if (lx as usize) + 1 < cs_u {
        if let Some(df) = primary {
            if df.get((lx + 1) as usize, ly as usize, lz as usize).density <= 0.0 { count += 1; }
        } else if read_density(fields, cs, wx + 1, wy, wz) <= 0.0 { count += 1; }
    } else if read_density(fields, cs, wx + 1, wy, wz) <= 0.0 { count += 1; }

    // -Y / +Y
    if ly > 0 {
        if let Some(df) = primary {
            if df.get(lx as usize, (ly - 1) as usize, lz as usize).density <= 0.0 { count += 1; }
        } else if read_density(fields, cs, wx, wy - 1, wz) <= 0.0 { count += 1; }
    } else if read_density(fields, cs, wx, wy - 1, wz) <= 0.0 { count += 1; }

    if (ly as usize) + 1 < cs_u {
        if let Some(df) = primary {
            if df.get(lx as usize, (ly + 1) as usize, lz as usize).density <= 0.0 { count += 1; }
        } else if read_density(fields, cs, wx, wy + 1, wz) <= 0.0 { count += 1; }
    } else if read_density(fields, cs, wx, wy + 1, wz) <= 0.0 { count += 1; }

    // -Z / +Z
    if lz > 0 {
        if let Some(df) = primary {
            if df.get(lx as usize, ly as usize, (lz - 1) as usize).density <= 0.0 { count += 1; }
        } else if read_density(fields, cs, wx, wy, wz - 1) <= 0.0 { count += 1; }
    } else if read_density(fields, cs, wx, wy, wz - 1) <= 0.0 { count += 1; }

    if (lz as usize) + 1 < cs_u {
        if let Some(df) = primary {
            if df.get(lx as usize, ly as usize, (lz + 1) as usize).density <= 0.0 { count += 1; }
        } else if read_density(fields, cs, wx, wy, wz + 1) <= 0.0 { count += 1; }
    } else if read_density(fields, cs, wx, wy, wz + 1) <= 0.0 { count += 1; }

    count
}

/// Find the natural cave floor's iso-surface Y in a column by interpolating
/// the density values at adjacent cells. Returns `Some(y_float)` if a real
/// air→solid transition is found in the scan window, `None` if the column
/// is fully solid (wall) or fully air (open shaft) within range.
///
/// Works for both legacy integer-Y placements (iso at +0.5 voxels) and
/// sub-voxel placements (iso at any frac).
pub fn natural_floor_y_iso(
    fields: &HashMap<(i32, i32, i32), DensityField>,
    cs: i32,
    wx: i32,
    base_y: i32,
    wz: i32,
    scan_up: i32,
    scan_down: i32,
) -> Option<f32> {
    let top = base_y + scan_up;
    let bot = base_y - scan_down;

    let mut prev = read_density(fields, cs, wx, top + 1, wz);
    for y in (bot..=top).rev() {
        let d = read_density(fields, cs, wx, y, wz);
        if d >= 0.0 && prev < 0.0 {
            let denom = (d - prev).max(1e-3);
            return Some(y as f32 + d / denom);
        }
        prev = d;
    }
    None
}

// ── Density write primitives (seam-aware) ────────────────────────────────

/// Apply a write decision to ALL chunks sharing this world cell. The decision
/// closure looks at current density+material and returns Some((new_density,
/// new_material)) or None to skip. Tracks every actual write in `written`.
///
/// Hot path: ~95% of cells inside a flatten/pile/brush zone are *interior*
/// (`lx,ly,lz` all > 0) and live in exactly one chunk. The fast path below
/// does the single-chunk write directly and skips `cell_locations` —
/// avoiding the 8-slot `[Option<...>; 8]` build, the 3 nested for-loops, and
/// the `into_iter().flatten()` machinery. Only cells that sit on a chunk
/// boundary (any of `lx`/`ly`/`lz` == 0) take the multi-chunk fan-out path.
pub fn write_all_locations<F>(
    fields: &mut HashMap<(i32, i32, i32), DensityField>,
    cs: i32,
    wx: i32, wy: i32, wz: i32,
    decide: F,
    dirty_set: &mut HashSet<(i32, i32, i32)>,
    written: &mut Vec<WrittenCell>,
    changed_count: &mut u32,
)
where
    F: Fn(f32, Material) -> Option<(f32, Material)>,
{
    // Interior fast path — single chunk, no fan-out needed.
    let lx_i = wx.rem_euclid(cs);
    let ly_i = wy.rem_euclid(cs);
    let lz_i = wz.rem_euclid(cs);
    if lx_i > 0 && ly_i > 0 && lz_i > 0 {
        let key = (wx.div_euclid(cs), wy.div_euclid(cs), wz.div_euclid(cs));
        if let Some(df) = fields.get_mut(&key) {
            let lx = lx_i as usize;
            let ly = ly_i as usize;
            let lz = lz_i as usize;
            let s = df.get_mut(lx, ly, lz);
            let orig_density = s.density;
            let orig_material = s.material;
            if let Some((new_d, new_m)) = decide(orig_density, orig_material) {
                if (orig_density - new_d).abs() > 1e-3 || orig_material != new_m {
                    *changed_count += 1;
                }
                s.density = new_d;
                s.material = new_m;
                dirty_set.insert(key);
                written.push(WrittenCell {
                    key,
                    lx, ly, lz,
                    new_density: new_d,
                    new_material: new_m,
                    orig_density,
                    orig_material,
                });
            }
        }
        return;
    }

    // Boundary fan-out path — cell is shared by 2/4/8 chunks; write to each.
    for slot in cell_locations(cs, wx, wy, wz).into_iter().flatten() {
        let (key, lx, ly, lz) = slot;
        if let Some(df) = fields.get_mut(&key) {
            let s = df.get_mut(lx, ly, lz);
            let orig_density = s.density;
            let orig_material = s.material;
            if let Some((new_d, new_m)) = decide(orig_density, orig_material) {
                if (orig_density - new_d).abs() > 1e-3 || orig_material != new_m {
                    *changed_count += 1;
                }
                s.density = new_d;
                s.material = new_m;
                dirty_set.insert(key);
                written.push(WrittenCell {
                    key,
                    lx, ly, lz,
                    new_density: new_d,
                    new_material: new_m,
                    orig_density,
                    orig_material,
                });
            }
        }
    }
}

/// Conditional raise: write if new density would be HIGHER than existing.
pub fn write_raise(
    fields: &mut HashMap<(i32, i32, i32), DensityField>,
    cs: i32,
    wx: i32, wy: i32, wz: i32,
    target_density: f32,
    target_material: Material,
    dirty_set: &mut HashSet<(i32, i32, i32)>,
    written: &mut Vec<WrittenCell>,
    changed_count: &mut u32,
) {
    write_all_locations(fields, cs, wx, wy, wz, |cur_d, cur_m| {
        if cur_d < target_density - 1e-3 {
            let mat = if target_density > 0.0 { target_material } else { cur_m };
            Some((target_density, mat))
        } else { None }
    }, dirty_set, written, changed_count);
}

/// Conditional lower: write if new density would be LOWER than existing.
pub fn write_lower(
    fields: &mut HashMap<(i32, i32, i32), DensityField>,
    cs: i32,
    wx: i32, wy: i32, wz: i32,
    target_density: f32,
    dirty_set: &mut HashSet<(i32, i32, i32)>,
    written: &mut Vec<WrittenCell>,
    changed_count: &mut u32,
) {
    write_all_locations(fields, cs, wx, wy, wz, |cur_d, cur_m| {
        if cur_d > target_density + 1e-3 {
            let mat = if target_density <= 0.0 { Material::Air } else { cur_m };
            Some((target_density, mat))
        } else { None }
    }, dirty_set, written, changed_count);
}

/// Unconditional force-write — used at boundary cells where we MUST set the
/// exact density to position the iso surface correctly.
///
/// Material assignment:
///   density >= 0 → use `target_material` (treat 0 as solid for material).
///   density <  0 → Material::Air.
///
/// CRITICAL: the `>= 0` (not `> 0`) check handles sub-voxel boundaries where
/// `d_solid == 0` exactly (frac=0 case in `subvoxel_boundary_densities`).
/// Without this, the boundary cell would get Material::Air despite being on
/// the solid side of the iso, causing hermite-extracted edges between two
/// such cells to inherit Air material → matte-black submeshes in UE.
pub fn write_force(
    fields: &mut HashMap<(i32, i32, i32), DensityField>,
    cs: i32,
    wx: i32, wy: i32, wz: i32,
    target_density: f32,
    target_material: Material,
    dirty_set: &mut HashSet<(i32, i32, i32)>,
    written: &mut Vec<WrittenCell>,
    changed_count: &mut u32,
) {
    write_all_locations(fields, cs, wx, wy, wz, |_cur_d, _cur_m| {
        let mat = if target_density >= 0.0 {
            target_material
        } else {
            Material::Air
        };
        Some((target_density, mat))
    }, dirty_set, written, changed_count);
}

/// Restore every written cell to its intended density+material. Run AFTER
/// `sync_boundary_density` to defeat its `min()` merging at chunk seams.
pub fn restore_written_cells(
    fields: &mut HashMap<(i32, i32, i32), DensityField>,
    written: &[WrittenCell],
) {
    for w in written {
        if let Some(df) = fields.get_mut(&w.key) {
            let s = df.get_mut(w.lx, w.ly, w.lz);
            if (s.density - w.new_density).abs() > 1e-3 {
                s.density = w.new_density;
                s.material = w.new_material;
            }
        }
    }
}

/// Roll back every written cell to its **pre-write** density+material.
/// Used by the cinematic-collapse preview flow: we run `place_collapse_pile`
/// to compute the final shape, ship a preview mesh, then call this to revert
/// the density store before the real impact-time commit.
pub fn rollback_written_cells(
    fields: &mut HashMap<(i32, i32, i32), DensityField>,
    written: &[WrittenCell],
) {
    // Walk in reverse so chronologically-later writes are reverted first;
    // ensures the original pre-pile-event state is restored even when the
    // same cell was touched multiple times during placement.
    for w in written.iter().rev() {
        if let Some(df) = fields.get_mut(&w.key) {
            let s = df.get_mut(w.lx, w.ly, w.lz);
            s.density = w.orig_density;
            s.material = w.orig_material;
        }
    }
}

/// Re-apply previously-rolled-back writes (the "commit" half of preview-
/// then-commit). Walks `written` in original order and writes the recorded
/// post-state back into the density store.
pub fn replay_written_cells(
    fields: &mut HashMap<(i32, i32, i32), DensityField>,
    written: &[WrittenCell],
) {
    for w in written {
        if let Some(df) = fields.get_mut(&w.key) {
            let s = df.get_mut(w.lx, w.ly, w.lz);
            s.density = w.new_density;
            s.material = w.new_material;
        }
    }
}

// ── Formation removal pass ───────────────────────────────────────────────

/// Configuration for the formation-removal pass.
///
/// The pass identifies and carves "thin features" (stalactites, stalagmites,
/// columns, flowstone tips) inside an axis-aligned cylinder around an
/// anchor point. Walls and ceilings are preserved by the air-neighbor
/// threshold — a wall has 4-6 solid face-neighbors while a thin formation
/// has at most 2.
pub struct FormationRemovalConfig {
    pub anchor_x: i32,
    pub anchor_z: i32,
    pub footprint_x: i32,    // half-extent in X (interior reaches ±this)
    pub footprint_z: i32,    // half-extent in Z
    pub radius_extra: i32,   // additional voxels beyond the footprint
    pub anchor_y: i32,       // center Y of the scan window
    pub max_above: i32,      // scan this many voxels above anchor_y
    pub scan_below: i32,     // scan this many voxels below anchor_y
    pub air_neighbors_threshold: u8, // ≥ this many air face-neighbors → thin
    pub max_iterations: u32, // erosion passes (each peels the current outer shell)
}

impl FormationRemovalConfig {
    pub fn default_for_pile(anchor_x: i32, anchor_y: i32, anchor_z: i32, half_x: i32, half_z: i32) -> Self {
        Self {
            anchor_x,
            anchor_z,
            footprint_x: half_x,
            footprint_z: half_z,
            radius_extra: 4,
            anchor_y,
            max_above: 12,
            scan_below: 4,
            air_neighbors_threshold: 4,
            max_iterations: 3,
        }
    }
}

/// Identify and carve thin formation features in a cylinder around an
/// anchor. Iterative erosion lets us peel chunkier formations (flowstone
/// slabs, column bases) without false-positiving into walls — within a
/// single iteration we snapshot+carve atomically; across iterations, the
/// outer thin shell becomes air and the next layer crosses the threshold.
///
/// Returns the number of cells actually carved.
///
/// Perf: iteration 0 does a full cylinder sweep. Iterations 1+ only re-test
/// the 6 face-neighbors of the previous iteration's victims. Rationale: a
/// cell can only newly cross the air-neighbor threshold in iter N+1 if at
/// least one of its face-neighbors was carved in iter N — i.e. it must be
/// a face-neighbor of an iter-N victim. Whole-pass cost on typical 3-iter
/// runs drops to roughly the iter-0 sweep + a handful of frontier checks.
pub fn formation_removal_pass(
    fields: &mut HashMap<(i32, i32, i32), DensityField>,
    cs: i32,
    cfg: &FormationRemovalConfig,
    dirty_set: &mut HashSet<(i32, i32, i32)>,
    written: &mut Vec<WrittenCell>,
    changed_count: &mut u32,
) -> u32 {
    let radius = cfg.footprint_x.max(cfg.footprint_z) + cfg.radius_extra;
    let radius_f = radius as f32;
    let radius_sq = radius_f * radius_f;
    let y_min = cfg.anchor_y - cfg.scan_below;
    let y_max = cfg.anchor_y + cfg.max_above;
    let mut total_carved = 0u32;

    // Cylinder bounds check (matches the iter-0 sweep filter).
    let in_cylinder = |wx: i32, wy: i32, wz: i32| -> bool {
        if wy < y_min || wy > y_max { return false; }
        let dx = (wx - cfg.anchor_x) as f32;
        let dz = (wz - cfg.anchor_z) as f32;
        dx * dx + dz * dz <= radius_sq
    };

    // Iteration 0: full cylinder sweep.
    let mut victims: Vec<(i32, i32, i32)> = Vec::new();
    for dx in -radius..=radius {
        for dz in -radius..=radius {
            let dist_sq = (dx * dx + dz * dz) as f32;
            if dist_sq > radius_sq { continue; }
            let wx = cfg.anchor_x + dx;
            let wz = cfg.anchor_z + dz;
            for y_off in -cfg.scan_below..=cfg.max_above {
                let wy = cfg.anchor_y + y_off;
                let d = read_density(fields, cs, wx, wy, wz);
                if d <= 0.0 { continue; }
                let air_neighbors = count_air_face_neighbors(fields, cs, wx, wy, wz);
                if air_neighbors >= cfg.air_neighbors_threshold {
                    victims.push((wx, wy, wz));
                }
            }
        }
    }

    if victims.is_empty() { return 0; }

    // Carve iteration-0 victims, then run frontier-based passes.
    for &(wx, wy, wz) in &victims {
        let before = *changed_count;
        write_lower(fields, cs, wx, wy, wz, -1.0, dirty_set, written, changed_count);
        if *changed_count > before { total_carved += 1; }
    }

    // Frontier passes: candidates = face-neighbors of previous victims, in cylinder.
    // Dedup via HashSet so a candidate touched by multiple victims is tested once.
    let mut prev_victims = victims;
    for _iter in 1..cfg.max_iterations {
        if prev_victims.is_empty() { break; }
        let mut candidates: HashSet<(i32, i32, i32)> = HashSet::with_capacity(prev_victims.len() * 6);
        for &(vx, vy, vz) in &prev_victims {
            // 6 face-neighbors. Filter to cylinder; the cell may itself be
            // air now (the victim it neighbors became air last iter), so we
            // re-check density inside the test loop.
            for (nx, ny, nz) in [
                (vx - 1, vy, vz), (vx + 1, vy, vz),
                (vx, vy - 1, vz), (vx, vy + 1, vz),
                (vx, vy, vz - 1), (vx, vy, vz + 1),
            ] {
                if in_cylinder(nx, ny, nz) {
                    candidates.insert((nx, ny, nz));
                }
            }
        }

        let mut next_victims: Vec<(i32, i32, i32)> = Vec::new();
        for &(wx, wy, wz) in &candidates {
            let d = read_density(fields, cs, wx, wy, wz);
            if d <= 0.0 { continue; }
            let air_neighbors = count_air_face_neighbors(fields, cs, wx, wy, wz);
            if air_neighbors >= cfg.air_neighbors_threshold {
                next_victims.push((wx, wy, wz));
            }
        }

        if next_victims.is_empty() { break; }
        for &(wx, wy, wz) in &next_victims {
            let before = *changed_count;
            write_lower(fields, cs, wx, wy, wz, -1.0, dirty_set, written, changed_count);
            if *changed_count > before { total_carved += 1; }
        }
        prev_victims = next_victims;
    }

    total_carved
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subvoxel_at_half_voxel_uses_classic_densities() {
        let (y, ds, da) = subvoxel_boundary_densities(10.5);
        assert_eq!(y, 10);
        assert!((ds - 1.0).abs() < 1e-3);
        assert!((da + 1.0).abs() < 1e-3);
    }

    #[test]
    fn subvoxel_at_quarter_voxel_positions_iso_correctly() {
        let (y, ds, da) = subvoxel_boundary_densities(10.25);
        assert_eq!(y, 10);
        let iso = y as f32 + ds / (ds - da);
        assert!((iso - 10.25).abs() < 0.01);
    }

    #[test]
    fn subvoxel_at_three_quarter_voxel_positions_iso_correctly() {
        let (y, ds, da) = subvoxel_boundary_densities(10.75);
        assert_eq!(y, 10);
        let iso = y as f32 + ds / (ds - da);
        assert!((iso - 10.75).abs() < 0.01);
    }

    #[test]
    fn cell_locations_interior_returns_one() {
        let locs = cell_locations(16, 5, 5, 5);
        assert_eq!(locs.iter().filter(|x| x.is_some()).count(), 1);
    }

    #[test]
    fn cell_locations_face_returns_two() {
        // wx=0 → lx=0, on face. Should fan to 2 chunks.
        let locs = cell_locations(16, 0, 5, 5);
        assert_eq!(locs.iter().filter(|x| x.is_some()).count(), 2);
    }

    #[test]
    fn cell_locations_corner_returns_eight() {
        // wx=0, wy=0, wz=0 — all on faces. Fan to 2*2*2 = 8.
        let locs = cell_locations(16, 0, 0, 0);
        assert_eq!(locs.iter().filter(|x| x.is_some()).count(), 8);
    }

    /// Frontier-iteration regression: a 3-cell stalactite hanging from a
    /// ceiling needs all 3 iterations to peel (threshold=5). Iter 1 carves
    /// only the bottom cell; iter 2 needs to find the middle cell, which is
    /// only reachable by walking face-neighbors of the previous victim.
    #[test]
    fn formation_removal_peels_thick_stalactite_over_iterations() {
        let cs = 16;
        let mut fields: HashMap<(i32, i32, i32), DensityField> = HashMap::new();
        let mut df = DensityField::new((cs + 1) as usize);
        // Carve everything to air.
        for z in 0..(cs + 1) as usize {
            for y in 0..(cs + 1) as usize {
                for x in 0..(cs + 1) as usize {
                    df.get_mut(x, y, z).density = -1.0;
                    df.get_mut(x, y, z).material = Material::Air;
                }
            }
        }
        // Ceiling at y=14 across the whole plane (solid Limestone).
        for z in 0..cs as usize {
            for x in 0..cs as usize {
                let s = df.get_mut(x, 14, z);
                s.density = 1.0;
                s.material = Material::Limestone;
            }
        }
        // Stalactite hanging at (8, ?, 8): y=11,12,13 solid.
        for y in 11..=13 {
            let s = df.get_mut(8, y, 8);
            s.density = 1.0;
            s.material = Material::Limestone;
        }
        fields.insert((0, 0, 0), df);

        let cfg = FormationRemovalConfig {
            anchor_x: 8,
            anchor_z: 8,
            footprint_x: 1,
            footprint_z: 1,
            radius_extra: 2,
            anchor_y: 12,
            max_above: 2,
            scan_below: 2,
            air_neighbors_threshold: 5,
            max_iterations: 3,
        };
        let mut dirty = HashSet::new();
        let mut written = Vec::new();
        let mut changed = 0u32;
        let carved = formation_removal_pass(
            &mut fields, cs, &cfg, &mut dirty, &mut written, &mut changed,
        );
        assert_eq!(carved, 3, "all 3 stalactite cells should be carved across iterations");

        let df = fields.get(&(0, 0, 0)).unwrap();
        for y in 11..=13 {
            assert!(df.get(8, y, 8).density <= 0.0, "stalactite cell at y={} not carved", y);
        }
        // Ceiling must remain solid (wall, 0 air face neighbors).
        assert!(df.get(8, 14, 8).density > 0.0, "ceiling cell incorrectly carved");
    }
}
