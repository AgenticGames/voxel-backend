//! All-in single-placement building flatten.
//!
//! Algorithm: explicit per-column target-Y ramp with sub-voxel boundary
//! density (3C, ±0 voxel surface alignment), plus a convex-hull buttress
//! made of capped-cone SDFs for cantilever columns (2C, fills cliffs to
//! nearest natural rock without filling ravines).
//!
//! For each (wx, wz) in (footprint + apron):
//!   1. Determine `target_y_float` in 4 cases:
//!        a. Interior footprint  -> base_y_float
//!        b. Apron, natural floor found (and <= base_y) -> lerp toward natural
//!        c. Apron, no natural floor (wall/void), but a buttress cone covers
//!           the column -> the cone's surface Y at this column
//!        d. Apron, no natural, no buttress -> skip (preserve wall / hover air)
//!   2. Write sub-voxel boundary densities at target_y_float (DC iso-surface
//!      lands exactly at base_y_float for interior, at the ramp Y for apron).
//!   3. Fill solid below the boundary up to FILL_DEPTH voxels.
//!   4. Force air above the boundary up to clearance.
//!
//! Buttress hull is built UPFRONT for every cantilever column in the footprint:
//! cast support rays in the lower hemisphere (Y <= base_y so no reverse ramps),
//! collect the closest hits, materialize them as capped cones. Apron columns
//! query the hull to discover whether a cantilevered support reaches them.

use std::collections::HashSet;

use glam::Vec3;
use voxel_core::material::Material;
use voxel_gen::config::GenerationConfig;
use voxel_noise::{simplex::Simplex3D, NoiseSource};

use crate::sdf::{find_support_rays, sdf_capped_cone};
use crate::store::{sync_boundary_density, ChunkStore};
use crate::types::ConvertedMesh;

// ── Tunables ───────────────────────────────────────────────────────────────

const APRON_FRAC: f32 = 0.50;          // 50% of footprint each side = 200% total
const APRON_MIN: i32 = 3;              // never less than 3 voxels of apron

// Formation removal pass — carves thin features (stalactites, stalagmites,
// columns, flowstone) in a cylinder around the building. Walls/ceilings are
// preserved by the air-neighbor threshold.
const FORMATION_EXTRA_RADIUS: i32 = 4; // beyond apron, additional cylinder
const FORMATION_MAX_ABOVE: i32 = 12;   // scan this many voxels above floor
const FORMATION_SCAN_BELOW: i32 = 4;   // scan this many voxels below floor (apron body)
const FORMATION_AIR_NEIGHBORS: u8 = 4; // ≥this many air face-neighbors → thin feature
const FORMATION_MAX_ITERATIONS: u32 = 3; // iterate to erode chunkier formations
const FILL_DEPTH: i32 = 6;             // depth of solid support below floor
const SUPPORT_CHECK_DEPTH: i32 = 4;    // column with no solid here = cantilever
const SUPPORT_RAY_COUNT: usize = 16;
const SUPPORT_RAY_UP_TOL: f32 = 0.05;
const SUPPORT_RAYS_PER_COL: usize = 3; // top-N closest hits per cantilever col
const BUTTRESS_R_BASE: f32 = 1.5;
const BUTTRESS_R_TIP: f32 = 0.8;
const RAMP_NOISE_AMP: f32 = 0.0;       // disabled — was causing per-column jitter
const ADJACENT_Y_CAP: f32 = 1.0;       // max apron Y delta per voxel (1:1 slope)

#[inline]
fn apron_radius_for(terrace_size: i32) -> i32 {
    ((terrace_size as f32) * APRON_FRAC).round().max(APRON_MIN as f32) as i32
}

#[inline]
fn cap_distance_for(terrace_size: i32) -> f32 {
    (apron_radius_for(terrace_size) as f32) * 2.0
}

// ── Sub-voxel density math (3C) ───────────────────────────────────────────

/// For a target surface at `target_y_float`, return the integer floor cell
/// and the densities to write at (floor_cell, floor_cell+1) such that the DC
/// iso surface lands exactly at `target_y_float`.
#[inline]
fn subvoxel_boundary_densities(target_y_float: f32) -> (i32, f32, f32) {
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

// ── Support hull (Phase 2 + 2C) ───────────────────────────────────────────

#[derive(Clone, Copy)]
struct SupportCone {
    base: Vec3,    // at the cantilever column, at base_y_float
    tip: Vec3,     // at the natural rock hit point
    r_base: f32,
    r_tip: f32,
}

#[derive(Default)]
struct SupportHull {
    cones: Vec<SupportCone>,
}

impl SupportHull {
    /// Returns Some(top_y) if a buttress cone covers the column at (wx, wz);
    /// the Y is the highest cone-surface Y in that column, used as a target Y
    /// for fill. Returns None if no cone covers this column.
    fn cone_top_in_column(&self, wx: f32, wz: f32, search_lo: f32, search_hi: f32) -> Option<f32> {
        if self.cones.is_empty() { return None; }
        // Step the column from search_hi down to search_lo, looking for the
        // highest Y where the union of cones gives a NEGATIVE SDF (inside).
        let mut y = search_hi;
        let step = 0.5;
        while y >= search_lo {
            let p = Vec3::new(wx, y, wz);
            let mut min_d = f32::INFINITY;
            for c in &self.cones {
                let d = sdf_capped_cone(p, c.base, c.tip, c.r_base, c.r_tip);
                if d < min_d { min_d = d; }
            }
            if min_d < 0.0 {
                return Some(y);
            }
            y -= step;
        }
        None
    }
}

/// Build the buttress hull by scanning every footprint column for cantilever
/// (no solid within SUPPORT_CHECK_DEPTH below the floor) and casting rays
/// from each cantilever column to the nearest natural rock.
fn build_support_hull(
    store: &ChunkStore,
    cs: i32,
    base: glam::IVec3,
    base_y_float: f32,
    terrace_size: i32,
) -> SupportHull {
    let cap_dist = cap_distance_for(terrace_size);
    let base_y_int = base_y_float.floor() as i32;
    let mut cones: Vec<SupportCone> = Vec::new();

    for dx in 0..terrace_size {
        for dz in 0..terrace_size {
            let wx = base.x + dx;
            let wz = base.z + dz;

            // Cantilever check.
            let mut has_support = false;
            for k in 1..=SUPPORT_CHECK_DEPTH {
                let y = base_y_int - k;
                let cx = wx.div_euclid(cs);
                let cy = y.div_euclid(cs);
                let cz = wz.div_euclid(cs);
                let lx = wx.rem_euclid(cs) as usize;
                let ly = y.rem_euclid(cs) as usize;
                let lz = wz.rem_euclid(cs) as usize;
                if let Some(df) = store.density_fields.get(&(cx, cy, cz)) {
                    if df.get(lx, ly, lz).density > 0.0 {
                        has_support = true;
                        break;
                    }
                }
            }
            if has_support { continue; }

            // Cantilever — search rays in lower hemisphere only.
            let origin = Vec3::new(wx as f32 + 0.5, base_y_float, wz as f32 + 0.5);
            let hits = find_support_rays(
                &store.density_fields, cs, origin, cap_dist,
                SUPPORT_RAY_COUNT, SUPPORT_RAY_UP_TOL,
            );
            for hit in hits.iter().take(SUPPORT_RAYS_PER_COL) {
                cones.push(SupportCone {
                    base: origin,
                    tip: hit.hit_pos,
                    r_base: BUTTRESS_R_BASE,
                    r_tip: BUTTRESS_R_TIP,
                });
            }
        }
    }
    SupportHull { cones }
}

// ── Per-column ramp Y resolution ──────────────────────────────────────────

/// Find the natural cave floor — returns the EXACT iso surface position by
/// linear interpolation between the densities of adjacent cells. This works
/// for both legacy integer-Y flattens (iso at +0.5) and sub-voxel flattens
/// (iso at any frac), so the apron lerps correctly toward existing surfaces.
fn natural_floor_y_iso(
    fields: &std::collections::HashMap<(i32, i32, i32), voxel_core::density::DensityField>,
    cs: i32,
    wx: i32,
    base_y: i32,
    wz: i32,
) -> Option<f32> {
    const SCAN_UP: i32 = 4;
    const SCAN_DOWN: i32 = 24;
    let cx = wx.div_euclid(cs);
    let cz = wz.div_euclid(cs);
    let lx = wx.rem_euclid(cs) as usize;
    let lz = wz.rem_euclid(cs) as usize;
    let sample_density = |y: i32| -> Option<f32> {
        let cy = y.div_euclid(cs);
        let ly = y.rem_euclid(cs) as usize;
        fields.get(&(cx, cy, cz)).map(|df| df.get(lx, ly, lz).density)
    };
    let top = base_y + SCAN_UP;
    let bot = base_y - SCAN_DOWN;

    // Walk top -> bottom. Look for transition from negative density above to
    // density >= 0 at this y. That's the iso boundary. Interpolate the exact
    // crossing position.
    let mut prev = sample_density(top + 1).unwrap_or(-1.0);
    for y in (bot..=top).rev() {
        let d = match sample_density(y) {
            Some(d) => d,
            None => { prev = -1.0; continue; }
        };
        if d >= 0.0 && prev < 0.0 {
            let denom = (d - prev).max(1e-3);
            return Some(y as f32 + d / denom);
        }
        prev = d;
    }
    None
}

#[inline]
fn ramp_y_noise(cfg: &GenerationConfig, wx: i32, wz: i32) -> f32 {
    let freq = cfg.noise.cavern_frequency;
    let s = Simplex3D::new(cfg.seed);
    let n = s.sample(wx as f64 * freq, 0.0, wz as f64 * freq) as f32;
    n * RAMP_NOISE_AMP
}

#[inline]
fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Resolve the target-Y for one column. Returns None if the column should
/// be skipped (wall preservation OR cantilever with no buttress).
fn resolve_target_y(
    store: &ChunkStore,
    cs: i32,
    base: glam::IVec3,
    base_y_float: f32,
    apron_radius: f32,
    cfg: &GenerationConfig,
    hull: &SupportHull,
    wx: i32,
    wz: i32,
    edge_dist: f32,
    in_interior: bool,
) -> Option<f32> {
    // Case (a): interior footprint
    if in_interior {
        return Some(base_y_float);
    }

    // Case (b): apron with a real natural floor at-or-below base_y
    if let Some(nat_y_iso) = natural_floor_y_iso(&store.density_fields, cs, wx, base.y, wz) {
        // Clamp natural at the base_y_float ceiling — apron only ramps DOWN.
        let nat_clamped = nat_y_iso.min(base_y_float);

        // ★ FLAT-MATCH: if the existing surface is already within ~1 voxel of
        // the building floor, skip the ramp entirely and keep the apron at
        // base_y_float. This prevents the "leveller surface gets bumpy when
        // a building is placed on it" artifact — the apron used to ramp
        // 0.5 voxels down even on perfectly-flat existing terrain.
        const FLAT_MATCH_THRESHOLD: f32 = 1.0;
        if (base_y_float - nat_clamped).abs() < FLAT_MATCH_THRESHOLD {
            return Some(base_y_float);
        }

        let t = (edge_dist / apron_radius).clamp(0.0, 1.0);
        let influence = 1.0 - smoothstep(t);
        let lerped = base_y_float * influence + nat_clamped * (1.0 - influence);
        let wobble = ramp_y_noise(cfg, wx, wz) * influence;
        let max_drop = ADJACENT_Y_CAP * edge_dist;
        let target = (lerped + wobble).max(base_y_float - max_drop);
        return Some(target);
    }

    // Case (c): cantilever buttress reaches this column
    let cone_search_lo = base_y_float - cap_distance_for(8);
    let cone_search_hi = base_y_float + 1.0;
    if let Some(top_y) = hull.cone_top_in_column(
        wx as f32 + 0.5, wz as f32 + 0.5, cone_search_lo, cone_search_hi,
    ) {
        return Some(top_y.min(base_y_float));
    }

    // Case (d): wall column or unsupported air — leave alone.
    None
}

// ── Cell write with seam-restore tracking ─────────────────────────────────

/// Tracks every cell we wrote, in either direction, so we can restore against
/// `sync_boundary_density`'s min() at chunk seams.
type WrittenCell = ((i32, i32, i32), usize, usize, usize, f32, Material);

/// A single world voxel cell can be stored in up to 8 chunk DensityFields
/// because each chunk includes its +cs boundary face (size = chunk_size + 1).
/// World cell (wx, wy, wz) lives in:
///   - Primary chunk (wx/cs, wy/cs, wz/cs) at local (wx%cs, wy%cs, wz%cs).
///   - For each axis where wx is a multiple of cs (i.e. on a chunk face),
///     ALSO in the previous chunk at local index = cs.
///
/// This returns every (chunk_key, lx, ly, lz) location for the cell. Skips
/// negative chunk coords if they exist (caller filters).
fn cell_locations(cs: i32, wx: i32, wy: i32, wz: i32) -> [Option<((i32, i32, i32), usize, usize, usize)>; 8] {
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

/// Apply a write decision to ALL chunks sharing this world cell. The decision
/// closure looks at current density and returns Some((new_density, new_material))
/// or None to skip. Tracks every write in `written` so post-sync restoration
/// keeps every shared location consistent across chunk seams.
fn write_all_locations(
    store: &mut ChunkStore,
    cs: i32,
    wx: i32, wy: i32, wz: i32,
    decide: impl Fn(f32, Material) -> Option<(f32, Material)> + Copy,
    dirty_set: &mut HashSet<(i32, i32, i32)>,
    written: &mut Vec<WrittenCell>,
    changed_count: &mut u32,
) {
    for slot in cell_locations(cs, wx, wy, wz).into_iter().flatten() {
        let (key, lx, ly, lz) = slot;
        if let Some(df) = store.density_fields.get_mut(&key) {
            let s = df.get_mut(lx, ly, lz);
            if let Some((new_d, new_m)) = decide(s.density, s.material) {
                if (s.density - new_d).abs() > 1e-3 || s.material != new_m {
                    *changed_count += 1;
                }
                s.density = new_d;
                s.material = new_m;
                dirty_set.insert(key);
                written.push((key, lx, ly, lz, new_d, new_m));
            }
        }
    }
}

/// Conditional raise: writes if new density would be HIGHER than existing.
fn write_raise(
    store: &mut ChunkStore,
    cs: i32,
    wx: i32, wy: i32, wz: i32,
    target_density: f32,
    host_material: Material,
    dirty_set: &mut HashSet<(i32, i32, i32)>,
    written: &mut Vec<WrittenCell>,
    changed_count: &mut u32,
) {
    write_all_locations(store, cs, wx, wy, wz, |cur_d, cur_m| {
        if cur_d < target_density - 1e-3 {
            let mat = if target_density > 0.0 { host_material } else { cur_m };
            Some((target_density, mat))
        } else { None }
    }, dirty_set, written, changed_count);
}

/// Conditional lower: writes if new density would be LOWER than existing.
fn write_lower(
    store: &mut ChunkStore,
    cs: i32,
    wx: i32, wy: i32, wz: i32,
    target_density: f32,
    dirty_set: &mut HashSet<(i32, i32, i32)>,
    written: &mut Vec<WrittenCell>,
    changed_count: &mut u32,
) {
    write_all_locations(store, cs, wx, wy, wz, |cur_d, cur_m| {
        if cur_d > target_density + 1e-3 {
            let mat = if target_density <= 0.0 { Material::Air } else { cur_m };
            Some((target_density, mat))
        } else { None }
    }, dirty_set, written, changed_count);
}

/// Unconditional force-write — used at boundary cells (target_y and
/// target_y+1) where we MUST set the exact sub-voxel density to position
/// the iso surface correctly. Writes to ALL chunks sharing the world cell.
fn write_force(
    store: &mut ChunkStore,
    cs: i32,
    wx: i32, wy: i32, wz: i32,
    target_density: f32,
    host_material: Material,
    dirty_set: &mut HashSet<(i32, i32, i32)>,
    written: &mut Vec<WrittenCell>,
    changed_count: &mut u32,
) {
    write_all_locations(store, cs, wx, wy, wz, |_cur_d, cur_m| {
        let mat = if target_density > 0.0 {
            host_material
        } else if target_density <= 0.0 {
            Material::Air
        } else {
            cur_m
        };
        Some((target_density, mat))
    }, dirty_set, written, changed_count);
}

// ── Formation removal pass ────────────────────────────────────────────────

/// Read density at a world cell. Returns 1.0 (deep solid) for unloaded cells
/// to be conservative — we never want to carve into unknown territory.
#[inline]
fn read_density(
    fields: &std::collections::HashMap<(i32, i32, i32), voxel_core::density::DensityField>,
    cs: i32,
    wx: i32, wy: i32, wz: i32,
) -> f32 {
    let cx = wx.div_euclid(cs);
    let cy = wy.div_euclid(cs);
    let cz = wz.div_euclid(cs);
    let lx = wx.rem_euclid(cs) as usize;
    let ly = wy.rem_euclid(cs) as usize;
    let lz = wz.rem_euclid(cs) as usize;
    fields.get(&(cx, cy, cz)).map(|df| df.get(lx, ly, lz).density).unwrap_or(1.0)
}

/// Count how many of the 6 face-adjacent neighbors are air (density ≤ 0).
#[inline]
fn count_air_face_neighbors(
    fields: &std::collections::HashMap<(i32, i32, i32), voxel_core::density::DensityField>,
    cs: i32,
    wx: i32, wy: i32, wz: i32,
) -> u8 {
    let mut n = 0u8;
    for (dx, dy, dz) in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)] {
        if read_density(fields, cs, wx + dx, wy + dy, wz + dz) <= 0.0 {
            n += 1;
        }
    }
    n
}

/// Identify and carve thin formation features in an extended cylinder around
/// the building. Iterates `FORMATION_MAX_ITERATIONS` times so that chunky
/// formations erode layer by layer:
///   - Iteration 1: cells with ≥4 air neighbors carve (thin tips/edges)
///   - Iteration 2: cells exposed by iter 1's carving now qualify and carve
///   - Iteration 3: same again
///
/// Within a single iteration, we snapshot victims FIRST then carve, so
/// carving doesn't cascade into walls within one pass. Across iterations,
/// the eroded boundary creeps inward — but a true wall has so many solid
/// neighbors that it never crosses the threshold even after multiple passes.
///
/// MUST be called BEFORE the main raise/carve passes so the first iteration
/// classifies cells against the natural cave state.
fn formation_removal_pass(
    store: &mut ChunkStore,
    cs: i32,
    base: glam::IVec3,
    base_y_float: f32,
    terrace_size: i32,
    apron_radius: i32,
    dirty_set: &mut HashSet<(i32, i32, i32)>,
    written: &mut Vec<WrittenCell>,
    changed_count: &mut u32,
) -> u32 {
    let base_y_int = base_y_float.floor() as i32;
    let formation_radius = apron_radius + FORMATION_EXTRA_RADIUS;
    let formation_radius_f = formation_radius as f32;
    let interior_max = terrace_size - 1;
    let mut total_carved = 0u32;

    for _iter in 0..FORMATION_MAX_ITERATIONS {
        let mut victims: Vec<(i32, i32, i32)> = Vec::new();
        for dx in -formation_radius..=(terrace_size + formation_radius - 1) {
            for dz in -formation_radius..=(terrace_size + formation_radius - 1) {
                let wx = base.x + dx;
                let wz = base.z + dz;
                let dx_out = 0.max(-dx).max(dx - interior_max) as f32;
                let dz_out = 0.max(-dz).max(dz - interior_max) as f32;
                let edge_dist = (dx_out * dx_out + dz_out * dz_out).sqrt();
                if edge_dist > formation_radius_f { continue; }

                for y_off in -FORMATION_SCAN_BELOW..=FORMATION_MAX_ABOVE {
                    let wy = base_y_int + y_off;
                    let d = read_density(&store.density_fields, cs, wx, wy, wz);
                    if d <= 0.0 { continue; }
                    let air_neighbors = count_air_face_neighbors(
                        &store.density_fields, cs, wx, wy, wz,
                    );
                    if air_neighbors >= FORMATION_AIR_NEIGHBORS {
                        victims.push((wx, wy, wz));
                    }
                }
            }
        }
        if victims.is_empty() {
            break; // converged — nothing left to carve
        }
        for (wx, wy, wz) in victims {
            let before = *changed_count;
            write_lower(store, cs, wx, wy, wz, -1.0, dirty_set, written, changed_count);
            if *changed_count > before {
                total_carved += 1;
            }
        }
    }
    total_carved
}

// ── Public entry ──────────────────────────────────────────────────────────

pub fn flatten_terrace_sdf(
    store: &mut ChunkStore,
    base: glam::IVec3,
    base_y_float: f32,
    host_material: Material,
    config: &GenerationConfig,
    world_scale: f32,
    terrace_size: i32,
    clearance_voxels: i32,
) -> Vec<((i32, i32, i32), ConvertedMesh)> {
    let cs = config.chunk_size as i32;
    let clear = clearance_voxels.max(2);
    let apron_radius = apron_radius_for(terrace_size);
    let apron_radius_f = apron_radius as f32;

    // Build buttress hull for cantilever columns (Phase 2 / 2C).
    let hull = build_support_hull(store, cs, base, base_y_float, terrace_size);

    let mut dirty_set: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut written: Vec<WrittenCell> = Vec::new();
    let mut changed_count = 0u32;

    // ★ Formation removal pass (BEFORE main raise/carve so air-neighbor
    // classification uses the natural cave state). Erases stalactites,
    // stalagmites, columns, flowstone tips inside an extended cylinder
    // around the building so they don't leave mangled stubs after carving.
    let formations_carved = formation_removal_pass(
        store, cs, base, base_y_float, terrace_size, apron_radius,
        &mut dirty_set, &mut written, &mut changed_count,
    );

    let extent = apron_radius;
    let interior_max = terrace_size - 1;

    for dx in -extent..(terrace_size + extent) {
        for dz in -extent..(terrace_size + extent) {
            let wx = base.x + dx;
            let wz = base.z + dz;

            let dx_out = 0.max(-dx).max(dx - interior_max) as f32;
            let dz_out = 0.max(-dz).max(dz - interior_max) as f32;
            let edge_dist = (dx_out * dx_out + dz_out * dz_out).sqrt();
            let in_interior = edge_dist <= 0.0;
            if !in_interior && edge_dist > apron_radius_f { continue; }

            let target_y_float = match resolve_target_y(
                store, cs, base, base_y_float, apron_radius_f, config,
                &hull, wx, wz, edge_dist, in_interior,
            ) {
                Some(y) => y,
                None => continue, // wall preservation / no support
            };

            let (target_y, d_solid, d_air) = subvoxel_boundary_densities(target_y_float);

            // Fill solid below the boundary (raise-only — don't carve into
            // existing solids that are already higher density).
            for y in (target_y - FILL_DEPTH)..target_y {
                write_raise(store, cs, wx, y, wz, 1.0,
                    host_material, &mut dirty_set, &mut written, &mut changed_count);
            }

            // ── Boundary cells: FORCE-WRITE the sub-voxel densities. ──
            // Without force-write, conditional raise/lower will skip when the
            // existing density is already saturated (1.0 or -1.0), leaving the
            // iso surface at +0.5 voxels instead of at base_y_float. This is
            // the "building floats 20 UU above the floor" bug.
            write_force(store, cs, wx, target_y, wz, d_solid,
                host_material, &mut dirty_set, &mut written, &mut changed_count);
            write_force(store, cs, wx, target_y + 1, wz, d_air,
                host_material, &mut dirty_set, &mut written, &mut changed_count);

            // Clearance volume above the boundary: lower-only (don't disturb
            // existing solids that aren't actually in the way).
            for y in (target_y + 2)..=(target_y + clear) {
                write_lower(store, cs, wx, y, wz, -1.0,
                    &mut dirty_set, &mut written, &mut changed_count);
            }

            if in_interior {
                store.terraced_cells.insert((wx, base.y, wz));
                store.terraced_columns.insert((wx, wz), base.y);
            }
        }
    }

    // Diagnostic dump to file — eprintln to stderr isn't captured when the
    // DLL is loaded by UE, so we write directly to a file UE can read.
    let cx_diag = base.x + terrace_size / 2;
    let cz_diag = base.z + terrace_size / 2;
    let center_y = base_y_float.floor() as i32;
    let read = |y: i32| -> f32 {
        let cx = cx_diag.div_euclid(cs);
        let cy = y.div_euclid(cs);
        let cz = cz_diag.div_euclid(cs);
        let lx = cx_diag.rem_euclid(cs) as usize;
        let ly = y.rem_euclid(cs) as usize;
        let lz = cz_diag.rem_euclid(cs) as usize;
        store.density_fields.get(&(cx, cy, cz))
            .map(|df| df.get(lx, ly, lz).density).unwrap_or(0.0)
    };
    let d_below = read(center_y);
    let d_above = read(center_y + 1);
    // Linear iso interpolation: surface at center_y + d_below/(d_below - d_above)
    // when d_below >= 0 (solid or zero) and d_above < 0 (air).
    let iso_y = if d_below >= 0.0 && d_above < 0.0 {
        let denom = (d_below - d_above).max(1e-6);
        center_y as f32 + d_below / denom
    } else { f32::NAN };
    let log_line = format!(
        "[flatten_sdf] base=({},{},{}) y_float={:.4} size={} (+{}apron) clearance={} cones={} formations_carved={} written={} cells changed={} voxels dirty={} chunks | center_col(wx={},wz={}): y{}={:.4} y{}={:.4} iso_y={:.4} (UE={:.2}) | base_y_float_UE={:.2}\n",
        base.x, base.y, base.z, base_y_float, terrace_size, apron_radius, clear,
        hull.cones.len(), formations_carved, written.len(), changed_count, dirty_set.len(),
        cx_diag, cz_diag,
        center_y, d_below, center_y + 1, d_above, iso_y, iso_y * world_scale,
        base_y_float * world_scale);
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true)
        .open("D:/Unreal Projects/Mithril2026/Saved/flatten_sdf_log.txt")
    {
        let _ = f.write_all(log_line.as_bytes());
    }

    let chunk_size = config.chunk_size;
    let mut dirty_chunks: Vec<_> = dirty_set
        .into_iter()
        .map(|key| (key, 0usize, 0usize, 0usize, chunk_size, chunk_size, chunk_size))
        .collect();

    let extra_dirty = sync_boundary_density(
        &mut store.density_fields, &dirty_chunks, config.chunk_size,
    );
    dirty_chunks.extend(extra_dirty);

    // Post-sync restoration. sync_boundary_density uses min() at chunk seams
    // which can pull our written values in either direction depending on what
    // the neighbor chunk had. Restore EVERY written cell to its intended
    // density+material so the iso surface lands exactly where we computed it
    // even at seam boundaries.
    for &(key, lx, ly, lz, intended_density, intended_material) in &written {
        if let Some(df) = store.density_fields.get_mut(&key) {
            let s = df.get_mut(lx, ly, lz);
            if (s.density - intended_density).abs() > 1e-3 {
                s.density = intended_density;
                s.material = intended_material;
            }
        }
    }

    let dirty_keys: Vec<_> = dirty_chunks.iter().map(|&(k, ..)| k).collect();
    store.modification_tracker.mark_dirty_many(&dirty_keys);

    store.remesh_dirty(&dirty_chunks, config, world_scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::density::DensityField;

    #[test]
    fn apron_radius_min_is_3() {
        assert_eq!(apron_radius_for(1), 3);
        assert_eq!(apron_radius_for(4), 3);
        assert!(apron_radius_for(10) >= 5);
    }

    #[test]
    fn subvoxel_at_half_voxel_uses_classic_densities() {
        // frac = 0.5 should give d_solid=1, d_air=-1 (legacy behavior).
        let (y, ds, da) = subvoxel_boundary_densities(10.5);
        assert_eq!(y, 10);
        assert!((ds - 1.0).abs() < 1e-3);
        assert!((da + 1.0).abs() < 1e-3);
    }

    #[test]
    fn subvoxel_at_quarter_voxel_positions_iso_correctly() {
        // frac = 0.25 → d_solid = 0.25/0.75 ≈ 0.333, d_air = -1.
        let (y, ds, da) = subvoxel_boundary_densities(10.25);
        assert_eq!(y, 10);
        let iso = y as f32 + ds / (ds - da);
        assert!((iso - 10.25).abs() < 0.01,
            "iso should be near 10.25, got {} (ds={}, da={})", iso, ds, da);
    }

    #[test]
    fn subvoxel_at_three_quarter_voxel_positions_iso_correctly() {
        // frac = 0.75 → d_solid = 1, d_air = -0.25/0.75 ≈ -0.333.
        let (y, ds, da) = subvoxel_boundary_densities(10.75);
        assert_eq!(y, 10);
        let iso = y as f32 + ds / (ds - da);
        assert!((iso - 10.75).abs() < 0.01,
            "iso should be near 10.75, got {} (ds={}, da={})", iso, ds, da);
    }

    fn make_flat_ground(ground_y: i32, chunks: i32) -> ChunkStore {
        let cs: usize = 16;
        let mut store = ChunkStore::new(cs as i32);
        for cx in -chunks..=chunks {
            for cz in -chunks..=chunks {
                for cy in -chunks..=chunks {
                    let mut df = DensityField::new(cs + 1);
                    for z in 0..=cs {
                        for y in 0..=cs {
                            for x in 0..=cs {
                                let wy = cy * cs as i32 + y as i32;
                                let s = df.get_mut(x, y, z);
                                if wy < ground_y {
                                    s.density = 1.0;
                                    s.material = Material::Granite;
                                } else {
                                    s.density = -1.0;
                                    s.material = Material::Air;
                                }
                            }
                        }
                    }
                    store.density_fields.insert((cx, cy, cz), df);
                }
            }
        }
        store
    }

    /// Sub-voxel surface placement: ask for iso at y=10.3, get iso at ±0.1.
    #[test]
    fn subvoxel_surface_lands_near_requested_y() {
        let mut store = make_flat_ground(10, 1);
        let cfg = GenerationConfig::default();

        let base = glam::IVec3::new(0, 10, 0);
        let base_y_float = 10.3;
        let _ = flatten_terrace_sdf(
            &mut store, base, base_y_float,
            Material::Granite, &cfg, 40.0, 4, 3,
        );

        let cs = cfg.chunk_size as i32;
        let center_x = base.x + 2;
        let center_z = base.z + 2;
        let sample_density = |y: i32| -> f32 {
            let cx = center_x.div_euclid(cs);
            let cy = y.div_euclid(cs);
            let cz = center_z.div_euclid(cs);
            let lx = center_x.rem_euclid(cs) as usize;
            let ly = y.rem_euclid(cs) as usize;
            let lz = center_z.rem_euclid(cs) as usize;
            store.density_fields.get(&(cx, cy, cz))
                .map(|df| df.get(lx, ly, lz).density)
                .unwrap_or(1.0)
        };
        let mut crossing_y: Option<f32> = None;
        for y in 5..15 {
            let d_lo = sample_density(y);
            let d_hi = sample_density(y + 1);
            if d_lo > 0.0 && d_hi <= 0.0 {
                let t = d_lo / (d_lo - d_hi);
                crossing_y = Some(y as f32 + t);
                break;
            }
        }
        let crossing = crossing_y.expect("iso surface should exist in column");
        assert!((crossing - base_y_float).abs() < 0.05,
            "iso surface should land within 0.05 of {}, got {}", base_y_float, crossing);
    }

    /// Force-write fixes the "frac<0.5 boundary stays at 1.0" bug.
    /// With existing density saturated at 1.0, boundary write must overwrite.
    #[test]
    fn boundary_force_writes_low_frac_density() {
        let mut store = make_flat_ground(10, 1);
        let cfg = GenerationConfig::default();

        // Place at y=10.2 (frac = 0.2). Existing rock density at y=10 is +1.0.
        // The boundary at target_y=10 should be FORCED to d_solid = 0.2/0.8 = 0.25.
        let base = glam::IVec3::new(0, 10, 0);
        let _ = flatten_terrace_sdf(
            &mut store, base, 10.2,
            Material::Granite, &cfg, 40.0, 4, 3,
        );

        let cs = cfg.chunk_size as i32;
        let cx = base.x.div_euclid(cs);
        let cy = 10i32.div_euclid(cs);
        let cz = base.z.div_euclid(cs);
        let lx = base.x.rem_euclid(cs) as usize;
        let ly = 10i32.rem_euclid(cs) as usize;
        let lz = base.z.rem_euclid(cs) as usize;
        let d10 = store.density_fields.get(&(cx, cy, cz))
            .map(|df| df.get(lx, ly, lz).density).unwrap_or(1.0);
        // For frac=0.2, d_solid = 0.2/0.8 = 0.25.
        assert!((d10 - 0.25).abs() < 0.01,
            "force-write should set boundary to ~0.25 even though rock was 1.0, got {}", d10);
    }

    /// Cantilever over a cliff: building extends past edge, ravine below
    /// stays untouched (no fill all the way down).
    #[test]
    fn cantilever_does_not_fill_ravine() {
        let cs: usize = 16;
        let mut store = ChunkStore::new(cs as i32);
        // Ground only where wx <= 8. Right side is a 30+ voxel ravine.
        for cx in -1..=1 {
            for cz in -1..=1 {
                for cy in -2..=1 {
                    let mut df = DensityField::new(cs + 1);
                    for z in 0..=cs {
                        for y in 0..=cs {
                            for x in 0..=cs {
                                let wx = cx * cs as i32 + x as i32;
                                let wy = cy * cs as i32 + y as i32;
                                let s = df.get_mut(x, y, z);
                                if wy < 10 && wx <= 8 {
                                    s.density = 1.0;
                                    s.material = Material::Granite;
                                } else {
                                    s.density = -1.0;
                                    s.material = Material::Air;
                                }
                            }
                        }
                    }
                    store.density_fields.insert((cx, cy, cz), df);
                }
            }
        }
        let cfg = GenerationConfig::default();
        let base = glam::IVec3::new(8, 10, 0);
        let _ = flatten_terrace_sdf(
            &mut store, base, 10.0,
            Material::Granite, &cfg, 40.0, 4, 3,
        );

        let cs_i = cfg.chunk_size as i32;
        let sample_at = |wx: i32, wy: i32, wz: i32| -> f32 {
            let cx = wx.div_euclid(cs_i);
            let cy = wy.div_euclid(cs_i);
            let cz = wz.div_euclid(cs_i);
            let lx = wx.rem_euclid(cs_i) as usize;
            let ly = wy.rem_euclid(cs_i) as usize;
            let lz = wz.rem_euclid(cs_i) as usize;
            store.density_fields.get(&(cx, cy, cz))
                .map(|df| df.get(lx, ly, lz).density)
                .unwrap_or(1.0)
        };
        // Deep below the cantilever, in the ravine: must remain air.
        let deep = sample_at(11, -5, 1);
        assert!(deep < 0.0,
            "deep ravine should remain air, got density={}", deep);
    }
}
