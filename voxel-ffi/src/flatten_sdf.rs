//! All-in single-placement building flatten.
//!
//! Algorithm: explicit per-column target-Y ramp with sub-voxel boundary
//! density (3C, ±0 voxel surface alignment), plus a convex-hull buttress
//! made of capped-cone SDFs for cantilever columns (2C, fills cliffs to
//! nearest natural rock without filling ravines).
//!
//! Most density-field operations are factored into `voxel_core::density_ops`
//! so they're shared with the collapse rubble pile placement.

use std::collections::HashSet;

use glam::Vec3;
use voxel_core::density::DensityField;
use voxel_core::density_ops::{
    self, FormationRemovalConfig, WrittenCell,
};
use voxel_core::material::Material;
use voxel_gen::config::GenerationConfig;

use crate::sdf::{compile_cone, find_support_rays, sdf_compiled_cone, CompiledCone};
use crate::store::{sync_boundary_density, ChunkStore};
use crate::types::ConvertedMesh;

// ── Tunables ───────────────────────────────────────────────────────────────

const APRON_FRAC: f32 = 0.50;          // 50% of footprint each side = 200% total
const APRON_MIN: i32 = 3;              // never less than 3 voxels of apron
const FILL_DEPTH: i32 = 6;             // depth of solid support below floor
const SUPPORT_CHECK_DEPTH: i32 = 4;    // column with no solid here = cantilever
const SUPPORT_RAY_COUNT: usize = 16;
const SUPPORT_RAY_UP_TOL: f32 = 0.05;
const SUPPORT_RAYS_PER_COL: usize = 3; // top-N closest hits per cantilever col
const BUTTRESS_R_BASE: f32 = 1.5;
const BUTTRESS_R_TIP: f32 = 0.8;
// Ramp noise disabled — was causing per-column jitter. If you re-enable it,
// reinstate the simplex sample inside `ramp_y_noise` AND hoist `Simplex3D::new`
// out of the per-column path.
const ADJACENT_Y_CAP: f32 = 1.0;       // max apron Y delta per voxel (1:1 slope)
const FLAT_MATCH_THRESHOLD: f32 = 1.0; // skip ramp if existing surface within this

// ── Carve rim smoothing ───────────────────────────────────────────────────
//
// The clearance pass leaves a knife-sharp edge where the carve boundary
// meets standing rock — a machine-cut bite in organic cave. A small
// Laplacian pass over a band straddling that boundary lets the cut read as
// an eroded alcove instead. Same idea as `mining::smooth_mine_boundary`,
// which has done this for mine walls for a long time.
//
// ⚠️ Deliberately NOT the pad. The floor's iso crossing lives between
// target_y and target_y+1, and the sub-voxel alignment that lands it exactly
// under the building is the thing belt drags took several rounds to get
// right. The band starts two voxels above pad height so those cells are
// never candidates.
//
// ⚠️ Cells the carve CLEARED are clamped to stay air. Blending is otherwise
// free to grow rock back into the clearance volume — lips over the floor,
// slivers in the space the player and the building occupy. Smoothing may
// round rock away; it may never put it back.
// ⚠️ Measured 2026-08-09: selecting this band GEOMETRICALLY — a fixed
// annulus around the footprint — smoothed 1761 cells to blend a cut that
// removed 643, and 776 against 127 on a small building. Most of the writes
// landed on untouched cave that the carve never touched. Worse, a blur
// applied inside a hard-edged region just relocates the hard edge to that
// region's boundary, leaving a ring seam where smoothing stops.
//
// So the band is now seeded from the columns that ACTUALLY removed rock,
// grown by RIM_SMOOTH_RADIUS, and the strength falls off to zero across
// that growth. No cut, no smoothing, and no discontinuity at the edge.
const RIM_SMOOTH_ITERATIONS: u32 = 2;
const RIM_SMOOTH_STRENGTH: f32 = 0.5;
const RIM_SMOOTH_RADIUS: i32 = 2;   // falloff distance from a cutting column

/// Destination for the per-placement carve diagnostic, or None to write
/// nothing. Read once from `MITHRIL_FLATTEN_LOG`.
///
/// This used to be `#[cfg(debug_assertions)]` with a path hardcoded to the
/// MAIN checkout. Both halves were wrong for how the project is actually
/// driven: release DLLs are what get shipped to the editor for visual
/// checks, so the diagnostic was dead exactly when it was needed, and every
/// parallel worktree would have logged over the same file.
fn flatten_log_path() -> Option<&'static str> {
    static LOG_PATH: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    LOG_PATH
        .get_or_init(|| std::env::var("MITHRIL_FLATTEN_LOG").ok().filter(|s| !s.is_empty()))
        .as_deref()
}

#[inline]
fn apron_radius_for(terrace_size: i32) -> i32 {
    ((terrace_size as f32) * APRON_FRAC).round().max(APRON_MIN as f32) as i32
}

#[inline]
fn cap_distance_for(terrace_size: i32) -> f32 {
    (apron_radius_for(terrace_size) as f32) * 2.0
}

// ── Support hull (cantilever buttress) ────────────────────────────────────

#[derive(Default)]
struct SupportHull {
    // Cones are stored in compiled form so the per-column sphere-trace below
    // skips the per-call sqrt for axis length and the sqrt+div for the
    // slanted-side normal. Each cone is sampled ~14 Y-steps × ~16-30 apron
    // columns per placement, so this saves real cycles on the hot path.
    cones: Vec<CompiledCone>,
}

impl SupportHull {
    fn cone_top_in_column(&self, wx: f32, wz: f32, search_lo: f32, search_hi: f32) -> Option<f32> {
        if self.cones.is_empty() { return None; }
        // Sphere trace: when we're outside every cone, the SDF tells us how
        // far we can step down before potentially entering one. Falls back to
        // a 0.25 floor so we don't stall in flat regions and never overshoot
        // the cone top by more than that floor (acceptable for buttress Y).
        let min_step = 0.25_f32;
        let mut y = search_hi;
        while y >= search_lo {
            let p = Vec3::new(wx, y, wz);
            let mut min_d = f32::INFINITY;
            for c in &self.cones {
                let d = sdf_compiled_cone(p, c);
                if d < min_d {
                    min_d = d;
                    if min_d <= 0.0 { return Some(y); }
                }
            }
            // min_d > 0 here (we didn't return). Step down by at least min_step,
            // but jump further when we're far from any cone surface.
            y -= min_d.max(min_step);
        }
        None
    }
}

fn build_support_hull(
    store: &ChunkStore,
    cs: i32,
    base: glam::IVec3,
    base_y_float: f32,
    terrace_size: i32,
) -> SupportHull {
    let cap_dist = cap_distance_for(terrace_size);
    let base_y_int = base_y_float.floor() as i32;
    let mut cones: Vec<CompiledCone> = Vec::new();

    for dx in 0..terrace_size {
        for dz in 0..terrace_size {
            let wx = base.x + dx;
            let wz = base.z + dz;

            let mut has_support = false;
            for k in 1..=SUPPORT_CHECK_DEPTH {
                let y = base_y_int - k;
                if density_ops::read_density(&store.density_fields, cs, wx, y, wz) > 0.0 {
                    has_support = true;
                    break;
                }
            }
            if has_support { continue; }

            let origin = Vec3::new(wx as f32 + 0.5, base_y_float, wz as f32 + 0.5);
            let hits = find_support_rays(
                &store.density_fields, cs, origin, cap_dist,
                SUPPORT_RAY_COUNT, SUPPORT_RAY_UP_TOL,
            );
            for hit in hits.iter().take(SUPPORT_RAYS_PER_COL) {
                if let Some(c) = compile_cone(origin, hit.hit_pos, BUTTRESS_R_BASE, BUTTRESS_R_TIP) {
                    cones.push(c);
                }
            }
        }
    }
    SupportHull { cones }
}

// ── Per-column ramp Y resolution ──────────────────────────────────────────

#[inline]
fn ramp_y_noise(_cfg: &GenerationConfig, _wx: i32, _wz: i32) -> f32 {
    // RAMP_NOISE_AMP is 0.0 (disabled — was causing per-column jitter).
    // Short-circuit so we don't pay for a per-column Simplex3D::new()
    // (ChaCha8Rng seed + 256-element shuffle + 512-byte perm table) inside
    // the apron loop. Re-enable the noise body if RAMP_NOISE_AMP ever
    // becomes nonzero — and at that point, hoist the Simplex3D out of the
    // per-column path.
    0.0
}

#[inline]
fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn resolve_target_y(
    fields: &std::collections::HashMap<(i32, i32, i32), DensityField>,
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
    if in_interior {
        return Some(base_y_float);
    }

    if let Some(nat_y_iso) = density_ops::natural_floor_y_iso(fields, cs, wx, base.y, wz, 4, 24) {
        let nat_clamped = nat_y_iso.min(base_y_float);
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

    let cone_search_lo = base_y_float - cap_distance_for(8);
    let cone_search_hi = base_y_float + 1.0;
    if let Some(top_y) = hull.cone_top_in_column(
        wx as f32 + 0.5, wz as f32 + 0.5, cone_search_lo, cone_search_hi,
    ) {
        return Some(top_y.min(base_y_float));
    }

    None
}

// ── Carve rim smoothing ───────────────────────────────────────────────────

/// Laplacian pass over the band straddling the carve boundary. Returns the
/// number of cell writes made.
///
/// Works in WORLD space via `read_density` rather than per-chunk like
/// `smooth_mine_boundary`. Chunk fields overlap by a cell, so a per-chunk
/// blur gives each copy of a shared boundary cell a different answer — its
/// neighbourhood is clipped differently in each — and the seam shows. Going
/// through `read_density`/`write_all_locations` means one value is computed
/// per world cell and fanned out to every copy.
#[allow(clippy::too_many_arguments)]
fn smooth_carve_rim(
    fields: &mut std::collections::HashMap<(i32, i32, i32), DensityField>,
    cs: i32,
    cut_columns: &[(i32, i32, i32)],
    carved_columns: &std::collections::HashSet<(i32, i32)>,
    clear: i32,
    host_material: Material,
    dirty_set: &mut HashSet<(i32, i32, i32)>,
    written: &mut Vec<WrittenCell>,
    changed_count: &mut u32,
) -> u32 {
    if RIM_SMOOTH_ITERATIONS == 0 || RIM_SMOOTH_STRENGTH <= 0.0 || cut_columns.is_empty() {
        return 0;
    }

    // Grow the cutting columns outward, weight falling to zero at the edge.
    // Keeping the STRONGEST weight and the LOWEST floor per column means
    // overlapping cut columns reinforce rather than fight.
    let mut band: std::collections::HashMap<(i32, i32), (i32, f32)> =
        std::collections::HashMap::new();
    for &(wx, wz, target_y) in cut_columns {
        for dx in -RIM_SMOOTH_RADIUS..=RIM_SMOOTH_RADIUS {
            for dz in -RIM_SMOOTH_RADIUS..=RIM_SMOOTH_RADIUS {
                let d = dx.abs().max(dz.abs());
                let w = 1.0 - (d as f32) / (RIM_SMOOTH_RADIUS as f32 + 1.0);
                let e = band.entry((wx + dx, wz + dz)).or_insert((target_y, 0.0));
                e.0 = e.0.min(target_y);
                if w > e.1 { e.1 = w; }
            }
        }
    }

    // (wx, wy, wz, cleared_by_the_carve, weight)
    let mut cells: Vec<(i32, i32, i32, bool, f32)> = Vec::new();
    for (&(wx, wz), &(target_y, w)) in &band {
        let cleared_col = carved_columns.contains(&(wx, wz));
        let top = target_y + clear;
        for y in (target_y + 2)..=top {
            // Taper the top two cells as well, so the band does not end in a
            // horizontal step the way its lateral edge used to.
            let vy = ((top - y + 1) as f32 / 3.0).min(1.0);
            cells.push((wx, y, wz, cleared_col, w * vy));
        }
    }

    const NEIGHBOURS: [(i32, i32, i32); 6] = [
        (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1),
    ];

    let mut writes = 0u32;
    for _ in 0..RIM_SMOOTH_ITERATIONS {
        // Double-buffered: every new value is computed against the field as
        // it stood at the start of the sweep, so the result does not depend
        // on the order cells happen to be visited in.
        let mut updates: Vec<(i32, i32, i32, f32)> = Vec::with_capacity(cells.len());
        for &(wx, wy, wz, cleared, weight) in &cells {
            let s = RIM_SMOOTH_STRENGTH * weight;
            if s <= 0.0 { continue; }
            let cur = density_ops::read_density(fields, cs, wx, wy, wz);
            let mut sum = 0.0f32;
            for (nx, ny, nz) in NEIGHBOURS {
                sum += density_ops::read_density(fields, cs, wx + nx, wy + ny, wz + nz);
            }
            let avg = sum / NEIGHBOURS.len() as f32;
            let mut new_d = cur * (1.0 - s) + avg * s;
            // Never re-solidify what the carve cleared.
            if cleared { new_d = new_d.min(-1e-3); }
            if (new_d - cur).abs() > 1e-3 {
                updates.push((wx, wy, wz, new_d));
            }
        }
        if updates.is_empty() { break; }
        for (wx, wy, wz, new_d) in updates {
            density_ops::write_all_locations(fields, cs, wx, wy, wz, |_cur_d, cur_m| {
                // Density moves; material only follows when the sign flips.
                // Repainting eroding natural rock with host_material would
                // stripe the cave with the building's stone.
                let new_m = if new_d >= 0.0 {
                    if cur_m.is_solid() { cur_m } else { host_material }
                } else {
                    Material::Air
                };
                Some((new_d, new_m))
            }, dirty_set, written, changed_count);
            writes += 1;
        }
    }
    writes
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
    let dirty_chunks = flatten_terrace_sdf_carve(
        store, base, base_y_float, host_material, config, world_scale,
        terrace_size, clearance_voxels,
    );
    store.remesh_dirty(&dirty_chunks, config, world_scale)
}

/// Carve-only variant: performs the full flatten (formation removal, SDF
/// carve, boundary sync, written-cell restore, dirty tracking) but does NOT
/// remesh. Returns the dirty chunk list (full-chunk bounds) for the caller
/// to feed to `ChunkStore::remesh_dirty`.
///
/// Exists for batch placements (belt drags): adjacent buildings share
/// chunks, so remeshing inside each per-building call redoes the full
/// hermite-extract + DC-solve + smooth + convert pipeline for the same
/// chunk once per building — with every result but the last overwritten.
/// Carving all buildings first and remeshing the union once produces
/// bit-identical meshes (densities are fully persistent after each carve;
/// meshing is pure derived output of the final density state).
pub fn flatten_terrace_sdf_carve(
    store: &mut ChunkStore,
    base: glam::IVec3,
    base_y_float: f32,
    host_material: Material,
    config: &GenerationConfig,
    world_scale: f32,
    terrace_size: i32,
    clearance_voxels: i32,
) -> Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> {
    let cs = config.chunk_size as i32;
    let clear = clearance_voxels.max(2);
    let apron_radius = apron_radius_for(terrace_size);
    let apron_radius_f = apron_radius as f32;

    let hull = build_support_hull(store, cs, base, base_y_float, terrace_size);

    let mut dirty_set: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut written: Vec<WrittenCell> = Vec::new();
    let mut changed_count = 0u32;

    // Formation removal pass — iterative thin-feature erosion in a cylinder
    // around the building. Carves stalactites/stalagmites/columns/flowstone
    // tips so they don't leave mangled stubs after the main carve.
    let formation_cfg = FormationRemovalConfig {
        anchor_x: base.x + terrace_size / 2,
        anchor_z: base.z + terrace_size / 2,
        footprint_x: terrace_size / 2,
        footprint_z: terrace_size / 2,
        radius_extra: 4,
        anchor_y: base_y_float.floor() as i32,
        max_above: 12,
        scan_below: 4,
        air_neighbors_threshold: 4,
        max_iterations: 3,
    };
    let formations_carved = density_ops::formation_removal_pass(
        &mut store.density_fields, cs, &formation_cfg,
        &mut dirty_set, &mut written, &mut changed_count,
    );

    let extent = apron_radius;
    let interior_max = terrace_size - 1;

    // How many columns the carve touches at all.
    let mut region_cols = 0u32;
    // Columns whose clearance pass actually removed SOLID ROCK, i.e. columns
    // that genuinely produced cut face. `region_cols` only counts columns
    // admitted to the carve — a column admitted over open air writes nothing
    // and shows nothing, so admitted-vs-round cannot tell us whether the
    // squared corners are cutting a wall or waving at thin air.
    let mut cut_cols = 0u32;
    let mut cut_cells = 0u32;
    // Seeds for the rim smoothing: the columns that genuinely produced cut
    // face, and every column the carve touched (so smoothing knows which
    // cells it must not re-solidify).
    let mut cut_columns: Vec<(i32, i32, i32)> = Vec::new();
    let mut carved_columns: std::collections::HashSet<(i32, i32)> =
        std::collections::HashSet::new();

    for dx in -extent..(terrace_size + extent) {
        for dz in -extent..(terrace_size + extent) {
            let wx = base.x + dx;
            let wz = base.z + dz;

            let dx_out = 0.max(-dx).max(dx - interior_max) as f32;
            let dz_out = 0.max(-dz).max(dz - interior_max) as f32;
            let edge_dist = (dx_out * dx_out + dz_out * dz_out).sqrt();
            let in_interior = edge_dist <= 0.0;
            if !in_interior && edge_dist > apron_radius_f { continue; }
            region_cols += 1;

            let target_y_float = match resolve_target_y(
                &store.density_fields, cs, base, base_y_float, apron_radius_f, config,
                &hull, wx, wz, edge_dist, in_interior,
            ) {
                Some(y) => y,
                None => continue,
            };

            let (target_y, d_solid, d_air) = density_ops::subvoxel_boundary_densities(target_y_float);

            for y in (target_y - FILL_DEPTH)..target_y {
                density_ops::write_raise(&mut store.density_fields, cs, wx, y, wz, 1.0,
                    host_material, &mut dirty_set, &mut written, &mut changed_count);
            }

            density_ops::write_force(&mut store.density_fields, cs, wx, target_y, wz, d_solid,
                host_material, &mut dirty_set, &mut written, &mut changed_count);
            density_ops::write_force(&mut store.density_fields, cs, wx, target_y + 1, wz, d_air,
                host_material, &mut dirty_set, &mut written, &mut changed_count);

            // write_lower is conditional, so changed_count only moves when
            // there was solid rock here to take away. Snapshotting it around
            // the clearance pass is what separates "cut a wall" from "swept
            // through air that was already empty".
            let changed_before_clearance = changed_count;
            for y in (target_y + 2)..=(target_y + clear) {
                density_ops::write_lower(&mut store.density_fields, cs, wx, y, wz, -1.0,
                    &mut dirty_set, &mut written, &mut changed_count);
            }
            let cut_here = changed_count - changed_before_clearance;
            carved_columns.insert((wx, wz));
            if cut_here > 0 {
                cut_cols += 1;
                cut_cells += cut_here;
                cut_columns.push((wx, wz, target_y));
            }

            if in_interior {
                store.terraced_cells.insert((wx, base.y, wz));
                store.terraced_columns.insert((wx, wz), base.y);
            }
        }
    }

    // Rim smoothing runs BEFORE sync_boundary_density on purpose. Its writes
    // land in `written` after the carve's, and restore_written_cells replays
    // that list in order — so the smoothed value is what survives the seam
    // merge. Smoothing after the restore instead would need a second sync,
    // whose min() merge would then start undoing the carve.
    let rim_writes = smooth_carve_rim(
        &mut store.density_fields, cs, &cut_columns, &carved_columns, clear,
        host_material, &mut dirty_set, &mut written, &mut changed_count,
    );

    // Diagnostic dump (file-based — eprintln isn't visible in UE).
    // Runtime-gated on MITHRIL_FLATTEN_LOG; see flatten_log_path().
    if let Some(log_path) = flatten_log_path() {
        let cx_diag = base.x + terrace_size / 2;
        let cz_diag = base.z + terrace_size / 2;
        let center_y = base_y_float.floor() as i32;
        let d_below = density_ops::read_density(&store.density_fields, cs, cx_diag, center_y, cz_diag);
        let d_above = density_ops::read_density(&store.density_fields, cs, cx_diag, center_y + 1, cz_diag);
        let iso_y = if d_below >= 0.0 && d_above < 0.0 {
            let denom = (d_below - d_above).max(1e-6);
            center_y as f32 + d_below / denom
        } else { f32::NAN };
        let log_line = format!(
            "[flatten_sdf] base=({},{},{}) y_float={:.4} size={} (+{}apron) clearance={} cones={} formations_carved={} written={} cells changed={} voxels dirty={} chunks | center_col(wx={},wz={}): y{}={:.4} y{}={:.4} iso_y={:.4} (UE={:.2}) | base_y_float_UE={:.2}\n\
             [flatten_sdf.cut] apron_radius={} cols={} | cols that removed rock={} ({} cells) | carve spans {}x{} voxels = {:.0}x{:.0} UU | cut face height <= {} voxels = {:.0} UU\n\
             [flatten_sdf.rim] smoothing iters={} strength={:.2} radius={} | seeded from {} cutting cols | cell writes={}\n",
            base.x, base.y, base.z, base_y_float, terrace_size, apron_radius, clear,
            hull.cones.len(), formations_carved, written.len(), changed_count, dirty_set.len(),
            cx_diag, cz_diag,
            center_y, d_below, center_y + 1, d_above, iso_y, iso_y * world_scale,
            base_y_float * world_scale,
            apron_radius, region_cols, cut_cols, cut_cells,
            terrace_size + 2 * apron_radius, terrace_size + 2 * apron_radius,
            (terrace_size + 2 * apron_radius) as f32 * world_scale,
            (terrace_size + 2 * apron_radius) as f32 * world_scale,
            clear - 1, (clear - 1) as f32 * world_scale,
            RIM_SMOOTH_ITERATIONS, RIM_SMOOTH_STRENGTH, RIM_SMOOTH_RADIUS, cut_cols,
            rim_writes);
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(log_path)
        {
            let _ = f.write_all(log_line.as_bytes());
        }
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

    density_ops::restore_written_cells(&mut store.density_fields, &written);

    let dirty_keys: Vec<_> = dirty_chunks.iter().map(|&(k, ..)| k).collect();
    store.modification_tracker.mark_dirty_many(&dirty_keys);

    dirty_chunks
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

    /// Flat floor with a solid wall rising from `wall_x` upward in +X — the
    /// geometry the rim smoothing actually has to deal with, and the case
    /// `make_flat_ground` cannot produce.
    fn make_ground_with_wall(ground_y: i32, wall_x: i32, wall_top_y: i32, chunks: i32) -> ChunkStore {
        let cs: usize = 16;
        let mut store = ChunkStore::new(cs as i32);
        for cx in -chunks..=chunks {
            for cz in -chunks..=chunks {
                for cy in -chunks..=chunks {
                    let mut df = DensityField::new(cs + 1);
                    for z in 0..=cs {
                        for y in 0..=cs {
                            for x in 0..=cs {
                                let wx = cx * cs as i32 + x as i32;
                                let wy = cy * cs as i32 + y as i32;
                                let solid = wy < ground_y || (wx >= wall_x && wy < wall_top_y);
                                let s = df.get_mut(x, y, z);
                                if solid {
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

    /// THE safety invariant for rim smoothing. Blending a carve boundary
    /// into the rock beside it will happily pull solid back across the
    /// boundary — which would leave lips hanging over the pad and slivers in
    /// the volume the building and the player stand in. Cleared cells are
    /// clamped to air for exactly this reason; if that clamp is ever
    /// loosened, this fails.
    #[test]
    fn rim_smoothing_never_puts_rock_back_in_the_clearance() {
        let ground_y = 10;
        let clearance = 4;
        // Wall starts 6 voxels along +X, so the carve's +X rim cuts into it.
        let mut store = make_ground_with_wall(ground_y, 6, 20, 1);
        let cfg = GenerationConfig::default();
        let base = glam::IVec3::new(0, ground_y, 0);

        let _ = flatten_terrace_sdf(
            &mut store, base, ground_y as f32,
            Material::Granite, &cfg, 30.0, 4, clearance,
        );

        let cs = cfg.chunk_size as i32;
        let apron = apron_radius_for(4);
        let mut solid_in_clearance = Vec::new();
        for dx in -apron..(4 + apron) {
            for dz in -apron..(4 + apron) {
                let dx_out = 0.max(-dx).max(dx - 3) as f32;
                let dz_out = 0.max(-dz).max(dz - 3) as f32;
                if (dx_out * dx_out + dz_out * dz_out).sqrt() > apron as f32 { continue; }
                // Columns buried in solid wall get no natural floor, so
                // resolve_target_y returns None and the carve SKIPS them —
                // they are legitimately still rock and are not part of the
                // cleared volume. Detect that by the floor cell: a carved
                // column has air directly above pad height.
                let above_pad = density_ops::read_density(
                    &store.density_fields, cs, base.x + dx, ground_y + 1, base.z + dz);
                if above_pad > 0.0 { continue; }
                for y in (ground_y + 2)..=(ground_y + clearance) {
                    let d = density_ops::read_density(
                        &store.density_fields, cs, base.x + dx, y, base.z + dz);
                    if d > 0.0 {
                        solid_in_clearance.push((base.x + dx, y, base.z + dz, d));
                    }
                }
            }
        }
        assert!(solid_in_clearance.is_empty(),
            "rim smoothing pulled rock back into the cleared volume at {:?}",
            &solid_in_clearance[..solid_in_clearance.len().min(6)]);
    }

    /// The rim pass must not disturb the pad. Its band starts two voxels up
    /// for this reason — the floor's iso crossing sits between target_y and
    /// target_y+1, and moving it puts buildings through the ground.
    #[test]
    fn rim_smoothing_leaves_the_pad_iso_alone() {
        let ground_y = 10;
        let mut store = make_ground_with_wall(ground_y, 6, 20, 1);
        let cfg = GenerationConfig::default();
        let base = glam::IVec3::new(0, ground_y, 0);
        let base_y_float = ground_y as f32;

        let _ = flatten_terrace_sdf(
            &mut store, base, base_y_float,
            Material::Granite, &cfg, 30.0, 4, 4,
        );

        let cs = cfg.chunk_size as i32;
        // Sample a pad column on the wall side, where smoothing is closest.
        let (px, pz) = (base.x + 3, base.z + 2);
        let mut crossing = None;
        for y in 5..18 {
            let lo = density_ops::read_density(&store.density_fields, cs, px, y, pz);
            let hi = density_ops::read_density(&store.density_fields, cs, px, y + 1, pz);
            if lo > 0.0 && hi <= 0.0 {
                crossing = Some(y as f32 + lo / (lo - hi));
                break;
            }
        }
        let crossing = crossing.expect("pad column should still have an iso surface");
        assert!((crossing - base_y_float).abs() < 0.05,
            "pad iso moved to {} (wanted {}) — the rim band is reaching the floor",
            crossing, base_y_float);
    }

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
            density_ops::read_density(&store.density_fields, cs, center_x, y, center_z)
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

    #[test]
    fn boundary_force_writes_low_frac_density() {
        let mut store = make_flat_ground(10, 1);
        let cfg = GenerationConfig::default();
        let base = glam::IVec3::new(0, 10, 0);
        let _ = flatten_terrace_sdf(
            &mut store, base, 10.2,
            Material::Granite, &cfg, 40.0, 4, 3,
        );
        let cs = cfg.chunk_size as i32;
        let d10 = density_ops::read_density(&store.density_fields, cs, base.x, 10, base.z);
        // For frac=0.2, d_solid = 0.2/0.8 = 0.25.
        assert!((d10 - 0.25).abs() < 0.01,
            "force-write should set boundary to ~0.25 even though rock was 1.0, got {}", d10);
    }

    /// Run a simulated belt-drag batch two ways and return
    /// (chunk remesh count, final store) for each:
    ///   old — full flatten (carve + remesh) per belt, as the batch handler
    ///         did before the carve/remesh split;
    ///   new — carve every belt, then ONE remesh of the deduped union, as
    ///         `handle_building_flatten_batch` does now.
    fn run_belt_drag_both_ways(
        n_belts: i32, ground_chunks: i32,
    ) -> ((usize, ChunkStore), (usize, ChunkStore)) {
        let cfg = GenerationConfig::default();
        let ws = 40.0;
        // Adjacent 2-voxel belts along +X at sub-voxel height, like a drag.
        let belts: Vec<glam::IVec3> =
            (0..n_belts).map(|i| glam::IVec3::new(i * 2, 10, 3)).collect();

        let mut store_old = make_flat_ground(10, ground_chunks);
        let mut old_remeshes = 0usize;
        for &base in &belts {
            old_remeshes += flatten_terrace_sdf(
                &mut store_old, base, 10.3, Material::Granite, &cfg, ws, 2, 3,
            ).len();
        }

        let mut store_new = make_flat_ground(10, ground_chunks);
        let mut dirty = Vec::new();
        for &base in &belts {
            dirty.extend(flatten_terrace_sdf_carve(
                &mut store_new, base, 10.3, Material::Granite, &cfg, ws, 2, 3,
            ));
        }
        dirty.sort_by_key(|&(k, ..)| k);
        dirty.dedup_by_key(|&mut (k, ..)| k);
        let new_remeshes = store_new.remesh_dirty(&dirty, &cfg, ws).len();

        ((old_remeshes, store_old), (new_remeshes, store_new))
    }

    /// The batch handler's carve-all-then-remesh-once ordering must produce
    /// bit-identical base meshes to the old remesh-inside-every-flatten
    /// ordering (densities persist per carve; meshing is pure output).
    #[test]
    fn batch_single_remesh_is_bit_identical_to_per_belt_remesh() {
        let ((old_n, store_old), (new_n, store_new)) = run_belt_drag_both_ways(12, 1);
        assert!(new_n < old_n,
            "deferred remesh should mesh fewer chunks ({} old vs {} new)", old_n, new_n);

        assert_eq!(store_old.base_meshes.len(), store_new.base_meshes.len(),
            "both orderings must mesh the same chunk set");
        for (key, old_mesh) in &store_old.base_meshes {
            let new_mesh = store_new.base_meshes.get(key)
                .unwrap_or_else(|| panic!("chunk {:?} missing from deferred-remesh store", key));
            assert_eq!(old_mesh.vertices.len(), new_mesh.vertices.len(),
                "vertex count mismatch in chunk {:?}", key);
            assert_eq!(old_mesh.triangles.len(), new_mesh.triangles.len(),
                "triangle count mismatch in chunk {:?}", key);
            for (a, b) in old_mesh.vertices.iter().zip(new_mesh.vertices.iter()) {
                assert_eq!(a.position.to_array().map(f32::to_bits),
                           b.position.to_array().map(f32::to_bits),
                           "vertex position differs in chunk {:?}", key);
                assert_eq!(a.normal.to_array().map(f32::to_bits),
                           b.normal.to_array().map(f32::to_bits),
                           "vertex normal differs in chunk {:?}", key);
                assert_eq!(a.material, b.material,
                    "vertex material differs in chunk {:?}", key);
            }
            for (a, b) in old_mesh.triangles.iter().zip(new_mesh.triangles.iter()) {
                assert_eq!(a.indices, b.indices, "triangle differs in chunk {:?}", key);
            }
        }
    }

    /// Wall-time A/B of the two orderings. Run with:
    ///   cargo test --release -p voxel-ffi bench_batch_flatten -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_batch_flatten_deferred_remesh() {
        // Warmup pass (rayon pool spin-up, allocator, caches).
        let t0 = std::time::Instant::now();
        let _ = run_belt_drag_both_ways(20, 2);

        // Time each ordering in isolation (setup excluded).
        let cfg = GenerationConfig::default();
        let ws = 40.0;
        let belts: Vec<glam::IVec3> =
            (0..20).map(|i| glam::IVec3::new(i * 2, 10, 3)).collect();

        let mut store_old = make_flat_ground(10, 2);
        let t_old = std::time::Instant::now();
        let mut old_remeshes = 0usize;
        for &base in &belts {
            old_remeshes += flatten_terrace_sdf(
                &mut store_old, base, 10.3, Material::Granite, &cfg, ws, 2, 3,
            ).len();
        }
        let old_ms = t_old.elapsed().as_secs_f64() * 1e3;

        let mut store_new = make_flat_ground(10, 2);
        let t_new = std::time::Instant::now();
        let mut dirty = Vec::new();
        for &base in &belts {
            dirty.extend(flatten_terrace_sdf_carve(
                &mut store_new, base, 10.3, Material::Granite, &cfg, ws, 2, 3,
            ));
        }
        dirty.sort_by_key(|&(k, ..)| k);
        dirty.dedup_by_key(|&mut (k, ..)| k);
        let new_remeshes = store_new.remesh_dirty(&dirty, &cfg, ws).len();
        let new_ms = t_new.elapsed().as_secs_f64() * 1e3;

        println!(
            "bench_batch_flatten (20-belt drag): old {:.2} ms / {} chunk remeshes, \
             new {:.2} ms / {} chunk remeshes ({:.0}% less wall time, total incl. carve) \
             [warmup {:.0} ms]",
            old_ms, old_remeshes, new_ms, new_remeshes,
            (1.0 - new_ms / old_ms) * 100.0,
            t0.elapsed().as_secs_f64() * 1e3,
        );
        assert!(new_remeshes < old_remeshes);
    }
}
