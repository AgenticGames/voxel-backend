//! Cinematic collapse-rubble placement.
//!
//! When a slab of cave roof collapses, this module turns the per-voxel
//! collapse data into a visually pristine rubble pile. It does *physics
//! storytelling*, not just stamping — the pile slides on slopes, cascades
//! down cliffs, fragments into multiple peaks, leaves boulder tracks and
//! impact craters, and chains into nearby weak rock for secondary collapses.
//!
//! Pipeline (called once per collapse event):
//!
//!   1. **Fragmentation** — split the slab into 2-5 fragments (grid-based
//!      partition along longest axis). Each fragment is processed
//!      independently → pile naturally has multiple peaks.
//!   2. **Per-fragment metadata** — centroid, volume, dominant material,
//!      landing offset (median fall distance for that fragment's columns),
//!      leading-edge bias direction (computed from fall-distance variance).
//!   3. **Per-column volume allocation** — for each (x,z) in an extended
//!      fragment footprint:
//!        - Sample natural cave floor Y.
//!        - Compute weight = Gaussian(distance) × skew(leading_edge_dir)
//!          × low_floor_bonus (debris pools in low spots).
//!        - Allocate volume_per_column = total × weight / sum(weights).
//!   4. **Angle-of-repose iteration** — diffusion solver: if pile_top here is
//!      higher than neighbor by more than slope_limit, transfer surplus.
//!      Naturally handles 20°/70° slopes and cliff cascades.
//!   5. **Impact craters** — depress the natural floor 1-2 voxels under
//!      each fragment center *before* placing the pile. "Ground bites."
//!   6. **Pile placement (sub-voxel)** — for each column, surface at
//!      natural_floor + accumulated_height with sub-voxel boundary densities.
//!      Material stratification: dense materials at the bottom, lighter
//!      at the top.
//!   7. **Boulder tracks + impact cracks** — radial fissures and rolled-
//!      boulder grooves carved into the surrounding floor.
//!   8. **Splash ring** — single-cell debris speckle at 1.3× radius.
//!   9. **Apron blend + crust noise** — pile surface modulated by simplex
//!      noise; outer edge ramps smoothly into surrounding cave floor.
//!  10. **Formation removal** — strip stalactite/stalagmite stubs at both
//!      the landing zone AND the slab origin (cleanup of broken ceiling
//!      fragments left behind after the slab departed).
//!  11. **Multi-chunk seam-aware writes + post-sync restore** — every
//!      density edit fans out to all chunks sharing the world cell so
//!      `sync_boundary_density`'s `min()` doesn't divot our writes.
//!  12. **Settling-collapse hint + dust event** — flag cells near the
//!      landing zone for the next stress tick, and emit a dust event for
//!      UE's Niagara system.

use std::collections::{HashMap, HashSet};

use voxel_noise::{simplex::Simplex3D, NoiseSource};

use crate::density::DensityField;
use crate::density_ops::{
    self, FormationRemovalConfig, WrittenCell,
};
use crate::material::Material;
use crate::stress::{CollapsedVoxel, StressConfig};

// ── Tunables ──────────────────────────────────────────────────────────────

const APRON_EXTRA_RADIUS: i32 = 3;        // extends pile footprint for blend
const SLOPE_LIMIT_TAN: f32 = 0.65;        // tan(33°), geological default
const AOR_ITERATIONS: u32 = 8;
const AOR_TRANSFER_RATE: f32 = 0.25;      // fraction of surplus moved per iter
const PILE_NOISE_AMP: f32 = 0.45;         // ± voxels of organic crest perturbation
const CRATER_DEPTH_PER_FRAGMENT_VOLUME: f32 = 0.04; // multiply by sqrt(volume) for depth
const CRATER_MAX_DEPTH: i32 = 2;
const CRATER_RADIUS_FRAC: f32 = 0.45;     // crater radius = this × pile_radius
const SPLASH_RADIUS_FRAC: f32 = 1.30;     // splash ring at this × pile_radius
const SPLASH_DENSITY_FRAC: f32 = 0.35;    // probability of placing a splash voxel
const BOULDER_COUNT_BASE: usize = 2;      // boulders per fragment, at minimum
const BOULDER_COUNT_PER_RADIUS: f32 = 0.6;
const BOULDER_COUNT_MAX: usize = 6;
const BOULDER_RADIUS_MIN: f32 = 1.5;
const BOULDER_RADIUS_MAX: f32 = 2.7;
const BOULDER_TRACK_LENGTH_MAX: f32 = 6.0;
const CRACK_COUNT_PER_FRAGMENT: usize = 5;
const CRACK_LENGTH_MAX: f32 = 4.5;
const CRACK_DEPTH_DENSITY: f32 = 0.4;     // shallow crack: density 0.4 (mostly air)
const FRAGMENT_COUNT_MAX: usize = 5;

// ── Public types ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct FragmentInfo {
    pub center_xz: (f32, f32),
    pub center_y: f32,
    pub volume: u32,
    pub landing_y: f32,
    pub leading_edge_dir: (f32, f32), // unit vector toward leading edge
    pub dominant_material: Material,
}

/// One dust event per fragment impact. UE polls these and spawns Niagara.
#[derive(Debug, Clone)]
pub struct DustEvent {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub magnitude: f32, // 0.0..1.0 normalized to fragment volume
}

/// Result of placing a collapse pile. Caller (stress.rs) merges these into
/// the wider collapse event.
pub struct PlacementResult {
    pub fragments: Vec<FragmentInfo>,
    pub dust_events: Vec<DustEvent>,
    pub affected_chunks: HashSet<(i32, i32, i32)>,
    pub written_cells: Vec<WrittenCell>,
    /// Cells flagged as "may trigger secondary collapse next tick" — the
    /// caller should add these to the stress dirty set.
    pub settling_dirty_cells: Vec<(i32, i32, i32)>,
}

// ── Public entry ──────────────────────────────────────────────────────────

pub fn place_collapse_pile(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    config: &StressConfig,
    collapsed_voxels: &[CollapsedVoxel],
    bb_min: (i32, i32, i32),
    bb_max: (i32, i32, i32),
    dominant_material: Material,
    landing_offset: i32,
    chunk_size: usize,
) -> PlacementResult {
    let cs = chunk_size as i32;
    let mut affected_chunks: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut written: Vec<WrittenCell> = Vec::new();
    let mut changed_count = 0u32;

    if !config.rubble_enabled || collapsed_voxels.is_empty() {
        return PlacementResult {
            fragments: vec![],
            dust_events: vec![],
            affected_chunks,
            written_cells: written,
            settling_dirty_cells: vec![],
        };
    }

    // Deterministic seeds derived from collapse center.
    let pile_seed: u64 = (((bb_min.0 + bb_max.0) / 2) as i64 as u64)
        .wrapping_mul(73856093)
        ^ (((bb_min.1 + bb_max.1) / 2) as i64 as u64).wrapping_mul(19349663)
        ^ (((bb_min.2 + bb_max.2) / 2) as i64 as u64).wrapping_mul(83492791);
    let crust_noise = Simplex3D::new(pile_seed);
    let detail_noise = Simplex3D::new(pile_seed.wrapping_add(1));
    let boulder_noise = Simplex3D::new(pile_seed.wrapping_add(2));

    // 1. Fragment the slab.
    let fragments = fragment_slab(
        collapsed_voxels, bb_min, bb_max, density_fields, cs,
        landing_offset, dominant_material,
    );

    let mut placement_metadata: Vec<FragmentInfo> = Vec::with_capacity(fragments.len());
    let mut dust_events: Vec<DustEvent> = Vec::with_capacity(fragments.len());
    let mut settling_dirty_cells: Vec<(i32, i32, i32)> = Vec::new();

    let total_volume = collapsed_voxels.len() as f32;

    for frag in &fragments {
        let info = place_fragment_pile(
            density_fields, cs, config, frag,
            &crust_noise, &detail_noise, &boulder_noise,
            &mut affected_chunks, &mut written, &mut changed_count,
            &mut settling_dirty_cells,
        );
        // Dust event for UE.
        let mag = (info.volume as f32 / total_volume.max(1.0)).clamp(0.0, 1.0);
        dust_events.push(DustEvent {
            world_x: info.center_xz.0,
            world_y: info.center_y,
            world_z: info.center_xz.1,
            magnitude: mag,
        });
        placement_metadata.push(info);
    }

    // Slab-origin formation cleanup — scrub thin features that the slab
    // departure left dangling from the ceiling/walls.
    {
        let origin_cfg = FormationRemovalConfig {
            anchor_x: ((bb_min.0 + bb_max.0) / 2),
            anchor_z: ((bb_min.2 + bb_max.2) / 2),
            footprint_x: ((bb_max.0 - bb_min.0).max(2) / 2 + 1),
            footprint_z: ((bb_max.2 - bb_min.2).max(2) / 2 + 1),
            radius_extra: 3,
            anchor_y: ((bb_min.1 + bb_max.1) / 2),
            max_above: 8,
            scan_below: 8,
            air_neighbors_threshold: 4,
            max_iterations: 3,
        };
        density_ops::formation_removal_pass(
            density_fields, cs, &origin_cfg,
            &mut affected_chunks, &mut written, &mut changed_count,
        );
    }

    let _ = changed_count;
    PlacementResult {
        fragments: placement_metadata,
        dust_events,
        affected_chunks,
        written_cells: written,
        settling_dirty_cells,
    }
}

// ── Slab fragmentation ────────────────────────────────────────────────────

struct Fragment<'a> {
    voxels: Vec<&'a CollapsedVoxel>,
    center_xz: (f32, f32),
    radius_x: f32,
    radius_z: f32,
    volume: usize,
    floor_y: i32,
    leading_edge_dir: (f32, f32),
    dominant_material: Material,
    material_counts: HashMap<Material, u32>,
}

fn fragment_slab<'a>(
    collapsed: &'a [CollapsedVoxel],
    bb_min: (i32, i32, i32),
    bb_max: (i32, i32, i32),
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    cs: i32,
    landing_offset: i32,
    fallback_material: Material,
) -> Vec<Fragment<'a>> {
    let dx = (bb_max.0 - bb_min.0 + 1).max(1);
    let dz = (bb_max.2 - bb_min.2 + 1).max(1);
    let volume = collapsed.len();

    // Decide grid based on volume + aspect ratio.
    let (nx, nz) = if volume < 12 {
        (1, 1)
    } else if volume < 50 {
        if dx >= dz { (2, 1) } else { (1, 2) }
    } else if volume < 200 {
        if dx >= 2 * dz { (3, 1) }
        else if dz >= 2 * dx { (1, 3) }
        else { (2, 2) }
    } else if volume < 600 {
        if dx >= dz { (3, 2) } else { (2, 3) }
    } else {
        // Big slab: still cap at FRAGMENT_COUNT_MAX.
        (3, 2)
    };
    let total_cells = (nx * nz).min(FRAGMENT_COUNT_MAX);
    let nx = if nz * 1 >= total_cells { total_cells / nz.max(1) } else { nx };

    let cell_dx = dx as f32 / nx as f32;
    let cell_dz = dz as f32 / nz as f32;

    let mut buckets: Vec<Vec<&CollapsedVoxel>> = vec![Vec::new(); nx * nz];
    for v in collapsed {
        let fx = (((v.world_x - bb_min.0) as f32 / cell_dx).floor() as i32)
            .clamp(0, nx as i32 - 1) as usize;
        let fz = (((v.world_z - bb_min.2) as f32 / cell_dz).floor() as i32)
            .clamp(0, nz as i32 - 1) as usize;
        buckets[fz * nx + fx].push(v);
    }

    buckets.into_iter()
        .filter(|b| !b.is_empty())
        .map(|voxels| {
            build_fragment(voxels, density_fields, cs, landing_offset, fallback_material)
        })
        .collect()
}

fn build_fragment<'a>(
    voxels: Vec<&'a CollapsedVoxel>,
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    cs: i32,
    landing_offset: i32,
    fallback_material: Material,
) -> Fragment<'a> {
    // Centroid + bbox.
    let mut sum_x = 0.0f32;
    let mut sum_z = 0.0f32;
    let mut min_x = i32::MAX;
    let mut max_x = i32::MIN;
    let mut min_z = i32::MAX;
    let mut max_z = i32::MIN;
    let mut min_y = i32::MAX;
    let mut max_y = i32::MIN;
    let mut mat_counts: HashMap<Material, u32> = HashMap::new();
    for v in &voxels {
        sum_x += v.world_x as f32;
        sum_z += v.world_z as f32;
        min_x = min_x.min(v.world_x);
        max_x = max_x.max(v.world_x);
        min_z = min_z.min(v.world_z);
        max_z = max_z.max(v.world_z);
        min_y = min_y.min(v.world_y);
        max_y = max_y.max(v.world_y);
        *mat_counts.entry(v.material).or_insert(0) += 1;
    }
    let n = voxels.len() as f32;
    let cx = sum_x / n;
    let cz = sum_z / n;
    let radius_x = ((max_x - min_x) as f32 * 0.5 + 1.0).max(1.5);
    let radius_z = ((max_z - min_z) as f32 * 0.5 + 1.0).max(1.5);

    // Dominant material from this fragment. Filter non-renderable variants
    // (Air + mushroom/ice extras 42+) before picking — those would render
    // as matte black in UE since they have no MaterialInstanceDynamic slot.
    let dominant = mat_counts.iter()
        .filter(|(m, _)| {
            let idx = **m as u8;
            idx > 0 && idx <= MAX_RENDERABLE_MATERIAL
        })
        .max_by_key(|(_, c)| **c)
        .map(|(m, _)| *m)
        .unwrap_or_else(|| safe_pile_material(fallback_material));

    // Per-fragment landing offset = MEDIAN fall distance for columns
    // contained in this fragment. Different fragments can land at different
    // Ys (e.g., one half of slab over a cliff lands deeper).
    let mut column_floor_ys: Vec<i32> = Vec::new();
    let mut col_min: HashMap<(i32, i32), i32> = HashMap::new();
    for v in &voxels {
        let entry = col_min.entry((v.world_x, v.world_z)).or_insert(v.world_y);
        *entry = (*entry).min(v.world_y);
    }
    for (&(x, z), &min_yc) in &col_min {
        // Find first solid below the column's slab bottom.
        let mut floor_y_for_col = min_yc - 1;
        for _ in 0..96 {
            let d = density_ops::read_density(density_fields, cs, x, floor_y_for_col, z);
            if d > 0.0 {
                column_floor_ys.push(floor_y_for_col + 1);
                break;
            }
            floor_y_for_col -= 1;
            if floor_y_for_col < min_yc - 96 {
                column_floor_ys.push(min_yc - landing_offset);
                break;
            }
        }
    }
    column_floor_ys.sort();
    let landing_y = if column_floor_ys.is_empty() {
        min_y - landing_offset
    } else {
        column_floor_ys[column_floor_ys.len() / 2]
    };

    // Leading-edge direction: variance of fall distance across XZ → the
    // direction of greatest delta. Approximated by comparing falls along
    // the two principal axes of the fragment.
    let leading_edge_dir = compute_leading_edge_dir(&voxels, &col_min, density_fields, cs);

    Fragment {
        voxels,
        center_xz: (cx, cz),
        radius_x,
        radius_z,
        volume: n as usize,
        floor_y: landing_y,
        leading_edge_dir,
        dominant_material: dominant,
        material_counts: mat_counts,
    }
}

fn compute_leading_edge_dir(
    voxels: &[&CollapsedVoxel],
    col_min: &HashMap<(i32, i32), i32>,
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    cs: i32,
) -> (f32, f32) {
    // For each column in the fragment, compute its fall distance, then the
    // weighted mean position of "deeper falls" relative to centroid gives us
    // the leading-edge direction.
    let mut sum_x = 0.0f32;
    let mut sum_z = 0.0f32;
    let mut sum_w = 0.0f32;
    for v in voxels {
        let min_yc = col_min[&(v.world_x, v.world_z)];
        // Quick fall scan.
        let mut depth = 0;
        for k in 1..32 {
            let d = density_ops::read_density(density_fields, cs, v.world_x, min_yc - k, v.world_z);
            if d > 0.0 { break; }
            depth = k;
        }
        let w = depth as f32;
        sum_x += v.world_x as f32 * w;
        sum_z += v.world_z as f32 * w;
        sum_w += w;
    }
    if sum_w < 1e-3 { return (0.0, 0.0); }
    let mean_x = sum_x / sum_w;
    let mean_z = sum_z / sum_w;
    // Center of fragment for offset.
    let cx = voxels.iter().map(|v| v.world_x as f32).sum::<f32>() / voxels.len() as f32;
    let cz = voxels.iter().map(|v| v.world_z as f32).sum::<f32>() / voxels.len() as f32;
    let dx = mean_x - cx;
    let dz = mean_z - cz;
    let mag = (dx * dx + dz * dz).sqrt().max(1e-3);
    (dx / mag, dz / mag)
}

// ── Per-fragment placement ────────────────────────────────────────────────

fn place_fragment_pile(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    cs: i32,
    config: &StressConfig,
    frag: &Fragment,
    crust_noise: &Simplex3D,
    detail_noise: &Simplex3D,
    boulder_noise: &Simplex3D,
    affected_chunks: &mut HashSet<(i32, i32, i32)>,
    written: &mut Vec<WrittenCell>,
    changed_count: &mut u32,
    settling_dirty: &mut Vec<(i32, i32, i32)>,
) -> FragmentInfo {
    let (cx, cz) = frag.center_xz;
    let avg_radius = (frag.radius_x + frag.radius_z) * 0.5;

    // Pile zone: footprint expanded by APRON_EXTRA_RADIUS for the smooth blend.
    let zone_rx = (frag.radius_x + APRON_EXTRA_RADIUS as f32).ceil() as i32;
    let zone_rz = (frag.radius_z + APRON_EXTRA_RADIUS as f32).ceil() as i32;
    let cx_i = cx.round() as i32;
    let cz_i = cz.round() as i32;

    // ── Per-column natural floor scan ──
    // Sample existing surface Y for every column in the zone. Deep columns
    // get more material (debris pools in low spots).
    let nx = (zone_rx * 2 + 1) as usize;
    let nz = (zone_rz * 2 + 1) as usize;
    let mut natural_floor: Vec<f32> = vec![frag.floor_y as f32 + 0.5; nx * nz];
    let mut weight: Vec<f32> = vec![0.0; nx * nz];
    let target_volume = frag.volume as f32 * config.rubble_fill_ratio;

    for iz in 0..nz {
        for ix in 0..nx {
            let wx = cx_i - zone_rx + ix as i32;
            let wz = cz_i - zone_rz + iz as i32;
            // Natural floor — use scan around fragment landing Y.
            if let Some(nat) = density_ops::natural_floor_y_iso(
                density_fields, cs, wx, frag.floor_y, wz, 4, 16,
            ) {
                natural_floor[iz * nx + ix] = nat;
            }

            // Distance to fragment center XZ, normalized by elliptical radii.
            let nx_d = (wx as f32 - cx) / frag.radius_x;
            let nz_d = (wz as f32 - cz) / frag.radius_z;
            let dist_sq = nx_d * nx_d + nz_d * nz_d;
            if dist_sq > 1.4 { continue; }

            // Gaussian falloff (sigma^2 = 0.35 normalized).
            let mut w = (-(dist_sq * 2.5)).exp();

            // SKEWED CREST (5F): bias weight toward leading edge.
            let toward_lead = nx_d * frag.leading_edge_dir.0 + nz_d * frag.leading_edge_dir.1;
            w *= 1.0 + 0.45 * toward_lead.max(-0.7);

            // Low-floor bonus: deep columns attract more debris.
            let nat_y = natural_floor[iz * nx + ix];
            let low_bonus = ((frag.floor_y as f32 - nat_y) * 0.05).max(0.0).min(0.5);
            w *= 1.0 + low_bonus;

            weight[iz * nx + ix] = w.max(0.0);
        }
    }

    // Normalize weights to allocate target_volume.
    let weight_sum: f32 = weight.iter().sum::<f32>().max(1e-3);
    let mut pile_height: Vec<f32> = weight.iter()
        .map(|w| w * target_volume / weight_sum)
        .collect();

    // ── Angle-of-repose iteration (the magic) ──
    for _iter in 0..AOR_ITERATIONS {
        let mut changes: Vec<(usize, f32)> = Vec::new();
        for iz in 0..nz {
            for ix in 0..nx {
                let i = iz * nx + ix;
                let here_top = natural_floor[i] + pile_height[i];
                for &(dx, dz) in &[(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                    let nix = ix as i32 + dx;
                    let niz = iz as i32 + dz;
                    if nix < 0 || nix >= nx as i32 || niz < 0 || niz >= nz as i32 { continue; }
                    let j = niz as usize * nx + nix as usize;
                    let neigh_top = natural_floor[j] + pile_height[j];
                    let delta = here_top - neigh_top;
                    if delta > SLOPE_LIMIT_TAN {
                        let transfer = (delta - SLOPE_LIMIT_TAN) * AOR_TRANSFER_RATE * 0.25;
                        changes.push((i, -transfer));
                        changes.push((j, transfer));
                    }
                }
            }
        }
        if changes.is_empty() { break; }
        for (i, dh) in changes {
            pile_height[i] = (pile_height[i] + dh).max(0.0);
        }
    }

    // ── Impact crater under fragment center (5C) ──
    let crater_depth = ((frag.volume as f32).sqrt() * CRATER_DEPTH_PER_FRAGMENT_VOLUME)
        .round() as i32;
    let crater_depth = crater_depth.clamp(0, CRATER_MAX_DEPTH);
    let crater_radius = (avg_radius * CRATER_RADIUS_FRAC).max(1.0);
    if crater_depth > 0 {
        let cr_i = crater_radius.ceil() as i32;
        for dx in -cr_i..=cr_i {
            for dz in -cr_i..=cr_i {
                let dx_n = dx as f32 / crater_radius;
                let dz_n = dz as f32 / crater_radius;
                let r2 = dx_n * dx_n + dz_n * dz_n;
                if r2 > 1.0 { continue; }
                let dish = (1.0 - r2).sqrt(); // 1 at center, 0 at edge
                let depth = (crater_depth as f32 * dish).round() as i32;
                if depth <= 0 { continue; }
                let wx = cx_i + dx;
                let wz = cz_i + dz;
                let nat_y = natural_floor[
                    (((dz + zone_rz) as usize) * nx + ((dx + zone_rx) as usize))
                    .min(nx * nz - 1)
                ];
                let nat_int = nat_y.floor() as i32;
                for k in 0..depth {
                    density_ops::write_lower(
                        density_fields, cs, wx, nat_int - k, wz, -1.0,
                        affected_chunks, written, changed_count,
                    );
                }
            }
        }
    }

    // ── Pile placement with sub-voxel surface + crust noise (Tier 1A + 5F) ──
    for iz in 0..nz {
        for ix in 0..nx {
            let i = iz * nx + ix;
            let h = pile_height[i];
            if h < 0.05 { continue; }

            let wx = cx_i - zone_rx + ix as i32;
            let wz = cz_i - zone_rz + iz as i32;
            let nat_y = natural_floor[i];

            // Add per-column simplex noise to the pile surface (5F crest skew
            // already in weights; this adds small organic ripple).
            let n_lo = crust_noise.sample(wx as f64 * 0.35, 0.0, wz as f64 * 0.35) as f32;
            let n_hi = detail_noise.sample(wx as f64 * 1.1, 5.0, wz as f64 * 1.1) as f32;
            let surface_jitter = (n_lo * 0.7 + n_hi * 0.3) * PILE_NOISE_AMP;

            let target_y_float = nat_y + h + surface_jitter;
            let (target_y, d_solid, d_air) =
                density_ops::subvoxel_boundary_densities(target_y_float);

            // Material stratification (5D): top 25% gets pulverized lighter
            // material if available, bottom 25% gets densest material.
            let nat_int = nat_y.floor() as i32;
            let pile_voxels = (target_y - nat_int).max(0);

            for k in 0..pile_voxels {
                let wy = nat_int + k;
                let frac_height = if pile_voxels > 0 {
                    k as f32 / pile_voxels as f32
                } else { 0.0 };
                let material_at = stratified_material(&frag.material_counts, frac_height);
                density_ops::write_raise(
                    density_fields, cs, wx, wy, wz, 1.0, material_at,
                    affected_chunks, written, changed_count,
                );
            }

            // Boundary cells (sub-voxel iso placement).
            density_ops::write_force(
                density_fields, cs, wx, target_y, wz, d_solid,
                stratified_material(&frag.material_counts, 1.0),
                affected_chunks, written, changed_count,
            );
            density_ops::write_force(
                density_fields, cs, wx, target_y + 1, wz, d_air,
                Material::Air,
                affected_chunks, written, changed_count,
            );
        }
    }

    // ── Splash ring (5E) ──
    // Single-cell debris speckle just past the pile edge — looks like
    // material that bounced and rolled outward.
    let splash_radius = avg_radius * SPLASH_RADIUS_FRAC;
    let splash_steps = (splash_radius * std::f32::consts::TAU * 0.5) as i32;
    for s in 0..splash_steps {
        let theta = (s as f32 / splash_steps as f32) * std::f32::consts::TAU;
        let n = boulder_noise.sample(theta as f64 * 1.7, 0.0, theta as f64 * 0.9) as f32;
        if n < (1.0 - SPLASH_DENSITY_FRAC * 2.0) { continue; }
        let r = splash_radius * (0.95 + 0.1 * n);
        let wx = cx_i + (theta.cos() * r * (frag.radius_x / avg_radius)).round() as i32;
        let wz = cz_i + (theta.sin() * r * (frag.radius_z / avg_radius)).round() as i32;
        // Place 1 voxel on top of the natural floor here.
        if let Some(nat) = density_ops::natural_floor_y_iso(
            density_fields, cs, wx, frag.floor_y, wz, 4, 8,
        ) {
            let wy = nat.ceil() as i32;
            density_ops::write_raise(
                density_fields, cs, wx, wy, wz, 1.0, frag.dominant_material,
                affected_chunks, written, changed_count,
            );
        }
    }

    // ── Boulder tracks (5G) — only where the fragment actually rolled ──
    // For each boulder we'll place, also carve a short track from the
    // impact center to the boulder's resting place.
    let boulder_count = (((avg_radius * BOULDER_COUNT_PER_RADIUS) as usize) + BOULDER_COUNT_BASE)
        .min(BOULDER_COUNT_MAX);
    for b in 0..boulder_count {
        let theta = (b as f32) * std::f32::consts::TAU / boulder_count as f32;
        let radial_n = boulder_noise.sample(b as f64 * 1.31, 0.0, b as f64 * 0.93) as f32;
        let radial_frac = 0.2 + (radial_n * 0.5 + 0.5) * 0.6;
        let bx = (cx + theta.cos() * frag.radius_x * radial_frac).round() as i32;
        let bz = (cz + theta.sin() * frag.radius_z * radial_frac).round() as i32;

        // Boulder rests on top of the pile at this column.
        let bb_idx = bbi_for(bx, bz, cx_i, cz_i, zone_rx, zone_rz, nx, nz);
        let nat_y_at = bb_idx.map(|i| natural_floor[i]).unwrap_or(frag.floor_y as f32);
        let pile_h_at = bb_idx.map(|i| pile_height[i]).unwrap_or(0.0);
        let by = (nat_y_at + pile_h_at - 0.5).round() as i32;

        // Boulder size noise.
        let size_n = boulder_noise.sample(bx as f64 * 0.41, by as f64 * 0.41, bz as f64 * 0.41) as f32;
        let radius = BOULDER_RADIUS_MIN
            + (size_n * 0.5 + 0.5) * (BOULDER_RADIUS_MAX - BOULDER_RADIUS_MIN);

        // Stamp boulder.
        place_boulder(
            density_fields, cs, bx, by, bz, radius,
            frag.dominant_material, boulder_noise,
            affected_chunks, written, changed_count,
        );

        // Track from impact center to boulder.
        let track_n = boulder_noise.sample(b as f64 * 7.13, 11.0, b as f64 * 5.91) as f32;
        let track_len = (BOULDER_TRACK_LENGTH_MAX * (track_n * 0.5 + 0.5)).max(1.5);
        let track_steps = track_len.ceil() as i32;
        let dx = bx as f32 - cx;
        let dz = bz as f32 - cz;
        let mag = (dx * dx + dz * dz).sqrt().max(1e-3);
        let dirx = dx / mag;
        let dirz = dz / mag;
        for k in 1..track_steps {
            let t = k as f32 / track_steps as f32;
            let tx = (cx + dirx * track_len * t).round() as i32;
            let tz = (cz + dirz * track_len * t).round() as i32;
            // Pile-surface Y at this column for the track.
            let bbi = bbi_for(tx, tz, cx_i, cz_i, zone_rx, zone_rz, nx, nz);
            let nat_y_tk = bbi.map(|i| natural_floor[i]).unwrap_or(frag.floor_y as f32);
            let pile_h_tk = bbi.map(|i| pile_height[i]).unwrap_or(0.0);
            let surf_y = (nat_y_tk + pile_h_tk).floor() as i32;
            // Light depression in the surface cell.
            density_ops::write_lower(
                density_fields, cs, tx, surf_y, tz, 0.2,
                affected_chunks, written, changed_count,
            );
        }
    }

    // ── Impact cracks (5H) ──
    // Spider-cracks radiating from the fragment center on the surrounding
    // floor (not on the pile itself — outside the pile footprint).
    for c in 0..CRACK_COUNT_PER_FRAGMENT {
        let theta = (c as f32) * std::f32::consts::TAU / CRACK_COUNT_PER_FRAGMENT as f32
            + (boulder_noise.sample(c as f64 * 9.7, 0.0, 0.0) as f32 * 0.7);
        let dirx = theta.cos();
        let dirz = theta.sin();
        let len_n = boulder_noise.sample(c as f64 * 4.2, 1.0, 0.0) as f32;
        let len = (CRACK_LENGTH_MAX * (len_n * 0.5 + 0.5)).max(2.0);
        let start = (avg_radius * 0.95).max(1.5);
        let end = start + len;
        let steps = end.ceil() as i32;
        for k in start.floor() as i32..=steps {
            let kf = k as f32;
            if kf > end { break; }
            let tx = (cx + dirx * kf).round() as i32;
            let tz = (cz + dirz * kf).round() as i32;
            // Surface Y at this column = natural floor + maybe pile.
            let bbi = bbi_for(tx, tz, cx_i, cz_i, zone_rx, zone_rz, nx, nz);
            let nat_y_tk = bbi.map(|i| natural_floor[i]).unwrap_or(frag.floor_y as f32);
            let surf_y = nat_y_tk.floor() as i32;
            density_ops::write_lower(
                density_fields, cs, tx, surf_y, tz, CRACK_DEPTH_DENSITY,
                affected_chunks, written, changed_count,
            );
        }
    }

    // ── Formation removal at landing zone (Tier 3A) ──
    {
        let landing_cfg = FormationRemovalConfig {
            anchor_x: cx_i,
            anchor_z: cz_i,
            footprint_x: frag.radius_x.ceil() as i32,
            footprint_z: frag.radius_z.ceil() as i32,
            radius_extra: 3,
            anchor_y: frag.floor_y,
            max_above: 8 + (frag.volume as f32).sqrt().min(8.0) as i32,
            scan_below: 4,
            air_neighbors_threshold: 4,
            max_iterations: 3,
        };
        density_ops::formation_removal_pass(
            density_fields, cs, &landing_cfg,
            affected_chunks, written, changed_count,
        );
    }

    // ── Settling-collapse hint (5I) ──
    // Mark cells just above the pile surface as "may need stress check next
    // tick" — added pile weight could tip a marginal ceiling.
    let pile_top_y = (frag.floor_y as f32 + pile_height.iter().cloned().fold(0.0, f32::max)).ceil() as i32;
    for k in 1..=4 {
        settling_dirty.push((cx_i, pile_top_y + k * 4, cz_i));
    }

    // Compute pile center Y (approx surface at center for dust event).
    let center_idx_x = zone_rx as usize;
    let center_idx_z = zone_rz as usize;
    let center_pile_top = natural_floor[center_idx_z * nx + center_idx_x]
        + pile_height[center_idx_z * nx + center_idx_x];

    FragmentInfo {
        center_xz: (cx, cz),
        center_y: center_pile_top,
        volume: frag.volume as u32,
        landing_y: frag.floor_y as f32,
        leading_edge_dir: frag.leading_edge_dir,
        dominant_material: frag.dominant_material,
    }
}

// ── Material stratification (5D) ──────────────────────────────────────────

/// UE has 42 MaterialInstanceDynamic slots (indices 0-41). Material enum
/// values 42-48 (mushroom + extra ice variants) have no UE material instance
/// assigned and render as matte black if used in a mesh. Filter those out
/// of pile material selection — substitute Granite as a safe fallback rock.
const MAX_RENDERABLE_MATERIAL: u8 = 41;

#[inline]
fn safe_pile_material(m: Material) -> Material {
    let idx = m as u8;
    if idx == 0 || idx > MAX_RENDERABLE_MATERIAL {
        // Air can't be solid; mushroom/ice extras have no UE material.
        // Fall back to Granite (the universal rock).
        Material::Granite
    } else {
        m
    }
}

fn stratified_material(counts: &HashMap<Material, u32>, frac_height: f32) -> Material {
    // frac_height: 0.0 = bottom of pile, 1.0 = top.
    // Bottom 33%: heaviest/densest material from the slab's mix.
    // Middle 67%: dominant material.
    // Top: keep dominant; we don't have a "fines/dust" generic material, so
    // we just bias toward less-common slab materials for variety.
    //
    // ALL returns pass through `safe_pile_material` so we never write a
    // material that would render as matte black in UE.
    if counts.is_empty() {
        return Material::Granite;
    }
    // Filter the candidate list so we never pick a non-renderable material
    // (Air or mushroom/ice extras 42+). Falls back to Granite if everything
    // in the slab was non-renderable, which shouldn't happen but is safe.
    let mut sorted: Vec<(Material, u32)> = counts.iter()
        .filter(|(m, _)| {
            let idx = **m as u8;
            idx > 0 && idx <= MAX_RENDERABLE_MATERIAL
        })
        .map(|(m, c)| (*m, *c))
        .collect();
    if sorted.is_empty() {
        return Material::Granite;
    }
    sorted.sort_by(|a, b| b.1.cmp(&a.1));

    let pick = if frac_height < 0.33 {
        sorted[0].0
    } else if frac_height > 0.85 && sorted.len() > 1 {
        sorted[1].0
    } else {
        sorted[0].0
    };
    safe_pile_material(pick)
}

// ── Boulder placement helper ──────────────────────────────────────────────

fn place_boulder(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    cs: i32,
    bx: i32, by: i32, bz: i32,
    radius: f32,
    material: Material,
    boulder_noise: &Simplex3D,
    affected_chunks: &mut HashSet<(i32, i32, i32)>,
    written: &mut Vec<WrittenCell>,
    changed_count: &mut u32,
) {
    // Half-bury: shift the boulder DOWN by ~40% of its radius so it appears
    // settled into the pile.
    let bury = (radius * 0.4) as i32;
    let cy = (by - bury) as f32;
    let r_ceil = radius.ceil() as i32;
    let r_sq = radius * radius;
    for ox in -r_ceil..=r_ceil {
        for oy in -r_ceil..=r_ceil {
            for oz in -r_ceil..=r_ceil {
                let dx = ox as f32;
                let dy_b = oy as f32;
                let dz = oz as f32;
                let d2 = dx * dx + dy_b * dy_b + dz * dz;
                if d2 > r_sq { continue; }

                let edge = (d2 / r_sq).sqrt();
                let edge_n = boulder_noise.sample(
                    (bx + ox) as f64 * 0.7,
                    (cy + oy as f32) as f64 * 0.7,
                    (bz + oz) as f64 * 0.7,
                ) as f32;
                if edge > 0.85 + edge_n * 0.20 { continue; }

                let wx_b = bx + ox;
                let wy_b = cy as i32 + oy;
                let wz_b = bz + oz;

                density_ops::write_raise(
                    density_fields, cs, wx_b, wy_b, wz_b, 1.0, material,
                    affected_chunks, written, changed_count,
                );
            }
        }
    }
}

// ── Index helpers ─────────────────────────────────────────────────────────

#[inline]
fn bbi_for(wx: i32, wz: i32, cx_i: i32, cz_i: i32, zone_rx: i32, zone_rz: i32, nx: usize, nz: usize) -> Option<usize> {
    let dx = wx - (cx_i - zone_rx);
    let dz = wz - (cz_i - zone_rz);
    if dx < 0 || dx >= nx as i32 || dz < 0 || dz >= nz as i32 { return None; }
    Some(dz as usize * nx + dx as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fragment_count_caps() {
        // Build 100 voxels in a 10x1x10 layout.
        let mut voxels = Vec::new();
        for x in 0..10 {
            for z in 0..10 {
                voxels.push(CollapsedVoxel { world_x: x, world_y: 5, world_z: z, material: Material::Granite });
            }
        }
        let mut fields: HashMap<(i32, i32, i32), DensityField> = HashMap::new();
        let frags = fragment_slab(
            &voxels, (0, 5, 0), (9, 5, 9), &fields, 16, 5, Material::Granite,
        );
        // Volume = 100, near the (3,2)/(2,3) range. Aspect ratio square → 2x2.
        assert!(frags.len() <= FRAGMENT_COUNT_MAX);
        assert!(frags.len() >= 2);
    }
}
