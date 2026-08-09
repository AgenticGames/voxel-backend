use std::collections::HashSet;

use voxel_core::density::DensityField;
use voxel_core::material::Material;
use voxel_gen::config::GenerationConfig;
use voxel_noise::{simplex::Simplex3D, NoiseSource};

use crate::store::{sync_boundary_density, ChunkStore};
use crate::types::ConvertedMesh;

// Building flatten apron — smooth ramp from flat building floor down to the
// natural cave floor, sized as a fraction of the building footprint.
//
// Strategy: per (x, z) compute a TARGET FLOOR Y. Inside the footprint Y =
// base.y. Outside, we scan the column for the natural cave floor and lerp Y
// from base.y → natural_floor_y over `apron_radius` voxels. Then for every
// column we (a) RAISE solid below the target Y so low spots fill in and
// (b) LOWER air above the target Y up to clearance so any small overhead
// rock between the ramp and the cave volume is removed.
//
// Constraints honored:
// - Apron radius is ~40% of footprint size (per user request: "no need to
//   ramp out further than 140% the size of the building"). Whatever drop is
//   needed happens in those few voxels; the angle gets steep when the cave
//   floor is far below, but the building stays sealed.
// - Wall columns (no clear "surface" found in the scan range) are NEVER
//   touched. This prevents the apron from tunneling through walls behind/
//   beside a building placed against rock.
// - Natural floor is clamped at base.y so the apron never ramps UP above
//   the building (a tall pillar nearby doesn't lift the surrounding floor).
// - Every cell we raise is recorded for post-sync restoration (sync uses
//   min() at chunk seams which would otherwise create the seam-divot users
//   were seeing).

const APRON_FRAC: f32 = 0.40;        // 40% of footprint = 140% total radius
const APRON_MIN: i32 = 2;            // never less than 2 voxels of apron
const NATURAL_SCAN_UP: i32 = 4;      // search above base.y for natural surface
const NATURAL_SCAN_DOWN: i32 = 24;   // and below — cave floors vary a lot
const RAMP_NOISE_AMP: f32 = 1.0;     // ± voxels of organic Y wobble
const FILL_DOWN_INTERIOR: i32 = 6;   // solid support depth under interior floor

#[inline]
fn apron_radius_for(terrace_size: i32) -> i32 {
    ((terrace_size as f32) * APRON_FRAC).round().max(APRON_MIN as f32) as i32
}

/// Find the natural cave floor Y in this column. Returns:
/// - `Some(y)` if we found a solid cell with air directly above it (a real
///   surface) within the scan range.
/// - `None` if no surface was found (column is fully solid → wall, or fully
///   air → no floor reachable). Apron cells with `None` are LEFT ALONE.
#[inline]
fn natural_floor_y(
    fields: &std::collections::HashMap<(i32, i32, i32), DensityField>,
    cs: i32,
    wx: i32,
    base_y: i32,
    wz: i32,
) -> Option<i32> {
    let cx = wx.div_euclid(cs);
    let cz = wz.div_euclid(cs);
    let lx = wx.rem_euclid(cs) as usize;
    let lz = wz.rem_euclid(cs) as usize;

    let sample_solid = |y: i32| -> Option<bool> {
        let cy = y.div_euclid(cs);
        let ly = y.rem_euclid(cs) as usize;
        fields.get(&(cx, cy, cz)).map(|df| df.get(lx, ly, lz).density > 0.0)
    };

    let top = base_y + NATURAL_SCAN_UP;
    let bot = base_y - NATURAL_SCAN_DOWN;
    let mut prev_air = match sample_solid(top + 1) { Some(s) => !s, None => true };
    for y in (bot..=top).rev() {
        match sample_solid(y) {
            Some(true) if prev_air => return Some(y),
            Some(s) => prev_air = !s,
            None => prev_air = true,
        }
    }
    None
}

/// Per-flatten noise context. Build the Simplex3D tables ONCE per flatten
/// and pass by reference into the apron loop, instead of per-column.
/// `Simplex3D::new` allocates a 512-byte permutation table seeded by
/// ChaCha8Rng + a 256-element shuffle; recreating that for every (wx, wz)
/// in the apron was burning thousands of cycles per column.
struct RampNoiseCtx {
    s_cavern: Simplex3D,
    s_detail: Simplex3D,
    freq: f64,
}

impl RampNoiseCtx {
    fn new(cfg: &GenerationConfig) -> Self {
        Self {
            s_cavern: Simplex3D::new(cfg.seed),
            s_detail: Simplex3D::new(cfg.seed.wrapping_add(1)),
            freq: cfg.noise.cavern_frequency,
        }
    }
}

#[inline]
fn ramp_y_noise(ctx: &RampNoiseCtx, wx: i32, wz: i32) -> f32 {
    let cavern = ctx.s_cavern.sample(wx as f64 * ctx.freq, 0.0, wz as f64 * ctx.freq) as f32;
    let detail = ctx.s_detail.sample(wx as f64 * ctx.freq * 2.5, 7.0, wz as f64 * ctx.freq * 2.5) as f32;
    (cavern * 0.7 + detail * 0.3) * RAMP_NOISE_AMP
}

#[inline]
fn apron_influence(edge_dist: f32, apron_radius: f32) -> f32 {
    let t = (edge_dist / apron_radius).clamp(0.0, 1.0);
    1.0 - (t * t * (3.0 - 2.0 * t))
}

/// Compute sub-voxel boundary densities for a target surface at `target_y_float`.
/// Returns (density_at_floor_voxel, density_at_voxel_above) such that the DC
/// iso-surface (where density crosses 0) lands exactly at `target_y_float`.
///
/// Interpolation: for an iso point at frac inside the boundary cell pair,
///   frac ≤ 0.5: solid density = frac/(1-frac), air density = -1
///   frac > 0.5: solid density = 1, air density = (frac-1)/frac
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

/// Build the ramp at a single (wx, wz) column. Records all RAISED cells in
/// `raised_cells` for post-sync restoration. `target_y_float` is the exact
/// sub-voxel Y for the boundary surface in this column.
fn apply_ramp_column(
    store: &mut ChunkStore,
    cs: i32,
    base_y: i32,
    base_y_float: f32,
    wx: i32,
    wz: i32,
    edge_dist: f32,
    in_interior: bool,
    clearance: i32,
    apron_radius: f32,
    host_material: Material,
    noise: &RampNoiseCtx,
    dirty_set: &mut HashSet<(i32, i32, i32)>,
    raised_cells: &mut Vec<((i32, i32, i32), usize, usize, usize)>,
    changed_count: &mut u32,
) {
    // Target Y (sub-voxel): interior uses exact base_y_float; apron lerps Y
    // to the clamped natural cave floor (integer) so apron stays cheap.
    let target_y_float = if in_interior {
        base_y_float
    } else {
        let nat_y = match natural_floor_y(&store.density_fields, cs, wx, base_y, wz) {
            Some(y) => y,
            None => return,
        };
        let nat_clamped = nat_y.min(base_y) as f32;
        let influence = apron_influence(edge_dist, apron_radius);
        let lerped = base_y_float * influence + nat_clamped * (1.0 - influence);
        let wobble = ramp_y_noise(noise, wx, wz) * influence;
        lerped + wobble
    };

    let (target_y, d_solid, d_air) = subvoxel_boundary_densities(target_y_float);

    let cx = wx.div_euclid(cs);
    let cz = wz.div_euclid(cs);
    let lx = wx.rem_euclid(cs) as usize;
    let lz = wz.rem_euclid(cs) as usize;

    // ── 1. RAISE: fill below target with solid (deep solid + sub-voxel boundary) ──
    let raise_lo = target_y - FILL_DOWN_INTERIOR;
    for y in raise_lo..=target_y {
        let cy = y.div_euclid(cs);
        let ly = y.rem_euclid(cs) as usize;
        let key = (cx, cy, cz);
        let target_density = if y == target_y { d_solid } else { 1.0 };
        if let Some(df) = store.density_fields.get_mut(&key) {
            let sample = df.get_mut(lx, ly, lz);
            if sample.density < target_density {
                *changed_count += 1;
                sample.density = target_density;
                if target_density > 0.0 {
                    sample.material = host_material;
                }
                dirty_set.insert(key);
                raised_cells.push((key, lx, ly, lz));
            }
        }
    }

    // ── 2. LOWER: sub-voxel boundary at target_y+1, then full air to clearance ──
    let carve_top = target_y + clearance;
    for y in (target_y + 1)..=carve_top {
        let cy = y.div_euclid(cs);
        let ly = y.rem_euclid(cs) as usize;
        let key = (cx, cy, cz);
        let target_density = if y == target_y + 1 { d_air } else { -1.0 };
        if let Some(df) = store.density_fields.get_mut(&key) {
            let sample = df.get_mut(lx, ly, lz);
            if sample.density > target_density {
                *changed_count += 1;
                sample.density = target_density;
                if target_density <= 0.0 {
                    sample.material = Material::Air;
                }
                dirty_set.insert(key);
            }
        }
    }
}

/// Flatten a building footprint with a smooth ramp into the natural cave floor.
/// `base_y_float` carries sub-voxel Y precision so the iso-surface lands
/// exactly where UE wants it (no float/sink).
pub fn flatten_terrace(
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

    let mut dirty_set: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut raised_cells: Vec<((i32, i32, i32), usize, usize, usize)> = Vec::new();
    let mut changed_count = 0u32;

    let extent = apron_radius;
    let interior_max = terrace_size - 1;
    // Build the noise context ONCE for the whole apron loop. Was previously
    // built per (wx, wz) inside ramp_y_noise — two PermutationTable allocs +
    // ChaCha8Rng seeds per column.
    let noise = RampNoiseCtx::new(config);

    for dx in -extent..(terrace_size + extent) {
        for dz in -extent..(terrace_size + extent) {
            let wx = base.x + dx;
            let wz = base.z + dz;

            let dx_out = 0.max(-dx).max(dx - interior_max) as f32;
            let dz_out = 0.max(-dz).max(dz - interior_max) as f32;
            let edge_dist = (dx_out * dx_out + dz_out * dz_out).sqrt();
            let in_interior = edge_dist <= 0.0;
            if !in_interior && edge_dist > apron_radius_f { continue; }

            apply_ramp_column(
                store, cs, base.y, base_y_float, wx, wz, edge_dist, in_interior, clear,
                apron_radius_f, host_material, &noise,
                &mut dirty_set, &mut raised_cells, &mut changed_count,
            );

            if dx >= 0 && dx < terrace_size && dz >= 0 && dz < terrace_size {
                store.terraced_cells.insert((wx, base.y, wz));
                store.terraced_columns.insert((wx, wz), base.y);
            }
        }
    }

    // Per-flatten diagnostic — silenced in release. UE swallows eprintln! anyway,
    // and the format!() it forces (12 args) ran on every legacy-path flatten,
    // which now means every DK2 zone tile and every conveyor batch entry.
    #[cfg(debug_assertions)]
    eprintln!("[voxel] flatten_terrace: base=({},{},{}), size={} (+{}apron), clearance={}, changed={} voxels, dirty={} chunks",
        base.x, base.y, base.z, terrace_size, apron_radius, clear, changed_count, dirty_set.len());

    let chunk_size = config.chunk_size;
    let mut dirty_chunks: Vec<_> = dirty_set
        .into_iter()
        .map(|key| (key, 0usize, 0usize, 0usize, chunk_size, chunk_size, chunk_size))
        .collect();

    let extra_dirty = sync_boundary_density(
        &mut store.density_fields, &dirty_chunks, config.chunk_size,
    );
    dirty_chunks.extend(extra_dirty);

    // Post-sync fix: restore EVERY raised cell. sync_boundary_density uses
    // min() across chunk seams which would otherwise pull our 1.0 cells back
    // down to whatever the neighbor chunk had — that's the seam divot the
    // user reported. Restoring all raised cells guarantees the ramp surface
    // is continuous across chunk boundaries.
    for &(key, lx, ly, lz) in &raised_cells {
        if let Some(density) = store.density_fields.get_mut(&key) {
            let sample = density.get_mut(lx, ly, lz);
            if sample.density < 1.0 {
                sample.density = 1.0;
            }
        }
    }

    let dirty_keys: Vec<_> = dirty_chunks.iter().map(|&(k, ..)| k).collect();
    store.modification_tracker.mark_dirty_many(&dirty_keys);

    store.remesh_dirty(&dirty_chunks, config, world_scale)
}

/// Flatten multiple terrace tiles in a single write lock + one remesh pass.
pub fn flatten_terrace_batch(
    store: &mut ChunkStore,
    tiles: &[(glam::IVec3, Material)],
    config: &GenerationConfig,
    world_scale: f32,
    terrace_size: i32,
) -> Vec<((i32, i32, i32), ConvertedMesh)> {
    if tiles.is_empty() {
        return Vec::new();
    }

    let cs = config.chunk_size as i32;
    let apron_radius = apron_radius_for(terrace_size);
    let apron_radius_f = apron_radius as f32;
    let extent = apron_radius;
    let interior_max = terrace_size - 1;
    let clear = 2;

    let mut dirty_set: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut raised_cells: Vec<((i32, i32, i32), usize, usize, usize)> = Vec::new();
    let mut changed_count = 0u32;

    // Build noise context ONCE for ALL tiles. Batch placements (DK2 zones,
    // conveyors) used to pay 2× Simplex3D::new per (wx, wz) per tile. With
    // dozens of conveyors per batch and ~100 columns per tile, that was
    // thousands of redundant ChaCha8Rng-seeded shuffle allocations.
    let noise = RampNoiseCtx::new(config);

    for (base, host_material) in tiles {
        // For zone-style flatten (no per-tile float Y), the integer base.y is
        // the surface — i.e. base_y_float == base.y as f32. The sub-voxel math
        // collapses to the legacy integer-Y placement (frac = 0).
        let base_y_float = base.y as f32;
        for dx in -extent..(terrace_size + extent) {
            for dz in -extent..(terrace_size + extent) {
                let wx = base.x + dx;
                let wz = base.z + dz;

                let dx_out = 0.max(-dx).max(dx - interior_max) as f32;
                let dz_out = 0.max(-dz).max(dz - interior_max) as f32;
                let edge_dist = (dx_out * dx_out + dz_out * dz_out).sqrt();
                let in_interior = edge_dist <= 0.0;
                if !in_interior && edge_dist > apron_radius_f { continue; }

                apply_ramp_column(
                    store, cs, base.y, base_y_float, wx, wz, edge_dist, in_interior, clear,
                    apron_radius_f, *host_material, &noise,
                    &mut dirty_set, &mut raised_cells, &mut changed_count,
                );

                if dx >= 0 && dx < terrace_size && dz >= 0 && dz < terrace_size {
                    store.terraced_cells.insert((wx, base.y, wz));
                    store.terraced_columns.insert((wx, wz), base.y);
                }
            }
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

    for &(key, lx, ly, lz) in &raised_cells {
        if let Some(density) = store.density_fields.get_mut(&key) {
            let sample = density.get_mut(lx, ly, lz);
            if sample.density < 1.0 {
                sample.density = 1.0;
            }
        }
    }

    let dirty_keys: Vec<_> = dirty_chunks.iter().map(|&(k, ..)| k).collect();
    store.modification_tracker.mark_dirty_many(&dirty_keys);

    store.remesh_dirty(&dirty_chunks, config, world_scale)
}

/// Query floor support for a flatten preview.
/// Checks cells in the 2-voxel column below the terrace floor (base.y-1, base.y-2).
/// A column counts as supported if any voxel in that range is solid.
/// Returns (solid_count, clearance_count) — supported columns and solid cells at dy=1,2 above base.
pub fn query_flatten_support(store: &ChunkStore, base: glam::IVec3, chunk_size: i32, terrace_size: i32) -> (u8, u8) {
    let mut solid_count = 0u8;
    let mut clearance_count = 0u8;
    for dx in 0..terrace_size {
        for dz in 0..terrace_size {
            let wx = base.x + dx;
            let wz = base.z + dz;

            // Check 4-voxel column below: any solid = supported
            let mut column_supported = false;
            for dy in 1..=4i32 {
                let check_y = base.y - dy;
                let cx = wx.div_euclid(chunk_size);
                let cy = check_y.div_euclid(chunk_size);
                let cz = wz.div_euclid(chunk_size);
                let lx = wx.rem_euclid(chunk_size) as usize;
                let ly = check_y.rem_euclid(chunk_size) as usize;
                let lz = wz.rem_euclid(chunk_size) as usize;
                if let Some(df) = store.density_fields.get(&(cx, cy, cz)) {
                    if df.get(lx, ly, lz).density > 0.0 {
                        column_supported = true;
                        break;
                    }
                }
            }
            if column_supported {
                solid_count += 1;
            }

            // Clearance (dy=1 and dy=2)
            for dy in 1i32..=2 {
                let vy = base.y + dy;
                let (cx2, cy2, cz2) = (wx.div_euclid(chunk_size), vy.div_euclid(chunk_size), wz.div_euclid(chunk_size));
                let (lx2, ly2, lz2) = (wx.rem_euclid(chunk_size) as usize, vy.rem_euclid(chunk_size) as usize, wz.rem_euclid(chunk_size) as usize);
                if let Some(df) = store.density_fields.get(&(cx2, cy2, cz2)) {
                    if df.get(lx2, ly2, lz2).density > 0.0 {
                        clearance_count = clearance_count.saturating_add(1);
                    }
                }
            }
        }
    }
    (solid_count, clearance_count)
}

/// Query floor support for a building placement footprint.
/// Returns (solid_count, total_columns, first_floor_material).
/// UE-space cell rect origin → Rust-space cell rect origin, for the cell-rect
/// building API.
///
/// THE coord contract, in one place, so the two engine methods that need it
/// cannot drift apart:
///     rust_x =  ue_x                  (same axis, same direction)
///     rust_z = -(ue_y + size_y)       (UE Y is NEGATED, so the half-open span
///                                      [ue_y, ue_y+size_y) becomes
///                                      [-(ue_y+size_y), -ue_y) in Rust Z)
/// UE Z (up) maps to Rust Y unchanged and is passed through separately.
///
/// ⚠️ This mapping is specific to THIS API. Other FFI entry points carry their
/// own conventions — never reuse this for them without re-deriving it.
#[inline]
pub fn ue_cell_rect_to_rust_xz(ue_min_x: i32, ue_min_y: i32, size_y: i32) -> (i32, i32) {
    (ue_min_x, -(ue_min_y + size_y))
}

pub fn query_building_support(store: &ChunkStore, base: glam::IVec3, chunk_size: i32, terrace_size: i32) -> (u8, u8, Material) {
    query_building_support_rect(store, base, chunk_size, terrace_size, terrace_size)
}

/// Rectangular form of `query_building_support`.
///
/// Buildings carry an authored footprint in cells that rotates with their yaw,
/// so a 2x4 machine turned 90 degrees is a 4x2 one — the support test has to
/// follow, or a rotated building would be checked against the wrong columns.
/// The square entry point above delegates here.
///
/// `base.y` is the TOP SOLID cell (not the first body cell); the columns
/// probed are the two directly beneath it, which is the long-standing
/// behaviour the 40%-support threshold in UE was tuned against.
pub fn query_building_support_rect(
    store: &ChunkStore,
    base: glam::IVec3,
    chunk_size: i32,
    size_x: i32,
    size_z: i32,
) -> (u8, u8, Material) {
    // Counts are u8 on the FFI boundary, so a footprint above 15x15 would wrap.
    // Saturate instead: a clamped count still reads as "well supported" against
    // the ratio test, whereas a wrapped one reads as "no floor at all".
    let total_columns = (size_x * size_z).clamp(0, u8::MAX as i32) as u8;
    let mut solid_count = 0u8;
    let mut first_mat = Material::Air;
    for dx in 0..size_x {
        for dz in 0..size_z {
            let wx = base.x + dx;
            let wz = base.z + dz;

            // Check 2-voxel column below: any solid = supported
            for dy in 1..=2i32 {
                let check_y = base.y - dy;
                let cx = wx.div_euclid(chunk_size);
                let cy = check_y.div_euclid(chunk_size);
                let cz = wz.div_euclid(chunk_size);
                let lx = wx.rem_euclid(chunk_size) as usize;
                let ly = check_y.rem_euclid(chunk_size) as usize;
                let lz = wz.rem_euclid(chunk_size) as usize;
                if let Some(df) = store.density_fields.get(&(cx, cy, cz)) {
                    let sample = df.get(lx, ly, lz);
                    if sample.density > 0.0 {
                        solid_count = solid_count.saturating_add(1);
                        if first_mat == Material::Air {
                            first_mat = sample.material;
                        }
                        break;
                    }
                }
            }
        }
    }
    (solid_count, total_columns, first_mat)
}

/// Query whether a terrace exists at the given base position.
/// Returns Some(material) of the floor if all cells are terraced, None otherwise.
/// Checks both `base.y` and `base.y - 1` because the mesh surface sits ~0.5
/// voxels above the floor, so building traces can snap to either Y or Y+1.
pub fn query_terrace(store: &ChunkStore, base: glam::IVec3, terrace_size: i32) -> Option<Material> {
    for y_offset in [0, -1] {
        let check_y = base.y + y_offset;
        let all_present = (0..terrace_size).all(|dx| {
            (0..terrace_size).all(|dz| {
                store.terraced_cells.contains(&(base.x + dx, check_y, base.z + dz))
            })
        });
        if all_present {
            let cs = 16i32;
            let cx = base.x.div_euclid(cs);
            let cy = check_y.div_euclid(cs);
            let cz = base.z.div_euclid(cs);
            let lx = base.x.rem_euclid(cs) as usize;
            let ly = check_y.rem_euclid(cs) as usize;
            let lz = base.z.rem_euclid(cs) as usize;
            return store.density_fields
                .get(&(cx, cy, cz))
                .map(|df| df.get(lx, ly, lz).material);
        }
    }
    None
}

/// Find the nearest terraced column within `search_radius` XY voxels and
/// `max_y_diff` vertical voxels of `approx_y`. Returns the floor Y if found.
pub fn query_nearby_terrace_y(
    store: &ChunkStore,
    base_x: i32,
    base_z: i32,
    approx_y: i32,
    search_radius: i32,
    max_y_diff: i32,
) -> Option<i32> {
    let mut best_dist_sq = i32::MAX;
    let mut best_y = None;
    for dx in -search_radius..=search_radius {
        for dz in -search_radius..=search_radius {
            if let Some(&y) = store.terraced_columns.get(&(base_x + dx, base_z + dz)) {
                if (y - approx_y).abs() <= max_y_diff {
                    let dist_sq = dx * dx + dz * dz;
                    if dist_sq < best_dist_sq {
                        best_dist_sq = dist_sq;
                        best_y = Some(y);
                    }
                }
            }
        }
    }
    best_y
}

#[cfg(test)]
mod cell_rect_tests {
    use super::*;
    use voxel_core::density::DensityField;

    /// The whole point of the cell-rect API: the Rust cells it probes must be
    /// EXACTLY the cells UE reserved — no more, no fewer, no off-by-one from
    /// the Y negation. This is the check that would have caught the
    /// "convert at each call site" coord bugs the hard way.
    #[test]
    fn ue_rect_maps_onto_exactly_the_same_cells() {
        for &(ue_min_y, size_y) in &[(0, 2), (0, 4), (-6, 2), (5, 3), (-1, 1), (100, 12)] {
            let (_, rz0) = ue_cell_rect_to_rust_xz(0, ue_min_y, size_y);
            let rust_span = rz0..(rz0 + size_y);

            for ue_y in ue_min_y..(ue_min_y + size_y) {
                // UE cell `ue_y` spans world [ue_y*C, (ue_y+1)*C). Negating puts
                // that span in Rust Z at floor index -ue_y - 1.
                let rust_cell = -ue_y - 1;
                assert!(
                    rust_span.contains(&rust_cell),
                    "UE cell {} (rect {}..{}) mapped to Rust {}, outside {:?}",
                    ue_y, ue_min_y, ue_min_y + size_y, rust_cell, rust_span
                );
            }
        }
    }

    /// X passes straight through; only Y is negated.
    #[test]
    fn ue_rect_x_axis_is_untouched() {
        for &x in &[-9, 0, 3, 77] {
            assert_eq!(ue_cell_rect_to_rust_xz(x, 0, 2).0, x);
        }
    }

    fn flat_ground(ground_y: i32) -> ChunkStore {
        let cs: usize = 16;
        let mut store = ChunkStore::new(cs as i32);
        for cx in -1..=1 {
            for cz in -1..=1 {
                for cy in -1..=1 {
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

    #[test]
    fn rect_support_counts_every_column() {
        let ground_y = 8;
        let store = flat_ground(ground_y);
        // base.y is the top SOLID cell, and the helper probes the two below it.
        let base = glam::IVec3::new(2, ground_y - 1, 2);

        let (solid, total, mat) = query_building_support_rect(&store, base, 16, 2, 4);
        assert_eq!(total, 8, "2x4 rect is 8 columns");
        assert_eq!(solid, 8, "flat ground supports every column");
        assert_eq!(mat, Material::Granite);

        // Rotating the footprint swaps the axes and must not change the verdict
        // on uniform ground — but it IS a different set of columns.
        let (solid_rot, total_rot, _) = query_building_support_rect(&store, base, 16, 4, 2);
        assert_eq!((solid_rot, total_rot), (8, 8));
    }

    /// The square entry point must stay a pure delegation — if these ever
    /// disagree, every existing caller silently changed behaviour.
    #[test]
    fn square_query_matches_rect_query() {
        let store = flat_ground(8);
        let base = glam::IVec3::new(0, 7, 0);
        for n in 1..=6 {
            assert_eq!(
                query_building_support(&store, base, 16, n),
                query_building_support_rect(&store, base, 16, n, n),
                "square/rect disagree at size {}", n
            );
        }
    }

    /// Unsupported air must report zero, not a wrapped or defaulted count —
    /// UE's 40% threshold turns a bogus high count into a building placed on
    /// nothing.
    #[test]
    fn rect_support_reports_zero_over_air() {
        let store = flat_ground(8);
        // Well above the ground: the two cells probed below are both air.
        let base = glam::IVec3::new(2, 14, 2);
        let (solid, total, _) = query_building_support_rect(&store, base, 16, 2, 2);
        assert_eq!(solid, 0);
        assert_eq!(total, 4);
    }
}
