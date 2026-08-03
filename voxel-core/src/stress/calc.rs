//! V1 per-voxel stress calculation, world<->chunk sampling helpers,
//! strut load accumulation, and the ground-connectivity relaxation pass.
//!
//! Behavior-preserving split of the former `stress.rs` god file.

use std::collections::{HashMap, HashSet};

use crate::density::DensityField;
use crate::material::Material;

use super::config::StressConfig;
use super::types::{
    SupportField, SupportScoreField, SupportType,
    BrokenStrutEvent, MAX_STRUT_RADIUS, STRUT_TUNING, HP_DAMAGE_SCALE,
};

/// Convert world coordinate to (chunk_key, local_coord).
pub fn world_to_chunk_local(wx: i32, wy: i32, wz: i32, chunk_size: usize) -> ((i32, i32, i32), usize, usize, usize) {
    let cs = chunk_size as i32;
    let cx = wx.div_euclid(cs);
    let cy = wy.div_euclid(cs);
    let cz = wz.div_euclid(cs);
    let lx = wx.rem_euclid(cs) as usize;
    let ly = wy.rem_euclid(cs) as usize;
    let lz = wz.rem_euclid(cs) as usize;
    ((cx, cy, cz), lx, ly, lz)
}

/// Sample density from world coordinates, looking up the correct chunk.
/// Returns None if the chunk is not loaded (treated as solid by caller).
pub fn sample_world(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    wx: i32, wy: i32, wz: i32,
    chunk_size: usize,
) -> Option<(f32, Material)> {
    let (key, lx, ly, lz) = world_to_chunk_local(wx, wy, wz, chunk_size);
    density_fields.get(&key).map(|df| {
        let sample = df.get(lx, ly, lz);
        (sample.density, sample.material)
    })
}

/// Visit every non-empty support-field chunk that could hold a strut within
/// `MAX_STRUT_RADIUS` of (wx,wy,wz). Shared by the three strut sweeps
/// (relief, load-accumulate, BFS halt) — keep them on this helper so their
/// neighborhoods can never drift apart.
///
/// Two walks, cheapest chosen per call (2026-08-03, radius 14 -> 56 resize):
/// - box-probe: one HashMap probe per chunk in the ±MAX_STRUT_RADIUS box.
///   Fine at radius 14 (≤8 probes @ cs=30), ~125 probes at radius 56.
/// - map-filter: iterate support_fields and range-filter — O(#strut chunks)
///   regardless of radius, and worlds hold a handful of strut chunks.
/// The map-filter path visits chunks in SORTED order so float-summing
/// callers (relief) stay deterministic — HashMap iteration order is not.
pub(crate) fn for_each_strut_chunk_in_range(
    support_fields: &HashMap<(i32, i32, i32), SupportField>,
    wx: i32, wy: i32, wz: i32,
    chunk_size: usize,
    mut visit: impl FnMut((i32, i32, i32), &SupportField),
) {
    let sr = MAX_STRUT_RADIUS as i32;
    let cs = chunk_size as i32;
    let kx0 = (wx - sr).div_euclid(cs); let kx1 = (wx + sr).div_euclid(cs);
    let ky0 = (wy - sr).div_euclid(cs); let ky1 = (wy + sr).div_euclid(cs);
    let kz0 = (wz - sr).div_euclid(cs); let kz1 = (wz + sr).div_euclid(cs);
    let box_chunks = ((kx1 - kx0 + 1) as usize)
        * ((ky1 - ky0 + 1) as usize)
        * ((kz1 - kz0 + 1) as usize);
    if support_fields.len() < box_chunks {
        let mut keys: Vec<(i32, i32, i32)> = support_fields.iter()
            .filter(|(k, sf)| {
                !sf.is_empty()
                    && k.0 >= kx0 && k.0 <= kx1
                    && k.1 >= ky0 && k.1 <= ky1
                    && k.2 >= kz0 && k.2 <= kz1
            })
            .map(|(&k, _)| k)
            .collect();
        keys.sort_unstable();
        for k in keys {
            if let Some(sf) = support_fields.get(&k) {
                visit(k, sf);
            }
        }
    } else {
        for ckx in kx0..=kx1 {
            for cky in ky0..=ky1 {
                for ckz in kz0..=kz1 {
                    if let Some(sf) = support_fields.get(&(ckx, cky, ckz)) {
                        if !sf.is_empty() {
                            visit((ckx, cky, ckz), sf);
                        }
                    }
                }
            }
        }
    }
}

/// Walk the same per-voxel strut neighborhood that `calc_voxel_stress_v2`
/// walks for stress reduction, and accumulate each strut's `hardness/dist`
/// contribution into `loads`. Keyed by (chunk, local_xyz). Only alive struts
/// (HP > 0) and only struts inside their per-tier radius are counted.
///
/// Cheap when no struts exist: the sweep pays min(#strut-chunks, box) probes
/// and iterates empty strut lists, so the cost is ~0 for early game.
pub(crate) fn accumulate_strut_load_at_voxel(
    support_fields: &HashMap<(i32, i32, i32), SupportField>,
    loads: &mut std::collections::HashMap<((i32, i32, i32), usize, usize, usize), f32>,
    wx: i32, wy: i32, wz: i32,
    chunk_size: usize,
) {
    // Inverted sweep — iterate actual struts via strut_cells() instead of
    // scanning the max-radius cube. See strut_relief_raw for the rationale
    // (and keep the two loops structurally identical: every strut that
    // relieves a voxel must also accumulate load for it).
    let cs = chunk_size as i32;
    for_each_strut_chunk_in_range(support_fields, wx, wy, wz, chunk_size, |skey, sf| {
        let (ckx, cky, ckz) = skey;
        for &(lx, ly, lz) in sf.strut_cells() {
            let dx = ckx * cs + lx as i32 - wx;
            let dy = cky * cs + ly as i32 - wy;
            let dz = ckz * cs + lz as i32 - wz;
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 == 0 { continue; }
            let support = sf.get(lx as usize, ly as usize, lz as usize);
            let tuning = STRUT_TUNING[support as u8 as usize];
            let r2 = (tuning.radius as i32) * (tuning.radius as i32);
            if d2 > r2 { continue; }
            if !sf.is_strut_alive(lx as usize, ly as usize, lz as usize) { continue; }
            let contribution = tuning.hardness / (d2 as f32).sqrt();
            *loads.entry((skey, lx as usize, ly as usize, lz as usize)).or_insert(0.0) += contribution;
        }
    });
}

/// Apply load-decay HP damage to every strut in `loads`. Each strut takes
/// `max(0, load_borne - tier_decay_threshold) * HP_DAMAGE_SCALE` HP damage;
/// HP is saturating at 0. Returns the list of struts that just hit 0 — caller
/// must clear those cells (via `SupportField::set(.., None)`) and emit
/// `StrutBroken` events.
pub(crate) fn apply_strut_load_damage(
    support_fields: &mut HashMap<(i32, i32, i32), SupportField>,
    loads: &std::collections::HashMap<((i32, i32, i32), usize, usize, usize), f32>,
) -> Vec<BrokenStrutEvent> {
    let mut broken = Vec::new();
    for (&(chunk_key, lx, ly, lz), &load) in loads.iter() {
        let sf = match support_fields.get_mut(&chunk_key) {
            Some(s) => s,
            None => continue,
        };
        let stype = sf.get(lx, ly, lz);
        if stype == SupportType::None { continue; }
        let tuning = STRUT_TUNING[stype as u8 as usize];
        let excess = load - tuning.hp_decay_threshold;
        if excess <= 0.0 { continue; }
        let damage = excess * HP_DAMAGE_SCALE;
        if sf.damage_hp(lx, ly, lz, damage) {
            broken.push(BrokenStrutEvent {
                chunk: chunk_key, lx, ly, lz, support_type: stype,
            });
        }
    }
    broken
}

/// Count contiguous solid voxels above (Y+) a position, capped at 32.
fn column_weight_above(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    wx: i32, wy: i32, wz: i32,
    chunk_size: usize,
) -> u32 {
    let mut count = 0u32;
    for dy in 1..=32i32 {
        let sy = wy + dy;
        match sample_world(density_fields, wx, sy, wz, chunk_size) {
            Some((_, mat)) => {
                if mat.is_solid() {
                    count += 1;
                } else {
                    break;
                }
            }
            // Unloaded = treat as solid (conservative)
            None => count += 1,
        }
    }
    count
}

/// Calculate stress for a single voxel at world coordinates.
///
/// Strut relief is NOT applied here — callers subtract
/// `strut_relief_final_legacy()` from the (zero-clamped) result so relief
/// surplus survives as negative stored stress and offsets painted stress.
pub fn calc_voxel_stress(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    config: &StressConfig,
    wx: i32, wy: i32, wz: i32,
    chunk_size: usize,
) -> f32 {
    // Only solid voxels have stress
    let (_, mat) = match sample_world(density_fields, wx, wy, wz, chunk_size) {
        Some((d, m)) => (d, m),
        None => return 0.0, // Unloaded
    };
    if !mat.is_solid() {
        return 0.0;
    }

    let hardness = config.material_hardness[mat as u8 as usize];
    if hardness <= 0.0 {
        return 0.0;
    }

    // 1. Column weight: gravity from above
    let weight = column_weight_above(density_fields, wx, wy, wz, chunk_size);
    let mut raw_stress = weight as f32 * config.gravity_weight;

    // 2. Support reduction from direct neighbors
    // Voxel below reduces stress
    match sample_world(density_fields, wx, wy - 1, wz, chunk_size) {
        Some((_, m)) if m.is_solid() => {
            raw_stress -= config.vertical_support_factor;
        }
        None => {
            // Unloaded = treat as solid support (conservative)
            raw_stress -= config.vertical_support_factor;
        }
        _ => {}
    }

    // 6-connected lateral neighbors reduce stress
    let lateral_offsets: [(i32, i32, i32); 4] = [
        (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1),
    ];
    for (dx, dy, dz) in &lateral_offsets {
        let nx = wx + dx;
        let ny = wy + dy;
        let nz = wz + dz;
        match sample_world(density_fields, nx, ny, nz, chunk_size) {
            Some((_, m)) if m.is_solid() => {
                raw_stress -= config.lateral_support_factor;
            }
            None => {
                raw_stress -= config.lateral_support_factor;
            }
            _ => {}
        }
    }

    // Clamp to non-negative before normalization. NOTE: strut relief is no
    // longer applied here — callers subtract `strut_relief_final_legacy()`
    // AFTER this clamp so relief surplus survives as negative stored stress
    // and can offset the painted overlay (`effective = stress + painted`).
    // The old inline subtraction died at this clamp, which made struts
    // powerless against map-authored painted stress.
    raw_stress = raw_stress.max(0.0);

    // 3. Normalize by material hardness
    raw_stress / hardness
}

/// Total strut stress relief at a voxel, in RAW stress units (the scale of
/// span/cross-section stress before hardness normalization).
///
/// Sums `hardness / dist` over every ALIVE strut whose per-tier radius
/// (`STRUT_TUNING[type].radius`, Copper=6 .. Mithril=14) reaches the voxel.
/// Near-free when no struts are nearby — the overwhelmingly common case.
pub(crate) fn strut_relief_raw(
    support_fields: &HashMap<(i32, i32, i32), SupportField>,
    wx: i32,
    wy: i32,
    wz: i32,
    chunk_size: usize,
) -> f32 {
    // Inverted sweep (2026-07-17): iterate the ACTUAL struts in the ≤8
    // chunks overlapping the max-radius box via each chunk's sparse
    // `strut_cells()` list, instead of scanning the (2sr+1)^3 cube around
    // the target voxel. The cube scan was fine at MAX_STRUT_RADIUS=5
    // (~7.5µs/voxel near struts) but cubed into ~145µs/voxel at the new
    // radius 14; the strut-list walk is O(nearby struts) — nanoseconds for
    // realistic strut counts, and the no-struts common case is the same
    // ≤8 HashMap probes the old any_supports_in_radius_box guard paid.
    let cs = chunk_size as i32;
    let mut relief = 0.0f32;
    for_each_strut_chunk_in_range(support_fields, wx, wy, wz, chunk_size, |(ckx, cky, ckz), sf| {
        for &(lx, ly, lz) in sf.strut_cells() {
            let dx = ckx * cs + lx as i32 - wx;
            let dy = cky * cs + ly as i32 - wy;
            let dz = ckz * cs + lz as i32 - wz;
            let d2 = dx * dx + dy * dy + dz * dz;
            // A strut occupying the target cell itself relieves
            // nothing there (matches the old scan's center skip).
            if d2 == 0 { continue; }
            let support = sf.get(lx as usize, ly as usize, lz as usize);
            let tuning = STRUT_TUNING[support as u8 as usize];
            let r2 = (tuning.radius as i32) * (tuning.radius as i32);
            if d2 > r2 { continue; }
            // Broken struts (HP=0) contribute nothing — the worker
            // tick will clear them, but until it does we must not
            // pretend they're still holding the rock up.
            if !sf.is_strut_alive(lx as usize, ly as usize, lz as usize) {
                continue;
            }
            relief += tuning.hardness / (d2 as f32).sqrt();
        }
    });
    relief
}

/// Strut relief in the legacy calc's FINAL stress units (raw / material
/// hardness — the legacy path has no depth-pressure factor). Returns 0 for
/// air or zero-hardness voxels. Callers subtract this from the clamped
/// `calc_voxel_stress` result; the difference can go negative, which is the
/// relief surplus that offsets painted stress at `effective()` read time.
pub(crate) fn strut_relief_final_legacy(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    support_fields: &HashMap<(i32, i32, i32), SupportField>,
    config: &StressConfig,
    wx: i32,
    wy: i32,
    wz: i32,
    chunk_size: usize,
) -> f32 {
    let mat = match sample_world(density_fields, wx, wy, wz, chunk_size) {
        Some((_, m)) if m.is_solid() => m,
        _ => return 0.0,
    };
    let hardness = config.material_hardness[mat as u8 as usize];
    if hardness <= 0.0 {
        return 0.0;
    }
    strut_relief_raw(support_fields, wx, wy, wz, chunk_size) / hardness
}

// ── V2 stress algorithm: two-pass ground connectivity + load accumulation ──

/// Minimum distance to nearest air voxel in 6 face-connected directions.
/// Returns 0 if the voxel itself is air, 1 if a face-neighbor is air, etc.
/// Returns `max_dist + 1` if no air found within range (deep interior).
pub(crate) fn min_distance_to_air(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    wx: i32, wy: i32, wz: i32,
    chunk_size: usize,
    max_dist: i32,
) -> i32 {
    let dirs: [(i32, i32, i32); 6] = [
        (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1),
    ];
    let mut best = max_dist + 1;
    for &(dx, dy, dz) in &dirs {
        for d in 1..=max_dist {
            let nx = wx + dx * d;
            let ny = wy + dy * d;
            let nz = wz + dz * d;
            match sample_world(density_fields, nx, ny, nz, chunk_size) {
                Some((_, mat)) if !mat.is_solid() => {
                    best = best.min(d);
                    break; // Found air in this direction
                }
                Some(_) => {} // Solid, keep searching
                None => break, // Unloaded, stop this direction
            }
        }
    }
    best
}

/// Count contiguous air voxels below a position (Y−), capped at 32.
pub fn count_air_below(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    wx: i32, wy: i32, wz: i32,
    chunk_size: usize,
) -> u32 {
    let mut count = 0u32;
    for dy in 1..=12i32 { // Capped at 12 — prevents deep cave stress explosion
        let sy = wy - dy;
        match sample_world(density_fields, wx, sy, wz, chunk_size) {
            Some((_, mat)) if mat.is_solid() => break,
            None => break, // Unloaded = assume solid (conservative)
            _ => count += 1,
        }
    }
    count
}

/// Find minimum lateral distance to a "grounded" voxel (support_score >= threshold).
/// Searches in 4 cardinal directions (X+, X−, Z+, Z−) up to max_dist.
/// Returns the minimum distance found, or max_dist if none found.
fn min_lateral_distance_to_grounded(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    support_scores: &HashMap<(i32, i32, i32), SupportScoreField>,
    wx: i32, wy: i32, wz: i32,
    chunk_size: usize,
    ground_threshold: f32,
    max_dist: u32,
) -> u32 {
    let directions: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    let mut min_dist = max_dist;

    for &(dx, dz) in &directions {
        for d in 1..=max_dist as i32 {
            let nx = wx + dx * d;
            let nz = wz + dz * d;
            // Must be solid to be a grounded support
            match sample_world(density_fields, nx, wy, nz, chunk_size) {
                Some((_, mat)) if mat.is_solid() => {
                    // Check support score
                    let (key, lx, ly, lz) = world_to_chunk_local(nx, wy, nz, chunk_size);
                    let score = support_scores
                        .get(&key)
                        .map(|sf| sf.get(lx, ly, lz))
                        .unwrap_or(1.0); // Unloaded = assume grounded
                    if score >= ground_threshold {
                        min_dist = min_dist.min(d as u32);
                        break;
                    }
                }
                Some(_) => break, // Hit air, stop this direction
                None => {
                    // Unloaded = assume grounded at this distance
                    min_dist = min_dist.min(d as u32);
                    break;
                }
            }
        }
    }
    min_dist
}

/// Measure the unsupported span for a solid surface voxel.
///
/// For each air face-neighbor, searches laterally from that air position through air
/// to find the distance to the nearest wall. Returns the MINIMUM distance found —
/// the nearest wall provides structural support regardless of what's in other directions.
///
/// This handles both ceiling voxels (air below → search laterally at cave level)
/// and wall voxels (air to the side → search laterally through cave).
pub(crate) fn measure_span_from_air(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    wx: i32, wy: i32, wz: i32,
    chunk_size: usize,
    max_dist: u32,
) -> u32 {
    let face_offsets: [(i32, i32, i32); 6] = [
        (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1),
    ];
    let lat_dirs: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    let mut best_span = max_dist;
    let mut found_any = false;

    for &(dx, dy, dz) in &face_offsets {
        let ax = wx + dx;
        let ay = wy + dy;
        let az = wz + dz;

        // Only start from air neighbors
        match sample_world(density_fields, ax, ay, az, chunk_size) {
            Some((_, mat)) if !mat.is_solid() => {}
            _ => continue,
        }

        // From this air position, search laterally for walls
        for &(ldx, ldz) in &lat_dirs {
            for d in 1..=max_dist as i32 {
                let nx = ax + ldx * d;
                let nz = az + ldz * d;
                match sample_world(density_fields, nx, ay, nz, chunk_size) {
                    Some((_, mat)) if mat.is_solid() => {
                        best_span = best_span.min(d as u32);
                        found_any = true;
                        break;
                    }
                    Some(_) => {} // Air — keep going
                    None => {
                        best_span = best_span.min(d as u32); // Unloaded = wall
                        found_any = true;
                        break;
                    }
                }
            }
        }
    }
    if found_any { best_span } else { max_dist }
}

/// Pass 1: Ground connectivity analysis via iterative relaxation.
///
/// For each solid voxel in the specified chunks, computes a `support_score` in [0.0, 1.0]:
/// - 1.0 = directly grounded (solid voxel below all the way down)
/// - 0.0 = completely unsupported (floating)
///
/// Support propagates vertically (0.95 per voxel) and laterally (0.7 per voxel)
/// over multiple relaxation iterations, modeling how walls and pillars support ceilings.
pub fn ground_connectivity_pass(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    chunk_keys: &[(i32, i32, i32)],
    chunk_size: usize,
    config: &StressConfig,
) -> HashMap<(i32, i32, i32), SupportScoreField> {
    // Expand chunk set to include immediate neighbors (needed for boundary propagation)
    let mut expanded_keys: HashSet<(i32, i32, i32)> = HashSet::new();
    for &(cx, cy, cz) in chunk_keys {
        for dz in -1..=1i32 {
            for dy in -1..=1i32 {
                for dx in -1..=1i32 {
                    let key = (cx + dx, cy + dy, cz + dz);
                    if density_fields.contains_key(&key) {
                        expanded_keys.insert(key);
                    }
                }
            }
        }
    }

    let cs = chunk_size;
    let grid_size = cs + 1; // DensityField is (chunk_size+1)^3

    // Initialize support scores via GLOBAL top-down column flood.
    // For each unique (wx, wz) column across dirty chunks, walk from max_wy to min_wy
    // using sample_world for cross-chunk reads. No per-chunk boundary artifacts.
    let mut scores: HashMap<(i32, i32, i32), SupportScoreField> = HashMap::new();
    for &key in &expanded_keys {
        scores.insert(key, SupportScoreField::new(grid_size));
    }

    let vert_decay = config.vertical_transfer_factor;

    // Collect unique (wx, wz) columns from DIRTY chunks only (not all expanded)
    let mut columns: HashSet<(i32, i32)> = HashSet::new();
    for &(cx, _, cz) in chunk_keys {
        for z in 0..grid_size {
            for x in 0..grid_size {
                columns.insert((cx * cs as i32 + x as i32, cz * cs as i32 + z as i32));
            }
        }
    }

    // Y range across all expanded chunks
    let mut min_wy = i32::MAX;
    let mut max_wy = i32::MIN;
    for &(_, cy, _) in &expanded_keys {
        min_wy = min_wy.min(cy * cs as i32);
        max_wy = max_wy.max(cy * cs as i32 + grid_size as i32 - 1);
    }

    // Global flood: each column walks top-to-bottom across all chunks.
    //
    // Perf: consecutive y values in the descending walk almost always land in
    // the same chunk (only `cy` changes when crossing a chunk boundary). Cache
    // the chunk_y and the looked-up DensityField/in_expanded flags so we only
    // re-fetch from `density_fields`/`expanded_keys` when `cy` changes. The
    // `scores.get_mut` write target ALSO only changes when `cy` does (its key is
    // the same `cached_key`), so cache it alongside `cached_df`: `scores` and
    // `density_fields` are DISTINCT maps, so holding a `&mut SupportScoreField`
    // from one while holding a `&DensityField` from the other is sound (no
    // aliasing, no unsafe). This elides the per-solid-cell SipHash probe of
    // `scores` (a 12-byte key re-hashed ~cs times per chunk-column segment)
    // down to one probe per `cy` crossing — the flood's write path was the last
    // per-cell HashMap probe left after the relaxation `scores.get` was hoisted.
    let cs_i32 = cs as i32;
    for &(wx, wz) in &columns {
        let mut current_score = 1.0f32;
        let mut in_air_gap = false;

        let mut cached_cy: Option<i32> = None;
        let mut cached_df: Option<&DensityField> = None;
        let mut cached_sf: Option<&mut SupportScoreField> = None;
        let mut cached_key = (0i32, 0i32, 0i32);

        for wy in (min_wy..=max_wy).rev() {
            let cx = wx.div_euclid(cs_i32);
            let cy = wy.div_euclid(cs_i32);
            let cz = wz.div_euclid(cs_i32);
            let lx = wx.rem_euclid(cs_i32) as usize;
            let ly = wy.rem_euclid(cs_i32) as usize;
            let lz = wz.rem_euclid(cs_i32) as usize;

            if cached_cy != Some(cy) {
                cached_cy = Some(cy);
                cached_key = (cx, cy, cz);
                cached_df = density_fields.get(&cached_key);
                // Only `expanded_keys` chunks get a `SupportScoreField`; for
                // others this is `None` and the write below is skipped, exactly
                // as the old `cached_in_expanded` guard did.
                cached_sf = scores.get_mut(&cached_key);
            }

            let is_solid = cached_df
                .map(|df| df.get(lx, ly, lz).material.is_solid())
                .unwrap_or(false);

            if !is_solid {
                in_air_gap = true;
                current_score = 0.0;
                continue;
            }

            if in_air_gap {
                current_score = 0.0;
                in_air_gap = false;
            }

            if let Some(sf) = cached_sf.as_deref_mut() {
                sf.set(lx, ly, lz, current_score);
            }

            current_score *= vert_decay;
        }
    }

    // Iterative relaxation
    let vert_transfer = config.vertical_transfer_factor;
    let lat_transfer = config.lateral_transfer_factor;

    for _iter in 0..config.support_propagation_iterations {
        // We need to read neighbor scores from the previous iteration,
        // so collect updates first, then apply.
        let mut updates: Vec<((i32, i32, i32), usize, usize, usize, f32)> = Vec::new();

        let cs_i32 = cs as i32;
        let last_local = grid_size - 1;
        for &key in &expanded_keys {
            let df = match density_fields.get(&key) {
                Some(d) => d,
                None => continue,
            };
            // Hoist the chunk's own score field outside the (z,y,x) loops:
            // every cell inside this chunk reads `scores.get(&key)` at least
            // once (for its `current_score`), and most neighbor reads also
            // resolve back to this same chunk because only voxels on a chunk
            // face cross into a different `SupportScoreField`. Caching it
            // saves grid_size^3 redundant HashMap lookups per chunk per
            // iteration (~29k for chunk_size=30) and an additional ~5×
            // savings on neighbor lookups whose key matches `key`.
            let current_sf = match scores.get(&key) {
                Some(sf) => sf,
                None => continue,
            };
            let (cx, cy, cz) = key;
            let chunk_origin_x = cx * cs_i32;
            let chunk_origin_y = cy * cs_i32;
            let chunk_origin_z = cz * cs_i32;

            for z in 0..grid_size {
                for y in 0..grid_size {
                    for x in 0..grid_size {
                        if !df.get(x, y, z).material.is_solid() {
                            continue;
                        }
                        let current_score = current_sf.get(x, y, z);
                        if current_score >= 1.0 {
                            continue; // Already fully grounded
                        }

                        let wx = chunk_origin_x + x as i32;
                        let wy = chunk_origin_y + y as i32;
                        let wz = chunk_origin_z + z as i32;

                        let mut best = current_score;

                        // Vertical transfer from below. Stay on the cached
                        // `current_sf` whenever the cell-below is in the
                        // same chunk (the common case — only y==0 crosses).
                        if y > 0 {
                            let below_score = current_sf.get(x, y - 1, z);
                            best = best.max(below_score * vert_transfer);
                        } else {
                            let bkey = (cx, cy - 1, cz);
                            if let Some(bsf) = scores.get(&bkey) {
                                let below_score = bsf.get(x, last_local, z);
                                best = best.max(below_score * vert_transfer);
                            } else {
                                // Unloaded neighbor = assume grounded
                                best = best.max(vert_transfer);
                            }
                        }

                        // Lateral transfer from 4 horizontal neighbors.
                        // Same-chunk reads use the cached field; only the 4
                        // cells on each face fall through to a HashMap lookup.
                        // -X
                        if x > 0 {
                            let n_score = current_sf.get(x - 1, y, z);
                            best = best.max(n_score * lat_transfer);
                        } else if let Some(nsf) = scores.get(&(cx - 1, cy, cz)) {
                            let n_score = nsf.get(last_local, y, z);
                            best = best.max(n_score * lat_transfer);
                        }
                        // +X
                        if x < last_local {
                            let n_score = current_sf.get(x + 1, y, z);
                            best = best.max(n_score * lat_transfer);
                        } else if let Some(nsf) = scores.get(&(cx + 1, cy, cz)) {
                            let n_score = nsf.get(0, y, z);
                            best = best.max(n_score * lat_transfer);
                        }
                        // -Z
                        if z > 0 {
                            let n_score = current_sf.get(x, y, z - 1);
                            best = best.max(n_score * lat_transfer);
                        } else if let Some(nsf) = scores.get(&(cx, cy, cz - 1)) {
                            let n_score = nsf.get(x, y, last_local);
                            best = best.max(n_score * lat_transfer);
                        }
                        // +Z
                        if z < last_local {
                            let n_score = current_sf.get(x, y, z + 1);
                            best = best.max(n_score * lat_transfer);
                        } else if let Some(nsf) = scores.get(&(cx, cy, cz + 1)) {
                            let n_score = nsf.get(x, y, 0);
                            best = best.max(n_score * lat_transfer);
                        }

                        // No above-transfer: ceiling rock must NOT bootstrap support
                        // from the unsupported mass above it. Support only comes from
                        // below (pillars/floor) and laterally (walls).

                        if best > current_score + 0.001 {
                            updates.push((key, x, y, z, best.min(1.0)));
                        }
                    }
                }
            }
        }

        // Apply updates
        if updates.is_empty() {
            break; // Converged early
        }
        for (key, x, y, z, val) in updates {
            if let Some(sf) = scores.get_mut(&key) {
                sf.set(x, y, z, val);
            }
        }
    }

    scores
}
