//! V1 per-voxel stress calculation, world<->chunk sampling helpers,
//! strut load accumulation, and the ground-connectivity relaxation pass.
//!
//! Behavior-preserving split of the former `stress.rs` god file.

use std::collections::{HashMap, HashSet};

use crate::density::DensityField;
use crate::material::Material;

use super::config::StressConfig;
use super::types::{
    any_supports_in_radius_box, SupportField, SupportScoreField, SupportType,
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

/// Sample support type from world coordinates, looking up the correct chunk.
pub(crate) fn sample_support(
    support_fields: &HashMap<(i32, i32, i32), SupportField>,
    wx: i32, wy: i32, wz: i32,
    chunk_size: usize,
) -> SupportType {
    let (key, lx, ly, lz) = world_to_chunk_local(wx, wy, wz, chunk_size);
    support_fields
        .get(&key)
        .map(|sf| sf.get(lx, ly, lz))
        .unwrap_or(SupportType::None)
}

/// Sample whether a strut at world coordinates is alive (HP > 0). False when
/// the cell has no strut, the chunk isn't loaded, or HP has been ground to 0.
/// Stress reduction and BFS halt both gate on this — broken-but-not-yet-cleared
/// struts must not pretend to be holding rock up.
pub(crate) fn sample_strut_alive(
    support_fields: &HashMap<(i32, i32, i32), SupportField>,
    wx: i32, wy: i32, wz: i32,
    chunk_size: usize,
) -> bool {
    let (key, lx, ly, lz) = world_to_chunk_local(wx, wy, wz, chunk_size);
    support_fields
        .get(&key)
        .map(|sf| sf.is_strut_alive(lx, ly, lz))
        .unwrap_or(false)
}

/// Walk the same per-voxel strut neighborhood that `calc_voxel_stress_v2`
/// walks for stress reduction, and accumulate each strut's `hardness/dist`
/// contribution into `loads`. Keyed by (chunk, local_xyz). Only alive struts
/// (HP > 0) and only struts inside their per-tier radius are counted.
///
/// Cheap when no struts exist: the `any_supports_in_radius_box` short-circuit
/// matches the stress calc's own guard, so the cost is ~0 for early game.
pub(crate) fn accumulate_strut_load_at_voxel(
    support_fields: &HashMap<(i32, i32, i32), SupportField>,
    loads: &mut std::collections::HashMap<((i32, i32, i32), usize, usize, usize), f32>,
    wx: i32, wy: i32, wz: i32,
    chunk_size: usize,
) {
    let sr = MAX_STRUT_RADIUS as i32;
    if !any_supports_in_radius_box(support_fields, wx, wy, wz, sr, chunk_size) {
        return;
    }
    let cs = chunk_size as i32;
    // Cache the last chunk's SupportField across the strut cube. The (2*sr+1)^3
    // box (11^3 = 1331 cells at sr=5) overwhelmingly stays inside the home chunk
    // plus a few neighbors, so iterating x-innermost lets consecutive cells reuse
    // one borrow instead of re-hashing the (i32,i32,i32) chunk key per cell. We
    // also read `get` and `is_strut_alive` off the *same* reference, so each live
    // strut cell costs one probe instead of two. Replaces sample_support +
    // sample_strut_alive (each a full world_to_chunk_local + HashMap probe) — the
    // 26-of-27-miss pattern those two left on this path.
    let mut cached_key: Option<(i32, i32, i32)> = None;
    let mut cached_sf: Option<&SupportField> = None;
    for dz in -sr..=sr {
        for dy in -sr..=sr {
            for dx in -sr..=sr {
                if dx == 0 && dy == 0 && dz == 0 { continue; }
                let sx = wx + dx;
                let sy = wy + dy;
                let sz = wz + dz;
                let skey = (sx.div_euclid(cs), sy.div_euclid(cs), sz.div_euclid(cs));
                if cached_key != Some(skey) {
                    cached_key = Some(skey);
                    cached_sf = support_fields.get(&skey);
                }
                let sf = match cached_sf {
                    Some(sf) => sf,
                    None => continue,
                };
                let slx = sx.rem_euclid(cs) as usize;
                let sly = sy.rem_euclid(cs) as usize;
                let slz = sz.rem_euclid(cs) as usize;
                let support = sf.get(slx, sly, slz);
                if support == SupportType::None { continue; }
                let tuning = STRUT_TUNING[support as u8 as usize];
                let r2 = (tuning.radius as i32) * (tuning.radius as i32);
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 > r2 { continue; }
                if !sf.is_strut_alive(slx, sly, slz) { continue; }
                let dist = (d2 as f32).sqrt();
                let contribution = tuning.hardness / dist;
                *loads.entry((skey, slx, sly, slz)).or_insert(0.0) += contribution;
            }
        }
    }
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
pub fn calc_voxel_stress(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    support_fields: &HashMap<(i32, i32, i32), SupportField>,
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

    // 3. Support structure bonus: nearby ALIVE struts reduce stress.
    //
    // Per-tier sphere of influence: each strut samples its own
    // `STRUT_TUNING[type].radius` (Copper=2 .. Mithril=5). We walk the MAX
    // radius bounding box and let the inner check filter out cells that
    // sit outside any individual strut's radius.
    //
    // Fast skip: if no chunk in the bounding box has any non-None supports,
    // the entire sweep below is pure waste. Cheap O(<=8) chunk lookups guard
    // the per-voxel HashMap walk. For early-game (0 struts placed in the
    // world) this short-circuits ~100% of stressed voxels.
    let sr = MAX_STRUT_RADIUS as i32;
    if any_supports_in_radius_box(support_fields, wx, wy, wz, sr, chunk_size) {
        for dz in -sr..=sr {
            for dy in -sr..=sr {
                for dx in -sr..=sr {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let support = sample_support(support_fields, wx + dx, wy + dy, wz + dz, chunk_size);
                    if support == SupportType::None { continue; }
                    let tuning = STRUT_TUNING[support as u8 as usize];
                    let r2 = (tuning.radius as i32) * (tuning.radius as i32);
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 > r2 { continue; }
                    // Broken struts (HP=0) contribute nothing — the worker
                    // tick will clear them, but until it does we must not
                    // pretend they're still holding the rock up.
                    if !sample_strut_alive(support_fields, wx + dx, wy + dy, wz + dz, chunk_size) {
                        continue;
                    }
                    let dist = (d2 as f32).sqrt();
                    raw_stress -= tuning.hardness / dist;
                }
            }
        }
    }

    // Clamp to non-negative before normalization
    raw_stress = raw_stress.max(0.0);

    // 4. Normalize by material hardness
    raw_stress / hardness
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

                        let _wx = chunk_origin_x + x as i32;
                        let _wy = chunk_origin_y + y as i32;
                        let _wz = chunk_origin_z + z as i32;

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
