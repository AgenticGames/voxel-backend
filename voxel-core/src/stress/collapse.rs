//! Collapse detection + execution: v1 flood-fill collapses, deferred pile
//! application, and the v2 coherent-slab collapse pipeline (incl. the
//! force-deadline cinematic variant).
//!
//! Behavior-preserving split of the former `stress.rs` god file.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU32, Ordering};

use voxel_noise::{simplex::Simplex3D, NoiseSource};

use crate::density::DensityField;
use crate::material::Material;

use super::config::StressConfig;
use super::types::{
    bfs_halt_damage,
    BrokenStrutEvent, CollapseEvent, CollapseEventV2,
    CollapseSlab, CollapsedVoxel, OverstressedVoxel, PendingPilePlacement,
    RubbleVoxel, StressField, SupportField, SupportType,
    MAX_STRUT_RADIUS, STRUT_TUNING,
};
use super::calc::{sample_world, world_to_chunk_local};

/// Why candidate regions were REJECTED by the natural pass, accumulated since
/// the last `take_collapse_skip_stats()` (2026-09-06). Debug readout only -
/// the worker prints them into stress_debug.txt so a "seeds but no event"
/// recalc names its filter instead of leaving a floating sheet unexplained.
static SKIP_MIN_REGION: AtomicU32 = AtomicU32::new(0);
static SKIP_NO_FALLABLE: AtomicU32 = AtomicU32::new(0);
static SKIP_GROUNDED: AtomicU32 = AtomicU32::new(0);

/// (below min_collapse_region, no fallable column, median-grounded) since the last call.
pub fn take_collapse_skip_stats() -> (u32, u32, u32) {
    (
        SKIP_MIN_REGION.swap(0, Ordering::Relaxed),
        SKIP_NO_FALLABLE.swap(0, Ordering::Relaxed),
        SKIP_GROUNDED.swap(0, Ordering::Relaxed),
    )
}

/// Detect contiguous overstressed regions via flood-fill (6-connected BFS)
/// and execute collapses: convert to Air, place rubble, mark dirty chunks.
pub fn detect_and_execute_collapses(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    stress_fields: &mut HashMap<(i32, i32, i32), StressField>,
    _support_fields: &HashMap<(i32, i32, i32), SupportField>,
    overstressed: &[OverstressedVoxel],
    config: &StressConfig,
    chunk_size: usize,
) -> Vec<CollapseEvent> {
    if overstressed.is_empty() {
        return Vec::new();
    }

    let mut visited: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut events = Vec::new();

    // Build set for quick lookup
    let overstressed_set: HashSet<(i32, i32, i32)> = overstressed
        .iter()
        .map(|v| (v.world_x, v.world_y, v.world_z))
        .collect();

    for ov in overstressed {
        let start = (ov.world_x, ov.world_y, ov.world_z);
        if visited.contains(&start) {
            continue;
        }

        // BFS flood-fill to find contiguous overstressed region
        let mut queue = VecDeque::new();
        let mut region: Vec<(i32, i32, i32)> = Vec::new();
        queue.push_back(start);
        visited.insert(start);

        while let Some(pos) = queue.pop_front() {
            if region.len() >= config.max_collapse_volume as usize {
                break;
            }
            region.push(pos);

            // Check 6-connected neighbors
            let offsets: [(i32, i32, i32); 6] = [
                (1, 0, 0), (-1, 0, 0),
                (0, 1, 0), (0, -1, 0),
                (0, 0, 1), (0, 0, -1),
            ];
            for (dx, dy, dz) in &offsets {
                let neighbor = (pos.0 + dx, pos.1 + dy, pos.2 + dz);
                if !visited.contains(&neighbor) && overstressed_set.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }

        if region.is_empty() {
            continue;
        }

        // Calculate center
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_z = 0.0f32;
        for &(x, y, z) in &region {
            sum_x += x as f32;
            sum_y += y as f32;
            sum_z += z as f32;
        }
        let n = region.len() as f32;
        let center = (sum_x / n, sum_y / n, sum_z / n);

        // Execute collapse: convert voxels to Air
        let mut collapsed_voxels = Vec::with_capacity(region.len());
        let mut affected_chunks_set = HashSet::new();

        for &(wx, wy, wz) in &region {
            let (key, lx, ly, lz) = world_to_chunk_local(wx, wy, wz, chunk_size);

            // Get original material before clearing
            let material = if let Some(df) = density_fields.get(&key) {
                df.get(lx, ly, lz).material
            } else {
                Material::Air
            };

            // Set to Air
            if let Some(df) = density_fields.get_mut(&key) {
                let sample = df.get_mut(lx, ly, lz);
                sample.density = -1.0;
                sample.material = Material::Air;
            }

            // Clear stress
            if let Some(sf) = stress_fields.get_mut(&key) {
                sf.set(lx, ly, lz, 0.0);
            }

            affected_chunks_set.insert(key);
            collapsed_voxels.push(CollapsedVoxel {
                world_x: wx,
                world_y: wy,
                world_z: wz,
                material,
            });
        }

        // Place rubble below collapsed region
        let mut rubble_voxels = Vec::new();
        if config.rubble_enabled {
            let rubble_count = (region.len() as f32 * config.rubble_fill_ratio) as usize;
            let mut placed = 0;

            for cv in &collapsed_voxels {
                if placed >= rubble_count {
                    break;
                }
                // Trace downward to find first solid surface
                let mut ry = cv.world_y - 1;
                let mut found_surface = false;
                for _ in 0..32 {
                    match sample_world(density_fields, cv.world_x, ry, cv.world_z, chunk_size) {
                        Some((_, mat)) if mat.is_solid() => {
                            // Place rubble one above solid surface
                            ry += 1;
                            found_surface = true;
                            break;
                        }
                        None => {
                            // Unloaded, stop
                            break;
                        }
                        _ => {
                            ry -= 1;
                        }
                    }
                }

                if found_surface && ry < cv.world_y {
                    let (rkey, rlx, rly, rlz) = world_to_chunk_local(
                        cv.world_x, ry, cv.world_z, chunk_size,
                    );
                    // Only place rubble in air voxels
                    let is_air = density_fields
                        .get(&rkey)
                        .map(|df| !df.get(rlx, rly, rlz).material.is_solid())
                        .unwrap_or(false);

                    if is_air {
                        if let Some(df) = density_fields.get_mut(&rkey) {
                            let sample = df.get_mut(rlx, rly, rlz);
                            sample.density = 1.0;
                            sample.material = cv.material;
                        }
                        affected_chunks_set.insert(rkey);
                        rubble_voxels.push(RubbleVoxel {
                            world_x: cv.world_x,
                            world_y: ry,
                            world_z: cv.world_z,
                            material: cv.material,
                        });
                        placed += 1;
                    }
                }
            }
        }

        events.push(CollapseEvent {
            volume: collapsed_voxels.len() as u32,
            collapsed_voxels,
            rubble_voxels,
            affected_chunks: affected_chunks_set.into_iter().collect(),
            center,
        });
    }

    events
}

// ── V2 collapse detection: coherent slab collapse ──

/// Detect contiguous overstressed regions and produce coherent falling slabs.
///
/// Key improvements over v1:
/// - Slab cohesion expansion: includes nearly-overstressed neighbors (>= slab_cohesion_threshold)
///   to prevent ragged holes
/// - Minimum region filter: skips tiny regions (< min_collapse_region)
/// - Uniform slab translation: entire slab drops as a unit, preserving shape
/// - Rubble preserves slab geometry at landing position
/// Apply a previously-deferred pile placement. Mutates density to add the
/// rubble pile, returns the chunks affected. The caller is responsible for
/// remeshing those chunks after this call.
pub fn apply_pending_pile(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    config: &StressConfig,
    pending: &PendingPilePlacement,
    chunk_size: usize,
) -> Vec<(i32, i32, i32)> {
    let pile_result = crate::collapse_pile::place_collapse_pile(
        density_fields, config, &pending.collapsed_voxels,
        pending.bb_min, pending.bb_max,
        pending.dominant_material, pending.landing_offset, chunk_size,
    );
    pile_result.affected_chunks.into_iter().collect()
}

/// Like `apply_pending_pile` but returns the full `PlacementResult` so the
/// caller can inspect `written_cells` (e.g. to extract a preview mesh and
/// then roll back the writes).
pub fn apply_pending_pile_with_result(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    config: &StressConfig,
    pending: &PendingPilePlacement,
    chunk_size: usize,
) -> crate::collapse_pile::PlacementResult {
    crate::collapse_pile::place_collapse_pile(
        density_fields, config, &pending.collapsed_voxels,
        pending.bb_min, pending.bb_max,
        pending.dominant_material, pending.landing_offset, chunk_size,
    )
}

pub fn detect_and_execute_collapses_v2(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    stress_fields: &mut HashMap<(i32, i32, i32), StressField>,
    support_fields: &mut HashMap<(i32, i32, i32), SupportField>,
    overstressed: &[OverstressedVoxel],
    config: &StressConfig,
    chunk_size: usize,
) -> Vec<CollapseEventV2> {
    let mut broken = Vec::new();
    detect_and_execute_collapses_v2_with_force(
        density_fields, stress_fields, support_fields,
        overstressed, config, chunk_size,
        false, // defer_pile
        false, // force_collapse
        true,  // halt_at_struts — sleep path: alive struts brace the slab
        &mut broken,
    )
}

/// Same as `detect_and_execute_collapses_v2` but with an option to defer
/// pile placement. When `defer_pile = true`, slab voxels are still cleared
/// (cave roof opens immediately) but the rubble pile is NOT placed —
/// instead, `pending_piles` is populated on each event so the caller can
/// apply piles later (e.g., scheduled at impact time for the cinematic).
///
/// This wrapper enables strut halt by default — alive struts halt the BFS
/// and take HP damage. Pass the returned events; broken struts are cleared
/// from `support_fields` and a fresh stress recalc will mark dirty cells.
/// Use `_with_force` directly when scripted triggers should bypass struts.
pub fn detect_and_execute_collapses_v2_with_options(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    stress_fields: &mut HashMap<(i32, i32, i32), StressField>,
    support_fields: &mut HashMap<(i32, i32, i32), SupportField>,
    overstressed: &[OverstressedVoxel],
    config: &StressConfig,
    chunk_size: usize,
    defer_pile: bool,
) -> Vec<CollapseEventV2> {
    let mut broken = Vec::new();
    detect_and_execute_collapses_v2_with_force(
        density_fields,
        stress_fields,
        support_fields,
        overstressed,
        config,
        chunk_size,
        defer_pile,
        false, // force_collapse — default off, natural filters apply
        true,  // halt_at_struts — alive struts brace the slab
        &mut broken,
    )
}

/// Like `detect_and_execute_collapses_v2_with_options` but exposes the
/// `force_collapse` flag and the `halt_at_struts` flag explicitly.
///
/// `force_collapse=true` bypasses the grounding filter (`landing_offset
/// <= 0`) and forces grounded regions to "fall" a default distance — used
/// by scripted editor triggers that need to collapse cave walls/pillars
/// the designer has authored to fall anyway. Combine with
/// `halt_at_struts=false` to let scripted slabs override struts entirely.
///
/// When `halt_at_struts=true`, BFS expansion stops at any voxel within
/// `MAX_STRUT_RADIUS` of an alive strut, AND each halting strut takes
/// `blocked_voxels * BFS_HALT_DAMAGE_SCALE` HP damage. Broken struts
/// are pushed to `broken_out` and cleared from `support_fields`.
pub fn detect_and_execute_collapses_v2_with_force(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    stress_fields: &mut HashMap<(i32, i32, i32), StressField>,
    support_fields: &mut HashMap<(i32, i32, i32), SupportField>,
    overstressed: &[OverstressedVoxel],
    config: &StressConfig,
    chunk_size: usize,
    defer_pile: bool,
    force_collapse: bool,
    halt_at_struts: bool,
    broken_out: &mut Vec<BrokenStrutEvent>,
) -> Vec<CollapseEventV2> {
    let (events, _hit) = detect_and_execute_collapses_v2_with_force_deadline(
        density_fields, stress_fields, support_fields,
        overstressed, config, chunk_size,
        defer_pile, force_collapse, halt_at_struts,
        broken_out, None,
    );
    events
}

/// Same as `_with_force` but lets the caller specify a wall-clock deadline.
/// When `deadline` is `Some(t)` and the per-seed loop crosses it, the
/// cascade returns whatever events accumulated so far AND `hit_deadline=true`.
/// Used by `voxel-sleep::systems::collapse::apply_collapse` to put a 10 s
/// cap on the cascade so a pathological dense world can't black-screen the
/// player forever waiting on Phase 4. Pass `None` to disable.
pub fn detect_and_execute_collapses_v2_with_force_deadline(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    stress_fields: &mut HashMap<(i32, i32, i32), StressField>,
    support_fields: &mut HashMap<(i32, i32, i32), SupportField>,
    overstressed: &[OverstressedVoxel],
    config: &StressConfig,
    chunk_size: usize,
    defer_pile: bool,
    force_collapse: bool,
    halt_at_struts: bool,
    broken_out: &mut Vec<BrokenStrutEvent>,
    deadline: Option<std::time::Instant>,
) -> (Vec<CollapseEventV2>, bool) {
    if overstressed.is_empty() {
        return (Vec::new(), false);
    }

    let mut visited: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut events = Vec::new();
    let mut hit_deadline = false;
    let mut seeds_processed: usize = 0;

    // Build set and stress lookup for quick access
    let overstressed_set: HashSet<(i32, i32, i32)> = overstressed
        .iter()
        .map(|v| (v.world_x, v.world_y, v.world_z))
        .collect();

    // Per-strut braced-workload tally, summed across all collapse events
    // processed in this call. Applied after the loop as HP damage. f32:
    // when N struts cover the same blocked voxel they SHARE the cost 1/N
    // each (2026-08-03 — overlapping struts previously each paid full
    // price for the same voxel).
    let mut strut_halt_counts: std::collections::HashMap<((i32,i32,i32), usize, usize, usize), f32> =
        std::collections::HashMap::new();

    for ov in overstressed {
        // Deadline check — happens at every seed so the longest a single
        // seed's BFS can stall is bounded by `config.max_collapse_volume`.
        // When `deadline` is None this is free.
        if let Some(t) = deadline {
            if std::time::Instant::now() >= t {
                hit_deadline = true;
                eprintln!(
                    "[stress] detect_and_execute_collapses_v2 hit deadline: processed {}/{} seeds, {} events accumulated",
                    seeds_processed, overstressed.len(), events.len()
                );
                break;
            }
        }
        seeds_processed += 1;

        let start = (ov.world_x, ov.world_y, ov.world_z);
        if visited.contains(&start) {
            continue;
        }

        // Check if the starting voxel can actually fall (has air below within 48 voxels)
        // BFS flood-fill: find contiguous region of overstressed voxels.
        // No "can fall" filter — all overstressed voxels join the region.
        // Fall eligibility is checked per-column in the landing computation.
        let mut queue = VecDeque::new();
        let mut region: Vec<(i32, i32, i32)> = Vec::new();
        queue.push_back(start);
        visited.insert(start);

        while let Some(pos) = queue.pop_front() {
            if region.len() >= config.max_collapse_volume as usize {
                break;
            }
            region.push(pos);

            // 26-connected BFS: face + edge + corner neighbors.
            for dz in -1..=1i32 {
                for dy in -1..=1i32 {
                    for dx in -1..=1i32 {
                        if dx == 0 && dy == 0 && dz == 0 { continue; }
                        let neighbor = (pos.0 + dx, pos.1 + dy, pos.2 + dz);
                        if visited.contains(&neighbor) {
                            continue;
                        }

                        // Strut halt: if the candidate voxel sits inside the
                        // sphere of influence of any alive strut, the strut
                        // braces it — skip expansion and tally the blocked
                        // cell for post-pass HP damage. Each blocking strut
                        // in range eats one count.
                        if halt_at_struts {
                            // Inverted sweep — iterate actual struts via each
                            // overlapping chunk's strut_cells() list instead
                            // of scanning the (2·sr_max+1)^3 cube per BFS
                            // frontier voxel (see strut_relief_raw). This
                            // path ran the old cube scan with UNCACHED
                            // per-cell probes — at radius 14 that was ~24k
                            // cells × 2 HashMap probes per frontier voxel.
                            let cs_i = chunk_size as i32;
                            let (nx, ny, nz) = neighbor;
                            // Two-phase: collect every alive strut covering
                            // this voxel first, then split the workload
                            // 1/N so co-bracing struts SHARE the damage.
                            let mut covering: Vec<((i32,i32,i32), usize, usize, usize)> = Vec::new();
                            super::calc::for_each_strut_chunk_in_range(
                                support_fields, nx, ny, nz, chunk_size,
                                |skey, sf| {
                                    let (ckx, cky, ckz) = skey;
                                    for &(lx, ly, lz) in sf.strut_cells() {
                                        let sdx = ckx * cs_i + lx as i32 - nx;
                                        let sdy = cky * cs_i + ly as i32 - ny;
                                        let sdz = ckz * cs_i + lz as i32 - nz;
                                        let d2 = sdx * sdx + sdy * sdy + sdz * sdz;
                                        if d2 == 0 { continue; }
                                        let support = sf.get(lx as usize, ly as usize, lz as usize);
                                        let tuning = STRUT_TUNING[support as u8 as usize];
                                        let r2 = (tuning.radius as i32) * (tuning.radius as i32);
                                        if d2 > r2 { continue; }
                                        if !sf.is_strut_alive(lx as usize, ly as usize, lz as usize) {
                                            continue;
                                        }
                                        covering.push((skey, lx as usize, ly as usize, lz as usize));
                                    }
                                });
                            let halted = !covering.is_empty();
                            if halted {
                                let share = 1.0f32 / covering.len() as f32;
                                for key in covering {
                                    *strut_halt_counts.entry(key).or_insert(0.0) += share;
                                }
                            }
                            if halted {
                                visited.insert(neighbor); // mark to avoid re-checking
                                continue;
                            }
                        }

                        // Include neighbor if:
                        //  - it's already in the overstressed seed set, OR
                        //  - (natural only) it's solid AND has natural stress
                        //    above the cohesion threshold.
                        //
                        // For force_collapse, the slab is EXACTLY the painted
                        // seed set. We do NOT BFS-expand into surrounding rock —
                        // earlier attempts at "expand to all connected solid"
                        // flooded ±2000 cells around the painted region, creating
                        // collapses far from where the designer painted. Stick
                        // to what was painted; if the designer wants a bigger
                        // slab, they paint more cells.
                        let include = if overstressed_set.contains(&neighbor) {
                            true
                        } else if force_collapse {
                            false
                        } else {
                            let (nkey, nlx, nly, nlz) = world_to_chunk_local(
                                neighbor.0, neighbor.1, neighbor.2, chunk_size,
                            );
                            let is_solid = density_fields
                                .get(&nkey)
                                .map(|df| df.get(nlx, nly, nlz).material.is_solid())
                                .unwrap_or(false);
                            // EFFECTIVE stress (base + painted overlay), 2026-09-06.
                            // Reading the stored base alone meant a painted band
                            // sitting at 0.85-0.99 (cracked on screen) could never
                            // ride along with the slab above and below it - the
                            // seed test uses base+painted, cohesion did not, and
                            // the mismatch left floating sheets inside painted
                            // set pieces.
                            let stress_val = stress_fields
                                .get(&nkey)
                                .map(|sf| sf.effective(nlx, nly, nlz))
                                .unwrap_or(0.0);
                            is_solid && stress_val >= config.slab_cohesion_threshold
                        };

                        if include {
                            visited.insert(neighbor);
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }

        // Minimum region filter
        if (region.len() as u32) < config.min_collapse_region {
            SKIP_MIN_REGION.fetch_add(1, Ordering::Relaxed);
            continue;
        }

        // Compute bounding box and center
        let mut bb_min = (i32::MAX, i32::MAX, i32::MAX);
        let mut bb_max = (i32::MIN, i32::MIN, i32::MIN);
        let mut sum = (0.0f32, 0.0f32, 0.0f32);
        let mut material_counts: HashMap<Material, u32> = HashMap::new();

        for &(x, y, z) in &region {
            bb_min.0 = bb_min.0.min(x);
            bb_min.1 = bb_min.1.min(y);
            bb_min.2 = bb_min.2.min(z);
            bb_max.0 = bb_max.0.max(x);
            bb_max.1 = bb_max.1.max(y);
            bb_max.2 = bb_max.2.max(z);
            sum.0 += x as f32;
            sum.1 += y as f32;
            sum.2 += z as f32;

            let (key, lx, ly, lz) = world_to_chunk_local(x, y, z, chunk_size);
            if let Some(df) = density_fields.get(&key) {
                let mat = df.get(lx, ly, lz).material;
                *material_counts.entry(mat).or_insert(0) += 1;
            }
        }
        let n = region.len() as f32;
        let center = (sum.0 / n, sum.1 / n, sum.2 / n);

        // Filter Air out of dominant_material — stress's BFS region can
        // include marginal air-classified cells, and "dominant=Air" then
        // propagates everywhere as matte-black mesh material in UE.
        let dominant_material = material_counts
            .into_iter()
            .filter(|(m, _)| (*m as u8) > 0)
            .max_by_key(|&(_, count)| count)
            .map(|(mat, _)| mat)
            .unwrap_or(Material::Granite);

        // Compute landing position using only columns with immediate air below
        // (actual ceiling surfaces). Wall/floor voxels in the region are ignored
        // for fall distance — they just get removed along with the slab.
        let region_set: HashSet<(i32, i32, i32)> = region.iter().copied().collect();
        let mut column_min_y: HashMap<(i32, i32), i32> = HashMap::new();
        for &(x, y, z) in &region {
            // Only include this column if the voxel at the bottom has air below
            let entry = column_min_y.entry((x, z)).or_insert(y);
            *entry = (*entry).min(y);
        }

        // Filter to only columns with air immediately below the slab
        let fallable_columns: Vec<((i32, i32), i32)> = column_min_y.iter()
            .filter(|&(&(x, z), &min_y)| {
                // Check if the voxel below the slab bottom in this column is air
                match sample_world(density_fields, x, min_y - 1, z, chunk_size) {
                    Some((_, mat)) => !mat.is_solid(),
                    None => false,
                }
            })
            .map(|(&k, &v)| (k, v))
            .collect();

        if fallable_columns.is_empty() && !force_collapse {
            SKIP_NO_FALLABLE.fetch_add(1, Ordering::Relaxed);
            continue; // No columns can fall — entire region is embedded in solid.
            // Scripted triggers (force_collapse=true) still proceed: the
            // grounding bypass below applies a default fall distance so the
            // cinematic plays even for rock with no natural fall path.
        }

        // Compute fall offset per column, then use MEDIAN (not minimum).
        // One wall column near the floor shouldn't anchor the whole ceiling slab.
        let mut column_offsets: Vec<i32> = Vec::with_capacity(fallable_columns.len());
        for &((x, z), min_y) in &fallable_columns {
            let mut floor_y = min_y - 1;
            let mut found = false;
            for _ in 0..64 {
                if region_set.contains(&(x, floor_y, z)) {
                    floor_y -= 1;
                    continue;
                }
                match sample_world(density_fields, x, floor_y, z, chunk_size) {
                    Some((_, mat)) if mat.is_solid() => {
                        column_offsets.push(min_y - (floor_y + 1));
                        found = true;
                        break;
                    }
                    None => {
                        column_offsets.push(min_y - (floor_y + 1));
                        found = true;
                        break;
                    }
                    _ => floor_y -= 1,
                }
            }
            if !found {
                column_offsets.push(64);
            }
        }

        column_offsets.sort();
        let mut landing_offset = if column_offsets.is_empty() {
            0
        } else {
            column_offsets[column_offsets.len() / 2] // median
        };

        if landing_offset <= 0 {
            if force_collapse {
                // Scripted trigger AND no natural fall path. Use a small
                // default so the pile lands close to the painted region
                // rather than tunneling far below into solid rock.
                landing_offset = 4;
            } else {
                SKIP_GROUNDED.fetch_add(1, Ordering::Relaxed);
                continue; // Median says slab is grounded
            }
        }
        // Note: for force_collapse with a meaningful natural fall distance,
        // trust the natural median. The earlier behavior (force min 8) put
        // the pile too far below the painted region — e.g. painted tunnel
        // ceiling, pile would land 320 UU below the tunnel floor inside
        // solid rock, making the cinematic look like "rock appeared
        // somewhere unrelated." Now the pile lands on the tunnel floor,
        // right under the hole, like a normal cave-in.

        let landing_y = bb_min.1 - landing_offset;

        // Record collapsed voxels with original material, then clear them.
        // Track slab-affected chunks separately from the wider affected_chunks
        // set so the worker can remesh roof chunks at fall-start time and
        // pile chunks at impact time (the cinematic-aligned split).
        let mut collapsed_voxels = Vec::with_capacity(region.len());
        let mut slab_chunks_set: HashSet<(i32, i32, i32)> = HashSet::new();
        let mut pile_chunks_set: HashSet<(i32, i32, i32)> = HashSet::new();
        let mut affected_chunks_set = HashSet::new();

        for &(wx, wy, wz) in &region {
            let (key, lx, ly, lz) = world_to_chunk_local(wx, wy, wz, chunk_size);

            let material = density_fields
                .get(&key)
                .map(|df| df.get(lx, ly, lz).material)
                .unwrap_or(Material::Air);

            // Set to Air (remove the slab from its original position)
            if let Some(df) = density_fields.get_mut(&key) {
                let sample = df.get_mut(lx, ly, lz);
                sample.density = -1.0;
                sample.material = Material::Air;
            }
            if let Some(sf) = stress_fields.get_mut(&key) {
                sf.set(lx, ly, lz, 0.0);
            }
            slab_chunks_set.insert(key);
            affected_chunks_set.insert(key);

            collapsed_voxels.push(CollapsedVoxel {
                world_x: wx, world_y: wy, world_z: wz, material,
            });
        }

        // Place rubble as a sealed elliptical cone, then add roughness on top.
        //
        // Three passes, each one ADDITIVE — no pass ever removes solid:
        //
        //  1. SEALED CONE BASE. Stack of fully-filled elliptical discs from
        //     the floor up. At height fraction f the disc radius is
        //     (1 - f) * full_radius. Every air cell inside the cone gets
        //     solid — no per-column breaks, no noise gating. Guaranteed sealed.
        //
        //  2. NOISE CRUST. Walk each (x, z) inside the cone footprint and
        //     stamp 0–2 extra voxels above its cone-top where simplex noise
        //     is positive. Pure additive — breaks the perfect-cone silhouette.
        //
        //  3. BOULDERS. A handful of half-buried solid spheres on the surface
        //     using the dominant slab material for chunky roughness.
        //
        // Pile sizing: cone volume = pi*R^2*H/3, solve for H from
        // collapsed_volume * rubble_fill_ratio. Seeded from collapse center
        // for determinism (multiplayer-safe).
        // Cinematic collapse pile (see crate::collapse_pile). Splits the slab
        // into fragments, runs angle-of-repose distribution for slope/cliff
        // handling, sub-voxel pile surface, material stratification, craters,
        // splash ring, boulder tracks, impact cracks, plus formation removal
        // at both landing zone AND slab origin. All multi-chunk seam-aware.
        let mut pending_piles_for_event: Vec<PendingPilePlacement> = Vec::new();
        let pile_result_opt = if defer_pile {
            // Save data for later application — pile cells NOT placed yet.
            // Worker will call apply_pending_pile at the cinematic impact time.
            pending_piles_for_event.push(PendingPilePlacement {
                collapsed_voxels: collapsed_voxels.clone(),
                bb_min,
                bb_max,
                dominant_material,
                landing_offset,
            });
            None
        } else {
            let pr = crate::collapse_pile::place_collapse_pile(
                density_fields, config, &collapsed_voxels,
                bb_min, bb_max, dominant_material, landing_offset, chunk_size,
            );
            for k in &pr.affected_chunks {
                pile_chunks_set.insert(*k);
                affected_chunks_set.insert(*k);
            }
            Some(pr)
        };

        // ── Chained collapse hint (Tier 5I) ──
        // The pile's added weight may tip a marginal ceiling. Add the
        // chunks containing the settling-hint cells to the affected_chunks
        // set so the cascade picks them up. Only available when pile was
        // placed inline (defer_pile=false). Worker handles deferred
        // settling separately by re-running stress after impact.
        if let Some(pr) = pile_result_opt {
            for &(wx, wy, wz) in &pr.settling_dirty_cells {
                let key = world_to_chunk_local(wx, wy, wz, chunk_size).0;
                affected_chunks_set.insert(key);
            }
            let _ = (pr.written_cells, pr.dust_events, pr.fragments);
        }

        if false {
            let pile_seed: u64 = (center.0 as i64 as u64)
                .wrapping_mul(73856093)
                ^ (center.1 as i64 as u64).wrapping_mul(19349663)
                ^ (center.2 as i64 as u64).wrapping_mul(83492791);

            let crust_noise = Simplex3D::new(pile_seed);
            let boulder_noise = Simplex3D::new(pile_seed.wrapping_add(1));

            let cx_f = center.0;
            let cz_f = center.2;
            // Footprint radii. Pad by 1 voxel + floor at 1.5 so even tiny
            // single-voxel slabs still produce a small visible pile.
            let radius_x = ((bb_max.0 - bb_min.0) as f32 * 0.5 + 1.0).max(1.5);
            let radius_z = ((bb_max.2 - bb_min.2) as f32 * 0.5 + 1.0).max(1.5);
            let avg_radius = (radius_x + radius_z) * 0.5;
            let slab_thickness = (bb_max.1 - bb_min.1 + 1).max(1) as f32;

            let target_volume = (collapsed_voxels.len() as f32 * config.rubble_fill_ratio)
                .max(1.0);

            // Cone volume = (pi * R^2 * H) / 3 → H = 3V / (pi * R^2).
            let cone_volume_factor = std::f32::consts::PI * radius_x * radius_z / 3.0;
            let cone_h_raw = target_volume / cone_volume_factor.max(0.5);
            // Allow slightly taller than slab so wide flat slabs still pile up.
            let cone_h_cap = (slab_thickness * 1.2).max(2.0);
            let cone_h_max = cone_h_raw.clamp(1.0, cone_h_cap);
            let cone_h_int = cone_h_max.ceil() as i32;

            let floor_y = bb_min.1 - landing_offset;
            let mut placed = 0usize;

            // ── Pass 1: SEALED CONE BASE ──
            //
            // For each layer dy: compute disc radius, fill EVERY air cell in
            // that ellipse. No noise, no per-column logic, no breaks. The
            // cone is closed by construction.
            for dy in 0..cone_h_int {
                let y = floor_y + dy;
                // f = 0 at base, 1 at top. Floor at 0.05 so the very tip is
                // still half a voxel wide instead of an infinitesimal point.
                let f = (dy as f32 / cone_h_max).clamp(0.0, 1.0);
                let shrink = (1.0 - f).max(0.05);
                let rx = (radius_x * shrink).max(0.5);
                let rz = (radius_z * shrink).max(0.5);

                let x0 = (cx_f - rx).floor() as i32 - 1;
                let x1 = (cx_f + rx).ceil() as i32 + 1;
                let z0 = (cz_f - rz).floor() as i32 - 1;
                let z1 = (cz_f + rz).ceil() as i32 + 1;

                for x in x0..=x1 {
                    for z in z0..=z1 {
                        let nx = (x as f32 - cx_f) / rx;
                        let nz = (z as f32 - cz_f) / rz;
                        if nx * nx + nz * nz > 1.0 { continue; }

                        let (rkey, rlx, rly, rlz) =
                            world_to_chunk_local(x, y, z, chunk_size);
                        let is_air = density_fields
                            .get(&rkey)
                            .map(|df| !df.get(rlx, rly, rlz).material.is_solid())
                            .unwrap_or(false);
                        if !is_air { continue; }

                        if let Some(df) = density_fields.get_mut(&rkey) {
                            let sample = df.get_mut(rlx, rly, rlz);
                            sample.density = 1.0;
                            sample.material = dominant_material;
                        }
                        affected_chunks_set.insert(rkey);
                        placed += 1;
                    }
                }
            }

            // ── Pass 2: NOISE CRUST ──
            //
            // For each (x, z) inside the cone footprint, find the cone-top Y
            // for that column and stamp 0..=2 extra voxels above where simplex
            // noise is positive. Pure additive; never subtracts from Pass 1.
            const CRUST_MAX_EXTRA: i32 = 2;
            let xmin = (cx_f - radius_x).floor() as i32 - 1;
            let xmax = (cx_f + radius_x).ceil() as i32 + 1;
            let zmin = (cz_f - radius_z).floor() as i32 - 1;
            let zmax = (cz_f + radius_z).ceil() as i32 + 1;

            for x in xmin..=xmax {
                for z in zmin..=zmax {
                    let nx = (x as f32 - cx_f) / radius_x;
                    let nz = (z as f32 - cz_f) / radius_z;
                    let r2 = nx * nx + nz * nz;
                    if r2 > 1.0 { continue; }

                    // Cone-top Y for this column: r(y)/R = 1 - f, so the
                    // column top in voxels = (1 - sqrt(r2)) * cone_h_max.
                    let column_h = ((1.0 - r2.sqrt()) * cone_h_max).max(0.0);
                    let column_top_dy = column_h.floor() as i32;

                    let n_lo = crust_noise.sample(
                        x as f64 * 0.22, 0.0, z as f64 * 0.22,
                    ) as f32;
                    let n_hi = crust_noise.sample(
                        x as f64 * 0.55, 7.0, z as f64 * 0.55,
                    ) as f32;
                    let n = n_lo * 0.7 + n_hi * 0.4;
                    if n <= 0.0 { continue; }

                    let extra = ((n * (CRUST_MAX_EXTRA as f32 + 0.5)).round() as i32)
                        .clamp(0, CRUST_MAX_EXTRA);
                    if extra <= 0 { continue; }

                    for k in 1..=extra {
                        let y = floor_y + column_top_dy + k;
                        let (rkey, rlx, rly, rlz) =
                            world_to_chunk_local(x, y, z, chunk_size);
                        let is_air = density_fields
                            .get(&rkey)
                            .map(|df| !df.get(rlx, rly, rlz).material.is_solid())
                            .unwrap_or(false);
                        if !is_air { continue; }

                        if let Some(df) = density_fields.get_mut(&rkey) {
                            let sample = df.get_mut(rlx, rly, rlz);
                            sample.density = 1.0;
                            sample.material = dominant_material;
                        }
                        affected_chunks_set.insert(rkey);
                        placed += 1;
                    }
                }
            }

            // ── Pass 3: BOULDERS ──
            //
            // 2–8 half-buried spheres around the cone, sitting on the cone-top
            // for their (x, z). Edge noise so they aren't perfect spheres.
            let boulder_count = ((avg_radius * 0.6) as usize).clamp(2, 8);
            for i in 0..boulder_count {
                // Sweep angles around the cone axis; offset radially via noise
                // so they don't land on a perfect ring.
                let theta = (i as f32) * std::f32::consts::TAU / (boulder_count as f32);
                let radial_n = boulder_noise.sample(
                    (i as f64) * 1.31, 0.0, (i as f64) * 0.93,
                ) as f32;
                // Radial fraction in [0.2, 0.7] of the footprint.
                let radial_frac = 0.2 + (radial_n * 0.5 + 0.5) * 0.5;
                let bx = (cx_f + theta.cos() * radius_x * radial_frac).round() as i32;
                let bz = (cz_f + theta.sin() * radius_z * radial_frac).round() as i32;

                // Cone-top Y at this (bx, bz) so boulders sit ON the pile.
                let nx = (bx as f32 - cx_f) / radius_x;
                let nz = (bz as f32 - cz_f) / radius_z;
                let br2 = (nx * nx + nz * nz).min(1.0);
                let column_h = ((1.0 - br2.sqrt()) * cone_h_max).max(0.0);
                let by = floor_y + column_h.floor() as i32;

                let size_n = boulder_noise.sample(
                    bx as f64 * 0.41, by as f64 * 0.41, bz as f64 * 0.41,
                ) as f32;
                let radius = 1.5 + (size_n * 0.5 + 0.5) * 1.2; // 1.5..2.7

                // Half-bury so it looks settled.
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

                            // Edge noise so boulders aren't perfect spheres.
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
                            if wy_b < floor_y { continue; }

                            let (rkey, rlx, rly, rlz) = world_to_chunk_local(
                                wx_b, wy_b, wz_b, chunk_size,
                            );
                            let is_air = density_fields
                                .get(&rkey)
                                .map(|df| !df.get(rlx, rly, rlz).material.is_solid())
                                .unwrap_or(false);
                            if !is_air { continue; }

                            if let Some(df) = density_fields.get_mut(&rkey) {
                                let sample = df.get_mut(rlx, rly, rlz);
                                sample.density = 1.0;
                                sample.material = dominant_material;
                            }
                            affected_chunks_set.insert(rkey);
                            placed += 1;
                        }
                    }
                }
            }

            let _ = placed;
        }

        let slab = CollapseSlab {
            voxels: collapsed_voxels,
            bb_min,
            bb_max,
            center,
            landing_y,
            fall_distance: landing_offset,
            dominant_material,
        };

        events.push(CollapseEventV2 {
            slabs: vec![slab],
            affected_chunks: affected_chunks_set.into_iter().collect(),
            slab_chunks: slab_chunks_set.into_iter().collect(),
            pile_chunks: pile_chunks_set.into_iter().collect(),
            pending_piles: pending_piles_for_event,
            total_volume: region.len() as u32,
            center,
        });
    }

    // Apply BFS-halt HP damage to each strut that braced part of a slab.
    // Each blocked voxel = BFS_HALT_DAMAGE_SCALE HP off the strut. Broken
    // struts get cleared and pushed to `broken_out` so the caller can emit
    // StrutBroken events to UE.
    if halt_at_struts && !strut_halt_counts.is_empty() {
        for ((chunk_key, lx, ly, lz), count) in strut_halt_counts {
            let stype = support_fields
                .get(&chunk_key)
                .map(|sf| sf.get(lx, ly, lz))
                .unwrap_or(SupportType::None);
            if stype == SupportType::None { continue; }
            // Per-tier resistance (2026-08-23): the global rate times the tier's
            // own `damage_taken_scale`, via the shared helper. Resolved AFTER
            // `stype` — the type has to be known first, which is why the lookup
            // moved above the damage calc.
            let damage = bfs_halt_damage(stype, count);
            if let Some(sf) = support_fields.get_mut(&chunk_key) {
                if sf.damage_hp(lx, ly, lz, damage) {
                    sf.set(lx, ly, lz, SupportType::None);
                    broken_out.push(BrokenStrutEvent {
                        chunk: chunk_key, lx, ly, lz, support_type: stype,
                    });
                }
            }
        }
    }

    (events, hit_deadline)
}
