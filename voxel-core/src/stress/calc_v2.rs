//! V2 two-pass stress calculation (ground connectivity + load accumulation)
//! and the region-recalc entry points (v1 + v2 variants).
//!
//! Behavior-preserving split of the former `stress.rs` god file.

use std::collections::{HashMap, HashSet};

use crate::density::DensityField;

use super::config::StressConfig;
use super::events::{
    in_any_event, pack_classification, StressDirtyEvent, SOURCE_CROSS_SECTION,
    SOURCE_NONE, SOURCE_SPAN, SURFACE_CEILING, SURFACE_FLOOR, SURFACE_INTERIOR,
    SURFACE_THIN, SURFACE_WALL,
};
use super::types::{
    OverstressedVoxel, StressField, StressResult,
    SupportField, SupportScoreField, SupportType,
};
use super::calc::{
    accumulate_strut_load_at_voxel, apply_strut_load_damage, calc_voxel_stress,
    ground_connectivity_pass, measure_span_from_air, min_distance_to_air,
    sample_world, strut_relief_final_legacy, strut_relief_raw, world_to_chunk_local,
};

/// Same-chunk fast path for a face/below neighbor solidity sample inside the
/// Pass-2 classification loop.
///
/// When the neighbor's local coords all fall STRICTLY inside the current
/// chunk's `[0, cs-1]` grid, `world_to_chunk_local` resolves them right back to
/// `(cx,cy,cz)` — i.e. the `df` we already hold — so reading `df` directly is
/// bit-identical to `sample_world` while skipping the per-voxel chunk-key
/// re-hash (SipHash over a 12-byte tuple) + `density_fields` HashMap probe. Only
/// genuine cross-chunk neighbors (a voxel on a chunk face stepping out of
/// `[0, cs-1]`, including the shared `cs` overlap row on an unchanged axis) fall
/// through to `sample_world`. Returns `None` for an unloaded chunk exactly as
/// `sample_world` does, so callers' `None` handling is unchanged.
///
/// Interior voxels are the common case during the initial-load / zone-stream /
/// save-load storm (recalc runs over every solid voxel), so this elides ~7
/// HashMap probes per grounded voxel almost everywhere but the chunk shell.
#[inline]
fn neighbor_solid_same_chunk(
    df: &DensityField,
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    cs: usize,
    nlx: i32, nly: i32, nlz: i32, // neighbor local coords in the current chunk
    wnx: i32, wny: i32, wnz: i32, // neighbor world coords (for the fallback)
) -> Option<bool> {
    let cs_i = cs as i32;
    if nlx >= 0 && nlx < cs_i && nly >= 0 && nly < cs_i && nlz >= 0 && nlz < cs_i {
        Some(df.get(nlx as usize, nly as usize, nlz as usize).material.is_solid())
    } else {
        sample_world(density_fields, wnx, wny, wnz, cs).map(|(_, m)| m.is_solid())
    }
}

/// V2 stress calculation for a single voxel using precomputed ground connectivity.
pub fn calc_voxel_stress_v2(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    support_scores: &HashMap<(i32, i32, i32), SupportScoreField>,
    config: &StressConfig,
    wx: i32, wy: i32, wz: i32,
    chunk_size: usize,
) -> (f32, u8) {
    // Only solid voxels have stress
    let mat = match sample_world(density_fields, wx, wy, wz, chunk_size) {
        Some((_, m)) if m.is_solid() => m,
        _ => return (0.0, pack_classification(SURFACE_INTERIOR, SOURCE_NONE)),
    };

    let hardness = config.material_hardness[mat as u8 as usize];
    if hardness <= 0.0 {
        return (0.0, pack_classification(SURFACE_INTERIOR, SOURCE_NONE));
    }

    // Get support score from ground connectivity pass
    let (key, lx, ly, lz) = world_to_chunk_local(wx, wy, wz, chunk_size);
    let support_score = support_scores
        .get(&key)
        .map(|sf| sf.get(lx, ly, lz))
        .unwrap_or(1.0); // Unloaded = assume grounded

    // Unsupported factor: 0.0 for fully grounded, 1.0 for floating.
    let unsupported = (1.0 - support_score).max(0.0);

    // Floor protection: solid below AND well-supported by the flood = stable floor.
    // Thick ceiling rock has solid below but LOW flood score (air gap broke chain) → NOT protected.
    // Floor rock has solid below AND HIGH flood score (connected to surface) → protected.
    // `below_solid` is reused by the surface classification below — computing the
    // `(wx, wy-1, wz)` sample once saves a redundant chunk-key resolve + HashMap
    // probe per voxel that reaches the span-stress path.
    let below_solid = sample_world(density_fields, wx, wy - 1, wz, chunk_size)
        .map(|(_, m)| m.is_solid())
        .unwrap_or(true);
    if below_solid && support_score >= 0.2 {
        return (0.0, pack_classification(SURFACE_FLOOR, SOURCE_NONE));
    }

    // Distance-to-air decay: stress attenuates as we go deeper into rock.
    // Surface voxels (1 cell from air) get full stress, deep interior gets none.
    // This prevents the span search from producing stress on voxels buried in solid rock
    // where the concept of "unsupported span" doesn't physically apply.
    let air_dist = min_distance_to_air(density_fields, wx, wy, wz, chunk_size, 2);
    let air_decay = if air_dist <= 1 {
        1.0  // At the cave surface: full stress
    } else if air_dist == 2 {
        0.5  // One cell deep: half stress
    } else {
        0.0  // 3+ cells deep: no surface stress
    };

    // Deep interior shortcut: no stress, classify as interior
    if air_decay <= 0.0 {
        return (0.0, pack_classification(SURFACE_INTERIOR, SOURCE_NONE));
    }

    // Track individual stress components for classification
    let mut raw_stress = 0.0f32;

    // Span penalty: measures widest unsupported air gap this voxel is exposed to.
    // Searches from each air face-neighbor laterally through air to find walls.
    // Near walls = low span = safe. Center of wide ceiling = high span = danger.
    let span_dist = measure_span_from_air(
        density_fields, wx, wy, wz, chunk_size, 20,
    );
    let span_stress = if span_dist > config.min_safe_span {
        (span_dist - config.min_safe_span) as f32 * config.span_weight * unsupported * air_decay
    } else { 0.0 };
    raw_stress += span_stress;

    // Cross-section penalty
    let face_offsets: [(i32, i32, i32); 6] = [
        (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1),
    ];
    let mut air_neighbors = 0u32;
    for &(dx, dy, dz) in &face_offsets {
        match sample_world(density_fields, wx + dx, wy + dy, wz + dz, chunk_size) {
            Some((_, m)) if !m.is_solid() => air_neighbors += 1,
            None => {}
            _ => {}
        }
    }
    let xsec_stress = if air_neighbors >= config.cross_section_min_faces {
        (air_neighbors - 1) as f32 * config.cross_section_weight
    } else { 0.0 };
    raw_stress += xsec_stress;

    // NOTE: strut relief is no longer subtracted here. The recalc loops
    // subtract `strut_relief_final_v2()` from this function's result so that
    // (a) relief surplus survives past the zero-clamp below as negative
    // stored stress, offsetting the painted overlay at `effective()` read
    // time, and (b) relief also reaches voxels this function early-returns
    // for (grounded floors, deep interior) — exactly where map-authored
    // painted stress usually lives. `stored - relief` distributes to the
    // same value the old inline subtraction produced wherever both applied.

    // Depth pressure: deeper rock is under more overburden compression.
    // At surface: 1.0x. At depth 100: 2.0x. At depth 200: 3.0x.
    // This makes narrow tunnels dangerous at depth even when span is safe.
    let depth = (config.surface_y - wy).max(0) as f32;
    let depth_factor = 1.0 + depth / config.depth_pressure_scale;

    let final_stress = (raw_stress.max(0.0) * depth_factor) / hardness;

    // Classify surface type using BOTH local air neighbors AND air_dist.
    // A voxel with air_neighbors==0 but air_dist<=4 is near the surface and may have
    // stress — classify by geometry (below_solid, computed once above) rather
    // than defaulting to INTERIOR.
    let surface_type = if air_neighbors >= 4 {
        SURFACE_THIN       // Stalactite/thin column (4+ air faces)
    } else if !below_solid {
        SURFACE_CEILING    // Air directly below
    } else if air_neighbors == 0 && final_stress <= 0.001 {
        SURFACE_INTERIOR   // Fully enclosed AND no stress = truly interior
    } else if below_solid && support_score >= 0.2 {
        SURFACE_FLOOR      // Solid below + moderately supported
    } else {
        SURFACE_WALL       // Near surface, solid below = wall/pillar
    };

    // Dominant stress source (gravity + overhang removed — span is primary)
    let dominant_source = if final_stress <= 0.001 {
        SOURCE_NONE
    } else if xsec_stress >= span_stress {
        SOURCE_CROSS_SECTION
    } else {
        SOURCE_SPAN
    };

    (final_stress, pack_classification(surface_type, dominant_source))
}

/// Strut relief at a voxel in FINAL v2 stress units — the raw sweep run
/// through the same depth-pressure × material-hardness transform as span
/// stress, so `calc_voxel_stress_v2() - strut_relief_final_v2()` equals the
/// old inline-subtraction result wherever both applied (distributivity).
/// Returns 0 for air / zero-hardness voxels.
///
/// The difference `stress - relief` may go NEGATIVE: that surplus is stored
/// in the stress field on purpose so `effective = stress + painted` lets
/// struts offset map-authored painted stress. Every effective-stress read
/// clamps the SUM at zero (`StressField::effective`), never the parts.
pub fn strut_relief_final_v2(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    support_fields: &HashMap<(i32, i32, i32), SupportField>,
    config: &StressConfig,
    wx: i32, wy: i32, wz: i32,
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
    let raw = strut_relief_raw(support_fields, wx, wy, wz, chunk_size);
    if raw <= 0.0 {
        return 0.0;
    }
    let depth = (config.surface_y - wy).max(0) as f32;
    let depth_factor = 1.0 + depth / config.depth_pressure_scale;
    (raw * depth_factor) / hardness
}

/// V2 stress recalculation: runs ground connectivity pass then per-voxel stress.
/// Operates on a set of dirty chunks (and their neighborhoods).
/// Used by overlay preview (V/C key) which needs full-chunk recalc.
pub fn recalc_stress_region_v2(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    stress_fields: &mut HashMap<(i32, i32, i32), StressField>,
    support_fields: &HashMap<(i32, i32, i32), SupportField>,
    config: &StressConfig,
    dirty_chunks: &[(i32, i32, i32)],
    chunk_size: usize,
) -> StressResult {
    recalc_stress_region_v2_filtered(
        density_fields, stress_fields, support_fields, config,
        dirty_chunks, &[], chunk_size,
    )
}

/// V2 stress recalculation with optional position-based filtering.
/// If `events` is non-empty, only voxels within any event's radius are recalculated.
/// If `events` is empty, all voxels in `dirty_chunks` are recalculated (full mode).
///
/// NOTE: this signature is the legacy entry — it cannot damage strut HP because
/// it takes `support_fields` as `&` (read-only). Callers that need HP decay
/// (the live worker stress pass) should call `recalc_stress_region_v2_with_load_decay`
/// below, which takes `&mut support_fields`. This wrapper exists for overlay /
/// preview / debug / test sites that just want stress numbers.
pub fn recalc_stress_region_v2_filtered(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    stress_fields: &mut HashMap<(i32, i32, i32), StressField>,
    support_fields: &HashMap<(i32, i32, i32), SupportField>,
    config: &StressConfig,
    dirty_chunks: &[(i32, i32, i32)],
    events: &[StressDirtyEvent],
    chunk_size: usize,
) -> StressResult {
    // Pass 1: ground connectivity on dirty chunks + neighbors
    let support_scores = ground_connectivity_pass(density_fields, dirty_chunks, chunk_size, config);

    let mut overstressed = Vec::new();
    let mut affected_chunks = HashSet::new();

    // Pass 2: calculate stress for voxels near surfaces in dirty chunks.
    // Deep interior voxels (fully surrounded by grounded solid) are skipped
    // for performance. Body extracted to `recalc_chunk_stress_voxels` so the
    // FFI's region-level VFX pre-population can run chunks in parallel.
    for &key in dirty_chunks {
        let wrote = recalc_chunk_stress_voxels(
            density_fields,
            support_fields,
            &support_scores,
            config,
            key,
            chunk_size,
            stress_fields.get_mut(&key),
            events,
            &mut overstressed,
        );
        if wrote {
            affected_chunks.insert(key);
        }
    }

    StressResult {
        overstressed,
        affected_chunks: affected_chunks.into_iter().collect(),
        broken_struts: Vec::new(), // read-only entry — no HP damage tracked
    }
}

/// Full (optionally event-filtered) stress recompute of ONE chunk's voxels —
/// the extracted per-chunk body of [`recalc_stress_region_v2_filtered`].
///
/// Extracted so the FFI's region-level VFX stress pre-population can run the
/// per-chunk pass for many chunks IN PARALLEL: every map input is read-only
/// and each call writes only its own `chunk_sf` + `overstressed`, so calls
/// for distinct chunks are data-race-free. `support_scores` must come from a
/// [`ground_connectivity_pass`] whose dirty set included `chunk`.
///
/// Returns true if any stress voxel was written (the caller's
/// "affected chunk" signal).
pub fn recalc_chunk_stress_voxels(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    support_fields: &HashMap<(i32, i32, i32), SupportField>,
    support_scores: &HashMap<(i32, i32, i32), SupportScoreField>,
    config: &StressConfig,
    chunk: (i32, i32, i32),
    chunk_size: usize,
    mut chunk_sf: Option<&mut StressField>,
    events: &[StressDirtyEvent],
    overstressed: &mut Vec<OverstressedVoxel>,
) -> bool {
    let (cx, cy, cz) = chunk;
    let df = match density_fields.get(&chunk) {
        Some(d) => d,
        None => return false,
    };
    let use_filter = !events.is_empty();
    let cs = chunk_size;
    let grid_size = cs + 1;

    // Hoist the per-chunk map lookups out of the gs³ voxel loop: the chunk
    // key is constant for the whole triple loop, so re-hashing it (SipHash
    // over a 12-byte tuple) ~30k× per chunk was pure overhead. Fetch the
    // support-score field once; the stress field arrives pre-fetched.
    let chunk_support = support_scores.get(&chunk);
    let mut wrote = false;

    for z in 0..grid_size {
        for y in 0..grid_size {
            for x in 0..grid_size {
                if !df.get(x, y, z).material.is_solid() {
                    if let Some(sf) = chunk_sf.as_deref_mut() {
                        sf.set(x, y, z, 0.0);
                        sf.set_class(x, y, z, 0); // Air = no classification
                    }
                    continue;
                }

                let wx = cx * cs as i32 + x as i32;
                let wy = cy * cs as i32 + y as i32;
                let wz = cz * cs as i32 + z as i32;

                // Position filter: skip voxels outside all mine event radii.
                // Their existing stress stays untouched — no phantom collapses.
                if use_filter && !in_any_event(events, wx, wy, wz) {
                    continue;
                }

                // Interior skip: fully grounded voxels get 0 stress but still classified.
                let my_support = chunk_support
                    .map(|sf| sf.get(x, y, z))
                    .unwrap_or(1.0);
                if my_support >= config.ground_threshold {
                    // Classify: is this a floor or deep interior? Same-chunk
                    // neighbors read `df` directly (see neighbor_solid_same_chunk).
                    let (xi, yi, zi) = (x as i32, y as i32, z as i32);
                    let below_solid = neighbor_solid_same_chunk(
                        df, density_fields, cs, xi, yi - 1, zi, wx, wy - 1, wz,
                    ).unwrap_or(true);
                    // Count air neighbors for wall detection
                    let mut air_n = 0u8;
                    for &(dx, dy, dz) in &[(1i32,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)] {
                        if neighbor_solid_same_chunk(
                            df, density_fields, cs, xi+dx, yi+dy, zi+dz, wx+dx, wy+dy, wz+dz,
                        ) == Some(false) {
                            air_n += 1;
                        }
                    }
                    let stype = if air_n == 0 { SURFACE_INTERIOR }
                        else if below_solid { SURFACE_FLOOR }
                        else { SURFACE_WALL };
                    // Grounded rock has zero organic stress, but the map
                    // editor may have painted stress onto it — store the
                    // strut relief surplus (negative) so nearby struts
                    // offset that painted load at effective() read time.
                    let painted = chunk_sf
                        .as_deref()
                        .map(|sf| sf.painted(x, y, z))
                        .unwrap_or(0.0);
                    let grounded_stress = if painted > 0.001 {
                        -strut_relief_final_v2(
                            density_fields, support_fields, config, wx, wy, wz, cs,
                        )
                    } else {
                        0.0
                    };
                    if let Some(sf) = chunk_sf.as_deref_mut() {
                        sf.set(x, y, z, grounded_stress);
                        sf.set_class(x, y, z, pack_classification(stype, SOURCE_NONE));
                    }
                    continue;
                }

                let (stress, classification) = calc_voxel_stress_v2(
                    density_fields, support_scores,
                    config, wx, wy, wz, cs,
                );

                // Painted overlay (creative-mode PaintStress brush) is
                // captured BEFORE the set, since set() doesn't touch it.
                let painted = chunk_sf
                    .as_deref()
                    .map(|sf| sf.painted(x, y, z))
                    .unwrap_or(0.0);
                // Strut relief applies after the calc's zero-clamp; the
                // surplus (negative stored stress) is what lets struts
                // offset painted stress. Skip the sweep when there is
                // nothing to relieve.
                let stress = if stress > 0.001 || painted > 0.001 {
                    stress - strut_relief_final_v2(
                        density_fields, support_fields, config, wx, wy, wz, cs,
                    )
                } else {
                    stress
                };
                if let Some(sf) = chunk_sf.as_deref_mut() {
                    sf.set(x, y, z, stress);
                    sf.set_class(x, y, z, classification);
                    wrote = true;
                }

                let eff = stress + painted;
                if eff >= 1.0 {
                    overstressed.push(OverstressedVoxel {
                        world_x: wx, world_y: wy, world_z: wz, stress: eff,
                    });
                }
            }
        }
    }

    wrote
}

/// V2 recalc that ALSO tracks per-strut load and damages HP / clears
/// broken struts. The hot live-mining + sleep path. Returns the standard
/// stress result with `broken_struts` populated when any HP hit 0.
///
/// Callers that don't need HP decay (overlay previews, debug, tests) should
/// use `recalc_stress_region_v2_filtered` above which takes `&` not `&mut`.
pub fn recalc_stress_region_v2_with_load_decay(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    stress_fields: &mut HashMap<(i32, i32, i32), StressField>,
    support_fields: &mut HashMap<(i32, i32, i32), SupportField>,
    config: &StressConfig,
    dirty_chunks: &[(i32, i32, i32)],
    events: &[StressDirtyEvent],
    chunk_size: usize,
) -> StressResult {
    let use_filter = !events.is_empty();
    let support_scores = ground_connectivity_pass(density_fields, dirty_chunks, chunk_size, config);

    let cs = chunk_size;
    let grid_size = cs + 1;
    let mut overstressed = Vec::new();
    let mut affected_chunks = HashSet::new();
    let mut loads: std::collections::HashMap<((i32, i32, i32), usize, usize, usize), f32> =
        std::collections::HashMap::new();

    for &(cx, cy, cz) in dirty_chunks {
        let df = match density_fields.get(&(cx, cy, cz)) {
            Some(d) => d,
            None => continue,
        };

        // Hoist the per-chunk map lookups out of the gs³ voxel loop (see
        // recalc_stress_region_v2_filtered for the rationale). The chunk key is
        // loop-invariant; re-hashing it per voxel was pure overhead.
        let chunk_support = support_scores.get(&(cx, cy, cz));
        let mut chunk_sf = stress_fields.get_mut(&(cx, cy, cz));

        for z in 0..grid_size {
            for y in 0..grid_size {
                for x in 0..grid_size {
                    if !df.get(x, y, z).material.is_solid() {
                        if let Some(sf) = chunk_sf.as_deref_mut() {
                            sf.set(x, y, z, 0.0);
                            sf.set_class(x, y, z, 0);
                        }
                        continue;
                    }

                    let wx = cx * cs as i32 + x as i32;
                    let wy = cy * cs as i32 + y as i32;
                    let wz = cz * cs as i32 + z as i32;

                    if use_filter && !in_any_event(events, wx, wy, wz) { continue; }

                    let my_support = chunk_support
                        .map(|sf| sf.get(x, y, z))
                        .unwrap_or(1.0);
                    if my_support >= config.ground_threshold {
                        // Same-chunk neighbors read `df` directly (see
                        // neighbor_solid_same_chunk) — skips ~7 HashMap probes
                        // per grounded voxel away from the chunk shell.
                        let (xi, yi, zi) = (x as i32, y as i32, z as i32);
                        let below_solid = neighbor_solid_same_chunk(
                            df, density_fields, cs, xi, yi - 1, zi, wx, wy - 1, wz,
                        ).unwrap_or(true);
                        let mut air_n = 0u8;
                        for &(dx, dy, dz) in &[(1i32,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)] {
                            if neighbor_solid_same_chunk(
                                df, density_fields, cs, xi+dx, yi+dy, zi+dz, wx+dx, wy+dy, wz+dz,
                            ) == Some(false) {
                                air_n += 1;
                            }
                        }
                        let stype = if air_n == 0 { SURFACE_INTERIOR }
                            else if below_solid { SURFACE_FLOOR }
                            else { SURFACE_WALL };
                        // Grounded rock: store the strut relief surplus
                        // (negative) when painted stress is present so struts
                        // offset the painted load at effective() read time.
                        let painted = chunk_sf
                            .as_deref()
                            .map(|sf| sf.painted(x, y, z))
                            .unwrap_or(0.0);
                        let grounded_stress = if painted > 0.001 {
                            -strut_relief_final_v2(
                                density_fields, support_fields, config, wx, wy, wz, cs,
                            )
                        } else {
                            0.0
                        };
                        if let Some(sf) = chunk_sf.as_deref_mut() {
                            sf.set(x, y, z, grounded_stress);
                            sf.set_class(x, y, z, pack_classification(stype, SOURCE_NONE));
                        }
                        // Struts bracing a painted grounded region are bearing
                        // that painted load — they decay for it, same as the
                        // span path below.
                        if painted > 0.001 {
                            accumulate_strut_load_at_voxel(support_fields, &mut loads, wx, wy, wz, cs);
                        }
                        continue;
                    }

                    let (stress, classification) = calc_voxel_stress_v2(
                        density_fields, &support_scores,
                        config, wx, wy, wz, cs,
                    );

                    // Painted overlay read BEFORE relief: both decide whether
                    // the strut sweep below has anything to do.
                    let painted = chunk_sf
                        .as_deref()
                        .map(|sf| sf.painted(x, y, z))
                        .unwrap_or(0.0);
                    // Strut relief applies after the calc's zero-clamp; the
                    // surplus (negative stored stress) offsets painted stress.
                    let stress = if stress > 0.001 || painted > 0.001 {
                        stress - strut_relief_final_v2(
                            density_fields, support_fields, config, wx, wy, wz, cs,
                        )
                    } else {
                        stress
                    };

                    // Record this voxel's incoming strut load so the post-pass
                    // HP-decay step knows which struts bore the weight. Only
                    // count load when the voxel had non-zero residual stress —
                    // a strut sitting in a stable region (test fixtures,
                    // grounded rock with no span) shouldn't wear down just
                    // by existing. Painted stress also counts so creative
                    // brushes can burn struts down deliberately. `stress` here
                    // is post-relief, matching the stored value the old code
                    // tested.
                    if stress > 0.001 || painted > 0.001 {
                        accumulate_strut_load_at_voxel(support_fields, &mut loads, wx, wy, wz, cs);
                    }

                    if let Some(sf) = chunk_sf.as_deref_mut() {
                        sf.set(x, y, z, stress);
                        sf.set_class(x, y, z, classification);
                        affected_chunks.insert((cx, cy, cz));
                    }

                    let eff = stress + painted;
                    if eff >= 1.0 {
                        overstressed.push(OverstressedVoxel {
                            world_x: wx, world_y: wy, world_z: wz, stress: eff,
                        });
                    }
                }
            }
        }
    }

    // Apply HP damage based on accumulated loads. Broken struts are cleared
    // from the support field so the *next* stress recalc (caller-driven) sees
    // the new world state.
    let broken = apply_strut_load_damage(support_fields, &loads);
    for ev in &broken {
        if let Some(sf) = support_fields.get_mut(&ev.chunk) {
            sf.set(ev.lx, ev.ly, ev.lz, SupportType::None);
            affected_chunks.insert(ev.chunk);
        }
    }

    StressResult {
        overstressed,
        affected_chunks: affected_chunks.into_iter().collect(),
        broken_struts: broken,
    }
}

/// Recalculate stress in a region around a changed world position.
/// Returns the list of overstressed voxels and affected chunks.
pub fn recalc_stress_region(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    stress_fields: &mut HashMap<(i32, i32, i32), StressField>,
    support_fields: &HashMap<(i32, i32, i32), SupportField>,
    config: &StressConfig,
    changed_world_pos: (i32, i32, i32),
    radius: u32,
    chunk_size: usize,
) -> StressResult {
    let (cwx, cwy, cwz) = changed_world_pos;
    let r = radius as i32;
    let mut overstressed = Vec::new();
    let mut affected_chunks = HashSet::new();

    for dz in -r..=r {
        for dy in -r..=r {
            for dx in -r..=r {
                let wx = cwx + dx;
                let wy = cwy + dy;
                let wz = cwz + dz;

                let stress = calc_voxel_stress(
                    density_fields, config, wx, wy, wz, chunk_size,
                );

                // Store stress value and fold in the painted overlay before the
                // overstressed test so creative-mode painted regions can drive
                // collapses just like organic geological stress.
                let (key, lx, ly, lz) = world_to_chunk_local(wx, wy, wz, chunk_size);
                let painted = stress_fields
                    .get(&key)
                    .map(|sf| sf.painted(lx, ly, lz))
                    .unwrap_or(0.0);
                // Strut relief after the calc's zero-clamp — surplus goes
                // negative and offsets painted stress at effective() reads.
                let stress = if stress > 0.001 || painted > 0.001 {
                    stress - strut_relief_final_legacy(
                        density_fields, support_fields, config, wx, wy, wz, chunk_size,
                    )
                } else {
                    stress
                };
                if let Some(sf) = stress_fields.get_mut(&key) {
                    sf.set(lx, ly, lz, stress);
                    affected_chunks.insert(key);
                }

                let eff = stress + painted;
                if eff >= 1.0 {
                    // Verify this is actually a solid voxel
                    if let Some((_, mat)) = sample_world(density_fields, wx, wy, wz, chunk_size) {
                        if mat.is_solid() {
                            overstressed.push(OverstressedVoxel {
                                world_x: wx,
                                world_y: wy,
                                world_z: wz,
                                stress: eff,
                            });
                        }
                    }
                }
            }
        }
    }

    StressResult {
        overstressed,
        affected_chunks: affected_chunks.into_iter().collect(),
        broken_struts: Vec::new(),
    }
}
