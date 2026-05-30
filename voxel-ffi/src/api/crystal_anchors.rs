//! Auto-split from the former monolithic `api.rs`. See `mod.rs`.
#[allow(unused_imports)]
use std::ffi::{c_char, c_void, CString};
#[allow(unused_imports)]
use std::ptr;
#[allow(unused_imports)]
use crate::engine::VoxelEngine;
#[allow(unused_imports)]
use crate::types::*;
use super::helpers::rust_pos_to_ue;

// ─── Crystal Growth Bridge (Crystal Anchor) FFI ─────────────────────────────
//
// Anchor placement & query API. The growth itself happens during deep sleep
// (Phase 3 wires that into the sleep handler) and exposes the grown list via
// `voxel_request_list_grown_crystal_bridges`.


/// Place a Crystal Anchor at the given UE world position. Implements the
/// state machine described in `crystal_anchors.rs`. Returns 0 if the engine
/// pointer is null, 1 otherwise — actual outcome is in `out_result.error_code`.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_place_crystal_anchor(
    engine: *mut c_void,
    ue_x: f32,
    ue_y: f32,
    ue_z: f32,
    ue_normal_x: f32,
    ue_normal_y: f32,
    ue_normal_z: f32,
    out_result: *mut FfiCrystalAnchorResult,
) -> u32 {
    if engine.is_null() || out_result.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let ws = engine.get_world_scale();
    let pos_rust = crate::convert::from_ue_world_pos(ue_x, ue_y, ue_z, ws);
    let normal_rust = crate::convert::from_ue_normal(ue_normal_x, ue_normal_y, ue_normal_z);

    let mut mgr = engine.crystal_anchors.lock().unwrap();
    let res = mgr.place_anchor(pos_rust, normal_rust);

    *out_result = FfiCrystalAnchorResult {
        error_code: res.error as u8,
        _padding: [0; 3],
        anchor_id: res.anchor_id,
        partner_id: res.partner_id,
        pair_token: res.pair_token,
        pair_completed: if res.pair_completed { 1 } else { 0 },
        _padding2: [0; 7],
    };
    1
}

/// Cancel an unpaired or paired anchor by id. If paired, both partners are
/// removed. Returns 1 if the anchor was found and removed, 0 otherwise.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_cancel_crystal_anchor(
    engine: *mut c_void,
    anchor_id: u64,
) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let mut mgr = engine.crystal_anchors.lock().unwrap();
    if mgr.cancel_anchor(anchor_id) {
        1
    } else {
        0
    }
}

/// Cancel the nearest unpaired anchor within `max_dist_ue` UE units of the
/// player's world position. Returns the cancelled anchor's id (0 if none).
#[no_mangle]
pub unsafe extern "C" fn voxel_request_cancel_nearest_crystal_anchor(
    engine: *mut c_void,
    player_ue_x: f32,
    player_ue_y: f32,
    player_ue_z: f32,
    max_dist_ue: f32,
) -> u64 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let ws = engine.get_world_scale();
    let pos_rust = crate::convert::from_ue_world_pos(player_ue_x, player_ue_y, player_ue_z, ws);
    let max_dist_voxels = max_dist_ue / ws;

    let mut mgr = engine.crystal_anchors.lock().unwrap();
    let nearest = mgr.nearest_unpaired(pos_rust, max_dist_voxels);
    match nearest {
        Some(id) if mgr.cancel_anchor(id) => id,
        _ => 0,
    }
}

/// Fill `out_buf` (capacity `out_capacity`) with up to N pending pairs, and
/// write the actual count to `out_count`. If `out_buf` is null, only the
/// count is written — caller can use this to size their buffer.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_list_pending_crystal_pairs(
    engine: *mut c_void,
    out_buf: *mut FfiCrystalBridgePair,
    out_capacity: u32,
    out_count: *mut u32,
) -> u32 {
    if engine.is_null() || out_count.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let ws = engine.get_world_scale();
    let mgr = engine.crystal_anchors.lock().unwrap();
    let pairs = mgr.list_pending_pairs();
    *out_count = pairs.len() as u32;

    if out_buf.is_null() {
        return 1;
    }
    let cap = out_capacity as usize;
    let copy_n = pairs.len().min(cap);
    let buf_slice = std::slice::from_raw_parts_mut(out_buf, copy_n);
    for (i, p) in pairs.iter().take(copy_n).enumerate() {
        buf_slice[i] = FfiCrystalBridgePair {
            pair_token: p.pair_token,
            anchor_a_id: p.anchor_a_id,
            anchor_b_id: p.anchor_b_id,
            anchor_a_pos_ue: rust_pos_to_ue(p.anchor_a_pos_rust, ws),
            anchor_b_pos_ue: rust_pos_to_ue(p.anchor_b_pos_rust, ws),
            midpoint_ue: rust_pos_to_ue(p.midpoint_rust, ws),
        };
    }
    1
}

/// Augment the cached morph manifest with "synthesize growth" entries for
/// the given chunks. Chunks already in the manifest are left alone (their
/// recorded voxel_changes win). New chunks get a stub ChunkDelta with
/// `synthesize_growth = true` plus the supplied growth sources — the morph
/// step procedurally animates them rising from air to their current state.
///
/// `sources_ue` (UE world coords) parameterize the reveal: each voxel's
/// spread = min-distance-to-any-source / max_dist, normalized [0,1].
/// Pass 2 sources (anchor A + anchor B) for bridges, 1 source (chunk
/// center) for radial reveal of other POIs, or 0 for y-axis fallback.
///
/// Called by UE before each POI play so the play's 3×3×3 showcase block
/// always animates with a pretty reveal pattern, even for POIs whose chunks
/// weren't sleep-affected (crystal bridges, pre-existing lava chambers).
///
/// Returns the number of new entries added (chunks that were not already
/// in the manifest). Returns 0 if no manifest is cached yet.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_augment_morph_synthesize(
    engine: *mut c_void,
    chunks_ue: *const FfiChunkCoord,
    chunk_count: u32,
    sources_ue: *const FfiVec3,
    source_count: u32,
    max_dist_ue: f32,
) -> u32 {
    if engine.is_null() || chunks_ue.is_null() || chunk_count == 0 {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let chunks_slice = std::slice::from_raw_parts(chunks_ue, chunk_count as usize);
    let sources_slice: &[FfiVec3] = if !sources_ue.is_null() && source_count > 0 {
        std::slice::from_raw_parts(sources_ue, source_count as usize)
    } else {
        &[]
    };
    engine.augment_morph_synthesize_ue_chunks(chunks_slice, sources_slice, max_dist_ue)
}

/// Fill `out_buf` with up to N top POIs (lava / water / stress / bridges)
/// from the continuous tracker. Unified ranking across all kinds — bridges
/// from the anchor manager are merged in at query time and compete on the
/// same score scale.
///
/// `out_capacity` is the buffer size. `out_count` is set to the actual
/// number of POIs available (may exceed capacity — caller can re-call with
/// a larger buffer if needed).
#[no_mangle]
pub unsafe extern "C" fn voxel_request_list_top_pois(
    engine: *mut c_void,
    out_buf: *mut FfiPoi,
    out_capacity: u32,
    out_count: *mut u32,
) -> u32 {
    if engine.is_null() || out_count.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let ws = engine.get_world_scale();
    let chunk_size = engine.chunk_size();

    // Block 1: try the new WorldMemory adapter FIRST. If it returns a
    // non-empty result, use it (strictly better data: sub-voxel centroids
    // + accurate extent radii). Falls back to the legacy POI tracker if
    // WorldMemory is empty (cold start before first drift tick).
    const CANDIDATE_LIMIT: usize = 64;
    let wm_pois = engine.world_memory.legacy_top_k_pois(CANDIDATE_LIMIT, false);
    let mut candidates: Vec<crate::poi_scanner::Poi> = if !wm_pois.is_empty() {
        // Project WorldMemory's LegacyPoi → poi_scanner::Poi. Same numeric
        // kind layout (0..=6) and Rust-coord centroids.
        wm_pois
            .into_iter()
            .map(|p| {
                let kind = match p.kind {
                    0 => crate::poi_scanner::PoiKind::Lava,
                    1 => crate::poi_scanner::PoiKind::Water,
                    2 => crate::poi_scanner::PoiKind::Stress,
                    3 => crate::poi_scanner::PoiKind::Bridge,
                    4 => crate::poi_scanner::PoiKind::CeilingDome,
                    5 => crate::poi_scanner::PoiKind::Chokepoint,
                    6 => crate::poi_scanner::PoiKind::WallNiche,
                    _ => crate::poi_scanner::PoiKind::Lava, // shouldn't happen
                };
                crate::poi_scanner::Poi {
                    kind,
                    score: p.score,
                    chunk_coord: (p.chunk_rust[0], p.chunk_rust[1], p.chunk_rust[2]),
                    center_world_rust: glam::Vec3::new(
                        p.centroid_rust[0],
                        p.centroid_rust[1],
                        p.centroid_rust[2],
                    ),
                    extent_radius_voxels: p.extent_radius_voxels,
                }
            })
            .collect()
    } else {
        // Cold-start fallback to the legacy tracker. UE migration in
        // Block 2 retires this branch.
        crate::poi_tracker::read_top_k(&engine.poi_tracker, CANDIDATE_LIMIT, chunk_size)
    };

    // Merge in grown crystal bridges with their own scoring (matches the
    // scanner's bridge math — see poi_scanner::BRIDGE_BASELINE_SCORE).
    {
        let mgr = engine.crystal_anchors.lock().unwrap();
        let cs_f = chunk_size as f32;
        for pair in mgr.list_grown_pairs() {
            let dist = (pair.anchor_b_pos_rust - pair.anchor_a_pos_rust).length();
            let score = crate::poi_scanner::BRIDGE_BASELINE_SCORE
                + dist * crate::poi_scanner::BRIDGE_LENGTH_BONUS_PER_VOXEL;
            let cx = (pair.midpoint_rust.x / cs_f).floor() as i32;
            let cy = (pair.midpoint_rust.y / cs_f).floor() as i32;
            let cz = (pair.midpoint_rust.z / cs_f).floor() as i32;
            candidates.push(crate::poi_scanner::Poi {
                kind: crate::poi_scanner::PoiKind::Bridge,
                score,
                chunk_coord: (cx, cy, cz),
                center_world_rust: pair.midpoint_rust,
                extent_radius_voxels: dist * 0.5,
            });
        }
    }

    // Re-sort + truncate. This is the final ranking the montage will use.
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    *out_count = candidates.len() as u32;
    if out_buf.is_null() {
        return 1;
    }
    let cap = out_capacity as usize;
    let copy_n = candidates.len().min(cap);
    let buf_slice = std::slice::from_raw_parts_mut(out_buf, copy_n);
    for (i, poi) in candidates.iter().take(copy_n).enumerate() {
        let (ue_cx, ue_cy, ue_cz) = crate::convert::rust_chunk_to_ue(
            poi.chunk_coord.0,
            poi.chunk_coord.1,
            poi.chunk_coord.2,
        );
        buf_slice[i] = FfiPoi {
            kind: poi.kind as u8,
            _padding: [0; 3],
            score: poi.score,
            chunk_coord_ue: FfiChunkCoord { x: ue_cx, y: ue_cy, z: ue_cz },
            center_ue: FfiVec3 {
                x: poi.center_world_rust.x * ws,
                y: -poi.center_world_rust.z * ws,
                z: poi.center_world_rust.y * ws,
            },
            extent_radius_ue: poi.extent_radius_voxels * ws,
        };
    }
    1
}

/// Fill `out_buf` with up to N grown bridges (state == Grown), and write the
/// actual count to `out_count`. Used by the sleep montage POI rotation.
/// (Phase 3 actually flips pairs into Grown; in Phase 2 this returns 0.)
#[no_mangle]
pub unsafe extern "C" fn voxel_request_list_grown_crystal_bridges(
    engine: *mut c_void,
    out_buf: *mut FfiCrystalBridgePair,
    out_capacity: u32,
    out_count: *mut u32,
) -> u32 {
    if engine.is_null() || out_count.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let ws = engine.get_world_scale();
    let mgr = engine.crystal_anchors.lock().unwrap();
    let pairs = mgr.list_grown_pairs();
    *out_count = pairs.len() as u32;

    if out_buf.is_null() {
        return 1;
    }
    let cap = out_capacity as usize;
    let copy_n = pairs.len().min(cap);
    let buf_slice = std::slice::from_raw_parts_mut(out_buf, copy_n);
    for (i, p) in pairs.iter().take(copy_n).enumerate() {
        buf_slice[i] = FfiCrystalBridgePair {
            pair_token: p.pair_token,
            anchor_a_id: p.anchor_a_id,
            anchor_b_id: p.anchor_b_id,
            anchor_a_pos_ue: rust_pos_to_ue(p.anchor_a_pos_rust, ws),
            anchor_b_pos_ue: rust_pos_to_ue(p.anchor_b_pos_rust, ws),
            midpoint_ue: rust_pos_to_ue(p.midpoint_rust, ws),
        };
    }
    1
}

