//! Auto-split from the former monolithic `api.rs`. See `mod.rs`.
#[allow(unused_imports)]
use std::ffi::{c_char, c_void, CString};
#[allow(unused_imports)]
use std::ptr;
#[allow(unused_imports)]
use crate::engine::VoxelEngine;
#[allow(unused_imports)]
use crate::types::*;

// ════════════════════════════════════════════════════════════════════
// ─── Block 1 FFI: World Memory + Cinema + Predictor ────────────────
// ════════════════════════════════════════════════════════════════════

/// Query the WorldMemory Scene store. Fills `out_buf` with at most
/// `out_capacity` Scenes (sorted by score desc), writes the available
/// count to `out_count`.
///
/// Filter: passed by value. `kind_mask=0xFFFFFFFF` = all kinds.
/// `include_topology=0` (default) filters out CeilingDome/Chokepoint/WallNiche.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_scenes(
    engine: *mut c_void,
    filter: FfiSceneFilter,
    out_buf: *mut FfiScene,
    out_capacity: u32,
    out_count: *mut u32,
) -> u32 {
    if engine.is_null() || out_count.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let ws = engine.get_world_scale();

    let scene_filter = voxel_world_memory::SceneFilter {
        kind_mask: filter.kind_mask,
        min_score: filter.min_score,
        min_confidence: filter.min_confidence,
        include_topology: filter.include_topology != 0,
    };

    let scenes = engine.world_memory.scenes(scene_filter, out_capacity as usize);
    *out_count = scenes.len() as u32;
    if out_buf.is_null() {
        return 1;
    }

    let cap = out_capacity as usize;
    let copy_n = scenes.len().min(cap);
    let buf_slice = std::slice::from_raw_parts_mut(out_buf, copy_n);
    for (i, s) in scenes.iter().take(copy_n).enumerate() {
        let to_ue = |rx: f32, ry: f32, rz: f32| FfiVec3 {
            x: rx * ws,
            y: -rz * ws,
            z: ry * ws,
        };
        buf_slice[i] = FfiScene {
            id: s.id.0,
            kind: s.kind as u8,
            _padding: [0; 3],
            score: s.score,
            confidence: s.confidence,
            age_secs: s.age_secs,
            centroid_ue: to_ue(s.centroid[0], s.centroid[1], s.centroid[2]),
            aabb_min_ue: to_ue(s.aabb.min[0], s.aabb.min[1], s.aabb.min[2]),
            aabb_max_ue: to_ue(s.aabb.max[0], s.aabb.max[1], s.aabb.max[2]),
            history_count: s.history.len() as u32,
            tag_mask: s.tags.0,
        };
    }
    1
}

/// Fetch the history ring buffer for a specific Scene. Returns the
/// number of available history entries; caller's buffer is filled with
/// up to `out_capacity` entries (oldest-first).
#[no_mangle]
pub unsafe extern "C" fn voxel_request_scene_history(
    engine: *mut c_void,
    scene_id: u64,
    out_buf: *mut FfiSceneHistory,
    out_capacity: u32,
    out_count: *mut u32,
) -> u32 {
    if engine.is_null() || out_count.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let scene = match engine
        .world_memory
        .scenes
        .get(&voxel_world_memory::scene::SceneId(scene_id))
    {
        Some(s) => s.value().clone(),
        None => {
            *out_count = 0;
            return 0;
        }
    };
    *out_count = scene.history.len() as u32;
    if out_buf.is_null() {
        return 1;
    }
    let cap = out_capacity as usize;
    let copy_n = scene.history.len().min(cap);
    let buf_slice = std::slice::from_raw_parts_mut(out_buf, copy_n);
    for (i, h) in scene.history.iter().take(copy_n).enumerate() {
        buf_slice[i] = FfiSceneHistory {
            tag: h.tag,
            _padding: [0; 3],
            at_secs: h.at_secs,
        };
    }
    1
}

/// Push a world event into the WorldMemory ingestion queue. Lock-free,
/// non-blocking. Returns 1 if accepted, 0 if the queue was full
/// (caller should not retry — events are advisory).
#[no_mangle]
pub unsafe extern "C" fn voxel_record_world_event(
    engine: *mut c_void,
    event: FfiWorldEvent,
) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let ws = engine.get_world_scale();
    // UE → Rust voxel coord conversion. World scale 40, axis swap:
    // (ux, uy, uz) → (ux/ws, uz/ws, -uy/ws).
    let rust_pos = [
        event.world_pos_ue.x / ws,
        event.world_pos_ue.z / ws,
        -event.world_pos_ue.y / ws,
    ];
    let kind_hint = voxel_world_memory::scene::SceneKind::from_u8(event.kind_hint);

    use voxel_world_memory::WorldEvent;
    let wm_event = match event.event_kind {
        0 => WorldEvent::brush_applied(rust_pos, kind_hint),
        1 => WorldEvent::anchor_placed(rust_pos, event.payload as u64),
        2 => WorldEvent::collapse_fired(rust_pos, event.payload),
        3 => WorldEvent::sleep_completed(event.payload, 0),
        4 => WorldEvent::lava_spread_at(rust_pos[0], rust_pos[1], rust_pos[2]),
        5 => WorldEvent::water_changed_at(rust_pos[0], rust_pos[1], rust_pos[2]),
        _ => return 0,
    };
    if engine.world_memory.record_event(wm_event) {
        1
    } else {
        0
    }
}

/// Compose shot candidates for a Scene. Fills both `out_buf` (the
/// candidates) and `waypoint_buf` (the parallel waypoint payload). Each
/// candidate's `waypoint_offset/_count` indexes into `waypoint_buf`.
///
/// `intent_mask`: bitmask of allowed `ShotIntent` discriminants. Use
/// `0xFFFFFFFF` for all intents.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_shot_candidates(
    engine: *mut c_void,
    scene_id: u64,
    intent_mask: u32,
    count: u32,
    out_buf: *mut FfiShotCandidate,
    out_capacity: u32,
    out_count: *mut u32,
    waypoint_buf: *mut FfiWaypoint,
    waypoint_capacity: u32,
    waypoint_count: *mut u32,
) -> u32 {
    if engine.is_null() || out_count.is_null() || waypoint_count.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let ws = engine.get_world_scale();

    let mask = voxel_cinema::IntentMask(intent_mask);
    let scene_sid = voxel_world_memory::scene::SceneId(scene_id);
    let candidates =
        crate::cinema_bridge::compose_for_engine(engine, scene_sid, mask, count as usize);

    *out_count = candidates.len() as u32;
    let total_waypoints: u32 = candidates.iter().map(|c| c.waypoints.len() as u32).sum();
    *waypoint_count = total_waypoints;

    if out_buf.is_null() || waypoint_buf.is_null() {
        return 1;
    }

    let cap = out_capacity as usize;
    let wp_cap = waypoint_capacity as usize;
    let copy_n = candidates.len().min(cap);

    let mut wp_cursor: usize = 0;
    let out_slice = std::slice::from_raw_parts_mut(out_buf, copy_n);
    let wp_slice = std::slice::from_raw_parts_mut(waypoint_buf, wp_cap);

    for (i, c) in candidates.iter().take(copy_n).enumerate() {
        let mut caption = [0u8; 64];
        let bytes = c.caption.as_bytes();
        let n = bytes.len().min(63);
        caption[..n].copy_from_slice(&bytes[..n]);
        // [n] stays 0 (NUL terminator).

        let waypoint_offset = wp_cursor as u32;
        let mut written = 0u32;
        for w in &c.waypoints {
            if wp_cursor >= wp_cap {
                break;
            }
            let to_ue = |rx: f32, ry: f32, rz: f32| FfiVec3 {
                x: rx * ws,
                y: -rz * ws,
                z: ry * ws,
            };
            wp_slice[wp_cursor] = FfiWaypoint {
                pos_ue: to_ue(w.pos[0], w.pos[1], w.pos[2]),
                look_at_ue: to_ue(w.look_at[0], w.look_at[1], w.look_at[2]),
                fov_deg: w.fov_deg,
                t_secs: w.t_secs,
                dof_focus_dist: w.dof_focus_dist * ws,
                dof_aperture: w.dof_aperture,
            };
            wp_cursor += 1;
            written += 1;
        }

        out_slice[i] = FfiShotCandidate {
            intent: c.intent as u8,
            _padding: [0; 3],
            score: c.score,
            waypoint_offset,
            waypoint_count: written,
            total_duration: c.total_duration,
            lighting: FfiLightingProfile {
                warmth: c.lighting.warmth,
                contrast: c.lighting.contrast,
                key_intensity: c.lighting.key_intensity,
                fill_ratio: c.lighting.fill_ratio,
                hero_position_intent: c.lighting.hero_position_intent as u8,
                _padding: [0; 3],
            },
            caption,
            audio_cue: c.audio_cue,
            _padding2: [0; 3],
        };
    }
    1
}

/// Poll the predictor cache. Fills `out_manifest` (single scalar struct)
/// plus optional parallel buffers for the variable-length payloads
/// (likely-changed chunks, lava cells, aureole block, scene hints).
/// Returns 0 if no prediction has been computed yet, 1 if data is filled.
///
/// Callers can pass null buffers to query just the counts (e.g. to size
/// their own buffers correctly).
#[no_mangle]
pub unsafe extern "C" fn voxel_poll_prediction_cache(
    engine: *mut c_void,
    out_manifest: *mut FfiPredictedManifest,
    chunks_changed_buf: *mut FfiChunkCoord,
    chunks_changed_capacity: u32,
    lava_cells_buf: *mut FfiChunkCoord,
    lava_cells_capacity: u32,
    aureole_block_buf: *mut FfiChunkCoord,
    aureole_block_capacity: u32,
    scene_hints_buf: *mut FfiSceneHint,
    scene_hints_capacity: u32,
) -> u32 {
    if engine.is_null() || out_manifest.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let ws = engine.get_world_scale();
    let chunk_size = engine.chunk_size() as f32;

    let guard = match engine.predict_cache.read() {
        Ok(g) => g,
        Err(_) => return 0,
    };
    let manifest = match guard.as_ref() {
        Some(m) => m,
        None => return 0,
    };

    // Conversion helpers — Rust voxel coord → UE world coord.
    let rust_pos_to_ue = |rx: i32, ry: i32, rz: i32| FfiVec3 {
        x: rx as f32 * ws,
        y: -(rz as f32) * ws,
        z: ry as f32 * ws,
    };
    let rust_chunk_to_ue_coord = |cx: i32, cy: i32, cz: i32| {
        let (ucx, ucy, ucz) = crate::convert::rust_chunk_to_ue(cx, cy, cz);
        FfiChunkCoord {
            x: ucx,
            y: ucy,
            z: ucz,
        }
    };

    let aureole = manifest.predicted_aureole_glimpse_pos;
    *out_manifest = FfiPredictedManifest {
        computed_at_secs: manifest.computed_at_secs,
        wall_ms: manifest.wall_ms,
        sleep_count: manifest.sleep_count,
        chunks_changed_count: manifest.chunks_likely_changed.len() as u32,
        lava_cells_count: manifest.predicted_lava_cells.len() as u32,
        aureole_block_count: manifest
            .predicted_aureole_block
            .as_ref()
            .map(|b| b.len() as u32)
            .unwrap_or(0),
        scene_hints_count: manifest.predicted_scene_hints.len() as u32,
        has_aureole_glimpse: aureole.is_some() as u8,
        _padding: [0; 3],
        aureole_glimpse_ue: aureole
            .map(|(rx, ry, rz)| rust_pos_to_ue(rx, ry, rz))
            .unwrap_or(FfiVec3 { x: 0.0, y: 0.0, z: 0.0 }),
    };

    // Fill payload buffers if provided.
    if !chunks_changed_buf.is_null() {
        let cap = chunks_changed_capacity as usize;
        let copy_n = manifest.chunks_likely_changed.len().min(cap);
        let buf = std::slice::from_raw_parts_mut(chunks_changed_buf, copy_n);
        for (i, c) in manifest.chunks_likely_changed.iter().take(copy_n).enumerate() {
            buf[i] = rust_chunk_to_ue_coord(c.0, c.1, c.2);
        }
    }
    if !lava_cells_buf.is_null() {
        let cap = lava_cells_capacity as usize;
        let copy_n = manifest.predicted_lava_cells.len().min(cap);
        let buf = std::slice::from_raw_parts_mut(lava_cells_buf, copy_n);
        // Lava cells are world voxel coords, not chunk coords — packed as
        // FfiChunkCoord for transport convenience.
        for (i, c) in manifest.predicted_lava_cells.iter().take(copy_n).enumerate() {
            buf[i] = FfiChunkCoord {
                x: c.0,
                y: c.1,
                z: c.2,
            };
        }
    }
    if !aureole_block_buf.is_null() {
        if let Some(block) = manifest.predicted_aureole_block.as_ref() {
            let cap = aureole_block_capacity as usize;
            let copy_n = block.len().min(cap);
            let buf = std::slice::from_raw_parts_mut(aureole_block_buf, copy_n);
            for (i, c) in block.iter().take(copy_n).enumerate() {
                buf[i] = rust_chunk_to_ue_coord(c.0, c.1, c.2);
            }
        }
    }
    if !scene_hints_buf.is_null() {
        let cap = scene_hints_capacity as usize;
        let copy_n = manifest.predicted_scene_hints.len().min(cap);
        let buf = std::slice::from_raw_parts_mut(scene_hints_buf, copy_n);
        for (i, h) in manifest.predicted_scene_hints.iter().take(copy_n).enumerate() {
            // h.chunk_coord is a Rust chunk coord (already chunk-space).
            buf[i] = FfiSceneHint {
                kind: h.kind,
                _padding: [0; 3],
                estimated_score: h.estimated_score,
                world_pos_ue: rust_pos_to_ue(h.world_pos.0, h.world_pos.1, h.world_pos.2),
                chunk_coord_ue: rust_chunk_to_ue_coord(
                    h.chunk_coord.0,
                    h.chunk_coord.1,
                    h.chunk_coord.2,
                ),
            };
        }
    }

    // Silence unused warning if chunk_size isn't read above.
    let _ = chunk_size;
    1
}

/// Poke the predictor's wake channel so it runs immediately instead of
/// waiting up to 60 s. UE calls this when the player approaches a bedroll.
/// Returns 1 if the signal was accepted, 0 if the channel was full
/// (predictor is already pending a tick — no need to bug it again).
#[no_mangle]
pub unsafe extern "C" fn voxel_request_predict_now(engine: *mut c_void) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    if engine.predict_wake_tx.try_send(()).is_ok() {
        1
    } else {
        0
    }
}

