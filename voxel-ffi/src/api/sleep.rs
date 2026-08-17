//! Auto-split from the former monolithic `api.rs`. See `mod.rs`.
#[allow(unused_imports)]
use std::ffi::{c_char, c_void, CString};
#[allow(unused_imports)]
use std::ptr;
#[allow(unused_imports)]
use crate::engine::VoxelEngine;
#[allow(unused_imports)]
use crate::types::*;

/// Update the sleep configuration from FFI config fields.
#[no_mangle]
pub unsafe extern "C" fn voxel_set_sleep_config(
    engine: *mut c_void,
    config: *const FfiEngineConfig,
) {
    if engine.is_null() || config.is_null() {
        return;
    }
    let engine = &*(engine as *const VoxelEngine);
    let ffi_config = &*config;
    let sleep_config = crate::engine::ffi_config_to_sleep(ffi_config);
    engine.update_sleep_config(sleep_config);
}

/// Set spider nest positions for sleep fossilization. UE calls this before voxel_start_sleep().
/// Coordinates are in UE world space (will be converted to Rust space).
/// Returns 1 on success, 0 on error.
#[no_mangle]
pub unsafe extern "C" fn voxel_set_sleep_nests(
    engine: *mut c_void,
    world_xs: *const i32,
    world_ys: *const i32,
    world_zs: *const i32,
    count: u32,
) -> u32 {
    if engine.is_null() || (count > 0 && (world_xs.is_null() || world_ys.is_null() || world_zs.is_null())) {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let positions: Vec<(i32, i32, i32)> = (0..count as usize)
        .map(|i| {
            let ux = *world_xs.add(i);
            let uy = *world_ys.add(i);
            let uz = *world_zs.add(i);
            crate::convert::ue_chunk_to_rust(ux, uy, uz)
        })
        .collect();
    engine.set_sleep_nests(positions);
    1
}

/// Set spider corpse positions for sleep fossilization. Coordinates are UE world space.
#[no_mangle]
pub unsafe extern "C" fn voxel_set_sleep_corpses(
    engine: *mut c_void,
    world_xs: *const i32,
    world_ys: *const i32,
    world_zs: *const i32,
    count: u32,
) -> u32 {
    if engine.is_null() || (count > 0 && (world_xs.is_null() || world_ys.is_null() || world_zs.is_null())) {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let positions: Vec<(i32, i32, i32)> = (0..count as usize)
        .map(|i| {
            let ux = *world_xs.add(i);
            let uy = *world_ys.add(i);
            let uz = *world_zs.add(i);
            crate::convert::ue_chunk_to_rust(ux, uy, uz)
        })
        .collect();
    engine.set_sleep_corpses(positions);
    1
}

/// Set the tagged top-POI chunks to simulate during the next sleep regardless
/// of chunk_radius distance. UE passes UE chunk coords (converted to Rust);
/// it should first ensure these chunks are streamed/generated so the sim has
/// density to work on. This is what gives distant POIs (a far lava spot, a
/// bridge across the map) REAL per-voxel reveal data instead of a synthesized
/// fallback. Call AFTER voxel_set_sleep_config and BEFORE voxel_start_sleep.
#[no_mangle]
pub unsafe extern "C" fn voxel_set_sleep_poi_chunks(
    engine: *mut c_void,
    chunk_xs: *const i32,
    chunk_ys: *const i32,
    chunk_zs: *const i32,
    count: u32,
) -> u32 {
    if engine.is_null() || (count > 0 && (chunk_xs.is_null() || chunk_ys.is_null() || chunk_zs.is_null())) {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let positions: Vec<(i32, i32, i32)> = (0..count as usize)
        .map(|i| {
            let ux = *chunk_xs.add(i);
            let uy = *chunk_ys.add(i);
            let uz = *chunk_zs.add(i);
            crate::convert::ue_chunk_to_rust(ux, uy, uz)
        })
        .collect();
    engine.set_sleep_extra_chunks(positions);
    1
}

/// Mark the chunks the sleep-montage is filming as protected from eviction.
/// While set, the worker's `unload()` refuses to drop these chunks' density, so
/// the camera planner's QuerySurface calls (rock-vs-air ray clamp) always have
/// real voxel data. UE pins the UE-side chunk actor, but only this stops Rust
/// from evicting the density underneath. Chunk coords are UE space; replaces any
/// prior set. Call `voxel_montage_clear_protected` at montage end.
#[no_mangle]
pub unsafe extern "C" fn voxel_montage_set_protected_chunks(
    engine: *mut c_void,
    chunk_xs: *const i32,
    chunk_ys: *const i32,
    chunk_zs: *const i32,
    count: u32,
) -> u32 {
    if engine.is_null() || (count > 0 && (chunk_xs.is_null() || chunk_ys.is_null() || chunk_zs.is_null())) {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let chunks: Vec<(i32, i32, i32)> = (0..count as usize)
        .map(|i| {
            let ux = *chunk_xs.add(i);
            let uy = *chunk_ys.add(i);
            let uz = *chunk_zs.add(i);
            crate::convert::ue_chunk_to_rust(ux, uy, uz)
        })
        .collect();
    engine.set_montage_protected(chunks);
    1
}

/// Release the montage-protected chunk set so normal streaming eviction resumes.
#[no_mangle]
pub unsafe extern "C" fn voxel_montage_clear_protected(engine: *mut c_void) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.clear_montage_protected();
    1
}

/// Release the sleep handler's deferred FAR remesh + seam pass (2026-08-18).
/// UE calls this when the montage reveal curtain rises (and again from
/// CleanupMontage as a backstop) — the far work then runs on the idle pool
/// during the prebuffered reveal instead of contending with the morph-step
/// prebuffer. Idempotent; the handler also self-releases after 30s.
#[no_mangle]
pub unsafe extern "C" fn voxel_sleep_far_work_go(engine: *mut c_void) -> u32 {
    if engine.is_null() {
        return 0;
    }
    crate::worker::sleep_morph::SLEEP_FAR_GO.store(true, std::sync::atomic::Ordering::Relaxed);
    1
}

/// Start a deep sleep cycle. player_chunk coordinates are in UE space.
/// Returns 1 on success, 0 if queue full.
#[no_mangle]
pub unsafe extern "C" fn voxel_start_sleep(
    engine: *mut c_void,
    player_cx: i32,
    player_cy: i32,
    player_cz: i32,
    sleep_count: u32,
) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let player_chunk = crate::convert::ue_chunk_to_rust(player_cx, player_cy, player_cz);
    engine.start_sleep(player_chunk, sleep_count)
}

/// Run only the aureole phase (contact metamorphism + lava solidification) for debugging.
/// Result is delivered via voxel_poll_sleep_result — same polling path as deep sleep.
/// Player chunk is in UE chunk coordinates (same convention as voxel_start_sleep).
#[no_mangle]
pub unsafe extern "C" fn voxel_start_aureole_only(
    engine: *mut c_void,
    player_cx: i32,
    player_cy: i32,
    player_cz: i32,
) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let player_chunk = crate::convert::ue_chunk_to_rust(player_cx, player_cy, player_cz);
    engine.start_aureole_only(player_chunk)
}

/// Poll for a completed sleep result.
/// Returns an FfiSleepResult with success=1 if a result is available, success=0 otherwise.
/// Dirty chunk meshes and collapse events are delivered through the normal voxel_poll_result
/// pipeline; this function only returns the summary statistics.
#[no_mangle]
pub unsafe extern "C" fn voxel_poll_sleep_result(engine: *mut c_void) -> FfiSleepResult {
    let empty = FfiSleepResult {
        success: 0,
        chunks_changed: 0,
        voxels_metamorphosed: 0,
        minerals_grown: 0,
        supports_degraded: 0,
        collapses_triggered: 0,
        acid_dissolved: 0,
        veins_deposited: 0,
        voxels_enriched: 0,
        formations_grown: 0,
        sulfide_dissolved: 0,
        coal_matured: 0,
        diamonds_formed: 0,
        voxels_silicified: 0,
        nests_fossilized: 0,
        channels_eroded: 0,
        corpses_fossilized: 0,
        lava_solidified: 0,
        dirty_chunks: ptr::null_mut(),
        dirty_chunk_count: 0,
        collapse_events: ptr::null_mut(),
        collapse_event_count: 0,
        profile_report: ptr::null_mut(),
        profile_report_length: 0,
        has_aureole_glimpse: 0,
        aureole_glimpse_x: 0, aureole_glimpse_y: 0, aureole_glimpse_z: 0,
        has_aureole_block: 0,
        aureole_block: ptr::null_mut(),
        aureole_block_count: 0,
        manifest_json: ptr::null_mut(),
        manifest_json_length: 0,
        lava_cells: ptr::null_mut(),
        lava_cell_count: 0,
        surface_changed_cells: ptr::null_mut(),
        surface_changed_cell_count: 0,
        surface_activity: [0u16; voxel_sleep::SURFACE_ACTIVITY_BUCKETS],
    };
    if engine.is_null() {
        return empty;
    }
    let engine = &*(engine as *const VoxelEngine);
    match engine.poll_sleep_complete() {
        Some(data) => {
            // Strip any interior null bytes so CString::new() cannot fail.
            let sanitized_report: String = data.profile_report.replace('\0', "");
            let (report_ptr, report_len) = {
                let cstr = CString::new(sanitized_report).unwrap_or_default();
                let len = cstr.as_bytes().len() as u32;
                (cstr.into_raw(), len)
            };
            let (manifest_ptr, manifest_len) = if data.manifest_json.is_empty() {
                (ptr::null_mut(), 0u32)
            } else {
                let cstr = CString::new(data.manifest_json).unwrap_or_default();
                let len = cstr.as_bytes().len() as u32;
                (cstr.into_raw(), len)
            };
            FfiSleepResult {
                success: 1,
                chunks_changed: data.chunks_changed,
                voxels_metamorphosed: data.voxels_metamorphosed,
                minerals_grown: data.minerals_grown,
                supports_degraded: data.supports_degraded,
                collapses_triggered: data.collapses_triggered,
                acid_dissolved: data.acid_dissolved,
                veins_deposited: data.veins_deposited,
                voxels_enriched: data.voxels_enriched,
                formations_grown: data.formations_grown,
                sulfide_dissolved: data.sulfide_dissolved,
                coal_matured: data.coal_matured,
                diamonds_formed: data.diamonds_formed,
                voxels_silicified: data.voxels_silicified,
                nests_fossilized: data.nests_fossilized,
                channels_eroded: data.channels_eroded,
                corpses_fossilized: data.corpses_fossilized,
                lava_solidified: data.lava_solidified,
                dirty_chunks: ptr::null_mut(),
                dirty_chunk_count: 0,
                collapse_events: ptr::null_mut(),
                collapse_event_count: 0,
                profile_report: report_ptr,
                profile_report_length: report_len,
                has_aureole_glimpse: if data.aureole_glimpse_pos.is_some() { 1 } else { 0 },
                aureole_glimpse_x: data.aureole_glimpse_pos.map_or(0, |p| p.0),
                aureole_glimpse_y: data.aureole_glimpse_pos.map_or(0, |p| p.1),
                aureole_glimpse_z: data.aureole_glimpse_pos.map_or(0, |p| p.2),
                has_aureole_block: if data.aureole_showcase_block.is_some() { 1 } else { 0 },
                aureole_block: data.aureole_showcase_block.as_ref().map_or(
                    ptr::null_mut(),
                    |b| {
                        let mut coords: Vec<FfiChunkCoord> = b.iter()
                            .map(|&(x, y, z)| FfiChunkCoord { x, y, z })
                            .collect();
                        let ptr = coords.as_mut_ptr();
                        std::mem::forget(coords);
                        ptr
                    },
                ),
                aureole_block_count: data.aureole_showcase_block.as_ref().map_or(0, |b| b.len() as u32),
                manifest_json: manifest_ptr,
                manifest_json_length: manifest_len,
                lava_cells: if data.lava_cells.is_empty() {
                    ptr::null_mut()
                } else {
                    let mut coords: Vec<FfiChunkCoord> = data.lava_cells.iter()
                        .map(|&(x, y, z)| FfiChunkCoord { x, y, z })
                        .collect();
                    let ptr = coords.as_mut_ptr();
                    std::mem::forget(coords);
                    ptr
                },
                lava_cell_count: data.lava_cells.len() as u32,
                surface_changed_cells: if data.surface_changed_cells.is_empty() {
                    ptr::null_mut()
                } else {
                    let mut coords: Vec<FfiChunkCoord> = data.surface_changed_cells.iter()
                        .map(|&(x, y, z)| FfiChunkCoord { x, y, z })
                        .collect();
                    let ptr = coords.as_mut_ptr();
                    std::mem::forget(coords);
                    ptr
                },
                surface_changed_cell_count: data.surface_changed_cells.len() as u32,
                surface_activity: data.surface_step_activity,
            }
        },
        None => empty,
    }
}

