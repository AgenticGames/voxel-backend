//! Auto-split from the former monolithic `api.rs`. See `mod.rs`.
#[allow(unused_imports)]
use std::ffi::{c_char, c_void, CString};
#[allow(unused_imports)]
use std::ptr;
#[allow(unused_imports)]
use crate::engine::VoxelEngine;
#[allow(unused_imports)]
use crate::types::*;

// ── Editor collapse triggers ──────────────────────────────────────────

/// Create a new editor collapse trigger.
///
/// * `activation_kind`: 0 = OnFirstMine (any mining inside `volumes[0]`
///   fires it), 1 = OnPillarLoss (each entry of `volumes` is one pillar).
/// * `name_ptr`/`name_len`: UTF-8 bytes (not NUL-terminated). Truncated to
///   1024 chars on the Rust side.
/// * `volumes`/`volume_count`: ≥1 volumes; first is the primary.
/// * `loss_condition` (only for kind=1): 0=Any, 1=NPillars, 2=AllPillars.
/// * `loss_n` (only for kind=1, loss_condition=1): N threshold.
/// * `loss_threshold` (only for kind=1): pillar "lost" when its current
///   solid-voxel count is below this fraction × baseline.
/// * `slab_voxels`/`slab_count`: world voxel coords that fall during the
///   cinematic.
/// * `pile_chunks`/`pile_count`: chunk coords where debris settles.
/// * `fall_distance_uu`: 0 = auto from geometry, otherwise an override.
///
/// Returns the new trigger id (≥1), or 0 on validation failure.
#[no_mangle]
pub unsafe extern "C" fn voxel_create_trigger(
    engine: *mut c_void,
    activation_kind: u8,
    name_ptr: *const u8,
    name_len: u32,
    volumes: *const crate::types::FfiVoxelAabb,
    volume_count: u32,
    loss_condition: u8,
    loss_n: u8,
    loss_threshold: f32,
    slab_voxels: *const crate::types::FfiVoxelCoord,
    slab_count: u32,
    pile_chunks: *const crate::types::FfiVoxelCoord,
    pile_count: u32,
    fall_distance_uu: f32,
) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);

    let name = if !name_ptr.is_null() && name_len > 0 {
        let slice = std::slice::from_raw_parts(name_ptr, name_len as usize);
        std::str::from_utf8(slice).unwrap_or("").to_string()
    } else {
        String::new()
    };
    let volumes_slice: &[crate::types::FfiVoxelAabb] = if !volumes.is_null() && volume_count > 0 {
        std::slice::from_raw_parts(volumes, volume_count as usize)
    } else {
        &[]
    };
    let slab_slice: &[crate::types::FfiVoxelCoord] = if !slab_voxels.is_null() && slab_count > 0 {
        std::slice::from_raw_parts(slab_voxels, slab_count as usize)
    } else {
        &[]
    };
    let pile_slice: &[crate::types::FfiVoxelCoord] = if !pile_chunks.is_null() && pile_count > 0 {
        std::slice::from_raw_parts(pile_chunks, pile_count as usize)
    } else {
        &[]
    };

    let volumes_owned: Vec<crate::triggers::VoxelAabb> = volumes_slice
        .iter()
        .map(|a| crate::triggers::VoxelAabb {
            min: (a.min.x, a.min.y, a.min.z),
            max: (a.max.x, a.max.y, a.max.z),
        })
        .collect();
    let slab_owned: Vec<(i32, i32, i32)> =
        slab_slice.iter().map(|c| (c.x, c.y, c.z)).collect();
    let pile_owned: Vec<(i32, i32, i32)> =
        pile_slice.iter().map(|c| (c.x, c.y, c.z)).collect();

    engine.create_trigger(
        activation_kind,
        &name,
        &volumes_owned,
        loss_condition,
        loss_n,
        loss_threshold,
        &slab_owned,
        &pile_owned,
        fall_distance_uu,
    )
}

/// Write all trigger ids into `out_ids` (caller-allocated). Returns the
/// actual trigger count. If the count exceeds `capacity`, callers should
/// re-allocate and call again.
#[no_mangle]
pub unsafe extern "C" fn voxel_list_trigger_ids(
    engine: *mut c_void,
    out_ids: *mut u32,
    capacity: u32,
) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let ids = engine.list_trigger_ids();
    let n = ids.len() as u32;
    if !out_ids.is_null() && capacity > 0 {
        let write_n = n.min(capacity) as usize;
        let dst = std::slice::from_raw_parts_mut(out_ids, write_n);
        dst.copy_from_slice(&ids[..write_n]);
    }
    n
}

/// Fill `out_info` with the metadata for trigger `id`. Returns 1 on
/// success, 0 if no trigger has that id.
#[no_mangle]
pub unsafe extern "C" fn voxel_get_trigger_info(
    engine: *mut c_void,
    id: u32,
    out_info: *mut crate::types::FfiTriggerInfo,
) -> u32 {
    if engine.is_null() || out_info.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    match engine.get_trigger_info(id) {
        Some(info) => {
            *out_info = info;
            1
        }
        None => 0,
    }
}

/// Remove a trigger by id. Returns 1 if it existed.
#[no_mangle]
pub unsafe extern "C" fn voxel_remove_trigger(engine: *mut c_void, id: u32) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    if engine.remove_trigger(id) { 1 } else { 0 }
}

/// Arm (`armed`=1) or disarm (`armed`=0) a trigger.
/// On re-arm (false→true), pillar baselines are recaptured against the
/// current density so the iteration test can repeat.
#[no_mangle]
pub unsafe extern "C" fn voxel_set_trigger_armed(
    engine: *mut c_void,
    id: u32,
    armed: u8,
) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    if engine.set_trigger_armed(id, armed != 0) { 1 } else { 0 }
}

/// Fire a trigger on the next stress process tick, bypassing
/// `should_fire` evaluation. Used by the editor "Fire Now" preview
/// button. The cinematic runs through the normal pipeline.
#[no_mangle]
pub unsafe extern "C" fn voxel_fire_trigger_now(engine: *mut c_void, id: u32) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    if engine.fire_trigger_now(id) { 1 } else { 0 }
}

/// Query the engine for every voxel with density > 0 inside a sphere
/// of `radius` voxels centered on `(center_x, center_y, center_z)`.
///
/// Fills `out_buf` with up to `capacity` coords; returns the TOTAL solid
/// count (may exceed capacity if the buffer was too small — caller can
/// re-allocate and retry, though for typical brush radii the (2r+1)^3
/// upper bound is a safe pre-allocation).
///
/// Cells in unloaded chunks are skipped (treated as not-solid). Used by
/// the Trigger Author "Slab" paint action so its wireframe markers match
/// the cinematic's eventual slab geometry — air cells get filtered at
/// stroke time instead of silently disappearing on synth.
#[no_mangle]
pub unsafe extern "C" fn voxel_query_solid_sphere(
    engine: *mut c_void,
    center_x: i32,
    center_y: i32,
    center_z: i32,
    radius: i32,
    out_buf: *mut crate::types::FfiVoxelCoord,
    capacity: u32,
) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let mut buf: Vec<(i32, i32, i32)> = Vec::new();
    engine.query_solid_voxels_in_sphere((center_x, center_y, center_z), radius, &mut buf);
    let total = buf.len() as u32;
    if !out_buf.is_null() && capacity > 0 {
        let write = (total.min(capacity)) as usize;
        let dst = std::slice::from_raw_parts_mut(out_buf, write);
        for i in 0..write {
            dst[i] = crate::types::FfiVoxelCoord {
                x: buf[i].0,
                y: buf[i].1,
                z: buf[i].2,
            };
        }
    }
    total
}

/// Apply pending save snapshots to already-loaded chunks (for mid-game load).
/// Patches density fields, re-extracts hermite data.
/// Returns chunk keys that were patched via out_keys/out_count.
/// Caller must request remesh for these chunks (e.g. via RequestPriorityGenerate).
/// Caller must free the returned buffer via voxel_free_chunk_keys.
#[no_mangle]
pub unsafe extern "C" fn voxel_apply_loaded_snapshots(
    engine: *mut c_void,
    out_count: *mut u32,
) -> *mut FfiChunkCoord {
    if engine.is_null() || out_count.is_null() {
        if !out_count.is_null() { *out_count = 0; }
        return ptr::null_mut();
    }
    let engine = &*(engine as *const VoxelEngine);
    let patched = engine.apply_loaded_snapshots();
    *out_count = patched.len() as u32;
    if patched.is_empty() {
        return ptr::null_mut();
    }
    let coords: Vec<FfiChunkCoord> = patched.iter().map(|&(x, y, z)| FfiChunkCoord { x, y, z }).collect();
    let mut boxed = coords.into_boxed_slice();
    let ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);
    ptr
}

/// Free a chunk keys buffer returned by voxel_apply_loaded_snapshots.
#[no_mangle]
pub unsafe extern "C" fn voxel_free_chunk_keys(keys: *mut FfiChunkCoord, count: u32) {
    if keys.is_null() || count == 0 { return; }
    let _ = Vec::from_raw_parts(keys, count as usize, count as usize);
}

/// Check if the world has any unsaved modifications (mining, flatten, sleep, collapse).
/// Returns 1 if modifications exist, 0 otherwise.
#[no_mangle]
pub unsafe extern "C" fn voxel_has_world_modifications(engine: *mut c_void) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    if engine.has_world_modifications() { 1 } else { 0 }
}

