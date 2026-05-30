//! Auto-split from the former monolithic `api.rs`. See `mod.rs`.
#[allow(unused_imports)]
use std::ffi::{c_char, c_void, CString};
#[allow(unused_imports)]
use std::ptr;
#[allow(unused_imports)]
use crate::engine::VoxelEngine;
#[allow(unused_imports)]
use crate::types::*;

/// Find a capsule-validated spawn location for the player.
/// Returns 1 if found (out pointers written), 0 if no suitable location.
/// All coordinates are UE world space. Clearance: height=13, radius=3 voxels.
#[no_mangle]
pub unsafe extern "C" fn voxel_find_spawn_location(
    engine: *mut c_void,
    target_x: f32,
    target_y: f32,
    target_z: f32,
    exclude_x: f32,
    exclude_y: f32,
    exclude_z: f32,
    exclude_radius: f32,
    world_scale: f32,
    out_x: *mut f32,
    out_y: *mut f32,
    out_z: *mut f32,
) -> u32 {
    if engine.is_null() || out_x.is_null() || out_y.is_null() || out_z.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    // Player capsule: 13 voxels tall, 3 voxels radius
    match engine.find_spawn_location(
        target_x, target_y, target_z,
        exclude_x, exclude_y, exclude_z,
        exclude_radius, world_scale, 13, 3,
    ) {
        Some((x, y, z)) => {
            *out_x = x;
            *out_y = y;
            *out_z = z;
            1
        }
        None => 0,
    }
}

/// Find a validated spawn location for the chrysalis (quest giver).
/// Returns 1 if found (out pointers written), 0 if no suitable location.
/// Clearance: height=4, radius=2 voxels. Prefers near walls but not clipping.
#[no_mangle]
pub unsafe extern "C" fn voxel_find_chrysalis_location(
    engine: *mut c_void,
    target_x: f32,
    target_y: f32,
    target_z: f32,
    exclude_x: f32,
    exclude_y: f32,
    exclude_z: f32,
    exclude_radius: f32,
    world_scale: f32,
    out_x: *mut f32,
    out_y: *mut f32,
    out_z: *mut f32,
) -> u32 {
    if engine.is_null() || out_x.is_null() || out_y.is_null() || out_z.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    // Chrysalis: 4 voxels tall, 2 voxels radius
    match engine.find_chrysalis_location(
        target_x, target_y, target_z,
        exclude_x, exclude_y, exclude_z,
        exclude_radius, world_scale, 4, 2,
    ) {
        Some((x, y, z)) => {
            *out_x = x;
            *out_y = y;
            *out_z = z;
            1
        }
        None => 0,
    }
}

/// Find spring, chrysalis, and spawn locations all in the same cavern.
/// All coordinates are UE world space. Returns 1 if all three found, 0 otherwise.
/// Geode-filtered: no positions inside crystal geodes.
/// Same-cavern: chrysalis and spawn are flood-fill constrained to the spring's cavern.
#[no_mangle]
pub unsafe extern "C" fn voxel_find_cavern_locations(
    engine: *mut c_void,
    player_x: f32,
    player_y: f32,
    player_z: f32,
    world_scale: f32,
    out_spring_x: *mut f32,
    out_spring_y: *mut f32,
    out_spring_z: *mut f32,
    out_chrysalis_x: *mut f32,
    out_chrysalis_y: *mut f32,
    out_chrysalis_z: *mut f32,
    out_spawn_x: *mut f32,
    out_spawn_y: *mut f32,
    out_spawn_z: *mut f32,
) -> u32 {
    if engine.is_null()
        || out_spring_x.is_null() || out_spring_y.is_null() || out_spring_z.is_null()
        || out_chrysalis_x.is_null() || out_chrysalis_y.is_null() || out_chrysalis_z.is_null()
        || out_spawn_x.is_null() || out_spawn_y.is_null() || out_spawn_z.is_null()
    {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    match engine.find_cavern_locations(player_x, player_y, player_z, world_scale) {
        Some(((sx, sy, sz), (cx, cy, cz), (px, py, pz))) => {
            *out_spring_x = sx;
            *out_spring_y = sy;
            *out_spring_z = sz;
            *out_chrysalis_x = cx;
            *out_chrysalis_y = cy;
            *out_chrysalis_z = cz;
            *out_spawn_x = px;
            *out_spawn_y = py;
            *out_spawn_z = pz;
            1
        }
        None => 0,
    }
}

/// Request priority generation for a single chunk (sent via mine channel for
/// immediate processing). Coords are UE space. Returns 1 on success, 0 if full.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_priority_generate(
    engine: *mut c_void,
    chunk: FfiChunkCoord,
) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.request_priority_generate(chunk.x, chunk.y, chunk.z)
}

/// Query the host rock material at a UE world position based on depth.
/// Returns the material id as u8.
#[no_mangle]
pub unsafe extern "C" fn voxel_query_host_rock_at(
    engine: *mut c_void,
    x: f32,
    y: f32,
    z: f32,
    scale: f32,
) -> u8 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.query_host_rock_at(x, y, z, scale)
}

