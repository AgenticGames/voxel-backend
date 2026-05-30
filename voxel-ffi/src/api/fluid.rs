//! Auto-split from the former monolithic `api.rs`. See `mod.rs`.
#[allow(unused_imports)]
use std::ffi::{c_char, c_void, CString};
#[allow(unused_imports)]
use std::ptr;
#[allow(unused_imports)]
use crate::engine::VoxelEngine;
#[allow(unused_imports)]
use crate::types::*;

/// Inject fluid at a UE world position.
/// Inject a fluid cell at a world position.
/// fluid_type: 1=Water, 2=Lava (3-9 specialized water sub-types).
/// is_source: 1=infinite source, 0=finite (drains/spreads).
/// max_flow_dist: bounded-flow limit when `is_source = 1`. 0 = unlimited (legacy).
/// >0 = source's children stop propagating beyond this hop count, with linear
/// taper across the last few cells (Minecraft-style hard length limit).
/// Ignored when `is_source = 0`.
#[no_mangle]
pub unsafe extern "C" fn voxel_add_fluid(
    engine: *mut c_void,
    world_x: f32,
    world_y: f32,
    world_z: f32,
    fluid_type: u8,
    is_source: u8,
    world_scale: f32,
    max_flow_dist: u8,
) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.add_fluid(world_x, world_y, world_z, fluid_type, is_source != 0, world_scale, max_flow_dist)
}

/// Find the best cavern spring location near the player.
/// Returns 1 if found (out pointers written), 0 if no suitable location.
/// All coordinates are UE world space.
#[no_mangle]
pub unsafe extern "C" fn voxel_find_spring(
    engine: *mut c_void,
    player_x: f32,
    player_y: f32,
    player_z: f32,
    world_scale: f32,
    out_x: *mut f32,
    out_y: *mut f32,
    out_z: *mut f32,
) -> u32 {
    if engine.is_null() || out_x.is_null() || out_y.is_null() || out_z.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    match engine.find_spring(player_x, player_y, player_z, world_scale) {
        Some((x, y, z)) => {
            *out_x = x;
            *out_y = y;
            *out_z = z;
            1
        }
        None => 0,
    }
}

/// Find a wall-adjacent air cell near a target, excluding a radius around a point.
/// Returns 1 if found (out pointers written), 0 if no suitable location.
/// All coordinates are UE world space.
#[no_mangle]
pub unsafe extern "C" fn voxel_find_wall_near(
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
    match engine.find_wall_near(
        target_x, target_y, target_z,
        exclude_x, exclude_y, exclude_z,
        exclude_radius, world_scale,
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

/// Query surface-facing ore voxels near a player position.
///
/// `material_filter` of `0xFF` returns any ore (`Material::is_ore`); any other
/// value matches a specific `Material as u8`. `radius_ue` is in UE units.
///
/// Returns a heap-allocated list. Caller MUST call `voxel_free_ore_voxel_list`
/// on the result regardless of `count` (empty results are still well-formed).
#[no_mangle]
pub unsafe extern "C" fn voxel_query_ore_voxels(
    engine: *mut c_void,
    player_x: f32,
    player_y: f32,
    player_z: f32,
    radius_ue: f32,
    material_filter: u8,
    max_results: u32,
    world_scale: f32,
) -> FfiOreVoxelList {
    let empty = FfiOreVoxelList {
        voxels: ptr::null_mut(),
        count: 0,
    };

    if engine.is_null() {
        return empty;
    }
    let engine = &*(engine as *const VoxelEngine);

    let hits = engine.find_ore_voxels(
        player_x,
        player_y,
        player_z,
        radius_ue,
        material_filter,
        max_results as usize,
        world_scale,
    );

    if hits.is_empty() {
        return empty;
    }

    let mut buf: Vec<FfiOreVoxel> = hits
        .into_iter()
        .map(|(x, y, z, m)| FfiOreVoxel {
            x,
            y,
            z,
            material_index: m,
            _pad: [0; 3],
        })
        .collect();

    let count = buf.len() as u32;
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);

    FfiOreVoxelList { voxels: ptr, count }
}

/// Free ore voxel list returned by `voxel_query_ore_voxels`.
/// Safe to call with an empty/null list.
#[no_mangle]
pub unsafe extern "C" fn voxel_free_ore_voxel_list(list: FfiOreVoxelList) {
    if !list.voxels.is_null() && list.count > 0 {
        drop(Vec::from_raw_parts(
            list.voxels,
            list.count as usize,
            list.count as usize,
        ));
    }
}

