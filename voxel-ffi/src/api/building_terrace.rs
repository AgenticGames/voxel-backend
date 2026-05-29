//! Auto-split from the former monolithic `api.rs`. See `mod.rs`.
#[allow(unused_imports)]
use std::ffi::{c_char, c_void, CString};
#[allow(unused_imports)]
use std::ptr;
#[allow(unused_imports)]
use crate::engine::VoxelEngine;
#[allow(unused_imports)]
use crate::types::*;

/// Query whether a 2x2 terrace exists at a UE world position.
/// Returns 1 if found (out_mat written), 0 if not found.
#[no_mangle]
pub unsafe extern "C" fn voxel_query_terrace(
    engine: *mut c_void,
    x: f32,
    y: f32,
    z: f32,
    scale: f32,
    out_mat: *mut u8,
) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    match engine.query_terrace(x, y, z, scale) {
        Some(mat) => {
            if !out_mat.is_null() {
                *out_mat = mat;
            }
            1
        }
        None => 0,
    }
}

/// Query floor support for a building placement.
/// footprint_voxels controls the NxN footprint (e.g. 4 = 4x4, 2 = 2x2).
/// Writes solid_count, total_columns, host material, and authoritative snapped UE position.
/// Returns 1 on success, 0 if engine null.
#[no_mangle]
pub unsafe extern "C" fn voxel_query_building_support(
    engine: *mut c_void,
    x: f32,
    y: f32,
    z: f32,
    scale: f32,
    footprint_voxels: i32,
    out_solid: *mut u8,
    out_total: *mut u8,
    out_mat: *mut u8,
    out_x: *mut f32,
    out_y: *mut f32,
    out_z: *mut f32,
) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let (solid, total, mat, sx, sy, sz) = engine.query_building_support(x, y, z, scale, footprint_voxels);
    if !out_solid.is_null() { *out_solid = solid; }
    if !out_total.is_null() { *out_total = total; }
    if !out_mat.is_null() { *out_mat = mat; }
    if !out_x.is_null() { *out_x = sx; }
    if !out_y.is_null() { *out_y = sy; }
    if !out_z.is_null() { *out_z = sz; }
    1
}

/// Request auto-terrace for a building placement.
/// footprint_voxels controls the NxN footprint (e.g. 4 = 4x4, 2 = 2x2).
/// clearance_voxels controls how many air voxels to carve above the floor.
/// Returns 1 on success, 0 if queue full.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_building_flatten(
    engine: *mut c_void,
    x: f32,
    y: f32,
    z: f32,
    scale: f32,
    footprint_voxels: i32,
    clearance_voxels: i32,
) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.request_building_flatten(x, y, z, scale, footprint_voxels, clearance_voxels)
}

/// Batch building flatten: flatten terrain under multiple buildings in one worker job.
/// All buildings share the same footprint and clearance. One seam pass for the lot.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_building_flatten_batch(
    engine: *mut c_void,
    xs: *const f32,
    ys: *const f32,
    zs: *const f32,
    count: u32,
    scale: f32,
    footprint_voxels: i32,
    clearance_voxels: i32,
) -> u32 {
    if engine.is_null() || xs.is_null() || ys.is_null() || zs.is_null() || count == 0 {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let n = count as usize;
    let xs = std::slice::from_raw_parts(xs, n);
    let ys = std::slice::from_raw_parts(ys, n);
    let zs = std::slice::from_raw_parts(zs, n);
    let positions: Vec<(f32, f32, f32)> = (0..n).map(|i| (xs[i], ys[i], zs[i])).collect();
    engine.request_building_flatten_batch(&positions, scale, footprint_voxels, clearance_voxels)
}

/// Query floor support for a flatten ghost preview.
/// Returns solid count. Writes snapped UE position to out pointers.
/// out_clearance_solids receives count of solid voxels in the 2-voxel clearance zone above.
#[no_mangle]
pub unsafe extern "C" fn voxel_query_flatten_support(
    engine: *mut c_void,
    x: f32,
    y: f32,
    z: f32,
    scale: f32,
    out_x: *mut f32,
    out_y: *mut f32,
    out_z: *mut f32,
    out_clearance_solids: *mut u8,
) -> u8 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let (count, clearance, sx, sy, sz) = engine.query_flatten_support(x, y, z, scale);
    if !out_x.is_null() { *out_x = sx; }
    if !out_y.is_null() { *out_y = sy; }
    if !out_z.is_null() { *out_z = sz; }
    if !out_clearance_solids.is_null() { *out_clearance_solids = clearance; }
    count
}

/// Query nearby existing terrace for Z-snap when extending terraces.
/// Returns 1 if a nearby terrace was found (writes snapped UE position), 0 otherwise.
#[no_mangle]
pub unsafe extern "C" fn voxel_query_nearby_terrace(
    engine: *mut c_void,
    x: f32,
    y: f32,
    z: f32,
    scale: f32,
    out_x: *mut f32,
    out_y: *mut f32,
    out_z: *mut f32,
) -> u8 {
    if engine.is_null() || out_x.is_null() || out_y.is_null() || out_z.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    match engine.query_nearby_terrace(x, y, z, scale) {
        Some((sx, sy, sz)) => {
            *out_x = sx;
            *out_y = sy;
            *out_z = sz;
            1
        }
        None => 0,
    }
}

