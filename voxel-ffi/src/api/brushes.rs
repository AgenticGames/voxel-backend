//! Auto-split from the former monolithic `api.rs`. See `mod.rs`.
#[allow(unused_imports)]
use std::ffi::{c_char, c_void, CString};
#[allow(unused_imports)]
use std::ptr;
#[allow(unused_imports)]
use crate::engine::VoxelEngine;
#[allow(unused_imports)]
use crate::types::*;

// ── Force Spawn Pool ──

/// Request force-spawning a pool at a UE world position.
/// fluid_type: 0=water, 1=lava. Coordinates are in UE world space (pre-scale).
/// Returns 1 on success, 0 if queue full.
#[no_mangle]
pub unsafe extern "C" fn voxel_force_spawn_pool(
    engine: *mut c_void,
    world_x: f32,
    world_y: f32,
    world_z: f32,
    fluid_type: u8,
    _world_scale: f32,
) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.request_force_spawn_pool(world_x, world_y, world_z, fluid_type)
}

/// Poll for a completed force-spawn pool result.
/// Returns null if not ready, otherwise a heap-allocated C string with JSON diagnostics.
/// Caller MUST call `voxel_free_force_spawn_result` on non-null returns.
#[no_mangle]
pub unsafe extern "C" fn voxel_poll_force_spawn_result(engine: *mut c_void) -> *mut c_char {
    if engine.is_null() {
        return ptr::null_mut();
    }
    let engine = &*(engine as *const VoxelEngine);
    match engine.poll_force_spawn_complete() {
        Some(json) => {
            match CString::new(json) {
                Ok(cstr) => cstr.into_raw(),
                Err(_) => ptr::null_mut(),
            }
        }
        None => ptr::null_mut(),
    }
}

/// Free a force-spawn pool result string. Safe to call with null pointer.
#[no_mangle]
pub unsafe extern "C" fn voxel_free_force_spawn_result(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}

/// Request mining a sphere and filling the bottom half with fluid.
/// fluid_type: 1=water, 2=lava. Coordinates are in UE world space.
/// Returns 1 on success, 0 if queue full.
#[no_mangle]
pub unsafe extern "C" fn voxel_mine_and_fill_fluid(
    engine: *mut c_void,
    world_x: f32,
    world_y: f32,
    world_z: f32,
    radius: f32,
    fluid_type: u8,
    world_scale: f32,
) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.mine_and_fill_fluid(world_x, world_y, world_z, radius, fluid_type, world_scale)
}

/// Creative-mode sphere brush: paint material (mode=0), carve (mode=1), or fill (mode=2).
/// `request.material` is ignored for carve mode. Returns 1 on success, 0 if queue full.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_brush_sphere(
    engine: *mut c_void,
    request: *const FfiBrushSphereRequest,
) -> u32 {
    if engine.is_null() || request.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.request_brush_sphere(*request)
}

/// Creative "OrePaint" brush — drops wall-exposed ore deposits inside the sphere
/// with Poisson-disk anti-clumping, weighted ore-type picks, and optional inward
/// "deep channel" tubes. Density is preserved; only `sample.material` is rewritten
/// where ore lands. Returns 1 on success, 0 if queue full.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_brush_ore_paint(
    engine: *mut c_void,
    request: *const FfiBrushOrePaintRequest,
) -> u32 {
    if engine.is_null() || request.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.request_brush_ore_paint(*request)
}

/// Creative "PaintStress" brush — additive sphere over the per-voxel painted-stress
/// overlay. Does NOT change density/material (no remesh emitted).
/// op: 0=add, 1=subtract, 2=clear. falloff: 0=constant, 1=linear, 2=smoothstep.
/// Returns 1 on success, 0 if queue full.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_brush_paint_stress(
    engine: *mut c_void,
    request: *const FfiBrushPaintStressRequest,
) -> u32 {
    if engine.is_null() || request.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.request_brush_paint_stress(*request)
}

/// Wipe every loaded chunk's painted-stress overlay back to empty.
/// "Clear All Painted" eraser button in the Atelier HUD.
/// Returns 1 on success, 0 if queue full.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_brush_clear_all_painted_stress(
    engine: *mut c_void,
) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.request_brush_clear_all_painted_stress()
}

/// Creative-mode tunnel brush: carves (or fills) a capsule along a polyline.
/// `points` is a UE-coord array of length `point_count` (>=2 required).
/// `material == 255` carves; otherwise fills with that material.
/// Returns 1 on success, 0 if queue full or invalid input.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_brush_tunnel(
    engine: *mut c_void,
    points: *const FfiVec3,
    point_count: u32,
    radius: f32,
    material: u8,
) -> u32 {
    if engine.is_null() || points.is_null() || point_count < 2 {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let pts_slice = std::slice::from_raw_parts(points, point_count as usize);
    let pts: Vec<(f32, f32, f32)> = pts_slice.iter().map(|p| (p.x, p.y, p.z)).collect();
    engine.request_brush_tunnel(&pts, radius, material)
}

/// Creative-mode formation placer (single stalactite/stalagmite/column/etc.).
/// Returns 1 on success, 0 if queue full.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_brush_formation(
    engine: *mut c_void,
    request: *const FfiBrushFormationRequest,
) -> u32 {
    if engine.is_null() || request.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.request_brush_formation(*request)
}

/// Creative-mode mushroom placer. Places one mushroom at the cursor anchor.
/// Returns 1 on success, 0 if queue full or invalid input.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_brush_place_mushroom(
    engine: *mut c_void,
    request: *const FfiBrushPlaceMushroomRequest,
) -> u32 {
    if engine.is_null() || request.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.request_brush_place_mushroom(*request)
}

/// Creative-mode mushroom sphere brush — scatters multiple mushrooms of one
/// kind within a radius, Bernoulli-sampled against viable surface voxels.
/// Returns 1 on success, 0 if queue full or invalid input.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_brush_place_mushroom_sphere(
    engine: *mut c_void,
    request: *const crate::types::FfiBrushPlaceMushroomSphereRequest,
) -> u32 {
    if engine.is_null() || request.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.request_brush_place_mushroom_sphere(*request)
}

/// Creative-mode formation stamp brush — runs the worldgen formation pipeline
/// (random mix of stalactites/columns/drapery/etc. per the live FormationConfig)
/// within a sphere. `seed` randomizes the pick.
/// Returns 1 on success, 0 if queue full.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_brush_formation_stamp(
    engine: *mut c_void,
    request: *const FfiBrushFormationStampRequest,
) -> u32 {
    if engine.is_null() || request.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.request_brush_formation_stamp(*request)
}

/// Creative-mode cavern stamp brush — chunk-snapped cave generator over a
/// NxMxK chunk region. Worms carve additively (existing edits survive),
/// optional pools/formations decorate the new surfaces.
/// Returns 1 on success, 0 if queue full.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_brush_cavern_stamp(
    engine: *mut c_void,
    request: *const FfiBrushCavernStampRequest,
) -> u32 {
    if engine.is_null() || request.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.request_brush_cavern_stamp(*request)
}

/// Creative-mode axis-aligned box brush (paint=0/carve=1/fill=2).
#[no_mangle]
pub unsafe extern "C" fn voxel_request_brush_box(
    engine: *mut c_void,
    request: *const FfiBrushBoxRequest,
) -> u32 {
    if engine.is_null() || request.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.request_brush_box(*request)
}

/// Creative-mode Y-axis-aligned cylinder brush (paint=0/carve=1/fill=2).
#[no_mangle]
pub unsafe extern "C" fn voxel_request_brush_cylinder(
    engine: *mut c_void,
    request: *const FfiBrushCylinderRequest,
) -> u32 {
    if engine.is_null() || request.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.request_brush_cylinder(*request)
}

/// Creative-mode smooth brush.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_brush_smooth(
    engine: *mut c_void,
    request: *const FfiBrushSmoothRequest,
) -> u32 {
    if engine.is_null() || request.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.request_brush_smooth(*request)
}

/// Creative-mode noise brush.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_brush_noise(
    engine: *mut c_void,
    request: *const FfiBrushNoiseRequest,
) -> u32 {
    if engine.is_null() || request.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.request_brush_noise(*request)
}

/// Undo the most recent creative-mode brush stroke. Returns 1 if queued.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_brush_undo(engine: *mut c_void) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.request_brush_undo()
}

/// Number of undo strokes currently available.
#[no_mangle]
pub unsafe extern "C" fn voxel_brush_undo_depth(engine: *mut c_void) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.undo_depth()
}

/// Sphere fluid brush (op: 0=fill, 1=clear, 2=pool-dig).
#[no_mangle]
pub unsafe extern "C" fn voxel_request_brush_fluid_sphere(
    engine: *mut c_void,
    request: *const FfiBrushFluidSphereRequest,
) -> u32 {
    if engine.is_null() || request.is_null() { return 0; }
    let engine = &*(engine as *const VoxelEngine);
    engine.request_brush_fluid_sphere(*request)
}

/// Box fluid brush (op: 0=fill, 1=clear).
#[no_mangle]
pub unsafe extern "C" fn voxel_request_brush_fluid_box(
    engine: *mut c_void,
    request: *const FfiBrushFluidBoxRequest,
) -> u32 {
    if engine.is_null() || request.is_null() { return 0; }
    let engine = &*(engine as *const VoxelEngine);
    engine.request_brush_fluid_box(*request)
}

/// River fluid brush — capsule chain along polyline of UE-world points.
/// `op == 2` carves the channel before filling.
/// `max_flow_dist`: bounded-flow limit when `is_source = 1`. 0 = unlimited.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_brush_fluid_river(
    engine: *mut c_void,
    points: *const FfiVec3,
    point_count: u32,
    radius: f32,
    fluid_type: u8,
    is_source: u8,
    op: u8,
    max_flow_dist: u8,
) -> u32 {
    if engine.is_null() || points.is_null() || point_count < 2 { return 0; }
    let engine = &*(engine as *const VoxelEngine);
    let pts_slice = std::slice::from_raw_parts(points, point_count as usize);
    let pts: Vec<(f32, f32, f32)> = pts_slice.iter().map(|p| (p.x, p.y, p.z)).collect();
    engine.request_brush_fluid_river(&pts, radius, fluid_type, is_source != 0, op, max_flow_dist)
}

/// Request flattening a 2x2 terrace at a UE world position.
/// Snaps to grid and uses depth-appropriate host rock.
/// Returns 1 on success, 0 if queue full.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_flatten(
    engine: *mut c_void,
    x: f32,
    y: f32,
    z: f32,
    scale: f32,
) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.request_flatten(x, y, z, scale)
}

/// Request batch-flattening of multiple terrace tiles in a single lock + remesh pass.
/// xs/ys/zs are parallel arrays of UE world positions; count is the array length.
/// Returns 1 on success, 0 if queue full or invalid args.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_flatten_batch(
    engine: *mut c_void,
    xs: *const f32,
    ys: *const f32,
    zs: *const f32,
    count: u32,
    scale: f32,
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
    engine.request_flatten_batch(&positions, scale)
}

