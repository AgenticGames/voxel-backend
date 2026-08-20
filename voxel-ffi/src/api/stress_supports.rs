//! Auto-split from the former monolithic `api.rs`. See `mod.rs`.
#[allow(unused_imports)]
use std::ffi::{c_char, c_void, CString};
#[allow(unused_imports)]
use std::ptr;
#[allow(unused_imports)]
use crate::engine::VoxelEngine;
#[allow(unused_imports)]
use crate::types::*;

/// Query the stress field for a chunk. Returns heap-allocated stress data.
/// Caller MUST call `voxel_free_stress_data` on the result.
/// Chunk coords are UE space.
#[no_mangle]
pub unsafe extern "C" fn voxel_query_stress(
    engine: *mut c_void,
    chunk_x: i32,
    chunk_y: i32,
    chunk_z: i32,
) -> FfiStressData {
    use crate::convert::ue_chunk_to_rust;

    if engine.is_null() {
        return FfiStressData {
            stress_values: ptr::null_mut(),
            classification: ptr::null_mut(),
            count: 0,
            valid: 0,
            painted_values: ptr::null_mut(),
        };
    }
    let engine = &*(engine as *const VoxelEngine);
    let key = ue_chunk_to_rust(chunk_x, chunk_y, chunk_z);

    match engine.query_stress(key) {
        Some(sf) => {
            let count = sf.stress.len() as u32;
            // The painted layer is optional — when the chunk has never had a
            // PaintStress stroke we return a null pointer so UE can short-circuit.
            let painted_ptr = if sf.painted_stress.is_empty() {
                ptr::null_mut()
            } else {
                let mut painted_data = sf.painted_stress.into_boxed_slice();
                let p = painted_data.as_mut_ptr();
                std::mem::forget(painted_data);
                p
            };

            let mut stress_data = sf.stress.into_boxed_slice();
            let stress_ptr = stress_data.as_mut_ptr();
            std::mem::forget(stress_data);

            let mut class_data = sf.classification.into_boxed_slice();
            let class_ptr = class_data.as_mut_ptr();
            std::mem::forget(class_data);

            FfiStressData {
                stress_values: stress_ptr,
                classification: class_ptr,
                count,
                valid: 1,
                painted_values: painted_ptr,
            }
        }
        None => FfiStressData {
            stress_values: ptr::null_mut(),
            classification: ptr::null_mut(),
            count: 0,
            valid: 0,
            painted_values: ptr::null_mut(),
        },
    }
}

/// Free stress data returned by `voxel_query_stress`.
#[no_mangle]
pub unsafe extern "C" fn voxel_free_stress_data(data: FfiStressData) {
    if !data.stress_values.is_null() && data.count > 0 {
        drop(Vec::from_raw_parts(
            data.stress_values,
            data.count as usize,
            data.count as usize,
        ));
    }
    if !data.classification.is_null() && data.count > 0 {
        drop(Vec::from_raw_parts(
            data.classification,
            data.count as usize,
            data.count as usize,
        ));
    }
    if !data.painted_values.is_null() && data.count > 0 {
        drop(Vec::from_raw_parts(
            data.painted_values,
            data.count as usize,
            data.count as usize,
        ));
    }
}

/// Synchronously recalculate stress on nearby chunks for V-key overlay preview.
/// Takes a UE chunk coordinate as center, recalcs the 3x3 at that Y + 3x3 at Y+1 (18 chunks).
/// Call before querying stress to ensure overlay has data.
#[no_mangle]
/// out_chunks: caller provides array of 27*3 i32s (27 chunks × xyz).
/// Returns actual count written.
pub unsafe extern "C" fn voxel_recalc_stress_preview(
    engine: *mut c_void,
    world_x: f32,
    world_y: f32,
    world_z: f32,
    out_chunks: *mut i32,
    out_count: *mut u32,
) {
    use crate::convert::from_ue_world_pos;

    if engine.is_null() {
        if !out_count.is_null() { *out_count = 0; }
        return;
    }
    let engine_ref = &*(engine as *const VoxelEngine);
    let world_scale = engine_ref.get_world_scale();
    let rust_pos = from_ue_world_pos(world_x, world_y, world_z, world_scale);

    let cs = engine_ref.chunk_size() as i32;
    let center = (
        (rust_pos.x.floor() as i32).div_euclid(cs),
        (rust_pos.y.floor() as i32).div_euclid(cs),
        (rust_pos.z.floor() as i32).div_euclid(cs),
    );
    let ue_keys = engine_ref.recalc_stress_preview(center);

    // Write UE chunk coords to output buffer
    if !out_chunks.is_null() && !out_count.is_null() {
        let max_out = 27usize;
        let write_count = ue_keys.len().min(max_out);
        for i in 0..write_count {
            *out_chunks.add(i * 3) = ue_keys[i].0;
            *out_chunks.add(i * 3 + 1) = ue_keys[i].1;
            *out_chunks.add(i * 3 + 2) = ue_keys[i].2;
        }
        *out_count = write_count as u32;
    }
}

/// Query stress at a single world position (UE coords).
/// Returns normalized stress value (>= 1.0 means overstressed).
#[no_mangle]
pub unsafe extern "C" fn voxel_query_stress_at(
    engine: *mut c_void,
    world_x: f32,
    world_y: f32,
    world_z: f32,
) -> f32 {
    use crate::convert::from_ue_world_pos;

    if engine.is_null() {
        return 0.0;
    }
    let engine = &*(engine as *const VoxelEngine);

    let chunk_size = engine.chunk_size();
    let world_scale = engine.get_world_scale();

    let rust_pos = from_ue_world_pos(world_x, world_y, world_z, world_scale);
    engine.query_stress_at(
        rust_pos.x as i32,
        rust_pos.y as i32,
        rust_pos.z as i32,
        chunk_size,
    )
}

/// Query the PAINTED-stress overlay at a single world position (UE coords).
/// Returns the authored/painted stress at that voxel (0.0 if none). Lets the
/// game tell authored tunnel collapses (painted stress) apart from natural ones.
#[no_mangle]
pub unsafe extern "C" fn voxel_query_painted_stress_at(
    engine: *mut c_void,
    world_x: f32,
    world_y: f32,
    world_z: f32,
) -> f32 {
    use crate::convert::from_ue_world_pos;

    if engine.is_null() {
        return 0.0;
    }
    let engine = &*(engine as *const VoxelEngine);

    let chunk_size = engine.chunk_size();
    let world_scale = engine.get_world_scale();

    let rust_pos = from_ue_world_pos(world_x, world_y, world_z, world_scale);
    engine.query_painted_stress_at(
        rust_pos.x as i32,
        rust_pos.y as i32,
        rust_pos.z as i32,
        chunk_size,
    )
}

/// Voxel-aware surface probe at a UE world point. Used by spider-nest /
/// wasp-hive placement validators to confirm a candidate is anchored to a
/// real surface of the right kind with enough cavity room.
///
/// `hint_*` is the caller's surface-normal hint in UE space (e.g. the
/// hit normal of an Unreal line trace); pass `(0, 0, 1)` if no hint is
/// available. The hint is used only as a fallback when the local density
/// gradient is flat (open air or solid interior).
///
/// Returns 1 on success (out_probe written), 0 if the engine pointer is
/// null, the out pointer is null, or the store lock couldn't be acquired.
#[no_mangle]
pub unsafe extern "C" fn voxel_query_surface(
    engine: *mut c_void,
    world_x: f32, world_y: f32, world_z: f32,
    hint_x: f32, hint_y: f32, hint_z: f32,
    out_probe: *mut FfiSurfaceProbe,
) -> u32 {
    if engine.is_null() || out_probe.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let probe = match engine.query_surface(world_x, world_y, world_z, hint_x, hint_y, hint_z) {
        Some(p) => p,
        None => return 0,
    };
    let world_scale = engine.get_world_scale();

    // Translate the Rust-space normal back to UE space using the same
    // swap-and-negate as `convert::convert_crystals_to_ue` (Rust nx,ny,nz
    // -> UE nx, -nz, ny).
    let normal_ue_x = probe.normal.x;
    let normal_ue_y = -probe.normal.z;
    let normal_ue_z = probe.normal.y;

    // Per-axis clearance is in voxel units along Rust axes (+X,-X,+Y,-Y,+Z,-Z).
    // UE axis mapping for clearance directions:
    //   UE +X  ⇔ Rust +X  → clearance_rust[0]
    //   UE -X  ⇔ Rust -X  → clearance_rust[1]
    //   UE +Y  ⇔ Rust -Z  → clearance_rust[5]
    //   UE -Y  ⇔ Rust +Z  → clearance_rust[4]
    //   UE +Z  ⇔ Rust +Y  → clearance_rust[2]
    //   UE -Z  ⇔ Rust -Y  → clearance_rust[3]
    let cr = &probe.clearance_rust;
    let clearance_ue = [
        cr[0] * world_scale, // +X
        cr[1] * world_scale, // -X
        cr[5] * world_scale, // +Y
        cr[4] * world_scale, // -Y
        cr[2] * world_scale, // +Z
        cr[3] * world_scale, // -Z
    ];

    *out_probe = FfiSurfaceProbe {
        kind: probe.kind as u8,
        _padding: [0; 3],
        normal_x: normal_ue_x,
        normal_y: normal_ue_y,
        normal_z: normal_ue_z,
        cavity_radius: probe.cavity_radius * world_scale,
        clearance_ue,
    };
    1
}

/// Cheap TRI-STATE solidity at a UE-world point: 0=air, 1=loaded-solid,
/// 2=unloaded (lock-busy → 2). For the sleep-montage camera planner — ~1000×
/// cheaper than `voxel_query_surface`. The ray clamp treats {1,2} as rock; the
/// camera-exposure check treats only 1 as enclosure (so unloaded void reads as
/// exposed).
#[no_mangle]
pub unsafe extern "C" fn voxel_is_solid_at(
    engine: *mut c_void,
    ue_x: f32,
    ue_y: f32,
    ue_z: f32,
) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.is_solid_at_ue(ue_x, ue_y, ue_z)
}

/// List all surface-exposed cells in a chunk whose effective stress has
/// crossed the collapse threshold (>= 1.0). UE uses the result to drive
/// per-chunk stress-crack decals and warning dust puffs at primed cells.
///
/// Chunk coords are UE space. Returns a heap-allocated list; caller MUST
/// call `voxel_free_overstressed_list` on the result.
#[no_mangle]
pub unsafe extern "C" fn voxel_list_overstressed_in_chunk(
    engine: *mut c_void,
    chunk_x: i32,
    chunk_y: i32,
    chunk_z: i32,
) -> FfiOverstressedList {
    use crate::convert::ue_chunk_to_rust;

    if engine.is_null() {
        // Engine null is "permanent" failure, not contention — caller should
        // treat as a real empty list (clear decals). valid=1 + count=0.
        return FfiOverstressedList { cells: ptr::null_mut(), count: 0, valid: 1 };
    }
    let engine = &*(engine as *const VoxelEngine);
    let key = ue_chunk_to_rust(chunk_x, chunk_y, chunk_z);
    let (cells, valid) = engine.enumerate_overstressed_in_chunk(key);

    if cells.is_empty() {
        return FfiOverstressedList { cells: ptr::null_mut(), count: 0, valid: if valid { 1 } else { 0 } };
    }
    let mut boxed = cells.into_boxed_slice();
    let count = boxed.len() as u32;
    let ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);
    FfiOverstressedList { cells: ptr, count, valid: 1 }
}

/// List all surface-exposed over-stress cells inside a UE world-space sphere.
/// Used by the mining post-recalc handler to fire a "you're undermining a
/// fragile area" dust burst at every primed cell within the mining impact zone.
///
/// `center_*` and `radius` are in UE units. Returns a heap-allocated list;
/// caller MUST call `voxel_free_overstressed_list` on the result.
#[no_mangle]
pub unsafe extern "C" fn voxel_list_overstressed_in_sphere(
    engine: *mut c_void,
    center_x: f32,
    center_y: f32,
    center_z: f32,
    radius: f32,
) -> FfiOverstressedList {
    if engine.is_null() || radius <= 0.0 {
        // Permanent failure → valid=1, count=0 (real "empty" answer).
        return FfiOverstressedList { cells: ptr::null_mut(), count: 0, valid: 1 };
    }
    let engine = &*(engine as *const VoxelEngine);
    let (cells, valid) = engine.enumerate_overstressed_in_sphere(center_x, center_y, center_z, radius);

    if cells.is_empty() {
        return FfiOverstressedList { cells: ptr::null_mut(), count: 0, valid: if valid { 1 } else { 0 } };
    }
    let mut boxed = cells.into_boxed_slice();
    let count = boxed.len() as u32;
    let ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);
    FfiOverstressedList { cells: ptr, count, valid: 1 }
}

/// Free a list returned by `voxel_list_overstressed_in_chunk` or
/// `voxel_list_overstressed_in_sphere`.
#[no_mangle]
pub unsafe extern "C" fn voxel_free_overstressed_list(list: FfiOverstressedList) {
    if !list.cells.is_null() && list.count > 0 {
        drop(Vec::from_raw_parts(
            list.cells,
            list.count as usize,
            list.count as usize,
        ));
    }
}

/// Synchronous "is a path possible from A to B" check. Used by the wasp
/// hive placement validator to confirm a hive can actually deploy wasps
/// to nearby cavity-center POIs. Runs A* under a brief store read lock
/// (~5-50 ms typical for 10_000-node budget); acceptable here because
/// placement validation is off the hot path (~once per cluster spawn).
///
/// `movement_mode`: 0 = Flying, 1 = Walking, 2 = Surface.
/// `max_nodes`: 0 = use default (10_000).
///
/// Returns the raw `voxel_path::PathStatus` u8:
///   0 = Success, 1 = NoPath, 2 = MaxNodesReached,
///   3 = PartiallyUnloaded, 4 = InvalidEndpoint.
/// Returns 1 (NoPath) on engine-pointer null; returns 255 on store lock
/// contention (the sync probe uses try_read and CANNOT wait — during a
/// streaming burst every probe in a burst-window used to come back as a
/// counterfeit NoPath, which made sense-trail endpoint resolution reject
/// every column right after a teleport). 255 means "could not check":
/// callers should defer to an async solve, whose worker blocks on the lock
/// properly, rather than treat the cell as invalid.
#[no_mangle]
pub unsafe extern "C" fn voxel_query_path_exists(
    engine: *mut c_void,
    from_x: f32, from_y: f32, from_z: f32,
    to_x: f32, to_y: f32, to_z: f32,
    agent_radius_ue: f32,
    movement_mode: u8,
    fine_cells: u8,
    unknown_open: u8,
    max_nodes: u32,
) -> u8 {
    if engine.is_null() {
        return 1; // NoPath
    }
    let engine = &*(engine as *const VoxelEngine);
    engine
        .query_path_exists(
            from_x, from_y, from_z,
            to_x, to_y, to_z,
            agent_radius_ue,
            movement_mode,
            fine_cells,
            unknown_open,
            max_nodes,
        )
        .unwrap_or(255)
}

/// Place a support structure at a UE world position.
/// support_type (2026-05-26+): 1=Copper, 2=Iron, 3=Steel, 4=Crystal, 5=Mithril.
/// Legacy IDs 6/7 from the pre-overhaul UE plugin (Steel/Crystal at 6/7) are
/// remapped via `SupportType::from_legacy_u8` — the in-flight DLL/editor pair
/// during the rollout doesn't garble.
/// Returns 1 on success (queued), 0 on failure.
#[no_mangle]
pub unsafe extern "C" fn voxel_place_support(
    engine: *mut c_void,
    world_x: f32,
    world_y: f32,
    world_z: f32,
    support_type: u8,
) -> u32 {
    use crate::convert::from_ue_world_pos;

    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    // Coord contract: UE world-space in, engine's LIVE world scale (config-
    // driven, 30 as of 2026-08) for the transform. This was hardcoded 15.0
    // in all three support fns — self-consistently wrong (place/query/remove
    // agreed with each other) but the support-field cells landed at 2x the
    // aimed voxel, so stress relief centered far from the visual strut.
    let world_scale = engine.get_world_scale();
    let rust_pos = from_ue_world_pos(world_x, world_y, world_z, world_scale);
    engine.request_place_support(
        rust_pos.x as i32,
        rust_pos.y as i32,
        rust_pos.z as i32,
        support_type,
    )
}

/// Remove a support structure at a UE world position.
/// Returns 1 on success (queued), 0 on failure.
#[no_mangle]
pub unsafe extern "C" fn voxel_remove_support(
    engine: *mut c_void,
    world_x: f32,
    world_y: f32,
    world_z: f32,
) -> u32 {
    use crate::convert::from_ue_world_pos;

    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    // Coord contract: UE world-space in, engine's LIVE world scale (config-
    // driven, 30 as of 2026-08) for the transform. This was hardcoded 15.0
    // in all three support fns — self-consistently wrong (place/query/remove
    // agreed with each other) but the support-field cells landed at 2x the
    // aimed voxel, so stress relief centered far from the visual strut.
    let world_scale = engine.get_world_scale();
    let rust_pos = from_ue_world_pos(world_x, world_y, world_z, world_scale);
    engine.request_remove_support(
        rust_pos.x as i32,
        rust_pos.y as i32,
        rust_pos.z as i32,
    )
}

/// Drain all pending broken-strut events. Caller passes a pointer to
/// `*mut FfiStrutBroken` (out_ptr) and a pointer to a u32 (out_count).
/// On return:
///   - `out_count` is set to the number of broken-strut events drained.
///   - `out_ptr` points to a heap-allocated array of length `out_count`,
///     or NULL when count is 0.
///   - Caller MUST call `voxel_free_struts_broken(ptr, count)` to release
///     the array when done.
///
/// Returns 1 on success, 0 if `engine` is null. A successful call with
/// count=0 is normal (no struts broken this frame) — `out_ptr` will be NULL
/// in that case and `voxel_free_struts_broken` is a no-op on NULL.
#[no_mangle]
pub unsafe extern "C" fn voxel_take_struts_broken(
    engine: *mut c_void,
    out_ptr: *mut *mut crate::types::FfiStrutBroken,
    out_count: *mut u32,
) -> u32 {
    if engine.is_null() || out_ptr.is_null() || out_count.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let mut drained = engine.drain_struts_broken();
    let count = drained.len() as u32;
    *out_count = count;
    if count == 0 {
        *out_ptr = std::ptr::null_mut();
    } else {
        drained.shrink_to_fit();
        let boxed = drained.into_boxed_slice();
        *out_ptr = Box::into_raw(boxed) as *mut crate::types::FfiStrutBroken;
    }
    1
}

/// Free a struts-broken array returned by `voxel_take_struts_broken`.
/// Safe to call with NULL ptr (no-op).
#[no_mangle]
pub unsafe extern "C" fn voxel_free_struts_broken(
    ptr: *mut crate::types::FfiStrutBroken,
    count: u32,
) {
    if ptr.is_null() || count == 0 { return; }
    let slice = std::slice::from_raw_parts_mut(ptr, count as usize);
    let _ = Box::from_raw(slice as *mut [crate::types::FfiStrutBroken]);
}

/// Synchronously query a strut's HP/type at a UE world position. UE uses this
/// for the aim-inspect HP-bar widget. Caller passes a pointer to a stack-
/// allocated `FfiStrutInfo`; no heap allocation, no paired free needed.
///
/// `valid==0`: store lock contended; UE should keep the existing bar value
///             and retry next frame.
/// `valid==1, support_type==0`: chunk loaded, no strut at this voxel.
/// `valid==1, support_type>=1`: strut found; read `hp` / `max_hp`.
#[no_mangle]
pub unsafe extern "C" fn voxel_query_strut_hp(
    engine: *mut c_void,
    world_x: f32,
    world_y: f32,
    world_z: f32,
    out_info: *mut crate::types::FfiStrutInfo,
) -> u32 {
    use crate::convert::from_ue_world_pos;

    if engine.is_null() || out_info.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    // Coord contract: UE world-space in, engine's LIVE world scale (config-
    // driven, 30 as of 2026-08) for the transform. This was hardcoded 15.0
    // in all three support fns — self-consistently wrong (place/query/remove
    // agreed with each other) but the support-field cells landed at 2x the
    // aimed voxel, so stress relief centered far from the visual strut.
    let world_scale = engine.get_world_scale();
    let rust_pos = from_ue_world_pos(world_x, world_y, world_z, world_scale);
    let (stype, hp, max_hp, valid) = engine.query_strut_hp(
        rust_pos.x as i32, rust_pos.y as i32, rust_pos.z as i32,
    );
    (*out_info) = crate::types::FfiStrutInfo {
        support_type: stype, _pad: [0; 1],
        hp, max_hp,
        valid, _pad2: [0; 1],
    };
    1
}

/// Set the stress configuration. Takes a pointer to FfiStressConfig.
#[no_mangle]
pub unsafe extern "C" fn voxel_set_stress_config(
    engine: *mut c_void,
    config: *const FfiStressConfig,
) {
    use voxel_gen::config::StressConfig;

    if engine.is_null() || config.is_null() {
        return;
    }
    let engine = &*(engine as *const VoxelEngine);
    let ffi_cfg = &*config;

    let stress_config = StressConfig {
        material_hardness: ffi_cfg.material_hardness,
        gravity_weight: ffi_cfg.gravity_weight,
        lateral_support_factor: ffi_cfg.lateral_support_factor,
        vertical_support_factor: ffi_cfg.vertical_support_factor,
        support_radius: ffi_cfg.support_radius,
        propagation_radius: ffi_cfg.propagation_radius,
        max_collapse_volume: ffi_cfg.max_collapse_volume,
        rubble_enabled: ffi_cfg.rubble_enabled != 0,
        rubble_fill_ratio: ffi_cfg.rubble_fill_ratio,
        warn_dust_threshold: ffi_cfg.warn_dust_threshold,
        warn_creak_threshold: ffi_cfg.warn_creak_threshold,
        warn_shake_threshold: ffi_cfg.warn_shake_threshold,
        support_hardness: ffi_cfg.support_hardness,
        // V2 fields
        lateral_transfer_factor: ffi_cfg.lateral_transfer_factor,
        vertical_transfer_factor: ffi_cfg.vertical_transfer_factor,
        support_propagation_iterations: ffi_cfg.support_propagation_iterations,
        ground_threshold: ffi_cfg.ground_threshold,
        overhang_weight: ffi_cfg.overhang_weight,
        span_weight: ffi_cfg.span_weight,
        min_safe_span: ffi_cfg.min_safe_span,
        min_collapse_region: ffi_cfg.min_collapse_region,
        slab_cohesion_threshold: ffi_cfg.slab_cohesion_threshold,
        cross_section_weight: ffi_cfg.cross_section_weight,
        cross_section_min_faces: ffi_cfg.cross_section_min_faces,
        surface_y: ffi_cfg.surface_y,
        depth_pressure_scale: ffi_cfg.depth_pressure_scale,
        mining_stress_scan_buffer: ffi_cfg.mining_stress_scan_buffer,
    };

    engine.update_stress_config(stress_config);
}

