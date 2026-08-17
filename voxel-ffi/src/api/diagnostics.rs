//! Auto-split from the former monolithic `api.rs`. See `mod.rs`.
#[allow(unused_imports)]
use std::ffi::{c_char, c_void, CString};
#[allow(unused_imports)]
use std::ptr;
#[allow(unused_imports)]
use crate::engine::VoxelEngine;
#[allow(unused_imports)]
use crate::types::*;

// ── Profiler API ──

/// Enable or disable the streaming profiler.
/// Returns the previous state (1=was enabled, 0=was disabled).
#[no_mangle]
pub unsafe extern "C" fn voxel_profiler_set_enabled(engine: *mut c_void, enabled: u32) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let was_enabled = engine.profiler_is_enabled();
    engine.profiler_set_enabled(enabled != 0);
    if was_enabled { 1 } else { 0 }
}

/// Begin a new profiling session. Resets all metrics and captures config snapshot.
/// Returns the session id (monotonically increasing).
#[no_mangle]
pub unsafe extern "C" fn voxel_profiler_begin_session(engine: *mut c_void) -> u64 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.profiler_begin_session()
}

/// End the current profiling session and return a plain-text report.
/// Returns a heap-allocated null-terminated UTF-8 string.
/// Caller MUST free with `voxel_profiler_free_report`.
/// Returns null if engine is null.
#[no_mangle]
pub unsafe extern "C" fn voxel_profiler_get_report(engine: *mut c_void) -> *mut c_char {
    if engine.is_null() {
        return ptr::null_mut();
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.profiler_end_session();
    engine.profiler_get_report_cstr()
}

/// Free a report string previously returned by `voxel_profiler_get_report`.
#[no_mangle]
pub unsafe extern "C" fn voxel_profiler_free_report(report: *mut c_char) {
    if report.is_null() {
        return;
    }
    drop(CString::from_raw(report));
}

/// Build a multi-line plain-text diagnostic dump for a single chunk.
/// Caller passes UE chunk coords (the same convention used by
/// voxel_query_stress, voxel_request_generate, etc.) — the FFI converts
/// to Rust chunk coords internally for HashMap lookup.
/// Returns a heap-allocated null-terminated UTF-8 string the caller must
/// free with `voxel_free_chunk_diagnostic`. Returns null if the engine
/// pointer is null.
#[no_mangle]
pub unsafe extern "C" fn voxel_get_chunk_diagnostic(
    engine: *mut c_void,
    ue_cx: i32,
    ue_cy: i32,
    ue_cz: i32,
) -> *mut c_char {
    if engine.is_null() {
        return ptr::null_mut();
    }
    let engine = &*(engine as *const VoxelEngine);
    let rust_chunk = crate::convert::ue_chunk_to_rust(ue_cx, ue_cy, ue_cz);
    let dump = engine.build_chunk_diagnostic_with_ue(rust_chunk, (ue_cx, ue_cy, ue_cz));
    match CString::new(dump) {
        Ok(cs) => cs.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Free a diagnostic string previously returned by `voxel_get_chunk_diagnostic`.
#[no_mangle]
pub unsafe extern "C" fn voxel_free_chunk_diagnostic(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    drop(CString::from_raw(s));
}

/// Diagnostic: force a single chunk to re-sync its boundaries with all
/// face-adjacent neighbors and re-mesh whatever changed. Used by the UE
/// "Force Resync" button on UVoxelChunkDiagnosticComponent.
/// Caller passes UE chunk coords; FFI converts to Rust chunk coords
/// internally (Rust store is keyed by the post-transform coord).
/// Returns 1 if queued, 0 if engine is null or the worker queue is full.
#[no_mangle]
pub unsafe extern "C" fn voxel_force_chunk_resync(
    engine: *mut c_void,
    ue_cx: i32,
    ue_cy: i32,
    ue_cz: i32,
) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let rust_chunk = crate::convert::ue_chunk_to_rust(ue_cx, ue_cy, ue_cz);
    engine.request_force_chunk_resync(rust_chunk.0, rust_chunk.1, rust_chunk.2)
}

/// Bulk force-resync (2026-08-18): one request for a whole chunk set — the
/// post-montage truth-restore. Caller passes UE chunk coords (an array of
/// i32 triples, len = count*3), ALREADY neighbor-expanded; each chunk is
/// remeshed exactly once via the slice-parallel path. Coord contract matches
/// voxel_force_chunk_resync (UE coords in, converted per-chunk here).
/// Returns 1 if queued, 0 if engine null / empty / queue full.
#[no_mangle]
pub unsafe extern "C" fn voxel_force_chunk_resync_batch(
    engine: *mut c_void,
    ue_coords: *const i32,
    count: u32,
) -> u32 {
    if engine.is_null() || ue_coords.is_null() || count == 0 {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let triples = std::slice::from_raw_parts(ue_coords, (count as usize) * 3);
    let chunks: Vec<(i32, i32, i32)> = triples
        .chunks_exact(3)
        .map(|t| crate::convert::ue_chunk_to_rust(t[0], t[1], t[2]))
        .collect();
    engine.request_force_chunk_resync_batch(chunks)
}

/// Lightweight zone scan: generates density fields (noise only, no worms/ores)
/// for chunks in a radius, then runs zone detection. Writes results into caller-provided buffer.
/// Returns 0 on success, non-zero on error.
#[no_mangle]
pub unsafe extern "C" fn voxel_scan_zones(
    engine: *mut c_void,
    center_x: i32, center_y: i32, center_z: i32,
    chunk_radius: i32,
    out_zones: *mut FfiZoneDescriptor,
    out_count: *mut u32,
    max_zones: u32,
) -> i32 {
    if engine.is_null() || out_zones.is_null() || out_count.is_null() {
        return -1;
    }
    let engine = &*(engine as *const VoxelEngine);
    let cfg = engine.config_snapshot();
    let ws = engine.get_world_scale();

    // Transform UE chunk coords to Rust coords
    let rust_center = crate::convert::ue_chunk_to_rust(center_x, center_y, center_z);

    let descriptors = voxel_gen::scan_zones_only(rust_center, chunk_radius, &cfg);

    let voxel_scale = cfg.effective_bounds() / cfg.chunk_size as f32;
    let scale = voxel_scale * ws;

    let count = descriptors.len().min(max_zones as usize);
    for (i, zd) in descriptors.iter().take(count).enumerate() {
        let ffi = FfiZoneDescriptor {
            zone_type: zd.zone_type as u8,
            // Rust Y-up → UE Z-up: swap Y↔Z, negate new Y
            center_x: zd.center.x * scale,
            center_y: -zd.center.z * scale,
            center_z: zd.center.y * scale,
            min_x: zd.world_min.x * scale,
            min_y: -zd.world_max.z * scale,  // negate + swap min/max
            min_z: zd.world_min.y * scale,
            max_x: zd.world_max.x * scale,
            max_y: -zd.world_min.z * scale,  // negate + swap min/max
            max_z: zd.world_max.y * scale,
        };
        *out_zones.add(i) = ffi;
    }
    *out_count = count as u32;

    0 // success
}

