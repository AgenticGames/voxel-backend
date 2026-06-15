//! Auto-split from the former monolithic `api.rs`. See `mod.rs`.
#[allow(unused_imports)]
use std::ffi::{c_char, c_void, CString};
#[allow(unused_imports)]
use std::ptr;
#[allow(unused_imports)]
use crate::engine::VoxelEngine;
#[allow(unused_imports)]
use crate::types::*;

// ─── Pathfinding FFI ─────────────────────────────────────────────
//
// Async pattern: caller submits a request via `voxel_path_request` and gets
// back a `request_id` (or 0 on failure). Path runs on a dedicated worker
// thread; caller polls `voxel_path_poll` with that id each frame. On a
// returned status of 1, the `FfiPathResult` is filled — caller MUST then call
// `voxel_path_free` to release the heap-allocated node array.

/// Submit a path request to the path-worker thread.
/// Returns: request id (>= 1) on success, 0 if the path channel is full or
/// any pointer is null.
#[no_mangle]
pub unsafe extern "C" fn voxel_path_request(
    engine: *mut c_void,
    request: *const crate::pathing::FfiPathRequest,
) -> u32 {
    if engine.is_null() || request.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let req = *request;
    engine.request_path(req)
}

/// Poll for a completed path result by request id.
/// Returns:
///   0 — request pending (still computing) OR result already collected
///   1 — result populated into `*out`; caller MUST call voxel_path_free
///   2 — unknown id (never submitted, or expired by TTL)
///
/// `out` must point to a caller-allocated FfiPathResult struct.
#[no_mangle]
pub unsafe extern "C" fn voxel_path_poll(
    engine: *mut c_void,
    request_id: u32,
    out: *mut crate::pathing::FfiPathResult,
) -> u32 {
    if engine.is_null() || out.is_null() || request_id == 0 {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    match engine.poll_path(request_id) {
        Some(result) => {
            *out = result;
            1
        }
        None => 0,
    }
}

/// Release the heap-allocated node array of an FfiPathResult previously
/// returned by `voxel_path_poll`. Idempotent on null / zero-count.
#[no_mangle]
pub unsafe extern "C" fn voxel_path_free(
    engine: *mut c_void,
    result: *mut crate::pathing::FfiPathResult,
) {
    if engine.is_null() || result.is_null() {
        return;
    }
    let engine = &*(engine as *const VoxelEngine);
    let r = &mut *result;
    engine.free_path_nodes(r.nodes, r.node_count);
    r.nodes = std::ptr::null_mut();
    r.node_count = 0;
}

/// Cross-species avoidance: replace the engine's obstacle-cell snapshot.
///
/// `cells` is a packed array of `(ue_x, ue_y, ue_z)` floats — one f32 triple
/// per agent position, `count` triples total (so the buffer is `count * 3`
/// f32s long). UE-space positions are converted to Rust pathing-cell coords
/// internally and stored in the engine's `occupied_cells` set. Subsequent
/// path requests treat those cells as obstacles (with self-exclusion at the
/// requester's own cell).
///
/// Call once per tick from UE (or whatever cadence makes sense — staleness
/// just means agents occasionally route through where a peer was a few frames
/// ago). Idempotent on null engine / zero count (clears the set).
#[no_mangle]
pub unsafe extern "C" fn voxel_path_set_obstacle_cells(
    engine: *mut c_void,
    cells: *const f32,
    count: u32,
) {
    if engine.is_null() {
        return;
    }
    let engine = &*(engine as *const VoxelEngine);

    // Widen to usize before any arithmetic so `count * 3` cannot wrap a u32
    // (a wrapped length would build an undersized slice and the loop below
    // would index past it). Cap the eager reservation so a bogus `count`
    // cannot force a multi-GB allocation; the set still grows as needed.
    let count = count as usize;
    let mut new_set: std::collections::HashSet<(i32, i32, i32)> =
        std::collections::HashSet::with_capacity(count.min(65_536));
    if !cells.is_null() && count > 0 {
        let cf = crate::pathing::PATH_CELL_FACTOR as f32;
        let world_scale = engine.get_world_scale();
        let slice = std::slice::from_raw_parts(cells, count * 3);
        for i in 0..count {
            let ue_x = slice[i * 3];
            let ue_y = slice[i * 3 + 1];
            let ue_z = slice[i * 3 + 2];
            // UE → voxel → cell.
            let voxel = crate::convert::from_ue_world_pos(ue_x, ue_y, ue_z, world_scale);
            let cell = (
                (voxel.x / cf).floor() as i32,
                (voxel.y / cf).floor() as i32,
                (voxel.z / cf).floor() as i32,
            );
            new_set.insert(cell);
        }
    }

    // Swap in the new snapshot as a fresh Arc. The write lock window is one
    // pointer assignment; in-flight path solves keep their own Arc clone, so
    // this call (on the UE game thread, ~10Hz) never waits on a solve and
    // never blocks one.
    if let Ok(mut guard) = engine.occupied_cells.write() {
        *guard = std::sync::Arc::new(new_set);
    }
}

