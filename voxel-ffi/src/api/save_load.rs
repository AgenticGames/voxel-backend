//! Auto-split from the former monolithic `api.rs`. See `mod.rs`.
#[allow(unused_imports)]
use std::ffi::{c_char, c_void, CString};
#[allow(unused_imports)]
use std::ptr;
#[allow(unused_imports)]
use crate::engine::VoxelEngine;
#[allow(unused_imports)]
use crate::types::*;

// ── Save/Load FFI ─────────────��────────────────────────────────────────

/// Export world modification data (mined/flattened/sleep-modified chunks) as a binary buffer.
/// Returns a heap-allocated buffer; caller must free via `voxel_free_save_buffer`.
/// Writes the buffer length to `out_len`.
/// Returns null if engine is null or no modifications exist.
#[no_mangle]
pub unsafe extern "C" fn voxel_save_world_data(
    engine: *mut c_void,
    out_len: *mut u32,
) -> *mut u8 {
    if engine.is_null() || out_len.is_null() {
        return ptr::null_mut();
    }
    let engine = &*(engine as *const VoxelEngine);
    let bytes = engine.export_save_data();
    if bytes.is_empty() {
        *out_len = 0;
        return ptr::null_mut();
    }
    *out_len = bytes.len() as u32;
    let mut boxed = bytes.into_boxed_slice();
    let ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);
    ptr
}

/// Free a save buffer returned by `voxel_save_world_data`.
#[no_mangle]
pub unsafe extern "C" fn voxel_free_save_buffer(buffer: *mut u8, len: u32) {
    if buffer.is_null() || len == 0 {
        return;
    }
    let _ = Vec::from_raw_parts(buffer, len as usize, len as usize);
}

/// Load world modification data from a binary buffer.
/// Must be called BEFORE chunk streaming begins so snapshots are applied during generation.
/// Returns 1 on success, 0 on failure (corrupt data or null engine).
#[no_mangle]
pub unsafe extern "C" fn voxel_load_world_data(
    engine: *mut c_void,
    buffer: *const u8,
    len: u32,
) -> u32 {
    if engine.is_null() || buffer.is_null() || len == 0 {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let data = std::slice::from_raw_parts(buffer, len as usize);
    if engine.import_save_data(data) { 1 } else { 0 }
}

/// Export player-placed fluid (lava/water sources, brush-painted pools)
/// as a binary buffer. Returns a heap-allocated buffer; caller must free
/// via `voxel_free_save_buffer` (same allocator as world data).
/// Returns null if engine is null or no fluid to save.
#[no_mangle]
pub unsafe extern "C" fn voxel_save_fluid_data(
    engine: *mut c_void,
    out_len: *mut u32,
) -> *mut u8 {
    if engine.is_null() || out_len.is_null() {
        return ptr::null_mut();
    }
    let engine = &*(engine as *const VoxelEngine);
    let bytes = engine.export_fluid_data();
    if bytes.is_empty() {
        *out_len = 0;
        return ptr::null_mut();
    }
    *out_len = bytes.len() as u32;
    let mut boxed = bytes.into_boxed_slice();
    let ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);
    ptr
}

/// Load player-placed fluid state from a binary buffer. Cells are queued
/// in the fluid thread and applied per-chunk as the chunks stream in.
/// Returns 1 on success (or empty payload), 0 on parse error.
#[no_mangle]
pub unsafe extern "C" fn voxel_load_fluid_data(
    engine: *mut c_void,
    buffer: *const u8,
    len: u32,
) -> u32 {
    if engine.is_null() {
        return 0;
    }
    if buffer.is_null() || len == 0 {
        return 1; // empty payload — nothing to do
    }
    let engine = &*(engine as *const VoxelEngine);
    let data = std::slice::from_raw_parts(buffer, len as usize);
    if engine.import_fluid_data(data) { 1 } else { 0 }
}

