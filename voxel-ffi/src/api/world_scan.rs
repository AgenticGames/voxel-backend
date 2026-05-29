//! Auto-split from the former monolithic `api.rs`. See `mod.rs`.
#[allow(unused_imports)]
use std::ffi::{c_char, c_void, CString};
#[allow(unused_imports)]
use std::ptr;
#[allow(unused_imports)]
use crate::engine::VoxelEngine;
#[allow(unused_imports)]
use crate::types::*;
use crate::engine::ffi_scan_config_to_scan_config;

/// Request a world scan. The scan runs on a worker thread and the result is
/// polled via `voxel_poll_scan_result`. Returns 1 on success, 0 if queue full.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_world_scan(engine: *mut c_void) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.request_world_scan()
}

/// Request a world scan with custom configuration.
/// Returns 1 on success, 0 if queue full.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_world_scan_with_config(
    engine: *mut c_void,
    config: *const FfiScanConfig,
) -> u32 {
    if engine.is_null() || config.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let ffi_config = &*config;
    let scan_config = ffi_scan_config_to_scan_config(ffi_config);
    engine.request_world_scan_with_config(scan_config)
}

/// Poll for a completed world scan result.
/// Returns success=0 if not ready, success=1 with heap-allocated JSON string if ready.
/// Caller MUST call `voxel_free_scan_result` on a successful result.
#[no_mangle]
pub unsafe extern "C" fn voxel_poll_scan_result(engine: *mut c_void) -> FfiWorldScanResult {
    let empty = FfiWorldScanResult {
        success: 0,
        json_report: ptr::null_mut(),
        json_length: 0,
        chunks_scanned: 0,
        total_issues: 0,
        total_errors: 0,
        total_warnings: 0,
    };
    if engine.is_null() {
        return empty;
    }
    let engine = &*(engine as *const VoxelEngine);
    match engine.poll_scan_complete() {
        Some(json) => {
            // Parse summary stats from the JSON for convenience
            let (chunks_scanned, total_issues, total_errors, total_warnings) =
                parse_scan_summary(&json);
            let json_len = json.len() as u32;
            match CString::new(json) {
                Ok(cstr) => {
                    let ptr = cstr.into_raw();
                    FfiWorldScanResult {
                        success: 1,
                        json_report: ptr,
                        json_length: json_len,
                        chunks_scanned,
                        total_issues,
                        total_errors,
                        total_warnings,
                    }
                }
                Err(_) => empty,
            }
        }
        None => empty,
    }
}

/// Free a scan result's JSON string. Safe to call with null pointer.
#[no_mangle]
pub unsafe extern "C" fn voxel_free_scan_result(result: *mut FfiWorldScanResult) {
    if result.is_null() {
        return;
    }
    let r = &*result;
    if !r.json_report.is_null() {
        drop(CString::from_raw(r.json_report));
    }
}

/// Parse summary stats from a scan JSON report string.
fn parse_scan_summary(json: &str) -> (u32, u32, u32, u32) {
    // Quick parse using serde_json::Value to extract summary fields
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(json) {
        let chunks = val.get("chunks_scanned").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let issues = val.get("issues").and_then(|v| v.as_array()).map(|a| a.len() as u32).unwrap_or(0);
        let summary = val.get("summary");
        let errors = summary.and_then(|s| s.get("total_errors")).and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let warnings = summary.and_then(|s| s.get("total_warnings")).and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        (chunks, issues, errors, warnings)
    } else {
        (0, 0, 0, 0)
    }
}

/// Free a sleep result's allocated memory (dirty_chunks and collapse_events arrays).
/// Safe to call with null pointers or zero counts.
#[no_mangle]
pub unsafe extern "C" fn voxel_free_sleep_result(result: *mut FfiSleepResult) {
    if result.is_null() {
        return;
    }
    let r = &*result;
    if !r.dirty_chunks.is_null() && r.dirty_chunk_count > 0 {
        // Reclaim FFI-allocated memory
        let _ = Vec::from_raw_parts(
            r.dirty_chunks,
            r.dirty_chunk_count as usize,
            r.dirty_chunk_count as usize,
        );
    }
    if !r.collapse_events.is_null() && r.collapse_event_count > 0 {
        // Reclaim FFI-allocated memory
        let _ = Vec::from_raw_parts(
            r.collapse_events,
            r.collapse_event_count as usize,
            r.collapse_event_count as usize,
        );
    }
    if !r.profile_report.is_null() {
        drop(CString::from_raw(r.profile_report));
    }
    if !r.manifest_json.is_null() {
        drop(CString::from_raw(r.manifest_json));
    }
    if !r.aureole_block.is_null() && r.aureole_block_count > 0 {
        // Reclaim FFI-allocated memory
        let _ = Vec::from_raw_parts(
            r.aureole_block,
            r.aureole_block_count as usize,
            r.aureole_block_count as usize,
        );
    }
    if !r.lava_cells.is_null() && r.lava_cell_count > 0 {
        // Reclaim FFI-allocated memory
        let _ = Vec::from_raw_parts(
            r.lava_cells,
            r.lava_cell_count as usize,
            r.lava_cell_count as usize,
        );
    }
    if !r.surface_changed_cells.is_null() && r.surface_changed_cell_count > 0 {
        // Reclaim FFI-allocated memory
        let _ = Vec::from_raw_parts(
            r.surface_changed_cells,
            r.surface_changed_cell_count as usize,
            r.surface_changed_cell_count as usize,
        );
    }
}

