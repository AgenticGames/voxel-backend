//! Auto-split from the former monolithic `api.rs`. See `mod.rs`.
#[allow(unused_imports)]
use std::ffi::{c_char, c_void, CString};
#[allow(unused_imports)]
use std::ptr;
#[allow(unused_imports)]
use crate::engine::VoxelEngine;
#[allow(unused_imports)]
use crate::types::*;

// ── Morph Step ──

/// Request a morph step for progressive showcase morphing.
/// chunks: pointer to array of FfiChunkCoord (Rust chunk coords)
/// chunk_count: number of chunks (typically 8)
/// Cache the morph manifest JSON (deserialized once, reused for all morph steps).
/// Must be called before voxel_request_morph_step. Returns 1 on success, 0 on parse error.
#[no_mangle]
pub unsafe extern "C" fn voxel_set_morph_manifest(
    engine: *mut c_void,
    manifest_json: *const std::ffi::c_char,
    manifest_len: u32,
) -> u32 {
    if engine.is_null() || manifest_json.is_null() { return 0; }
    let engine = &*(engine as *const VoxelEngine);
    let json_bytes = std::slice::from_raw_parts(manifest_json as *const u8, manifest_len as usize);
    let json_str = match std::str::from_utf8(json_bytes) {
        Ok(s) => s,
        Err(_) => return 0,
    };
    if engine.set_morph_manifest(json_str) { 1 } else { 0 }
}

/// Clear cached morph manifest (call after morph sequence completes).
#[no_mangle]
pub unsafe extern "C" fn voxel_clear_morph_manifest(engine: *mut c_void) {
    if engine.is_null() { return; }
    let engine = &*(engine as *const VoxelEngine);
    engine.clear_morph_manifest();
}

/// Request a morph step using the cached manifest.
/// step: current step (0..total_steps)
/// total_steps: total number of morph steps
/// Returns 1 on success, 0 if queue full.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_morph_step(
    engine: *mut c_void,
    chunks: *const FfiChunkCoord,
    chunk_count: u32,
    step: u32,
    total_steps: u32,
) -> u32 {
    if engine.is_null() || chunks.is_null() || chunk_count == 0 {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);

    let chunk_slice = std::slice::from_raw_parts(chunks, chunk_count as usize);
    let chunk_vec: Vec<(i32, i32, i32)> = chunk_slice.iter()
        .map(|c| (c.x, c.y, c.z))
        .collect();

    engine.request_morph_step(chunk_vec, step, total_steps)
}

/// Poll for a completed morph step result.
/// Returns null if no result ready, or a heap-allocated FfiMorphResult.
/// Caller MUST call voxel_free_morph_result when done.
#[no_mangle]
pub unsafe extern "C" fn voxel_poll_morph_result(engine: *mut c_void) -> *mut FfiMorphResult {
    if engine.is_null() {
        return ptr::null_mut();
    }
    let engine = &*(engine as *const VoxelEngine);

    match engine.poll_morph_result() {
        Some(result) => {
            let chunk_count = result.meshes.len() as u32;

            // Allocate array of FfiMeshData
            let mut mesh_array: Vec<FfiMeshData> = Vec::with_capacity(result.meshes.len());
            for converted in result.meshes {
                let vert_count = converted.positions.len() as u32;
                let idx_count = converted.indices.len() as u32;
                let sub_count = converted.submeshes.len() as u32;

                let mut positions = converted.positions.into_boxed_slice();
                let mut normals = converted.normals.into_boxed_slice();
                let mut material_ids = converted.material_ids.into_boxed_slice();
                let mut indices = converted.indices.into_boxed_slice();
                let mut submeshes = converted.submeshes.into_boxed_slice();

                let mesh = FfiMeshData {
                    positions: positions.as_mut_ptr(),
                    normals: normals.as_mut_ptr(),
                    material_ids: material_ids.as_mut_ptr(),
                    vertex_count: vert_count,
                    indices: indices.as_mut_ptr(),
                    index_count: idx_count,
                    submeshes: submeshes.as_mut_ptr(),
                    submesh_count: sub_count,
                };

                // Leak the boxes so FFI owns them
                std::mem::forget(positions);
                std::mem::forget(normals);
                std::mem::forget(material_ids);
                std::mem::forget(indices);
                std::mem::forget(submeshes);

                mesh_array.push(mesh);
            }

            let mut mesh_box = mesh_array.into_boxed_slice();
            let meshes_ptr = mesh_box.as_mut_ptr();
            std::mem::forget(mesh_box);

            let ffi_result = Box::new(FfiMorphResult {
                step: result.step,
                total_steps: result.total_steps,
                chunk_count,
                meshes: meshes_ptr,
            });
            Box::into_raw(ffi_result)
        }
        None => ptr::null_mut(),
    }
}

/// Free a morph result and all its internal mesh data.
#[no_mangle]
pub unsafe extern "C" fn voxel_free_morph_result(result: *mut FfiMorphResult) {
    if result.is_null() {
        return;
    }
    let r = Box::from_raw(result);
    if !r.meshes.is_null() && r.chunk_count > 0 {
        let mesh_slice = std::slice::from_raw_parts_mut(r.meshes, r.chunk_count as usize);
        // Reclaim FFI-allocated memory for each mesh's buffers
        for mesh in mesh_slice.iter() {
            if !mesh.positions.is_null() && mesh.vertex_count > 0 {
                let _ = Vec::from_raw_parts(mesh.positions, mesh.vertex_count as usize, mesh.vertex_count as usize);
            }
            if !mesh.normals.is_null() && mesh.vertex_count > 0 {
                let _ = Vec::from_raw_parts(mesh.normals, mesh.vertex_count as usize, mesh.vertex_count as usize);
            }
            if !mesh.material_ids.is_null() && mesh.vertex_count > 0 {
                let _ = Vec::from_raw_parts(mesh.material_ids, mesh.vertex_count as usize, mesh.vertex_count as usize);
            }
            if !mesh.indices.is_null() && mesh.index_count > 0 {
                let _ = Vec::from_raw_parts(mesh.indices, mesh.index_count as usize, mesh.index_count as usize);
            }
            if !mesh.submeshes.is_null() && mesh.submesh_count > 0 {
                let _ = Vec::from_raw_parts(mesh.submeshes, mesh.submesh_count as usize, mesh.submesh_count as usize);
            }
        }
        // Reclaim the mesh array itself
        let _ = Vec::from_raw_parts(r.meshes, r.chunk_count as usize, r.chunk_count as usize);
    }
}

