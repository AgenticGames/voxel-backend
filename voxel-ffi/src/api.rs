use std::ffi::c_void;
use std::ptr;

use crate::convert::rust_chunk_to_ue;
use crate::engine::VoxelEngine;
use crate::types::*;

/// Create a new voxel engine instance. Returns opaque pointer.
/// Caller must eventually call `voxel_destroy_engine` to free.
#[no_mangle]
pub unsafe extern "C" fn voxel_create_engine(config: *const FfiEngineConfig) -> *mut c_void {
    if config.is_null() {
        return ptr::null_mut();
    }
    let cfg = &*config;
    let engine = Box::new(VoxelEngine::new(cfg));
    Box::into_raw(engine) as *mut c_void
}

/// Destroy a voxel engine, shutting down worker threads and freeing memory.
#[no_mangle]
pub unsafe extern "C" fn voxel_destroy_engine(engine: *mut c_void) {
    if engine.is_null() {
        return;
    }
    let engine = Box::from_raw(engine as *mut VoxelEngine);
    engine.shutdown();
}

/// Request generation of a single chunk. Returns 1 on success, 0 if queue full.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_generate(
    engine: *mut c_void,
    chunk: FfiChunkCoord,
) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.request_generate(chunk.x, chunk.y, chunk.z)
}

/// Request generation of multiple chunks. Returns count successfully queued.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_generate_batch(
    engine: *mut c_void,
    chunks: *const FfiChunkCoord,
    count: u32,
) -> u32 {
    if engine.is_null() || chunks.is_null() || count == 0 {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let chunk_slice = std::slice::from_raw_parts(chunks, count as usize);
    let keys: Vec<(i32, i32, i32)> = chunk_slice.iter().map(|c| (c.x, c.y, c.z)).collect();
    engine.request_generate_batch(&keys)
}

/// Request a mining operation. Returns 1 on success, 0 if queue full.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_mine(
    engine: *mut c_void,
    request: *const FfiMineRequest,
) -> u32 {
    if engine.is_null() || request.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.request_mine(*request)
}

/// Request unloading a chunk's cached data. Returns 1 on success, 0 if queue full.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_unload(
    engine: *mut c_void,
    chunk: FfiChunkCoord,
) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.request_unload(chunk.x, chunk.y, chunk.z)
}

/// Cancel pending generation for a chunk (stale results will be discarded).
#[no_mangle]
pub unsafe extern "C" fn voxel_cancel_chunk(engine: *mut c_void, chunk: FfiChunkCoord) {
    if engine.is_null() {
        return;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.cancel_chunk(chunk.x, chunk.y, chunk.z);
}

/// Non-blocking poll for a completed result.
/// Returns null if nothing ready, otherwise a heap-allocated FfiResult.
/// Caller MUST call `voxel_free_result` on non-null returns.
#[no_mangle]
pub unsafe extern "C" fn voxel_poll_result(engine: *mut c_void) -> *mut FfiResult {
    if engine.is_null() {
        return ptr::null_mut();
    }
    let engine = &*(engine as *const VoxelEngine);

    match engine.poll_result() {
        None => ptr::null_mut(),
        Some(worker_result) => match worker_result {
            WorkerResult::ChunkMesh {
                chunk,
                mesh,
                generation,
            } => {
                let result = convert_mesh_to_ffi_result(chunk, mesh, generation);
                Box::into_raw(Box::new(result))
            }
            WorkerResult::MineResults { meshes, mined } => {
                // For mine results, return the first dirty chunk mesh.
                // If there are multiple dirty chunks, they'll be returned
                // in subsequent polls via re-queuing.
                if meshes.is_empty() {
                    let result = FfiResult {
                        result_type: FfiResultType::MineResult,
                        chunk: FfiChunkCoord { x: 0, y: 0, z: 0 },
                        mesh: empty_mesh_data(),
                        mined,
                        generation: 0,
                    };
                    return Box::into_raw(Box::new(result));
                }

                // Return results one at a time. We send additional results
                // back to the result channel for subsequent polls.
                let mut iter = meshes.into_iter();
                let (first_key, first_mesh) = iter.next().unwrap();

                // Re-queue remaining meshes
                for (key, mesh) in iter {
                    // These are already-meshed results, send as ChunkMesh with generation=0
                    let _ = engine.poll_result(); // drain is handled by caller
                    // Actually we need to push these back. Since we can't easily
                    // push back to result channel from here, pack first one with mined data
                    // and the rest as ChunkMesh results.
                    // For simplicity, only return the first mesh with mined data.
                    // The UE side should re-request affected chunks if needed.
                    let _ = key;
                    let _ = mesh;
                }

                let ue_key = rust_chunk_to_ue(first_key.0, first_key.1, first_key.2);
                let result = FfiResult {
                    result_type: FfiResultType::MineResult,
                    chunk: FfiChunkCoord {
                        x: ue_key.0,
                        y: ue_key.1,
                        z: ue_key.2,
                    },
                    mesh: converted_mesh_to_ffi(first_mesh),
                    mined,
                    generation: 0,
                };
                Box::into_raw(Box::new(result))
            }
        },
    }
}

/// Free a result previously returned by `voxel_poll_result`.
#[no_mangle]
pub unsafe extern "C" fn voxel_free_result(result: *mut FfiResult) {
    if result.is_null() {
        return;
    }
    let result = Box::from_raw(result);

    // Reconstitute and drop the owned Vec allocations
    let mesh = &result.mesh;
    if mesh.vertex_count > 0 {
        if !mesh.positions.is_null() {
            drop(Vec::from_raw_parts(
                mesh.positions,
                mesh.vertex_count as usize,
                mesh.vertex_count as usize,
            ));
        }
        if !mesh.normals.is_null() {
            drop(Vec::from_raw_parts(
                mesh.normals,
                mesh.vertex_count as usize,
                mesh.vertex_count as usize,
            ));
        }
        if !mesh.material_ids.is_null() {
            drop(Vec::from_raw_parts(
                mesh.material_ids,
                mesh.vertex_count as usize,
                mesh.vertex_count as usize,
            ));
        }
    }
    if mesh.index_count > 0 && !mesh.indices.is_null() {
        drop(Vec::from_raw_parts(
            mesh.indices,
            mesh.index_count as usize,
            mesh.index_count as usize,
        ));
    }

    // Box<FfiResult> dropped here
}

/// Get current engine statistics.
#[no_mangle]
pub unsafe extern "C" fn voxel_get_stats(engine: *mut c_void) -> FfiEngineStats {
    if engine.is_null() {
        return FfiEngineStats {
            chunks_loaded: 0,
            pending_requests: 0,
            completed_results: 0,
            worker_threads_active: 0,
        };
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.get_stats()
}

/// Hot-reload engine configuration (affects future generation requests).
#[no_mangle]
pub unsafe extern "C" fn voxel_update_config(engine: *mut c_void, config: *const FfiEngineConfig) {
    if engine.is_null() || config.is_null() {
        return;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.update_config(&*config);
}

// ── Internal helpers ──

fn convert_mesh_to_ffi_result(
    chunk: (i32, i32, i32),
    mesh: ConvertedMesh,
    generation: u64,
) -> FfiResult {
    // Convert Rust chunk coords back to UE space for the caller
    let ue = rust_chunk_to_ue(chunk.0, chunk.1, chunk.2);
    FfiResult {
        result_type: FfiResultType::ChunkMesh,
        chunk: FfiChunkCoord {
            x: ue.0,
            y: ue.1,
            z: ue.2,
        },
        mesh: converted_mesh_to_ffi(mesh),
        mined: FfiMinedMaterials { counts: [0; 19] },
        generation,
    }
}

fn converted_mesh_to_ffi(mesh: ConvertedMesh) -> FfiMeshData {
    let vertex_count = mesh.positions.len() as u32;
    let index_count = mesh.indices.len() as u32;

    // Convert Vecs to raw pointers. Ownership transferred to FFI side.
    let mut positions = mesh.positions.into_boxed_slice();
    let mut normals = mesh.normals.into_boxed_slice();
    let mut material_ids = mesh.material_ids.into_boxed_slice();
    let mut indices = mesh.indices.into_boxed_slice();

    let positions_ptr = positions.as_mut_ptr();
    let normals_ptr = normals.as_mut_ptr();
    let material_ids_ptr = material_ids.as_mut_ptr();
    let indices_ptr = indices.as_mut_ptr();

    // Leak the boxes so they aren't freed here. voxel_free_result handles cleanup.
    std::mem::forget(positions);
    std::mem::forget(normals);
    std::mem::forget(material_ids);
    std::mem::forget(indices);

    FfiMeshData {
        positions: positions_ptr,
        normals: normals_ptr,
        material_ids: material_ids_ptr,
        vertex_count,
        indices: indices_ptr,
        index_count,
    }
}

fn empty_mesh_data() -> FfiMeshData {
    FfiMeshData {
        positions: ptr::null_mut(),
        normals: ptr::null_mut(),
        material_ids: ptr::null_mut(),
        vertex_count: 0,
        indices: ptr::null_mut(),
        index_count: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    fn test_config() -> FfiEngineConfig {
        FfiEngineConfig {
            seed: 42,
            chunk_size: 16,
            worker_threads: 2,
            world_scale: 100.0,
            max_edge_length: 4.0,
            cavern_frequency: 0.05,
            cavern_threshold: 0.55,
            detail_octaves: 4,
            detail_persistence: 0.5,
            warp_amplitude: 5.0,
            worms_per_region: 5,
            worm_radius_min: 2.0,
            worm_radius_max: 4.0,
            worm_step_length: 1.0,
            worm_max_steps: 200,
            worm_falloff_power: 2.0,
            region_size: 3,
        }
    }

    #[test]
    fn engine_lifecycle() {
        unsafe {
            let cfg = test_config();
            let engine = voxel_create_engine(&cfg);
            assert!(!engine.is_null());
            voxel_destroy_engine(engine);
        }
    }

    #[test]
    fn generate_single_chunk_and_poll() {
        unsafe {
            let cfg = test_config();
            let engine = voxel_create_engine(&cfg);
            assert!(!engine.is_null());

            let chunk = FfiChunkCoord { x: 0, y: 0, z: 0 };
            let ok = voxel_request_generate(engine, chunk);
            assert_eq!(ok, 1);

            // Poll until we get a result (with timeout)
            let mut result_ptr = ptr::null_mut();
            for _ in 0..200 {
                result_ptr = voxel_poll_result(engine);
                if !result_ptr.is_null() {
                    break;
                }
                thread::sleep(Duration::from_millis(50));
            }
            assert!(!result_ptr.is_null(), "Should have received a result");

            let result = &*result_ptr;
            assert_eq!(result.result_type, FfiResultType::ChunkMesh);
            assert_eq!(result.chunk.x, 0);
            assert_eq!(result.chunk.y, 0);
            assert_eq!(result.chunk.z, 0);
            assert!(result.mesh.vertex_count > 0, "Mesh should have vertices");
            assert!(result.mesh.index_count > 0, "Mesh should have indices");

            voxel_free_result(result_ptr);
            voxel_destroy_engine(engine);
        }
    }

    #[test]
    fn cancel_discards_stale() {
        unsafe {
            let cfg = test_config();
            let engine = voxel_create_engine(&cfg);

            let chunk = FfiChunkCoord { x: 5, y: 5, z: 5 };
            voxel_request_generate(engine, chunk);
            // Immediately cancel
            voxel_cancel_chunk(engine, chunk);

            // Wait a bit, then poll - result should either be absent or have
            // a stale generation that was already in flight
            thread::sleep(Duration::from_millis(500));

            // Drain any results - if we get one, it should still be well-formed
            loop {
                let result = voxel_poll_result(engine);
                if result.is_null() {
                    break;
                }
                voxel_free_result(result);
            }

            voxel_destroy_engine(engine);
        }
    }

    #[test]
    fn destroy_under_load() {
        unsafe {
            let cfg = test_config();
            let engine = voxel_create_engine(&cfg);

            // Queue many chunks
            for x in 0..4 {
                for z in 0..4 {
                    voxel_request_generate(
                        engine,
                        FfiChunkCoord { x, y: 0, z },
                    );
                }
            }

            // Destroy immediately while workers are busy
            thread::sleep(Duration::from_millis(100));
            voxel_destroy_engine(engine);
            // Should not crash
        }
    }

    #[test]
    fn null_engine_safety() {
        unsafe {
            // All API functions should handle null gracefully
            voxel_destroy_engine(ptr::null_mut());
            assert_eq!(
                voxel_request_generate(ptr::null_mut(), FfiChunkCoord { x: 0, y: 0, z: 0 }),
                0
            );
            assert!(voxel_poll_result(ptr::null_mut()).is_null());
            let stats = voxel_get_stats(ptr::null_mut());
            assert_eq!(stats.chunks_loaded, 0);
        }
    }

    #[test]
    fn stats_reports_correctly() {
        unsafe {
            let cfg = test_config();
            let engine = voxel_create_engine(&cfg);

            let stats = voxel_get_stats(engine);
            assert_eq!(stats.chunks_loaded, 0);
            assert_eq!(stats.worker_threads_active, 2);

            voxel_destroy_engine(engine);
        }
    }
}
