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
            WorkerResult::MinedMaterials { mined } => {
                let result = FfiResult {
                    result_type: FfiResultType::MineResult,
                    chunk: FfiChunkCoord { x: 0, y: 0, z: 0 },
                    mesh: empty_mesh_data(),
                    mined,
                    generation: 0,
                    fluid_mesh: empty_fluid_mesh_data(),
                };
                Box::into_raw(Box::new(result))
            }
            WorkerResult::FluidMesh { chunk, mesh } => {
                let ue = rust_chunk_to_ue(chunk.0, chunk.1, chunk.2);
                let result = FfiResult {
                    result_type: FfiResultType::FluidMesh,
                    chunk: FfiChunkCoord { x: ue.0, y: ue.1, z: ue.2 },
                    mesh: empty_mesh_data(),
                    mined: FfiMinedMaterials { counts: [0; 19] },
                    generation: 0,
                    fluid_mesh: converted_fluid_mesh_to_ffi(mesh),
                };
                Box::into_raw(Box::new(result))
            }
            WorkerResult::SolidifyRequest { .. } => {
                // SolidifyRequest is handled engine-internally; skip for now
                ptr::null_mut()
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

    // Free fluid mesh data if present
    let fluid = &result.fluid_mesh;
    if fluid.vertex_count > 0 {
        if !fluid.positions.is_null() {
            drop(Vec::from_raw_parts(
                fluid.positions,
                fluid.vertex_count as usize,
                fluid.vertex_count as usize,
            ));
        }
        if !fluid.normals.is_null() {
            drop(Vec::from_raw_parts(
                fluid.normals,
                fluid.vertex_count as usize,
                fluid.vertex_count as usize,
            ));
        }
        if !fluid.fluid_types.is_null() {
            drop(Vec::from_raw_parts(
                fluid.fluid_types,
                fluid.vertex_count as usize,
                fluid.vertex_count as usize,
            ));
        }
    }
    if fluid.index_count > 0 && !fluid.indices.is_null() {
        drop(Vec::from_raw_parts(
            fluid.indices,
            fluid.index_count as usize,
            fluid.index_count as usize,
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

/// Inject fluid at a UE world position.
/// fluid_type: 1=Water, 2=Lava. is_source: 1=infinite source, 0=finite.
/// Returns 1 on success, 0 if queue full.
#[no_mangle]
pub unsafe extern "C" fn voxel_add_fluid(
    engine: *mut c_void,
    world_x: f32,
    world_y: f32,
    world_z: f32,
    fluid_type: u8,
    is_source: u8,
    world_scale: f32,
) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.add_fluid(world_x, world_y, world_z, fluid_type, is_source != 0, world_scale)
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
        fluid_mesh: empty_fluid_mesh_data(),
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

fn empty_fluid_mesh_data() -> FfiFluidMeshData {
    FfiFluidMeshData {
        positions: ptr::null_mut(),
        normals: ptr::null_mut(),
        fluid_types: ptr::null_mut(),
        vertex_count: 0,
        indices: ptr::null_mut(),
        index_count: 0,
    }
}

fn converted_fluid_mesh_to_ffi(mesh: ConvertedFluidMesh) -> FfiFluidMeshData {
    let vertex_count = mesh.positions.len() as u32;
    let index_count = mesh.indices.len() as u32;

    let mut positions = mesh.positions.into_boxed_slice();
    let mut normals = mesh.normals.into_boxed_slice();
    let mut fluid_types = mesh.fluid_types.into_boxed_slice();
    let mut indices = mesh.indices.into_boxed_slice();

    let positions_ptr = positions.as_mut_ptr();
    let normals_ptr = normals.as_mut_ptr();
    let fluid_types_ptr = fluid_types.as_mut_ptr();
    let indices_ptr = indices.as_mut_ptr();

    std::mem::forget(positions);
    std::mem::forget(normals);
    std::mem::forget(fluid_types);
    std::mem::forget(indices);

    FfiFluidMeshData {
        positions: positions_ptr,
        normals: normals_ptr,
        fluid_types: fluid_types_ptr,
        vertex_count,
        indices: indices_ptr,
        index_count,
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
            max_edge_length: 5.0,
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
            // Host Rock
            host_sandstone_depth: 200.0,
            host_granite_depth: 160.0,
            host_basalt_depth: 20.0,
            host_slate_depth: -150.0,
            host_boundary_noise_amp: 8.0,
            host_boundary_noise_freq: 0.03,
            host_basalt_intrusion_freq: 0.02,
            host_basalt_intrusion_thresh: 0.85,
            host_basalt_intrusion_depth_max: 10.0,
            // Banded Iron
            iron_band_frequency: 0.2,
            iron_noise_perturbation: 1.0,
            iron_noise_frequency: 0.15,
            iron_threshold: 1.2,
            iron_depth_min: -200.0,
            iron_depth_max: 200.0,
            // Copper
            copper_frequency: 0.009,
            copper_threshold: 0.72,
            copper_depth_min: -30.0,
            copper_depth_max: 200.0,
            // Malachite
            malachite_frequency: 0.8,
            malachite_threshold: 0.1,
            malachite_depth_min: -200.0,
            malachite_depth_max: -30.0,
            // Quartz
            quartz_frequency: 0.01,
            quartz_threshold: 0.67,
            quartz_depth_min: -200.0,
            quartz_depth_max: 200.0,
            // Gold
            gold_frequency: 0.08,
            gold_threshold: 0.87,
            gold_depth_min: -200.0,
            gold_depth_max: 200.0,
            // Pyrite
            pyrite_frequency: 0.05,
            pyrite_threshold: 0.92,
            pyrite_depth_min: -200.0,
            pyrite_depth_max: 200.0,
            // Kimberlite
            kimb_pipe_freq_2d: 0.008,
            kimb_pipe_threshold: 0.9,
            kimb_depth_min: -200.0,
            kimb_depth_max: -30.0,
            kimb_diamond_threshold: 0.75,
            kimb_diamond_frequency: 0.2,
            // Sulfide
            sulfide_frequency: 0.5,
            sulfide_threshold: 0.2,
            sulfide_tin_threshold: 0.5,
            sulfide_depth_min: -200.0,
            sulfide_depth_max: -20.0,
            // Geode
            geode_frequency: 0.009,
            geode_center_threshold: 0.94,
            geode_shell_thickness: 0.01,
            geode_hollow_factor: -0.20,
            geode_depth_min: -200.0,
            geode_depth_max: 200.0,
            // Fluid
            fluid_tick_rate: 15.0,
            fluid_lava_tick_divisor: 4,
            fluid_water_spring_threshold: 0.97,
            fluid_lava_source_threshold: 0.98,
            fluid_lava_depth_max: -50.0,
            fluid_water_noise_frequency: 0.05,
            fluid_water_depth_min: -9999.0,
            fluid_water_depth_max: 9999.0,
            fluid_water_flow_rate: 0.25,
            fluid_water_spread_rate: 0.125,
            fluid_lava_noise_frequency: 0.03,
            fluid_lava_depth_min: -9999.0,
            fluid_lava_flow_rate: 0.1,
            fluid_lava_spread_rate: 0.125,
            fluid_cavern_source_bias: 0.0,
            fluid_tunnel_bend_threshold: 0.0,
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
