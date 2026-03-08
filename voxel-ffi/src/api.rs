use std::ffi::{c_char, c_void, CString};
use std::ptr;

use crate::convert::rust_chunk_to_ue;
use crate::engine::{ffi_scan_config_to_scan_config, VoxelEngine};
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
                crystal_data,
            } => {
                let result = convert_mesh_to_ffi_result(chunk, mesh, generation, crystal_data);
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
                    crystal_data: empty_crystal_data(),
                };
                Box::into_raw(Box::new(result))
            }
            WorkerResult::FluidMesh { chunk, mesh } => {
                let ue = rust_chunk_to_ue(chunk.0, chunk.1, chunk.2);
                let result = FfiResult {
                    result_type: FfiResultType::FluidMesh,
                    chunk: FfiChunkCoord { x: ue.0, y: ue.1, z: ue.2 },
                    mesh: empty_mesh_data(),
                    mined: FfiMinedMaterials { counts: [0; 22] },
                    generation: 0,
                    fluid_mesh: converted_fluid_mesh_to_ffi(mesh),
                    crystal_data: empty_crystal_data(),
                };
                Box::into_raw(Box::new(result))
            }
            WorkerResult::Error { chunk, generation } => {
                let ue = rust_chunk_to_ue(chunk.0, chunk.1, chunk.2);
                let result = FfiResult {
                    result_type: FfiResultType::Error,
                    chunk: FfiChunkCoord { x: ue.0, y: ue.1, z: ue.2 },
                    mesh: empty_mesh_data(),
                    mined: FfiMinedMaterials { counts: [0; 22] },
                    generation,
                    fluid_mesh: empty_fluid_mesh_data(),
                    crystal_data: empty_crystal_data(),
                };
                Box::into_raw(Box::new(result))
            }
            WorkerResult::SolidifyRequest { .. } => {
                // SolidifyRequest is handled engine-internally; skip for now
                ptr::null_mut()
            }
            WorkerResult::CollapseResult { events, meshes } => {
                // Send each collapse-remeshed chunk as a ChunkMesh result first
                for (chunk, mesh) in meshes {
                    let r = convert_mesh_to_ffi_result(chunk, mesh, 0, Vec::new());
                    let _ = Box::into_raw(Box::new(r));
                    // Note: in practice these go through the result channel,
                    // but for the poll API we emit the collapse event.
                }

                if events.is_empty() {
                    return ptr::null_mut();
                }

                // Return the first collapse event (UE polls repeatedly)
                let ev = events[0];
                let result = FfiResult {
                    result_type: FfiResultType::CollapseResult,
                    chunk: FfiChunkCoord {
                        x: ev.center_x as i32,
                        y: ev.center_y as i32,
                        z: ev.center_z as i32,
                    },
                    mesh: empty_mesh_data(),
                    mined: FfiMinedMaterials { counts: [0; 22] },
                    generation: ev.volume as u64,
                    fluid_mesh: empty_fluid_mesh_data(),
                    crystal_data: empty_crystal_data(),
                };
                Box::into_raw(Box::new(result))
            }
            WorkerResult::SupportResult { success, meshes } => {
                // Send each remeshed chunk as a ChunkMesh
                for (chunk, mesh) in meshes {
                    let r = convert_mesh_to_ffi_result(chunk, mesh, 0, Vec::new());
                    let _ = Box::into_raw(Box::new(r));
                }
                // Return as a mine result with success indicator
                let result = FfiResult {
                    result_type: FfiResultType::MineResult,
                    chunk: FfiChunkCoord { x: 0, y: 0, z: 0 },
                    mesh: empty_mesh_data(),
                    mined: FfiMinedMaterials { counts: [if success { 1 } else { 0 }; 22] },
                    generation: 0,
                    fluid_mesh: empty_fluid_mesh_data(),
                    crystal_data: empty_crystal_data(),
                };
                Box::into_raw(Box::new(result))
            }
            WorkerResult::SleepComplete { .. } => {
                // This should have been intercepted by engine.poll_result().
                // If it somehow reaches here, ignore it.
                ptr::null_mut()
            }
            WorkerResult::ScanComplete { .. } => {
                // This should have been intercepted by engine.poll_result().
                // If it somehow reaches here, ignore it.
                ptr::null_mut()
            }
            WorkerResult::ForceSpawnPoolComplete { .. } => {
                // Intercepted by engine.poll_result(); ignore if it reaches here.
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
    if mesh.submesh_count > 0 && !mesh.submeshes.is_null() {
        drop(Vec::from_raw_parts(
            mesh.submeshes,
            mesh.submesh_count as usize,
            mesh.submesh_count as usize,
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
        if !fluid.uvs.is_null() {
            drop(Vec::from_raw_parts(
                fluid.uvs,
                fluid.vertex_count as usize,
                fluid.vertex_count as usize,
            ));
        }
        if !fluid.flow_directions.is_null() {
            drop(Vec::from_raw_parts(
                fluid.flow_directions,
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

    // Free crystal data if present
    let crystals = &result.crystal_data;
    if crystals.count > 0 && !crystals.placements.is_null() {
        drop(Vec::from_raw_parts(
            crystals.placements,
            crystals.count as usize,
            crystals.count as usize,
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

/// Find the best cavern spring location near the player.
/// Returns 1 if found (out pointers written), 0 if no suitable location.
/// All coordinates are UE world space.
#[no_mangle]
pub unsafe extern "C" fn voxel_find_spring(
    engine: *mut c_void,
    player_x: f32,
    player_y: f32,
    player_z: f32,
    world_scale: f32,
    out_x: *mut f32,
    out_y: *mut f32,
    out_z: *mut f32,
) -> u32 {
    if engine.is_null() || out_x.is_null() || out_y.is_null() || out_z.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    match engine.find_spring(player_x, player_y, player_z, world_scale) {
        Some((x, y, z)) => {
            *out_x = x;
            *out_y = y;
            *out_z = z;
            1
        }
        None => 0,
    }
}

/// Find a wall-adjacent air cell near a target, excluding a radius around a point.
/// Returns 1 if found (out pointers written), 0 if no suitable location.
/// All coordinates are UE world space.
#[no_mangle]
pub unsafe extern "C" fn voxel_find_wall_near(
    engine: *mut c_void,
    target_x: f32,
    target_y: f32,
    target_z: f32,
    exclude_x: f32,
    exclude_y: f32,
    exclude_z: f32,
    exclude_radius: f32,
    world_scale: f32,
    out_x: *mut f32,
    out_y: *mut f32,
    out_z: *mut f32,
) -> u32 {
    if engine.is_null() || out_x.is_null() || out_y.is_null() || out_z.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    match engine.find_wall_near(
        target_x, target_y, target_z,
        exclude_x, exclude_y, exclude_z,
        exclude_radius, world_scale,
    ) {
        Some((x, y, z)) => {
            *out_x = x;
            *out_y = y;
            *out_z = z;
            1
        }
        None => 0,
    }
}

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
            count: 0,
            valid: 0,
        };
    }
    let engine = &*(engine as *const VoxelEngine);
    let key = ue_chunk_to_rust(chunk_x, chunk_y, chunk_z);

    match engine.query_stress(key) {
        Some(sf) => {
            let count = sf.stress.len() as u32;
            let mut data = sf.stress.into_boxed_slice();
            let ptr = data.as_mut_ptr();
            std::mem::forget(data);
            FfiStressData {
                stress_values: ptr,
                count,
                valid: 1,
            }
        }
        None => FfiStressData {
            stress_values: ptr::null_mut(),
            count: 0,
            valid: 0,
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

    // Get world scale and chunk size from the engine config
    let chunk_size = 16usize; // Standard chunk size
    let world_scale = 15.0f32; // Standard world scale

    let rust_pos = from_ue_world_pos(world_x, world_y, world_z, world_scale);
    engine.query_stress_at(
        rust_pos.x as i32,
        rust_pos.y as i32,
        rust_pos.z as i32,
        chunk_size,
    )
}

/// Place a support structure at a UE world position.
/// support_type: 1=SlateStrut, 2=GraniteStrut, 3=LimestoneStrut, 4=CopperStrut,
///               5=IronStrut, 6=SteelStrut, 7=CrystalStrut.
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
    let world_scale = 15.0f32;
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
    let world_scale = 15.0f32;
    let rust_pos = from_ue_world_pos(world_x, world_y, world_z, world_scale);
    engine.request_remove_support(
        rust_pos.x as i32,
        rust_pos.y as i32,
        rust_pos.z as i32,
    )
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
    };

    engine.update_stress_config(stress_config);
}

/// Update the sleep configuration from FFI config fields.
#[no_mangle]
pub unsafe extern "C" fn voxel_set_sleep_config(
    engine: *mut c_void,
    config: *const FfiEngineConfig,
) {
    if engine.is_null() || config.is_null() {
        return;
    }
    let engine = &*(engine as *const VoxelEngine);
    let ffi_config = &*config;
    let sleep_config = crate::engine::ffi_config_to_sleep(ffi_config);
    engine.update_sleep_config(sleep_config);
}

/// Set spider nest positions for sleep fossilization. UE calls this before voxel_start_sleep().
/// Coordinates are in UE world space (will be converted to Rust space).
/// Returns 1 on success, 0 on error.
#[no_mangle]
pub unsafe extern "C" fn voxel_set_sleep_nests(
    engine: *mut c_void,
    world_xs: *const i32,
    world_ys: *const i32,
    world_zs: *const i32,
    count: u32,
) -> u32 {
    if engine.is_null() || (count > 0 && (world_xs.is_null() || world_ys.is_null() || world_zs.is_null())) {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let positions: Vec<(i32, i32, i32)> = (0..count as usize)
        .map(|i| {
            let ux = *world_xs.add(i);
            let uy = *world_ys.add(i);
            let uz = *world_zs.add(i);
            crate::convert::ue_chunk_to_rust(ux, uy, uz)
        })
        .collect();
    engine.set_sleep_nests(positions);
    1
}

/// Start a deep sleep cycle. player_chunk coordinates are in UE space.
/// Returns 1 on success, 0 if queue full.
#[no_mangle]
pub unsafe extern "C" fn voxel_start_sleep(
    engine: *mut c_void,
    player_cx: i32,
    player_cy: i32,
    player_cz: i32,
    sleep_count: u32,
) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let player_chunk = crate::convert::ue_chunk_to_rust(player_cx, player_cy, player_cz);
    engine.start_sleep(player_chunk, sleep_count)
}

/// Poll for a completed sleep result.
/// Returns an FfiSleepResult with success=1 if a result is available, success=0 otherwise.
/// Dirty chunk meshes and collapse events are delivered through the normal voxel_poll_result
/// pipeline; this function only returns the summary statistics.
#[no_mangle]
pub unsafe extern "C" fn voxel_poll_sleep_result(engine: *mut c_void) -> FfiSleepResult {
    let empty = FfiSleepResult {
        success: 0,
        chunks_changed: 0,
        voxels_metamorphosed: 0,
        minerals_grown: 0,
        supports_degraded: 0,
        collapses_triggered: 0,
        acid_dissolved: 0,
        veins_deposited: 0,
        voxels_enriched: 0,
        formations_grown: 0,
        sulfide_dissolved: 0,
        coal_matured: 0,
        diamonds_formed: 0,
        voxels_silicified: 0,
        nests_fossilized: 0,
        dirty_chunks: ptr::null_mut(),
        dirty_chunk_count: 0,
        collapse_events: ptr::null_mut(),
        collapse_event_count: 0,
        profile_report: ptr::null_mut(),
        profile_report_length: 0,
    };
    if engine.is_null() {
        return empty;
    }
    let engine = &*(engine as *const VoxelEngine);
    match engine.poll_sleep_complete() {
        Some(data) => {
            let (report_ptr, report_len) = match CString::new(data.profile_report) {
                Ok(cstr) => {
                    let len = cstr.as_bytes().len() as u32;
                    (cstr.into_raw(), len)
                }
                Err(_) => (ptr::null_mut(), 0u32),
            };
            FfiSleepResult {
                success: 1,
                chunks_changed: data.chunks_changed,
                voxels_metamorphosed: data.voxels_metamorphosed,
                minerals_grown: data.minerals_grown,
                supports_degraded: data.supports_degraded,
                collapses_triggered: data.collapses_triggered,
                acid_dissolved: data.acid_dissolved,
                veins_deposited: data.veins_deposited,
                voxels_enriched: data.voxels_enriched,
                formations_grown: data.formations_grown,
                sulfide_dissolved: data.sulfide_dissolved,
                coal_matured: data.coal_matured,
                diamonds_formed: data.diamonds_formed,
                voxels_silicified: data.voxels_silicified,
                nests_fossilized: data.nests_fossilized,
                dirty_chunks: ptr::null_mut(),
                dirty_chunk_count: 0,
                collapse_events: ptr::null_mut(),
                collapse_event_count: 0,
                profile_report: report_ptr,
                profile_report_length: report_len,
            }
        },
        None => empty,
    }
}

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
        let _ = Vec::from_raw_parts(
            r.dirty_chunks,
            r.dirty_chunk_count as usize,
            r.dirty_chunk_count as usize,
        );
    }
    if !r.collapse_events.is_null() && r.collapse_event_count > 0 {
        let _ = Vec::from_raw_parts(
            r.collapse_events,
            r.collapse_event_count as usize,
            r.collapse_event_count as usize,
        );
    }
    if !r.profile_report.is_null() {
        drop(CString::from_raw(r.profile_report));
    }
}

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

/// Query floor support for a building placement (4x4 footprint).
/// Writes solid_count, total_columns, and host material to out pointers.
/// Returns 1 on success, 0 if engine null.
#[no_mangle]
pub unsafe extern "C" fn voxel_query_building_support(
    engine: *mut c_void,
    x: f32,
    y: f32,
    z: f32,
    scale: f32,
    out_solid: *mut u8,
    out_total: *mut u8,
    out_mat: *mut u8,
) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    let (solid, total, mat) = engine.query_building_support(x, y, z, scale);
    if !out_solid.is_null() { *out_solid = solid; }
    if !out_total.is_null() { *out_total = total; }
    if !out_mat.is_null() { *out_mat = mat; }
    1
}

/// Request auto-terrace for a building placement (4x4 footprint).
/// Returns 1 on success, 0 if queue full.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_building_flatten(
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
    engine.request_building_flatten(x, y, z, scale)
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

/// Find a capsule-validated spawn location for the player.
/// Returns 1 if found (out pointers written), 0 if no suitable location.
/// All coordinates are UE world space. Clearance: height=13, radius=3 voxels.
#[no_mangle]
pub unsafe extern "C" fn voxel_find_spawn_location(
    engine: *mut c_void,
    target_x: f32,
    target_y: f32,
    target_z: f32,
    exclude_x: f32,
    exclude_y: f32,
    exclude_z: f32,
    exclude_radius: f32,
    world_scale: f32,
    out_x: *mut f32,
    out_y: *mut f32,
    out_z: *mut f32,
) -> u32 {
    if engine.is_null() || out_x.is_null() || out_y.is_null() || out_z.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    // Player capsule: 13 voxels tall, 3 voxels radius
    match engine.find_spawn_location(
        target_x, target_y, target_z,
        exclude_x, exclude_y, exclude_z,
        exclude_radius, world_scale, 13, 3,
    ) {
        Some((x, y, z)) => {
            *out_x = x;
            *out_y = y;
            *out_z = z;
            1
        }
        None => 0,
    }
}

/// Find a validated spawn location for the chrysalis (quest giver).
/// Returns 1 if found (out pointers written), 0 if no suitable location.
/// Clearance: height=4, radius=2 voxels. Prefers near walls but not clipping.
#[no_mangle]
pub unsafe extern "C" fn voxel_find_chrysalis_location(
    engine: *mut c_void,
    target_x: f32,
    target_y: f32,
    target_z: f32,
    exclude_x: f32,
    exclude_y: f32,
    exclude_z: f32,
    exclude_radius: f32,
    world_scale: f32,
    out_x: *mut f32,
    out_y: *mut f32,
    out_z: *mut f32,
) -> u32 {
    if engine.is_null() || out_x.is_null() || out_y.is_null() || out_z.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    // Chrysalis: 4 voxels tall, 2 voxels radius
    match engine.find_chrysalis_location(
        target_x, target_y, target_z,
        exclude_x, exclude_y, exclude_z,
        exclude_radius, world_scale, 4, 2,
    ) {
        Some((x, y, z)) => {
            *out_x = x;
            *out_y = y;
            *out_z = z;
            1
        }
        None => 0,
    }
}

/// Find spring, chrysalis, and spawn locations all in the same cavern.
/// All coordinates are UE world space. Returns 1 if all three found, 0 otherwise.
/// Geode-filtered: no positions inside crystal geodes.
/// Same-cavern: chrysalis and spawn are flood-fill constrained to the spring's cavern.
#[no_mangle]
pub unsafe extern "C" fn voxel_find_cavern_locations(
    engine: *mut c_void,
    player_x: f32,
    player_y: f32,
    player_z: f32,
    world_scale: f32,
    out_spring_x: *mut f32,
    out_spring_y: *mut f32,
    out_spring_z: *mut f32,
    out_chrysalis_x: *mut f32,
    out_chrysalis_y: *mut f32,
    out_chrysalis_z: *mut f32,
    out_spawn_x: *mut f32,
    out_spawn_y: *mut f32,
    out_spawn_z: *mut f32,
) -> u32 {
    if engine.is_null()
        || out_spring_x.is_null() || out_spring_y.is_null() || out_spring_z.is_null()
        || out_chrysalis_x.is_null() || out_chrysalis_y.is_null() || out_chrysalis_z.is_null()
        || out_spawn_x.is_null() || out_spawn_y.is_null() || out_spawn_z.is_null()
    {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    match engine.find_cavern_locations(player_x, player_y, player_z, world_scale) {
        Some(((sx, sy, sz), (cx, cy, cz), (px, py, pz))) => {
            *out_spring_x = sx;
            *out_spring_y = sy;
            *out_spring_z = sz;
            *out_chrysalis_x = cx;
            *out_chrysalis_y = cy;
            *out_chrysalis_z = cz;
            *out_spawn_x = px;
            *out_spawn_y = py;
            *out_spawn_z = pz;
            1
        }
        None => 0,
    }
}

/// Request priority generation for a single chunk (sent via mine channel for
/// immediate processing). Coords are UE space. Returns 1 on success, 0 if full.
#[no_mangle]
pub unsafe extern "C" fn voxel_request_priority_generate(
    engine: *mut c_void,
    chunk: FfiChunkCoord,
) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.request_priority_generate(chunk.x, chunk.y, chunk.z)
}

/// Query the host rock material at a UE world position based on depth.
/// Returns the material id as u8.
#[no_mangle]
pub unsafe extern "C" fn voxel_query_host_rock_at(
    engine: *mut c_void,
    x: f32,
    y: f32,
    z: f32,
    scale: f32,
) -> u8 {
    if engine.is_null() {
        return 0;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.query_host_rock_at(x, y, z, scale)
}

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

// ── Internal helpers ──

fn convert_mesh_to_ffi_result(
    chunk: (i32, i32, i32),
    mesh: ConvertedMesh,
    generation: u64,
    crystal_data: Vec<FfiCrystalPlacement>,
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
        mined: FfiMinedMaterials { counts: [0; 22] },
        generation,
        fluid_mesh: empty_fluid_mesh_data(),
        crystal_data: convert_crystal_vec_to_ffi(crystal_data),
    }
}

fn convert_crystal_vec_to_ffi(data: Vec<FfiCrystalPlacement>) -> FfiCrystalData {
    if data.is_empty() {
        return FfiCrystalData { placements: std::ptr::null_mut(), count: 0 };
    }
    let count = data.len() as u32;
    let mut boxed = data.into_boxed_slice();
    let ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);
    FfiCrystalData { placements: ptr, count }
}

fn empty_crystal_data() -> FfiCrystalData {
    FfiCrystalData { placements: std::ptr::null_mut(), count: 0 }
}

fn converted_mesh_to_ffi(mesh: ConvertedMesh) -> FfiMeshData {
    let vertex_count = mesh.positions.len() as u32;
    let index_count = mesh.indices.len() as u32;
    let submesh_count = mesh.submeshes.len() as u32;

    let mut positions = mesh.positions.into_boxed_slice();
    let mut normals = mesh.normals.into_boxed_slice();
    let mut material_ids = mesh.material_ids.into_boxed_slice();
    let mut indices = mesh.indices.into_boxed_slice();
    let mut submeshes = mesh.submeshes.into_boxed_slice();

    let positions_ptr = positions.as_mut_ptr();
    let normals_ptr = normals.as_mut_ptr();
    let material_ids_ptr = material_ids.as_mut_ptr();
    let indices_ptr = indices.as_mut_ptr();
    let submeshes_ptr = submeshes.as_mut_ptr();

    std::mem::forget(positions);
    std::mem::forget(normals);
    std::mem::forget(material_ids);
    std::mem::forget(indices);
    std::mem::forget(submeshes);

    FfiMeshData {
        positions: positions_ptr,
        normals: normals_ptr,
        material_ids: material_ids_ptr,
        vertex_count,
        indices: indices_ptr,
        index_count,
        submeshes: submeshes_ptr,
        submesh_count,
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
        submeshes: ptr::null_mut(),
        submesh_count: 0,
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
        uvs: ptr::null_mut(),
        flow_directions: ptr::null_mut(),
    }
}

fn converted_fluid_mesh_to_ffi(mesh: ConvertedFluidMesh) -> FfiFluidMeshData {
    let vertex_count = mesh.positions.len() as u32;
    let index_count = mesh.indices.len() as u32;

    let mut positions = mesh.positions.into_boxed_slice();
    let mut normals = mesh.normals.into_boxed_slice();
    let mut fluid_types = mesh.fluid_types.into_boxed_slice();
    let mut indices = mesh.indices.into_boxed_slice();
    let mut uvs = mesh.uvs.into_boxed_slice();
    let mut flow_directions = mesh.flow_directions.into_boxed_slice();

    let positions_ptr = positions.as_mut_ptr();
    let normals_ptr = normals.as_mut_ptr();
    let fluid_types_ptr = fluid_types.as_mut_ptr();
    let indices_ptr = indices.as_mut_ptr();
    let uvs_ptr = uvs.as_mut_ptr();
    let flow_directions_ptr = flow_directions.as_mut_ptr();

    std::mem::forget(positions);
    std::mem::forget(normals);
    std::mem::forget(fluid_types);
    std::mem::forget(indices);
    std::mem::forget(uvs);
    std::mem::forget(flow_directions);

    FfiFluidMeshData {
        positions: positions_ptr,
        normals: normals_ptr,
        fluid_types: fluid_types_ptr,
        vertex_count,
        indices: indices_ptr,
        index_count,
        uvs: uvs_ptr,
        flow_directions: flow_directions_ptr,
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
            world_scale: 15.0,
            max_edge_length: 5.0,
            cavern_frequency: 0.05,
            cavern_threshold: 0.80,
            detail_octaves: 4,
            detail_persistence: 0.5,
            warp_amplitude: 5.0,
            worms_per_region: 5.0,
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
            fluid_water_spring_threshold: 2.0,
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
            // Mine
            mine_smooth_iterations: 2,
            mine_smooth_strength: 0.3,
            mine_min_triangle_area: 0.01,
            mine_dirty_expand: 2,
            // Bounds
            bounds_size: 0.0,
            // Ore Visual Quality
            ore_domain_warp_strength: 0.0,
            ore_warp_frequency: 0.02,
            ore_edge_falloff: 0.0,
            ore_detail_weight: 0.0,
            // Mesh Smoothing
            mesh_smooth_iterations: 0,
            mesh_smooth_strength: 0.3,
            mesh_boundary_smooth: 0.3,
            mesh_recalc_normals: 1,
            // Pool Config
            pool_enabled: 1,
            pool_placement_freq: 0.08,
            pool_placement_thresh: 0.75,
            pool_chance: 0.3,
            pool_min_area: 4,
            pool_max_radius: 4,
            pool_basin_depth: 2,
            pool_rim_height: 1,
            pool_water_pct: 0.75,
            pool_lava_pct: 0.25,
            pool_empty_pct: 0.0,
            pool_min_air_above: 3,
            pool_max_cave_height: 20,
            pool_min_floor_thickness: 2,
            pool_min_ground_depth: 2,
            pool_max_y_step: 2,
            pool_footprint_y_tolerance: 2,
            // Formation Config
            formation_enabled: 1,
            formation_placement_frequency: 0.15,
            formation_placement_threshold: 0.3,
            formation_stalactite_chance: 0.15,
            formation_stalagmite_chance: 0.12,
            formation_flowstone_chance: 0.1,
            formation_column_chance: 0.08,
            formation_column_max_gap: 8,
            formation_length_min: 2.0,
            formation_length_max: 5.0,
            formation_radius_min: 0.3,
            formation_radius_max: 0.8,
            formation_max_radius: 1.0,
            formation_column_radius_min: 0.4,
            formation_column_radius_max: 1.0,
            formation_flowstone_length_min: 2.0,
            formation_flowstone_length_max: 5.0,
            formation_flowstone_thickness: 0.5,
            formation_min_air_gap: 3,
            formation_min_clearance: 2,
            formation_smoothness: 0.85,
            formation_mega_column_chance: 0.03,
            formation_mega_column_min_gap: 12,
            formation_mega_column_radius_min: 3.0,
            formation_mega_column_radius_max: 5.0,
            formation_mega_column_noise_strength: 0.3,
            formation_mega_column_ring_frequency: 0.8,
            formation_drapery_chance: 0.06,
            formation_drapery_length_min: 3.0,
            formation_drapery_length_max: 8.0,
            formation_drapery_wave_frequency: 1.5,
            formation_drapery_wave_amplitude: 0.4,
            formation_rimstone_chance: 0.04,
            formation_rimstone_dam_height_min: 1.0,
            formation_rimstone_dam_height_max: 1.5,
            formation_rimstone_pool_depth: 1.0,
            formation_rimstone_min_slope: 0.05,
            formation_shield_chance: 0.008,
            formation_shield_radius_min: 1.5,
            formation_shield_radius_max: 3.0,
            formation_shield_max_tilt: 30.0,
            formation_shield_stalactite_chance: 0.5,
            // Cauldron
            formation_cauldron_chance: 0.03,
            formation_cauldron_radius_min: 2.0,
            formation_cauldron_radius_max: 3.0,
            formation_cauldron_depth: 3.0,
            formation_cauldron_lip_height: 0.8,
            formation_cauldron_rim_stalagmite_count_min: 3,
            formation_cauldron_rim_stalagmite_count_max: 5,
            formation_cauldron_rim_stalagmite_scale: 0.5,
            formation_cauldron_floor_noise: 0.3,
            formation_cauldron_water_chance: 0.5,
            formation_cauldron_lava_chance: 0.2,
            // Geological Realism Toggles (all off in tests)
            ore_iron_sedimentary_only: 0,
            ore_iron_depth_fade: 0,
            ore_copper_supergene: 0,
            ore_copper_granite_contact: 0,
            ore_malachite_depth_bias: 0,
            ore_kimberlite_carrot_taper: 0,
            ore_diamond_depth_grade: 0,
            ore_sulfide_gossan_cap: 0,
            ore_sulfide_disseminated: 0,
            ore_pyrite_ore_halo: 0,
            ore_quartz_planar_veins: 0,
            ore_gold_bonanza: 0,
            ore_geode_volcanic_host: 0,
            ore_geode_depth_scaling: 0,
            // Coal
            ore_coal_frequency: 0.03,
            ore_coal_threshold: 0.62,
            ore_coal_depth_min: 10.0,
            ore_coal_depth_max: 80.0,
            ore_coal_sedimentary_host: 0,
            ore_coal_shallow_ceiling: 0,
            ore_coal_depth_enrichment: 0,
            // Ore Detail
            ore_detail_multiplier: 1,
            ore_protrusion: 0.0,
            // Crystal Config
            crystal_enabled: 1,
            // Iron crystals
            crystal_iron_enabled: 1,
            crystal_iron_chance: 0.25,
            crystal_iron_density_threshold: 0.3,
            crystal_iron_scale_min: 0.6,
            crystal_iron_scale_max: 1.4,
            crystal_iron_small_weight: 0.5,
            crystal_iron_medium_weight: 0.35,
            crystal_iron_large_weight: 0.15,
            crystal_iron_normal_alignment: 0.7,
            crystal_iron_cluster_size: 4,
            crystal_iron_cluster_radius: 1.0,
            crystal_iron_surface_offset: 0.1,
            crystal_iron_vein_enabled: 0,
            crystal_iron_vein_frequency: 0.0,
            crystal_iron_vein_thickness: 0.0,
            crystal_iron_vein_octaves: 0,
            crystal_iron_vein_lacunarity: 0.0,
            crystal_iron_vein_warp_strength: 0.0,
            crystal_iron_vein_density: 0.0,
            // Copper crystals
            crystal_copper_enabled: 1,
            crystal_copper_chance: 0.3,
            crystal_copper_density_threshold: 0.3,
            crystal_copper_scale_min: 0.4,
            crystal_copper_scale_max: 1.2,
            crystal_copper_small_weight: 0.5,
            crystal_copper_medium_weight: 0.35,
            crystal_copper_large_weight: 0.15,
            crystal_copper_normal_alignment: 0.7,
            crystal_copper_cluster_size: 3,
            crystal_copper_cluster_radius: 0.8,
            crystal_copper_surface_offset: 0.1,
            crystal_copper_vein_enabled: 0,
            crystal_copper_vein_frequency: 0.0,
            crystal_copper_vein_thickness: 0.0,
            crystal_copper_vein_octaves: 0,
            crystal_copper_vein_lacunarity: 0.0,
            crystal_copper_vein_warp_strength: 0.0,
            crystal_copper_vein_density: 0.0,
            // Malachite crystals
            crystal_malachite_enabled: 1,
            crystal_malachite_chance: 0.35,
            crystal_malachite_density_threshold: 0.25,
            crystal_malachite_scale_min: 0.5,
            crystal_malachite_scale_max: 1.3,
            crystal_malachite_small_weight: 0.5,
            crystal_malachite_medium_weight: 0.35,
            crystal_malachite_large_weight: 0.15,
            crystal_malachite_normal_alignment: 0.7,
            crystal_malachite_cluster_size: 3,
            crystal_malachite_cluster_radius: 0.8,
            crystal_malachite_surface_offset: 0.1,
            crystal_malachite_vein_enabled: 0,
            crystal_malachite_vein_frequency: 0.0,
            crystal_malachite_vein_thickness: 0.0,
            crystal_malachite_vein_octaves: 0,
            crystal_malachite_vein_lacunarity: 0.0,
            crystal_malachite_vein_warp_strength: 0.0,
            crystal_malachite_vein_density: 0.0,
            // Tin crystals
            crystal_tin_enabled: 1,
            crystal_tin_chance: 0.2,
            crystal_tin_density_threshold: 0.3,
            crystal_tin_scale_min: 0.5,
            crystal_tin_scale_max: 1.0,
            crystal_tin_small_weight: 0.5,
            crystal_tin_medium_weight: 0.35,
            crystal_tin_large_weight: 0.15,
            crystal_tin_normal_alignment: 0.7,
            crystal_tin_cluster_size: 3,
            crystal_tin_cluster_radius: 0.8,
            crystal_tin_surface_offset: 0.1,
            crystal_tin_vein_enabled: 0,
            crystal_tin_vein_frequency: 0.0,
            crystal_tin_vein_thickness: 0.0,
            crystal_tin_vein_octaves: 0,
            crystal_tin_vein_lacunarity: 0.0,
            crystal_tin_vein_warp_strength: 0.0,
            crystal_tin_vein_density: 0.0,
            // Gold crystals
            crystal_gold_enabled: 1,
            crystal_gold_chance: 0.4,
            crystal_gold_density_threshold: 0.3,
            crystal_gold_scale_min: 0.3,
            crystal_gold_scale_max: 0.8,
            crystal_gold_small_weight: 0.5,
            crystal_gold_medium_weight: 0.35,
            crystal_gold_large_weight: 0.15,
            crystal_gold_normal_alignment: 0.7,
            crystal_gold_cluster_size: 5,
            crystal_gold_cluster_radius: 0.6,
            crystal_gold_surface_offset: 0.1,
            crystal_gold_vein_enabled: 0,
            crystal_gold_vein_frequency: 0.0,
            crystal_gold_vein_thickness: 0.0,
            crystal_gold_vein_octaves: 0,
            crystal_gold_vein_lacunarity: 0.0,
            crystal_gold_vein_warp_strength: 0.0,
            crystal_gold_vein_density: 0.0,
            // Diamond crystals
            crystal_diamond_enabled: 1,
            crystal_diamond_chance: 0.5,
            crystal_diamond_density_threshold: 0.2,
            crystal_diamond_scale_min: 0.3,
            crystal_diamond_scale_max: 1.0,
            crystal_diamond_small_weight: 0.5,
            crystal_diamond_medium_weight: 0.35,
            crystal_diamond_large_weight: 0.15,
            crystal_diamond_normal_alignment: 0.7,
            crystal_diamond_cluster_size: 3,
            crystal_diamond_cluster_radius: 0.5,
            crystal_diamond_surface_offset: 0.1,
            crystal_diamond_vein_enabled: 0,
            crystal_diamond_vein_frequency: 0.0,
            crystal_diamond_vein_thickness: 0.0,
            crystal_diamond_vein_octaves: 0,
            crystal_diamond_vein_lacunarity: 0.0,
            crystal_diamond_vein_warp_strength: 0.0,
            crystal_diamond_vein_density: 0.0,
            // Kimberlite crystals
            crystal_kimberlite_enabled: 1,
            crystal_kimberlite_chance: 0.15,
            crystal_kimberlite_density_threshold: 0.3,
            crystal_kimberlite_scale_min: 0.8,
            crystal_kimberlite_scale_max: 2.0,
            crystal_kimberlite_small_weight: 0.5,
            crystal_kimberlite_medium_weight: 0.35,
            crystal_kimberlite_large_weight: 0.15,
            crystal_kimberlite_normal_alignment: 0.7,
            crystal_kimberlite_cluster_size: 2,
            crystal_kimberlite_cluster_radius: 1.2,
            crystal_kimberlite_surface_offset: 0.1,
            crystal_kimberlite_vein_enabled: 0,
            crystal_kimberlite_vein_frequency: 0.0,
            crystal_kimberlite_vein_thickness: 0.0,
            crystal_kimberlite_vein_octaves: 0,
            crystal_kimberlite_vein_lacunarity: 0.0,
            crystal_kimberlite_vein_warp_strength: 0.0,
            crystal_kimberlite_vein_density: 0.0,
            // Sulfide crystals
            crystal_sulfide_enabled: 1,
            crystal_sulfide_chance: 0.2,
            crystal_sulfide_density_threshold: 0.3,
            crystal_sulfide_scale_min: 0.5,
            crystal_sulfide_scale_max: 1.2,
            crystal_sulfide_small_weight: 0.5,
            crystal_sulfide_medium_weight: 0.35,
            crystal_sulfide_large_weight: 0.15,
            crystal_sulfide_normal_alignment: 0.7,
            crystal_sulfide_cluster_size: 3,
            crystal_sulfide_cluster_radius: 0.8,
            crystal_sulfide_surface_offset: 0.1,
            crystal_sulfide_vein_enabled: 0,
            crystal_sulfide_vein_frequency: 0.0,
            crystal_sulfide_vein_thickness: 0.0,
            crystal_sulfide_vein_octaves: 0,
            crystal_sulfide_vein_lacunarity: 0.0,
            crystal_sulfide_vein_warp_strength: 0.0,
            crystal_sulfide_vein_density: 0.0,
            // Quartz crystals
            crystal_quartz_enabled: 1,
            crystal_quartz_chance: 0.4,
            crystal_quartz_density_threshold: 0.3,
            crystal_quartz_scale_min: 0.4,
            crystal_quartz_scale_max: 1.5,
            crystal_quartz_small_weight: 0.5,
            crystal_quartz_medium_weight: 0.35,
            crystal_quartz_large_weight: 0.15,
            crystal_quartz_normal_alignment: 0.7,
            crystal_quartz_cluster_size: 4,
            crystal_quartz_cluster_radius: 0.7,
            crystal_quartz_surface_offset: 0.1,
            crystal_quartz_vein_enabled: 0,
            crystal_quartz_vein_frequency: 0.0,
            crystal_quartz_vein_thickness: 0.0,
            crystal_quartz_vein_octaves: 0,
            crystal_quartz_vein_lacunarity: 0.0,
            crystal_quartz_vein_warp_strength: 0.0,
            crystal_quartz_vein_density: 0.0,
            // Pyrite crystals
            crystal_pyrite_enabled: 1,
            crystal_pyrite_chance: 0.3,
            crystal_pyrite_density_threshold: 0.3,
            crystal_pyrite_scale_min: 0.3,
            crystal_pyrite_scale_max: 0.9,
            crystal_pyrite_small_weight: 0.5,
            crystal_pyrite_medium_weight: 0.35,
            crystal_pyrite_large_weight: 0.15,
            crystal_pyrite_normal_alignment: 0.7,
            crystal_pyrite_cluster_size: 5,
            crystal_pyrite_cluster_radius: 0.5,
            crystal_pyrite_surface_offset: 0.1,
            crystal_pyrite_vein_enabled: 0,
            crystal_pyrite_vein_frequency: 0.0,
            crystal_pyrite_vein_thickness: 0.0,
            crystal_pyrite_vein_octaves: 0,
            crystal_pyrite_vein_lacunarity: 0.0,
            crystal_pyrite_vein_warp_strength: 0.0,
            crystal_pyrite_vein_density: 0.0,
            // Amethyst crystals
            crystal_amethyst_enabled: 1,
            crystal_amethyst_chance: 0.45,
            crystal_amethyst_density_threshold: 0.2,
            crystal_amethyst_scale_min: 0.4,
            crystal_amethyst_scale_max: 1.4,
            crystal_amethyst_small_weight: 0.5,
            crystal_amethyst_medium_weight: 0.35,
            crystal_amethyst_large_weight: 0.15,
            crystal_amethyst_normal_alignment: 0.7,
            crystal_amethyst_cluster_size: 4,
            crystal_amethyst_cluster_radius: 0.8,
            crystal_amethyst_surface_offset: 0.1,
            crystal_amethyst_vein_enabled: 0,
            crystal_amethyst_vein_frequency: 0.0,
            crystal_amethyst_vein_thickness: 0.0,
            crystal_amethyst_vein_octaves: 0,
            crystal_amethyst_vein_lacunarity: 0.0,
            crystal_amethyst_vein_warp_strength: 0.0,
            crystal_amethyst_vein_density: 0.0,
            // Coal crystals
            crystal_coal_enabled: 1,
            crystal_coal_chance: 0.1,
            crystal_coal_density_threshold: 0.3,
            crystal_coal_scale_min: 0.3,
            crystal_coal_scale_max: 0.7,
            crystal_coal_small_weight: 0.5,
            crystal_coal_medium_weight: 0.35,
            crystal_coal_large_weight: 0.15,
            crystal_coal_normal_alignment: 0.7,
            crystal_coal_cluster_size: 2,
            crystal_coal_cluster_radius: 0.5,
            crystal_coal_surface_offset: 0.1,
            crystal_coal_vein_enabled: 0,
            crystal_coal_vein_frequency: 0.0,
            crystal_coal_vein_thickness: 0.0,
            crystal_coal_vein_octaves: 0,
            crystal_coal_vein_lacunarity: 0.0,
            crystal_coal_vein_warp_strength: 0.0,
            crystal_coal_vein_density: 0.0,
            // Sleep Config
            sleep_time_budget_ms: 8000,
            sleep_chunk_radius: 1,
            sleep_metamorphism_enabled: 1,
            sleep_minerals_enabled: 1,
            sleep_collapse_enabled: 1,
            sleep_count: 1,
            // Sleep Metamorphism
            sleep_limestone_to_marble_prob: 0.40,
            sleep_limestone_to_marble_depth: -50.0,
            sleep_limestone_to_marble_enabled: 1,
            sleep_sandstone_to_granite_prob: 0.25,
            sleep_sandstone_to_granite_depth: -100.0,
            sleep_sandstone_to_granite_min_neighbors: 4,
            sleep_sandstone_to_granite_enabled: 1,
            sleep_slate_to_marble_prob: 0.60,
            sleep_slate_to_marble_enabled: 1,
            sleep_granite_to_basalt_prob: 0.15,
            sleep_granite_to_basalt_min_air: 2,
            sleep_granite_to_basalt_enabled: 1,
            sleep_iron_to_pyrite_prob: 0.35,
            sleep_iron_to_pyrite_search_radius: 2,
            sleep_iron_to_pyrite_enabled: 1,
            sleep_copper_to_malachite_prob: 0.50,
            sleep_copper_to_malachite_enabled: 1,
            // Sleep Minerals
            sleep_crystal_growth_max: 2,
            sleep_crystal_growth_enabled: 1,
            sleep_crystal_growth_prob: 0.3,
            sleep_malachite_stalactite_max: 1,
            sleep_malachite_stalactite_enabled: 1,
            sleep_malachite_stalactite_prob: 0.2,
            sleep_quartz_extension_prob: 0.10,
            sleep_quartz_extension_max: 1,
            sleep_quartz_extension_enabled: 1,
            sleep_calcite_infill_max: 1,
            sleep_calcite_infill_depth: -30.0,
            sleep_calcite_infill_min_faces: 3,
            sleep_calcite_infill_enabled: 1,
            sleep_calcite_infill_prob: 0.15,
            sleep_pyrite_crust_max: 1,
            sleep_pyrite_crust_min_solid: 2,
            sleep_pyrite_crust_enabled: 1,
            sleep_pyrite_crust_prob: 0.1,
            sleep_growth_density_min: 0.3,
            sleep_growth_density_max: 0.6,
            // Sleep Collapse
            sleep_strut_survival: [0.0, 0.25, 0.30, 0.25, 0.55, 0.70, 0.85, 0.95],
            sleep_stress_multiplier: 1.5,
            sleep_max_cascade_iterations: 8,
            sleep_rubble_fill_ratio: 0.40,
            sleep_min_stress_for_cascade: 0.7,
            sleep_rubble_material_match: 1,
            sleep_support_stress_penalty: 1.0,
            sleep_collapse_sub_enabled: 1,
            // New 4-phase + Groundwater fields
            sleep_groundwater_enabled: 1,
            sleep_groundwater_strength: 0.3,
            sleep_groundwater_depth_scale: 0.02,
            sleep_groundwater_drip_multiplier: 2.0,
            sleep_phase1_enabled: 1,
            sleep_phase2_enabled: 1,
            sleep_phase3_enabled: 1,
            sleep_phase4_enabled: 1,
            sleep_acid_dissolution_prob: 0.60,
            sleep_copper_oxidation_prob: 0.50,
            sleep_basalt_crust_prob: 0.70,
            sleep_aureole_radius: 8,
            sleep_contact_marble_prob: 0.80,
            sleep_water_erosion_prob: 0.05,
            sleep_water_erosion_enabled: 1,
            sleep_vein_deposition_prob: 0.25,
            sleep_vein_max_distance: 16,
            sleep_vein_max_per_source: 12,
            sleep_flowstone_prob: 0.10,
            sleep_enrichment_prob: 0.15,
            sleep_vein_thickening_prob: 0.10,
            sleep_stalactite_growth_prob: 0.10,
            sleep_new_collapse_enabled: 1,
            sleep_new_stress_multiplier: 1.5,
            sleep_new_min_stress_cascade: 0.7,
            sleep_new_rubble_fill_ratio: 0.40,
            // Groundwater power controls
            sleep_gw_erosion_power: 1.0,
            sleep_gw_flowstone_power: 1.0,
            sleep_gw_enrichment_power: 1.0,
            sleep_gw_soft_rock_mult: 1.0,
            sleep_gw_hard_rock_mult: 0.15,
            // Water Table Config
            water_table_enabled: 1,
            water_table_base_y: 170.0,
            water_table_noise_amplitude: 15.0,
            water_table_noise_frequency: 0.008,
            water_table_spring_flow_rate: 0.8,
            water_table_min_porosity: 0.5,
            water_table_drip_noise_frequency: 0.15,
            water_table_drip_noise_threshold: 0.7,
            water_table_drip_level: 0.4,
            water_table_max_springs: 8,
            water_table_max_drips: 12,
            // Pipe Lava Config
            pipe_lava_enabled: 1,
            pipe_lava_activation_depth: -80.0,
            pipe_lava_max_per_chunk: 6,
            pipe_lava_depth_scaling: 0.5,
            // Lava Tube Config
            lava_tube_enabled: 1,
            lava_tube_tubes_per_region: 2.0,
            lava_tube_depth_min: -250.0,
            lava_tube_depth_max: -50.0,
            lava_tube_radius_min: 2.0,
            lava_tube_radius_max: 4.0,
            lava_tube_max_steps: 150,
            lava_tube_step_length: 1.5,
            lava_tube_active_depth: -120.0,
            lava_tube_pipe_connection_radius: 20.0,
            // Hydrothermal Config
            hydrothermal_enabled: 1,
            hydrothermal_radius: 8,
            hydrothermal_max_per_chunk: 4,
            // River Config
            river_enabled: 1,
            river_rivers_per_region: 1.0,
            river_width_min: 3.0,
            river_width_max: 6.0,
            river_height: 2.5,
            river_max_steps: 300,
            river_step_length: 1.5,
            river_layer_restriction: 1,
            river_downslope_bias: 0.02,
            // Artesian Config
            artesian_enabled: 1,
            artesian_aquifer_y_center: -15.0,
            artesian_aquifer_thickness: 3.0,
            artesian_aquifer_noise_freq: 0.01,
            artesian_aquifer_noise_threshold: 0.3,
            artesian_pressure_noise_freq: 0.02,
            artesian_max_per_chunk: 3,
            // Fluid Sources Toggle
            fluid_sources_enabled: 1,
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

            // Poll results until we find the ChunkMesh (fluid meshes may arrive first)
            let mut found_chunk = false;
            let result = &*result_ptr;
            if result.result_type == FfiResultType::ChunkMesh {
                found_chunk = true;
            } else {
                voxel_free_result(result_ptr);
                // Keep polling for the ChunkMesh
                for _ in 0..200 {
                    result_ptr = voxel_poll_result(engine);
                    if !result_ptr.is_null() {
                        let r = &*result_ptr;
                        if r.result_type == FfiResultType::ChunkMesh {
                            found_chunk = true;
                            break;
                        }
                        voxel_free_result(result_ptr);
                    } else {
                        thread::sleep(Duration::from_millis(50));
                    }
                }
            }
            assert!(found_chunk, "Should have received a ChunkMesh result");

            let result = &*result_ptr;
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
