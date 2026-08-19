//! Auto-split from the former monolithic `api.rs`. See `mod.rs`.
#[allow(unused_imports)]
use std::ffi::{c_char, c_void, CString};
#[allow(unused_imports)]
use std::ptr;
#[allow(unused_imports)]
use crate::engine::VoxelEngine;
#[allow(unused_imports)]
use crate::types::*;
use crate::convert::rust_chunk_to_ue;
use super::helpers::*;

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

    // FFI-level skip loop (2026-08-19): arms below that produce "nothing for
    // UE" (should-not-reach intercept fallbacks, empty collapse/stress
    // payloads, the CollapseSlabResult aggregate no-op) must NOT surface as
    // null — UE's ProcessResults reads null as "queue empty" and ends the
    // whole tick's result drain, so one such item made applies land in
    // bursts. Loop to the next UE-facing result instead. [POLL-SLOW] notes
    // >=20ms totals; engine.poll_result() times its own sections.
    let t_poll = std::time::Instant::now();
    let out = loop {
        match engine.poll_result() {
            None => break ptr::null_mut(),
            Some(worker_result) => match worker_result {
            WorkerResult::ChunkMesh {
                chunk,
                mesh,
                generation,
                crystal_data,
                mushroom_data,
                zone_descriptors,
            } => {
                let t_conv = std::time::Instant::now();
                let result = convert_mesh_to_ffi_result(chunk, mesh, generation, crystal_data, mushroom_data, zone_descriptors);
                let conv_ms = t_conv.elapsed().as_secs_f64() * 1000.0;
                if conv_ms >= 20.0 {
                    crate::panic_log::note(&format!("[POLL-SLOW] convert_mesh_to_ffi_result took {:.1}ms", conv_ms));
                }
                break Box::into_raw(Box::new(result));
            }
            WorkerResult::MineBatchMesh { meshes } => {
                // Convert batch to individual results — send first one now, rest get re-queued
                let iter = meshes.into_iter();
                // Re-queue remaining for next polls
                for (chunk, mesh, crystal_data, mushroom_data) in iter {
                    engine.requeue_result(WorkerResult::ChunkMesh {
                        chunk, mesh, generation: 0, crystal_data, mushroom_data, zone_descriptors: Vec::new(),
                    });
                }
                // Signal UE to drain all results this frame
                let result = FfiResult {
                    result_type: FfiResultType::MineResult,
                    chunk: FfiChunkCoord { x: 0, y: 0, z: 0 },
                    mesh: empty_mesh_data(),
                    mined: FfiMinedMaterials { counts: [0; 64] },
                    generation: 0,
                    fluid_mesh: empty_fluid_mesh_data(),
                    crystal_data: empty_crystal_data(),
                    zone_data: empty_zone_data(),
                    mushroom_data: empty_mushroom_data(),
                    slab_fall: FfiSlabFallData::default(),
                };
                break Box::into_raw(Box::new(result));
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
                    zone_data: empty_zone_data(),
                    mushroom_data: empty_mushroom_data(),
                    slab_fall: FfiSlabFallData::default(),
                };
                break Box::into_raw(Box::new(result));
            }
            WorkerResult::FluidMesh { chunk, mesh } => {
                let ue = rust_chunk_to_ue(chunk.0, chunk.1, chunk.2);
                let result = FfiResult {
                    result_type: FfiResultType::FluidMesh,
                    chunk: FfiChunkCoord { x: ue.0, y: ue.1, z: ue.2 },
                    mesh: empty_mesh_data(),
                    mined: FfiMinedMaterials { counts: [0; 64] },
                    generation: 0,
                    fluid_mesh: converted_fluid_mesh_to_ffi(mesh),
                    crystal_data: empty_crystal_data(),
                    zone_data: empty_zone_data(),
                    mushroom_data: empty_mushroom_data(),
                    slab_fall: FfiSlabFallData::default(),
                };
                break Box::into_raw(Box::new(result));
            }
            WorkerResult::Error { chunk, generation } => {
                let ue = rust_chunk_to_ue(chunk.0, chunk.1, chunk.2);
                let result = FfiResult {
                    result_type: FfiResultType::Error,
                    chunk: FfiChunkCoord { x: ue.0, y: ue.1, z: ue.2 },
                    mesh: empty_mesh_data(),
                    mined: FfiMinedMaterials { counts: [0; 64] },
                    generation,
                    fluid_mesh: empty_fluid_mesh_data(),
                    crystal_data: empty_crystal_data(),
                    zone_data: empty_zone_data(),
                    mushroom_data: empty_mushroom_data(),
                    slab_fall: FfiSlabFallData::default(),
                };
                break Box::into_raw(Box::new(result));
            }
            WorkerResult::SolidifyRequest { .. } => {
                // SolidifyRequest is handled engine-internally; skip for now
                continue;
            }
            WorkerResult::LavaQuench { .. } => {
                // LavaQuench is intercepted by engine.poll_result() and
                // re-dispatched to the worker as WorkerRequest::ApplyLavaQuench.
                // If it reaches here, the intercept missed — just drop it.
                continue;
            }
            WorkerResult::CollapseResult { mut events } => {
                if events.is_empty() {
                    continue;
                }

                // Sort by volume descending — biggest slab first
                events.sort_by(|a, b| b.volume.cmp(&a.volume));

                // Requeue additional events so UE receives them on subsequent polls
                if events.len() > 1 {
                    for ev in &events[1..] {
                        engine.requeue_result(WorkerResult::CollapseResult {
                            events: vec![*ev],
                        });
                    }
                }

                // Return the largest collapse event
                let ev = events[0];
                let result = FfiResult {
                    result_type: FfiResultType::CollapseResult,
                    chunk: FfiChunkCoord {
                        x: ev.center_x as i32,
                        y: ev.center_y as i32,
                        z: ev.center_z as i32,
                    },
                    mesh: empty_mesh_data(),
                    mined: FfiMinedMaterials { counts: [0; 64] },
                    generation: ev.volume as u64,
                    fluid_mesh: empty_fluid_mesh_data(),
                    crystal_data: empty_crystal_data(),
                    zone_data: empty_zone_data(),
                    mushroom_data: empty_mushroom_data(),
                    slab_fall: FfiSlabFallData::default(),
                };
                break Box::into_raw(Box::new(result));
            }
            WorkerResult::SleepComplete { .. } => {
                // This should have been intercepted by engine.poll_result().
                // If it somehow reaches here, ignore it.
                continue;
            }
            WorkerResult::ScanComplete { .. } => {
                // This should have been intercepted by engine.poll_result().
                // If it somehow reaches here, ignore it.
                continue;
            }
            WorkerResult::ForceSpawnPoolComplete { .. } => {
                // Intercepted by engine.poll_result(); ignore if it reaches here.
                continue;
            }
            WorkerResult::MorphMeshes { .. } => {
                // Intercepted by engine.poll_result(); ignore if it reaches here.
                continue;
            }
            WorkerResult::StressWarnings { warnings } => {
                if warnings.is_empty() {
                    continue;
                }
                // Pack summary into FfiResult:
                // Chunk = position of highest-stress warning (UE coords, as integers)
                // Generation = (warning_count << 32) | (max_stress * 1000) — packed
                // Mined.counts[0] = dust_count, [1] = creak_count, [2] = shake_count
                let top = &warnings[0]; // Already sorted by stress descending
                let dust_count = warnings.iter().filter(|w| w.warning_type == 1).count() as u32;
                let creak_count = warnings.iter().filter(|w| w.warning_type == 2).count() as u32;
                let shake_count = warnings.iter().filter(|w| w.warning_type == 3).count() as u32;
                let mut mined = FfiMinedMaterials { counts: [0; 64] };
                mined.counts[0] = dust_count;
                mined.counts[1] = creak_count;
                mined.counts[2] = shake_count;
                mined.counts[3] = warnings.len() as u32;

                let result = FfiResult {
                    result_type: FfiResultType::StressWarnings,
                    chunk: FfiChunkCoord {
                        x: top.world_x as i32,
                        y: top.world_y as i32,
                        z: top.world_z as i32,
                    },
                    mesh: empty_mesh_data(),
                    mined,
                    generation: (top.stress * 1000.0) as u64,
                    fluid_mesh: empty_fluid_mesh_data(),
                    crystal_data: empty_crystal_data(),
                    zone_data: empty_zone_data(),
                    mushroom_data: empty_mushroom_data(),
                    slab_fall: FfiSlabFallData::default(),
                };
                break Box::into_raw(Box::new(result));
            }
            WorkerResult::CollapseSlabResult { .. } => {
                // Aggregate variant — individual slabs are emitted via SlabFall
                // (one result per slab fragment). Keep this as a no-op for now.
                continue;
            }
            WorkerResult::SlabFall { mesh, fall_data } => {
                // Individual falling-slab visual — real DC mesh + fall metadata.
                // Populate the mesh field (UE builds a ProcMesh from it) and
                // the slab_fall metadata block.
                let result = FfiResult {
                    result_type: FfiResultType::CollapseSlabResult,
                    chunk: FfiChunkCoord {
                        x: fall_data.land_x as i32,
                        y: fall_data.land_y as i32,
                        z: fall_data.land_z as i32,
                    },
                    mesh: converted_mesh_to_ffi(mesh),
                    mined: FfiMinedMaterials { counts: [0; 64] },
                    generation: fall_data.volume as u64,
                    fluid_mesh: empty_fluid_mesh_data(),
                    crystal_data: empty_crystal_data(),
                    zone_data: empty_zone_data(),
                    mushroom_data: empty_mushroom_data(),
                    slab_fall: fall_data,
                };
                break Box::into_raw(Box::new(result));
            }
            WorkerResult::CollapseWarning { center_ue, bounds_extent_ue, severity, eta_ms, volume } => {
                // Localized pre-collapse warning. Drives Acts 1-2 of the
                // cinematic — UE spawns a warning FX actor at the centre with
                // the given bounds and ETA.
                let mut fall = FfiSlabFallData::default();
                fall.spawn_x = center_ue.0;
                fall.spawn_y = center_ue.1;
                fall.spawn_z = center_ue.2;
                fall.bounds_extent_x = bounds_extent_ue.0;
                fall.bounds_extent_y = bounds_extent_ue.1;
                fall.bounds_extent_z = bounds_extent_ue.2;
                fall.volume = volume;
                fall.warning_severity = severity;
                fall.warning_eta_ms = eta_ms;
                let result = FfiResult {
                    result_type: FfiResultType::CollapseWarning,
                    chunk: FfiChunkCoord {
                        x: center_ue.0 as i32,
                        y: center_ue.1 as i32,
                        z: center_ue.2 as i32,
                    },
                    mesh: empty_mesh_data(),
                    mined: FfiMinedMaterials { counts: [0; 64] },
                    generation: volume as u64,
                    fluid_mesh: empty_fluid_mesh_data(),
                    crystal_data: empty_crystal_data(),
                    zone_data: empty_zone_data(),
                    mushroom_data: empty_mushroom_data(),
                    slab_fall: fall,
                };
                break Box::into_raw(Box::new(result));
            }
            // PathComputed is intercepted inside engine.poll_result() and
            // stashed into path_results — it never reaches this match in
            // practice. Listed here only for exhaustiveness.
            WorkerResult::PathComputed { .. } => continue,
            // StrutsBroken is intercepted in engine.poll_result() and stashed
            // for UE to drain via voxel_take_struts_broken — reaching this
            // arm means the intercept missed; safe to drop.
            WorkerResult::StrutsBroken { .. } => continue,
            WorkerResult::PilePreviewTier { mesh, fall_data } => {
                // One tier of the pre-commit pile preview. fall_data carries
                // tier_index in pile_tier_index, spawn_x/y/z is the pile
                // anchor used by UE to correlate 4 tiers into one debris actor.
                let result = FfiResult {
                    result_type: FfiResultType::CollapsePilePreviewTier,
                    chunk: FfiChunkCoord {
                        x: fall_data.spawn_x as i32,
                        y: fall_data.spawn_y as i32,
                        z: fall_data.spawn_z as i32,
                    },
                    mesh: converted_mesh_to_ffi(mesh),
                    mined: FfiMinedMaterials { counts: [0; 64] },
                    generation: fall_data.volume as u64,
                    fluid_mesh: empty_fluid_mesh_data(),
                    crystal_data: empty_crystal_data(),
                    zone_data: empty_zone_data(),
                    mushroom_data: empty_mushroom_data(),
                    slab_fall: fall_data,
                };
                break Box::into_raw(Box::new(result));
            }
            },
        }
    };
    let poll_ms = t_poll.elapsed().as_secs_f64() * 1000.0;
    if poll_ms >= 20.0 {
        crate::panic_log::note(&format!(
            "[POLL-SLOW] voxel_poll_result took {:.1}ms total (engine skip-loop + conversion)",
            poll_ms));
    }
    out
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

    // Free zone data if present
    let zones = &result.zone_data;
    if zones.count > 0 && !zones.descriptors.is_null() {
        drop(Vec::from_raw_parts(
            zones.descriptors,
            zones.count as usize,
            zones.count as usize,
        ));
    }

    // Free mushroom data if present
    let mush = &result.mushroom_data;
    if mush.count > 0 && !mush.instances.is_null() {
        drop(Vec::from_raw_parts(
            mush.instances,
            mush.count as usize,
            mush.count as usize,
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
            workers_alive: 0,
            panics_observed: 0,
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

/// Hot-reload fluid config at runtime.
/// flow_solid_threshold and fractional_capacity kept in C signature for UE ABI
/// compatibility but are ignored — binary cell classification is always used.
#[no_mangle]
pub unsafe extern "C" fn voxel_update_fluid_config(
    engine: *mut c_void,
    _flow_solid_threshold: u8,
    _fractional_capacity: u8,
    source_grace_ticks: u16,
) {
    if engine.is_null() {
        return;
    }
    let engine = &*(engine as *const VoxelEngine);
    engine.update_fluid_config(source_grace_ticks);
}

