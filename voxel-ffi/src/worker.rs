use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};
use dashmap::DashMap;
use voxel_core::dual_contouring::mesh_gen::generate_mesh;
use voxel_core::dual_contouring::solve::solve_dc_vertices;
use voxel_core::stress::SupportType;
use voxel_fluid::FluidEvent;
use voxel_gen::config::{GenerationConfig, StressConfig};
use voxel_gen::hermite_extract::extract_hermite_data;
use voxel_gen::region_gen::{
    self, generate_region_densities, region_chunks, region_key, ChunkSeamData,
};

use crate::convert::{convert_mesh_to_ue_scaled, from_ue_normal, from_ue_world_pos};
use crate::profiler::{ChunkTimings, StreamingProfiler};
use crate::store::{extract_solid_mask, ChunkStore};
use crate::types::{FfiCollapseEvent, WorkerRequest, WorkerResult};

/// Worker thread main loop. Each worker pulls from shared channels.
pub fn worker_loop(
    shutdown: Arc<AtomicBool>,
    generate_rx: Receiver<WorkerRequest>,
    mine_rx: Receiver<WorkerRequest>,
    result_tx: Sender<WorkerResult>,
    store: Arc<RwLock<ChunkStore>>,
    config: Arc<RwLock<GenerationConfig>>,
    stress_config: Arc<RwLock<StressConfig>>,
    generation_counters: Arc<DashMap<(i32, i32, i32), AtomicU64>>,
    world_scale: f32,
    fluid_event_tx: Sender<FluidEvent>,
    profiler: Arc<StreamingProfiler>,
    worker_id: usize,
) {
    while !shutdown.load(Ordering::Relaxed) {
        // Priority 1: mine requests (non-blocking)
        if let Ok(req) = mine_rx.try_recv() {
            handle_request(
                req, &result_tx, &store, &config, &stress_config, &generation_counters,
                world_scale, &fluid_event_tx, &profiler, worker_id, &generate_rx,
            );
            continue;
        }

        // Priority 2: generate requests (blocking with timeout)
        let idle_start = Instant::now();
        match generate_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(req) => {
                profiler.record_worker_idle(worker_id, idle_start.elapsed());
                handle_request(
                    req, &result_tx, &store, &config, &stress_config, &generation_counters,
                    world_scale, &fluid_event_tx, &profiler, worker_id, &generate_rx,
                );
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                profiler.record_worker_idle(worker_id, idle_start.elapsed());
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }
}

fn handle_request(
    req: WorkerRequest,
    result_tx: &Sender<WorkerResult>,
    store: &Arc<RwLock<ChunkStore>>,
    config: &Arc<RwLock<GenerationConfig>>,
    stress_config: &Arc<RwLock<StressConfig>>,
    generation_counters: &Arc<DashMap<(i32, i32, i32), AtomicU64>>,
    world_scale: f32,
    fluid_event_tx: &Sender<FluidEvent>,
    profiler: &Arc<StreamingProfiler>,
    worker_id: usize,
    generate_rx: &Receiver<WorkerRequest>,
) {
    match req {
        WorkerRequest::PriorityGenerate { chunk, generation } |
        WorkerRequest::Generate { chunk, generation } => {
            let chunk_start = Instant::now();
            let profiling = profiler.is_enabled();

            // Check if this generation is still current (stale detection)
            if let Some(counter) = generation_counters.get(&chunk) {
                if counter.load(Ordering::Relaxed) != generation {
                    profiler.record_stale_skip(worker_id);
                    return; // Stale request, skip
                }
            }

            let cfg = config.read().unwrap().clone();
            let rk = region_key(chunk.0, chunk.1, chunk.2, cfg.region_size);

            // Timing accumulators
            let mut t_region_density = Duration::ZERO;
            let mut t_hermite = Duration::ZERO;
            let mut t_store_read_wait = Duration::ZERO;
            let mut t_store_write_wait = Duration::ZERO;
            let mut t_dc_solve = Duration::ZERO;
            let mut t_mesh_gen = Duration::ZERO;
            let mut was_slow_path = false;

            // Fast path: region generated AND this chunk has data → mesh under one read lock
            let mesh_result = {
                let t0 = Instant::now();
                let s = store.read().unwrap();
                if profiling { t_store_read_wait += t0.elapsed(); }

                if s.is_region_generated(&rk) && s.has_density(&chunk) {
                    let density = s.density_fields.get(&chunk).unwrap();
                    let hermite = s.hermite_data.get(&chunk).unwrap();
                    let cell_size = density.size - 1;

                    let t1 = Instant::now();
                    let dc_verts = solve_dc_vertices(hermite, cell_size);
                    if profiling { t_dc_solve += t1.elapsed(); }

                    let t2 = Instant::now();
                    let m = generate_mesh(hermite, &dc_verts, cell_size, cfg.max_edge_length, cfg.mine.min_triangle_area);
                    if profiling { t_mesh_gen += t2.elapsed(); }

                    let b_edges = region_gen::extract_boundary_edges(hermite, cfg.chunk_size);
                    Some((m, dc_verts, b_edges))
                } else {
                    None
                }
            };

            let (mesh, dc_vertices, boundary_edges) = if let Some(result) = mesh_result {
                result
            } else {
                // Slow path: (re)generate region densities
                was_slow_path = true;

                let t0 = Instant::now();
                let coords = region_chunks(rk, cfg.region_size);
                let (densities, _pools) = generate_region_densities(&coords, &cfg);
                if profiling { t_region_density += t0.elapsed(); }

                    // Pre-extract hermite data BEFORE acquiring write lock (expensive part)
                let t2 = Instant::now();
                let keyed_data: Vec<_> = densities
                    .into_iter()
                    .map(|(key, density)| {
                        let hermite = extract_hermite_data(&density);
                        (key, density, hermite)
                    })
                    .collect();
                if profiling { t_hermite += t2.elapsed(); }

                // Write lock held only for fast inserts
                {
                    let t1 = Instant::now();
                    let mut s = store.write().unwrap();
                    if profiling { t_store_write_wait += t1.elapsed(); }

                    if !s.is_region_generated(&rk) || !s.has_density(&chunk) {
                        for (key, density, hermite) in keyed_data {
                            if !s.has_density(&key) {
                                s.insert(key, density, hermite);
                            }
                        }
                        s.mark_region_generated(rk);
                    }
                }

                // Mesh under fresh read lock
                let t3 = Instant::now();
                let s = store.read().unwrap();
                if profiling { t_store_read_wait += t3.elapsed(); }

                let density = match s.density_fields.get(&chunk) {
                    Some(d) => d,
                    None => {
                        profiler.record_error();
                        let _ = result_tx.send(WorkerResult::Error { chunk, generation });
                        return;
                    }
                };
                let hermite = match s.hermite_data.get(&chunk) {
                    Some(h) => h,
                    None => {
                        profiler.record_error();
                        let _ = result_tx.send(WorkerResult::Error { chunk, generation });
                        return;
                    }
                };
                let cell_size = density.size - 1;

                let t4 = Instant::now();
                let dc_verts = solve_dc_vertices(hermite, cell_size);
                if profiling { t_dc_solve += t4.elapsed(); }

                let t5 = Instant::now();
                let m = generate_mesh(hermite, &dc_verts, cell_size, cfg.max_edge_length, cfg.mine.min_triangle_area);
                if profiling { t_mesh_gen += t5.elapsed(); }

                let b_edges = region_gen::extract_boundary_edges(hermite, cfg.chunk_size);
                (m, dc_verts, b_edges)
            };

            // Cache seam data and base mesh for this chunk
            {
                let mut s = store.write().unwrap();
                s.add_seam_data(
                    chunk,
                    ChunkSeamData {
                        dc_vertices,
                        world_origin: glam::Vec3::ZERO,
                        boundary_edges,
                    },
                );
                s.base_meshes.insert(chunk, mesh.clone());
            }

            // Extract solid mask and send to fluid thread
            {
                let s = store.read().unwrap();
                if let Some(density) = s.density_fields.get(&chunk) {
                    let mask = extract_solid_mask(density, cfg.chunk_size);
                    let _ = fluid_event_tx.send(FluidEvent::SolidMaskUpdate {
                        chunk,
                        mask,
                    });
                    let _ = fluid_event_tx.send(FluidEvent::PlaceSources { chunk });
                }
            }

            // Convert to UE coordinates and send initial result (no seams yet)
            let t_coord_start = Instant::now();
            let mut converted = convert_mesh_to_ue_scaled(&mesh, cfg.voxel_scale(), world_scale);
            crate::convert::bucket_mesh_by_material(&mut converted);
            let t_coord_transform = if profiling { t_coord_start.elapsed() } else { Duration::ZERO };

            // Capture mesh complexity before sending
            let vertex_count = converted.positions.len() as u32;
            let triangle_count = (converted.indices.len() / 3) as u32;
            // Count unique material sections
            let section_count = {
                let mut mats: Vec<u8> = converted.material_ids.iter().copied().collect();
                mats.sort_unstable();
                mats.dedup();
                mats.len() as u32
            };
            let mesh_bytes = (converted.positions.len() * std::mem::size_of::<crate::types::FfiVec3>()
                + converted.normals.len() * std::mem::size_of::<crate::types::FfiVec3>()
                + converted.material_ids.len()
                + converted.indices.len() * std::mem::size_of::<u32>()) as u32;

            let t_send_start = Instant::now();
            let _ = result_tx.send(WorkerResult::ChunkMesh {
                chunk,
                mesh: converted,
                generation,
            });
            let t_send_block = if profiling { t_send_start.elapsed() } else { Duration::ZERO };

            // Try to generate seams for this chunk and its neighbors
            let seam_timings = incremental_seam_pass(chunk, &cfg, store, result_tx, world_scale);
            let t_seam_pass = if profiling { seam_timings.total } else { Duration::ZERO };

            // Record profiling data
            if profiling {
                let gen_queue_len = generate_rx.len() as u32;
                let result_queue_len = 0u32; // Sender doesn't expose queue length

                let timings = ChunkTimings {
                    region_density: t_region_density,
                    hermite: t_hermite,
                    dc_solve: t_dc_solve,
                    mesh_gen: t_mesh_gen,
                    seam_pass: t_seam_pass,
                    coord_transform: t_coord_transform,
                    store_read_wait: t_store_read_wait,
                    store_write_wait: t_store_write_wait,
                    total: chunk_start.elapsed(),
                    was_slow_path,
                    vertex_count,
                    triangle_count,
                    section_count,
                    mesh_bytes,
                    seam_quad_gen: seam_timings.quad_gen,
                    seam_mesh_retrieve: seam_timings.mesh_retrieve,
                    seam_convert: seam_timings.convert,
                    seam_candidates_tried: seam_timings.candidates_tried,
                    seam_candidates_sent: seam_timings.candidates_sent,
                    send_block: t_send_block,
                    coarse_skip: vertex_count == 0 && triangle_count == 0,
                };
                profiler.record_chunk_with_coord(
                    worker_id, chunk, timings, gen_queue_len, result_queue_len,
                );
            }
        }
        WorkerRequest::Flatten { base_x, base_y, base_z, host_material } => {
            let cfg = config.read().unwrap().clone();
            let mat = voxel_core::material::Material::from_u8(host_material);
            let mut s = store.write().unwrap();
            let meshes = s.flatten_terrace(glam::IVec3::new(base_x, base_y, base_z), mat, &cfg, world_scale);
            drop(s);
            for (key, mesh) in meshes {
                let _ = result_tx.send(WorkerResult::ChunkMesh { chunk: key, mesh, generation: 0 });
            }
        }
        WorkerRequest::Mine { request } => {
            let cfg = config.read().unwrap().clone();

            // Convert UE world position to Rust coordinates
            let center = from_ue_world_pos(
                request.world_x, request.world_y, request.world_z, world_scale,
            );
            let radius = request.radius / world_scale;

            let mut s = store.write().unwrap();
            let (meshes, mined) = if request.mode == 0 {
                s.mine_sphere(center, radius, &cfg, world_scale)
            } else {
                let normal = from_ue_normal(
                    request.normal_x, request.normal_y, request.normal_z,
                );
                s.mine_peel(center, normal, radius, &cfg, world_scale)
            };
            drop(s);

            // Collect dirty chunk keys for seam regeneration
            let dirty_keys: Vec<(i32, i32, i32)> = meshes.iter().map(|(k, _)| *k).collect();

            // Send each dirty chunk mesh individually so UE can update existing actors
            for (key, mesh) in meshes {
                let _ = result_tx.send(WorkerResult::ChunkMesh {
                    chunk: key,
                    mesh,
                    generation: 0,
                });
            }

            // Send mined material counts separately
            let _ = result_tx.send(WorkerResult::MinedMaterials { mined });

            // Post-mine stress update: recalc stress + cascade collapses
            let stress_cfg = stress_config.read().unwrap().clone();
            {
                let mut s = store.write().unwrap();
                let (collapse_events, collapse_dirty) = s.post_mine_stress_update(
                    center, &stress_cfg, cfg.chunk_size,
                );

                // If collapses happened, remesh affected chunks and send collapse events
                if !collapse_events.is_empty() {
                    let ffi_events: Vec<FfiCollapseEvent> = collapse_events.iter().map(|e| {
                        // Convert center from Rust coords to UE coords for the FFI
                        FfiCollapseEvent {
                            center_x: e.center.0 * world_scale,
                            center_y: -e.center.2 * world_scale,
                            center_z: e.center.1 * world_scale,
                            volume: e.volume,
                        }
                    }).collect();

                    // Remesh collapse-dirty chunks
                    let collapse_bounds: Vec<_> = collapse_dirty.iter().map(|&key| {
                        (key, 0usize, 0usize, 0usize, cfg.chunk_size, cfg.chunk_size, cfg.chunk_size)
                    }).collect();
                    let collapse_meshes = s.remesh_dirty(&collapse_bounds, &cfg, world_scale);

                    let _ = result_tx.send(WorkerResult::CollapseResult {
                        events: ffi_events,
                        meshes: collapse_meshes,
                    });
                }
            }

            // Send terrain modifications to fluid thread
            {
                let s = store.read().unwrap();
                for &key in &dirty_keys {
                    if let Some(density) = s.density_fields.get(&key) {
                        let mask = extract_solid_mask(density, cfg.chunk_size);
                        let _ = fluid_event_tx.send(FluidEvent::TerrainModified {
                            chunk: key,
                            mask,
                        });
                    }
                }
            }

            // Regenerate seams for dirty chunks and their neighbors
            for key in dirty_keys {
                let _ = incremental_seam_pass(key, &cfg, store, result_tx, world_scale);
            }
        }
        WorkerRequest::Unload { chunk } => {
            let mut s = store.write().unwrap();
            s.unload(chunk);
            generation_counters.remove(&chunk);
            let _ = fluid_event_tx.send(FluidEvent::ChunkUnloaded { chunk });
        }
        WorkerRequest::PlaceSupport { world_x, world_y, world_z, support_type } => {
            let cfg = config.read().unwrap().clone();
            let stress_cfg = stress_config.read().unwrap().clone();
            let st = SupportType::from_u8(support_type);

            let mut s = store.write().unwrap();
            let (success, collapse_events, dirty_bounds) = s.place_support(
                (world_x, world_y, world_z), st, &stress_cfg, cfg.chunk_size,
            );

            // Remesh affected chunks
            let remesh_bounds: Vec<_> = dirty_bounds.iter().map(|&(key, (min_x, min_y, min_z, max_x, max_y, max_z))| {
                (key, min_x, min_y, min_z, max_x, max_y, max_z)
            }).collect();
            let meshes = s.remesh_dirty(&remesh_bounds, &cfg, world_scale);
            drop(s);

            // Send collapse events if any
            if !collapse_events.is_empty() {
                let ffi_events: Vec<FfiCollapseEvent> = collapse_events.iter().map(|e| {
                    FfiCollapseEvent {
                        center_x: e.center.0 * world_scale,
                        center_y: -e.center.2 * world_scale,
                        center_z: e.center.1 * world_scale,
                        volume: e.volume,
                    }
                }).collect();
                let _ = result_tx.send(WorkerResult::CollapseResult {
                    events: ffi_events,
                    meshes: Vec::new(),
                });
            }

            // Send support result with remeshed chunks
            let mesh_pairs: Vec<_> = meshes.into_iter().collect();
            let _ = result_tx.send(WorkerResult::SupportResult {
                success,
                meshes: mesh_pairs,
            });
        }
        WorkerRequest::RemoveSupport { world_x, world_y, world_z } => {
            let cfg = config.read().unwrap().clone();
            let stress_cfg = stress_config.read().unwrap().clone();

            let mut s = store.write().unwrap();
            let (removed, collapse_events, dirty_bounds) = s.remove_support(
                (world_x, world_y, world_z), &stress_cfg, cfg.chunk_size,
            );

            // Remesh affected chunks
            let remesh_bounds: Vec<_> = dirty_bounds.iter().map(|&(key, (min_x, min_y, min_z, max_x, max_y, max_z))| {
                (key, min_x, min_y, min_z, max_x, max_y, max_z)
            }).collect();
            let meshes = s.remesh_dirty(&remesh_bounds, &cfg, world_scale);
            drop(s);

            // Send collapse events if any
            if !collapse_events.is_empty() {
                let ffi_events: Vec<FfiCollapseEvent> = collapse_events.iter().map(|e| {
                    FfiCollapseEvent {
                        center_x: e.center.0 * world_scale,
                        center_y: -e.center.2 * world_scale,
                        center_z: e.center.1 * world_scale,
                        volume: e.volume,
                    }
                }).collect();
                let _ = result_tx.send(WorkerResult::CollapseResult {
                    events: ffi_events,
                    meshes: Vec::new(),
                });
            }

            // Send support result
            let mesh_pairs: Vec<_> = meshes.into_iter().collect();
            let _ = result_tx.send(WorkerResult::SupportResult {
                success: removed.is_some(),
                meshes: mesh_pairs,
            });
        }
        WorkerRequest::Sleep { player_chunk, sleep_count } => {
            let cfg = config.read().unwrap().clone();
            let sleep_config = voxel_sleep::SleepConfig::default();

            let mut s = store.write().unwrap();

            // Use helper to get three simultaneous &mut borrows (borrow checker
            // cannot split borrows through method calls on the same struct).
            let (density_fields, stress_fields, support_fields) = s.sleep_fields_mut();

            // Execute the sleep cycle on the mutable store fields
            let sleep_result = voxel_sleep::execute_sleep(
                &sleep_config,
                density_fields,
                stress_fields,
                support_fields,
                player_chunk,
                sleep_count,
                None, // No progress channel for now
            );

            // Remesh all dirty chunks (full chunk bounds)
            let dirty_bounds: Vec<_> = sleep_result.dirty_chunks.iter().map(|&key| {
                (key, 0usize, 0usize, 0usize, cfg.chunk_size, cfg.chunk_size, cfg.chunk_size)
            }).collect();
            let meshes = s.remesh_dirty(&dirty_bounds, &cfg, world_scale);
            drop(s);

            // Send each dirty chunk mesh through the normal ChunkMesh pipeline
            // so UE auto-remeshes existing chunk actors
            for (chunk, mesh) in meshes {
                let _ = result_tx.send(WorkerResult::ChunkMesh {
                    chunk,
                    mesh,
                    generation: 0, // Sleep remesh
                });
            }

            // Send collapse events through the normal CollapseResult pipeline
            if !sleep_result.collapse_events.is_empty() {
                let ffi_events: Vec<FfiCollapseEvent> = sleep_result.collapse_events.iter().map(|e| {
                    FfiCollapseEvent {
                        center_x: e.center.0 * world_scale,
                        center_y: -e.center.2 * world_scale,  // Rust Y-up -> UE Z-up
                        center_z: e.center.1 * world_scale,
                        volume: e.volume,
                    }
                }).collect();
                let _ = result_tx.send(WorkerResult::CollapseResult {
                    events: ffi_events,
                    meshes: Vec::new(), // Meshes already sent above
                });
            }

            // Send sleep completion stats (intercepted by engine.poll_result)
            let _ = result_tx.send(WorkerResult::SleepComplete {
                chunks_changed: sleep_result.chunks_changed,
                voxels_metamorphosed: sleep_result.voxels_metamorphosed,
                minerals_grown: sleep_result.minerals_grown,
                supports_degraded: sleep_result.supports_degraded,
                collapses_triggered: sleep_result.collapses_triggered,
            });

            // Regenerate seams for dirty chunks
            for &key in &sleep_result.dirty_chunks {
                let _ = incremental_seam_pass(key, &cfg, store, result_tx, world_scale);
            }
        }
    }
}

/// Timing breakdown from the seam pass.
struct SeamPassTimings {
    pub total: Duration,
    pub quad_gen: Duration,
    pub mesh_retrieve: Duration,
    pub convert: Duration,
    pub candidates_tried: u32,
    pub candidates_sent: u32,
}

/// After meshing chunk C, attempt seam generation for C and its 6 face-adjacent
/// neighbors. Any chunk that produces non-empty seam quads gets combined with
/// the cached base mesh and re-sent.
///
/// generate_chunk_seam_quads gracefully handles missing neighbors — it simply
/// skips quads where neighbor data isn't available yet. So calling it repeatedly
/// as neighbors arrive is safe and produces progressively more complete seams.
fn incremental_seam_pass(
    chunk: (i32, i32, i32),
    cfg: &GenerationConfig,
    store: &Arc<RwLock<ChunkStore>>,
    result_tx: &Sender<WorkerResult>,
    world_scale: f32,
) -> SeamPassTimings {
    let pass_start = Instant::now();
    let mut t_quad_gen = Duration::ZERO;
    let mut t_mesh_retrieve = Duration::ZERO;
    let mut t_convert = Duration::ZERO;
    let mut candidates_tried: u32 = 0;
    let mut candidates_sent: u32 = 0;

    let candidates = [
        chunk,
        (chunk.0 - 1, chunk.1, chunk.2),
        (chunk.0 + 1, chunk.1, chunk.2),
        (chunk.0, chunk.1 - 1, chunk.2),
        (chunk.0, chunk.1 + 1, chunk.2),
        (chunk.0, chunk.1, chunk.2 - 1),
        (chunk.0, chunk.1, chunk.2 + 1),
    ];

    // Batch: acquire ONE read lock, generate all seam quads + clone base meshes
    let mut to_send: Vec<((i32, i32, i32), voxel_core::mesh::Mesh)> = Vec::new();
    {
        let t0 = Instant::now();
        let s = store.read().unwrap();
        let lock_wait = t0.elapsed();
        t_mesh_retrieve += lock_wait; // attribute lock wait to mesh_retrieve

        for &target in &candidates {
            if !s.chunk_seam_data.contains_key(&target) {
                continue;
            }

            let tq = Instant::now();
            let seam_mesh = region_gen::generate_chunk_seam_quads(target, &s.chunk_seam_data, cfg.chunk_size);
            t_quad_gen += tq.elapsed();
            candidates_tried += 1;

            if seam_mesh.triangles.is_empty() {
                continue;
            }

            let tm = Instant::now();
            let base = match s.base_meshes.get(&target) {
                Some(m) => m.clone(),
                None => continue,
            };
            let mut mesh = base;
            mesh.append(seam_mesh);
            t_mesh_retrieve += tm.elapsed();

            to_send.push((target, mesh));
        }
    } // read lock released

    // Convert and send outside the lock (non-blocking sends)
    for (target, combined) in to_send {
        let t2 = Instant::now();
        let mut converted = convert_mesh_to_ue_scaled(&combined, cfg.voxel_scale(), world_scale);
        crate::convert::bucket_mesh_by_material(&mut converted);
        t_convert += t2.elapsed();

        let _ = result_tx.try_send(WorkerResult::ChunkMesh {
            chunk: target,
            mesh: converted,
            generation: 0,
        });
        candidates_sent += 1;
    }

    SeamPassTimings {
        total: pass_start.elapsed(),
        quad_gen: t_quad_gen,
        mesh_retrieve: t_mesh_retrieve,
        convert: t_convert,
        candidates_tried,
        candidates_sent,
    }
}
