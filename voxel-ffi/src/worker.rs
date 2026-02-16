use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender};
use dashmap::DashMap;
use voxel_core::dual_contouring::mesh_gen::generate_mesh;
use voxel_core::dual_contouring::solve::solve_dc_vertices;
use voxel_gen::config::GenerationConfig;
use voxel_gen::hermite_extract::extract_hermite_data;
use voxel_gen::region_gen::{
    self, generate_region_densities, region_chunks, region_key, ChunkSeamData,
};

use crate::convert::{convert_mesh_to_ue, from_ue_normal, from_ue_world_pos};
use crate::store::ChunkStore;
use crate::types::{WorkerRequest, WorkerResult};

/// Worker thread main loop. Each worker pulls from shared channels.
pub fn worker_loop(
    shutdown: Arc<AtomicBool>,
    generate_rx: Receiver<WorkerRequest>,
    mine_rx: Receiver<WorkerRequest>,
    result_tx: Sender<WorkerResult>,
    store: Arc<RwLock<ChunkStore>>,
    config: Arc<RwLock<GenerationConfig>>,
    generation_counters: Arc<DashMap<(i32, i32, i32), AtomicU64>>,
    world_scale: f32,
) {
    while !shutdown.load(Ordering::Relaxed) {
        // Priority 1: mine requests (non-blocking)
        if let Ok(req) = mine_rx.try_recv() {
            handle_request(
                req, &result_tx, &store, &config, &generation_counters, world_scale,
            );
            continue;
        }

        // Priority 2: generate requests (blocking with timeout)
        match generate_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(req) => {
                handle_request(
                    req, &result_tx, &store, &config, &generation_counters, world_scale,
                );
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }
}

fn handle_request(
    req: WorkerRequest,
    result_tx: &Sender<WorkerResult>,
    store: &Arc<RwLock<ChunkStore>>,
    config: &Arc<RwLock<GenerationConfig>>,
    generation_counters: &Arc<DashMap<(i32, i32, i32), AtomicU64>>,
    world_scale: f32,
) {
    match req {
        WorkerRequest::Generate { chunk, generation } => {
            // Check if this generation is still current (stale detection)
            if let Some(counter) = generation_counters.get(&chunk) {
                if counter.load(Ordering::Relaxed) != generation {
                    return; // Stale request, skip
                }
            }

            let cfg = config.read().unwrap().clone();
            let rk = region_key(chunk.0, chunk.1, chunk.2, cfg.region_size);

            // Check if this chunk's density is already cached (from a region batch)
            let needs_region = {
                let s = store.read().unwrap();
                !s.is_region_generated(&rk)
            };

            if needs_region {
                // Generate ALL densities for this region with global worms
                let coords = region_chunks(rk, cfg.region_size);
                let densities = generate_region_densities(&coords, &cfg);

                // Store all densities + hermite data
                let mut s = store.write().unwrap();
                if !s.is_region_generated(&rk) {
                    for (key, density) in densities {
                        if !s.has_density(&key) {
                            let hermite = extract_hermite_data(&density);
                            s.insert(key, density, hermite);
                        }
                    }
                    s.mark_region_generated(rk);
                }
            }

            // Mesh the requested chunk from cached density + extract seam data
            let (mesh, dc_vertices, boundary_edges) = {
                let s = store.read().unwrap();
                let density = match s.density_fields.get(&chunk) {
                    Some(d) => d,
                    None => return,
                };
                let hermite = match s.hermite_data.get(&chunk) {
                    Some(h) => h,
                    None => return,
                };
                let cell_size = density.size - 1;
                let dc_verts = solve_dc_vertices(hermite, cell_size);
                let m = generate_mesh(hermite, &dc_verts, cell_size, cfg.max_edge_length);
                let b_edges = region_gen::extract_boundary_edges(hermite, cfg.chunk_size);
                (m, dc_verts, b_edges)
            };

            // Cache seam data for this chunk
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
            }

            // Convert to UE coordinates and send initial result (no seams yet)
            let converted = convert_mesh_to_ue(&mesh, world_scale);
            let _ = result_tx.send(WorkerResult::ChunkMesh {
                chunk,
                mesh: converted,
                generation,
            });

            // Try to generate seams for this chunk and its neighbors
            incremental_seam_pass(chunk, &cfg, store, result_tx, world_scale);
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
        }
        WorkerRequest::Unload { chunk } => {
            let mut s = store.write().unwrap();
            s.unload(chunk);
            generation_counters.remove(&chunk);
        }
    }
}

/// After meshing chunk C, attempt seam generation for C and its 6 face-adjacent
/// neighbors. Any chunk that produces non-empty seam quads gets re-meshed
/// (from cached hermite) with seams appended and re-sent.
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
) {
    // Collect this chunk + its 6 face-adjacent neighbors to try seam generation
    let candidates = [
        chunk,
        (chunk.0 - 1, chunk.1, chunk.2),
        (chunk.0 + 1, chunk.1, chunk.2),
        (chunk.0, chunk.1 - 1, chunk.2),
        (chunk.0, chunk.1 + 1, chunk.2),
        (chunk.0, chunk.1, chunk.2 - 1),
        (chunk.0, chunk.1, chunk.2 + 1),
    ];

    for &target in &candidates {
        // Generate seam quads for this target chunk using all available neighbor data
        let seam_mesh = {
            let s = store.read().unwrap();
            // Skip if this chunk doesn't have seam data (hasn't been meshed yet)
            if !s.chunk_seam_data.contains_key(&target) {
                continue;
            }
            region_gen::generate_chunk_seam_quads(target, &s.chunk_seam_data, cfg.chunk_size)
        };

        if seam_mesh.triangles.is_empty() {
            continue;
        }

        // Re-mesh the chunk from cached hermite data, append seam quads
        let combined = {
            let s = store.read().unwrap();
            let density = match s.density_fields.get(&target) {
                Some(d) => d,
                None => continue,
            };
            let hermite = match s.hermite_data.get(&target) {
                Some(h) => h,
                None => continue,
            };
            let cell_size = density.size - 1;
            let dc_vertices = solve_dc_vertices(hermite, cell_size);
            let mut mesh = generate_mesh(hermite, &dc_vertices, cell_size, cfg.max_edge_length);
            mesh.append(seam_mesh);
            mesh
        };

        let converted = convert_mesh_to_ue(&combined, world_scale);
        let _ = result_tx.send(WorkerResult::ChunkMesh {
            chunk: target,
            mesh: converted,
            generation: 0, // Seam update
        });
    }
}
