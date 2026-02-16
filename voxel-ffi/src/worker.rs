use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender};
use dashmap::DashMap;
use voxel_core::chunk::ChunkCoord;
use voxel_core::dual_contouring::mesh_gen::generate_mesh;
use voxel_core::dual_contouring::solve::solve_dc_vertices;
use voxel_gen::config::GenerationConfig;
use voxel_gen::hermite_extract::extract_hermite_data;

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
            let coord = ChunkCoord::new(chunk.0, chunk.1, chunk.2);

            // Generate density field (includes worm carving)
            let density = voxel_gen::generate_density(coord, &cfg);

            // Extract hermite data
            let hermite = extract_hermite_data(&density);
            let cell_size = density.size - 1;

            // Solve DC vertices and generate mesh
            let dc_vertices = solve_dc_vertices(&hermite, cell_size);
            let mesh = generate_mesh(&hermite, &dc_vertices, cell_size, cfg.max_edge_length);

            // Vertices stay in LOCAL chunk space [0..chunk_size].
            // The UE actor's world position handles the chunk offset.

            // Store density + hermite for future mining
            {
                let mut s = store.write().unwrap();
                s.insert(chunk, density, hermite);
            }

            // Convert to UE coordinates
            let converted = convert_mesh_to_ue(&mesh, world_scale);

            let _ = result_tx.send(WorkerResult::ChunkMesh {
                chunk,
                mesh: converted,
                generation,
            });
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

            let _ = result_tx.send(WorkerResult::MineResults { meshes, mined });
        }
        WorkerRequest::Unload { chunk } => {
            let mut s = store.write().unwrap();
            s.unload(chunk);
            generation_counters.remove(&chunk);
        }
    }
}
