//! Worker-thread request handling, decomposed from the former monolithic
//! `worker.rs` into cohesive submodules. This module owns the worker loop and
//! the thin `handle_request` dispatcher; the actual per-request work lives in
//! the handler submodules. Pure code-movement — behavior is unchanged.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};
use dashmap::DashMap;
use voxel_fluid::FluidEvent;
use voxel_gen::config::{GenerationConfig, StressConfig};

use crate::convert::{from_ue_normal, from_ue_world_pos};
use crate::profiler::StreamingProfiler;
use crate::store::ChunkStore;
use crate::types::{WorkerRequest, WorkerResult};

mod brush;
mod generate;
mod pathing;
mod scan_support;
mod seam;
mod sleep_morph;
mod stress;

// Re-exports so external callers (engine.rs) keep `crate::worker::worker_loop`
// and `crate::worker::path_worker_loop` resolving unchanged.
pub use pathing::path_worker_loop;

use seam::batched_seam_pass_mine;
use stress::try_process_stress_queue;

/// Shared context threaded into every request handler. Holds borrowed
/// references to the worker's channels, stores, and config so each extracted
/// handler can bind the exact locals the original match-arm body referenced.
pub(crate) struct HandlerCtx<'a> {
    pub result_tx: &'a Sender<WorkerResult>,
    pub store: &'a Arc<RwLock<ChunkStore>>,
    pub config: &'a Arc<RwLock<GenerationConfig>>,
    pub stress_config: &'a Arc<RwLock<StressConfig>>,
    pub generation_counters: &'a Arc<DashMap<(i32, i32, i32), AtomicU64>>,
    pub world_scale: f32,
    pub fluid_event_tx: &'a Sender<FluidEvent>,
    pub profiler: &'a Arc<StreamingProfiler>,
    pub worker_id: usize,
    pub generate_rx: &'a Receiver<WorkerRequest>,
    pub mine_rx: &'a Receiver<WorkerRequest>,
    pub morph_manifest: &'a Arc<Mutex<Option<voxel_sleep::ChangeManifest>>>,
    pub regions_in_flight: &'a Arc<DashMap<(i32, i32, i32), Arc<Mutex<()>>>>,
    pub crystal_anchors: &'a Arc<Mutex<crate::crystal_anchors::CrystalAnchorManager>>,
}

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
    morph_manifest: Arc<Mutex<Option<voxel_sleep::ChangeManifest>>>,
    regions_in_flight: Arc<DashMap<(i32, i32, i32), Arc<Mutex<()>>>>,
    crystal_anchors: Arc<Mutex<crate::crystal_anchors::CrystalAnchorManager>>,
) {
    while !shutdown.load(Ordering::Relaxed) {
        // Priority 1: mine requests (non-blocking)
        if let Ok(req) = mine_rx.try_recv() {
            handle_request(
                req, &result_tx, &store, &config, &stress_config, &generation_counters,
                world_scale, &fluid_event_tx, &profiler, worker_id, &generate_rx, &mine_rx, &morph_manifest,
                &regions_in_flight, &crystal_anchors,
            );
            continue;
        }

        // Priority 1.5: deferred stress recalculation (only worker 0 handles this)
        if worker_id == 0 {
            if try_process_stress_queue(&store, &stress_config, &config, &result_tx, &fluid_event_tx, world_scale) {
                continue;
            }
        }

        // Priority 2: generate requests (blocking with timeout)
        let idle_start = Instant::now();
        match generate_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(req) => {
                profiler.record_worker_idle(worker_id, idle_start.elapsed());
                handle_request(
                    req, &result_tx, &store, &config, &stress_config, &generation_counters,
                    world_scale, &fluid_event_tx, &profiler, worker_id, &generate_rx, &mine_rx, &morph_manifest,
                    &regions_in_flight, &crystal_anchors,
                );
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                profiler.record_worker_idle(worker_id, idle_start.elapsed());
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }
}

/// Check mine queue and handle any pending mine request. Returns true if one was handled.
pub(crate) fn try_handle_mine(
    mine_rx: &Receiver<WorkerRequest>,
    result_tx: &Sender<WorkerResult>,
    store: &Arc<RwLock<ChunkStore>>,
    config: &Arc<RwLock<GenerationConfig>>,
    world_scale: f32,
    fluid_event_tx: &Sender<FluidEvent>,
) -> bool {
    if let Ok(req) = mine_rx.try_recv() {
        match req {
            WorkerRequest::Mine { request } => {
                // Inline mine handling (same as handle_request Mine branch)
                let cfg = config.read().unwrap().clone();
                let center = from_ue_world_pos(
                    request.world_x, request.world_y, request.world_z, world_scale,
                );
                let radius = request.radius / world_scale;
                let mut s = store.write().unwrap();
                let outcome = if request.mode == 0 {
                    crate::mining::mine_sphere(&mut s, center, radius, &cfg, world_scale)
                } else {
                    let normal = from_ue_normal(request.normal_x, request.normal_y, request.normal_z);
                    crate::mining::mine_peel(&mut s, center, normal, radius, &cfg, world_scale)
                };
                let dirty_keys: Vec<(i32, i32, i32)> = outcome.meshes.into_iter().map(|(k, _)| k).collect();
                let mined = outcome.mined;
                // Fix B: crystal recompute only for chunks where a material flip
                // actually occurred. Boundary-sync chunks (density tweaks only) keep
                // their existing crystal placements — no recompute needed.
                for &key in &outcome.flipped_chunks {
                    if let Some(density) = s.density_fields.get(&key) {
                        let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                        let placements = voxel_gen::compute_crystals(coord, density, &cfg);
                        s.crystal_placements.insert(key, placements);
                    }
                }
                // Queue position-based stress recalculation at mine point
                let stress_center = (center.x as i32, center.y as i32, center.z as i32);
                let stress_radius = radius as i32 + 4; // mine radius + air decay(2) + small margin
                s.queue_stress_dirty(stress_center, stress_radius);
                drop(s);
                let _ = result_tx.send(WorkerResult::MinedMaterials { mined });
                batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
                return true;
            }
            _ => {} // non-mine request, ignore
        }
    }
    false
}

/// Thin dispatcher: builds a `HandlerCtx` and routes each `WorkerRequest`
/// variant to its extracted handler. An early `return` inside a handler is
/// equivalent to falling off the matched arm — the dispatcher does no
/// post-arm work.
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
    mine_rx: &Receiver<WorkerRequest>,
    morph_manifest: &Arc<Mutex<Option<voxel_sleep::ChangeManifest>>>,
    regions_in_flight: &Arc<DashMap<(i32, i32, i32), Arc<Mutex<()>>>>,
    crystal_anchors: &Arc<Mutex<crate::crystal_anchors::CrystalAnchorManager>>,
) {
    let ctx = HandlerCtx {
        result_tx,
        store,
        config,
        stress_config,
        generation_counters,
        world_scale,
        fluid_event_tx,
        profiler,
        worker_id,
        generate_rx,
        mine_rx,
        morph_manifest,
        regions_in_flight,
        crystal_anchors,
    };
    match req {
        // ComputePath is handled exclusively by the dedicated path-worker
        // (see `path_worker_loop`). If it ever lands here it means routing
        // confusion — silently drop rather than panic.
        WorkerRequest::ComputePath { .. } => {}
        WorkerRequest::PriorityGenerate { chunk, generation } |
        WorkerRequest::Generate { chunk, generation } => {
            generate::handle_generate(&ctx, chunk, generation);
        }
        WorkerRequest::Flatten { base_x, base_y, base_z, host_material } => {
            brush::handle_flatten(&ctx, base_x, base_y, base_z, host_material);
        }
        WorkerRequest::FlattenBatch { tiles } => {
            brush::handle_flatten_batch(&ctx, tiles);
        }
        WorkerRequest::BuildingFlattenBatch { buildings } => {
            brush::handle_building_flatten_batch(&ctx, buildings);
        }
        WorkerRequest::BuildingFlatten { base_x, base_y, base_z, base_y_float, host_material, footprint_voxels, clearance_voxels } => {
            brush::handle_building_flatten(&ctx, base_x, base_y, base_z, base_y_float, host_material, footprint_voxels, clearance_voxels);
        }
        WorkerRequest::Mine { request } => {
            brush::handle_mine(&ctx, request);
        }
        WorkerRequest::MineAndFillFluid { world_x, world_y, world_z, radius, fluid_type, world_scale: ws } => {
            brush::handle_mine_and_fill_fluid(&ctx, world_x, world_y, world_z, radius, fluid_type, ws);
        }
        WorkerRequest::BrushSphere { request } => {
            brush::handle_brush_sphere(&ctx, request);
        }
        WorkerRequest::BrushTunnel { points, radius, material } => {
            brush::handle_brush_tunnel(&ctx, points, radius, material);
        }
        WorkerRequest::BrushPlaceMushroom { center_rust, kind, search_radius, scale, yaw } => {
            brush::handle_brush_place_mushroom(&ctx, center_rust, kind, search_radius, scale, yaw);
        }
        WorkerRequest::BrushPlaceMushroomSphere { center_rust, radius, density, clustering, kind, seed } => {
            brush::handle_brush_place_mushroom_sphere(&ctx, center_rust, radius, density, clustering, kind, seed);
        }
        WorkerRequest::BrushEraseMushroomSphere { center_rust, radius, kind_filter } => {
            brush::handle_brush_erase_mushroom_sphere(&ctx, center_rust, radius, kind_filter);
        }
        WorkerRequest::BrushFormation { center_rust, formation_type, material, height, radius } => {
            brush::handle_brush_formation(&ctx, center_rust, formation_type, material, height, radius);
        }
        WorkerRequest::ForceChunkResync { chunk } => {
            brush::handle_force_chunk_resync(&ctx, chunk);
        }
        WorkerRequest::BrushCavernStamp { chunk_origin, extent, decorate, fluids, seed } => {
            brush::handle_brush_cavern_stamp(&ctx, chunk_origin, extent, decorate, fluids, seed);
        }
        WorkerRequest::BrushFormationStamp { center_rust, radius, seed } => {
            brush::handle_brush_formation_stamp(&ctx, center_rust, radius, seed);
        }
        WorkerRequest::BrushBox { center_rust, half_ext_rust, yaw_rad, op, material } => {
            brush::handle_brush_box(&ctx, center_rust, half_ext_rust, yaw_rad, op, material);
        }
        WorkerRequest::BrushCylinder { center_rust, radius, height, op, material } => {
            brush::handle_brush_cylinder(&ctx, center_rust, radius, height, op, material);
        }
        WorkerRequest::BrushSmooth { center_rust, radius, iterations, strength } => {
            brush::handle_brush_smooth(&ctx, center_rust, radius, iterations, strength);
        }
        WorkerRequest::BrushOrePaint {
            center_rust,
            radius,
            cluster_size,
            min_spacing,
            channel_prob,
            channel_length,
            channel_radius,
            density,
            seed,
            weights,
        } => {
            brush::handle_brush_ore_paint(&ctx, center_rust, radius, cluster_size, min_spacing, channel_prob, channel_length, channel_radius, density, seed, weights);
        }
        WorkerRequest::BrushPaintStress { center_rust, radius, amount, cap, op, falloff } => {
            brush::handle_brush_paint_stress(&ctx, center_rust, radius, amount, cap, op, falloff);
        }
        WorkerRequest::BrushClearAllPaintedStress => {
            brush::handle_brush_clear_all_painted_stress(&ctx);
        }
        WorkerRequest::BrushUndo => {
            brush::handle_brush_undo(&ctx);
        }
        WorkerRequest::BrushNoise { center_rust, radius, frequency, strength, seed } => {
            brush::handle_brush_noise(&ctx, center_rust, radius, frequency, strength, seed);
        }
        WorkerRequest::BrushFluidSphere { center_rust, radius, fluid_type, is_source, op, max_flow_dist } => {
            brush::handle_brush_fluid_sphere(&ctx, center_rust, radius, fluid_type, is_source, op, max_flow_dist);
        }
        WorkerRequest::BrushFluidBox { center_rust, half_ext_rust, fluid_type, is_source, op, max_flow_dist } => {
            brush::handle_brush_fluid_box(&ctx, center_rust, half_ext_rust, fluid_type, is_source, op, max_flow_dist);
        }
        WorkerRequest::BrushFluidRiver { points, radius, fluid_type, is_source, op, max_flow_dist } => {
            brush::handle_brush_fluid_river(&ctx, points, radius, fluid_type, is_source, op, max_flow_dist);
        }
        WorkerRequest::ApplyLavaQuench { obsidian, scoria, drained_water } => {
            brush::handle_apply_lava_quench(&ctx, obsidian, scoria, drained_water);
        }
        WorkerRequest::Unload { chunk } => {
            brush::handle_unload(&ctx, chunk);
        }
        WorkerRequest::PlaceSupport { world_x, world_y, world_z, support_type } => {
            scan_support::handle_place_support(&ctx, world_x, world_y, world_z, support_type);
        }
        WorkerRequest::RemoveSupport { world_x, world_y, world_z } => {
            scan_support::handle_remove_support(&ctx, world_x, world_y, world_z);
        }
        WorkerRequest::Sleep { player_chunk, sleep_count, sleep_config: sc } => {
            sleep_morph::handle_sleep(&ctx, player_chunk, sleep_count, sc);
        }
        WorkerRequest::AureoleOnly { player_chunk, sleep_config: sc } => {
            sleep_morph::handle_aureole_only(&ctx, player_chunk, sc);
        }
        WorkerRequest::MorphStep { chunks, step, total_steps } => {
            sleep_morph::handle_morph_step(&ctx, chunks, step, total_steps);
        }
        WorkerRequest::WorldScan => {
            scan_support::handle_world_scan(&ctx);
        }
        WorkerRequest::WorldScanWithConfig { config: scan_config } => {
            scan_support::handle_world_scan_with_config(&ctx, scan_config);
        }
        WorkerRequest::ForceSpawnPool { world_x, world_y, world_z, fluid_type } => {
            scan_support::handle_force_spawn_pool(&ctx, world_x, world_y, world_z, fluid_type);
        }
    }
}
