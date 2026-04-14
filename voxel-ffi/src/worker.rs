use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use std::collections::HashSet;

use crossbeam_channel::{Receiver, Sender};
use dashmap::DashMap;
use rayon::prelude::*;
use voxel_core::dual_contouring::mesh_gen::generate_mesh;
use voxel_core::dual_contouring::solve::solve_dc_vertices;
use voxel_core::stress::SupportType;
use voxel_fluid::FluidEvent;
use voxel_gen::config::{GenerationConfig, StressConfig};
use voxel_gen::hermite_extract::extract_hermite_data;
use voxel_gen::region_gen::{
    self, generate_region_densities, region_chunks, region_key, sync_region_boundary_densities,
    ChunkSeamData, RegionTimings,
};

use crate::convert::{convert_mesh_to_ue_scaled, from_ue_normal, from_ue_world_pos};
use crate::engine::terrace_size_for_scale;
use crate::profiler::{ChunkTimings, StreamingProfiler};
use crate::store::ChunkStore;
use crate::types::{FfiCollapseEvent, FfiCrystalPlacement, FfiZoneDescriptor, WorkerRequest, WorkerResult};

/// Map SpringType → FluidType u8 for debug-colored water rendering.
fn spring_type_to_fluid_u8(st: &voxel_gen::springs::SpringType) -> u8 {
    use voxel_gen::springs::SpringType;
    match st {
        SpringType::SpringLine => 3,  // WaterSpringLine (cyan)
        SpringType::VadoseDrip => 4,  // WaterDrip (purple)
        SpringType::AquiferBreach => 5, // WaterBreach (yellow-green)
        SpringType::RiverSource => 6, // WaterRiver (green)
        SpringType::Artesian => 7,    // WaterArtesian (silver)
    }
}

/// Retrieve existing crystal data from ChunkStore for a chunk, converted to UE coords.
/// Used by remesh/seam/mining paths that don't recompute crystals from density.
fn retrieve_crystal_data(
    store: &Arc<RwLock<ChunkStore>>,
    key: (i32, i32, i32),
    voxel_scale: f32,
    world_scale: f32,
) -> Vec<FfiCrystalPlacement> {
    let s = store.read().unwrap();
    match s.crystal_placements.get(&key) {
        Some(placements) if !placements.is_empty() => {
            crate::convert::convert_crystals_to_ue(placements, voxel_scale, world_scale)
        }
        _ => Vec::new(),
    }
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
) {
    while !shutdown.load(Ordering::Relaxed) {
        // Priority 1: mine requests (non-blocking)
        if let Ok(req) = mine_rx.try_recv() {
            handle_request(
                req, &result_tx, &store, &config, &stress_config, &generation_counters,
                world_scale, &fluid_event_tx, &profiler, worker_id, &generate_rx, &mine_rx, &morph_manifest,
            );
            continue;
        }

        // Priority 1.5: deferred stress recalculation (only worker 0 handles this)
        if worker_id == 0 {
            if try_process_stress_queue(&store, &stress_config, &config, &result_tx, world_scale) {
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
fn try_handle_mine(
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
                let (meshes, mined) = if request.mode == 0 {
                    crate::mining::mine_sphere(&mut s, center, radius, &cfg, world_scale)
                } else {
                    let normal = from_ue_normal(request.normal_x, request.normal_y, request.normal_z);
                    crate::mining::mine_peel(&mut s, center, normal, radius, &cfg, world_scale)
                };
                let dirty_keys: Vec<(i32, i32, i32)> = meshes.into_iter().map(|(k, _)| k).collect();
                // Crystal recompute
                for &key in &dirty_keys {
                    if let Some(density) = s.density_fields.get(&key) {
                        let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                        let placements = voxel_gen::compute_crystals(coord, density, &cfg);
                        s.crystal_placements.insert(key, placements);
                    }
                }
                // Queue position-based stress recalculation at mine point
                let stress_center = (center.x as i32, center.y as i32, center.z as i32);
                let stress_radius = radius as i32 + 22; // mine radius + span search(20) + air decay(2)
                s.queue_stress_dirty(stress_center, stress_radius);
                drop(s);
                let _ = result_tx.send(WorkerResult::MinedMaterials { mined });
                batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, world_scale);
                return true;
            }
            _ => {} // non-mine request, ignore
        }
    }
    false
}

/// Deferred stress recalculation: checks if the stress dirty queue timer has elapsed,
/// runs the v2 stress algorithm, emits warnings, and triggers collapse if needed.
/// Returns true if work was done (so worker loop can continue instead of sleeping).
fn try_process_stress_queue(
    store: &Arc<RwLock<ChunkStore>>,
    stress_config: &Arc<RwLock<StressConfig>>,
    config: &Arc<RwLock<GenerationConfig>>,
    result_tx: &Sender<WorkerResult>,
    world_scale: f32,
) -> bool {
    use voxel_core::stress::{
        recalc_stress_region_v2_filtered, detect_and_execute_collapses_v2,
    };
    use crate::types::FfiStressWarning;
    use std::io::Write;
    use std::collections::HashSet;

    let debug_path = "D:/Unreal Projects/Mithril2026/Saved/stress_debug.txt";
    let mut dbg = |msg: String| {
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(debug_path) {
            let _ = writeln!(f, "[{:.2}] {}", std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs_f64() % 10000.0, msg);
        }
    };

    // Check if stress dirty queue is ready (timer elapsed)
    let events = {
        let mut s = store.write().unwrap();
        s.drain_stress_dirty(0.4) // 400ms deferred timer
    };

    let events = match events {
        Some(e) if !e.is_empty() => e,
        _ => return false,
    };

    let cfg = config.read().unwrap().clone();
    let stress_cfg = stress_config.read().unwrap().clone();
    let chunk_size = cfg.chunk_size;

    // Derive affected chunks from events (union of all event bounding boxes)
    let dirty_chunks: Vec<(i32, i32, i32)> = {
        let s = store.read().unwrap();
        let mut chunk_set: HashSet<(i32, i32, i32)> = HashSet::new();
        for event in &events {
            for key in event.affected_chunks(chunk_size) {
                if s.density_fields.contains_key(&key) {
                    chunk_set.insert(key);
                }
            }
        }
        chunk_set.into_iter().collect()
    };

    if dirty_chunks.is_empty() { return false; }

    dbg(format!("=== STRESS RECALC START === events={} derived_chunks={} chunk_size={}",
        events.len(), dirty_chunks.len(), chunk_size));
    for (i, e) in events.iter().enumerate() {
        dbg(format!("  event[{}]: center=({},{},{}) radius={}", i, e.center.0, e.center.1, e.center.2, e.radius));
    }
    dbg(format!("  config: span_w={:.3} min_safe_span={} min_collapse={} slab_cohesion={:.2} max_vol={} depth_scale={:.0}",
        stress_cfg.span_weight, stress_cfg.min_safe_span,
        stress_cfg.min_collapse_region, stress_cfg.slab_cohesion_threshold, stress_cfg.max_collapse_volume,
        stress_cfg.depth_pressure_scale));

    let recalc_start = std::time::Instant::now();

    // Run v2 stress recalculation — only voxels within event radii are recalculated
    let result = {
        let mut s = store.write().unwrap();
        let (density, stress, support) = s.sleep_fields_mut();
        recalc_stress_region_v2_filtered(
            density,
            stress,
            support,
            &stress_cfg,
            &dirty_chunks,
            &events,
            chunk_size,
        )
    };

    let recalc_ms = recalc_start.elapsed().as_secs_f64() * 1000.0;

    // Count stress distribution for DIRTY CHUNKS ONLY (what we just recalculated)
    {
        let s = store.read().unwrap();
        let mut air = 0u32;
        let mut stress_zero = 0u32;
        let mut stress_dust = 0u32;   // 0.01 .. 0.4
        let mut stress_creak = 0u32;  // 0.4 .. 0.6
        let mut stress_shake = 0u32;  // 0.6 .. 0.8
        let mut stress_danger = 0u32; // 0.8 .. 1.0
        let mut stress_over = 0u32;   // >= 1.0
        let grid_size = chunk_size + 1;
        for &key in &dirty_chunks {
            if let (Some(df), Some(ssf)) = (s.density_fields.get(&key), s.stress_fields.get(&key)) {
                for z in 0..grid_size {
                    for y in 0..grid_size {
                        for x in 0..grid_size {
                            if !df.get(x, y, z).material.is_solid() { air += 1; continue; }
                            let stress = ssf.get(x, y, z);
                            if stress <= 0.001 { stress_zero += 1; }
                            else if stress < 0.4 { stress_dust += 1; }
                            else if stress < 0.6 { stress_creak += 1; }
                            else if stress < 0.8 { stress_shake += 1; }
                            else if stress < 1.0 { stress_danger += 1; }
                            else { stress_over += 1; }
                        }
                    }
                }
            }
        }
        let solid = stress_zero + stress_dust + stress_creak + stress_shake + stress_danger + stress_over;
        dbg(format!("  recalc {:.1}ms — {} dirty chunks, {} solid: {} zero, {} dust(<0.4), {} creak(<0.6), {} shake(<0.8), {} danger(<1.0), {} OVER(1.0+)",
            recalc_ms, dirty_chunks.len(), solid, stress_zero, stress_dust, stress_creak, stress_shake, stress_danger, stress_over));
    }
    dbg(format!("  overstressed={} affected_chunks={}",
        result.overstressed.len(), result.affected_chunks.len()));

    // Log stress distribution
    if !result.overstressed.is_empty() {
        let max_stress = result.overstressed.iter().map(|v| v.stress).fold(0.0f32, f32::max);
        let min_stress = result.overstressed.iter().map(|v| v.stress).fold(f32::MAX, f32::min);
        let avg_stress: f32 = result.overstressed.iter().map(|v| v.stress).sum::<f32>() / result.overstressed.len() as f32;
        dbg(format!("  overstressed: min={:.2} avg={:.2} max={:.2}", min_stress, avg_stress, max_stress));
        // Log top 5 by stress
        let mut sorted: Vec<_> = result.overstressed.iter().collect();
        sorted.sort_by(|a, b| b.stress.partial_cmp(&a.stress).unwrap_or(std::cmp::Ordering::Equal));
        // Detailed stress breakdown for top 5 voxels
        let s = store.read().unwrap();
        for (i, ov) in sorted.iter().take(5).enumerate() {
            let (wx, wy, wz) = (ov.world_x, ov.world_y, ov.world_z);
            // Reconstruct components
            let (key, lx, ly, lz) = voxel_core::stress::world_to_chunk_local(wx, wy, wz, chunk_size);
            let mat = s.density_fields.get(&key)
                .map(|df| df.get(lx, ly, lz).material)
                .unwrap_or(voxel_core::material::Material::Air);
            let hardness = stress_cfg.material_hardness[mat as u8 as usize];
            let air_below = voxel_core::stress::count_air_below(&s.density_fields, wx, wy, wz, chunk_size);

            // Count air face-neighbors for cross-section
            let mut air_faces = 0u32;
            for &(dx, dy, dz) in &[(1i32,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)] {
                if let Some((_, m)) = voxel_core::stress::sample_world(&s.density_fields, wx+dx, wy+dy, wz+dz, chunk_size) {
                    if !m.is_solid() { air_faces += 1; }
                }
            }

            dbg(format!("  top[{}]: ({},{},{}) stress={:.3} | mat={:?} hard={:.2} air_below={} air_faces={} | raw≈grav({:.2})+oh({:.2})+xsec({:.2})/h({:.2})",
                i, wx, wy, wz, ov.stress, mat, hardness, air_below, air_faces,
                stress_cfg.gravity_weight, air_below as f32 * stress_cfg.overhang_weight,
                if air_faces >= 2 { (air_faces - 1) as f32 * 0.15 } else { 0.0 },
                hardness));
        }
        drop(s);
    }

    // Emit stress warnings for UE — scan ALL voxels with stress above dust threshold.
    // This is push-based: Rust tells UE where the stress is, UE checks proximity to player.
    {
        let s = store.read().unwrap();
        let mut warnings = Vec::new();
        let mut dust_count = 0u32;
        let mut creak_count = 0u32;
        let mut shake_count = 0u32;
        let grid_size = chunk_size + 1;

        for &(cx, cy, cz) in &dirty_chunks {
            let sf = match s.stress_fields.get(&(cx, cy, cz)) {
                Some(f) => f,
                None => continue,
            };
            for z in 0..grid_size {
                for y in 0..grid_size {
                    for x in 0..grid_size {
                        let stress = sf.get(x, y, z);
                        if stress < stress_cfg.warn_dust_threshold {
                            continue;
                        }
                        let warning_type = if stress >= stress_cfg.warn_shake_threshold {
                            shake_count += 1; 3u8
                        } else if stress >= stress_cfg.warn_creak_threshold {
                            creak_count += 1; 2u8
                        } else {
                            dust_count += 1; 1u8
                        };
                        let wx = cx * chunk_size as i32 + x as i32;
                        let wy = cy * chunk_size as i32 + y as i32;
                        let wz = cz * chunk_size as i32 + z as i32;
                        warnings.push(FfiStressWarning {
                            world_x: wx as f32 * world_scale,
                            world_y: -(wz as f32) * world_scale,
                            world_z: wy as f32 * world_scale,
                            stress,
                            warning_type,
                        });
                    }
                }
            }
        }
        if dust_count + creak_count + shake_count > 0 {
            dbg(format!("  warnings: dust={} creak={} shake={} (scanned from stress fields)",
                dust_count, creak_count, shake_count));
        }
        // Send the highest-stress warnings (sorted, capped at 128)
        warnings.sort_by(|a, b| b.stress.partial_cmp(&a.stress).unwrap_or(std::cmp::Ordering::Equal));
        warnings.truncate(128);
        if !warnings.is_empty() {
            let _ = result_tx.send(WorkerResult::StressWarnings { warnings });
        }
    }

    // Run collapse detection if there are overstressed voxels
    if !result.overstressed.is_empty() {
        let mut s = store.write().unwrap();

        let collapse_start = std::time::Instant::now();
        let events = {
            let (density, stress, support) = s.sleep_fields_mut();
            detect_and_execute_collapses_v2(
                density,
                stress,
                support,
                &result.overstressed,
                &stress_cfg,
                chunk_size,
            )
        };
        let collapse_ms = collapse_start.elapsed().as_secs_f64() * 1000.0;

        if !events.is_empty() {
            let total_voxels: u32 = events.iter().map(|e| e.total_volume).sum();
            let total_slabs: usize = events.iter().map(|e| e.slabs.len()).sum();
            dbg(format!("  === COLLAPSE === events={} total_slabs={} total_voxels={} in {:.1}ms",
                events.len(), total_slabs, total_voxels, collapse_ms));

            for (ei, event) in events.iter().enumerate() {
                dbg(format!("  event[{}]: vol={} slabs={} center=({:.1},{:.1},{:.1}) affected_chunks={}",
                    ei, event.total_volume, event.slabs.len(),
                    event.center.0, event.center.1, event.center.2,
                    event.affected_chunks.len()));
                for (si, slab) in event.slabs.iter().enumerate() {
                    dbg(format!("    slab[{}]: voxels={} fall={} bb=({},{},{})→({},{},{}) mat={:?}",
                        si, slab.voxels.len(), slab.fall_distance,
                        slab.bb_min.0, slab.bb_min.1, slab.bb_min.2,
                        slab.bb_max.0, slab.bb_max.1, slab.bb_max.2,
                        slab.dominant_material));
                }
            }

            // Collect all affected chunks for remeshing
            let mut all_dirty: Vec<((i32,i32,i32), usize, usize, usize, usize, usize, usize)> = Vec::new();
            for event in &events {
                for &key in &event.affected_chunks {
                    all_dirty.push((key, 0, 0, 0, chunk_size, chunk_size, chunk_size));
                }
            }
            all_dirty.sort_by_key(|&(k, ..)| k);
            all_dirty.dedup_by_key(|k| k.0);
            dbg(format!("  remeshing {} chunks", all_dirty.len()));

            let _base_meshes = s.remesh_dirty(&all_dirty, &cfg, world_scale);
            drop(s);

            let mut ffi_events: Vec<FfiCollapseEvent> = events.iter().map(|e| {
                FfiCollapseEvent {
                    center_x: e.center.0 * world_scale,
                    center_y: -e.center.2 * world_scale,
                    center_z: e.center.1 * world_scale,
                    volume: e.total_volume,
                }
            }).collect();
            ffi_events.sort_by(|a, b| b.volume.cmp(&a.volume));

            // Collapse region size distribution
            let mut vol_hist = [0u32; 5]; // [1-5, 6-15, 16-50, 51-150, 150+]
            for e in &events {
                let bucket = if e.total_volume <= 5 { 0 }
                    else if e.total_volume <= 15 { 1 }
                    else if e.total_volume <= 50 { 2 }
                    else if e.total_volume <= 150 { 3 }
                    else { 4 };
                vol_hist[bucket] += 1;
            }
            dbg(format!("  size distribution: tiny(1-5)={} small(6-15)={} medium(16-50)={} large(51-150)={} huge(150+)={}",
                vol_hist[0], vol_hist[1], vol_hist[2], vol_hist[3], vol_hist[4]));

            dbg(format!("  sending {} collapse events to UE (largest vol={})",
                ffi_events.len(), ffi_events.first().map(|e| e.volume).unwrap_or(0)));

            let _ = result_tx.send(WorkerResult::CollapseResult {
                events: ffi_events,
                meshes: Vec::new(),
            });

            let dirty_keys: Vec<(i32, i32, i32)> = all_dirty.iter().map(|&(k, ..)| k).collect();
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, world_scale);
            dbg("  seam pass complete".to_string());
        } else {
            dbg(format!("  no collapses triggered (overstressed={} but regions too small or grounded) in {:.1}ms",
                result.overstressed.len(), collapse_ms));
        }
    } else {
        dbg(format!("  no overstressed voxels — stress is stable"));
    }

    dbg("=== STRESS RECALC END ===".to_string());
    true
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
    mine_rx: &Receiver<WorkerRequest>,
    morph_manifest: &Arc<Mutex<Option<voxel_sleep::ChangeManifest>>>,
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
            let mut t_mesh_smooth = Duration::ZERO;
            let mut was_slow_path = false;
            let mut region_timings = RegionTimings::default();
            let mut t_worm_forward_sharing = Duration::ZERO;
            let mut t_worm_backward_carve = Duration::ZERO;
            let mut t_worm_backward_remesh = Duration::ZERO;
            let mut backward_dirty_count: u32 = 0;

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
                    t_dc_solve += t1.elapsed();

                    let t2 = Instant::now();
                    let mut m = generate_mesh(hermite, &dc_verts, cell_size);
                    t_mesh_gen += t2.elapsed();

                    let t_s = Instant::now();
                    m.smooth(cfg.mesh_smooth_iterations, cfg.mesh_smooth_strength, cfg.mesh_boundary_smooth, Some(cell_size));
                    if cfg.mesh_recalc_normals > 0 { m.recalculate_normals(); }
                    t_mesh_smooth += t_s.elapsed();

                    let b_edges = region_gen::extract_boundary_edges(hermite, cfg.chunk_size);
                    Some((m, dc_verts, b_edges))
                } else {
                    None
                }
            };

            let mut pool_fluid_seeds: Vec<voxel_gen::pools::FluidSeed> = Vec::new();
            let mut region_river_springs: Vec<((i32, i32, i32), voxel_gen::springs::SpringDescriptor)> = Vec::new();
            let mut region_zone_descriptors: Vec<FfiZoneDescriptor> = Vec::new();

            let (mesh, dc_vertices, boundary_edges) = if let Some(result) = mesh_result {
                result
            } else {
                // Slow path: (re)generate region densities
                was_slow_path = true;

                let t0 = Instant::now();
                let coords = region_chunks(rk, cfg.region_size);
                let (mut densities, _pools, fluid_seeds, worm_paths, rt, river_springs, zones) = generate_region_densities(&coords, &cfg);
                pool_fluid_seeds = fluid_seeds;
                region_river_springs = river_springs;

                // Convert zone descriptors to FFI format with coordinate transform
                {
                    let voxel_scale = cfg.effective_bounds() / cfg.chunk_size as f32;
                    let scale = voxel_scale * world_scale;
                    region_zone_descriptors = zones.iter().map(|zd| {
                        FfiZoneDescriptor {
                            zone_type: zd.zone_type as u8,
                            // Rust Y-up → UE Z-up: swap Y↔Z, negate new Y
                            center_x: zd.center.x * scale,
                            center_y: -zd.center.z * scale,
                            center_z: zd.center.y * scale,
                            min_x: zd.world_min.x * scale,
                            min_y: -zd.world_max.z * scale,
                            min_z: zd.world_min.y * scale,
                            max_x: zd.world_max.x * scale,
                            max_y: -zd.world_min.z * scale,
                            max_z: zd.world_max.y * scale,
                        }
                    }).collect();
                }
                t_region_density += t0.elapsed();
                region_timings = rt;
                if profiling {
                }
                // Forward sharing: apply worm paths from already-generated regions
                // into our new density fields (before hermite extraction)
                let t_fwd = Instant::now();
                {
                    let s = store.read().unwrap();
                    let stored = s.get_all_region_worm_paths();
                    let mut external: Vec<&[voxel_gen::worm::path::WormSegment]> = Vec::new();
                    for (rk_other, paths) in stored {
                        if *rk_other == rk { continue; }
                        for path in paths {
                            external.push(path);
                        }
                    }
                    if !external.is_empty() {
                        let as_vecs: Vec<Vec<voxel_gen::worm::path::WormSegment>> =
                            external.into_iter().map(|s| s.to_vec()).collect();
                        region_gen::apply_external_worm_paths(&mut densities, &as_vecs, &cfg);
                        // Recompute metadata after carving external worms
                        for density in densities.values_mut() {
                            density.compute_metadata();
                        }
                        // Re-sync intra-region boundaries broken by external worm carving
                        sync_region_boundary_densities(&mut densities, cfg.chunk_size);
                    }
                }
                if profiling { t_worm_forward_sharing = t_fwd.elapsed(); }

                // Check for pending mine requests between phases
                try_handle_mine(mine_rx, result_tx, store, config, world_scale, fluid_event_tx);

                // Pre-extract hermite data BEFORE acquiring write lock (expensive part)
                let t2 = Instant::now();
                let keyed_data: Vec<_> = densities
                    .into_par_iter()
                    .map(|(key, density)| {
                        let hermite = extract_hermite_data(&density);
                        (key, density, hermite)
                    })
                    .collect();
                t_hermite += t2.elapsed();

                // Write lock held only for fast inserts + worm path storage
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
                    s.store_region_worms(rk, worm_paths.clone());
                }
                // Backward sharing: carve new worms into already-loaded chunks
                // from other regions, then re-extract hermite and re-mesh
                if !worm_paths.is_empty() {
                    let eb = cfg.effective_bounds();
                    let mut backward_dirty: Vec<(i32, i32, i32)> = Vec::new();
                    let t_bwd_carve = Instant::now();
                    // Phase 1: Carve worms + compute_metadata (write lock)
                    {
                        let mut s = store.write().unwrap();
                        for path in &worm_paths {
                            if path.is_empty() { continue; }
                            let (path_min, path_max) = region_gen::worm_path_aabb(path);
                            let min_cx = (path_min.x / eb).floor() as i32;
                            let max_cx = (path_max.x / eb).floor() as i32;
                            let min_cy = (path_min.y / eb).floor() as i32;
                            let max_cy = (path_max.y / eb).floor() as i32;
                            let min_cz = (path_min.z / eb).floor() as i32;
                            let max_cz = (path_max.z / eb).floor() as i32;

                            for cz in min_cz..=max_cz {
                                for cy in min_cy..=max_cy {
                                    for cx in min_cx..=max_cx {
                                        let key = (cx, cy, cz);
                                        if coords.contains(&key) { continue; }
                                        if let Some(density) = s.density_fields.get_mut(&key) {
                                            let coord = voxel_core::chunk::ChunkCoord::new(cx, cy, cz);
                                            if !voxel_gen::worm::carve::worm_overlaps_chunk(
                                                path,
                                                coord.world_origin_bounds(eb),
                                                density.size,
                                            ) {
                                                continue;
                                            }
                                            voxel_gen::worm::carve::carve_worm_into_density(
                                                density,
                                                path,
                                                coord.world_origin_bounds(eb),
                                                cfg.worm.falloff_power,
                                            );
                                            density.compute_metadata();
                                            if !backward_dirty.contains(&key) {
                                                backward_dirty.push(key);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Sync backward-carved chunks with their loaded neighbors
                    if !backward_dirty.is_empty() {
                        let extra_dirty = {
                            let mut s = store.write().unwrap();
                            s.sync_cross_region_densities(&backward_dirty, cfg.chunk_size)
                        };
                        for key in extra_dirty {
                            if !backward_dirty.contains(&key) {
                                backward_dirty.push(key);
                            }
                        }
                    }
                    // Phase 2: Extract hermite (read lock — doesn't block other readers)
                    if !backward_dirty.is_empty() {
                        let hermite_updates: Vec<_> = {
                            let s = store.read().unwrap();
                            backward_dirty.iter().filter_map(|&key| {
                                s.density_fields.get(&key).map(|d| (key, extract_hermite_data(d)))
                            }).collect()
                        };
                        // Phase 3: Store hermite results (brief write lock)
                        {
                            let mut s = store.write().unwrap();
                            for (key, hermite) in hermite_updates {
                                s.hermite_data.insert(key, hermite);
                            }
                        }
                    }
                    if profiling { t_worm_backward_carve = t_bwd_carve.elapsed(); }
                    backward_dirty_count = backward_dirty.len() as u32;

                    // Check for pending mine requests between phases
                    try_handle_mine(mine_rx, result_tx, store, config, world_scale, fluid_event_tx);

                    let t_bwd_remesh = Instant::now();
                    for &key in &backward_dirty {
                        // 1. Solve DC + generate mesh from updated hermite (read lock)
                        let computed = {
                            let s = store.read().unwrap();
                            let density = s.density_fields.get(&key);
                            let hermite = s.hermite_data.get(&key);
                            if let (Some(density), Some(hermite)) = (density, hermite) {
                                let cell_size = density.size - 1;
                                let dc_verts = solve_dc_vertices(hermite, cell_size);
                                let mut m = generate_mesh(hermite, &dc_verts, cell_size);
                                m.smooth(cfg.mesh_smooth_iterations, cfg.mesh_smooth_strength,
                                         cfg.mesh_boundary_smooth, Some(cell_size));
                                if cfg.mesh_recalc_normals > 0 { m.recalculate_normals(); }
                                let b_edges = region_gen::extract_boundary_edges(hermite, cfg.chunk_size);
                                Some((dc_verts, m, b_edges))
                            } else { None }
                        };
                        if let Some((dc_verts, mesh, b_edges)) = computed {
                            // 2. Update seam data + base mesh (write lock)
                            {
                                let mut s = store.write().unwrap();
                                s.add_seam_data(key, ChunkSeamData {
                                    dc_vertices: dc_verts,
                                    world_origin: glam::Vec3::ZERO,
                                    boundary_edges: b_edges,
                                });
                                s.base_meshes.insert(key, mesh.clone());
                            }
                            // 3. Send updated base mesh to UE
                            let mut converted = convert_mesh_to_ue_scaled(&mesh, cfg.voxel_scale(), world_scale);
                            crate::convert::bucket_mesh_by_material(&mut converted);
                            if !converted.indices.is_empty() {
                                let crystal_data = retrieve_crystal_data(store, key, cfg.voxel_scale(), world_scale);
                                let _ = result_tx.send(WorkerResult::ChunkMesh {
                                    chunk: key, mesh: converted, generation: 0, crystal_data,
                                    zone_descriptors: Vec::new(),
                                });
                            }
                        }
                    }
                    if profiling { t_worm_backward_remesh = t_bwd_remesh.elapsed(); }
                }

                // Cross-region boundary density sync: ensure region edge chunks
                // match their already-loaded neighbors from other regions
                {
                    // Phase 1: Sync densities (write lock)
                    let all_dirty_keys: Vec<(i32, i32, i32)>;
                    {
                        let mut s = store.write().unwrap();
                        all_dirty_keys = s.sync_cross_region_densities(&coords, cfg.chunk_size);
                    }

                    if !all_dirty_keys.is_empty() {
                        // Phase 2: Extract hermite for all dirty chunks (read lock)
                        let hermite_updates: Vec<_> = {
                            let s = store.read().unwrap();
                            all_dirty_keys.iter().filter_map(|&key| {
                                s.density_fields.get(&key).map(|d| (key, extract_hermite_data(d)))
                            }).collect()
                        };

                        // Phase 3: Store hermite results (brief write lock)
                        {
                            let mut s = store.write().unwrap();
                            for (key, hermite) in hermite_updates {
                                s.hermite_data.insert(key, hermite);
                            }
                        }

                        // Phase 4: Remesh non-region dirty chunks
                        let region_set: HashSet<_> = coords.iter().copied().collect();
                        for &key in all_dirty_keys.iter().filter(|k| !region_set.contains(k)) {
                            let computed = {
                                let s = store.read().unwrap();
                                let density = s.density_fields.get(&key);
                                let hermite = s.hermite_data.get(&key);
                                if let (Some(density), Some(hermite)) = (density, hermite) {
                                    let cell_size = density.size - 1;
                                    let dc_verts = solve_dc_vertices(hermite, cell_size);
                                    let mut m = generate_mesh(hermite, &dc_verts, cell_size);
                                    m.smooth(cfg.mesh_smooth_iterations, cfg.mesh_smooth_strength,
                                             cfg.mesh_boundary_smooth, Some(cell_size));
                                    if cfg.mesh_recalc_normals > 0 { m.recalculate_normals(); }
                                    let b_edges = region_gen::extract_boundary_edges(hermite, cfg.chunk_size);
                                    Some((dc_verts, m, b_edges))
                                } else { None }
                            };
                            if let Some((dc_verts, mesh, b_edges)) = computed {
                                {
                                    let mut s = store.write().unwrap();
                                    s.add_seam_data(key, ChunkSeamData {
                                        dc_vertices: dc_verts,
                                        world_origin: glam::Vec3::ZERO,
                                        boundary_edges: b_edges,
                                    });
                                    s.base_meshes.insert(key, mesh.clone());
                                }
                                let mut converted = convert_mesh_to_ue_scaled(&mesh, cfg.voxel_scale(), world_scale);
                                crate::convert::bucket_mesh_by_material(&mut converted);
                                if !converted.indices.is_empty() {
                                    let crystal_data = retrieve_crystal_data(store, key, cfg.voxel_scale(), world_scale);
                                    let _ = result_tx.send(WorkerResult::ChunkMesh {
                                        chunk: key, mesh: converted, generation: 0, crystal_data,
                                        zone_descriptors: Vec::new(),
                                    });
                                }
                            }
                        }
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
                let mut m = generate_mesh(hermite, &dc_verts, cell_size);
                if profiling { t_mesh_gen += t5.elapsed(); }

                let t_s = Instant::now();
                m.smooth(cfg.mesh_smooth_iterations, cfg.mesh_smooth_strength, cfg.mesh_boundary_smooth, Some(cell_size));
                if cfg.mesh_recalc_normals > 0 { m.recalculate_normals(); }
                if profiling { t_mesh_smooth += t_s.elapsed(); }

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

            // Extract density values and send to fluid thread
            {
                let s = store.read().unwrap();
                if let Some(density) = s.density_fields.get(&chunk) {
                    let densities: Vec<f32> = density.samples.iter().map(|s| s.density).collect();
                    let _ = fluid_event_tx.send(FluidEvent::DensityUpdate {
                        chunk,
                        densities,
                    });
                    if cfg.fluid_sources_enabled {
                        let _ = fluid_event_tx.send(FluidEvent::PlaceSources { chunk });
                    }

                    if cfg.fluid_sources_enabled {
                        // Detect geological springs (spring lines + vadose drips)
                        let mut geo_springs: Vec<(u8, u8, u8, f32, u8)> = Vec::new();
                        {
                            let springs = voxel_gen::springs::detect_spring_lines(
                                density, chunk, cfg.chunk_size,
                                &cfg.water_table, &cfg.ore.host_rock, cfg.seed,
                            );
                            for s in &springs {
                                geo_springs.push((s.lx, s.ly, s.lz, s.level, spring_type_to_fluid_u8(&s.source_type)));
                            }
                            let drips = voxel_gen::springs::detect_vadose_drips(
                                density, chunk, cfg.chunk_size,
                                &cfg.water_table, cfg.seed,
                            );
                            for d in &drips {
                                geo_springs.push((d.lx, d.ly, d.lz, d.level, spring_type_to_fluid_u8(&d.source_type)));
                            }
                            // Hydrothermal springs near heat sources
                            let hydro = voxel_gen::springs::detect_hydrothermal_springs(
                                density, chunk, cfg.chunk_size,
                                &cfg.water_table, &cfg.hydrothermal, cfg.seed,
                            );
                            for h in &hydro {
                                // Hydrothermal springs use SpringLine type but render as amber
                                geo_springs.push((h.lx, h.ly, h.lz, h.level, 8)); // WaterHydrothermal
                            }
                            // Artesian springs from confined aquifer
                            let artesian = voxel_gen::springs::detect_artesian_springs(
                                density, chunk, cfg.chunk_size,
                                &cfg.artesian, cfg.seed,
                            );
                            for a in &artesian {
                                geo_springs.push((a.lx, a.ly, a.lz, a.level, spring_type_to_fluid_u8(&a.source_type)));
                            }
                            // River springs from carve_rivers (collected during region gen)
                            for (rs_chunk, rs_desc) in &region_river_springs {
                                if *rs_chunk == chunk {
                                    geo_springs.push((rs_desc.lx, rs_desc.ly, rs_desc.lz, rs_desc.level, spring_type_to_fluid_u8(&rs_desc.source_type)));
                                }
                            }
                        }
                        if !geo_springs.is_empty() {
                            let _ = fluid_event_tx.send(FluidEvent::PlaceGeologicalSprings {
                                chunk,
                                springs: geo_springs,
                            });
                        }

                        // Detect pipe lava sources near kimberlite
                        let pipe_lava = voxel_gen::springs::detect_pipe_lava(
                            density, chunk, cfg.chunk_size,
                            &cfg.pipe_lava, cfg.seed,
                        );
                        for lv in &pipe_lava {
                            let _ = fluid_event_tx.send(FluidEvent::AddFluid {
                                chunk,
                                x: lv.lx,
                                y: lv.ly,
                                z: lv.lz,
                                fluid_type: voxel_fluid::cell::FluidType::Lava,
                                level: lv.level,
                                is_source: true,
                            });
                        }
                    }
                }
            }

            // Inject pool fluid seeds into the fluid simulation
            // When fluid_sources_enabled is off, only inject cauldron seeds (is_source == false)
            if was_slow_path {
                for seed in &pool_fluid_seeds {
                    if !cfg.fluid_sources_enabled && seed.is_source {
                        continue; // skip infinite pool sources when toggle is off
                    }
                    let _ = fluid_event_tx.send(FluidEvent::AddFluid {
                        chunk: seed.chunk,
                        x: seed.lx,
                        y: seed.ly,
                        z: seed.lz,
                        fluid_type: match seed.fluid_type {
                            voxel_gen::pools::PoolFluid::Water => voxel_fluid::cell::FluidType::WaterPool,
                            voxel_gen::pools::PoolFluid::Lava => voxel_fluid::cell::FluidType::Lava,
                        },
                        level: 1.0,
                        is_source: seed.is_source,
                    });
                }
            }

            // Compute crystal placements and store them
            let crystal_data = {
                let placements_opt = {
                    let s = store.read().unwrap();
                    s.density_fields.get(&chunk).map(|density| {
                        let coord = voxel_core::chunk::ChunkCoord::new(chunk.0, chunk.1, chunk.2);
                        voxel_gen::compute_crystals(coord, density, &cfg)
                    })
                };
                if let Some(placements) = placements_opt {
                    if !placements.is_empty() {
                        eprintln!("[CRYSTAL] Chunk {:?}: {} placements (first ore_type={}, scale={:.2})",
                            chunk, placements.len(),
                            placements[0].ore_type, placements[0].scale);
                    }
                    let ue_crystals = crate::convert::convert_crystals_to_ue(&placements, cfg.voxel_scale(), world_scale);
                    let mut sw = store.write().unwrap();
                    sw.crystal_placements.insert(chunk, placements);
                    ue_crystals
                } else {
                    Vec::new()
                }
            };

            // Gate 1: replace mesh with hardcoded test cube
            #[cfg(feature = "diag-gate-1")]
            let mesh = crate::convert::diagnostic_test_cube();

            // Convert to UE coordinates and send initial result (no seams yet)
            let t_coord_start = Instant::now();
            let mut converted = convert_mesh_to_ue_scaled(&mesh, cfg.voxel_scale(), world_scale);
            crate::convert::bucket_mesh_by_material(&mut converted);
            let t_coord_transform = if profiling { t_coord_start.elapsed() } else { Duration::ZERO };

            if !converted.positions.is_empty() && converted.indices.is_empty() {
                eprintln!("[WARN] Chunk {:?}: {} vertices but 0 indices (all triangles filtered)",
                    chunk, converted.positions.len());
            }

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

            // Capture per-submesh details for profiler before sending (converted is moved)
            let submesh_details: Vec<(u8, u32, u32)> = if profiling {
                converted.submeshes.iter().map(|s| (s.material_id, s.vertex_count, s.index_count)).collect()
            } else {
                Vec::new()
            };

            let t_send_start = Instant::now();
            let _ = result_tx.send(WorkerResult::ChunkMesh {
                chunk,
                mesh: converted,
                generation,
                crystal_data,
                zone_descriptors: std::mem::take(&mut region_zone_descriptors),
            });
            let t_send_block = if profiling { t_send_start.elapsed() } else { Duration::ZERO };

            // Try to generate seams for this chunk and its neighbors
            // Gate 3: skip seam pass entirely
            #[cfg(feature = "diag-gate-3")]
            let seam_timings = SeamPassTimings {
                total: Duration::ZERO,
                quad_gen: Duration::ZERO,
                mesh_retrieve: Duration::ZERO,
                convert: Duration::ZERO,
                candidates_tried: 0,
                candidates_sent: 0,
            };
            #[cfg(not(feature = "diag-gate-3"))]
            let seam_timings = incremental_seam_pass(chunk, &cfg, store, result_tx, world_scale);
            let t_seam_pass = if profiling { seam_timings.total } else { Duration::ZERO };

            // Write pipeline timing report
            {
                let total = chunk_start.elapsed();
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true)
                    .open("D:/Unreal Projects/Mithril2026/Saved/gen_perf.txt")
                {
                    let rt = &region_timings;
                    let _ = writeln!(f, "chunk({},{},{}) total={:.1}ms slow={} | region_total={:.1} [base_density={:.1} caverns={:.1} worm_plan={:.1} worm_carve={:.1} zones={:.1} pools={:.1} formations={:.1} boundary={:.1} metadata={:.1} worms={}] hermite={:.1} dc={:.1} mesh={:.1} smooth={:.1} seam={:.1} worm_fwd={:.1} bwd_carve={:.1} bwd_remesh={:.1} bwd_dirty={}",
                        chunk.0, chunk.1, chunk.2,
                        total.as_secs_f64() * 1000.0,
                        was_slow_path,
                        t_region_density.as_secs_f64() * 1000.0,
                        rt.base_density.as_secs_f64() * 1000.0,
                        rt.cavern_centers.as_secs_f64() * 1000.0,
                        rt.worm_planning.as_secs_f64() * 1000.0,
                        rt.worm_carving.as_secs_f64() * 1000.0,
                        rt.zones.as_secs_f64() * 1000.0,
                        rt.pools.as_secs_f64() * 1000.0,
                        rt.formations.as_secs_f64() * 1000.0,
                        rt.boundary_sync.as_secs_f64() * 1000.0,
                        rt.metadata.as_secs_f64() * 1000.0,
                        rt.worm_count,
                        t_hermite.as_secs_f64() * 1000.0,
                        t_dc_solve.as_secs_f64() * 1000.0,
                        t_mesh_gen.as_secs_f64() * 1000.0,
                        t_mesh_smooth.as_secs_f64() * 1000.0,
                        t_seam_pass.as_secs_f64() * 1000.0,
                        t_worm_forward_sharing.as_secs_f64() * 1000.0,
                        t_worm_backward_carve.as_secs_f64() * 1000.0,
                        t_worm_backward_remesh.as_secs_f64() * 1000.0,
                        backward_dirty_count,
                    );
                }
            }

            // Record profiling data
            if profiling {
                let gen_queue_len = generate_rx.len() as u32;
                let result_queue_len = 0u32; // Sender doesn't expose queue length

                let timings = ChunkTimings {
                    region_density: t_region_density,
                    hermite: t_hermite,
                    dc_solve: t_dc_solve,
                    mesh_gen: t_mesh_gen,
                    mesh_smooth: t_mesh_smooth,
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
                    worm_base_density: region_timings.base_density,
                    worm_cavern_centers: region_timings.cavern_centers,
                    worm_planning: region_timings.worm_planning,
                    worm_carving: region_timings.worm_carving,
                    worm_pools: region_timings.pools,
                    worm_formations: region_timings.formations,
                    worm_forward_sharing: t_worm_forward_sharing,
                    worm_backward_carve: t_worm_backward_carve,
                    worm_backward_remesh: t_worm_backward_remesh,
                    worm_count: region_timings.worm_count,
                    worm_segment_count: region_timings.worm_segment_count,
                    backward_dirty_count,
                    stress_update: Duration::ZERO,
                    collapse_detect: Duration::ZERO,
                    collapse_remesh: Duration::ZERO,
                };
                profiler.record_chunk_with_coord(
                    worker_id, chunk, timings, gen_queue_len, result_queue_len,
                );
                if !submesh_details.is_empty() {
                    profiler.attach_submesh_info(submesh_details);
                }
            }
        }
        WorkerRequest::Flatten { base_x, base_y, base_z, host_material } => {
            let cfg = config.read().unwrap().clone();
            let mat = voxel_core::material::Material::from_u8(host_material);
            let mut s = store.write().unwrap();
            let ts = terrace_size_for_scale(world_scale);
            let meshes = crate::terrain_ops::flatten_terrace(&mut s, glam::IVec3::new(base_x, base_y, base_z), mat, &cfg, world_scale, ts, 2);
            let dirty_keys: Vec<(i32, i32, i32)> = meshes.into_iter().map(|(k, _)| k).collect();
            drop(s);

            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, world_scale);
        }
        WorkerRequest::FlattenBatch { tiles } => {
            let cfg = config.read().unwrap().clone();
            let mut s = store.write().unwrap();
            let ts = terrace_size_for_scale(world_scale);
            let meshes = crate::terrain_ops::flatten_terrace_batch(&mut s, &tiles, &cfg, world_scale, ts);
            let dirty_keys: Vec<_> = meshes.into_iter().map(|(k, _)| k).collect();
            drop(s);
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, world_scale);
        }
        WorkerRequest::BuildingFlattenBatch { buildings } => {
            let cfg = config.read().unwrap().clone();
            let mut s = store.write().unwrap();
            let mut all_dirty: Vec<(i32, i32, i32)> = Vec::new();
            for &(bx, by, bz, host_mat, footprint, clearance) in &buildings {
                let mat = voxel_core::material::Material::from_u8(host_mat);
                let bts = footprint.max(1);
                let meshes = crate::terrain_ops::flatten_terrace(
                    &mut s,
                    glam::IVec3::new(bx, by, bz),
                    mat,
                    &cfg,
                    world_scale,
                    bts,
                    clearance.max(2),
                );
                all_dirty.extend(meshes.into_iter().map(|(k, _)| k));
            }
            // Deduplicate dirty keys
            all_dirty.sort();
            all_dirty.dedup();
            drop(s);
            // Single seam pass for all flattens combined
            batched_seam_pass_mine(&all_dirty, &cfg, store, result_tx, world_scale);
        }
        WorkerRequest::BuildingFlatten { base_x, base_y, base_z, host_material, footprint_voxels, clearance_voxels } => {
            let cfg = config.read().unwrap().clone();
            let mat = voxel_core::material::Material::from_u8(host_material);
            let mut s = store.write().unwrap();
            let bts = footprint_voxels.max(1);

            // No surface correction — UE-side QueryBuildingSupport already found the
            // correct surface Y. Double-correcting here causes inconsistent floor heights
            // when adjacent placements have different terrain at their center columns.

            let meshes = crate::terrain_ops::flatten_terrace(&mut s, glam::IVec3::new(base_x, base_y, base_z), mat, &cfg, world_scale, bts, clearance_voxels);
            let dirty_keys: Vec<(i32, i32, i32)> = meshes.into_iter().map(|(k, _)| k).collect();
            drop(s);

            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, world_scale);
        }
        WorkerRequest::Mine { request } => {
            {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true)
                    .open("D:/Unreal Projects/Mithril2026/Saved/mine_debug.txt")
                {
                    let _ = writeln!(f, "[MINE] request: ({},{},{}) r={} mode={}",
                        request.world_x, request.world_y, request.world_z, request.radius, request.mode);
                }
            }
            let cfg = config.read().unwrap().clone();

            let center = from_ue_world_pos(
                request.world_x, request.world_y, request.world_z, world_scale,
            );
            let radius = request.radius / world_scale;

            let mut s = store.write().unwrap();
            {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true)
                    .open("D:/Unreal Projects/Mithril2026/Saved/mine_debug.txt")
                {
                    let _ = writeln!(f, "[MINE] rust coords: ({:.1},{:.1},{:.1}) r={:.1}, store chunks={}",
                        center.x, center.y, center.z, radius, s.density_fields.len());
                }
            }
            let (meshes, mined) = if request.mode == 0 {
                crate::mining::mine_sphere(&mut s, center, radius, &cfg, world_scale)
            } else {
                let normal = from_ue_normal(
                    request.normal_x, request.normal_y, request.normal_z,
                );
                crate::mining::mine_peel(&mut s, center, normal, radius, &cfg, world_scale)
            };
            {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true)
                    .open("D:/Unreal Projects/Mithril2026/Saved/mine_debug.txt")
                {
                    let _ = writeln!(f, "[MINE] complete: {} dirty chunks", meshes.len());
                }
            }
            drop(s);

            // Collect dirty chunk keys — don't send meshes yet (seam pass will send
            // them with seam quads included, avoiding a seamless→seamed flash)
            let dirty_keys: Vec<(i32, i32, i32)> = meshes.into_iter().map(|(k, _)| k).collect();

            // Recompute crystal placements for dirty chunks so batched_seam_pass
            // picks up the updated data via retrieve_crystal_data
            for &key in &dirty_keys {
                let s = store.read().unwrap();
                if let Some(density) = s.density_fields.get(&key) {
                    let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                    let placements = voxel_gen::compute_crystals(coord, density, &cfg);
                    drop(s);
                    store.write().unwrap().crystal_placements.insert(key, placements);
                } else {
                    drop(s);
                }
            }

            // Send mined material counts separately
            let _ = result_tx.send(WorkerResult::MinedMaterials { mined });

            // Queue position-based stress recalculation at mine point
            {
                let mut s = store.write().unwrap();
                let stress_center = (center.x as i32, center.y as i32, center.z as i32);
                let stress_radius = radius as i32 + 22;
                s.queue_stress_dirty(stress_center, stress_radius);
            }

            // Send terrain modifications to fluid thread + detect aquifer breaches
            {
                // Approximate mined cells: cells near the mine center in each dirty chunk
                let mine_cx = (center.x / cfg.chunk_size as f32).floor() as i32;
                let mine_cy = (center.y / cfg.chunk_size as f32).floor() as i32;
                let mine_cz = (center.z / cfg.chunk_size as f32).floor() as i32;
                let mine_lx = ((center.x - mine_cx as f32 * cfg.chunk_size as f32) as usize).min(cfg.chunk_size - 1);
                let mine_ly = ((center.y - mine_cy as f32 * cfg.chunk_size as f32) as usize).min(cfg.chunk_size - 1);
                let mine_lz = ((center.z - mine_cz as f32 * cfg.chunk_size as f32) as usize).min(cfg.chunk_size - 1);
                let mined_cells = vec![(mine_lx, mine_ly, mine_lz)];

                let s = store.read().unwrap();
                for &key in &dirty_keys {
                    if let Some(density) = s.density_fields.get(&key) {
                        let densities: Vec<f32> = density.samples.iter().map(|s| s.density).collect();
                        let _ = fluid_event_tx.send(FluidEvent::TerrainModified {
                            chunk: key,
                            densities,
                        });

                        // Detect aquifer breaches near the mined area
                        if cfg.fluid_sources_enabled {
                            let breaches = voxel_gen::springs::detect_aquifer_breaches(
                                density, key, cfg.chunk_size,
                                &cfg.water_table, &cfg.ore.host_rock, cfg.seed,
                                &mined_cells,
                            );
                            for b in &breaches {
                                let _ = fluid_event_tx.send(FluidEvent::AddFluid {
                                    chunk: key,
                                    x: b.lx,
                                    y: b.ly,
                                    z: b.lz,
                                    fluid_type: voxel_fluid::cell::FluidType::WaterBreach,
                                    level: b.level,
                                    is_source: true,
                                });
                            }
                        }
                    }
                }
            }

            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, world_scale);
        }
        WorkerRequest::MineAndFillFluid { world_x, world_y, world_z, radius, fluid_type, world_scale: ws } => {
            let cfg = config.read().unwrap().clone();

            // Convert UE world position to Rust coordinates
            let center = from_ue_world_pos(world_x, world_y, world_z, ws);
            let rust_radius = radius / ws;

            // Step 1: Mine the sphere (same as normal pick)
            let mut s = store.write().unwrap();
            let (meshes, mined) = crate::mining::mine_sphere(&mut s, center, rust_radius, &cfg, ws);
            drop(s);

            let dirty_keys: Vec<(i32, i32, i32)> = meshes.into_iter().map(|(k, _)| k).collect();

            // Step 2: Fill bottom half with non-source fluid
            {
                let s = store.read().unwrap();
                let eb = cfg.effective_bounds();
                let vs = cfg.voxel_scale();
                let r2 = rust_radius * rust_radius;
                let ft = voxel_fluid::cell::FluidType::from_u8(fluid_type);

                let min_cx = ((center.x - rust_radius) / eb).floor() as i32;
                let max_cx = ((center.x + rust_radius) / eb).floor() as i32;
                let min_cy = ((center.y - rust_radius) / eb).floor() as i32;
                let max_cy = ((center.y + rust_radius) / eb).floor() as i32;
                let min_cz = ((center.z - rust_radius) / eb).floor() as i32;
                let max_cz = ((center.z + rust_radius) / eb).floor() as i32;

                for cz in min_cz..=max_cz {
                    for cy in min_cy..=max_cy {
                        for cx in min_cx..=max_cx {
                            let key = (cx, cy, cz);
                            if let Some(density) = s.density_fields.get(&key) {
                                let origin = glam::Vec3::new(cx as f32 * eb, cy as f32 * eb, cz as f32 * eb);
                                let grid_center = (center - origin) / vs;
                                let grid_radius = rust_radius / vs;
                                let lo_x = ((grid_center.x - grid_radius).floor() as i32).max(0) as usize;
                                let hi_x = ((grid_center.x + grid_radius).ceil() as usize + 1).min(density.size);
                                let lo_y = ((grid_center.y - grid_radius).floor() as i32).max(0) as usize;
                                let hi_y = ((grid_center.y + grid_radius).ceil() as usize + 1).min(density.size);
                                let lo_z = ((grid_center.z - grid_radius).floor() as i32).max(0) as usize;
                                let hi_z = ((grid_center.z + grid_radius).ceil() as usize + 1).min(density.size);

                                for z in lo_z..hi_z {
                                    for y in lo_y..hi_y {
                                        for x in lo_x..hi_x {
                                            let world_pos = origin + glam::Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                            let dist2 = (world_pos - center).length_squared();
                                            // Bottom half of sphere (Rust Y-up: below center) and cell is air
                                            if dist2 <= r2 && world_pos.y < center.y && density.get(x, y, z).density <= 0.0 {
                                                eprintln!("[FLUID_FILL]   placing fluid at chunk({},{},{}) local({},{},{}) world({:.0},{:.0},{:.0})",
                                                    key.0, key.1, key.2, x, y, z, world_pos.x, world_pos.y, world_pos.z);
                                                let _ = fluid_event_tx.send(FluidEvent::AddFluid {
                                                    chunk: key,
                                                    x: x as u8,
                                                    y: y as u8,
                                                    z: z as u8,
                                                    fluid_type: ft,
                                                    level: 1.0,
                                                    is_source: false,
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Step 3: Store updated crystal data (seam pass will pick it up)
            for &key in &dirty_keys {
                let s = store.read().unwrap();
                if let Some(density) = s.density_fields.get(&key) {
                    let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                    let placements = voxel_gen::compute_crystals(coord, density, &cfg);
                    drop(s);
                    store.write().unwrap().crystal_placements.insert(key, placements);
                } else {
                    drop(s);
                }
            }

            // Step 4: Send mined material counts
            let _ = result_tx.send(WorkerResult::MinedMaterials { mined });

            // Step 5: Send terrain modifications to fluid thread + detect aquifer breaches
            {
                let mine_cx = (center.x / cfg.chunk_size as f32).floor() as i32;
                let mine_cy = (center.y / cfg.chunk_size as f32).floor() as i32;
                let mine_cz = (center.z / cfg.chunk_size as f32).floor() as i32;
                let mine_lx = ((center.x - mine_cx as f32 * cfg.chunk_size as f32) as usize).min(cfg.chunk_size - 1);
                let mine_ly = ((center.y - mine_cy as f32 * cfg.chunk_size as f32) as usize).min(cfg.chunk_size - 1);
                let mine_lz = ((center.z - mine_cz as f32 * cfg.chunk_size as f32) as usize).min(cfg.chunk_size - 1);
                let mined_cells = vec![(mine_lx, mine_ly, mine_lz)];

                let s = store.read().unwrap();
                for &key in &dirty_keys {
                    if let Some(density) = s.density_fields.get(&key) {
                        let densities: Vec<f32> = density.samples.iter().map(|s| s.density).collect();
                        let _ = fluid_event_tx.send(FluidEvent::TerrainModified {
                            chunk: key,
                            densities,
                        });

                        if cfg.fluid_sources_enabled {
                            let breaches = voxel_gen::springs::detect_aquifer_breaches(
                                density, key, cfg.chunk_size,
                                &cfg.water_table, &cfg.ore.host_rock, cfg.seed,
                                &mined_cells,
                            );
                            for b in &breaches {
                                let _ = fluid_event_tx.send(FluidEvent::AddFluid {
                                    chunk: key,
                                    x: b.lx,
                                    y: b.ly,
                                    z: b.lz,
                                    fluid_type: voxel_fluid::cell::FluidType::WaterBreach,
                                    level: b.level,
                                    is_source: true,
                                });
                            }
                        }
                    }
                }
            }

            // Step 6: Regenerate seams
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, ws);
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
            let (success, _collapse_events, dirty_bounds) = s.place_support(
                (world_x, world_y, world_z), st, &stress_cfg, cfg.chunk_size,
            );

            // Remesh affected chunks
            let remesh_bounds: Vec<_> = dirty_bounds.iter().map(|&(key, (min_x, min_y, min_z, max_x, max_y, max_z))| {
                (key, min_x, min_y, min_z, max_x, max_y, max_z)
            }).collect();
            let meshes = s.remesh_dirty(&remesh_bounds, &cfg, world_scale);
            drop(s);

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
            let (removed, _collapse_events, dirty_bounds) = s.remove_support(
                (world_x, world_y, world_z), &stress_cfg, cfg.chunk_size,
            );

            // Remesh affected chunks
            let remesh_bounds: Vec<_> = dirty_bounds.iter().map(|&(key, (min_x, min_y, min_z, max_x, max_y, max_z))| {
                (key, min_x, min_y, min_z, max_x, max_y, max_z)
            }).collect();
            let meshes = s.remesh_dirty(&remesh_bounds, &cfg, world_scale);
            drop(s);

            // Send support result
            let mesh_pairs: Vec<_> = meshes.into_iter().collect();
            let _ = result_tx.send(WorkerResult::SupportResult {
                success: removed.is_some(),
                meshes: mesh_pairs,
            });
        }
        WorkerRequest::Sleep { player_chunk, sleep_count, sleep_config: sc } => {
            let cfg = config.read().unwrap().clone();
            let sleep_config = sc;
            let t_worker_start = Instant::now();

            // Request fluid snapshot for geological processes
            let (snap_tx, snap_rx) = crossbeam_channel::bounded(1);
            let _ = fluid_event_tx.send(voxel_fluid::FluidEvent::SnapshotRequest { reply_tx: snap_tx });
            let mut fluid_snapshot = snap_rx.recv().unwrap_or_else(|_| voxel_fluid::FluidSnapshot::default());

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
                &mut fluid_snapshot,
                player_chunk,
                sleep_count,
                None, // No progress channel for now
            );

            // Drain solidified lava from the real fluid system
            if sleep_result.lava_solidified > 0 {
                let lava_chunks: Vec<(i32, i32, i32)> = fluid_snapshot.chunks.keys().copied().collect();
                let _ = fluid_event_tx.send(voxel_fluid::FluidEvent::DrainLavaChunks { chunks: lava_chunks });
            }

            // Remesh all dirty chunks (full chunk bounds)
            let t_remesh = Instant::now();
            let dirty_count = sleep_result.dirty_chunks.len();
            let mut dirty_bounds: Vec<_> = sleep_result.dirty_chunks.iter().map(|&key| {
                (key, 0usize, 0usize, 0usize, cfg.chunk_size, cfg.chunk_size, cfg.chunk_size)
            }).collect();

            // NOTE: Do NOT call sync_boundaries here. Sleep uses set_voxel_synced()
            // which already keeps boundary overlap voxels consistent. Running
            // sync_boundary_density on top of that causes material bleeding:
            // its average_boundary_voxel() picks material by density comparison,
            // which can propagate hornfels/skarn to distant chunk boundaries.

            // However, set_voxel_synced writes mirror copies into neighbor chunks'
            // density fields. Those neighbors need remeshing too. Add all 26
            // face/edge/corner neighbors of dirty chunks that are loaded.
            {
                let dirty_set: std::collections::HashSet<(i32,i32,i32)> =
                    sleep_result.dirty_chunks.iter().copied().collect();
                let mut extra: Vec<((i32,i32,i32), usize, usize, usize, usize, usize, usize)> = Vec::new();
                for &(cx, cy, cz) in &sleep_result.dirty_chunks {
                    for dx in -1i32..=1 {
                        for dy in -1i32..=1 {
                            for dz in -1i32..=1 {
                                if dx == 0 && dy == 0 && dz == 0 { continue; }
                                let nk = (cx + dx, cy + dy, cz + dz);
                                if !dirty_set.contains(&nk) && s.density_fields.contains_key(&nk) {
                                    extra.push((nk, 0, 0, 0, cfg.chunk_size, cfg.chunk_size, cfg.chunk_size));
                                }
                            }
                        }
                    }
                }
                // Deduplicate
                extra.sort_by_key(|e| e.0);
                extra.dedup_by_key(|e| e.0);
                dirty_bounds.extend(extra);
            }

            let meshes = s.remesh_dirty(&dirty_bounds, &cfg, world_scale);
            drop(s);
            let t_remesh_elapsed = t_remesh.elapsed();

            // Send each dirty chunk mesh through the normal ChunkMesh pipeline
            // so UE auto-remeshes existing chunk actors
            let t_mesh_send = Instant::now();
            eprintln!("[SLEEP_REMESH] Sending {} chunk meshes (from {} dirty + neighbors)", meshes.len(), dirty_count);
            for (chunk, mesh) in meshes {
                let ue = crate::convert::rust_chunk_to_ue(chunk.0, chunk.1, chunk.2);
                let vert_count = mesh.positions.len() / 3;
                eprintln!("[SLEEP_REMESH]   Rust({},{},{}) → UE({},{},{})  verts={}",
                    chunk.0, chunk.1, chunk.2, ue.0, ue.1, ue.2, vert_count);
                let crystal_data = retrieve_crystal_data(store, chunk, cfg.voxel_scale(), world_scale);
                let _ = result_tx.send(WorkerResult::ChunkMesh {
                    chunk,
                    mesh,
                    generation: 0, // Sleep remesh
                    crystal_data,
                    zone_descriptors: Vec::new(),
                });
            }
            let t_mesh_send_elapsed = t_mesh_send.elapsed();

            // Send collapse events through the normal CollapseResult pipeline
            let t_collapse_send = Instant::now();
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
            let t_collapse_send_elapsed = t_collapse_send.elapsed();

            // Regenerate seams for dirty chunks
            let t_seam = Instant::now();
            let seam_count = sleep_result.dirty_chunks.len();
            batched_seam_pass(&sleep_result.dirty_chunks, &cfg, store, result_tx, world_scale);
            let t_seam_elapsed = t_seam.elapsed();

            // Build combined profile report with worker timings appended
            let t_worker_total = t_worker_start.elapsed();
            let worker_post_total = t_remesh_elapsed + t_mesh_send_elapsed + t_collapse_send_elapsed + t_seam_elapsed;
            let dur_ms = |d: Duration| d.as_secs_f64() * 1000.0;
            let mut report = sleep_result.profile_report.clone();
            use std::fmt::Write as FmtWrite;
            let _ = writeln!(report);
            let _ = writeln!(report, "─── Worker Post-Processing ─────────────────────────");
            let _ = writeln!(report, "  Remesh ({} chunks):  {:.2} ms", dirty_count, dur_ms(t_remesh_elapsed));
            let _ = writeln!(report, "  Mesh send:           {:.2} ms", dur_ms(t_mesh_send_elapsed));
            let _ = writeln!(report, "  Collapse events:     {:.2} ms", dur_ms(t_collapse_send_elapsed));
            let _ = writeln!(report, "  Seam regen ({}):     {:.2} ms", seam_count, dur_ms(t_seam_elapsed));
            let _ = writeln!(report, "  Worker post total:   {:.2} ms", dur_ms(worker_post_total));
            let _ = writeln!(report);
            let _ = writeln!(report, "═══════════════════════════════════════════════════════");
            let _ = writeln!(report, "  GRAND TOTAL (worker): {:.2} ms", dur_ms(t_worker_total));
            let _ = writeln!(report, "═══════════════════════════════════════════════════════");

            // Compact & serialize manifest for morph system — filter to aureole block only
            // Compact manifest (merge multi-phase changes per voxel) but don't filter by block —
            // cinematic mode uses a player-aimed block that differs from Rust's showcase block.
            // Manifest is cached once via set_morph_manifest, so full size (~30MB) is acceptable.
            let mut compact_manifest = sleep_result.manifest.clone();
            compact_manifest.compact();
            let manifest_json = compact_manifest.to_json().unwrap_or_default();

            // Send sleep completion stats (intercepted by engine.poll_result)
            let _ = result_tx.send(WorkerResult::SleepComplete {
                chunks_changed: sleep_result.chunks_changed,
                voxels_metamorphosed: sleep_result.voxels_metamorphosed,
                minerals_grown: sleep_result.minerals_grown,
                supports_degraded: sleep_result.supports_degraded,
                collapses_triggered: sleep_result.collapses_triggered,
                acid_dissolved: sleep_result.acid_dissolved,
                veins_deposited: sleep_result.veins_deposited,
                voxels_enriched: sleep_result.voxels_enriched,
                formations_grown: sleep_result.formations_grown,
                sulfide_dissolved: sleep_result.sulfide_dissolved,
                coal_matured: sleep_result.coal_matured,
                diamonds_formed: sleep_result.diamonds_formed,
                voxels_silicified: sleep_result.voxels_silicified,
                nests_fossilized: sleep_result.nests_fossilized,
                channels_eroded: sleep_result.channels_eroded,
                corpses_fossilized: sleep_result.corpses_fossilized,
                lava_solidified: sleep_result.lava_solidified,
                profile_report: report,
                aureole_glimpse_pos: sleep_result.aureole_glimpse_pos,
                aureole_showcase_block: sleep_result.aureole_showcase_block,
                manifest_json,
                lava_cells: sleep_result.lava_cells,
            });
        }
        WorkerRequest::AureoleOnly { player_chunk, sleep_config: sc } => {
            let cfg = config.read().unwrap().clone();

            // Request fluid snapshot (same as Sleep)
            let (snap_tx, snap_rx) = crossbeam_channel::bounded(1);
            let _ = fluid_event_tx.send(voxel_fluid::FluidEvent::SnapshotRequest { reply_tx: snap_tx });
            let mut fluid_snapshot = snap_rx.recv().unwrap_or_else(|_| voxel_fluid::FluidSnapshot::default());

            let mut s = store.write().unwrap();

            // Run aureole-only (no stress/support fields needed)
            let sleep_result = voxel_sleep::execute_aureole_only(
                &sc,
                &mut s.density_fields,
                &mut fluid_snapshot,
                player_chunk,
            );

            // Drain solidified lava if any
            if sleep_result.lava_solidified > 0 {
                let lava_chunks: Vec<(i32, i32, i32)> = fluid_snapshot.chunks.keys().copied().collect();
                let _ = fluid_event_tx.send(voxel_fluid::FluidEvent::DrainLavaChunks { chunks: lava_chunks });
            }

            // Remesh dirty chunks (same pattern as Sleep)
            let dirty_count = sleep_result.dirty_chunks.len();
            let mut dirty_bounds: Vec<_> = sleep_result.dirty_chunks.iter().map(|&key| {
                (key, 0usize, 0usize, 0usize, cfg.chunk_size, cfg.chunk_size, cfg.chunk_size)
            }).collect();

            // Expand to 26-neighbors so boundary voxels remesh correctly
            {
                let dirty_set: std::collections::HashSet<(i32,i32,i32)> =
                    sleep_result.dirty_chunks.iter().copied().collect();
                let mut extra: Vec<((i32,i32,i32), usize, usize, usize, usize, usize, usize)> = Vec::new();
                for &(cx, cy, cz) in &sleep_result.dirty_chunks {
                    for dx in -1i32..=1 {
                        for dy in -1i32..=1 {
                            for dz in -1i32..=1 {
                                if dx == 0 && dy == 0 && dz == 0 { continue; }
                                let nk = (cx + dx, cy + dy, cz + dz);
                                if !dirty_set.contains(&nk) && s.density_fields.contains_key(&nk) {
                                    extra.push((nk, 0, 0, 0, cfg.chunk_size, cfg.chunk_size, cfg.chunk_size));
                                }
                            }
                        }
                    }
                }
                extra.sort_by_key(|e| e.0);
                extra.dedup_by_key(|e| e.0);
                dirty_bounds.extend(extra);
            }

            let meshes = s.remesh_dirty(&dirty_bounds, &cfg, world_scale);
            drop(s);

            eprintln!("[AUREOLE_REMESH] Sending {} chunk meshes (from {} dirty + neighbors)", meshes.len(), dirty_count);
            for (chunk, mesh) in meshes {
                let ue = crate::convert::rust_chunk_to_ue(chunk.0, chunk.1, chunk.2);
                let vert_count = mesh.positions.len() / 3;
                eprintln!("[AUREOLE_REMESH]   Rust({},{},{}) → UE({},{},{})  verts={}",
                    chunk.0, chunk.1, chunk.2, ue.0, ue.1, ue.2, vert_count);
                let crystal_data = retrieve_crystal_data(store, chunk, cfg.voxel_scale(), world_scale);
                let _ = result_tx.send(WorkerResult::ChunkMesh {
                    chunk,
                    mesh,
                    generation: 0,
                    crystal_data,
                    zone_descriptors: Vec::new(),
                });
            }

            // Build report
            let mut report = sleep_result.profile_report.clone();
            use std::fmt::Write as FmtWrite;
            let _ = writeln!(report, "\nRemeshed {} dirty chunks (+neighbors)", dirty_count);

            // Deliver via SleepComplete so UE can parse debug lines and display results
            let _ = result_tx.send(WorkerResult::SleepComplete {
                chunks_changed: sleep_result.chunks_changed,
                voxels_metamorphosed: sleep_result.voxels_metamorphosed,
                minerals_grown: 0,
                supports_degraded: 0,
                collapses_triggered: 0,
                acid_dissolved: 0,
                veins_deposited: 0,
                voxels_enriched: 0,
                formations_grown: 0,
                sulfide_dissolved: 0,
                coal_matured: sleep_result.coal_matured,
                diamonds_formed: sleep_result.diamonds_formed,
                voxels_silicified: sleep_result.voxels_silicified,
                nests_fossilized: 0,
                channels_eroded: sleep_result.channels_eroded,
                corpses_fossilized: 0,
                lava_solidified: sleep_result.lava_solidified,
                profile_report: report,
                aureole_glimpse_pos: sleep_result.aureole_glimpse_pos,
                aureole_showcase_block: sleep_result.aureole_showcase_block,
                manifest_json: String::new(), // Aureole-only doesn't need morph
                lava_cells: sleep_result.lava_cells,
            });
        }
        WorkerRequest::MorphStep { chunks, step, total_steps } => {
            let cfg = config.read().unwrap().clone();

            // Borrow cached manifest (set once via set_morph_manifest, reused for all 30 steps)
            // Hold lock for duration of step — acceptable since only one morph runs at a time
            let manifest_guard = morph_manifest.lock().unwrap();
            let manifest = match manifest_guard.as_ref() {
                Some(m) => m,
                None => {
                    eprintln!("[MORPH] No cached manifest — call set_morph_manifest first");
                    drop(manifest_guard);
                    let _ = result_tx.send(WorkerResult::MorphMeshes {
                        step, total_steps, meshes: Vec::new(),
                    });
                    return;
                }
            };

            let t = if total_steps > 0 { step as f32 / total_steps as f32 } else { 1.0 };

            // Force ALL chunks active every step to prevent seam cracks between
            // active (morph-updated) and inactive (stale) neighbors.
            // Parallelized mesh gen (rayon) keeps this fast.
            let active: Vec<bool> = vec![true; chunks.len()];

            let s = store.read().unwrap();

            // Phase 1: Clone active density fields and apply manifest interpolation
            let mut density_fields: Vec<Option<voxel_core::density::DensityField>> = Vec::with_capacity(chunks.len());
            for (i, &key) in chunks.iter().enumerate() {
                if !active[i] {
                    density_fields.push(None); // Skip — existing mesh preserved
                    continue;
                }
                match s.density_fields.get(&key) {
                    Some(d) => {
                        let mut df = d.clone();
                        if let Some(delta) = manifest.chunk_deltas.get(&key) {
                            for change in &delta.voxel_changes {
                                let sample = df.get_mut(change.lx, change.ly, change.lz);
                                let old_d = change.old_density;
                                let new_d = change.new_density;
                                // Per-voxel spreading: voxels near heat source (spread=0)
                                // transform first, farthest (spread=1) start at t=0.6
                                let delay_factor = 0.6_f32;
                                let voxel_delay = change.spread_distance * delay_factor;
                                let voxel_t = ((t - voxel_delay) / (1.0 - voxel_delay)).clamp(0.0, 1.0);
                                sample.density = old_d + (new_d - old_d) * voxel_t;
                                let old_mat = voxel_core::material::Material::from_u8(change.old_material);
                                let new_mat = voxel_core::material::Material::from_u8(change.new_material);
                                sample.material = if voxel_t >= 0.5 { new_mat } else { old_mat };
                            }
                        }
                        density_fields.push(Some(df));
                    }
                    None => density_fields.push(None),
                }
            }
            drop(s);

            // Phase 2: Sync boundaries between adjacent chunks in the block.
            // Generalized boundary sync: for each pair of chunks that are adjacent
            // (differ by 1 in exactly one axis), sync the shared face.
            {
                let cs_val = density_fields.iter().flatten().next()
                    .map(|df| df.size - 1).unwrap_or(16);

                let sync_face = |dfs: &mut [Option<voxel_core::density::DensityField>], a: usize, b: usize, axis: usize, cs: usize| {
                    let boundary: Vec<(usize, usize, f32, voxel_core::material::Material)> = {
                        let src = match &dfs[a] { Some(d) => d, None => return };
                        let mut out = Vec::with_capacity((cs + 1) * (cs + 1));
                        for u in 0..=cs {
                            for v in 0..=cs {
                                let s = match axis {
                                    0 => src.get(cs, u, v),
                                    1 => src.get(u, cs, v),
                                    _ => src.get(u, v, cs),
                                };
                                out.push((u, v, s.density, s.material));
                            }
                        }
                        out
                    };
                    if let Some(dst) = &mut dfs[b] {
                        for &(u, v, density, material) in &boundary {
                            let s = match axis {
                                0 => dst.get_mut(0, u, v),
                                1 => dst.get_mut(u, 0, v),
                                _ => dst.get_mut(u, v, 0),
                            };
                            s.density = density;
                            s.material = material;
                        }
                    }
                };

                // Find adjacent pairs: chunks[a] and chunks[b] are adjacent if they
                // differ by exactly 1 in one axis and 0 in the others
                for a in 0..chunks.len() {
                    for b in (a + 1)..chunks.len() {
                        let (ax, ay, az) = chunks[a];
                        let (bx, by, bz) = chunks[b];
                        let dx = bx - ax;
                        let dy = by - ay;
                        let dz = bz - az;
                        if dx == 1 && dy == 0 && dz == 0 {
                            sync_face(&mut density_fields, a, b, 0, cs_val);
                        } else if dx == 0 && dy == 1 && dz == 0 {
                            sync_face(&mut density_fields, a, b, 1, cs_val);
                        } else if dx == 0 && dy == 0 && dz == 1 {
                            sync_face(&mut density_fields, a, b, 2, cs_val);
                        }
                    }
                }
            }

            // Phase 3: Mesh all density fields + build seam data (parallelized with rayon)
            let chunk_size = cfg.chunk_size;

            // Parallel mesh generation — each chunk independently computes hermite + DC + mesh + smooth
            type BEdges = Vec<(voxel_core::hermite::EdgeKey, voxel_core::hermite::EdgeIntersection)>;
            let mesh_results: Vec<Option<(voxel_core::mesh::Mesh, Vec<glam::Vec3>, BEdges)>> =
                density_fields.par_iter().map(|df_opt| {
                    match df_opt {
                        Some(df) => {
                            let h = extract_hermite_data(df);
                            let cell_size = df.size - 1;
                            let dc_verts = solve_dc_vertices(&h, cell_size);
                            let mut mesh = generate_mesh(&h, &dc_verts, cell_size);
                            mesh.smooth(cfg.mesh_smooth_iterations, cfg.mesh_smooth_strength, cfg.mesh_boundary_smooth, Some(cell_size));
                            let boundary_edges = region_gen::extract_boundary_edges(&h, chunk_size);
                            Some((mesh, dc_verts, boundary_edges))
                        }
                        None => None,
                    }
                }).collect();

            // Collect results back into sequential structures (seam_data_map is not thread-safe)
            let mut base_meshes: Vec<Option<voxel_core::mesh::Mesh>> = Vec::with_capacity(chunks.len());
            let mut seam_data_map: std::collections::HashMap<(i32,i32,i32), ChunkSeamData> =
                std::collections::HashMap::new();

            for (i, result) in mesh_results.into_iter().enumerate() {
                match result {
                    Some((mesh, dc_verts, boundary_edges)) => {
                        seam_data_map.insert(chunks[i], ChunkSeamData {
                            dc_vertices: dc_verts,
                            world_origin: glam::Vec3::ZERO,
                            boundary_edges,
                        });
                        base_meshes.push(Some(mesh));
                    }
                    None => {
                        base_meshes.push(None);
                    }
                }
            }

            // Phase 4: Generate seam quads and append to base meshes, then convert
            let mut meshes = Vec::with_capacity(chunks.len());
            for (i, base_opt) in base_meshes.into_iter().enumerate() {
                match base_opt {
                    Some(mut mesh) => {
                        // Generate seam quads for this chunk using neighbors in the block
                        let seam_mesh = region_gen::generate_chunk_seam_quads(
                            chunks[i], &seam_data_map, chunk_size);
                        if !seam_mesh.triangles.is_empty() {
                            mesh.append(seam_mesh);
                        }
                        if cfg.mesh_recalc_normals > 0 { mesh.recalculate_normals(); }

                        let mut converted = convert_mesh_to_ue_scaled(&mesh, cfg.voxel_scale(), world_scale);
                        crate::convert::bucket_mesh_by_material(&mut converted);
                        meshes.push(converted);
                    }
                    None => {
                        meshes.push(crate::types::ConvertedMesh {
                            positions: Vec::new(),
                            normals: Vec::new(),
                            material_ids: Vec::new(),
                            indices: Vec::new(),
                            submeshes: Vec::new(),
                        });
                    }
                }
            }

            eprintln!("[MORPH] Step {}/{}: meshed {} chunks", step, total_steps, meshes.len());

            let _ = result_tx.send(WorkerResult::MorphMeshes {
                step, total_steps, meshes,
            });
        }
        WorkerRequest::WorldScan => {
            let cfg = config.read().unwrap().clone();
            let s = store.read().unwrap();

            // Convert store data to scan-compatible types
            let scan_seam_data: std::collections::HashMap<(i32,i32,i32), voxel_core::world_scan::ScanSeamData> =
                s.chunk_seam_data.iter().map(|(&k, v)| {
                    (k, voxel_core::world_scan::ScanSeamData {
                        boundary_edges: v.boundary_edges.clone(),
                    })
                }).collect();

            let scan_worm_paths: Vec<Vec<voxel_core::world_scan::ScanWormSegment>> =
                s.region_worm_paths.values().flat_map(|paths| {
                    paths.iter().map(|path| {
                        path.iter().map(|seg| voxel_core::world_scan::ScanWormSegment {
                            position: [seg.position.x, seg.position.y, seg.position.z],
                            radius: seg.radius,
                        }).collect::<Vec<_>>()
                    })
                }).collect();

            let result = voxel_core::world_scan::scan_world(
                &s.density_fields,
                &s.base_meshes,
                &scan_seam_data,
                &scan_worm_paths,
                cfg.chunk_size,
            );

            let json = result.to_json_string();
            drop(s);

            let _ = result_tx.send(WorkerResult::ScanComplete { json_report: json });
        }
        WorkerRequest::WorldScanWithConfig { config: scan_config } => {
            let cfg = config.read().unwrap().clone();
            let s = store.read().unwrap();

            let scan_seam_data: std::collections::HashMap<(i32,i32,i32), voxel_core::world_scan::ScanSeamData> =
                s.chunk_seam_data.iter().map(|(&k, v)| {
                    (k, voxel_core::world_scan::ScanSeamData {
                        boundary_edges: v.boundary_edges.clone(),
                    })
                }).collect();

            let scan_worm_paths: Vec<Vec<voxel_core::world_scan::ScanWormSegment>> =
                s.region_worm_paths.values().flat_map(|paths| {
                    paths.iter().map(|path| {
                        path.iter().map(|seg| voxel_core::world_scan::ScanWormSegment {
                            position: [seg.position.x, seg.position.y, seg.position.z],
                            radius: seg.radius,
                        }).collect::<Vec<_>>()
                    })
                }).collect();

            let result = voxel_core::world_scan::scan_world_with_config(
                &s.density_fields,
                &s.base_meshes,
                Some(&s.hermite_data),
                &scan_seam_data,
                &scan_worm_paths,
                cfg.chunk_size,
                &scan_config,
            );

            let json = result.to_json_string();
            drop(s);

            let _ = result_tx.send(WorkerResult::ScanComplete { json_report: json });
        }
        WorkerRequest::ForceSpawnPool { world_x, world_y, world_z, fluid_type } => {
            let cfg = config.read().unwrap().clone();

            // Convert UE world position to Rust coordinates
            let center = from_ue_world_pos(world_x, world_y, world_z, world_scale);
            let chunk_size = cfg.chunk_size;

            // Compute chunk coordinate and local position
            let cx = (center.x / chunk_size as f32).floor() as i32;
            let cy = (center.y / chunk_size as f32).floor() as i32;
            let cz = (center.z / chunk_size as f32).floor() as i32;
            let key = (cx, cy, cz);

            let lx = ((center.x - cx as f32 * chunk_size as f32) as usize).min(chunk_size);
            let ly = ((center.y - cy as f32 * chunk_size as f32) as usize).min(chunk_size);
            let lz = ((center.z - cz as f32 * chunk_size as f32) as usize).min(chunk_size);

            // Check if chunk is loaded
            let has_density = {
                let s = store.read().unwrap();
                s.density_fields.contains_key(&key)
            };

            if !has_density {
                let json = serde_json::json!({
                    "error": "chunk not loaded",
                    "chunk": [cx, cy, cz],
                }).to_string();
                let _ = result_tx.send(WorkerResult::ForceSpawnPoolComplete { json_report: json });
                return;
            }

            let pool_fluid = if fluid_type == 1 {
                voxel_gen::pools::PoolFluid::Lava
            } else {
                voxel_gen::pools::PoolFluid::Water
            };

            // Force spawn pool: carve basin + generate diagnostics
            let (diagnostics, fluid_seeds) = {
                let mut s = store.write().unwrap();
                let density = s.density_fields.get_mut(&key).unwrap();
                let world_origin = glam::Vec3::new(
                    cx as f32 * chunk_size as f32,
                    cy as f32 * chunk_size as f32,
                    cz as f32 * chunk_size as f32,
                );
                voxel_gen::pools::force_spawn_pool(
                    density,
                    &cfg.pools,
                    world_origin,
                    cfg.seed,
                    lx,
                    ly,
                    lz,
                    pool_fluid,
                    key,
                )
            };

            // Remesh the dirty chunk
            {
                let cs = chunk_size;
                let remesh_bounds = vec![(key, 0usize, 0usize, 0usize, cs, cs, cs)];
                let mut s = store.write().unwrap();
                let meshes = s.remesh_dirty(&remesh_bounds, &cfg, world_scale);
                drop(s);

                for (mkey, mesh) in meshes {
                    let crystal_data = retrieve_crystal_data(store, mkey, cfg.voxel_scale(), world_scale);
                    let _ = result_tx.send(WorkerResult::ChunkMesh {
                        chunk: mkey,
                        mesh,
                        generation: 0,
                        crystal_data,
                        zone_descriptors: Vec::new(),
                    });
                }
            }

            // Inject fluid seeds (skip when fluid sources disabled)
            for seed in &fluid_seeds {
                if !cfg.fluid_sources_enabled {
                    continue; // skip fluid when sources disabled
                }
                let ft = match seed.fluid_type {
                    voxel_gen::pools::PoolFluid::Water => voxel_fluid::cell::FluidType::WaterPool,
                    voxel_gen::pools::PoolFluid::Lava => voxel_fluid::cell::FluidType::Lava,
                };
                let _ = fluid_event_tx.send(FluidEvent::AddFluid {
                    chunk: seed.chunk,
                    x: seed.lx,
                    y: seed.ly,
                    z: seed.lz,
                    fluid_type: ft,
                    level: 1.0,
                    is_source: seed.is_source,
                });
            }

            // Run incremental seam pass
            let _ = incremental_seam_pass(key, &cfg, store, result_tx, world_scale);

            // Send diagnostics as JSON
            let json = serde_json::to_string(&diagnostics).unwrap_or_else(|_| "{}".to_string());
            let _ = result_tx.send(WorkerResult::ForceSpawnPoolComplete { json_report: json });
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

/// After meshing chunk C, attempt seam generation for C and its full
/// 26-neighborhood (face, edge, and corner neighbors). Any chunk that produces
/// non-empty seam quads gets combined with the cached base mesh and re-sent.
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

    let mut candidates = Vec::with_capacity(27);
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                candidates.push((chunk.0 + dx, chunk.1 + dy, chunk.2 + dz));
            }
        }
    }

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
            if cfg.mesh_recalc_normals > 0 { mesh.recalculate_normals(); }
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

        if converted.indices.is_empty() {
            continue;  // Don't overwrite base mesh with empty seam update
        }

        let crystal_data = retrieve_crystal_data(store, target, cfg.voxel_scale(), world_scale);
        let _ = result_tx.send(WorkerResult::ChunkMesh {
            chunk: target,
            mesh: converted,
            generation: 0,
            crystal_data,
            zone_descriptors: Vec::new(),
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

/// Deduplicated seam pass for multiple dirty chunks.
/// Computes the union of all 27-neighborhoods and runs each candidate exactly once,
/// avoiding the N× duplication when overlapping neighborhoods re-generate the same seams.
///
/// Dirty chunks are guaranteed to have their mesh sent even if they have no seam quads,
/// since callers rely on this function as the sole sender for mine/flatten results.
fn batched_seam_pass(
    dirty_keys: &[(i32, i32, i32)],
    cfg: &GenerationConfig,
    store: &Arc<RwLock<ChunkStore>>,
    result_tx: &Sender<WorkerResult>,
    world_scale: f32,
) {
    batched_seam_pass_inner(dirty_keys, cfg, store, result_tx, world_scale, false);
}

fn batched_seam_pass_mine(
    dirty_keys: &[(i32, i32, i32)],
    cfg: &GenerationConfig,
    store: &Arc<RwLock<ChunkStore>>,
    result_tx: &Sender<WorkerResult>,
    world_scale: f32,
) {
    batched_seam_pass_inner(dirty_keys, cfg, store, result_tx, world_scale, true);
}

fn batched_seam_pass_inner(
    dirty_keys: &[(i32, i32, i32)],
    cfg: &GenerationConfig,
    store: &Arc<RwLock<ChunkStore>>,
    result_tx: &Sender<WorkerResult>,
    world_scale: f32,
    batch_as_mine: bool,
) {
    let dirty_set: HashSet<(i32, i32, i32)> = dirty_keys.iter().copied().collect();

    let mut candidates: HashSet<(i32, i32, i32)> = HashSet::new();
    for &key in dirty_keys {
        for dx in -1..=1i32 {
            for dy in -1..=1i32 {
                for dz in -1..=1i32 {
                    candidates.insert((key.0 + dx, key.1 + dy, key.2 + dz));
                }
            }
        }
    }

    let mut to_send: Vec<((i32, i32, i32), voxel_core::mesh::Mesh)> = Vec::new();
    let mut sent_keys: HashSet<(i32, i32, i32)> = HashSet::new();
    {
        let s = store.read().unwrap();
        for &target in &candidates {
            if !s.chunk_seam_data.contains_key(&target) {
                continue;
            }
            let seam_mesh = region_gen::generate_chunk_seam_quads(target, &s.chunk_seam_data, cfg.chunk_size);
            let base = match s.base_meshes.get(&target) {
                Some(m) => m.clone(),
                None => continue,
            };
            if seam_mesh.triangles.is_empty() {
                // No seam quads — only send the base mesh if this is a dirty chunk
                // (must still receive its updated mesh after mine/flatten)
                if dirty_set.contains(&target) {
                    let mut mesh = base;
                    if cfg.mesh_recalc_normals > 0 {
                        mesh.recalculate_normals();
                    }
                    to_send.push((target, mesh));
                    sent_keys.insert(target);
                }
                continue;
            }
            let mut mesh = base;
            mesh.append(seam_mesh);
            if cfg.mesh_recalc_normals > 0 {
                mesh.recalculate_normals();
            }
            to_send.push((target, mesh));
            sent_keys.insert(target);
        }

        // Fallback: dirty chunks that have a base mesh but no seam data entry at all
        for &key in dirty_keys {
            if sent_keys.contains(&key) {
                continue;
            }
            if let Some(base) = s.base_meshes.get(&key) {
                let mut mesh = base.clone();
                if cfg.mesh_recalc_normals > 0 {
                    mesh.recalculate_normals();
                }
                to_send.push((key, mesh));
            }
        }
    }

    if batch_as_mine {
        // Send all mine mesh updates as one atomic result — no pop-in
        let mut batch = Vec::new();
        for (target, combined) in to_send {
            let mut converted = convert_mesh_to_ue_scaled(&combined, cfg.voxel_scale(), world_scale);
            crate::convert::bucket_mesh_by_material(&mut converted);
            if converted.indices.is_empty() { continue; }
            let crystal_data = retrieve_crystal_data(store, target, cfg.voxel_scale(), world_scale);
            batch.push((target, converted, crystal_data));
        }
        if !batch.is_empty() {
            let _ = result_tx.send(WorkerResult::MineBatchMesh { meshes: batch });
        }
    } else {
        for (target, combined) in to_send {
            let mut converted = convert_mesh_to_ue_scaled(&combined, cfg.voxel_scale(), world_scale);
            crate::convert::bucket_mesh_by_material(&mut converted);
            if converted.indices.is_empty() { continue; }
            let crystal_data = retrieve_crystal_data(store, target, cfg.voxel_scale(), world_scale);
            let _ = result_tx.send(WorkerResult::ChunkMesh {
                chunk: target, mesh: converted, generation: 0, crystal_data,
                zone_descriptors: Vec::new(),
            });
        }
    }
}
