use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
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
use crate::engine::{terrace_size_for_scale, building_terrace_size_for_scale};
use crate::profiler::{ChunkTimings, StreamingProfiler};
use crate::store::ChunkStore;
use crate::types::{FfiCollapseEvent, FfiCrystalPlacement, WorkerRequest, WorkerResult};

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
                    if profiling { t_dc_solve += t1.elapsed(); }

                    let t2 = Instant::now();
                    let mut m = generate_mesh(hermite, &dc_verts, cell_size);
                    if profiling { t_mesh_gen += t2.elapsed(); }

                    let t_s = Instant::now();
                    m.smooth(cfg.mesh_smooth_iterations, cfg.mesh_smooth_strength, cfg.mesh_boundary_smooth, Some(cell_size));
                    if cfg.mesh_recalc_normals > 0 { m.recalculate_normals(); }
                    if profiling { t_mesh_smooth += t_s.elapsed(); }

                    let b_edges = region_gen::extract_boundary_edges(hermite, cfg.chunk_size);
                    Some((m, dc_verts, b_edges))
                } else {
                    None
                }
            };

            let mut pool_fluid_seeds: Vec<voxel_gen::pools::FluidSeed> = Vec::new();
            let mut region_river_springs: Vec<((i32, i32, i32), voxel_gen::springs::SpringDescriptor)> = Vec::new();

            let (mesh, dc_vertices, boundary_edges) = if let Some(result) = mesh_result {
                result
            } else {
                // Slow path: (re)generate region densities
                was_slow_path = true;

                let t0 = Instant::now();
                let coords = region_chunks(rk, cfg.region_size);
                let (mut densities, _pools, fluid_seeds, worm_paths, rt, river_springs) = generate_region_densities(&coords, &cfg);
                pool_fluid_seeds = fluid_seeds;
                region_river_springs = river_springs;
                if profiling {
                    t_region_density += t0.elapsed();
                    region_timings = rt;
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

                // Pre-extract hermite data BEFORE acquiring write lock (expensive part)
                let t2 = Instant::now();
                let keyed_data: Vec<_> = densities
                    .into_par_iter()
                    .map(|(key, density)| {
                        let hermite = extract_hermite_data(&density);
                        (key, density, hermite)
                    })
                    .collect();
                if profiling { t_hermite += t2.elapsed(); }

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
            let meshes = s.flatten_terrace(glam::IVec3::new(base_x, base_y, base_z), mat, &cfg, world_scale, ts);

            // Collect dirty chunk keys for seam regeneration
            let dirty_keys: Vec<(i32, i32, i32)> = meshes.iter().map(|(k, _)| *k).collect();

            drop(s);
            for (key, mesh) in meshes {
                let crystal_data = retrieve_crystal_data(store, key, cfg.voxel_scale(), world_scale);
                let _ = result_tx.send(WorkerResult::ChunkMesh { chunk: key, mesh, generation: 0, crystal_data });
            }

            // Regenerate seams for dirty chunks and their neighbors (matches mining path)
            for key in dirty_keys {
                let _ = incremental_seam_pass(key, &cfg, store, result_tx, world_scale);
            }
        }
        WorkerRequest::FlattenBatch { tiles } => {
            let cfg = config.read().unwrap().clone();
            let mut s = store.write().unwrap();
            let ts = terrace_size_for_scale(world_scale);
            let meshes = s.flatten_terrace_batch(&tiles, &cfg, world_scale, ts);
            let dirty_keys: Vec<_> = meshes.iter().map(|(k, _)| *k).collect();
            drop(s);
            for (key, mesh) in meshes {
                let crystal_data = retrieve_crystal_data(store, key, cfg.voxel_scale(), world_scale);
                let _ = result_tx.send(WorkerResult::ChunkMesh { chunk: key, mesh, generation: 0, crystal_data });
            }
            for key in dirty_keys {
                let _ = incremental_seam_pass(key, &cfg, store, result_tx, world_scale);
            }
        }
        WorkerRequest::BuildingFlatten { base_x, mut base_y, base_z, host_material } => {
            let cfg = config.read().unwrap().clone();
            let mat = voxel_core::material::Material::from_u8(host_material);
            let mut s = store.write().unwrap();
            let bts = building_terrace_size_for_scale(world_scale);
            let cs = cfg.chunk_size as i32;

            // Surface correction: if base_y is inside solid rock, scan upward
            // to find the actual surface (first air voxel above solid)
            let check_x = base_x + bts / 2;
            let check_z = base_z + bts / 2;
            let cx = check_x.div_euclid(cs);
            let cz = check_z.div_euclid(cs);
            let lx = check_x.rem_euclid(cs) as usize;
            let lz = check_z.rem_euclid(cs) as usize;

            let cy0 = base_y.div_euclid(cs);
            let ly0 = base_y.rem_euclid(cs) as usize;
            if let Some(df) = s.density_fields.get(&(cx, cy0, cz)) {
                if df.get(lx, ly0, lz).density > 0.0 {
                    // Inside solid rock — scan up to find air
                    let original_y = base_y;
                    for dy in 1..=4 {
                        let sy = base_y + dy;
                        let scy = sy.div_euclid(cs);
                        let sly = sy.rem_euclid(cs) as usize;
                        if let Some(df2) = s.density_fields.get(&(cx, scy, cz)) {
                            if df2.get(lx, sly, lz).density <= 0.0 {
                                base_y = sy - 1; // Surface is one below first air
                                break;
                            }
                        }
                    }
                    if base_y != original_y {
                        eprintln!("[voxel] BuildingFlatten: surface-corrected base_y {} -> {}", original_y, base_y);
                    }
                }
            }

            let meshes = s.flatten_terrace(glam::IVec3::new(base_x, base_y, base_z), mat, &cfg, world_scale, bts);

            let dirty_keys: Vec<(i32, i32, i32)> = meshes.iter().map(|(k, _)| *k).collect();

            drop(s);
            for (key, mesh) in meshes {
                let crystal_data = retrieve_crystal_data(store, key, cfg.voxel_scale(), world_scale);
                let _ = result_tx.send(WorkerResult::ChunkMesh { chunk: key, mesh, generation: 0, crystal_data });
            }

            for key in dirty_keys {
                let _ = incremental_seam_pass(key, &cfg, store, result_tx, world_scale);
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
                // Recompute crystal placements from post-mine density so newly
                // exposed ore surfaces get crystals and mined-away surfaces lose them.
                let crystal_data = {
                    let s = store.read().unwrap();
                    if let Some(density) = s.density_fields.get(&key) {
                        let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                        let placements = voxel_gen::compute_crystals(coord, density, &cfg);
                        let ue_crystals = crate::convert::convert_crystals_to_ue(
                            &placements, cfg.voxel_scale(), world_scale,
                        );
                        drop(s);
                        store.write().unwrap().crystal_placements.insert(key, placements);
                        ue_crystals
                    } else {
                        drop(s);
                        Vec::new()
                    }
                };
                let _ = result_tx.send(WorkerResult::ChunkMesh {
                    chunk: key,
                    mesh,
                    generation: 0,
                    crystal_data,
                });
            }

            // Send mined material counts separately
            let _ = result_tx.send(WorkerResult::MinedMaterials { mined });

            // Stress/collapse deferred to sleep-only — no live stress update

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

            // Regenerate seams for dirty chunks and their neighbors
            for key in dirty_keys {
                let _ = incremental_seam_pass(key, &cfg, store, result_tx, world_scale);
            }
        }
        WorkerRequest::MineAndFillFluid { world_x, world_y, world_z, radius, fluid_type, world_scale: ws } => {
            let cfg = config.read().unwrap().clone();

            // Convert UE world position to Rust coordinates
            let center = from_ue_world_pos(world_x, world_y, world_z, ws);
            let rust_radius = radius / ws;

            // Step 1: Mine the sphere (same as normal pick)
            let mut s = store.write().unwrap();
            let (meshes, mined) = s.mine_sphere(center, rust_radius, &cfg, ws);
            drop(s);

            // Collect dirty chunk keys for seam regeneration
            let dirty_keys: Vec<(i32, i32, i32)> = meshes.iter().map(|(k, _)| *k).collect();

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

            // Step 3: Send each dirty chunk mesh + crystal recompute
            for (key, mesh) in meshes {
                let crystal_data = {
                    let s = store.read().unwrap();
                    if let Some(density) = s.density_fields.get(&key) {
                        let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                        let placements = voxel_gen::compute_crystals(coord, density, &cfg);
                        let ue_crystals = crate::convert::convert_crystals_to_ue(
                            &placements, cfg.voxel_scale(), ws,
                        );
                        drop(s);
                        store.write().unwrap().crystal_placements.insert(key, placements);
                        ue_crystals
                    } else {
                        drop(s);
                        Vec::new()
                    }
                };
                let _ = result_tx.send(WorkerResult::ChunkMesh {
                    chunk: key,
                    mesh,
                    generation: 0,
                    crystal_data,
                });
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
            for key in dirty_keys {
                let _ = incremental_seam_pass(key, &cfg, store, result_tx, ws);
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

            // Remesh all dirty chunks (full chunk bounds)
            let t_remesh = Instant::now();
            let dirty_count = sleep_result.dirty_chunks.len();
            let dirty_bounds: Vec<_> = sleep_result.dirty_chunks.iter().map(|&key| {
                (key, 0usize, 0usize, 0usize, cfg.chunk_size, cfg.chunk_size, cfg.chunk_size)
            }).collect();
            let meshes = s.remesh_dirty(&dirty_bounds, &cfg, world_scale);
            drop(s);
            let t_remesh_elapsed = t_remesh.elapsed();

            // Send each dirty chunk mesh through the normal ChunkMesh pipeline
            // so UE auto-remeshes existing chunk actors
            let t_mesh_send = Instant::now();
            for (chunk, mesh) in meshes {
                let crystal_data = retrieve_crystal_data(store, chunk, cfg.voxel_scale(), world_scale);
                let _ = result_tx.send(WorkerResult::ChunkMesh {
                    chunk,
                    mesh,
                    generation: 0, // Sleep remesh
                    crystal_data,
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
            for &key in &sleep_result.dirty_chunks {
                let _ = incremental_seam_pass(key, &cfg, store, result_tx, world_scale);
            }
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
