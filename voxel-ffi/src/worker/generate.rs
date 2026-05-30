//! Region/chunk generation request handler (`Generate` / `PriorityGenerate`).
//!
//! Pure code-movement out of the former monolithic `worker.rs`; this is the
//! merged Generate/PriorityGenerate match-arm body of the old `handle_request`.
//! Behavior is unchanged.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use std::collections::HashSet;

use rayon::prelude::*;
use voxel_core::dual_contouring::mesh_gen::generate_mesh;
use voxel_core::dual_contouring::solve::solve_dc_vertices;
use voxel_fluid::FluidEvent;
use voxel_gen::hermite_extract::extract_hermite_data;
use voxel_gen::region_gen::{
    self, generate_region_densities, region_chunks, region_key, sync_region_boundary_densities,
    ChunkSeamData, RegionTimings,
};

use crate::convert::convert_mesh_to_ue_scaled;
use crate::profiler::ChunkTimings;
use crate::types::{FfiZoneDescriptor, WorkerResult};

use super::seam::{
    batched_seam_pass, hash_mesh, incremental_seam_pass, spring_type_to_fluid_u8, SeamPassTimings,
};
use super::try_handle_mine;

pub(super) fn handle_generate(ctx: &super::HandlerCtx<'_>, chunk: (i32, i32, i32), generation: u64) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let generation_counters = ctx.generation_counters;
    let world_scale = ctx.world_scale;
    let fluid_event_tx = ctx.fluid_event_tx;
    let profiler = ctx.profiler;
    let worker_id = ctx.worker_id;
    let generate_rx = ctx.generate_rx;
    let mine_rx = ctx.mine_rx;
    let regions_in_flight = ctx.regions_in_flight;
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

            // Tracked across the slow-path block so the per-snapshot smart
            // re-seam below (after incremental_seam_pass) knows which chunks
            // had snapshots restored during this region regen. Required to
            // force-clear stale hashes for their backward face neighbors —
            // otherwise hash-skip suppresses the new combined meshes.
            let mut applied_snapshots: Vec<(i32, i32, i32)> = Vec::new();

            // Tracks chunks whose seam_data + base_mesh were freshly regenerated
            // by the post-sync DC re-solve (STALE-SEAM_DATA fix). Their old
            // dc_vertices baked against pre-sync hermite would have produced
            // degenerate seams when neighbors stitched against them; after
            // re-solving we need to ship the new combined meshes via a final
            // batched_seam_pass so UE actually receives the correct geometry.
            let mut sync_remeshed: Vec<(i32, i32, i32)> = Vec::new();

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
                // Fix A: Per-region mutex — blocks if another worker is
                // generating this region, preventing redundant slow-path work.
                let region_mutex = regions_in_flight
                    .entry(rk)
                    .or_insert_with(|| Arc::new(Mutex::new(())))
                    .clone();
                // The mutex guards (), not data — there's no invariant a
                // panicked worker could have violated. Recover from
                // poisoning so a single panic in region-gen code (e.g. an
                // OOB in voxel-gen) does NOT cascade through every other
                // worker that touches the same region. Without this, every
                // peer hits PoisonError on lock().unwrap() and the whole
                // pool dies — which is exactly how PANIC #1 took out 4
                // peers via worker.rs:596 PoisonError.
                let _region_guard = region_mutex.lock().unwrap_or_else(|p| p.into_inner());

                // Re-check fast path under the guard — the owner may have
                // just finished generating this region while we were waiting.
                let retry_result = {
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

                if let Some(result) = retry_result {
                    result
                } else {
                // Slow path: (re)generate region densities.
                // Region guard is held throughout — other workers for this
                // region block on the mutex until we finish.
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

                // Pre-extract hermite data BEFORE acquiring write lock (expensive part).
                // Round 6 Fix B experiment: tried serial here to avoid rayon contention
                // across 8 workers. Reverted — measured +58% regression on initial_load.
                // Parallelism wins even with contention.
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
                        let mut newly_inserted: Vec<(i32, i32, i32)> = Vec::new();
                        for (key, density, hermite) in keyed_data {
                            if !s.has_density(&key) {
                                s.insert(key, density, hermite);
                                newly_inserted.push(key);
                                // Apply saved snapshot if loading a saved game
                                if s.apply_pending_snapshot(key) {
                                    applied_snapshots.push(key);
                                    // Density was patched — re-extract hermite from patched data.
                                    // (May be re-extracted again below if re-sync touches this chunk.)
                                    if let Some(df) = s.density_fields.get(&key) {
                                        let df_clone = df.clone();
                                        let new_hermite = extract_hermite_data(&df_clone);
                                        s.hermite_data.insert(key, new_hermite);
                                    }
                                }
                            }
                        }
                        s.mark_region_generated(rk);

                        // ── CRITICAL: re-sync ALL region chunks after insert ──
                        //
                        // Two bug classes covered here:
                        //
                        // 1) SNAPSHOT-OVERWRITE: apply_pending_snapshot rewrites
                        //    all 29 791 cells of restored chunks — including
                        //    boundary cells that sync_region_boundary_densities
                        //    just made consistent in the temp HashMap. Without
                        //    re-sync, restored chunk boundaries diverge from
                        //    neighbors → seam holes / mesh gaps.
                        //
                        // 2) STALE-PRE-EXISTING: the insert loop above uses
                        //    `if !s.has_density(&key)` so it SKIPS chunks
                        //    already in store. If region regen runs a second
                        //    time (e.g. region marker cleared by an unload but
                        //    some chunks stayed loaded, OR config changed
                        //    mid-session like BlankCanvas being toggled), the
                        //    fresh sync'd density gets discarded for those
                        //    pre-existing chunks. They keep their old values
                        //    and disagree with their freshly-streamed
                        //    neighbors → 961-cell face mismatches in the
                        //    diagnostic.
                        //
                        // Fix: re-sync EVERY chunk in `coords` against its 26
                        // neighbors using the store's current density. This
                        // cleans up both bug classes regardless of which
                        // chunks were freshly inserted vs already-existing,
                        // and regardless of whether snapshots were applied.
                        //
                        // sync_chunk_full_boundaries is idempotent (min(a,b) =
                        // min(min(a,b),b)) so duplicate work between adjacent
                        // chunks in the region is harmless. Hermite
                        // re-extraction only fires for chunks whose density
                        // actually changed, so the cost scales with the size
                        // of inconsistency — typically zero on a clean region.
                        {
                            use std::collections::HashSet;
                            let mut hermite_dirty: HashSet<(i32, i32, i32)> = HashSet::new();
                            for &chunk_key in coords.iter() {
                                if !s.density_fields.contains_key(&chunk_key) {
                                    continue;
                                }
                                let resync_dirty = s
                                    .sync_chunk_full_boundaries(chunk_key, cfg.chunk_size);
                                for k in resync_dirty {
                                    hermite_dirty.insert(k);
                                }
                            }
                            for key in hermite_dirty {
                                // Re-extract hermite from patched density.
                                let new_hermite_opt = s.density_fields.get(&key).map(|df| {
                                    let df_clone = df.clone();
                                    extract_hermite_data(&df_clone)
                                });
                                let Some(new_hermite) = new_hermite_opt else { continue; };

                                // STALE-SEAM_DATA fix: if this chunk was previously
                                // fully meshed (has seam_data + base_mesh from a prior
                                // load), the dc_vertices in its seam_data were solved
                                // against the OLD hermite. After sync touched its
                                // boundary, the dc_vertices are now stale — neighbors
                                // looking up cell positions from this chunk's seam_data
                                // get coordinates that don't match the current density,
                                // producing degenerate/missing seam quads at the shared
                                // face. Symptom: chunks render with their base mesh but
                                // have a black gap toward this neighbor that only
                                // Force Resync clears.
                                //
                                // Re-solve DC + regen base_mesh + boundary_edges
                                // against the fresh hermite so seam computations against
                                // this chunk produce correct geometry. Track the keys so
                                // a batched seam pass can re-emit downstream.
                                let prev_meshed = s.chunk_seam_data.contains_key(&key)
                                    && s.base_meshes.contains_key(&key);
                                if prev_meshed {
                                    let cell_size = cfg.chunk_size;
                                    let dc_verts = solve_dc_vertices(&new_hermite, cell_size);
                                    let mut mesh = generate_mesh(&new_hermite, &dc_verts, cell_size);
                                    mesh.smooth(
                                        cfg.mesh_smooth_iterations,
                                        cfg.mesh_smooth_strength,
                                        cfg.mesh_boundary_smooth,
                                        Some(cell_size),
                                    );
                                    if cfg.mesh_recalc_normals > 0 { mesh.recalculate_normals(); }
                                    let b_edges = region_gen::extract_boundary_edges(&new_hermite, cfg.chunk_size);
                                    s.hermite_data.insert(key, new_hermite);
                                    s.base_meshes.insert(key, mesh);
                                    s.add_seam_data(key, ChunkSeamData {
                                        dc_vertices: dc_verts,
                                        world_origin: glam::Vec3::ZERO,
                                        boundary_edges: b_edges,
                                    });
                                    sync_remeshed.push(key);
                                } else {
                                    s.hermite_data.insert(key, new_hermite);
                                }
                            }
                        }
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
                                        // Critical: skip chunks the user has hand-edited.
                                        // Backward worm carving writes directly into live
                                        // density without going through the snapshot
                                        // preserve/restore mechanism — without this guard,
                                        // entering a fresh region next to a hand-authored
                                        // area destroys the player's custom geometry
                                        // (walls, structures, painted detail) by carving
                                        // worm tunnels through it.
                                        if s.modification_tracker.dirty_chunks.contains(&key) {
                                            continue;
                                        }
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
                        if let Some((dc_verts, mesh, _b_edges)) = computed {
                            // Update seam data + base mesh (write lock).
                            // No base-only send — batched_seam_pass below is the sole
                            // sender. A base-only send here races with other workers'
                            // seam passes: if their combined-mesh send arrives first
                            // and records the hash, this base-only arrives later and
                            // wipes seams via ClearAllMeshSections on UE, while
                            // batched_seam_pass then hash-skips the redo.
                            let mut s = store.write().unwrap();
                            s.add_seam_data(key, ChunkSeamData {
                                dc_vertices: dc_verts,
                                world_origin: glam::Vec3::ZERO,
                                boundary_edges: _b_edges,
                            });
                            s.base_meshes.insert(key, mesh);
                        }
                    }
                    if profiling { t_worm_backward_remesh = t_bwd_remesh.elapsed(); }

                    // Seam pass for backward-carved chunks: the base-only sends above
                    // land on UE via ClearAllMeshSections, wiping any seams those chunks
                    // had before. Re-stitch seams now so the chunks don't show gaps.
                    if !backward_dirty.is_empty() {
                        batched_seam_pass(&backward_dirty, &cfg, store, result_tx, fluid_event_tx, world_scale);
                    }
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

                    let non_region_dirty: Vec<(i32, i32, i32)> = {
                        let region_set: HashSet<_> = coords.iter().copied().collect();
                        all_dirty_keys.iter().copied().filter(|k| !region_set.contains(k)).collect()
                    };

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
                                // No base-only send — see backward-carve comment above.
                                let mut s = store.write().unwrap();
                                s.add_seam_data(key, ChunkSeamData {
                                    dc_vertices: dc_verts,
                                    world_origin: glam::Vec3::ZERO,
                                    boundary_edges: b_edges,
                                });
                                s.base_meshes.insert(key, mesh);
                            }
                        }
                    }

                    // Seam pass for cross-region dirty chunks: sole sender.
                    if !non_region_dirty.is_empty() {
                        batched_seam_pass(&non_region_dirty, &cfg, store, result_tx, fluid_event_tx, world_scale);
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
                }
            };

            // Targeted dump: when worldgen finishes for the suspect chunk(s),
            // write the full mesh + density stats to a focused log so we can
            // see what produced the visible "slate cube wall" without sifting
            // through every chunk in the world. Targets are hardcoded; flip
            // the empty array off when not debugging.
            {
                const TARGETS: &[(i32, i32, i32)] = &[
                    (-2, -1, -1), // user-confirmed slate-cube chunk
                    (-1, -1, -1), (-3, -1, -1),
                    (-2, 0, -1), (-2, -2, -1),
                    (-2, -1, 0), (-2, -1, -2),
                ];
                if TARGETS.contains(&chunk) {
                    use std::io::Write;
                    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true)
                        .open("D:/Unreal Projects/Mithril2026/Saved/chunk_dump.txt")
                    {
                        let s = store.read().unwrap();
                        let df = s.density_fields.get(&chunk);
                        let _ = writeln!(f, "\n=== chunk {:?} ===", chunk);
                        // Sample density slices on each chunk boundary face — if the
                        // cells just INSIDE a boundary are uniformly solid (e.g. all 1.0)
                        // and the boundary cell got sync'd to air, DC produces a flat
                        // wall. If they have natural noise variation (0.3, 0.7, 0.5),
                        // the wall is just a normal rock face.
                        if let Some(df) = df {
                            let cs = cfg.chunk_size;
                            // For each face, sample density at lx=cs-1 (just inside)
                            // and lx=cs (boundary itself), in a 5x5 grid spread.
                            let _ = writeln!(f, "boundary slice +X (interior cell x=cs-1=29):");
                            for v in (0..=cs).step_by(cs/4) {
                                let mut row = String::new();
                                for u in (0..=cs).step_by(cs/4) {
                                    let s_in = df.get(cs-1, u, v);
                                    let s_b = df.get(cs, u, v);
                                    row += &format!(" [{:>5.2}|{:>5.2}]", s_in.density, s_b.density);
                                }
                                let _ = writeln!(f, "  y/z={}:{}", v, row);
                            }
                            let _ = writeln!(f, "boundary slice +Y (interior cell y=cs-1=29):");
                            for v in (0..=cs).step_by(cs/4) {
                                let mut row = String::new();
                                for u in (0..=cs).step_by(cs/4) {
                                    let s_in = df.get(u, cs-1, v);
                                    let s_b = df.get(u, cs, v);
                                    row += &format!(" [{:>5.2}|{:>5.2}]", s_in.density, s_b.density);
                                }
                                let _ = writeln!(f, "  x/z={}:{}", v, row);
                            }
                            let _ = writeln!(f, "boundary slice -Z (interior cell z=1, vs boundary z=0):");
                            for v in (0..=cs).step_by(cs/4) {
                                let mut row = String::new();
                                for u in (0..=cs).step_by(cs/4) {
                                    let s_in = df.get(u, v, 1);
                                    let s_b = df.get(u, v, 0);
                                    row += &format!(" [{:>5.2}|{:>5.2}]", s_in.density, s_b.density);
                                }
                                let _ = writeln!(f, "  x/y={}:{}", v, row);
                            }
                        }
                        if let Some(df) = df {
                            let total = df.samples.len();
                            let mut solid = 0u32;
                            let mut air = 0u32;
                            let mut min_d = f32::INFINITY;
                            let mut max_d = f32::NEG_INFINITY;
                            let mut mat_counts: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
                            for sm in &df.samples {
                                if sm.density > 0.0 { solid += 1; } else { air += 1; }
                                if sm.density < min_d { min_d = sm.density; }
                                if sm.density > max_d { max_d = sm.density; }
                                *mat_counts.entry(format!("{:?}", sm.material)).or_insert(0) += 1;
                            }
                            let _ = writeln!(f, "density: total={} solid={} air={} min={:.3} max={:.3}",
                                total, solid, air, min_d, max_d);
                            let _ = writeln!(f, "materials: {:?}", mat_counts);
                        } else {
                            let _ = writeln!(f, "(no density field in store)");
                        }

                        // Mesh stats
                        let _ = writeln!(f, "mesh: vertices={} triangles={}",
                            mesh.vertices.len(), mesh.triangles.len());
                        let mut vmin = glam::Vec3::splat(f32::INFINITY);
                        let mut vmax = glam::Vec3::splat(f32::NEG_INFINITY);
                        for v in &mesh.vertices {
                            vmin = vmin.min(v.position);
                            vmax = vmax.max(v.position);
                        }
                        let _ = writeln!(f, "mesh AABB: min=({:.2},{:.2},{:.2}) max=({:.2},{:.2},{:.2})",
                            vmin.x, vmin.y, vmin.z, vmax.x, vmax.y, vmax.z);

                        // Coplanar tri counts: bin all triangles by which axis-aligned
                        // plane they're on. A flat planar wall = many tris in one bin.
                        let eps = 0.05_f32;
                        let mut x_bins: std::collections::HashMap<i32, u32> = std::collections::HashMap::new();
                        let mut y_bins: std::collections::HashMap<i32, u32> = std::collections::HashMap::new();
                        let mut z_bins: std::collections::HashMap<i32, u32> = std::collections::HashMap::new();
                        for tri in &mesh.triangles {
                            let p0 = mesh.vertices[tri.indices[0] as usize].position;
                            let p1 = mesh.vertices[tri.indices[1] as usize].position;
                            let p2 = mesh.vertices[tri.indices[2] as usize].position;
                            if (p0.x - p1.x).abs() < eps && (p1.x - p2.x).abs() < eps {
                                *x_bins.entry((p0.x * 2.0).round() as i32).or_insert(0) += 1;
                            }
                            if (p0.y - p1.y).abs() < eps && (p1.y - p2.y).abs() < eps {
                                *y_bins.entry((p0.y * 2.0).round() as i32).or_insert(0) += 1;
                            }
                            if (p0.z - p1.z).abs() < eps && (p1.z - p2.z).abs() < eps {
                                *z_bins.entry((p0.z * 2.0).round() as i32).or_insert(0) += 1;
                            }
                        }
                        let mut x_top: Vec<_> = x_bins.iter().collect();
                        x_top.sort_by_key(|(_, &c)| std::cmp::Reverse(c));
                        let mut y_top: Vec<_> = y_bins.iter().collect();
                        y_top.sort_by_key(|(_, &c)| std::cmp::Reverse(c));
                        let mut z_top: Vec<_> = z_bins.iter().collect();
                        z_top.sort_by_key(|(_, &c)| std::cmp::Reverse(c));
                        let _ = writeln!(f, "coplanar tris top X-planes (x*2 → count): {:?}",
                            x_top.iter().take(5).collect::<Vec<_>>());
                        let _ = writeln!(f, "coplanar tris top Y-planes: {:?}",
                            y_top.iter().take(5).collect::<Vec<_>>());
                        let _ = writeln!(f, "coplanar tris top Z-planes: {:?}",
                            z_top.iter().take(5).collect::<Vec<_>>());
                        let _ = writeln!(f, "boundary edges: {}", boundary_edges.len());
                        let _ = writeln!(f, "dc_vertices: {} (NaN-count={})",
                            dc_vertices.len(),
                            dc_vertices.iter().filter(|v| v.x.is_nan()).count());
                    }
                }
            }

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
                                max_flow_dist: 0, // legacy procedural source — unbounded
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
                        max_flow_dist: 0, // legacy procedural source — unbounded
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

            // Compute mushroom placements and store them (mirrors crystal flow).
            // Skip regeneration when the store already has an entry for this
            // chunk — that means `apply_pending_snapshot` restored saved or
            // session-preserved placements, and worldgen would clobber them.
            let mushroom_data = {
                let (placements_opt, from_save) = {
                    let s = store.read().unwrap();
                    if let Some(saved) = s.mushroom_placements.get(&chunk) {
                        (Some(saved.clone()), true)
                    } else {
                        let p = s.density_fields.get(&chunk).map(|density| {
                            let coord = voxel_core::chunk::ChunkCoord::new(chunk.0, chunk.1, chunk.2);
                            voxel_gen::compute_mushrooms(coord, density, &cfg)
                        });
                        (p, false)
                    }
                };
                if let Some(placements) = placements_opt {
                    let ue_mushrooms = crate::convert::convert_mushrooms_to_ue(&placements, cfg.voxel_scale(), world_scale);
                    if !from_save {
                        let mut sw = store.write().unwrap();
                        sw.mushroom_placements.insert(chunk, placements);
                    }
                    ue_mushrooms
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

            // Record the hash of the base-only mesh we're about to send. Without
            // this, a cross-worker race can cause UE to end up with base-only:
            //   W2 sends combined_A first (from its own seam pass as neighbor),
            //   records hash_combined. W1 sends base-only A afterwards (this
            //   line). W1's incremental_seam_pass computes combined_A (same
            //   content as W2's) and hash-matches W2's recorded hash → skips.
            //   UE channel order: combined, base-only, (nothing). UE ends on
            //   base-only. Recording hash_base here means W1's subsequent
            //   incremental sees hash_base (different from combined) and sends.
            //
            // Hash is computed BEFORE acquiring the write lock — hashing a base
            // mesh iterates every triangle, and holding the write lock during
            // that iteration serializes all other workers waiting for store access.
            let base_hash = hash_mesh(&mesh);
            {
                let mut s = store.write().unwrap();
                s.last_sent_mesh_hash.insert(chunk, base_hash);
            }

            // Debug throttle: GLOBAL mutex so all 8 workers serialize through
            // a single send queue. Each chunk waits its turn → real one-at-a-time
            // streaming the user can observe. File-controlled delay (no rebuild).
            {
                use std::sync::Mutex;
                static THROTTLE: Mutex<()> = Mutex::new(());
                let delay_ms = std::fs::read_to_string(
                    "D:/Unreal Projects/Mithril2026/Saved/voxel_stream_delay_ms.txt"
                ).ok().and_then(|s| s.trim().parse::<u64>().ok()).unwrap_or(0);
                if delay_ms > 0 {
                    let _guard = THROTTLE.lock().unwrap_or_else(|p| p.into_inner());
                    use std::io::Write;
                    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true)
                        .open("D:/Unreal Projects/Mithril2026/Saved/voxel_stream_log.txt")
                    {
                        const WS: f32 = 40.0;
                        let ux = chunk.0 as f32 * cfg.chunk_size as f32 * WS;
                        let uy = -(chunk.2 as f32) * cfg.chunk_size as f32 * WS;
                        let uz = chunk.1 as f32 * cfg.chunk_size as f32 * WS;
                        let _ = writeln!(f,
                            "[STREAM] sending chunk_rust=({},{},{}) chunk_ue=({:.0},{:.0},{:.0}) delay_ms={}",
                            chunk.0, chunk.1, chunk.2, ux, uy, uz, delay_ms
                        );
                    }
                    std::thread::sleep(std::time::Duration::from_millis(delay_ms));
                }
            }

            // ── Isolated VFX-only stress compute (no collapse) ──
            //
            // Worldgen builds density / mesh / crystals but never computes a
            // stress field, so a freshly streamed or loaded chunk has all-zero
            // stress. The crack-decal + warning-dust overlay reads that field
            // via `enumerate_overstressed_in_chunk`, so it shows NOTHING until
            // the player mines nearby and the position-based queue finally
            // recalculates — the "cracks only appear once I hit the area" bug.
            //
            // Compute this chunk's stress now so the overlay can light up the
            // already-stressed (large-span / thin-feature) parts of the cave
            // the instant the chunk appears — on initial load, zone stream-in,
            // and save-load alike (all bring-in paths funnel through here).
            //
            // CRITICAL — this stays VFX-only and must NEVER collapse on load:
            // `recalc_stress_region_v2` only WRITES stress numbers (it takes
            // support_fields read-only, so it cannot decay strut HP) and
            // returns an overstressed list we deliberately DISCARD. It does not
            // call detect_and_execute_collapses_*. Collapse remains exclusive
            // to the mining stress queue, which only ever falls cells it
            // freshly recomputes inside the mine radius — pre-populating the
            // field here cannot trigger a spurious cave-in.
            //
            // Computed under a READ lock into a local one-chunk map (cloning
            // the existing field preserves any save-restored painted-stress
            // overlay, which `set()` never touches), then committed with a
            // brief WRITE lock — so the span-search pass doesn't hold the store
            // write lock and serialize every other worker during the
            // initial-load storm.
            {
                use voxel_core::stress::{recalc_stress_region_v2, StressField};
                let stress_cfg = ctx.stress_config.read().unwrap().clone();
                let gs = cfg.chunk_size + 1;
                let mut dbg_solid = 0u32;
                let mut dbg_ge10 = 0u32;
                let mut dbg_ge15 = 0u32;
                let mut dbg_max = 0.0f32;
                let computed: Option<StressField> = {
                    let s = store.read().unwrap();
                    if let Some(existing) = s.stress_fields.get(&chunk) {
                        let mut local: std::collections::HashMap<(i32, i32, i32), StressField> =
                            std::collections::HashMap::new();
                        local.insert(chunk, existing.clone());
                        // dirty=[chunk], events=[] → unfiltered full recompute
                        // of this one chunk's surface voxels. Neighbour density
                        // is read where loaded (the whole region is present);
                        // cross-region edges refine when the adjacent region
                        // streams in and runs its own compute.
                        recalc_stress_region_v2(
                            &s.density_fields,
                            &mut local,
                            &s.support_fields,
                            &stress_cfg,
                            &[chunk],
                            cfg.chunk_size,
                        );
                        // TEMP VFX diagnostic — distribution over the fresh field.
                        if let (Some(df), Some(lf)) =
                            (s.density_fields.get(&chunk), local.get(&chunk))
                        {
                            for z in 0..gs { for y in 0..gs { for x in 0..gs {
                                if !df.get(x, y, z).material.is_solid() { continue; }
                                dbg_solid += 1;
                                let e = lf.effective(x, y, z);
                                if e > dbg_max { dbg_max = e; }
                                if e >= 1.0 { dbg_ge10 += 1; }
                                if e >= 1.5 { dbg_ge15 += 1; }
                            }}}
                        }
                        local.remove(&chunk)
                    } else {
                        None
                    }
                };
                if let Some(sf) = computed {
                    let mut s = store.write().unwrap();
                    s.stress_fields.insert(chunk, sf);
                }
                // TEMP VFX diagnostic — one line per generated chunk with solids.
                if dbg_solid > 0 {
                    use std::io::Write;
                    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true)
                        .open("D:/Unreal Projects/Mithril2026/Saved/stress_vfx_gen.txt")
                    {
                        let _ = writeln!(f, "[VFXGEN] rust_chunk=({},{},{}) solid={} max={:.2} ge1.0={} ge1.5={}",
                            chunk.0, chunk.1, chunk.2, dbg_solid, dbg_max, dbg_ge10, dbg_ge15);
                    }
                }
            }

            let t_send_start = Instant::now();
            let _ = result_tx.send(WorkerResult::ChunkMesh {
                chunk,
                mesh: converted,
                generation,
                crystal_data,
                mushroom_data,
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

            // ── Per-snapshot smart re-seam ────────────────────────────────
            //
            // When chunks in this region had their pending snapshots applied
            // during region regen, every backward face neighbor of those
            // chunks may have a STALE last_sent_mesh_hash recorded from when
            // it was originally meshed without these chunks loaded. The
            // hash skip in incremental_seam_pass / batched_seam_pass then
            // suppresses the new combined meshes that include the seams to
            // the freshly-restored chunks — leading to chunks that show
            // their base mesh in UE but never receive the seam quads
            // pointing toward the restored neighbor.
            //
            // Force-clear the hash for every snapshot-applied chunk and
            // its 3 backward face neighbors (-X/-Y/-Z), then run a
            // batched_seam_pass over the union. batched_seam_pass expands
            // the dirty set into its 27-neighborhood, so this also catches
            // diagonal/edge neighbors. Hash-skip would have re-suppressed
            // any genuine no-op so we have to remove the hashes BEFORE
            // running the pass — relying on hash-skip after force-clear
            // gives us cheap dedup for chunks whose seams genuinely
            // didn't change.
            //
            // Cost: a batched_seam_pass over <region_size>³ chunks is
            // single-digit ms in the worker thread; the game thread sees
            // small mesh updates trickle through. See
            // [streaming-perf.md] notes — these are existing-actor updates
            // and bypass per-tick spawn budgets.
            if !applied_snapshots.is_empty() {
                use std::collections::HashSet;
                let bwd_offsets: [(i32, i32, i32); 3] = [(-1, 0, 0), (0, -1, 0), (0, 0, -1)];
                let mut targets: HashSet<(i32, i32, i32)> = HashSet::new();
                for &k in &applied_snapshots {
                    targets.insert(k);
                    for &(dx, dy, dz) in &bwd_offsets {
                        targets.insert((k.0 + dx, k.1 + dy, k.2 + dz));
                    }
                }
                {
                    let mut s = store.write().unwrap();
                    for &t in &targets {
                        s.last_sent_mesh_hash.remove(&t);
                    }
                }
                let target_vec: Vec<(i32, i32, i32)> = targets.into_iter().collect();
                batched_seam_pass(&target_vec, &cfg, store, result_tx, fluid_event_tx, world_scale);
            }

            // ── STALE-SEAM_DATA fix: ship the regenerated meshes ──
            //
            // The post-sync DC re-solve in the region-regen block freshened
            // base_mesh + seam_data for every sync-dirty chunk that already
            // had a mesh. Now we have to:
            //   (a) drop their last_sent_mesh_hash so the upcoming
            //       batched_seam_pass can't be hash-skipped, and
            //   (b) run batched_seam_pass over the union, which will fan
            //       out into their 27-neighborhood and emit every combined
            //       mesh whose dc_vertices are now valid.
            //
            // Without this, the regenerated seam_data sits in the store
            // unused — UE keeps showing the stale combined or base-only.
            if !sync_remeshed.is_empty() {
                {
                    let mut s = store.write().unwrap();
                    for &k in &sync_remeshed {
                        s.last_sent_mesh_hash.remove(&k);
                        // Also drop hashes of the 3 backward face neighbors —
                        // their previously-sent combined included this chunk's
                        // stale dc_vertices, so it's also out of date.
                        let bwd: [(i32,i32,i32); 3] = [(-1,0,0),(0,-1,0),(0,0,-1)];
                        for &(dx,dy,dz) in &bwd {
                            s.last_sent_mesh_hash.remove(&(k.0+dx, k.1+dy, k.2+dz));
                        }
                    }
                }
                batched_seam_pass(&sync_remeshed, &cfg, store, result_tx, fluid_event_tx, world_scale);
            }

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
