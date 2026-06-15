//! Terrain-edit + brush + mining + fluid request handlers.
//!
//! Pure code-movement out of the former monolithic `worker.rs`. Each function
//! is one match-arm body of the old `handle_request`, threading the same
//! context via `super::HandlerCtx`. Behavior is unchanged.

use std::collections::HashSet;

use voxel_core::dual_contouring::mesh_gen::generate_mesh;
use voxel_core::dual_contouring::solve::solve_dc_vertices;
use voxel_fluid::FluidEvent;
use voxel_gen::hermite_extract::extract_hermite_data;
use voxel_gen::region_gen::ChunkSeamData;

use crate::convert::{convert_mesh_to_ue_scaled, from_ue_normal, from_ue_world_pos};
use crate::engine::terrace_size_for_scale;
use crate::types::WorkerResult;

use super::seam::{
    batched_seam_pass, batched_seam_pass_mine, hash_mesh,
    prune_destroyed_mushrooms_for_chunks, recompute_crystals_for_chunks,
};

pub(super) fn handle_flatten(ctx: &super::HandlerCtx<'_>, base_x: i32, base_y: i32, base_z: i32, host_material: u8) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let world_scale = ctx.world_scale;
    let fluid_event_tx = ctx.fluid_event_tx;
            let cfg = config.read().unwrap().clone();
            let mat = voxel_core::material::Material::from_u8(host_material);
            let mut s = store.write().unwrap();
            let ts = terrace_size_for_scale(world_scale);
            // DK2-style zone flatten: integer Y is fine (no per-tile float).
            let meshes = crate::terrain_ops::flatten_terrace(&mut s, glam::IVec3::new(base_x, base_y, base_z), base_y as f32, mat, &cfg, world_scale, ts, 2);
            let dirty_keys: Vec<(i32, i32, i32)> = meshes.into_iter().map(|(k, _)| k).collect();
            drop(s);

            recompute_crystals_for_chunks(store, &cfg, &dirty_keys);
            prune_destroyed_mushrooms_for_chunks(store, &dirty_keys);
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
}

pub(super) fn handle_flatten_batch(ctx: &super::HandlerCtx<'_>, tiles: Vec<(glam::IVec3, voxel_core::material::Material)>) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let world_scale = ctx.world_scale;
    let fluid_event_tx = ctx.fluid_event_tx;
            let cfg = config.read().unwrap().clone();
            let mut s = store.write().unwrap();
            let ts = terrace_size_for_scale(world_scale);
            let meshes = crate::terrain_ops::flatten_terrace_batch(&mut s, &tiles, &cfg, world_scale, ts);
            let dirty_keys: Vec<_> = meshes.into_iter().map(|(k, _)| k).collect();
            drop(s);
            recompute_crystals_for_chunks(store, &cfg, &dirty_keys);
            prune_destroyed_mushrooms_for_chunks(store, &dirty_keys);
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
}

pub(super) fn handle_building_flatten_batch(ctx: &super::HandlerCtx<'_>, buildings: Vec<(i32, i32, i32, f32, u8, i32, i32)>) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let world_scale = ctx.world_scale;
    let fluid_event_tx = ctx.fluid_event_tx;
            // Cheap path: per-tile call to the legacy ramp flatten. Each tile
            // carries its own sub-voxel base_y_float so conveyors don't
            // float/sink (3A density tweak applied inside apply_ramp_column).
            let cfg = config.read().unwrap().clone();
            let mut s = store.write().unwrap();
            let mut all_dirty: Vec<(i32, i32, i32)> = Vec::new();
            for &(bx, by, bz, by_f, host_mat, footprint, clearance) in &buildings {
                let mat = voxel_core::material::Material::from_u8(host_mat);
                let bts = footprint.max(1);
                let meshes = crate::terrain_ops::flatten_terrace(
                    &mut s,
                    glam::IVec3::new(bx, by, bz),
                    by_f,
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
            recompute_crystals_for_chunks(store, &cfg, &all_dirty);
            prune_destroyed_mushrooms_for_chunks(store, &all_dirty);
            batched_seam_pass_mine(&all_dirty, &cfg, store, result_tx, fluid_event_tx, world_scale);
}

pub(super) fn handle_building_flatten(ctx: &super::HandlerCtx<'_>, base_x: i32, base_y: i32, base_z: i32, base_y_float: f32, host_material: u8, footprint_voxels: i32, clearance_voxels: i32) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let world_scale = ctx.world_scale;
    let fluid_event_tx = ctx.fluid_event_tx;
            let cfg = config.read().unwrap().clone();
            let mat = voxel_core::material::Material::from_u8(host_material);
            let mut s = store.write().unwrap();
            let bts = footprint_voxels.max(1);

            // Single placement: route to the all-in SDF flatten (1C+3C+2C).
            // base_y_float carries the exact sub-voxel Y so the iso surface
            // lands exactly where UE wants it (no float/sink). Cantilever
            // columns get convex-hull buttresses to nearby natural rock.
            let meshes = crate::flatten_sdf::flatten_terrace_sdf(
                &mut s,
                glam::IVec3::new(base_x, base_y, base_z),
                base_y_float,
                mat,
                &cfg,
                world_scale,
                bts,
                clearance_voxels,
            );
            let dirty_keys: Vec<(i32, i32, i32)> = meshes.into_iter().map(|(k, _)| k).collect();
            drop(s);

            recompute_crystals_for_chunks(store, &cfg, &dirty_keys);
            prune_destroyed_mushrooms_for_chunks(store, &dirty_keys);
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
}

pub(super) fn handle_mine(ctx: &super::HandlerCtx<'_>, request: crate::types::FfiMineRequest) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let stress_config = ctx.stress_config;
    let world_scale = ctx.world_scale;
    let fluid_event_tx = ctx.fluid_event_tx;
            let cfg = config.read().unwrap().clone();

            let center = from_ue_world_pos(
                request.world_x, request.world_y, request.world_z, world_scale,
            );
            let radius = request.radius / world_scale;

            let mut s = store.write().unwrap();
            let store_chunks_count = s.density_fields.len();
            let outcome = if request.mode == 0 {
                crate::mining::mine_sphere(&mut s, center, radius, &cfg, world_scale)
            } else {
                let normal = from_ue_normal(
                    request.normal_x, request.normal_y, request.normal_z,
                );
                crate::mining::mine_peel(&mut s, center, normal, radius, &cfg, world_scale)
            };
            let dirty_count = outcome.meshes.len();
            let mined = outcome.mined;
            let flipped_chunks = outcome.flipped_chunks;
            let meshes = outcome.meshes;
            drop(s);

            // Perf: all mine_debug.txt I/O is now after drop(s), consolidated into
            // ONE file open (was 3 opens, 2 of them under the write lock — blocking
            // every other worker for ~200-800μs on a contended NTFS handle cache).
            {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true)
                    .open("D:/Unreal Projects/Mithril2026/Saved/mine_debug.txt")
                {
                    let _ = writeln!(f, "[MINE] request: ({},{},{}) r={} mode={} | rust coords: ({:.1},{:.1},{:.1}) r={:.1}, store chunks={} | complete: {} dirty chunks",
                        request.world_x, request.world_y, request.world_z, request.radius, request.mode,
                        center.x, center.y, center.z, radius, store_chunks_count,
                        dirty_count);
                }
            }

            // Collect dirty chunk keys — don't send meshes yet (seam pass will send
            // them with seam quads included, avoiding a seamless→seamed flash)
            let dirty_keys: Vec<(i32, i32, i32)> = meshes.into_iter().map(|(k, _)| k).collect();

            // Recompute crystal placements for dirty chunks so batched_seam_pass
            // picks up the updated data via retrieve_crystal_data.
            // Single read lock for all computes, single write lock for all inserts —
            // replaces N× per-key read+write acquisition (~2-5ms saved per mine with
            // many dirty chunks under contended locking).
            // Fix B: iterate only flipped_chunks (material actually changed).
            // Boundary-sync extras in dirty_keys have density tweaks only, no new
            // material placement, so their crystal layout is unchanged.
            let new_placements: Vec<_> = {
                let s = store.read().unwrap();
                flipped_chunks.iter().filter_map(|&key| {
                    s.density_fields.get(&key).map(|density| {
                        let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                        (key, voxel_gen::compute_crystals(coord, density, &cfg))
                    })
                }).collect()
            };
            // Merge crystal insert + stress-dirty queue into a single write lock.
            // Previously these were two separate write lock acquisitions sandwiching
            // a channel send — every mine paid for two lock round-trips on the
            // contended store. Channel send moved after the write lock; ordering
            // is independent (stress + crystal writes don't depend on the send).
            {
                let mut s = store.write().unwrap();
                for (key, placements) in new_placements {
                    s.crystal_placements.insert(key, placements);
                }
                let stress_center = (center.x as i32, center.y as i32, center.z as i32);
                // ⚠ Cinematic mining pipeline. Scan radius = mine radius
                // (rust voxels) + `mining_stress_scan_buffer` (tuned via UE
                // Codex). This sphere is what gets stress-recomputed; a
                // 26-connected BFS then walks through cohesive pre-stressed
                // rock so the actual slab-detection reach is sphere + that
                // chain. NOT the legacy `propagation_radius` — that one only
                // serves the sleep collapse path now.
                // Note: `cfg` in this scope is the GenerationConfig clone.
                // The StressConfig lives in a separate Arc<RwLock<…>> —
                // hold the read lock only long enough to copy the u32 out.
                let scan_buffer = stress_config.read().unwrap().mining_stress_scan_buffer;
                let stress_radius = radius as i32 + scan_buffer as i32;
                s.queue_stress_dirty(stress_center, stress_radius);
            }

            // Mushroom destruction: drop any whose anchor voxel was mined.
            // Uses dirty_keys (all chunks whose density actually changed in this op).
            prune_destroyed_mushrooms_for_chunks(store, &dirty_keys);

            // Send mined material counts (outside the store lock)
            let _ = result_tx.send(WorkerResult::MinedMaterials { mined });

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
                                    max_flow_dist: 0, // procedural breach — unbounded
                                });
                            }
                        }
                    }
                }
            }

            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
}

pub(super) fn handle_mine_and_fill_fluid(ctx: &super::HandlerCtx<'_>, world_x: f32, world_y: f32, world_z: f32, radius: f32, fluid_type: u8, ws: f32) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let fluid_event_tx = ctx.fluid_event_tx;
            let cfg = config.read().unwrap().clone();

            // Convert UE world position to Rust coordinates
            let center = from_ue_world_pos(world_x, world_y, world_z, ws);
            let rust_radius = radius / ws;

            // Step 1: Mine the sphere (same as normal pick)
            let mut s = store.write().unwrap();
            let outcome = crate::mining::mine_sphere(&mut s, center, rust_radius, &cfg, ws);
            drop(s);
            let meshes = outcome.meshes;
            let mined = outcome.mined;

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
                                                    max_flow_dist: 0, // non-source — irrelevant
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
                                    max_flow_dist: 0, // procedural breach — unbounded
                                });
                            }
                        }
                    }
                }
            }

            // Step 6: Regenerate seams
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, ws);
}

pub(super) fn handle_brush_sphere(ctx: &super::HandlerCtx<'_>, request: crate::types::FfiBrushSphereRequest) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let world_scale = ctx.world_scale;
    let fluid_event_tx = ctx.fluid_event_tx;
            let cfg = config.read().unwrap().clone();
            let center = from_ue_world_pos(
                request.world_x, request.world_y, request.world_z, world_scale,
            );
            let radius = request.radius / world_scale;

            let mut s = store.write().unwrap();
            let outcome = match request.mode {
                0 => {
                    let mat = voxel_core::material::Material::from_u8(request.material);
                    crate::brushes::paint_material_sphere(
                        &mut s, center, radius, mat, &cfg, world_scale,
                    )
                }
                1 => crate::brushes::carve_sphere(&mut s, center, radius, &cfg, world_scale),
                2 => {
                    let mat = voxel_core::material::Material::from_u8(request.material);
                    crate::brushes::fill_sphere(
                        &mut s, center, radius, mat, &cfg, world_scale,
                    )
                }
                _ => crate::brushes::BrushOutcome { meshes: Vec::new(), flipped_chunks: Vec::new() },
            };
            drop(s);

            let dirty_keys: Vec<(i32, i32, i32)> =
                outcome.meshes.into_iter().map(|(k, _)| k).collect();
            // Recompute crystal placements on flipped chunks (mirrors mine path).
            let new_placements: Vec<_> = {
                let s = store.read().unwrap();
                outcome.flipped_chunks.iter().filter_map(|&key| {
                    s.density_fields.get(&key).map(|density| {
                        let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                        (key, voxel_gen::compute_crystals(coord, density, &cfg))
                    })
                }).collect()
            };
            {
                let mut s = store.write().unwrap();
                for (key, placements) in new_placements {
                    s.crystal_placements.insert(key, placements);
                }
            }
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
}

pub(super) fn handle_brush_tunnel(ctx: &super::HandlerCtx<'_>, points: Vec<glam::Vec3>, radius: f32, material: Option<u8>) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let world_scale = ctx.world_scale;
    let fluid_event_tx = ctx.fluid_event_tx;
            let cfg = config.read().unwrap().clone();
            let mat = material.map(voxel_core::material::Material::from_u8);

            let mut s = store.write().unwrap();
            let outcome = crate::brushes::tunnel(
                &mut s, &points, radius, mat, &cfg, world_scale,
            );
            drop(s);

            let dirty_keys: Vec<(i32, i32, i32)> =
                outcome.meshes.into_iter().map(|(k, _)| k).collect();
            let new_placements: Vec<_> = {
                let s = store.read().unwrap();
                outcome.flipped_chunks.iter().filter_map(|&key| {
                    s.density_fields.get(&key).map(|density| {
                        let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                        (key, voxel_gen::compute_crystals(coord, density, &cfg))
                    })
                }).collect()
            };
            {
                let mut s = store.write().unwrap();
                for (key, placements) in new_placements {
                    s.crystal_placements.insert(key, placements);
                }
            }
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
}

pub(super) fn handle_brush_place_mushroom(ctx: &super::HandlerCtx<'_>, center_rust: glam::Vec3, kind: u8, search_radius: f32, scale: f32, yaw: f32) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let world_scale = ctx.world_scale;
    let fluid_event_tx = ctx.fluid_event_tx;
            let cfg = config.read().unwrap().clone();
            let mut s = store.write().unwrap();
            let placed_chunk = crate::brushes::place_mushroom_at_world(
                &mut s, center_rust, kind, search_radius, scale, yaw, &cfg,
            );
            drop(s);

            // No density change — but UE needs the new mushroom_data. Bypass
            // the hash-skip in `batched_seam_pass_inner` by clearing the
            // last-sent hash for that chunk, then trigger a seam pass on it.
            // The seam pass will re-emit the same combined mesh + the new
            // mushroom_data, and UE's ApplyMushroomData hash-skip lets the
            // crystal+mesh apply paths short-circuit.
            if let Some(chunk_key) = placed_chunk {
                {
                    let mut s = store.write().unwrap();
                    s.last_sent_mesh_hash.remove(&chunk_key);
                    s.modification_tracker.mark_dirty(chunk_key);
                }
                let dirty = [chunk_key];
                batched_seam_pass_mine(&dirty, &cfg, store, result_tx, fluid_event_tx, world_scale);
            }
}

pub(super) fn handle_brush_place_mushroom_sphere(ctx: &super::HandlerCtx<'_>, center_rust: glam::Vec3, radius: f32, density: f32, clustering: f32, kind: u8, seed: u64) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let world_scale = ctx.world_scale;
    let fluid_event_tx = ctx.fluid_event_tx;
            let cfg = config.read().unwrap().clone();
            let mut s = store.write().unwrap();
            let affected = crate::brushes::place_mushrooms_brush_sphere(
                &mut s, center_rust, kind, radius, density, clustering, seed, &cfg,
            );
            drop(s);

            if !affected.is_empty() {
                {
                    let mut s = store.write().unwrap();
                    for key in &affected {
                        s.last_sent_mesh_hash.remove(key);
                    }
                    // Mark dirty so collect_save_data picks up the new
                    // mushroom_placements for these chunks (mushroom edits
                    // don't otherwise touch the density dirty tracker).
                    let dirty_vec: Vec<(i32, i32, i32)> = affected.iter().copied().collect();
                    s.modification_tracker.mark_dirty_many(&dirty_vec);
                }
                let dirty: Vec<(i32, i32, i32)> = affected.into_iter().collect();
                batched_seam_pass_mine(&dirty, &cfg, store, result_tx, fluid_event_tx, world_scale);
            }
}

pub(super) fn handle_brush_erase_mushroom_sphere(ctx: &super::HandlerCtx<'_>, center_rust: glam::Vec3, radius: f32, kind_filter: u8) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let world_scale = ctx.world_scale;
    let fluid_event_tx = ctx.fluid_event_tx;
            let cfg = config.read().unwrap().clone();
            let mut s = store.write().unwrap();
            let affected = crate::brushes::erase_mushrooms_brush_sphere(
                &mut s, center_rust, kind_filter, radius, &cfg,
            );
            drop(s);

            if !affected.is_empty() {
                {
                    let mut s = store.write().unwrap();
                    for key in &affected {
                        s.last_sent_mesh_hash.remove(key);
                    }
                    let dirty_vec: Vec<(i32, i32, i32)> = affected.iter().copied().collect();
                    s.modification_tracker.mark_dirty_many(&dirty_vec);
                }
                let dirty: Vec<(i32, i32, i32)> = affected.into_iter().collect();
                batched_seam_pass_mine(&dirty, &cfg, store, result_tx, fluid_event_tx, world_scale);
            }
}

pub(super) fn handle_brush_formation(ctx: &super::HandlerCtx<'_>, center_rust: glam::Vec3, formation_type: u8, material: u8, height: f32, radius: f32) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let world_scale = ctx.world_scale;
    let fluid_event_tx = ctx.fluid_event_tx;
            let cfg = config.read().unwrap().clone();
            let mat = voxel_core::material::Material::from_u8(material);

            let mut s = store.write().unwrap();
            let outcome = crate::brushes::place_formation(
                &mut s, center_rust, formation_type, height, radius, mat, &cfg, world_scale,
            );
            drop(s);

            let dirty_keys: Vec<(i32, i32, i32)> =
                outcome.meshes.into_iter().map(|(k, _)| k).collect();
            let new_placements: Vec<_> = {
                let s = store.read().unwrap();
                outcome.flipped_chunks.iter().filter_map(|&key| {
                    s.density_fields.get(&key).map(|density| {
                        let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                        (key, voxel_gen::compute_crystals(coord, density, &cfg))
                    })
                }).collect()
            };
            {
                let mut s = store.write().unwrap();
                for (key, placements) in new_placements {
                    s.crystal_placements.insert(key, placements);
                }
            }
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
}

pub(super) fn handle_force_chunk_resync(ctx: &super::HandlerCtx<'_>, chunk: (i32, i32, i32)) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let world_scale = ctx.world_scale;
            let cfg = config.read().unwrap().clone();
            let neighbors_face: [(i32, i32, i32); 6] = [
                (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
            ];
            // Trace file so we can confirm the request reached the worker
            // and what the seam pass actually emitted. Lives next to other
            // worker logs the user already collects.
            let log_line = |line: String| {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true)
                    .open("D:/Unreal Projects/Mithril2026/Saved/voxel_force_resync.txt")
                {
                    let _ = writeln!(f, "[ForceResync] {}", line);
                }
            };
            log_line(format!("BEGIN chunk={:?}", chunk));

            // Phase 1: sync boundaries
            {
                let mut s = store.write().unwrap();
                let modified = s.sync_chunk_full_boundaries(chunk, cfg.chunk_size);
                log_line(format!("sync_chunk_full_boundaries → {} chunks modified", modified.len()));
            }

            // Phase 2: collect targets (chunk + 6 face neighbors that exist)
            let mut targets: Vec<(i32, i32, i32)> = vec![chunk];
            {
                let s = store.read().unwrap();
                for &(dx, dy, dz) in &neighbors_face {
                    let n = (chunk.0 + dx, chunk.1 + dy, chunk.2 + dz);
                    if s.density_fields.contains_key(&n) {
                        targets.push(n);
                    }
                }
            }
            log_line(format!("targets: {:?}", targets));

            // Phase 3: regenerate base mesh + hermite + seam_data for each
            // target.
            let chunk_size = cfg.chunk_size;
            for &target in &targets {
                let density_clone = {
                    let s = store.read().unwrap();
                    s.density_fields.get(&target).cloned()
                };
                let Some(density) = density_clone else { continue; };
                let hermite = extract_hermite_data(&density);
                let cell_size = density.size - 1;
                let dc_vertices = solve_dc_vertices(&hermite, cell_size);
                let mut mesh = generate_mesh(&hermite, &dc_vertices, cell_size);
                mesh.smooth(
                    cfg.mesh_smooth_iterations,
                    cfg.mesh_smooth_strength,
                    cfg.mesh_boundary_smooth,
                    Some(cell_size),
                );
                if cfg.mesh_recalc_normals > 0 { mesh.recalculate_normals(); }
                let boundary_edges = voxel_gen::region_gen::extract_boundary_edges(&hermite, chunk_size);
                let base_verts = mesh.vertices.len();
                {
                    let mut s = store.write().unwrap();
                    s.hermite_data.insert(target, hermite);
                    s.base_meshes.insert(target, std::sync::Arc::new(mesh));
                    s.add_seam_data(target, ChunkSeamData {
                        dc_vertices,
                        world_origin: glam::Vec3::ZERO,
                        boundary_edges,
                    });
                }
                log_line(format!("regenerated target {:?}: base verts={}", target, base_verts));
            }

            // Phase 4: combine base + seam quads and send DIRECTLY as
            // ChunkMesh results. Bypasses the MineBatchMesh requeue dance
            // which seems to drop seam updates for Force Resync. Also
            // forcibly invalidates last_sent_mesh_hash so hash-skip can't
            // suppress the send (the user explicitly asked for a refresh).
            for &target in &targets {
                let combined: Option<voxel_core::mesh::Mesh> = {
                    let s = store.read().unwrap();
                    let base = match s.base_meshes.get(&target) { Some(m) => (**m).clone(), None => { continue; } };
                    let seam = voxel_gen::region_gen::generate_chunk_seam_quads(target, &s.chunk_seam_data, cfg.chunk_size);
                    let seam_tris = seam.triangles.len();
                    let mut combined = base;
                    combined.append(seam);
                    if cfg.mesh_recalc_normals > 0 { combined.recalculate_normals(); }
                    log_line(format!("target {:?}: base+seam → {} verts / {} tris (seam_tris={})",
                        target, combined.vertices.len(), combined.triangles.len(), seam_tris));
                    Some(combined)
                };
                let Some(combined) = combined else { continue; };

                // Force-invalidate hash so send isn't skipped.
                {
                    let mut s = store.write().unwrap();
                    s.last_sent_mesh_hash.remove(&target);
                }

                let mut converted = convert_mesh_to_ue_scaled(&combined, cfg.voxel_scale(), world_scale);
                crate::convert::bucket_mesh_by_material(&mut converted);
                let v = converted.positions.len();
                let i = converted.indices.len();
                log_line(format!("target {:?}: sending ChunkMesh — {} verts / {} indices", target, v, i));
                let _ = result_tx.send(WorkerResult::ChunkMesh {
                    chunk: target,
                    mesh: converted,
                    generation: 0,
                    crystal_data: Vec::new(),
                    mushroom_data: Vec::new(),
                    zone_descriptors: Vec::new(),
                });

                // Record new hash so subsequent passes don't redo it
                let h = hash_mesh(&combined);
                {
                    let mut s = store.write().unwrap();
                    s.last_sent_mesh_hash.insert(target, h);
                }
            }
            log_line(format!("END chunk={:?}", chunk));
}

pub(super) fn handle_brush_cavern_stamp(ctx: &super::HandlerCtx<'_>, chunk_origin: (i32, i32, i32), extent: (u8, u8, u8), decorate: bool, fluids: bool, seed: u32) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let world_scale = ctx.world_scale;
    let fluid_event_tx = ctx.fluid_event_tx;
            let cfg = config.read().unwrap().clone();

            let mut s = store.write().unwrap();
            let outcome = crate::brushes::cavern_stamp_brush(
                &mut s, chunk_origin, extent, decorate, fluids, seed as u64, &cfg, world_scale,
            );
            drop(s);

            let dirty_keys: Vec<(i32, i32, i32)> =
                outcome.meshes.into_iter().map(|(k, _)| k).collect();
            let new_placements: Vec<_> = {
                let s = store.read().unwrap();
                outcome.flipped_chunks.iter().filter_map(|&key| {
                    s.density_fields.get(&key).map(|density| {
                        let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                        (key, voxel_gen::compute_crystals(coord, density, &cfg))
                    })
                }).collect()
            };
            {
                let mut s = store.write().unwrap();
                for (key, placements) in new_placements {
                    s.crystal_placements.insert(key, placements);
                }
            }
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
}

pub(super) fn handle_brush_formation_stamp(ctx: &super::HandlerCtx<'_>, center_rust: glam::Vec3, radius: f32, seed: u32) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let world_scale = ctx.world_scale;
    let fluid_event_tx = ctx.fluid_event_tx;
            let cfg = config.read().unwrap().clone();

            let mut s = store.write().unwrap();
            let outcome = crate::brushes::random_formations_brush(
                &mut s, center_rust, radius, seed as u64, &cfg, world_scale,
            );
            drop(s);

            let dirty_keys: Vec<(i32, i32, i32)> =
                outcome.meshes.into_iter().map(|(k, _)| k).collect();
            let new_placements: Vec<_> = {
                let s = store.read().unwrap();
                outcome.flipped_chunks.iter().filter_map(|&key| {
                    s.density_fields.get(&key).map(|density| {
                        let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                        (key, voxel_gen::compute_crystals(coord, density, &cfg))
                    })
                }).collect()
            };
            {
                let mut s = store.write().unwrap();
                for (key, placements) in new_placements {
                    s.crystal_placements.insert(key, placements);
                }
            }
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
}

pub(super) fn handle_brush_box(ctx: &super::HandlerCtx<'_>, center_rust: glam::Vec3, half_ext_rust: glam::Vec3, yaw_rad: f32, op: u8, material: u8) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let world_scale = ctx.world_scale;
    let fluid_event_tx = ctx.fluid_event_tx;
            let cfg = config.read().unwrap().clone();
            let mat = voxel_core::material::Material::from_u8(material);

            let mut s = store.write().unwrap();
            let outcome = crate::brushes::box_brush(
                &mut s, center_rust, half_ext_rust, yaw_rad, op, mat, &cfg, world_scale,
            );
            drop(s);

            let dirty_keys: Vec<(i32, i32, i32)> =
                outcome.meshes.into_iter().map(|(k, _)| k).collect();
            let new_placements: Vec<_> = {
                let s = store.read().unwrap();
                outcome.flipped_chunks.iter().filter_map(|&key| {
                    s.density_fields.get(&key).map(|density| {
                        let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                        (key, voxel_gen::compute_crystals(coord, density, &cfg))
                    })
                }).collect()
            };
            {
                let mut s = store.write().unwrap();
                for (key, placements) in new_placements {
                    s.crystal_placements.insert(key, placements);
                }
            }
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
}

pub(super) fn handle_brush_cylinder(ctx: &super::HandlerCtx<'_>, center_rust: glam::Vec3, radius: f32, height: f32, op: u8, material: u8) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let world_scale = ctx.world_scale;
    let fluid_event_tx = ctx.fluid_event_tx;
            let cfg = config.read().unwrap().clone();
            let mat = voxel_core::material::Material::from_u8(material);

            let mut s = store.write().unwrap();
            let outcome = crate::brushes::cylinder_brush(
                &mut s, center_rust, radius, height, op, mat, &cfg, world_scale,
            );
            drop(s);

            let dirty_keys: Vec<(i32, i32, i32)> =
                outcome.meshes.into_iter().map(|(k, _)| k).collect();
            let new_placements: Vec<_> = {
                let s = store.read().unwrap();
                outcome.flipped_chunks.iter().filter_map(|&key| {
                    s.density_fields.get(&key).map(|density| {
                        let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                        (key, voxel_gen::compute_crystals(coord, density, &cfg))
                    })
                }).collect()
            };
            {
                let mut s = store.write().unwrap();
                for (key, placements) in new_placements {
                    s.crystal_placements.insert(key, placements);
                }
            }
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
}

pub(super) fn handle_brush_smooth(ctx: &super::HandlerCtx<'_>, center_rust: glam::Vec3, radius: f32, iterations: u32, strength: f32) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let world_scale = ctx.world_scale;
    let fluid_event_tx = ctx.fluid_event_tx;
            let cfg = config.read().unwrap().clone();
            let mut s = store.write().unwrap();
            let outcome = crate::brushes::smooth_brush(
                &mut s, center_rust, radius, iterations, strength, &cfg, world_scale,
            );
            drop(s);
            let dirty_keys: Vec<(i32, i32, i32)> =
                outcome.meshes.into_iter().map(|(k, _)| k).collect();
            // Smooth doesn't change material, so crystal recompute isn't strictly needed —
            // but keep consistent with other brush handlers for safety.
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
}

pub(super) fn handle_brush_ore_paint(ctx: &super::HandlerCtx<'_>, center_rust: glam::Vec3, radius: f32, cluster_size: f32, min_spacing: f32, channel_prob: f32, channel_length: f32, channel_radius: f32, density: f32, seed: u32, weights: crate::brushes::OreWeights) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let world_scale = ctx.world_scale;
    let fluid_event_tx = ctx.fluid_event_tx;
            let cfg = config.read().unwrap().clone();
            let mut s = store.write().unwrap();
            let outcome = crate::brushes::paint_ore_deposits(
                &mut s,
                center_rust,
                radius,
                weights,
                cluster_size,
                min_spacing,
                channel_prob,
                channel_length,
                channel_radius,
                density,
                seed,
                &cfg,
                world_scale,
            );
            drop(s);

            let dirty_keys: Vec<(i32, i32, i32)> =
                outcome.meshes.into_iter().map(|(k, _)| k).collect();
            // Material flipped on solid voxels — recompute crystal placements
            // exactly like BrushSphere does so quartz/amethyst/crystal-y ores
            // get their geode visual contribution refreshed.
            let new_placements: Vec<_> = {
                let s = store.read().unwrap();
                outcome.flipped_chunks.iter().filter_map(|&key| {
                    s.density_fields.get(&key).map(|density_field| {
                        let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                        (key, voxel_gen::compute_crystals(coord, density_field, &cfg))
                    })
                }).collect()
            };
            {
                let mut s = store.write().unwrap();
                for (key, placements) in new_placements {
                    s.crystal_placements.insert(key, placements);
                }
            }
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
}

pub(super) fn handle_brush_paint_stress(ctx: &super::HandlerCtx<'_>, center_rust: glam::Vec3, radius: f32, amount: f32, cap: f32, op: u8, falloff: u8) {
    let store = ctx.store;
    let config = ctx.config;
    let world_scale = ctx.world_scale;
            let cfg = config.read().unwrap().clone();
            let mut s = store.write().unwrap();
            let _outcome = crate::brushes::paint_stress_sphere(
                &mut s, center_rust, radius, amount, falloff, op, cap, &cfg, world_scale,
            );
            // PaintStress does not change density/material → no remesh or seam
            // pass is needed. The UE side picks up the updated painted overlay
            // on the next `voxel_query_stress` call (and the V-key overlay
            // recalc preview already drives that path).
            drop(s);
}

pub(super) fn handle_brush_clear_all_painted_stress(ctx: &super::HandlerCtx<'_>) {
    let store = ctx.store;
            let mut s = store.write().unwrap();
            // Wipe every loaded chunk's painted overlay. Unloaded chunks with
            // preserved snapshots are NOT cleared — they retain whatever paint
            // was captured at unload. Hand-flagged in the Atelier UI as
            // "Clear All Painted" (loaded chunks only); if users need a true
            // world-wide wipe they'd have to walk near every chunk first.
            let touched: Vec<(i32, i32, i32)> = s
                .stress_fields
                .iter_mut()
                .filter_map(|(k, sf)| {
                    if sf.has_painted_layer() {
                        sf.clear_all_painted();
                        Some(*k)
                    } else {
                        None
                    }
                })
                .collect();
            if !touched.is_empty() {
                s.modification_tracker.mark_dirty_many(&touched);
            }
            drop(s);
}

pub(super) fn handle_brush_undo(ctx: &super::HandlerCtx<'_>) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let world_scale = ctx.world_scale;
    let fluid_event_tx = ctx.fluid_event_tx;
            let cfg = config.read().unwrap().clone();
            let mut s = store.write().unwrap();
            let outcome = crate::brushes::apply_undo(&mut s, &cfg, world_scale);
            drop(s);
            if let Some(outcome) = outcome {
                let dirty_keys: Vec<(i32, i32, i32)> =
                    outcome.meshes.into_iter().map(|(k, _)| k).collect();
                let new_placements: Vec<_> = {
                    let s = store.read().unwrap();
                    outcome.flipped_chunks.iter().filter_map(|&key| {
                        s.density_fields.get(&key).map(|density| {
                            let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                            (key, voxel_gen::compute_crystals(coord, density, &cfg))
                        })
                    }).collect()
                };
                {
                    let mut s = store.write().unwrap();
                    for (key, placements) in new_placements {
                        s.crystal_placements.insert(key, placements);
                    }
                }
                batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
            }
}

pub(super) fn handle_brush_noise(ctx: &super::HandlerCtx<'_>, center_rust: glam::Vec3, radius: f32, frequency: f32, strength: f32, seed: u32) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let world_scale = ctx.world_scale;
    let fluid_event_tx = ctx.fluid_event_tx;
            let cfg = config.read().unwrap().clone();
            let mut s = store.write().unwrap();
            let outcome = crate::brushes::noise_brush(
                &mut s, center_rust, radius, frequency, strength, seed, &cfg, world_scale,
            );
            drop(s);
            let dirty_keys: Vec<(i32, i32, i32)> =
                outcome.meshes.into_iter().map(|(k, _)| k).collect();
            let new_placements: Vec<_> = {
                let s = store.read().unwrap();
                outcome.flipped_chunks.iter().filter_map(|&key| {
                    s.density_fields.get(&key).map(|density| {
                        let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                        (key, voxel_gen::compute_crystals(coord, density, &cfg))
                    })
                }).collect()
            };
            {
                let mut s = store.write().unwrap();
                for (key, placements) in new_placements {
                    s.crystal_placements.insert(key, placements);
                }
            }
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
}

pub(super) fn handle_brush_fluid_sphere(ctx: &super::HandlerCtx<'_>, center_rust: glam::Vec3, radius: f32, fluid_type: u8, is_source: bool, op: u8, max_flow_dist: u8) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let world_scale = ctx.world_scale;
    let fluid_event_tx = ctx.fluid_event_tx;
            // op semantics:
            //   0 = fill air (no carve)
            //   1 = clear (level=0)
            //   2 = pool-dig (carve solid + bottom-half non-source — designer's "make a basin")
            //   3 = carve + full fill (carve solid + fill entire sphere; respects is_source)
            let cfg = config.read().unwrap().clone();
            let ft = voxel_fluid::cell::FluidType::from_u8(fluid_type);
            let level = if op == 1 { 0.0 } else { 1.0 };
            let does_carve = op == 2 || op == 3;
            let bottom_half_only = op == 2;
            let force_non_source = op == 2; // pool-dig body always drains

            // Carve first (if applicable).
            let mut dig_outcome = None;
            if does_carve {
                let mut s = store.write().unwrap();
                dig_outcome = Some(crate::brushes::carve_sphere(&mut s, center_rust, radius, &cfg, world_scale));
                drop(s);
            }

            let cells = {
                let s = store.read().unwrap();
                crate::brushes::collect_fluid_cells_in_sphere(&s, center_rust, radius, bottom_half_only, &cfg)
            };

            // Refresh the fluid thread's density cache for every chunk this
            // brush touches. The fluid thread keeps its own density cache and
            // AddFluid checks `cell_capacity > MIN_LEVEL` against it. The
            // cache only refreshes when the worker sends DensityUpdate (at
            // gen time) or TerrainModified (after density changes).
            //
            // This needs to fire BOTH when carving (cells just changed
            // solid→air, fluid thread's cache still says solid) AND when
            // painting without carving (the cache may be stale from any
            // prior carve / config change / save-load cycle that didn't
            // notify the fluid thread). Without this, AddFluid silently
            // rejects every event because cell_capacity reads 0.
            //
            // Driving the chunk list from `cells` (rather than just the
            // carve outcome) covers paint-only mode too — the worker has
            // already filtered the cell list to "store thinks this is
            // air", so refreshing those chunks brings the fluid thread's
            // cache into agreement before AddFluid lands.
            {
                use std::collections::HashSet;
                let mut chunks_to_refresh: HashSet<(i32, i32, i32)> = HashSet::new();
                for c in &cells {
                    chunks_to_refresh.insert(c.chunk);
                }
                if let Some(ref outcome) = dig_outcome {
                    for (key, _) in &outcome.meshes {
                        chunks_to_refresh.insert(*key);
                    }
                }
                let s = store.read().unwrap();
                for key in chunks_to_refresh {
                    if let Some(density) = s.density_fields.get(&key) {
                        let densities: Vec<f32> = density.samples.iter().map(|s| s.density).collect();
                        let _ = fluid_event_tx.send(FluidEvent::TerrainModified {
                            chunk: key,
                            densities,
                        });
                    }
                }
            }

            for cell in cells {
                let cell_is_source = if force_non_source { false } else { is_source };
                let _ = fluid_event_tx.send(FluidEvent::AddFluid {
                    chunk: cell.chunk,
                    x: cell.x, y: cell.y, z: cell.z,
                    fluid_type: ft,
                    level,
                    is_source: cell_is_source,
                    // Bounded-flow only meaningful on sources.
                    max_flow_dist: if cell_is_source { max_flow_dist } else { 0 },
                });
            }

            if let Some(outcome) = dig_outcome {
                let dirty_keys: Vec<(i32, i32, i32)> =
                    outcome.meshes.into_iter().map(|(k, _)| k).collect();
                if !dirty_keys.is_empty() {
                    batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
                }
            }
}

pub(super) fn handle_brush_fluid_box(ctx: &super::HandlerCtx<'_>, center_rust: glam::Vec3, half_ext_rust: glam::Vec3, fluid_type: u8, is_source: bool, op: u8, max_flow_dist: u8) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let world_scale = ctx.world_scale;
    let fluid_event_tx = ctx.fluid_event_tx;
            let cfg = config.read().unwrap().clone();
            let ft = voxel_fluid::cell::FluidType::from_u8(fluid_type);
            let level = if op == 1 { 0.0 } else { 1.0 };

            // Op 2 (carve+fill): carve the box region first, then fill all air cells inside.
            // Fluid box brush is unrotated for now — add yaw_rad to FfiBrushFluidBoxRequest if needed.
            let mut dig_outcome = None;
            if op == 2 {
                let mut s = store.write().unwrap();
                dig_outcome = Some(crate::brushes::box_brush(
                    &mut s, center_rust, half_ext_rust, /*yaw_rad*/0.0, 1, // op=1 (carve)
                    voxel_core::material::Material::Air, &cfg, world_scale));
                drop(s);
            }

            let cells = {
                let s = store.read().unwrap();
                crate::brushes::collect_fluid_cells_in_box(&s, center_rust, half_ext_rust, &cfg)
            };

            // Refresh fluid thread density cache for every chunk touched —
            // covers carve mode AND paint-only mode. See BrushFluidSphere
            // for full rationale.
            {
                use std::collections::HashSet;
                let mut chunks_to_refresh: HashSet<(i32, i32, i32)> = HashSet::new();
                for c in &cells {
                    chunks_to_refresh.insert(c.chunk);
                }
                if let Some(ref outcome) = dig_outcome {
                    for (key, _) in &outcome.meshes {
                        chunks_to_refresh.insert(*key);
                    }
                }
                let s = store.read().unwrap();
                for key in chunks_to_refresh {
                    if let Some(density) = s.density_fields.get(&key) {
                        let densities: Vec<f32> = density.samples.iter().map(|s| s.density).collect();
                        let _ = fluid_event_tx.send(FluidEvent::TerrainModified {
                            chunk: key,
                            densities,
                        });
                    }
                }
            }

            for cell in cells {
                let cell_is_source = if op == 2 { false } else { is_source };
                let _ = fluid_event_tx.send(FluidEvent::AddFluid {
                    chunk: cell.chunk,
                    x: cell.x, y: cell.y, z: cell.z,
                    fluid_type: ft,
                    level,
                    // Carve-and-fill semantics: the body becomes non-source so it drains/spreads.
                    is_source: cell_is_source,
                    max_flow_dist: if cell_is_source { max_flow_dist } else { 0 },
                });
            }
            if let Some(outcome) = dig_outcome {
                let dirty_keys: Vec<(i32, i32, i32)> =
                    outcome.meshes.into_iter().map(|(k, _)| k).collect();
                if !dirty_keys.is_empty() {
                    batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
                }
            }
}

pub(super) fn handle_brush_fluid_river(ctx: &super::HandlerCtx<'_>, points: Vec<glam::Vec3>, radius: f32, fluid_type: u8, is_source: bool, op: u8, max_flow_dist: u8) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let world_scale = ctx.world_scale;
    let fluid_event_tx = ctx.fluid_event_tx;
            let cfg = config.read().unwrap().clone();
            let ft = voxel_fluid::cell::FluidType::from_u8(fluid_type);

            // Op 2: carve a channel first using the existing tunnel brush, then fill it.
            let mut dig_outcome = None;
            if op == 2 {
                let mut s = store.write().unwrap();
                dig_outcome = Some(crate::brushes::tunnel(&mut s, &points, radius, None, &cfg, world_scale));
                drop(s);
            }

            let cells = {
                let s = store.read().unwrap();
                crate::brushes::collect_fluid_cells_in_capsule_chain(&s, &points, radius, &cfg)
            };

            // Refresh fluid thread density cache for every chunk touched —
            // covers carve mode AND paint-only mode. See BrushFluidSphere
            // for full rationale.
            {
                use std::collections::HashSet;
                let mut chunks_to_refresh: HashSet<(i32, i32, i32)> = HashSet::new();
                for c in &cells {
                    chunks_to_refresh.insert(c.chunk);
                }
                if let Some(ref outcome) = dig_outcome {
                    for (key, _) in &outcome.meshes {
                        chunks_to_refresh.insert(*key);
                    }
                }
                let s = store.read().unwrap();
                for key in chunks_to_refresh {
                    if let Some(density) = s.density_fields.get(&key) {
                        let densities: Vec<f32> = density.samples.iter().map(|s| s.density).collect();
                        let _ = fluid_event_tx.send(FluidEvent::TerrainModified {
                            chunk: key,
                            densities,
                        });
                    }
                }
            }
            for cell in cells {
                let _ = fluid_event_tx.send(FluidEvent::AddFluid {
                    chunk: cell.chunk,
                    x: cell.x, y: cell.y, z: cell.z,
                    fluid_type: ft,
                    level: 1.0,
                    is_source,
                    max_flow_dist: if is_source { max_flow_dist } else { 0 },
                });
            }
            if let Some(outcome) = dig_outcome {
                let dirty_keys: Vec<(i32, i32, i32)> =
                    outcome.meshes.into_iter().map(|(k, _)| k).collect();
                if !dirty_keys.is_empty() {
                    batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
                }
            }
}

pub(super) fn handle_apply_lava_quench(ctx: &super::HandlerCtx<'_>, obsidian: Vec<((i32, i32, i32), usize, usize, usize)>, scoria: Vec<((i32, i32, i32), usize, usize, usize)>, _drained_water: Vec<((i32, i32, i32), usize, usize, usize)>) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let world_scale = ctx.world_scale;
    let fluid_event_tx = ctx.fluid_event_tx;
            // Apply the live lava↔water quench plan emitted by the fluid sim.
            // - obsidian cells: Material::Obsidian (glassy quench skin, hardness 0.85)
            // - scoria cells:   Material::Scoria   (steam-altered halo, hardness 0.45)
            // The fluid grid already drained these cells locally; here we make
            // the underlying voxels solid + remesh + nudge the fluid sim's
            // density cache so it knows the cells are now solid.
            if obsidian.is_empty() && scoria.is_empty() {
                return;
            }
            let cfg = config.read().unwrap().clone();
            let cs = cfg.chunk_size as i32;

            let mut s = store.write().unwrap();
            let mut dirty_set: HashSet<(i32, i32, i32)> = HashSet::new();
            let mut written: Vec<voxel_core::density_ops::WrittenCell> = Vec::new();
            let mut changed: u32 = 0;

            // Helper: convert local (key, lx, ly, lz) → world coords + write
            // through density_ops::write_all_locations so chunk-boundary
            // mirrors stay in sync.
            for (key, lx, ly, lz) in &obsidian {
                let wx = key.0 * cs + *lx as i32;
                let wy = key.1 * cs + *ly as i32;
                let wz = key.2 * cs + *lz as i32;
                voxel_core::density_ops::write_all_locations(
                    &mut s.density_fields, cs, wx, wy, wz,
                    |_old_d, _old_m| Some((1.0, voxel_core::material::Material::Obsidian)),
                    &mut dirty_set, &mut written, &mut changed,
                );
            }
            for (key, lx, ly, lz) in &scoria {
                let wx = key.0 * cs + *lx as i32;
                let wy = key.1 * cs + *ly as i32;
                let wz = key.2 * cs + *lz as i32;
                voxel_core::density_ops::write_all_locations(
                    &mut s.density_fields, cs, wx, wy, wz,
                    |_old_d, old_m| {
                        // Don't overwrite obsidian we just placed at the same
                        // boundary position (rim wins over halo).
                        if old_m == voxel_core::material::Material::Obsidian {
                            None
                        } else {
                            Some((1.0, voxel_core::material::Material::Scoria))
                        }
                    },
                    &mut dirty_set, &mut written, &mut changed,
                );
            }

            // Persist these changes across save/load.
            let dirty_chunks: Vec<(i32, i32, i32)> = dirty_set.iter().copied().collect();
            s.modification_tracker.mark_dirty_many(&dirty_chunks);

            // Remesh affected chunks (full chunk bounds — we touched solid voxels).
            let dirty_bounds: Vec<_> = dirty_chunks.iter().map(|&k| {
                (k, 0usize, 0usize, 0usize, cfg.chunk_size, cfg.chunk_size, cfg.chunk_size)
            }).collect();
            let meshes = s.remesh_dirty(&dirty_bounds, &cfg, world_scale);

            // Capture density for the fluid TerrainModified events while we
            // still hold the lock, so the fluid sim's density cache is updated
            // and stops trying to flow into the new solid voxels.
            let mut terrain_updates: Vec<((i32,i32,i32), Vec<f32>)> = Vec::new();
            for &k in &dirty_chunks {
                if let Some(df) = s.density_fields.get(&k) {
                    let densities: Vec<f32> = df.samples.iter().map(|s| s.density).collect();
                    terrain_updates.push((k, densities));
                }
            }
            drop(s);

            // Ship remeshed chunks back to UE through the normal pipeline.
            for (chunk, mesh) in meshes {
                let _ = result_tx.send(WorkerResult::ChunkMesh {
                    chunk,
                    mesh,
                    generation: 0,
                    crystal_data: Vec::new(),
                    mushroom_data: Vec::new(),
                    zone_descriptors: Vec::new(),
                });
            }

            // Notify fluid sim of new solid voxels so its density cache
            // matches the world state.
            for (chunk, densities) in terrain_updates {
                let _ = fluid_event_tx.send(FluidEvent::TerrainModified {
                    chunk, densities,
                });
            }

            // Seam pass so chunk-boundary quads tween correctly between
            // the new solid (1.0) and adjacent air/density values.
            if !dirty_chunks.is_empty() {
                batched_seam_pass(&dirty_chunks, &cfg, store, result_tx, fluid_event_tx, world_scale);
            }
}

pub(super) fn handle_unload(ctx: &super::HandlerCtx<'_>, chunk: (i32, i32, i32)) {
    let store = ctx.store;
    let generation_counters = ctx.generation_counters;
    let fluid_event_tx = ctx.fluid_event_tx;
            let mut s = store.write().unwrap();
            s.unload(chunk);
            generation_counters.remove(&chunk);
            let _ = fluid_event_tx.send(FluidEvent::ChunkUnloaded { chunk });
}
