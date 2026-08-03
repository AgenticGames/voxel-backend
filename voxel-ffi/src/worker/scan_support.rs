//! Support placement/removal, world-scan diagnostics, and force-spawn-pool
//! request handlers.
//!
//! Pure code-movement out of the former monolithic `worker.rs`; each function
//! is one match-arm body of the old `handle_request`. Behavior is unchanged.

use voxel_core::stress::SupportType;
use voxel_fluid::FluidEvent;

use crate::convert::from_ue_world_pos;
use crate::types::WorkerResult;

use super::seam::{incremental_seam_pass, retrieve_crystal_data, retrieve_mushroom_data};

pub(super) fn handle_place_support(ctx: &super::HandlerCtx<'_>, world_x: i32, world_y: i32, world_z: i32, support_type: u8) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let stress_config = ctx.stress_config;
            let cfg = config.read().unwrap().clone();
            let stress_cfg = stress_config.read().unwrap().clone();
            // Defensive: if an old UE editor (pre-2026-05-26) sends a
            // legacy SupportType byte (Slate/Granite/Limestone at 1/2/3 or
            // Copper/Iron/Steel/Crystal at 4/5/6/7), remap to the new
            // lineup so the in-flight transition window can't crash. New
            // UE plugins send 1-5 directly and pass through.
            let st = SupportType::from_legacy_u8(support_type);

            let mut s = store.write().unwrap();
            let (success, _collapse_events, _dirty_bounds) = s.place_support(
                (world_x, world_y, world_z), st, &stress_cfg, cfg.chunk_size,
            );
            drop(s);

            // No remesh: a strut only writes the support field (stress model) —
            // the density is untouched, so remesh_dirty produced an identical
            // base-only mesh that never reached UE anyway (leaked in
            // voxel_poll_result) and would wipe seams if it had. Stress relief
            // lands via the queue_stress_dirty call inside place_support.
            let _ = result_tx.send(WorkerResult::SupportResult { success });
}

pub(super) fn handle_remove_support(ctx: &super::HandlerCtx<'_>, world_x: i32, world_y: i32, world_z: i32) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let stress_config = ctx.stress_config;
            let cfg = config.read().unwrap().clone();
            let stress_cfg = stress_config.read().unwrap().clone();

            let mut s = store.write().unwrap();
            let (removed, _collapse_events, _dirty_bounds) = s.remove_support(
                (world_x, world_y, world_z), &stress_cfg, cfg.chunk_size,
            );
            drop(s);

            // No remesh — same reasoning as handle_place_support above.
            let _ = result_tx.send(WorkerResult::SupportResult {
                success: removed.is_some(),
            });
}

pub(super) fn handle_world_scan(ctx: &super::HandlerCtx<'_>) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
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

            // scan_world takes plain `Mesh` maps; materialize one from the
            // Arc-wrapped store cache. Deep clone is fine here — world scan is
            // a rare manual diagnostic, not a hot path.
            let scan_base_meshes: std::collections::HashMap<(i32, i32, i32), voxel_core::mesh::Mesh> =
                s.base_meshes.iter().map(|(&k, v)| (k, (**v).clone())).collect();

            let result = voxel_core::world_scan::scan_world(
                &s.density_fields,
                &scan_base_meshes,
                &scan_seam_data,
                &scan_worm_paths,
                cfg.chunk_size,
            );

            let json = result.to_json_string();
            drop(s);

            let _ = result_tx.send(WorkerResult::ScanComplete { json_report: json });
}

pub(super) fn handle_world_scan_with_config(ctx: &super::HandlerCtx<'_>, scan_config: voxel_core::world_scan::ScanConfig) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
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

            // Plain-Mesh map for scan_world_with_config (see handle_world_scan).
            let scan_base_meshes: std::collections::HashMap<(i32, i32, i32), voxel_core::mesh::Mesh> =
                s.base_meshes.iter().map(|(&k, v)| (k, (**v).clone())).collect();

            let result = voxel_core::world_scan::scan_world_with_config(
                &s.density_fields,
                &scan_base_meshes,
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

pub(super) fn handle_force_spawn_pool(ctx: &super::HandlerCtx<'_>, world_x: f32, world_y: f32, world_z: f32, fluid_type: u8) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let world_scale = ctx.world_scale;
    let fluid_event_tx = ctx.fluid_event_tx;
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
                    let mushroom_data = retrieve_mushroom_data(store, mkey, cfg.voxel_scale(), world_scale);
                    let _ = result_tx.send(WorkerResult::ChunkMesh {
                        chunk: mkey,
                        mesh,
                        generation: 0,
                        crystal_data,
                        mushroom_data,
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
                    max_flow_dist: 0, // procedural pool seed — unbounded
                });
            }

            // Run incremental seam pass
            let _ = incremental_seam_pass(key, &cfg, store, result_tx, world_scale);

            // Send diagnostics as JSON
            let json = serde_json::to_string(&diagnostics).unwrap_or_else(|_| "{}".to_string());
            let _ = result_tx.send(WorkerResult::ForceSpawnPoolComplete { json_report: json });
}
