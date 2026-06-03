//! Sleep cycle, aureole-only preview, and morph-step request handlers.
//!
//! Pure code-movement out of the former monolithic `worker.rs`; each function
//! is one match-arm body of the old `handle_request`. Behavior is unchanged.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use rayon::prelude::*;
use voxel_core::dual_contouring::mesh_gen::generate_mesh;
use voxel_core::dual_contouring::solve::solve_dc_vertices;
use voxel_fluid::FluidEvent;
use voxel_gen::hermite_extract::extract_hermite_data;
use voxel_gen::region_gen::{self, ChunkSeamData};

use crate::convert::convert_mesh_to_ue_scaled;
use crate::types::{FfiCollapseEvent, WorkerResult};

use super::seam::{batched_seam_pass, retrieve_crystal_data, retrieve_mushroom_data};

/// Morph reveal mode. `false` (default) = CPU per-step reveal: the worker re-meshes
/// the morph block each step and the geometry itself animates, so the per-vertex
/// `reveal_t` dissolve attribute is NEVER read by UE — baking it is pure dead work.
/// `true` = GPU dissolve reveal: UE meshes once at the final state and the material
/// dissolves the mesh in as `MorphProgress` sweeps, which DOES consume `reveal_t`.
/// Set from the `r.Dormancy.GpuReveal` CVar via `voxel_set_morph_gpu_reveal`. Process-
/// global (one engine/morph at a time) so it needs no per-worker plumbing.
pub static MORPH_GPU_REVEAL: AtomicBool = AtomicBool::new(false);

pub(super) fn handle_sleep(ctx: &super::HandlerCtx<'_>, player_chunk: (i32, i32, i32), sleep_count: u32, sc: voxel_sleep::SleepConfig) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let world_scale = ctx.world_scale;
    let fluid_event_tx = ctx.fluid_event_tx;
    let crystal_anchors = ctx.crystal_anchors;
            let cfg = config.read().unwrap().clone();
            let sleep_config = sc;
            let t_worker_start = Instant::now();
            crate::panic_log::note(&format!("[SLEEP_TRACE] enter Sleep handler player_chunk=({},{},{})", player_chunk.0, player_chunk.1, player_chunk.2));

            // Request fluid snapshot for geological processes
            let (snap_tx, snap_rx) = crossbeam_channel::bounded(1);
            crate::panic_log::note("[SLEEP_TRACE] sending SnapshotRequest");
            let _ = fluid_event_tx.send(voxel_fluid::FluidEvent::SnapshotRequest { reply_tx: snap_tx });
            let mut fluid_snapshot = snap_rx.recv().unwrap_or_else(|_| voxel_fluid::FluidSnapshot::default());
            crate::panic_log::note(&format!("[SLEEP_TRACE] got fluid snapshot ({} chunks)", fluid_snapshot.chunks.len()));

            crate::panic_log::note("[SLEEP_TRACE] acquiring store write lock");
            let mut s = store.write().unwrap();
            crate::panic_log::note("[SLEEP_TRACE] store write lock acquired");

            // Crystal Growth Bridge — grow any pending pairs BEFORE the
            // geological-time pass, so the new voxels are part of world state
            // when the sleep cinematic snapshots chunks.
            {
                let mut anchor_mgr = crystal_anchors.lock().unwrap();
                crate::crystal_anchors::grow_pending_bridges(
                    &mut anchor_mgr,
                    &mut s,
                    &cfg,
                    world_scale,
                );
                if anchor_mgr.anchor_count() > 0 {
                    crate::panic_log::note(&format!(
                        "[SLEEP_TRACE] crystal anchors: {} total, {} pairs grown this cycle",
                        anchor_mgr.anchor_count(),
                        anchor_mgr.list_grown_pairs().len(),
                    ));
                }
            }

            // Use helper to get three simultaneous &mut borrows (borrow checker
            // cannot split borrows through method calls on the same struct).
            let (density_fields, stress_fields, support_fields) = s.sleep_fields_mut();
            crate::panic_log::note(&format!("[SLEEP_TRACE] entering execute_sleep ({} density chunks)", density_fields.len()));

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
            crate::panic_log::note(&format!("[SLEEP_TRACE] execute_sleep returned: chunks_changed={} dirty_chunks={} metamorphosed={}", sleep_result.chunks_changed, sleep_result.dirty_chunks.len(), sleep_result.voxels_metamorphosed));

            // POI ranking: NOT done here — the continuous tracker thread
            // (poi_tracker.rs) maintains a live score map that survives chunk
            // unload. UE queries top-K via voxel_request_list_top_pois which
            // sorts the tracker map + merges live bridges from the anchor
            // manager at query time.

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

            // Mark sleep-modified chunks for save persistence.
            // (Re-locking `store.write()` here would deadlock — the outer scope
            // already holds the write guard `s` from before execute_sleep.)
            s.modification_tracker.mark_dirty_many(&sleep_result.dirty_chunks);

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
            // ROOT FIX (2026-05-29): remesh_dirty returns BASE-only meshes (it
            // refreshes seam DATA in the store but does not append seam QUADS to
            // the returned mesh). Sending those directly leaves EVERY sleep-
            // touched chunk + neighbour SEAMLESS — the "seams vanish the moment
            // the montage begins, across the whole cave, and never recover" bug
            // (normal mining does a follow-up seam combine; the sleep path never
            // did). remesh_dirty already updated chunk_seam_data above, so here
            // we re-combine base + seam quads per chunk (same as the mining /
            // force-resync seam pass) before sending, so post-sleep chunks carry
            // their seams.
            let dirty_keys: Vec<(i32, i32, i32)> = meshes.iter().map(|(k, _)| *k).collect();
            let mut dbg_total_seam_tris = 0usize;
            let mut dbg_chunks_with_seams = 0usize;
            let mut dbg_chunks_no_seams = 0usize;
            eprintln!("[SLEEP_SEAM] base+seam remesh: {} chunks (from {} dirty + neighbors)", dirty_keys.len(), dirty_count);
            for chunk in dirty_keys {
                let (converted, dbg_seam_tris) = {
                    let s = store.read().unwrap();
                    let base = match s.base_meshes.get(&chunk) { Some(m) => m.clone(), None => continue };
                    let seam = region_gen::generate_chunk_seam_quads(chunk, &s.chunk_seam_data, cfg.chunk_size);
                    let st = seam.triangles.len();
                    let mut combined = base;
                    if !seam.triangles.is_empty() { combined.append(seam); }
                    if cfg.mesh_recalc_normals > 0 { combined.recalculate_normals(); }
                    let mut c = convert_mesh_to_ue_scaled(&combined, cfg.voxel_scale(), world_scale);
                    crate::convert::bucket_mesh_by_material(&mut c);
                    (c, st)
                };
                dbg_total_seam_tris += dbg_seam_tris;
                if dbg_seam_tris > 0 { dbg_chunks_with_seams += 1; } else if !converted.positions.is_empty() { dbg_chunks_no_seams += 1; }
                let crystal_data = retrieve_crystal_data(store, chunk, cfg.voxel_scale(), world_scale);
                let mushroom_data = retrieve_mushroom_data(store, chunk, cfg.voxel_scale(), world_scale);
                let _ = result_tx.send(WorkerResult::ChunkMesh {
                    chunk,
                    mesh: converted,
                    generation: 0, // Sleep remesh
                    crystal_data,
                    mushroom_data,
                    zone_descriptors: Vec::new(),
                });
            }
            // [SLEEP_SEAM DBG 2026-05-29] Verify the post-sleep remesh carries
            // seams. If chunks_with_seams is high and seams are STILL missing
            // in-world, the loss is downstream (UE apply); if it's ~0, the seam
            // generation itself is failing for the sleep region.
            eprintln!("[SLEEP_SEAM] sent base+seam: {} chunks with seams, {} solid chunks WITHOUT seams, {} total seam tris",
                dbg_chunks_with_seams, dbg_chunks_no_seams, dbg_total_seam_tris);
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
            batched_seam_pass(&sleep_result.dirty_chunks, &cfg, store, result_tx, fluid_event_tx, world_scale);
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
            crate::panic_log::note("[SLEEP_TRACE] cloning + compacting manifest");
            let mut compact_manifest = sleep_result.manifest.clone();
            compact_manifest.compact();
            crate::panic_log::note("[SLEEP_TRACE] serializing manifest to JSON");
            let manifest_json = compact_manifest.to_json().unwrap_or_default();
            crate::panic_log::note(&format!("[SLEEP_TRACE] manifest JSON serialized ({} bytes), sending SleepComplete", manifest_json.len()));

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
                surface_changed_cells: sleep_result.surface_changed_cells,
                surface_step_activity: sleep_result.surface_step_activity,
            });
}

pub(super) fn handle_aureole_only(ctx: &super::HandlerCtx<'_>, player_chunk: (i32, i32, i32), sc: voxel_sleep::SleepConfig) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let world_scale = ctx.world_scale;
    let fluid_event_tx = ctx.fluid_event_tx;
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
                let mushroom_data = retrieve_mushroom_data(store, chunk, cfg.voxel_scale(), world_scale);
                let _ = result_tx.send(WorkerResult::ChunkMesh {
                    chunk,
                    mesh,
                    generation: 0,
                    crystal_data,
                    mushroom_data,
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
                surface_changed_cells: sleep_result.surface_changed_cells,
                surface_step_activity: sleep_result.surface_step_activity,
            });
}

pub(super) fn handle_morph_step(ctx: &super::HandlerCtx<'_>, chunks: Vec<(i32, i32, i32)>, step: u32, total_steps: u32) {
    let result_tx = ctx.result_tx;
    let store = ctx.store;
    let config = ctx.config;
    let world_scale = ctx.world_scale;
    let morph_manifest = ctx.morph_manifest;
    let morph_snapshot = ctx.morph_snapshot;
            let cfg = config.read().unwrap().clone();
            // CHUNK-DBG: print on first step only to avoid log spam
            if step == 1 {
                eprintln!("[CHUNK-DBG] Rust MorphStep step={}/{} chunks.len={}", step, total_steps, chunks.len());
                for (i, c) in chunks.iter().enumerate() {
                    if i < 5 || i == chunks.len() / 2 || i >= chunks.len() - 3 {
                        eprintln!("[CHUNK-DBG]  morph[{}/{}] key=({},{},{})", i, chunks.len(), c.0, c.1, c.2);
                    }
                }
            }

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

            // R5 (morph-snapshot): refresh the per-play snapshot if the chunk set
            // changed, taking the store read lock ONCE here. Every subsequent step
            // for this play meshes from the snapshot and never touches the store
            // lock — which would otherwise stall behind a generation slow-path write
            // (150-380ms hold) and freeze the on-screen reveal.
            let mut snap = morph_snapshot.lock().unwrap();
            if snap.keys != chunks {
                snap.densities.clear();
                snap.neighbor_seams.clear();
                let chunks_set: std::collections::HashSet<(i32, i32, i32)> = chunks.iter().copied().collect();
                let s = store.read().unwrap();
                for &key in &chunks {
                    if let Some(d) = s.density_fields.get(&key) {
                        snap.densities.insert(key, d.clone());
                    }
                }
                for &c in &chunks {
                    for dx in -1..=1i32 {
                        for dy in -1..=1i32 {
                            for dz in -1..=1i32 {
                                let n = (c.0 + dx, c.1 + dy, c.2 + dz);
                                if chunks_set.contains(&n) { continue; }
                                if let Some(data) = s.chunk_seam_data.get(&n) {
                                    snap.neighbor_seams.insert(n, data.clone());
                                }
                            }
                        }
                    }
                }
                drop(s);
                snap.keys = chunks.clone();
            }
            // Out-of-block neighbor seams for Phase 3 (cloned once from the snapshot).
            let neighbor_seam_snapshot: std::collections::HashMap<(i32, i32, i32), ChunkSeamData> =
                snap.neighbor_seams.clone();

            // Phase 1: Clone active density fields and apply manifest interpolation
            let mut density_fields: Vec<Option<voxel_core::density::DensityField>> = Vec::with_capacity(chunks.len());
            for (i, &key) in chunks.iter().enumerate() {
                if !active[i] {
                    density_fields.push(None); // Skip — existing mesh preserved
                    continue;
                }
                match snap.densities.get(&key) {
                    Some(d) => {
                        let mut df = d.clone();
                        if let Some(delta) = manifest.chunk_deltas.get(&key) {
                            if delta.voxel_changes.is_empty() && delta.synthesize_growth {
                                // Synthesized "rise from air" animation — used by
                                // POI plays for chunks that weren't sleep-affected.
                                // No recorded per-voxel diff, so we animate every
                                // solid voxel from -1.0 density up to its current
                                // state.
                                //
                                // Spread mode:
                                //   - growth_sources non-empty → per-voxel spread =
                                //     min-distance to any source / max_dist. Bridges
                                //     pass 2 anchors so growth radiates inward; other
                                //     POIs pass 1 chunk-center source for a radial
                                //     reveal.
                                //   - growth_sources empty → fall back to y-axis
                                //     gradient ("rising from below").
                                let size = df.size;
                                let y_norm_denom = (size as f32 - 1.0).max(1.0);
                                let cs = cfg.chunk_size as f32;
                                let chunk_origin_x = key.0 as f32 * cs;
                                let chunk_origin_y = key.1 as f32 * cs;
                                let chunk_origin_z = key.2 as f32 * cs;
                                let use_sources = !delta.growth_sources.is_empty();
                                let inv_max_dist = if delta.growth_source_max_dist > 0.0 {
                                    1.0 / delta.growth_source_max_dist
                                } else {
                                    0.0
                                };

                                for z in 0..size {
                                    for y in 0..size {
                                        for x in 0..size {
                                            let sample = df.get_mut(x, y, z);
                                            let target = sample.density;
                                            if target <= 0.0 {
                                                continue;
                                            }
                                            let spread = if use_sources {
                                                let wx = chunk_origin_x + x as f32;
                                                let wy = chunk_origin_y + y as f32;
                                                let wz = chunk_origin_z + z as f32;
                                                let mut best = f32::MAX;
                                                for s in &delta.growth_sources {
                                                    let dx = wx - s.0;
                                                    let dy = wy - s.1;
                                                    let dz = wz - s.2;
                                                    let d2 = dx * dx + dy * dy + dz * dz;
                                                    if d2 < best {
                                                        best = d2;
                                                    }
                                                }
                                                (best.sqrt() * inv_max_dist).clamp(0.0, 1.0)
                                            } else {
                                                y as f32 / y_norm_denom
                                            };
                                            // Stretch t by 1.5 so the wave fully
                                            // covers all voxels by t=1.0 even with
                                            // the spread bias.
                                            let voxel_t = ((t * 1.5) - spread * 0.5)
                                                .clamp(0.0, 1.0);
                                            sample.density = -1.0 + (target + 1.0) * voxel_t;
                                        }
                                    }
                                }
                            } else {
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
                        }
                        density_fields.push(Some(df));
                    }
                    None => density_fields.push(None),
                }
            }
            drop(snap);

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
                            // Empty-chunk early-out: no sign-changing edge anywhere in the
                            // chunk (incl. the boundary faces toward neighbors) means there
                            // is NO surface here — fully air (ahead of the reveal front) or
                            // fully solid (behind it). The DC solve/mesh/smooth would just
                            // produce an empty mesh after two big per-cell memsets, so skip
                            // them. This is the SAFE form of front-only meshing: a chunk that
                            // touches air has boundary edges → is NOT empty → still meshes, so
                            // its seam quads are never dropped (no cracks/holes). Returning
                            // None matches the inactive-chunk path: UE's DrainPendingMorph
                            // preserves the prior mesh on VertexCount==0 (any stale geometry
                            // is interior/occluded and is torn down at montage end).
                            //
                            // SAFETY IS COUPLED to two invariants — if either changes this
                            // early-out can silently start dropping seams (holes):
                            //   1. extract_hermite_data walks the apron/boundary-face edges
                            //      (density `size` = chunk_size+1, the +face plane at size-1),
                            //      so a chunk touching air across ANY face is non-empty here.
                            //   2. region_gen::extract_boundary_edges uses +face-only seam
                            //      ownership, so a skipped (surfaceless) chunk is never the
                            //      missing `+` neighbor a meshed chunk needs.
                            if h.edges.is_empty() {
                                return None;
                            }
                            let cell_size = df.size - 1;
                            let dc_verts = solve_dc_vertices(&h, cell_size);
                            let mut mesh = generate_mesh(&h, &dc_verts, cell_size);
                            // Morph-only: cap smoothing at 1 iteration. The reveal mesh is
                            // transient (dissolving in), so a 2nd smoothing pass isn't
                            // perceptible mid-cinematic — worldgen meshing keeps cfg's value.
                            mesh.smooth(cfg.mesh_smooth_iterations.min(1), cfg.mesh_smooth_strength, cfg.mesh_boundary_smooth, Some(cell_size));
                            let boundary_edges = region_gen::extract_boundary_edges(&h, chunk_size);
                            Some((mesh, dc_verts, boundary_edges))
                        }
                        None => None,
                    }
                }).collect();

            // Collect results back into sequential structures (seam_data_map is not thread-safe).
            // Pre-populate with the global store's seam data for out-of-block neighbors so
            // edge chunks in the morph block can seam against their stored neighbors.
            // Without this, `generate_chunk_seam_quads` skips every quad that references
            // a chunk outside the morph block (returns `valid = false` on missing lookup)
            // — leaving visible seam gaps along the morph-block perimeter, the exact
            // behaviour the recent brush-side seam fix addressed. Brushes already pull
            // from `store.chunk_seam_data`; we mirror that here for the morph path.
            // Mid-morph the out-of-block DC vertices are the post-sleep snapshot (t=1),
            // so the seam will be slightly inconsistent until t→1 — acceptable trade-off
            // vs. fully missing quads. At t=1 it matches the dirty-chunks state exactly.
            // INTERIM (2026-05-29): full seams EVERY step (no holes). The boundary
            // geometry mostly doesn't morph, so these read as already-transformed —
            // but that beats see-through gaps. Out-of-block neighbours come from the
            // per-play snapshot (post-sleep t=1); in-block dc_verts are added below.
            // R5: out-of-block neighbor seams come from the per-play snapshot, so this
            // step takes NO store read lock — it can't stall behind a generation write.
            // `neighbor_seam_snapshot` already IS exactly the out-of-block perimeter
            // shell (the snapshot filtered in-block keys when it was built), so we MOVE
            // it straight into the seam map instead of re-iterating the 26-neighborhood
            // and per-entry-cloning every value again — that was a second full deep
            // clone of the whole shell (every ChunkSeamData's dc_vertices + edges Vecs)
            // every step. Behavior-identical: same keys, same data.
            let mut seam_data_map: std::collections::HashMap<(i32, i32, i32), ChunkSeamData> =
                neighbor_seam_snapshot;
            let mut base_meshes: Vec<Option<voxel_core::mesh::Mesh>> = Vec::with_capacity(chunks.len());

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

            // Phase 4: Generate seam quads and append to base meshes, then convert.
            //
            // GPU-reveal bake: per vertex we also compute reveal_t in [0,1] (0 = reveals
            // first, 1 = last) in the SAME Rust voxel-world space as the Phase-1 growth
            // (vertex world pos = key*chunk_size + local). The UE material dissolves the
            // mesh in as MorphProgress sweeps 0->1, so the CPU no longer re-meshes per
            // step. Synthesized-growth chunks bake the exact `spread` field (radial to
            // growth_sources, or a chunk-local rising y-gradient). Recorded voxel_changes
            // chunks bake reveal_t from PROXIMITY TO ACTUAL CHANGES: unchanged rock is
            // reveal_t=0 (present from the start, never dissolves), only changed voxels
            // (+1-voxel dilation to catch DC surface verts straddling them) reveal in
            // spread_distance order — so existing geometry doesn't vanish/pop, matching
            // the old per-step morph's behaviour for unchanged terrain.
            let cs_f = cfg.chunk_size as f32;
            let mut meshes = Vec::with_capacity(chunks.len());
            for (i, base_opt) in base_meshes.into_iter().enumerate() {
                match base_opt {
                    Some(mut mesh) => {
                        // INTERIM: full seams every step (no holes). Boundary
                        // geometry mostly doesn't morph so these read as
                        // already-transformed, but solid-with-seams beats gaps.
                        let seam_mesh = region_gen::generate_chunk_seam_quads(
                            chunks[i], &seam_data_map, chunk_size);
                        if !seam_mesh.triangles.is_empty() {
                            mesh.append(seam_mesh);
                        }
                        if cfg.mesh_recalc_normals > 0 { mesh.recalculate_normals(); }

                        let mut converted = convert_mesh_to_ue_scaled(&mesh, cfg.voxel_scale(), world_scale);

                        // Bake per-vertex reveal_t (aligned with converted.positions BEFORE
                        // bucketing — convert preserves vertex order; bucket reorders both).
                        // ONLY when the GPU dissolve reveal is active: in the default CPU
                        // per-step reveal the geometry itself animates and UE never reads
                        // reveal_t, so this whole bake (a per-vertex pass + a ~119 KB
                        // change_field alloc for recorded chunks) is dead work every step.
                        // When skipped, converted.reveal_t stays empty → bucket_mesh_by_material
                        // early-outs on it → the FFI ships a null reveal_t pointer → UE falls
                        // back to no dissolve (exactly the CPU path's expectation).
                        if MORPH_GPU_REVEAL.load(Ordering::Relaxed) {
                            let key = chunks[i];
                            let origin = glam::Vec3::new(key.0 as f32 * cs_f, key.1 as f32 * cs_f, key.2 as f32 * cs_f);
                            let delta = manifest.chunk_deltas.get(&key);
                            let synth = delta.map(|d| d.synthesize_growth).unwrap_or(false);
                            let sources: &[(f32, f32, f32)] = delta.map(|d| d.growth_sources.as_slice()).unwrap_or(&[]);
                            let inv_max = delta
                                .map(|d| if d.growth_source_max_dist > 0.0 { 1.0 / d.growth_source_max_dist } else { 0.0 })
                                .unwrap_or(0.0);

                            // Recorded-changes chunks: per-voxel reveal field so ONLY the
                            // changed voxels reveal (dilated 1 voxel); unchanged terrain
                            // stays reveal_t=0. -1.0 = "unchanged / present from start".
                            let fsize = cfg.chunk_size as usize + 1;
                            let mut change_field: Vec<f32> = Vec::new();
                            if !synth {
                                if let Some(d) = delta {
                                    change_field = vec![-1.0f32; fsize * fsize * fsize];
                                    for ch in &d.voxel_changes {
                                        let sd = ch.spread_distance.clamp(0.0, 1.0);
                                        for dz in -1..=1i32 { for dy in -1..=1i32 { for dx in -1..=1i32 {
                                            let x = ch.lx as i32 + dx;
                                            let y = ch.ly as i32 + dy;
                                            let z = ch.lz as i32 + dz;
                                            if x < 0 || y < 0 || z < 0
                                                || x >= fsize as i32 || y >= fsize as i32 || z >= fsize as i32 { continue; }
                                            let fi = (z as usize * fsize + y as usize) * fsize + x as usize;
                                            if change_field[fi] < 0.0 || sd < change_field[fi] { change_field[fi] = sd; }
                                        }}}
                                    }
                                }
                            }

                            let mut rt = Vec::with_capacity(mesh.vertices.len());
                            for v in &mesh.vertices {
                                let spread = if synth && !sources.is_empty() {
                                    // Radial: min distance to any growth source / max_dist.
                                    let wp = origin + v.position;
                                    let mut best = f32::MAX;
                                    for s in sources {
                                        let d = wp - glam::Vec3::new(s.0, s.1, s.2);
                                        let d2 = d.length_squared();
                                        if d2 < best { best = d2; }
                                    }
                                    (best.sqrt() * inv_max).clamp(0.0, 1.0)
                                } else if synth {
                                    // Rising y-gradient (chunk-local), mirrors Phase-1 fallback.
                                    (v.position.y / cs_f).clamp(0.0, 1.0)
                                } else if !change_field.is_empty() {
                                    // Recorded: reveal only near actual changes; unchanged = 0.
                                    let xi = (v.position.x.round() as i32).clamp(0, fsize as i32 - 1) as usize;
                                    let yi = (v.position.y.round() as i32).clamp(0, fsize as i32 - 1) as usize;
                                    let zi = (v.position.z.round() as i32).clamp(0, fsize as i32 - 1) as usize;
                                    let f = change_field[(zi * fsize + yi) * fsize + xi];
                                    if f >= 0.0 { f } else { 0.0 }
                                } else {
                                    0.0
                                };
                                rt.push(spread);
                            }
                            converted.reveal_t = rt;
                        }

                        crate::convert::bucket_mesh_by_material(&mut converted);
                        // CHUNK-DBG: log vertex bounding box for first step
                        if step == 1 && (i < 3 || i == chunks.len() / 2 || i >= chunks.len() - 2) {
                            let key = chunks[i];
                            let mut mn = (f32::MAX, f32::MAX, f32::MAX);
                            let mut mx = (f32::MIN, f32::MIN, f32::MIN);
                            for p in converted.positions.iter().take(50) {
                                if p.x < mn.0 { mn.0 = p.x; } if p.y < mn.1 { mn.1 = p.y; } if p.z < mn.2 { mn.2 = p.z; }
                                if p.x > mx.0 { mx.0 = p.x; } if p.y > mx.1 { mx.1 = p.y; } if p.z > mx.2 { mx.2 = p.z; }
                            }
                            eprintln!("[CHUNK-DBG]  meshed[{}/{}] key=({},{},{}) verts={} vbox=[({:.0},{:.0},{:.0})..({:.0},{:.0},{:.0})]",
                                i, chunks.len(), key.0, key.1, key.2, converted.positions.len(),
                                mn.0, mn.1, mn.2, mx.0, mx.1, mx.2);
                        }
                        meshes.push(converted);
                    }
                    None => {
                        meshes.push(crate::types::ConvertedMesh {
                            positions: Vec::new(),
                            normals: Vec::new(),
                            material_ids: Vec::new(),
                            indices: Vec::new(),
                            submeshes: Vec::new(),
                            reveal_t: Vec::new(),
                        });
                    }
                }
            }

            eprintln!("[MORPH] Step {}/{}: meshed {} chunks", step, total_steps, meshes.len());

            let _ = result_tx.send(WorkerResult::MorphMeshes {
                step, total_steps, meshes,
            });
}
