//! Sleep cycle, aureole-only preview, and morph-step request handlers.
//!
//! Pure code-movement out of the former monolithic `worker.rs`; each function
//! is one match-arm body of the old `handle_request`. Behavior is unchanged.

use std::time::{Duration, Instant};

use rayon::prelude::*;
use voxel_core::dual_contouring::mesh_gen::{compute_cell_normals, generate_mesh};
use voxel_core::dual_contouring::solve::solve_dc_vertices;
use voxel_fluid::FluidEvent;
use voxel_gen::hermite_extract::extract_hermite_data;
use voxel_gen::region_gen::{self, ChunkSeamData};

use crate::convert::convert_mesh_to_ue_scaled;
use crate::types::{FfiCollapseEvent, WorkerResult};

use super::seam::{batched_seam_pass, retrieve_crystal_data, retrieve_mushroom_data};

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
                    let base = match s.base_meshes.get(&chunk) { Some(m) => (**m).clone(), None => continue };
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
    // Entry note (2026-08-05): run-5 saw a morph request dequeue and produce
    // NOTHING (no result, no log, worker back to idle) — UE timed the step out
    // at 4s. This line pins dequeue-vs-handler when it recurs.
    crate::panic_log::note(&format!(
        "[MORPH-REQ] step {}/{} dequeued by worker {} ({} chunks)",
        step, total_steps, ctx.worker_id, chunks.len()));
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
                    if let Ok(mut mq) = ctx.morph_results.lock() {
                        mq.push_back(crate::engine::MorphStepResult {
                            step, total_steps, meshes: Vec::new(),
                        });
                    }
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
                // Recolor fast-path: reset the per-play base cache and classify
                // whether the WHOLE block is a pure recolor (no chunk moves the DC
                // surface). If so, we mesh once and recolor per step (see below).
                snap.base_meshes.clear();
                snap.vertex_change.clear();
                snap.base_reveal_t.clear();
                snap.last_materials.clear();
                snap.base_built = false;
                // Classify the block. The fast path is gated ONLY on the absence of
                // synthesized growth (the POI "rise from air" genuinely animates geometry
                // from -1.0 → solid and must keep the full per-step DC path). Sparse
                // sign-flip changes (water/acid EROSION, solid↔air) are TOLERATED: they're
                // excluded from the recolor (see the capture below) so they freeze at the
                // pre-reveal state and the montage-end ForceChunkResync opens them — the
                // bulk metamorphism/hydrothermal recolor still meshes once + recolors.
                // (Block-level "every chunk pure_recolor" was too brittle: one erosion
                // voxel anywhere disqualified the whole 32-chunk aureole block.)
                let mut synth_count = 0u32;
                let mut signflip_count = 0u32;
                let mut norecord_count = 0u32;
                for key in &chunks {
                    match manifest.chunk_deltas.get(key) {
                        None => norecord_count += 1,
                        Some(d) => {
                            if d.synthesize_growth {
                                synth_count += 1;
                            } else if d.voxel_changes.iter().any(|c| (c.old_density > 0.0) != (c.new_density > 0.0)) {
                                signflip_count += 1;
                            }
                        }
                    }
                }
                snap.block_recolor = synth_count == 0;
                crate::panic_log::note(&format!(
                    "[MORPH-CLASS] chunks={} block_recolor={} synth={} signflip={} norecord={}",
                    chunks.len(), snap.block_recolor, synth_count, signflip_count, norecord_count));
                // ── Fidelity census (2026-08-05 playtest: montage-end != reality;
                // player-made Scoria absent from every displayed frame). One run of
                // these lines answers: is the material in the live snapshot, what
                // did the manifest record about it, and do the sleep's most-changed
                // chunks sit OUTSIDE the filmed block entirely.
                {
                    use voxel_core::material::Material;
                    let mut snap_scoria = 0usize;
                    let mut snap_chunks_with = 0usize;
                    for d in snap.densities.values() {
                        let c = d.samples.iter().filter(|s| s.material == Material::Scoria).count();
                        if c > 0 { snap_chunks_with += 1; }
                        snap_scoria += c;
                    }
                    let mut old_scoria = 0usize;
                    let mut new_scoria = 0usize;
                    let mut scoria_to: std::collections::BTreeMap<u8, usize> = Default::default();
                    for d in manifest.chunk_deltas.values() {
                        for c in &d.voxel_changes {
                            if c.old_material == Material::Scoria as u8 {
                                old_scoria += 1;
                                *scoria_to.entry(c.new_material).or_insert(0) += 1;
                            }
                            if c.new_material == Material::Scoria as u8 { new_scoria += 1; }
                        }
                    }
                    crate::panic_log::note(&format!(
                        "[MORPH-CENSUS] snapshot scoria_voxels={} in {}/{} block chunks; manifest old=Scoria changes={} (to {:?}) new=Scoria={}",
                        snap_scoria, snap_chunks_with, chunks.len(), old_scoria, scoria_to, new_scoria));
                    // Block coverage: the sleep's most-changed chunks vs the filmed block.
                    let mut counts: Vec<((i32, i32, i32), usize)> = manifest.chunk_deltas.iter()
                        .map(|(k, d)| (*k, d.voxel_changes.len())).collect();
                    counts.sort_by(|a, b| b.1.cmp(&a.1));
                    let top: Vec<String> = counts.iter().take(10)
                        .map(|(k, n)| format!("({},{},{}):{}{}", k.0, k.1, k.2, n,
                            if chunks_set.contains(k) { "" } else { "|OUT" }))
                        .collect();
                    let changed_total = counts.iter().filter(|(_, n)| *n > 0).count();
                    let changed_in_block = counts.iter().filter(|(k, n)| *n > 0 && chunks_set.contains(k)).count();
                    crate::panic_log::note(&format!(
                        "[MORPH-CENSUS] manifest chunks_with_changes={} (in filmed block: {}); top10 by changes: {}",
                        changed_total, changed_in_block, top.join(" ")));
                }
            }
            // Out-of-block neighbor seams for Phase 3 (cloned once from the snapshot).
            // Arc-valued since the store maps went Arc (f15ae6f) — cloning the
            // snapshot map is refcount bumps, not ~316KB/chunk deep copies.
            let neighbor_seam_snapshot: std::collections::HashMap<(i32, i32, i32), std::sync::Arc<ChunkSeamData>> =
                snap.neighbor_seams.clone();

            // ── Recolor fast path ────────────────────────────────────────────
            // Pure-recolor block whose base meshes are cached: skip ALL dual-
            // contouring + seam work. Each step only reassigns per-vertex material
            // by the reveal progress and re-buckets. The FIRST step of a recolor
            // play falls through to the full pipeline below (which builds + caches
            // the base via `building_base`); every step after returns here.
            let block_recolor = snap.block_recolor;
            let building_base = block_recolor && !snap.base_built;
            if block_recolor && snap.base_built {
                let recolor_t0 = std::time::Instant::now();
                let force_all = step >= total_steps; // final step ships every chunk
                let (meshes, shipped) =
                    recolor_cached_meshes(&mut snap, &chunks, &cfg, world_scale, t, force_all);
                drop(snap);
                drop(manifest_guard);
                // panic_log, not eprintln — stderr is invisible under a GUI app.
                crate::panic_log::note(&format!(
                    "[MORPH-STEP] {}/{} recolored {} chunks ({} shipped, rest unchanged) in {}ms (direct push)",
                    step, total_steps, meshes.len(), shipped, recolor_t0.elapsed().as_millis()));
                // Census: the FINAL displayed frame's material truth (t=1).
                if step >= total_steps {
                    use voxel_core::material::Material;
                    let sc: usize = meshes.iter()
                        .map(|m| m.material_ids.iter().filter(|&&id| id == Material::Scoria as u8).count())
                        .sum();
                    let verts: usize = meshes.iter().map(|m| m.material_ids.len()).sum();
                    crate::panic_log::note(&format!(
                        "[MORPH-CENSUS] final step: verts={} scoria_verts={}", verts, sc));
                }
                // Direct push (2026-08-05): result_tx routed morph steps through
                // the MAIN result queue, where they sat behind fluid/gen results
                // drained under UE's throttled reveal budgets (~0.5-1s extra
                // latency per step — the reveal ran 15.8s against a 5.95s arc).
                if let Ok(mut mq) = ctx.morph_results.lock() {
                    mq.push_back(crate::engine::MorphStepResult { step, total_steps, meshes });
                }
                return;
            }

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
            let mesh_results: Vec<Option<(voxel_core::mesh::Mesh, Vec<glam::Vec3>, Vec<glam::Vec3>, BEdges)>> =
                density_fields.par_iter().map(|df_opt| {
                    match df_opt {
                        Some(df) => {
                            let h = extract_hermite_data(df);
                            let cell_size = df.size - 1;
                            let dc_verts = solve_dc_vertices(&h, cell_size);
                            let mut mesh = generate_mesh(&h, &dc_verts, cell_size);
                            mesh.smooth(cfg.mesh_smooth_iterations, cfg.mesh_smooth_strength, cfg.mesh_boundary_smooth, Some(cell_size));
                            let boundary_edges = region_gen::extract_boundary_edges(&h, chunk_size);
                            let d_normals = compute_cell_normals(&h, cell_size);
                            Some((mesh, dc_verts, d_normals, boundary_edges))
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
            let chunks_set: std::collections::HashSet<(i32, i32, i32)> =
                chunks.iter().copied().collect();
            let mut seam_data_map: std::collections::HashMap<(i32, i32, i32), std::sync::Arc<ChunkSeamData>> = {
                // INTERIM (2026-05-29): full seams EVERY step (no holes). The
                // boundary geometry mostly doesn't morph, so these read as
                // already-transformed — but that beats see-through gaps. Proper
                // reveal-timed seams pending diagnosis of the morph/seam path.
                // Out-of-block neighbours come from the store (post-sleep t=1);
                // in-block dc_verts are added below.
                // R5: out-of-block neighbor seams come from the per-play snapshot
                // (cloned once above), so this step takes NO store read lock — it
                // can't stall behind a generation write. Behavior-preserving: the
                // morph already treats out-of-block seams as the post-sleep t=1 state.
                let mut map = std::collections::HashMap::new();
                for &c in &chunks {
                    for dx in -1..=1i32 {
                        for dy in -1..=1i32 {
                            for dz in -1..=1i32 {
                                let n = (c.0 + dx, c.1 + dy, c.2 + dz);
                                if chunks_set.contains(&n) { continue; }
                                if let Some(data) = neighbor_seam_snapshot.get(&n) {
                                    // Arc clone — cheap snapshot, no deep copy.
                                    map.insert(n, std::sync::Arc::clone(data));
                                }
                            }
                        }
                    }
                }
                map
            };
            let mut base_meshes: Vec<Option<voxel_core::mesh::Mesh>> = Vec::with_capacity(chunks.len());

            for (i, result) in mesh_results.into_iter().enumerate() {
                match result {
                    Some((mesh, dc_verts, d_normals, boundary_edges)) => {
                        seam_data_map.insert(chunks[i], std::sync::Arc::new(ChunkSeamData {
                            dc_vertices: dc_verts,
                            dc_normals: d_normals,
                            world_origin: glam::Vec3::ZERO,
                            boundary_edges,
                        }));
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
            // When building the recolor base this step, capture per-chunk
            // (base mesh, vertex→voxel lookup, reveal_t) to store into the
            // snapshot after the loop so every later step can recolor from cache.
            type BaseCapture = ((i32, i32, i32), voxel_core::mesh::Mesh, Vec<Option<(f32, u8, u8)>>, Vec<f32>);
            let mut base_capture: Vec<BaseCapture> = Vec::new();
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
                        {
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

                        // Capture the base for the recolor fast path (pre-bucket:
                        // mesh + reveal_t are still aligned with mesh.vertices order).
                        if building_base {
                            let fsize_l = cfg.chunk_size as usize + 1;
                            // ±1-dilated per-voxel change field (spread, old_mat, new_mat),
                            // identical dilation + closest-change-wins rule as the reveal_t
                            // change_field above — so a surface vertex whose nearest cell
                            // corner is unchanged but which straddles a changed voxel still
                            // recolors (matching the dissolve, no stale rim at recolor/air
                            // boundaries). None = unchanged neighborhood.
                            let mut field: Vec<Option<(f32, u8, u8)>> = vec![None; fsize_l * fsize_l * fsize_l];
                            if let Some(d) = manifest.chunk_deltas.get(&chunks[i]) {
                                for ch in &d.voxel_changes {
                                    // Sign-flip (erosion solid↔air) voxels are NOT recolored —
                                    // their geometry is frozen at the base state, so flipping
                                    // their material (e.g. to Air) would paint an air-material
                                    // patch on a still-solid surface. Skip → keep base material.
                                    if (ch.old_density > 0.0) != (ch.new_density > 0.0) { continue; }
                                    let sd = ch.spread_distance.clamp(0.0, 1.0);
                                    for dz in -1..=1i32 { for dy in -1..=1i32 { for dx in -1..=1i32 {
                                        let x = ch.lx as i32 + dx;
                                        let y = ch.ly as i32 + dy;
                                        let z = ch.lz as i32 + dz;
                                        if x < 0 || y < 0 || z < 0
                                            || x >= fsize_l as i32 || y >= fsize_l as i32 || z >= fsize_l as i32 { continue; }
                                        let fi = (z as usize * fsize_l + y as usize) * fsize_l + x as usize;
                                        let better = match field[fi] { Some((s, _, _)) => sd < s, None => true };
                                        if better { field[fi] = Some((sd, ch.old_material, ch.new_material)); }
                                    }}}
                                }
                            }
                            let vchange: Vec<Option<(f32, u8, u8)>> = mesh.vertices.iter().map(|v| {
                                let xi = (v.position.x.round() as i32).clamp(0, fsize_l as i32 - 1) as usize;
                                let yi = (v.position.y.round() as i32).clamp(0, fsize_l as i32 - 1) as usize;
                                let zi = (v.position.z.round() as i32).clamp(0, fsize_l as i32 - 1) as usize;
                                field[(zi * fsize_l + yi) * fsize_l + xi]
                            }).collect();
                            base_capture.push((chunks[i], mesh.clone(), vchange, converted.reveal_t.clone()));
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

            // Store the captured base so every subsequent step of this recolor play
            // takes the fast path. Re-lock the snapshot (it was dropped at line ~587
            // before Phase 3); guard on keys still matching (single morph at a time).
            if building_base {
                let mut snap2 = morph_snapshot.lock().unwrap();
                if snap2.keys == chunks {
                    for (k, m, vc, rt) in base_capture {
                        // Diff-skip seed: the base paint is the last shipped state.
                        snap2.last_materials.insert(k, m.vertices.iter().map(|v| v.material as u8).collect());
                        snap2.base_meshes.insert(k, m);
                        snap2.vertex_change.insert(k, vc);
                        snap2.base_reveal_t.insert(k, rt);
                    }
                    snap2.base_built = true;
                    eprintln!("[MORPH] Recolor base cached for {} chunks — later steps skip DC", snap2.base_meshes.len());
                    crate::panic_log::note(&format!(
                        "[MORPH-FASTPATH] base cached for {} chunks at step {}/{} — later steps recolor-only (no DC)",
                        snap2.base_meshes.len(), step, total_steps));
                    // Census: does the DISPLAYED base carry the live-store materials?
                    {
                        use voxel_core::material::Material;
                        let mut base_scoria = 0usize;
                        let mut verts = 0usize;
                        let mut covered = 0usize;
                        for m in snap2.base_meshes.values() {
                            base_scoria += m.vertices.iter().filter(|v| v.material == Material::Scoria).count();
                            verts += m.vertices.len();
                        }
                        for vc in snap2.vertex_change.values() {
                            covered += vc.iter().filter(|e| e.is_some()).count();
                        }
                        crate::panic_log::note(&format!(
                            "[MORPH-CENSUS] base meshes: verts={} scoria_verts={} vchange_covered={} ({:.1}%) at build t={:.2}",
                            verts, base_scoria, covered,
                            if verts > 0 { covered as f32 * 100.0 / verts as f32 } else { 0.0 },
                            t));
                    }
                }
            }

            if let Ok(mut mq) = ctx.morph_results.lock() {
                mq.push_back(crate::engine::MorphStepResult { step, total_steps, meshes });
            }
}

/// Recolor fast path: rebuild each chunk's ConvertedMesh from the cached base
/// (meshed ONCE) by reassigning per-vertex material to the reveal progress `t`,
/// then converting + bucketing. NO dual-contouring, NO seam-gen — geometry is
/// frozen (the block is a pure recolor), only material IDs (and thus the bucketed
/// submeshes/index order) change. Mirrors the recorded-change material rule used
/// by Phase 1 (`material = new if voxel_t>=0.5 else old`, staggered by spread).
fn recolor_cached_meshes(
    snap: &mut super::MorphSnapshot,
    chunks: &[(i32, i32, i32)],
    cfg: &voxel_gen::config::GenerationConfig,
    world_scale: f32,
    t: f32,
    force_all: bool,
) -> (Vec<crate::types::ConvertedMesh>, usize) {
    use rayon::prelude::*;
    // Shared reborrow for the parallel read phase; `snap` is only mutated in
    // the sequential post-pass below (last_materials updates).
    let snap_ref: &super::MorphSnapshot = &*snap;
    // Parallel per chunk (2026-08-05): each chunk's clone→recolor→convert→
    // bucket is pure and independent, and the sequential loop WAS the reveal's
    // pacing ceiling (~0.8-1.4s/step for ~40 chunks against the 350ms step
    // budget — the whole montage overran its camera arc ~2.7x). par_iter's
    // indexed collect preserves input order, so the UE index-pair contract
    // (Meshes[i] ↔ ShowcaseChunkActors[i]) is untouched.
    //
    // Diff-skip: a chunk whose per-vertex material assignment is IDENTICAL to
    // the last shipped step returns an empty mesh (UE preserves the actor's
    // prior sections on VertexCount==0), so only the recolor FRONT pays the
    // convert+bucket and — far more importantly — UE's per-section rebuild.
    let empty = || crate::types::ConvertedMesh {
        positions: Vec::new(), normals: Vec::new(), material_ids: Vec::new(),
        indices: Vec::new(), submeshes: Vec::new(), reveal_t: Vec::new(),
    };
    let results: Vec<(crate::types::ConvertedMesh, Option<Vec<u8>>)> = chunks
        .par_iter()
        .map(|&key| {
            let base = match snap_ref.base_meshes.get(&key) {
                Some(m) => m,
                None => return (empty(), None), // no surface at base build
            };
            // This step's material assignment, in base-vertex order.
            let mut mats: Vec<u8> = base.vertices.iter().map(|v| v.material as u8).collect();
            if let Some(vchange) = snap_ref.vertex_change.get(&key) {
                for (vi, m) in mats.iter_mut().enumerate() {
                    if let Some(Some((spread, old_m, new_m))) = vchange.get(vi).copied() {
                        let voxel_delay = spread * 0.6;
                        let voxel_t = ((t - voxel_delay) / (1.0 - voxel_delay)).clamp(0.0, 1.0);
                        *m = if voxel_t >= 0.5 { new_m } else { old_m };
                    }
                }
            }
            if !force_all {
                if let Some(prev) = snap_ref.last_materials.get(&key) {
                    if *prev == mats {
                        return (empty(), None); // unchanged since last shipped step
                    }
                }
            }
            // Clone the frozen base geometry and stamp this step's materials.
            let mut mesh = base.clone();
            for (vi, v) in mesh.vertices.iter_mut().enumerate() {
                v.material = voxel_core::material::Material::from_u8(mats[vi]);
            }
            let mut converted = convert_mesh_to_ue_scaled(&mesh, cfg.voxel_scale(), world_scale);
            // Re-attach the frozen (spread-only) reveal_t baked at base build.
            if let Some(rt) = snap_ref.base_reveal_t.get(&key) {
                if rt.len() == converted.positions.len() {
                    converted.reveal_t = rt.clone();
                }
            }
            crate::convert::bucket_mesh_by_material(&mut converted);
            (converted, Some(mats))
        })
        .collect();

    let mut meshes = Vec::with_capacity(results.len());
    let mut shipped = 0usize;
    for (i, (mesh, mats)) in results.into_iter().enumerate() {
        if let Some(m) = mats {
            snap.last_materials.insert(chunks[i], m);
            shipped += 1;
        }
        meshes.push(mesh);
    }
    (meshes, shipped)
}
