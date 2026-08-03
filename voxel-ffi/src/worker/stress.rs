//! Deferred stress recalculation + collapse cascade (worker 0 only).
//!
//! Pure code-movement out of the former monolithic `worker.rs`; behavior is
//! unchanged. Visibility widened to `pub(crate)` so the worker loop can call it.

use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use crossbeam_channel::Sender;
use voxel_fluid::FluidEvent;
use voxel_gen::config::{GenerationConfig, StressConfig};

use crate::store::ChunkStore;
use crate::types::{FfiSlabFallData, WorkerResult};

use super::seam::{
    batched_seam_pass_mine, prune_destroyed_mushrooms_for_chunks,
    recompute_crystals_for_chunks, split_slab_for_visual, throttle_collapse_remesh,
};

/// Deferred stress recalculation: checks if the stress dirty queue timer has elapsed,
/// runs the v2 stress algorithm, emits warnings, and triggers collapse if needed.
/// Returns true if work was done (so worker loop can continue instead of sleeping).
pub(crate) fn try_process_stress_queue(
    store: &Arc<RwLock<ChunkStore>>,
    stress_config: &Arc<RwLock<StressConfig>>,
    config: &Arc<RwLock<GenerationConfig>>,
    result_tx: &Sender<WorkerResult>,
    fluid_event_tx: &Sender<FluidEvent>,
    world_scale: f32,
) -> bool {
    use voxel_core::stress::{
        recalc_stress_region_v2_with_load_decay, detect_and_execute_collapses_v2,
    };
    let _ = detect_and_execute_collapses_v2; // legacy import retained for symmetry
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

    // Keep a parallel reference for editor-trigger evaluation later — the
    // local `events` gets shadowed by the v2 collapse events vec further
    // down, but trigger eval still needs to know which AABBs were mined.
    let mined_dirty_events: Vec<voxel_core::stress::StressDirtyEvent> = events.clone();

    let cfg = config.read().unwrap().clone();
    let stress_cfg = stress_config.read().unwrap().clone();
    let chunk_size = cfg.chunk_size;

    // Derive affected chunks from events (union of all event bounding boxes)
    let mut dirty_chunks: Vec<(i32, i32, i32)> = {
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

    // Run v2 stress recalculation — only voxels within event radii are recalculated.
    // Use the load-decay variant so live mining wears nearby struts. Broken
    // struts are accumulated into `result.broken_struts` and forwarded as a
    // `StrutsBroken` worker result so UE can play breaking VFX + refresh the
    // crack overlay around each broken strut's world position.
    let mut result = {
        let mut s = store.write().unwrap();
        let (density, stress, support) = s.sleep_fields_mut();
        recalc_stress_region_v2_with_load_decay(
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

    // Forward load-decay broken struts to UE before the collapse pass, so the
    // breaking VFX plays right when stress crosses the strut's tolerance —
    // even if no slab fall is triggered. UE deduplicates by world voxel
    // position vs PlacedSupports, so emitting twice (load-decay here + later
    // BFS-halt) is harmless.
    if !result.broken_struts.is_empty() {
        let ffi_struts: Vec<crate::types::FfiStrutBroken> = result.broken_struts
            .iter()
            .map(|ev| {
                let cs = chunk_size as i32;
                crate::types::FfiStrutBroken {
                    world_x: ev.chunk.0 * cs + ev.lx as i32,
                    world_y: ev.chunk.1 * cs + ev.ly as i32,
                    world_z: ev.chunk.2 * cs + ev.lz as i32,
                    support_type: ev.support_type as u8,
                    source: 0, // load decay
                    _pad: [0; 2],
                }
            })
            .collect();
        let _ = result_tx.send(WorkerResult::StrutsBroken { struts: ffi_struts });
    }

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

    // ── Editor trigger pre-pass ──
    //
    // CRITICAL: evaluate scripted triggers BEFORE the warning scan below so
    // the seed cells get their stress written into the stress field. Without
    // this, the dust/creak/shake warnings (and the on-screen cracks, smoke,
    // wall-shake effects on UE side) appear only at the natural overstressed
    // cells around the mine point — which is NOT where the designer painted.
    // Writing the seed stress here makes the cinematic warning visuals show
    // up at the painted region instead of at the mining point.
    let (early_has_armed, early_has_force) = {
        let s = store.read().unwrap();
        (
            s.triggers.iter().any(|t| t.armed),
            !s.force_fire_trigger_ids.is_empty(),
        )
    };
    // Build the painted-seed overstressed list AND write seed stress into
    // the stress field in one pass. The result feeds both:
    //   (a) the warning scan below — so cracks/dust/shake appear at the
    //       painted region instead of only at the mine point.
    //   (b) the force_collapse detect call further down (post-pass).
    let mut trigger_seed_overstressed: Vec<voxel_core::stress::OverstressedVoxel> = Vec::new();
    let mut trigger_extra_dirty_chunks: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut trigger_eval_log_pending: Vec<String> = Vec::new();
    if early_has_armed || early_has_force {
        let mined_aabbs: Vec<crate::triggers::VoxelAabb> = mined_dirty_events
            .iter()
            .map(|ev| {
                let r = ev.radius.max(1);
                crate::triggers::VoxelAabb {
                    min: (ev.center.0 - r, ev.center.1 - r, ev.center.2 - r),
                    max: (ev.center.0 + r, ev.center.1 + r, ev.center.2 + r),
                }
            })
            .collect();

        let mut s = store.write().unwrap();
        let forced_ids: Vec<u32> = std::mem::take(&mut s.force_fire_trigger_ids);
        let mut fired_ids: Vec<u32> = forced_ids.clone();
        for trig in &s.triggers {
            if !trig.armed || forced_ids.contains(&trig.id) {
                continue;
            }
            let fires = mined_aabbs
                .iter()
                .any(|aabb| trig.should_fire(aabb, &s.density_fields, chunk_size));
            if fires {
                fired_ids.push(trig.id);
            }
        }
        trigger_eval_log_pending.push(format!(
            "  TRIGGER eval: fired_ids={:?} (from {} mined AABBs, force-fire={})",
            fired_ids, mined_aabbs.len(), forced_ids.len()
        ));

        for tid in &fired_ids {
            let trig_clone = match s.find_trigger(*tid) {
                Some(t) => t.clone(),
                None => continue,
            };
            let mut seeded_count: u32 = 0;
            let cs = chunk_size as i32;
            for &(wx, wy, wz) in &trig_clone.target_slab_voxels {
                if !crate::engine::cell_has_solid_center(&s, wx, wy, wz, cs) {
                    continue;
                }
                let cx = wx.div_euclid(cs);
                let cy = wy.div_euclid(cs);
                let cz = wz.div_euclid(cs);
                let lx = wx.rem_euclid(cs) as usize;
                let ly = wy.rem_euclid(cs) as usize;
                let lz = wz.rem_euclid(cs) as usize;
                // Write stress into the field so the warning scan picks it
                // up (cracks/dust/shake at the painted region, not just the
                // mine point).
                if let Some(sf) = s.stress_fields.get_mut(&(cx, cy, cz)) {
                    sf.set(lx, ly, lz, 1.5);
                }
                trigger_extra_dirty_chunks.insert((cx, cy, cz));
                // Push onto the forced overstressed list for the force
                // collapse pass downstream.
                trigger_seed_overstressed.push(voxel_core::stress::OverstressedVoxel {
                    world_x: wx,
                    world_y: wy,
                    world_z: wz,
                    stress: 1.5,
                });
                seeded_count += 1;
            }
            // Disarm now — this trigger has fired.
            if let Some(t) = s.find_trigger_mut(*tid) {
                t.armed = false;
            }
            trigger_eval_log_pending.push(format!(
                "  TRIGGER {} ('{}') fired: {}/{} solid seed cells (stress_field bumped + queued for force collapse)",
                tid, trig_clone.name, seeded_count, trig_clone.target_slab_voxels.len()
            ));
        }
    }
    for line in &trigger_eval_log_pending {
        dbg(line.clone());
    }

    // Merge seed chunks into dirty_chunks so the warning scan visits them.
    // Mining-driven dirty_chunks alone might not cover seed locations far
    // from the mine point.
    if !trigger_extra_dirty_chunks.is_empty() {
        let existing: HashSet<(i32, i32, i32)> = dirty_chunks.iter().copied().collect();
        for k in &trigger_extra_dirty_chunks {
            if !existing.contains(k) {
                dirty_chunks.push(*k);
            }
        }
    }

    // Emit stress warnings for UE — EDGE-TRIGGERED (playtest #209): a cell only
    // warns when its tier ROSE since the last scan, so re-recalcs near a
    // long-standing overhang don't shake the camera on every mining swing for
    // stress the player didn't cause. Tier drops are recorded silently so a
    // future re-crossing warns again. Push-based: Rust says where, UE checks
    // proximity to the player.
    {
        let mut s = store.write().unwrap();
        let mut warnings = Vec::new();
        let mut dust_count = 0u32;
        let mut creak_count = 0u32;
        let mut shake_count = 0u32;
        let mut suppressed = 0u32;
        let grid_size = chunk_size + 1;

        for &(cx, cy, cz) in &dirty_chunks {
            let sf = match s.stress_fields.get_mut(&(cx, cy, cz)) {
                Some(f) => f,
                None => continue,
            };
            for z in 0..grid_size {
                for y in 0..grid_size {
                    for x in 0..grid_size {
                        let stress = sf.get(x, y, z);
                        let tier = if stress < stress_cfg.warn_dust_threshold {
                            0u8
                        } else if stress >= stress_cfg.warn_shake_threshold {
                            3u8
                        } else if stress >= stress_cfg.warn_creak_threshold {
                            2u8
                        } else {
                            1u8
                        };
                        let prev = sf.warned_tier(x, y, z);
                        if tier != prev {
                            sf.set_warned_tier(x, y, z, tier);
                        }
                        if tier == 0 {
                            continue;
                        }
                        if tier <= prev {
                            suppressed += 1;
                            continue;   // already announced at this tier or higher
                        }
                        match tier {
                            3 => shake_count += 1,
                            2 => creak_count += 1,
                            _ => dust_count += 1,
                        }
                        let wx = cx * chunk_size as i32 + x as i32;
                        let wy = cy * chunk_size as i32 + y as i32;
                        let wz = cz * chunk_size as i32 + z as i32;
                        warnings.push(FfiStressWarning {
                            world_x: wx as f32 * world_scale,
                            world_y: -(wz as f32) * world_scale,
                            world_z: wy as f32 * world_scale,
                            stress,
                            warning_type: tier,
                        });
                    }
                }
            }
        }
        if dust_count + creak_count + shake_count + suppressed > 0 {
            dbg(format!("  warnings: dust={} creak={} shake={} (edge-triggered; {} unchanged cells suppressed)",
                dust_count, creak_count, shake_count, suppressed));
        }
        // Send the highest-stress warnings (sorted, capped at 128)
        warnings.sort_by(|a, b| b.stress.partial_cmp(&a.stress).unwrap_or(std::cmp::Ordering::Equal));
        warnings.truncate(128);
        if !warnings.is_empty() {
            let _ = result_tx.send(WorkerResult::StressWarnings { warnings });
        }
    }

    // Trigger eval was done in the pre-pass above (so warning visuals
    // appear at the painted region). Here we just need to run the natural
    // + forced collapse detections using already-collected data.
    if !result.overstressed.is_empty() || !trigger_seed_overstressed.is_empty() {
        let mut s = store.write().unwrap();

        let collapse_start = std::time::Instant::now();
        let mut events: Vec<voxel_core::stress::CollapseEventV2> = Vec::new();
        // Broken-strut events accumulated across both natural + scripted
        // passes — forwarded as a `StrutsBroken` result so UE can play the
        // breaking VFX + refresh crack overlay around each broken strut.
        let mut broken_struts: Vec<voxel_core::stress::BrokenStrutEvent> = Vec::new();
        // Natural collapse pass: only stress-recalc'd overstressed cells.
        // Filters (size / grounding / cohesion) apply normally so player
        // mining doesn't trigger spurious cave-ins on supported rock.
        // Strut halt is ENABLED here — alive struts brace the slab and
        // take HP damage proportional to blocked volume.
        // #214 gate: a batch whose every event forbids collapse (strut
        // placement) rewrites stress + decals but must not EXECUTE the
        // latent overstress it surfaced — placing a brace was turning into
        // the very cave-in the player was bracing against. Any mining/
        // removal event in the same batch re-enables the pass.
        let batch_allows_collapse = mined_dirty_events.iter().any(|e| e.allow_collapse);
        if !result.overstressed.is_empty() && batch_allows_collapse {
            let (density, stress, support) = s.sleep_fields_mut();
            let natural = voxel_core::stress::detect_and_execute_collapses_v2_with_force(
                density, stress, support,
                &result.overstressed,
                &stress_cfg, chunk_size,
                true,  // defer_pile
                false, // force_collapse — natural filters apply
                true,  // halt_at_struts — alive struts brace the slab
                &mut broken_struts,
            );
            events.extend(natural);
        }
        // Scripted-trigger pass: force_collapse=true. Bypasses the
        // grounding filter; designer-painted regions on cave walls or
        // pillars fall even though they're physically supported.
        // Scripted triggers also bypass the strut-halt — the designer's
        // authored event overrides player struts (otherwise a single Crystal
        // Strut could veto a boss-room collapse cinematic).
        if !trigger_seed_overstressed.is_empty() {
            let (density, stress, support) = s.sleep_fields_mut();
            let mut _scripted_broken: Vec<voxel_core::stress::BrokenStrutEvent> = Vec::new();
            let forced = voxel_core::stress::detect_and_execute_collapses_v2_with_force(
                density, stress, support,
                &trigger_seed_overstressed,
                &stress_cfg, chunk_size,
                true,  // defer_pile
                true,  // force_collapse
                false, // halt_at_struts — scripted: ignore struts
                &mut _scripted_broken,
            );
            events.extend(forced);
        }
        let collapse_ms = collapse_start.elapsed().as_secs_f64() * 1000.0;

        // BFS-halt broken struts (the slab couldn't be braced anymore) ride
        // back to UE so the breaking VFX + crack overlay can refresh around
        // each one. Source=1 distinguishes from load-decay (source=0).
        if !broken_struts.is_empty() {
            let ffi_struts: Vec<crate::types::FfiStrutBroken> = broken_struts
                .iter()
                .map(|ev| {
                    let cs = chunk_size as i32;
                    crate::types::FfiStrutBroken {
                        world_x: ev.chunk.0 * cs + ev.lx as i32,
                        world_y: ev.chunk.1 * cs + ev.ly as i32,
                        world_z: ev.chunk.2 * cs + ev.lz as i32,
                        support_type: ev.support_type as u8,
                        source: 1, // BFS halt
                        _pad: [0; 2],
                    }
                })
                .collect();
            let _ = result_tx.send(WorkerResult::StrutsBroken { struts: ffi_struts });
        }

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

            // Mark collapse-modified chunks for save persistence
            let collapse_keys: Vec<_> = all_dirty.iter().map(|&(k, ..)| k).collect();
            s.modification_tracker.mark_dirty_many(&collapse_keys);

            let _base_meshes = s.remesh_dirty(&all_dirty, &cfg, world_scale);
            drop(s);

            // ── Cinematic emission ──
            //
            // For each collapse event:
            //   1. Emit a CollapseWarning result carrying AABB + ETA so UE can
            //      play Acts 1-2 (warning shake, dust, cracks) BEFORE the
            //      falling slab actor appears.
            //   2. For each slab in the event, extract a real DC mesh (works
            //      post-pile because the slab struct preserves voxel data),
            //      then emit a SlabFall result with the mesh + fall metadata.
            //      UE spawns one AVoxelCollapseSlabActor per slab fragment.
            //
            // The old CollapseResult emission is dropped — the slab+warning
            // pipeline replaces it. UE consumers should switch over.

            // Collapse region size distribution
            let mut vol_hist = [0u32; 5];
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

            // Snapshot density field reference for slab extraction.
            let s_ref = store.read().unwrap();
            let voxel_scale = cfg.voxel_scale();

            // Open the cinematic diagnostics log. UE writes phase transitions
            // to the same file (different prefix) so the timeline can be
            // reconstructed end-to-end.
            use std::io::Write;
            let log_path = "D:/Unreal Projects/Mithril2026/Saved/collapse_log.txt";
            let mut log_file = std::fs::OpenOptions::new()
                .create(true).append(true).open(log_path).ok();
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis()).unwrap_or(0);
            if let Some(f) = &mut log_file {
                let _ = writeln!(f, "");
                let _ = writeln!(f, "================================================================");
                let _ = writeln!(f, "[{}][R] === COLLAPSE BURST: {} events ===", now_ms, events.len());
                let _ = writeln!(f, "================================================================");
            }

            for (event_idx, event) in events.iter().enumerate() {
                // Compute event bounds in UE units for the warning event.
                let mut ev_min = (f32::MAX, f32::MAX, f32::MAX);
                let mut ev_max = (f32::MIN, f32::MIN, f32::MIN);
                for slab in &event.slabs {
                    let lo_ue = (slab.bb_min.0 as f32 * world_scale,
                                 -(slab.bb_min.2 as f32) * world_scale,
                                 slab.bb_min.1 as f32 * world_scale);
                    let hi_ue = (slab.bb_max.0 as f32 * world_scale,
                                 -(slab.bb_max.2 as f32) * world_scale,
                                 slab.bb_max.1 as f32 * world_scale);
                    ev_min.0 = ev_min.0.min(lo_ue.0.min(hi_ue.0));
                    ev_min.1 = ev_min.1.min(lo_ue.1.min(hi_ue.1));
                    ev_min.2 = ev_min.2.min(lo_ue.2.min(hi_ue.2));
                    ev_max.0 = ev_max.0.max(lo_ue.0.max(hi_ue.0));
                    ev_max.1 = ev_max.1.max(lo_ue.1.max(hi_ue.1));
                    ev_max.2 = ev_max.2.max(lo_ue.2.max(hi_ue.2));
                }
                let center_ue = (
                    event.center.0 * world_scale,
                    -event.center.2 * world_scale,
                    event.center.1 * world_scale,
                );
                let extent_ue = (
                    (ev_max.0 - ev_min.0) * 0.5,
                    (ev_max.1 - ev_min.1) * 0.5,
                    (ev_max.2 - ev_min.2) * 0.5,
                );

                // Volume-scaled warning ETA (1.5s for tiny → 3.5s for huge).
                let eta_ms = (1500 + (event.total_volume.min(200) as u32) * 8).min(3500);
                let severity = if event.total_volume >= 80 { 4 }
                    else if event.total_volume >= 30 { 3 }
                    else if event.total_volume >= 10 { 2 }
                    else { 1 };

                // ── Diagnostic log: event header ──
                if let Some(f) = &mut log_file {
                    let _ = writeln!(f, "[{}][R] --- EVENT[{}] ---", now_ms, event_idx);
                    let _ = writeln!(f, "[{}][R]   center_rust=({:.2},{:.2},{:.2}) center_ue=({:.1},{:.1},{:.1})",
                        now_ms, event.center.0, event.center.1, event.center.2,
                        center_ue.0, center_ue.1, center_ue.2);
                    let _ = writeln!(f, "[{}][R]   total_volume={} slab_count={} affected_chunks={}",
                        now_ms, event.total_volume, event.slabs.len(), event.affected_chunks.len());
                    let _ = writeln!(f, "[{}][R]   bounds_ue: min=({:.1},{:.1},{:.1}) max=({:.1},{:.1},{:.1}) extent=({:.1},{:.1},{:.1})",
                        now_ms,
                        ev_min.0, ev_min.1, ev_min.2,
                        ev_max.0, ev_max.1, ev_max.2,
                        extent_ue.0, extent_ue.1, extent_ue.2);
                    let _ = writeln!(f, "[{}][R]   warning: severity={} eta_ms={} (PreFallWarning duration)",
                        now_ms, severity, eta_ms);
                }

                let _ = result_tx.send(WorkerResult::CollapseWarning {
                    center_ue,
                    bounds_extent_ue: extent_ue,
                    severity,
                    eta_ms,
                    volume: event.total_volume,
                });

                // Per-slab falling visuals — split each event slab into 1-5
                // visual sub-fragments via grid partition so big collapses
                // read as multiple chunks breaking off rather than one
                // monolithic block. Each sub-slab gets its own falling-slab
                // actor in UE with independent tumble axis + impact zone.
                let visual_slabs: Vec<voxel_core::stress::CollapseSlab> = event.slabs.iter()
                    .flat_map(|s| split_slab_for_visual(s).into_iter())
                    .collect();
                if let Some(f) = &mut log_file {
                    let _ = writeln!(f, "[{}][R]   visual_split: {} event slab(s) → {} sub-slab(s)",
                        now_ms, event.slabs.len(), visual_slabs.len());
                }
                for (slab_idx, slab) in visual_slabs.iter().enumerate() {
                    // Material breakdown for the slab (top 4 materials).
                    let mut mat_counts: std::collections::HashMap<voxel_core::material::Material, u32> = std::collections::HashMap::new();
                    for v in &slab.voxels {
                        *mat_counts.entry(v.material).or_insert(0) += 1;
                    }
                    let mut mat_breakdown: Vec<(voxel_core::material::Material, u32)> =
                        mat_counts.into_iter().collect();
                    mat_breakdown.sort_by(|a, b| b.1.cmp(&a.1));
                    let breakdown_str = mat_breakdown.iter().take(4)
                        .map(|(m, c)| format!("{:?}={}", m, c))
                        .collect::<Vec<_>>().join(", ");

                    let mesh_opt = crate::slab::extract_slab_mesh(
                        slab, &s_ref.density_fields, chunk_size,
                        voxel_scale, world_scale,
                    );

                    if let Some(f) = &mut log_file {
                        let _ = writeln!(f, "[{}][R]   slab[{}]:", now_ms, slab_idx);
                        let _ = writeln!(f, "[{}][R]     voxels={} fall_dist={} dom_mat={:?}",
                            now_ms, slab.voxels.len(), slab.fall_distance,
                            slab.dominant_material);
                        let _ = writeln!(f, "[{}][R]     materials: {}", now_ms, breakdown_str);
                        let _ = writeln!(f, "[{}][R]     bb_rust: min=({},{},{}) max=({},{},{})",
                            now_ms, slab.bb_min.0, slab.bb_min.1, slab.bb_min.2,
                            slab.bb_max.0, slab.bb_max.1, slab.bb_max.2);
                        match &mesh_opt {
                            Some(m) => {
                                let _ = writeln!(f, "[{}][R]     mesh extracted: positions={} indices={} ({} tris)",
                                    now_ms, m.positions.len(), m.indices.len(), m.indices.len() / 3);
                            }
                            None => {
                                let _ = writeln!(f, "[{}][R]     ★ MESH EXTRACTION FAILED — slab actor will fall back to procedural box",
                                    now_ms);
                            }
                        }
                    }

                    let Some(mesh) = mesh_opt else { continue };

                    let center_rust = slab.center;
                    let spawn_ue = (
                        center_rust.0 * world_scale,
                        -center_rust.2 * world_scale,
                        center_rust.1 * world_scale,
                    );
                    let land_ue_z = (center_rust.1 - slab.fall_distance as f32) * world_scale;
                    let land_ue = (spawn_ue.0, spawn_ue.1, land_ue_z);
                    let slab_extent_ue = (
                        (slab.bb_max.0 - slab.bb_min.0).max(1) as f32 * world_scale * 0.5,
                        (slab.bb_max.2 - slab.bb_min.2).max(1) as f32 * world_scale * 0.5,
                        (slab.bb_max.1 - slab.bb_min.1).max(1) as f32 * world_scale * 0.5,
                    );

                    // ── Principal horizontal axis = leading-edge direction ──
                    // Compute the 2×2 XZ covariance of the slab voxels and
                    // pick the dominant eigenvector. Magnitude scaled by
                    // anisotropy so chunky cubes get ~0 (no preferred tilt)
                    // and long thin slabs get ~1 (strong domino direction).
                    let (lead_ue_x, lead_ue_y) = {
                        let cx_r = center_rust.0;
                        let cz_r = center_rust.2;
                        let mut cov_xx = 0.0f32;
                        let mut cov_xz = 0.0f32;
                        let mut cov_zz = 0.0f32;
                        for v in &slab.voxels {
                            let dx = v.world_x as f32 - cx_r;
                            let dz = v.world_z as f32 - cz_r;
                            cov_xx += dx * dx;
                            cov_xz += dx * dz;
                            cov_zz += dz * dz;
                        }
                        let n = (slab.voxels.len().max(1)) as f32;
                        cov_xx /= n; cov_xz /= n; cov_zz /= n;

                        let trace = cov_xx + cov_zz;
                        let det = cov_xx * cov_zz - cov_xz * cov_xz;
                        let disc = (trace * trace - 4.0 * det).max(0.0);
                        let s = disc.sqrt();
                        let lambda_max = (trace + s) * 0.5;
                        let lambda_min = (trace - s) * 0.5;

                        // Eigenvector for lambda_max: v = (cov_xz, lambda_max - cov_xx)
                        let ex = cov_xz;
                        let ez = lambda_max - cov_xx;
                        let mag = (ex * ex + ez * ez).sqrt();
                        let (lead_rx, lead_rz) = if mag > 1e-4 {
                            (ex / mag, ez / mag)
                        } else if cov_xx >= cov_zz {
                            (1.0, 0.0)
                        } else {
                            (0.0, 1.0)
                        };

                        // Anisotropy in [0..1]: 0 = round, 1 = elongated.
                        let aniso = if lambda_max > 1e-6 {
                            (1.0 - (lambda_min / lambda_max).max(0.0)).clamp(0.0, 1.0)
                        } else { 0.0 };

                        // Rust X → UE X; Rust Z → -UE Y. Vertical Y unused
                        // because tilting is horizontal (no fall component).
                        (lead_rx * aniso, -lead_rz * aniso)
                    };

                    let mut fall = FfiSlabFallData::default();
                    fall.spawn_x = spawn_ue.0;
                    fall.spawn_y = spawn_ue.1;
                    fall.spawn_z = spawn_ue.2;
                    fall.land_x = land_ue.0;
                    fall.land_y = land_ue.1;
                    fall.land_z = land_ue.2;
                    fall.fall_distance = slab.fall_distance as f32 * world_scale;
                    fall.bounds_extent_x = slab_extent_ue.0;
                    fall.bounds_extent_y = slab_extent_ue.1;
                    fall.bounds_extent_z = slab_extent_ue.2;
                    fall.volume = slab.voxels.len() as u32;
                    fall.dominant_material = slab.dominant_material as u8;
                    // ★ CRITICAL FIX: propagate the warning ETA to every slab
                    // so the slab actor's PreFallWarning + ImminentDetach phases
                    // actually run. Without this, every slab skipped straight
                    // to Falling, rendering the cinematic invisible.
                    fall.warning_severity = severity;
                    fall.warning_eta_ms = eta_ms;
                    fall.leading_edge_dir_x = lead_ue_x;
                    fall.leading_edge_dir_y = lead_ue_y;

                    if let Some(f) = &mut log_file {
                        let _ = writeln!(f, "[{}][R]     spawn_ue=({:.1},{:.1},{:.1}) land_ue=({:.1},{:.1},{:.1}) fall_dist_ue={:.1}",
                            now_ms, spawn_ue.0, spawn_ue.1, spawn_ue.2,
                            land_ue.0, land_ue.1, land_ue.2,
                            fall.fall_distance);
                        let _ = writeln!(f, "[{}][R]     extent_ue=({:.1},{:.1},{:.1}) → emitting SlabFall (eta={}ms severity={})",
                            now_ms, slab_extent_ue.0, slab_extent_ue.1, slab_extent_ue.2,
                            eta_ms, severity);
                    }

                    let _ = result_tx.send(WorkerResult::SlabFall {
                        mesh,
                        fall_data: fall,
                    });
                }
            }
            drop(s_ref);
            if let Some(f) = &mut log_file {
                let _ = writeln!(f, "[{}][R] === BURST COMPLETE ===", now_ms);
            }

            dbg(format!("  emitted {} CollapseWarning + per-slab SlabFall events", events.len()));

            let dirty_keys: Vec<(i32, i32, i32)> = all_dirty.iter().map(|&(k, ..)| k).collect();

            // Crystal recompute can fire NOW — UE will pick up fresh placements
            // when each chunk mesh arrives in its scheduled batch.
            recompute_crystals_for_chunks(store, &cfg, &dirty_keys);
            prune_destroyed_mushrooms_for_chunks(store, &dirty_keys);

            // ── Per-event cinematic-aligned chunk-mesh scheduling. ──
            //
            // Density was already mutated (slab cells cleared + pile placed)
            // inside detect_and_execute_collapses_v2. If we send chunk meshes
            // for ALL affected chunks now, UE shows the cave roof hole AND
            // the floor pile immediately, before the falling-slab actor has
            // visually impacted.
            //
            // Fix: per event, split affected chunks into:
            //   - SLAB chunks (cave roof hole) → send at fall-start time
            //     so the roof opens up just as the slab actor visually
            //     detaches.
            //   - PILE chunks (floor pile) → send at impact time so the
            //     pile appears under the slab right when it lands.
            //   - chunks in BOTH → send at impact (the later time wins,
            //     and the dust burst masks the transition).
            //
            // Each event has its own timing — short falls update fast,
            // long falls take longer. They DON'T wait for each other.
            const SLAB_GRAVITY: f32 = 327.0; // matches UE's slab actor

            for event in &events {
                let eta_ms = (1500u32 + (event.total_volume.min(200) as u32) * 8).min(3500);
                // Imminent-detach beat sits between warning and falling.
                // Bumped from 600ms → 1600ms so the fracture cracks have a
                // full second of unobstructed peak-intensity visibility
                // BEFORE the falling slab pulls the player's eye away.
                // MUST stay in sync with `AVoxelCollapseSlabActor::ImminentDuration`
                // on the UE side (currently 1.6s).
                let imminent_ms = 1600u32;
                let max_fall_uu = event.slabs.iter()
                    .map(|s| s.fall_distance)
                    .max().unwrap_or(0) as f32 * world_scale;
                let fall_dur_ms = if max_fall_uu > 0.0 {
                    ((2.0 * max_fall_uu / SLAB_GRAVITY).sqrt() * 1000.0).clamp(500.0, 6000.0) as u32
                } else { 1000 };
                let fall_start_ms = eta_ms + imminent_ms;
                let impact_ms = fall_start_ms + fall_dur_ms;

                // Split chunk sets:
                let pile_set: std::collections::HashSet<(i32, i32, i32)> =
                    event.pile_chunks.iter().copied().collect();
                let slab_only: Vec<(i32, i32, i32)> = event.slab_chunks.iter()
                    .copied()
                    .filter(|k| !pile_set.contains(k))
                    .collect();
                let pile_or_both: Vec<(i32, i32, i32)> = event.pile_chunks.clone();

                dbg(format!("  event center=({:.1},{:.1},{:.1}) vol={} slab_only_chunks={} pile_chunks={} fall_start={}ms impact={}ms",
                    event.center.0, event.center.1, event.center.2, event.total_volume,
                    slab_only.len(), pile_or_both.len(), fall_start_ms, impact_ms));

                // Schedule slab-chunk remesh at fall-start (cave roof opens).
                // At this point, density has slab cells cleared but NO pile
                // yet (deferred), so the remesh shows roof hole only.
                //
                // ★ CRITICAL: must call `remesh_dirty` BEFORE the seam pass
                // because `batched_seam_pass_mine` reads from the cached
                // `base_meshes`. The slab cells were cleared earlier
                // (inside `detect_and_execute_collapses_v2_with_options`)
                // but the cached mesh still shows the slab in place. Without
                // the explicit re-DC, the seam pass publishes stale meshes
                // and the cave roof stays visually intact until something
                // else (e.g., the pile thread's neighborhood seam at
                // impact + PILE_PREVIEW_MS) eventually re-DCs the area.
                // Symptom: "see-through wall" / cave roof opens 1.2s late.
                if !event.slab_chunks.is_empty() {
                    let store_c = std::sync::Arc::clone(store);
                    let cfg_c = cfg.clone();
                    let tx_c = result_tx.clone();
                    let fluid_tx_c = fluid_event_tx.clone();
                    let keys = event.slab_chunks.clone();
                    let ws = world_scale;
                    let event_center_for_log = event.center;
                    std::thread::spawn(move || {
                        let log_line = |msg: String| {
                            use std::io::Write;
                            let now_ms = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_millis()).unwrap_or(0);
                            if let Ok(mut f) = std::fs::OpenOptions::new()
                                .create(true).append(true)
                                .open("D:/Unreal Projects/Mithril2026/Saved/collapse_log.txt")
                            {
                                let _ = writeln!(f, "[{}][R-SLABFX] {}", now_ms, msg);
                            }
                        };

                        std::thread::sleep(std::time::Duration::from_millis(fall_start_ms as u64));
                        log_line(format!(
                            "fall_start fired for event center=({:.1},{:.1},{:.1}) slab_chunks={}",
                            event_center_for_log.0, event_center_for_log.1, event_center_for_log.2,
                            keys.len()));

                        // Stagger consecutive collapse remeshes via the
                        // global rate limiter so the game thread isn't
                        // dumped with 6 events × N chunks at once.
                        let rate_wait = throttle_collapse_remesh();
                        if rate_wait > 0 {
                            log_line(format!("rate-limited: extra wait {}ms", rate_wait));
                        }

                        // Refresh base_meshes cache so the seam pass
                        // publishes the post-clear mesh, not the pre-clear
                        // (slab-still-there) mesh.
                        let cs = cfg_c.chunk_size;
                        let dirty_bounds: Vec<_> = keys.iter()
                            .map(|&k| (k, 0usize, 0usize, 0usize, cs, cs, cs))
                            .collect();
                        {
                            let mut s = store_c.write().unwrap();
                            let _ = s.remesh_dirty(&dirty_bounds, &cfg_c, ws);
                        }
                        log_line(format!("remeshed {} slab chunks (cache refreshed)", keys.len()));

                        // HISMs anchored on now-cleared roof surfaces must
                        // recompute too — otherwise crystals stay floating
                        // where the slab used to be.
                        recompute_crystals_for_chunks(&store_c, &cfg_c, &keys);
                        prune_destroyed_mushrooms_for_chunks(&store_c, &keys);

                        // Force fresh send by clearing the last-sent-hash so
                        // the seam pass doesn't skip thinking nothing changed.
                        {
                            let mut s = store_c.write().unwrap();
                            for k in &keys {
                                s.last_sent_mesh_hash.remove(k);
                            }
                        }

                        batched_seam_pass_mine(&keys, &cfg_c, &store_c, &tx_c, &fluid_tx_c, ws);
                        log_line(format!("batched_seam_pass_mine called (cave roof now open)"));
                    });
                }

                // Schedule pile placement at impact, then a 4-tier preview
                // reveal over PILE_PREVIEW_MS, then the real chunk remesh.
                //
                // Sequence (deferred thread):
                //   t = impact_ms                   — apply pile to density;
                //                                     extract 4 tier preview
                //                                     meshes; emit them.
                //   t = impact_ms + 0..PREVIEW_MS   — UE animates debris
                //                                     actor reveal.
                //   t = impact_ms + PREVIEW_MS      — recompute crystals,
                //                                     remesh pile_chunks, run
                //                                     seam pass; the real
                //                                     chunk mesh now matches
                //                                     the pile and the debris
                //                                     actor despawns.
                //
                // Density is applied immediately so subsequent gameplay
                // queries (mining, flatten, stress) see the correct world;
                // only the chunk MESH lags by PREVIEW_MS, which is the
                // illusion window.
                if !event.pending_piles.is_empty() {
                    // 6 tiers × 200 ms/tier = 1200 ms total reveal. Each tier
                    // is one cumulative slice of the heap; UE swaps which one
                    // is visible at each slot boundary with a small dust puff
                    // per swap. After the full window the chunk remesh fires
                    // and the debris actor self-destructs.
                    const PILE_PREVIEW_MS: u32 = 1200;
                    let store_c = std::sync::Arc::clone(store);
                    let cfg_c = cfg.clone();
                    let stress_cfg_c = stress_cfg.clone();
                    let tx_c = result_tx.clone();
                    let fluid_tx_c = fluid_event_tx.clone();
                    let pending = event.pending_piles.clone();
                    let cs_c = chunk_size;
                    let ws = world_scale;
                    let pending_count = pending.len();
                    let event_center = event.center;
                    std::thread::spawn(move || {
                        // Helper to write to the cinematic diagnostics log
                        // (same file Rust+UE share, tagged [R-DEFERRED]).
                        let log_line = |msg: String| {
                            use std::io::Write;
                            let now_ms = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_millis()).unwrap_or(0);
                            if let Ok(mut f) = std::fs::OpenOptions::new()
                                .create(true).append(true)
                                .open("D:/Unreal Projects/Mithril2026/Saved/collapse_log.txt")
                            {
                                let _ = writeln!(f, "[{}][R-DEFERRED] {}", now_ms, msg);
                            }
                        };

                        log_line(format!(
                            "thread-spawn impact_ms={} pending_count={} center=({:.1},{:.1},{:.1})",
                            impact_ms, pending_count,
                            event_center.0, event_center.1, event_center.2));

                        std::thread::sleep(std::time::Duration::from_millis(impact_ms as u64));
                        log_line(format!("woke after sleep, applying {} piles", pending_count));

                        // ── Apply pile + collect written_cells for preview ──
                        let mut pile_chunks: Vec<(i32, i32, i32)> = Vec::new();
                        let mut all_written: Vec<voxel_core::density_ops::WrittenCell> = Vec::new();
                        {
                            let mut s = match store_c.write() {
                                Ok(s) => s,
                                Err(e) => {
                                    log_line(format!("FAILED to acquire store write lock: {:?}", e));
                                    return;
                                }
                            };
                            for (i, p) in pending.iter().enumerate() {
                                let pr = voxel_core::stress::apply_pending_pile_with_result(
                                    &mut s.density_fields, &stress_cfg_c,
                                    p, cs_c,
                                );
                                log_line(format!(
                                    "  pile[{}]: collapsed_voxels={} bb=({},{},{})..({},{},{}) → {} chunks affected, {} written cells",
                                    i, p.collapsed_voxels.len(),
                                    p.bb_min.0, p.bb_min.1, p.bb_min.2,
                                    p.bb_max.0, p.bb_max.1, p.bb_max.2,
                                    pr.affected_chunks.len(), pr.written_cells.len()));
                                for k in pr.affected_chunks {
                                    pile_chunks.push(k);
                                }
                                all_written.extend(pr.written_cells);
                            }
                        }
                        pile_chunks.sort();
                        pile_chunks.dedup();
                        log_line(format!("total pile_chunks (deduped)={}, total written_cells={}",
                            pile_chunks.len(), all_written.len()));

                        if pile_chunks.is_empty() {
                            log_line("⚠ pile_chunks is empty — nothing to remesh".to_string());
                            return;
                        }

                        // ── Extract N-tier preview meshes & emit ──
                        // Use the first pending pile's spawn coord as the
                        // anchor UE keys the debris actor by. (Each event
                        // has a single pending pile in practice; if there
                        // were multiple, we'd loop and emit per-pile.)
                        // Tier count comes from PILE_PREVIEW_TIER_COUNT
                        // (currently 8) so this code stays correct on bumps.
                        let voxel_scale = 1.0_f32;
                        let tier_meshes = crate::pile_preview::extract_pile_tier_meshes(
                            &all_written, cs_c, voxel_scale, ws,
                        );
                        // Anchor in UE coords: pile bbox center at landing Y.
                        let anchor_p = &pending[0];
                        let cx = ((anchor_p.bb_min.0 + anchor_p.bb_max.0) as f32 * 0.5
                            + anchor_p.landing_offset as f32 * 0.0) as f32;
                        let cy = ((anchor_p.bb_min.1 + anchor_p.bb_max.1) as f32 * 0.5
                            - anchor_p.landing_offset as f32) as f32;
                        let cz = ((anchor_p.bb_min.2 + anchor_p.bb_max.2) as f32 * 0.5) as f32;
                        // Convert Rust voxel coord → UE world (Y-up→Z-up).
                        let ue_x = cx * ws;
                        let ue_y = -cz * ws;
                        let ue_z = cy * ws;
                        let bbox_x = ((anchor_p.bb_max.0 - anchor_p.bb_min.0).max(1)) as f32 * ws;
                        let bbox_y = ((anchor_p.bb_max.1 - anchor_p.bb_min.1).max(1)) as f32 * ws;
                        let bbox_z = ((anchor_p.bb_max.2 - anchor_p.bb_min.2).max(1)) as f32 * ws;
                        let total_volume: u32 = pending.iter()
                            .map(|p| p.collapsed_voxels.len() as u32).sum();
                        let dom_mat = anchor_p.dominant_material as u8;

                        for (tier_idx, mesh) in tier_meshes.into_iter().enumerate() {
                            let mut fall = FfiSlabFallData::default();
                            fall.spawn_x = ue_x;
                            fall.spawn_y = ue_y;
                            fall.spawn_z = ue_z;
                            fall.land_x = ue_x;
                            fall.land_y = ue_y;
                            fall.land_z = ue_z;
                            fall.bounds_extent_x = bbox_x;
                            fall.bounds_extent_y = bbox_z; // Y-up→Z-up swap
                            fall.bounds_extent_z = bbox_y;
                            fall.volume = total_volume;
                            fall.dominant_material = dom_mat;
                            fall.pile_tier_index = tier_idx as u8;
                            fall.warning_eta_ms = PILE_PREVIEW_MS;
                            let _ = tx_c.send(WorkerResult::PilePreviewTier {
                                mesh,
                                fall_data: fall,
                            });
                        }
                        log_line(format!("emitted {} tier preview meshes anchor=({:.1},{:.1},{:.1})",
                            crate::pile_preview::PILE_PREVIEW_TIER_COUNT,
                            ue_x, ue_y, ue_z));

                        // ── Hold the preview window ──
                        std::thread::sleep(std::time::Duration::from_millis(PILE_PREVIEW_MS as u64));

                        // ── Throttle through the global rate limiter ──
                        // Multi-region collapses queue 6+ deferred threads
                        // that all wake within ~1.6 s after the first
                        // impact. Without throttling they'd all dump their
                        // chunk remesh + seam pass results onto the game
                        // thread back-to-back, producing the "2 second
                        // freeze" the user reported. Rate limiter spreads
                        // these batches by `COLLAPSE_REMESH_GAP_MS` each.
                        let rate_wait = throttle_collapse_remesh();
                        if rate_wait > 0 {
                            log_line(format!("rate-limited: extra wait {}ms before remesh", rate_wait));
                        }

                        // ── Real commit: crystals + remesh + seam ──
                        recompute_crystals_for_chunks(&store_c, &cfg_c, &pile_chunks);
                        prune_destroyed_mushrooms_for_chunks(&store_c, &pile_chunks);
                        log_line("crystals recomputed".to_string());

                        // ★ CRITICAL: re-DC the chunks so `base_meshes` cache
                        // reflects the just-placed pile. `batched_seam_pass_mine`
                        // reads from `base_meshes` — without this remesh, the
                        // seam pass would publish stale meshes.
                        let cs = cfg_c.chunk_size;
                        let dirty_bounds: Vec<_> = pile_chunks.iter()
                            .map(|&k| (k, 0usize, 0usize, 0usize, cs, cs, cs))
                            .collect();
                        {
                            let mut s = store_c.write().unwrap();
                            let _ = s.remesh_dirty(&dirty_bounds, &cfg_c, ws);
                        }
                        log_line(format!("remeshed {} pile chunks (base_meshes cache refreshed)",
                            pile_chunks.len()));

                        // Force fresh send by clearing the last-sent-hash.
                        {
                            let mut s = store_c.write().unwrap();
                            let mut cleared = 0u32;
                            for k in &pile_chunks {
                                if s.last_sent_mesh_hash.remove(k).is_some() {
                                    cleared += 1;
                                }
                            }
                            log_line(format!("cleared {} stale hashes from {} pile chunks",
                                cleared, pile_chunks.len()));
                        }

                        batched_seam_pass_mine(&pile_chunks, &cfg_c, &store_c, &tx_c, &fluid_tx_c, ws);
                        log_line(format!("batched_seam_pass_mine called for {} chunks", pile_chunks.len()));
                    });
                } else {
                    if let Some(f) = &mut log_file {
                        let _ = writeln!(f, "[{}][R] ⚠ event has no pending_piles — pile defer was a no-op (pile already placed inline?)", now_ms);
                    }
                }
            }
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
