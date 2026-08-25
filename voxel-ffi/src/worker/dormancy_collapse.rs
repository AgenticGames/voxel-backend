//! Dormancy world collapse (2026-08-25): when dormancy is triggered, every
//! LOADED area sitting at >= 75% effective stress collapses straight to its
//! end state — no slab-fall cinematic, no camera shake. The player finds the
//! cave-ins when they re-explore.
//!
//! Two phases (2026-08-25 evening — "share the work"):
//!
//! **Phase 1** runs in the sleep handler's FAR-work window, strictly AFTER
//! the reveal curtain rises (`SLEEP_FAR_GO`), on the otherwise-idle worker
//! pool — zero montage-critical-path cost. It excludes only the FILMED
//! block (from the morph snapshot; falls back to the full montage-protected
//! set if the snapshot is empty) plus the player-adjacent chunks, and
//! STASHES the filmed-block seeds on the store for phase 2.
//!
//! **Phase 2** runs after the montage ends. Release signals (any of):
//! `voxel_sleep_far_work_go` fired while the gate is ALREADY open (= the
//! CleanupMontage backstop call — UE's normal cleanup never calls
//! clear_protected, it re-pushes the residency-only set), an actual
//! `voxel_montage_clear_protected`, or the stranded fallback. Worker 0's
//! idle path then re-validates the stashed seeds, cascades them, and
//! publishes the remesh for BOTH phases via the mine-collapse path
//! (remesh_dirty then batched_seam_pass_mine — the seam pass reads the
//! base_meshes cache, so the remesh MUST come first).
//!
//! Player-adjacent chunks are exempt in BOTH phases — the player is at the
//! wake point for all of it, and rubble on their head has no counterplay.
//!
//! Known limitation: with `r.Dormancy.MaxPoiPlays > 0` (default 0), later
//! POI plays film blocks phase 1 cannot identify (the snapshot holds only
//! the current play), so a collapse could land in a to-be-filmed POI block
//! and the rewind would snap at resync. Revisit if POI plays return.

use std::collections::HashSet;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use crossbeam_channel::Sender;
use voxel_core::stress::{
    detect_and_execute_collapses_v2_with_force_deadline, BrokenStrutEvent, OverstressedVoxel,
};
use voxel_fluid::FluidEvent;
use voxel_gen::config::{GenerationConfig, StressConfig};

use crate::store::{ChunkStore, DormancyPhase2Pending};
use crate::types::{FfiStrutBroken, WorkerResult};

use super::seam::batched_seam_pass_mine;

/// Effective-stress threshold at which an area collapses during dormancy.
/// 0.75 (user call 2026-08-25, was 0.85): deliberately BELOW the 0.85
/// crack-decal / creak tier, so dormancy also drops marginal rock that never
/// showed cracks — geological time is dramatic. Also matches the default
/// `slab_cohesion_threshold` (0.75), so seed clusters and their cohesion
/// expansion agree.
const DORMANCY_COLLAPSE_THRESHOLD: f32 = 0.75;

/// Wall-clock budget for a cascade (each phase gets its own). Phase 1 runs
/// behind the reveal so this is not a UX budget; phase 2 holds the store
/// write lock on worker 0 post-montage, so the deadline bounds how long the
/// mine lane can stall. Partial processing is safe: unprocessed areas stay
/// standing.
const DORMANCY_COLLAPSE_BUDGET: Duration = Duration::from_secs(6);

/// Chunks within this Chebyshev radius of the player are exempt in both
/// phases. Dropping a slab (or burying the floor in its rubble pile) at the
/// spot the player wakes up on is a no-counterplay death; everything
/// farther out is fair geology.
const PLAYER_EXCLUDE_CHEB: i32 = 2;

/// If the montage-end signal never arrives (hard abort before any curtain),
/// phase 2 still fires after this long so the stash cannot strand. Normal
/// montages end ~15s after arming; 45s clears the slowest multi-play runs.
const PHASE2_STRANDED_FALLBACK: Duration = Duration::from_secs(45);

// ─────────────────────────────────────────────────────────────────────────
// Phase 1 — reveal window (called from the sleep handler after SLEEP_FAR_GO)
// ─────────────────────────────────────────────────────────────────────────

/// Scan every loaded chunk for cells at >= the dormancy threshold and
/// collapse the eligible slabs to their end state — DENSITY WRITES ONLY.
/// Both the filmed-block seeds AND the remesh of every phase-1 collapse
/// chunk are stashed for phase 2: publishing the collapse meshes during
/// the reveal measurably stretched the black hold by contending with the
/// resync drain, so the montage window ships only its baseline mesh volume.
pub(crate) fn apply_dormancy_world_collapse(
    ctx: &super::HandlerCtx<'_>,
    player_chunk: (i32, i32, i32),
) {
    let t0 = Instant::now();
    let stress_cfg = ctx.stress_config.read().unwrap().clone();
    let chunk_size = ctx.config.read().unwrap().chunk_size;

    // The deferred zone: the block the morph is actively filming. The
    // snapshot's keys are the current play's block in Rust chunk space,
    // populated by step 0 — which always precedes curtain-up. Empty
    // snapshot (no morph ran) → conservative fallback to the full
    // montage-protected set.
    let filmed: HashSet<(i32, i32, i32)> = {
        let snap = ctx.morph_snapshot.lock().unwrap();
        snap.keys.iter().copied().collect()
    };
    let mut used_fallback = false;

    // ── Scan (read lock — writers wait, readers don't) ──────────────────
    let (eligible, deferred, stats) = {
        let s = ctx.store.read().unwrap();
        let zone: &HashSet<(i32, i32, i32)> = if filmed.is_empty() {
            used_fallback = true;
            &s.montage_protected
        } else {
            &filmed
        };
        collect_dormancy_seeds(
            &s.stress_fields,
            &s.density_fields,
            zone,
            player_chunk,
            chunk_size,
        )
    };
    let scan_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let deferred_count = deferred.len();
    let out = if eligible.is_empty() {
        None
    } else {
        Some(run_dormancy_cascade(
            ctx.store,
            &stress_cfg,
            chunk_size,
            ctx.world_scale,
            ctx.result_tx,
            eligible,
            player_chunk,
            "phase1",
        ))
    };
    let phase1_dirty: Vec<(i32, i32, i32)> =
        out.as_ref().map(|o| o.dirty.clone()).unwrap_or_default();

    // Stash for phase 2: the filmed-block seeds AND the phase-1 remesh work.
    if deferred_count > 0 || !phase1_dirty.is_empty() {
        let mut s = ctx.store.write().unwrap();
        s.dormancy_phase2_pending = Some(DormancyPhase2Pending {
            seeds: deferred,
            pending_remesh: phase1_dirty,
            player_chunk,
            armed_at: Instant::now(),
        });
        s.dormancy_phase2_go = false;
    }

    match &out {
        None => crate::panic_log::note(&format!(
            "[DORMANCY-COLLAPSE] phase1: no eligible cells >= {:.2} eff in {} chunks (skipped: {} player-adjacent, {} filmed{}; {} seeds deferred to phase 2) — scan {:.0}ms",
            DORMANCY_COLLAPSE_THRESHOLD, stats.chunks_scanned, stats.skipped_player,
            stats.skipped_zone, if used_fallback { " [fallback: full protected set]" } else { "" },
            deferred_count, scan_ms
        )),
        Some(out) => crate::panic_log::note(&format!(
            "[DORMANCY-COLLAPSE] phase1: {} seeds in {} chunks (skipped: {} player-adjacent, {} filmed{}; {} seeds deferred) -> {} collapses, {} voxels, {} dirty chunks (remesh deferred to phase 2), {} struts broken — scan {:.0}ms cascade {:.0}ms{}",
            out.seed_count, stats.chunks_scanned, stats.skipped_player, stats.skipped_zone,
            if used_fallback { " [fallback: full protected set]" } else { "" },
            deferred_count,
            out.events, out.voxels, out.dirty.len(), out.struts_broken,
            scan_ms, out.cascade_ms,
            if out.hit_deadline { " (HIT DEADLINE — remaining areas left standing)" } else { "" }
        )),
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Phase 2 — post-montage (worker 0 idle path)
// ─────────────────────────────────────────────────────────────────────────

/// If phase-2 work is armed and released (montage protection cleared, or
/// the stranded fallback elapsed), re-validate the stashed filmed-block
/// seeds and cascade them, then remesh + seam-stitch via the mine-collapse
/// path. Returns true if work was done.
pub(crate) fn try_run_phase2(
    store: &Arc<RwLock<ChunkStore>>,
    stress_config: &Arc<RwLock<StressConfig>>,
    config: &Arc<RwLock<GenerationConfig>>,
    result_tx: &Sender<WorkerResult>,
    fluid_event_tx: &Sender<FluidEvent>,
    world_scale: f32,
) -> bool {
    // Cheap gate under a read lock — this runs every worker-0 idle tick.
    {
        let s = match store.read() {
            Ok(s) => s,
            Err(_) => return false,
        };
        match &s.dormancy_phase2_pending {
            None => return false,
            Some(p) => {
                if !s.dormancy_phase2_go && p.armed_at.elapsed() < PHASE2_STRANDED_FALLBACK {
                    return false;
                }
            }
        }
    }

    let pending = {
        let mut s = store.write().unwrap();
        s.dormancy_phase2_go = false;
        s.dormancy_phase2_pending.take()
    };
    let Some(pending) = pending else { return false };
    let latency_ms = pending.armed_at.elapsed().as_secs_f64() * 1000.0;

    let stress_cfg = stress_config.read().unwrap().clone();
    let cfg = config.read().unwrap().clone();
    let chunk_size = cfg.chunk_size;

    // Re-validate: phase 1's own collapses (and their queued recalcs) may
    // have changed the world under these seeds during the reveal.
    let stashed = pending.seeds.len();
    let seeds: Vec<OverstressedVoxel> = {
        let s = store.read().unwrap();
        pending
            .seeds
            .into_iter()
            .filter(|v| {
                let (key, lx, ly, lz) = voxel_core::stress::world_to_chunk_local(
                    v.world_x, v.world_y, v.world_z, chunk_size,
                );
                let solid = s
                    .density_fields
                    .get(&key)
                    .map(|df| df.get(lx, ly, lz).material.is_solid())
                    .unwrap_or(false);
                if !solid {
                    return false;
                }
                s.stress_fields
                    .get(&key)
                    .map(|sf| sf.effective(lx, ly, lz) >= DORMANCY_COLLAPSE_THRESHOLD)
                    .unwrap_or(false)
            })
            .collect()
    };

    let out = if seeds.is_empty() {
        None
    } else {
        Some(run_dormancy_cascade(
            store,
            &stress_cfg,
            chunk_size,
            world_scale,
            result_tx,
            seeds,
            pending.player_chunk,
            "phase2",
        ))
    };

    // Publish: phase-1's deferred remesh + phase-2's own cascade dirty.
    // Mine-path remesh recipe: remesh_dirty refreshes the base_meshes cache,
    // and MUST run before batched_seam_pass_mine (which publishes combined
    // base+seam meshes FROM that cache — skipping the remesh publishes stale
    // geometry, see worker/stress.rs slab path). Sliced write locks so
    // post-montage mine traffic interleaves.
    let mut remesh: Vec<(i32, i32, i32)> = pending.pending_remesh;
    if let Some(o) = &out {
        remesh.extend(o.dirty.iter().copied());
    }
    remesh.sort();
    remesh.dedup();
    let t_remesh = Instant::now();
    if !remesh.is_empty() {
        for slice in remesh.chunks(24) {
            let dirty_bounds: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> =
                slice
                    .iter()
                    .map(|&key| (key, 0usize, 0usize, 0usize, chunk_size, chunk_size, chunk_size))
                    .collect();
            let mut s = store.write().unwrap();
            let _ = s.remesh_dirty(&dirty_bounds, &cfg, world_scale);
        }
        batched_seam_pass_mine(&remesh, &cfg, store, result_tx, fluid_event_tx, world_scale);
    }
    let remesh_ms = t_remesh.elapsed().as_secs_f64() * 1000.0;

    match &out {
        None => crate::panic_log::note(&format!(
            "[DORMANCY-COLLAPSE] phase2: 0 of {} stashed seeds still valid ({:.0}ms after arming); published {} deferred phase-1 chunks (remesh+seam {:.0}ms)",
            stashed, latency_ms, remesh.len(), remesh_ms
        )),
        Some(o) => crate::panic_log::note(&format!(
            "[DORMANCY-COLLAPSE] phase2: {} of {} stashed seeds valid ({:.0}ms after arming) -> {} collapses, {} voxels, {} struts broken — cascade {:.0}ms; published {} chunks total (incl deferred phase-1, remesh+seam {:.0}ms){}",
            o.seed_count, stashed, latency_ms,
            o.events, o.voxels, o.struts_broken, o.cascade_ms,
            remesh.len(), remesh_ms,
            if o.hit_deadline { " (HIT DEADLINE — remaining areas left standing)" } else { "" }
        )),
    }

    true
}

// ─────────────────────────────────────────────────────────────────────────
// Shared cascade + bookkeeping
// ─────────────────────────────────────────────────────────────────────────

struct CascadeOutcome {
    dirty: Vec<(i32, i32, i32)>,
    seed_count: usize,
    events: usize,
    voxels: usize,
    struts_broken: usize,
    hit_deadline: bool,
    cascade_ms: f64,
}

/// Existing v2 slab machinery, straight to end state: slab -> air, pile
/// lands below. halt_at_struts=true so braced areas stay up (struts pay
/// BFS-halt HP as usual); force_collapse=false so grounded rock stays put.
/// Marks dirty chunks for save persistence, queues no_collapse stress
/// recalcs per event, and ships StrutsBroken so UE strut visuals stay in
/// sync. Does NOT remesh — phase 1 folds dirty into the sleep FAR pass,
/// phase 2 runs the mine-path remesh itself.
#[allow(clippy::too_many_arguments)]
fn run_dormancy_cascade(
    store: &Arc<RwLock<ChunkStore>>,
    stress_cfg: &StressConfig,
    chunk_size: usize,
    world_scale: f32,
    result_tx: &Sender<WorkerResult>,
    mut seeds: Vec<OverstressedVoxel>,
    player_chunk: (i32, i32, i32),
    label: &str,
) -> CascadeOutcome {
    // Nearest-first: if the deadline bites, the areas the player will reach
    // soonest are the ones guaranteed to have collapsed.
    let pcw = (
        player_chunk.0 * chunk_size as i32,
        player_chunk.1 * chunk_size as i32,
        player_chunk.2 * chunk_size as i32,
    );
    seeds.sort_by_key(|v| {
        let dx = (v.world_x - pcw.0) as i64;
        let dy = (v.world_y - pcw.1) as i64;
        let dz = (v.world_z - pcw.2) as i64;
        dx * dx + dy * dy + dz * dz
    });
    let seed_count = seeds.len();

    let t_cascade = Instant::now();
    let deadline = t_cascade + DORMANCY_COLLAPSE_BUDGET;
    let mut broken: Vec<BrokenStrutEvent> = Vec::new();

    let mut dirty: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut voxels_collapsed = 0usize;
    let (events, hit_deadline) = {
        let mut s = store.write().unwrap();
        let (events, hit_deadline) = {
            let (density_fields, stress_fields, support_fields) = s.sleep_fields_mut();
            detect_and_execute_collapses_v2_with_force_deadline(
                density_fields,
                stress_fields,
                support_fields,
                &seeds,
                stress_cfg,
                chunk_size,
                false, // defer_pile — place rubble immediately (end state)
                false, // force_collapse — grounded slabs don't fall
                true,  // halt_at_struts — braced areas survive dormancy
                &mut broken,
                Some(deadline),
            )
        };

        for ev in &events {
            for k in &ev.affected_chunks {
                dirty.insert(*k);
            }
            for slab in &ev.slabs {
                voxels_collapsed += slab.voxels.len();
            }
        }

        if !dirty.is_empty() {
            let dirty_vec: Vec<(i32, i32, i32)> = dirty.iter().copied().collect();
            // Save persistence — without this the cave-ins evaporate on reload.
            s.modification_tracker.mark_dirty_many(&dirty_vec);
            // Collapsed cells carry stale stress (they are air now) and the
            // new cavity edges carry fresh spans: queue deferred position
            // recalcs so warnings + crack decals re-evaluate on the normal
            // deferred path. no_collapse: this pass IS the collapse.
            for ev in &events {
                let center = (
                    ev.center.0.round() as i32,
                    ev.center.1.round() as i32,
                    ev.center.2.round() as i32,
                );
                let mut max_cheb = 0i32;
                for slab in &ev.slabs {
                    for cv in &slab.voxels {
                        let d = (cv.world_x - center.0)
                            .abs()
                            .max((cv.world_y - center.1).abs())
                            .max((cv.world_z - center.2).abs());
                        max_cheb = max_cheb.max(d + slab.fall_distance.abs());
                    }
                }
                let radius = (max_cheb + 6).min(chunk_size as i32 * 3);
                s.queue_stress_dirty_no_collapse(center, radius);
            }
        }
        (events, hit_deadline)
    };

    // Struts that broke bracing slabs ride the existing StrutsBroken result
    // so UE's placed-strut visuals + crack overlays stay in sync (dedup by
    // world position UE-side, same as the mine path).
    if !broken.is_empty() {
        let cs = chunk_size as i32;
        let ffi_struts: Vec<FfiStrutBroken> = broken
            .iter()
            .map(|ev| FfiStrutBroken {
                world_x: ev.chunk.0 * cs + ev.lx as i32,
                world_y: ev.chunk.1 * cs + ev.ly as i32,
                world_z: ev.chunk.2 * cs + ev.lz as i32,
                support_type: ev.support_type as u8,
                source: 1, // BFS halt
                _pad: [0; 2],
            })
            .collect();
        let _ = result_tx.send(WorkerResult::StrutsBroken { struts: ffi_struts });
    }

    // Per-event locations (UE coords) — verification + forensics: BugItGo to
    // center_ue (with Ghost) and the cave-in should be visible.
    for (i, ev) in events.iter().enumerate() {
        let vox: usize = ev.slabs.iter().map(|sl| sl.voxels.len()).sum();
        let fall = ev.slabs.first().map(|sl| sl.fall_distance).unwrap_or(0);
        crate::panic_log::note(&format!(
            "[DORMANCY-COLLAPSE-EVENT] {} {}: center_rust=({:.0},{:.0},{:.0}) center_ue=({:.0},{:.0},{:.0}) voxels={} fall={} chunks={}",
            label, i, ev.center.0, ev.center.1, ev.center.2,
            ev.center.0 * world_scale, -ev.center.2 * world_scale, ev.center.1 * world_scale,
            vox, fall, ev.affected_chunks.len()
        ));
    }

    CascadeOutcome {
        dirty: dirty.into_iter().collect(),
        seed_count,
        events: events.len(),
        voxels: voxels_collapsed,
        struts_broken: broken.len(),
        hit_deadline,
        cascade_ms: t_cascade.elapsed().as_secs_f64() * 1000.0,
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Scan
// ─────────────────────────────────────────────────────────────────────────

struct ScanStats {
    chunks_scanned: usize,
    skipped_player: usize,
    skipped_zone: usize,
}

/// Pure scan: every solid cell at >= `DORMANCY_COLLAPSE_THRESHOLD` effective
/// stress across the given fields. Player-adjacent chunks are DROPPED
/// (exempt in both phases); cells inside `deferred_zone` chunks are returned
/// separately for the post-montage phase. Split out of the store plumbing
/// for testability.
fn collect_dormancy_seeds(
    stress_fields: &std::collections::HashMap<(i32, i32, i32), voxel_core::stress::StressField>,
    density_fields: &std::collections::HashMap<(i32, i32, i32), voxel_core::density::DensityField>,
    deferred_zone: &HashSet<(i32, i32, i32)>,
    player_chunk: (i32, i32, i32),
    chunk_size: usize,
) -> (Vec<OverstressedVoxel>, Vec<OverstressedVoxel>, ScanStats) {
    let mut eligible = Vec::new();
    let mut deferred = Vec::new();
    let mut stats = ScanStats {
        chunks_scanned: 0,
        skipped_player: 0,
        skipped_zone: 0,
    };

    for (&key, sf) in stress_fields.iter() {
        let cheb = (key.0 - player_chunk.0)
            .abs()
            .max((key.1 - player_chunk.1).abs())
            .max((key.2 - player_chunk.2).abs());
        if cheb <= PLAYER_EXCLUDE_CHEB {
            stats.skipped_player += 1;
            continue;
        }
        let in_zone = deferred_zone.contains(&key);
        if in_zone {
            stats.skipped_zone += 1;
        }
        let Some(df) = density_fields.get(&key) else {
            continue;
        };
        if !in_zone {
            stats.chunks_scanned += 1;
        }
        let sink: &mut Vec<OverstressedVoxel> =
            if in_zone { &mut deferred } else { &mut eligible };
        // 0..chunk_size (not the +1 overlap row) — overlap cells are owned
        // by the neighbor chunk's scan.
        for z in 0..chunk_size {
            for y in 0..chunk_size {
                for x in 0..chunk_size {
                    let eff = sf.effective(x, y, z);
                    if eff < DORMANCY_COLLAPSE_THRESHOLD {
                        continue;
                    }
                    if !df.get(x, y, z).material.is_solid() {
                        continue;
                    }
                    sink.push(OverstressedVoxel {
                        world_x: key.0 * chunk_size as i32 + x as i32,
                        world_y: key.1 * chunk_size as i32 + y as i32,
                        world_z: key.2 * chunk_size as i32 + z as i32,
                        stress: eff,
                    });
                }
            }
        }
    }
    (eligible, deferred, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use voxel_core::density::DensityField;
    use voxel_core::material::Material;
    use voxel_core::stress::StressField;

    const CHUNK_SIZE: usize = 16;
    const FIELD_SIZE: usize = 17;

    fn solid_density() -> DensityField {
        let mut df = DensityField::new(FIELD_SIZE);
        for sample in df.samples.iter_mut() {
            sample.density = 1.0;
            sample.material = Material::Granite;
        }
        df
    }

    fn world(
        keys: &[(i32, i32, i32)],
    ) -> (
        HashMap<(i32, i32, i32), StressField>,
        HashMap<(i32, i32, i32), DensityField>,
    ) {
        let mut sf = HashMap::new();
        let mut df = HashMap::new();
        for &k in keys {
            sf.insert(k, StressField::new(FIELD_SIZE));
            df.insert(k, solid_density());
        }
        (sf, df)
    }

    #[test]
    fn threshold_is_respected() {
        // Player far away at (10,10,10) so nothing is player-excluded.
        let (mut sf, df) = world(&[(0, 0, 0)]);
        {
            let f = sf.get_mut(&(0, 0, 0)).unwrap();
            f.set(3, 3, 3, 0.74); // just under — must NOT seed
            f.set(5, 5, 5, 0.76); // over — must seed
        }
        let (eligible, deferred, stats) = collect_dormancy_seeds(
            &sf, &df, &HashSet::new(), (10, 10, 10), CHUNK_SIZE,
        );
        assert_eq!(stats.chunks_scanned, 1);
        assert!(deferred.is_empty());
        assert_eq!(eligible.len(), 1, "only the 0.76 cell should seed");
        assert_eq!(
            (eligible[0].world_x, eligible[0].world_y, eligible[0].world_z),
            (5, 5, 5)
        );
    }

    #[test]
    fn painted_stress_counts_toward_effective() {
        let (mut sf, df) = world(&[(0, 0, 0)]);
        {
            let f = sf.get_mut(&(0, 0, 0)).unwrap();
            f.set(4, 4, 4, 0.50);
            f.set_painted(4, 4, 4, 0.30); // eff 0.80 — must seed
        }
        let (eligible, _, _) = collect_dormancy_seeds(
            &sf, &df, &HashSet::new(), (10, 10, 10), CHUNK_SIZE,
        );
        assert_eq!(eligible.len(), 1, "base+painted crossing 0.75 should seed");
    }

    #[test]
    fn air_cells_do_not_seed() {
        let (mut sf, mut df) = world(&[(0, 0, 0)]);
        sf.get_mut(&(0, 0, 0)).unwrap().set(6, 6, 6, 0.95);
        {
            let d = df.get_mut(&(0, 0, 0)).unwrap();
            let s = d.get_mut(6, 6, 6);
            s.density = -1.0;
            s.material = Material::Air;
        }
        let (eligible, deferred, _) = collect_dormancy_seeds(
            &sf, &df, &HashSet::new(), (10, 10, 10), CHUNK_SIZE,
        );
        assert!(eligible.is_empty() && deferred.is_empty(),
            "stale stress on an air cell must not seed");
    }

    #[test]
    fn player_adjacent_chunks_are_dropped_entirely() {
        let (mut sf, df) = world(&[(0, 0, 0), (5, 0, 0)]);
        sf.get_mut(&(0, 0, 0)).unwrap().set(3, 3, 3, 0.95);
        sf.get_mut(&(5, 0, 0)).unwrap().set(3, 3, 3, 0.95);
        // Player at (1,0,0): chunk (0,0,0) is cheb 1 <= 2 -> dropped (not
        // even deferred); chunk (5,0,0) is cheb 4 -> eligible.
        let (eligible, deferred, stats) = collect_dormancy_seeds(
            &sf, &df, &HashSet::new(), (1, 0, 0), CHUNK_SIZE,
        );
        assert_eq!(stats.skipped_player, 1);
        assert!(deferred.is_empty(), "player-adjacent seeds are never deferred");
        assert_eq!(eligible.len(), 1);
        assert_eq!(eligible[0].world_x, 5 * CHUNK_SIZE as i32 + 3);
    }

    #[test]
    fn filmed_zone_seeds_are_deferred_not_dropped() {
        let (mut sf, df) = world(&[(4, 0, 0), (5, 0, 0)]);
        sf.get_mut(&(4, 0, 0)).unwrap().set(3, 3, 3, 0.95);
        sf.get_mut(&(5, 0, 0)).unwrap().set(3, 3, 3, 0.95);
        let mut zone = HashSet::new();
        zone.insert((4, 0, 0));
        let (eligible, deferred, stats) = collect_dormancy_seeds(
            &sf, &df, &zone, (20, 20, 20), CHUNK_SIZE,
        );
        assert_eq!(stats.skipped_zone, 1);
        assert_eq!(eligible.len(), 1, "non-zone seed stays eligible");
        assert_eq!(eligible[0].world_x, 5 * CHUNK_SIZE as i32 + 3);
        assert_eq!(deferred.len(), 1, "zone seed is deferred for phase 2");
        assert_eq!(deferred[0].world_x, 4 * CHUNK_SIZE as i32 + 3);
    }
}
