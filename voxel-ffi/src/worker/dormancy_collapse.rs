//! Dormancy world collapse (2026-08-25): when dormancy is triggered, loaded
//! areas that geological time deems unstable collapse straight to their end
//! state — no slab-fall cinematic, no camera shake. The player finds the
//! cave-ins when they re-explore.
//!
//! ## Dormancy stress recipe ("geological time uses geological stress")
//!
//! The PLAY-time stress config deliberately disables the span model
//! (MinSafeSpan 333 / SpanWeight 0.001 in the live JSON) so mining feels
//! stable — which also means a tunnel roof reads 0.00 stress and no
//! dormancy threshold can ever seed it. This pass therefore does NOT read
//! the stored play-stress: it RECOMPUTES stress with a dormancy-only
//! overlay (span model alive, see `dormancy_stress_config`) into scratch,
//! per surface-shell cell, and seeds cells whose
//! `recomputed + painted >= 0.75`. Roofs and wide tunnel ceilings become
//! seeds. The play-time stress fields, crack overlays and warnings are
//! untouched.
//!
//! STRUT TIERS (user spec 2026-08-25) replace the old relief math:
//! COPPER struts are all SACRIFICED at event start (broken + reported to
//! UE) but their radius keeps a posthumous shadow — evaluated at
//! a-little-safer-than-Geological dials, so big domes still fall there
//! while rooms/corridors survive. IRON AND ABOVE never break and make
//! their radius fully immune (seed-skip + cascade halt + break-restore).
//!
//! Seeding is a UNION of two classes: (a) the span recipe over air-below
//! ceiling/deck cells, and (b) the CRACK TIER — any cell whose STORED
//! play-effective stress crosses the threshold (what the player saw
//! cracking before the sleep falls, whatever its geometry; the cascade's
//! grounded filter handles the rest).
//!
//! Buildings have NO immunity (user call 2026-08-25): pads collapse like
//! anywhere else — iron+ struts are the only real protection. Player-DUG
//! tunnels are fair game: strut them or lose them.
//!
//! ## Two phases (montage-timer protection)
//!
//! **Phase 1** runs in the sleep handler's FAR-work window, strictly AFTER
//! the reveal curtain rises (`SLEEP_FAR_GO`), on the otherwise-idle worker
//! pool — zero montage-critical-path cost. Scan (sliced read locks, rayon
//! per slice, nearest-first, budget-capped) + cascade; DENSITY WRITES ONLY.
//! The filmed block's seeds (morph snapshot keys; fallback: whole protected
//! set) and ALL remesh work are stashed for phase 2.
//!
//! **Phase 2** runs after the montage ends. Release signals (any of):
//! `voxel_sleep_far_work_go` fired while the gate is ALREADY open (= the
//! CleanupMontage backstop call — UE's normal cleanup never calls
//! clear_protected, it re-pushes the residency-only set), an actual
//! `voxel_montage_clear_protected`, or the stranded fallback. Worker 0's
//! idle path then re-validates the stashed seeds under the SAME dormancy
//! recipe, cascades them, and publishes the remesh for BOTH phases via the
//! mine-collapse path (remesh_dirty then batched_seam_pass_mine — the seam
//! pass reads the base_meshes cache, so the remesh MUST come first).
//!
//! Player-adjacent chunks are exempt in BOTH phases — the player is at the
//! wake point for all of it, and rubble on their head has no counterplay.
//!
//! Known limitation: with `r.Dormancy.MaxPoiPlays > 0` (default 0), later
//! POI plays film blocks phase 1 cannot identify (the snapshot holds only
//! the current play), so a collapse could land in a to-be-filmed POI block
//! and the rewind would snap at resync. Revisit if POI plays return.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use crossbeam_channel::Sender;
use rayon::prelude::*;
use voxel_core::density::DensityField;
use voxel_core::stress::{
    detect_and_execute_collapses_v2_with_force_deadline, measure_span_from_air, sample_world,
    world_to_chunk_local,
    BrokenStrutEvent, OverstressedVoxel, StressField, SupportField, SupportType, STRUT_TUNING,
};
use voxel_fluid::FluidEvent;
use voxel_gen::config::{GenerationConfig, StressConfig};

use crate::store::{ChunkStore, DormancyPhase2Pending};
use crate::types::{FfiStrutBroken, WorkerResult};

use super::seam::batched_seam_pass_mine;

// ── "ANGRY MOUNTAIN, 4-LAYER" preset (user final 2026-08-25 — Wrathful
// Peak "kills it", dialed back): every material's roofs fall from
// ~4-voxel-wide passages except granite (granite corridors are the only
// safe unstrutted digs); V-overlay green-yellow+ (stored >= 0.45) falls;
// 4-voxel-thick slabs; single events to 12k voxels. STRUT TIERS (user
// spec): COPPER struts ALL break during the event but posthumously
// soften their braced radius to a-little-safer-than-Geological dials;
// IRON AND ABOVE never break and make their radius fully immune.
// Buildings have NO immunity (pads collapse like anywhere else).

/// Span-recipe threshold at which a recomputed ceiling cell seeds.
const DORMANCY_COLLAPSE_THRESHOLD: f32 = 0.75;

/// Crack-tier threshold on STORED play-effective stress (the V-overlay
/// value: green 0 → yellow 0.5 → red 1.0). 0.40 = green-yellow and
/// hotter falls — what the player can SEE straining comes down.
const DORMANCY_STORED_THRESHOLD: f32 = 0.45;

/// Span model, dormancy overlay (play config has this at 0.001 — dead).
/// 0.50 with min_safe 1: span-2 falls in coal (1.25) and limestone
/// (0.77); granite holds until span 3 (1.25) — granite corridors are the
/// safe digs, narrow passages survive everywhere.
const DORMANCY_SPAN_WEIGHT: f32 = 0.50;

/// Spans at or below this many voxels are safe regardless of material.
const DORMANCY_MIN_SAFE_SPAN: u32 = 1;

/// Roof slab thickness: an over-threshold exposed ceiling cell also seeds
/// this many cells straight up (stopping at air or the chunk top). 4-layer
/// slabs make "giant hole in the roof + obvious pile" instead of a peeled
/// 1-voxel sheet. Upper layers deliberately ignore the play model's
/// air-decay — thickness IS the point.
const DORMANCY_SEED_DEPTH_LAYERS: usize = 4;

/// Copper shadow — the posthumous protection a sacrificed copper strut
/// leaves over its radius: "a little safer than Geological" (Geological
/// was span_w .30 / min_safe 1 / stored .55). Big domes still fall in a
/// copper zone; rooms and corridors survive.
const DORMANCY_COPPER_SPAN_WEIGHT: f32 = 0.25;
const DORMANCY_COPPER_MIN_SAFE_SPAN: u32 = 1;
const DORMANCY_COPPER_STORED_THRESHOLD: f32 = 0.60;

/// Wall-clock budget for the SCAN (support pass + per-cell recompute).
/// Nearest-first, sliced; chunks not reached simply survive this dormancy
/// (logged). Runs behind the reveal, so this is throughput shaping, not UX.
const DORMANCY_SCAN_BUDGET: Duration = Duration::from_secs(4);

/// Wall-clock budget for a cascade (each phase gets its own). Phase 1 runs
/// behind the reveal; phase 2 holds the store write lock on worker 0
/// post-montage, so the deadline bounds how long the mine lane can stall.
/// Partial processing is safe: unprocessed areas stay standing.
const DORMANCY_COLLAPSE_BUDGET: Duration = Duration::from_secs(10);

/// Chunks within this Chebyshev radius of the player are exempt in both
/// phases. Dropping a slab (or burying the floor in its rubble pile) at the
/// spot the player wakes up on is a no-counterplay death; everything
/// farther out is fair geology.
const PLAYER_EXCLUDE_CHEB: i32 = 2;

/// If the montage-end signal never arrives (hard abort before any curtain),
/// phase 2 still fires after this long so the stash cannot strand. Normal
/// montages end ~15s after arming; 45s clears the slowest multi-play runs.
const PHASE2_STRANDED_FALLBACK: Duration = Duration::from_secs(45);

/// Recalc trickle pacing: events fed per interval ≈ 40/s, so a ~700-event
/// wave spreads over ~17s of background instead of one burst.
const RECALC_TRICKLE_PER_FEED: usize = 8;
const RECALC_TRICKLE_INTERVAL: Duration = Duration::from_millis(200);

/// Per-event [DORMANCY-COLLAPSE-EVENT] notes are capped (top N by volume +
/// a summary line): 668 formatted lines produced 300-400ms log-flush
/// stalls in the 2026-08-25 trace.
const EVENT_LOG_CAP: usize = 20;

/// The dormancy stress recipe: clone the LIVE config (keeps the material
/// hardness table + every cascade parameter the designers tuned) and wake
/// the span model up. The support-propagation trio + ground threshold are
/// pinned to the v2 canon values so a play-side support retune can't
/// silently change what geological time does.
fn dormancy_stress_config(live: &StressConfig) -> StressConfig {
    let mut c = live.clone();
    c.span_weight = DORMANCY_SPAN_WEIGHT;
    c.min_safe_span = DORMANCY_MIN_SAFE_SPAN;
    c.lateral_transfer_factor = 0.7;
    c.vertical_transfer_factor = 0.95;
    c.support_propagation_iterations = 2;
    c.ground_threshold = 0.80;
    // Angry Mountain cascade overrides (dormancy-only; live play values
    // untouched): smaller clusters qualify, single events can be dome-sized,
    // and piles keep most of the fallen mass so the aftermath reads.
    c.min_collapse_region = 6;
    c.max_collapse_volume = 12000;
    c.rubble_fill_ratio = 0.80;
    c
}

// ─────────────────────────────────────────────────────────────────────────
// Phase 1 — reveal window (called from the sleep handler after SLEEP_FAR_GO)
// ─────────────────────────────────────────────────────────────────────────

/// Scan every loaded chunk under the dormancy stress recipe and collapse
/// the eligible slabs to their end state — DENSITY WRITES ONLY. Both the
/// filmed-block seeds AND the remesh of every phase-1 collapse chunk are
/// stashed for phase 2 (publishing collapse meshes during the reveal
/// measurably stretched the black hold by contending with the resync
/// drain).
pub(crate) fn apply_dormancy_world_collapse(
    ctx: &super::HandlerCtx<'_>,
    player_chunk: (i32, i32, i32),
) {
    let t0 = Instant::now();
    let stress_cfg = ctx.stress_config.read().unwrap().clone();
    let dcfg = dormancy_stress_config(&stress_cfg);
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

    // ── Strut census, then the COPPER SACRIFICE ─────────────────────
    // Index alive struts BEFORE breaking copper: the scan needs the copper
    // positions for the posthumous shadow protection.
    let strut_idx = {
        let s = ctx.store.read().unwrap();
        build_strut_index(&s.support_fields, chunk_size)
    };
    let copper_broken: usize = if strut_idx.copper.is_empty() {
        0
    } else {
        let broken = {
            let mut s = ctx.store.write().unwrap();
            break_all_copper(&mut s.support_fields)
        };
        let cs_i = chunk_size as i32;
        let ffi_struts: Vec<FfiStrutBroken> = broken
            .iter()
            .map(|&(ck, lx, ly, lz)| FfiStrutBroken {
                world_x: ck.0 * cs_i + lx as i32,
                world_y: ck.1 * cs_i + ly as i32,
                world_z: ck.2 * cs_i + lz as i32,
                support_type: SupportType::Copper as u8,
                source: 1,
                _pad: [0; 2],
            })
            .collect();
        let count = ffi_struts.len();
        let _ = ctx.result_tx.send(WorkerResult::StrutsBroken { struts: ffi_struts });
        count
    };

    // ── Scan: sliced read locks, rayon per slice, budget-capped ─────────
    let scan_deadline = Instant::now() + DORMANCY_SCAN_BUDGET;
    let (eligible, deferred, stats) = {
        let s = ctx.store.read().unwrap();
        let zone: &HashSet<(i32, i32, i32)> = if filmed.is_empty() {
            used_fallback = true;
            &s.montage_protected
        } else {
            &filmed
        };
        collect_dormancy_seeds(
            &s.density_fields,
            &s.stress_fields,
            &dcfg,
            &strut_idx,
            zone,
            player_chunk,
            chunk_size,
            scan_deadline,
        )
    };
    let scan_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let deferred_count = deferred.len();
    let out = if eligible.is_empty() {
        None
    } else {
        Some(run_dormancy_cascade(
            ctx.store,
            &dcfg,
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

    // Stash for phase 2: the filmed-block seeds, the phase-1 remesh work,
    // AND the phase-1 stress recalcs (queueing those during the reveal put
    // a worker-0 recalc batch in front of the montage resync = black hold).
    let phase1_recalcs: Vec<((i32, i32, i32), i32)> =
        out.as_ref().map(|o| o.recalcs.clone()).unwrap_or_default();
    if deferred_count > 0 || !phase1_dirty.is_empty() {
        let mut s = ctx.store.write().unwrap();
        s.dormancy_phase2_pending = Some(DormancyPhase2Pending {
            seeds: deferred,
            pending_remesh: phase1_dirty,
            pending_recalcs: phase1_recalcs,
            player_chunk,
            armed_at: Instant::now(),
        });
        s.dormancy_phase2_go = false;
    }

    let ctx_line = format!(
        "recipe span_w={:.2} min_span={} thr={:.2} stored_thr={:.2} layers={}; {} chunks scanned ({} unscanned/budget), skipped: {} player-adjacent, {} filmed{}; copper struts sacrificed: {}, iron+ shields: {}; {} seeds deferred; ceiling span hist 0-2:{} 3-4:{} 5-6:{} 7-9:{} 10-14:{} 15+:{}",
        DORMANCY_SPAN_WEIGHT, DORMANCY_MIN_SAFE_SPAN, DORMANCY_COLLAPSE_THRESHOLD,
        DORMANCY_STORED_THRESHOLD, DORMANCY_SEED_DEPTH_LAYERS,
        stats.chunks_scanned, stats.chunks_unscanned,
        stats.skipped_player, stats.skipped_zone,
        if used_fallback { " [fallback: full protected set]" } else { "" },
        copper_broken, strut_idx.ironplus.len(),
        deferred_count,
        stats.span_hist[0], stats.span_hist[1], stats.span_hist[2],
        stats.span_hist[3], stats.span_hist[4], stats.span_hist[5]
    );
    match &out {
        None => crate::panic_log::note(&format!(
            "[DORMANCY-COLLAPSE] phase1: no eligible cells ({}) — scan {:.0}ms",
            ctx_line, scan_ms
        )),
        Some(out) => crate::panic_log::note(&format!(
            "[DORMANCY-COLLAPSE] phase1: {} seeds ({}) -> {} collapses, {} voxels, {} dirty chunks (remesh deferred to phase 2), {} struts broken — scan {:.0}ms cascade {:.0}ms{}",
            out.seed_count, ctx_line,
            out.events, out.voxels, out.dirty.len(), out.struts_broken,
            scan_ms, out.cascade_ms,
            if out.hit_deadline { " (HIT DEADLINE — remaining areas left standing)" } else { "" }
        )),
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Phase 2 — post-montage (worker 0 idle path)
// ─────────────────────────────────────────────────────────────────────────

/// If phase-2 work is armed and released (montage over, or the stranded
/// fallback elapsed), re-validate the stashed filmed-block seeds under the
/// dormancy recipe, cascade them, then publish the remesh for BOTH phases
/// via the mine-collapse path. Returns true if work was done.
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
    let dcfg = dormancy_stress_config(&stress_cfg);
    let cfg = config.read().unwrap().clone();
    let chunk_size = cfg.chunk_size;

    // Re-validate under the SAME dormancy recipe (stored play-stress reads
    // 0 for roofs, so the phase-1 check cannot be reused): phase-1's own
    // collapses may have changed spans/solidity under these seeds during
    // the reveal.
    let stashed = pending.seeds.len();
    let seeds: Vec<OverstressedVoxel> = {
        let s = store.read().unwrap();
        // Post-copper-sacrifice index: only iron+ remains alive, and its
        // radius is immune. Copper shadow zones ceased with their struts;
        // deferred seeds already passed the shadow rules at phase-1 time.
        let strut_idx = build_strut_index(&s.support_fields, chunk_size);
        pending
            .seeds
            .into_iter()
            .filter(|v| {
                if strut_cover(&strut_idx, v.world_x, v.world_y, v.world_z)
                    == StrutCover::IronPlus
                {
                    return false;
                }
                let (key, lx, ly, lz) =
                    world_to_chunk_local(v.world_x, v.world_y, v.world_z, chunk_size);
                let mat = match s.density_fields.get(&key) {
                    Some(df) => {
                        let c = df.get(lx, ly, lz);
                        if !c.material.is_solid() {
                            return false;
                        }
                        c.material
                    }
                    None => return false,
                };
                // Crack-tier union, mirroring the scan: stored effective
                // crossing the threshold validates regardless of geometry.
                let stored_eff = s
                    .stress_fields
                    .get(&key)
                    .map(|sf| sf.effective(lx, ly, lz))
                    .unwrap_or(0.0);
                if stored_eff >= DORMANCY_STORED_THRESHOLD {
                    return true;
                }
                // Span seeds may be UPPER slab layers (no air directly
                // below): walk down up to the layer depth to find the
                // exposed ceiling cell this seed rode on, and re-evaluate
                // the recipe there.
                let mut exposed_y = None;
                for dy in 0..DORMANCY_SEED_DEPTH_LAYERS as i32 {
                    let cy = v.world_y - dy;
                    let solid_here = matches!(
                        sample_world(&s.density_fields, v.world_x, cy, v.world_z, chunk_size),
                        Some((_, m)) if m.is_solid()
                    );
                    if !solid_here {
                        break;
                    }
                    let below_air = matches!(
                        sample_world(&s.density_fields, v.world_x, cy - 1, v.world_z, chunk_size),
                        Some((_, m)) if !m.is_solid()
                    );
                    if below_air {
                        exposed_y = Some(cy);
                        break;
                    }
                }
                let Some(ey) = exposed_y else { return false };
                let emat = match sample_world(&s.density_fields, v.world_x, ey, v.world_z, chunk_size)
                {
                    Some((_, m)) => m,
                    None => return false,
                };
                let _ = mat;
                let air_faces =
                    count_air_faces(&s.density_fields, v.world_x, ey, v.world_z, chunk_size);
                let (stress, _) = dormancy_cell_stress(
                    &s.density_fields, &dcfg, emat,
                    v.world_x, ey, v.world_z, chunk_size, air_faces,
                );
                let painted = s
                    .stress_fields
                    .get(&key)
                    .map(|sf| sf.painted(lx, ly, lz))
                    .unwrap_or(0.0);
                let eff = stress + painted;
                eff >= DORMANCY_COLLAPSE_THRESHOLD
            })
            .collect()
    };

    let out = if seeds.is_empty() {
        None
    } else {
        Some(run_dormancy_cascade(
            store,
            &dcfg,
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

    // Park the deferred stress recalcs — phase-1's stash plus phase-2's
    // own — in the TRICKLE BACKLOG rather than the live queue: dumping
    // ~700 events at once made the stress worker recalc them as one giant
    // batch, and the warning/crack/FX wave that followed was a top source
    // of the ~12s post-montage lag (traced 2026-08-25). Worker 0's idle
    // path (`try_trickle_dormancy_recalcs`) feeds them in gradually.
    {
        let mut recalcs = pending.pending_recalcs;
        if let Some(o) = &out {
            recalcs.extend(o.recalcs.iter().copied());
        }
        if !recalcs.is_empty() {
            let mut s = store.write().unwrap();
            s.dormancy_recalc_total = recalcs.len();
            s.dormancy_recalc_started = Some(Instant::now());
            s.dormancy_recalc_backlog = recalcs; // nearest-first already
            s.dormancy_recalc_last_feed = None;
        }
    }

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

/// Feed a few parked recalc events into the live stress queue (worker 0
/// idle path). Paced by `RECALC_TRICKLE_*`; held while a dormancy's phase 2
/// is still pending (montage in progress). Returns true if it fed.
pub(crate) fn try_trickle_dormancy_recalcs(store: &Arc<RwLock<ChunkStore>>) -> bool {
    {
        let s = match store.read() {
            Ok(s) => s,
            Err(_) => return false,
        };
        if s.dormancy_recalc_backlog.is_empty() {
            return false;
        }
        if s.dormancy_phase2_pending.is_some() {
            return false;
        }
        if let Some(t) = s.dormancy_recalc_last_feed {
            if t.elapsed() < RECALC_TRICKLE_INTERVAL {
                return false;
            }
        }
    }
    let mut s = store.write().unwrap();
    if s.dormancy_recalc_backlog.is_empty() {
        return false;
    }
    let n = RECALC_TRICKLE_PER_FEED.min(s.dormancy_recalc_backlog.len());
    let batch: Vec<((i32, i32, i32), i32)> = s.dormancy_recalc_backlog.drain(..n).collect();
    for (center, radius) in batch {
        s.queue_stress_dirty_no_collapse(center, radius);
    }
    s.dormancy_recalc_last_feed = Some(Instant::now());
    if s.dormancy_recalc_backlog.is_empty() {
        let total = s.dormancy_recalc_total;
        let secs = s
            .dormancy_recalc_started
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        crate::panic_log::note(&format!(
            "[DORMANCY-COLLAPSE] recalc trickle complete: {} events fed over {:.1}s",
            total, secs
        ));
    }
    true
}

// ─────────────────────────────────────────────────────────────────────────
// Shared cascade + bookkeeping
// ─────────────────────────────────────────────────────────────────────────

struct CascadeOutcome {
    dirty: Vec<(i32, i32, i32)>,
    /// Deferred stress-recalc events for the collapse sites — the CALLER
    /// queues these (phase 1 stashes for phase 2; phase 2 queues at once).
    recalcs: Vec<((i32, i32, i32), i32)>,
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
/// The cascade's cohesion expansion reads the STORED (play) stress, which
/// is ~0 at roofs — so a roof slab is exactly the seeded surface sheet,
/// naturally 1-2 voxels thick (air_decay limits seeds to the shell).
/// Marks dirty chunks for save persistence, queues no_collapse stress
/// recalcs per event, and ships StrutsBroken so UE strut visuals stay in
/// sync. Does NOT remesh — phase 2 publishes for both phases.
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
        }

        // Iron+ never breaks during dormancy (user spec): restore any the
        // BFS-halt damage felled and drop them from the report. Copper was
        // sacrificed before the cascade, so anything here is iron-and-above.
        broken.retain(|ev| {
            if (ev.support_type as u8) >= SupportType::Iron as u8 {
                if let Some(sf) = s.support_fields.get_mut(&ev.chunk) {
                    sf.set(
                        ev.lx as usize,
                        ev.ly as usize,
                        ev.lz as usize,
                        ev.support_type,
                    );
                }
                false
            } else {
                true
            }
        });
        (events, hit_deadline)
    };

    // Collapsed cells carry stale stress (they are air now) and the new
    // cavity edges carry fresh spans: compute deferred position recalcs so
    // warnings + crack decals re-evaluate on the normal deferred path
    // (PLAY config — dormancy stress never persists). no_collapse: this
    // pass IS the collapse. NOT queued here — the caller decides when:
    // phase 1 stashes them for phase 2 (queueing 500+ events during the
    // reveal put a multi-second worker-0 recalc batch in front of the
    // montage resync and stretched the black hold ~5s); phase 2 queues
    // immediately (montage over).
    let recalcs: Vec<((i32, i32, i32), i32)> = events
        .iter()
        .map(|ev| {
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
            (center, (max_cheb + 6).min(chunk_size as i32 * 3))
        })
        .collect();

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

    // Per-event locations (UE coords) — verification + forensics: Ghost +
    // BugItGo to center_ue and the cave-in should be visible. Capped to the
    // top EVENT_LOG_CAP by volume (see the const's rationale).
    let ev_vox = |ev: &voxel_core::stress::CollapseEventV2| -> usize {
        ev.slabs.iter().map(|sl| sl.voxels.len()).sum()
    };
    let mut order: Vec<usize> = (0..events.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(ev_vox(&events[i])));
    for &i in order.iter().take(EVENT_LOG_CAP) {
        let ev = &events[i];
        let fall = ev.slabs.first().map(|sl| sl.fall_distance).unwrap_or(0);
        crate::panic_log::note(&format!(
            "[DORMANCY-COLLAPSE-EVENT] {} {}: center_rust=({:.0},{:.0},{:.0}) center_ue=({:.0},{:.0},{:.0}) voxels={} fall={} chunks={}",
            label, i, ev.center.0, ev.center.1, ev.center.2,
            ev.center.0 * world_scale, -ev.center.2 * world_scale, ev.center.1 * world_scale,
            ev_vox(ev), fall, ev.affected_chunks.len()
        ));
    }
    if events.len() > EVENT_LOG_CAP {
        let rest_vox: usize = order.iter().skip(EVENT_LOG_CAP).map(|&i| ev_vox(&events[i])).sum();
        crate::panic_log::note(&format!(
            "[DORMANCY-COLLAPSE-EVENT] {}: ... and {} smaller events ({} voxels) not listed",
            label, events.len() - EVENT_LOG_CAP, rest_vox
        ));
    }

    CascadeOutcome {
        dirty: dirty.into_iter().collect(),
        recalcs,
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
    chunks_unscanned: usize,
    skipped_player: usize,
    skipped_zone: usize,
    /// Span histogram over every air-below (ceiling/deck) cell scanned —
    /// buckets 0-2 / 3-4 / 5-6 / 7-9 / 10-14 / 15-20. THE tuning
    /// instrument for `DORMANCY_SPAN_WEIGHT` / `DORMANCY_MIN_SAFE_SPAN`.
    span_hist: [usize; 6],
}

fn span_bucket(span: u32) -> usize {
    match span {
        0..=2 => 0,
        3..=4 => 1,
        5..=6 => 2,
        7..=9 => 3,
        10..=14 => 4,
        _ => 5,
    }
}

/// Air face-neighbor count for a solid cell (all six via `sample_world`;
/// unloaded reads as not-air).
fn count_air_faces(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    wx: i32,
    wy: i32,
    wz: i32,
    chunk_size: usize,
) -> u32 {
    let offs: [(i32, i32, i32); 6] =
        [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)];
    let mut n = 0u32;
    for &(dx, dy, dz) in &offs {
        if let Some((_, m)) = sample_world(density_fields, wx + dx, wy + dy, wz + dz, chunk_size) {
            if !m.is_solid() {
                n += 1;
            }
        }
    }
    n
}

/// Reduced dormancy stress for an air-below (ceiling/deck) surface cell —
/// mirrors `calc_voxel_stress_v2` under this cell class's invariants:
/// air_decay=1 (surface cell), depth_factor≈1 (depth pressure disabled),
/// floor-protect can't apply (air below) — and pins `unsupported` to 1.0:
/// the dormancy recipe's geological-fatigue assumption. After 1.25M years,
/// hanging rock gets no credit for ground connectivity; struts are the
/// counterplay (relief is subtracted by the caller, and the cascade still
/// halts at alive struts). This removes the ground-connectivity flood
/// entirely — the r1 scan spent its whole budget re-flooding slice halos
/// (422 of 3234 chunks in 4s).
/// Returns (stress_before_relief_and_paint, span).
fn dormancy_cell_stress(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    dcfg: &StressConfig,
    mat: voxel_core::material::Material,
    wx: i32,
    wy: i32,
    wz: i32,
    chunk_size: usize,
    air_faces: u32,
) -> (f32, u32) {
    let hardness = dcfg.material_hardness[mat as u8 as usize];
    if hardness <= 0.0 {
        return (0.0, 0);
    }
    let span = measure_span_from_air(density_fields, wx, wy, wz, chunk_size, 20);
    let span_stress = if span > dcfg.min_safe_span {
        (span - dcfg.min_safe_span) as f32 * dcfg.span_weight
    } else {
        0.0
    };
    let xsec_stress = if air_faces >= dcfg.cross_section_min_faces {
        (air_faces - 1) as f32 * dcfg.cross_section_weight
    } else {
        0.0
    };
    ((span_stress + xsec_stress) / hardness, span)
}

/// Alive struts indexed for the dormancy tier rules (world coords + r²).
struct StrutIndex {
    /// Copper: sacrificed at event start; radius keeps the posthumous
    /// "a little safer than Geological" shadow.
    copper: Vec<(i32, i32, i32, i64)>,
    /// Iron and above: unbreakable, radius fully immune.
    ironplus: Vec<(i32, i32, i32, i64)>,
}

#[derive(Clone, Copy, PartialEq)]
enum StrutCover {
    Open,
    Copper,
    IronPlus,
}

fn build_strut_index(
    support_fields: &HashMap<(i32, i32, i32), SupportField>,
    chunk_size: usize,
) -> StrutIndex {
    let cs = chunk_size as i32;
    let mut idx = StrutIndex { copper: Vec::new(), ironplus: Vec::new() };
    for (&key, sf) in support_fields.iter() {
        for &(lx, ly, lz) in sf.strut_cells() {
            let (lx, ly, lz) = (lx as usize, ly as usize, lz as usize);
            let ty = sf.get(lx, ly, lz);
            if ty == SupportType::None || !sf.is_strut_alive(lx, ly, lz) {
                continue;
            }
            let r = STRUT_TUNING[ty as u8 as usize].radius as i64;
            let entry = (
                key.0 * cs + lx as i32,
                key.1 * cs + ly as i32,
                key.2 * cs + lz as i32,
                r * r,
            );
            if ty == SupportType::Copper {
                idx.copper.push(entry);
            } else {
                idx.ironplus.push(entry);
            }
        }
    }
    idx
}

fn strut_cover(idx: &StrutIndex, wx: i32, wy: i32, wz: i32) -> StrutCover {
    let hit = |s: &(i32, i32, i32, i64)| -> bool {
        let dx = (wx - s.0) as i64;
        let dy = (wy - s.1) as i64;
        let dz = (wz - s.2) as i64;
        dx * dx + dy * dy + dz * dz <= s.3
    };
    if idx.ironplus.iter().any(hit) {
        return StrutCover::IronPlus;
    }
    if idx.copper.iter().any(hit) {
        return StrutCover::Copper;
    }
    StrutCover::Open
}

/// The copper sacrifice: every alive copper strut breaks at dormancy.
/// Returns (chunk, lx, ly, lz) of each broken strut for the UE report.
/// `set()` maintains the strut-cell list, so no manual rebuild is needed.
fn break_all_copper(
    support_fields: &mut HashMap<(i32, i32, i32), SupportField>,
) -> Vec<((i32, i32, i32), usize, usize, usize)> {
    let mut broken = Vec::new();
    for (&key, sf) in support_fields.iter_mut() {
        let cells: Vec<(u8, u8, u8)> = sf.strut_cells().to_vec();
        for (lx, ly, lz) in cells {
            let (lx, ly, lz) = (lx as usize, ly as usize, lz as usize);
            if sf.get(lx, ly, lz) == SupportType::Copper && sf.is_strut_alive(lx, ly, lz) {
                sf.set(lx, ly, lz, SupportType::None);
                broken.push((key, lx, ly, lz));
            }
        }
    }
    broken
}

/// Dormancy scan: recompute stress under the dormancy recipe for every
/// surface-shell cell of every loaded chunk (nearest-first, sliced, rayon
/// per slice, budget-capped). Strut tiers apply per cell: iron+ radius is
/// skipped outright, copper radius evaluates under the softer shadow
/// dials (copper itself was already sacrificed by the caller). The
/// player-adjacent chunks are DROPPED; cells inside `deferred_zone`
/// chunks are returned separately for the post-montage phase. Stored
/// play-stress feeds only the crack tier + painted overlay; never written.
#[allow(clippy::too_many_arguments)]
fn collect_dormancy_seeds(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    stress_fields: &HashMap<(i32, i32, i32), StressField>,
    dcfg: &StressConfig,
    strut_idx: &StrutIndex,
    deferred_zone: &HashSet<(i32, i32, i32)>,
    player_chunk: (i32, i32, i32),
    chunk_size: usize,
    deadline: Instant,
) -> (Vec<OverstressedVoxel>, Vec<OverstressedVoxel>, ScanStats) {
    // Copper-shadow recipe: same live-derived config, softer span dials.
    let copper_shadow = {
        let mut c = dcfg.clone();
        c.span_weight = DORMANCY_COPPER_SPAN_WEIGHT;
        c.min_safe_span = DORMANCY_COPPER_MIN_SAFE_SPAN;
        c
    };
    let copper_shadow = &copper_shadow;
    let mut eligible = Vec::new();
    let mut deferred = Vec::new();
    let mut stats = ScanStats {
        chunks_scanned: 0,
        chunks_unscanned: 0,
        skipped_player: 0,
        skipped_zone: 0,
        span_hist: [0; 6],
    };

    // Candidate list: every chunk with density (span stress needs no prior
    // stress field). Nearest-first so the budget favors what the player
    // will see soonest.
    let cheb = |k: &(i32, i32, i32)| -> i32 {
        (k.0 - player_chunk.0)
            .abs()
            .max((k.1 - player_chunk.1).abs())
            .max((k.2 - player_chunk.2).abs())
    };
    let mut candidates: Vec<(i32, i32, i32)> = Vec::with_capacity(density_fields.len());
    for &key in density_fields.keys() {
        if cheb(&key) <= PLAYER_EXCLUDE_CHEB {
            stats.skipped_player += 1;
            continue;
        }
        candidates.push(key);
    }
    candidates.sort_by_key(|k| (cheb(k), *k));

    let cs = chunk_size;
    let mut idx = 0usize;
    while idx < candidates.len() {
        if Instant::now() >= deadline {
            stats.chunks_unscanned = candidates.len() - idx;
            break;
        }
        let slice = &candidates[idx..(idx + 128).min(candidates.len())];
        idx += slice.len();

        let per_chunk: Vec<((i32, i32, i32), Vec<OverstressedVoxel>, [usize; 6])> = slice
            .par_iter()
            .map(|&key| {
                let df = &density_fields[&key];
                let sf_painted = stress_fields.get(&key);
                let mut out = Vec::new();
                let mut hist = [0usize; 6];
                for z in 0..cs {
                    for y in 0..cs {
                        for x in 0..cs {
                            let cell = df.get(x, y, z);
                            if !cell.material.is_solid() {
                                continue;
                            }
                            let wx = key.0 * cs as i32 + x as i32;
                            let wy = key.1 * cs as i32 + y as i32;
                            let wz = key.2 * cs as i32 + z as i32;
                            // CRACK TIER (union, restored 2026-08-25 after
                            // the span rework dropped it): any cell whose
                            // STORED play-effective stress crosses the
                            // threshold seeds regardless of geometry —
                            // "it was cracking before I slept" must fall,
                            // same as the first live-verified builds. The
                            // cascade's grounded/min-region filters handle
                            // fallability; stored stress is already
                            // strut-relief-aware (relief is baked in as
                            // negative stored surplus).
                            // Strut tiers: iron+ radius is immune; copper
                            // radius evaluates under the softer shadow.
                            let cover = strut_cover(strut_idx, wx, wy, wz);
                            if cover == StrutCover::IronPlus {
                                continue;
                            }
                            let (stored_thr, cell_cfg) = match cover {
                                StrutCover::Copper => {
                                    (DORMANCY_COPPER_STORED_THRESHOLD, copper_shadow)
                                }
                                _ => (DORMANCY_STORED_THRESHOLD, dcfg),
                            };
                            let stored_eff =
                                sf_painted.map(|s| s.effective(x, y, z)).unwrap_or(0.0);
                            if stored_eff >= stored_thr {
                                out.push(OverstressedVoxel {
                                    world_x: wx,
                                    world_y: wy,
                                    world_z: wz,
                                    stress: stored_eff,
                                });
                                continue;
                            }
                            // SPAN RECIPE: only cells with AIR BELOW can
                            // fall (ceilings, bridge decks). Floors and
                            // walls could never pass the cascade's
                            // grounded filter anyway.
                            let below_air = if y > 0 {
                                !df.get(x, y - 1, z).material.is_solid()
                            } else {
                                matches!(
                                    sample_world(density_fields, wx, wy - 1, wz, cs),
                                    Some((_, m)) if !m.is_solid()
                                )
                            };
                            if !below_air {
                                continue;
                            }
                            let air_faces = count_air_faces(density_fields, wx, wy, wz, cs);
                            let (stress, span) = dormancy_cell_stress(
                                density_fields, cell_cfg, cell.material, wx, wy, wz, cs,
                                air_faces,
                            );
                            hist[span_bucket(span)] += 1;
                            let painted =
                                sf_painted.map(|s| s.painted(x, y, z)).unwrap_or(0.0);
                            let eff = stress + painted;
                            if eff < DORMANCY_COLLAPSE_THRESHOLD {
                                continue;
                            }
                            out.push(OverstressedVoxel {
                                world_x: wx,
                                world_y: wy,
                                world_z: wz,
                                stress: eff,
                            });
                            // Slab thickness: seed straight up into the
                            // roof (same chunk; stop at air = thin roof).
                            // Duplicates with a stored-tier push of the
                            // same cell are harmless (cascade dedups).
                            for dy in 1..DORMANCY_SEED_DEPTH_LAYERS {
                                let uy = y + dy;
                                if uy >= cs || !df.get(x, uy, z).material.is_solid() {
                                    break;
                                }
                                out.push(OverstressedVoxel {
                                    world_x: wx,
                                    world_y: wy + dy as i32,
                                    world_z: wz,
                                    stress: eff,
                                });
                            }
                        }
                    }
                }
                (key, out, hist)
            })
            .collect();

        for (key, cells, hist) in per_chunk {
            for (i, n) in hist.iter().enumerate() {
                stats.span_hist[i] += n;
            }
            if deferred_zone.contains(&key) {
                stats.skipped_zone += 1;
                deferred.extend(cells);
            } else {
                stats.chunks_scanned += 1;
                eligible.extend(cells);
            }
        }
    }

    (eligible, deferred, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::material::Material;

    const CHUNK_SIZE: usize = 16;
    const FIELD_SIZE: usize = 17;
    const FAR_FUTURE: Duration = Duration::from_secs(3600);

    fn solid_density(mat: Material) -> DensityField {
        let mut df = DensityField::new(FIELD_SIZE);
        for sample in df.samples.iter_mut() {
            sample.density = 1.0;
            sample.material = mat;
        }
        df
    }

    /// Carve an air box [min..max] (inclusive) into a chunk's density.
    fn carve(df: &mut DensityField, min: (usize, usize, usize), max: (usize, usize, usize)) {
        for z in min.2..=max.2 {
            for y in min.1..=max.1 {
                for x in min.0..=max.0 {
                    let s = df.get_mut(x, y, z);
                    s.density = -1.0;
                    s.material = Material::Air;
                }
            }
        }
    }

    /// A single-chunk world with a wide flat room carved in it: air from
    /// y=3..=6, x/z 2..=13 — the roof at y=7 has up to ~6 voxels of clear
    /// span at its center.
    fn room_world(
        mat: Material,
    ) -> (
        HashMap<(i32, i32, i32), DensityField>,
        HashMap<(i32, i32, i32), StressField>,
        HashMap<(i32, i32, i32), SupportField>,
    ) {
        let mut df_map = HashMap::new();
        let mut df = solid_density(mat);
        carve(&mut df, (2, 3, 2), (13, 6, 13));
        df_map.insert((0, 0, 0), df);
        let mut sf_map = HashMap::new();
        sf_map.insert((0, 0, 0), StressField::new(FIELD_SIZE));
        let mut su_map = HashMap::new();
        su_map.insert((0, 0, 0), SupportField::new(FIELD_SIZE));
        (df_map, sf_map, su_map)
    }

    fn cfg(span_weight: f32, min_safe_span: u32) -> StressConfig {
        let mut c = StressConfig::default();
        c.span_weight = span_weight;
        c.min_safe_span = min_safe_span;
        c.lateral_transfer_factor = 0.7;
        c.vertical_transfer_factor = 0.95;
        c.support_propagation_iterations = 2;
        c.ground_threshold = 0.80;
        c
    }

    fn scan(
        df: &HashMap<(i32, i32, i32), DensityField>,
        sf: &HashMap<(i32, i32, i32), StressField>,
        su: &HashMap<(i32, i32, i32), SupportField>,
        c: &StressConfig,
        zone: &HashSet<(i32, i32, i32)>,
        player: (i32, i32, i32),
    ) -> (Vec<OverstressedVoxel>, Vec<OverstressedVoxel>, ScanStats) {
        let idx = build_strut_index(su, CHUNK_SIZE);
        collect_dormancy_seeds(
            df, sf, c, &idx, zone, player, CHUNK_SIZE,
            Instant::now() + FAR_FUTURE,
        )
    }

    #[test]
    fn wide_roof_seeds_under_span_recipe_and_not_under_play_recipe() {
        let (df, sf, su) = room_world(Material::Granite);
        let none = HashSet::new();

        // Span model alive (cranked so material hardness can't matter):
        // the roof over the wide room must produce seeds.
        let alive = cfg(0.5, 2);
        let (seeds, _, _) = scan(&df, &sf, &su, &alive, &none, (10, 10, 10));
        assert!(
            !seeds.is_empty(),
            "wide room roof should seed under the span recipe"
        );
        // Every seed should be roof material sitting ABOVE the room's air
        // (y >= 7), never the floor below it.
        for v in &seeds {
            assert!(
                v.world_y > 6,
                "seed at y={} — floor/wall below the room must not seed",
                v.world_y
            );
        }

        // Play-like config (span dead): the same geometry produces nothing.
        let dead = cfg(0.001, 333);
        let (seeds, _, _) = scan(&df, &sf, &su, &dead, &none, (10, 10, 10));
        assert!(
            seeds.is_empty(),
            "with the span model dead the roof must not seed (got {})",
            seeds.len()
        );
    }

    #[test]
    fn narrow_corridor_does_not_seed() {
        // 3-wide corridor: span from any roof cell <= 2 = min_safe_span.
        let mut df_map = HashMap::new();
        let mut df = solid_density(Material::Granite);
        carve(&mut df, (7, 3, 2), (9, 6, 13));
        df_map.insert((0, 0, 0), df);
        let mut sf_map = HashMap::new();
        sf_map.insert((0, 0, 0), StressField::new(FIELD_SIZE));
        let mut su_map = HashMap::new();
        su_map.insert((0, 0, 0), SupportField::new(FIELD_SIZE));

        let alive = cfg(0.5, 2);
        let none = HashSet::new();
        let (seeds, _, _) = scan(&df_map, &sf_map, &su_map, &alive, &none, (10, 10, 10));
        assert!(
            seeds.is_empty(),
            "narrow corridor roof must stay below threshold (got {} seeds)",
            seeds.len()
        );
    }

    #[test]
    fn painted_stress_seeds_surface_cells() {
        // Span dead — only the painted overlay drives (via stored
        // effective, so the STORED threshold applies). Painted 0.44 stays
        // under, 0.46 seeds.
        let (df, mut sf, su) = room_world(Material::Granite);
        {
            let f = sf.get_mut(&(0, 0, 0)).unwrap();
            f.set_painted(5, 7, 5, 0.44);
            f.set_painted(9, 7, 9, 0.46);
        }
        let dead = cfg(0.001, 333);
        let none = HashSet::new();
        let (seeds, _, _) = scan(&df, &sf, &su, &dead, &none, (10, 10, 10));
        assert_eq!(seeds.len(), 1, "only the 0.46 painted cell should seed");
        assert_eq!(
            (seeds[0].world_x, seeds[0].world_y, seeds[0].world_z),
            (9, 7, 9)
        );
    }

    #[test]
    fn stored_crack_tier_seeds_regardless_of_geometry() {
        // A FLOOR cell (solid below — the span recipe would never touch it)
        // whose STORED play-effective stress crosses the threshold must
        // still seed: "it was cracking before I slept" falls.
        let (df, mut sf, su) = room_world(Material::Granite);
        {
            let f = sf.get_mut(&(0, 0, 0)).unwrap();
            f.set(5, 2, 5, 0.44); // floor cell, just under — must NOT seed
            f.set(9, 2, 9, 0.46); // floor cell, over — must seed via union
        }
        let dead = cfg(0.001, 333);
        let none = HashSet::new();
        let (seeds, _, _) = scan(&df, &sf, &su, &dead, &none, (10, 10, 10));
        assert_eq!(seeds.len(), 1, "only the 0.46 stored-stress cell seeds");
        assert_eq!(
            (seeds[0].world_x, seeds[0].world_y, seeds[0].world_z),
            (9, 2, 9)
        );
    }

    #[test]
    fn roof_seeds_pull_upper_layers() {
        // An over-threshold ceiling cell also seeds cells straight up
        // (slab thickness). The room roof is at y=7; layers land at 8, 9.
        let (df, sf, su) = room_world(Material::Granite);
        let alive = cfg(0.5, 2);
        let none = HashSet::new();
        let (seeds, _, _) = scan(&df, &sf, &su, &alive, &none, (10, 10, 10));
        assert!(
            seeds.iter().any(|v| v.world_y == 8),
            "second slab layer (y=8) should be seeded"
        );
        assert!(
            seeds.iter().any(|v| v.world_y == 9),
            "third slab layer (y=9) should be seeded"
        );
    }

    #[test]
    fn player_adjacent_chunks_are_dropped_entirely() {
        let (df0, sf0, su0) = room_world(Material::Granite);
        let (df5, sf5, su5) = room_world(Material::Granite);
        let mut df: HashMap<_, _> = df0;
        df.insert((5, 0, 0), df5[&(0, 0, 0)].clone());
        let mut sf = sf0;
        sf.insert((5, 0, 0), sf5[&(0, 0, 0)].clone());
        let mut su = su0;
        su.insert((5, 0, 0), su5[&(0, 0, 0)].clone());

        // Player at (1,0,0): chunk (0,0,0) is cheb 1 <= 2 -> dropped;
        // chunk (5,0,0) is cheb 4 -> eligible.
        let alive = cfg(0.5, 2);
        let none = HashSet::new();
        let (seeds, deferred, stats) = scan(&df, &sf, &su, &alive, &none, (1, 0, 0));
        assert_eq!(stats.skipped_player, 1);
        assert!(deferred.is_empty());
        assert!(!seeds.is_empty());
        assert!(
            seeds.iter().all(|v| v.world_x >= 5 * CHUNK_SIZE as i32),
            "all seeds must come from the far chunk"
        );
    }

    #[test]
    fn zone_seeds_deferred() {
        let (dfa, sfa, sua) = room_world(Material::Granite);
        let mut df: HashMap<_, _> = HashMap::new();
        let mut sf = HashMap::new();
        let mut su = HashMap::new();
        for &k in &[(4, 0, 0), (12, 0, 0)] {
            df.insert(k, dfa[&(0, 0, 0)].clone());
            sf.insert(k, sfa[&(0, 0, 0)].clone());
            su.insert(k, sua[&(0, 0, 0)].clone());
        }
        let alive = cfg(0.5, 2);
        let mut zone = HashSet::new();
        zone.insert((4, 0, 0));

        let (seeds, deferred, stats) = scan(&df, &sf, &su, &alive, &zone, (40, 40, 40));
        assert_eq!(stats.skipped_zone, 1);
        assert!(!deferred.is_empty(), "zone chunk's roof seeds are deferred");
        assert!(
            deferred.iter().all(|v| v.world_x < 5 * CHUNK_SIZE as i32),
            "deferred seeds come from the zone chunk"
        );
        assert!(!seeds.is_empty(), "the plain chunk still seeds");
        assert!(
            seeds.iter().all(|v| v.world_x >= 12 * CHUNK_SIZE as i32),
            "eligible seeds come only from the non-zone chunk"
        );
    }

    #[test]
    fn recalc_trickle_paces_and_respects_pending() {
        use crate::store::{ChunkStore, DormancyPhase2Pending};
        let store = Arc::new(RwLock::new(ChunkStore::new(6)));
        {
            let mut s = store.write().unwrap();
            s.dormancy_recalc_backlog = (0..20).map(|i| ((i, 0, 0), 10)).collect();
            s.dormancy_recalc_total = 20;
            s.dormancy_recalc_started = Some(Instant::now());
        }
        // Feeds immediately when never fed before.
        assert!(try_trickle_dormancy_recalcs(&store));
        assert_eq!(store.read().unwrap().dormancy_recalc_backlog.len(), 12);
        // Within the interval: no feed.
        assert!(!try_trickle_dormancy_recalcs(&store));
        assert_eq!(store.read().unwrap().dormancy_recalc_backlog.len(), 12);
        // Force the interval to have elapsed.
        store.write().unwrap().dormancy_recalc_last_feed =
            Some(Instant::now() - Duration::from_secs(1));
        assert!(try_trickle_dormancy_recalcs(&store));
        assert_eq!(store.read().unwrap().dormancy_recalc_backlog.len(), 4);
        // A pending phase 2 (montage in progress) holds the trickle.
        {
            let mut s = store.write().unwrap();
            s.dormancy_recalc_last_feed = Some(Instant::now() - Duration::from_secs(1));
            s.dormancy_phase2_pending = Some(DormancyPhase2Pending {
                seeds: Vec::new(),
                pending_remesh: Vec::new(),
                pending_recalcs: Vec::new(),
                player_chunk: (0, 0, 0),
                armed_at: Instant::now(),
            });
        }
        assert!(!try_trickle_dormancy_recalcs(&store));
        assert_eq!(store.read().unwrap().dormancy_recalc_backlog.len(), 4);
    }

    #[test]
    fn iron_strut_radius_is_immune() {
        let (df, sf, mut su) = room_world(Material::Granite);
        // Iron strut in the roof over the room center.
        su.get_mut(&(0, 0, 0)).unwrap().set(8, 8, 8, SupportType::Iron);
        let alive = cfg(0.5, 2);
        let none = HashSet::new();
        let (seeds, _, _) = scan(&df, &sf, &su, &alive, &none, (10, 10, 10));
        let r = STRUT_TUNING[SupportType::Iron as u8 as usize].radius as i64;
        let r2 = r * r;
        assert!(
            seeds.iter().all(|v| {
                let dx = (v.world_x - 8) as i64;
                let dy = (v.world_y - 8) as i64;
                let dz = (v.world_z - 8) as i64;
                dx * dx + dy * dy + dz * dz > r2
            }),
            "no seed may sit inside the iron strut's radius"
        );
    }

    #[test]
    fn copper_shadow_softens_and_copper_breaks() {
        // 6-wide corridor: span ~3 from roof center. At cfg(0.6, 1) the
        // open roof seeds ((3-1)*0.6/0.8 = 1.5); under the copper shadow
        // ((3-1)*0.25/0.8 = 0.625) it survives.
        let mut df_map = HashMap::new();
        let mut df = solid_density(Material::Granite);
        carve(&mut df, (5, 3, 2), (10, 6, 13));
        df_map.insert((0, 0, 0), df);
        let mut sf_map = HashMap::new();
        sf_map.insert((0, 0, 0), StressField::new(FIELD_SIZE));
        let mut su_map = HashMap::new();
        su_map.insert((0, 0, 0), SupportField::new(FIELD_SIZE));

        let alive = cfg(0.6, 1);
        let none = HashSet::new();
        let (seeds, _, _) = scan(&df_map, &sf_map, &su_map, &alive, &none, (10, 10, 10));
        assert!(!seeds.is_empty(), "unstrutted corridor roof must seed");

        // Copper strut mid-corridor: its (large) radius shadows the roof.
        su_map.get_mut(&(0, 0, 0)).unwrap().set(8, 5, 8, SupportType::Copper);
        let (seeds, _, _) = scan(&df_map, &sf_map, &su_map, &alive, &none, (10, 10, 10));
        assert!(
            seeds.is_empty(),
            "copper shadow must hold the corridor roof (got {} seeds)",
            seeds.len()
        );

        // And the sacrifice: break_all_copper clears copper, spares iron.
        su_map.get_mut(&(0, 0, 0)).unwrap().set(2, 2, 2, SupportType::Iron);
        let broken = break_all_copper(&mut su_map);
        assert_eq!(broken.len(), 1, "exactly the copper strut breaks");
        let sf = su_map.get(&(0, 0, 0)).unwrap();
        assert_eq!(sf.get(8, 5, 8), SupportType::None, "copper cleared");
        assert_eq!(sf.get(2, 2, 2), SupportType::Iron, "iron untouched");
    }
}
