//! Worker-thread request handling, decomposed from the former monolithic
//! `worker.rs` into cohesive submodules. This module owns the worker loop and
//! the thin `handle_request` dispatcher; the actual per-request work lives in
//! the handler submodules. Pure code-movement — behavior is unchanged.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};
use dashmap::DashMap;
use voxel_fluid::FluidEvent;
use voxel_gen::config::{GenerationConfig, StressConfig};

use crate::convert::{from_ue_normal, from_ue_world_pos};
use crate::profiler::StreamingProfiler;
use crate::store::ChunkStore;
use crate::types::{WorkerRequest, WorkerResult};

mod brush;
mod dormancy_collapse;
mod generate;
pub mod heartbeat;
mod pathing;
pub mod region_stress;
mod scan_support;
mod seam;
pub(crate) mod sleep_morph; // pub(crate): the FFI (api/sleep.rs) pokes SLEEP_FAR_GO
mod stress;

pub use region_stress::DeferredRegionStress;
pub use seam::PendingSeams;

// Re-exports so external callers (engine.rs) keep `crate::worker::worker_loop`
// and `crate::worker::path_worker_loop` resolving unchanged.
pub use pathing::path_worker_loop;

use seam::batched_seam_pass_mine;
use stress::try_process_stress_queue;

/// R5 (morph-snapshot): a one-time, per-play clone of the morph block's base
/// densities + out-of-block neighbor seam data. A morph step meshes the reveal
/// from THIS instead of taking the store read lock on every step — otherwise the
/// step's read acquisition stalls behind generation's slow-path write lock
/// (150-380ms holds) and the on-screen reveal freezes. Rebuilt when the play's
/// chunk set changes; reset to default at montage end (clear_morph_manifest). The
/// block is montage-protected (never unloaded) and the morph already meshes from
/// a post-sleep t=1 clone, so snapshotting is behavior-preserving.
#[derive(Default)]
pub struct MorphSnapshot {
    pub keys: Vec<(i32, i32, i32)>,
    pub densities: std::collections::HashMap<(i32, i32, i32), voxel_core::density::DensityField>,
    pub neighbor_seams: std::collections::HashMap<(i32, i32, i32), std::sync::Arc<voxel_gen::region_gen::ChunkSeamData>>,

    // ── Recolor fast-path cache (mesh-once + per-step recolor) ───────────────
    // The deep-sleep montage morph is almost entirely a per-voxel MATERIAL flip
    // (metamorphism, hydrothermal ore) — density never moves the DC surface, so
    // the triangles are byte-identical at every reveal step. For a block where
    // EVERY chunk is a pure recolor we mesh ONCE (capturing the base mesh below)
    // and each subsequent step only reassigns per-vertex material + re-buckets —
    // no dual-contouring, no seam-gen. All cleared/reset when `keys` changes.
    /// True iff every chunk in this play is a pure recolor → fast path eligible.
    pub block_recolor: bool,
    /// Set once the base meshes have been captured for the current `keys`.
    pub base_built: bool,
    /// Per-chunk base mesh (Rust space, base + seam quads, normals recalc'd,
    /// PRE-convert / PRE-bucket). Per step we clone it, reassign per-vertex
    /// material by the reveal progress, then convert + bucket.
    pub base_meshes: std::collections::HashMap<(i32, i32, i32), voxel_core::mesh::Mesh>,
    /// Per cached vertex (aligned to `base_meshes[key].vertices`), the recorded change
    /// that drives its recolor — `(spread_distance, old_material, new_material)` — or
    /// None if the vertex's voxel (and its ±1 neighborhood) is unchanged. Computed once
    /// at base build using the SAME ±1-dilated field + round/clamp as the reveal_t bake,
    /// so material flips and the dissolve track the recolor/air boundary identically (no
    /// stale-color rim). Per step we just pick old vs new from the cached change by `t`.
    pub vertex_change: std::collections::HashMap<(i32, i32, i32), Vec<Option<(f32, u8, u8)>>>,
    /// Per-vertex reveal_t (spread-only, t-independent), baked once at base build.
    pub base_reveal_t: std::collections::HashMap<(i32, i32, i32), Vec<f32>>,
    // (Diff-skip is now STATELESS — 2026-08-05 lookahead: each step computes
    // materials at t(step) and t(prev_step from the request) and ships only
    // chunks that differ. The old `last_materials` cache was order-dependent
    // and would corrupt under ahead-of-display step computation.)
    /// Hybrid morph (2026-08-05 playtest 3: "still no change down in the hole
    /// until the montage is over"): chunks whose sign-flip (solid↔air) change
    /// count crosses the GEO threshold — geometry genuinely moves there, which
    /// the frozen-recolor path cannot show. These re-run the faithful per-step
    /// DC morph (interpolated densities) INSIDE the recolor fast path; the
    /// rest keep the cached recolor + diff-skip.
    pub geo_chunks: std::collections::HashSet<(i32, i32, i32)>,
    /// Seam data captured at base build for every in-block chunk — the geo
    /// chunks' per-step seam generation needs their frozen-recolor neighbors'
    /// DC vertices without re-running DC on them.
    pub base_seams: std::collections::HashMap<(i32, i32, i32), std::sync::Arc<voxel_gen::region_gen::ChunkSeamData>>,
}

/// Per-region generation claim used by `handle_generate`'s slow path.
///
/// `gate` is the region-generation mutex, held by the owning worker while it
/// runs the slow path. Non-owners do NOT block on it (blocking idled workers
/// 3.5-5.2s per region during virgin-terrain floods, serializing region
/// generation across the pool — measured 2026-07-17): they park their
/// (chunk, generation) in `waiters` and return to the queue, typically
/// becoming owners of OTHER regions. The owner re-dispatches parked waiters
/// via [`ParkedGenerates`] the moment the region's densities are committed.
///
/// `done` closes the park/drain race: it is set under the `waiters` lock,
/// exactly once, at drain time. A would-be parker that observes `done`
/// re-dispatches itself through the pool instead of parking into a list
/// nobody will drain again.
#[derive(Default)]
pub struct RegionInFlight {
    pub(crate) gate: Mutex<()>,
    pub(crate) waiters: Mutex<RegionWaiters>,
}

#[derive(Default)]
pub(crate) struct RegionWaiters {
    pub(crate) done: bool,
    pub(crate) list: Vec<((i32, i32, i32), u64)>,
}

/// Generate requests parked because their region was mid-generation by
/// another worker (region-convoy fix). The worker loop pops these with
/// priority over fresh queue pulls — parked chunks are the oldest
/// outstanding requests (nearest the player) and, post-commit, mesh via the
/// fast path. A shared pool rather than a channel re-send on purpose: the
/// generate channel is bounded(256) and sits full during exactly the floods
/// that park chunks, so blocking sends from workers could deadlock the pool.
pub struct ParkedGenerates {
    queue: Mutex<std::collections::VecDeque<((i32, i32, i32), u64)>>,
}

impl ParkedGenerates {
    pub fn new() -> Self {
        Self { queue: Mutex::new(std::collections::VecDeque::new()) }
    }

    pub fn push(&self, chunk: (i32, i32, i32), generation: u64) {
        self.queue.lock().unwrap_or_else(|p| p.into_inner()).push_back((chunk, generation));
    }

    pub fn push_batch(&self, items: Vec<((i32, i32, i32), u64)>) {
        self.queue.lock().unwrap_or_else(|p| p.into_inner()).extend(items);
    }

    pub fn pop(&self) -> Option<((i32, i32, i32), u64)> {
        self.queue.lock().unwrap_or_else(|p| p.into_inner()).pop_front()
    }
}

impl Default for ParkedGenerates {
    fn default() -> Self {
        Self::new()
    }
}

/// Counting semaphore bounding how many workers may run the region slow path
/// (generate_region_densities) CONCURRENTLY. Unbounded concurrency maximizes
/// drain throughput but inflates each region's wall-clock 2.5-6x through CPU
/// oversubscription (8 owners x rayon inside each) — and the saved-position
/// restore's async ground-wait teleport has a ~10s deadline that individual
/// region latency must meet. Measured 2026-07-17: uncapped, base_density avg
/// went 1030 -> 2436 ms and the restore reliably missed its deadline (player
/// free-fell through absent ground into an unrecoverable generation chase);
/// pre-convoy serial gen landed it at +8.6-9.9s, right at the edge. Capping
/// at half the pool keeps regions near serial speed while still generating
/// several regions in parallel, and leaves the other workers free to mesh
/// parked fast-path chunks.
pub struct SlowPathPermits {
    state: Mutex<usize>,
    cv: std::sync::Condvar,
}

impl SlowPathPermits {
    pub fn new(max_concurrent: usize) -> Self {
        Self { state: Mutex::new(max_concurrent), cv: std::sync::Condvar::new() }
    }

    /// Returns false if `shutdown` was raised while waiting — the caller must
    /// NOT run the slow path. A plain condvar wait here serialized teardown:
    /// permit waiters couldn't see the shutdown flag, so each woke only when
    /// a peer finished its whole region and then ran its OWN region before
    /// exiting — 6-8 workers x 10-20s chase regions = the 133s PIE-exit hang
    /// observed 2026-07-18.
    pub(crate) fn acquire(&self, shutdown: &AtomicBool) -> bool {
        let mut avail = self.state.lock().unwrap_or_else(|p| p.into_inner());
        while *avail == 0 {
            if shutdown.load(Ordering::Relaxed) {
                return false;
            }
            let (g, _) = self
                .cv
                .wait_timeout(avail, Duration::from_millis(100))
                .unwrap_or_else(|p| p.into_inner());
            avail = g;
        }
        if shutdown.load(Ordering::Relaxed) {
            return false;
        }
        *avail -= 1;
        true
    }

    pub(crate) fn release(&self) {
        let mut avail = self.state.lock().unwrap_or_else(|p| p.into_inner());
        *avail += 1;
        drop(avail);
        self.cv.notify_one();
    }
}

/// Drain a region's parked waiters into the shared re-dispatch pool and
/// retire the map entry. Called by the region owner at density-commit time,
/// by a gate claimant whose retry fast path hit (region committed by a
/// predecessor while it raced for the gate), and by the idle-sweep safety
/// net below (owner panicked mid-region). Setting `done` under the waiters
/// lock closes the race against concurrent parkers.
pub(crate) fn drain_region_waiters(
    rk: (i32, i32, i32),
    entry: &Arc<RegionInFlight>,
    regions_in_flight: &DashMap<(i32, i32, i32), Arc<RegionInFlight>>,
    parked_generates: &ParkedGenerates,
) {
    let drained = {
        let mut w = entry.waiters.lock().unwrap_or_else(|p| p.into_inner());
        w.done = true;
        std::mem::take(&mut w.list)
    };
    regions_in_flight.remove(&rk);
    if !drained.is_empty() {
        parked_generates.push_batch(drained);
    }
}

/// Shared context threaded into every request handler. Holds borrowed
/// references to the worker's channels, stores, and config so each extracted
/// handler can bind the exact locals the original match-arm body referenced.
pub(crate) struct HandlerCtx<'a> {
    pub result_tx: &'a Sender<WorkerResult>,
    pub store: &'a Arc<RwLock<ChunkStore>>,
    pub config: &'a Arc<RwLock<GenerationConfig>>,
    pub stress_config: &'a Arc<RwLock<StressConfig>>,
    pub generation_counters: &'a Arc<DashMap<(i32, i32, i32), AtomicU64>>,
    pub world_scale: f32,
    pub fluid_event_tx: &'a Sender<FluidEvent>,
    pub profiler: &'a Arc<StreamingProfiler>,
    pub worker_id: usize,
    pub generate_rx: &'a Receiver<WorkerRequest>,
    pub mine_rx: &'a Receiver<WorkerRequest>,
    // Requeue handle for the mid-generate preemption drain (try_handle_mine):
    // non-Mine requests it pulls go BACK on the channel instead of vanishing.
    pub mine_tx: &'a Sender<WorkerRequest>,
    pub morph_manifest: &'a Arc<Mutex<Option<voxel_sleep::ChangeManifest>>>,
    pub morph_snapshot: &'a Arc<Mutex<MorphSnapshot>>,
    // Direct push target for finished morph steps (2026-08-05): the shared
    // engine queue voxel_poll_morph_result pops. Bypasses result_tx — a morph
    // result on the main channel sat behind fluid/gen traffic that UE drains
    // under throttled reveal budgets (~0.5-1s added latency per step).
    pub morph_results: &'a Arc<Mutex<std::collections::VecDeque<crate::engine::MorphStepResult>>>,
    pub regions_in_flight: &'a Arc<DashMap<(i32, i32, i32), Arc<RegionInFlight>>>,
    pub crystal_anchors: &'a Arc<Mutex<crate::crystal_anchors::CrystalAnchorManager>>,
    pub deferred_region_stress: &'a Arc<DeferredRegionStress>,
    pub pending_seams: &'a Arc<seam::PendingSeams>,
    pub parked_generates: &'a Arc<ParkedGenerates>,
    pub slow_path_permits: &'a Arc<SlowPathPermits>,
    pub shutdown: &'a Arc<AtomicBool>,
}

/// Worker thread main loop. Each worker pulls from shared channels.
pub fn worker_loop(
    shutdown: Arc<AtomicBool>,
    generation_paused: Arc<AtomicBool>,
    generate_rx: Receiver<WorkerRequest>,
    mine_rx: Receiver<WorkerRequest>,
    mine_tx: Sender<WorkerRequest>,
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
    morph_snapshot: Arc<Mutex<MorphSnapshot>>,
    morph_results: Arc<Mutex<std::collections::VecDeque<crate::engine::MorphStepResult>>>,
    regions_in_flight: Arc<DashMap<(i32, i32, i32), Arc<RegionInFlight>>>,
    crystal_anchors: Arc<Mutex<crate::crystal_anchors::CrystalAnchorManager>>,
    // Per-worker activity heartbeat (this worker writes only `heartbeats[worker_id]`).
    // Read by the stall monitor to pinpoint a wedged worker when a sleep request
    // never gets dequeued. See `heartbeat.rs`.
    heartbeats: Arc<Vec<heartbeat::WorkerHeartbeat>>,
    deferred_region_stress: Arc<DeferredRegionStress>,
    pending_seams: Arc<seam::PendingSeams>,
    parked_generates: Arc<ParkedGenerates>,
    slow_path_permits: Arc<SlowPathPermits>,
) {
    let hb = &heartbeats[worker_id];
    // Orphan-store sweep state (worker 0 only): region-grain generation
    // commits all 216 chunk densities per region, but UE only ever unloads
    // the chunks it made actors for — the rest linger in the store forever
    // (observed 3327 stored vs 318 UE-loaded, multi-GB store, and the store
    // Drop alone took ~90s of the 2026-07-18 PIE-exit hang). A stored chunk
    // with no generation_counters entry was never UE-requested (or was
    // already unloaded) = orphan. Two-round grace so chunks about to be
    // requested by the advancing ring aren't evicted prematurely.
    let mut last_orphan_sweep = Instant::now();
    let mut orphan_candidates: std::collections::HashSet<(i32, i32, i32)> =
        std::collections::HashSet::new();
    while !shutdown.load(Ordering::Relaxed) {
        // Priority 1: mine requests (non-blocking). Stamp the heartbeat around
        // the handler so a stall snapshot names exactly what wedged us here.
        if let Ok(req) = mine_rx.try_recv() {
            let (act, coord) = heartbeat::classify(&req);
            hb.enter(act, coord);
            handle_request(
                req, &result_tx, &store, &config, &stress_config, &generation_counters,
                world_scale, &fluid_event_tx, &profiler, worker_id, &generate_rx, &mine_rx, &mine_tx, &morph_manifest, &morph_snapshot, &morph_results,
                &regions_in_flight, &crystal_anchors, &deferred_region_stress, &pending_seams, &parked_generates, &slow_path_permits, &shutdown,
            );
            hb.idle();
            continue;
        }

        // Priority 1.5: deferred stress recalculation (only worker 0 handles this)
        if worker_id == 0 {
            hb.enter(heartbeat::activity::STRESS, (0, 0, 0));
            let did_stress = try_process_stress_queue(&store, &stress_config, &config, &result_tx, &fluid_event_tx, world_scale);
            hb.idle();
            if did_stress {
                continue;
            }
            // Priority 1.6: dormancy-collapse phase 2 — the filmed-block
            // seeds deferred at curtain-up, released when CleanupMontage
            // clears montage protection (voxel_montage_clear_protected).
            hb.enter(heartbeat::activity::STRESS, (0, 0, 0));
            let did_phase2 = dormancy_collapse::try_run_phase2(
                &store, &stress_config, &config, &result_tx, &fluid_event_tx, world_scale,
            );
            hb.idle();
            if did_phase2 {
                continue;
            }
            // Priority 1.7: trickle-feed parked dormancy recalc events into
            // the live stress queue (a few per interval — the burst version
            // was a top source of the post-montage lag).
            if dormancy_collapse::try_trickle_dormancy_recalcs(&store) {
                continue;
            }
        }

        // Reveal pause: while a sleep-montage morph reveal is on screen, UE pauses
        // generation so the morph's parallel (rayon) mesh-gen gets the full core
        // count. The POI gen "storm" otherwise steals cores and stutters the reveal.
        // Mine (morph/sleep) above still runs every loop — we only stop pulling
        // generates here; queued gens stay buffered and resume the instant UE clears
        // the flag between plays.
        if generation_paused.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(2));
            continue;
        }

        // Priority 1.75: re-dispatched parked generates (region-convoy fix).
        // These are the oldest outstanding requests — their region's
        // densities were just committed by an owner, so they mesh via the
        // fast path. Handle them before pulling fresh queue work.
        if let Some((chunk, generation)) = parked_generates.pop() {
            let req = WorkerRequest::Generate { chunk, generation };
            let (act, coord) = heartbeat::classify(&req);
            hb.enter(act, coord);
            handle_request(
                req, &result_tx, &store, &config, &stress_config, &generation_counters,
                world_scale, &fluid_event_tx, &profiler, worker_id, &generate_rx, &mine_rx, &mine_tx, &morph_manifest, &morph_snapshot, &morph_results,
                &regions_in_flight, &crystal_anchors, &deferred_region_stress, &pending_seams, &parked_generates, &slow_path_permits, &shutdown,
            );
            hb.idle();
            continue;
        }

        // Priority 2: generate requests (blocking with timeout)
        let idle_start = Instant::now();
        match generate_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(req) => {
                profiler.record_worker_idle(worker_id, idle_start.elapsed());
                let (act, coord) = heartbeat::classify(&req);
                hb.enter(act, coord);
                handle_request(
                    req, &result_tx, &store, &config, &stress_config, &generation_counters,
                    world_scale, &fluid_event_tx, &profiler, worker_id, &generate_rx, &mine_rx, &mine_tx, &morph_manifest, &morph_snapshot, &morph_results,
                    &regions_in_flight, &crystal_anchors, &deferred_region_stress, &pending_seams, &parked_generates, &slow_path_permits, &shutdown,
                );
                hb.idle();
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                profiler.record_worker_idle(worker_id, idle_start.elapsed());
                // Orphan-store sweep (see state above). Worker 0, queue idle,
                // at most once per 60s. `unload()` preserves modified-chunk
                // snapshots and no-ops on montage-protected keys, so eviction
                // has the exact semantics of a normal UE-driven unload.
                if worker_id == 0 && last_orphan_sweep.elapsed() >= Duration::from_secs(60) {
                    last_orphan_sweep = Instant::now();
                    let current: std::collections::HashSet<(i32, i32, i32)> = {
                        let s = store.read().unwrap();
                        s.density_fields
                            .keys()
                            .filter(|k| !generation_counters.contains_key(*k))
                            .copied()
                            .collect()
                    };
                    let evict: Vec<(i32, i32, i32)> =
                        current.intersection(&orphan_candidates).copied().collect();
                    if !evict.is_empty() {
                        let mut evicted = 0usize;
                        {
                            let mut s = store.write().unwrap();
                            for k in &evict {
                                // Re-check under the write lock — a request
                                // may have landed since the read snapshot.
                                if generation_counters.contains_key(k) {
                                    continue;
                                }
                                s.unload(*k);
                                evicted += 1;
                            }
                        }
                        for k in &evict {
                            let _ = fluid_event_tx.send(FluidEvent::ChunkUnloaded { chunk: *k });
                        }
                        let remain = store.read().unwrap().density_fields.len();
                        eprintln!(
                            "[ORPHAN_SWEEP] evicted {} never-requested stored chunks ({} remain)",
                            evicted, remain
                        );
                    }
                    orphan_candidates = current;
                }
                // Region-convoy safety net: normal owners drain their
                // waiters at region commit; a worker that panicked
                // mid-region leaves them parked forever. With the queue
                // idle, re-dispatch the waiters of any region that has no
                // active owner (gate free). Entries WITHOUT waiters are
                // left alone — a freshly created entry's owner may simply
                // not have claimed its gate yet.
                if !regions_in_flight.is_empty() {
                    let keys: Vec<(i32, i32, i32)> =
                        regions_in_flight.iter().map(|e| *e.key()).collect();
                    for rk in keys {
                        let entry = match regions_in_flight.get(&rk) {
                            Some(e) => Arc::clone(e.value()),
                            None => continue,
                        };
                        let guard = match entry.gate.try_lock() {
                            Ok(g) => g,
                            Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
                            Err(std::sync::TryLockError::WouldBlock) => continue,
                        };
                        let has_waiters = !entry
                            .waiters
                            .lock()
                            .unwrap_or_else(|p| p.into_inner())
                            .list
                            .is_empty();
                        if has_waiters {
                            drain_region_waiters(rk, &entry, &regions_in_flight, &parked_generates);
                        }
                        drop(guard);
                    }
                }
                // Generate queue is idle — flush deferred bulk-load seam
                // passes FIRST (player-visible geometry; one bounded batch
                // per idle window so mine_rx is re-checked between batches),
                // then fall through to deferred region stress (VFX-only).
                if !pending_seams.is_empty() {
                    hb.enter(heartbeat::activity::SEAM, (0, 0, 0));
                    let cfg = config.read().unwrap().clone();
                    seam::drain_pending_seams(
                        &pending_seams, &cfg, &store, &result_tx, &fluid_event_tx, world_scale,
                    );
                    hb.idle();
                    continue;
                }
                // Drain one deferred region-stress compute (VFX crack/dust
                // pre-population). Capped concurrency + rayon-parallel
                // internals keep this from starving a mine request that lands
                // mid-compute; during the load flood the Timeout branch never
                // fires, so the loading window is never taxed (the inline
                // version froze the loading-screen counter for ~5s while all
                // 8 workers chewed regions).
                if !deferred_region_stress.is_empty() {
                    hb.enter(heartbeat::activity::STRESS, (0, 0, 0));
                    region_stress::try_process_deferred_region_stress(
                        &deferred_region_stress, &store, &stress_config, &config,
                    );
                    hb.idle();
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }
}

/// Check mine queue and handle any pending mine request. Returns true if one was handled.
pub(crate) fn try_handle_mine(
    mine_rx: &Receiver<WorkerRequest>,
    mine_requeue_tx: &Sender<WorkerRequest>,
    result_tx: &Sender<WorkerResult>,
    store: &Arc<RwLock<ChunkStore>>,
    config: &Arc<RwLock<GenerationConfig>>,
    stress_config: &Arc<RwLock<StressConfig>>,
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
                let outcome = if request.mode == 0 {
                    crate::mining::mine_sphere(&mut s, center, radius, &cfg, world_scale)
                } else {
                    let normal = from_ue_normal(request.normal_x, request.normal_y, request.normal_z);
                    crate::mining::mine_peel(&mut s, center, normal, radius, &cfg, world_scale)
                };
                let dirty_keys: Vec<(i32, i32, i32)> = outcome.meshes.into_iter().map(|(k, _)| k).collect();
                let mined = outcome.mined;
                // Fix B: crystal recompute only for chunks where a material flip
                // actually occurred. Boundary-sync chunks (density tweaks only) keep
                // their existing crystal placements — no recompute needed.
                for &key in &outcome.flipped_chunks {
                    if let Some(density) = s.density_fields.get(&key) {
                        let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                        let placements = voxel_gen::compute_crystals(coord, density, &cfg);
                        s.crystal_placements.insert(key, placements);
                    }
                }
                // Queue position-based stress recalculation at mine point.
                // ⚠ Must match `brush::handle_mine` — this preemption path drains
                // the SAME WorkerRequest::Mine mid-generate, so a mine landing
                // while a worker is busy must get the identical scan sphere.
                // (Was hardcoded `+4` until 2026-07-31, which silently shrank the
                // cinematic scan radius for any mine that arrived during region
                // generation and desynced it from the debug visualizer.)
                let stress_center = (center.x as i32, center.y as i32, center.z as i32);
                let scan_buffer = stress_config.read().unwrap().mining_stress_scan_buffer;
                let stress_radius = radius as i32 + scan_buffer as i32;
                s.queue_stress_dirty(stress_center, stress_radius);
                drop(s);
                let _ = result_tx.send(WorkerResult::MinedMaterials { mined });
                batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
                return true;
            }
            // ⚠ NOT ours to eat (2026-08-05): this preemption drain used to
            // `_ => {}` DISCARD any non-Mine request it pulled off the shared
            // mine channel — a MorphStep (or PlaceSupport / Sleep / WorldScan)
            // arriving while a worker was mid-Generate simply VANISHED. Cost:
            // the dormancy reveal's first step died to the 4s timeout in every
            // run that overlapped the montage's POI gen burst. Requeue at the
            // tail instead; the main worker loops handle it properly.
            other => {
                crate::panic_log::note("[MINE-PREEMPT] non-mine request pulled mid-generate — requeued");
                let _ = mine_requeue_tx.try_send(other);
            }
        }
    }
    false
}

/// Dedicated morph-step lane (2026-08-13). Morph steps used to ride the
/// shared mine channel: during a procedural sleep's aftermath (halo
/// generation bursts, stress recalcs on the solidified block, the quench
/// cascade) every pool worker sat on multi-second jobs and the reveal
/// starved in bursts — heartbeats showed mine_wait 2.0s with 7 workers on
/// 2-5s jobs while the camera visibly froze 3 times (one 4.3s). One thread
/// owns `morph_rx`; steps never queue behind anything else. Mirrors the
/// path-lane precedent (path_tx). The loop reuses `handle_request` so the
/// morph handler itself stays single-sourced.
pub fn morph_worker_loop(
    shutdown: Arc<AtomicBool>,
    morph_rx: Receiver<WorkerRequest>,
    generate_rx: Receiver<WorkerRequest>,
    mine_rx: Receiver<WorkerRequest>,
    mine_tx: Sender<WorkerRequest>,
    result_tx: Sender<WorkerResult>,
    store: Arc<RwLock<ChunkStore>>,
    config: Arc<RwLock<GenerationConfig>>,
    stress_config: Arc<RwLock<StressConfig>>,
    generation_counters: Arc<DashMap<(i32, i32, i32), AtomicU64>>,
    world_scale: f32,
    fluid_event_tx: Sender<FluidEvent>,
    profiler: Arc<StreamingProfiler>,
    morph_manifest: Arc<Mutex<Option<voxel_sleep::ChangeManifest>>>,
    morph_snapshot: Arc<Mutex<MorphSnapshot>>,
    morph_results: Arc<Mutex<std::collections::VecDeque<crate::engine::MorphStepResult>>>,
    regions_in_flight: Arc<DashMap<(i32, i32, i32), Arc<RegionInFlight>>>,
    crystal_anchors: Arc<Mutex<crate::crystal_anchors::CrystalAnchorManager>>,
    deferred_region_stress: Arc<DeferredRegionStress>,
    pending_seams: Arc<seam::PendingSeams>,
    parked_generates: Arc<ParkedGenerates>,
    slow_path_permits: Arc<SlowPathPermits>,
) {
    // Distinct id in [MORPH-REQ] logs so "dequeued by worker 98" reads as
    // the dedicated lane at a glance.
    const MORPH_WORKER_ID: usize = 98;
    while !shutdown.load(Ordering::Relaxed) {
        match morph_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(req) => {
                handle_request(
                    req, &result_tx, &store, &config, &stress_config, &generation_counters,
                    world_scale, &fluid_event_tx, &profiler, MORPH_WORKER_ID, &generate_rx, &mine_rx, &mine_tx,
                    &morph_manifest, &morph_snapshot, &morph_results, &regions_in_flight, &crystal_anchors,
                    &deferred_region_stress, &pending_seams, &parked_generates, &slow_path_permits, &shutdown,
                );
            }
            Err(_) => {} // timeout — re-check shutdown
        }
    }
}

/// Thin dispatcher: builds a `HandlerCtx` and routes each `WorkerRequest`
/// variant to its extracted handler. An early `return` inside a handler is
/// equivalent to falling off the matched arm — the dispatcher does no
/// post-arm work.
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
    mine_tx: &Sender<WorkerRequest>,
    morph_manifest: &Arc<Mutex<Option<voxel_sleep::ChangeManifest>>>,
    morph_snapshot: &Arc<Mutex<MorphSnapshot>>,
    morph_results: &Arc<Mutex<std::collections::VecDeque<crate::engine::MorphStepResult>>>,
    regions_in_flight: &Arc<DashMap<(i32, i32, i32), Arc<RegionInFlight>>>,
    crystal_anchors: &Arc<Mutex<crate::crystal_anchors::CrystalAnchorManager>>,
    deferred_region_stress: &Arc<DeferredRegionStress>,
    pending_seams: &Arc<seam::PendingSeams>,
    parked_generates: &Arc<ParkedGenerates>,
    slow_path_permits: &Arc<SlowPathPermits>,
    shutdown: &Arc<AtomicBool>,
) {
    let ctx = HandlerCtx {
        result_tx,
        store,
        config,
        stress_config,
        generation_counters,
        world_scale,
        fluid_event_tx,
        profiler,
        worker_id,
        generate_rx,
        mine_rx,
        mine_tx,
        morph_manifest,
        morph_snapshot,
        morph_results,
        regions_in_flight,
        crystal_anchors,
        deferred_region_stress,
        pending_seams,
        parked_generates,
        slow_path_permits,
        shutdown,
    };
    match req {
        // ComputePath is handled exclusively by the dedicated path-worker
        // (see `path_worker_loop`). If it ever lands here it means routing
        // confusion — silently drop rather than panic.
        WorkerRequest::ComputePath { .. } => {}
        WorkerRequest::PriorityGenerate { chunk, generation } => {
            // Priority requests (spawn/restore ground chunks) never park —
            // see the is_priority blocking arm in handle_generate.
            generate::handle_generate(&ctx, chunk, generation, true);
        }
        WorkerRequest::Generate { chunk, generation } => {
            generate::handle_generate(&ctx, chunk, generation, false);
        }
        WorkerRequest::Flatten { base_x, base_y, base_z, host_material } => {
            brush::handle_flatten(&ctx, base_x, base_y, base_z, host_material);
        }
        WorkerRequest::FlattenBatch { tiles } => {
            brush::handle_flatten_batch(&ctx, tiles);
        }
        WorkerRequest::BuildingFlattenBatch { buildings } => {
            brush::handle_building_flatten_batch(&ctx, buildings);
        }
        WorkerRequest::BuildingFlatten { base_x, base_y, base_z, base_y_float, host_material, footprint_x, footprint_z, clearance_voxels } => {
            brush::handle_building_flatten(&ctx, base_x, base_y, base_z, base_y_float, host_material, footprint_x, footprint_z, clearance_voxels);
        }
        WorkerRequest::Mine { request } => {
            brush::handle_mine(&ctx, request);
        }
        WorkerRequest::MineAndFillFluid { world_x, world_y, world_z, radius, fluid_type, world_scale: ws } => {
            brush::handle_mine_and_fill_fluid(&ctx, world_x, world_y, world_z, radius, fluid_type, ws);
        }
        WorkerRequest::BrushSphere { request } => {
            brush::handle_brush_sphere(&ctx, request);
        }
        WorkerRequest::BrushTunnel { points, radius, material } => {
            brush::handle_brush_tunnel(&ctx, points, radius, material);
        }
        WorkerRequest::BrushPlaceMushroom { center_rust, kind, search_radius, scale, yaw } => {
            brush::handle_brush_place_mushroom(&ctx, center_rust, kind, search_radius, scale, yaw);
        }
        WorkerRequest::BrushPlaceMushroomSphere { center_rust, radius, density, clustering, kind, seed } => {
            brush::handle_brush_place_mushroom_sphere(&ctx, center_rust, radius, density, clustering, kind, seed);
        }
        WorkerRequest::BrushEraseMushroomSphere { center_rust, radius, kind_filter } => {
            brush::handle_brush_erase_mushroom_sphere(&ctx, center_rust, radius, kind_filter);
        }
        WorkerRequest::BrushFormation { center_rust, formation_type, material, height, radius } => {
            brush::handle_brush_formation(&ctx, center_rust, formation_type, material, height, radius);
        }
        WorkerRequest::ForceChunkResync { chunk } => {
            brush::handle_force_chunk_resync(&ctx, chunk);
        }
        WorkerRequest::ForceChunkResyncBatch { chunks } => {
            brush::handle_force_chunk_resync_batch(&ctx, chunks);
        }
        WorkerRequest::BrushCavernStamp { chunk_origin, extent, decorate, fluids, seed } => {
            brush::handle_brush_cavern_stamp(&ctx, chunk_origin, extent, decorate, fluids, seed);
        }
        WorkerRequest::BrushFormationStamp { center_rust, radius, seed } => {
            brush::handle_brush_formation_stamp(&ctx, center_rust, radius, seed);
        }
        WorkerRequest::BrushBox { center_rust, half_ext_rust, yaw_rad, op, material } => {
            brush::handle_brush_box(&ctx, center_rust, half_ext_rust, yaw_rad, op, material);
        }
        WorkerRequest::BrushCylinder { center_rust, radius, height, op, material } => {
            brush::handle_brush_cylinder(&ctx, center_rust, radius, height, op, material);
        }
        WorkerRequest::BrushSmooth { center_rust, radius, iterations, strength } => {
            brush::handle_brush_smooth(&ctx, center_rust, radius, iterations, strength);
        }
        WorkerRequest::BrushOrePaint {
            center_rust,
            radius,
            cluster_size,
            min_spacing,
            channel_prob,
            channel_length,
            channel_radius,
            density,
            seed,
            weights,
        } => {
            brush::handle_brush_ore_paint(&ctx, center_rust, radius, cluster_size, min_spacing, channel_prob, channel_length, channel_radius, density, seed, weights);
        }
        WorkerRequest::BrushPaintStress { center_rust, radius, amount, cap, op, falloff } => {
            brush::handle_brush_paint_stress(&ctx, center_rust, radius, amount, cap, op, falloff);
        }
        WorkerRequest::BrushClearAllPaintedStress => {
            brush::handle_brush_clear_all_painted_stress(&ctx);
        }
        WorkerRequest::BrushUndo => {
            brush::handle_brush_undo(&ctx);
        }
        WorkerRequest::BrushNoise { center_rust, radius, frequency, strength, seed } => {
            brush::handle_brush_noise(&ctx, center_rust, radius, frequency, strength, seed);
        }
        WorkerRequest::BrushFluidSphere { center_rust, radius, fluid_type, is_source, op, max_flow_dist } => {
            brush::handle_brush_fluid_sphere(&ctx, center_rust, radius, fluid_type, is_source, op, max_flow_dist);
        }
        WorkerRequest::BrushFluidBox { center_rust, half_ext_rust, fluid_type, is_source, op, max_flow_dist } => {
            brush::handle_brush_fluid_box(&ctx, center_rust, half_ext_rust, fluid_type, is_source, op, max_flow_dist);
        }
        WorkerRequest::BrushFluidRiver { points, radius, fluid_type, is_source, op, max_flow_dist } => {
            brush::handle_brush_fluid_river(&ctx, points, radius, fluid_type, is_source, op, max_flow_dist);
        }
        WorkerRequest::ApplyLavaQuench { obsidian, scoria, drained_water } => {
            brush::handle_apply_lava_quench(&ctx, obsidian, scoria, drained_water);
        }
        WorkerRequest::Unload { chunk } => {
            brush::handle_unload(&ctx, chunk);
        }
        WorkerRequest::PlaceSupport { world_x, world_y, world_z, support_type } => {
            scan_support::handle_place_support(&ctx, world_x, world_y, world_z, support_type);
        }
        WorkerRequest::RemoveSupport { world_x, world_y, world_z } => {
            scan_support::handle_remove_support(&ctx, world_x, world_y, world_z);
        }
        WorkerRequest::Sleep { player_chunk, sleep_count, sleep_config: sc } => {
            sleep_morph::handle_sleep(&ctx, player_chunk, sleep_count, sc);
        }
        WorkerRequest::AureoleOnly { player_chunk, sleep_config: sc } => {
            sleep_morph::handle_aureole_only(&ctx, player_chunk, sc);
        }
        WorkerRequest::MorphStep { chunks, step, total_steps, prev_step } => {
            sleep_morph::handle_morph_step(&ctx, chunks, step, total_steps, prev_step);
        }
        WorkerRequest::WorldScan => {
            scan_support::handle_world_scan(&ctx);
        }
        WorkerRequest::WorldScanWithConfig { config: scan_config } => {
            scan_support::handle_world_scan_with_config(&ctx, scan_config);
        }
        WorkerRequest::ForceSpawnPool { world_x, world_y, world_z, fluid_type } => {
            scan_support::handle_force_spawn_pool(&ctx, world_x, world_y, world_z, fluid_type);
        }
    }
}

#[cfg(test)]
mod region_park_tests {
    use super::*;
    use dashmap::DashMap;

    #[test]
    fn drain_moves_waiters_to_pool_sets_done_and_retires_entry() {
        let map: DashMap<(i32, i32, i32), Arc<RegionInFlight>> = DashMap::new();
        let entry = map
            .entry((0, 0, 0))
            .or_insert_with(|| Arc::new(RegionInFlight::default()))
            .clone();
        entry.waiters.lock().unwrap().list.push(((1, 2, 3), 7));
        entry.waiters.lock().unwrap().list.push(((1, 2, 4), 7));

        let pool = ParkedGenerates::new();
        drain_region_waiters((0, 0, 0), &entry, &map, &pool);

        assert!(entry.waiters.lock().unwrap().done, "drain must set done");
        assert!(map.get(&(0, 0, 0)).is_none(), "drain must retire the map entry");
        assert_eq!(pool.pop(), Some(((1, 2, 3), 7)), "waiters re-dispatch FIFO");
        assert_eq!(pool.pop(), Some(((1, 2, 4), 7)));
        assert_eq!(pool.pop(), None);
    }

    #[test]
    fn parker_protocol_refuses_to_park_after_done() {
        // Mirrors the parker-side check in generate.rs: a would-be parker
        // that observes `done` must NOT push into the (never again drained)
        // list — it re-dispatches itself through the pool instead.
        let entry = Arc::new(RegionInFlight::default());
        entry.waiters.lock().unwrap().done = true;

        let parked = {
            let mut w = entry.waiters.lock().unwrap();
            if w.done {
                false
            } else {
                w.list.push(((0, 0, 0), 1));
                true
            }
        };
        assert!(!parked);
        assert!(entry.waiters.lock().unwrap().list.is_empty());
    }
}
