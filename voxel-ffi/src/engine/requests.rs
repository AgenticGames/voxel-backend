use std::collections::HashSet;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};

use crossbeam_channel::{bounded, Receiver, Sender};
use dashmap::DashMap;
use voxel_fluid::FluidConfig;
use voxel_fluid::FluidEvent;
use voxel_core::stress::StressField;
use voxel_core::world_scan::ScanConfig;
use voxel_gen::config::{
    BandedIronConfig, CrystalConfig, FormationConfig, GenerationConfig, GeodeConfig, HostRockConfig,
    KimberlitePipeConfig, MineConfig, NoiseConfig, OreConfig, OreCrystalConfig, OreVeinParams,
    PoolConfig, StressConfig, SulfideBlobConfig, WormConfig,
};

use crate::convert::ue_chunk_to_rust;
use crate::pathing::{
    build_request_from_ue, FfiPathNode, FfiPathRequest, FfiPathResult, PathResultStore,
    StashedPathResult,
};
use crate::profiler::StreamingProfiler;
use crate::store::ChunkStore;
use crate::types::*;
use crate::worker::{path_worker_loop, worker_loop};

use super::{MorphStepResult, SleepCompleteData, VoxelEngine, PATH_RESULT_TTL_SECS};

impl VoxelEngine {
    /// Queue a single chunk for generation. Coords are UE space, converted internally.
    /// Returns 1 on success, 0 if queue full.
    pub fn request_generate(&self, cx: i32, cy: i32, cz: i32) -> u32 {
        let key = ue_chunk_to_rust(cx, cy, cz);
        // Bump the counter BEFORE sending. If we sent first and stored the new
        // value after, a worker could pick the job off the queue between the
        // send and the store, observe the old counter value, fail the
        // stale-detection check at worker.rs:528, and silently `return`.
        // No result reaches UE → PendingRequests leaks the coord forever.
        let generation = self
            .generation_counters
            .entry(key)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed)
            + 1;

        match self.generate_tx.try_send(WorkerRequest::Generate {
            chunk: key,
            generation,
        }) {
            Ok(()) => {
                self.profiler.record_request(key);
                1
            }
            Err(_) => 0,
        }
    }

    /// Queue multiple chunks for generation. Returns count successfully queued.
    pub fn request_generate_batch(&self, chunks: &[(i32, i32, i32)]) -> u32 {
        let mut count = 0;
        for &(cx, cy, cz) in chunks {
            count += self.request_generate(cx, cy, cz);
        }
        count
    }

    /// Queue a mine request. Returns 1 on success, 0 if queue full.
    pub fn request_mine(&self, request: FfiMineRequest) -> u32 {
        match self.mine_tx.try_send(WorkerRequest::Mine { request }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Request unloading a chunk's cached data. Coords are UE space.
    pub fn request_unload(&self, cx: i32, cy: i32, cz: i32) -> u32 {
        let key = ue_chunk_to_rust(cx, cy, cz);
        match self
            .generate_tx
            .try_send(WorkerRequest::Unload { chunk: key })
        {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Cancel any pending generation for a chunk by bumping its generation counter.
    /// Workers will see the stale generation and skip. Coords are UE space.
    pub fn cancel_chunk(&self, cx: i32, cy: i32, cz: i32) {
        let key = ue_chunk_to_rust(cx, cy, cz);
        self.generation_counters
            .entry(key)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Non-blocking poll for a completed result. Returns None if nothing ready.
    /// SleepComplete results are intercepted and stored internally; they are
    /// retrieved via `poll_sleep_complete()` instead.
    pub fn requeue_result(&self, result: WorkerResult) {
        self.priority_results.lock().unwrap().push_back(result);
    }

    pub fn poll_result(&self) -> Option<WorkerResult> {
        // [POLL-SLOW] section timing (2026-08-19): the UE game thread calls this
        // every result drain; profiler caught single calls blocking 112-188ms
        // during generation storms (P50 1us / P99 0.1ms — bimodal lock hit).
        // When a call exceeds 20ms, note WHICH section ate it so the contended
        // lock is named instead of guessed. Cost when fast: two Instant reads.
        let t_start = std::time::Instant::now();

        // Flush re-dispatches an earlier poll couldn't place (mine channel was
        // full). try_send only — the game thread must never block here; see
        // `pending_mine_redispatch` in engine/mod.rs for the 2026-08-13
        // gridlock this prevents. Stop at the first Full so ordering holds.
        {
            let mut pending = self.pending_mine_redispatch.lock().unwrap();
            while let Some(req) = pending.pop_front() {
                match self.mine_tx.try_send(req) {
                    Ok(()) => {}
                    Err(crossbeam_channel::TrySendError::Full(req)) => {
                        pending.push_front(req);
                        break;
                    }
                    Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
                        pending.clear();
                        break;
                    }
                }
            }
        }
        let t_flush = t_start.elapsed();

        // Priority results first (mine batch expansions)
        if let Some(r) = self.priority_results.lock().unwrap().pop_front() {
            let total = t_start.elapsed();
            if total.as_millis() >= 20 {
                crate::panic_log::note(&format!(
                    "[POLL-SLOW] prio-hit total={:.1}ms flush={:.1}ms prio_lock={:.1}ms",
                    total.as_secs_f64() * 1000.0,
                    t_flush.as_secs_f64() * 1000.0,
                    (total - t_flush).as_secs_f64() * 1000.0));
            }
            return Some(r);
        }
        let t_prio = t_start.elapsed();

        // Intercept-skip loop (2026-08-19): intercepted kinds (sleep/morph/
        // scan/quench/path/strut) used to make this fn return None, which the
        // FFI surfaced as null and UE's ProcessResults read as "queue empty" -
        // ONE intercepted item ended the whole tick's result drain and applies
        // landed in bursts. Keep consuming until a UE-facing result or a
        // genuinely empty channel. Every intercepted arm is a cheap stash.
        let mut intercepted = 0u32;
        let polled = loop {
            match self.result_rx.try_recv() {
            Ok(WorkerResult::SleepComplete {
                chunks_changed,
                voxels_metamorphosed,
                minerals_grown,
                supports_degraded,
                collapses_triggered,
                acid_dissolved,
                veins_deposited,
                voxels_enriched,
                formations_grown,
                sulfide_dissolved,
                coal_matured,
                diamonds_formed,
                voxels_silicified,
                nests_fossilized,
                channels_eroded,
                corpses_fossilized,
                lava_solidified,
                profile_report,
                aureole_glimpse_pos,
                aureole_showcase_block,
                manifest_json,
                lava_cells,
                surface_changed_cells,
                surface_step_activity,
            }) => {
                // ─── Block 1: invalidate predictor cache + record event ───
                // Real sleep result is authoritative; prediction is now stale.
                if let Ok(mut pc) = self.predict_cache.write() {
                    *pc = None;
                }
                // Record a SleepCompleted event so the drift loop knows to
                // re-score modified chunks within ~16 ms (vs. waiting for
                // its next 2 s tick).
                let manifest_bytes = manifest_json.len() as u32;
                let _ = self.world_memory.record_event(
                    voxel_world_memory::WorldEvent::sleep_completed(
                        chunks_changed,
                        manifest_bytes,
                    ),
                );

                if let Ok(mut sc) = self.sleep_complete.lock() {
                    *sc = Some(SleepCompleteData {
                        chunks_changed,
                        voxels_metamorphosed,
                        minerals_grown,
                        supports_degraded,
                        collapses_triggered,
                        acid_dissolved,
                        veins_deposited,
                        voxels_enriched,
                        formations_grown,
                        sulfide_dissolved,
                        coal_matured,
                        diamonds_formed,
                        voxels_silicified,
                        nests_fossilized,
                        channels_eroded,
                        corpses_fossilized,
                        lava_solidified,
                        profile_report,
                        aureole_glimpse_pos,
                        aureole_showcase_block,
                        manifest_json,
                        lava_cells,
                        surface_changed_cells,
                        surface_step_activity,
                    });
                }
                // Don't expose to the FfiResult pipeline; UE polls via voxel_poll_sleep_result
                intercepted += 1;
                continue;
            }
            Ok(WorkerResult::MorphMeshes { step, total_steps, meshes }) => {
                if let Ok(mut mq) = self.morph_results.lock() {
                    mq.push_back(MorphStepResult { step, total_steps, meshes });
                }
                intercepted += 1;
                continue;
            }
            Ok(WorkerResult::ScanComplete { json_report }) => {
                if let Ok(mut sc) = self.scan_complete.lock() {
                    *sc = Some(json_report);
                }
                // Don't expose to the FfiResult pipeline; UE polls via voxel_poll_scan_result
                intercepted += 1;
                continue;
            }
            Ok(WorkerResult::ForceSpawnPoolComplete { json_report }) => {
                if let Ok(mut fc) = self.force_spawn_complete.lock() {
                    *fc = Some(json_report);
                }
                intercepted += 1;
                continue;
            }
            Ok(WorkerResult::LavaQuench { obsidian, scoria, drained_water }) => {
                // Re-dispatch to the worker thread so the voxel writes +
                // remesh happen off the API/game thread. The worker emits
                // ChunkMesh results normally; UE picks them up via poll.
                //
                // try_send ONLY — this runs on the game thread inside the
                // result drain, and a blocking send here caused the 2026-08-13
                // engine-wide gridlock (see `pending_mine_redispatch`). On a
                // full channel the request is deferred and flushed by the next
                // poll_result call; the quench is never lost.
                if !obsidian.is_empty() || !scoria.is_empty() || !drained_water.is_empty() {
                    let req = WorkerRequest::ApplyLavaQuench {
                        obsidian, scoria, drained_water,
                    };
                    if let Err(crossbeam_channel::TrySendError::Full(req)) = self.mine_tx.try_send(req) {
                        crate::panic_log::note("[QUENCH-DEFER] mine channel full — lava-quench re-dispatch deferred to next poll");
                        self.pending_mine_redispatch.lock().unwrap().push_back(req);
                    }
                }
                intercepted += 1;
                continue;
            }
            Ok(WorkerResult::PathComputed { request_id, status, nodes_ue }) => {
                // Stash for `voxel_path_poll`; UE polls per-request_id rather
                // than from the mesh result pipeline. TTL-prune happens here.
                if let Ok(mut store) = self.path_results.lock() {
                    use voxel_path::PathStatus;
                    let status_enum = match status {
                        0 => PathStatus::Success,
                        1 => PathStatus::NoPath,
                        2 => PathStatus::MaxNodesReached,
                        3 => PathStatus::PartiallyUnloaded,
                        4 => PathStatus::InvalidEndpoint,
                        _ => PathStatus::NoPath,
                    };
                    store.stash(StashedPathResult {
                        request_id,
                        status: status_enum,
                        nodes_ue,
                        stash_time: std::time::Instant::now(),
                    });
                    store.prune(PATH_RESULT_TTL_SECS);
                }
                intercepted += 1;
                continue;
            }
            Ok(WorkerResult::StrutsBroken { struts }) => {
                if let Ok(mut stash) = self.strut_broken_stash.lock() {
                    stash.extend(struts);
                }
                intercepted += 1;
                continue;
            }
            Ok(other) => break Some(other),
            Err(_) => break None,
            }
        };
        let total = t_start.elapsed();
        if total.as_millis() >= 20 {
            crate::panic_log::note(&format!(
                "[POLL-SLOW] total={:.1}ms flush={:.1}ms prio_lock={:.1}ms recv_arm={:.1}ms intercepted={} got={}",
                total.as_secs_f64() * 1000.0,
                t_flush.as_secs_f64() * 1000.0,
                (t_prio - t_flush).as_secs_f64() * 1000.0,
                (total - t_prio).as_secs_f64() * 1000.0,
                intercepted,
                if polled.is_some() { "result" } else { "empty" }));
        }
        polled
    }

    /// Drain all pending broken-strut events. Called by UE per-frame via
    /// `voxel_take_struts_broken`. The stash is emptied — UE owns the
    /// returned data.
    pub fn drain_struts_broken(&self) -> Vec<crate::types::FfiStrutBroken> {
        let mut stash = match self.strut_broken_stash.lock() {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        std::mem::take(&mut *stash)
    }
}
