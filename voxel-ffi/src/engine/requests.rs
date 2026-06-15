use std::sync::atomic::{AtomicU64, Ordering};


use crate::convert::ue_chunk_to_rust;
use crate::pathing::StashedPathResult;
use crate::types::*;

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
        // Priority results first (mine batch expansions)
        if let Some(r) = self.priority_results.lock().unwrap().pop_front() {
            return Some(r);
        }
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
                None
            }
            Ok(WorkerResult::MorphMeshes { step, total_steps, meshes }) => {
                if let Ok(mut mq) = self.morph_results.lock() {
                    mq.push_back(MorphStepResult { step, total_steps, meshes });
                }
                None
            }
            Ok(WorkerResult::ScanComplete { json_report }) => {
                if let Ok(mut sc) = self.scan_complete.lock() {
                    *sc = Some(json_report);
                }
                // Don't expose to the FfiResult pipeline; UE polls via voxel_poll_scan_result
                None
            }
            Ok(WorkerResult::ForceSpawnPoolComplete { json_report }) => {
                if let Ok(mut fc) = self.force_spawn_complete.lock() {
                    *fc = Some(json_report);
                }
                None
            }
            Ok(WorkerResult::LavaQuench { obsidian, scoria, drained_water }) => {
                // Re-dispatch to the worker thread so the voxel writes +
                // remesh happen off the API/game thread. The worker emits
                // ChunkMesh results normally; UE picks them up via poll.
                if !obsidian.is_empty() || !scoria.is_empty() || !drained_water.is_empty() {
                    let _ = self.mine_tx.send(WorkerRequest::ApplyLavaQuench {
                        obsidian, scoria, drained_water,
                    });
                }
                None
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
                None
            }
            Ok(WorkerResult::StrutsBroken { struts }) => {
                if let Ok(mut stash) = self.strut_broken_stash.lock() {
                    stash.extend(struts);
                }
                None
            }
            Ok(other) => Some(other),
            Err(_) => None,
        }
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
