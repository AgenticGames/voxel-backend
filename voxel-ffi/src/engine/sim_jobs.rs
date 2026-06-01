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

use super::{MorphStepResult, SleepCompleteData, VoxelEngine};

impl VoxelEngine {
    /// Set spider nest positions for fossilization during sleep.
    pub fn set_sleep_nests(&self, positions: Vec<(i32, i32, i32)>) {
        if let Ok(mut sc) = self.sleep_config.write() {
            sc.nest_positions = positions;
        }
    }

    /// Set spider corpse positions for fossilization during sleep.
    pub fn set_sleep_corpses(&self, positions: Vec<(i32, i32, i32)>) {
        if let Ok(mut sc) = self.sleep_config.write() {
            sc.corpse_positions = positions;
        }
    }

    /// Set the tagged top-POI chunks to simulate regardless of chunk_radius
    /// (Rust chunk coords). UE sets these before start_sleep (after ensuring
    /// they're streamed/generated) so distant points of interest get REAL
    /// per-voxel reveal data in the morph manifest. Must be set AFTER
    /// update_sleep_config (which resets the config) — same ordering rule as
    /// nests/corpses.
    pub fn set_sleep_extra_chunks(&self, positions: Vec<(i32, i32, i32)>) {
        if let Ok(mut sc) = self.sleep_config.write() {
            sc.extra_sim_chunks = positions;
        }
    }

    /// Start a deep sleep cycle. Sends request through the mine channel
    /// (which has exclusive write-lock priority).
    /// Returns 1 on success, 0 if queue full.
    pub fn start_sleep(&self, player_chunk: (i32, i32, i32), sleep_count: u32) -> u32 {
        let sc = self.sleep_config.read().unwrap().clone();
        match self.mine_tx.try_send(WorkerRequest::Sleep {
            player_chunk,
            sleep_count,
            sleep_config: sc,
        }) {
            Ok(()) => {
                // Trace dispatch so a stall is bracketed: this line (+ mine-queue
                // depth) vs the worker's "[SLEEP_TRACE] enter Sleep handler" bound
                // the dequeue latency. If the latter never appears, the stall
                // monitor's [WORKER_STALL] snapshots show which workers wedged.
                crate::panic_log::note(&format!(
                    "[SLEEP_TRACE] dispatched Sleep to mine channel (player_chunk={:?}, mine_q_after={})",
                    player_chunk,
                    self.mine_tx.len()
                ));
                1
            }
            Err(_) => {
                crate::panic_log::note(
                    "[SLEEP_TRACE] StartSleep REJECTED — mine channel full (try_send failed)",
                );
                0
            }
        }
    }

    /// Replace the set of chunks the sleep-montage cinematic is actively
    /// filming. While populated, `ChunkStore::unload` refuses to evict their
    /// density, so the camera planner's voxel queries (rock-vs-air ray clamp)
    /// always have real data instead of "unloaded"(=solid) garbage. Coords are
    /// Rust chunk coords. Replaces any prior set.
    pub fn set_montage_protected(&self, chunks: Vec<(i32, i32, i32)>) {
        if let Ok(mut s) = self.store.write() {
            s.montage_protected = chunks.into_iter().collect();
        }
    }

    /// Clear the montage-protected set (call at montage end so normal streaming
    /// eviction resumes).
    pub fn clear_montage_protected(&self) {
        if let Ok(mut s) = self.store.write() {
            s.montage_protected.clear();
        }
    }

    /// Trigger an aureole-only debug run. Uses the same result polling as sleep.
    pub fn start_aureole_only(&self, player_chunk: (i32, i32, i32)) -> u32 {
        let sc = self.sleep_config.read().unwrap().clone();
        match self.mine_tx.try_send(WorkerRequest::AureoleOnly {
            player_chunk,
            sleep_config: sc,
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Poll for a completed sleep result. Returns None if no sleep has completed yet.
    pub fn poll_sleep_complete(&self) -> Option<SleepCompleteData> {
        let mut sc = self.sleep_complete.lock().ok()?;
        sc.take()
    }

    /// Cache a morph manifest (deserialized once, reused for all 30 steps).
    pub fn set_morph_manifest(&self, json: &str) -> bool {
        match voxel_sleep::ChangeManifest::from_json(json) {
            Ok(m) => {
                *self.morph_manifest.lock().unwrap() = Some(m);
                true
            }
            Err(e) => {
                eprintln!("[MORPH] Failed to cache manifest: {}", e);
                false
            }
        }
    }

    /// Clear cached morph manifest.
    pub fn clear_morph_manifest(&self) {
        *self.morph_manifest.lock().unwrap() = None;
    }

    /// Add `synthesize_growth = true` stubs to the cached morph manifest
    /// for the given UE-space chunk coords. Used by the sleep-montage POI
    /// plays so every POI chunk animates even if it wasn't sleep-affected.
    ///
    /// `ue_sources` (UE world coords) and `max_dist_ue` parameterize the
    /// reveal animation: each voxel's spread = min-distance-to-any-source /
    /// max_dist, normalized [0,1]. Bridges pass 2 sources (anchors) so growth
    /// radiates inward; other POIs pass 1 source (chunk center) for a
    /// radial reveal. Empty `ue_sources` falls back to a y-axis gradient.
    ///
    /// Returns the count of new entries added (chunks not already in
    /// manifest). Returns 0 if no manifest is cached.
    pub fn augment_morph_synthesize_ue_chunks(
        &self,
        ue_chunks: &[FfiChunkCoord],
        ue_sources: &[FfiVec3],
        max_dist_ue: f32,
    ) -> u32 {
        let mut guard = self.morph_manifest.lock().unwrap();
        let manifest = match guard.as_mut() {
            Some(m) => m,
            None => return 0,
        };

        // Convert UE sources to Rust voxel-space tuples (world voxel coords).
        let ws = self.world_scale;
        let growth_sources: Vec<(f32, f32, f32)> = ue_sources
            .iter()
            .map(|s| {
                let v = crate::convert::from_ue_world_pos(s.x, s.y, s.z, ws);
                (v.x, v.y, v.z)
            })
            .collect();
        let growth_source_max_dist = if max_dist_ue > 0.0 { max_dist_ue / ws } else { 0.0 };

        let mut added = 0u32;
        for c in ue_chunks {
            let rust_chunk = crate::convert::ue_chunk_to_rust(c.x, c.y, c.z);
            if !manifest.chunk_deltas.contains_key(&rust_chunk) {
                manifest.chunk_deltas.insert(
                    rust_chunk,
                    voxel_sleep::ChunkDelta {
                        voxel_changes: Vec::new(),
                        support_changes: Vec::new(),
                        synthesize_growth: true,
                        growth_sources: growth_sources.clone(),
                        growth_source_max_dist,
                    },
                );
                added += 1;
            }
        }
        added
    }

    /// Request a morph step. Uses the cached manifest (set via set_morph_manifest).
    ///
    /// Sent through the MINE (priority) channel, not generate_tx. The montage's
    /// morph reveal is latency-critical — on generate_tx the step queued behind
    /// the chunk-generation backlog (P2, FIFO), and that backlog grows as the
    /// montage pins/streams every POI volume, so the per-step wait climbed (the
    /// ~1.3s -> 3.7s morph-step-0 growth). Morph only runs while the player is
    /// asleep, so nothing else uses mine_tx then (no mining / brushes; the sleep
    /// request that started the montage already completed) — zero new contention.
    /// Matches how Sleep / WorldScan are already routed for immediate processing.
    /// The morph's chunks are guaranteed resident before this is called, so
    /// jumping ahead of queued Generate jobs has no ordering hazard.
    /// Returns 1 on success, 0 if queue full.
    pub fn request_morph_step(
        &self,
        chunks: Vec<(i32, i32, i32)>,
        step: u32,
        total_steps: u32,
    ) -> u32 {
        match self.mine_tx.try_send(WorkerRequest::MorphStep {
            chunks,
            step,
            total_steps,
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Poll for a completed morph step result. Returns None if no morph has completed yet.
    pub fn poll_morph_result(&self) -> Option<MorphStepResult> {
        let mut mq = self.morph_results.lock().ok()?;
        mq.pop_front()
    }

    /// Request a world scan. Sent through the mine channel for immediate processing.
    /// Returns 1 on success, 0 if queue full.
    pub fn request_world_scan(&self) -> u32 {
        match self.mine_tx.try_send(WorkerRequest::WorldScan) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Request a world scan with custom configuration. Sent through the mine
    /// channel for immediate processing. Returns 1 on success, 0 if queue full.
    pub fn request_world_scan_with_config(&self, config: ScanConfig) -> u32 {
        match self.mine_tx.try_send(WorkerRequest::WorldScanWithConfig { config }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Poll for a completed scan result. Returns the JSON report string if ready.
    pub fn poll_scan_complete(&self) -> Option<String> {
        let mut sc = self.scan_complete.lock().ok()?;
        sc.take()
    }

    /// Request force-spawning a pool at a UE world position. Sent through the mine channel.
    /// fluid_type: 0=water, 1=lava. Returns 1 on success, 0 if queue full.
    pub fn request_force_spawn_pool(&self, world_x: f32, world_y: f32, world_z: f32, fluid_type: u8) -> u32 {
        match self.mine_tx.try_send(WorkerRequest::ForceSpawnPool {
            world_x,
            world_y,
            world_z,
            fluid_type,
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Request mining a sphere and filling the bottom half with fluid.
    /// Sent through the mine channel. Returns 1 on success, 0 if queue full.
    pub fn mine_and_fill_fluid(&self, world_x: f32, world_y: f32, world_z: f32, radius: f32, fluid_type: u8, world_scale: f32) -> u32 {
        match self.mine_tx.try_send(WorkerRequest::MineAndFillFluid {
            world_x,
            world_y,
            world_z,
            radius,
            fluid_type,
            world_scale,
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Poll for a completed force-spawn pool result. Returns the JSON diagnostics string if ready.
    pub fn poll_force_spawn_complete(&self) -> Option<String> {
        let mut fc = self.force_spawn_complete.lock().ok()?;
        fc.take()
    }

    /// Get current engine statistics.
    pub fn get_stats(&self) -> FfiEngineStats {
        let chunks_loaded = self.store.try_read().map(|s| s.chunks_loaded()).unwrap_or(0);
        FfiEngineStats {
            chunks_loaded: chunks_loaded as u32,
            pending_requests: self.generate_tx.len() as u32,
            completed_results: self.result_rx.len() as u32,
            worker_threads_active: self.workers.len() as u32,
            workers_alive: crate::panic_log::workers_alive() as u32,
            panics_observed: crate::panic_log::panic_count() as u32,
        }
    }

    /// Inject fluid at a UE world position. Computes chunk + local cell automatically.
    /// fluid_type: 1=Water, 2=Lava. Returns 1 on success, 0 on failure.
    pub fn add_fluid(&self, world_x: f32, world_y: f32, world_z: f32,
                     fluid_type: u8, is_source: bool, world_scale: f32,
                     max_flow_dist: u8) -> u32 {
        use crate::convert::from_ue_world_pos;
        use voxel_fluid::cell::FluidType;

        let (chunk_size, eb) = self.config.read()
            .map(|c| (c.chunk_size, c.effective_bounds()))
            .unwrap_or((16, 16.0));
        let cs = eb;

        // Convert UE world pos -> Rust voxel pos
        let rust_pos = from_ue_world_pos(world_x, world_y, world_z, world_scale);

        // Compute chunk coord and local cell
        let cx = (rust_pos.x / cs).floor() as i32;
        let cy = (rust_pos.y / cs).floor() as i32;
        let cz = (rust_pos.z / cs).floor() as i32;

        let lx = ((rust_pos.x - cx as f32 * cs) as i32).clamp(0, chunk_size as i32 - 1) as u8;
        let ly = ((rust_pos.y - cy as f32 * cs) as i32).clamp(0, chunk_size as i32 - 1) as u8;
        let lz = ((rust_pos.z - cz as f32 * cs) as i32).clamp(0, chunk_size as i32 - 1) as u8;

        let ft = FluidType::from_u8(fluid_type);

        match self.fluid_event_tx.try_send(FluidEvent::AddFluid {
            chunk: (cx, cy, cz),
            x: lx,
            y: ly,
            z: lz,
            fluid_type: ft,
            level: voxel_fluid::cell::MAX_LEVEL,
            is_source,
            max_flow_dist: if is_source { max_flow_dist } else { 0 },
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }
}
