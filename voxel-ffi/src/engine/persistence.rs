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

use super::{cell_has_solid_center, VoxelEngine};

impl VoxelEngine {
    // ── Profiler API ──

    /// Enable or disable the streaming profiler.
    pub fn profiler_set_enabled(&self, enabled: bool) {
        self.profiler.set_enabled(enabled);
    }

    /// Check if profiler is enabled.
    pub fn profiler_is_enabled(&self) -> bool {
        self.profiler.is_enabled()
    }

    /// Begin a new profiling session. Resets metrics, captures config snapshot.
    pub fn profiler_begin_session(&self) -> u64 {
        let config_snapshot = if let Ok(cfg) = self.config.read() {
            format!(
                "seed={}\nchunk_size={}\nworkers={}\nworld_scale={:.1}\nregion_size={}\n\
                 cavern_freq={:.4}\ncavern_thresh={:.2}\ndetail_octaves={}\ndetail_persistence={:.2}\nwarp_amplitude={:.1}\n\
                 worms_per_region={:.1}\nworm_radius_min={:.1}\nworm_radius_max={:.1}\nworm_step_length={:.1}\nworm_max_steps={}\nworm_falloff_power={:.1}\n\
                 ore_domain_warp_strength={:.2}\nore_warp_frequency={:.4}\nore_edge_falloff={:.4}\nore_detail_weight={:.2}\n\
                 mesh_smooth_iterations={}\nmesh_smooth_strength={:.2}\nmesh_boundary_smooth={:.2}\nmesh_recalc_normals={}",
                cfg.seed, cfg.chunk_size,
                self.workers.len(), self.world_scale, cfg.region_size,
                cfg.noise.cavern_frequency, cfg.noise.cavern_threshold,
                cfg.noise.detail_octaves, cfg.noise.detail_persistence, cfg.noise.warp_amplitude,
                cfg.worm.worms_per_region, cfg.worm.radius_min, cfg.worm.radius_max,
                cfg.worm.step_length, cfg.worm.max_steps, cfg.worm.falloff_power,
                cfg.ore.ore_domain_warp_strength, cfg.ore.ore_warp_frequency,
                cfg.ore.ore_edge_falloff, cfg.ore.ore_detail_weight,
                cfg.mesh_smooth_iterations, cfg.mesh_smooth_strength,
                cfg.mesh_boundary_smooth, cfg.mesh_recalc_normals,
            )
        } else {
            "(config unavailable)".to_string()
        };
        self.profiler.begin_session(config_snapshot)
    }

    /// End the current profiling session.
    pub fn profiler_end_session(&self) {
        self.profiler.end_session();
    }

    /// Generate a plain-text profiling report.
    pub fn profiler_get_report(&self) -> String {
        self.profiler.generate_report()
    }

    /// Generate report as C string for FFI. Caller must free with voxel_profiler_free_report.
    pub fn profiler_get_report_cstr(&self) -> *mut std::ffi::c_char {
        self.profiler.generate_report_cstr()
    }

    /// Get a reference to the profiler (for FFI poll instrumentation).
    pub fn profiler(&self) -> &Arc<StreamingProfiler> {
        &self.profiler
    }

    /// Build a multi-line plain-text diagnostic dump for a single chunk —
    /// density boundary slices, mesh stats, coplanar-tri analysis, edit
    /// state. Used by the UE creative-mode "Chunk Diagnostic" component
    /// so the user can copy-paste a problem chunk's stats into chat.
    pub fn build_chunk_diagnostic(&self, chunk: (i32, i32, i32)) -> String {
        // Backward-compatible wrapper — derives UE coord via the inverse
        // transform. Prefer build_chunk_diagnostic_with_ue when calling
        // from FFI (the caller has the UE coord directly).
        let ue_chunk = crate::convert::rust_chunk_to_ue(chunk.0, chunk.1, chunk.2);
        self.build_chunk_diagnostic_with_ue(chunk, ue_chunk)
    }

    /// Same as `build_chunk_diagnostic` but accepts both the Rust (HashMap-key)
    /// chunk coord AND the original UE chunk coord, so the dump can label
    /// them correctly without relying on the inverse transform. The two
    /// must satisfy `rust = ue_chunk_to_rust(ue)`.
    pub fn build_chunk_diagnostic_with_ue(
        &self,
        rust_chunk: (i32, i32, i32),
        ue_chunk: (i32, i32, i32),
    ) -> String {
        let cfg = self.config.read().unwrap();
        let chunk_size = cfg.chunk_size;
        let voxel_scale = cfg.voxel_scale();
        drop(cfg);
        let store = self.store.read().unwrap();
        crate::diagnostics::build_chunk_diagnostic_with_ue(
            &store,
            rust_chunk,
            ue_chunk,
            chunk_size,
            voxel_scale,
            self.world_scale,
        )
    }

    /// Gracefully shut down all workers and wait for them to finish.
    // ── Save/Load ──────────────────────────────────────────────────────

    /// Export world modification data as a binary buffer for saving.
    /// Returns the serialized bytes (caller manages the memory).
    pub fn export_save_data(&self) -> Vec<u8> {
        let store = self.store.read().unwrap();
        let mut data = store.collect_save_data();
        // Crystal Growth Bridge state lives outside the store; merge it in.
        data.crystal_anchors_json = self.crystal_anchors.lock().unwrap().to_json_string();
        // Block 1 v7: WorldMemory state also lives outside the store.
        data.world_memory_blob = voxel_world_memory::persist::serialize_blob(&self.world_memory);
        data.serialize()
    }

    /// Import world modification data from a binary buffer.
    /// Must be called BEFORE chunk streaming begins so that pending snapshots
    /// are applied as chunks are generated.
    /// Returns true on success.
    pub fn import_save_data(&self, bytes: &[u8]) -> bool {
        match crate::delta::WorldSaveData::deserialize(bytes) {
            Ok(data) => {
                // 2026-09-07: Shipping has no stderr - the stall log is the only
                // witness a player machine keeps of whether the world came back.
                crate::panic_log::note(&format!(
                    "[SAVE-IMPORT] ok: {} bytes, {} chunk snapshots",
                    bytes.len(), data.chunk_snapshots.len()));
                let anchor_json = data.crystal_anchors_json.clone();
                let world_memory_blob = data.world_memory_blob.clone();
                {
                    let mut store = self.store.write().unwrap();
                    store.load_save_data(data);
                }
                // Restore Crystal Anchor manager state from the JSON blob.
                let restored = crate::crystal_anchors::CrystalAnchorManager::from_json_string(&anchor_json);
                *self.crystal_anchors.lock().unwrap() = restored;
                // Block 1 v7: restore WorldMemory blob. On bad bytes the
                // loader returns Err and we keep WorldMemory empty — the
                // drift thread repopulates from live state.
                if !world_memory_blob.is_empty() {
                    if let Err(e) = voxel_world_memory::persist::load_blob(&self.world_memory, &world_memory_blob) {
                        eprintln!("[voxel-ffi] WorldMemory blob load failed: {e} — starting empty");
                    }
                }
                true
            }
            Err(e) => {
                eprintln!("[voxel-ffi] Failed to import save data: {e}");
                crate::panic_log::note(&format!("[SAVE-IMPORT] FAILED: {} bytes: {e}", bytes.len()));
                false
            }
        }
    }

    /// Apply pending snapshots to already-loaded density fields and re-extract hermite.
    /// Returns chunk keys that were patched (caller should request remesh for these).
    pub fn apply_loaded_snapshots(&self) -> Vec<(i32, i32, i32)> {
        use voxel_gen::hermite_extract::extract_hermite_data;
        let mut store = self.store.write().unwrap();
        let keys: Vec<(i32, i32, i32)> = match &store.pending_snapshots {
            Some(data) => data.chunk_snapshots.keys().copied().collect(),
            None => return Vec::new(),
        };
        let mut patched = Vec::new();
        for key in keys {
            if store.density_fields.contains_key(&key) {
                if store.apply_pending_snapshot(key) {
                    // Re-extract hermite from patched density
                    if let Some(df) = store.density_fields.get(&key) {
                        let new_hermite = extract_hermite_data(df);
                        store.hermite_data.insert(key, new_hermite);
                    }
                    patched.push(key);
                }
            }
        }
        // Per-snapshot smart re-seam companion: drop last_sent_mesh_hash for
        // every patched chunk and its 3 backward face neighbors. The caller
        // will RequestPriorityGenerate each patched chunk; that path's
        // incremental_seam_pass would otherwise hash-skip the backward
        // neighbors' updated combined meshes (their hashes were recorded
        // before the snapshots existed). Mirrors the in-flight regen
        // counterpart in worker.rs's GenerateChunk handler.
        if !patched.is_empty() {
            let bwd_offsets: [(i32, i32, i32); 3] = [(-1, 0, 0), (0, -1, 0), (0, 0, -1)];
            for &k in &patched {
                store.last_sent_mesh_hash.remove(&k);
                for &(dx, dy, dz) in &bwd_offsets {
                    store.last_sent_mesh_hash.remove(&(k.0 + dx, k.1 + dy, k.2 + dz));
                }
            }
            eprintln!("[voxel-ffi] Applied {} loaded snapshots for mid-game reload", patched.len());
        }
        patched
    }

    // ── Editor collapse triggers ───────────────────────────────────────

    /// Create a new editor collapse trigger. Returns the newly assigned id,
    /// or 0 on validation failure (no slab voxels, no volumes, etc.).
    ///
    /// `activation_kind`: 0 = OnFirstMine (uses `volumes[0]` as the trigger
    /// volume), 1 = OnPillarLoss (each entry of `volumes` is one pillar).
    /// The trigger is armed by default; pillar baselines are captured
    /// immediately against the current density.
    /// Helper: snapshot the engine's chunk_size from the live config.
    pub(crate) fn cached_chunk_size(&self) -> usize {
        self.config.read().unwrap().chunk_size
    }

    /// Fill `out` with the world voxel coords of every CELL whose center
    /// density is positive (i.e. the cell has rock at its interior) inside
    /// a sphere of `radius` voxels centered on `center`. Used by the
    /// in-engine Trigger Author brush so its slab paint only registers
    /// cells that the cinematic will actually treat as solid mass.
    ///
    /// Cell-center density = average of the cell's 8 corner samples. This
    /// is the same definition the DC mesh extraction uses to decide
    /// whether a cell has interior rock, so paint markers and cinematic
    /// slab will always agree. A single-corner read (the previous test)
    /// missed every cave-ceiling cell because its bottom corner sits in
    /// the cave air while the top four corners are in the rock above.
    ///
    /// Cells in unloaded chunks are skipped (treated as not-solid).
    pub fn query_solid_voxels_in_sphere(
        &self,
        center: (i32, i32, i32),
        radius: i32,
        out: &mut Vec<(i32, i32, i32)>,
    ) {
        out.clear();
        if radius < 0 {
            return;
        }
        let chunk_size = self.cached_chunk_size() as i32;
        let store = self.store.read().unwrap();
        let r2 = radius * radius;
        for dz in -radius..=radius {
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    if dx * dx + dy * dy + dz * dz > r2 {
                        continue;
                    }
                    let wx = center.0 + dx;
                    let wy = center.1 + dy;
                    let wz = center.2 + dz;
                    if cell_has_solid_center(&store, wx, wy, wz, chunk_size) {
                        out.push((wx, wy, wz));
                    }
                }
            }
        }
    }

    /// Export fluid simulation state (player-placed lava/water) as a binary
    /// buffer. Asks the fluid thread for a snapshot via `SnapshotRequest` and
    /// hands the payload to `crate::fluid_save::serialize`. Returns an empty
    /// vec on timeout or shutdown — callers should treat that as "no fluid
    /// to save" and not write a file (load skips missing files).
    pub fn export_fluid_data(&self) -> Vec<u8> {
        use crossbeam_channel::bounded;
        let (reply_tx, reply_rx) = bounded::<voxel_fluid::FluidSnapshot>(1);
        // The request rides the same bounded event channel the streaming
        // flood fills; one failed try_send used to abandon the save's fluid
        // outright (the slot got no fluid file — silent data loss on the
        // next load). A save is an explicit user action: a short bounded
        // retry on the game thread is the right trade.
        let mut event = voxel_fluid::FluidEvent::SnapshotRequest { reply_tx };
        let mut queued = false;
        for _ in 0..20 {
            match self.fluid_event_tx.try_send(event) {
                Ok(()) => {
                    queued = true;
                    break;
                }
                Err(crossbeam_channel::TrySendError::Full(ev)) => {
                    event = ev;
                    std::thread::sleep(std::time::Duration::from_millis(25));
                }
                Err(crossbeam_channel::TrySendError::Disconnected(_)) => break,
            }
        }
        if !queued {
            crate::panic_log::note("[FLUID-SAVE] snapshot request never enqueued - fluid NOT in this save");
            eprintln!("[voxel-ffi] export_fluid_data: failed to enqueue snapshot request");
            return Vec::new();
        }
        match reply_rx.recv_timeout(std::time::Duration::from_millis(2000)) {
            Ok(snapshot) => crate::fluid_save::serialize(&snapshot),
            Err(_) => {
                crate::panic_log::note("[FLUID-SAVE] snapshot timed out - fluid NOT in this save");
                eprintln!("[voxel-ffi] export_fluid_data: fluid thread snapshot timed out");
                Vec::new()
            }
        }
    }

    /// Import fluid simulation state from a binary buffer. The cells are
    /// queued in the fluid thread's pending-load map and applied per-chunk
    /// the moment that chunk's density arrives — this avoids landing fluid
    /// in cells whose `cell_capacity` would later read as solid.
    pub fn import_fluid_data(&self, bytes: &[u8]) -> bool {
        if bytes.is_empty() {
            return true; // nothing to do is success
        }
        match crate::fluid_save::deserialize(bytes) {
            Ok(per_chunk) => {
                let chunks = per_chunk.len();
                let cells: usize = per_chunk.values().map(|v| v.len()).sum();
                // 2026-08-19: this was a try_send per chunk with the result
                // discarded — under the load-time streaming flood the bounded
                // event channel is routinely full, so whole chunks of saved
                // fluid silently vanished (the same save restored all / none /
                // a-sixth of its fluid across three loads). The stash cannot
                // drop; the sim drains it every iteration.
                {
                    let mut stash = match self.fluid_import_stash.lock() {
                        Ok(g) => g,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    stash.extend(per_chunk);
                }
                crate::panic_log::note(&format!(
                    "[FLUID-IMPORT] stashed {chunks} chunks / {cells} cells for restore"
                ));
                eprintln!(
                    "[voxel-ffi] import_fluid_data: stashed {chunks} chunks / {cells} cells",
                );
                true
            }
            Err(e) => {
                eprintln!("[voxel-ffi] import_fluid_data: parse error {e:?}");
                false
            }
        }
    }

    /// Check if the world has any unsaved modifications.
    pub fn has_world_modifications(&self) -> bool {
        match self.store.try_read() {
            Ok(s) => s.has_modifications(),
            Err(_) => false,
        }
    }

}
