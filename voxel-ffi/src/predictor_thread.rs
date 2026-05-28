//! Sleep predictor thread loop (voxel-ffi side).
//!
//! Spawned once from `VoxelEngine::new`. Wakes every 60 s OR on a wake
//! signal via `predict_wake_rx`. Each cycle: snapshot density/stress/
//! support HashMaps under a brief read lock, release, run
//! `voxel_sleep::predict::predict_next_sleep`, write the result to
//! `predict_cache`.
//!
//! The cache is a hint, not authoritative state — when real `execute_sleep`
//! lands it overwrites the cache in `worker.rs`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

use crossbeam_channel::{select, Receiver, Sender};
use voxel_fluid::{FluidEvent, FluidSnapshot};
use voxel_gen::config::GenerationConfig;
use voxel_sleep::predict::{predict_next_sleep, PredictSnapshot, PredictedManifest};
use voxel_sleep::SleepConfig;

use crate::store::ChunkStore;

/// Maximum interval between automatic prediction cycles. UE-poked wake
/// signals can fire sooner.
const PERIODIC_INTERVAL_SECS: u64 = 60;
/// Timeout for fluid snapshot request before we proceed with the last
/// cached value.
const FLUID_SNAPSHOT_TIMEOUT_MS: u64 = 250;

/// Predictor thread main. Catch-unwind + respawn handled by caller.
pub fn predictor_thread_loop(
    shutdown: Arc<AtomicBool>,
    store: Arc<RwLock<ChunkStore>>,
    fluid_event_tx: Sender<FluidEvent>,
    config: Arc<RwLock<GenerationConfig>>,
    sleep_config: Arc<RwLock<SleepConfig>>,
    predict_cache: Arc<RwLock<Option<PredictedManifest>>>,
    wake_rx: Receiver<()>,
) {
    crate::panic_log::worker_started();
    let mut sleep_count_counter: u32 = 0;
    let mut cached_fluid: FluidSnapshot = FluidSnapshot::default();

    // Shutdown channel — synthesize one to put inside select! alongside wake_rx.
    // We poll shutdown after every wake/timeout instead.
    let interval = Duration::from_secs(PERIODIC_INTERVAL_SECS);

    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
        // Wait for either wake signal or the periodic timer. The channel
        // disconnect (engine.shutdown drops predict_wake_tx) returns
        // Err(_) here — treat as an immediate exit so we don't sit
        // through the 60-second default interval during PIE teardown.
        let do_predict;
        select! {
            recv(wake_rx) -> msg => {
                match msg {
                    Ok(_) => { do_predict = true; }
                    Err(_) => { break; } // channel closed — engine shutting down
                }
            }
            default(interval) => { do_predict = true; }
        }
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
        if !do_predict {
            continue;
        }

        sleep_count_counter = sleep_count_counter.wrapping_add(1);

        // Read a fresh fluid snapshot (so the predictor sees current lava).
        let (snap_tx, snap_rx) = crossbeam_channel::bounded::<FluidSnapshot>(1);
        if fluid_event_tx
            .send(FluidEvent::SnapshotRequest { reply_tx: snap_tx })
            .is_ok()
        {
            if let Ok(snap) =
                snap_rx.recv_timeout(Duration::from_millis(FLUID_SNAPSHOT_TIMEOUT_MS))
            {
                cached_fluid = snap;
            }
        }

        let (chunk_size, chunk_radius) = {
            let cfg = config.read().ok();
            let sc = sleep_config.read().ok();
            let cs = cfg.map(|c| c.chunk_size).unwrap_or(16);
            let r = sc.map(|s| s.chunk_radius).unwrap_or(3);
            (cs, r as u32)
        };

        // Build PredictSnapshot by cloning the relevant chunks under a
        // brief read lock. Target: <5 ms read-lock hold for chunk_radius=3.
        let snapshot = {
            let g = match store.read() {
                Ok(g) => g,
                Err(_) => continue,
            };
            // Use player chunk = (0,0,0) for now — UE will be able to
            // push the real player_chunk via a future FFI call. For Block 1,
            // origin is fine since the predictor's output is keyed by
            // chunk coords and UE can filter against its known player.
            let player_chunk = (0i32, 0i32, 0i32);
            let r = chunk_radius as i32;

            let mut density_fields = std::collections::HashMap::new();
            let mut stress_fields = std::collections::HashMap::new();
            let mut support_fields = std::collections::HashMap::new();
            for (&key, df) in &g.density_fields {
                let dx = (key.0 - player_chunk.0).abs();
                let dy = (key.1 - player_chunk.1).abs();
                let dz = (key.2 - player_chunk.2).abs();
                if dx.max(dy).max(dz) <= r {
                    density_fields.insert(key, df.clone());
                    if let Some(sf) = g.stress_fields.get(&key) {
                        stress_fields.insert(key, sf.clone());
                    }
                    if let Some(sf) = g.support_fields.get(&key) {
                        support_fields.insert(key, sf.clone());
                    }
                }
            }
            drop(g);

            PredictSnapshot::new(
                density_fields,
                stress_fields,
                support_fields,
                cached_fluid.clone(),
                player_chunk,
                sleep_count_counter,
                chunk_size,
                chunk_radius,
            )
        };

        // Run prediction (no locks held — owned scratch).
        let manifest = predict_next_sleep(&snapshot);

        // Stash in cache.
        if let Ok(mut g) = predict_cache.write() {
            *g = Some(manifest);
        }
    }
}
