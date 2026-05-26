//! World-memory drift thread loop (voxel-ffi side).
//!
//! Spawned once from `VoxelEngine::new`. Runs on a 2 s cadence, mirrors
//! the existing POI tracker pattern in `voxel-ffi/src/poi_tracker.rs` but
//! produces per-cell-weighted `ChunkScored` entries for the new
//! `voxel-world-memory` Scene store.
//!
//! Block 1 scope: stress signals + lava cells (lava from fluid snapshot
//! cached every N ticks). Block 2 will fold in water + cross-chunk
//! topology votes and retire the legacy `poi_tracker` once UE consumes
//! the Scene API directly.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

use crossbeam_channel::{bounded, Sender};
use voxel_fluid::{FluidEvent, FluidSnapshot};
use voxel_gen::config::GenerationConfig;
use voxel_world_memory::cluster::ChunkScored;
use voxel_world_memory::drift::{drift_tick, DriftCtx};
use voxel_world_memory::scene::SceneKind;
use voxel_world_memory::scoring::{
    aggregate_signals, CellSignal, WEIGHT_LAVA_CELL, WEIGHT_STRESS_CELL,
};
use voxel_world_memory::WorldMemory;

use crate::store::ChunkStore;

/// Chunks scanned per tick. Matches POI tracker's `SCAN_BUDGET_PER_TICK = 16`.
const SCAN_BUDGET_PER_TICK: usize = 16;
/// Tick interval in milliseconds. Matches POI tracker's `TICK_DURATION_MS`.
const TICK_DURATION_MS: u64 = 2000;
/// Refresh fluid snapshot every N ticks (~6 s at 2 s tick). Matches POI
/// tracker's `FLUID_SNAPSHOT_EVERY_N_TICKS = 3`.
const FLUID_SNAPSHOT_EVERY_N_TICKS: u64 = 3;
const FLUID_SNAPSHOT_TIMEOUT_MS: u64 = 200;
/// Drain at most this many events from the WorldMemory event queue per tick.
/// Prevents a brush burst from monopolizing the scan budget.
const MAX_EVENTS_PER_TICK: usize = 64;
/// High-stress per-cell threshold. Cells above this contribute to the
/// chunk's Stress signal weight. Mirrors the legacy POI scanner's per-cell
/// threshold at voxel-ffi/src/poi_scanner.rs (~0.7).
const HIGH_STRESS_THRESHOLD: f32 = 0.7;
/// Lava cell level threshold (matches existing fluid scoring).
const LAVA_LEVEL_MIN: f32 = 0.10;

/// Drift thread main. Catch panics and respawn up to 8 times — same
/// pattern as the POI tracker.
pub fn world_memory_drift_loop(
    shutdown: Arc<AtomicBool>,
    store: Arc<RwLock<ChunkStore>>,
    fluid_event_tx: Sender<FluidEvent>,
    config: Arc<RwLock<GenerationConfig>>,
    world_memory: Arc<WorldMemory>,
) {
    let tick_dur = Duration::from_millis(TICK_DURATION_MS);
    let mut tick: u64 = 0;
    let mut cursor: usize = 0;
    let mut cached_fluid: FluidSnapshot = FluidSnapshot::default();
    crate::panic_log::worker_started();

    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
        thread::sleep(tick_dur);
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
        tick += 1;

        // Snapshot chunk keys (brief read lock).
        let chunk_keys: Vec<(i32, i32, i32)> = match store.read() {
            Ok(g) => {
                let mut keys: Vec<_> = g.density_fields.keys().copied().collect();
                keys.sort_unstable();
                keys
            }
            Err(_) => continue,
        };
        if chunk_keys.is_empty() {
            continue;
        }

        // Refresh fluid snapshot every N ticks.
        if tick % FLUID_SNAPSHOT_EVERY_N_TICKS == 0 {
            let (snap_tx, snap_rx) = bounded::<FluidSnapshot>(1);
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
        }

        // Pick the scan budget (round-robin across chunk_keys).
        let budget = SCAN_BUDGET_PER_TICK.min(chunk_keys.len());
        let mut budget_chunks: Vec<(i32, i32, i32)> = Vec::with_capacity(budget);
        for _ in 0..budget {
            budget_chunks.push(chunk_keys[cursor % chunk_keys.len()]);
            cursor = cursor.wrapping_add(1);
        }

        let chunk_size_voxels = config
            .read()
            .map(|c| c.chunk_size as u32)
            .unwrap_or(16) as u32;

        // Build per-chunk score entries. Holds the read lock for the
        // whole budget — RwLock is read-shared so concurrent readers are
        // fine. Writers (sleep, brushes) wait briefly. Budget × density
        // size keeps us well under 5 ms p99.
        let chunk_scored: Vec<ChunkScored> = match store.read() {
            Ok(s) => budget_chunks
                .iter()
                .filter_map(|&coord| {
                    let signals = build_chunk_signals(
                        &s,
                        coord,
                        &cached_fluid,
                        chunk_size_voxels as usize,
                    );
                    if signals.is_empty() {
                        return None;
                    }
                    let aggregated = aggregate_signals(&signals);
                    // One ChunkScored per dominant kind found in this chunk.
                    let strongest = aggregated.into_iter().max_by(|a, b| {
                        a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal)
                    })?;
                    Some(ChunkScored {
                        chunk_coord: coord,
                        entry: strongest,
                    })
                })
                .collect(),
            Err(_) => Vec::new(),
        };

        // Drain events + invoke drift_tick.
        let events = world_memory.drain_events(MAX_EVENTS_PER_TICK);
        let now_secs = world_memory.elapsed_secs() as u32;
        let mut ctx = DriftCtx::new(now_secs, chunk_size_voxels);
        ctx.fresh_scores = chunk_scored;
        ctx.events = events;
        // Block 1: keep topology kinds off — UE doesn't handle them yet.
        ctx.include_topology = false;
        let _stats = drift_tick(&world_memory, &ctx);
    }
}

/// Produce per-cell signals for one chunk. Block 1 covers stress + lava;
/// water + topology added in Block 2.
fn build_chunk_signals(
    store: &ChunkStore,
    coord: (i32, i32, i32),
    fluid: &FluidSnapshot,
    chunk_size_voxels: usize,
) -> Vec<CellSignal> {
    let mut out = Vec::with_capacity(64);

    // Stress signals — walk the stress field, emit one signal per
    // cell whose stress exceeds the threshold.
    if let Some(sf) = store.stress_fields.get(&coord) {
        let stride = chunk_size_voxels;
        for z in 0..stride {
            for y in 0..stride {
                for x in 0..stride {
                    let idx = z * stride * stride + y * stride + x;
                    if idx < sf.stress.len() && sf.stress[idx] >= HIGH_STRESS_THRESHOLD {
                        out.push(CellSignal {
                            kind: SceneKind::Stress,
                            weight: WEIGHT_STRESS_CELL,
                            local_pos: [x as u32, y as u32, z as u32],
                        });
                    }
                }
            }
        }
    }

    // Lava signals from the cached fluid snapshot.
    if let Some(cells) = fluid.chunks.get(&coord) {
        let fcs = fluid.chunk_size;
        for z in 0..fcs {
            for y in 0..fcs {
                for x in 0..fcs {
                    let idx = z * fcs * fcs + y * fcs + x;
                    if idx < cells.len()
                        && cells[idx].level > LAVA_LEVEL_MIN
                        && cells[idx].fluid_type.is_lava()
                    {
                        out.push(CellSignal {
                            kind: SceneKind::Lava,
                            weight: WEIGHT_LAVA_CELL,
                            local_pos: [x as u32, y as u32, z as u32],
                        });
                    }
                }
            }
        }
    }

    out
}
