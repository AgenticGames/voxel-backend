//! World Memory — persistent, semantically-clustered "things the world has done."
//!
//! Replaces the per-chunk threshold-binary POI tracker (in voxel-ffi's
//! `poi_tracker` + `poi_scanner`) with **Scenes**: clusters of adjacent
//! same-kind chunks with sub-voxel centroids, AABB, score, confidence, age,
//! event history, and free-form tags. Scenes survive chunk unload (cached in
//! a DashMap keyed by stable id) and persist into the save file (opaque blob
//! handed to `voxel-ffi`'s delta.rs at v7).
//!
//! Two ingestion paths:
//!   - **Drift** — a 2 s / 16-chunk background scan loop (mirrors the existing
//!     `voxel-poi-tracker` cadence in `voxel-ffi/src/poi_tracker.rs`). Catches
//!     ambient/background changes.
//!   - **Events** — `record_event(WorldEvent)` from the live worker paths.
//!     Brushes, anchor placements, collapses, and sleep completions push
//!     events into a lock-free queue drained by the drift loop within ~16 ms.
//!
//! FFI consumers query via `WorldMemory::scenes(filter, top_k)` (rich) or
//! `WorldMemory::legacy_top_k_pois(k)` (backward-compat adapter that lets the
//! existing `voxel_request_list_top_pois` keep working with strictly better
//! data — sub-voxel centroids, accurate extent radii).
//!
//! All Block 1 work is **additive**: the old POI tracker keeps running as a
//! cold-start fallback. Removal is Block 2's job.

pub mod adapter;
pub mod cluster;
pub mod drift;
pub mod event;
pub mod persist;
pub mod scene;
pub mod scoring;
pub mod topology;

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender};
use dashmap::DashMap;

pub use adapter::LegacyPoi;
pub use cluster::ClusterCtx;
pub use event::WorldEvent;
pub use scene::{Scene, SceneFilter, SceneHistoryEvent, SceneId, SceneKind};
pub use scoring::{CellSignal, ScoreContext};

/// Inbound event queue capacity. Sized generously — bursts during a sleep
/// cycle can record hundreds of voxel-change events in <16 ms. Overflow drops
/// the oldest event silently (events are hints, not authoritative state).
const EVENT_QUEUE_CAPACITY: usize = 4096;

/// Root struct. Cloneable via `Arc` for sharing between the engine, the
/// drift thread, and the FFI surface.
pub struct WorldMemory {
    /// All currently-tracked scenes. Key: stable `SceneId`.
    pub scenes: DashMap<SceneId, Scene>,
    /// Reference instant — every age/timestamp is relative to this.
    pub start_instant: Instant,
    /// Monotonic id allocator for new scenes.
    next_scene_id: AtomicU64,
    /// Lock-free event ingestion — pushed by live worker threads, drained
    /// by the drift loop.
    event_tx: Sender<WorldEvent>,
    event_rx: Receiver<WorldEvent>,
}

impl WorldMemory {
    pub fn new() -> Self {
        let (event_tx, event_rx) = crossbeam_channel::bounded(EVENT_QUEUE_CAPACITY);
        Self {
            scenes: DashMap::new(),
            start_instant: Instant::now(),
            next_scene_id: AtomicU64::new(1),
            event_tx,
            event_rx,
        }
    }

    /// Push an event for the drift loop to merge into Scene state. Lock-free.
    /// Returns `true` if accepted, `false` if the queue was full (caller
    /// should not retry — events are advisory).
    pub fn record_event(&self, event: WorldEvent) -> bool {
        self.event_tx.try_send(event).is_ok()
    }

    /// Drain the event queue into a Vec for batch processing by the drift
    /// loop. Bounded by `max` so a single tick can't be starved.
    pub fn drain_events(&self, max: usize) -> Vec<WorldEvent> {
        let mut out = Vec::with_capacity(max.min(64));
        for _ in 0..max {
            match self.event_rx.try_recv() {
                Ok(e) => out.push(e),
                Err(_) => break,
            }
        }
        out
    }

    pub fn elapsed_secs(&self) -> u64 {
        self.start_instant.elapsed().as_secs()
    }

    pub fn tracked_scene_count(&self) -> usize {
        self.scenes.len()
    }

    /// Allocate a fresh `SceneId`. Monotonically increasing; never re-used
    /// across the lifetime of a `WorldMemory` instance (including after
    /// load — the loader picks up where the save left off).
    pub fn alloc_scene_id(&self) -> SceneId {
        SceneId(self.next_scene_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Force-set the next id allocator (used by `persist::load_blob` so
    /// post-load id allocations don't collide with persisted ids).
    pub fn set_next_scene_id(&self, id: u64) {
        // Take the max — never go backwards.
        let mut cur = self.next_scene_id.load(Ordering::Relaxed);
        while cur < id {
            match self.next_scene_id.compare_exchange(
                cur,
                id,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(actual) => cur = actual,
            }
        }
    }

    /// Top-K scenes matching the filter, sorted by score descending.
    pub fn scenes(&self, filter: SceneFilter, top_k: usize) -> Vec<Scene> {
        let mut all: Vec<Scene> = self
            .scenes
            .iter()
            .filter(|entry| filter.matches(entry.value()))
            .map(|entry| entry.value().clone())
            .collect();
        all.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        all.truncate(top_k);
        all
    }

    /// Adapter for the legacy `voxel_request_list_top_pois` FFI. Returns
    /// scenes projected to the old `LegacyPoi` shape so UE compiles
    /// unchanged.
    pub fn legacy_top_k_pois(&self, k: usize, include_topology: bool) -> Vec<LegacyPoi> {
        adapter::legacy_top_k_pois(self, k, include_topology)
    }
}

impl Default for WorldMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod lib_tests {
    use super::*;

    #[test]
    fn worldmemory_basics() {
        let wm = WorldMemory::new();
        assert_eq!(wm.tracked_scene_count(), 0);
        let id1 = wm.alloc_scene_id();
        let id2 = wm.alloc_scene_id();
        assert!(id2.0 > id1.0);
    }

    #[test]
    fn record_event_accepts_under_capacity() {
        let wm = WorldMemory::new();
        for _ in 0..10 {
            assert!(wm.record_event(WorldEvent::lava_spread_at(0.0, 0.0, 0.0)));
        }
        let drained = wm.drain_events(100);
        assert_eq!(drained.len(), 10);
    }

    #[test]
    fn record_event_drops_on_overflow() {
        let wm = WorldMemory::new();
        let mut accepted = 0;
        for _ in 0..EVENT_QUEUE_CAPACITY * 2 {
            if wm.record_event(WorldEvent::lava_spread_at(0.0, 0.0, 0.0)) {
                accepted += 1;
            }
        }
        // Some were dropped — exact count depends on rx not being drained,
        // but accepted must be <= capacity.
        assert!(accepted <= EVENT_QUEUE_CAPACITY);
        assert!(accepted >= EVENT_QUEUE_CAPACITY / 2);
    }

    #[test]
    fn set_next_scene_id_never_goes_backwards() {
        let wm = WorldMemory::new();
        let _ = wm.alloc_scene_id();
        let _ = wm.alloc_scene_id();
        let _ = wm.alloc_scene_id();
        wm.set_next_scene_id(100);
        assert_eq!(wm.alloc_scene_id().0, 100);
        // Try to set it backwards — should be a no-op.
        wm.set_next_scene_id(5);
        assert_eq!(wm.alloc_scene_id().0, 101);
    }
}
