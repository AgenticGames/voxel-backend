//! Continuous (background-thread) Point-of-Interest tracker.
//!
//! Runs alongside the main worker pool, scoring loaded chunks at a budget
//! per tick (default 16 chunks every 2 seconds, round-robin). Per-chunk
//! scores accumulate in a [`DashMap`] keyed by chunk coord — entries survive
//! chunk unload so the sleep montage can revisit chunks the player explored
//! 20 minutes ago. TTL prunes anything older than 30 minutes of session
//! time so the map stays bounded.
//!
//! The FFI top-K query (`voxel_request_list_top_pois`) reads from this
//! map at query time, sorts by score, merges live bridges from the anchor
//! manager, and returns the unified top-K. The tracker thread itself never
//! handles bridges — they're a small live in-memory list pulled per query.
//!
//! Thread safety: tracker takes brief `RwLock<ChunkStore>` reads (one for
//! the key snapshot, one per chunk for stress). All reads are <1ms total
//! per tick so they never starve writers. Fluid data comes from a snapshot
//! requested via the existing fluid event channel.
//!
//! Tunables (constants below) — all easy to tweak without touching the
//! algorithm: scan budget, tick duration, TTL, prune cadence, min score.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, Sender};
use dashmap::DashMap;
use voxel_fluid::{FluidEvent, FluidSnapshot};
use voxel_gen::config::GenerationConfig;

use crate::poi_scanner::{
    count_fluid_voxels, count_high_stress_voxels, score_from_votes, PoiKind,
};
use crate::store::ChunkStore;

// ─── Tunables ───────────────────────────────────────────────────────────

/// Chunks scored per scan tick. Higher → faster coverage of new exploration
/// but more time spent holding read locks each tick.
const SCAN_BUDGET_PER_TICK: usize = 16;
/// Milliseconds between scan ticks. 2s = budget covers 8 chunks/sec.
const TICK_DURATION_MS: u64 = 2000;
/// Entries not re-scored within this many seconds get pruned. 30 minutes
/// matches "the player is unlikely to want to revisit something they last
/// saw half an hour ago."
const TTL_SECS: u64 = 30 * 60;
/// Run the TTL prune every N ticks (≈20s at default tick rate). Cheap, but
/// not worth doing every tick.
const PRUNE_EVERY_N_TICKS: u64 = 10;
/// Score must reach this to ADD a fresh entry or REFRESH an existing one
/// at full strength. Filters noise.
const MIN_REGISTRABLE_SCORE: f32 = 30.0;
/// Score must fall below this for an entry to be EVICTED. Together with
/// the register threshold above this gives a hysteresis band (30→15)
/// that prevents thrashing for chunks oscillating around the threshold.
const MIN_KEEP_SCORE: f32 = 15.0;
/// When a chunk's current-tick score drops below MIN_REGISTRABLE but it
/// has an existing entry, decay the stored score by this factor each tick
/// instead of overwriting (smooth fade-out, not insta-evict).
const SCORE_DECAY_PER_TICK: f32 = 0.6;
/// Fluid snapshots cost a Clone of the entire fluid state on the fluid
/// thread. Refresh every N ticks instead of every tick — fluid is slow
/// enough that staleness of 6s is fine, and this cuts pressure 3×.
const FLUID_SNAPSHOT_EVERY_N_TICKS: u64 = 3;
/// Max time the tracker is willing to wait for the fluid thread to reply
/// with a snapshot. If fluid is overwhelmed, we reuse the stale cached
/// snapshot (stress scoring still happens against fresh data).
const FLUID_SNAPSHOT_TIMEOUT_MS: u64 = 200;

// ─── Data ──────────────────────────────────────────────────────────────

/// Per-chunk record stored in the tracker's DashMap.
#[derive(Debug, Clone, Copy)]
pub struct ChunkPoiScore {
    pub best_kind: PoiKind,
    pub best_score: f32,
    /// Seconds since tracker start (`tracker.elapsed_secs()`).
    pub last_scored_secs: u64,
}

pub struct PoiTracker {
    /// All currently-tracked chunks. Read by FFI at query time, written by
    /// the tracker thread.
    pub scores: DashMap<(i32, i32, i32), ChunkPoiScore>,
    /// Reference instant — every `last_scored_secs` is relative to this.
    pub start_instant: Instant,
    /// Bumped every scan tick (telemetry).
    pub tick_counter: AtomicU64,
}

impl PoiTracker {
    pub fn new() -> Self {
        Self {
            scores: DashMap::new(),
            start_instant: Instant::now(),
            tick_counter: AtomicU64::new(0),
        }
    }

    pub fn elapsed_secs(&self) -> u64 {
        self.start_instant.elapsed().as_secs()
    }

    pub fn tracked_chunk_count(&self) -> usize {
        self.scores.len()
    }
}

pub type SharedPoiTracker = Arc<PoiTracker>;

pub fn new_tracker() -> SharedPoiTracker {
    Arc::new(PoiTracker::new())
}

// ─── Background loop ───────────────────────────────────────────────────

/// Tracker thread main. Spawned once from [`crate::engine::VoxelEngine::new`]
/// and runs until `shutdown` is set.
///
/// Per-tick flow:
///   1. Snapshot all loaded chunk keys (one brief read lock).
///   2. Optionally refresh fluid snapshot (every N ticks).
///   3. Read stress fields for the budget under ONE batched read lock — the
///      whole pass holds the lock <~2ms, then releases for scoring & writes.
///   4. Score each chunk via `score_chunk` (shared with [`poi_scanner`]).
///   5. Apply hysteresis: high score → insert/refresh, low score → decay or
///      keep the existing entry until it falls below the keep threshold.
///   6. Every N ticks, TTL-prune entries last touched >30min ago.
///
/// Chunk-unload race: a chunk may be unloaded between the key snapshot
/// (step 1) and the batched stress read (step 3). `stress_fields.get`
/// returns None for unloaded chunks; we treat as zero stress (no false
/// signal — fluid scoring still runs from the snapshot). The existing
/// score entry stays put and decays over subsequent ticks via hysteresis.
pub fn poi_tracker_loop(
    shutdown: Arc<AtomicBool>,
    store: Arc<RwLock<ChunkStore>>,
    fluid_event_tx: Sender<FluidEvent>,
    _config: Arc<RwLock<GenerationConfig>>,
    tracker: SharedPoiTracker,
) {
    let mut cursor: usize = 0;
    let tick_dur = Duration::from_millis(TICK_DURATION_MS);
    // Cached fluid snapshot — refreshed every FLUID_SNAPSHOT_EVERY_N_TICKS.
    let mut cached_fluid_snap: FluidSnapshot = FluidSnapshot::default();
    crate::panic_log::worker_started();

    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
        thread::sleep(tick_dur);
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        let tick = tracker.tick_counter.fetch_add(1, Ordering::Relaxed);

        // ── Step 1: snapshot chunk keys under a brief read lock ──
        let chunk_keys: Vec<(i32, i32, i32)> = {
            match store.read() {
                Ok(g) => {
                    let mut keys: Vec<(i32, i32, i32)> = g.density_fields.keys().copied().collect();
                    // Stable order for round-robin determinism
                    keys.sort_unstable();
                    keys
                }
                Err(_) => continue, // poisoned, skip this tick
            }
        };
        if chunk_keys.is_empty() {
            continue;
        }

        // ── Step 2: refresh fluid snapshot every N ticks ──
        // Fluid state changes on the order of seconds in normal play; a
        // 6-second-stale snapshot is fine and triples the time between
        // expensive fluid-state clones on the fluid thread.
        if tick % FLUID_SNAPSHOT_EVERY_N_TICKS == 0 {
            let (snap_tx, snap_rx) = bounded::<FluidSnapshot>(1);
            if fluid_event_tx
                .send(FluidEvent::SnapshotRequest { reply_tx: snap_tx })
                .is_ok()
            {
                if let Ok(snap) =
                    snap_rx.recv_timeout(Duration::from_millis(FLUID_SNAPSHOT_TIMEOUT_MS))
                {
                    cached_fluid_snap = snap;
                }
                // Timeout → keep the previous cached snapshot (better than
                // an empty default, which would force-evict all fluid POIs).
            }
        }

        // ── Step 3: pick the budget of chunks for this tick ──
        let budget = SCAN_BUDGET_PER_TICK.min(chunk_keys.len());
        let mut budget_chunks: Vec<(i32, i32, i32)> = Vec::with_capacity(budget);
        for _ in 0..budget {
            budget_chunks.push(chunk_keys[cursor % chunk_keys.len()]);
            cursor = cursor.wrapping_add(1);
        }

        let now_secs = tracker.elapsed_secs();

        // ── Step 4: batched stress read under one lock ──
        // Holding the lock for the whole budget (~16 chunks × stress-count =
        // ~1ms total) is fine — RwLock is read-shared so other readers don't
        // block. The sleep handler's write would wait briefly, but its work
        // takes seconds so 1-2ms is rounding error.
        let stress_votes: Vec<usize> = match store.read() {
            Ok(s) => budget_chunks
                .iter()
                .map(|coord| {
                    s.stress_fields
                        .get(coord)
                        .map(count_high_stress_voxels)
                        .unwrap_or(0)
                })
                .collect(),
            Err(_) => vec![0; budget_chunks.len()],
        };

        // ── Step 5: score each chunk + apply hysteresis ──
        for (idx, chunk_coord) in budget_chunks.iter().enumerate() {
            let (lava_votes, water_votes) = cached_fluid_snap
                .chunks
                .get(chunk_coord)
                .map(|cells| count_fluid_voxels(cells))
                .unwrap_or((0, 0));
            let breakdown = score_from_votes(lava_votes, water_votes, stress_votes[idx]);
            let (best_kind, best_score) = breakdown.best();

            apply_hysteresis(
                &tracker.scores,
                *chunk_coord,
                best_kind,
                best_score,
                now_secs,
            );
        }

        // ── Step 6: TTL prune ──
        if tick % PRUNE_EVERY_N_TICKS == 0 {
            tracker
                .scores
                .retain(|_, entry| now_secs.saturating_sub(entry.last_scored_secs) < TTL_SECS);
        }
    }
    crate::panic_log::worker_exited();
}

/// Score-update logic with hysteresis:
///   - new ≥ MIN_REGISTRABLE → insert/refresh entry at full strength
///   - new < MIN_REGISTRABLE + existing entry → decay existing by SCORE_DECAY
///   - decayed score < MIN_KEEP → evict
///   - no existing entry + low new score → no-op
///
/// Result: a chunk that hits a high score once persists for several ticks
/// after the signal drops, fading out gracefully. Prevents the thrashing
/// where a chunk oscillates around the threshold and gets inserted/evicted
/// every tick.
fn apply_hysteresis(
    scores: &DashMap<(i32, i32, i32), ChunkPoiScore>,
    coord: (i32, i32, i32),
    new_kind: PoiKind,
    new_score: f32,
    now_secs: u64,
) {
    if new_score >= MIN_REGISTRABLE_SCORE {
        scores.insert(
            coord,
            ChunkPoiScore {
                best_kind: new_kind,
                best_score: new_score,
                last_scored_secs: now_secs,
            },
        );
        return;
    }

    // New signal is weak. If there's an existing entry, decay it; else nothing.
    if let Some(mut existing) = scores.get_mut(&coord) {
        let decayed = existing.best_score * SCORE_DECAY_PER_TICK;
        if decayed < MIN_KEEP_SCORE {
            drop(existing); // release the entry guard before remove
            scores.remove(&coord);
        } else {
            existing.best_score = decayed;
            existing.last_scored_secs = now_secs;
        }
    }
}


// ─── Query API ─────────────────────────────────────────────────────────

/// Read top-K POIs from the tracker, sorted descending by score. Bridges
/// are appended by the caller before truncation — they're not in this map.
pub fn read_top_k(tracker: &PoiTracker, k: usize, chunk_size: usize) -> Vec<crate::poi_scanner::Poi> {
    if k == 0 {
        return Vec::new();
    }
    let cs_f = chunk_size as f32;
    let mut entries: Vec<((i32, i32, i32), ChunkPoiScore)> = tracker
        .scores
        .iter()
        .map(|kv| (*kv.key(), *kv.value()))
        .collect();
    entries.sort_by(|a, b| {
        b.1.best_score
            .partial_cmp(&a.1.best_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    entries.truncate(k);
    entries
        .into_iter()
        .map(|(coord, sc)| crate::poi_scanner::Poi {
            kind: sc.best_kind,
            score: sc.best_score,
            chunk_coord: coord,
            center_world_rust: glam::Vec3::new(
                coord.0 as f32 * cs_f + cs_f * 0.5,
                coord.1 as f32 * cs_f + cs_f * 0.5,
                coord.2 as f32 * cs_f + cs_f * 0.5,
            ),
            extent_radius_voxels: cs_f * 0.5,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poi_scanner::PoiKind;

    #[test]
    fn empty_tracker_returns_empty() {
        let t = new_tracker();
        let v = read_top_k(&t, 5, 16);
        assert!(v.is_empty());
    }

    #[test]
    fn insert_and_read_top_k() {
        let t = new_tracker();
        t.scores.insert(
            (0, 0, 0),
            ChunkPoiScore {
                best_kind: PoiKind::Lava,
                best_score: 200.0,
                last_scored_secs: 0,
            },
        );
        t.scores.insert(
            (1, 0, 0),
            ChunkPoiScore {
                best_kind: PoiKind::Stress,
                best_score: 100.0,
                last_scored_secs: 0,
            },
        );
        t.scores.insert(
            (2, 0, 0),
            ChunkPoiScore {
                best_kind: PoiKind::Water,
                best_score: 50.0,
                last_scored_secs: 0,
            },
        );

        let top2 = read_top_k(&t, 2, 16);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].kind, PoiKind::Lava);
        assert!((top2[0].score - 200.0).abs() < 1e-3);
        assert_eq!(top2[1].kind, PoiKind::Stress);
    }

    #[test]
    fn hysteresis_inserts_above_register_threshold() {
        let scores: DashMap<(i32, i32, i32), ChunkPoiScore> = DashMap::new();
        apply_hysteresis(&scores, (0, 0, 0), PoiKind::Lava, 100.0, 0);
        assert_eq!(scores.len(), 1);
        assert!((scores.get(&(0, 0, 0)).unwrap().best_score - 100.0).abs() < 1e-3);
    }

    #[test]
    fn hysteresis_decays_below_threshold_keeps_above_keep() {
        let scores: DashMap<(i32, i32, i32), ChunkPoiScore> = DashMap::new();
        // Seed with high score
        apply_hysteresis(&scores, (0, 0, 0), PoiKind::Lava, 100.0, 0);
        // Subsequent tick reports low signal → decay (100 × 0.6 = 60)
        apply_hysteresis(&scores, (0, 0, 0), PoiKind::Lava, 0.0, 1);
        let after = scores.get(&(0, 0, 0)).unwrap().best_score;
        assert!((after - 60.0).abs() < 1e-3, "expected 60, got {}", after);
    }

    #[test]
    fn hysteresis_evicts_after_enough_decay() {
        let scores: DashMap<(i32, i32, i32), ChunkPoiScore> = DashMap::new();
        // Seed at just-above-keep so one decay step drops below MIN_KEEP_SCORE
        apply_hysteresis(&scores, (0, 0, 0), PoiKind::Lava, 30.0, 0);
        // 30 × 0.6 = 18, still ≥ 15 → keep
        apply_hysteresis(&scores, (0, 0, 0), PoiKind::Lava, 0.0, 1);
        assert!(scores.contains_key(&(0, 0, 0)));
        // 18 × 0.6 = 10.8, < 15 → evict
        apply_hysteresis(&scores, (0, 0, 0), PoiKind::Lava, 0.0, 2);
        assert!(!scores.contains_key(&(0, 0, 0)));
    }

    #[test]
    fn hysteresis_no_op_when_low_and_no_existing() {
        let scores: DashMap<(i32, i32, i32), ChunkPoiScore> = DashMap::new();
        apply_hysteresis(&scores, (0, 0, 0), PoiKind::Lava, 5.0, 0);
        assert!(scores.is_empty());
    }
}
