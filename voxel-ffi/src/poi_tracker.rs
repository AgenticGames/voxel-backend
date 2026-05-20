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
    PoiKind, FLUID_MIN_LEVEL, MIN_LAVA_VOTES, MIN_STRESS_VOTES, MIN_WATER_VOTES,
    SCORE_PER_LAVA_VOXEL, SCORE_PER_STRESS_VOXEL, SCORE_PER_WATER_VOXEL, STRESS_HIGH_THRESHOLD,
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
/// Below this raw score, don't register the chunk at all — filters noise.
const MIN_REGISTRABLE_SCORE: f32 = 30.0;
/// Max time the tracker is willing to wait for the fluid thread to reply
/// with a snapshot. If fluid is overwhelmed, we skip fluid scoring this
/// tick (stress scoring still happens).
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
pub fn poi_tracker_loop(
    shutdown: Arc<AtomicBool>,
    store: Arc<RwLock<ChunkStore>>,
    fluid_event_tx: Sender<FluidEvent>,
    config: Arc<RwLock<GenerationConfig>>,
    tracker: SharedPoiTracker,
) {
    let mut cursor: usize = 0;
    let tick_dur = Duration::from_millis(TICK_DURATION_MS);
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

        // ── Step 2: fresh fluid snapshot (cap wait so we never starve) ──
        let (snap_tx, snap_rx) = bounded::<FluidSnapshot>(1);
        let _ = fluid_event_tx.send(FluidEvent::SnapshotRequest { reply_tx: snap_tx });
        let fluid_snap = snap_rx
            .recv_timeout(Duration::from_millis(FLUID_SNAPSHOT_TIMEOUT_MS))
            .unwrap_or_default();

        // ── Step 3: pick the budget of chunks for this tick ──
        let budget = SCAN_BUDGET_PER_TICK.min(chunk_keys.len());
        let mut budget_chunks: Vec<(i32, i32, i32)> = Vec::with_capacity(budget);
        for _ in 0..budget {
            budget_chunks.push(chunk_keys[cursor % chunk_keys.len()]);
            cursor = cursor.wrapping_add(1);
        }

        let _chunk_size = config.read().map(|c| c.chunk_size).unwrap_or(16);
        let now_secs = tracker.elapsed_secs();

        // ── Step 4: score each chunk in the budget ──
        // For stress, take a fresh short read lock per chunk. Holding the
        // lock for the entire budget would let a sleep-handler write starve
        // briefly. One lock per chunk = max ~16 short reads, each O(chunk).
        for chunk_coord in &budget_chunks {
            let stress_count = {
                match store.read() {
                    Ok(s) => s
                        .stress_fields
                        .get(chunk_coord)
                        .map(|sf| {
                            sf.stress
                                .iter()
                                .filter(|&&v| v > STRESS_HIGH_THRESHOLD)
                                .count()
                        })
                        .unwrap_or(0),
                    Err(_) => 0,
                }
            };

            // Fluid: from cached snapshot, no lock needed
            let (lava_count, water_count) = match fluid_snap.chunks.get(chunk_coord) {
                Some(cells) => {
                    let mut lava = 0usize;
                    let mut water = 0usize;
                    for c in cells {
                        if c.level < FLUID_MIN_LEVEL {
                            continue;
                        }
                        if c.fluid_type.is_lava() {
                            lava += 1;
                        } else if c.fluid_type.is_water() {
                            water += 1;
                        }
                    }
                    (lava, water)
                }
                None => (0, 0),
            };

            let lava_score = if lava_count >= MIN_LAVA_VOTES {
                lava_count as f32 * SCORE_PER_LAVA_VOXEL
            } else {
                0.0
            };
            let water_score = if water_count >= MIN_WATER_VOTES {
                water_count as f32 * SCORE_PER_WATER_VOXEL
            } else {
                0.0
            };
            let stress_score = if stress_count >= MIN_STRESS_VOTES {
                stress_count as f32 * SCORE_PER_STRESS_VOXEL
            } else {
                0.0
            };

            let (best_kind, best_score) = pick_best(lava_score, water_score, stress_score);

            if best_score >= MIN_REGISTRABLE_SCORE {
                tracker.scores.insert(
                    *chunk_coord,
                    ChunkPoiScore {
                        best_kind,
                        best_score,
                        last_scored_secs: now_secs,
                    },
                );
            } else {
                // Score dropped below threshold (e.g. fluid drained, stress
                // relieved) — actively evict so the FFI top-K doesn't keep
                // returning a chunk that's no longer interesting.
                tracker.scores.remove(chunk_coord);
            }
        }

        // ── Step 5: TTL prune ──
        if tick % PRUNE_EVERY_N_TICKS == 0 {
            tracker
                .scores
                .retain(|_, entry| now_secs.saturating_sub(entry.last_scored_secs) < TTL_SECS);
        }
    }
    crate::panic_log::worker_exited();
}

fn pick_best(lava: f32, water: f32, stress: f32) -> (PoiKind, f32) {
    if lava >= water && lava >= stress {
        (PoiKind::Lava, lava)
    } else if water >= stress {
        (PoiKind::Water, water)
    } else {
        (PoiKind::Stress, stress)
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
    fn pick_best_picks_correctly() {
        assert_eq!(pick_best(100.0, 50.0, 10.0).0, PoiKind::Lava);
        assert_eq!(pick_best(0.0, 80.0, 30.0).0, PoiKind::Water);
        assert_eq!(pick_best(0.0, 0.0, 60.0).0, PoiKind::Stress);
        // All-zero ties default to Lava (first branch). Filtered out by
        // MIN_REGISTRABLE_SCORE anyway, so the tie-break is academic.
        assert_eq!(pick_best(0.0, 0.0, 0.0).0, PoiKind::Lava);
    }
}
