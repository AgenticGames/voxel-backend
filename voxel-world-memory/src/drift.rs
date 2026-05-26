//! Drift loop — background scan that refreshes Scene scores from live world
//! state on a 2 s cadence, mirroring the existing `voxel-poi-tracker` loop in
//! `voxel-ffi/src/poi_tracker.rs:144-288`.
//!
//! Block 1 skeleton: the integration with `ChunkStore` lives in voxel-ffi.
//! This module defines the per-tick *algorithm* that takes
//! already-collected per-chunk score entries (produced by voxel-ffi which
//! holds the ChunkStore read lock) and merges them into the
//! `WorldMemory.scenes` map with hysteresis.
//!
//! Splitting the algorithm here (no ChunkStore dep) from the IO in voxel-ffi
//! keeps this crate independent of FFI and trivially unit-testable.

use crate::cluster::{cluster_chunks, ChunkScored, ClusterCtx, ClusterOutput};
use crate::event::WorldEvent;
use crate::scene::{Scene, SceneKind};
use crate::WorldMemory;

// ── Tunables (mirror the legacy POI tracker's hysteresis values) ─────

/// Below this score, a fresh chunk-scoring won't register a new Scene.
/// Filters noise. Mirrors `MIN_REGISTRABLE_SCORE = 30.0` in voxel-ffi POI
/// tracker, but in per-cell-weighted scale (so 30.0 / scale ≈ 3.0).
pub const MIN_REGISTRABLE_SCORE: f32 = 3.0;

/// When a Scene's tick-score falls below this, decay-instead-of-overwrite
/// the stored score. Mirrors `MIN_KEEP_SCORE` in the legacy tracker.
pub const MIN_KEEP_SCORE: f32 = 1.5;

/// Decay factor for the hysteresis band. Mirrors `SCORE_DECAY_PER_TICK`
/// in the legacy tracker.
pub const SCORE_DECAY_PER_TICK: f32 = 0.6;

/// TTL in seconds. Entries not refreshed within this window get pruned.
/// Mirrors `TTL_SECS = 30 * 60` in the legacy tracker.
pub const TTL_SECS: u32 = 30 * 60;

/// Drift-tick context. Caller (voxel-ffi) populates this from the live
/// ChunkStore data + current monotonic time.
pub struct DriftCtx {
    pub now_secs: u32,
    pub chunk_size: u32,
    /// Per-chunk per-kind scores collected from the live scan + topology
    /// detection. Empty if no chunks were scanned this tick.
    pub fresh_scores: Vec<ChunkScored>,
    /// Events drained from the WorldMemory event queue (call
    /// `wm.drain_events(N)` to populate). Limit to ~64 per tick so a flood
    /// can't starve the scan.
    pub events: Vec<WorldEvent>,
    /// Whether to include topology kinds in the cluster pass (gated by
    /// engine config `enable_topology_scenes`).
    pub include_topology: bool,
}

impl DriftCtx {
    pub fn new(now_secs: u32, chunk_size: u32) -> Self {
        Self {
            now_secs,
            chunk_size,
            fresh_scores: Vec::new(),
            events: Vec::new(),
            include_topology: false,
        }
    }
}

/// Apply one drift tick to the WorldMemory. Returns the number of scenes
/// inserted/updated/pruned. Caller (voxel-ffi drift thread) calls this
/// every TICK_DURATION_MS.
///
/// Algorithm:
///  1. Cluster fresh per-chunk scores into Scenes (cluster.rs).
///  2. Merge into `wm.scenes`: if a candidate-scene overlaps an existing
///     Scene's chunk set, refresh that Scene's score (full or decayed
///     based on hysteresis). Otherwise insert as new.
///  3. Process events: bump score on Scenes containing the event's
///     position; create lightweight "hint" Scenes for events that don't
///     overlap any existing Scene.
///  4. Prune any Scenes whose `last_seen_secs` is older than TTL.
pub fn drift_tick(wm: &WorldMemory, ctx: &DriftCtx) -> DriftStats {
    let mut stats = DriftStats::default();

    // ── 1. Cluster fresh scores ────────────────────────────────────────
    let next_id_start = wm.alloc_scene_id().0;
    // alloc_scene_id bumped — set back to allow cluster_chunks to allocate
    // sequentially from there.
    wm.set_next_scene_id(next_id_start);

    let cluster_ctx = ClusterCtx {
        chunk_size: ctx.chunk_size,
        next_scene_id: next_id_start,
        now_secs: ctx.now_secs,
    };

    // Filter input to non-topology kinds unless include_topology is set.
    let filtered_input: Vec<ChunkScored> = if ctx.include_topology {
        ctx.fresh_scores.clone()
    } else {
        ctx.fresh_scores
            .iter()
            .filter(|c| !c.entry.kind.is_topology())
            .copied()
            .collect()
    };

    let ClusterOutput { scenes, next_scene_id } = cluster_chunks(&filtered_input, cluster_ctx);
    wm.set_next_scene_id(next_scene_id);

    // ── 2. Merge into wm.scenes (hysteresis) ──────────────────────────
    for fresh_scene in scenes {
        let existing = find_overlapping_scene(wm, &fresh_scene);
        if let Some(existing_id) = existing {
            // Refresh the existing scene.
            if let Some(mut entry) = wm.scenes.get_mut(&existing_id) {
                if fresh_scene.score >= MIN_REGISTRABLE_SCORE {
                    // Full refresh — overwrite score, extend AABB.
                    entry.score = fresh_scene.score;
                    entry.aabb = entry.aabb.union(&fresh_scene.aabb);
                    entry.centroid = fresh_scene.centroid;
                    entry.confidence = (entry.confidence + 0.1).min(1.0);
                    entry.last_seen_secs = ctx.now_secs;
                    // Merge chunk lists (dedup).
                    for chunk in &fresh_scene.chunks {
                        if !entry.chunks.contains(chunk) {
                            entry.chunks.push(*chunk);
                        }
                    }
                    entry.record_history(1 /* refreshed */, ctx.now_secs);
                    stats.refreshed += 1;
                } else if fresh_scene.score >= MIN_KEEP_SCORE {
                    // Weak signal — decay rather than overwrite.
                    entry.score *= SCORE_DECAY_PER_TICK;
                    entry.last_seen_secs = ctx.now_secs;
                    stats.decayed += 1;
                }
                // Below MIN_KEEP_SCORE: skip, will TTL-prune later.
            }
        } else if fresh_scene.score >= MIN_REGISTRABLE_SCORE {
            // Insert as new scene.
            stats.inserted += 1;
            wm.scenes.insert(fresh_scene.id, fresh_scene);
        }
    }

    // ── 3. Process events ──────────────────────────────────────────────
    for ev in &ctx.events {
        if let (Some(pos), Some(kind)) = (ev.world_pos(), ev.kind_hint()) {
            // Bump matching scenes whose AABB contains the event pos.
            let event_pos_vec = glam::Vec3::new(pos[0], pos[1], pos[2]);
            let mut bumped = false;
            for mut entry in wm.scenes.iter_mut() {
                if entry.value().kind == kind && aabb_contains(&entry.value().aabb, event_pos_vec)
                {
                    let s = entry.value_mut();
                    s.score += 1.0; // small bump
                    s.last_seen_secs = ctx.now_secs;
                    s.record_history(2 /* event-promoted */, ctx.now_secs);
                    bumped = true;
                }
            }
            if !bumped && kind != SceneKind::Bridge {
                // Light event for a non-tracked region — record a tiny
                // ephemeral scene that the next drift scan will either
                // confirm (full Scene) or TTL-prune.
                let id = wm.alloc_scene_id();
                let mut s = Scene::new(id, kind, event_pos_vec);
                s.score = MIN_REGISTRABLE_SCORE * 0.8; // sub-threshold
                s.confidence = 0.2;
                s.last_seen_secs = ctx.now_secs;
                s.record_history(2 /* event-promoted */, ctx.now_secs);
                wm.scenes.insert(id, s);
                stats.event_seeded += 1;
            }
        }
    }

    // ── 4. TTL prune ───────────────────────────────────────────────────
    let ttl = TTL_SECS;
    let now = ctx.now_secs;
    let stale: Vec<_> = wm
        .scenes
        .iter()
        .filter(|e| now.saturating_sub(e.value().last_seen_secs) > ttl)
        .map(|e| *e.key())
        .collect();
    for id in stale {
        wm.scenes.remove(&id);
        stats.pruned += 1;
    }

    stats
}

/// Find an existing Scene whose chunk-list overlaps `fresh.chunks` AND
/// kind matches. Returns the first match's id, or None.
fn find_overlapping_scene(wm: &WorldMemory, fresh: &Scene) -> Option<crate::scene::SceneId> {
    for entry in wm.scenes.iter() {
        if entry.value().kind != fresh.kind {
            continue;
        }
        for ch in &entry.value().chunks {
            if fresh.chunks.contains(ch) {
                return Some(*entry.key());
            }
        }
    }
    None
}

fn aabb_contains(bb: &crate::scene::Aabb, p: glam::Vec3) -> bool {
    p.x >= bb.min[0]
        && p.x <= bb.max[0]
        && p.y >= bb.min[1]
        && p.y <= bb.max[1]
        && p.z >= bb.min[2]
        && p.z <= bb.max[2]
}

/// Drift-tick statistics for telemetry.
#[derive(Debug, Default, Clone, Copy)]
pub struct DriftStats {
    pub inserted: u32,
    pub refreshed: u32,
    pub decayed: u32,
    pub event_seeded: u32,
    pub pruned: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scoring::ChunkScoreEntry;

    fn fresh(chunk: (i32, i32, i32), kind: SceneKind, score: f32) -> ChunkScored {
        ChunkScored {
            chunk_coord: chunk,
            entry: ChunkScoreEntry {
                kind,
                score,
                centroid_local: [15.0, 15.0, 15.0],
                cell_count: (score as u32).max(1),
            },
        }
    }

    #[test]
    fn fresh_score_above_min_registers_new_scene() {
        let wm = WorldMemory::new();
        let mut ctx = DriftCtx::new(0, 30);
        ctx.fresh_scores
            .push(fresh((0, 0, 0), SceneKind::Lava, 100.0));
        let stats = drift_tick(&wm, &ctx);
        assert_eq!(stats.inserted, 1);
        assert_eq!(wm.tracked_scene_count(), 1);
    }

    #[test]
    fn fresh_score_below_min_does_not_register() {
        let wm = WorldMemory::new();
        let mut ctx = DriftCtx::new(0, 30);
        ctx.fresh_scores.push(fresh((0, 0, 0), SceneKind::Lava, 1.0));
        let stats = drift_tick(&wm, &ctx);
        assert_eq!(stats.inserted, 0);
        assert_eq!(wm.tracked_scene_count(), 0);
    }

    #[test]
    fn second_tick_refreshes_existing_scene() {
        let wm = WorldMemory::new();
        // First tick — register.
        {
            let mut ctx = DriftCtx::new(0, 30);
            ctx.fresh_scores
                .push(fresh((0, 0, 0), SceneKind::Lava, 50.0));
            drift_tick(&wm, &ctx);
        }
        assert_eq!(wm.tracked_scene_count(), 1);
        let confidence_before = wm.scenes.iter().next().unwrap().value().confidence;

        // Second tick — refresh.
        {
            let mut ctx = DriftCtx::new(5, 30);
            ctx.fresh_scores
                .push(fresh((0, 0, 0), SceneKind::Lava, 80.0));
            let stats = drift_tick(&wm, &ctx);
            assert_eq!(stats.refreshed, 1);
            assert_eq!(stats.inserted, 0);
        }
        assert_eq!(wm.tracked_scene_count(), 1);
        let s = wm.scenes.iter().next().unwrap().value().clone();
        assert!((s.score - 80.0).abs() < 1e-3);
        assert!(s.confidence > confidence_before, "confidence should grow");
        assert_eq!(s.last_seen_secs, 5);
    }

    #[test]
    fn weak_tick_decays_existing() {
        let wm = WorldMemory::new();
        {
            let mut ctx = DriftCtx::new(0, 30);
            ctx.fresh_scores
                .push(fresh((0, 0, 0), SceneKind::Lava, 50.0));
            drift_tick(&wm, &ctx);
        }
        // Weak follow-up — in hysteresis band (between MIN_KEEP and MIN_REG).
        {
            let mut ctx = DriftCtx::new(5, 30);
            ctx.fresh_scores
                .push(fresh((0, 0, 0), SceneKind::Lava, 2.0)); // < MIN_REG=3, > MIN_KEEP=1.5
            let stats = drift_tick(&wm, &ctx);
            assert_eq!(stats.decayed, 1);
        }
        let s = wm.scenes.iter().next().unwrap().value().clone();
        // 50 * 0.6 = 30
        assert!((s.score - 30.0).abs() < 1e-3);
    }

    #[test]
    fn event_in_aabb_bumps_score() {
        let wm = WorldMemory::new();
        {
            let mut ctx = DriftCtx::new(0, 30);
            ctx.fresh_scores
                .push(fresh((0, 0, 0), SceneKind::Lava, 50.0));
            drift_tick(&wm, &ctx);
        }
        let score_before = wm.scenes.iter().next().unwrap().value().score;

        // Event at world pos within the Scene's chunk (chunk_size=30 → AABB [0,0,0] to [30,30,30]).
        let mut ctx = DriftCtx::new(5, 30);
        ctx.events
            .push(WorldEvent::lava_spread_at(15.0, 15.0, 15.0));
        drift_tick(&wm, &ctx);

        let score_after = wm.scenes.iter().next().unwrap().value().score;
        assert!(score_after > score_before);
    }

    #[test]
    fn event_outside_any_scene_seeds_ephemeral() {
        let wm = WorldMemory::new();
        let mut ctx = DriftCtx::new(0, 30);
        ctx.events.push(WorldEvent::lava_spread_at(99.0, 99.0, 99.0));
        let stats = drift_tick(&wm, &ctx);
        assert_eq!(stats.event_seeded, 1);
        assert_eq!(wm.tracked_scene_count(), 1);
        let s = wm.scenes.iter().next().unwrap().value().clone();
        assert_eq!(s.kind, SceneKind::Lava);
        assert!(s.confidence < 0.5); // low confidence
    }

    #[test]
    fn ttl_prunes_stale_scenes() {
        let wm = WorldMemory::new();
        {
            let mut ctx = DriftCtx::new(0, 30);
            ctx.fresh_scores
                .push(fresh((0, 0, 0), SceneKind::Lava, 50.0));
            drift_tick(&wm, &ctx);
        }
        assert_eq!(wm.tracked_scene_count(), 1);

        // Fast-forward past TTL with no refresh.
        let ctx = DriftCtx::new(TTL_SECS + 100, 30);
        let stats = drift_tick(&wm, &ctx);
        assert_eq!(stats.pruned, 1);
        assert_eq!(wm.tracked_scene_count(), 0);
    }

    #[test]
    fn topology_filtered_unless_flag_set() {
        let wm = WorldMemory::new();
        let mut ctx = DriftCtx::new(0, 30);
        ctx.fresh_scores
            .push(fresh((0, 0, 0), SceneKind::CeilingDome, 50.0));
        ctx.include_topology = false;
        let stats = drift_tick(&wm, &ctx);
        assert_eq!(stats.inserted, 0); // CeilingDome filtered out

        ctx.include_topology = true;
        let stats = drift_tick(&wm, &ctx);
        assert_eq!(stats.inserted, 1);
    }
}
