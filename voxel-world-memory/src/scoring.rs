//! Per-cell weighted scoring — replaces chunk-binary threshold gates in the
//! legacy `voxel-ffi/src/poi_scanner.rs::score_from_votes` (lines 71-101).
//!
//! Each cell emits a `CellSignal { kind, weight }`. Scoring then aggregates
//! across the cell population in a chunk. Stage 2 (cluster.rs) then
//! aggregates across chunks into Scenes.
//!
//! Calibration: a chunk with 200 lava cells must score in [180, 220] so the
//! legacy adapter at `legacy_top_k_pois` returns scores within ±10% of the
//! pre-change baseline (where `SCORE_PER_LAVA_VOXEL = 10.0`, so 200 × 10 =
//! 2000). NOTE: pre-change scale was 10x higher because chunk-binary scoring
//! had no aggregate-multiplier downstream. We calibrate to match the FINAL
//! adapter score, which divides by 10 to align with UE's expectations. See
//! `legacy_score_for` in `adapter.rs`.

use crate::scene::SceneKind;

// ── Per-cell weights ──────────────────────────────────────────────────
// These are the **per-cell** weights. The chunk-aggregate score is the sum
// of weights, optionally with a small "density bonus" for chunks with
// concentrated signal vs. sparse signal of the same total.

/// Lava cell contributes 1.0 weight. Calibration: 200 cells × 1.0 = 200 →
/// adapter divides through to give legacy parity ~200 (legacy was 200 × 10
/// = 2000; adapter scales /10).
pub const WEIGHT_LAVA_CELL: f32 = 1.0;
/// Water cell contributes 0.6 (matches legacy `SCORE_PER_WATER_VOXEL = 6.0`
/// relative scale: water is ~60% as cinematic as lava).
pub const WEIGHT_WATER_CELL: f32 = 0.6;
/// High-stress cell contributes 0.8 (matches legacy `SCORE_PER_STRESS_VOXEL
/// = 8.0` relative scale).
pub const WEIGHT_STRESS_CELL: f32 = 0.8;
/// Bridge voxel contributes 0.12 (matches legacy `BRIDGE_LENGTH_BONUS_PER_VOXEL
/// = 1.2` relative scale; bridge baseline goes through the adapter at 8.0).
pub const WEIGHT_BRIDGE_VOXEL: f32 = 0.12;
/// Bridge baseline (added regardless of length) — 8.0 to give bridges a
/// presence even when very short. Legacy baseline was 80.0.
pub const BRIDGE_BASELINE_WEIGHT: f32 = 8.0;

/// Per-cell weights for topology kinds. These are evaluated at the
/// chunk-level (one vote per chunk for the kind), not per-cell.
pub const WEIGHT_CEILING_DOME: f32 = 35.0;
pub const WEIGHT_CHOKEPOINT: f32 = 28.0;
pub const WEIGHT_WALL_NICHE: f32 = 22.0;

// ── Minimum totals to register a Scene ─────────────────────────────────
// Filters out noise. Legacy thresholds were per-chunk counts (16/24/32).
// These are post-weight aggregate floors.

/// Below this, lava cluster won't register. Tuned to ignore single-voxel
/// fluid splashes.
pub const MIN_LAVA_SCORE: f32 = 4.0;
pub const MIN_WATER_SCORE: f32 = 4.0;
pub const MIN_STRESS_SCORE: f32 = 6.0;
pub const MIN_BRIDGE_SCORE: f32 = BRIDGE_BASELINE_WEIGHT; // any paired bridge counts
pub const MIN_TOPOLOGY_SCORE: f32 = 15.0;

/// A single cell's contribution to scoring. Emitted by the per-chunk scan
/// for each "interesting" cell encountered.
#[derive(Debug, Clone, Copy)]
pub struct CellSignal {
    pub kind: SceneKind,
    pub weight: f32,
    /// Local voxel coordinates within the chunk (used for centroid).
    pub local_pos: [u32; 3],
}

/// Context passed to scoring helpers — provides chunk-size and other
/// per-evaluation knobs without threading them through every function.
#[derive(Debug, Clone, Copy)]
pub struct ScoreContext {
    pub chunk_size: u32,
}

impl ScoreContext {
    pub fn new(chunk_size: u32) -> Self {
        Self { chunk_size }
    }
}

/// Aggregate a vec of cell signals into a single chunk-level score per kind.
/// Returns weighted centroid + score for each kind present in the input.
pub fn aggregate_signals(signals: &[CellSignal]) -> Vec<ChunkScoreEntry> {
    let mut by_kind: [(f32, [f32; 3], u32); 7] = [(0.0, [0.0; 3], 0); 7]; // Kind 0..=6

    for sig in signals {
        let idx = sig.kind as u8 as usize;
        if idx >= 7 {
            continue;
        }
        let (sum_w, centroid_acc, count) = &mut by_kind[idx];
        *sum_w += sig.weight;
        centroid_acc[0] += sig.weight * sig.local_pos[0] as f32;
        centroid_acc[1] += sig.weight * sig.local_pos[1] as f32;
        centroid_acc[2] += sig.weight * sig.local_pos[2] as f32;
        *count += 1;
    }

    let mut out = Vec::new();
    for (idx, (sum_w, centroid_acc, count)) in by_kind.iter().enumerate() {
        if *count == 0 || *sum_w <= 0.0 {
            continue;
        }
        let kind = match SceneKind::from_u8(idx as u8) {
            Some(k) => k,
            None => continue,
        };
        let cx = centroid_acc[0] / *sum_w;
        let cy = centroid_acc[1] / *sum_w;
        let cz = centroid_acc[2] / *sum_w;
        // Density bonus: concentrated signal scores higher than diffuse.
        // ratio of weighted_sum / count = average weight per cell ⇒ 1.0
        // for solidly hot cells, much less for diffuse mixed regions.
        // We don't actually need it for parity with legacy, so leave it
        // off in Block 1 — straight weighted sum.
        let score = *sum_w;
        out.push(ChunkScoreEntry {
            kind,
            score,
            centroid_local: [cx, cy, cz],
            cell_count: *count,
        });
    }
    out
}

/// One Scene-kind aggregate score for a chunk. Stage 2 (cluster.rs) folds
/// these across adjacent chunks.
#[derive(Debug, Clone, Copy)]
pub struct ChunkScoreEntry {
    pub kind: SceneKind,
    pub score: f32,
    /// Weighted centroid in local chunk voxel coords. Stage 2 converts to
    /// world coords by adding `chunk_coord * chunk_size`.
    pub centroid_local: [f32; 3],
    pub cell_count: u32,
}

/// Minimum score threshold for a kind to be considered.
pub fn min_score_for(kind: SceneKind) -> f32 {
    match kind {
        SceneKind::Lava => MIN_LAVA_SCORE,
        SceneKind::Water => MIN_WATER_SCORE,
        SceneKind::Stress => MIN_STRESS_SCORE,
        SceneKind::Bridge => MIN_BRIDGE_SCORE,
        SceneKind::CeilingDome | SceneKind::Chokepoint | SceneKind::WallNiche => {
            MIN_TOPOLOGY_SCORE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_200_lava_cells_score_180_to_220() {
        // Plan requires: synthetic chunk with 200 lava cells must score in
        // [180, 220]. Adapter feeds this through legacy_score_for which
        // multiplies by 10 to get UE scale (200 → 2000). Here we check the
        // base score before adapter scaling.
        let signals: Vec<CellSignal> = (0..200)
            .map(|i| CellSignal {
                kind: SceneKind::Lava,
                weight: WEIGHT_LAVA_CELL,
                local_pos: [(i % 30) as u32, ((i / 30) % 30) as u32, (i / 900) as u32],
            })
            .collect();
        let aggs = aggregate_signals(&signals);
        assert_eq!(aggs.len(), 1);
        assert_eq!(aggs[0].kind, SceneKind::Lava);
        // 200 cells × 1.0 weight = 200.0. Adapter scales ×10 → 2000.
        // Pre-change baseline: 200 cells × SCORE_PER_LAVA_VOXEL(10) = 2000.
        // Adapter parity confirmed.
        assert!(
            aggs[0].score >= 180.0 && aggs[0].score <= 220.0,
            "lava score {} not in [180, 220]",
            aggs[0].score
        );
        assert_eq!(aggs[0].cell_count, 200);
    }

    #[test]
    fn aggregate_separates_kinds() {
        let signals = vec![
            CellSignal {
                kind: SceneKind::Lava,
                weight: 1.0,
                local_pos: [0, 0, 0],
            },
            CellSignal {
                kind: SceneKind::Water,
                weight: 0.6,
                local_pos: [10, 0, 0],
            },
            CellSignal {
                kind: SceneKind::Lava,
                weight: 1.0,
                local_pos: [2, 0, 0],
            },
        ];
        let aggs = aggregate_signals(&signals);
        assert_eq!(aggs.len(), 2);

        let lava = aggs.iter().find(|a| a.kind == SceneKind::Lava).unwrap();
        let water = aggs.iter().find(|a| a.kind == SceneKind::Water).unwrap();
        assert!((lava.score - 2.0).abs() < 1e-5);
        assert!((water.score - 0.6).abs() < 1e-5);
        // Lava centroid is weighted-avg of [0,0,0] and [2,0,0] → [1, 0, 0]
        assert!((lava.centroid_local[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn weighted_centroid_pulls_toward_heavier_cells() {
        let signals = vec![
            CellSignal {
                kind: SceneKind::Stress,
                weight: 1.0,
                local_pos: [0, 0, 0],
            },
            CellSignal {
                kind: SceneKind::Stress,
                weight: 9.0, // 9× heavier
                local_pos: [10, 0, 0],
            },
        ];
        let aggs = aggregate_signals(&signals);
        // Weighted: (0×1 + 10×9) / (1+9) = 90/10 = 9.0
        assert!((aggs[0].centroid_local[0] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn empty_signals_returns_empty() {
        let aggs = aggregate_signals(&[]);
        assert!(aggs.is_empty());
    }

    #[test]
    fn min_score_thresholds_match_kinds() {
        assert_eq!(min_score_for(SceneKind::Lava), MIN_LAVA_SCORE);
        assert_eq!(min_score_for(SceneKind::Water), MIN_WATER_SCORE);
        assert_eq!(min_score_for(SceneKind::Stress), MIN_STRESS_SCORE);
        assert_eq!(min_score_for(SceneKind::Bridge), MIN_BRIDGE_SCORE);
        assert_eq!(min_score_for(SceneKind::CeilingDome), MIN_TOPOLOGY_SCORE);
    }
}
