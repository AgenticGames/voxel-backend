//! Cluster — face-adjacency union-find over per-chunk score entries to
//! produce Scenes spanning multiple chunks of the same kind.
//!
//! Adjacency rule: two chunks of the same kind merge if their coordinates
//! differ by exactly 1 on exactly one axis (face-adjacent, NOT diagonal).
//! Diagonal-adjacency would smear unrelated POIs together too aggressively.

use std::collections::HashMap;

use glam::Vec3;

use crate::scene::{Aabb, Scene, SceneId, SceneKind};
use crate::scoring::ChunkScoreEntry;

/// Per-chunk input to clustering. Wraps the chunk coord (Rust space) and
/// the `ChunkScoreEntry` produced by `scoring::aggregate_signals`.
#[derive(Debug, Clone, Copy)]
pub struct ChunkScored {
    pub chunk_coord: (i32, i32, i32),
    pub entry: ChunkScoreEntry,
}

/// Cluster context — passed by the drift loop to `cluster_chunks`. Provides
/// chunk-size for centroid world-conversion.
#[derive(Debug, Clone, Copy)]
pub struct ClusterCtx {
    pub chunk_size: u32,
    /// Monotonic id allocator base (drift loop advances and writes back).
    pub next_scene_id: u64,
    pub now_secs: u32,
}

/// Result of clustering: one Scene per fused cluster, plus the new
/// `next_scene_id` for the caller to write back.
#[derive(Debug)]
pub struct ClusterOutput {
    pub scenes: Vec<Scene>,
    pub next_scene_id: u64,
}

/// Face-adjacency union-find. Input is per-chunk per-kind score entries;
/// output is one Scene per (kind, connected-cluster).
pub fn cluster_chunks(input: &[ChunkScored], ctx: ClusterCtx) -> ClusterOutput {
    if input.is_empty() {
        return ClusterOutput {
            scenes: Vec::new(),
            next_scene_id: ctx.next_scene_id,
        };
    }

    // Group by kind first so we only union within same-kind chunks.
    let mut by_kind: HashMap<SceneKind, Vec<usize>> = HashMap::new();
    for (i, c) in input.iter().enumerate() {
        by_kind.entry(c.entry.kind).or_default().push(i);
    }

    let mut next_id = ctx.next_scene_id;
    let mut scenes_out = Vec::new();

    for (kind, indices) in by_kind {
        // Union-find within this kind. Map: chunk_coord → index in this
        // kind's indices.
        let coord_to_local: HashMap<(i32, i32, i32), usize> = indices
            .iter()
            .enumerate()
            .map(|(local_i, &input_i)| (input[input_i].chunk_coord, local_i))
            .collect();

        let n = indices.len();
        let mut parent: Vec<usize> = (0..n).collect();

        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]]; // path compression
                x = parent[x];
            }
            x
        }

        fn union(parent: &mut [usize], a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb {
                parent[ra] = rb;
            }
        }

        // For each chunk in this kind, check its 6 face neighbors.
        for (local_i, &input_i) in indices.iter().enumerate() {
            let (cx, cy, cz) = input[input_i].chunk_coord;
            const NEIGHBORS: [(i32, i32, i32); 6] = [
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            ];
            for (dx, dy, dz) in NEIGHBORS {
                let nb = (cx + dx, cy + dy, cz + dz);
                if let Some(&nb_local) = coord_to_local.get(&nb) {
                    union(&mut parent, local_i, nb_local);
                }
            }
        }

        // Collect clusters by root.
        let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
        for local_i in 0..n {
            let root = find(&mut parent, local_i);
            clusters.entry(root).or_default().push(local_i);
        }

        // Build one Scene per cluster.
        for (_root, members) in clusters {
            let mut total_score = 0.0f32;
            let mut weighted_centroid = Vec3::ZERO;
            let mut aabb = Aabb::empty();
            let mut chunks = Vec::with_capacity(members.len());

            for local_i in &members {
                let input_i = indices[*local_i];
                let c = &input[input_i];
                let cs = ctx.chunk_size as f32;
                let world_centroid = Vec3::new(
                    c.chunk_coord.0 as f32 * cs + c.entry.centroid_local[0],
                    c.chunk_coord.1 as f32 * cs + c.entry.centroid_local[1],
                    c.chunk_coord.2 as f32 * cs + c.entry.centroid_local[2],
                );
                total_score += c.entry.score;
                weighted_centroid += world_centroid * c.entry.score;

                // Extend AABB by the chunk's world-space bounding cube
                // (rather than just the centroid) so the Scene's extent
                // covers its spatial footprint, not just the average.
                let chunk_min = Vec3::new(
                    c.chunk_coord.0 as f32 * cs,
                    c.chunk_coord.1 as f32 * cs,
                    c.chunk_coord.2 as f32 * cs,
                );
                let chunk_max = chunk_min + Vec3::splat(cs);
                aabb.extend_point(chunk_min);
                aabb.extend_point(chunk_max);
                chunks.push(c.chunk_coord);
            }

            let centroid = if total_score > 0.0 {
                weighted_centroid / total_score
            } else {
                aabb.center()
            };

            let id = SceneId(next_id);
            next_id += 1;
            let mut s = Scene::new(id, kind, centroid);
            s.score = total_score;
            s.aabb = aabb;
            // Confidence: single-chunk = 0.4, two-chunk = 0.6, 3+ = saturate to 1.0
            s.confidence = match members.len() {
                1 => 0.4,
                2 => 0.6,
                3 => 0.8,
                _ => 1.0,
            };
            s.age_secs = 0;
            s.last_seen_secs = ctx.now_secs;
            s.chunks = chunks;
            s.record_history(0 /* created */, ctx.now_secs);
            scenes_out.push(s);
        }
    }

    ClusterOutput {
        scenes: scenes_out,
        next_scene_id: next_id,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scoring::ChunkScoreEntry;

    fn entry(kind: SceneKind, score: f32) -> ChunkScoreEntry {
        ChunkScoreEntry {
            kind,
            score,
            centroid_local: [15.0, 15.0, 15.0], // chunk center for chunk_size=30
            cell_count: 200,
        }
    }

    #[test]
    fn three_face_adjacent_chunks_fuse_into_one_scene() {
        let input = vec![
            ChunkScored {
                chunk_coord: (0, 0, 0),
                entry: entry(SceneKind::Lava, 100.0),
            },
            ChunkScored {
                chunk_coord: (1, 0, 0),
                entry: entry(SceneKind::Lava, 100.0),
            },
            ChunkScored {
                chunk_coord: (2, 0, 0),
                entry: entry(SceneKind::Lava, 100.0),
            },
        ];
        let ctx = ClusterCtx {
            chunk_size: 30,
            next_scene_id: 1,
            now_secs: 0,
        };
        let out = cluster_chunks(&input, ctx);
        assert_eq!(out.scenes.len(), 1);
        let s = &out.scenes[0];
        assert_eq!(s.kind, SceneKind::Lava);
        assert!((s.score - 300.0).abs() < 1e-3);
        assert_eq!(s.chunks.len(), 3);
        assert!(s.confidence > 0.7); // 3-chunk → high confidence
        assert_eq!(out.next_scene_id, 2);
    }

    #[test]
    fn non_adjacent_chunks_stay_separate() {
        let input = vec![
            ChunkScored {
                chunk_coord: (0, 0, 0),
                entry: entry(SceneKind::Lava, 100.0),
            },
            ChunkScored {
                chunk_coord: (5, 0, 0), // 4 chunks apart
                entry: entry(SceneKind::Lava, 100.0),
            },
        ];
        let ctx = ClusterCtx {
            chunk_size: 30,
            next_scene_id: 1,
            now_secs: 0,
        };
        let out = cluster_chunks(&input, ctx);
        assert_eq!(out.scenes.len(), 2);
        assert_eq!(out.next_scene_id, 3);
    }

    #[test]
    fn diagonal_only_adjacency_stays_separate() {
        let input = vec![
            ChunkScored {
                chunk_coord: (0, 0, 0),
                entry: entry(SceneKind::Lava, 100.0),
            },
            ChunkScored {
                chunk_coord: (1, 1, 0), // diagonal, not face-adjacent
                entry: entry(SceneKind::Lava, 100.0),
            },
        ];
        let ctx = ClusterCtx {
            chunk_size: 30,
            next_scene_id: 1,
            now_secs: 0,
        };
        let out = cluster_chunks(&input, ctx);
        assert_eq!(out.scenes.len(), 2, "diagonal adjacency must NOT fuse");
    }

    #[test]
    fn different_kinds_never_fuse_even_if_adjacent() {
        let input = vec![
            ChunkScored {
                chunk_coord: (0, 0, 0),
                entry: entry(SceneKind::Lava, 100.0),
            },
            ChunkScored {
                chunk_coord: (1, 0, 0),
                entry: entry(SceneKind::Water, 100.0),
            },
        ];
        let ctx = ClusterCtx {
            chunk_size: 30,
            next_scene_id: 1,
            now_secs: 0,
        };
        let out = cluster_chunks(&input, ctx);
        assert_eq!(out.scenes.len(), 2);
    }

    #[test]
    fn weighted_centroid_in_world_space() {
        // Two chunks, second has 9× the score → centroid pulled to ~90% of
        // the way toward the heavier chunk.
        let input = vec![
            ChunkScored {
                chunk_coord: (0, 0, 0),
                entry: ChunkScoreEntry {
                    kind: SceneKind::Lava,
                    score: 10.0,
                    centroid_local: [15.0, 15.0, 15.0],
                    cell_count: 10,
                },
            },
            ChunkScored {
                chunk_coord: (1, 0, 0),
                entry: ChunkScoreEntry {
                    kind: SceneKind::Lava,
                    score: 90.0,
                    centroid_local: [15.0, 15.0, 15.0],
                    cell_count: 90,
                },
            },
        ];
        let ctx = ClusterCtx {
            chunk_size: 30,
            next_scene_id: 1,
            now_secs: 0,
        };
        let out = cluster_chunks(&input, ctx);
        assert_eq!(out.scenes.len(), 1);
        // Centroid_x: (15 * 10 + 45 * 90) / 100 = (150 + 4050) / 100 = 42.0
        assert!(
            (out.scenes[0].centroid[0] - 42.0).abs() < 1e-3,
            "centroid_x = {}",
            out.scenes[0].centroid[0]
        );
    }

    #[test]
    fn aabb_covers_chunk_footprints() {
        let input = vec![
            ChunkScored {
                chunk_coord: (0, 0, 0),
                entry: entry(SceneKind::Lava, 100.0),
            },
            ChunkScored {
                chunk_coord: (2, 0, 0),
                entry: entry(SceneKind::Lava, 100.0),
            },
            ChunkScored {
                chunk_coord: (1, 0, 0),
                entry: entry(SceneKind::Lava, 100.0),
            },
        ];
        let ctx = ClusterCtx {
            chunk_size: 30,
            next_scene_id: 1,
            now_secs: 0,
        };
        let out = cluster_chunks(&input, ctx);
        assert_eq!(out.scenes.len(), 1);
        let bb = &out.scenes[0].aabb;
        assert!((bb.min[0] - 0.0).abs() < 1e-3);
        assert!((bb.max[0] - 90.0).abs() < 1e-3); // 3 chunks × 30 voxels
    }
}
