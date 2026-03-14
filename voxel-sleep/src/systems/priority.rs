use std::collections::HashSet;

/// Priority tiers for chunk processing during sleep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkTier {
    /// Player chunk + 6 face-adjacent: all transforms + full cascade
    Critical,
    /// 2-ring neighbors + chunks with supports: metamorphism + minerals + lightweight collapse
    Important,
    /// All other loaded chunks: metamorphism only
    Cosmetic,
}

/// Classify loaded chunks into priority tiers based on player position.
pub fn classify_chunks(
    player_chunk: (i32, i32, i32),
    loaded_chunks: &[(i32, i32, i32)],
    chunks_with_supports: &HashSet<(i32, i32, i32)>,
) -> Vec<((i32, i32, i32), ChunkTier)> {
    let (px, py, pz) = player_chunk;
    let mut result = Vec::with_capacity(loaded_chunks.len());

    for &(cx, cy, cz) in loaded_chunks {
        let dx = (cx - px).abs();
        let dy = (cy - py).abs();
        let dz = (cz - pz).abs();
        let chebyshev = dx.max(dy).max(dz);

        let tier = if chebyshev <= 1 {
            // Player chunk + 6 face-adjacent (actually 26-connected at distance 1)
            // But plan says "6 face-adjacent" which is Manhattan distance 1
            let manhattan = dx + dy + dz;
            if manhattan <= 1 {
                ChunkTier::Critical
            } else {
                ChunkTier::Important
            }
        } else if chebyshev <= 2 || chunks_with_supports.contains(&(cx, cy, cz)) {
            ChunkTier::Important
        } else {
            ChunkTier::Cosmetic
        };

        result.push(((cx, cy, cz), tier));
    }

    // Sort: Critical first, then Important, then Cosmetic
    result.sort_by_key(|(_, tier)| match tier {
        ChunkTier::Critical => 0,
        ChunkTier::Important => 1,
        ChunkTier::Cosmetic => 2,
    });

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_player_chunk_is_critical() {
        let loaded = vec![(0, 0, 0), (1, 0, 0), (-1, 0, 0), (2, 0, 0)];
        let supports = HashSet::new();
        let classified = classify_chunks((0, 0, 0), &loaded, &supports);

        let critical: Vec<_> = classified.iter()
            .filter(|(_, t)| *t == ChunkTier::Critical)
            .collect();
        assert!(critical.iter().any(|((x, y, z), _)| *x == 0 && *y == 0 && *z == 0));
        assert!(critical.iter().any(|((x, y, z), _)| *x == 1 && *y == 0 && *z == 0));
    }

    #[test]
    fn test_support_chunk_is_important() {
        let loaded = vec![(0, 0, 0), (5, 5, 5)];
        let mut supports = HashSet::new();
        supports.insert((5, 5, 5));
        let classified = classify_chunks((0, 0, 0), &loaded, &supports);

        let far_chunk = classified.iter().find(|((x, y, z), _)| *x == 5 && *y == 5 && *z == 5).unwrap();
        assert_eq!(far_chunk.1, ChunkTier::Important);
    }

    #[test]
    fn test_far_chunk_is_cosmetic() {
        let loaded = vec![(0, 0, 0), (10, 0, 0)];
        let supports = HashSet::new();
        let classified = classify_chunks((0, 0, 0), &loaded, &supports);

        let far_chunk = classified.iter().find(|((x, _, _), _)| *x == 10).unwrap();
        assert_eq!(far_chunk.1, ChunkTier::Cosmetic);
    }

    #[test]
    fn test_sorted_by_priority() {
        let loaded = vec![(10, 0, 0), (0, 0, 0), (2, 0, 0), (1, 0, 0)];
        let supports = HashSet::new();
        let classified = classify_chunks((0, 0, 0), &loaded, &supports);

        // First entries should be Critical
        assert_eq!(classified[0].1, ChunkTier::Critical);
    }
}
