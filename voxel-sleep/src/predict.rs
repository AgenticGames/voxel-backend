//! Sleep predictor — fast forward-pass that estimates the *next* deep-sleep
//! outcome from a snapshot of current world state. Produces a
//! [`PredictedManifest`] that UE consumes to pre-warm chunks, pre-compose
//! shots, and remove the "Time passes…" loading-screen wait.
//!
//! The predictor does NOT mutate world state. It takes owned cloned
//! scratch HashMaps (built via [`PredictSnapshot::clone_from_store`]) and
//! emits a manifest of *likely* changes.
//!
//! **Speed target**: <300 ms p99 on a 343-chunk world (`chunk_radius=3`).
//! The full `execute_sleep` takes 1.5–4 s; the predictor sacrifices voxel
//! fidelity for ~10–30× speedup.
//!
//! **RNG isolation**: predictor seeds [`ChaCha8Rng`] with
//! `sleep_count * 7919 + 42 + 0x1_0000_0000`. The real `execute_sleep` uses
//! `sleep_count * 7919 + 42` (see `lib.rs::execute_sleep` line ~355). Same
//! base formula + different offset = uncorrelated samples, so predicted
//! voxel positions don't trivially appear in the real sim (which would be
//! cheating).
//!
//! **Output is a hint, not authoritative state**. When the real sleep
//! lands, it overwrites the cache. If the predictor's `lava_cells` differs
//! from the real result's `lava_cells`, the real one wins.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use voxel_core::density::DensityField;
use voxel_core::material::Material;
use voxel_core::stress::{StressField, SupportField};
use voxel_fluid::FluidSnapshot;

use crate::phases::aureole::{build_heat_map, HeatSource, HeatSourceType};

/// RNG seed offset that distinguishes the predictor from real
/// `execute_sleep` (see lib.rs line ~355).
pub const PREDICTOR_RNG_OFFSET: u64 = 0x1_0000_0000;

/// Compute the predictor's RNG seed for a given sleep_count.
pub fn predictor_seed(sleep_count: u32) -> u64 {
    sleep_count as u64 * 7919 + 42 + PREDICTOR_RNG_OFFSET
}

/// Owned scratch state passed to the predictor. The drift thread builds
/// this under a brief read lock on `ChunkStore`, then releases the lock
/// before running prediction (so writers don't block on the ~200 ms scan).
pub struct PredictSnapshot {
    /// Cloned density fields. Sized to ~76 KB per chunk at chunk_size=30,
    /// so a `chunk_radius=3` snapshot (7³ = 343 chunks) is ~26 MB transient.
    pub density_fields: HashMap<(i32, i32, i32), DensityField>,
    pub stress_fields: HashMap<(i32, i32, i32), StressField>,
    pub support_fields: HashMap<(i32, i32, i32), SupportField>,
    pub fluid_snapshot: FluidSnapshot,
    pub player_chunk: (i32, i32, i32),
    pub sleep_count: u32,
    pub chunk_size: usize,
    /// Chebyshev radius around `player_chunk` to consider. Matches the
    /// `SleepConfig.chunk_radius`.
    pub chunk_radius: u32,
}

impl PredictSnapshot {
    /// Build a snapshot manually (used by tests + the engine-side helper
    /// that takes the read lock).
    pub fn new(
        density_fields: HashMap<(i32, i32, i32), DensityField>,
        stress_fields: HashMap<(i32, i32, i32), StressField>,
        support_fields: HashMap<(i32, i32, i32), SupportField>,
        fluid_snapshot: FluidSnapshot,
        player_chunk: (i32, i32, i32),
        sleep_count: u32,
        chunk_size: usize,
        chunk_radius: u32,
    ) -> Self {
        Self {
            density_fields,
            stress_fields,
            support_fields,
            fluid_snapshot,
            player_chunk,
            sleep_count,
            chunk_size,
            chunk_radius,
        }
    }

    /// Filter loaded chunks to those within `chunk_radius` Chebyshev
    /// distance of the player. Matches `execute_sleep` line ~365.
    pub fn loaded_in_range(&self) -> Vec<(i32, i32, i32)> {
        let r = self.chunk_radius as i32;
        let p = self.player_chunk;
        let mut out: Vec<_> = self
            .density_fields
            .keys()
            .copied()
            .filter(|&(cx, cy, cz)| {
                let dx = (cx - p.0).abs();
                let dy = (cy - p.1).abs();
                let dz = (cz - p.2).abs();
                dx.max(dy).max(dz) <= r
            })
            .collect();
        out.sort();
        out
    }
}

/// A predicted scene hint — what the cinematographer can pre-compose
/// against before the real sleep completes.
#[derive(Debug, Clone, Copy)]
pub struct SceneHint {
    /// Numeric Scene kind matching `voxel-world-memory::SceneKind`
    /// discriminants (0=Lava, 1=Water, 2=Stress, 3=Bridge).
    pub kind: u8,
    pub world_pos: (i32, i32, i32),
    pub estimated_score: f32,
    pub chunk_coord: (i32, i32, i32),
}

/// The predictor's output — handed to the UE prediction cache via FFI.
#[derive(Debug, Clone)]
pub struct PredictedManifest {
    pub chunks_likely_changed: Vec<(i32, i32, i32)>,
    pub predicted_lava_cells: Vec<(i32, i32, i32)>,
    pub predicted_aureole_glimpse_pos: Option<(i32, i32, i32)>,
    pub predicted_aureole_block: Option<Vec<(i32, i32, i32)>>,
    pub predicted_scene_hints: Vec<SceneHint>,
    /// Seconds since UNIX epoch when this manifest was computed. UE can
    /// reject stale predictions on its side.
    pub computed_at_secs: u64,
    pub wall_ms: u32,
    /// Sleep count used to seed the predictor. UE can match this against
    /// the real-sleep result's sleep_count to detect mismatches.
    pub sleep_count: u32,
}

impl Default for PredictedManifest {
    fn default() -> Self {
        Self {
            chunks_likely_changed: Vec::new(),
            predicted_lava_cells: Vec::new(),
            predicted_aureole_glimpse_pos: None,
            predicted_aureole_block: None,
            predicted_scene_hints: Vec::new(),
            computed_at_secs: 0,
            wall_ms: 0,
            sleep_count: 0,
        }
    }
}

/// Run the prediction pass on an owned snapshot. Returns a
/// [`PredictedManifest`] suitable for caching + handing to UE.
///
/// Algorithm:
///   1. Build heat map (lava cells + kimberlite voxels). O(fluid + density).
///   2. Cluster heat sources into zones via BFS. Pick the biggest as the
///      predicted aureole glimpse. O(N log N) on heat-source count.
///   3. Identify chunks within metamorphic radius of any heat source —
///      these are "likely changed."
///   4. Scan stress fields for high-stress chunks → "likely collapse."
///   5. Aggregate water cells per chunk → Water scene hints.
///   6. Pack into `PredictedManifest`.
///
/// No mutation of the snapshot. Deterministic given (snapshot, sleep_count).
pub fn predict_next_sleep(snap: &PredictSnapshot) -> PredictedManifest {
    let t_start = Instant::now();
    let mut _rng = ChaCha8Rng::seed_from_u64(predictor_seed(snap.sleep_count));

    let chunks_in_range = snap.loaded_in_range();

    // 1. Heat map (reuses existing voxel-sleep helper)
    let heat_map = build_heat_map(
        &snap.density_fields,
        &snap.fluid_snapshot,
        &chunks_in_range,
        snap.chunk_size,
    );

    // 2. Cluster heat sources into zones via face-adjacency BFS. Pick the
    //    biggest zone's centroid as the predicted aureole glimpse.
    let (zones, lava_cells) = cluster_heat_zones(&heat_map);
    let (predicted_aureole_glimpse_pos, predicted_aureole_block) = pick_aureole(&zones, snap.chunk_size);

    // 3. Chunks likely changed: any chunk that contains, or is within
    //    METAMORPHIC_CHUNK_RADIUS of, a heat source.
    const METAMORPHIC_CHUNK_RADIUS: i32 = 1;
    let cs = snap.chunk_size as i32;
    let mut likely: HashSet<(i32, i32, i32)> = HashSet::new();
    for src in &heat_map {
        let (wx, wy, wz) = src.pos;
        let bcx = wx.div_euclid(cs);
        let bcy = wy.div_euclid(cs);
        let bcz = wz.div_euclid(cs);
        for dx in -METAMORPHIC_CHUNK_RADIUS..=METAMORPHIC_CHUNK_RADIUS {
            for dy in -METAMORPHIC_CHUNK_RADIUS..=METAMORPHIC_CHUNK_RADIUS {
                for dz in -METAMORPHIC_CHUNK_RADIUS..=METAMORPHIC_CHUNK_RADIUS {
                    likely.insert((bcx + dx, bcy + dy, bcz + dz));
                }
            }
        }
    }

    // 4. High-stress chunks (collapse risk).
    let mut stress_hints: Vec<SceneHint> = Vec::new();
    for chunk in &chunks_in_range {
        if let Some(sf) = snap.stress_fields.get(chunk) {
            let avg_stress = avg_chunk_stress(sf);
            // Threshold tuned to match the legacy POI scanner's high-stress
            // detection at voxel-ffi/src/poi_scanner.rs (per-cell threshold
            // ~0.7). avg > 0.5 is the chunk-level proxy.
            if avg_stress > 0.5 {
                likely.insert(*chunk);
                let cx = chunk.0 * cs + cs / 2;
                let cy = chunk.1 * cs + cs / 2;
                let cz = chunk.2 * cs + cs / 2;
                stress_hints.push(SceneHint {
                    kind: 2, // Stress
                    world_pos: (cx, cy, cz),
                    estimated_score: (avg_stress * 80.0).min(200.0),
                    chunk_coord: *chunk,
                });
            }
        }
    }

    // 5. Scene hints from lava + water + zones.
    let mut scene_hints: Vec<SceneHint> = Vec::new();
    for zone in &zones {
        scene_hints.push(SceneHint {
            kind: 0, // Lava
            world_pos: zone.centroid,
            estimated_score: (zone.cell_count as f32 * 10.0).min(2000.0),
            chunk_coord: world_to_chunk(zone.centroid, cs),
        });
    }
    // Water scene hints from fluid snapshot.
    let water_hints = water_scene_hints(&snap.fluid_snapshot, cs);
    scene_hints.extend(water_hints);
    scene_hints.extend(stress_hints);

    // Aggregate, dedupe and sort for determinism.
    let mut likely_vec: Vec<_> = likely.into_iter().collect();
    likely_vec.sort();
    scene_hints.sort_by(|a, b| {
        b.estimated_score
            .partial_cmp(&a.estimated_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let wall_ms = t_start.elapsed().as_millis().min(u32::MAX as u128) as u32;
    let computed_at_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    PredictedManifest {
        chunks_likely_changed: likely_vec,
        predicted_lava_cells: lava_cells,
        predicted_aureole_glimpse_pos,
        predicted_aureole_block,
        predicted_scene_hints: scene_hints,
        computed_at_secs,
        wall_ms,
        sleep_count: snap.sleep_count,
    }
}

// ── Internal helpers ────────────────────────────────────────────────

struct HeatZone {
    centroid: (i32, i32, i32),
    cell_count: u32,
}

const FACE_OFFSETS: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

/// Cluster heat sources by face-adjacency. Returns (zones, all_lava_cells).
fn cluster_heat_zones(heat_map: &[HeatSource]) -> (Vec<HeatZone>, Vec<(i32, i32, i32)>) {
    // Extract lava cells (kimberlite voxels are heat sources but not "lava cells"
    // for the montage's purposes).
    let mut lava_positions: Vec<(i32, i32, i32)> = heat_map
        .iter()
        .filter(|h| h.source_type == HeatSourceType::Lava)
        .map(|h| h.pos)
        .collect();
    lava_positions.sort();
    lava_positions.dedup();

    // Cluster ALL heat sources (lava + kimberlite) into zones for the
    // aureole glimpse — kimberlite contributes too (matches the real
    // `apply_aureole` behavior).
    let mut all_positions: Vec<(i32, i32, i32)> = heat_map.iter().map(|h| h.pos).collect();
    all_positions.sort();
    all_positions.dedup();
    let pos_set: HashSet<(i32, i32, i32)> = all_positions.iter().copied().collect();
    let mut visited: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut zones = Vec::new();

    const MIN_ZONE_SIZE: u32 = 4;

    for &pos in &all_positions {
        if visited.contains(&pos) {
            continue;
        }
        let mut queue = VecDeque::new();
        let mut component: Vec<(i32, i32, i32)> = Vec::new();
        queue.push_back(pos);
        visited.insert(pos);
        while let Some(cur) = queue.pop_front() {
            component.push(cur);
            for &(dx, dy, dz) in &FACE_OFFSETS {
                let nb = (cur.0 + dx, cur.1 + dy, cur.2 + dz);
                if pos_set.contains(&nb) && visited.insert(nb) {
                    queue.push_back(nb);
                }
            }
        }
        if (component.len() as u32) < MIN_ZONE_SIZE {
            continue;
        }
        let n = component.len() as i64;
        let (sx, sy, sz) = component.iter().fold((0i64, 0i64, 0i64), |(ax, ay, az), &(x, y, z)| {
            (ax + x as i64, ay + y as i64, az + z as i64)
        });
        let centroid = ((sx / n) as i32, (sy / n) as i32, (sz / n) as i32);
        zones.push(HeatZone {
            centroid,
            cell_count: component.len() as u32,
        });
    }
    // Sort by size desc for picking biggest.
    zones.sort_by(|a, b| b.cell_count.cmp(&a.cell_count));
    (zones, lava_positions)
}

fn pick_aureole(
    zones: &[HeatZone],
    chunk_size: usize,
) -> (Option<(i32, i32, i32)>, Option<Vec<(i32, i32, i32)>>) {
    let cs = chunk_size as i32;
    let biggest = match zones.first() {
        Some(z) => z,
        None => return (None, None),
    };
    let glimpse = biggest.centroid;
    let gx = glimpse.0.div_euclid(cs);
    let gy = glimpse.1.div_euclid(cs);
    let gz = glimpse.2.div_euclid(cs);
    let mut block = Vec::with_capacity(27);
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                block.push((gx + dx, gy + dy, gz + dz));
            }
        }
    }
    (Some(glimpse), Some(block))
}

fn avg_chunk_stress(sf: &StressField) -> f32 {
    if sf.stress.is_empty() {
        return 0.0;
    }
    let sum: f32 = sf.stress.iter().sum();
    sum / sf.stress.len() as f32
}

fn water_scene_hints(snap: &FluidSnapshot, chunk_size_voxels: i32) -> Vec<SceneHint> {
    // Aggregate water cells per chunk; emit a hint per chunk with non-trivial
    // water (≥ 24 cells, matching legacy MIN_WATER_VOTES at
    // voxel-ffi/src/poi_scanner.rs:43).
    let cs = snap.chunk_size as i32;
    let mut by_chunk: HashMap<(i32, i32, i32), (u32, i64, i64, i64)> = HashMap::new();
    for (&chunk_key, cells) in &snap.chunks {
        let (cx, cy, cz) = chunk_key;
        let mut count = 0u32;
        let mut sx: i64 = 0;
        let mut sy: i64 = 0;
        let mut sz: i64 = 0;
        for z in 0..snap.chunk_size {
            for y in 0..snap.chunk_size {
                for x in 0..snap.chunk_size {
                    let idx = z * snap.chunk_size * snap.chunk_size + y * snap.chunk_size + x;
                    if idx < cells.len() && cells[idx].level > 0.01 && cells[idx].fluid_type.is_water()
                    {
                        let wx = cx * cs + x as i32;
                        let wy = cy * cs + y as i32;
                        let wz = cz * cs + z as i32;
                        count += 1;
                        sx += wx as i64;
                        sy += wy as i64;
                        sz += wz as i64;
                    }
                }
            }
        }
        if count >= 24 {
            by_chunk.insert(chunk_key, (count, sx, sy, sz));
        }
    }
    by_chunk
        .into_iter()
        .map(|(chunk, (count, sx, sy, sz))| SceneHint {
            kind: 1, // Water
            world_pos: (
                (sx / count as i64) as i32,
                (sy / count as i64) as i32,
                (sz / count as i64) as i32,
            ),
            estimated_score: (count as f32 * 6.0).min(2000.0),
            chunk_coord: (
                chunk.0 * cs / chunk_size_voxels.max(1),
                chunk.1 * cs / chunk_size_voxels.max(1),
                chunk.2 * cs / chunk_size_voxels.max(1),
            ),
        })
        .collect()
}

fn world_to_chunk(pos: (i32, i32, i32), chunk_size: i32) -> (i32, i32, i32) {
    (
        pos.0.div_euclid(chunk_size),
        pos.1.div_euclid(chunk_size),
        pos.2.div_euclid(chunk_size),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use voxel_fluid::cell::{FluidCell, FluidType};

    fn empty_snapshot(player_chunk: (i32, i32, i32), sleep_count: u32) -> PredictSnapshot {
        PredictSnapshot::new(
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            FluidSnapshot {
                chunks: HashMap::new(),
                chunk_size: 30,
            },
            player_chunk,
            sleep_count,
            30,
            3,
        )
    }

    fn fluid_snapshot_with_lava(positions: &[(i32, i32, i32)]) -> FluidSnapshot {
        let cs = 30usize;
        let mut chunks: HashMap<(i32, i32, i32), Vec<FluidCell>> = HashMap::new();
        for &(wx, wy, wz) in positions {
            let cx = wx.div_euclid(cs as i32);
            let cy = wy.div_euclid(cs as i32);
            let cz = wz.div_euclid(cs as i32);
            let lx = wx.rem_euclid(cs as i32) as usize;
            let ly = wy.rem_euclid(cs as i32) as usize;
            let lz = wz.rem_euclid(cs as i32) as usize;
            let entry = chunks
                .entry((cx, cy, cz))
                .or_insert_with(|| vec![FluidCell::default(); cs * cs * cs]);
            let idx = lz * cs * cs + ly * cs + lx;
            entry[idx].level = 1.0;
            entry[idx].fluid_type = FluidType::Lava;
        }
        FluidSnapshot {
            chunks,
            chunk_size: cs,
        }
    }

    fn density_field(size: usize, material: Material) -> DensityField {
        let mut df = DensityField::new(size);
        for y in 0..size {
            for z in 0..size {
                for x in 0..size {
                    df.get_mut(x, y, z).material = material;
                }
            }
        }
        df
    }

    #[test]
    fn predictor_seed_offset_distinct_from_real() {
        // Real sleep at sleep_count=0 would seed with 42.
        let real_seed = 0u64 * 7919 + 42;
        let pred_seed = predictor_seed(0);
        assert_ne!(real_seed, pred_seed);
        assert_eq!(pred_seed, 42 + 0x1_0000_0000);
    }

    #[test]
    fn empty_world_produces_empty_manifest() {
        let snap = empty_snapshot((0, 0, 0), 0);
        let m = predict_next_sleep(&snap);
        assert!(m.chunks_likely_changed.is_empty());
        assert!(m.predicted_lava_cells.is_empty());
        assert!(m.predicted_aureole_glimpse_pos.is_none());
        assert!(m.predicted_scene_hints.is_empty());
        assert!(m.wall_ms <= 100); // empty world is fast
    }

    #[test]
    fn lava_cluster_drives_aureole_glimpse() {
        // 8 face-adjacent lava cells in a 2x2x2 cube → 1 zone.
        let positions: Vec<_> = (0..2)
            .flat_map(|x| (0..2).flat_map(move |y| (0..2).map(move |z| (x, y, z))))
            .collect();
        let mut snap = empty_snapshot((0, 0, 0), 1);
        snap.fluid_snapshot = fluid_snapshot_with_lava(&positions);
        // Insert dummy density fields so heat_map's chunk iteration has data
        // (in our case, only fluid lava cells produce heat sources, so this
        // is mostly a smoke check for the chunk listing).
        snap.density_fields
            .insert((0, 0, 0), density_field(31, Material::Granite));

        let m = predict_next_sleep(&snap);
        assert_eq!(m.predicted_lava_cells.len(), 8);
        assert!(m.predicted_aureole_glimpse_pos.is_some());
        // Glimpse centroid should be near (0.5, 0.5, 0.5) — i64 div by n
        // rounds toward zero, so (0..2 avg → 0).
        let (gx, gy, gz) = m.predicted_aureole_glimpse_pos.unwrap();
        assert!(gx.abs() <= 1, "gx={}", gx);
        assert!(gy.abs() <= 1, "gy={}", gy);
        assert!(gz.abs() <= 1, "gz={}", gz);
        assert_eq!(m.predicted_aureole_block.as_ref().unwrap().len(), 27);
        assert!(!m.chunks_likely_changed.is_empty());
    }

    #[test]
    fn small_lava_zone_below_min_size_ignored() {
        // 2 lava cells (below MIN_ZONE_SIZE = 4) → no aureole zone.
        let positions = [(0, 0, 0), (1, 0, 0)];
        let mut snap = empty_snapshot((0, 0, 0), 0);
        snap.fluid_snapshot = fluid_snapshot_with_lava(&positions);
        snap.density_fields
            .insert((0, 0, 0), density_field(31, Material::Granite));
        let m = predict_next_sleep(&snap);
        assert!(m.predicted_aureole_glimpse_pos.is_none());
    }

    #[test]
    fn high_stress_chunk_emits_likely_changed_and_hint() {
        let mut snap = empty_snapshot((0, 0, 0), 0);
        snap.density_fields
            .insert((0, 0, 0), density_field(31, Material::Granite));
        // Construct a StressField with avg > 0.5 (threshold).
        let mut sf = StressField::new(30);
        for cell in sf.stress.iter_mut() {
            *cell = 0.8;
        }
        snap.stress_fields.insert((0, 0, 0), sf);

        let m = predict_next_sleep(&snap);
        assert!(m.chunks_likely_changed.contains(&(0, 0, 0)));
        assert!(m
            .predicted_scene_hints
            .iter()
            .any(|h| h.kind == 2 /* Stress */));
    }

    #[test]
    fn determinism_same_input_same_output() {
        let mut snap = empty_snapshot((0, 0, 0), 42);
        snap.density_fields
            .insert((0, 0, 0), density_field(31, Material::Granite));
        snap.fluid_snapshot = fluid_snapshot_with_lava(&[(5, 5, 5), (6, 5, 5), (5, 6, 5), (5, 5, 6)]);
        let m1 = predict_next_sleep(&snap);
        let m2 = predict_next_sleep(&snap);
        assert_eq!(m1.chunks_likely_changed, m2.chunks_likely_changed);
        assert_eq!(m1.predicted_lava_cells, m2.predicted_lava_cells);
        assert_eq!(m1.predicted_aureole_glimpse_pos, m2.predicted_aureole_glimpse_pos);
    }

    #[test]
    fn loaded_in_range_uses_chebyshev() {
        let mut snap = empty_snapshot((10, 10, 10), 0);
        snap.chunk_radius = 2;
        // In range:
        snap.density_fields
            .insert((10, 10, 10), density_field(31, Material::Granite));
        snap.density_fields
            .insert((12, 10, 10), density_field(31, Material::Granite));
        // Out of range (Chebyshev distance 3):
        snap.density_fields
            .insert((13, 10, 10), density_field(31, Material::Granite));
        let in_range = snap.loaded_in_range();
        assert_eq!(in_range.len(), 2);
        assert!(in_range.contains(&(10, 10, 10)));
        assert!(in_range.contains(&(12, 10, 10)));
        assert!(!in_range.contains(&(13, 10, 10)));
    }

    #[test]
    fn wall_ms_is_populated() {
        let snap = empty_snapshot((0, 0, 0), 0);
        let m = predict_next_sleep(&snap);
        // u32 wall_ms is set; even an empty snapshot may report 0 or 1 ms.
        // Just confirm it's a real value, not magic.
        assert!(m.wall_ms < 30_000); // < 30s sanity bound
    }
}
