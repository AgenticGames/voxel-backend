use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use voxel_core::density::DensityField;
use voxel_core::material::Material;
use voxel_core::stress::{SupportField, SupportType};

/// Records a single voxel change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoxelChange {
    /// Local coordinates within the chunk
    pub lx: usize,
    pub ly: usize,
    pub lz: usize,
    pub old_material: u8,
    pub old_density: f32,
    pub new_material: u8,
    pub new_density: f32,
    /// Normalized distance from heat source (0.0 = at source, 1.0 = farthest).
    /// Controls spreading morph animation order during sleep montage.
    #[serde(default)]
    pub spread_distance: f32,
}

/// Records a single support change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupportChange {
    pub lx: usize,
    pub ly: usize,
    pub lz: usize,
    pub old_support: u8,
    pub new_support: u8,
}

/// All changes for a single chunk.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChunkDelta {
    pub voxel_changes: Vec<VoxelChange>,
    pub support_changes: Vec<SupportChange>,
    /// True for chunks injected post-sleep by the montage POI system that
    /// have no per-voxel diff data — the morph step then runs a procedural
    /// "rise from air to current state" animation across all solid voxels
    /// instead of interpolating recorded VoxelChanges. Used for POI plays
    /// when the POI's chunks weren't affected by execute_sleep (e.g. crystal
    /// bridges, or a lava chunk that was already there).
    #[serde(default)]
    pub synthesize_growth: bool,
    /// Growth "sources" in WORLD voxel coordinates (Rust Y-up). When non-empty
    /// and `synthesize_growth = true`, the morph step computes each voxel's
    /// reveal timing from its min-distance to any source (normalized by
    /// `growth_source_max_dist`). Bridges get 2 sources (anchor A + anchor B)
    /// so the bridge animates growing inward from each end; other POIs get 1
    /// source (chunk center) for a radial reveal.
    /// When empty, the synthesize path falls back to a y-axis gradient
    /// ("rise from below").
    #[serde(default)]
    pub growth_sources: Vec<(f32, f32, f32)>,
    /// Distance (in world voxels) at which the spread normalization saturates
    /// to 1.0. Smaller → tighter, faster reveal; larger → slower, more
    /// gradual. Bridge plays use half the bridge length so anchors animate
    /// at t≈0 and midpoint at t≈1.
    #[serde(default)]
    pub growth_source_max_dist: f32,
}

impl ChunkDelta {
    /// True when this chunk's morph is a PURE RECOLOR — no change moves the DC
    /// surface and there's no synthesized growth. A dual-contouring surface only
    /// exists/moves where density crosses the 0.0 solid/air threshold, so a change
    /// shifts geometry iff it flips a voxel's sign across 0.0. Metamorphism and
    /// hydrothermal ore deposition leave density untouched (old_density==new_density,
    /// both >0) → no surface movement → qualify. Erosion (solid→air) and formation
    /// growth (air→solid) flip the sign → do NOT qualify. Synthesized "rise from air"
    /// growth animates density up from -1.0 → never a pure recolor.
    ///
    /// Pure-recolor chunks can be meshed ONCE and recolored per reveal step (per-vertex
    /// material reassign) instead of re-running dual-contouring every step, since the
    /// triangles are byte-identical at every step.
    ///
    /// We require density EQUALITY (not merely same-sign): the recolor fast path freezes
    /// the DC geometry, but a DC vertex position derives from the density-interpolated
    /// edge crossing `t = da/(da-db)`, so even a same-sign density change (e.g. 1.0→0.8)
    /// would shift the surface sub-voxel in the full pipeline and diverge if frozen. All
    /// real recolor producers (metamorphism, hydrothermal ore, gypsum, deeptime non-
    /// formation) record `new_density == old_density` exactly (set_voxel_synced material-
    /// only, density `None`); every genuine geometry change flips the sign (Air↔solid).
    /// So this is the same classification in practice today, but self-contained — a future
    /// change that recorded a same-sign density delta correctly falls to the slow path
    /// instead of silently freezing geometry that should move.
    pub fn is_pure_recolor(&self) -> bool {
        !self.synthesize_growth
            && self
                .voxel_changes
                .iter()
                .all(|c| (c.old_density - c.new_density).abs() < 1e-6)
    }
}

/// Custom serde module for HashMap<(i32,i32,i32), ChunkDelta> using string keys.
/// JSON requires string keys, so we serialize tuple keys as "x,y,z".
mod chunk_deltas_serde {
    use super::{ChunkDelta, HashMap};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(
        map: &HashMap<(i32, i32, i32), ChunkDelta>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let string_map: HashMap<String, &ChunkDelta> = map
            .iter()
            .map(|((x, y, z), v)| (format!("{},{},{}", x, y, z), v))
            .collect();
        string_map.serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<HashMap<(i32, i32, i32), ChunkDelta>, D::Error> {
        let string_map: HashMap<String, ChunkDelta> = HashMap::deserialize(deserializer)?;
        string_map
            .into_iter()
            .map(|(k, v)| {
                let parts: Vec<&str> = k.split(',').collect();
                if parts.len() != 3 {
                    return Err(serde::de::Error::custom(format!(
                        "invalid chunk key: '{}'", k
                    )));
                }
                let x = parts[0].parse::<i32>().map_err(serde::de::Error::custom)?;
                let y = parts[1].parse::<i32>().map_err(serde::de::Error::custom)?;
                let z = parts[2].parse::<i32>().map_err(serde::de::Error::custom)?;
                Ok(((x, y, z), v))
            })
            .collect()
    }
}

/// Manifest tracking all world modifications across sleep cycles.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChangeManifest {
    #[serde(with = "chunk_deltas_serde")]
    pub chunk_deltas: HashMap<(i32, i32, i32), ChunkDelta>,
    pub sleep_count: u32,
}

/// Stable-sort + in-place run-length coalesce for `compact()`. Keeps
/// first.old_* + last.new_* per (lx,ly,lz) run, preserving the first
/// entry's spread_distance (aureole-driven reveal order).
fn compact_voxel_changes(changes: &mut Vec<VoxelChange>) {
    let n = changes.len();
    if n <= 1 {
        return;
    }
    changes.sort_by_key(|c| (c.lx, c.ly, c.lz));
    let mut write = 0usize;
    let mut read = 0usize;
    while read < n {
        let start = read;
        let key = (changes[start].lx, changes[start].ly, changes[start].lz);
        let mut end = start + 1;
        while end < n {
            let c = &changes[end];
            if (c.lx, c.ly, c.lz) != key {
                break;
            }
            end += 1;
        }
        let first = changes[start].clone();
        let last_new_material = changes[end - 1].new_material;
        let last_new_density = changes[end - 1].new_density;
        changes[write] = VoxelChange {
            lx: first.lx,
            ly: first.ly,
            lz: first.lz,
            old_material: first.old_material,
            old_density: first.old_density,
            new_material: last_new_material,
            new_density: last_new_density,
            spread_distance: first.spread_distance,
        };
        write += 1;
        read = end;
    }
    changes.truncate(write);
}

/// Stable-sort + in-place run-length coalesce for support changes
/// (mirrors `compact_voxel_changes`).
fn compact_support_changes(changes: &mut Vec<SupportChange>) {
    let n = changes.len();
    if n <= 1 {
        return;
    }
    changes.sort_by_key(|c| (c.lx, c.ly, c.lz));
    let mut write = 0usize;
    let mut read = 0usize;
    while read < n {
        let start = read;
        let key = (changes[start].lx, changes[start].ly, changes[start].lz);
        let mut end = start + 1;
        while end < n {
            let c = &changes[end];
            if (c.lx, c.ly, c.lz) != key {
                break;
            }
            end += 1;
        }
        let first = changes[start].clone();
        let last_new_support = changes[end - 1].new_support;
        changes[write] = SupportChange {
            lx: first.lx,
            ly: first.ly,
            lz: first.lz,
            old_support: first.old_support,
            new_support: last_new_support,
        };
        write += 1;
        read = end;
    }
    changes.truncate(write);
}

impl ChangeManifest {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a voxel change in the manifest (spread_distance defaults to 0.0).
    pub fn record_voxel_change(
        &mut self,
        chunk: (i32, i32, i32),
        lx: usize, ly: usize, lz: usize,
        old_material: Material, old_density: f32,
        new_material: Material, new_density: f32,
    ) {
        self.record_voxel_change_with_spread(chunk, lx, ly, lz, old_material, old_density, new_material, new_density, 0.0);
    }

    /// Record a voxel change with explicit spread_distance for morph animation ordering.
    pub fn record_voxel_change_with_spread(
        &mut self,
        chunk: (i32, i32, i32),
        lx: usize, ly: usize, lz: usize,
        old_material: Material, old_density: f32,
        new_material: Material, new_density: f32,
        spread_distance: f32,
    ) {
        let delta = self.chunk_deltas.entry(chunk).or_default();
        delta.voxel_changes.push(VoxelChange {
            lx, ly, lz,
            old_material: old_material as u8,
            old_density,
            new_material: new_material as u8,
            new_density,
            spread_distance,
        });
    }

    /// Record a support change in the manifest.
    pub fn record_support_change(
        &mut self,
        chunk: (i32, i32, i32),
        lx: usize, ly: usize, lz: usize,
        old_support: SupportType, new_support: SupportType,
    ) {
        let delta = self.chunk_deltas.entry(chunk).or_default();
        delta.support_changes.push(SupportChange {
            lx, ly, lz,
            old_support: old_support as u8,
            new_support: new_support as u8,
        });
    }

    /// Mirror boundary-face voxel changes into neighbor chunks so the montage
    /// morph can rewind BOTH sides of a chunk seam.
    ///
    /// During sleep, a voxel on a chunk boundary has its DENSITY written into
    /// both adjacent chunks (via `set_voxel_synced`), but the manifest
    /// VoxelChange is recorded on only the owning chunk. The morph reveal
    /// rewinds from the final (post-sleep) state per recorded change, so the
    /// un-recorded mirror side stays at its final value — and the seam, which
    /// stitches DC vertices across that boundary, reads fully-transformed from
    /// step 0. This pass copies each boundary change into the neighbor chunk(s)
    /// at the mirror local coord (faces → 1 neighbor, edges → 3, corners → 7),
    /// matching `set_voxel_synced` semantics. Call BEFORE `compact()` so any
    /// duplicate (lx,ly,lz) runs in the neighbor coalesce.
    pub fn mirror_boundary_changes(&mut self, chunk_size: usize) {
        let cs = chunk_size;
        // Collect first — can't mutate chunk_deltas while iterating it.
        let mut mirrored: Vec<((i32, i32, i32), VoxelChange)> = Vec::new();
        for (&chunk, delta) in self.chunk_deltas.iter() {
            // Synthesize-growth chunks animate procedurally (no per-voxel diff);
            // don't seed them with mirrored changes (would flip them off that path).
            if delta.synthesize_growth {
                continue;
            }
            for vc in &delta.voxel_changes {
                // Per-axis mirror: (neighbor offset, mirror local coord) or None.
                let mx = if vc.lx == 0 { Some((-1i32, cs)) }
                    else if vc.lx == cs { Some((1i32, 0usize)) } else { None };
                let my = if vc.ly == 0 { Some((-1i32, cs)) }
                    else if vc.ly == cs { Some((1i32, 0usize)) } else { None };
                let mz = if vc.lz == 0 { Some((-1i32, cs)) }
                    else if vc.lz == cs { Some((1i32, 0usize)) } else { None };
                if mx.is_none() && my.is_none() && mz.is_none() {
                    continue; // interior voxel — no seam involvement
                }
                // 2^3 combinations of (primary, mirror) per axis; skip all-primary.
                for xm in 0..2u8 {
                    for ym in 0..2u8 {
                        for zm in 0..2u8 {
                            if xm == 0 && ym == 0 && zm == 0 {
                                continue;
                            }
                            let (ox, nlx) = if xm == 1 {
                                match mx { Some(v) => v, None => continue }
                            } else { (0i32, vc.lx) };
                            let (oy, nly) = if ym == 1 {
                                match my { Some(v) => v, None => continue }
                            } else { (0i32, vc.ly) };
                            let (oz, nlz) = if zm == 1 {
                                match mz { Some(v) => v, None => continue }
                            } else { (0i32, vc.lz) };
                            let nkey = (chunk.0 + ox, chunk.1 + oy, chunk.2 + oz);
                            let mut nvc = vc.clone();
                            nvc.lx = nlx;
                            nvc.ly = nly;
                            nvc.lz = nlz;
                            mirrored.push((nkey, nvc));
                        }
                    }
                }
            }
        }
        for (nkey, vc) in mirrored {
            let delta = self.chunk_deltas.entry(nkey).or_default();
            if delta.synthesize_growth {
                continue;
            }
            delta.voxel_changes.push(vc);
        }
    }

    /// Merge another manifest's changes (from a sleep result) into this one.
    pub fn merge_sleep_changes(&mut self, other: &ChangeManifest) {
        for (chunk, delta) in &other.chunk_deltas {
            let target = self.chunk_deltas.entry(*chunk).or_default();
            target.voxel_changes.extend(delta.voxel_changes.iter().cloned());
            target.support_changes.extend(delta.support_changes.iter().cloned());
        }
        self.sleep_count += other.sleep_count;
    }

    /// Apply this manifest's deltas on top of a freshly generated density field.
    pub fn apply_to_chunk(&self, chunk: (i32, i32, i32), density: &mut DensityField) {
        if let Some(delta) = self.chunk_deltas.get(&chunk) {
            for change in &delta.voxel_changes {
                let sample = density.get_mut(change.lx, change.ly, change.lz);
                sample.material = Material::from_u8(change.new_material);
                sample.density = change.new_density;
            }
        }
    }

    /// Apply support changes on top of a freshly initialized support field.
    pub fn apply_supports_to_chunk(&self, chunk: (i32, i32, i32), supports: &mut SupportField) {
        if let Some(delta) = self.chunk_deltas.get(&chunk) {
            for change in &delta.support_changes {
                supports.set(change.lx, change.ly, change.lz, SupportType::from_u8(change.new_support));
            }
        }
    }

    /// Compact: coalesce multiple changes to the same voxel into one entry.
    /// Keeps the FIRST change's old_material/old_density (true pre-sleep state)
    /// and the LAST change's new_material/new_density (final post-sleep state).
    /// spread_distance is taken from the first change (aureole-driven spread order).
    ///
    /// Implementation: stable sort by (lx, ly, lz) + in-place run-length
    /// coalesce. The previous version built TWO `HashMap<(usize,usize,usize),
    /// usize>` per chunk (first_idx + last_idx) plus a Vec of sorted keys plus
    /// a fresh output Vec — ~3N HashMap ops + 2K lookups + a separate keys.sort.
    /// A deep sleep modifying ~500 chunks with thousands of changes each made
    /// that visible in profile traces. Sort + linear scan does the same work
    /// in one allocation pass, with no hashing, and is much more cache-friendly.
    pub fn compact(&mut self) {
        for delta in self.chunk_deltas.values_mut() {
            compact_voxel_changes(&mut delta.voxel_changes);
            compact_support_changes(&mut delta.support_changes);
        }
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialize from JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_apply() {
        let mut manifest = ChangeManifest::new();
        manifest.record_voxel_change(
            (0, 0, 0), 5, 5, 5,
            Material::Limestone, 1.0,
            Material::Marble, 1.0,
        );

        let mut df = DensityField::new(17);
        // Fill with limestone
        for s in df.samples.iter_mut() {
            s.density = 1.0;
            s.material = Material::Limestone;
        }

        manifest.apply_to_chunk((0, 0, 0), &mut df);
        assert_eq!(df.get(5, 5, 5).material, Material::Marble);
        // Other voxels unchanged
        assert_eq!(df.get(0, 0, 0).material, Material::Limestone);
    }

    #[test]
    fn test_compact() {
        let mut manifest = ChangeManifest::new();
        // Two changes to same voxel -- compact should keep first.old + last.new
        manifest.record_voxel_change(
            (0, 0, 0), 3, 3, 3,
            Material::Limestone, 1.0,
            Material::Granite, 1.0,
        );
        manifest.record_voxel_change(
            (0, 0, 0), 3, 3, 3,
            Material::Granite, 1.0,
            Material::Marble, 0.8,
        );

        manifest.compact();
        let delta = manifest.chunk_deltas.get(&(0, 0, 0)).unwrap();
        assert_eq!(delta.voxel_changes.len(), 1);
        // old_material from FIRST change (true pre-sleep state)
        assert_eq!(delta.voxel_changes[0].old_material, Material::Limestone as u8);
        assert_eq!(delta.voxel_changes[0].old_density, 1.0);
        // new_material from LAST change (final post-sleep state)
        assert_eq!(delta.voxel_changes[0].new_material, Material::Marble as u8);
        assert_eq!(delta.voxel_changes[0].new_density, 0.8);
    }

    #[test]
    fn test_json_roundtrip() {
        let mut manifest = ChangeManifest::new();
        manifest.sleep_count = 3;
        manifest.record_voxel_change(
            (1, 2, 3), 5, 5, 5,
            Material::Copper, 1.0,
            Material::Malachite, 1.0,
        );

        let json = manifest.to_json().unwrap();
        let restored = ChangeManifest::from_json(&json).unwrap();
        assert_eq!(restored.sleep_count, 3);
        assert!(restored.chunk_deltas.contains_key(&(1, 2, 3)));
    }

    #[test]
    fn test_merge() {
        let mut m1 = ChangeManifest::new();
        m1.sleep_count = 1;
        m1.record_voxel_change(
            (0, 0, 0), 1, 1, 1,
            Material::Limestone, 1.0,
            Material::Marble, 1.0,
        );

        let mut m2 = ChangeManifest::new();
        m2.sleep_count = 1;
        m2.record_voxel_change(
            (0, 0, 0), 2, 2, 2,
            Material::Copper, 1.0,
            Material::Malachite, 1.0,
        );

        m1.merge_sleep_changes(&m2);
        assert_eq!(m1.sleep_count, 2);
        let delta = m1.chunk_deltas.get(&(0, 0, 0)).unwrap();
        assert_eq!(delta.voxel_changes.len(), 2);
    }
}
