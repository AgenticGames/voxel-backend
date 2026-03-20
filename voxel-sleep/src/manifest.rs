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
    pub fn compact(&mut self) {
        for delta in self.chunk_deltas.values_mut() {
            // For voxel changes: merge first.old + last.new per (lx, ly, lz)
            let mut first_idx: HashMap<(usize, usize, usize), usize> = HashMap::new();
            let mut last_idx: HashMap<(usize, usize, usize), usize> = HashMap::new();
            for (i, change) in delta.voxel_changes.iter().enumerate() {
                let key = (change.lx, change.ly, change.lz);
                first_idx.entry(key).or_insert(i);
                last_idx.insert(key, i);
            }
            let mut keys: Vec<(usize, usize, usize)> = first_idx.keys().copied().collect();
            keys.sort();
            let mut compacted = Vec::with_capacity(keys.len());
            for key in keys {
                let fi = first_idx[&key];
                let li = last_idx[&key];
                let first = &delta.voxel_changes[fi];
                let last = &delta.voxel_changes[li];
                compacted.push(VoxelChange {
                    lx: first.lx,
                    ly: first.ly,
                    lz: first.lz,
                    old_material: first.old_material,
                    old_density: first.old_density,
                    new_material: last.new_material,
                    new_density: last.new_density,
                    spread_distance: first.spread_distance,
                });
            }
            delta.voxel_changes = compacted;

            // Same for support changes: keep first.old + last.new
            let mut first_s: HashMap<(usize, usize, usize), usize> = HashMap::new();
            let mut last_s: HashMap<(usize, usize, usize), usize> = HashMap::new();
            for (i, change) in delta.support_changes.iter().enumerate() {
                let key = (change.lx, change.ly, change.lz);
                first_s.entry(key).or_insert(i);
                last_s.insert(key, i);
            }
            let mut keys_s: Vec<(usize, usize, usize)> = first_s.keys().copied().collect();
            keys_s.sort();
            let mut compacted_s = Vec::with_capacity(keys_s.len());
            for key in keys_s {
                let fi = first_s[&key];
                let li = last_s[&key];
                let first = &delta.support_changes[fi];
                let last = &delta.support_changes[li];
                compacted_s.push(SupportChange {
                    lx: first.lx,
                    ly: first.ly,
                    lz: first.lz,
                    old_support: first.old_support,
                    new_support: last.new_support,
                });
            }
            delta.support_changes = compacted_s;
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
