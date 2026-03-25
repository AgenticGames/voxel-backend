//! Zone detection: air volume census, cavern clustering, and zone type selection.

use std::collections::{HashMap, HashSet, VecDeque};

use glam::Vec3;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::Rng;

use voxel_core::material::Material;

use crate::config::{ZoneConfig, ZoneType};
use crate::density::DensityField;

/// Per-chunk air volume statistics.
#[derive(Debug, Clone)]
pub struct ChunkAirStats {
    pub chunk_key: (i32, i32, i32),
    pub air_count: u32,
    pub air_bbox_min: [u8; 3],
    pub air_bbox_max: [u8; 3],
    pub avg_floor_y: f32,
    pub dominant_host_rock: Material,
    pub world_y_center: f32,
}

/// A cluster of connected chunks sharing a large air volume.
#[derive(Debug, Clone)]
pub struct CavernVolume {
    pub chunk_keys: Vec<(i32, i32, i32)>,
    pub total_air: u32,
    pub world_center: Vec3,
    pub world_bbox_min: Vec3,
    pub world_bbox_max: Vec3,
    pub dominant_material: Material,
    pub avg_depth: f32,
}

/// Compute air statistics for all density fields in the region.
pub fn compute_air_stats(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    effective_bounds: f32,
) -> HashMap<(i32, i32, i32), ChunkAirStats> {
    let mut stats = HashMap::new();

    let mut sorted_keys: Vec<_> = density_fields.keys().copied().collect();
    sorted_keys.sort();

    for key in sorted_keys {
        let density = &density_fields[&key];
        let size = density.size;
        let mut air_count = 0u32;
        let mut bbox_min = [255u8; 3];
        let mut bbox_max = [0u8; 3];
        let mut floor_y_sum = 0.0f64;
        let mut floor_count = 0u32;
        let mut material_counts: HashMap<Material, u32> = HashMap::new();

        for z in 1..size - 1 {
            for y in 1..size - 1 {
                for x in 1..size - 1 {
                    let idx = z * size * size + y * size + x;
                    let sample = &density.samples[idx];
                    if sample.density <= 0.0 {
                        air_count += 1;
                        bbox_min[0] = bbox_min[0].min(x as u8);
                        bbox_min[1] = bbox_min[1].min(y as u8);
                        bbox_min[2] = bbox_min[2].min(z as u8);
                        bbox_max[0] = bbox_max[0].max(x as u8);
                        bbox_max[1] = bbox_max[1].max(y as u8);
                        bbox_max[2] = bbox_max[2].max(z as u8);

                        // Check if solid below → floor-adjacent air
                        if y > 0 {
                            let below_idx = z * size * size + (y - 1) * size + x;
                            if density.samples[below_idx].density > 0.0 {
                                floor_y_sum += y as f64;
                                floor_count += 1;
                            }
                        }
                    } else if sample.material.is_host_rock() {
                        *material_counts.entry(sample.material).or_insert(0) += 1;
                    }
                }
            }
        }

        let dominant_host_rock = material_counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(mat, _)| mat)
            .unwrap_or(Material::Limestone);

        let avg_floor_y = if floor_count > 0 {
            (floor_y_sum / floor_count as f64) as f32
        } else {
            (size / 2) as f32
        };

        let world_y_center = key.1 as f32 * effective_bounds
            + (size as f32 / 2.0) * (effective_bounds / (size - 1) as f32);

        stats.insert(key, ChunkAirStats {
            chunk_key: key,
            air_count,
            air_bbox_min: bbox_min,
            air_bbox_max: bbox_max,
            avg_floor_y,
            dominant_host_rock,
            world_y_center,
        });
    }

    stats
}

/// Cluster adjacent chunks with sufficient air into CavernVolumes via BFS.
pub fn cluster_cavern_volumes(
    air_stats: &HashMap<(i32, i32, i32), ChunkAirStats>,
    effective_bounds: f32,
    min_air_threshold: u32,
) -> Vec<CavernVolume> {
    let eligible: HashSet<(i32, i32, i32)> = air_stats
        .iter()
        .filter(|(_, s)| s.air_count >= min_air_threshold)
        .map(|(k, _)| *k)
        .collect();

    let mut visited: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut volumes = Vec::new();

    // Sort for deterministic iteration
    let mut eligible_sorted: Vec<_> = eligible.iter().copied().collect();
    eligible_sorted.sort();

    for start in eligible_sorted {
        if visited.contains(&start) {
            continue;
        }

        // BFS flood fill over 26-neighbors
        let mut cluster = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited.insert(start);

        while let Some(key) = queue.pop_front() {
            cluster.push(key);
            for dz in -1..=1i32 {
                for dy in -1..=1i32 {
                    for dx in -1..=1i32 {
                        if dx == 0 && dy == 0 && dz == 0 {
                            continue;
                        }
                        let neighbor = (key.0 + dx, key.1 + dy, key.2 + dz);
                        if eligible.contains(&neighbor) && !visited.contains(&neighbor) {
                            visited.insert(neighbor);
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }

        // Compute volume stats
        let total_air: u32 = cluster.iter()
            .map(|k| air_stats[k].air_count)
            .sum();

        let mut bbox_min = Vec3::splat(f32::MAX);
        let mut bbox_max = Vec3::splat(f32::MIN);
        let mut center_sum = Vec3::ZERO;
        let mut depth_sum = 0.0f32;
        let mut mat_counts: HashMap<Material, u32> = HashMap::new();

        for k in &cluster {
            let s = &air_stats[k];
            let chunk_world_min = Vec3::new(
                k.0 as f32 * effective_bounds,
                k.1 as f32 * effective_bounds,
                k.2 as f32 * effective_bounds,
            );
            let chunk_world_max = chunk_world_min + Vec3::splat(effective_bounds);
            bbox_min = bbox_min.min(chunk_world_min);
            bbox_max = bbox_max.max(chunk_world_max);
            center_sum += (chunk_world_min + chunk_world_max) * 0.5;
            depth_sum += s.world_y_center;
            *mat_counts.entry(s.dominant_host_rock).or_insert(0) += s.air_count;
        }

        let count = cluster.len() as f32;
        let world_center = center_sum / count;
        let avg_depth = depth_sum / count;

        let dominant_material = mat_counts
            .into_iter()
            .max_by_key(|&(_, c)| c)
            .map(|(m, _)| m)
            .unwrap_or(Material::Limestone);

        volumes.push(CavernVolume {
            chunk_keys: cluster,
            total_air,
            world_center,
            world_bbox_min: bbox_min,
            world_bbox_max: bbox_max,
            dominant_material,
            avg_depth,
        });
    }

    // Sort volumes by total_air descending for priority (largest volumes get first pick)
    volumes.sort_by(|a, b| b.total_air.cmp(&a.total_air));
    volumes
}

/// Select a zone type for a CavernVolume based on eligibility and RNG.
/// Returns None if no zone is selected (all rolls fail).
pub fn select_zone_type(
    volume: &CavernVolume,
    config: &ZoneConfig,
    global_seed: u64,
) -> Option<ZoneType> {
    // Deterministic seed from volume's minimum chunk coordinate
    let min_key = volume.chunk_keys.iter().copied().min().unwrap_or((0, 0, 0));
    let zone_seed = global_seed
        .wrapping_add(0xCAFE_20DE)
        .wrapping_add((min_key.0 as u64).wrapping_mul(0x9E3779B97F4A7C15))
        ^ (min_key.1 as u64).wrapping_mul(0x517CC1B727220A95)
        ^ (min_key.2 as u64).wrapping_mul(0x6C62272E07BB0142);
    let mut rng = ChaCha8Rng::seed_from_u64(zone_seed);

    let extent = volume.world_bbox_max - volume.world_bbox_min;
    let max_dim = extent.x.max(extent.y).max(extent.z);
    let min_dim = extent.x.min(extent.y).min(extent.z);
    let aspect_ratio = if min_dim > 0.0 { max_dim / min_dim } else { 1.0 };

    // Priority-ordered zone type checks
    let candidates: &[(ZoneType, u32, f32, Option<fn(&CavernVolume, f32) -> bool>)] = &[
        (ZoneType::Cathedral, config.cathedral_min_air, config.cathedral_chance,
         Some(|v: &CavernVolume, _| v.dominant_material == Material::Limestone || v.dominant_material == Material::Marble)),
        (ZoneType::SubterraneanLake, config.lake_min_air, config.lake_chance, None),
        (ZoneType::RiverCanyon, config.canyon_min_air, config.canyon_chance,
         Some(|_: &CavernVolume, ar: f32| ar >= 2.5)), // elongated shape required
        (ZoneType::LavaTubeGallery, config.lava_gallery_min_air, config.lava_gallery_chance,
         Some(|v: &CavernVolume, _| v.avg_depth < -50.0 || v.dominant_material == Material::Basalt)),
        (ZoneType::GeothermalTerraces, config.terraces_min_air, config.terraces_chance,
         Some(|v: &CavernVolume, _| v.avg_depth < -30.0)),
        (ZoneType::BioluminescentGrotto, config.bioluminescent_min_air, config.bioluminescent_chance, None),
        (ZoneType::FrozenGrotto, config.frozen_min_air, config.frozen_chance,
         Some(|v: &CavernVolume, _| v.avg_depth > 100.0)),
    ];

    for &(zone_type, min_air, chance, ref extra_check) in candidates {
        if chance <= 0.0 {
            continue;
        }
        if volume.total_air < min_air {
            continue;
        }
        if let Some(check) = extra_check {
            if !check(volume, aspect_ratio) {
                continue;
            }
        }
        let roll: f32 = rng.gen();
        if roll < chance {
            return Some(zone_type);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::density::DensityField;

    fn make_density_with_air(size: usize, air_count: usize) -> DensityField {
        let mut df = DensityField::new(size);
        // Fill center with air
        let mut placed = 0;
        for z in 1..size - 1 {
            for y in 1..size - 1 {
                for x in 1..size - 1 {
                    if placed < air_count {
                        let idx = z * size * size + y * size + x;
                        df.samples[idx].density = -1.0;
                        df.samples[idx].material = Material::Air;
                        placed += 1;
                    }
                }
            }
        }
        df
    }

    #[test]
    fn air_stats_counts_air() {
        let mut fields = HashMap::new();
        fields.insert((0, 0, 0), make_density_with_air(17, 100));
        let stats = compute_air_stats(&fields, 16.0);
        assert_eq!(stats[&(0, 0, 0)].air_count, 100);
    }

    #[test]
    fn clustering_groups_adjacent() {
        let mut fields = HashMap::new();
        fields.insert((0, 0, 0), make_density_with_air(17, 200));
        fields.insert((1, 0, 0), make_density_with_air(17, 200));
        fields.insert((2, 0, 0), make_density_with_air(17, 200));
        // Disconnected chunk
        fields.insert((5, 5, 5), make_density_with_air(17, 200));

        let stats = compute_air_stats(&fields, 16.0);
        let volumes = cluster_cavern_volumes(&stats, 16.0, 64);
        assert_eq!(volumes.len(), 2);
        // Largest cluster should have 3 chunks
        assert_eq!(volumes[0].chunk_keys.len(), 3);
        assert_eq!(volumes[0].total_air, 600);
    }

    #[test]
    fn zone_selection_deterministic() {
        let volume = CavernVolume {
            chunk_keys: vec![(0, 0, 0), (1, 0, 0)],
            total_air: 3000,
            world_center: Vec3::new(16.0, 8.0, 8.0),
            world_bbox_min: Vec3::ZERO,
            world_bbox_max: Vec3::new(32.0, 16.0, 16.0),
            dominant_material: Material::Limestone,
            avg_depth: 0.0,
        };
        let config = ZoneConfig { enabled: true, ..ZoneConfig::default() };
        let r1 = select_zone_type(&volume, &config, 42);
        let r2 = select_zone_type(&volume, &config, 42);
        assert_eq!(r1, r2);
    }
}
