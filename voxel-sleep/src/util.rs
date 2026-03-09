//! Shared utility functions for sleep phase modules.

use std::collections::{HashMap, HashSet, VecDeque};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use voxel_core::density::DensityField;
use voxel_core::material::Material;
use voxel_core::stress::world_to_chunk_local;

/// 6-connected face-neighbor offsets.
pub const FACE_OFFSETS: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

/// Look up material at a world coordinate, returning None if the chunk is not loaded.
pub fn sample_material(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    wx: i32,
    wy: i32,
    wz: i32,
    chunk_size: usize,
) -> Option<Material> {
    let (chunk_key, lx, ly, lz) = world_to_chunk_local(wx, wy, wz, chunk_size);
    density_fields
        .get(&chunk_key)
        .map(|df| df.get(lx, ly, lz).material)
}

/// Count 6-connected neighbors matching a predicate.
pub fn count_neighbors(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    wx: i32,
    wy: i32,
    wz: i32,
    chunk_size: usize,
    predicate: impl Fn(Material) -> bool,
) -> u32 {
    let mut count = 0u32;
    for &(dx, dy, dz) in &FACE_OFFSETS {
        if let Some(mat) = sample_material(density_fields, wx + dx, wy + dy, wz + dz, chunk_size) {
            if predicate(mat) {
                count += 1;
            }
        }
    }
    count
}

/// Vein deposition probability multiplier based on tunnel aperture (air neighbor count).
/// Wider tunnels = more fluid flow = richer veins; narrow cracks = less deposition.
pub fn aperture_multiplier(air_neighbors: u32) -> f32 {
    match air_neighbors {
        0 => 0.0,
        1 => 1.40,
        2 => 1.15,
        3 => 1.00,
        4 => 0.65,
        5 => 0.40,
        _ => 0.20,
    }
}

/// Check if any voxel within Manhattan distance <= radius has the given material.
pub fn has_material_within_radius(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    wx: i32,
    wy: i32,
    wz: i32,
    chunk_size: usize,
    radius: i32,
    target: Material,
) -> bool {
    for dx in -radius..=radius {
        for dy in -radius..=radius {
            for dz in -radius..=radius {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                if dx.abs() + dy.abs() + dz.abs() > radius {
                    continue;
                }
                if let Some(mat) =
                    sample_material(density_fields, wx + dx, wy + dy, wz + dz, chunk_size)
                {
                    if mat == target {
                        return true;
                    }
                }
            }
        }
    }
    false
}

// ──────────────────────────────────────────────────────────────
// Vein Growth System
// ──────────────────────────────────────────────────────────────

/// Directional bias for vein growth morphology.
#[derive(Debug, Clone)]
pub enum VeinBias {
    /// Iron: favor XZ spread (banded layers)
    Horizontal,
    /// Copper: pick 2 directions when possible (dendritic)
    Branching,
    /// Quartz: compress one axis (0=X, 1=Y, 2=Z) for sheet veins
    Planar(u8),
    /// Gold/Sulfide: tight spherical clusters
    Compact,
}

/// Parameters for growing a connected vein deposit.
#[derive(Debug, Clone)]
pub struct VeinGrowthParams {
    pub ore: Material,
    pub min_size: u32,
    pub max_size: u32,
    pub bias: VeinBias,
}

/// Grow a connected ore vein FROM a seed point INTO surrounding host rock.
///
/// Returns world-coordinate positions to convert to ore. The seed itself
/// is included in the result.
pub fn grow_vein(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    seed: (i32, i32, i32),
    params: &VeinGrowthParams,
    chunk_size: usize,
    rng: &mut ChaCha8Rng,
) -> Vec<(i32, i32, i32)> {
    let target_size = rng.gen_range(params.min_size..=params.max_size) as usize;
    let mut result = vec![seed];
    let mut visited: HashSet<(i32, i32, i32)> = HashSet::new();
    visited.insert(seed);
    let mut frontier: Vec<(i32, i32, i32)> = vec![seed];

    while result.len() < target_size && !frontier.is_empty() {
        // Pick frontier voxel
        let fi = match params.bias {
            VeinBias::Branching => rng.gen_range(0..frontier.len()),
            _ => frontier.len() - 1, // stack-like (depth-first)
        };
        let (fx, fy, fz) = frontier[fi];

        // Collect valid neighbors: solid host rock, not visited, not ore, not air
        let mut candidates: Vec<((i32, i32, i32), f32)> = Vec::new();
        for &(dx, dy, dz) in &FACE_OFFSETS {
            let n = (fx + dx, fy + dy, fz + dz);
            if visited.contains(&n) {
                continue;
            }
            if let Some(mat) = sample_material(density_fields, n.0, n.1, n.2, chunk_size) {
                if !mat.is_host_rock() {
                    continue;
                }
                // Directional weight
                let weight = match &params.bias {
                    VeinBias::Horizontal => {
                        if dy != 0 { 0.3 } else { 1.0 }
                    }
                    VeinBias::Planar(axis) => {
                        let on_compressed = match axis {
                            0 => dx != 0,
                            1 => dy != 0,
                            _ => dz != 0,
                        };
                        if on_compressed { 0.3 } else { 1.0 }
                    }
                    VeinBias::Branching | VeinBias::Compact => 1.0,
                };
                candidates.push((n, weight));
            }
        }

        if candidates.is_empty() {
            frontier.swap_remove(fi);
            continue;
        }

        // Pick how many to add
        let pick_count = match params.bias {
            VeinBias::Branching => 2.min(candidates.len()),
            _ => 1,
        };

        for _ in 0..pick_count {
            if candidates.is_empty() || result.len() >= target_size {
                break;
            }
            // Weighted random selection
            let total_weight: f32 = candidates.iter().map(|(_, w)| w).sum();
            if total_weight <= 0.0 {
                break;
            }
            let mut roll = rng.gen::<f32>() * total_weight;
            let mut chosen_idx = 0;
            for (i, &(_, w)) in candidates.iter().enumerate() {
                roll -= w;
                if roll <= 0.0 {
                    chosen_idx = i;
                    break;
                }
            }
            let (pos, _) = candidates.remove(chosen_idx);
            visited.insert(pos);
            result.push(pos);
            frontier.push(pos);
        }
    }

    result
}

/// Default vein size range for a given ore type.
pub fn default_vein_size(ore: Material) -> (u32, u32) {
    match ore {
        Material::Iron | Material::Copper => (10, 25),
        Material::Tin | Material::Quartz | Material::Pyrite => (6, 15),
        Material::Gold | Material::Sulfide => (3, 8),
        Material::Malachite => (4, 10),
        _ => (4, 12),
    }
}

/// Vein sizes for sleep hydrothermal deposits (thinner than world-gen defaults).
/// Hydrothermal veins precipitate from fluids on cave walls — thin stringers, not ore bodies.
pub fn sleep_vein_size(ore: Material) -> (u32, u32) {
    match ore {
        Material::Iron => (2, 5),
        Material::Copper => (2, 4),
        Material::Tin | Material::Quartz => (1, 3),
        Material::Gold => (1, 2),
        Material::Sulfide => (1, 3),
        Material::Malachite => (1, 3),
        Material::Pyrite => (1, 3),
        _ => (1, 3),
    }
}

/// Default vein bias for a given ore type.
pub fn default_vein_bias(ore: Material, rng: &mut ChaCha8Rng) -> VeinBias {
    match ore {
        Material::Iron => VeinBias::Horizontal,
        Material::Copper => VeinBias::Branching,
        Material::Quartz => VeinBias::Planar(rng.gen_range(0..3)),
        _ => VeinBias::Compact,
    }
}

/// BFS through solid rock from a solid starting position to find nearby air voxels.
///
/// Returns `(air_position, bfs_distance)` pairs sorted by distance (closest first).
/// Used by Phase 3 to find air pathways from submerged heat sources (lava in solid bowls).
pub fn find_air_from_solid(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    start: (i32, i32, i32),
    max_radius: u32,
    chunk_size: usize,
) -> Vec<((i32, i32, i32), u32)> {
    let mut queue: VecDeque<((i32, i32, i32), u32)> = VecDeque::new();
    let mut visited: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut air_results: Vec<((i32, i32, i32), u32)> = Vec::new();

    visited.insert(start);
    queue.push_back((start, 0));

    while let Some(((sx, sy, sz), dist)) = queue.pop_front() {
        if dist >= max_radius {
            continue;
        }
        for &(dx, dy, dz) in &FACE_OFFSETS {
            let n = (sx + dx, sy + dy, sz + dz);
            if !visited.insert(n) {
                continue;
            }
            if let Some(mat) = sample_material(density_fields, n.0, n.1, n.2, chunk_size) {
                let next_dist = dist + 1;
                if !mat.is_solid() {
                    // Found air — record it
                    air_results.push((n, next_dist));
                } else {
                    // Solid — continue BFS through rock
                    queue.push_back((n, next_dist));
                }
            }
        }
    }

    air_results.sort_by_key(|&(_, d)| d);
    air_results
}
