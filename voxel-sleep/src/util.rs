//! Shared utility functions for sleep phase modules.

use std::collections::HashMap;
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
