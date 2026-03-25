//! Subterranean Lake zone: massive still lake with calcite beaches and rocky islands.

use std::collections::HashMap;

use glam::Vec3;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::config::ZoneConfig;
use crate::density::DensityField;
use crate::pools::FluidSeed;

use super::detect::CavernVolume;
use super::shapes;
use super::{ZoneAnchor, world_to_fluid_seed};

pub fn generate(
    volume: &CavernVolume,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    config: &ZoneConfig,
    global_seed: u64,
    effective_bounds: f32,
) -> (Vec<ZoneAnchor>, Vec<FluidSeed>) {
    let min_key = volume.chunk_keys.iter().copied().min().unwrap_or((0, 0, 0));
    let zone_seed = global_seed.wrapping_add(0x1A4E_0000u64).wrapping_add(min_key.0 as u64 * 37);
    let _rng = ChaCha8Rng::seed_from_u64(zone_seed);

    let center = volume.world_center;
    let extent = volume.world_bbox_max - volume.world_bbox_min;
    let floor_y = volume.world_bbox_min.y + extent.y * 0.3;

    // Derive chunk_size from first available density field
    let chunk_size = density_fields.values().next().map(|d| d.size - 1).unwrap_or(16);

    // Step 1: Carve a basin below water level
    let basin_semi = Vec3::new(
        extent.x * 0.4,
        config.lake_depth as f32 + 2.0,
        extent.z * 0.4,
    );
    let basin_center = Vec3::new(center.x, floor_y, center.z);
    shapes::carve_ellipsoid(density_fields, basin_center, basin_semi, 1.5, effective_bounds);

    // Step 2: Place water fluid seeds across the lake floor
    let mut fluid_seeds = Vec::new();
    let spacing = 2.0;
    let x_start = center.x - basin_semi.x * 0.8;
    let x_end = center.x + basin_semi.x * 0.8;
    let z_start = center.z - basin_semi.z * 0.8;
    let z_end = center.z + basin_semi.z * 0.8;

    let mut fx = x_start;
    while fx <= x_end {
        let mut fz = z_start;
        while fz <= z_end {
            fluid_seeds.push(world_to_fluid_seed(fx, floor_y - 1.0, fz, effective_bounds, chunk_size, false));
            fz += spacing;
        }
        fx += spacing;
    }

    (Vec::new(), fluid_seeds)
}
