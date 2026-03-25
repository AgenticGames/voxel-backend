//! Geothermal Terraces zone: stepped travertine pools with overflow channels.

use std::collections::HashMap;

use glam::Vec3;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::Rng;

use voxel_core::material::Material;

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
    let zone_seed = global_seed.wrapping_add(0x7E22ACE5).wrapping_add(min_key.1 as u64 * 53);
    let mut rng = ChaCha8Rng::seed_from_u64(zone_seed);

    let center = volume.world_center;
    let extent = volume.world_bbox_max - volume.world_bbox_min;

    // Determine number of tiers
    let vertical_extent = extent.y;
    let step_height = config.terrace_step_height;
    let num_tiers = ((vertical_extent / step_height) as u32)
        .max(config.terrace_tiers_min)
        .min(config.terrace_tiers_max);

    let chunk_size = density_fields.values().next().map(|d| d.size - 1).unwrap_or(16);
    let mut fluid_seeds = Vec::new();

    // Step 1: Stain zone walls with Travertine
    shapes::apply_surface_material(
        density_fields,
        volume.world_bbox_min,
        volume.world_bbox_max,
        Material::Travertine,
        effective_bounds,
    );

    // Step 2: Carve tiered shelves with basins (top → bottom)
    for tier in 0..num_tiers {
        let tier_y = volume.world_bbox_max.y - (tier as f32 + 0.5) * step_height;
        let tier_width = extent.x * rng.gen_range(0.3..0.5);
        let tier_depth = extent.z * rng.gen_range(0.3..0.5);

        // Carve flat shelf
        let shelf_semi = Vec3::new(tier_width, 1.0, tier_depth);
        let shelf_center = Vec3::new(
            center.x + rng.gen_range(-2.0..2.0),
            tier_y,
            center.z + rng.gen_range(-2.0..2.0),
        );
        shapes::carve_ellipsoid(density_fields, shelf_center, shelf_semi, 3.0, effective_bounds);

        // Carve basin below shelf
        let basin_semi = Vec3::new(
            tier_width * 0.8,
            config.terrace_basin_depth as f32,
            tier_depth * 0.8,
        );
        let basin_center = Vec3::new(shelf_center.x, tier_y - 1.0, shelf_center.z);
        shapes::carve_ellipsoid(density_fields, basin_center, basin_semi, 2.0, effective_bounds);

        // Place water seeds in basin
        let spacing = 2.0;
        let mut fx = shelf_center.x - tier_width * 0.6;
        while fx <= shelf_center.x + tier_width * 0.6 {
            let mut fz = shelf_center.z - tier_depth * 0.6;
            while fz <= shelf_center.z + tier_depth * 0.6 {
                fluid_seeds.push(world_to_fluid_seed(
                    fx, tier_y - config.terrace_basin_depth as f32, fz,
                    effective_bounds, chunk_size, false,
                ));
                fz += spacing;
            }
            fx += spacing;
        }
    }

    (Vec::new(), fluid_seeds)
}
