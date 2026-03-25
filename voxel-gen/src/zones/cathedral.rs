//! Cathedral Cavern zone: mega-dome with stalagmite forest, breakdown boulders, flowstone walls.

use std::collections::HashMap;

use glam::Vec3;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::Rng;

use crate::config::ZoneConfig;
use crate::density::DensityField;
use crate::pools::FluidSeed;

use super::detect::CavernVolume;
use super::shapes;
use super::ZoneAnchor;

pub fn generate(
    volume: &CavernVolume,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    config: &ZoneConfig,
    global_seed: u64,
    effective_bounds: f32,
) -> (Vec<ZoneAnchor>, Vec<FluidSeed>) {
    let min_key = volume.chunk_keys.iter().copied().min().unwrap_or((0, 0, 0));
    let zone_seed = global_seed.wrapping_add(0xCA7ED0A1).wrapping_add(min_key.0 as u64 * 31);
    let mut rng = ChaCha8Rng::seed_from_u64(zone_seed);

    let center = volume.world_center;
    let extent = volume.world_bbox_max - volume.world_bbox_min;

    // Step 1: Carve dome (oblate ellipsoid)
    let dome_scale = config.cathedral_dome_scale;
    let semi_axes = Vec3::new(
        extent.x * 0.5 * dome_scale,
        extent.y * 0.5 * dome_scale * 0.7, // oblate: shorter vertically
        extent.z * 0.5 * dome_scale,
    );
    shapes::carve_ellipsoid(density_fields, center, semi_axes, 2.0, effective_bounds);

    // Step 2: Breakdown boulders on the floor
    let boulder_count = rng.gen_range(config.cathedral_boulder_count_min..=config.cathedral_boulder_count_max);
    let floor_y = center.y - semi_axes.y * 0.8; // approximate floor level
    for _ in 0..boulder_count {
        let bx = center.x + rng.gen_range(-semi_axes.x * 0.6..semi_axes.x * 0.6);
        let bz = center.z + rng.gen_range(-semi_axes.z * 0.6..semi_axes.z * 0.6);
        let radius = rng.gen_range(1.5..3.0);
        shapes::write_solid_sphere(
            density_fields,
            Vec3::new(bx, floor_y + radius, bz),
            radius,
            volume.dominant_material,
            effective_bounds,
        );
    }

    // Step 3: Mega-stalagmites on the floor
    let stalag_count = (semi_axes.x * semi_axes.z * config.cathedral_mega_stalagmite_chance * 0.02) as u32;
    for _ in 0..stalag_count.min(20) {
        let sx = center.x + rng.gen_range(-semi_axes.x * 0.7..semi_axes.x * 0.7);
        let sz = center.z + rng.gen_range(-semi_axes.z * 0.7..semi_axes.z * 0.7);
        let height = rng.gen_range(6.0..15.0);
        let radius = rng.gen_range(1.5..3.0);
        shapes::write_cone(
            density_fields,
            Vec3::new(sx, floor_y, sz),
            height,
            radius,
            1.0, // growing up
            volume.dominant_material,
            2.0,
            effective_bounds,
        );
    }

    (Vec::new(), Vec::new())
}
