//! Underground River Canyon zone: deep vertical slot with bridges and river.

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
use super::{ZoneAnchor, world_to_fluid_seed};

pub fn generate(
    volume: &CavernVolume,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    config: &ZoneConfig,
    global_seed: u64,
    effective_bounds: f32,
) -> (Vec<ZoneAnchor>, Vec<FluidSeed>) {
    let min_key = volume.chunk_keys.iter().copied().min().unwrap_or((0, 0, 0));
    let zone_seed = global_seed.wrapping_add(0xCA01_0000u64).wrapping_add(min_key.2 as u64 * 41);
    let mut rng = ChaCha8Rng::seed_from_u64(zone_seed);

    let extent = volume.world_bbox_max - volume.world_bbox_min;
    let center = volume.world_center;

    // Determine principal (longest) axis
    let (axis, length) = if extent.x >= extent.y && extent.x >= extent.z {
        (0, extent.x)
    } else if extent.z >= extent.x && extent.z >= extent.y {
        (2, extent.z)
    } else {
        (1, extent.y)
    };

    let width = rng.gen_range(config.canyon_width_min..=config.canyon_width_max);
    let height = rng.gen_range(config.canyon_height_min..=config.canyon_height_max);
    let steps = (length / 1.0).ceil() as usize;

    let chunk_size = density_fields.values().next().map(|d| d.size - 1).unwrap_or(16);
    let mut fluid_seeds = Vec::new();

    // Carve canyon slot along principal axis
    for step in 0..steps {
        let t = step as f32 / steps as f32;
        let noise_offset = (t * 7.0).sin() * 1.5; // gentle winding

        let pos = match axis {
            0 => Vec3::new(
                volume.world_bbox_min.x + t * extent.x,
                center.y,
                center.z + noise_offset,
            ),
            2 => Vec3::new(
                center.x + noise_offset,
                center.y,
                volume.world_bbox_min.z + t * extent.z,
            ),
            _ => Vec3::new(
                center.x + noise_offset,
                volume.world_bbox_min.y + t * extent.y,
                center.z,
            ),
        };

        // Carve a tall narrow ellipse
        let slot_semi = match axis {
            0 => Vec3::new(0.8, height * 0.5, width * 0.5),
            2 => Vec3::new(width * 0.5, height * 0.5, 0.8),
            _ => Vec3::new(width * 0.5, 0.8, height * 0.5),
        };
        shapes::carve_ellipsoid(density_fields, pos, slot_semi, 2.0, effective_bounds);

        // Place river water at canyon floor
        if step % 2 == 0 {
            fluid_seeds.push(world_to_fluid_seed(
                pos.x, pos.y - height * 0.4, pos.z,
                effective_bounds, chunk_size, false,
            ));
        }
    }

    // Place rock bridges
    let bridge_count = (length / 15.0).ceil() as u32;
    for _ in 0..bridge_count.min(4) {
        let roll: f32 = rng.gen();
        if roll < config.canyon_bridge_chance {
            let bt = rng.gen_range(0.2..0.8);
            let bridge_pos = match axis {
                0 => Vec3::new(
                    volume.world_bbox_min.x + bt * extent.x,
                    center.y + rng.gen_range(-height * 0.1..height * 0.2),
                    center.z,
                ),
                _ => Vec3::new(
                    center.x,
                    center.y + rng.gen_range(-height * 0.1..height * 0.2),
                    volume.world_bbox_min.z + bt * extent.z,
                ),
            };
            shapes::write_solid_sphere(
                density_fields,
                bridge_pos,
                width * 0.6,
                volume.dominant_material,
                effective_bounds,
            );
        }
    }

    (Vec::new(), fluid_seeds)
}
