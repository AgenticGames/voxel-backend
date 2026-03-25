//! Frozen Grotto zone: ice floor, frozen waterfalls, ice stalactites, surface ice layer.

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
use super::ZoneAnchor;

pub fn generate(
    volume: &CavernVolume,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    config: &ZoneConfig,
    global_seed: u64,
    effective_bounds: f32,
) -> (Vec<ZoneAnchor>, Vec<FluidSeed>) {
    let min_key = volume.chunk_keys.iter().copied().min().unwrap_or((0, 0, 0));
    let zone_seed = global_seed.wrapping_add(0xF002_0000u64).wrapping_add(min_key.2 as u64 * 59);
    let mut rng = ChaCha8Rng::seed_from_u64(zone_seed);

    // Step 1: Apply surface ice layer to all solid surfaces in zone
    shapes::apply_surface_material(
        density_fields,
        volume.world_bbox_min,
        volume.world_bbox_max,
        Material::Ice,
        effective_bounds,
    );

    // Step 2: Ice stalactites and stalagmites
    let eb = effective_bounds;
    let mut cone_params: Vec<(Vec3, f32, f32, f32)> = Vec::new(); // (anchor, length, radius, direction)

    {
        for &key in &volume.chunk_keys {
            if let Some(density) = density_fields.get(&key) {
                let size = density.size;
                let vs = eb / (size - 1) as f32;
                let origin = Vec3::new(key.0 as f32 * eb, key.1 as f32 * eb, key.2 as f32 * eb);

                for z in 1..size - 1 {
                    for y in 1..size - 1 {
                        for x in 1..size - 1 {
                            let idx = z * size * size + y * size + x;
                            if density.samples[idx].density <= 0.0 {
                                continue;
                            }

                            let below_idx = z * size * size + (y - 1) * size + x;
                            let above_idx = z * size * size + (y + 1) * size + x;

                            // Ceiling → ice stalactite (hanging down)
                            if y > 0 && density.samples[below_idx].density <= 0.0 {
                                let roll: f32 = rng.gen();
                                if roll < config.frozen_ice_stalactite_chance {
                                    let pos = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                    let length = rng.gen_range(1.0..4.0);
                                    let radius = rng.gen_range(0.3..1.0);
                                    cone_params.push((pos, length, radius, -1.0));
                                }
                            }

                            // Floor → ice stalagmite (growing up)
                            if y + 1 < size && density.samples[above_idx].density <= 0.0 {
                                let roll: f32 = rng.gen();
                                if roll < config.frozen_ice_stalactite_chance * 0.7 {
                                    let pos = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                    let length = rng.gen_range(1.0..3.0);
                                    let radius = rng.gen_range(0.3..0.8);
                                    cone_params.push((pos, length, radius, 1.0));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (anchor, length, radius, direction) in cone_params {
        shapes::write_cone(
            density_fields,
            anchor,
            length,
            radius,
            direction,
            Material::Ice,
            2.0,
            effective_bounds,
        );
    }

    (Vec::new(), Vec::new())
}
