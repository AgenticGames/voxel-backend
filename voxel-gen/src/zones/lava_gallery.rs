//! Lava Tube Gallery zone: smooth basalt tunnels with lavacicles and benches.

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
    let zone_seed = global_seed.wrapping_add(0x1A0A_70BE).wrapping_add(min_key.0 as u64 * 43);
    let mut rng = ChaCha8Rng::seed_from_u64(zone_seed);

    // Step 1: Replace all host rock with Basalt
    shapes::replace_material_in_aabb(
        density_fields,
        volume.world_bbox_min,
        volume.world_bbox_max,
        Material::Basalt,
        effective_bounds,
    );

    // Step 2: Lavacicles (short downward cones on ceiling surfaces)
    // Collect positions from read-only pass, then write in batch
    let eb = effective_bounds;
    let mut lavacicle_params: Vec<(Vec3, f32, f32)> = Vec::new();
    {
        let mut rng2 = ChaCha8Rng::seed_from_u64(zone_seed.wrapping_add(1));
        for &key in &volume.chunk_keys {
            if let Some(density) = density_fields.get(&key) {
                let size = density.size;
                let vs = eb / (size - 1) as f32;
                let origin = Vec3::new(key.0 as f32 * eb, key.1 as f32 * eb, key.2 as f32 * eb);

                for z in 1..size - 1 {
                    for y in 1..size - 1 {
                        for x in 1..size - 1 {
                            let idx = z * size * size + y * size + x;
                            if density.samples[idx].density > 0.0 {
                                let below_idx = z * size * size + (y - 1) * size + x;
                                if density.samples[below_idx].density <= 0.0 {
                                    let roll: f32 = rng2.gen();
                                    if roll < config.lava_gallery_lavacicle_chance {
                                        let world_pos = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                        let length = rng2.gen_range(0.5..2.0);
                                        let radius = rng2.gen_range(0.3..0.8);
                                        lavacicle_params.push((world_pos, length, radius));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (anchor, length, radius) in lavacicle_params {
        shapes::write_cone(
            density_fields,
            anchor,
            length,
            radius,
            -1.0, // hanging down
            Material::Basalt,
            2.0,
            effective_bounds,
        );
    }

    (Vec::new(), Vec::new())
}
