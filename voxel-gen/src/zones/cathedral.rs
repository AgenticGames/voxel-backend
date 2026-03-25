//! Cathedral Cavern zone: mega-dome with multi-material decoration.
//!
//! Materials: Host rock (walls, boulders), Flowstone (amber calcite drapes on
//! walls with downward slope), Moonmilk (soft white patches on ceiling, deposited
//! by bacteria). Stalagmites/stalactites inherit host rock.

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
    let zone_seed = global_seed.wrapping_add(0xCA7E_D0A1).wrapping_add(min_key.0 as u64 * 31);
    let mut rng = ChaCha8Rng::seed_from_u64(zone_seed);
    let eb = effective_bounds;

    let center = volume.world_center;
    let extent = volume.world_bbox_max - volume.world_bbox_min;

    // Step 1: Carve dome (oblate ellipsoid)
    let dome_scale = config.cathedral_dome_scale;
    let semi_axes = Vec3::new(
        extent.x * 0.5 * dome_scale,
        extent.y * 0.5 * dome_scale * 0.7,
        extent.z * 0.5 * dome_scale,
    );
    shapes::carve_ellipsoid(density_fields, center, semi_axes, 2.0, effective_bounds);

    // Step 2: Moonmilk on ceiling surfaces + Flowstone on wall surfaces
    {
        let mut rng_mat = ChaCha8Rng::seed_from_u64(zone_seed.wrapping_add(10));
        for &key in &volume.chunk_keys {
            if let Some(density) = density_fields.get_mut(&key) {
                let size = density.size;
                let vs = eb / (size - 1) as f32;
                let origin = Vec3::new(key.0 as f32 * eb, key.1 as f32 * eb, key.2 as f32 * eb);

                for z in 1..size - 1 {
                    for y in 1..size - 1 {
                        for x in 1..size - 1 {
                            let idx = z * size * size + y * size + x;
                            if density.samples[idx].density <= 0.0 { continue; }

                            let wp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                            // Only inside dome
                            let offset = wp - center;
                            let nd = (offset.x / semi_axes.x).powi(2)
                                + (offset.y / semi_axes.y).powi(2)
                                + (offset.z / semi_axes.z).powi(2);
                            if nd > 1.5 { continue; } // just outside dome surface

                            let below_idx = z * size * size + (y - 1) * size + x;
                            let above_idx = z * size * size + (y + 1) * size + x;

                            let is_ceiling = y > 0 && density.samples[below_idx].density <= 0.0;
                            let is_floor = y + 1 < size && density.samples[above_idx].density <= 0.0;

                            let is_wall = !is_ceiling && !is_floor && [
                                (x + 1, y, z), (x.wrapping_sub(1), y, z),
                                (x, y, z + 1), (x, y, z.wrapping_sub(1)),
                            ].iter().any(|&(nx, ny, nz)| {
                                nx < size && nz < size && {
                                    let ni = nz * size * size + ny * size + nx;
                                    density.samples[ni].density <= 0.0
                                }
                            });

                            if is_ceiling {
                                // Moonmilk: soft white calcium carbonate on ceiling
                                let roll: f32 = rng_mat.gen();
                                if roll < 0.35 {
                                    density.samples[idx].material = Material::Moonmilk;
                                }
                            } else if is_wall {
                                // Flowstone: amber calcite drapes on walls
                                let roll: f32 = rng_mat.gen();
                                if roll < config.cathedral_flowstone_coverage {
                                    density.samples[idx].material = Material::Flowstone;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Step 3: Breakdown boulders on the floor
    let boulder_count = rng.gen_range(config.cathedral_boulder_count_min..=config.cathedral_boulder_count_max);
    let floor_y = center.y - semi_axes.y * 0.8;
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

    // Step 4: Mega-stalagmites on the floor (Flowstone material for dramatic amber pillars)
    let stalag_count = (semi_axes.x * semi_axes.z * config.cathedral_mega_stalagmite_chance * 0.02) as u32;
    for _ in 0..stalag_count.min(20) {
        let sx = center.x + rng.gen_range(-semi_axes.x * 0.7..semi_axes.x * 0.7);
        let sz = center.z + rng.gen_range(-semi_axes.z * 0.7..semi_axes.z * 0.7);
        let height = rng.gen_range(6.0..15.0);
        let radius = rng.gen_range(1.5..3.0);
        // Alternate between host rock and flowstone for visual variety
        let mat = if rng.gen::<f32>() < 0.4 { Material::Flowstone } else { volume.dominant_material };
        shapes::write_cone(
            density_fields,
            Vec3::new(sx, floor_y, sz),
            height, radius,
            1.0, mat, 2.0,
            effective_bounds,
        );
    }

    (Vec::new(), Vec::new())
}
