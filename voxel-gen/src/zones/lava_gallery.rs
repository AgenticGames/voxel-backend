//! Lava Tube Gallery zone: smooth volcanic tunnels with multi-material palette.
//!
//! Materials: Basalt (primary walls), Obsidian (glossy patches where lava cooled
//! fast against existing rock, near openings), Pumice (ceiling pockets from
//! trapped gas), Scoria (rough broken lava on floors).

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
    let eb = effective_bounds;

    // Step 1: Replace all host rock with volcanic materials based on position
    // Walls → Basalt, Floor → Scoria, Ceiling pockets → Pumice, Smooth patches → Obsidian
    {
        let mut rng = ChaCha8Rng::seed_from_u64(zone_seed);
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
                            if !density.samples[idx].material.is_host_rock()
                                && !density.samples[idx].material.is_ore() { continue; }

                            let above_idx = z * size * size + (y + 1) * size + x;
                            let below_idx = z * size * size + (y - 1) * size + x;

                            let is_floor = y + 1 < size && density.samples[above_idx].density <= 0.0;
                            let is_ceiling = y > 0 && density.samples[below_idx].density <= 0.0;

                            // Check any neighbor for air (surface voxel)
                            let has_air = is_floor || is_ceiling || [
                                (x + 1, y, z), (x.wrapping_sub(1), y, z),
                                (x, y, z + 1), (x, y, z.wrapping_sub(1)),
                            ].iter().any(|&(nx, ny, nz)| {
                                nx < size && nz < size && {
                                    let ni = nz * size * size + ny * size + nx;
                                    density.samples[ni].density <= 0.0
                                }
                            });

                            if is_floor {
                                // Floor: Scoria (rough broken lava)
                                density.samples[idx].material = Material::Scoria;
                            } else if is_ceiling {
                                // Ceiling: mostly Basalt, with 20% Pumice pockets
                                let roll: f32 = rng.gen();
                                density.samples[idx].material = if roll < 0.20 {
                                    Material::Pumice
                                } else {
                                    Material::Basalt
                                };
                            } else if has_air {
                                // Wall surface: mostly Basalt, 15% Obsidian (glossy patches)
                                let roll: f32 = rng.gen();
                                density.samples[idx].material = if roll < 0.15 {
                                    Material::Obsidian
                                } else {
                                    Material::Basalt
                                };
                            } else {
                                // Interior solid: Basalt
                                density.samples[idx].material = Material::Basalt;
                            }
                        }
                    }
                }
            }
        }
    }

    // Step 2: Lavacicles (short basalt cones hanging from ceiling)
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
        // Alternate between Basalt and Obsidian for lavacicles
        let mat = if (anchor.x * 7.0 + anchor.z * 13.0).sin() > 0.3 {
            Material::Obsidian
        } else {
            Material::Basalt
        };
        shapes::write_cone(
            density_fields, anchor, length, radius,
            -1.0, mat, 2.0, effective_bounds,
        );
    }

    (Vec::new(), Vec::new())
}
