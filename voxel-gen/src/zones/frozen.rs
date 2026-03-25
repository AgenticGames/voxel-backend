//! Frozen Grotto zone: multi-ice-type chamber with frozen waterfalls and frost.
//!
//! Materials: Ice (clear blue, stalactites/stalagmites), Hoarfrost (white crystalline
//! surface frost on walls/ceiling), BlackIce (dense dark patches on floor),
//! Permafrost (frozen earth transition at zone edges).

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

    let eb = effective_bounds;
    let center = volume.world_center;
    let extent = volume.world_bbox_max - volume.world_bbox_min;

    // Step 1: Permafrost transition layer at zone edges (outer 30% of zone)
    // Replace host rock near zone boundary with frozen earth
    let edge_min = volume.world_bbox_min;
    let edge_max = volume.world_bbox_max;
    let inner_shrink = 0.3; // inner zone starts at 30% inward
    let inner_min = edge_min + extent * inner_shrink;
    let inner_max = edge_max - extent * inner_shrink;

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
                        if wp.x < edge_min.x || wp.x > edge_max.x
                            || wp.y < edge_min.y || wp.y > edge_max.y
                            || wp.z < edge_min.z || wp.z > edge_max.z { continue; }

                        let in_inner = wp.x >= inner_min.x && wp.x <= inner_max.x
                            && wp.y >= inner_min.y && wp.y <= inner_max.y
                            && wp.z >= inner_min.z && wp.z <= inner_max.z;

                        // Check if adjacent to air (surface voxel)
                        let has_air = [
                            (x + 1, y, z), (x.wrapping_sub(1), y, z),
                            (x, y + 1, z), (x, y.wrapping_sub(1), z),
                            (x, y, z + 1), (x, y, z.wrapping_sub(1)),
                        ].iter().any(|&(nx, ny, nz)| {
                            nx < size && ny < size && nz < size && {
                                let ni = nz * size * size + ny * size + nx;
                                density.samples[ni].density <= 0.0
                            }
                        });

                        if has_air {
                            if !in_inner {
                                // Edge zone: Permafrost (frozen earth transition)
                                density.samples[idx].material = Material::Permafrost;
                            } else {
                                // Inner zone surface: Hoarfrost (white crystalline frost)
                                density.samples[idx].material = Material::Hoarfrost;
                            }
                        } else if !in_inner && density.samples[idx].material.is_host_rock() {
                            // Subsurface edge: Permafrost
                            density.samples[idx].material = Material::Permafrost;
                        }
                    }
                }
            }
        }
    }

    // Step 2: BlackIce patches on floor (dense dark ice, smooth walkable)
    // Collect floor positions, then apply
    let mut floor_positions: Vec<Vec3> = Vec::new();
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
                            let above_idx = z * size * size + (y + 1) * size + x;
                            if y + 1 < size && density.samples[above_idx].density <= 0.0 {
                                let roll: f32 = rng.gen();
                                if roll < 0.25 { // 25% of floor gets BlackIce
                                    let wp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                    floor_positions.push(wp);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Apply BlackIce to collected floor positions (need mutable access)
    for &key in &volume.chunk_keys {
        if let Some(density) = density_fields.get_mut(&key) {
            let size = density.size;
            let vs = eb / (size - 1) as f32;
            let origin = Vec3::new(key.0 as f32 * eb, key.1 as f32 * eb, key.2 as f32 * eb);
            for z in 1..size - 1 {
                for y in 1..size - 1 {
                    for x in 1..size - 1 {
                        let idx = z * size * size + y * size + x;
                        if density.samples[idx].density > 0.0 {
                            let above_idx = z * size * size + (y + 1) * size + x;
                            if y + 1 < size && density.samples[above_idx].density <= 0.0 {
                                let wp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                // Check if this position was selected for BlackIce
                                if floor_positions.iter().any(|fp| (fp.x - wp.x).abs() < vs * 0.5 && (fp.y - wp.y).abs() < vs * 0.5 && (fp.z - wp.z).abs() < vs * 0.5) {
                                    density.samples[idx].material = Material::BlackIce;
                                    // Also set 1 voxel below to BlackIce for depth
                                    if y > 1 {
                                        let below_idx = z * size * size + (y - 1) * size + x;
                                        if density.samples[below_idx].density > 0.0 {
                                            density.samples[below_idx].material = Material::BlackIce;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Step 3: Ice stalactites and stalagmites
    let mut cone_params: Vec<(Vec3, f32, f32, f32)> = Vec::new();
    {
        let mut rng2 = ChaCha8Rng::seed_from_u64(zone_seed.wrapping_add(100));
        for &key in &volume.chunk_keys {
            if let Some(density) = density_fields.get(&key) {
                let size = density.size;
                let vs = eb / (size - 1) as f32;
                let origin = Vec3::new(key.0 as f32 * eb, key.1 as f32 * eb, key.2 as f32 * eb);

                for z in 1..size - 1 {
                    for y in 1..size - 1 {
                        for x in 1..size - 1 {
                            let idx = z * size * size + y * size + x;
                            if density.samples[idx].density <= 0.0 { continue; }

                            let below_idx = z * size * size + (y - 1) * size + x;
                            let above_idx = z * size * size + (y + 1) * size + x;

                            // Ceiling → ice stalactite
                            if y > 0 && density.samples[below_idx].density <= 0.0 {
                                let roll: f32 = rng2.gen();
                                if roll < config.frozen_ice_stalactite_chance {
                                    let pos = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                    let length = rng2.gen_range(1.0..4.0);
                                    let radius = rng2.gen_range(0.3..1.0);
                                    cone_params.push((pos, length, radius, -1.0));
                                }
                            }

                            // Floor → ice stalagmite
                            if y + 1 < size && density.samples[above_idx].density <= 0.0 {
                                let roll: f32 = rng2.gen();
                                if roll < config.frozen_ice_stalactite_chance * 0.7 {
                                    let pos = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                    let length = rng2.gen_range(1.0..3.0);
                                    let radius = rng2.gen_range(0.3..0.8);
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
            Material::Ice, // clear blue ice for stalactites
            2.0,
            effective_bounds,
        );
    }

    (Vec::new(), Vec::new())
}
