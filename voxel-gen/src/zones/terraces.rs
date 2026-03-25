//! Geothermal Terraces zone: stepped pools with multi-mineral palette.
//!
//! Materials: Travertine (rim walls, terrace structure), Sinter (basin floors,
//! siliceous thermal deposits), Sulfur (bright yellow patches near vents on walls).

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
    let zone_seed = global_seed.wrapping_add(0x7E22_ACE5).wrapping_add(min_key.1 as u64 * 53);
    let mut rng = ChaCha8Rng::seed_from_u64(zone_seed);

    let center = volume.world_center;
    let extent = volume.world_bbox_max - volume.world_bbox_min;
    let chunk_size = density_fields.values().next().map(|d| d.size - 1).unwrap_or(16);
    let eb = effective_bounds;

    // Determine number of tiers
    let vertical_extent = extent.y;
    let step_height = config.terrace_step_height;
    let num_tiers = ((vertical_extent / step_height) as u32)
        .max(config.terrace_tiers_min)
        .min(config.terrace_tiers_max);

    let mut fluid_seeds = Vec::new();

    // Step 1: Zone-wide mineral staining
    // Wall surfaces: mostly Travertine, with Sulfur patches and Sinter bands
    {
        let mut rng_stain = ChaCha8Rng::seed_from_u64(zone_seed.wrapping_add(50));
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
                            if wp.x < volume.world_bbox_min.x || wp.x > volume.world_bbox_max.x
                                || wp.y < volume.world_bbox_min.y || wp.y > volume.world_bbox_max.y
                                || wp.z < volume.world_bbox_min.z || wp.z > volume.world_bbox_max.z { continue; }

                            // Check for air neighbor (surface)
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
                                let roll: f32 = rng_stain.gen();
                                // Depth-based staining: Sulfur more common deep, Sinter mid, Travertine top
                                let depth_factor = (volume.world_bbox_max.y - wp.y) / extent.y;
                                if roll < 0.08 + depth_factor * 0.12 {
                                    // Sulfur increases with depth (vent proximity)
                                    density.samples[idx].material = Material::Sulfur;
                                } else if roll < 0.25 {
                                    // Sinter bands (siliceous deposit)
                                    density.samples[idx].material = Material::Sinter;
                                } else {
                                    // Primary surface: Travertine
                                    density.samples[idx].material = Material::Travertine;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Step 2: Carve tiered shelves with basins
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

        // Write Sinter as basin floor material (siliceous thermal deposit)
        let sinter_center = Vec3::new(shelf_center.x, tier_y - config.terrace_basin_depth as f32, shelf_center.z);
        shapes::write_solid_sphere(
            density_fields, sinter_center,
            tier_width * 0.7, Material::Sinter, effective_bounds,
        );

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
