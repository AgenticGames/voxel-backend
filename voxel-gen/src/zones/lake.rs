//! Subterranean Lake zone: massive still lake with Tufa shores.
//!
//! Materials: Host rock (walls, islands), Tufa (porous pale limestone deposited
//! at the waterline from CO2 outgassing — forms the shore ring).

use std::collections::HashMap;

use glam::Vec3;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

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
    let zone_seed = global_seed.wrapping_add(0x1A4E_0000u64).wrapping_add(min_key.0 as u64 * 37);
    let _rng = ChaCha8Rng::seed_from_u64(zone_seed);
    let eb = effective_bounds;

    let center = volume.world_center;
    let extent = volume.world_bbox_max - volume.world_bbox_min;
    let floor_y = volume.world_bbox_min.y + extent.y * 0.3;
    let chunk_size = density_fields.values().next().map(|d| d.size - 1).unwrap_or(16);

    // Step 1: Carve a basin below water level
    let basin_semi = Vec3::new(
        extent.x * 0.4,
        config.lake_depth as f32 + 2.0,
        extent.z * 0.4,
    );
    let basin_center = Vec3::new(center.x, floor_y, center.z);
    shapes::carve_ellipsoid(density_fields, basin_center, basin_semi, 1.5, effective_bounds);

    // Step 2: Tufa shore ring — replace solid material near the water surface
    // Tufa forms at the waterline where CO2 outgasses from the lake
    let water_y = floor_y + config.lake_depth as f32;
    let tufa_band = 2.0; // voxels above and below waterline
    {
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

                            // Check if near waterline Y band
                            let y_dist = (wp.y - water_y).abs();
                            if y_dist > tufa_band { continue; }

                            // Check if near basin edge (distance from center in XZ)
                            let dx = (wp.x - center.x) / basin_semi.x;
                            let dz = (wp.z - center.z) / basin_semi.z;
                            let xz_dist = (dx * dx + dz * dz).sqrt();
                            if xz_dist < 0.5 || xz_dist > 1.3 { continue; } // only the shore ring

                            // Check if surface voxel (adjacent to air)
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
                                density.samples[idx].material = Material::Tufa;
                            }
                        }
                    }
                }
            }
        }
    }

    // Step 3: Place water fluid seeds across the lake floor
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
