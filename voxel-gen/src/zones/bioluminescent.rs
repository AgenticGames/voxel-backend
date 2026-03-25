//! Bioluminescent Grotto zone: anchor points for UE emissive rendering.

use std::collections::HashMap;

use glam::Vec3;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::Rng;

use crate::config::ZoneConfig;
use crate::density::DensityField;
use crate::pools::FluidSeed;

use super::detect::CavernVolume;
use super::ZoneAnchor;

pub fn generate(
    volume: &CavernVolume,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    config: &ZoneConfig,
    global_seed: u64,
    effective_bounds: f32,
) -> (Vec<ZoneAnchor>, Vec<FluidSeed>) {
    let min_key = volume.chunk_keys.iter().copied().min().unwrap_or((0, 0, 0));
    let zone_seed = global_seed.wrapping_add(0xB10_6100).wrapping_add(min_key.1 as u64 * 47);
    let mut rng = ChaCha8Rng::seed_from_u64(zone_seed);

    let eb = effective_bounds;
    let mut anchors = Vec::new();
    let max_anchors = config.bio_max_anchors as usize;

    // Scan ceiling and wall surfaces, sub-sample for anchor placement
    for &key in &volume.chunk_keys {
        if anchors.len() >= max_anchors {
            break;
        }
        if let Some(density) = density_fields.get(&key) {
            let size = density.size;
            let vs = eb / (size - 1) as f32;
            let origin = Vec3::new(key.0 as f32 * eb, key.1 as f32 * eb, key.2 as f32 * eb);

            for z in 1..size - 1 {
                for y in 1..size - 1 {
                    for x in 1..size - 1 {
                        if anchors.len() >= max_anchors {
                            break;
                        }
                        let idx = z * size * size + y * size + x;
                        if density.samples[idx].density <= 0.0 {
                            continue;
                        }

                        // Check for ceiling (solid with air below)
                        let below_idx = z * size * size + (y - 1) * size + x;
                        if density.samples[below_idx].density <= 0.0 {
                            let roll: f32 = rng.gen();
                            if roll < config.bio_anchor_density {
                                let pos = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                anchors.push(ZoneAnchor {
                                    px: pos.x, py: pos.y, pz: pos.z,
                                    nx: 0.0, ny: -1.0, nz: 0.0,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    (anchors, Vec::new())
}
