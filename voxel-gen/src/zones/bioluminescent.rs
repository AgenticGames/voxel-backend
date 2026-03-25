//! Bioluminescent Grotto zone: noise+seed Mycelium spread, mushroom forests, water sources.
//!
//! Materials: Mycelium (clustered fungal networks on floor/walls), Glowstone (ceiling),
//! MushroomStalk (pale cream), MushroomGill (glowing underside), PurpleCap/TealCap/AmberCap
//! (color-variety caps clustered by type).

use std::collections::HashMap;

use glam::Vec3;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::Rng;

use voxel_core::material::Material;
use voxel_noise::NoiseSource;
use voxel_noise::simplex::Simplex3D;

use crate::config::ZoneConfig;
use crate::density::DensityField;
use crate::pools::FluidSeed;

use super::detect::CavernVolume;
use super::shapes;
use super::{ZoneAnchor, world_to_fluid_seed};

/// Cap color varieties — noise partitions the zone into regions, each gets one color.
const CAP_MATERIALS: [Material; 3] = [Material::PurpleCap, Material::TealCap, Material::AmberCap];

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
    let chunk_size = density_fields.values().next().map(|d| d.size - 1).unwrap_or(16);
    let max_anchors = config.bio_max_anchors as usize;

    let center = volume.world_center;
    let extent = volume.world_bbox_max - volume.world_bbox_min;

    // ── Step 1: Pick 3-6 seed points on the floor for Mycelium growth centers ──
    let num_seeds = rng.gen_range(3u32..=6);
    let mut mycelium_seeds: Vec<Vec3> = Vec::new();

    // Collect floor positions to pick seed points from
    let mut floor_positions: Vec<Vec3> = Vec::new();
    for &key in &volume.chunk_keys {
        if let Some(density) = density_fields.get(&key) {
            let size = density.size;
            let vs = eb / (size - 1) as f32;
            let origin = Vec3::new(key.0 as f32 * eb, key.1 as f32 * eb, key.2 as f32 * eb);
            for z in (1..size - 1).step_by(2) {
                for y in 0..size {
                    for x in (1..size - 1).step_by(2) {
                        let idx = z * size * size + y * size + x;
                        if density.samples[idx].density > 0.0 {
                            let above_idx = z * size * size + (y + 1) * size + x;
                            if y + 1 < size && density.samples[above_idx].density <= 0.0 {
                                let wp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                floor_positions.push(wp);
                            }
                        }
                    }
                }
            }
        }
    }

    // Pick seed points spread across the zone
    if !floor_positions.is_empty() {
        for _ in 0..num_seeds {
            let idx = rng.gen_range(0..floor_positions.len());
            mycelium_seeds.push(floor_positions[idx]);
        }
    }

    // ── Step 2: Noise + seed hybrid Mycelium spread ──
    // Simplex noise creates organic patch shapes, seeds intensify growth nearby
    let mycelium_noise = Simplex3D::new(zone_seed.wrapping_add(100));
    let mycelium_freq = 0.15; // patch scale
    let mycelium_threshold = 0.3; // noise must exceed this for any mycelium
    let seed_radius = extent.x.min(extent.z) * 0.35; // how far seeds influence

    for &key in &volume.chunk_keys {
        if let Some(density) = density_fields.get_mut(&key) {
            let size = density.size;
            let vs = eb / (size - 1) as f32;
            let origin = Vec3::new(key.0 as f32 * eb, key.1 as f32 * eb, key.2 as f32 * eb);

            for z in 0..size {
                for y in 0..size {
                    for x in 0..size {
                        let idx = z * size * size + y * size + x;
                        if density.samples[idx].density <= 0.0 { continue; }
                        // Preserve ores — only replace boring host rock
                        if density.samples[idx].material.is_ore() { continue; }

                        let wp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);

                        let above_idx = z * size * size + (y + 1) * size + x;
                        let below_idx = z * size * size + (y - 1) * size + x;
                        let is_floor = y + 1 < size && density.samples[above_idx].density <= 0.0;
                        let is_ceiling = y > 0 && density.samples[below_idx].density <= 0.0;
                        let is_wall = !is_floor && !is_ceiling && [
                            (x + 1, y, z), (x.wrapping_sub(1), y, z),
                            (x, y, z + 1), (x, y, z.wrapping_sub(1)),
                        ].iter().any(|&(nx, ny, nz)| {
                            nx < size && nz < size && {
                                let ni = nz * size * size + ny * size + nx;
                                density.samples[ni].density <= 0.0
                            }
                        });

                        if !is_floor && !is_ceiling && !is_wall { continue; }

                        // Sample noise at world position
                        let noise_val = mycelium_noise.sample(
                            wp.x as f64 * mycelium_freq,
                            wp.y as f64 * mycelium_freq * 0.5, // compress Y for horizontal spread
                            wp.z as f64 * mycelium_freq,
                        ) as f32 * 0.5 + 0.5; // normalize to 0..1

                        // Seed proximity boost: closer to a seed → higher probability
                        let seed_boost = mycelium_seeds.iter()
                            .map(|s| {
                                let d = (wp - *s).length();
                                (1.0 - (d / seed_radius).min(1.0)).powi(2) // quadratic falloff
                            })
                            .fold(0.0f32, f32::max);

                        let combined = noise_val + seed_boost * 0.5;

                        if is_floor && combined > mycelium_threshold {
                            density.samples[idx].material = Material::Mycelium;
                        } else if is_wall && combined > mycelium_threshold + 0.15 {
                            // Walls need higher threshold — mycelium climbs from floor
                            density.samples[idx].material = Material::Mycelium;
                        } else if is_ceiling && combined > mycelium_threshold + 0.35 {
                            // Ceiling: sparse Glowstone at high-combined areas
                            density.samples[idx].material = Material::Glowstone;
                        }
                    }
                }
            }
        }
    }

    // ── Step 3: Place mushroom clusters near seed points ──
    // Noise partitions the zone into 3 color regions
    let color_noise = Simplex3D::new(zone_seed.wrapping_add(200));

    let mut mushroom_params: Vec<(Vec3, f32, f32, f32, f32, Material)> = Vec::new();

    for seed_pos in &mycelium_seeds {
        // 2-4 mushrooms per seed cluster
        let cluster_count = rng.gen_range(2u32..=4);
        let cluster_radius = 3.0;

        // Determine cap color for this cluster based on noise at seed position
        let cn = color_noise.sample(
            seed_pos.x as f64 * 0.05,
            seed_pos.y as f64 * 0.05,
            seed_pos.z as f64 * 0.05,
        ) as f32;
        let cap_idx = if cn < -0.33 { 0 } else if cn < 0.33 { 1 } else { 2 };
        let cap_material = CAP_MATERIALS[cap_idx];

        for _ in 0..cluster_count {
            // Scatter within cluster radius
            let offset_x = rng.gen_range(-cluster_radius..cluster_radius);
            let offset_z = rng.gen_range(-cluster_radius..cluster_radius);
            let mush_base = Vec3::new(
                seed_pos.x + offset_x,
                seed_pos.y, // floor Y
                seed_pos.z + offset_z,
            );

            let stalk_height = rng.gen_range(4.0..10.0);
            let stalk_radius = rng.gen_range(0.6..1.2);
            let cap_radius = rng.gen_range(2.0..5.0);
            let cap_thickness = rng.gen_range(1.0..2.5);

            mushroom_params.push((mush_base, stalk_height, stalk_radius, cap_radius, cap_thickness, cap_material));
        }
    }

    // Write mushrooms
    for (base, stalk_h, stalk_r, cap_r, cap_t, cap_mat) in &mushroom_params {
        shapes::write_mushroom(
            density_fields,
            *base,
            *stalk_h,
            *stalk_r,
            *cap_r,
            *cap_t,
            Material::MushroomStalk,
            *cap_mat,
            Material::MushroomGill,
            effective_bounds,
        );
    }

    // ── Step 4: Organic noise-shaped water features ──
    // Irregular puddles and meandering streams between seed points.
    // Uses 2D noise to create organic blob boundaries — water placed where
    // noise exceeds threshold within range of seed points.
    let mut fluid_seeds = Vec::new();
    let water_noise = Simplex3D::new(zone_seed.wrapping_add(300));
    let water_freq = 0.25; // organic blob scale
    let water_threshold = 0.08; // lower = more water coverage

    // 80% of mycelium seeds get water, plus extra for large zones
    let mut water_seeds: Vec<(Vec3, f32)> = Vec::new(); // (center, max_radius)
    for seed_pos in &mycelium_seeds {
        let roll: f32 = rng.gen();
        if roll < 0.8 {
            let pool_max_r = rng.gen_range(4.0..10.0);
            water_seeds.push((*seed_pos, pool_max_r));
        }
    }
    // Add extra water seeds proportional to zone volume
    let extra_waters = (volume.total_air / 500).min(8);
    for _ in 0..extra_waters {
        if !floor_positions.is_empty() {
            let idx = rng.gen_range(0..floor_positions.len());
            let pool_max_r = rng.gen_range(3.0..8.0);
            water_seeds.push((floor_positions[idx], pool_max_r));
        }
    }

    // Connect ALL adjacent seed pairs with streams (not just 1-2)
    if water_seeds.len() >= 2 {
        let stream_count = water_seeds.len() - 1;
        for i in 0..stream_count {
            let a = water_seeds[i].0;
            let b = water_seeds[(i + 1) % water_seeds.len()].0;
            // Add intermediate points along the stream path
            let steps = ((a - b).length() / 3.0).ceil() as u32;
            for s in 1..steps {
                let t = s as f32 / steps as f32;
                let mid = a.lerp(b, t);
                // Narrow stream: small radius
                water_seeds.push((mid, rng.gen_range(1.0..3.0)));
            }
        }
    }

    // Scan floor area and place water where noise creates organic blobs near water seeds
    for &key in &volume.chunk_keys {
        if let Some(density) = density_fields.get(&key) {
            let size = density.size;
            let vs = eb / (size - 1) as f32;
            let origin = Vec3::new(key.0 as f32 * eb, key.1 as f32 * eb, key.2 as f32 * eb);

            for z in 0..size {
                for y in 0..size {
                    for x in 0..size {
                        let idx = z * size * size + y * size + x;
                        if density.samples[idx].density <= 0.0 { continue; }

                        // Only floor surfaces
                        let above_idx = z * size * size + (y + 1) * size + x;
                        if !(y + 1 < size && density.samples[above_idx].density <= 0.0) { continue; }

                        let wp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);

                        // Check proximity to any water seed
                        let mut best_influence = 0.0f32;
                        for &(center, max_r) in &water_seeds {
                            let dx = wp.x - center.x;
                            let dz = wp.z - center.z;
                            let dist = (dx * dx + dz * dz).sqrt();
                            if dist < max_r {
                                let proximity = 1.0 - dist / max_r;
                                best_influence = best_influence.max(proximity);
                            }
                        }

                        if best_influence <= 0.0 { continue; }

                        // Sample noise for organic boundary shape
                        let n = water_noise.sample(
                            wp.x as f64 * water_freq,
                            0.0, // flat 2D noise on XZ plane
                            wp.z as f64 * water_freq,
                        ) as f32 * 0.5 + 0.5;

                        // Water placed where noise + proximity exceeds threshold
                        // This creates organic blobs: noise provides the irregular boundary,
                        // proximity ensures they're near seed points
                        if n * best_influence > water_threshold {
                            fluid_seeds.push(world_to_fluid_seed(
                                wp.x, wp.y - 0.5, wp.z, effective_bounds, chunk_size, false,
                            ));
                        }
                    }
                }
            }
        }
    }

    // ── Step 5: Generate anchor positions for UE emissive rendering ──
    let mut anchors = Vec::new();
    for &key in &volume.chunk_keys {
        if anchors.len() >= max_anchors { break; }
        if let Some(density) = density_fields.get(&key) {
            let size = density.size;
            let vs = eb / (size - 1) as f32;
            let origin = Vec3::new(key.0 as f32 * eb, key.1 as f32 * eb, key.2 as f32 * eb);

            for z in 0..size {
                for y in 0..size {
                    for x in 0..size {
                        if anchors.len() >= max_anchors { break; }
                        let idx = z * size * size + y * size + x;
                        let mat = density.samples[idx].material;
                        // Anchors from Glowstone ceiling points AND MushroomGill
                        if mat == Material::Glowstone || mat == Material::MushroomGill {
                            let pos = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                            let ny = if mat == Material::Glowstone { -1.0 } else { -1.0 };
                            anchors.push(ZoneAnchor {
                                px: pos.x, py: pos.y, pz: pos.z,
                                nx: 0.0, ny, nz: 0.0,
                            });
                        }
                    }
                }
            }
        }
    }

    (anchors, fluid_seeds)
}
