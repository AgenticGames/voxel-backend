//! Frozen Grotto zone: Eisriesenwelt-inspired ice cave generation.
//!
//! 5-step algorithm:
//! 1. Seed points + noise setup (growth centers for ice coverage)
//! 2. Noise-driven ice coating (organic material layering by surface type)
//! 3. Icicle clusters at seed points (stalactites/stalagmites/columns)
//! 4. Frozen waterfalls (wall cascades + floor-to-ceiling columns)
//! 5. BlackIce floor enhancement (noise-driven pools, not random scatter)
//!
//! Materials: Ice (clear blue), Hoarfrost (white crystalline frost),
//! BlackIce (dark glossy floor patches), Permafrost (frozen earth edges).

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
    let extent = volume.world_bbox_max - volume.world_bbox_min;
    let seed_radius = extent.length() * 0.4; // influence radius = 40% of zone extent

    // ── Step 1: Seed Points + Noise Setup ──
    // Collect floor positions for seeding
    let mut all_floors: Vec<Vec3> = Vec::new();
    for &key in &volume.chunk_keys {
        if let Some(density) = density_fields.get(&key) {
            let size = density.size;
            let vs = eb / (size - 1) as f32;
            let origin = Vec3::new(key.0 as f32 * eb, key.1 as f32 * eb, key.2 as f32 * eb);
            for z in (0..size).step_by(3) {
                for y in 0..size {
                    for x in (0..size).step_by(3) {
                        let idx = z * size * size + y * size + x;
                        if density.samples[idx].density <= 0.0 { continue; }
                        let above_idx = z * size * size + (y + 1) * size + x;
                        if y + 1 < size && density.samples[above_idx].density <= 0.0 {
                            let wp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                            all_floors.push(wp);
                        }
                    }
                }
            }
        }
    }

    // Pick 4-8 ice growth seed points
    let num_seeds = rng.gen_range(4u32..=8).min(all_floors.len() as u32);
    let mut ice_seeds: Vec<Vec3> = Vec::new();
    for _ in 0..num_seeds {
        if all_floors.is_empty() { break; }
        let idx = rng.gen_range(0..all_floors.len());
        ice_seeds.push(all_floors.swap_remove(idx));
    }

    // Two noise layers
    let frost_noise = Simplex3D::new(zone_seed.wrapping_add(200));
    let ice_detail = Simplex3D::new(zone_seed.wrapping_add(300));
    let frost_freq = 0.12f64;
    let detail_freq = 0.25f64;

    // ── Step 2: Noise-Driven Ice Coating ──
    // Organic material layering based on surface type + noise + seed proximity
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
                        if density.samples[idx].material.is_ore() { continue; }

                        let wp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);

                        // Surface classification
                        let above_idx = z * size * size + (y + 1) * size + x;
                        let below_idx = if y > 0 { z * size * size + (y - 1) * size + x } else { idx };
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

                        // Noise + seed proximity
                        let noise_val = frost_noise.sample(
                            wp.x as f64 * frost_freq,
                            wp.y as f64 * frost_freq * 0.5,
                            wp.z as f64 * frost_freq,
                        ) as f32 * 0.5 + 0.5;

                        let seed_boost = ice_seeds.iter()
                            .map(|s| {
                                let d = (wp - *s).length();
                                (1.0 - (d / seed_radius).min(1.0)).powi(2)
                            })
                            .fold(0.0f32, f32::max);

                        let frost_value = noise_val + seed_boost * 0.5;

                        // Material assignment by surface type + frost intensity
                        let target_mat = if is_floor {
                            if frost_value > 0.65 { Some(Material::BlackIce) }
                            else if frost_value > 0.45 { Some(Material::Hoarfrost) }
                            else if frost_value > 0.25 { Some(Material::Permafrost) }
                            else { None }
                        } else if is_wall {
                            if frost_value > 0.70 { Some(Material::Hoarfrost) }
                            else if frost_value > 0.50 { Some(Material::Ice) }
                            else if frost_value > 0.30 { Some(Material::Permafrost) }
                            else { None }
                        } else if is_ceiling {
                            if frost_value > 0.60 { Some(Material::Ice) }
                            else if frost_value > 0.35 { Some(Material::Hoarfrost) }
                            else { None }
                        } else {
                            None
                        };

                        if let Some(mat) = target_mat {
                            if !density.samples[idx].material.is_ore() {
                                density.samples[idx].material = mat;
                            } else {
                                // Grow ice around ores (same hug pattern as bio zone)
                                for &(nx, ny, nz) in &[
                                    (x + 1, y, z), (x.wrapping_sub(1), y, z),
                                    (x, y + 1, z), (x, y.wrapping_sub(1), z),
                                    (x, y, z + 1), (x, y, z.wrapping_sub(1)),
                                ] {
                                    if nx < size && ny < size && nz < size {
                                        let ni = nz * size * size + ny * size + nx;
                                        if density.samples[ni].density > 0.0
                                            && !density.samples[ni].material.is_ore()
                                        {
                                            density.samples[ni].material = mat;
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

    // ── Step 3: Icicle Clusters at Seed Points ──
    // Larger, denser formations than old random scatter
    let mut cone_params: Vec<(Vec3, f32, f32, f32)> = Vec::new(); // (anchor, length, radius, direction)
    let mut ceiling_positions: Vec<(Vec3, f32)> = Vec::new(); // (pos, height_to_floor) for column detection

    {
        let mut rng2 = ChaCha8Rng::seed_from_u64(zone_seed.wrapping_add(100));

        for seed_pos in &ice_seeds {
            let cluster_radius = rng2.gen_range(4.0..8.0);
            let num_icicles = rng2.gen_range(5u32..=12);

            for _ in 0..num_icicles {
                let offset = Vec3::new(
                    rng2.gen_range(-cluster_radius..cluster_radius),
                    0.0,
                    rng2.gen_range(-cluster_radius..cluster_radius),
                );
                let candidate = *seed_pos + offset;

                // Check noise density — only place if ice_detail is high enough
                let detail_val = ice_detail.sample(
                    candidate.x as f64 * detail_freq,
                    candidate.y as f64 * detail_freq,
                    candidate.z as f64 * detail_freq,
                ) as f32;
                if detail_val < -0.2 { continue; } // skip sparse areas

                // Stalactite (70% ceiling) or stalagmite (30% floor)
                let roll: f32 = rng2.gen();
                if roll < 0.7 {
                    // Ceiling icicle
                    let length = rng2.gen_range(2.0..6.0);
                    let radius = rng2.gen_range(0.3..1.2);
                    // Offset upward to ceiling approximate
                    let ceiling_pos = candidate + Vec3::new(0.0, 8.0, 0.0);
                    cone_params.push((ceiling_pos, length, radius, -1.0));
                    ceiling_positions.push((ceiling_pos, length));
                } else {
                    // Floor stalagmite
                    let length = rng2.gen_range(1.0..4.0);
                    let radius = rng2.gen_range(0.3..0.8);
                    cone_params.push((candidate, length, radius, 1.0));
                }
            }
        }
    }

    // Write all cones
    for (anchor, length, radius, direction) in &cone_params {
        shapes::write_cone(
            density_fields, *anchor, *length, *radius, *direction,
            Material::Ice, 2.0, effective_bounds,
        );
    }

    // ── Step 4: Frozen Waterfalls ──

    // 4a: Wall Cascades — find tall wall surfaces
    let num_cascades = rng.gen_range(2u32..=4).min(config.frozen_waterfall_count.max(2));
    let mut cascade_count = 0u32;
    let mut cascade_candidates: Vec<(Vec3, Vec3)> = Vec::new(); // (top_pos, wall_normal)

    // Scan for wall voxels to find cascade sites
    for &key in &volume.chunk_keys {
        if cascade_candidates.len() >= 20 { break; } // enough candidates
        if let Some(density) = density_fields.get(&key) {
            let size = density.size;
            let vs = eb / (size - 1) as f32;
            let origin = Vec3::new(key.0 as f32 * eb, key.1 as f32 * eb, key.2 as f32 * eb);

            // Check a few positions per chunk for tall walls
            for z in (2..size - 2).step_by(4) {
                for x in (2..size - 2).step_by(4) {
                    // Find top of a wall column
                    let mut wall_height = 0;
                    let mut top_y = 0usize;
                    let mut normal = Vec3::ZERO;
                    for y in (0..size).rev() {
                        let idx = z * size * size + y * size + x;
                        if density.samples[idx].density <= 0.0 { continue; }
                        // Check if it's a wall (has horizontal air neighbor)
                        let mut has_h_air = false;
                        let mut n = Vec3::ZERO;
                        for &(nx, nz, dx, dz) in &[
                            (x + 1, z, 1.0f32, 0.0f32), (x.wrapping_sub(1), z, -1.0, 0.0),
                            (x, z + 1, 0.0, 1.0), (x, z.wrapping_sub(1), 0.0, -1.0),
                        ] {
                            if nx < size && nz < size {
                                let ni = nz * size * size + y * size + nx;
                                if density.samples[ni].density <= 0.0 {
                                    has_h_air = true;
                                    n = Vec3::new(dx, 0.0, dz);
                                }
                            }
                        }
                        if has_h_air {
                            if wall_height == 0 { top_y = y; normal = n; }
                            wall_height += 1;
                        } else if wall_height > 0 {
                            break;
                        }
                    }
                    if wall_height >= 6 {
                        let wp = origin + Vec3::new(x as f32 * vs, top_y as f32 * vs, z as f32 * vs);
                        cascade_candidates.push((wp, normal));
                    }
                }
            }
        }
    }

    // Place cascades at random candidates
    for _ in 0..num_cascades {
        if cascade_candidates.is_empty() { break; }
        let idx = rng.gen_range(0..cascade_candidates.len());
        let (top_pos, wall_normal) = cascade_candidates.swap_remove(idx);
        let height = rng.gen_range(5.0..12.0);
        let width = rng.gen_range(3.0..5.0);
        let thickness = rng.gen_range(1.0..2.0);

        shapes::write_wall_cascade(
            density_fields, top_pos, wall_normal, height, width, thickness,
            Material::Ice, effective_bounds,
        );
        cascade_count += 1;
    }

    // 4b: Floor-to-Ceiling Ice Columns (1-3)
    let num_columns = rng.gen_range(1u32..=3);
    let mut column_count = 0u32;

    for seed_pos in &ice_seeds {
        if column_count >= num_columns { break; }

        // Find floor and ceiling height at this seed position
        let mut floor_y = None;
        let mut ceiling_y = None;

        let cx = (seed_pos.x / eb).floor() as i32;
        let cy = (seed_pos.y / eb).floor() as i32;
        let cz = (seed_pos.z / eb).floor() as i32;

        // Search vertically for floor and ceiling
        for dy in -2..=2 {
            let check_cy = cy + dy;
            if let Some(density) = density_fields.get(&(cx, check_cy, cz)) {
                let size = density.size;
                let vs = eb / (size - 1) as f32;
                let local_x = ((seed_pos.x - cx as f32 * eb) / vs).round() as usize;
                let local_z = ((seed_pos.z - cz as f32 * eb) / vs).round() as usize;
                let lx = local_x.min(size - 1);
                let lz = local_z.min(size - 1);

                for y in 0..size {
                    let idx = lz * size * size + y * size + lx;
                    let world_y = check_cy as f32 * eb + y as f32 * vs;
                    if density.samples[idx].density > 0.0 {
                        if y + 1 < size {
                            let above = lz * size * size + (y + 1) * size + lx;
                            if density.samples[above].density <= 0.0 {
                                floor_y = Some(world_y);
                            }
                        }
                        if y > 0 {
                            let below = lz * size * size + (y - 1) * size + lx;
                            if density.samples[below].density <= 0.0 {
                                if ceiling_y.is_none() || world_y < ceiling_y.unwrap() {
                                    ceiling_y = Some(world_y);
                                }
                            }
                        }
                    }
                }
            }
        }

        if let (Some(fy), Some(cy_val)) = (floor_y, ceiling_y) {
            let gap = cy_val - fy;
            if gap > 4.0 && gap < 15.0 {
                let base = Vec3::new(seed_pos.x, fy, seed_pos.z);
                let top = Vec3::new(seed_pos.x, cy_val, seed_pos.z);
                let radius = rng.gen_range(1.5..3.0);

                shapes::write_ice_column(
                    density_fields, base, top, radius, 0.3,
                    Material::Ice, effective_bounds,
                );
                column_count += 1;
            }
        }
    }

    // ── Step 5: BlackIce Floor Enhancement ──
    // Noise-driven connected pools instead of random 25% scatter
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

                        let above_idx = z * size * size + (y + 1) * size + x;
                        if !(y + 1 < size && density.samples[above_idx].density <= 0.0) { continue; }

                        let wp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);

                        // Noise-driven BlackIce: connected pools near seeds
                        let detail_val = ice_detail.sample(
                            wp.x as f64 * detail_freq,
                            wp.y as f64 * detail_freq * 0.3, // compress Y for horizontal pools
                            wp.z as f64 * detail_freq,
                        ) as f32 * 0.5 + 0.5;

                        let seed_proximity = ice_seeds.iter()
                            .map(|s| {
                                let d = (wp - *s).length();
                                (1.0 - (d / (seed_radius * 0.5)).min(1.0)).powi(2)
                            })
                            .fold(0.0f32, f32::max);

                        let combined = detail_val + seed_proximity * 0.4;

                        if combined > 0.55 {
                            // Only override if already an ice material (don't replace ore)
                            let current = density.samples[idx].material;
                            if current == Material::Permafrost || current == Material::Hoarfrost
                                || current == Material::Ice || current == Material::BlackIce
                            {
                                density.samples[idx].material = Material::BlackIce;
                                // 2 voxels deep for visual depth
                                if y > 1 {
                                    let below = z * size * size + (y - 1) * size + x;
                                    if density.samples[below].density > 0.0
                                        && !density.samples[below].material.is_ore()
                                    {
                                        density.samples[below].material = Material::BlackIce;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ── Step 6: Ice Pit Traps ──
    // 1-2 hidden pits under BlackIce floor: bowl-shaped cavity with ice spike stalagmites,
    // thin ice cap on top with a small gap crack. BlackIce sliding pulls players toward center.
    let num_pits = rng.gen_range(1u32..=2);
    let mut pit_count = 0u32;

    // Find BlackIce floor positions far enough from zone edges
    let pit_margin = extent.length() * 0.15;
    let mut pit_candidates: Vec<Vec3> = Vec::new();
    for &key in &volume.chunk_keys {
        if let Some(density) = density_fields.get(&key) {
            let size = density.size;
            let vs = eb / (size - 1) as f32;
            let origin = Vec3::new(key.0 as f32 * eb, key.1 as f32 * eb, key.2 as f32 * eb);
            for z in (2..size - 2).step_by(4) {
                for x in (2..size - 2).step_by(4) {
                    for y in 0..size {
                        let idx = z * size * size + y * size + x;
                        if density.samples[idx].density <= 0.0 { continue; }
                        if density.samples[idx].material != Material::BlackIce { continue; }
                        let above = z * size * size + (y + 1) * size + x;
                        if y + 1 < size && density.samples[above].density <= 0.0 {
                            let wp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                            // Must be away from zone edges
                            let to_center = (wp - volume.world_center).length();
                            if to_center < extent.length() * 0.35 {
                                pit_candidates.push(wp);
                            }
                        }
                    }
                }
            }
        }
    }

    for _ in 0..num_pits {
        if pit_candidates.is_empty() { break; }
        let idx = rng.gen_range(0..pit_candidates.len());
        let pit_center = pit_candidates.swap_remove(idx);
        // Remove nearby candidates so pits don't overlap
        pit_candidates.retain(|p| (p.x - pit_center.x).abs() > 8.0 || (p.z - pit_center.z).abs() > 8.0);

        let pit_radius = rng.gen_range(3.0..5.0);
        let pit_depth = rng.gen_range(4.0..6.0);
        let crack_radius = 0.8; // small center gap for player to fall through

        // Carve bowl-shaped pit beneath the floor
        for &key in &volume.chunk_keys {
            if let Some(density) = density_fields.get_mut(&key) {
                let size = density.size;
                let vs = eb / (size - 1) as f32;
                let origin = Vec3::new(key.0 as f32 * eb, key.1 as f32 * eb, key.2 as f32 * eb);

                for z in 0..size {
                    for y in 0..size {
                        for x in 0..size {
                            let wp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                            let dx = wp.x - pit_center.x;
                            let dz = wp.z - pit_center.z;
                            let dist_xz = (dx * dx + dz * dz).sqrt();
                            let dy = pit_center.y - wp.y; // positive = below pit surface

                            if dist_xz > pit_radius + 1.0 { continue; }
                            if dy < 0.5 || dy > pit_depth + 1.0 { continue; }

                            let idx = z * size * size + y * size + x;

                            // Bowl profile: deeper in center, shallow at edges
                            let t_radius = (dist_xz / pit_radius).min(1.0);
                            let max_depth_here = pit_depth * (1.0 - t_radius * t_radius); // parabolic bowl

                            if dy <= max_depth_here && density.samples[idx].density > 0.0 {
                                density.samples[idx].density = -1.0;
                                density.samples[idx].material = Material::Air;
                            }
                        }
                    }
                }
            }
        }

        // Punch a small crack hole in the ice cap at center
        for &key in &volume.chunk_keys {
            if let Some(density) = density_fields.get_mut(&key) {
                let size = density.size;
                let vs = eb / (size - 1) as f32;
                let origin = Vec3::new(key.0 as f32 * eb, key.1 as f32 * eb, key.2 as f32 * eb);

                for z in 0..size {
                    for y in 0..size {
                        for x in 0..size {
                            let wp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                            let dx = wp.x - pit_center.x;
                            let dz = wp.z - pit_center.z;
                            let dist_xz = (dx * dx + dz * dz).sqrt();
                            let dy = (wp.y - pit_center.y).abs();

                            // Crack hole: remove ice cap within crack_radius of center
                            if dist_xz < crack_radius && dy < 1.5 {
                                let idx = z * size * size + y * size + x;
                                if density.samples[idx].density > 0.0 {
                                    density.samples[idx].density = -1.0;
                                    density.samples[idx].material = Material::Air;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Place ice spike stalagmites in the pit floor
        let num_spikes = rng.gen_range(5u32..=10);
        for _ in 0..num_spikes {
            let spike_offset = Vec3::new(
                rng.gen_range(-pit_radius * 0.7..pit_radius * 0.7),
                0.0,
                rng.gen_range(-pit_radius * 0.7..pit_radius * 0.7),
            );
            let spike_base = Vec3::new(
                pit_center.x + spike_offset.x,
                pit_center.y - pit_depth + 0.5, // pit floor
                pit_center.z + spike_offset.z,
            );
            let spike_len = rng.gen_range(1.5..3.5);
            let spike_rad = rng.gen_range(0.2..0.6);

            shapes::write_cone(
                density_fields, spike_base, spike_len, spike_rad, 1.0,
                Material::Ice, 2.5, effective_bounds,
            );
        }

        pit_count += 1;
    }

    let _ = (cascade_count, column_count, pit_count); // suppress unused warnings

    (Vec::new(), Vec::new())
}
