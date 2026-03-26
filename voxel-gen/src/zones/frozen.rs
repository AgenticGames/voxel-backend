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

use crate::config::{ZoneConfig, ZoneType};
use crate::density::DensityField;
use crate::pools::FluidSeed;
use crate::worm::path::WormSegment;

use super::detect::CavernVolume;
use super::shapes;
use super::{ZoneAnchor, ZoneBounds, ZoneDescriptor};

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

// ═══════════════════════════════════════════════════════════════════════════════
// Frozen Mega-Vault: 12×7×12 chunk fissure network carved from solid rock
// ═══════════════════════════════════════════════════════════════════════════════

/// Try to place a frozen mega-vault. Returns None if roll fails or no valid location found.
/// Carves 2-4 parallel fissures with ice walls, connecting tunnels, and tiered ledges.
/// Natural entrances are worm tunnels that intersect the vault boundary.
pub fn try_place_mega_vault(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    config: &ZoneConfig,
    global_seed: u64,
    effective_bounds: f32,
    worm_paths: &[Vec<WormSegment>],
) -> Option<(ZoneDescriptor, ZoneBounds)> {
    let mut rng = ChaCha8Rng::seed_from_u64(global_seed.wrapping_add(0xAE6A_F001));
    let eb = effective_bounds;

    // Roll for mega-vault (3% chance)
    let roll: f32 = rng.gen();
    if roll > config.frozen_mega_chance {
        return None;
    }

    // Vault dimensions in chunks
    let vault_cx = 12i32;
    let vault_cy = 7i32;
    let vault_cz = 12i32;

    // Find a valid location: scan for a block of mostly-solid chunks at depth 80-160
    let all_keys: Vec<(i32, i32, i32)> = density_fields.keys().copied().collect();
    if all_keys.is_empty() { return None; }

    // Find the Y range that corresponds to depth 80-160
    let target_y_min = (-160.0 / eb).floor() as i32; // deeper = more negative Y
    let target_y_max = (-80.0 / eb).floor() as i32;

    // Find X/Z bounds of available chunks
    let min_x = all_keys.iter().map(|k| k.0).min().unwrap();
    let max_x = all_keys.iter().map(|k| k.0).max().unwrap();
    let min_z = all_keys.iter().map(|k| k.2).min().unwrap();
    let max_z = all_keys.iter().map(|k| k.2).max().unwrap();

    // Try random positions for the vault
    let mut vault_origin: Option<(i32, i32, i32)> = None;
    for _ in 0..20 {
        let ox = rng.gen_range(min_x..=(max_x - vault_cx + 1).max(min_x));
        let oy = rng.gen_range(target_y_min..=(target_y_max - vault_cy + 1).max(target_y_min));
        let oz = rng.gen_range(min_z..=(max_z - vault_cz + 1).max(min_z));

        // Check that most chunks in the vault region exist and are mostly solid
        let mut total = 0u32;
        let mut solid = 0u32;
        for cy in oy..oy + vault_cy {
            for cz in oz..oz + vault_cz {
                for cx in ox..ox + vault_cx {
                    if density_fields.contains_key(&(cx, cy, cz)) {
                        total += 1;
                        // Quick check: sample center voxel
                        if let Some(d) = density_fields.get(&(cx, cy, cz)) {
                            let mid = d.size / 2;
                            let mid_idx = mid * d.size * d.size + mid * d.size + mid;
                            if d.samples[mid_idx].density > 0.0 {
                                solid += 1;
                            }
                        }
                    }
                }
            }
        }

        // Need >70% of chunks to exist and be solid
        let needed = (vault_cx * vault_cy * vault_cz) as u32;
        if total >= needed * 7 / 10 && solid >= total * 7 / 10 {
            vault_origin = Some((ox, oy, oz));
            break;
        }
    }

    let (ox, oy, oz) = vault_origin?;

    let world_min = Vec3::new(ox as f32 * eb, oy as f32 * eb, oz as f32 * eb);
    let world_max = Vec3::new(
        (ox + vault_cx) as f32 * eb,
        (oy + vault_cy) as f32 * eb,
        (oz + vault_cz) as f32 * eb,
    );
    let world_center = (world_min + world_max) * 0.5;

    // Noise for organic fissure shapes
    let fissure_noise = Simplex3D::new(global_seed.wrapping_add(0xF155_0001));
    let ledge_noise = Simplex3D::new(global_seed.wrapping_add(0xF155_0002));
    let fissure_freq = 0.08f64;

    // Fissure parameters (in voxels)
    let num_fissures = rng.gen_range(2u32..=4);
    let fissure_width = rng.gen_range(4.0f32..6.0);
    let wall_thickness = rng.gen_range(8.0f32..12.0);
    let vault_voxels_x = vault_cx as f32 * eb / (eb / 16.0); // approximate
    let vault_voxels_y = vault_cy as f32 * 16.0;
    let vault_voxels_z = vault_cz as f32 * 16.0;

    // Fissures run along Z axis (longest horizontal), spaced along X
    let total_fissure_width = num_fissures as f32 * fissure_width
        + (num_fissures as f32 - 1.0) * wall_thickness;
    let fissure_start_x = world_center.x - total_fissure_width * 0.5;

    // Carve fissures
    let margin_y = 2.0; // leave some solid at top/bottom of vault
    let fissure_floor_y = world_min.y + margin_y;
    let fissure_ceil_y = world_max.y - margin_y;
    let fissure_min_z = world_min.z + 2.0;
    let fissure_max_z = world_max.z - 2.0;

    // Store fissure center X positions for tunnel/ledge carving
    let mut fissure_centers_x: Vec<f32> = Vec::new();

    for fi in 0..num_fissures {
        let center_x = fissure_start_x + fi as f32 * (fissure_width + wall_thickness) + fissure_width * 0.5;
        fissure_centers_x.push(center_x);

        // Carve this fissure through all chunks it overlaps
        for cy in oy..oy + vault_cy {
            for cz in oz..oz + vault_cz {
                for cx in ox..ox + vault_cx {
                    if let Some(density) = density_fields.get_mut(&(cx, cy, cz)) {
                        let size = density.size;
                        let vs = eb / (size - 1) as f32;
                        let origin = Vec3::new(cx as f32 * eb, cy as f32 * eb, cz as f32 * eb);

                        for z in 0..size {
                            for y in 0..size {
                                for x in 0..size {
                                    let wp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);

                                    // Vertical bounds
                                    if wp.y < fissure_floor_y || wp.y > fissure_ceil_y { continue; }
                                    if wp.z < fissure_min_z || wp.z > fissure_max_z { continue; }

                                    // Noise-driven floor/ceiling undulation
                                    let floor_offset = fissure_noise.sample(
                                        wp.z as f64 * fissure_freq, 0.0, fi as f64 * 100.0,
                                    ) as f32 * 3.0;
                                    let ceil_offset = fissure_noise.sample(
                                        wp.z as f64 * fissure_freq, 100.0, fi as f64 * 100.0,
                                    ) as f32 * 3.0;
                                    let local_floor = fissure_floor_y + floor_offset.abs();
                                    let local_ceil = fissure_ceil_y - ceil_offset.abs();

                                    if wp.y < local_floor || wp.y > local_ceil { continue; }

                                    // Fissure width with noise waviness
                                    let waver = fissure_noise.sample(
                                        wp.z as f64 * fissure_freq * 0.5,
                                        wp.y as f64 * 0.02,
                                        (fi as f64 + 0.5) * 100.0,
                                    ) as f32 * 2.0;
                                    let dist_from_center = (wp.x - center_x - waver).abs();

                                    if dist_from_center < fissure_width * 0.5 {
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
            }
        }
    }

    // Carve connecting tunnels between adjacent fissures
    let tunnel_spacing = rng.gen_range(20.0f32..30.0);
    for fi in 0..(num_fissures - 1) as usize {
        let left_x = fissure_centers_x[fi];
        let right_x = fissure_centers_x[fi + 1];
        let wall_center_x = (left_x + right_x) * 0.5;

        let mut z_pos = fissure_min_z + rng.gen_range(5.0..15.0);
        while z_pos < fissure_max_z - 5.0 {
            // Tunnel height varies
            let tunnel_y = fissure_floor_y + rng.gen_range(2.0..(fissure_ceil_y - fissure_floor_y - 4.0).max(3.0));
            let tunnel_h = 3.0f32; // 3 voxels tall
            let tunnel_w = 3.0f32; // 3 voxels wide (along Z)

            // 35% chance to block this tunnel with ice
            let blocked: f32 = rng.gen();
            let is_blocked = blocked < 0.35;

            // Carve tunnel from left fissure to right fissure through the wall
            for cy in oy..oy + vault_cy {
                for cz in oz..oz + vault_cz {
                    for cx in ox..ox + vault_cx {
                        if let Some(density) = density_fields.get_mut(&(cx, cy, cz)) {
                            let size = density.size;
                            let vs = eb / (size - 1) as f32;
                            let origin = Vec3::new(cx as f32 * eb, cy as f32 * eb, cz as f32 * eb);

                            for z in 0..size {
                                for y in 0..size {
                                    for x in 0..size {
                                        let wp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);

                                        // Tunnel bounds
                                        if (wp.z - z_pos).abs() > tunnel_w * 0.5 { continue; }
                                        if wp.y < tunnel_y || wp.y > tunnel_y + tunnel_h { continue; }
                                        if wp.x < left_x - fissure_width * 0.3 || wp.x > right_x + fissure_width * 0.3 { continue; }

                                        // Only carve through the wall area (not the fissures themselves)
                                        let in_wall = wp.x > left_x + fissure_width * 0.3 && wp.x < right_x - fissure_width * 0.3;

                                        if in_wall {
                                            let idx = z * size * size + y * size + x;
                                            if is_blocked {
                                                // Sealed with thick ice — player must mine through
                                                density.samples[idx].density = 1.0;
                                                density.samples[idx].material = Material::Ice;
                                            } else if density.samples[idx].density > 0.0 {
                                                density.samples[idx].density = -1.0;
                                                density.samples[idx].material = Material::Air;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            z_pos += tunnel_spacing + rng.gen_range(-5.0..5.0);
        }
    }

    // Carve tiered ledges along fissure walls
    let ledge_spacing = 5.0f32;
    for &center_x in &fissure_centers_x {
        let mut ledge_y = fissure_floor_y + ledge_spacing;
        while ledge_y < fissure_ceil_y - ledge_spacing {
            // Ledge on left wall
            for side in &[-1.0f32, 1.0f32] {
                let wall_x = center_x + side * fissure_width * 0.5;

                // Noise-driven: not every position gets a ledge
                let ledge_val = ledge_noise.sample(
                    wall_x as f64 * 0.1, ledge_y as f64 * 0.1, center_x as f64 * 0.05,
                ) as f32;
                if ledge_val < -0.1 { continue; } // skip ~40% of positions

                let ledge_depth = rng.gen_range(2.0f32..3.0);
                let ledge_length = rng.gen_range(8.0f32..15.0);
                let ledge_z_start = fissure_min_z + rng.gen_range(3.0..(fissure_max_z - fissure_min_z - ledge_length - 3.0).max(4.0));

                for cy in oy..oy + vault_cy {
                    for cz in oz..oz + vault_cz {
                        for cx in ox..ox + vault_cx {
                            if let Some(density) = density_fields.get_mut(&(cx, cy, cz)) {
                                let size = density.size;
                                let vs = eb / (size - 1) as f32;
                                let origin = Vec3::new(cx as f32 * eb, cy as f32 * eb, cz as f32 * eb);

                                for z in 0..size {
                                    for y in 0..size {
                                        for x in 0..size {
                                            let wp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);

                                            if wp.z < ledge_z_start || wp.z > ledge_z_start + ledge_length { continue; }
                                            if wp.y < ledge_y || wp.y > ledge_y + 1.0 { continue; } // 1 voxel thick shelf

                                            // Ledge extends from wall into fissure
                                            let dist_from_wall = if *side < 0.0 {
                                                wall_x - wp.x // left wall: ledge goes right (positive X)
                                            } else {
                                                wp.x - wall_x // right wall: ledge goes left (negative X)
                                            };

                                            if dist_from_wall >= 0.0 && dist_from_wall < ledge_depth {
                                                let idx = z * size * size + y * size + x;
                                                // Write solid Ice for the ledge
                                                density.samples[idx].density = 0.9;
                                                density.samples[idx].material = Material::Ice;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            ledge_y += ledge_spacing + rng.gen_range(-1.0..2.0);
        }
    }

    // Fill all solid within vault bounds with Ice material (ice walls between fissures)
    for cy in oy..oy + vault_cy {
        for cz in oz..oz + vault_cz {
            for cx in ox..ox + vault_cx {
                if let Some(density) = density_fields.get_mut(&(cx, cy, cz)) {
                    let size = density.size;
                    for idx in 0..size * size * size {
                        if density.samples[idx].density > 0.0
                            && density.samples[idx].material.is_host_rock()
                        {
                            density.samples[idx].material = Material::Ice;
                        }
                    }
                }
            }
        }
    }

    // Find worm tunnels intersecting vault boundary for natural entrances
    let mut has_entrance = false;
    for path in worm_paths {
        for seg in path {
            let p = seg.position;
            // Check if worm segment is near the vault boundary (within 2 chunks)
            let near_boundary = (p.x - world_min.x).abs() < eb * 2.0
                || (p.x - world_max.x).abs() < eb * 2.0
                || (p.z - world_min.z).abs() < eb * 2.0
                || (p.z - world_max.z).abs() < eb * 2.0;
            let in_y_range = p.y >= world_min.y && p.y <= world_max.y;
            if near_boundary && in_y_range {
                has_entrance = true;
                break;
            }
        }
        if has_entrance { break; }
    }

    // If no worm tunnels intersect, carve connector tunnels to nearest worm
    if !has_entrance {
        // Find the nearest worm segment to the vault center
        let mut best_seg: Option<Vec3> = None;
        let mut best_dist = f32::MAX;
        for path in worm_paths {
            for seg in path {
                let d = (seg.position - world_center).length();
                if d < best_dist && d > 0.0 {
                    best_dist = d;
                    best_seg = Some(seg.position);
                }
            }
        }

        if let Some(target) = best_seg {
            // Carve a straight tunnel from vault boundary toward the worm
            let dir = (target - world_center).normalize();
            let start = world_center;
            let steps = (best_dist / 1.0).min(80.0) as u32; // max 80 voxels of connector

            for s in 0..steps {
                let pos = start + dir * s as f32;
                let ck = ((pos.x / eb).floor() as i32, (pos.y / eb).floor() as i32, (pos.z / eb).floor() as i32);
                if let Some(density) = density_fields.get_mut(&ck) {
                    let size = density.size;
                    let vs = eb / (size - 1) as f32;
                    let origin = Vec3::new(ck.0 as f32 * eb, ck.1 as f32 * eb, ck.2 as f32 * eb);

                    for z in 0..size {
                        for y in 0..size {
                            for x in 0..size {
                                let wp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                let d = (wp - pos).length();
                                if d < 2.5 { // 2.5 voxel radius tunnel
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
        }
    }

    let descriptor = ZoneDescriptor {
        zone_type: ZoneType::FrozenGrotto,
        world_min,
        world_max,
        center: world_center,
        anchors: Vec::new(),
    };

    let zone_bounds = ZoneBounds {
        world_min,
        world_max,
        zone_type: ZoneType::FrozenGrotto,
    };

    Some((descriptor, zone_bounds))
}
