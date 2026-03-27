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
#[allow(unused_imports)]
use crate::config::ZoneType;
use crate::density::DensityField;
use crate::pools::FluidSeed;
#[allow(unused_imports)]
use crate::worm::path::WormSegment;

use super::detect::CavernVolume;
use super::shapes;
use super::ZoneAnchor;
#[allow(unused_imports)]
use super::{ZoneBounds, ZoneDescriptor};

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
// Frozen Mega-Vault (LEGACY)
// ═══════════════════════════════════════════════════════════════════════════════
//
// This multi-pass implementation is superseded by the Blueprint + Per-Chunk-Apply
// architecture in `mega_blueprint.rs` + `mega_apply.rs`. The new system:
//   1. Pre-computes all geometry parametrically in ~2ms (MegaVaultBlueprint::generate)
//   2. Applies to each chunk with ONE pass through 17^3 voxels (apply_vault_to_chunk)
//
// Kept as dead code for reference. The entry point in zones/mod.rs now calls
// the blueprint system directly instead of try_place_mega_vault().

/// Legacy mega-vault entry point. No longer called -- see mega_blueprint + mega_apply.
#[allow(dead_code)]
pub fn try_place_mega_vault(
    _density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    _config: &ZoneConfig,
    _global_seed: u64,
    _effective_bounds: f32,
    _worm_paths: &[Vec<WormSegment>],
) -> Option<(ZoneDescriptor, ZoneBounds)> {
    // Superseded by mega_blueprint::MegaVaultBlueprint + mega_apply::apply_vault_to_chunk
    None
}

// Legacy implementation below kept for reference only.
#[allow(dead_code, unreachable_code, unused_variables, unused_mut)]
fn try_place_mega_vault_legacy(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    config: &ZoneConfig,
    global_seed: u64,
    effective_bounds: f32,
    _worm_paths: &[Vec<WormSegment>],
) -> Option<(ZoneDescriptor, ZoneBounds)> {
    let mut rng = ChaCha8Rng::seed_from_u64(global_seed.wrapping_add(0xAE6A_F001));
    let eb = effective_bounds;

    // Logging helper — writes to a file since eprintln doesn't reach UE
    fn vault_log(msg: &str) {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true)
            .open("D:/Unreal Projects/Mithril2026/Saved/mega_vault_log.txt")
        {
            let _ = writeln!(f, "{}", msg);
        }
    }

    return None;

    // Roll for mega-vault
    let roll: f32 = rng.gen();
    if roll > config.frozen_mega_chance {
        vault_log(&format!("[MegaVault] Roll {:.3} > chance {:.3}, skipping", roll, config.frozen_mega_chance));
        return None;
    }

    // Vault dimensions in chunks
    let vault_cx = 8i32;  // 30% wider for fat ledges
    let vault_cy = 9i32;  // 120% deeper (was 4)
    let vault_cz = 6i32;

    // Place vault centered on (0,0,0) — always overlaps initial streaming region
    let ox = -vault_cx / 2;
    let oy = -vault_cy / 2;
    let oz = -vault_cz / 2;

    // Log all incoming chunk keys
    let all_keys: Vec<(i32,i32,i32)> = density_fields.keys().copied().collect();
    vault_log(&format!("[MegaVault] Roll {:.3} PASSED. Vault bounds: ({},{},{}) to ({},{},{})",
        roll, ox, oy, oz, ox+vault_cx, oy+vault_cy, oz+vault_cz));
    vault_log(&format!("[MegaVault] Region has {} chunks. Sample keys: {:?}",
        all_keys.len(), &all_keys[..all_keys.len().min(20)]));

    // Log Y range of incoming chunks
    let min_y = all_keys.iter().map(|k| k.1).min().unwrap_or(0);
    let max_y = all_keys.iter().map(|k| k.1).max().unwrap_or(0);
    let min_x = all_keys.iter().map(|k| k.0).min().unwrap_or(0);
    let max_x = all_keys.iter().map(|k| k.0).max().unwrap_or(0);
    let min_z = all_keys.iter().map(|k| k.2).min().unwrap_or(0);
    let max_z = all_keys.iter().map(|k| k.2).max().unwrap_or(0);
    vault_log(&format!("[MegaVault] Region range: X={}..{} Y={}..{} Z={}..{}",
        min_x, max_x, min_y, max_y, min_z, max_z));

    let world_min = Vec3::new(ox as f32 * eb, oy as f32 * eb, oz as f32 * eb);
    let world_max = Vec3::new(
        (ox + vault_cx) as f32 * eb,
        (oy + vault_cy) as f32 * eb,
        (oz + vault_cz) as f32 * eb,
    );
    let world_center = (world_min + world_max) * 0.5;

    // Check if any existing chunks overlap the vault
    let overlapping: Vec<(i32, i32, i32)> = density_fields.keys()
        .filter(|&&(cx, cy, cz)| {
            cx >= ox && cx < ox + vault_cx
            && cy >= oy && cy < oy + vault_cy
            && cz >= oz && cz < oz + vault_cz
        })
        .copied()
        .collect();

    vault_log(&format!("[MegaVault] Overlapping chunks: {} of {} needed",
        overlapping.len(), vault_cx * vault_cy * vault_cz));

    if overlapping.is_empty() {
        vault_log(&format!("[MegaVault] NO OVERLAP — vault at ({},{},{}) doesn't intersect any streamed chunks", ox, oy, oz));
        return None;
    }

    vault_log(&format!("[MegaVault] CARVING {} fissures into {} chunks!", rng.gen_range(2u32..=3), overlapping.len()));

    // Noise for organic fissure shapes
    let fissure_noise = Simplex3D::new(global_seed.wrapping_add(0xF155_0001));
    let ledge_noise = Simplex3D::new(global_seed.wrapping_add(0xF155_0002));
    let fissure_freq = 0.08f64;

    // Fissure parameters — wide fissures, thick walls for cramped tunnels
    let num_fissures = rng.gen_range(2u32..=3);
    let fissure_width = eb * 1.5;      // 1.5 chunks wide per fissure
    let wall_thickness = eb * 1.6;     // 60% thicker walls for tunnel carving

    let total_fissure_width = num_fissures as f32 * fissure_width
        + (num_fissures as f32 - 1.0) * wall_thickness;
    let fissure_start_x = world_center.x - total_fissure_width * 0.5;

    let margin_y = eb * 0.25;
    let fissure_floor_y = world_min.y + margin_y;
    let fissure_ceil_y = world_max.y - margin_y;
    let margin_z = eb * 0.15;
    let fissure_min_z = world_min.z + margin_z;
    let fissure_max_z = world_max.z - margin_z;

    vault_log(&format!("[MegaVault] eb={}, fissure_width={:.1}, wall={:.1}, num={}, total_span={:.1}",
        eb, fissure_width, wall_thickness, num_fissures, total_fissure_width));

    let mut fissure_centers_x: Vec<f32> = Vec::new();

    // Carve fissures into overlapping chunks only
    for fi in 0..num_fissures {
        let center_x = fissure_start_x + fi as f32 * (fissure_width + wall_thickness) + fissure_width * 0.5;
        fissure_centers_x.push(center_x);

        for &key in &overlapping {
            if let Some(density) = density_fields.get_mut(&key) {
                let size = density.size;
                let vs = eb / (size - 1) as f32;
                let origin = Vec3::new(key.0 as f32 * eb, key.1 as f32 * eb, key.2 as f32 * eb);

                for z in 0..size { for y in 0..size { for x in 0..size {
                    let wp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                    if wp.y < fissure_floor_y || wp.y > fissure_ceil_y { continue; }
                    if wp.z < fissure_min_z || wp.z > fissure_max_z { continue; }

                    let floor_offset = fissure_noise.sample(
                        wp.z as f64 * fissure_freq, 0.0, fi as f64 * 100.0,
                    ) as f32 * eb * 0.2; // undulate ±20% of a chunk
                    let ceil_offset = fissure_noise.sample(
                        wp.z as f64 * fissure_freq, 100.0, fi as f64 * 100.0,
                    ) as f32 * eb * 0.2;
                    if wp.y < fissure_floor_y + floor_offset.abs() { continue; }
                    if wp.y > fissure_ceil_y - ceil_offset.abs() { continue; }

                    let waver = fissure_noise.sample(
                        wp.z as f64 * fissure_freq * 0.5,
                        wp.y as f64 * 0.02,
                        (fi as f64 + 0.5) * 100.0,
                    ) as f32 * eb * 0.3; // waver up to ±30% of a chunk

                    if (wp.x - center_x - waver).abs() < fissure_width * 0.5 {
                        let idx = z * size * size + y * size + x;
                        if density.samples[idx].density > 0.0 {
                            density.samples[idx].density = -1.0;
                            density.samples[idx].material = Material::Air;
                        }
                    }
                }}}
            }
        }
    }

    // Carve connecting tunnels — scaled to chunk units
    let tunnel_spacing = eb * rng.gen_range(1.0f32..2.0); // 1-2 chunks between tunnels
    for fi in 0..(num_fissures - 1) as usize {
        let left_x = fissure_centers_x[fi];
        let right_x = fissure_centers_x[fi + 1];

        let mut z_pos = fissure_min_z + eb * 0.5;
        while z_pos < fissure_max_z - eb * 0.5 {
            let vert_range = (fissure_ceil_y - fissure_floor_y - eb * 0.5).max(eb * 0.3);
            let tunnel_y = fissure_floor_y + rng.gen_range(eb * 0.2..vert_range);
            let tunnel_h = 3.0f32;
            let tunnel_w = 3.0f32;
            let blocked: f32 = rng.gen();
            let is_blocked = blocked < 0.35;

            for &key in &overlapping {
                if let Some(density) = density_fields.get_mut(&key) {
                    let size = density.size;
                    let vs = eb / (size - 1) as f32;
                    let origin = Vec3::new(key.0 as f32 * eb, key.1 as f32 * eb, key.2 as f32 * eb);

                    for z in 0..size { for y in 0..size { for x in 0..size {
                        let wp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                        if (wp.z - z_pos).abs() > tunnel_w * 0.5 { continue; }
                        if wp.y < tunnel_y || wp.y > tunnel_y + tunnel_h { continue; }
                        if wp.x < left_x - fissure_width * 0.3 || wp.x > right_x + fissure_width * 0.3 { continue; }

                        let in_wall = wp.x > left_x + fissure_width * 0.3 && wp.x < right_x - fissure_width * 0.3;
                        if in_wall {
                            let idx = z * size * size + y * size + x;
                            if is_blocked {
                                density.samples[idx].density = 1.0;
                                density.samples[idx].material = Material::Ice;
                            } else if density.samples[idx].density > 0.0 {
                                density.samples[idx].density = -1.0;
                                density.samples[idx].material = Material::Air;
                            }
                        }
                    }}}
                }
            }
            z_pos += tunnel_spacing + rng.gen_range(-eb * 0.3..eb * 0.3);
        }
    }

    // Seal worm holes: fill ALL air in vault bounds with solid, then re-carve fissures
    // Much faster than per-voxel fissure membership check with noise samples
    for &key in &overlapping {
        if let Some(density) = density_fields.get_mut(&key) {
            let size = density.size;
            for idx in 0..size * size * size {
                if density.samples[idx].density <= 0.0 {
                    density.samples[idx].density = 1.0;
                    density.samples[idx].material = Material::Ice;
                }
            }
        }
    }

    // Re-carve fissures (same logic as above, now cutting through sealed worm holes)
    for fi in 0..num_fissures {
        let center_x = fissure_centers_x[fi as usize];
        for &key in &overlapping {
            if let Some(density) = density_fields.get_mut(&key) {
                let size = density.size;
                let vs = eb / (size - 1) as f32;
                let origin = Vec3::new(key.0 as f32 * eb, key.1 as f32 * eb, key.2 as f32 * eb);
                for z in 0..size { for y in 0..size { for x in 0..size {
                    let wp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                    if wp.y < fissure_floor_y || wp.y > fissure_ceil_y { continue; }
                    if wp.z < fissure_min_z || wp.z > fissure_max_z { continue; }
                    let floor_offset = fissure_noise.sample(
                        wp.z as f64 * fissure_freq, 0.0, fi as f64 * 100.0,
                    ) as f32 * eb * 0.2;
                    let ceil_offset = fissure_noise.sample(
                        wp.z as f64 * fissure_freq, 100.0, fi as f64 * 100.0,
                    ) as f32 * eb * 0.2;
                    if wp.y < fissure_floor_y + floor_offset.abs() { continue; }
                    if wp.y > fissure_ceil_y - ceil_offset.abs() { continue; }
                    let waver = fissure_noise.sample(
                        wp.z as f64 * fissure_freq * 0.5, wp.y as f64 * 0.02,
                        (fi as f64 + 0.5) * 100.0,
                    ) as f32 * eb * 0.3;
                    if (wp.x - center_x - waver).abs() < fissure_width * 0.5 {
                        let idx = z * size * size + y * size + x;
                        if density.samples[idx].density > 0.0 {
                            density.samples[idx].density = -1.0;
                            density.samples[idx].material = Material::Air;
                        }
                    }
                }}}
            }
        }
    }

    // Winding paths along fissure walls — organic walkways that snake along Z
    // with noise-driven Y wander, occasional cross-bridges between walls
    let path_noise = Simplex3D::new(global_seed.wrapping_add(0xF155_0004));
    let mut stalagmite_params: Vec<(Vec3, f32, f32)> = Vec::new(); // collected during path walk
    let path_freq = 0.06f64;

    for (fi, &center_x) in fissure_centers_x.iter().enumerate() {
        // 2-3 path tiers per fissure at different heights
        let num_tiers = rng.gen_range(10u32..=12);
        let tier_spacing = (fissure_ceil_y - fissure_floor_y) / (num_tiers as f32 + 1.0);

        for tier in 0..num_tiers {
            let base_y = fissure_floor_y + tier_spacing * (tier as f32 + 1.0);
            let side: f32 = if rng.gen_bool(0.5) { -1.0 } else { 1.0 };
            let wall_x = center_x + side * fissure_width * 0.5;
            let path_width = rng.gen_range(4.6f32..6.6); // +15% wider again
            let path_thickness = rng.gen_range(2.5f32..3.5); // slightly thicker too

            // Walk along Z, with noise-driven Y wander
            let z_step = 1.0f32; // sample every voxel along Z
            let mut z = fissure_min_z + rng.gen_range(1.0..3.0);
            let fissure_z_len = fissure_max_z - fissure_min_z;

            // Occasionally bridge across to the other wall
            // 1-2 cross-bridges per tier path
            let num_bridges = rng.gen_range(1u32..=2);
            let mut bridge_zs: Vec<f32> = Vec::new();
            for _ in 0..num_bridges {
                bridge_zs.push(fissure_min_z + rng.gen_range(fissure_z_len * 0.15..fissure_z_len * 0.85));
            }
            let bridge_width = rng.gen_range(2.5f32..4.0);

            while z < fissure_max_z - 1.0 {
                // Noise-driven Y offset: path wanders up and down ±3 voxels
                let y_wander = path_noise.sample(
                    z as f64 * path_freq,
                    (fi as f64 + tier as f64 * 10.0) * 50.0,
                    side as f64 * 100.0,
                ) as f32 * 4.0;
                let path_y = base_y + y_wander;

                // Noise-driven width variation + periodic thickness wave
                let width_noise = path_noise.sample(
                    z as f64 * path_freq * 2.0,
                    100.0 + fi as f64 * 50.0,
                    tier as f64 * 30.0,
                ) as f32;
                // Thickness wave: every ~15 voxels, bulge out 30-40% wider
                // Bulge wave: 15% more frequent, flatter peak (+30% length), apex gets extra wide
                let raw_wave = ((z as f64 * 0.46).sin() * 0.5 + 0.5) as f32; // 15% higher freq
                let wave = if raw_wave > 0.5 {
                    // Above midpoint: flatten the peak (hold wider longer)
                    let peak_t = (raw_wave - 0.5) * 2.0; // 0..1 within peak zone
                    let flattened = 0.5 + peak_t.powf(0.6) * 0.5; // flatter top = longer bulge
                    // Apex bonus: extra wide at the very top
                    if raw_wave > 0.85 { flattened * 1.25 } else { flattened }
                } else {
                    raw_wave
                };
                let bulge = path_width * 0.70 * wave; // +10% wider bulges
                let local_width = path_width + width_noise * 2.0 + bulge;

                // Tunnel mode: occasionally the path dives INTO the wall as a carved tunnel
                // Uses a slower noise to create ~20% tunnel sections along the path
                let tunnel_val = path_noise.sample(
                    z as f64 * 0.03, // slow frequency = long tunnel sections
                    200.0 + fi as f64 * 70.0,
                    tier as f64 * 40.0 + side as f64 * 50.0,
                ) as f32;
                let is_tunnel = tunnel_val > -0.1; // ~55% of path is tunneled (6x more)
                let tunnel_depth = 16.0f32; // +8 deeper into the wall
                let tunnel_height = (path_thickness + 2.0) * 2.5; // 2.5x headroom

                // Spawn stalagmite on bulge peaks — more at apex, only on ledges not tunnels
                let stag_chance = if wave > 0.95 { 0.7 } else if wave > 0.7 { 0.4 } else { 0.0 };
                if stag_chance > 0.0 && !is_tunnel && rng.gen::<f32>() < stag_chance {
                    // Place stalagmite BEYOND the ledge edge — create its own platform
                    let platform_extend = rng.gen_range(2.5f32..4.0); // extra platform beyond bulge
                    let stag_x = if side < 0.0 {
                        wall_x + local_width + platform_extend * 0.5 // past the ledge edge
                    } else {
                        wall_x - local_width - platform_extend * 0.5
                    };
                    let stag_base = Vec3::new(stag_x, path_y + path_thickness, z);
                    let stag_len = rng.gen_range(3.0..7.0);
                    let stag_rad = rng.gen_range(0.8..1.8);
                    stalagmite_params.push((stag_base, stag_len, stag_rad));

                    // Write a small ice platform under the stalagmite
                    let plat_radius = stag_rad + platform_extend;
                    for &key in &overlapping {
                        if let Some(density) = density_fields.get_mut(&key) {
                            let size = density.size;
                            let vs = eb / (size - 1) as f32;
                            let origin = Vec3::new(key.0 as f32 * eb, key.1 as f32 * eb, key.2 as f32 * eb);
                            for vz in 0..size { for vy in 0..size { for vx in 0..size {
                                let wp = origin + Vec3::new(vx as f32 * vs, vy as f32 * vs, vz as f32 * vs);
                                let dx = wp.x - stag_base.x;
                                let dz = wp.z - stag_base.z;
                                let dist_h = (dx * dx + dz * dz).sqrt();
                                if dist_h > plat_radius { continue; }
                                // Platform is 2 voxels thick at the ledge Y level
                                if wp.y < path_y || wp.y > path_y + path_thickness { continue; }
                                let idx = vz * size * size + vy * size + vx;
                                if 0.85 > density.samples[idx].density {
                                    density.samples[idx].density = 0.85;
                                    density.samples[idx].material = Material::IceSheet;
                                }
                            }}}
                        }
                    }
                }

                // Write path segment at this Z position
                for &key in &overlapping {
                    if let Some(density) = density_fields.get_mut(&key) {
                        let size = density.size;
                        let vs = eb / (size - 1) as f32;
                        let origin = Vec3::new(key.0 as f32 * eb, key.1 as f32 * eb, key.2 as f32 * eb);

                        for vy in 0..size { for vx in 0..size { for vz in 0..size {
                            let wp = origin + Vec3::new(vx as f32 * vs, vy as f32 * vs, vz as f32 * vs);

                            if (wp.z - z).abs() > z_step * 1.5 { continue; }

                            if is_tunnel {
                                // TUNNEL MODE: rounded cross-section carved into the wall
                                let into_wall = if side < 0.0 { wall_x - wp.x } else { wp.x - wall_x };
                                if into_wall < -3.0 || into_wall > tunnel_depth + 2.0 { continue; }

                                // Entrance flare: wider opening near the wall face
                                let entrance_t = (into_wall / (tunnel_depth * 0.3)).clamp(0.0, 1.0);
                                let entrance_scale = 1.0 + (1.0 - entrance_t) * 0.5; // 50% wider at mouth

                                let floor_wobble = path_noise.sample(
                                    wp.z as f64 * 0.15, 500.0 + fi as f64 * 30.0, tier as f64 * 20.0,
                                ) as f32 * 1.5;
                                let tunnel_center_y = path_y + tunnel_height * 0.45 + floor_wobble;
                                let tunnel_center_x_depth = tunnel_depth * 0.5;

                                // Elliptical cross-section with entrance flare
                                let rx = tunnel_depth * 0.55;
                                let ry = tunnel_height * 0.5 * entrance_scale; // taller at entrance
                                let dx_norm = (into_wall - tunnel_center_x_depth) / rx;
                                let dy_norm = (wp.y - tunnel_center_y) / ry;
                                let dist_sq = dx_norm * dx_norm + dy_norm * dy_norm;

                                if dist_sq < 1.0 {
                                    let idx = vz * size * size + vy * size + vx;
                                    if density.samples[idx].density > 0.0 {
                                        density.samples[idx].density = -1.0;
                                        density.samples[idx].material = Material::Air;
                                    }
                                }
                                continue;
                            }

                            // LEDGE MODE: protruding path, thicker at wall for structural support
                            let protrusion = if side < 0.0 { wp.x - wall_x } else { wall_x - wp.x };
                            if protrusion < -2.0 || protrusion > local_width { continue; }

                            // Buttress: ledge is thicker near the wall, thins toward the edge
                            // At wall (protrusion=0): full thickness + 50% extra underside
                            // At edge (protrusion=local_width): just path_thickness
                            let t_across = (protrusion / local_width).clamp(0.0, 1.0);
                            let buttress_extra = path_thickness * 0.5 * (1.0 - t_across); // tapers to 0 at edge
                            let local_top = path_y + path_thickness;
                            let local_bottom = path_y - buttress_extra; // extends DOWN near wall

                            if wp.y < local_bottom || wp.y > local_top { continue; }

                            let idx = vz * size * size + vy * size + vx;
                            let edge_fade = if protrusion > local_width - 1.5 {
                                (local_width - protrusion) / 1.5
                            } else { 1.0 };
                            let d = (0.9 * edge_fade).max(0.1);
                            if d > density.samples[idx].density {
                                density.samples[idx].density = d;
                                // Top surface = Hoarfrost, underside/sides = IceSheet
                                let is_top = wp.y > path_y + path_thickness - 1.0;
                                density.samples[idx].material = if is_top {
                                    Material::Hoarfrost
                                } else {
                                    Material::IceSheet
                                };
                            }
                        }}}
                    }
                }

                // Organic cross-bridge: verify far wall exists, then build winding path
                let near_bridge = bridge_zs.iter().any(|bz| (z - *bz).abs() < 2.0);
                if near_bridge {
                    let start_y = path_y;
                    let other_wall_x = center_x - side * fissure_width * 0.5;
                    let end_y_offset = path_noise.sample(
                        z as f64 * 0.1, 300.0 + fi as f64 * 40.0, tier as f64 * 20.0,
                    ) as f32 * 3.0;
                    let end_y = start_y + end_y_offset;

                    // Check if far wall has solid at the landing point
                    let far_check_pos = Vec3::new(other_wall_x, end_y + 1.0, z);
                    let far_ck = (
                        (far_check_pos.x / eb).floor() as i32,
                        (far_check_pos.y / eb).floor() as i32,
                        (far_check_pos.z / eb).floor() as i32,
                    );
                    let far_wall_solid = density_fields.get(&far_ck).map_or(false, |d| {
                        let size = d.size;
                        let vs = eb / (size - 1) as f32;
                        let lx = ((far_check_pos.x - far_ck.0 as f32 * eb) / vs).round() as usize;
                        let ly = ((far_check_pos.y - far_ck.1 as f32 * eb) / vs).round() as usize;
                        let lz = ((far_check_pos.z - far_ck.2 as f32 * eb) / vs).round() as usize;
                        if lx < size && ly < size && lz < size {
                            d.samples[lz * size * size + ly * size + lx].density > 0.0
                        } else { false }
                    });

                    // Determine bridge completion: full, collapsed stub, or skip
                    let max_t = if far_wall_solid {
                        1.0f32 // full bridge
                    } else {
                        // No far wall — make a collapsed stub (<50%)
                        let stub = rng.gen_range(0.2f32..0.45);
                        stub
                    };

                    let cross_steps = 20u32;
                    let dx_per_step = (other_wall_x - wall_x) / cross_steps as f32;
                    let max_step = ((max_t * cross_steps as f32) as u32).min(cross_steps);

                    for step in 0..=max_step {
                        let t = step as f32 / cross_steps as f32;
                        let bridge_x = wall_x + dx_per_step * step as f32;
                        let bridge_y = start_y + (end_y - start_y) * t;
                        let z_wander = path_noise.sample(
                            t as f64 * 3.0, 400.0 + fi as f64 * 60.0, tier as f64 * 25.0,
                        ) as f32 * 2.5;
                        let bridge_z_local = z + z_wander;
                        let width_at_t = bridge_width * (0.6 + 0.4 * (1.0 - (2.0 * t - 1.0).abs()));

                        // Jagged edge at collapse point: last 15% of stub gets noise-eroded
                        let at_collapse_edge = !far_wall_solid && t > max_t * 0.85;
                        let collapse_noise = if at_collapse_edge {
                            path_noise.sample(
                                bridge_x as f64 * 0.3, bridge_y as f64 * 0.3, bridge_z_local as f64 * 0.3,
                            ) as f32
                        } else { 1.0 };
                        if at_collapse_edge && collapse_noise < 0.0 { continue; } // jagged holes

                        for &key in &overlapping {
                            if let Some(density) = density_fields.get_mut(&key) {
                                let size = density.size;
                                let vs = eb / (size - 1) as f32;
                                let origin = Vec3::new(key.0 as f32 * eb, key.1 as f32 * eb, key.2 as f32 * eb);

                                for vy in 0..size { for vx in 0..size { for vz in 0..size {
                                    let wp = origin + Vec3::new(vx as f32 * vs, vy as f32 * vs, vz as f32 * vs);
                                    if (wp.x - bridge_x).abs() > 1.5 { continue; }
                                    if (wp.z - bridge_z_local).abs() > width_at_t * 0.5 { continue; }
                                    if wp.y < bridge_y || wp.y > bridge_y + path_thickness { continue; }

                                    let idx = vz * size * size + vy * size + vx;
                                    if 0.85 > density.samples[idx].density {
                                        density.samples[idx].density = 0.85;
                                        let is_top = wp.y > bridge_y + path_thickness - 1.0;
                                        density.samples[idx].material = if is_top {
                                            Material::BlackIce
                                        } else {
                                            Material::IceSheet
                                        };
                                    }
                                }}}
                            }
                        }
                    }

                    // Landing on the other side — only if bridge completed fully
                    if far_wall_solid {
                        let landing_depth = rng.gen_range(4.0f32..7.0);
                        let landing_width = rng.gen_range(4.0f32..6.0);
                        let landing_height = rng.gen_range(3.5f32..5.0);
                        for &key in &overlapping {
                            if let Some(density) = density_fields.get_mut(&key) {
                                let size = density.size;
                                let vs = eb / (size - 1) as f32;
                                let origin = Vec3::new(key.0 as f32 * eb, key.1 as f32 * eb, key.2 as f32 * eb);

                                for vy in 0..size { for vx in 0..size { for vz in 0..size {
                                    let wp = origin + Vec3::new(vx as f32 * vs, vy as f32 * vs, vz as f32 * vs);
                                    let into_far_wall = if side < 0.0 { other_wall_x - wp.x } else { wp.x - other_wall_x };
                                    if into_far_wall < -1.0 || into_far_wall > landing_depth { continue; }
                                    if (wp.z - z).abs() > landing_width * 0.5 { continue; }
                                    if wp.y < end_y - 0.5 || wp.y > end_y + landing_height { continue; }

                                    let idx = vz * size * size + vy * size + vx;
                                    if density.samples[idx].density > 0.0 {
                                        density.samples[idx].density = -1.0;
                                        density.samples[idx].material = Material::Air;
                                    }
                                }}}
                            }
                        }
                    } // end if far_wall_solid
                } // end if near_bridge

                z += z_step;
            }
        }
    }

    // Tier-connecting tunnels: carved INTO the wall, sloping between tiers
    // 3x more frequent, deeper, wider — player travels up/down inside the ice
    let ramp_noise = Simplex3D::new(global_seed.wrapping_add(0xF155_0006));
    for (fi, &center_x) in fissure_centers_x.iter().enumerate() {
        let num_tiers_local = rng.gen_range(6u32..=8);
        let tier_spacing_local = (fissure_ceil_y - fissure_floor_y) / (num_tiers_local as f32 + 1.0);

        for tier in 0..num_tiers_local.saturating_sub(1) {
            if rng.gen_bool(0.15) { continue; } // 85% of tier pairs get a tunnel (3x more)

            let lower_y = fissure_floor_y + tier_spacing_local * (tier as f32 + 1.0);
            let upper_y = fissure_floor_y + tier_spacing_local * (tier as f32 + 2.0);
            let z_range = (fissure_max_z - fissure_min_z - eb * 0.8).max(eb * 0.5);
            let ramp_z = fissure_min_z + rng.gen_range(eb * 0.3..z_range);
            let ramp_side: f32 = if rng.gen_bool(0.5) { -1.0 } else { 1.0 };
            let ramp_wall_x = center_x + ramp_side * fissure_width * 0.5;
            let ramp_length = rng.gen_range(eb * 0.8..eb * 2.0); // longer tunnel
            let tunnel_depth = rng.gen_range(14.0f32..18.0); // +8 deeper into wall
            let tunnel_radius = rng.gen_range(5.0f32..7.0); // 2.5x headroom

            for &key in &overlapping {
                if let Some(density) = density_fields.get_mut(&key) {
                    let size = density.size;
                    let vs = eb / (size - 1) as f32;
                    let origin = Vec3::new(key.0 as f32 * eb, key.1 as f32 * eb, key.2 as f32 * eb);

                    for vz in 0..size { for vy in 0..size { for vx in 0..size {
                        let wp = origin + Vec3::new(vx as f32 * vs, vy as f32 * vs, vz as f32 * vs);
                        if wp.z < ramp_z || wp.z > ramp_z + ramp_length { continue; }

                        // Sloping Y along Z: smooth interpolation
                        let t = (wp.z - ramp_z) / ramp_length;
                        let t_smooth = t * t * (3.0 - 2.0 * t); // smoothstep for gentle slope
                        let tunnel_center_y = lower_y + (upper_y - lower_y) * t_smooth;

                        // Tunnel goes INTO the wall (negative protrusion = into wall)
                        let into_wall = if ramp_side < 0.0 { ramp_wall_x - wp.x } else { wp.x - ramp_wall_x };
                        if into_wall < -1.5 || into_wall > tunnel_depth { continue; }

                        // Rounded cross-section: elliptical
                        let tunnel_cx = tunnel_depth * 0.5;
                        let dx_norm = (into_wall - tunnel_cx) / (tunnel_depth * 0.5);
                        let dy_norm = (wp.y - tunnel_center_y) / tunnel_radius;
                        // Add noise wobble to the tunnel shape
                        let wobble = ramp_noise.sample(
                            wp.z as f64 * 0.12, wp.y as f64 * 0.1, fi as f64 * 40.0 + tier as f64 * 10.0,
                        ) as f32 * 0.3;
                        let dist_sq = dx_norm * dx_norm + dy_norm * dy_norm + wobble;

                        if dist_sq < 1.0 {
                            let idx = vz * size * size + vy * size + vx;
                            if density.samples[idx].density > 0.0 {
                                density.samples[idx].density = -1.0;
                                density.samples[idx].material = Material::Air;
                            }
                        }
                    }}}
                }
            }
        }
    }

    // Paint materials: low-frequency noise for large connected patches
    let mat_noise = Simplex3D::new(global_seed.wrapping_add(0xF155_0003));
    let mat_freq = 0.04f64; // very low freq = large blobs, not speckles

    for &key in &overlapping {
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
                        let above = if y + 1 < size { z * size * size + (y + 1) * size + x } else { idx };
                        let below = if y > 0 { z * size * size + (y - 1) * size + x } else { idx };
                        let is_floor = y + 1 < size && density.samples[above].density <= 0.0;
                        let is_ceiling = y > 0 && density.samples[below].density <= 0.0;
                        let is_wall = !is_floor && !is_ceiling && [
                            (x + 1, y, z), (x.wrapping_sub(1), y, z),
                            (x, y, z + 1), (x, y, z.wrapping_sub(1)),
                        ].iter().any(|&(nx, ny, nz)| {
                            nx < size && nz < size && {
                                let ni = nz * size * size + ny * size + nx;
                                density.samples[ni].density <= 0.0
                            }
                        });
                        let is_interior = !is_floor && !is_ceiling && !is_wall;

                        let noise_val = mat_noise.sample(
                            wp.x as f64 * mat_freq,
                            wp.y as f64 * mat_freq * 0.5,
                            wp.z as f64 * mat_freq,
                        ) as f32 * 0.5 + 0.5;

                        // Check if this voxel is on a ledge (has air below AND is horizontal-ish)
                        let below2 = if y > 1 { z * size * size + (y - 2) * size + x } else { idx };
                        let on_ledge = is_floor && y > 1 && density.samples[below2].density <= 0.0;

                        // Check if underside of a ledge (solid with air below)
                        let is_ledge_underside = is_ceiling && y + 1 < size && {
                            let above2 = z * size * size + (y + 1) * size + x;
                            density.samples[above2].density > 0.0
                        };

                        let mat = if on_ledge {
                            // Ledge tops: Hoarfrost accumulation with BlackIce patches
                            if noise_val > 0.55 { Material::BlackIce }
                            else { Material::Hoarfrost }
                        } else if is_ledge_underside {
                            // Ledge undersides: chunky IceSheet
                            Material::IceSheet
                        } else if is_floor {
                            // Fissure floor: large BlackIce pools
                            if noise_val > 0.45 { Material::BlackIce }
                            else { Material::Permafrost }
                        } else if is_ceiling {
                            // Ceiling: mostly Ice with Hoarfrost patches
                            if noise_val > 0.6 { Material::Hoarfrost }
                            else { Material::Ice }
                        } else if is_wall {
                            // Walls: IceSheet deep in walls, Ice on surface, Permafrost at edges
                            if noise_val > 0.65 { Material::Hoarfrost }
                            else if noise_val < 0.2 { Material::Permafrost }
                            else if noise_val < 0.4 { Material::IceSheet }
                            else { Material::Ice }
                        } else {
                            // Deep interior — thick wall cores
                            Material::IceSheet
                        };

                        density.samples[idx].material = mat;
                    }
                }
            }
        }
    }

    // Spawn stalagmites on ledge bulge sections (upward ice spikes)
    // Collected during path generation, written here
    // stalagmite_params: (base_pos, length, radius)
    for (anchor, length, radius) in &stalagmite_params {
        shapes::write_cone(
            density_fields, *anchor, *length, *radius, 1.0, // +1 = growing UP
            Material::IceSheet, 1.8, effective_bounds,
        );
        // 50% get FrozenGlow tips
        if rng.gen_bool(0.5) {
            let tip_pos = *anchor + Vec3::new(0.0, *length - 1.0, 0.0);
            let tip_len = 1.0f32.min(*length * 0.25);
            shapes::write_cone(
                density_fields, tip_pos, tip_len, radius * 0.3, 1.0,
                Material::FrozenGlow, 2.5, effective_bounds,
            );
        }
    }

    // Spawn icicles: ceiling + under paths/bridges/overhangs
    let mut icicle_params: Vec<(Vec3, f32, f32)> = Vec::new();

    // Ceiling icicles — 4x density
    for &center_x in &fissure_centers_x {
        let mut z_pos = fissure_min_z + rng.gen_range(0.5..1.5);
        while z_pos < fissure_max_z {
            // Multiple icicles per Z position across the fissure width
            for _ in 0..rng.gen_range(1u32..=3) {
                if rng.gen::<f32>() < 0.8 {
                    let ix = center_x + rng.gen_range(-fissure_width * 0.45..fissure_width * 0.45);
                    let iy = fissure_ceil_y - rng.gen_range(0.0..eb * 0.1);
                    let length = rng.gen_range(3.0..14.0);
                    let radius = rng.gen_range(0.4..2.0);
                    icicle_params.push((Vec3::new(ix, iy, z_pos), length, radius));
                }
            }
            z_pos += rng.gen_range(0.5..1.2); // +70% more icicles
        }
    }

    // Under-path/bridge icicles: scan for solid voxels with air below (overhangs)
    for &key in &overlapping {
        if let Some(density) = density_fields.get(&key) {
            let size = density.size;
            let vs = eb / (size - 1) as f32;
            let origin = Vec3::new(key.0 as f32 * eb, key.1 as f32 * eb, key.2 as f32 * eb);
            for z in (0..size).step_by(3) {
                for x in (0..size).step_by(3) {
                    for y in 1..size {
                        let idx = z * size * size + y * size + x;
                        let below = z * size * size + (y - 1) * size + x;
                        if density.samples[idx].density > 0.0 && density.samples[below].density <= 0.0 {
                            // This is an overhang — solid with air below
                            let mat = density.samples[idx].material;
                            if mat == Material::Ice || mat == Material::IceSheet || mat == Material::Hoarfrost {
                                if rng.gen::<f32>() < 0.85 { // +70% more overhang icicles
                                    let wp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                    let length = rng.gen_range(2.0..6.0);
                                    let radius = rng.gen_range(0.3..0.8);
                                    icicle_params.push((wp, length, radius));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Write icicles: IceSheet body, 50% get FrozenGlow tips
    for (i, (anchor, length, radius)) in icicle_params.iter().enumerate() {
        shapes::write_cone(
            density_fields, *anchor, *length, *radius, -1.0,
            Material::IceSheet, 2.0, effective_bounds,
        );
        // Only half get glowing tips
        if i % 2 == 0 {
            let tip_pos = *anchor + Vec3::new(0.0, -(*length - 1.5), 0.0);
            let tip_len = 1.5f32.min(*length * 0.3);
            shapes::write_cone(
                density_fields, tip_pos, tip_len, radius * 0.4, -1.0,
                Material::FrozenGlow, 2.5, effective_bounds,
            );
        }
    }

    // Organic end walls: add noise to the Z boundaries so they're not flat
    let end_noise = Simplex3D::new(global_seed.wrapping_add(0xF155_0005));
    for &key in &overlapping {
        if let Some(density) = density_fields.get_mut(&key) {
            let size = density.size;
            let vs = eb / (size - 1) as f32;
            let origin = Vec3::new(key.0 as f32 * eb, key.1 as f32 * eb, key.2 as f32 * eb);
            for z in 0..size { for y in 0..size { for x in 0..size {
                let wp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                // Near Z min/max boundaries: add noise-driven solid bumps
                let near_z_min = wp.z - world_min.z;
                let near_z_max = world_max.z - wp.z;
                let near_y_min = wp.y - world_min.y;
                if near_z_min < eb * 0.5 || near_z_max < eb * 0.5 || near_y_min < eb * 0.3 {
                    let n = end_noise.sample(
                        wp.x as f64 * 0.08, wp.y as f64 * 0.08, wp.z as f64 * 0.08,
                    ) as f32;
                    let idx = z * size * size + y * size + x;
                    // Add bumpy solid near boundaries where there's air
                    if density.samples[idx].density <= 0.0 && n > 0.2 {
                        let boundary_dist = near_z_min.min(near_z_max).min(near_y_min);
                        if boundary_dist < eb * 0.3 {
                            density.samples[idx].density = 0.6;
                            density.samples[idx].material = Material::Ice;
                        }
                    }
                }
            }}}
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
