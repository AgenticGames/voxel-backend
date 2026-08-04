//! Cavern Zone system — large-scale themed underground zones.
//!
//! Zones detect large air volumes after worm carving and reshape them into
//! themed areas (Cathedral Cavern, Subterranean Lake, Lava Tube Gallery, etc.).
//! They take priority over the smaller formation system, which is banned from
//! generating within zone bounds.

pub mod detect;
pub mod shapes;
pub mod cathedral;
pub mod lake;
pub mod canyon;
pub mod lava_gallery;
pub mod bioluminescent;
pub mod terraces;
pub mod frozen;
pub mod mega_blueprint;
pub mod mega_apply;

use std::collections::HashMap;

use glam::Vec3;
use rand::{SeedableRng, Rng};
use serde::{Deserialize, Serialize};
use voxel_core::material::Material;

use crate::config::{ZoneConfig, ZoneType};
use crate::density::DensityField;
use crate::pools::{FluidSeed, PoolFluid};
use crate::worm::path::WormSegment;

/// Convert a world-space position to a FluidSeed (chunk-local coordinates).
pub fn world_to_fluid_seed(wx: f32, wy: f32, wz: f32, effective_bounds: f32, chunk_size: usize, is_lava: bool) -> FluidSeed {
    let cx = (wx / effective_bounds).floor() as i32;
    let cy = (wy / effective_bounds).floor() as i32;
    let cz = (wz / effective_bounds).floor() as i32;
    let vs = effective_bounds / chunk_size as f32;
    let lx = ((wx - cx as f32 * effective_bounds) / vs).floor().max(0.0).min((chunk_size - 1) as f32) as u8;
    let ly = ((wy - cy as f32 * effective_bounds) / vs).floor().max(0.0).min((chunk_size - 1) as f32) as u8;
    let lz = ((wz - cz as f32 * effective_bounds) / vs).floor().max(0.0).min((chunk_size - 1) as f32) as u8;
    FluidSeed {
        chunk: (cx, cy, cz),
        lx, ly, lz,
        fluid_type: if is_lava { PoolFluid::Lava } else { PoolFluid::Water },
        is_source: true,
        // Zone fluid bodies stay unbounded for now — they are large deliberate
        // lakes, and bounding them is a separate (riskier) decision from the
        // 2026-08-04 pool-containment bundle.
        max_flow_dist: 0,
    }
}

/// Bounding box of a placed zone, used to exclude formations/pools.
#[derive(Debug, Clone)]
pub struct ZoneBounds {
    pub world_min: Vec3,
    pub world_max: Vec3,
    pub zone_type: ZoneType,
}

impl ZoneBounds {
    /// Check if a world-space point falls inside this zone's AABB.
    pub fn contains(&self, pos: Vec3) -> bool {
        pos.x >= self.world_min.x && pos.x <= self.world_max.x
            && pos.y >= self.world_min.y && pos.y <= self.world_max.y
            && pos.z >= self.world_min.z && pos.z <= self.world_max.z
    }
}

/// Check if a world-space point is inside any zone (except Bioluminescent, which allows formations).
pub fn is_in_exclusion_zone(pos: Vec3, zone_bounds: &[ZoneBounds]) -> bool {
    zone_bounds.iter().any(|z| {
        z.zone_type != ZoneType::BioluminescentGrotto && z.contains(pos)
    })
}

/// Anchor point for UE rendering (bioluminescent lights, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoneAnchor {
    pub px: f32, pub py: f32, pub pz: f32,
    pub nx: f32, pub ny: f32, pub nz: f32,
}

/// Descriptor for a placed zone, returned to caller for UE/viewer consumption.
#[derive(Debug, Clone)]
pub struct ZoneDescriptor {
    pub zone_type: ZoneType,
    pub world_min: Vec3,
    pub world_max: Vec3,
    pub center: Vec3,
    pub anchors: Vec<ZoneAnchor>,
}

/// Main entry point: detect and place zones across a region's density fields.
///
/// Called after worm carving + lava tubes + rivers, before pools and formations.
/// Returns placed zone descriptors, zone exclusion bounds, and any fluid seeds.
pub fn place_zones(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    config: &ZoneConfig,
    global_seed: u64,
    effective_bounds: f32,
    _worm_paths: &[Vec<WormSegment>],
) -> (Vec<ZoneDescriptor>, Vec<ZoneBounds>, Vec<FluidSeed>) {
    if !config.enabled {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    use std::time::Instant;
    let zones_total_start = Instant::now();

    let mut descriptors = Vec::new();
    let mut bounds = Vec::new();
    let mut fluid_seeds = Vec::new();

    // Phase 0: Frozen Mega-Vault via Blueprint + Per-Chunk-Apply pattern.
    // The blueprint pre-computes all vault geometry parametrically (~2ms),
    // then each overlapping chunk gets a single-pass apply (~O(17^3) per chunk).
    let t_blueprint_start = Instant::now();
    {
        let eb = effective_bounds;
        // Use cached blueprint (OnceLock) — generates once, reused for all regions
        let bp_opt = mega_blueprint::get_or_create_blueprint(
            config.frozen_mega_chance, global_seed, eb,
        );
        if let Some(bp) = bp_opt {
            // Apply to overlapping chunks IN PARALLEL
            // Extract overlapping density fields, apply via rayon, put back
            let overlapping_keys: Vec<(i32, i32, i32)> = density_fields.keys()
                .filter(|&&k| bp.overlaps_chunk(k))
                .copied()
                .collect();

            // Extract: remove from HashMap so we can mutably process in parallel
            let mut extracted: Vec<((i32, i32, i32), DensityField)> = overlapping_keys.iter()
                .filter_map(|k| density_fields.remove(k).map(|d| (*k, d)))
                .collect();

            // Parallel apply — limit to half CPU cores to avoid starving OS/audio/UE
            // Use global rayon pool (no custom thread pool creation per-call)
            use rayon::prelude::*;
            extracted.par_iter_mut().for_each(|(key, density)| {
                mega_apply::apply_vault_to_chunk(density, *key, bp, eb);
            });

            // Put back
            for (key, density) in extracted {
                density_fields.insert(key, density);
            }

            // Generate entrance tunnels: scan outward from vault faces to find caves
            if !overlapping_keys.is_empty() {
                let mut entrance_chains: Vec<mega_blueprint::TunnelWaypointChain> = Vec::new();
                let entrance_radius = 6.0f32;
                let max_search = eb * 8.0; // search up to 8 chunks outward
                let mut e_rng = rand_chacha::ChaCha8Rng::seed_from_u64(
                    global_seed.wrapping_add(0xE0E0_0001));

                // 6 search directions: ±X (fissure long walls), ±Z (tapered ends), -Y (floor), +Y (ceiling)
                let search_dirs: Vec<(Vec3, Vec3)> = vec![
                    // (start_face_center, direction)
                    (Vec3::new(bp.world_min.x, bp.world_center.y, bp.world_center.z), Vec3::new(-1.0, 0.0, 0.0)),
                    (Vec3::new(bp.world_max.x, bp.world_center.y, bp.world_center.z), Vec3::new(1.0, 0.0, 0.0)),
                    (Vec3::new(bp.world_center.x, bp.world_center.y, bp.world_min.z), Vec3::new(0.0, 0.0, -1.0)),
                    (Vec3::new(bp.world_center.x, bp.world_center.y, bp.world_max.z), Vec3::new(0.0, 0.0, 1.0)),
                    (Vec3::new(bp.world_center.x, bp.world_min.y, bp.world_center.z), Vec3::new(0.0, -1.0, 0.0)), // floor
                    (Vec3::new(bp.world_center.x, bp.world_max.y * 0.7, bp.world_center.z), Vec3::new(0.0, 1.0, 0.0)), // above
                ];

                for (face_center, search_dir) in &search_dirs {
                    // Search outward for cave air
                    let step_size = 2.0f32;
                    let mut best_air_pos: Option<Vec3> = None;
                    let mut best_air_size = 0u32;

                    for step in 1..(max_search / step_size) as i32 {
                        let probe = *face_center + *search_dir * step as f32 * step_size;
                        let ck = ((probe.x / eb).floor() as i32, (probe.y / eb).floor() as i32, (probe.z / eb).floor() as i32);

                        if let Some(density) = density_fields.get(&ck) {
                            let s = density.size;
                            let vs_local = eb / (s - 1) as f32;
                            let local = probe - Vec3::new(ck.0 as f32 * eb, ck.1 as f32 * eb, ck.2 as f32 * eb);
                            let gx = (local.x / vs_local).round().clamp(0.0, (s - 1) as f32) as usize;
                            let gy = (local.y / vs_local).round().clamp(0.0, (s - 1) as f32) as usize;
                            let gz = (local.z / vs_local).round().clamp(0.0, (s - 1) as f32) as usize;
                            let gi = gz * s * s + gy * s + gx;

                            if gi < density.samples.len() && density.samples[gi].density <= 0.0 {
                                // Found air — count nearby air for "size"
                                let mut air_count = 0u32;
                                for dx in -2i32..=2 {
                                    for dy in -2i32..=2 {
                                        for dz in -2i32..=2 {
                                            let nx = gx as i32 + dx;
                                            let ny = gy as i32 + dy;
                                            let nz = gz as i32 + dz;
                                            if nx >= 0 && nx < s as i32 && ny >= 0 && ny < s as i32 && nz >= 0 && nz < s as i32 {
                                                let ni = nz as usize * s * s + ny as usize * s + nx as usize;
                                                if density.samples[ni].density <= 0.0 { air_count += 1; }
                                            }
                                        }
                                    }
                                }
                                if air_count > best_air_size {
                                    best_air_size = air_count;
                                    best_air_pos = Some(probe);
                                }
                                if air_count > 30 { break; } // found a big cave, stop searching
                            }
                        }
                    }

                    // Build entrance tunnel from vault face to cave
                    if let Some(cave_pos) = best_air_pos {
                        let tunnel_start = *face_center;
                        let tunnel_end = cave_pos;
                        let tunnel_vec = tunnel_end - tunnel_start;
                        let tunnel_len = tunnel_vec.length();
                        if tunnel_len < 3.0 { continue; }

                        let steps = (tunnel_len / 3.0).ceil() as u32;
                        let mut chain_wps = Vec::new();

                        for si in 0..=steps {
                            let t = si as f32 / steps as f32;
                            let pos = tunnel_start + tunnel_vec * t;
                            // Slight wander
                            let wobble_y = ((t * 5.0) as f64).sin() as f32 * 2.0;
                            let wobble_perp = ((t * 3.0) as f64).cos() as f32 * 1.5;
                            chain_wps.push(pos + Vec3::new(
                                if search_dir.x.abs() < 0.5 { wobble_perp } else { 0.0 },
                                wobble_y,
                                if search_dir.z.abs() < 0.5 { wobble_perp } else { 0.0 },
                            ));
                        }

                        // Flared exit mouth: big opening where it meets the cave
                        let exit_center = tunnel_end;
                        for angle in 0..10 {
                            let a = angle as f32 * 0.628;
                            let flare_r = entrance_radius * 3.0; // huge mouth
                            let perp1 = if search_dir.y.abs() > 0.5 {
                                Vec3::new(a.cos() * flare_r, 0.0, a.sin() * flare_r)
                            } else if search_dir.x.abs() > 0.5 {
                                Vec3::new(0.0, a.sin() * flare_r, a.cos() * flare_r)
                            } else {
                                Vec3::new(a.cos() * flare_r, a.sin() * flare_r, 0.0)
                            };
                            chain_wps.push(exit_center + perp1);
                        }

                        // Branch: 2-3 sub-tunnels if cave is big enough
                        if best_air_size > 15 {
                            let num_branches = e_rng.gen_range(2u32..=3);
                            for _ in 0..num_branches {
                                let branch_dir = Vec3::new(
                                    e_rng.gen_range(-1.0f32..1.0),
                                    e_rng.gen_range(-0.3f32..0.3),
                                    e_rng.gen_range(-1.0f32..1.0),
                                ).normalize();
                                for bsi in 1..=5 {
                                    let bpos = cave_pos + branch_dir * bsi as f32 * 4.0;
                                    chain_wps.push(bpos);
                                }
                            }
                        }

                        entrance_chains.push(mega_blueprint::TunnelWaypointChain {
                            waypoints: chain_wps,
                            radius: entrance_radius,
                            is_blocked: false,
                            source_path: None,
                            priority: 2, // highest — never discard
                            chambers: Vec::new(),
                        });
                    }
                }

                // Apply entrance chains: carve sphere chains + spread ice materials
                let all_keys: Vec<(i32,i32,i32)> = density_fields.keys().copied().collect();
                for chain in &entrance_chains {
                    for &key in &all_keys {
                        if let Some(density) = density_fields.get_mut(&key) {
                            let chunk_origin = Vec3::new(key.0 as f32 * eb, key.1 as f32 * eb, key.2 as f32 * eb);
                            let chunk_max = chunk_origin + Vec3::splat(eb + 20.0);
                            let chunk_min = chunk_origin - Vec3::splat(20.0);

                            // Quick AABB check
                            let any_near = chain.waypoints.iter().any(|wp| {
                                wp.x >= chunk_min.x && wp.x <= chunk_max.x
                                && wp.y >= chunk_min.y && wp.y <= chunk_max.y
                                && wp.z >= chunk_min.z && wp.z <= chunk_max.z
                            });
                            if !any_near { continue; }

                            let s = density.size;
                            let vs_local = eb / (s - 1) as f32;

                            for (wi, wp) in chain.waypoints.iter().enumerate() {
                                let wobble = ((wi as f32 * 0.7).sin() * 0.3 + 1.0) * chain.radius;
                                let local = *wp - chunk_origin;
                                let gc = Vec3::new(local.x / vs_local, local.y / vs_local, local.z / vs_local);
                                let gr = (wobble / vs_local).ceil() as i32 + 1;
                                let lo_x = (gc.x as i32 - gr).max(0) as usize;
                                let hi_x = ((gc.x as i32 + gr) as usize).min(s - 1);
                                let lo_y = (gc.y as i32 - gr).max(0) as usize;
                                let hi_y = ((gc.y as i32 + gr) as usize).min(s - 1);
                                let lo_z = (gc.z as i32 - gr).max(0) as usize;
                                let hi_z = ((gc.z as i32 + gr) as usize).min(s - 1);

                                for z in lo_z..=hi_z { for y in lo_y..=hi_y { for x in lo_x..=hi_x {
                                    let vwp = chunk_origin + Vec3::new(x as f32 * vs_local, y as f32 * vs_local, z as f32 * vs_local);
                                    let dist = (vwp - *wp).length();
                                    if dist < wobble {
                                        let idx = z * s * s + y * s + x;
                                        if density.samples[idx].density > 0.0 {
                                            density.samples[idx].density = -1.0;
                                            density.samples[idx].material = Material::Air;
                                        }
                                    }
                                    // Spread ice materials on walls near the entrance
                                    if dist < wobble * 1.5 && dist >= wobble {
                                        let idx = z * s * s + y * s + x;
                                        if density.samples[idx].density > 0.0
                                            && density.samples[idx].material.is_host_rock()
                                        {
                                            density.samples[idx].material = Material::Permafrost;
                                        }
                                    }
                                }}}
                            }
                        }
                    }
                }
            }

            if !overlapping_keys.is_empty() {
                let desc = ZoneDescriptor {
                    zone_type: ZoneType::FrozenGrotto,
                    world_min: bp.world_min,
                    world_max: bp.world_max,
                    center: bp.world_center,
                    anchors: Vec::new(),
                };
                let zone_bounds = ZoneBounds {
                    world_min: bp.world_min,
                    world_max: bp.world_max,
                    zone_type: ZoneType::FrozenGrotto,
                };
                descriptors.push(desc);
                bounds.push(zone_bounds);
            }
        }
    }

    let t_vault = t_blueprint_start.elapsed();
    let t_air_start = Instant::now();

    // Step 1: Compute per-chunk air statistics
    let air_stats = detect::compute_air_stats(density_fields, effective_bounds);
    let t_air = t_air_start.elapsed();

    let t_cluster_start = Instant::now();
    // Step 2: Cluster into CavernVolumes
    let volumes = detect::cluster_cavern_volumes(&air_stats, effective_bounds, 64);
    let t_cluster = t_cluster_start.elapsed();

    let t_zonegen_start = Instant::now();
    let mut zone_gen_count = 0u32;
    // Step 3: Select zone types and generate — skip volumes inside existing zone bounds
    for volume in &volumes {
        // Don't place zones that overlap the mega-vault or other existing zones
        let overlaps_existing = bounds.iter().any(|b| b.contains(volume.world_center));
        if overlaps_existing { continue; }

        let zone_type = detect::select_zone_type(volume, config, global_seed);
        if let Some(zt) = zone_type {
            zone_gen_count += 1;
            let (desc, zone_bounds, seeds) = generate_zone(
                zt, volume, density_fields, config, global_seed, effective_bounds,
            );
            descriptors.push(desc);
            bounds.push(zone_bounds);
            fluid_seeds.extend(seeds);
        }
    }

    let t_zonegen = t_zonegen_start.elapsed();
    let zones_total = zones_total_start.elapsed();

    // Detailed zones timing log
    {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true)
            .open("D:/Unreal Projects/Mithril2026/Saved/zones_perf.txt")
        {
            let _ = writeln!(f, "place_zones total={:.1}ms | vault_bp+apply={:.1} air_stats={:.1} clustering={:.1} zone_gen={:.1} ({}zones {} volumes) chunks={}",
                zones_total.as_secs_f64() * 1000.0,
                t_vault.as_secs_f64() * 1000.0,
                t_air.as_secs_f64() * 1000.0,
                t_cluster.as_secs_f64() * 1000.0,
                t_zonegen.as_secs_f64() * 1000.0,
                zone_gen_count,
                volumes.len(),
                density_fields.len(),
            );
        }
    }

    (descriptors, bounds, fluid_seeds)
}

/// Dispatch to zone-specific generation.
fn generate_zone(
    zone_type: ZoneType,
    volume: &detect::CavernVolume,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    config: &ZoneConfig,
    global_seed: u64,
    effective_bounds: f32,
) -> (ZoneDescriptor, ZoneBounds, Vec<FluidSeed>) {
    let zone_bounds = ZoneBounds {
        world_min: volume.world_bbox_min,
        world_max: volume.world_bbox_max,
        zone_type,
    };

    let (anchors, fluid_seeds) = match zone_type {
        ZoneType::Cathedral => cathedral::generate(volume, density_fields, config, global_seed, effective_bounds),
        ZoneType::SubterraneanLake => lake::generate(volume, density_fields, config, global_seed, effective_bounds),
        ZoneType::RiverCanyon => canyon::generate(volume, density_fields, config, global_seed, effective_bounds),
        ZoneType::LavaTubeGallery => lava_gallery::generate(volume, density_fields, config, global_seed, effective_bounds),
        ZoneType::BioluminescentGrotto => bioluminescent::generate(volume, density_fields, config, global_seed, effective_bounds),
        ZoneType::GeothermalTerraces => terraces::generate(volume, density_fields, config, global_seed, effective_bounds),
        ZoneType::FrozenGrotto => frozen::generate(volume, density_fields, config, global_seed, effective_bounds),
    };

    let descriptor = ZoneDescriptor {
        zone_type,
        world_min: volume.world_bbox_min,
        world_max: volume.world_bbox_max,
        center: volume.world_center,
        anchors,
    };

    (descriptor, zone_bounds, fluid_seeds)
}
