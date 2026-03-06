//! Underground river generator — carves wide, flat passages through limestone
//! following the water table surface. Rivers are seeded at "swallow holes" where
//! the water table intersects the limestone layer ceiling.

use glam::Vec3;
use voxel_core::density::DensityField;
use voxel_core::material::Material;
use voxel_noise::simplex::Simplex3D;
use voxel_noise::NoiseSource;

use crate::config::{RiverConfig, WaterTableConfig};
use crate::density::water_table_y_at;
use crate::seed;
use crate::springs::SpringDescriptor;
use crate::springs::SpringType;

/// Segment of an underground river path.
#[derive(Debug, Clone)]
struct RiverSegment {
    position: Vec3,
    width: f32,
    height: f32,
}

/// Carve underground rivers into the density field and return water source descriptors.
pub fn carve_rivers(
    density: &mut DensityField,
    config: &RiverConfig,
    water_config: &WaterTableConfig,
    world_origin: (f64, f64, f64),
    global_seed: u64,
    chunk_seed: u64,
) -> Vec<SpringDescriptor> {
    if !config.enabled {
        return Vec::new();
    }

    let grid_size = density.size;
    let chunk_size = grid_size - 1;

    // Region-deterministic seed
    let rx = (world_origin.0 / (chunk_size as f64 * 4.0)).floor() as i32;
    let ry = (world_origin.1 / (chunk_size as f64 * 4.0)).floor() as i32;
    let rz = (world_origin.2 / (chunk_size as f64 * 4.0)).floor() as i32;
    let r_seed = seed::region_seed(global_seed, rx, ry, rz);

    // Determine river count for this region
    let river_rng = Simplex3D::new(r_seed.wrapping_add(850));
    let rng_val = river_rng.sample(rx as f64 * 0.1, ry as f64 * 0.1, rz as f64 * 0.1) * 0.5 + 0.5;
    let river_count = (config.rivers_per_region as f64 * rng_val * 2.0).floor() as u32;

    if river_count == 0 {
        return Vec::new();
    }

    let mut water_sources = Vec::new();
    let _ = chunk_seed; // available for future per-chunk variation

    for r in 0..river_count {
        let r_s = r_seed.wrapping_add(r as u64 * 5000 + 860);
        let pos_noise = Simplex3D::new(r_s);

        // Seed position — in limestone layer, near water table
        let region_x = rx as f64 * chunk_size as f64 * 4.0;
        let region_z = rz as f64 * chunk_size as f64 * 4.0;
        let region_extent = chunk_size as f64 * 4.0;

        let sx = region_x
            + (pos_noise.sample(r as f64, 0.0, 0.0) * 0.5 + 0.5) * region_extent;
        let sz = region_z
            + (pos_noise.sample(0.0, 0.0, r as f64) * 0.5 + 0.5) * region_extent;

        // Y follows water table with slight offset into limestone
        let sy = if water_config.enabled {
            water_table_y_at(sx, sz, water_config, global_seed) - 2.0
        } else {
            170.0 // fallback: limestone layer
        };

        // Layer restriction: only in limestone range (160-200 default)
        if config.layer_restriction && (sy < 155.0 || sy > 210.0) {
            continue;
        }

        let segments = generate_river_path(
            r_s,
            Vec3::new(sx as f32, sy as f32, sz as f32),
            config,
            water_config,
            global_seed,
        );

        // Carve each segment
        for seg in &segments {
            let local_x = seg.position.x as f64 - world_origin.0;
            let local_y = seg.position.y as f64 - world_origin.1;
            let local_z = seg.position.z as f64 - world_origin.2;

            let half_w = seg.width / 2.0;
            let half_h = seg.height / 2.0;
            let max_r = half_w.max(half_h);

            let min_x = ((local_x - max_r as f64).floor() as i32).max(0);
            let max_x = ((local_x + max_r as f64).ceil() as i32).min(grid_size as i32 - 1);
            let min_y = ((local_y - max_r as f64).floor() as i32).max(0);
            let max_y = ((local_y + max_r as f64).ceil() as i32).min(grid_size as i32 - 1);
            let min_z = ((local_z - max_r as f64).floor() as i32).max(0);
            let max_z = ((local_z + max_r as f64).ceil() as i32).min(grid_size as i32 - 1);

            for gz in min_z..=max_z {
                for gy in min_y..=max_y {
                    for gx in min_x..=max_x {
                        let dx = gx as f64 - local_x;
                        let dy = gy as f64 - local_y;
                        let dz = gz as f64 - local_z;

                        // Elliptical cross-section (wider than tall)
                        let ellipse_dist = (dx / half_w as f64).powi(2)
                            + (dy / half_h as f64).powi(2)
                            + (dz / half_w as f64).powi(2);

                        if ellipse_dist < 1.0 {
                            let vs = density.get_mut(gx as usize, gy as usize, gz as usize);
                            vs.density = -1.0;
                            vs.material = Material::Air;
                        }
                    }
                }
            }

            // Place water source at floor of carved passage
            let floor_y = (local_y - half_h as f64).round() as i32;
            let cx = (local_x.round() as i32).clamp(0, grid_size as i32 - 1) as usize;
            let cy = floor_y.clamp(0, grid_size as i32 - 1) as usize;
            let cz = (local_z.round() as i32).clamp(0, grid_size as i32 - 1) as usize;

            if cx < chunk_size && cy < chunk_size && cz < chunk_size {
                if density.get(cx, cy, cz).density < 0.0 {
                    water_sources.push(SpringDescriptor {
                        lx: cx as u8,
                        ly: cy as u8,
                        lz: cz as u8,
                        level: 0.6,
                        source_type: SpringType::RiverSource,
                    });
                }
            }
        }
    }

    water_sources
}

/// Generate a river path — wide gentle meanders following the water table.
fn generate_river_path(
    seed: u64,
    start: Vec3,
    config: &RiverConfig,
    water_config: &WaterTableConfig,
    global_seed: u64,
) -> Vec<RiverSegment> {
    let yaw_noise = Simplex3D::new(seed.wrapping_add(1));
    let width_noise = Simplex3D::new(seed.wrapping_add(3));

    let mut segments = Vec::with_capacity(config.max_steps as usize);
    let mut pos = start;
    let mut yaw: f32 = yaw_noise.sample(0.0, 0.0, (seed % 10000) as f64 * 0.01) as f32 * std::f32::consts::PI;

    for step in 0..config.max_steps {
        let t = step as f64 / config.max_steps as f64;

        // Taper at endpoints
        let taper = if t < 0.15 {
            t / 0.15
        } else if t > 0.85 {
            (1.0 - t) / 0.15
        } else {
            1.0
        };
        let taper = taper.max(0.25) as f32;

        let w_val = width_noise.sample(pos.x as f64 * 0.03, pos.z as f64 * 0.03, 0.0) as f32;
        let width = (config.width_min + (config.width_max - config.width_min) * (w_val * 0.5 + 0.5)) * taper;

        segments.push(RiverSegment {
            position: pos,
            width,
            height: config.height * taper,
        });

        // Gentle yaw meander — wider turns than worms
        let yaw_delta = yaw_noise.sample(t * 2.0, pos.z as f64 * 0.05, 0.0) as f32;
        yaw += yaw_delta * 0.15;

        // Y follows water table with gentle downslope
        let target_y = if water_config.enabled {
            water_table_y_at(
                pos.x as f64 + yaw.cos() as f64 * config.step_length as f64,
                pos.z as f64 + yaw.sin() as f64 * config.step_length as f64,
                water_config,
                global_seed,
            ) as f32
                - 2.0
        } else {
            pos.y - config.downslope_bias as f32
        };

        let pitch = ((target_y - pos.y) / config.step_length).clamp(-0.087, 0.087); // ±5°

        let dir = Vec3::new(
            yaw.cos() * (1.0 - pitch.abs()),
            pitch,
            yaw.sin() * (1.0 - pitch.abs()),
        )
        .normalize();

        pos += dir * config.step_length;
    }

    segments
}
