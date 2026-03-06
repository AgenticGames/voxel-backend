//! Lava tube generator — carves basalt-lined tunnels at depth, some connected to
//! kimberlite pipes, some independent. Active tubes below a threshold depth are
//! lava-filled; drained tubes above that depth are empty traversable tunnels.

use glam::Vec3;
use voxel_core::density::DensityField;
use voxel_core::material::Material;
use voxel_noise::simplex::Simplex3D;
use voxel_noise::NoiseSource;

use crate::config::LavaTubeConfig;
use crate::seed;
use crate::springs::LavaDescriptor;

/// Segment of a lava tube path.
#[derive(Debug, Clone)]
pub struct TubeSegment {
    pub position: Vec3,
    pub radius: f32,
}

/// Carve lava tubes into the density field and return lava source descriptors
/// for active (lava-filled) segments.
pub fn carve_lava_tubes(
    density: &mut DensityField,
    config: &LavaTubeConfig,
    world_origin: (f64, f64, f64),
    global_seed: u64,
    chunk_seed: u64,
) -> Vec<LavaDescriptor> {
    if !config.enabled {
        return Vec::new();
    }

    let grid_size = density.size;
    let chunk_size = grid_size - 1;

    // Region-deterministic seed (same approach as worms)
    let rx = (world_origin.0 / (chunk_size as f64 * 4.0)).floor() as i32;
    let ry = (world_origin.1 / (chunk_size as f64 * 4.0)).floor() as i32;
    let rz = (world_origin.2 / (chunk_size as f64 * 4.0)).floor() as i32;
    let r_seed = seed::region_seed(global_seed, rx, ry, rz);

    // Determine how many tubes in this region
    let tube_rng = Simplex3D::new(r_seed.wrapping_add(800));
    let rng_val = tube_rng.sample(rx as f64 * 0.1, ry as f64 * 0.1, rz as f64 * 0.1) * 0.5 + 0.5;
    let tube_count = (config.tubes_per_region as f64 * rng_val * 2.0).floor() as u32;

    if tube_count == 0 {
        return Vec::new();
    }

    let mut lava_descriptors = Vec::new();

    for t in 0..tube_count {
        let t_seed = r_seed.wrapping_add(t as u64 * 3000 + 900);
        let pos_noise = Simplex3D::new(t_seed);

        // Determine tube start position in world coordinates
        let region_x = rx as f64 * chunk_size as f64 * 4.0;
        let region_z = rz as f64 * chunk_size as f64 * 4.0;
        let region_extent = chunk_size as f64 * 4.0;

        let sx = region_x
            + (pos_noise.sample(t as f64, 0.0, 0.0) * 0.5 + 0.5) * region_extent;
        let sy = config.depth_min
            + (pos_noise.sample(0.0, t as f64, 0.0) * 0.5 + 0.5)
                * (config.depth_max - config.depth_min);
        let sz = region_z
            + (pos_noise.sample(0.0, 0.0, t as f64) * 0.5 + 0.5) * region_extent;

        // Generate tube path
        let segments = generate_tube_path(
            t_seed,
            Vec3::new(sx as f32, sy as f32, sz as f32),
            config.step_length,
            config.max_steps,
            config.radius_min,
            config.radius_max,
        );

        // Carve each segment into the density field
        for seg in &segments {
            let local_x = seg.position.x as f64 - world_origin.0;
            let local_y = seg.position.y as f64 - world_origin.1;
            let local_z = seg.position.z as f64 - world_origin.2;

            // Oval cross-section: wider than tall
            let rx_radius = seg.radius * 1.3;
            let ry_radius = seg.radius * 0.7;
            let rz_radius = seg.radius * 1.3;
            let max_r = rx_radius.max(ry_radius).max(rz_radius);

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

                        // Ellipsoidal distance
                        let ellipse_dist = (dx / rx_radius as f64).powi(2)
                            + (dy / ry_radius as f64).powi(2)
                            + (dz / rz_radius as f64).powi(2);

                        if ellipse_dist < 0.7 {
                            // Interior → Air
                            let vs = density.get_mut(gx as usize, gy as usize, gz as usize);
                            vs.density = -1.0;
                            vs.material = Material::Air;
                        } else if ellipse_dist < 1.0 {
                            // Boundary → Basalt (tube wall is cooled lava)
                            let vs = density.get_mut(gx as usize, gy as usize, gz as usize);
                            if vs.density > 0.0 {
                                vs.material = Material::Basalt;
                            }
                        }
                    }
                }
            }

            // Active tube check: below active_depth → lava source candidates
            let wy = seg.position.y as f64;
            let is_active = wy < config.active_depth;

            if is_active {
                // Place lava in carved air at this segment
                let cx = (local_x.round() as i32).clamp(0, grid_size as i32 - 1) as usize;
                let cy = (local_y.round() as i32).clamp(0, grid_size as i32 - 1) as usize;
                let cz = (local_z.round() as i32).clamp(0, grid_size as i32 - 1) as usize;

                if cx < chunk_size && cy < chunk_size && cz < chunk_size {
                    if density.get(cx, cy, cz).density < 0.0 {
                        lava_descriptors.push(LavaDescriptor {
                            lx: cx as u8,
                            ly: cy as u8,
                            lz: cz as u8,
                            level: 1.0,
                        });
                    }
                }
            }
        }
    }

    lava_descriptors
}

/// Generate a lava tube path — roughly horizontal with gentle downslope bias.
fn generate_tube_path(
    seed: u64,
    start: Vec3,
    step_length: f32,
    max_steps: u32,
    radius_min: f32,
    radius_max: f32,
) -> Vec<TubeSegment> {
    let yaw_noise = Simplex3D::new(seed.wrapping_add(1));
    let pitch_noise = Simplex3D::new(seed.wrapping_add(2));
    let radius_noise = Simplex3D::new(seed.wrapping_add(3));

    let mut segments = Vec::with_capacity(max_steps as usize);
    let mut pos = start;
    let mut yaw: f32 = yaw_noise.sample(0.0, 0.0, (seed % 10000) as f64 * 0.01) as f32 * std::f32::consts::PI;
    let mut pitch: f32 = 0.0;

    for step in 0..max_steps {
        let t = step as f64 / max_steps as f64;

        // Taper radius at endpoints
        let taper = if t < 0.15 {
            t / 0.15
        } else if t > 0.85 {
            (1.0 - t) / 0.15
        } else {
            1.0
        };
        let taper = taper.max(0.25) as f32;

        let r_val = radius_noise.sample(pos.x as f64 * 0.05, pos.y as f64 * 0.05, pos.z as f64 * 0.05) as f32;
        let radius = (radius_min + (radius_max - radius_min) * (r_val * 0.5 + 0.5)) * taper;

        segments.push(TubeSegment {
            position: pos,
            radius,
        });

        // Update direction — much flatter than worm caves
        let yaw_delta = yaw_noise.sample(
            t * 3.0,
            pos.y as f64 * 0.1,
            pos.z as f64 * 0.1,
        ) as f32;
        let pitch_delta = pitch_noise.sample(
            t * 3.0,
            pos.x as f64 * 0.1,
            pos.z as f64 * 0.1,
        ) as f32;

        yaw += yaw_delta * 0.2; // gentler turning than worms
        pitch += pitch_delta * 0.05;
        pitch = pitch.clamp(-0.26, 0.26); // ±15° max pitch
        pitch -= 0.005; // gentle downslope bias

        let dir = Vec3::new(
            yaw.cos() * pitch.cos(),
            pitch.sin(),
            yaw.sin() * pitch.cos(),
        );

        pos += dir * step_length;
    }

    segments
}
