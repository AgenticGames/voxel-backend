//! Shape primitive helpers: write cone, cylinder, shelf, drapery, dam,
//! shield, and cauldron geometry into the density field.

use glam::Vec3;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use voxel_core::material::Material;
use voxel_noise::simplex::Simplex3D;
use voxel_noise::NoiseSource;

use crate::config::FormationConfig;
use crate::density::DensityField;

/// Write a cone shape (stalactite or stalagmite) into the density field.
///
/// direction: -1 = downward (stalactite), +1 = upward (stalagmite)
pub(super) fn write_cone(
    density: &mut DensityField,
    anchor_x: usize,
    anchor_y: usize,
    anchor_z: usize,
    length: f32,
    radius: f32,
    direction: i32,
    material: Material,
    smoothness: f32,
    size: usize,
) {
    let r_ceil = radius.ceil() as i32;
    let l_ceil = length.ceil() as i32;

    let ax = anchor_x as f32;
    let ay = anchor_y as f32;
    let az = anchor_z as f32;

    // Bounding box in grid space
    let min_x = (anchor_x as i32 - r_ceil).max(0) as usize;
    let max_x = ((anchor_x as i32 + r_ceil) as usize + 1).min(size);
    let min_z = (anchor_z as i32 - r_ceil).max(0) as usize;
    let max_z = ((anchor_z as i32 + r_ceil) as usize + 1).min(size);

    let (min_y, max_y) = if direction < 0 {
        // Stalactite grows down from anchor
        let min = (anchor_y as i32 - l_ceil).max(0) as usize;
        (min, anchor_y + 1)
    } else {
        // Stalagmite grows up from anchor
        let max = ((anchor_y as i32 + l_ceil) as usize + 1).min(size);
        (anchor_y, max)
    };

    for z in min_z..max_z {
        for y in min_y..max_y {
            for x in min_x..max_x {
                let dy = if direction < 0 {
                    ay - y as f32 // distance downward from anchor
                } else {
                    y as f32 - ay // distance upward from anchor
                };

                if dy < 0.0 || dy > length {
                    continue;
                }

                let t = dy / length; // 0=base, 1=tip
                let max_r = radius * (1.0 - t); // linear taper
                if max_r <= 0.0 {
                    continue;
                }

                let dx = x as f32 - ax;
                let dz = z as f32 - az;
                let dist_h = (dx * dx + dz * dz).sqrt();

                if dist_h < max_r {
                    let new_density = ((max_r - dist_h) * smoothness).min(1.0);
                    let sample = density.get_mut(x, y, z);
                    if new_density > sample.density {
                        sample.density = new_density;
                        sample.material = material;
                    }
                }
            }
        }
    }
}

/// Write a column (floor-to-ceiling cylinder) with noise-modulated radius.
pub(super) fn write_column(
    density: &mut DensityField,
    cx: usize,
    cz: usize,
    floor_y: usize,
    ceil_y: usize,
    base_radius: f32,
    material: Material,
    smoothness: f32,
    column_noise: &Simplex3D,
    world_origin: Vec3,
    size: usize,
) {
    let r_ceil = (base_radius * 1.2).ceil() as i32; // account for noise expansion

    let min_x = (cx as i32 - r_ceil).max(0) as usize;
    let max_x = ((cx as i32 + r_ceil) as usize + 1).min(size);
    let min_z = (cz as i32 - r_ceil).max(0) as usize;
    let max_z = ((cz as i32 + r_ceil) as usize + 1).min(size);

    // Column spans from floor_y+1 to ceil_y-1 (the air space between)
    let min_y = if floor_y + 1 < size { floor_y + 1 } else { return };
    let max_y = ceil_y.min(size);
    if min_y >= max_y {
        return;
    }

    let ax = cx as f32;
    let az = cz as f32;

    for z in min_z..max_z {
        for y in min_y..max_y {
            for x in min_x..max_x {
                // Modulate radius with noise for organic look
                let wx = world_origin.x + x as f32;
                let wy = world_origin.y + y as f32;
                let wz = world_origin.z + z as f32;
                let noise_val = column_noise.sample(
                    wx as f64 * 0.3,
                    wy as f64 * 0.3,
                    wz as f64 * 0.3,
                ) as f32;
                let radius = base_radius * (1.0 + noise_val * 0.2); // ±20% variation

                let dx = x as f32 - ax;
                let dz = z as f32 - az;
                let dist_h = (dx * dx + dz * dz).sqrt();

                if dist_h < radius {
                    let new_density = ((radius - dist_h) * smoothness).min(1.0);
                    let sample = density.get_mut(x, y, z);
                    if new_density > sample.density {
                        sample.density = new_density;
                        sample.material = material;
                    }
                }
            }
        }
    }
}

/// Write a flowstone shelf extending from a wall surface.
pub(super) fn write_flowstone(
    density: &mut DensityField,
    anchor_x: usize,
    anchor_y: usize,
    anchor_z: usize,
    normal_x: f32,
    normal_z: f32,
    length: f32,
    thickness: f32,
    material: Material,
    smoothness: f32,
    size: usize,
) {
    let l_ceil = length.ceil() as usize;
    let t_ceil = thickness.ceil() as usize;

    // Bounding box: extend in normal direction + some perpendicular spread
    let extent = (l_ceil + 1).max(t_ceil + 1) as i32;

    let min_x = (anchor_x as i32 - extent).max(0) as usize;
    let max_x = ((anchor_x as i32 + extent) as usize + 1).min(size);
    let min_y = (anchor_y as i32 - t_ceil as i32 - 1).max(0) as usize;
    let max_y = ((anchor_y as i32 + t_ceil as i32) as usize + 2).min(size);
    let min_z = (anchor_z as i32 - extent).max(0) as usize;
    let max_z = ((anchor_z as i32 + extent) as usize + 1).min(size);

    let ax = anchor_x as f32;
    let ay = anchor_y as f32;
    let az = anchor_z as f32;

    for z in min_z..max_z {
        for y in min_y..max_y {
            for x in min_x..max_x {
                let dx = x as f32 - ax;
                let dz = z as f32 - az;

                // Project onto normal direction
                let along_normal = dx * normal_x + dz * normal_z;
                if along_normal < 0.0 || along_normal > length {
                    continue;
                }

                // Perpendicular distance from normal axis
                let perp = (dx - along_normal * normal_x).powi(2)
                    + (dz - along_normal * normal_z).powi(2);
                let perp = perp.sqrt();

                // Shelf width tapers: full width at anchor, narrow at tip
                let t = along_normal / length;
                let width_at_t = 1.0 * (1.0 - t * 0.5); // gentle taper
                if perp > width_at_t {
                    continue;
                }

                // Thickness reduces as shelf extends outward
                let thickness_at_t = thickness * (1.0 - t * 0.7); // 30% at tip
                let dy = (y as f32 - ay).abs();
                if dy > thickness_at_t {
                    continue;
                }

                let fill = ((thickness_at_t - dy) * smoothness).min(1.0);
                let new_density = fill * (1.0 - perp / width_at_t.max(0.01));
                let new_density = new_density.min(1.0);

                if new_density > 0.0 {
                    let sample = density.get_mut(x, y, z);
                    if new_density > sample.density {
                        sample.density = new_density;
                        sample.material = material;
                    }
                }
            }
        }
    }
}

/// Write a mega-column: fat cylinder with noise modulation and ring bumps.
pub(super) fn write_mega_column(
    density: &mut DensityField,
    cx: usize,
    cz: usize,
    floor_y: usize,
    ceil_y: usize,
    base_radius: f32,
    material: Material,
    config: &FormationConfig,
    column_noise: &Simplex3D,
    world_origin: Vec3,
    size: usize,
) {
    let r_ceil = (base_radius as usize) + 2;
    let min_x = cx.saturating_sub(r_ceil);
    let max_x = (cx + r_ceil + 1).min(size);
    let min_z = cz.saturating_sub(r_ceil);
    let max_z = (cz + r_ceil + 1).min(size);

    // Column spans from floor_y+1 to ceil_y-1 (the air space between)
    let min_y = if floor_y + 1 < size { floor_y + 1 } else { return };
    let max_y = ceil_y.min(size);
    if min_y >= max_y {
        return;
    }

    let height = (ceil_y - floor_y).max(1) as f32;

    for iz in min_z..max_z {
        for iy in min_y..max_y {
            for ix in min_x..max_x {
                let dx = ix as f32 - cx as f32;
                let dz = iz as f32 - cz as f32;
                let dist = (dx * dx + dz * dz).sqrt();

                // Noise modulation on radius
                let wx = world_origin.x + ix as f32;
                let wy = world_origin.y + iy as f32;
                let wz = world_origin.z + iz as f32;
                let n = column_noise.sample(
                    wx as f64 * 0.15,
                    wy as f64 * 0.15,
                    wz as f64 * 0.15,
                ) as f32;
                let effective_radius =
                    base_radius * (1.0 + n * config.mega_column_noise_strength);

                // Ring bumps
                let ring = 0.3
                    * (iy as f32 * config.mega_column_ring_frequency * std::f32::consts::TAU)
                        .sin();
                let r = effective_radius + ring;

                // Bulge at floor/ceiling junctions
                let y_frac = (iy - floor_y) as f32 / height;
                let junction_bulge = if y_frac < 0.1 || y_frac > 0.9 {
                    0.5
                } else {
                    0.0
                };
                let final_r = r + junction_bulge;

                if dist < final_r {
                    let falloff = 1.0 - (dist / final_r).powi(2);
                    let sample = density.get_mut(ix, iy, iz);
                    let new_density = falloff * config.smoothness;
                    if new_density > sample.density {
                        sample.density = new_density;
                        sample.material = material;
                    }
                }
            }
        }
    }
}

/// Write a drapery: thin hanging sheet from ceiling following slope direction.
pub(super) fn write_drapery(
    density: &mut DensityField,
    anchor_x: usize,
    anchor_y: usize,
    anchor_z: usize,
    direction_x: f32,
    direction_z: f32,
    length: f32,
    material: Material,
    config: &FormationConfig,
    size: usize,
) {
    let steps = (length * 2.0) as usize;
    if steps == 0 {
        return;
    }

    for step in 0..=steps {
        let t = step as f32 / steps as f32;
        let cx = anchor_x as f32 + direction_x * t * length;
        let drop = t * t * length * 0.25; // gentle drop — calcite sheets hang stiffly
        let wave =
            (step as f32 * config.drapery_wave_frequency).sin() * config.drapery_wave_amplitude;
        let cy = anchor_y as f32 - drop - wave.max(0.0);
        let cz = anchor_z as f32 + direction_z * t * length;

        let ix = cx.round() as i32;
        let iy = cy.round() as i32;
        let iz = cz.round() as i32;

        if ix < 0 || iy < 0 || iz < 0 {
            continue;
        }
        let ix = ix as usize;
        let iy = iy as usize;
        let iz = iz as usize;

        if ix < size && iy < size && iz < size {
            let sample = density.get_mut(ix, iy, iz);
            if config.smoothness > sample.density {
                sample.density = config.smoothness;
                sample.material = material;
            }
            // Also fill the voxel above for thickness
            if iy + 1 < size {
                let sample2 = density.get_mut(ix, iy + 1, iz);
                let half_smooth = config.smoothness * 0.5;
                if half_smooth > sample2.density {
                    sample2.density = half_smooth;
                    sample2.material = material;
                }
            }
        }
    }
}

/// Write a rimstone dam: curved wall on sloped floor with basin behind it.
pub(super) fn write_rimstone_dam(
    density: &mut DensityField,
    anchor_x: usize,
    anchor_y: usize,
    anchor_z: usize,
    slope_x: f32,
    slope_z: f32,
    dam_height: f32,
    pool_depth: f32,
    material: Material,
    config: &FormationConfig,
    size: usize,
) {
    // Dam perpendicular to slope direction
    let perp_x = -slope_z;
    let perp_z = slope_x;
    let arc_radius = 3.0_f32;

    // Variable dam count scales with height — taller dams form more terraces
    let dam_count = (1 + (dam_height * 1.5) as usize).min(4);
    let mut cumulative_offset = 0.0_f32;
    for dam_idx in 0..dam_count {
        let spacing = 1.5 + dam_idx as f32 * 0.7; // increasing spacing downslope
        if dam_idx > 0 {
            cumulative_offset += spacing;
        }
        let base_x = anchor_x as f32 + slope_x * cumulative_offset;
        let base_z = anchor_z as f32 + slope_z * cumulative_offset;
        let base_y = anchor_y as f32 - cumulative_offset * 0.3; // follows slope down

        // Arc across the slope
        for arc_step in -3..=3i32 {
            let t = arc_step as f32;
            let wx = (base_x + perp_x * t).round() as i32;
            let wz = (base_z + perp_z * t).round() as i32;
            let wy = base_y.round() as i32;

            if wx < 0 || wz < 0 || wy < 0 {
                continue;
            }
            let wx = wx as usize;
            let wz = wz as usize;
            let wy = wy as usize;

            if wx >= size || wz >= size {
                continue;
            }

            // Build dam wall
            let h = (dam_height * (1.0 - (t / arc_radius).powi(2)).max(0.0)) as usize;
            for dy in 0..=h {
                if wy + dy < size {
                    let sample = density.get_mut(wx, wy + dy, wz);
                    if config.smoothness > sample.density {
                        sample.density = config.smoothness;
                        sample.material = material;
                    }
                }
            }

            // Carve basin behind dam (upslope side)
            let basin_x = (wx as f32 - slope_x * 1.5).round() as i32;
            let basin_z = (wz as f32 - slope_z * 1.5).round() as i32;
            if basin_x >= 0 && basin_z >= 0 {
                let basin_x = basin_x as usize;
                let basin_z = basin_z as usize;
                if basin_x < size && basin_z < size && wy > 0 {
                    let depth = pool_depth.min(wy as f32) as usize;
                    for dy in 0..depth {
                        if wy - dy > 0 {
                            let sample = density.get_mut(basin_x, wy - dy, basin_z);
                            sample.density = -0.5; // carve
                        }
                    }
                }
            }
        }
    }
}

/// Write a cave shield: oriented disc projecting from wall.
pub(super) fn write_cave_shield(
    density: &mut DensityField,
    anchor_x: usize,
    anchor_y: usize,
    anchor_z: usize,
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
    tilt_x: f32,
    tilt_y: f32,
    radius: f32,
    material: Material,
    config: &FormationConfig,
    size: usize,
    rng: &mut ChaCha8Rng,
) {
    // Bias normal upward ~30° from wall (real shields project upward from cracks)
    let biased_nx = normal_x;
    let biased_ny = normal_y + 0.577; // tan(30°) ≈ 0.577
    let biased_nz = normal_z;
    let bmag = (biased_nx * biased_nx + biased_ny * biased_ny + biased_nz * biased_nz).sqrt();
    let (normal_x, normal_y, normal_z) = if bmag > 0.001 {
        (biased_nx / bmag, biased_ny / bmag, biased_nz / bmag)
    } else {
        (normal_x, normal_y, normal_z)
    };

    // Build orthogonal basis from biased normal
    let up = if normal_y.abs() < 0.9 {
        (0.0_f32, 1.0, 0.0)
    } else {
        (1.0, 0.0, 0.0)
    };
    let tangent_x = normal_y * up.2 - normal_z * up.1;
    let tangent_y = normal_z * up.0 - normal_x * up.2;
    let tangent_z = normal_x * up.1 - normal_y * up.0;
    let tmag =
        (tangent_x * tangent_x + tangent_y * tangent_y + tangent_z * tangent_z).sqrt();
    if tmag <= 0.001 {
        return; // degenerate
    }
    let (tx, ty, tz) = (tangent_x / tmag, tangent_y / tmag, tangent_z / tmag);
    let bx = normal_y * tz - normal_z * ty;
    let by = normal_z * tx - normal_x * tz;
    let bz = normal_x * ty - normal_y * tx;

    // Apply tilt
    let tilt_rad_x = tilt_x.to_radians();
    let tilt_rad_y = tilt_y.to_radians();

    let r_ceil = radius.ceil() as i32;
    for du in -r_ceil..=r_ceil {
        for dv in -r_ceil..=r_ceil {
            let u = du as f32;
            let v = dv as f32;
            let rdist = (u * u + v * v).sqrt();
            if rdist > radius {
                continue;
            }

            // Apply tilt offset in normal direction
            let tilt_offset = u * tilt_rad_x.sin() + v * tilt_rad_y.sin();

            let px = anchor_x as f32 + tx * u + bx * v + normal_x * tilt_offset;
            let py = anchor_y as f32 + ty * u + by * v + normal_y * tilt_offset;
            let pz = anchor_z as f32 + tz * u + bz * v + normal_z * tilt_offset;

            let ix = px.round() as i32;
            let iy = py.round() as i32;
            let iz = pz.round() as i32;

            if ix < 0 || iy < 0 || iz < 0 {
                continue;
            }
            let ix = ix as usize;
            let iy = iy as usize;
            let iz = iz as usize;

            if ix < size && iy < size && iz < size {
                let falloff = 1.0 - (rdist / radius).powi(2);
                let new_density = falloff * config.smoothness;
                let sample = density.get_mut(ix, iy, iz);
                if new_density > sample.density {
                    sample.density = new_density;
                    sample.material = material;
                }
            }
        }
    }

    // Stalactite from lowest disc edge (real shields often have stalactites hanging below)
    if rng.gen::<f32>() < config.shield_stalactite_chance {
        // Sample 8 points around the disc edge, find lowest Y
        let mut lowest_y = f32::MAX;
        let mut lowest_pos = (anchor_x as f32, anchor_y as f32, anchor_z as f32);
        for i in 0..8 {
            let angle = i as f32 * std::f32::consts::TAU / 8.0;
            let eu = angle.cos() * radius;
            let ev = angle.sin() * radius;
            let tilt_offset = eu * tilt_rad_x.sin() + ev * tilt_rad_y.sin();
            let py = anchor_y as f32 + ty * eu + by * ev + normal_y * tilt_offset;
            if py < lowest_y {
                lowest_y = py;
                lowest_pos = (
                    anchor_x as f32 + tx * eu + bx * ev + normal_x * tilt_offset,
                    py,
                    anchor_z as f32 + tz * eu + bz * ev + normal_z * tilt_offset,
                );
            }
        }

        // Write a small downward cone from the lowest edge point
        let stalk_len = rng.gen_range(2.0_f32..=3.0);
        let stalk_radius = rng.gen_range(0.3_f32..=0.5);
        let base_ix = lowest_pos.0.round() as i32;
        let base_iy = lowest_pos.1.round() as i32;
        let base_iz = lowest_pos.2.round() as i32;
        if base_ix >= 0 && base_iy >= 0 && base_iz >= 0 {
            let bx_u = base_ix as usize;
            let by_u = base_iy as usize;
            let bz_u = base_iz as usize;
            if bx_u < size && by_u < size && bz_u < size {
                write_cone(
                    density,
                    bx_u,
                    by_u,
                    bz_u,
                    stalk_len,
                    stalk_radius,
                    -1, // downward
                    material,
                    config.smoothness,
                    size,
                );
            }
        }
    }
}

/// Write a cauldron: bowl-shaped depression carved into the floor with a raised lip
/// and small stalagmites on the rim.
pub(super) fn write_cauldron(
    density: &mut DensityField,
    anchor_x: usize,
    anchor_y: usize,
    anchor_z: usize,
    radius: f32,
    depth: f32,
    lip_height: f32,
    rim_stalagmite_count: u32,
    rim_stalagmite_scale: f32,
    floor_noise_amp: f32,
    material: Material,
    config: &FormationConfig,
    world_seed: u64,
    size: usize,
) {
    let r_ceil = (radius + 1.0).ceil() as i32; // +1 for lip ring
    let d_ceil = depth.ceil() as i32;
    let ax = anchor_x as f32;
    let az = anchor_z as f32;
    let ay = anchor_y as f32;

    let floor_noise = voxel_noise::simplex::Simplex3D::new(world_seed.wrapping_add(300));

    // Phase 1: Carve basin — circular footprint with steep walls and flat bottom
    let min_x = (anchor_x as i32 - r_ceil).max(0) as usize;
    let max_x = ((anchor_x as i32 + r_ceil) as usize + 1).min(size);
    let min_z = (anchor_z as i32 - r_ceil).max(0) as usize;
    let max_z = ((anchor_z as i32 + r_ceil) as usize + 1).min(size);
    let min_y = (anchor_y as i32 - d_ceil).max(0) as usize;
    let max_y = ((anchor_y as f32 + lip_height.ceil()) as usize + 1).min(size);

    for iz in min_z..max_z {
        for iy in min_y..max_y {
            for ix in min_x..max_x {
                let dx = ix as f32 - ax;
                let dz = iz as f32 - az;
                let dist_h = (dx * dx + dz * dz).sqrt();

                if dist_h >= radius {
                    continue;
                }

                let fy = iy as f32;

                if fy <= ay {
                    // Below-floor carve: existing steep-wall + flat-bottom profile
                    let rim_factor = dist_h / radius;
                    let carve_depth_at_r = depth * (1.0 - rim_factor.powf(0.3).min(1.0));
                    let dy_below = ay - fy;
                    if dy_below > carve_depth_at_r {
                        continue;
                    }
                } else {
                    // Above-floor carve: clear rock inside the basin up to lip_height.
                    // Taper the carve height near the lip ring (0.85r..r) so we
                    // don't cut a sharp vertical edge into the lip Phase 2 builds.
                    let dy_above = fy - ay;
                    let max_carve_h = if dist_h < radius * 0.85 {
                        // Interior: carve up to full lip_height
                        lip_height
                    } else {
                        // Transition zone (0.85r..r): linearly taper from lip_height → 0
                        let t = (dist_h - radius * 0.85) / (radius * 0.15);
                        lip_height * (1.0 - t.min(1.0))
                    };
                    if dy_above > max_carve_h {
                        continue;
                    }
                }

                // Add noise to basin floor
                let noise_val = if floor_noise_amp > 0.0 {
                    let n = floor_noise.sample(
                        ix as f64 * 0.5,
                        iy as f64 * 0.5,
                        iz as f64 * 0.5,
                    ) as f32;
                    n * floor_noise_amp
                } else {
                    0.0
                };

                // Only carve (make density more negative), don't add solid
                let carve_val = -0.8 + noise_val * 0.2;
                let sample = density.get_mut(ix, iy, iz);
                if carve_val < sample.density {
                    sample.density = carve_val;
                }
            }
        }
    }

    // Phase 1.5: Seal basin shell — fill air below walls and floor so the cauldron
    // is watertight even when placed on slopes or above pre-existing caves.

    // Walls: outer ring extending downward from anchor to carve depth
    for iz in min_z..max_z {
        for ix in min_x..max_x {
            let dx = ix as f32 - ax;
            let dz = iz as f32 - az;
            let dist_h = (dx * dx + dz * dz).sqrt();

            // Same ring as the lip: between radius*0.85 and radius+1
            if dist_h < radius * 0.85 || dist_h > radius + 1.0 {
                continue;
            }

            // Wall extends down to the deepest carve at this radial distance
            let rim_factor = (dist_h / radius).min(1.0);
            let carve_depth_at_r = depth * (1.0 - rim_factor.powf(0.3).min(1.0));
            let bottom_y = (ay - carve_depth_at_r).floor() as i32;
            let wall_min_y = bottom_y.max(0) as usize;

            for iy in wall_min_y..anchor_y {
                let sample = density.get_mut(ix, iy, iz);
                if sample.density < 0.5 {
                    sample.density = 0.8;
                    sample.material = Material::Limestone;
                }
            }
        }
    }

    // Floor: 3-voxel-thick disc at the bottom of the carved basin
    for iz in min_z..max_z {
        for ix in min_x..max_x {
            let dx = ix as f32 - ax;
            let dz = iz as f32 - az;
            let dist_h = (dx * dx + dz * dz).sqrt();

            if dist_h >= radius {
                continue;
            }

            let rim_factor = dist_h / radius;
            let carve_depth_at_r = depth * (1.0 - rim_factor.powf(0.3).min(1.0));
            let bottom_y = (ay - carve_depth_at_r).floor() as i32;

            // 3 voxels thick: bottom_y, bottom_y-1, bottom_y-2
            for offset in 0..3_i32 {
                let iy = bottom_y - offset;
                if iy < 0 || iy as usize >= size {
                    continue;
                }
                let sample = density.get_mut(ix, iy as usize, iz);
                if sample.density < 0.5 {
                    sample.density = 0.8;
                    sample.material = Material::Limestone;
                }
            }
        }
    }

    // Phase 1.6/1.7 removed — boundary sealing handled by binary cell classification
    // in the fluid sim (center density > 0 = solid cap, no fractional capacity).

    // Phase 2: Build lip — ring from radius to radius+1, tapered height
    let lip_min_y = anchor_y;
    let lip_max_y = ((anchor_y as f32 + lip_height.ceil()) as usize + 1).min(size);
    for iz in min_z..max_z {
        for iy in lip_min_y..lip_max_y {
            for ix in min_x..max_x {
                let dx = ix as f32 - ax;
                let dz = iz as f32 - az;
                let dist_h = (dx * dx + dz * dz).sqrt();

                // Lip ring: between radius and radius+1
                if dist_h < radius * 0.85 || dist_h > radius + 1.0 {
                    continue;
                }

                // Taper height: full at inner edge, zero at outer edge
                let ring_t = (dist_h - radius * 0.85) / (radius * 0.15 + 1.0);
                let lip_h_at_r = lip_height * (1.0 - ring_t.max(0.0).min(1.0));
                let dy_above = iy as f32 - ay;
                if dy_above < 0.0 || dy_above > lip_h_at_r {
                    continue;
                }

                let new_density = ((lip_h_at_r - dy_above) * config.smoothness).min(1.0);
                let sample = density.get_mut(ix, iy, iz);
                if new_density > sample.density {
                    sample.density = new_density;
                    sample.material = material;
                }
            }
        }
    }

    // Phase 3: Rim stalagmites — evenly spaced around rim with jitter
    if rim_stalagmite_count > 0 {
        let scaled_length_min = config.length_min * rim_stalagmite_scale;
        let scaled_length_max = config.length_max * rim_stalagmite_scale;
        let scaled_radius_min = config.radius_min * rim_stalagmite_scale;
        let scaled_radius_max = config.radius_max * rim_stalagmite_scale;

        let mut stag_rng = ChaCha8Rng::seed_from_u64(
            world_seed.wrapping_add(301).wrapping_add(anchor_x as u64 * 7 + anchor_z as u64 * 13),
        );

        for i in 0..rim_stalagmite_count {
            let base_angle = (i as f32 / rim_stalagmite_count as f32) * std::f32::consts::TAU;
            let jitter = stag_rng.gen_range(-0.3_f32..=0.3);
            let angle = base_angle + jitter;

            let sx = (ax + angle.cos() * radius).round() as i32;
            let sz = (az + angle.sin() * radius).round() as i32;
            let sy = anchor_y as i32 + 1; // on top of floor

            if sx < 0 || sz < 0 || sy < 0 {
                continue;
            }
            let sx = sx as usize;
            let sz = sz as usize;
            let sy = sy as usize;
            if sx >= size || sz >= size || sy >= size {
                continue;
            }

            let stag_length = stag_rng.gen_range(scaled_length_min..=scaled_length_max);
            let stag_radius = stag_rng.gen_range(scaled_radius_min..=scaled_radius_max);

            write_cone(
                density,
                sx, sy, sz,
                stag_length,
                stag_radius,
                1, // upward
                material,
                config.smoothness,
                size,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::density::DensityField;

    #[test]
    fn test_write_cone_creates_solid() {
        let mut field = DensityField::new(17);
        // Make everything air
        for s in &mut field.samples {
            s.density = -1.0;
            s.material = Material::Air;
        }

        // Write a stalactite cone at center
        write_cone(
            &mut field,
            8, 14, 8,  // anchor near top
            5.0,       // length
            1.5,       // radius
            -1,        // downward
            Material::Limestone,
            2.0,
            17,
        );

        // Check that some voxels below anchor became solid
        let filled = (0..17)
            .filter(|&y| field.get(8, y, 8).density > 0.0)
            .count();
        assert!(filled > 0, "Cone should fill some voxels");
    }
}
