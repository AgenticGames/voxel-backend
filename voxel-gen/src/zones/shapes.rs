//! Shared shape primitives for zone generation.
//!
//! Provides dome carving, slot carving, shelf writing, material replacement,
//! and ice layer application used by multiple zone types.

use std::collections::HashMap;

use glam::Vec3;
use voxel_core::material::Material;

use crate::density::DensityField;

/// Smoothstep interpolation (cubic Hermite, 0→1 as t goes 0→1).
pub fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Carve an ellipsoidal dome into the density field.
/// Only carves voxels that already have air within `proximity` voxels (prevents punching into solid).
/// `semi_axes` = (rx, ry, rz) half-extents of the ellipsoid.
pub fn carve_ellipsoid(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    center: Vec3,
    semi_axes: Vec3,
    falloff_power: f32,
    effective_bounds: f32,
) {
    let max_r = semi_axes.x.max(semi_axes.y).max(semi_axes.z);
    let min_cx = ((center.x - max_r) / effective_bounds).floor() as i32;
    let max_cx = ((center.x + max_r) / effective_bounds).floor() as i32;
    let min_cy = ((center.y - max_r) / effective_bounds).floor() as i32;
    let max_cy = ((center.y + max_r) / effective_bounds).floor() as i32;
    let min_cz = ((center.z - max_r) / effective_bounds).floor() as i32;
    let max_cz = ((center.z + max_r) / effective_bounds).floor() as i32;

    for cz in min_cz..=max_cz {
        for cy in min_cy..=max_cy {
            for cx in min_cx..=max_cx {
                if let Some(density) = density_fields.get_mut(&(cx, cy, cz)) {
                    let origin = Vec3::new(
                        cx as f32 * effective_bounds,
                        cy as f32 * effective_bounds,
                        cz as f32 * effective_bounds,
                    );
                    let size = density.size;
                    let vs = effective_bounds / (size - 1) as f32;

                    for z in 0..size {
                        for y in 0..size {
                            for x in 0..size {
                                let world_pos = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                let offset = world_pos - center;
                                // Normalized distance within ellipsoid
                                let nd = (offset.x / semi_axes.x).powi(2)
                                    + (offset.y / semi_axes.y).powi(2)
                                    + (offset.z / semi_axes.z).powi(2);
                                if nd < 1.0 {
                                    let idx = z * size * size + y * size + x;
                                    let t = nd.sqrt(); // 0 at center, 1 at surface
                                    let falloff = smoothstep(t).powf(falloff_power);
                                    let new_density = density.samples[idx].density * falloff - (1.0 - falloff);
                                    if new_density < density.samples[idx].density {
                                        density.samples[idx].density = new_density;
                                        if new_density <= 0.0 {
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
}

/// Replace all solid materials within an AABB with a target material.
pub fn replace_material_in_aabb(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    world_min: Vec3,
    world_max: Vec3,
    target_material: Material,
    effective_bounds: f32,
) {
    let min_cx = (world_min.x / effective_bounds).floor() as i32;
    let max_cx = (world_max.x / effective_bounds).floor() as i32;
    let min_cy = (world_min.y / effective_bounds).floor() as i32;
    let max_cy = (world_max.y / effective_bounds).floor() as i32;
    let min_cz = (world_min.z / effective_bounds).floor() as i32;
    let max_cz = (world_max.z / effective_bounds).floor() as i32;

    for cz in min_cz..=max_cz {
        for cy in min_cy..=max_cy {
            for cx in min_cx..=max_cx {
                if let Some(density) = density_fields.get_mut(&(cx, cy, cz)) {
                    let origin = Vec3::new(
                        cx as f32 * effective_bounds,
                        cy as f32 * effective_bounds,
                        cz as f32 * effective_bounds,
                    );
                    let size = density.size;
                    let vs = effective_bounds / (size - 1) as f32;

                    for z in 0..size {
                        for y in 0..size {
                            for x in 0..size {
                                let world_pos = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                if world_pos.x >= world_min.x && world_pos.x <= world_max.x
                                    && world_pos.y >= world_min.y && world_pos.y <= world_max.y
                                    && world_pos.z >= world_min.z && world_pos.z <= world_max.z
                                {
                                    let idx = z * size * size + y * size + x;
                                    if density.samples[idx].density > 0.0 && density.samples[idx].material.is_solid() {
                                        density.samples[idx].material = target_material;
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

/// Apply a surface ice layer: for solid voxels within 1 voxel of air inside an AABB, set material to Ice.
pub fn apply_surface_material(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    world_min: Vec3,
    world_max: Vec3,
    material: Material,
    effective_bounds: f32,
) {
    let min_cx = (world_min.x / effective_bounds).floor() as i32;
    let max_cx = (world_max.x / effective_bounds).floor() as i32;
    let min_cy = (world_min.y / effective_bounds).floor() as i32;
    let max_cy = (world_max.y / effective_bounds).floor() as i32;
    let min_cz = (world_min.z / effective_bounds).floor() as i32;
    let max_cz = (world_max.z / effective_bounds).floor() as i32;

    for cz in min_cz..=max_cz {
        for cy in min_cy..=max_cy {
            for cx in min_cx..=max_cx {
                if let Some(density) = density_fields.get_mut(&(cx, cy, cz)) {
                    let origin = Vec3::new(
                        cx as f32 * effective_bounds,
                        cy as f32 * effective_bounds,
                        cz as f32 * effective_bounds,
                    );
                    let size = density.size;
                    let vs = effective_bounds / (size - 1) as f32;

                    for z in 0..size {
                        for y in 0..size {
                            for x in 0..size {
                                let world_pos = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                if world_pos.x < world_min.x || world_pos.x > world_max.x
                                    || world_pos.y < world_min.y || world_pos.y > world_max.y
                                    || world_pos.z < world_min.z || world_pos.z > world_max.z
                                {
                                    continue;
                                }

                                let idx = z * size * size + y * size + x;
                                if density.samples[idx].density <= 0.0 {
                                    continue; // skip air
                                }

                                // Check 6 neighbors for air
                                let has_air_neighbor = [
                                    (x + 1, y, z), (x.wrapping_sub(1), y, z),
                                    (x, y + 1, z), (x, y.wrapping_sub(1), z),
                                    (x, y, z + 1), (x, y, z.wrapping_sub(1)),
                                ].iter().any(|&(nx, ny, nz)| {
                                    if nx < size && ny < size && nz < size {
                                        let ni = nz * size * size + ny * size + nx;
                                        density.samples[ni].density <= 0.0
                                    } else {
                                        false
                                    }
                                });

                                if has_air_neighbor {
                                    density.samples[idx].material = material;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Write a solid sphere (positive density) at a world position.
pub fn write_solid_sphere(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    center: Vec3,
    radius: f32,
    material: Material,
    effective_bounds: f32,
) {
    let min_cx = ((center.x - radius) / effective_bounds).floor() as i32;
    let max_cx = ((center.x + radius) / effective_bounds).floor() as i32;
    let min_cy = ((center.y - radius) / effective_bounds).floor() as i32;
    let max_cy = ((center.y + radius) / effective_bounds).floor() as i32;
    let min_cz = ((center.z - radius) / effective_bounds).floor() as i32;
    let max_cz = ((center.z + radius) / effective_bounds).floor() as i32;

    let r2 = radius * radius;

    for cz in min_cz..=max_cz {
        for cy in min_cy..=max_cy {
            for cx in min_cx..=max_cx {
                if let Some(density) = density_fields.get_mut(&(cx, cy, cz)) {
                    let origin = Vec3::new(
                        cx as f32 * effective_bounds,
                        cy as f32 * effective_bounds,
                        cz as f32 * effective_bounds,
                    );
                    let size = density.size;
                    let vs = effective_bounds / (size - 1) as f32;

                    for z in 0..size {
                        for y in 0..size {
                            for x in 0..size {
                                let world_pos = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                let dist2 = (world_pos - center).length_squared();
                                if dist2 < r2 {
                                    let idx = z * size * size + y * size + x;
                                    let t = (dist2 / r2).sqrt();
                                    let new_density = 1.0 - smoothstep(t);
                                    if new_density > density.samples[idx].density {
                                        density.samples[idx].density = new_density;
                                        density.samples[idx].material = material;
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

/// Write a solid cone (stalactite/stalagmite shape) at a world position.
/// `direction`: -1.0 = hanging down (stalactite), +1.0 = growing up (stalagmite).
pub fn write_cone(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    anchor: Vec3,
    length: f32,
    base_radius: f32,
    direction: f32,
    material: Material,
    smoothness: f32,
    effective_bounds: f32,
) {
    let tip = anchor + Vec3::new(0.0, direction * length, 0.0);
    let min_y = anchor.y.min(tip.y) - 1.0;
    let max_y = anchor.y.max(tip.y) + 1.0;

    let min_cx = ((anchor.x - base_radius - 1.0) / effective_bounds).floor() as i32;
    let max_cx = ((anchor.x + base_radius + 1.0) / effective_bounds).floor() as i32;
    let min_cy = ((min_y) / effective_bounds).floor() as i32;
    let max_cy = ((max_y) / effective_bounds).floor() as i32;
    let min_cz = ((anchor.z - base_radius - 1.0) / effective_bounds).floor() as i32;
    let max_cz = ((anchor.z + base_radius + 1.0) / effective_bounds).floor() as i32;

    for cz in min_cz..=max_cz {
        for cy in min_cy..=max_cy {
            for cx in min_cx..=max_cx {
                if let Some(density) = density_fields.get_mut(&(cx, cy, cz)) {
                    let origin = Vec3::new(
                        cx as f32 * effective_bounds,
                        cy as f32 * effective_bounds,
                        cz as f32 * effective_bounds,
                    );
                    let size = density.size;
                    let vs = effective_bounds / (size - 1) as f32;

                    for z in 0..size {
                        for y in 0..size {
                            for x in 0..size {
                                let world_pos = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                // Distance along cone axis (0 at anchor, 1 at tip)
                                let t = (world_pos.y - anchor.y) / (direction * length);
                                if t < 0.0 || t > 1.0 {
                                    continue;
                                }
                                // Radius tapers linearly from base_radius to 0
                                let max_r = base_radius * (1.0 - t);
                                let dist_h = ((world_pos.x - anchor.x).powi(2) + (world_pos.z - anchor.z).powi(2)).sqrt();
                                if dist_h > max_r + 1.0 {
                                    continue;
                                }
                                let falloff = ((max_r - dist_h) * smoothness).min(1.0).max(0.0);
                                if falloff > 0.0 {
                                    let idx = z * size * size + y * size + x;
                                    if falloff > density.samples[idx].density {
                                        density.samples[idx].density = falloff;
                                        density.samples[idx].material = material;
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

/// Write a mushroom shape: clearance carve + stalk (cylinder) + cap (dome) + gill ring (glow underside).
/// `base` is the floor anchor point. Stalk grows upward.
/// Carves clearance air around the mushroom first to prevent blending with terrain.
pub fn write_mushroom(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    base: Vec3,
    stalk_height: f32,
    stalk_radius: f32,
    cap_radius: f32,
    cap_thickness: f32,
    stalk_material: Material,
    cap_material: Material,
    gill_material: Material,
    effective_bounds: f32,
) {
    let cap_center = base + Vec3::new(0.0, stalk_height, 0.0);
    let clearance_r = cap_radius + 1.5;
    let max_y_top = cap_center.y + cap_thickness + 1.5;
    let min_y_bot = base.y - 1.0;

    let min_cx = ((base.x - clearance_r) / effective_bounds).floor() as i32;
    let max_cx = ((base.x + clearance_r) / effective_bounds).floor() as i32;
    let min_cy = ((min_y_bot) / effective_bounds).floor() as i32;
    let max_cy = ((max_y_top) / effective_bounds).floor() as i32;
    let min_cz = ((base.z - clearance_r) / effective_bounds).floor() as i32;
    let max_cz = ((base.z + clearance_r) / effective_bounds).floor() as i32;

    // Pass 1: Clearance carve — remove terrain around mushroom so it doesn't blend
    for cz in min_cz..=max_cz {
        for cy in min_cy..=max_cy {
            for cx in min_cx..=max_cx {
                if let Some(density) = density_fields.get_mut(&(cx, cy, cz)) {
                    let origin = Vec3::new(
                        cx as f32 * effective_bounds,
                        cy as f32 * effective_bounds,
                        cz as f32 * effective_bounds,
                    );
                    let size = density.size;
                    let vs = effective_bounds / (size - 1) as f32;

                    for z in 0..size {
                        for y in 0..size {
                            for x in 0..size {
                                let wp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                let dx = wp.x - base.x;
                                let dz = wp.z - base.z;
                                let dist_xz = (dx * dx + dz * dz).sqrt();
                                let dy = wp.y - base.y;
                                let idx = z * size * size + y * size + x;

                                // Only carve in the mushroom's vertical range (above base)
                                if dy < 0.5 || dy > stalk_height + cap_thickness + 1.0 {
                                    continue;
                                }

                                // Carve clearance cylinder around cap area
                                if dist_xz < clearance_r && density.samples[idx].density > 0.0 {
                                    // Don't carve where the mushroom itself will be written
                                    let is_stalk = dist_xz <= stalk_radius + 0.3;
                                    let cap_dy = wp.y - cap_center.y;
                                    let is_cap = cap_dy >= 0.0 && cap_dy <= cap_thickness
                                        && dist_xz <= cap_radius * (1.0 - (cap_dy / cap_thickness).max(0.0).powi(2));

                                    if !is_stalk && !is_cap {
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

    // Pass 2: Write mushroom geometry (stalk → cap → gill)
    for cz in min_cz..=max_cz {
        for cy in min_cy..=max_cy {
            for cx in min_cx..=max_cx {
                if let Some(density) = density_fields.get_mut(&(cx, cy, cz)) {
                    let origin = Vec3::new(
                        cx as f32 * effective_bounds,
                        cy as f32 * effective_bounds,
                        cz as f32 * effective_bounds,
                    );
                    let size = density.size;
                    let vs = effective_bounds / (size - 1) as f32;

                    for z in 0..size {
                        for y in 0..size {
                            for x in 0..size {
                                let wp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                let dx = wp.x - base.x;
                                let dz = wp.z - base.z;
                                let dist_xz = (dx * dx + dz * dz).sqrt();
                                let dy = wp.y - base.y;
                                let cap_dy = wp.y - cap_center.y;
                                let idx = z * size * size + y * size + x;

                                // Stalk: cylinder from base up to cap
                                if dy >= -0.5 && dy <= stalk_height && dist_xz <= stalk_radius + 0.5 {
                                    let falloff = ((stalk_radius + 0.5 - dist_xz) * 2.0).min(1.0).max(0.0);
                                    if falloff > 0.0 && falloff > density.samples[idx].density {
                                        density.samples[idx].density = falloff;
                                        density.samples[idx].material = stalk_material;
                                    }
                                }

                                // Cap: dome AT and ABOVE cap center only (no downward overlap with gill)
                                if cap_dy >= 0.0 && cap_dy <= cap_thickness && dist_xz <= cap_radius + 0.5 {
                                    let t = cap_dy / cap_thickness;
                                    let dome_r = cap_radius * (1.0 - t * t);
                                    if dist_xz <= dome_r + 0.5 {
                                        let falloff = ((dome_r + 0.5 - dist_xz) * 2.0).min(1.0).max(0.0);
                                        if falloff > 0.0 && falloff > density.samples[idx].density {
                                            density.samples[idx].density = falloff;
                                            density.samples[idx].material = cap_material;
                                        }
                                    }
                                }

                                // Gill: strict flat underside only — narrow band below cap, not at edges
                                if cap_dy >= -0.5 && cap_dy < -0.1
                                    && dist_xz >= stalk_radius + 0.5
                                    && dist_xz <= cap_radius * 0.85
                                {
                                    let falloff = 0.9;
                                    if falloff > density.samples[idx].density {
                                        density.samples[idx].density = falloff;
                                        density.samples[idx].material = gill_material;
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
