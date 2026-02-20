//! Cave formation placement: stalactites, stalagmites, columns, and flowstone shelves.
//!
//! Scans the density field for ceiling/floor/wall surfaces adjacent to air,
//! uses noise + seeded RNG for sparse deterministic placement, then writes
//! cone/cylinder/shelf shapes as positive density into air voxels.
//! Formations inherit the host rock material from their anchor surface.

use std::collections::HashMap;

use glam::Vec3;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use voxel_core::material::Material;
use voxel_noise::simplex::Simplex3D;
use voxel_noise::NoiseSource;

use crate::config::FormationConfig;
use crate::density::DensityField;

/// Type of surface detected adjacent to air.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SurfaceKind {
    /// Solid with air below (stalactite anchor)
    Ceiling,
    /// Solid with air above (stalagmite anchor)
    Floor,
    /// Solid with air at an X/Z neighbor (flowstone anchor)
    Wall,
}

/// A detected surface point where a formation can be anchored.
#[derive(Debug, Clone)]
struct SurfacePoint {
    x: usize,
    y: usize,
    z: usize,
    kind: SurfaceKind,
    material: Material,
}

/// A formation to be written into the density field.
#[derive(Debug)]
enum Formation {
    Stalactite {
        anchor_x: usize,
        anchor_y: usize,
        anchor_z: usize,
        length: f32,
        radius: f32,
        material: Material,
    },
    Stalagmite {
        anchor_x: usize,
        anchor_y: usize,
        anchor_z: usize,
        length: f32,
        radius: f32,
        material: Material,
    },
    Column {
        x: usize,
        z: usize,
        floor_y: usize,
        ceil_y: usize,
        radius: f32,
        material: Material,
    },
    Flowstone {
        anchor_x: usize,
        anchor_y: usize,
        anchor_z: usize,
        normal_x: f32,
        normal_z: f32,
        length: f32,
        thickness: f32,
        material: Material,
    },
}

/// Place cave formations into the density field after worm carving.
///
/// Called between worm carving and hermite extraction so formations become
/// natural parts of the mesh via dual contouring.
pub fn place_formations(
    density: &mut DensityField,
    config: &FormationConfig,
    world_origin: Vec3,
    world_seed: u64,
    chunk_seed: u64,
) {
    if !config.enabled {
        return;
    }

    let size = density.size;
    if size < 4 {
        return;
    }

    // Step A: Detect surfaces
    let surfaces = detect_surfaces(density, size);
    if surfaces.is_empty() {
        return;
    }

    // Step B: Noise filter for world-coherent sparse placement
    let placement_noise = Simplex3D::new(world_seed.wrapping_add(200));
    let surfaces: Vec<SurfacePoint> = surfaces
        .into_iter()
        .filter(|sp| {
            let wx = world_origin.x + sp.x as f32;
            let wy = world_origin.y + sp.y as f32;
            let wz = world_origin.z + sp.z as f32;
            let n = placement_noise.sample(
                wx as f64 * config.placement_frequency,
                wy as f64 * config.placement_frequency,
                wz as f64 * config.placement_frequency,
            );
            n * 0.5 + 0.5 > config.placement_threshold
        })
        .collect();

    if surfaces.is_empty() {
        return;
    }

    // Step C: RNG filter + Step D: Column detection + Step E: Dimension rolls
    let mut rng = ChaCha8Rng::seed_from_u64(chunk_seed.wrapping_add(0xF0E4A710));
    let column_noise = Simplex3D::new(world_seed.wrapping_add(201));

    // Separate surfaces by kind for column detection
    let mut ceiling_by_xz: HashMap<(usize, usize), Vec<&SurfacePoint>> = HashMap::new();
    let mut floor_by_xz: HashMap<(usize, usize), Vec<&SurfacePoint>> = HashMap::new();
    let mut wall_surfaces: Vec<&SurfacePoint> = Vec::new();
    let mut ceiling_surfaces: Vec<&SurfacePoint> = Vec::new();
    let mut floor_surfaces: Vec<&SurfacePoint> = Vec::new();

    for sp in &surfaces {
        match sp.kind {
            SurfaceKind::Ceiling => {
                ceiling_by_xz
                    .entry((sp.x, sp.z))
                    .or_default()
                    .push(sp);
                ceiling_surfaces.push(sp);
            }
            SurfaceKind::Floor => {
                floor_by_xz.entry((sp.x, sp.z)).or_default().push(sp);
                floor_surfaces.push(sp);
            }
            SurfaceKind::Wall => {
                wall_surfaces.push(sp);
            }
        }
    }

    let mut formations: Vec<Formation> = Vec::new();
    let mut used_ceiling: HashMap<(usize, usize, usize), bool> = HashMap::new();
    let mut used_floor: HashMap<(usize, usize, usize), bool> = HashMap::new();

    // Step D: Column detection — where ceiling and floor align at same (x,z)
    // Sort keys for deterministic iteration
    let mut xz_keys: Vec<(usize, usize)> = ceiling_by_xz.keys().copied().collect();
    xz_keys.sort();

    for (x, z) in xz_keys {
        let ceilings = match ceiling_by_xz.get(&(x, z)) {
            Some(c) => c,
            None => continue,
        };
        let floors = match floor_by_xz.get(&(x, z)) {
            Some(f) => f,
            None => continue,
        };

        for ceil_pt in ceilings {
            for floor_pt in floors {
                // Ceiling is solid with air below, so air starts at ceil_y - 1
                // Floor is solid with air above, so air starts at floor_y + 1
                if ceil_pt.y <= floor_pt.y {
                    continue; // ceiling must be above floor
                }
                let gap = ceil_pt.y - floor_pt.y - 1; // air voxels between
                if gap < config.min_air_gap || gap > config.column_max_gap {
                    continue;
                }
                if rng.gen::<f32>() < config.column_chance {
                    let radius = rng.gen_range(config.column_radius_min..=config.column_radius_max)
                        .min(config.max_radius);
                    formations.push(Formation::Column {
                        x,
                        z,
                        floor_y: floor_pt.y,
                        ceil_y: ceil_pt.y,
                        radius,
                        material: ceil_pt.material,
                    });
                    used_ceiling.insert((ceil_pt.x, ceil_pt.y, ceil_pt.z), true);
                    used_floor.insert((floor_pt.x, floor_pt.y, floor_pt.z), true);
                }
            }
        }
    }

    // Step C continued: RNG filter for individual stalactites/stalagmites/flowstone
    for sp in &ceiling_surfaces {
        if used_ceiling.contains_key(&(sp.x, sp.y, sp.z)) {
            continue;
        }
        if rng.gen::<f32>() >= config.stalactite_chance {
            continue;
        }
        // Measure air gap downward
        let air_gap = measure_air_down(density, sp.x, sp.y, sp.z, size);
        if air_gap < config.min_air_gap {
            continue;
        }
        let max_len = (air_gap - config.min_clearance) as f32;
        if max_len <= 0.0 {
            continue;
        }
        let length = rng
            .gen_range(config.length_min..=config.length_max)
            .min(max_len);
        let radius = rng
            .gen_range(config.radius_min..=config.radius_max)
            .min(config.max_radius);
        formations.push(Formation::Stalactite {
            anchor_x: sp.x,
            anchor_y: sp.y,
            anchor_z: sp.z,
            length,
            radius,
            material: sp.material,
        });
    }

    for sp in &floor_surfaces {
        if used_floor.contains_key(&(sp.x, sp.y, sp.z)) {
            continue;
        }
        if rng.gen::<f32>() >= config.stalagmite_chance {
            continue;
        }
        // Measure air gap upward
        let air_gap = measure_air_up(density, sp.x, sp.y, sp.z, size);
        if air_gap < config.min_air_gap {
            continue;
        }
        let max_len = (air_gap - config.min_clearance) as f32;
        if max_len <= 0.0 {
            continue;
        }
        let length = rng
            .gen_range(config.length_min..=config.length_max)
            .min(max_len);
        let radius = rng
            .gen_range(config.radius_min..=config.radius_max)
            .min(config.max_radius);
        formations.push(Formation::Stalagmite {
            anchor_x: sp.x,
            anchor_y: sp.y,
            anchor_z: sp.z,
            length,
            radius,
            material: sp.material,
        });
    }

    for sp in &wall_surfaces {
        if rng.gen::<f32>() >= config.flowstone_chance {
            continue;
        }
        // Determine outward normal direction (into air)
        let (nx, nz) = find_wall_normal(density, sp.x, sp.y, sp.z, size);
        if nx == 0.0 && nz == 0.0 {
            continue;
        }
        // Measure available air space in normal direction
        let air_extent = measure_air_extent(density, sp.x, sp.y, sp.z, nx, nz, size);
        if air_extent < 2 {
            continue;
        }
        let max_len = (air_extent / 2) as f32; // cap to half corridor
        let length = rng
            .gen_range(config.flowstone_length_min..=config.flowstone_length_max)
            .min(max_len);
        let thickness = config.flowstone_thickness;
        formations.push(Formation::Flowstone {
            anchor_x: sp.x,
            anchor_y: sp.y,
            anchor_z: sp.z,
            normal_x: nx,
            normal_z: nz,
            length,
            thickness,
            material: sp.material,
        });
    }

    // Step F: Write shapes into density field
    for formation in &formations {
        match formation {
            Formation::Stalactite {
                anchor_x,
                anchor_y,
                anchor_z,
                length,
                radius,
                material,
            } => {
                write_cone(
                    density,
                    *anchor_x,
                    *anchor_y,
                    *anchor_z,
                    *length,
                    *radius,
                    -1, // grows downward
                    *material,
                    config.smoothness,
                    size,
                );
            }
            Formation::Stalagmite {
                anchor_x,
                anchor_y,
                anchor_z,
                length,
                radius,
                material,
            } => {
                write_cone(
                    density,
                    *anchor_x,
                    *anchor_y,
                    *anchor_z,
                    *length,
                    *radius,
                    1, // grows upward
                    *material,
                    config.smoothness,
                    size,
                );
            }
            Formation::Column {
                x,
                z,
                floor_y,
                ceil_y,
                radius,
                material,
            } => {
                write_column(
                    density,
                    *x,
                    *z,
                    *floor_y,
                    *ceil_y,
                    *radius,
                    *material,
                    config.smoothness,
                    &column_noise,
                    world_origin,
                    size,
                );
            }
            Formation::Flowstone {
                anchor_x,
                anchor_y,
                anchor_z,
                normal_x,
                normal_z,
                length,
                thickness,
                material,
            } => {
                write_flowstone(
                    density,
                    *anchor_x,
                    *anchor_y,
                    *anchor_z,
                    *normal_x,
                    *normal_z,
                    *length,
                    *thickness,
                    *material,
                    config.smoothness,
                    size,
                );
            }
        }
    }
}

/// Scan voxels at indices 1..size-1 (avoid boundary) for solid voxels adjacent to air.
fn detect_surfaces(density: &DensityField, size: usize) -> Vec<SurfacePoint> {
    let mut surfaces = Vec::new();
    let scan_max = size - 1;

    for z in 1..scan_max {
        for y in 1..scan_max {
            for x in 1..scan_max {
                let sample = density.get(x, y, z);
                if sample.density <= 0.0 {
                    continue; // air voxel, not a surface
                }
                let material = sample.material;

                // Check if air below → ceiling surface
                if y > 0 && density.get(x, y - 1, z).density <= 0.0 {
                    surfaces.push(SurfacePoint {
                        x,
                        y,
                        z,
                        kind: SurfaceKind::Ceiling,
                        material,
                    });
                }

                // Check if air above → floor surface
                if y + 1 < size && density.get(x, y + 1, z).density <= 0.0 {
                    surfaces.push(SurfacePoint {
                        x,
                        y,
                        z,
                        kind: SurfaceKind::Floor,
                        material,
                    });
                }

                // Check X/Z neighbors for wall surface
                let has_air_neighbor = (x > 0 && density.get(x - 1, y, z).density <= 0.0)
                    || (x + 1 < size && density.get(x + 1, y, z).density <= 0.0)
                    || (z > 0 && density.get(x, y, z - 1).density <= 0.0)
                    || (z + 1 < size && density.get(x, y, z + 1).density <= 0.0);

                if has_air_neighbor {
                    // Only add wall if this is NOT already a ceiling or floor
                    // (avoid double-counting corners)
                    let is_ceiling = y > 0 && density.get(x, y - 1, z).density <= 0.0;
                    let is_floor = y + 1 < size && density.get(x, y + 1, z).density <= 0.0;
                    if !is_ceiling && !is_floor {
                        surfaces.push(SurfacePoint {
                            x,
                            y,
                            z,
                            kind: SurfaceKind::Wall,
                            material,
                        });
                    }
                }
            }
        }
    }

    surfaces
}

/// Measure contiguous air voxels downward from anchor at (x, anchor_y, z).
/// anchor_y is the solid voxel; air starts at anchor_y - 1.
fn measure_air_down(density: &DensityField, x: usize, anchor_y: usize, z: usize, _size: usize) -> usize {
    if anchor_y == 0 {
        return 0;
    }
    let mut gap = 0;
    let mut cy = anchor_y - 1;
    loop {
        if density.get(x, cy, z).density > 0.0 {
            break;
        }
        gap += 1;
        if cy == 0 {
            break;
        }
        cy -= 1;
    }
    gap
}

/// Measure contiguous air voxels upward from anchor at (x, anchor_y, z).
/// anchor_y is the solid voxel; air starts at anchor_y + 1.
fn measure_air_up(density: &DensityField, x: usize, anchor_y: usize, z: usize, size: usize) -> usize {
    let mut gap = 0;
    let mut cy = anchor_y + 1;
    while cy < size {
        if density.get(x, cy, z).density > 0.0 {
            break;
        }
        gap += 1;
        cy += 1;
    }
    gap
}

/// Find the dominant horizontal normal direction from a wall surface into air.
fn find_wall_normal(
    density: &DensityField,
    x: usize,
    y: usize,
    z: usize,
    size: usize,
) -> (f32, f32) {
    let mut nx: f32 = 0.0;
    let mut nz: f32 = 0.0;

    if x > 0 && density.get(x - 1, y, z).density <= 0.0 {
        nx -= 1.0;
    }
    if x + 1 < size && density.get(x + 1, y, z).density <= 0.0 {
        nx += 1.0;
    }
    if z > 0 && density.get(x, y, z - 1).density <= 0.0 {
        nz -= 1.0;
    }
    if z + 1 < size && density.get(x, y, z + 1).density <= 0.0 {
        nz += 1.0;
    }

    // Normalize
    let len = (nx * nx + nz * nz).sqrt();
    if len > 0.0 {
        (nx / len, nz / len)
    } else {
        (0.0, 0.0)
    }
}

/// Measure available air space from a wall surface in the normal direction.
fn measure_air_extent(
    density: &DensityField,
    x: usize,
    y: usize,
    z: usize,
    nx: f32,
    nz: f32,
    size: usize,
) -> usize {
    let mut extent = 0;
    for step in 1..size {
        let sx = (x as f32 + nx * step as f32).round() as i32;
        let sz = (z as f32 + nz * step as f32).round() as i32;
        if sx < 0 || sx >= size as i32 || sz < 0 || sz >= size as i32 {
            break;
        }
        if density.get(sx as usize, y, sz as usize).density > 0.0 {
            break;
        }
        extent += 1;
    }
    extent
}

/// Write a cone shape (stalactite or stalagmite) into the density field.
///
/// direction: -1 = downward (stalactite), +1 = upward (stalagmite)
fn write_cone(
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
fn write_column(
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
fn write_flowstone(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FormationConfig;

    /// Helper: create a density field with a horizontal cave (air slab).
    /// Solid above y_ceil and below y_floor, air in between.
    fn make_cave_field(size: usize, y_floor: usize, y_ceil: usize) -> DensityField {
        let mut field = DensityField::new(size);
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let sample = field.get_mut(x, y, z);
                    if y > y_floor && y < y_ceil {
                        // Air
                        sample.density = -1.0;
                        sample.material = Material::Air;
                    } else {
                        // Solid
                        sample.density = 1.0;
                        sample.material = Material::Limestone;
                    }
                }
            }
        }
        field
    }

    #[test]
    fn test_detect_surfaces_finds_ceiling_and_floor() {
        let field = make_cave_field(17, 4, 12);
        let surfaces = detect_surfaces(&field, 17);

        let has_ceiling = surfaces.iter().any(|s| s.kind == SurfaceKind::Ceiling);
        let has_floor = surfaces.iter().any(|s| s.kind == SurfaceKind::Floor);

        assert!(has_ceiling, "Should detect ceiling surfaces");
        assert!(has_floor, "Should detect floor surfaces");
    }

    #[test]
    fn test_place_formations_no_panic() {
        let mut field = make_cave_field(17, 3, 13);
        let config = FormationConfig::default();
        place_formations(
            &mut field,
            &config,
            Vec3::ZERO,
            42,
            12345,
        );
        // Should complete without panic
    }

    #[test]
    fn test_place_formations_adds_density() {
        let mut field = make_cave_field(17, 3, 13);
        let config = FormationConfig {
            enabled: true,
            placement_threshold: 0.0, // accept everything
            stalactite_chance: 1.0,
            stalagmite_chance: 1.0,
            column_chance: 1.0,
            flowstone_chance: 1.0,
            ..FormationConfig::default()
        };

        // Count air voxels before
        let air_before: usize = field
            .samples
            .iter()
            .filter(|s| s.density <= 0.0)
            .count();

        place_formations(
            &mut field,
            &config,
            Vec3::ZERO,
            42,
            12345,
        );

        // Count air voxels after — should have fewer (some filled by formations)
        let air_after: usize = field
            .samples
            .iter()
            .filter(|s| s.density <= 0.0)
            .count();

        assert!(
            air_after < air_before,
            "Formations should fill some air voxels: before={air_before}, after={air_after}"
        );
    }

    #[test]
    fn test_disabled_formations_noop() {
        let mut field = make_cave_field(17, 3, 13);
        let original: Vec<f32> = field.samples.iter().map(|s| s.density).collect();

        let config = FormationConfig {
            enabled: false,
            ..FormationConfig::default()
        };
        place_formations(&mut field, &config, Vec3::ZERO, 42, 12345);

        let after: Vec<f32> = field.samples.iter().map(|s| s.density).collect();
        assert_eq!(original, after, "Disabled formations should not modify field");
    }

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

    #[test]
    fn test_formations_deterministic() {
        let config = FormationConfig {
            enabled: true,
            placement_threshold: 0.0,
            stalactite_chance: 1.0,
            stalagmite_chance: 1.0,
            ..FormationConfig::default()
        };

        let mut field1 = make_cave_field(17, 3, 13);
        place_formations(&mut field1, &config, Vec3::ZERO, 42, 12345);

        let mut field2 = make_cave_field(17, 3, 13);
        place_formations(&mut field2, &config, Vec3::ZERO, 42, 12345);

        let d1: Vec<f32> = field1.samples.iter().map(|s| s.density).collect();
        let d2: Vec<f32> = field2.samples.iter().map(|s| s.density).collect();
        assert_eq!(d1, d2, "Same seeds should produce identical formations");
    }
}
