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
    MegaColumn {
        x: usize,
        z: usize,
        floor_y: usize,
        ceil_y: usize,
        base_radius: f32,
        material: Material,
    },
    Drapery {
        anchor_x: usize,
        anchor_y: usize,
        anchor_z: usize,
        direction_x: f32,
        direction_z: f32,
        length: f32,
        material: Material,
    },
    RimstoneDam {
        anchor_x: usize,
        anchor_y: usize,
        anchor_z: usize,
        slope_x: f32,
        slope_z: f32,
        dam_height: f32,
        pool_depth: f32,
        material: Material,
    },
    CaveShield {
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

    // Step B2: Material preference gate — formations cluster in carbonate rocks
    let mut rng = ChaCha8Rng::seed_from_u64(chunk_seed.wrapping_add(0xF0E4A710));
    let surfaces: Vec<SurfacePoint> = surfaces
        .into_iter()
        .filter(|sp| {
            if sp.material.is_carbonate() {
                true // limestone/marble: 100%
            } else if sp.material == Material::Sandstone {
                rng.gen::<f32>() < 0.15 // secondary calcite deposits
            } else {
                rng.gen::<f32>() < 0.03 // granite/basalt/slate/ores: very rare
            }
        })
        .collect();

    if surfaces.is_empty() {
        return;
    }

    // Step C: RNG filter + Step D: Column detection + Step E: Dimension rolls
    // (rng continues from material gate — deterministic sequence preserved)
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
                if gap < config.min_air_gap {
                    continue;
                }

                // Mega-Column: large gap + separate chance roll
                if gap >= config.mega_column_min_gap
                    && rng.gen::<f32>() < config.mega_column_chance
                {
                    let base_radius = rng
                        .gen_range(config.mega_column_radius_min..=config.mega_column_radius_max);
                    formations.push(Formation::MegaColumn {
                        x,
                        z,
                        floor_y: floor_pt.y,
                        ceil_y: ceil_pt.y,
                        base_radius,
                        material: ceil_pt.material,
                    });
                    used_ceiling.insert((ceil_pt.x, ceil_pt.y, ceil_pt.z), true);
                    used_floor.insert((floor_pt.x, floor_pt.y, floor_pt.z), true);
                } else if gap <= config.column_max_gap
                    && rng.gen::<f32>() < config.column_chance
                {
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

    // Drapery placement: ceiling surfaces with slope transitions
    for sp in &ceiling_surfaces {
        if used_ceiling.contains_key(&(sp.x, sp.y, sp.z)) {
            continue;
        }
        if rng.gen::<f32>() >= config.drapery_chance {
            continue;
        }
        if let Some((dx, dz)) = detect_ceiling_slope(density, sp.x, sp.y, sp.z, size) {
            let length = rng.gen_range(config.drapery_length_min..=config.drapery_length_max);
            formations.push(Formation::Drapery {
                anchor_x: sp.x,
                anchor_y: sp.y,
                anchor_z: sp.z,
                direction_x: dx,
                direction_z: dz,
                length,
                material: sp.material,
            });
        }
    }

    // Rimstone Dam placement: floor surfaces with sufficient slope
    for sp in &floor_surfaces {
        if used_floor.contains_key(&(sp.x, sp.y, sp.z)) {
            continue;
        }
        if rng.gen::<f32>() >= config.rimstone_chance {
            continue;
        }
        if let Some((grad_x, grad_z, magnitude)) =
            detect_floor_slope(density, sp.x, sp.y, sp.z, size)
        {
            if magnitude >= config.rimstone_min_slope {
                let dam_height =
                    rng.gen_range(config.rimstone_dam_height_min..=config.rimstone_dam_height_max);
                let norm = magnitude.max(0.001);
                formations.push(Formation::RimstoneDam {
                    anchor_x: sp.x,
                    anchor_y: sp.y,
                    anchor_z: sp.z,
                    slope_x: grad_x / norm,
                    slope_z: grad_z / norm,
                    dam_height,
                    pool_depth: config.rimstone_pool_depth,
                    material: sp.material,
                });
            }
        }
    }

    // Cave Shield placement: wall surfaces
    for sp in &wall_surfaces {
        if rng.gen::<f32>() >= config.shield_chance {
            continue;
        }
        let (nx, ny, nz) = find_wall_normal_3d(density, sp.x, sp.y, sp.z, size);
        // Only place shields on mostly-horizontal walls (avoid floor/ceiling normals)
        if ny.abs() > 0.7 {
            continue;
        }
        let radius = rng.gen_range(config.shield_radius_min..=config.shield_radius_max);
        let tilt_x = rng.gen_range(-config.shield_max_tilt..=config.shield_max_tilt);
        let tilt_y = rng.gen_range(-config.shield_max_tilt..=config.shield_max_tilt);
        formations.push(Formation::CaveShield {
            anchor_x: sp.x,
            anchor_y: sp.y,
            anchor_z: sp.z,
            normal_x: nx,
            normal_y: ny,
            normal_z: nz,
            tilt_x,
            tilt_y,
            radius,
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
            Formation::MegaColumn {
                x,
                z,
                floor_y,
                ceil_y,
                base_radius,
                material,
            } => {
                write_mega_column(
                    density,
                    *x,
                    *z,
                    *floor_y,
                    *ceil_y,
                    *base_radius,
                    *material,
                    config,
                    &column_noise,
                    world_origin,
                    size,
                );
            }
            Formation::Drapery {
                anchor_x,
                anchor_y,
                anchor_z,
                direction_x,
                direction_z,
                length,
                material,
            } => {
                write_drapery(
                    density,
                    *anchor_x,
                    *anchor_y,
                    *anchor_z,
                    *direction_x,
                    *direction_z,
                    *length,
                    *material,
                    config,
                    size,
                );
            }
            Formation::RimstoneDam {
                anchor_x,
                anchor_y,
                anchor_z,
                slope_x,
                slope_z,
                dam_height,
                pool_depth,
                material,
            } => {
                write_rimstone_dam(
                    density,
                    *anchor_x,
                    *anchor_y,
                    *anchor_z,
                    *slope_x,
                    *slope_z,
                    *dam_height,
                    *pool_depth,
                    *material,
                    config,
                    size,
                );
            }
            Formation::CaveShield {
                anchor_x,
                anchor_y,
                anchor_z,
                normal_x,
                normal_y,
                normal_z,
                tilt_x,
                tilt_y,
                radius,
                material,
            } => {
                write_cave_shield(
                    density,
                    *anchor_x,
                    *anchor_y,
                    *anchor_z,
                    *normal_x,
                    *normal_y,
                    *normal_z,
                    *tilt_x,
                    *tilt_y,
                    *radius,
                    *material,
                    config,
                    size,
                    &mut rng,
                );
            }
        }
    }
}

/// Detect ceiling slope at a position by comparing ceiling Y at adjacent X/Z.
/// Returns slope direction (dx, dz) if slope delta >= 1.
fn detect_ceiling_slope(
    density: &DensityField,
    x: usize,
    y: usize,
    z: usize,
    size: usize,
) -> Option<(f32, f32)> {
    if x < 1 || x >= size - 1 || z < 1 || z >= size - 1 {
        return None;
    }

    // Find ceiling Y at a given (px, pz) by scanning upward from y
    let find_ceil = |px: usize, pz: usize| -> Option<usize> {
        for py in y..size.min(y + 6) {
            if density.get(px, py, pz).density > 0.0 {
                return Some(py);
            }
        }
        None
    };

    let cy = find_ceil(x, z)?;
    let cx_pos = find_ceil(x + 1, z).unwrap_or(cy);
    let cx_neg = find_ceil(x.saturating_sub(1), z).unwrap_or(cy);
    let cz_pos = find_ceil(x, z + 1).unwrap_or(cy);
    let cz_neg = find_ceil(x, z.saturating_sub(1)).unwrap_or(cy);

    let dx = cx_pos as f32 - cx_neg as f32;
    let dz = cz_pos as f32 - cz_neg as f32;

    let mag = (dx * dx + dz * dz).sqrt();
    if mag >= 1.0 {
        Some((dx / mag, dz / mag))
    } else {
        None
    }
}

/// Detect floor slope using central differences with wide sampling.
/// Returns (grad_x, grad_z, magnitude).
///
/// Uses a sampling radius of 3 voxels (±3) to catch gentle slopes that
/// change only 1 voxel over several horizontal steps. Also searches both
/// up and down from the anchor Y to find floor surfaces at neighbor columns.
fn detect_floor_slope(
    density: &DensityField,
    x: usize,
    y: usize,
    z: usize,
    size: usize,
) -> Option<(f32, f32, f32)> {
    let radius: usize = 3;
    if x < radius || x >= size - radius || z < radius || z >= size - radius {
        return None;
    }

    // Find nearest floor surface at (px, pz): solid with air above,
    // searching both up and down from anchor y.
    let find_floor = |px: usize, pz: usize| -> Option<usize> {
        let search_range = 4usize;
        for dy in 0..=search_range {
            // Check y - dy
            if dy <= y {
                let py = y - dy;
                if py + 1 < size
                    && density.get(px, py, pz).density > 0.0
                    && density.get(px, py + 1, pz).density <= 0.0
                {
                    return Some(py);
                }
            }
            // Check y + dy (skip dy=0 to avoid double-checking)
            if dy > 0 {
                let py = y + dy;
                if py + 1 < size
                    && density.get(px, py, pz).density > 0.0
                    && density.get(px, py + 1, pz).density <= 0.0
                {
                    return Some(py);
                }
            }
        }
        None
    };

    let fy = find_floor(x, z)?;
    let fx_pos = find_floor(x + radius, z).unwrap_or(fy);
    let fx_neg = find_floor(x - radius, z).unwrap_or(fy);
    let fz_pos = find_floor(x, z + radius).unwrap_or(fy);
    let fz_neg = find_floor(x, z - radius).unwrap_or(fy);

    let r = radius as f32;
    let grad_x = (fx_pos as f32 - fx_neg as f32) / (2.0 * r);
    let grad_z = (fz_pos as f32 - fz_neg as f32) / (2.0 * r);
    let magnitude = (grad_x * grad_x + grad_z * grad_z).sqrt();

    Some((grad_x, grad_z, magnitude))
}

/// Find 3D wall normal at a position using density gradient.
fn find_wall_normal_3d(
    density: &DensityField,
    x: usize,
    y: usize,
    z: usize,
    size: usize,
) -> (f32, f32, f32) {
    let sample = |px: usize, py: usize, pz: usize| -> f32 {
        if px < size && py < size && pz < size {
            density.get(px, py, pz).density
        } else {
            0.0
        }
    };

    let gx = sample(x + 1, y, z) - sample(x.saturating_sub(1), y, z);
    let gy = sample(x, y + 1, z) - sample(x, y.saturating_sub(1), z);
    let gz = sample(x, y, z + 1) - sample(x, y, z.saturating_sub(1));

    let mag = (gx * gx + gy * gy + gz * gz).sqrt();
    if mag > 0.001 {
        (gx / mag, gy / mag, gz / mag)
    } else {
        (1.0, 0.0, 0.0)
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

/// Write a mega-column: fat cylinder with noise modulation and ring bumps.
fn write_mega_column(
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
fn write_drapery(
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
fn write_rimstone_dam(
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
fn write_cave_shield(
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

    #[test]
    fn test_mega_column_fills_large_gap() {
        // Create a field with a large gap (>= 12 voxels between floor and ceiling)
        // size 32 gives us room: floor at y=3, ceiling at y=28 => gap = 24
        let size = 32;
        let mut field = make_cave_field(size, 3, 28);
        let config = FormationConfig {
            enabled: true,
            placement_threshold: 0.0,
            mega_column_chance: 1.0,
            mega_column_min_gap: 12,
            mega_column_radius_min: 2.0,
            mega_column_radius_max: 3.0,
            column_chance: 0.0, // disable regular columns
            stalactite_chance: 0.0,
            stalagmite_chance: 0.0,
            flowstone_chance: 0.0,
            drapery_chance: 0.0,
            rimstone_chance: 0.0,
            shield_chance: 0.0,
            ..FormationConfig::default()
        };

        let air_before: usize = field
            .samples
            .iter()
            .filter(|s| s.density <= 0.0)
            .count();

        place_formations(&mut field, &config, Vec3::ZERO, 42, 99999);

        let air_after: usize = field
            .samples
            .iter()
            .filter(|s| s.density <= 0.0)
            .count();

        assert!(
            air_after < air_before,
            "MegaColumn should fill some air voxels: before={air_before}, after={air_after}"
        );
    }

    #[test]
    fn test_drapery_thin_sheet() {
        // Create a field with a ceiling slope transition:
        // left half ceiling at y=12, right half at y=14
        let size = 17;
        let mut field = DensityField::new(size);
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let sample = field.get_mut(x, y, z);
                    let ceil = if x < size / 2 { 12 } else { 14 };
                    if y > 3 && y < ceil {
                        sample.density = -1.0;
                        sample.material = Material::Air;
                    } else {
                        sample.density = 1.0;
                        sample.material = Material::Limestone;
                    }
                }
            }
        }

        let config = FormationConfig {
            enabled: true,
            placement_threshold: 0.0,
            drapery_chance: 1.0,
            stalactite_chance: 0.0,
            stalagmite_chance: 0.0,
            column_chance: 0.0,
            mega_column_chance: 0.0,
            flowstone_chance: 0.0,
            rimstone_chance: 0.0,
            shield_chance: 0.0,
            ..FormationConfig::default()
        };

        let air_before: usize = field
            .samples
            .iter()
            .filter(|s| s.density <= 0.0)
            .count();

        place_formations(&mut field, &config, Vec3::ZERO, 42, 55555);

        let air_after: usize = field
            .samples
            .iter()
            .filter(|s| s.density <= 0.0)
            .count();

        assert!(
            air_after < air_before,
            "Drapery should fill some air voxels: before={air_before}, after={air_after}"
        );
    }

    #[test]
    fn test_rimstone_dam_on_slope() {
        // Create a field with a sloped floor: floor Y varies by X
        let size = 17;
        let mut field = DensityField::new(size);
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let sample = field.get_mut(x, y, z);
                    // Floor slopes: y_floor = 3 + x/3
                    let floor_y = 3 + x / 3;
                    let ceil_y = 14;
                    if y > floor_y && y < ceil_y {
                        sample.density = -1.0;
                        sample.material = Material::Air;
                    } else {
                        sample.density = 1.0;
                        sample.material = Material::Limestone;
                    }
                }
            }
        }

        let config = FormationConfig {
            enabled: true,
            placement_threshold: 0.0,
            rimstone_chance: 1.0,
            rimstone_min_slope: 0.1, // low threshold for sloped floor
            stalactite_chance: 0.0,
            stalagmite_chance: 0.0,
            column_chance: 0.0,
            mega_column_chance: 0.0,
            flowstone_chance: 0.0,
            drapery_chance: 0.0,
            shield_chance: 0.0,
            ..FormationConfig::default()
        };

        place_formations(&mut field, &config, Vec3::ZERO, 42, 77777);

        // Rimstone dams both fill (dam walls) and carve (basins), so check density changed
        let densities_before: Vec<f32> = make_cave_field(size, 3, 14)
            .samples
            .iter()
            .map(|s| s.density)
            .collect();
        let densities_after: Vec<f32> = field.samples.iter().map(|s| s.density).collect();
        assert_ne!(
            densities_before, densities_after,
            "RimstoneDam should modify the density field"
        );
    }

    #[test]
    fn test_rimstone_dam_gentle_slope() {
        // Gentle slope: 1 voxel rise per 4 horizontal steps (floor_y = 3 + x/4).
        // The old ±1 sampling missed this; the ±3 radius should detect it.
        let size = 17;
        let mut field = DensityField::new(size);
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let sample = field.get_mut(x, y, z);
                    let floor_y = 3 + x / 4;
                    let ceil_y = 14;
                    if y > floor_y && y < ceil_y {
                        sample.density = -1.0;
                        sample.material = Material::Air;
                    } else {
                        sample.density = 1.0;
                        sample.material = Material::Limestone;
                    }
                }
            }
        }

        let config = FormationConfig {
            enabled: true,
            placement_threshold: 0.0,
            rimstone_chance: 1.0,
            rimstone_min_slope: 0.05, // match new default
            stalactite_chance: 0.0,
            stalagmite_chance: 0.0,
            column_chance: 0.0,
            mega_column_chance: 0.0,
            flowstone_chance: 0.0,
            drapery_chance: 0.0,
            shield_chance: 0.0,
            ..FormationConfig::default()
        };

        place_formations(&mut field, &config, Vec3::ZERO, 42, 88888);

        // Verify density field was modified (dams were placed)
        let mut reference = DensityField::new(size);
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let sample = reference.get_mut(x, y, z);
                    let floor_y = 3 + x / 4;
                    let ceil_y = 14;
                    if y > floor_y && y < ceil_y {
                        sample.density = -1.0;
                        sample.material = Material::Air;
                    } else {
                        sample.density = 1.0;
                        sample.material = Material::Limestone;
                    }
                }
            }
        }
        let densities_before: Vec<f32> = reference.samples.iter().map(|s| s.density).collect();
        let densities_after: Vec<f32> = field.samples.iter().map(|s| s.density).collect();
        assert_ne!(
            densities_before, densities_after,
            "RimstoneDam should spawn on gentle slope (1 voxel per 4 horizontal)"
        );
    }

    #[test]
    fn test_cave_shield_disc() {
        // Create a field with wall surfaces (vertical solid/air boundary)
        let size = 17;
        let mut field = DensityField::new(size);
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let sample = field.get_mut(x, y, z);
                    // Vertical cave: air for x in 5..12
                    if x > 4 && x < 12 {
                        sample.density = -1.0;
                        sample.material = Material::Air;
                    } else {
                        sample.density = 1.0;
                        sample.material = Material::Limestone;
                    }
                }
            }
        }

        let config = FormationConfig {
            enabled: true,
            placement_threshold: 0.0,
            shield_chance: 1.0,
            shield_radius_min: 2.0,
            shield_radius_max: 3.0,
            stalactite_chance: 0.0,
            stalagmite_chance: 0.0,
            column_chance: 0.0,
            mega_column_chance: 0.0,
            flowstone_chance: 0.0,
            drapery_chance: 0.0,
            rimstone_chance: 0.0,
            ..FormationConfig::default()
        };

        let air_before: usize = field
            .samples
            .iter()
            .filter(|s| s.density <= 0.0)
            .count();

        place_formations(&mut field, &config, Vec3::ZERO, 42, 88888);

        let air_after: usize = field
            .samples
            .iter()
            .filter(|s| s.density <= 0.0)
            .count();

        assert!(
            air_after < air_before,
            "CaveShield should fill some air voxels: before={air_before}, after={air_after}"
        );
    }

    #[test]
    fn test_new_formations_deterministic() {
        let config = FormationConfig {
            enabled: true,
            placement_threshold: 0.0,
            stalactite_chance: 1.0,
            stalagmite_chance: 1.0,
            column_chance: 1.0,
            mega_column_chance: 1.0,
            flowstone_chance: 1.0,
            drapery_chance: 1.0,
            rimstone_chance: 1.0,
            shield_chance: 1.0,
            ..FormationConfig::default()
        };

        let mut field1 = make_cave_field(17, 3, 13);
        place_formations(&mut field1, &config, Vec3::ZERO, 42, 12345);

        let mut field2 = make_cave_field(17, 3, 13);
        place_formations(&mut field2, &config, Vec3::ZERO, 42, 12345);

        let d1: Vec<f32> = field1.samples.iter().map(|s| s.density).collect();
        let d2: Vec<f32> = field2.samples.iter().map(|s| s.density).collect();
        assert_eq!(
            d1, d2,
            "Same seeds should produce identical formations with all types enabled"
        );
    }
}
