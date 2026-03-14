//! Cave formation placement: stalactites, stalagmites, columns, and flowstone shelves.
//!
//! Scans the density field for ceiling/floor/wall surfaces adjacent to air,
//! uses noise + seeded RNG for sparse deterministic placement, then writes
//! cone/cylinder/shelf shapes as positive density into air voxels.
//! Formations inherit the host rock material from their anchor surface.

mod surface;
mod shapes;

use std::collections::HashMap;

use glam::Vec3;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use voxel_noise::simplex::Simplex3D;
use voxel_noise::NoiseSource;

use crate::config::FormationConfig;
use crate::density::DensityField;
use crate::pools::{FluidSeed, PoolFluid};

use surface::{
    SurfaceKind, SurfacePoint,
    detect_surfaces, detect_ceiling_slope, detect_floor_slope,
    find_wall_normal, find_wall_normal_3d,
    measure_air_down, measure_air_up, measure_solid_down, measure_air_extent,
};
use shapes::{
    write_cone, write_column, write_flowstone, write_mega_column,
    write_drapery, write_rimstone_dam, write_cave_shield, write_cauldron,
};

/// A formation to be written into the density field.
#[derive(Debug)]
enum Formation {
    Stalactite {
        anchor_x: usize,
        anchor_y: usize,
        anchor_z: usize,
        length: f32,
        radius: f32,
        material: voxel_core::material::Material,
    },
    Stalagmite {
        anchor_x: usize,
        anchor_y: usize,
        anchor_z: usize,
        length: f32,
        radius: f32,
        material: voxel_core::material::Material,
    },
    Column {
        x: usize,
        z: usize,
        floor_y: usize,
        ceil_y: usize,
        radius: f32,
        material: voxel_core::material::Material,
    },
    Flowstone {
        anchor_x: usize,
        anchor_y: usize,
        anchor_z: usize,
        normal_x: f32,
        normal_z: f32,
        length: f32,
        thickness: f32,
        material: voxel_core::material::Material,
    },
    MegaColumn {
        x: usize,
        z: usize,
        floor_y: usize,
        ceil_y: usize,
        base_radius: f32,
        material: voxel_core::material::Material,
    },
    Drapery {
        anchor_x: usize,
        anchor_y: usize,
        anchor_z: usize,
        direction_x: f32,
        direction_z: f32,
        length: f32,
        material: voxel_core::material::Material,
    },
    RimstoneDam {
        anchor_x: usize,
        anchor_y: usize,
        anchor_z: usize,
        slope_x: f32,
        slope_z: f32,
        dam_height: f32,
        pool_depth: f32,
        material: voxel_core::material::Material,
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
        material: voxel_core::material::Material,
    },
    Cauldron {
        anchor_x: usize,
        anchor_y: usize,
        anchor_z: usize,
        radius: f32,
        depth: f32,
        lip_height: f32,
        rim_stalagmite_count: u32,
        rim_stalagmite_scale: f32,
        floor_noise: f32,
        material: voxel_core::material::Material,
        fluid_type: Option<PoolFluid>,
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
    chunk_coord: (i32, i32, i32),
) -> Vec<FluidSeed> {
    if !config.enabled {
        return Vec::new();
    }

    let mut fluid_seeds = Vec::new();

    let size = density.size;
    if size < 4 {
        return fluid_seeds;
    }

    // Step A: Detect surfaces
    let surfaces = detect_surfaces(density, size);
    if surfaces.is_empty() {
        return fluid_seeds;
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
        return fluid_seeds;
    }

    // Step B2: Material preference gate — formations cluster in carbonate rocks
    let mut rng = ChaCha8Rng::seed_from_u64(chunk_seed.wrapping_add(0xF0E4A710));
    let surfaces: Vec<SurfacePoint> = surfaces
        .into_iter()
        .filter(|sp| {
            if sp.material.is_carbonate() {
                true // limestone/marble: 100%
            } else if sp.material == voxel_core::material::Material::Sandstone {
                rng.gen::<f32>() < 0.15 // secondary calcite deposits
            } else {
                rng.gen::<f32>() < 0.03 // granite/basalt/slate/ores: very rare
            }
        })
        .collect();

    if surfaces.is_empty() {
        return fluid_seeds;
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
                } else if rng.gen::<f32>() < config.column_chance {
                    // Normal column — any gap >= min_air_gap qualifies.
                    // No max-gap cap: write_column fills floor_y+1..ceil_y-1 at any height.
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

    // Cauldron placement: floor surfaces with enough solid below
    for sp in &floor_surfaces {
        if used_floor.contains_key(&(sp.x, sp.y, sp.z)) {
            continue;
        }
        if rng.gen::<f32>() >= config.cauldron_chance {
            continue;
        }
        let radius = rng.gen_range(config.cauldron_radius_min..=config.cauldron_radius_max);
        let r_ceil = radius.ceil() as usize;
        // Check chunk edge clearance
        if sp.x < r_ceil + 1 || sp.x + r_ceil + 1 >= size
            || sp.z < r_ceil + 1 || sp.z + r_ceil + 1 >= size
        {
            continue;
        }
        // Check air above for lip clearance
        let air_up = measure_air_up(density, sp.x, sp.y, sp.z, size);
        if air_up < 3 {
            continue;
        }
        // Check enough solid below to carve into
        let solid_down = measure_solid_down(density, sp.x, sp.y, sp.z);
        if (solid_down as f32) < config.cauldron_depth + 1.0 {
            continue;
        }
        let rim_count = rng.gen_range(config.cauldron_rim_stalagmite_count_min..=config.cauldron_rim_stalagmite_count_max);
        // Roll fluid type: water first, then lava if water fails
        let cauldron_fluid = if rng.gen::<f32>() < config.cauldron_water_chance {
            Some(PoolFluid::Water)
        } else if rng.gen::<f32>() < config.cauldron_lava_chance {
            Some(PoolFluid::Lava)
        } else {
            None
        };
        formations.push(Formation::Cauldron {
            anchor_x: sp.x,
            anchor_y: sp.y,
            anchor_z: sp.z,
            radius,
            depth: config.cauldron_depth,
            lip_height: config.cauldron_lip_height,
            rim_stalagmite_count: rim_count,
            rim_stalagmite_scale: config.cauldron_rim_stalagmite_scale,
            floor_noise: config.cauldron_floor_noise,
            material: voxel_core::material::Material::Limestone,
            fluid_type: cauldron_fluid,
        });
        used_floor.insert((sp.x, sp.y, sp.z), true);
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
            Formation::Cauldron {
                anchor_x,
                anchor_y,
                anchor_z,
                radius,
                depth,
                lip_height,
                rim_stalagmite_count,
                rim_stalagmite_scale,
                floor_noise,
                material,
                fluid_type,
            } => {
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("D:/cargo-target/sentinel.log") {
                    use std::io::Write;
                    let _ = writeln!(f,
                        "[SENTINEL] Cauldron formation detected at ({},{},{}) chunk {:?} radius={:.1} depth={:.1} fluid={:?}",
                        anchor_x, anchor_y, anchor_z, chunk_coord, radius, depth, fluid_type
                    );
                }
                write_cauldron(
                    density,
                    *anchor_x,
                    *anchor_y,
                    *anchor_z,
                    *radius,
                    *depth,
                    *lip_height,
                    *rim_stalagmite_count,
                    *rim_stalagmite_scale,
                    *floor_noise,
                    *material,
                    config,
                    world_seed,
                    size,
                );
                if let Some(ft) = fluid_type {
                    generate_cauldron_fluid_seeds(
                        &mut fluid_seeds,
                        *anchor_x,
                        *anchor_y,
                        *anchor_z,
                        *radius,
                        *depth,
                        *lip_height,
                        config.cauldron_wall_inset,
                        config.cauldron_floor_inset,
                        *ft,
                        chunk_coord,
                        size,
                    );
                }
            }
        }
    }

    fluid_seeds
}

/// Generate fluid seeds to fill a cauldron basin with non-source fluid.
fn generate_cauldron_fluid_seeds(
    seeds: &mut Vec<FluidSeed>,
    anchor_x: usize,
    anchor_y: usize,
    anchor_z: usize,
    radius: f32,
    depth: f32,
    lip_height: f32,
    wall_inset: f32,
    floor_inset: i32,
    fluid_type: PoolFluid,
    chunk_coord: (i32, i32, i32),
    size: usize,
) {
    // Inset fill region: keep fluid wall_inset cells away from walls
    // to avoid placing fluid in boundary cells with mixed solid/air corners.
    let fill_radius = radius - wall_inset;
    if fill_radius < 1.0 {
        return; // too small to hold fluid safely
    }
    let r_ceil = fill_radius.ceil() as i32;
    let ay = anchor_y as f32;
    let lip_ceil = lip_height.ceil() as i32;

    for dz in -r_ceil..=r_ceil {
        for dx in -r_ceil..=r_ceil {
            let dist_h = ((dx * dx + dz * dz) as f32).sqrt();
            if dist_h >= fill_radius {
                continue;
            }
            let gx = anchor_x as i32 + dx;
            let gz = anchor_z as i32 + dz;
            if gx < 0 || gx >= size as i32 || gz < 0 || gz >= size as i32 {
                continue;
            }

            // Same profile as write_cauldron Phase 1 carve (use original radius for depth profile)
            let rim_factor = dist_h / radius;
            let carve_depth_at_r = depth * (1.0 - rim_factor.powf(0.3).min(1.0));
            let bottom_y = (ay - carve_depth_at_r).floor() as i32;

            // Fill from basin bottom+floor_inset+lip_ceil up to anchor_y+lip_ceil
            // floor_inset raises fill above floor boundary cells
            // lip_ceil raises fill into the lip ring area
            let min_y = (bottom_y + floor_inset + lip_ceil).max(0);
            let max_y = anchor_y as i32 + lip_ceil; // exclusive
            for iy in min_y..max_y {
                if iy >= size as i32 {
                    continue;
                }
                seeds.push(FluidSeed {
                    chunk: chunk_coord,
                    lx: gx as u8,
                    ly: iy as u8,
                    lz: gz as u8,
                    fluid_type,
                    is_source: false,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FormationConfig;
    use voxel_core::material::Material;

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
    fn test_place_formations_no_panic() {
        let mut field = make_cave_field(17, 3, 13);
        let config = FormationConfig::default();
        place_formations(
            &mut field,
            &config,
            Vec3::ZERO,
            42,
            12345,
            (0, 0, 0),
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
            (0, 0, 0),
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
        place_formations(&mut field, &config, Vec3::ZERO, 42, 12345, (0, 0, 0));

        let after: Vec<f32> = field.samples.iter().map(|s| s.density).collect();
        assert_eq!(original, after, "Disabled formations should not modify field");
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
        place_formations(&mut field1, &config, Vec3::ZERO, 42, 12345, (0, 0, 0));

        let mut field2 = make_cave_field(17, 3, 13);
        place_formations(&mut field2, &config, Vec3::ZERO, 42, 12345, (0, 0, 0));

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

        place_formations(&mut field, &config, Vec3::ZERO, 42, 99999, (0, 0, 0));

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

        place_formations(&mut field, &config, Vec3::ZERO, 42, 55555, (0, 0, 0));

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

        place_formations(&mut field, &config, Vec3::ZERO, 42, 77777, (0, 0, 0));

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

        place_formations(&mut field, &config, Vec3::ZERO, 42, 88888, (0, 0, 0));

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

        place_formations(&mut field, &config, Vec3::ZERO, 42, 88888, (0, 0, 0));

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
        place_formations(&mut field1, &config, Vec3::ZERO, 42, 12345, (0, 0, 0));

        let mut field2 = make_cave_field(17, 3, 13);
        place_formations(&mut field2, &config, Vec3::ZERO, 42, 12345, (0, 0, 0));

        let d1: Vec<f32> = field1.samples.iter().map(|s| s.density).collect();
        let d2: Vec<f32> = field2.samples.iter().map(|s| s.density).collect();
        assert_eq!(
            d1, d2,
            "Same seeds should produce identical formations with all types enabled"
        );
    }

    #[test]
    fn test_column_medium_gap() {
        // Cave with gap of 8 voxels (floor_y=3, ceil_y=12, gap=8)
        // Well within normal column range, should produce columns.
        let mut field = make_cave_field(17, 3, 12);
        let config = FormationConfig {
            enabled: true,
            placement_threshold: 0.0,
            column_chance: 1.0,
            mega_column_chance: 0.0,
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

        place_formations(&mut field, &config, Vec3::ZERO, 42, 11111, (0, 0, 0));

        let air_after: usize = field
            .samples
            .iter()
            .filter(|s| s.density <= 0.0)
            .count();

        assert!(
            air_after < air_before,
            "Normal column should fill air in medium-gap cave: before={air_before}, after={air_after}"
        );
    }

    #[test]
    fn test_column_fallback_large_gap() {
        // Cave with gap of 15 voxels (floor_y=3, ceil_y=19, gap=15)
        // This gap exceeds the old column_max_gap (10) and qualifies for
        // mega-columns, but mega_column_chance=0 so the mega roll always fails.
        // Before the fix, this would fall to `else if gap <= 10` which is false,
        // producing zero formations. After the fix, it falls through to a normal column.
        let size = 24;
        let mut field = make_cave_field(size, 3, 19);
        let config = FormationConfig {
            enabled: true,
            placement_threshold: 0.0,
            column_chance: 1.0,
            mega_column_chance: 0.0, // mega roll always fails
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

        place_formations(&mut field, &config, Vec3::ZERO, 42, 22222, (0, 0, 0));

        let air_after: usize = field
            .samples
            .iter()
            .filter(|s| s.density <= 0.0)
            .count();

        assert!(
            air_after < air_before,
            "Normal column should spawn in large-gap cave when mega-column fails: before={air_before}, after={air_after}"
        );
    }

    #[test]
    fn test_cauldron_carves_basin() {
        // Create a field with thick floor (y=0..8 solid, y=9..16 air)
        let size = 17;
        let mut field = make_cave_field(size, 8, 15);
        let config = FormationConfig {
            enabled: true,
            placement_threshold: 0.0,
            cauldron_chance: 1.0,
            cauldron_lip_height: 0.0,           // disable lip to isolate carving
            cauldron_rim_stalagmite_count_min: 0, // disable stalagmites
            cauldron_rim_stalagmite_count_max: 0,
            stalactite_chance: 0.0,
            stalagmite_chance: 0.0,
            column_chance: 0.0,
            flowstone_chance: 0.0,
            drapery_chance: 0.0,
            rimstone_chance: 0.0,
            shield_chance: 0.0,
            mega_column_chance: 0.0,
            ..FormationConfig::default()
        };

        // Count solid voxels at/below floor before
        let solid_before: usize = field
            .samples
            .iter()
            .filter(|s| s.density > 0.0)
            .count();

        place_formations(&mut field, &config, Vec3::ZERO, 42, 55555, (0, 0, 0));

        let solid_after: usize = field
            .samples
            .iter()
            .filter(|s| s.density > 0.0)
            .count();

        // Cauldron carves a basin, so solid count should decrease
        assert!(
            solid_after < solid_before,
            "Cauldron should carve some solid voxels: before={solid_before}, after={solid_after}"
        );
    }

    #[test]
    fn test_cauldron_fluid_seeds_inset_from_walls_and_floor() {
        use crate::pools::PoolFluid;

        let mut seeds = Vec::new();
        let anchor_x = 8;
        let anchor_y = 8;
        let anchor_z = 8;
        let radius = 4.0;
        let depth = 3.0;
        let size = 17;

        generate_cauldron_fluid_seeds(
            &mut seeds,
            anchor_x,
            anchor_y,
            anchor_z,
            radius,
            depth,
            0.8, // lip_height
            1.0, // wall_inset
            1,   // floor_inset
            PoolFluid::Water,
            (0, 0, 0),
            size,
        );

        assert!(!seeds.is_empty(), "Should generate some fluid seeds");

        let lip_ceil = 1i32; // ceil(0.8) = 1

        for seed in seeds.iter() {
            let dx = seed.lx as f32 - anchor_x as f32;
            let dz = seed.lz as f32 - anchor_z as f32;
            let dist_h = (dx * dx + dz * dz).sqrt();

            // Fluid should be inset: at least wall_inset cells from basin wall
            assert!(
                dist_h < radius - 1.0,
                "Seed at ({},{},{}) dist_h={:.2} should be < fill_radius={:.2}",
                seed.lx, seed.ly, seed.lz, dist_h, radius - 1.0,
            );

            // Fluid should be raised above floor boundary + lip_ceil
            let rim_factor = dist_h / radius;
            let carve_depth_at_r = depth * (1.0 - rim_factor.powf(0.3).min(1.0));
            let bottom_y = (anchor_y as f32 - carve_depth_at_r).floor() as i32;
            assert!(
                (seed.ly as i32) >= bottom_y + 1 + lip_ceil,
                "Seed y={} should be >= bottom_y+floor_inset+lip_ceil={} (floor inset + lip raise)",
                seed.ly, bottom_y + 1 + lip_ceil,
            );
        }
    }

    #[test]
    fn test_cauldron_fluid_seeds_small_radius_skipped() {
        use crate::pools::PoolFluid;

        let mut seeds = Vec::new();
        // radius=1.5 → fill_radius=0.5 < 1.0 → should skip
        generate_cauldron_fluid_seeds(
            &mut seeds, 8, 8, 8, 1.5, 2.0, 0.8, 1.0, 1, PoolFluid::Water, (0, 0, 0), 17,
        );
        assert!(seeds.is_empty(), "Small cauldron (radius<2) should produce no seeds");
    }
}
