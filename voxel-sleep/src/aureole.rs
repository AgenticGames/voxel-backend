//! Phase 2: "The Aureole" — 100,000 years.
//!
//! Contact metamorphism around heat sources (lava, kimberlite).
//! Water erosion along fluid pathways.

use std::collections::HashMap;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use voxel_core::density::DensityField;
use voxel_core::material::Material;
use voxel_core::stress::world_to_chunk_local;
use voxel_fluid::FluidSnapshot;
use voxel_fluid::cell::FluidType;

use crate::config::{AureoleConfig, GroundwaterConfig};
use crate::groundwater::ambient_moisture;
use crate::manifest::ChangeManifest;
use crate::util::{FACE_OFFSETS, sample_material};
use crate::TransformEntry;

/// Type of heat source for coal maturation decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeatSourceType {
    Lava,
    Kimberlite,
}

/// A heat source with position and type.
#[derive(Debug, Clone)]
pub struct HeatSource {
    pub pos: (i32, i32, i32),
    pub source_type: HeatSourceType,
}

/// Heat source positions in world coordinates.
pub type HeatMap = Vec<HeatSource>;

/// Result of the aureole phase.
#[derive(Debug, Default)]
pub struct AureoleResult {
    pub voxels_metamorphosed: u32,
    pub channels_eroded: u32,
    pub coal_matured: u32,
    pub diamonds_formed: u32,
    pub voxels_silicified: u32,
    pub manifest: ChangeManifest,
    pub glimpse_chunk: Option<(i32, i32, i32)>,
    pub transform_log: Vec<TransformEntry>,
}

/// Build a heat map: collect all lava cell positions from fluid snapshot
/// plus kimberlite voxels from density fields.
pub fn build_heat_map(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    fluid_snapshot: &FluidSnapshot,
    chunks: &[(i32, i32, i32)],
    chunk_size: usize,
) -> HeatMap {
    let mut heat_sources: Vec<HeatSource> = Vec::new();
    let field_size = chunk_size + 1;

    // Lava cells from fluid snapshot
    let cs = fluid_snapshot.chunk_size;
    for (&chunk_key, cells) in &fluid_snapshot.chunks {
        let (cx, cy, cz) = chunk_key;
        for z in 0..cs {
            for y in 0..cs {
                for x in 0..cs {
                    let idx = z * cs * cs + y * cs + x;
                    let cell = &cells[idx];
                    if cell.level > 0.001 && cell.fluid_type == FluidType::Lava {
                        let wx = cx * (cs as i32) + x as i32;
                        let wy = cy * (cs as i32) + y as i32;
                        let wz = cz * (cs as i32) + z as i32;
                        heat_sources.push(HeatSource { pos: (wx, wy, wz), source_type: HeatSourceType::Lava });
                    }
                }
            }
        }
    }

    // Kimberlite voxels from density fields
    for &chunk_key in chunks {
        let (cx, cy, cz) = chunk_key;
        let df = match density_fields.get(&chunk_key) {
            Some(df) => df,
            None => continue,
        };

        for lz in 0..field_size {
            for ly in 0..field_size {
                for lx in 0..field_size {
                    if df.get(lx, ly, lz).material == Material::Kimberlite {
                        let wx = cx * (chunk_size as i32) + lx as i32;
                        let wy = cy * (chunk_size as i32) + ly as i32;
                        let wz = cz * (chunk_size as i32) + lz as i32;
                        heat_sources.push(HeatSource { pos: (wx, wy, wz), source_type: HeatSourceType::Kimberlite });
                    }
                }
            }
        }
    }

    heat_sources
}

/// Execute Phase 2: contact metamorphism aureoles + water erosion.
pub fn apply_aureole(
    config: &AureoleConfig,
    groundwater: &GroundwaterConfig,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    fluid_snapshot: &FluidSnapshot,
    heat_map: &HeatMap,
    _chunks: &[(i32, i32, i32)],
    chunk_size: usize,
    rng: &mut ChaCha8Rng,
) -> AureoleResult {
    let mut result = AureoleResult::default();

    struct Candidate {
        chunk_key: (i32, i32, i32),
        lx: usize, ly: usize, lz: usize,
        old_material: Material,
        density: f32,
        new_material: Material,
    }

    let mut candidates: Vec<Candidate> = Vec::new();
    let radius = config.aureole_radius as i32;

    // --- Contact Metamorphism + Coal Maturation + Silicification ---
    let mut coal_matured = 0u32;
    let mut diamonds_formed = 0u32;
    let mut voxels_silicified = 0u32;

    // Check if water exists anywhere in the fluid snapshot (for silicification)
    let has_water_in_snapshot = fluid_snapshot.chunks.values().any(|cells| {
        cells.iter().any(|cell| cell.level > 0.001 && cell.fluid_type == FluidType::Water)
    });

    if config.metamorphism_enabled && !heat_map.is_empty() {
        // For each heat source, scan within aureole_radius using Chebyshev distance
        // Use a set to avoid duplicate transformations
        let mut transformed: std::collections::HashSet<(i32, i32, i32)> = std::collections::HashSet::new();

        for heat in heat_map {
            let (hx, hy, hz) = heat.pos;
            for dx in -radius..=radius {
                for dy in -radius..=radius {
                    for dz in -radius..=radius {
                        let dist = dx.abs().max(dy.abs()).max(dz.abs());
                        if dist == 0 || dist > radius {
                            continue;
                        }

                        let wx = hx + dx;
                        let wy = hy + dy;
                        let wz = hz + dz;

                        if transformed.contains(&(wx, wy, wz)) {
                            continue;
                        }

                        let mat = match sample_material(density_fields, wx, wy, wz, chunk_size) {
                            Some(m) => m,
                            None => continue,
                        };

                        let new_mat = match dist {
                            // Contact zone (0-2)
                            d if d <= 2 => {
                                // Coal maturation: Coal → Diamond (kimberlite contact, dist <=1) or Coal → Graphite
                                if config.coal_maturation_enabled && mat == Material::Coal {
                                    if heat.source_type == HeatSourceType::Kimberlite && d <= 1 && rng.gen::<f32>() < config.graphite_to_diamond_prob {
                                        diamonds_formed += 1;
                                        Some(Material::Diamond)
                                    } else if rng.gen::<f32>() < config.coal_to_graphite_prob {
                                        coal_matured += 1;
                                        Some(Material::Graphite)
                                    } else {
                                        None
                                    }
                                } else {
                                    match mat {
                                        Material::Limestone if rng.gen::<f32>() < config.contact_limestone_to_marble_prob => Some(Material::Marble),
                                        Material::Sandstone if rng.gen::<f32>() < config.contact_sandstone_to_granite_prob => Some(Material::Granite),
                                        _ => None,
                                    }
                                }
                            },
                            // Mid aureole (3-5)
                            d if d <= 5 => {
                                // Coal maturation in mid aureole
                                if config.coal_maturation_enabled && mat == Material::Coal && rng.gen::<f32>() < config.coal_to_graphite_mid_prob {
                                    coal_matured += 1;
                                    Some(Material::Graphite)
                                }
                                // Silicification (requires water)
                                else if config.silicification_enabled && has_water_in_snapshot {
                                    match mat {
                                        Material::Limestone if rng.gen::<f32>() < config.silicification_limestone_prob => {
                                            voxels_silicified += 1;
                                            Some(Material::Quartz)
                                        },
                                        Material::Sandstone if rng.gen::<f32>() < config.silicification_sandstone_prob => {
                                            voxels_silicified += 1;
                                            Some(Material::Quartz)
                                        },
                                        _ => match mat {
                                            Material::Limestone if rng.gen::<f32>() < config.mid_limestone_to_marble_prob => Some(Material::Marble),
                                            Material::Sandstone if rng.gen::<f32>() < config.mid_sandstone_to_granite_prob => Some(Material::Granite),
                                            _ => None,
                                        },
                                    }
                                } else {
                                    match mat {
                                        Material::Limestone if rng.gen::<f32>() < config.mid_limestone_to_marble_prob => Some(Material::Marble),
                                        Material::Sandstone if rng.gen::<f32>() < config.mid_sandstone_to_granite_prob => Some(Material::Granite),
                                        _ => None,
                                    }
                                }
                            },
                            // Outer aureole (6+)
                            _ => {
                                // Silicification in outer aureole (weaker, requires water)
                                if config.silicification_enabled && has_water_in_snapshot {
                                    match mat {
                                        Material::Limestone if rng.gen::<f32>() < config.silicification_limestone_prob * 0.5 => {
                                            voxels_silicified += 1;
                                            Some(Material::Quartz)
                                        },
                                        Material::Sandstone if rng.gen::<f32>() < config.silicification_sandstone_prob * 0.5 => {
                                            voxels_silicified += 1;
                                            Some(Material::Quartz)
                                        },
                                        _ => match mat {
                                            Material::Limestone if rng.gen::<f32>() < config.outer_limestone_to_marble_prob => Some(Material::Marble),
                                            Material::Slate if rng.gen::<f32>() < config.outer_slate_to_marble_prob => Some(Material::Marble),
                                            _ => None,
                                        },
                                    }
                                } else {
                                    match mat {
                                        Material::Limestone if rng.gen::<f32>() < config.outer_limestone_to_marble_prob => Some(Material::Marble),
                                        Material::Slate if rng.gen::<f32>() < config.outer_slate_to_marble_prob => Some(Material::Marble),
                                        _ => None,
                                    }
                                }
                            },
                        };

                        if let Some(new_material) = new_mat {
                            transformed.insert((wx, wy, wz));
                            let (chunk_key, lx, ly, lz) = world_to_chunk_local(wx, wy, wz, chunk_size);
                            if let Some(df) = density_fields.get(&chunk_key) {
                                let sample = df.get(lx, ly, lz);
                                candidates.push(Candidate {
                                    chunk_key, lx, ly, lz,
                                    old_material: mat,
                                    density: sample.density,
                                    new_material,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    let metamorphism_count = candidates.len() as u32;

    // --- Water Erosion ---
    let mut erosion_count = 0u32;
    if config.water_erosion_enabled && !fluid_snapshot.chunks.is_empty() {
        let cs = fluid_snapshot.chunk_size;
        for (&chunk_key, cells) in &fluid_snapshot.chunks {
            let (cx, cy, cz) = chunk_key;
            for z in 0..cs {
                for y in 0..cs {
                    for x in 0..cs {
                        let idx = z * cs * cs + y * cs + x;
                        let cell = &cells[idx];
                        if cell.level <= 0.001 || cell.fluid_type != FluidType::Water {
                            continue;
                        }

                        let wx = cx * (cs as i32) + x as i32;
                        let wy = cy * (cs as i32) + y as i32;
                        let wz = cz * (cs as i32) + z as i32;

                        // Check solid neighbors for erodible host rocks
                        for &(dx, dy, dz) in &FACE_OFFSETS {
                            let nx = wx + dx;
                            let ny = wy + dy;
                            let nz = wz + dz;
                            if let Some(mat) = sample_material(density_fields, nx, ny, nz, chunk_size) {
                                if (mat == Material::Limestone || mat == Material::Sandstone)
                                    && rng.gen::<f32>() < config.water_erosion_prob
                                {
                                    let (ck, elx, ely, elz) = world_to_chunk_local(nx, ny, nz, chunk_size);
                                    if let Some(df) = density_fields.get(&ck) {
                                        let sample = df.get(elx, ely, elz);
                                        candidates.push(Candidate {
                                            chunk_key: ck,
                                            lx: elx, ly: ely, lz: elz,
                                            old_material: mat,
                                            density: sample.density,
                                            new_material: Material::Air,
                                        });
                                        erosion_count += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // --- Ambient Groundwater Erosion ---
    // Erodes any host rock that's air-adjacent; porosity_of() naturally
    // reduces the rate for hard rocks (granite 0.2, basalt 0.1).
    let mut ambient_erosion_count = 0u32;
    if config.water_erosion_enabled && groundwater.enabled {
        let field_size = chunk_size + 1;
        let chunk_keys: Vec<(i32, i32, i32)> = density_fields.keys().copied().collect();
        for chunk_key in chunk_keys {
            let (cx, cy, cz) = chunk_key;
            let df = match density_fields.get(&chunk_key) {
                Some(df) => df,
                None => continue,
            };

            for lz in 0..field_size {
                for ly in 0..field_size {
                    for lx in 0..field_size {
                        let sample = df.get(lx, ly, lz);
                        let mat = sample.material;
                        // Any host rock is eligible — porosity handles the rate
                        if !matches!(mat,
                            Material::Limestone | Material::Sandstone |
                            Material::Granite | Material::Basalt |
                            Material::Slate | Material::Marble
                        ) {
                            continue;
                        }

                        let wx = cx * (chunk_size as i32) + lx as i32;
                        let wy = cy * (chunk_size as i32) + ly as i32;
                        let wz = cz * (chunk_size as i32) + lz as i32;

                        // Must be air-adjacent
                        let mut has_air = false;
                        let mut has_air_below = false;
                        for &(dx, dy, dz) in &FACE_OFFSETS {
                            if let Some(neighbor) = sample_material(density_fields, wx + dx, wy + dy, wz + dz, chunk_size) {
                                if !neighbor.is_solid() {
                                    has_air = true;
                                    if dy == -1 { has_air_below = true; }
                                }
                            }
                        }
                        if !has_air {
                            continue;
                        }

                        let moisture = ambient_moisture(groundwater, wy, mat, has_air_below);
                        if moisture > 0.0 && rng.gen::<f32>() < config.water_erosion_prob * moisture {
                            candidates.push(Candidate {
                                chunk_key,
                                lx, ly, lz,
                                old_material: mat,
                                density: sample.density,
                                new_material: Material::Air,
                            });
                            ambient_erosion_count += 1;
                        }
                    }
                }
            }
        }
    }

    // --- Apply all candidates ---
    for c in &candidates {
        if let Some(df) = density_fields.get_mut(&c.chunk_key) {
            let sample = df.get_mut(c.lx, c.ly, c.lz);
            sample.material = c.new_material;
            if c.new_material == Material::Air {
                sample.density = -1.0;
            }
        }

        let new_density = if c.new_material == Material::Air { -1.0 } else { c.density };
        result.manifest.record_voxel_change(
            c.chunk_key, c.lx, c.ly, c.lz,
            c.old_material, c.density,
            c.new_material, new_density,
        );

        if result.glimpse_chunk.is_none() {
            result.glimpse_chunk = Some(c.chunk_key);
        }
    }

    result.voxels_metamorphosed = metamorphism_count;
    result.channels_eroded = erosion_count + ambient_erosion_count;
    result.coal_matured = coal_matured;
    result.diamonds_formed = diamonds_formed;
    result.voxels_silicified = voxels_silicified;

    // Build transform log
    if metamorphism_count > 0 {
        result.transform_log.push(TransformEntry {
            description: format!(
                "The Aureole \u{2014} 100,000 years: {} voxels metamorphosed around {} heat sources (radius {})",
                metamorphism_count, heat_map.len(), config.aureole_radius
            ),
            count: metamorphism_count,
        });
    }
    if coal_matured > 0 || diamonds_formed > 0 {
        result.transform_log.push(TransformEntry {
            description: format!(
                "The Aureole \u{2014} 100,000 years: {} coal \u{2192} graphite, {} \u{2192} diamond",
                coal_matured, diamonds_formed
            ),
            count: coal_matured + diamonds_formed,
        });
    }
    if voxels_silicified > 0 {
        result.transform_log.push(TransformEntry {
            description: format!(
                "The Aureole \u{2014} 100,000 years: {} voxels silicified (water-enabled)",
                voxels_silicified
            ),
            count: voxels_silicified,
        });
    }
    if erosion_count > 0 {
        result.transform_log.push(TransformEntry {
            description: format!("The Aureole \u{2014} 100,000 years: {} channels widened by water erosion", erosion_count),
            count: erosion_count,
        });
    }
    if ambient_erosion_count > 0 {
        result.transform_log.push(TransformEntry {
            description: format!("The Aureole \u{2014} 100,000 years: {} voxels eroded by ambient groundwater", ambient_erosion_count),
            count: ambient_erosion_count,
        });
    }

    result
}
