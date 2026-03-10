//! Phase 2: "The Aureole" — 100,000 years.
//!
//! Contact metamorphism around heat sources (lava, kimberlite).
//! Water erosion along fluid pathways.

use std::collections::{HashMap, HashSet};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use voxel_core::density::DensityField;
use voxel_core::material::Material;
use voxel_core::stress::world_to_chunk_local;
use voxel_fluid::FluidSnapshot;


use crate::config::{AureoleConfig, GroundwaterConfig};
use crate::groundwater::ambient_moisture;
use crate::manifest::ChangeManifest;
use crate::util::{FACE_OFFSETS, sample_material};
use crate::{Bottleneck, PhaseDiagnostics, ResourceCensus, TransformEntry};

/// Count water cells within Chebyshev radius of a world position.
fn count_water_in_radius(
    fluid_snapshot: &FluidSnapshot,
    cx: i32, cy: i32, cz: i32,
    radius: i32,
) -> u32 {
    let cs = fluid_snapshot.chunk_size as i32;
    if cs == 0 { return 0; }
    let mut count = 0u32;
    let chunk_min = (
        (cx - radius).div_euclid(cs),
        (cy - radius).div_euclid(cs),
        (cz - radius).div_euclid(cs),
    );
    let chunk_max = (
        (cx + radius).div_euclid(cs),
        (cy + radius).div_euclid(cs),
        (cz + radius).div_euclid(cs),
    );
    for ckx in chunk_min.0..=chunk_max.0 {
        for cky in chunk_min.1..=chunk_max.1 {
            for ckz in chunk_min.2..=chunk_max.2 {
                if let Some(cells) = fluid_snapshot.chunks.get(&(ckx, cky, ckz)) {
                    for lz in 0..cs {
                        for ly in 0..cs {
                            for lx in 0..cs {
                                let wx = ckx * cs + lx;
                                let wy = cky * cs + ly;
                                let wz = ckz * cs + lz;
                                let dist = (wx - cx).abs().max((wy - cy).abs()).max((wz - cz).abs());
                                if dist <= radius {
                                    let idx = (lz * cs * cs + ly * cs + lx) as usize;
                                    if idx < cells.len() && cells[idx].level > 0.001 && cells[idx].fluid_type.is_water() {
                                        count += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    count
}

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
    pub diagnostics: PhaseDiagnostics,
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
                    if cell.level > 0.001 && cell.fluid_type.is_lava() {
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
    fluid_snapshot: &mut FluidSnapshot,
    heat_map: &HeatMap,
    _chunks: &[(i32, i32, i32)],
    chunk_size: usize,
    rng: &mut ChaCha8Rng,
    census: &ResourceCensus,
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
    let mut theoretical_max = 0u32;
    let radius = config.aureole_radius as i32;

    // --- Contact Metamorphism + Coal Maturation + Silicification ---
    let mut coal_matured = 0u32;
    let mut diamonds_formed = 0u32;
    let mut voxels_silicified = 0u32;

    if config.metamorphism_enabled && !heat_map.is_empty() {
        // Compute heat strength: count nearby heat sources within radius 3
        // Larger clusters sustain wider aureoles; isolated cells dissipate heat fast
        let heat_positions: HashSet<(i32, i32, i32)> = heat_map.iter().map(|h| h.pos).collect();
        let heat_strength: Vec<u32> = heat_map.iter().map(|h| {
            let (hx, hy, hz) = h.pos;
            let mut count = 0u32;
            for dx in -3i32..=3 {
                for dy in -3i32..=3 {
                    for dz in -3i32..=3 {
                        if dx == 0 && dy == 0 && dz == 0 { continue; }
                        if heat_positions.contains(&(hx + dx, hy + dy, hz + dz)) {
                            count += 1;
                        }
                    }
                }
            }
            count
        }).collect();

        // For each heat source, scan within effective radius using Chebyshev distance
        // Use a set to avoid duplicate transformations
        let mut transformed: std::collections::HashSet<(i32, i32, i32)> = std::collections::HashSet::new();

        for (hi, heat) in heat_map.iter().enumerate() {
            let (hx, hy, hz) = heat.pos;
            // Scale effective aureole radius by local heat clustering
            let neighbors = heat_strength[hi];
            let effective_radius: i32 = if neighbors >= 31 {
                radius // massive intrusion — full config radius
            } else if neighbors >= 16 {
                6
            } else if neighbors >= 6 {
                5
            } else if neighbors >= 1 {
                3
            } else {
                2 // isolated cell, heat dissipates fast
            };
            // Per-heat-source water proximity check for silicification
            let water_search_radius = effective_radius * config.silicification_water_radius_mult as i32;
            let local_water_count = count_water_in_radius(fluid_snapshot, hx, hy, hz, water_search_radius);
            let has_local_water = local_water_count > 0;
            // Magnitude scaling: more water = stronger silicification (0.0-1.0)
            let water_magnitude = (local_water_count as f32 / 20.0).min(1.0);
            for dx in -effective_radius..=effective_radius {
                for dy in -effective_radius..=effective_radius {
                    for dz in -effective_radius..=effective_radius {
                        let dist = dx.abs().max(dy.abs()).max(dz.abs());
                        if dist == 0 || dist > effective_radius {
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

                        // Count transformable materials as theoretical candidates
                        if matches!(mat, Material::Limestone | Material::Sandstone | Material::Slate | Material::Coal) {
                            theoretical_max += 1;
                        }

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
                                        Material::Limestone if rng.gen::<f32>() < config.contact_limestone_to_garnet_prob => Some(Material::Garnet),
                                        Material::Sandstone if rng.gen::<f32>() < config.contact_sandstone_to_granite_prob => Some(Material::Granite),
                                        Material::Slate if rng.gen::<f32>() < config.contact_slate_to_hornfels_prob => Some(Material::Hornfels),
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
                                else if config.silicification_enabled && has_local_water {
                                    match mat {
                                        Material::Limestone if rng.gen::<f32>() < config.silicification_limestone_prob * water_magnitude => {
                                            voxels_silicified += 1;
                                            Some(Material::Quartz)
                                        },
                                        Material::Sandstone if rng.gen::<f32>() < config.silicification_sandstone_prob * water_magnitude => {
                                            voxels_silicified += 1;
                                            Some(Material::Quartz)
                                        },
                                        _ => match mat {
                                            Material::Limestone if rng.gen::<f32>() < config.mid_limestone_to_diopside_prob => Some(Material::Diopside),
                                            Material::Sandstone if rng.gen::<f32>() < config.mid_sandstone_to_granite_prob => Some(Material::Granite),
                                            Material::Slate if rng.gen::<f32>() < config.mid_slate_to_hornfels_prob => Some(Material::Hornfels),
                                            _ => None,
                                        },
                                    }
                                } else {
                                    match mat {
                                        Material::Limestone if rng.gen::<f32>() < config.mid_limestone_to_diopside_prob => Some(Material::Diopside),
                                        Material::Sandstone if rng.gen::<f32>() < config.mid_sandstone_to_granite_prob => Some(Material::Granite),
                                        Material::Slate if rng.gen::<f32>() < config.mid_slate_to_hornfels_prob => Some(Material::Hornfels),
                                        _ => None,
                                    }
                                }
                            },
                            // Outer aureole (6+)
                            _ => {
                                // Silicification in outer aureole (weaker, requires water)
                                if config.silicification_enabled && has_local_water {
                                    match mat {
                                        Material::Limestone if rng.gen::<f32>() < config.silicification_limestone_prob * 0.5 * water_magnitude => {
                                            voxels_silicified += 1;
                                            Some(Material::Quartz)
                                        },
                                        Material::Sandstone if rng.gen::<f32>() < config.silicification_sandstone_prob * 0.5 * water_magnitude => {
                                            voxels_silicified += 1;
                                            Some(Material::Quartz)
                                        },
                                        _ => match mat {
                                            Material::Limestone if rng.gen::<f32>() < config.outer_limestone_to_marble_prob => Some(Material::Marble),
                                            Material::Slate if rng.gen::<f32>() < config.outer_slate_to_hornfels_prob => Some(Material::Hornfels),
                                            _ => None,
                                        },
                                    }
                                } else {
                                    match mat {
                                        Material::Limestone if rng.gen::<f32>() < config.outer_limestone_to_marble_prob => Some(Material::Marble),
                                        Material::Slate if rng.gen::<f32>() < config.outer_slate_to_hornfels_prob => Some(Material::Hornfels),
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
        // Collect water cell positions and levels first (avoids borrow conflict for drain)
        let water_cells: Vec<((i32, i32, i32), usize, f32, bool)> = fluid_snapshot.chunks.iter()
            .flat_map(|(&chunk_key, cells)| {
                let (cx, cy, cz) = chunk_key;
                (0..cs).flat_map(move |z| (0..cs).flat_map(move |y| (0..cs).map(move |x| {
                    let idx = z * cs * cs + y * cs + x;
                    let cell = &cells[idx];
                    let wx = cx * (cs as i32) + x as i32;
                    let wy = cy * (cs as i32) + y as i32;
                    let wz = cz * (cs as i32) + z as i32;
                    ((wx, wy, wz), idx, cell.level, cell.fluid_type.is_water() && cell.level > 0.001)
                })))
            })
            .filter(|(_, _, _, valid)| *valid)
            .collect();

        for &((wx, wy, wz), _idx, level, _) in &water_cells {
            // Scale erosion probability by water cell level (more water = stronger erosion)
            let level_factor = level.min(1.0);
            for &(dx, dy, dz) in &FACE_OFFSETS {
                let nx = wx + dx;
                let ny = wy + dy;
                let nz = wz + dz;
                if let Some(mat) = sample_material(density_fields, nx, ny, nz, chunk_size) {
                    if mat == Material::Limestone || mat == Material::Sandstone {
                        theoretical_max += 1;
                    }
                    if (mat == Material::Limestone || mat == Material::Sandstone)
                        && rng.gen::<f32>() < config.water_erosion_prob * level_factor
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

        // Drain water cells used for erosion (0.05 per voxel eroded, skip sources)
        if erosion_count > 0 {
            let drain_total = erosion_count as f32 * 0.05;
            let per_cell = drain_total / water_cells.len().max(1) as f32;
            for &((wx, wy, wz), _idx, _level, _) in &water_cells {
                let fck = (wx.div_euclid(cs as i32), wy.div_euclid(cs as i32), wz.div_euclid(cs as i32));
                let flx = wx.rem_euclid(cs as i32) as usize;
                let fly = wy.rem_euclid(cs as i32) as usize;
                let flz = wz.rem_euclid(cs as i32) as usize;
                let fidx = flz * cs * cs + fly * cs + flx;
                if let Some(cells) = fluid_snapshot.chunks.get_mut(&fck) {
                    if fidx < cells.len() && !cells[fidx].is_source && cells[fidx].level > 0.001 {
                        cells[fidx].level = (cells[fidx].level - per_cell).max(0.0);
                    }
                }
            }
        }
    }

    // --- Ambient Groundwater Erosion ---
    // Only limestone/sandstone dissolve in water (karst dissolution).
    // Granite/basalt/slate/marble don't erode — they're too hard.
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
                        if !matches!(mat, Material::Limestone | Material::Sandstone) {
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
                        if moisture > 0.0 && rng.gen::<f32>() < config.water_erosion_prob * moisture * groundwater.erosion_power * groundwater.soft_rock_mult {
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
    let mut conversions: std::collections::BTreeMap<(u8, u8), u32> = std::collections::BTreeMap::new();
    for c in &candidates {
        *conversions.entry((c.old_material as u8, c.new_material as u8)).or_insert(0) += 1;
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

    // --- Diagnostics ---
    let actual_output = candidates.len() as u32;
    result.diagnostics = PhaseDiagnostics {
        conversions,
        theoretical_max,
        actual_output,
        bottlenecks: compute_aureole_bottlenecks(census, heat_map),
    };

    result
}

fn compute_aureole_bottlenecks(census: &ResourceCensus, heat_map: &HeatMap) -> Vec<Bottleneck> {
    let mut bottlenecks = Vec::new();

    if heat_map.is_empty() {
        bottlenecks.push(Bottleneck {
            severity: 1.0,
            description: "No heat sources \u{2014} metamorphism requires lava or kimberlite".into(),
        });
    } else if heat_map.len() < 5 {
        bottlenecks.push(Bottleneck {
            severity: 0.6,
            description: format!(
                "Only {} heat sources \u{2014} more lava/kimberlite increases aureole coverage",
                heat_map.len()
            ),
        });
    }

    let exposed_limestone = census.exposed_surfaces_by_material.get(&(Material::Limestone as u8)).copied().unwrap_or(0);
    let exposed_sandstone = census.exposed_surfaces_by_material.get(&(Material::Sandstone as u8)).copied().unwrap_or(0);
    let target_rock = exposed_limestone + exposed_sandstone;
    if target_rock == 0 {
        bottlenecks.push(Bottleneck {
            severity: 0.8,
            description: "No exposed limestone or sandstone \u{2014} metamorphism targets are scarce".into(),
        });
    }

    let exposed_coal = census.exposed_surfaces_by_material.get(&(Material::Coal as u8)).copied().unwrap_or(0);
    if exposed_coal == 0 && !heat_map.is_empty() {
        bottlenecks.push(Bottleneck {
            severity: 0.3,
            description: "No coal near heat sources \u{2014} coal maturation to graphite/diamond needs proximity".into(),
        });
    }

    if census.water.cell_count == 0 {
        bottlenecks.push(Bottleneck {
            severity: 0.5,
            description: "No water detected \u{2014} silicification and erosion need moisture".into(),
        });
    }

    bottlenecks.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap_or(std::cmp::Ordering::Equal));
    bottlenecks.truncate(3);
    bottlenecks
}
