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

use crate::config::{AureoleConfig, GroundwaterConfig};
use crate::groundwater::ambient_moisture;
use crate::manifest::ChangeManifest;
use crate::util::{FACE_OFFSETS, sample_material};
use crate::{Bottleneck, PhaseDiagnostics, ResourceCensus, TransformEntry};

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

    // Contact metamorphism / aureole code removed — will be redesigned.

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

    result.channels_eroded = erosion_count + ambient_erosion_count;

    // Build transform log
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

fn compute_aureole_bottlenecks(census: &ResourceCensus, _heat_map: &HeatMap) -> Vec<Bottleneck> {
    let mut bottlenecks = Vec::new();

    if census.water.cell_count == 0 {
        bottlenecks.push(Bottleneck {
            severity: 0.5,
            description: "No water detected \u{2014} erosion needs moisture".into(),
        });
    }

    bottlenecks
}
