//! Phase 4: "The Deep Time" — 1,250,000 years.
//!
//! Supergene enrichment (water-driven ore concentration),
//! vein thickening, mature formations, and structural collapse.

use std::collections::HashMap;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use voxel_core::density::DensityField;
use voxel_core::material::Material;
use voxel_core::stress::{StressField, SupportField, world_to_chunk_local};
use voxel_fluid::FluidSnapshot;
use voxel_fluid::cell::FluidType;

use crate::aureole::HeatMap;
use crate::collapse::{apply_collapse, CollapseResult};
use crate::config::DeepTimeConfig;
use crate::manifest::ChangeManifest;
use crate::util::{FACE_OFFSETS, sample_material};
use crate::TransformEntry;

/// Result of the deep time phase.
#[derive(Debug)]
pub struct DeepTimeResult {
    pub voxels_enriched: u32,
    pub veins_thickened: u32,
    pub formations_grown: u32,
    pub supports_degraded: u32,
    pub collapses_triggered: u32,
    pub collapse_result: Option<CollapseResult>,
    pub manifest: ChangeManifest,
    pub glimpse_chunk: Option<(i32, i32, i32)>,
    pub transform_log: Vec<TransformEntry>,
}

impl Default for DeepTimeResult {
    fn default() -> Self {
        Self {
            voxels_enriched: 0,
            veins_thickened: 0,
            formations_grown: 0,
            supports_degraded: 0,
            collapses_triggered: 0,
            collapse_result: None,
            manifest: ChangeManifest::default(),
            glimpse_chunk: None,
            transform_log: Vec::new(),
        }
    }
}

/// Check if material is an ore (deposited by veins or natural).
fn is_ore(mat: Material) -> bool {
    matches!(mat,
        Material::Copper | Material::Iron | Material::Gold |
        Material::Tin | Material::Sulfide | Material::Quartz |
        Material::Pyrite | Material::Malachite
    )
}

/// Check if material is a host rock (transformable by enrichment).
fn is_host_rock(mat: Material) -> bool {
    matches!(mat,
        Material::Sandstone | Material::Limestone | Material::Granite |
        Material::Basalt | Material::Slate | Material::Marble
    )
}

/// Execute Phase 4: enrichment, thickening, formations, collapse.
pub fn apply_deeptime(
    config: &DeepTimeConfig,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    stress_fields: &mut HashMap<(i32, i32, i32), StressField>,
    support_fields: &mut HashMap<(i32, i32, i32), SupportField>,
    fluid_snapshot: &FluidSnapshot,
    _heat_map: &HeatMap,
    chunks: &[(i32, i32, i32)],
    chunk_size: usize,
    stress_config: &voxel_core::stress::StressConfig,
    rng: &mut ChaCha8Rng,
) -> DeepTimeResult {
    let mut result = DeepTimeResult::default();
    let field_size = chunk_size + 1;

    struct Candidate {
        chunk_key: (i32, i32, i32),
        lx: usize, ly: usize, lz: usize,
        old_material: Material,
        old_density: f32,
        new_material: Material,
        change_type: u8, // 0=enrichment, 1=thickening, 2=formation
    }

    let mut candidates: Vec<Candidate> = Vec::new();

    // --- Supergene Enrichment ---
    if config.enrichment_enabled && !fluid_snapshot.chunks.is_empty() {
        let cs = fluid_snapshot.chunk_size;
        let search_radius = config.enrichment_search_radius;
        let mut enrichment_per_chunk: HashMap<(i32, i32, i32), u32> = HashMap::new();

        for (&fchunk, cells) in &fluid_snapshot.chunks {
            let (cx, cy, cz) = fchunk;
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

                        // Check solid voxels 1-3 positions directly above (ceiling)
                        for dy_offset in 1..=3i32 {
                            let above_y = wy + dy_offset;
                            let above_mat = match sample_material(density_fields, wx, above_y, wz, chunk_size) {
                                Some(m) => m,
                                None => continue,
                            };

                            if !is_host_rock(above_mat) {
                                continue;
                            }

                            // Check if ore exists within search radius
                            let mut found_ore: Option<Material> = None;
                            'search: for sdx in -search_radius..=search_radius {
                                for sdy in -search_radius..=search_radius {
                                    for sdz in -search_radius..=search_radius {
                                        if sdx.abs() + sdy.abs() + sdz.abs() > search_radius {
                                            continue;
                                        }
                                        let sx = wx + sdx;
                                        let sy = above_y + sdy;
                                        let sz = wz + sdz;
                                        if let Some(m) = sample_material(density_fields, sx, sy, sz, chunk_size) {
                                            if matches!(m, Material::Copper | Material::Iron | Material::Gold) {
                                                found_ore = Some(m);
                                                break 'search;
                                            }
                                        }
                                    }
                                }
                            }

                            if let Some(ore) = found_ore {
                                let (ck, _, _, _) = world_to_chunk_local(wx, above_y, wz, chunk_size);
                                let count = enrichment_per_chunk.entry(ck).or_insert(0);
                                if *count < config.max_enrichment_per_chunk
                                    && rng.gen::<f32>() < config.enrichment_prob
                                {
                                    let (ck2, elx, ely, elz) = world_to_chunk_local(wx, above_y, wz, chunk_size);
                                    if let Some(df) = density_fields.get(&ck2) {
                                        let sample = df.get(elx, ely, elz);
                                        candidates.push(Candidate {
                                            chunk_key: ck2, lx: elx, ly: ely, lz: elz,
                                            old_material: above_mat,
                                            old_density: sample.density,
                                            new_material: ore,
                                            change_type: 0,
                                        });
                                        *count += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // --- Vein Thickening ---
    if config.vein_thickening_enabled {
        let mut thickening_per_chunk: HashMap<(i32, i32, i32), u32> = HashMap::new();

        for &chunk_key in chunks {
            let (cx, cy, cz) = chunk_key;
            let df = match density_fields.get(&chunk_key) {
                Some(df) => df,
                None => continue,
            };

            for lz in 0..field_size {
                for ly in 0..field_size {
                    for lx in 0..field_size {
                        let sample = df.get(lx, ly, lz);
                        if !is_ore(sample.material) {
                            continue;
                        }

                        let wx = cx * (chunk_size as i32) + lx as i32;
                        let wy = cy * (chunk_size as i32) + ly as i32;
                        let wz = cz * (chunk_size as i32) + lz as i32;

                        // Check if this ore is air-adjacent (surface ore)
                        let mut has_air_neighbor = false;
                        for &(dx, dy, dz) in &FACE_OFFSETS {
                            if let Some(mat) = sample_material(density_fields, wx + dx, wy + dy, wz + dz, chunk_size) {
                                if !mat.is_solid() {
                                    has_air_neighbor = true;
                                    break;
                                }
                            }
                        }

                        if !has_air_neighbor {
                            continue;
                        }

                        // Try to grow into adjacent host rock
                        if rng.gen::<f32>() >= config.vein_thickening_prob {
                            continue;
                        }

                        for &(dx, dy, dz) in &FACE_OFFSETS {
                            let nx = wx + dx;
                            let ny = wy + dy;
                            let nz = wz + dz;
                            if let Some(mat) = sample_material(density_fields, nx, ny, nz, chunk_size) {
                                if is_host_rock(mat) {
                                    let (ck, _, _, _) = world_to_chunk_local(nx, ny, nz, chunk_size);
                                    let count = thickening_per_chunk.entry(ck).or_insert(0);
                                    if *count < config.vein_thickening_max_per_chunk {
                                        let (ck2, tlx, tly, tlz) = world_to_chunk_local(nx, ny, nz, chunk_size);
                                        if let Some(tdf) = density_fields.get(&ck2) {
                                            let tsample = tdf.get(tlx, tly, tlz);
                                            candidates.push(Candidate {
                                                chunk_key: ck2, lx: tlx, ly: tly, lz: tlz,
                                                old_material: mat,
                                                old_density: tsample.density,
                                                new_material: sample.material, // Same ore type
                                                change_type: 1,
                                            });
                                            *count += 1;
                                        }
                                    }
                                    break; // Only thicken in one direction per ore voxel
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // --- Mature Formations (stalactite growth, column formation) ---
    if config.mature_formations_enabled {
        for &chunk_key in chunks {
            let (cx, cy, cz) = chunk_key;
            let df = match density_fields.get(&chunk_key) {
                Some(df) => df,
                None => continue,
            };

            for lz in 0..field_size {
                for ly in 0..field_size {
                    for lx in 0..field_size {
                        let sample = df.get(lx, ly, lz);
                        if sample.material != Material::Air {
                            continue;
                        }

                        let wx = cx * (chunk_size as i32) + lx as i32;
                        let wy = cy * (chunk_size as i32) + ly as i32;
                        let wz = cz * (chunk_size as i32) + lz as i32;

                        // Stalactite growth: air below limestone ceiling
                        if let Some(above_mat) = sample_material(density_fields, wx, wy + 1, wz, chunk_size) {
                            if above_mat == Material::Limestone
                                && rng.gen::<f32>() < config.stalactite_growth_prob
                            {
                                candidates.push(Candidate {
                                    chunk_key, lx, ly, lz,
                                    old_material: Material::Air,
                                    old_density: sample.density,
                                    new_material: Material::Limestone,
                                    change_type: 2,
                                });
                                continue;
                            }
                        }

                        // Column formation: stalactite meets stalagmite (limestone above AND below)
                        let above = sample_material(density_fields, wx, wy + 1, wz, chunk_size);
                        let below = sample_material(density_fields, wx, wy - 1, wz, chunk_size);
                        if above == Some(Material::Limestone) && below == Some(Material::Limestone)
                            && rng.gen::<f32>() < config.column_formation_prob
                        {
                            candidates.push(Candidate {
                                chunk_key, lx, ly, lz,
                                old_material: Material::Air,
                                old_density: sample.density,
                                new_material: Material::Limestone,
                                change_type: 2,
                            });
                        }
                    }
                }
            }
        }
    }

    // --- Apply all candidates ---
    let mut enrichment_count = 0u32;
    let mut thickening_count = 0u32;
    let mut formation_count = 0u32;

    for c in &candidates {
        if let Some(df) = density_fields.get_mut(&c.chunk_key) {
            let sample = df.get_mut(c.lx, c.ly, c.lz);
            sample.material = c.new_material;
            if c.change_type == 2 {
                // Formation growth gets partial density for smooth DC mesh
                sample.density = rng.gen_range(0.3..=0.6);
            }
        }

        let new_density = if c.change_type == 2 { 0.45 } else { c.old_density };
        result.manifest.record_voxel_change(
            c.chunk_key, c.lx, c.ly, c.lz,
            c.old_material, c.old_density,
            c.new_material, new_density,
        );

        if result.glimpse_chunk.is_none() {
            result.glimpse_chunk = Some(c.chunk_key);
        }

        match c.change_type {
            0 => enrichment_count += 1,
            1 => thickening_count += 1,
            2 => formation_count += 1,
            _ => {}
        }
    }

    result.voxels_enriched = enrichment_count;
    result.veins_thickened = thickening_count;
    result.formations_grown = formation_count;

    // --- Structural Collapse (delegated to existing collapse.rs) ---
    if config.collapse.collapse_enabled {
        let collapse_result = apply_collapse(
            &config.collapse,
            stress_config,
            density_fields,
            stress_fields,
            support_fields,
            chunks,
            chunk_size,
            rng,
        );
        result.supports_degraded = collapse_result.supports_degraded;
        result.collapses_triggered = collapse_result.collapses_triggered;
        result.manifest.merge_sleep_changes(&collapse_result.manifest);
        result.transform_log.extend(collapse_result.transform_log.iter().map(|t| {
            TransformEntry {
                description: format!("The Deep Time \u{2014} 1,250,000 years: {}", t.description),
                count: t.count,
            }
        }));
        if result.glimpse_chunk.is_none() {
            result.glimpse_chunk = collapse_result.glimpse_chunk;
        }
        result.collapse_result = Some(collapse_result);
    }

    // Build transform log for non-collapse changes
    if enrichment_count > 0 {
        result.transform_log.insert(0, TransformEntry {
            description: format!(
                "The Deep Time \u{2014} 1,250,000 years: Supergene enrichment concentrated {} ore voxels",
                enrichment_count
            ),
            count: enrichment_count,
        });
    }
    if thickening_count > 0 {
        result.transform_log.insert(if enrichment_count > 0 { 1 } else { 0 }, TransformEntry {
            description: format!(
                "The Deep Time \u{2014} 1,250,000 years: {} vein voxels thickened",
                thickening_count
            ),
            count: thickening_count,
        });
    }
    if formation_count > 0 {
        let insert_pos = (if enrichment_count > 0 { 1 } else { 0 }) + (if thickening_count > 0 { 1 } else { 0 });
        result.transform_log.insert(insert_pos, TransformEntry {
            description: format!(
                "The Deep Time \u{2014} 1,250,000 years: {} mature formations grown",
                formation_count
            ),
            count: formation_count,
        });
    }

    result
}
