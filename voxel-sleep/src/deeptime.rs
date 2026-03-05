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
use crate::config::{DeepTimeConfig, GroundwaterConfig};
use crate::groundwater::ambient_moisture;
use crate::manifest::ChangeManifest;
use crate::util::{FACE_OFFSETS, sample_material, count_neighbors, has_material_within_radius};
use crate::TransformEntry;

/// Result of the deep time phase.
#[derive(Debug)]
pub struct DeepTimeResult {
    pub voxels_enriched: u32,
    pub veins_thickened: u32,
    pub formations_grown: u32,
    pub supports_degraded: u32,
    pub collapses_triggered: u32,
    pub nests_fossilized: u32,
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
            nests_fossilized: 0,
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
    groundwater: &GroundwaterConfig,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    stress_fields: &mut HashMap<(i32, i32, i32), StressField>,
    support_fields: &mut HashMap<(i32, i32, i32), SupportField>,
    fluid_snapshot: &FluidSnapshot,
    _heat_map: &HeatMap,
    chunks: &[(i32, i32, i32)],
    chunk_size: usize,
    stress_config: &voxel_core::stress::StressConfig,
    nest_positions: &[(i32, i32, i32)],
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

    // --- Ambient Groundwater Enrichment ---
    let mut ambient_enrichment_count = 0u32;
    if config.enrichment_enabled && groundwater.enabled {
        let search_radius = config.enrichment_search_radius;
        let mut enrichment_per_chunk: HashMap<(i32, i32, i32), u32> = HashMap::new();

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
                        if !is_host_rock(sample.material) {
                            continue;
                        }

                        let wx = cx * (chunk_size as i32) + lx as i32;
                        let wy = cy * (chunk_size as i32) + ly as i32;
                        let wz = cz * (chunk_size as i32) + lz as i32;

                        // Drip zone: must have air below
                        let below = sample_material(density_fields, wx, wy - 1, wz, chunk_size);
                        let has_air_below = matches!(below, Some(m) if !m.is_solid());
                        if !has_air_below {
                            continue;
                        }

                        let moisture = ambient_moisture(groundwater, wy, sample.material, true);
                        if moisture <= 0.0 {
                            continue;
                        }

                        // Search radius for nearby ore
                        let mut found_ore: Option<Material> = None;
                        'asearch: for sdx in -search_radius..=search_radius {
                            for sdy in -search_radius..=search_radius {
                                for sdz in -search_radius..=search_radius {
                                    if sdx.abs() + sdy.abs() + sdz.abs() > search_radius {
                                        continue;
                                    }
                                    let sx = wx + sdx;
                                    let sy = wy + sdy;
                                    let sz = wz + sdz;
                                    if let Some(m) = sample_material(density_fields, sx, sy, sz, chunk_size) {
                                        if matches!(m, Material::Copper | Material::Iron | Material::Gold) {
                                            found_ore = Some(m);
                                            break 'asearch;
                                        }
                                    }
                                }
                            }
                        }

                        if let Some(ore) = found_ore {
                            let count = enrichment_per_chunk.entry(chunk_key).or_insert(0);
                            if *count < config.max_enrichment_per_chunk
                                && rng.gen::<f32>() < config.enrichment_prob * moisture
                            {
                                candidates.push(Candidate {
                                    chunk_key, lx, ly, lz,
                                    old_material: sample.material,
                                    old_density: sample.density,
                                    new_material: ore,
                                    change_type: 0,
                                });
                                *count += 1;
                                ambient_enrichment_count += 1;
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

    // --- Nest Fossilization ---
    let mut nests_fossilized = 0u32;
    if config.nest_fossilization.enabled && !nest_positions.is_empty() {
        let nf = &config.nest_fossilization;
        let nest_radius = nf.nest_radius as i32;

        for &(nx, ny, nz) in nest_positions {
            // Check if nest position is in a loaded chunk
            let host_mat = match sample_material(density_fields, nx, ny, nz, chunk_size) {
                Some(m) => m,
                None => continue,
            };

            // Check if buried (0 air neighbors)
            let air_count = count_neighbors(density_fields, nx, ny, nz, chunk_size, |m| !m.is_solid());
            let is_buried = air_count == 0;
            if nf.buried_required && !is_buried {
                continue;
            }

            // Check for adjacent water
            let has_water = {
                let cs = fluid_snapshot.chunk_size;
                let mut found = false;
                if cs > 0 {
                    for &(dx, dy, dz) in &FACE_OFFSETS {
                        let nwx = nx + dx;
                        let nwy = ny + dy;
                        let nwz = nz + dz;
                        let fck = (
                            nwx.div_euclid(cs as i32),
                            nwy.div_euclid(cs as i32),
                            nwz.div_euclid(cs as i32),
                        );
                        if let Some(cells) = fluid_snapshot.chunks.get(&fck) {
                            let flx = nwx.rem_euclid(cs as i32) as usize;
                            let fly = nwy.rem_euclid(cs as i32) as usize;
                            let flz = nwz.rem_euclid(cs as i32) as usize;
                            let idx = flz * cs * cs + fly * cs + flx;
                            if idx < cells.len() {
                                let cell = &cells[idx];
                                if cell.level > 0.001 && cell.fluid_type == FluidType::Water {
                                    found = true;
                                    break;
                                }
                            }
                        }
                    }
                }
                found
            };

            // Check for nearby lava
            let has_lava = {
                let cs = fluid_snapshot.chunk_size;
                let mut found = false;
                if cs > 0 {
                    for &(dx, dy, dz) in &FACE_OFFSETS {
                        let nwx = nx + dx;
                        let nwy = ny + dy;
                        let nwz = nz + dz;
                        let fck = (
                            nwx.div_euclid(cs as i32),
                            nwy.div_euclid(cs as i32),
                            nwz.div_euclid(cs as i32),
                        );
                        if let Some(cells) = fluid_snapshot.chunks.get(&fck) {
                            let flx = nwx.rem_euclid(cs as i32) as usize;
                            let fly = nwy.rem_euclid(cs as i32) as usize;
                            let flz = nwz.rem_euclid(cs as i32) as usize;
                            let idx = flz * cs * cs + fly * cs + flx;
                            if idx < cells.len() {
                                let cell = &cells[idx];
                                if cell.level > 0.001 && cell.fluid_type == FluidType::Lava {
                                    found = true;
                                    break;
                                }
                            }
                        }
                    }
                }
                found
            };

            // Near lava → nothing
            if has_lava {
                continue;
            }

            // Decision tree for fossilization material
            let iron_rich = matches!(host_mat, Material::Basalt)
                || has_material_within_radius(density_fields, nx, ny, nz, chunk_size, 3, Material::Iron)
                || has_material_within_radius(density_fields, nx, ny, nz, chunk_size, 3, Material::Sulfide)
                || has_material_within_radius(density_fields, nx, ny, nz, chunk_size, 3, Material::Pyrite);
            let silica_rich = matches!(host_mat, Material::Granite | Material::Sandstone);

            let target_mat = if is_buried || !nf.buried_required {
                if (has_water || !nf.water_required_for_pyrite) && iron_rich && rng.gen::<f32>() < nf.pyrite_prob {
                    Some(Material::Pyrite)
                } else if (has_water || !nf.water_required_for_opal) && silica_rich && rng.gen::<f32>() < nf.opal_prob {
                    Some(Material::Opal)
                } else if (has_water || !nf.water_required_for_opal) && host_mat == Material::Limestone && !iron_rich && rng.gen::<f32>() < 0.25 {
                    Some(Material::Opal)
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(target) = target_mat {
                // Replace voxels within nest_radius
                for rdx in -nest_radius..=nest_radius {
                    for rdy in -nest_radius..=nest_radius {
                        for rdz in -nest_radius..=nest_radius {
                            if rdx.abs() + rdy.abs() + rdz.abs() > nest_radius {
                                continue;
                            }
                            let wx = nx + rdx;
                            let wy = ny + rdy;
                            let wz = nz + rdz;
                            if let Some(mat) = sample_material(density_fields, wx, wy, wz, chunk_size) {
                                if mat.is_solid() {
                                    let (ck, lx, ly, lz) = world_to_chunk_local(wx, wy, wz, chunk_size);
                                    if let Some(df) = density_fields.get(&ck) {
                                        let sample = df.get(lx, ly, lz);
                                        candidates.push(Candidate {
                                            chunk_key: ck, lx, ly, lz,
                                            old_material: mat,
                                            old_density: sample.density,
                                            new_material: target,
                                            change_type: 3, // fossilization
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
                nests_fossilized += 1;
            }
        }
    }

    // --- Apply all candidates ---
    let mut enrichment_count = 0u32;
    let mut thickening_count = 0u32;
    let mut formation_count = 0u32;
    let mut fossilization_count = 0u32;

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
            3 => fossilization_count += 1,
            _ => {}
        }
    }

    result.voxels_enriched = enrichment_count;
    result.veins_thickened = thickening_count;
    result.formations_grown = formation_count;
    result.nests_fossilized = nests_fossilized;

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
    if nests_fossilized > 0 {
        result.transform_log.insert(0, TransformEntry {
            description: format!(
                "The Deep Time \u{2014} 1,250,000 years: {} spider nests fossilized ({} voxels replaced)",
                nests_fossilized, fossilization_count
            ),
            count: nests_fossilized,
        });
    }
    if enrichment_count > 0 {
        result.transform_log.insert(0, TransformEntry {
            description: format!(
                "The Deep Time \u{2014} 1,250,000 years: Supergene enrichment concentrated {} ore voxels",
                enrichment_count
            ),
            count: enrichment_count,
        });
    }
    if ambient_enrichment_count > 0 {
        result.transform_log.insert(if enrichment_count > 0 { 1 } else { 0 }, TransformEntry {
            description: format!(
                "The Deep Time \u{2014} 1,250,000 years: {} ore voxels enriched by ambient groundwater",
                ambient_enrichment_count
            ),
            count: ambient_enrichment_count,
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
