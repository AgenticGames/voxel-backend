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


use crate::aureole::HeatMap;
use crate::collapse::{apply_collapse, CollapseResult};
use crate::config::{DeepTimeConfig, GroundwaterConfig};
use crate::groundwater::{ambient_moisture, is_fracture_site};
use crate::manifest::ChangeManifest;
use crate::util::{FACE_OFFSETS, sample_material, count_neighbors, has_material_within_radius, grow_vein, default_vein_bias, sleep_vein_size, VeinGrowthParams, VeinBias};
use crate::{Bottleneck, PhaseDiagnostics, ResourceCensus, TransformEntry};

/// Result of the deep time phase.
#[derive(Debug)]
pub struct DeepTimeResult {
    pub voxels_enriched: u32,
    pub veins_thickened: u32,
    pub formations_grown: u32,
    pub supports_degraded: u32,
    pub collapses_triggered: u32,
    pub nests_fossilized: u32,
    pub corpses_fossilized: u32,
    pub collapse_result: Option<CollapseResult>,
    pub manifest: ChangeManifest,
    pub glimpse_chunk: Option<(i32, i32, i32)>,
    pub transform_log: Vec<TransformEntry>,
    pub diagnostics: PhaseDiagnostics,
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
            corpses_fossilized: 0,
            collapse_result: None,
            manifest: ChangeManifest::default(),
            glimpse_chunk: None,
            transform_log: Vec::new(),
            diagnostics: PhaseDiagnostics::default(),
        }
    }
}

/// Check if a Slate or Hornfels barrier exists within scan_depth below a position.
fn has_slate_barrier_below(
    density_fields: &HashMap<(i32,i32,i32), DensityField>,
    wx: i32, wy: i32, wz: i32,
    scan_depth: i32,
    chunk_size: usize,
) -> bool {
    for dy in 1..=scan_depth {
        if let Some(mat) = sample_material(density_fields, wx, wy - dy, wz, chunk_size) {
            if matches!(mat, Material::Slate | Material::Hornfels) {
                return true;
            }
        }
    }
    false
}

/// Check if a Slate or Hornfels barrier exists within scan_depth above a position.
fn has_slate_barrier_above(
    density_fields: &HashMap<(i32,i32,i32), DensityField>,
    wx: i32, wy: i32, wz: i32,
    scan_depth: i32,
    chunk_size: usize,
) -> bool {
    for dy in 1..=scan_depth {
        if let Some(mat) = sample_material(density_fields, wx, wy + dy, wz, chunk_size) {
            if matches!(mat, Material::Slate | Material::Hornfels) {
                return true;
            }
        }
    }
    false
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
    fluid_snapshot: &mut FluidSnapshot,
    _heat_map: &HeatMap,
    chunks: &[(i32, i32, i32)],
    chunk_size: usize,
    stress_config: &voxel_core::stress::StressConfig,
    nest_positions: &[(i32, i32, i32)],
    corpse_positions: &[(i32, i32, i32)],
    sleep_count: u32,
    rng: &mut ChaCha8Rng,
    census: &ResourceCensus,
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
    let mut theoretical_max = 0u32;

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
                        if cell.level <= 0.001 || !cell.fluid_type.is_water() {
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

                            theoretical_max += 1;

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
                                            if matches!(m, Material::Copper | Material::Iron | Material::Gold | Material::Sulfide | Material::Pyrite) {
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
                                    let bias = default_vein_bias(ore, rng);
                                    let (sv_min, sv_max) = sleep_vein_size(ore);
                                    let params = VeinGrowthParams {
                                        ore,
                                        min_size: sv_min,
                                        max_size: sv_max,
                                        bias,
                                    };
                                    let cluster = grow_vein(density_fields, (wx, above_y, wz), &params, chunk_size, rng);
                                    for &pos in &cluster {
                                        if *count >= config.max_enrichment_per_chunk {
                                            break;
                                        }
                                        let (ck2, elx, ely, elz) = world_to_chunk_local(pos.0, pos.1, pos.2, chunk_size);
                                        if let Some(df) = density_fields.get(&ck2) {
                                            let sample = df.get(elx, ely, elz);
                                            candidates.push(Candidate {
                                                chunk_key: ck2, lx: elx, ly: ely, lz: elz,
                                                old_material: sample.material,
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
    }

    // --- Ambient Groundwater Enrichment ---
    // Groundwater dissolves trace minerals from host rock over geological time
    // and re-deposits them at drip zones (ceiling voxels with air below).
    // Soft rock (limestone/sandstone): full drip-zone enrichment.
    // Hard rock (granite/basalt/slate/marble): ONLY at fracture sites (1-2 air neighbors).
    // If nearby ore exists, concentrate that. Otherwise, deposit trace minerals
    // naturally present in the host rock geochemistry.
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
                        let mat = sample.material;
                        if !is_host_rock(mat) {
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

                        let is_soft = mat.is_soft_rock();
                        let is_hard = mat.is_hard_rock();

                        // Hard rock: enrichment ONLY at fracture sites (1-2 air neighbors)
                        if is_hard {
                            let air_count = count_neighbors(density_fields, wx, wy, wz, chunk_size, |m| !m.is_solid());
                            if !is_fracture_site(air_count) {
                                continue;
                            }
                        }

                        let moisture = ambient_moisture(groundwater, wy, mat, true);
                        // Slate aquitard effect
                        let moisture = if config.slate_aquitard_enabled {
                            if has_slate_barrier_above(density_fields, wx, wy, wz, config.slate_aquitard_scan_depth, chunk_size) {
                                moisture * config.slate_aquitard_concentration // water pools above slate
                            } else if has_slate_barrier_below(density_fields, wx, wy, wz, config.slate_aquitard_scan_depth, chunk_size) {
                                moisture * config.slate_aquitard_factor // nearly dry below slate
                            } else {
                                moisture
                            }
                        } else {
                            moisture
                        };
                        if moisture <= 0.0 {
                            continue;
                        }

                        theoretical_max += 1;

                        // Search radius for nearby ore — concentrate if found
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
                                        if matches!(m, Material::Copper | Material::Iron | Material::Gold | Material::Sulfide | Material::Pyrite) {
                                            found_ore = Some(m);
                                            break 'asearch;
                                        }
                                    }
                                }
                            }
                        }

                        // If no nearby ore, derive trace mineral from host rock geochemistry
                        let ore = found_ore.unwrap_or_else(|| {
                            match mat {
                                Material::Granite | Material::Basalt => {
                                    if rng.gen::<f32>() < 0.6 { Material::Iron } else { Material::Copper }
                                }
                                Material::Limestone => {
                                    if rng.gen::<f32>() < 0.5 { Material::Iron } else { Material::Malachite }
                                }
                                Material::Sandstone => Material::Iron,
                                Material::Slate | Material::Marble => {
                                    if rng.gen::<f32>() < 0.5 { Material::Copper } else { Material::Quartz }
                                }
                                _ => Material::Iron,
                            }
                        });

                        // Power multiplier based on rock type
                        let rock_mult = if is_soft { groundwater.soft_rock_mult } else if is_hard { groundwater.hard_rock_mult } else { 1.0 };
                        let eff_prob = if found_ore.is_some() {
                            config.enrichment_prob * moisture * groundwater.enrichment_power * rock_mult
                        } else {
                            // Trace enrichment: weaker (halved) since there's no ore source
                            config.enrichment_prob * moisture * groundwater.enrichment_power * rock_mult * 0.5
                        };

                        let count = enrichment_per_chunk.entry(chunk_key).or_insert(0);
                        if *count < config.max_enrichment_per_chunk
                            && rng.gen::<f32>() < eff_prob
                        {
                            let bias = default_vein_bias(ore, rng);
                            let (sv_min, sv_max) = sleep_vein_size(ore);
                            let params = VeinGrowthParams {
                                ore,
                                min_size: sv_min,
                                max_size: sv_max,
                                bias,
                            };
                            let cluster = grow_vein(density_fields, (wx, wy, wz), &params, chunk_size, rng);
                            for &pos in &cluster {
                                if *count >= config.max_enrichment_per_chunk {
                                    break;
                                }
                                let (ck2, elx, ely, elz) = world_to_chunk_local(pos.0, pos.1, pos.2, chunk_size);
                                if let Some(df) = density_fields.get(&ck2) {
                                    let esample = df.get(elx, ely, elz);
                                    candidates.push(Candidate {
                                        chunk_key: ck2, lx: elx, ly: ely, lz: elz,
                                        old_material: esample.material,
                                        old_density: esample.density,
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

                        theoretical_max += 1;

                        // Try to grow into adjacent host rock
                        if rng.gen::<f32>() >= config.vein_thickening_prob {
                            continue;
                        }

                        // Find first host-rock neighbor to seed vein growth from
                        let mut growth_seed: Option<(i32, i32, i32)> = None;
                        for &(dx, dy, dz) in &FACE_OFFSETS {
                            let nx = wx + dx;
                            let ny = wy + dy;
                            let nz = wz + dz;
                            if let Some(mat) = sample_material(density_fields, nx, ny, nz, chunk_size) {
                                if is_host_rock(mat) {
                                    growth_seed = Some((nx, ny, nz));
                                    break;
                                }
                            }
                        }

                        if let Some(seed_pos) = growth_seed {
                            let (ck, _, _, _) = world_to_chunk_local(seed_pos.0, seed_pos.1, seed_pos.2, chunk_size);
                            let count = thickening_per_chunk.entry(ck).or_insert(0);
                            if *count < config.vein_thickening_max_per_chunk {
                                let params = VeinGrowthParams {
                                    ore: sample.material,
                                    min_size: config.vein_thickening_growth_min,
                                    max_size: config.vein_thickening_growth_max,
                                    bias: VeinBias::Compact,
                                };
                                let growth = grow_vein(density_fields, seed_pos, &params, chunk_size, rng);
                                for &pos in &growth {
                                    if *count >= config.vein_thickening_max_per_chunk {
                                        break;
                                    }
                                    let (ck2, tlx, tly, tlz) = world_to_chunk_local(pos.0, pos.1, pos.2, chunk_size);
                                    if let Some(tdf) = density_fields.get(&ck2) {
                                        let tsample = tdf.get(tlx, tly, tlz);
                                        candidates.push(Candidate {
                                            chunk_key: ck2, lx: tlx, ly: tly, lz: tlz,
                                            old_material: tsample.material,
                                            old_density: tsample.density,
                                            new_material: sample.material,
                                            change_type: 1,
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

    // --- Mature Formations (stalactite growth, column formation) ---
    // Stalactites and columns are calcite speleothems — only form under limestone ceilings.
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

                        // Stalactite growth: only under limestone ceiling (calcite precipitation)
                        if let Some(above_mat) = sample_material(density_fields, wx, wy + 1, wz, chunk_size) {
                            if above_mat == Material::Limestone {
                                theoretical_max += 1;
                            }
                            // Skip formation if Slate aquitard blocks water from above
                            if config.slate_aquitard_enabled && has_slate_barrier_above(density_fields, wx, wy, wz, config.slate_aquitard_scan_depth, chunk_size) {
                                continue;
                            }
                            if above_mat == Material::Limestone
                                && rng.gen::<f32>() < config.stalactite_growth_prob * groundwater.flowstone_power * groundwater.soft_rock_mult
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

                        // Column formation: limestone above AND limestone below
                        let above = sample_material(density_fields, wx, wy + 1, wz, chunk_size);
                        let below = sample_material(density_fields, wx, wy - 1, wz, chunk_size);
                        if above == Some(Material::Limestone) && below == Some(Material::Limestone)
                            && rng.gen::<f32>() < config.column_formation_prob * groundwater.flowstone_power * groundwater.soft_rock_mult
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

            // Check air neighbors (burial no longer required — over 1.25Ma, material self-buries)
            let air_count = count_neighbors(density_fields, nx, ny, nz, chunk_size, |m| !m.is_solid());
            let is_buried = air_count == 0;

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
                                if cell.level > 0.001 && cell.fluid_type.is_water() {
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
                                if cell.level > 0.001 && cell.fluid_type.is_lava() {
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

    // --- Corpse Fossilization ---
    let mut corpses_fossilized = 0u32;
    if config.corpse_fossilization.enabled && !corpse_positions.is_empty() && sleep_count >= config.corpse_fossilization.min_sleep_cycles {
        let cf = &config.corpse_fossilization;
        let corpse_radius = cf.corpse_radius as i32;

        for &(cx_pos, cy_pos, cz_pos) in corpse_positions {
            let host_mat = match sample_material(density_fields, cx_pos, cy_pos, cz_pos, chunk_size) {
                Some(m) => m,
                None => continue,
            };

            // Check for adjacent water
            let has_water = if cf.water_required {
                let cs = fluid_snapshot.chunk_size;
                let mut found = false;
                if cs > 0 {
                    for &(dx, dy, dz) in &FACE_OFFSETS {
                        let nwx = cx_pos + dx;
                        let nwy = cy_pos + dy;
                        let nwz = cz_pos + dz;
                        let fck = (
                            nwx.div_euclid(cs as i32),
                            nwy.div_euclid(cs as i32),
                            nwz.div_euclid(cs as i32),
                        );
                        if let Some(cells) = fluid_snapshot.chunks.get(&fck) {
                            let flx = nwx.rem_euclid(cs as i32) as usize;
                            let fly = nwy.rem_euclid(cs as i32) as usize;
                            let flz = nwz.rem_euclid(cs as i32) as usize;
                            let fidx = flz * cs * cs + fly * cs + flx;
                            if fidx < cells.len() && cells[fidx].level > 0.001 && cells[fidx].fluid_type.is_water() {
                                found = true;
                                break;
                            }
                        }
                    }
                }
                found
            } else {
                true
            };

            if !has_water {
                continue;
            }

            // Check for nearby lava (destroys corpses)
            let has_lava = {
                let cs = fluid_snapshot.chunk_size;
                let mut found = false;
                if cs > 0 {
                    for &(dx, dy, dz) in &FACE_OFFSETS {
                        let nwx = cx_pos + dx;
                        let nwy = cy_pos + dy;
                        let nwz = cz_pos + dz;
                        let fck = (
                            nwx.div_euclid(cs as i32),
                            nwy.div_euclid(cs as i32),
                            nwz.div_euclid(cs as i32),
                        );
                        if let Some(cells) = fluid_snapshot.chunks.get(&fck) {
                            let flx = nwx.rem_euclid(cs as i32) as usize;
                            let fly = nwy.rem_euclid(cs as i32) as usize;
                            let flz = nwz.rem_euclid(cs as i32) as usize;
                            let fidx = flz * cs * cs + fly * cs + flx;
                            if fidx < cells.len() && cells[fidx].level > 0.001 && cells[fidx].fluid_type.is_lava() {
                                found = true;
                                break;
                            }
                        }
                    }
                }
                found
            };
            if has_lava { continue; }

            // Material selection for corpse fossilization
            let iron_rich = matches!(host_mat, Material::Basalt)
                || has_material_within_radius(density_fields, cx_pos, cy_pos, cz_pos, chunk_size, 3, Material::Iron)
                || has_material_within_radius(density_fields, cx_pos, cy_pos, cz_pos, chunk_size, 3, Material::Sulfide)
                || has_material_within_radius(density_fields, cx_pos, cy_pos, cz_pos, chunk_size, 3, Material::Pyrite);

            let target_mat = if iron_rich && rng.gen::<f32>() < cf.pyrite_prob {
                // Iron-rich + water → Pyrite replacement mineralization (like real insect fossils)
                Some(Material::Pyrite)
            } else if host_mat == Material::Limestone && rng.gen::<f32>() < cf.calcium_prob {
                // Limestone host → calcium carbonate preservation
                Some(Material::Limestone)
            } else {
                None
            };

            if let Some(target) = target_mat {
                for rdx in -corpse_radius..=corpse_radius {
                    for rdy in -corpse_radius..=corpse_radius {
                        for rdz in -corpse_radius..=corpse_radius {
                            if rdx.abs() + rdy.abs() + rdz.abs() > corpse_radius {
                                continue;
                            }
                            let wx = cx_pos + rdx;
                            let wy = cy_pos + rdy;
                            let wz = cz_pos + rdz;
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
                                            change_type: 4, // corpse fossilization
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
                corpses_fossilized += 1;

                // Drain nearby water for fossilization (0.1 per corpse)
                let cs = fluid_snapshot.chunk_size;
                if cs > 0 {
                    for &(dx, dy, dz) in &FACE_OFFSETS {
                        let nwx = cx_pos + dx;
                        let nwy = cy_pos + dy;
                        let nwz = cz_pos + dz;
                        let fck = (
                            nwx.div_euclid(cs as i32),
                            nwy.div_euclid(cs as i32),
                            nwz.div_euclid(cs as i32),
                        );
                        if let Some(cells) = fluid_snapshot.chunks.get_mut(&fck) {
                            let flx = nwx.rem_euclid(cs as i32) as usize;
                            let fly = nwy.rem_euclid(cs as i32) as usize;
                            let flz = nwz.rem_euclid(cs as i32) as usize;
                            let fidx = flz * cs * cs + fly * cs + flx;
                            if fidx < cells.len() && !cells[fidx].is_source && cells[fidx].level > 0.001 && cells[fidx].fluid_type.is_water() {
                                cells[fidx].level = (cells[fidx].level - 0.1).max(0.0);
                            }
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
    let mut fossilization_count = 0u32;
    let mut corpse_fossil_count = 0u32;
    let mut conversions: std::collections::BTreeMap<(u8, u8), u32> = std::collections::BTreeMap::new();

    for c in &candidates {
        *conversions.entry((c.old_material as u8, c.new_material as u8)).or_insert(0) += 1;
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
            4 => corpse_fossil_count += 1,
            _ => {}
        }
    }

    result.voxels_enriched = enrichment_count;
    result.veins_thickened = thickening_count;
    result.formations_grown = formation_count;
    result.nests_fossilized = nests_fossilized;
    result.corpses_fossilized = corpses_fossilized;

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
    if corpses_fossilized > 0 {
        result.transform_log.insert(0, TransformEntry {
            description: format!(
                "The Deep Time \u{2014} 1,250,000 years: {} spider corpses fossilized ({} voxels replaced)",
                corpses_fossilized, corpse_fossil_count
            ),
            count: corpses_fossilized,
        });
    }
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

    // --- Diagnostics ---
    let actual_output = candidates.len() as u32;
    result.diagnostics = PhaseDiagnostics {
        conversions,
        theoretical_max,
        actual_output,
        bottlenecks: compute_deeptime_bottlenecks(census, nest_positions),
    };

    result
}

fn compute_deeptime_bottlenecks(census: &ResourceCensus, nest_positions: &[(i32, i32, i32)]) -> Vec<Bottleneck> {
    let mut bottlenecks = Vec::new();

    if census.water.cell_count == 0 {
        bottlenecks.push(Bottleneck {
            severity: 0.8,
            description: "No water \u{2014} supergene enrichment needs water percolating through rock".into(),
        });
    } else if census.water.cell_count < 10 {
        bottlenecks.push(Bottleneck {
            severity: 0.5,
            description: format!(
                "Only {} water cells \u{2014} more water above rock increases enrichment yield",
                census.water.cell_count
            ),
        });
    }

    // Check surface ore for vein thickening
    let surface_ore: u32 = census.exposed_ore.values().sum();
    if surface_ore == 0 {
        bottlenecks.push(Bottleneck {
            severity: 0.6,
            description: "No exposed ore voxels \u{2014} vein thickening needs air-adjacent ore".into(),
        });
    }

    // Check limestone ceilings for formations
    let exposed_limestone = census.exposed_surfaces_by_material.get(&(Material::Limestone as u8)).copied().unwrap_or(0);
    if exposed_limestone == 0 {
        bottlenecks.push(Bottleneck {
            severity: 0.4,
            description: "No exposed limestone \u{2014} stalactite/column formation requires limestone ceilings".into(),
        });
    }

    if nest_positions.is_empty() {
        bottlenecks.push(Bottleneck {
            severity: 0.2,
            description: "No spider nests \u{2014} nest fossilization has no targets".into(),
        });
    }

    bottlenecks.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap_or(std::cmp::Ordering::Equal));
    bottlenecks.truncate(3);
    bottlenecks
}
