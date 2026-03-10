//! Phase 3: "The Veins" — 500,000 years.
//!
//! THE GAMEPLAY PAYOFF: Hydrothermal ore deposition through player tunnels.
//! BFS from heat sources through air (player tunnels = fracture pathways),
//! depositing temperature-zonated ores on tunnel walls.
//! Also: cave formation growth (crystal, calcite, flowstone).

use std::collections::{HashMap, HashSet, VecDeque};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use voxel_core::density::DensityField;
use voxel_core::material::Material;
use voxel_core::stress::world_to_chunk_local;
use voxel_fluid::FluidSnapshot;


use crate::aureole::HeatMap;
use crate::config::{VeinConfig, GroundwaterConfig};
use crate::groundwater::ambient_moisture;
use crate::manifest::ChangeManifest;
use crate::util::{FACE_OFFSETS, sample_material, count_neighbors, aperture_multiplier, find_air_from_solid, grow_vein, sleep_vein_size, default_vein_bias, VeinGrowthParams};
use crate::{Bottleneck, PhaseDiagnostics, ResourceCensus, TransformEntry};

/// Result of the veins phase.
#[derive(Debug, Default)]
pub struct VeinResult {
    pub veins_deposited: u32,
    pub formations_grown: u32,
    pub manifest: ChangeManifest,
    pub glimpse_chunk: Option<(i32, i32, i32)>,
    pub transform_log: Vec<TransformEntry>,
    pub diagnostics: PhaseDiagnostics,
}

/// Select ore type by BFS distance from heat source and host rock (temperature zonation + skarn/slate).
fn select_ore_by_distance_and_host(
    config: &VeinConfig,
    distance: u32,
    host_rock: Material,
    rng: &mut ChaCha8Rng,
) -> Material {
    // Per-host ore selection (limestone skarn = copper-iron, slate = gold-pyrite Bendigo)
    if config.host_rock_ore_enabled {
        match host_rock {
            Material::Limestone | Material::Garnet | Material::Diopside => {
                if distance < config.hypothermal_max {
                    let r = rng.gen::<f32>();
                    if r < 0.50 { return Material::Copper; }
                    else if r < 0.80 { return Material::Iron; }
                    else { return Material::Pyrite; }
                } else if distance < config.mesothermal_max {
                    let r = rng.gen::<f32>();
                    if r < 0.45 { return Material::Copper; }
                    else if r < 0.80 { return Material::Iron; }
                    else { return Material::Pyrite; }
                }
                // epithermal falls through to default
            },
            Material::Slate | Material::Hornfels => {
                if distance < config.hypothermal_max {
                    let r = rng.gen::<f32>();
                    if r < 0.40 { return Material::Quartz; }
                    else if r < 0.70 { return Material::Pyrite; }
                    else { return Material::Tin; }
                } else if distance < config.mesothermal_max {
                    let r = rng.gen::<f32>();
                    if r < 0.25 { return Material::Gold; }
                    else if r < 0.60 { return Material::Pyrite; }
                    else if r < 0.85 { return Material::Iron; }
                    else { return Material::Copper; }
                } else {
                    let r = rng.gen::<f32>();
                    if r < 0.50 { return Material::Gold; }
                    else if r < 0.80 { return Material::Pyrite; }
                    else { return Material::Sulfide; }
                }
            },
            _ => {},
        }
    }

    // Default behavior (all other hosts)
    if distance < config.hypothermal_max {
        let r = rng.gen::<f32>();
        if r < 0.35 { Material::Tin }
        else if r < 0.65 { Material::Quartz }
        else { Material::Pyrite }
    } else if distance < config.mesothermal_max {
        let r = rng.gen::<f32>();
        if r < 0.40 { Material::Copper }
        else if r < 0.75 { Material::Iron }
        else { Material::Pyrite }
    } else {
        let r = rng.gen::<f32>();
        if r < 0.30 { Material::Gold }
        else if r < 0.55 { Material::Sulfide }
        else { Material::Pyrite }
    }
}

/// Check if a material is a host rock (transformable by vein deposition).
fn is_host_rock(mat: Material) -> bool {
    matches!(mat,
        Material::Sandstone | Material::Limestone | Material::Granite |
        Material::Basalt | Material::Slate | Material::Marble |
        Material::Hornfels | Material::Garnet | Material::Diopside
    )
}

/// Execute Phase 3: hydrothermal vein injection + formation growth.
pub fn apply_veins(
    config: &VeinConfig,
    groundwater: &GroundwaterConfig,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    fluid_snapshot: &mut FluidSnapshot,
    heat_map: &HeatMap,
    chunks: &[(i32, i32, i32)],
    chunk_size: usize,
    rng: &mut ChaCha8Rng,
    census: &ResourceCensus,
) -> VeinResult {
    let mut result = VeinResult::default();
    let field_size = chunk_size + 1;

    struct VeinCandidate {
        chunk_key: (i32, i32, i32),
        lx: usize, ly: usize, lz: usize,
        old_material: Material,
        old_density: f32,
        new_material: Material,
    }

    let mut vein_candidates: Vec<VeinCandidate> = Vec::new();
    let mut theoretical_max = 0u32;

    // --- Hydrothermal Vein Deposition ---
    if config.vein_enabled && !heat_map.is_empty() {
        let max_dist = config.vein_max_distance;
        let max_per_source = config.max_vein_voxels_per_source;
        let mut global_deposited: HashSet<(i32, i32, i32)> = HashSet::new();

        for heat in heat_map {
            let (hx, hy, hz) = heat.pos;
            // BFS through air voxels from heat source
            let mut queue: VecDeque<((i32, i32, i32), u32)> = VecDeque::new();
            let mut visited: HashSet<(i32, i32, i32)> = HashSet::new();
            let mut source_deposited = 0u32;

            // Try direct air neighbors first (fast path for exposed lava/kimberlite)
            for &(dx, dy, dz) in &FACE_OFFSETS {
                let nx = hx + dx;
                let ny = hy + dy;
                let nz = hz + dz;
                if let Some(mat) = sample_material(density_fields, nx, ny, nz, chunk_size) {
                    if !mat.is_solid() && visited.insert((nx, ny, nz)) {
                        queue.push_back(((nx, ny, nz), 1));
                    }
                }
            }

            // Fallback: search through solid rock to find air (submerged lava in solid bowls)
            if queue.is_empty() {
                let air_seeds = find_air_from_solid(
                    density_fields, (hx, hy, hz),
                    config.heat_source_search_radius, chunk_size,
                );
                for (pos, dist) in air_seeds {
                    if visited.insert(pos) {
                        queue.push_back((pos, dist));
                    }
                }
            }

            while let Some(((ax, ay, az), dist)) = queue.pop_front() {
                if source_deposited >= max_per_source {
                    break;
                }

                // Check solid neighbors of this air voxel for deposition sites
                for &(dx, dy, dz) in &FACE_OFFSETS {
                    let nx = ax + dx;
                    let ny = ay + dy;
                    let nz = az + dz;

                    if global_deposited.contains(&(nx, ny, nz)) {
                        continue;
                    }

                    if let Some(mat) = sample_material(density_fields, nx, ny, nz, chunk_size) {
                        if is_host_rock(mat) {
                            theoretical_max += 1;
                        }
                        let air_n = count_neighbors(density_fields, ax, ay, az, chunk_size, |m| !m.is_solid());
                        let eff_prob = config.vein_deposition_prob * if config.aperture_scaling_enabled {
                            aperture_multiplier(air_n)
                        } else { 1.0 };
                        if is_host_rock(mat) && rng.gen::<f32>() < eff_prob {
                            let ore = select_ore_by_distance_and_host(config, dist, mat, rng);

                            // Epithermal rarity: precious metals deposit less frequently
                            let rarity_factor = if dist >= config.mesothermal_max {
                                config.epithermal_rarity
                            } else {
                                1.0
                            };
                            if rng.gen::<f32>() >= rarity_factor {
                                continue; // fluid too dilute for precious metals
                            }

                            let (min_sz, max_sz) = sleep_vein_size(ore);
                            let bias = default_vein_bias(ore, rng);
                            let params = VeinGrowthParams { ore, min_size: min_sz, max_size: max_sz, bias };
                            let vein_positions = grow_vein(density_fields, (nx, ny, nz), &params, chunk_size, rng);

                            for &vpos in &vein_positions {
                                if global_deposited.contains(&vpos) {
                                    continue;
                                }
                                if source_deposited >= max_per_source {
                                    break;
                                }
                                let (ck, lx, ly, lz) = world_to_chunk_local(vpos.0, vpos.1, vpos.2, chunk_size);
                                if let Some(df) = density_fields.get(&ck) {
                                    let sample = df.get(lx, ly, lz);
                                    let old_mat = sample.material;
                                    vein_candidates.push(VeinCandidate {
                                        chunk_key: ck, lx, ly, lz,
                                        old_material: old_mat,
                                        old_density: sample.density,
                                        new_material: ore,
                                    });
                                    global_deposited.insert(vpos);
                                    source_deposited += 1;
                                }
                            }

                            // Co-deposit pyrite gangue mineral alongside non-Pyrite ores
                            let pyrite_codeposit_prob = if matches!(mat, Material::Slate | Material::Hornfels) {
                                config.slate_pyrite_codeposit_prob
                            } else {
                                0.25
                            };
                            if ore != Material::Pyrite && rng.gen::<f32>() < pyrite_codeposit_prob && source_deposited < max_per_source {
                                // Pick first host-rock neighbor of the vein seed for pyrite co-deposit
                                for &(pdx, pdy, pdz) in &FACE_OFFSETS {
                                    let pnx = nx + pdx;
                                    let pny = ny + pdy;
                                    let pnz = nz + pdz;
                                    if global_deposited.contains(&(pnx, pny, pnz)) { continue; }
                                    if let Some(pmat) = sample_material(density_fields, pnx, pny, pnz, chunk_size) {
                                        if is_host_rock(pmat) {
                                            let pyrite_params = VeinGrowthParams {
                                                ore: Material::Pyrite,
                                                min_size: 1,
                                                max_size: 3,
                                                bias: default_vein_bias(Material::Pyrite, rng),
                                            };
                                            let pyrite_vein = grow_vein(density_fields, (pnx, pny, pnz), &pyrite_params, chunk_size, rng);
                                            for &pvpos in &pyrite_vein {
                                                if global_deposited.contains(&pvpos) { continue; }
                                                if source_deposited >= max_per_source { break; }
                                                let (pck, plx, ply, plz) = world_to_chunk_local(pvpos.0, pvpos.1, pvpos.2, chunk_size);
                                                if let Some(pdf) = density_fields.get(&pck) {
                                                    let psample = pdf.get(plx, ply, plz);
                                                    vein_candidates.push(VeinCandidate {
                                                        chunk_key: pck, lx: plx, ly: ply, lz: plz,
                                                        old_material: psample.material,
                                                        old_density: psample.density,
                                                        new_material: Material::Pyrite,
                                                    });
                                                    global_deposited.insert(pvpos);
                                                    source_deposited += 1;
                                                }
                                            }
                                            break;
                                        }
                                    }
                                }
                            }

                            // Quartz co-deposit for gold in Slate (saddle reef)
                            if matches!(mat, Material::Slate | Material::Hornfels)
                                && ore == Material::Gold
                                && rng.gen::<f32>() < config.slate_quartz_vein_prob
                                && source_deposited < max_per_source
                            {
                                for &(qdx, qdy, qdz) in &FACE_OFFSETS {
                                    let qnx = nx + qdx;
                                    let qny = ny + qdy;
                                    let qnz = nz + qdz;
                                    if global_deposited.contains(&(qnx, qny, qnz)) { continue; }
                                    if let Some(qmat) = sample_material(density_fields, qnx, qny, qnz, chunk_size) {
                                        if is_host_rock(qmat) {
                                            let quartz_params = VeinGrowthParams {
                                                ore: Material::Quartz,
                                                min_size: 1,
                                                max_size: 3,
                                                bias: default_vein_bias(Material::Quartz, rng),
                                            };
                                            let quartz_vein = grow_vein(density_fields, (qnx, qny, qnz), &quartz_params, chunk_size, rng);
                                            for &qvpos in &quartz_vein {
                                                if global_deposited.contains(&qvpos) { continue; }
                                                if source_deposited >= max_per_source { break; }
                                                let (qck, qlx, qly, qlz) = world_to_chunk_local(qvpos.0, qvpos.1, qvpos.2, chunk_size);
                                                if let Some(qdf) = density_fields.get(&qck) {
                                                    let qsample = qdf.get(qlx, qly, qlz);
                                                    vein_candidates.push(VeinCandidate {
                                                        chunk_key: qck, lx: qlx, ly: qly, lz: qlz,
                                                        old_material: qsample.material,
                                                        old_density: qsample.density,
                                                        new_material: Material::Quartz,
                                                    });
                                                    global_deposited.insert(qvpos);
                                                    source_deposited += 1;
                                                }
                                            }
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Expand BFS through air voxels
                if dist < max_dist {
                    for &(dx, dy, dz) in &FACE_OFFSETS {
                        let nx = ax + dx;
                        let ny = ay + dy;
                        let nz = az + dz;
                        if !visited.contains(&(nx, ny, nz)) {
                            if let Some(mat) = sample_material(density_fields, nx, ny, nz, chunk_size) {
                                if !mat.is_solid() {
                                    visited.insert((nx, ny, nz));
                                    queue.push_back(((nx, ny, nz), dist + 1));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Apply vein candidates
    let mut conversions: std::collections::BTreeMap<(u8, u8), u32> = std::collections::BTreeMap::new();
    for c in &vein_candidates {
        *conversions.entry((c.old_material as u8, c.new_material as u8)).or_insert(0) += 1;
        if let Some(df) = density_fields.get_mut(&c.chunk_key) {
            let sample = df.get_mut(c.lx, c.ly, c.lz);
            sample.material = c.new_material;
            // Preserve density (solid → solid)
        }

        result.manifest.record_voxel_change(
            c.chunk_key, c.lx, c.ly, c.lz,
            c.old_material, c.old_density,
            c.new_material, c.old_density,
        );

        if result.glimpse_chunk.is_none() {
            result.glimpse_chunk = Some(c.chunk_key);
        }

        result.veins_deposited += 1;
    }

    // --- Cave Formation Growth ---
    struct GrowthCandidate {
        chunk_key: (i32, i32, i32),
        lx: usize, ly: usize, lz: usize,
        new_material: Material,
        growth_type: u8, // 0=crystal, 1=calcite, 2=flowstone
    }

    let mut growth_candidates: Vec<GrowthCandidate> = Vec::new();
    let _chunk_set: HashSet<(i32, i32, i32)> = chunks.iter().copied().collect();

    for &chunk_key in chunks {
        let (cx, cy, cz) = chunk_key;
        let df = match density_fields.get(&chunk_key) {
            Some(df) => df,
            None => continue,
        };

        let mut crystal_count = 0u32;
        let mut calcite_count = 0u32;

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

                    // Crystal growth: air with 2+ Crystal/Amethyst neighbors
                    if config.crystal_growth_enabled && crystal_count < config.crystal_growth_max_per_chunk {
                        let crystal_neighbors = count_neighbors(
                            density_fields, wx, wy, wz, chunk_size,
                            |m| m == Material::Crystal || m == Material::Amethyst,
                        );
                        if crystal_neighbors >= 2 && rng.gen::<f32>() < config.crystal_growth_prob {
                            growth_candidates.push(GrowthCandidate {
                                chunk_key, lx, ly, lz,
                                new_material: Material::Crystal,
                                growth_type: 0,
                            });
                            crystal_count += 1;
                            continue;
                        }
                    }

                    // Calcite infill: air with 3+ Limestone faces
                    if config.calcite_infill_enabled && calcite_count < config.calcite_infill_max_per_chunk {
                        let ls_neighbors = count_neighbors(
                            density_fields, wx, wy, wz, chunk_size,
                            |m| m == Material::Limestone,
                        );
                        if ls_neighbors >= 3 && rng.gen::<f32>() < config.calcite_infill_prob {
                            growth_candidates.push(GrowthCandidate {
                                chunk_key, lx, ly, lz,
                                new_material: Material::Limestone,
                                growth_type: 1,
                            });
                            calcite_count += 1;
                            continue;
                        }
                    }
                }
            }
        }
    }

    // Flowstone: air along water paths (adjacent to water flow cells)
    if config.flowstone_enabled && !fluid_snapshot.chunks.is_empty() {
        let cs = fluid_snapshot.chunk_size;
        let mut flowstone_per_chunk: HashMap<(i32, i32, i32), u32> = HashMap::new();

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

                        // Check air neighbors for flowstone deposition
                        // Flowstone = calcite precipitation → requires carbonate host rock nearby
                        for &(dx, dy, dz) in &FACE_OFFSETS {
                            let nx = wx + dx;
                            let ny = wy + dy;
                            let nz = wz + dz;

                            if let Some(mat) = sample_material(density_fields, nx, ny, nz, chunk_size) {
                                if mat == Material::Air && rng.gen::<f32>() < config.flowstone_prob {
                                    // Verify at least one solid neighbor of target air is carbonate
                                    let has_carbonate = FACE_OFFSETS.iter().any(|&(dx2, dy2, dz2)| {
                                        if let Some(nm) = sample_material(density_fields, nx + dx2, ny + dy2, nz + dz2, chunk_size) {
                                            matches!(nm, Material::Limestone | Material::Sandstone | Material::Marble)
                                        } else {
                                            false
                                        }
                                    });
                                    if !has_carbonate { continue; }
                                    let (ck, _, _, _) = world_to_chunk_local(nx, ny, nz, chunk_size);
                                    let count = flowstone_per_chunk.entry(ck).or_insert(0);
                                    if *count < config.flowstone_max_per_chunk {
                                        let (ck2, flx, fly, flz) = world_to_chunk_local(nx, ny, nz, chunk_size);
                                        growth_candidates.push(GrowthCandidate {
                                            chunk_key: ck2, lx: flx, ly: fly, lz: flz,
                                            new_material: Material::Limestone,
                                            growth_type: 2,
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

    // --- Ambient Groundwater Flowstone ---
    // Only limestone/sandstone ceilings produce calcite flowstone.
    // Flowstone = calcite precipitation from carbonate-saturated water.
    let mut ambient_flowstone_count = 0u32;
    if config.flowstone_enabled && groundwater.enabled {
        let mut flowstone_per_chunk: HashMap<(i32, i32, i32), u32> = HashMap::new();
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

                        // Only limestone/sandstone ceilings (calcite source)
                        let ceiling_mat = sample_material(density_fields, wx, wy + 1, wz, chunk_size);
                        let ceiling = match ceiling_mat {
                            Some(m) if matches!(m, Material::Limestone | Material::Sandstone) => m,
                            _ => continue,
                        };

                        let moisture = ambient_moisture(groundwater, wy + 1, ceiling, true);
                        if moisture > 0.0 && rng.gen::<f32>() < config.flowstone_prob * moisture * groundwater.flowstone_power * groundwater.soft_rock_mult {
                            let count = flowstone_per_chunk.entry(chunk_key).or_insert(0);
                            if *count < config.flowstone_max_per_chunk {
                                growth_candidates.push(GrowthCandidate {
                                    chunk_key, lx, ly, lz,
                                    new_material: Material::Limestone, // calcite flowstone
                                    growth_type: 2,
                                });
                                *count += 1;
                                ambient_flowstone_count += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    // Apply growth candidates
    let mut formation_total = 0u32;
    let mut crystal_total = 0u32;
    let mut calcite_total = 0u32;
    let mut flowstone_total = 0u32;

    for g in &growth_candidates {
        *conversions.entry((Material::Air as u8, g.new_material as u8)).or_insert(0) += 1;
        let new_density = rng.gen_range(config.growth_density_min..=config.growth_density_max);
        if let Some(df) = density_fields.get_mut(&g.chunk_key) {
            let sample = df.get_mut(g.lx, g.ly, g.lz);
            let old_material = sample.material;
            let old_density = sample.density;
            sample.material = g.new_material;
            sample.density = new_density;

            result.manifest.record_voxel_change(
                g.chunk_key, g.lx, g.ly, g.lz,
                old_material, old_density,
                g.new_material, new_density,
            );
        }

        formation_total += 1;
        match g.growth_type {
            0 => crystal_total += 1,
            1 => calcite_total += 1,
            2 => flowstone_total += 1,
            _ => {}
        }

        if result.glimpse_chunk.is_none() {
            result.glimpse_chunk = Some(g.chunk_key);
        }
    }

    result.formations_grown = formation_total;

    // Build transform log
    if result.veins_deposited > 0 {
        result.transform_log.push(TransformEntry {
            description: format!(
                "The Veins \u{2014} 500,000 years: {} ore voxels deposited from {} heat sources",
                result.veins_deposited, heat_map.len()
            ),
            count: result.veins_deposited,
        });
    }
    if formation_total > 0 {
        let mut parts = Vec::new();
        if crystal_total > 0 { parts.push(format!("{} crystal", crystal_total)); }
        if calcite_total > 0 { parts.push(format!("{} calcite", calcite_total)); }
        if flowstone_total > 0 { parts.push(format!("{} flowstone", flowstone_total)); }
        result.transform_log.push(TransformEntry {
            description: format!(
                "The Veins \u{2014} 500,000 years: {} formations grown ({})",
                formation_total, parts.join(", ")
            ),
            count: formation_total,
        });
    }
    if ambient_flowstone_count > 0 {
        result.transform_log.push(TransformEntry {
            description: format!(
                "The Veins \u{2014} 500,000 years: {} flowstone deposited by ambient groundwater",
                ambient_flowstone_count
            ),
            count: ambient_flowstone_count,
        });
    }

    // --- Diagnostics ---
    let actual_output = result.veins_deposited + formation_total;
    result.diagnostics = PhaseDiagnostics {
        conversions,
        theoretical_max,
        actual_output,
        bottlenecks: compute_veins_bottlenecks(census, heat_map),
    };

    result
}

fn compute_veins_bottlenecks(census: &ResourceCensus, heat_map: &HeatMap) -> Vec<Bottleneck> {
    let mut bottlenecks = Vec::new();

    let total_heat = census.heat_source_lava + census.heat_source_kimberlite;
    if total_heat == 0 {
        bottlenecks.push(Bottleneck {
            severity: 1.0,
            description: "No heat sources \u{2014} hydrothermal veins require lava or kimberlite".into(),
        });
    } else if heat_map.is_empty() {
        bottlenecks.push(Bottleneck {
            severity: 0.9,
            description: "Heat sources have no air pathways \u{2014} mine tunnels toward heat sources".into(),
        });
    }

    if census.fissure_count > census.open_wall_count * 3 && census.fissure_count > 20 {
        bottlenecks.push(Bottleneck {
            severity: 0.4,
            description: format!(
                "{} fissures vs {} open walls \u{2014} mostly tight cracks, wider tunnels deposit more ore",
                census.fissure_count, census.open_wall_count
            ),
        });
    }

    // Check host rock availability near air
    let host_exposed: u32 = [
        Material::Sandstone as u8, Material::Limestone as u8,
        Material::Granite as u8, Material::Basalt as u8,
        Material::Slate as u8, Material::Marble as u8,
    ].iter().map(|m| census.exposed_surfaces_by_material.get(m).copied().unwrap_or(0)).sum();

    if host_exposed == 0 {
        bottlenecks.push(Bottleneck {
            severity: 0.8,
            description: "No exposed host rock adjacent to air \u{2014} mine into rock to create deposition surfaces".into(),
        });
    }

    if census.water.cell_count == 0 {
        bottlenecks.push(Bottleneck {
            severity: 0.3,
            description: "No water \u{2014} flowstone formation requires active water flow".into(),
        });
    }

    bottlenecks.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap_or(std::cmp::Ordering::Equal));
    bottlenecks.truncate(3);
    bottlenecks
}
