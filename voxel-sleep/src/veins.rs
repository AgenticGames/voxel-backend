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
use voxel_fluid::cell::FluidType;

use crate::aureole::HeatMap;
use crate::config::{VeinConfig, GroundwaterConfig};
use crate::groundwater::ambient_moisture;
use crate::manifest::ChangeManifest;
use crate::util::{FACE_OFFSETS, sample_material, count_neighbors, aperture_multiplier};
use crate::TransformEntry;

/// Result of the veins phase.
#[derive(Debug, Default)]
pub struct VeinResult {
    pub veins_deposited: u32,
    pub formations_grown: u32,
    pub manifest: ChangeManifest,
    pub glimpse_chunk: Option<(i32, i32, i32)>,
    pub transform_log: Vec<TransformEntry>,
}

/// Select ore type by BFS distance from heat source (temperature zonation).
fn select_ore_by_distance(
    config: &VeinConfig,
    distance: u32,
    rng: &mut ChaCha8Rng,
) -> Material {
    if distance < config.hypothermal_max {
        // Hypothermal (high-temperature): Tin / Quartz
        if rng.gen::<f32>() < 0.40 { Material::Tin } else { Material::Quartz }
    } else if distance < config.mesothermal_max {
        // Mesothermal (medium-temperature): Copper / Iron
        if rng.gen::<f32>() < 0.50 { Material::Copper } else { Material::Iron }
    } else {
        // Epithermal (low-temperature): Gold / Sulfide
        if rng.gen::<f32>() < 0.40 { Material::Gold } else { Material::Sulfide }
    }
}

/// Check if a material is a host rock (transformable by vein deposition).
fn is_host_rock(mat: Material) -> bool {
    matches!(mat,
        Material::Sandstone | Material::Limestone | Material::Granite |
        Material::Basalt | Material::Slate | Material::Marble
    )
}

/// Execute Phase 3: hydrothermal vein injection + formation growth.
pub fn apply_veins(
    config: &VeinConfig,
    groundwater: &GroundwaterConfig,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    fluid_snapshot: &FluidSnapshot,
    heat_map: &HeatMap,
    chunks: &[(i32, i32, i32)],
    chunk_size: usize,
    rng: &mut ChaCha8Rng,
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

            // Start BFS from neighbors of the heat source that are air
            for &(dx, dy, dz) in &FACE_OFFSETS {
                let nx = hx + dx;
                let ny = hy + dy;
                let nz = hz + dz;
                if let Some(mat) = sample_material(density_fields, nx, ny, nz, chunk_size) {
                    if !mat.is_solid() && !visited.contains(&(nx, ny, nz)) {
                        visited.insert((nx, ny, nz));
                        queue.push_back(((nx, ny, nz), 1));
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
                        let air_n = count_neighbors(density_fields, ax, ay, az, chunk_size, |m| !m.is_solid());
                        let eff_prob = config.vein_deposition_prob * if config.aperture_scaling_enabled {
                            aperture_multiplier(air_n)
                        } else { 1.0 };
                        if is_host_rock(mat) && rng.gen::<f32>() < eff_prob {
                            let ore = select_ore_by_distance(config, dist, rng);
                            let (ck, lx, ly, lz) = world_to_chunk_local(nx, ny, nz, chunk_size);
                            if let Some(df) = density_fields.get(&ck) {
                                let sample = df.get(lx, ly, lz);
                                vein_candidates.push(VeinCandidate {
                                    chunk_key: ck, lx, ly, lz,
                                    old_material: mat,
                                    old_density: sample.density,
                                    new_material: ore,
                                });
                                global_deposited.insert((nx, ny, nz));
                                source_deposited += 1;
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
    for c in &vein_candidates {
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
                        if cell.level <= 0.001 || cell.fluid_type != FluidType::Water {
                            continue;
                        }

                        let wx = cx * (cs as i32) + x as i32;
                        let wy = cy * (cs as i32) + y as i32;
                        let wz = cz * (cs as i32) + z as i32;

                        // Check air neighbors for flowstone deposition
                        for &(dx, dy, dz) in &FACE_OFFSETS {
                            let nx = wx + dx;
                            let ny = wy + dy;
                            let nz = wz + dz;

                            if let Some(mat) = sample_material(density_fields, nx, ny, nz, chunk_size) {
                                if mat == Material::Air && rng.gen::<f32>() < config.flowstone_prob {
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

                        // Check for limestone/sandstone ceiling (drip site)
                        let ceiling_mat = sample_material(density_fields, wx, wy + 1, wz, chunk_size);
                        let is_drip_site = matches!(ceiling_mat, Some(Material::Limestone) | Some(Material::Sandstone));
                        if !is_drip_site {
                            continue;
                        }

                        let ceiling = ceiling_mat.unwrap();
                        let moisture = ambient_moisture(groundwater, wy + 1, ceiling, true);
                        if moisture > 0.0 && rng.gen::<f32>() < config.flowstone_prob * moisture {
                            let count = flowstone_per_chunk.entry(chunk_key).or_insert(0);
                            if *count < config.flowstone_max_per_chunk {
                                growth_candidates.push(GrowthCandidate {
                                    chunk_key, lx, ly, lz,
                                    new_material: Material::Limestone,
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

    result
}
