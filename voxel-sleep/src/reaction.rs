//! Phase 1: "The Reaction" — 10,000 years.
//!
//! Fast chemistry: acid dissolution of limestone near exposed pyrite,
//! copper oxidation to malachite, basalt crust formation around lava.

use std::collections::{HashMap, HashSet, VecDeque};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use voxel_core::density::DensityField;
use voxel_core::material::Material;
use voxel_core::stress::world_to_chunk_local;
use voxel_fluid::FluidSnapshot;


use crate::config::ReactionConfig;
use crate::manifest::ChangeManifest;
use crate::util::{FACE_OFFSETS, sample_material, count_neighbors};
use crate::{Bottleneck, PhaseDiagnostics, ResourceCensus, TransformEntry};

/// Result of the reaction phase.
#[derive(Debug, Default)]
pub struct ReactionResult {
    pub acid_dissolved: u32,
    pub sulfide_dissolved: u32,
    pub voxels_oxidized: u32,
    pub gypsum_deposited: u32,
    pub manifest: ChangeManifest,
    pub glimpse_chunk: Option<(i32, i32, i32)>,
    pub transform_log: Vec<TransformEntry>,
    pub diagnostics: PhaseDiagnostics,
}

/// Execute Phase 1: acid dissolution, copper oxidation, basalt crust.
pub fn apply_reaction(
    config: &ReactionConfig,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    fluid_snapshot: &mut FluidSnapshot,
    chunks: &[(i32, i32, i32)],
    chunk_size: usize,
    rng: &mut ChaCha8Rng,
    census: &ResourceCensus,
) -> ReactionResult {
    let field_size = chunk_size + 1;
    let mut result = ReactionResult::default();

    // Collect all candidates first (two-pass: scan then apply)
    struct Candidate {
        chunk_key: (i32, i32, i32),
        lx: usize, ly: usize, lz: usize,
        old_material: Material,
        old_density: f32,
        new_material: Material,
        new_density: f32,
        change_type: u8, // 0=acid, 1=oxidation, 2=basalt_crust
    }

    let mut candidates: Vec<Candidate> = Vec::new();
    let mut theoretical_max = 0u32;

    // --- Acid Dissolution: BFS through limestone from exposed pyrite ---
    if config.acid_dissolution_enabled {
        // Step 1: Find all exposed pyrite sites (pyrite with >= 1 air neighbor)
        let mut pyrite_sites: Vec<(i32, i32, i32)> = Vec::new();

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
                        if sample.material != Material::Pyrite {
                            continue;
                        }

                        let wx = cx * (chunk_size as i32) + lx as i32;
                        let wy = cy * (chunk_size as i32) + ly as i32;
                        let wz = cz * (chunk_size as i32) + lz as i32;

                        let air_count = count_neighbors(
                            density_fields, wx, wy, wz, chunk_size,
                            |m| !m.is_solid(),
                        );
                        if air_count >= 1 {
                            pyrite_sites.push((wx, wy, wz));
                        }
                    }
                }
            }
        }

        // Step 2: BFS from each pyrite site through connected limestone
        let max_depth = config.acid_dissolution_radius as i32;
        let mut dissolved_set: HashSet<(i32, i32, i32)> = HashSet::new();
        let mut acid_bfs_visited: HashSet<(i32, i32, i32)> = HashSet::new();

        for &(px, py, pz) in &pyrite_sites {
            // Start BFS from limestone neighbors of this pyrite
            let mut queue: VecDeque<((i32, i32, i32), i32)> = VecDeque::new();
            let mut visited: HashSet<(i32, i32, i32)> = HashSet::new();
            let mut source_dissolved: u32 = 0;

            for &(dx, dy, dz) in &FACE_OFFSETS {
                let nx = px + dx;
                let ny = py + dy;
                let nz = pz + dz;
                if let Some(mat) = sample_material(density_fields, nx, ny, nz, chunk_size) {
                    if mat == Material::Limestone && !visited.contains(&(nx, ny, nz)) {
                        visited.insert((nx, ny, nz));
                        queue.push_back(((nx, ny, nz), 1));
                    }
                }
            }

            while let Some(((wx, wy, wz), depth)) = queue.pop_front() {
                if depth > max_depth {
                    continue;
                }

                acid_bfs_visited.insert((wx, wy, wz));

                // Roll probability for dissolution
                if rng.gen::<f32>() < config.acid_dissolution_prob {
                    if dissolved_set.insert((wx, wy, wz)) {
                        source_dissolved += 1;
                        if source_dissolved >= config.acid_max_dissolved_per_source {
                            break;
                        }
                    }
                }

                // Expand BFS to limestone neighbors
                if depth < max_depth {
                    for &(dx, dy, dz) in &FACE_OFFSETS {
                        let nx = wx + dx;
                        let ny = wy + dy;
                        let nz = wz + dz;
                        if !visited.contains(&(nx, ny, nz)) {
                            if let Some(mat) = sample_material(density_fields, nx, ny, nz, chunk_size) {
                                if mat == Material::Limestone {
                                    visited.insert((nx, ny, nz));
                                    queue.push_back(((nx, ny, nz), depth + 1));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Convert dissolved positions to candidates
        for &(wx, wy, wz) in &dissolved_set {
            let (chunk_key, lx, ly, lz) = world_to_chunk_local(wx, wy, wz, chunk_size);
            if let Some(df) = density_fields.get(&chunk_key) {
                let sample = df.get(lx, ly, lz);
                candidates.push(Candidate {
                    chunk_key, lx, ly, lz,
                    old_material: sample.material,
                    old_density: sample.density,
                    new_material: Material::Air,
                    new_density: -1.0,
                    change_type: 0,
                });
            }
        }

        theoretical_max += acid_bfs_visited.len() as u32;
    }

    // --- Copper Oxidation: copper + air neighbor → malachite ---
    if config.copper_oxidation_enabled {
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
                        if sample.material != Material::Copper {
                            continue;
                        }

                        let wx = cx * (chunk_size as i32) + lx as i32;
                        let wy = cy * (chunk_size as i32) + ly as i32;
                        let wz = cz * (chunk_size as i32) + lz as i32;

                        let air_count = count_neighbors(
                            density_fields, wx, wy, wz, chunk_size,
                            |m| !m.is_solid(),
                        );
                        if air_count >= 1 {
                            theoretical_max += 1;
                        }
                        if air_count >= 1 && rng.gen::<f32>() < config.copper_oxidation_prob {
                            candidates.push(Candidate {
                                chunk_key, lx, ly, lz,
                                old_material: sample.material,
                                old_density: sample.density,
                                new_material: Material::Malachite,
                                new_density: sample.density,
                                change_type: 1,
                            });
                        }
                    }
                }
            }
        }
    }

    // --- Basalt Crust: solid voxels adjacent to lava → basalt ---
    if config.basalt_crust_enabled && !fluid_snapshot.chunks.is_empty() {
        let cs = fluid_snapshot.chunk_size;
        // Collect all lava positions from fluid snapshot
        let mut lava_positions: Vec<(i32, i32, i32)> = Vec::new();
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
                            lava_positions.push((wx, wy, wz));
                        }
                    }
                }
            }
        }

        // For each lava position, check solid neighbors
        let mut crust_set: HashSet<(i32, i32, i32)> = HashSet::new();
        for &(lx, ly, lz) in &lava_positions {
            for &(dx, dy, dz) in &FACE_OFFSETS {
                let nx = lx + dx;
                let ny = ly + dy;
                let nz = lz + dz;
                if crust_set.contains(&(nx, ny, nz)) {
                    continue;
                }
                if let Some(mat) = sample_material(density_fields, nx, ny, nz, chunk_size) {
                    if mat.is_solid() && mat != Material::Basalt && mat != Material::Kimberlite {
                        theoretical_max += 1;
                        if rng.gen::<f32>() < config.basalt_crust_prob {
                            crust_set.insert((nx, ny, nz));
                        }
                    }
                }
            }
        }

        for &(wx, wy, wz) in &crust_set {
            let (chunk_key, lx, ly, lz) = world_to_chunk_local(wx, wy, wz, chunk_size);
            if let Some(df) = density_fields.get(&chunk_key) {
                let sample = df.get(lx, ly, lz);
                candidates.push(Candidate {
                    chunk_key, lx, ly, lz,
                    old_material: sample.material,
                    old_density: sample.density,
                    new_material: Material::Basalt,
                    new_density: sample.density,
                    change_type: 2,
                });
            }
        }
    }

    // --- Sulfide Acid Dissolution: BFS through limestone from exposed sulfide ---
    if config.sulfide_acid_enabled {
        // Step 1: Find all exposed sulfide sites (sulfide with >= 1 air neighbor)
        let mut sulfide_sites: Vec<(i32, i32, i32, bool)> = Vec::new(); // (x, y, z, has_water)

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
                        if sample.material != Material::Sulfide {
                            continue;
                        }

                        let wx = cx * (chunk_size as i32) + lx as i32;
                        let wy = cy * (chunk_size as i32) + ly as i32;
                        let wz = cz * (chunk_size as i32) + lz as i32;

                        let air_count = count_neighbors(
                            density_fields, wx, wy, wz, chunk_size,
                            |m| !m.is_solid(),
                        );
                        if air_count >= 1 {
                            // Check for adjacent water
                            let has_water = {
                                let cs = fluid_snapshot.chunk_size;
                                let mut found = false;
                                for &(dx, dy, dz) in &FACE_OFFSETS {
                                    let nwx = wx + dx;
                                    let nwy = wy + dy;
                                    let nwz = wz + dz;
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
                                found
                            };
                            sulfide_sites.push((wx, wy, wz, has_water));
                        }
                    }
                }
            }
        }

        // Step 2: BFS from each sulfide site through connected limestone
        let base_radius = config.sulfide_acid_radius;
        let mut sulfide_dissolved_set: HashSet<(i32, i32, i32)> = HashSet::new();
        let mut sulfide_bfs_visited: HashSet<(i32, i32, i32)> = HashSet::new();

        for &(sx, sy, sz, has_water) in &sulfide_sites {
            // Sulfide oxidation requires water as reactant (FeS₂ + 7O₂ + 2H₂O → ...)
            if !has_water {
                continue;
            }
            let effective_radius = (base_radius as f32 * config.sulfide_water_amplification * config.limestone_acid_radius_boost) as i32;

            let mut queue: VecDeque<((i32, i32, i32), i32)> = VecDeque::new();
            let mut visited: HashSet<(i32, i32, i32)> = HashSet::new();
            let mut source_dissolved: u32 = 0;

            for &(dx, dy, dz) in &FACE_OFFSETS {
                let nx = sx + dx;
                let ny = sy + dy;
                let nz = sz + dz;
                if let Some(mat) = sample_material(density_fields, nx, ny, nz, chunk_size) {
                    if mat == Material::Limestone && !visited.contains(&(nx, ny, nz)) {
                        visited.insert((nx, ny, nz));
                        queue.push_back(((nx, ny, nz), 1));
                    }
                }
            }

            while let Some(((wx, wy, wz), depth)) = queue.pop_front() {
                if depth > effective_radius {
                    continue;
                }

                sulfide_bfs_visited.insert((wx, wy, wz));

                if rng.gen::<f32>() < config.sulfide_acid_prob {
                    if sulfide_dissolved_set.insert((wx, wy, wz)) {
                        source_dissolved += 1;
                        if source_dissolved >= config.acid_max_dissolved_per_source {
                            break;
                        }
                    }
                }

                if depth < effective_radius {
                    for &(dx, dy, dz) in &FACE_OFFSETS {
                        let nx = wx + dx;
                        let ny = wy + dy;
                        let nz = wz + dz;
                        if !visited.contains(&(nx, ny, nz)) {
                            if let Some(mat) = sample_material(density_fields, nx, ny, nz, chunk_size) {
                                if mat == Material::Limestone {
                                    visited.insert((nx, ny, nz));
                                    queue.push_back(((nx, ny, nz), depth + 1));
                                }
                            }
                        }
                    }
                }
            }
        }

        theoretical_max += sulfide_bfs_visited.len() as u32;

        // Convert sulfide dissolved positions to candidates
        for &(wx, wy, wz) in &sulfide_dissolved_set {
            let (chunk_key, lx, ly, lz) = world_to_chunk_local(wx, wy, wz, chunk_size);
            if let Some(df) = density_fields.get(&chunk_key) {
                let sample = df.get(lx, ly, lz);
                candidates.push(Candidate {
                    chunk_key, lx, ly, lz,
                    old_material: sample.material,
                    old_density: sample.density,
                    new_material: Material::Air,
                    new_density: -1.0,
                    change_type: 3, // sulfide acid
                });
            }
        }

        // Gypsum deposition: CaCO₃ + H₂SO₄ → CaSO₄·2H₂O (gypsum forms on void walls)
        if config.gypsum_enabled {
            let dissolved_positions: Vec<(i32, i32, i32)> = sulfide_dissolved_set.iter().copied().collect();
            for &(wx, wy, wz) in &dissolved_positions {
                // Check face-adjacent voxels of the new void
                for &(dx, dy, dz) in &FACE_OFFSETS {
                    let nx = wx + dx;
                    let ny = wy + dy;
                    let nz = wz + dz;
                    if sulfide_dissolved_set.contains(&(nx, ny, nz)) { continue; }
                    if let Some(mat) = sample_material(density_fields, nx, ny, nz, chunk_size) {
                        if mat == Material::Limestone && rng.gen::<f32>() < config.gypsum_deposition_prob {
                            let (gck, glx, gly, glz) = world_to_chunk_local(nx, ny, nz, chunk_size);
                            if let Some(df) = density_fields.get(&gck) {
                                let gsample = df.get(glx, gly, glz);
                                candidates.push(Candidate {
                                    chunk_key: gck, lx: glx, ly: gly, lz: glz,
                                    old_material: gsample.material,
                                    old_density: gsample.density,
                                    new_material: Material::Gypsum,
                                    new_density: gsample.density,
                                    change_type: 4, // gypsum deposition
                                });
                            }
                        }
                    }
                }
            }
        }

        // Drain water near sulfide sites (acid consumes water: 0.1 per BFS step)
        let cs = fluid_snapshot.chunk_size;
        for &(sx, sy, sz, has_water) in &sulfide_sites {
            if !has_water { continue; }
            let drain_amount = 0.1;
            for &(dx, dy, dz) in &FACE_OFFSETS {
                let nwx = sx + dx;
                let nwy = sy + dy;
                let nwz = sz + dz;
                let fck = (
                    nwx.div_euclid(cs as i32),
                    nwy.div_euclid(cs as i32),
                    nwz.div_euclid(cs as i32),
                );
                if let Some(cells) = fluid_snapshot.chunks.get_mut(&fck) {
                    let flx = nwx.rem_euclid(cs as i32) as usize;
                    let fly = nwy.rem_euclid(cs as i32) as usize;
                    let flz = nwz.rem_euclid(cs as i32) as usize;
                    let idx = flz * cs * cs + fly * cs + flx;
                    if idx < cells.len() && !cells[idx].is_source && cells[idx].level > 0.001 && cells[idx].fluid_type.is_water() {
                        cells[idx].level = (cells[idx].level - drain_amount).max(0.0);
                    }
                }
            }
        }
    }

    // --- Apply all candidates ---
    let mut acid_count = 0u32;
    let mut oxidation_count = 0u32;
    let mut basalt_count = 0u32;
    let mut sulfide_acid_count = 0u32;
    let mut gypsum_count = 0u32;
    let mut conversions: std::collections::BTreeMap<(u8, u8), u32> = std::collections::BTreeMap::new();

    for c in &candidates {
        *conversions.entry((c.old_material as u8, c.new_material as u8)).or_insert(0) += 1;
        if let Some(df) = density_fields.get_mut(&c.chunk_key) {
            let sample = df.get_mut(c.lx, c.ly, c.lz);
            sample.material = c.new_material;
            sample.density = c.new_density;
        }

        result.manifest.record_voxel_change(
            c.chunk_key, c.lx, c.ly, c.lz,
            c.old_material, c.old_density,
            c.new_material, c.new_density,
        );

        if result.glimpse_chunk.is_none() {
            result.glimpse_chunk = Some(c.chunk_key);
        }

        match c.change_type {
            0 => acid_count += 1,
            1 => oxidation_count += 1,
            2 => basalt_count += 1,
            3 => sulfide_acid_count += 1,
            4 => gypsum_count += 1,
            _ => {}
        }
    }

    result.acid_dissolved = acid_count;
    result.sulfide_dissolved = sulfide_acid_count;
    result.voxels_oxidized = oxidation_count + basalt_count;
    result.gypsum_deposited = gypsum_count;

    // Build transform log
    if acid_count > 0 {
        result.transform_log.push(TransformEntry {
            description: format!("The Reaction \u{2014} 10,000 years: Acid dissolved {} limestone voxels", acid_count),
            count: acid_count,
        });
    }
    if oxidation_count > 0 {
        result.transform_log.push(TransformEntry {
            description: format!("The Reaction \u{2014} 10,000 years: {} copper surfaces oxidized to malachite", oxidation_count),
            count: oxidation_count,
        });
    }
    if basalt_count > 0 {
        result.transform_log.push(TransformEntry {
            description: format!("The Reaction \u{2014} 10,000 years: {} basalt crust formed around lava", basalt_count),
            count: basalt_count,
        });
    }
    if sulfide_acid_count > 0 {
        result.transform_log.push(TransformEntry {
            description: format!("The Reaction \u{2014} 10,000 years: Sulfide acid dissolved {} limestone voxels", sulfide_acid_count),
            count: sulfide_acid_count,
        });
    }
    if gypsum_count > 0 {
        result.transform_log.push(TransformEntry {
            description: format!("The Reaction \u{2014} 10,000 years: {} gypsum crusts deposited on acid-dissolved walls", gypsum_count),
            count: gypsum_count,
        });
    }

    // --- Diagnostics ---
    let actual_output = candidates.len() as u32;
    result.diagnostics = PhaseDiagnostics {
        conversions,
        theoretical_max,
        actual_output,
        bottlenecks: compute_reaction_bottlenecks(census),
    };

    result
}

pub(crate) fn compute_reaction_bottlenecks(census: &ResourceCensus) -> Vec<Bottleneck> {
    use voxel_core::material::Material;
    let mut bottlenecks = Vec::new();

    let exposed_pyrite = census.exposed_ore.get(&(Material::Pyrite as u8)).copied().unwrap_or(0);
    let exposed_sulfide = census.exposed_ore.get(&(Material::Sulfide as u8)).copied().unwrap_or(0);
    let exposed_copper = census.exposed_ore.get(&(Material::Copper as u8)).copied().unwrap_or(0);

    if exposed_pyrite == 0 && exposed_sulfide == 0 {
        bottlenecks.push(Bottleneck {
            severity: 1.0,
            description: "No exposed pyrite or sulfide \u{2014} acid dissolution cannot start".into(),
        });
    } else if exposed_pyrite + exposed_sulfide < 5 {
        bottlenecks.push(Bottleneck {
            severity: 0.7,
            description: format!(
                "Only {} exposed pyrite + {} sulfide cells \u{2014} mine more to expose acid sources",
                exposed_pyrite, exposed_sulfide
            ),
        });
    }

    if census.water.cell_count == 0 {
        bottlenecks.push(Bottleneck {
            severity: 0.5,
            description: "No water detected \u{2014} sulfide acid gets 2\u{00d7} radius with water contact".into(),
        });
    }

    if exposed_copper == 0 {
        bottlenecks.push(Bottleneck {
            severity: 0.4,
            description: "No exposed copper \u{2014} oxidation to malachite needs air-adjacent copper".into(),
        });
    } else if exposed_copper < 5 {
        bottlenecks.push(Bottleneck {
            severity: 0.2,
            description: format!(
                "Only {} exposed copper cells \u{2014} mine more to expose copper for oxidation",
                exposed_copper
            ),
        });
    }

    if census.lava.cell_count == 0 {
        bottlenecks.push(Bottleneck {
            severity: 0.3,
            description: "No lava detected \u{2014} basalt crust formation needs active lava".into(),
        });
    }

    bottlenecks.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap_or(std::cmp::Ordering::Equal));
    bottlenecks.truncate(3);
    bottlenecks
}
