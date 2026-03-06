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
use crate::TransformEntry;

/// Result of the reaction phase.
#[derive(Debug, Default)]
pub struct ReactionResult {
    pub acid_dissolved: u32,
    pub sulfide_dissolved: u32,
    pub voxels_oxidized: u32,
    pub manifest: ChangeManifest,
    pub glimpse_chunk: Option<(i32, i32, i32)>,
    pub transform_log: Vec<TransformEntry>,
}

/// Execute Phase 1: acid dissolution, copper oxidation, basalt crust.
pub fn apply_reaction(
    config: &ReactionConfig,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    fluid_snapshot: &FluidSnapshot,
    chunks: &[(i32, i32, i32)],
    chunk_size: usize,
    rng: &mut ChaCha8Rng,
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

        for &(px, py, pz) in &pyrite_sites {
            // Start BFS from limestone neighbors of this pyrite
            let mut queue: VecDeque<((i32, i32, i32), i32)> = VecDeque::new();
            let mut visited: HashSet<(i32, i32, i32)> = HashSet::new();

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

                // Roll probability for dissolution
                if rng.gen::<f32>() < config.acid_dissolution_prob {
                    dissolved_set.insert((wx, wy, wz));
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

        for &(sx, sy, sz, has_water) in &sulfide_sites {
            let effective_radius = if has_water {
                (base_radius as f32 * config.sulfide_water_amplification) as i32
            } else {
                base_radius as i32
            };

            let mut queue: VecDeque<((i32, i32, i32), i32)> = VecDeque::new();
            let mut visited: HashSet<(i32, i32, i32)> = HashSet::new();

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

                if rng.gen::<f32>() < config.sulfide_acid_prob {
                    sulfide_dissolved_set.insert((wx, wy, wz));
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
    }

    // --- Apply all candidates ---
    let mut acid_count = 0u32;
    let mut oxidation_count = 0u32;
    let mut basalt_count = 0u32;
    let mut sulfide_acid_count = 0u32;

    for c in &candidates {
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
            _ => {}
        }
    }

    result.acid_dissolved = acid_count;
    result.sulfide_dissolved = sulfide_acid_count;
    result.voxels_oxidized = oxidation_count + basalt_count;

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

    result
}
