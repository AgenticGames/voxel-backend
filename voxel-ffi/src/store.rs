use std::collections::{HashMap, HashSet};

use glam::Vec3;
use voxel_core::dual_contouring::mesh_gen::generate_mesh;
use voxel_core::dual_contouring::solve::solve_dc_vertices;
use voxel_core::hermite::HermiteData;
use voxel_core::material::Material;
use voxel_gen::config::GenerationConfig;
use voxel_gen::density::DensityField;
use voxel_gen::hermite_extract::{extract_hermite_data, patch_hermite_data};
use voxel_gen::region_gen::{self, ChunkSeamData};

use crate::convert::convert_mesh_to_ue;
use crate::types::{ConvertedMesh, FfiMinedMaterials};

/// Per-chunk cached data needed for mining and re-meshing.
pub struct ChunkStore {
    pub density_fields: HashMap<(i32, i32, i32), DensityField>,
    pub hermite_data: HashMap<(i32, i32, i32), HermiteData>,
    /// Tracks which regions have had their densities generated (with global worms).
    generated_regions: HashSet<(i32, i32, i32)>,
    /// Per-chunk seam data (DC vertices + boundary edges) for seam stitching.
    pub chunk_seam_data: HashMap<(i32, i32, i32), ChunkSeamData>,
}

impl ChunkStore {
    pub fn new() -> Self {
        Self {
            density_fields: HashMap::new(),
            hermite_data: HashMap::new(),
            generated_regions: HashSet::new(),
            chunk_seam_data: HashMap::new(),
        }
    }

    pub fn has_density(&self, key: &(i32, i32, i32)) -> bool {
        self.density_fields.contains_key(key)
    }

    pub fn is_region_generated(&self, region_key: &(i32, i32, i32)) -> bool {
        self.generated_regions.contains(region_key)
    }

    pub fn mark_region_generated(&mut self, region_key: (i32, i32, i32)) {
        self.generated_regions.insert(region_key);
    }

    pub fn chunks_loaded(&self) -> usize {
        self.density_fields.len()
    }

    pub fn insert(&mut self, key: (i32, i32, i32), density: DensityField, hermite: HermiteData) {
        self.density_fields.insert(key, density);
        self.hermite_data.insert(key, hermite);
    }

    pub fn unload(&mut self, key: (i32, i32, i32)) {
        self.density_fields.remove(&key);
        self.hermite_data.remove(&key);
        self.chunk_seam_data.remove(&key);
    }

    /// Cache seam data for a chunk.
    pub fn add_seam_data(
        &mut self,
        chunk: (i32, i32, i32),
        seam_data: ChunkSeamData,
    ) {
        self.chunk_seam_data.insert(chunk, seam_data);
    }

    /// Mine a sphere: set solid voxels within radius to Air.
    /// Returns the re-meshed dirty chunks (in UE coords) and mined material counts.
    pub fn mine_sphere(
        &mut self,
        center: Vec3,
        radius: f32,
        config: &GenerationConfig,
        world_scale: f32,
    ) -> (Vec<((i32, i32, i32), ConvertedMesh)>, FfiMinedMaterials) {
        let cs = config.chunk_size as f32;
        let r2 = radius * radius;
        let mut mined_counts = [0u32; 19];

        let min_cx = ((center.x - radius) / cs).floor() as i32;
        let max_cx = ((center.x + radius) / cs).floor() as i32;
        let min_cy = ((center.y - radius) / cs).floor() as i32;
        let max_cy = ((center.y + radius) / cs).floor() as i32;
        let min_cz = ((center.z - radius) / cs).floor() as i32;
        let max_cz = ((center.z + radius) / cs).floor() as i32;

        let mut dirty_chunks: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> =
            Vec::new();

        for cz in min_cz..=max_cz {
            for cy in min_cy..=max_cy {
                for cx in min_cx..=max_cx {
                    if let Some(density) = self.density_fields.get_mut(&(cx, cy, cz)) {
                        let origin =
                            Vec3::new(cx as f32 * cs, cy as f32 * cs, cz as f32 * cs);
                        let mut changed = false;

                        let local_center = center - origin;
                        let lo_x = ((local_center.x - radius).floor() as i32).max(0) as usize;
                        let hi_x =
                            ((local_center.x + radius).ceil() as usize + 1).min(density.size);
                        let lo_y = ((local_center.y - radius).floor() as i32).max(0) as usize;
                        let hi_y =
                            ((local_center.y + radius).ceil() as usize + 1).min(density.size);
                        let lo_z = ((local_center.z - radius).floor() as i32).max(0) as usize;
                        let hi_z =
                            ((local_center.z + radius).ceil() as usize + 1).min(density.size);

                        for z in lo_z..hi_z {
                            for y in lo_y..hi_y {
                                for x in lo_x..hi_x {
                                    let world_pos =
                                        origin + Vec3::new(x as f32, y as f32, z as f32);
                                    let dist2 = (world_pos - center).length_squared();
                                    if dist2 <= r2 {
                                        let sample = density.get_mut(x, y, z);
                                        if sample.material.is_solid() {
                                            mined_counts[sample.material as u8 as usize] += 1;
                                            // SDF: smooth gradient following sphere curvature
                                            // instead of flat -1.0 which kills DC normals
                                            let sdf = dist2.sqrt() - radius;
                                            sample.density = sdf.min(sample.density);
                                            sample.material = Material::Air;
                                            changed = true;
                                        }
                                    }
                                }
                            }
                        }

                        if changed {
                            let d_min_x = lo_x.saturating_sub(1);
                            let d_min_y = lo_y.saturating_sub(1);
                            let d_min_z = lo_z.saturating_sub(1);
                            let d_max_x = hi_x.min(density.size - 1);
                            let d_max_y = hi_y.min(density.size - 1);
                            let d_max_z = hi_z.min(density.size - 1);
                            dirty_chunks.push((
                                (cx, cy, cz),
                                d_min_x, d_min_y, d_min_z,
                                d_max_x, d_max_y, d_max_z,
                            ));
                        }
                    }
                }
            }
        }

        let meshes = self.remesh_dirty(&dirty_chunks, config, world_scale);
        (meshes, FfiMinedMaterials { counts: mined_counts })
    }

    /// Mine by peeling: only remove surface voxels within radius.
    pub fn mine_peel(
        &mut self,
        center: Vec3,
        normal: Vec3,
        radius: f32,
        config: &GenerationConfig,
        world_scale: f32,
    ) -> (Vec<((i32, i32, i32), ConvertedMesh)>, FfiMinedMaterials) {
        let cs = config.chunk_size as f32;
        let r2 = radius * radius;
        let mut mined_counts = [0u32; 19];
        let adjusted_center = center - normal * 0.5;

        let min_cx = ((adjusted_center.x - radius) / cs).floor() as i32;
        let max_cx = ((adjusted_center.x + radius) / cs).floor() as i32;
        let min_cy = ((adjusted_center.y - radius) / cs).floor() as i32;
        let max_cy = ((adjusted_center.y + radius) / cs).floor() as i32;
        let min_cz = ((adjusted_center.z - radius) / cs).floor() as i32;
        let max_cz = ((adjusted_center.z + radius) / cs).floor() as i32;

        let mut dirty_chunks: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> =
            Vec::new();

        for cz in min_cz..=max_cz {
            for cy in min_cy..=max_cy {
                for cx in min_cx..=max_cx {
                    if let Some(density) = self.density_fields.get_mut(&(cx, cy, cz)) {
                        let origin =
                            Vec3::new(cx as f32 * cs, cy as f32 * cs, cz as f32 * cs);
                        let mut changed = false;

                        let local_center = adjusted_center - origin;
                        let lo_x = ((local_center.x - radius).floor() as i32).max(0) as usize;
                        let hi_x =
                            ((local_center.x + radius).ceil() as usize + 1).min(density.size);
                        let lo_y = ((local_center.y - radius).floor() as i32).max(0) as usize;
                        let hi_y =
                            ((local_center.y + radius).ceil() as usize + 1).min(density.size);
                        let lo_z = ((local_center.z - radius).floor() as i32).max(0) as usize;
                        let hi_z =
                            ((local_center.z + radius).ceil() as usize + 1).min(density.size);

                        // First pass: collect voxels to peel
                        let mut to_peel: Vec<(usize, usize, usize)> = Vec::new();
                        for z in lo_z..hi_z {
                            for y in lo_y..hi_y {
                                for x in lo_x..hi_x {
                                    let world_pos =
                                        origin + Vec3::new(x as f32, y as f32, z as f32);
                                    let dist2 = (world_pos - adjusted_center).length_squared();
                                    if dist2 > r2 {
                                        continue;
                                    }

                                    let sample = density.get(x, y, z);
                                    if !sample.material.is_solid() {
                                        continue;
                                    }

                                    let near_surface =
                                        sample.density < 0.5
                                            || has_air_neighbor(density, x, y, z);
                                    if near_surface {
                                        to_peel.push((x, y, z));
                                    }
                                }
                            }
                        }

                        // Second pass: apply peeling with SDF gradient
                        for (x, y, z) in to_peel {
                            let world_pos =
                                origin + Vec3::new(x as f32, y as f32, z as f32);
                            let dist = (world_pos - adjusted_center).length();
                            let sdf = dist - radius;
                            let sample = density.get_mut(x, y, z);
                            mined_counts[sample.material as u8 as usize] += 1;
                            sample.density = sdf.min(sample.density);
                            sample.material = Material::Air;
                            changed = true;
                        }

                        if changed {
                            let d_min_x = lo_x.saturating_sub(1);
                            let d_min_y = lo_y.saturating_sub(1);
                            let d_min_z = lo_z.saturating_sub(1);
                            let d_max_x = hi_x.min(density.size - 1);
                            let d_max_y = hi_y.min(density.size - 1);
                            let d_max_z = hi_z.min(density.size - 1);
                            dirty_chunks.push((
                                (cx, cy, cz),
                                d_min_x, d_min_y, d_min_z,
                                d_max_x, d_max_y, d_max_z,
                            ));
                        }
                    }
                }
            }
        }

        let meshes = self.remesh_dirty(&dirty_chunks, config, world_scale);
        (meshes, FfiMinedMaterials { counts: mined_counts })
    }

    /// Re-mesh dirty chunks using incremental hermite patching.
    /// Returns converted meshes in UE coordinate space.
    /// Also updates seam data so seam stitching reflects post-mining geometry.
    fn remesh_dirty(
        &mut self,
        dirty_chunks: &[((i32, i32, i32), usize, usize, usize, usize, usize, usize)],
        config: &GenerationConfig,
        world_scale: f32,
    ) -> Vec<((i32, i32, i32), ConvertedMesh)> {
        let max_edge_length = config.max_edge_length;
        let chunk_size = config.chunk_size;
        let mut results = Vec::with_capacity(dirty_chunks.len());

        for &(key, min_x, min_y, min_z, max_x, max_y, max_z) in dirty_chunks {
            let density = match self.density_fields.get(&key) {
                Some(d) => d,
                None => continue,
            };

            let hermite = match self.hermite_data.get_mut(&key) {
                Some(h) => {
                    patch_hermite_data(h, density, min_x, min_y, min_z, max_x, max_y, max_z);
                    h
                }
                None => {
                    let h = extract_hermite_data(density);
                    self.hermite_data.insert(key, h);
                    self.hermite_data.get(&key).unwrap()
                }
            };

            let cell_size = density.size - 1;
            let dc_vertices = solve_dc_vertices(hermite, cell_size);
            let mesh = generate_mesh(hermite, &dc_vertices, cell_size, max_edge_length);

            // Update seam data so seam stitching uses post-mining geometry
            let boundary_edges = region_gen::extract_boundary_edges(hermite, chunk_size);
            self.chunk_seam_data.insert(
                key,
                ChunkSeamData {
                    dc_vertices,
                    world_origin: Vec3::ZERO,
                    boundary_edges,
                },
            );

            let converted = convert_mesh_to_ue(&mesh, world_scale);
            results.push((key, converted));
        }

        results
    }

    /// Find the best spring location (wall seep / ceiling drip) near the player.
    /// Scans all loaded density fields for open cavern cells adjacent to a wall.
    /// Returns the world-space (Rust coords) position of the best candidate.
    pub fn find_spring_location(&self, player_pos: Vec3, chunk_size: usize) -> Option<Vec3> {
        let cs = chunk_size as f32;
        let mut best_score: f32 = -1.0;
        let mut best_pos: Option<Vec3> = None;

        for (&(cx, cy, cz), density) in &self.density_fields {
            let origin = Vec3::new(cx as f32 * cs, cy as f32 * cs, cz as f32 * cs);

            // Skip border cells (1..chunk_size-1) to avoid neighbor OOB on the +1 grid
            for z in 1..(chunk_size - 1) {
                for y in 1..(chunk_size - 1) {
                    for x in 1..(chunk_size - 1) {
                        let sample = density.get(x, y, z);
                        // Must be air
                        if sample.material.is_solid() {
                            continue;
                        }

                        // Count air neighbors in 3x3x3 Moore neighborhood
                        let mut air_count: u32 = 0;
                        for dz in -1i32..=1 {
                            for dy in -1i32..=1 {
                                for dx in -1i32..=1 {
                                    if dx == 0 && dy == 0 && dz == 0 {
                                        continue;
                                    }
                                    let nx = (x as i32 + dx) as usize;
                                    let ny = (y as i32 + dy) as usize;
                                    let nz = (z as i32 + dz) as usize;
                                    if !density.get(nx, ny, nz).material.is_solid() {
                                        air_count += 1;
                                    }
                                }
                            }
                        }

                        // Cavern test: need >= 15 air neighbors out of 26
                        if air_count < 15 {
                            continue;
                        }

                        // Check face neighbors for wall/ceiling adjacency
                        let solid_above = density.get(x, y + 1, z).material.is_solid();
                        let solid_below = density.get(x, y.wrapping_sub(1), z).material.is_solid();
                        let solid_xp = density.get(x + 1, y, z).material.is_solid();
                        let solid_xn = density.get(x.wrapping_sub(1), y, z).material.is_solid();
                        let solid_zp = density.get(x, y, z + 1).material.is_solid();
                        let solid_zn = density.get(x, y, z.wrapping_sub(1)).material.is_solid();

                        let has_solid_side = solid_xp || solid_xn || solid_zp || solid_zn;
                        let air_below = !solid_below;

                        // Must have at least one solid face neighbor
                        if !solid_above && !has_solid_side {
                            continue;
                        }

                        // Scoring
                        let wall_bonus = if has_solid_side && !solid_above {
                            2.0_f32 // wall seep (preferred)
                        } else if solid_above {
                            1.5 // ceiling drip
                        } else {
                            1.0
                        };

                        let air_below_bonus = if air_below { 1.5_f32 } else { 1.0 };

                        let world_pos = origin + Vec3::new(x as f32, y as f32, z as f32);
                        let distance = (world_pos - player_pos).length();

                        let score = (air_count as f32) * wall_bonus * air_below_bonus
                            / (1.0 + distance);

                        if score > best_score {
                            best_score = score;
                            best_pos = Some(world_pos);
                        }
                    }
                }
            }
        }

        best_pos
    }
}

/// Extract a solid mask bitfield from a density field.
///
/// One bit per voxel for the inner chunk_size^3 grid (not the +1 border).
/// Solid = density > 0.0 or material.is_solid().
pub fn extract_solid_mask(density: &DensityField, chunk_size: usize) -> Vec<u64> {
    let total = chunk_size * chunk_size * chunk_size;
    let num_words = (total + 63) / 64;
    let mut mask = vec![0u64; num_words];

    for z in 0..chunk_size {
        for y in 0..chunk_size {
            for x in 0..chunk_size {
                let sample = density.get(x, y, z);
                if sample.material.is_solid() {
                    let idx = z * chunk_size * chunk_size + y * chunk_size + x;
                    let word = idx / 64;
                    let bit = idx % 64;
                    mask[word] |= 1u64 << bit;
                }
            }
        }
    }

    mask
}

fn has_air_neighbor(density: &DensityField, x: usize, y: usize, z: usize) -> bool {
    let s = density.size;
    let neighbors = [
        (x.wrapping_sub(1), y, z),
        (x + 1, y, z),
        (x, y.wrapping_sub(1), z),
        (x, y + 1, z),
        (x, y, z.wrapping_sub(1)),
        (x, y, z + 1),
    ];
    for (nx, ny, nz) in neighbors {
        if nx < s && ny < s && nz < s && !density.get(nx, ny, nz).material.is_solid() {
            return true;
        }
    }
    false
}
