use std::collections::HashMap;
use glam::Vec3;
use rayon::prelude::*;
use voxel_core::chunk::ChunkCoord;
use voxel_core::dual_contouring::mesh_gen::generate_mesh;
use voxel_core::dual_contouring::solve::solve_dc_vertices;
use voxel_core::export::{mesh_to_json_multi, JsonMesh};
use voxel_core::hermite::HermiteData;
use voxel_core::material::Material;
use voxel_core::mesh::Mesh;
use voxel_core::stress::{StressField, SupportField};
use voxel_gen::config::GenerationConfig;
use voxel_gen::density::DensityField;
use voxel_gen::hermite_extract::{extract_hermite_data, patch_hermite_data};
use voxel_gen::pools::{self, PoolDescriptor};
use voxel_gen::region_gen::{self, ChunkSeamData};

/// Keeps density fields alive for re-meshing after mining.
pub struct GeneratedRegion {
    pub config: GenerationConfig,
    pub density_fields: HashMap<(i32, i32, i32), DensityField>,
    pub chunk_meshes: HashMap<(i32, i32, i32), Mesh>,
    hermite_data: HashMap<(i32, i32, i32), HermiteData>,
    chunk_seam_data: HashMap<(i32, i32, i32), ChunkSeamData>,
    seam_mesh: Mesh,
    pub stress_fields: HashMap<(i32, i32, i32), StressField>,
    pub support_fields: HashMap<(i32, i32, i32), SupportField>,
    pub pool_descriptors: Vec<PoolDescriptor>,
}

pub struct MineResult {
    pub mined_materials: HashMap<Material, u32>,
}

impl GeneratedRegion {
    /// Generate a region using the same 6-phase pipeline as generate.rs.
    /// Phases 1 and 6 are parallelized across chunks with rayon.
    pub fn generate(
        config: GenerationConfig,
        range_min: (i32, i32, i32),
        range_max: (i32, i32, i32),
        closed: bool,
    ) -> Self {
        let gs = config.chunk_size;

        // Build coordinate list for all chunks
        let coords: Vec<(i32, i32, i32)> = (range_min.2..range_max.2)
            .flat_map(|cz| {
                (range_min.1..range_max.1).flat_map(move |cy| {
                    (range_min.0..range_max.0).map(move |cx| (cx, cy, cz))
                })
            })
            .collect();

        // Phases 1-5: Generate density fields with global worm carving + pools (shared pipeline)
        let (mut density_fields, pool_descriptors, _worm_paths) = region_gen::generate_region_densities(&coords, &config);

        // Phase 5: Seal boundary faces
        if closed {
            for (&(cx, cy, cz), density) in &mut density_fields {
                density.clamp_boundary_faces(
                    cx == range_min.0,
                    cx == range_max.0 - 1,
                    cy == range_min.1,
                    cy == range_max.1 - 1,
                    cz == range_min.2,
                    cz == range_max.2 - 1,
                );
            }
        }

        // Phase 6: Extract hermite, mesh per chunk, collect seam data (PARALLEL)
        let chunk_results: Vec<_> = coords
            .par_iter()
            .map(|&(cx, cy, cz)| {
                let density = &density_fields[&(cx, cy, cz)];
                let coord = ChunkCoord::new(cx, cy, cz);

                let hermite = extract_hermite_data(density);
                let cell_size = density.size - 1;
                let dc_vertices = solve_dc_vertices(&hermite, cell_size);
                let mut mesh = generate_mesh(&hermite, &dc_vertices, cell_size, config.max_edge_length, 0.0);

                let boundary_edges = region_gen::extract_boundary_edges(&hermite, gs);

                let world_origin = coord.world_origin_sized(gs);
                for v in &mut mesh.vertices {
                    v.position += world_origin;
                }

                ((cx, cy, cz), hermite, mesh, ChunkSeamData {
                    dc_vertices,
                    world_origin,
                    boundary_edges,
                })
            })
            .collect();

        let mut chunk_meshes: HashMap<(i32, i32, i32), Mesh> = HashMap::with_capacity(coords.len());
        let mut hermite_data: HashMap<(i32, i32, i32), HermiteData> = HashMap::with_capacity(coords.len());
        let mut chunk_seam_data: HashMap<(i32, i32, i32), ChunkSeamData> = HashMap::with_capacity(coords.len());
        for (key, hermite, mesh, seam_data) in chunk_results {
            chunk_meshes.insert(key, mesh);
            hermite_data.insert(key, hermite);
            chunk_seam_data.insert(key, seam_data);
        }

        // Generate seam mesh to fill gaps between adjacent chunks
        let seam_mesh = region_gen::generate_seam_mesh(&chunk_seam_data, gs);

        // Initialize stress and support fields for sleep system
        let chunk_size = config.chunk_size;
        let grid_size = chunk_size + 1;
        let mut stress_fields = HashMap::with_capacity(coords.len());
        let mut support_fields = HashMap::with_capacity(coords.len());
        for &key in density_fields.keys() {
            stress_fields.insert(key, StressField::new(grid_size));
            support_fields.insert(key, SupportField::new(grid_size));
        }

        GeneratedRegion {
            config,
            density_fields,
            chunk_meshes,
            hermite_data,
            chunk_seam_data,
            seam_mesh,
            stress_fields,
            support_fields,
            pool_descriptors,
        }
    }

    /// Run a deep sleep cycle on this region.
    /// Returns (sleep_result, updated_mesh_json).
    pub fn apply_sleep(&mut self, sleep_config: &voxel_sleep::SleepConfig) -> (voxel_sleep::SleepResult, JsonMesh) {

        // Use center chunk as player position
        let player_chunk = (0, 0, 0);

        let result = voxel_sleep::execute_sleep(
            sleep_config,
            &mut self.density_fields,
            &mut self.stress_fields,
            &mut self.support_fields,
            player_chunk,
            1, // sleep_count
            None, // no progress channel
        );

        // Re-mesh all dirty chunks
        if !result.dirty_chunks.is_empty() {
            // Build full-chunk dirty bounds for each dirty chunk
            let gs = self.config.chunk_size;
            let dirty_bounds: Vec<_> = result.dirty_chunks.iter()
                .filter(|key| self.density_fields.contains_key(key))
                .map(|&key| (key, 0, 0, 0, gs, gs, gs))
                .collect();
            self.remesh_dirty_parallel(&dirty_bounds);
        }

        let mesh_json = self.to_json_mesh();
        (result, mesh_json)
    }

    /// Mine a sphere: set solid voxels within radius to Air, return what was mined.
    pub fn mine_sphere(&mut self, center: Vec3, radius: f32) -> MineResult {
        let mut mined = HashMap::new();
        let cs = self.config.chunk_size as f32;
        let r2 = radius * radius;

        // Find affected chunks
        let min_cx = ((center.x - radius) / cs).floor() as i32;
        let max_cx = ((center.x + radius) / cs).floor() as i32;
        let min_cy = ((center.y - radius) / cs).floor() as i32;
        let max_cy = ((center.y + radius) / cs).floor() as i32;
        let min_cz = ((center.z - radius) / cs).floor() as i32;
        let max_cz = ((center.z + radius) / cs).floor() as i32;

        // Track dirty chunks with their local dirty bounds for incremental hermite patching
        let mut dirty_chunks: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> = Vec::new();

        for cz in min_cz..=max_cz {
            for cy in min_cy..=max_cy {
                for cx in min_cx..=max_cx {
                    if let Some(density) = self.density_fields.get_mut(&(cx, cy, cz)) {
                        let origin = Vec3::new(cx as f32 * cs, cy as f32 * cs, cz as f32 * cs);
                        let mut changed = false;

                        // Bounded iteration: only check voxels within the mine radius
                        let local_center = center - origin;
                        let lo_x = ((local_center.x - radius).floor() as i32).max(0) as usize;
                        let hi_x = ((local_center.x + radius).ceil() as usize + 1).min(density.size);
                        let lo_y = ((local_center.y - radius).floor() as i32).max(0) as usize;
                        let hi_y = ((local_center.y + radius).ceil() as usize + 1).min(density.size);
                        let lo_z = ((local_center.z - radius).floor() as i32).max(0) as usize;
                        let hi_z = ((local_center.z + radius).ceil() as usize + 1).min(density.size);

                        for z in lo_z..hi_z {
                            for y in lo_y..hi_y {
                                for x in lo_x..hi_x {
                                    let world_pos = origin
                                        + Vec3::new(x as f32, y as f32, z as f32);
                                    let dist2 = (world_pos - center).length_squared();
                                    if dist2 <= r2 {
                                        let sample = density.get_mut(x, y, z);
                                        if sample.material.is_solid() {
                                            *mined.entry(sample.material).or_insert(0) += 1;
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
                            // Expand dirty bounds by 1 voxel for edge adjacency, clamp to grid
                            let d_min_x = lo_x.saturating_sub(1);
                            let d_min_y = lo_y.saturating_sub(1);
                            let d_min_z = lo_z.saturating_sub(1);
                            let d_max_x = hi_x.min(density.size - 1);
                            let d_max_y = hi_y.min(density.size - 1);
                            let d_max_z = hi_z.min(density.size - 1);
                            dirty_chunks.push(((cx, cy, cz), d_min_x, d_min_y, d_min_z, d_max_x, d_max_y, d_max_z));
                        }
                    }
                }
            }
        }

        // Re-mesh dirty chunks in parallel using incremental hermite patching
        self.remesh_dirty_parallel(&dirty_chunks);

        MineResult {
            mined_materials: mined,
        }
    }

    /// Mine by peeling: only remove voxels near the surface (within small radius, near air).
    pub fn mine_peel(&mut self, center: Vec3, normal: Vec3, radius: f32) -> MineResult {
        let mut mined = HashMap::new();
        let cs = self.config.chunk_size as f32;
        let r2 = radius * radius;
        // Offset the center slightly along the normal into the surface
        let adjusted_center = center - normal * 0.5;

        let min_cx = ((adjusted_center.x - radius) / cs).floor() as i32;
        let max_cx = ((adjusted_center.x + radius) / cs).floor() as i32;
        let min_cy = ((adjusted_center.y - radius) / cs).floor() as i32;
        let max_cy = ((adjusted_center.y + radius) / cs).floor() as i32;
        let min_cz = ((adjusted_center.z - radius) / cs).floor() as i32;
        let max_cz = ((adjusted_center.z + radius) / cs).floor() as i32;

        let mut dirty_chunks: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> = Vec::new();

        for cz in min_cz..=max_cz {
            for cy in min_cy..=max_cy {
                for cx in min_cx..=max_cx {
                    if let Some(density) = self.density_fields.get_mut(&(cx, cy, cz)) {
                        let origin = Vec3::new(cx as f32 * cs, cy as f32 * cs, cz as f32 * cs);
                        let mut changed = false;

                        // Bounded iteration for peel
                        let local_center = adjusted_center - origin;
                        let lo_x = ((local_center.x - radius).floor() as i32).max(0) as usize;
                        let hi_x = ((local_center.x + radius).ceil() as usize + 1).min(density.size);
                        let lo_y = ((local_center.y - radius).floor() as i32).max(0) as usize;
                        let hi_y = ((local_center.y + radius).ceil() as usize + 1).min(density.size);
                        let lo_z = ((local_center.z - radius).floor() as i32).max(0) as usize;
                        let hi_z = ((local_center.z + radius).ceil() as usize + 1).min(density.size);

                        // First pass: collect voxels to peel
                        let mut to_peel: Vec<(usize, usize, usize)> = Vec::new();
                        for z in lo_z..hi_z {
                            for y in lo_y..hi_y {
                                for x in lo_x..hi_x {
                                    let world_pos = origin
                                        + Vec3::new(x as f32, y as f32, z as f32);
                                    let dist2 =
                                        (world_pos - adjusted_center).length_squared();
                                    if dist2 > r2 {
                                        continue;
                                    }

                                    let sample = density.get(x, y, z);
                                    if !sample.material.is_solid() {
                                        continue;
                                    }

                                    let near_surface = sample.density < 0.5
                                        || has_air_neighbor(density, x, y, z);
                                    if near_surface {
                                        to_peel.push((x, y, z));
                                    }
                                }
                            }
                        }

                        // Second pass: apply peeling with SDF gradient
                        for (x, y, z) in to_peel {
                            let world_pos = origin + Vec3::new(x as f32, y as f32, z as f32);
                            let dist = (world_pos - adjusted_center).length();
                            let sdf = dist - radius;
                            let sample = density.get_mut(x, y, z);
                            *mined.entry(sample.material).or_insert(0) += 1;
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
                            dirty_chunks.push(((cx, cy, cz), d_min_x, d_min_y, d_min_z, d_max_x, d_max_y, d_max_z));
                        }
                    }
                }
            }
        }

        self.remesh_dirty_parallel(&dirty_chunks);

        MineResult {
            mined_materials: mined,
        }
    }

    /// Re-extract hermite data and re-mesh a single chunk (full extraction fallback).
    fn remesh_chunk(&mut self, cx: i32, cy: i32, cz: i32) {
        if let Some(density) = self.density_fields.get(&(cx, cy, cz)) {
            let gs = self.config.chunk_size;
            let coord = ChunkCoord::new(cx, cy, cz);
            let hermite = extract_hermite_data(density);
            let cell_size = density.size - 1;
            let dc_vertices = solve_dc_vertices(&hermite, cell_size);
            let mut mesh = generate_mesh(&hermite, &dc_vertices, cell_size, self.config.max_edge_length, 0.0);

            let boundary_edges = region_gen::extract_boundary_edges(&hermite, gs);

            let world_origin = coord.world_origin_sized(gs);
            for v in &mut mesh.vertices {
                v.position += world_origin;
            }

            self.chunk_meshes.insert((cx, cy, cz), mesh);
            self.hermite_data.insert((cx, cy, cz), hermite);
            self.chunk_seam_data.insert((cx, cy, cz), ChunkSeamData {
                dc_vertices,
                world_origin,
                boundary_edges,
            });
        }
    }

    /// Remesh multiple dirty chunks in parallel, then rebuild combined mesh.
    /// Each chunk's hermite data is incrementally patched in its dirty region.
    fn remesh_dirty_parallel(
        &mut self,
        dirty_chunks: &[((i32, i32, i32), usize, usize, usize, usize, usize, usize)],
    ) {
        if dirty_chunks.is_empty() {
            return;
        }

        let gs = self.config.chunk_size;
        let max_edge_length = self.config.max_edge_length;

        // Extract data for dirty chunks so we can process in parallel
        let mut work: Vec<(
            (i32, i32, i32),
            DensityField,
            HermiteData,
            usize, usize, usize, usize, usize, usize,
        )> = Vec::with_capacity(dirty_chunks.len());

        for &(key, min_x, min_y, min_z, max_x, max_y, max_z) in dirty_chunks {
            let density = match self.density_fields.remove(&key) {
                Some(d) => d,
                None => continue,
            };
            let hermite = match self.hermite_data.remove(&key) {
                Some(h) => h,
                None => {
                    // No cached hermite — do full extraction
                    let h = extract_hermite_data(&density);
                    self.density_fields.insert(key, density);
                    self.hermite_data.insert(key, h);
                    // Fall back to full remesh for this chunk
                    self.remesh_chunk(key.0, key.1, key.2);
                    continue;
                }
            };
            work.push((key, density, hermite, min_x, min_y, min_z, max_x, max_y, max_z));
        }

        // Process all dirty chunks in parallel
        let results: Vec<_> = work
            .into_par_iter()
            .map(|(key, density, mut hermite, min_x, min_y, min_z, max_x, max_y, max_z)| {
                let coord = ChunkCoord::new(key.0, key.1, key.2);

                // Incremental hermite patch
                patch_hermite_data(&mut hermite, &density, min_x, min_y, min_z, max_x, max_y, max_z);

                let cell_size = density.size - 1;
                let dc_vertices = solve_dc_vertices(&hermite, cell_size);
                let mut mesh = generate_mesh(&hermite, &dc_vertices, cell_size, max_edge_length, 0.0);

                let boundary_edges = region_gen::extract_boundary_edges(&hermite, gs);

                let world_origin = coord.world_origin_sized(gs);
                for v in &mut mesh.vertices {
                    v.position += world_origin;
                }

                let seam_data = ChunkSeamData {
                    dc_vertices,
                    world_origin,
                    boundary_edges,
                };

                (key, density, hermite, mesh, seam_data)
            })
            .collect();

        // Write results back + collect dirty keys
        let mut dirty_keys = Vec::with_capacity(results.len());
        for (key, density, hermite, mesh, seam_data) in results {
            self.density_fields.insert(key, density);
            self.hermite_data.insert(key, hermite);
            self.chunk_meshes.insert(key, mesh);
            self.chunk_seam_data.insert(key, seam_data);
            dirty_keys.push(key);
        }

        self.rebuild_dirty(&dirty_keys);
    }

    /// Incremental rebuild: only regenerate seam mesh for dirty chunks.
    /// No combined mesh rebuild needed — to_json_mesh reads directly from chunk meshes.
    fn rebuild_dirty(&mut self, dirty_chunks: &[(i32, i32, i32)]) {
        if dirty_chunks.is_empty() {
            return;
        }

        let gs = self.config.chunk_size;

        // Regenerate full seam mesh (seam gen is cheap compared to hermite/DC).
        self.seam_mesh = region_gen::generate_seam_mesh(&self.chunk_seam_data, gs);
    }

    /// Check pool containment after mining — remove drained pools.
    /// A pool is drained if any of its rim voxels have been mined to air.
    pub fn check_pool_containment(&mut self) {
        let cs = self.config.chunk_size as f32;
        self.pool_descriptors.retain(|pool| {
            // Find which chunk this pool belongs to
            let cx = (pool.world_x / cs).floor() as i32;
            let cy = (pool.world_y / cs).floor() as i32;
            let cz = (pool.world_z / cs).floor() as i32;
            if let Some(density) = self.density_fields.get(&(cx, cy, cz)) {
                let world_origin = Vec3::new(cx as f32 * cs, cy as f32 * cs, cz as f32 * cs);
                pools::is_pool_contained(density, pool, world_origin)
            } else {
                false // chunk not found, pool can't exist
            }
        });
    }

    /// Get the combined mesh as JSON (direct from chunk meshes + seam, no intermediate copy).
    pub fn to_json_mesh(&self) -> JsonMesh {
        let mut keys: Vec<_> = self.chunk_meshes.keys().collect();
        keys.sort();
        let mut meshes: Vec<&Mesh> = keys.iter().map(|k| &self.chunk_meshes[k]).collect();
        meshes.push(&self.seam_mesh);
        mesh_to_json_multi(&meshes)
    }
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
        if nx < s && ny < s && nz < s {
            if !density.get(nx, ny, nz).material.is_solid() {
                return true;
            }
        }
    }
    false
}
