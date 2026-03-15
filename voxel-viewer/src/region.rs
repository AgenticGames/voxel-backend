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
use voxel_gen::density::{self as gen_density, DensityField};
use voxel_gen::hermite_extract::{extract_hermite_data, patch_hermite_data};
use voxel_gen::pools::{self, FluidSeed, PoolDescriptor, PoolFluid};
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
    pub fluid_seeds: Vec<FluidSeed>,
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
        let (mut density_fields, pool_descriptors, fluid_seeds, _worm_paths, _timings, _river_springs) = region_gen::generate_region_densities(&coords, &config);

        // Phase 5b: Ore detail supersampling — regenerate ore chunks at higher resolution
        let multiplier = config.ore_detail_multiplier.max(1).min(4) as usize;
        if multiplier > 1 {
            let ore_chunks: Vec<(i32, i32, i32)> = coords
                .par_iter()
                .filter(|&&key| {
                    density_fields.get(&key)
                        .map(|d| gen_density::has_exposed_ore(d))
                        .unwrap_or(false)
                })
                .copied()
                .collect();

            if !ore_chunks.is_empty() {
                let eb = config.effective_bounds();
                let hires_results: Vec<_> = ore_chunks
                    .par_iter()
                    .map(|&(cx, cy, cz)| {
                        let mut hires_config = config.clone();
                        hires_config.chunk_size = gs * multiplier;
                        hires_config.bounds_size = eb;
                        let coord = ChunkCoord::new(cx, cy, cz);
                        let origin = coord.world_origin_bounds(eb);
                        let density = gen_density::generate_density_field(&hires_config, origin);
                        ((cx, cy, cz), density)
                    })
                    .collect();
                for (key, density) in hires_results {
                    density_fields.insert(key, density);
                }
            }
        }

        // Phase 5c: Seal boundary faces
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
        // Each chunk meshes at its own resolution (base or high-res).
        // Seam data always uses base resolution for cross-chunk stitching.
        let eb = config.effective_bounds();
        let chunk_results: Vec<_> = coords
            .par_iter()
            .map(|&(cx, cy, cz)| {
                let density = &density_fields[&(cx, cy, cz)];
                let coord = ChunkCoord::new(cx, cy, cz);
                let is_hires = density.size > gs + 1;

                let hermite = extract_hermite_data(density);
                let cell_size = density.size - 1;
                let dc_vertices = solve_dc_vertices(&hermite, cell_size);
                let mut mesh = generate_mesh(&hermite, &dc_vertices, cell_size);

                // Scale vertices to world coordinates
                let world_origin = coord.world_origin_bounds(eb);
                let voxel_scale = eb / cell_size as f32;
                for v in &mut mesh.vertices {
                    v.position = v.position * voxel_scale + world_origin;
                }

                // Seam data: always at base resolution (dc_vertices in local grid coords)
                let (seam_hermite, seam_dc, seam_boundary) = if is_hires {
                    // Downsample high-res density to base resolution for seam data
                    let factor = cell_size / gs;
                    let base_density = density.downsample(factor);
                    let base_hermite = extract_hermite_data(&base_density);
                    let base_dc = solve_dc_vertices(&base_hermite, gs);
                    let base_boundary = region_gen::extract_boundary_edges(&base_hermite, gs);
                    (base_hermite, base_dc, base_boundary)
                } else {
                    let boundary_edges = region_gen::extract_boundary_edges(&hermite, gs);
                    (hermite, dc_vertices, boundary_edges)
                };

                let seam_data = ChunkSeamData {
                    dc_vertices: seam_dc,
                    world_origin,
                    boundary_edges: seam_boundary,
                };

                ((cx, cy, cz), seam_hermite, mesh, seam_data)
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
            fluid_seeds,
        }
    }

    /// Run a deep sleep cycle on this region.
    /// Returns (sleep_result, updated_mesh_json).
    pub fn apply_sleep(&mut self, sleep_config: &voxel_sleep::SleepConfig) -> (voxel_sleep::SleepResult, JsonMesh) {

        // Use center chunk as player position
        let player_chunk = (0, 0, 0);

        // Build FluidSnapshot from generation fluid seeds (lava pools, water springs)
        let mut fluid = voxel_fluid::FluidSnapshot::default();
        let cs = self.config.chunk_size;
        let mut lava_count = 0u32;
        let mut water_count = 0u32;
        for fs in &self.fluid_seeds {
            let cells = fluid.chunks.entry(fs.chunk).or_insert_with(|| {
                vec![voxel_fluid::cell::FluidCell {
                    level: 0.0,
                    fluid_type: voxel_fluid::cell::FluidType::Water,
                    is_source: false,
                    grace_ticks: 0,
                    stagnant_ticks: 0,
                }; cs * cs * cs]
            });
            let idx = fs.lz as usize * cs * cs + fs.ly as usize * cs + fs.lx as usize;
            if idx < cells.len() {
                let ft = match fs.fluid_type {
                    PoolFluid::Water => { water_count += 1; voxel_fluid::cell::FluidType::Water },
                    PoolFluid::Lava => { lava_count += 1; voxel_fluid::cell::FluidType::Lava },
                };
                cells[idx] = voxel_fluid::cell::FluidCell {
                    level: 1.0,
                    fluid_type: ft,
                    is_source: fs.is_source,
                    grace_ticks: 0,
                    stagnant_ticks: 0,
                };
            }
        }
        eprintln!("Sleep: Injected {} lava + {} water fluid seeds from generation", lava_count, water_count);

        // If no lava from generation, inject ONE continuous diagonal lava tube
        // using world coordinates to avoid per-chunk duplication
        if lava_count == 0 {
            let mut injected = 0u32;
            let mut lava_in_solid = 0u32;
            let mut lava_in_air = 0u32;
            let pipe_radius = 2i32;
            let world_max = (cs * 3) as i32; // 48 for 3x3x3 region

            // Walk world diagonal from (0,0,0) to (47,47,47), one step at a time
            for i in 0..world_max {
                // Center of pipe at world (i, i, i)
                for dz in -pipe_radius..=pipe_radius {
                    for dy in -pipe_radius..=pipe_radius {
                        for dx in -pipe_radius..=pipe_radius {
                            if dx * dx + dy * dy + dz * dz > pipe_radius * pipe_radius + 2 {
                                continue; // cylindrical
                            }
                            let wx = i + dx;
                            let wy = i + dy;
                            let wz = i + dz;
                            if wx < 0 || wx >= world_max || wy < 0 || wy >= world_max || wz < 0 || wz >= world_max {
                                continue;
                            }
                            // Convert world → chunk + local
                            let chunk_key = (wx / cs as i32, wy / cs as i32, wz / cs as i32);
                            let lx = (wx % cs as i32) as usize;
                            let ly = (wy % cs as i32) as usize;
                            let lz = (wz % cs as i32) as usize;

                            if !self.density_fields.contains_key(&chunk_key) {
                                continue;
                            }

                            let cells = fluid.chunks.entry(chunk_key).or_insert_with(|| {
                                vec![voxel_fluid::cell::FluidCell {
                                    level: 0.0,
                                    fluid_type: voxel_fluid::cell::FluidType::Water,
                                    is_source: false,
                                    grace_ticks: 0,
                                    stagnant_ticks: 0,
                                }; cs * cs * cs]
                            });

                            let idx = lz * cs * cs + ly * cs + lx;
                            if cells[idx].level < 0.5 {
                                cells[idx] = voxel_fluid::cell::FluidCell {
                                    level: 1.0,
                                    fluid_type: voxel_fluid::cell::FluidType::Lava,
                                    is_source: true,
                                    grace_ticks: 0,
                                    stagnant_ticks: 0,
                                };
                                injected += 1;

                                // Check what's at this position
                                if let Some(df) = self.density_fields.get(&chunk_key) {
                                    if df.get(lx, ly, lz).material.is_solid() { lava_in_solid += 1; } else { lava_in_air += 1; }
                                }
                            }
                        }
                    }
                }
            }
            eprintln!("Sleep: Injected {} lava cells as ONE diagonal pipe (world 0,0,0 → {},{},{})",
                injected, world_max - 1, world_max - 1, world_max - 1);
            eprintln!("Sleep: Lava positions: {} in solid rock, {} in air", lava_in_solid, lava_in_air);
        }
        let result = voxel_sleep::execute_sleep(
            sleep_config,
            &mut self.density_fields,
            &mut self.stress_fields,
            &mut self.support_fields,
            &mut fluid,
            player_chunk,
            1, // sleep_count
            None, // no progress channel
        );

        // Sync boundary density planes between dirty chunks and their face neighbors
        if !result.dirty_chunks.is_empty() {
            let cs = self.config.chunk_size;
            let offsets = [(1i32,0i32,0i32),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)];
            for &dk in &result.dirty_chunks {
                for &(ox, oy, oz) in &offsets {
                    let nk = (dk.0+ox, dk.1+oy, dk.2+oz);
                    if !self.density_fields.contains_key(&nk) { continue; }
                    let (axis, src_idx, dst_idx) = if ox == 1 { (0, cs, 0) }
                        else if ox == -1 { (0, 0, cs) }
                        else if oy == 1 { (1, cs, 0) }
                        else if oy == -1 { (1, 0, cs) }
                        else if oz == 1 { (2, cs, 0) }
                        else { (2, 0, cs) };
                    for a in 0..=cs {
                        for b in 0..=cs {
                            let (sx, sy, sz) = match axis {
                                0 => (src_idx, a, b), 1 => (a, src_idx, b), _ => (a, b, src_idx),
                            };
                            let (dx, dy, dz) = match axis {
                                0 => (dst_idx, a, b), 1 => (a, dst_idx, b), _ => (a, b, dst_idx),
                            };
                            let sample = self.density_fields[&dk].get(sx, sy, sz);
                            let mat = sample.material;
                            let dens = sample.density;
                            if let Some(ndf) = self.density_fields.get_mut(&nk) {
                                let dst = ndf.get_mut(dx, dy, dz);
                                dst.material = mat;
                                dst.density = dens;
                            }
                        }
                    }
                }
            }
        }

        // Re-mesh all dirty chunks
        if !result.dirty_chunks.is_empty() {
            // Build full-chunk dirty bounds for each dirty chunk
            let dirty_bounds: Vec<_> = result.dirty_chunks.iter()
                .filter(|key| self.density_fields.contains_key(key))
                .map(|&key| {
                    let cell_size = self.density_fields[&key].size - 1;
                    (key, 0, 0, 0, cell_size, cell_size, cell_size)
                })
                .collect();
            self.remesh_dirty_parallel(&dirty_bounds);
        }

        let mesh_json = self.to_json_mesh();
        (result, mesh_json)
    }

    /// Mine a sphere: set solid voxels within radius to Air, return what was mined.
    pub fn mine_sphere(&mut self, center: Vec3, radius: f32) -> MineResult {
        let mut mined = HashMap::new();
        let eb = self.config.effective_bounds();
        let r2 = radius * radius;

        // Find affected chunks
        let min_cx = ((center.x - radius) / eb).floor() as i32;
        let max_cx = ((center.x + radius) / eb).floor() as i32;
        let min_cy = ((center.y - radius) / eb).floor() as i32;
        let max_cy = ((center.y + radius) / eb).floor() as i32;
        let min_cz = ((center.z - radius) / eb).floor() as i32;
        let max_cz = ((center.z + radius) / eb).floor() as i32;

        // Track dirty chunks with their local dirty bounds for incremental hermite patching
        let mut dirty_chunks: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> = Vec::new();

        for cz in min_cz..=max_cz {
            for cy in min_cy..=max_cy {
                for cx in min_cx..=max_cx {
                    if let Some(density) = self.density_fields.get_mut(&(cx, cy, cz)) {
                        let origin = Vec3::new(cx as f32 * eb, cy as f32 * eb, cz as f32 * eb);
                        let vs = eb / (density.size - 1) as f32;
                        let mut changed = false;

                        // Bounded iteration: only check voxels within the mine radius
                        let local_center = center - origin;
                        let lo_x = ((local_center.x / vs - radius / vs).floor() as i32).max(0) as usize;
                        let hi_x = (((local_center.x + radius) / vs).ceil() as usize + 1).min(density.size);
                        let lo_y = ((local_center.y / vs - radius / vs).floor() as i32).max(0) as usize;
                        let hi_y = (((local_center.y + radius) / vs).ceil() as usize + 1).min(density.size);
                        let lo_z = ((local_center.z / vs - radius / vs).floor() as i32).max(0) as usize;
                        let hi_z = (((local_center.z + radius) / vs).ceil() as usize + 1).min(density.size);

                        for z in lo_z..hi_z {
                            for y in lo_y..hi_y {
                                for x in lo_x..hi_x {
                                    let world_pos = origin
                                        + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
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
        let eb = self.config.effective_bounds();
        let r2 = radius * radius;
        // Offset the center slightly along the normal into the surface
        let adjusted_center = center - normal * 0.5;

        let min_cx = ((adjusted_center.x - radius) / eb).floor() as i32;
        let max_cx = ((adjusted_center.x + radius) / eb).floor() as i32;
        let min_cy = ((adjusted_center.y - radius) / eb).floor() as i32;
        let max_cy = ((adjusted_center.y + radius) / eb).floor() as i32;
        let min_cz = ((adjusted_center.z - radius) / eb).floor() as i32;
        let max_cz = ((adjusted_center.z + radius) / eb).floor() as i32;

        let mut dirty_chunks: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> = Vec::new();

        for cz in min_cz..=max_cz {
            for cy in min_cy..=max_cy {
                for cx in min_cx..=max_cx {
                    if let Some(density) = self.density_fields.get_mut(&(cx, cy, cz)) {
                        let origin = Vec3::new(cx as f32 * eb, cy as f32 * eb, cz as f32 * eb);
                        let vs = eb / (density.size - 1) as f32;
                        let mut changed = false;

                        // Bounded iteration for peel
                        let local_center = adjusted_center - origin;
                        let lo_x = (((local_center.x - radius) / vs).floor() as i32).max(0) as usize;
                        let hi_x = (((local_center.x + radius) / vs).ceil() as usize + 1).min(density.size);
                        let lo_y = (((local_center.y - radius) / vs).floor() as i32).max(0) as usize;
                        let hi_y = (((local_center.y + radius) / vs).ceil() as usize + 1).min(density.size);
                        let lo_z = (((local_center.z - radius) / vs).floor() as i32).max(0) as usize;
                        let hi_z = (((local_center.z + radius) / vs).ceil() as usize + 1).min(density.size);

                        // First pass: collect voxels to peel
                        let mut to_peel: Vec<(usize, usize, usize)> = Vec::new();
                        for z in lo_z..hi_z {
                            for y in lo_y..hi_y {
                                for x in lo_x..hi_x {
                                    let world_pos = origin
                                        + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
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
                            let world_pos = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
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
            let eb = self.config.effective_bounds();
            let coord = ChunkCoord::new(cx, cy, cz);
            let hermite = extract_hermite_data(density);
            let cell_size = density.size - 1;
            let dc_vertices = solve_dc_vertices(&hermite, cell_size);
            let mut mesh = generate_mesh(&hermite, &dc_vertices, cell_size);

            let is_hires = density.size > gs + 1;

            let world_origin = coord.world_origin_bounds(eb);
            let voxel_scale = eb / cell_size as f32;
            for v in &mut mesh.vertices {
                v.position = v.position * voxel_scale + world_origin;
            }

            // Seam data always at base resolution
            let (seam_dc, seam_boundary) = if is_hires {
                let factor = cell_size / gs;
                let base_density = density.downsample(factor);
                let base_hermite = extract_hermite_data(&base_density);
                let base_dc = solve_dc_vertices(&base_hermite, gs);
                let base_boundary = region_gen::extract_boundary_edges(&base_hermite, gs);
                (base_dc, base_boundary)
            } else {
                let boundary_edges = region_gen::extract_boundary_edges(&hermite, gs);
                (dc_vertices, boundary_edges)
            };

            self.chunk_meshes.insert((cx, cy, cz), mesh);
            self.hermite_data.insert((cx, cy, cz), hermite);
            self.chunk_seam_data.insert((cx, cy, cz), ChunkSeamData {
                dc_vertices: seam_dc,
                world_origin,
                boundary_edges: seam_boundary,
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
        let eb = self.config.effective_bounds();
        let results: Vec<_> = work
            .into_par_iter()
            .map(|(key, density, mut hermite, min_x, min_y, min_z, max_x, max_y, max_z)| {
                let coord = ChunkCoord::new(key.0, key.1, key.2);

                // Incremental hermite patch
                patch_hermite_data(&mut hermite, &density, min_x, min_y, min_z, max_x, max_y, max_z);

                let cell_size = density.size - 1;
                let is_hires = density.size > gs + 1;
                let dc_vertices = solve_dc_vertices(&hermite, cell_size);
                let mut mesh = generate_mesh(&hermite, &dc_vertices, cell_size);

                let world_origin = coord.world_origin_bounds(eb);
                let voxel_scale = eb / cell_size as f32;
                for v in &mut mesh.vertices {
                    v.position = v.position * voxel_scale + world_origin;
                }

                // Seam data always at base resolution
                let (seam_dc, seam_boundary) = if is_hires {
                    let factor = cell_size / gs;
                    let base_density = density.downsample(factor);
                    let base_hermite = extract_hermite_data(&base_density);
                    let base_dc = solve_dc_vertices(&base_hermite, gs);
                    let base_boundary = region_gen::extract_boundary_edges(&base_hermite, gs);
                    (base_dc, base_boundary)
                } else {
                    let boundary_edges = region_gen::extract_boundary_edges(&hermite, gs);
                    (dc_vertices, boundary_edges)
                };

                let seam_data = ChunkSeamData {
                    dc_vertices: seam_dc,
                    world_origin,
                    boundary_edges: seam_boundary,
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
        let eb = self.config.effective_bounds();
        self.pool_descriptors.retain(|pool| {
            // Find which chunk this pool belongs to
            let cx = (pool.world_x / eb).floor() as i32;
            let cy = (pool.world_y / eb).floor() as i32;
            let cz = (pool.world_z / eb).floor() as i32;
            if let Some(density) = self.density_fields.get(&(cx, cy, cz)) {
                let world_origin = Vec3::new(cx as f32 * eb, cy as f32 * eb, cz as f32 * eb);
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
