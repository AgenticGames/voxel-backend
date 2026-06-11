//! Region-based density generation with global worm carving and seam stitching.
//!
//! Shared between voxel-viewer (batch) and voxel-ffi (streaming).
//! A "region" is a deterministic group of chunks that share global worm connections.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use glam::Vec3;
use rayon::prelude::*;
use voxel_core::chunk::ChunkCoord;
use voxel_core::hermite::{EdgeIntersection, EdgeKey, HermiteData};
use voxel_core::mesh::{Mesh, Triangle, Vertex};

use voxel_core::material::Material;
use voxel_core::octree::node::VoxelSample;

use crate::config::GenerationConfig;
use crate::density::{DensityField, generate_density_field};
use crate::pools::{FluidSeed, PoolDescriptor};
use crate::springs::SpringDescriptor;
use crate::worm;
use crate::worm::path::WormSegment;

/// Timing breakdown of region generation phases.
#[derive(Debug, Clone, Default)]
pub struct RegionTimings {
    pub base_density: Duration,
    pub cavern_centers: Duration,
    pub worm_planning: Duration,
    pub worm_carving: Duration,
    pub pools: Duration,
    pub zones: Duration,
    pub formations: Duration,
    pub boundary_sync: Duration,
    pub metadata: Duration,
    pub worm_count: u32,
    pub worm_segment_count: u32,
}

/// Compute the deterministic region key for a chunk coordinate.
pub fn region_key(cx: i32, cy: i32, cz: i32, region_size: i32) -> (i32, i32, i32) {
    (
        cx.div_euclid(region_size),
        cy.div_euclid(region_size),
        cz.div_euclid(region_size),
    )
}

/// Get all chunk coordinates belonging to a region.
pub fn region_chunks(region: (i32, i32, i32), region_size: i32) -> Vec<(i32, i32, i32)> {
    let base_x = region.0 * region_size;
    let base_y = region.1 * region_size;
    let base_z = region.2 * region_size;
    let mut coords = Vec::with_capacity((region_size * region_size * region_size) as usize);
    for dz in 0..region_size {
        for dy in 0..region_size {
            for dx in 0..region_size {
                coords.push((base_x + dx, base_y + dy, base_z + dz));
            }
        }
    }
    coords
}

/// Generate density fields for a set of chunks with global worm carving.
///
/// This is the shared multi-chunk pipeline used by both the web viewer and FFI engine.
/// Worms are planned globally across all provided chunks, then carved into every
/// chunk they overlap — producing cross-chunk tunnels.
///
/// Phases:
/// 1. Generate base density fields (parallel via rayon)
/// 2. Collect cavern centers from ALL chunks
/// 3. Plan global worm connections (deterministic seed)
/// 4. Carve worms across all overlapping chunks
pub fn generate_region_densities(
    coords: &[(i32, i32, i32)],
    config: &GenerationConfig,
) -> (HashMap<(i32, i32, i32), DensityField>, Vec<PoolDescriptor>, Vec<FluidSeed>, Vec<Vec<WormSegment>>, RegionTimings, Vec<((i32, i32, i32), SpringDescriptor)>, Vec<crate::zones::ZoneDescriptor>) {
    let eb = config.effective_bounds();
    let chunk_size_f = eb;
    let mut timings = RegionTimings::default();

    // Phase 1: Generate base density fields (parallel via rayon).
    // Round 6 Fix B experiment: tried going serial here to avoid rayon pool
    // contention across 8 FFI workers hitting this simultaneously. Reverted —
    // measured +58% regression on initial_load wall time. par_iter's speedup
    // (even with contention) outweighs the serial overhead. Kept as-is.
    let t0 = Instant::now();
    let coarse_skipped = std::sync::atomic::AtomicU32::new(0);
    let slowest_chunk_ms = std::sync::atomic::AtomicU64::new(0);
    let mut density_fields: HashMap<(i32, i32, i32), DensityField> = coords
        .par_iter()
        .map(|&(cx, cy, cz)| {
            let chunk_t0 = Instant::now();
            let coord = ChunkCoord::new(cx, cy, cz);
            let origin = coord.world_origin_bounds(eb);
            let density = match crate::density::try_coarse_solid_check(config, origin) {
                Some(solid_field) => {
                    coarse_skipped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    solid_field
                }
                None => generate_density_field(config, origin),
            };
            let ms = (chunk_t0.elapsed().as_secs_f64() * 1000.0) as u64;
            slowest_chunk_ms.fetch_max(ms, std::sync::atomic::Ordering::Relaxed);
            ((cx, cy, cz), density)
        })
        .collect();
    timings.base_density = t0.elapsed();

    // Log base density breakdown
    {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true)
            .open("D:/Unreal Projects/Mithril2026/Saved/density_perf.txt")
        {
            let _ = writeln!(f, "base_density: {:.1}ms for {} chunks | coarse_skipped={} slowest_chunk={}ms",
                timings.base_density.as_secs_f64() * 1000.0,
                coords.len(),
                coarse_skipped.load(std::sync::atomic::Ordering::Relaxed),
                slowest_chunk_ms.load(std::sync::atomic::Ordering::Relaxed),
            );
        }
    }

    // Blank-canvas: skip all decoration phases. Each density field is already
    // a uniform host-rock block. Run boundary sync + metadata only.
    if config.blank_canvas {
        let t_bsync = Instant::now();
        sync_region_boundary_densities(&mut density_fields, config.chunk_size);
        timings.boundary_sync = t_bsync.elapsed();

        let t6 = Instant::now();
        for density in density_fields.values_mut() {
            density.compute_metadata();
        }
        timings.metadata = t6.elapsed();

        return (
            density_fields,
            Vec::new(),       // pool descriptors
            Vec::new(),       // fluid seeds
            Vec::new(),       // worm paths
            timings,
            Vec::new(),       // river springs
            Vec::new(),       // zone descriptors
        );
    }

    // Phase 2: Collect cavern centers from ALL chunks
    let t1 = Instant::now();
    let all_cavern_centers: Vec<Vec3> = coords
        .par_iter()
        .flat_map(|&(cx, cy, cz)| {
            let density = &density_fields[&(cx, cy, cz)];
            let coord = ChunkCoord::new(cx, cy, cz);
            let flat = density.densities();
            worm::connect::find_cavern_centers(&flat, density.size, coord.world_origin_bounds(eb))
        })
        .collect();
    timings.cavern_centers = t1.elapsed();

    // Phase 3: Plan global worm connections with deterministic region seed
    let t2 = Instant::now();
    let num_chunks = coords.len() as u32;
    let global_worm_seed = region_worm_seed(config.seed, coords);
    let total_worms = (config.worm.worms_per_region * num_chunks as f32).ceil() as u32;
    let connections =
        worm::connect::plan_worm_connections(global_worm_seed, &all_cavern_centers, total_worms);
    timings.worm_planning = t2.elapsed();

    // Phase 4: Generate worm paths and carve across all chunks they touch
    let t3 = Instant::now();
    let mut all_worm_paths: Vec<Vec<WormSegment>> = Vec::new();
    for (i, (worm_start, worm_end)) in connections.iter().enumerate() {
        let worm_seed = global_worm_seed
            .wrapping_add(0x1000)
            .wrapping_add(i as u64 * 1000);
        let segments = worm::path::generate_worm_path(
            worm_seed,
            *worm_start,
            *worm_end,
            config.worm.step_length,
            config.worm.max_steps,
            config.worm.radius_min,
            config.worm.radius_max,
        );
        if segments.is_empty() {
            continue;
        }

        // Validate: path endpoint must land within 50 voxels (2000 UU) of target cavern
        if let Some(last) = segments.last() {
            if last.position.distance(*worm_end) > 50.0 {
                continue;
            }
        }

        // Find bounding box of worm path to limit chunk iteration
        let (path_min, path_max) = worm_path_aabb(&segments);

        let min_cx = (path_min.x / chunk_size_f).floor() as i32;
        let max_cx = (path_max.x / chunk_size_f).floor() as i32;
        let min_cy = (path_min.y / chunk_size_f).floor() as i32;
        let max_cy = (path_max.y / chunk_size_f).floor() as i32;
        let min_cz = (path_min.z / chunk_size_f).floor() as i32;
        let max_cz = (path_max.z / chunk_size_f).floor() as i32;

        for cz in min_cz..=max_cz {
            for cy in min_cy..=max_cy {
                for cx in min_cx..=max_cx {
                    if let Some(density) = density_fields.get_mut(&(cx, cy, cz)) {
                        let coord = ChunkCoord::new(cx, cy, cz);
                        worm::carve::carve_worm_into_density(
                            density,
                            &segments,
                            coord.world_origin_bounds(eb),
                            config.worm.falloff_power,
                        );
                    }
                }
            }
        }

        // Carve junction spheres at start and end to guarantee openings
        let junction_radius = config.worm.radius_max * 1.5;
        for &junction_center in &[*worm_start, *worm_end] {
            let j_min_cx = ((junction_center.x - junction_radius) / chunk_size_f).floor() as i32;
            let j_max_cx = ((junction_center.x + junction_radius) / chunk_size_f).floor() as i32;
            let j_min_cy = ((junction_center.y - junction_radius) / chunk_size_f).floor() as i32;
            let j_max_cy = ((junction_center.y + junction_radius) / chunk_size_f).floor() as i32;
            let j_min_cz = ((junction_center.z - junction_radius) / chunk_size_f).floor() as i32;
            let j_max_cz = ((junction_center.z + junction_radius) / chunk_size_f).floor() as i32;
            for jcz in j_min_cz..=j_max_cz {
                for jcy in j_min_cy..=j_max_cy {
                    for jcx in j_min_cx..=j_max_cx {
                        if let Some(density) = density_fields.get_mut(&(jcx, jcy, jcz)) {
                            let coord = ChunkCoord::new(jcx, jcy, jcz);
                            worm::carve::carve_junction_sphere(
                                density,
                                junction_center,
                                junction_radius,
                                coord.world_origin_bounds(eb),
                                config.worm.falloff_power,
                            );
                        }
                    }
                }
            }
        }

        timings.worm_segment_count += segments.len() as u32;
        all_worm_paths.push(segments);
    }
    timings.worm_carving = t3.elapsed();
    timings.worm_count = all_worm_paths.len() as u32;

    // Phase 4b: Carve lava tubes + underground rivers per chunk
    let mut all_river_springs: Vec<((i32, i32, i32), SpringDescriptor)> = Vec::new();
    {
        let mut sorted_keys: Vec<_> = density_fields.keys().copied().collect();
        sorted_keys.sort();
        for &(cx, cy, cz) in &sorted_keys {
            let coord = ChunkCoord::new(cx, cy, cz);
            let c_seed = crate::seed::chunk_seed(config.seed, coord);
            let origin = coord.world_origin_bounds(eb);
            let wo = (origin.x as f64, origin.y as f64, origin.z as f64);
            if let Some(density) = density_fields.get_mut(&(cx, cy, cz)) {
                crate::lava_tubes::carve_lava_tubes(
                    density, &config.lava_tubes, wo, config.seed, c_seed,
                );
                let river_springs = crate::rivers::carve_rivers(
                    density, &config.rivers, &config.water_table, wo, config.seed, c_seed,
                );
                for rs in river_springs {
                    all_river_springs.push(((cx, cy, cz), rs));
                }
            }
        }
    }

    // Phase 4z: Detect and place cavern zones
    let t_zones = Instant::now();
    let (zone_descriptors, zone_bounds, zone_fluid_seeds) = crate::zones::place_zones(
        &mut density_fields,
        &config.zones,
        config.seed,
        eb,
        &all_worm_paths,
    );
    let mut all_fluid_seeds: Vec<crate::pools::FluidSeed> = zone_fluid_seeds;
    timings.zones = t_zones.elapsed();

    // Phase 5: Place cave pools per chunk (sort keys for determinism, skip zone areas)
    let t4 = Instant::now();
    let mut all_pool_descriptors = Vec::new();
    if config.pools.enabled {
        let mut sorted_keys: Vec<_> = density_fields.keys().copied().collect();
        sorted_keys.sort();
        for &(cx, cy, cz) in &sorted_keys {
            let coord = ChunkCoord::new(cx, cy, cz);
            let c_seed = crate::seed::chunk_seed(config.seed, coord);
            if let Some(density) = density_fields.get_mut(&(cx, cy, cz)) {
                // Skip pools entirely if chunk center is inside a zone exclusion area
                let chunk_center = glam::Vec3::new(
                    cx as f32 * eb + eb * 0.5,
                    cy as f32 * eb + eb * 0.5,
                    cz as f32 * eb + eb * 0.5,
                );
                if !crate::zones::is_in_exclusion_zone(chunk_center, &zone_bounds) {
                    let (mut pools, mut seeds) = crate::pools::place_pools(
                        density,
                        &config.pools,
                        coord.world_origin_bounds(eb),
                        config.seed,
                        c_seed,
                        (cx, cy, cz),
                    );
                    all_pool_descriptors.append(&mut pools);
                    all_fluid_seeds.append(&mut seeds);
                }
            }
        }
    }
    timings.pools = t4.elapsed();

    // Phase 6: Place cave formations per chunk
    let t5 = Instant::now();
    if config.formations.enabled {
        let mut sorted_keys: Vec<_> = density_fields.keys().copied().collect();
        sorted_keys.sort();
        for &(cx, cy, cz) in &sorted_keys {
            let coord = ChunkCoord::new(cx, cy, cz);
            let c_seed = crate::seed::chunk_seed(config.seed, coord);
            if let Some(density) = density_fields.get_mut(&(cx, cy, cz)) {
                // Skip formations if chunk center is inside a zone exclusion area
                let chunk_center = glam::Vec3::new(
                    cx as f32 * eb + eb * 0.5,
                    cy as f32 * eb + eb * 0.5,
                    cz as f32 * eb + eb * 0.5,
                );
                if !crate::zones::is_in_exclusion_zone(chunk_center, &zone_bounds) {
                    let formation_seeds = crate::formations::place_formations(
                        density,
                        &config.formations,
                        coord.world_origin_bounds(eb),
                        config.seed,
                        c_seed,
                        (cx, cy, cz),
                    );
                    all_fluid_seeds.extend(formation_seeds);
                }
            }
        }
    }
    timings.formations = t5.elapsed();

    // Phase 6b: Apply ore protrusion (push ore surfaces outward with smooth falloff)
    if config.ore_protrusion > 0.0 {
        for density in density_fields.values_mut() {
            crate::density::apply_ore_protrusion(density, config.ore_protrusion);
        }
    }

    // Phase 6c: Sync boundary densities between adjacent chunks in the region
    let t_bsync = Instant::now();
    sync_region_boundary_densities(&mut density_fields, config.chunk_size);
    timings.boundary_sync = t_bsync.elapsed();

    // Phase 7: Compute cached metadata for all density fields (search optimization)
    let t6 = Instant::now();
    for density in density_fields.values_mut() {
        density.compute_metadata();
    }
    timings.metadata = t6.elapsed();

    (density_fields, all_pool_descriptors, all_fluid_seeds, all_worm_paths, timings, all_river_springs, zone_descriptors)
}

/// Compute a deterministic worm seed for a set of coordinates.
/// Uses the minimum coordinate corner to ensure the same region always gets the same seed.
fn region_worm_seed(base_seed: u64, coords: &[(i32, i32, i32)]) -> u64 {
    if coords.is_empty() {
        return base_seed;
    }
    let min_x = coords.iter().map(|c| c.0).min().unwrap();
    let min_y = coords.iter().map(|c| c.1).min().unwrap();
    let min_z = coords.iter().map(|c| c.2).min().unwrap();
    base_seed
        .wrapping_add(0xF1F0_CAFE)
        .wrapping_add((min_x as u64).wrapping_mul(0x9E3779B97F4A7C15))
        ^ (min_y as u64).wrapping_mul(0x517CC1B727220A95)
        ^ (min_z as u64).wrapping_mul(0x6C62272E07BB0142)
}

/// Compute the axis-aligned bounding box of a worm path, expanded by the max segment radius.
pub fn worm_path_aabb(segments: &[WormSegment]) -> (Vec3, Vec3) {
    let max_r = segments.iter().map(|s| s.radius).fold(0.0f32, f32::max);
    let mut path_min = segments[0].position;
    let mut path_max = segments[0].position;
    for seg in segments {
        path_min = path_min.min(seg.position);
        path_max = path_max.max(seg.position);
    }
    path_min -= Vec3::splat(max_r);
    path_max += Vec3::splat(max_r);
    (path_min, path_max)
}

/// Apply worm paths from external regions into density fields.
///
/// For each worm path, computes AABB, finds overlapping chunks in `density_fields`,
/// and carves worm segments into them. Used for cross-region worm sharing.
pub fn apply_external_worm_paths(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    worm_paths: &[Vec<WormSegment>],
    config: &GenerationConfig,
) {
    let eb = config.effective_bounds();
    for path in worm_paths {
        if path.is_empty() {
            continue;
        }
        let (path_min, path_max) = worm_path_aabb(path);

        let min_cx = (path_min.x / eb).floor() as i32;
        let max_cx = (path_max.x / eb).floor() as i32;
        let min_cy = (path_min.y / eb).floor() as i32;
        let max_cy = (path_max.y / eb).floor() as i32;
        let min_cz = (path_min.z / eb).floor() as i32;
        let max_cz = (path_max.z / eb).floor() as i32;

        for cz in min_cz..=max_cz {
            for cy in min_cy..=max_cy {
                for cx in min_cx..=max_cx {
                    if let Some(density) = density_fields.get_mut(&(cx, cy, cz)) {
                        let coord = ChunkCoord::new(cx, cy, cz);
                        worm::carve::carve_worm_into_density(
                            density,
                            path,
                            coord.world_origin_bounds(eb),
                            config.worm.falloff_power,
                        );
                    }
                }
            }
        }
    }
}

// ── Seam stitching ──────────────────────────────────────────────────────────

/// Per-chunk data needed for seam stitching between adjacent chunks.
#[derive(Clone)]
pub struct ChunkSeamData {
    pub dc_vertices: Vec<Vec3>,
    pub world_origin: Vec3,
    pub boundary_edges: Vec<(EdgeKey, EdgeIntersection)>,
}

/// Extract boundary edges from hermite data that need cross-chunk seam quads.
/// Collects edges on the positive faces (x=gs, y=gs, z=gs boundaries).
pub fn extract_boundary_edges(
    hermite: &HermiteData,
    gs: usize,
) -> Vec<(EdgeKey, EdgeIntersection)> {
    let mut edges = Vec::new();
    for (key, intersection) in hermite.edges.iter() {
        let x = key.x() as usize;
        let y = key.y() as usize;
        let z = key.z() as usize;
        let axis = key.axis();

        let is_boundary = match axis {
            0 => y == gs || z == gs,
            1 => x == gs || z == gs,
            2 => x == gs || y == gs,
            _ => false,
        };

        if is_boundary {
            edges.push((key, intersection.clone()));
        }
    }
    edges
}

/// Generate seam mesh by emitting quads for boundary edges using DC vertices
/// from adjacent chunks.
pub fn generate_seam_mesh(
    chunks: &HashMap<(i32, i32, i32), ChunkSeamData>,
    gs: usize,
) -> Mesh {
    let mut mesh = Mesh::new();
    let gs_i = gs as i32;

    for (&(cx, cy, cz), chunk) in chunks {
        for (edge_key, intersection) in &chunk.boundary_edges {
            let ex = edge_key.x() as i32;
            let ey = edge_key.y() as i32;
            let ez = edge_key.z() as i32;
            let axis = edge_key.axis() as usize;

            // Compute the 4 cell positions for this edge's quad
            let cells = match axis {
                0 => [
                    (ex, ey - 1, ez - 1),
                    (ex, ey, ez - 1),
                    (ex, ey, ez),
                    (ex, ey - 1, ez),
                ],
                1 => [
                    (ex - 1, ey, ez - 1),
                    (ex, ey, ez - 1),
                    (ex, ey, ez),
                    (ex - 1, ey, ez),
                ],
                2 => [
                    (ex - 1, ey - 1, ez),
                    (ex, ey - 1, ez),
                    (ex, ey, ez),
                    (ex - 1, ey, ez),
                ],
                _ => continue,
            };

            // Skip if any cell has negative coordinates
            if cells.iter().any(|&(x, y, z)| x < 0 || y < 0 || z < 0) {
                continue;
            }

            // Look up DC vertex for each cell from the appropriate chunk
            let mut positions = [Vec3::ZERO; 4];
            let mut valid = true;

            for (i, &(cell_x, cell_y, cell_z)) in cells.iter().enumerate() {
                let chunk_dx = if cell_x >= gs_i { 1 } else { 0 };
                let chunk_dy = if cell_y >= gs_i { 1 } else { 0 };
                let chunk_dz = if cell_z >= gs_i { 1 } else { 0 };

                let neighbor_key = (cx + chunk_dx, cy + chunk_dy, cz + chunk_dz);

                let lx = (cell_x - chunk_dx * gs_i) as usize;
                let ly = (cell_y - chunk_dy * gs_i) as usize;
                let lz = (cell_z - chunk_dz * gs_i) as usize;

                if let Some(neighbor) = chunks.get(&neighbor_key) {
                    let cell_idx = lz * gs * gs + ly * gs + lx;
                    if cell_idx >= neighbor.dc_vertices.len() {
                        valid = false;
                        break;
                    }
                    let pos = neighbor.dc_vertices[cell_idx];
                    if pos.x.is_nan() {
                        // Cell has no sign changes (entirely solid or air).
                        // Use cell center as fallback to cap the surface.
                        let fallback = Vec3::new(
                            lx as f32 + 0.5,
                            ly as f32 + 0.5,
                            lz as f32 + 0.5,
                        );
                        positions[i] = fallback + neighbor.world_origin;
                    } else {
                        positions[i] = pos + neighbor.world_origin;
                    }
                } else {
                    valid = false;
                    break;
                }
            }

            if !valid {
                continue;
            }

            // Emit the quad as 2 triangles
            let base = mesh.vertices.len() as u32;
            for pos in &positions {
                mesh.vertices.push(Vertex {
                    position: *pos,
                    normal: intersection.normal,
                    material: intersection.material,
                });
            }

            let axis_dir = match axis {
                0 => Vec3::X,
                1 => Vec3::Y,
                _ => Vec3::Z,
            };
            let normal_dot = intersection.normal.dot(axis_dir);

            let (tri_a, tri_b) = if normal_dot > 0.0 {
                ([base, base + 1, base + 2], [base, base + 2, base + 3])
            } else {
                ([base + 2, base + 1, base], [base + 3, base + 2, base])
            };

            if !is_degenerate(&mesh.vertices, tri_a) {
                mesh.triangles.push(Triangle { indices: tri_a });
            }
            if !is_degenerate(&mesh.vertices, tri_b) {
                mesh.triangles.push(Triangle { indices: tri_b });
            }
        }
    }

    mesh
}

/// Generate seam quads for a SINGLE chunk in that chunk's LOCAL coordinate space.
///
/// Used by the FFI streaming path where each chunk is a separate actor.
/// Only processes boundary edges owned by `chunk_key`. Vertices from neighbor
/// chunks are offset relative to the owning chunk (e.g., a neighbor at +X has
/// its DC vertices shifted by +gs along X).
/// Generic over the map's value type so callers can pass either owned
/// `ChunkSeamData` or `Arc<ChunkSeamData>` (the FFI store Arc-wraps entries so
/// seam passes can snapshot them and run quad-gen without holding the store
/// read lock).
pub fn generate_chunk_seam_quads<S: std::borrow::Borrow<ChunkSeamData>>(
    chunk_key: (i32, i32, i32),
    all_seam_data: &HashMap<(i32, i32, i32), S>,
    gs: usize,
) -> Mesh {
    let chunk_data: &ChunkSeamData = match all_seam_data.get(&chunk_key) {
        Some(data) => data.borrow(),
        None => return Mesh::new(),
    };

    let gs_i = gs as i32;
    let gs_f = gs as f32;
    let mut mesh = Mesh::new();

    // Hoist the per-cell neighbor lookup out of the hot edge loop.
    // Every quad cell resolves to `chunk_key + (dx, dy, dz)` where each delta is
    // provably 0 or 1: cell coords range over [0, gs] (edge coords are 0..=gs,
    // cells are coord or coord-1), so `cell >= gs` flips a delta to at most 1.
    // That means the whole loop references only the 8 chunks in this 2^3 block —
    // yet the original code hash-probed `all_seam_data.get(&neighbor_key)` once
    // per cell (4x per boundary edge). Resolve those 8 references ONCE here and
    // index them by `dx | dy<<1 | dz<<2` inside the loop. Same lookup semantics
    // (a missing chunk yields None -> the quad is dropped), zero hashing per edge.
    let neighbors: [Option<&ChunkSeamData>; 8] = std::array::from_fn(|i| {
        let dx = (i & 1) as i32;
        let dy = ((i >> 1) & 1) as i32;
        let dz = ((i >> 2) & 1) as i32;
        all_seam_data
            .get(&(chunk_key.0 + dx, chunk_key.1 + dy, chunk_key.2 + dz))
            .map(|s| s.borrow())
    });

    for (edge_key, intersection) in &chunk_data.boundary_edges {
        let ex = edge_key.x() as i32;
        let ey = edge_key.y() as i32;
        let ez = edge_key.z() as i32;
        let axis = edge_key.axis() as usize;

        let cells = match axis {
            0 => [
                (ex, ey - 1, ez - 1),
                (ex, ey, ez - 1),
                (ex, ey, ez),
                (ex, ey - 1, ez),
            ],
            1 => [
                (ex - 1, ey, ez - 1),
                (ex, ey, ez - 1),
                (ex, ey, ez),
                (ex - 1, ey, ez),
            ],
            2 => [
                (ex - 1, ey - 1, ez),
                (ex, ey - 1, ez),
                (ex, ey, ez),
                (ex - 1, ey, ez),
            ],
            _ => continue,
        };

        if cells.iter().any(|&(x, y, z)| x < 0 || y < 0 || z < 0) {
            continue;
        }

        let mut positions = [Vec3::ZERO; 4];
        let mut valid = true;

        for (i, &(cell_x, cell_y, cell_z)) in cells.iter().enumerate() {
            let chunk_dx = if cell_x >= gs_i { 1 } else { 0 };
            let chunk_dy = if cell_y >= gs_i { 1 } else { 0 };
            let chunk_dz = if cell_z >= gs_i { 1 } else { 0 };

            let lx = (cell_x - chunk_dx * gs_i) as usize;
            let ly = (cell_y - chunk_dy * gs_i) as usize;
            let lz = (cell_z - chunk_dz * gs_i) as usize;

            let neighbor_slot = (chunk_dx | (chunk_dy << 1) | (chunk_dz << 2)) as usize;
            if let Some(neighbor) = neighbors[neighbor_slot] {
                let cell_idx = lz * gs * gs + ly * gs + lx;
                if cell_idx >= neighbor.dc_vertices.len() {
                    valid = false;
                    break;
                }
                let pos = neighbor.dc_vertices[cell_idx];
                // Offset neighbor DC vertex into THIS chunk's local space
                let offset = Vec3::new(
                    chunk_dx as f32 * gs_f,
                    chunk_dy as f32 * gs_f,
                    chunk_dz as f32 * gs_f,
                );
                if pos.x.is_nan() {
                    // Cell has no sign changes (entirely solid or air).
                    // Use cell center as fallback to cap the surface at the
                    // chunk boundary instead of leaving a hole.
                    let fallback = Vec3::new(
                        lx as f32 + 0.5,
                        ly as f32 + 0.5,
                        lz as f32 + 0.5,
                    );
                    positions[i] = fallback + offset;
                } else {
                    positions[i] = pos + offset;
                }
            } else {
                valid = false;
                break;
            }
        }

        if !valid {
            continue;
        }

        let base = mesh.vertices.len() as u32;
        for pos in &positions {
            mesh.vertices.push(Vertex {
                position: *pos,
                normal: intersection.normal,
                material: intersection.material,
            });
        }

        let axis_dir = match axis {
            0 => Vec3::X,
            1 => Vec3::Y,
            _ => Vec3::Z,
        };
        let normal_dot = intersection.normal.dot(axis_dir);

        let (tri_a, tri_b) = if normal_dot > 0.0 {
            ([base, base + 1, base + 2], [base, base + 2, base + 3])
        } else {
            ([base + 2, base + 1, base], [base + 3, base + 2, base])
        };

        if !is_degenerate(&mesh.vertices, tri_a) {
            mesh.triangles.push(Triangle { indices: tri_a });
        }
        if !is_degenerate(&mesh.vertices, tri_b) {
            mesh.triangles.push(Triangle { indices: tri_b });
        }
    }

    mesh
}

fn is_degenerate(vertices: &[Vertex], indices: [u32; 3]) -> bool {
    let v0 = vertices[indices[0] as usize].position;
    let v1 = vertices[indices[1] as usize].position;
    let v2 = vertices[indices[2] as usize].position;
    (v1 - v0).cross(v2 - v0).length_squared() < 1e-10
}

// ── Intra-region boundary sync ──────────────────────────────────────────────

/// Sync boundary densities between adjacent chunks within a region.
/// Ensures overlapping boundary voxels have identical density/material,
/// preventing seam gaps from chunk-local modifications (formations, pools, ore protrusion).
pub fn sync_region_boundary_densities(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
) {
    let gs = chunk_size;
    let keys: Vec<_> = density_fields.keys().copied().collect();

    // key -> dense bucket index, built once (one hash per field). Lets every
    // boundary write be routed to a per-field bucket so the apply pass resolves
    // each field's HashMap slot EXACTLY ONCE, instead of one `get_mut` SipHash
    // per individual sample write. The old flat `updates` vec re-hashed an
    // edge/corner field's 12-byte key thousands of times per region (every pair
    // that touched it pushed its key, and the apply loop hashed each push).
    let key_index: HashMap<(i32, i32, i32), usize> =
        keys.iter().enumerate().map(|(i, &k)| (k, i)).collect();

    // 13 "forward" neighbor offsets (first nonzero component is positive).
    // Processing only forward directions avoids syncing each pair twice.
    let offsets: [(i32, i32, i32); 13] = [
        // Faces (3)
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        // Edges (6)
        (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1),
        // Corners (4)
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
    ];

    // Per-field write buckets. Pushes are appended in the SAME outer-loop
    // traversal order that previously filled the flat `updates` vec, so for any
    // single cell the sequence of writes — and thus the last-writer-wins result —
    // is byte-for-byte preserved (bucketing is a stable partition by target key).
    let mut buckets: Vec<Vec<(usize, usize, usize, f32, Material)>> =
        vec![Vec::new(); keys.len()];

    for (ki, &(cx, cy, cz)) in keys.iter().enumerate() {
        // `f_a` is invariant across all 13 offsets for this key — resolve it once
        // instead of re-hashing `(cx,cy,cz)` inside the boundary-pair loop.
        let f_a = &density_fields[&(cx, cy, cz)];
        for &(dx, dy, dz) in &offsets {
            let neighbor = (cx + dx, cy + dy, cz + dz);
            // One `.get` replaces the old `contains_key` + `[&neighbor]` (which
            // hashed the same 12-byte key twice); resolved once per neighbor here
            // rather than re-hashed per boundary pair below.
            let Some(f_b) = density_fields.get(&neighbor) else {
                continue;
            };
            // Neighbor's bucket index — one hash per (key, offset), not per pair.
            let ni = key_index[&neighbor];

            let x_pairs = axis_boundary_pairs(dx, gs);
            let y_pairs = axis_boundary_pairs(dy, gs);
            let z_pairs = axis_boundary_pairs(dz, gs);

            for &(az, bz) in &z_pairs {
                for &(ay, by) in &y_pairs {
                    for &(ax, bx) in &x_pairs {
                        let sample_a = f_a.get(ax, ay, az);
                        let sample_b = f_b.get(bx, by, bz);
                        let (d, m) = avg_boundary(sample_a, sample_b);
                        buckets[ki].push((ax, ay, az, d, m));
                        buckets[ni].push((bx, by, bz, d, m));
                    }
                }
            }
        }
    }

    for (ki, &key) in keys.iter().enumerate() {
        if buckets[ki].is_empty() {
            continue;
        }
        // Single HashMap probe per field; all of this field's writes follow.
        let field = density_fields
            .get_mut(&key)
            .expect("key originates from density_fields.keys()");
        for (x, y, z, d, m) in buckets[ki].drain(..) {
            let sample = field.get_mut(x, y, z);
            sample.density = d;
            sample.material = m;
        }
    }

    // ── Gradient-blend pass (systemic seam-cliff fix) ────────────────────
    // The min-rule above can leave a 1-cell cliff: interior cell at
    // (gs-1) keeps a hard +1.0 (from a formation/pool/zone/coarse-bypass
    // write) while the boundary cell (gs) gets sync'd to neighbor air
    // (~-0.1). DC's QEF then projects every vertex to the same boundary
    // plane → coplanar triangles → flat-wall artifact ("slate cube"
    // bug). Per-system fixes (smooth SDF in write_mega_column, etc.)
    // treat one symptom; this pass attacks the geometric cause for ALL
    // density writers — formations, pools, zones, springs, surface
    // formations, and any future writer.
    //
    // Heuristic: when the interior cell is strongly positive AND the
    // synced boundary is negative AND the delta is large, blend the
    // interior partway toward the boundary. The conditions together
    // identify "hard write next to natural air" — natural noise rarely
    // produces strongly-positive cells right next to negative ones,
    // because noise is smooth.
    let cliff_interior_min: f32 = 0.5;   // interior must be at least this solid
    let cliff_boundary_max: f32 = 0.0;   // synced boundary must be air
    let cliff_delta_min: f32 = 0.7;      // and delta this large
    let blend_factor: f32 = 0.5;          // pull interior 50% toward boundary
    let face_offsets: [(i32, i32, i32); 3] = [(1, 0, 0), (0, 1, 0), (0, 0, 1)];
    // Same per-field bucketing as the min-rule pass: each field's slot is probed
    // once at apply, and per-cell write order is preserved (stable partition).
    let mut grad_buckets: Vec<Vec<(usize, usize, usize, f32)>> =
        vec![Vec::new(); keys.len()];

    for (ki, &(cx, cy, cz)) in keys.iter().enumerate() {
        // `f_a` is invariant across the 3 face offsets; resolve once per key.
        let f_a = &density_fields[&(cx, cy, cz)];
        for &(dx, dy, dz) in &face_offsets {
            let neighbor = (cx + dx, cy + dy, cz + dz);
            // Resolve the neighbor field once per face instead of re-hashing the
            // key inside the (gs+1)^2 (u,v) scan below.
            let Some(f_b) = density_fields.get(&neighbor) else {
                continue;
            };
            let ni = key_index[&neighbor];
            for u in 0..=gs {
                for v in 0..=gs {
                    let (a_int_d, a_bnd_d, b_int_d, b_bnd_d, a_int_xyz, b_int_xyz) = if dx == 1 {
                        (
                            f_a.get(gs - 1, u, v).density,
                            f_a.get(gs, u, v).density,
                            f_b.get(1, u, v).density,
                            f_b.get(0, u, v).density,
                            (gs - 1, u, v),
                            (1, u, v),
                        )
                    } else if dy == 1 {
                        (
                            f_a.get(u, gs - 1, v).density,
                            f_a.get(u, gs, v).density,
                            f_b.get(u, 1, v).density,
                            f_b.get(u, 0, v).density,
                            (u, gs - 1, v),
                            (u, 1, v),
                        )
                    } else {
                        (
                            f_a.get(u, v, gs - 1).density,
                            f_a.get(u, v, gs).density,
                            f_b.get(u, v, 1).density,
                            f_b.get(u, v, 0).density,
                            (u, v, gs - 1),
                            (u, v, 1),
                        )
                    };

                    // a-side cliff: a's interior is hard-solid, a's boundary was sync'd to air
                    if a_int_d > cliff_interior_min
                        && a_bnd_d < cliff_boundary_max
                        && a_int_d - a_bnd_d > cliff_delta_min
                    {
                        let new_d = a_int_d + (a_bnd_d - a_int_d) * blend_factor;
                        grad_buckets[ki].push((a_int_xyz.0, a_int_xyz.1, a_int_xyz.2, new_d));
                    }
                    // b-side cliff: same check from b's perspective
                    if b_int_d > cliff_interior_min
                        && b_bnd_d < cliff_boundary_max
                        && b_int_d - b_bnd_d > cliff_delta_min
                    {
                        let new_d = b_int_d + (b_bnd_d - b_int_d) * blend_factor;
                        grad_buckets[ni].push((b_int_xyz.0, b_int_xyz.1, b_int_xyz.2, new_d));
                    }
                }
            }
        }
    }

    for (ki, &key) in keys.iter().enumerate() {
        if grad_buckets[ki].is_empty() {
            continue;
        }
        let field = density_fields
            .get_mut(&key)
            .expect("key originates from density_fields.keys()");
        for (x, y, z, d) in grad_buckets[ki].drain(..) {
            let sample = field.get_mut(x, y, z);
            sample.density = d;
            // material left alone — only soften the density gradient.
        }
    }
}

/// Brush entry point: run the cave-carving phases (worms ± lava tubes/rivers)
/// and optionally the decoration phases (pools + formations) on a list of
/// existing chunk density fields. Used by the creative-mode "Cavern Stamp"
/// brush.
///
/// Modifies only chunks whose key is in `coords`. Worm planning ranges over
/// caverns inside those chunks; carving stays inside the brush region.
///
/// `decorate`: run pools + formations after carving.
/// `fluids`:   run lava tubes + rivers (small specialized carves).
///
/// Worms carve additively (write negative density), so existing user edits
/// in the chunks survive — solid → cave where worms pass, cave stays cave.
/// Decoration phases write material into newly-created surfaces.
///
/// The brush seed (`seed`) drives all randomness — same seed + same input
/// density gives the same caves, so users can re-roll vibes deterministically.
pub fn carve_caverns_into_existing(
    coords: &[(i32, i32, i32)],
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    config: &GenerationConfig,
    seed: u64,
    decorate: bool,
    fluids: bool,
) {
    use std::collections::HashSet;
    let eb = config.effective_bounds();
    let chunk_size_f = eb;
    let coord_set: HashSet<(i32, i32, i32)> = coords.iter().copied().collect();

    // Phase 2: cavern centers from brush chunks only
    let mut all_cavern_centers: Vec<Vec3> = Vec::new();
    for &c in coords {
        if let Some(df) = density_fields.get(&c) {
            let coord = ChunkCoord::new(c.0, c.1, c.2);
            let flat = df.densities();
            all_cavern_centers.extend(worm::connect::find_cavern_centers(
                &flat,
                df.size,
                coord.world_origin_bounds(eb),
            ));
        }
    }

    // Phase 3: plan worms with brush seed
    let num_chunks = coords.len() as u32;
    let total_worms = (config.worm.worms_per_region * num_chunks as f32).ceil() as u32;
    let connections = if all_cavern_centers.len() >= 2 && total_worms > 0 {
        worm::connect::plan_worm_connections(seed, &all_cavern_centers, total_worms)
    } else {
        Vec::new()
    };

    // Phase 4: generate + carve worm paths (restricted to brush chunks)
    let max_endpoint_drift = 50.0;
    for (i, (worm_start, worm_end)) in connections.iter().enumerate() {
        let worm_seed = seed.wrapping_add(0x1000).wrapping_add(i as u64 * 1000);
        let segments = worm::path::generate_worm_path(
            worm_seed,
            *worm_start,
            *worm_end,
            config.worm.step_length,
            config.worm.max_steps,
            config.worm.radius_min,
            config.worm.radius_max,
        );
        if segments.is_empty() {
            continue;
        }
        if let Some(last) = segments.last() {
            if last.position.distance(*worm_end) > max_endpoint_drift {
                continue;
            }
        }
        let (path_min, path_max) = worm_path_aabb(&segments);
        let min_cx = (path_min.x / chunk_size_f).floor() as i32;
        let max_cx = (path_max.x / chunk_size_f).floor() as i32;
        let min_cy = (path_min.y / chunk_size_f).floor() as i32;
        let max_cy = (path_max.y / chunk_size_f).floor() as i32;
        let min_cz = (path_min.z / chunk_size_f).floor() as i32;
        let max_cz = (path_max.z / chunk_size_f).floor() as i32;
        for cz in min_cz..=max_cz {
            for cy in min_cy..=max_cy {
                for cx in min_cx..=max_cx {
                    let key = (cx, cy, cz);
                    if !coord_set.contains(&key) {
                        continue;
                    }
                    if let Some(density) = density_fields.get_mut(&key) {
                        let coord = ChunkCoord::new(cx, cy, cz);
                        worm::carve::carve_worm_into_density(
                            density,
                            &segments,
                            coord.world_origin_bounds(eb),
                            config.worm.falloff_power,
                        );
                    }
                }
            }
        }
        // Junction spheres at start/end (also brush-restricted)
        let junction_radius = config.worm.radius_max * 1.5;
        for &jc in &[*worm_start, *worm_end] {
            let j_min_cx = ((jc.x - junction_radius) / chunk_size_f).floor() as i32;
            let j_max_cx = ((jc.x + junction_radius) / chunk_size_f).floor() as i32;
            let j_min_cy = ((jc.y - junction_radius) / chunk_size_f).floor() as i32;
            let j_max_cy = ((jc.y + junction_radius) / chunk_size_f).floor() as i32;
            let j_min_cz = ((jc.z - junction_radius) / chunk_size_f).floor() as i32;
            let j_max_cz = ((jc.z + junction_radius) / chunk_size_f).floor() as i32;
            for jcz in j_min_cz..=j_max_cz {
                for jcy in j_min_cy..=j_max_cy {
                    for jcx in j_min_cx..=j_max_cx {
                        let key = (jcx, jcy, jcz);
                        if !coord_set.contains(&key) {
                            continue;
                        }
                        if let Some(density) = density_fields.get_mut(&key) {
                            let coord = ChunkCoord::new(jcx, jcy, jcz);
                            worm::carve::carve_junction_sphere(
                                density,
                                jc,
                                junction_radius,
                                coord.world_origin_bounds(eb),
                                config.worm.falloff_power,
                            );
                        }
                    }
                }
            }
        }
    }

    // Phase 4b: lava tubes + rivers (per chunk, optional)
    if fluids {
        for &c in coords {
            let coord = ChunkCoord::new(c.0, c.1, c.2);
            let c_seed = crate::seed::chunk_seed(seed, coord);
            let origin = coord.world_origin_bounds(eb);
            let wo = (origin.x as f64, origin.y as f64, origin.z as f64);
            if let Some(density) = density_fields.get_mut(&c) {
                let _ = crate::lava_tubes::carve_lava_tubes(
                    density,
                    &config.lava_tubes,
                    wo,
                    seed,
                    c_seed,
                );
                let _ = crate::rivers::carve_rivers(
                    density,
                    &config.rivers,
                    &config.water_table,
                    wo,
                    seed,
                    c_seed,
                );
            }
        }
    }

    // Phase 5 + 6: pools + formations (optional, decoration)
    if decorate {
        for &c in coords {
            let coord = ChunkCoord::new(c.0, c.1, c.2);
            let c_seed = crate::seed::chunk_seed(seed, coord);
            if let Some(density) = density_fields.get_mut(&c) {
                if config.pools.enabled {
                    let (_pools, _seeds) = crate::pools::place_pools(
                        density,
                        &config.pools,
                        coord.world_origin_bounds(eb),
                        seed,
                        c_seed,
                        c,
                    );
                }
                if config.formations.enabled {
                    let _formation_seeds = crate::formations::place_formations(
                        density,
                        &config.formations,
                        coord.world_origin_bounds(eb),
                        seed,
                        c_seed,
                        c,
                    );
                }
            }
        }
    }
}

/// For a neighbor offset component, return the boundary coordinate pairs (a, b).
fn axis_boundary_pairs(d: i32, gs: usize) -> Vec<(usize, usize)> {
    if d == 1 {
        vec![(gs, 0)]
    } else if d == -1 {
        vec![(0, gs)]
    } else {
        (0..=gs).map(|i| (i, i)).collect()
    }
}

/// Density uses min (carved side wins); material preserves solid when possible.
fn avg_boundary(a: &VoxelSample, b: &VoxelSample) -> (f32, Material) {
    let density = a.density.min(b.density);
    let material = if a.material.is_solid() && b.material.is_solid() {
        if a.density >= b.density { a.material } else { b.material }
    } else if a.material.is_solid() {
        a.material
    } else if b.material.is_solid() {
        b.material
    } else {
        Material::Air
    };
    if !material.is_solid() && density > 0.0 {
        (0.0, material)
    } else {
        (density, material)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GenerationConfig;

    #[test]
    fn region_key_deterministic() {
        assert_eq!(region_key(0, 0, 0, 3), (0, 0, 0));
        assert_eq!(region_key(2, 2, 2, 3), (0, 0, 0));
        assert_eq!(region_key(3, 0, 0, 3), (1, 0, 0));
        assert_eq!(region_key(-1, 0, 0, 3), (-1, 0, 0));
        assert_eq!(region_key(-3, 0, 0, 3), (-1, 0, 0));
    }

    #[test]
    fn region_chunks_correct_count() {
        let chunks = region_chunks((0, 0, 0), 3);
        assert_eq!(chunks.len(), 27);
        assert!(chunks.contains(&(0, 0, 0)));
        assert!(chunks.contains(&(2, 2, 2)));
        assert!(!chunks.contains(&(3, 0, 0)));
    }

    /// Inlined replica of the PRE-hoist `sync_region_boundary_densities` body
    /// (the per-cell `density_fields[&key]` / `density_fields[&neighbor]` lookup
    /// form), used to prove the hoisted version is bit-identical. Kept byte-for-byte
    /// equivalent to the shipped function except that every inner lookup re-hashes
    /// the key, as the original did.
    fn sync_region_boundary_densities_prehoist(
        density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
        chunk_size: usize,
    ) {
        let gs = chunk_size;
        let keys: Vec<_> = density_fields.keys().copied().collect();
        let mut updates: Vec<((i32, i32, i32), usize, usize, usize, f32, Material)> = Vec::new();
        let offsets: [(i32, i32, i32); 13] = [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1),
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        ];
        for &(cx, cy, cz) in &keys {
            for &(dx, dy, dz) in &offsets {
                let neighbor = (cx + dx, cy + dy, cz + dz);
                if !density_fields.contains_key(&neighbor) {
                    continue;
                }
                let x_pairs = axis_boundary_pairs(dx, gs);
                let y_pairs = axis_boundary_pairs(dy, gs);
                let z_pairs = axis_boundary_pairs(dz, gs);
                for &(az, bz) in &z_pairs {
                    for &(ay, by) in &y_pairs {
                        for &(ax, bx) in &x_pairs {
                            let sample_a = density_fields[&(cx, cy, cz)].get(ax, ay, az);
                            let sample_b = density_fields[&neighbor].get(bx, by, bz);
                            let (d, m) = avg_boundary(sample_a, sample_b);
                            updates.push(((cx, cy, cz), ax, ay, az, d, m));
                            updates.push((neighbor, bx, by, bz, d, m));
                        }
                    }
                }
            }
        }
        for (key, x, y, z, d, m) in updates {
            if let Some(field) = density_fields.get_mut(&key) {
                let sample = field.get_mut(x, y, z);
                sample.density = d;
                sample.material = m;
            }
        }
        let cliff_interior_min: f32 = 0.5;
        let cliff_boundary_max: f32 = 0.0;
        let cliff_delta_min: f32 = 0.7;
        let blend_factor: f32 = 0.5;
        let face_offsets: [(i32, i32, i32); 3] = [(1, 0, 0), (0, 1, 0), (0, 0, 1)];
        let mut grad_updates: Vec<((i32, i32, i32), usize, usize, usize, f32)> = Vec::new();
        for &(cx, cy, cz) in &keys {
            for &(dx, dy, dz) in &face_offsets {
                let neighbor = (cx + dx, cy + dy, cz + dz);
                if !density_fields.contains_key(&neighbor) {
                    continue;
                }
                for u in 0..=gs {
                    for v in 0..=gs {
                        let (a_int_d, a_bnd_d, b_int_d, b_bnd_d, a_int_xyz, b_int_xyz) = if dx == 1 {
                            let f_a = &density_fields[&(cx, cy, cz)];
                            let f_b = &density_fields[&neighbor];
                            (f_a.get(gs - 1, u, v).density, f_a.get(gs, u, v).density,
                             f_b.get(1, u, v).density, f_b.get(0, u, v).density,
                             (gs - 1, u, v), (1, u, v))
                        } else if dy == 1 {
                            let f_a = &density_fields[&(cx, cy, cz)];
                            let f_b = &density_fields[&neighbor];
                            (f_a.get(u, gs - 1, v).density, f_a.get(u, gs, v).density,
                             f_b.get(u, 1, v).density, f_b.get(u, 0, v).density,
                             (u, gs - 1, v), (u, 1, v))
                        } else {
                            let f_a = &density_fields[&(cx, cy, cz)];
                            let f_b = &density_fields[&neighbor];
                            (f_a.get(u, v, gs - 1).density, f_a.get(u, v, gs).density,
                             f_b.get(u, v, 1).density, f_b.get(u, v, 0).density,
                             (u, v, gs - 1), (u, v, 1))
                        };
                        if a_int_d > cliff_interior_min && a_bnd_d < cliff_boundary_max
                            && a_int_d - a_bnd_d > cliff_delta_min
                        {
                            let new_d = a_int_d + (a_bnd_d - a_int_d) * blend_factor;
                            grad_updates.push(((cx, cy, cz), a_int_xyz.0, a_int_xyz.1, a_int_xyz.2, new_d));
                        }
                        if b_int_d > cliff_interior_min && b_bnd_d < cliff_boundary_max
                            && b_int_d - b_bnd_d > cliff_delta_min
                        {
                            let new_d = b_int_d + (b_bnd_d - b_int_d) * blend_factor;
                            grad_updates.push((neighbor, b_int_xyz.0, b_int_xyz.1, b_int_xyz.2, new_d));
                        }
                    }
                }
            }
        }
        for (key, x, y, z, d) in grad_updates {
            if let Some(field) = density_fields.get_mut(&key) {
                field.get_mut(x, y, z).density = d;
            }
        }
    }

    /// The hoisted `sync_region_boundary_densities` must produce bit-identical
    /// output to the pre-hoist replica. `HashMap::clone` preserves iteration order
    /// within a process, so both clones below visit `keys()` in the same order —
    /// making the (intrinsically order-dependent, at shared edge/corner cells)
    /// output directly comparable.
    #[test]
    fn boundary_sync_hoist_is_bit_identical() {
        let cs = 4usize; // size = 5
        let mut region: HashMap<(i32, i32, i32), DensityField> = HashMap::new();
        // 2^3 block: interior cell shares boundary faces/edges/corners with all
        // forward neighbors, so edge/corner cells receive multiple updates.
        for cz in 0..2 {
            for cy in 0..2 {
                for cx in 0..2 {
                    let mut f = DensityField::new(cs + 1);
                    let ox = cx * cs as i32;
                    let oy = cy * cs as i32;
                    let oz = cz * cs as i32;
                    for z in 0..=cs {
                        for y in 0..=cs {
                            for x in 0..=cs {
                                // A bumpy field that straddles the cliff thresholds
                                // and varies per global coord (so neighbors differ).
                                let gx = ox + x as i32;
                                let gy = oy + y as i32;
                                let gz = oz + z as i32;
                                let d = (((gx * 7 + gy * 13 + gz * 5) % 11) as f32 / 5.0) - 1.0;
                                let s = f.get_mut(x, y, z);
                                s.density = d;
                                s.material = if d > 0.0 { Material::Limestone } else { Material::Air };
                            }
                        }
                    }
                    region.insert((cx, cy, cz), f);
                }
            }
        }

        let mut a = region.clone();
        let mut b = region.clone();
        sync_region_boundary_densities(&mut a, cs); // hoisted (shipped)
        sync_region_boundary_densities_prehoist(&mut b, cs); // original

        let mut keys: Vec<_> = a.keys().copied().collect();
        keys.sort();
        for k in keys {
            let fa = &a[&k];
            let fb = &b[&k];
            assert_eq!(fa.samples.len(), fb.samples.len());
            for (i, (sa, sb)) in fa.samples.iter().zip(fb.samples.iter()).enumerate() {
                assert_eq!(
                    sa.density.to_bits(),
                    sb.density.to_bits(),
                    "density mismatch at chunk {k:?} sample {i}"
                );
                assert_eq!(sa.material, sb.material, "material mismatch at chunk {k:?} sample {i}");
            }
        }
    }

    #[test]
    fn generate_region_produces_all_chunks() {
        let config = GenerationConfig {
            chunk_size: 16,
            ..GenerationConfig::default()
        };
        let coords = region_chunks((0, 0, 0), 2);
        let (densities, _pools, _seeds, _worms, _timings, _river_springs, _zones) = generate_region_densities(&coords, &config);
        assert_eq!(densities.len(), 8);
        for &c in &coords {
            assert!(densities.contains_key(&c));
        }
    }
}
