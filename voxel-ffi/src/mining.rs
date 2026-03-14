use glam::Vec3;
use voxel_core::material::Material;
use voxel_gen::config::GenerationConfig;
use voxel_gen::density::DensityField;

use crate::store::{sync_boundary_density, ChunkStore};
use crate::types::{ConvertedMesh, FfiMinedMaterials};

/// Mine a sphere: set solid voxels within radius to Air.
/// Returns the re-meshed dirty chunks (in UE coords) and mined material counts.
pub fn mine_sphere(
    store: &mut ChunkStore,
    center: Vec3,
    radius: f32,
    config: &GenerationConfig,
    world_scale: f32,
) -> (Vec<((i32, i32, i32), ConvertedMesh)>, FfiMinedMaterials) {
    let eb = config.effective_bounds();
    let vs = config.voxel_scale();
    let r2 = radius * radius;
    let mut mined_counts = [0u32; 27];

    let min_cx = ((center.x - radius) / eb).floor() as i32;
    let max_cx = ((center.x + radius) / eb).floor() as i32;
    let min_cy = ((center.y - radius) / eb).floor() as i32;
    let max_cy = ((center.y + radius) / eb).floor() as i32;
    let min_cz = ((center.z - radius) / eb).floor() as i32;
    let max_cz = ((center.z + radius) / eb).floor() as i32;

    let mut dirty_chunks: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> =
        Vec::new();

    for cz in min_cz..=max_cz {
        for cy in min_cy..=max_cy {
            for cx in min_cx..=max_cx {
                if let Some(density) = store.density_fields.get_mut(&(cx, cy, cz)) {
                    let origin =
                        Vec3::new(cx as f32 * eb, cy as f32 * eb, cz as f32 * eb);
                    let mut changed = false;

                    // Convert world-space center to grid-space for this chunk
                    let grid_center = (center - origin) / vs;
                    let grid_radius = radius / vs;
                    let lo_x = ((grid_center.x - grid_radius).floor() as i32).max(0) as usize;
                    let hi_x =
                        ((grid_center.x + grid_radius).ceil() as usize + 1).min(density.size);
                    let lo_y = ((grid_center.y - grid_radius).floor() as i32).max(0) as usize;
                    let hi_y =
                        ((grid_center.y + grid_radius).ceil() as usize + 1).min(density.size);
                    let lo_z = ((grid_center.z - grid_radius).floor() as i32).max(0) as usize;
                    let hi_z =
                        ((grid_center.z + grid_radius).ceil() as usize + 1).min(density.size);

                    for z in lo_z..hi_z {
                        for y in lo_y..hi_y {
                            for x in lo_x..hi_x {
                                let world_pos =
                                    origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                let dist2 = (world_pos - center).length_squared();
                                if dist2 <= r2 {
                                    let sample = density.get_mut(x, y, z);
                                    if sample.material.is_solid() {
                                        mined_counts[sample.material as u8 as usize] += 1;
                                        // SDF: smooth gradient following sphere curvature
                                        // instead of flat -1.0 which kills DC normals
                                        let sdf = dist2.sqrt() - radius;
                                        sample.density = sdf.min(sample.density);
                                        if sample.density <= 0.0 {
                                            sample.material = Material::Air;
                                        }
                                        changed = true;
                                    }
                                }
                            }
                        }
                    }

                    if changed {
                        let expand = config.mine.dirty_expand as usize;
                        let d_min_x = lo_x.saturating_sub(expand);
                        let d_min_y = lo_y.saturating_sub(expand);
                        let d_min_z = lo_z.saturating_sub(expand);
                        let d_max_x = (hi_x + expand).min(density.size - 1);
                        let d_max_y = (hi_y + expand).min(density.size - 1);
                        let d_max_z = (hi_z + expand).min(density.size - 1);
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

    // Post-mine Laplacian density smoothing
    for &(key, min_x, min_y, min_z, max_x, max_y, max_z) in &dirty_chunks {
        if let Some(density) = store.density_fields.get_mut(&key) {
            smooth_mine_boundary(
                density,
                min_x, min_y, min_z, max_x, max_y, max_z,
                config.mine.smooth_iterations,
                config.mine.smooth_strength,
            );
        }
    }

    // Sync boundary density between dirty chunks and face neighbors
    let extra_dirty = sync_boundary_density(
        &mut store.density_fields, &dirty_chunks, config.chunk_size,
    );
    dirty_chunks.extend(extra_dirty);

    let meshes = store.remesh_dirty(&dirty_chunks, config, world_scale);
    (meshes, FfiMinedMaterials { counts: mined_counts })
}

/// Mine by peeling: only remove surface voxels within radius.
pub fn mine_peel(
    store: &mut ChunkStore,
    center: Vec3,
    normal: Vec3,
    radius: f32,
    config: &GenerationConfig,
    world_scale: f32,
) -> (Vec<((i32, i32, i32), ConvertedMesh)>, FfiMinedMaterials) {
    let eb = config.effective_bounds();
    let vs = config.voxel_scale();
    let r2 = radius * radius;
    let mut mined_counts = [0u32; 27];
    let adjusted_center = center - normal * 0.5;

    let min_cx = ((adjusted_center.x - radius) / eb).floor() as i32;
    let max_cx = ((adjusted_center.x + radius) / eb).floor() as i32;
    let min_cy = ((adjusted_center.y - radius) / eb).floor() as i32;
    let max_cy = ((adjusted_center.y + radius) / eb).floor() as i32;
    let min_cz = ((adjusted_center.z - radius) / eb).floor() as i32;
    let max_cz = ((adjusted_center.z + radius) / eb).floor() as i32;

    let mut dirty_chunks: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> =
        Vec::new();

    for cz in min_cz..=max_cz {
        for cy in min_cy..=max_cy {
            for cx in min_cx..=max_cx {
                if let Some(density) = store.density_fields.get_mut(&(cx, cy, cz)) {
                    let origin =
                        Vec3::new(cx as f32 * eb, cy as f32 * eb, cz as f32 * eb);
                    let mut changed = false;

                    let grid_center = (adjusted_center - origin) / vs;
                    let grid_radius = radius / vs;
                    let lo_x = ((grid_center.x - grid_radius).floor() as i32).max(0) as usize;
                    let hi_x =
                        ((grid_center.x + grid_radius).ceil() as usize + 1).min(density.size);
                    let lo_y = ((grid_center.y - grid_radius).floor() as i32).max(0) as usize;
                    let hi_y =
                        ((grid_center.y + grid_radius).ceil() as usize + 1).min(density.size);
                    let lo_z = ((grid_center.z - grid_radius).floor() as i32).max(0) as usize;
                    let hi_z =
                        ((grid_center.z + grid_radius).ceil() as usize + 1).min(density.size);

                    // First pass: collect voxels to peel
                    let mut to_peel: Vec<(usize, usize, usize)> = Vec::new();
                    for z in lo_z..hi_z {
                        for y in lo_y..hi_y {
                            for x in lo_x..hi_x {
                                let world_pos =
                                    origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
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
                            origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                        let dist = (world_pos - adjusted_center).length();
                        let sdf = dist - radius;
                        let sample = density.get_mut(x, y, z);
                        let original_material = sample.material;
                        mined_counts[sample.material as u8 as usize] += 1;
                        sample.density = sdf.min(sample.density);
                        if sample.density <= 0.0 {
                            sample.material = Material::Air;
                        } else {
                            sample.material = original_material;
                        }
                        changed = true;
                    }

                    if changed {
                        let expand = config.mine.dirty_expand as usize;
                        let d_min_x = lo_x.saturating_sub(expand);
                        let d_min_y = lo_y.saturating_sub(expand);
                        let d_min_z = lo_z.saturating_sub(expand);
                        let d_max_x = (hi_x + expand).min(density.size - 1);
                        let d_max_y = (hi_y + expand).min(density.size - 1);
                        let d_max_z = (hi_z + expand).min(density.size - 1);
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

    // Post-mine Laplacian density smoothing
    for &(key, min_x, min_y, min_z, max_x, max_y, max_z) in &dirty_chunks {
        if let Some(density) = store.density_fields.get_mut(&key) {
            smooth_mine_boundary(
                density,
                min_x, min_y, min_z, max_x, max_y, max_z,
                config.mine.smooth_iterations,
                config.mine.smooth_strength,
            );
        }
    }

    // Sync boundary density between dirty chunks and face neighbors
    let extra_dirty = sync_boundary_density(
        &mut store.density_fields, &dirty_chunks, config.chunk_size,
    );
    dirty_chunks.extend(extra_dirty);

    let meshes = store.remesh_dirty(&dirty_chunks, config, world_scale);
    (meshes, FfiMinedMaterials { counts: mined_counts })
}

/// Laplacian smoothing of density values near the mine boundary.
/// Only affects voxels near the air/solid interface within the expanded dirty region.
/// Uses double-buffering to avoid order-dependent results.
pub(crate) fn smooth_mine_boundary(
    density: &mut DensityField,
    min_x: usize, min_y: usize, min_z: usize,
    max_x: usize, max_y: usize, max_z: usize,
    iterations: u32,
    strength: f32,
) {
    if iterations == 0 || strength <= 0.0 {
        return;
    }
    let size = density.size;

    for _ in 0..iterations {
        // Collect smoothed values for surface voxels (double-buffer)
        let mut updates: Vec<(usize, usize, usize, f32)> = Vec::new();

        for z in min_z..=max_z.min(size - 1) {
            for y in min_y..=max_y.min(size - 1) {
                for x in min_x..=max_x.min(size - 1) {
                    // Only smooth near the surface: solid with air neighbor, or air with solid neighbor
                    let is_solid = density.get(x, y, z).material.is_solid();
                    let near_surface = if is_solid {
                        has_air_neighbor(density, x, y, z)
                    } else {
                        has_solid_neighbor(density, x, y, z)
                    };

                    if !near_surface {
                        continue;
                    }

                    // Average of 6 face neighbors (clamped to bounds)
                    let mut sum = 0.0f32;
                    let mut count = 0u32;
                    let neighbors: [(i32, i32, i32); 6] = [
                        (-1, 0, 0), (1, 0, 0),
                        (0, -1, 0), (0, 1, 0),
                        (0, 0, -1), (0, 0, 1),
                    ];
                    for (dx, dy, dz) in neighbors {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        let nz = z as i32 + dz;
                        if nx >= 0 && nx < size as i32 && ny >= 0 && ny < size as i32 && nz >= 0 && nz < size as i32 {
                            sum += density.get(nx as usize, ny as usize, nz as usize).density;
                            count += 1;
                        }
                    }

                    if count > 0 {
                        let avg = sum / count as f32;
                        let old = density.get(x, y, z).density;
                        let new_val = (1.0 - strength) * old + strength * avg;
                        updates.push((x, y, z, new_val));
                    }
                }
            }
        }

        // Apply all updates (density, with invariant enforcement)
        for (x, y, z, new_density) in updates {
            let sample = density.get_mut(x, y, z);
            sample.density = new_density;
            // Enforce invariant: Air material must have non-positive density
            if !sample.material.is_solid() && sample.density > 0.0 {
                sample.density = 0.0;
            }
        }
    }
}

fn has_solid_neighbor(density: &DensityField, x: usize, y: usize, z: usize) -> bool {
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
        if nx < s && ny < s && nz < s && density.get(nx, ny, nz).material.is_solid() {
            return true;
        }
    }
    false
}

pub(crate) fn has_air_neighbor(density: &DensityField, x: usize, y: usize, z: usize) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::material::Material;
    use voxel_gen::density::DensityField;

    /// Full hermite re-extraction after mining produces a valid mesh with no NaN vertices,
    /// and edges in unmodified regions match the original extraction.
    #[test]
    fn test_full_reextract_matches_initial() {
        use voxel_gen::hermite_extract::extract_hermite_data;

        let chunk_size = 4usize;
        let size = chunk_size + 1;

        // Create a density field with a surface: solid below y=2, air above
        let mut field = DensityField::new(size);
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let s = field.get_mut(x, y, z);
                    if y <= 2 {
                        s.density = 1.0;
                        s.material = Material::Limestone;
                    } else {
                        s.density = -1.0;
                        s.material = Material::Air;
                    }
                }
            }
        }

        // Initial full extraction → mesh A
        let hermite_a = extract_hermite_data(&field);
        let cell_size = size - 1;
        let dc_a = voxel_core::dual_contouring::solve::solve_dc_vertices(&hermite_a, cell_size);
        let mesh_a = voxel_core::dual_contouring::mesh_gen::generate_mesh(&hermite_a, &dc_a, cell_size);

        // Verify mesh A is valid
        assert!(!mesh_a.vertices.is_empty(), "initial mesh should have vertices");
        for v in &mesh_a.vertices {
            assert!(!v.position[0].is_nan() && !v.position[1].is_nan() && !v.position[2].is_nan(),
                "initial mesh has NaN vertex");
        }

        // Mine a sphere: carve out y=1..2, x=1..3, z=1..3
        for z in 1..=3 {
            for y in 1..=2 {
                for x in 1..=3 {
                    let s = field.get_mut(x, y, z);
                    s.density = -1.0;
                    s.material = Material::Air;
                }
            }
        }

        // Smooth the mined region
        smooth_mine_boundary(&mut field, 1, 1, 1, 3, 2, 3, 2, 0.5);

        // Full re-extraction after mining → mesh B
        let hermite_b = extract_hermite_data(&field);
        let dc_b = voxel_core::dual_contouring::solve::solve_dc_vertices(&hermite_b, cell_size);
        let mesh_b = voxel_core::dual_contouring::mesh_gen::generate_mesh(&hermite_b, &dc_b, cell_size);

        // Verify mesh B has no NaN vertices
        for v in &mesh_b.vertices {
            assert!(!v.position[0].is_nan() && !v.position[1].is_nan() && !v.position[2].is_nan(),
                "post-mine mesh has NaN vertex");
        }

        // Verify mesh B still has geometry (mining carved some but not all)
        assert!(!mesh_b.vertices.is_empty(), "post-mine mesh should have vertices");

        // Verify edges outside the mined region are consistent:
        // Edges at z=0 (untouched) should match between A and B
        for (key, edge_b) in &hermite_b.edges {
            let z = key.z();
            let y = key.y();
            if z == 0 && y <= 2 {
                // This edge is in the z=0 slice, below the surface — density unchanged
                if let Some(edge_a) = hermite_a.edges.get(&key) {
                    assert!(
                        (edge_b.t - edge_a.t).abs() < 1e-5,
                        "edge at ({},{},{}) axis {} has t={} but original had t={}",
                        key.x(), key.y(), key.z(), key.axis(), edge_b.t, edge_a.t
                    );
                }
            }
        }
    }
}
