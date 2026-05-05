//! Slab mesh extraction for the collapse system.
//!
//! Before a collapsing slab's voxels are removed from the density field,
//! this module extracts a ProceduralMesh of just the slab region so UE
//! can spawn a temporary falling actor.

use std::collections::HashMap;

use voxel_core::density::DensityField as CoreDensityField;
use voxel_core::dual_contouring::mesh_gen::generate_mesh;
use voxel_core::dual_contouring::solve::solve_dc_vertices;
use voxel_core::material::Material;
use voxel_core::stress::CollapseSlab;
use voxel_gen::density::DensityField;
use voxel_gen::hermite_extract::extract_hermite_data;

use crate::convert::{convert_mesh_to_ue_scaled, bucket_mesh_by_material};
use crate::types::ConvertedMesh;

/// Extract a mesh from a slab's voxels before they are cleared from the density field.
///
/// Creates a temporary DensityField covering the slab bounding box, copies only
/// the slab voxels as solid (everything else air), then runs the standard
/// hermite extraction + dual contouring pipeline on it.
///
/// Returns the ConvertedMesh in UE coordinate space, or None if the slab is
/// too small to produce geometry.
pub fn extract_slab_mesh(
    slab: &CollapseSlab,
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
    voxel_scale: f32,
    world_scale: f32,
) -> Option<ConvertedMesh> {
    if slab.voxels.is_empty() {
        return None;
    }

    // Compute padded bounding box (1 voxel padding for correct surface normals)
    let pad = 1i32;
    let min_x = slab.bb_min.0 - pad;
    let min_y = slab.bb_min.1 - pad;
    let min_z = slab.bb_min.2 - pad;
    let max_x = slab.bb_max.0 + pad;
    let max_y = slab.bb_max.1 + pad;
    let max_z = slab.bb_max.2 + pad;

    let size_x = (max_x - min_x + 1) as usize;
    let size_y = (max_y - min_y + 1) as usize;
    let size_z = (max_z - min_z + 1) as usize;

    // DensityField requires cubic size (all dimensions equal) for the DC grid
    let grid_size = size_x.max(size_y).max(size_z);
    if grid_size < 2 || grid_size > 64 {
        return None; // Too small or too large
    }

    // Create temporary density field: air everywhere, then fill slab voxels as solid
    let mut temp_df = DensityField::new(grid_size);
    for sample in temp_df.samples.iter_mut() {
        sample.density = -1.0;
        sample.material = Material::Air;
    }

    // Copy slab voxels from the slab's own data. The CollapseSlab preserves
    // the original material per voxel, so we use that directly. Some slab
    // voxels can have material=Air (when stress's BFS region included
    // marginal air-classified cells); those would render as matte black in
    // UE if we wrote them as solid+Air, so we substitute the dominant slab
    // material. If even the dominant is Air, fall back to Granite (a safe
    // rock material that always has a UE material instance).
    let _ = density_fields;
    let safe_dominant = if (slab.dominant_material as u8) > 0 {
        slab.dominant_material
    } else {
        // Find the first non-Air material in the slab voxels.
        slab.voxels.iter()
            .map(|v| v.material)
            .find(|m| (*m as u8) > 0)
            .unwrap_or(Material::Granite)
    };
    for cv in &slab.voxels {
        let lx = (cv.world_x - min_x) as usize;
        let ly = (cv.world_y - min_y) as usize;
        let lz = (cv.world_z - min_z) as usize;
        if lx >= grid_size || ly >= grid_size || lz >= grid_size { continue; }
        let sample = temp_df.get_mut(lx, ly, lz);
        sample.density = 1.0;
        // Substitute Air with the slab's dominant rock so the mesh isn't black.
        sample.material = if (cv.material as u8) > 0 { cv.material } else { safe_dominant };
    }

    // Run the standard mesh extraction pipeline
    let hermite = extract_hermite_data(&temp_df);
    if hermite.edges.is_empty() {
        return None; // No surface to extract
    }

    let cell_size = grid_size - 1;
    let dc_vertices = solve_dc_vertices(&hermite, cell_size);
    let mut mesh = generate_mesh(&hermite, &dc_vertices, cell_size);

    if mesh.vertices.is_empty() {
        return None;
    }

    // Light smoothing for the slab mesh
    mesh.smooth(1, 0.3, 0.0, Some(cell_size));
    mesh.recalculate_normals();

    // Offset mesh vertices so they're in the correct world position.
    // The temp DF grid starts at (min_x, min_y, min_z) in world coords,
    // so we add that offset before converting to UE space.
    for v in &mut mesh.vertices {
        v.position.x += min_x as f32;
        v.position.y += min_y as f32;
        v.position.z += min_z as f32;
    }

    let mut converted = convert_mesh_to_ue_scaled(&mesh, voxel_scale, world_scale);
    bucket_mesh_by_material(&mut converted);

    if converted.positions.is_empty() {
        return None;
    }

    Some(converted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::stress::CollapsedVoxel;

    #[test]
    fn extract_slab_mesh_from_solid_block() {
        // Create a 3x3x3 solid block as a slab
        let mut density_fields = HashMap::new();
        let mut df = DensityField::new(17);
        // Fill chunk (0,0,0) as solid granite
        for sample in df.samples.iter_mut() {
            sample.density = 1.0;
            sample.material = Material::Granite;
        }
        density_fields.insert((0, 0, 0), df);

        let mut voxels = Vec::new();
        for z in 5..8 {
            for y in 5..8 {
                for x in 5..8 {
                    voxels.push(CollapsedVoxel {
                        world_x: x, world_y: y, world_z: z,
                        material: Material::Granite,
                    });
                }
            }
        }

        let slab = CollapseSlab {
            voxels,
            bb_min: (5, 5, 5),
            bb_max: (7, 7, 7),
            center: (6.0, 6.0, 6.0),
            landing_y: 0,
            fall_distance: 5,
            dominant_material: Material::Granite,
        };

        let result = extract_slab_mesh(&slab, &density_fields, 16, 1.0, 1.0);
        assert!(result.is_some(), "Should extract a mesh from a solid block slab");
        let mesh = result.unwrap();
        assert!(!mesh.positions.is_empty(), "Mesh should have vertices");
        assert!(!mesh.indices.is_empty(), "Mesh should have indices");
    }
}
