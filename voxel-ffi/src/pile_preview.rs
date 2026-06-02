//! Pile preview mesh extraction for the cinematic-collapse buildup illusion.
//!
//! After `place_collapse_pile` writes the debris cells (so we know the final
//! pile shape) but BEFORE we commit the writes to the live density store,
//! we extract a 4-tier preview mesh and roll back. UE shows the tiers in
//! sequence over ~0.4 s so the heap appears to grow into place instead of
//! snapping in. The real density commit + remesh fires at the end of the
//! reveal.
//!
//! Each tier mesh is **cumulative**: tier 0 is the bottom 25 % of the heap;
//! tier 3 is the full pile. UE swaps which tier is visible per frame.
//!
//! Extraction strategy mirrors `slab::extract_slab_mesh`:
//!
//!   1. Walk `PlacementResult.written_cells`, dedup to unique world cells,
//!      and keep only those that ended up solid (density >= 0, material != Air).
//!   2. Compute world-space bounding box. Determine 4 cumulative Y cutoffs.
//!   3. For each tier, build a temporary cubic DensityField sized to fit
//!      the tier subset, fill solid cells, run hermite extract + DC, smooth
//!      and convert to UE space.
//!
//! Returns a `Vec<ConvertedMesh>` of length 4 (some tiers may be empty if
//! the heap is very thin; the caller still emits an empty FFI message so UE
//! sees exactly 4 tiers per pile).

use std::collections::HashMap;

use voxel_core::density_ops::WrittenCell;
use voxel_core::dual_contouring::mesh_gen::generate_mesh;
use voxel_core::dual_contouring::solve::solve_dc_vertices;
use voxel_core::material::Material;
use voxel_gen::density::DensityField;
use voxel_gen::hermite_extract::extract_hermite_data;

use crate::convert::{bucket_mesh_by_material, convert_mesh_to_ue_scaled};
use crate::types::ConvertedMesh;

/// Number of preview tiers (cumulative reveal). Bumping this changes how
/// granular the buildup looks; the deferred worker emits one result per
/// tier, and UE swaps which section is visible across the reveal window.
/// At 8 tiers × 150 ms slot = 1200 ms total reveal, the heap rises in
/// distinct visible steps without stuttering.
pub const PILE_PREVIEW_TIER_COUNT: usize = 8;

/// Maximum grid size of a single tier's temp DC field. Large piles get
/// downsampled implicitly because grid_size clamps; if a pile spans more
/// than 64 voxels in any axis it falls back to no preview.
const MAX_TIER_GRID: usize = 96;

/// Extract `PILE_PREVIEW_TIER_COUNT` cumulative tier meshes from the cells
/// written by `place_collapse_pile`.
///
/// `chunk_size` is the live engine chunk size (used to convert WrittenCell
/// chunk-local coords back to world coords). `voxel_scale` and `world_scale`
/// match `extract_slab_mesh`.
///
/// Returns a Vec of length 4. Empty tiers are returned as `ConvertedMesh`
/// with empty positions/indices (the caller emits an empty FFI mesh so UE
/// can keep its 4-tier accumulator simple).
pub fn extract_pile_tier_meshes(
    written_cells: &[WrittenCell],
    chunk_size: usize,
    voxel_scale: f32,
    world_scale: f32,
) -> Vec<ConvertedMesh> {
    let cs = chunk_size as i32;

    // 1. Dedup written cells to unique world cells, keeping only those that
    //    ended up solid (density >= 0, non-Air). pile writes through
    //    `write_force` fan out to multi-chunk seam locations, so the same
    //    world cell shows up multiple times. We just take the first
    //    occurrence — they all have the same target density+material.
    let mut solid_cells: HashMap<(i32, i32, i32), (f32, Material)> = HashMap::new();
    for w in written_cells {
        if w.new_density < 0.0 || (w.new_material as u8) == 0 {
            continue;
        }
        let wx = w.key.0 * cs + w.lx as i32;
        let wy = w.key.1 * cs + w.ly as i32;
        let wz = w.key.2 * cs + w.lz as i32;
        solid_cells.entry((wx, wy, wz)).or_insert((w.new_density, w.new_material));
    }

    if solid_cells.is_empty() {
        return empty_tiers();
    }

    // 2. Bounding box of the solid pile cells.
    let mut min_x = i32::MAX; let mut max_x = i32::MIN;
    let mut min_y = i32::MAX; let mut max_y = i32::MIN;
    let mut min_z = i32::MAX; let mut max_z = i32::MIN;
    for &(wx, wy, wz) in solid_cells.keys() {
        min_x = min_x.min(wx); max_x = max_x.max(wx);
        min_y = min_y.min(wy); max_y = max_y.max(wy);
        min_z = min_z.min(wz); max_z = max_z.max(wz);
    }

    let pile_h = (max_y - min_y).max(0) as f32 + 1.0;

    let mut out: Vec<ConvertedMesh> = Vec::with_capacity(PILE_PREVIEW_TIER_COUNT);

    for tier in 0..PILE_PREVIEW_TIER_COUNT {
        // Cumulative cutoff: tier 0 covers bottom 1/4, tier 3 covers the
        // full pile. We use a fractional cutoff and round up so the top
        // tier always includes max_y.
        let frac = (tier + 1) as f32 / PILE_PREVIEW_TIER_COUNT as f32;
        let cutoff_y = if tier + 1 == PILE_PREVIEW_TIER_COUNT {
            max_y
        } else {
            (min_y as f32 + frac * pile_h - 0.5).floor() as i32
        };

        // Filter cells for this tier (cumulative).
        let mut tier_cells: Vec<((i32, i32, i32), (f32, Material))> = solid_cells.iter()
            .filter(|&(&(_, wy, _), _)| wy <= cutoff_y)
            .map(|(k, v)| (*k, *v))
            .collect();

        if tier_cells.is_empty() {
            out.push(empty_mesh());
            continue;
        }

        // Tier-local bbox (could be smaller than the full pile).
        let mut tmin_x = i32::MAX; let mut tmax_x = i32::MIN;
        let mut tmin_y = i32::MAX; let mut tmax_y = i32::MIN;
        let mut tmin_z = i32::MAX; let mut tmax_z = i32::MIN;
        for ((wx, wy, wz), _) in &tier_cells {
            tmin_x = tmin_x.min(*wx); tmax_x = tmax_x.max(*wx);
            tmin_y = tmin_y.min(*wy); tmax_y = tmax_y.max(*wy);
            tmin_z = tmin_z.min(*wz); tmax_z = tmax_z.max(*wz);
        }

        // 1-voxel padding for surface normals.
        let pad = 1i32;
        tmin_x -= pad; tmax_x += pad;
        tmin_y -= pad; tmax_y += pad;
        tmin_z -= pad; tmax_z += pad;

        let size_x = (tmax_x - tmin_x + 1) as usize;
        let size_y = (tmax_y - tmin_y + 1) as usize;
        let size_z = (tmax_z - tmin_z + 1) as usize;
        let grid_size = size_x.max(size_y).max(size_z);
        if grid_size < 2 || grid_size > MAX_TIER_GRID {
            out.push(empty_mesh());
            continue;
        }

        // Build temp DF with air defaults, fill the tier cells solid.
        let mut temp_df = DensityField::new(grid_size);
        for sample in temp_df.samples.iter_mut() {
            sample.density = -1.0;
            sample.material = Material::Air;
        }
        for ((wx, wy, wz), (density, material)) in tier_cells.drain(..) {
            let lx = (wx - tmin_x) as usize;
            let ly = (wy - tmin_y) as usize;
            let lz = (wz - tmin_z) as usize;
            if lx >= grid_size || ly >= grid_size || lz >= grid_size { continue; }
            let s = temp_df.get_mut(lx, ly, lz);
            // Use the recorded target density (sub-voxel boundary aware).
            // Clamp to >= 0 so we always keep the cell solid for the preview.
            s.density = density.max(0.05);
            s.material = if (material as u8) > 0 && (material as u8) <= 41 {
                material
            } else {
                Material::Granite
            };
        }

        // Run the standard mesh extraction pipeline.
        let hermite = extract_hermite_data(&temp_df);
        if hermite.edges.is_empty() {
            out.push(empty_mesh());
            continue;
        }

        let cell_size = grid_size - 1;
        let dc_vertices = solve_dc_vertices(&hermite, cell_size);
        let mut mesh = generate_mesh(&hermite, &dc_vertices, cell_size);

        if mesh.vertices.is_empty() {
            out.push(empty_mesh());
            continue;
        }

        mesh.smooth(1, 0.3, 0.0, Some(cell_size));
        mesh.recalculate_normals();

        // World offset: temp grid starts at (tmin_x, tmin_y, tmin_z).
        for v in &mut mesh.vertices {
            v.position.x += tmin_x as f32;
            v.position.y += tmin_y as f32;
            v.position.z += tmin_z as f32;
        }

        let mut converted = convert_mesh_to_ue_scaled(&mesh, voxel_scale, world_scale);
        bucket_mesh_by_material(&mut converted);
        if converted.positions.is_empty() {
            out.push(empty_mesh());
        } else {
            out.push(converted);
        }
    }

    out
}

fn empty_tiers() -> Vec<ConvertedMesh> {
    (0..PILE_PREVIEW_TIER_COUNT).map(|_| empty_mesh()).collect()
}

fn empty_mesh() -> ConvertedMesh {
    ConvertedMesh {
        positions: Vec::new(),
        normals: Vec::new(),
        material_ids: Vec::new(),
        indices: Vec::new(),
        submeshes: Vec::new(),
        reveal_t: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cell(key: (i32,i32,i32), lx: usize, ly: usize, lz: usize, d: f32, m: Material) -> WrittenCell {
        WrittenCell {
            key, lx, ly, lz,
            new_density: d,
            new_material: m,
            orig_density: -1.0,
            orig_material: Material::Air,
        }
    }

    #[test]
    fn extract_returns_four_tiers_for_simple_heap() {
        // Simulate a 4x4x4 heap of granite cells. Each cell at world
        // (x, y, z) for y in 0..4 with density 1.0.
        let mut written: Vec<WrittenCell> = Vec::new();
        let cs = 16i32;
        for y in 0..4 {
            for z in 0..4 {
                for x in 0..4 {
                    // Pretend chunk (0,0,0) so local coords == world coords.
                    written.push(make_cell((0, 0, 0), x, y, z, 1.0, Material::Granite));
                }
            }
        }
        let tiers = extract_pile_tier_meshes(&written, cs as usize, 1.0, 40.0);
        assert_eq!(tiers.len(), PILE_PREVIEW_TIER_COUNT);
        // Tier 3 (full) should have geometry.
        assert!(!tiers[3].positions.is_empty(), "top tier should have a mesh");
    }

    #[test]
    fn extract_returns_empty_tiers_for_no_cells() {
        let tiers = extract_pile_tier_meshes(&[], 16, 1.0, 40.0);
        assert_eq!(tiers.len(), PILE_PREVIEW_TIER_COUNT);
        for t in &tiers {
            assert!(t.positions.is_empty());
        }
    }

    #[test]
    fn extract_skips_air_and_zero_material_cells() {
        let mut written: Vec<WrittenCell> = Vec::new();
        // One Air cell, one Granite cell.
        written.push(make_cell((0, 0, 0), 0, 0, 0, -1.0, Material::Air));
        written.push(make_cell((0, 0, 0), 1, 0, 0, 1.0, Material::Granite));
        let tiers = extract_pile_tier_meshes(&written, 16, 1.0, 40.0);
        assert_eq!(tiers.len(), PILE_PREVIEW_TIER_COUNT);
    }
}
