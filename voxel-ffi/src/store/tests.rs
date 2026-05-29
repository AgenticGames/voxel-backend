//! Unit tests for `ChunkStore` boundary sync and spatial search.
//!
//! Split out of the original `store.rs` god file (behavior-preserving).

use std::collections::HashMap;

use glam::Vec3;
use voxel_core::material::Material;
use voxel_core::octree::node::VoxelSample;
use voxel_gen::density::DensityField;

use super::boundary::average_boundary_voxel;
use super::*;


/// Helper: create a solid density field of given grid size (chunk_size+1).
fn make_solid_field(size: usize) -> DensityField {
    let mut field = DensityField::new(size);
    for s in &mut field.samples {
        s.density = 1.0;
        s.material = Material::Limestone;
    }
    field
}

/// Mine asymmetric patterns in two adjacent chunks and smooth independently,
/// then verify sync_boundary_density makes the overlap match.
#[test]
fn test_boundary_density_sync_after_mine() {
    let chunk_size = 4usize;
    let size = chunk_size + 1; // grid size = 5

    // Two adjacent chunks along X: A=(0,0,0), B=(1,0,0)
    let mut fields: HashMap<(i32, i32, i32), DensityField> = HashMap::new();
    fields.insert((0, 0, 0), make_solid_field(size));
    fields.insert((1, 0, 0), make_solid_field(size));

    // Asymmetric mining: carve a wide tunnel in A but narrow in B.
    // This creates different neighbor patterns so smoothing diverges at overlap.
    // Chunk A: carve x=2..4, y=0..4, z=1..3  (wide, reaching overlap at x=4)
    for z in 1..=3 {
        for y in 0..size {
            for x in 2..size {
                let s = fields.get_mut(&(0, 0, 0)).unwrap().get_mut(x, y, z);
                s.density = -1.0;
                s.material = Material::Air;
            }
        }
    }
    // Chunk B: carve only x=0, y=2..2, z=2..2  (narrow, overlap at x=0)
    {
        let s = fields.get_mut(&(1, 0, 0)).unwrap().get_mut(0, 2, 2);
        s.density = -1.0;
        s.material = Material::Air;
    }

    // Smooth each chunk independently (simulating post-mine smoothing)
    crate::mining::smooth_mine_boundary(
        fields.get_mut(&(0, 0, 0)).unwrap(),
        1, 0, 0, chunk_size, chunk_size, chunk_size,
        3, 0.5,
    );
    crate::mining::smooth_mine_boundary(
        fields.get_mut(&(1, 0, 0)).unwrap(),
        0, 0, 0, 1, chunk_size, chunk_size,
        3, 0.5,
    );

    // Before sync: overlap voxels should differ due to asymmetric carving
    let mut any_differ = false;
    for y in 0..size {
        for z in 0..size {
            let a = fields[&(0, 0, 0)].get(chunk_size, y, z).density;
            let b = fields[&(1, 0, 0)].get(0, y, z).density;
            if (a - b).abs() > 1e-6 {
                any_differ = true;
            }
        }
    }
    assert!(any_differ, "smoothing should have desynchronized at least some overlap voxels");

    // Run boundary sync
    let dirty_chunks = vec![
        ((0, 0, 0), 1usize, 0usize, 0usize, chunk_size, chunk_size, chunk_size),
        ((1, 0, 0), 0usize, 0usize, 0usize, 1usize, chunk_size, chunk_size),
    ];
    let extra = sync_boundary_density(&mut fields, &dirty_chunks, chunk_size);

    // Both chunks were already dirty, so no extra neighbors expected
    assert!(extra.is_empty(), "both chunks already dirty, no extras expected");

    // After sync: overlap voxels must match exactly
    for y in 0..size {
        for z in 0..size {
            let a = fields[&(0, 0, 0)].get(chunk_size, y, z);
            let b = fields[&(1, 0, 0)].get(0, y, z);
            assert!(
                (a.density - b.density).abs() < 1e-6,
                "density mismatch at overlap y={y} z={z}: A={} B={}",
                a.density, b.density
            );
            assert_eq!(
                a.material, b.material,
                "material mismatch at overlap y={y} z={z}"
            );
        }
    }
}

/// Mine one chunk with dirty_expand reaching the boundary; verify
/// the neighbor gets added to extra_dirty and overlaps match after sync.
#[test]
fn test_boundary_sync_single_chunk_dirty_expand() {
    let chunk_size = 4usize;
    let size = chunk_size + 1;

    let mut fields: HashMap<(i32, i32, i32), DensityField> = HashMap::new();
    fields.insert((0, 0, 0), make_solid_field(size));
    fields.insert((1, 0, 0), make_solid_field(size));

    // Mine near the +X face of chunk A only (x=3,4)
    for y in 1..=3 {
        for z in 1..=3 {
            for x in 3..=4 {
                let s = fields.get_mut(&(0, 0, 0)).unwrap().get_mut(x, y, z);
                s.density = -1.0;
                s.material = Material::Air;
            }
        }
    }

    // Smooth only chunk A
    crate::mining::smooth_mine_boundary(
        fields.get_mut(&(0, 0, 0)).unwrap(),
        2, 0, 0, 4, 4, 4,
        2, 0.5,
    );

    // Only chunk A is dirty, with max_x reaching chunk_size (the overlap face)
    let dirty_chunks = vec![
        ((0, 0, 0), 2usize, 0usize, 0usize, chunk_size, chunk_size, chunk_size),
    ];
    let extra = sync_boundary_density(&mut fields, &dirty_chunks, chunk_size);

    // Neighbor (1,0,0) should be added as extra dirty
    assert_eq!(extra.len(), 1, "neighbor should be added as extra dirty");
    assert_eq!(extra[0].0, (1, 0, 0));

    // After sync: overlap voxels must match
    for y in 0..size {
        for z in 0..size {
            let a = fields[&(0, 0, 0)].get(chunk_size, y, z);
            let b = fields[&(1, 0, 0)].get(0, y, z);
            assert!(
                (a.density - b.density).abs() < 1e-6,
                "density mismatch at overlap y={y} z={z}: A={} B={}",
                a.density, b.density
            );
            assert_eq!(
                a.material, b.material,
                "material mismatch at overlap y={y} z={z}"
            );
        }
    }
}

#[test]
fn test_boundary_voxel_solid_plus_air_preserves_solid() {
    let solid = VoxelSample { density: 0.8, material: Material::Granite };
    let air = VoxelSample { density: -0.5, material: Material::Air };

    // Solid + Air → preserves solid material
    let (d, m) = average_boundary_voxel(&solid, &air);
    assert_eq!(m, Material::Granite, "solid+air should preserve solid material");
    assert!((d - (-0.5)).abs() < 1e-6, "density should be min of the two");

    // Air + Solid → preserves solid material
    let (d2, m2) = average_boundary_voxel(&air, &solid);
    assert_eq!(m2, Material::Granite, "air+solid should preserve solid material");
    assert!((d2 - (-0.5)).abs() < 1e-6, "density should be min of the two");
}

#[test]
fn test_boundary_voxel_solid_solid_picks_higher_density() {
    let a = VoxelSample { density: 0.9, material: Material::Granite };
    let b = VoxelSample { density: 0.5, material: Material::Iron };

    let (_, m) = average_boundary_voxel(&a, &b);
    assert_eq!(m, Material::Granite, "should pick material with higher density");

    // Swap: b has higher density
    let (_, m2) = average_boundary_voxel(&b, &a);
    assert_eq!(m2, Material::Granite, "should pick material with higher density (a)");
}

#[test]
fn test_boundary_voxel_air_plus_air_stays_air() {
    let a = VoxelSample { density: -1.0, material: Material::Air };
    let b = VoxelSample { density: -0.3, material: Material::Air };

    let (_, m) = average_boundary_voxel(&a, &b);
    assert_eq!(m, Material::Air, "air+air should remain air");
}

/// Place one ore voxel at a known position with an air neighbor (surface).
fn place_surface_ore(
    field: &mut DensityField,
    ore: Material,
    x: usize,
    y: usize,
    z: usize,
    air_side: (i32, i32, i32),
) {
    let s = field.get_mut(x, y, z);
    s.density = 1.0;
    s.material = ore;
    let nx = (x as i32 + air_side.0) as usize;
    let ny = (y as i32 + air_side.1) as usize;
    let nz = (z as i32 + air_side.2) as usize;
    let n = field.get_mut(nx, ny, nz);
    n.density = -1.0;
    n.material = Material::Air;
}

#[test]
fn test_find_ore_voxels_filter_surface_sort_truncate() {
    let chunk_size = 4usize;
    let size = chunk_size + 1; // grid = 5
    let eb = chunk_size as f32; // voxel_scale = 1.0

    let mut store = ChunkStore::new(8);

    // Chunk A at (0,0,0):
    //   - Tin at (1,1,1) with air neighbor at (1,2,1)  -> SURFACE, world ~ (1.5,1.5,1.5)
    //   - Iron at (3,2,2) with air neighbor at (2,2,2) -> SURFACE, world ~ (3.5,2.5,2.5)
    //   - Tin BURIED at (2,1,1) — neighbors are all solid     -> SHOULD BE EXCLUDED
    let mut a = make_solid_field(size);
    place_surface_ore(&mut a, Material::Tin, 1, 1, 1, (0, 1, 0));
    place_surface_ore(&mut a, Material::Iron, 3, 2, 2, (-1, 0, 0));
    a.get_mut(2, 1, 1).material = Material::Tin; // buried, no air neighbor
    a.compute_metadata();
    assert!(a.has_ore_material, "chunk metadata must flag ores");
    store.density_fields.insert((0, 0, 0), a);

    // Chunk B at (1,0,0): one far Tin voxel.
    // World pos for (x=2,y=2,z=2) here = (1*4 + 2.5, 2.5, 2.5) = (6.5, 2.5, 2.5).
    let mut b = make_solid_field(size);
    place_surface_ore(&mut b, Material::Tin, 2, 2, 2, (0, -1, 0));
    b.compute_metadata();
    store.density_fields.insert((1, 0, 0), b);

    // Chunk C at (0,1,0): all-solid Limestone, NO ore. Must be skipped by metadata gate.
    let mut c = make_solid_field(size);
    c.compute_metadata();
    assert!(!c.has_ore_material);
    store.density_fields.insert((0, 1, 0), c);

    // Player near chunk A origin.
    let player = Vec3::new(0.5, 1.0, 1.0);

    // ── Test 1: filter on Tin specifically — should return 2 Tin voxels, near first.
    let tin_only = store.find_ore_voxels(
        player,
        100.0,
        Material::Tin as u8,
        16,
        chunk_size,
        eb,
    );
    assert_eq!(tin_only.len(), 2, "should find both Tin surface voxels and skip the buried one");
    for (_, m) in &tin_only {
        assert_eq!(*m, Material::Tin as u8);
    }
    // First result should be the closer Tin (chunk A).
    let d0 = (tin_only[0].0 - player).length();
    let d1 = (tin_only[1].0 - player).length();
    assert!(d0 <= d1, "results must be sorted by distance");
    assert!(d0 < 2.0, "near tin should be within ~2 units of player, got {d0}");

    // ── Test 2: any-ore filter (0xFF) — should return Tin x2 + Iron x1.
    let any_ore = store.find_ore_voxels(
        player,
        100.0,
        0xFF,
        16,
        chunk_size,
        eb,
    );
    assert_eq!(any_ore.len(), 3, "any-ore should return all 3 surface ore voxels");
    let mut materials: Vec<u8> = any_ore.iter().map(|(_, m)| *m).collect();
    materials.sort();
    assert_eq!(materials, vec![Material::Iron as u8, Material::Tin as u8, Material::Tin as u8]);

    // ── Test 3: max_results truncation.
    let capped = store.find_ore_voxels(player, 100.0, 0xFF, 1, chunk_size, eb);
    assert_eq!(capped.len(), 1);

    // ── Test 4: radius excludes far voxels.
    // The chunk-B Tin is at world (6.5,2.5,2.5), distance from player ~6.3.
    let near_only = store.find_ore_voxels(player, 3.0, Material::Tin as u8, 16, chunk_size, eb);
    assert_eq!(near_only.len(), 1, "radius=3 should exclude far Tin in chunk B");

    // ── Test 5: zero max_results returns empty quickly.
    let empty = store.find_ore_voxels(player, 100.0, 0xFF, 0, chunk_size, eb);
    assert!(empty.is_empty());
}
