use crate::types::{ConvertedMesh, FfiCrystalPlacement, FfiSubmesh, FfiVec3};
use voxel_core::mesh::Mesh;
use voxel_gen::CrystalPlacement;

/// Convert a Rust-space mesh (Y-up, right-hand) to UE space (Z-up, left-hand).
///
/// Assumes mesh vertices already have world_origin added to their positions
/// (this is done during mesh generation in the worker).
///
/// Position transform: swap Y<->Z, negate new Y, then scale.
/// Normal transform: swap Y<->Z, negate new Y (no scale).
/// Winding: swap indices[0] and indices[2] to flip triangle facing.
pub fn convert_mesh_to_ue(mesh: &Mesh, scale: f32) -> ConvertedMesh {
    convert_mesh_to_ue_scaled(mesh, 1.0, scale)
}

/// Convert mesh with independent voxel_scale and world_scale.
/// Vertex positions are in grid-space [0, chunk_size]. They get scaled by
/// voxel_scale first (to world-space voxel units), then by world_scale (to UE units).
pub fn convert_mesh_to_ue_scaled(mesh: &Mesh, voxel_scale: f32, world_scale: f32) -> ConvertedMesh {
    let combined_scale = voxel_scale * world_scale;
    let vert_count = mesh.vertices.len();
    let mut positions = Vec::with_capacity(vert_count);
    let mut normals = Vec::with_capacity(vert_count);
    let mut material_ids = Vec::with_capacity(vert_count);

    for v in &mesh.vertices {
        let p = v.position;
        positions.push(FfiVec3 {
            x: p.x * combined_scale,
            y: -p.z * combined_scale,
            z: p.y * combined_scale,
        });

        let n = v.normal;
        normals.push(FfiVec3 {
            x: n.x,
            y: -n.z,
            z: n.y,
        });

        material_ids.push(v.material as u8);
    }

    let tri_count = mesh.triangles.len();
    let mut indices = Vec::with_capacity(tri_count * 3);
    for tri in &mesh.triangles {
        let i0 = tri.indices[0] as usize;
        let i1 = tri.indices[1] as usize;
        let i2 = tri.indices[2] as usize;

        // Compute geometric face normal from cross product
        let p0 = mesh.vertices[i0].position;
        let p1 = mesh.vertices[i1].position;
        let p2 = mesh.vertices[i2].position;
        let face_normal = (p1 - p0).cross(p2 - p0);

        // Compare with average vertex normal to determine correct winding
        let avg_normal = mesh.vertices[i0].normal
            + mesh.vertices[i1].normal
            + mesh.vertices[i2].normal;

        if face_normal.dot(avg_normal) < 0.0 {
            // Face normal opposes vertex normals — flip winding
            indices.push(tri.indices[2]);
            indices.push(tri.indices[1]);
            indices.push(tri.indices[0]);
        } else {
            indices.push(tri.indices[0]);
            indices.push(tri.indices[1]);
            indices.push(tri.indices[2]);
        }
    }

    ConvertedMesh {
        positions,
        normals,
        material_ids,
        indices,
        submeshes: Vec::new(),
    }
}

/// Convert UE world position back to Rust coordinate space (for mining).
pub fn from_ue_world_pos(x: f32, y: f32, z: f32, scale: f32) -> glam::Vec3 {
    glam::Vec3::new(x / scale, z / scale, -y / scale)
}

/// Convert UE normal direction back to Rust coordinate space.
pub fn from_ue_normal(x: f32, y: f32, z: f32) -> glam::Vec3 {
    glam::Vec3::new(x, z, -y)
}

/// Convert UE chunk coordinates to Rust chunk coordinates.
/// UE is Z-up left-hand, Rust is Y-up right-hand.
/// UE (x, y, z) -> Rust (x, z, -y)
pub fn ue_chunk_to_rust(cx: i32, cy: i32, cz: i32) -> (i32, i32, i32) {
    (cx, cz, -cy)
}

/// Convert Rust chunk coordinates back to UE chunk coordinates.
/// Rust (x, y, z) -> UE (x, -z, y)
pub fn rust_chunk_to_ue(cx: i32, cy: i32, cz: i32) -> (i32, i32, i32) {
    (cx, -cz, cy)
}

/// Rearrange a ConvertedMesh so vertices and indices are grouped by material.
/// Populates the submeshes field with offset/count for each material section.
pub fn bucket_mesh_by_material(mesh: &mut ConvertedMesh) {
    if mesh.indices.is_empty() {
        return;
    }

    // Group triangles by material of their first vertex
    let mut buckets: std::collections::BTreeMap<u8, Vec<u32>> = std::collections::BTreeMap::new();
    let tri_count = mesh.indices.len() / 3;
    for tri in 0..tri_count {
        let first_vert = mesh.indices[tri * 3] as usize;
        let mat_id = mesh.material_ids[first_vert];
        buckets.entry(mat_id).or_default().push(tri as u32);
    }

    let mut new_positions = Vec::with_capacity(mesh.positions.len());
    let mut new_normals = Vec::with_capacity(mesh.normals.len());
    let mut new_material_ids = Vec::with_capacity(mesh.material_ids.len());
    let mut new_indices = Vec::with_capacity(mesh.indices.len());
    let mut submeshes = Vec::with_capacity(buckets.len());

    // Per-bucket vertex remap. Original indices are dense (0..vert_count), so we
    // dedup with a flat array instead of a fresh SipHash `HashMap` per material
    // bucket. An `epoch` tag per slot records which bucket last wrote it, so the
    // table is "reset" for each bucket by bumping `epoch` — no per-bucket
    // allocation and no clearing pass. `epoch` starts at 0 and the tags are
    // 0-initialised, so the first bucket (epoch 1) sees every slot as absent.
    // (At most ~256 material buckets, so `epoch` never wraps.)
    let vert_count = mesh.positions.len();
    let mut remap_idx: Vec<u32> = vec![0u32; vert_count];
    let mut remap_epoch: Vec<u32> = vec![0u32; vert_count];
    let mut epoch: u32 = 0;

    for (mat_id, triangles) in &buckets {
        let vertex_offset = new_positions.len() as u32;
        let index_offset = new_indices.len() as u32;

        // Remap vertices for this material section. Bump the epoch so all slots
        // from previous buckets read as absent.
        epoch += 1;

        for &tri_idx in triangles {
            for corner in 0..3 {
                let orig_idx = mesh.indices[tri_idx as usize * 3 + corner];
                let oi = orig_idx as usize;
                let new_idx = if remap_epoch[oi] == epoch {
                    remap_idx[oi]
                } else {
                    let idx = new_positions.len() as u32;
                    remap_epoch[oi] = epoch;
                    remap_idx[oi] = idx;
                    new_positions.push(mesh.positions[oi]);
                    new_normals.push(mesh.normals[oi]);
                    new_material_ids.push(mesh.material_ids[oi]);
                    idx
                };
                new_indices.push(new_idx);
            }
        }

        let vertex_count = new_positions.len() as u32 - vertex_offset;
        let index_count = new_indices.len() as u32 - index_offset;

        submeshes.push(FfiSubmesh {
            material_id: *mat_id,
            vertex_offset,
            vertex_count,
            index_offset,
            index_count,
        });
    }

    mesh.positions = new_positions;
    mesh.normals = new_normals;
    mesh.material_ids = new_material_ids;
    mesh.indices = new_indices;
    mesh.submeshes = submeshes;
}

/// Convert mushroom placements from Rust Y-up to UE Z-up coordinates.
/// Same swap-and-negate rule as crystals.
pub fn convert_mushrooms_to_ue(
    placements: &[voxel_gen::MushroomPlacement],
    voxel_scale: f32,
    world_scale: f32,
) -> Vec<crate::types::FfiMushroomInstance> {
    let combined_scale = voxel_scale * world_scale;
    placements.iter().map(|p| {
        crate::types::FfiMushroomInstance {
            x: p.x * combined_scale,
            y: -p.z * combined_scale,
            z: p.y * combined_scale,
            normal_x: p.normal_x,
            normal_y: -p.normal_z,
            normal_z: p.normal_y,
            scale: p.scale,
            yaw: p.yaw,
            kind: p.kind as u8,
            anchor_lx: p.anchor_lx,
            anchor_ly: p.anchor_ly,
            anchor_lz: p.anchor_lz,
        }
    }).collect()
}

/// Convert crystal placements from Rust Y-up to UE Z-up coordinates.
/// Position transform: (x, y, z) -> (x * scale, -z * scale, y * scale)
/// Normal transform: (nx, ny, nz) -> (nx, -nz, ny)
pub fn convert_crystals_to_ue(
    placements: &[CrystalPlacement],
    voxel_scale: f32,
    world_scale: f32,
) -> Vec<FfiCrystalPlacement> {
    let combined_scale = voxel_scale * world_scale;
    placements.iter().map(|p| {
        FfiCrystalPlacement {
            x: p.x * combined_scale,
            y: -p.z * combined_scale,
            z: p.y * combined_scale,
            normal_x: p.normal_x,
            normal_y: -p.normal_z,
            normal_z: p.normal_y,
            ore_type: p.ore_type,
            size_class: p.size_class,
            scale: p.scale,
        }
    }).collect()
}

/// Gate 1 diagnostic: hardcoded 8-vertex cube (12 triangles, single material).
/// Every chunk becomes an identical cube, testing whether the bug is in mesh
/// generation vs transport/rendering.
#[cfg(feature = "diag-gate-1")]
pub fn diagnostic_test_cube() -> Mesh {
    use voxel_core::material::Material;
    use voxel_core::mesh::{Triangle, Vertex};

    // Unit cube from (1,1,1) to (2,2,2) — safely inside a chunk cell
    let corners = [
        glam::Vec3::new(1.0, 1.0, 1.0), // 0: ---
        glam::Vec3::new(2.0, 1.0, 1.0), // 1: +--
        glam::Vec3::new(2.0, 2.0, 1.0), // 2: ++-
        glam::Vec3::new(1.0, 2.0, 1.0), // 3: -+-
        glam::Vec3::new(1.0, 1.0, 2.0), // 4: --+
        glam::Vec3::new(2.0, 1.0, 2.0), // 5: +-+
        glam::Vec3::new(2.0, 2.0, 2.0), // 6: +++
        glam::Vec3::new(1.0, 2.0, 2.0), // 7: -++
    ];

    // 6 faces, each with an outward normal
    let faces: [([usize; 4], glam::Vec3); 6] = [
        ([0, 3, 2, 1], glam::Vec3::NEG_Z), // front  (z-)
        ([4, 5, 6, 7], glam::Vec3::Z),      // back   (z+)
        ([0, 4, 7, 3], glam::Vec3::NEG_X),  // left   (x-)
        ([1, 2, 6, 5], glam::Vec3::X),      // right  (x+)
        ([0, 1, 5, 4], glam::Vec3::NEG_Y),  // bottom (y-)
        ([3, 7, 6, 2], glam::Vec3::Y),      // top    (y+)
    ];

    let mut vertices = Vec::with_capacity(24);
    let mut triangles = Vec::with_capacity(12);

    for (quad, normal) in &faces {
        let base = vertices.len() as u32;
        for &ci in quad {
            vertices.push(Vertex {
                position: corners[ci],
                normal: *normal,
                material: Material::Limestone,
            });
        }
        triangles.push(Triangle { indices: [base, base + 1, base + 2] });
        triangles.push(Triangle { indices: [base, base + 2, base + 3] });
    }

    Mesh { vertices, triangles }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use voxel_core::material::Material;
    use voxel_core::mesh::{Mesh, Triangle, Vertex};

    #[test]
    fn coordinate_roundtrip() {
        let scale = 100.0;
        let original = Vec3::new(5.0, 10.0, -3.0);

        // Rust -> UE
        let ue_x = original.x * scale;
        let ue_y = -original.z * scale;
        let ue_z = original.y * scale;

        // UE -> Rust
        let back = from_ue_world_pos(ue_x, ue_y, ue_z, scale);
        assert!((back - original).length() < 1e-5);
    }

    #[test]
    fn normal_roundtrip() {
        let original = Vec3::new(0.0, 1.0, 0.0); // Y-up in Rust

        // Rust -> UE normal: (x, -z, y) = (0, 0, 1) => Z-up in UE
        let ue_n = FfiVec3 {
            x: original.x,
            y: -original.z,
            z: original.y,
        };
        assert!((ue_n.z - 1.0).abs() < 1e-5);

        // UE -> Rust
        let back = from_ue_normal(ue_n.x, ue_n.y, ue_n.z);
        assert!((back - original).length() < 1e-5);
    }

    // Build a small multi-material mesh whose vertices are shared across
    // material boundaries (the case that forces the same original index to be
    // remapped in more than one bucket).
    fn multimat_mesh() -> ConvertedMesh {
        // 6 vertices; position.x encodes vertex identity, normal.z mirrors it.
        let positions: Vec<FfiVec3> = (0..6)
            .map(|i| FfiVec3 { x: i as f32, y: 0.0, z: 0.0 })
            .collect();
        let normals: Vec<FfiVec3> = (0..6)
            .map(|i| FfiVec3 { x: 0.0, y: 0.0, z: i as f32 })
            .collect();
        let material_ids: Vec<u8> = vec![1, 1, 2, 2, 0, 0];
        // 4 triangles; first-vertex material decides the bucket. Vertices 1,2,3
        // appear in triangles of differing first-vertex materials, so they get
        // remapped independently per bucket.
        let indices: Vec<u32> = vec![
            0, 1, 2, // bucket mat=1
            1, 2, 3, // bucket mat=1
            2, 3, 4, // bucket mat=2
            4, 5, 1, // bucket mat=0
        ];
        ConvertedMesh { positions, normals, material_ids, indices, submeshes: Vec::new() }
    }

    #[test]
    fn bucket_preserves_triangles_and_partitions() {
        let src = multimat_mesh();
        // Original triangles encoded as vertex-identity triples (pos.x).
        let orig_tris: Vec<[u32; 3]> = src
            .indices
            .chunks(3)
            .map(|c| {
                [
                    src.positions[c[0] as usize].x as u32,
                    src.positions[c[1] as usize].x as u32,
                    src.positions[c[2] as usize].x as u32,
                ]
            })
            .collect();

        let mut out = multimat_mesh();
        bucket_mesh_by_material(&mut out);

        // Decoded output triangles must reproduce the original triangle set.
        let decoded: Vec<[u32; 3]> = out
            .indices
            .chunks(3)
            .map(|c| {
                [
                    out.positions[c[0] as usize].x as u32,
                    out.positions[c[1] as usize].x as u32,
                    out.positions[c[2] as usize].x as u32,
                ]
            })
            .collect();
        let mut a = orig_tris.clone();
        let mut b = decoded.clone();
        a.sort();
        b.sort();
        assert_eq!(a, b, "bucketing must preserve the triangle set exactly");

        // Each output vertex keeps its paired attributes through the remap.
        for (i, p) in out.positions.iter().enumerate() {
            assert_eq!(
                p.x as u32, out.normals[i].z as u32,
                "vertex attrs must stay paired through the remap"
            );
        }

        // Submeshes partition the index buffer contiguously and completely,
        // and every index points inside its submesh's own vertex span.
        assert!(!out.submeshes.is_empty());
        let mut covered = 0u32;
        for sm in &out.submeshes {
            assert_eq!(sm.index_offset, covered, "submesh ranges must be contiguous");
            covered += sm.index_count;
            let vlo = sm.vertex_offset;
            let vhi = sm.vertex_offset + sm.vertex_count;
            for &ix in &out.indices
                [sm.index_offset as usize..(sm.index_offset + sm.index_count) as usize]
            {
                assert!(ix >= vlo && ix < vhi, "index must stay within its submesh vertex span");
            }
        }
        assert_eq!(covered as usize, out.indices.len(), "submeshes must cover all indices");
    }

    #[test]
    fn winding_corrected_by_normal() {
        // Triangle with CCW winding matching Y-up normal → keep winding
        let mesh = Mesh {
            vertices: vec![
                Vertex {
                    position: Vec3::new(0.0, 0.0, 0.0),
                    normal: Vec3::Y,
                    material: Material::Limestone,
                },
                Vertex {
                    position: Vec3::new(1.0, 0.0, 0.0),
                    normal: Vec3::Y,
                    material: Material::Limestone,
                },
                Vertex {
                    position: Vec3::new(0.0, 0.0, 1.0),
                    normal: Vec3::Y,
                    material: Material::Limestone,
                },
            ],
            triangles: vec![Triangle {
                indices: [0, 1, 2],
            }],
        };

        let converted = convert_mesh_to_ue(&mesh, 100.0);
        // face_normal = (1,0,0)x(0,0,1) = (0,-1,0), dot with (0,3,0) < 0 → flip
        assert_eq!(converted.indices, vec![2, 1, 0]);

        // Opposite winding → face_normal agrees with vertex normals → keep
        let mesh2 = Mesh {
            vertices: mesh.vertices.clone(),
            triangles: vec![Triangle {
                indices: [0, 2, 1],
            }],
        };
        let converted2 = convert_mesh_to_ue(&mesh2, 100.0);
        assert_eq!(converted2.indices, vec![0, 2, 1]);
    }
}
