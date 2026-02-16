use crate::types::{ConvertedMesh, FfiVec3};
use voxel_core::mesh::Mesh;

/// Convert a Rust-space mesh (Y-up, right-hand) to UE space (Z-up, left-hand).
///
/// Assumes mesh vertices already have world_origin added to their positions
/// (this is done during mesh generation in the worker).
///
/// Position transform: swap Y<->Z, negate new Y, then scale.
/// Normal transform: swap Y<->Z, negate new Y (no scale).
/// Winding: swap indices[0] and indices[2] to flip triangle facing.
pub fn convert_mesh_to_ue(mesh: &Mesh, scale: f32) -> ConvertedMesh {
    let vert_count = mesh.vertices.len();
    let mut positions = Vec::with_capacity(vert_count);
    let mut normals = Vec::with_capacity(vert_count);
    let mut material_ids = Vec::with_capacity(vert_count);

    for v in &mesh.vertices {
        let p = v.position;
        positions.push(FfiVec3 {
            x: p.x * scale,
            y: -p.z * scale,
            z: p.y * scale,
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
        // Swap winding: indices[0] <-> indices[2]
        indices.push(tri.indices[2]);
        indices.push(tri.indices[1]);
        indices.push(tri.indices[0]);
    }

    ConvertedMesh {
        positions,
        normals,
        material_ids,
        indices,
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

    #[test]
    fn winding_order_swapped() {
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
                    position: Vec3::new(0.0, 1.0, 0.0),
                    normal: Vec3::Y,
                    material: Material::Limestone,
                },
            ],
            triangles: vec![Triangle {
                indices: [0, 1, 2],
            }],
        };

        let converted = convert_mesh_to_ue(&mesh, 100.0);
        // Winding should be reversed: [2, 1, 0]
        assert_eq!(converted.indices, vec![2, 1, 0]);
    }
}
