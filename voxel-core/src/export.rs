use std::io::Write;
use serde::Serialize;
use crate::material::Material;
use crate::mesh::Mesh;

/// Write a mesh to OBJ format
pub fn write_obj<W: Write>(mesh: &Mesh, writer: &mut W) -> std::io::Result<()> {
    for v in &mesh.vertices {
        writeln!(writer, "v {} {} {}", v.position.x, v.position.y, v.position.z)?;
    }
    for v in &mesh.vertices {
        writeln!(writer, "vn {} {} {}", v.normal.x, v.normal.y, v.normal.z)?;
    }
    for tri in &mesh.triangles {
        // OBJ is 1-indexed
        let i0 = tri.indices[0] + 1;
        let i1 = tri.indices[1] + 1;
        let i2 = tri.indices[2] + 1;
        writeln!(writer, "f {}//{} {}//{} {}//{}", i0, i0, i1, i1, i2, i2)?;
    }
    Ok(())
}

/// Write multiple meshes (multi-chunk) to OBJ
pub fn write_multi_obj<W: Write>(meshes: &[Mesh], writer: &mut W) -> std::io::Result<()> {
    let mut vertex_offset = 0u32;
    for (i, mesh) in meshes.iter().enumerate() {
        writeln!(writer, "g chunk_{}", i)?;
        for v in &mesh.vertices {
            writeln!(writer, "v {} {} {}", v.position.x, v.position.y, v.position.z)?;
        }
        for v in &mesh.vertices {
            writeln!(writer, "vn {} {} {}", v.normal.x, v.normal.y, v.normal.z)?;
        }
        for tri in &mesh.triangles {
            let i0 = tri.indices[0] + vertex_offset + 1;
            let i1 = tri.indices[1] + vertex_offset + 1;
            let i2 = tri.indices[2] + vertex_offset + 1;
            writeln!(writer, "f {}//{} {}//{} {}//{}", i0, i0, i1, i1, i2, i2)?;
        }
        vertex_offset += mesh.vertices.len() as u32;
    }
    Ok(())
}

/// JSON mesh format for Three.js BufferGeometry consumption.
#[derive(Debug, Clone, Serialize)]
pub struct JsonMesh {
    pub positions: Vec<f32>,
    pub normals: Vec<f32>,
    pub material_ids: Vec<u8>,
    pub indices: Vec<u32>,
    pub palette: Vec<PaletteEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PaletteEntry {
    pub id: u8,
    pub name: &'static str,
    pub color: String,
}

/// Convert a Mesh to JsonMesh with flat arrays for direct Three.js consumption.
pub fn mesh_to_json(mesh: &Mesh) -> JsonMesh {
    let vert_count = mesh.vertices.len();
    let mut positions = Vec::with_capacity(vert_count * 3);
    let mut normals = Vec::with_capacity(vert_count * 3);
    let mut material_ids = Vec::with_capacity(vert_count);

    for v in &mesh.vertices {
        positions.push(v.position.x);
        positions.push(v.position.y);
        positions.push(v.position.z);
        normals.push(v.normal.x);
        normals.push(v.normal.y);
        normals.push(v.normal.z);
        material_ids.push(v.material as u8);
    }

    let mut indices = Vec::with_capacity(mesh.triangles.len() * 3);
    for tri in &mesh.triangles {
        indices.push(tri.indices[0]);
        indices.push(tri.indices[1]);
        indices.push(tri.indices[2]);
    }

    // Build palette from all solid materials
    let palette: Vec<PaletteEntry> = std::iter::once(Material::Air)
        .chain(Material::all_solid().iter().copied())
        .map(|m| PaletteEntry {
            id: m as u8,
            name: m.display_name(),
            color: format!("#{:06X}", m.color_hex()),
        })
        .collect();

    JsonMesh {
        positions,
        normals,
        material_ids,
        indices,
        palette,
    }
}

/// Convert multiple meshes directly to JsonMesh without building an intermediate combined mesh.
/// Avoids the ~25MB combined mesh copy that happens on every mine.
pub fn mesh_to_json_multi(meshes: &[&Mesh]) -> JsonMesh {
    let total_verts: usize = meshes.iter().map(|m| m.vertices.len()).sum();
    let total_tris: usize = meshes.iter().map(|m| m.triangles.len()).sum();

    let mut positions = Vec::with_capacity(total_verts * 3);
    let mut normals = Vec::with_capacity(total_verts * 3);
    let mut material_ids = Vec::with_capacity(total_verts);
    let mut indices = Vec::with_capacity(total_tris * 3);

    let mut vertex_offset = 0u32;

    for mesh in meshes {
        for v in &mesh.vertices {
            positions.push(v.position.x);
            positions.push(v.position.y);
            positions.push(v.position.z);
            normals.push(v.normal.x);
            normals.push(v.normal.y);
            normals.push(v.normal.z);
            material_ids.push(v.material as u8);
        }
        for tri in &mesh.triangles {
            indices.push(tri.indices[0] + vertex_offset);
            indices.push(tri.indices[1] + vertex_offset);
            indices.push(tri.indices[2] + vertex_offset);
        }
        vertex_offset += mesh.vertices.len() as u32;
    }

    let palette: Vec<PaletteEntry> = std::iter::once(Material::Air)
        .chain(Material::all_solid().iter().copied())
        .map(|m| PaletteEntry {
            id: m as u8,
            name: m.display_name(),
            color: format!("#{:06X}", m.color_hex()),
        })
        .collect();

    JsonMesh {
        positions,
        normals,
        material_ids,
        indices,
        palette,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{Vertex, Triangle};
    use crate::material::Material;
    use glam::Vec3;

    fn make_triangle_mesh() -> Mesh {
        Mesh {
            vertices: vec![
                Vertex { position: Vec3::new(0.0, 0.0, 0.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(1.0, 0.0, 0.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(0.0, 1.0, 0.0), normal: Vec3::Y, material: Material::Limestone },
            ],
            triangles: vec![Triangle { indices: [0, 1, 2] }],
        }
    }

    #[test]
    fn obj_single_mesh_format() {
        let mesh = make_triangle_mesh();
        let mut buf = Vec::new();
        write_obj(&mesh, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        assert!(output.contains("v 0 0 0"));
        assert!(output.contains("v 1 0 0"));
        assert!(output.contains("v 0 1 0"));
        assert!(output.contains("vn 0 1 0"));
        // OBJ is 1-indexed
        assert!(output.contains("f 1//1 2//2 3//3"));
    }

    #[test]
    fn obj_vertex_count_matches() {
        let mesh = make_triangle_mesh();
        let mut buf = Vec::new();
        write_obj(&mesh, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        let v_count = output.lines().filter(|l| l.starts_with("v ")).count();
        let vn_count = output.lines().filter(|l| l.starts_with("vn ")).count();
        let f_count = output.lines().filter(|l| l.starts_with("f ")).count();

        assert_eq!(v_count, 3);
        assert_eq!(vn_count, 3);
        assert_eq!(f_count, 1);
    }

    #[test]
    fn obj_multi_mesh_offsets() {
        let mesh1 = make_triangle_mesh();
        let mesh2 = make_triangle_mesh();
        let mut buf = Vec::new();
        write_multi_obj(&[mesh1, mesh2], &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        // First chunk: f 1//1 2//2 3//3
        // Second chunk: offset by 3, so f 4//4 5//5 6//6
        assert!(output.contains("g chunk_0"));
        assert!(output.contains("g chunk_1"));
        assert!(output.contains("f 1//1 2//2 3//3"));
        assert!(output.contains("f 4//4 5//5 6//6"));
    }

    #[test]
    fn obj_empty_mesh() {
        let mesh = Mesh::new();
        let mut buf = Vec::new();
        write_obj(&mesh, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.is_empty());
    }
}
