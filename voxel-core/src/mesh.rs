use glam::Vec3;
use crate::material::Material;

#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub material: Material,
}

#[derive(Debug, Clone, Copy)]
pub struct Triangle {
    pub indices: [u32; 3],
}

#[derive(Debug, Clone, Default)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub triangles: Vec<Triangle>,
}

impl Mesh {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// Merge another mesh into this one, adjusting triangle indices accordingly
    pub fn merge(&mut self, other: &Mesh) {
        let offset = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&other.vertices);
        for tri in &other.triangles {
            self.triangles.push(Triangle {
                indices: [
                    tri.indices[0] + offset,
                    tri.indices[1] + offset,
                    tri.indices[2] + offset,
                ],
            });
        }
    }

    /// Append another mesh, consuming it
    pub fn append(&mut self, mut other: Mesh) {
        let offset = self.vertices.len() as u32;
        self.vertices.append(&mut other.vertices);
        for tri in &other.triangles {
            self.triangles.push(Triangle {
                indices: [
                    tri.indices[0] + offset,
                    tri.indices[1] + offset,
                    tri.indices[2] + offset,
                ],
            });
        }
    }

    /// Split mesh into per-material submeshes.
    pub fn split_by_material(&self) -> Vec<(u8, Mesh)> {
        use std::collections::BTreeMap;
        let mut buckets: BTreeMap<u8, Vec<usize>> = BTreeMap::new();
        for (i, tri) in self.triangles.iter().enumerate() {
            let mat = self.vertices[tri.indices[0] as usize].material as u8;
            buckets.entry(mat).or_default().push(i);
        }

        buckets.into_iter().map(|(mat, tri_indices)| {
            let mut remap = std::collections::HashMap::new();
            let mut verts = Vec::new();
            let mut tris = Vec::new();
            for tri_idx in tri_indices {
                let orig_tri = &self.triangles[tri_idx];
                let mut new_indices = [0u32; 3];
                for (c, &orig_idx) in orig_tri.indices.iter().enumerate() {
                    let new_idx = *remap.entry(orig_idx).or_insert_with(|| {
                        let idx = verts.len() as u32;
                        verts.push(self.vertices[orig_idx as usize]);
                        idx
                    });
                    new_indices[c] = new_idx;
                }
                tris.push(Triangle { indices: new_indices });
            }
            (mat, Mesh { vertices: verts, triangles: tris })
        }).collect()
    }

    /// Check for degenerate triangles (zero area)
    pub fn has_degenerate_triangles(&self) -> bool {
        for tri in &self.triangles {
            let v0 = self.vertices[tri.indices[0] as usize].position;
            let v1 = self.vertices[tri.indices[1] as usize].position;
            let v2 = self.vertices[tri.indices[2] as usize].position;
            let cross = (v1 - v0).cross(v2 - v0);
            if cross.length_squared() < 1e-12 {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tri_mesh() -> Mesh {
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
    fn merge_adjusts_indices() {
        let mut a = make_tri_mesh();
        let b = make_tri_mesh();
        a.merge(&b);

        assert_eq!(a.vertex_count(), 6);
        assert_eq!(a.triangle_count(), 2);
        // Second triangle indices should be offset by 3
        assert_eq!(a.triangles[1].indices, [3, 4, 5]);
    }

    #[test]
    fn append_adjusts_indices() {
        let mut a = make_tri_mesh();
        let b = make_tri_mesh();
        a.append(b);

        assert_eq!(a.vertex_count(), 6);
        assert_eq!(a.triangle_count(), 2);
        assert_eq!(a.triangles[1].indices, [3, 4, 5]);
    }

    #[test]
    fn merge_empty_into_mesh() {
        let mut a = make_tri_mesh();
        let empty = Mesh::new();
        a.merge(&empty);
        assert_eq!(a.vertex_count(), 3);
        assert_eq!(a.triangle_count(), 1);
    }

    #[test]
    fn merge_into_empty() {
        let mut empty = Mesh::new();
        let b = make_tri_mesh();
        empty.merge(&b);
        assert_eq!(empty.vertex_count(), 3);
        assert_eq!(empty.triangle_count(), 1);
        assert_eq!(empty.triangles[0].indices, [0, 1, 2]);
    }

    #[test]
    fn degenerate_detection() {
        let degenerate = Mesh {
            vertices: vec![
                Vertex { position: Vec3::ZERO, normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::ZERO, normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::ZERO, normal: Vec3::Y, material: Material::Limestone },
            ],
            triangles: vec![Triangle { indices: [0, 1, 2] }],
        };
        assert!(degenerate.has_degenerate_triangles());

        let valid = make_tri_mesh();
        assert!(!valid.has_degenerate_triangles());
    }
}
