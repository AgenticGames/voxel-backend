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

    /// Weld vertices at the same position into a single vertex.
    /// Duplicate vertices within `epsilon` distance are merged so that
    /// `recalculate_normals()` accumulates contributions from ALL adjacent
    /// triangles, eliminating hard lighting edges at chunk seams.
    /// The first vertex encountered at each position wins (base mesh vertices
    /// appear before appended seam vertices, preserving correct material).
    pub fn weld_vertices(&mut self, epsilon: f32) {
        if self.vertices.is_empty() { return; }

        use std::collections::HashMap;

        let inv = 1.0 / epsilon;

        // Quantize positions and find canonical index for each vertex
        let mut canonical: HashMap<(i64, i64, i64), u32> = HashMap::new();
        let mut remap: Vec<u32> = Vec::with_capacity(self.vertices.len());

        for (i, v) in self.vertices.iter().enumerate() {
            let key = (
                (v.position.x * inv).round() as i64,
                (v.position.y * inv).round() as i64,
                (v.position.z * inv).round() as i64,
            );
            let canon = *canonical.entry(key).or_insert(i as u32);
            remap.push(canon);
        }

        // Remap triangle indices and remove degenerate triangles
        let mut new_triangles = Vec::with_capacity(self.triangles.len());
        for tri in &self.triangles {
            let i0 = remap[tri.indices[0] as usize];
            let i1 = remap[tri.indices[1] as usize];
            let i2 = remap[tri.indices[2] as usize];
            if i0 != i1 && i1 != i2 && i2 != i0 {
                new_triangles.push(Triangle { indices: [i0, i1, i2] });
            }
        }

        // Compact: find which vertices are still referenced
        let mut referenced = vec![false; self.vertices.len()];
        for tri in &new_triangles {
            for &idx in &tri.indices {
                referenced[idx as usize] = true;
            }
        }

        // Build compaction map: old index → new index
        let mut compact_map: Vec<u32> = vec![0; self.vertices.len()];
        let mut new_vertices = Vec::new();
        for (old_idx, &is_ref) in referenced.iter().enumerate() {
            if is_ref {
                compact_map[old_idx] = new_vertices.len() as u32;
                new_vertices.push(self.vertices[old_idx]);
            }
        }

        // Remap triangle indices through compaction
        for tri in &mut new_triangles {
            for idx in &mut tri.indices {
                *idx = compact_map[*idx as usize];
            }
        }

        self.vertices = new_vertices;
        self.triangles = new_triangles;
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

    /// Laplacian smoothing: iteratively blend vertices toward neighbor average.
    /// Material-boundary vertices use reduced strength to preserve ore outlines.
    /// When `chunk_cell_size` is set, vertices in boundary cells (any coord < 1.0
    /// or > cell_size - 1) are pinned so they match seam quad positions exactly.
    pub fn smooth(&mut self, iterations: u32, strength: f32, boundary_smooth: f32, chunk_cell_size: Option<usize>) {
        if iterations == 0 || self.vertices.is_empty() { return; }

        let vert_count = self.vertices.len();
        let tri_count = self.triangles.len();

        // Build adjacency: for each vertex, collect unique neighbor vertex indices
        let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); vert_count];
        for t in 0..tri_count {
            let i0 = self.triangles[t].indices[0] as usize;
            let i1 = self.triangles[t].indices[1] as usize;
            let i2 = self.triangles[t].indices[2] as usize;
            for &(a, b) in &[(i0, i1), (i1, i2), (i2, i0)] {
                if !adjacency[a].contains(&(b as u32)) { adjacency[a].push(b as u32); }
                if !adjacency[b].contains(&(a as u32)) { adjacency[b].push(a as u32); }
            }
        }

        // Identify material-boundary vertices (shared by triangles with different materials)
        let mut is_boundary = vec![false; vert_count];
        let mut vert_materials: Vec<Vec<u8>> = vec![Vec::new(); vert_count];
        for t in 0..tri_count {
            let mat = self.vertices[self.triangles[t].indices[0] as usize].material as u8;
            for &vi in &self.triangles[t].indices {
                let vi = vi as usize;
                if !vert_materials[vi].contains(&mat) {
                    vert_materials[vi].push(mat);
                }
            }
        }
        for vi in 0..vert_count {
            is_boundary[vi] = vert_materials[vi].len() > 1;
        }

        // Identify chunk-edge vertices to pin (skip during smoothing)
        let is_chunk_edge: Vec<bool> = if let Some(cell_size) = chunk_cell_size {
            let lo = 1.0_f32;
            let hi = (cell_size - 1) as f32;
            self.vertices.iter().map(|v| {
                let p = v.position;
                p.x < lo || p.y < lo || p.z < lo || p.x > hi || p.y > hi || p.z > hi
            }).collect()
        } else {
            vec![false; vert_count]
        };

        // Iterative smoothing
        for _ in 0..iterations {
            let old_positions: Vec<Vec3> = self.vertices.iter().map(|v| v.position).collect();
            for vi in 0..vert_count {
                if is_chunk_edge[vi] { continue; }

                let neighbors = &adjacency[vi];
                if neighbors.is_empty() { continue; }

                let pos = old_positions[vi];
                let mut avg = Vec3::ZERO;
                for &ni in neighbors {
                    avg += old_positions[ni as usize];
                }
                avg /= neighbors.len() as f32;

                let s = if is_boundary[vi] { boundary_smooth } else { strength };
                self.vertices[vi].position = pos + (avg - pos) * s;
            }
        }
    }

    /// Recalculate area-weighted vertex normals from triangle geometry.
    pub fn recalculate_normals(&mut self) {
        if self.vertices.is_empty() || self.triangles.is_empty() { return; }

        // Zero all normals
        for v in &mut self.vertices {
            v.normal = Vec3::ZERO;
        }

        for tri in &self.triangles {
            let i0 = tri.indices[0] as usize;
            let i1 = tri.indices[1] as usize;
            let i2 = tri.indices[2] as usize;

            let p0 = self.vertices[i0].position;
            let p1 = self.vertices[i1].position;
            let p2 = self.vertices[i2].position;

            // Cross product (un-normalized = area-weighted)
            let normal = (p1 - p0).cross(p2 - p0);

            self.vertices[i0].normal += normal;
            self.vertices[i1].normal += normal;
            self.vertices[i2].normal += normal;
        }

        // Normalize
        for v in &mut self.vertices {
            let len = v.normal.length();
            if len > 1e-10 {
                v.normal /= len;
            }
        }
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

    #[test]
    fn smooth_zero_iterations_noop() {
        let mut mesh = make_tri_mesh();
        let orig: Vec<Vec3> = mesh.vertices.iter().map(|v| v.position).collect();
        mesh.smooth(0, 0.5, 0.3, None);
        for (i, v) in mesh.vertices.iter().enumerate() {
            assert_eq!(v.position, orig[i]);
        }
    }

    #[test]
    fn smooth_moves_vertices() {
        // Two triangles sharing an edge — smoothing should move interior vertex
        let mut mesh = Mesh {
            vertices: vec![
                Vertex { position: Vec3::new(0.0, 0.0, 0.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(2.0, 0.0, 0.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(1.0, 2.0, 0.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(1.0, -2.0, 0.0), normal: Vec3::Y, material: Material::Limestone },
            ],
            triangles: vec![
                Triangle { indices: [0, 1, 2] },
                Triangle { indices: [0, 3, 1] },
            ],
        };
        let orig_pos = mesh.vertices[2].position;
        mesh.smooth(1, 0.5, 0.3, None);
        // Vertex 2 should have moved toward its neighbors
        assert_ne!(mesh.vertices[2].position, orig_pos);
    }

    #[test]
    fn recalculate_normals_produces_unit_normals() {
        let mut mesh = make_tri_mesh();
        // Zero out normals
        for v in &mut mesh.vertices {
            v.normal = Vec3::ZERO;
        }
        mesh.recalculate_normals();
        for v in &mesh.vertices {
            let len = v.normal.length();
            assert!((len - 1.0).abs() < 1e-5, "Normal length should be ~1.0, got {len}");
        }
    }

    #[test]
    fn weld_vertices_merges_duplicates() {
        // Two triangles with 6 verts, but 2 positions are shared (indices 1==3, 2==4)
        let mut mesh = Mesh {
            vertices: vec![
                Vertex { position: Vec3::new(0.0, 0.0, 0.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(1.0, 0.0, 0.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(0.0, 1.0, 0.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(1.0, 0.0, 0.0), normal: Vec3::Y, material: Material::Granite }, // dup of 1
                Vertex { position: Vec3::new(0.0, 1.0, 0.0), normal: Vec3::Y, material: Material::Granite }, // dup of 2
                Vertex { position: Vec3::new(1.0, 1.0, 0.0), normal: Vec3::Y, material: Material::Granite },
            ],
            triangles: vec![
                Triangle { indices: [0, 1, 2] },
                Triangle { indices: [3, 4, 5] },
            ],
        };
        mesh.weld_vertices(1e-4);
        assert_eq!(mesh.vertex_count(), 4, "Should merge 2 duplicate pairs into 4 unique verts");
        assert_eq!(mesh.triangle_count(), 2, "Both triangles should survive (non-degenerate)");
    }

    #[test]
    fn weld_vertices_removes_degenerate_triangles() {
        // Triangle where two verts share the same position — becomes degenerate after weld
        let mut mesh = Mesh {
            vertices: vec![
                Vertex { position: Vec3::new(0.0, 0.0, 0.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(1.0, 0.0, 0.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(0.0, 0.0, 0.0), normal: Vec3::Y, material: Material::Limestone }, // dup of 0
            ],
            triangles: vec![Triangle { indices: [0, 1, 2] }],
        };
        mesh.weld_vertices(1e-4);
        assert_eq!(mesh.triangle_count(), 0, "Degenerate triangle should be removed");
    }

    #[test]
    fn weld_vertices_noop_no_duplicates() {
        let mut mesh = make_tri_mesh();
        let orig_vc = mesh.vertex_count();
        let orig_tc = mesh.triangle_count();
        mesh.weld_vertices(1e-4);
        assert_eq!(mesh.vertex_count(), orig_vc);
        assert_eq!(mesh.triangle_count(), orig_tc);
    }

    #[test]
    fn weld_vertices_empty_mesh() {
        let mut mesh = Mesh::new();
        mesh.weld_vertices(1e-4);
        assert_eq!(mesh.vertex_count(), 0);
        assert_eq!(mesh.triangle_count(), 0);
    }

    #[test]
    fn weld_after_append_produces_smooth_normals() {
        // Simulate base mesh + seam mesh sharing an edge.
        // Triangle A in XY plane, Triangle B angled into Z — shared edge at x=1.
        let base = Mesh {
            vertices: vec![
                Vertex { position: Vec3::new(0.0, 0.0, 0.0), normal: Vec3::ZERO, material: Material::Limestone },
                Vertex { position: Vec3::new(1.0, 0.0, 0.0), normal: Vec3::ZERO, material: Material::Limestone },
                Vertex { position: Vec3::new(1.0, 1.0, 0.0), normal: Vec3::ZERO, material: Material::Limestone },
            ],
            triangles: vec![Triangle { indices: [0, 1, 2] }],
        };
        let seam = Mesh {
            vertices: vec![
                Vertex { position: Vec3::new(1.0, 0.0, 0.0), normal: Vec3::ZERO, material: Material::Limestone },
                Vertex { position: Vec3::new(2.0, 0.0, 1.0), normal: Vec3::ZERO, material: Material::Limestone },
                Vertex { position: Vec3::new(1.0, 1.0, 0.0), normal: Vec3::ZERO, material: Material::Limestone },
            ],
            triangles: vec![Triangle { indices: [0, 1, 2] }],
        };
        let mut mesh = base;
        mesh.append(seam);
        assert_eq!(mesh.vertex_count(), 6);

        mesh.weld_vertices(1e-4);
        assert_eq!(mesh.vertex_count(), 4, "Two shared-edge verts should merge");

        mesh.recalculate_normals();
        // The shared vertices (at (1,0,0) and (1,1,0)) should have normals that
        // are NOT purely along Z — they should blend contributions from both triangles.
        for v in &mesh.vertices {
            if (v.position - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-3
                || (v.position - Vec3::new(1.0, 1.0, 0.0)).length() < 1e-3
            {
                assert!(v.normal.z.abs() < 0.99,
                    "Shared vertex normal should blend both triangles, got {:?}", v.normal);
            }
        }
    }
}
