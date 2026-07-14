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
    ///
    /// Face normals are accumulated into POSITION buckets (bit-exact position
    /// match), not vertex indices: seam quads duplicate boundary vertices
    /// instead of indexing into the base mesh, and index-based accumulation
    /// gave the coincident copies different one-sided normals — a visible
    /// lighting crease along every chunk seam. Bucketing by position gives all
    /// coincident copies the same full-ring average.
    ///
    /// Thin sheets and shells (1-cell-thick rock): DC emits ONE welded vertex
    /// layer serving BOTH faces, so a bucket's front/back contributions nearly
    /// cancel — the residual is a small, essentially random vector that
    /// normalizes to a full-strength garbage normal (checkered black/bright
    /// facets under any light; verified in-game 2026-07-14). Such buckets are
    /// SPLIT per facing side: triangles on each side get their own vertex copy
    /// (same position/material) carrying that side's average normal. Vertex
    /// count grows slightly on sheets; positions and triangle count are
    /// untouched.
    pub fn recalculate_normals(&mut self) {
        if self.vertices.is_empty() || self.triangles.is_empty() { return; }

        // A bucket is "two-sided" when its summed normal is much shorter than
        // the total contributed magnitude (opposing faces eat each other).
        // 0.35 keeps sharp creases (~90-120 degrees) smooth-shaded while
        // catching genuine front/back cancellation.
        const CANCEL_RATIO: f32 = 0.35;

        // -0.0 and +0.0 are the same position but different bits; `+ 0.0`
        // canonicalizes -0.0 to +0.0 before taking the bit pattern.
        #[inline]
        fn pos_key(p: Vec3) -> (u32, u32, u32) {
            ((p.x + 0.0).to_bits(), (p.y + 0.0).to_bits(), (p.z + 0.0).to_bits())
        }

        let vert_count = self.vertices.len();
        let mut bucket_of: Vec<u32> = Vec::with_capacity(vert_count);
        let mut bucket_ids: std::collections::HashMap<(u32, u32, u32), u32> =
            std::collections::HashMap::with_capacity(vert_count);
        for v in &self.vertices {
            let next = bucket_ids.len() as u32;
            bucket_of.push(*bucket_ids.entry(pos_key(v.position)).or_insert(next));
        }
        let bucket_count = bucket_ids.len();

        // Per-bucket: a reference direction (first face normal seen), plus
        // accumulators for faces agreeing/opposing it and total magnitude.
        let mut reference = vec![Vec3::ZERO; bucket_count];
        let mut acc_with = vec![Vec3::ZERO; bucket_count];
        let mut acc_against = vec![Vec3::ZERO; bucket_count];
        let mut magnitude = vec![0.0f32; bucket_count];

        let face_normal = |vertices: &[Vertex], tri: &Triangle| {
            let p0 = vertices[tri.indices[0] as usize].position;
            let p1 = vertices[tri.indices[1] as usize].position;
            let p2 = vertices[tri.indices[2] as usize].position;
            (p1 - p0).cross(p2 - p0) // un-normalized = area-weighted
        };

        for tri in &self.triangles {
            let n = face_normal(&self.vertices, tri);
            for &i in &tri.indices {
                let b = bucket_of[i as usize] as usize;
                if reference[b] == Vec3::ZERO {
                    reference[b] = n;
                }
                if n.dot(reference[b]) >= 0.0 {
                    acc_with[b] += n;
                } else {
                    acc_against[b] += n;
                }
                magnitude[b] += n.length();
            }
        }

        // Resolve each bucket: a single blended normal, or a per-side split.
        // split_normals[b] = Some((with_side, against_side)) marks a split.
        let mut single = vec![Vec3::ZERO; bucket_count];
        let mut split: Vec<Option<(Vec3, Vec3)>> = vec![None; bucket_count];
        for b in 0..bucket_count {
            let sum = acc_with[b] + acc_against[b];
            let len = sum.length();
            if magnitude[b] <= 1e-10 {
                continue; // unreferenced — vertices keep their prior normal
            }
            if len > CANCEL_RATIO * magnitude[b] {
                single[b] = sum / len;
            } else {
                let w = acc_with[b];
                let a = acc_against[b];
                let w_n = if w.length() > 1e-10 { w.normalize() } else { Vec3::ZERO };
                let a_n = if a.length() > 1e-10 { a.normalize() } else { Vec3::ZERO };
                split[b] = Some((w_n, a_n));
            }
        }

        // Apply single-normal buckets.
        for (vi, v) in self.vertices.iter_mut().enumerate() {
            let b = bucket_of[vi] as usize;
            if split[b].is_none() && single[b] != Vec3::ZERO {
                v.normal = single[b];
            }
        }

        // Apply splits: each (original vertex, side) pair resolves to one
        // final vertex index — the first side encountered keeps the original
        // index, the other side gets a duplicated vertex.
        let mut side_map: std::collections::HashMap<(u32, bool), u32> =
            std::collections::HashMap::new();
        let tri_count = self.triangles.len();
        for t in 0..tri_count {
            let n = face_normal(&self.vertices, &self.triangles[t]);
            for c in 0..3 {
                let orig = self.triangles[t].indices[c];
                let b = bucket_of[orig as usize] as usize;
                let Some((with_n, against_n)) = split[b] else { continue };
                let is_with = n.dot(reference[b]) >= 0.0;
                let desired = if is_with { with_n } else { against_n };
                if desired == Vec3::ZERO {
                    continue; // degenerate side — leave the prior normal
                }
                let key = (orig, is_with);
                let final_idx = match side_map.get(&key) {
                    Some(&idx) => idx,
                    None => {
                        let idx = if side_map.contains_key(&(orig, !is_with)) {
                            // Other side already claimed the original — duplicate.
                            let dup = self.vertices.len() as u32;
                            let mut v = self.vertices[orig as usize];
                            v.normal = desired;
                            self.vertices.push(v);
                            dup
                        } else {
                            self.vertices[orig as usize].normal = desired;
                            orig
                        };
                        side_map.insert(key, idx);
                        idx
                    }
                };
                self.triangles[t].indices[c] = final_idx;
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
    fn recalc_unifies_coincident_duplicate_vertices() {
        // The seam-append pattern: the second triangle references coincident
        // DUPLICATE copies of the shared edge's vertices instead of indexing
        // the first triangle's. Position-bucketed accumulation must give the
        // duplicates identical full-ring normals (index-based accumulation
        // gave each copy a one-sided normal = lighting crease on chunk seams).
        let mut mesh = Mesh {
            vertices: vec![
                Vertex { position: Vec3::new(0.0, 0.0, 0.0), normal: Vec3::Y, material: Material::Limestone }, // 0: A
                Vertex { position: Vec3::new(1.0, 0.0, 0.0), normal: Vec3::Y, material: Material::Limestone }, // 1: B
                Vertex { position: Vec3::new(0.0, 1.0, 0.0), normal: Vec3::Y, material: Material::Limestone }, // 2: C
                Vertex { position: Vec3::new(1.0, 0.0, 0.0), normal: Vec3::Y, material: Material::Limestone }, // 3: B duplicate
                Vertex { position: Vec3::new(0.0, 1.0, 0.0), normal: Vec3::Y, material: Material::Limestone }, // 4: C duplicate
                Vertex { position: Vec3::new(1.0, 1.0, 1.0), normal: Vec3::Y, material: Material::Limestone }, // 5: D
            ],
            triangles: vec![
                Triangle { indices: [0, 1, 2] }, // A-B-C
                Triangle { indices: [3, 5, 4] }, // B'-D-C' (tilted plane)
            ],
        };
        mesh.recalculate_normals();

        let nb = mesh.vertices[1].normal;
        let nb_dup = mesh.vertices[3].normal;
        let nc = mesh.vertices[2].normal;
        let nc_dup = mesh.vertices[4].normal;
        assert!((nb - nb_dup).length() < 1e-6, "duplicate of B must share B's normal: {nb:?} vs {nb_dup:?}");
        assert!((nc - nc_dup).length() < 1e-6, "duplicate of C must share C's normal: {nc:?} vs {nc_dup:?}");
        // And the shared normal must be a blend of BOTH faces, not either face alone.
        let face1 = Vec3::new(0.0, 0.0, 1.0);
        assert!((nb - face1).length() > 1e-3, "shared normal must include the second face's contribution");
        assert!(nb.length() > 0.9, "shared normal must be unit-ish, got {nb:?}");
    }

    #[test]
    fn recalc_splits_thin_sheet_vertices_per_side() {
        // Thin-sheet pattern: front and back faces reference the SAME welded
        // vertices with opposite winding, so area-weighted face normals cancel.
        // The bucket must SPLIT: each triangle ends up with vertices whose
        // normals match its own facing, and no vertex ships a zero or
        // garbage-residual normal.
        let mut mesh = Mesh {
            vertices: vec![
                Vertex { position: Vec3::new(0.0, 0.0, 0.0), normal: Vec3::X, material: Material::Limestone },
                Vertex { position: Vec3::new(1.0, 0.0, 0.0), normal: Vec3::X, material: Material::Limestone },
                Vertex { position: Vec3::new(0.0, 1.0, 0.0), normal: Vec3::X, material: Material::Limestone },
            ],
            triangles: vec![
                Triangle { indices: [0, 1, 2] },
                Triangle { indices: [0, 2, 1] },
            ],
        };
        mesh.recalculate_normals();

        assert_eq!(mesh.vertex_count(), 6, "each side must get its own vertex copies");
        assert_eq!(mesh.triangle_count(), 2, "triangle count must not change");
        for tri in &mesh.triangles {
            let p0 = mesh.vertices[tri.indices[0] as usize].position;
            let p1 = mesh.vertices[tri.indices[1] as usize].position;
            let p2 = mesh.vertices[tri.indices[2] as usize].position;
            let face_n = (p1 - p0).cross(p2 - p0).normalize();
            for &i in &tri.indices {
                let v = &mesh.vertices[i as usize];
                assert!((v.normal - face_n).length() < 1e-5,
                    "split vertex normal {:?} must match its side's face normal {:?}", v.normal, face_n);
                assert!((v.normal.length() - 1.0).abs() < 1e-5, "must be unit length");
            }
        }
        // Positions must be preserved exactly (duplicates coincide).
        for v in &mesh.vertices {
            assert!(v.position == Vec3::new(0.0, 0.0, 0.0)
                || v.position == Vec3::new(1.0, 0.0, 0.0)
                || v.position == Vec3::new(0.0, 1.0, 0.0));
        }
    }
}
