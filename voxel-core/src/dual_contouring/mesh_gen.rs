use glam::Vec3;
use crate::mesh::{Mesh, Vertex, Triangle};
use crate::hermite::HermiteData;

/// Generate mesh from dual contouring vertices and hermite data.
///
/// For each sign-changing edge in the hermite data, find the 4 cells sharing that edge,
/// emit a quad connecting their DC vertices, then split into two triangles.
///
/// `dc_vertices`: array indexed by cell index (linearized [z][y][x]), containing the DC vertex
/// position for each cell. Cells without a vertex should have a sentinel or be absent.
///
/// `grid_size`: number of cells along each axis (the density grid is grid_size+1 on each axis)
pub fn generate_mesh(hermite: &HermiteData, dc_vertices: &[glam::Vec3], grid_size: usize) -> Mesh {
    let mut mesh = Mesh::new();
    // Per-cell vertex dedup. `cell_idx` is a DENSE linearized cell index in
    // `0..dc_vertices.len()` (== grid_size^3 from `solve_dc_vertices`), so a flat sentinel
    // array is the right structure — was a `FastHashMap<usize,u32>` that hash-probed once
    // per quad corner (~4 * edge_count probes/chunk, plus per-call alloc + growth/rehash)
    // over a dense key. `u32::MAX` = "no vertex assigned yet for this cell"; a real index
    // can never reach u32::MAX (would need 4e9 verts/chunk). One alloc, raw array indexing.
    let mut vertex_map = vec![u32::MAX; dc_vertices.len()];
    // Per-vertex hermite-normal accumulator, indexed like mesh.vertices. Every
    // sign-changing edge that touches a cell contributes its normal and the
    // average is applied after the loop — the old behavior kept whichever edge
    // happened to come FIRST in hash-map iteration order (an arbitrary pick
    // among up to 12 candidates, visibly patchy on curved walls when
    // mesh_recalc_normals is off).
    let mut normal_accum: Vec<Vec3> = Vec::new();

    for (edge_key, intersection) in hermite.edges.iter() {
        let x = edge_key.x() as usize;
        let y = edge_key.y() as usize;
        let z = edge_key.z() as usize;
        let axis = edge_key.axis() as usize;

        // Find the 4 cells that share this edge
        // An edge along axis X at (x,y,z) is shared by cells:
        //   (x, y, z), (x, y-1, z), (x, y, z-1), (x, y-1, z-1) for axis 0 (x-axis)
        // etc.
        let cell_indices = match axis {
            0 => {
                // X-axis edge at (x, y, z): shared by cells
                // (x, y, z), (x, y-1, z), (x, y, z-1), (x, y-1, z-1)
                get_quad_cells_x(x, y, z, grid_size)
            }
            1 => {
                // Y-axis edge at (x, y, z): shared by cells
                // (x, y, z), (x-1, y, z), (x, y, z-1), (x-1, y, z-1)
                get_quad_cells_y(x, y, z, grid_size)
            }
            2 => {
                // Z-axis edge at (x, y, z): shared by cells
                // (x, y, z), (x-1, y, z), (x, y-1, z), (x-1, y-1, z)
                get_quad_cells_z(x, y, z, grid_size)
            }
            _ => continue,
        };

        let cell_indices = match cell_indices {
            Some(c) => c,
            None => continue,
        };

        // Get or create vertex indices for each of the 4 cells.
        // Track which cells are valid (have a DC vertex).
        let mut quad_verts = [0u32; 4];
        let mut valid_mask = [false; 4];
        let mut valid_count = 0u32;
        for (i, &cell_idx) in cell_indices.iter().enumerate() {
            if cell_idx >= dc_vertices.len() {
                continue;
            }
            let pos = dc_vertices[cell_idx];
            if pos.x.is_nan() {
                continue;
            }
            let mut vi = vertex_map[cell_idx];
            if vi == u32::MAX {
                vi = mesh.vertices.len() as u32;
                mesh.vertices.push(Vertex {
                    position: pos,
                    normal: intersection.normal,
                    material: intersection.material,
                });
                normal_accum.push(Vec3::ZERO);
                vertex_map[cell_idx] = vi;
            }
            normal_accum[vi as usize] += intersection.normal;
            quad_verts[i] = vi;
            valid_mask[i] = true;
            valid_count += 1;
        }

        if valid_count == 4 {
            // Full quad: split into two triangles
            let normal_dot = intersection.normal.dot(axis_direction(axis));
            let (tri_a, tri_b) = if normal_dot > 0.0 {
                ([quad_verts[0], quad_verts[1], quad_verts[2]],
                 [quad_verts[0], quad_verts[2], quad_verts[3]])
            } else {
                ([quad_verts[2], quad_verts[1], quad_verts[0]],
                 [quad_verts[3], quad_verts[2], quad_verts[0]])
            };

            if !is_degenerate_tri(&mesh.vertices, tri_a) {
                mesh.triangles.push(Triangle { indices: tri_a });
            }

            if !is_degenerate_tri(&mesh.vertices, tri_b) {
                mesh.triangles.push(Triangle { indices: tri_b });
            }
        } else if valid_count == 3 {
            // Partial quad: one cell is fully air (no DC vertex).
            // Emit a single triangle from the 3 valid vertices to fill the gap.
            let mut tri = [0u32; 3];
            let mut j = 0;
            for i in 0..4 {
                if valid_mask[i] {
                    tri[j] = quad_verts[i];
                    j += 1;
                }
            }

            // Ensure correct winding: face normal should agree with intersection normal
            let v0 = mesh.vertices[tri[0] as usize].position;
            let v1 = mesh.vertices[tri[1] as usize].position;
            let v2 = mesh.vertices[tri[2] as usize].position;
            let face_normal = (v1 - v0).cross(v2 - v0);
            if face_normal.dot(intersection.normal) < 0.0 {
                tri.swap(1, 2);
            }

            if !is_degenerate_tri(&mesh.vertices, tri) {
                mesh.triangles.push(Triangle { indices: tri });
            }
        }
        // else: 2 or fewer valid vertices, skip entirely
    }

    // Apply the averaged hermite normals. A cancelled sum (thin sheet: the
    // cell is crossed from both sides by opposing edges) keeps the first
    // edge's normal — any consistent nonzero side is fine; the far side is
    // lit via the two-sided material's facing flip.
    for (v, accum) in mesh.vertices.iter_mut().zip(&normal_accum) {
        let len = accum.length();
        if len > 1e-6 {
            v.normal = *accum / len;
        }
    }

    mesh
}

fn axis_direction(axis: usize) -> Vec3 {
    match axis {
        0 => Vec3::X,
        1 => Vec3::Y,
        2 => Vec3::Z,
        _ => Vec3::ZERO,
    }
}

#[inline]
fn cell_index(x: usize, y: usize, z: usize, size: usize) -> usize {
    z * size * size + y * size + x
}

fn get_quad_cells_x(x: usize, y: usize, z: usize, size: usize) -> Option<[usize; 4]> {
    // Cells are indexed 0..size-1; edges at y=size or z=size lack valid adjacent cells
    if y == 0 || z == 0 || y >= size || z >= size || x >= size {
        return None;
    }
    Some([
        cell_index(x, y - 1, z - 1, size),
        cell_index(x, y, z - 1, size),
        cell_index(x, y, z, size),
        cell_index(x, y - 1, z, size),
    ])
}

fn get_quad_cells_y(x: usize, y: usize, z: usize, size: usize) -> Option<[usize; 4]> {
    if x == 0 || z == 0 || x >= size || z >= size || y >= size {
        return None;
    }
    Some([
        cell_index(x - 1, y, z - 1, size),
        cell_index(x, y, z - 1, size),
        cell_index(x, y, z, size),
        cell_index(x - 1, y, z, size),
    ])
}

fn get_quad_cells_z(x: usize, y: usize, z: usize, size: usize) -> Option<[usize; 4]> {
    if x == 0 || y == 0 || x >= size || y >= size || z >= size {
        return None;
    }
    Some([
        cell_index(x - 1, y - 1, z, size),
        cell_index(x, y - 1, z, size),
        cell_index(x, y, z, size),
        cell_index(x - 1, y, z, size),
    ])
}

/// Check if a triangle is degenerate (zero or near-zero area)
fn is_degenerate_tri(vertices: &[Vertex], indices: [u32; 3]) -> bool {
    let v0 = vertices[indices[0] as usize].position;
    let v1 = vertices[indices[1] as usize].position;
    let v2 = vertices[indices[2] as usize].position;
    let cross = (v1 - v0).cross(v2 - v0);
    cross.length_squared() < 1e-10
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::hermite::{EdgeIntersection, HermiteData, EdgeKey};
    use crate::material::Material;

    #[test]
    fn empty_hermite_empty_mesh() {
        let hermite = HermiteData::default();
        let dc_vertices = vec![];
        let mesh = generate_mesh(&hermite, &dc_vertices, 4);
        assert!(mesh.is_empty());
    }

    #[test]
    fn single_edge_produces_triangles() {
        let grid_size = 4;
        let mut hermite = HermiteData::default();

        // Add a Z-axis edge at (1, 1, 1) — shared by cells (0,0,1), (1,0,1), (1,1,1), (0,1,1)
        let key = EdgeKey::new(1, 1, 1, 2);
        hermite.edges.insert(key, EdgeIntersection {
            t: 0.5,
            normal: Vec3::Z,
            material: Material::Limestone,
        });

        // Create DC vertices for all cells in the grid
        let total_cells = grid_size * grid_size * grid_size;
        let mut dc_vertices = vec![Vec3::ZERO; total_cells];
        // Place DC vertices at cell centers
        for z in 0..grid_size {
            for y in 0..grid_size {
                for x in 0..grid_size {
                    let idx = cell_index(x, y, z, grid_size);
                    dc_vertices[idx] = Vec3::new(
                        x as f32 + 0.5,
                        y as f32 + 0.5,
                        z as f32 + 0.5,
                    );
                }
            }
        }

        let mesh = generate_mesh(&hermite, &dc_vertices, grid_size);
        assert_eq!(mesh.triangle_count(), 2, "One quad = 2 triangles");
        assert!(mesh.vertex_count() <= 4, "At most 4 vertices for one quad");
    }

    #[test]
    fn dedup_shares_vertices_across_edges() {
        // Two Z-axis edges whose quads overlap in two cells. The flat-array vertex map
        // must collapse the 8 corner references (2 edges x 4 corners) to the 6 DISTINCT
        // cells — i.e. a cell touched by both edges yields exactly one shared vertex.
        let grid_size = 4;
        let mut hermite = HermiteData::default();
        for &ex in &[1u8, 2u8] {
            hermite.edges.insert(EdgeKey::new(ex, 1, 1, 2), EdgeIntersection {
                t: 0.5,
                normal: Vec3::Z,
                material: Material::Limestone,
            });
        }
        // All cells valid (non-NaN) so every referenced corner produces a vertex.
        let total = grid_size * grid_size * grid_size;
        let mut dc_vertices = vec![Vec3::ZERO; total];
        for z in 0..grid_size { for y in 0..grid_size { for x in 0..grid_size {
            dc_vertices[cell_index(x, y, z, grid_size)] =
                Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
        }}}

        let mesh = generate_mesh(&hermite, &dc_vertices, grid_size);

        // Edge (1,1,1) cells: (0,0,1)(1,0,1)(1,1,1)(0,1,1); edge (2,1,1) cells:
        // (1,0,1)(2,0,1)(2,1,1)(1,1,1). Shared: (1,0,1),(1,1,1) -> 6 distinct cells.
        assert_eq!(mesh.vertex_count(), 6,
            "8 corner refs across 2 edges must dedup to 6 distinct cells, got {}", mesh.vertex_count());
        // Every triangle index is in-range (no dangling/duplicate vertex).
        for tri in &mesh.triangles {
            for &i in &tri.indices {
                assert!((i as usize) < mesh.vertex_count(), "index {i} out of range");
            }
        }
    }

    #[test]
    fn cell_index_correct() {
        assert_eq!(cell_index(0, 0, 0, 4), 0);
        assert_eq!(cell_index(1, 0, 0, 4), 1);
        assert_eq!(cell_index(0, 1, 0, 4), 4);
        assert_eq!(cell_index(0, 0, 1, 4), 16);
        assert_eq!(cell_index(3, 3, 3, 4), 63);
    }
}
