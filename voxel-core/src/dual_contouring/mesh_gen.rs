use glam::Vec3;
use crate::mesh::{Mesh, Vertex, Triangle};
use crate::hermite::{HermiteData, FastHashMap};

/// Generate mesh from dual contouring vertices and hermite data.
///
/// For each sign-changing edge in the hermite data, find the 4 cells sharing that edge,
/// emit a quad connecting their DC vertices, then split into two triangles.
///
/// `dc_vertices`: array indexed by cell index (linearized [z][y][x]), containing the DC vertex
/// position for each cell. Cells without a vertex should have a sentinel or be absent.
///
/// `grid_size`: number of cells along each axis (the density grid is grid_size+1 on each axis)
pub fn generate_mesh(hermite: &HermiteData, dc_vertices: &[glam::Vec3], grid_size: usize, max_edge_length: f32, min_triangle_area: f32) -> Mesh {
    let mut mesh = Mesh::new();
    let mut vertex_map: FastHashMap<usize, u32> = FastHashMap::default();

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
            let vi = *vertex_map.entry(cell_idx).or_insert_with(|| {
                let idx = mesh.vertices.len() as u32;
                mesh.vertices.push(Vertex {
                    position: pos,
                    normal: intersection.normal,
                    material: intersection.material,
                });
                idx
            });
            quad_verts[i] = vi;
            valid_mask[i] = true;
            valid_count += 1;
        }

        let max_edge_sq = max_edge_length * max_edge_length;

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

            // Gate 2: skip stretched and thin filters, keep only degenerate check
            #[cfg(feature = "diag-gate-2")]
            let pass_a = !is_degenerate_tri(&mesh.vertices, tri_a);
            #[cfg(not(feature = "diag-gate-2"))]
            let pass_a = !is_degenerate_tri(&mesh.vertices, tri_a) && !is_stretched_tri(&mesh.vertices, tri_a, max_edge_sq) && !is_thin_tri(&mesh.vertices, tri_a, min_triangle_area);
            if pass_a {
                mesh.triangles.push(Triangle { indices: tri_a });
            }

            #[cfg(feature = "diag-gate-2")]
            let pass_b = !is_degenerate_tri(&mesh.vertices, tri_b);
            #[cfg(not(feature = "diag-gate-2"))]
            let pass_b = !is_degenerate_tri(&mesh.vertices, tri_b) && !is_stretched_tri(&mesh.vertices, tri_b, max_edge_sq) && !is_thin_tri(&mesh.vertices, tri_b, min_triangle_area);
            if pass_b {
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

            // Gate 2: skip stretched and thin filters, keep only degenerate check
            #[cfg(feature = "diag-gate-2")]
            let pass_tri = !is_degenerate_tri(&mesh.vertices, tri);
            #[cfg(not(feature = "diag-gate-2"))]
            let pass_tri = !is_degenerate_tri(&mesh.vertices, tri) && !is_stretched_tri(&mesh.vertices, tri, max_edge_sq) && !is_thin_tri(&mesh.vertices, tri, min_triangle_area);
            if pass_tri {
                mesh.triangles.push(Triangle { indices: tri });
            }
        }
        // else: 2 or fewer valid vertices, skip entirely
    }

    // Fallback: if full filter chain removed ALL triangles but vertices exist,
    // retry with degenerate-only filter to rescue some geometry
    if mesh.triangles.is_empty() && !vertex_map.is_empty() {
        for (edge_key, intersection) in hermite.edges.iter() {
            let x = edge_key.x() as usize;
            let y = edge_key.y() as usize;
            let z = edge_key.z() as usize;
            let axis = edge_key.axis() as usize;

            let cell_indices = match axis {
                0 => get_quad_cells_x(x, y, z, grid_size),
                1 => get_quad_cells_y(x, y, z, grid_size),
                2 => get_quad_cells_z(x, y, z, grid_size),
                _ => continue,
            };
            let cell_indices = match cell_indices {
                Some(c) => c,
                None => continue,
            };

            let mut quad_verts = [0u32; 4];
            let mut valid_count = 0u32;
            for (i, &cell_idx) in cell_indices.iter().enumerate() {
                if let Some(&vi) = vertex_map.get(&cell_idx) {
                    quad_verts[i] = vi;
                    valid_count += 1;
                }
            }

            if valid_count == 4 {
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
                let mut tri = [0u32; 3];
                let mut j = 0;
                for i in 0..4 {
                    if vertex_map.contains_key(&cell_indices[i]) {
                        tri[j] = quad_verts[i];
                        j += 1;
                    }
                }
                if j == 3 {
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
            }
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

/// Check if any edge of a triangle exceeds the maximum allowed length squared.
fn is_stretched_tri(vertices: &[Vertex], indices: [u32; 3], max_edge_len_sq: f32) -> bool {
    let v0 = vertices[indices[0] as usize].position;
    let v1 = vertices[indices[1] as usize].position;
    let v2 = vertices[indices[2] as usize].position;
    (v1 - v0).length_squared() > max_edge_len_sq
        || (v2 - v1).length_squared() > max_edge_len_sq
        || (v2 - v0).length_squared() > max_edge_len_sq
}

/// Check if a triangle's area is below the minimum threshold (thin sliver filter).
/// When min_area <= 0.0, this check is disabled (always returns false).
fn is_thin_tri(vertices: &[Vertex], indices: [u32; 3], min_area: f32) -> bool {
    if min_area <= 0.0 {
        return false;
    }
    let v0 = vertices[indices[0] as usize].position;
    let v1 = vertices[indices[1] as usize].position;
    let v2 = vertices[indices[2] as usize].position;
    let area = (v1 - v0).cross(v2 - v0).length() * 0.5;
    area < min_area
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
        let mesh = generate_mesh(&hermite, &dc_vertices, 4, 4.0, 0.0);
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

        let mesh = generate_mesh(&hermite, &dc_vertices, grid_size, 4.0, 0.0);
        assert_eq!(mesh.triangle_count(), 2, "One quad = 2 triangles");
        assert!(mesh.vertex_count() <= 4, "At most 4 vertices for one quad");
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
