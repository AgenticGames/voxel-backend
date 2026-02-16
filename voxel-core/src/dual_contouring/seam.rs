use std::collections::HashMap;
use glam::Vec3;
use crate::mesh::{Mesh, Vertex, Triangle};
use crate::hermite::{EdgeKey, EdgeIntersection, HermiteData};

/// Which chunk face (for boundary data extraction).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkFace {
    NegX, PosX,
    NegY, PosY,
    NegZ, PosZ,
}

/// Boundary data for one face of a chunk, used for cross-chunk seam stitching.
/// Stores the DC vertices from cells touching the face, and hermite edges
/// on/near the face that were skipped by single-chunk mesh generation.
#[derive(Debug, Clone, Default)]
pub struct ChunkBoundaryData {
    /// DC vertex positions at the boundary, keyed by local cell (x,y,z)
    pub boundary_vertices: HashMap<(u8, u8, u8), Vec3>,
    /// Edge intersections at the boundary, keyed by EdgeKey
    pub boundary_edges: HashMap<EdgeKey, EdgeIntersection>,
    /// Normal for each boundary vertex
    pub boundary_normals: HashMap<(u8, u8, u8), Vec3>,
}

impl ChunkBoundaryData {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a boundary vertex
    pub fn add_vertex(&mut self, pos: (u8, u8, u8), vertex: Vec3, normal: Vec3) {
        self.boundary_vertices.insert(pos, vertex);
        self.boundary_normals.insert(pos, normal);
    }

    /// Add a boundary edge intersection
    pub fn add_edge(&mut self, key: EdgeKey, intersection: EdgeIntersection) {
        self.boundary_edges.insert(key, intersection);
    }
}

/// Extract boundary data for a specific face of a chunk.
///
/// `hermite`: the chunk's hermite data
/// `dc_vertices`: the chunk's DC vertices (grid_size^3, indexed [z*gs^2 + y*gs + x])
/// `grid_size`: number of cells along each axis (e.g., 16)
/// `face`: which face to extract data for
///
/// For the PosX face, collects:
/// - DC vertices in the last cell layer (x = grid_size-1)
/// - Hermite edges that lie on the face and were skipped by mesh_gen
///   (Y-edges and Z-edges at x = grid_size, plus X-edges whose adjacent cells
///   extend past x = grid_size-1)
pub fn extract_boundary_data(
    hermite: &HermiteData,
    dc_vertices: &[Vec3],
    grid_size: usize,
    face: ChunkFace,
) -> ChunkBoundaryData {
    let mut data = ChunkBoundaryData::new();
    let gs = grid_size;

    // Collect DC vertices from the boundary cell layer
    match face {
        ChunkFace::NegX => {
            // x=0 layer
            for z in 0..gs { for y in 0..gs {
                let idx = z * gs * gs + y * gs;
                let v = dc_vertices[idx];
                if !v.x.is_nan() {
                    data.add_vertex((0, y as u8, z as u8), v, Vec3::X);
                }
            }}
        }
        ChunkFace::PosX => {
            // x=grid_size-1 layer
            let x = gs - 1;
            for z in 0..gs { for y in 0..gs {
                let idx = z * gs * gs + y * gs + x;
                let v = dc_vertices[idx];
                if !v.x.is_nan() {
                    data.add_vertex((x as u8, y as u8, z as u8), v, Vec3::NEG_X);
                }
            }}
        }
        ChunkFace::NegY => {
            for z in 0..gs { for x in 0..gs {
                let idx = z * gs * gs + x;
                let v = dc_vertices[idx];
                if !v.x.is_nan() {
                    data.add_vertex((x as u8, 0, z as u8), v, Vec3::Y);
                }
            }}
        }
        ChunkFace::PosY => {
            let y = gs - 1;
            for z in 0..gs { for x in 0..gs {
                let idx = z * gs * gs + y * gs + x;
                let v = dc_vertices[idx];
                if !v.x.is_nan() {
                    data.add_vertex((x as u8, y as u8, z as u8), v, Vec3::NEG_Y);
                }
            }}
        }
        ChunkFace::NegZ => {
            for y in 0..gs { for x in 0..gs {
                let idx = y * gs + x;
                let v = dc_vertices[idx];
                if !v.x.is_nan() {
                    data.add_vertex((x as u8, y as u8, 0), v, Vec3::Z);
                }
            }}
        }
        ChunkFace::PosZ => {
            let z = gs - 1;
            for y in 0..gs { for x in 0..gs {
                let idx = z * gs * gs + y * gs + x;
                let v = dc_vertices[idx];
                if !v.x.is_nan() {
                    data.add_vertex((x as u8, y as u8, z as u8), v, Vec3::NEG_Z);
                }
            }}
        }
    }

    // Collect hermite edges on/near the boundary that were skipped by mesh_gen.
    // These are edges where at least one of the 4 adjacent cells falls outside the grid.
    for (edge_key, intersection) in hermite.iter() {
        let x = edge_key.x() as usize;
        let y = edge_key.y() as usize;
        let z = edge_key.z() as usize;
        let axis = edge_key.axis();

        let is_boundary = match face {
            ChunkFace::NegX => is_boundary_edge_neg(axis, x, y, z, gs, 0),
            ChunkFace::PosX => is_boundary_edge_pos(axis, x, y, z, gs, 0),
            ChunkFace::NegY => is_boundary_edge_neg(axis, x, y, z, gs, 1),
            ChunkFace::PosY => is_boundary_edge_pos(axis, x, y, z, gs, 1),
            ChunkFace::NegZ => is_boundary_edge_neg(axis, x, y, z, gs, 2),
            ChunkFace::PosZ => is_boundary_edge_pos(axis, x, y, z, gs, 2),
        };

        if is_boundary {
            data.add_edge(edge_key, intersection.clone());
        }
    }

    data
}

/// Check if an edge at (x,y,z) with given axis is a boundary edge for the negative face
/// along `face_axis`. A boundary edge is one that mesh_gen skips because some adjacent
/// cells fall at coordinate -1 (< 0) on the face_axis.
fn is_boundary_edge_neg(axis: u8, x: usize, y: usize, z: usize, _gs: usize, face_axis: u8) -> bool {
    // The adjacent cells for each edge type have offsets on the two perpendicular axes.
    // For an edge along axis A at (x,y,z), the perpendicular axes depend on A.
    // The adjacent cells have offsets (0,0), (-1,0), (0,-1), (-1,-1) on the two perp axes.
    // A boundary edge for NegX face: face_axis=0, means some cell has x-component = -1.
    // This happens when the edge's perpendicular axes include face_axis AND the edge
    // position on that face_axis is 0.

    match axis {
        0 => {
            // X-edge: perp axes are Y and Z. Cell offsets on (y, z).
            // Cells: (x, y+dy, z+dz) for dy,dz in {0,-1}
            // Boundary for NegY (face_axis=1): y==0
            // Boundary for NegZ (face_axis=2): z==0
            match face_axis {
                1 => y == 0,
                2 => z == 0,
                _ => false, // X-edge has no cell offset along X
            }
        }
        1 => {
            // Y-edge: perp axes are X and Z. Cell offsets on (x, z).
            match face_axis {
                0 => x == 0,
                2 => z == 0,
                _ => false,
            }
        }
        2 => {
            // Z-edge: perp axes are X and Y. Cell offsets on (x, y).
            match face_axis {
                0 => x == 0,
                1 => y == 0,
                _ => false,
            }
        }
        _ => false,
    }
}

/// Check if an edge is a boundary edge for the positive face along `face_axis`.
/// This happens when some adjacent cell would be at coordinate >= gs (outside the grid).
fn is_boundary_edge_pos(axis: u8, x: usize, y: usize, z: usize, gs: usize, face_axis: u8) -> bool {
    match axis {
        0 => {
            // X-edge: perp axes Y, Z. Cells at y, y-1 and z, z-1.
            // Max cell coord is y (or z) when offset is 0.
            // Outside when y >= gs or z >= gs.
            match face_axis {
                0 => x >= gs, // X-edge at x >= gs: cell x >= gs
                1 => y >= gs,
                2 => z >= gs,
                _ => false,
            }
        }
        1 => {
            match face_axis {
                0 => x >= gs,
                1 => y >= gs,
                2 => z >= gs,
                _ => false,
            }
        }
        2 => {
            match face_axis {
                0 => x >= gs,
                1 => y >= gs,
                2 => z >= gs,
                _ => false,
            }
        }
        _ => false,
    }
}

/// Stitch seams between two adjacent chunks along a shared face.
///
/// `a_data`: boundary data from the negative-side chunk (PosX/PosY/PosZ face)
/// `b_data`: boundary data from the positive-side chunk (NegX/NegY/NegZ face)
/// `a_dc_vertices`: DC vertices from chunk A
/// `b_dc_vertices`: DC vertices from chunk B
/// `grid_size`: cells per axis
/// `shared_axis`: 0=X, 1=Y, 2=Z — the axis perpendicular to the shared face
/// `a_world_offset`: world-space offset of chunk A's origin
/// `b_world_offset`: world-space offset of chunk B's origin
///
/// For each sign-changing edge on the shared face, finds the 4 adjacent cells
/// (2 from A, 2 from B) and emits a quad connecting their DC vertices.
#[allow(clippy::too_many_arguments)]
pub fn stitch_seam(
    a_dc_vertices: &[Vec3],
    b_dc_vertices: &[Vec3],
    a_hermite: &HermiteData,
    _b_hermite: &HermiteData,
    grid_size: usize,
    shared_axis: usize,
    a_world_offset: Vec3,
    b_world_offset: Vec3,
) -> Mesh {
    let mut mesh = Mesh::new();
    let mut vertex_map: HashMap<(bool, usize), u32> = HashMap::new(); // (is_chunk_b, cell_idx) -> mesh vertex idx
    let gs = grid_size;

    // Collect boundary edges from both chunks that lie on the shared face.
    // Edges from chunk A's positive face and chunk B's negative face.
    //
    // For shared_axis=0 (X boundary): collect edges that are NOT along X axis
    // (Y-edges and Z-edges) at position x=gs (on chunk A's side).
    // These same edges appear at x=0 in chunk B.
    //
    // Also collect X-edges near the boundary whose adjacent cells span both chunks.

    // Process edges from chunk A's hermite data that lie on the shared boundary
    for (edge_key, intersection) in a_hermite.iter() {
        let ex = edge_key.x() as usize;
        let ey = edge_key.y() as usize;
        let ez = edge_key.z() as usize;
        let axis = edge_key.axis() as usize;

        // Check if this edge lies on A's positive boundary for the shared axis
        if !is_edge_on_pos_boundary(axis, ex, ey, ez, gs, shared_axis) {
            continue;
        }

        // Find the 4 adjacent cells: 2 from chunk A, 2 from chunk B
        let quad = get_boundary_quad_cells(
            axis, ex, ey, ez, gs, shared_axis,
        );

        let quad = match quad {
            Some(q) => q,
            None => continue,
        };

        // Get or create mesh vertices for each of the 4 cells
        let mut quad_verts = [0u32; 4];
        let mut valid = true;

        for (i, &(is_b, cell_idx)) in quad.iter().enumerate() {
            let dc_verts = if is_b { b_dc_vertices } else { a_dc_vertices };
            let world_off = if is_b { b_world_offset } else { a_world_offset };

            if cell_idx >= dc_verts.len() {
                valid = false;
                break;
            }
            let local_pos = dc_verts[cell_idx];
            if local_pos.x.is_nan() {
                valid = false;
                break;
            }

            let world_pos = local_pos + world_off;

            let vi = *vertex_map.entry((is_b, cell_idx)).or_insert_with(|| {
                let idx = mesh.vertices.len() as u32;
                mesh.vertices.push(Vertex {
                    position: world_pos,
                    normal: intersection.normal,
                    material: intersection.material,
                });
                idx
            });
            quad_verts[i] = vi;
        }

        if !valid {
            continue;
        }

        // Determine winding order based on normal direction
        let edge_axis_dir = match axis {
            0 => Vec3::X,
            1 => Vec3::Y,
            _ => Vec3::Z,
        };
        let normal_dot = intersection.normal.dot(edge_axis_dir);

        let (tri_a, tri_b) = if normal_dot > 0.0 {
            ([quad_verts[0], quad_verts[1], quad_verts[2]],
             [quad_verts[0], quad_verts[2], quad_verts[3]])
        } else {
            ([quad_verts[2], quad_verts[1], quad_verts[0]],
             [quad_verts[3], quad_verts[2], quad_verts[0]])
        };

        if !is_degenerate(&mesh.vertices, tri_a) {
            mesh.triangles.push(Triangle { indices: tri_a });
        }
        if !is_degenerate(&mesh.vertices, tri_b) {
            mesh.triangles.push(Triangle { indices: tri_b });
        }
    }

    mesh
}

/// Check if an edge at (ex, ey, ez) with given axis lies on the positive boundary
/// of the chunk along shared_axis. These are edges that mesh_gen would skip because
/// their adjacent cells extend past the chunk boundary.
fn is_edge_on_pos_boundary(
    edge_axis: usize, ex: usize, ey: usize, ez: usize,
    gs: usize, shared_axis: usize,
) -> bool {
    if edge_axis == shared_axis {
        // An edge parallel to the shared axis at the boundary:
        // For shared_axis=0 (X): X-edges whose perp cell coords are at the boundary
        // These aren't the primary boundary edges -- they're handled differently
        return false;
    }

    // An edge perpendicular to shared_axis at the boundary position.
    // For shared_axis=0: Y-edges or Z-edges at x = gs (the positive boundary)
    // For shared_axis=1: X-edges or Z-edges at y = gs
    // For shared_axis=2: X-edges or Y-edges at z = gs
    match shared_axis {
        0 => {
            // Y-edges and Z-edges: their adjacent cells use x offsets.
            // A Y-edge at (x,y,z) has cells (x,y,z), (x-1,y,z), (x,y,z-1), (x-1,y,z-1)
            // A Z-edge at (x,y,z) has cells (x,y,z), (x-1,y,z), (x,y-1,z), (x-1,y-1,z)
            // At x=gs, cell x=gs is in chunk B (at x=0). At x=gs-1, all cells are in A... wait.
            // Actually: the LAST cell in A along X is at x=gs-1.
            // Y-edge at x=gs: cells (gs,y,z-1), (gs-1,y,z-1), (gs,y,z), (gs-1,y,z)
            // Cell gs is outside A, needs to come from B's x=0.
            ex == gs
        }
        1 => ey == gs,
        2 => ez == gs,
        _ => false,
    }
}

/// For an edge on the positive boundary, compute the 4 adjacent cell indices.
/// Returns [(is_chunk_b, cell_index); 4] or None if invalid.
///
/// The edge is at position (ex, ey, ez) with axis `edge_axis` on the boundary of
/// chunk A along `shared_axis`. Some cells are in chunk A, some in chunk B.
fn get_boundary_quad_cells(
    edge_axis: usize, ex: usize, ey: usize, ez: usize,
    gs: usize, shared_axis: usize,
) -> Option<[(bool, usize); 4]> {
    // For each edge type, the 4 adjacent cells have offsets on two perpendicular axes.
    // One of those perp axes is the shared_axis, along which some cells cross into chunk B.

    match (edge_axis, shared_axis) {
        // Y-edge crossing X boundary: perp axes are X and Z
        // Y-edge at (ex,ey,ez): cells (x,y,z), (x-1,y,z), (x,y,z-1), (x-1,y,z-1)
        // where offsets are on X and Z.
        // At ex=gs: x=gs is in B (at x=0), x-1=gs-1 is in A.
        (1, 0) => {
            if ey >= gs || ez == 0 { return None; }
            let a_x = gs - 1; // chunk A's last cell in X
            let b_x = 0usize; // chunk B's first cell in X
            Some([
                (false, cell_idx(a_x, ey, ez - 1, gs)),  // (ex-1, ey, ez-1)
                (true,  cell_idx(b_x, ey, ez - 1, gs)),  // (ex=gs -> B's 0, ey, ez-1)
                (true,  cell_idx(b_x, ey, ez, gs)),       // (ex=gs -> B's 0, ey, ez)
                (false, cell_idx(a_x, ey, ez, gs)),       // (ex-1, ey, ez)
            ])
        }
        // Z-edge crossing X boundary: perp axes are X and Y
        // Z-edge at (ex,ey,ez): cells (x-1,y-1,z), (x,y-1,z), (x,y,z), (x-1,y,z)
        (2, 0) => {
            if ez >= gs || ey == 0 { return None; }
            let a_x = gs - 1;
            let b_x = 0usize;
            Some([
                (false, cell_idx(a_x, ey - 1, ez, gs)),  // (ex-1, ey-1, ez)
                (true,  cell_idx(b_x, ey - 1, ez, gs)),  // (ex -> B's 0, ey-1, ez)
                (true,  cell_idx(b_x, ey, ez, gs)),       // (ex -> B's 0, ey, ez)
                (false, cell_idx(a_x, ey, ez, gs)),       // (ex-1, ey, ez)
            ])
        }
        // X-edge crossing Y boundary: perp axes are Y and Z
        // X-edge at (ex,ey,ez): cells (ex,y,z), (ex,y-1,z), (ex,y,z-1), (ex,y-1,z-1)
        (0, 1) => {
            if ex >= gs || ez == 0 { return None; }
            let a_y = gs - 1;
            let b_y = 0usize;
            Some([
                (false, cell_idx(ex, a_y, ez - 1, gs)),
                (true,  cell_idx(ex, b_y, ez - 1, gs)),
                (true,  cell_idx(ex, b_y, ez, gs)),
                (false, cell_idx(ex, a_y, ez, gs)),
            ])
        }
        // Z-edge crossing Y boundary: perp axes are X and Y
        // Z-edge at (ex,ey,ez): cells (x-1,y-1,z), (x,y-1,z), (x,y,z), (x-1,y,z)
        (2, 1) => {
            if ez >= gs || ex == 0 { return None; }
            let a_y = gs - 1;
            let b_y = 0usize;
            Some([
                (false, cell_idx(ex - 1, a_y, ez, gs)),
                (true,  cell_idx(ex - 1, b_y, ez, gs)),
                (true,  cell_idx(ex, b_y, ez, gs)),
                (false, cell_idx(ex, a_y, ez, gs)),
            ])
        }
        // X-edge crossing Z boundary: perp axes are Y and Z
        // X-edge at (ex,ey,ez): cells (ex,y,z), (ex,y-1,z), (ex,y,z-1), (ex,y-1,z-1)
        (0, 2) => {
            if ex >= gs || ey == 0 { return None; }
            let a_z = gs - 1;
            let b_z = 0usize;
            Some([
                (false, cell_idx(ex, ey - 1, a_z, gs)),
                (true,  cell_idx(ex, ey - 1, b_z, gs)),
                (true,  cell_idx(ex, ey, b_z, gs)),
                (false, cell_idx(ex, ey, a_z, gs)),
            ])
        }
        // Y-edge crossing Z boundary: perp axes are X and Z
        // Y-edge at (ex,ey,ez): cells (x,y,z), (x-1,y,z), (x,y,z-1), (x-1,y,z-1)
        (1, 2) => {
            if ey >= gs || ex == 0 { return None; }
            let a_z = gs - 1;
            let b_z = 0usize;
            Some([
                (false, cell_idx(ex - 1, ey, a_z, gs)),
                (true,  cell_idx(ex - 1, ey, b_z, gs)),
                (true,  cell_idx(ex, ey, b_z, gs)),
                (false, cell_idx(ex, ey, a_z, gs)),
            ])
        }
        _ => None,
    }
}

#[inline]
fn cell_idx(x: usize, y: usize, z: usize, gs: usize) -> usize {
    z * gs * gs + y * gs + x
}

fn is_degenerate(vertices: &[Vertex], indices: [u32; 3]) -> bool {
    let v0 = vertices[indices[0] as usize].position;
    let v1 = vertices[indices[1] as usize].position;
    let v2 = vertices[indices[2] as usize].position;
    (v1 - v0).cross(v2 - v0).length_squared() < 1e-10
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Material;
    use crate::hermite::{EdgeKey, EdgeIntersection, HermiteData};

    #[test]
    fn empty_boundaries_empty_mesh() {
        let a = HermiteData::default();
        let b = HermiteData::default();
        let gs = 4;
        let a_verts = vec![Vec3::new(f32::NAN, f32::NAN, f32::NAN); gs * gs * gs];
        let b_verts = a_verts.clone();
        let mesh = stitch_seam(
            &a_verts, &b_verts,
            &a, &b, gs, 0,
            Vec3::ZERO, Vec3::new(gs as f32, 0.0, 0.0),
        );
        assert!(mesh.is_empty());
    }

    #[test]
    fn extract_boundary_collects_vertices() {
        let gs = 4;
        let mut dc_vertices = vec![Vec3::new(f32::NAN, f32::NAN, f32::NAN); gs * gs * gs];
        // Place a real vertex at (3, 1, 1) -- on the PosX boundary
        dc_vertices[1 * gs * gs + 1 * gs + 3] = Vec3::new(3.5, 1.5, 1.5);

        let hermite = HermiteData::default();
        let data = extract_boundary_data(&hermite, &dc_vertices, gs, ChunkFace::PosX);
        assert!(data.boundary_vertices.contains_key(&(3, 1, 1)));
    }

    #[test]
    fn stitch_produces_seam_geometry() {
        // Create two adjacent chunks with a surface crossing the boundary.
        // Chunk A is at origin, Chunk B is at (gs, 0, 0).
        let gs = 4;
        let nan = Vec3::new(f32::NAN, f32::NAN, f32::NAN);

        // Create hermite data with a Y-edge at x=gs (on A's positive X boundary)
        let mut a_hermite = HermiteData::default();
        // Y-edge at (gs, 1, 2): this edge is on the +X boundary of chunk A
        // Its adjacent cells are (gs-1, 1, 1), (gs, 1, 1), (gs, 1, 2), (gs-1, 1, 2)
        // gs-1 is in A, gs is in B (as B's x=0)
        a_hermite.edges.insert(
            EdgeKey::new(gs as u8, 1, 2, 1),
            EdgeIntersection {
                t: 0.5,
                normal: Vec3::Z,
                material: Material::Limestone,
            },
        );

        let b_hermite = HermiteData::default();

        // Create DC vertices for both chunks
        let mut a_verts = vec![nan; gs * gs * gs];
        let mut b_verts = vec![nan; gs * gs * gs];

        // Place vertices in cells used by the boundary quad
        // Chunk A: cells (gs-1, 1, 1) and (gs-1, 1, 2)
        a_verts[1 * gs * gs + 1 * gs + (gs - 1)] = Vec3::new(3.5, 1.5, 1.5);
        a_verts[2 * gs * gs + 1 * gs + (gs - 1)] = Vec3::new(3.5, 1.5, 2.5);

        // Chunk B: cells (0, 1, 1) and (0, 1, 2)
        b_verts[1 * gs * gs + 1 * gs + 0] = Vec3::new(0.5, 1.5, 1.5);
        b_verts[2 * gs * gs + 1 * gs + 0] = Vec3::new(0.5, 1.5, 2.5);

        let mesh = stitch_seam(
            &a_verts, &b_verts,
            &a_hermite, &b_hermite,
            gs, 0,
            Vec3::ZERO, Vec3::new(gs as f32, 0.0, 0.0),
        );

        assert_eq!(mesh.triangle_count(), 2, "One quad = 2 triangles");
        assert!(mesh.vertex_count() <= 4, "At most 4 vertices for one quad");
    }

    #[test]
    fn stitch_with_no_surface_produces_nothing() {
        let gs = 4;
        let nan = Vec3::new(f32::NAN, f32::NAN, f32::NAN);
        let a_hermite = HermiteData::default();
        let b_hermite = HermiteData::default();
        let a_verts = vec![nan; gs * gs * gs];
        let b_verts = vec![nan; gs * gs * gs];

        let mesh = stitch_seam(
            &a_verts, &b_verts,
            &a_hermite, &b_hermite,
            gs, 0,
            Vec3::ZERO, Vec3::new(gs as f32, 0.0, 0.0),
        );
        assert!(mesh.is_empty());
    }

    #[test]
    fn boundary_edges_detected_for_neg_face() {
        let gs = 4;
        let mut hermite = HermiteData::default();
        // Z-edge at (0, 2, 1): on the NegX face since x=0
        hermite.edges.insert(
            EdgeKey::new(0, 2, 1, 2),
            EdgeIntersection {
                t: 0.5,
                normal: Vec3::X,
                material: Material::Limestone,
            },
        );

        let dc_vertices = vec![Vec3::new(f32::NAN, f32::NAN, f32::NAN); gs * gs * gs];
        let data = extract_boundary_data(&hermite, &dc_vertices, gs, ChunkFace::NegX);
        assert_eq!(data.boundary_edges.len(), 1, "Should detect boundary edge at x=0");
    }
}
