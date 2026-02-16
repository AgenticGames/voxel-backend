use std::collections::HashMap;
use glam::Vec3;
use crate::octree::node::OctreeNode;
use crate::dual_contouring::qef::QefData;
use crate::mesh::{Mesh, Vertex, Triangle};

/// Bottom-up LOD build via QEF merging.
/// Traverses the octree bottom-up and creates LOD vertices at Branch nodes
/// by merging the QEFs of their children.
pub fn build_lod(root: &mut OctreeNode) {
    build_lod_recursive(root, Vec3::ZERO, 1.0);
}

fn build_lod_recursive(node: &mut OctreeNode, origin: Vec3, size: f32) -> Option<QefData> {
    match node {
        OctreeNode::Empty { .. } => None,
        OctreeNode::Leaf { corners, dc_vertex } => {
            // Build QEF from sign-changing edges of this leaf
            let mut qef = QefData::new();
            let has_sign_change = corners.iter().any(|c| c.density <= 0.0)
                && corners.iter().any(|c| c.density > 0.0);

            if !has_sign_change {
                return None;
            }

            // Check each of the 12 edges for sign changes
            let edge_pairs: [(usize, usize, Vec3, Vec3); 12] = [
                // X-edges
                (0, 1, origin, origin + Vec3::new(size, 0.0, 0.0)),
                (2, 3, origin + Vec3::new(0.0, size, 0.0), origin + Vec3::new(size, size, 0.0)),
                (4, 5, origin + Vec3::new(0.0, 0.0, size), origin + Vec3::new(size, 0.0, size)),
                (6, 7, origin + Vec3::new(0.0, size, size), origin + Vec3::new(size, size, size)),
                // Y-edges
                (0, 2, origin, origin + Vec3::new(0.0, size, 0.0)),
                (1, 3, origin + Vec3::new(size, 0.0, 0.0), origin + Vec3::new(size, size, 0.0)),
                (4, 6, origin + Vec3::new(0.0, 0.0, size), origin + Vec3::new(0.0, size, size)),
                (5, 7, origin + Vec3::new(size, 0.0, size), origin + Vec3::new(size, size, size)),
                // Z-edges
                (0, 4, origin, origin + Vec3::new(0.0, 0.0, size)),
                (1, 5, origin + Vec3::new(size, 0.0, 0.0), origin + Vec3::new(size, 0.0, size)),
                (2, 6, origin + Vec3::new(0.0, size, 0.0), origin + Vec3::new(0.0, size, size)),
                (3, 7, origin + Vec3::new(size, size, 0.0), origin + Vec3::new(size, size, size)),
            ];

            for (a_idx, b_idx, pos_a, pos_b) in &edge_pairs {
                let da = corners[*a_idx].density;
                let db = corners[*b_idx].density;
                if (da > 0.0) != (db > 0.0) {
                    let t = da / (da - db);
                    let intersection = *pos_a + (*pos_b - *pos_a) * t;
                    let edge_dir = (*pos_b - *pos_a).normalize_or_zero();
                    let normal = if da < db { edge_dir } else { -edge_dir };
                    qef.add(intersection, normal);
                }
            }

            if qef.count > 0 {
                let vertex = qef.solve_clamped(origin, origin + Vec3::splat(size));
                *dc_vertex = Some(vertex);
                Some(qef)
            } else {
                None
            }
        }
        OctreeNode::Branch { children, lod_vertex, lod_qef } => {
            let half = size / 2.0;
            let child_origins = [
                origin,
                origin + Vec3::new(half, 0.0, 0.0),
                origin + Vec3::new(0.0, half, 0.0),
                origin + Vec3::new(half, half, 0.0),
                origin + Vec3::new(0.0, 0.0, half),
                origin + Vec3::new(half, 0.0, half),
                origin + Vec3::new(0.0, half, half),
                origin + Vec3::new(half, half, half),
            ];

            let mut merged_qef = QefData::new();
            let mut has_data = false;

            for (i, child) in children.iter_mut().enumerate() {
                if let Some(child_qef) = build_lod_recursive(child, child_origins[i], half) {
                    merged_qef.merge(&child_qef);
                    has_data = true;
                }
            }

            if has_data {
                let vertex = merged_qef.solve_clamped(origin, origin + Vec3::splat(size));
                *lod_vertex = Some(vertex);
                *lod_qef = Some(merged_qef.clone());
                Some(merged_qef)
            } else {
                None
            }
        }
    }
}

/// An active cell at a given LOD level with its DC vertex and corner signs.
struct LodCell {
    origin: Vec3,
    size: f32,
    vertex: Vec3,
    /// Sign at each of the 8 corners: true = positive (outside), false = negative (inside)
    corner_signs: [bool; 8],
}

/// Extract mesh at given LOD level.
/// `lod_level`: 0 = full detail (leaves), 1+ = coarser
///
/// Collects active cells at the target LOD depth, then finds adjacent cell pairs
/// that share sign-changing edges and emits quads connecting their DC vertices.
pub fn extract_at_lod(root: &OctreeNode, lod_level: u32) -> Mesh {
    // Step 1: Collect all active cells at the target LOD level
    let mut cells: Vec<LodCell> = Vec::new();
    collect_cells(root, lod_level, 0, Vec3::ZERO, 1.0, &mut cells);

    if cells.is_empty() {
        return Mesh::new();
    }

    // Step 2: Build spatial index for fast neighbor lookups.
    // Key: quantized origin position -> cell index
    let mut cell_map: HashMap<(i32, i32, i32), usize> = HashMap::with_capacity(cells.len());
    for (i, cell) in cells.iter().enumerate() {
        let key = quantize_pos(cell.origin);
        cell_map.insert(key, i);
    }

    // Step 3: For each pair of adjacent cells sharing a sign-changing edge, emit a quad.
    let mut mesh = Mesh::new();
    let mut vertex_map: HashMap<usize, u32> = HashMap::new();

    for (i, cell) in cells.iter().enumerate() {
        let s = cell.size;

        // Check +X neighbor: the cell at (origin.x + size, origin.y, origin.z)
        let nx_key = quantize_pos(cell.origin + Vec3::new(s, 0.0, 0.0));
        if let Some(&j) = cell_map.get(&nx_key) {
            let neighbor = &cells[j];
            // For cells sharing the +X/-X face, check sign-changing edges along the face.
            // The shared face has 4 edges (2 Y-edges and 2 Z-edges).
            // Each edge connects two corners: one from cell and one from neighbor.
            // A sign change along the edge means we need a quad from the 4 cells sharing that edge.
            //
            // For the X-face between cell i and neighbor j:
            // Y-edges on the face: bottom-Z and top-Z
            // Z-edges on the face: bottom-Y and top-Y
            emit_face_quads_x(cell, i, neighbor, j, &cells, &cell_map, &mut mesh, &mut vertex_map);
        }

        // Check +Y neighbor
        let ny_key = quantize_pos(cell.origin + Vec3::new(0.0, s, 0.0));
        if let Some(&j) = cell_map.get(&ny_key) {
            let neighbor = &cells[j];
            emit_face_quads_y(cell, i, neighbor, j, &cells, &cell_map, &mut mesh, &mut vertex_map);
        }

        // Check +Z neighbor
        let nz_key = quantize_pos(cell.origin + Vec3::new(0.0, 0.0, s));
        if let Some(&j) = cell_map.get(&nz_key) {
            let neighbor = &cells[j];
            emit_face_quads_z(cell, i, neighbor, j, &cells, &cell_map, &mut mesh, &mut vertex_map);
        }
    }

    mesh
}

/// Collect active cells from the octree at the target LOD depth.
fn collect_cells(
    node: &OctreeNode,
    target_lod: u32,
    current_depth: u32,
    origin: Vec3,
    size: f32,
    cells: &mut Vec<LodCell>,
) {
    match node {
        OctreeNode::Empty { .. } => {}
        OctreeNode::Leaf { corners, dc_vertex } => {
            if let Some(vertex) = dc_vertex {
                let corner_signs = std::array::from_fn(|i| corners[i].density > 0.0);
                cells.push(LodCell {
                    origin,
                    size,
                    vertex: *vertex,
                    corner_signs,
                });
            }
        }
        OctreeNode::Branch { children, lod_vertex, .. } => {
            if target_lod > 0 && current_depth + 1 >= target_lod {
                // Use LOD vertex at this depth
                if let Some(vertex) = lod_vertex {
                    // Compute aggregate corner signs from children
                    let corner_signs = aggregate_corner_signs(children, origin, size);
                    cells.push(LodCell {
                        origin,
                        size,
                        vertex: *vertex,
                        corner_signs,
                    });
                }
            } else {
                let half = size / 2.0;
                let child_origins = [
                    origin,
                    origin + Vec3::new(half, 0.0, 0.0),
                    origin + Vec3::new(0.0, half, 0.0),
                    origin + Vec3::new(half, half, 0.0),
                    origin + Vec3::new(0.0, 0.0, half),
                    origin + Vec3::new(half, 0.0, half),
                    origin + Vec3::new(0.0, half, half),
                    origin + Vec3::new(half, half, half),
                ];
                for (i, child) in children.iter().enumerate() {
                    collect_cells(child, target_lod, current_depth + 1, child_origins[i], half, cells);
                }
            }
        }
    }
}

/// Get the sign (positive = outside) at a specific corner of an octree node.
fn get_corner_sign(node: &OctreeNode, corner: usize) -> bool {
    match node {
        OctreeNode::Empty { material } => {
            use crate::material::Material;
            // Air is positive (outside), Stone is negative (inside)
            *material == Material::Air
        }
        OctreeNode::Leaf { corners, .. } => corners[corner].density > 0.0,
        OctreeNode::Branch { children, .. } => {
            // The corner of the branch maps to a specific corner of one child.
            // In Morton order: child i's corners map to the parent's corners.
            // Child 0 corner 0 = parent corner 0
            // Child 7 corner 7 = parent corner 7
            // Each child i owns the corner at the same position i in the parent.
            get_corner_sign(&children[corner], corner)
        }
    }
}

/// Compute aggregate corner signs for a branch node from its children.
/// Each of the 8 corners of the parent corresponds to a specific corner of a child.
fn aggregate_corner_signs(children: &[OctreeNode; 8], _origin: Vec3, _size: f32) -> [bool; 8] {
    // Parent corner i corresponds to child i's corner i (the outermost corner)
    std::array::from_fn(|i| get_corner_sign(&children[i], i))
}

fn quantize_pos(pos: Vec3) -> (i32, i32, i32) {
    // Multiply by 1024 for sub-unit precision, then round
    let scale = 1024.0;
    (
        (pos.x * scale).round() as i32,
        (pos.y * scale).round() as i32,
        (pos.z * scale).round() as i32,
    )
}

/// Get or create a mesh vertex for a cell
fn get_or_create_vertex(
    cell_idx: usize,
    cells: &[LodCell],
    mesh: &mut Mesh,
    vertex_map: &mut HashMap<usize, u32>,
    normal: Vec3,
) -> u32 {
    *vertex_map.entry(cell_idx).or_insert_with(|| {
        let idx = mesh.vertices.len() as u32;
        mesh.vertices.push(Vertex {
            position: cells[cell_idx].vertex,
            normal,
            material: crate::material::Material::Limestone,
        });
        idx
    })
}

/// Emit quads for sign-changing edges on the +X face between cell i and neighbor j.
/// The shared face has corners from cell's +X side and neighbor's -X side.
/// Cell corners at +X face: 1(+x,0,0), 3(+x,+y,0), 5(+x,0,+z), 7(+x,+y,+z)
/// Neighbor corners at -X face: 0(0,0,0), 2(0,+y,0), 4(0,0,+z), 6(0,+y,+z)
fn emit_face_quads_x(
    cell: &LodCell, i: usize,
    neighbor: &LodCell, j: usize,
    cells: &[LodCell],
    cell_map: &HashMap<(i32, i32, i32), usize>,
    mesh: &mut Mesh,
    vertex_map: &mut HashMap<usize, u32>,
) {
    let s = cell.size;
    // The 4 edges on the shared face (Y-edges and Z-edges):
    // Y-edge at bottom-Z: between corners (1,0,0)-(3,+y,0) i.e. cell[1]-cell[3] or neighbor[0]-neighbor[2]
    // Y-edge at top-Z:    between corners (5,0,+z)-(7,+y,+z) i.e. cell[5]-cell[7] or neighbor[4]-neighbor[6]
    // Z-edge at bottom-Y: between corners (1,0,0)-(5,0,+z) i.e. cell[1]-cell[5] or neighbor[0]-neighbor[4]
    // Z-edge at top-Y:    between corners (3,+y,0)-(7,+y,+z) i.e. cell[3]-cell[7] or neighbor[2]-neighbor[6]

    // For a Y-edge on this face at bottom-Z:
    // Sign change? cell corner 1 vs cell corner 3 (or equiv: neighbor corner 0 vs neighbor corner 2)
    let sign_1 = cell.corner_signs[1]; // = neighbor corner 0
    let sign_3 = cell.corner_signs[3]; // = neighbor corner 2
    if sign_1 != sign_3 {
        // This Y-edge has a sign change. The 4 cells sharing this edge are:
        // cell i, neighbor j, -Z neighbor of i, -Z neighbor of j
        let nz_i_key = quantize_pos(cell.origin - Vec3::new(0.0, 0.0, s));
        let nz_j_key = quantize_pos(neighbor.origin - Vec3::new(0.0, 0.0, s));
        if let (Some(&ki), Some(&kj)) = (cell_map.get(&nz_i_key), cell_map.get(&nz_j_key)) {
            let normal = if sign_1 { Vec3::Y } else { Vec3::NEG_Y };
            emit_quad(i, j, kj, ki, cells, mesh, vertex_map, normal);
        }
    }

    // Y-edge at top-Z
    let sign_5 = cell.corner_signs[5];
    let sign_7 = cell.corner_signs[7];
    if sign_5 != sign_7 {
        let pz_i_key = quantize_pos(cell.origin + Vec3::new(0.0, 0.0, s));
        let pz_j_key = quantize_pos(neighbor.origin + Vec3::new(0.0, 0.0, s));
        if let (Some(&ki), Some(&kj)) = (cell_map.get(&pz_i_key), cell_map.get(&pz_j_key)) {
            let normal = if sign_5 { Vec3::Y } else { Vec3::NEG_Y };
            emit_quad(i, j, kj, ki, cells, mesh, vertex_map, normal);
        }
    }

    // Z-edge at bottom-Y
    if sign_1 != cell.corner_signs[5] {
        let ny_i_key = quantize_pos(cell.origin - Vec3::new(0.0, s, 0.0));
        let ny_j_key = quantize_pos(neighbor.origin - Vec3::new(0.0, s, 0.0));
        if let (Some(&ki), Some(&kj)) = (cell_map.get(&ny_i_key), cell_map.get(&ny_j_key)) {
            let normal = if sign_1 { Vec3::Z } else { Vec3::NEG_Z };
            emit_quad(i, j, kj, ki, cells, mesh, vertex_map, normal);
        }
    }

    // Z-edge at top-Y
    if sign_3 != cell.corner_signs[7] {
        let py_i_key = quantize_pos(cell.origin + Vec3::new(0.0, s, 0.0));
        let py_j_key = quantize_pos(neighbor.origin + Vec3::new(0.0, s, 0.0));
        if let (Some(&ki), Some(&kj)) = (cell_map.get(&py_i_key), cell_map.get(&py_j_key)) {
            let normal = if sign_3 { Vec3::Z } else { Vec3::NEG_Z };
            emit_quad(i, j, kj, ki, cells, mesh, vertex_map, normal);
        }
    }
}

/// Emit quads for sign-changing edges on the +Y face between cell i and neighbor j.
fn emit_face_quads_y(
    cell: &LodCell, i: usize,
    neighbor: &LodCell, j: usize,
    cells: &[LodCell],
    cell_map: &HashMap<(i32, i32, i32), usize>,
    mesh: &mut Mesh,
    vertex_map: &mut HashMap<usize, u32>,
) {
    let s = cell.size;
    // +Y face corners: cell 2(0,+y,0), 3(+x,+y,0), 6(0,+y,+z), 7(+x,+y,+z)
    // Neighbor -Y face: neighbor 0(0,0,0), 1(+x,0,0), 4(0,0,+z), 5(+x,0,+z)

    // X-edge at bottom-Z: cell[2] vs cell[3]
    let sign_2 = cell.corner_signs[2];
    let sign_3 = cell.corner_signs[3];
    if sign_2 != sign_3 {
        let nz_i_key = quantize_pos(cell.origin - Vec3::new(0.0, 0.0, s));
        let nz_j_key = quantize_pos(neighbor.origin - Vec3::new(0.0, 0.0, s));
        if let (Some(&ki), Some(&kj)) = (cell_map.get(&nz_i_key), cell_map.get(&nz_j_key)) {
            let normal = if sign_2 { Vec3::X } else { Vec3::NEG_X };
            emit_quad(i, j, kj, ki, cells, mesh, vertex_map, normal);
        }
    }

    // X-edge at top-Z: cell[6] vs cell[7]
    let sign_6 = cell.corner_signs[6];
    let sign_7 = cell.corner_signs[7];
    if sign_6 != sign_7 {
        let pz_i_key = quantize_pos(cell.origin + Vec3::new(0.0, 0.0, s));
        let pz_j_key = quantize_pos(neighbor.origin + Vec3::new(0.0, 0.0, s));
        if let (Some(&ki), Some(&kj)) = (cell_map.get(&pz_i_key), cell_map.get(&pz_j_key)) {
            let normal = if sign_6 { Vec3::X } else { Vec3::NEG_X };
            emit_quad(i, j, kj, ki, cells, mesh, vertex_map, normal);
        }
    }

    // Z-edge at bottom-X: cell[2] vs cell[6]
    if sign_2 != sign_6 {
        let nx_i_key = quantize_pos(cell.origin - Vec3::new(s, 0.0, 0.0));
        let nx_j_key = quantize_pos(neighbor.origin - Vec3::new(s, 0.0, 0.0));
        if let (Some(&ki), Some(&kj)) = (cell_map.get(&nx_i_key), cell_map.get(&nx_j_key)) {
            let normal = if sign_2 { Vec3::Z } else { Vec3::NEG_Z };
            emit_quad(i, j, kj, ki, cells, mesh, vertex_map, normal);
        }
    }

    // Z-edge at top-X: cell[3] vs cell[7]
    if sign_3 != sign_7 {
        let px_i_key = quantize_pos(cell.origin + Vec3::new(s, 0.0, 0.0));
        let px_j_key = quantize_pos(neighbor.origin + Vec3::new(s, 0.0, 0.0));
        if let (Some(&ki), Some(&kj)) = (cell_map.get(&px_i_key), cell_map.get(&px_j_key)) {
            let normal = if sign_3 { Vec3::Z } else { Vec3::NEG_Z };
            emit_quad(i, j, kj, ki, cells, mesh, vertex_map, normal);
        }
    }
}

/// Emit quads for sign-changing edges on the +Z face between cell i and neighbor j.
fn emit_face_quads_z(
    cell: &LodCell, i: usize,
    neighbor: &LodCell, j: usize,
    cells: &[LodCell],
    cell_map: &HashMap<(i32, i32, i32), usize>,
    mesh: &mut Mesh,
    vertex_map: &mut HashMap<usize, u32>,
) {
    let s = cell.size;
    // +Z face corners: cell 4(0,0,+z), 5(+x,0,+z), 6(0,+y,+z), 7(+x,+y,+z)

    // X-edge at bottom-Y: cell[4] vs cell[5]
    let sign_4 = cell.corner_signs[4];
    let sign_5 = cell.corner_signs[5];
    if sign_4 != sign_5 {
        let ny_i_key = quantize_pos(cell.origin - Vec3::new(0.0, s, 0.0));
        let ny_j_key = quantize_pos(neighbor.origin - Vec3::new(0.0, s, 0.0));
        if let (Some(&ki), Some(&kj)) = (cell_map.get(&ny_i_key), cell_map.get(&ny_j_key)) {
            let normal = if sign_4 { Vec3::X } else { Vec3::NEG_X };
            emit_quad(i, j, kj, ki, cells, mesh, vertex_map, normal);
        }
    }

    // X-edge at top-Y: cell[6] vs cell[7]
    let sign_6 = cell.corner_signs[6];
    let sign_7 = cell.corner_signs[7];
    if sign_6 != sign_7 {
        let py_i_key = quantize_pos(cell.origin + Vec3::new(0.0, s, 0.0));
        let py_j_key = quantize_pos(neighbor.origin + Vec3::new(0.0, s, 0.0));
        if let (Some(&ki), Some(&kj)) = (cell_map.get(&py_i_key), cell_map.get(&py_j_key)) {
            let normal = if sign_6 { Vec3::X } else { Vec3::NEG_X };
            emit_quad(i, j, kj, ki, cells, mesh, vertex_map, normal);
        }
    }

    // Y-edge at bottom-X: cell[4] vs cell[6]
    if sign_4 != sign_6 {
        let nx_i_key = quantize_pos(cell.origin - Vec3::new(s, 0.0, 0.0));
        let nx_j_key = quantize_pos(neighbor.origin - Vec3::new(s, 0.0, 0.0));
        if let (Some(&ki), Some(&kj)) = (cell_map.get(&nx_i_key), cell_map.get(&nx_j_key)) {
            let normal = if sign_4 { Vec3::Y } else { Vec3::NEG_Y };
            emit_quad(i, j, kj, ki, cells, mesh, vertex_map, normal);
        }
    }

    // Y-edge at top-X: cell[5] vs cell[7]
    if sign_5 != sign_7 {
        let px_i_key = quantize_pos(cell.origin + Vec3::new(s, 0.0, 0.0));
        let px_j_key = quantize_pos(neighbor.origin + Vec3::new(s, 0.0, 0.0));
        if let (Some(&ki), Some(&kj)) = (cell_map.get(&px_i_key), cell_map.get(&px_j_key)) {
            let normal = if sign_5 { Vec3::Y } else { Vec3::NEG_Y };
            emit_quad(i, j, kj, ki, cells, mesh, vertex_map, normal);
        }
    }
}

/// Emit a quad (two triangles) connecting 4 cells' DC vertices.
fn emit_quad(
    a: usize, b: usize, c: usize, d: usize,
    cells: &[LodCell],
    mesh: &mut Mesh,
    vertex_map: &mut HashMap<usize, u32>,
    normal: Vec3,
) {
    let va = get_or_create_vertex(a, cells, mesh, vertex_map, normal);
    let vb = get_or_create_vertex(b, cells, mesh, vertex_map, normal);
    let vc = get_or_create_vertex(c, cells, mesh, vertex_map, normal);
    let vd = get_or_create_vertex(d, cells, mesh, vertex_map, normal);

    // Split quad into two triangles, checking for degeneracy
    let tri1 = [va, vb, vc];
    let tri2 = [va, vc, vd];

    if !is_degenerate(mesh, tri1) {
        mesh.triangles.push(Triangle { indices: tri1 });
    }
    if !is_degenerate(mesh, tri2) {
        mesh.triangles.push(Triangle { indices: tri2 });
    }
}

fn is_degenerate(mesh: &Mesh, indices: [u32; 3]) -> bool {
    let v0 = mesh.vertices[indices[0] as usize].position;
    let v1 = mesh.vertices[indices[1] as usize].position;
    let v2 = mesh.vertices[indices[2] as usize].position;
    (v1 - v0).cross(v2 - v0).length_squared() < 1e-10
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::octree::builder::build_octree;
    use crate::octree::node::OctreeConfig;

    fn make_sphere_grid(size: usize, radius: f32) -> Vec<f32> {
        let grid_size = size + 1;
        let center = size as f32 / 2.0;
        let mut grid = vec![0.0f32; grid_size * grid_size * grid_size];
        for z in 0..grid_size {
            for y in 0..grid_size {
                for x in 0..grid_size {
                    let dx = x as f32 - center;
                    let dy = y as f32 - center;
                    let dz = z as f32 - center;
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    let idx = z * grid_size * grid_size + y * grid_size + x;
                    grid[idx] = dist - radius;
                }
            }
        }
        grid
    }

    #[test]
    fn build_lod_on_sphere() {
        let size = 4;
        let grid = make_sphere_grid(size, 1.5);
        let config = OctreeConfig { max_depth: 4, error_threshold: 0.01 };
        let mut root = build_octree(&grid, size, &config);
        build_lod(&mut root);

        // After LOD build, branch nodes should have lod_vertex set
        match &root {
            OctreeNode::Branch { lod_vertex, .. } => {
                assert!(lod_vertex.is_some(), "Root branch should have LOD vertex after build_lod");
            }
            _ => {} // Might not be a branch for small grids
        }
    }

    #[test]
    fn extract_at_lod_zero_produces_triangles() {
        let size = 4;
        let grid = make_sphere_grid(size, 1.5);
        let config = OctreeConfig { max_depth: 4, error_threshold: 0.01 };
        let mut root = build_octree(&grid, size, &config);
        build_lod(&mut root);

        let mesh = extract_at_lod(&root, 0);
        assert!(mesh.vertex_count() > 0, "Should have vertices");
        assert!(mesh.triangle_count() > 0,
            "Should produce triangles, got {} vertices {} triangles",
            mesh.vertex_count(), mesh.triangle_count());
    }

    #[test]
    fn extract_at_coarse_lod_fewer_vertices() {
        let size = 8;
        let grid = make_sphere_grid(size, 3.0);
        let config = OctreeConfig { max_depth: 4, error_threshold: 0.01 };
        let mut root = build_octree(&grid, size, &config);
        build_lod(&mut root);

        let mesh_full = extract_at_lod(&root, 0);
        let mesh_coarse = extract_at_lod(&root, 2);

        // Full detail should have at least as many vertices as coarse
        assert!(mesh_full.vertex_count() >= mesh_coarse.vertex_count(),
            "Full detail {} should have >= vertices than coarse {}",
            mesh_full.vertex_count(), mesh_coarse.vertex_count());
    }

    #[test]
    fn uniform_field_no_lod_vertices() {
        let size = 2;
        let grid = vec![1.0f32; 3 * 3 * 3];
        let config = OctreeConfig::default();
        let mut root = build_octree(&grid, size, &config);
        build_lod(&mut root);
        let mesh = extract_at_lod(&root, 0);
        assert!(mesh.is_empty(), "Uniform field should produce no LOD vertices");
    }
}
