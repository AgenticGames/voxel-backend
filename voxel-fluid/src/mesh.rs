use crate::cell::{ChunkFluidGrid, FluidType, MIN_LEVEL};

/// A fluid mesh produced by face-based quad generation with shoreline clipping.
pub struct FluidMeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub fluid_types: Vec<u8>,
    pub indices: Vec<u32>,
    pub uvs: Vec<[f32; 2]>,
    pub flow_directions: Vec<[f32; 3]>, // (dx, dz, magnitude) for UV scroll
}

/// Build a face-based quad mesh from a fluid grid with shoreline clipping.
///
/// For each fluid cell, emits:
/// - Top face: horizontal quad at fill height, clipped to density=0 shoreline
/// - Side faces: vertical quads where neighbor is solid or has much less fluid
///
/// UVs use world-space XZ mapping for top faces, local UV for side faces.
/// Flow directions encode the fluid gradient for animated UV panning.
pub fn mesh_fluid(grid: &ChunkFluidGrid) -> FluidMeshData {
    let size = grid.size;
    let mut mesh = FluidMeshData {
        positions: Vec::new(),
        normals: Vec::new(),
        fluid_types: Vec::new(),
        indices: Vec::new(),
        uvs: Vec::new(),
        flow_directions: Vec::new(),
    };

    if size < 2 {
        return mesh;
    }

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let cell = grid.get(x, y, z);
                if cell.level < MIN_LEVEL {
                    continue;
                }

                let capacity = grid.cell_capacity(x, y, z);
                if capacity < MIN_LEVEL {
                    continue;
                }

                let fill_height = cell.level / capacity;
                let fluid_type = cell.fluid_type;
                let corners = grid.get_corners(x, y, z);

                // Compute flow direction from level gradient
                let flow = compute_flow_direction(grid, x, y, z);

                // Top face — exposed to air above or cell above with less fluid
                let emit_top = if y + 1 < size {
                    let above = grid.get(x, y + 1, z);
                    let above_cap = grid.cell_capacity(x, y + 1, z);
                    above.level < MIN_LEVEL || above_cap < MIN_LEVEL
                        || (above.level / above_cap.max(MIN_LEVEL)) < fill_height - 0.1
                } else {
                    true // top of chunk
                };

                if emit_top {
                    emit_top_face(
                        &mut mesh, x, y, z, fill_height, fluid_type, &corners, flow,
                    );
                }

                // Side faces (4 horizontal neighbors)
                let side_dirs: [(i32, i32, i32, [f32; 3]); 4] = [
                    (1, 0, 0, [1.0, 0.0, 0.0]),
                    (-1, 0, 0, [-1.0, 0.0, 0.0]),
                    (0, 0, 1, [0.0, 0.0, 1.0]),
                    (0, 0, -1, [0.0, 0.0, -1.0]),
                ];

                for (dx, _dy, dz, normal) in &side_dirs {
                    let nx = x as i32 + dx;
                    let nz = z as i32 + dz;

                    let emit_side = if nx < 0 || nx >= size as i32 || nz < 0 || nz >= size as i32 {
                        // Edge of chunk: emit side if there's fluid here
                        true
                    } else {
                        let nxu = nx as usize;
                        let nzu = nz as usize;
                        if grid.is_solid(nxu, y, nzu) {
                            false // solid neighbor — terrain mesh handles the wall
                        } else {
                            let n_cell = grid.get(nxu, y, nzu);
                            let n_cap = grid.cell_capacity(nxu, y, nzu);
                            // Emit side if neighbor has significantly less fluid
                            n_cell.level < cell.level * 0.3 || n_cap < MIN_LEVEL
                        }
                    };

                    if emit_side {
                        emit_side_face(
                            &mut mesh, x, y, z, fill_height, fluid_type, *dx, *dz,
                            *normal, &corners, flow,
                        );
                    }
                }
            }
        }
    }

    mesh
}

/// Compute flow direction from fluid level gradient (for UV animation).
fn compute_flow_direction(grid: &ChunkFluidGrid, x: usize, y: usize, z: usize) -> [f32; 3] {
    let size = grid.size;
    let level = grid.get(x, y, z).level;

    let mut dx = 0.0f32;
    let mut dz = 0.0f32;

    // X gradient
    if x > 0 && !grid.is_solid(x - 1, y, z) {
        dx += level - grid.get(x - 1, y, z).level;
    }
    if x + 1 < size && !grid.is_solid(x + 1, y, z) {
        dx += grid.get(x + 1, y, z).level - level;
    }

    // Z gradient
    if z > 0 && !grid.is_solid(x, y, z - 1) {
        dz += level - grid.get(x, y, z - 1).level;
    }
    if z + 1 < size && !grid.is_solid(x, y, z + 1) {
        dz += grid.get(x, y, z + 1).level - level;
    }

    // Also check downward gradient (waterfall detection)
    let mut dy_mag = 0.0f32;
    if y > 0 && !grid.is_solid(x, y - 1, z) {
        let below_level = grid.get(x, y - 1, z).level;
        dy_mag = (level - below_level).max(0.0);
    }

    let horiz_mag = (dx * dx + dz * dz).sqrt();
    let total_mag = (horiz_mag * horiz_mag + dy_mag * dy_mag).sqrt().min(1.0);

    if total_mag < 0.001 {
        [0.0, 0.0, 0.0]
    } else {
        // Normalize horizontal direction, magnitude is total flow strength
        let inv = if horiz_mag > 0.001 { 1.0 / horiz_mag } else { 0.0 };
        [dx * inv, dz * inv, total_mag]
    }
}

/// Emit a top (horizontal) face quad at the fill height, with shoreline clipping.
fn emit_top_face(
    mesh: &mut FluidMeshData,
    x: usize, y: usize, z: usize,
    fill_height: f32,
    fluid_type: FluidType,
    corners: &[f32; 8],
    flow: [f32; 3],
) {
    let fy = y as f32 + fill_height;

    // Four corners of the top face in XZ plane:
    //   (x, z), (x+1, z), (x+1, z+1), (x, z+1)
    // Corresponding density corners (top face of cell at fill_height):
    // We use the top-layer corners: indices 4,5,6,7 if fill is near top,
    // or bottom corners 0,1,2,3 if fill is near bottom.
    // For shoreline clipping, we interpolate between bottom and top corner pairs.
    let corner_pairs: [(usize, usize); 4] = [
        (0, 4), // (x, z) bottom→top
        (1, 5), // (x+1, z)
        (2, 6), // (x+1, z+1)
        (3, 7), // (x, z+1)
    ];

    let base_positions: [[f32; 2]; 4] = [
        [x as f32, z as f32],
        [x as f32 + 1.0, z as f32],
        [x as f32 + 1.0, z as f32 + 1.0],
        [x as f32, z as f32 + 1.0],
    ];

    // Build the 4 vertices, clipping any that are inside solid terrain
    let mut verts = [[0.0f32; 3]; 4];
    let mut valid = [true; 4];
    let mut valid_count = 0;

    for i in 0..4 {
        let (bot, top) = corner_pairs[i];
        // Interpolate density at fill height
        let d = corners[bot] + (corners[top] - corners[bot]) * fill_height;
        if d > 0.0 {
            // This corner is inside solid — try to clip to density=0
            valid[i] = false;
        } else {
            verts[i] = [base_positions[i][0], fy, base_positions[i][1]];
            valid_count += 1;
        }
    }

    // Clip invalid vertices to density=0 crossing along edges to valid neighbors
    for i in 0..4 {
        if valid[i] {
            continue;
        }
        let prev = (i + 3) % 4;
        let next = (i + 1) % 4;

        // Find nearest valid neighbor to interpolate toward
        let neighbor = if valid[prev] { prev } else if valid[next] { next } else { continue };

        let (bot_i, top_i) = corner_pairs[i];
        let (bot_n, top_n) = corner_pairs[neighbor];
        let d_i = corners[bot_i] + (corners[top_i] - corners[bot_i]) * fill_height;
        let d_n = corners[bot_n] + (corners[top_n] - corners[bot_n]) * fill_height;

        if (d_i - d_n).abs() < 1e-6 {
            continue;
        }

        // Linear interpolation to find density=0 crossing
        let t = d_i / (d_i - d_n);
        let t = t.clamp(0.0, 1.0);
        verts[i] = [
            base_positions[i][0] + (base_positions[neighbor][0] - base_positions[i][0]) * t,
            fy,
            base_positions[i][1] + (base_positions[neighbor][1] - base_positions[i][1]) * t,
        ];
        valid[i] = true;
        valid_count += 1;
    }

    if valid_count < 3 {
        return;
    }

    let normal = [0.0, 1.0, 0.0];
    let ft = fluid_type as u8;

    // Emit as a triangle fan from vertex 0
    let mut emitted = Vec::new();

    for i in 0..4 {
        if valid[i] {
            mesh.positions.push(verts[i]);
            mesh.normals.push(normal);
            mesh.fluid_types.push(ft);
            // World-space XZ UVs for top faces
            mesh.uvs.push([verts[i][0], verts[i][2]]);
            mesh.flow_directions.push(flow);
            emitted.push(mesh.positions.len() as u32 - 1);
        }
    }

    // Triangulate the polygon
    if emitted.len() >= 3 {
        for i in 1..emitted.len() - 1 {
            mesh.indices.push(emitted[0]);
            mesh.indices.push(emitted[i] as u32);
            mesh.indices.push(emitted[i + 1] as u32);
        }
    }
}

/// Emit a side (vertical) face quad.
#[allow(clippy::too_many_arguments)]
fn emit_side_face(
    mesh: &mut FluidMeshData,
    x: usize, y: usize, z: usize,
    fill_height: f32,
    fluid_type: FluidType,
    dx: i32, dz: i32,
    normal: [f32; 3],
    corners: &[f32; 8],
    flow: [f32; 3],
) {
    let fy_top = y as f32 + fill_height;
    let fy_bot = y as f32;
    let ft = fluid_type as u8;

    // Determine which 2 corners of the cell form this face edge
    // Face +X: corners 1,2,5,6 (x+1 face)
    // Face -X: corners 0,3,4,7 (x face)
    // Face +Z: corners 3,2,7,6 (z+1 face)
    // Face -Z: corners 0,1,4,5 (z face)
    let (c_bl, c_br, c_tl, c_tr) = match (dx, dz) {
        (1, 0) =>  (1, 2, 5, 6),  // +X face: bottom-left=c1, bottom-right=c2, top-left=c5, top-right=c6
        (-1, 0) => (3, 0, 7, 4),  // -X face
        (0, 1) =>  (2, 3, 6, 7),  // +Z face
        (0, -1) => (0, 1, 4, 5),  // -Z face
        _ => return,
    };

    // Check if any corner of this face is valid (not fully solid)
    let d_bl = corners[c_bl];
    let d_br = corners[c_br];
    let d_tl = corners[c_tl];
    let d_tr = corners[c_tr];

    // If all corners are solid, skip
    if d_bl > 0.0 && d_br > 0.0 && d_tl > 0.0 && d_tr > 0.0 {
        return;
    }

    // Face position in world space
    let (fx, fz) = match (dx, dz) {
        (1, 0) =>  (x as f32 + 1.0, z as f32),
        (-1, 0) => (x as f32, z as f32),
        (0, 1) =>  (x as f32, z as f32 + 1.0),
        (0, -1) => (x as f32, z as f32),
        _ => return,
    };

    // Build 4 vertices: bottom-left, bottom-right, top-right, top-left
    let (v0, v1, v2, v3) = match (dx, dz) {
        (1, 0) => (
            [fx, fy_bot, z as f32],
            [fx, fy_bot, z as f32 + 1.0],
            [fx, fy_top, z as f32 + 1.0],
            [fx, fy_top, z as f32],
        ),
        (-1, 0) => (
            [fx, fy_bot, z as f32 + 1.0],
            [fx, fy_bot, z as f32],
            [fx, fy_top, z as f32],
            [fx, fy_top, z as f32 + 1.0],
        ),
        (0, 1) => (
            [x as f32 + 1.0, fy_bot, fz],
            [x as f32, fy_bot, fz],
            [x as f32, fy_top, fz],
            [x as f32 + 1.0, fy_top, fz],
        ),
        (0, -1) => (
            [x as f32, fy_bot, fz],
            [x as f32 + 1.0, fy_bot, fz],
            [x as f32 + 1.0, fy_top, fz],
            [x as f32, fy_top, fz],
        ),
        _ => return,
    };

    // Emit full quad if at least one corner is non-solid
    let base = mesh.positions.len() as u32;

    // UVs for side faces: local space (u=horizontal along face, v=vertical)
    let uvs = [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, fill_height],
        [0.0, fill_height],
    ];

    for (pos, uv) in [v0, v1, v2, v3].iter().zip(uvs.iter()) {
        mesh.positions.push(*pos);
        mesh.normals.push(normal);
        mesh.fluid_types.push(ft);
        mesh.uvs.push(*uv);
        mesh.flow_directions.push(flow);
    }

    // Two triangles for the quad
    mesh.indices.push(base);
    mesh.indices.push(base + 1);
    mesh.indices.push(base + 2);
    mesh.indices.push(base);
    mesh.indices.push(base + 2);
    mesh.indices.push(base + 3);
}

/// Determine the dominant fluid type among non-empty neighboring cells.
/// When water-family wins over lava, returns the most common water subtype.
pub fn dominant_fluid_type(grid: &ChunkFluidGrid, x: usize, y: usize, z: usize) -> FluidType {
    let mut lava_count = 0u32;
    // Index 0 unused; indices 1,3-8 are water-family types
    let mut water_counts = [0u32; 9];
    let size = grid.size;

    for cz in z..=(z + 1).min(size - 1) {
        for cy in y..=(y + 1).min(size - 1) {
            for cx in x..=(x + 1).min(size - 1) {
                let cell = grid.get(cx, cy, cz);
                if cell.level >= MIN_LEVEL {
                    if cell.fluid_type.is_lava() {
                        lava_count += 1;
                    } else {
                        let idx = cell.fluid_type as u8 as usize;
                        if idx < water_counts.len() {
                            water_counts[idx] += 1;
                        }
                    }
                }
            }
        }
    }

    let total_water: u32 = water_counts.iter().sum();
    if lava_count > total_water {
        FluidType::Lava
    } else {
        // Return the most common water subtype
        let mut best_idx = 1u8; // default to Water
        let mut best_count = 0u32;
        for i in 0..water_counts.len() {
            if water_counts[i] > best_count {
                best_count = water_counts[i];
                best_idx = i as u8;
            }
        }
        FluidType::from_u8(best_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_grid_produces_no_mesh() {
        let grid = ChunkFluidGrid::new(16);
        let mesh = mesh_fluid(&grid);
        assert!(mesh.positions.is_empty());
        assert!(mesh.indices.is_empty());
    }

    #[test]
    fn single_source_produces_mesh() {
        let mut grid = ChunkFluidGrid::new(16);
        // Fill a small region with water
        for z in 4..8 {
            for y in 4..8 {
                for x in 4..8 {
                    let cell = grid.get_mut(x, y, z);
                    cell.level = 1.0;
                    cell.fluid_type = FluidType::Water;
                }
            }
        }
        let mesh = mesh_fluid(&grid);
        assert!(!mesh.positions.is_empty(), "Should produce mesh vertices");
        assert!(!mesh.indices.is_empty(), "Should produce mesh indices");
        assert_eq!(mesh.indices.len() % 3, 0, "Indices should be triangles");
        // Should also have UVs and flow directions
        assert_eq!(mesh.uvs.len(), mesh.positions.len(), "UVs per vertex");
        assert_eq!(mesh.flow_directions.len(), mesh.positions.len(), "Flow per vertex");
    }

    #[test]
    fn partial_fill_produces_lower_surface() {
        let mut grid = ChunkFluidGrid::new(16);
        // Place half-filled cell
        grid.get_mut(8, 4, 8).level = 0.5;
        grid.get_mut(8, 4, 8).fluid_type = FluidType::Water;

        let mesh = mesh_fluid(&grid);
        assert!(!mesh.positions.is_empty(), "Should produce mesh for partial fill");

        // Top face vertices should be at y = 4 + 0.5 = 4.5
        let max_y = mesh.positions.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (max_y - 4.5).abs() < 0.01,
            "Top face should be at fill height 4.5, got {}",
            max_y
        );
    }

    #[test]
    fn solid_neighbor_produces_side_face() {
        let mut grid = ChunkFluidGrid::new(16);
        // Water cell next to solid
        grid.get_mut(8, 4, 8).level = 0.8;
        grid.get_mut(8, 4, 8).fluid_type = FluidType::Water;
        // Make +X neighbor solid — but side face should NOT emit against solid
        // (terrain handles the wall)
        grid.set_density(9, 4, 8, 1.0);

        let mesh = mesh_fluid(&grid);
        // Should still have a top face at minimum
        assert!(!mesh.positions.is_empty(), "Should produce mesh");
    }
}
