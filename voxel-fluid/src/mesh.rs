use crate::cell::{ChunkFluidGrid, FluidType, MIN_LEVEL};
use crate::tables::{CORNER_OFFSETS, EDGE_TABLE, EDGE_VERTICES, TRI_TABLE};

/// Isosurface threshold for fluid meshing.
pub(crate) const ISO_LEVEL: f32 = 0.15;
/// Tiny SDF value for out-of-bounds samples — places boundary faces near chunk edge.
const BOUNDARY_SDF: f32 = 0.001;
/// Field value for out-of-bounds samples — just below ISO_LEVEL so MC places face at edge.
const BOUNDARY_FIELD: f32 = ISO_LEVEL - BOUNDARY_SDF;
/// Y offset for MC mesh to avoid z-fighting with Surface Nets.
const MC_Y_OFFSET: f32 = 0.05;
/// Surface Nets fluid type: WaterBreach (yellow-green, highly visible for A/B).
const SN_FLUID_TYPE: u8 = 5;
/// Marching Cubes fluid type: WaterDrip (purple).
const MC_FLUID_TYPE: u8 = 4;

/// Surface Nets edge table: 12 edges as pairs of corner indices.
/// Binary encoding: bit 0=X, bit 1=Y, bit 2=Z.
const CUBE_EDGES: [[usize; 2]; 12] = [
    [0b000, 0b001],
    [0b000, 0b010],
    [0b000, 0b100],
    [0b001, 0b011],
    [0b001, 0b101],
    [0b010, 0b011],
    [0b010, 0b110],
    [0b011, 0b111],
    [0b100, 0b101],
    [0b100, 0b110],
    [0b101, 0b111],
    [0b110, 0b111],
];

/// Corner positions for Surface Nets (binary encoding: bit 0=X, bit 1=Y, bit 2=Z).
const SN_CORNERS: [[f32; 3]; 8] = [
    [0.0, 0.0, 0.0], // 0b000
    [1.0, 0.0, 0.0], // 0b001
    [0.0, 1.0, 0.0], // 0b010
    [1.0, 1.0, 0.0], // 0b011
    [0.0, 0.0, 1.0], // 0b100
    [1.0, 0.0, 1.0], // 0b101
    [0.0, 1.0, 1.0], // 0b110
    [1.0, 1.0, 1.0], // 0b111
];

/// Fluid levels from neighboring chunks at the 3 positive boundary faces.
/// Used to create seamless mesh at chunk edges instead of sealing the isosurface.
pub struct BoundaryLevels {
    /// Neighbor's x=0 face pre-baked field values, indexed [z * size + y]. size*size values.
    pub pos_x: Option<Vec<f32>>,
    /// Neighbor's y=0 face pre-baked field values, indexed [z * size + x]. size*size values.
    pub pos_y: Option<Vec<f32>>,
    /// Neighbor's z=0 face pre-baked field values, indexed [y * size + x]. size*size values.
    pub pos_z: Option<Vec<f32>>,
    /// Whether the neighbor cell at x=0 face has actual fluid (level >= MIN_LEVEL).
    pub pos_x_fluid: Option<Vec<bool>>,
    /// Whether the neighbor cell at y=0 face has actual fluid.
    pub pos_y_fluid: Option<Vec<bool>>,
    /// Whether the neighbor cell at z=0 face has actual fluid.
    pub pos_z_fluid: Option<Vec<bool>>,
    pub size: usize,
}

impl BoundaryLevels {
    /// Create empty boundary levels (no neighbor data — mesh seals at edges).
    pub fn empty(size: usize) -> Self {
        Self {
            pos_x: None,
            pos_y: None,
            pos_z: None,
            pos_x_fluid: None,
            pos_y_fluid: None,
            pos_z_fluid: None,
            size,
        }
    }

    /// Get the fluid level at an out-of-bounds coordinate from neighbor data.
    /// Returns None for multi-axis overflow or if no neighbor data exists.
    pub fn get_level(&self, x: usize, y: usize, z: usize) -> Option<f32> {
        let size = self.size;
        let x_over = x >= size;
        let y_over = y >= size;
        let z_over = z >= size;

        // Multi-axis overflow or no overflow — no data
        if (x_over as u8 + y_over as u8 + z_over as u8) != 1 {
            return None;
        }

        if x_over {
            // x == size: look up pos_x face at [z * size + y]
            self.pos_x.as_ref().map(|v| v[z * size + y])
        } else if y_over {
            // y == size: look up pos_y face at [z * size + x]
            self.pos_y.as_ref().map(|v| v[z * size + x])
        } else {
            // z == size: look up pos_z face at [y * size + x]
            self.pos_z.as_ref().map(|v| v[y * size + x])
        }
    }

    /// Get whether the cell at an out-of-bounds coordinate has actual fluid.
    /// Returns None for multi-axis overflow or if no neighbor data exists.
    pub fn get_has_fluid(&self, x: usize, y: usize, z: usize) -> Option<bool> {
        let size = self.size;
        let x_over = x >= size;
        let y_over = y >= size;
        let z_over = z >= size;

        if (x_over as u8 + y_over as u8 + z_over as u8) != 1 {
            return None;
        }

        if x_over {
            self.pos_x_fluid.as_ref().map(|v| v[z * size + y])
        } else if y_over {
            self.pos_y_fluid.as_ref().map(|v| v[z * size + x])
        } else {
            self.pos_z_fluid.as_ref().map(|v| v[y * size + x])
        }
    }
}

/// A fluid mesh produced by isosurface extraction (MC + Surface Nets).
pub struct FluidMeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub fluid_types: Vec<u8>,
    pub indices: Vec<u32>,
    pub uvs: Vec<[f32; 2]>,
    pub flow_directions: Vec<[f32; 3]>, // (dx, dz, magnitude) for UV scroll
}

/// Build dual isosurface mesh: Surface Nets (teal) + Marching Cubes (purple).
///
/// Both meshers extract the same isosurface from the fluid grid at ISO_LEVEL.
/// They output different fluid_type values so UE renders them with different
/// materials for A/B comparison.
pub fn mesh_fluid(grid: &ChunkFluidGrid, boundary: &BoundaryLevels) -> FluidMeshData {
    let size = grid.size;
    if size < 2 {
        return FluidMeshData {
            positions: Vec::new(),
            normals: Vec::new(),
            fluid_types: Vec::new(),
            indices: Vec::new(),
            uvs: Vec::new(),
            flow_directions: Vec::new(),
        };
    }

    let mut mesh_a = mesh_fluid_surface_nets(grid, boundary);
    let mesh_b = mesh_fluid_mc(grid, boundary);

    // Merge mesh_b into mesh_a
    let offset = mesh_a.positions.len() as u32;
    mesh_a.positions.extend_from_slice(&mesh_b.positions);
    mesh_a.normals.extend_from_slice(&mesh_b.normals);
    mesh_a.fluid_types.extend_from_slice(&mesh_b.fluid_types);
    mesh_a.uvs.extend_from_slice(&mesh_b.uvs);
    mesh_a
        .flow_directions
        .extend_from_slice(&mesh_b.flow_directions);
    for &idx in &mesh_b.indices {
        mesh_a.indices.push(idx + offset);
    }

    mesh_a
}

/// Check if the cell above (x,y+1,z) is non-solid and has real fluid.
/// Used to detect floor-adjacent cells that should be extended.
#[inline]
fn has_fluid_above(grid: &ChunkFluidGrid, x: usize, y: usize, z: usize) -> bool {
    let size = grid.size;
    if y + 1 >= size {
        return false;
    }
    grid.grid_point_density(x, y + 1, z) <= 0.0 && grid.get(x, y + 1, z).level >= ISO_LEVEL
}

/// Sample the scalar field for MC meshing.
/// Returns fluid level for air cells. Solid cells return 1.0 (treated as "inside")
/// so no isosurface forms at rock/fluid boundaries — only at fluid/air boundaries.
/// Floor extension: non-solid cells with low fluid sitting on solid rock (or at chunk
/// bottom boundary) with fluid above get boosted to 1.0 to close the visual gap.
/// Out-of-bounds coordinates return BOUNDARY_FIELD to close mesh at chunk edges.
#[inline]
fn sample_field(grid: &ChunkFluidGrid, x: usize, y: usize, z: usize, boundary: &BoundaryLevels) -> f32 {
    let size = grid.size;
    if x >= size || y >= size || z >= size {
        // Pre-baked boundary value includes density + floor extension
        if let Some(level) = boundary.get_level(x, y, z) {
            return level;
        }
        // Fallback: density check at boundary coords
        if grid.grid_point_density(x, y, z) > 0.0 {
            return 1.0;
        }
        return BOUNDARY_FIELD;
    }
    if grid.grid_point_density(x, y, z) > 0.0 {
        1.0 // inside — prevents surface at rock/fluid boundary
    } else {
        let level = grid.get(x, y, z).level;
        // Floor extension: low-fluid cell on solid rock (or chunk bottom) with fluid above
        if level < ISO_LEVEL && (y == 0 || grid.grid_point_density(x, y - 1, z) > 0.0) && has_fluid_above(grid, x, y, z) {
            1.0 // boost to close floor gap
        } else {
            level
        }
    }
}

/// Sample the SDF for Surface Nets: ISO_LEVEL - fluid_level.
/// Negative = fluid present, positive = air only.
/// Solid cells return -1.0 (treated as "inside") so no isosurface at rock/fluid boundary.
/// Floor extension: non-solid cells with low fluid on solid rock (or at chunk bottom)
/// with fluid above get boosted to -1.0 to close the visual gap.
/// Out-of-bounds coordinates return BOUNDARY_SDF to close mesh at chunk edges.
#[inline]
fn sample_sdf(grid: &ChunkFluidGrid, x: usize, y: usize, z: usize, boundary: &BoundaryLevels) -> f32 {
    let size = grid.size;
    if x >= size || y >= size || z >= size {
        // Pre-baked boundary value includes density + floor extension (convert to SDF)
        if let Some(level) = boundary.get_level(x, y, z) {
            return ISO_LEVEL - level;
        }
        // Fallback: density check at boundary coords
        if grid.grid_point_density(x, y, z) > 0.0 {
            return -1.0;
        }
        return BOUNDARY_SDF;
    }
    if grid.grid_point_density(x, y, z) > 0.0 {
        -1.0 // inside — prevents surface at rock/fluid boundary
    } else {
        let level = grid.get(x, y, z).level;
        // Floor extension: low-fluid cell on solid rock (or chunk bottom) with fluid above
        if level < ISO_LEVEL && (y == 0 || grid.grid_point_density(x, y - 1, z) > 0.0) && has_fluid_above(grid, x, y, z) {
            -1.0 // boost to close floor gap
        } else {
            ISO_LEVEL - level
        }
    }
}

/// Returns true if any in-bounds corner of the unit cube at (x,y,z) has real fluid,
/// or if any corner is a floor extension cell (non-solid, low fluid, on solid rock
/// with fluid above). Used to skip cubes in pure rock/air regions — without this,
/// treating solid as "inside" would generate phantom water surfaces on every cave wall.
/// Out-of-bounds corners are treated as having no fluid.
#[inline]
fn cube_has_fluid(grid: &ChunkFluidGrid, x: usize, y: usize, z: usize, boundary: &BoundaryLevels) -> bool {
    let size = grid.size;
    for dz in 0..=1usize {
        for dy in 0..=1usize {
            for dx in 0..=1usize {
                let cx = x + dx;
                let cy = y + dy;
                let cz = z + dz;
                if cx >= size || cy >= size || cz >= size {
                    // Multi-axis overflow: skip
                    let overflows = (cx >= size) as u8 + (cy >= size) as u8 + (cz >= size) as u8;
                    if overflows > 1 {
                        continue;
                    }
                    // Single-axis: check pre-baked has_fluid flag
                    if let Some(has) = boundary.get_has_fluid(cx, cy, cz) {
                        if has {
                            return true;
                        }
                    }
                    continue;
                }
                if !(grid.grid_point_density(cx, cy, cz) > 0.0) {
                    let level = grid.get(cx, cy, cz).level;
                    if level >= MIN_LEVEL {
                        return true;
                    }
                    // Floor extension: low-fluid cell on solid rock (or chunk bottom) with fluid above
                    if level < ISO_LEVEL && (cy == 0 || grid.grid_point_density(cx, cy - 1, cz) > 0.0) && has_fluid_above(grid, cx, cy, cz) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Marching Cubes fluid mesher.
/// Produces triangulated isosurface at ISO_LEVEL with fluid_type=4 (purple).
/// All positions offset by +MC_Y_OFFSET in Y to sit above Surface Nets mesh.
fn mesh_fluid_mc(grid: &ChunkFluidGrid, boundary: &BoundaryLevels) -> FluidMeshData {
    let size = grid.size;
    let mut mesh = FluidMeshData {
        positions: Vec::new(),
        normals: Vec::new(),
        fluid_types: Vec::new(),
        indices: Vec::new(),
        uvs: Vec::new(),
        flow_directions: Vec::new(),
    };

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                // Skip cubes with no actual fluid — prevents phantom surfaces on cave walls
                if !cube_has_fluid(grid, x, y, z, boundary) {
                    continue;
                }

                // Sample 8 corners using Paul Bourke ordering (CORNER_OFFSETS)
                let mut corner_vals = [0.0f32; 8];
                for (i, off) in CORNER_OFFSETS.iter().enumerate() {
                    corner_vals[i] = sample_field(grid, x + off[0], y + off[1], z + off[2], boundary);
                }

                // Build cube index: bit i set if corner i >= ISO_LEVEL
                let mut cube_index: usize = 0;
                for i in 0..8 {
                    if corner_vals[i] >= ISO_LEVEL {
                        cube_index |= 1 << i;
                    }
                }

                let edge_mask = EDGE_TABLE[cube_index];
                if edge_mask == 0 {
                    continue;
                }

                // Interpolate edge vertices at ISO_LEVEL crossings
                let mut edge_verts = [[0.0f32; 3]; 12];
                for e in 0..12 {
                    if edge_mask & (1 << e) != 0 {
                        let [c0, c1] = EDGE_VERTICES[e];
                        let v0 = corner_vals[c0];
                        let v1 = corner_vals[c1];
                        let t = if (v1 - v0).abs() > 1e-6 {
                            (ISO_LEVEL - v0) / (v1 - v0)
                        } else {
                            0.5
                        };
                        let t = t.clamp(0.0, 1.0);
                        let p0 = CORNER_OFFSETS[c0];
                        let p1 = CORNER_OFFSETS[c1];
                        edge_verts[e] = [
                            x as f32 + p0[0] as f32 + (p1[0] as f32 - p0[0] as f32) * t,
                            y as f32 + p0[1] as f32 + (p1[1] as f32 - p0[1] as f32) * t
                                + MC_Y_OFFSET,
                            z as f32 + p0[2] as f32 + (p1[2] as f32 - p0[2] as f32) * t,
                        ];
                    }
                }

                // Emit triangles from TRI_TABLE
                let tri_row = &TRI_TABLE[cube_index];
                let flow = compute_flow_direction(grid, x, y, z);
                let mut i = 0;
                while i < 15 && tri_row[i] >= 0 {
                    let v0 = edge_verts[tri_row[i] as usize];
                    let v1 = edge_verts[tri_row[i + 1] as usize];
                    let v2 = edge_verts[tri_row[i + 2] as usize];

                    // Cross-product normal, skip degenerate triangles
                    let ax = v1[0] - v0[0];
                    let ay = v1[1] - v0[1];
                    let az = v1[2] - v0[2];
                    let bx = v2[0] - v0[0];
                    let by = v2[1] - v0[1];
                    let bz = v2[2] - v0[2];
                    let nx = ay * bz - az * by;
                    let ny = az * bx - ax * bz;
                    let nz = ax * by - ay * bx;
                    let len = (nx * nx + ny * ny + nz * nz).sqrt();

                    if len >= 1e-4 {
                        let normal = [nx / len, ny / len, nz / len];
                        let base = mesh.positions.len() as u32;

                        for &v in &[v0, v1, v2] {
                            mesh.positions.push(v);
                            mesh.normals.push(normal);
                            mesh.fluid_types.push(MC_FLUID_TYPE);
                            mesh.uvs.push([v[0], v[2]]);
                            mesh.flow_directions.push(flow);
                        }

                        mesh.indices.push(base);
                        mesh.indices.push(base + 1);
                        mesh.indices.push(base + 2);
                    }

                    i += 3;
                }
            }
        }
    }

    mesh
}

/// Naive Surface Nets fluid mesher.
/// Produces smoothed isosurface at ISO_LEVEL with fluid_type=1 (teal).
fn mesh_fluid_surface_nets(grid: &ChunkFluidGrid, boundary: &BoundaryLevels) -> FluidMeshData {
    let size = grid.size;
    let mut mesh = FluidMeshData {
        positions: Vec::new(),
        normals: Vec::new(),
        fluid_types: Vec::new(),
        indices: Vec::new(),
        uvs: Vec::new(),
        flow_directions: Vec::new(),
    };

    // Precompute SDF field: (size+1)^3 samples
    let fs = size + 1; // field_size
    let mut sdf = vec![0.0f32; fs * fs * fs];
    for z in 0..fs {
        for y in 0..fs {
            for x in 0..fs {
                sdf[z * fs * fs + y * fs + x] = sample_sdf(grid, x, y, z, boundary);
            }
        }
    }

    // cell_vertex_map: vertex index for each cell, or -1 if none
    let mut cell_vertex: Vec<i32> = vec![-1; size * size * size];

    // Pass 1: Place a vertex in each cell that has mixed-sign corners
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                // Skip cells with no actual fluid — prevents phantom surfaces on cave walls
                if !cube_has_fluid(grid, x, y, z, boundary) {
                    continue;
                }

                // Sample 8 corners using binary encoding (SN_CORNERS)
                let mut corner_sdf = [0.0f32; 8];
                for (i, corner) in SN_CORNERS.iter().enumerate() {
                    let sx = x + corner[0] as usize;
                    let sy = y + corner[1] as usize;
                    let sz = z + corner[2] as usize;
                    corner_sdf[i] = sdf[sz * fs * fs + sy * fs + sx];
                }

                // Skip if all corners have the same sign
                let all_pos = corner_sdf.iter().all(|&v| v > 0.0);
                let all_neg = corner_sdf.iter().all(|&v| v <= 0.0);
                if all_pos || all_neg {
                    continue;
                }

                // Average all edge-crossing positions to find vertex
                let mut sum = [0.0f32; 3];
                let mut count = 0u32;

                for &[c0, c1] in &CUBE_EDGES {
                    let v0 = corner_sdf[c0];
                    let v1 = corner_sdf[c1];
                    if (v0 > 0.0) != (v1 > 0.0) {
                        let t = if (v1 - v0).abs() > 1e-6 {
                            v0 / (v0 - v1)
                        } else {
                            0.5
                        };
                        let t = t.clamp(0.0, 1.0);
                        let p0 = SN_CORNERS[c0];
                        let p1 = SN_CORNERS[c1];
                        sum[0] += p0[0] + (p1[0] - p0[0]) * t;
                        sum[1] += p0[1] + (p1[1] - p0[1]) * t;
                        sum[2] += p0[2] + (p1[2] - p0[2]) * t;
                        count += 1;
                    }
                }

                if count == 0 {
                    continue;
                }

                let inv = 1.0 / count as f32;
                let lp = [sum[0] * inv, sum[1] * inv, sum[2] * inv]; // local position in [0,1]^3
                let pos = [x as f32 + lp[0], y as f32 + lp[1], z as f32 + lp[2]];

                // Smooth normal via SDF gradient (bilinear interpolation of corner differences)
                let normal = sdf_gradient(&corner_sdf, lp);
                let flow = compute_flow_direction(grid, x, y, z);

                let vert_idx = mesh.positions.len() as i32;
                cell_vertex[z * size * size + y * size + x] = vert_idx;

                mesh.positions.push(pos);
                mesh.normals.push(normal);
                mesh.fluid_types.push(SN_FLUID_TYPE);
                mesh.uvs.push([pos[0], pos[2]]);
                mesh.flow_directions.push(flow);
            }
        }
    }

    // Pass 2: Generate quads from lattice edges with sign changes.
    // For each SDF grid edge that crosses the isosurface, emit a quad
    // connecting the 4 cells that share that edge (if all have vertices).

    // Helper to look up vertex index for a cell
    let cv = |x: usize, y: usize, z: usize| -> i32 { cell_vertex[z * size * size + y * size + x] };

    // X-edges: iterate y in [1, size-1], z in [1, size-1], x in [0, size-1]
    for z in 1..size {
        for y in 1..size {
            for x in 0..size {
                let s0 = sdf[z * fs * fs + y * fs + x];
                let s1 = sdf[z * fs * fs + y * fs + (x + 1)];
                if (s0 > 0.0) == (s1 > 0.0) {
                    continue;
                }
                // 4 cells sharing this X-edge
                let a = cv(x, y, z);
                let b = cv(x, y - 1, z);
                let c = cv(x, y, z - 1);
                let d = cv(x, y - 1, z - 1);
                if a < 0 || b < 0 || c < 0 || d < 0 {
                    continue;
                }
                // Winding: if s0 < 0 (fluid side), wind one way; else the other
                if s0 <= 0.0 {
                    emit_quad(&mut mesh, a as u32, b as u32, d as u32, c as u32);
                } else {
                    emit_quad(&mut mesh, a as u32, c as u32, d as u32, b as u32);
                }
            }
        }
    }

    // Y-edges: iterate x in [1, size-1], z in [1, size-1], y in [0, size-1]
    for z in 1..size {
        for y in 0..size {
            for x in 1..size {
                let s0 = sdf[z * fs * fs + y * fs + x];
                let s1 = sdf[z * fs * fs + (y + 1) * fs + x];
                if (s0 > 0.0) == (s1 > 0.0) {
                    continue;
                }
                let a = cv(x, y, z);
                let b = cv(x, y, z - 1);
                let c = cv(x - 1, y, z);
                let d = cv(x - 1, y, z - 1);
                if a < 0 || b < 0 || c < 0 || d < 0 {
                    continue;
                }
                if s0 <= 0.0 {
                    emit_quad(&mut mesh, a as u32, c as u32, d as u32, b as u32);
                } else {
                    emit_quad(&mut mesh, a as u32, b as u32, d as u32, c as u32);
                }
            }
        }
    }

    // Z-edges: iterate x in [1, size-1], y in [1, size-1], z in [0, size-1]
    for z in 0..size {
        for y in 1..size {
            for x in 1..size {
                let s0 = sdf[z * fs * fs + y * fs + x];
                let s1 = sdf[(z + 1) * fs * fs + y * fs + x];
                if (s0 > 0.0) == (s1 > 0.0) {
                    continue;
                }
                let a = cv(x, y, z);
                let b = cv(x - 1, y, z);
                let c = cv(x, y - 1, z);
                let d = cv(x - 1, y - 1, z);
                if a < 0 || b < 0 || c < 0 || d < 0 {
                    continue;
                }
                if s0 <= 0.0 {
                    emit_quad(&mut mesh, a as u32, b as u32, d as u32, c as u32);
                } else {
                    emit_quad(&mut mesh, a as u32, c as u32, d as u32, b as u32);
                }
            }
        }
    }

    mesh
}

/// Emit a quad as two triangles, split along the shorter diagonal.
fn emit_quad(mesh: &mut FluidMeshData, a: u32, b: u32, c: u32, d: u32) {
    let pa = mesh.positions[a as usize];
    let pb = mesh.positions[b as usize];
    let pc = mesh.positions[c as usize];
    let pd = mesh.positions[d as usize];

    // Split along shorter diagonal: AC vs BD
    let diag_ac = dist_sq(pa, pc);
    let diag_bd = dist_sq(pb, pd);

    if diag_ac <= diag_bd {
        // Split along AC: triangles ABC, ACD
        mesh.indices.extend_from_slice(&[a, b, c, a, c, d]);
    } else {
        // Split along BD: triangles ABD, BCD
        mesh.indices.extend_from_slice(&[a, b, d, b, c, d]);
    }
}

#[inline]
fn dist_sq(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// Compute SDF gradient at position `s` within a unit cube via bilinear interpolation.
/// `corners` are SDF values at the 8 corners (binary encoding: bit 0=X, bit 1=Y, bit 2=Z).
/// Returns normalized gradient vector pointing away from fluid (toward positive SDF).
fn sdf_gradient(corners: &[f32; 8], s: [f32; 3]) -> [f32; 3] {
    let sx = s[0];
    let sy = s[1];
    let sz = s[2];

    // X gradient: 4 parallel X-edges, bilinear interpolation
    let d00 = corners[0b001] - corners[0b000]; // (1,0,0) - (0,0,0)
    let d10 = corners[0b101] - corners[0b100]; // (1,0,1) - (0,0,1)
    let d01 = corners[0b011] - corners[0b010]; // (1,1,0) - (0,1,0)
    let d11 = corners[0b111] - corners[0b110]; // (1,1,1) - (0,1,1)
    let gx = (1.0 - sy) * (1.0 - sz) * d00
        + (1.0 - sy) * sz * d10
        + sy * (1.0 - sz) * d01
        + sy * sz * d11;

    // Y gradient: 4 parallel Y-edges
    let d00 = corners[0b010] - corners[0b000]; // (0,1,0) - (0,0,0)
    let d10 = corners[0b110] - corners[0b100]; // (0,1,1) - (0,0,1)
    let d01 = corners[0b011] - corners[0b001]; // (1,1,0) - (1,0,0)
    let d11 = corners[0b111] - corners[0b101]; // (1,1,1) - (1,0,1)
    let gy = (1.0 - sx) * (1.0 - sz) * d00
        + (1.0 - sx) * sz * d10
        + sx * (1.0 - sz) * d01
        + sx * sz * d11;

    // Z gradient: 4 parallel Z-edges
    let d00 = corners[0b100] - corners[0b000]; // (0,0,1) - (0,0,0)
    let d10 = corners[0b101] - corners[0b001]; // (1,0,1) - (1,0,0)
    let d01 = corners[0b110] - corners[0b010]; // (0,1,1) - (0,1,0)
    let d11 = corners[0b111] - corners[0b011]; // (1,1,1) - (1,1,0)
    let gz = (1.0 - sx) * (1.0 - sy) * d00
        + sx * (1.0 - sy) * d10
        + (1.0 - sx) * sy * d01
        + sx * sy * d11;

    let len = (gx * gx + gy * gy + gz * gz).sqrt();
    if len > 1e-6 {
        [gx / len, gy / len, gz / len]
    } else {
        [0.0, 1.0, 0.0] // default up normal
    }
}

/// Compute flow direction from fluid level gradient (for UV animation).
fn compute_flow_direction(grid: &ChunkFluidGrid, x: usize, y: usize, z: usize) -> [f32; 3] {
    let size = grid.size;
    let level = grid.get(x, y, z).level;

    let mut dx = 0.0f32;
    let mut dz = 0.0f32;

    // X gradient
    if x > 0 && grid.grid_point_density(x - 1, y, z) <= 0.0 {
        dx += level - grid.get(x - 1, y, z).level;
    }
    if x + 1 < size && grid.grid_point_density(x + 1, y, z) <= 0.0 {
        dx += grid.get(x + 1, y, z).level - level;
    }

    // Z gradient
    if z > 0 && grid.grid_point_density(x, y, z - 1) <= 0.0 {
        dz += level - grid.get(x, y, z - 1).level;
    }
    if z + 1 < size && grid.grid_point_density(x, y, z + 1) <= 0.0 {
        dz += grid.get(x, y, z + 1).level - level;
    }

    // Downward gradient (waterfall detection)
    let mut dy_mag = 0.0f32;
    if y > 0 && grid.grid_point_density(x, y - 1, z) <= 0.0 {
        let below_level = grid.get(x, y - 1, z).level;
        dy_mag = (level - below_level).max(0.0);
    }

    let horiz_mag = (dx * dx + dz * dz).sqrt();
    let total_mag = (horiz_mag * horiz_mag + dy_mag * dy_mag).sqrt().min(1.0);

    if total_mag < 0.001 {
        [0.0, 0.0, 0.0]
    } else {
        let inv = if horiz_mag > 0.001 {
            1.0 / horiz_mag
        } else {
            0.0
        };
        [dx * inv, dz * inv, total_mag]
    }
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

    fn make_fluid_grid(regions: &[(std::ops::Range<usize>, std::ops::Range<usize>, std::ops::Range<usize>, f32)]) -> ChunkFluidGrid {
        let mut grid = ChunkFluidGrid::new(16);
        for (xr, yr, zr, level) in regions {
            for z in zr.clone() {
                for y in yr.clone() {
                    for x in xr.clone() {
                        let cell = grid.get_mut(x, y, z);
                        cell.level = *level;
                        cell.fluid_type = FluidType::Water;
                    }
                }
            }
        }
        grid
    }

    fn no_boundary() -> BoundaryLevels {
        BoundaryLevels::empty(16)
    }

    #[test]
    fn empty_grid_produces_no_mesh() {
        let grid = ChunkFluidGrid::new(16);
        let mesh = mesh_fluid(&grid, &no_boundary());
        assert!(mesh.positions.is_empty());
        assert!(mesh.indices.is_empty());
    }

    #[test]
    fn single_source_produces_mesh() {
        let grid = make_fluid_grid(&[(4..8, 4..8, 4..8, 1.0)]);
        let mesh = mesh_fluid(&grid, &no_boundary());
        assert!(!mesh.positions.is_empty(), "Should produce mesh vertices");
        assert!(!mesh.indices.is_empty(), "Should produce mesh indices");
        assert_eq!(mesh.indices.len() % 3, 0, "Indices should be triangles");
        assert_eq!(mesh.uvs.len(), mesh.positions.len(), "UVs per vertex");
        assert_eq!(
            mesh.flow_directions.len(),
            mesh.positions.len(),
            "Flow per vertex"
        );
    }

    #[test]
    fn test_mc_produces_mesh() {
        let grid = make_fluid_grid(&[(6..10, 6..10, 6..10, 1.0)]);
        let mesh = mesh_fluid_mc(&grid, &no_boundary());
        assert!(!mesh.positions.is_empty(), "MC should produce vertices");
        assert!(!mesh.indices.is_empty(), "MC should produce indices");
        assert_eq!(mesh.indices.len() % 3, 0, "MC indices should be triangles");
        // All fluid types should be MC_FLUID_TYPE (4)
        for &ft in &mesh.fluid_types {
            assert_eq!(ft, MC_FLUID_TYPE, "MC fluid type should be 4");
        }
    }

    #[test]
    fn test_surface_nets_produces_mesh() {
        let grid = make_fluid_grid(&[(6..10, 6..10, 6..10, 1.0)]);
        let mesh = mesh_fluid_surface_nets(&grid, &no_boundary());
        assert!(!mesh.positions.is_empty(), "SN should produce vertices");
        assert!(!mesh.indices.is_empty(), "SN should produce indices");
        assert_eq!(mesh.indices.len() % 3, 0, "SN indices should be triangles");
        // All fluid types should be SN_FLUID_TYPE (1)
        for &ft in &mesh.fluid_types {
            assert_eq!(ft, SN_FLUID_TYPE, "SN fluid type should be 1");
        }
    }

    #[test]
    fn test_dual_mesher_both_present() {
        let grid = make_fluid_grid(&[(4..12, 4..12, 4..12, 1.0)]);
        let mesh = mesh_fluid(&grid, &no_boundary());
        let has_sn = mesh.fluid_types.iter().any(|&ft| ft == SN_FLUID_TYPE);
        let has_mc = mesh.fluid_types.iter().any(|&ft| ft == MC_FLUID_TYPE);
        assert!(has_sn, "Merged mesh should have Surface Nets (type 1) vertices");
        assert!(has_mc, "Merged mesh should have Marching Cubes (type 4) vertices");
    }

    #[test]
    fn test_mc_no_mesh_for_empty() {
        let grid = ChunkFluidGrid::new(16);
        let mesh = mesh_fluid_mc(&grid, &no_boundary());
        assert!(mesh.positions.is_empty(), "Empty grid should produce no MC geometry");
        assert!(mesh.indices.is_empty());
    }

    #[test]
    fn test_sn_no_mesh_for_empty() {
        let grid = ChunkFluidGrid::new(16);
        let mesh = mesh_fluid_surface_nets(&grid, &no_boundary());
        assert!(mesh.positions.is_empty(), "Empty grid should produce no SN geometry");
        assert!(mesh.indices.is_empty());
    }

    #[test]
    fn test_sn_flat_pool_is_flat() {
        // Uniform fluid level across a large area should produce a flat top surface
        let grid = make_fluid_grid(&[(2..14, 2..6, 2..14, 1.0)]);
        let mesh = mesh_fluid_surface_nets(&grid, &no_boundary());
        assert!(!mesh.positions.is_empty(), "Flat pool should produce SN vertices");

        // SN generates a full 3D isosurface (top, bottom, sides).
        // Check that the TOP surface vertices (highest Y region) are flat.
        let max_y = mesh.positions.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max);
        let top_ys: Vec<f32> = mesh
            .positions
            .iter()
            .filter(|p| p[1] > max_y - 0.1)
            .map(|p| p[1])
            .collect();
        assert!(!top_ys.is_empty(), "Should have top-surface vertices");
        let top_min = top_ys.iter().cloned().fold(f32::INFINITY, f32::min);
        let top_max = top_ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (top_max - top_min) < 0.01,
            "Top surface should be flat, got Y range {:.3} to {:.3}",
            top_min,
            top_max
        );
    }

    #[test]
    fn test_mc_y_offset_applied() {
        let grid = make_fluid_grid(&[(6..10, 6..10, 6..10, 1.0)]);
        let mc_mesh = mesh_fluid_mc(&grid, &no_boundary());
        // MC positions should have the Y offset applied
        // At minimum, some vertex Y values should differ from integer positions by MC_Y_OFFSET
        assert!(
            mc_mesh.positions.iter().any(|p| {
                let frac = p[1] - p[1].floor();
                (frac - MC_Y_OFFSET).abs() < 0.01 || frac > 0.01
            }),
            "MC mesh should have Y offset applied"
        );
    }

    #[test]
    fn test_solid_no_phantom_surface() {
        // A grid with solid rock and air but NO fluid should produce no mesh at all.
        // This verifies cube_has_fluid() prevents phantom surfaces at cave walls.
        let mut grid = ChunkFluidGrid::new(16);
        // Make bottom half solid, top half air
        for z in 0..16 {
            for y in 0..8 {
                for x in 0..16 {
                    grid.set_density(x, y, z, 1.0); // solid
                }
            }
        }
        let mesh = mesh_fluid(&grid, &no_boundary());
        assert!(
            mesh.positions.is_empty(),
            "Solid/air boundary with no fluid should produce no mesh, got {} verts",
            mesh.positions.len()
        );
    }

    #[test]
    fn test_fluid_on_rock_floor_not_floating() {
        // Fluid sitting on a solid floor should produce mesh.
        // With full fluid levels at the boundary, floor extension doesn't trigger
        // (only triggers for low-fluid transition cells).
        let mut grid = ChunkFluidGrid::new(16);
        // Solid floor at y=0..4
        for z in 4..12 {
            for y in 0..4 {
                for x in 4..12 {
                    grid.set_density(x, y, z, 1.0);
                }
            }
        }
        // Fluid at y=4..6 (sitting on the solid floor, full level)
        for z in 4..12 {
            for y in 4..6 {
                for x in 4..12 {
                    let cell = grid.get_mut(x, y, z);
                    cell.level = 1.0;
                    cell.fluid_type = FluidType::Water;
                }
            }
        }
        let mesh = mesh_fluid(&grid, &no_boundary());
        assert!(!mesh.positions.is_empty(), "Should produce fluid mesh");

        // The minimum Y of vertices should be near the solid/fluid boundary (y≈3-4).
        let min_y = mesh.positions.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
        assert!(
            min_y >= 2.9,
            "Mesh should not extend deep into solid rock, min_y = {:.2}",
            min_y
        );
    }

    #[test]
    fn test_floor_extension_mc() {
        // Production scenario: solid floor, transition cell with near-zero fluid,
        // then real fluid above. MC mesh should extend down through the transition cell.
        let mut grid = ChunkFluidGrid::new(16);
        // Solid floor at y=0..4
        for z in 4..12 {
            for y in 0..4 {
                for x in 4..12 {
                    grid.set_density(x, y, z, 1.0);
                }
            }
        }
        // Transition cell at y=4: near-zero fluid (mimics averaged density boundary)
        for z in 4..12 {
            for x in 4..12 {
                let cell = grid.get_mut(x, 4, z);
                cell.level = 0.02; // below ISO_LEVEL
                cell.fluid_type = FluidType::Water;
            }
        }
        // Real fluid at y=5..8
        for z in 4..12 {
            for y in 5..8 {
                for x in 4..12 {
                    let cell = grid.get_mut(x, y, z);
                    cell.level = 1.0;
                    cell.fluid_type = FluidType::Water;
                }
            }
        }
        let mesh = mesh_fluid_mc(&grid, &no_boundary());
        assert!(!mesh.positions.is_empty(), "MC should produce vertices");
        let min_y = mesh.positions.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
        // Without floor extension, mesh bottom would be at y≈4.9 (above transition cell).
        // With extension, transition cell gets boosted, so mesh extends down to y≈3-4.
        assert!(
            min_y < 4.9,
            "MC mesh should extend below the transition cell gap, min_y = {:.2}",
            min_y
        );
    }

    #[test]
    fn test_floor_extension_sn() {
        // Same production scenario for Surface Nets.
        let mut grid = ChunkFluidGrid::new(16);
        // Solid floor at y=0..4
        for z in 4..12 {
            for y in 0..4 {
                for x in 4..12 {
                    grid.set_density(x, y, z, 1.0);
                }
            }
        }
        // Transition cell at y=4: near-zero fluid
        for z in 4..12 {
            for x in 4..12 {
                let cell = grid.get_mut(x, 4, z);
                cell.level = 0.02;
                cell.fluid_type = FluidType::Water;
            }
        }
        // Real fluid at y=5..8
        for z in 4..12 {
            for y in 5..8 {
                for x in 4..12 {
                    let cell = grid.get_mut(x, y, z);
                    cell.level = 1.0;
                    cell.fluid_type = FluidType::Water;
                }
            }
        }
        let mesh = mesh_fluid_surface_nets(&grid, &no_boundary());
        assert!(!mesh.positions.is_empty(), "SN should produce vertices");
        let min_y = mesh.positions.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
        // SN should also extend below the transition cell
        assert!(
            min_y < 4.9,
            "SN mesh should extend below the transition cell gap, min_y = {:.2}",
            min_y
        );
    }

    #[test]
    fn test_no_phantom_walls() {
        // Solid cells beside (not below) fluid should NOT generate phantom mesh.
        // Wall of solid rock next to fluid, with no solid floor below the fluid.
        let mut grid = ChunkFluidGrid::new(16);
        // Solid wall at x=0..4, y=4..12, z=4..12
        for z in 4..12 {
            for y in 4..12 {
                for x in 0..4 {
                    grid.set_density(x, y, z, 1.0);
                }
            }
        }
        // Fluid at x=4..8, y=4..8, z=4..8 (floating in air, next to wall)
        for z in 4..8 {
            for y in 4..8 {
                for x in 4..8 {
                    let cell = grid.get_mut(x, y, z);
                    cell.level = 1.0;
                    cell.fluid_type = FluidType::Water;
                }
            }
        }
        let mesh = mesh_fluid(&grid, &no_boundary());
        assert!(!mesh.positions.is_empty(), "Should produce fluid mesh");
        // The mesh should not generate phantom water on the wall face.
        // Solid wall cells return 1.0 ("inside"), same as fluid cells,
        // so no isosurface forms at the wall/fluid boundary.
        // Vertices can approach x=4 (the boundary) but should not go into x<3.
        let min_x = mesh.positions.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
        assert!(
            min_x >= 3.0,
            "Mesh should not extend deep into solid wall, min_x = {:.2}",
            min_x
        );
    }

    #[test]
    fn test_boundary_levels_seamless() {
        // Two adjacent fluid grids: mesh with boundary levels should not seal at the shared edge.
        // Grid A has fluid at x=12..16, Grid B (the +X neighbor) has fluid at x=0..4.
        // Without boundary data, mesh seals at x=16. With it, the surface should be open.
        let grid_a = make_fluid_grid(&[(12..16, 4..8, 4..8, 1.0)]);

        // Build boundary levels from the neighbor's x=0 face
        let mut boundary = BoundaryLevels::empty(16);
        let mut pos_x_levels = vec![0.0f32; 16 * 16];
        let mut pos_x_fluid = vec![false; 16 * 16];
        for z in 4..8 {
            for y in 4..8 {
                pos_x_levels[z * 16 + y] = 1.0; // neighbor has fluid at x=0
                pos_x_fluid[z * 16 + y] = true;
            }
        }
        boundary.pos_x = Some(pos_x_levels);
        boundary.pos_x_fluid = Some(pos_x_fluid);

        let mesh_with_boundary = mesh_fluid(&grid_a, &boundary);
        let mesh_without_boundary = mesh_fluid(&grid_a, &no_boundary());

        // With boundary data, the mesh should have fewer vertices at the +X face
        // (the surface is open there instead of sealed). Alternatively, the vertex count
        // should differ, indicating the boundary data changed the mesh.
        assert!(
            mesh_with_boundary.positions.len() != mesh_without_boundary.positions.len(),
            "Boundary levels should change the mesh at the shared edge (with={}, without={})",
            mesh_with_boundary.positions.len(),
            mesh_without_boundary.positions.len()
        );
    }
}
