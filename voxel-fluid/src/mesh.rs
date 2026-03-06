use crate::cell::{ChunkFluidGrid, FluidType, MIN_LEVEL};
use crate::tables::{CORNER_OFFSETS, EDGE_TABLE, EDGE_VERTICES, TRI_TABLE};

/// Isosurface threshold for fluid meshing.
const ISO_LEVEL: f32 = 0.15;
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
pub fn mesh_fluid(grid: &ChunkFluidGrid) -> FluidMeshData {
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

    let mut mesh_a = mesh_fluid_surface_nets(grid);
    let mesh_b = mesh_fluid_mc(grid);

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

/// Sample the scalar field for MC meshing.
/// Returns fluid level for air cells. Solid cells return 1.0 (treated as "inside")
/// so no isosurface forms at rock/fluid boundaries — only at fluid/air boundaries.
/// Boundary-clamped to prevent chunk-edge holes.
#[inline]
fn sample_field(grid: &ChunkFluidGrid, x: usize, y: usize, z: usize) -> f32 {
    let size = grid.size;
    let cx = x.min(size - 1);
    let cy = y.min(size - 1);
    let cz = z.min(size - 1);
    if grid.is_solid(cx, cy, cz) {
        1.0 // inside — prevents surface at rock/fluid boundary
    } else {
        grid.get(cx, cy, cz).level
    }
}

/// Sample the SDF for Surface Nets: ISO_LEVEL - fluid_level.
/// Negative = fluid present, positive = air only.
/// Solid cells return -1.0 (treated as "inside") so no isosurface at rock/fluid boundary.
#[inline]
fn sample_sdf(grid: &ChunkFluidGrid, x: usize, y: usize, z: usize) -> f32 {
    let size = grid.size;
    let cx = x.min(size - 1);
    let cy = y.min(size - 1);
    let cz = z.min(size - 1);
    if grid.is_solid(cx, cy, cz) {
        -1.0 // inside — prevents surface at rock/fluid boundary
    } else {
        ISO_LEVEL - grid.get(cx, cy, cz).level
    }
}

/// Returns true if any corner of the unit cube at (x,y,z) has real fluid.
/// Used to skip cubes in pure rock/air regions — without this, treating solid as
/// "inside" would generate phantom water surfaces on every cave wall.
#[inline]
fn cube_has_fluid(grid: &ChunkFluidGrid, x: usize, y: usize, z: usize) -> bool {
    let size = grid.size;
    for dz in 0..=1usize {
        for dy in 0..=1usize {
            for dx in 0..=1usize {
                let cx = (x + dx).min(size - 1);
                let cy = (y + dy).min(size - 1);
                let cz = (z + dz).min(size - 1);
                if !grid.is_solid(cx, cy, cz) && grid.get(cx, cy, cz).level >= MIN_LEVEL {
                    return true;
                }
            }
        }
    }
    false
}

/// Marching Cubes fluid mesher.
/// Produces triangulated isosurface at ISO_LEVEL with fluid_type=4 (purple).
/// All positions offset by +MC_Y_OFFSET in Y to sit above Surface Nets mesh.
fn mesh_fluid_mc(grid: &ChunkFluidGrid) -> FluidMeshData {
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
                if !cube_has_fluid(grid, x, y, z) {
                    continue;
                }

                // Sample 8 corners using Paul Bourke ordering (CORNER_OFFSETS)
                let mut corner_vals = [0.0f32; 8];
                for (i, off) in CORNER_OFFSETS.iter().enumerate() {
                    corner_vals[i] = sample_field(grid, x + off[0], y + off[1], z + off[2]);
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
fn mesh_fluid_surface_nets(grid: &ChunkFluidGrid) -> FluidMeshData {
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
                sdf[z * fs * fs + y * fs + x] = sample_sdf(grid, x, y, z);
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
                if !cube_has_fluid(grid, x, y, z) {
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

    // Downward gradient (waterfall detection)
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

    #[test]
    fn empty_grid_produces_no_mesh() {
        let grid = ChunkFluidGrid::new(16);
        let mesh = mesh_fluid(&grid);
        assert!(mesh.positions.is_empty());
        assert!(mesh.indices.is_empty());
    }

    #[test]
    fn single_source_produces_mesh() {
        let grid = make_fluid_grid(&[(4..8, 4..8, 4..8, 1.0)]);
        let mesh = mesh_fluid(&grid);
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
        let mesh = mesh_fluid_mc(&grid);
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
        let mesh = mesh_fluid_surface_nets(&grid);
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
        let mesh = mesh_fluid(&grid);
        let has_sn = mesh.fluid_types.iter().any(|&ft| ft == SN_FLUID_TYPE);
        let has_mc = mesh.fluid_types.iter().any(|&ft| ft == MC_FLUID_TYPE);
        assert!(has_sn, "Merged mesh should have Surface Nets (type 1) vertices");
        assert!(has_mc, "Merged mesh should have Marching Cubes (type 4) vertices");
    }

    #[test]
    fn test_mc_no_mesh_for_empty() {
        let grid = ChunkFluidGrid::new(16);
        let mesh = mesh_fluid_mc(&grid);
        assert!(mesh.positions.is_empty(), "Empty grid should produce no MC geometry");
        assert!(mesh.indices.is_empty());
    }

    #[test]
    fn test_sn_no_mesh_for_empty() {
        let grid = ChunkFluidGrid::new(16);
        let mesh = mesh_fluid_surface_nets(&grid);
        assert!(mesh.positions.is_empty(), "Empty grid should produce no SN geometry");
        assert!(mesh.indices.is_empty());
    }

    #[test]
    fn test_sn_flat_pool_is_flat() {
        // Uniform fluid level across a large area should produce a flat top surface
        let grid = make_fluid_grid(&[(2..14, 2..6, 2..14, 1.0)]);
        let mesh = mesh_fluid_surface_nets(&grid);
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
        let mc_mesh = mesh_fluid_mc(&grid);
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
        let mesh = mesh_fluid(&grid);
        assert!(
            mesh.positions.is_empty(),
            "Solid/air boundary with no fluid should produce no mesh, got {} verts",
            mesh.positions.len()
        );
    }

    #[test]
    fn test_fluid_on_rock_floor_not_floating() {
        // Fluid sitting on a solid floor should produce a surface only at the top (air boundary).
        // No surface should form at the bottom (rock boundary).
        let mut grid = ChunkFluidGrid::new(16);
        // Solid floor at y=0..4
        for z in 4..12 {
            for y in 0..4 {
                for x in 4..12 {
                    grid.set_density(x, y, z, 1.0);
                }
            }
        }
        // Fluid at y=4..6 (sitting on the solid floor)
        for z in 4..12 {
            for y in 4..6 {
                for x in 4..12 {
                    let cell = grid.get_mut(x, y, z);
                    cell.level = 1.0;
                    cell.fluid_type = FluidType::Water;
                }
            }
        }
        let mesh = mesh_fluid(&grid);
        assert!(!mesh.positions.is_empty(), "Should produce fluid mesh");

        // The minimum Y of vertices should be near the solid/fluid boundary (y≈3-4),
        // NOT deep inside the solid (y<3). The isosurface wraps at corners where
        // solid/fluid/air meet, so vertices can dip to y≈3 at boundary cubes.
        // With solid treated as "inside", no surface forms at the rock/fluid interface.
        let min_y = mesh.positions.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
        assert!(
            min_y >= 2.9,
            "Mesh should not extend deep into solid rock, min_y = {:.2}",
            min_y
        );
    }
}
