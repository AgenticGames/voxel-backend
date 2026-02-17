use crate::cell::{ChunkFluidGrid, FluidType, MIN_LEVEL};
use crate::tables::{CORNER_OFFSETS, EDGE_TABLE, EDGE_VERTICES, TRI_TABLE};

/// A fluid mesh produced by marching cubes.
pub struct FluidMeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub fluid_types: Vec<u8>,
    pub indices: Vec<u32>,
}

/// Build a marching cubes mesh from a fluid grid.
///
/// Uses the fluid level as the scalar field. The iso-level is `ISO_LEVEL` (0.0625).
/// Cells that are solid in the solid_mask are skipped.
/// The resulting positions are in local chunk space (0..chunk_size).
pub fn mesh_fluid(grid: &ChunkFluidGrid) -> FluidMeshData {
    let size = grid.size;
    let mut mesh = FluidMeshData {
        positions: Vec::new(),
        normals: Vec::new(),
        fluid_types: Vec::new(),
        indices: Vec::new(),
    };

    if size < 2 {
        return mesh;
    }

    // Build a scalar field of fluid levels, sized (size+1)^3 so we have corners
    // for every cell. We use the fluid levels directly; solid cells get 0.
    let field = |x: usize, y: usize, z: usize| -> f32 {
        if x >= size || y >= size || z >= size {
            return 0.0;
        }
        if grid.is_solid(x, y, z) {
            return 0.0;
        }
        grid.get(x, y, z).level
    };

    // March through cells (size-1 cubes along each axis)
    for z in 0..size - 1 {
        for y in 0..size - 1 {
            for x in 0..size - 1 {
                // Skip if entirely inside solid
                let any_non_solid = CORNER_OFFSETS.iter().any(|c| {
                    let cx = x + c[0];
                    let cy = y + c[1];
                    let cz = z + c[2];
                    cx < size && cy < size && cz < size && !grid.is_solid(cx, cy, cz)
                });
                if !any_non_solid {
                    continue;
                }

                // Sample 8 corners
                let mut corners = [0.0f32; 8];
                for (i, c) in CORNER_OFFSETS.iter().enumerate() {
                    corners[i] = field(x + c[0], y + c[1], z + c[2]);
                }

                // Build cube index
                let mut cube_index: u8 = 0;
                for i in 0..8 {
                    if corners[i] >= ISO_LEVEL {
                        cube_index |= 1 << i;
                    }
                }

                let edge_bits = EDGE_TABLE[cube_index as usize];
                if edge_bits == 0 {
                    continue;
                }

                // Determine dominant fluid type for this cell
                let dominant_type = dominant_fluid_type(grid, x, y, z);

                // Interpolate edge vertices
                let mut edge_verts = [[0.0f32; 3]; 12];
                for e in 0..12 {
                    if edge_bits & (1 << e) != 0 {
                        let [v0, v1] = EDGE_VERTICES[e];
                        let c0 = CORNER_OFFSETS[v0];
                        let c1 = CORNER_OFFSETS[v1];
                        let val0 = corners[v0];
                        let val1 = corners[v1];
                        let t = if (val1 - val0).abs() < 1e-6 {
                            0.5
                        } else {
                            (ISO_LEVEL - val0) / (val1 - val0)
                        };
                        let t = t.clamp(0.0, 1.0);
                        edge_verts[e] = [
                            x as f32 + c0[0] as f32 + (c1[0] as f32 - c0[0] as f32) * t,
                            y as f32 + c0[1] as f32 + (c1[1] as f32 - c0[1] as f32) * t,
                            z as f32 + c0[2] as f32 + (c1[2] as f32 - c0[2] as f32) * t,
                        ];
                    }
                }

                // Emit triangles
                let tri_row = &TRI_TABLE[cube_index as usize];
                let mut i = 0;
                while i < 16 && tri_row[i] >= 0 {
                    let e0 = tri_row[i] as usize;
                    let e1 = tri_row[i + 1] as usize;
                    let e2 = tri_row[i + 2] as usize;

                    let p0 = edge_verts[e0];
                    let p1 = edge_verts[e1];
                    let p2 = edge_verts[e2];

                    // Compute face normal
                    let u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
                    let v = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
                    let mut n = [
                        u[1] * v[2] - u[2] * v[1],
                        u[2] * v[0] - u[0] * v[2],
                        u[0] * v[1] - u[1] * v[0],
                    ];
                    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                    if len > 1e-8 {
                        n[0] /= len;
                        n[1] /= len;
                        n[2] /= len;
                    }

                    let base = mesh.positions.len() as u32;
                    mesh.positions.push(p0);
                    mesh.positions.push(p1);
                    mesh.positions.push(p2);
                    mesh.normals.push(n);
                    mesh.normals.push(n);
                    mesh.normals.push(n);
                    mesh.fluid_types.push(dominant_type as u8);
                    mesh.fluid_types.push(dominant_type as u8);
                    mesh.fluid_types.push(dominant_type as u8);
                    mesh.indices.push(base);
                    mesh.indices.push(base + 1);
                    mesh.indices.push(base + 2);

                    i += 3;
                }
            }
        }
    }

    mesh
}

/// Iso-level for fluid surface extraction.
const ISO_LEVEL: f32 = 0.0625;

/// Determine the dominant fluid type among non-empty neighboring cells.
fn dominant_fluid_type(grid: &ChunkFluidGrid, x: usize, y: usize, z: usize) -> FluidType {
    let mut water = 0;
    let mut lava = 0;
    let size = grid.size;

    for cz in z..=(z + 1).min(size - 1) {
        for cy in y..=(y + 1).min(size - 1) {
            for cx in x..=(x + 1).min(size - 1) {
                let cell = grid.get(cx, cy, cz);
                if cell.level >= MIN_LEVEL {
                    match cell.fluid_type {
                        FluidType::Water => water += 1,
                        FluidType::Lava => lava += 1,
                    }
                }
            }
        }
    }

    if lava > water {
        FluidType::Lava
    } else {
        FluidType::Water
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
    }
}
