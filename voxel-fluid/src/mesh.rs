use crate::cell::{ChunkFluidGrid, FluidType, MIN_LEVEL};
use crate::tables::{CORNER_OFFSETS, EDGE_TABLE, EDGE_VERTICES, TRI_TABLE};

/// Isosurface threshold for fluid meshing.
const ISO_LEVEL: f32 = 0.15;
/// Tiny SDF value for out-of-bounds samples — places boundary faces near chunk edge.
const BOUNDARY_SDF: f32 = 0.001;
/// Field value for out-of-bounds samples — just below ISO_LEVEL so MC places face at edge.
const BOUNDARY_FIELD: f32 = ISO_LEVEL - BOUNDARY_SDF;
/// Fluid levels from neighboring chunks at the 3 positive boundary faces.
/// Used to create seamless mesh at chunk edges instead of sealing the isosurface.
pub struct BoundaryLevels {
    /// Neighbor's x=0 face fluid levels, indexed [z * size + y]. size*size values.
    pub pos_x: Option<Vec<f32>>,
    /// Neighbor's y=0 face fluid levels, indexed [z * size + x]. size*size values.
    pub pos_y: Option<Vec<f32>>,
    /// Neighbor's z=0 face fluid levels, indexed [y * size + x]. size*size values.
    pub pos_z: Option<Vec<f32>>,
    pub size: usize,
}

impl BoundaryLevels {
    /// Create empty boundary levels (no neighbor data — mesh seals at edges).
    pub fn empty(size: usize) -> Self {
        Self {
            pos_x: None,
            pos_y: None,
            pos_z: None,
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
}

/// A fluid mesh produced by Marching Cubes isosurface extraction.
pub struct FluidMeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub fluid_types: Vec<u8>,
    pub indices: Vec<u32>,
    pub uvs: Vec<[f32; 2]>,
    pub flow_directions: Vec<[f32; 3]>, // (dx, dz, magnitude) for UV scroll
}

/// Build fluid isosurface mesh via Marching Cubes.
///
/// Extracts the isosurface at ISO_LEVEL and passes through the actual geological
/// FluidType from each cell so UE can render distinct colors per water source.
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

    mesh_fluid_mc(grid, boundary)
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
        // Density at boundary coords is valid via grid_point_density (handles up to size)
        if grid.grid_point_density(x, y, z) > 0.0 {
            return 1.0;
        }
        // Try neighbor fluid level
        if let Some(level) = boundary.get_level(x, y, z) {
            return level;
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
                    // Single-axis: check density + boundary level
                    if !(grid.grid_point_density(cx, cy, cz) > 0.0) {
                        if let Some(level) = boundary.get_level(cx, cy, cz) {
                            if level >= MIN_LEVEL {
                                return true;
                            }
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
/// Produces triangulated isosurface at ISO_LEVEL, passing through the actual
/// geological FluidType from each cell via dominant_fluid_type().
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
                            y as f32 + p0[1] as f32 + (p1[1] as f32 - p0[1] as f32) * t,
                            z as f32 + p0[2] as f32 + (p1[2] as f32 - p0[2] as f32) * t,
                        ];
                    }
                }

                // Emit triangles from TRI_TABLE
                let tri_row = &TRI_TABLE[cube_index];
                let flow = compute_flow_direction(grid, x, y, z);
                let ft = dominant_fluid_type(grid, x, y, z) as u8;
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
                            mesh.fluid_types.push(ft);
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
    let mut water_counts = [0u32; 10];
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
        // make_fluid_grid sets FluidType::Water (=1), so dominant_fluid_type returns Water
        for &ft in &mesh.fluid_types {
            assert_eq!(ft, FluidType::Water as u8, "MC fluid type should passthrough Water (1)");
        }
    }

    #[test]
    fn test_mc_no_mesh_for_empty() {
        let grid = ChunkFluidGrid::new(16);
        let mesh = mesh_fluid_mc(&grid, &no_boundary());
        assert!(mesh.positions.is_empty(), "Empty grid should produce no MC geometry");
        assert!(mesh.indices.is_empty());
    }

    #[test]
    fn test_mc_passthrough_subtype() {
        // Grid with WaterRiver cells should produce fluid_type=6 in the mesh
        let mut grid = ChunkFluidGrid::new(16);
        for z in 6..10 {
            for y in 6..10 {
                for x in 6..10 {
                    let cell = grid.get_mut(x, y, z);
                    cell.level = 1.0;
                    cell.fluid_type = FluidType::WaterRiver;
                }
            }
        }
        let mesh = mesh_fluid(&grid, &no_boundary());
        assert!(!mesh.positions.is_empty(), "Should produce vertices");
        for &ft in &mesh.fluid_types {
            assert_eq!(ft, FluidType::WaterRiver as u8, "Should passthrough WaterRiver (6)");
        }
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
        for z in 4..8 {
            for y in 4..8 {
                pos_x_levels[z * 16 + y] = 1.0; // neighbor has fluid at x=0
            }
        }
        boundary.pos_x = Some(pos_x_levels);

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
