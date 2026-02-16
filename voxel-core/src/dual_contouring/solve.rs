use glam::Vec3;
use crate::hermite::{HermiteData, FastHashMap};
use crate::dual_contouring::qef::QefData;

/// Solve per-cell QEF vertices from hermite data on a flat grid.
///
/// `grid_size` is the number of cells along each axis (density grid is grid_size + 1).
/// Returns a Vec of length grid_size^3, indexed as [z * grid_size^2 + y * grid_size + x].
/// Cells with no sign-changing edges get a NAN sentinel (skipped by mesh_gen).
pub fn solve_dc_vertices(hermite: &HermiteData, grid_size: usize) -> Vec<Vec3> {
    let total = grid_size * grid_size * grid_size;
    // Use sparse map with identity hasher for fast integer key lookup
    let mut qefs: FastHashMap<usize, QefData> = FastHashMap::with_capacity_and_hasher(
        hermite.edges.len() / 2,
        Default::default(),
    );

    for (edge_key, intersection) in hermite.edges.iter() {
        let ex = edge_key.x() as usize;
        let ey = edge_key.y() as usize;
        let ez = edge_key.z() as usize;
        let axis = edge_key.axis();

        // Compute the intersection position in grid space
        let intersection_pos = match axis {
            0 => Vec3::new(ex as f32 + intersection.t, ey as f32, ez as f32),
            1 => Vec3::new(ex as f32, ey as f32 + intersection.t, ez as f32),
            2 => Vec3::new(ex as f32, ey as f32, ez as f32 + intersection.t),
            _ => continue,
        };

        // Accumulate this intersection into all adjacent cells (no heap allocation)
        let adj = match axis {
            0 => adjacent_cells_x(ex, ey, ez, grid_size),
            1 => adjacent_cells_y(ex, ey, ez, grid_size),
            2 => adjacent_cells_z(ex, ey, ez, grid_size),
            _ => AdjacentCells { cells: [0; 4], count: 0 },
        };

        for i in 0..adj.count {
            qefs.entry(adj.cells[i])
                .or_default()
                .add(intersection_pos, intersection.normal);
        }
    }

    // Solve each cell's QEF, clamping to cell bounds.
    // Cells with no intersections get NAN sentinel (skipped by mesh_gen).
    let sentinel = Vec3::new(f32::NAN, f32::NAN, f32::NAN);
    let mut vertices = vec![sentinel; total];
    let gs2 = grid_size * grid_size;
    for (&idx, qef) in &qefs {
        let x = idx % grid_size;
        let y = (idx / grid_size) % grid_size;
        let z = idx / gs2;
        let min_bound = Vec3::new(x as f32, y as f32, z as f32);
        let max_bound = min_bound + Vec3::ONE;
        vertices[idx] = qef.solve_clamped(min_bound, max_bound);
    }

    vertices
}

/// Fixed-capacity collection of adjacent cell indices (avoids heap allocation).
struct AdjacentCells {
    cells: [usize; 4],
    count: usize,
}

/// Get cell indices adjacent to an X-axis edge at grid position (x, y, z).
/// An X-edge is shared by cells: (x, y, z), (x, y-1, z), (x, y, z-1), (x, y-1, z-1)
#[inline]
fn adjacent_cells_x(x: usize, y: usize, z: usize, size: usize) -> AdjacentCells {
    let mut adj = AdjacentCells { cells: [0; 4], count: 0 };
    let offsets: [(i32, i32); 4] = [(0, 0), (-1, 0), (0, -1), (-1, -1)];
    for (dy, dz) in offsets {
        let cy = y as i32 + dy;
        let cz = z as i32 + dz;
        if cy >= 0 && cy < size as i32 && cz >= 0 && cz < size as i32 && x < size {
            adj.cells[adj.count] = cz as usize * size * size + cy as usize * size + x;
            adj.count += 1;
        }
    }
    adj
}

/// Get cell indices adjacent to a Y-axis edge at grid position (x, y, z).
#[inline]
fn adjacent_cells_y(x: usize, y: usize, z: usize, size: usize) -> AdjacentCells {
    let mut adj = AdjacentCells { cells: [0; 4], count: 0 };
    let offsets: [(i32, i32); 4] = [(0, 0), (-1, 0), (0, -1), (-1, -1)];
    for (dx, dz) in offsets {
        let cx = x as i32 + dx;
        let cz = z as i32 + dz;
        if cx >= 0 && cx < size as i32 && cz >= 0 && cz < size as i32 && y < size {
            adj.cells[adj.count] = cz as usize * size * size + y * size + cx as usize;
            adj.count += 1;
        }
    }
    adj
}

/// Get cell indices adjacent to a Z-axis edge at grid position (x, y, z).
#[inline]
fn adjacent_cells_z(x: usize, y: usize, z: usize, size: usize) -> AdjacentCells {
    let mut adj = AdjacentCells { cells: [0; 4], count: 0 };
    let offsets: [(i32, i32); 4] = [(0, 0), (-1, 0), (0, -1), (-1, -1)];
    for (dx, dy) in offsets {
        let cx = x as i32 + dx;
        let cy = y as i32 + dy;
        if cx >= 0 && cx < size as i32 && cy >= 0 && cy < size as i32 && z < size {
            adj.cells[adj.count] = z * size * size + cy as usize * size + cx as usize;
            adj.count += 1;
        }
    }
    adj
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hermite::{EdgeIntersection, EdgeKey, HermiteData};
    use crate::material::Material;

    #[test]
    fn empty_hermite_gives_nan_vertices() {
        let hermite = HermiteData::default();
        let verts = solve_dc_vertices(&hermite, 4);
        assert_eq!(verts.len(), 64);
        for v in &verts {
            assert!(v.x.is_nan(), "Empty cells should have NAN sentinel");
        }
    }

    #[test]
    fn single_edge_produces_vertex_in_cell() {
        let mut hermite = HermiteData::default();
        // X-edge at (0, 1, 1) with sign change at t=0.5
        hermite.edges.insert(
            EdgeKey::new(0, 1, 1, 0),
            EdgeIntersection {
                t: 0.5,
                normal: Vec3::Y,
                material: Material::Limestone,
            },
        );

        let grid_size = 4;
        let verts = solve_dc_vertices(&hermite, grid_size);
        // Cell (0,1,1) should have a non-zero vertex
        let idx = 1 * grid_size * grid_size + 1 * grid_size + 0;
        assert_ne!(verts[idx], Vec3::ZERO, "Cell (0,1,1) should have a DC vertex");
    }

    #[test]
    fn sphere_sdf_produces_vertices() {
        // Create hermite data for a simple sphere
        let grid_size = 8;
        let mut hermite = HermiteData::default();
        let center = 4.0f32;
        let radius = 2.5f32;

        // Walk edges and detect sign changes
        for z in 0..grid_size + 1 {
            for y in 0..grid_size + 1 {
                for x in 0..grid_size {
                    let da = sdf_sphere(x as f32, y as f32, z as f32, center, radius);
                    let db = sdf_sphere(x as f32 + 1.0, y as f32, z as f32, center, radius);
                    if (da > 0.0) != (db > 0.0) {
                        let t = da / (da - db);
                        hermite.edges.insert(
                            EdgeKey::new(x as u8, y as u8, z as u8, 0),
                            EdgeIntersection {
                                t,
                                normal: Vec3::X * if da < db { 1.0 } else { -1.0 },
                                material: Material::Limestone,
                            },
                        );
                    }
                }
            }
        }
        for z in 0..grid_size + 1 {
            for y in 0..grid_size {
                for x in 0..grid_size + 1 {
                    let da = sdf_sphere(x as f32, y as f32, z as f32, center, radius);
                    let db = sdf_sphere(x as f32, y as f32 + 1.0, z as f32, center, radius);
                    if (da > 0.0) != (db > 0.0) {
                        let t = da / (da - db);
                        hermite.edges.insert(
                            EdgeKey::new(x as u8, y as u8, z as u8, 1),
                            EdgeIntersection {
                                t,
                                normal: Vec3::Y * if da < db { 1.0 } else { -1.0 },
                                material: Material::Limestone,
                            },
                        );
                    }
                }
            }
        }
        for z in 0..grid_size {
            for y in 0..grid_size + 1 {
                for x in 0..grid_size + 1 {
                    let da = sdf_sphere(x as f32, y as f32, z as f32, center, radius);
                    let db = sdf_sphere(x as f32, y as f32, z as f32 + 1.0, center, radius);
                    if (da > 0.0) != (db > 0.0) {
                        let t = da / (da - db);
                        hermite.edges.insert(
                            EdgeKey::new(x as u8, y as u8, z as u8, 2),
                            EdgeIntersection {
                                t,
                                normal: Vec3::Z * if da < db { 1.0 } else { -1.0 },
                                material: Material::Limestone,
                            },
                        );
                    }
                }
            }
        }

        let verts = solve_dc_vertices(&hermite, grid_size);
        let real_verts: Vec<_> = verts.iter().filter(|v| !v.x.is_nan()).collect();
        assert!(
            real_verts.len() > 10,
            "Sphere should produce many DC vertices, got {}",
            real_verts.len()
        );

        // All real vertices should be roughly within the grid
        for v in &real_verts {
            assert!(v.x >= 0.0 && v.x <= grid_size as f32, "x out of range: {}", v.x);
            assert!(v.y >= 0.0 && v.y <= grid_size as f32, "y out of range: {}", v.y);
            assert!(v.z >= 0.0 && v.z <= grid_size as f32, "z out of range: {}", v.z);
        }
    }

    fn sdf_sphere(x: f32, y: f32, z: f32, center: f32, radius: f32) -> f32 {
        let dx = x - center;
        let dy = y - center;
        let dz = z - center;
        (dx * dx + dy * dy + dz * dz).sqrt() - radius
    }
}
