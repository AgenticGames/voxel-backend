use std::collections::HashMap;
use std::collections::VecDeque;
use voxel_core::mesh::Mesh;

/// Maximum allowed edge length squared for any triangle.
/// Adjacent DC cells span at most 2 cells; with vertices clamped to cell bounds
/// the theoretical max is sqrt(12) ≈ 3.46.  Use 4.0 as a generous threshold.
const MAX_EDGE_LEN_SQ: f32 = 4.0 * 4.0;

/// Check mesh validity: all indices in bounds, no degenerate triangles,
/// no stretched triangles, normals not zero.
pub fn validate_mesh(mesh: &Mesh) -> bool {
    // Check all indices are in bounds
    for tri in &mesh.triangles {
        for &idx in &tri.indices {
            if idx as usize >= mesh.vertices.len() {
                return false;
            }
        }
    }

    // Check no degenerate triangles
    if mesh.has_degenerate_triangles() {
        return false;
    }

    // Check no stretched triangles (edges spanning far beyond their cell neighbourhood)
    if has_stretched_triangles(mesh) {
        return false;
    }

    // Check normals are not zero
    for v in &mesh.vertices {
        if v.normal.length_squared() < 1e-12 {
            return false;
        }
    }

    true
}

/// Returns true if any triangle has an edge longer than the maximum allowed length.
fn has_stretched_triangles(mesh: &Mesh) -> bool {
    for tri in &mesh.triangles {
        let v0 = mesh.vertices[tri.indices[0] as usize].position;
        let v1 = mesh.vertices[tri.indices[1] as usize].position;
        let v2 = mesh.vertices[tri.indices[2] as usize].position;
        if (v1 - v0).length_squared() > MAX_EDGE_LEN_SQ
            || (v2 - v1).length_squared() > MAX_EDGE_LEN_SQ
            || (v2 - v0).length_squared() > MAX_EDGE_LEN_SQ
        {
            return true;
        }
    }
    false
}

/// Flood-fill navigability check on the density field.
///
/// Finds the largest connected component of air voxels (density < 0)
/// and checks that it contains at least 80% of all air voxels.
/// This allows small disconnected pockets while ensuring the main
/// cave network is well-connected.
pub fn check_navigability(density: &[f32], size: usize) -> bool {
    let total = size * size * size;
    let mut component_id = vec![0u32; total];
    let mut component_sizes: Vec<usize> = vec![0]; // component 0 = unvisited
    let mut current_component = 0u32;
    let mut air_count = 0usize;

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = z * size * size + y * size + x;
                if density[idx] < 0.0 {
                    air_count += 1;
                    if component_id[idx] == 0 {
                        // New unvisited air voxel — flood fill a new component
                        current_component += 1;
                        component_sizes.push(0);
                        let mut queue = VecDeque::new();
                        component_id[idx] = current_component;
                        component_sizes[current_component as usize] = 1;
                        queue.push_back((x, y, z));

                        while let Some((cx, cy, cz)) = queue.pop_front() {
                            let neighbors = [
                                (cx.wrapping_sub(1), cy, cz),
                                (cx + 1, cy, cz),
                                (cx, cy.wrapping_sub(1), cz),
                                (cx, cy + 1, cz),
                                (cx, cy, cz.wrapping_sub(1)),
                                (cx, cy, cz + 1),
                            ];
                            for (nx, ny, nz) in neighbors {
                                if nx < size && ny < size && nz < size {
                                    let n_idx = nz * size * size + ny * size + nx;
                                    if component_id[n_idx] == 0 && density[n_idx] < 0.0 {
                                        component_id[n_idx] = current_component;
                                        component_sizes[current_component as usize] += 1;
                                        queue.push_back((nx, ny, nz));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // No air = vacuously navigable
    if air_count == 0 {
        return true;
    }

    // Pass if the largest component has >= 80% of all air
    let largest = component_sizes.iter().copied().max().unwrap_or(0);
    largest as f64 / air_count as f64 >= 0.80
}

/// Watertight check: every edge is shared by exactly 2 triangles.
///
/// Builds an edge-to-face adjacency map. For a watertight mesh, every
/// edge (pair of vertex indices, order-independent) should appear exactly twice.
pub fn check_watertight(mesh: &Mesh) -> bool {
    if mesh.triangles.is_empty() {
        return true;
    }

    let mut edge_counts: HashMap<(u32, u32), u32> = HashMap::new();

    for tri in &mesh.triangles {
        let i = tri.indices;
        // Three edges per triangle, store as (min, max) for order independence
        let edges = [
            (i[0].min(i[1]), i[0].max(i[1])),
            (i[1].min(i[2]), i[1].max(i[2])),
            (i[0].min(i[2]), i[0].max(i[2])),
        ];

        for edge in edges {
            *edge_counts.entry(edge).or_insert(0) += 1;
        }
    }

    // Every edge should have exactly 2 adjacent triangles
    edge_counts.values().all(|&count| count == 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::mesh::{Mesh, Vertex, Triangle};
    use voxel_core::material::Material;
    use glam::Vec3;

    #[test]
    fn test_validate_empty_mesh() {
        let mesh = Mesh::new();
        assert!(validate_mesh(&mesh));
    }

    #[test]
    fn test_validate_valid_mesh() {
        let mesh = Mesh {
            vertices: vec![
                Vertex { position: Vec3::new(0.0, 0.0, 0.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(1.0, 0.0, 0.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(0.0, 1.0, 0.0), normal: Vec3::Y, material: Material::Limestone },
            ],
            triangles: vec![Triangle { indices: [0, 1, 2] }],
        };
        assert!(validate_mesh(&mesh));
    }

    #[test]
    fn test_validate_invalid_index() {
        let mesh = Mesh {
            vertices: vec![
                Vertex { position: Vec3::ZERO, normal: Vec3::Y, material: Material::Limestone },
            ],
            triangles: vec![Triangle { indices: [0, 1, 2] }], // indices 1,2 out of bounds
        };
        assert!(!validate_mesh(&mesh));
    }

    #[test]
    fn test_navigability_all_solid() {
        let size = 4;
        let density = vec![1.0f32; size * size * size];
        assert!(check_navigability(&density, size));
    }

    #[test]
    fn test_navigability_connected() {
        let size = 4;
        let mut density = vec![1.0f32; size * size * size];
        // Create a connected line of air
        for x in 0..4 {
            density[0 * size * size + 0 * size + x] = -1.0;
        }
        assert!(check_navigability(&density, size));
    }

    #[test]
    fn test_navigability_disconnected() {
        let size = 4;
        let mut density = vec![1.0f32; size * size * size];
        // Two separate air voxels with solid between them (50/50 split, below 80% threshold)
        density[0 * size * size + 0 * size + 0] = -1.0;
        density[3 * size * size + 3 * size + 3] = -1.0;
        assert!(!check_navigability(&density, size));
    }

    #[test]
    fn test_navigability_mostly_connected() {
        let size = 4;
        let mut density = vec![1.0f32; size * size * size];
        // Large connected region (10 voxels) + 1 stray = 91% in main, passes
        for x in 0..4 {
            density[0 * size * size + 0 * size + x] = -1.0; // 4 connected
        }
        for x in 0..4 {
            density[0 * size * size + 1 * size + x] = -1.0; // 4 more connected
        }
        density[0 * size * size + 2 * size + 0] = -1.0; // +1 connected
        density[0 * size * size + 2 * size + 1] = -1.0; // +1 connected
        // One stray disconnected voxel
        density[3 * size * size + 3 * size + 3] = -1.0;
        // 10/11 = 90.9% in largest component, > 80% threshold
        assert!(check_navigability(&density, size));
    }

    #[test]
    fn test_validate_stretched_mesh_fails() {
        // Triangle with an edge spanning 10 units — should fail validation
        let mesh = Mesh {
            vertices: vec![
                Vertex { position: Vec3::new(0.0, 0.0, 0.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(1.0, 0.0, 0.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(0.0, 10.0, 0.0), normal: Vec3::Y, material: Material::Limestone },
            ],
            triangles: vec![Triangle { indices: [0, 1, 2] }],
        };
        assert!(!validate_mesh(&mesh), "Stretched triangle should fail validation");
    }

    #[test]
    fn test_watertight_empty() {
        let mesh = Mesh::new();
        assert!(check_watertight(&mesh));
    }

    #[test]
    fn test_watertight_tetrahedron() {
        // A tetrahedron has 4 triangles and 6 edges, each shared by exactly 2 triangles
        let mesh = Mesh {
            vertices: vec![
                Vertex { position: Vec3::new(0.0, 0.0, 0.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(1.0, 0.0, 0.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(0.5, 1.0, 0.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(0.5, 0.5, 1.0), normal: Vec3::Y, material: Material::Limestone },
            ],
            triangles: vec![
                Triangle { indices: [0, 1, 2] },
                Triangle { indices: [0, 1, 3] },
                Triangle { indices: [1, 2, 3] },
                Triangle { indices: [0, 2, 3] },
            ],
        };
        assert!(check_watertight(&mesh));
    }

    #[test]
    fn test_watertight_open_mesh() {
        // Single triangle is NOT watertight (each edge appears only once)
        let mesh = Mesh {
            vertices: vec![
                Vertex { position: Vec3::new(0.0, 0.0, 0.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(1.0, 0.0, 0.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(0.0, 1.0, 0.0), normal: Vec3::Y, material: Material::Limestone },
            ],
            triangles: vec![Triangle { indices: [0, 1, 2] }],
        };
        assert!(!check_watertight(&mesh));
    }
}
