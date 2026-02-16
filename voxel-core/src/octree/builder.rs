use crate::octree::node::{OctreeConfig, OctreeNode, VoxelSample};
use crate::material::Material;

/// Build octree from a flat density grid.
/// `samples`: density values for a (size+1)^3 grid (corner samples for `size^3` cells)
/// `size`: number of cells along each axis (must be a power of 2)
/// `config`: octree construction parameters
pub fn build_octree(samples: &[f32], size: usize, config: &OctreeConfig) -> OctreeNode {
    let grid_size = size + 1;
    assert_eq!(samples.len(), grid_size * grid_size * grid_size,
        "Expected {} samples for grid size {}, got {}",
        grid_size * grid_size * grid_size, size, samples.len());

    build_recursive(samples, grid_size, 0, 0, 0, size, 0, config)
}

fn sample_index(x: usize, y: usize, z: usize, grid_size: usize) -> usize {
    z * grid_size * grid_size + y * grid_size + x
}

#[allow(clippy::too_many_arguments)]
fn build_recursive(
    samples: &[f32],
    grid_size: usize,
    x: usize, y: usize, z: usize,
    size: usize,
    depth: u32,
    config: &OctreeConfig,
) -> OctreeNode {
    if size == 1 || depth >= config.max_depth {
        // Leaf node: sample 8 corners of this cell
        let corners = [
            sample_at(samples, grid_size, x, y, z),
            sample_at(samples, grid_size, x + 1, y, z),
            sample_at(samples, grid_size, x, y + 1, z),
            sample_at(samples, grid_size, x + 1, y + 1, z),
            sample_at(samples, grid_size, x, y, z + 1),
            sample_at(samples, grid_size, x + 1, y, z + 1),
            sample_at(samples, grid_size, x, y + 1, z + 1),
            sample_at(samples, grid_size, x + 1, y + 1, z + 1),
        ];

        // Check if all corners have the same sign (uniform region)
        let all_positive = corners.iter().all(|c| c.density > 0.0);
        let all_negative = corners.iter().all(|c| c.density <= 0.0);

        if all_positive {
            return OctreeNode::Empty { material: Material::Air };
        }
        if all_negative {
            return OctreeNode::Empty { material: Material::Limestone };
        }

        return OctreeNode::Leaf {
            corners,
            dc_vertex: None,
        };
    }

    let half = size / 2;
    if half == 0 {
        // Can't subdivide further
        let corners = [
            sample_at(samples, grid_size, x, y, z),
            sample_at(samples, grid_size, x + 1, y, z),
            sample_at(samples, grid_size, x, y + 1, z),
            sample_at(samples, grid_size, x + 1, y + 1, z),
            sample_at(samples, grid_size, x, y, z + 1),
            sample_at(samples, grid_size, x + 1, y, z + 1),
            sample_at(samples, grid_size, x, y + 1, z + 1),
            sample_at(samples, grid_size, x + 1, y + 1, z + 1),
        ];
        return OctreeNode::Leaf {
            corners,
            dc_vertex: None,
        };
    }

    // Build 8 children in Morton order
    let children = Box::new([
        build_recursive(samples, grid_size, x,        y,        z,        half, depth + 1, config),
        build_recursive(samples, grid_size, x + half,  y,        z,        half, depth + 1, config),
        build_recursive(samples, grid_size, x,        y + half,  z,        half, depth + 1, config),
        build_recursive(samples, grid_size, x + half,  y + half,  z,        half, depth + 1, config),
        build_recursive(samples, grid_size, x,        y,        z + half,  half, depth + 1, config),
        build_recursive(samples, grid_size, x + half,  y,        z + half,  half, depth + 1, config),
        build_recursive(samples, grid_size, x,        y + half,  z + half,  half, depth + 1, config),
        build_recursive(samples, grid_size, x + half,  y + half,  z + half,  half, depth + 1, config),
    ]);

    // Try to collapse: if all children are Empty with the same material, collapse
    if let Some(mat) = all_children_same_material(&children) {
        return OctreeNode::Empty { material: mat };
    }

    OctreeNode::Branch {
        children,
        lod_vertex: None,
        lod_qef: None,
    }
}

fn sample_at(samples: &[f32], grid_size: usize, x: usize, y: usize, z: usize) -> VoxelSample {
    let idx = sample_index(x.min(grid_size - 1), y.min(grid_size - 1), z.min(grid_size - 1), grid_size);
    let density = samples[idx];
    VoxelSample {
        density,
        material: if density <= 0.0 { Material::Limestone } else { Material::Air },
    }
}

fn all_children_same_material(children: &[OctreeNode; 8]) -> Option<Material> {
    let first_mat = match &children[0] {
        OctreeNode::Empty { material } => *material,
        _ => return None,
    };
    for child in &children[1..] {
        match child {
            OctreeNode::Empty { material } if *material == first_mat => {}
            _ => return None,
        }
    }
    Some(first_mat)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_positive_collapses_to_empty() {
        // All densities positive => all air => single Empty node
        let size = 2;
        let grid = vec![1.0f32; 3 * 3 * 3]; // (size+1)^3
        let config = OctreeConfig { max_depth: 4, error_threshold: 0.01 };
        let root = build_octree(&grid, size, &config);
        match root {
            OctreeNode::Empty { material } => assert_eq!(material, Material::Air),
            _ => panic!("Expected Empty node for uniform positive field"),
        }
    }

    #[test]
    fn uniform_negative_collapses_to_empty_stone() {
        let size = 2;
        let grid = vec![-1.0f32; 3 * 3 * 3];
        let config = OctreeConfig::default();
        let root = build_octree(&grid, size, &config);
        match root {
            OctreeNode::Empty { material } => assert_eq!(material, Material::Limestone),
            _ => panic!("Expected Empty(Stone) node for uniform negative field"),
        }
    }

    #[test]
    fn mixed_field_produces_leaves() {
        // Create a field with a sign change: half positive, half negative
        let size = 2;
        let grid_size = size + 1;
        let mut grid = vec![0.0f32; grid_size * grid_size * grid_size];
        for z in 0..grid_size {
            for y in 0..grid_size {
                for x in 0..grid_size {
                    let idx = sample_index(x, y, z, grid_size);
                    // Positive above y=1, negative below
                    grid[idx] = y as f32 - 1.0;
                }
            }
        }
        let config = OctreeConfig { max_depth: 4, error_threshold: 0.01 };
        let root = build_octree(&grid, size, &config);
        // Should not be collapsed since there's a sign change
        match root {
            OctreeNode::Empty { .. } => panic!("Expected non-Empty node for mixed field"),
            _ => {} // Branch or Leaf is fine
        }
    }

    #[test]
    fn correct_sample_count() {
        let size = 4;
        let grid_size = size + 1;
        let grid = vec![1.0f32; grid_size * grid_size * grid_size];
        let config = OctreeConfig { max_depth: 4, error_threshold: 0.01 };
        // Should not panic
        let _ = build_octree(&grid, size, &config);
    }

    #[test]
    fn depth_limit_creates_leaves() {
        let size = 4;
        let grid_size = size + 1;
        let mut grid = vec![0.0f32; grid_size * grid_size * grid_size];
        // Alternating pattern that can't collapse
        for z in 0..grid_size {
            for y in 0..grid_size {
                for x in 0..grid_size {
                    let idx = sample_index(x, y, z, grid_size);
                    grid[idx] = if (x + y + z) % 2 == 0 { 1.0 } else { -1.0 };
                }
            }
        }
        let config = OctreeConfig { max_depth: 1, error_threshold: 0.01 };
        let root = build_octree(&grid, size, &config);
        // With max_depth=1, should create leaves quickly
        match root {
            OctreeNode::Branch { .. } => {} // expected
            OctreeNode::Leaf { .. } => {}   // also fine at depth limit
            OctreeNode::Empty { .. } => panic!("Should not collapse alternating field"),
        }
    }
}
