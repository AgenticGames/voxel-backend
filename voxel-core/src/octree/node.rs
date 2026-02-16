use glam::Vec3;
use crate::material::Material;
use crate::dual_contouring::qef::QefData;

/// Configuration for octree construction
#[derive(Debug, Clone)]
pub struct OctreeConfig {
    pub max_depth: u32,
    pub error_threshold: f32,
}

impl Default for OctreeConfig {
    fn default() -> Self {
        Self {
            max_depth: 4,
            error_threshold: 0.01,
        }
    }
}

/// Sample at a voxel corner
#[derive(Debug, Clone, Copy)]
pub struct VoxelSample {
    pub density: f32,
    pub material: Material,
}

impl Default for VoxelSample {
    fn default() -> Self {
        Self {
            density: 1.0,
            material: Material::Limestone,
        }
    }
}

/// Octree node enum — children in Morton order
#[derive(Debug, Clone)]
pub enum OctreeNode {
    Empty {
        material: Material,
    },
    Leaf {
        corners: [VoxelSample; 8],
        dc_vertex: Option<Vec3>,
    },
    Branch {
        children: Box<[OctreeNode; 8]>,
        lod_vertex: Option<Vec3>,
        lod_qef: Option<QefData>,
    },
}

impl Default for OctreeNode {
    fn default() -> Self {
        OctreeNode::Empty {
            material: Material::Air,
        }
    }
}
