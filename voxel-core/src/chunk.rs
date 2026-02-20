use serde::{Deserialize, Serialize};
use crate::octree::node::OctreeNode;

/// Signed-integer chunk coordinates for infinite streaming
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChunkCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl ChunkCoord {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// World-space origin of this chunk (chunk_size = 16)
    pub fn world_origin(self) -> glam::Vec3 {
        self.world_origin_sized(16)
    }

    /// World-space origin of this chunk with configurable chunk_size
    pub fn world_origin_sized(self, chunk_size: usize) -> glam::Vec3 {
        let cs = chunk_size as f32;
        glam::Vec3::new(
            self.x as f32 * cs,
            self.y as f32 * cs,
            self.z as f32 * cs,
        )
    }

    /// World-space origin using explicit bounds_size (decoupled from voxel count).
    pub fn world_origin_bounds(self, bounds_size: f32) -> glam::Vec3 {
        glam::Vec3::new(
            self.x as f32 * bounds_size,
            self.y as f32 * bounds_size,
            self.z as f32 * bounds_size,
        )
    }
}

/// A chunk containing its octree and metadata
#[derive(Debug, Clone)]
pub struct Chunk {
    pub coord: ChunkCoord,
    pub octree: OctreeNode,
}
