pub mod cell;
pub mod mesh;
pub mod sim;
pub mod sources;
pub mod tables;
pub mod thread;

use crate::mesh::FluidMeshData;

/// Configuration for the fluid simulation.
#[derive(Debug, Clone)]
pub struct FluidConfig {
    pub seed: u64,
    pub chunk_size: usize,
    pub tick_rate: f32,
    pub lava_tick_divisor: u8,
    pub water_spring_threshold: f64,
    pub lava_source_threshold: f64,
    pub lava_depth_max: f64,
}

impl Default for FluidConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            chunk_size: 16,
            tick_rate: 15.0,
            lava_tick_divisor: 4,
            water_spring_threshold: 0.97,
            lava_source_threshold: 0.98,
            lava_depth_max: -50.0,
        }
    }
}

/// Events sent from the voxel engine workers to the fluid simulation thread.
pub enum FluidEvent {
    /// A chunk's solid mask was generated (after density generation).
    SolidMaskUpdate {
        chunk: (i32, i32, i32),
        mask: Vec<u64>,
    },
    /// Place initial fluid sources in a newly generated chunk.
    PlaceSources {
        chunk: (i32, i32, i32),
    },
    /// Terrain was modified by mining; solid mask updated.
    TerrainModified {
        chunk: (i32, i32, i32),
        mask: Vec<u64>,
    },
    /// A chunk was unloaded; remove its fluid data.
    ChunkUnloaded {
        chunk: (i32, i32, i32),
    },
    /// Inject fluid at a specific cell (debug / scripted spawning).
    AddFluid {
        chunk: (i32, i32, i32),
        x: u8,
        y: u8,
        z: u8,
        fluid_type: cell::FluidType,
        level: f32,
        is_source: bool,
    },
}

/// Results sent from the fluid simulation thread back to the engine.
pub enum FluidResult {
    /// A fluid mesh update for a chunk.
    FluidMesh {
        chunk: (i32, i32, i32),
        mesh: FluidMeshData,
    },
    /// Request to solidify lava cells into basalt in the terrain.
    SolidifyRequest {
        positions: Vec<((i32, i32, i32), usize, usize, usize)>,
    },
}
