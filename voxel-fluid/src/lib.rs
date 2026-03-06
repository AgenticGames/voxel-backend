pub mod cell;
pub mod mesh;
pub mod sim;
pub mod sources;
pub mod tables;
pub mod thread;

use std::collections::HashMap;
use crate::cell::FluidCell;
use crate::mesh::FluidMeshData;

/// Lightweight snapshot of all fluid cells for sleep system queries.
#[derive(Debug, Clone)]
pub struct FluidSnapshot {
    pub chunks: HashMap<(i32, i32, i32), Vec<FluidCell>>,
    pub chunk_size: usize,
}

impl Default for FluidSnapshot {
    fn default() -> Self {
        Self {
            chunks: HashMap::new(),
            chunk_size: 16,
        }
    }
}

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
    // Water
    pub water_noise_frequency: f64,
    pub water_depth_min: f64,
    pub water_depth_max: f64,
    pub water_flow_rate: f32,
    pub water_spread_rate: f32,
    // Lava
    pub lava_noise_frequency: f64,
    pub lava_depth_min: f64,
    pub lava_flow_rate: f32,
    pub lava_spread_rate: f32,
    // General
    pub cavern_source_bias: f64,
    pub tunnel_bend_threshold: f64,
}

impl Default for FluidConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            chunk_size: 16,
            tick_rate: 15.0,
            lava_tick_divisor: 4,
            water_spring_threshold: 2.0,
            lava_source_threshold: 0.98,
            lava_depth_max: -50.0,
            water_noise_frequency: 0.05,
            water_depth_min: -9999.0,
            water_depth_max: 9999.0,
            water_flow_rate: 0.25,
            water_spread_rate: 0.125,
            lava_noise_frequency: 0.03,
            lava_depth_min: -9999.0,
            lava_flow_rate: 0.1,
            lava_spread_rate: 0.125,
            cavern_source_bias: 0.0,
            tunnel_bend_threshold: 0.0,
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
    /// Place geological springs (spring lines, drips) in a chunk.
    /// Springs are detected by the worker thread which has access to the DensityField.
    PlaceGeologicalSprings {
        chunk: (i32, i32, i32),
        springs: Vec<(u8, u8, u8, f32)>, // (lx, ly, lz, level)
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
    /// Request a snapshot of all fluid cells (used by sleep system).
    /// Response sent via the dedicated reply channel.
    SnapshotRequest {
        reply_tx: crossbeam_channel::Sender<FluidSnapshot>,
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
