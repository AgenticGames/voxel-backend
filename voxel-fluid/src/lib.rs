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
    // Water substeps per tick (information propagates N cells per tick)
    pub water_substeps: u8,
    // General
    pub cavern_source_bias: f64,
    pub tunnel_bend_threshold: f64,
    // New: flow animation / density threshold
    pub flow_anim_speed: f32,
    pub solid_threshold: f32,
    // Solid corner threshold for is_mostly_solid guard (1-8, default 6)
    pub solid_corner_threshold: u8,
    // Grace period for non-source fluid (e.g. cauldron water)
    pub source_grace_ticks: u16,    // ticks of source-like behavior (default 50 = 5s at 10Hz)
    // Upward pressure equalization
    pub water_pressure_rate: f32,
    pub lava_pressure_rate: f32,
    // Mesh post-processing
    pub mesh_smooth_iterations: u32,
    pub mesh_smooth_strength: f32,
    pub mesh_qef_refinement: bool,
    pub mesh_recalc_normals: bool,
    // Rim/skirt fix bundle (2026-08-04) — three independently toggleable
    // flags so they can be A/B tested live, then defaulted on.
    /// Release hysteresis-held cells that have been stagnant below the mesh
    /// iso threshold: a settled pool's drained rim cells otherwise stay in
    /// the mesh forever (raised phantom ring above the real surface).
    pub mesh_sticky_release: bool,
    /// On floor-contact rim edges (vertical, solid-below) use a whisker
    /// recess instead of the full ROCK_RECESS_T, so the sheet hugs the floor
    /// from just below the surface instead of diving 0.1 cells into it.
    pub mesh_floor_clamp: bool,
    /// Drop triangles whose three vertices are all at/inside the terrain —
    /// the buried closure skirt is only ever visible from inside the ground
    /// (or through terrain cracks), and is the "under the floor" sheet.
    pub mesh_buried_cull: bool,
}

impl Default for FluidConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            chunk_size: 16,
            tick_rate: 10.0,
            lava_tick_divisor: 4,
            water_spring_threshold: 2.0,
            lava_source_threshold: 0.98,
            lava_depth_max: -50.0,
            water_noise_frequency: 0.05,
            water_depth_min: -9999.0,
            water_depth_max: 9999.0,
            water_flow_rate: 2.0,
            water_spread_rate: 2.0,
            lava_noise_frequency: 0.03,
            lava_depth_min: -9999.0,
            lava_flow_rate: 0.1,
            lava_spread_rate: 0.125,
            water_substeps: 6,
            cavern_source_bias: 0.0,
            tunnel_bend_threshold: 0.0,
            flow_anim_speed: 1.0,
            solid_threshold: 0.0,
            solid_corner_threshold: 6,
            source_grace_ticks: 50,
            water_pressure_rate: 0.3,
            lava_pressure_rate: 0.1,
            mesh_smooth_iterations: 2,
            mesh_smooth_strength: 0.3,
            mesh_qef_refinement: true,
            mesh_recalc_normals: true,
            mesh_sticky_release: false,
            mesh_floor_clamp: false,
            mesh_buried_cull: false,
        }
    }
}

/// Events sent from the voxel engine workers to the fluid simulation thread.
pub enum FluidEvent {
    /// A chunk's density field was generated — provides raw 17^3 density values.
    DensityUpdate {
        chunk: (i32, i32, i32),
        densities: Vec<f32>, // 17^3 = 4913 raw density values
    },
    /// Place initial fluid sources in a newly generated chunk.
    PlaceSources {
        chunk: (i32, i32, i32),
    },
    /// Terrain was modified by mining; density values updated.
    TerrainModified {
        chunk: (i32, i32, i32),
        densities: Vec<f32>, // 17^3 = 4913 raw density values
    },
    /// A chunk was unloaded; remove its fluid data.
    ChunkUnloaded {
        chunk: (i32, i32, i32),
    },
    /// Place geological springs (spring lines, drips) in a chunk.
    /// Springs are detected by the worker thread which has access to the DensityField.
    PlaceGeologicalSprings {
        chunk: (i32, i32, i32),
        springs: Vec<(u8, u8, u8, f32, u8)>, // (lx, ly, lz, level, fluid_type_u8)
    },
    /// Inject fluid at a specific cell (debug / scripted spawning).
    /// `max_flow_dist`: bounded-flow limit when `is_source = true`.
    /// 0 = unlimited (legacy behavior, used by procedural sources for backward
    /// compat). >0 = source's children stop propagating beyond this hop count
    /// (Minecraft-style hard length limit, with linear taper across the last
    /// `chunk::TAPER_HOPS` cells). Ignored for non-source placements.
    /// Procedural pipe-lava vents for one chunk (kimberlite-adjacent).
    /// Separate from AddFluid so the fluid thread can once-guard it — the
    /// worker re-sends on every stream-in, and generic AddFluid must keep
    /// working for player brushes (bug #216 refill class).
    PlacePipeLava {
        chunk: (i32, i32, i32),
        cells: Vec<(u8, u8, u8, f32)>,
    },
    /// Gen-time fluid seeds for one chunk (pools, formation cauldron fills,
    /// zone lakes). Once-guarded like the other procedural placements: store
    /// eviction makes a region re-generate via the slow path on return
    /// flights, and re-injecting seeds refilled every basin to gen-fresh
    /// full (#216, the "no dormancy involved" refills).
    /// cells: (x, y, z, fluid_type_u8, is_source, max_flow_dist).
    PlaceSeedFluids {
        chunk: (i32, i32, i32),
        cells: Vec<(u8, u8, u8, u8, bool, u8)>,
    },
    AddFluid {
        chunk: (i32, i32, i32),
        x: u8,
        y: u8,
        z: u8,
        fluid_type: cell::FluidType,
        level: f32,
        is_source: bool,
        max_flow_dist: u8,
    },
    /// Update fluid simulation config at runtime (grace period).
    UpdateFluidConfig {
        source_grace_ticks: u16,
    },
    /// Update the simulation *rate* knobs at runtime — how fast the sim ticks
    /// and how fast each fluid moves per tick. Sent by
    /// `VoxelEngine::update_config`, which UE calls on every VoxelConfig.json
    /// reload (the O-menu codex writes that file), so lava/water speed is
    /// tunable live instead of only at world creation.
    ///
    /// Deliberately does NOT carry source/threshold/mesh fields — those come
    /// from other config files on their own paths and would be clobbered.
    UpdateFluidRates {
        tick_rate: f32,
        lava_tick_divisor: u8,
        water_flow_rate: f32,
        water_spread_rate: f32,
        lava_flow_rate: f32,
        lava_spread_rate: f32,
    },
    /// Update the rim/skirt mesh flags at runtime (same reload path as
    /// UpdateFluidRates). The handler dirty-sweeps every fluid grid so a
    /// settled pool re-meshes immediately — without that, toggling a flag
    /// would only show on chunks the sim happens to touch next.
    UpdateFluidMeshFlags {
        sticky_release: bool,
        floor_clamp: bool,
        buried_cull: bool,
    },
    /// Request a snapshot of all fluid cells (used by sleep system).
    /// Response sent via the dedicated reply channel.
    SnapshotRequest {
        reply_tx: crossbeam_channel::Sender<FluidSnapshot>,
    },
    /// Drain (zero out) all lava cells in the given chunks.
    /// Used after sleep solidification converts lava to basalt.
    DrainLavaChunks {
        chunks: Vec<(i32, i32, i32)>,
    },
    /// Restore fluid state from a save file. The fluid thread holds these
    /// entries until the matching chunk receives a DensityUpdate or
    /// TerrainModified event (so cell_capacity matches the post-load
    /// terrain), then applies them via the same path as AddFluid.
    PendingFluidLoad {
        chunk: (i32, i32, i32),
        cells: Vec<PendingFluidCell>,
    },
}

/// One fluid cell waiting to be applied once the chunk's density arrives.
#[derive(Debug, Clone, Copy)]
pub struct PendingFluidCell {
    /// Linear cell index inside the chunk's flat array (z*size² + y*size + x).
    pub idx: u32,
    pub fluid_type: cell::FluidType,
    pub level: f32,
    pub is_source: bool,
    pub max_flow_dist: u8,
}

/// Results sent from the fluid simulation thread back to the engine.
pub enum FluidResult {
    /// A fluid mesh update for a chunk.
    FluidMesh {
        chunk: (i32, i32, i32),
        mesh: FluidMeshData,
    },
    /// **DEPRECATED**: legacy single-list quench (lava→basalt). Replaced by
    /// `LavaQuench` which carries the full Obsidian + Scoria + drain plan.
    /// Kept so older callers don't fail to compile.
    SolidifyRequest {
        positions: Vec<((i32, i32, i32), usize, usize, usize)>,
    },
    /// Live lava↔water contact solidification plan, produced by the fluid
    /// sim each tick. The worker thread applies these voxel writes:
    /// `obsidian` cells become Material::Obsidian voxels (glassy quench skin),
    /// `scoria` cells become Material::Scoria (steam-altered halo, thicker
    /// for bigger lava chambers), `drained_water` cells get their fluid level
    /// zeroed in the density field's neighbor sync. `pillow_sources` tracks
    /// lava SOURCE cells currently in contact — the fluid sim grows a
    /// pillow mound around each of them over many ticks.
    LavaQuench {
        obsidian: Vec<((i32, i32, i32), usize, usize, usize)>,
        scoria: Vec<((i32, i32, i32), usize, usize, usize)>,
        drained_water: Vec<((i32, i32, i32), usize, usize, usize)>,
    },
}
