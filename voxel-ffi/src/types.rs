/// All `#[repr(C)]` FFI types for the voxel engine DLL interface.

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiChunkCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

/// SoA layout for UE ProceduralMeshComponent.
/// Pointers are owned by the Rust side and freed via `voxel_free_result`.
#[repr(C)]
pub struct FfiMeshData {
    pub positions: *mut FfiVec3,
    pub normals: *mut FfiVec3,
    pub material_ids: *mut u8,
    pub vertex_count: u32,
    pub indices: *mut u32,
    pub index_count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiMinedMaterials {
    pub counts: [u32; 19],
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiResultType {
    None = 0,
    ChunkMesh = 1,
    MineResult = 2,
    Error = 3,
}

#[repr(C)]
pub struct FfiResult {
    pub result_type: FfiResultType,
    pub chunk: FfiChunkCoord,
    pub mesh: FfiMeshData,
    pub mined: FfiMinedMaterials,
    pub generation: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiEngineConfig {
    pub seed: u64,
    pub chunk_size: u32,
    pub worker_threads: u32,
    pub world_scale: f32,
    pub max_edge_length: f32,
    // Noise
    pub cavern_frequency: f64,
    pub cavern_threshold: f64,
    pub detail_octaves: u32,
    pub detail_persistence: f64,
    pub warp_amplitude: f64,
    // Worm
    pub worms_per_region: u32,
    pub worm_radius_min: f32,
    pub worm_radius_max: f32,
    pub worm_step_length: f32,
    pub worm_max_steps: u32,
    pub worm_falloff_power: f32,
    pub region_size: u32,
    // ── Ore Config (52 fields) ──
    // Host Rock (9)
    pub host_sandstone_depth: f64,
    pub host_granite_depth: f64,
    pub host_basalt_depth: f64,
    pub host_slate_depth: f64,
    pub host_boundary_noise_amp: f64,
    pub host_boundary_noise_freq: f64,
    pub host_basalt_intrusion_freq: f64,
    pub host_basalt_intrusion_thresh: f64,
    pub host_basalt_intrusion_depth_max: f64,
    // Banded Iron (6)
    pub iron_band_frequency: f64,
    pub iron_noise_perturbation: f64,
    pub iron_noise_frequency: f64,
    pub iron_threshold: f64,
    pub iron_depth_min: f64,
    pub iron_depth_max: f64,
    // Copper (4)
    pub copper_frequency: f64,
    pub copper_threshold: f64,
    pub copper_depth_min: f64,
    pub copper_depth_max: f64,
    // Malachite (4)
    pub malachite_frequency: f64,
    pub malachite_threshold: f64,
    pub malachite_depth_min: f64,
    pub malachite_depth_max: f64,
    // Quartz (4)
    pub quartz_frequency: f64,
    pub quartz_threshold: f64,
    pub quartz_depth_min: f64,
    pub quartz_depth_max: f64,
    // Gold (4)
    pub gold_frequency: f64,
    pub gold_threshold: f64,
    pub gold_depth_min: f64,
    pub gold_depth_max: f64,
    // Pyrite (4)
    pub pyrite_frequency: f64,
    pub pyrite_threshold: f64,
    pub pyrite_depth_min: f64,
    pub pyrite_depth_max: f64,
    // Kimberlite (6)
    pub kimb_pipe_freq_2d: f64,
    pub kimb_pipe_threshold: f64,
    pub kimb_depth_min: f64,
    pub kimb_depth_max: f64,
    pub kimb_diamond_threshold: f64,
    pub kimb_diamond_frequency: f64,
    // Sulfide (5)
    pub sulfide_frequency: f64,
    pub sulfide_threshold: f64,
    pub sulfide_tin_threshold: f64,
    pub sulfide_depth_min: f64,
    pub sulfide_depth_max: f64,
    // Geode (6)
    pub geode_frequency: f64,
    pub geode_center_threshold: f64,
    pub geode_shell_thickness: f64,
    pub geode_hollow_factor: f32,
    pub geode_depth_min: f64,
    pub geode_depth_max: f64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiMineRequest {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub radius: f32,
    pub mode: u8, // 0=sphere, 1=peel
    pub normal_x: f32,
    pub normal_y: f32,
    pub normal_z: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiEngineStats {
    pub chunks_loaded: u32,
    pub pending_requests: u32,
    pub completed_results: u32,
    pub worker_threads_active: u32,
}

// ── Internal (non-FFI) types ──

/// Converted mesh data in UE coordinate space, ready to be handed out via FFI.
pub struct ConvertedMesh {
    pub positions: Vec<FfiVec3>,
    pub normals: Vec<FfiVec3>,
    pub material_ids: Vec<u8>,
    pub indices: Vec<u32>,
}

/// Messages sent to worker threads.
pub enum WorkerRequest {
    Generate {
        chunk: (i32, i32, i32),
        generation: u64,
    },
    Mine {
        request: FfiMineRequest,
    },
    Unload {
        chunk: (i32, i32, i32),
    },
}

/// Results sent back from worker threads.
pub enum WorkerResult {
    ChunkMesh {
        chunk: (i32, i32, i32),
        mesh: ConvertedMesh,
        generation: u64,
    },
    MinedMaterials {
        mined: FfiMinedMaterials,
    },
}
