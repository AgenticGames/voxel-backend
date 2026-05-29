//! Core geometry + mesh + result FFI types.

use super::*;

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

/// Voxel coordinate (3 × i32). Identical layout to FfiChunkCoord — kept as
/// a distinct type so call sites self-document whether they're passing a
/// per-voxel position or a chunk key.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiVoxelCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

/// Inclusive AABB in world voxel coordinates. Matches
/// `crate::triggers::VoxelAabb`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiVoxelAabb {
    pub min: FfiVoxelCoord,
    pub max: FfiVoxelCoord,
}

/// Summary record for an editor collapse trigger. Returned by
/// `voxel_get_trigger_info`. Names are stored inline (truncated to 63
/// chars + NUL) so UE never has to free per-trigger heap allocations.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiTriggerInfo {
    pub id: u32,
    pub armed: u8,
    /// 0 = OnFirstMine, 1 = OnPillarLoss
    pub activation_kind: u8,
    /// For OnPillarLoss: number of pillars (max 8 reported here; the trigger
    /// internally can hold more). For OnFirstMine: 1 (the trigger volume).
    pub volume_count: u8,
    /// 0 = AnyPillar, 1 = NPillars, 2 = AllPillars. Ignored for OnFirstMine.
    pub loss_condition: u8,
    pub loss_n: u8,
    pub _padding: [u8; 3],
    pub loss_threshold: f32,
    pub fall_distance_uu: f32,
    pub slab_voxel_count: u32,
    pub pile_chunk_count: u32,
    /// Primary volume: trigger_volume (OnFirstMine) or volumes[0] (OnPillarLoss).
    pub primary_volume: FfiVoxelAabb,
    /// Up to 8 pillar volumes inline (OnPillarLoss only). Unused entries are
    /// zeroed. For triggers with >8 pillars, only the first 8 are reported.
    pub pillar_volumes: [FfiVoxelAabb; 8],
    /// Name as UTF-8, NUL-terminated, max 63 chars + NUL.
    pub name: [u8; 64],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiSubmesh {
    pub material_id: u8,
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub index_offset: u32,
    pub index_count: u32,
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
    pub submeshes: *mut FfiSubmesh,
    pub submesh_count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiMinedMaterials {
    pub counts: [u32; 64],
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiResultType {
    None = 0,
    ChunkMesh = 1,
    MineResult = 2,
    Error = 3,
    FluidMesh = 4,
    SolidifyRequest = 5,
    CollapseResult = 6,
    StressWarnings = 7,
    /// Per-slab collapse mesh + fall data. Each result is one slab; multi-
    /// fragment events are queued as N consecutive results.
    CollapseSlabResult = 8,
    /// Localized pre-collapse warning. Drives Acts 1-2 of the cinematic.
    CollapseWarning = 9,
    /// One tier of a 4-tier pile-buildup preview. Cinematic Act 4 — sent in
    /// 4 sequential messages right before the density commit, tier_index 0..3
    /// stored in `slab_fall.pile_tier_index`. UE accumulates by spawn loc and
    /// reveals tiers over `slab_fall.warning_eta_ms` ms.
    CollapsePilePreviewTier = 10,
}

// NOTE: StrutsBroken does not appear in this enum. UE drains broken-strut
// events via `voxel_take_struts_broken` (see api.rs) — the engine stashes
// them in a take-once buffer keyed by world voxel position, avoiding both
// the heap allocation inside the polled FfiResult and the ordering issue
// of interleaving struts with mesh/collapse results.

/// SoA layout for fluid mesh data. Pointers owned by Rust, freed via `voxel_free_result`.
#[repr(C)]
pub struct FfiFluidMeshData {
    pub positions: *mut FfiVec3,
    pub normals: *mut FfiVec3,
    pub fluid_types: *mut u8,
    pub vertex_count: u32,
    pub indices: *mut u32,
    pub index_count: u32,
    pub uvs: *mut [f32; 2],
    pub flow_directions: *mut FfiVec3,
}

#[repr(C)]
pub struct FfiResult {
    pub result_type: FfiResultType,
    pub chunk: FfiChunkCoord,
    pub mesh: FfiMeshData,
    pub mined: FfiMinedMaterials,
    pub generation: u64,
    pub fluid_mesh: FfiFluidMeshData,
    pub crystal_data: FfiCrystalData,
    pub zone_data: FfiZoneData,
    pub mushroom_data: FfiMushroomData,
    pub slab_fall: FfiSlabFallData,
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
    /// Spawn-time worker count. Static — does NOT decrement when a worker
    /// thread panics out. Use `workers_alive` to see live thread count.
    pub worker_threads_active: u32,
    /// Live count of worker threads currently inside the run loop. Drops
    /// below `worker_threads_active` when a worker exhausts its respawn
    /// budget after repeated panics.
    pub workers_alive: u32,
    /// Process-wide cumulative panic count since DLL load. Any nonzero
    /// value means `voxel_panic.log` has details — most likely the cause
    /// of any "stuck queue" symptom.
    pub panics_observed: u32,
}
