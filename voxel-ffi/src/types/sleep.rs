//! Sleep / morph result FFI types.

use super::*;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiSleepProgress {
    pub phase: u8,            // 0=reaction, 1=aureole, 2=veins, 3=deeptime, 4=done
    pub progress_pct: f32,    // 0.0 - 1.0
    pub chunks_processed: u32,
    pub chunks_total: u32,
    pub glimpse_chunk: FfiChunkCoord,  // Chunk where interesting transform happened
    pub glimpse_type: u8,     // 0=none, 1=acid_dissolution, 2=metamorphism, 3=vein_deposit, 4=enrichment, 5=collapse
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct FfiSleepResult {
    pub success: u32,
    pub chunks_changed: u32,
    pub voxels_metamorphosed: u32,
    pub minerals_grown: u32,
    pub supports_degraded: u32,
    pub collapses_triggered: u32,
    pub acid_dissolved: u32,
    pub veins_deposited: u32,
    pub voxels_enriched: u32,
    pub formations_grown: u32,
    pub sulfide_dissolved: u32,
    pub coal_matured: u32,
    pub diamonds_formed: u32,
    pub voxels_silicified: u32,
    pub nests_fossilized: u32,
    pub channels_eroded: u32,
    pub corpses_fossilized: u32,
    pub lava_solidified: u32,
    pub dirty_chunks: *mut FfiChunkCoord,
    pub dirty_chunk_count: u32,
    pub collapse_events: *mut FfiCollapseEvent,
    pub collapse_event_count: u32,
    pub profile_report: *mut std::ffi::c_char,
    pub profile_report_length: u32,
    pub has_aureole_glimpse: u32,
    pub aureole_glimpse_x: i32,
    pub aureole_glimpse_y: i32,
    pub aureole_glimpse_z: i32,
    // Showcase block coords (heap-allocated, 27 entries for 3x3x3 block)
    pub has_aureole_block: u32,
    pub aureole_block: *mut FfiChunkCoord,
    pub aureole_block_count: u32,
    // Compacted manifest JSON for morph system
    pub manifest_json: *mut std::ffi::c_char,
    pub manifest_json_length: u32,
    // Lava cell world voxel positions (for montage lava mesh)
    pub lava_cells: *mut FfiChunkCoord,
    pub lava_cell_count: u32,
    // Surface-exposed changed voxel world positions (for montage camera
    // framing — see SleepResult::surface_changed_cells). Packed as
    // FfiChunkCoord for transport (it's voxel coords, not chunk coords).
    // APPENDED AT END — UE's FVoxelSleepResult must mirror this exact order.
    pub surface_changed_cells: *mut FfiChunkCoord,
    pub surface_changed_cell_count: u32,
    // Per-t surface-activity histogram (fixed 64 buckets, no alloc/free).
    // bucket b ≈ reveal time t = b/64. The montage culls dead reveal steps
    // (steps whose t-window holds <1% of total activity). UE mirrors this as
    // uint16 SurfaceActivity[64] — same fixed length, same position at END.
    pub surface_activity: [u16; voxel_sleep::SURFACE_ACTIVITY_BUCKETS],
}

/// Morph step result: 8 meshes (one per showcase chunk) for progressive morphing.
/// Heap-allocated array of FfiMeshData — caller must free via voxel_free_morph_result.
#[repr(C)]
pub struct FfiMorphResult {
    pub step: u32,
    pub total_steps: u32,
    pub chunk_count: u32,
    pub meshes: *mut FfiMeshData,  // heap array, length = chunk_count
}
