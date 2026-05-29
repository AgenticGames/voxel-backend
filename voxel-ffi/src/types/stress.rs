//! Stress / collapse / strut / slab-fall / ore-voxel FFI types.

use super::*;

/// Cinematic-collapse metadata. When `result_type == CollapseSlabResult` the
/// `mesh` field carries the actual DC-extracted slab mesh and this struct
/// carries the metadata needed by the falling-slab actor (spawn pos, landing
/// pos, aspect ratio for tumbling, volume for shake/dust scaling).
///
/// When `result_type == CollapseWarning` the same struct conveys severity +
/// ETA-to-collapse + the bounds of the about-to-fall region so UE can spawn
/// localized warning FX (cracks, dust, creaking).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiSlabFallData {
    // Slab fall data (CollapseSlabResult)
    pub spawn_x: f32,
    pub spawn_y: f32,
    pub spawn_z: f32,
    pub land_x: f32,
    pub land_y: f32,
    pub land_z: f32,
    pub fall_distance: f32,
    /// Bounding-box extents in UE units. Used by the slab actor to compute
    /// aspect ratio for volume-aware fall behavior (tumble, drop speed).
    pub bounds_extent_x: f32,
    pub bounds_extent_y: f32,
    pub bounds_extent_z: f32,
    pub volume: u32,
    pub dominant_material: u8,
    // Warning event fields (CollapseWarning)
    /// 0=none, 1=dust, 2=creak, 3=shake, 4=imminent
    pub warning_severity: u8,
    /// Tier index (0..7) for CollapsePilePreviewTier results. Otherwise 0.
    pub pile_tier_index: u8,
    pub _padding: [u8; 1],
    /// Estimated milliseconds until the actual collapse fires. UE uses this
    /// to time the warning state-machine (act 1 → act 2 → fall).
    /// For CollapsePilePreviewTier, this is the total tier-reveal duration ms.
    pub warning_eta_ms: u32,
    /// Leading-edge horizontal unit vector in **UE world space**, indicating
    /// the direction the slab "leans" — long edge offset from the centroid
    /// in the direction it will tip while falling. Magnitude in [0..1]; a
    /// magnitude of 0 means no preferred tilt direction (chunky cube).
    /// UE uses this to pick the tumble axis so long thin slabs tip like
    /// dominoes mid-fall instead of randomly jittering.
    pub leading_edge_dir_x: f32,
    pub leading_edge_dir_y: f32,
}

impl Default for FfiSlabFallData {
    fn default() -> Self {
        Self {
            spawn_x: 0.0, spawn_y: 0.0, spawn_z: 0.0,
            land_x: 0.0, land_y: 0.0, land_z: 0.0,
            fall_distance: 0.0,
            bounds_extent_x: 0.0, bounds_extent_y: 0.0, bounds_extent_z: 0.0,
            volume: 0,
            dominant_material: 0,
            warning_severity: 0,
            pile_tier_index: 0,
            _padding: [0; 1],
            warning_eta_ms: 0,
            leading_edge_dir_x: 0.0,
            leading_edge_dir_y: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct FfiStressData {
    pub stress_values: *mut f32,
    pub classification: *mut u8,  // Per-voxel: top 4 = surface type, bottom 4 = stress source
    pub count: u32,
    pub valid: u32,
    /// Player-painted additive stress overlay (creative PaintStress brush).
    /// `painted_values` is null if the chunk has no painted layer (treat as
    /// all-zeros). When non-null, length matches `count` and the effective
    /// stress at voxel i is `stress_values[i] + painted_values[i]`.
    pub painted_values: *mut f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiCollapseEvent {
    pub center_x: f32,
    pub center_y: f32,
    pub center_z: f32,
    pub volume: u32,
}

/// Per-voxel stress warning sent to UE for visual/audio feedback.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiStressWarning {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub stress: f32,
    pub warning_type: u8, // 0=none, 1=dust, 2=creak, 3=shake
}

/// A coherent collapse slab with mesh data for animated falling.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct FfiCollapseSlab {
    /// Slab mesh data (ProceduralMesh on UE side)
    pub positions: *mut FfiVec3,
    pub normals: *mut FfiVec3,
    pub material_ids: *mut u8,
    pub vertex_count: u32,
    pub indices: *mut u32,
    pub index_count: u32,
    pub submeshes: *mut FfiSubmesh,
    pub submesh_count: u32,
    /// Spawn position in UE world space (where slab appears initially)
    pub spawn_x: f32,
    pub spawn_y: f32,
    pub spawn_z: f32,
    /// Landing position in UE world space (where slab comes to rest)
    pub land_x: f32,
    pub land_y: f32,
    pub land_z: f32,
    /// Fall distance in UE world units
    pub fall_distance: f32,
    /// Slab volume (number of voxels)
    pub volume: u32,
    /// Dominant material index
    pub dominant_material: u8,
}

// SAFETY: FfiCollapseSlab's raw pointers are exclusively owned by the result
// and only dereferenced on the FFI boundary. Not shared across threads.
unsafe impl Send for FfiCollapseSlab {}
unsafe impl Sync for FfiCollapseSlab {}

/// V2 collapse event with coherent slab data for animated falling.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct FfiCollapseEventV2 {
    pub center_x: f32,
    pub center_y: f32,
    pub center_z: f32,
    pub total_volume: u32,
    pub slabs: *mut FfiCollapseSlab,
    pub slab_count: u32,
}

unsafe impl Send for FfiCollapseEventV2 {}
unsafe impl Sync for FfiCollapseEventV2 {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiStressConfig {
    pub material_hardness: [f32; 50],
    pub gravity_weight: f32,
    pub lateral_support_factor: f32,
    pub vertical_support_factor: f32,
    pub support_radius: u32,
    pub propagation_radius: u32,
    pub max_collapse_volume: u32,
    pub rubble_enabled: u32,  // bool as u32 for C ABI
    pub rubble_fill_ratio: f32,
    pub warn_dust_threshold: f32,
    pub warn_creak_threshold: f32,
    pub warn_shake_threshold: f32,
    /// LEGACY ABI slot — kept for layout stability. Pre-2026-05-26 stress
    /// system used a single per-tier hardness array. New system uses
    /// `STRUT_TUNING` per-tier struct (in voxel-core/src/stress.rs). UE side
    /// should set this to all zeros; internal math ignores it now.
    pub support_hardness: [f32; 6],
    // V2 fields
    pub lateral_transfer_factor: f32,
    pub vertical_transfer_factor: f32,
    pub support_propagation_iterations: u32,
    pub ground_threshold: f32,
    pub overhang_weight: f32,
    pub span_weight: f32,
    pub min_safe_span: u32,
    pub min_collapse_region: u32,
    pub slab_cohesion_threshold: f32,
    pub cross_section_weight: f32,
    pub cross_section_min_faces: u32,
    pub surface_y: i32,
    pub depth_pressure_scale: f32,
    // Cinematic mining pipeline scan buffer. UE-tuneable. See
    // voxel-core/src/stress.rs StressConfig::mining_stress_scan_buffer for the
    // semantic note and worker.rs WorkerRequest::Mine for where it's used.
    // ⚠ Two stress systems coexist: this drives the SlabFall pipeline,
    // `propagation_radius` above only drives the legacy sleep pipeline.
    pub mining_stress_scan_buffer: u32,
}

/// One surface-facing ore voxel returned by `voxel_query_ore_voxels`.
/// Position is in UE world space; material_index is the raw `Material as u8`.
/// Layout: 12 bytes for x/y/z + 1 byte material + 3 bytes tail padding = 16 bytes.
/// UE side must mirror with `float X, Y, Z; uint8 MaterialIndex; uint8 _pad[3];`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiOreVoxel {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub material_index: u8,
    pub _pad: [u8; 3],
}

/// Result list for `voxel_query_ore_voxels`. Caller MUST free via
/// `voxel_free_ore_voxel_list`. `voxels` is null and `count` is 0 when empty.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct FfiOreVoxelList {
    pub voxels: *mut FfiOreVoxel,
    pub count: u32,
}

// SAFETY: pointer is exclusively owned by the result and only dereferenced
// on the FFI boundary by the UE caller. Not shared across threads.
unsafe impl Send for FfiOreVoxelList {}
unsafe impl Sync for FfiOreVoxelList {}

/// One broken strut event for `WorkerResult::StrutsBroken`.
/// Position is reported in WORLD VOXEL coords (Rust frame). UE converts to
/// world space via `RustToUE` + WorldScale and indexes `PlacedSupports`.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FfiStrutBroken {
    pub world_x: i32,
    pub world_y: i32,
    pub world_z: i32,
    /// SupportType byte (Copper=1 .. Mithril=5). 0 = None, never emitted.
    pub support_type: u8,
    /// Why the strut broke: 0 = load decay (recalc-time HP exhaustion),
    /// 1 = BFS halt (cinematic mining absorbed the slab).
    pub source: u8,
    pub _pad: [u8; 2],
}

/// FFI inspect result for `voxel_query_strut_hp`. UE renders a small HP bar
/// over the strut when the player aims at one. Lock-contention is signalled
/// via `valid` (0 = retry blocked, 1 = ok, type may still be None).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FfiStrutInfo {
    /// SupportType byte at the queried voxel. 0 = no strut here.
    pub support_type: u8,
    pub _pad: [u8; 1],
    pub hp: u16,
    pub max_hp: u16,
    /// 0 = lock contended (UE should preserve prior bar / hide).
    /// 1 = read OK (treat `support_type==0` as "no strut here, hide the bar").
    pub valid: u8,
    pub _pad2: [u8; 1],
}

/// A single voxel cell that has crossed the collapse stress threshold
/// (effective stress >= 1.0). Used to drive UE-side stress-crack decals
/// and per-cell warning dust effects on chunks that are primed to collapse.
///
/// Position + normal are already in UE world space. `stress` is the
/// effective stress value (base + painted overlay) — typically 1.0-2.0.
/// Interior (non-surface-exposed) cells are filtered out by the caller
/// since they have no visible surface to decorate.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiOverstressedCell {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub normal_x: f32,
    pub normal_y: f32,
    pub normal_z: f32,
    pub stress: f32,
    /// Surface kind from the existing surface-probe enum: 1=Floor, 2=Ceiling,
    /// 3=Wall, 4=Thin (matches the classification top-4 bits). Used by UE to
    /// tweak decal scale + orientation hints per surface type.
    pub surface_kind: u8,
    pub _padding: [u8; 3],
}

/// Heap-allocated list of overstressed cells returned by
/// `voxel_list_overstressed_in_chunk` and `voxel_list_overstressed_in_region`.
/// Caller MUST call `voxel_free_overstressed_list` to release.
///
/// `valid` distinguishes "store was read OK" from "store lock was contended":
///   valid=1, count>0  -> N over-stress cells found
///   valid=1, count=0  -> store read OK, no over-stress cells in this region
///   valid=0           -> store lock contended after retries; caller should
///                        SKIP its overlay refresh (preserve existing decals)
///                        rather than clear to empty. Avoids the "decals
///                        disappear on paint" race where the brush worker
///                        held the write lock when UE polled.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiOverstressedList {
    pub cells: *mut FfiOverstressedCell,
    pub count: u32,
    pub valid: u32,
}
