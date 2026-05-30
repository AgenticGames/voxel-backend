//! Decoration FFI types (crystals, mushrooms, zones, crystal-anchor bridge).

use super::*;

/// Single crystal placement in UE coordinate space.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiCrystalPlacement {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub normal_x: f32,
    pub normal_y: f32,
    pub normal_z: f32,
    pub ore_type: u8,
    pub size_class: u8,
    pub scale: f32,
}

/// Crystal placement data for a chunk. Pointer owned by Rust, freed via voxel_free_result.
///
/// `hash` is a stable hash of the placement set (FNV-style over each
/// placement's bytes). UE caches the last applied hash per chunk and
/// **skips** `ApplyCrystalData` (the expensive HISM rebuild +
/// `Foliage Create Proxy`) when the incoming hash matches — at scale
/// (30-event burst) this dropped HISM proxy rebuilds from ~11K to ~1K.
/// `hash` of zero means "skip the optimization, just apply" (used as a
/// safety value when computation is uncertain).
#[repr(C)]
pub struct FfiCrystalData {
    pub placements: *mut FfiCrystalPlacement,
    pub count: u32,
    pub _padding: u32,   // align hash to 8 bytes for repr(C) parity with UE
    pub hash: u64,
}

/// One placed mushroom instance in chunk-relative voxel coordinates
/// (Rust Y-up). UE applies the world-scale + coord swap on the consumer
/// side. `kind` is `MushroomKind as u8` — see voxel-gen `MushroomKind`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiMushroomInstance {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub normal_x: f32,
    pub normal_y: f32,
    pub normal_z: f32,
    pub scale: f32,
    pub yaw: f32,
    pub kind: u8,
    pub anchor_lx: u8,
    pub anchor_ly: u8,
    pub anchor_lz: u8,
}

/// Mushroom placement data for a chunk. Pointer owned by Rust, freed via
/// `voxel_free_result`. Mirrors `FfiCrystalData` layout (with the same
/// hash-skip optimization for HISM rebuild cost).
#[repr(C)]
pub struct FfiMushroomData {
    pub instances: *mut FfiMushroomInstance,
    pub count: u32,
    pub _padding: u32,
    pub hash: u64,
}

/// Zone descriptor for UE consumption — one per detected zone in a region.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiZoneDescriptor {
    pub zone_type: u8,
    pub center_x: f32,
    pub center_y: f32,
    pub center_z: f32,
    pub min_x: f32,
    pub min_y: f32,
    pub min_z: f32,
    pub max_x: f32,
    pub max_y: f32,
    pub max_z: f32,
}

/// Zone data container. Pointer owned by Rust, freed via voxel_free_result.
#[repr(C)]
pub struct FfiZoneData {
    pub descriptors: *mut FfiZoneDescriptor,
    pub count: u32,
}

// FfiZoneDescriptor is defined near the top of this file, alongside FfiZoneData.

/// Anchor point for zone rendering (bioluminescent lights, etc.).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiZoneAnchor {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
}

// ─── Crystal Growth Bridge (Crystal Anchor) FFI structs ─────────────────────
// All position fields are UE world space (Z-up left-hand, world_scale units).
// Mirror of `crate::crystal_anchors::PlaceAnchorError` for the FFI layer.

/// Result of voxel_request_place_crystal_anchor. `error_code` mirrors
/// `crate::crystal_anchors::PlaceAnchorError`:
///     0 = Ok, 1 = TooFarFromPartner, 2 = CapReached,
///     3 = NoSolidUnder, 4 = DuplicateTooClose.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiCrystalAnchorResult {
    pub error_code: u8,
    pub _padding: [u8; 3],
    /// Set only when error_code == 0. Otherwise 0.
    pub anchor_id: u64,
    pub partner_id: u64,
    pub pair_token: u64,
    /// 1 if this throw completed a pair, else 0.
    pub pair_completed: u8,
    pub _padding2: [u8; 7],
}

/// One pending or grown bridge pair (same layout for both query types).
/// UE-space positions; midpoint is the arch-lifted bridge focal point.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiCrystalBridgePair {
    pub pair_token: u64,
    pub anchor_a_id: u64,
    pub anchor_b_id: u64,
    pub anchor_a_pos_ue: FfiVec3,
    pub anchor_b_pos_ue: FfiVec3,
    pub midpoint_ue: FfiVec3,
}
