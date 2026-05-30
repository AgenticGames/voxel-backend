//! POI / Scene / cinema / predictor / surface-probe FFI types.

use super::*;

/// One Point-of-Interest from the sleep-time scanner. `kind` mirrors
/// `crate::poi_scanner::PoiKind`: 0=Bridge, 1=Lava, 2=Water, 3=Stress,
/// 4=CeilingDome, 5=Chokepoint, 6=WallNiche.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiPoi {
    pub kind: u8,
    pub _padding: [u8; 3],
    pub score: f32,
    pub chunk_coord_ue: FfiChunkCoord,
    pub center_ue: FfiVec3,
    /// "Radius of interest" in UE units — half the bridge length for Bridge
    /// POIs, half-chunk for the per-chunk kinds. The montage camera uses
    /// this to size its orbit so wide bridges get a wide orbit.
    pub extent_radius_ue: f32,
}

// ─── Block 1: voxel-world-memory Scene FFI ──────────────────────────
// New richer Scene model — sub-voxel centroid, AABB, history, tags.
// `voxel_request_scenes` returns these. The legacy `voxel_request_list_top_pois`
// keeps returning `FfiPoi` for UE backward-compat.

/// A semantically-clustered Scene (one fused region) — replaces the
/// flat per-chunk `FfiPoi` for rich consumers. `kind` matches
/// `voxel_world_memory::SceneKind`: same 0..=6 layout as PoiKind.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiScene {
    pub id: u64,
    pub kind: u8,
    pub _padding: [u8; 3],
    pub score: f32,
    pub confidence: f32,
    pub age_secs: u32,
    /// Sub-voxel weighted centroid in UE world space.
    pub centroid_ue: FfiVec3,
    pub aabb_min_ue: FfiVec3,
    pub aabb_max_ue: FfiVec3,
    /// Number of history events captured for this Scene (bounded by
    /// SCENE_HISTORY_CAP=16). UE queries
    /// `voxel_request_scene_history` to get the events themselves.
    pub history_count: u32,
    /// Bitmask of `SceneTags::*` (FRESH/PLAYER_PLACED/NATURAL/SLEEP_EVOLVED).
    pub tag_mask: u64,
}

/// One historical event in a Scene's ring buffer. Tag interpretation:
///   0 = created, 1 = refreshed-via-scan, 2 = event-promoted,
///   3 = cluster-merged.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiSceneHistory {
    pub tag: u8,
    pub _padding: [u8; 3],
    /// Seconds since engine boot.
    pub at_secs: u32,
}

/// Filter for `voxel_request_scenes`. `kind_mask`: bit N set ⇒ kind with
/// discriminant N included. `0xFFFFFFFF` = all kinds. `include_topology`:
/// when 0, CeilingDome/Chokepoint/WallNiche are filtered out (UE doesn't
/// handle them until Block 2). `min_score` and `min_confidence` apply
/// after kind filtering.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiSceneFilter {
    pub kind_mask: u32,
    pub min_score: f32,
    pub min_confidence: f32,
    /// 0 = exclude, nonzero = include topology kinds.
    pub include_topology: u8,
    pub _padding: [u8; 3],
}

/// Push-event payload for `voxel_record_world_event`. `event_kind`:
///   0 = BrushApplied, 1 = AnchorPlaced, 2 = CollapseFired,
///   3 = SleepCompleted, 4 = LavaSpread, 5 = WaterChanged.
/// `kind_hint`: when applicable, hints at the Scene kind affected
/// (0..=6 matching SceneKind). For events that don't have a kind hint,
/// use `0xFF`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiWorldEvent {
    pub event_kind: u8,
    pub kind_hint: u8,
    pub _padding: [u8; 2],
    pub world_pos_ue: FfiVec3,
    /// Auxiliary payload field — interpretation depends on event_kind.
    /// BrushApplied: unused (0). AnchorPlaced: anchor_id (u32 truncated
    /// from u64). CollapseFired: affected_chunks count. SleepCompleted:
    /// dirty_chunk_count.
    pub payload: u32,
}

// ─── Block 1: voxel-cinema Shot Candidate FFI ───────────────────────

/// One waypoint on a camera spline. The full spline lives in a parallel
/// `FfiWaypoint[]` buffer indexed by `FfiShotCandidate.waypoint_offset/_count`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiWaypoint {
    pub pos_ue: FfiVec3,
    pub look_at_ue: FfiVec3,
    pub fov_deg: f32,
    pub t_secs: f32,
    pub dof_focus_dist: f32,
    pub dof_aperture: f32,
}

/// Lighting profile — UE realizes these into a 5-7 point-light rig.
/// `hero_position_intent`: 0=AboveBehind, 1=Below, 2=Frontal,
/// 3=BehindSubject, 4=None.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiLightingProfile {
    pub warmth: f32,
    pub contrast: f32,
    pub key_intensity: f32,
    pub fill_ratio: f32,
    pub hero_position_intent: u8,
    pub _padding: [u8; 3],
}

/// A composed shot candidate. Waypoints live in a parallel buffer (see
/// `FfiWaypoint`). `intent`: 0=SafeOrbit, 1=BridgeTraveling, 2=BridgeAerial,
/// 3=LavaDescent, 4=LavaTopdown, 5=WaterFlowFollow, 6=StressCascade,
/// 7=DomeRevealUp, 8=ChokepointPull, 9=WallNicheStrafe.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiShotCandidate {
    pub intent: u8,
    pub _padding: [u8; 3],
    pub score: f32,
    /// Offset into the caller's waypoint buffer where this shot's
    /// waypoints start.
    pub waypoint_offset: u32,
    pub waypoint_count: u32,
    pub total_duration: f32,
    pub lighting: FfiLightingProfile,
    /// Null-terminated ASCII caption (≤63 chars + NUL).
    pub caption: [u8; 64],
    /// Audio cue tag (UE owns the asset mapping).
    pub audio_cue: u8,
    pub _padding2: [u8; 3],
}

// ─── Block 1: voxel-sleep Predicted Manifest FFI ────────────────────

/// Predictor cache snapshot. UE polls via `voxel_poll_prediction_cache`.
/// Variable-length payload arrays live in parallel caller-owned buffers;
/// this struct just carries the counts + scalar metadata.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiPredictedManifest {
    /// Unix seconds when prediction was computed. UE can reject stale
    /// predictions on its side.
    pub computed_at_secs: u64,
    pub wall_ms: u32,
    pub sleep_count: u32,
    /// Number of likely-changed chunks. Caller passes a parallel buffer
    /// of size N to `voxel_poll_prediction_cache`; this is filled in.
    pub chunks_changed_count: u32,
    pub lava_cells_count: u32,
    pub aureole_block_count: u32,
    pub scene_hints_count: u32,
    /// 1 if the predictor produced a non-empty aureole glimpse, else 0.
    pub has_aureole_glimpse: u8,
    pub _padding: [u8; 3],
    /// Predicted aureole-glimpse position in UE world coords (valid only
    /// if `has_aureole_glimpse == 1`).
    pub aureole_glimpse_ue: FfiVec3,
}

/// Per-scene-hint entry returned by the predictor. Parallel buffer to
/// `FfiPredictedManifest.scene_hints_count`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiSceneHint {
    pub kind: u8,
    pub _padding: [u8; 3],
    pub estimated_score: f32,
    pub world_pos_ue: FfiVec3,
    pub chunk_coord_ue: FfiChunkCoord,
}

/// Voxel-aware surface probe result. Position-independent classification
/// of "what's at this world point": surface kind, averaged normal,
/// largest empty cavity radius around it, and per-axis clearance.
///
/// All output fields are in **UE world space**: normal is the UE-space
/// unit vector, distances are UE units (`world_scale * voxel_units`).
///
/// `kind` mirrors `crate::surface_probe::SurfaceKind`:
///   0 = Solid (inside rock)
///   1 = AirOpen (no solid within 2 voxels in any direction)
///   2 = Floor (rock below, near-vertical up-normal)
///   3 = Wall (rock alongside, near-horizontal normal)
///   4 = Ceiling (rock above, near-vertical down-normal)
///   5 = Overhang (slanted between Floor/Wall or Wall/Ceiling)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiSurfaceProbe {
    pub kind: u8,
    pub _padding: [u8; 3],
    pub normal_x: f32,
    pub normal_y: f32,
    pub normal_z: f32,
    /// Largest empty-sphere radius centered on probe point, in UE units,
    /// capped at the probe's max sampling reach.
    pub cavity_radius: f32,
    /// Distance to nearest solid along UE axes, in UE units, in order
    /// +X, -X, +Y, -Y, +Z, -Z. Capped at the probe's max sampling reach.
    pub clearance_ue: [f32; 6],
}
