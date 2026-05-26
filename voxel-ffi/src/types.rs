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
    pub worms_per_region: f32,
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
    // ── Fluid Config (16 fields) ──
    pub fluid_tick_rate: f32,
    pub fluid_lava_tick_divisor: u8,
    pub fluid_water_spring_threshold: f64,
    pub fluid_lava_source_threshold: f64,
    pub fluid_lava_depth_max: f64,
    // New fluid fields
    pub fluid_water_noise_frequency: f64,
    pub fluid_water_depth_min: f64,
    pub fluid_water_depth_max: f64,
    pub fluid_water_flow_rate: f32,
    pub fluid_water_spread_rate: f32,
    pub fluid_lava_noise_frequency: f64,
    pub fluid_lava_depth_min: f64,
    pub fluid_lava_flow_rate: f32,
    pub fluid_lava_spread_rate: f32,
    pub fluid_cavern_source_bias: f64,
    pub fluid_tunnel_bend_threshold: f64,
    // ── Mine Config (4 fields) ──
    pub mine_smooth_iterations: u32,
    pub mine_smooth_strength: f32,
    pub mine_min_triangle_area: f32,
    pub mine_dirty_expand: u32,
    // ── Bounds Size ──
    pub bounds_size: f32,
    // ── Ore Visual Quality (4 fields) ──
    pub ore_domain_warp_strength: f64,
    pub ore_warp_frequency: f64,
    pub ore_edge_falloff: f64,
    pub ore_detail_weight: f64,
    // ── Mesh Smoothing (4 fields) ──
    pub mesh_smooth_iterations: u32,
    pub mesh_smooth_strength: f32,
    pub mesh_boundary_smooth: f32,
    pub mesh_recalc_normals: u32,
    // ── Pool Config (12 fields) ──
    pub pool_enabled: u8,           // 0=disabled, nonzero=enabled
    pub pool_placement_freq: f64,
    pub pool_placement_thresh: f64,
    pub pool_chance: f32,
    pub pool_min_area: u32,
    pub pool_max_radius: u32,
    pub pool_basin_depth: u32,
    pub pool_rim_height: u32,
    pub pool_water_pct: f32,
    pub pool_lava_pct: f32,
    pub pool_empty_pct: f32,
    pub pool_min_air_above: u32,
    pub pool_max_cave_height: u32,
    pub pool_min_floor_thickness: u32,
    pub pool_min_ground_depth: u32,
    pub pool_max_y_step: u32,
    pub pool_footprint_y_tolerance: u32,
    // ── Formation Config (42 fields) ──
    pub formation_enabled: u8,
    pub formation_placement_frequency: f32,
    pub formation_placement_threshold: f32,
    pub formation_stalactite_chance: f32,
    pub formation_stalagmite_chance: f32,
    pub formation_flowstone_chance: f32,
    pub formation_column_chance: f32,
    pub formation_column_max_gap: u32,
    pub formation_length_min: f32,
    pub formation_length_max: f32,
    pub formation_radius_min: f32,
    pub formation_radius_max: f32,
    pub formation_max_radius: f32,
    pub formation_column_radius_min: f32,
    pub formation_column_radius_max: f32,
    pub formation_flowstone_length_min: f32,
    pub formation_flowstone_length_max: f32,
    pub formation_flowstone_thickness: f32,
    pub formation_min_air_gap: u32,
    pub formation_min_clearance: u32,
    pub formation_smoothness: f32,
    // New formation fields
    pub formation_mega_column_chance: f32,
    pub formation_mega_column_min_gap: u32,
    pub formation_mega_column_radius_min: f32,
    pub formation_mega_column_radius_max: f32,
    pub formation_mega_column_noise_strength: f32,
    pub formation_mega_column_ring_frequency: f32,
    pub formation_drapery_chance: f32,
    pub formation_drapery_length_min: f32,
    pub formation_drapery_length_max: f32,
    pub formation_drapery_wave_frequency: f32,
    pub formation_drapery_wave_amplitude: f32,
    pub formation_rimstone_chance: f32,
    pub formation_rimstone_dam_height_min: f32,
    pub formation_rimstone_dam_height_max: f32,
    pub formation_rimstone_pool_depth: f32,
    pub formation_rimstone_min_slope: f32,
    pub formation_shield_chance: f32,
    pub formation_shield_radius_min: f32,
    pub formation_shield_radius_max: f32,
    pub formation_shield_max_tilt: f32,
    pub formation_shield_stalactite_chance: f32,
    // Cauldron (11 fields)
    pub formation_cauldron_chance: f32,
    pub formation_cauldron_radius_min: f32,
    pub formation_cauldron_radius_max: f32,
    pub formation_cauldron_depth: f32,
    pub formation_cauldron_lip_height: f32,
    pub formation_cauldron_rim_stalagmite_count_min: u32,
    pub formation_cauldron_rim_stalagmite_count_max: u32,
    pub formation_cauldron_rim_stalagmite_scale: f32,
    pub formation_cauldron_floor_noise: f32,
    pub formation_cauldron_water_chance: f32,
    pub formation_cauldron_lava_chance: f32,
    // ── Geological Realism Toggles (14 fields, u8 booleans) ──
    pub ore_iron_sedimentary_only: u8,
    pub ore_iron_depth_fade: u8,
    pub ore_copper_supergene: u8,
    pub ore_copper_granite_contact: u8,
    pub ore_malachite_depth_bias: u8,
    pub ore_kimberlite_carrot_taper: u8,
    pub ore_diamond_depth_grade: u8,
    pub ore_sulfide_gossan_cap: u8,
    pub ore_sulfide_disseminated: u8,
    pub ore_pyrite_ore_halo: u8,
    pub ore_quartz_planar_veins: u8,
    pub ore_gold_bonanza: u8,
    pub ore_geode_volcanic_host: u8,
    pub ore_geode_depth_scaling: u8,
    // Coal (4 params + 3 toggles)
    pub ore_coal_frequency: f64,
    pub ore_coal_threshold: f64,
    pub ore_coal_depth_min: f64,
    pub ore_coal_depth_max: f64,
    pub ore_coal_sedimentary_host: u8,
    pub ore_coal_shallow_ceiling: u8,
    pub ore_coal_depth_enrichment: u8,
    // ── Ore Detail ──
    pub ore_detail_multiplier: u32,
    pub ore_protrusion: f32,
    // ── Crystal Config (229 fields: 1 master + 12 ores × 19 fields) ──
    pub crystal_enabled: u8,
    // Iron crystals
    pub crystal_iron_enabled: u8,
    pub crystal_iron_chance: f32,
    pub crystal_iron_density_threshold: f32,
    pub crystal_iron_scale_min: f32,
    pub crystal_iron_scale_max: f32,
    pub crystal_iron_small_weight: f32,
    pub crystal_iron_medium_weight: f32,
    pub crystal_iron_large_weight: f32,
    pub crystal_iron_normal_alignment: f32,
    pub crystal_iron_cluster_size: u32,
    pub crystal_iron_cluster_radius: f32,
    pub crystal_iron_surface_offset: f32,
    pub crystal_iron_vein_enabled: u8,
    pub crystal_iron_vein_frequency: f32,
    pub crystal_iron_vein_thickness: f32,
    pub crystal_iron_vein_octaves: u32,
    pub crystal_iron_vein_lacunarity: f32,
    pub crystal_iron_vein_warp_strength: f32,
    pub crystal_iron_vein_density: f32,
    // Copper crystals
    pub crystal_copper_enabled: u8,
    pub crystal_copper_chance: f32,
    pub crystal_copper_density_threshold: f32,
    pub crystal_copper_scale_min: f32,
    pub crystal_copper_scale_max: f32,
    pub crystal_copper_small_weight: f32,
    pub crystal_copper_medium_weight: f32,
    pub crystal_copper_large_weight: f32,
    pub crystal_copper_normal_alignment: f32,
    pub crystal_copper_cluster_size: u32,
    pub crystal_copper_cluster_radius: f32,
    pub crystal_copper_surface_offset: f32,
    pub crystal_copper_vein_enabled: u8,
    pub crystal_copper_vein_frequency: f32,
    pub crystal_copper_vein_thickness: f32,
    pub crystal_copper_vein_octaves: u32,
    pub crystal_copper_vein_lacunarity: f32,
    pub crystal_copper_vein_warp_strength: f32,
    pub crystal_copper_vein_density: f32,
    // Malachite crystals
    pub crystal_malachite_enabled: u8,
    pub crystal_malachite_chance: f32,
    pub crystal_malachite_density_threshold: f32,
    pub crystal_malachite_scale_min: f32,
    pub crystal_malachite_scale_max: f32,
    pub crystal_malachite_small_weight: f32,
    pub crystal_malachite_medium_weight: f32,
    pub crystal_malachite_large_weight: f32,
    pub crystal_malachite_normal_alignment: f32,
    pub crystal_malachite_cluster_size: u32,
    pub crystal_malachite_cluster_radius: f32,
    pub crystal_malachite_surface_offset: f32,
    pub crystal_malachite_vein_enabled: u8,
    pub crystal_malachite_vein_frequency: f32,
    pub crystal_malachite_vein_thickness: f32,
    pub crystal_malachite_vein_octaves: u32,
    pub crystal_malachite_vein_lacunarity: f32,
    pub crystal_malachite_vein_warp_strength: f32,
    pub crystal_malachite_vein_density: f32,
    // Tin crystals
    pub crystal_tin_enabled: u8,
    pub crystal_tin_chance: f32,
    pub crystal_tin_density_threshold: f32,
    pub crystal_tin_scale_min: f32,
    pub crystal_tin_scale_max: f32,
    pub crystal_tin_small_weight: f32,
    pub crystal_tin_medium_weight: f32,
    pub crystal_tin_large_weight: f32,
    pub crystal_tin_normal_alignment: f32,
    pub crystal_tin_cluster_size: u32,
    pub crystal_tin_cluster_radius: f32,
    pub crystal_tin_surface_offset: f32,
    pub crystal_tin_vein_enabled: u8,
    pub crystal_tin_vein_frequency: f32,
    pub crystal_tin_vein_thickness: f32,
    pub crystal_tin_vein_octaves: u32,
    pub crystal_tin_vein_lacunarity: f32,
    pub crystal_tin_vein_warp_strength: f32,
    pub crystal_tin_vein_density: f32,
    // Gold crystals
    pub crystal_gold_enabled: u8,
    pub crystal_gold_chance: f32,
    pub crystal_gold_density_threshold: f32,
    pub crystal_gold_scale_min: f32,
    pub crystal_gold_scale_max: f32,
    pub crystal_gold_small_weight: f32,
    pub crystal_gold_medium_weight: f32,
    pub crystal_gold_large_weight: f32,
    pub crystal_gold_normal_alignment: f32,
    pub crystal_gold_cluster_size: u32,
    pub crystal_gold_cluster_radius: f32,
    pub crystal_gold_surface_offset: f32,
    pub crystal_gold_vein_enabled: u8,
    pub crystal_gold_vein_frequency: f32,
    pub crystal_gold_vein_thickness: f32,
    pub crystal_gold_vein_octaves: u32,
    pub crystal_gold_vein_lacunarity: f32,
    pub crystal_gold_vein_warp_strength: f32,
    pub crystal_gold_vein_density: f32,
    // Diamond crystals
    pub crystal_diamond_enabled: u8,
    pub crystal_diamond_chance: f32,
    pub crystal_diamond_density_threshold: f32,
    pub crystal_diamond_scale_min: f32,
    pub crystal_diamond_scale_max: f32,
    pub crystal_diamond_small_weight: f32,
    pub crystal_diamond_medium_weight: f32,
    pub crystal_diamond_large_weight: f32,
    pub crystal_diamond_normal_alignment: f32,
    pub crystal_diamond_cluster_size: u32,
    pub crystal_diamond_cluster_radius: f32,
    pub crystal_diamond_surface_offset: f32,
    pub crystal_diamond_vein_enabled: u8,
    pub crystal_diamond_vein_frequency: f32,
    pub crystal_diamond_vein_thickness: f32,
    pub crystal_diamond_vein_octaves: u32,
    pub crystal_diamond_vein_lacunarity: f32,
    pub crystal_diamond_vein_warp_strength: f32,
    pub crystal_diamond_vein_density: f32,
    // Kimberlite crystals
    pub crystal_kimberlite_enabled: u8,
    pub crystal_kimberlite_chance: f32,
    pub crystal_kimberlite_density_threshold: f32,
    pub crystal_kimberlite_scale_min: f32,
    pub crystal_kimberlite_scale_max: f32,
    pub crystal_kimberlite_small_weight: f32,
    pub crystal_kimberlite_medium_weight: f32,
    pub crystal_kimberlite_large_weight: f32,
    pub crystal_kimberlite_normal_alignment: f32,
    pub crystal_kimberlite_cluster_size: u32,
    pub crystal_kimberlite_cluster_radius: f32,
    pub crystal_kimberlite_surface_offset: f32,
    pub crystal_kimberlite_vein_enabled: u8,
    pub crystal_kimberlite_vein_frequency: f32,
    pub crystal_kimberlite_vein_thickness: f32,
    pub crystal_kimberlite_vein_octaves: u32,
    pub crystal_kimberlite_vein_lacunarity: f32,
    pub crystal_kimberlite_vein_warp_strength: f32,
    pub crystal_kimberlite_vein_density: f32,
    // Sulfide crystals
    pub crystal_sulfide_enabled: u8,
    pub crystal_sulfide_chance: f32,
    pub crystal_sulfide_density_threshold: f32,
    pub crystal_sulfide_scale_min: f32,
    pub crystal_sulfide_scale_max: f32,
    pub crystal_sulfide_small_weight: f32,
    pub crystal_sulfide_medium_weight: f32,
    pub crystal_sulfide_large_weight: f32,
    pub crystal_sulfide_normal_alignment: f32,
    pub crystal_sulfide_cluster_size: u32,
    pub crystal_sulfide_cluster_radius: f32,
    pub crystal_sulfide_surface_offset: f32,
    pub crystal_sulfide_vein_enabled: u8,
    pub crystal_sulfide_vein_frequency: f32,
    pub crystal_sulfide_vein_thickness: f32,
    pub crystal_sulfide_vein_octaves: u32,
    pub crystal_sulfide_vein_lacunarity: f32,
    pub crystal_sulfide_vein_warp_strength: f32,
    pub crystal_sulfide_vein_density: f32,
    // Quartz crystals
    pub crystal_quartz_enabled: u8,
    pub crystal_quartz_chance: f32,
    pub crystal_quartz_density_threshold: f32,
    pub crystal_quartz_scale_min: f32,
    pub crystal_quartz_scale_max: f32,
    pub crystal_quartz_small_weight: f32,
    pub crystal_quartz_medium_weight: f32,
    pub crystal_quartz_large_weight: f32,
    pub crystal_quartz_normal_alignment: f32,
    pub crystal_quartz_cluster_size: u32,
    pub crystal_quartz_cluster_radius: f32,
    pub crystal_quartz_surface_offset: f32,
    pub crystal_quartz_vein_enabled: u8,
    pub crystal_quartz_vein_frequency: f32,
    pub crystal_quartz_vein_thickness: f32,
    pub crystal_quartz_vein_octaves: u32,
    pub crystal_quartz_vein_lacunarity: f32,
    pub crystal_quartz_vein_warp_strength: f32,
    pub crystal_quartz_vein_density: f32,
    // Pyrite crystals
    pub crystal_pyrite_enabled: u8,
    pub crystal_pyrite_chance: f32,
    pub crystal_pyrite_density_threshold: f32,
    pub crystal_pyrite_scale_min: f32,
    pub crystal_pyrite_scale_max: f32,
    pub crystal_pyrite_small_weight: f32,
    pub crystal_pyrite_medium_weight: f32,
    pub crystal_pyrite_large_weight: f32,
    pub crystal_pyrite_normal_alignment: f32,
    pub crystal_pyrite_cluster_size: u32,
    pub crystal_pyrite_cluster_radius: f32,
    pub crystal_pyrite_surface_offset: f32,
    pub crystal_pyrite_vein_enabled: u8,
    pub crystal_pyrite_vein_frequency: f32,
    pub crystal_pyrite_vein_thickness: f32,
    pub crystal_pyrite_vein_octaves: u32,
    pub crystal_pyrite_vein_lacunarity: f32,
    pub crystal_pyrite_vein_warp_strength: f32,
    pub crystal_pyrite_vein_density: f32,
    // Amethyst crystals
    pub crystal_amethyst_enabled: u8,
    pub crystal_amethyst_chance: f32,
    pub crystal_amethyst_density_threshold: f32,
    pub crystal_amethyst_scale_min: f32,
    pub crystal_amethyst_scale_max: f32,
    pub crystal_amethyst_small_weight: f32,
    pub crystal_amethyst_medium_weight: f32,
    pub crystal_amethyst_large_weight: f32,
    pub crystal_amethyst_normal_alignment: f32,
    pub crystal_amethyst_cluster_size: u32,
    pub crystal_amethyst_cluster_radius: f32,
    pub crystal_amethyst_surface_offset: f32,
    pub crystal_amethyst_vein_enabled: u8,
    pub crystal_amethyst_vein_frequency: f32,
    pub crystal_amethyst_vein_thickness: f32,
    pub crystal_amethyst_vein_octaves: u32,
    pub crystal_amethyst_vein_lacunarity: f32,
    pub crystal_amethyst_vein_warp_strength: f32,
    pub crystal_amethyst_vein_density: f32,
    // Coal crystals
    pub crystal_coal_enabled: u8,
    pub crystal_coal_chance: f32,
    pub crystal_coal_density_threshold: f32,
    pub crystal_coal_scale_min: f32,
    pub crystal_coal_scale_max: f32,
    pub crystal_coal_small_weight: f32,
    pub crystal_coal_medium_weight: f32,
    pub crystal_coal_large_weight: f32,
    pub crystal_coal_normal_alignment: f32,
    pub crystal_coal_cluster_size: u32,
    pub crystal_coal_cluster_radius: f32,
    pub crystal_coal_surface_offset: f32,
    pub crystal_coal_vein_enabled: u8,
    pub crystal_coal_vein_frequency: f32,
    pub crystal_coal_vein_thickness: f32,
    pub crystal_coal_vein_octaves: u32,
    pub crystal_coal_vein_lacunarity: f32,
    pub crystal_coal_vein_warp_strength: f32,
    pub crystal_coal_vein_density: f32,
    // ── Sleep Config ──
    // Top-level sleep
    pub sleep_time_budget_ms: u32,
    pub sleep_chunk_radius: u32,
    pub sleep_metamorphism_enabled: u8,
    pub sleep_minerals_enabled: u8,
    pub sleep_collapse_enabled: u8,
    pub sleep_count: u32,
    // Metamorphism
    pub sleep_limestone_to_marble_prob: f32,
    pub sleep_limestone_to_marble_depth: f32,
    pub sleep_limestone_to_marble_enabled: u8,
    pub sleep_sandstone_to_granite_prob: f32,
    pub sleep_sandstone_to_granite_depth: f32,
    pub sleep_sandstone_to_granite_min_neighbors: u32,
    pub sleep_sandstone_to_granite_enabled: u8,
    pub sleep_slate_to_marble_prob: f32,
    pub sleep_slate_to_marble_enabled: u8,
    pub sleep_granite_to_basalt_prob: f32,
    pub sleep_granite_to_basalt_min_air: u32,
    pub sleep_granite_to_basalt_enabled: u8,
    pub sleep_iron_to_pyrite_prob: f32,
    pub sleep_iron_to_pyrite_search_radius: u32,
    pub sleep_iron_to_pyrite_enabled: u8,
    pub sleep_copper_to_malachite_prob: f32,
    pub sleep_copper_to_malachite_enabled: u8,
    // Minerals
    pub sleep_crystal_growth_max: u32,
    pub sleep_crystal_growth_enabled: u8,
    pub sleep_crystal_growth_prob: f32,
    pub sleep_malachite_stalactite_max: u32,
    pub sleep_malachite_stalactite_enabled: u8,
    pub sleep_malachite_stalactite_prob: f32,
    pub sleep_quartz_extension_prob: f32,
    pub sleep_quartz_extension_max: u32,
    pub sleep_quartz_extension_enabled: u8,
    pub sleep_calcite_infill_max: u32,
    pub sleep_calcite_infill_depth: f32,
    pub sleep_calcite_infill_min_faces: u32,
    pub sleep_calcite_infill_enabled: u8,
    pub sleep_calcite_infill_prob: f32,
    pub sleep_pyrite_crust_max: u32,
    pub sleep_pyrite_crust_min_solid: u32,
    pub sleep_pyrite_crust_enabled: u8,
    pub sleep_pyrite_crust_prob: f32,
    pub sleep_growth_density_min: f32,
    pub sleep_growth_density_max: f32,
    // Collapse — strut_survival shrunk from [f32; 8] to [f32; 6] on 2026-05-26.
    // 6 entries: None=0, Copper=1, Iron=2, Steel=3, Crystal=4, Mithril=5.
    pub sleep_strut_survival: [f32; 6],
    pub sleep_stress_multiplier: f32,
    pub sleep_max_cascade_iterations: u32,
    pub sleep_rubble_fill_ratio: f32,
    pub sleep_min_stress_for_cascade: f32,
    pub sleep_rubble_material_match: u8,
    pub sleep_support_stress_penalty: f32,
    pub sleep_collapse_sub_enabled: u8,
    // ── New 4-phase + Groundwater fields (appended) ──
    // Groundwater (4)
    pub sleep_groundwater_enabled: u8,
    pub sleep_groundwater_strength: f32,
    pub sleep_groundwater_depth_scale: f32,
    pub sleep_groundwater_drip_multiplier: f32,
    // Phase enables (4)
    pub sleep_phase1_enabled: u8,
    pub sleep_phase2_enabled: u8,
    pub sleep_phase3_enabled: u8,
    pub sleep_phase4_enabled: u8,
    // Phase 1: Reaction (3)
    pub sleep_acid_dissolution_prob: f32,
    pub sleep_copper_oxidation_prob: f32,
    pub sleep_basalt_crust_prob: f32,
    // Phase 2: Aureole (4)
    pub sleep_aureole_radius: u32,
    pub sleep_contact_marble_prob: f32,
    pub sleep_water_erosion_prob: f32,
    pub sleep_water_erosion_enabled: u8,
    // Phase 3: Veins (4)
    pub sleep_vein_deposition_prob: f32,
    pub sleep_vein_max_distance: u32,
    pub sleep_vein_max_per_source: u32,
    pub sleep_flowstone_prob: f32,
    // Phase 4: Deep Time (3)
    pub sleep_enrichment_prob: f32,
    pub sleep_vein_thickening_prob: f32,
    pub sleep_stalactite_growth_prob: f32,
    // Collapse (new, separate from legacy collapse fields above)
    pub sleep_new_collapse_enabled: u8,
    pub sleep_new_stress_multiplier: f32,
    pub sleep_new_min_stress_cascade: f32,
    pub sleep_new_rubble_fill_ratio: f32,
    // Groundwater power controls (5)
    pub sleep_gw_erosion_power: f32,
    pub sleep_gw_flowstone_power: f32,
    pub sleep_gw_enrichment_power: f32,
    pub sleep_gw_soft_rock_mult: f32,
    pub sleep_gw_hard_rock_mult: f32,
    // ── Water Table Config (11 fields) ──
    pub water_table_enabled: u8,
    pub water_table_base_y: f64,
    pub water_table_noise_amplitude: f64,
    pub water_table_noise_frequency: f64,
    pub water_table_spring_flow_rate: f32,
    pub water_table_min_porosity: f32,
    pub water_table_drip_noise_frequency: f64,
    pub water_table_drip_noise_threshold: f64,
    pub water_table_drip_level: f32,
    pub water_table_max_springs: u32,
    pub water_table_max_drips: u32,
    // ── Pipe Lava Config (4 fields) ──
    pub pipe_lava_enabled: u8,
    pub pipe_lava_activation_depth: f64,
    pub pipe_lava_max_per_chunk: u32,
    pub pipe_lava_depth_scaling: f64,
    // ── Lava Tube Config (10 fields) ──
    pub lava_tube_enabled: u8,
    pub lava_tube_tubes_per_region: f32,
    pub lava_tube_depth_min: f64,
    pub lava_tube_depth_max: f64,
    pub lava_tube_radius_min: f32,
    pub lava_tube_radius_max: f32,
    pub lava_tube_max_steps: u32,
    pub lava_tube_step_length: f32,
    pub lava_tube_active_depth: f64,
    pub lava_tube_pipe_connection_radius: f32,
    // ── Hydrothermal Config (3 fields) ──
    pub hydrothermal_enabled: u8,
    pub hydrothermal_radius: u32,
    pub hydrothermal_max_per_chunk: u32,
    // ── River Config (9 fields) ──
    pub river_enabled: u8,
    pub river_rivers_per_region: f32,
    pub river_width_min: f32,
    pub river_width_max: f32,
    pub river_height: f32,
    pub river_max_steps: u32,
    pub river_step_length: f32,
    pub river_layer_restriction: u8,
    pub river_downslope_bias: f64,
    // ── Artesian Config (7 fields) ──
    pub artesian_enabled: u8,
    pub artesian_aquifer_y_center: f64,
    pub artesian_aquifer_thickness: f64,
    pub artesian_aquifer_noise_freq: f64,
    pub artesian_aquifer_noise_threshold: f64,
    pub artesian_pressure_noise_freq: f64,
    pub artesian_max_per_chunk: u32,
    // ── Fluid Sources Toggle ──
    pub fluid_sources_enabled: u8,
    // ── Fluid Tuning ──
    pub fluid_solid_corner_threshold: u8,
    // ── Fluid Flow Capacity (DEPRECATED — binary classification always used, kept for ABI) ──
    pub fluid_flow_solid_threshold: u8,
    pub fluid_fractional_capacity: u8,
    // ── Cauldron Inset Tuning ──
    pub formation_cauldron_wall_inset: f32,
    pub formation_cauldron_floor_inset: i32,
    // ── Grace Period ──
    pub fluid_source_grace_ticks: u16,
    // ── Acid Dissolution Cap ──
    pub sleep_acid_max_dissolved_per_source: u32,
    // ── Vein Deposit Spacing ──
    pub sleep_vein_deposit_spacing: u32,
    // ── Lava Solidification ──
    pub sleep_lava_solidification_enabled: u8,
    // ── Aureole Zone Config (10 fields) ──
    pub sleep_zone_enabled: u8,
    pub sleep_heat_multiplier: f32,
    pub sleep_radius_scale: f32,
    pub sleep_water_boost_max: f32,
    pub sleep_water_search_radius_mult: f32,
    pub sleep_large_vein_base_size: u32,
    pub sleep_small_vein_base_size: u32,
    pub sleep_min_lava_zone_size: u32,
    pub sleep_garnet_pocket_size: u32,
    pub sleep_diopside_pocket_size: u32,
    pub sleep_max_aureole_radius: f32,
    // ── New Sleep Fields (Phase A overhaul — ~90 fields) ──
    // Top-level sleep
    pub sleep_accumulation_enabled: u8,
    pub sleep_accumulation_iterations: u32,
    // Groundwater (depth_baseline + 6 porosities)
    pub sleep_gw_depth_baseline: f32,
    pub sleep_gw_porosity_limestone: f32,
    pub sleep_gw_porosity_sandstone: f32,
    pub sleep_gw_porosity_slate: f32,
    pub sleep_gw_porosity_marble: f32,
    pub sleep_gw_porosity_granite: f32,
    pub sleep_gw_porosity_basalt: f32,
    // Phase 1: Reaction (11 missing fields)
    pub sleep_acid_dissolution_radius: u32,
    pub sleep_acid_dissolution_enabled: u8,
    pub sleep_copper_oxidation_enabled: u8,
    pub sleep_basalt_crust_enabled: u8,
    pub sleep_sulfide_acid_enabled: u8,
    pub sleep_sulfide_acid_prob: f32,
    pub sleep_sulfide_acid_radius: u32,
    pub sleep_sulfide_water_amplification: f32,
    pub sleep_limestone_acid_radius_boost: f32,
    pub sleep_gypsum_deposition_prob: f32,
    pub sleep_gypsum_enabled: u8,
    // Phase 2: Aureole (20 missing fields)
    pub sleep_contact_sandstone_to_granite_prob: f32,
    pub sleep_mid_limestone_to_marble_prob: f32,
    pub sleep_mid_sandstone_to_granite_prob: f32,
    pub sleep_outer_limestone_to_marble_prob: f32,
    pub sleep_aureole_metamorphism_enabled: u8,
    pub sleep_coal_maturation_enabled: u8,
    pub sleep_coal_to_graphite_prob: f32,
    pub sleep_coal_to_graphite_mid_prob: f32,
    pub sleep_graphite_to_diamond_prob: f32,
    pub sleep_silicification_enabled: u8,
    pub sleep_silicification_limestone_prob: f32,
    pub sleep_silicification_sandstone_prob: f32,
    pub sleep_silicification_water_radius_mult: u32,
    pub sleep_contact_limestone_to_garnet_prob: f32,
    pub sleep_mid_limestone_to_garnet_prob: f32,
    pub sleep_mid_limestone_to_diopside_prob: f32,
    pub sleep_recrystallization_prob: f32,
    pub sleep_contact_slate_to_hornfels_prob: f32,
    pub sleep_mid_slate_to_hornfels_prob: f32,
    pub sleep_outer_slate_to_hornfels_prob: f32,
    // Phase 3: Veins (29 missing fields)
    pub sleep_vein_enabled: u8,
    pub sleep_hypothermal_height: u32,
    pub sleep_mesothermal_height: u32,
    pub sleep_epithermal_height: u32,
    pub sleep_horizontal_spread: u32,
    pub sleep_veins_per_zone_min: u32,
    // DEPRECATED: replaced by sleep_vein_size_min/max at end of struct (kept for ABI padding)
    pub sleep_vein_climb_height_min: u32,
    pub sleep_vein_climb_height_max: u32,
    pub sleep_vein_wall_width_min: u32,
    pub sleep_vein_wall_width_max: u32,
    pub sleep_vein_rock_depth_min: u32,
    pub sleep_vein_rock_depth_max: u32,
    pub sleep_heat_direction_bias: f32,
    pub sleep_epithermal_rarity: f32,
    pub sleep_vein_crystal_growth_enabled: u8,
    pub sleep_vein_crystal_growth_prob: f32,
    pub sleep_vein_crystal_growth_max_per_chunk: u32,
    pub sleep_vein_calcite_infill_enabled: u8,
    pub sleep_vein_calcite_infill_prob: f32,
    pub sleep_vein_calcite_infill_max_per_chunk: u32,
    pub sleep_vein_flowstone_enabled: u8,
    pub sleep_vein_flowstone_max_per_chunk: u32,
    pub sleep_vein_growth_density_min: f32,
    pub sleep_vein_growth_density_max: f32,
    pub sleep_aperture_scaling_enabled: u8,
    pub sleep_host_rock_ore_enabled: u8,
    pub sleep_slate_pyrite_codeposit_prob: f32,
    pub sleep_slate_quartz_vein_prob: f32,
    pub sleep_wall_rock_alteration_prob: f32,
    // Phase 4: Deep Time (31 missing fields)
    pub sleep_max_enrichment_per_chunk: u32,
    pub sleep_enrichment_search_radius: i32,
    pub sleep_enrichment_enabled: u8,
    pub sleep_enrichment_cluster_min: u32,
    pub sleep_enrichment_cluster_max: u32,
    pub sleep_vein_thickening_enabled: u8,
    pub sleep_vein_thickening_max_per_chunk: u32,
    pub sleep_vein_thickening_water_radius: f32,
    pub sleep_vein_thickening_coat_depth: u32,
    pub sleep_vein_thickening_finger_interval: u32,
    pub sleep_vein_thickening_finger_length_min: u32,
    pub sleep_vein_thickening_finger_length_max: u32,
    pub sleep_vein_thickening_finger_taper: f32,
    pub sleep_mature_formations_enabled: u8,
    pub sleep_column_formation_prob: f32,
    // Nest fossilization (7 fields)
    pub sleep_nest_fossil_enabled: u8,
    pub sleep_nest_fossil_radius: u32,
    pub sleep_nest_fossil_pyrite_prob: f32,
    pub sleep_nest_fossil_opal_prob: f32,
    pub sleep_nest_fossil_buried_required: u8,
    pub sleep_nest_fossil_water_pyrite: u8,
    pub sleep_nest_fossil_water_opal: u8,
    // Corpse fossilization (6 fields)
    pub sleep_corpse_fossil_enabled: u8,
    pub sleep_corpse_fossil_radius: u32,
    pub sleep_corpse_fossil_pyrite_prob: f32,
    pub sleep_corpse_fossil_calcium_prob: f32,
    pub sleep_corpse_fossil_water_required: u8,
    pub sleep_corpse_fossil_min_cycles: u32,
    // Slate aquitard (3 fields)
    pub sleep_slate_aquitard_enabled: u8,
    pub sleep_slate_aquitard_factor: f32,
    pub sleep_slate_aquitard_concentration: f32,
    // ── Vein scaling + spikes + ore global scale ──
    // Min vein height
    pub sleep_min_vein_height: u32,
    // Water volume scaling (4 fields)
    pub sleep_water_volume_radius: u32,
    pub sleep_water_volume_max_cells: u32,
    pub sleep_water_volume_vein_mult: f32,
    pub sleep_water_volume_amount_mult: f32,
    // Lava volume scaling (4 fields)
    pub sleep_lava_volume_radius: u32,
    pub sleep_lava_volume_max_cells: u32,
    pub sleep_lava_volume_vein_mult: f32,
    pub sleep_lava_volume_amount_mult: f32,
    // Spike intrusions (6 fields)
    pub sleep_spike_enabled: u8,
    pub sleep_spike_count_min: u32,
    pub sleep_spike_count_max: u32,
    pub sleep_spike_length_min: u32,
    pub sleep_spike_length_max: u32,
    pub sleep_spike_taper: f32,
    // Ore global scale
    pub ore_global_scale: f32,
    // ── Aureole deposit detail settings ──
    pub sleep_aureole_vein_count: u32,
    pub sleep_aureole_vein_min: u32,
    pub sleep_aureole_vein_max: u32,
    pub sleep_garnet_compact_size: u32,
    pub sleep_diopside_compact_size: u32,
    pub sleep_garnet_pocket_count: u32,
    pub sleep_diopside_pocket_count: u32,
    pub sleep_aureole_vein_spread: f32,
    // Aureole lava volume scaling
    pub sleep_aureole_lava_max_cells: u32,
    pub sleep_aureole_lava_deposit_mult: f32,
    pub sleep_aureole_lava_count_mult: f32,
    // Aureole water boost exposure
    pub sleep_aureole_water_search_radius: u32,
    pub sleep_aureole_water_max_cells: u32,
    pub sleep_aureole_water_deposit_mult: f32,
    // Aureole vein shape
    pub sleep_aureole_wall_climbing: u8,
    pub sleep_aureole_weight_up: f32,
    pub sleep_aureole_weight_depth: f32,
    pub sleep_aureole_weight_lateral: f32,
    pub sleep_aureole_surface_ratio: f32,
    // Hydrothermal vein shape
    pub sleep_vein_spread: f32,
    pub sleep_vein_size_min: u32,
    pub sleep_vein_size_max: u32,
    pub sleep_vein_weight_up: f32,
    pub sleep_vein_weight_depth: f32,
    pub sleep_vein_weight_lateral: f32,
    pub sleep_vein_surface_ratio: f32,
    // Water proximity bias
    pub sleep_water_proximity_bias: f32,
    // Min connectivity
    pub sleep_vein_min_connectivity: u32,
    pub sleep_aureole_min_connectivity: u32,
    // Weight down
    pub sleep_vein_weight_down: f32,
    pub sleep_aureole_weight_down: f32,
    // Aureole per-N-cells scaling
    pub sleep_aureole_veins_per_n_cells: f32,
    pub sleep_aureole_garnet_per_n_cells: f32,
    pub sleep_aureole_diopside_per_n_cells: f32,
    pub sleep_aureole_cells_per_extra: u32,

    // ── Cavern Zone Config (45 fields) ──
    pub zone_enabled: u8,
    // Per-type spawn probabilities
    pub zone_cathedral_chance: f32,
    pub zone_lake_chance: f32,
    pub zone_canyon_chance: f32,
    pub zone_lava_gallery_chance: f32,
    pub zone_bioluminescent_chance: f32,
    pub zone_terraces_chance: f32,
    pub zone_frozen_chance: f32,
    // Per-type minimum air thresholds
    pub zone_cathedral_min_air: u32,
    pub zone_lake_min_air: u32,
    pub zone_canyon_min_air: u32,
    pub zone_lava_gallery_min_air: u32,
    pub zone_bioluminescent_min_air: u32,
    pub zone_terraces_min_air: u32,
    pub zone_frozen_min_air: u32,
    // Cathedral
    pub zone_cathedral_dome_scale: f32,
    pub zone_cathedral_boulder_count_min: u32,
    pub zone_cathedral_boulder_count_max: u32,
    pub zone_cathedral_mega_stalagmite_chance: f32,
    pub zone_cathedral_flowstone_coverage: f32,
    // Lake
    pub zone_lake_depth: u32,
    pub zone_lake_beach_width: f32,
    pub zone_lake_island_min_radius: f32,
    // Canyon
    pub zone_canyon_width_min: f32,
    pub zone_canyon_width_max: f32,
    pub zone_canyon_height_min: f32,
    pub zone_canyon_height_max: f32,
    pub zone_canyon_bridge_chance: f32,
    // Lava Gallery
    pub zone_lava_gallery_bench_spacing: f32,
    pub zone_lava_gallery_lavacicle_chance: f32,
    // Bioluminescent
    pub zone_bio_anchor_density: f32,
    pub zone_bio_max_anchors: u32,
    // Terraces
    pub zone_terrace_tiers_min: u32,
    pub zone_terrace_tiers_max: u32,
    pub zone_terrace_step_height: f32,
    pub zone_terrace_rim_height: f32,
    pub zone_terrace_basin_depth: u32,
    // Frozen
    pub zone_frozen_floor_depth: u32,
    pub zone_frozen_waterfall_count: u32,
    pub zone_frozen_ice_stalactite_chance: f32,
    pub zone_frozen_mega_chance: f32,
    // ── Blank-Canvas Mode ──
    // 1 = skip all decoration phases (caverns, worms, pools, formations, zones)
    // and emit uniform host rock for hand-authoring with creative brushes.
    pub blank_canvas: u8,
    // ── Basalt aureole (Amphibolite) deposit settings ──
    // Appended at the end to avoid shifting offsets of pre-existing fields.
    pub sleep_amphibolite_pyrite_pocket_count: u32,
    pub sleep_amphibolite_garnet_pocket_count: u32,
    pub sleep_amphibolite_pyrite_compact_size: u32,
    pub sleep_aureole_amphibolite_pyrite_per_n_cells: f32,
    pub sleep_aureole_amphibolite_garnet_per_n_cells: f32,
    // ── Hydrothermal water-boost v2 (Phase 1 BFS + connected supply network) ──
    // Appended at end (FFI sync rule: never insert into middle).
    pub sleep_aureole_water_phase1_weight: f32,
    pub sleep_aureole_water_phase2_weight: f32,
    pub sleep_aureole_water_network_max_hops: u32,
    pub sleep_aureole_water_to_lava_ratio: f32,
    pub sleep_aureole_water_phase1_max_floor: u32,
    pub sleep_aureole_water_count_mult: f32,
    // ── Mushroom Decoration (16 fields) ──
    // Appended at end (FFI sync rule: never insert into middle).
    pub mushroom_enabled: u8,
    pub _mushroom_pad: [u8; 3],
    pub mushroom_global_density: f32,
    pub mushroom_cluster_frequency: f64,
    pub mushroom_cluster_threshold: f32,
    pub mushroom_min_spacing_voxels: f32,
    pub mushroom_ghost_tower_routing_share: f32,
    // Per-kind: enabled, spawn_chance, scale_min, scale_max (4 each × 4 kinds = 16 — but
    // packed into the same layout as KindConfig to keep parity with engine.rs mapping).
    pub mushroom_turkey_tail_enabled: u8,
    pub _mushroom_pad_tt: [u8; 3],
    pub mushroom_turkey_tail_spawn_chance: f32,
    pub mushroom_turkey_tail_scale_min: f32,
    pub mushroom_turkey_tail_scale_max: f32,
    pub mushroom_foxfire_enabled: u8,
    pub _mushroom_pad_fx: [u8; 3],
    pub mushroom_foxfire_spawn_chance: f32,
    pub mushroom_foxfire_scale_min: f32,
    pub mushroom_foxfire_scale_max: f32,
    pub mushroom_green_pepe_enabled: u8,
    pub _mushroom_pad_gp: [u8; 3],
    pub mushroom_green_pepe_spawn_chance: f32,
    pub mushroom_green_pepe_scale_min: f32,
    pub mushroom_green_pepe_scale_max: f32,
    pub mushroom_ghost_tower_enabled: u8,
    pub _mushroom_pad_gt: [u8; 3],
    pub mushroom_ghost_tower_spawn_chance: f32,
    pub mushroom_ghost_tower_scale_min: f32,
    pub mushroom_ghost_tower_scale_max: f32,
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

/// Creative-mode brush request. Sphere ops (paint/carve/fill) use this struct.
/// World coords/radius are in UE space; the worker converts them.
///
/// `mode`: 0 = paint material on solid voxels (no shape change)
///         1 = carve sphere (set solid → air)
///         2 = fill sphere (set air → solid with `material`, also overwrites
///             material on already-solid voxels in range)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushSphereRequest {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub radius: f32,
    pub mode: u8,
    pub material: u8,   // ignored for carve mode
    pub _pad: [u8; 2],
}

/// Tunnel-along-polyline brush. `points` are UE world coords; the worker
/// converts each point. If `material == 255` the tunnel carves; otherwise
/// it fills with that material.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushTunnelRequest {
    pub points: *const FfiVec3,
    pub point_count: u32,
    pub radius: f32,
    pub material: u8,
    pub _pad: [u8; 3],
}

/// Place a single hand-authored formation at a UE world position.
/// `formation_type`: 0=Stalactite, 1=Stalagmite, 2=Column, 3=Drapery,
///                   4=Flowstone, 5=Shield, 6=RimstoneDam
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushFormationRequest {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub formation_type: u8,
    pub material: u8,
    pub _pad: [u8; 2],
    pub height: f32,    // UE units
    pub radius: f32,    // UE units
}

/// Place a single mushroom instance at a UE world position. Does NOT touch
/// density — the brush picks the nearest solid voxel within `search_radius`
/// (UE units) as the anchor, infers the surface face from its air-neighbor
/// pattern, and inserts a `MushroomPlacement` into the chunk's store.
/// `kind` is the `MushroomKind` enum value (0=TurkeyTail, 1=Foxfire,
/// 2=GreenPepe, 3=GhostTower). `scale` is the instance scale; pass 0.0 to
/// use the kind's configured `scale_min..scale_max` random range.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushPlaceMushroomRequest {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub kind: u8,
    pub _pad: [u8; 3],
    pub search_radius: f32,  // UE units — radius to scan for an anchor voxel
    pub scale: f32,          // 0.0 = randomize per kind config
    pub yaw_radians: f32,    // 0.0 = randomize
}

/// Sphere-area mushroom brush — scatters multiple mushrooms within a radius.
/// `radius` is in UE units; `density` is 0..1 Bernoulli per viable surface
/// voxel. `kind` constrains placement to that species' preferred face
/// (TurkeyTail→walls, Foxfire→ceilings, GreenPepe/GhostTower→floors). `seed`
/// randomizes the pattern.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushPlaceMushroomSphereRequest {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub radius: f32,
    pub density: f32,
    pub clustering: f32, // 0..1 — local noise gate strength
    pub kind: u8,
    pub op: u8,          // 0=place, 1=erase (kind acts as filter; 255=any)
    pub _pad: [u8; 2],
    pub seed: u64,
}

/// Formation Stamp brush — runs the full worldgen formation pipeline
/// (random mix of stalactites/columns/drapery/etc. picked per the live
/// FormationConfig) on chunks overlapping a sphere, anchored within it.
/// `seed` randomizes the pick so re-stamping gives a different vibe.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushFormationStampRequest {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub radius: f32,    // UE units
    pub seed: u32,
}

/// Cavern Stamp brush — chunk-snapped cave generator. Runs worldgen worm
/// carving (additively — existing edits in the chunks survive) on a NxMxK
/// chunk region, optionally with lava tubes/rivers + pools/formations.
/// `chunk_x/y/z` is the lo-corner chunk in Rust chunk coords.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushCavernStampRequest {
    pub chunk_x: i32,
    pub chunk_y: i32,
    pub chunk_z: i32,
    pub extent_x: u8,
    pub extent_y: u8,
    pub extent_z: u8,
    pub decorate: u8,  // 0/1 — also run pools + formations
    pub fluids: u8,    // 0/1 — also run lava tubes + rivers
    pub seed: u32,
}

/// Axis-aligned-or-yawed box brush. `op`: 0=paint, 1=carve, 2=fill.
/// Half-extents in UE units. `yaw_deg`: rotation around UE vertical (Z) axis,
/// in degrees. 0 = AABB (no rotation).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushBoxRequest {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub half_x: f32,
    pub half_y: f32,
    pub half_z: f32,
    pub op: u8,
    pub material: u8,
    pub _pad: [u8; 2],
    pub yaw_deg: f32,
}

/// Y-axis-aligned cylinder brush. `op`: 0=paint, 1=carve, 2=fill.
/// `radius` and `height` in UE units; `height` is the full cylinder height.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushCylinderRequest {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub radius: f32,
    pub height: f32,
    pub op: u8,
    pub material: u8,
    pub _pad: [u8; 2],
}

/// Creative "OrePaint" brush — drops wall-exposed ore deposits inside the sphere
/// with even (Poisson-disk) spacing, weighted ore-type picks, and optional
/// inward "deep channel" tubes for each cluster. World coords are UE space.
/// Per-ore weights match `OreWeights` in `brushes.rs`:
/// `[iron, copper, malachite, tin, gold, diamond, kimberlite, sulfide,
///   quartz, pyrite, amethyst, crystal, coal]`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushOrePaintRequest {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub radius: f32,         // UE units — overall brush sphere
    pub cluster_size: f32,   // voxels — radius of each ore knob
    pub min_spacing: f32,    // voxels — minimum distance between cluster anchors
    pub channel_prob: f32,   // 0..1 — per-anchor chance to extend a deep channel
    pub channel_length: f32, // voxels — typical tube length into rock
    pub channel_radius: f32, // voxels — tube radius
    pub density: f32,        // 0..1 — fraction of wall candidates to keep as anchors
    pub seed: u32,
    pub weights: [u8; 13],   // per-ore frequency weights (see OreWeights ordering)
    pub _pad: [u8; 3],
}

/// Creative-mode "PaintStress" brush — additively writes into the per-voxel
/// painted-stress overlay (`StressField::painted_stress`) inside a sphere.
/// Does not change density/material, so no remesh is emitted; the new stress
/// is folded into `effective()` reads and drives extra collapses during sleep.
///
/// `op`:    0 = add, 1 = subtract, 2 = clear (zero the painted overlay inside the sphere)
/// `falloff`: 0 = constant, 1 = linear, 2 = smoothstep
/// `amount`: peak per-stroke additive (typical 0.2–0.8)
/// `cap`:    per-cell accumulation ceiling (typical 2.0)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushPaintStressRequest {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub radius: f32,    // UE units
    pub amount: f32,
    pub cap: f32,
    pub op: u8,         // 0=add, 1=sub, 2=clear
    pub falloff: u8,    // 0=constant, 1=linear, 2=smoothstep
    pub _pad: [u8; 2],
}

/// Smooth brush — Laplacian average of density in a sphere. Material preserved.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushSmoothRequest {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub radius: f32,
    pub iterations: u32,
    pub strength: f32,  // 0..1
}

/// Noise brush — perturb density by hash-based 3D noise within a sphere.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushNoiseRequest {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub radius: f32,
    pub frequency: f32,
    pub strength: f32,
    pub seed: u32,
}

/// Sphere fluid brush — places (or clears) fluid within a sphere.
/// `op`: 0=fill (level=1.0), 1=clear (level=0.0), 2=pool-dig (carve solid + fill bottom half non-source), 3=carve+full fill
/// `fluid_type`: 1=Water, 2=Lava, 3-9=specialized water sub-types
/// `is_source`: nonzero = treat placed fluid as an infinite source (spring); 0 = drains naturally
/// `max_flow_dist`: bounded-flow limit when placed as a source. 0 = unlimited (legacy
/// behavior). >0 = source's children stop propagating beyond this hop count, with
/// linear taper across the last `chunk::TAPER_HOPS` cells.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushFluidSphereRequest {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub radius: f32,
    pub fluid_type: u8,
    pub is_source: u8,
    pub op: u8,
    pub max_flow_dist: u8,
}

/// Box fluid brush — fills (or clears) fluid within an axis-aligned box.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushFluidBoxRequest {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub half_x: f32,
    pub half_y: f32,
    pub half_z: f32,
    pub fluid_type: u8,
    pub is_source: u8,
    pub op: u8,            // 0=fill, 1=clear, 2=carve+fill
    pub max_flow_dist: u8, // 0 = unlimited
}

/// Capsule-chain (river/spline) fluid brush. Points are UE world coords.
/// Fills air voxels along the path; if `op == 2`, also carves the channel first.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushFluidRiverRequest {
    pub points: *const FfiVec3,
    pub point_count: u32,
    pub radius: f32,
    pub fluid_type: u8,
    pub is_source: u8,
    pub op: u8,            // 0=fill (only), 2=carve channel + fill
    pub max_flow_dist: u8, // 0 = unlimited
}

// (removed FfiBrushFluidStreamRequest — replaced by bounded sources via max_flow_dist on FluidCell)

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

// ── Internal (non-FFI) types ──

/// Converted mesh data in UE coordinate space, ready to be handed out via FFI.
pub struct ConvertedMesh {
    pub positions: Vec<FfiVec3>,
    pub normals: Vec<FfiVec3>,
    pub material_ids: Vec<u8>,
    pub indices: Vec<u32>,
    pub submeshes: Vec<FfiSubmesh>,
}

/// Converted fluid mesh data in UE coordinate space.
pub struct ConvertedFluidMesh {
    pub positions: Vec<FfiVec3>,
    pub normals: Vec<FfiVec3>,
    pub fluid_types: Vec<u8>,
    pub indices: Vec<u32>,
    pub uvs: Vec<[f32; 2]>,
    pub flow_directions: Vec<FfiVec3>,
}

/// Messages sent to worker threads.
pub enum WorkerRequest {
    /// Apply a lava↔water quench plan from the fluid sim.
    /// Writes Obsidian voxels at the contact rim and Scoria voxels in the
    /// inward halo, then remeshes the affected chunks. Sources are NOT in
    /// this list — they stay alive and grow via the fluid sim's pillow state.
    ApplyLavaQuench {
        obsidian: Vec<((i32, i32, i32), usize, usize, usize)>,
        scoria: Vec<((i32, i32, i32), usize, usize, usize)>,
        drained_water: Vec<((i32, i32, i32), usize, usize, usize)>,
    },
    Generate {
        chunk: (i32, i32, i32),
        generation: u64,
    },
    PriorityGenerate {
        chunk: (i32, i32, i32),
        generation: u64,
    },
    Mine {
        request: FfiMineRequest,
    },
    Flatten {
        base_x: i32,
        base_y: i32,
        base_z: i32,
        host_material: u8,
    },
    FlattenBatch {
        tiles: Vec<(glam::IVec3, voxel_core::material::Material)>,
    },
    BuildingFlatten {
        base_x: i32,
        base_y: i32,
        base_z: i32,
        // Exact sub-voxel Y from UE (in voxel units). The integer base_y is
        // floor(base_y_float). Used by the SDF flatten to position the iso
        // surface within sub-voxel precision so buildings don't float/sink.
        base_y_float: f32,
        host_material: u8,
        footprint_voxels: i32,
        clearance_voxels: i32,
    },
    BuildingFlattenBatch {
        // (base_x, base_y, base_z, base_y_float, host_material, footprint, clearance)
        buildings: Vec<(i32, i32, i32, f32, u8, i32, i32)>,
    },
    Unload {
        chunk: (i32, i32, i32),
    },
    PlaceSupport {
        world_x: i32,
        world_y: i32,
        world_z: i32,
        support_type: u8,
    },
    RemoveSupport {
        world_x: i32,
        world_y: i32,
        world_z: i32,
    },
    Sleep {
        player_chunk: (i32, i32, i32),
        sleep_count: u32,
        sleep_config: voxel_sleep::SleepConfig,
    },
    AureoleOnly {
        player_chunk: (i32, i32, i32),
        sleep_config: voxel_sleep::SleepConfig,
    },
    WorldScan,
    WorldScanWithConfig {
        config: voxel_core::world_scan::ScanConfig,
    },
    ForceSpawnPool {
        world_x: f32,
        world_y: f32,
        world_z: f32,
        fluid_type: u8,
    },
    MineAndFillFluid {
        world_x: f32,
        world_y: f32,
        world_z: f32,
        radius: f32,
        fluid_type: u8,
        world_scale: f32,
    },
    MorphStep {
        chunks: Vec<(i32, i32, i32)>,
        step: u32,
        total_steps: u32,
    },
    /// Sphere brush (paint/carve/fill) at a UE world position.
    BrushSphere {
        request: FfiBrushSphereRequest,
    },
    /// Tunnel-along-polyline brush.
    BrushTunnel {
        /// Points in Rust world coords (already converted from UE).
        points: Vec<glam::Vec3>,
        radius: f32,           // Rust world units
        material: Option<u8>,  // None = carve, Some(u8) = fill with material
    },
    /// Place a single formation primitive at a UE world position.
    BrushFormation {
        center_rust: glam::Vec3,
        formation_type: u8,
        material: u8,
        height: f32,   // Rust world units
        radius: f32,   // Rust world units
    },
    /// Formation Stamp brush — runs the full worldgen formation pipeline,
    /// spatially clipped to a sphere, with per-call randomized seed.
    BrushFormationStamp {
        center_rust: glam::Vec3,
        radius: f32,   // Rust world units
        seed: u32,
    },
    /// Cavern Stamp brush — chunk-snapped cave generator over NxMxK chunks.
    BrushCavernStamp {
        chunk_origin: (i32, i32, i32),
        extent: (u8, u8, u8),
        decorate: bool,
        fluids: bool,
        seed: u32,
    },
    /// Diagnostic action: re-sync a single chunk's boundaries with all 6
    /// face neighbors and re-mesh anything modified. Used by the UE
    /// "Force Resync" button to repair seam mismatches without a full
    /// quit+reload — handy when the in-memory state has drifted from
    /// neighbors via mid-session brush/snapshot/regen paths.
    ForceChunkResync { chunk: (i32, i32, i32) },
    /// Axis-aligned-or-yawed box brush.
    BrushBox {
        center_rust: glam::Vec3,
        half_ext_rust: glam::Vec3,
        yaw_rad: f32,
        op: u8,
        material: u8,
    },
    /// Y-axis-aligned cylinder brush.
    BrushCylinder {
        center_rust: glam::Vec3,
        radius: f32,    // Rust world units
        height: f32,    // Rust world units
        op: u8,
        material: u8,
    },
    /// Smooth brush.
    BrushSmooth {
        center_rust: glam::Vec3,
        radius: f32,
        iterations: u32,
        strength: f32,
    },
    /// Noise brush.
    BrushNoise {
        center_rust: glam::Vec3,
        radius: f32,
        frequency: f32,
        strength: f32,
        seed: u32,
    },
    /// Creative "OrePaint" brush — wall-exposed ore deposits + optional channels.
    BrushOrePaint {
        center_rust: glam::Vec3,
        radius: f32,           // Rust world units
        cluster_size: f32,     // voxels
        min_spacing: f32,      // voxels
        channel_prob: f32,     // 0..1
        channel_length: f32,   // voxels
        channel_radius: f32,   // voxels
        density: f32,          // 0..1
        seed: u32,
        weights: crate::brushes::OreWeights,
    },
    /// Creative "PaintStress" brush — additive sphere over the painted-stress overlay.
    BrushPaintStress {
        center_rust: glam::Vec3,
        radius: f32,    // Rust world units
        amount: f32,
        cap: f32,
        op: u8,
        falloff: u8,
    },
    /// Creative "Clear all painted stress" — wipes the painted overlay in
    /// every loaded chunk's StressField. No spatial argument; this is the
    /// "nuke everything" eraser for when the per-sphere Clear op is too
    /// fiddly. Atelier exposes a single button.
    BrushClearAllPaintedStress,
    /// Undo the most recent brush stroke (any creative brush).
    BrushUndo,
    /// Sphere fluid brush.
    BrushFluidSphere {
        center_rust: glam::Vec3,
        radius: f32,
        fluid_type: u8,
        is_source: bool,
        op: u8,         // 0=fill, 1=clear, 2=pool-dig, 3=carve+full fill
        max_flow_dist: u8,
    },
    /// Box fluid brush.
    BrushFluidBox {
        center_rust: glam::Vec3,
        half_ext_rust: glam::Vec3,
        fluid_type: u8,
        is_source: bool,
        op: u8,         // 0=fill, 1=clear, 2=carve+fill
        max_flow_dist: u8,
    },
    /// Place a single mushroom at the cursor position (creative brush).
    BrushPlaceMushroom {
        center_rust: glam::Vec3,
        kind: u8,
        search_radius: f32, // Rust voxel units
        scale: f32,         // 0 = randomize
        yaw: f32,           // 0 = randomize
    },
    /// Sphere-area mushroom brush — scatters N mushrooms of one kind within
    /// a radius via Bernoulli sampling against surface voxels of the kind's
    /// preferred face. `clustering` (0..1) raises a local Simplex-noise gate
    /// so high values produce tight family pockets instead of flat scatter.
    BrushPlaceMushroomSphere {
        center_rust: glam::Vec3,
        radius: f32,    // Rust voxel units
        density: f32,   // 0..1
        clustering: f32, // 0..1
        kind: u8,
        seed: u64,
    },
    /// Sphere-area mushroom eraser. `kind_filter == 255` erases any kind;
    /// otherwise only matching kind is removed.
    BrushEraseMushroomSphere {
        center_rust: glam::Vec3,
        radius: f32,    // Rust voxel units
        kind_filter: u8,
    },
    /// River (capsule chain) fluid brush.
    BrushFluidRiver {
        points: Vec<glam::Vec3>,
        radius: f32,
        fluid_type: u8,
        is_source: bool,
        op: u8,         // 0=fill, 2=carve channel + fill
        max_flow_dist: u8,
    },
    // (removed BrushFluidStream — replaced by bounded sources via max_flow_dist on FluidCell)
    /// Pathfinding request — runs A* against the live ChunkStore on a path
    /// worker thread, sends back `WorkerResult::PathComputed`. The internal
    /// request is pre-converted to Rust voxel coordinates by the FFI entry.
    ComputePath {
        request: crate::pathing::PathRequestInternal,
    },
}

/// Results sent back from worker threads.
pub enum WorkerResult {
    ChunkMesh {
        chunk: (i32, i32, i32),
        mesh: ConvertedMesh,
        generation: u64,
        crystal_data: Vec<FfiCrystalPlacement>,
        mushroom_data: Vec<FfiMushroomInstance>,
        zone_descriptors: Vec<FfiZoneDescriptor>,
    },
    Error {
        chunk: (i32, i32, i32),
        generation: u64,
    },
    MinedMaterials {
        mined: FfiMinedMaterials,
    },
    /// All mine mesh updates in one atomic result — prevents pop-in
    MineBatchMesh {
        meshes: Vec<((i32, i32, i32), ConvertedMesh, Vec<FfiCrystalPlacement>, Vec<FfiMushroomInstance>)>,
    },
    FluidMesh {
        chunk: (i32, i32, i32),
        mesh: ConvertedFluidMesh,
    },
    SolidifyRequest {
        positions: Vec<((i32, i32, i32), usize, usize, usize)>,
    },
    /// Live lava↔water quench plan from the fluid sim — apply Obsidian +
    /// Scoria voxel writes, drain water cells, and remesh affected chunks.
    LavaQuench {
        obsidian: Vec<((i32, i32, i32), usize, usize, usize)>,
        scoria: Vec<((i32, i32, i32), usize, usize, usize)>,
        drained_water: Vec<((i32, i32, i32), usize, usize, usize)>,
    },
    CollapseResult {
        events: Vec<FfiCollapseEvent>,
        meshes: Vec<((i32, i32, i32), ConvertedMesh)>,
    },
    SupportResult {
        success: bool,
        meshes: Vec<((i32, i32, i32), ConvertedMesh)>,
    },
    SleepComplete {
        chunks_changed: u32,
        voxels_metamorphosed: u32,
        minerals_grown: u32,
        supports_degraded: u32,
        collapses_triggered: u32,
        acid_dissolved: u32,
        veins_deposited: u32,
        voxels_enriched: u32,
        formations_grown: u32,
        sulfide_dissolved: u32,
        coal_matured: u32,
        diamonds_formed: u32,
        voxels_silicified: u32,
        nests_fossilized: u32,
        channels_eroded: u32,
        corpses_fossilized: u32,
        lava_solidified: u32,
        profile_report: String,
        aureole_glimpse_pos: Option<(i32, i32, i32)>,
        aureole_showcase_block: Option<Vec<(i32, i32, i32)>>,
        manifest_json: String,
        lava_cells: Vec<(i32, i32, i32)>,
    },
    MorphMeshes {
        step: u32,
        total_steps: u32,
        meshes: Vec<ConvertedMesh>,
    },
    ScanComplete {
        json_report: String,
    },
    ForceSpawnPoolComplete {
        json_report: String,
    },
    /// V2 stress warnings for live feedback (dust/creak/shake positions).
    StressWarnings {
        warnings: Vec<FfiStressWarning>,
    },
    /// V2 collapse result with coherent slab data for animated falling.
    CollapseSlabResult {
        events: Vec<FfiCollapseEventV2>,
        meshes: Vec<((i32, i32, i32), ConvertedMesh)>,
    },
    /// One falling-slab visual: real DC mesh + fall metadata. Cinematic Act 3.
    /// Each fragment of a multi-fragment collapse becomes one of these.
    SlabFall {
        mesh: ConvertedMesh,
        fall_data: FfiSlabFallData,
    },
    /// Localized pre-collapse warning. Cinematic Acts 1-2.
    CollapseWarning {
        center_ue: (f32, f32, f32),
        bounds_extent_ue: (f32, f32, f32),
        severity: u8, // 1=dust, 2=creak, 3=shake, 4=imminent
        eta_ms: u32,
        volume: u32,
    },
    /// One tier of the 4-tier pile-buildup preview. UE buffers by spawn loc,
    /// builds a debris actor when it has 4 tiers, animates reveal.
    PilePreviewTier {
        mesh: ConvertedMesh,
        fall_data: FfiSlabFallData,
    },
    /// A* result for a path request — stashed into the engine's
    /// `path_results` map keyed by `request_id` and never touched by mesh
    /// consumers (intercepted in `poll_result`).
    PathComputed {
        request_id: u32,
        status: u8,
        nodes_ue: Vec<crate::pathing::PathNodeUE>,
    },
    /// One or more struts hit 0 HP. UE should look up the corresponding
    /// `AVoxelSupportActor` via `PlacedSupports` (keyed by voxel position),
    /// play the breaking VFX, and refresh the crack overlay in a sphere
    /// around the strut so the newly-unsupported walls flash red.
    StrutsBroken {
        struts: Vec<FfiStrutBroken>,
    },
}

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

/// FFI result for world scan. JSON report is passed as a heap-allocated string.
#[repr(C)]
pub struct FfiWorldScanResult {
    pub success: u32,
    pub json_report: *mut std::ffi::c_char,
    pub json_length: u32,
    pub chunks_scanned: u32,
    pub total_issues: u32,
    pub total_errors: u32,
    pub total_warnings: u32,
}

/// FFI-safe scan configuration. Uses u32 for booleans (C ABI compatibility).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiScanConfig {
    // Enable flags (u32: 0=false, nonzero=true)
    pub enable_density_seam: u32,
    pub enable_mesh_topology: u32,
    pub enable_seam_completeness: u32,
    pub enable_navigability: u32,
    pub enable_worm_truncation: u32,
    pub enable_thin_walls: u32,
    pub enable_winding_consistency: u32,
    pub enable_degenerate_triangles: u32,
    pub enable_worm_carve_verify: u32,
    pub enable_self_intersection: u32,
    pub enable_seam_mesh_quality: u32,
    // Accuracy params
    pub density_subsample_count: u32,
    pub raymarch_rays_per_chunk: u32,
    pub raymarch_step_size: f32,
    pub max_vertex_zero_crossing_dist: f32,
    pub min_passage_width: f32,
    pub min_triangle_area: f32,
    pub max_edge_length: f32,
    pub thin_wall_max_thickness: u32,
    pub self_intersection_tri_limit: u32,
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
