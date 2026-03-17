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
    pub counts: [u32; 27],
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
}

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
#[repr(C)]
pub struct FfiCrystalData {
    pub placements: *mut FfiCrystalPlacement,
    pub count: u32,
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
    // Collapse
    pub sleep_strut_survival: [f32; 8],
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

#[repr(C)]
#[derive(Debug, Clone)]
pub struct FfiStressData {
    pub stress_values: *mut f32,
    pub count: u32,
    pub valid: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiCollapseEvent {
    pub center_x: f32,
    pub center_y: f32,
    pub center_z: f32,
    pub volume: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiStressConfig {
    pub material_hardness: [f32; 27],
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
    pub support_hardness: [f32; 8],
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
        host_material: u8,
        footprint_voxels: i32,
        clearance_voxels: i32,
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
        manifest_json: String,
        step: u32,
        total_steps: u32,
    },
}

/// Results sent back from worker threads.
pub enum WorkerResult {
    ChunkMesh {
        chunk: (i32, i32, i32),
        mesh: ConvertedMesh,
        generation: u64,
        crystal_data: Vec<FfiCrystalPlacement>,
    },
    Error {
        chunk: (i32, i32, i32),
        generation: u64,
    },
    MinedMaterials {
        mined: FfiMinedMaterials,
    },
    FluidMesh {
        chunk: (i32, i32, i32),
        mesh: ConvertedFluidMesh,
    },
    SolidifyRequest {
        positions: Vec<((i32, i32, i32), usize, usize, usize)>,
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
