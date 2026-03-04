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
    pub counts: [u32; 20],
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
    // ── Crystal Config (145 fields: 1 master + 12 ores × 12 fields) ──
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
    pub material_hardness: [f32; 20],
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
    pub phase: u8,            // 0=metamorphism, 1=minerals, 2=collapse, 3=done
    pub progress_pct: f32,    // 0.0 - 1.0
    pub chunks_processed: u32,
    pub chunks_total: u32,
    pub glimpse_chunk: FfiChunkCoord,  // Chunk where interesting transform happened
    pub glimpse_type: u8,     // 0=none, 1=metamorphism, 2=mineral, 3=collapse
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
    pub dirty_chunks: *mut FfiChunkCoord,
    pub dirty_chunk_count: u32,
    pub collapse_events: *mut FfiCollapseEvent,
    pub collapse_event_count: u32,
    pub profile_report: *mut std::ffi::c_char,
    pub profile_report_length: u32,
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
    WorldScan,
    WorldScanWithConfig {
        config: voxel_core::world_scan::ScanConfig,
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
        profile_report: String,
    },
    ScanComplete {
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
