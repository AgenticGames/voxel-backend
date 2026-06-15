
use voxel_fluid::FluidConfig;
use voxel_fluid::FluidEvent;
use voxel_core::world_scan::ScanConfig;
use voxel_gen::config::{
    BandedIronConfig, CrystalConfig, FormationConfig, GenerationConfig, GeodeConfig, HostRockConfig,
    KimberlitePipeConfig, MineConfig, NoiseConfig, OreConfig, OreCrystalConfig, OreVeinParams,
    PoolConfig, StressConfig, SulfideBlobConfig, WormConfig,
};

use crate::types::*;

use super::VoxelEngine;

impl VoxelEngine {
    /// Update the stress configuration.
    pub fn update_stress_config(&self, new_config: StressConfig) {
        if let Ok(mut cfg) = self.stress_config.write() {
            *cfg = new_config;
        }
    }

    /// Update the sleep configuration.
    pub fn update_sleep_config(&self, new_config: voxel_sleep::SleepConfig) {
        if let Ok(mut cfg) = self.sleep_config.write() {
            *cfg = new_config;
        }
    }

    /// Hot-reload configuration (affects future generation requests).
    pub fn update_config(&self, ffi_config: &FfiEngineConfig) {
        let new_config = ffi_config_to_generation(ffi_config);
        if let Ok(mut cfg) = self.config.write() {
            *cfg = new_config;
        }
    }

    /// Hot-reload fluid configuration at runtime.
    pub fn update_fluid_config(&self, source_grace_ticks: u16) {
        let _ = self.fluid_event_tx.try_send(FluidEvent::UpdateFluidConfig {
            source_grace_ticks,
        });
    }
}

/// Convert FFI config struct to internal GenerationConfig.
/// The main function is a thin orchestrator; each config section is built by a dedicated helper.
pub fn ffi_config_to_generation(c: &FfiEngineConfig) -> GenerationConfig {
    GenerationConfig {
        seed: c.seed,
        chunk_size: c.chunk_size as usize,
        noise: ffi_to_noise_config(c),
        worm: ffi_to_worm_config(c),
        ore: ffi_to_ore_config(c),
        formations: ffi_to_formation_config(c),
        pools: ffi_to_pool_config(c),
        mine: ffi_to_mine_config(c),
        crystals: ffi_to_crystal_config(c),
        zones: ffi_to_zone_config(c),
        water_table: ffi_to_water_table_config(c),
        pipe_lava: ffi_to_pipe_lava_config(c),
        lava_tubes: ffi_to_lava_tube_config(c),
        hydrothermal: ffi_to_hydrothermal_config(c),
        rivers: ffi_to_river_config(c),
        artesian: ffi_to_artesian_config(c),
        mushrooms: ffi_to_mushroom_config(c),
        octree_max_depth: 4,
        region_size: if c.region_size == 0 { 3 } else { c.region_size as i32 },
        bounds_size: c.bounds_size,
        mesh_smooth_iterations: c.mesh_smooth_iterations,
        mesh_smooth_strength: if c.mesh_smooth_strength > 0.0 { c.mesh_smooth_strength } else { 0.3 },
        mesh_boundary_smooth: if c.mesh_boundary_smooth > 0.0 { c.mesh_boundary_smooth } else { 0.3 },
        mesh_recalc_normals: c.mesh_recalc_normals,
        ore_detail_multiplier: if c.ore_detail_multiplier > 0 { c.ore_detail_multiplier.min(4) } else { 1 },
        ore_protrusion: c.ore_protrusion.max(0.0).min(0.5),
        fluid_sources_enabled: c.fluid_sources_enabled != 0,
        blank_canvas: c.blank_canvas != 0,
    }
}

// ── ffi_config_to_generation sub-functions ────────────────────────────────────

fn ffi_to_noise_config(c: &FfiEngineConfig) -> NoiseConfig {
    NoiseConfig {
        cavern_frequency: c.cavern_frequency,
        cavern_threshold: c.cavern_threshold,
        detail_octaves: c.detail_octaves,
        detail_persistence: c.detail_persistence,
        warp_amplitude: c.warp_amplitude,
    }
}

fn ffi_to_worm_config(c: &FfiEngineConfig) -> WormConfig {
    WormConfig {
        worms_per_region: c.worms_per_region,
        radius_min: c.worm_radius_min,
        radius_max: c.worm_radius_max,
        step_length: c.worm_step_length,
        max_steps: c.worm_max_steps,
        falloff_power: c.worm_falloff_power,
    }
}

fn ffi_to_ore_config(c: &FfiEngineConfig) -> OreConfig {
    OreConfig {
        host_rock: HostRockConfig {
            sandstone_depth: c.host_sandstone_depth,
            granite_depth: c.host_granite_depth,
            basalt_depth: c.host_basalt_depth,
            slate_depth: c.host_slate_depth,
            boundary_noise_amplitude: c.host_boundary_noise_amp,
            boundary_noise_frequency: c.host_boundary_noise_freq,
            basalt_intrusion_frequency: c.host_basalt_intrusion_freq,
            basalt_intrusion_threshold: c.host_basalt_intrusion_thresh,
            basalt_intrusion_depth_max: c.host_basalt_intrusion_depth_max,
        },
        iron: BandedIronConfig {
            band_frequency: c.iron_band_frequency,
            noise_perturbation: c.iron_noise_perturbation,
            noise_frequency: c.iron_noise_frequency,
            threshold: c.iron_threshold,
            depth_min: c.iron_depth_min,
            depth_max: c.iron_depth_max,
        },
        copper: OreVeinParams {
            frequency: c.copper_frequency,
            threshold: c.copper_threshold,
            depth_min: c.copper_depth_min,
            depth_max: c.copper_depth_max,
        },
        malachite: OreVeinParams {
            frequency: c.malachite_frequency,
            threshold: c.malachite_threshold,
            depth_min: c.malachite_depth_min,
            depth_max: c.malachite_depth_max,
        },
        quartz: OreVeinParams {
            frequency: c.quartz_frequency,
            threshold: c.quartz_threshold,
            depth_min: c.quartz_depth_min,
            depth_max: c.quartz_depth_max,
        },
        gold: OreVeinParams {
            frequency: c.gold_frequency,
            threshold: c.gold_threshold,
            depth_min: c.gold_depth_min,
            depth_max: c.gold_depth_max,
        },
        pyrite: OreVeinParams {
            frequency: c.pyrite_frequency,
            threshold: c.pyrite_threshold,
            depth_min: c.pyrite_depth_min,
            depth_max: c.pyrite_depth_max,
        },
        kimberlite: KimberlitePipeConfig {
            pipe_frequency_2d: c.kimb_pipe_freq_2d,
            pipe_threshold: c.kimb_pipe_threshold,
            depth_min: c.kimb_depth_min,
            depth_max: c.kimb_depth_max,
            diamond_threshold: c.kimb_diamond_threshold,
            diamond_frequency: c.kimb_diamond_frequency,
        },
        sulfide: SulfideBlobConfig {
            frequency: c.sulfide_frequency,
            threshold: c.sulfide_threshold,
            tin_threshold: c.sulfide_tin_threshold,
            depth_min: c.sulfide_depth_min,
            depth_max: c.sulfide_depth_max,
        },
        geode: GeodeConfig {
            frequency: c.geode_frequency,
            center_threshold: c.geode_center_threshold,
            shell_thickness: c.geode_shell_thickness,
            hollow_factor: c.geode_hollow_factor,
            depth_min: c.geode_depth_min,
            depth_max: c.geode_depth_max,
        },
        coal: OreVeinParams {
            frequency: if c.ore_coal_frequency > 0.0 { c.ore_coal_frequency } else { 0.03 },
            threshold: if c.ore_coal_threshold > 0.0 { c.ore_coal_threshold } else { 0.62 },
            depth_min: c.ore_coal_depth_min,
            depth_max: if c.ore_coal_depth_max > 0.0 { c.ore_coal_depth_max } else { 80.0 },
        },
        ore_domain_warp_strength: c.ore_domain_warp_strength,
        ore_warp_frequency: if c.ore_warp_frequency > 0.0 { c.ore_warp_frequency } else { 0.02 },
        ore_edge_falloff: c.ore_edge_falloff,
        ore_detail_weight: c.ore_detail_weight,
        iron_sedimentary_only: c.ore_iron_sedimentary_only != 0,
        iron_depth_fade: c.ore_iron_depth_fade != 0,
        copper_supergene: c.ore_copper_supergene != 0,
        copper_granite_contact: c.ore_copper_granite_contact != 0,
        malachite_depth_bias: c.ore_malachite_depth_bias != 0,
        kimberlite_carrot_taper: c.ore_kimberlite_carrot_taper != 0,
        diamond_depth_grade: c.ore_diamond_depth_grade != 0,
        sulfide_gossan_cap: c.ore_sulfide_gossan_cap != 0,
        sulfide_disseminated: c.ore_sulfide_disseminated != 0,
        pyrite_ore_halo: c.ore_pyrite_ore_halo != 0,
        quartz_planar_veins: c.ore_quartz_planar_veins != 0,
        gold_bonanza: c.ore_gold_bonanza != 0,
        geode_volcanic_host: c.ore_geode_volcanic_host != 0,
        geode_depth_scaling: c.ore_geode_depth_scaling != 0,
        coal_sedimentary_host: c.ore_coal_sedimentary_host != 0,
        coal_shallow_ceiling: c.ore_coal_shallow_ceiling != 0,
        coal_depth_enrichment: c.ore_coal_depth_enrichment != 0,
        ore_global_scale: if c.ore_global_scale >= 0.0 { c.ore_global_scale } else { 1.0 },
    }
}

fn ffi_to_formation_config(c: &FfiEngineConfig) -> FormationConfig {
    let def = FormationConfig::default();
    FormationConfig {
        enabled: c.formation_enabled != 0,
        placement_frequency: if c.formation_placement_frequency > 0.0 { c.formation_placement_frequency as f64 } else { def.placement_frequency },
        placement_threshold: if c.formation_placement_threshold > 0.0 { c.formation_placement_threshold as f64 } else { def.placement_threshold },
        stalactite_chance: if c.formation_stalactite_chance > 0.0 { c.formation_stalactite_chance } else { def.stalactite_chance },
        stalagmite_chance: if c.formation_stalagmite_chance > 0.0 { c.formation_stalagmite_chance } else { def.stalagmite_chance },
        flowstone_chance: if c.formation_flowstone_chance > 0.0 { c.formation_flowstone_chance } else { def.flowstone_chance },
        column_chance: if c.formation_column_chance > 0.0 { c.formation_column_chance } else { def.column_chance },
        column_max_gap: if c.formation_column_max_gap > 0 { c.formation_column_max_gap as usize } else { def.column_max_gap },
        length_min: if c.formation_length_min > 0.0 { c.formation_length_min } else { def.length_min },
        length_max: if c.formation_length_max > 0.0 { c.formation_length_max } else { def.length_max },
        radius_min: if c.formation_radius_min > 0.0 { c.formation_radius_min } else { def.radius_min },
        radius_max: if c.formation_radius_max > 0.0 { c.formation_radius_max } else { def.radius_max },
        max_radius: if c.formation_max_radius > 0.0 { c.formation_max_radius } else { def.max_radius },
        column_radius_min: if c.formation_column_radius_min > 0.0 { c.formation_column_radius_min } else { def.column_radius_min },
        column_radius_max: if c.formation_column_radius_max > 0.0 { c.formation_column_radius_max } else { def.column_radius_max },
        flowstone_length_min: if c.formation_flowstone_length_min > 0.0 { c.formation_flowstone_length_min } else { def.flowstone_length_min },
        flowstone_length_max: if c.formation_flowstone_length_max > 0.0 { c.formation_flowstone_length_max } else { def.flowstone_length_max },
        flowstone_thickness: if c.formation_flowstone_thickness > 0.0 { c.formation_flowstone_thickness } else { def.flowstone_thickness },
        min_air_gap: if c.formation_min_air_gap > 0 { c.formation_min_air_gap as usize } else { def.min_air_gap },
        min_clearance: if c.formation_min_clearance > 0 { c.formation_min_clearance as usize } else { def.min_clearance },
        smoothness: if c.formation_smoothness > 0.0 { c.formation_smoothness } else { def.smoothness },
        // Mega-Column
        mega_column_chance: if c.formation_mega_column_chance > 0.0 { c.formation_mega_column_chance } else { def.mega_column_chance },
        mega_column_min_gap: if c.formation_mega_column_min_gap > 0 { c.formation_mega_column_min_gap as usize } else { def.mega_column_min_gap },
        mega_column_radius_min: if c.formation_mega_column_radius_min > 0.0 { c.formation_mega_column_radius_min } else { def.mega_column_radius_min },
        mega_column_radius_max: if c.formation_mega_column_radius_max > 0.0 { c.formation_mega_column_radius_max } else { def.mega_column_radius_max },
        mega_column_noise_strength: if c.formation_mega_column_noise_strength > 0.0 { c.formation_mega_column_noise_strength } else { def.mega_column_noise_strength },
        mega_column_ring_frequency: if c.formation_mega_column_ring_frequency > 0.0 { c.formation_mega_column_ring_frequency } else { def.mega_column_ring_frequency },
        // Drapery
        drapery_chance: if c.formation_drapery_chance > 0.0 { c.formation_drapery_chance } else { def.drapery_chance },
        drapery_length_min: if c.formation_drapery_length_min > 0.0 { c.formation_drapery_length_min } else { def.drapery_length_min },
        drapery_length_max: if c.formation_drapery_length_max > 0.0 { c.formation_drapery_length_max } else { def.drapery_length_max },
        drapery_wave_frequency: if c.formation_drapery_wave_frequency > 0.0 { c.formation_drapery_wave_frequency } else { def.drapery_wave_frequency },
        drapery_wave_amplitude: if c.formation_drapery_wave_amplitude > 0.0 { c.formation_drapery_wave_amplitude } else { def.drapery_wave_amplitude },
        // Rimstone Dam
        rimstone_chance: if c.formation_rimstone_chance > 0.0 { c.formation_rimstone_chance } else { def.rimstone_chance },
        rimstone_dam_height_min: if c.formation_rimstone_dam_height_min > 0.0 { c.formation_rimstone_dam_height_min } else { def.rimstone_dam_height_min },
        rimstone_dam_height_max: if c.formation_rimstone_dam_height_max > 0.0 { c.formation_rimstone_dam_height_max } else { def.rimstone_dam_height_max },
        rimstone_pool_depth: if c.formation_rimstone_pool_depth > 0.0 { c.formation_rimstone_pool_depth } else { def.rimstone_pool_depth },
        rimstone_min_slope: if c.formation_rimstone_min_slope > 0.0 { c.formation_rimstone_min_slope } else { def.rimstone_min_slope },
        // Cave Shield
        shield_chance: if c.formation_shield_chance > 0.0 { c.formation_shield_chance } else { def.shield_chance },
        shield_radius_min: if c.formation_shield_radius_min > 0.0 { c.formation_shield_radius_min } else { def.shield_radius_min },
        shield_radius_max: if c.formation_shield_radius_max > 0.0 { c.formation_shield_radius_max } else { def.shield_radius_max },
        shield_max_tilt: if c.formation_shield_max_tilt > 0.0 { c.formation_shield_max_tilt } else { def.shield_max_tilt },
        shield_stalactite_chance: if c.formation_shield_stalactite_chance > 0.0 { c.formation_shield_stalactite_chance } else { def.shield_stalactite_chance },
        // Cauldron
        cauldron_chance: if c.formation_cauldron_chance > 0.0 { c.formation_cauldron_chance } else { def.cauldron_chance },
        cauldron_radius_min: if c.formation_cauldron_radius_min > 0.0 { c.formation_cauldron_radius_min } else { def.cauldron_radius_min },
        cauldron_radius_max: if c.formation_cauldron_radius_max > 0.0 { c.formation_cauldron_radius_max } else { def.cauldron_radius_max },
        cauldron_depth: if c.formation_cauldron_depth > 0.0 { c.formation_cauldron_depth } else { def.cauldron_depth },
        cauldron_lip_height: if c.formation_cauldron_lip_height > 0.0 { c.formation_cauldron_lip_height } else { def.cauldron_lip_height },
        cauldron_rim_stalagmite_count_min: if c.formation_cauldron_rim_stalagmite_count_min > 0 { c.formation_cauldron_rim_stalagmite_count_min } else { def.cauldron_rim_stalagmite_count_min },
        cauldron_rim_stalagmite_count_max: if c.formation_cauldron_rim_stalagmite_count_max > 0 { c.formation_cauldron_rim_stalagmite_count_max } else { def.cauldron_rim_stalagmite_count_max },
        cauldron_rim_stalagmite_scale: if c.formation_cauldron_rim_stalagmite_scale > 0.0 { c.formation_cauldron_rim_stalagmite_scale } else { def.cauldron_rim_stalagmite_scale },
        cauldron_floor_noise: if c.formation_cauldron_floor_noise >= 0.0 { c.formation_cauldron_floor_noise } else { def.cauldron_floor_noise },
        cauldron_water_chance: if c.formation_cauldron_water_chance >= 0.0 { c.formation_cauldron_water_chance } else { def.cauldron_water_chance },
        cauldron_lava_chance: if c.formation_cauldron_lava_chance >= 0.0 { c.formation_cauldron_lava_chance } else { def.cauldron_lava_chance },
        cauldron_wall_inset: if c.formation_cauldron_wall_inset > 0.0 { c.formation_cauldron_wall_inset } else { def.cauldron_wall_inset },
        cauldron_floor_inset: if c.formation_cauldron_floor_inset > 0 { c.formation_cauldron_floor_inset } else { def.cauldron_floor_inset },
    }
}

fn ffi_to_pool_config(c: &FfiEngineConfig) -> PoolConfig {
    // Pool fields pass through directly — the C++ struct already has correct
    // default initializers, and placement_threshold legitimately uses negative
    // values (e.g. -1.0 = accept all noise values).
    let fluid_specified = c.pool_water_pct > 0.0 || c.pool_lava_pct > 0.0 || c.pool_empty_pct > 0.0;
    PoolConfig {
        enabled: c.pool_enabled != 0,
        placement_frequency: if c.pool_placement_freq != 0.0 { c.pool_placement_freq } else { 0.08 },
        placement_threshold: c.pool_placement_thresh, // can be negative
        pool_chance: if c.pool_chance > 0.0 { c.pool_chance } else { 0.3 },
        min_area: if c.pool_min_area > 0 { c.pool_min_area as usize } else { 4 },
        max_radius: if c.pool_max_radius > 0 { c.pool_max_radius as usize } else { 4 },
        basin_depth: if c.pool_basin_depth > 0 { c.pool_basin_depth as usize } else { 2 },
        rim_height: if c.pool_rim_height > 0 { c.pool_rim_height as usize } else { 1 },
        water_pct: if fluid_specified { c.pool_water_pct } else { 0.75 },
        lava_pct: if fluid_specified { c.pool_lava_pct } else { 0.25 },
        empty_pct: c.pool_empty_pct,
        min_air_above: if c.pool_min_air_above > 0 { c.pool_min_air_above as usize } else { 3 },
        max_cave_height: if c.pool_max_cave_height > 0 { c.pool_max_cave_height as usize } else { 20 },
        min_floor_thickness: if c.pool_min_floor_thickness > 0 { c.pool_min_floor_thickness as usize } else { 2 },
        min_ground_depth: if c.pool_min_ground_depth > 0 { c.pool_min_ground_depth as usize } else { 2 },
        max_y_step: if c.pool_max_y_step > 0 { c.pool_max_y_step as usize } else { 2 },
        footprint_y_tolerance: if c.pool_footprint_y_tolerance > 0 { c.pool_footprint_y_tolerance as usize } else { 2 },
    }
}

fn ffi_to_mine_config(c: &FfiEngineConfig) -> MineConfig {
    MineConfig {
        smooth_iterations: if c.mine_smooth_iterations == 0 && c.mine_smooth_strength == 0.0 {
            2 // default
        } else {
            c.mine_smooth_iterations
        },
        smooth_strength: if c.mine_smooth_strength > 0.0 { c.mine_smooth_strength } else { 0.3 },
        dirty_expand: if c.mine_dirty_expand > 0 { c.mine_dirty_expand } else { 2 },
    }
}

/// Build a single OreCrystalConfig from a set of raw FFI scalar values.
/// All fields follow the same defaulting pattern: use the FFI value if > 0, else the type default.
fn ffi_to_ore_crystal_config(
    enabled: u8,
    chance: f32,
    density_threshold: f32,
    scale_min: f32,
    scale_max: f32,
    small_weight: f32,
    medium_weight: f32,
    large_weight: f32,
    normal_alignment: f32,
    cluster_size: u32,
    cluster_radius: f32,
    surface_offset: f32,
    vein_enabled: u8,
    vein_frequency: f32,
    vein_thickness: f32,
    vein_octaves: u32,
    vein_lacunarity: f32,
    vein_warp_strength: f32,
    vein_density: f32,
) -> OreCrystalConfig {
    let def = OreCrystalConfig::default();
    OreCrystalConfig {
        enabled: enabled != 0,
        chance: if chance > 0.0 { chance } else { def.chance },
        density_threshold,
        scale_min: if scale_min > 0.0 { scale_min } else { def.scale_min },
        scale_max: if scale_max > 0.0 { scale_max } else { def.scale_max },
        small_weight: if small_weight > 0.0 { small_weight } else { def.small_weight },
        medium_weight: if medium_weight > 0.0 { medium_weight } else { def.medium_weight },
        large_weight: if large_weight > 0.0 { large_weight } else { def.large_weight },
        normal_alignment,
        cluster_size: if cluster_size > 0 { cluster_size } else { def.cluster_size },
        cluster_radius: if cluster_radius > 0.0 { cluster_radius } else { def.cluster_radius },
        surface_offset,
        vein_enabled: vein_enabled != 0,
        vein_frequency: if vein_frequency > 0.0 { vein_frequency } else { def.vein_frequency },
        vein_thickness: if vein_thickness > 0.0 { vein_thickness } else { def.vein_thickness },
        vein_octaves: if vein_octaves > 0 { vein_octaves } else { def.vein_octaves },
        vein_lacunarity: if vein_lacunarity > 0.0 { vein_lacunarity } else { def.vein_lacunarity },
        vein_warp_strength,
        vein_density: if vein_density > 0.0 { vein_density } else { def.vein_density },
    }
}

fn ffi_to_crystal_config(c: &FfiEngineConfig) -> CrystalConfig {
    CrystalConfig {
        enabled: c.crystal_enabled != 0,
        iron: ffi_to_ore_crystal_config(
            c.crystal_iron_enabled, c.crystal_iron_chance, c.crystal_iron_density_threshold,
            c.crystal_iron_scale_min, c.crystal_iron_scale_max,
            c.crystal_iron_small_weight, c.crystal_iron_medium_weight, c.crystal_iron_large_weight,
            c.crystal_iron_normal_alignment,
            c.crystal_iron_cluster_size, c.crystal_iron_cluster_radius, c.crystal_iron_surface_offset,
            c.crystal_iron_vein_enabled, c.crystal_iron_vein_frequency, c.crystal_iron_vein_thickness,
            c.crystal_iron_vein_octaves, c.crystal_iron_vein_lacunarity, c.crystal_iron_vein_warp_strength,
            c.crystal_iron_vein_density,
        ),
        copper: ffi_to_ore_crystal_config(
            c.crystal_copper_enabled, c.crystal_copper_chance, c.crystal_copper_density_threshold,
            c.crystal_copper_scale_min, c.crystal_copper_scale_max,
            c.crystal_copper_small_weight, c.crystal_copper_medium_weight, c.crystal_copper_large_weight,
            c.crystal_copper_normal_alignment,
            c.crystal_copper_cluster_size, c.crystal_copper_cluster_radius, c.crystal_copper_surface_offset,
            c.crystal_copper_vein_enabled, c.crystal_copper_vein_frequency, c.crystal_copper_vein_thickness,
            c.crystal_copper_vein_octaves, c.crystal_copper_vein_lacunarity, c.crystal_copper_vein_warp_strength,
            c.crystal_copper_vein_density,
        ),
        malachite: ffi_to_ore_crystal_config(
            c.crystal_malachite_enabled, c.crystal_malachite_chance, c.crystal_malachite_density_threshold,
            c.crystal_malachite_scale_min, c.crystal_malachite_scale_max,
            c.crystal_malachite_small_weight, c.crystal_malachite_medium_weight, c.crystal_malachite_large_weight,
            c.crystal_malachite_normal_alignment,
            c.crystal_malachite_cluster_size, c.crystal_malachite_cluster_radius, c.crystal_malachite_surface_offset,
            c.crystal_malachite_vein_enabled, c.crystal_malachite_vein_frequency, c.crystal_malachite_vein_thickness,
            c.crystal_malachite_vein_octaves, c.crystal_malachite_vein_lacunarity, c.crystal_malachite_vein_warp_strength,
            c.crystal_malachite_vein_density,
        ),
        tin: ffi_to_ore_crystal_config(
            c.crystal_tin_enabled, c.crystal_tin_chance, c.crystal_tin_density_threshold,
            c.crystal_tin_scale_min, c.crystal_tin_scale_max,
            c.crystal_tin_small_weight, c.crystal_tin_medium_weight, c.crystal_tin_large_weight,
            c.crystal_tin_normal_alignment,
            c.crystal_tin_cluster_size, c.crystal_tin_cluster_radius, c.crystal_tin_surface_offset,
            c.crystal_tin_vein_enabled, c.crystal_tin_vein_frequency, c.crystal_tin_vein_thickness,
            c.crystal_tin_vein_octaves, c.crystal_tin_vein_lacunarity, c.crystal_tin_vein_warp_strength,
            c.crystal_tin_vein_density,
        ),
        gold: ffi_to_ore_crystal_config(
            c.crystal_gold_enabled, c.crystal_gold_chance, c.crystal_gold_density_threshold,
            c.crystal_gold_scale_min, c.crystal_gold_scale_max,
            c.crystal_gold_small_weight, c.crystal_gold_medium_weight, c.crystal_gold_large_weight,
            c.crystal_gold_normal_alignment,
            c.crystal_gold_cluster_size, c.crystal_gold_cluster_radius, c.crystal_gold_surface_offset,
            c.crystal_gold_vein_enabled, c.crystal_gold_vein_frequency, c.crystal_gold_vein_thickness,
            c.crystal_gold_vein_octaves, c.crystal_gold_vein_lacunarity, c.crystal_gold_vein_warp_strength,
            c.crystal_gold_vein_density,
        ),
        diamond: ffi_to_ore_crystal_config(
            c.crystal_diamond_enabled, c.crystal_diamond_chance, c.crystal_diamond_density_threshold,
            c.crystal_diamond_scale_min, c.crystal_diamond_scale_max,
            c.crystal_diamond_small_weight, c.crystal_diamond_medium_weight, c.crystal_diamond_large_weight,
            c.crystal_diamond_normal_alignment,
            c.crystal_diamond_cluster_size, c.crystal_diamond_cluster_radius, c.crystal_diamond_surface_offset,
            c.crystal_diamond_vein_enabled, c.crystal_diamond_vein_frequency, c.crystal_diamond_vein_thickness,
            c.crystal_diamond_vein_octaves, c.crystal_diamond_vein_lacunarity, c.crystal_diamond_vein_warp_strength,
            c.crystal_diamond_vein_density,
        ),
        kimberlite: ffi_to_ore_crystal_config(
            c.crystal_kimberlite_enabled, c.crystal_kimberlite_chance, c.crystal_kimberlite_density_threshold,
            c.crystal_kimberlite_scale_min, c.crystal_kimberlite_scale_max,
            c.crystal_kimberlite_small_weight, c.crystal_kimberlite_medium_weight, c.crystal_kimberlite_large_weight,
            c.crystal_kimberlite_normal_alignment,
            c.crystal_kimberlite_cluster_size, c.crystal_kimberlite_cluster_radius, c.crystal_kimberlite_surface_offset,
            c.crystal_kimberlite_vein_enabled, c.crystal_kimberlite_vein_frequency, c.crystal_kimberlite_vein_thickness,
            c.crystal_kimberlite_vein_octaves, c.crystal_kimberlite_vein_lacunarity, c.crystal_kimberlite_vein_warp_strength,
            c.crystal_kimberlite_vein_density,
        ),
        sulfide: ffi_to_ore_crystal_config(
            c.crystal_sulfide_enabled, c.crystal_sulfide_chance, c.crystal_sulfide_density_threshold,
            c.crystal_sulfide_scale_min, c.crystal_sulfide_scale_max,
            c.crystal_sulfide_small_weight, c.crystal_sulfide_medium_weight, c.crystal_sulfide_large_weight,
            c.crystal_sulfide_normal_alignment,
            c.crystal_sulfide_cluster_size, c.crystal_sulfide_cluster_radius, c.crystal_sulfide_surface_offset,
            c.crystal_sulfide_vein_enabled, c.crystal_sulfide_vein_frequency, c.crystal_sulfide_vein_thickness,
            c.crystal_sulfide_vein_octaves, c.crystal_sulfide_vein_lacunarity, c.crystal_sulfide_vein_warp_strength,
            c.crystal_sulfide_vein_density,
        ),
        quartz: ffi_to_ore_crystal_config(
            c.crystal_quartz_enabled, c.crystal_quartz_chance, c.crystal_quartz_density_threshold,
            c.crystal_quartz_scale_min, c.crystal_quartz_scale_max,
            c.crystal_quartz_small_weight, c.crystal_quartz_medium_weight, c.crystal_quartz_large_weight,
            c.crystal_quartz_normal_alignment,
            c.crystal_quartz_cluster_size, c.crystal_quartz_cluster_radius, c.crystal_quartz_surface_offset,
            c.crystal_quartz_vein_enabled, c.crystal_quartz_vein_frequency, c.crystal_quartz_vein_thickness,
            c.crystal_quartz_vein_octaves, c.crystal_quartz_vein_lacunarity, c.crystal_quartz_vein_warp_strength,
            c.crystal_quartz_vein_density,
        ),
        pyrite: ffi_to_ore_crystal_config(
            c.crystal_pyrite_enabled, c.crystal_pyrite_chance, c.crystal_pyrite_density_threshold,
            c.crystal_pyrite_scale_min, c.crystal_pyrite_scale_max,
            c.crystal_pyrite_small_weight, c.crystal_pyrite_medium_weight, c.crystal_pyrite_large_weight,
            c.crystal_pyrite_normal_alignment,
            c.crystal_pyrite_cluster_size, c.crystal_pyrite_cluster_radius, c.crystal_pyrite_surface_offset,
            c.crystal_pyrite_vein_enabled, c.crystal_pyrite_vein_frequency, c.crystal_pyrite_vein_thickness,
            c.crystal_pyrite_vein_octaves, c.crystal_pyrite_vein_lacunarity, c.crystal_pyrite_vein_warp_strength,
            c.crystal_pyrite_vein_density,
        ),
        amethyst: ffi_to_ore_crystal_config(
            c.crystal_amethyst_enabled, c.crystal_amethyst_chance, c.crystal_amethyst_density_threshold,
            c.crystal_amethyst_scale_min, c.crystal_amethyst_scale_max,
            c.crystal_amethyst_small_weight, c.crystal_amethyst_medium_weight, c.crystal_amethyst_large_weight,
            c.crystal_amethyst_normal_alignment,
            c.crystal_amethyst_cluster_size, c.crystal_amethyst_cluster_radius, c.crystal_amethyst_surface_offset,
            c.crystal_amethyst_vein_enabled, c.crystal_amethyst_vein_frequency, c.crystal_amethyst_vein_thickness,
            c.crystal_amethyst_vein_octaves, c.crystal_amethyst_vein_lacunarity, c.crystal_amethyst_vein_warp_strength,
            c.crystal_amethyst_vein_density,
        ),
        coal: ffi_to_ore_crystal_config(
            c.crystal_coal_enabled, c.crystal_coal_chance, c.crystal_coal_density_threshold,
            c.crystal_coal_scale_min, c.crystal_coal_scale_max,
            c.crystal_coal_small_weight, c.crystal_coal_medium_weight, c.crystal_coal_large_weight,
            c.crystal_coal_normal_alignment,
            c.crystal_coal_cluster_size, c.crystal_coal_cluster_radius, c.crystal_coal_surface_offset,
            c.crystal_coal_vein_enabled, c.crystal_coal_vein_frequency, c.crystal_coal_vein_thickness,
            c.crystal_coal_vein_octaves, c.crystal_coal_vein_lacunarity, c.crystal_coal_vein_warp_strength,
            c.crystal_coal_vein_density,
        ),
    }
}

fn ffi_to_water_table_config(c: &FfiEngineConfig) -> voxel_gen::config::WaterTableConfig {
    voxel_gen::config::WaterTableConfig {
        enabled: c.water_table_enabled != 0,
        base_y: if c.water_table_base_y != 0.0 { c.water_table_base_y } else { 170.0 },
        noise_amplitude: if c.water_table_noise_amplitude != 0.0 { c.water_table_noise_amplitude } else { 15.0 },
        noise_frequency: if c.water_table_noise_frequency > 0.0 { c.water_table_noise_frequency } else { 0.008 },
        spring_flow_rate: if c.water_table_spring_flow_rate > 0.0 { c.water_table_spring_flow_rate } else { 0.8 },
        min_porosity_for_spring: if c.water_table_min_porosity > 0.0 { c.water_table_min_porosity } else { 0.5 },
        drip_noise_frequency: if c.water_table_drip_noise_frequency > 0.0 { c.water_table_drip_noise_frequency } else { 0.15 },
        drip_noise_threshold: if c.water_table_drip_noise_threshold > 0.0 { c.water_table_drip_noise_threshold } else { 0.7 },
        drip_level: if c.water_table_drip_level > 0.0 { c.water_table_drip_level } else { 0.4 },
        max_springs_per_chunk: if c.water_table_max_springs > 0 { c.water_table_max_springs } else { 8 },
        max_drips_per_chunk: if c.water_table_max_drips > 0 { c.water_table_max_drips } else { 12 },
    }
}

fn ffi_to_pipe_lava_config(c: &FfiEngineConfig) -> voxel_gen::config::PipeLavaConfig {
    voxel_gen::config::PipeLavaConfig {
        enabled: c.pipe_lava_enabled != 0,
        activation_depth: if c.pipe_lava_activation_depth != 0.0 { c.pipe_lava_activation_depth } else { -80.0 },
        max_lava_per_chunk: if c.pipe_lava_max_per_chunk > 0 { c.pipe_lava_max_per_chunk } else { 6 },
        depth_scaling: if c.pipe_lava_depth_scaling > 0.0 { c.pipe_lava_depth_scaling } else { 0.5 },
    }
}

fn ffi_to_lava_tube_config(c: &FfiEngineConfig) -> voxel_gen::config::LavaTubeConfig {
    voxel_gen::config::LavaTubeConfig {
        enabled: c.lava_tube_enabled != 0,
        tubes_per_region: if c.lava_tube_tubes_per_region > 0.0 { c.lava_tube_tubes_per_region } else { 2.0 },
        depth_min: if c.lava_tube_depth_min != 0.0 { c.lava_tube_depth_min } else { -250.0 },
        depth_max: if c.lava_tube_depth_max != 0.0 { c.lava_tube_depth_max } else { -50.0 },
        radius_min: if c.lava_tube_radius_min > 0.0 { c.lava_tube_radius_min } else { 2.0 },
        radius_max: if c.lava_tube_radius_max > 0.0 { c.lava_tube_radius_max } else { 4.0 },
        max_steps: if c.lava_tube_max_steps > 0 { c.lava_tube_max_steps } else { 150 },
        step_length: if c.lava_tube_step_length > 0.0 { c.lava_tube_step_length } else { 1.5 },
        active_depth: if c.lava_tube_active_depth != 0.0 { c.lava_tube_active_depth } else { -120.0 },
        pipe_connection_radius: if c.lava_tube_pipe_connection_radius > 0.0 { c.lava_tube_pipe_connection_radius } else { 20.0 },
    }
}

fn ffi_to_hydrothermal_config(c: &FfiEngineConfig) -> voxel_gen::config::HydrothermalConfig {
    voxel_gen::config::HydrothermalConfig {
        enabled: c.hydrothermal_enabled != 0,
        radius: if c.hydrothermal_radius > 0 { c.hydrothermal_radius } else { 8 },
        max_per_chunk: if c.hydrothermal_max_per_chunk > 0 { c.hydrothermal_max_per_chunk } else { 4 },
    }
}

fn ffi_to_river_config(c: &FfiEngineConfig) -> voxel_gen::config::RiverConfig {
    voxel_gen::config::RiverConfig {
        enabled: c.river_enabled != 0,
        rivers_per_region: if c.river_rivers_per_region > 0.0 { c.river_rivers_per_region } else { 1.0 },
        width_min: if c.river_width_min > 0.0 { c.river_width_min } else { 3.0 },
        width_max: if c.river_width_max > 0.0 { c.river_width_max } else { 6.0 },
        height: if c.river_height > 0.0 { c.river_height } else { 2.5 },
        max_steps: if c.river_max_steps > 0 { c.river_max_steps } else { 300 },
        step_length: if c.river_step_length > 0.0 { c.river_step_length } else { 1.5 },
        layer_restriction: c.river_layer_restriction != 0,
        downslope_bias: if c.river_downslope_bias > 0.0 { c.river_downslope_bias } else { 0.02 },
    }
}

fn ffi_to_artesian_config(c: &FfiEngineConfig) -> voxel_gen::config::ArtesianConfig {
    voxel_gen::config::ArtesianConfig {
        enabled: c.artesian_enabled != 0,
        aquifer_y_center: if c.artesian_aquifer_y_center != 0.0 { c.artesian_aquifer_y_center } else { -15.0 },
        aquifer_thickness: if c.artesian_aquifer_thickness > 0.0 { c.artesian_aquifer_thickness } else { 3.0 },
        aquifer_noise_freq: if c.artesian_aquifer_noise_freq > 0.0 { c.artesian_aquifer_noise_freq } else { 0.01 },
        aquifer_noise_threshold: if c.artesian_aquifer_noise_threshold > 0.0 { c.artesian_aquifer_noise_threshold } else { 0.3 },
        pressure_noise_freq: if c.artesian_pressure_noise_freq > 0.0 { c.artesian_pressure_noise_freq } else { 0.02 },
        max_per_chunk: if c.artesian_max_per_chunk > 0 { c.artesian_max_per_chunk } else { 3 },
    }
}

fn ffi_to_zone_config(c: &FfiEngineConfig) -> voxel_gen::config::ZoneConfig {
    voxel_gen::config::ZoneConfig {
        enabled: c.zone_enabled != 0,
        cathedral_chance: if c.zone_cathedral_chance > 0.0 { c.zone_cathedral_chance } else { 0.15 },
        lake_chance: if c.zone_lake_chance > 0.0 { c.zone_lake_chance } else { 0.12 },
        canyon_chance: if c.zone_canyon_chance > 0.0 { c.zone_canyon_chance } else { 0.10 },
        lava_gallery_chance: if c.zone_lava_gallery_chance > 0.0 { c.zone_lava_gallery_chance } else { 0.08 },
        bioluminescent_chance: if c.zone_bioluminescent_chance > 0.0 { c.zone_bioluminescent_chance } else { 0.10 },
        terraces_chance: if c.zone_terraces_chance > 0.0 { c.zone_terraces_chance } else { 0.08 },
        frozen_chance: if c.zone_frozen_chance > 0.0 { c.zone_frozen_chance } else { 0.06 },
        cathedral_min_air: if c.zone_cathedral_min_air > 0 { c.zone_cathedral_min_air } else { 2000 },
        lake_min_air: if c.zone_lake_min_air > 0 { c.zone_lake_min_air } else { 1500 },
        canyon_min_air: if c.zone_canyon_min_air > 0 { c.zone_canyon_min_air } else { 800 },
        lava_gallery_min_air: if c.zone_lava_gallery_min_air > 0 { c.zone_lava_gallery_min_air } else { 600 },
        bioluminescent_min_air: if c.zone_bioluminescent_min_air > 0 { c.zone_bioluminescent_min_air } else { 400 },
        terraces_min_air: if c.zone_terraces_min_air > 0 { c.zone_terraces_min_air } else { 1000 },
        frozen_min_air: if c.zone_frozen_min_air > 0 { c.zone_frozen_min_air } else { 600 },
        cathedral_dome_scale: if c.zone_cathedral_dome_scale > 0.0 { c.zone_cathedral_dome_scale } else { 0.7 },
        cathedral_boulder_count_min: if c.zone_cathedral_boulder_count_min > 0 { c.zone_cathedral_boulder_count_min } else { 3 },
        cathedral_boulder_count_max: if c.zone_cathedral_boulder_count_max > 0 { c.zone_cathedral_boulder_count_max } else { 8 },
        cathedral_mega_stalagmite_chance: if c.zone_cathedral_mega_stalagmite_chance > 0.0 { c.zone_cathedral_mega_stalagmite_chance } else { 0.4 },
        cathedral_flowstone_coverage: if c.zone_cathedral_flowstone_coverage > 0.0 { c.zone_cathedral_flowstone_coverage } else { 0.3 },
        lake_depth: if c.zone_lake_depth > 0 { c.zone_lake_depth } else { 4 },
        lake_beach_width: if c.zone_lake_beach_width > 0.0 { c.zone_lake_beach_width } else { 3.0 },
        lake_island_min_radius: if c.zone_lake_island_min_radius > 0.0 { c.zone_lake_island_min_radius } else { 2.0 },
        canyon_width_min: if c.zone_canyon_width_min > 0.0 { c.zone_canyon_width_min } else { 3.0 },
        canyon_width_max: if c.zone_canyon_width_max > 0.0 { c.zone_canyon_width_max } else { 6.0 },
        canyon_height_min: if c.zone_canyon_height_min > 0.0 { c.zone_canyon_height_min } else { 12.0 },
        canyon_height_max: if c.zone_canyon_height_max > 0.0 { c.zone_canyon_height_max } else { 25.0 },
        canyon_bridge_chance: if c.zone_canyon_bridge_chance > 0.0 { c.zone_canyon_bridge_chance } else { 0.3 },
        lava_gallery_bench_spacing: if c.zone_lava_gallery_bench_spacing > 0.0 { c.zone_lava_gallery_bench_spacing } else { 4.0 },
        lava_gallery_lavacicle_chance: if c.zone_lava_gallery_lavacicle_chance > 0.0 { c.zone_lava_gallery_lavacicle_chance } else { 0.15 },
        bio_anchor_density: if c.zone_bio_anchor_density > 0.0 { c.zone_bio_anchor_density } else { 0.1 },
        bio_max_anchors: if c.zone_bio_max_anchors > 0 { c.zone_bio_max_anchors } else { 50 },
        terrace_tiers_min: if c.zone_terrace_tiers_min > 0 { c.zone_terrace_tiers_min } else { 3 },
        terrace_tiers_max: if c.zone_terrace_tiers_max > 0 { c.zone_terrace_tiers_max } else { 7 },
        terrace_step_height: if c.zone_terrace_step_height > 0.0 { c.zone_terrace_step_height } else { 4.0 },
        terrace_rim_height: if c.zone_terrace_rim_height > 0.0 { c.zone_terrace_rim_height } else { 1.5 },
        terrace_basin_depth: if c.zone_terrace_basin_depth > 0 { c.zone_terrace_basin_depth } else { 2 },
        frozen_floor_depth: if c.zone_frozen_floor_depth > 0 { c.zone_frozen_floor_depth } else { 2 },
        frozen_waterfall_count: if c.zone_frozen_waterfall_count > 0 { c.zone_frozen_waterfall_count } else { 2 },
        frozen_ice_stalactite_chance: if c.zone_frozen_ice_stalactite_chance > 0.0 { c.zone_frozen_ice_stalactite_chance } else { 0.3 },
        frozen_mega_chance: if c.zone_frozen_mega_chance > 0.0 { c.zone_frozen_mega_chance } else { 0.03 },
    }
}

fn ffi_to_mushroom_config(c: &FfiEngineConfig) -> voxel_gen::config::MushroomConfig {
    use voxel_gen::config::{KindConfig, MushroomConfig};
    // For each per-kind block, use fallbacks if the FFI gave zeros — that
    // matches the rest of the helpers and keeps an uninitialized UE config
    // from producing zero-density everywhere.
    let mk_kind = |enabled: u8, chance: f32, smin: f32, smax: f32, default: KindConfig| -> KindConfig {
        KindConfig {
            enabled: enabled != 0,
            spawn_chance: if chance > 0.0 { chance } else { default.spawn_chance },
            scale_min: if smin > 0.0 { smin } else { default.scale_min },
            scale_max: if smax > 0.0 { smax } else { default.scale_max },
        }
    };
    let d = MushroomConfig::default();
    MushroomConfig {
        enabled: c.mushroom_enabled != 0,
        global_density: if c.mushroom_global_density > 0.0 { c.mushroom_global_density } else { d.global_density },
        cluster_frequency: if c.mushroom_cluster_frequency > 0.0 { c.mushroom_cluster_frequency } else { d.cluster_frequency },
        cluster_threshold: if c.mushroom_cluster_threshold != 0.0 { c.mushroom_cluster_threshold } else { d.cluster_threshold },
        min_spacing_voxels: if c.mushroom_min_spacing_voxels > 0.0 { c.mushroom_min_spacing_voxels } else { d.min_spacing_voxels },
        ghost_tower_routing_share: if c.mushroom_ghost_tower_routing_share > 0.0 { c.mushroom_ghost_tower_routing_share } else { d.ghost_tower_routing_share },
        turkey_tail: mk_kind(
            c.mushroom_turkey_tail_enabled,
            c.mushroom_turkey_tail_spawn_chance,
            c.mushroom_turkey_tail_scale_min,
            c.mushroom_turkey_tail_scale_max,
            d.turkey_tail.clone(),
        ),
        foxfire: mk_kind(
            c.mushroom_foxfire_enabled,
            c.mushroom_foxfire_spawn_chance,
            c.mushroom_foxfire_scale_min,
            c.mushroom_foxfire_scale_max,
            d.foxfire.clone(),
        ),
        green_pepe: mk_kind(
            c.mushroom_green_pepe_enabled,
            c.mushroom_green_pepe_spawn_chance,
            c.mushroom_green_pepe_scale_min,
            c.mushroom_green_pepe_scale_max,
            d.green_pepe.clone(),
        ),
        ghost_tower: mk_kind(
            c.mushroom_ghost_tower_enabled,
            c.mushroom_ghost_tower_spawn_chance,
            c.mushroom_ghost_tower_scale_min,
            c.mushroom_ghost_tower_scale_max,
            d.ghost_tower.clone(),
        ),
    }
}

/// Debug: log pool config as received from FFI (temporary diagnostic).
pub(crate) fn debug_log_pool_config(c: &FfiEngineConfig) {
    eprintln!("[FFI-POOL] enabled={} freq={} thresh={} chance={} min_area={} max_radius={} \
              basin_depth={} rim_height={} water={} lava={} empty={} air_above={} \
              max_cave_height={} min_floor_thickness={} min_ground_depth={}",
        c.pool_enabled, c.pool_placement_freq, c.pool_placement_thresh,
        c.pool_chance, c.pool_min_area, c.pool_max_radius,
        c.pool_basin_depth, c.pool_rim_height,
        c.pool_water_pct, c.pool_lava_pct, c.pool_empty_pct, c.pool_min_air_above,
        c.pool_max_cave_height, c.pool_min_floor_thickness, c.pool_min_ground_depth);
}

/// Convert FFI config to FluidConfig.
pub(crate) fn ffi_config_to_fluid(c: &FfiEngineConfig) -> FluidConfig {
    FluidConfig {
        seed: c.seed,
        chunk_size: c.chunk_size as usize,
        tick_rate: if c.fluid_tick_rate > 0.0 { c.fluid_tick_rate } else { 10.0 },
        lava_tick_divisor: if c.fluid_lava_tick_divisor > 0 { c.fluid_lava_tick_divisor } else { 4 },
        water_spring_threshold: if c.fluid_water_spring_threshold > 0.0 { c.fluid_water_spring_threshold } else { 2.0 },
        lava_source_threshold: if c.fluid_lava_source_threshold > 0.0 { c.fluid_lava_source_threshold } else { 0.98 },
        lava_depth_max: if c.fluid_lava_depth_max != 0.0 { c.fluid_lava_depth_max } else { -50.0 },
        water_noise_frequency: if c.fluid_water_noise_frequency > 0.0 { c.fluid_water_noise_frequency } else { 0.05 },
        water_depth_min: if c.fluid_water_depth_min != 0.0 { c.fluid_water_depth_min } else { -9999.0 },
        water_depth_max: if c.fluid_water_depth_max != 0.0 { c.fluid_water_depth_max } else { 9999.0 },
        water_flow_rate: if c.fluid_water_flow_rate > 0.0 { c.fluid_water_flow_rate } else { 2.0 },
        water_spread_rate: if c.fluid_water_spread_rate > 0.0 { c.fluid_water_spread_rate } else { 2.0 },
        lava_noise_frequency: if c.fluid_lava_noise_frequency > 0.0 { c.fluid_lava_noise_frequency } else { 0.03 },
        lava_depth_min: if c.fluid_lava_depth_min != 0.0 { c.fluid_lava_depth_min } else { -9999.0 },
        lava_flow_rate: if c.fluid_lava_flow_rate > 0.0 { c.fluid_lava_flow_rate } else { 0.1 },
        lava_spread_rate: if c.fluid_lava_spread_rate > 0.0 { c.fluid_lava_spread_rate } else { 0.125 },
        cavern_source_bias: c.fluid_cavern_source_bias,
        tunnel_bend_threshold: c.fluid_tunnel_bend_threshold,
        water_substeps: 6,
        flow_anim_speed: 1.0,
        solid_threshold: 0.0,
        solid_corner_threshold: if c.fluid_solid_corner_threshold > 0 { c.fluid_solid_corner_threshold } else { 6 },
        // flow_solid_threshold and fractional_capacity removed — binary classification always used
        source_grace_ticks: if c.fluid_source_grace_ticks > 0 { c.fluid_source_grace_ticks } else { 50 },
        water_pressure_rate: 0.3,
        lava_pressure_rate: 0.1,
        mesh_smooth_iterations: 2,
        mesh_smooth_strength: 0.3,
        mesh_qef_refinement: true,
        mesh_recalc_normals: true,
    }
}

/// Convert FFI scan config to internal ScanConfig.
pub fn ffi_scan_config_to_scan_config(c: &FfiScanConfig) -> ScanConfig {
    ScanConfig {
        enable_density_seam: c.enable_density_seam != 0,
        enable_mesh_topology: c.enable_mesh_topology != 0,
        enable_seam_completeness: c.enable_seam_completeness != 0,
        enable_navigability: c.enable_navigability != 0,
        enable_worm_truncation: c.enable_worm_truncation != 0,
        enable_thin_walls: c.enable_thin_walls != 0,
        enable_winding_consistency: c.enable_winding_consistency != 0,
        enable_degenerate_triangles: c.enable_degenerate_triangles != 0,
        enable_worm_carve_verify: c.enable_worm_carve_verify != 0,
        enable_self_intersection: c.enable_self_intersection != 0,
        enable_seam_mesh_quality: c.enable_seam_mesh_quality != 0,
        density_subsample_count: c.density_subsample_count,
        raymarch_rays_per_chunk: c.raymarch_rays_per_chunk,
        raymarch_step_size: c.raymarch_step_size,
        max_vertex_zero_crossing_dist: c.max_vertex_zero_crossing_dist,
        min_passage_width: c.min_passage_width,
        min_triangle_area: c.min_triangle_area,
        max_edge_length: c.max_edge_length,
        thin_wall_max_thickness: c.thin_wall_max_thickness,
        self_intersection_tri_limit: c.self_intersection_tri_limit,
    }
}

/// Convert FFI config to SleepConfig.
pub fn ffi_config_to_sleep(c: &FfiEngineConfig) -> voxel_sleep::SleepConfig {
    use voxel_sleep::config::{CollapseConfig, DeepTimeConfig, GroundwaterConfig, MetamorphismConfig, MineralConfig, ReactionConfig, AureoleConfig, VeinConfig};
    // Build collapse config from new FFI fields (fall back to legacy fields if new are zero)
    let new_collapse = CollapseConfig {
        strut_survival: if c.sleep_strut_survival[1..].iter().any(|&v| v > 0.0) {
            c.sleep_strut_survival
        } else {
            CollapseConfig::default().strut_survival
        },
        stress_multiplier: if c.sleep_new_stress_multiplier > 0.0 { c.sleep_new_stress_multiplier }
            else if c.sleep_stress_multiplier > 0.0 { c.sleep_stress_multiplier } else { 0.8 },
        max_cascade_iterations: 3, // not exposed in new UI
        rubble_fill_ratio: if c.sleep_new_rubble_fill_ratio > 0.0 { c.sleep_new_rubble_fill_ratio }
            else if c.sleep_rubble_fill_ratio > 0.0 { c.sleep_rubble_fill_ratio } else { 0.65 },
        min_stress_for_cascade: if c.sleep_new_min_stress_cascade > 0.0 { c.sleep_new_min_stress_cascade }
            else if c.sleep_min_stress_for_cascade > 0.0 { c.sleep_min_stress_for_cascade } else { 0.95 },
        rubble_material_match: true,
        support_stress_penalty: if c.sleep_support_stress_penalty > 0.0 { c.sleep_support_stress_penalty } else { 1.0 },
        collapse_enabled: c.sleep_new_collapse_enabled != 0,
    };
    // Also build legacy collapse for backward compat
    let legacy_collapse = CollapseConfig {
        strut_survival: if c.sleep_strut_survival[1..].iter().any(|&v| v > 0.0) {
            c.sleep_strut_survival
        } else {
            CollapseConfig::default().strut_survival
        },
        stress_multiplier: if c.sleep_stress_multiplier > 0.0 { c.sleep_stress_multiplier } else { 0.8 },
        max_cascade_iterations: if c.sleep_max_cascade_iterations > 0 { c.sleep_max_cascade_iterations } else { 3 },
        rubble_fill_ratio: if c.sleep_rubble_fill_ratio > 0.0 { c.sleep_rubble_fill_ratio } else { 0.65 },
        min_stress_for_cascade: if c.sleep_min_stress_for_cascade > 0.0 { c.sleep_min_stress_for_cascade } else { 0.95 },
        rubble_material_match: c.sleep_rubble_material_match != 0,
        support_stress_penalty: if c.sleep_support_stress_penalty > 0.0 { c.sleep_support_stress_penalty } else { 1.0 },
        collapse_enabled: c.sleep_collapse_sub_enabled != 0,
    };
    voxel_sleep::SleepConfig {
        time_budget_ms: if c.sleep_time_budget_ms > 0 { c.sleep_time_budget_ms } else { 8000 },
        chunk_radius: c.sleep_chunk_radius.min(10),
        sleep_count: if c.sleep_count > 0 { c.sleep_count } else { 1 },
        accumulation_enabled: c.sleep_accumulation_enabled != 0,
        // Defensive cap: a misaligned FFI struct can deliver garbage f32 bits
        // (e.g. 0x42C20000 = f32 97.0 → u32 1,120,010,240). Anything over ~100
        // would already blow the wall-clock budget, so reject silly values and
        // log so we notice. Real values are 1..~30.
        accumulation_iterations: if c.sleep_accumulation_iterations > 0 && c.sleep_accumulation_iterations < 1000 {
            c.sleep_accumulation_iterations
        } else {
            if c.sleep_accumulation_iterations >= 1000 {
                eprintln!("[engine] sleep_accumulation_iterations garbage value {} (likely FFI misalignment) — clamping to 3", c.sleep_accumulation_iterations);
            }
            3
        },
        lava_solidification_enabled: c.sleep_lava_solidification_enabled != 0,
        nest_positions: Vec::new(),
        corpse_positions: Vec::new(),
        extra_sim_chunks: Vec::new(), // set later via voxel_set_sleep_poi_chunks
        phase1_enabled: c.sleep_phase1_enabled != 0,
        phase2_enabled: c.sleep_phase2_enabled != 0,
        phase3_enabled: c.sleep_phase3_enabled != 0,
        phase4_enabled: c.sleep_phase4_enabled != 0,
        groundwater: GroundwaterConfig {
            enabled: c.sleep_groundwater_enabled != 0,
            strength: if c.sleep_groundwater_strength > 0.0 { c.sleep_groundwater_strength } else { 0.3 },
            depth_baseline: c.sleep_gw_depth_baseline,
            depth_scale: if c.sleep_groundwater_depth_scale > 0.0 { c.sleep_groundwater_depth_scale } else { 0.02 },
            drip_zone_multiplier: if c.sleep_groundwater_drip_multiplier > 0.0 { c.sleep_groundwater_drip_multiplier } else { 2.0 },
            porosity_limestone: if c.sleep_gw_porosity_limestone > 0.0 { c.sleep_gw_porosity_limestone } else { 1.0 },
            porosity_sandstone: if c.sleep_gw_porosity_sandstone > 0.0 { c.sleep_gw_porosity_sandstone } else { 0.8 },
            porosity_slate: if c.sleep_gw_porosity_slate > 0.0 { c.sleep_gw_porosity_slate } else { 0.5 },
            porosity_marble: if c.sleep_gw_porosity_marble > 0.0 { c.sleep_gw_porosity_marble } else { 0.3 },
            porosity_granite: if c.sleep_gw_porosity_granite > 0.0 { c.sleep_gw_porosity_granite } else { 0.2 },
            porosity_basalt: if c.sleep_gw_porosity_basalt > 0.0 { c.sleep_gw_porosity_basalt } else { 0.1 },
            erosion_power: if c.sleep_gw_erosion_power > 0.0 { c.sleep_gw_erosion_power } else { 1.0 },
            flowstone_power: if c.sleep_gw_flowstone_power > 0.0 { c.sleep_gw_flowstone_power } else { 1.0 },
            enrichment_power: if c.sleep_gw_enrichment_power > 0.0 { c.sleep_gw_enrichment_power } else { 1.0 },
            soft_rock_mult: if c.sleep_gw_soft_rock_mult > 0.0 { c.sleep_gw_soft_rock_mult } else { 1.0 },
            hard_rock_mult: if c.sleep_gw_hard_rock_mult > 0.0 { c.sleep_gw_hard_rock_mult } else { 0.15 },
        },
        reaction: ReactionConfig {
            acid_dissolution_prob: if c.sleep_acid_dissolution_prob > 0.0 { c.sleep_acid_dissolution_prob } else { 0.25 },
            acid_dissolution_radius: if c.sleep_acid_dissolution_radius > 0 { c.sleep_acid_dissolution_radius } else { 3 },
            acid_dissolution_enabled: c.sleep_acid_dissolution_enabled != 0,
            acid_max_dissolved_per_source: if c.sleep_acid_max_dissolved_per_source > 0 { c.sleep_acid_max_dissolved_per_source } else { 30 },
            copper_oxidation_prob: if c.sleep_copper_oxidation_prob > 0.0 { c.sleep_copper_oxidation_prob } else { 0.0012 },
            copper_oxidation_enabled: c.sleep_copper_oxidation_enabled != 0,
            basalt_crust_prob: if c.sleep_basalt_crust_prob > 0.0 { c.sleep_basalt_crust_prob } else { 0.001 },
            basalt_crust_enabled: c.sleep_basalt_crust_enabled != 0,
            sulfide_acid_enabled: c.sleep_sulfide_acid_enabled != 0,
            sulfide_acid_prob: if c.sleep_sulfide_acid_prob > 0.0 { c.sleep_sulfide_acid_prob } else { 0.60 },
            sulfide_acid_radius: if c.sleep_sulfide_acid_radius > 0 { c.sleep_sulfide_acid_radius } else { 2 },
            sulfide_water_amplification: if c.sleep_sulfide_water_amplification > 0.0 { c.sleep_sulfide_water_amplification } else { 2.0 },
            limestone_acid_radius_boost: if c.sleep_limestone_acid_radius_boost > 0.0 { c.sleep_limestone_acid_radius_boost } else { 1.5 },
            gypsum_deposition_prob: if c.sleep_gypsum_deposition_prob > 0.0 { c.sleep_gypsum_deposition_prob } else { 0.18 },
            gypsum_enabled: c.sleep_gypsum_enabled != 0,
        },
        aureole: AureoleConfig {
            aureole_radius: if c.sleep_aureole_radius > 0 { c.sleep_aureole_radius } else { 10 },
            contact_limestone_to_marble_prob: if c.sleep_contact_marble_prob > 0.0 { c.sleep_contact_marble_prob } else { 0.18 },
            contact_sandstone_to_granite_prob: if c.sleep_contact_sandstone_to_granite_prob > 0.0 { c.sleep_contact_sandstone_to_granite_prob } else { 0.50 },
            mid_limestone_to_marble_prob: if c.sleep_mid_limestone_to_marble_prob > 0.0 { c.sleep_mid_limestone_to_marble_prob } else { 0.15 },
            mid_sandstone_to_granite_prob: if c.sleep_mid_sandstone_to_granite_prob > 0.0 { c.sleep_mid_sandstone_to_granite_prob } else { 0.25 },
            outer_limestone_to_marble_prob: if c.sleep_outer_limestone_to_marble_prob > 0.0 { c.sleep_outer_limestone_to_marble_prob } else { 0.30 },
            water_erosion_prob: if c.sleep_water_erosion_prob > 0.0 { c.sleep_water_erosion_prob } else { 0.05 },
            water_erosion_enabled: c.sleep_water_erosion_enabled != 0,
            metamorphism_enabled: c.sleep_aureole_metamorphism_enabled != 0,
            coal_maturation_enabled: c.sleep_coal_maturation_enabled != 0,
            coal_to_graphite_prob: if c.sleep_coal_to_graphite_prob > 0.0 { c.sleep_coal_to_graphite_prob } else { 0.70 },
            coal_to_graphite_mid_prob: if c.sleep_coal_to_graphite_mid_prob > 0.0 { c.sleep_coal_to_graphite_mid_prob } else { 0.35 },
            graphite_to_diamond_prob: if c.sleep_graphite_to_diamond_prob > 0.0 { c.sleep_graphite_to_diamond_prob } else { 0.15 },
            silicification_enabled: c.sleep_silicification_enabled != 0,
            silicification_limestone_prob: if c.sleep_silicification_limestone_prob > 0.0 { c.sleep_silicification_limestone_prob } else { 0.55 },
            silicification_sandstone_prob: if c.sleep_silicification_sandstone_prob > 0.0 { c.sleep_silicification_sandstone_prob } else { 0.15 },
            silicification_water_radius_mult: if c.sleep_silicification_water_radius_mult > 0 { c.sleep_silicification_water_radius_mult } else { 3 },
            contact_limestone_to_garnet_prob: if c.sleep_contact_limestone_to_garnet_prob > 0.0 { c.sleep_contact_limestone_to_garnet_prob } else { 0.65 },
            mid_limestone_to_garnet_prob: if c.sleep_mid_limestone_to_garnet_prob > 0.0 { c.sleep_mid_limestone_to_garnet_prob } else { 0.30 },
            mid_limestone_to_diopside_prob: if c.sleep_mid_limestone_to_diopside_prob > 0.0 { c.sleep_mid_limestone_to_diopside_prob } else { 0.65 },
            recrystallization_prob: if c.sleep_recrystallization_prob > 0.0 { c.sleep_recrystallization_prob } else { 0.70 },
            contact_slate_to_hornfels_prob: if c.sleep_contact_slate_to_hornfels_prob > 0.0 { c.sleep_contact_slate_to_hornfels_prob } else { 0.90 },
            mid_slate_to_hornfels_prob: if c.sleep_mid_slate_to_hornfels_prob > 0.0 { c.sleep_mid_slate_to_hornfels_prob } else { 0.60 },
            outer_slate_to_hornfels_prob: if c.sleep_outer_slate_to_hornfels_prob > 0.0 { c.sleep_outer_slate_to_hornfels_prob } else { 0.25 },
            zone_enabled: c.sleep_zone_enabled != 0,
            heat_multiplier: if c.sleep_heat_multiplier > 0.0 { c.sleep_heat_multiplier } else { 1.0 },
            radius_scale: if c.sleep_radius_scale > 0.0 { c.sleep_radius_scale } else { 1.0 },
            water_boost_max: if c.sleep_water_boost_max > 0.0 { c.sleep_water_boost_max } else { 0.6 },
            water_search_radius_mult: if c.sleep_water_search_radius_mult > 0.0 { c.sleep_water_search_radius_mult } else { 2.0 },
            large_vein_base_size: if c.sleep_large_vein_base_size > 0 { c.sleep_large_vein_base_size } else { 15 },
            small_vein_base_size: if c.sleep_small_vein_base_size > 0 { c.sleep_small_vein_base_size } else { 6 },
            min_lava_zone_size: if c.sleep_min_lava_zone_size > 0 { c.sleep_min_lava_zone_size } else { 5 },
            garnet_pocket_size: if c.sleep_garnet_pocket_size > 0 { c.sleep_garnet_pocket_size } else { 4 },
            diopside_pocket_size: if c.sleep_diopside_pocket_size > 0 { c.sleep_diopside_pocket_size } else { 4 },
            max_radius: if c.sleep_max_aureole_radius > 0.0 { c.sleep_max_aureole_radius } else { 10.0 },
            aureole_vein_count: if c.sleep_aureole_vein_count > 0 { c.sleep_aureole_vein_count } else { 8 },
            aureole_vein_min: if c.sleep_aureole_vein_min > 0 { c.sleep_aureole_vein_min } else { 6 },
            aureole_vein_max: if c.sleep_aureole_vein_max > 0 { c.sleep_aureole_vein_max } else { 20 },
            garnet_compact_size: if c.sleep_garnet_compact_size > 0 { c.sleep_garnet_compact_size } else { 8 },
            diopside_compact_size: if c.sleep_diopside_compact_size > 0 { c.sleep_diopside_compact_size } else { 8 },
            garnet_pocket_count: if c.sleep_garnet_pocket_count > 0 { c.sleep_garnet_pocket_count } else { 2 },
            diopside_pocket_count: if c.sleep_diopside_pocket_count > 0 { c.sleep_diopside_pocket_count } else { 1 },
            aureole_vein_spread: c.sleep_aureole_vein_spread,
            aureole_lava_volume_max_cells: if c.sleep_aureole_lava_max_cells > 0 { c.sleep_aureole_lava_max_cells } else { 10000 },
            aureole_lava_deposit_mult: c.sleep_aureole_lava_deposit_mult,
            aureole_lava_count_mult: c.sleep_aureole_lava_count_mult,
            aureole_water_search_radius: if c.sleep_aureole_water_search_radius > 0 { c.sleep_aureole_water_search_radius } else { 45 },
            aureole_water_max_cells: if c.sleep_aureole_water_max_cells > 0 { c.sleep_aureole_water_max_cells } else { 30 },
            aureole_water_deposit_mult: c.sleep_aureole_water_deposit_mult,
            aureole_wall_climbing: c.sleep_aureole_wall_climbing != 0,
            aureole_weight_up: if c.sleep_aureole_weight_up > 0.0 { c.sleep_aureole_weight_up } else { 3.0 },
            aureole_weight_depth: if c.sleep_aureole_weight_depth > 0.0 { c.sleep_aureole_weight_depth } else { 2.0 },
            aureole_weight_lateral: if c.sleep_aureole_weight_lateral > 0.0 { c.sleep_aureole_weight_lateral } else { 1.5 },
            aureole_surface_ratio: if c.sleep_aureole_surface_ratio > 0.0 { c.sleep_aureole_surface_ratio } else { 0.5 },
            aureole_min_connectivity: if c.sleep_aureole_min_connectivity > 0 { c.sleep_aureole_min_connectivity } else { 1 },
            aureole_weight_down: if c.sleep_aureole_weight_down > 0.0 { c.sleep_aureole_weight_down } else { 1.5 },
            aureole_veins_per_n_cells: c.sleep_aureole_veins_per_n_cells,
            aureole_garnet_per_n_cells: c.sleep_aureole_garnet_per_n_cells,
            aureole_diopside_per_n_cells: c.sleep_aureole_diopside_per_n_cells,
            aureole_cells_per_extra: if c.sleep_aureole_cells_per_extra > 0 { c.sleep_aureole_cells_per_extra } else { 90 },
            amphibolite_pyrite_pocket_count: if c.sleep_amphibolite_pyrite_pocket_count > 0 { c.sleep_amphibolite_pyrite_pocket_count } else { 2 },
            amphibolite_garnet_pocket_count: if c.sleep_amphibolite_garnet_pocket_count > 0 { c.sleep_amphibolite_garnet_pocket_count } else { 1 },
            amphibolite_pyrite_compact_size: if c.sleep_amphibolite_pyrite_compact_size > 0 { c.sleep_amphibolite_pyrite_compact_size } else { 8 },
            aureole_amphibolite_pyrite_per_n_cells: c.sleep_aureole_amphibolite_pyrite_per_n_cells,
            aureole_amphibolite_garnet_per_n_cells: c.sleep_aureole_amphibolite_garnet_per_n_cells,
            // Hydrothermal water-boost v2: defaults applied when UE side passes 0
            aureole_water_phase1_weight: if c.sleep_aureole_water_phase1_weight > 0.0 { c.sleep_aureole_water_phase1_weight } else { 1.0 },
            aureole_water_phase2_weight: if c.sleep_aureole_water_phase2_weight > 0.0 { c.sleep_aureole_water_phase2_weight } else { 0.25 },
            aureole_water_network_max_hops: if c.sleep_aureole_water_network_max_hops > 0 { c.sleep_aureole_water_network_max_hops } else { 50 },
            aureole_water_to_lava_ratio: if c.sleep_aureole_water_to_lava_ratio > 0.0 { c.sleep_aureole_water_to_lava_ratio } else { 1.2 },
            aureole_water_phase1_max_floor: if c.sleep_aureole_water_phase1_max_floor > 0 { c.sleep_aureole_water_phase1_max_floor } else { 50 },
            aureole_water_count_mult: if c.sleep_aureole_water_count_mult > 0.0 { c.sleep_aureole_water_count_mult } else { 1.0 },
        },
        veins: VeinConfig {
            vein_deposition_prob: if c.sleep_vein_deposition_prob > 0.0 { c.sleep_vein_deposition_prob } else { 0.85 },
            vein_enabled: c.sleep_vein_enabled != 0,
            convergence_radius: if c.sleep_vein_max_distance > 0 { c.sleep_vein_max_distance as f32 } else { 70.0 },
            hypothermal_height: if c.sleep_hypothermal_height > 0 { c.sleep_hypothermal_height } else { 25 },
            mesothermal_height: if c.sleep_mesothermal_height > 0 { c.sleep_mesothermal_height } else { 45 },
            epithermal_height: if c.sleep_epithermal_height > 0 { c.sleep_epithermal_height } else { 65 },
            horizontal_spread: if c.sleep_horizontal_spread > 0 { c.sleep_horizontal_spread } else { 20 },
            veins_per_zone_min: if c.sleep_veins_per_zone_min > 0 { c.sleep_veins_per_zone_min } else { 2 },
            veins_per_zone_max: if c.sleep_vein_max_per_source > 0 { c.sleep_vein_max_per_source } else { 4 },
            vein_size_min: if c.sleep_vein_size_min > 0 { c.sleep_vein_size_min } else { 8 },
            vein_size_max: if c.sleep_vein_size_max > 0 { c.sleep_vein_size_max } else { 30 },
            heat_direction_bias: if c.sleep_heat_direction_bias > 0.0 { c.sleep_heat_direction_bias } else { 0.3 },
            convergence_spacing: if c.sleep_vein_deposit_spacing > 0 { c.sleep_vein_deposit_spacing } else { 25 },
            epithermal_rarity: if c.sleep_epithermal_rarity > 0.0 { c.sleep_epithermal_rarity } else { 0.55 },
            crystal_growth_enabled: c.sleep_vein_crystal_growth_enabled != 0,
            crystal_growth_prob: if c.sleep_vein_crystal_growth_prob > 0.0 { c.sleep_vein_crystal_growth_prob } else { 0.30 },
            crystal_growth_max_per_chunk: if c.sleep_vein_crystal_growth_max_per_chunk > 0 { c.sleep_vein_crystal_growth_max_per_chunk } else { 4 },
            calcite_infill_enabled: c.sleep_vein_calcite_infill_enabled != 0,
            calcite_infill_prob: if c.sleep_vein_calcite_infill_prob > 0.0 { c.sleep_vein_calcite_infill_prob } else { 0.15 },
            calcite_infill_max_per_chunk: if c.sleep_vein_calcite_infill_max_per_chunk > 0 { c.sleep_vein_calcite_infill_max_per_chunk } else { 4 },
            flowstone_enabled: c.sleep_vein_flowstone_enabled != 0,
            flowstone_prob: if c.sleep_flowstone_prob > 0.0 { c.sleep_flowstone_prob } else { 0.10 },
            flowstone_max_per_chunk: if c.sleep_vein_flowstone_max_per_chunk > 0 { c.sleep_vein_flowstone_max_per_chunk } else { 3 },
            growth_density_min: if c.sleep_vein_growth_density_min > 0.0 { c.sleep_vein_growth_density_min } else { 0.3 },
            growth_density_max: if c.sleep_vein_growth_density_max > 0.0 { c.sleep_vein_growth_density_max } else { 0.6 },
            aperture_scaling_enabled: c.sleep_aperture_scaling_enabled != 0,
            host_rock_ore_enabled: c.sleep_host_rock_ore_enabled != 0,
            slate_pyrite_codeposit_prob: if c.sleep_slate_pyrite_codeposit_prob > 0.0 { c.sleep_slate_pyrite_codeposit_prob } else { 0.25 },
            slate_quartz_vein_prob: if c.sleep_slate_quartz_vein_prob > 0.0 { c.sleep_slate_quartz_vein_prob } else { 0.30 },
            wall_rock_alteration_prob: if c.sleep_wall_rock_alteration_prob > 0.0 { c.sleep_wall_rock_alteration_prob } else { 0.18 },
            min_vein_height: if c.sleep_min_vein_height > 0 { c.sleep_min_vein_height } else { 3 },
            water_volume_radius: if c.sleep_water_volume_radius > 0 { c.sleep_water_volume_radius } else { 8 },
            water_volume_max_cells: if c.sleep_water_volume_max_cells > 0 { c.sleep_water_volume_max_cells } else { 50 },
            water_volume_vein_mult: c.sleep_water_volume_vein_mult,
            water_volume_amount_mult: c.sleep_water_volume_amount_mult,
            lava_volume_radius: if c.sleep_lava_volume_radius > 0 { c.sleep_lava_volume_radius } else { 8 },
            lava_volume_max_cells: if c.sleep_lava_volume_max_cells > 0 { c.sleep_lava_volume_max_cells } else { 30 },
            lava_volume_vein_mult: c.sleep_lava_volume_vein_mult,
            lava_volume_amount_mult: c.sleep_lava_volume_amount_mult,
            spike_enabled: c.sleep_spike_enabled != 0,
            spike_count_min: if c.sleep_spike_count_min > 0 { c.sleep_spike_count_min } else { 4 },
            spike_count_max: if c.sleep_spike_count_max > 0 { c.sleep_spike_count_max } else { 10 },
            spike_length_min: if c.sleep_spike_length_min > 0 { c.sleep_spike_length_min } else { 2 },
            spike_length_max: if c.sleep_spike_length_max > 0 { c.sleep_spike_length_max } else { 5 },
            spike_taper: if c.sleep_spike_taper > 0.0 { c.sleep_spike_taper } else { 0.7 },
            vein_spread: c.sleep_vein_spread,
            vein_weight_up: if c.sleep_vein_weight_up > 0.0 { c.sleep_vein_weight_up } else { 3.0 },
            vein_weight_depth: if c.sleep_vein_weight_depth > 0.0 { c.sleep_vein_weight_depth } else { 2.0 },
            vein_weight_lateral: if c.sleep_vein_weight_lateral > 0.0 { c.sleep_vein_weight_lateral } else { 1.5 },
            vein_surface_ratio: if c.sleep_vein_surface_ratio > 0.0 { c.sleep_vein_surface_ratio } else { 0.5 },
            vein_min_connectivity: if c.sleep_vein_min_connectivity > 0 { c.sleep_vein_min_connectivity } else { 1 },
            vein_weight_down: c.sleep_vein_weight_down,
            water_proximity_bias: c.sleep_water_proximity_bias,
        },
        deeptime: DeepTimeConfig {
            enrichment_prob: if c.sleep_enrichment_prob > 0.0 { c.sleep_enrichment_prob } else { 0.90 },
            max_enrichment_per_chunk: if c.sleep_max_enrichment_per_chunk > 0 { c.sleep_max_enrichment_per_chunk } else { 400 },
            enrichment_search_radius: if c.sleep_enrichment_search_radius != 0 { c.sleep_enrichment_search_radius } else { 12 },
            enrichment_enabled: c.sleep_enrichment_enabled != 0,
            // Off by default — Block 3.3 disabled ambient groundwater
            // enrichment (Phase 4 step 2) due to catastrophic perf on
            // dense worlds. No FFI knob yet; flip in code once a perf
            // fix lands.
            ambient_enrichment_enabled: false,
            enrichment_cluster_min: if c.sleep_enrichment_cluster_min > 0 { c.sleep_enrichment_cluster_min } else { 3 },
            enrichment_cluster_max: if c.sleep_enrichment_cluster_max > 0 { c.sleep_enrichment_cluster_max } else { 30 },
            vein_thickening_enabled: c.sleep_vein_thickening_enabled != 0,
            vein_thickening_max_per_chunk: if c.sleep_vein_thickening_max_per_chunk > 0 { c.sleep_vein_thickening_max_per_chunk } else { 100 },
            vein_thickening_water_radius: if c.sleep_vein_thickening_water_radius > 0.0 { c.sleep_vein_thickening_water_radius } else { 40.0 },
            vein_thickening_coat_depth: if c.sleep_vein_thickening_coat_depth > 0 { c.sleep_vein_thickening_coat_depth } else { 1 },
            vein_thickening_finger_interval: if c.sleep_vein_thickening_finger_interval > 0 { c.sleep_vein_thickening_finger_interval } else { 5 },
            vein_thickening_finger_length_min: if c.sleep_vein_thickening_finger_length_min > 0 { c.sleep_vein_thickening_finger_length_min } else { 3 },
            vein_thickening_finger_length_max: if c.sleep_vein_thickening_finger_length_max > 0 { c.sleep_vein_thickening_finger_length_max } else { 5 },
            vein_thickening_finger_taper: if c.sleep_vein_thickening_finger_taper > 0.0 { c.sleep_vein_thickening_finger_taper } else { 0.7 },
            mature_formations_enabled: c.sleep_mature_formations_enabled != 0,
            stalactite_growth_prob: if c.sleep_stalactite_growth_prob > 0.0 { c.sleep_stalactite_growth_prob } else { 0.10 },
            column_formation_prob: if c.sleep_column_formation_prob > 0.0 { c.sleep_column_formation_prob } else { 0.05 },
            slate_zone_top: c.host_slate_depth,
            slate_zone_bottom: c.host_granite_depth,
            collapse: new_collapse,
            nest_fossilization: voxel_sleep::config::NestFossilizationConfig {
                enabled: c.sleep_nest_fossil_enabled != 0,
                nest_radius: if c.sleep_nest_fossil_radius > 0 { c.sleep_nest_fossil_radius } else { 2 },
                pyrite_prob: if c.sleep_nest_fossil_pyrite_prob > 0.0 { c.sleep_nest_fossil_pyrite_prob } else { 0.60 },
                opal_prob: if c.sleep_nest_fossil_opal_prob > 0.0 { c.sleep_nest_fossil_opal_prob } else { 0.40 },
                buried_required: c.sleep_nest_fossil_buried_required != 0,
                water_required_for_pyrite: c.sleep_nest_fossil_water_pyrite != 0,
                water_required_for_opal: c.sleep_nest_fossil_water_opal != 0,
            },
            corpse_fossilization: voxel_sleep::config::CorpseFossilizationConfig {
                enabled: c.sleep_corpse_fossil_enabled != 0,
                corpse_radius: if c.sleep_corpse_fossil_radius > 0 { c.sleep_corpse_fossil_radius } else { 1 },
                pyrite_prob: if c.sleep_corpse_fossil_pyrite_prob > 0.0 { c.sleep_corpse_fossil_pyrite_prob } else { 0.50 },
                calcium_prob: if c.sleep_corpse_fossil_calcium_prob > 0.0 { c.sleep_corpse_fossil_calcium_prob } else { 0.40 },
                water_required: c.sleep_corpse_fossil_water_required != 0,
                min_sleep_cycles: if c.sleep_corpse_fossil_min_cycles > 0 { c.sleep_corpse_fossil_min_cycles } else { 2 },
            },
            slate_aquitard_enabled: c.sleep_slate_aquitard_enabled != 0,
            slate_aquitard_factor: if c.sleep_slate_aquitard_factor > 0.0 { c.sleep_slate_aquitard_factor } else { 0.05 },
            slate_aquitard_concentration: if c.sleep_slate_aquitard_concentration > 0.0 { c.sleep_slate_aquitard_concentration } else { 2.0 },
        },
        // Legacy fields (kept for backward compat, old FFI fields still map here)
        metamorphism_enabled: c.sleep_metamorphism_enabled != 0,
        minerals_enabled: c.sleep_minerals_enabled != 0,
        collapse_enabled: c.sleep_collapse_enabled != 0,
        metamorphism: MetamorphismConfig {
            limestone_to_marble_prob: if c.sleep_limestone_to_marble_prob > 0.0 { c.sleep_limestone_to_marble_prob } else { 0.40 },
            limestone_to_marble_depth: if c.sleep_limestone_to_marble_depth != 0.0 { c.sleep_limestone_to_marble_depth } else { -50.0 },
            limestone_to_marble_enabled: c.sleep_limestone_to_marble_enabled != 0,
            sandstone_to_granite_prob: if c.sleep_sandstone_to_granite_prob > 0.0 { c.sleep_sandstone_to_granite_prob } else { 0.25 },
            sandstone_to_granite_depth: if c.sleep_sandstone_to_granite_depth != 0.0 { c.sleep_sandstone_to_granite_depth } else { -100.0 },
            sandstone_to_granite_min_neighbors: if c.sleep_sandstone_to_granite_min_neighbors > 0 { c.sleep_sandstone_to_granite_min_neighbors } else { 4 },
            sandstone_to_granite_enabled: c.sleep_sandstone_to_granite_enabled != 0,
            slate_to_marble_prob: if c.sleep_slate_to_marble_prob > 0.0 { c.sleep_slate_to_marble_prob } else { 0.60 },
            slate_to_marble_enabled: c.sleep_slate_to_marble_enabled != 0,
            granite_to_basalt_prob: if c.sleep_granite_to_basalt_prob > 0.0 { c.sleep_granite_to_basalt_prob } else { 0.15 },
            granite_to_basalt_min_air: if c.sleep_granite_to_basalt_min_air > 0 { c.sleep_granite_to_basalt_min_air } else { 2 },
            granite_to_basalt_enabled: c.sleep_granite_to_basalt_enabled != 0,
            iron_to_pyrite_prob: if c.sleep_iron_to_pyrite_prob > 0.0 { c.sleep_iron_to_pyrite_prob } else { 0.35 },
            iron_to_pyrite_search_radius: if c.sleep_iron_to_pyrite_search_radius > 0 { c.sleep_iron_to_pyrite_search_radius } else { 2 },
            iron_to_pyrite_enabled: c.sleep_iron_to_pyrite_enabled != 0,
            copper_to_malachite_prob: if c.sleep_copper_to_malachite_prob > 0.0 { c.sleep_copper_to_malachite_prob } else { 0.50 },
            copper_to_malachite_enabled: c.sleep_copper_to_malachite_enabled != 0,
        },
        minerals: MineralConfig {
            crystal_growth_max: if c.sleep_crystal_growth_max > 0 { c.sleep_crystal_growth_max } else { 2 },
            crystal_growth_enabled: c.sleep_crystal_growth_enabled != 0,
            crystal_growth_prob: if c.sleep_crystal_growth_prob > 0.0 { c.sleep_crystal_growth_prob } else { 0.3 },
            malachite_stalactite_max: if c.sleep_malachite_stalactite_max > 0 { c.sleep_malachite_stalactite_max } else { 1 },
            malachite_stalactite_enabled: c.sleep_malachite_stalactite_enabled != 0,
            malachite_stalactite_prob: if c.sleep_malachite_stalactite_prob > 0.0 { c.sleep_malachite_stalactite_prob } else { 0.2 },
            quartz_extension_prob: if c.sleep_quartz_extension_prob > 0.0 { c.sleep_quartz_extension_prob } else { 0.10 },
            quartz_extension_max: if c.sleep_quartz_extension_max > 0 { c.sleep_quartz_extension_max } else { 1 },
            quartz_extension_enabled: c.sleep_quartz_extension_enabled != 0,
            calcite_infill_max: if c.sleep_calcite_infill_max > 0 { c.sleep_calcite_infill_max } else { 1 },
            calcite_infill_depth: if c.sleep_calcite_infill_depth != 0.0 { c.sleep_calcite_infill_depth } else { -30.0 },
            calcite_infill_min_faces: if c.sleep_calcite_infill_min_faces > 0 { c.sleep_calcite_infill_min_faces } else { 3 },
            calcite_infill_enabled: c.sleep_calcite_infill_enabled != 0,
            calcite_infill_prob: if c.sleep_calcite_infill_prob > 0.0 { c.sleep_calcite_infill_prob } else { 0.15 },
            pyrite_crust_max: if c.sleep_pyrite_crust_max > 0 { c.sleep_pyrite_crust_max } else { 1 },
            pyrite_crust_min_solid: if c.sleep_pyrite_crust_min_solid > 0 { c.sleep_pyrite_crust_min_solid } else { 2 },
            pyrite_crust_enabled: c.sleep_pyrite_crust_enabled != 0,
            pyrite_crust_prob: if c.sleep_pyrite_crust_prob > 0.0 { c.sleep_pyrite_crust_prob } else { 0.1 },
            growth_density_min: if c.sleep_growth_density_min > 0.0 { c.sleep_growth_density_min } else { 0.3 },
            growth_density_max: if c.sleep_growth_density_max > 0.0 { c.sleep_growth_density_max } else { 0.6 },
        },
        collapse: legacy_collapse,
        stress: {
            let mut sc = voxel_core::stress::StressConfig::default();
            sc.propagation_radius = 4;
            sc.max_collapse_volume = 50;
            sc
        },
    }
}
