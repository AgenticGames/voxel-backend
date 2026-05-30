    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn dump_engine_config_offsets() {
        use std::mem::offset_of;
        macro_rules! off { ($f:ident) => { eprintln!("{:40} @ {}", stringify!($f), offset_of!(FfiEngineConfig, $f)); }; }
        eprintln!("=== FfiEngineConfig layout ===");
        eprintln!("total size = {}  (UE saw 3488 → 8-byte mismatch)", std::mem::size_of::<FfiEngineConfig>());
        // Sentinels at known struct positions
        off!(seed);
        off!(chunk_size);
        off!(region_size);
        // Bracket the legacy sleep* block (right before "new sleep fields"):
        off!(sleep_time_budget_ms);
        off!(sleep_chunk_radius);
        off!(sleep_count);
        off!(sleep_strut_survival);
        off!(sleep_stress_multiplier);
        off!(sleep_max_cascade_iterations);
        off!(sleep_rubble_fill_ratio);
        off!(sleep_min_stress_for_cascade);
        off!(sleep_rubble_material_match);
        off!(sleep_support_stress_penalty);
        off!(sleep_collapse_sub_enabled);
        off!(sleep_groundwater_enabled);
        off!(sleep_groundwater_strength);
        off!(sleep_groundwater_depth_scale);
        off!(sleep_groundwater_drip_multiplier);
        off!(sleep_phase1_enabled);
        off!(sleep_phase2_enabled);
        off!(sleep_phase3_enabled);
        off!(sleep_phase4_enabled);
        off!(sleep_acid_dissolution_prob);
        off!(sleep_flowstone_prob);
        off!(sleep_enrichment_prob);
        off!(sleep_vein_thickening_prob);
        off!(sleep_stalactite_growth_prob);
        off!(sleep_new_collapse_enabled);
        off!(sleep_new_stress_multiplier);
        off!(sleep_new_min_stress_cascade);
        off!(sleep_new_rubble_fill_ratio);
        off!(sleep_gw_erosion_power);
        off!(sleep_gw_flowstone_power);
        off!(sleep_gw_enrichment_power);
        off!(sleep_gw_soft_rock_mult);
        off!(sleep_gw_hard_rock_mult);
        off!(water_table_enabled);
        off!(water_table_base_y);
        off!(water_table_max_drips);
        off!(pipe_lava_enabled);
        off!(lava_tube_enabled);
        off!(hydrothermal_enabled);
        off!(river_enabled);
        off!(artesian_enabled);
        off!(fluid_sources_enabled);
        off!(fluid_solid_corner_threshold);
        off!(fluid_flow_solid_threshold);
        off!(fluid_fractional_capacity);
        off!(formation_cauldron_wall_inset);
        off!(formation_cauldron_floor_inset);
        off!(fluid_source_grace_ticks);
        off!(sleep_acid_max_dissolved_per_source);
        off!(sleep_vein_deposit_spacing);
        off!(sleep_lava_solidification_enabled);
        off!(sleep_zone_enabled);
        off!(sleep_heat_multiplier);
        off!(sleep_radius_scale);
        off!(sleep_water_boost_max);
        off!(sleep_water_search_radius_mult);
        off!(sleep_large_vein_base_size);
        off!(sleep_small_vein_base_size);
        off!(sleep_min_lava_zone_size);
        off!(sleep_garnet_pocket_size);
        off!(sleep_diopside_pocket_size);
        off!(sleep_max_aureole_radius);
        off!(sleep_accumulation_enabled);
        off!(sleep_accumulation_iterations);
        off!(sleep_gw_depth_baseline);
        off!(sleep_gw_porosity_limestone);
        off!(mushroom_enabled);
        eprintln!("=== end ===");
    }

    fn test_config() -> FfiEngineConfig {
        FfiEngineConfig {
            seed: 42,
            chunk_size: 16,
            worker_threads: 2,
            world_scale: 15.0,
            max_edge_length: 5.0,
            cavern_frequency: 0.05,
            cavern_threshold: 0.80,
            detail_octaves: 4,
            detail_persistence: 0.5,
            warp_amplitude: 5.0,
            worms_per_region: 5.0,
            worm_radius_min: 2.0,
            worm_radius_max: 4.0,
            worm_step_length: 1.0,
            worm_max_steps: 200,
            worm_falloff_power: 2.0,
            region_size: 3,
            // Host Rock
            host_sandstone_depth: 200.0,
            host_granite_depth: 160.0,
            host_basalt_depth: 20.0,
            host_slate_depth: -150.0,
            host_boundary_noise_amp: 8.0,
            host_boundary_noise_freq: 0.03,
            host_basalt_intrusion_freq: 0.02,
            host_basalt_intrusion_thresh: 0.85,
            host_basalt_intrusion_depth_max: 10.0,
            // Banded Iron
            iron_band_frequency: 0.2,
            iron_noise_perturbation: 1.0,
            iron_noise_frequency: 0.15,
            iron_threshold: 1.2,
            iron_depth_min: -200.0,
            iron_depth_max: 200.0,
            // Copper
            copper_frequency: 0.009,
            copper_threshold: 0.72,
            copper_depth_min: -30.0,
            copper_depth_max: 200.0,
            // Malachite
            malachite_frequency: 0.8,
            malachite_threshold: 0.1,
            malachite_depth_min: -200.0,
            malachite_depth_max: -30.0,
            // Quartz
            quartz_frequency: 0.01,
            quartz_threshold: 0.67,
            quartz_depth_min: -200.0,
            quartz_depth_max: 200.0,
            // Gold
            gold_frequency: 0.08,
            gold_threshold: 0.87,
            gold_depth_min: -200.0,
            gold_depth_max: 200.0,
            // Pyrite
            pyrite_frequency: 0.05,
            pyrite_threshold: 0.92,
            pyrite_depth_min: -200.0,
            pyrite_depth_max: 200.0,
            // Kimberlite
            kimb_pipe_freq_2d: 0.008,
            kimb_pipe_threshold: 0.9,
            kimb_depth_min: -200.0,
            kimb_depth_max: -30.0,
            kimb_diamond_threshold: 0.75,
            kimb_diamond_frequency: 0.2,
            // Sulfide
            sulfide_frequency: 0.5,
            sulfide_threshold: 0.2,
            sulfide_tin_threshold: 0.5,
            sulfide_depth_min: -200.0,
            sulfide_depth_max: -20.0,
            // Geode
            geode_frequency: 0.009,
            geode_center_threshold: 0.94,
            geode_shell_thickness: 0.01,
            geode_hollow_factor: -0.20,
            geode_depth_min: -200.0,
            geode_depth_max: 200.0,
            // Fluid
            fluid_tick_rate: 15.0,
            fluid_lava_tick_divisor: 4,
            fluid_water_spring_threshold: 2.0,
            fluid_lava_source_threshold: 0.98,
            fluid_lava_depth_max: -50.0,
            fluid_water_noise_frequency: 0.05,
            fluid_water_depth_min: -9999.0,
            fluid_water_depth_max: 9999.0,
            fluid_water_flow_rate: 0.25,
            fluid_water_spread_rate: 0.125,
            fluid_lava_noise_frequency: 0.03,
            fluid_lava_depth_min: -9999.0,
            fluid_lava_flow_rate: 0.1,
            fluid_lava_spread_rate: 0.125,
            fluid_cavern_source_bias: 0.0,
            fluid_tunnel_bend_threshold: 0.0,
            // Mine
            mine_smooth_iterations: 2,
            mine_smooth_strength: 0.3,
            mine_min_triangle_area: 0.01,
            mine_dirty_expand: 2,
            // Bounds
            bounds_size: 0.0,
            // Ore Visual Quality
            ore_domain_warp_strength: 0.0,
            ore_warp_frequency: 0.02,
            ore_edge_falloff: 0.0,
            ore_detail_weight: 0.0,
            // Mesh Smoothing
            mesh_smooth_iterations: 0,
            mesh_smooth_strength: 0.3,
            mesh_boundary_smooth: 0.3,
            mesh_recalc_normals: 1,
            // Pool Config
            pool_enabled: 1,
            pool_placement_freq: 0.08,
            pool_placement_thresh: 0.75,
            pool_chance: 0.3,
            pool_min_area: 4,
            pool_max_radius: 4,
            pool_basin_depth: 2,
            pool_rim_height: 1,
            pool_water_pct: 0.75,
            pool_lava_pct: 0.25,
            pool_empty_pct: 0.0,
            pool_min_air_above: 3,
            pool_max_cave_height: 20,
            pool_min_floor_thickness: 2,
            pool_min_ground_depth: 2,
            pool_max_y_step: 2,
            pool_footprint_y_tolerance: 2,
            // Formation Config
            formation_enabled: 1,
            formation_placement_frequency: 0.15,
            formation_placement_threshold: 0.3,
            formation_stalactite_chance: 0.15,
            formation_stalagmite_chance: 0.12,
            formation_flowstone_chance: 0.1,
            formation_column_chance: 0.08,
            formation_column_max_gap: 8,
            formation_length_min: 2.0,
            formation_length_max: 5.0,
            formation_radius_min: 0.3,
            formation_radius_max: 0.8,
            formation_max_radius: 1.0,
            formation_column_radius_min: 0.4,
            formation_column_radius_max: 1.0,
            formation_flowstone_length_min: 2.0,
            formation_flowstone_length_max: 5.0,
            formation_flowstone_thickness: 0.5,
            formation_min_air_gap: 3,
            formation_min_clearance: 2,
            formation_smoothness: 0.85,
            formation_mega_column_chance: 0.03,
            formation_mega_column_min_gap: 12,
            formation_mega_column_radius_min: 3.0,
            formation_mega_column_radius_max: 5.0,
            formation_mega_column_noise_strength: 0.3,
            formation_mega_column_ring_frequency: 0.8,
            formation_drapery_chance: 0.06,
            formation_drapery_length_min: 3.0,
            formation_drapery_length_max: 8.0,
            formation_drapery_wave_frequency: 1.5,
            formation_drapery_wave_amplitude: 0.4,
            formation_rimstone_chance: 0.04,
            formation_rimstone_dam_height_min: 1.0,
            formation_rimstone_dam_height_max: 1.5,
            formation_rimstone_pool_depth: 1.0,
            formation_rimstone_min_slope: 0.05,
            formation_shield_chance: 0.008,
            formation_shield_radius_min: 1.5,
            formation_shield_radius_max: 3.0,
            formation_shield_max_tilt: 30.0,
            formation_shield_stalactite_chance: 0.5,
            // Cauldron
            formation_cauldron_chance: 0.03,
            formation_cauldron_radius_min: 2.0,
            formation_cauldron_radius_max: 3.0,
            formation_cauldron_depth: 3.0,
            formation_cauldron_lip_height: 0.8,
            formation_cauldron_rim_stalagmite_count_min: 3,
            formation_cauldron_rim_stalagmite_count_max: 5,
            formation_cauldron_rim_stalagmite_scale: 0.5,
            formation_cauldron_floor_noise: 0.3,
            formation_cauldron_water_chance: 0.5,
            formation_cauldron_lava_chance: 0.2,
            // Geological Realism Toggles (all off in tests)
            ore_iron_sedimentary_only: 0,
            ore_iron_depth_fade: 0,
            ore_copper_supergene: 0,
            ore_copper_granite_contact: 0,
            ore_malachite_depth_bias: 0,
            ore_kimberlite_carrot_taper: 0,
            ore_diamond_depth_grade: 0,
            ore_sulfide_gossan_cap: 0,
            ore_sulfide_disseminated: 0,
            ore_pyrite_ore_halo: 0,
            ore_quartz_planar_veins: 0,
            ore_gold_bonanza: 0,
            ore_geode_volcanic_host: 0,
            ore_geode_depth_scaling: 0,
            // Coal
            ore_coal_frequency: 0.03,
            ore_coal_threshold: 0.62,
            ore_coal_depth_min: 10.0,
            ore_coal_depth_max: 80.0,
            ore_coal_sedimentary_host: 0,
            ore_coal_shallow_ceiling: 0,
            ore_coal_depth_enrichment: 0,
            // Ore Detail
            ore_detail_multiplier: 1,
            ore_protrusion: 0.0,
            // Crystal Config
            crystal_enabled: 1,
            // Iron crystals
            crystal_iron_enabled: 1,
            crystal_iron_chance: 0.25,
            crystal_iron_density_threshold: 0.3,
            crystal_iron_scale_min: 0.6,
            crystal_iron_scale_max: 1.4,
            crystal_iron_small_weight: 0.5,
            crystal_iron_medium_weight: 0.35,
            crystal_iron_large_weight: 0.15,
            crystal_iron_normal_alignment: 0.7,
            crystal_iron_cluster_size: 4,
            crystal_iron_cluster_radius: 1.0,
            crystal_iron_surface_offset: 0.1,
            crystal_iron_vein_enabled: 0,
            crystal_iron_vein_frequency: 0.0,
            crystal_iron_vein_thickness: 0.0,
            crystal_iron_vein_octaves: 0,
            crystal_iron_vein_lacunarity: 0.0,
            crystal_iron_vein_warp_strength: 0.0,
            crystal_iron_vein_density: 0.0,
            // Copper crystals
            crystal_copper_enabled: 1,
            crystal_copper_chance: 0.3,
            crystal_copper_density_threshold: 0.3,
            crystal_copper_scale_min: 0.4,
            crystal_copper_scale_max: 1.2,
            crystal_copper_small_weight: 0.5,
            crystal_copper_medium_weight: 0.35,
            crystal_copper_large_weight: 0.15,
            crystal_copper_normal_alignment: 0.7,
            crystal_copper_cluster_size: 3,
            crystal_copper_cluster_radius: 0.8,
            crystal_copper_surface_offset: 0.1,
            crystal_copper_vein_enabled: 0,
            crystal_copper_vein_frequency: 0.0,
            crystal_copper_vein_thickness: 0.0,
            crystal_copper_vein_octaves: 0,
            crystal_copper_vein_lacunarity: 0.0,
            crystal_copper_vein_warp_strength: 0.0,
            crystal_copper_vein_density: 0.0,
            // Malachite crystals
            crystal_malachite_enabled: 1,
            crystal_malachite_chance: 0.35,
            crystal_malachite_density_threshold: 0.25,
            crystal_malachite_scale_min: 0.5,
            crystal_malachite_scale_max: 1.3,
            crystal_malachite_small_weight: 0.5,
            crystal_malachite_medium_weight: 0.35,
            crystal_malachite_large_weight: 0.15,
            crystal_malachite_normal_alignment: 0.7,
            crystal_malachite_cluster_size: 3,
            crystal_malachite_cluster_radius: 0.8,
            crystal_malachite_surface_offset: 0.1,
            crystal_malachite_vein_enabled: 0,
            crystal_malachite_vein_frequency: 0.0,
            crystal_malachite_vein_thickness: 0.0,
            crystal_malachite_vein_octaves: 0,
            crystal_malachite_vein_lacunarity: 0.0,
            crystal_malachite_vein_warp_strength: 0.0,
            crystal_malachite_vein_density: 0.0,
            // Tin crystals
            crystal_tin_enabled: 1,
            crystal_tin_chance: 0.2,
            crystal_tin_density_threshold: 0.3,
            crystal_tin_scale_min: 0.5,
            crystal_tin_scale_max: 1.0,
            crystal_tin_small_weight: 0.5,
            crystal_tin_medium_weight: 0.35,
            crystal_tin_large_weight: 0.15,
            crystal_tin_normal_alignment: 0.7,
            crystal_tin_cluster_size: 3,
            crystal_tin_cluster_radius: 0.8,
            crystal_tin_surface_offset: 0.1,
            crystal_tin_vein_enabled: 0,
            crystal_tin_vein_frequency: 0.0,
            crystal_tin_vein_thickness: 0.0,
            crystal_tin_vein_octaves: 0,
            crystal_tin_vein_lacunarity: 0.0,
            crystal_tin_vein_warp_strength: 0.0,
            crystal_tin_vein_density: 0.0,
            // Gold crystals
            crystal_gold_enabled: 1,
            crystal_gold_chance: 0.4,
            crystal_gold_density_threshold: 0.3,
            crystal_gold_scale_min: 0.3,
            crystal_gold_scale_max: 0.8,
            crystal_gold_small_weight: 0.5,
            crystal_gold_medium_weight: 0.35,
            crystal_gold_large_weight: 0.15,
            crystal_gold_normal_alignment: 0.7,
            crystal_gold_cluster_size: 5,
            crystal_gold_cluster_radius: 0.6,
            crystal_gold_surface_offset: 0.1,
            crystal_gold_vein_enabled: 0,
            crystal_gold_vein_frequency: 0.0,
            crystal_gold_vein_thickness: 0.0,
            crystal_gold_vein_octaves: 0,
            crystal_gold_vein_lacunarity: 0.0,
            crystal_gold_vein_warp_strength: 0.0,
            crystal_gold_vein_density: 0.0,
            // Diamond crystals
            crystal_diamond_enabled: 1,
            crystal_diamond_chance: 0.5,
            crystal_diamond_density_threshold: 0.2,
            crystal_diamond_scale_min: 0.3,
            crystal_diamond_scale_max: 1.0,
            crystal_diamond_small_weight: 0.5,
            crystal_diamond_medium_weight: 0.35,
            crystal_diamond_large_weight: 0.15,
            crystal_diamond_normal_alignment: 0.7,
            crystal_diamond_cluster_size: 3,
            crystal_diamond_cluster_radius: 0.5,
            crystal_diamond_surface_offset: 0.1,
            crystal_diamond_vein_enabled: 0,
            crystal_diamond_vein_frequency: 0.0,
            crystal_diamond_vein_thickness: 0.0,
            crystal_diamond_vein_octaves: 0,
            crystal_diamond_vein_lacunarity: 0.0,
            crystal_diamond_vein_warp_strength: 0.0,
            crystal_diamond_vein_density: 0.0,
            // Kimberlite crystals
            crystal_kimberlite_enabled: 1,
            crystal_kimberlite_chance: 0.15,
            crystal_kimberlite_density_threshold: 0.3,
            crystal_kimberlite_scale_min: 0.8,
            crystal_kimberlite_scale_max: 2.0,
            crystal_kimberlite_small_weight: 0.5,
            crystal_kimberlite_medium_weight: 0.35,
            crystal_kimberlite_large_weight: 0.15,
            crystal_kimberlite_normal_alignment: 0.7,
            crystal_kimberlite_cluster_size: 2,
            crystal_kimberlite_cluster_radius: 1.2,
            crystal_kimberlite_surface_offset: 0.1,
            crystal_kimberlite_vein_enabled: 0,
            crystal_kimberlite_vein_frequency: 0.0,
            crystal_kimberlite_vein_thickness: 0.0,
            crystal_kimberlite_vein_octaves: 0,
            crystal_kimberlite_vein_lacunarity: 0.0,
            crystal_kimberlite_vein_warp_strength: 0.0,
            crystal_kimberlite_vein_density: 0.0,
            // Sulfide crystals
            crystal_sulfide_enabled: 1,
            crystal_sulfide_chance: 0.2,
            crystal_sulfide_density_threshold: 0.3,
            crystal_sulfide_scale_min: 0.5,
            crystal_sulfide_scale_max: 1.2,
            crystal_sulfide_small_weight: 0.5,
            crystal_sulfide_medium_weight: 0.35,
            crystal_sulfide_large_weight: 0.15,
            crystal_sulfide_normal_alignment: 0.7,
            crystal_sulfide_cluster_size: 3,
            crystal_sulfide_cluster_radius: 0.8,
            crystal_sulfide_surface_offset: 0.1,
            crystal_sulfide_vein_enabled: 0,
            crystal_sulfide_vein_frequency: 0.0,
            crystal_sulfide_vein_thickness: 0.0,
            crystal_sulfide_vein_octaves: 0,
            crystal_sulfide_vein_lacunarity: 0.0,
            crystal_sulfide_vein_warp_strength: 0.0,
            crystal_sulfide_vein_density: 0.0,
            // Quartz crystals
            crystal_quartz_enabled: 1,
            crystal_quartz_chance: 0.4,
            crystal_quartz_density_threshold: 0.3,
            crystal_quartz_scale_min: 0.4,
            crystal_quartz_scale_max: 1.5,
            crystal_quartz_small_weight: 0.5,
            crystal_quartz_medium_weight: 0.35,
            crystal_quartz_large_weight: 0.15,
            crystal_quartz_normal_alignment: 0.7,
            crystal_quartz_cluster_size: 4,
            crystal_quartz_cluster_radius: 0.7,
            crystal_quartz_surface_offset: 0.1,
            crystal_quartz_vein_enabled: 0,
            crystal_quartz_vein_frequency: 0.0,
            crystal_quartz_vein_thickness: 0.0,
            crystal_quartz_vein_octaves: 0,
            crystal_quartz_vein_lacunarity: 0.0,
            crystal_quartz_vein_warp_strength: 0.0,
            crystal_quartz_vein_density: 0.0,
            // Pyrite crystals
            crystal_pyrite_enabled: 1,
            crystal_pyrite_chance: 0.3,
            crystal_pyrite_density_threshold: 0.3,
            crystal_pyrite_scale_min: 0.3,
            crystal_pyrite_scale_max: 0.9,
            crystal_pyrite_small_weight: 0.5,
            crystal_pyrite_medium_weight: 0.35,
            crystal_pyrite_large_weight: 0.15,
            crystal_pyrite_normal_alignment: 0.7,
            crystal_pyrite_cluster_size: 5,
            crystal_pyrite_cluster_radius: 0.5,
            crystal_pyrite_surface_offset: 0.1,
            crystal_pyrite_vein_enabled: 0,
            crystal_pyrite_vein_frequency: 0.0,
            crystal_pyrite_vein_thickness: 0.0,
            crystal_pyrite_vein_octaves: 0,
            crystal_pyrite_vein_lacunarity: 0.0,
            crystal_pyrite_vein_warp_strength: 0.0,
            crystal_pyrite_vein_density: 0.0,
            // Amethyst crystals
            crystal_amethyst_enabled: 1,
            crystal_amethyst_chance: 0.45,
            crystal_amethyst_density_threshold: 0.2,
            crystal_amethyst_scale_min: 0.4,
            crystal_amethyst_scale_max: 1.4,
            crystal_amethyst_small_weight: 0.5,
            crystal_amethyst_medium_weight: 0.35,
            crystal_amethyst_large_weight: 0.15,
            crystal_amethyst_normal_alignment: 0.7,
            crystal_amethyst_cluster_size: 4,
            crystal_amethyst_cluster_radius: 0.8,
            crystal_amethyst_surface_offset: 0.1,
            crystal_amethyst_vein_enabled: 0,
            crystal_amethyst_vein_frequency: 0.0,
            crystal_amethyst_vein_thickness: 0.0,
            crystal_amethyst_vein_octaves: 0,
            crystal_amethyst_vein_lacunarity: 0.0,
            crystal_amethyst_vein_warp_strength: 0.0,
            crystal_amethyst_vein_density: 0.0,
            // Coal crystals
            crystal_coal_enabled: 1,
            crystal_coal_chance: 0.1,
            crystal_coal_density_threshold: 0.3,
            crystal_coal_scale_min: 0.3,
            crystal_coal_scale_max: 0.7,
            crystal_coal_small_weight: 0.5,
            crystal_coal_medium_weight: 0.35,
            crystal_coal_large_weight: 0.15,
            crystal_coal_normal_alignment: 0.7,
            crystal_coal_cluster_size: 2,
            crystal_coal_cluster_radius: 0.5,
            crystal_coal_surface_offset: 0.1,
            crystal_coal_vein_enabled: 0,
            crystal_coal_vein_frequency: 0.0,
            crystal_coal_vein_thickness: 0.0,
            crystal_coal_vein_octaves: 0,
            crystal_coal_vein_lacunarity: 0.0,
            crystal_coal_vein_warp_strength: 0.0,
            crystal_coal_vein_density: 0.0,
            // Sleep Config
            sleep_time_budget_ms: 8000,
            sleep_chunk_radius: 1,
            sleep_metamorphism_enabled: 1,
            sleep_minerals_enabled: 1,
            sleep_collapse_enabled: 1,
            sleep_count: 1,
            // Sleep Metamorphism
            sleep_limestone_to_marble_prob: 0.40,
            sleep_limestone_to_marble_depth: -50.0,
            sleep_limestone_to_marble_enabled: 1,
            sleep_sandstone_to_granite_prob: 0.25,
            sleep_sandstone_to_granite_depth: -100.0,
            sleep_sandstone_to_granite_min_neighbors: 4,
            sleep_sandstone_to_granite_enabled: 1,
            sleep_slate_to_marble_prob: 0.60,
            sleep_slate_to_marble_enabled: 1,
            sleep_granite_to_basalt_prob: 0.15,
            sleep_granite_to_basalt_min_air: 2,
            sleep_granite_to_basalt_enabled: 1,
            sleep_iron_to_pyrite_prob: 0.35,
            sleep_iron_to_pyrite_search_radius: 2,
            sleep_iron_to_pyrite_enabled: 1,
            sleep_copper_to_malachite_prob: 0.50,
            sleep_copper_to_malachite_enabled: 1,
            // Sleep Minerals
            sleep_crystal_growth_max: 2,
            sleep_crystal_growth_enabled: 1,
            sleep_crystal_growth_prob: 0.3,
            sleep_malachite_stalactite_max: 1,
            sleep_malachite_stalactite_enabled: 1,
            sleep_malachite_stalactite_prob: 0.2,
            sleep_quartz_extension_prob: 0.10,
            sleep_quartz_extension_max: 1,
            sleep_quartz_extension_enabled: 1,
            sleep_calcite_infill_max: 1,
            sleep_calcite_infill_depth: -30.0,
            sleep_calcite_infill_min_faces: 3,
            sleep_calcite_infill_enabled: 1,
            sleep_calcite_infill_prob: 0.15,
            sleep_pyrite_crust_max: 1,
            sleep_pyrite_crust_min_solid: 2,
            sleep_pyrite_crust_enabled: 1,
            sleep_pyrite_crust_prob: 0.1,
            sleep_growth_density_min: 0.3,
            sleep_growth_density_max: 0.6,
            // Sleep Collapse
            sleep_strut_survival: [0.0, 0.50, 0.70, 0.85, 0.95, 0.99],
            sleep_stress_multiplier: 1.5,
            sleep_max_cascade_iterations: 8,
            sleep_rubble_fill_ratio: 0.40,
            sleep_min_stress_for_cascade: 0.7,
            sleep_rubble_material_match: 1,
            sleep_support_stress_penalty: 1.0,
            sleep_collapse_sub_enabled: 1,
            // New 4-phase + Groundwater fields
            sleep_groundwater_enabled: 1,
            sleep_groundwater_strength: 0.3,
            sleep_groundwater_depth_scale: 0.02,
            sleep_groundwater_drip_multiplier: 2.0,
            sleep_phase1_enabled: 1,
            sleep_phase2_enabled: 1,
            sleep_phase3_enabled: 1,
            sleep_phase4_enabled: 1,
            sleep_acid_dissolution_prob: 0.25,
            sleep_copper_oxidation_prob: 0.0012,
            sleep_basalt_crust_prob: 0.001,
            sleep_acid_max_dissolved_per_source: 30,
            sleep_vein_deposit_spacing: 25,  // now convergence_spacing
            sleep_lava_solidification_enabled: 1,
            sleep_zone_enabled: 1,
            sleep_heat_multiplier: 1.0,
            sleep_radius_scale: 1.0,
            sleep_water_boost_max: 0.6,
            sleep_water_search_radius_mult: 2.0,
            sleep_large_vein_base_size: 15,
            sleep_small_vein_base_size: 6,
            sleep_min_lava_zone_size: 5,
            sleep_garnet_pocket_size: 4,
            sleep_diopside_pocket_size: 4,
            sleep_max_aureole_radius: 10.0,
            sleep_aureole_radius: 10,
            sleep_contact_marble_prob: 0.18,
            sleep_water_erosion_prob: 0.05,
            sleep_water_erosion_enabled: 1,
            sleep_vein_deposition_prob: 0.85,
            sleep_vein_max_distance: 70,    // now convergence_radius
            sleep_vein_max_per_source: 4,   // now veins_per_zone_max

            sleep_flowstone_prob: 0.10,
            sleep_enrichment_prob: 0.90,
            sleep_vein_thickening_prob: 0.35,
            sleep_stalactite_growth_prob: 0.10,
            sleep_new_collapse_enabled: 1,
            sleep_new_stress_multiplier: 1.5,
            sleep_new_min_stress_cascade: 0.7,
            sleep_new_rubble_fill_ratio: 0.40,
            // Groundwater power controls
            sleep_gw_erosion_power: 1.0,
            sleep_gw_flowstone_power: 1.0,
            sleep_gw_enrichment_power: 1.0,
            sleep_gw_soft_rock_mult: 1.0,
            sleep_gw_hard_rock_mult: 0.15,
            // Water Table Config
            water_table_enabled: 1,
            water_table_base_y: 170.0,
            water_table_noise_amplitude: 15.0,
            water_table_noise_frequency: 0.008,
            water_table_spring_flow_rate: 0.8,
            water_table_min_porosity: 0.5,
            water_table_drip_noise_frequency: 0.15,
            water_table_drip_noise_threshold: 0.7,
            water_table_drip_level: 0.4,
            water_table_max_springs: 8,
            water_table_max_drips: 12,
            // Pipe Lava Config
            pipe_lava_enabled: 1,
            pipe_lava_activation_depth: -80.0,
            pipe_lava_max_per_chunk: 6,
            pipe_lava_depth_scaling: 0.5,
            // Lava Tube Config
            lava_tube_enabled: 1,
            lava_tube_tubes_per_region: 2.0,
            lava_tube_depth_min: -250.0,
            lava_tube_depth_max: -50.0,
            lava_tube_radius_min: 2.0,
            lava_tube_radius_max: 4.0,
            lava_tube_max_steps: 150,
            lava_tube_step_length: 1.5,
            lava_tube_active_depth: -120.0,
            lava_tube_pipe_connection_radius: 20.0,
            // Hydrothermal Config
            hydrothermal_enabled: 1,
            hydrothermal_radius: 8,
            hydrothermal_max_per_chunk: 4,
            // River Config
            river_enabled: 1,
            river_rivers_per_region: 1.0,
            river_width_min: 3.0,
            river_width_max: 6.0,
            river_height: 2.5,
            river_max_steps: 300,
            river_step_length: 1.5,
            river_layer_restriction: 1,
            river_downslope_bias: 0.02,
            // Artesian Config
            artesian_enabled: 1,
            artesian_aquifer_y_center: -15.0,
            artesian_aquifer_thickness: 3.0,
            artesian_aquifer_noise_freq: 0.01,
            artesian_aquifer_noise_threshold: 0.3,
            artesian_pressure_noise_freq: 0.02,
            artesian_max_per_chunk: 3,
            // Fluid Sources Toggle
            fluid_sources_enabled: 1,
            // Fluid Tuning
            fluid_solid_corner_threshold: 6,
            fluid_flow_solid_threshold: 6,
            fluid_fractional_capacity: 1,
            // Cauldron Inset Tuning
            formation_cauldron_wall_inset: 1.0,
            formation_cauldron_floor_inset: 1,
            // Grace Period
            fluid_source_grace_ticks: 50,
            // ── New Sleep Fields (Phase A overhaul) ──
            // Top-level sleep
            sleep_accumulation_enabled: 1,
            sleep_accumulation_iterations: 3,
            // Groundwater (depth_baseline + 6 porosities)
            sleep_gw_depth_baseline: 0.0,
            sleep_gw_porosity_limestone: 1.0,
            sleep_gw_porosity_sandstone: 0.8,
            sleep_gw_porosity_slate: 0.5,
            sleep_gw_porosity_marble: 0.3,
            sleep_gw_porosity_granite: 0.2,
            sleep_gw_porosity_basalt: 0.1,
            // Phase 1: Reaction (11 fields)
            sleep_acid_dissolution_radius: 3,
            sleep_acid_dissolution_enabled: 1,
            sleep_copper_oxidation_enabled: 1,
            sleep_basalt_crust_enabled: 1,
            sleep_sulfide_acid_enabled: 1,
            sleep_sulfide_acid_prob: 0.60,
            sleep_sulfide_acid_radius: 2,
            sleep_sulfide_water_amplification: 2.0,
            sleep_limestone_acid_radius_boost: 1.5,
            sleep_gypsum_deposition_prob: 0.18,
            sleep_gypsum_enabled: 1,
            // Phase 2: Aureole (20 fields)
            sleep_contact_sandstone_to_granite_prob: 0.50,
            sleep_mid_limestone_to_marble_prob: 0.15,
            sleep_mid_sandstone_to_granite_prob: 0.25,
            sleep_outer_limestone_to_marble_prob: 0.30,
            sleep_aureole_metamorphism_enabled: 1,
            sleep_coal_maturation_enabled: 1,
            sleep_coal_to_graphite_prob: 0.70,
            sleep_coal_to_graphite_mid_prob: 0.35,
            sleep_graphite_to_diamond_prob: 0.15,
            sleep_silicification_enabled: 1,
            sleep_silicification_limestone_prob: 0.55,
            sleep_silicification_sandstone_prob: 0.15,
            sleep_silicification_water_radius_mult: 3,
            sleep_contact_limestone_to_garnet_prob: 0.65,
            sleep_mid_limestone_to_garnet_prob: 0.30,
            sleep_mid_limestone_to_diopside_prob: 0.65,
            sleep_recrystallization_prob: 0.70,
            sleep_contact_slate_to_hornfels_prob: 0.90,
            sleep_mid_slate_to_hornfels_prob: 0.60,
            sleep_outer_slate_to_hornfels_prob: 0.25,
            // Phase 3: Veins (29 fields)
            sleep_vein_enabled: 1,
            sleep_hypothermal_height: 25,
            sleep_mesothermal_height: 45,
            sleep_epithermal_height: 65,
            sleep_horizontal_spread: 20,
            sleep_veins_per_zone_min: 2,
            sleep_vein_climb_height_min: 6,
            sleep_vein_climb_height_max: 12,
            sleep_vein_wall_width_min: 2,
            sleep_vein_wall_width_max: 3,
            sleep_vein_rock_depth_min: 1,
            sleep_vein_rock_depth_max: 3,
            sleep_heat_direction_bias: 0.3,
            sleep_epithermal_rarity: 0.55,
            sleep_vein_crystal_growth_enabled: 1,
            sleep_vein_crystal_growth_prob: 0.30,
            sleep_vein_crystal_growth_max_per_chunk: 4,
            sleep_vein_calcite_infill_enabled: 1,
            sleep_vein_calcite_infill_prob: 0.15,
            sleep_vein_calcite_infill_max_per_chunk: 4,
            sleep_vein_flowstone_enabled: 1,
            sleep_vein_flowstone_max_per_chunk: 3,
            sleep_vein_growth_density_min: 0.3,
            sleep_vein_growth_density_max: 0.6,
            sleep_aperture_scaling_enabled: 1,
            sleep_host_rock_ore_enabled: 1,
            sleep_slate_pyrite_codeposit_prob: 0.25,
            sleep_slate_quartz_vein_prob: 0.30,
            sleep_wall_rock_alteration_prob: 0.18,
            // Phase 4: Deep Time (31 fields)
            sleep_max_enrichment_per_chunk: 400,
            sleep_enrichment_search_radius: 12,
            sleep_enrichment_enabled: 1,
            sleep_enrichment_cluster_min: 3,
            sleep_enrichment_cluster_max: 30,
            sleep_vein_thickening_enabled: 1,
            sleep_vein_thickening_max_per_chunk: 100,
            sleep_vein_thickening_water_radius: 40.0,
            sleep_vein_thickening_coat_depth: 1,
            sleep_vein_thickening_finger_interval: 5,
            sleep_vein_thickening_finger_length_min: 3,
            sleep_vein_thickening_finger_length_max: 5,
            sleep_vein_thickening_finger_taper: 0.7,
            sleep_mature_formations_enabled: 1,
            sleep_column_formation_prob: 0.05,
            // Nest fossilization
            sleep_nest_fossil_enabled: 1,
            sleep_nest_fossil_radius: 2,
            sleep_nest_fossil_pyrite_prob: 0.60,
            sleep_nest_fossil_opal_prob: 0.40,
            sleep_nest_fossil_buried_required: 0,
            sleep_nest_fossil_water_pyrite: 1,
            sleep_nest_fossil_water_opal: 1,
            // Corpse fossilization
            sleep_corpse_fossil_enabled: 1,
            sleep_corpse_fossil_radius: 1,
            sleep_corpse_fossil_pyrite_prob: 0.50,
            sleep_corpse_fossil_calcium_prob: 0.40,
            sleep_corpse_fossil_water_required: 1,
            sleep_corpse_fossil_min_cycles: 2,
            // Slate aquitard
            sleep_slate_aquitard_enabled: 1,
            sleep_slate_aquitard_factor: 0.05,
            sleep_slate_aquitard_concentration: 2.0,
            // Vein scaling + spikes + ore global scale
            sleep_min_vein_height: 3,
            sleep_water_volume_radius: 8,
            sleep_water_volume_max_cells: 50,
            sleep_water_volume_vein_mult: 1.0,
            sleep_water_volume_amount_mult: 1.0,
            sleep_lava_volume_radius: 8,
            sleep_lava_volume_max_cells: 30,
            sleep_lava_volume_vein_mult: 0.5,
            sleep_lava_volume_amount_mult: 0.5,
            sleep_spike_enabled: 1,
            sleep_spike_count_min: 4,
            sleep_spike_count_max: 10,
            sleep_spike_length_min: 2,
            sleep_spike_length_max: 5,
            sleep_spike_taper: 0.7,
            ore_global_scale: 1.0,
            // Aureole deposit detail
            sleep_aureole_vein_count: 8,
            sleep_aureole_vein_min: 6,
            sleep_aureole_vein_max: 20,
            sleep_garnet_compact_size: 8,
            sleep_diopside_compact_size: 8,
            sleep_garnet_pocket_count: 2,
            sleep_diopside_pocket_count: 1,
            sleep_aureole_vein_spread: 0.5,
            sleep_aureole_lava_max_cells: 10000,
            sleep_aureole_lava_deposit_mult: 1.0,
            sleep_aureole_lava_count_mult: 0.5,
            sleep_aureole_water_search_radius: 45,
            sleep_aureole_water_max_cells: 30,
            sleep_aureole_water_deposit_mult: 1.0,
            // Aureole vein shape
            sleep_aureole_wall_climbing: 1,
            sleep_aureole_weight_up: 3.0,
            sleep_aureole_weight_depth: 2.0,
            sleep_aureole_weight_lateral: 1.5,
            sleep_aureole_surface_ratio: 0.5,
            // Hydrothermal vein shape
            sleep_vein_spread: 0.5,
            sleep_vein_size_min: 8,
            sleep_vein_size_max: 30,
            sleep_vein_weight_up: 3.0,
            sleep_vein_weight_depth: 2.0,
            sleep_vein_weight_lateral: 1.5,
            sleep_vein_surface_ratio: 0.5,
            sleep_water_proximity_bias: 2.0,
            sleep_vein_min_connectivity: 1,
            sleep_aureole_min_connectivity: 1,
            sleep_vein_weight_down: 0.3,
            sleep_aureole_weight_down: 1.5,
            sleep_aureole_veins_per_n_cells: 1.0,
            sleep_aureole_garnet_per_n_cells: 0.5,
            sleep_aureole_diopside_per_n_cells: 0.3,
            sleep_aureole_cells_per_extra: 90,

            // Zone Config (defaults — zones disabled in test)
            zone_enabled: 0,
            zone_cathedral_chance: 0.15,
            zone_lake_chance: 0.12,
            zone_canyon_chance: 0.10,
            zone_lava_gallery_chance: 0.08,
            zone_bioluminescent_chance: 0.10,
            zone_terraces_chance: 0.08,
            zone_frozen_chance: 0.06,
            zone_cathedral_min_air: 2000,
            zone_lake_min_air: 1500,
            zone_canyon_min_air: 800,
            zone_lava_gallery_min_air: 600,
            zone_bioluminescent_min_air: 400,
            zone_terraces_min_air: 1000,
            zone_frozen_min_air: 600,
            zone_cathedral_dome_scale: 0.7,
            zone_cathedral_boulder_count_min: 3,
            zone_cathedral_boulder_count_max: 8,
            zone_cathedral_mega_stalagmite_chance: 0.4,
            zone_cathedral_flowstone_coverage: 0.3,
            zone_lake_depth: 4,
            zone_lake_beach_width: 3.0,
            zone_lake_island_min_radius: 2.0,
            zone_canyon_width_min: 3.0,
            zone_canyon_width_max: 6.0,
            zone_canyon_height_min: 12.0,
            zone_canyon_height_max: 25.0,
            zone_canyon_bridge_chance: 0.3,
            zone_lava_gallery_bench_spacing: 4.0,
            zone_lava_gallery_lavacicle_chance: 0.15,
            zone_bio_anchor_density: 0.1,
            zone_bio_max_anchors: 50,
            zone_terrace_tiers_min: 3,
            zone_terrace_tiers_max: 7,
            zone_terrace_step_height: 4.0,
            zone_terrace_rim_height: 1.5,
            zone_terrace_basin_depth: 2,
            zone_frozen_floor_depth: 2,
            zone_frozen_waterfall_count: 2,
            zone_frozen_ice_stalactite_chance: 0.3,
            zone_frozen_mega_chance: 0.03,
            blank_canvas: 0,
            // Basalt aureole (Amphibolite) deposits
            sleep_amphibolite_pyrite_pocket_count: 2,
            sleep_amphibolite_garnet_pocket_count: 1,
            sleep_amphibolite_pyrite_compact_size: 8,
            sleep_aureole_amphibolite_pyrite_per_n_cells: 0.4,
            sleep_aureole_amphibolite_garnet_per_n_cells: 0.2,
            // Hydrothermal water-boost v2
            sleep_aureole_water_phase1_weight: 1.0,
            sleep_aureole_water_phase2_weight: 0.25,
            sleep_aureole_water_network_max_hops: 50,
            sleep_aureole_water_to_lava_ratio: 1.2,
            sleep_aureole_water_phase1_max_floor: 50,
            sleep_aureole_water_count_mult: 1.0,
            // Mushroom decoration
            mushroom_enabled: 1,
            _mushroom_pad: [0; 3],
            mushroom_global_density: 0.04,
            mushroom_cluster_frequency: 0.05,
            mushroom_cluster_threshold: -0.15,
            mushroom_min_spacing_voxels: 1.5,
            mushroom_ghost_tower_routing_share: 0.06,
            mushroom_turkey_tail_enabled: 1,
            _mushroom_pad_tt: [0; 3],
            mushroom_turkey_tail_spawn_chance: 0.35,
            mushroom_turkey_tail_scale_min: 0.6,
            mushroom_turkey_tail_scale_max: 1.1,
            mushroom_foxfire_enabled: 1,
            _mushroom_pad_fx: [0; 3],
            mushroom_foxfire_spawn_chance: 0.25,
            mushroom_foxfire_scale_min: 0.5,
            mushroom_foxfire_scale_max: 1.0,
            mushroom_green_pepe_enabled: 1,
            _mushroom_pad_gp: [0; 3],
            mushroom_green_pepe_spawn_chance: 0.4,
            mushroom_green_pepe_scale_min: 0.7,
            mushroom_green_pepe_scale_max: 1.2,
            mushroom_ghost_tower_enabled: 1,
            _mushroom_pad_gt: [0; 3],
            mushroom_ghost_tower_spawn_chance: 0.5,
            mushroom_ghost_tower_scale_min: 1.5,
            mushroom_ghost_tower_scale_max: 3.0,
        }
    }

    #[test]
    fn engine_lifecycle() {
        unsafe {
            let cfg = test_config();
            let engine = voxel_create_engine(&cfg);
            assert!(!engine.is_null());
            voxel_destroy_engine(engine);
        }
    }

    #[test]
    fn generate_single_chunk_and_poll() {
        unsafe {
            let cfg = test_config();
            let engine = voxel_create_engine(&cfg);
            assert!(!engine.is_null());

            let chunk = FfiChunkCoord { x: 0, y: 0, z: 0 };
            let ok = voxel_request_generate(engine, chunk);
            assert_eq!(ok, 1);

            // Poll until we get a result (with timeout)
            let mut result_ptr = ptr::null_mut();
            for _ in 0..200 {
                result_ptr = voxel_poll_result(engine);
                if !result_ptr.is_null() {
                    break;
                }
                thread::sleep(Duration::from_millis(50));
            }
            assert!(!result_ptr.is_null(), "Should have received a result");

            // Poll results until we find the ChunkMesh (fluid meshes may arrive first)
            let mut found_chunk = false;
            let result = &*result_ptr;
            if result.result_type == FfiResultType::ChunkMesh {
                found_chunk = true;
            } else {
                voxel_free_result(result_ptr);
                // Keep polling for the ChunkMesh
                for _ in 0..200 {
                    result_ptr = voxel_poll_result(engine);
                    if !result_ptr.is_null() {
                        let r = &*result_ptr;
                        if r.result_type == FfiResultType::ChunkMesh {
                            found_chunk = true;
                            break;
                        }
                        voxel_free_result(result_ptr);
                    } else {
                        thread::sleep(Duration::from_millis(50));
                    }
                }
            }
            assert!(found_chunk, "Should have received a ChunkMesh result");

            let result = &*result_ptr;
            assert_eq!(result.chunk.x, 0);
            assert_eq!(result.chunk.y, 0);
            assert_eq!(result.chunk.z, 0);
            assert!(result.mesh.vertex_count > 0, "Mesh should have vertices");
            assert!(result.mesh.index_count > 0, "Mesh should have indices");

            voxel_free_result(result_ptr);
            voxel_destroy_engine(engine);
        }
    }

    #[test]
    fn cancel_discards_stale() {
        unsafe {
            let cfg = test_config();
            let engine = voxel_create_engine(&cfg);

            let chunk = FfiChunkCoord { x: 5, y: 5, z: 5 };
            voxel_request_generate(engine, chunk);
            // Immediately cancel
            voxel_cancel_chunk(engine, chunk);

            // Wait a bit, then poll - result should either be absent or have
            // a stale generation that was already in flight
            thread::sleep(Duration::from_millis(500));

            // Drain any results - if we get one, it should still be well-formed
            loop {
                let result = voxel_poll_result(engine);
                if result.is_null() {
                    break;
                }
                voxel_free_result(result);
            }

            voxel_destroy_engine(engine);
        }
    }

    #[test]
    fn destroy_under_load() {
        unsafe {
            let cfg = test_config();
            let engine = voxel_create_engine(&cfg);

            // Queue many chunks
            for x in 0..4 {
                for z in 0..4 {
                    voxel_request_generate(
                        engine,
                        FfiChunkCoord { x, y: 0, z },
                    );
                }
            }

            // Destroy immediately while workers are busy
            thread::sleep(Duration::from_millis(100));
            voxel_destroy_engine(engine);
            // Should not crash
        }
    }

    #[test]
    fn null_engine_safety() {
        unsafe {
            // All API functions should handle null gracefully
            voxel_destroy_engine(ptr::null_mut());
            assert_eq!(
                voxel_request_generate(ptr::null_mut(), FfiChunkCoord { x: 0, y: 0, z: 0 }),
                0
            );
            assert!(voxel_poll_result(ptr::null_mut()).is_null());
            let stats = voxel_get_stats(ptr::null_mut());
            assert_eq!(stats.chunks_loaded, 0);
        }
    }

    #[test]
    fn stats_reports_correctly() {
        unsafe {
            let cfg = test_config();
            let engine = voxel_create_engine(&cfg);

            let stats = voxel_get_stats(engine);
            assert_eq!(stats.chunks_loaded, 0);
            // Worker-thread count grows as background subsystems are added
            // (2 generate/mine + path-worker were the original 3; Block 1+
            // added poi-tracker, drift, predictor, etc., now ~8). The exact
            // number is an implementation detail that keeps changing — assert
            // the meaningful invariant instead: the core workers are spawned.
            assert!(
                stats.worker_threads_active >= 3,
                "expected at least 3 worker threads, got {}",
                stats.worker_threads_active
            );

            voxel_destroy_engine(engine);
        }
    }
