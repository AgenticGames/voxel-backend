//! Form-body parsing for the demo generate endpoint.
//!
//! Extracted verbatim from voxel-viewer's `serve_generate` so the native
//! HTTP server and the browser/WASM demo parse identical parameter sets.

use voxel_gen::config::GenerationConfig;
use voxel_sleep::config::SleepConfig;

/// Everything a generate request configures.
pub struct GenerateRequest {
    pub config: GenerationConfig,
    pub sleep_config: SleepConfig,
    pub chunks: (u32, u32, u32),
    pub closed: bool,
    pub seed: u64,
}

/// Parse an `application/x-www-form-urlencoded` generate body
/// (`seed=42&chunks_x=6&...`) into full generation + sleep configs.
pub fn parse_generate_form(body: &str) -> GenerateRequest {
    let mut seed = 42u64;
    let mut chunks_x = 3u32;
    let mut chunks_y = 3u32;
    let mut chunks_z = 1u32;
    let mut closed = false;

    let mut cavern_freq: Option<f64> = None;
    let mut cavern_threshold: Option<f64> = None;
    let mut detail_octaves: Option<u32> = None;
    let mut detail_persistence: Option<f64> = None;
    let mut warp_amplitude: Option<f64> = None;
    let mut worms_per_region: Option<f32> = None;
    let mut worm_radius_min: Option<f32> = None;
    let mut worm_radius_max: Option<f32> = None;
    let mut worm_step_length: Option<f32> = None;
    let mut worm_max_steps: Option<u32> = None;
    let mut worm_falloff_power: Option<f32> = None;
    let mut chunk_size: Option<usize> = None;
    let mut sandstone_depth: Option<f64> = None;
    let mut granite_depth: Option<f64> = None;
    let mut basalt_depth: Option<f64> = None;
    let mut slate_depth: Option<f64> = None;
    let mut iron_band_freq: Option<f64> = None;
    let mut iron_noise_freq: Option<f64> = None;
    let mut iron_perturbation: Option<f64> = None;
    let mut iron_threshold: Option<f64> = None;
    let mut copper_freq: Option<f64> = None;
    let mut copper_threshold: Option<f64> = None;
    let mut malachite_freq: Option<f64> = None;
    let mut malachite_threshold: Option<f64> = None;
    let mut kimberlite_pipe_freq: Option<f64> = None;
    let mut kimberlite_pipe_threshold: Option<f64> = None;
    let mut diamond_freq: Option<f64> = None;
    let mut diamond_threshold: Option<f64> = None;
    let mut sulfide_freq: Option<f64> = None;
    let mut sulfide_threshold: Option<f64> = None;
    let mut tin_threshold: Option<f64> = None;
    let mut pyrite_freq: Option<f64> = None;
    let mut pyrite_threshold: Option<f64> = None;
    let mut quartz_freq: Option<f64> = None;
    let mut quartz_threshold: Option<f64> = None;
    let mut gold_threshold: Option<f64> = None;
    let mut geode_freq: Option<f64> = None;
    let mut geode_center_threshold: Option<f64> = None;
    let mut geode_shell_thickness: Option<f64> = None;
    let mut geode_hollow_factor: Option<f32> = None;
    // Pool settings
    let mut pools_enabled: Option<bool> = None;
    let mut pool_placement_freq: Option<f64> = None;
    let mut pool_placement_threshold: Option<f64> = None;
    let mut pool_chance: Option<f32> = None;
    let mut pool_min_area: Option<usize> = None;
    let mut pool_max_radius: Option<usize> = None;
    let mut pool_basin_depth: Option<usize> = None;
    let mut pool_rim_height: Option<usize> = None;
    let mut pool_water_pct: Option<f32> = None;
    let mut pool_lava_pct: Option<f32> = None;
    let mut pool_empty_pct: Option<f32> = None;
    let mut pool_min_air_above: Option<usize> = None;
    let mut pool_max_cave_height: Option<usize> = None;
    let mut pool_min_floor_thickness: Option<usize> = None;
    let mut pool_min_ground_depth: Option<usize> = None;
    let mut pool_max_y_step: Option<usize> = None;
    let mut pool_footprint_y_tolerance: Option<usize> = None;
    // Formation settings
    let mut formations_enabled: Option<bool> = None;
    let mut form_placement_frequency: Option<f64> = None;
    let mut form_placement_threshold: Option<f64> = None;
    let mut form_stalactite_chance: Option<f32> = None;
    let mut form_stalagmite_chance: Option<f32> = None;
    let mut form_flowstone_chance: Option<f32> = None;
    let mut form_column_chance: Option<f32> = None;
    let mut form_column_max_gap: Option<usize> = None;
    let mut form_length_min: Option<f32> = None;
    let mut form_length_max: Option<f32> = None;
    let mut form_radius_min: Option<f32> = None;
    let mut form_radius_max: Option<f32> = None;
    let mut form_max_radius: Option<f32> = None;
    let mut form_column_radius_min: Option<f32> = None;
    let mut form_column_radius_max: Option<f32> = None;
    let mut form_flowstone_length_min: Option<f32> = None;
    let mut form_flowstone_length_max: Option<f32> = None;
    let mut form_flowstone_thickness: Option<f32> = None;
    let mut form_min_air_gap: Option<usize> = None;
    let mut form_min_clearance: Option<usize> = None;
    let mut form_smoothness: Option<f32> = None;
    // Mega-Column settings
    let mut form_mega_column_chance: Option<f32> = None;
    let mut form_mega_column_min_gap: Option<usize> = None;
    let mut form_mega_column_radius_min: Option<f32> = None;
    let mut form_mega_column_radius_max: Option<f32> = None;
    let mut form_mega_column_noise_strength: Option<f32> = None;
    let mut form_mega_column_ring_frequency: Option<f32> = None;
    // Drapery settings
    let mut form_drapery_chance: Option<f32> = None;
    let mut form_drapery_length_min: Option<f32> = None;
    let mut form_drapery_length_max: Option<f32> = None;
    let mut form_drapery_wave_frequency: Option<f32> = None;
    let mut form_drapery_wave_amplitude: Option<f32> = None;
    // Rimstone Dam settings
    let mut form_rimstone_chance: Option<f32> = None;
    let mut form_rimstone_dam_height_min: Option<f32> = None;
    let mut form_rimstone_dam_height_max: Option<f32> = None;
    let mut form_rimstone_pool_depth: Option<f32> = None;
    let mut form_rimstone_min_slope: Option<f32> = None;
    // Cave Shield settings
    let mut form_shield_chance: Option<f32> = None;
    let mut form_shield_radius_min: Option<f32> = None;
    let mut form_shield_radius_max: Option<f32> = None;
    let mut form_shield_max_tilt: Option<f32> = None;
    let mut form_shield_stalactite_chance: Option<f32> = None;
    // Cauldron settings
    let mut form_cauldron_chance: Option<f32> = None;
    let mut form_cauldron_radius_min: Option<f32> = None;
    let mut form_cauldron_radius_max: Option<f32> = None;
    let mut form_cauldron_depth: Option<f32> = None;
    let mut form_cauldron_lip_height: Option<f32> = None;
    let mut form_cauldron_rim_stal_min: Option<u32> = None;
    let mut form_cauldron_rim_stal_max: Option<u32> = None;
    let mut form_cauldron_rim_stal_scale: Option<f32> = None;
    let mut form_cauldron_floor_noise: Option<f32> = None;
    let mut form_cauldron_water_chance: Option<f32> = None;
    let mut form_cauldron_lava_chance: Option<f32> = None;
    // Stress settings
    let mut stress_gravity: Option<f32> = None;
    let mut stress_lateral: Option<f32> = None;
    let mut stress_vertical: Option<f32> = None;
    let mut stress_prop_radius: Option<u32> = None;
    let mut stress_max_collapse: Option<u32> = None;
    // Sleep collapse settings (strut survival per type).
    // 2026-05-26: removed Slate/Granite/Limestone (collapsed to Copper); added Mithril.
    // Legacy slate/granite/limestone params still accepted on the query string
    // and routed to Copper to keep bookmarks alive.
    let mut collapse_slate: Option<f32> = None;
    let mut collapse_granite: Option<f32> = None;
    let mut collapse_limestone: Option<f32> = None;
    let mut collapse_copper: Option<f32> = None;
    let mut collapse_iron: Option<f32> = None;
    let mut collapse_steel: Option<f32> = None;
    let mut collapse_crystal: Option<f32> = None;
    let mut collapse_mithril: Option<f32> = None;
    let mut collapse_stress_mult: Option<f32> = None;
    let mut collapse_max_cascade: Option<u32> = None;
    let mut collapse_rubble: Option<f32> = None;
    // Geological realism toggles
    let mut iron_sedimentary_only: Option<bool> = None;
    let mut iron_depth_fade: Option<bool> = None;
    let mut copper_supergene: Option<bool> = None;
    let mut copper_granite_contact: Option<bool> = None;
    let mut malachite_depth_bias: Option<bool> = None;
    let mut kimberlite_carrot_taper: Option<bool> = None;
    let mut diamond_depth_grade: Option<bool> = None;
    let mut sulfide_gossan_cap: Option<bool> = None;
    let mut sulfide_disseminated: Option<bool> = None;
    let mut pyrite_ore_halo: Option<bool> = None;
    let mut quartz_planar_veins: Option<bool> = None;
    let mut gold_bonanza: Option<bool> = None;
    let mut geode_volcanic_host: Option<bool> = None;
    let mut geode_depth_scaling: Option<bool> = None;
    // Coal
    let mut coal_freq: Option<f64> = None;
    let mut coal_threshold: Option<f64> = None;
    let mut coal_depth_min: Option<f64> = None;
    let mut coal_depth_max: Option<f64> = None;
    let mut coal_sedimentary_host: Option<bool> = None;
    let mut coal_shallow_ceiling: Option<bool> = None;
    let mut coal_depth_enrichment: Option<bool> = None;
    let mut ore_detail_multiplier: Option<u32> = None;
    let mut ore_protrusion: Option<f32> = None;

    for pair in body.split('&') {
        let mut kv = pair.splitn(2, '=');
        let key = kv.next().unwrap_or("");
        let val = kv.next().unwrap_or("");
        match key {
            "seed" => { seed = val.parse().unwrap_or(42); }
            "chunks_x" => { chunks_x = val.parse().unwrap_or(3); }
            "chunks_y" => { chunks_y = val.parse().unwrap_or(3); }
            "chunks_z" => { chunks_z = val.parse().unwrap_or(1); }
            "closed" => { closed = val == "1" || val == "true"; }
            "cavern_freq" => { cavern_freq = val.parse().ok(); }
            "cavern_threshold" => { cavern_threshold = val.parse().ok(); }
            "detail_octaves" => { detail_octaves = val.parse().ok(); }
            "detail_persistence" => { detail_persistence = val.parse().ok(); }
            "warp_amplitude" => { warp_amplitude = val.parse().ok(); }
            "worms_per_region" => { worms_per_region = val.parse().ok(); }
            "worm_radius_min" => { worm_radius_min = val.parse().ok(); }
            "worm_radius_max" => { worm_radius_max = val.parse().ok(); }
            "worm_step_length" => { worm_step_length = val.parse().ok(); }
            "worm_max_steps" => { worm_max_steps = val.parse().ok(); }
            "worm_falloff_power" => { worm_falloff_power = val.parse().ok(); }
            "chunk_size" => { chunk_size = val.parse().ok(); }
            "sandstone_depth" => { sandstone_depth = val.parse().ok(); }
            "granite_depth" => { granite_depth = val.parse().ok(); }
            "basalt_depth" => { basalt_depth = val.parse().ok(); }
            "slate_depth" => { slate_depth = val.parse().ok(); }
            "iron_band_freq" => { iron_band_freq = val.parse().ok(); }
            "iron_noise_freq" => { iron_noise_freq = val.parse().ok(); }
            "iron_perturbation" => { iron_perturbation = val.parse().ok(); }
            "iron_threshold" => { iron_threshold = val.parse().ok(); }
            "copper_freq" => { copper_freq = val.parse().ok(); }
            "copper_threshold" => { copper_threshold = val.parse().ok(); }
            "malachite_freq" => { malachite_freq = val.parse().ok(); }
            "malachite_threshold" => { malachite_threshold = val.parse().ok(); }
            "kimberlite_pipe_freq" => { kimberlite_pipe_freq = val.parse().ok(); }
            "kimberlite_pipe_threshold" => { kimberlite_pipe_threshold = val.parse().ok(); }
            "diamond_freq" => { diamond_freq = val.parse().ok(); }
            "diamond_threshold" => { diamond_threshold = val.parse().ok(); }
            "sulfide_freq" => { sulfide_freq = val.parse().ok(); }
            "sulfide_threshold" => { sulfide_threshold = val.parse().ok(); }
            "tin_threshold" => { tin_threshold = val.parse().ok(); }
            "pyrite_freq" => { pyrite_freq = val.parse().ok(); }
            "pyrite_threshold" => { pyrite_threshold = val.parse().ok(); }
            "quartz_freq" => { quartz_freq = val.parse().ok(); }
            "quartz_threshold" => { quartz_threshold = val.parse().ok(); }
            "gold_threshold" => { gold_threshold = val.parse().ok(); }
            "geode_freq" => { geode_freq = val.parse().ok(); }
            "geode_center_threshold" => { geode_center_threshold = val.parse().ok(); }
            "geode_shell_thickness" => { geode_shell_thickness = val.parse().ok(); }
            "geode_hollow_factor" => { geode_hollow_factor = val.parse().ok(); }
            // Pool settings
            "pools_enabled" => { pools_enabled = Some(val == "1" || val == "true"); }
            "pool_placement_freq" => { pool_placement_freq = val.parse().ok(); }
            "pool_placement_threshold" => { pool_placement_threshold = val.parse().ok(); }
            "pool_chance" => { pool_chance = val.parse().ok(); }
            "pool_min_area" => { pool_min_area = val.parse().ok(); }
            "pool_max_radius" => { pool_max_radius = val.parse().ok(); }
            "pool_basin_depth" => { pool_basin_depth = val.parse().ok(); }
            "pool_rim_height" => { pool_rim_height = val.parse().ok(); }
            "pool_water_pct" => { pool_water_pct = val.parse().ok(); }
            "pool_lava_pct" => { pool_lava_pct = val.parse().ok(); }
            "pool_empty_pct" => { pool_empty_pct = val.parse().ok(); }
            "pool_min_air_above" => { pool_min_air_above = val.parse().ok(); }
            "pool_max_cave_height" => { pool_max_cave_height = val.parse().ok(); }
            "pool_min_floor_thickness" => { pool_min_floor_thickness = val.parse().ok(); }
            "pool_min_ground_depth" => { pool_min_ground_depth = val.parse().ok(); }
            "pool_max_y_step" => { pool_max_y_step = val.parse().ok(); }
            "pool_footprint_y_tolerance" => { pool_footprint_y_tolerance = val.parse().ok(); }
            // Formation settings
            "formations_enabled" => { formations_enabled = Some(val == "1" || val == "true"); }
            "form_placement_frequency" => { form_placement_frequency = val.parse().ok(); }
            "form_placement_threshold" => { form_placement_threshold = val.parse().ok(); }
            "form_stalactite_chance" => { form_stalactite_chance = val.parse().ok(); }
            "form_stalagmite_chance" => { form_stalagmite_chance = val.parse().ok(); }
            "form_flowstone_chance" => { form_flowstone_chance = val.parse().ok(); }
            "form_column_chance" => { form_column_chance = val.parse().ok(); }
            "form_column_max_gap" => { form_column_max_gap = val.parse().ok(); }
            "form_length_min" => { form_length_min = val.parse().ok(); }
            "form_length_max" => { form_length_max = val.parse().ok(); }
            "form_radius_min" => { form_radius_min = val.parse().ok(); }
            "form_radius_max" => { form_radius_max = val.parse().ok(); }
            "form_max_radius" => { form_max_radius = val.parse().ok(); }
            "form_column_radius_min" => { form_column_radius_min = val.parse().ok(); }
            "form_column_radius_max" => { form_column_radius_max = val.parse().ok(); }
            "form_flowstone_length_min" => { form_flowstone_length_min = val.parse().ok(); }
            "form_flowstone_length_max" => { form_flowstone_length_max = val.parse().ok(); }
            "form_flowstone_thickness" => { form_flowstone_thickness = val.parse().ok(); }
            "form_min_air_gap" => { form_min_air_gap = val.parse().ok(); }
            "form_min_clearance" => { form_min_clearance = val.parse().ok(); }
            "form_smoothness" => { form_smoothness = val.parse().ok(); }
            // Mega-Column settings
            "form_mega_column_chance" => { form_mega_column_chance = val.parse().ok(); }
            "form_mega_column_min_gap" => { form_mega_column_min_gap = val.parse().ok(); }
            "form_mega_column_radius_min" => { form_mega_column_radius_min = val.parse().ok(); }
            "form_mega_column_radius_max" => { form_mega_column_radius_max = val.parse().ok(); }
            "form_mega_column_noise_strength" => { form_mega_column_noise_strength = val.parse().ok(); }
            "form_mega_column_ring_frequency" => { form_mega_column_ring_frequency = val.parse().ok(); }
            // Drapery settings
            "form_drapery_chance" => { form_drapery_chance = val.parse().ok(); }
            "form_drapery_length_min" => { form_drapery_length_min = val.parse().ok(); }
            "form_drapery_length_max" => { form_drapery_length_max = val.parse().ok(); }
            "form_drapery_wave_frequency" => { form_drapery_wave_frequency = val.parse().ok(); }
            "form_drapery_wave_amplitude" => { form_drapery_wave_amplitude = val.parse().ok(); }
            // Rimstone Dam settings
            "form_rimstone_chance" => { form_rimstone_chance = val.parse().ok(); }
            "form_rimstone_dam_height_min" => { form_rimstone_dam_height_min = val.parse().ok(); }
            "form_rimstone_dam_height_max" => { form_rimstone_dam_height_max = val.parse().ok(); }
            "form_rimstone_pool_depth" => { form_rimstone_pool_depth = val.parse().ok(); }
            "form_rimstone_min_slope" => { form_rimstone_min_slope = val.parse().ok(); }
            // Cave Shield settings
            "form_shield_chance" => { form_shield_chance = val.parse().ok(); }
            "form_shield_radius_min" => { form_shield_radius_min = val.parse().ok(); }
            "form_shield_radius_max" => { form_shield_radius_max = val.parse().ok(); }
            "form_shield_max_tilt" => { form_shield_max_tilt = val.parse().ok(); }
            "form_shield_stalactite_chance" => { form_shield_stalactite_chance = val.parse().ok(); }
            // Cauldron settings
            "form_cauldron_chance" => { form_cauldron_chance = val.parse().ok(); }
            "form_cauldron_radius_min" => { form_cauldron_radius_min = val.parse().ok(); }
            "form_cauldron_radius_max" => { form_cauldron_radius_max = val.parse().ok(); }
            "form_cauldron_depth" => { form_cauldron_depth = val.parse().ok(); }
            "form_cauldron_lip_height" => { form_cauldron_lip_height = val.parse().ok(); }
            "form_cauldron_rim_stal_min" => { form_cauldron_rim_stal_min = val.parse().ok(); }
            "form_cauldron_rim_stal_max" => { form_cauldron_rim_stal_max = val.parse().ok(); }
            "form_cauldron_rim_stal_scale" => { form_cauldron_rim_stal_scale = val.parse().ok(); }
            "form_cauldron_floor_noise" => { form_cauldron_floor_noise = val.parse().ok(); }
            "form_cauldron_water_chance" => { form_cauldron_water_chance = val.parse().ok(); }
            "form_cauldron_lava_chance" => { form_cauldron_lava_chance = val.parse().ok(); }
            // Stress settings
            "stress_gravity" => { stress_gravity = val.parse().ok(); }
            "stress_lateral" => { stress_lateral = val.parse().ok(); }
            "stress_vertical" => { stress_vertical = val.parse().ok(); }
            "stress_prop_radius" => { stress_prop_radius = val.parse().ok(); }
            "stress_max_collapse" => { stress_max_collapse = val.parse().ok(); }
            // Sleep collapse settings
            "collapse_slate" => { collapse_slate = val.parse().ok(); }
            "collapse_granite" => { collapse_granite = val.parse().ok(); }
            "collapse_limestone" => { collapse_limestone = val.parse().ok(); }
            "collapse_copper" => { collapse_copper = val.parse().ok(); }
            "collapse_iron" => { collapse_iron = val.parse().ok(); }
            "collapse_steel" => { collapse_steel = val.parse().ok(); }
            "collapse_crystal" => { collapse_crystal = val.parse().ok(); }
            "collapse_mithril" => { collapse_mithril = val.parse().ok(); }
            "collapse_stress_mult" => { collapse_stress_mult = val.parse().ok(); }
            "collapse_max_cascade" => { collapse_max_cascade = val.parse().ok(); }
            "collapse_rubble" => { collapse_rubble = val.parse().ok(); }
            // Geological realism toggles
            "iron_sedimentary_only" => { iron_sedimentary_only = Some(val == "1" || val == "true"); }
            "iron_depth_fade" => { iron_depth_fade = Some(val == "1" || val == "true"); }
            "copper_supergene" => { copper_supergene = Some(val == "1" || val == "true"); }
            "copper_granite_contact" => { copper_granite_contact = Some(val == "1" || val == "true"); }
            "malachite_depth_bias" => { malachite_depth_bias = Some(val == "1" || val == "true"); }
            "kimberlite_carrot_taper" => { kimberlite_carrot_taper = Some(val == "1" || val == "true"); }
            "diamond_depth_grade" => { diamond_depth_grade = Some(val == "1" || val == "true"); }
            "sulfide_gossan_cap" => { sulfide_gossan_cap = Some(val == "1" || val == "true"); }
            "sulfide_disseminated" => { sulfide_disseminated = Some(val == "1" || val == "true"); }
            "pyrite_ore_halo" => { pyrite_ore_halo = Some(val == "1" || val == "true"); }
            "quartz_planar_veins" => { quartz_planar_veins = Some(val == "1" || val == "true"); }
            "gold_bonanza" => { gold_bonanza = Some(val == "1" || val == "true"); }
            "geode_volcanic_host" => { geode_volcanic_host = Some(val == "1" || val == "true"); }
            "geode_depth_scaling" => { geode_depth_scaling = Some(val == "1" || val == "true"); }
            // Coal
            "coal_freq" => { coal_freq = val.parse().ok(); }
            "coal_threshold" => { coal_threshold = val.parse().ok(); }
            "coal_depth_min" => { coal_depth_min = val.parse().ok(); }
            "coal_depth_max" => { coal_depth_max = val.parse().ok(); }
            "coal_sedimentary_host" => { coal_sedimentary_host = Some(val == "1" || val == "true"); }
            "coal_shallow_ceiling" => { coal_shallow_ceiling = Some(val == "1" || val == "true"); }
            "coal_depth_enrichment" => { coal_depth_enrichment = Some(val == "1" || val == "true"); }
            "ore_detail_multiplier" => { ore_detail_multiplier = val.parse().ok(); }
            "ore_protrusion" => { ore_protrusion = val.parse().ok(); }
            _ => {}
        }
    }

    let chunks_x = chunks_x.min(8);
    let chunks_y = chunks_y.min(8);
    let chunks_z = chunks_z.min(8);

    let mut config = GenerationConfig {
        seed,
        ..Default::default()
    };
    if let Some(v) = chunk_size { config.chunk_size = v.clamp(4, 64); }
    if let Some(v) = cavern_freq { config.noise.cavern_frequency = v; }
    if let Some(v) = cavern_threshold { config.noise.cavern_threshold = v; }
    if let Some(v) = detail_octaves { config.noise.detail_octaves = v; }
    if let Some(v) = detail_persistence { config.noise.detail_persistence = v; }
    if let Some(v) = warp_amplitude { config.noise.warp_amplitude = v; }
    if let Some(v) = worms_per_region { config.worm.worms_per_region = v; }
    if let Some(v) = worm_radius_min { config.worm.radius_min = v; }
    if let Some(v) = worm_radius_max { config.worm.radius_max = v; }
    if let Some(v) = worm_step_length { config.worm.step_length = v; }
    if let Some(v) = worm_max_steps { config.worm.max_steps = v; }
    if let Some(v) = worm_falloff_power { config.worm.falloff_power = v; }
    // Host rock
    if let Some(v) = sandstone_depth { config.ore.host_rock.sandstone_depth = v; }
    if let Some(v) = granite_depth { config.ore.host_rock.granite_depth = v; }
    if let Some(v) = basalt_depth { config.ore.host_rock.basalt_depth = v; }
    if let Some(v) = slate_depth { config.ore.host_rock.slate_depth = v; }
    // Banded iron
    if let Some(v) = iron_band_freq { config.ore.iron.band_frequency = v; }
    if let Some(v) = iron_noise_freq { config.ore.iron.noise_frequency = v; }
    if let Some(v) = iron_perturbation { config.ore.iron.noise_perturbation = v; }
    if let Some(v) = iron_threshold { config.ore.iron.threshold = v; }
    // Copper
    if let Some(v) = copper_freq { config.ore.copper.frequency = v; }
    if let Some(v) = copper_threshold { config.ore.copper.threshold = v; }
    // Malachite
    if let Some(v) = malachite_freq { config.ore.malachite.frequency = v; }
    if let Some(v) = malachite_threshold { config.ore.malachite.threshold = v; }
    // Kimberlite
    if let Some(v) = kimberlite_pipe_freq { config.ore.kimberlite.pipe_frequency_2d = v; }
    if let Some(v) = kimberlite_pipe_threshold { config.ore.kimberlite.pipe_threshold = v; }
    if let Some(v) = diamond_freq { config.ore.kimberlite.diamond_frequency = v; }
    if let Some(v) = diamond_threshold { config.ore.kimberlite.diamond_threshold = v; }
    // Sulfide
    if let Some(v) = sulfide_freq { config.ore.sulfide.frequency = v; }
    if let Some(v) = sulfide_threshold { config.ore.sulfide.threshold = v; }
    if let Some(v) = tin_threshold { config.ore.sulfide.tin_threshold = v; }
    // Pyrite
    if let Some(v) = pyrite_freq { config.ore.pyrite.frequency = v; }
    if let Some(v) = pyrite_threshold { config.ore.pyrite.threshold = v; }
    // Quartz
    if let Some(v) = quartz_freq { config.ore.quartz.frequency = v; config.ore.gold.frequency = v; }
    if let Some(v) = quartz_threshold { config.ore.quartz.threshold = v; }
    // Gold
    if let Some(v) = gold_threshold { config.ore.gold.threshold = v; }
    // Geode
    if let Some(v) = geode_freq { config.ore.geode.frequency = v; }
    if let Some(v) = geode_center_threshold { config.ore.geode.center_threshold = v; }
    if let Some(v) = geode_shell_thickness { config.ore.geode.shell_thickness = v; }
    if let Some(v) = geode_hollow_factor { config.ore.geode.hollow_factor = v; }
    // Geological realism toggles
    if let Some(v) = iron_sedimentary_only { config.ore.iron_sedimentary_only = v; }
    if let Some(v) = iron_depth_fade { config.ore.iron_depth_fade = v; }
    if let Some(v) = copper_supergene { config.ore.copper_supergene = v; }
    if let Some(v) = copper_granite_contact { config.ore.copper_granite_contact = v; }
    if let Some(v) = malachite_depth_bias { config.ore.malachite_depth_bias = v; }
    if let Some(v) = kimberlite_carrot_taper { config.ore.kimberlite_carrot_taper = v; }
    if let Some(v) = diamond_depth_grade { config.ore.diamond_depth_grade = v; }
    if let Some(v) = sulfide_gossan_cap { config.ore.sulfide_gossan_cap = v; }
    if let Some(v) = sulfide_disseminated { config.ore.sulfide_disseminated = v; }
    if let Some(v) = pyrite_ore_halo { config.ore.pyrite_ore_halo = v; }
    if let Some(v) = quartz_planar_veins { config.ore.quartz_planar_veins = v; }
    if let Some(v) = gold_bonanza { config.ore.gold_bonanza = v; }
    if let Some(v) = geode_volcanic_host { config.ore.geode_volcanic_host = v; }
    if let Some(v) = geode_depth_scaling { config.ore.geode_depth_scaling = v; }
    // Coal
    if let Some(v) = coal_freq { config.ore.coal.frequency = v; }
    if let Some(v) = coal_threshold { config.ore.coal.threshold = v; }
    if let Some(v) = coal_depth_min { config.ore.coal.depth_min = v; }
    if let Some(v) = coal_depth_max { config.ore.coal.depth_max = v; }
    if let Some(v) = coal_sedimentary_host { config.ore.coal_sedimentary_host = v; }
    if let Some(v) = coal_shallow_ceiling { config.ore.coal_shallow_ceiling = v; }
    if let Some(v) = coal_depth_enrichment { config.ore.coal_depth_enrichment = v; }
    if let Some(v) = ore_detail_multiplier { config.ore_detail_multiplier = v.max(1).min(4); }
    if let Some(v) = ore_protrusion { config.ore_protrusion = v.max(0.0).min(0.5); }
    // Pool settings
    if let Some(v) = pools_enabled { config.pools.enabled = v; }
    if let Some(v) = pool_placement_freq { config.pools.placement_frequency = v; }
    if let Some(v) = pool_placement_threshold { config.pools.placement_threshold = v; }
    if let Some(v) = pool_chance { config.pools.pool_chance = v; }
    if let Some(v) = pool_min_area { config.pools.min_area = v; }
    if let Some(v) = pool_max_radius { config.pools.max_radius = v; }
    if let Some(v) = pool_basin_depth { config.pools.basin_depth = v; }
    if let Some(v) = pool_rim_height { config.pools.rim_height = v; }
    if let Some(v) = pool_water_pct { config.pools.water_pct = v; }
    if let Some(v) = pool_lava_pct { config.pools.lava_pct = v; }
    if let Some(v) = pool_empty_pct { config.pools.empty_pct = v; }
    if let Some(v) = pool_min_air_above { config.pools.min_air_above = v; }
    if let Some(v) = pool_max_cave_height { config.pools.max_cave_height = v; }
    if let Some(v) = pool_min_floor_thickness { config.pools.min_floor_thickness = v; }
    if let Some(v) = pool_min_ground_depth { config.pools.min_ground_depth = v; }
    if let Some(v) = pool_max_y_step { config.pools.max_y_step = v; }
    if let Some(v) = pool_footprint_y_tolerance { config.pools.footprint_y_tolerance = v; }
    // Formation settings
    if let Some(v) = formations_enabled { config.formations.enabled = v; }
    if let Some(v) = form_placement_frequency { config.formations.placement_frequency = v; }
    if let Some(v) = form_placement_threshold { config.formations.placement_threshold = v; }
    if let Some(v) = form_stalactite_chance { config.formations.stalactite_chance = v; }
    if let Some(v) = form_stalagmite_chance { config.formations.stalagmite_chance = v; }
    if let Some(v) = form_flowstone_chance { config.formations.flowstone_chance = v; }
    if let Some(v) = form_column_chance { config.formations.column_chance = v; }
    if let Some(v) = form_column_max_gap { config.formations.column_max_gap = v; }
    if let Some(v) = form_length_min { config.formations.length_min = v; }
    if let Some(v) = form_length_max { config.formations.length_max = v; }
    if let Some(v) = form_radius_min { config.formations.radius_min = v; }
    if let Some(v) = form_radius_max { config.formations.radius_max = v; }
    if let Some(v) = form_max_radius { config.formations.max_radius = v; }
    if let Some(v) = form_column_radius_min { config.formations.column_radius_min = v; }
    if let Some(v) = form_column_radius_max { config.formations.column_radius_max = v; }
    if let Some(v) = form_flowstone_length_min { config.formations.flowstone_length_min = v; }
    if let Some(v) = form_flowstone_length_max { config.formations.flowstone_length_max = v; }
    if let Some(v) = form_flowstone_thickness { config.formations.flowstone_thickness = v; }
    if let Some(v) = form_min_air_gap { config.formations.min_air_gap = v; }
    if let Some(v) = form_min_clearance { config.formations.min_clearance = v; }
    if let Some(v) = form_smoothness { config.formations.smoothness = v; }
    // Mega-Column settings
    if let Some(v) = form_mega_column_chance { config.formations.mega_column_chance = v; }
    if let Some(v) = form_mega_column_min_gap { config.formations.mega_column_min_gap = v; }
    if let Some(v) = form_mega_column_radius_min { config.formations.mega_column_radius_min = v; }
    if let Some(v) = form_mega_column_radius_max { config.formations.mega_column_radius_max = v; }
    if let Some(v) = form_mega_column_noise_strength { config.formations.mega_column_noise_strength = v; }
    if let Some(v) = form_mega_column_ring_frequency { config.formations.mega_column_ring_frequency = v; }
    // Drapery settings
    if let Some(v) = form_drapery_chance { config.formations.drapery_chance = v; }
    if let Some(v) = form_drapery_length_min { config.formations.drapery_length_min = v; }
    if let Some(v) = form_drapery_length_max { config.formations.drapery_length_max = v; }
    if let Some(v) = form_drapery_wave_frequency { config.formations.drapery_wave_frequency = v; }
    if let Some(v) = form_drapery_wave_amplitude { config.formations.drapery_wave_amplitude = v; }
    // Rimstone Dam settings
    if let Some(v) = form_rimstone_chance { config.formations.rimstone_chance = v; }
    if let Some(v) = form_rimstone_dam_height_min { config.formations.rimstone_dam_height_min = v; }
    if let Some(v) = form_rimstone_dam_height_max { config.formations.rimstone_dam_height_max = v; }
    if let Some(v) = form_rimstone_pool_depth { config.formations.rimstone_pool_depth = v; }
    if let Some(v) = form_rimstone_min_slope { config.formations.rimstone_min_slope = v; }
    // Cave Shield settings
    if let Some(v) = form_shield_chance { config.formations.shield_chance = v; }
    if let Some(v) = form_shield_radius_min { config.formations.shield_radius_min = v; }
    if let Some(v) = form_shield_radius_max { config.formations.shield_radius_max = v; }
    if let Some(v) = form_shield_max_tilt { config.formations.shield_max_tilt = v; }
    if let Some(v) = form_shield_stalactite_chance { config.formations.shield_stalactite_chance = v; }
    // Cauldron settings
    if let Some(v) = form_cauldron_chance { config.formations.cauldron_chance = v; }
    if let Some(v) = form_cauldron_radius_min { config.formations.cauldron_radius_min = v; }
    if let Some(v) = form_cauldron_radius_max { config.formations.cauldron_radius_max = v; }
    if let Some(v) = form_cauldron_depth { config.formations.cauldron_depth = v; }
    if let Some(v) = form_cauldron_lip_height { config.formations.cauldron_lip_height = v; }
    if let Some(v) = form_cauldron_rim_stal_min { config.formations.cauldron_rim_stalagmite_count_min = v; }
    if let Some(v) = form_cauldron_rim_stal_max { config.formations.cauldron_rim_stalagmite_count_max = v; }
    if let Some(v) = form_cauldron_rim_stal_scale { config.formations.cauldron_rim_stalagmite_scale = v; }
    if let Some(v) = form_cauldron_floor_noise { config.formations.cauldron_floor_noise = v; }
    if let Some(v) = form_cauldron_water_chance { config.formations.cauldron_water_chance = v; }
    if let Some(v) = form_cauldron_lava_chance { config.formations.cauldron_lava_chance = v; }
    // Disable cavern zones in the web viewer — they dominate small previews
    // and obscure the ore veins / formations that are the real showcase.
    config.zones.enabled = false;

    // Build sleep config from UI overrides (stress settings embedded in sleep config)
    let mut sleep_cfg = SleepConfig::default();
    // Enable aureole + veins for testing, disable reaction/deeptime/accumulation
    sleep_cfg.phase1_enabled = false;  // reaction OFF
    sleep_cfg.phase2_enabled = true;   // aureole ON
    sleep_cfg.phase3_enabled = true;   // veins ON
    sleep_cfg.phase4_enabled = false;  // deeptime OFF
    sleep_cfg.accumulation_enabled = false; // accumulation OFF
    // Stress tuning
    if let Some(v) = stress_gravity { sleep_cfg.stress.gravity_weight = v; }
    if let Some(v) = stress_lateral { sleep_cfg.stress.lateral_support_factor = v; }
    if let Some(v) = stress_vertical { sleep_cfg.stress.vertical_support_factor = v; }
    if let Some(v) = stress_prop_radius { sleep_cfg.stress.propagation_radius = v; }
    if let Some(v) = stress_max_collapse { sleep_cfg.stress.max_collapse_volume = v; }
    // Sleep collapse — strut lineup overhauled 2026-05-26.
    // Old (Slate/Granite/Limestone) removed; new lineup is Copper..Mithril.
    // Legacy query params for the old stones map onto Copper (T1) so older
    // dashboard bookmarks don't 500.
    let copper_legacy_or_new = collapse_copper.or(collapse_slate).or(collapse_granite).or(collapse_limestone);
    if let Some(v) = copper_legacy_or_new { sleep_cfg.collapse.strut_survival[1] = v; }
    if let Some(v) = collapse_iron { sleep_cfg.collapse.strut_survival[2] = v; }
    if let Some(v) = collapse_steel { sleep_cfg.collapse.strut_survival[3] = v; }
    if let Some(v) = collapse_crystal { sleep_cfg.collapse.strut_survival[4] = v; }
    if let Some(v) = collapse_mithril { sleep_cfg.collapse.strut_survival[5] = v; }
    if let Some(v) = collapse_stress_mult { sleep_cfg.collapse.stress_multiplier = v; }
    if let Some(v) = collapse_max_cascade { sleep_cfg.collapse.max_cascade_iterations = v; }
    if let Some(v) = collapse_rubble { sleep_cfg.collapse.rubble_fill_ratio = v; }

    GenerateRequest {
        config,
        sleep_config: sleep_cfg,
        chunks: (chunks_x, chunks_y, chunks_z),
        closed,
        seed,
    }
}
