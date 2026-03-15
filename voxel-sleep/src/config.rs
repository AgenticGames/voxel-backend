use serde::{Deserialize, Serialize};
use voxel_core::stress::StressConfig;

fn default_true() -> bool { true }
fn default_one() -> f32 { 1.0 }
fn default_accumulation_iterations() -> u32 { 3 }
fn default_convergence_radius() -> f32 { 70.0 }
fn default_hypothermal_height() -> u32 { 25 }
fn default_mesothermal_height() -> u32 { 45 }
fn default_epithermal_height() -> u32 { 65 }
fn default_horizontal_spread() -> u32 { 20 }
fn default_veins_per_zone_min() -> u32 { 2 }
fn default_veins_per_zone_max() -> u32 { 4 }
fn default_vein_climb_height_min() -> u32 { 6 }
fn default_vein_climb_height_max() -> u32 { 12 }
fn default_vein_wall_width_min() -> u32 { 2 }
fn default_vein_wall_width_max() -> u32 { 3 }
fn default_vein_rock_depth_min() -> u32 { 1 }
fn default_vein_rock_depth_max() -> u32 { 3 }
fn default_heat_direction_bias() -> f32 { 0.3 }
fn default_convergence_spacing() -> u32 { 25 }
fn default_enrichment_cluster_min() -> u32 { 2 }
fn default_enrichment_cluster_max() -> u32 { 6 }
fn default_vein_thickening_water_radius() -> f32 { 40.0 }
fn default_vein_thickening_coat_depth() -> u32 { 1 }
fn default_vein_thickening_finger_interval() -> u32 { 5 }
fn default_vein_thickening_finger_length_min() -> u32 { 3 }
fn default_vein_thickening_finger_length_max() -> u32 { 5 }
fn default_vein_thickening_finger_taper() -> f32 { 0.7 }
fn default_water_boost_max() -> f32 { 0.6 }
fn default_water_search_mult() -> f32 { 2.0 }
fn default_large_vein() -> u32 { 15 }
fn default_small_vein() -> u32 { 6 }
fn default_min_zone() -> u32 { 5 }
fn default_garnet_pocket() -> u32 { 4 }
fn default_diopside_pocket() -> u32 { 4 }
fn default_max_aureole_radius() -> f32 { 10.0 }

/// Top-level sleep configuration — 4-phase geological time simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepConfig {
    /// Phase 1: The Reaction (acid dissolution, oxidation)
    pub reaction: ReactionConfig,
    /// Phase 2: The Aureole (contact metamorphism, water erosion)
    pub aureole: AureoleConfig,
    /// Phase 3: The Veins (hydrothermal ore deposition, formation growth)
    pub veins: VeinConfig,
    /// Phase 4: The Deep Time (enrichment, thickening, formations, collapse)
    pub deeptime: DeepTimeConfig,
    /// Ambient groundwater model (depth + porosity + drip zones)
    pub groundwater: GroundwaterConfig,
    /// Shared stress config
    pub stress: StressConfig,
    /// Chunk radius from player to process
    pub chunk_radius: u32,
    /// Per-phase enables
    pub phase1_enabled: bool,
    pub phase2_enabled: bool,
    pub phase3_enabled: bool,
    pub phase4_enabled: bool,
    /// Time budget (unused currently, reserved for future)
    pub time_budget_ms: u32,
    /// Sleep cycle number (for deterministic RNG seeding)
    pub sleep_count: u32,
    /// Enable accumulation pass after Phase 4 (re-runs Phase 1-3 with scaled params)
    #[serde(default = "default_true")]
    pub accumulation_enabled: bool,
    /// Number of accumulation iterations (each represents a fraction of remaining time)
    #[serde(default = "default_accumulation_iterations")]
    pub accumulation_iterations: u32,
    /// Convert all lava fluid cells to solid basalt after sleep cycle
    #[serde(default = "default_true")]
    pub lava_solidification_enabled: bool,
    /// Spider nest world positions (set by FFI before sleep starts)
    #[serde(skip)]
    pub nest_positions: Vec<(i32, i32, i32)>,
    /// Spider corpse world positions (set by FFI before sleep starts)
    #[serde(skip)]
    pub corpse_positions: Vec<(i32, i32, i32)>,

    // --- Legacy fields (kept for FFI backward compat during transition) ---
    #[serde(skip)]
    pub metamorphism: MetamorphismConfig,
    #[serde(skip)]
    pub minerals: MineralConfig,
    #[serde(skip)]
    pub collapse: CollapseConfig,
    #[serde(skip)]
    pub metamorphism_enabled: bool,
    #[serde(skip)]
    pub minerals_enabled: bool,
    #[serde(skip)]
    pub collapse_enabled: bool,
}

impl Default for SleepConfig {
    fn default() -> Self {
        Self {
            reaction: ReactionConfig::default(),
            aureole: AureoleConfig::default(),
            veins: VeinConfig::default(),
            deeptime: DeepTimeConfig::default(),
            groundwater: GroundwaterConfig::default(),
            stress: StressConfig::default(),
            chunk_radius: 1,
            phase1_enabled: true,
            phase2_enabled: true,
            phase3_enabled: true,
            phase4_enabled: true,
            time_budget_ms: 8000,
            sleep_count: 1,
            accumulation_enabled: true,
            accumulation_iterations: 3,
            lava_solidification_enabled: true,
            nest_positions: Vec::new(),
            corpse_positions: Vec::new(),
            // Legacy defaults
            metamorphism: MetamorphismConfig::default(),
            minerals: MineralConfig::default(),
            collapse: CollapseConfig::default(),
            metamorphism_enabled: true,
            minerals_enabled: true,
            collapse_enabled: true,
        }
    }
}

// ──────────────────────────────────────────────────────────────
// Ambient Groundwater Model
// ──────────────────────────────────────────────────────────────

/// Ambient groundwater config — depth-based moisture for passive geological effects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundwaterConfig {
    pub enabled: bool,
    /// Overall passive/active strength slider (0.0 = off, 1.0 = full)
    pub strength: f32,
    /// Y-level where groundwater starts (moisture increases below this)
    pub depth_baseline: f32,
    /// Moisture gain per unit depth below baseline
    pub depth_scale: f32,
    /// Multiplier for drip zones (ceilings with air below)
    pub drip_zone_multiplier: f32,
    // Porosity per host rock type (NOT exposed in UI)
    pub porosity_limestone: f32,
    pub porosity_sandstone: f32,
    pub porosity_slate: f32,
    pub porosity_marble: f32,
    pub porosity_granite: f32,
    pub porosity_basalt: f32,
    // Power controls for per-effect tuning
    pub erosion_power: f32,
    pub flowstone_power: f32,
    pub enrichment_power: f32,
    pub soft_rock_mult: f32,
    pub hard_rock_mult: f32,
}

impl Default for GroundwaterConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strength: 0.3,
            depth_baseline: 0.0,
            depth_scale: 0.02,
            drip_zone_multiplier: 2.0,
            porosity_limestone: 1.0,
            porosity_sandstone: 0.8,
            porosity_slate: 0.5,
            porosity_marble: 0.3,
            porosity_granite: 0.2,
            porosity_basalt: 0.1,
            erosion_power: 1.0,
            flowstone_power: 1.0,
            enrichment_power: 1.0,
            soft_rock_mult: 1.0,
            hard_rock_mult: 0.15,
        }
    }
}

// ──────────────────────────────────────────────────────────────
// Phase 1: The Reaction (10,000 years)
// ──────────────────────────────────────────────────────────────

/// Config for Phase 1: acid dissolution, surface oxidation, basalt crust.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactionConfig {
    // Acid dissolution (BFS through limestone from exposed pyrite)
    pub acid_dissolution_prob: f32,
    pub acid_dissolution_radius: u32,
    pub acid_dissolution_enabled: bool,
    pub acid_max_dissolved_per_source: u32,
    // Copper oxidation (copper + air → malachite)
    pub copper_oxidation_prob: f32,
    pub copper_oxidation_enabled: bool,
    // Basalt crust (solid adjacent to lava → basalt)
    pub basalt_crust_prob: f32,
    pub basalt_crust_enabled: bool,
    // Sulfide acid dissolution (BFS through limestone from exposed sulfide)
    pub sulfide_acid_enabled: bool,
    pub sulfide_acid_prob: f32,
    pub sulfide_acid_radius: u32,
    pub sulfide_water_amplification: f32,
    // Gypsum deposition (acid dissolution byproduct)
    pub limestone_acid_radius_boost: f32,
    pub gypsum_deposition_prob: f32,
    pub gypsum_enabled: bool,
}

impl Default for ReactionConfig {
    fn default() -> Self {
        Self {
            acid_dissolution_prob: 0.25,
            acid_dissolution_radius: 3,
            acid_dissolution_enabled: true,
            acid_max_dissolved_per_source: 30,
            copper_oxidation_prob: 0.0012,
            copper_oxidation_enabled: true,
            basalt_crust_prob: 0.001,
            basalt_crust_enabled: true,
            sulfide_acid_enabled: true,
            sulfide_acid_prob: 0.60,
            sulfide_acid_radius: 2,
            sulfide_water_amplification: 2.0,
            limestone_acid_radius_boost: 1.5,
            gypsum_deposition_prob: 0.18,
            gypsum_enabled: true,
        }
    }
}

// ──────────────────────────────────────────────────────────────
// Phase 2: The Aureole (100,000 years)
// ──────────────────────────────────────────────────────────────

/// Config for Phase 2: contact metamorphism around heat sources + water erosion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AureoleConfig {
    pub aureole_radius: u32,
    // Contact zone (0-2 voxels from heat)
    pub contact_limestone_to_marble_prob: f32,
    pub contact_sandstone_to_granite_prob: f32,
    // Mid aureole (3-5 voxels from heat)
    pub mid_limestone_to_marble_prob: f32,
    pub mid_sandstone_to_granite_prob: f32,
    // Outer aureole (6-8 voxels from heat)
    pub outer_limestone_to_marble_prob: f32,
    // Water erosion
    pub water_erosion_prob: f32,
    pub water_erosion_enabled: bool,
    pub metamorphism_enabled: bool,
    // Coal maturation (coal → graphite → diamond)
    pub coal_maturation_enabled: bool,
    pub coal_to_graphite_prob: f32,
    pub coal_to_graphite_mid_prob: f32,
    pub graphite_to_diamond_prob: f32,
    // Quartz silicification (hydrothermal silica replacement)
    pub silicification_enabled: bool,
    pub silicification_limestone_prob: f32,
    pub silicification_sandstone_prob: f32,
    /// Water search radius multiplier for silicification (radius = effective_radius * this)
    pub silicification_water_radius_mult: u32,
    // Skarn + hornfels metamorphism
    pub contact_limestone_to_garnet_prob: f32,
    pub mid_limestone_to_garnet_prob: f32,
    pub mid_limestone_to_diopside_prob: f32,
    // Post-acid recrystallization (fills acid-dissolved air near heat with metamorphic minerals)
    pub recrystallization_prob: f32,
    pub contact_slate_to_hornfels_prob: f32,
    pub mid_slate_to_hornfels_prob: f32,
    pub outer_slate_to_hornfels_prob: f32,
    // Lava zone contact metamorphism + ore veins
    #[serde(default = "default_true")]
    pub zone_enabled: bool,
    #[serde(default = "default_one")]
    pub heat_multiplier: f32,
    #[serde(default = "default_one")]
    pub radius_scale: f32,
    #[serde(default = "default_water_boost_max")]
    pub water_boost_max: f32,
    #[serde(default = "default_water_search_mult")]
    pub water_search_radius_mult: f32,
    #[serde(default = "default_large_vein")]
    pub large_vein_base_size: u32,
    #[serde(default = "default_small_vein")]
    pub small_vein_base_size: u32,
    #[serde(default = "default_min_zone")]
    pub min_lava_zone_size: u32,
    #[serde(default = "default_garnet_pocket")]
    pub garnet_pocket_size: u32,
    #[serde(default = "default_diopside_pocket")]
    pub diopside_pocket_size: u32,
    /// Maximum base radius for aureole metamorphic sphere (before water boost).
    #[serde(default = "default_max_aureole_radius")]
    pub max_radius: f32,
    // ── Aureole deposit detail settings ──
    /// Number of ore vein seed points per zone (was hardcoded 8)
    pub aureole_vein_count: u32,
    /// Min vein size for aureole ore deposits
    pub aureole_vein_min: u32,
    /// Max vein size for aureole ore deposits
    pub aureole_vein_max: u32,
    /// Garnet compact deposit radius (Compact bias target size)
    pub garnet_compact_size: u32,
    /// Diopside compact deposit radius
    pub diopside_compact_size: u32,
    /// Number of garnet pockets per zone
    pub garnet_pocket_count: u32,
    /// Number of diopside pockets per zone
    pub diopside_pocket_count: u32,
    // ── Aureole vein geometry (wall-climbing, like hydrothermal) ──
    /// Use wall-climbing bias for aureole ore veins (0/1). When enabled, veins streak up walls.
    pub aureole_wall_climbing: bool,
    /// Min height of aureole wall-climbing veins
    pub aureole_climb_height_min: u32,
    /// Max height of aureole wall-climbing veins
    pub aureole_climb_height_max: u32,
    /// Min visible width on wall face
    pub aureole_wall_width_min: u32,
    /// Max visible width on wall face
    pub aureole_wall_width_max: u32,
    /// Min depth into solid rock behind wall
    pub aureole_rock_depth_min: u32,
    /// Max depth into solid rock behind wall
    pub aureole_rock_depth_max: u32,
    /// Surface exposure: min air-face neighbors required for seed eligibility (higher = more visible)
    pub aureole_min_surface_exposure: u32,
    // ── Lava volume scaling for aureole ──
    /// Radius to count lava cells for zone volume scaling
    pub aureole_lava_volume_max_cells: u32,
    /// Deposit size multiplier bonus at max lava volume
    pub aureole_lava_deposit_mult: f32,
    /// Vein count multiplier bonus at max lava volume
    pub aureole_lava_count_mult: f32,
    /// Vein seed spread factor (0.0 = random placement, 1.0 = max spread apart)
    pub aureole_vein_spread: f32,
    // ── Water boost exposure ──
    /// Radius for water search around lava zone (in voxels, 0 = face-neighbor only)
    pub aureole_water_search_radius: u32,
    /// Max water cells counted before diminishing returns
    pub aureole_water_max_cells: u32,
    /// Deposit size multiplier bonus at max water
    pub aureole_water_deposit_mult: f32,
}

impl Default for AureoleConfig {
    fn default() -> Self {
        Self {
            aureole_radius: 10,
            contact_limestone_to_marble_prob: 0.18,
            contact_sandstone_to_granite_prob: 0.50,
            mid_limestone_to_marble_prob: 0.15,
            mid_sandstone_to_granite_prob: 0.25,
            outer_limestone_to_marble_prob: 0.30,
            water_erosion_prob: 0.05,
            water_erosion_enabled: true,
            metamorphism_enabled: true,
            coal_maturation_enabled: true,
            coal_to_graphite_prob: 0.70,
            coal_to_graphite_mid_prob: 0.35,
            graphite_to_diamond_prob: 0.15,
            silicification_enabled: true,
            silicification_limestone_prob: 0.55,
            silicification_sandstone_prob: 0.15,
            silicification_water_radius_mult: 3,
            contact_limestone_to_garnet_prob: 0.65,
            mid_limestone_to_garnet_prob: 0.30,
            mid_limestone_to_diopside_prob: 0.65,
            recrystallization_prob: 0.70,
            contact_slate_to_hornfels_prob: 0.90,
            mid_slate_to_hornfels_prob: 0.60,
            outer_slate_to_hornfels_prob: 0.25,
            zone_enabled: true,
            heat_multiplier: 1.0,
            radius_scale: 1.0,
            water_boost_max: 0.6,
            water_search_radius_mult: 2.0,
            large_vein_base_size: 15,
            small_vein_base_size: 6,
            min_lava_zone_size: 5,
            garnet_pocket_size: 4,
            diopside_pocket_size: 4,
            max_radius: 10.0,
            aureole_vein_count: 8,
            aureole_vein_min: 6,
            aureole_vein_max: 20,
            garnet_compact_size: 8,
            diopside_compact_size: 8,
            garnet_pocket_count: 2,
            diopside_pocket_count: 1,
            aureole_wall_climbing: true,
            aureole_climb_height_min: 4,
            aureole_climb_height_max: 10,
            aureole_wall_width_min: 2,
            aureole_wall_width_max: 3,
            aureole_rock_depth_min: 1,
            aureole_rock_depth_max: 3,
            aureole_min_surface_exposure: 1,
            aureole_vein_spread: 0.5,
            aureole_lava_volume_max_cells: 50,
            aureole_lava_deposit_mult: 1.0,
            aureole_lava_count_mult: 0.5,
            aureole_water_search_radius: 3,
            aureole_water_max_cells: 30,
            aureole_water_deposit_mult: 0.5,
        }
    }
}

// ──────────────────────────────────────────────────────────────
// Phase 3: The Veins (500,000 years)
// ──────────────────────────────────────────────────────────────

/// Config for Phase 3: water-heat convergence vein deposition + formation growth.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VeinConfig {
    // Water-heat convergence vein deposition
    pub vein_deposition_prob: f32,
    pub vein_enabled: bool,
    /// Max 3D distance between water and heat source for activation
    #[serde(default = "default_convergence_radius")]
    pub convergence_radius: f32,
    /// Height of hypothermal zone above water (0..hypothermal_height)
    #[serde(default = "default_hypothermal_height")]
    pub hypothermal_height: u32,
    /// Height of mesothermal zone above water (hypothermal..mesothermal_height)
    #[serde(default = "default_mesothermal_height")]
    pub mesothermal_height: u32,
    /// Height of epithermal zone above water (mesothermal..epithermal_height)
    #[serde(default = "default_epithermal_height")]
    pub epithermal_height: u32,
    /// Max XZ spread from water source for wall site scanning
    #[serde(default = "default_horizontal_spread")]
    pub horizontal_spread: u32,
    /// Min veins to place per temperature zone per convergence area
    #[serde(default = "default_veins_per_zone_min")]
    pub veins_per_zone_min: u32,
    /// Max veins to place per temperature zone per convergence area
    #[serde(default = "default_veins_per_zone_max")]
    pub veins_per_zone_max: u32,
    /// Min height of individual wall-climbing veins
    #[serde(default = "default_vein_climb_height_min")]
    pub vein_climb_height_min: u32,
    /// Max height of individual wall-climbing veins
    #[serde(default = "default_vein_climb_height_max")]
    pub vein_climb_height_max: u32,
    /// Min visible width on wall face
    #[serde(default = "default_vein_wall_width_min")]
    pub vein_wall_width_min: u32,
    /// Max visible width on wall face
    #[serde(default = "default_vein_wall_width_max")]
    pub vein_wall_width_max: u32,
    /// Min depth into solid rock behind wall
    #[serde(default = "default_vein_rock_depth_min")]
    pub vein_rock_depth_min: u32,
    /// Max depth into solid rock behind wall
    #[serde(default = "default_vein_rock_depth_max")]
    pub vein_rock_depth_max: u32,
    /// Preference for heat-facing walls (0.0 = none, 1.0 = strong)
    #[serde(default = "default_heat_direction_bias")]
    pub heat_direction_bias: f32,
    /// Min distance between processed water cells (spatial deduplication)
    #[serde(default = "default_convergence_spacing")]
    pub convergence_spacing: u32,
    /// Epithermal rarity: probability that Gold/Sulfide actually deposits (0.0-1.0)
    pub epithermal_rarity: f32,
    // Formation growth
    pub crystal_growth_enabled: bool,
    pub crystal_growth_prob: f32,
    pub crystal_growth_max_per_chunk: u32,
    pub calcite_infill_enabled: bool,
    pub calcite_infill_prob: f32,
    pub calcite_infill_max_per_chunk: u32,
    pub flowstone_enabled: bool,
    pub flowstone_prob: f32,
    pub flowstone_max_per_chunk: u32,
    /// Density range for new growths
    pub growth_density_min: f32,
    pub growth_density_max: f32,
    /// Aperture scaling: wider tunnels = richer vein deposition
    pub aperture_scaling_enabled: bool,
    // Per-host rock ore selection
    pub host_rock_ore_enabled: bool,
    pub slate_pyrite_codeposit_prob: f32,
    pub slate_quartz_vein_prob: f32,
    /// Wall-rock alteration: probability that vein deposition converts adjacent limestone to garnet/diopside
    pub wall_rock_alteration_prob: f32,
    /// Minimum Y offset above water before hypothermal zone starts
    pub min_vein_height: u32,
    // Water volume scaling
    /// Radius to count water cells around each activated water source
    pub water_volume_radius: u32,
    /// Max water cells that contribute to scaling (diminishing returns cap)
    pub water_volume_max_cells: u32,
    /// Vein size multiplier bonus at max water volume (0.0 = no scaling)
    pub water_volume_vein_mult: f32,
    /// Veins-per-zone multiplier bonus at max water volume
    pub water_volume_amount_mult: f32,
    // Lava/heat volume scaling
    /// Radius to count heat sources around each convergence point
    pub lava_volume_radius: u32,
    /// Max heat cells that contribute to scaling
    pub lava_volume_max_cells: u32,
    /// Vein size multiplier bonus at max lava volume
    pub lava_volume_vein_mult: f32,
    /// Veins-per-zone multiplier bonus at max lava volume
    pub lava_volume_amount_mult: f32,
    /// Vein seed spread factor (0.0 = random, 1.0 = max spread apart)
    pub vein_spread: f32,
    // Spike/tendril intrusions ("centipede" look)
    /// Enable spikey tendrils radiating from vein bodies
    pub spike_enabled: bool,
    /// Min number of spikes per vein body
    pub spike_count_min: u32,
    /// Max number of spikes per vein body
    pub spike_count_max: u32,
    /// Min spike tendril length (voxels)
    pub spike_length_min: u32,
    /// Max spike tendril length (voxels)
    pub spike_length_max: u32,
    /// Per-step survival probability for spike decay (taper)
    pub spike_taper: f32,
}

impl Default for VeinConfig {
    fn default() -> Self {
        Self {
            vein_deposition_prob: 0.85,
            vein_enabled: true,
            convergence_radius: 70.0,
            hypothermal_height: 25,
            mesothermal_height: 45,
            epithermal_height: 65,
            horizontal_spread: 20,
            veins_per_zone_min: 2,
            veins_per_zone_max: 4,
            vein_climb_height_min: 6,
            vein_climb_height_max: 12,
            vein_wall_width_min: 2,
            vein_wall_width_max: 3,
            vein_rock_depth_min: 1,
            vein_rock_depth_max: 3,
            heat_direction_bias: 0.3,
            convergence_spacing: 25,
            epithermal_rarity: 0.55,
            crystal_growth_enabled: true,
            crystal_growth_prob: 0.30,
            crystal_growth_max_per_chunk: 4,
            calcite_infill_enabled: true,
            calcite_infill_prob: 0.15,
            calcite_infill_max_per_chunk: 4,
            flowstone_enabled: true,
            flowstone_prob: 0.10,
            flowstone_max_per_chunk: 3,
            growth_density_min: 0.3,
            growth_density_max: 0.6,
            aperture_scaling_enabled: true,
            host_rock_ore_enabled: true,
            slate_pyrite_codeposit_prob: 0.25,
            slate_quartz_vein_prob: 0.30,
            wall_rock_alteration_prob: 0.18,
            min_vein_height: 3,
            water_volume_radius: 8,
            water_volume_max_cells: 50,
            water_volume_vein_mult: 1.0,
            water_volume_amount_mult: 1.0,
            lava_volume_radius: 8,
            lava_volume_max_cells: 30,
            lava_volume_vein_mult: 0.5,
            lava_volume_amount_mult: 0.5,
            vein_spread: 0.5,
            spike_enabled: true,
            spike_count_min: 4,
            spike_count_max: 10,
            spike_length_min: 2,
            spike_length_max: 5,
            spike_taper: 0.7,
        }
    }
}

// ──────────────────────────────────────────────────────────────
// Phase 4: The Deep Time (1,250,000 years)
// ──────────────────────────────────────────────────────────────

/// Config for Phase 4: supergene enrichment, vein thickening, mature formations, collapse.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepTimeConfig {
    // Supergene enrichment
    pub enrichment_prob: f32,
    pub max_enrichment_per_chunk: u32,
    pub enrichment_search_radius: i32,
    pub enrichment_enabled: bool,
    /// Minimum cluster size when enrichment triggers grow_vein()
    #[serde(default = "default_enrichment_cluster_min")]
    pub enrichment_cluster_min: u32,
    /// Maximum cluster size when enrichment triggers grow_vein()
    #[serde(default = "default_enrichment_cluster_max")]
    pub enrichment_cluster_max: u32,
    // Vein thickening (water-proximity coating + fracture fingers)
    pub vein_thickening_enabled: bool,
    pub vein_thickening_max_per_chunk: u32,
    /// Max Euclidean distance to water for hydrothermal influence
    #[serde(default = "default_vein_thickening_water_radius")]
    pub vein_thickening_water_radius: f32,
    /// Coating depth into host rock (1 or 2 voxels)
    #[serde(default = "default_vein_thickening_coat_depth")]
    pub vein_thickening_coat_depth: u32,
    /// Every N-th surface ore voxel spawns a fracture finger
    #[serde(default = "default_vein_thickening_finger_interval")]
    pub vein_thickening_finger_interval: u32,
    /// Minimum fracture finger length
    #[serde(default = "default_vein_thickening_finger_length_min")]
    pub vein_thickening_finger_length_min: u32,
    /// Maximum fracture finger length
    #[serde(default = "default_vein_thickening_finger_length_max")]
    pub vein_thickening_finger_length_max: u32,
    /// Per-step survival probability decay for fracture fingers
    #[serde(default = "default_vein_thickening_finger_taper")]
    pub vein_thickening_finger_taper: f32,
    // Mature formations
    pub mature_formations_enabled: bool,
    pub stalactite_growth_prob: f32,
    pub column_formation_prob: f32,
    // Collapse (embedded)
    pub collapse: CollapseConfig,
    // Nest fossilization
    pub nest_fossilization: NestFossilizationConfig,
    // Corpse fossilization
    pub corpse_fossilization: CorpseFossilizationConfig,
    // Slate aquitard (blocks vertical water flow)
    pub slate_aquitard_enabled: bool,
    pub slate_zone_top: f64,
    pub slate_zone_bottom: f64,
    pub slate_aquitard_factor: f32,
    pub slate_aquitard_concentration: f32,
}

impl Default for DeepTimeConfig {
    fn default() -> Self {
        Self {
            enrichment_prob: 0.90,
            max_enrichment_per_chunk: 400,
            enrichment_search_radius: 12,
            enrichment_enabled: true,
            enrichment_cluster_min: 3,
            enrichment_cluster_max: 30,
            vein_thickening_enabled: true,
            vein_thickening_max_per_chunk: 100,
            vein_thickening_water_radius: 40.0,
            vein_thickening_coat_depth: 1,
            vein_thickening_finger_interval: 5,
            vein_thickening_finger_length_min: 3,
            vein_thickening_finger_length_max: 5,
            vein_thickening_finger_taper: 0.7,
            mature_formations_enabled: true,
            stalactite_growth_prob: 0.10,
            column_formation_prob: 0.05,
            collapse: CollapseConfig::default(),
            nest_fossilization: NestFossilizationConfig::default(),
            corpse_fossilization: CorpseFossilizationConfig::default(),
            slate_aquitard_enabled: true,
            slate_zone_top: 130.0,
            slate_zone_bottom: -100.0,
            slate_aquitard_factor: 0.05,
            slate_aquitard_concentration: 2.0,
        }
    }
}

/// Config for spider corpse fossilization (Phase 4 sub-stage).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpseFossilizationConfig {
    pub enabled: bool,
    pub corpse_radius: u32,
    pub pyrite_prob: f32,
    pub calcium_prob: f32,
    pub water_required: bool,
    pub min_sleep_cycles: u32,
}

impl Default for CorpseFossilizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            corpse_radius: 1,
            pyrite_prob: 0.50,
            calcium_prob: 0.40,
            water_required: true,
            min_sleep_cycles: 2,
        }
    }
}

/// Config for spider nest fossilization (Phase 4 sub-stage).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NestFossilizationConfig {
    pub enabled: bool,
    pub nest_radius: u32,
    pub pyrite_prob: f32,
    pub opal_prob: f32,
    pub buried_required: bool,
    pub water_required_for_pyrite: bool,
    pub water_required_for_opal: bool,
}

impl Default for NestFossilizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            nest_radius: 2,
            pyrite_prob: 0.60,
            opal_prob: 0.40,
            buried_required: false,
            water_required_for_pyrite: true,
            water_required_for_opal: true,
        }
    }
}

// ──────────────────────────────────────────────────────────────
// Legacy configs (kept for backward compat with old FFI/metamorphism/minerals)
// ──────────────────────────────────────────────────────────────

/// Metamorphism transformation probabilities (legacy — used by old metamorphism.rs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetamorphismConfig {
    pub limestone_to_marble_prob: f32,
    pub limestone_to_marble_depth: f32,
    pub limestone_to_marble_enabled: bool,
    pub sandstone_to_granite_prob: f32,
    pub sandstone_to_granite_depth: f32,
    pub sandstone_to_granite_min_neighbors: u32,
    pub sandstone_to_granite_enabled: bool,
    pub slate_to_marble_prob: f32,
    pub slate_to_marble_enabled: bool,
    pub granite_to_basalt_prob: f32,
    pub granite_to_basalt_min_air: u32,
    pub granite_to_basalt_enabled: bool,
    pub iron_to_pyrite_prob: f32,
    pub iron_to_pyrite_search_radius: u32,
    pub iron_to_pyrite_enabled: bool,
    pub copper_to_malachite_prob: f32,
    pub copper_to_malachite_enabled: bool,
}

impl Default for MetamorphismConfig {
    fn default() -> Self {
        Self {
            limestone_to_marble_prob: 0.40,
            limestone_to_marble_depth: -50.0,
            limestone_to_marble_enabled: true,
            sandstone_to_granite_prob: 0.25,
            sandstone_to_granite_depth: -100.0,
            sandstone_to_granite_min_neighbors: 4,
            sandstone_to_granite_enabled: true,
            slate_to_marble_prob: 0.60,
            slate_to_marble_enabled: true,
            granite_to_basalt_prob: 0.15,
            granite_to_basalt_min_air: 2,
            granite_to_basalt_enabled: true,
            iron_to_pyrite_prob: 0.35,
            iron_to_pyrite_search_radius: 2,
            iron_to_pyrite_enabled: true,
            copper_to_malachite_prob: 0.50,
            copper_to_malachite_enabled: true,
        }
    }
}

/// Mineral growth configuration (legacy — used by old minerals.rs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MineralConfig {
    pub crystal_growth_max: u32,
    pub crystal_growth_enabled: bool,
    pub crystal_growth_prob: f32,
    pub malachite_stalactite_max: u32,
    pub malachite_stalactite_enabled: bool,
    pub malachite_stalactite_prob: f32,
    pub quartz_extension_prob: f32,
    pub quartz_extension_max: u32,
    pub quartz_extension_enabled: bool,
    pub calcite_infill_max: u32,
    pub calcite_infill_depth: f32,
    pub calcite_infill_min_faces: u32,
    pub calcite_infill_enabled: bool,
    pub calcite_infill_prob: f32,
    pub pyrite_crust_max: u32,
    pub pyrite_crust_min_solid: u32,
    pub pyrite_crust_enabled: bool,
    pub pyrite_crust_prob: f32,
    pub growth_density_min: f32,
    pub growth_density_max: f32,
}

impl Default for MineralConfig {
    fn default() -> Self {
        Self {
            crystal_growth_max: 2,
            crystal_growth_enabled: true,
            crystal_growth_prob: 0.3,
            malachite_stalactite_max: 1,
            malachite_stalactite_enabled: true,
            malachite_stalactite_prob: 0.2,
            quartz_extension_prob: 0.10,
            quartz_extension_max: 1,
            quartz_extension_enabled: true,
            calcite_infill_max: 1,
            calcite_infill_depth: -30.0,
            calcite_infill_min_faces: 3,
            calcite_infill_enabled: true,
            calcite_infill_prob: 0.15,
            pyrite_crust_max: 1,
            pyrite_crust_min_solid: 2,
            pyrite_crust_enabled: true,
            pyrite_crust_prob: 0.1,
            growth_density_min: 0.3,
            growth_density_max: 0.6,
        }
    }
}

/// Structural collapse configuration for sleep cycles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollapseConfig {
    pub strut_survival: [f32; 8],
    pub stress_multiplier: f32,
    pub max_cascade_iterations: u32,
    pub rubble_fill_ratio: f32,
    pub min_stress_for_cascade: f32,
    pub rubble_material_match: bool,
    pub support_stress_penalty: f32,
    pub collapse_enabled: bool,
}

impl Default for CollapseConfig {
    fn default() -> Self {
        Self {
            strut_survival: [
                0.0,   // None (unused)
                0.25,  // SlateStrut
                0.30,  // GraniteStrut
                0.25,  // LimestoneStrut
                0.55,  // CopperStrut
                0.70,  // IronStrut
                0.85,  // SteelStrut
                0.95,  // CrystalStrut
            ],
            stress_multiplier: 0.8,
            max_cascade_iterations: 3,
            rubble_fill_ratio: 0.65,
            min_stress_for_cascade: 0.95,
            rubble_material_match: true,
            support_stress_penalty: 1.0,
            collapse_enabled: true,
        }
    }
}
