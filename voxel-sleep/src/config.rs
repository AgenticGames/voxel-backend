use serde::{Deserialize, Serialize};
use voxel_core::stress::StressConfig;

/// Top-level sleep configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepConfig {
    pub metamorphism: MetamorphismConfig,
    pub minerals: MineralConfig,
    pub collapse: CollapseConfig,
    pub stress: StressConfig,
    pub time_budget_ms: u32,
    pub chunk_radius: u32,
    pub metamorphism_enabled: bool,
    pub minerals_enabled: bool,
    pub collapse_enabled: bool,
    pub sleep_count: u32,
}

impl Default for SleepConfig {
    fn default() -> Self {
        Self {
            metamorphism: MetamorphismConfig::default(),
            minerals: MineralConfig::default(),
            collapse: CollapseConfig::default(),
            stress: StressConfig::default(),
            time_budget_ms: 8000,
            chunk_radius: 1,
            metamorphism_enabled: true,
            minerals_enabled: true,
            collapse_enabled: true,
            sleep_count: 1,
        }
    }
}

/// Metamorphism transformation probabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetamorphismConfig {
    /// Limestone -> Marble when deep or adjacent to Basalt/Kimberlite
    pub limestone_to_marble_prob: f32,
    pub limestone_to_marble_depth: f32,
    pub limestone_to_marble_enabled: bool,
    /// Sandstone -> Granite when very deep with 4+ solid neighbors
    pub sandstone_to_granite_prob: f32,
    pub sandstone_to_granite_depth: f32,
    pub sandstone_to_granite_min_neighbors: u32,
    pub sandstone_to_granite_enabled: bool,
    /// Slate -> Marble when adjacent to Kimberlite pipe
    pub slate_to_marble_prob: f32,
    pub slate_to_marble_enabled: bool,
    /// Granite -> Basalt when adjacent to 2+ air voxels (cooling)
    pub granite_to_basalt_prob: f32,
    pub granite_to_basalt_min_air: u32,
    pub granite_to_basalt_enabled: bool,
    /// Iron -> Pyrite when adjacent to Sulfide within 2 voxels
    pub iron_to_pyrite_prob: f32,
    pub iron_to_pyrite_search_radius: u32,
    pub iron_to_pyrite_enabled: bool,
    /// Copper -> Malachite when adjacent to 1+ air voxel (oxidation)
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

/// Mineral growth configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MineralConfig {
    /// Crystal/Amethyst grows into air with 2+ crystal neighbors
    pub crystal_growth_max: u32,
    pub crystal_growth_enabled: bool,
    pub crystal_growth_prob: f32,
    /// Malachite stalactite: copper above air, limestone nearby
    pub malachite_stalactite_max: u32,
    pub malachite_stalactite_enabled: bool,
    pub malachite_stalactite_prob: f32,
    /// Quartz vein extension: 10% probability per quartz terminus
    pub quartz_extension_prob: f32,
    pub quartz_extension_max: u32,
    pub quartz_extension_enabled: bool,
    /// Calcite infill: limestone surrounds air with 3+ faces, depth < -30
    pub calcite_infill_max: u32,
    pub calcite_infill_depth: f32,
    pub calcite_infill_min_faces: u32,
    pub calcite_infill_enabled: bool,
    pub calcite_infill_prob: f32,
    /// Pyrite crust: grows outward from pyrite with 2+ solid behind
    pub pyrite_crust_max: u32,
    pub pyrite_crust_min_solid: u32,
    pub pyrite_crust_enabled: bool,
    pub pyrite_crust_prob: f32,
    /// Density range for new mineral growths (for smooth DC meshes)
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
    /// Per-strut-type survival probabilities per sleep cycle (indexed by SupportType as u8, [0] unused).
    /// [0]=None, [1]=SlateStrut, [2]=GraniteStrut, [3]=LimestoneStrut,
    /// [4]=CopperStrut, [5]=IronStrut, [6]=SteelStrut, [7]=CrystalStrut
    pub strut_survival: [f32; 8],
    /// Stress multiplier applied after support degradation (geological pressure)
    pub stress_multiplier: f32,
    /// Max collapse cascade iterations (higher than real-time's 5)
    pub max_cascade_iterations: u32,
    /// Rubble fill ratio for geological collapses
    pub rubble_fill_ratio: f32,
    /// Minimum stress value to trigger a collapse cascade
    pub min_stress_for_cascade: f32,
    /// Whether rubble material matches the collapsed material
    pub rubble_material_match: bool,
    /// Stress penalty applied when a support is removed
    pub support_stress_penalty: f32,
    /// Master enable for collapse phase
    pub collapse_enabled: bool,
}

impl Default for CollapseConfig {
    fn default() -> Self {
        Self {
            strut_survival: [
                0.0,   // None (unused)
                0.25,  // SlateStrut (Tier 1 - stone, weakest)
                0.30,  // GraniteStrut (Tier 1)
                0.25,  // LimestoneStrut (Tier 1)
                0.55,  // CopperStrut (Tier 2)
                0.70,  // IronStrut (Tier 3)
                0.85,  // SteelStrut (Tier 4)
                0.95,  // CrystalStrut (Tier 5 - most durable)
            ],
            stress_multiplier: 1.5,
            max_cascade_iterations: 8,
            rubble_fill_ratio: 0.40,
            min_stress_for_cascade: 0.7,
            rubble_material_match: true,
            support_stress_penalty: 1.0,
            collapse_enabled: true,
        }
    }
}
