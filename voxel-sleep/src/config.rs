use serde::{Deserialize, Serialize};

/// Top-level sleep configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepConfig {
    pub metamorphism: MetamorphismConfig,
    pub minerals: MineralConfig,
    pub collapse: CollapseConfig,
    pub time_budget_ms: u32,
}

impl Default for SleepConfig {
    fn default() -> Self {
        Self {
            metamorphism: MetamorphismConfig::default(),
            minerals: MineralConfig::default(),
            collapse: CollapseConfig::default(),
            time_budget_ms: 8000,
        }
    }
}

/// Metamorphism transformation probabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetamorphismConfig {
    /// Limestone -> Marble when deep or adjacent to Basalt/Kimberlite
    pub limestone_to_marble_prob: f32,
    pub limestone_to_marble_depth: f32,
    /// Sandstone -> Granite when very deep with 4+ solid neighbors
    pub sandstone_to_granite_prob: f32,
    pub sandstone_to_granite_depth: f32,
    pub sandstone_to_granite_min_neighbors: u32,
    /// Slate -> Marble when adjacent to Kimberlite pipe
    pub slate_to_marble_prob: f32,
    /// Granite -> Basalt when adjacent to 2+ air voxels (cooling)
    pub granite_to_basalt_prob: f32,
    pub granite_to_basalt_min_air: u32,
    /// Iron -> Pyrite when adjacent to Sulfide within 2 voxels
    pub iron_to_pyrite_prob: f32,
    pub iron_to_pyrite_search_radius: u32,
    /// Copper -> Malachite when adjacent to 1+ air voxel (oxidation)
    pub copper_to_malachite_prob: f32,
}

impl Default for MetamorphismConfig {
    fn default() -> Self {
        Self {
            limestone_to_marble_prob: 0.40,
            limestone_to_marble_depth: -50.0,
            sandstone_to_granite_prob: 0.25,
            sandstone_to_granite_depth: -100.0,
            sandstone_to_granite_min_neighbors: 4,
            slate_to_marble_prob: 0.60,
            granite_to_basalt_prob: 0.15,
            granite_to_basalt_min_air: 2,
            iron_to_pyrite_prob: 0.35,
            iron_to_pyrite_search_radius: 2,
            copper_to_malachite_prob: 0.50,
        }
    }
}

/// Mineral growth configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MineralConfig {
    /// Crystal/Amethyst grows into air with 2+ crystal neighbors
    pub crystal_growth_max: u32,
    /// Malachite stalactite: copper above air, limestone nearby
    pub malachite_stalactite_max: u32,
    /// Quartz vein extension: 10% probability per quartz terminus
    pub quartz_extension_prob: f32,
    pub quartz_extension_max: u32,
    /// Calcite infill: limestone surrounds air with 3+ faces, depth < -30
    pub calcite_infill_max: u32,
    pub calcite_infill_depth: f32,
    pub calcite_infill_min_faces: u32,
    /// Pyrite crust: grows outward from pyrite with 2+ solid behind
    pub pyrite_crust_max: u32,
    pub pyrite_crust_min_solid: u32,
    /// Density range for new mineral growths (for smooth DC meshes)
    pub growth_density_min: f32,
    pub growth_density_max: f32,
}

impl Default for MineralConfig {
    fn default() -> Self {
        Self {
            crystal_growth_max: 2,
            malachite_stalactite_max: 1,
            quartz_extension_prob: 0.10,
            quartz_extension_max: 1,
            calcite_infill_max: 1,
            calcite_infill_depth: -30.0,
            calcite_infill_min_faces: 3,
            pyrite_crust_max: 1,
            pyrite_crust_min_solid: 2,
            growth_density_min: 0.3,
            growth_density_max: 0.6,
        }
    }
}

/// Structural collapse configuration for sleep cycles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollapseConfig {
    /// Survival probability per sleep for WoodBeam supports
    pub wood_beam_survival: f32,
    /// Survival probability per sleep for MetalBeam supports
    pub metal_beam_survival: f32,
    /// Survival probability per sleep for Reinforcement supports
    pub reinforcement_survival: f32,
    /// Stress multiplier applied after support degradation (geological pressure)
    pub stress_multiplier: f32,
    /// Max collapse cascade iterations (higher than real-time's 5)
    pub max_cascade_iterations: u32,
    /// Rubble fill ratio for geological collapses
    pub rubble_fill_ratio: f32,
}

impl Default for CollapseConfig {
    fn default() -> Self {
        Self {
            wood_beam_survival: 0.30,
            metal_beam_survival: 0.70,
            reinforcement_survival: 0.85,
            stress_multiplier: 1.5,
            max_cascade_iterations: 8,
            rubble_fill_ratio: 0.40,
        }
    }
}
