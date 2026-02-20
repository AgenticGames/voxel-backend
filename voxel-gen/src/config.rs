use serde::{Deserialize, Serialize};

// Re-export StressConfig from voxel-core for backward compatibility
pub use voxel_core::stress::StressConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub seed: u64,
    pub chunk_size: usize,
    pub noise: NoiseConfig,
    pub worm: WormConfig,
    pub ore: OreConfig,
    pub formations: FormationConfig,
    pub octree_max_depth: u32,
    /// Maximum edge length for triangle filtering (removes stretched artifacts).
    pub max_edge_length: f32,
    /// Region size in chunks per axis for global worm planning (default 3).
    pub region_size: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseConfig {
    pub cavern_frequency: f64,
    pub cavern_threshold: f64,
    pub detail_octaves: u32,
    pub detail_persistence: f64,
    pub warp_amplitude: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WormConfig {
    pub worms_per_region: u32,
    pub radius_min: f32,
    pub radius_max: f32,
    pub step_length: f32,
    pub max_steps: u32,
    pub falloff_power: f32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            chunk_size: 16,
            noise: NoiseConfig::default(),
            worm: WormConfig::default(),
            ore: OreConfig::default(),
            formations: FormationConfig::default(),
            octree_max_depth: 4,
            max_edge_length: 5.0,
            region_size: 3,
        }
    }
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            cavern_frequency: 0.05,
            cavern_threshold: 0.55,
            detail_octaves: 4,
            detail_persistence: 0.5,
            warp_amplitude: 5.0,
        }
    }
}

impl Default for WormConfig {
    fn default() -> Self {
        Self {
            worms_per_region: 5,
            radius_min: 2.0,
            radius_max: 4.0,
            step_length: 1.0,
            max_steps: 200,
            falloff_power: 2.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OreVeinParams {
    pub frequency: f64,
    pub threshold: f64,
    pub depth_min: f64,
    pub depth_max: f64,
}

// ── New config structs for deposit morphologies ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostRockConfig {
    /// Above this Y → Sandstone
    pub sandstone_depth: f64,
    /// Limestone→Granite boundary
    pub granite_depth: f64,
    /// Granite→Slate boundary (Basalt is intrusions, not a layer)
    pub basalt_depth: f64,
    /// Slate→Marble boundary
    pub slate_depth: f64,
    /// Noise amplitude for perturbing layer boundaries
    pub boundary_noise_amplitude: f64,
    /// Noise frequency for boundary perturbation
    pub boundary_noise_frequency: f64,
    /// 2D noise frequency for basalt intrusion columns
    pub basalt_intrusion_frequency: f64,
    /// Threshold for basalt columns (higher = rarer)
    pub basalt_intrusion_threshold: f64,
    /// Basalt columns only below this Y
    pub basalt_intrusion_depth_max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandedIronConfig {
    /// Sine wave frequency for horizontal bands
    pub band_frequency: f64,
    /// Noise amplitude added to sine for natural edges
    pub noise_perturbation: f64,
    /// Frequency of the perturbation noise
    pub noise_frequency: f64,
    /// Combined value must exceed this
    pub threshold: f64,
    pub depth_min: f64,
    pub depth_max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KimberlitePipeConfig {
    /// 2D noise frequency for pipe center locations (low = widely spaced)
    pub pipe_frequency_2d: f64,
    /// Threshold for pipe centers (very high = rare)
    pub pipe_threshold: f64,
    /// Pipes only below this depth
    pub depth_min: f64,
    /// Pipes extend up to this depth
    pub depth_max: f64,
    /// Additional noise threshold for diamond within pipe
    pub diamond_threshold: f64,
    /// Noise frequency for diamond distribution
    pub diamond_frequency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SulfideBlobConfig {
    /// Low frequency for large blobs
    pub frequency: f64,
    /// Threshold for sulfide blob boundary
    pub threshold: f64,
    /// Higher threshold: tin pockets within sulfide
    pub tin_threshold: f64,
    pub depth_min: f64,
    pub depth_max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeodeConfig {
    /// Noise frequency for geode center detection
    pub frequency: f64,
    /// Very high threshold = rare small regions
    pub center_threshold: f64,
    /// Thickness of the crystal shell (in noise-space units)
    pub shell_thickness: f64,
    /// Density value for geode interior (negative = hollow)
    pub hollow_factor: f32,
    pub depth_min: f64,
    pub depth_max: f64,
}

// ── Main ore config ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OreConfig {
    /// Banded iron formation — horizontal sine-wave layers
    pub iron: BandedIronConfig,
    /// Dendritic copper — branching tendrils (shallow, RidgedMulti)
    pub copper: OreVeinParams,
    /// Malachite — green copper indicator zones (deep)
    pub malachite: OreVeinParams,
    /// Quartz reef veins — narrow ridged structures (host for gold)
    pub quartz: OreVeinParams,
    /// Gold — only inside quartz reef, higher threshold
    pub gold: OreVeinParams,
    /// Pyrite — indicator near copper/gold, lower threshold halo
    pub pyrite: OreVeinParams,
    /// Kimberlite pipes — vertical cylindrical intrusions with diamonds
    pub kimberlite: KimberlitePipeConfig,
    /// Massive sulfide blobs — large irregular deposits with tin pockets
    pub sulfide: SulfideBlobConfig,
    /// Geodes — crystal-lined hollow pockets
    pub geode: GeodeConfig,
    /// Host rock depth layering
    pub host_rock: HostRockConfig,
}

impl Default for HostRockConfig {
    fn default() -> Self {
        Self {
            sandstone_depth: 200.0,
            granite_depth: 160.0,
            basalt_depth: 20.0,
            slate_depth: -150.0,
            boundary_noise_amplitude: 8.0,
            boundary_noise_frequency: 0.03,
            basalt_intrusion_frequency: 0.02,
            basalt_intrusion_threshold: 0.85,
            basalt_intrusion_depth_max: 10.0,
        }
    }
}

impl Default for BandedIronConfig {
    fn default() -> Self {
        Self {
            band_frequency: 0.2,
            noise_perturbation: 1.0,
            noise_frequency: 0.15,
            threshold: 1.2,
            depth_min: -200.0,
            depth_max: 200.0,
        }
    }
}

impl Default for KimberlitePipeConfig {
    fn default() -> Self {
        Self {
            pipe_frequency_2d: 0.008,
            pipe_threshold: 0.9,
            depth_min: -200.0,
            depth_max: -30.0,
            diamond_threshold: 0.75,
            diamond_frequency: 0.2,
        }
    }
}

impl Default for SulfideBlobConfig {
    fn default() -> Self {
        Self {
            frequency: 0.5,
            threshold: 0.2,
            tin_threshold: 0.5,
            depth_min: -200.0,
            depth_max: -20.0,
        }
    }
}

impl Default for GeodeConfig {
    fn default() -> Self {
        Self {
            frequency: 0.009,
            center_threshold: 0.94,
            shell_thickness: 0.01,
            hollow_factor: -0.20,
            depth_min: -200.0,
            depth_max: 200.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormationConfig {
    /// Master switch for cave formations
    pub enabled: bool,
    /// Noise frequency for placement field
    pub placement_frequency: f64,
    /// Higher = sparser placement
    pub placement_threshold: f64,
    /// Per-surface probability for stalactites
    pub stalactite_chance: f32,
    /// Per-surface probability for stalagmites
    pub stalagmite_chance: f32,
    /// Per-surface probability for flowstone shelves
    pub flowstone_chance: f32,
    /// Probability when ceiling+floor align
    pub column_chance: f32,
    /// Max gap for column attempt
    pub column_max_gap: usize,
    /// Cone length range (min)
    pub length_min: f32,
    /// Cone length range (max)
    pub length_max: f32,
    /// Cone base radius range (min)
    pub radius_min: f32,
    /// Cone base radius range (max)
    pub radius_max: f32,
    /// Navigation safety cap on radius
    pub max_radius: f32,
    /// Column radius range (min)
    pub column_radius_min: f32,
    /// Column radius range (max)
    pub column_radius_max: f32,
    /// Flowstone shelf extent (min)
    pub flowstone_length_min: f32,
    /// Flowstone shelf extent (max)
    pub flowstone_length_max: f32,
    /// Flowstone vertical extent
    pub flowstone_thickness: f32,
    /// Minimum air gap for any formation
    pub min_air_gap: usize,
    /// Clearance maintained below/above formation tip
    pub min_clearance: usize,
    /// Density blending factor
    pub smoothness: f32,
}

impl Default for FormationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            placement_frequency: 0.15,
            placement_threshold: 0.65,
            stalactite_chance: 0.12,
            stalagmite_chance: 0.08,
            flowstone_chance: 0.05,
            column_chance: 0.15,
            column_max_gap: 10,
            length_min: 2.0,
            length_max: 5.0,
            radius_min: 0.6,
            radius_max: 1.2,
            max_radius: 1.5,
            column_radius_min: 0.5,
            column_radius_max: 1.0,
            flowstone_length_min: 1.5,
            flowstone_length_max: 3.0,
            flowstone_thickness: 0.8,
            min_air_gap: 5,
            min_clearance: 3,
            smoothness: 2.0,
        }
    }
}

impl Default for OreConfig {
    fn default() -> Self {
        Self {
            iron: BandedIronConfig::default(),
            copper: OreVeinParams {
                frequency: 0.009,
                threshold: 0.72,
                depth_min: -30.0,
                depth_max: 200.0,
            },
            malachite: OreVeinParams {
                frequency: 0.8,
                threshold: 0.1,
                depth_min: -200.0,
                depth_max: -30.0,
            },
            quartz: OreVeinParams {
                frequency: 0.01,
                threshold: 0.67,
                depth_min: -200.0,
                depth_max: 200.0,
            },
            gold: OreVeinParams {
                frequency: 0.08,
                threshold: 0.87,
                depth_min: -200.0,
                depth_max: 200.0,
            },
            pyrite: OreVeinParams {
                frequency: 0.05,
                threshold: 0.92,
                depth_min: -200.0,
                depth_max: 200.0,
            },
            kimberlite: KimberlitePipeConfig::default(),
            sulfide: SulfideBlobConfig::default(),
            geode: GeodeConfig::default(),
            host_rock: HostRockConfig::default(),
        }
    }
}

