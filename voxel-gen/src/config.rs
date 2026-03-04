use serde::{Deserialize, Serialize};

// Re-export StressConfig from voxel-core for backward compatibility
pub use voxel_core::stress::StressConfig;
use voxel_core::material::Material;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MineConfig {
    /// Laplacian blur passes after mining (default: 2)
    pub smooth_iterations: u32,
    /// Per-pass blend factor 0.0-1.0 (default: 0.3)
    pub smooth_strength: f32,
    /// Extra voxels to expand dirty region (default: 2)
    pub dirty_expand: u32,
}

impl Default for MineConfig {
    fn default() -> Self {
        Self {
            smooth_iterations: 2,
            smooth_strength: 0.3,
            dirty_expand: 2,
        }
    }
}

/// Per-ore crystal placement tuning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OreCrystalConfig {
    /// Enable crystals for this ore type
    pub enabled: bool,
    /// Per-surface-point spawn probability (0-1)
    pub chance: f32,
    /// Noise value must exceed this to allow placement (0-1)
    pub density_threshold: f32,
    /// Minimum instance scale
    pub scale_min: f32,
    /// Maximum instance scale
    pub scale_max: f32,
    /// Weight for small size class
    pub small_weight: f32,
    /// Weight for medium size class
    pub medium_weight: f32,
    /// Weight for large size class
    pub large_weight: f32,
    /// Lerp factor: 0=random direction, 1=surface normal
    pub normal_alignment: f32,
    /// Max crystals per seed point
    pub cluster_size: u32,
    /// Cluster scatter radius (voxels)
    pub cluster_radius: f32,
    /// Offset along surface normal (voxels); 0 = flush with surface
    pub surface_offset: f32,
}

impl Default for OreCrystalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            chance: 0.3,
            density_threshold: 0.4,
            scale_min: 0.3,
            scale_max: 1.0,
            small_weight: 0.6,
            medium_weight: 0.3,
            large_weight: 0.1,
            normal_alignment: 0.7,
            cluster_size: 3,
            cluster_radius: 0.5,
            surface_offset: 0.1,
        }
    }
}

impl OreCrystalConfig {
    /// Builder: set spawn chance
    pub fn with_chance(mut self, chance: f32) -> Self {
        self.chance = chance;
        self
    }

    /// Builder: set scale range
    pub fn with_scale(mut self, min: f32, max: f32) -> Self {
        self.scale_min = min;
        self.scale_max = max;
        self
    }

    /// Builder: set cluster parameters
    pub fn with_cluster(mut self, size: u32, radius: f32) -> Self {
        self.cluster_size = size;
        self.cluster_radius = radius;
        self
    }

    /// Builder: set density threshold
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.density_threshold = threshold;
        self
    }
}

/// Crystal placement configuration for all ore types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrystalConfig {
    /// Master switch for crystal placement
    pub enabled: bool,
    pub iron: OreCrystalConfig,
    pub copper: OreCrystalConfig,
    pub malachite: OreCrystalConfig,
    pub tin: OreCrystalConfig,
    pub gold: OreCrystalConfig,
    pub diamond: OreCrystalConfig,
    pub kimberlite: OreCrystalConfig,
    pub sulfide: OreCrystalConfig,
    pub quartz: OreCrystalConfig,
    pub pyrite: OreCrystalConfig,
    pub amethyst: OreCrystalConfig,
    pub coal: OreCrystalConfig,
}

impl Default for CrystalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            iron: OreCrystalConfig::default().with_chance(0.2),
            copper: OreCrystalConfig::default().with_chance(0.25),
            malachite: OreCrystalConfig::default().with_chance(0.3),
            tin: OreCrystalConfig::default().with_chance(0.15),
            gold: OreCrystalConfig::default()
                .with_chance(0.4)
                .with_scale(0.2, 0.6),
            diamond: OreCrystalConfig::default()
                .with_chance(0.5)
                .with_scale(0.3, 0.8)
                .with_cluster(5, 0.4),
            kimberlite: OreCrystalConfig::default().with_chance(0.1),
            sulfide: OreCrystalConfig::default().with_chance(0.15),
            quartz: OreCrystalConfig::default()
                .with_chance(0.35)
                .with_scale(0.4, 1.2)
                .with_cluster(4, 0.6),
            pyrite: OreCrystalConfig::default()
                .with_chance(0.25)
                .with_scale(0.2, 0.7),
            amethyst: OreCrystalConfig::default()
                .with_chance(0.45)
                .with_scale(0.3, 1.0)
                .with_cluster(5, 0.5),
            coal: OreCrystalConfig::default()
                .with_chance(0.05)
                .with_scale(0.2, 0.4),
        }
    }
}

impl CrystalConfig {
    /// Look up the per-ore config for a given material.
    /// Returns a default (disabled) config for non-ore materials.
    pub fn ore_config(&self, mat: Material) -> &OreCrystalConfig {
        match mat {
            Material::Iron => &self.iron,
            Material::Copper => &self.copper,
            Material::Malachite => &self.malachite,
            Material::Tin => &self.tin,
            Material::Gold => &self.gold,
            Material::Diamond => &self.diamond,
            Material::Kimberlite => &self.kimberlite,
            Material::Sulfide => &self.sulfide,
            Material::Quartz => &self.quartz,
            Material::Pyrite => &self.pyrite,
            Material::Amethyst => &self.amethyst,
            Material::Coal => &self.coal,
            _ => &self.iron, // fallback; caller should pre-filter
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub seed: u64,
    pub chunk_size: usize,
    pub noise: NoiseConfig,
    pub worm: WormConfig,
    pub ore: OreConfig,
    pub formations: FormationConfig,
    pub pools: PoolConfig,
    pub mine: MineConfig,
    pub crystals: CrystalConfig,
    pub octree_max_depth: u32,
    /// Region size in chunks per axis for global worm planning (default 3).
    pub region_size: i32,
    /// World-space extent per chunk. 0.0 means use chunk_size (backward compatible).
    pub bounds_size: f32,
    /// Laplacian smooth passes. 0=off, 1-3=smoother
    pub mesh_smooth_iterations: u32,
    /// Per-pass blend factor 0.0-1.0
    pub mesh_smooth_strength: f32,
    /// Smooth factor at material boundaries. Lower=preserve edges
    pub mesh_boundary_smooth: f32,
    /// Recalculate area-weighted normals. 0=off, 1=on
    pub mesh_recalc_normals: u32,
    /// Grid resolution multiplier for chunks containing exposed ore (1-4).
    /// 1 = base resolution, 2 = 4x triangles, 3 = 9x, 4 = 16x.
    pub ore_detail_multiplier: u32,
    /// Density offset pushing ore surfaces outward (0.0-0.5).
    /// Makes ore deposits physically bulge from cave walls.
    pub ore_protrusion: f32,
}

impl GenerationConfig {
    /// Effective world-space extent per chunk.
    pub fn effective_bounds(&self) -> f32 {
        if self.bounds_size > 0.0 {
            self.bounds_size
        } else {
            self.chunk_size as f32
        }
    }

    /// Ratio of world-space extent to grid voxel count.
    /// Each grid step covers voxel_scale() world units.
    pub fn voxel_scale(&self) -> f32 {
        self.effective_bounds() / self.chunk_size as f32
    }
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
    pub worms_per_region: f32,
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
            pools: PoolConfig::default(),
            mine: MineConfig::default(),
            crystals: CrystalConfig::default(),
            octree_max_depth: 4,
            region_size: 3,
            bounds_size: 0.0,
            mesh_smooth_iterations: 0,
            mesh_smooth_strength: 0.3,
            mesh_boundary_smooth: 0.3,
            mesh_recalc_normals: 1,
            ore_detail_multiplier: 1,
            ore_protrusion: 0.0,
        }
    }
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            cavern_frequency: 0.05,
            cavern_threshold: 0.80,
            detail_octaves: 4,
            detail_persistence: 0.5,
            warp_amplitude: 5.0,
        }
    }
}

impl Default for WormConfig {
    fn default() -> Self {
        Self {
            worms_per_region: 5.0,
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
    /// Coal seams — layered sedimentary deposits
    pub coal: OreVeinParams,
    /// Host rock depth layering
    pub host_rock: HostRockConfig,
    /// Domain warp amplitude for ore shapes. 0=off, 2-5=organic
    pub ore_domain_warp_strength: f64,
    /// Frequency of ore warp noise. Higher=tighter warping
    pub ore_warp_frequency: f64,
    /// Transition zone at ore boundaries. 0=hard, 0.03-0.08=soft
    pub ore_edge_falloff: f64,
    /// Multi-freq detail blend. 0=off, 0.15-0.3=natural
    pub ore_detail_weight: f64,
    // ── Geological realism toggles (all default false) ──
    /// Iron bands only in sandstone/limestone (real BIF geology)
    pub iron_sedimentary_only: bool,
    /// Iron bands thin near surface (real BIF is deep-seated)
    pub iron_depth_fade: bool,
    /// Copper richer at shallow depths (supergene enrichment)
    pub copper_supergene: bool,
    /// Copper concentrates near granite (porphyry deposits)
    pub copper_granite_contact: bool,
    /// Malachite denser near top of range (oxidation front)
    pub malachite_depth_bias: bool,
    /// Kimberlite pipes narrow with depth (carrot shape)
    pub kimberlite_carrot_taper: bool,
    /// Diamond concentration increases with depth
    pub diamond_depth_grade: bool,
    /// Sulfide absent near surface (gossan cap)
    pub sulfide_gossan_cap: bool,
    /// Scattered sulfide halo around main deposits
    pub sulfide_disseminated: bool,
    /// Pyrite clusters near sulfide/copper deposits
    pub pyrite_ore_halo: bool,
    /// Quartz veins become sheet-like (planar fault veins)
    pub quartz_planar_veins: bool,
    /// Gold concentrates in richest vein cores (bonanza zones)
    pub gold_bonanza: bool,
    /// Geodes only in basalt/granite (vesicular lava cavities)
    pub geode_volcanic_host: bool,
    /// Deeper geodes grow thicker crystal shells
    pub geode_depth_scaling: bool,
    /// Coal only forms in sandstone and limestone (real coal is exclusively sedimentary)
    pub coal_sedimentary_host: bool,
    /// Coal thins and vanishes near the surface, modeling erosion of shallow seams
    pub coal_shallow_ceiling: bool,
    /// Coal seams thicken with depth — deeper burial produces thicker, higher-rank deposits
    pub coal_depth_enrichment: bool,
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

    // Mega-Column settings
    pub mega_column_chance: f32,
    pub mega_column_min_gap: usize,
    pub mega_column_radius_min: f32,
    pub mega_column_radius_max: f32,
    pub mega_column_noise_strength: f32,
    pub mega_column_ring_frequency: f32,

    // Drapery settings
    pub drapery_chance: f32,
    pub drapery_length_min: f32,
    pub drapery_length_max: f32,
    pub drapery_wave_frequency: f32,
    pub drapery_wave_amplitude: f32,

    // Rimstone Dam settings
    pub rimstone_chance: f32,
    pub rimstone_dam_height_min: f32,
    pub rimstone_dam_height_max: f32,
    pub rimstone_pool_depth: f32,
    pub rimstone_min_slope: f32,

    // Cave Shield settings
    pub shield_chance: f32,
    pub shield_radius_min: f32,
    pub shield_radius_max: f32,
    pub shield_max_tilt: f32,
    pub shield_stalactite_chance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    /// Master switch for cave pool generation
    pub enabled: bool,
    /// Noise frequency for site selection (world-coherent)
    pub placement_frequency: f64,
    /// Noise threshold for site selection (higher = fewer pools)
    pub placement_threshold: f64,
    /// Per-site RNG roll probability (0-1)
    pub pool_chance: f32,
    /// Minimum floor cells to qualify as a pool site
    pub min_area: usize,
    /// Maximum pool radius in voxels
    pub max_radius: usize,
    /// Voxels carved below floor for basin
    pub basin_depth: usize,
    /// Solid rim height above floor
    pub rim_height: usize,
    /// Probability weight for water pools (normalized with lava_pct + empty_pct)
    pub water_pct: f32,
    /// Probability weight for lava pools (normalized with water_pct + empty_pct)
    pub lava_pct: f32,
    /// Probability weight for empty (skip site entirely)
    pub empty_pct: f32,
    /// Minimum air voxels above pool surface
    pub min_air_above: usize,
    /// Max distance upward to find a solid ceiling; if none found, site is open sky (not a cave)
    pub max_cave_height: usize,
    /// Minimum solid voxels required below basin bottom for structural support
    pub min_floor_thickness: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            placement_frequency: 0.08,
            placement_threshold: 0.75,
            pool_chance: 0.3,
            min_area: 6,
            max_radius: 4,
            basin_depth: 2,
            rim_height: 1,
            water_pct: 0.75,
            lava_pct: 0.25,
            empty_pct: 0.0,
            min_air_above: 3,
            max_cave_height: 20,
            min_floor_thickness: 2,
        }
    }
}

impl Default for FormationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            placement_frequency: 0.15,
            placement_threshold: 0.65,
            stalactite_chance: 0.12,
            stalagmite_chance: 0.10,
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
            // Mega-Column defaults
            mega_column_chance: 0.03,
            mega_column_min_gap: 12,
            mega_column_radius_min: 3.0,
            mega_column_radius_max: 5.0,
            mega_column_noise_strength: 0.3,
            mega_column_ring_frequency: 0.8,
            // Drapery defaults
            drapery_chance: 0.06,
            drapery_length_min: 3.0,
            drapery_length_max: 8.0,
            drapery_wave_frequency: 1.5,
            drapery_wave_amplitude: 0.4,
            // Rimstone Dam defaults
            rimstone_chance: 0.04,
            rimstone_dam_height_min: 1.0,
            rimstone_dam_height_max: 1.5,
            rimstone_pool_depth: 1.0,
            rimstone_min_slope: 0.05,
            // Cave Shield defaults
            shield_chance: 0.008,
            shield_radius_min: 1.5,
            shield_radius_max: 3.0,
            shield_max_tilt: 30.0,
            shield_stalactite_chance: 0.5,
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
            coal: OreVeinParams {
                frequency: 0.03,
                threshold: 0.62,
                depth_min: 10.0,
                depth_max: 80.0,
            },
            host_rock: HostRockConfig::default(),
            ore_domain_warp_strength: 0.0,
            ore_warp_frequency: 0.02,
            ore_edge_falloff: 0.0,
            ore_detail_weight: 0.0,
            iron_sedimentary_only: false,
            iron_depth_fade: false,
            copper_supergene: false,
            copper_granite_contact: false,
            malachite_depth_bias: false,
            kimberlite_carrot_taper: false,
            diamond_depth_grade: false,
            sulfide_gossan_cap: false,
            sulfide_disseminated: false,
            pyrite_ore_halo: false,
            quartz_planar_veins: false,
            gold_bonanza: false,
            geode_volcanic_host: false,
            geode_depth_scaling: false,
            coal_sedimentary_host: false,
            coal_shallow_ceiling: false,
            coal_depth_enrichment: false,
        }
    }
}

