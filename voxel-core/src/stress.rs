use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::density::DensityField;
use crate::material::Material;

/// Serde helper for [f32; 47] (serde doesn't impl Serialize/Deserialize for arrays > 32).
mod serde_f32_array_47 {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    pub fn serialize<S: Serializer>(arr: &[f32; 47], s: S) -> Result<S::Ok, S::Error> {
        arr.as_slice().serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[f32; 47], D::Error> {
        let v: Vec<f32> = Vec::deserialize(d)?;
        v.try_into().map_err(|v: Vec<f32>| serde::de::Error::custom(
            format!("expected 47 elements, got {}", v.len())))
    }
}

/// Per-voxel stress values for a chunk. Same layout as DensityField (17^3 for chunk_size=16).
#[derive(Debug, Clone)]
pub struct StressField {
    pub stress: Vec<f32>,
    pub size: usize,
}

impl StressField {
    pub fn new(size: usize) -> Self {
        Self {
            stress: vec![0.0; size * size * size],
            size,
        }
    }

    #[inline]
    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        z * self.size * self.size + y * self.size + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> f32 {
        self.stress[self.index(x, y, z)]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, val: f32) {
        let idx = self.index(x, y, z);
        self.stress[idx] = val;
    }
}

/// Support type enum (NOT a Material variant).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportType {
    None = 0,
    SlateStrut = 1,
    GraniteStrut = 2,
    LimestoneStrut = 3,
    CopperStrut = 4,
    IronStrut = 5,
    SteelStrut = 6,
    CrystalStrut = 7,
}

impl SupportType {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => SupportType::SlateStrut,
            2 => SupportType::GraniteStrut,
            3 => SupportType::LimestoneStrut,
            4 => SupportType::CopperStrut,
            5 => SupportType::IronStrut,
            6 => SupportType::SteelStrut,
            7 => SupportType::CrystalStrut,
            _ => SupportType::None,
        }
    }
}

/// Per-voxel support data for a chunk.
#[derive(Debug, Clone)]
pub struct SupportField {
    pub supports: Vec<SupportType>,
    pub size: usize,
}

impl SupportField {
    pub fn new(size: usize) -> Self {
        Self {
            supports: vec![SupportType::None; size * size * size],
            size,
        }
    }

    #[inline]
    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        z * self.size * self.size + y * self.size + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> SupportType {
        self.supports[self.index(x, y, z)]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, support_type: SupportType) {
        let idx = self.index(x, y, z);
        self.supports[idx] = support_type;
    }

    #[inline]
    pub fn has_support(&self, x: usize, y: usize, z: usize) -> bool {
        self.get(x, y, z) != SupportType::None
    }
}

/// Default hardness per Material (index by Material as u8).
/// Air = 0.0 (no resistance). Higher = harder to collapse.
pub const DEFAULT_MATERIAL_HARDNESS: [f32; 47] = [
    0.0,   // Air
    0.45,  // Sandstone (soft)
    0.55,  // Limestone
    0.80,  // Granite (hard)
    0.75,  // Basalt
    0.60,  // Slate
    0.65,  // Marble
    0.50,  // Iron
    0.45,  // Copper
    0.40,  // Malachite
    0.40,  // Tin
    0.55,  // Gold
    0.90,  // Diamond
    0.70,  // Kimberlite
    0.50,  // Sulfide
    0.65,  // Quartz
    0.55,  // Pyrite
    0.60,  // Amethyst
    0.70,  // Crystal
    0.30,  // Coal (soft sedimentary)
    0.30,  // Graphite
    0.40,  // Opal
    0.75,  // Hornfels (hard metamorphic)
    0.72,  // Garnet (hard silicate)
    0.65,  // Diopside (calc-silicate)
    0.25,  // Gypsum (soft evaporite)
    0.70,  // Skarn (hard metamorphic)
    0.15,  // Ice (very soft)
    0.35,  // Travertine (soft deposited limestone)
    0.20,  // Permafrost (frozen earth)
    0.10,  // Hoarfrost (delicate crystalline)
    0.25,  // BlackIce (dense ice)
    0.85,  // Obsidian (hard volcanic glass)
    0.20,  // Pumice (lightweight porous)
    0.45,  // Scoria (rough volcanic)
    0.40,  // Sinter (siliceous deposit)
    0.15,  // Sulfur (soft crystalline)
    0.50,  // Flowstone (calcite drape)
    0.10,  // Moonmilk (very soft calcium carbonate)
    0.30,  // Tufa (porous limestone)
    0.05,  // Mycelium (organic, extremely soft)
    0.35,  // Glowstone (luminous mineral)
    0.08,  // MushroomStalk (organic, soft)
    0.05,  // MushroomGill (organic, very soft)
    0.10,  // PurpleCap (organic, soft)
    0.10,  // TealCap (organic, soft)
    0.10,  // AmberCap (organic, soft)
];

/// Support hardness values (how much stress each support type absorbs).
pub const SUPPORT_HARDNESS: [f32; 8] = [
    0.0,   // None
    0.95,  // SlateStrut (Tier 1)
    0.95,  // GraniteStrut (Tier 1)
    0.95,  // LimestoneStrut (Tier 1)
    1.10,  // CopperStrut (Tier 2)
    1.30,  // IronStrut (Tier 3)
    1.50,  // SteelStrut (Tier 4)
    1.80,  // CrystalStrut (Tier 5)
];

// ── StressConfig (moved from voxel-gen) ──

/// Configuration for the structural stress and collapse system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressConfig {
    /// Per-material hardness thresholds (indexed by Material as u8).
    #[serde(with = "serde_f32_array_47")]
    pub material_hardness: [f32; 47],
    /// Weight per solid voxel above (column load factor).
    pub gravity_weight: f32,
    /// Contribution factor for lateral (side) neighbors.
    pub lateral_support_factor: f32,
    /// Contribution factor for voxel directly below.
    pub vertical_support_factor: f32,
    /// Effect radius of support structures.
    pub support_radius: u32,
    /// BFS recalc radius around changed voxels.
    pub propagation_radius: u32,
    /// Maximum voxels per single collapse event.
    pub max_collapse_volume: u32,
    /// Whether rubble placement is enabled.
    pub rubble_enabled: bool,
    /// Fraction of collapsed volume placed as rubble below.
    pub rubble_fill_ratio: f32,
    /// Stress threshold for dust warning (60%).
    pub warn_dust_threshold: f32,
    /// Stress threshold for creak warning (80%).
    pub warn_creak_threshold: f32,
    /// Stress threshold for shake warning (90%).
    pub warn_shake_threshold: f32,
    /// Per-support-type hardness values (indexed by SupportType as u8).
    pub support_hardness: [f32; 8],

    // ── V2 algorithm fields ──

    /// How much support transfers laterally per voxel in ground connectivity pass.
    pub lateral_transfer_factor: f32,
    /// How much support transfers vertically downward per voxel.
    pub vertical_transfer_factor: f32,
    /// Number of relaxation iterations for ground connectivity analysis.
    pub support_propagation_iterations: u32,
    /// Support score above this threshold counts as "grounded".
    pub ground_threshold: f32,
    /// Stress penalty per voxel of air below (overhang penalty).
    pub overhang_weight: f32,
    /// Stress penalty per voxel of unsupported span beyond min_safe_span.
    pub span_weight: f32,
    /// Spans shorter than this get no span penalty.
    pub min_safe_span: u32,
    /// Minimum overstressed region size to trigger collapse.
    pub min_collapse_region: u32,
    /// Voxels with stress >= this threshold are included in slab cohesion expansion.
    pub slab_cohesion_threshold: f32,
}

impl Default for StressConfig {
    fn default() -> Self {
        Self {
            material_hardness: DEFAULT_MATERIAL_HARDNESS,
            gravity_weight: 0.15,
            lateral_support_factor: 0.3,
            vertical_support_factor: 1.0,
            support_radius: 3,
            propagation_radius: 8,
            max_collapse_volume: 200,
            rubble_enabled: true,
            rubble_fill_ratio: 0.4,
            warn_dust_threshold: 0.6,
            warn_creak_threshold: 0.8,
            warn_shake_threshold: 0.9,
            support_hardness: SUPPORT_HARDNESS,
            // V2 defaults
            lateral_transfer_factor: 0.7,
            vertical_transfer_factor: 0.95,
            support_propagation_iterations: 4,
            ground_threshold: 0.8,
            overhang_weight: 0.06,
            span_weight: 0.03,
            min_safe_span: 6,
            min_collapse_region: 4,
            slab_cohesion_threshold: 0.85,
        }
    }
}

// ── Stress calculation types and functions (moved from voxel-ffi) ──

/// A voxel that has been identified as overstressed.
#[derive(Debug, Clone)]
pub struct OverstressedVoxel {
    pub world_x: i32,
    pub world_y: i32,
    pub world_z: i32,
    pub stress: f32,
}

/// Result of a stress recalculation.
#[derive(Debug, Clone)]
pub struct StressResult {
    pub overstressed: Vec<OverstressedVoxel>,
    pub affected_chunks: Vec<(i32, i32, i32)>,
}

/// A single collapsed voxel.
#[derive(Debug, Clone)]
pub struct CollapsedVoxel {
    pub world_x: i32,
    pub world_y: i32,
    pub world_z: i32,
    pub material: Material,
}

/// A rubble voxel placed after collapse.
#[derive(Debug, Clone)]
pub struct RubbleVoxel {
    pub world_x: i32,
    pub world_y: i32,
    pub world_z: i32,
    pub material: Material,
}

/// A collapse event containing all collapsed and rubble voxels.
#[derive(Debug, Clone)]
pub struct CollapseEvent {
    pub collapsed_voxels: Vec<CollapsedVoxel>,
    pub rubble_voxels: Vec<RubbleVoxel>,
    pub affected_chunks: Vec<(i32, i32, i32)>,
    pub center: (f32, f32, f32),
    pub volume: u32,
}

// ── V2 collapse types ──

/// A contiguous slab of voxels that collapses as a unit.
#[derive(Debug, Clone)]
pub struct CollapseSlab {
    /// All voxels in this slab (world coords + original material).
    pub voxels: Vec<CollapsedVoxel>,
    /// Bounding box min corner (world coords).
    pub bb_min: (i32, i32, i32),
    /// Bounding box max corner (world coords).
    pub bb_max: (i32, i32, i32),
    /// Center of mass (world coords).
    pub center: (f32, f32, f32),
    /// Y position where the bottom of the slab lands after falling.
    pub landing_y: i32,
    /// Fall distance in voxels.
    pub fall_distance: i32,
    /// Most common material in the slab.
    pub dominant_material: Material,
}

/// Enhanced collapse event with coherent slab data for animated falling.
#[derive(Debug, Clone)]
pub struct CollapseEventV2 {
    pub slabs: Vec<CollapseSlab>,
    pub affected_chunks: Vec<(i32, i32, i32)>,
    pub total_volume: u32,
    pub center: (f32, f32, f32),
}

/// Per-voxel support score field for a chunk (ground connectivity pass output).
#[derive(Debug, Clone)]
pub struct SupportScoreField {
    pub scores: Vec<f32>,
    pub size: usize,
}

impl SupportScoreField {
    pub fn new(size: usize) -> Self {
        Self {
            scores: vec![0.0; size * size * size],
            size,
        }
    }

    #[inline]
    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        z * self.size * self.size + y * self.size + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> f32 {
        self.scores[self.index(x, y, z)]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, val: f32) {
        let idx = self.index(x, y, z);
        self.scores[idx] = val;
    }
}

/// Convert world coordinate to (chunk_key, local_coord).
pub fn world_to_chunk_local(wx: i32, wy: i32, wz: i32, chunk_size: usize) -> ((i32, i32, i32), usize, usize, usize) {
    let cs = chunk_size as i32;
    let cx = wx.div_euclid(cs);
    let cy = wy.div_euclid(cs);
    let cz = wz.div_euclid(cs);
    let lx = wx.rem_euclid(cs) as usize;
    let ly = wy.rem_euclid(cs) as usize;
    let lz = wz.rem_euclid(cs) as usize;
    ((cx, cy, cz), lx, ly, lz)
}

/// Sample density from world coordinates, looking up the correct chunk.
/// Returns None if the chunk is not loaded (treated as solid by caller).
fn sample_world(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    wx: i32, wy: i32, wz: i32,
    chunk_size: usize,
) -> Option<(f32, Material)> {
    let (key, lx, ly, lz) = world_to_chunk_local(wx, wy, wz, chunk_size);
    density_fields.get(&key).map(|df| {
        let sample = df.get(lx, ly, lz);
        (sample.density, sample.material)
    })
}

/// Sample support type from world coordinates, looking up the correct chunk.
fn sample_support(
    support_fields: &HashMap<(i32, i32, i32), SupportField>,
    wx: i32, wy: i32, wz: i32,
    chunk_size: usize,
) -> SupportType {
    let (key, lx, ly, lz) = world_to_chunk_local(wx, wy, wz, chunk_size);
    support_fields
        .get(&key)
        .map(|sf| sf.get(lx, ly, lz))
        .unwrap_or(SupportType::None)
}

/// Count contiguous solid voxels above (Y+) a position, capped at 32.
fn column_weight_above(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    wx: i32, wy: i32, wz: i32,
    chunk_size: usize,
) -> u32 {
    let mut count = 0u32;
    for dy in 1..=32i32 {
        let sy = wy + dy;
        match sample_world(density_fields, wx, sy, wz, chunk_size) {
            Some((_, mat)) => {
                if mat.is_solid() {
                    count += 1;
                } else {
                    break;
                }
            }
            // Unloaded = treat as solid (conservative)
            None => count += 1,
        }
    }
    count
}

/// Calculate stress for a single voxel at world coordinates.
pub fn calc_voxel_stress(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    support_fields: &HashMap<(i32, i32, i32), SupportField>,
    config: &StressConfig,
    wx: i32, wy: i32, wz: i32,
    chunk_size: usize,
) -> f32 {
    // Only solid voxels have stress
    let (_, mat) = match sample_world(density_fields, wx, wy, wz, chunk_size) {
        Some((d, m)) => (d, m),
        None => return 0.0, // Unloaded
    };
    if !mat.is_solid() {
        return 0.0;
    }

    let hardness = config.material_hardness[mat as u8 as usize];
    if hardness <= 0.0 {
        return 0.0;
    }

    // 1. Column weight: gravity from above
    let weight = column_weight_above(density_fields, wx, wy, wz, chunk_size);
    let mut raw_stress = weight as f32 * config.gravity_weight;

    // 2. Support reduction from direct neighbors
    // Voxel below reduces stress
    match sample_world(density_fields, wx, wy - 1, wz, chunk_size) {
        Some((_, m)) if m.is_solid() => {
            raw_stress -= config.vertical_support_factor;
        }
        None => {
            // Unloaded = treat as solid support (conservative)
            raw_stress -= config.vertical_support_factor;
        }
        _ => {}
    }

    // 6-connected lateral neighbors reduce stress
    let lateral_offsets: [(i32, i32, i32); 4] = [
        (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1),
    ];
    for (dx, dy, dz) in &lateral_offsets {
        let nx = wx + dx;
        let ny = wy + dy;
        let nz = wz + dz;
        match sample_world(density_fields, nx, ny, nz, chunk_size) {
            Some((_, m)) if m.is_solid() => {
                raw_stress -= config.lateral_support_factor;
            }
            None => {
                raw_stress -= config.lateral_support_factor;
            }
            _ => {}
        }
    }

    // 3. Support structure bonus: nearby supports reduce stress
    let sr = config.support_radius as i32;
    for dz in -sr..=sr {
        for dy in -sr..=sr {
            for dx in -sr..=sr {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                let support = sample_support(support_fields, wx + dx, wy + dy, wz + dz, chunk_size);
                if support != SupportType::None {
                    let dist = ((dx * dx + dy * dy + dz * dz) as f32).sqrt();
                    let support_value = config.support_hardness[support as u8 as usize];
                    raw_stress -= support_value / dist;
                }
            }
        }
    }

    // Clamp to non-negative before normalization
    raw_stress = raw_stress.max(0.0);

    // 4. Normalize by material hardness
    raw_stress / hardness
}

// ── V2 stress algorithm: two-pass ground connectivity + load accumulation ──

/// Count contiguous air voxels below a position (Y−), capped at 32.
fn count_air_below(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    wx: i32, wy: i32, wz: i32,
    chunk_size: usize,
) -> u32 {
    let mut count = 0u32;
    for dy in 1..=32i32 {
        let sy = wy - dy;
        match sample_world(density_fields, wx, sy, wz, chunk_size) {
            Some((_, mat)) if mat.is_solid() => break,
            None => break, // Unloaded = assume solid (conservative)
            _ => count += 1,
        }
    }
    count
}

/// Find minimum lateral distance to a "grounded" voxel (support_score >= threshold).
/// Searches in 4 cardinal directions (X+, X−, Z+, Z−) up to max_dist.
/// Returns the minimum distance found, or max_dist if none found.
fn min_lateral_distance_to_grounded(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    support_scores: &HashMap<(i32, i32, i32), SupportScoreField>,
    wx: i32, wy: i32, wz: i32,
    chunk_size: usize,
    ground_threshold: f32,
    max_dist: u32,
) -> u32 {
    let directions: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    let mut min_dist = max_dist;

    for &(dx, dz) in &directions {
        for d in 1..=max_dist as i32 {
            let nx = wx + dx * d;
            let nz = wz + dz * d;
            // Must be solid to be a grounded support
            match sample_world(density_fields, nx, wy, nz, chunk_size) {
                Some((_, mat)) if mat.is_solid() => {
                    // Check support score
                    let (key, lx, ly, lz) = world_to_chunk_local(nx, wy, nz, chunk_size);
                    let score = support_scores
                        .get(&key)
                        .map(|sf| sf.get(lx, ly, lz))
                        .unwrap_or(1.0); // Unloaded = assume grounded
                    if score >= ground_threshold {
                        min_dist = min_dist.min(d as u32);
                        break;
                    }
                }
                Some(_) => break, // Hit air, stop this direction
                None => {
                    // Unloaded = assume grounded at this distance
                    min_dist = min_dist.min(d as u32);
                    break;
                }
            }
        }
    }
    min_dist
}

/// Pass 1: Ground connectivity analysis via iterative relaxation.
///
/// For each solid voxel in the specified chunks, computes a `support_score` in [0.0, 1.0]:
/// - 1.0 = directly grounded (solid voxel below all the way down)
/// - 0.0 = completely unsupported (floating)
///
/// Support propagates vertically (0.95 per voxel) and laterally (0.7 per voxel)
/// over multiple relaxation iterations, modeling how walls and pillars support ceilings.
pub fn ground_connectivity_pass(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    chunk_keys: &[(i32, i32, i32)],
    chunk_size: usize,
    config: &StressConfig,
) -> HashMap<(i32, i32, i32), SupportScoreField> {
    // Expand chunk set to include immediate neighbors (needed for boundary propagation)
    let mut expanded_keys: HashSet<(i32, i32, i32)> = HashSet::new();
    for &(cx, cy, cz) in chunk_keys {
        for dz in -1..=1i32 {
            for dy in -1..=1i32 {
                for dx in -1..=1i32 {
                    let key = (cx + dx, cy + dy, cz + dz);
                    if density_fields.contains_key(&key) {
                        expanded_keys.insert(key);
                    }
                }
            }
        }
    }

    let cs = chunk_size;
    let grid_size = cs + 1; // DensityField is (chunk_size+1)^3

    // Initialize support scores
    let mut scores: HashMap<(i32, i32, i32), SupportScoreField> = HashMap::new();
    for &key in &expanded_keys {
        let mut sf = SupportScoreField::new(grid_size);
        if let Some(df) = density_fields.get(&key) {
            let (cx, cy, cz) = key;
            for z in 0..grid_size {
                for y in 0..grid_size {
                    for x in 0..grid_size {
                        let sample = df.get(x, y, z);
                        if !sample.material.is_solid() {
                            continue; // Air voxels have 0 score
                        }
                        // Check if voxel below is solid
                        let wx = cx * cs as i32 + x as i32;
                        let wy = cy * cs as i32 + y as i32;
                        let wz = cz * cs as i32 + z as i32;
                        let below_solid = match sample_world(density_fields, wx, wy - 1, wz, cs) {
                            Some((_, mat)) => mat.is_solid(),
                            None => true, // Unloaded = assume solid (conservative grounding)
                        };
                        if below_solid {
                            sf.set(x, y, z, 1.0);
                        }
                    }
                }
            }
        }
        scores.insert(key, sf);
    }

    // Iterative relaxation
    let vert_transfer = config.vertical_transfer_factor;
    let lat_transfer = config.lateral_transfer_factor;

    for _iter in 0..config.support_propagation_iterations {
        // We need to read neighbor scores from the previous iteration,
        // so collect updates first, then apply.
        let mut updates: Vec<((i32, i32, i32), usize, usize, usize, f32)> = Vec::new();

        for &key in &expanded_keys {
            let df = match density_fields.get(&key) {
                Some(d) => d,
                None => continue,
            };
            let (cx, cy, cz) = key;

            for z in 0..grid_size {
                for y in 0..grid_size {
                    for x in 0..grid_size {
                        if !df.get(x, y, z).material.is_solid() {
                            continue;
                        }
                        let current_score = scores.get(&key).unwrap().get(x, y, z);
                        if current_score >= 1.0 {
                            continue; // Already fully grounded
                        }

                        let wx = cx * cs as i32 + x as i32;
                        let wy = cy * cs as i32 + y as i32;
                        let wz = cz * cs as i32 + z as i32;

                        let mut best = current_score;

                        // Vertical transfer from below
                        let (bkey, blx, bly, blz) = world_to_chunk_local(wx, wy - 1, wz, cs);
                        if let Some(bsf) = scores.get(&bkey) {
                            let below_score = bsf.get(blx, bly, blz);
                            best = best.max(below_score * vert_transfer);
                        } else {
                            // Unloaded neighbor = assume grounded
                            best = best.max(vert_transfer);
                        }

                        // Lateral transfer from 4 horizontal neighbors
                        let neighbors: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
                        for &(dx, dz) in &neighbors {
                            let (nkey, nlx, nly, nlz) = world_to_chunk_local(
                                wx + dx, wy, wz + dz, cs,
                            );
                            if let Some(nsf) = scores.get(&nkey) {
                                let n_score = nsf.get(nlx, nly, nlz);
                                best = best.max(n_score * lat_transfer);
                            }
                            // Don't assume grounded for unloaded lateral neighbors
                        }

                        // Also check vertical transfer from above (for hanging structures)
                        let (akey, alx, aly, alz) = world_to_chunk_local(wx, wy + 1, wz, cs);
                        if let Some(asf) = scores.get(&akey) {
                            let above_score = asf.get(alx, aly, alz);
                            // Upward transfer is weaker — hanging structures lose support fast
                            best = best.max(above_score * lat_transfer * 0.5);
                        }

                        if best > current_score + 0.001 {
                            updates.push((key, x, y, z, best.min(1.0)));
                        }
                    }
                }
            }
        }

        // Apply updates
        if updates.is_empty() {
            break; // Converged early
        }
        for (key, x, y, z, val) in updates {
            if let Some(sf) = scores.get_mut(&key) {
                sf.set(x, y, z, val);
            }
        }
    }

    scores
}

/// V2 stress calculation for a single voxel using precomputed ground connectivity.
pub fn calc_voxel_stress_v2(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    support_fields: &HashMap<(i32, i32, i32), SupportField>,
    support_scores: &HashMap<(i32, i32, i32), SupportScoreField>,
    config: &StressConfig,
    wx: i32, wy: i32, wz: i32,
    chunk_size: usize,
) -> f32 {
    // Only solid voxels have stress
    let mat = match sample_world(density_fields, wx, wy, wz, chunk_size) {
        Some((_, m)) if m.is_solid() => m,
        _ => return 0.0,
    };

    let hardness = config.material_hardness[mat as u8 as usize];
    if hardness <= 0.0 {
        return 0.0;
    }

    // Get support score from ground connectivity pass
    let (key, lx, ly, lz) = world_to_chunk_local(wx, wy, wz, chunk_size);
    let support_score = support_scores
        .get(&key)
        .map(|sf| sf.get(lx, ly, lz))
        .unwrap_or(1.0); // Unloaded = assume grounded

    // Unsupported factor: 0.0 for fully grounded, 1.0 for floating.
    // All load penalties are scaled by this — grounded voxels experience near-zero stress.
    let unsupported = (1.0 - support_score).max(0.0);

    // 1. Effective load above: only count solid voxels above until hitting a
    //    well-supported voxel. This gives "local ceiling thickness" — the mass
    //    that would actually fall, not the entire mountain above a small tunnel.
    let effective_load = {
        let mut count = 0u32;
        for dy in 1..=32i32 {
            let sy = wy + dy;
            match sample_world(density_fields, wx, sy, wz, chunk_size) {
                Some((_, m)) if m.is_solid() => {
                    // Check if this voxel above is well-grounded
                    let (ak, alx, aly, alz) = world_to_chunk_local(wx, sy, wz, chunk_size);
                    let above_score = support_scores
                        .get(&ak)
                        .map(|sf| sf.get(alx, aly, alz))
                        .unwrap_or(1.0);
                    if above_score >= config.ground_threshold {
                        break; // Hit a grounded layer, stop counting
                    }
                    count += 1;
                }
                Some(_) => break, // Air above
                None => break,    // Unloaded = stop counting
            }
        }
        count
    };
    let mut raw_stress = effective_load as f32 * config.gravity_weight * unsupported;

    // 2. Overhang penalty: count air below, scaled by unsupported factor
    let air_below = count_air_below(density_fields, wx, wy, wz, chunk_size);
    raw_stress += air_below as f32 * config.overhang_weight * unsupported;

    // 3. Span penalty: distance to nearest grounded voxel, scaled by unsupported factor
    let span_dist = min_lateral_distance_to_grounded(
        density_fields, support_scores, wx, wy, wz, chunk_size,
        config.ground_threshold, 20, // max search distance
    );
    if span_dist > config.min_safe_span {
        raw_stress += (span_dist - config.min_safe_span) as f32 * config.span_weight * unsupported;
    }

    // 4. Support structure bonus: nearby struts reduce stress
    let sr = config.support_radius as i32;
    for dz in -sr..=sr {
        for dy in -sr..=sr {
            for dx in -sr..=sr {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                let support = sample_support(support_fields, wx + dx, wy + dy, wz + dz, chunk_size);
                if support != SupportType::None {
                    let dist = ((dx * dx + dy * dy + dz * dz) as f32).sqrt();
                    let support_value = config.support_hardness[support as u8 as usize];
                    raw_stress -= support_value / dist;
                }
            }
        }
    }

    // Clamp to non-negative, normalize by material hardness
    raw_stress.max(0.0) / hardness
}

/// V2 stress recalculation: runs ground connectivity pass then per-voxel stress.
/// Operates on a set of dirty chunks (and their neighborhoods).
pub fn recalc_stress_region_v2(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    stress_fields: &mut HashMap<(i32, i32, i32), StressField>,
    support_fields: &HashMap<(i32, i32, i32), SupportField>,
    config: &StressConfig,
    dirty_chunks: &[(i32, i32, i32)],
    chunk_size: usize,
) -> StressResult {
    // Pass 1: ground connectivity on dirty chunks + neighbors
    let support_scores = ground_connectivity_pass(density_fields, dirty_chunks, chunk_size, config);

    let cs = chunk_size;
    let grid_size = cs + 1;
    let mut overstressed = Vec::new();
    let mut affected_chunks = HashSet::new();

    // Pass 2: calculate stress for all voxels in dirty chunks
    for &(cx, cy, cz) in dirty_chunks {
        let df = match density_fields.get(&(cx, cy, cz)) {
            Some(d) => d,
            None => continue,
        };

        for z in 0..grid_size {
            for y in 0..grid_size {
                for x in 0..grid_size {
                    if !df.get(x, y, z).material.is_solid() {
                        // Air voxels have 0 stress
                        if let Some(sf) = stress_fields.get_mut(&(cx, cy, cz)) {
                            sf.set(x, y, z, 0.0);
                        }
                        continue;
                    }

                    let wx = cx * cs as i32 + x as i32;
                    let wy = cy * cs as i32 + y as i32;
                    let wz = cz * cs as i32 + z as i32;

                    let stress = calc_voxel_stress_v2(
                        density_fields, support_fields, &support_scores,
                        config, wx, wy, wz, cs,
                    );

                    if let Some(sf) = stress_fields.get_mut(&(cx, cy, cz)) {
                        sf.set(x, y, z, stress);
                        affected_chunks.insert((cx, cy, cz));
                    }

                    if stress >= 1.0 {
                        overstressed.push(OverstressedVoxel {
                            world_x: wx, world_y: wy, world_z: wz, stress,
                        });
                    }
                }
            }
        }
    }

    StressResult {
        overstressed,
        affected_chunks: affected_chunks.into_iter().collect(),
    }
}

/// Recalculate stress in a region around a changed world position.
/// Returns the list of overstressed voxels and affected chunks.
pub fn recalc_stress_region(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    stress_fields: &mut HashMap<(i32, i32, i32), StressField>,
    support_fields: &HashMap<(i32, i32, i32), SupportField>,
    config: &StressConfig,
    changed_world_pos: (i32, i32, i32),
    radius: u32,
    chunk_size: usize,
) -> StressResult {
    let (cwx, cwy, cwz) = changed_world_pos;
    let r = radius as i32;
    let mut overstressed = Vec::new();
    let mut affected_chunks = HashSet::new();

    for dz in -r..=r {
        for dy in -r..=r {
            for dx in -r..=r {
                let wx = cwx + dx;
                let wy = cwy + dy;
                let wz = cwz + dz;

                let stress = calc_voxel_stress(
                    density_fields, support_fields, config, wx, wy, wz, chunk_size,
                );

                // Store stress value
                let (key, lx, ly, lz) = world_to_chunk_local(wx, wy, wz, chunk_size);
                if let Some(sf) = stress_fields.get_mut(&key) {
                    sf.set(lx, ly, lz, stress);
                    affected_chunks.insert(key);
                }

                // Check for overstress
                if stress >= 1.0 {
                    // Verify this is actually a solid voxel
                    if let Some((_, mat)) = sample_world(density_fields, wx, wy, wz, chunk_size) {
                        if mat.is_solid() {
                            overstressed.push(OverstressedVoxel {
                                world_x: wx,
                                world_y: wy,
                                world_z: wz,
                                stress,
                            });
                        }
                    }
                }
            }
        }
    }

    StressResult {
        overstressed,
        affected_chunks: affected_chunks.into_iter().collect(),
    }
}

/// Detect contiguous overstressed regions via flood-fill (6-connected BFS)
/// and execute collapses: convert to Air, place rubble, mark dirty chunks.
pub fn detect_and_execute_collapses(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    stress_fields: &mut HashMap<(i32, i32, i32), StressField>,
    _support_fields: &HashMap<(i32, i32, i32), SupportField>,
    overstressed: &[OverstressedVoxel],
    config: &StressConfig,
    chunk_size: usize,
) -> Vec<CollapseEvent> {
    if overstressed.is_empty() {
        return Vec::new();
    }

    let mut visited: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut events = Vec::new();

    // Build set for quick lookup
    let overstressed_set: HashSet<(i32, i32, i32)> = overstressed
        .iter()
        .map(|v| (v.world_x, v.world_y, v.world_z))
        .collect();

    for ov in overstressed {
        let start = (ov.world_x, ov.world_y, ov.world_z);
        if visited.contains(&start) {
            continue;
        }

        // BFS flood-fill to find contiguous overstressed region
        let mut queue = VecDeque::new();
        let mut region: Vec<(i32, i32, i32)> = Vec::new();
        queue.push_back(start);
        visited.insert(start);

        while let Some(pos) = queue.pop_front() {
            if region.len() >= config.max_collapse_volume as usize {
                break;
            }
            region.push(pos);

            // Check 6-connected neighbors
            let offsets: [(i32, i32, i32); 6] = [
                (1, 0, 0), (-1, 0, 0),
                (0, 1, 0), (0, -1, 0),
                (0, 0, 1), (0, 0, -1),
            ];
            for (dx, dy, dz) in &offsets {
                let neighbor = (pos.0 + dx, pos.1 + dy, pos.2 + dz);
                if !visited.contains(&neighbor) && overstressed_set.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }

        if region.is_empty() {
            continue;
        }

        // Calculate center
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_z = 0.0f32;
        for &(x, y, z) in &region {
            sum_x += x as f32;
            sum_y += y as f32;
            sum_z += z as f32;
        }
        let n = region.len() as f32;
        let center = (sum_x / n, sum_y / n, sum_z / n);

        // Execute collapse: convert voxels to Air
        let mut collapsed_voxels = Vec::with_capacity(region.len());
        let mut affected_chunks_set = HashSet::new();

        for &(wx, wy, wz) in &region {
            let (key, lx, ly, lz) = world_to_chunk_local(wx, wy, wz, chunk_size);

            // Get original material before clearing
            let material = if let Some(df) = density_fields.get(&key) {
                df.get(lx, ly, lz).material
            } else {
                Material::Air
            };

            // Set to Air
            if let Some(df) = density_fields.get_mut(&key) {
                let sample = df.get_mut(lx, ly, lz);
                sample.density = -1.0;
                sample.material = Material::Air;
            }

            // Clear stress
            if let Some(sf) = stress_fields.get_mut(&key) {
                sf.set(lx, ly, lz, 0.0);
            }

            affected_chunks_set.insert(key);
            collapsed_voxels.push(CollapsedVoxel {
                world_x: wx,
                world_y: wy,
                world_z: wz,
                material,
            });
        }

        // Place rubble below collapsed region
        let mut rubble_voxels = Vec::new();
        if config.rubble_enabled {
            let rubble_count = (region.len() as f32 * config.rubble_fill_ratio) as usize;
            let mut placed = 0;

            for cv in &collapsed_voxels {
                if placed >= rubble_count {
                    break;
                }
                // Trace downward to find first solid surface
                let mut ry = cv.world_y - 1;
                let mut found_surface = false;
                for _ in 0..32 {
                    match sample_world(density_fields, cv.world_x, ry, cv.world_z, chunk_size) {
                        Some((_, mat)) if mat.is_solid() => {
                            // Place rubble one above solid surface
                            ry += 1;
                            found_surface = true;
                            break;
                        }
                        None => {
                            // Unloaded, stop
                            break;
                        }
                        _ => {
                            ry -= 1;
                        }
                    }
                }

                if found_surface && ry < cv.world_y {
                    let (rkey, rlx, rly, rlz) = world_to_chunk_local(
                        cv.world_x, ry, cv.world_z, chunk_size,
                    );
                    // Only place rubble in air voxels
                    let is_air = density_fields
                        .get(&rkey)
                        .map(|df| !df.get(rlx, rly, rlz).material.is_solid())
                        .unwrap_or(false);

                    if is_air {
                        if let Some(df) = density_fields.get_mut(&rkey) {
                            let sample = df.get_mut(rlx, rly, rlz);
                            sample.density = 1.0;
                            sample.material = cv.material;
                        }
                        affected_chunks_set.insert(rkey);
                        rubble_voxels.push(RubbleVoxel {
                            world_x: cv.world_x,
                            world_y: ry,
                            world_z: cv.world_z,
                            material: cv.material,
                        });
                        placed += 1;
                    }
                }
            }
        }

        events.push(CollapseEvent {
            volume: collapsed_voxels.len() as u32,
            collapsed_voxels,
            rubble_voxels,
            affected_chunks: affected_chunks_set.into_iter().collect(),
            center,
        });
    }

    events
}

// ── V2 collapse detection: coherent slab collapse ──

/// Detect contiguous overstressed regions and produce coherent falling slabs.
///
/// Key improvements over v1:
/// - Slab cohesion expansion: includes nearly-overstressed neighbors (>= slab_cohesion_threshold)
///   to prevent ragged holes
/// - Minimum region filter: skips tiny regions (< min_collapse_region)
/// - Uniform slab translation: entire slab drops as a unit, preserving shape
/// - Rubble preserves slab geometry at landing position
pub fn detect_and_execute_collapses_v2(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    stress_fields: &mut HashMap<(i32, i32, i32), StressField>,
    _support_fields: &HashMap<(i32, i32, i32), SupportField>,
    overstressed: &[OverstressedVoxel],
    config: &StressConfig,
    chunk_size: usize,
) -> Vec<CollapseEventV2> {
    if overstressed.is_empty() {
        return Vec::new();
    }

    let mut visited: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut events = Vec::new();

    // Build set and stress lookup for quick access
    let overstressed_set: HashSet<(i32, i32, i32)> = overstressed
        .iter()
        .map(|v| (v.world_x, v.world_y, v.world_z))
        .collect();

    for ov in overstressed {
        let start = (ov.world_x, ov.world_y, ov.world_z);
        if visited.contains(&start) {
            continue;
        }

        // BFS flood-fill: find contiguous region of overstressed + nearly-overstressed voxels
        let mut queue = VecDeque::new();
        let mut region: Vec<(i32, i32, i32)> = Vec::new();
        queue.push_back(start);
        visited.insert(start);

        while let Some(pos) = queue.pop_front() {
            if region.len() >= config.max_collapse_volume as usize {
                break;
            }
            region.push(pos);

            let offsets: [(i32, i32, i32); 6] = [
                (1, 0, 0), (-1, 0, 0),
                (0, 1, 0), (0, -1, 0),
                (0, 0, 1), (0, 0, -1),
            ];
            for (dx, dy, dz) in &offsets {
                let neighbor = (pos.0 + dx, pos.1 + dy, pos.2 + dz);
                if visited.contains(&neighbor) {
                    continue;
                }

                // Include if overstressed OR if solid with stress >= cohesion threshold
                let include = if overstressed_set.contains(&neighbor) {
                    true
                } else {
                    // Check if solid and nearly overstressed (cohesion expansion)
                    let (nkey, nlx, nly, nlz) = world_to_chunk_local(
                        neighbor.0, neighbor.1, neighbor.2, chunk_size,
                    );
                    let is_solid = density_fields
                        .get(&nkey)
                        .map(|df| df.get(nlx, nly, nlz).material.is_solid())
                        .unwrap_or(false);
                    let stress_val = stress_fields
                        .get(&nkey)
                        .map(|sf| sf.get(nlx, nly, nlz))
                        .unwrap_or(0.0);
                    is_solid && stress_val >= config.slab_cohesion_threshold
                };

                if include {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }

        // Minimum region filter
        if (region.len() as u32) < config.min_collapse_region {
            continue;
        }

        // Compute bounding box and center
        let mut bb_min = (i32::MAX, i32::MAX, i32::MAX);
        let mut bb_max = (i32::MIN, i32::MIN, i32::MIN);
        let mut sum = (0.0f32, 0.0f32, 0.0f32);
        let mut material_counts: HashMap<Material, u32> = HashMap::new();

        for &(x, y, z) in &region {
            bb_min.0 = bb_min.0.min(x);
            bb_min.1 = bb_min.1.min(y);
            bb_min.2 = bb_min.2.min(z);
            bb_max.0 = bb_max.0.max(x);
            bb_max.1 = bb_max.1.max(y);
            bb_max.2 = bb_max.2.max(z);
            sum.0 += x as f32;
            sum.1 += y as f32;
            sum.2 += z as f32;

            let (key, lx, ly, lz) = world_to_chunk_local(x, y, z, chunk_size);
            if let Some(df) = density_fields.get(&key) {
                let mat = df.get(lx, ly, lz).material;
                *material_counts.entry(mat).or_insert(0) += 1;
            }
        }
        let n = region.len() as f32;
        let center = (sum.0 / n, sum.1 / n, sum.2 / n);

        let dominant_material = material_counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(mat, _)| mat)
            .unwrap_or(Material::Granite);

        // Compute landing position: for each (x, z) column in the slab,
        // trace down from the slab's bottom-most voxel in that column
        // to find the highest floor surface. The slab landing_y is the
        // maximum across all columns (so the slab sits on the highest floor).
        let region_set: HashSet<(i32, i32, i32)> = region.iter().copied().collect();
        let mut column_min_y: HashMap<(i32, i32), i32> = HashMap::new();
        for &(x, y, z) in &region {
            let entry = column_min_y.entry((x, z)).or_insert(y);
            *entry = (*entry).min(y);
        }

        let mut landing_offset = i32::MAX; // How far the slab drops (min across columns)
        for (&(x, z), &min_y) in &column_min_y {
            // Trace down from one below the slab bottom in this column
            let mut floor_y = min_y - 1;
            let mut found = false;
            for _ in 0..64 {
                // Don't count other slab voxels as floor
                if region_set.contains(&(x, floor_y, z)) {
                    floor_y -= 1;
                    continue;
                }
                match sample_world(density_fields, x, floor_y, z, chunk_size) {
                    Some((_, mat)) if mat.is_solid() => {
                        // Floor found: slab bottom rests one above this
                        let this_offset = min_y - (floor_y + 1);
                        landing_offset = landing_offset.min(this_offset);
                        found = true;
                        break;
                    }
                    None => {
                        // Unloaded = assume floor here
                        let this_offset = min_y - (floor_y + 1);
                        landing_offset = landing_offset.min(this_offset);
                        found = true;
                        break;
                    }
                    _ => floor_y -= 1,
                }
            }
            if !found {
                // No floor found within 64 voxels, use max trace distance
                landing_offset = landing_offset.min(min_y - (min_y - 64));
            }
        }

        if landing_offset <= 0 {
            continue; // Slab is already grounded, no fall
        }

        let landing_y = bb_min.1 - landing_offset;

        // Record collapsed voxels with original material, then clear them
        let mut collapsed_voxels = Vec::with_capacity(region.len());
        let mut affected_chunks_set = HashSet::new();

        for &(wx, wy, wz) in &region {
            let (key, lx, ly, lz) = world_to_chunk_local(wx, wy, wz, chunk_size);

            let material = density_fields
                .get(&key)
                .map(|df| df.get(lx, ly, lz).material)
                .unwrap_or(Material::Air);

            // Set to Air (remove the slab from its original position)
            if let Some(df) = density_fields.get_mut(&key) {
                let sample = df.get_mut(lx, ly, lz);
                sample.density = -1.0;
                sample.material = Material::Air;
            }
            if let Some(sf) = stress_fields.get_mut(&key) {
                sf.set(lx, ly, lz, 0.0);
            }
            affected_chunks_set.insert(key);

            collapsed_voxels.push(CollapsedVoxel {
                world_x: wx, world_y: wy, world_z: wz, material,
            });
        }

        // Place rubble as a natural mound at landing position.
        // Instead of 1:1 translation (which creates flat block copies), we build a
        // mound shape: tallest at center, tapering toward edges, with ~70% fill.
        {
            let cx_f = center.0;
            let cz_f = center.2;
            // Compute max radius of the collapse footprint
            let max_radius = {
                let dx = (bb_max.0 - bb_min.0) as f32 * 0.5;
                let dz = (bb_max.2 - bb_min.2) as f32 * 0.5;
                dx.max(dz).max(1.0)
            };
            // Slab thickness = how many Y layers the slab spans
            let slab_thickness = (bb_max.1 - bb_min.1 + 1).max(1) as f32;

            // Target rubble fill: 70% of collapsed volume, capped
            let target_fill = (collapsed_voxels.len() as f32 * config.rubble_fill_ratio).ceil() as usize;
            let mut placed = 0usize;

            // Sort voxels by distance to center (place center first for mound shape)
            let mut sorted_voxels: Vec<&CollapsedVoxel> = collapsed_voxels.iter()
                .filter(|cv| cv.material != Material::Air)
                .collect();
            sorted_voxels.sort_by(|a, b| {
                let da = (a.world_x as f32 - cx_f).powi(2) + (a.world_z as f32 - cz_f).powi(2);
                let db = (b.world_x as f32 - cx_f).powi(2) + (b.world_z as f32 - cz_f).powi(2);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });

            for cv in &sorted_voxels {
                if placed >= target_fill {
                    break;
                }

                // Distance from center as 0..1
                let dist_xz = ((cv.world_x as f32 - cx_f).powi(2) + (cv.world_z as f32 - cz_f).powi(2)).sqrt();
                let norm_dist = (dist_xz / max_radius).min(1.0);

                // Mound height: tallest at center, tapers to 0 at edges
                // Uses a smooth falloff: (1 - dist^1.5) * thickness * 0.6
                let mound_height = ((1.0 - norm_dist.powf(1.5)) * slab_thickness * 0.6).round() as i32;

                // Place at floor level + mound height offset
                let base_y = bb_min.1 - landing_offset; // floor level
                let place_y = base_y + mound_height.max(0);

                let (rkey, rlx, rly, rlz) = world_to_chunk_local(
                    cv.world_x, place_y, cv.world_z, chunk_size,
                );

                let is_air = density_fields
                    .get(&rkey)
                    .map(|df| !df.get(rlx, rly, rlz).material.is_solid())
                    .unwrap_or(false);

                if is_air {
                    if let Some(df) = density_fields.get_mut(&rkey) {
                        let sample = df.get_mut(rlx, rly, rlz);
                        sample.density = 1.0;
                        sample.material = cv.material;
                    }
                    affected_chunks_set.insert(rkey);
                    placed += 1;
                }
            }
        }

        let slab = CollapseSlab {
            voxels: collapsed_voxels,
            bb_min,
            bb_max,
            center,
            landing_y,
            fall_distance: landing_offset,
            dominant_material,
        };

        events.push(CollapseEventV2 {
            slabs: vec![slab],
            affected_chunks: affected_chunks_set.into_iter().collect(),
            total_volume: region.len() as u32,
            center,
        });
    }

    events
}

/// V2 post-change stress update: runs ground connectivity + collapse detection with cascade.
pub fn post_change_stress_update_v2(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    stress_fields: &mut HashMap<(i32, i32, i32), StressField>,
    support_fields: &HashMap<(i32, i32, i32), SupportField>,
    config: &StressConfig,
    dirty_chunks: &[(i32, i32, i32)],
    chunk_size: usize,
    max_iterations: u32,
) -> (Vec<CollapseEventV2>, HashSet<(i32, i32, i32)>) {
    let mut all_events = Vec::new();
    let mut all_dirty_chunks = HashSet::new();
    let mut current_dirty: Vec<(i32, i32, i32)> = dirty_chunks.to_vec();

    for _iteration in 0..max_iterations {
        // Recalculate stress using v2 algorithm
        let result = recalc_stress_region_v2(
            density_fields, stress_fields, support_fields,
            config, &current_dirty, chunk_size,
        );

        for key in &result.affected_chunks {
            all_dirty_chunks.insert(*key);
        }

        if result.overstressed.is_empty() {
            break;
        }

        // Execute v2 collapses (coherent slabs)
        let events = detect_and_execute_collapses_v2(
            density_fields, stress_fields, support_fields,
            &result.overstressed, config, chunk_size,
        );

        if events.is_empty() {
            break;
        }

        // Collect newly affected chunks for cascade iteration
        let mut cascade_dirty = HashSet::new();
        for event in &events {
            for key in &event.affected_chunks {
                all_dirty_chunks.insert(*key);
                cascade_dirty.insert(*key);
            }
        }

        all_events.extend(events);
        current_dirty = cascade_dirty.into_iter().collect();
    }

    (all_events, all_dirty_chunks)
}

/// After mining or support changes, run stress recalculation and collapse detection
/// with cascade (max iterations configurable, default 5).
/// Returns collapse events and all dirty chunks that need remeshing.
pub fn post_change_stress_update(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    stress_fields: &mut HashMap<(i32, i32, i32), StressField>,
    support_fields: &HashMap<(i32, i32, i32), SupportField>,
    config: &StressConfig,
    changed_world_pos: (i32, i32, i32),
    chunk_size: usize,
) -> (Vec<CollapseEvent>, HashSet<(i32, i32, i32)>) {
    post_change_stress_update_with_iterations(
        density_fields, stress_fields, support_fields,
        config, changed_world_pos, chunk_size, 5,
    )
}

/// Same as post_change_stress_update but with configurable max cascade iterations.
pub fn post_change_stress_update_with_iterations(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    stress_fields: &mut HashMap<(i32, i32, i32), StressField>,
    support_fields: &HashMap<(i32, i32, i32), SupportField>,
    config: &StressConfig,
    changed_world_pos: (i32, i32, i32),
    chunk_size: usize,
    max_iterations: u32,
) -> (Vec<CollapseEvent>, HashSet<(i32, i32, i32)>) {
    let mut all_events = Vec::new();
    let mut all_dirty_chunks = HashSet::new();
    let mut center = changed_world_pos;

    for _iteration in 0..max_iterations {
        // Recalculate stress in region
        let result = recalc_stress_region(
            density_fields,
            stress_fields,
            support_fields,
            config,
            center,
            config.propagation_radius,
            chunk_size,
        );

        for key in &result.affected_chunks {
            all_dirty_chunks.insert(*key);
        }

        if result.overstressed.is_empty() {
            break;
        }

        // Execute collapses
        let events = detect_and_execute_collapses(
            density_fields,
            stress_fields,
            support_fields,
            &result.overstressed,
            config,
            chunk_size,
        );

        if events.is_empty() {
            break;
        }

        // Track dirty chunks from collapse events
        for event in &events {
            for key in &event.affected_chunks {
                all_dirty_chunks.insert(*key);
            }
            // Update center for next cascade iteration
            center = (
                event.center.0 as i32,
                event.center.1 as i32,
                event.center.2 as i32,
            );
        }

        all_events.extend(events);
    }

    (all_events, all_dirty_chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stress_field_basic() {
        let mut sf = StressField::new(17);
        assert_eq!(sf.stress.len(), 17 * 17 * 17);
        assert_eq!(sf.get(0, 0, 0), 0.0);
        sf.set(5, 5, 5, 0.75);
        assert!((sf.get(5, 5, 5) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn support_field_basic() {
        let mut sf = SupportField::new(17);
        assert_eq!(sf.supports.len(), 17 * 17 * 17);
        assert!(!sf.has_support(0, 0, 0));
        sf.set(3, 3, 3, SupportType::SlateStrut);
        assert!(sf.has_support(3, 3, 3));
        assert_eq!(sf.get(3, 3, 3), SupportType::SlateStrut);
    }

    #[test]
    fn support_type_from_u8() {
        assert_eq!(SupportType::from_u8(0), SupportType::None);
        assert_eq!(SupportType::from_u8(1), SupportType::SlateStrut);
        assert_eq!(SupportType::from_u8(2), SupportType::GraniteStrut);
        assert_eq!(SupportType::from_u8(3), SupportType::LimestoneStrut);
        assert_eq!(SupportType::from_u8(4), SupportType::CopperStrut);
        assert_eq!(SupportType::from_u8(5), SupportType::IronStrut);
        assert_eq!(SupportType::from_u8(6), SupportType::SteelStrut);
        assert_eq!(SupportType::from_u8(7), SupportType::CrystalStrut);
        assert_eq!(SupportType::from_u8(255), SupportType::None);
    }

    #[test]
    fn hardness_tables_correct_length() {
        assert_eq!(DEFAULT_MATERIAL_HARDNESS.len(), 47);
        assert_eq!(SUPPORT_HARDNESS.len(), 8);
    }

    fn make_density_field(size: usize, fill_solid: bool) -> DensityField {
        let mut df = DensityField::new(size);
        if fill_solid {
            for sample in df.samples.iter_mut() {
                sample.density = 1.0;
                sample.material = Material::Granite;
            }
        }
        df
    }

    /// Create a 3x3x3 grid of chunks centered on (0,0,0) to avoid boundary effects.
    fn make_solid_world() -> (HashMap<(i32,i32,i32), DensityField>, HashMap<(i32,i32,i32), StressField>, HashMap<(i32,i32,i32), SupportField>) {
        let mut density_fields = HashMap::new();
        let mut stress_fields = HashMap::new();
        let support_fields = HashMap::new();
        for cz in -1..=1 {
            for cy in -1..=1 {
                for cx in -1..=1 {
                    density_fields.insert((cx, cy, cz), make_density_field(17, true));
                    stress_fields.insert((cx, cy, cz), StressField::new(17));
                }
            }
        }
        (density_fields, stress_fields, support_fields)
    }

    fn make_air_world() -> (HashMap<(i32,i32,i32), DensityField>, HashMap<(i32,i32,i32), StressField>, HashMap<(i32,i32,i32), SupportField>) {
        let mut density_fields = HashMap::new();
        let mut stress_fields = HashMap::new();
        let support_fields = HashMap::new();
        for cz in -1..=1 {
            for cy in -1..=1 {
                for cx in -1..=1 {
                    let mut df = DensityField::new(17);
                    // Default VoxelSample is Limestone/solid, so explicitly set to Air
                    for sample in df.samples.iter_mut() {
                        sample.density = -1.0;
                        sample.material = Material::Air;
                    }
                    density_fields.insert((cx, cy, cz), df);
                    stress_fields.insert((cx, cy, cz), StressField::new(17));
                }
            }
        }
        (density_fields, stress_fields, support_fields)
    }

    fn default_config() -> StressConfig {
        StressConfig::default()
    }

    #[test]
    fn air_voxel_has_zero_stress() {
        let (density_fields, mut stress_fields, support_fields) = make_air_world();
        let config = default_config();

        let result = recalc_stress_region(
            &density_fields, &mut stress_fields, &support_fields,
            &config, (8, 8, 8), 4, 16,
        );

        assert!(result.overstressed.is_empty());
    }

    #[test]
    fn supported_voxel_low_stress() {
        let (density_fields, _, support_fields) = make_solid_world();
        let config = default_config();

        let stress = calc_voxel_stress(
            &density_fields, &support_fields, &config, 8, 8, 8, 16,
        );

        assert!(stress > 0.0, "Deep solid voxel should have positive stress");
        assert!(stress.is_finite(), "Stress should be finite");
    }

    #[test]
    fn surface_voxel_low_stress() {
        let (mut density_fields, _, support_fields) = make_solid_world();
        let config = default_config();

        if let Some(df) = density_fields.get_mut(&(0, 0, 0)) {
            for z in 0..17 {
                for y in 10..17 {
                    for x in 0..17 {
                        let sample = df.get_mut(x, y, z);
                        sample.density = -1.0;
                        sample.material = Material::Air;
                    }
                }
            }
        }
        if let Some(df) = density_fields.get_mut(&(0, 1, 0)) {
            for sample in df.samples.iter_mut() {
                sample.density = -1.0;
                sample.material = Material::Air;
            }
        }

        let stress = calc_voxel_stress(
            &density_fields, &support_fields, &config, 8, 9, 8, 16,
        );

        assert!(stress < 1.0, "Surface voxel should not be overstressed, got {}", stress);
    }

    #[test]
    fn unsupported_ceiling_high_stress() {
        let (mut density_fields, _, support_fields) = make_solid_world();
        let config = default_config();

        if let Some(df) = density_fields.get_mut(&(0, 0, 0)) {
            for z in 0..17 {
                for y in 0..8 {
                    for x in 0..17 {
                        let sample = df.get_mut(x, y, z);
                        sample.density = -1.0;
                        sample.material = Material::Air;
                    }
                }
            }
        }
        if let Some(df) = density_fields.get_mut(&(0, -1, 0)) {
            for sample in df.samples.iter_mut() {
                sample.density = -1.0;
                sample.material = Material::Air;
            }
        }

        let stress = calc_voxel_stress(
            &density_fields, &support_fields, &config, 8, 8, 8, 16,
        );

        assert!(stress > 0.0, "Ceiling voxel should have stress > 0");
    }

    #[test]
    fn support_structure_reduces_stress() {
        let (mut density_fields, _, _) = make_solid_world();
        let mut support_fields_empty = HashMap::new();
        let mut support_fields_with = HashMap::new();
        let config = default_config();

        if let Some(df) = density_fields.get_mut(&(0, 0, 0)) {
            for z in 0..17 {
                for y in 0..8 {
                    for x in 0..17 {
                        let sample = df.get_mut(x, y, z);
                        sample.density = -1.0;
                        sample.material = Material::Air;
                    }
                }
            }
        }

        for cz in -1..=1 {
            for cy in -1..=1 {
                for cx in -1..=1 {
                    support_fields_empty.insert((cx, cy, cz), SupportField::new(17));
                    support_fields_with.insert((cx, cy, cz), SupportField::new(17));
                }
            }
        }

        if let Some(sf) = support_fields_with.get_mut(&(0, 0, 0)) {
            sf.set(8, 7, 8, SupportType::SteelStrut);
        }

        let stress_without = calc_voxel_stress(
            &density_fields, &support_fields_empty, &config, 8, 8, 8, 16,
        );
        let stress_with = calc_voxel_stress(
            &density_fields, &support_fields_with, &config, 8, 8, 8, 16,
        );

        assert!(
            stress_with < stress_without,
            "Support should reduce stress: with={}, without={}",
            stress_with, stress_without
        );
    }

    #[test]
    fn world_to_chunk_local_works() {
        let (key, lx, ly, lz) = world_to_chunk_local(20, 5, -3, 16);
        assert_eq!(key, (1, 0, -1));
        assert_eq!(lx, 4);
        assert_eq!(ly, 5);
        assert_eq!(lz, 13);
    }

    #[test]
    fn collapse_converts_to_air() {
        let mut density_fields = HashMap::new();
        let mut stress_fields = HashMap::new();
        let support_fields = HashMap::new();
        let config = default_config();

        let df = make_density_field(17, true);
        density_fields.insert((0, 0, 0), df);
        stress_fields.insert((0, 0, 0), StressField::new(17));

        let overstressed = vec![OverstressedVoxel {
            world_x: 5,
            world_y: 5,
            world_z: 5,
            stress: 1.5,
        }];

        let events = detect_and_execute_collapses(
            &mut density_fields, &mut stress_fields, &support_fields,
            &overstressed, &config, 16,
        );

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].collapsed_voxels.len(), 1);

        // Verify voxel is now air
        let df = density_fields.get(&(0, 0, 0)).unwrap();
        assert_eq!(df.get(5, 5, 5).material, Material::Air);
    }

    // ── V2 algorithm tests ──

    /// Helper: carve a horizontal tunnel (air) at given y range across a chunk.
    /// Returns world with solid above/below and air in between.
    fn make_tunnel_world(tunnel_y_min: usize, tunnel_y_max: usize)
        -> (HashMap<(i32,i32,i32), DensityField>, HashMap<(i32,i32,i32), StressField>, HashMap<(i32,i32,i32), SupportField>)
    {
        let (mut density_fields, stress_fields, support_fields) = make_solid_world();
        // Carve air in center chunk
        if let Some(df) = density_fields.get_mut(&(0, 0, 0)) {
            for z in 0..17 {
                for y in tunnel_y_min..=tunnel_y_max {
                    for x in 0..17 {
                        let sample = df.get_mut(x, y, z);
                        sample.density = -1.0;
                        sample.material = Material::Air;
                    }
                }
            }
        }
        // Also carve in adjacent chunks for wider tunnel
        for &cx in &[-1, 1] {
            if let Some(df) = density_fields.get_mut(&(cx, 0, 0)) {
                for z in 0..17 {
                    for y in tunnel_y_min..=tunnel_y_max {
                        for x in 0..17 {
                            let sample = df.get_mut(x, y, z);
                            sample.density = -1.0;
                            sample.material = Material::Air;
                        }
                    }
                }
            }
        }
        for &cz in &[-1, 1] {
            if let Some(df) = density_fields.get_mut(&(0, 0, cz)) {
                for z in 0..17 {
                    for y in tunnel_y_min..=tunnel_y_max {
                        for x in 0..17 {
                            let sample = df.get_mut(x, y, z);
                            sample.density = -1.0;
                            sample.material = Material::Air;
                        }
                    }
                }
            }
        }
        (density_fields, stress_fields, support_fields)
    }

    #[test]
    fn v2_ground_connectivity_grounded_voxels() {
        let (density_fields, _, _) = make_solid_world();
        let config = default_config();
        let scores = ground_connectivity_pass(
            &density_fields, &[(0, 0, 0)], 16, &config,
        );
        // A voxel deep in solid rock should be fully grounded
        let score = scores.get(&(0, 0, 0)).unwrap().get(8, 8, 8);
        assert!(score >= 0.95, "Deep solid voxel should be grounded, got {}", score);
    }

    #[test]
    fn v2_ground_connectivity_ceiling_low_score() {
        // Carve a wide tunnel: air from y=0 to y=7, solid ceiling at y=8+
        let (density_fields, _, _) = make_tunnel_world(0, 7);
        let config = default_config();
        let scores = ground_connectivity_pass(
            &density_fields, &[(0, 0, 0)], 16, &config,
        );
        // A ceiling voxel at y=8 above wide air should have low support score
        // (it's not directly grounded — nothing solid below in its chunk)
        let ceiling_score = scores.get(&(0, 0, 0)).unwrap().get(8, 8, 8);
        // Score should be less than ground_threshold (0.8) for a wide unsupported ceiling
        assert!(ceiling_score < 0.95,
            "Wide ceiling voxel should have reduced support, got {}", ceiling_score);
    }

    #[test]
    fn v2_small_tunnel_stable() {
        // A narrow 4-wide tunnel should NOT produce overstressed voxels
        let (density_fields, mut stress_fields, support_fields) = make_solid_world();
        let mut config = default_config();
        config.min_safe_span = 6;
        // Carve a 4-wide tunnel in center chunk only (narrow)
        let mut df_clone = density_fields.clone();
        if let Some(df) = df_clone.get_mut(&(0, 0, 0)) {
            for z in 6..10 { // 4 voxels wide in Z
                for y in 4..8 {   // 4 voxels tall
                    for x in 4..12 { // 8 voxels long in X
                        let sample = df.get_mut(x, y, z);
                        sample.density = -1.0;
                        sample.material = Material::Air;
                    }
                }
            }
        }
        let result = recalc_stress_region_v2(
            &df_clone, &mut stress_fields, &support_fields,
            &config, &[(0, 0, 0)], 16,
        );
        assert!(result.overstressed.is_empty(),
            "Narrow 4-wide tunnel should not produce overstressed voxels, got {}",
            result.overstressed.len());
    }

    #[test]
    fn v2_slab_coherence() {
        // Create a slab scenario and verify collapsed region is contiguous
        let (mut density_fields, mut stress_fields, support_fields) = make_solid_world();
        let config = default_config();

        // Create a group of overstressed voxels in a 3x1x3 pattern
        let mut overstressed = Vec::new();
        for x in 5..8 {
            for z in 5..8 {
                overstressed.push(OverstressedVoxel {
                    world_x: x, world_y: 10, world_z: z, stress: 1.5,
                });
            }
        }
        // Floor below at y=0..5
        if let Some(df) = density_fields.get_mut(&(0, 0, 0)) {
            for z in 0..17 {
                for y in 6..10 { // Air gap between floor and slab
                    for x in 0..17 {
                        let sample = df.get_mut(x, y, z);
                        sample.density = -1.0;
                        sample.material = Material::Air;
                    }
                }
            }
        }

        let events = detect_and_execute_collapses_v2(
            &mut density_fields, &mut stress_fields, &support_fields,
            &overstressed, &config, 16,
        );

        assert_eq!(events.len(), 1, "Should produce exactly 1 collapse event");
        assert_eq!(events[0].slabs.len(), 1, "Should produce 1 slab");
        let slab = &events[0].slabs[0];
        assert_eq!(slab.voxels.len(), 9, "Slab should contain 9 voxels (3x1x3)");
        assert!(slab.fall_distance > 0, "Slab should have positive fall distance");
    }

    #[test]
    fn v2_slab_landing_preserves_shape() {
        // Slab at y=10, floor at y=5, should land at y=6
        let (mut density_fields, mut stress_fields, support_fields) = make_solid_world();
        let config = default_config();

        // Carve air from y=6 to y=9
        if let Some(df) = density_fields.get_mut(&(0, 0, 0)) {
            for z in 0..17 {
                for y in 6..10 {
                    for x in 0..17 {
                        let sample = df.get_mut(x, y, z);
                        sample.density = -1.0;
                        sample.material = Material::Air;
                    }
                }
            }
        }

        let overstressed = vec![
            OverstressedVoxel { world_x: 8, world_y: 10, world_z: 8, stress: 1.5 },
            OverstressedVoxel { world_x: 9, world_y: 10, world_z: 8, stress: 1.5 },
            OverstressedVoxel { world_x: 8, world_y: 10, world_z: 9, stress: 1.5 },
            OverstressedVoxel { world_x: 9, world_y: 10, world_z: 9, stress: 1.5 },
        ];

        let events = detect_and_execute_collapses_v2(
            &mut density_fields, &mut stress_fields, &support_fields,
            &overstressed, &config, 16,
        );

        assert_eq!(events.len(), 1);
        let slab = &events[0].slabs[0];
        assert_eq!(slab.fall_distance, 4, "Should fall 4 voxels (10 → 6)");

        // Verify rubble was placed as a mound near the landing area
        // With mound shape, center voxels are placed higher, edge voxels lower
        let df = density_fields.get(&(0, 0, 0)).unwrap();
        // At least some rubble should exist in the landing zone (y=6..8)
        let mut rubble_count = 0;
        for y in 6..9 {
            for z in 7..11 {
                for x in 7..11 {
                    if df.get(x, y, z).material.is_solid() {
                        rubble_count += 1;
                    }
                }
            }
        }
        assert!(rubble_count > 0, "Should have rubble in the landing zone");
        // Original position should be air
        assert_eq!(df.get(8, 10, 8).material, Material::Air,
            "Original slab position should be air");
    }

    #[test]
    fn v2_strut_reduces_stress() {
        // Create a wide tunnel so ceiling has actual stress, then verify strut reduces it
        let (mut density_fields, _, _) = make_solid_world();
        let mut support_fields_empty = HashMap::new();
        let mut support_fields_with = HashMap::new();
        let config = default_config();

        // Carve a wide tunnel: air from y=0..7 across 3 chunks in X and Z
        // This creates a wide unsupported ceiling at y=8
        for &cz in &[-1, 0, 1] {
            for &cx in &[-1, 0, 1] {
                if let Some(df) = density_fields.get_mut(&(cx, 0, cz)) {
                    for z in 0..17 {
                        for y in 0..8 {
                            for x in 0..17 {
                                let sample = df.get_mut(x, y, z);
                                sample.density = -1.0;
                                sample.material = Material::Air;
                            }
                        }
                    }
                }
                if let Some(df) = density_fields.get_mut(&(cx, -1, cz)) {
                    for sample in df.samples.iter_mut() {
                        sample.density = -1.0;
                        sample.material = Material::Air;
                    }
                }
            }
        }

        for cz in -1..=1 {
            for cy in -1..=1 {
                for cx in -1..=1 {
                    support_fields_empty.insert((cx, cy, cz), SupportField::new(17));
                    support_fields_with.insert((cx, cy, cz), SupportField::new(17));
                }
            }
        }
        // Place a steel strut just below the ceiling
        if let Some(sf) = support_fields_with.get_mut(&(0, 0, 0)) {
            sf.set(8, 7, 8, SupportType::SteelStrut);
        }

        let scores = ground_connectivity_pass(
            &density_fields, &[(0, 0, 0)], 16, &config,
        );

        let stress_without = calc_voxel_stress_v2(
            &density_fields, &support_fields_empty, &scores, &config, 8, 8, 8, 16,
        );
        let stress_with = calc_voxel_stress_v2(
            &density_fields, &support_fields_with, &scores, &config, 8, 8, 8, 16,
        );

        assert!(stress_without > 0.0,
            "Wide ceiling should have positive stress without strut, got {}", stress_without);
        assert!(stress_with < stress_without,
            "Strut should reduce v2 stress: with={}, without={}", stress_with, stress_without);
    }
}
