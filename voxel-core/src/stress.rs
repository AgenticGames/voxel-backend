use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::density::DensityField;
use crate::material::Material;

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
pub const DEFAULT_MATERIAL_HARDNESS: [f32; 27] = [
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
    pub material_hardness: [f32; 27],
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
        assert_eq!(DEFAULT_MATERIAL_HARDNESS.len(), 27);
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
}
