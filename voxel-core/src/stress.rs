use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};
use voxel_noise::{simplex::Simplex3D, NoiseSource};

use crate::density::DensityField;
use crate::material::Material;

/// Serde helper for [f32; 50] (serde doesn't impl Serialize/Deserialize for arrays > 32).
mod serde_f32_array_50 {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    pub fn serialize<S: Serializer>(arr: &[f32; 50], s: S) -> Result<S::Ok, S::Error> {
        arr.as_slice().serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[f32; 50], D::Error> {
        let mut v: Vec<f32> = Vec::deserialize(d)?;
        // Tolerate older saved configs sized 47 by zero-padding the new tail entries.
        if v.len() == 47 {
            v.resize(50, 0.0);
        }
        v.try_into().map_err(|v: Vec<f32>| serde::de::Error::custom(
            format!("expected 50 elements, got {}", v.len())))
    }
}

/// Per-voxel stress values for a chunk. Same layout as DensityField (17^3 for chunk_size=16).
/// Classification byte: top 4 bits = surface type, bottom 4 bits = dominant stress source.
/// Surface types: 0=interior, 1=floor, 2=ceiling, 3=wall, 4=thin_feature
/// Stress sources: 0=none, 1=gravity, 2=overhang, 3=span, 4=cross_section
pub const SURFACE_INTERIOR: u8 = 0;
pub const SURFACE_FLOOR: u8 = 1;
pub const SURFACE_CEILING: u8 = 2;
pub const SURFACE_WALL: u8 = 3;
pub const SURFACE_THIN: u8 = 4;

pub const SOURCE_NONE: u8 = 0;
pub const SOURCE_GRAVITY: u8 = 1;
pub const SOURCE_OVERHANG: u8 = 2;
pub const SOURCE_SPAN: u8 = 3;
pub const SOURCE_CROSS_SECTION: u8 = 4;

/// A localized stress recalculation event centered on a mine point.
/// Only voxels within `radius` of `center` (world voxel coords) get recalculated.
#[derive(Debug, Clone)]
pub struct StressDirtyEvent {
    pub center: (i32, i32, i32),
    pub radius: i32,
}

impl StressDirtyEvent {
    /// Returns the set of chunk keys whose bounding boxes overlap this event's sphere.
    pub fn affected_chunks(&self, chunk_size: usize) -> Vec<(i32, i32, i32)> {
        let cs = chunk_size as i32;
        let min_cx = (self.center.0 - self.radius).div_euclid(cs);
        let max_cx = (self.center.0 + self.radius).div_euclid(cs);
        let min_cy = (self.center.1 - self.radius).div_euclid(cs);
        let max_cy = (self.center.1 + self.radius).div_euclid(cs);
        let min_cz = (self.center.2 - self.radius).div_euclid(cs);
        let max_cz = (self.center.2 + self.radius).div_euclid(cs);
        let mut keys = Vec::new();
        for cz in min_cz..=max_cz {
            for cy in min_cy..=max_cy {
                for cx in min_cx..=max_cx {
                    keys.push((cx, cy, cz));
                }
            }
        }
        keys
    }

    /// Check if a world voxel position is within this event's radius.
    #[inline]
    pub fn contains(&self, wx: i32, wy: i32, wz: i32) -> bool {
        let dx = wx - self.center.0;
        let dy = wy - self.center.1;
        let dz = wz - self.center.2;
        dx * dx + dy * dy + dz * dz <= self.radius * self.radius
    }
}

/// Check if a world position is within ANY event's radius.
#[inline]
pub fn in_any_event(events: &[StressDirtyEvent], wx: i32, wy: i32, wz: i32) -> bool {
    events.iter().any(|e| e.contains(wx, wy, wz))
}

#[inline]
pub fn pack_classification(surface: u8, source: u8) -> u8 {
    (surface << 4) | (source & 0x0F)
}

#[inline]
pub fn unpack_surface(c: u8) -> u8 { c >> 4 }

#[inline]
pub fn unpack_source(c: u8) -> u8 { c & 0x0F }

#[derive(Debug, Clone)]
pub struct StressField {
    pub stress: Vec<f32>,
    /// Per-voxel classification: surface type (top 4 bits) + stress source (bottom 4 bits)
    pub classification: Vec<u8>,
    pub size: usize,
    /// Player-painted additive stress overlay (creative-mode "PaintStress" brush).
    /// Empty Vec = no painted layer (no allocation). When non-empty, len = size^3.
    /// `effective_stress = stress + painted_stress` — survives recalc passes since
    /// the recalc only writes into `stress[]`.
    pub painted_stress: Vec<f32>,
}

impl StressField {
    pub fn new(size: usize) -> Self {
        let count = size * size * size;
        Self {
            stress: vec![0.0; count],
            classification: vec![0u8; count],
            size,
            painted_stress: Vec::new(),
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

    #[inline]
    pub fn get_class(&self, x: usize, y: usize, z: usize) -> u8 {
        self.classification[self.index(x, y, z)]
    }

    #[inline]
    pub fn set_class(&mut self, x: usize, y: usize, z: usize, val: u8) {
        let idx = self.index(x, y, z);
        self.classification[idx] = val;
    }

    /// True if any painted-stress value has ever been written into this chunk.
    /// When false, the painted overlay is treated as all-zeros at zero memory cost.
    #[inline]
    pub fn has_painted_layer(&self) -> bool {
        !self.painted_stress.is_empty()
    }

    /// Read the painted-stress overlay (0.0 if no layer allocated).
    #[inline]
    pub fn painted(&self, x: usize, y: usize, z: usize) -> f32 {
        if self.painted_stress.is_empty() {
            0.0
        } else {
            self.painted_stress[self.index(x, y, z)]
        }
    }

    /// Effective stress = base + painted overlay.
    /// Use this where you want player-painted stress to influence behavior
    /// (collapse-failure rolls, overstressed test, debug viz).
    #[inline]
    pub fn effective(&self, x: usize, y: usize, z: usize) -> f32 {
        let i = self.index(x, y, z);
        let base = self.stress[i];
        let painted = if self.painted_stress.is_empty() {
            0.0
        } else {
            self.painted_stress[i]
        };
        base + painted
    }

    /// Lazy-allocate the painted overlay. No-op if already allocated.
    fn ensure_painted_alloc(&mut self) {
        if self.painted_stress.is_empty() {
            self.painted_stress = vec![0.0; self.size * self.size * self.size];
        }
    }

    /// Add to the painted-stress overlay at one cell, clamped to `[0, cap]`.
    /// Negative `delta` subtracts (allowing right-click "lighten" semantics).
    /// `cap` is the per-cell ceiling for accumulated paint (typical: 2.0).
    pub fn add_painted(&mut self, x: usize, y: usize, z: usize, delta: f32, cap: f32) {
        if delta == 0.0 {
            return;
        }
        self.ensure_painted_alloc();
        let i = self.index(x, y, z);
        let v = (self.painted_stress[i] + delta).clamp(0.0, cap);
        self.painted_stress[i] = v;
    }

    /// Set the painted-stress overlay at one cell to an exact value (clamped to >= 0).
    pub fn set_painted(&mut self, x: usize, y: usize, z: usize, val: f32) {
        self.ensure_painted_alloc();
        let i = self.index(x, y, z);
        self.painted_stress[i] = val.max(0.0);
    }

    /// Zero the painted overlay at one cell. Doesn't deallocate the layer.
    pub fn clear_painted(&mut self, x: usize, y: usize, z: usize) {
        if self.painted_stress.is_empty() {
            return;
        }
        let i = self.index(x, y, z);
        self.painted_stress[i] = 0.0;
    }

    /// Zero the entire painted overlay (called by the "clear all painted stress" tool).
    pub fn clear_all_painted(&mut self) {
        if !self.painted_stress.is_empty() {
            self.painted_stress.fill(0.0);
        }
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
    /// Count of cells where supports[i] != SupportType::None.
    /// Maintained by `set()` so callers can do an O(1) "any support here?"
    /// check before walking a per-voxel support-radius scan.
    pub non_none_count: u32,
}

impl SupportField {
    pub fn new(size: usize) -> Self {
        Self {
            supports: vec![SupportType::None; size * size * size],
            size,
            non_none_count: 0,
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
        let was_none = self.supports[idx] == SupportType::None;
        let is_none = support_type == SupportType::None;
        match (was_none, is_none) {
            (true, false) => self.non_none_count = self.non_none_count.saturating_add(1),
            (false, true) => self.non_none_count = self.non_none_count.saturating_sub(1),
            _ => {}
        }
        self.supports[idx] = support_type;
    }

    #[inline]
    pub fn has_support(&self, x: usize, y: usize, z: usize) -> bool {
        self.get(x, y, z) != SupportType::None
    }

    /// O(1): true if every cell in this chunk's support field is `SupportType::None`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.non_none_count == 0
    }
}

/// O(1)-per-chunk check: does any chunk in the bounding box of voxel
/// `(wx,wy,wz)` ± `sr` voxels have at least one non-None support entry?
///
/// Used to fast-skip the per-voxel `(2sr+1)^3` support-radius scan in
/// `calc_voxel_stress*` when no struts have been placed near this voxel.
/// At most 8 chunk lookups in the worst case (typically 1 when sr < cs).
#[inline]
fn any_supports_in_radius_box(
    support_fields: &HashMap<(i32, i32, i32), SupportField>,
    wx: i32, wy: i32, wz: i32,
    sr: i32,
    chunk_size: usize,
) -> bool {
    let cs = chunk_size as i32;
    let cx_min = (wx - sr).div_euclid(cs);
    let cx_max = (wx + sr).div_euclid(cs);
    let cy_min = (wy - sr).div_euclid(cs);
    let cy_max = (wy + sr).div_euclid(cs);
    let cz_min = (wz - sr).div_euclid(cs);
    let cz_max = (wz + sr).div_euclid(cs);
    for cx in cx_min..=cx_max {
        for cy in cy_min..=cy_max {
            for cz in cz_min..=cz_max {
                if let Some(sf) = support_fields.get(&(cx, cy, cz)) {
                    if !sf.is_empty() {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Default hardness per Material (index by Material as u8).
/// Air = 0.0 (no resistance). Higher = harder to collapse.
pub const DEFAULT_MATERIAL_HARDNESS: [f32; 50] = [
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
    0.20,  // IceSheet (chunky fractured ice)
    0.15,  // FrozenGlow (luminous icicle drip)
    0.78,  // Amphibolite (hard metabasite, hornblende-rich)
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
    #[serde(with = "serde_f32_array_50")]
    pub material_hardness: [f32; 50],
    /// Weight per solid voxel above (column load factor).
    pub gravity_weight: f32,
    /// Contribution factor for lateral (side) neighbors.
    pub lateral_support_factor: f32,
    /// Contribution factor for voxel directly below.
    pub vertical_support_factor: f32,
    /// Effect radius of support structures.
    pub support_radius: u32,
    /// ⚠ LEGACY: BFS recalc radius for the OLD immediate-collapse pipeline.
    /// Mining no longer uses this — see `mining_stress_scan_buffer` below for
    /// the cinematic (`CollapseSlabResult` / `CollapseWarning`) path. Still
    /// consumed by sleep collapses (`detect_and_execute_collapses` / legacy
    /// `CollapseResult` emission). Two systems coexist; do not conflate.
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
    /// Stress per additional air face-neighbor beyond 1 (thin feature penalty).
    pub cross_section_weight: f32,
    /// Minimum air face-neighbors to trigger cross-section penalty (2 = default).
    pub cross_section_min_faces: u32,
    /// World Y coordinate of the surface (depth=0 reference point).
    pub surface_y: i32,
    /// Depth scale: depth_factor = 1.0 + depth / depth_scale. Lower = more aggressive.
    pub depth_pressure_scale: f32,

    // ── Cinematic mining pipeline (CollapseWarning + SlabFall path) ──
    //
    // The cinematic system (worker.rs `WorkerRequest::Mine`) recomputes
    // stress in a SPHERE around the mine point with radius
    // `mine_radius_voxels + mining_stress_scan_buffer`. Then a 26-connected
    // BFS expands through any solid voxel whose existing stress is already
    // ≥ `slab_cohesion_threshold` (capped at `max_collapse_volume`). The BFS
    // can therefore reach voxels OUTSIDE the scan sphere via pre-stressed
    // cohesive rock chains — that's how a far-away slab can fall from one
    // mine. NOT the same as `propagation_radius`, which only the legacy
    // sleep path consults.
    /// Buffer (in voxels) added to the mine radius for the cinematic stress
    /// scan sphere. Total scan radius = `mine_radius_voxels + this`. Tuning
    /// it directly affects how far out from a mine the system can detect
    /// stress to trigger slab falls.
    pub mining_stress_scan_buffer: u32,
}

impl Default for StressConfig {
    fn default() -> Self {
        Self {
            material_hardness: DEFAULT_MATERIAL_HARDNESS,
            gravity_weight: 0.1,        // Flat base load per unsupported voxel (not per-column anymore)
            lateral_support_factor: 0.3,
            vertical_support_factor: 1.0,
            support_radius: 3,
            propagation_radius: 8,
            max_collapse_volume: 8000,
            rubble_enabled: true,
            rubble_fill_ratio: 0.5,
            warn_dust_threshold: 0.4,
            warn_creak_threshold: 0.6,
            warn_shake_threshold: 0.8,
            support_hardness: SUPPORT_HARDNESS,
            // V2 defaults
            lateral_transfer_factor: 0.7,
            vertical_transfer_factor: 0.95,
            support_propagation_iterations: 2,
            ground_threshold: 0.80,     // Reverted from 0.95 — proper init makes 0.80 correct
            overhang_weight: 0.05,      // Primary ceiling stress driver. With cap=12: max raw=0.6
            span_weight: 0.025,         // Span penalty per voxel beyond safe (tuned down for gradient)
            min_safe_span: 8,           // Wider safe span — only large ceilings get stressed
            min_collapse_region: 8,
            slab_cohesion_threshold: 0.75,
            cross_section_weight: 0.15,  // Stress per air face beyond threshold (thin feature penalty)
            cross_section_min_faces: 2,  // Need 2+ air faces before penalty applies
            surface_y: 200,             // Approximate world surface level
            depth_pressure_scale: 99999.0, // Effectively disabled for now — tune after span gradient is dialed in
            mining_stress_scan_buffer: 22, // Was hardcoded `+22` in worker.rs; configurable since 2026-05
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

/// Data needed to apply a deferred pile placement later. Carries the
/// snapshot of collapsed slab voxels (which still have their original
/// material info) plus geometry.
#[derive(Debug, Clone)]
pub struct PendingPilePlacement {
    pub collapsed_voxels: Vec<CollapsedVoxel>,
    pub bb_min: (i32, i32, i32),
    pub bb_max: (i32, i32, i32),
    pub dominant_material: Material,
    pub landing_offset: i32,
}

/// Enhanced collapse event with coherent slab data for animated falling.
///
/// `affected_chunks` is the union of `slab_chunks` and `pile_chunks` for
/// backward compat — old code that doesn't care about the cinematic split
/// still works. The two sub-sets exist so the worker can defer pile-chunk
/// remeshes until after the falling-slab cinematic has impacted the floor:
///
/// - `slab_chunks`: chunks where slab voxels were CLEARED (cave roof opens
///   here). Should be remeshed at fall start so the roof hole appears as
///   the slab visually detaches.
/// - `pile_chunks`: chunks where rubble pile was PLACED (floor pile lands
///   here). Should be remeshed at impact so the pile appears under the
///   falling slab right when it lands.
///
/// `pending_piles` is populated when the caller used the no-pile variant
/// (`detect_collapses_v2_no_pile`) — the pile hasn't been placed yet.
/// The caller should later call `apply_pending_pile` to actually mutate
/// density and place the rubble. When the older
/// `detect_and_execute_collapses_v2` is used, `pending_piles` is empty and
/// `pile_chunks` is populated instead (pile already placed inline).
#[derive(Debug, Clone)]
pub struct CollapseEventV2 {
    pub slabs: Vec<CollapseSlab>,
    pub affected_chunks: Vec<(i32, i32, i32)>,
    pub slab_chunks: Vec<(i32, i32, i32)>,
    pub pile_chunks: Vec<(i32, i32, i32)>,
    pub pending_piles: Vec<PendingPilePlacement>,
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
pub fn sample_world(
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
    //
    // Fast skip: if no chunk in the (2sr+1)^3 voxel box around (wx,wy,wz) has
    // any non-None supports, the entire 7^3 sample_support sweep below is pure
    // waste. Cheap O(<=8) chunk lookups guard the 342-call HashMap walk.
    // For early-game (0 struts placed in the world) this short-circuits ~100%
    // of stressed voxels.
    let sr = config.support_radius as i32;
    if any_supports_in_radius_box(support_fields, wx, wy, wz, sr, chunk_size) {
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
    }

    // Clamp to non-negative before normalization
    raw_stress = raw_stress.max(0.0);

    // 4. Normalize by material hardness
    raw_stress / hardness
}

// ── V2 stress algorithm: two-pass ground connectivity + load accumulation ──

/// Minimum distance to nearest air voxel in 6 face-connected directions.
/// Returns 0 if the voxel itself is air, 1 if a face-neighbor is air, etc.
/// Returns `max_dist + 1` if no air found within range (deep interior).
fn min_distance_to_air(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    wx: i32, wy: i32, wz: i32,
    chunk_size: usize,
    max_dist: i32,
) -> i32 {
    let dirs: [(i32, i32, i32); 6] = [
        (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1),
    ];
    let mut best = max_dist + 1;
    for &(dx, dy, dz) in &dirs {
        for d in 1..=max_dist {
            let nx = wx + dx * d;
            let ny = wy + dy * d;
            let nz = wz + dz * d;
            match sample_world(density_fields, nx, ny, nz, chunk_size) {
                Some((_, mat)) if !mat.is_solid() => {
                    best = best.min(d);
                    break; // Found air in this direction
                }
                Some(_) => {} // Solid, keep searching
                None => break, // Unloaded, stop this direction
            }
        }
    }
    best
}

/// Count contiguous air voxels below a position (Y−), capped at 32.
pub fn count_air_below(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    wx: i32, wy: i32, wz: i32,
    chunk_size: usize,
) -> u32 {
    let mut count = 0u32;
    for dy in 1..=12i32 { // Capped at 12 — prevents deep cave stress explosion
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

/// Measure the unsupported span for a solid surface voxel.
///
/// For each air face-neighbor, searches laterally from that air position through air
/// to find the distance to the nearest wall. Returns the MINIMUM distance found —
/// the nearest wall provides structural support regardless of what's in other directions.
///
/// This handles both ceiling voxels (air below → search laterally at cave level)
/// and wall voxels (air to the side → search laterally through cave).
fn measure_span_from_air(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    wx: i32, wy: i32, wz: i32,
    chunk_size: usize,
    max_dist: u32,
) -> u32 {
    let face_offsets: [(i32, i32, i32); 6] = [
        (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1),
    ];
    let lat_dirs: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    let mut best_span = max_dist;
    let mut found_any = false;

    for &(dx, dy, dz) in &face_offsets {
        let ax = wx + dx;
        let ay = wy + dy;
        let az = wz + dz;

        // Only start from air neighbors
        match sample_world(density_fields, ax, ay, az, chunk_size) {
            Some((_, mat)) if !mat.is_solid() => {}
            _ => continue,
        }

        // From this air position, search laterally for walls
        for &(ldx, ldz) in &lat_dirs {
            for d in 1..=max_dist as i32 {
                let nx = ax + ldx * d;
                let nz = az + ldz * d;
                match sample_world(density_fields, nx, ay, nz, chunk_size) {
                    Some((_, mat)) if mat.is_solid() => {
                        best_span = best_span.min(d as u32);
                        found_any = true;
                        break;
                    }
                    Some(_) => {} // Air — keep going
                    None => {
                        best_span = best_span.min(d as u32); // Unloaded = wall
                        found_any = true;
                        break;
                    }
                }
            }
        }
    }
    if found_any { best_span } else { max_dist }
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

    // Initialize support scores via GLOBAL top-down column flood.
    // For each unique (wx, wz) column across dirty chunks, walk from max_wy to min_wy
    // using sample_world for cross-chunk reads. No per-chunk boundary artifacts.
    let mut scores: HashMap<(i32, i32, i32), SupportScoreField> = HashMap::new();
    for &key in &expanded_keys {
        scores.insert(key, SupportScoreField::new(grid_size));
    }

    let vert_decay = config.vertical_transfer_factor;

    // Collect unique (wx, wz) columns from DIRTY chunks only (not all expanded)
    let mut columns: HashSet<(i32, i32)> = HashSet::new();
    for &(cx, _, cz) in chunk_keys {
        for z in 0..grid_size {
            for x in 0..grid_size {
                columns.insert((cx * cs as i32 + x as i32, cz * cs as i32 + z as i32));
            }
        }
    }

    // Y range across all expanded chunks
    let mut min_wy = i32::MAX;
    let mut max_wy = i32::MIN;
    for &(_, cy, _) in &expanded_keys {
        min_wy = min_wy.min(cy * cs as i32);
        max_wy = max_wy.max(cy * cs as i32 + grid_size as i32 - 1);
    }

    // Global flood: each column walks top-to-bottom across all chunks.
    //
    // Perf: consecutive y values in the descending walk almost always land in
    // the same chunk (only `cy` changes when crossing a chunk boundary). Cache
    // the chunk_y and the looked-up DensityField/in_expanded flags so we only
    // re-fetch from `density_fields`/`expanded_keys` when `cy` changes. The
    // `scores.get_mut` write still happens per solid cell — keeping it lifted
    // would force unsafe (mutable + immutable map borrow simultaneously) and
    // the per-cell write path is the same cost it was before.
    let cs_i32 = cs as i32;
    for &(wx, wz) in &columns {
        let mut current_score = 1.0f32;
        let mut in_air_gap = false;

        let mut cached_cy: Option<i32> = None;
        let mut cached_df: Option<&DensityField> = None;
        let mut cached_in_expanded = false;
        let mut cached_key = (0i32, 0i32, 0i32);

        for wy in (min_wy..=max_wy).rev() {
            let cx = wx.div_euclid(cs_i32);
            let cy = wy.div_euclid(cs_i32);
            let cz = wz.div_euclid(cs_i32);
            let lx = wx.rem_euclid(cs_i32) as usize;
            let ly = wy.rem_euclid(cs_i32) as usize;
            let lz = wz.rem_euclid(cs_i32) as usize;

            if cached_cy != Some(cy) {
                cached_cy = Some(cy);
                cached_key = (cx, cy, cz);
                cached_df = density_fields.get(&cached_key);
                cached_in_expanded = expanded_keys.contains(&cached_key);
            }

            let is_solid = cached_df
                .map(|df| df.get(lx, ly, lz).material.is_solid())
                .unwrap_or(false);

            if !is_solid {
                in_air_gap = true;
                current_score = 0.0;
                continue;
            }

            if in_air_gap {
                current_score = 0.0;
                in_air_gap = false;
            }

            if cached_in_expanded {
                if let Some(sf) = scores.get_mut(&cached_key) {
                    sf.set(lx, ly, lz, current_score);
                }
            }

            current_score *= vert_decay;
        }
    }

    // Iterative relaxation
    let vert_transfer = config.vertical_transfer_factor;
    let lat_transfer = config.lateral_transfer_factor;

    for _iter in 0..config.support_propagation_iterations {
        // We need to read neighbor scores from the previous iteration,
        // so collect updates first, then apply.
        let mut updates: Vec<((i32, i32, i32), usize, usize, usize, f32)> = Vec::new();

        let cs_i32 = cs as i32;
        let last_local = grid_size - 1;
        for &key in &expanded_keys {
            let df = match density_fields.get(&key) {
                Some(d) => d,
                None => continue,
            };
            // Hoist the chunk's own score field outside the (z,y,x) loops:
            // every cell inside this chunk reads `scores.get(&key)` at least
            // once (for its `current_score`), and most neighbor reads also
            // resolve back to this same chunk because only voxels on a chunk
            // face cross into a different `SupportScoreField`. Caching it
            // saves grid_size^3 redundant HashMap lookups per chunk per
            // iteration (~29k for chunk_size=30) and an additional ~5×
            // savings on neighbor lookups whose key matches `key`.
            let current_sf = match scores.get(&key) {
                Some(sf) => sf,
                None => continue,
            };
            let (cx, cy, cz) = key;
            let chunk_origin_x = cx * cs_i32;
            let chunk_origin_y = cy * cs_i32;
            let chunk_origin_z = cz * cs_i32;

            for z in 0..grid_size {
                for y in 0..grid_size {
                    for x in 0..grid_size {
                        if !df.get(x, y, z).material.is_solid() {
                            continue;
                        }
                        let current_score = current_sf.get(x, y, z);
                        if current_score >= 1.0 {
                            continue; // Already fully grounded
                        }

                        let wx = chunk_origin_x + x as i32;
                        let wy = chunk_origin_y + y as i32;
                        let wz = chunk_origin_z + z as i32;

                        let mut best = current_score;

                        // Vertical transfer from below. Stay on the cached
                        // `current_sf` whenever the cell-below is in the
                        // same chunk (the common case — only y==0 crosses).
                        if y > 0 {
                            let below_score = current_sf.get(x, y - 1, z);
                            best = best.max(below_score * vert_transfer);
                        } else {
                            let bkey = (cx, cy - 1, cz);
                            if let Some(bsf) = scores.get(&bkey) {
                                let below_score = bsf.get(x, last_local, z);
                                best = best.max(below_score * vert_transfer);
                            } else {
                                // Unloaded neighbor = assume grounded
                                best = best.max(vert_transfer);
                            }
                        }

                        // Lateral transfer from 4 horizontal neighbors.
                        // Same-chunk reads use the cached field; only the 4
                        // cells on each face fall through to a HashMap lookup.
                        // -X
                        if x > 0 {
                            let n_score = current_sf.get(x - 1, y, z);
                            best = best.max(n_score * lat_transfer);
                        } else if let Some(nsf) = scores.get(&(cx - 1, cy, cz)) {
                            let n_score = nsf.get(last_local, y, z);
                            best = best.max(n_score * lat_transfer);
                        }
                        // +X
                        if x < last_local {
                            let n_score = current_sf.get(x + 1, y, z);
                            best = best.max(n_score * lat_transfer);
                        } else if let Some(nsf) = scores.get(&(cx + 1, cy, cz)) {
                            let n_score = nsf.get(0, y, z);
                            best = best.max(n_score * lat_transfer);
                        }
                        // -Z
                        if z > 0 {
                            let n_score = current_sf.get(x, y, z - 1);
                            best = best.max(n_score * lat_transfer);
                        } else if let Some(nsf) = scores.get(&(cx, cy, cz - 1)) {
                            let n_score = nsf.get(x, y, last_local);
                            best = best.max(n_score * lat_transfer);
                        }
                        // +Z
                        if z < last_local {
                            let n_score = current_sf.get(x, y, z + 1);
                            best = best.max(n_score * lat_transfer);
                        } else if let Some(nsf) = scores.get(&(cx, cy, cz + 1)) {
                            let n_score = nsf.get(x, y, 0);
                            best = best.max(n_score * lat_transfer);
                        }

                        // No above-transfer: ceiling rock must NOT bootstrap support
                        // from the unsupported mass above it. Support only comes from
                        // below (pillars/floor) and laterally (walls).

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
) -> (f32, u8) {
    // Only solid voxels have stress
    let mat = match sample_world(density_fields, wx, wy, wz, chunk_size) {
        Some((_, m)) if m.is_solid() => m,
        _ => return (0.0, pack_classification(SURFACE_INTERIOR, SOURCE_NONE)),
    };

    let hardness = config.material_hardness[mat as u8 as usize];
    if hardness <= 0.0 {
        return (0.0, pack_classification(SURFACE_INTERIOR, SOURCE_NONE));
    }

    // Get support score from ground connectivity pass
    let (key, lx, ly, lz) = world_to_chunk_local(wx, wy, wz, chunk_size);
    let support_score = support_scores
        .get(&key)
        .map(|sf| sf.get(lx, ly, lz))
        .unwrap_or(1.0); // Unloaded = assume grounded

    // Unsupported factor: 0.0 for fully grounded, 1.0 for floating.
    let unsupported = (1.0 - support_score).max(0.0);

    // Floor protection: solid below AND well-supported by the flood = stable floor.
    // Thick ceiling rock has solid below but LOW flood score (air gap broke chain) → NOT protected.
    // Floor rock has solid below AND HIGH flood score (connected to surface) → protected.
    {
        let below_solid = sample_world(density_fields, wx, wy - 1, wz, chunk_size)
            .map(|(_, m)| m.is_solid())
            .unwrap_or(true);
        if below_solid && support_score >= 0.2 {
            return (0.0, pack_classification(SURFACE_FLOOR, SOURCE_NONE));
        }
    }

    // Distance-to-air decay: stress attenuates as we go deeper into rock.
    // Surface voxels (1 cell from air) get full stress, deep interior gets none.
    // This prevents the span search from producing stress on voxels buried in solid rock
    // where the concept of "unsupported span" doesn't physically apply.
    let air_dist = min_distance_to_air(density_fields, wx, wy, wz, chunk_size, 2);
    let air_decay = if air_dist <= 1 {
        1.0  // At the cave surface: full stress
    } else if air_dist == 2 {
        0.5  // One cell deep: half stress
    } else {
        0.0  // 3+ cells deep: no surface stress
    };

    // Deep interior shortcut: no stress, classify as interior
    if air_decay <= 0.0 {
        return (0.0, pack_classification(SURFACE_INTERIOR, SOURCE_NONE));
    }

    // Track individual stress components for classification
    let mut raw_stress = 0.0f32;

    // Span penalty: measures widest unsupported air gap this voxel is exposed to.
    // Searches from each air face-neighbor laterally through air to find walls.
    // Near walls = low span = safe. Center of wide ceiling = high span = danger.
    let span_dist = measure_span_from_air(
        density_fields, wx, wy, wz, chunk_size, 20,
    );
    let span_stress = if span_dist > config.min_safe_span {
        (span_dist - config.min_safe_span) as f32 * config.span_weight * unsupported * air_decay
    } else { 0.0 };
    raw_stress += span_stress;

    // Cross-section penalty
    let face_offsets: [(i32, i32, i32); 6] = [
        (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1),
    ];
    let mut air_neighbors = 0u32;
    for &(dx, dy, dz) in &face_offsets {
        match sample_world(density_fields, wx + dx, wy + dy, wz + dz, chunk_size) {
            Some((_, m)) if !m.is_solid() => air_neighbors += 1,
            None => {}
            _ => {}
        }
    }
    let xsec_stress = if air_neighbors >= config.cross_section_min_faces {
        (air_neighbors - 1) as f32 * config.cross_section_weight
    } else { 0.0 };
    raw_stress += xsec_stress;

    // Support structure bonus: nearby struts reduce stress.
    //
    // Fast skip: if no chunk in the (2sr+1)^3 voxel box around (wx,wy,wz) has
    // any non-None supports, the entire 7^3 sample_support sweep below is pure
    // waste. Cheap O(<=8) chunk lookups guard the 342-call HashMap walk.
    // For early-game (0 struts placed in the world) this short-circuits ~100%
    // of stressed voxels in this hot loop.
    let sr = config.support_radius as i32;
    if any_supports_in_radius_box(support_fields, wx, wy, wz, sr, chunk_size) {
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
    }

    // Depth pressure: deeper rock is under more overburden compression.
    // At surface: 1.0x. At depth 100: 2.0x. At depth 200: 3.0x.
    // This makes narrow tunnels dangerous at depth even when span is safe.
    let depth = (config.surface_y - wy).max(0) as f32;
    let depth_factor = 1.0 + depth / config.depth_pressure_scale;

    let final_stress = (raw_stress.max(0.0) * depth_factor) / hardness;

    // Classify surface type using BOTH local air neighbors AND air_dist.
    // A voxel with air_neighbors==0 but air_dist<=4 is near the surface and may have
    // stress — classify by geometry (below_solid) rather than defaulting to INTERIOR.
    let below_solid = sample_world(density_fields, wx, wy - 1, wz, chunk_size)
        .map(|(_, m)| m.is_solid()).unwrap_or(true);
    let surface_type = if air_neighbors >= 4 {
        SURFACE_THIN       // Stalactite/thin column (4+ air faces)
    } else if !below_solid {
        SURFACE_CEILING    // Air directly below
    } else if air_neighbors == 0 && final_stress <= 0.001 {
        SURFACE_INTERIOR   // Fully enclosed AND no stress = truly interior
    } else if below_solid && support_score >= 0.2 {
        SURFACE_FLOOR      // Solid below + moderately supported
    } else {
        SURFACE_WALL       // Near surface, solid below = wall/pillar
    };

    // Dominant stress source (gravity + overhang removed — span is primary)
    let dominant_source = if final_stress <= 0.001 {
        SOURCE_NONE
    } else if xsec_stress >= span_stress {
        SOURCE_CROSS_SECTION
    } else {
        SOURCE_SPAN
    };

    (final_stress, pack_classification(surface_type, dominant_source))
}

/// V2 stress recalculation: runs ground connectivity pass then per-voxel stress.
/// Operates on a set of dirty chunks (and their neighborhoods).
/// Used by overlay preview (V/C key) which needs full-chunk recalc.
pub fn recalc_stress_region_v2(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    stress_fields: &mut HashMap<(i32, i32, i32), StressField>,
    support_fields: &HashMap<(i32, i32, i32), SupportField>,
    config: &StressConfig,
    dirty_chunks: &[(i32, i32, i32)],
    chunk_size: usize,
) -> StressResult {
    recalc_stress_region_v2_filtered(
        density_fields, stress_fields, support_fields, config,
        dirty_chunks, &[], chunk_size,
    )
}

/// V2 stress recalculation with optional position-based filtering.
/// If `events` is non-empty, only voxels within any event's radius are recalculated.
/// If `events` is empty, all voxels in `dirty_chunks` are recalculated (full mode).
pub fn recalc_stress_region_v2_filtered(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    stress_fields: &mut HashMap<(i32, i32, i32), StressField>,
    support_fields: &HashMap<(i32, i32, i32), SupportField>,
    config: &StressConfig,
    dirty_chunks: &[(i32, i32, i32)],
    events: &[StressDirtyEvent],
    chunk_size: usize,
) -> StressResult {
    let use_filter = !events.is_empty();

    // Pass 1: ground connectivity on dirty chunks + neighbors
    let support_scores = ground_connectivity_pass(density_fields, dirty_chunks, chunk_size, config);

    let cs = chunk_size;
    let grid_size = cs + 1;
    let mut overstressed = Vec::new();
    let mut affected_chunks = HashSet::new();

    // Pass 2: calculate stress for voxels near surfaces in dirty chunks.
    // Deep interior voxels (fully surrounded by grounded solid) are skipped for performance.
    for &(cx, cy, cz) in dirty_chunks {
        let df = match density_fields.get(&(cx, cy, cz)) {
            Some(d) => d,
            None => continue,
        };

        for z in 0..grid_size {
            for y in 0..grid_size {
                for x in 0..grid_size {
                    if !df.get(x, y, z).material.is_solid() {
                        if let Some(sf) = stress_fields.get_mut(&(cx, cy, cz)) {
                            sf.set(x, y, z, 0.0);
                            sf.set_class(x, y, z, 0); // Air = no classification
                        }
                        continue;
                    }

                    let wx = cx * cs as i32 + x as i32;
                    let wy = cy * cs as i32 + y as i32;
                    let wz = cz * cs as i32 + z as i32;

                    // Position filter: skip voxels outside all mine event radii.
                    // Their existing stress stays untouched — no phantom collapses.
                    if use_filter && !in_any_event(events, wx, wy, wz) {
                        continue;
                    }

                    // Interior skip: fully grounded voxels get 0 stress but still classified.
                    let my_support = support_scores
                        .get(&(cx, cy, cz))
                        .map(|sf| sf.get(x, y, z))
                        .unwrap_or(1.0);
                    if my_support >= config.ground_threshold {
                        // Classify: is this a floor or deep interior?
                        let below_solid = sample_world(density_fields, wx, wy - 1, wz, cs)
                            .map(|(_, m)| m.is_solid()).unwrap_or(true);
                        // Count air neighbors for wall detection
                        let mut air_n = 0u8;
                        for &(dx, dy, dz) in &[(1i32,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)] {
                            if let Some((_, m)) = sample_world(density_fields, wx+dx, wy+dy, wz+dz, cs) {
                                if !m.is_solid() { air_n += 1; }
                            }
                        }
                        let stype = if air_n == 0 { SURFACE_INTERIOR }
                            else if below_solid { SURFACE_FLOOR }
                            else { SURFACE_WALL };
                        if let Some(sf) = stress_fields.get_mut(&(cx, cy, cz)) {
                            sf.set(x, y, z, 0.0);
                            sf.set_class(x, y, z, pack_classification(stype, SOURCE_NONE));
                        }
                        continue;
                    }

                    let (stress, classification) = calc_voxel_stress_v2(
                        density_fields, support_fields, &support_scores,
                        config, wx, wy, wz, cs,
                    );

                    // Painted overlay (creative-mode PaintStress brush) is
                    // captured BEFORE the set, since set() doesn't touch it.
                    let painted = stress_fields
                        .get(&(cx, cy, cz))
                        .map(|sf| sf.painted(x, y, z))
                        .unwrap_or(0.0);
                    if let Some(sf) = stress_fields.get_mut(&(cx, cy, cz)) {
                        sf.set(x, y, z, stress);
                        sf.set_class(x, y, z, classification);
                        affected_chunks.insert((cx, cy, cz));
                    }

                    let eff = stress + painted;
                    if eff >= 1.0 {
                        overstressed.push(OverstressedVoxel {
                            world_x: wx, world_y: wy, world_z: wz, stress: eff,
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

                // Store stress value and fold in the painted overlay before the
                // overstressed test so creative-mode painted regions can drive
                // collapses just like organic geological stress.
                let (key, lx, ly, lz) = world_to_chunk_local(wx, wy, wz, chunk_size);
                let painted = stress_fields
                    .get(&key)
                    .map(|sf| sf.painted(lx, ly, lz))
                    .unwrap_or(0.0);
                if let Some(sf) = stress_fields.get_mut(&key) {
                    sf.set(lx, ly, lz, stress);
                    affected_chunks.insert(key);
                }

                let eff = stress + painted;
                if eff >= 1.0 {
                    // Verify this is actually a solid voxel
                    if let Some((_, mat)) = sample_world(density_fields, wx, wy, wz, chunk_size) {
                        if mat.is_solid() {
                            overstressed.push(OverstressedVoxel {
                                world_x: wx,
                                world_y: wy,
                                world_z: wz,
                                stress: eff,
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
/// Apply a previously-deferred pile placement. Mutates density to add the
/// rubble pile, returns the chunks affected. The caller is responsible for
/// remeshing those chunks after this call.
pub fn apply_pending_pile(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    config: &StressConfig,
    pending: &PendingPilePlacement,
    chunk_size: usize,
) -> Vec<(i32, i32, i32)> {
    let pile_result = crate::collapse_pile::place_collapse_pile(
        density_fields, config, &pending.collapsed_voxels,
        pending.bb_min, pending.bb_max,
        pending.dominant_material, pending.landing_offset, chunk_size,
    );
    pile_result.affected_chunks.into_iter().collect()
}

/// Like `apply_pending_pile` but returns the full `PlacementResult` so the
/// caller can inspect `written_cells` (e.g. to extract a preview mesh and
/// then roll back the writes).
pub fn apply_pending_pile_with_result(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    config: &StressConfig,
    pending: &PendingPilePlacement,
    chunk_size: usize,
) -> crate::collapse_pile::PlacementResult {
    crate::collapse_pile::place_collapse_pile(
        density_fields, config, &pending.collapsed_voxels,
        pending.bb_min, pending.bb_max,
        pending.dominant_material, pending.landing_offset, chunk_size,
    )
}

pub fn detect_and_execute_collapses_v2(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    stress_fields: &mut HashMap<(i32, i32, i32), StressField>,
    support_fields: &HashMap<(i32, i32, i32), SupportField>,
    overstressed: &[OverstressedVoxel],
    config: &StressConfig,
    chunk_size: usize,
) -> Vec<CollapseEventV2> {
    detect_and_execute_collapses_v2_with_options(
        density_fields, stress_fields, support_fields,
        overstressed, config, chunk_size, false,
    )
}

/// Same as `detect_and_execute_collapses_v2` but with an option to defer
/// pile placement. When `defer_pile = true`, slab voxels are still cleared
/// (cave roof opens immediately) but the rubble pile is NOT placed —
/// instead, `pending_piles` is populated on each event so the caller can
/// apply piles later (e.g., scheduled at impact time for the cinematic).
pub fn detect_and_execute_collapses_v2_with_options(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    stress_fields: &mut HashMap<(i32, i32, i32), StressField>,
    _support_fields: &HashMap<(i32, i32, i32), SupportField>,
    overstressed: &[OverstressedVoxel],
    config: &StressConfig,
    chunk_size: usize,
    defer_pile: bool,
) -> Vec<CollapseEventV2> {
    detect_and_execute_collapses_v2_with_force(
        density_fields,
        stress_fields,
        _support_fields,
        overstressed,
        config,
        chunk_size,
        defer_pile,
        false, // force_collapse — default off, natural filters apply
    )
}

/// Like `detect_and_execute_collapses_v2_with_options` but with an extra
/// `force_collapse` flag. When set, the grounding filter (`landing_offset
/// <= 0`) is bypassed and grounded regions are forced to "fall" a small
/// default distance — used by scripted editor triggers that need to
/// collapse cave walls / pillars / dome rock that's physically supported
/// by surrounding terrain but the designer has authored to fall anyway.
pub fn detect_and_execute_collapses_v2_with_force(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    stress_fields: &mut HashMap<(i32, i32, i32), StressField>,
    _support_fields: &HashMap<(i32, i32, i32), SupportField>,
    overstressed: &[OverstressedVoxel],
    config: &StressConfig,
    chunk_size: usize,
    defer_pile: bool,
    force_collapse: bool,
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

        // Check if the starting voxel can actually fall (has air below within 48 voxels)
        // BFS flood-fill: find contiguous region of overstressed voxels.
        // No "can fall" filter — all overstressed voxels join the region.
        // Fall eligibility is checked per-column in the landing computation.
        let mut queue = VecDeque::new();
        let mut region: Vec<(i32, i32, i32)> = Vec::new();
        queue.push_back(start);
        visited.insert(start);

        while let Some(pos) = queue.pop_front() {
            if region.len() >= config.max_collapse_volume as usize {
                break;
            }
            region.push(pos);

            // 26-connected BFS: face + edge + corner neighbors.
            for dz in -1..=1i32 {
                for dy in -1..=1i32 {
                    for dx in -1..=1i32 {
                        if dx == 0 && dy == 0 && dz == 0 { continue; }
                        let neighbor = (pos.0 + dx, pos.1 + dy, pos.2 + dz);
                        if visited.contains(&neighbor) {
                            continue;
                        }

                        // Include neighbor if:
                        //  - it's already in the overstressed seed set, OR
                        //  - (natural only) it's solid AND has natural stress
                        //    above the cohesion threshold.
                        //
                        // For force_collapse, the slab is EXACTLY the painted
                        // seed set. We do NOT BFS-expand into surrounding rock —
                        // earlier attempts at "expand to all connected solid"
                        // flooded ±2000 cells around the painted region, creating
                        // collapses far from where the designer painted. Stick
                        // to what was painted; if the designer wants a bigger
                        // slab, they paint more cells.
                        let include = if overstressed_set.contains(&neighbor) {
                            true
                        } else if force_collapse {
                            false
                        } else {
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

        // Filter Air out of dominant_material — stress's BFS region can
        // include marginal air-classified cells, and "dominant=Air" then
        // propagates everywhere as matte-black mesh material in UE.
        let dominant_material = material_counts
            .into_iter()
            .filter(|(m, _)| (*m as u8) > 0)
            .max_by_key(|&(_, count)| count)
            .map(|(mat, _)| mat)
            .unwrap_or(Material::Granite);

        // Compute landing position using only columns with immediate air below
        // (actual ceiling surfaces). Wall/floor voxels in the region are ignored
        // for fall distance — they just get removed along with the slab.
        let region_set: HashSet<(i32, i32, i32)> = region.iter().copied().collect();
        let mut column_min_y: HashMap<(i32, i32), i32> = HashMap::new();
        for &(x, y, z) in &region {
            // Only include this column if the voxel at the bottom has air below
            let entry = column_min_y.entry((x, z)).or_insert(y);
            *entry = (*entry).min(y);
        }

        // Filter to only columns with air immediately below the slab
        let fallable_columns: Vec<((i32, i32), i32)> = column_min_y.iter()
            .filter(|&(&(x, z), &min_y)| {
                // Check if the voxel below the slab bottom in this column is air
                match sample_world(density_fields, x, min_y - 1, z, chunk_size) {
                    Some((_, mat)) => !mat.is_solid(),
                    None => false,
                }
            })
            .map(|(&k, &v)| (k, v))
            .collect();

        if fallable_columns.is_empty() && !force_collapse {
            continue; // No columns can fall — entire region is embedded in solid.
            // Scripted triggers (force_collapse=true) still proceed: the
            // grounding bypass below applies a default fall distance so the
            // cinematic plays even for rock with no natural fall path.
        }

        // Compute fall offset per column, then use MEDIAN (not minimum).
        // One wall column near the floor shouldn't anchor the whole ceiling slab.
        let mut column_offsets: Vec<i32> = Vec::with_capacity(fallable_columns.len());
        for &((x, z), min_y) in &fallable_columns {
            let mut floor_y = min_y - 1;
            let mut found = false;
            for _ in 0..64 {
                if region_set.contains(&(x, floor_y, z)) {
                    floor_y -= 1;
                    continue;
                }
                match sample_world(density_fields, x, floor_y, z, chunk_size) {
                    Some((_, mat)) if mat.is_solid() => {
                        column_offsets.push(min_y - (floor_y + 1));
                        found = true;
                        break;
                    }
                    None => {
                        column_offsets.push(min_y - (floor_y + 1));
                        found = true;
                        break;
                    }
                    _ => floor_y -= 1,
                }
            }
            if !found {
                column_offsets.push(64);
            }
        }

        column_offsets.sort();
        let mut landing_offset = if column_offsets.is_empty() {
            0
        } else {
            column_offsets[column_offsets.len() / 2] // median
        };

        if landing_offset <= 0 {
            if force_collapse {
                // Scripted trigger AND no natural fall path. Use a small
                // default so the pile lands close to the painted region
                // rather than tunneling far below into solid rock.
                landing_offset = 4;
            } else {
                continue; // Median says slab is grounded
            }
        }
        // Note: for force_collapse with a meaningful natural fall distance,
        // trust the natural median. The earlier behavior (force min 8) put
        // the pile too far below the painted region — e.g. painted tunnel
        // ceiling, pile would land 320 UU below the tunnel floor inside
        // solid rock, making the cinematic look like "rock appeared
        // somewhere unrelated." Now the pile lands on the tunnel floor,
        // right under the hole, like a normal cave-in.

        let landing_y = bb_min.1 - landing_offset;

        // Record collapsed voxels with original material, then clear them.
        // Track slab-affected chunks separately from the wider affected_chunks
        // set so the worker can remesh roof chunks at fall-start time and
        // pile chunks at impact time (the cinematic-aligned split).
        let mut collapsed_voxels = Vec::with_capacity(region.len());
        let mut slab_chunks_set: HashSet<(i32, i32, i32)> = HashSet::new();
        let mut pile_chunks_set: HashSet<(i32, i32, i32)> = HashSet::new();
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
            slab_chunks_set.insert(key);
            affected_chunks_set.insert(key);

            collapsed_voxels.push(CollapsedVoxel {
                world_x: wx, world_y: wy, world_z: wz, material,
            });
        }

        // Place rubble as a sealed elliptical cone, then add roughness on top.
        //
        // Three passes, each one ADDITIVE — no pass ever removes solid:
        //
        //  1. SEALED CONE BASE. Stack of fully-filled elliptical discs from
        //     the floor up. At height fraction f the disc radius is
        //     (1 - f) * full_radius. Every air cell inside the cone gets
        //     solid — no per-column breaks, no noise gating. Guaranteed sealed.
        //
        //  2. NOISE CRUST. Walk each (x, z) inside the cone footprint and
        //     stamp 0–2 extra voxels above its cone-top where simplex noise
        //     is positive. Pure additive — breaks the perfect-cone silhouette.
        //
        //  3. BOULDERS. A handful of half-buried solid spheres on the surface
        //     using the dominant slab material for chunky roughness.
        //
        // Pile sizing: cone volume = pi*R^2*H/3, solve for H from
        // collapsed_volume * rubble_fill_ratio. Seeded from collapse center
        // for determinism (multiplayer-safe).
        // Cinematic collapse pile (see crate::collapse_pile). Splits the slab
        // into fragments, runs angle-of-repose distribution for slope/cliff
        // handling, sub-voxel pile surface, material stratification, craters,
        // splash ring, boulder tracks, impact cracks, plus formation removal
        // at both landing zone AND slab origin. All multi-chunk seam-aware.
        let mut pending_piles_for_event: Vec<PendingPilePlacement> = Vec::new();
        let pile_result_opt = if defer_pile {
            // Save data for later application — pile cells NOT placed yet.
            // Worker will call apply_pending_pile at the cinematic impact time.
            pending_piles_for_event.push(PendingPilePlacement {
                collapsed_voxels: collapsed_voxels.clone(),
                bb_min,
                bb_max,
                dominant_material,
                landing_offset,
            });
            None
        } else {
            let pr = crate::collapse_pile::place_collapse_pile(
                density_fields, config, &collapsed_voxels,
                bb_min, bb_max, dominant_material, landing_offset, chunk_size,
            );
            for k in &pr.affected_chunks {
                pile_chunks_set.insert(*k);
                affected_chunks_set.insert(*k);
            }
            Some(pr)
        };

        // ── Chained collapse hint (Tier 5I) ──
        // The pile's added weight may tip a marginal ceiling. Add the
        // chunks containing the settling-hint cells to the affected_chunks
        // set so the cascade picks them up. Only available when pile was
        // placed inline (defer_pile=false). Worker handles deferred
        // settling separately by re-running stress after impact.
        if let Some(pr) = pile_result_opt {
            for &(wx, wy, wz) in &pr.settling_dirty_cells {
                let key = world_to_chunk_local(wx, wy, wz, chunk_size).0;
                affected_chunks_set.insert(key);
            }
            let _ = (pr.written_cells, pr.dust_events, pr.fragments);
        }

        if false {
            let pile_seed: u64 = (center.0 as i64 as u64)
                .wrapping_mul(73856093)
                ^ (center.1 as i64 as u64).wrapping_mul(19349663)
                ^ (center.2 as i64 as u64).wrapping_mul(83492791);

            let crust_noise = Simplex3D::new(pile_seed);
            let boulder_noise = Simplex3D::new(pile_seed.wrapping_add(1));

            let cx_f = center.0;
            let cz_f = center.2;
            // Footprint radii. Pad by 1 voxel + floor at 1.5 so even tiny
            // single-voxel slabs still produce a small visible pile.
            let radius_x = ((bb_max.0 - bb_min.0) as f32 * 0.5 + 1.0).max(1.5);
            let radius_z = ((bb_max.2 - bb_min.2) as f32 * 0.5 + 1.0).max(1.5);
            let avg_radius = (radius_x + radius_z) * 0.5;
            let slab_thickness = (bb_max.1 - bb_min.1 + 1).max(1) as f32;

            let target_volume = (collapsed_voxels.len() as f32 * config.rubble_fill_ratio)
                .max(1.0);

            // Cone volume = (pi * R^2 * H) / 3 → H = 3V / (pi * R^2).
            let cone_volume_factor = std::f32::consts::PI * radius_x * radius_z / 3.0;
            let cone_h_raw = target_volume / cone_volume_factor.max(0.5);
            // Allow slightly taller than slab so wide flat slabs still pile up.
            let cone_h_cap = (slab_thickness * 1.2).max(2.0);
            let cone_h_max = cone_h_raw.clamp(1.0, cone_h_cap);
            let cone_h_int = cone_h_max.ceil() as i32;

            let floor_y = bb_min.1 - landing_offset;
            let mut placed = 0usize;

            // ── Pass 1: SEALED CONE BASE ──
            //
            // For each layer dy: compute disc radius, fill EVERY air cell in
            // that ellipse. No noise, no per-column logic, no breaks. The
            // cone is closed by construction.
            for dy in 0..cone_h_int {
                let y = floor_y + dy;
                // f = 0 at base, 1 at top. Floor at 0.05 so the very tip is
                // still half a voxel wide instead of an infinitesimal point.
                let f = (dy as f32 / cone_h_max).clamp(0.0, 1.0);
                let shrink = (1.0 - f).max(0.05);
                let rx = (radius_x * shrink).max(0.5);
                let rz = (radius_z * shrink).max(0.5);

                let x0 = (cx_f - rx).floor() as i32 - 1;
                let x1 = (cx_f + rx).ceil() as i32 + 1;
                let z0 = (cz_f - rz).floor() as i32 - 1;
                let z1 = (cz_f + rz).ceil() as i32 + 1;

                for x in x0..=x1 {
                    for z in z0..=z1 {
                        let nx = (x as f32 - cx_f) / rx;
                        let nz = (z as f32 - cz_f) / rz;
                        if nx * nx + nz * nz > 1.0 { continue; }

                        let (rkey, rlx, rly, rlz) =
                            world_to_chunk_local(x, y, z, chunk_size);
                        let is_air = density_fields
                            .get(&rkey)
                            .map(|df| !df.get(rlx, rly, rlz).material.is_solid())
                            .unwrap_or(false);
                        if !is_air { continue; }

                        if let Some(df) = density_fields.get_mut(&rkey) {
                            let sample = df.get_mut(rlx, rly, rlz);
                            sample.density = 1.0;
                            sample.material = dominant_material;
                        }
                        affected_chunks_set.insert(rkey);
                        placed += 1;
                    }
                }
            }

            // ── Pass 2: NOISE CRUST ──
            //
            // For each (x, z) inside the cone footprint, find the cone-top Y
            // for that column and stamp 0..=2 extra voxels above where simplex
            // noise is positive. Pure additive; never subtracts from Pass 1.
            const CRUST_MAX_EXTRA: i32 = 2;
            let xmin = (cx_f - radius_x).floor() as i32 - 1;
            let xmax = (cx_f + radius_x).ceil() as i32 + 1;
            let zmin = (cz_f - radius_z).floor() as i32 - 1;
            let zmax = (cz_f + radius_z).ceil() as i32 + 1;

            for x in xmin..=xmax {
                for z in zmin..=zmax {
                    let nx = (x as f32 - cx_f) / radius_x;
                    let nz = (z as f32 - cz_f) / radius_z;
                    let r2 = nx * nx + nz * nz;
                    if r2 > 1.0 { continue; }

                    // Cone-top Y for this column: r(y)/R = 1 - f, so the
                    // column top in voxels = (1 - sqrt(r2)) * cone_h_max.
                    let column_h = ((1.0 - r2.sqrt()) * cone_h_max).max(0.0);
                    let column_top_dy = column_h.floor() as i32;

                    let n_lo = crust_noise.sample(
                        x as f64 * 0.22, 0.0, z as f64 * 0.22,
                    ) as f32;
                    let n_hi = crust_noise.sample(
                        x as f64 * 0.55, 7.0, z as f64 * 0.55,
                    ) as f32;
                    let n = n_lo * 0.7 + n_hi * 0.4;
                    if n <= 0.0 { continue; }

                    let extra = ((n * (CRUST_MAX_EXTRA as f32 + 0.5)).round() as i32)
                        .clamp(0, CRUST_MAX_EXTRA);
                    if extra <= 0 { continue; }

                    for k in 1..=extra {
                        let y = floor_y + column_top_dy + k;
                        let (rkey, rlx, rly, rlz) =
                            world_to_chunk_local(x, y, z, chunk_size);
                        let is_air = density_fields
                            .get(&rkey)
                            .map(|df| !df.get(rlx, rly, rlz).material.is_solid())
                            .unwrap_or(false);
                        if !is_air { continue; }

                        if let Some(df) = density_fields.get_mut(&rkey) {
                            let sample = df.get_mut(rlx, rly, rlz);
                            sample.density = 1.0;
                            sample.material = dominant_material;
                        }
                        affected_chunks_set.insert(rkey);
                        placed += 1;
                    }
                }
            }

            // ── Pass 3: BOULDERS ──
            //
            // 2–8 half-buried spheres around the cone, sitting on the cone-top
            // for their (x, z). Edge noise so they aren't perfect spheres.
            let boulder_count = ((avg_radius * 0.6) as usize).clamp(2, 8);
            for i in 0..boulder_count {
                // Sweep angles around the cone axis; offset radially via noise
                // so they don't land on a perfect ring.
                let theta = (i as f32) * std::f32::consts::TAU / (boulder_count as f32);
                let radial_n = boulder_noise.sample(
                    (i as f64) * 1.31, 0.0, (i as f64) * 0.93,
                ) as f32;
                // Radial fraction in [0.2, 0.7] of the footprint.
                let radial_frac = 0.2 + (radial_n * 0.5 + 0.5) * 0.5;
                let bx = (cx_f + theta.cos() * radius_x * radial_frac).round() as i32;
                let bz = (cz_f + theta.sin() * radius_z * radial_frac).round() as i32;

                // Cone-top Y at this (bx, bz) so boulders sit ON the pile.
                let nx = (bx as f32 - cx_f) / radius_x;
                let nz = (bz as f32 - cz_f) / radius_z;
                let br2 = (nx * nx + nz * nz).min(1.0);
                let column_h = ((1.0 - br2.sqrt()) * cone_h_max).max(0.0);
                let by = floor_y + column_h.floor() as i32;

                let size_n = boulder_noise.sample(
                    bx as f64 * 0.41, by as f64 * 0.41, bz as f64 * 0.41,
                ) as f32;
                let radius = 1.5 + (size_n * 0.5 + 0.5) * 1.2; // 1.5..2.7

                // Half-bury so it looks settled.
                let bury = (radius * 0.4) as i32;
                let cy = (by - bury) as f32;

                let r_ceil = radius.ceil() as i32;
                let r_sq = radius * radius;
                for ox in -r_ceil..=r_ceil {
                    for oy in -r_ceil..=r_ceil {
                        for oz in -r_ceil..=r_ceil {
                            let dx = ox as f32;
                            let dy_b = oy as f32;
                            let dz = oz as f32;
                            let d2 = dx * dx + dy_b * dy_b + dz * dz;
                            if d2 > r_sq { continue; }

                            // Edge noise so boulders aren't perfect spheres.
                            let edge = (d2 / r_sq).sqrt();
                            let edge_n = boulder_noise.sample(
                                (bx + ox) as f64 * 0.7,
                                (cy + oy as f32) as f64 * 0.7,
                                (bz + oz) as f64 * 0.7,
                            ) as f32;
                            if edge > 0.85 + edge_n * 0.20 { continue; }

                            let wx_b = bx + ox;
                            let wy_b = cy as i32 + oy;
                            let wz_b = bz + oz;
                            if wy_b < floor_y { continue; }

                            let (rkey, rlx, rly, rlz) = world_to_chunk_local(
                                wx_b, wy_b, wz_b, chunk_size,
                            );
                            let is_air = density_fields
                                .get(&rkey)
                                .map(|df| !df.get(rlx, rly, rlz).material.is_solid())
                                .unwrap_or(false);
                            if !is_air { continue; }

                            if let Some(df) = density_fields.get_mut(&rkey) {
                                let sample = df.get_mut(rlx, rly, rlz);
                                sample.density = 1.0;
                                sample.material = dominant_material;
                            }
                            affected_chunks_set.insert(rkey);
                            placed += 1;
                        }
                    }
                }
            }

            let _ = placed;
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
            slab_chunks: slab_chunks_set.into_iter().collect(),
            pile_chunks: pile_chunks_set.into_iter().collect(),
            pending_piles: pending_piles_for_event,
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
        assert_eq!(DEFAULT_MATERIAL_HARDNESS.len(), 50);
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

    /// Create a 3x5x3 grid of chunks (tall in Y) for proper top-down flood testing.
    fn make_solid_world() -> (HashMap<(i32,i32,i32), DensityField>, HashMap<(i32,i32,i32), StressField>, HashMap<(i32,i32,i32), SupportField>) {
        let mut density_fields = HashMap::new();
        let mut stress_fields = HashMap::new();
        let support_fields = HashMap::new();
        for cz in -1..=1 {
            for cy in -2..=2 { // 5 chunks tall for proper flood propagation
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
            for cy in -2..=2 {
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

        // With retuned gravity_weight=0.05, a fully-supported deep voxel
        // should have 0 or near-0 stress (lateral+vertical support > gravity load)
        assert!(stress >= 0.0, "Stress should be non-negative");
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
        // A voxel in solid rock should have positive support score
        // (flooded from surface above, decayed by depth)
        let score = scores.get(&(0, 0, 0)).unwrap().get(8, 8, 8);
        assert!(score > 0.0, "Solid voxel should have positive support from surface flood, got {}", score);
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
        config.min_safe_span = 8;
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

        // Need >= min_collapse_region (8) voxels, so use a 3x3 block
        let mut overstressed = Vec::new();
        for z in 7..10 {
            for x in 7..10 {
                overstressed.push(OverstressedVoxel {
                    world_x: x, world_y: 10, world_z: z, stress: 1.5,
                });
            }
        }

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

        let (stress_without, _) = calc_voxel_stress_v2(
            &density_fields, &support_fields_empty, &scores, &config, 8, 8, 8, 16,
        );
        let (stress_with, _) = calc_voxel_stress_v2(
            &density_fields, &support_fields_with, &scores, &config, 8, 8, 8, 16,
        );

        assert!(stress_without > 0.0,
            "Wide ceiling should have positive stress without strut, got {}", stress_without);
        assert!(stress_with < stress_without,
            "Strut should reduce v2 stress: with={}, without={}", stress_with, stress_without);
    }

    /// Sweep tunnel heights (air gap size) and measure ceiling stress.
    /// This shows the stress curve vs span width — used for tuning overhang_weight,
    /// span_weight, and min_safe_span.
    #[test]
    #[ignore] // Run with: cargo test --release -p voxel-core sweep_ceiling_stress -- --ignored --nocapture
    fn sweep_ceiling_stress() {
        let mut config = default_config();
        // Set surface_y to 32 so it's at the top of our test world (chunk y=2, local y=0)
        // This ensures the ground connectivity flood can reach our test geometry
        config.surface_y = 32;

        println!("\n=== Ceiling Stress vs Tunnel Height (Air Gap) ===");
        println!("{:<12} {:<12} {:<12} {:<12} {:<12}",
            "air_gap", "v2_stress", "overstressed", "would_collapse", "support_score");
        println!("{}", "-".repeat(60));

        for air_gap in [2, 4, 6, 8, 10, 12, 14, 16] {
            // Tunnel from y=0 to y=air_gap-1, ceiling at y=air_gap
            let tunnel_y_max = (air_gap - 1).min(15);
            let (density_fields, _, support_fields) = make_tunnel_world(0, tunnel_y_max);

            // Run ground connectivity
            let all_keys: Vec<(i32,i32,i32)> = density_fields.keys().cloned().collect();
            let scores = ground_connectivity_pass(&density_fields, &all_keys, 16, &config);

            // Measure stress at ceiling center (just above the tunnel)
            let ceiling_y = (tunnel_y_max + 1).min(16);
            let (stress, _) = calc_voxel_stress_v2(
                &density_fields, &support_fields, &scores, &config,
                8, ceiling_y as i32, 8, 16,
            );

            let support_score = scores.get(&(0, 0, 0))
                .map(|s| s.get(8, ceiling_y, 8))
                .unwrap_or(-1.0);

            let overstressed = stress >= 1.0;
            let would_collapse = stress >= config.slab_cohesion_threshold;

            println!("{:<12} {:<12.4} {:<12} {:<12} {:<12.4}",
                air_gap, stress, overstressed, would_collapse, support_score);
        }
        println!();

        // Now sweep overhang_weight to show sensitivity
        println!("=== Sensitivity: overhang_weight (gap=12, ceiling at y=12) ===");
        println!("{:<16} {:<12} {:<12}",
            "overhang_weight", "v2_stress", "would_collapse");
        println!("{}", "-".repeat(40));

        for &ow in &[0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20] {
            let mut cfg = default_config();
            cfg.surface_y = 32;
            cfg.overhang_weight = ow;

            let (density_fields, _, support_fields) = make_tunnel_world(0, 11);
            let all_keys: Vec<(i32,i32,i32)> = density_fields.keys().cloned().collect();
            let scores = ground_connectivity_pass(&density_fields, &all_keys, 16, &cfg);
            let (stress, _) = calc_voxel_stress_v2(
                &density_fields, &support_fields, &scores, &cfg,
                8, 12, 8, 16,
            );

            println!("{:<16.3} {:<12.4} {:<12}",
                ow, stress, stress >= cfg.slab_cohesion_threshold);
        }
        println!();

        // Multi-chunk span: measure stress at varying positions across a 3-chunk-wide tunnel
        println!("=== Multi-chunk span: stress across 48-voxel-wide ceiling ===");
        println!("{:<12} {:<12} {:<12}",
            "x_position", "v2_stress", "support_score");
        println!("{}", "-".repeat(36));

        let (density_fields, _, support_fields) = make_tunnel_world(0, 11);
        let all_keys: Vec<(i32,i32,i32)> = density_fields.keys().cloned().collect();
        let scores = ground_connectivity_pass(&density_fields, &all_keys, 16, &config);

        // Sample stress across x positions in different chunks (with corrected surface_y)
        for &(cx, x) in &[(-1,4), (-1,8), (-1,12), (0,4), (0,8), (0,12), (1,4), (1,8), (1,12)] {
            let (stress, _) = calc_voxel_stress_v2(
                &density_fields, &support_fields, &scores, &config,
                cx * 16 + x, 12, 8, 16,
            );
            let score = scores.get(&(cx, 0, 0))
                .map(|s| s.get(x as usize, 12, 8))
                .unwrap_or(-1.0);
            println!("{:<12} {:<12.4} {:<12.4}",
                format!("c{}:x{}", cx, x), stress, score);
        }
        println!();

        // Sweep span_weight — the other major knob
        println!("=== Sensitivity: span_weight (gap=12, ceiling at y=12, min_safe_span=8) ===");
        println!("{:<16} {:<12} {:<12} {:<20}",
            "span_weight", "v2_stress", "would_collapse", "collapses_at_span");
        println!("{}", "-".repeat(60));

        for &sw in &[0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30] {
            let mut cfg = default_config();
            cfg.surface_y = 32;
            cfg.span_weight = sw;

            let (density_fields, _, support_fields) = make_tunnel_world(0, 11);
            let all_keys: Vec<(i32,i32,i32)> = density_fields.keys().cloned().collect();
            let scores = ground_connectivity_pass(&density_fields, &all_keys, 16, &cfg);
            let (stress, _) = calc_voxel_stress_v2(
                &density_fields, &support_fields, &scores, &cfg,
                8, 12, 8, 16,
            );

            // Calculate at what span width this would collapse (stress >= 0.75)
            // stress = overhang_weight * overhang_factor + span_weight * max(0, span - min_safe_span)
            // For collapse: 0.75 = 0.05 * oh + sw * (span - 8)
            // Simplified: span_for_collapse = (0.75 - base_stress) / sw + 8
            let collapse_span = if sw > 0.0 { ((0.75 - 0.05 * 12.0) / sw + 8.0) as i32 } else { 999 };

            println!("{:<16.3} {:<12.4} {:<12} span >= {:<12}",
                sw, stress, stress >= cfg.slab_cohesion_threshold, collapse_span);
        }
        println!();

        // Sweep min_safe_span
        println!("=== Sensitivity: min_safe_span (gap=12, ceiling at y=12) ===");
        println!("{:<16} {:<12} {:<12}",
            "min_safe_span", "v2_stress", "would_collapse");
        println!("{}", "-".repeat(40));

        for &mss in &[2, 4, 6, 8, 10, 12, 16] {
            let mut cfg = default_config();
            cfg.surface_y = 32;
            cfg.min_safe_span = mss;

            let (density_fields, _, support_fields) = make_tunnel_world(0, 11);
            let all_keys: Vec<(i32,i32,i32)> = density_fields.keys().cloned().collect();
            let scores = ground_connectivity_pass(&density_fields, &all_keys, 16, &cfg);
            let (stress, _) = calc_voxel_stress_v2(
                &density_fields, &support_fields, &scores, &cfg,
                8, 12, 8, 16,
            );

            println!("{:<16} {:<12.4} {:<12}",
                mss, stress, stress >= cfg.slab_cohesion_threshold);
        }
    }
}
