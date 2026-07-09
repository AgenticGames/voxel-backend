//! Core stress/support field types, support-tier tuning tables, hardness
//! constants, and collapse/result data structures.
//!
//! Behavior-preserving split of the former `stress.rs` god file.

use std::collections::HashMap;

use crate::material::Material;

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

    /// Effective stress = max(0, base + painted overlay).
    /// Use this where you want player-painted stress to influence behavior
    /// (collapse-failure rolls, overstressed test, debug viz).
    ///
    /// Base stress can be NEGATIVE: the recalc passes store strut relief
    /// surplus below zero so struts can offset painted stress. The clamp
    /// belongs on the SUM — clamping the parts would reintroduce the
    /// painted floor struts used to be powerless against.
    #[inline]
    pub fn effective(&self, x: usize, y: usize, z: usize) -> f32 {
        let i = self.index(x, y, z);
        let base = self.stress[i];
        let painted = if self.painted_stress.is_empty() {
            0.0
        } else {
            self.painted_stress[i]
        };
        (base + painted).max(0.0)
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

/// Support (strut) type enum.
///
/// 5-tier metal/crystal lineup (overhauled 2026-05-26): Copper → Iron → Steel →
/// Crystal → Mithril. The 3 stone struts from the Feb 2026 lineup
/// (Slate/Granite/Limestone) were dropped — they were functionally identical
/// at hardness=0.95 and didn't pay for their slot in the type table. Legacy
/// IDs 1/2/3 in older saves migrate to Copper (1) on load — see
/// `voxel-ffi/src/delta.rs` save-format v4 read path.
///
/// Fits in 3 bits, leaving room if a future tier is needed.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportType {
    None = 0,
    Copper = 1,
    Iron = 2,
    Steel = 3,
    Crystal = 4,
    Mithril = 5,
}

impl SupportType {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => SupportType::Copper,
            2 => SupportType::Iron,
            3 => SupportType::Steel,
            4 => SupportType::Crystal,
            5 => SupportType::Mithril,
            _ => SupportType::None,
        }
    }

    /// Map a legacy (pre-2026-05-26) SupportType byte to the new lineup.
    /// Legacy values 1/2/3 (Slate/Granite/Limestone) all collapse to Copper.
    /// Legacy 4/5/6/7 (Copper/Iron/Steel/Crystal) shift down one slot.
    /// Used by ChunkSnapshot v3→v4 migration and the UE save-loader.
    pub fn from_legacy_u8(v: u8) -> Self {
        match v {
            1 | 2 | 3 => SupportType::Copper,   // Slate/Granite/Limestone → Copper
            4 => SupportType::Copper,           // Old Copper (was idx 4)
            5 => SupportType::Iron,             // Old Iron (was idx 5)
            6 => SupportType::Steel,            // Old Steel (was idx 6)
            7 => SupportType::Crystal,          // Old Crystal (was idx 7)
            _ => SupportType::None,
        }
    }
}

/// Per-tier tuning. One entry per `SupportType` variant (index 0 = None, unused).
///
/// - `hardness`: flat reduction this strut subtracts from raw stress, falling off
///   as `hardness / dist` per neighbor voxel inside `radius`. ×10 the Feb 2026
///   values so the reduction actually moves the v2 span-stress needle.
/// - `radius`: per-strut sphere of influence (voxels). Replaces the old single
///   `StressConfig::support_radius` — each tier owns its reach.
/// - `max_hp`: initial HP assigned at placement. Decremented per recalc by
///   `max(0, load_borne - hp_decay_threshold) * HP_DAMAGE_SCALE`.
/// - `hp_decay_threshold`: load-borne amount the strut can absorb each recalc
///   tick before HP starts ticking down. Higher tiers idle under more load.
#[derive(Debug, Clone, Copy)]
pub struct StrutTuning {
    pub hardness: f32,
    pub radius: u8,
    pub max_hp: u16,
    pub hp_decay_threshold: f32,
}

/// Default per-tier tuning table. Indexed by `SupportType as usize`.
///
/// Tier shape (gentle ramp, each tier owns one differentiating axis):
/// - Copper  (T1): starter — cheapest, smallest radius, low HP.
/// - Iron    (T2): balanced workhorse — bread & butter for active mining.
/// - Steel   (T3): wide-radius specialist — covers more area per strut.
/// - Crystal (T4): HP tank — holds the vault under brutal load.
/// - Mithril (T5): endgame all-rounder — Spider Queen room anchor.
pub const STRUT_TUNING: [StrutTuning; 6] = [
    // None
    StrutTuning { hardness: 0.0, radius: 0, max_hp: 0,    hp_decay_threshold: 0.0 },
    // Copper (T1)
    StrutTuning { hardness:  8.0, radius: 2, max_hp:   50, hp_decay_threshold: 0.5 },
    // Iron (T2)
    StrutTuning { hardness: 14.0, radius: 3, max_hp:  150, hp_decay_threshold: 1.0 },
    // Steel (T3) — wide radius for area coverage
    StrutTuning { hardness: 18.0, radius: 4, max_hp:  300, hp_decay_threshold: 1.5 },
    // Crystal (T4) — HP tank
    StrutTuning { hardness: 25.0, radius: 3, max_hp:  800, hp_decay_threshold: 2.0 },
    // Mithril (T5) — endgame
    StrutTuning { hardness: 35.0, radius: 5, max_hp: 2000, hp_decay_threshold: 2.5 },
];

/// Maximum `radius` value across all tiers — used as the bounding box for
/// `any_supports_in_radius_box` short-circuits when the caller doesn't yet
/// know which specific tier sits where. Recompute if STRUT_TUNING changes.
pub const MAX_STRUT_RADIUS: u8 = 5;

/// Per-recalc HP-damage scale applied after subtracting `hp_decay_threshold`.
/// Tune to taste — higher = struts break faster under sustained load.
pub const HP_DAMAGE_SCALE: f32 = 1.0;

/// HP damage per voxel blocked by a strut during the cinematic mining BFS halt.
/// One mine event that would have peeled 200 voxels into a single Crystal Strut
/// (HP 800) chews through 200 * 0.5 = 100 HP — strut survives ~8 such saves.
pub const BFS_HALT_DAMAGE_SCALE: f32 = 0.5;

/// Per-voxel support data for a chunk.
///
/// Two parallel arrays:
/// - `supports[i]`: `SupportType` byte (None=0..Mithril=5). 1 byte/cell.
/// - `support_hp[i]`: lazy `Vec<u16>`. Empty when chunk has no struts; allocated
///   on first non-None `set()`. 2 bytes/cell when populated. Indexed identically.
#[derive(Debug, Clone)]
pub struct SupportField {
    pub supports: Vec<SupportType>,
    /// Per-voxel HP. Lazy-allocated: empty Vec when `non_none_count == 0`,
    /// `size^3` entries otherwise. Indexed identically to `supports`.
    pub support_hp: Vec<u16>,
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
            support_hp: Vec::new(),
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

    /// Read HP at one cell. Returns 0 if the cell has no strut OR the HP array
    /// is not yet allocated (no struts placed in this chunk yet).
    #[inline]
    pub fn get_hp(&self, x: usize, y: usize, z: usize) -> u16 {
        if self.support_hp.is_empty() {
            0
        } else {
            self.support_hp[self.index(x, y, z)]
        }
    }

    fn ensure_hp_alloc(&mut self) {
        if self.support_hp.is_empty() {
            self.support_hp = vec![0u16; self.size * self.size * self.size];
        }
    }

    /// Place / clear / replace a strut. Initializes HP to `STRUT_TUNING[type].max_hp`
    /// when placing a NEW strut (or REPLACING a different type). Preserves HP when
    /// the same type is re-`set()` at the same cell (so a touch-up doesn't refill).
    /// Clears HP to 0 when setting back to None.
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, support_type: SupportType) {
        let idx = self.index(x, y, z);
        let was = self.supports[idx];
        let was_none = was == SupportType::None;
        let is_none = support_type == SupportType::None;
        match (was_none, is_none) {
            (true, false) => self.non_none_count = self.non_none_count.saturating_add(1),
            (false, true) => self.non_none_count = self.non_none_count.saturating_sub(1),
            _ => {}
        }
        self.supports[idx] = support_type;
        // HP bookkeeping: only touch if state actually changed (different type
        // or transition to/from None). Same-type re-set preserves HP — important
        // for cinematic strut replacement that re-runs through set() each tick.
        if was != support_type {
            if is_none {
                if !self.support_hp.is_empty() {
                    self.support_hp[idx] = 0;
                }
            } else {
                self.ensure_hp_alloc();
                self.support_hp[idx] = STRUT_TUNING[support_type as usize].max_hp;
            }
        }
    }

    /// Set HP directly (used by save/load restore, debugger pokes, tests).
    /// No-op if the cell has no strut. Allocates the HP array if needed.
    pub fn set_hp(&mut self, x: usize, y: usize, z: usize, hp: u16) {
        if self.supports[self.index(x, y, z)] == SupportType::None {
            return;
        }
        self.ensure_hp_alloc();
        let idx = self.index(x, y, z);
        self.support_hp[idx] = hp;
    }

    /// Subtract `amount` HP from the cell, saturating at 0. Returns `true`
    /// when this call brought HP from `>0` down to `0` (caller should emit
    /// a StrutBroken event + clear the support). Returns `false` if no
    /// strut, already broken, or HP > 0 after subtract.
    pub fn damage_hp(&mut self, x: usize, y: usize, z: usize, amount: f32) -> bool {
        if amount <= 0.0 { return false; }
        let idx = self.index(x, y, z);
        if self.supports[idx] == SupportType::None { return false; }
        if self.support_hp.is_empty() { return false; } // shouldn't happen if support set
        let prev = self.support_hp[idx];
        if prev == 0 { return false; }
        let dec = amount.round().clamp(0.0, u16::MAX as f32) as u16;
        let next = prev.saturating_sub(dec);
        self.support_hp[idx] = next;
        next == 0
    }

    #[inline]
    pub fn has_support(&self, x: usize, y: usize, z: usize) -> bool {
        self.get(x, y, z) != SupportType::None
    }

    /// True when the cell has a strut with HP > 0. Used by the v2 stress
    /// reducer and the cinematic BFS halt to ignore broken struts that
    /// haven't yet been cleared by the worker tick.
    #[inline]
    pub fn is_strut_alive(&self, x: usize, y: usize, z: usize) -> bool {
        let idx = self.index(x, y, z);
        if self.supports[idx] == SupportType::None { return false; }
        if self.support_hp.is_empty() { return true; } // backwards-compat: assume alive
        self.support_hp[idx] > 0
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
pub(crate) fn any_supports_in_radius_box(
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

/// LEGACY: Pre-2026-05-26 single-array support hardness. Superseded by
/// `STRUT_TUNING` per-tier struct above which also owns radius + HP. Kept as
/// a back-compat ABI const for any external snapshot blob that referenced it
/// — internal code paths now sample STRUT_TUNING directly.
#[deprecated(note = "use STRUT_TUNING[type].hardness instead")]
pub const SUPPORT_HARDNESS: [f32; 6] = [
    0.0,                              // None
    STRUT_TUNING[1].hardness,         // Copper
    STRUT_TUNING[2].hardness,         // Iron
    STRUT_TUNING[3].hardness,         // Steel
    STRUT_TUNING[4].hardness,         // Crystal
    STRUT_TUNING[5].hardness,         // Mithril
];

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
    /// Struts whose HP hit 0 during this recalc (load-decay) — caller should
    /// emit `StrutBroken` events to the UE side AND clear them from the
    /// support field. The set is empty when no struts were over their decay
    /// threshold or no load-tracking was enabled.
    pub broken_struts: Vec<BrokenStrutEvent>,
}

/// One strut whose HP fell to 0 during a stress recalc tick.
/// Cell coords are local within `chunk`; sleep/worker callers convert to world.
#[derive(Debug, Clone, Copy)]
pub struct BrokenStrutEvent {
    pub chunk: (i32, i32, i32),
    pub lx: usize,
    pub ly: usize,
    pub lz: usize,
    pub support_type: SupportType,
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
