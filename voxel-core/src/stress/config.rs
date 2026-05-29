//! `StressConfig` (moved from voxel-gen) plus its serde array helper.
//!
//! Behavior-preserving split of the former `stress.rs` god file.

use serde::{Deserialize, Serialize};

use super::types::{DEFAULT_MATERIAL_HARDNESS, SUPPORT_HARDNESS};

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
    /// LEGACY: Pre-2026-05-26 single sphere-of-influence radius for struts.
    /// Superseded by `STRUT_TUNING[type].radius` (per-tier). Internal stress
    /// math now samples STRUT_TUNING directly; this field is kept only to
    /// preserve FFI struct layout for in-flight DLL/editor pairs and to
    /// document where the old knob used to live.
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
    /// LEGACY: Pre-2026-05-26 per-tier hardness array. Superseded by
    /// `STRUT_TUNING[type].hardness`. Mirrored here for ABI compatibility
    /// only; internal math ignores this field.
    pub support_hardness: [f32; 6],

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
            #[allow(deprecated)]
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
