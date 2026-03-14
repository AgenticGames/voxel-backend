use std::collections::HashMap;
use std::time::{Duration, Instant};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use voxel_core::density::DensityField;
use voxel_core::material::Material;
use voxel_core::stress::{
    StressField, SupportField, SupportType, StressConfig, CollapseEvent,
    calc_voxel_stress, post_change_stress_update_with_iterations,
    world_to_chunk_local,
};
use crate::config::CollapseConfig;
use crate::manifest::ChangeManifest;
use crate::TransformEntry;

/// Per-sub-step timing data from the collapse pass.
#[derive(Debug, Clone, Default)]
pub struct CollapseTimings {
    pub support_degradation: Duration,
    pub stress_amplification: Duration,
    pub collapse_cascade: Duration,
}

/// Result of the structural collapse pass.
#[derive(Debug)]
pub struct CollapseResult {
    pub supports_degraded: u32,
    pub collapses_triggered: u32,
    pub collapse_events: Vec<CollapseEvent>,
    pub manifest: ChangeManifest,
    pub dirty_chunks: Vec<(i32, i32, i32)>,
    pub glimpse_chunk: Option<(i32, i32, i32)>,
    pub transform_log: Vec<TransformEntry>,
    pub timings: CollapseTimings,
}

impl Default for CollapseResult {
    fn default() -> Self {
        Self {
            supports_degraded: 0,
            collapses_triggered: 0,
            collapse_events: Vec::new(),
            manifest: ChangeManifest::default(),
            dirty_chunks: Vec::new(),
            glimpse_chunk: None,
            transform_log: Vec::new(),
            timings: CollapseTimings {
                support_degradation: Duration::ZERO,
                stress_amplification: Duration::ZERO,
                collapse_cascade: Duration::ZERO,
            },
        }
    }
}

/// Degrade supports and trigger geological collapses.
pub fn apply_collapse(
    config: &CollapseConfig,
    stress_config: &StressConfig,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    stress_fields: &mut HashMap<(i32, i32, i32), StressField>,
    support_fields: &mut HashMap<(i32, i32, i32), SupportField>,
    chunks: &[(i32, i32, i32)],
    chunk_size: usize,
    rng: &mut ChaCha8Rng,
) -> CollapseResult {
    let mut result = CollapseResult::default();

    // Override stress config's rubble_fill_ratio with collapse config's value
    let mut local_stress_config = stress_config.clone();
    local_stress_config.rubble_fill_ratio = config.rubble_fill_ratio;

    // ── Step 1: Support Degradation ──
    let t_step1 = Instant::now();
    for &chunk_key in chunks {
        if let Some(support_field) = support_fields.get_mut(&chunk_key) {
            // Also need the stress field for this chunk to read current stress
            for z in 0..chunk_size {
                for y in 0..chunk_size {
                    for x in 0..chunk_size {
                        let support = support_field.get(x, y, z);
                        if support == SupportType::None {
                            continue;
                        }

                        let survival_rate = config.strut_survival[support as u8 as usize];

                        // Get stress at this voxel
                        let stress_at_voxel = stress_fields
                            .get(&chunk_key)
                            .map(|sf| sf.get(x, y, z))
                            .unwrap_or(0.0);

                        // Apply stress penalty to failure chance
                        let base_failure = 1.0 - survival_rate;
                        let actual_failure = (base_failure * (1.0 + stress_at_voxel)).min(1.0);

                        if rng.gen::<f32>() < actual_failure {
                            let old_support = support;
                            support_field.set(x, y, z, SupportType::None);
                            result.manifest.record_support_change(
                                chunk_key, x, y, z,
                                old_support, SupportType::None,
                            );
                            result.supports_degraded += 1;
                        }
                    }
                }
            }
        }
    }

    result.timings.support_degradation = t_step1.elapsed();

    // ── Step 2: Stress Amplification ──
    let t_step2 = Instant::now();
    for &chunk_key in chunks {
        let (cx, cy, cz) = chunk_key;
        for z in 0..chunk_size {
            for y in 0..chunk_size {
                for x in 0..chunk_size {
                    // Check if this is a solid voxel
                    let is_solid = density_fields
                        .get(&chunk_key)
                        .map(|df| df.get(x, y, z).material.is_solid())
                        .unwrap_or(false);

                    if !is_solid {
                        continue;
                    }

                    // Convert to world coordinates
                    let wx = cx * chunk_size as i32 + x as i32;
                    let wy = cy * chunk_size as i32 + y as i32;
                    let wz = cz * chunk_size as i32 + z as i32;

                    // Recalculate stress using the core function
                    let base_stress = calc_voxel_stress(
                        density_fields,
                        support_fields,
                        &local_stress_config,
                        wx, wy, wz,
                        chunk_size,
                    );

                    // Amplify by the stress multiplier
                    let amplified = base_stress * config.stress_multiplier;

                    // Store amplified stress
                    if let Some(sf) = stress_fields.get_mut(&chunk_key) {
                        sf.set(x, y, z, amplified);
                    }
                }
            }
        }
    }

    result.timings.stress_amplification = t_step2.elapsed();

    // ── Step 3: Collapse Cascade ──
    let t_step3 = Instant::now();
    let mut total_collapsed_voxels: u32 = 0;
    let mut all_dirty = std::collections::HashSet::new();

    for &chunk_key in chunks {
        let (cx, cy, cz) = chunk_key;
        // Chunk center world position
        let center_wx = cx * chunk_size as i32 + (chunk_size / 2) as i32;
        let center_wy = cy * chunk_size as i32 + (chunk_size / 2) as i32;
        let center_wz = cz * chunk_size as i32 + (chunk_size / 2) as i32;

        let (events, dirty_chunks) = post_change_stress_update_with_iterations(
            density_fields,
            stress_fields,
            support_fields,
            &local_stress_config,
            (center_wx, center_wy, center_wz),
            chunk_size,
            config.max_cascade_iterations,
        );

        for key in &dirty_chunks {
            all_dirty.insert(*key);
        }

        for event in &events {
            // Set glimpse_chunk to the first chunk where a collapse occurred
            if result.glimpse_chunk.is_none() {
                if let Some(first_chunk) = event.affected_chunks.first() {
                    result.glimpse_chunk = Some(*first_chunk);
                }
            }

            // Record collapsed voxels in manifest (solid -> Air with density -1.0)
            for cv in &event.collapsed_voxels {
                let (ck, lx, ly, lz) = world_to_chunk_local(
                    cv.world_x, cv.world_y, cv.world_z, chunk_size,
                );
                result.manifest.record_voxel_change(
                    ck, lx, ly, lz,
                    cv.material, 1.0,
                    Material::Air, -1.0,
                );
            }

            // Record rubble voxels in manifest (air -> solid with density 1.0)
            for rv in &event.rubble_voxels {
                let (rk, rlx, rly, rlz) = world_to_chunk_local(
                    rv.world_x, rv.world_y, rv.world_z, chunk_size,
                );
                result.manifest.record_voxel_change(
                    rk, rlx, rly, rlz,
                    Material::Air, -1.0,
                    rv.material, 1.0,
                );
            }

            total_collapsed_voxels += event.collapsed_voxels.len() as u32;
        }

        result.collapses_triggered += events.len() as u32;
        result.collapse_events.extend(events);
    }

    result.timings.collapse_cascade = t_step3.elapsed();

    result.dirty_chunks = all_dirty.into_iter().collect();

    // ── Build transform log ──
    result.transform_log.push(TransformEntry {
        description: "Supports degraded".to_string(),
        count: result.supports_degraded,
    });
    result.transform_log.push(TransformEntry {
        description: "Collapses triggered".to_string(),
        count: result.collapses_triggered,
    });
    result.transform_log.push(TransformEntry {
        description: "Voxels collapsed".to_string(),
        count: total_collapsed_voxels,
    });

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha8Rng;
    use rand::SeedableRng;

    const CHUNK_SIZE: usize = 16;
    const FIELD_SIZE: usize = 17;

    /// Create a fully solid density field.
    fn make_solid_density() -> DensityField {
        let mut df = DensityField::new(FIELD_SIZE);
        for sample in df.samples.iter_mut() {
            sample.density = 1.0;
            sample.material = Material::Granite;
        }
        df
    }

    /// Create a 3x3x3 grid of solid chunks centered on (0,0,0).
    fn make_solid_world() -> (
        HashMap<(i32, i32, i32), DensityField>,
        HashMap<(i32, i32, i32), StressField>,
        HashMap<(i32, i32, i32), SupportField>,
    ) {
        let mut density_fields = HashMap::new();
        let mut stress_fields = HashMap::new();
        let mut support_fields = HashMap::new();
        for cz in -1..=1 {
            for cy in -1..=1 {
                for cx in -1..=1 {
                    density_fields.insert((cx, cy, cz), make_solid_density());
                    stress_fields.insert((cx, cy, cz), StressField::new(FIELD_SIZE));
                    support_fields.insert((cx, cy, cz), SupportField::new(FIELD_SIZE));
                }
            }
        }
        (density_fields, stress_fields, support_fields)
    }

    /// Create a collapse config where slate struts always fail.
    fn config_slate_always_fails() -> CollapseConfig {
        let mut survival = CollapseConfig::default().strut_survival;
        survival[SupportType::SlateStrut as usize] = 0.0;
        CollapseConfig {
            strut_survival: survival,
            stress_multiplier: 1.5,
            max_cascade_iterations: 8,
            rubble_fill_ratio: 0.40,
            ..CollapseConfig::default()
        }
    }

    /// Create a collapse config where crystal struts always survive.
    fn config_crystal_always_survives() -> CollapseConfig {
        let mut survival = CollapseConfig::default().strut_survival;
        survival[SupportType::CrystalStrut as usize] = 1.0;
        CollapseConfig {
            strut_survival: survival,
            stress_multiplier: 1.5,
            max_cascade_iterations: 8,
            rubble_fill_ratio: 0.40,
            ..CollapseConfig::default()
        }
    }

    #[test]
    fn test_slate_strut_degrades() {
        let (mut density_fields, mut stress_fields, mut support_fields) = make_solid_world();

        // Place a SlateStrut at (0,0,0) chunk, local (5,5,5)
        support_fields.get_mut(&(0, 0, 0)).unwrap().set(5, 5, 5, SupportType::SlateStrut);

        let config = config_slate_always_fails();
        let stress_config = StressConfig::default();
        let chunks = vec![(0, 0, 0)];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = apply_collapse(
            &config, &stress_config,
            &mut density_fields, &mut stress_fields, &mut support_fields,
            &chunks, CHUNK_SIZE, &mut rng,
        );

        // With survival=0.0, the SlateStrut should definitely fail
        assert!(result.supports_degraded > 0, "SlateStrut with 0% survival should degrade");
        // Verify the support is now None
        assert_eq!(
            support_fields.get(&(0, 0, 0)).unwrap().get(5, 5, 5),
            SupportType::None,
            "Degraded SlateStrut should be None"
        );
    }

    #[test]
    fn test_crystal_strut_survives() {
        let (mut density_fields, mut stress_fields, mut support_fields) = make_solid_world();

        // Place CrystalStrut at (0,0,0) chunk, local (5,5,5)
        support_fields.get_mut(&(0, 0, 0)).unwrap().set(5, 5, 5, SupportType::CrystalStrut);

        let config = config_crystal_always_survives();
        let stress_config = StressConfig::default();
        let chunks = vec![(0, 0, 0)];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = apply_collapse(
            &config, &stress_config,
            &mut density_fields, &mut stress_fields, &mut support_fields,
            &chunks, CHUNK_SIZE, &mut rng,
        );

        // With survival=1.0, crystal strut should never fail
        assert_eq!(result.supports_degraded, 0, "CrystalStrut with 100% survival should not degrade");
        assert_eq!(
            support_fields.get(&(0, 0, 0)).unwrap().get(5, 5, 5),
            SupportType::CrystalStrut,
            "CrystalStrut should still be present"
        );
    }

    #[test]
    fn test_stress_amplification() {
        let (mut density_fields, mut stress_fields, mut support_fields) = make_solid_world();
        let stress_config = StressConfig::default();
        let multiplier = 1.5f32;
        let config = CollapseConfig {
            strut_survival: [1.0; 8],    // No supports degrade
            stress_multiplier: multiplier,
            max_cascade_iterations: 0,   // No cascade -- only test amplification
            rubble_fill_ratio: 0.40,
            ..CollapseConfig::default()
        };

        // Carve a void below chunk (0,0,0) to create ceiling stress
        // Set chunk (0,-1,0) to air
        if let Some(df) = density_fields.get_mut(&(0, -1, 0)) {
            for sample in df.samples.iter_mut() {
                sample.density = -1.0;
                sample.material = Material::Air;
            }
        }
        // Also set bottom half of (0,0,0) to air for an overhang
        if let Some(df) = density_fields.get_mut(&(0, 0, 0)) {
            for z in 0..FIELD_SIZE {
                for y in 0..8 {
                    for x in 0..FIELD_SIZE {
                        let sample = df.get_mut(x, y, z);
                        sample.density = -1.0;
                        sample.material = Material::Air;
                    }
                }
            }
        }

        // First, calculate baseline stress for a ceiling voxel using the core function
        let wx = 0 * CHUNK_SIZE as i32 + 8;
        let wy = 0 * CHUNK_SIZE as i32 + 8;
        let wz = 0 * CHUNK_SIZE as i32 + 8;
        let baseline_stress = calc_voxel_stress(
            &density_fields, &support_fields, &stress_config,
            wx, wy, wz, CHUNK_SIZE,
        );

        let chunks = vec![(0, 0, 0)];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        apply_collapse(
            &config, &stress_config,
            &mut density_fields, &mut stress_fields, &mut support_fields,
            &chunks, CHUNK_SIZE, &mut rng,
        );

        // Check stress at the ceiling voxel after amplification
        let stored_stress = stress_fields.get(&(0, 0, 0)).unwrap().get(8, 8, 8);

        // Only check amplification for voxels that actually have non-zero stress
        if baseline_stress > 0.0 {
            let expected = baseline_stress * multiplier;
            assert!(
                (stored_stress - expected).abs() < 0.01,
                "Stress should be amplified: baseline={}, expected={}, got={}",
                baseline_stress, expected, stored_stress
            );
        }
    }

    #[test]
    fn test_collapse_records_manifest() {
        let (mut density_fields, mut stress_fields, mut support_fields) = make_solid_world();

        // Place several SlateStruts in chunk (0,0,0)
        let sf = support_fields.get_mut(&(0, 0, 0)).unwrap();
        sf.set(3, 3, 3, SupportType::SlateStrut);
        sf.set(5, 5, 5, SupportType::SlateStrut);
        sf.set(7, 7, 7, SupportType::SlateStrut);

        let config = config_slate_always_fails();
        let stress_config = StressConfig::default();
        let chunks = vec![(0, 0, 0)];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = apply_collapse(
            &config, &stress_config,
            &mut density_fields, &mut stress_fields, &mut support_fields,
            &chunks, CHUNK_SIZE, &mut rng,
        );

        // With slate_strut_survival=0.0, all 3 should fail
        assert_eq!(result.supports_degraded, 3);

        // Verify manifest has support changes recorded
        let delta = result.manifest.chunk_deltas.get(&(0, 0, 0));
        assert!(delta.is_some(), "Manifest should have entries for chunk (0,0,0)");
        let delta = delta.unwrap();
        assert_eq!(
            delta.support_changes.len(), 3,
            "Manifest should record 3 support changes, got {}",
            delta.support_changes.len()
        );
        // Each should be SlateStrut -> None
        for change in &delta.support_changes {
            assert_eq!(change.old_support, SupportType::SlateStrut as u8);
            assert_eq!(change.new_support, SupportType::None as u8);
        }
    }

    #[test]
    fn test_deterministic() {
        // Run collapse twice with the same seed, verify identical results
        let config = config_slate_always_fails();
        let stress_config = StressConfig::default();
        let chunks = vec![(0, 0, 0)];

        let run = || {
            let (mut density_fields, mut stress_fields, mut support_fields) = make_solid_world();
            // Place some supports of different types
            let sf = support_fields.get_mut(&(0, 0, 0)).unwrap();
            sf.set(3, 3, 3, SupportType::SlateStrut);
            sf.set(5, 5, 5, SupportType::CopperStrut);
            sf.set(8, 8, 8, SupportType::CrystalStrut);

            let mut rng = ChaCha8Rng::seed_from_u64(42);
            let result = apply_collapse(
                &config, &stress_config,
                &mut density_fields, &mut stress_fields, &mut support_fields,
                &chunks, CHUNK_SIZE, &mut rng,
            );
            (result.supports_degraded, result.collapses_triggered, result.collapse_events.len())
        };

        let (deg1, col1, ev1) = run();
        let (deg2, col2, ev2) = run();

        assert_eq!(deg1, deg2, "Supports degraded should be deterministic");
        assert_eq!(col1, col2, "Collapses triggered should be deterministic");
        assert_eq!(ev1, ev2, "Collapse events count should be deterministic");
    }
}
