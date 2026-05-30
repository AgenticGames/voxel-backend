//! High-level post-change stress update orchestration (v1 + v2 cascade).
//!
//! Behavior-preserving split of the former `stress.rs` god file.

use std::collections::{HashMap, HashSet};

use crate::density::DensityField;

use super::config::StressConfig;
use super::types::{CollapseEvent, CollapseEventV2, StressField, SupportField};
use super::calc_v2::{
    recalc_stress_region, recalc_stress_region_v2,
};
use super::collapse::{
    detect_and_execute_collapses, detect_and_execute_collapses_v2,
};

/// V2 post-change stress update: runs ground connectivity + collapse detection with cascade.
/// `support_fields` is `&mut` so strut HP can decay and broken struts can be
/// cleared during the cascade. Callers that don't want HP decay should use
/// the lower-level `recalc_stress_region_v2_filtered` + manual collapse.
pub fn post_change_stress_update_v2(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    stress_fields: &mut HashMap<(i32, i32, i32), StressField>,
    support_fields: &mut HashMap<(i32, i32, i32), SupportField>,
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
