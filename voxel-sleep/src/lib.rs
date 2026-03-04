pub mod config;
pub mod manifest;
pub mod metamorphism;
pub mod minerals;
pub mod collapse;
pub mod priority;

use std::collections::{HashMap, HashSet};
use std::fmt::Write as FmtWrite;
use std::time::{Duration, Instant};
use crossbeam_channel::Sender;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use voxel_core::density::DensityField;
use voxel_core::stress::{StressField, SupportField, SupportType};

use crate::collapse::apply_collapse;
use crate::metamorphism::apply_metamorphism;
use crate::minerals::apply_mineral_growth;
use crate::priority::{classify_chunks, ChunkTier};

pub use collapse::CollapseTimings;
pub use config::SleepConfig;
pub use manifest::ChangeManifest;

/// Progress report during sleep processing.
#[derive(Debug, Clone)]
pub struct SleepProgress {
    /// 0=metamorphism, 1=minerals, 2=collapse, 3=done
    pub phase: u8,
    /// 0.0 - 1.0
    pub progress_pct: f32,
    pub chunks_processed: u32,
    pub chunks_total: u32,
    /// Chunk where something interesting happened (for montage display)
    pub glimpse_chunk: Option<(i32, i32, i32)>,
    /// 0=none, 1=metamorphism, 2=mineral_growth, 3=collapse
    pub glimpse_type: u8,
}

/// Per-phase timing data from the sleep pipeline.
#[derive(Debug, Clone)]
pub struct SleepTimings {
    pub total: Duration,
    pub chunk_filter: Duration,
    pub chunk_classify: Duration,
    pub metamorphism: Duration,
    pub minerals: Duration,
    pub collapse_total: Duration,
    pub collapse_sub: Option<CollapseTimings>,
    pub aggregation: Duration,
    pub loaded_chunks: u32,
    pub critical_chunks: u32,
    pub important_chunks: u32,
    pub cosmetic_chunks: u32,
}

/// Results of a deep sleep cycle.
#[derive(Debug, Clone)]
pub struct SleepResult {
    pub success: bool,
    pub chunks_changed: u32,
    pub voxels_metamorphosed: u32,
    pub minerals_grown: u32,
    pub supports_degraded: u32,
    pub collapses_triggered: u32,
    pub dirty_chunks: Vec<(i32, i32, i32)>,
    pub collapse_events: Vec<voxel_core::stress::CollapseEvent>,
    /// Detailed log of transformations for UI display
    pub transform_log: Vec<TransformEntry>,
    /// Change manifest recording all modifications for persistence
    pub manifest: ChangeManifest,
    /// Formatted profiling report text
    pub profile_report: String,
    /// Structured timing data
    pub timings: SleepTimings,
}

/// A single transformation entry for the log.
#[derive(Debug, Clone)]
pub struct TransformEntry {
    pub description: String,
    pub count: u32,
}

/// Execute a deep sleep cycle, advancing geological time.
///
/// Takes mutable refs to the world data (extracted by FFI or viewer layer).
/// Pure logic -- no FFI dependency.
///
/// Processing order by tier:
/// - Critical (player chunk + 6 face-adjacent): all transforms + full collapse cascade
/// - Important (2-ring neighbors + chunks with supports): metamorphism + minerals + lightweight collapse
/// - Cosmetic (remaining loaded chunks): metamorphism only
pub fn execute_sleep(
    config: &SleepConfig,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    stress_fields: &mut HashMap<(i32, i32, i32), StressField>,
    support_fields: &mut HashMap<(i32, i32, i32), SupportField>,
    player_chunk: (i32, i32, i32),
    sleep_count: u32,
    progress_tx: Option<&Sender<SleepProgress>>,
) -> SleepResult {
    let chunk_size: usize = 16;
    let t_total = Instant::now();

    // Deterministic RNG seeded from sleep_count (each sleep cycle is reproducible)
    let mut rng = ChaCha8Rng::seed_from_u64(sleep_count as u64 * 7919 + 42);

    // Identify which chunks have supports for tier classification
    let chunks_with_supports: HashSet<(i32, i32, i32)> = support_fields
        .iter()
        .filter(|(_, sf)| {
            sf.supports.iter().any(|s| *s != SupportType::None)
        })
        .map(|(k, _)| *k)
        .collect();

    // Filter chunks by Chebyshev distance from player chunk
    let t_filter = Instant::now();
    let radius = config.chunk_radius as i32;
    let mut loaded_chunks: Vec<(i32, i32, i32)> = density_fields.keys()
        .copied()
        .filter(|&(cx, cy, cz)| {
            let dx = (cx - player_chunk.0).abs();
            let dy = (cy - player_chunk.1).abs();
            let dz = (cz - player_chunk.2).abs();
            dx.max(dy).max(dz) <= radius
        })
        .collect();
    loaded_chunks.sort();
    let t_filter_elapsed = t_filter.elapsed();

    // Classify filtered chunks into priority tiers
    let t_classify = Instant::now();
    let classified = classify_chunks(player_chunk, &loaded_chunks, &chunks_with_supports);
    let t_classify_elapsed = t_classify.elapsed();

    let critical_chunks: Vec<(i32, i32, i32)> = classified.iter()
        .filter(|(_, t)| *t == ChunkTier::Critical)
        .map(|(c, _)| *c)
        .collect();
    let important_chunks: Vec<(i32, i32, i32)> = classified.iter()
        .filter(|(_, t)| *t == ChunkTier::Important)
        .map(|(c, _)| *c)
        .collect();
    let cosmetic_chunks: Vec<(i32, i32, i32)> = classified.iter()
        .filter(|(_, t)| *t == ChunkTier::Cosmetic)
        .map(|(c, _)| *c)
        .collect();

    // All chunks that get metamorphism (all tiers)
    let all_chunks: Vec<(i32, i32, i32)> = classified.iter().map(|(c, _)| *c).collect();
    // Chunks that get minerals (critical + important)
    let mineral_chunks: Vec<(i32, i32, i32)> = critical_chunks.iter()
        .chain(important_chunks.iter())
        .copied()
        .collect();
    // Chunks that get collapse (critical + important)
    let collapse_chunks: Vec<(i32, i32, i32)> = mineral_chunks.clone();

    let total_chunks = all_chunks.len() as u32;
    let mut result_manifest = ChangeManifest::new();
    let mut all_dirty: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut transform_log: Vec<TransformEntry> = Vec::new();
    let total_metamorphosed;
    let total_minerals;
    let total_supports_degraded;
    let total_collapses;
    let all_collapse_events;

    // --- Phase 0: Metamorphism (all chunks) ---
    let t_meta = Instant::now();
    send_progress(progress_tx, 0, 0.0, 0, total_chunks, None, 0);

    if config.metamorphism_enabled {
        let meta_result = apply_metamorphism(
            &config.metamorphism,
            density_fields,
            &all_chunks,
            chunk_size,
            &mut rng,
        );
        total_metamorphosed = meta_result.voxels_transformed;
        result_manifest.merge_sleep_changes(&meta_result.manifest);
        transform_log.extend(meta_result.transform_log);
        if meta_result.voxels_transformed > 0 {
            // Mark all metamorphism-affected chunks as dirty
            for key in meta_result.manifest.chunk_deltas.keys() {
                all_dirty.insert(*key);
            }
        }

        send_progress(
            progress_tx, 0, 1.0, total_chunks, total_chunks,
            meta_result.glimpse_chunk, if meta_result.glimpse_chunk.is_some() { 1 } else { 0 },
        );
    } else {
        total_metamorphosed = 0;
        send_progress(progress_tx, 0, 1.0, total_chunks, total_chunks, None, 0);
    }

    let t_meta_elapsed = t_meta.elapsed();

    // --- Phase 1: Mineral growth (critical + important chunks) ---
    let t_minerals = Instant::now();
    send_progress(progress_tx, 1, 0.0, 0, mineral_chunks.len() as u32, None, 0);

    if config.minerals_enabled {
        let mineral_result = apply_mineral_growth(
            &config.minerals,
            density_fields,
            &mineral_chunks,
            chunk_size,
            &mut rng,
        );
        total_minerals = mineral_result.minerals_grown;
        result_manifest.merge_sleep_changes(&mineral_result.manifest);
        transform_log.extend(mineral_result.transform_log);
        if mineral_result.minerals_grown > 0 {
            for key in mineral_result.manifest.chunk_deltas.keys() {
                all_dirty.insert(*key);
            }
        }

        send_progress(
            progress_tx, 1, 1.0, mineral_chunks.len() as u32, mineral_chunks.len() as u32,
            mineral_result.glimpse_chunk, if mineral_result.glimpse_chunk.is_some() { 2 } else { 0 },
        );
    } else {
        total_minerals = 0;
        send_progress(progress_tx, 1, 1.0, mineral_chunks.len() as u32, mineral_chunks.len() as u32, None, 0);
    }

    let t_minerals_elapsed = t_minerals.elapsed();

    // --- Phase 2: Structural collapse (critical + important chunks) ---
    let t_collapse = Instant::now();
    send_progress(progress_tx, 2, 0.0, 0, collapse_chunks.len() as u32, None, 0);

    let collapse_sub_timings;
    if config.collapse_enabled {
        let stress_config = &config.stress;
        let collapse_result = apply_collapse(
            &config.collapse,
            stress_config,
            density_fields,
            stress_fields,
            support_fields,
            &collapse_chunks,
            chunk_size,
            &mut rng,
        );
        total_supports_degraded = collapse_result.supports_degraded;
        total_collapses = collapse_result.collapses_triggered;
        collapse_sub_timings = Some(collapse_result.timings.clone());
        all_collapse_events = collapse_result.collapse_events;
        result_manifest.merge_sleep_changes(&collapse_result.manifest);
        transform_log.extend(collapse_result.transform_log);
        for key in &collapse_result.dirty_chunks {
            all_dirty.insert(*key);
        }
        for key in collapse_result.manifest.chunk_deltas.keys() {
            all_dirty.insert(*key);
        }

        send_progress(
            progress_tx, 2, 1.0, collapse_chunks.len() as u32, collapse_chunks.len() as u32,
            collapse_result.glimpse_chunk, if collapse_result.glimpse_chunk.is_some() { 3 } else { 0 },
        );
    } else {
        total_supports_degraded = 0;
        total_collapses = 0;
        collapse_sub_timings = None;
        all_collapse_events = Vec::new();
        send_progress(progress_tx, 2, 1.0, collapse_chunks.len() as u32, collapse_chunks.len() as u32, None, 0);
    }
    let t_collapse_elapsed = t_collapse.elapsed();

    // --- Aggregation ---
    let t_agg = Instant::now();

    // --- Done ---
    send_progress(progress_tx, 3, 1.0, total_chunks, total_chunks, None, 0);

    // Filter out zero-count log entries
    transform_log.retain(|e| e.count > 0);

    result_manifest.sleep_count = sleep_count;

    let dirty_chunks: Vec<(i32, i32, i32)> = all_dirty.into_iter().collect();
    let t_agg_elapsed = t_agg.elapsed();
    let t_total_elapsed = t_total.elapsed();

    let timings = SleepTimings {
        total: t_total_elapsed,
        chunk_filter: t_filter_elapsed,
        chunk_classify: t_classify_elapsed,
        metamorphism: t_meta_elapsed,
        minerals: t_minerals_elapsed,
        collapse_total: t_collapse_elapsed,
        collapse_sub: collapse_sub_timings,
        aggregation: t_agg_elapsed,
        loaded_chunks: loaded_chunks.len() as u32,
        critical_chunks: critical_chunks.len() as u32,
        important_chunks: important_chunks.len() as u32,
        cosmetic_chunks: cosmetic_chunks.len() as u32,
    };

    let profile_report = build_sleep_profile_report(
        &timings,
        sleep_count,
        config.chunk_radius,
        total_metamorphosed,
        total_minerals,
        total_supports_degraded,
        total_collapses,
        all_chunks.len() as u32,
        mineral_chunks.len() as u32,
        collapse_chunks.len() as u32,
        dirty_chunks.len() as u32,
    );

    SleepResult {
        success: true,
        chunks_changed: dirty_chunks.len() as u32,
        voxels_metamorphosed: total_metamorphosed,
        minerals_grown: total_minerals,
        supports_degraded: total_supports_degraded,
        collapses_triggered: total_collapses,
        dirty_chunks,
        collapse_events: all_collapse_events,
        transform_log,
        manifest: result_manifest,
        profile_report,
        timings,
    }
}

fn dur_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn build_sleep_profile_report(
    timings: &SleepTimings,
    sleep_count: u32,
    chunk_radius: u32,
    voxels_metamorphosed: u32,
    minerals_grown: u32,
    supports_degraded: u32,
    collapses_triggered: u32,
    meta_chunk_count: u32,
    mineral_chunk_count: u32,
    collapse_chunk_count: u32,
    dirty_chunk_count: u32,
) -> String {
    let mut s = String::with_capacity(2048);

    let _ = writeln!(s);
    let _ = writeln!(s, "═══════════════════════════════════════════════════════");
    let _ = writeln!(s, "  DEEP SLEEP PROFILING REPORT");
    let _ = writeln!(s, "  Sleep Count: {} | Chunk Radius: {}", sleep_count, chunk_radius);
    let _ = writeln!(s, "═══════════════════════════════════════════════════════");

    let _ = writeln!(s);
    let _ = writeln!(s, "─── Chunk Setup ────────────────────────────────────");
    let _ = writeln!(s, "  Loaded chunks:       {}", timings.loaded_chunks);
    let _ = writeln!(s, "  Critical:            {}", timings.critical_chunks);
    let _ = writeln!(s, "  Important:           {}", timings.important_chunks);
    let _ = writeln!(s, "  Cosmetic:            {}", timings.cosmetic_chunks);
    let _ = writeln!(s, "  Filter + sort:       {:.2} ms", dur_ms(timings.chunk_filter));
    let _ = writeln!(s, "  Classification:      {:.2} ms", dur_ms(timings.chunk_classify));

    let _ = writeln!(s);
    let _ = writeln!(s, "─── Phase 0: Metamorphism ──────────────────────────");
    let _ = writeln!(s, "  Chunks processed:    {}", meta_chunk_count);
    let _ = writeln!(s, "  Voxels transformed:  {}", voxels_metamorphosed);
    let _ = writeln!(s, "  Duration:            {:.2} ms", dur_ms(timings.metamorphism));

    let _ = writeln!(s);
    let _ = writeln!(s, "─── Phase 1: Mineral Growth ────────────────────────");
    let _ = writeln!(s, "  Chunks processed:    {}", mineral_chunk_count);
    let _ = writeln!(s, "  Minerals grown:      {}", minerals_grown);
    let _ = writeln!(s, "  Duration:            {:.2} ms", dur_ms(timings.minerals));

    let _ = writeln!(s);
    let _ = writeln!(s, "─── Phase 2: Structural Collapse ───────────────────");
    let _ = writeln!(s, "  Chunks processed:    {}", collapse_chunk_count);
    if let Some(ref ct) = timings.collapse_sub {
        let _ = writeln!(s, "  Support degradation: {:.2} ms  ({} degraded)", dur_ms(ct.support_degradation), supports_degraded);
        let _ = writeln!(s, "  Stress amplification: {:.2} ms", dur_ms(ct.stress_amplification));
        let _ = writeln!(s, "  Collapse cascade:    {:.2} ms  ({} collapses)", dur_ms(ct.collapse_cascade), collapses_triggered);
    }
    let _ = writeln!(s, "  Phase total:         {:.2} ms", dur_ms(timings.collapse_total));

    let _ = writeln!(s);
    let _ = writeln!(s, "─── Aggregation ────────────────────────────────────");
    let _ = writeln!(s, "  Duration:            {:.2} ms", dur_ms(timings.aggregation));
    let _ = writeln!(s, "  Dirty chunks:        {}", dirty_chunk_count);

    let _ = writeln!(s);
    let _ = writeln!(s, "═══ execute_sleep() total: {:.2} ms ════════════════", dur_ms(timings.total));

    s
}

/// Send a progress update if a channel is provided.
fn send_progress(
    tx: Option<&Sender<SleepProgress>>,
    phase: u8,
    progress_pct: f32,
    chunks_processed: u32,
    chunks_total: u32,
    glimpse_chunk: Option<(i32, i32, i32)>,
    glimpse_type: u8,
) {
    if let Some(tx) = tx {
        let _ = tx.try_send(SleepProgress {
            phase,
            progress_pct,
            chunks_processed,
            chunks_total,
            glimpse_chunk,
            glimpse_type,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::stress::{StressField, SupportField, SupportType};

    fn make_test_world(chunk_size: usize) -> (
        HashMap<(i32, i32, i32), DensityField>,
        HashMap<(i32, i32, i32), StressField>,
        HashMap<(i32, i32, i32), SupportField>,
    ) {
        let grid_size = chunk_size + 1;
        let coords = vec![(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)];
        let config = voxel_gen::config::GenerationConfig {
            seed: 42,
            ..Default::default()
        };
        let (density_fields, _pools, _seeds, _worms, _timings) = voxel_gen::region_gen::generate_region_densities(&coords, &config);
        let mut stress_fields = HashMap::new();
        let mut support_fields = HashMap::new();
        for &key in density_fields.keys() {
            stress_fields.insert(key, StressField::new(grid_size));
            support_fields.insert(key, SupportField::new(grid_size));
        }
        (density_fields, stress_fields, support_fields)
    }

    #[test]
    fn test_execute_sleep_basic() {
        let (mut density_fields, mut stress_fields, mut support_fields) = make_test_world(16);
        let config = SleepConfig::default();
        let result = execute_sleep(
            &config,
            &mut density_fields,
            &mut stress_fields,
            &mut support_fields,
            (0, 0, 0),
            1,
            None,
        );
        assert!(result.success);
        // Should have some metamorphism on a real generated world
        // (exact counts depend on noise, but should run without panic)
    }

    #[test]
    fn test_execute_sleep_deterministic() {
        let (mut df1, mut sf1, mut sup1) = make_test_world(16);
        let (mut df2, mut sf2, mut sup2) = make_test_world(16);
        let config = SleepConfig::default();

        let r1 = execute_sleep(&config, &mut df1, &mut sf1, &mut sup1, (0, 0, 0), 1, None);
        let r2 = execute_sleep(&config, &mut df2, &mut sf2, &mut sup2, (0, 0, 0), 1, None);

        assert_eq!(r1.voxels_metamorphosed, r2.voxels_metamorphosed);
        assert_eq!(r1.minerals_grown, r2.minerals_grown);
        assert_eq!(r1.supports_degraded, r2.supports_degraded);
        assert_eq!(r1.collapses_triggered, r2.collapses_triggered);
    }

    #[test]
    fn test_multiple_sleep_cycles() {
        let (mut density_fields, mut stress_fields, mut support_fields) = make_test_world(16);
        let config = SleepConfig::default();

        for cycle in 1..=3 {
            let result = execute_sleep(
                &config,
                &mut density_fields,
                &mut stress_fields,
                &mut support_fields,
                (0, 0, 0),
                cycle,
                None,
            );
            assert!(result.success);
        }
    }

    #[test]
    fn test_sleep_with_supports() {
        let (mut density_fields, mut stress_fields, mut support_fields) = make_test_world(16);

        // Place some strut supports
        if let Some(sf) = support_fields.get_mut(&(0, 0, 0)) {
            sf.set(5, 5, 5, SupportType::SlateStrut);
            sf.set(6, 5, 5, SupportType::SlateStrut);
            sf.set(7, 5, 5, SupportType::CopperStrut);
        }

        let config = SleepConfig::default();
        let result = execute_sleep(
            &config,
            &mut density_fields,
            &mut stress_fields,
            &mut support_fields,
            (0, 0, 0),
            1,
            None,
        );
        assert!(result.success);
        // Should have processed the supports (exact degradation depends on RNG + stress)
    }

    #[test]
    fn test_sleep_manifest_records_changes() {
        let (mut density_fields, mut stress_fields, mut support_fields) = make_test_world(16);
        let config = SleepConfig::default();

        let result = execute_sleep(
            &config,
            &mut density_fields,
            &mut stress_fields,
            &mut support_fields,
            (0, 0, 0),
            1,
            None,
        );

        // If there were metamorphism changes, the manifest should have entries
        if result.voxels_metamorphosed > 0 {
            assert!(!result.manifest.chunk_deltas.is_empty());
        }
    }
}
