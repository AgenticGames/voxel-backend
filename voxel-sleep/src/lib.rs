pub mod config;
pub mod manifest;
pub mod util;
pub mod metamorphism;
pub mod minerals;
pub mod collapse;
pub mod priority;
pub mod reaction;
pub mod aureole;
pub mod veins;
pub mod deeptime;
pub mod groundwater;

use std::collections::{HashMap, HashSet};
use std::fmt::Write as FmtWrite;
use std::time::{Duration, Instant};
use crossbeam_channel::Sender;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use voxel_core::density::DensityField;
use voxel_core::stress::{StressField, SupportField, SupportType};
use voxel_fluid::FluidSnapshot;

use crate::aureole::build_heat_map;
use crate::priority::{classify_chunks, ChunkTier};
use crate::reaction::apply_reaction;
use crate::aureole::apply_aureole;
use crate::veins::apply_veins;
use crate::deeptime::apply_deeptime;

pub use collapse::CollapseTimings;
pub use config::SleepConfig;
pub use manifest::ChangeManifest;

/// Progress report during sleep processing.
#[derive(Debug, Clone)]
pub struct SleepProgress {
    /// 0=reaction, 1=aureole, 2=veins, 3=deeptime, 4=done
    pub phase: u8,
    /// Phase display name
    pub phase_name: &'static str,
    /// Geological years for this phase
    pub phase_years: u32,
    /// 0.0 - 1.0
    pub progress_pct: f32,
    pub chunks_processed: u32,
    pub chunks_total: u32,
    /// Chunk where something interesting happened (for montage display)
    pub glimpse_chunk: Option<(i32, i32, i32)>,
    /// 0=none, 1=acid_dissolution, 2=metamorphism, 3=vein_deposit, 4=enrichment, 5=collapse
    pub glimpse_type: u8,
    /// Summary of what happened in this phase
    pub phase_summary: String,
}

/// Per-phase timing data from the sleep pipeline.
#[derive(Debug, Clone)]
pub struct SleepTimings {
    pub total: Duration,
    pub chunk_filter: Duration,
    pub chunk_classify: Duration,
    pub fluid_snapshot: Duration,
    pub heat_map: Duration,
    pub reaction: Duration,
    pub reaction_acid: Duration,
    pub reaction_oxidation: Duration,
    pub aureole: Duration,
    pub aureole_metamorphism: Duration,
    pub aureole_erosion: Duration,
    pub veins: Duration,
    pub veins_hydrothermal: Duration,
    pub veins_formations: Duration,
    pub deeptime: Duration,
    pub deeptime_enrichment: Duration,
    pub deeptime_thickening: Duration,
    pub deeptime_formations: Duration,
    pub deeptime_collapse: Duration,
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
    // Phase-specific counts
    pub acid_dissolved: u32,
    pub voxels_oxidized: u32,
    pub voxels_metamorphosed: u32,
    pub veins_deposited: u32,
    pub formations_grown: u32,
    pub voxels_enriched: u32,
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
    // Legacy fields for FFI backward compat
    pub minerals_grown: u32,
}

/// A single transformation entry for the log.
#[derive(Debug, Clone)]
pub struct TransformEntry {
    pub description: String,
    pub count: u32,
}

/// Execute a deep sleep cycle — 4-phase geological time simulation (1.25 million years).
///
/// Phase 1: "The Reaction" (10,000 years) — acid dissolution, oxidation
/// Phase 2: "The Aureole" (100,000 years) — contact metamorphism, water erosion
/// Phase 3: "The Veins" (500,000 years) — hydrothermal ore deposition, formation growth
/// Phase 4: "The Deep Time" (1,250,000 years) — enrichment, thickening, formations, collapse
pub fn execute_sleep(
    config: &SleepConfig,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    stress_fields: &mut HashMap<(i32, i32, i32), StressField>,
    support_fields: &mut HashMap<(i32, i32, i32), SupportField>,
    fluid_snapshot: &FluidSnapshot,
    player_chunk: (i32, i32, i32),
    sleep_count: u32,
    progress_tx: Option<&Sender<SleepProgress>>,
) -> SleepResult {
    let chunk_size: usize = 16;
    let t_total = Instant::now();

    // Deterministic RNG seeded from sleep_count
    let mut rng = ChaCha8Rng::seed_from_u64(sleep_count as u64 * 7919 + 42);

    // Identify chunks with supports for tier classification
    let chunks_with_supports: HashSet<(i32, i32, i32)> = support_fields
        .iter()
        .filter(|(_, sf)| sf.supports.iter().any(|s| *s != SupportType::None))
        .map(|(k, _)| *k)
        .collect();

    // Filter chunks by Chebyshev distance
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

    // Classify into priority tiers
    let t_classify = Instant::now();
    let classified = classify_chunks(player_chunk, &loaded_chunks, &chunks_with_supports);
    let t_classify_elapsed = t_classify.elapsed();

    let critical_chunks: Vec<(i32, i32, i32)> = classified.iter()
        .filter(|(_, t)| *t == ChunkTier::Critical).map(|(c, _)| *c).collect();
    let important_chunks: Vec<(i32, i32, i32)> = classified.iter()
        .filter(|(_, t)| *t == ChunkTier::Important).map(|(c, _)| *c).collect();
    let cosmetic_chunks: Vec<(i32, i32, i32)> = classified.iter()
        .filter(|(_, t)| *t == ChunkTier::Cosmetic).map(|(c, _)| *c).collect();

    // All chunks for phases 1-2 (acid/aureole spread widely)
    let all_chunks: Vec<(i32, i32, i32)> = classified.iter().map(|(c, _)| *c).collect();
    // Critical + Important for phases 3-4
    let mineral_chunks: Vec<(i32, i32, i32)> = critical_chunks.iter()
        .chain(important_chunks.iter()).copied().collect();
    let collapse_chunks: Vec<(i32, i32, i32)> = mineral_chunks.clone();

    let total_chunks = all_chunks.len() as u32;
    let mut result_manifest = ChangeManifest::new();
    let mut all_dirty: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut transform_log: Vec<TransformEntry> = Vec::new();

    // --- Heat map computation ---
    let t_heat = Instant::now();
    let heat_map = build_heat_map(density_fields, fluid_snapshot, &all_chunks, chunk_size);
    let t_heat_elapsed = t_heat.elapsed();

    // Accumulators
    let mut total_acid_dissolved = 0u32;
    let mut total_oxidized = 0u32;
    let mut total_metamorphosed = 0u32;
    let mut total_veins = 0u32;
    let mut total_formations = 0u32;
    let mut total_enriched = 0u32;
    let mut total_supports_degraded = 0u32;
    let mut total_collapses = 0u32;
    let mut all_collapse_events: Vec<voxel_core::stress::CollapseEvent> = Vec::new();
    let mut collapse_sub_timings: Option<CollapseTimings> = None;

    // ═══ Phase 1: The Reaction (10,000 years) ═══
    let t_p1 = Instant::now();
    send_progress(progress_tx, 0, "The Reaction", 10_000, 0.0, 0, total_chunks, None, 0, String::new());

    if config.phase1_enabled {
        let reaction_result = apply_reaction(
            &config.reaction, density_fields, fluid_snapshot,
            &all_chunks, chunk_size, &mut rng,
        );
        total_acid_dissolved = reaction_result.acid_dissolved;
        total_oxidized = reaction_result.voxels_oxidized;
        result_manifest.merge_sleep_changes(&reaction_result.manifest);
        transform_log.extend(reaction_result.transform_log);
        for key in reaction_result.manifest.chunk_deltas.keys() {
            all_dirty.insert(*key);
        }
        send_progress(progress_tx, 0, "The Reaction", 10_000, 1.0, total_chunks, total_chunks,
            reaction_result.glimpse_chunk, if reaction_result.glimpse_chunk.is_some() { 1 } else { 0 },
            format!("Acid dissolved {} voxels, {} oxidized", total_acid_dissolved, total_oxidized));
    } else {
        send_progress(progress_tx, 0, "The Reaction", 10_000, 1.0, total_chunks, total_chunks, None, 0, String::new());
    }
    let t_p1_elapsed = t_p1.elapsed();

    // ═══ Phase 2: The Aureole (100,000 years) ═══
    let t_p2 = Instant::now();
    send_progress(progress_tx, 1, "The Aureole", 100_000, 0.0, 0, total_chunks, None, 0, String::new());

    if config.phase2_enabled {
        let aureole_result = apply_aureole(
            &config.aureole, &config.groundwater, density_fields, fluid_snapshot,
            &heat_map, &all_chunks, chunk_size, &mut rng,
        );
        total_metamorphosed = aureole_result.voxels_metamorphosed;
        result_manifest.merge_sleep_changes(&aureole_result.manifest);
        transform_log.extend(aureole_result.transform_log);
        for key in aureole_result.manifest.chunk_deltas.keys() {
            all_dirty.insert(*key);
        }
        send_progress(progress_tx, 1, "The Aureole", 100_000, 1.0, total_chunks, total_chunks,
            aureole_result.glimpse_chunk, if aureole_result.glimpse_chunk.is_some() { 2 } else { 0 },
            format!("{} voxels metamorphosed, {} eroded", total_metamorphosed, aureole_result.channels_eroded));
    } else {
        send_progress(progress_tx, 1, "The Aureole", 100_000, 1.0, total_chunks, total_chunks, None, 0, String::new());
    }
    let t_p2_elapsed = t_p2.elapsed();

    // ═══ Phase 3: The Veins (500,000 years) ═══
    let t_p3 = Instant::now();
    send_progress(progress_tx, 2, "The Veins", 500_000, 0.0, 0, mineral_chunks.len() as u32, None, 0, String::new());

    if config.phase3_enabled {
        let vein_result = apply_veins(
            &config.veins, &config.groundwater, density_fields, fluid_snapshot,
            &heat_map, &mineral_chunks, chunk_size, &mut rng,
        );
        total_veins = vein_result.veins_deposited;
        total_formations += vein_result.formations_grown;
        result_manifest.merge_sleep_changes(&vein_result.manifest);
        transform_log.extend(vein_result.transform_log);
        for key in vein_result.manifest.chunk_deltas.keys() {
            all_dirty.insert(*key);
        }
        send_progress(progress_tx, 2, "The Veins", 500_000, 1.0,
            mineral_chunks.len() as u32, mineral_chunks.len() as u32,
            vein_result.glimpse_chunk, if vein_result.glimpse_chunk.is_some() { 3 } else { 0 },
            format!("{} ore voxels deposited, {} formations", total_veins, vein_result.formations_grown));
    } else {
        send_progress(progress_tx, 2, "The Veins", 500_000, 1.0,
            mineral_chunks.len() as u32, mineral_chunks.len() as u32, None, 0, String::new());
    }
    let t_p3_elapsed = t_p3.elapsed();

    // ═══ Phase 4: The Deep Time (1,250,000 years) ═══
    let t_p4 = Instant::now();
    send_progress(progress_tx, 3, "The Deep Time", 1_250_000, 0.0, 0, collapse_chunks.len() as u32, None, 0, String::new());

    if config.phase4_enabled {
        let dt_result = apply_deeptime(
            &config.deeptime, &config.groundwater, density_fields, stress_fields, support_fields,
            fluid_snapshot, &heat_map, &collapse_chunks, chunk_size,
            &config.stress, &mut rng,
        );
        total_enriched = dt_result.voxels_enriched;
        total_formations += dt_result.formations_grown;
        total_supports_degraded = dt_result.supports_degraded;
        total_collapses = dt_result.collapses_triggered;
        result_manifest.merge_sleep_changes(&dt_result.manifest);
        transform_log.extend(dt_result.transform_log);
        for key in dt_result.manifest.chunk_deltas.keys() {
            all_dirty.insert(*key);
        }
        if let Some(ref cr) = dt_result.collapse_result {
            all_collapse_events = cr.collapse_events.clone();
            collapse_sub_timings = Some(cr.timings.clone());
            for key in &cr.dirty_chunks {
                all_dirty.insert(*key);
            }
        }
        send_progress(progress_tx, 3, "The Deep Time", 1_250_000, 1.0,
            collapse_chunks.len() as u32, collapse_chunks.len() as u32,
            dt_result.glimpse_chunk,
            if dt_result.collapses_triggered > 0 { 5 } else if dt_result.glimpse_chunk.is_some() { 4 } else { 0 },
            format!("{} enriched, {} collapses", total_enriched, total_collapses));
    } else {
        send_progress(progress_tx, 3, "The Deep Time", 1_250_000, 1.0,
            collapse_chunks.len() as u32, collapse_chunks.len() as u32, None, 0, String::new());
    }
    let t_p4_elapsed = t_p4.elapsed();

    // --- Done ---
    let t_agg = Instant::now();
    send_progress(progress_tx, 4, "Complete", 1_250_000, 1.0, total_chunks, total_chunks, None, 0, String::new());

    transform_log.retain(|e| e.count > 0);
    result_manifest.sleep_count = sleep_count;
    let dirty_chunks: Vec<(i32, i32, i32)> = all_dirty.into_iter().collect();
    let t_agg_elapsed = t_agg.elapsed();
    let t_total_elapsed = t_total.elapsed();

    let timings = SleepTimings {
        total: t_total_elapsed,
        chunk_filter: t_filter_elapsed,
        chunk_classify: t_classify_elapsed,
        fluid_snapshot: Duration::ZERO, // Snapshot acquired externally
        heat_map: t_heat_elapsed,
        reaction: t_p1_elapsed,
        reaction_acid: t_p1_elapsed,     // Sub-timing not split yet
        reaction_oxidation: Duration::ZERO,
        aureole: t_p2_elapsed,
        aureole_metamorphism: t_p2_elapsed,
        aureole_erosion: Duration::ZERO,
        veins: t_p3_elapsed,
        veins_hydrothermal: t_p3_elapsed,
        veins_formations: Duration::ZERO,
        deeptime: t_p4_elapsed,
        deeptime_enrichment: Duration::ZERO,
        deeptime_thickening: Duration::ZERO,
        deeptime_formations: Duration::ZERO,
        deeptime_collapse: Duration::ZERO,
        collapse_sub: collapse_sub_timings,
        aggregation: t_agg_elapsed,
        loaded_chunks: loaded_chunks.len() as u32,
        critical_chunks: critical_chunks.len() as u32,
        important_chunks: important_chunks.len() as u32,
        cosmetic_chunks: cosmetic_chunks.len() as u32,
    };

    let profile_report = build_sleep_profile_report(
        &timings, sleep_count, config.chunk_radius,
        total_acid_dissolved, total_oxidized, total_metamorphosed,
        total_veins, total_formations, total_enriched,
        total_supports_degraded, total_collapses,
        all_chunks.len() as u32, mineral_chunks.len() as u32,
        collapse_chunks.len() as u32, dirty_chunks.len() as u32,
        heat_map.len() as u32,
    );

    SleepResult {
        success: true,
        chunks_changed: dirty_chunks.len() as u32,
        acid_dissolved: total_acid_dissolved,
        voxels_oxidized: total_oxidized,
        voxels_metamorphosed: total_metamorphosed,
        veins_deposited: total_veins,
        formations_grown: total_formations,
        voxels_enriched: total_enriched,
        supports_degraded: total_supports_degraded,
        collapses_triggered: total_collapses,
        dirty_chunks,
        collapse_events: all_collapse_events,
        transform_log,
        manifest: result_manifest,
        profile_report,
        timings,
        // Legacy
        minerals_grown: total_formations,
    }
}

fn dur_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn build_sleep_profile_report(
    timings: &SleepTimings,
    sleep_count: u32,
    chunk_radius: u32,
    acid_dissolved: u32,
    voxels_oxidized: u32,
    voxels_metamorphosed: u32,
    veins_deposited: u32,
    formations_grown: u32,
    voxels_enriched: u32,
    supports_degraded: u32,
    collapses_triggered: u32,
    _all_chunk_count: u32,
    _mineral_chunk_count: u32,
    _collapse_chunk_count: u32,
    dirty_chunk_count: u32,
    heat_source_count: u32,
) -> String {
    let mut s = String::with_capacity(4096);

    let _ = writeln!(s);
    let _ = writeln!(s, "\u{2550}\u{2550}\u{2550} Deep Sleep Profile \u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}");
    let _ = writeln!(s, "  Duration: 1,250,000 years (sleep cycle #{})", sleep_count);
    let _ = writeln!(s, "  Loaded chunks: {} ({} critical, {} important, {} cosmetic)",
        timings.loaded_chunks, timings.critical_chunks, timings.important_chunks, timings.cosmetic_chunks);
    let _ = writeln!(s, "  Heat sources: {} | Chunk radius: {}", heat_source_count, chunk_radius);

    let _ = writeln!(s);
    let _ = writeln!(s, "\u{2500}\u{2500}\u{2500} Phase 1: The Reaction (10,000 years) \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}");
    let _ = writeln!(s, "  Acid dissolution:    {:.2} ms  ({} voxels dissolved)", dur_ms(timings.reaction_acid), acid_dissolved);
    let _ = writeln!(s, "  Surface oxidation:   {:.2} ms  ({} voxels oxidized)", dur_ms(timings.reaction_oxidation), voxels_oxidized);
    let _ = writeln!(s, "  Phase 1 total:       {:.2} ms", dur_ms(timings.reaction));

    let _ = writeln!(s);
    let _ = writeln!(s, "\u{2500}\u{2500}\u{2500} Phase 2: The Aureole (100,000 years) \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}");
    let _ = writeln!(s, "  Contact metamorphism: {:.2} ms  ({} voxels transformed)", dur_ms(timings.aureole_metamorphism), voxels_metamorphosed);
    let _ = writeln!(s, "  Water erosion:        {:.2} ms", dur_ms(timings.aureole_erosion));
    let _ = writeln!(s, "  Phase 2 total:       {:.2} ms", dur_ms(timings.aureole));

    let _ = writeln!(s);
    let _ = writeln!(s, "\u{2500}\u{2500}\u{2500} Phase 3: The Veins (500,000 years) \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}");
    let _ = writeln!(s, "  Hydrothermal veins:  {:.2} ms  ({} ore voxels from {} heat sources)", dur_ms(timings.veins_hydrothermal), veins_deposited, heat_source_count);
    let _ = writeln!(s, "  Formation growth:    {:.2} ms  ({} formations grown)", dur_ms(timings.veins_formations), formations_grown);
    let _ = writeln!(s, "  Phase 3 total:       {:.2} ms", dur_ms(timings.veins));

    let _ = writeln!(s);
    let _ = writeln!(s, "\u{2500}\u{2500}\u{2500} Phase 4: The Deep Time (1,250,000 years) \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}");
    let _ = writeln!(s, "  Supergene enrichment: {:.2} ms  ({} voxels enriched)", dur_ms(timings.deeptime_enrichment), voxels_enriched);
    let _ = writeln!(s, "  Vein thickening:      {:.2} ms", dur_ms(timings.deeptime_thickening));
    let _ = writeln!(s, "  Formations:           {:.2} ms", dur_ms(timings.deeptime_formations));
    if let Some(ref ct) = timings.collapse_sub {
        let _ = writeln!(s, "  Collapse:            {:.2} ms", dur_ms(timings.deeptime_collapse));
        let _ = writeln!(s, "    Support degradation: {:.2} ms  ({} degraded)", dur_ms(ct.support_degradation), supports_degraded);
        let _ = writeln!(s, "    Stress amplification: {:.2} ms", dur_ms(ct.stress_amplification));
        let _ = writeln!(s, "    Collapse cascade:    {:.2} ms  ({} collapses)", dur_ms(ct.collapse_cascade), collapses_triggered);
    }
    let _ = writeln!(s, "  Phase 4 total:       {:.2} ms", dur_ms(timings.deeptime));

    let _ = writeln!(s);
    let _ = writeln!(s, "\u{2500}\u{2500}\u{2500} Summary \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}");
    let _ = writeln!(s, "  Heat map computation:  {:.2} ms", dur_ms(timings.heat_map));
    let _ = writeln!(s, "  Chunk filter/classify: {:.2} ms", dur_ms(timings.chunk_filter + timings.chunk_classify));
    let _ = writeln!(s, "  Aggregation:           {:.2} ms", dur_ms(timings.aggregation));
    let _ = writeln!(s, "  Dirty chunks:          {}", dirty_chunk_count);
    let _ = writeln!(s, "  TOTAL:                {:.2} ms", dur_ms(timings.total));
    let _ = writeln!(s, "\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}");

    s
}

/// Send a progress update if a channel is provided.
fn send_progress(
    tx: Option<&Sender<SleepProgress>>,
    phase: u8,
    phase_name: &'static str,
    phase_years: u32,
    progress_pct: f32,
    chunks_processed: u32,
    chunks_total: u32,
    glimpse_chunk: Option<(i32, i32, i32)>,
    glimpse_type: u8,
    phase_summary: String,
) {
    if let Some(tx) = tx {
        let _ = tx.try_send(SleepProgress {
            phase,
            phase_name,
            phase_years,
            progress_pct,
            chunks_processed,
            chunks_total,
            glimpse_chunk,
            glimpse_type,
            phase_summary,
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
        let fluid = FluidSnapshot::default();
        let result = execute_sleep(
            &config,
            &mut density_fields,
            &mut stress_fields,
            &mut support_fields,
            &fluid,
            (0, 0, 0),
            1,
            None,
        );
        assert!(result.success);
    }

    #[test]
    fn test_execute_sleep_deterministic() {
        let (mut df1, mut sf1, mut sup1) = make_test_world(16);
        let (mut df2, mut sf2, mut sup2) = make_test_world(16);
        let config = SleepConfig::default();
        let fluid = FluidSnapshot::default();

        let r1 = execute_sleep(&config, &mut df1, &mut sf1, &mut sup1, &fluid, (0, 0, 0), 1, None);
        let r2 = execute_sleep(&config, &mut df2, &mut sf2, &mut sup2, &fluid, (0, 0, 0), 1, None);

        assert_eq!(r1.acid_dissolved, r2.acid_dissolved);
        assert_eq!(r1.voxels_metamorphosed, r2.voxels_metamorphosed);
        assert_eq!(r1.veins_deposited, r2.veins_deposited);
        assert_eq!(r1.supports_degraded, r2.supports_degraded);
        assert_eq!(r1.collapses_triggered, r2.collapses_triggered);
    }

    #[test]
    fn test_multiple_sleep_cycles() {
        let (mut density_fields, mut stress_fields, mut support_fields) = make_test_world(16);
        let config = SleepConfig::default();
        let fluid = FluidSnapshot::default();

        for cycle in 1..=3 {
            let result = execute_sleep(
                &config,
                &mut density_fields,
                &mut stress_fields,
                &mut support_fields,
                &fluid,
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
        let fluid = FluidSnapshot::default();

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
            &fluid,
            (0, 0, 0),
            1,
            None,
        );
        assert!(result.success);
    }

    #[test]
    fn test_sleep_manifest_records_changes() {
        let (mut density_fields, mut stress_fields, mut support_fields) = make_test_world(16);
        let config = SleepConfig::default();
        let fluid = FluidSnapshot::default();

        let result = execute_sleep(
            &config,
            &mut density_fields,
            &mut stress_fields,
            &mut support_fields,
            &fluid,
            (0, 0, 0),
            1,
            None,
        );

        // If there were any changes, the manifest should have entries
        if result.chunks_changed > 0 {
            assert!(!result.manifest.chunk_deltas.is_empty());
        }
    }
}
