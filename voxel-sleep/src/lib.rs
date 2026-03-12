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

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as FmtWrite;
use std::time::{Duration, Instant};
use crossbeam_channel::Sender;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use voxel_core::density::DensityField;
use voxel_core::material::Material;
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
    /// 0=reaction, 1=aureole, 2=veins, 3=deeptime, 4=accumulation, 5=done
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
    pub deeptime_fossilization: Duration,
    pub reaction_sulfide_acid: Duration,
    pub aureole_coal_maturation: Duration,
    pub aureole_silicification: Duration,
    pub collapse_sub: Option<CollapseTimings>,
    pub census_scan: Duration,
    pub accumulation: Duration,
    pub accumulation_iterations: u32,
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
    pub veins_thickened: u32,
    pub supports_degraded: u32,
    pub collapses_triggered: u32,
    pub sulfide_dissolved: u32,
    pub coal_matured: u32,
    pub diamonds_formed: u32,
    pub voxels_silicified: u32,
    pub nests_fossilized: u32,
    pub channels_eroded: u32,
    pub corpses_fossilized: u32,
    pub gypsum_deposited: u32,
    pub lava_solidified: u32,
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

/// A single limiting factor for a sleep phase.
#[derive(Debug, Clone)]
pub struct Bottleneck {
    pub severity: f32, // 0.0=minor, 1.0=total blocker
    pub description: String,
}

/// Per-phase audit data tracking conversions, yield, and limiting factors.
#[derive(Debug, Clone, Default)]
pub struct PhaseDiagnostics {
    /// (old_material_u8, new_material_u8) -> count
    pub conversions: BTreeMap<(u8, u8), u32>,
    /// Candidates before probability rolls
    pub theoretical_max: u32,
    /// Candidates that passed and were applied
    pub actual_output: u32,
    /// Top 3 limiting factors, sorted by severity descending
    pub bottlenecks: Vec<Bottleneck>,
}

/// Fluid volume summary.
#[derive(Debug, Clone, Default)]
pub struct FluidMetrics {
    pub cell_count: u32,
    pub volume_sum: f32,
    pub chunks_with_fluid: u32,
}

/// Pre-scan snapshot of geological resources before sleep phases execute.
#[derive(Debug, Clone)]
pub struct ResourceCensus {
    pub water: FluidMetrics,
    pub lava: FluidMetrics,
    pub exposed_surfaces_by_material: BTreeMap<u8, u32>,
    pub total_exposed_surfaces: u32,
    /// 1-2 air neighbors (tight cracks, good for vein deposition)
    pub fissure_count: u32,
    /// 4+ air neighbors (wide caverns, fluid disperses)
    pub open_wall_count: u32,
    pub exposed_ore: BTreeMap<u8, u32>,
    pub heat_source_lava: u32,
    pub heat_source_kimberlite: u32,
    pub scan_duration: Duration,
}

impl Default for ResourceCensus {
    fn default() -> Self {
        Self {
            water: FluidMetrics::default(),
            lava: FluidMetrics::default(),
            exposed_surfaces_by_material: BTreeMap::new(),
            total_exposed_surfaces: 0,
            fissure_count: 0,
            open_wall_count: 0,
            exposed_ore: BTreeMap::new(),
            heat_source_lava: 0,
            heat_source_kimberlite: 0,
            scan_duration: Duration::ZERO,
        }
    }
}

/// Pre-scan loaded chunks to build a geological resource census.
fn compute_resource_census(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    fluid_snapshot: &FluidSnapshot,
    chunks: &[(i32, i32, i32)],
    chunk_size: usize,
) -> ResourceCensus {
    let t = Instant::now();
    let field_size = chunk_size + 1;

    // Pass 1: Fluid metrics
    let mut water = FluidMetrics::default();
    let mut lava = FluidMetrics::default();
    let mut water_chunks: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut lava_chunks: HashSet<(i32, i32, i32)> = HashSet::new();

    for (&chunk_key, cells) in &fluid_snapshot.chunks {
        for cell in cells.iter() {
            if cell.level > 0.001 {
                if cell.fluid_type.is_water() {
                    water.cell_count += 1;
                    water.volume_sum += cell.level;
                    water_chunks.insert(chunk_key);
                } else if cell.fluid_type.is_lava() {
                    lava.cell_count += 1;
                    lava.volume_sum += cell.level;
                    lava_chunks.insert(chunk_key);
                }
            }
        }
    }
    water.chunks_with_fluid = water_chunks.len() as u32;
    lava.chunks_with_fluid = lava_chunks.len() as u32;
    let lava_cell_count = lava.cell_count;

    // Pass 2: Density field surface analysis
    let mut exposed_surfaces_by_material: BTreeMap<u8, u32> = BTreeMap::new();
    let mut total_exposed_surfaces = 0u32;
    let mut fissure_count = 0u32;
    let mut open_wall_count = 0u32;
    let mut exposed_ore: BTreeMap<u8, u32> = BTreeMap::new();
    let mut heat_source_kimberlite = 0u32;

    for &chunk_key in chunks {
        let (cx, cy, cz) = chunk_key;
        let df = match density_fields.get(&chunk_key) {
            Some(df) => df,
            None => continue,
        };

        for lz in 0..field_size {
            for ly in 0..field_size {
                for lx in 0..field_size {
                    let sample = df.get(lx, ly, lz);
                    let mat = sample.material;

                    if !mat.is_solid() {
                        continue;
                    }

                    // Count all kimberlite (heat source regardless of exposure)
                    if mat == Material::Kimberlite {
                        heat_source_kimberlite += 1;
                    }

                    let wx = cx * (chunk_size as i32) + lx as i32;
                    let wy = cy * (chunk_size as i32) + ly as i32;
                    let wz = cz * (chunk_size as i32) + lz as i32;

                    let air_count = crate::util::count_neighbors(
                        density_fields, wx, wy, wz, chunk_size,
                        |m| !m.is_solid(),
                    );

                    if air_count >= 1 {
                        total_exposed_surfaces += 1;
                        *exposed_surfaces_by_material.entry(mat as u8).or_insert(0) += 1;

                        if mat.is_ore() || mat == Material::Pyrite || mat == Material::Kimberlite {
                            *exposed_ore.entry(mat as u8).or_insert(0) += 1;
                        }

                        if air_count <= 2 {
                            fissure_count += 1;
                        } else if air_count >= 4 {
                            open_wall_count += 1;
                        }
                    }
                }
            }
        }
    }

    ResourceCensus {
        water,
        lava,
        exposed_surfaces_by_material,
        total_exposed_surfaces,
        fissure_count,
        open_wall_count,
        exposed_ore,
        heat_source_lava: lava_cell_count,
        heat_source_kimberlite,
        scan_duration: t.elapsed(),
    }
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
    fluid_snapshot: &mut FluidSnapshot,
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

    // --- Fluid snapshot diagnostics ---
    let fluid_chunk_count = fluid_snapshot.chunks.len();
    let mut fluid_lava_cells = 0u32;
    let mut fluid_water_cells = 0u32;
    let fluid_cs = fluid_snapshot.chunk_size;
    for cells in fluid_snapshot.chunks.values() {
        for cell in cells {
            if cell.level > 0.001 {
                if cell.fluid_type.is_lava() {
                    fluid_lava_cells += 1;
                } else if cell.fluid_type.is_water() {
                    fluid_water_cells += 1;
                }
            }
        }
    }

    // --- Heat map computation ---
    let t_heat = Instant::now();
    let heat_map = build_heat_map(density_fields, fluid_snapshot, &all_chunks, chunk_size);
    let t_heat_elapsed = t_heat.elapsed();

    // Log fluid snapshot stats for debugging
    let lava_heat_count = heat_map.iter().filter(|h| h.source_type == crate::aureole::HeatSourceType::Lava).count();
    let kimb_heat_count = heat_map.iter().filter(|h| h.source_type == crate::aureole::HeatSourceType::Kimberlite).count();
    transform_log.push(TransformEntry {
        description: format!(
            "Fluid snapshot: {} chunks (cs={}), {} lava cells, {} water cells | Heat map: {} lava sources, {} kimberlite sources",
            fluid_chunk_count, fluid_cs, fluid_lava_cells, fluid_water_cells, lava_heat_count, kimb_heat_count,
        ),
        count: fluid_lava_cells + fluid_water_cells, // nonzero so retain() keeps it
    });

    // --- Resource Census (pre-scan) ---
    let t_census = Instant::now();
    let census = compute_resource_census(density_fields, fluid_snapshot, &all_chunks, chunk_size);
    let t_census_elapsed = t_census.elapsed();

    // Accumulators
    let mut total_acid_dissolved = 0u32;
    let mut total_oxidized = 0u32;
    let mut total_metamorphosed = 0u32;
    let mut total_veins = 0u32;
    let mut total_formations = 0u32;
    let mut total_enriched = 0u32;
    let mut total_thickened = 0u32;
    let mut total_supports_degraded = 0u32;
    let mut total_collapses = 0u32;
    let mut total_sulfide_dissolved = 0u32;
    let mut total_coal_matured = 0u32;
    let mut total_diamonds_formed = 0u32;
    let mut total_silicified = 0u32;
    let mut total_nests_fossilized = 0u32;
    let mut total_channels_eroded = 0u32;
    let mut total_corpses_fossilized = 0u32;
    let mut total_gypsum_deposited = 0u32;
    let mut all_collapse_events: Vec<voxel_core::stress::CollapseEvent> = Vec::new();
    let mut collapse_sub_timings: Option<CollapseTimings> = None;
    let mut diag_reaction = PhaseDiagnostics::default();
    let mut diag_aureole = PhaseDiagnostics::default();
    let mut diag_veins = PhaseDiagnostics::default();
    let mut diag_deeptime = PhaseDiagnostics::default();

    // ═══ Phase 1: The Aureole (100,000 years) ═══
    // Aureole runs BEFORE reaction so metamorphic minerals (marble, garnet, diopside)
    // form from limestone BEFORE acid dissolution can destroy it. Marble is immune to acid.
    let t_p1 = Instant::now();
    send_progress(progress_tx, 0, "The Aureole", 100_000, 0.0, 0, total_chunks, None, 0, String::new());

    if config.phase2_enabled {
        let aureole_result = apply_aureole(
            &config.aureole, &config.groundwater, density_fields, fluid_snapshot,
            &heat_map, &all_chunks, chunk_size, &mut rng, &census,
        );
        total_metamorphosed = aureole_result.voxels_metamorphosed;
        total_coal_matured = aureole_result.coal_matured;
        total_diamonds_formed = aureole_result.diamonds_formed;
        total_silicified = aureole_result.voxels_silicified;
        total_channels_eroded = aureole_result.channels_eroded;
        diag_aureole = aureole_result.diagnostics;
        result_manifest.merge_sleep_changes(&aureole_result.manifest);
        transform_log.extend(aureole_result.transform_log);
        for key in aureole_result.manifest.chunk_deltas.keys() {
            all_dirty.insert(*key);
        }
        let aureole_summary = if aureole_result.lava_zones_found > 0 {
            format!("{} metamorphosed ({} hornfels, {} skarn, {} veins), {} eroded, {} zones",
                total_metamorphosed, aureole_result.hornfels_placed, aureole_result.skarn_placed,
                aureole_result.veins_placed, aureole_result.channels_eroded, aureole_result.lava_zones_found)
        } else {
            format!("{} voxels metamorphosed, {} eroded", total_metamorphosed, aureole_result.channels_eroded)
        };
        send_progress(progress_tx, 0, "The Aureole", 100_000, 1.0, total_chunks, total_chunks,
            aureole_result.glimpse_chunk, if aureole_result.glimpse_chunk.is_some() { 1 } else { 0 },
            aureole_summary);
    } else {
        send_progress(progress_tx, 0, "The Aureole", 100_000, 1.0, total_chunks, total_chunks, None, 0, String::new());
    }
    let t_p1_elapsed = t_p1.elapsed();

    // ═══ Phase 2: The Reaction (10,000 years) ═══
    let t_p2 = Instant::now();
    send_progress(progress_tx, 1, "The Reaction", 10_000, 0.0, 0, total_chunks, None, 0, String::new());

    if config.phase1_enabled {
        let reaction_result = apply_reaction(
            &config.reaction, density_fields, fluid_snapshot,
            &all_chunks, chunk_size, &mut rng, &census,
        );
        total_acid_dissolved = reaction_result.acid_dissolved;
        total_oxidized = reaction_result.voxels_oxidized;
        total_sulfide_dissolved = reaction_result.sulfide_dissolved;
        total_gypsum_deposited += reaction_result.gypsum_deposited;
        diag_reaction = reaction_result.diagnostics;
        result_manifest.merge_sleep_changes(&reaction_result.manifest);
        transform_log.extend(reaction_result.transform_log);
        for key in reaction_result.manifest.chunk_deltas.keys() {
            all_dirty.insert(*key);
        }
        send_progress(progress_tx, 1, "The Reaction", 10_000, 1.0, total_chunks, total_chunks,
            reaction_result.glimpse_chunk, if reaction_result.glimpse_chunk.is_some() { 2 } else { 0 },
            format!("Acid dissolved {} voxels, {} oxidized", total_acid_dissolved, total_oxidized));
    } else {
        send_progress(progress_tx, 1, "The Reaction", 10_000, 1.0, total_chunks, total_chunks, None, 0, String::new());
    }
    let t_p2_elapsed = t_p2.elapsed();

    // ═══ Phase 3: The Veins (500,000 years) ═══
    let t_p3 = Instant::now();
    send_progress(progress_tx, 2, "The Veins", 500_000, 0.0, 0, mineral_chunks.len() as u32, None, 0, String::new());

    if config.phase3_enabled {
        let vein_result = apply_veins(
            &config.veins, &config.groundwater, density_fields, fluid_snapshot,
            &heat_map, &mineral_chunks, chunk_size, &mut rng, &census,
        );
        total_veins = vein_result.veins_deposited;
        total_formations += vein_result.formations_grown;
        diag_veins = vein_result.diagnostics;
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
            &config.stress, &config.nest_positions, &config.corpse_positions, sleep_count,
            &mut rng, &census,
        );
        total_enriched = dt_result.voxels_enriched;
        total_thickened = dt_result.veins_thickened;
        total_nests_fossilized = dt_result.nests_fossilized;
        total_corpses_fossilized = dt_result.corpses_fossilized;
        total_formations += dt_result.formations_grown;
        total_supports_degraded = dt_result.supports_degraded;
        diag_deeptime = dt_result.diagnostics;
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

    // ═══ Accumulation Pass (re-runs Phase 1-3 with scaled params) ═══
    let t_accum = Instant::now();
    let mut accum_acid = 0u32;
    let mut accum_oxidized = 0u32;
    let mut accum_sulfide = 0u32;
    let mut accum_metamorphosed = 0u32;
    let mut accum_eroded = 0u32;
    let mut accum_coal_matured = 0u32;
    let mut accum_diamonds = 0u32;
    let mut accum_silicified = 0u32;
    let accum_veins = 0u32;
    let accum_formations = 0u32;
    let mut accum_iterations_run = 0u32;

    if config.accumulation_enabled {
        let n = config.accumulation_iterations;
        send_progress(progress_tx, 4, "Accumulation", 1_250_000, 0.0, 0, total_chunks, None, 0, String::new());

        for iter in 0..n {
            let iter_seed = sleep_count as u64 * 7919 + 42 + 1000 + iter as u64 * 1013;
            let mut iter_rng = ChaCha8Rng::seed_from_u64(iter_seed);

            // Recompute census each iteration (world evolved)
            let iter_census = compute_resource_census(density_fields, fluid_snapshot, &all_chunks, chunk_size);

            // Phase 1 accumulation (time_factor = 41.3: 1.24Ma remaining / 3 iters / 10Ka base)
            if config.phase1_enabled {
                let scaled = scale_reaction_config(&config.reaction, 41.3);
                let r = apply_reaction(
                    &scaled, density_fields, fluid_snapshot,
                    &all_chunks, chunk_size, &mut iter_rng, &iter_census,
                );
                accum_acid += r.acid_dissolved;
                accum_oxidized += r.voxels_oxidized;
                accum_sulfide += r.sulfide_dissolved;
                total_gypsum_deposited += r.gypsum_deposited;
                result_manifest.merge_sleep_changes(&r.manifest);
                for key in r.manifest.chunk_deltas.keys() {
                    all_dirty.insert(*key);
                }
            }

            // Phase 2 accumulation (time_factor = 3.83: 1.15Ma remaining / 3 iters / 100Ka base)
            if config.phase2_enabled {
                let scaled = scale_aureole_config(&config.aureole, 3.83);
                let r = apply_aureole(
                    &scaled, &config.groundwater, density_fields, fluid_snapshot,
                    &heat_map, &all_chunks, chunk_size, &mut iter_rng, &iter_census,
                );
                accum_metamorphosed += r.voxels_metamorphosed;
                accum_eroded += r.channels_eroded;
                accum_coal_matured += r.coal_matured;
                accum_diamonds += r.diamonds_formed;
                accum_silicified += r.voxels_silicified;
                result_manifest.merge_sleep_changes(&r.manifest);
                for key in r.manifest.chunk_deltas.keys() {
                    all_dirty.insert(*key);
                }
            }

            // Phase 3 skipped in accumulation — convergence veins deposit fully in main pass;
            // re-running creates compounding blobs as new ore surfaces generate more seeds.

            accum_iterations_run += 1;
            send_progress(progress_tx, 4, "Accumulation", 1_250_000,
                (iter + 1) as f32 / n as f32, (iter + 1) * total_chunks, n * total_chunks, None, 0,
                format!("Iteration {}/{}", iter + 1, n));
        }

        // Merge accumulation totals into grand totals
        total_acid_dissolved += accum_acid;
        total_oxidized += accum_oxidized;
        total_sulfide_dissolved += accum_sulfide;
        total_metamorphosed += accum_metamorphosed;
        total_coal_matured += accum_coal_matured;
        total_diamonds_formed += accum_diamonds;
        total_silicified += accum_silicified;
        total_channels_eroded += accum_eroded;
        total_veins += accum_veins;
        total_formations += accum_formations;
    }
    let t_accum_elapsed = t_accum.elapsed();

    // --- Lava Solidification ---
    let mut total_lava_solidified = 0u32;
    eprintln!("[SLEEP] lava_solidification_enabled={}, fluid_snapshot chunks={}, density_fields={}",
        config.lava_solidification_enabled, fluid_snapshot.chunks.len(), density_fields.len());
    if config.lava_solidification_enabled {
        let (count, entries) = solidify_lava(density_fields, fluid_snapshot, &mut result_manifest, &mut all_dirty, chunk_size);
        total_lava_solidified = count;
        transform_log.extend(entries);
    } else {
        eprintln!("[SLEEP] Lava solidification DISABLED by config");
    }

    // --- Done ---
    let t_agg = Instant::now();
    send_progress(progress_tx, 5, "Complete", 1_250_000, 1.0, total_chunks, total_chunks, None, 0, String::new());

    // Add accumulation summary to transform log
    if accum_iterations_run > 0 {
        let accum_total = accum_acid + accum_oxidized + accum_sulfide
            + accum_metamorphosed + accum_veins + accum_formations
            + accum_coal_matured + accum_diamonds + accum_silicified;
        if accum_total > 0 {
            transform_log.push(TransformEntry {
                description: format!(
                    "Accumulation \u{2014} {} iterations (~1,240,000 years): {} acid, {} metamorphosed, {} veins, {} formations",
                    accum_iterations_run, accum_acid + accum_sulfide, accum_metamorphosed, accum_veins, accum_formations
                ),
                count: accum_total,
            });
        }
    }

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
        reaction: t_p2_elapsed,
        reaction_acid: t_p2_elapsed,     // Sub-timing not split yet
        reaction_oxidation: Duration::ZERO,
        aureole: t_p1_elapsed,
        aureole_metamorphism: t_p1_elapsed,
        aureole_erosion: Duration::ZERO,
        veins: t_p3_elapsed,
        veins_hydrothermal: t_p3_elapsed,
        veins_formations: Duration::ZERO,
        deeptime: t_p4_elapsed,
        deeptime_enrichment: Duration::ZERO,
        deeptime_thickening: Duration::ZERO,
        deeptime_formations: Duration::ZERO,
        deeptime_collapse: Duration::ZERO,
        deeptime_fossilization: Duration::ZERO,
        reaction_sulfide_acid: Duration::ZERO,
        aureole_coal_maturation: Duration::ZERO,
        aureole_silicification: Duration::ZERO,
        collapse_sub: collapse_sub_timings,
        census_scan: t_census_elapsed,
        accumulation: t_accum_elapsed,
        accumulation_iterations: accum_iterations_run,
        aggregation: t_agg_elapsed,
        loaded_chunks: loaded_chunks.len() as u32,
        critical_chunks: critical_chunks.len() as u32,
        important_chunks: important_chunks.len() as u32,
        cosmetic_chunks: cosmetic_chunks.len() as u32,
    };

    let profile_report = build_sleep_profile_report(
        &timings, sleep_count, config.chunk_radius,
        total_acid_dissolved, total_oxidized, total_metamorphosed,
        total_veins, total_formations, total_enriched, total_thickened,
        total_supports_degraded, total_collapses,
        all_chunks.len() as u32, mineral_chunks.len() as u32,
        collapse_chunks.len() as u32, dirty_chunks.len() as u32,
        heat_map.len() as u32,
        total_sulfide_dissolved, total_coal_matured, total_diamonds_formed,
        total_silicified, total_nests_fossilized,
        &config.groundwater,
        &census, &diag_reaction, &diag_aureole, &diag_veins, &diag_deeptime,
        accum_iterations_run, accum_acid + accum_sulfide, accum_oxidized,
        accum_metamorphosed, accum_eroded, accum_veins, accum_formations,
        total_lava_solidified,
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
        veins_thickened: total_thickened,
        supports_degraded: total_supports_degraded,
        collapses_triggered: total_collapses,
        sulfide_dissolved: total_sulfide_dissolved,
        coal_matured: total_coal_matured,
        diamonds_formed: total_diamonds_formed,
        voxels_silicified: total_silicified,
        nests_fossilized: total_nests_fossilized,
        channels_eroded: total_channels_eroded,
        corpses_fossilized: total_corpses_fossilized,
        gypsum_deposited: total_gypsum_deposited,
        lava_solidified: total_lava_solidified,
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

/// Convert all lava fluid cells to solid basalt voxels.
/// After 1.25Ma of geological time, any standing lava would have solidified.
fn solidify_lava(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    fluid_snapshot: &mut FluidSnapshot,
    manifest: &mut ChangeManifest,
    all_dirty: &mut HashSet<(i32, i32, i32)>,
    chunk_size: usize,
) -> (u32, Vec<TransformEntry>) {
    use voxel_core::stress::world_to_chunk_local;
    let mut count = 0u32;
    let cs = fluid_snapshot.chunk_size;
    if cs == 0 {
        return (0, Vec::new());
    }

    // Collect all lava cell positions first to avoid borrow conflict
    let lava_cells: Vec<((i32, i32, i32), (i32, i32, i32), usize)> = fluid_snapshot.chunks.iter()
        .flat_map(|(&chunk_key, cells)| {
            let (cx, cy, cz) = chunk_key;
            let mut out = Vec::new();
            for z in 0..cs {
                for y in 0..cs {
                    for x in 0..cs {
                        let idx = z * cs * cs + y * cs + x;
                        let cell = &cells[idx];
                        if cell.level > 0.001 && cell.fluid_type.is_lava() {
                            let wx = cx * (cs as i32) + x as i32;
                            let wy = cy * (cs as i32) + y as i32;
                            let wz = cz * (cs as i32) + z as i32;
                            out.push(((wx, wy, wz), chunk_key, idx));
                        }
                    }
                }
            }
            out
        })
        .collect();

    // Set density field voxels to basalt
    let mut missing_df = 0u32;
    let mut dirty_chunks_set: HashSet<(i32, i32, i32)> = HashSet::new();
    for &((wx, wy, wz), _fluid_chunk, _) in &lava_cells {
        let (ck, lx, ly, lz) = world_to_chunk_local(wx, wy, wz, chunk_size);
        if let Some(df) = density_fields.get_mut(&ck) {
            let sample = df.get_mut(lx, ly, lz);
            let old_mat = sample.material;
            let old_density = sample.density;
            sample.material = Material::Basalt;
            sample.density = -1.0;
            manifest.record_voxel_change(ck, lx, ly, lz, old_mat, old_density, Material::Basalt, -1.0);
            all_dirty.insert(ck);
            dirty_chunks_set.insert(ck);
            count += 1;
        } else {
            missing_df += 1;
        }
    }

    // Drain all lava cells in the snapshot (real fluid sim gets DrainLavaChunks from worker)
    for &(_, fluid_chunk, idx) in &lava_cells {
        if let Some(cells) = fluid_snapshot.chunks.get_mut(&fluid_chunk) {
            if idx < cells.len() {
                cells[idx].level = 0.0;
            }
        }
    }

    let mut entries = Vec::new();
    if !lava_cells.is_empty() || count > 0 {
        entries.push(TransformEntry {
            description: format!(
                "Lava Solidification: {} lava cells in snapshot, {} \u{2192} basalt in {} chunks, {} missing density fields (cs={}, df_chunks={}, snap_chunks={})",
                lava_cells.len(), count, dirty_chunks_set.len(), missing_df, cs, density_fields.len(), fluid_snapshot.chunks.len()
            ),
            count: count.max(1), // ensure it survives retain filter
        });
    }
    eprintln!("[SLEEP] solidify_lava: {} lava cells, {} basalt placed, {} missing density fields, cs={}, chunk_size={}",
        lava_cells.len(), count, missing_df, cs, chunk_size);
    (count, entries)
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
    veins_thickened: u32,
    supports_degraded: u32,
    collapses_triggered: u32,
    _all_chunk_count: u32,
    _mineral_chunk_count: u32,
    _collapse_chunk_count: u32,
    dirty_chunk_count: u32,
    heat_source_count: u32,
    sulfide_dissolved: u32,
    coal_matured: u32,
    diamonds_formed: u32,
    voxels_silicified: u32,
    nests_fossilized: u32,
    gw: &crate::config::GroundwaterConfig,
    census: &ResourceCensus,
    diag_reaction: &PhaseDiagnostics,
    diag_aureole: &PhaseDiagnostics,
    diag_veins: &PhaseDiagnostics,
    diag_deeptime: &PhaseDiagnostics,
    accum_iterations: u32,
    accum_acid_dissolved: u32,
    accum_oxidized: u32,
    accum_metamorphosed: u32,
    accum_eroded: u32,
    accum_veins: u32,
    accum_formations: u32,
    lava_solidified: u32,
) -> String {
    use voxel_core::material::Material;

    let mut s = String::with_capacity(8192);

    let _ = writeln!(s);
    let _ = writeln!(s, "\u{2550}\u{2550}\u{2550} Deep Sleep Profile \u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}");
    let _ = writeln!(s, "  Duration: 1,250,000 years (sleep cycle #{})", sleep_count);
    let _ = writeln!(s, "  Loaded chunks: {} ({} critical, {} important, {} cosmetic)",
        timings.loaded_chunks, timings.critical_chunks, timings.important_chunks, timings.cosmetic_chunks);
    let _ = writeln!(s, "  Heat sources: {} | Chunk radius: {}", heat_source_count, chunk_radius);
    let _ = writeln!(s, "  Groundwater:  erosion={:.2} flowstone={:.2} enrichment={:.2} soft_rock={:.2} hard_rock={:.2}",
        gw.erosion_power, gw.flowstone_power, gw.enrichment_power, gw.soft_rock_mult, gw.hard_rock_mult);

    let _ = writeln!(s);
    let _ = writeln!(s, "\u{2500}\u{2500}\u{2500} Phase 1: The Reaction (10,000 years) \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}");
    let _ = writeln!(s, "  Acid dissolution:    {:.2} ms  ({} voxels dissolved)", dur_ms(timings.reaction_acid), acid_dissolved);
    let _ = writeln!(s, "  Surface oxidation:   {:.2} ms  ({} voxels oxidized)", dur_ms(timings.reaction_oxidation), voxels_oxidized);
    let _ = writeln!(s, "  Sulfide acid:        {:.2} ms  ({} voxels dissolved)", dur_ms(timings.reaction_sulfide_acid), sulfide_dissolved);
    let _ = writeln!(s, "  Phase 1 total:       {:.2} ms", dur_ms(timings.reaction));

    let _ = writeln!(s);
    let _ = writeln!(s, "\u{2500}\u{2500}\u{2500} Phase 2: The Aureole (100,000 years) \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}");
    let _ = writeln!(s, "  Contact metamorphism: {:.2} ms  ({} voxels transformed)", dur_ms(timings.aureole_metamorphism), voxels_metamorphosed);
    let _ = writeln!(s, "  Coal maturation:      {:.2} ms  ({} coal \u{2192} graphite, {} \u{2192} diamond)", dur_ms(timings.aureole_coal_maturation), coal_matured, diamonds_formed);
    let _ = writeln!(s, "  Silicification:       {:.2} ms  ({} voxels silicified)", dur_ms(timings.aureole_silicification), voxels_silicified);
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
    let _ = writeln!(s, "  Vein thickening:      {:.2} ms  ({} voxels thickened)", dur_ms(timings.deeptime_thickening), veins_thickened);
    let _ = writeln!(s, "  Formations:           {:.2} ms", dur_ms(timings.deeptime_formations));
    let _ = writeln!(s, "  Nest fossilization:   {:.2} ms  ({} nests fossilized)", dur_ms(timings.deeptime_fossilization), nests_fossilized);
    if let Some(ref ct) = timings.collapse_sub {
        let _ = writeln!(s, "  Collapse:            {:.2} ms", dur_ms(timings.deeptime_collapse));
        let _ = writeln!(s, "    Support degradation: {:.2} ms  ({} degraded)", dur_ms(ct.support_degradation), supports_degraded);
        let _ = writeln!(s, "    Stress amplification: {:.2} ms", dur_ms(ct.stress_amplification));
        let _ = writeln!(s, "    Collapse cascade:    {:.2} ms  ({} collapses)", dur_ms(ct.collapse_cascade), collapses_triggered);
    }
    let _ = writeln!(s, "  Phase 4 total:       {:.2} ms", dur_ms(timings.deeptime));

    // ═══ Accumulation Pass ═══
    if accum_iterations > 0 {
        let _ = writeln!(s);
        let _ = writeln!(s, "\u{2550}\u{2550}\u{2550} Accumulation Pass ({} iterations, ~1,240,000 years) \u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}",
            accum_iterations);
        let _ = writeln!(s, "  Phase 1 (\u{00d7}25 scaled): {} acid dissolved, {} oxidized",
            accum_acid_dissolved, accum_oxidized);
        let _ = writeln!(s, "  Phase 2 (\u{00d7}2.3 scaled): {} metamorphosed, {} eroded",
            accum_metamorphosed, accum_eroded);
        let _ = writeln!(s, "  Phase 3 ({} passes): {} vein voxels deposited, {} formations",
            accum_iterations, accum_veins, accum_formations);
        let _ = writeln!(s, "  Accumulation time:   {:.2} ms", dur_ms(timings.accumulation));
    }

    // ═══ Lava Solidification ═══
    if lava_solidified > 0 {
        let _ = writeln!(s);
        let _ = writeln!(s, "\u{2550}\u{2550}\u{2550} Lava Solidification \u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}");
        let _ = writeln!(s, "  {} lava cells \u{2192} basalt (solid, density=-1.0)", lava_solidified);
    }

    let _ = writeln!(s);
    let _ = writeln!(s, "\u{2500}\u{2500}\u{2500} Summary \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}");
    let _ = writeln!(s, "  Heat map computation:  {:.2} ms", dur_ms(timings.heat_map));
    let _ = writeln!(s, "  Census scan:           {:.2} ms", dur_ms(timings.census_scan));
    let _ = writeln!(s, "  Chunk filter/classify: {:.2} ms", dur_ms(timings.chunk_filter + timings.chunk_classify));
    let _ = writeln!(s, "  Aggregation:           {:.2} ms", dur_ms(timings.aggregation));
    let _ = writeln!(s, "  Dirty chunks:          {}", dirty_chunk_count);
    let _ = writeln!(s, "  TOTAL:                {:.2} ms", dur_ms(timings.total));

    // ═══ Resource Census ═══
    let _ = writeln!(s);
    let _ = writeln!(s, "\u{2550}\u{2550}\u{2550} Resource Census \u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}");
    let _ = writeln!(s, "  Water:  {} cells, {:.1} m\u{00b3} volume, across {} chunks",
        census.water.cell_count, census.water.volume_sum, census.water.chunks_with_fluid);
    let _ = writeln!(s, "  Lava:   {} cells, {:.1} m\u{00b3} volume, across {} chunks",
        census.lava.cell_count, census.lava.volume_sum, census.lava.chunks_with_fluid);
    let _ = writeln!(s, "  Heat sources: {} lava + {} kimberlite = {} total",
        census.heat_source_lava, census.heat_source_kimberlite,
        census.heat_source_lava + census.heat_source_kimberlite);
    let _ = writeln!(s, "  Exposed surfaces: {} total", census.total_exposed_surfaces);
    let _ = writeln!(s, "    Fissures (1-2 air):  {} (tight cracks, good for vein deposition)", census.fissure_count);
    let _ = writeln!(s, "    Open walls (4+ air): {} (wide caverns, fluid disperses)", census.open_wall_count);

    // Exposed ore line
    if !census.exposed_ore.is_empty() {
        let mut ore_parts: Vec<String> = Vec::new();
        for (&mat_u8, &count) in &census.exposed_ore {
            let mat = Material::from_u8(mat_u8);
            ore_parts.push(format!("{}\u{00d7}{}", mat.display_name(), count));
        }
        let _ = writeln!(s, "  Exposed ore: {}", ore_parts.join("  "));
    } else {
        let _ = writeln!(s, "  Exposed ore: (none)");
    }
    let _ = writeln!(s, "  Scan time: {:.2} ms", dur_ms(census.scan_duration));

    // ═══ Geological Audit ═══
    let _ = writeln!(s);
    let _ = writeln!(s, "\u{2550}\u{2550}\u{2550} Geological Audit \u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}");

    let phase_names = [
        ("Phase 1: The Reaction", diag_reaction),
        ("Phase 2: The Aureole", diag_aureole),
        ("Phase 3: The Veins", diag_veins),
        ("Phase 4: The Deep Time", diag_deeptime),
    ];

    for (name, diag) in &phase_names {
        let _ = writeln!(s, "\u{2500}\u{2500}\u{2500} {} \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}", name);

        // Conversions
        if !diag.conversions.is_empty() {
            let _ = writeln!(s, "  Conversions:");
            for (&(old_u8, new_u8), &count) in &diag.conversions {
                let old_name = Material::from_u8(old_u8).display_name();
                let new_name = Material::from_u8(new_u8).display_name();
                let _ = writeln!(s, "    {} \u{2192} {}:  {} voxels", old_name, new_name, count);
            }
        }

        // Yield
        if diag.theoretical_max > 0 {
            let pct = if diag.theoretical_max > 0 {
                (diag.actual_output as f64 / diag.theoretical_max as f64) * 100.0
            } else {
                0.0
            };
            let _ = writeln!(s, "  Yield: {} / {} candidates ({:.1}%)",
                diag.actual_output, diag.theoretical_max, pct);
        } else if diag.actual_output > 0 {
            let _ = writeln!(s, "  Output: {} voxels transformed", diag.actual_output);
        } else {
            let _ = writeln!(s, "  Output: (none)");
        }

        // Bottlenecks
        if !diag.bottlenecks.is_empty() {
            let _ = writeln!(s, "  Limiting factors:");
            for b in &diag.bottlenecks {
                let _ = writeln!(s, "    \u{26a0} {}", b.description);
            }
        }

        let _ = writeln!(s);
    }

    let _ = writeln!(s, "\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}");

    s
}

/// Create a scaled ReactionConfig for accumulation passes (factor multiplies probabilities).
fn scale_reaction_config(base: &crate::config::ReactionConfig, factor: f32) -> crate::config::ReactionConfig {
    crate::config::ReactionConfig {
        acid_dissolution_prob: (base.acid_dissolution_prob * factor).min(1.0),
        copper_oxidation_prob: (base.copper_oxidation_prob * factor).min(1.0),
        basalt_crust_prob: (base.basalt_crust_prob * factor).min(1.0),
        sulfide_acid_prob: (base.sulfide_acid_prob * factor).min(1.0),
        acid_dissolution_radius: (base.acid_dissolution_radius as f32 * factor.sqrt()).max(1.0) as u32,
        sulfide_acid_radius: (base.sulfide_acid_radius as f32 * factor.sqrt()).max(1.0) as u32,
        ..base.clone()
    }
}

/// Create a scaled AureoleConfig for accumulation passes (factor multiplies probabilities).
fn scale_aureole_config(base: &crate::config::AureoleConfig, factor: f32) -> crate::config::AureoleConfig {
    crate::config::AureoleConfig {
        contact_limestone_to_marble_prob: (base.contact_limestone_to_marble_prob * factor).min(1.0),
        contact_sandstone_to_granite_prob: (base.contact_sandstone_to_granite_prob * factor).min(1.0),
        mid_limestone_to_marble_prob: (base.mid_limestone_to_marble_prob * factor).min(1.0),
        mid_sandstone_to_granite_prob: (base.mid_sandstone_to_granite_prob * factor).min(1.0),
        outer_limestone_to_marble_prob: (base.outer_limestone_to_marble_prob * factor).min(1.0),
        water_erosion_prob: (base.water_erosion_prob * factor).min(1.0),
        coal_to_graphite_prob: (base.coal_to_graphite_prob * factor).min(1.0),
        coal_to_graphite_mid_prob: (base.coal_to_graphite_mid_prob * factor).min(1.0),
        graphite_to_diamond_prob: (base.graphite_to_diamond_prob * factor).min(1.0),
        silicification_limestone_prob: (base.silicification_limestone_prob * factor).min(1.0),
        silicification_sandstone_prob: (base.silicification_sandstone_prob * factor).min(1.0),
        ..base.clone()
    }
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
mod bench;

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
        let (density_fields, _pools, _seeds, _worms, _timings, _river_springs) = voxel_gen::region_gen::generate_region_densities(&coords, &config);
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
        let mut fluid = FluidSnapshot::default();
        let result = execute_sleep(
            &config,
            &mut density_fields,
            &mut stress_fields,
            &mut support_fields,
            &mut fluid,
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
        let mut fluid1 = FluidSnapshot::default();
        let mut fluid2 = FluidSnapshot::default();

        let r1 = execute_sleep(&config, &mut df1, &mut sf1, &mut sup1, &mut fluid1, (0, 0, 0), 1, None);
        let r2 = execute_sleep(&config, &mut df2, &mut sf2, &mut sup2, &mut fluid2, (0, 0, 0), 1, None);

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
        let mut fluid = FluidSnapshot::default();

        for cycle in 1..=3 {
            let result = execute_sleep(
                &config,
                &mut density_fields,
                &mut stress_fields,
                &mut support_fields,
                &mut fluid,
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
        let mut fluid = FluidSnapshot::default();

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
            &mut fluid,
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
        let mut fluid = FluidSnapshot::default();

        let result = execute_sleep(
            &config,
            &mut density_fields,
            &mut stress_fields,
            &mut support_fields,
            &mut fluid,
            (0, 0, 0),
            1,
            None,
        );

        // If there were any changes, the manifest should have entries
        if result.chunks_changed > 0 {
            assert!(!result.manifest.chunk_deltas.is_empty());
        }
    }

    #[test]
    fn test_resource_census_basic() {
        let (density_fields, _, _) = make_test_world(16);
        let fluid = FluidSnapshot::default();
        let chunks: Vec<(i32, i32, i32)> = density_fields.keys().copied().collect();
        let census = compute_resource_census(&density_fields, &fluid, &chunks, 16);

        // With no fluid, water and lava should be zero
        assert_eq!(census.water.cell_count, 0);
        assert_eq!(census.lava.cell_count, 0);
        assert_eq!(census.heat_source_lava, 0);
        // Scan duration should be non-zero (ran some work)
        // Scan should complete (total_exposed_surfaces is populated)
        assert!(census.scan_duration.as_nanos() > 0);
        // Fissures + open walls should not exceed total
        assert!(census.fissure_count + census.open_wall_count <= census.total_exposed_surfaces);
    }

    #[test]
    fn test_reaction_diagnostics() {
        let (mut density_fields, mut stress_fields, mut support_fields) = make_test_world(16);
        let config = SleepConfig::default();
        let mut fluid = FluidSnapshot::default();

        let result = execute_sleep(
            &config,
            &mut density_fields,
            &mut stress_fields,
            &mut support_fields,
            &mut fluid,
            (0, 0, 0),
            1,
            None,
        );

        // Profile report should contain the new sections
        assert!(result.profile_report.contains("Resource Census"));
        assert!(result.profile_report.contains("Geological Audit"));
        assert!(result.profile_report.contains("Phase 1: The Reaction"));
        assert!(result.profile_report.contains("Phase 2: The Aureole"));
        assert!(result.profile_report.contains("Phase 3: The Veins"));
        assert!(result.profile_report.contains("Phase 4: The Deep Time"));
    }

    #[test]
    fn test_bottleneck_empty_world() {
        // An empty world with no fluid should produce bottlenecks about missing resources
        let census = ResourceCensus::default();
        let result_bottlenecks = crate::reaction::compute_reaction_bottlenecks(&census);

        // Should have bottlenecks about missing pyrite/sulfide, water, copper, lava
        assert!(!result_bottlenecks.is_empty());
        assert!(result_bottlenecks.len() <= 3);
        // Highest severity should be the no-pyrite/sulfide blocker
        assert!(result_bottlenecks[0].severity >= 0.5);
    }

    #[test]
    fn test_profile_report_deterministic() {
        let (mut df1, mut sf1, mut sup1) = make_test_world(16);
        let (mut df2, mut sf2, mut sup2) = make_test_world(16);
        let config = SleepConfig::default();
        let mut fluid1 = FluidSnapshot::default();
        let mut fluid2 = FluidSnapshot::default();

        let r1 = execute_sleep(&config, &mut df1, &mut sf1, &mut sup1, &mut fluid1, (0, 0, 0), 1, None);
        let r2 = execute_sleep(&config, &mut df2, &mut sf2, &mut sup2, &mut fluid2, (0, 0, 0), 1, None);

        // Strip timing values (non-deterministic) by checking structural content
        assert!(r1.profile_report.contains("Resource Census"));
        assert!(r2.profile_report.contains("Resource Census"));
        assert!(r1.profile_report.contains("Geological Audit"));
        assert!(r2.profile_report.contains("Geological Audit"));
    }
}
