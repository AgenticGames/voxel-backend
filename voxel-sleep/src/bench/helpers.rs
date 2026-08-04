use super::*;

use std::collections::{BTreeMap, HashMap};
use voxel_core::density::DensityField;
use voxel_core::material::Material;
use voxel_core::octree::node::VoxelSample;
use voxel_core::stress::{StressField, SupportField};
use voxel_fluid::{FluidSnapshot, cell::{FluidCell, FluidType}};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

use crate::config::SleepConfig;
use crate::execute_sleep;
use crate::util::{sleep_vein_size, default_vein_size};

pub(crate) fn mat_id(m: Material) -> u8 { m as u8 }
pub(crate) fn mat_name(id: u8) -> &'static str {
    match id {
        0 => "Air",
        1 => "Sandstone",
        2 => "Limestone",
        3 => "Granite",
        4 => "Basalt",
        5 => "Slate",
        6 => "Marble",
        7 => "Iron",
        8 => "Copper",
        9 => "Malachite",
        10 => "Tin",
        11 => "Gold",
        12 => "Diamond",
        13 => "Kimberlite",
        14 => "Sulfide",
        15 => "Quartz",
        16 => "Pyrite",
        17 => "Amethyst",
        18 => "Crystal",
        19 => "Coal",
        20 => "Graphite",
        21 => "Opal",
        22 => "Hornfels",
        23 => "Garnet",
        24 => "Diopside",
        25 => "Gypsum",
        _ => "Unknown",
    }
}

// ─── Helper: UE-matching config (collapse OFF for benchmarking) ────────────

pub(crate) fn make_ue_config() -> SleepConfig {
    let mut cfg = SleepConfig::default();
    // Veins (water-heat convergence model)
    cfg.veins.vein_deposition_prob = 0.85;
    cfg.veins.convergence_radius = 70.0;
    cfg.veins.veins_per_zone_max = 4;
    // Enrichment
    cfg.deeptime.enrichment_prob = 0.90;
    cfg.deeptime.max_enrichment_per_chunk = 400;
    cfg.deeptime.enrichment_cluster_max = 30;
    cfg.deeptime.enrichment_search_radius = 12;
    cfg.deeptime.vein_thickening_coat_depth = 1;
    cfg.deeptime.vein_thickening_finger_interval = 5;
    // Reaction (defaults: acid_prob=0.25, acid_cap=30, copper_ox=0.001, basalt=0.03, gypsum=0.18)
    // Aureole (defaults: radius=10, marble=0.90/0.60/0.30, garnet=0.35, diopside=0.80, recryst=0.70)
    // Stress/collapse
    cfg.stress.propagation_radius = 4;
    cfg.stress.max_collapse_volume = 50;
    // Collapse OFF — isolate geological effects from structural destruction
    cfg.deeptime.collapse.collapse_enabled = false;
    cfg
}

// ─── Helper: Material census ───────────────────────────────────────────────

pub(crate) fn count_materials(density_fields: &HashMap<(i32, i32, i32), DensityField>) -> MatMap<u32> {
    let mut counts: MatMap<u32> = BTreeMap::new();
    for df in density_fields.values() {
        let size = df.size;
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let sample = df.get(x, y, z);
                    if sample.density > 0.0 {
                        *counts.entry(sample.material as u8).or_insert(0) += 1;
                    }
                }
            }
        }
    }
    counts
}

pub(crate) fn material_delta(before: &MatMap<u32>, after: &MatMap<u32>) -> MatMap<i64> {
    let mut delta: MatMap<i64> = BTreeMap::new();
    let mut all_ids: std::collections::BTreeSet<u8> = before.keys().copied().collect();
    all_ids.extend(after.keys());
    for id in all_ids {
        let b = *before.get(&id).unwrap_or(&0) as i64;
        let a = *after.get(&id).unwrap_or(&0) as i64;
        let d = a - b;
        if d != 0 {
            delta.insert(id, d);
        }
    }
    delta
}

// ─── Helper: Stats ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(crate) struct Stats {
    pub(crate) min: f64,
    pub(crate) max: f64,
    pub(crate) avg: f64,
    pub(crate) stddev: f64,
    pub(crate) p95: f64,
}

pub(crate) fn compute_stats(values: &[f64]) -> Stats {
    if values.is_empty() {
        return Stats { min: 0.0, max: 0.0, avg: 0.0, stddev: 0.0, p95: 0.0 };
    }
    let n = values.len() as f64;
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let avg = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - avg).powi(2)).sum::<f64>() / n;
    let stddev = variance.sqrt();
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95_idx = ((values.len() as f64) * 0.95).ceil() as usize;
    let p95 = sorted[p95_idx.min(sorted.len() - 1)];
    Stats { min, max, avg, stddev, p95 }
}

// ─── Helper: Fluid utilities ───────────────────────────────────────────────

pub(crate) fn empty_fluid_cells() -> Vec<FluidCell> {
    vec![FluidCell {
        level: 0.0,
        fluid_type: FluidType::Water,
        is_source: false,
        grace_ticks: 0,
        stagnant_ticks: 0, drain_ticks: 0, hops_from_source: 255, max_flow_dist: 0,
    }; 4096]
}

/// Inject water sources into surface-adjacent air voxels (air with solid face neighbor).
/// Distributes evenly across all chunks. Returns count placed.
pub(crate) fn inject_water_sources(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    fluid: &mut FluidSnapshot,
    target_count: usize,
) -> usize {
    if target_count == 0 { return 0; }
    let chunk_size = 16usize;

    // Collect candidate positions: air voxels with at least one solid face neighbor
    let mut candidates: Vec<((i32, i32, i32), usize, usize, usize)> = Vec::new();
    let mut chunk_keys: Vec<_> = density_fields.keys().copied().collect();
    chunk_keys.sort();

    for &key in &chunk_keys {
        let df = &density_fields[&key];
        for lz in 0..chunk_size {
            for ly in 0..chunk_size {
                for lx in 0..chunk_size {
                    if df.get(lx, ly, lz).density > 0.0 { continue; }
                    let has_solid = [(1i32,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
                        .iter()
                        .any(|&(dx, dy, dz)| {
                            let nx = lx as i32 + dx;
                            let ny = ly as i32 + dy;
                            let nz = lz as i32 + dz;
                            nx >= 0 && nx < chunk_size as i32 &&
                            ny >= 0 && ny < chunk_size as i32 &&
                            nz >= 0 && nz < chunk_size as i32 &&
                            df.get(nx as usize, ny as usize, nz as usize).density > 0.0
                        });
                    if has_solid {
                        candidates.push((key, lx, ly, lz));
                    }
                }
            }
        }
    }

    if candidates.is_empty() { return 0; }
    let step = (candidates.len() / target_count).max(1);
    let mut placed = 0;
    for i in (0..candidates.len()).step_by(step) {
        if placed >= target_count { break; }
        let (key, lx, ly, lz) = candidates[i];
        let cells = fluid.chunks.entry(key).or_insert_with(empty_fluid_cells);
        let idx = lz * 16 * 16 + ly * 16 + lx;
        if idx < cells.len() && cells[idx].level < 0.01 {
            cells[idx] = FluidCell {
                level: 1.0,
                fluid_type: FluidType::Water,
                is_source: true,
                grace_ticks: 0,
                stagnant_ticks: 0, drain_ticks: 0, hops_from_source: 255, max_flow_dist: 0,
            };
            placed += 1;
        }
    }
    placed
}
