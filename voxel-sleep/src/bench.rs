/// Sleep benchmark & tuning test suite.
///
/// All tests are `#[ignore]` — run with:
///   cargo test -p voxel-sleep bench_ -- --ignored --nocapture

use std::collections::{BTreeMap, HashMap};
use voxel_core::density::DensityField;
use voxel_core::material::Material;
use voxel_core::octree::node::VoxelSample;
use voxel_core::stress::{StressField, SupportField};
use voxel_fluid::{FluidSnapshot, cell::{FluidCell, FluidType}};

use crate::config::SleepConfig;
use crate::execute_sleep;
use crate::util::{sleep_vein_size, default_vein_size};

// Use u8 keys for maps since Material doesn't implement Ord.
type MatMap<V> = BTreeMap<u8, V>;

fn mat_id(m: Material) -> u8 { m as u8 }
fn mat_name(id: u8) -> &'static str {
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
        _ => "Unknown",
    }
}

// ─── Helper: UE-matching config ────────────────────────────────────────────

/// Build a SleepConfig that matches what the FFI layer sends to UE.
fn make_ue_config() -> SleepConfig {
    let mut cfg = SleepConfig::default();
    cfg.veins.vein_deposition_prob = 0.25;
    cfg.veins.vein_max_distance = 16;
    cfg.veins.max_vein_voxels_per_source = 12;
    cfg.deeptime.enrichment_prob = 0.15;
    cfg.deeptime.vein_thickening_prob = 0.10;
    // accumulation_iterations stays at Rust default (3) now that engine.rs is fixed
    cfg
}

// ─── Helper: Material census ───────────────────────────────────────────────

fn count_materials(density_fields: &HashMap<(i32, i32, i32), DensityField>) -> MatMap<u32> {
    let mut counts: MatMap<u32> = BTreeMap::new();
    for df in density_fields.values() {
        let size = df.size; // 17
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

fn material_delta(before: &MatMap<u32>, after: &MatMap<u32>) -> MatMap<i64> {
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
struct Stats {
    min: f64,
    max: f64,
    avg: f64,
    stddev: f64,
    p95: f64,
}

fn compute_stats(values: &[f64]) -> Stats {
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

// ─── Helper: Realistic world (3×3×3 via region gen) ────────────────────────

fn make_realistic_world(seed: u64) -> (
    HashMap<(i32, i32, i32), DensityField>,
    HashMap<(i32, i32, i32), StressField>,
    HashMap<(i32, i32, i32), SupportField>,
    FluidSnapshot,
) {
    let grid_size = 17; // chunk_size(16) + 1
    let mut coords = Vec::new();
    for cx in 0..3i32 {
        for cy in 0..3i32 {
            for cz in 0..3i32 {
                coords.push((cx, cy, cz));
            }
        }
    }

    let config = voxel_gen::config::GenerationConfig {
        seed,
        ..Default::default()
    };
    let (density_fields, _pools, fluid_seeds, _worms, _timings, _springs) =
        voxel_gen::region_gen::generate_region_densities(&coords, &config);

    let mut stress_fields = HashMap::new();
    let mut support_fields = HashMap::new();
    for &key in density_fields.keys() {
        stress_fields.insert(key, StressField::new(grid_size));
        support_fields.insert(key, SupportField::new(grid_size));
    }

    // Convert FluidSeeds → FluidSnapshot
    let mut fluid = FluidSnapshot::default();
    for fs in &fluid_seeds {
        let cells = fluid.chunks.entry(fs.chunk).or_insert_with(|| vec![
            FluidCell {
                level: 0.0,
                fluid_type: FluidType::Water,
                is_source: false,
                grace_ticks: 0,
                stagnant_ticks: 0,
            };
            4096
        ]);
        let idx = fs.lz as usize * 16 * 16 + fs.ly as usize * 16 + fs.lx as usize;
        if idx < cells.len() {
            let ft = match fs.fluid_type {
                voxel_gen::pools::PoolFluid::Water => FluidType::Water,
                voxel_gen::pools::PoolFluid::Lava => FluidType::Lava,
            };
            cells[idx] = FluidCell {
                level: 1.0,
                fluid_type: ft,
                is_source: fs.is_source,
                grace_ticks: 0,
                stagnant_ticks: 0,
            };
        }
    }

    (density_fields, stress_fields, support_fields, fluid)
}

// ─── Helper: Slate world (synthetic for aureole tests) ─────────────────────

fn make_slate_world(lava_positions: &[(i32, i32, i32)]) -> (
    HashMap<(i32, i32, i32), DensityField>,
    HashMap<(i32, i32, i32), StressField>,
    HashMap<(i32, i32, i32), SupportField>,
    FluidSnapshot,
) {
    let grid_size = 17;
    let mut density_fields = HashMap::new();
    let mut stress_fields = HashMap::new();
    let mut support_fields = HashMap::new();

    for cx in 0..3i32 {
        for cy in 0..3i32 {
            for cz in 0..3i32 {
                let mut df = DensityField::new(grid_size);
                // Fill all solid Slate
                for z in 0..grid_size {
                    for y in 0..grid_size {
                        for x in 0..grid_size {
                            let idx = df.index(x, y, z);
                            df.samples[idx] = VoxelSample {
                                density: 1.0,
                                material: Material::Slate,
                            };
                        }
                    }
                }
                // Carve air channel at y=7,8 in center chunk for exposed surfaces
                if cx == 1 && cy == 1 && cz == 1 {
                    for z in 0..grid_size {
                        for y in 7..=8 {
                            for x in 0..grid_size {
                                let idx = df.index(x, y, z);
                                df.samples[idx] = VoxelSample {
                                    density: 0.0,
                                    material: Material::Air,
                                };
                            }
                        }
                    }
                }
                density_fields.insert((cx, cy, cz), df);
                stress_fields.insert((cx, cy, cz), StressField::new(grid_size));
                support_fields.insert((cx, cy, cz), SupportField::new(grid_size));
            }
        }
    }

    // Place lava cells in fluid snapshot
    let mut fluid = FluidSnapshot::default();
    for &(wx, wy, wz) in lava_positions {
        let chunk = (wx.div_euclid(16), wy.div_euclid(16), wz.div_euclid(16));
        let lx = wx.rem_euclid(16) as usize;
        let ly = wy.rem_euclid(16) as usize;
        let lz = wz.rem_euclid(16) as usize;
        let cells = fluid.chunks.entry(chunk).or_insert_with(|| vec![
            FluidCell {
                level: 0.0,
                fluid_type: FluidType::Water,
                is_source: false,
                grace_ticks: 0,
                stagnant_ticks: 0,
            };
            4096
        ]);
        let idx = lz * 16 * 16 + ly * 16 + lx;
        if idx < cells.len() {
            cells[idx] = FluidCell {
                level: 1.0,
                fluid_type: FluidType::Lava,
                is_source: true,
                grace_ticks: 0,
                stagnant_ticks: 0,
            };
        }
    }

    (density_fields, stress_fields, support_fields, fluid)
}

// ─── Helper: BFS lava cluster ──────────────────────────────────────────────

fn build_lava_cluster(center: (i32, i32, i32), count: usize) -> Vec<(i32, i32, i32)> {
    use std::collections::{HashSet, VecDeque};
    let mut placed = HashSet::new();
    let mut result = Vec::new();
    let mut queue = VecDeque::new();
    queue.push_back(center);
    placed.insert(center);

    while result.len() < count {
        let pos = match queue.pop_front() {
            Some(p) => p,
            None => break,
        };
        result.push(pos);
        for &(dx, dy, dz) in &[(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)] {
            let n = (pos.0 + dx, pos.1 + dy, pos.2 + dz);
            if !placed.contains(&n) {
                placed.insert(n);
                queue.push_back(n);
            }
        }
    }
    result
}

// ─── Material IDs for iteration ────────────────────────────────────────────

// All non-Air material u8 IDs (1..=21)
const ALL_MAT_IDS: [u8; 21] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21];

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: Full statistical profile (200 runs)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn bench_sleep_statistics() {
    const RUNS: u32 = 200;
    let config = make_ue_config();

    eprintln!("\n=== Generating realistic world (seed=42, 3x3x3 chunks) ===");
    let (template_density, template_stress, template_support, fluid) = make_realistic_world(42);
    let before_census = count_materials(&template_density);

    eprintln!("=== Running {} sleep cycles ===", RUNS);

    let mut all_after_census: Vec<MatMap<u32>> = Vec::new();
    let mut all_deltas: Vec<MatMap<i64>> = Vec::new();
    let mut all_total_ms: Vec<f64> = Vec::new();
    let mut all_phase_ms: Vec<[f64; 5]> = Vec::new();
    let mut all_results: Vec<crate::SleepResult> = Vec::new();

    for i in 0..RUNS {
        let mut density = template_density.clone();
        let mut stress = template_stress.clone();
        let mut support = template_support.clone();

        let result = execute_sleep(
            &config,
            &mut density,
            &mut stress,
            &mut support,
            &fluid,
            (1, 1, 1),
            i,
            None,
        );

        let after = count_materials(&density);
        let delta = material_delta(&before_census, &after);

        let t = &result.timings;
        let total_ms = t.total.as_secs_f64() * 1000.0;
        let phase_ms = [
            t.reaction.as_secs_f64() * 1000.0,
            t.aureole.as_secs_f64() * 1000.0,
            t.veins.as_secs_f64() * 1000.0,
            t.deeptime.as_secs_f64() * 1000.0,
            t.accumulation.as_secs_f64() * 1000.0,
        ];

        all_after_census.push(after);
        all_deltas.push(delta);
        all_total_ms.push(total_ms);
        all_phase_ms.push(phase_ms);
        all_results.push(result);

        if (i + 1) % 50 == 0 {
            eprintln!("  ... completed {}/{}", i + 1, RUNS);
        }
    }

    // ── Table 1: Material Deltas ──
    eprintln!("\n{:=<68}", "= MATERIAL DELTAS (200 runs) ");
    eprintln!("{:<14} {:>9} {:>10} {:>10} {:>10}",
        "Material", "Before", "AfterAvg", "DeltaAvg", "DeltaStd");
    eprintln!("{:-<54}", "");
    for &mid in &ALL_MAT_IDS {
        let before_val = *before_census.get(&mid).unwrap_or(&0) as f64;
        let after_vals: Vec<f64> = all_after_census.iter()
            .map(|c| *c.get(&mid).unwrap_or(&0) as f64)
            .collect();
        let delta_vals: Vec<f64> = all_deltas.iter()
            .map(|d| *d.get(&mid).unwrap_or(&0) as f64)
            .collect();
        let after_s = compute_stats(&after_vals);
        let delta_s = compute_stats(&delta_vals);
        if before_val > 0.0 || delta_s.avg.abs() > 0.1 {
            eprintln!("{:<14} {:>9.0} {:>10.1} {:>+10.1} {:>10.1}",
                mat_name(mid), before_val, after_s.avg, delta_s.avg, delta_s.stddev);
        }
    }

    // ── Table 2: Ore Ratios ──
    eprintln!("\n{:=<60}", "= ORE RATIOS (200 runs) ");
    eprintln!("{:<14} {:>9} {:>9} {:>9} {:>9}", "Ratio", "Avg", "Stddev", "Min", "Max");
    eprintln!("{:-<50}", "");
    let ratios: &[(&str, u8, u8)] = &[
        ("Gold/Iron", mat_id(Material::Gold), mat_id(Material::Iron)),
        ("Sulfide/Iron", mat_id(Material::Sulfide), mat_id(Material::Iron)),
        ("Copper/Iron", mat_id(Material::Copper), mat_id(Material::Iron)),
    ];
    for &(name, num_id, den_id) in ratios {
        let ratio_vals: Vec<f64> = all_after_census.iter().map(|c| {
            let num = *c.get(&num_id).unwrap_or(&0) as f64;
            let den = *c.get(&den_id).unwrap_or(&1) as f64;
            if den > 0.0 { num / den } else { 0.0 }
        }).collect();
        let s = compute_stats(&ratio_vals);
        eprintln!("{:<14} {:>9.4} {:>9.4} {:>9.4} {:>9.4}", name, s.avg, s.stddev, s.min, s.max);
    }

    // ── Table 3: Marble Production ──
    let marble_id = mat_id(Material::Marble);
    let marble_deltas: Vec<f64> = all_deltas.iter()
        .map(|d| *d.get(&marble_id).unwrap_or(&0) as f64)
        .collect();
    let marble_stats = compute_stats(&marble_deltas);
    eprintln!("\n{:=<50}", "= MARBLE PRODUCTION (200 runs) ");
    eprintln!("  delta avg:  {:+.1}", marble_stats.avg);
    eprintln!("  delta std:  {:.1}", marble_stats.stddev);
    eprintln!("  delta min:  {:+.0}", marble_stats.min);
    eprintln!("  delta max:  {:+.0}", marble_stats.max);

    // ── Table 4: Timing ──
    let timing_stats = compute_stats(&all_total_ms);
    eprintln!("\n{:=<50}", "= TIMING (200 runs) ");
    eprintln!("  total_ms min: {:>10.1}", timing_stats.min);
    eprintln!("  total_ms avg: {:>10.1}", timing_stats.avg);
    eprintln!("  total_ms max: {:>10.1}", timing_stats.max);
    eprintln!("  total_ms p95: {:>10.1}", timing_stats.p95);

    // ── Table 5: Phase Breakdown ──
    let phase_names = ["Reaction", "Aureole", "Veins", "DeepTime", "Accumulation"];
    eprintln!("\n{:=<50}", "= PHASE BREAKDOWN (200 runs) ");
    eprintln!("{:<14} {:>10} {:>8}", "Phase", "Avg ms", "Avg %");
    eprintln!("{:-<34}", "");
    for p in 0..5 {
        let phase_vals: Vec<f64> = all_phase_ms.iter().map(|ms| ms[p]).collect();
        let pct_vals: Vec<f64> = all_phase_ms.iter().enumerate().map(|(i, ms)| {
            if all_total_ms[i] > 0.0 { ms[p] / all_total_ms[i] * 100.0 } else { 0.0 }
        }).collect();
        let ms_s = compute_stats(&phase_vals);
        let pct_s = compute_stats(&pct_vals);
        eprintln!("{:<14} {:>10.2} {:>7.1}%", phase_names[p], ms_s.avg, pct_s.avg);
    }

    // ── Table 6: SleepResult Counters ──
    eprintln!("\n{:=<50}", "= SLEEP RESULT COUNTERS (200 runs) ");
    eprintln!("{:<22} {:>10} {:>10}", "Counter", "Avg", "Stddev");
    eprintln!("{:-<44}", "");
    let counter_extractors: &[(&str, fn(&crate::SleepResult) -> f64)] = &[
        ("chunks_changed",      |r| r.chunks_changed as f64),
        ("acid_dissolved",      |r| r.acid_dissolved as f64),
        ("voxels_oxidized",     |r| r.voxels_oxidized as f64),
        ("voxels_metamorphosed",|r| r.voxels_metamorphosed as f64),
        ("veins_deposited",     |r| r.veins_deposited as f64),
        ("formations_grown",    |r| r.formations_grown as f64),
        ("voxels_enriched",     |r| r.voxels_enriched as f64),
        ("supports_degraded",   |r| r.supports_degraded as f64),
        ("collapses_triggered", |r| r.collapses_triggered as f64),
        ("sulfide_dissolved",   |r| r.sulfide_dissolved as f64),
        ("coal_matured",        |r| r.coal_matured as f64),
        ("diamonds_formed",     |r| r.diamonds_formed as f64),
        ("voxels_silicified",   |r| r.voxels_silicified as f64),
    ];
    for &(name, extractor) in counter_extractors {
        let vals: Vec<f64> = all_results.iter().map(|r| extractor(r)).collect();
        let s = compute_stats(&vals);
        eprintln!("{:<22} {:>10.1} {:>10.1}", name, s.avg, s.stddev);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: Aureole heat scaling (5 cluster sizes × 20 runs)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn bench_aureole_heat_scaling() {
    let cluster_sizes = [1, 5, 20, 100, 500];
    // Center lava in chunk (1,1,1) at local (8,7,8) = world (24,23,24)
    let lava_center = (24, 23, 24);
    let marble_id = mat_id(Material::Marble);

    eprintln!("\n{:=<50}", "= AUREOLE HEAT SCALING ");
    eprintln!("{:<14} {:>12} {:>12}", "ClusterSize", "MarbleAvg", "MarbleStd");
    eprintln!("{:-<40}", "");

    for &size in &cluster_sizes {
        let lava_positions = build_lava_cluster(lava_center, size);

        let mut marble_deltas = Vec::new();
        for run in 0..20u32 {
            let (mut density, mut stress, mut support, fluid) = make_slate_world(&lava_positions);

            let mut cfg = make_ue_config();
            cfg.phase1_enabled = false;
            cfg.phase2_enabled = true;
            cfg.phase3_enabled = false;
            cfg.phase4_enabled = false;
            cfg.accumulation_enabled = false;

            let before = count_materials(&density);
            execute_sleep(&cfg, &mut density, &mut stress, &mut support, &fluid, (1,1,1), run, None);
            let after = count_materials(&density);
            let delta = material_delta(&before, &after);

            marble_deltas.push(*delta.get(&marble_id).unwrap_or(&0) as f64);
        }

        let s = compute_stats(&marble_deltas);
        eprintln!("{:<14} {:>+12.1} {:>12.1}", size, s.avg, s.stddev);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: Epithermal rarity sweep (10 values × 50 runs)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn bench_epithermal_rarity_sweep() {
    eprintln!("\n=== Generating realistic world for epithermal sweep ===");
    let (template_density, template_stress, template_support, fluid) = make_realistic_world(42);
    let before_census = count_materials(&template_density);

    let gold_id = mat_id(Material::Gold);
    let sulfide_id = mat_id(Material::Sulfide);
    let iron_id = mat_id(Material::Iron);
    let copper_id = mat_id(Material::Copper);

    eprintln!("\n{:=<70}", "= EPITHERMAL RARITY SWEEP (0.1-1.0, 50 runs each) ");
    eprintln!("{:<8} {:>12} {:>12} {:>12} {:>12}",
        "Rarity", "Gold a+/-s", "Sulf a+/-s", "Iron a+/-s", "Copr a+/-s");
    eprintln!("{:-<58}", "");

    for step in 1..=10u32 {
        let rarity = step as f32 * 0.1;

        let mut gold_deltas = Vec::new();
        let mut sulfide_deltas = Vec::new();
        let mut iron_deltas = Vec::new();
        let mut copper_deltas = Vec::new();

        for run in 0..50u32 {
            let mut density = template_density.clone();
            let mut stress = template_stress.clone();
            let mut support = template_support.clone();

            let mut cfg = make_ue_config();
            cfg.veins.epithermal_rarity = rarity;

            execute_sleep(&cfg, &mut density, &mut stress, &mut support, &fluid, (1,1,1), run, None);
            let after = count_materials(&density);
            let delta = material_delta(&before_census, &after);

            gold_deltas.push(*delta.get(&gold_id).unwrap_or(&0) as f64);
            sulfide_deltas.push(*delta.get(&sulfide_id).unwrap_or(&0) as f64);
            iron_deltas.push(*delta.get(&iron_id).unwrap_or(&0) as f64);
            copper_deltas.push(*delta.get(&copper_id).unwrap_or(&0) as f64);
        }

        let gs = compute_stats(&gold_deltas);
        let ss = compute_stats(&sulfide_deltas);
        let is = compute_stats(&iron_deltas);
        let cs = compute_stats(&copper_deltas);

        eprintln!("{:<8.1} {:>+5.0}+/-{:<4.0} {:>+5.0}+/-{:<4.0} {:>+5.0}+/-{:<4.0} {:>+5.0}+/-{:<4.0}",
            rarity,
            gs.avg, gs.stddev,
            ss.avg, ss.stddev,
            is.avg, is.stddev,
            cs.avg, cs.stddev);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: Vein size comparison (50 runs)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn bench_vein_size_comparison() {
    let ores = [
        Material::Iron, Material::Copper, Material::Tin, Material::Gold,
        Material::Sulfide, Material::Malachite, Material::Quartz, Material::Pyrite,
    ];

    eprintln!("\n{:=<60}", "= VEIN SIZE PARAMETERS ");
    eprintln!("{:<14} {:>16} {:>16}", "Ore", "Sleep min-max", "Default min-max");
    eprintln!("{:-<48}", "");
    for &ore in &ores {
        let (smin, smax) = sleep_vein_size(ore);
        let (dmin, dmax) = default_vein_size(ore);
        eprintln!("{:<14} {:>7}-{:<7} {:>7}-{:<7}",
            format!("{:?}", ore), smin, smax, dmin, dmax);
    }

    eprintln!("\n=== Generating realistic world for vein comparison ===");
    let (template_density, template_stress, template_support, fluid) = make_realistic_world(42);
    let before_census = count_materials(&template_density);

    eprintln!("=== Running 50 sleep cycles ===");

    let mut ore_deltas: BTreeMap<u8, Vec<f64>> = BTreeMap::new();
    for &ore in &ores {
        ore_deltas.insert(ore as u8, Vec::new());
    }

    for run in 0..50u32 {
        let mut density = template_density.clone();
        let mut stress = template_stress.clone();
        let mut support = template_support.clone();
        let cfg = make_ue_config();

        execute_sleep(&cfg, &mut density, &mut stress, &mut support, &fluid, (1,1,1), run, None);
        let after = count_materials(&density);
        let delta = material_delta(&before_census, &after);

        for &ore in &ores {
            let id = ore as u8;
            ore_deltas.get_mut(&id).unwrap().push(*delta.get(&id).unwrap_or(&0) as f64);
        }
    }

    eprintln!("\n{:=<70}", "= VEIN SIZE vs ACTUAL DEPOSITS (50 runs) ");
    eprintln!("{:<14} {:>14} {:>16} {:>14}", "Ore", "ConfigRange", "ActualAvgDelta", "ActualStddev");
    eprintln!("{:-<60}", "");
    for &ore in &ores {
        let (smin, smax) = sleep_vein_size(ore);
        let id = ore as u8;
        let vals = ore_deltas.get(&id).unwrap();
        let s = compute_stats(vals);
        eprintln!("{:<14} {:>6}-{:<6} {:>+16.1} {:>14.1}",
            format!("{:?}", ore), smin, smax, s.avg, s.stddev);
    }
}
