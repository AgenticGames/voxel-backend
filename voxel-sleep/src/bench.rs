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

// ─── Helper: UE-matching config (collapse OFF for benchmarking) ────────────

fn make_ue_config() -> SleepConfig {
    let mut cfg = SleepConfig::default();
    cfg.veins.vein_deposition_prob = 0.35;
    cfg.veins.vein_max_distance = 16;
    cfg.veins.max_vein_voxels_per_source = 20;
    cfg.deeptime.enrichment_prob = 0.25;
    cfg.deeptime.vein_thickening_prob = 0.20;
    cfg.reaction.copper_oxidation_prob = 0.15;
    cfg.stress.propagation_radius = 4;
    cfg.stress.max_collapse_volume = 50;
    // Collapse OFF — isolate geological effects from structural destruction
    cfg.deeptime.collapse.collapse_enabled = false;
    cfg
}

// ─── Helper: Material census ───────────────────────────────────────────────

fn count_materials(density_fields: &HashMap<(i32, i32, i32), DensityField>) -> MatMap<u32> {
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

// ─── Helper: Fluid utilities ───────────────────────────────────────────────

fn empty_fluid_cells() -> Vec<FluidCell> {
    vec![FluidCell {
        level: 0.0,
        fluid_type: FluidType::Water,
        is_source: false,
        grace_ticks: 0,
        stagnant_ticks: 0,
    }; 4096]
}

/// Inject water sources into surface-adjacent air voxels (air with solid face neighbor).
/// Distributes evenly across all chunks. Returns count placed.
fn inject_water_sources(
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
                stagnant_ticks: 0,
            };
            placed += 1;
        }
    }
    placed
}

// ─── Helper: Realistic world (3×3×3 via region gen) ────────────────────────

fn make_realistic_world(seed: u64, water_count: usize) -> (
    HashMap<(i32, i32, i32), DensityField>,
    HashMap<(i32, i32, i32), StressField>,
    HashMap<(i32, i32, i32), SupportField>,
    FluidSnapshot,
) {
    let grid_size = 17;
    let mut coords = Vec::new();
    for cx in 0..3i32 {
        for cy in -5..-2i32 {
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
    let mut gen_water = 0u32;
    let mut gen_lava = 0u32;
    for fs in &fluid_seeds {
        match fs.fluid_type {
            voxel_gen::pools::PoolFluid::Water => gen_water += 1,
            voxel_gen::pools::PoolFluid::Lava => gen_lava += 1,
        }
        let cells = fluid.chunks.entry(fs.chunk).or_insert_with(empty_fluid_cells);
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

    // Inject additional water sources
    let water_placed = inject_water_sources(&density_fields, &mut fluid, water_count);

    eprintln!("  World: {} chunks, gen fluid: {} water + {} lava, injected: {} water",
        density_fields.len(), gen_water, gen_lava, water_placed);

    (density_fields, stress_fields, support_fields, fluid)
}

// ─── Helper: Synthetic world (controlled material + air channel) ───────────

fn make_synthetic_world(
    base_material: Material,
    lava_positions: &[(i32, i32, i32)],
    water_positions: &[(i32, i32, i32)],
) -> (
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
                for z in 0..grid_size {
                    for y in 0..grid_size {
                        for x in 0..grid_size {
                            let idx = df.index(x, y, z);
                            df.samples[idx] = VoxelSample {
                                density: 1.0,
                                material: base_material,
                            };
                        }
                    }
                }
                // Carve air channel at y=7,8 in center chunk
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

    let mut fluid = FluidSnapshot::default();

    // Place lava cells
    for &(wx, wy, wz) in lava_positions {
        let chunk = (wx.div_euclid(16), wy.div_euclid(16), wz.div_euclid(16));
        let lx = wx.rem_euclid(16) as usize;
        let ly = wy.rem_euclid(16) as usize;
        let lz = wz.rem_euclid(16) as usize;
        let cells = fluid.chunks.entry(chunk).or_insert_with(empty_fluid_cells);
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

    // Place water cells
    for &(wx, wy, wz) in water_positions {
        let chunk = (wx.div_euclid(16), wy.div_euclid(16), wz.div_euclid(16));
        let lx = wx.rem_euclid(16) as usize;
        let ly = wy.rem_euclid(16) as usize;
        let lz = wz.rem_euclid(16) as usize;
        let cells = fluid.chunks.entry(chunk).or_insert_with(empty_fluid_cells);
        let idx = lz * 16 * 16 + ly * 16 + lx;
        if idx < cells.len() {
            cells[idx] = FluidCell {
                level: 1.0,
                fluid_type: FluidType::Water,
                is_source: true,
                grace_ticks: 0,
                stagnant_ticks: 0,
            };
        }
    }

    (density_fields, stress_fields, support_fields, fluid)
}

// ─── Helper: BFS fluid cluster ─────────────────────────────────────────────

fn build_fluid_cluster(center: (i32, i32, i32), count: usize) -> Vec<(i32, i32, i32)> {
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

// ─── Fluid config descriptor ───────────────────────────────────────────────

struct FluidConfig {
    name: &'static str,
    water_count: usize,
}

const FLUID_CONFIGS: &[FluidConfig] = &[
    FluidConfig { name: "Dry",     water_count: 0 },
    FluidConfig { name: "Damp",    water_count: 30 },
    FluidConfig { name: "Wet",     water_count: 100 },
    FluidConfig { name: "Flooded", water_count: 300 },
];

// All non-Air material u8 IDs (1..=21)
const ALL_MAT_IDS: [u8; 21] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21];

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: Full statistical profile across fluid configs (4 × 50 = 200 runs)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn bench_sleep_statistics() {
    const RUNS_PER_CONFIG: u32 = 50;

    eprintln!("\n{:=<80}", "= SLEEP STATISTICS (collapse OFF, 50 runs × 4 fluid configs) ");

    // ── Summary comparison table ──
    eprintln!("\n{:<10} {:>5} {:>5} | {:>7} {:>7} {:>7} {:>7} {:>7} | {:>7} {:>7} {:>7} {:>7} | {:>6}",
        "Config", "Water", "Lava",
        "Marble", "Iron", "Copper", "Gold", "Sulfide",
        "Eroded", "Flowst", "Enrich", "Silici",
        "ms");
    eprintln!("{:-<115}", "");

    for fc in FLUID_CONFIGS {
        eprintln!("\n--- Generating world for '{}' (water={}) ---", fc.name, fc.water_count);
        let (template_density, template_stress, template_support, fluid) =
            make_realistic_world(42, fc.water_count);
        let before_census = count_materials(&template_density);

        // Count lava in fluid snapshot
        let lava_count: u32 = fluid.chunks.values()
            .flat_map(|cells| cells.iter())
            .filter(|c| c.level > 0.01 && c.fluid_type.is_lava())
            .count() as u32;

        let mut all_deltas: Vec<MatMap<i64>> = Vec::new();
        let mut all_total_ms: Vec<f64> = Vec::new();
        let mut all_results: Vec<crate::SleepResult> = Vec::new();

        for i in 0..RUNS_PER_CONFIG {
            let mut density = template_density.clone();
            let mut stress = template_stress.clone();
            let mut support = template_support.clone();
            let config = make_ue_config();

            let result = execute_sleep(
                &config, &mut density, &mut stress, &mut support,
                &fluid, (1, -4, 1), i, None,
            );

            let after = count_materials(&density);
            all_deltas.push(material_delta(&before_census, &after));
            all_total_ms.push(result.timings.total.as_secs_f64() * 1000.0);
            all_results.push(result);
        }

        // Compute averages for key materials
        let avg_delta = |mid: u8| -> f64 {
            let vals: Vec<f64> = all_deltas.iter()
                .map(|d| *d.get(&mid).unwrap_or(&0) as f64).collect();
            compute_stats(&vals).avg
        };

        let avg_counter = |f: fn(&crate::SleepResult) -> f64| -> f64 {
            let vals: Vec<f64> = all_results.iter().map(|r| f(r)).collect();
            compute_stats(&vals).avg
        };

        let timing = compute_stats(&all_total_ms);

        eprintln!("{:<10} {:>5} {:>5} | {:>+7.0} {:>+7.0} {:>+7.0} {:>+7.0} {:>+7.0} | {:>7.1} {:>7.1} {:>7.1} {:>7.1} | {:>6.0}",
            fc.name, fc.water_count, lava_count,
            avg_delta(mat_id(Material::Marble)),
            avg_delta(mat_id(Material::Iron)),
            avg_delta(mat_id(Material::Copper)),
            avg_delta(mat_id(Material::Gold)),
            avg_delta(mat_id(Material::Sulfide)),
            avg_counter(|r| r.acid_dissolved as f64),
            avg_counter(|r| r.formations_grown as f64),
            avg_counter(|r| r.voxels_enriched as f64),
            avg_counter(|r| r.voxels_silicified as f64),
            timing.avg,
        );

        // ── Detailed material deltas for this config ──
        eprintln!("\n  {:=<60}", format!("= {} MATERIAL DELTAS ", fc.name));
        eprintln!("  {:<14} {:>10} {:>+12} {:>10}", "Material", "Before", "DeltaAvg", "DeltaStd");
        eprintln!("  {:-<48}", "");
        for &mid in &ALL_MAT_IDS {
            let before_val = *before_census.get(&mid).unwrap_or(&0) as f64;
            let delta_vals: Vec<f64> = all_deltas.iter()
                .map(|d| *d.get(&mid).unwrap_or(&0) as f64).collect();
            let ds = compute_stats(&delta_vals);
            if before_val > 0.0 || ds.avg.abs() > 0.1 {
                eprintln!("  {:<14} {:>10.0} {:>+12.1} {:>10.1}",
                    mat_name(mid), before_val, ds.avg, ds.stddev);
            }
        }

        // ── Counters for this config ──
        eprintln!("\n  {:=<50}", format!("= {} COUNTERS ", fc.name));
        let counter_extractors: &[(&str, fn(&crate::SleepResult) -> f64)] = &[
            ("acid_dissolved",      |r| r.acid_dissolved as f64),
            ("voxels_oxidized",     |r| r.voxels_oxidized as f64),
            ("voxels_metamorphosed",|r| r.voxels_metamorphosed as f64),
            ("veins_deposited",     |r| r.veins_deposited as f64),
            ("formations_grown",    |r| r.formations_grown as f64),
            ("voxels_enriched",     |r| r.voxels_enriched as f64),
            ("sulfide_dissolved",   |r| r.sulfide_dissolved as f64),
            ("coal_matured",        |r| r.coal_matured as f64),
            ("diamonds_formed",     |r| r.diamonds_formed as f64),
            ("voxels_silicified",   |r| r.voxels_silicified as f64),
        ];
        eprintln!("  {:<22} {:>10} {:>10}", "Counter", "Avg", "Stddev");
        eprintln!("  {:-<44}", "");
        for &(name, extractor) in counter_extractors {
            let vals: Vec<f64> = all_results.iter().map(|r| extractor(r)).collect();
            let s = compute_stats(&vals);
            if s.avg.abs() > 0.01 {
                eprintln!("  {:<22} {:>10.1} {:>10.1}", name, s.avg, s.stddev);
            }
        }

        // ── Timing breakdown ──
        eprintln!("\n  Timing: avg={:.0}ms  p95={:.0}ms  min={:.0}ms  max={:.0}ms",
            timing.avg, timing.p95, timing.min, timing.max);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: Aureole heat scaling (5 lava sizes × 3 water levels × 10 runs)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn bench_aureole_heat_scaling() {
    let lava_sizes = [1, 5, 20, 100, 500];
    let water_counts = [0usize, 50, 200];
    // Lava centered at (24,23,24) in air channel of chunk (1,1,1)
    let lava_center = (24, 23, 24);
    let marble_id = mat_id(Material::Marble);
    let quartz_id = mat_id(Material::Quartz);
    let air_id = mat_id(Material::Air);

    // Test with both Limestone (shows silicification + erosion) and Slate (heat-only)
    let base_materials = [
        (Material::Limestone, "Limestone"),
        (Material::Slate, "Slate"),
    ];

    for &(base_mat, mat_label) in &base_materials {
        eprintln!("\n{:=<90}", format!("= AUREOLE SCALING — {} base (collapse OFF, 10 runs each) ", mat_label));
        eprintln!("{:<8} {:>6} | {:>10} {:>10} | {:>10} {:>10} | {:>8} {:>8}",
            "Lava", "Water", "MarbleAvg", "MarbleStd", "QuartzAvg", "QuartzStd", "Eroded", "ms");
        eprintln!("{:-<90}", "");

        for &lava_size in &lava_sizes {
            let lava_positions = build_fluid_cluster(lava_center, lava_size);

            for &wc in &water_counts {
                // Place water at y=23 along z=16..32, x=16..32 (floor of air channel)
                let water_positions: Vec<(i32, i32, i32)> = (16..32i32)
                    .flat_map(|x| (16..32i32).map(move |z| (x, 23, z)))
                    .filter(|pos| !lava_positions.contains(pos))
                    .take(wc)
                    .collect();

                let mut marble_deltas = Vec::new();
                let mut quartz_deltas = Vec::new();
                let mut air_deltas = Vec::new();
                let mut timing_ms = Vec::new();

                for run in 0..10u32 {
                    let (mut density, mut stress, mut support, fluid) =
                        make_synthetic_world(base_mat, &lava_positions, &water_positions);

                    let mut cfg = make_ue_config();
                    cfg.phase1_enabled = false;
                    cfg.phase2_enabled = true;
                    cfg.phase3_enabled = false;
                    cfg.phase4_enabled = false;
                    cfg.accumulation_enabled = false;

                    let before = count_materials(&density);
                    let result = execute_sleep(
                        &cfg, &mut density, &mut stress, &mut support,
                        &fluid, (1, 1, 1), run, None,
                    );
                    let after = count_materials(&density);
                    let delta = material_delta(&before, &after);

                    marble_deltas.push(*delta.get(&marble_id).unwrap_or(&0) as f64);
                    quartz_deltas.push(*delta.get(&quartz_id).unwrap_or(&0) as f64);
                    air_deltas.push(*delta.get(&air_id).unwrap_or(&0) as f64);
                    timing_ms.push(result.timings.total.as_secs_f64() * 1000.0);
                }

                let ms = compute_stats(&marble_deltas);
                let qs = compute_stats(&quartz_deltas);
                let es = compute_stats(&air_deltas);
                let ts = compute_stats(&timing_ms);
                eprintln!("{:<8} {:>6} | {:>+10.1} {:>10.1} | {:>+10.1} {:>10.1} | {:>+8.1} {:>8.0}",
                    lava_size, wc, ms.avg, ms.stddev, qs.avg, qs.stddev, es.avg, ts.avg);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: Epithermal rarity sweep (2 fluid configs × 5 rarities × 25 runs)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn bench_epithermal_rarity_sweep() {
    let fluid_configs = [
        ("Dry", 0usize),
        ("Wet", 100),
    ];
    let rarities = [0.1f32, 0.3, 0.5, 0.7, 1.0];

    let gold_id = mat_id(Material::Gold);
    let sulfide_id = mat_id(Material::Sulfide);
    let iron_id = mat_id(Material::Iron);
    let copper_id = mat_id(Material::Copper);

    eprintln!("\n{:=<90}", "= EPITHERMAL RARITY SWEEP (collapse OFF, 25 runs each) ");

    for &(fname, wcount) in &fluid_configs {
        eprintln!("\n--- Generating world for '{}' (water={}) ---", fname, wcount);
        let (template_density, template_stress, template_support, fluid) =
            make_realistic_world(42, wcount);
        let before_census = count_materials(&template_density);

        eprintln!("\n  {:<8} {:>12} {:>12} {:>12} {:>12}",
            "Rarity", "Gold a+/-s", "Sulf a+/-s", "Iron a+/-s", "Copr a+/-s");
        eprintln!("  {:-<60}", "");

        for &rarity in &rarities {
            let mut gold_d = Vec::new();
            let mut sulf_d = Vec::new();
            let mut iron_d = Vec::new();
            let mut copr_d = Vec::new();

            for run in 0..25u32 {
                let mut density = template_density.clone();
                let mut stress = template_stress.clone();
                let mut support = template_support.clone();

                let mut cfg = make_ue_config();
                cfg.veins.epithermal_rarity = rarity;

                execute_sleep(
                    &cfg, &mut density, &mut stress, &mut support,
                    &fluid, (1, -4, 1), run, None,
                );
                let after = count_materials(&density);
                let delta = material_delta(&before_census, &after);

                gold_d.push(*delta.get(&gold_id).unwrap_or(&0) as f64);
                sulf_d.push(*delta.get(&sulfide_id).unwrap_or(&0) as f64);
                iron_d.push(*delta.get(&iron_id).unwrap_or(&0) as f64);
                copr_d.push(*delta.get(&copper_id).unwrap_or(&0) as f64);
            }

            let gs = compute_stats(&gold_d);
            let ss = compute_stats(&sulf_d);
            let is = compute_stats(&iron_d);
            let cs = compute_stats(&copr_d);

            eprintln!("  {:<8.1} {:>+5.0}+/-{:<4.0} {:>+5.0}+/-{:<4.0} {:>+5.0}+/-{:<4.0} {:>+5.0}+/-{:<4.0}",
                rarity,
                gs.avg, gs.stddev,
                ss.avg, ss.stddev,
                is.avg, is.stddev,
                cs.avg, cs.stddev);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: Vein size comparison across fluid configs (2 × 25 = 50 runs)
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

    let fluid_configs = [
        ("Dry", 0usize),
        ("Wet", 100),
    ];

    for &(fname, wcount) in &fluid_configs {
        eprintln!("\n--- Generating world for '{}' (water={}) ---", fname, wcount);
        let (template_density, template_stress, template_support, fluid) =
            make_realistic_world(42, wcount);
        let before_census = count_materials(&template_density);

        let mut ore_deltas: BTreeMap<u8, Vec<f64>> = BTreeMap::new();
        for &ore in &ores {
            ore_deltas.insert(ore as u8, Vec::new());
        }

        for run in 0..25u32 {
            let mut density = template_density.clone();
            let mut stress = template_stress.clone();
            let mut support = template_support.clone();
            let cfg = make_ue_config();

            execute_sleep(
                &cfg, &mut density, &mut stress, &mut support,
                &fluid, (1, -4, 1), run, None,
            );
            let after = count_materials(&density);
            let delta = material_delta(&before_census, &after);

            for &ore in &ores {
                let id = ore as u8;
                ore_deltas.get_mut(&id).unwrap().push(*delta.get(&id).unwrap_or(&0) as f64);
            }
        }

        eprintln!("\n{:=<70}", format!("= {} VEIN DEPOSITS (25 runs) ", fname));
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
}
