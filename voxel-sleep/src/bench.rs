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
        22 => "Hornfels",
        23 => "Garnet",
        24 => "Diopside",
        25 => "Gypsum",
        _ => "Unknown",
    }
}

// ─── Helper: UE-matching config (collapse OFF for benchmarking) ────────────

fn make_ue_config() -> SleepConfig {
    let mut cfg = SleepConfig::default();
    cfg.veins.vein_deposition_prob = 0.35;
    cfg.veins.vein_max_distance = 22;
    cfg.veins.max_vein_voxels_per_source = 20;
    cfg.deeptime.enrichment_prob = 0.25;
    cfg.deeptime.vein_thickening_prob = 0.20;
    cfg.reaction.copper_oxidation_prob = 0.15;
    cfg.reaction.acid_dissolution_prob = 0.45;
    cfg.reaction.acid_max_dissolved_per_source = 80;
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
        for cy in -2..1i32 {
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

// ─── Helper: Seed material into world at world-space positions ──────────────

/// Overwrite voxels at the given world-space positions with `material`.
/// Returns how many voxels were actually placed (chunk existed + coords in range).
fn seed_material(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    positions: &[(i32, i32, i32)],
    material: Material,
    chunk_size: usize,
) -> usize {
    let cs = chunk_size as i32;
    let mut placed = 0;
    for &(wx, wy, wz) in positions {
        let chunk = (wx.div_euclid(cs), wy.div_euclid(cs), wz.div_euclid(cs));
        let lx = wx.rem_euclid(cs) as usize;
        let ly = wy.rem_euclid(cs) as usize;
        let lz = wz.rem_euclid(cs) as usize;
        if let Some(df) = density_fields.get_mut(&chunk) {
            let sample = df.get_mut(lx, ly, lz);
            sample.material = material;
            sample.density = 1.0;
            placed += 1;
        }
    }
    placed
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
const ALL_MAT_IDS: [u8; 25] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25];

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: Full statistical profile across fluid configs (4 × 50 = 200 runs)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn bench_sleep_statistics() {
    const RUNS_PER_CONFIG: u32 = 50;

    eprintln!("\n{:=<80}", "= SLEEP STATISTICS (collapse OFF, 50 runs × 4 fluid configs) ");

    // ── Summary comparison table ──
    eprintln!("\n{:<10} {:>5} {:>5} | {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} | {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} | {:>6}",
        "Config", "Water", "Lava",
        "Marble", "Iron", "Copper", "Gold", "Sulfide", "Pyrite",
        "Eroded", "ChErode", "Flowst", "Enrich", "Silici", "Corpse",
        "ms");
    eprintln!("{:-<145}", "");

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
            let mut run_fluid = fluid.clone();
            let mut config = make_ue_config();
            // Inject test fossilization targets
            config.nest_positions = vec![
                (8, -24, 8), (24, -20, 24), (40, -16, 12), (16, -28, 32), (32, -22, 20),
            ];
            config.corpse_positions = vec![
                (10, -24, 10), (12, -20, 22), (38, -18, 14), (20, -26, 30), (28, -22, 18),
                (6, -24, 6), (26, -20, 26), (36, -16, 16), (14, -28, 28), (30, -22, 22),
            ];

            let result = execute_sleep(
                &config, &mut density, &mut stress, &mut support,
                &mut run_fluid, (1, -1, 1), i, None,
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

        eprintln!("{:<10} {:>5} {:>5} | {:>+7.0} {:>+7.0} {:>+7.0} {:>+7.0} {:>+7.0} {:>+7.0} | {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} | {:>6.0}",
            fc.name, fc.water_count, lava_count,
            avg_delta(mat_id(Material::Marble)),
            avg_delta(mat_id(Material::Iron)),
            avg_delta(mat_id(Material::Copper)),
            avg_delta(mat_id(Material::Gold)),
            avg_delta(mat_id(Material::Sulfide)),
            avg_delta(mat_id(Material::Pyrite)),
            avg_counter(|r| r.acid_dissolved as f64),
            avg_counter(|r| r.channels_eroded as f64),
            avg_counter(|r| r.formations_grown as f64),
            avg_counter(|r| r.voxels_enriched as f64),
            avg_counter(|r| r.voxels_silicified as f64),
            avg_counter(|r| r.corpses_fossilized as f64),
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
            ("channels_eroded",    |r| r.channels_eroded as f64),
            ("corpses_fossilized", |r| r.corpses_fossilized as f64),
            ("nests_fossilized",   |r| r.nests_fossilized as f64),
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
                    let (mut density, mut stress, mut support, mut fluid) =
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
                        &mut fluid, (1, 1, 1), run, None,
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
                let mut run_fluid = fluid.clone();

                let mut cfg = make_ue_config();
                cfg.veins.epithermal_rarity = rarity;

                execute_sleep(
                    &cfg, &mut density, &mut stress, &mut support,
                    &mut run_fluid, (1, -1, 1), run, None,
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
            let mut run_fluid = fluid.clone();
            let cfg = make_ue_config();

            execute_sleep(
                &cfg, &mut density, &mut stress, &mut support,
                &mut run_fluid, (1, -1, 1), run, None,
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

// ═══════════════════════════════════════════════════════════════════════════
// Test 5: Limestone dramatic sleep (15 iterations × 3 cycles)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn bench_limestone_sleep_dramatic() {
    const ITERATIONS: u32 = 15;
    const CYCLES: u32 = 3;

    eprintln!("\n{:=<80}", "= LIMESTONE DRAMATIC SLEEP (15 iters × 3 cycles) ");

    // Build seeding positions (world-space, center chunk is (1,1,1) → world 16..31)
    // Air channel at y=23,24 in chunk (1,1,1) → world y = 16+7 = 23, 16+8 = 24
    let lava_cluster = build_fluid_cluster((24, 20, 24), 20);

    // Water spread across air channel floor (step_by(2) ensures coverage over seeded ores)
    let water_positions: Vec<(i32, i32, i32)> = (16..30i32).step_by(2)
        .flat_map(|x| (16..30i32).step_by(2).map(move |z| (x, 23, z)))
        .take(30)
        .collect();

    // Pyrite at y=22 (floor, one below air channel)
    let pyrite_positions: Vec<(i32, i32, i32)> = (18..26i32)
        .flat_map(|x| (18..26i32).map(move |z| (x, 22, z)))
        .step_by(4)
        .take(16)
        .collect();

    // Sulfide at y=22 (interleaved with pyrite)
    let sulfide_positions: Vec<(i32, i32, i32)> = (19..26i32)
        .flat_map(|x| (19..26i32).map(move |z| (x, 22, z)))
        .step_by(5)
        .take(9)
        .collect();

    // Copper at y=25 (ceiling, one above air channel)
    let copper_positions: Vec<(i32, i32, i32)> = (18..26i32)
        .flat_map(|x| (18..26i32).map(move |z| (x, 25, z)))
        .step_by(5)
        .take(12)
        .collect();

    // Gold at y=25
    let gold_positions: Vec<(i32, i32, i32)> = (20..26i32)
        .flat_map(|x| (20..26i32).map(move |z| (x, 25, z)))
        .step_by(6)
        .take(6)
        .collect();

    // Accumulators across all iterations
    let mut all_acid: Vec<u32> = Vec::new();
    let mut all_sulfide: Vec<u32> = Vec::new();
    let mut all_gypsum: Vec<u32> = Vec::new();
    let mut all_formations: Vec<u32> = Vec::new();
    let mut all_silicified: Vec<u32> = Vec::new();
    let mut all_veins: Vec<u32> = Vec::new();
    let mut all_enriched: Vec<u32> = Vec::new();
    let mut all_metamorphosed: Vec<u32> = Vec::new();
    let mut all_deltas: Vec<MatMap<i64>> = Vec::new();

    for iter in 0..ITERATIONS {
        let (mut density, mut stress, mut support, mut fluid) =
            make_synthetic_world(Material::Limestone, &lava_cluster, &water_positions);

        // Seed ore minerals
        let n_pyr = seed_material(&mut density, &pyrite_positions, Material::Pyrite, 16);
        let n_sul = seed_material(&mut density, &sulfide_positions, Material::Sulfide, 16);
        let n_cu = seed_material(&mut density, &copper_positions, Material::Copper, 16);
        let n_au = seed_material(&mut density, &gold_positions, Material::Gold, 16);
        if iter == 0 {
            eprintln!("  Seeded: {} pyrite, {} sulfide, {} copper, {} gold", n_pyr, n_sul, n_cu, n_au);
        }

        let before = count_materials(&density);

        let mut total_acid = 0u32;
        let mut total_sulfide = 0u32;
        let mut total_gypsum = 0u32;
        let mut total_formations = 0u32;
        let mut total_silicified = 0u32;
        let mut total_veins = 0u32;
        let mut total_enriched = 0u32;
        let mut total_metamorphosed = 0u32;

        for cycle in 0..CYCLES {
            let mut cfg = make_ue_config();
            cfg.sleep_count = cycle + 1;
            let result = execute_sleep(
                &cfg, &mut density, &mut stress, &mut support,
                &mut fluid, (1, 1, 1), iter * CYCLES + cycle, None,
            );
            total_acid += result.acid_dissolved;
            total_sulfide += result.sulfide_dissolved;
            total_gypsum += result.gypsum_deposited;
            total_formations += result.formations_grown;
            total_silicified += result.voxels_silicified;
            total_veins += result.veins_deposited;
            total_enriched += result.voxels_enriched;
            total_metamorphosed += result.voxels_metamorphosed;
        }

        let after = count_materials(&density);
        let delta = material_delta(&before, &after);

        eprintln!("  [iter {:>2}] acid={:<4} sulfide={:<4} gypsum={:<4} meta={:<4} form={:<4} silic={:<4} veins={:<4} enrich={:<4}",
            iter, total_acid, total_sulfide, total_gypsum, total_metamorphosed,
            total_formations, total_silicified, total_veins, total_enriched);

        all_acid.push(total_acid);
        all_sulfide.push(total_sulfide);
        all_gypsum.push(total_gypsum);
        all_formations.push(total_formations);
        all_silicified.push(total_silicified);
        all_veins.push(total_veins);
        all_enriched.push(total_enriched);
        all_metamorphosed.push(total_metamorphosed);
        all_deltas.push(delta);
    }

    // ── Averages ──
    eprintln!("\n--- Averages ---");
    let avg_u32 = |v: &[u32]| -> (f64, u32, u32) {
        let sum: u32 = v.iter().sum();
        let avg = sum as f64 / v.len() as f64;
        let min = *v.iter().min().unwrap();
        let max = *v.iter().max().unwrap();
        (avg, min, max)
    };
    let print_avg = |name: &str, v: &[u32]| {
        let (avg, min, max) = avg_u32(v);
        eprintln!("  {:<22} avg={:<8.1} min={:<6} max={}", name, avg, min, max);
    };
    print_avg("acid_dissolved:", &all_acid);
    print_avg("sulfide_dissolved:", &all_sulfide);
    print_avg("gypsum_deposited:", &all_gypsum);
    print_avg("metamorphosed:", &all_metamorphosed);
    print_avg("formations_grown:", &all_formations);
    print_avg("silicified:", &all_silicified);
    print_avg("veins_deposited:", &all_veins);
    print_avg("enriched:", &all_enriched);

    // ── Material Deltas (avg) ──
    eprintln!("\n--- Material Deltas (avg) ---");
    for &mid in &ALL_MAT_IDS {
        let vals: Vec<f64> = all_deltas.iter()
            .map(|d| *d.get(&mid).unwrap_or(&0) as f64).collect();
        let s = compute_stats(&vals);
        if s.avg.abs() > 0.1 || s.max.abs() > 0.1 {
            eprintln!("  {:<14} {:>+10.1}", mat_name(mid), s.avg);
        }
    }

    // ── Process Health ──
    eprintln!("\n--- Process Health ---");
    let n = ITERATIONS as usize;

    let count_pass = |v: &[u32], pred: fn(u32) -> bool| -> usize {
        v.iter().filter(|&&x| pred(x)).count()
    };
    let gt0 = |x: u32| x > 0;

    let delta_pass = |mid: u8, pred: fn(i64) -> bool| -> usize {
        all_deltas.iter().filter(|d| pred(*d.get(&mid).unwrap_or(&0))).count()
    };
    let dgt0 = |x: i64| x > 0;

    // Dramatic volume: sum of absolute deltas > 200
    let dramatic_count = all_deltas.iter().filter(|d| {
        let vol: i64 = d.values().map(|v| v.abs()).sum();
        vol > 200
    }).count();

    struct HealthCheck {
        name: &'static str,
        actual: usize,
        target: usize,
        total: usize,
    }
    let checks = vec![
        HealthCheck { name: "Acid dissolution fires",  actual: count_pass(&all_acid, gt0),    target: n,      total: n },
        HealthCheck { name: "Sulfide acid fires",       actual: count_pass(&all_sulfide, gt0), target: n,      total: n },
        HealthCheck { name: "Gypsum forms",             actual: count_pass(&all_gypsum, gt0),  target: 10,     total: n },
        HealthCheck { name: "Marble forms",             actual: delta_pass(mat_id(Material::Marble), dgt0),   target: n,  total: n },
        HealthCheck { name: "Garnet forms",             actual: delta_pass(mat_id(Material::Garnet), dgt0),   target: 12, total: n },
        HealthCheck { name: "Diopside forms",           actual: delta_pass(mat_id(Material::Diopside), dgt0), target: 10, total: n },
        HealthCheck { name: "Stalactites/formations",   actual: count_pass(&all_formations, gt0),              target: n,  total: n },
        HealthCheck { name: "Malachite forms",          actual: delta_pass(mat_id(Material::Malachite), dgt0), target: 10, total: n },
        HealthCheck { name: "Silicification fires",     actual: count_pass(&all_silicified, gt0),              target: 8,  total: n },
        HealthCheck { name: "Veins deposit",            actual: count_pass(&all_veins, gt0),                   target: n,  total: n },
        HealthCheck { name: "Enrichment fires",         actual: count_pass(&all_enriched, gt0),                target: 10, total: n },
        HealthCheck { name: "Dramatic volume (>200)",   actual: dramatic_count,                                target: n,  total: n },
    ];

    let mut any_fail = false;
    for c in &checks {
        let status = if c.actual >= c.target { "PASS" } else { "FAIL" };
        if c.actual < c.target { any_fail = true; }
        eprintln!("  [{}] {} ({}/{}){}", status, c.name, c.actual, c.total,
            if c.actual < c.target { " ← needs tuning" } else { "" });
    }

    // Hard asserts on critical invariants
    let acid_count = count_pass(&all_acid, gt0);
    assert!(acid_count >= n, "CRITICAL: Acid dissolution should ALWAYS fire on limestone ({}/{})", acid_count, n);
    let marble_count = delta_pass(mat_id(Material::Marble), dgt0);
    assert!(marble_count >= n, "CRITICAL: Marble should ALWAYS form from limestone ({}/{})", marble_count, n);

    if any_fail {
        eprintln!("\n  ⚠ Some non-critical processes underperforming — may need config tuning");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 6: Slate dramatic sleep (15 iterations × 3 cycles)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn bench_slate_sleep_dramatic() {
    const ITERATIONS: u32 = 15;
    const CYCLES: u32 = 3;

    eprintln!("\n{:=<80}", "= SLATE DRAMATIC SLEEP (15 iters × 3 cycles) ");

    // Same geometry as limestone test
    let lava_cluster = build_fluid_cluster((24, 20, 24), 20);

    // Water spread across air channel floor (step_by(2) ensures coverage over seeded ores)
    let water_positions: Vec<(i32, i32, i32)> = (16..30i32).step_by(2)
        .flat_map(|x| (16..30i32).step_by(2).map(move |z| (x, 23, z)))
        .take(30)
        .collect();

    let pyrite_positions: Vec<(i32, i32, i32)> = (18..26i32)
        .flat_map(|x| (18..26i32).map(move |z| (x, 22, z)))
        .step_by(4)
        .take(16)
        .collect();

    let sulfide_positions: Vec<(i32, i32, i32)> = (19..26i32)
        .flat_map(|x| (19..26i32).map(move |z| (x, 22, z)))
        .step_by(5)
        .take(9)
        .collect();

    let copper_positions: Vec<(i32, i32, i32)> = (18..26i32)
        .flat_map(|x| (18..26i32).map(move |z| (x, 25, z)))
        .step_by(5)
        .take(12)
        .collect();

    let gold_positions: Vec<(i32, i32, i32)> = (20..26i32)
        .flat_map(|x| (20..26i32).map(move |z| (x, 25, z)))
        .step_by(6)
        .take(6)
        .collect();

    let mut all_acid: Vec<u32> = Vec::new();
    let mut all_sulfide_acid: Vec<u32> = Vec::new();
    let mut all_gypsum: Vec<u32> = Vec::new();
    let mut all_formations: Vec<u32> = Vec::new();
    let mut all_veins: Vec<u32> = Vec::new();
    let mut all_enriched: Vec<u32> = Vec::new();
    let mut all_metamorphosed: Vec<u32> = Vec::new();
    let mut all_deltas: Vec<MatMap<i64>> = Vec::new();

    for iter in 0..ITERATIONS {
        let (mut density, mut stress, mut support, mut fluid) =
            make_synthetic_world(Material::Slate, &lava_cluster, &water_positions);

        let n_pyr = seed_material(&mut density, &pyrite_positions, Material::Pyrite, 16);
        let n_sul = seed_material(&mut density, &sulfide_positions, Material::Sulfide, 16);
        let n_cu = seed_material(&mut density, &copper_positions, Material::Copper, 16);
        let n_au = seed_material(&mut density, &gold_positions, Material::Gold, 16);
        if iter == 0 {
            eprintln!("  Seeded: {} pyrite, {} sulfide, {} copper, {} gold", n_pyr, n_sul, n_cu, n_au);
        }

        let before = count_materials(&density);

        let mut total_acid = 0u32;
        let mut total_sulfide_acid = 0u32;
        let mut total_gypsum = 0u32;
        let mut total_formations = 0u32;
        let mut total_veins = 0u32;
        let mut total_enriched = 0u32;
        let mut total_metamorphosed = 0u32;

        for cycle in 0..CYCLES {
            let mut cfg = make_ue_config();
            cfg.sleep_count = cycle + 1;
            let result = execute_sleep(
                &cfg, &mut density, &mut stress, &mut support,
                &mut fluid, (1, 1, 1), iter * CYCLES + cycle, None,
            );
            total_acid += result.acid_dissolved;
            total_sulfide_acid += result.sulfide_dissolved;
            total_gypsum += result.gypsum_deposited;
            total_formations += result.formations_grown;
            total_veins += result.veins_deposited;
            total_enriched += result.voxels_enriched;
            total_metamorphosed += result.voxels_metamorphosed;
        }

        let after = count_materials(&density);
        let delta = material_delta(&before, &after);

        eprintln!("  [iter {:>2}] acid={:<4} sulfide_acid={:<4} gypsum={:<4} meta={:<4} form={:<4} veins={:<4} enrich={:<4}",
            iter, total_acid, total_sulfide_acid, total_gypsum, total_metamorphosed,
            total_formations, total_veins, total_enriched);

        all_acid.push(total_acid);
        all_sulfide_acid.push(total_sulfide_acid);
        all_gypsum.push(total_gypsum);
        all_formations.push(total_formations);
        all_veins.push(total_veins);
        all_enriched.push(total_enriched);
        all_metamorphosed.push(total_metamorphosed);
        all_deltas.push(delta);
    }

    // ── Averages ──
    eprintln!("\n--- Averages ---");
    let avg_u32 = |v: &[u32]| -> (f64, u32, u32) {
        let sum: u32 = v.iter().sum();
        let avg = sum as f64 / v.len() as f64;
        let min = *v.iter().min().unwrap();
        let max = *v.iter().max().unwrap();
        (avg, min, max)
    };
    let print_avg = |name: &str, v: &[u32]| {
        let (avg, min, max) = avg_u32(v);
        eprintln!("  {:<22} avg={:<8.1} min={:<6} max={}", name, avg, min, max);
    };
    print_avg("acid_dissolved:", &all_acid);
    print_avg("sulfide_acid:", &all_sulfide_acid);
    print_avg("gypsum_deposited:", &all_gypsum);
    print_avg("metamorphosed:", &all_metamorphosed);
    print_avg("formations_grown:", &all_formations);
    print_avg("veins_deposited:", &all_veins);
    print_avg("enriched:", &all_enriched);

    // ── Material Deltas (avg) ──
    eprintln!("\n--- Material Deltas (avg) ---");
    for &mid in &ALL_MAT_IDS {
        let vals: Vec<f64> = all_deltas.iter()
            .map(|d| *d.get(&mid).unwrap_or(&0) as f64).collect();
        let s = compute_stats(&vals);
        if s.avg.abs() > 0.1 || s.max.abs() > 0.1 {
            eprintln!("  {:<14} {:>+10.1}", mat_name(mid), s.avg);
        }
    }

    // ── Process Health ──
    eprintln!("\n--- Process Health ---");
    let n = ITERATIONS as usize;

    let count_pass = |v: &[u32], pred: fn(u32) -> bool| -> usize {
        v.iter().filter(|&&x| pred(x)).count()
    };
    let eq0 = |x: u32| x == 0;
    let gt0 = |x: u32| x > 0;

    let delta_pass = |mid: u8, pred: fn(i64) -> bool| -> usize {
        all_deltas.iter().filter(|d| pred(*d.get(&mid).unwrap_or(&0))).count()
    };
    let deq0 = |x: i64| x == 0;
    let dgt0 = |x: i64| x > 0;

    struct HealthCheck {
        name: &'static str,
        actual: usize,
        target: usize,
        total: usize,
    }
    let checks = vec![
        HealthCheck { name: "Acid does NOT fire",       actual: count_pass(&all_acid, eq0),                    target: n,  total: n },
        HealthCheck { name: "Gypsum does NOT form",     actual: count_pass(&all_gypsum, eq0),                  target: n,  total: n },
        HealthCheck { name: "Hornfels forms",           actual: delta_pass(mat_id(Material::Hornfels), dgt0),  target: n,  total: n },
        HealthCheck { name: "Marble does NOT form",     actual: delta_pass(mat_id(Material::Marble), deq0),    target: n,  total: n },
        HealthCheck { name: "No stalactites",           actual: count_pass(&all_formations, eq0),              target: 13, total: n },
        HealthCheck { name: "Gold veins",               actual: delta_pass(mat_id(Material::Gold), dgt0),      target: 8,  total: n },
        HealthCheck { name: "Pyrite co-deposition",     actual: delta_pass(mat_id(Material::Pyrite), dgt0),    target: 12, total: n },
        HealthCheck { name: "Quartz veins",             actual: delta_pass(mat_id(Material::Quartz), dgt0),    target: 8,  total: n },
        HealthCheck { name: "Veins deposit",            actual: count_pass(&all_veins, gt0),                   target: n,  total: n },
        HealthCheck { name: "Cu oxidation (malachite)", actual: delta_pass(mat_id(Material::Malachite), dgt0), target: 8,  total: n },
        HealthCheck { name: "Enrichment fires",         actual: count_pass(&all_enriched, gt0),                target: 8,  total: n },
    ];

    let mut any_fail = false;
    for c in &checks {
        let status = if c.actual >= c.target { "PASS" } else { "FAIL" };
        if c.actual < c.target { any_fail = true; }
        eprintln!("  [{}] {} ({}/{}){}", status, c.name, c.actual, c.total,
            if c.actual < c.target { " ← needs tuning" } else { "" });
    }

    // Hard asserts on critical invariants — slate is immune to acid/gypsum, must form hornfels not marble
    let acid_count = count_pass(&all_acid, eq0);
    assert!(acid_count == n, "CRITICAL: Acid should NEVER fire on slate ({} fired out of {})", n - acid_count, n);
    let gypsum_count = count_pass(&all_gypsum, eq0);
    assert!(gypsum_count == n, "CRITICAL: Gypsum should NEVER form from slate ({} formed out of {})", n - gypsum_count, n);
    let marble_count = delta_pass(mat_id(Material::Marble), deq0);
    assert!(marble_count == n, "CRITICAL: Marble should NEVER form from slate ({} formed out of {})", n - marble_count, n);
    let hornfels_count = delta_pass(mat_id(Material::Hornfels), dgt0);
    assert!(hornfels_count >= n, "CRITICAL: Hornfels should ALWAYS form from slate near heat ({}/{})", hornfels_count, n);

    if any_fail {
        eprintln!("\n  ⚠ Some non-critical processes underperforming — may need config tuning");
    }
}
