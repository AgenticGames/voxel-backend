use super::*;

use std::collections::BTreeMap;
use voxel_core::material::Material;

use crate::execute_sleep;
use crate::util::{sleep_vein_size, default_vein_size};

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
