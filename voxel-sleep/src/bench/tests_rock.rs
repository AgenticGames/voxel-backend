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

// ═══════════════════════════════════════════════════════════════════════════
// Test 7: Limestone realistic (10 iterations × 3 cycles, UE5 world gen)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn bench_limestone_realistic() {
    const ITERATIONS: u32 = 10;
    const CYCLES: u32 = 3;
    // Limestone zone: Y=130..250 (UE5 depths: sandstone=250, slate=130).
    // Midpoint=190. chunk_y=12 → world Y=192..208. Radius 4: Y=128..256.
    const CENTER: (i32, i32, i32) = (1, 12, 1);
    const RADIUS: i32 = 4;

    eprintln!("\n{:=<80}", "= LIMESTONE REALISTIC (10 iters × 3 cycles, radius 4, UE5 gen) ");
    eprintln!("  Center chunk: {:?}, Radius: {}, Grid: {}×{}×{} = {} chunks",
        CENTER, RADIUS, RADIUS * 2 + 1, RADIUS * 2 + 1, RADIUS * 2 + 1,
        (RADIUS * 2 + 1).pow(3));
    eprintln!("  World Y: {}..{}", (CENTER.1 - RADIUS) * 16, (CENTER.1 + RADIUS + 1) * 16);

    let mut all_acid: Vec<u32> = Vec::new();
    let mut all_sulfide: Vec<u32> = Vec::new();
    let mut all_gypsum: Vec<u32> = Vec::new();
    let mut all_formations: Vec<u32> = Vec::new();
    let mut all_silicified: Vec<u32> = Vec::new();
    let mut all_veins: Vec<u32> = Vec::new();
    let mut all_enriched: Vec<u32> = Vec::new();
    let mut all_metamorphosed: Vec<u32> = Vec::new();
    let mut all_deltas: Vec<MatMap<i64>> = Vec::new();
    let mut all_befores: Vec<MatMap<u32>> = Vec::new();
    let mut all_water_cells: Vec<u32> = Vec::new();
    let mut all_lava_cells: Vec<u32> = Vec::new();

    for iter in 0..ITERATIONS {
        let seed = 42 + iter as u64;
        let gen_config = make_ue_gen_config(seed);
        eprintln!("\n  [iter {:>2}] seed={}", iter, seed);

        let (mut density, mut stress, mut support, mut fluid) =
            make_realistic_world_at(&gen_config, CENTER, RADIUS);

        // Add 4 water + 4 lava patches (10 cells each, lava ≥8 Manhattan from water)
        let (wp, lp) = inject_fluid_patches(&density, &mut fluid, 4, 10, 8, seed * 1000 + 7);

        // Count total fluid cells (gen + injected)
        let (total_water, total_lava) = count_fluid_cells(&fluid);
        all_water_cells.push(total_water);
        all_lava_cells.push(total_lava);
        eprintln!("  Fluid: {} water + {} lava (injected {} + {})", total_water, total_lava, wp, lp);

        let before = count_materials(&density);
        if iter == 0 {
            print_material_census("Before Census (iter 0):", &before);
        }
        all_befores.push(before.clone());

        // Nest/corpse positions inside the chunk volume
        let wy_base = (CENTER.1 - 1) * 16;
        let mut cfg = make_ue_config();
        cfg.nest_positions = vec![
            (8, wy_base + 8, 8), (24, wy_base + 20, 24), (40, wy_base + 32, 12),
            (16, wy_base + 24, 32), (32, wy_base + 16, 20),
        ];
        cfg.corpse_positions = vec![
            (10, wy_base + 8, 10), (12, wy_base + 20, 22), (38, wy_base + 32, 14),
            (20, wy_base + 24, 30), (28, wy_base + 16, 18),
            (6, wy_base + 12, 6), (26, wy_base + 22, 26), (36, wy_base + 28, 16),
            (14, wy_base + 26, 28), (30, wy_base + 18, 22),
        ];

        let mut total_acid = 0u32;
        let mut total_sulfide = 0u32;
        let mut total_gypsum = 0u32;
        let mut total_formations = 0u32;
        let mut total_silicified = 0u32;
        let mut total_veins = 0u32;
        let mut total_enriched = 0u32;
        let mut total_metamorphosed = 0u32;

        for cycle in 0..CYCLES {
            cfg.sleep_count = cycle + 1;
            let result = execute_sleep(
                &cfg, &mut density, &mut stress, &mut support,
                &mut fluid, CENTER, iter * CYCLES + cycle, None,
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

        eprintln!("    acid={:<4} sulfide={:<4} gypsum={:<4} meta={:<4} form={:<4} silic={:<4} veins={:<4} enrich={:<4}",
            total_acid, total_sulfide, total_gypsum, total_metamorphosed,
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

    print_realistic_report(
        "Limestone", ITERATIONS, CYCLES,
        &all_acid, &all_sulfide, &all_gypsum, &all_metamorphosed,
        &all_formations, &all_silicified, &all_veins, &all_enriched,
        &all_deltas, &all_befores, &all_water_cells, &all_lava_cells,
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 8: Slate realistic (10 iterations × 3 cycles, UE5 world gen)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn bench_slate_realistic() {
    const ITERATIONS: u32 = 10;
    const CYCLES: u32 = 3;
    // Slate zone: Y=-100..130 (UE5 depths: slate=130, granite=-100).
    // Midpoint=15. chunk_y=1. Radius 4: Y=-48..80.
    const CENTER: (i32, i32, i32) = (1, 1, 1);
    const RADIUS: i32 = 4;

    eprintln!("\n{:=<80}", "= SLATE REALISTIC (10 iters × 3 cycles, radius 4, UE5 gen) ");
    eprintln!("  Center chunk: {:?}, Radius: {}, Grid: {}×{}×{} = {} chunks",
        CENTER, RADIUS, RADIUS * 2 + 1, RADIUS * 2 + 1, RADIUS * 2 + 1,
        (RADIUS * 2 + 1).pow(3));
    eprintln!("  World Y: {}..{}", (CENTER.1 - RADIUS) * 16, (CENTER.1 + RADIUS + 1) * 16);

    let mut all_acid: Vec<u32> = Vec::new();
    let mut all_sulfide_acid: Vec<u32> = Vec::new();
    let mut all_gypsum: Vec<u32> = Vec::new();
    let mut all_formations: Vec<u32> = Vec::new();
    let mut all_silicified: Vec<u32> = Vec::new();
    let mut all_veins: Vec<u32> = Vec::new();
    let mut all_enriched: Vec<u32> = Vec::new();
    let mut all_metamorphosed: Vec<u32> = Vec::new();
    let mut all_deltas: Vec<MatMap<i64>> = Vec::new();
    let mut all_befores: Vec<MatMap<u32>> = Vec::new();
    let mut all_water_cells: Vec<u32> = Vec::new();
    let mut all_lava_cells: Vec<u32> = Vec::new();

    for iter in 0..ITERATIONS {
        let seed = 42 + iter as u64;
        let gen_config = make_ue_gen_config(seed);
        eprintln!("\n  [iter {:>2}] seed={}", iter, seed);

        let (mut density, mut stress, mut support, mut fluid) =
            make_realistic_world_at(&gen_config, CENTER, RADIUS);

        // Add 4 water + 4 lava patches (10 cells each, lava ≥8 Manhattan from water)
        let (wp, lp) = inject_fluid_patches(&density, &mut fluid, 4, 10, 8, seed * 1000 + 7);

        // Count total fluid cells (gen + injected)
        let (total_water, total_lava) = count_fluid_cells(&fluid);
        all_water_cells.push(total_water);
        all_lava_cells.push(total_lava);
        eprintln!("  Fluid: {} water + {} lava (injected {} + {})", total_water, total_lava, wp, lp);

        let before = count_materials(&density);
        if iter == 0 {
            print_material_census("Before Census (iter 0):", &before);
        }
        all_befores.push(before.clone());

        // Nest/corpse positions inside the chunk volume
        let wy_base = (CENTER.1 - 1) * 16;
        let mut cfg = make_ue_config();
        cfg.nest_positions = vec![
            (8, wy_base + 8, 8), (24, wy_base + 20, 24), (40, wy_base + 32, 12),
            (16, wy_base + 24, 32), (32, wy_base + 16, 20),
        ];
        cfg.corpse_positions = vec![
            (10, wy_base + 8, 10), (12, wy_base + 20, 22), (38, wy_base + 32, 14),
            (20, wy_base + 24, 30), (28, wy_base + 16, 18),
            (6, wy_base + 12, 6), (26, wy_base + 22, 26), (36, wy_base + 28, 16),
            (14, wy_base + 26, 28), (30, wy_base + 18, 22),
        ];

        let mut total_acid = 0u32;
        let mut total_sulfide_acid = 0u32;
        let mut total_gypsum = 0u32;
        let mut total_formations = 0u32;
        let mut total_silicified = 0u32;
        let mut total_veins = 0u32;
        let mut total_enriched = 0u32;
        let mut total_metamorphosed = 0u32;

        for cycle in 0..CYCLES {
            cfg.sleep_count = cycle + 1;
            let result = execute_sleep(
                &cfg, &mut density, &mut stress, &mut support,
                &mut fluid, CENTER, iter * CYCLES + cycle, None,
            );
            total_acid += result.acid_dissolved;
            total_sulfide_acid += result.sulfide_dissolved;
            total_gypsum += result.gypsum_deposited;
            total_formations += result.formations_grown;
            total_silicified += result.voxels_silicified;
            total_veins += result.veins_deposited;
            total_enriched += result.voxels_enriched;
            total_metamorphosed += result.voxels_metamorphosed;
        }

        let after = count_materials(&density);
        let delta = material_delta(&before, &after);

        eprintln!("    acid={:<4} sulfide_acid={:<4} gypsum={:<4} meta={:<4} form={:<4} silic={:<4} veins={:<4} enrich={:<4}",
            total_acid, total_sulfide_acid, total_gypsum, total_metamorphosed,
            total_formations, total_silicified, total_veins, total_enriched);

        all_acid.push(total_acid);
        all_sulfide_acid.push(total_sulfide_acid);
        all_gypsum.push(total_gypsum);
        all_formations.push(total_formations);
        all_silicified.push(total_silicified);
        all_veins.push(total_veins);
        all_enriched.push(total_enriched);
        all_metamorphosed.push(total_metamorphosed);
        all_deltas.push(delta);
    }

    print_realistic_report(
        "Slate", ITERATIONS, CYCLES,
        &all_acid, &all_sulfide_acid, &all_gypsum, &all_metamorphosed,
        &all_formations, &all_silicified, &all_veins, &all_enriched,
        &all_deltas, &all_befores, &all_water_cells, &all_lava_cells,
    );
}

// ===================================================================
// Test 9: Limestone realistic SUPER (8x fluid, 2.5x cauldrons)
// ===================================================================

#[test]
#[ignore]
fn bench_limestone_realistic_super() {
    const ITERATIONS: u32 = 10;
    const CYCLES: u32 = 3;
    const CENTER: (i32, i32, i32) = (1, 12, 1);
    const RADIUS: i32 = 4;

    eprintln!("\n{:=<80}", "= LIMESTONE SUPER (8x fluid, 2.5x cauldrons, 10 iters x 3 cycles) ");
    eprintln!("  Center chunk: {:?}, Radius: {}, Grid: {}x{}x{} = {} chunks",
        CENTER, RADIUS, RADIUS * 2 + 1, RADIUS * 2 + 1, RADIUS * 2 + 1,
        (RADIUS * 2 + 1).pow(3));
    eprintln!("  World Y: {}..{}", (CENTER.1 - RADIUS) * 16, (CENTER.1 + RADIUS + 1) * 16);

    let mut all_acid: Vec<u32> = Vec::new();
    let mut all_sulfide: Vec<u32> = Vec::new();
    let mut all_gypsum: Vec<u32> = Vec::new();
    let mut all_formations: Vec<u32> = Vec::new();
    let mut all_silicified: Vec<u32> = Vec::new();
    let mut all_veins: Vec<u32> = Vec::new();
    let mut all_enriched: Vec<u32> = Vec::new();
    let mut all_metamorphosed: Vec<u32> = Vec::new();
    let mut all_deltas: Vec<MatMap<i64>> = Vec::new();
    let mut all_befores: Vec<MatMap<u32>> = Vec::new();
    let mut all_water_cells: Vec<u32> = Vec::new();
    let mut all_lava_cells: Vec<u32> = Vec::new();

    for iter in 0..ITERATIONS {
        let seed = 42 + iter as u64;
        let mut gen_config = make_ue_gen_config(seed);
        gen_config.formations.cauldron_chance = 0.20; // 2.5x (0.08 -> 0.20)
        eprintln!("\n  [iter {:>2}] seed={}", iter, seed);

        let (mut density, mut stress, mut support, mut fluid) =
            make_realistic_world_at(&gen_config, CENTER, RADIUS);

        // 8x fluid: 8 patches of 40 cells each (vs 4x10 normal)
        let (wp, lp) = inject_fluid_patches(&density, &mut fluid, 8, 40, 8, seed * 1000 + 7);

        let (total_water, total_lava) = count_fluid_cells(&fluid);
        all_water_cells.push(total_water);
        all_lava_cells.push(total_lava);
        eprintln!("  Fluid: {} water + {} lava (injected {} + {})", total_water, total_lava, wp, lp);

        let before = count_materials(&density);
        if iter == 0 {
            print_material_census("Before Census (iter 0):", &before);
        }
        all_befores.push(before.clone());

        let wy_base = (CENTER.1 - 1) * 16;
        let mut cfg = make_ue_config();
        cfg.nest_positions = vec![
            (8, wy_base + 8, 8), (24, wy_base + 20, 24), (40, wy_base + 32, 12),
            (16, wy_base + 24, 32), (32, wy_base + 16, 20),
        ];
        cfg.corpse_positions = vec![
            (10, wy_base + 8, 10), (12, wy_base + 20, 22), (38, wy_base + 32, 14),
            (20, wy_base + 24, 30), (28, wy_base + 16, 18),
            (6, wy_base + 12, 6), (26, wy_base + 22, 26), (36, wy_base + 28, 16),
            (14, wy_base + 26, 28), (30, wy_base + 18, 22),
        ];

        let mut total_acid = 0u32;
        let mut total_sulfide = 0u32;
        let mut total_gypsum = 0u32;
        let mut total_formations = 0u32;
        let mut total_silicified = 0u32;
        let mut total_veins = 0u32;
        let mut total_enriched = 0u32;
        let mut total_metamorphosed = 0u32;

        for cycle in 0..CYCLES {
            cfg.sleep_count = cycle + 1;
            let result = execute_sleep(
                &cfg, &mut density, &mut stress, &mut support,
                &mut fluid, CENTER, iter * CYCLES + cycle, None,
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

        eprintln!("    acid={:<5} sulfide={:<4} gypsum={:<5} meta={:<5} form={:<5} silic={:<5} veins={:<5} enrich={:<5}",
            total_acid, total_sulfide, total_gypsum, total_metamorphosed,
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

    print_realistic_report(
        "Limestone", ITERATIONS, CYCLES,
        &all_acid, &all_sulfide, &all_gypsum, &all_metamorphosed,
        &all_formations, &all_silicified, &all_veins, &all_enriched,
        &all_deltas, &all_befores, &all_water_cells, &all_lava_cells,
    );
}

// ===================================================================
// Test 10: Slate realistic SUPER (8x fluid, 2.5x cauldrons)
// ===================================================================

#[test]
#[ignore]
fn bench_slate_realistic_super() {
    const ITERATIONS: u32 = 10;
    const CYCLES: u32 = 3;
    const CENTER: (i32, i32, i32) = (1, 1, 1);
    const RADIUS: i32 = 4;

    eprintln!("\n{:=<80}", "= SLATE SUPER (8x fluid, 2.5x cauldrons, 10 iters x 3 cycles) ");
    eprintln!("  Center chunk: {:?}, Radius: {}, Grid: {}x{}x{} = {} chunks",
        CENTER, RADIUS, RADIUS * 2 + 1, RADIUS * 2 + 1, RADIUS * 2 + 1,
        (RADIUS * 2 + 1).pow(3));
    eprintln!("  World Y: {}..{}", (CENTER.1 - RADIUS) * 16, (CENTER.1 + RADIUS + 1) * 16);

    let mut all_acid: Vec<u32> = Vec::new();
    let mut all_sulfide_acid: Vec<u32> = Vec::new();
    let mut all_gypsum: Vec<u32> = Vec::new();
    let mut all_formations: Vec<u32> = Vec::new();
    let mut all_silicified: Vec<u32> = Vec::new();
    let mut all_veins: Vec<u32> = Vec::new();
    let mut all_enriched: Vec<u32> = Vec::new();
    let mut all_metamorphosed: Vec<u32> = Vec::new();
    let mut all_deltas: Vec<MatMap<i64>> = Vec::new();
    let mut all_befores: Vec<MatMap<u32>> = Vec::new();
    let mut all_water_cells: Vec<u32> = Vec::new();
    let mut all_lava_cells: Vec<u32> = Vec::new();

    for iter in 0..ITERATIONS {
        let seed = 42 + iter as u64;
        let mut gen_config = make_ue_gen_config(seed);
        gen_config.formations.cauldron_chance = 0.20; // 2.5x (0.08 -> 0.20)
        eprintln!("\n  [iter {:>2}] seed={}", iter, seed);

        let (mut density, mut stress, mut support, mut fluid) =
            make_realistic_world_at(&gen_config, CENTER, RADIUS);

        // 8x fluid: 8 patches of 40 cells each (vs 4x10 normal)
        let (wp, lp) = inject_fluid_patches(&density, &mut fluid, 8, 40, 8, seed * 1000 + 7);

        let (total_water, total_lava) = count_fluid_cells(&fluid);
        all_water_cells.push(total_water);
        all_lava_cells.push(total_lava);
        eprintln!("  Fluid: {} water + {} lava (injected {} + {})", total_water, total_lava, wp, lp);

        let before = count_materials(&density);
        if iter == 0 {
            print_material_census("Before Census (iter 0):", &before);
        }
        all_befores.push(before.clone());

        let wy_base = (CENTER.1 - 1) * 16;
        let mut cfg = make_ue_config();
        cfg.nest_positions = vec![
            (8, wy_base + 8, 8), (24, wy_base + 20, 24), (40, wy_base + 32, 12),
            (16, wy_base + 24, 32), (32, wy_base + 16, 20),
        ];
        cfg.corpse_positions = vec![
            (10, wy_base + 8, 10), (12, wy_base + 20, 22), (38, wy_base + 32, 14),
            (20, wy_base + 24, 30), (28, wy_base + 16, 18),
            (6, wy_base + 12, 6), (26, wy_base + 22, 26), (36, wy_base + 28, 16),
            (14, wy_base + 26, 28), (30, wy_base + 18, 22),
        ];

        let mut total_acid = 0u32;
        let mut total_sulfide_acid = 0u32;
        let mut total_gypsum = 0u32;
        let mut total_formations = 0u32;
        let mut total_silicified = 0u32;
        let mut total_veins = 0u32;
        let mut total_enriched = 0u32;
        let mut total_metamorphosed = 0u32;

        for cycle in 0..CYCLES {
            cfg.sleep_count = cycle + 1;
            let result = execute_sleep(
                &cfg, &mut density, &mut stress, &mut support,
                &mut fluid, CENTER, iter * CYCLES + cycle, None,
            );
            total_acid += result.acid_dissolved;
            total_sulfide_acid += result.sulfide_dissolved;
            total_gypsum += result.gypsum_deposited;
            total_formations += result.formations_grown;
            total_silicified += result.voxels_silicified;
            total_veins += result.veins_deposited;
            total_enriched += result.voxels_enriched;
            total_metamorphosed += result.voxels_metamorphosed;
        }

        let after = count_materials(&density);
        let delta = material_delta(&before, &after);

        eprintln!("    acid={:<5} sulfide_acid={:<4} gypsum={:<5} meta={:<5} form={:<5} silic={:<5} veins={:<5} enrich={:<5}",
            total_acid, total_sulfide_acid, total_gypsum, total_metamorphosed,
            total_formations, total_silicified, total_veins, total_enriched);

        all_acid.push(total_acid);
        all_sulfide_acid.push(total_sulfide_acid);
        all_gypsum.push(total_gypsum);
        all_formations.push(total_formations);
        all_silicified.push(total_silicified);
        all_veins.push(total_veins);
        all_enriched.push(total_enriched);
        all_metamorphosed.push(total_metamorphosed);
        all_deltas.push(delta);
    }

    print_realistic_report(
        "Slate", ITERATIONS, CYCLES,
        &all_acid, &all_sulfide_acid, &all_gypsum, &all_metamorphosed,
        &all_formations, &all_silicified, &all_veins, &all_enriched,
        &all_deltas, &all_befores, &all_water_cells, &all_lava_cells,
    );
}
