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

// ─── Helper: Print material census ────────────────────────────────────────

pub(crate) fn print_material_census(label: &str, counts: &MatMap<u32>) {
    let total: u32 = counts.values().sum();
    eprintln!("\n  {}", label);
    eprintln!("  Total solid voxels: {}", total);
    for (&mid, &count) in counts {
        if count > 0 {
            let pct = count as f64 / total as f64 * 100.0;
            eprintln!("    {:<14} {:>8} ({:>5.1}%)", mat_name(mid), count, pct);
        }
    }
}

/// Count water and lava cells in a FluidSnapshot.
pub(crate) fn count_fluid_cells(fluid: &FluidSnapshot) -> (u32, u32) {
    let mut water = 0u32;
    let mut lava = 0u32;
    for cells in fluid.chunks.values() {
        for cell in cells {
            if cell.level > 0.01 {
                match cell.fluid_type {
                    FluidType::Lava => lava += 1,
                    _ => water += 1,
                }
            }
        }
    }
    (water, lava)
}

/// Average a BTreeMap of u32 counts across multiple samples.
pub(crate) fn average_mat_maps(maps: &[MatMap<u32>]) -> MatMap<f64> {
    let mut result: MatMap<f64> = BTreeMap::new();
    let n = maps.len() as f64;
    for map in maps {
        for (&mid, &count) in map {
            *result.entry(mid).or_insert(0.0) += count as f64;
        }
    }
    for val in result.values_mut() {
        *val /= n;
    }
    result
}

// ─── Helper: Per-mineral source system + expected level ───────────────────

pub(crate) fn material_source(mid: u8) -> &'static str {
    match mid {
        1  => "consumed",      // Sandstone
        2  => "Acid-/Meta-",   // Limestone consumed by acid + metamorphism
        3  => "Meta",          // Granite (from sandstone metamorphism)
        4  => "Meta",          // Basalt (crust / consumed)
        5  => "Meta-",         // Slate consumed by metamorphism
        6  => "Meta",          // Marble
        7  => "Veins+Enr",     // Iron
        8  => "Veins+Enr",     // Copper
        9  => "CuOx",          // Malachite
        10 => "Veins",         // Tin
        11 => "Veins+Enr",     // Gold
        14 => "Veins",         // Sulfide
        15 => "Veins+Sil",     // Quartz (veins + silicification)
        16 => "Veins",         // Pyrite
        19 => "Meta-",         // Coal consumed (→ Graphite)
        20 => "Meta",          // Graphite (coal maturation)
        22 => "Meta",          // Hornfels
        23 => "Meta",          // Garnet (skarn)
        24 => "Meta",          // Diopside (skarn)
        25 => "Acid",          // Gypsum
        _  => "",
    }
}

pub(crate) fn material_expected_limestone(mid: u8) -> &'static str {
    match mid {
        1  => "boundary",   // Sandstone at depth boundary
        2  => "HOST",       // Limestone is the host rock
        3  => "",           // Granite
        4  => "low",        // Basalt
        5  => "boundary",   // Slate at depth boundary
        6  => "vhigh",      // Marble (100% meta magnitude)
        7  => "high",       // Iron (skarn + MVT veins)
        8  => "high",       // Copper (Cu-skarn, veins)
        9  => "high",       // Malachite (best carbonate buffering)
        10 => "low",        // Tin (Sn-skarn, less common)
        11 => "med",        // Gold (Au-skarn, veins)
        14 => "high",       // Sulfide (veins)
        15 => "med-high",   // Quartz (silicification 100% + veins)
        16 => "high",       // Pyrite (veins + gangue)
        19 => "",           // Coal (not in limestone depth)
        20 => "",           // Graphite
        22 => "none",       // Hornfels (wrong host)
        23 => "high",       // Garnet (skarn inner zone)
        24 => "high",       // Diopside (skarn mid zone)
        25 => "vhigh",      // Gypsum (acid dissolution, 35-40% of change)
        _  => "",
    }
}

pub(crate) fn material_expected_slate(mid: u8) -> &'static str {
    match mid {
        1  => "boundary",   // Sandstone at depth boundary
        2  => "",           // Limestone (not in slate zone)
        3  => "",           // Granite
        4  => "low",        // Basalt (consumed by meta)
        5  => "HOST",       // Slate is the host rock
        6  => "none",       // Marble (wrong host! slate→hornfels)
        7  => "high",       // Iron (veins, 110% magnitude)
        8  => "high",       // Copper (cassiterite-chalcopyrite veins)
        9  => "low-med",    // Malachite (Cu oxidation at 50%)
        10 => "high",       // Tin (slate is a top Sn host, 30% in hypothermal)
        11 => "high",       // Gold (Bendigo-type, world-class!)
        14 => "high",       // Sulfide (veins)
        15 => "med",        // Quartz (veins + silic at 50%)
        16 => "med-high",   // Pyrite (veins, on cleavage planes)
        19 => "present",    // Coal (in world gen, consumed → graphite)
        20 => "med",        // Graphite (coal maturation near heat)
        22 => "vhigh",      // Hornfels (THE signature product!)
        23 => "none-low",   // Garnet (almandine possible in inner zone)
        24 => "none",       // Diopside (wrong host)
        25 => "none",       // Gypsum (acid immune!)
        _  => "",
    }
}

/// Print the full realistic bench report (shared by all 4 realistic tests).
pub(crate) fn print_realistic_report(
    rock_type: &str,
    iterations: u32,
    cycles: u32,
    all_acid: &[u32],
    all_sulfide_acid: &[u32],
    all_gypsum: &[u32],
    all_metamorphosed: &[u32],
    all_formations: &[u32],
    all_silicified: &[u32],
    all_veins: &[u32],
    all_enriched: &[u32],
    all_deltas: &[MatMap<i64>],
    all_befores: &[MatMap<u32>],
    all_water_cells: &[u32],
    all_lava_cells: &[u32],
) {
    let n = iterations as usize;
    let expected_fn: fn(u8) -> &'static str = match rock_type {
        "Limestone" => material_expected_limestone,
        "Slate" => material_expected_slate,
        _ => material_expected_limestone,
    };

    // ── Fluid Summary ──
    let avg_water: f64 = all_water_cells.iter().sum::<u32>() as f64 / n as f64;
    let avg_lava: f64 = all_lava_cells.iter().sum::<u32>() as f64 / n as f64;
    eprintln!("\n--- Fluid Cells (avg over {} iterations) ---", n);
    eprintln!("  Water: avg={:.0} min={} max={}", avg_water,
        all_water_cells.iter().min().unwrap(), all_water_cells.iter().max().unwrap());
    eprintln!("  Lava:  avg={:.0} min={} max={}", avg_lava,
        all_lava_cells.iter().min().unwrap(), all_lava_cells.iter().max().unwrap());

    // ── Process Counters ──
    eprintln!("\n--- Process Counters (avg over {} iterations x {} cycles) ---", n, cycles);
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
    print_avg("acid_dissolved:", all_acid);
    print_avg("sulfide_acid:", all_sulfide_acid);
    print_avg("gypsum_deposited:", all_gypsum);
    print_avg("metamorphosed:", all_metamorphosed);
    print_avg("formations_grown:", all_formations);
    print_avg("silicified:", all_silicified);
    print_avg("veins_deposited:", all_veins);
    print_avg("enriched:", all_enriched);

    // ── System Contribution Breakdown ──
    let total_acid: f64 = all_acid.iter().sum::<u32>() as f64 / n as f64;
    let total_meta: f64 = all_metamorphosed.iter().sum::<u32>() as f64 / n as f64;
    let total_veins: f64 = all_veins.iter().sum::<u32>() as f64 / n as f64;
    let total_silic: f64 = all_silicified.iter().sum::<u32>() as f64 / n as f64;
    let total_form: f64 = all_formations.iter().sum::<u32>() as f64 / n as f64;
    let total_enrich: f64 = all_enriched.iter().sum::<u32>() as f64 / n as f64;
    let total_gypsum: f64 = all_gypsum.iter().sum::<u32>() as f64 / n as f64;
    let grand_total = total_acid + total_meta + total_veins + total_silic + total_form + total_enrich;
    let pct = |v: f64| if grand_total > 0.0 { v / grand_total * 100.0 } else { 0.0 };

    eprintln!("\n--- System Contribution (% of total voxel changes, avg) ---");
    eprintln!("  {:<26} {:>8} {:>8}", "System", "Avg", "% Total");
    eprintln!("  {:-<44}", "");
    let systems: &[(&str, f64)] = &[
        ("Acid dissolution", total_acid),
        ("Contact metamorphism", total_meta),
        ("Hydrothermal veins", total_veins),
        ("Silicification", total_silic),
        ("Formations (speleothems)", total_form),
        ("Supergene enrichment", total_enrich),
    ];
    for &(name, val) in systems {
        if val > 0.1 {
            eprintln!("  {:<26} {:>8.0} {:>7.1}%", name, val, pct(val));
        }
    }
    eprintln!("  {:<26} {:>8.0} {:>7}",   "TOTAL", grand_total, "100%");
    if total_gypsum > 0.1 {
        eprintln!("  (Gypsum deposited: {:.0} — byproduct of acid dissolution)", total_gypsum);
    }

    // ── World Gen vs Sleep Production (with Source + Expected columns) ──
    let avg_before = average_mat_maps(all_befores);
    let avg_total: f64 = avg_before.values().sum();

    eprintln!("\n--- World Gen vs Sleep Production (avg over {} iterations) ---", n);
    eprintln!("  {:<14} {:>10} {:>8} {:>12} {:>10} {:>10} {:>8}",
        "Material", "Gen Avg", "Gen %", "Sleep Delta", "% of Gen", "Source", "Expected");
    eprintln!("  {:-<80}", "");
    for &mid in &ALL_MAT_IDS {
        let gen_avg = *avg_before.get(&mid).unwrap_or(&0.0);
        let delta_vals: Vec<f64> = all_deltas.iter()
            .map(|d| *d.get(&mid).unwrap_or(&0) as f64).collect();
        let delta_avg = delta_vals.iter().sum::<f64>() / delta_vals.len() as f64;
        let src = material_source(mid);
        let exp = expected_fn(mid);
        if gen_avg > 0.1 || delta_avg.abs() > 0.1 {
            let gen_pct = gen_avg / avg_total * 100.0;
            let sleep_pct = if gen_avg > 0.1 { delta_avg / gen_avg * 100.0 } else { f64::INFINITY };
            if sleep_pct.is_finite() {
                eprintln!("  {:<14} {:>10.0} {:>7.2}% {:>+12.1} {:>+9.1}% {:>10} {:>8}",
                    mat_name(mid), gen_avg, gen_pct, delta_avg, sleep_pct, src, exp);
            } else {
                eprintln!("  {:<14} {:>10.0} {:>7.2}% {:>+12.1} {:>10} {:>10} {:>8}",
                    mat_name(mid), gen_avg, gen_pct, delta_avg, "NEW", src, exp);
            }
        }
    }

    // ── Process Activity ──
    eprintln!("\n--- Process Activity ({} iterations) ---", n);
    let count_active = |v: &[u32]| v.iter().filter(|&&x| x > 0).count();
    let delta_active = |mid: u8| all_deltas.iter().filter(|d| *d.get(&mid).unwrap_or(&0) > 0).count();

    let processes = [
        ("Acid dissolution",      count_active(all_acid)),
        ("Sulfide acid",           count_active(all_sulfide_acid)),
        ("Gypsum deposition",     count_active(all_gypsum)),
        ("Contact metamorphism",  count_active(all_metamorphosed)),
        ("Formations grown",      count_active(all_formations)),
        ("Silicification",        count_active(all_silicified)),
        ("Vein deposition",       count_active(all_veins)),
        ("Supergene enrichment",  count_active(all_enriched)),
    ];
    for (name, active) in &processes {
        eprintln!("  {:<24} {}/{}", name, active, n);
    }

    let minerals = [
        ("Marble",    mat_id(Material::Marble)),
        ("Garnet",    mat_id(Material::Garnet)),
        ("Diopside",  mat_id(Material::Diopside)),
        ("Hornfels",  mat_id(Material::Hornfels)),
        ("Gypsum",    mat_id(Material::Gypsum)),
        ("Malachite", mat_id(Material::Malachite)),
        ("Pyrite",    mat_id(Material::Pyrite)),
        ("Iron",      mat_id(Material::Iron)),
        ("Copper",    mat_id(Material::Copper)),
        ("Gold",      mat_id(Material::Gold)),
        ("Quartz",    mat_id(Material::Quartz)),
        ("Sulfide",   mat_id(Material::Sulfide)),
        ("Tin",       mat_id(Material::Tin)),
        ("Basalt",    mat_id(Material::Basalt)),
    ];
    eprintln!("\n  Mineral production (iters with delta > 0):");
    for (name, mid) in &minerals {
        let active = delta_active(*mid);
        if active > 0 {
            eprintln!("    {:<14} {}/{}", name, active, n);
        }
    }

    // ── Geology Reference (from geological-realism-per-rock.md) ──
    eprintln!("\n--- Geology Reference: {} (expected system contributions) ---", rock_type);
    match rock_type {
        "Limestone" => {
            eprintln!("  Sulfuric acid speleogenesis:  ~35-40%  (100% magnitude — THE dominant cave process)");
            eprintln!("  Contact metamorphism (skarn): ~20-25%  (100% — marble + garnet + diopside)");
            eprintln!("  Karst dissolution:            ~15-20%  (100% — textbook karst rock)");
            eprintln!("  Hydrothermal veins (MVT):     ~10-15%  (100% — carbonate buffering precipitates metals)");
            eprintln!("  Silicification:                ~5-10%  (100% — silica replaces CaCO3)");
            eprintln!("  Speleothems/flowstone:         ~3-5%   (100% — 300+ formation varieties)");
            eprintln!("  Cu oxidation:                  ~3-5%   (100% — best malachite/azurite formation)");
            eprintln!("  Supergene enrichment:          ~1-3%   (100% — high pH precipitates metals)");
        },
        "Slate" => {
            eprintln!("  Contact metamorphism:         ~35-45%  (90% magnitude — spotted slate -> hornfels)");
            eprintln!("  Hydrothermal veins:           ~25-35%  (110% — Bendigo gold! world-class deposits)");
            eprintln!("  Supergene enrichment:         ~10-15%  (80% — slate acts as TRAP layer)");
            eprintln!("  Cu oxidation:                  ~5-10%  (50% — where Cu exists in veins)");
            eprintln!("  Silicification:                ~3-5%   (50% — along fractures only)");
            eprintln!("  Pyrite growth on cleavage:     ~3-5%   (moderate — visually striking cubes)");
            eprintln!("  Sulfuric acid:                 ~0-2%   (2% — nearly IMMUNE, clay+quartz resists)");
            eprintln!("  Flowstone/speleothems:         ~0-1%   (5% — only imported calcite)");
            eprintln!("  Karst dissolution:              ~0%    (<1% — insoluble in water)");
        },
        _ => {},
    }

    eprintln!("\n  Report-only mode. Use these numbers to set targets.");
}
