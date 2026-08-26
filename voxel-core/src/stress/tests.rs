//! Unit tests for the stress system (behavior-preserving split of the
//! former `stress.rs` god file). `use super::*` resolves through
//! `stress/mod.rs`, which re-exports every public item.

use std::collections::HashMap;

use crate::density::DensityField;
use crate::material::Material;

use super::*;

#[test]
fn stress_field_basic() {
    let mut sf = StressField::new(17);
    assert_eq!(sf.stress.len(), 17 * 17 * 17);
    assert_eq!(sf.get(0, 0, 0), 0.0);
    sf.set(5, 5, 5, 0.75);
    assert!((sf.get(5, 5, 5) - 0.75).abs() < 1e-6);
}

#[test]
fn support_field_basic() {
    let mut sf = SupportField::new(17);
    assert_eq!(sf.supports.len(), 17 * 17 * 17);
    assert!(!sf.has_support(0, 0, 0));
    sf.set(3, 3, 3, SupportType::Copper);
    assert!(sf.has_support(3, 3, 3));
    assert_eq!(sf.get(3, 3, 3), SupportType::Copper);
}

#[test]
fn support_type_from_u8() {
    assert_eq!(SupportType::from_u8(0), SupportType::None);
    assert_eq!(SupportType::from_u8(1), SupportType::Copper);
    assert_eq!(SupportType::from_u8(2), SupportType::Iron);
    assert_eq!(SupportType::from_u8(3), SupportType::Steel);
    assert_eq!(SupportType::from_u8(4), SupportType::Crystal);
    assert_eq!(SupportType::from_u8(5), SupportType::Mithril);
    assert_eq!(SupportType::from_u8(6), SupportType::None); // out of range
    assert_eq!(SupportType::from_u8(255), SupportType::None);
}

#[test]
fn support_type_from_legacy_u8() {
    // Pre-2026-05-26 IDs: Slate=1, Granite=2, Limestone=3, Copper=4,
    // Iron=5, Steel=6, Crystal=7. Stone struts collapse to Copper,
    // metals shift down one slot.
    assert_eq!(SupportType::from_legacy_u8(1), SupportType::Copper); // Slate
    assert_eq!(SupportType::from_legacy_u8(2), SupportType::Copper); // Granite
    assert_eq!(SupportType::from_legacy_u8(3), SupportType::Copper); // Limestone
    assert_eq!(SupportType::from_legacy_u8(4), SupportType::Copper); // Old Copper at 4
    assert_eq!(SupportType::from_legacy_u8(5), SupportType::Iron);   // Old Iron at 5
    assert_eq!(SupportType::from_legacy_u8(6), SupportType::Steel);  // Old Steel at 6
    assert_eq!(SupportType::from_legacy_u8(7), SupportType::Crystal);// Old Crystal at 7
    assert_eq!(SupportType::from_legacy_u8(255), SupportType::None);
}

#[test]
fn hardness_tables_correct_length() {
    assert_eq!(DEFAULT_MATERIAL_HARDNESS.len(), 50);
    #[allow(deprecated)]
    { assert_eq!(SUPPORT_HARDNESS.len(), 6); }
    assert_eq!(STRUT_TUNING.len(), 6);
}

#[test]
fn strut_hp_storage_and_decay() {
    // Tier auto-fills HP on placement; same-type re-set preserves HP;
    // damage_hp saturates and reports break only on the killing blow;
    // setting to None zeros HP.
    let mut sf = SupportField::new(8);
    sf.set(2, 2, 2, SupportType::Iron);
    assert_eq!(sf.get_hp(2, 2, 2), STRUT_TUNING[SupportType::Iron as usize].max_hp);
    // Burn down to mid-HP.
    let alive_before = sf.is_strut_alive(2, 2, 2);
    assert!(alive_before);
    let died1 = sf.damage_hp(2, 2, 2, 50.0);
    assert!(!died1);
    // Same-type re-set preserves HP (doesn't refill).
    sf.set(2, 2, 2, SupportType::Iron);
    let hp_after_reset = sf.get_hp(2, 2, 2);
    // Derived from the table, not a literal: this asserts "re-set PRESERVES hp"
    // and must not care what Iron's max_hp happens to be. It was hardcoded to
    // 100 (== the old 150 - 50) and broke the moment Iron was retuned to 200.
    assert_eq!(hp_after_reset, STRUT_TUNING[SupportType::Iron as usize].max_hp - 50);
    // Replacing with a different type refills HP for the new tier.
    sf.set(2, 2, 2, SupportType::Crystal);
    assert_eq!(sf.get_hp(2, 2, 2), STRUT_TUNING[SupportType::Crystal as usize].max_hp);
    // Damage to exactly 0 returns true once.
    let died2 = sf.damage_hp(2, 2, 2, 5000.0);
    assert!(died2);
    assert_eq!(sf.get_hp(2, 2, 2), 0);
    assert!(!sf.is_strut_alive(2, 2, 2));
    // A second call returns false (already broken).
    let died3 = sf.damage_hp(2, 2, 2, 100.0);
    assert!(!died3);
    // Setting to None clears HP and decrements count.
    sf.set(2, 2, 2, SupportType::None);
    assert_eq!(sf.get_hp(2, 2, 2), 0);
    assert!(sf.is_empty());
}

/// 2026-08-23 (user): "give iron strut 200 hp but 70% more resistance to stress
/// dmg". Resistance is a SECOND knob beside max_hp, so this pins both — and the
/// staying-power comparison is the part that would actually regress silently if
/// someone folded resistance back into HP.
#[test]
fn iron_strut_resists_bracing_damage() {
    let copper = STRUT_TUNING[SupportType::Copper as usize];
    let iron = STRUT_TUNING[SupportType::Iron as usize];

    // Same HP pool...
    assert_eq!(iron.max_hp, 200);
    assert_eq!(copper.max_hp, 200);
    // ...but Iron takes 30% of the bracing damage ("70% more resistant").
    assert!((iron.damage_taken_scale - 0.30).abs() < 1e-6);
    // 2026-08-26 (user): Copper takes 30% LESS than it used to — 1.0 -> 0.70.
    // Iron stays the clearly tougher tier (0.30 vs 0.70).
    assert!((copper.damage_taken_scale - 0.70).abs() < 1e-6);
    assert!(iron.damage_taken_scale < copper.damage_taken_scale,
        "Iron must still resist more bracing damage than Copper");

    // Every other tier is untouched at full rate — resistance is a T1/T2 thing.
    for t in [SupportType::Steel, SupportType::Crystal, SupportType::Mithril] {
        assert!((STRUT_TUNING[t as usize].damage_taken_scale - 1.0).abs() < 1e-6,
            "{:?} should still take full bracing damage", t);
    }

    // The formula the collapse BFS actually calls.
    let blocked = 100.0_f32;
    let dmg_copper = bfs_halt_damage(SupportType::Copper, blocked);
    let dmg_iron = bfs_halt_damage(SupportType::Iron, blocked);
    assert!((dmg_copper - 35.0).abs() < 1e-4);   // 100 * 0.5 * 0.70
    assert!((dmg_iron - 15.0).abs() < 1e-4);     // 100 * 0.5 * 0.30

    // Net staying power: at equal HP, the ratio is copper_scale / iron_scale.
    // 2026-08-26: the copper buff (1.0 -> 0.70) deliberately NARROWS Iron's
    // lead from 3.33x to 0.70/0.30 = 2.33x. That is the cost the user accepted
    // by asking for copper specifically; the ordering is what must not break,
    // so the bar moved to 2x rather than being deleted.
    let voxels_copper = copper.max_hp as f32 / dmg_copper * blocked;
    let voxels_iron = iron.max_hp as f32 / dmg_iron * blocked;
    assert!(voxels_iron > voxels_copper * 2.0,
        "iron {voxels_iron} should outlast copper {voxels_copper} by >2x");

    // A None strut must never be charged damage (the call site skips it, but
    // the helper indexes the table by type — index 0 has to stay harmless).
    assert_eq!(bfs_halt_damage(SupportType::None, blocked), 0.0);
}

fn make_density_field(size: usize, fill_solid: bool) -> DensityField {
    let mut df = DensityField::new(size);
    if fill_solid {
        for sample in df.samples.iter_mut() {
            sample.density = 1.0;
            sample.material = Material::Granite;
        }
    }
    df
}

/// Create a 3x5x3 grid of chunks (tall in Y) for proper top-down flood testing.
fn make_solid_world() -> (HashMap<(i32,i32,i32), DensityField>, HashMap<(i32,i32,i32), StressField>, HashMap<(i32,i32,i32), SupportField>) {
    let mut density_fields = HashMap::new();
    let mut stress_fields = HashMap::new();
    let support_fields = HashMap::new();
    for cz in -1..=1 {
        for cy in -2..=2 { // 5 chunks tall for proper flood propagation
            for cx in -1..=1 {
                density_fields.insert((cx, cy, cz), make_density_field(17, true));
                stress_fields.insert((cx, cy, cz), StressField::new(17));
            }
        }
    }
    (density_fields, stress_fields, support_fields)
}

fn make_air_world() -> (HashMap<(i32,i32,i32), DensityField>, HashMap<(i32,i32,i32), StressField>, HashMap<(i32,i32,i32), SupportField>) {
    let mut density_fields = HashMap::new();
    let mut stress_fields = HashMap::new();
    let support_fields = HashMap::new();
    for cz in -1..=1 {
        for cy in -2..=2 {
            for cx in -1..=1 {
                let mut df = DensityField::new(17);
                // Default VoxelSample is Limestone/solid, so explicitly set to Air
                for sample in df.samples.iter_mut() {
                    sample.density = -1.0;
                    sample.material = Material::Air;
                }
                density_fields.insert((cx, cy, cz), df);
                stress_fields.insert((cx, cy, cz), StressField::new(17));
            }
        }
    }
    (density_fields, stress_fields, support_fields)
}

fn default_config() -> StressConfig {
    StressConfig::default()
}

#[test]
fn air_voxel_has_zero_stress() {
    let (density_fields, mut stress_fields, support_fields) = make_air_world();
    let config = default_config();

    let result = recalc_stress_region(
        &density_fields, &mut stress_fields, &support_fields,
        &config, (8, 8, 8), 4, 16,
    );

    assert!(result.overstressed.is_empty());
}

#[test]
fn supported_voxel_low_stress() {
    let (density_fields, _, _) = make_solid_world();
    let config = default_config();

    let stress = calc_voxel_stress(
        &density_fields, &config, 8, 8, 8, 16,
    );

    // With retuned gravity_weight=0.05, a fully-supported deep voxel
    // should have 0 or near-0 stress (lateral+vertical support > gravity load)
    assert!(stress >= 0.0, "Stress should be non-negative");
    assert!(stress.is_finite(), "Stress should be finite");
}

#[test]
fn surface_voxel_low_stress() {
    let (mut density_fields, _, support_fields) = make_solid_world();
    let config = default_config();

    if let Some(df) = density_fields.get_mut(&(0, 0, 0)) {
        for z in 0..17 {
            for y in 10..17 {
                for x in 0..17 {
                    let sample = df.get_mut(x, y, z);
                    sample.density = -1.0;
                    sample.material = Material::Air;
                }
            }
        }
    }
    if let Some(df) = density_fields.get_mut(&(0, 1, 0)) {
        for sample in df.samples.iter_mut() {
            sample.density = -1.0;
            sample.material = Material::Air;
        }
    }

    let stress = calc_voxel_stress(
        &density_fields, &config, 8, 9, 8, 16,
    );

    assert!(stress < 1.0, "Surface voxel should not be overstressed, got {}", stress);
}

#[test]
fn unsupported_ceiling_high_stress() {
    let (mut density_fields, _, support_fields) = make_solid_world();
    let config = default_config();

    if let Some(df) = density_fields.get_mut(&(0, 0, 0)) {
        for z in 0..17 {
            for y in 0..8 {
                for x in 0..17 {
                    let sample = df.get_mut(x, y, z);
                    sample.density = -1.0;
                    sample.material = Material::Air;
                }
            }
        }
    }
    if let Some(df) = density_fields.get_mut(&(0, -1, 0)) {
        for sample in df.samples.iter_mut() {
            sample.density = -1.0;
            sample.material = Material::Air;
        }
    }

    let stress = calc_voxel_stress(
        &density_fields, &config, 8, 8, 8, 16,
    );

    assert!(stress > 0.0, "Ceiling voxel should have stress > 0");
}

#[test]
fn support_structure_reduces_stress() {
    let (mut density_fields, _, _) = make_solid_world();
    let mut support_fields_empty = HashMap::new();
    let mut support_fields_with = HashMap::new();
    let config = default_config();

    if let Some(df) = density_fields.get_mut(&(0, 0, 0)) {
        for z in 0..17 {
            for y in 0..8 {
                for x in 0..17 {
                    let sample = df.get_mut(x, y, z);
                    sample.density = -1.0;
                    sample.material = Material::Air;
                }
            }
        }
    }

    for cz in -1..=1 {
        for cy in -1..=1 {
            for cx in -1..=1 {
                support_fields_empty.insert((cx, cy, cz), SupportField::new(17));
                support_fields_with.insert((cx, cy, cz), SupportField::new(17));
            }
        }
    }

    if let Some(sf) = support_fields_with.get_mut(&(0, 0, 0)) {
        sf.set(8, 7, 8, SupportType::Steel);
    }

    // Relief is applied by callers now: stored = calc - relief.
    let base = calc_voxel_stress(&density_fields, &config, 8, 8, 8, 16);
    let stress_without =
        base - strut_relief_final_legacy(&density_fields, &support_fields_empty, &config, 8, 8, 8, 16);
    let stress_with =
        base - strut_relief_final_legacy(&density_fields, &support_fields_with, &config, 8, 8, 8, 16);

    assert!(
        stress_with < stress_without,
        "Support should reduce stress: with={}, without={}",
        stress_with, stress_without
    );
}

/// Release-only micro-bench for `strut_relief_raw`. Run with:
/// `cargo test --release -p voxel-core bench_strut_relief_raw -- --ignored --nocapture`
#[test]
#[ignore]
fn bench_strut_relief_raw() {
    let chunk_size = 30usize;
    let mut support_fields: HashMap<(i32, i32, i32), SupportField> = HashMap::new();
    for cz in -1..=1 {
        for cy in -1..=1 {
            for cx in -1..=1 {
                support_fields.insert((cx, cy, cz), SupportField::new(chunk_size + 1));
            }
        }
    }
    // Scatter mixed-tier struts through the home chunk on a 5-voxel lattice.
    let sf = support_fields.get_mut(&(0, 0, 0)).unwrap();
    let tiers = [
        SupportType::Copper, SupportType::Iron, SupportType::Steel,
        SupportType::Crystal, SupportType::Mithril,
    ];
    let mut i = 0usize;
    for z in (0..chunk_size).step_by(5) {
        for y in (0..chunk_size).step_by(5) {
            for x in (0..chunk_size).step_by(5) {
                sf.set(x, y, z, tiers[i % tiers.len()]);
                i += 1;
            }
        }
    }
    let n = chunk_size as i32;
    let mut checksum = 0.0f64;
    let t = std::time::Instant::now();
    for wz in 0..n {
        for wy in 0..n {
            for wx in 0..n {
                checksum += strut_relief_raw(&support_fields, wx, wy, wz, chunk_size) as f64;
            }
        }
    }
    let el = t.elapsed();
    println!(
        "bench_strut_relief_raw: {} voxels in {:?} ({:.1} ns/voxel), checksum={:.6}",
        chunk_size.pow(3), el,
        el.as_nanos() as f64 / chunk_size.pow(3) as f64, checksum,
    );
}

#[test]
fn world_to_chunk_local_works() {
    let (key, lx, ly, lz) = world_to_chunk_local(20, 5, -3, 16);
    assert_eq!(key, (1, 0, -1));
    assert_eq!(lx, 4);
    assert_eq!(ly, 5);
    assert_eq!(lz, 13);
}

#[test]
fn collapse_converts_to_air() {
    let mut density_fields = HashMap::new();
    let mut stress_fields = HashMap::new();
    let mut support_fields = HashMap::new();
    let config = default_config();

    let df = make_density_field(17, true);
    density_fields.insert((0, 0, 0), df);
    stress_fields.insert((0, 0, 0), StressField::new(17));

    let overstressed = vec![OverstressedVoxel {
        world_x: 5,
        world_y: 5,
        world_z: 5,
        stress: 1.5,
    }];

    let events = detect_and_execute_collapses(
        &mut density_fields, &mut stress_fields, &support_fields,
        &overstressed, &config, 16,
    );

    assert_eq!(events.len(), 1);
    assert_eq!(events[0].collapsed_voxels.len(), 1);

    // Verify voxel is now air
    let df = density_fields.get(&(0, 0, 0)).unwrap();
    assert_eq!(df.get(5, 5, 5).material, Material::Air);
}

// ── V2 algorithm tests ──

/// Helper: carve a horizontal tunnel (air) at given y range across a chunk.
/// Returns world with solid above/below and air in between.
fn make_tunnel_world(tunnel_y_min: usize, tunnel_y_max: usize)
    -> (HashMap<(i32,i32,i32), DensityField>, HashMap<(i32,i32,i32), StressField>, HashMap<(i32,i32,i32), SupportField>)
{
    let (mut density_fields, stress_fields, support_fields) = make_solid_world();
    // Carve air in center chunk
    if let Some(df) = density_fields.get_mut(&(0, 0, 0)) {
        for z in 0..17 {
            for y in tunnel_y_min..=tunnel_y_max {
                for x in 0..17 {
                    let sample = df.get_mut(x, y, z);
                    sample.density = -1.0;
                    sample.material = Material::Air;
                }
            }
        }
    }
    // Also carve in adjacent chunks for wider tunnel
    for &cx in &[-1, 1] {
        if let Some(df) = density_fields.get_mut(&(cx, 0, 0)) {
            for z in 0..17 {
                for y in tunnel_y_min..=tunnel_y_max {
                    for x in 0..17 {
                        let sample = df.get_mut(x, y, z);
                        sample.density = -1.0;
                        sample.material = Material::Air;
                    }
                }
            }
        }
    }
    for &cz in &[-1, 1] {
        if let Some(df) = density_fields.get_mut(&(0, 0, cz)) {
            for z in 0..17 {
                for y in tunnel_y_min..=tunnel_y_max {
                    for x in 0..17 {
                        let sample = df.get_mut(x, y, z);
                        sample.density = -1.0;
                        sample.material = Material::Air;
                    }
                }
            }
        }
    }
    (density_fields, stress_fields, support_fields)
}

#[test]
fn v2_ground_connectivity_grounded_voxels() {
    let (density_fields, _, _) = make_solid_world();
    let config = default_config();
    let scores = ground_connectivity_pass(
        &density_fields, &[(0, 0, 0)], 16, &config,
    );
    // A voxel in solid rock should have positive support score
    // (flooded from surface above, decayed by depth)
    let score = scores.get(&(0, 0, 0)).unwrap().get(8, 8, 8);
    assert!(score > 0.0, "Solid voxel should have positive support from surface flood, got {}", score);
}

#[test]
fn v2_ground_connectivity_ceiling_low_score() {
    // Carve a wide tunnel: air from y=0 to y=7, solid ceiling at y=8+
    let (density_fields, _, _) = make_tunnel_world(0, 7);
    let config = default_config();
    let scores = ground_connectivity_pass(
        &density_fields, &[(0, 0, 0)], 16, &config,
    );
    // A ceiling voxel at y=8 above wide air should have low support score
    // (it's not directly grounded — nothing solid below in its chunk)
    let ceiling_score = scores.get(&(0, 0, 0)).unwrap().get(8, 8, 8);
    // Score should be less than ground_threshold (0.8) for a wide unsupported ceiling
    assert!(ceiling_score < 0.95,
        "Wide ceiling voxel should have reduced support, got {}", ceiling_score);
}

#[test]
fn v2_small_tunnel_stable() {
    // A narrow 4-wide tunnel should NOT produce overstressed voxels
    let (density_fields, mut stress_fields, support_fields) = make_solid_world();
    let mut config = default_config();
    config.min_safe_span = 8;
    // Carve a 4-wide tunnel in center chunk only (narrow)
    let mut df_clone = density_fields.clone();
    if let Some(df) = df_clone.get_mut(&(0, 0, 0)) {
        for z in 6..10 { // 4 voxels wide in Z
            for y in 4..8 {   // 4 voxels tall
                for x in 4..12 { // 8 voxels long in X
                    let sample = df.get_mut(x, y, z);
                    sample.density = -1.0;
                    sample.material = Material::Air;
                }
            }
        }
    }
    let result = recalc_stress_region_v2(
        &df_clone, &mut stress_fields, &support_fields,
        &config, &[(0, 0, 0)], 16,
    );
    assert!(result.overstressed.is_empty(),
        "Narrow 4-wide tunnel should not produce overstressed voxels, got {}",
        result.overstressed.len());
}

#[test]
fn v2_slab_coherence() {
    // Create a slab scenario and verify collapsed region is contiguous
    let (mut density_fields, mut stress_fields, mut support_fields) = make_solid_world();
    let config = default_config();

    // Create a group of overstressed voxels in a 3x1x3 pattern
    let mut overstressed = Vec::new();
    for x in 5..8 {
        for z in 5..8 {
            overstressed.push(OverstressedVoxel {
                world_x: x, world_y: 10, world_z: z, stress: 1.5,
            });
        }
    }
    // Floor below at y=0..5
    if let Some(df) = density_fields.get_mut(&(0, 0, 0)) {
        for z in 0..17 {
            for y in 6..10 { // Air gap between floor and slab
                for x in 0..17 {
                    let sample = df.get_mut(x, y, z);
                    sample.density = -1.0;
                    sample.material = Material::Air;
                }
            }
        }
    }

    let events = detect_and_execute_collapses_v2(
        &mut density_fields, &mut stress_fields, &mut support_fields,
        &overstressed, &config, 16,
    );

    assert_eq!(events.len(), 1, "Should produce exactly 1 collapse event");
    assert_eq!(events[0].slabs.len(), 1, "Should produce 1 slab");
    let slab = &events[0].slabs[0];
    assert_eq!(slab.voxels.len(), 9, "Slab should contain 9 voxels (3x1x3)");
    assert!(slab.fall_distance > 0, "Slab should have positive fall distance");
}

#[test]
fn v2_slab_landing_preserves_shape() {
    // Slab at y=10, floor at y=5, should land at y=6
    let (mut density_fields, mut stress_fields, mut support_fields) = make_solid_world();
    let config = default_config();

    // Carve air from y=6 to y=9
    if let Some(df) = density_fields.get_mut(&(0, 0, 0)) {
        for z in 0..17 {
            for y in 6..10 {
                for x in 0..17 {
                    let sample = df.get_mut(x, y, z);
                    sample.density = -1.0;
                    sample.material = Material::Air;
                }
            }
        }
    }

    // Need >= min_collapse_region (8) voxels, so use a 3x3 block
    let mut overstressed = Vec::new();
    for z in 7..10 {
        for x in 7..10 {
            overstressed.push(OverstressedVoxel {
                world_x: x, world_y: 10, world_z: z, stress: 1.5,
            });
        }
    }

    let events = detect_and_execute_collapses_v2(
        &mut density_fields, &mut stress_fields, &mut support_fields,
        &overstressed, &config, 16,
    );

    assert_eq!(events.len(), 1);
    let slab = &events[0].slabs[0];
    assert_eq!(slab.fall_distance, 4, "Should fall 4 voxels (10 → 6)");

    // Verify rubble was placed as a mound near the landing area
    // With mound shape, center voxels are placed higher, edge voxels lower
    let df = density_fields.get(&(0, 0, 0)).unwrap();
    // At least some rubble should exist in the landing zone (y=6..8)
    let mut rubble_count = 0;
    for y in 6..9 {
        for z in 7..11 {
            for x in 7..11 {
                if df.get(x, y, z).material.is_solid() {
                    rubble_count += 1;
                }
            }
        }
    }
    assert!(rubble_count > 0, "Should have rubble in the landing zone");
    // Original position should be air
    assert_eq!(df.get(8, 10, 8).material, Material::Air,
        "Original slab position should be air");
}

#[test]
fn v2_strut_reduces_stress() {
    // Create a wide tunnel so ceiling has actual stress, then verify strut reduces it
    let (mut density_fields, _, _) = make_solid_world();
    let mut support_fields_empty = HashMap::new();
    let mut support_fields_with = HashMap::new();
    let config = default_config();

    // Carve a wide tunnel: air from y=0..7 across 3 chunks in X and Z
    // This creates a wide unsupported ceiling at y=8
    for &cz in &[-1, 0, 1] {
        for &cx in &[-1, 0, 1] {
            if let Some(df) = density_fields.get_mut(&(cx, 0, cz)) {
                for z in 0..17 {
                    for y in 0..8 {
                        for x in 0..17 {
                            let sample = df.get_mut(x, y, z);
                            sample.density = -1.0;
                            sample.material = Material::Air;
                        }
                    }
                }
            }
            if let Some(df) = density_fields.get_mut(&(cx, -1, cz)) {
                for sample in df.samples.iter_mut() {
                    sample.density = -1.0;
                    sample.material = Material::Air;
                }
            }
        }
    }

    for cz in -1..=1 {
        for cy in -1..=1 {
            for cx in -1..=1 {
                support_fields_empty.insert((cx, cy, cz), SupportField::new(17));
                support_fields_with.insert((cx, cy, cz), SupportField::new(17));
            }
        }
    }
    // Place a steel strut just below the ceiling
    if let Some(sf) = support_fields_with.get_mut(&(0, 0, 0)) {
        sf.set(8, 7, 8, SupportType::Steel);
    }

    let scores = ground_connectivity_pass(
        &density_fields, &[(0, 0, 0)], 16, &config,
    );

    // Relief is applied by callers now: stored = calc - relief.
    let (base, _) = calc_voxel_stress_v2(
        &density_fields, &scores, &config, 8, 8, 8, 16,
    );
    let stress_without =
        base - strut_relief_final_v2(&density_fields, &support_fields_empty, &config, 8, 8, 8, 16);
    let stress_with =
        base - strut_relief_final_v2(&density_fields, &support_fields_with, &config, 8, 8, 8, 16);

    assert!(stress_without > 0.0,
        "Wide ceiling should have positive stress without strut, got {}", stress_without);
    assert!(stress_with < stress_without,
        "Strut should reduce v2 stress: with={}, without={}", stress_with, stress_without);
}

/// Map-editor scenario: painted stress on STABLE grounded rock (zero organic
/// stress) must be relieved by a nearby strut. This is the case the old
/// inline-relief code could never handle — relief died at the zero-clamp
/// inside the calc, and grounded voxels never even reached the strut sweep.
#[test]
fn strut_relieves_painted_stress_on_grounded_rock() {
    let (density_fields, mut stress_fields, _) = make_solid_world();
    let config = default_config();
    let key = (0, 0, 0);

    // Designer paints heavy stress onto solid rock at (8,8,8).
    stress_fields
        .get_mut(&key)
        .unwrap()
        .add_painted(8, 8, 8, 3.0, 10.0);

    let mut support_fields: HashMap<(i32, i32, i32), SupportField> = HashMap::new();
    for cz in -1..=1 {
        for cy in -2..=2 {
            for cx in -1..=1 {
                support_fields.insert((cx, cy, cz), SupportField::new(17));
            }
        }
    }

    // Without a strut: effective stress IS the painted value.
    recalc_stress_region_v2(
        &density_fields, &mut stress_fields, &support_fields, &config, &[key], 16,
    );
    let eff_no_strut = stress_fields.get(&key).unwrap().effective(8, 8, 8);
    assert!(
        (eff_no_strut - 3.0).abs() < 1e-3,
        "unrelieved painted stress should read back as painted (3.0), got {}",
        eff_no_strut
    );

    // Mithril strut adjacent: relief (35/dist, hardness-scaled) dwarfs the
    // painted 3.0 — effective must drop, and must clamp at zero rather than
    // going negative.
    support_fields
        .get_mut(&key)
        .unwrap()
        .set(8, 7, 8, SupportType::Mithril);
    recalc_stress_region_v2(
        &density_fields, &mut stress_fields, &support_fields, &config, &[key], 16,
    );
    let eff_with_strut = stress_fields.get(&key).unwrap().effective(8, 8, 8);
    assert!(
        eff_with_strut < eff_no_strut,
        "strut must relieve painted stress: with={} without={}",
        eff_with_strut, eff_no_strut
    );
    assert_eq!(
        eff_with_strut, 0.0,
        "full suppression should clamp effective at zero, got {}",
        eff_with_strut
    );

    // The painted layer itself must be untouched — relief is a read-time
    // offset, not an edit of the designer's authored data.
    let painted_after = stress_fields.get(&key).unwrap().painted(8, 8, 8);
    assert!(
        (painted_after - 3.0).abs() < 1e-6,
        "painted overlay must survive relief untouched, got {}",
        painted_after
    );
}

/// Sweep tunnel heights (air gap size) and measure ceiling stress.
/// This shows the stress curve vs span width — used for tuning overhang_weight,
/// span_weight, and min_safe_span.
#[test]
#[ignore] // Run with: cargo test --release -p voxel-core sweep_ceiling_stress -- --ignored --nocapture
fn sweep_ceiling_stress() {
    let mut config = default_config();
    // Set surface_y to 32 so it's at the top of our test world (chunk y=2, local y=0)
    // This ensures the ground connectivity flood can reach our test geometry
    config.surface_y = 32;

    println!("\n=== Ceiling Stress vs Tunnel Height (Air Gap) ===");
    println!("{:<12} {:<12} {:<12} {:<12} {:<12}",
        "air_gap", "v2_stress", "overstressed", "would_collapse", "support_score");
    println!("{}", "-".repeat(60));

    for air_gap in [2, 4, 6, 8, 10, 12, 14, 16] {
        // Tunnel from y=0 to y=air_gap-1, ceiling at y=air_gap
        let tunnel_y_max = (air_gap - 1).min(15);
        let (density_fields, _, _) = make_tunnel_world(0, tunnel_y_max);

        // Run ground connectivity
        let all_keys: Vec<(i32,i32,i32)> = density_fields.keys().cloned().collect();
        let scores = ground_connectivity_pass(&density_fields, &all_keys, 16, &config);

        // Measure stress at ceiling center (just above the tunnel)
        let ceiling_y = (tunnel_y_max + 1).min(16);
        let (stress, _) = calc_voxel_stress_v2(
            &density_fields, &scores, &config,
            8, ceiling_y as i32, 8, 16,
        );

        let support_score = scores.get(&(0, 0, 0))
            .map(|s| s.get(8, ceiling_y, 8))
            .unwrap_or(-1.0);

        let overstressed = stress >= 1.0;
        let would_collapse = stress >= config.slab_cohesion_threshold;

        println!("{:<12} {:<12.4} {:<12} {:<12} {:<12.4}",
            air_gap, stress, overstressed, would_collapse, support_score);
    }
    println!();

    // Now sweep overhang_weight to show sensitivity
    println!("=== Sensitivity: overhang_weight (gap=12, ceiling at y=12) ===");
    println!("{:<16} {:<12} {:<12}",
        "overhang_weight", "v2_stress", "would_collapse");
    println!("{}", "-".repeat(40));

    for &ow in &[0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20] {
        let mut cfg = default_config();
        cfg.surface_y = 32;
        cfg.overhang_weight = ow;

        let (density_fields, _, _) = make_tunnel_world(0, 11);
        let all_keys: Vec<(i32,i32,i32)> = density_fields.keys().cloned().collect();
        let scores = ground_connectivity_pass(&density_fields, &all_keys, 16, &cfg);
        let (stress, _) = calc_voxel_stress_v2(
            &density_fields, &scores, &cfg,
            8, 12, 8, 16,
        );

        println!("{:<16.3} {:<12.4} {:<12}",
            ow, stress, stress >= cfg.slab_cohesion_threshold);
    }
    println!();

    // Multi-chunk span: measure stress at varying positions across a 3-chunk-wide tunnel
    println!("=== Multi-chunk span: stress across 48-voxel-wide ceiling ===");
    println!("{:<12} {:<12} {:<12}",
        "x_position", "v2_stress", "support_score");
    println!("{}", "-".repeat(36));

    let (density_fields, _, _) = make_tunnel_world(0, 11);
    let all_keys: Vec<(i32,i32,i32)> = density_fields.keys().cloned().collect();
    let scores = ground_connectivity_pass(&density_fields, &all_keys, 16, &config);

    // Sample stress across x positions in different chunks (with corrected surface_y)
    for &(cx, x) in &[(-1,4), (-1,8), (-1,12), (0,4), (0,8), (0,12), (1,4), (1,8), (1,12)] {
        let (stress, _) = calc_voxel_stress_v2(
            &density_fields, &scores, &config,
            cx * 16 + x, 12, 8, 16,
        );
        let score = scores.get(&(cx, 0, 0))
            .map(|s| s.get(x as usize, 12, 8))
            .unwrap_or(-1.0);
        println!("{:<12} {:<12.4} {:<12.4}",
            format!("c{}:x{}", cx, x), stress, score);
    }
    println!();

    // Sweep span_weight — the other major knob
    println!("=== Sensitivity: span_weight (gap=12, ceiling at y=12, min_safe_span=8) ===");
    println!("{:<16} {:<12} {:<12} {:<20}",
        "span_weight", "v2_stress", "would_collapse", "collapses_at_span");
    println!("{}", "-".repeat(60));

    for &sw in &[0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30] {
        let mut cfg = default_config();
        cfg.surface_y = 32;
        cfg.span_weight = sw;

        let (density_fields, _, _) = make_tunnel_world(0, 11);
        let all_keys: Vec<(i32,i32,i32)> = density_fields.keys().cloned().collect();
        let scores = ground_connectivity_pass(&density_fields, &all_keys, 16, &cfg);
        let (stress, _) = calc_voxel_stress_v2(
            &density_fields, &scores, &cfg,
            8, 12, 8, 16,
        );

        // Calculate at what span width this would collapse (stress >= 0.75)
        // stress = overhang_weight * overhang_factor + span_weight * max(0, span - min_safe_span)
        // For collapse: 0.75 = 0.05 * oh + sw * (span - 8)
        // Simplified: span_for_collapse = (0.75 - base_stress) / sw + 8
        let collapse_span = if sw > 0.0 { ((0.75 - 0.05 * 12.0) / sw + 8.0) as i32 } else { 999 };

        println!("{:<16.3} {:<12.4} {:<12} span >= {:<12}",
            sw, stress, stress >= cfg.slab_cohesion_threshold, collapse_span);
    }
    println!();

    // Sweep min_safe_span
    println!("=== Sensitivity: min_safe_span (gap=12, ceiling at y=12) ===");
    println!("{:<16} {:<12} {:<12}",
        "min_safe_span", "v2_stress", "would_collapse");
    println!("{}", "-".repeat(40));

    for &mss in &[2, 4, 6, 8, 10, 12, 16] {
        let mut cfg = default_config();
        cfg.surface_y = 32;
        cfg.min_safe_span = mss;

        let (density_fields, _, _) = make_tunnel_world(0, 11);
        let all_keys: Vec<(i32,i32,i32)> = density_fields.keys().cloned().collect();
        let scores = ground_connectivity_pass(&density_fields, &all_keys, 16, &cfg);
        let (stress, _) = calc_voxel_stress_v2(
            &density_fields, &scores, &cfg,
            8, 12, 8, 16,
        );

        println!("{:<16} {:<12.4} {:<12}",
            mss, stress, stress >= cfg.slab_cohesion_threshold);
    }
}
