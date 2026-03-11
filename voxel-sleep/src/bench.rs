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
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

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
    // Veins (defaults: vein_deposition_prob=0.85, max_vein_voxels=80, search_radius=20)
    cfg.veins.vein_deposition_prob = 0.85;
    cfg.veins.max_vein_voxels_per_source = 80;
    cfg.veins.heat_source_search_radius = 20;
    // Enrichment
    cfg.deeptime.enrichment_prob = 0.90;
    cfg.deeptime.max_enrichment_per_chunk = 400;
    cfg.deeptime.enrichment_cluster_max = 30;
    cfg.deeptime.enrichment_search_radius = 12;
    cfg.deeptime.vein_thickening_prob = 0.35;
    cfg.deeptime.vein_thickening_growth_max = 8;
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

// ─── Helper: UE5-matching generation config (from VoxelConfig.json + FormationConfig.json) ──

fn make_ue_gen_config(seed: u64) -> voxel_gen::config::GenerationConfig {
    use voxel_gen::config::*;

    GenerationConfig {
        seed,
        chunk_size: 16,
        // bounds_size=0 → uses chunk_size=16 for noise coords (matches sleep coord system)
        bounds_size: 0.0,
        region_size: 3,
        mesh_smooth_iterations: 3,
        mesh_smooth_strength: 0.4,
        mesh_boundary_smooth: 0.3,
        mesh_recalc_normals: 0,
        ore_detail_multiplier: 1,
        ore_protrusion: 0.0,
        fluid_sources_enabled: false,  // UE5 has this OFF; cauldron seeds only
        octree_max_depth: 4,

        // Noise — VoxelConfig.json
        noise: NoiseConfig {
            cavern_frequency: 0.004,
            cavern_threshold: 0.75,
            detail_octaves: 12,
            detail_persistence: 0.5,
            warp_amplitude: 2.0,
        },

        // Worms — UE5 has 0 worms per region
        worm: WormConfig {
            worms_per_region: 0.0,
            radius_min: 4.0,
            radius_max: 5.0,
            step_length: 0.7,
            max_steps: 300,
            falloff_power: 2.0,
        },

        ore: OreConfig {
            host_rock: HostRockConfig {
                sandstone_depth: 250.0,
                granite_depth: -100.0,
                basalt_depth: -200.0,
                slate_depth: 130.0,
                boundary_noise_amplitude: 13.0,
                boundary_noise_frequency: 0.08,
                basalt_intrusion_frequency: 0.02,
                basalt_intrusion_threshold: 0.85,
                basalt_intrusion_depth_max: 10.0,
            },
            iron: BandedIronConfig {
                band_frequency: 0.2,
                noise_perturbation: 1.0,
                noise_frequency: 0.11,
                threshold: 1.35,
                depth_min: -700.0,
                depth_max: 700.0,
            },
            copper: OreVeinParams { frequency: 0.011, threshold: 0.91, depth_min: -220.0, depth_max: 350.0 },
            malachite: OreVeinParams { frequency: 0.8, threshold: 0.94, depth_min: -200.0, depth_max: -30.0 },
            quartz: OreVeinParams { frequency: 0.01, threshold: 0.88, depth_min: -200.0, depth_max: 200.0 },
            gold: OreVeinParams { frequency: 0.08, threshold: 0.96, depth_min: -200.0, depth_max: 200.0 },
            pyrite: OreVeinParams { frequency: 0.05, threshold: 0.98, depth_min: -200.0, depth_max: 200.0 },
            kimberlite: KimberlitePipeConfig {
                pipe_frequency_2d: 0.008,
                pipe_threshold: 0.94,
                depth_min: -200.0,
                depth_max: -30.0,
                diamond_threshold: 0.93,
                diamond_frequency: 0.10,
            },
            sulfide: SulfideBlobConfig {
                frequency: 0.5,
                threshold: 0.90,
                tin_threshold: 0.5,
                depth_min: -200.0,
                depth_max: -20.0,
            },
            geode: GeodeConfig {
                frequency: 0.002,
                center_threshold: 0.98,
                shell_thickness: 0.01,
                hollow_factor: -0.1,
                depth_min: -200.0,
                depth_max: 200.0,
            },
            coal: OreVeinParams { frequency: 0.03, threshold: 0.62, depth_min: 10.0, depth_max: 80.0 },
            ore_domain_warp_strength: 5.0,
            ore_warp_frequency: 0.01,
            ore_edge_falloff: 0.08,
            ore_detail_weight: 0.2,
            // All geological realism toggles OFF (matching UE5 defaults)
            iron_sedimentary_only: false,
            iron_depth_fade: false,
            copper_supergene: false,
            copper_granite_contact: false,
            malachite_depth_bias: false,
            kimberlite_carrot_taper: false,
            diamond_depth_grade: false,
            sulfide_gossan_cap: false,
            sulfide_disseminated: false,
            pyrite_ore_halo: false,
            quartz_planar_veins: false,
            gold_bonanza: false,
            geode_volcanic_host: false,
            geode_depth_scaling: false,
            coal_sedimentary_host: false,
            coal_shallow_ceiling: false,
            coal_depth_enrichment: false,
        },

        // Formation config — FormationConfig.json
        formations: FormationConfig {
            enabled: true,
            placement_frequency: 0.26,
            placement_threshold: 0.25,
            stalactite_chance: 0.31,
            stalagmite_chance: 0.22,
            flowstone_chance: 0.02,
            column_chance: 1.0,
            column_max_gap: 200,
            length_min: 5.0,
            length_max: 8.0,
            radius_min: 1.9,
            radius_max: 2.4,
            max_radius: 3.1,
            column_radius_min: 15.5,
            column_radius_max: 27.5,
            flowstone_length_min: 2.7,
            flowstone_length_max: 4.5,
            flowstone_thickness: 1.1,
            min_air_gap: 3,
            min_clearance: 4,
            smoothness: 4.0,
            mega_column_chance: 0.30,
            mega_column_min_gap: 12,
            mega_column_radius_min: 16.0,
            mega_column_radius_max: 26.0,
            mega_column_noise_strength: 0.3,
            mega_column_ring_frequency: 0.8,
            drapery_chance: 0.02,
            drapery_length_min: 5.0,
            drapery_length_max: 6.0,
            drapery_wave_frequency: 3.5,
            drapery_wave_amplitude: 1.8,
            rimstone_chance: 0.06,
            rimstone_dam_height_min: 1.2,
            rimstone_dam_height_max: 2.1,
            rimstone_pool_depth: 1.0,
            rimstone_min_slope: 0.15,
            shield_chance: 0.41,
            shield_radius_min: 5.0,
            shield_radius_max: 8.0,
            shield_max_tilt: 35.0,
            shield_stalactite_chance: 0.5,
            cauldron_chance: 0.08,
            cauldron_radius_min: 5.0,
            cauldron_radius_max: 8.0,
            cauldron_depth: 6.0,
            cauldron_lip_height: 4.0,
            cauldron_rim_stalagmite_count_min: 2,
            cauldron_rim_stalagmite_count_max: 5,
            cauldron_rim_stalagmite_scale: 0.7,
            cauldron_floor_noise: 0.3,
            cauldron_water_chance: 0.65,
            cauldron_lava_chance: 0.80,
            cauldron_wall_inset: 1.0,
            cauldron_floor_inset: 1,
        },

        // Pools disabled in UE5
        pools: PoolConfig { enabled: false, ..PoolConfig::default() },

        // Water table — from TimeSkipConfig.json
        water_table: WaterTableConfig {
            enabled: true,
            base_y: 170.0,
            noise_amplitude: 15.0,
            noise_frequency: 0.008,
            spring_flow_rate: 0.8,
            min_porosity_for_spring: 0.5,
            drip_noise_frequency: 0.15,
            drip_noise_threshold: 0.7,
            drip_level: 0.4,
            max_springs_per_chunk: 8,
            max_drips_per_chunk: 12,
        },

        pipe_lava: PipeLavaConfig {
            enabled: true,
            activation_depth: -80.0,
            max_lava_per_chunk: 6,
            depth_scaling: 0.5,
        },

        lava_tubes: LavaTubeConfig {
            enabled: true,
            tubes_per_region: 2.0,
            depth_min: -250.0,
            depth_max: -50.0,
            radius_min: 2.0,
            radius_max: 4.0,
            max_steps: 150,
            step_length: 1.5,
            active_depth: -120.0,
            pipe_connection_radius: 20.0,
        },

        hydrothermal: HydrothermalConfig { enabled: true, radius: 8, max_per_chunk: 4 },

        rivers: RiverConfig {
            enabled: true,
            rivers_per_region: 1.0,
            width_min: 3.0,
            width_max: 6.0,
            height: 2.5,
            max_steps: 300,
            step_length: 1.5,
            layer_restriction: true,
            downslope_bias: 0.02,
        },

        artesian: ArtesianConfig {
            enabled: true,
            aquifer_y_center: -15.0,
            aquifer_thickness: 3.0,
            aquifer_noise_freq: 0.01,
            aquifer_noise_threshold: 0.3,
            pressure_noise_freq: 0.02,
            max_per_chunk: 3,
        },

        mine: MineConfig::default(),
        crystals: CrystalConfig::default(),
    }
}

// ─── Helper: Generate 3×3×3 world at arbitrary center chunk ──────────────

fn make_realistic_world_at(
    gen_config: &voxel_gen::config::GenerationConfig,
    center_chunk: (i32, i32, i32),
    radius: i32,
) -> (
    HashMap<(i32, i32, i32), DensityField>,
    HashMap<(i32, i32, i32), StressField>,
    HashMap<(i32, i32, i32), SupportField>,
    FluidSnapshot,
) {
    let grid_size = 17;
    let (cx, cy, cz) = center_chunk;
    let mut coords = Vec::new();
    for dx in -radius..=radius {
        for dy in -radius..=radius {
            for dz in -radius..=radius {
                coords.push((cx + dx, cy + dy, cz + dz));
            }
        }
    }

    let (density_fields, _pools, fluid_seeds, _worms, _timings, _springs) =
        voxel_gen::region_gen::generate_region_densities(&coords, gen_config);

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

    eprintln!("  World: {} chunks, gen fluid: {} water + {} lava",
        density_fields.len(), gen_water, gen_lava);

    (density_fields, stress_fields, support_fields, fluid)
}

// ─── Helper: Inject 4 water + 4 lava patches ─────────────────────────────

/// Place water and lava patches in surface-adjacent air voxels.
/// Lava patches are placed at least `min_gap` (Manhattan) voxels from any water patch.
/// Returns (water_cells_placed, lava_cells_placed).
fn inject_fluid_patches(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    fluid: &mut FluidSnapshot,
    patch_count: usize,
    cells_per_patch: usize,
    min_gap: i32,
    rng_seed: u64,
) -> (usize, usize) {
    let chunk_size = 16usize;
    let mut rng = ChaCha8Rng::seed_from_u64(rng_seed);

    // Collect surface-adjacent air candidates
    let mut candidates: Vec<(i32, i32, i32)> = Vec::new();
    let mut chunk_keys: Vec<_> = density_fields.keys().copied().collect();
    chunk_keys.sort();

    for &(cx, cy, cz) in &chunk_keys {
        let df = &density_fields[&(cx, cy, cz)];
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
                        let wx = cx * 16 + lx as i32;
                        let wy = cy * 16 + ly as i32;
                        let wz = cz * 16 + lz as i32;
                        candidates.push((wx, wy, wz));
                    }
                }
            }
        }
    }

    if candidates.len() < patch_count * 2 { return (0, 0); }
    candidates.shuffle(&mut rng);

    let manhattan = |a: (i32,i32,i32), b: (i32,i32,i32)| -> i32 {
        (a.0 - b.0).abs() + (a.1 - b.1).abs() + (a.2 - b.2).abs()
    };

    // Pick water patch centers (spread out via stride)
    let step = candidates.len() / patch_count;
    let water_centers: Vec<(i32, i32, i32)> = (0..patch_count)
        .map(|i| candidates[i * step])
        .collect();

    // Pick lava centers: at least min_gap from any water center and from each other
    let mut lava_centers: Vec<(i32, i32, i32)> = Vec::new();
    for &c in &candidates {
        if lava_centers.len() >= patch_count { break; }
        let far_water = water_centers.iter().all(|w| manhattan(c, *w) >= min_gap);
        let far_lava = lava_centers.iter().all(|l| manhattan(c, *l) >= min_gap);
        if far_water && far_lava { lava_centers.push(c); }
    }

    // Place water
    let mut water_placed = 0;
    for &center in &water_centers {
        let cluster = build_fluid_cluster(center, cells_per_patch);
        for &(wx, wy, wz) in &cluster {
            let chunk = (wx.div_euclid(16), wy.div_euclid(16), wz.div_euclid(16));
            let lx = wx.rem_euclid(16) as usize;
            let ly = wy.rem_euclid(16) as usize;
            let lz = wz.rem_euclid(16) as usize;
            if let Some(df) = density_fields.get(&chunk) {
                if df.get(lx, ly, lz).density > 0.0 { continue; }
            } else { continue; }
            let cells = fluid.chunks.entry(chunk).or_insert_with(empty_fluid_cells);
            let idx = lz * 16 * 16 + ly * 16 + lx;
            if idx < cells.len() && cells[idx].level < 0.01 {
                cells[idx] = FluidCell {
                    level: 1.0, fluid_type: FluidType::Water,
                    is_source: true, grace_ticks: 0, stagnant_ticks: 0,
                };
                water_placed += 1;
            }
        }
    }

    // Place lava
    let mut lava_placed = 0;
    for &center in &lava_centers {
        let cluster = build_fluid_cluster(center, cells_per_patch);
        for &(wx, wy, wz) in &cluster {
            let chunk = (wx.div_euclid(16), wy.div_euclid(16), wz.div_euclid(16));
            let lx = wx.rem_euclid(16) as usize;
            let ly = wy.rem_euclid(16) as usize;
            let lz = wz.rem_euclid(16) as usize;
            if let Some(df) = density_fields.get(&chunk) {
                if df.get(lx, ly, lz).density > 0.0 { continue; }
            } else { continue; }
            let cells = fluid.chunks.entry(chunk).or_insert_with(empty_fluid_cells);
            let idx = lz * 16 * 16 + ly * 16 + lx;
            if idx < cells.len() && cells[idx].level < 0.01 {
                cells[idx] = FluidCell {
                    level: 1.0, fluid_type: FluidType::Lava,
                    is_source: true, grace_ticks: 0, stagnant_ticks: 0,
                };
                lava_placed += 1;
            }
        }
    }

    (water_placed, lava_placed)
}

// ─── Helper: Print material census ────────────────────────────────────────

fn print_material_census(label: &str, counts: &MatMap<u32>) {
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
fn count_fluid_cells(fluid: &FluidSnapshot) -> (u32, u32) {
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
fn average_mat_maps(maps: &[MatMap<u32>]) -> MatMap<f64> {
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

fn material_source(mid: u8) -> &'static str {
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

fn material_expected_limestone(mid: u8) -> &'static str {
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

fn material_expected_slate(mid: u8) -> &'static str {
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
fn print_realistic_report(
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

// ===================================================================
// Test 11: Mining Exploitation Test
// Tests whether players can amplify geological production by mining
// tunnels/caverns near fluid sources before sleeping.
// ===================================================================

/// Carve a line of air voxels (simulating player mining) in world coordinates.
/// Returns how many voxels were actually carved (were solid before).
fn carve_air(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    positions: &[(i32, i32, i32)],
) -> usize {
    let mut carved = 0;
    for &(wx, wy, wz) in positions {
        let chunk = (wx.div_euclid(16), wy.div_euclid(16), wz.div_euclid(16));
        let lx = wx.rem_euclid(16) as usize;
        let ly = wy.rem_euclid(16) as usize;
        let lz = wz.rem_euclid(16) as usize;
        if let Some(df) = density_fields.get_mut(&chunk) {
            let sample = df.get_mut(lx, ly, lz);
            if sample.density > 0.0 {
                sample.density = 0.0;
                sample.material = Material::Air;
                carved += 1;
            }
        }
    }
    carved
}

/// Find world-coords of fluid cells of a given type. Returns up to `limit` positions.
fn find_fluid_positions(
    fluid: &FluidSnapshot,
    fluid_type: FluidType,
    limit: usize,
) -> Vec<(i32, i32, i32)> {
    let mut result = Vec::new();
    let mut keys: Vec<_> = fluid.chunks.keys().copied().collect();
    keys.sort();
    for &(cx, cy, cz) in &keys {
        let cells = &fluid.chunks[&(cx, cy, cz)];
        for (idx, cell) in cells.iter().enumerate() {
            if cell.level > 0.01 && std::mem::discriminant(&cell.fluid_type) == std::mem::discriminant(&fluid_type) {
                let lx = idx % 16;
                let ly = (idx / 16) % 16;
                let lz = idx / (16 * 16);
                let wx = cx * 16 + lx as i32;
                let wy = cy * 16 + ly as i32;
                let wz = cz * 16 + lz as i32;
                result.push((wx, wy, wz));
                if result.len() >= limit { return result; }
            }
        }
    }
    result
}

/// Generate a 3D box of positions (world coords).
fn box_positions(cx: i32, cy: i32, cz: i32, rx: i32, ry: i32, rz: i32) -> Vec<(i32, i32, i32)> {
    let mut result = Vec::new();
    for z in (cz - rz)..=(cz + rz) {
        for y in (cy - ry)..=(cy + ry) {
            for x in (cx - rx)..=(cx + rx) {
                result.push((x, y, z));
            }
        }
    }
    result
}

/// Run one exploitation scenario: generate world, apply mining strategy, run sleep,
/// return total ore delta and total positive delta.
fn run_exploit_scenario(
    rock_type: &str,
    scenario_name: &str,
    mine_fn: &dyn Fn(
        &mut HashMap<(i32, i32, i32), DensityField>,
        &FluidSnapshot,
    ) -> usize,
) -> (i64, i64) {
    const CYCLES: u32 = 3;

    let center: (i32, i32, i32) = match rock_type {
        "Limestone" => (1, 12, 1),
        _ => (1, 1, 1),
    };
    let radius = 4i32;
    let seed = 42u64;

    let mut gen_config = make_ue_gen_config(seed);
    gen_config.formations.cauldron_chance = 0.20;

    let (mut density, mut stress, mut support, mut fluid) =
        make_realistic_world_at(&gen_config, center, radius);

    // 8x fluid (super variant)
    inject_fluid_patches(&density, &mut fluid, 8, 40, 8, seed * 1000 + 7);

    // Apply mining strategy
    let voxels_carved = mine_fn(&mut density, &fluid);

    let before = count_materials(&density);

    // Set up sleep config
    let wy_base = (center.1 - 1) * 16;
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

    let mut total_veins = 0u32;
    let mut total_meta = 0u32;
    let mut total_acid = 0u32;
    let mut total_enrich = 0u32;

    for cycle in 0..CYCLES {
        cfg.sleep_count = cycle + 1;
        let result = execute_sleep(
            &cfg, &mut density, &mut stress, &mut support,
            &mut fluid, center, cycle, None,
        );
        total_veins += result.veins_deposited;
        total_meta += result.voxels_metamorphosed;
        total_acid += result.acid_dissolved;
        total_enrich += result.voxels_enriched;
    }

    let after = count_materials(&density);
    let delta = material_delta(&before, &after);

    // Sum ore deltas (positive = production)
    let ore_ids: &[u8] = &[
        7, 8, 9, 10, 11, 14, 15, 16, // Iron, Copper, Malachite, Tin, Gold, Sulfide, Quartz, Pyrite
    ];
    let total_ore: i64 = ore_ids.iter()
        .map(|&mid| *delta.get(&mid).unwrap_or(&0))
        .filter(|&v| v > 0)
        .sum();

    // Sum all positive deltas (total new material)
    let total_new: i64 = delta.values().filter(|&&v| v > 0).sum();

    eprintln!("    {:<36} carved={:<6} veins={:<6} meta={:<6} acid={:<5} enrich={:<5} ore+={:<6} total+={:<6}",
        scenario_name, voxels_carved, total_veins, total_meta, total_acid, total_enrich,
        total_ore, total_new);

    (total_ore, total_new)
}

#[test]
#[ignore]
fn bench_exploit_mining() {
    eprintln!("\n{:=<90}", "= EXPLOITATION TEST: Mining near fluid sources (super config, seed=42) ");
    eprintln!("  Tests whether player mining strategies amplify geological production.\n");

    for rock_type in &["Limestone", "Slate"] {
        eprintln!("  === {} ===", rock_type);
        eprintln!("    {:<36} {:<12} {:<12} {:<11} {:<11} {:<11} {:<12} {:<12}",
            "Scenario", "Carved", "Veins", "Meta", "Acid", "Enrich", "Ore+", "Total+");
        eprintln!("    {:-<110}", "");

        // 1. Baseline -- no mining
        let (base_ore, base_total) = run_exploit_scenario(
            rock_type, "Baseline (no mining)", &|_density, _fluid| 0,
        );

        // 2. Small tunnels through lava (3x3, length 20)
        let (ore2, _) = run_exploit_scenario(
            rock_type, "3x3 tunnels through lava", &|density, fluid| {
                let lava_pos = find_fluid_positions(fluid, FluidType::Lava, 4);
                let mut carved = 0;
                for &(lx, ly, lz) in &lava_pos {
                    let mut positions = Vec::new();
                    for dx in -10..=10 {
                        for dy in -1..=1 {
                            for dz in -1..=1 {
                                positions.push((lx + dx, ly + dy, lz + dz));
                            }
                        }
                    }
                    carved += carve_air(density, &positions);
                }
                carved
            },
        );

        // 3. 3x3 tunnel UNDER lava (2 blocks below)
        let (ore3, _) = run_exploit_scenario(
            rock_type, "3x3 tunnel under lava (Y-3)", &|density, fluid| {
                let lava_pos = find_fluid_positions(fluid, FluidType::Lava, 4);
                let mut carved = 0;
                for &(lx, ly, lz) in &lava_pos {
                    let mut positions = Vec::new();
                    for dx in -10..=10 {
                        for dy in -1..=1 {
                            for dz in -1..=1 {
                                positions.push((lx + dx, ly - 3 + dy, lz + dz));
                            }
                        }
                    }
                    carved += carve_air(density, &positions);
                }
                carved
            },
        );

        // 4. 3x3 tunnel ABOVE lava (2 blocks above)
        let (ore4, _) = run_exploit_scenario(
            rock_type, "3x3 tunnel above lava (Y+3)", &|density, fluid| {
                let lava_pos = find_fluid_positions(fluid, FluidType::Lava, 4);
                let mut carved = 0;
                for &(lx, ly, lz) in &lava_pos {
                    let mut positions = Vec::new();
                    for dx in -10..=10 {
                        for dy in -1..=1 {
                            for dz in -1..=1 {
                                positions.push((lx + dx, ly + 3 + dy, lz + dz));
                            }
                        }
                    }
                    carved += carve_air(density, &positions);
                }
                carved
            },
        );

        // 5. Giant 7x7x7 cavern next to each lava source
        let (ore5, _) = run_exploit_scenario(
            rock_type, "7x7x7 cavern next to lava", &|density, fluid| {
                let lava_pos = find_fluid_positions(fluid, FluidType::Lava, 4);
                let mut carved = 0;
                for &(lx, ly, lz) in &lava_pos {
                    let positions = box_positions(lx + 5, ly, lz, 3, 3, 3);
                    carved += carve_air(density, &positions);
                }
                carved
            },
        );

        // 6. Channel connecting water to lava
        let (ore6, _) = run_exploit_scenario(
            rock_type, "3x3 channel: water->lava", &|density, fluid| {
                let water_pos = find_fluid_positions(fluid, FluidType::Water, 1);
                let lava_pos = find_fluid_positions(fluid, FluidType::Lava, 1);
                if water_pos.is_empty() || lava_pos.is_empty() { return 0; }
                let (wx, wy, wz) = water_pos[0];
                let (lx, ly, lz) = lava_pos[0];
                let steps = ((lx - wx).abs() + (ly - wy).abs() + (lz - wz).abs()).max(1);
                let mut positions = Vec::new();
                for i in 0..=steps {
                    let t = i as f32 / steps as f32;
                    let cx = wx + ((lx - wx) as f32 * t) as i32;
                    let cy = wy + ((ly - wy) as f32 * t) as i32;
                    let cz = wz + ((lz - wz) as f32 * t) as i32;
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            positions.push((cx, cy + dy, cz + dz));
                        }
                    }
                }
                carve_air(density, &positions)
            },
        );

        // 7. Strip mine: 32x3x32 slab near first lava
        let (ore7, _) = run_exploit_scenario(
            rock_type, "32x3x32 strip mine near lava", &|density, fluid| {
                let lava_pos = find_fluid_positions(fluid, FluidType::Lava, 1);
                if lava_pos.is_empty() { return 0; }
                let (lx, ly, lz) = lava_pos[0];
                let positions = box_positions(lx, ly - 2, lz, 16, 1, 16);
                carve_air(density, &positions)
            },
        );

        // 8. Many small 1x1 shafts (Swiss cheese)
        let (ore8, _) = run_exploit_scenario(
            rock_type, "20 vertical shafts near lava", &|density, fluid| {
                let lava_pos = find_fluid_positions(fluid, FluidType::Lava, 20);
                let mut carved = 0;
                for (i, &(lx, ly, lz)) in lava_pos.iter().enumerate() {
                    let offset_x = (i as i32 % 5) * 3;
                    let offset_z = (i as i32 / 5) * 3;
                    let mut positions = Vec::new();
                    for dy in -5..=5 {
                        positions.push((lx + offset_x, ly + dy, lz + offset_z));
                    }
                    carved += carve_air(density, &positions);
                }
                carved
            },
        );

        // 9. Ring mine around lava (expose maximum surface area)
        let (ore9, _) = run_exploit_scenario(
            rock_type, "Ring mine around lava (r=3)", &|density, fluid| {
                let lava_pos = find_fluid_positions(fluid, FluidType::Lava, 4);
                let mut carved = 0;
                for &(lx, ly, lz) in &lava_pos {
                    let mut positions = Vec::new();
                    for dx in -3..=3i32 {
                        for dy in -3..=3i32 {
                            for dz in -3..=3i32 {
                                let dist = dx.abs() + dy.abs() + dz.abs();
                                if dist >= 2 && dist <= 3 {
                                    positions.push((lx + dx, ly + dy, lz + dz));
                                }
                            }
                        }
                    }
                    carved += carve_air(density, &positions);
                }
                carved
            },
        );

        // Summary
        eprintln!();
        if base_ore > 0 {
            eprintln!("    Baseline ore: {}    Baseline total new material: {}", base_ore, base_total);
            let scenarios = [
                ("Tunnels through lava", ore2),
                ("Tunnel under lava", ore3),
                ("Tunnel above lava", ore4),
                ("7x7x7 cavern", ore5),
                ("Water->lava channel", ore6),
                ("32x3x32 strip mine", ore7),
                ("20 vertical shafts", ore8),
                ("Ring mine around lava", ore9),
            ];
            eprintln!("    {:<30} {:>10} {:>10}", "Scenario", "Ore+", "vs Base");
            eprintln!("    {:-<52}", "");
            for (name, ore) in &scenarios {
                let pct = if base_ore > 0 { (*ore as f64 / base_ore as f64 - 1.0) * 100.0 } else { 0.0 };
                let indicator = if pct > 25.0 { " *** EXPLOIT" }
                    else if pct > 10.0 { " ** notable" }
                    else if pct < -10.0 { " (reduced)" }
                    else { "" };
                eprintln!("    {:<30} {:>10} {:>+9.1}%{}", name, ore, pct, indicator);
            }
        } else {
            eprintln!("    Baseline ore=0 -- cannot compare ratios.");
        }
        eprintln!();
    }

    eprintln!("  Legend: *** EXPLOIT = >25% ore gain vs baseline (needs balancing)");
    eprintln!("          ** notable = >10% ore gain (monitor)");
    eprintln!("          (reduced)  = >10% ore loss (mining hurts production)");
}
