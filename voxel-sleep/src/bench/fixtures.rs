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

// ─── Helper: Realistic world (3×3×3 via region gen) ────────────────────────

pub(crate) fn make_realistic_world(seed: u64, water_count: usize) -> (
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
    let (density_fields, _pools, fluid_seeds, _worms, _timings, _springs, _zones) =
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
                stagnant_ticks: 0, hops_from_source: 255, max_flow_dist: 0,
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

pub(crate) fn make_ue_gen_config(seed: u64) -> voxel_gen::config::GenerationConfig {
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
        blank_canvas: false,
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
            ore_global_scale: 1.0,
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
        zones: ZoneConfig::default(),
        mushrooms: MushroomConfig::default(),
    }
}

// ─── Helper: Generate 3×3×3 world at arbitrary center chunk ──────────────

pub(crate) fn make_realistic_world_at(
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

    let (density_fields, _pools, fluid_seeds, _worms, _timings, _springs, _zones) =
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
                stagnant_ticks: 0, hops_from_source: 255, max_flow_dist: 0,
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
pub(crate) fn inject_fluid_patches(
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
                    is_source: true, grace_ticks: 0, stagnant_ticks: 0, hops_from_source: 255, max_flow_dist: 0,
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
                    is_source: true, grace_ticks: 0, stagnant_ticks: 0, hops_from_source: 255, max_flow_dist: 0,
                };
                lava_placed += 1;
            }
        }
    }

    (water_placed, lava_placed)
}

// ─── Helper: Synthetic world (controlled material + air channel) ───────────

pub(crate) fn make_synthetic_world(
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
                stagnant_ticks: 0, hops_from_source: 255, max_flow_dist: 0,
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
                stagnant_ticks: 0, hops_from_source: 255, max_flow_dist: 0,
            };
        }
    }

    (density_fields, stress_fields, support_fields, fluid)
}

// ─── Helper: BFS fluid cluster ─────────────────────────────────────────────

pub(crate) fn build_fluid_cluster(center: (i32, i32, i32), count: usize) -> Vec<(i32, i32, i32)> {
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
pub(crate) fn seed_material(
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

pub(crate) struct FluidConfig {
    pub(crate) name: &'static str,
    pub(crate) water_count: usize,
}

pub(crate) const FLUID_CONFIGS: &[FluidConfig] = &[
    FluidConfig { name: "Dry",     water_count: 0 },
    FluidConfig { name: "Damp",    water_count: 30 },
    FluidConfig { name: "Wet",     water_count: 100 },
    FluidConfig { name: "Flooded", water_count: 300 },
];

// All non-Air material u8 IDs (1..=21)
pub(crate) const ALL_MAT_IDS: [u8; 25] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25];
