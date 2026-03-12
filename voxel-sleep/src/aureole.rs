//! Phase 2: "The Aureole" — 100,000 years.
//!
//! Lava-zone-centric contact metamorphism: cluster heat sources into zones,
//! compute heat → metamorphic sphere (Hornfels/Skarn) → deposit ore veins.
//! Water erosion along fluid pathways.

use std::collections::{HashMap, HashSet, VecDeque};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use voxel_core::density::DensityField;
use voxel_core::material::Material;
use voxel_core::stress::world_to_chunk_local;
use voxel_fluid::FluidSnapshot;

use crate::config::{AureoleConfig, GroundwaterConfig};
use crate::groundwater::ambient_moisture;
use crate::manifest::ChangeManifest;
use crate::util::{FACE_OFFSETS, sample_material, grow_vein, VeinGrowthParams, VeinBias, default_vein_bias};
use crate::{Bottleneck, PhaseDiagnostics, ResourceCensus, TransformEntry};

/// Type of heat source for coal maturation decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeatSourceType {
    Lava,
    Kimberlite,
}

/// A heat source with position and type.
#[derive(Debug, Clone)]
pub struct HeatSource {
    pub pos: (i32, i32, i32),
    pub source_type: HeatSourceType,
}

/// Heat source positions in world coordinates.
pub type HeatMap = Vec<HeatSource>;

/// Result of the aureole phase.
#[derive(Debug, Default)]
pub struct AureoleResult {
    pub voxels_metamorphosed: u32,
    pub channels_eroded: u32,
    pub coal_matured: u32,
    pub diamonds_formed: u32,
    pub voxels_silicified: u32,
    pub lava_zones_found: u32,
    pub hornfels_placed: u32,
    pub skarn_placed: u32,
    pub veins_placed: u32,
    pub manifest: ChangeManifest,
    pub glimpse_chunk: Option<(i32, i32, i32)>,
    pub transform_log: Vec<TransformEntry>,
    pub diagnostics: PhaseDiagnostics,
}

/// Build a heat map: collect all lava cell positions from fluid snapshot
/// plus kimberlite voxels from density fields.
pub fn build_heat_map(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    fluid_snapshot: &FluidSnapshot,
    chunks: &[(i32, i32, i32)],
    chunk_size: usize,
) -> HeatMap {
    let mut heat_sources: Vec<HeatSource> = Vec::new();
    let field_size = chunk_size + 1;

    // Lava cells from fluid snapshot
    let cs = fluid_snapshot.chunk_size;
    for (&chunk_key, cells) in &fluid_snapshot.chunks {
        let (cx, cy, cz) = chunk_key;
        for z in 0..cs {
            for y in 0..cs {
                for x in 0..cs {
                    let idx = z * cs * cs + y * cs + x;
                    let cell = &cells[idx];
                    if cell.level > 0.001 && cell.fluid_type.is_lava() {
                        let wx = cx * (cs as i32) + x as i32;
                        let wy = cy * (cs as i32) + y as i32;
                        let wz = cz * (cs as i32) + z as i32;
                        heat_sources.push(HeatSource { pos: (wx, wy, wz), source_type: HeatSourceType::Lava });
                    }
                }
            }
        }
    }

    // Kimberlite voxels from density fields
    for &chunk_key in chunks {
        let (cx, cy, cz) = chunk_key;
        let df = match density_fields.get(&chunk_key) {
            Some(df) => df,
            None => continue,
        };

        for lz in 0..field_size {
            for ly in 0..field_size {
                for lx in 0..field_size {
                    if df.get(lx, ly, lz).material == Material::Kimberlite {
                        let wx = cx * (chunk_size as i32) + lx as i32;
                        let wy = cy * (chunk_size as i32) + ly as i32;
                        let wz = cz * (chunk_size as i32) + lz as i32;
                        heat_sources.push(HeatSource { pos: (wx, wy, wz), source_type: HeatSourceType::Kimberlite });
                    }
                }
            }
        }
    }

    heat_sources
}

// ──────────────────────────────────────────────────────────────
// Lava Zone Clustering
// ──────────────────────────────────────────────────────────────

struct LavaZone {
    cells: Vec<(i32, i32, i32)>,
    centroid: (i32, i32, i32),
}

/// Cluster all heat sources (lava + kimberlite) into connected components via BFS.
fn cluster_lava_zones(heat_map: &HeatMap, min_zone_size: u32) -> Vec<LavaZone> {
    // Collect sorted positions for determinism
    let mut positions: Vec<(i32, i32, i32)> = heat_map.iter().map(|h| h.pos).collect();
    positions.sort();
    positions.dedup();

    let pos_set: HashSet<(i32, i32, i32)> = positions.iter().copied().collect();
    let mut visited: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut zones = Vec::new();

    for &pos in &positions {
        if visited.contains(&pos) {
            continue;
        }

        // BFS flood-fill
        let mut queue = VecDeque::new();
        let mut component = Vec::new();
        queue.push_back(pos);
        visited.insert(pos);

        while let Some(current) = queue.pop_front() {
            component.push(current);
            for &(dx, dy, dz) in &FACE_OFFSETS {
                let neighbor = (current.0 + dx, current.1 + dy, current.2 + dz);
                if pos_set.contains(&neighbor) && visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }

        if (component.len() as u32) < min_zone_size {
            continue;
        }

        // Compute centroid
        let n = component.len() as i64;
        let (sx, sy, sz) = component.iter().fold((0i64, 0i64, 0i64), |(ax, ay, az), &(x, y, z)| {
            (ax + x as i64, ay + y as i64, az + z as i64)
        });
        let centroid = ((sx / n) as i32, (sy / n) as i32, (sz / n) as i32);

        zones.push(LavaZone { cells: component, centroid });
    }

    // Sort by centroid for determinism
    zones.sort_by(|a, b| a.centroid.cmp(&b.centroid));
    zones
}

// ──────────────────────────────────────────────────────────────
// Aureole Type Detection
// ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum AureoleType {
    Slate,
    Limestone,
}

/// Determine whether the zone is predominantly limestone-hosted or slate-hosted.
fn determine_aureole_type(
    centroid: (i32, i32, i32),
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
) -> AureoleType {
    let (cx, cy, cz) = centroid;
    let mut limestone_count = 0u32;
    let mut other_count = 0u32;

    for dx in -3..=3i32 {
        for dy in -3..=3i32 {
            for dz in -3..=3i32 {
                if let Some(mat) = sample_material(density_fields, cx + dx, cy + dy, cz + dz, chunk_size) {
                    if mat == Material::Limestone {
                        limestone_count += 1;
                    } else if mat.is_host_rock() && mat != Material::Limestone {
                        other_count += 1;
                    }
                }
            }
        }
    }

    if limestone_count > other_count {
        AureoleType::Limestone
    } else {
        AureoleType::Slate
    }
}

// ──────────────────────────────────────────────────────────────
// Water Boost
// ──────────────────────────────────────────────────────────────

/// Compute water boost multiplier by counting water cells near the zone centroid.
fn compute_water_boost(
    centroid: (i32, i32, i32),
    lava_count: u32,
    fluid_snapshot: &FluidSnapshot,
    search_radius: f32,
    max_boost: f32,
) -> f32 {
    let (cx, cy, cz) = centroid;
    let r = search_radius.ceil() as i32;
    let r_sq = search_radius * search_radius;
    let cs = fluid_snapshot.chunk_size;
    let mut water_count = 0u32;

    for dx in -r..=r {
        for dy in -r..=r {
            for dz in -r..=r {
                let dist_sq = (dx * dx + dy * dy + dz * dz) as f32;
                if dist_sq > r_sq {
                    continue;
                }
                let wx = cx + dx;
                let wy = cy + dy;
                let wz = cz + dz;
                let fck = (
                    wx.div_euclid(cs as i32),
                    wy.div_euclid(cs as i32),
                    wz.div_euclid(cs as i32),
                );
                let flx = wx.rem_euclid(cs as i32) as usize;
                let fly = wy.rem_euclid(cs as i32) as usize;
                let flz = wz.rem_euclid(cs as i32) as usize;
                let fidx = flz * cs * cs + fly * cs + flx;
                if let Some(cells) = fluid_snapshot.chunks.get(&fck) {
                    if fidx < cells.len() && cells[fidx].fluid_type.is_water() && cells[fidx].level > 0.001 {
                        water_count += 1;
                    }
                }
            }
        }
    }

    let ratio = (water_count as f32 / lava_count.max(1) as f32).min(1.0);
    1.0 + ratio * max_boost
}

// ──────────────────────────────────────────────────────────────
// Metamorphic Sphere Placement
// ──────────────────────────────────────────────────────────────

/// Place metamorphic sphere: limestone → Skarn, other host rock → Hornfels.
/// Returns (hornfels_count, skarn_count).
fn place_metamorphic_sphere(
    centroid: (i32, i32, i32),
    radius: f32,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
    manifest: &mut ChangeManifest,
) -> (u32, u32) {
    let (cx, cy, cz) = centroid;
    let r = radius.ceil() as i32;
    let r_sq = radius * radius;
    let mut hornfels_count = 0u32;
    let mut skarn_count = 0u32;

    for dx in -r..=r {
        for dy in -r..=r {
            for dz in -r..=r {
                let dist_sq = (dx * dx + dy * dy + dz * dz) as f32;
                if dist_sq > r_sq {
                    continue;
                }
                let wx = cx + dx;
                let wy = cy + dy;
                let wz = cz + dz;
                let (key, lx, ly, lz) = world_to_chunk_local(wx, wy, wz, chunk_size);
                let (old_mat, old_density) = match density_fields.get(&key) {
                    Some(df) => {
                        let s = df.get(lx, ly, lz);
                        (s.material, s.density)
                    }
                    None => continue,
                };

                let new_mat = if old_mat == Material::Limestone {
                    Material::Skarn
                } else if old_mat.is_host_rock()
                    && old_mat != Material::Hornfels
                    && old_mat != Material::Skarn
                {
                    Material::Hornfels
                } else {
                    continue;
                };

                if let Some(df) = density_fields.get_mut(&key) {
                    df.get_mut(lx, ly, lz).material = new_mat;
                }
                manifest.record_voxel_change(key, lx, ly, lz, old_mat, old_density, new_mat, old_density);

                if new_mat == Material::Skarn {
                    skarn_count += 1;
                } else {
                    hornfels_count += 1;
                }
            }
        }
    }

    (hornfels_count, skarn_count)
}

// ──────────────────────────────────────────────────────────────
// Vein Seed Distribution (golden-angle spiral)
// ──────────────────────────────────────────────────────────────

/// Distribute vein seed positions by raycasting outward from centroid using golden-angle spiral.
fn distribute_vein_seeds(
    centroid: (i32, i32, i32),
    radius: f32,
    count: usize,
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
) -> Vec<(i32, i32, i32)> {
    let golden_angle = std::f32::consts::PI * (3.0 - (5.0_f32).sqrt());
    let (cx, cy, cz) = centroid;
    let mut seeds = Vec::new();

    // Overshoot: generate 3× the requested count, take first `count` hits
    let attempt_count = count * 3;

    for i in 0..attempt_count {
        if seeds.len() >= count {
            break;
        }
        let y_frac = 1.0 - 2.0 * (i as f32) / (attempt_count as f32 - 1.0).max(1.0);
        let r_at = (1.0 - y_frac * y_frac).sqrt();
        let theta = golden_angle * i as f32;

        let dx = r_at * theta.cos();
        let dy = y_frac;
        let dz = r_at * theta.sin();

        // Raycast outward from centroid
        let max_steps = (radius * 2.0).ceil().max(20.0) as i32;
        let mut last_solid = None;

        for step in 1..=max_steps {
            let wx = cx + (dx * step as f32).round() as i32;
            let wy = cy + (dy * step as f32).round() as i32;
            let wz = cz + (dz * step as f32).round() as i32;

            if let Some(mat) = sample_material(density_fields, wx, wy, wz, chunk_size) {
                if mat.is_solid() {
                    last_solid = Some((wx, wy, wz));
                } else if last_solid.is_some() {
                    // Found air/lava boundary — use the last solid as seed
                    break;
                }
            }
        }

        if let Some(seed) = last_solid {
            seeds.push(seed);
        }
    }

    seeds
}

// ──────────────────────────────────────────────────────────────
// Ore Vein + Pocket Placement
// ──────────────────────────────────────────────────────────────

/// Write vein voxels into density fields, returns count of voxels placed.
fn apply_vein_to_world(
    positions: &[(i32, i32, i32)],
    material: Material,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
    manifest: &mut ChangeManifest,
) -> u32 {
    let mut count = 0u32;
    for &(wx, wy, wz) in positions {
        let (key, lx, ly, lz) = world_to_chunk_local(wx, wy, wz, chunk_size);
        let (old_mat, old_density) = match density_fields.get(&key) {
            Some(df) => {
                let s = df.get(lx, ly, lz);
                (s.material, s.density)
            }
            None => continue,
        };

        // Only place into host rock (including Hornfels/Skarn)
        if !old_mat.is_host_rock() {
            continue;
        }

        if let Some(df) = density_fields.get_mut(&key) {
            df.get_mut(lx, ly, lz).material = material;
        }
        manifest.record_voxel_change(key, lx, ly, lz, old_mat, old_density, material, old_density);
        count += 1;
    }
    count
}

/// Compute vein size bounds scaled by heat level.
fn scaled_vein_size(base: u32, heat_level: f32, large: bool) -> (u32, u32) {
    let heat_scale = heat_level.sqrt() / 5.0; // 25 cells → 1.0×
    let scale_factor = if large { 0.5 } else { 0.3 };
    let max = base + (base as f32 * heat_scale * scale_factor) as u32;
    (base, max.max(base + 1))
}

/// Place ore veins for a Slate-hosted aureole zone.
fn place_slate_veins(
    centroid: (i32, i32, i32),
    radius: f32,
    heat_level: f32,
    config: &AureoleConfig,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
    manifest: &mut ChangeManifest,
    rng: &mut ChaCha8Rng,
    skarn_count: u32,
) -> u32 {
    let seeds = distribute_vein_seeds(centroid, radius, 8, density_fields, chunk_size);
    if seeds.is_empty() {
        return 0;
    }

    let ores = [Material::Copper, Material::Iron, Material::Tin];
    // Pick 2 of 3
    let skip = rng.gen_range(0..3usize);
    let ore_a = ores[(skip + 1) % 3];
    let ore_b = ores[(skip + 2) % 3];

    let (large_min, large_max) = scaled_vein_size(config.large_vein_base_size, heat_level, true);
    let (small_min, small_max) = scaled_vein_size(config.small_vein_base_size, heat_level, false);

    let mut total_placed = 0u32;
    let assignments: Vec<(Material, u32, u32)> = vec![
        (ore_a, large_min, large_max),  // seed 0
        (ore_a, large_min, large_max),  // seed 1
        (ore_a, large_min, large_max),  // seed 2
        (ore_b, large_min, large_max),  // seed 3
        (ore_b, large_min, large_max),  // seed 4
        (Material::Pyrite, small_min, small_max), // seed 5
        (Material::Pyrite, small_min, small_max), // seed 6
        (Material::Pyrite, small_min, small_max), // seed 7
    ];

    for (i, &seed) in seeds.iter().enumerate() {
        if i >= assignments.len() {
            break;
        }
        let (ore, min_sz, max_sz) = assignments[i];
        let bias = default_vein_bias(ore, rng);
        let params = VeinGrowthParams { ore, min_size: min_sz, max_size: max_sz, bias, exclude_aureole: false };
        let positions = grow_vein(density_fields, seed, &params, chunk_size, rng);
        total_placed += apply_vein_to_world(&positions, ore, density_fields, chunk_size, manifest);
    }

    // Bonus: if skarn exists (slate aureole reached limestone), place Garnet/Diopside pockets in skarn region
    if skarn_count > 0 {
        let (pocket_min, pocket_max) = scaled_vein_size(config.garnet_pocket_size, heat_level, false);
        // Try to find a skarn voxel to seed pockets
        let r = radius.ceil() as i32;
        let (cx, cy, cz) = centroid;
        let mut skarn_seeds: Vec<(i32, i32, i32)> = Vec::new();
        for dx in -r..=r {
            for dy in -r..=r {
                for dz in -r..=r {
                    if (dx * dx + dy * dy + dz * dz) as f32 > radius * radius {
                        continue;
                    }
                    let wx = cx + dx;
                    let wy = cy + dy;
                    let wz = cz + dz;
                    if let Some(mat) = sample_material(density_fields, wx, wy, wz, chunk_size) {
                        if mat == Material::Skarn {
                            skarn_seeds.push((wx, wy, wz));
                        }
                    }
                }
            }
        }
        skarn_seeds.sort();
        if !skarn_seeds.is_empty() {
            // Place 1-2 garnet pockets + 1-2 diopside pockets
            let garnet_seed = skarn_seeds[rng.gen_range(0..skarn_seeds.len())];
            let params = VeinGrowthParams {
                ore: Material::Garnet,
                min_size: pocket_min,
                max_size: pocket_max,
                bias: VeinBias::Compact,
                exclude_aureole: false,
            };
            let positions = grow_vein(density_fields, garnet_seed, &params, chunk_size, rng);
            total_placed += apply_vein_to_world(&positions, Material::Garnet, density_fields, chunk_size, manifest);

            let diopside_seed = skarn_seeds[rng.gen_range(0..skarn_seeds.len())];
            let (dp_min, dp_max) = scaled_vein_size(config.diopside_pocket_size, heat_level, false);
            let params = VeinGrowthParams {
                ore: Material::Diopside,
                min_size: dp_min,
                max_size: dp_max,
                bias: VeinBias::Compact,
                exclude_aureole: false,
            };
            let positions = grow_vein(density_fields, diopside_seed, &params, chunk_size, rng);
            total_placed += apply_vein_to_world(&positions, Material::Diopside, density_fields, chunk_size, manifest);
        }
    }

    total_placed
}

/// Place ore veins for a Limestone-hosted (Skarn) aureole zone.
fn place_limestone_veins(
    centroid: (i32, i32, i32),
    radius: f32,
    heat_level: f32,
    config: &AureoleConfig,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
    manifest: &mut ChangeManifest,
    rng: &mut ChaCha8Rng,
) -> u32 {
    let seeds = distribute_vein_seeds(centroid, radius, 8, density_fields, chunk_size);
    if seeds.is_empty() {
        return 0;
    }

    let ores = [Material::Copper, Material::Iron, Material::Gold];
    let skip = rng.gen_range(0..3usize);
    let ore_a = ores[(skip + 1) % 3];
    let ore_b = ores[(skip + 2) % 3];

    let (large_min, large_max) = scaled_vein_size(config.large_vein_base_size, heat_level, true);
    let (garnet_min, garnet_max) = scaled_vein_size(config.garnet_pocket_size, heat_level, false);
    let (diopside_min, diopside_max) = scaled_vein_size(config.diopside_pocket_size, heat_level, false);

    let mut total_placed = 0u32;

    // Assignments: 3 ore_a (large) + 2 ore_b (large) + 2 Garnet (inner) + 1 Diopside (outer)
    for (i, &seed) in seeds.iter().enumerate() {
        if i >= 8 {
            break;
        }
        let (ore, min_sz, max_sz) = match i {
            0..=2 => (ore_a, large_min, large_max),
            3..=4 => (ore_b, large_min, large_max),
            5..=6 => (Material::Garnet, garnet_min, garnet_max),
            _ => (Material::Diopside, diopside_min, diopside_max),
        };
        let bias = if ore == Material::Garnet || ore == Material::Diopside {
            VeinBias::Compact
        } else {
            default_vein_bias(ore, rng)
        };
        let params = VeinGrowthParams { ore, min_size: min_sz, max_size: max_sz, bias, exclude_aureole: false };
        let positions = grow_vein(density_fields, seed, &params, chunk_size, rng);
        total_placed += apply_vein_to_world(&positions, ore, density_fields, chunk_size, manifest);
    }

    total_placed
}

// ──────────────────────────────────────────────────────────────
// Main Entry Point
// ──────────────────────────────────────────────────────────────

/// Execute Phase 2: contact metamorphism aureoles + water erosion.
pub fn apply_aureole(
    config: &AureoleConfig,
    groundwater: &GroundwaterConfig,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    fluid_snapshot: &mut FluidSnapshot,
    heat_map: &HeatMap,
    _chunks: &[(i32, i32, i32)],
    chunk_size: usize,
    rng: &mut ChaCha8Rng,
    census: &ResourceCensus,
) -> AureoleResult {
    let mut result = AureoleResult::default();

    struct Candidate {
        chunk_key: (i32, i32, i32),
        lx: usize, ly: usize, lz: usize,
        old_material: Material,
        density: f32,
        new_material: Material,
    }

    let mut candidates: Vec<Candidate> = Vec::new();
    let mut theoretical_max = 0u32;

    // ═══ Lava Zone Contact Metamorphism + Ore Veins ═══
    if config.zone_enabled && config.metamorphism_enabled && !heat_map.is_empty() {
        let zones = cluster_lava_zones(heat_map, config.min_lava_zone_size);
        result.lava_zones_found = zones.len() as u32;

        for zone in &zones {
            let heat = zone.cells.len() as f32 * config.heat_multiplier;
            let base_radius = (heat.sqrt() * config.radius_scale).min(config.max_radius);

            let water_boost = compute_water_boost(
                zone.centroid,
                zone.cells.len() as u32,
                fluid_snapshot,
                base_radius * config.water_search_radius_mult,
                config.water_boost_max,
            );
            let final_radius = base_radius * water_boost;

            if final_radius < 1.0 {
                continue;
            }

            let aureole_type = determine_aureole_type(zone.centroid, density_fields, chunk_size);

            // Pass 1: metamorphic sphere (applied immediately to density_fields)
            let (hornfels_n, skarn_n) = place_metamorphic_sphere(
                zone.centroid,
                final_radius,
                density_fields,
                chunk_size,
                &mut result.manifest,
            );
            result.hornfels_placed += hornfels_n;
            result.skarn_placed += skarn_n;
            result.voxels_metamorphosed += hornfels_n + skarn_n;

            if result.glimpse_chunk.is_none() && (hornfels_n > 0 || skarn_n > 0) {
                let (key, _, _, _) = world_to_chunk_local(
                    zone.centroid.0, zone.centroid.1, zone.centroid.2, chunk_size,
                );
                result.glimpse_chunk = Some(key);
            }

            // Pass 2: ore veins + pockets (grow into just-placed metamorphic rock)
            let veins_placed = match aureole_type {
                AureoleType::Slate => place_slate_veins(
                    zone.centroid, final_radius, heat, config,
                    density_fields, chunk_size, &mut result.manifest, rng, skarn_n,
                ),
                AureoleType::Limestone => place_limestone_veins(
                    zone.centroid, final_radius, heat, config,
                    density_fields, chunk_size, &mut result.manifest, rng,
                ),
            };
            result.veins_placed += veins_placed;
        }

        // Add transform log entry for zone metamorphism
        if result.lava_zones_found > 0 {
            let total_meta = result.hornfels_placed + result.skarn_placed + result.veins_placed;
            result.transform_log.push(TransformEntry {
                description: format!(
                    "The Aureole \u{2014} 100,000 years: {} lava zones, {} hornfels, {} skarn, {} ore vein voxels",
                    result.lava_zones_found, result.hornfels_placed, result.skarn_placed, result.veins_placed
                ),
                count: total_meta,
            });
        }
    }

    // --- Water Erosion ---
    let mut erosion_count = 0u32;
    if config.water_erosion_enabled && !fluid_snapshot.chunks.is_empty() {
        let cs = fluid_snapshot.chunk_size;
        // Collect water cell positions and levels first (avoids borrow conflict for drain)
        let water_cells: Vec<((i32, i32, i32), usize, f32, bool)> = fluid_snapshot.chunks.iter()
            .flat_map(|(&chunk_key, cells)| {
                let (cx, cy, cz) = chunk_key;
                (0..cs).flat_map(move |z| (0..cs).flat_map(move |y| (0..cs).map(move |x| {
                    let idx = z * cs * cs + y * cs + x;
                    let cell = &cells[idx];
                    let wx = cx * (cs as i32) + x as i32;
                    let wy = cy * (cs as i32) + y as i32;
                    let wz = cz * (cs as i32) + z as i32;
                    ((wx, wy, wz), idx, cell.level, cell.fluid_type.is_water() && cell.level > 0.001)
                })))
            })
            .filter(|(_, _, _, valid)| *valid)
            .collect();

        for &((wx, wy, wz), _idx, level, _) in &water_cells {
            // Scale erosion probability by water cell level (more water = stronger erosion)
            let level_factor = level.min(1.0);
            for &(dx, dy, dz) in &FACE_OFFSETS {
                let nx = wx + dx;
                let ny = wy + dy;
                let nz = wz + dz;
                if let Some(mat) = sample_material(density_fields, nx, ny, nz, chunk_size) {
                    if mat == Material::Limestone || mat == Material::Sandstone {
                        theoretical_max += 1;
                    }
                    if (mat == Material::Limestone || mat == Material::Sandstone)
                        && rng.gen::<f32>() < config.water_erosion_prob * level_factor
                    {
                        let (ck, elx, ely, elz) = world_to_chunk_local(nx, ny, nz, chunk_size);
                        if let Some(df) = density_fields.get(&ck) {
                            let sample = df.get(elx, ely, elz);
                            candidates.push(Candidate {
                                chunk_key: ck,
                                lx: elx, ly: ely, lz: elz,
                                old_material: mat,
                                density: sample.density,
                                new_material: Material::Air,
                            });
                            erosion_count += 1;
                        }
                    }
                }
            }
        }

        // Drain water cells used for erosion (0.05 per voxel eroded, skip sources)
        if erosion_count > 0 {
            let drain_total = erosion_count as f32 * 0.05;
            let per_cell = drain_total / water_cells.len().max(1) as f32;
            for &((wx, wy, wz), _idx, _level, _) in &water_cells {
                let fck = (wx.div_euclid(cs as i32), wy.div_euclid(cs as i32), wz.div_euclid(cs as i32));
                let flx = wx.rem_euclid(cs as i32) as usize;
                let fly = wy.rem_euclid(cs as i32) as usize;
                let flz = wz.rem_euclid(cs as i32) as usize;
                let fidx = flz * cs * cs + fly * cs + flx;
                if let Some(cells) = fluid_snapshot.chunks.get_mut(&fck) {
                    if fidx < cells.len() && !cells[fidx].is_source && cells[fidx].level > 0.001 {
                        cells[fidx].level = (cells[fidx].level - per_cell).max(0.0);
                    }
                }
            }
        }
    }

    // --- Ambient Groundwater Erosion ---
    // Only limestone/sandstone dissolve in water (karst dissolution).
    // Granite/basalt/slate/marble don't erode — they're too hard.
    let mut ambient_erosion_count = 0u32;
    if config.water_erosion_enabled && groundwater.enabled {
        let field_size = chunk_size + 1;
        let chunk_keys: Vec<(i32, i32, i32)> = density_fields.keys().copied().collect();
        for chunk_key in chunk_keys {
            let (cx, cy, cz) = chunk_key;
            let df = match density_fields.get(&chunk_key) {
                Some(df) => df,
                None => continue,
            };

            for lz in 0..field_size {
                for ly in 0..field_size {
                    for lx in 0..field_size {
                        let sample = df.get(lx, ly, lz);
                        let mat = sample.material;
                        if !matches!(mat, Material::Limestone | Material::Sandstone) {
                            continue;
                        }

                        let wx = cx * (chunk_size as i32) + lx as i32;
                        let wy = cy * (chunk_size as i32) + ly as i32;
                        let wz = cz * (chunk_size as i32) + lz as i32;

                        // Must be air-adjacent
                        let mut has_air = false;
                        let mut has_air_below = false;
                        for &(dx, dy, dz) in &FACE_OFFSETS {
                            if let Some(neighbor) = sample_material(density_fields, wx + dx, wy + dy, wz + dz, chunk_size) {
                                if !neighbor.is_solid() {
                                    has_air = true;
                                    if dy == -1 { has_air_below = true; }
                                }
                            }
                        }
                        if !has_air {
                            continue;
                        }

                        let moisture = ambient_moisture(groundwater, wy, mat, has_air_below);
                        if moisture > 0.0 && rng.gen::<f32>() < config.water_erosion_prob * moisture * groundwater.erosion_power * groundwater.soft_rock_mult {
                            candidates.push(Candidate {
                                chunk_key,
                                lx, ly, lz,
                                old_material: mat,
                                density: sample.density,
                                new_material: Material::Air,
                            });
                            ambient_erosion_count += 1;
                        }
                    }
                }
            }
        }
    }

    // --- Apply all erosion candidates ---
    let mut conversions: std::collections::BTreeMap<(u8, u8), u32> = std::collections::BTreeMap::new();
    for c in &candidates {
        *conversions.entry((c.old_material as u8, c.new_material as u8)).or_insert(0) += 1;
        if let Some(df) = density_fields.get_mut(&c.chunk_key) {
            let sample = df.get_mut(c.lx, c.ly, c.lz);
            sample.material = c.new_material;
            if c.new_material == Material::Air {
                sample.density = -1.0;
            }
        }

        let new_density = if c.new_material == Material::Air { -1.0 } else { c.density };
        result.manifest.record_voxel_change(
            c.chunk_key, c.lx, c.ly, c.lz,
            c.old_material, c.density,
            c.new_material, new_density,
        );

        if result.glimpse_chunk.is_none() {
            result.glimpse_chunk = Some(c.chunk_key);
        }
    }

    // Also count metamorphic conversions in the diagnostics
    for (_key, delta) in &result.manifest.chunk_deltas {
        for change in &delta.voxel_changes {
            let from = change.old_material as u8;
            let to = change.new_material as u8;
            if from != to && to != Material::Air as u8 {
                *conversions.entry((from, to)).or_insert(0) += 1;
            }
        }
    }

    result.channels_eroded = erosion_count + ambient_erosion_count;

    // Build transform log for erosion
    if erosion_count > 0 {
        result.transform_log.push(TransformEntry {
            description: format!("The Aureole \u{2014} 100,000 years: {} channels widened by water erosion", erosion_count),
            count: erosion_count,
        });
    }
    if ambient_erosion_count > 0 {
        result.transform_log.push(TransformEntry {
            description: format!("The Aureole \u{2014} 100,000 years: {} voxels eroded by ambient groundwater", ambient_erosion_count),
            count: ambient_erosion_count,
        });
    }

    // --- Diagnostics ---
    let actual_output = candidates.len() as u32 + result.hornfels_placed + result.skarn_placed + result.veins_placed;
    result.diagnostics = PhaseDiagnostics {
        conversions,
        theoretical_max,
        actual_output,
        bottlenecks: compute_aureole_bottlenecks(census, heat_map),
    };

    result
}

fn compute_aureole_bottlenecks(census: &ResourceCensus, heat_map: &HeatMap) -> Vec<Bottleneck> {
    let mut bottlenecks = Vec::new();

    if census.water.cell_count == 0 {
        bottlenecks.push(Bottleneck {
            severity: 0.5,
            description: "No water detected \u{2014} erosion needs moisture".into(),
        });
    }

    if heat_map.is_empty() {
        bottlenecks.push(Bottleneck {
            severity: 0.8,
            description: "No lava or kimberlite \u{2014} no aureole zones possible".into(),
        });
    }

    bottlenecks
}
