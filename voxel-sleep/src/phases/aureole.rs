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
use crate::systems::groundwater::ambient_moisture;
use crate::manifest::ChangeManifest;
use crate::util::{FACE_OFFSETS, sample_material, set_voxel_synced, grow_vein, VeinGrowthParams, VeinBias, default_vein_bias};
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
    /// Debug: lava zone centroids and BFS depths (voxel coords) for visualization.
    pub debug_zones: Vec<(i32, i32, i32, i32)>, // (cx, cy, cz, depth)
    /// Debug: detailed zone placement log lines for profile report.
    pub debug_lines: Vec<String>,
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

/// Determine whether the zone is predominantly limestone-hosted or slate-hosted
/// by sampling face-neighbors of lava cells directly (avoids centroid-in-air problem).
fn determine_aureole_type(
    zone: &LavaZone,
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
) -> AureoleType {
    let mut limestone_count = 0u32;
    let mut other_count = 0u32;

    // Sample up to 200 cells for perf
    let limit = zone.cells.len().min(200);
    for &(cx, cy, cz) in &zone.cells[..limit] {
        for &(dx, dy, dz) in &FACE_OFFSETS {
            if let Some(mat) = sample_material(density_fields, cx + dx, cy + dy, cz + dz, chunk_size) {
                if mat == Material::Limestone {
                    limestone_count += 1;
                } else if mat.is_host_rock() && mat != Material::Limestone {
                    other_count += 1;
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

/// Compute water boost multiplier by counting water cells near lava cells.
/// Uses configurable search radius and max cell cap for exposed scaling.
fn compute_water_boost(
    zone: &LavaZone,
    fluid_snapshot: &FluidSnapshot,
    config: &AureoleConfig,
) -> (f32, f32) {
    let cs = fluid_snapshot.chunk_size;
    let mut water_count = 0u32;
    let search_r = config.aureole_water_search_radius.max(1) as i32;
    let max_cells = config.aureole_water_max_cells.max(1);

    // Deduplicate: only check unique positions within search_r of any lava cell
    let mut checked: HashSet<(i32, i32, i32)> = HashSet::new();
    let lava_set: HashSet<(i32, i32, i32)> = zone.cells.iter().copied().collect();

    for &(lx, ly, lz) in &zone.cells {
        // Search within radius around each lava cell
        for wdx in -search_r..=search_r {
            for wdy in -search_r..=search_r {
                for wdz in -search_r..=search_r {
                    if wdx * wdx + wdy * wdy + wdz * wdz > search_r * search_r { continue; }
                    let wp = (lx + wdx, ly + wdy, lz + wdz);
                    if lava_set.contains(&wp) { continue; }
                    if !checked.insert(wp) { continue; }

                    let fck = (
                        wp.0.div_euclid(cs as i32),
                        wp.1.div_euclid(cs as i32),
                        wp.2.div_euclid(cs as i32),
                    );
                    let flx = wp.0.rem_euclid(cs as i32) as usize;
                    let fly = wp.1.rem_euclid(cs as i32) as usize;
                    let flz = wp.2.rem_euclid(cs as i32) as usize;
                    let fidx = flz * cs * cs + fly * cs + flx;
                    if let Some(cells) = fluid_snapshot.chunks.get(&fck) {
                        if fidx < cells.len() && cells[fidx].fluid_type.is_water() && cells[fidx].level > 0.001 {
                            water_count += 1;
                            if water_count >= max_cells { break; }
                        }
                    }
                }
                if water_count >= max_cells { break; }
            }
            if water_count >= max_cells { break; }
        }
        if water_count >= max_cells { break; }
    }

    let water_frac = (water_count.min(max_cells) as f32) / (max_cells as f32);
    // Two return values: legacy boost (for shell radius) and deposit multiplier
    let legacy_boost = 1.0 + water_frac * config.water_boost_max;
    let deposit_mult = 1.0 + water_frac * config.aureole_water_deposit_mult;
    (legacy_boost, deposit_mult)
}

// ──────────────────────────────────────────────────────────────
// Metamorphic Shell Placement (multi-source BFS from lava cells)
// ──────────────────────────────────────────────────────────────

/// Place metamorphic shell via BFS from every lava cell outward into solid rock.
/// Limestone → Skarn, other host rock → Hornfels. Air gaps block propagation.
/// Returns (hornfels_count, skarn_count, set of converted world positions).
///
/// Uses `set_voxel_synced` so overlapping boundary voxels in adjacent chunks
/// are updated immediately, preventing `sync_boundary_density` from reverting
/// material-only changes at chunk boundaries.
fn place_metamorphic_shell(
    zone: &LavaZone,
    max_depth: i32,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
    manifest: &mut ChangeManifest,
) -> (u32, u32, HashSet<(i32, i32, i32)>) {
    let mut hornfels_count = 0u32;
    let mut skarn_count = 0u32;
    let mut converted: HashSet<(i32, i32, i32)> = HashSet::new();

    // Build lava position set for O(1) lookup
    let lava_set: HashSet<(i32, i32, i32)> = zone.cells.iter().copied().collect();

    // Multi-source BFS: seed with all lava cells at distance 0
    let mut queue: VecDeque<((i32, i32, i32), i32)> = VecDeque::new();
    let mut visited: HashSet<(i32, i32, i32)> = HashSet::new();
    for &pos in &zone.cells {
        queue.push_back((pos, 0));
        visited.insert(pos);
    }

    while let Some((pos, dist)) = queue.pop_front() {
        for &(dx, dy, dz) in &FACE_OFFSETS {
            let n = (pos.0 + dx, pos.1 + dy, pos.2 + dz);
            if !visited.insert(n) {
                continue;
            }
            if lava_set.contains(&n) {
                continue; // already a lava cell
            }
            let next_dist = dist + 1;
            if next_dist > max_depth {
                continue;
            }

            let (key, lx, ly, lz) = world_to_chunk_local(n.0, n.1, n.2, chunk_size);
            let (mat, density) = match density_fields.get(&key) {
                Some(df) => {
                    let s = df.get(lx, ly, lz);
                    (s.material, s.density)
                }
                None => continue,
            };

            if !mat.is_solid() {
                // Air/non-solid: don't enqueue (aureole doesn't cross air gaps)
                continue;
            }

            if mat == Material::Hornfels || mat == Material::Skarn {
                // Already metamorphosed — continue BFS through but don't re-convert
                queue.push_back((n, next_dist));
                continue;
            }

            let new_mat = if mat == Material::Limestone {
                Material::Skarn
            } else if mat.is_host_rock() {
                Material::Hornfels
            } else {
                // Non-host-rock solid (ore, etc.) — block BFS
                continue;
            };

            // Convert with boundary sync
            set_voxel_synced(density_fields, key, lx, ly, lz, new_mat, None, chunk_size);
            manifest.record_voxel_change(key, lx, ly, lz, mat, density, new_mat, density);
            converted.insert(n);

            if new_mat == Material::Skarn {
                skarn_count += 1;
            } else {
                hornfels_count += 1;
            }

            queue.push_back((n, next_dist));
        }
    }

    (hornfels_count, skarn_count, converted)
}

// ──────────────────────────────────────────────────────────────
// Aureole Boundary Seed Finding
// ──────────────────────────────────────────────────────────────

/// Find vein seed positions at the aureole boundary: converted voxels that have
/// at least one air face-neighbor (visible) AND at least one unconverted host-rock
/// face-neighbor (vein can grow into). Seeds are placed where players will see them.
fn find_aureole_boundary_seeds(
    converted: &HashSet<(i32, i32, i32)>,
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
    count: usize,
    rng: &mut ChaCha8Rng,
    spread: f32,
) -> Vec<(i32, i32, i32)> {
    // Collect boundary candidates with air-neighbor count for weighting
    let mut candidates: Vec<((i32, i32, i32), u32)> = Vec::new();

    for &pos in converted {
        let mut air_count = 0u32;
        let mut has_host = false;
        for &(dx, dy, dz) in &FACE_OFFSETS {
            let n = (pos.0 + dx, pos.1 + dy, pos.2 + dz);
            if let Some(mat) = sample_material(density_fields, n.0, n.1, n.2, chunk_size) {
                if !mat.is_solid() {
                    air_count += 1;
                } else if mat.is_host_rock()
                    && mat != Material::Hornfels
                    && mat != Material::Skarn
                    && !converted.contains(&n)
                {
                    has_host = true;
                }
            }
        }
        if air_count >= 1 && has_host {
            candidates.push((pos, air_count));
        }
    }

    // Sort for determinism
    candidates.sort_by(|a, b| a.0.cmp(&b.0));

    if candidates.is_empty() {
        return Vec::new();
    }

    if count >= candidates.len() {
        return candidates.into_iter().map(|(pos, _)| pos).collect();
    }

    // Weighted random selection with spread-based repulsion
    let mut selected: Vec<(i32, i32, i32)> = Vec::with_capacity(count);
    let mut remaining = candidates;
    for _ in 0..count {
        if remaining.is_empty() {
            break;
        }
        // Compute weights: base air-count + spread repulsion from already-selected
        let weights: Vec<f32> = remaining.iter().map(|&(pos, air_w)| {
            let mut w = air_w as f32;
            if spread > 0.0 && !selected.is_empty() {
                // Find min distance to any selected seed
                let min_dist = selected.iter().map(|&s| {
                    let dx = (pos.0 - s.0) as f32;
                    let dy = (pos.1 - s.1) as f32;
                    let dz = (pos.2 - s.2) as f32;
                    (dx * dx + dy * dy + dz * dz).sqrt()
                }).fold(f32::MAX, f32::min);
                // Boost weight by distance (further = better) scaled by spread factor
                w *= 1.0 + spread * min_dist * 0.5;
            }
            w.max(0.01)
        }).collect();
        let total_weight: f32 = weights.iter().sum();
        if total_weight <= 0.0 {
            break;
        }
        let mut roll = rng.gen::<f32>() * total_weight;
        let mut chosen = 0;
        for (i, &w) in weights.iter().enumerate() {
            roll -= w;
            if roll <= 0.0 {
                chosen = i;
                break;
            }
        }
        let (pos, _) = remaining.remove(chosen);
        selected.push(pos);
    }

    selected
}

// ──────────────────────────────────────────────────────────────
// Ore Vein + Pocket Placement
// ──────────────────────────────────────────────────────────────

/// Write vein voxels into density fields, returns count of voxels placed.
/// Uses boundary-synced writes to prevent chunk-edge seams.
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

        set_voxel_synced(density_fields, key, lx, ly, lz, material, None, chunk_size);
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
    converted: &HashSet<(i32, i32, i32)>,
    config: &AureoleConfig,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
    manifest: &mut ChangeManifest,
    rng: &mut ChaCha8Rng,
    skarn_count: u32,
    deposit_mult: f32,
    count_mult: f32,
) -> u32 {
    let vein_count = ((config.aureole_vein_count as f32 * count_mult).round() as usize).max(1);
    let seeds = find_aureole_boundary_seeds(converted, density_fields, chunk_size, vein_count, rng, config.aureole_vein_spread);
    if seeds.is_empty() {
        return 0;
    }

    let ores = [Material::Copper, Material::Iron, Material::Tin];
    let skip = rng.gen_range(0..3usize);
    let ore_a = ores[(skip + 1) % 3];
    let ore_b = ores[(skip + 2) % 3];

    let vein_min = ((config.aureole_vein_min as f32 * deposit_mult).round() as u32).max(2);
    let vein_max = ((config.aureole_vein_max as f32 * deposit_mult).round() as u32).max(vein_min + 1);
    let small_min = ((config.small_vein_base_size as f32 * deposit_mult).round() as u32).max(2);
    let small_max = ((small_min as f32 * 1.5).round() as u32).max(small_min + 1);

    let mut total_placed = 0u32;

    for (i, &seed) in seeds.iter().enumerate() {
        // First 60% ore_a, next 20% ore_b, last 20% pyrite
        let (ore, min_sz, max_sz) = if (i as f32) < (seeds.len() as f32 * 0.6) {
            (ore_a, vein_min, vein_max)
        } else if (i as f32) < (seeds.len() as f32 * 0.8) {
            (ore_b, vein_min, vein_max)
        } else {
            (Material::Pyrite, small_min, small_max)
        };
        let (actual_min, actual_max, bias) = if config.aureole_wall_climbing {
            // Compute target from geometry params (like hydrothermal)
            let height = rng.gen_range(config.aureole_climb_height_min..=config.aureole_climb_height_max);
            let width = rng.gen_range(config.aureole_wall_width_min..=config.aureole_wall_width_max);
            let depth = rng.gen_range(config.aureole_rock_depth_min..=config.aureole_rock_depth_max);
            let geo_target = (((height * width * depth) as f32 / 4.0) * deposit_mult).round() as u32;
            let geo_target = geo_target.max(4).min(120);
            // Find wall normal from seed
            let wall_normal = FACE_OFFSETS.iter()
                .find(|&&(dx, dy, dz)| {
                    sample_material(density_fields, seed.0 + dx, seed.1 + dy, seed.2 + dz, chunk_size)
                        .map_or(false, |m| !m.is_solid())
                })
                .copied()
                .unwrap_or((0, 1, 0));
            ((geo_target * 8) / 10, geo_target, VeinBias::WallClimbing {
                        wall_normal,
                        weight_up: config.aureole_weight_up,
                        weight_into: config.aureole_weight_into,
                        weight_lateral: config.aureole_weight_lateral,
                        weight_down: config.aureole_weight_down,
                        weight_toward_air: config.aureole_weight_toward_air,
                    })
        } else {
            (min_sz, max_sz, default_vein_bias(ore, rng))
        };
        let params = VeinGrowthParams { ore, min_size: actual_min, max_size: actual_max, bias, exclude_aureole: false };
        let positions = grow_vein(density_fields, seed, &params, chunk_size, rng);
        total_placed += apply_vein_to_world(&positions, ore, density_fields, chunk_size, manifest);
    }

    // Bonus: if skarn exists (slate aureole reached limestone), place compact Garnet/Diopside pockets
    if skarn_count > 0 {
        let mut skarn_seeds: Vec<(i32, i32, i32)> = converted.iter()
            .filter(|&&pos| {
                sample_material(density_fields, pos.0, pos.1, pos.2, chunk_size)
                    .map_or(false, |m| m == Material::Skarn)
            })
            .copied()
            .collect();
        skarn_seeds.sort();
        if !skarn_seeds.is_empty() {
            let g_size = ((config.garnet_compact_size as f32 * deposit_mult).round() as u32).max(3);
            for _ in 0..config.garnet_pocket_count {
                let garnet_seed = skarn_seeds[rng.gen_range(0..skarn_seeds.len())];
                let params = VeinGrowthParams {
                    ore: Material::Garnet,
                    min_size: (g_size * 8) / 10,
                    max_size: g_size,
                    bias: VeinBias::Compact,
                    exclude_aureole: false,
                };
                let positions = grow_vein(density_fields, garnet_seed, &params, chunk_size, rng);
                total_placed += apply_vein_to_world(&positions, Material::Garnet, density_fields, chunk_size, manifest);
            }

            let d_size = ((config.diopside_compact_size as f32 * deposit_mult).round() as u32).max(3);
            for _ in 0..config.diopside_pocket_count {
                let diopside_seed = skarn_seeds[rng.gen_range(0..skarn_seeds.len())];
                let params = VeinGrowthParams {
                    ore: Material::Diopside,
                    min_size: (d_size * 8) / 10,
                    max_size: d_size,
                    bias: VeinBias::Compact,
                    exclude_aureole: false,
                };
                let positions = grow_vein(density_fields, diopside_seed, &params, chunk_size, rng);
                total_placed += apply_vein_to_world(&positions, Material::Diopside, density_fields, chunk_size, manifest);
            }
        }
    }

    total_placed
}

/// Place ore veins for a Limestone-hosted (Skarn) aureole zone.
fn place_limestone_veins(
    converted: &HashSet<(i32, i32, i32)>,
    config: &AureoleConfig,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
    manifest: &mut ChangeManifest,
    rng: &mut ChaCha8Rng,
    deposit_mult: f32,
    count_mult: f32,
) -> u32 {
    // Ore veins at boundary seeds
    let ore_count = ((config.aureole_vein_count as f32 * count_mult * 0.6).round() as usize).max(1);
    let seeds = find_aureole_boundary_seeds(converted, density_fields, chunk_size, ore_count, rng, config.aureole_vein_spread);
    if seeds.is_empty() {
        return 0;
    }

    let ores = [Material::Copper, Material::Iron, Material::Gold];
    let skip = rng.gen_range(0..3usize);
    let ore_a = ores[(skip + 1) % 3];
    let ore_b = ores[(skip + 2) % 3];

    let vein_min = ((config.aureole_vein_min as f32 * deposit_mult).round() as u32).max(2);
    let vein_max = ((config.aureole_vein_max as f32 * deposit_mult).round() as u32).max(vein_min + 1);

    let mut total_placed = 0u32;

    for (i, &seed) in seeds.iter().enumerate() {
        let ore = if (i as f32) < (seeds.len() as f32 * 0.6) { ore_a } else { ore_b };
        let (actual_min, actual_max, bias) = if config.aureole_wall_climbing {
            let height = rng.gen_range(config.aureole_climb_height_min..=config.aureole_climb_height_max);
            let width = rng.gen_range(config.aureole_wall_width_min..=config.aureole_wall_width_max);
            let depth = rng.gen_range(config.aureole_rock_depth_min..=config.aureole_rock_depth_max);
            let geo_target = (((height * width * depth) as f32 / 4.0) * deposit_mult).round() as u32;
            let geo_target = geo_target.max(4).min(120);
            let wall_normal = FACE_OFFSETS.iter()
                .find(|&&(dx, dy, dz)| {
                    sample_material(density_fields, seed.0 + dx, seed.1 + dy, seed.2 + dz, chunk_size)
                        .map_or(false, |m| !m.is_solid())
                })
                .copied()
                .unwrap_or((0, 1, 0));
            ((geo_target * 8) / 10, geo_target, VeinBias::WallClimbing {
                        wall_normal,
                        weight_up: config.aureole_weight_up,
                        weight_into: config.aureole_weight_into,
                        weight_lateral: config.aureole_weight_lateral,
                        weight_down: config.aureole_weight_down,
                        weight_toward_air: config.aureole_weight_toward_air,
                    })
        } else {
            (vein_min, vein_max, default_vein_bias(ore, rng))
        };
        let params = VeinGrowthParams { ore, min_size: actual_min, max_size: actual_max, bias, exclude_aureole: false };
        let positions = grow_vein(density_fields, seed, &params, chunk_size, rng);
        total_placed += apply_vein_to_world(&positions, ore, density_fields, chunk_size, manifest);
    }

    // Compact Garnet + Diopside pockets (placed into skarn zones)
    let mut skarn_seeds: Vec<(i32, i32, i32)> = converted.iter()
        .filter(|&&pos| {
            sample_material(density_fields, pos.0, pos.1, pos.2, chunk_size)
                .map_or(false, |m| m == Material::Skarn)
        })
        .copied()
        .collect();
    skarn_seeds.sort();
    if !skarn_seeds.is_empty() {
        let g_size = ((config.garnet_compact_size as f32 * deposit_mult).round() as u32).max(3);
        for _ in 0..config.garnet_pocket_count {
            let seed = skarn_seeds[rng.gen_range(0..skarn_seeds.len())];
            let params = VeinGrowthParams {
                ore: Material::Garnet,
                min_size: (g_size * 8) / 10,
                max_size: g_size,
                bias: VeinBias::Compact,
                exclude_aureole: false,
            };
            let positions = grow_vein(density_fields, seed, &params, chunk_size, rng);
            total_placed += apply_vein_to_world(&positions, Material::Garnet, density_fields, chunk_size, manifest);
        }

        let d_size = ((config.diopside_compact_size as f32 * deposit_mult).round() as u32).max(3);
        for _ in 0..config.diopside_pocket_count {
            let seed = skarn_seeds[rng.gen_range(0..skarn_seeds.len())];
            let params = VeinGrowthParams {
                ore: Material::Diopside,
                min_size: (d_size * 8) / 10,
                max_size: d_size,
                bias: VeinBias::Compact,
                exclude_aureole: false,
            };
            let positions = grow_vein(density_fields, seed, &params, chunk_size, rng);
            total_placed += apply_vein_to_world(&positions, Material::Diopside, density_fields, chunk_size, manifest);
        }
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
            // Compute BFS depth from zone size using ln() for sensible shell thicknesses
            // 5 cells→2, 50→4, 200→5, 958→7 voxels
            let cell_count = zone.cells.len() as f32;
            let base_depth = (cell_count.ln().max(1.0) * config.radius_scale * config.heat_multiplier)
                .min(config.max_radius)
                .max(2.0);
            let (water_boost, water_deposit_mult) = compute_water_boost(zone, fluid_snapshot, config);
            let final_depth = (base_depth * water_boost).ceil() as i32;

            // Lava volume scaling: fraction of zone cells vs max_cells cap
            let lava_max = config.aureole_lava_volume_max_cells.max(1);
            let lava_frac = (cell_count.min(lava_max as f32)) / (lava_max as f32);
            let lava_deposit_mult = 1.0 + lava_frac * config.aureole_lava_deposit_mult;
            let lava_count_mult = 1.0 + lava_frac * config.aureole_lava_count_mult;
            let combined_deposit_mult = lava_deposit_mult * water_deposit_mult;
            let combined_count_mult = lava_count_mult;

            if final_depth < 1 {
                continue;
            }

            let aureole_type = determine_aureole_type(zone, density_fields, chunk_size);

            // Pass 1: metamorphic shell via BFS from lava cells
            let (hornfels_n, skarn_n, converted) = place_metamorphic_shell(
                zone,
                final_depth,
                density_fields,
                chunk_size,
                &mut result.manifest,
            );
            result.hornfels_placed += hornfels_n;
            result.skarn_placed += skarn_n;
            result.voxels_metamorphosed += hornfels_n + skarn_n;

            // Record debug zone info (centroid in voxel coords + BFS depth)
            result.debug_zones.push((zone.centroid.0, zone.centroid.1, zone.centroid.2, final_depth));

            // Compute lava extent and hornfels placement extent for diagnostics
            if !zone.cells.is_empty() {
                let (mut lmin, mut lmax) = (zone.cells[0], zone.cells[0]);
                for &c in &zone.cells {
                    lmin = (lmin.0.min(c.0), lmin.1.min(c.1), lmin.2.min(c.2));
                    lmax = (lmax.0.max(c.0), lmax.1.max(c.1), lmax.2.max(c.2));
                }
                let (mut hmin, mut hmax) = ((i32::MAX, i32::MAX, i32::MAX), (i32::MIN, i32::MIN, i32::MIN));
                for &c in &converted {
                    hmin = (hmin.0.min(c.0), hmin.1.min(c.1), hmin.2.min(c.2));
                    hmax = (hmax.0.max(c.0), hmax.1.max(c.1), hmax.2.max(c.2));
                }
                result.debug_lines.push(format!(
                    "[ZONE_DIAG] zone_idx={} cells={} depth={} centroid=({},{},{}) lava_min=({},{},{}) lava_max=({},{},{}) hornfels={} skarn={} placed_min=({},{},{}) placed_max=({},{},{})",
                    result.debug_zones.len() - 1, zone.cells.len(), final_depth,
                    zone.centroid.0, zone.centroid.1, zone.centroid.2,
                    lmin.0, lmin.1, lmin.2, lmax.0, lmax.1, lmax.2,
                    hornfels_n, skarn_n,
                    hmin.0, hmin.1, hmin.2, hmax.0, hmax.1, hmax.2,
                ));
                // Parseable bounding boxes for UE debug visualization
                if hornfels_n + skarn_n > 0 {
                    // Placement extent (where hornfels/skarn was actually placed)
                    result.debug_lines.push(format!(
                        "[AUREOLE_BOX] {} {} {} {} {} {}",
                        hmin.0, hmin.1, hmin.2, hmax.0, hmax.1, hmax.2,
                    ));
                    // Lava extent (inner zone)
                    result.debug_lines.push(format!(
                        "[LAVA_BOX] {} {} {} {} {} {}",
                        lmin.0, lmin.1, lmin.2, lmax.0, lmax.1, lmax.2,
                    ));
                    // Zone centroid as a single point (for precision alignment check)
                    result.debug_lines.push(format!(
                        "[CENTROID_PT] {} {} {}",
                        zone.centroid.0, zone.centroid.1, zone.centroid.2,
                    ));
                }
            }

            if result.glimpse_chunk.is_none() && (hornfels_n > 0 || skarn_n > 0) {
                let (key, _, _, _) = world_to_chunk_local(
                    zone.centroid.0, zone.centroid.1, zone.centroid.2, chunk_size,
                );
                result.glimpse_chunk = Some(key);
            }

            // Pass 2: ore veins + pockets (grow into just-placed metamorphic rock)
            let veins_placed = match aureole_type {
                AureoleType::Slate => place_slate_veins(
                    &converted, config,
                    density_fields, chunk_size, &mut result.manifest, rng, skarn_n,
                    combined_deposit_mult, combined_count_mult,
                ),
                AureoleType::Limestone => place_limestone_veins(
                    &converted, config,
                    density_fields, chunk_size, &mut result.manifest, rng,
                    combined_deposit_mult, combined_count_mult,
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
        let new_density = if c.new_material == Material::Air { -1.0 } else { c.density };
        set_voxel_synced(density_fields, c.chunk_key, c.lx, c.ly, c.lz, c.new_material, Some(new_density), chunk_size);

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
