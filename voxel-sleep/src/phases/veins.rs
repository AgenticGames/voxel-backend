//! Phase 3: "The Veins" — 500,000 years.
//!
//! THE GAMEPLAY PAYOFF: Water-heat convergence hydrothermal ore deposition.
//! Heat sources within convergence_radius of water "activate" that water.
//! Veins grow upward from activated water onto solid walls/ceilings as
//! visible climbing streaks that thicken into the rock.
//! Also: cave formation growth (crystal, calcite, flowstone).

use std::collections::{HashMap, HashSet};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use voxel_core::density::DensityField;
use voxel_core::material::Material;
use voxel_core::stress::world_to_chunk_local;
use voxel_fluid::FluidSnapshot;


use crate::phases::aureole::HeatMap;
use crate::config::{VeinConfig, GroundwaterConfig};
use crate::systems::groundwater::ambient_moisture;
use crate::manifest::ChangeManifest;
use crate::util::{FACE_OFFSETS, sample_material, set_voxel_synced, count_neighbors, grow_vein, VeinGrowthParams, VeinBias};
use crate::{Bottleneck, PhaseDiagnostics, ResourceCensus, TransformEntry};

/// Result of the veins phase.
#[derive(Debug, Default)]
pub struct VeinResult {
    pub veins_deposited: u32,
    pub formations_grown: u32,
    pub manifest: ChangeManifest,
    pub glimpse_chunk: Option<(i32, i32, i32)>,
    /// Exact world voxel position of the most intense vein deposit
    pub glimpse_pos: Option<(i32, i32, i32)>,
    pub transform_log: Vec<TransformEntry>,
    pub diagnostics: PhaseDiagnostics,
}

// ──────────────────────────────────────────────────────────────
// Temperature zones & helper types
// ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
enum TemperatureZone { Hypothermal, Mesothermal, Epithermal }

struct ActivatedWater {
    pos: (i32, i32, i32),
    heat_pos: (i32, i32, i32),
}

struct WallSite {
    pos: (i32, i32, i32),
    wall_normal: (i32, i32, i32),
    host_rock: Material,
    heat_score: f32,
}

/// Select ore type by temperature zone and host rock (replaces old distance-based selection).
fn select_ore_by_zone_and_host(
    config: &VeinConfig,
    zone: TemperatureZone,
    host_rock: Material,
    rng: &mut ChaCha8Rng,
) -> Material {
    // Ore selection by temperature zone and host rock.
    // Each zone has a DISTINCT signature so the player can read what they're looking at:
    //   Hypothermal (deep/hot):  Tin + Iron dominant — the "industrial metals" zone
    //   Mesothermal (mid):       Copper + Quartz dominant — the "mid-depth" zone
    //   Epithermal (shallow/cool): Gold + Sulfide dominant — the "precious metals" zone
    if config.host_rock_ore_enabled {
        match host_rock {
            Material::Limestone | Material::Garnet | Material::Diopside | Material::Marble => {
                match zone {
                    TemperatureZone::Hypothermal => {
                        // Tin 35%, Iron 35%, Copper 15%, Pyrite 15%
                        let r = rng.gen::<f32>();
                        if r < 0.35 { return Material::Tin; }
                        else if r < 0.70 { return Material::Iron; }
                        else if r < 0.85 { return Material::Copper; }
                        else { return Material::Pyrite; }
                    }
                    TemperatureZone::Mesothermal => {
                        // Copper 40%, Quartz 25%, Iron 15%, Pyrite 15%, Tin 5%
                        let r = rng.gen::<f32>();
                        if r < 0.40 { return Material::Copper; }
                        else if r < 0.65 { return Material::Quartz; }
                        else if r < 0.80 { return Material::Iron; }
                        else if r < 0.95 { return Material::Pyrite; }
                        else { return Material::Tin; }
                    }
                    TemperatureZone::Epithermal => {
                        // Gold 30%, Sulfide 35%, Pyrite 25%, Quartz 10%
                        let r = rng.gen::<f32>();
                        if r < 0.30 { return Material::Gold; }
                        else if r < 0.65 { return Material::Sulfide; }
                        else if r < 0.90 { return Material::Pyrite; }
                        else { return Material::Quartz; }
                    }
                }
            },
            Material::Slate | Material::Hornfels => {
                match zone {
                    TemperatureZone::Hypothermal => {
                        // Tin 40%, Iron 30%, Copper 15%, Pyrite 15%
                        let r = rng.gen::<f32>();
                        if r < 0.40 { return Material::Tin; }
                        else if r < 0.70 { return Material::Iron; }
                        else if r < 0.85 { return Material::Copper; }
                        else { return Material::Pyrite; }
                    }
                    TemperatureZone::Mesothermal => {
                        // Copper 35%, Quartz 25%, Iron 15%, Tin 10%, Pyrite 15%
                        let r = rng.gen::<f32>();
                        if r < 0.35 { return Material::Copper; }
                        else if r < 0.60 { return Material::Quartz; }
                        else if r < 0.75 { return Material::Iron; }
                        else if r < 0.85 { return Material::Tin; }
                        else { return Material::Pyrite; }
                    }
                    TemperatureZone::Epithermal => {
                        // Gold 25%, Sulfide 30%, Quartz 20%, Pyrite 25%
                        let r = rng.gen::<f32>();
                        if r < 0.25 { return Material::Gold; }
                        else if r < 0.55 { return Material::Sulfide; }
                        else if r < 0.75 { return Material::Quartz; }
                        else { return Material::Pyrite; }
                    }
                }
            },
            _ => {},
        }
    }

    // Default behavior (all other hosts — Granite/Sandstone/Basalt)
    match zone {
        TemperatureZone::Hypothermal => {
            // Tin 40%, Iron 30%, Pyrite 30%
            let r = rng.gen::<f32>();
            if r < 0.40 { Material::Tin }
            else if r < 0.70 { Material::Iron }
            else { Material::Pyrite }
        }
        TemperatureZone::Mesothermal => {
            // Copper 40%, Quartz 30%, Pyrite 30%
            let r = rng.gen::<f32>();
            if r < 0.40 { Material::Copper }
            else if r < 0.70 { Material::Quartz }
            else { Material::Pyrite }
        }
        TemperatureZone::Epithermal => {
            // Gold 35%, Sulfide 35%, Pyrite 30%
            let r = rng.gen::<f32>();
            if r < 0.35 { Material::Gold }
            else if r < 0.70 { Material::Sulfide }
            else { Material::Pyrite }
        }
    }
}

/// Check if a material is a host rock (transformable by vein deposition).
fn is_host_rock(mat: Material) -> bool {
    matches!(mat,
        Material::Sandstone | Material::Limestone | Material::Granite |
        Material::Basalt | Material::Slate | Material::Marble |
        Material::Garnet | Material::Diopside
        // Hornfels excluded — aureole metamorphic product, Phase 3 should not start veins here
    )
}

// ──────────────────────────────────────────────────────────────
// Core algorithm
// ──────────────────────────────────────────────────────────────

/// Execute Phase 3: water-heat convergence vein deposition + formation growth.
pub fn apply_veins(
    config: &VeinConfig,
    groundwater: &GroundwaterConfig,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    fluid_snapshot: &mut FluidSnapshot,
    heat_map: &HeatMap,
    chunks: &[(i32, i32, i32)],
    chunk_size: usize,
    rng: &mut ChaCha8Rng,
    census: &ResourceCensus,
) -> VeinResult {
    let mut result = VeinResult::default();
    let field_size = chunk_size + 1;

    struct VeinCandidate {
        chunk_key: (i32, i32, i32),
        lx: usize, ly: usize, lz: usize,
        old_material: Material,
        old_density: f32,
        new_material: Material,
    }

    let mut vein_candidates: Vec<VeinCandidate> = Vec::new();
    let mut theoretical_max = 0u32;
    let mut convergence_count = 0u32;

    // Diagnostic: capture first few water/heat/vein/wall-site positions for debugging
    let mut diag_water_positions: Vec<(i32, i32, i32)> = Vec::new();
    let mut diag_heat_positions: Vec<(i32, i32, i32)> = Vec::new();
    let mut diag_vein_positions: Vec<(i32, i32, i32)> = Vec::new();
    let mut diag_water_raw: Vec<((i32, i32, i32), usize, usize, usize)> = Vec::new();
    let mut diag_wall_sites: Vec<((i32,i32,i32), (i32,i32,i32), Material, f32)> = Vec::new(); // (pos, normal, host, score)
    let mut diag_total_wall_sites: u32 = 0;
    let diag_chunk_size = chunk_size;
    let diag_fluid_cs = fluid_snapshot.chunk_size;

    // --- Water-Heat Convergence Vein Deposition ---
    if config.vein_enabled && !heat_map.is_empty() && !fluid_snapshot.chunks.is_empty() {
        let convergence_radius = config.convergence_radius;
        let convergence_radius_sq = convergence_radius * convergence_radius;

        // 1. Spatial-bucket heat sources by chunk for fast lookup
        let search_chunks = (convergence_radius / chunk_size as f32).ceil() as i32 + 1;
        let mut heat_buckets: HashMap<(i32, i32, i32), Vec<(i32, i32, i32)>> = HashMap::new();
        for heat in heat_map {
            let (hx, hy, hz) = heat.pos;
            let bucket_key = (
                hx.div_euclid(chunk_size as i32),
                hy.div_euclid(chunk_size as i32),
                hz.div_euclid(chunk_size as i32),
            );
            heat_buckets.entry(bucket_key).or_default().push((hx, hy, hz));
        }

        // 2. Find activated water cells
        let mut activated: Vec<ActivatedWater> = Vec::new();
        let mut spacing_grid: HashSet<(i32, i32, i32)> = HashSet::new();
        let spacing = config.convergence_spacing.max(1) as i32;
        let cs = fluid_snapshot.chunk_size;

        // Collect fluid chunk keys and sort for determinism
        let mut fluid_chunk_keys: Vec<(i32, i32, i32)> = fluid_snapshot.chunks.keys().copied().collect();
        fluid_chunk_keys.sort();

        for fchunk in &fluid_chunk_keys {
            let cells = match fluid_snapshot.chunks.get(fchunk) {
                Some(c) => c,
                None => continue,
            };
            let (cx, cy, cz) = *fchunk;
            for z in 0..cs {
                for y in 0..cs {
                    for x in 0..cs {
                        let idx = z * cs * cs + y * cs + x;
                        let cell = &cells[idx];
                        if cell.level <= 0.001 || !cell.fluid_type.is_water() {
                            continue;
                        }

                        let wx = cx * (cs as i32) + x as i32;
                        let wy = cy * (cs as i32) + y as i32;
                        let wz = cz * (cs as i32) + z as i32;

                        // Grid-based spacing check
                        let grid_key = (
                            wx.div_euclid(spacing),
                            wy.div_euclid(spacing),
                            wz.div_euclid(spacing),
                        );
                        if !spacing_grid.insert(grid_key) {
                            continue;
                        }

                        // Search nearby heat buckets for nearest heat source
                        let water_bucket = (
                            wx.div_euclid(chunk_size as i32),
                            wy.div_euclid(chunk_size as i32),
                            wz.div_euclid(chunk_size as i32),
                        );
                        let mut nearest_heat: Option<(i32, i32, i32)> = None;
                        let mut nearest_dist_sq = f32::MAX;

                        for bx in (water_bucket.0 - search_chunks)..=(water_bucket.0 + search_chunks) {
                            for by in (water_bucket.1 - search_chunks)..=(water_bucket.1 + search_chunks) {
                                for bz in (water_bucket.2 - search_chunks)..=(water_bucket.2 + search_chunks) {
                                    if let Some(heats) = heat_buckets.get(&(bx, by, bz)) {
                                        for &hp in heats {
                                            let dx = (hp.0 - wx) as f32;
                                            let dy = (hp.1 - wy) as f32;
                                            let dz = (hp.2 - wz) as f32;
                                            let dist_sq = dx * dx + dy * dy + dz * dz;
                                            if dist_sq <= convergence_radius_sq && dist_sq < nearest_dist_sq {
                                                nearest_dist_sq = dist_sq;
                                                nearest_heat = Some(hp);
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if let Some(heat_pos) = nearest_heat {
                            if diag_water_positions.len() < 3 {
                                diag_water_raw.push((*fchunk, x, y, z));
                                diag_water_positions.push((wx, wy, wz));
                                diag_heat_positions.push(heat_pos);
                            }
                            activated.push(ActivatedWater { pos: (wx, wy, wz), heat_pos });
                        }
                    }
                }
            }
        }

        // Sort activated for determinism (HashMap iteration is nondeterministic)
        activated.sort_by(|a, b| a.pos.cmp(&b.pos));
        convergence_count = activated.len() as u32;

        // Diagnostic: always log water/heat stats even if 0 activations
        {
            let total_water_cells: u32 = fluid_snapshot.chunks.values()
                .map(|cells| cells.iter().filter(|c| c.level > 0.001 && c.fluid_type.is_water()).count() as u32)
                .sum();
            let total_heat = heat_map.len();
            let first_water = fluid_snapshot.chunks.iter()
                .flat_map(|(&ck, cells)| {
                    let cs_i = cs as i32;
                    cells.iter().enumerate().filter(|(_, c)| c.level > 0.001 && c.fluid_type.is_water())
                        .map(move |(idx, _)| {
                            let lx = idx % cs;
                            let ly = (idx / cs) % cs;
                            let lz = idx / (cs * cs);
                            (ck.0 * cs_i + lx as i32, ck.1 * cs_i + ly as i32, ck.2 * cs_i + lz as i32)
                        })
                }).next();
            let first_heat = heat_map.first().map(|h| h.pos);
            let dist = match (first_water, first_heat) {
                (Some(w), Some(h)) => {
                    let dx = (w.0 - h.0) as f32;
                    let dy = (w.1 - h.1) as f32;
                    let dz = (w.2 - h.2) as f32;
                    (dx*dx + dy*dy + dz*dz).sqrt()
                }
                _ => -1.0,
            };
            result.transform_log.push(TransformEntry {
                description: format!(
                    "[VEIN_COORD_DEBUG] PRE-ACTIVATION: water_cells={} heat_sources={} activated={} convergence_radius={:.0} spacing={} first_water={:?} first_heat={:?} dist={:.1}",
                    total_water_cells, total_heat, convergence_count, convergence_radius, spacing, first_water, first_heat, dist
                ),
                count: 1,
            });
        }

        // 3. For each activated water cell, deposit veins above
        let mut global_deposited: HashSet<(i32, i32, i32)> = HashSet::new();
        let zones = [TemperatureZone::Hypothermal, TemperatureZone::Mesothermal, TemperatureZone::Epithermal];

        for aw in &activated {
            let (water_x, water_y, water_z) = aw.pos;
            let (heat_x, _heat_y, heat_z) = aw.heat_pos;

            // XZ direction from water toward heat (for bias scoring)
            let hdx = (heat_x - water_x) as f32;
            let hdz = (heat_z - water_z) as f32;
            let heat_dir_len = (hdx * hdx + hdz * hdz).sqrt();
            let (heat_dir_x, heat_dir_z) = if heat_dir_len > 0.001 {
                (hdx / heat_dir_len, hdz / heat_dir_len)
            } else {
                (0.0, 0.0)
            };

            let spread = config.horizontal_spread as i32;

            // --- Water/Lava volume scaling ---
            // Count nearby water cells for volume bonus
            let water_frac = if config.water_volume_max_cells > 0 && config.water_volume_radius > 0 {
                let wr = config.water_volume_radius as i32;
                let mut water_count = 0u32;
                let wcs = fluid_snapshot.chunk_size as i32;
                for wdx in -wr..=wr {
                    for wdy in -wr..=wr {
                        for wdz in -wr..=wr {
                            if wdx * wdx + wdy * wdy + wdz * wdz > wr * wr { continue; }
                            let sx = water_x + wdx;
                            let sy = water_y + wdy;
                            let sz = water_z + wdz;
                            let fck = (sx.div_euclid(wcs), sy.div_euclid(wcs), sz.div_euclid(wcs));
                            if let Some(cells) = fluid_snapshot.chunks.get(&fck) {
                                let lx = sx.rem_euclid(wcs) as usize;
                                let ly = sy.rem_euclid(wcs) as usize;
                                let lz = sz.rem_euclid(wcs) as usize;
                                let idx = lz * (wcs as usize) * (wcs as usize) + ly * (wcs as usize) + lx;
                                if idx < cells.len() && cells[idx].level > 0.001 && cells[idx].fluid_type.is_water() {
                                    water_count += 1;
                                }
                            }
                        }
                    }
                }
                (water_count.min(config.water_volume_max_cells) as f32) / (config.water_volume_max_cells as f32)
            } else { 0.0 };

            // Count nearby heat sources for lava volume bonus
            let lava_frac = if config.lava_volume_max_cells > 0 && config.lava_volume_radius > 0 {
                let lr = config.lava_volume_radius as i32;
                let lr_sq = lr * lr;
                let mut heat_count = 0u32;
                for heat in heat_map.iter() {
                    let dx = heat.pos.0 - aw.heat_pos.0;
                    let dy = heat.pos.1 - aw.heat_pos.1;
                    let dz = heat.pos.2 - aw.heat_pos.2;
                    if dx * dx + dy * dy + dz * dz <= lr_sq {
                        heat_count += 1;
                        if heat_count >= config.lava_volume_max_cells { break; }
                    }
                }
                (heat_count.min(config.lava_volume_max_cells) as f32) / (config.lava_volume_max_cells as f32)
            } else { 0.0 };

            // Combined scaling multipliers (1.0 base + bonus fraction * max_mult)
            let volume_vein_mult = 1.0 + water_frac * config.water_volume_vein_mult + lava_frac * config.lava_volume_vein_mult;
            let volume_amount_mult = 1.0 + water_frac * config.water_volume_amount_mult + lava_frac * config.lava_volume_amount_mult;

            // Pre-select dominant ore per zone for this convergence area (coherent vein bodies)
            // Sample nearby host rock to determine the dominant ore tables
            let nearby_host = FACE_OFFSETS.iter()
                .filter_map(|&(dx, dy, dz)| sample_material(density_fields, water_x + dx, water_y + dy, water_z + dz, chunk_size))
                .find(|m| is_host_rock(*m))
                .unwrap_or(Material::Slate);
            let dominant_hypo = select_ore_by_zone_and_host(config, TemperatureZone::Hypothermal, nearby_host, rng);
            let dominant_meso = select_ore_by_zone_and_host(config, TemperatureZone::Mesothermal, nearby_host, rng);
            let dominant_epi = select_ore_by_zone_and_host(config, TemperatureZone::Epithermal, nearby_host, rng);

            for &zone in &zones {
                let (y_min, y_max) = match zone {
                    TemperatureZone::Hypothermal => (config.min_vein_height as i32, config.hypothermal_height as i32),
                    TemperatureZone::Mesothermal => (config.hypothermal_height as i32, config.mesothermal_height as i32),
                    TemperatureZone::Epithermal => (config.mesothermal_height as i32, config.epithermal_height as i32),
                };
                if y_min >= y_max { continue; }

                let base_veins = rng.gen_range(config.veins_per_zone_min..=config.veins_per_zone_max);
                let num_veins = ((base_veins as f32 * volume_amount_mult).round() as u32).max(1);
                // No candidate cap — scan box is already bounded by horizontal_spread

                // Find wall sites: scan box above water
                let mut candidates: Vec<WallSite> = Vec::new();
                let scan_x_min = water_x - spread;
                let scan_x_max = water_x + spread;
                let scan_z_min = water_z - spread;
                let scan_z_max = water_z + spread;
                let scan_y_min = water_y + y_min;
                let scan_y_max = water_y + y_max;

                // Check which chunks overlap the scan box and iterate only loaded ones
                let chunk_x_min = scan_x_min.div_euclid(chunk_size as i32);
                let chunk_x_max = scan_x_max.div_euclid(chunk_size as i32);
                let chunk_y_min = scan_y_min.div_euclid(chunk_size as i32);
                let chunk_y_max = scan_y_max.div_euclid(chunk_size as i32);
                let chunk_z_min = scan_z_min.div_euclid(chunk_size as i32);
                let chunk_z_max = scan_z_max.div_euclid(chunk_size as i32);

                // Diagnostic: log scan chunk range for first activated water
                if diag_water_positions.len() <= 1 && diag_wall_sites.is_empty() {
                    let mut scan_found = Vec::new();
                    let mut scan_missing = Vec::new();
                    for sx in chunk_x_min..=chunk_x_max {
                        for sy in chunk_y_min..=chunk_y_max {
                            for sz in chunk_z_min..=chunk_z_max {
                                if density_fields.contains_key(&(sx, sy, sz)) {
                                    scan_found.push((sx, sy, sz));
                                } else {
                                    scan_missing.push((sx, sy, sz));
                                }
                            }
                        }
                    }
                    if !diag_water_positions.is_empty() {
                        // Store as a transform_log entry directly
                        result.transform_log.push(TransformEntry {
                            description: format!(
                                "[VEIN_COORD_DEBUG] scan_range: X=[{}..{}] Y=[{}..{}] Z=[{}..{}] found={} missing={} scan_world: X=[{},{}] Y=[{},{}] Z=[{},{}]",
                                chunk_x_min, chunk_x_max, chunk_y_min, chunk_y_max, chunk_z_min, chunk_z_max,
                                scan_found.len(), scan_missing.len(),
                                scan_x_min, scan_x_max, scan_y_min, scan_y_max, scan_z_min, scan_z_max
                            ),
                            count: 1,
                        });
                    }
                }

                // Build chunk list sorted by distance from water (closest first)
                // so the max_candidates cap doesn't bias toward low-coordinate chunks
                let water_chunk = (
                    water_x.div_euclid(chunk_size as i32),
                    water_y.div_euclid(chunk_size as i32),
                    water_z.div_euclid(chunk_size as i32),
                );
                let mut scan_chunks: Vec<(i32, i32, i32)> = Vec::new();
                for ckx in chunk_x_min..=chunk_x_max {
                    for cky in chunk_y_min..=chunk_y_max {
                        for ckz in chunk_z_min..=chunk_z_max {
                            scan_chunks.push((ckx, cky, ckz));
                        }
                    }
                }
                scan_chunks.sort_by(|a, b| {
                    let da = (a.0 - water_chunk.0).pow(2) + (a.1 - water_chunk.1).pow(2) + (a.2 - water_chunk.2).pow(2);
                    let db = (b.0 - water_chunk.0).pow(2) + (b.1 - water_chunk.1).pow(2) + (b.2 - water_chunk.2).pow(2);
                    da.cmp(&db)
                });

                for &(ckx, cky, ckz) in &scan_chunks {
                            let ck = (ckx, cky, ckz);
                            let df = match density_fields.get(&ck) {
                                Some(df) => df,
                                None => continue,
                            };

                            // Iterate voxels within this chunk that overlap the scan box
                            let local_x_min = (scan_x_min - ckx * chunk_size as i32).max(0) as usize;
                            let local_x_max = ((scan_x_max - ckx * chunk_size as i32) as usize).min(field_size - 1);
                            let local_y_min = (scan_y_min - cky * chunk_size as i32).max(0) as usize;
                            let local_y_max = ((scan_y_max - cky * chunk_size as i32) as usize).min(field_size - 1);
                            let local_z_min = (scan_z_min - ckz * chunk_size as i32).max(0) as usize;
                            let local_z_max = ((scan_z_max - ckz * chunk_size as i32) as usize).min(field_size - 1);

                            for lz in local_z_min..=local_z_max {
                                for ly in local_y_min..=local_y_max {
                                    for lx in local_x_min..=local_x_max {
                                        let sample = df.get(lx, ly, lz);
                                        let mat = sample.material;
                                        if !is_host_rock(mat) { continue; }
                                        if global_deposited.contains(&(
                                            ckx * chunk_size as i32 + lx as i32,
                                            cky * chunk_size as i32 + ly as i32,
                                            ckz * chunk_size as i32 + lz as i32,
                                        )) { continue; }

                                        let wx = ckx * chunk_size as i32 + lx as i32;
                                        let wy = cky * chunk_size as i32 + ly as i32;
                                        let wz = ckz * chunk_size as i32 + lz as i32;

                                        // Check for at least 1 air face neighbor (wall site)
                                        let mut wall_normal = (0i32, 0i32, 0i32);
                                        let mut has_air_face = false;
                                        for &(dx, dy, dz) in &FACE_OFFSETS {
                                            if let Some(nm) = sample_material(density_fields, wx + dx, wy + dy, wz + dz, chunk_size) {
                                                if !nm.is_solid() {
                                                    wall_normal = (dx, dy, dz);
                                                    has_air_face = true;
                                                    break;
                                                }
                                            }
                                        }
                                        if !has_air_face { continue; }

                                        theoretical_max += 1;

                                        // Score: base + heat direction bias
                                        let site_dx = (wx - water_x) as f32;
                                        let site_dz = (wz - water_z) as f32;
                                        let site_len = (site_dx * site_dx + site_dz * site_dz).sqrt();
                                        let dot = if site_len > 0.001 {
                                            (site_dx * heat_dir_x + site_dz * heat_dir_z) / site_len
                                        } else {
                                            0.0
                                        };
                                        // Water proximity: prefer sites closer to directly above water
                                        let xz_dist = ((site_dx * site_dx + site_dz * site_dz).sqrt()).max(1.0);
                                        let proximity_bonus = config.water_proximity_bias / xz_dist;
                                        let heat_score = 1.0 + config.heat_direction_bias * dot + proximity_bonus;

                                        candidates.push(WallSite {
                                            pos: (wx, wy, wz),
                                            wall_normal,
                                            host_rock: mat,
                                            heat_score,
                                        });

                                    }
                                }
                            }
                }

                if candidates.is_empty() { continue; }

                // Select num_veins sites with minimum spacing (15 voxels apart for distinct streaks)
                let min_site_spacing = 15i32;
                let mut selected: Vec<usize> = Vec::new();

                let mut chosen_positions: Vec<(i32, i32, i32)> = Vec::new();
                for _ in 0..num_veins {
                    if candidates.is_empty() { break; }

                    // Weighted random selection with spread-based repulsion
                    let weights: Vec<f32> = candidates.iter().map(|s| {
                        let mut w = s.heat_score.max(0.01);
                        if config.vein_spread > 0.0 && !chosen_positions.is_empty() {
                            let min_dist = chosen_positions.iter().map(|&cp| {
                                let dx = (s.pos.0 - cp.0) as f32;
                                let dy = (s.pos.1 - cp.1) as f32;
                                let dz = (s.pos.2 - cp.2) as f32;
                                (dx * dx + dy * dy + dz * dz).sqrt()
                            }).fold(f32::MAX, f32::min);
                            w *= 1.0 + config.vein_spread * min_dist * 0.3;
                        }
                        w
                    }).collect();
                    let total_weight: f32 = weights.iter().sum();
                    if total_weight <= 0.0 { break; }
                    let mut roll = rng.gen::<f32>() * total_weight;
                    let mut chosen_idx = 0;
                    for (i, &w) in weights.iter().enumerate() {
                        roll -= w;
                        if roll <= 0.0 {
                            chosen_idx = i;
                            break;
                        }
                    }

                    let chosen = candidates.swap_remove(chosen_idx);
                    chosen_positions.push(chosen.pos);
                    selected.push(0); // placeholder, we process immediately
                    diag_total_wall_sites += 1;
                    if diag_wall_sites.len() < 5 {
                        diag_wall_sites.push((chosen.pos, chosen.wall_normal, chosen.host_rock, chosen.heat_score));
                    }

                    // Remove candidates too close to chosen
                    candidates.retain(|s| {
                        let dx = (s.pos.0 - chosen.pos.0).abs();
                        let dy = (s.pos.1 - chosen.pos.1).abs();
                        let dz = (s.pos.2 - chosen.pos.2).abs();
                        dx + dy + dz >= min_site_spacing
                    });

                    // Select ore: 75% dominant for coherent vein bodies, 25% random for variety
                    let dominant = match zone {
                        TemperatureZone::Hypothermal => dominant_hypo,
                        TemperatureZone::Mesothermal => dominant_meso,
                        TemperatureZone::Epithermal => dominant_epi,
                    };
                    let ore = if rng.gen::<f32>() < 0.75 {
                        dominant
                    } else {
                        select_ore_by_zone_and_host(config, zone, chosen.host_rock, rng)
                    };

                    // Epithermal rarity filter
                    if matches!(zone, TemperatureZone::Epithermal) {
                        if rng.gen::<f32>() >= config.epithermal_rarity {
                            continue;
                        }
                    }

                    // Deposition probability check
                    if rng.gen::<f32>() >= config.vein_deposition_prob {
                        continue;
                    }

                    // Vein size from config, scaled by water/lava volume
                    let base_min = ((config.vein_size_min as f32 * volume_vein_mult).round() as u32).max(2);
                    let base_max = ((config.vein_size_max as f32 * volume_vein_mult).round() as u32).max(base_min + 1);

                    let params = VeinGrowthParams {
                        ore,
                        min_size: base_min,
                        max_size: base_max,
                        bias: VeinBias::WallClimbing {
                            wall_normal: chosen.wall_normal,
                            weight_up: config.vein_weight_up,
                            weight_depth: config.vein_weight_depth,
                            weight_lateral: config.vein_weight_lateral,
                            surface_ratio: config.vein_surface_ratio,
                        },
                        exclude_aureole: true,
                    };
                    let vein_positions = grow_vein(density_fields, chosen.pos, &params, chunk_size, rng);

                    for &vpos in &vein_positions {
                        if global_deposited.contains(&vpos) { continue; }
                        let (ck, lx, ly, lz) = world_to_chunk_local(vpos.0, vpos.1, vpos.2, chunk_size);
                        if let Some(df) = density_fields.get(&ck) {
                            let sample = df.get(lx, ly, lz);
                            let old_mat = sample.material;
                            vein_candidates.push(VeinCandidate {
                                chunk_key: ck, lx, ly, lz,
                                old_material: old_mat,
                                old_density: sample.density,
                                new_material: ore,
                            });
                            global_deposited.insert(vpos);
                            if diag_vein_positions.len() < 5 {
                                diag_vein_positions.push(vpos);
                            }
                        }
                    }

                    // Spike/tendril intrusions ("centipede" look)
                    if config.spike_enabled && config.spike_count_max > 0 && !vein_positions.is_empty() {
                        let spike_count = rng.gen_range(config.spike_count_min..=config.spike_count_max);
                        // Collect surface voxels of the vein body (have at least one non-ore neighbor)
                        let vein_set: HashSet<(i32, i32, i32)> = vein_positions.iter().copied().collect();
                        let mut surface_voxels: Vec<(i32, i32, i32)> = Vec::new();
                        for &vp in &vein_positions {
                            for &(dx, dy, dz) in &FACE_OFFSETS {
                                let nb = (vp.0 + dx, vp.1 + dy, vp.2 + dz);
                                if !vein_set.contains(&nb) && !global_deposited.contains(&nb) {
                                    surface_voxels.push(vp);
                                    break;
                                }
                            }
                        }
                        // Pick spike origins spread along the vein
                        let stride = if surface_voxels.len() > spike_count as usize {
                            surface_voxels.len() / spike_count as usize
                        } else { 1 };
                        let mut spike_origins: Vec<(i32, i32, i32)> = Vec::new();
                        for i in (0..surface_voxels.len()).step_by(stride.max(1)) {
                            if spike_origins.len() >= spike_count as usize { break; }
                            spike_origins.push(surface_voxels[i]);
                        }
                        // Grow each spike as a thin tendril into host rock
                        for origin in spike_origins {
                            let spike_len = rng.gen_range(config.spike_length_min..=config.spike_length_max);
                            // Pick a random direction (prefer directions away from vein center & into rock)
                            let mut dirs: Vec<(i32, i32, i32)> = FACE_OFFSETS.iter()
                                .filter(|&&(dx, dy, dz)| {
                                    let nb = (origin.0 + dx, origin.1 + dy, origin.2 + dz);
                                    !vein_set.contains(&nb) && !global_deposited.contains(&nb)
                                })
                                .copied()
                                .collect();
                            if dirs.is_empty() { continue; }
                            let dir_idx = rng.gen_range(0..dirs.len());
                            let spike_dir = dirs[dir_idx];
                            // Walk in the chosen direction with taper
                            let mut pos = origin;
                            for step in 0..spike_len {
                                let next = (pos.0 + spike_dir.0, pos.1 + spike_dir.1, pos.2 + spike_dir.2);
                                if global_deposited.contains(&next) { break; }
                                // Check target is host rock
                                if let Some(mat) = sample_material(density_fields, next.0, next.1, next.2, chunk_size) {
                                    if !is_host_rock(mat) { break; }
                                } else { break; }
                                // Taper check: probability decays each step
                                if step > 0 && rng.gen::<f32>() >= config.spike_taper { break; }
                                let (ck, slx, sly, slz) = world_to_chunk_local(next.0, next.1, next.2, chunk_size);
                                if let Some(df) = density_fields.get(&ck) {
                                    let s = df.get(slx, sly, slz);
                                    vein_candidates.push(VeinCandidate {
                                        chunk_key: ck, lx: slx, ly: sly, lz: slz,
                                        old_material: s.material,
                                        old_density: s.density,
                                        new_material: ore,
                                    });
                                    global_deposited.insert(next);
                                }
                                // Occasionally branch sideways for more organic look
                                if rng.gen::<f32>() < 0.25 {
                                    let perp_dirs: Vec<(i32, i32, i32)> = FACE_OFFSETS.iter()
                                        .filter(|&&(dx, dy, dz)| (dx, dy, dz) != spike_dir && (dx, dy, dz) != (-spike_dir.0, -spike_dir.1, -spike_dir.2))
                                        .copied()
                                        .collect();
                                    if !perp_dirs.is_empty() {
                                        let branch = perp_dirs[rng.gen_range(0..perp_dirs.len())];
                                        let bn = (next.0 + branch.0, next.1 + branch.1, next.2 + branch.2);
                                        if !global_deposited.contains(&bn) {
                                            if let Some(bmat) = sample_material(density_fields, bn.0, bn.1, bn.2, chunk_size) {
                                                if is_host_rock(bmat) {
                                                    let (bck, blx, bly, blz) = world_to_chunk_local(bn.0, bn.1, bn.2, chunk_size);
                                                    if let Some(bdf) = density_fields.get(&bck) {
                                                        let bs = bdf.get(blx, bly, blz);
                                                        vein_candidates.push(VeinCandidate {
                                                            chunk_key: bck, lx: blx, ly: bly, lz: blz,
                                                            old_material: bs.material,
                                                            old_density: bs.density,
                                                            new_material: ore,
                                                        });
                                                        global_deposited.insert(bn);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                pos = next;
                            }
                        }
                    }

                    // Wall-rock alteration: ore-bearing fluid alters adjacent limestone
                    if config.wall_rock_alteration_prob > 0.0
                        && rng.gen::<f32>() < config.wall_rock_alteration_prob
                    {
                        let alter_mat = match zone {
                            TemperatureZone::Hypothermal => {
                                if rng.gen::<f32>() < 0.30 { Material::Diopside } else { Material::Garnet }
                            }
                            TemperatureZone::Mesothermal => {
                                let r = rng.gen::<f32>();
                                if r < 0.25 { Material::Garnet } else { Material::Diopside }
                            }
                            TemperatureZone::Epithermal => {
                                let r = rng.gen::<f32>();
                                if r < 0.20 { Material::Garnet }
                                else if r < 0.80 { Material::Diopside }
                                else { Material::Marble }
                            }
                        };
                        for &(adx, ady, adz) in &FACE_OFFSETS {
                            let anx = chosen.pos.0 + adx;
                            let any = chosen.pos.1 + ady;
                            let anz = chosen.pos.2 + adz;
                            if global_deposited.contains(&(anx, any, anz)) { continue; }
                            if let Some(amat) = sample_material(density_fields, anx, any, anz, chunk_size) {
                                if amat == Material::Limestone {
                                    let (ack, alx, aly, alz) = world_to_chunk_local(anx, any, anz, chunk_size);
                                    if let Some(adf) = density_fields.get(&ack) {
                                        let asample = adf.get(alx, aly, alz);
                                        vein_candidates.push(VeinCandidate {
                                            chunk_key: ack, lx: alx, ly: aly, lz: alz,
                                            old_material: amat,
                                            old_density: asample.density,
                                            new_material: alter_mat,
                                        });
                                        global_deposited.insert((anx, any, anz));
                                    }
                                    break;
                                }
                            }
                        }
                    }

                    // Pyrite co-deposition alongside non-Pyrite ores
                    let pyrite_codeposit_prob = if matches!(chosen.host_rock, Material::Slate | Material::Hornfels) {
                        config.slate_pyrite_codeposit_prob
                    } else {
                        0.10
                    };
                    if ore != Material::Pyrite && rng.gen::<f32>() < pyrite_codeposit_prob {
                        for &(pdx, pdy, pdz) in &FACE_OFFSETS {
                            let pnx = chosen.pos.0 + pdx;
                            let pny = chosen.pos.1 + pdy;
                            let pnz = chosen.pos.2 + pdz;
                            if global_deposited.contains(&(pnx, pny, pnz)) { continue; }
                            if let Some(pmat) = sample_material(density_fields, pnx, pny, pnz, chunk_size) {
                                if is_host_rock(pmat) {
                                    let pyrite_params = VeinGrowthParams {
                                        ore: Material::Pyrite,
                                        min_size: 1,
                                        max_size: 3,
                                        bias: VeinBias::Compact,
                                        exclude_aureole: true,
                                    };
                                    let pyrite_vein = grow_vein(density_fields, (pnx, pny, pnz), &pyrite_params, chunk_size, rng);
                                    for &pvpos in &pyrite_vein {
                                        if global_deposited.contains(&pvpos) { continue; }
                                        let (pck, plx, ply, plz) = world_to_chunk_local(pvpos.0, pvpos.1, pvpos.2, chunk_size);
                                        if let Some(pdf) = density_fields.get(&pck) {
                                            let psample = pdf.get(plx, ply, plz);
                                            vein_candidates.push(VeinCandidate {
                                                chunk_key: pck, lx: plx, ly: ply, lz: plz,
                                                old_material: psample.material,
                                                old_density: psample.density,
                                                new_material: Material::Pyrite,
                                            });
                                            global_deposited.insert(pvpos);
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                    }

                    // Quartz co-deposit for gold in Slate (saddle reef)
                    if matches!(chosen.host_rock, Material::Slate | Material::Hornfels)
                        && ore == Material::Gold
                        && rng.gen::<f32>() < config.slate_quartz_vein_prob
                    {
                        for &(qdx, qdy, qdz) in &FACE_OFFSETS {
                            let qnx = chosen.pos.0 + qdx;
                            let qny = chosen.pos.1 + qdy;
                            let qnz = chosen.pos.2 + qdz;
                            if global_deposited.contains(&(qnx, qny, qnz)) { continue; }
                            if let Some(qmat) = sample_material(density_fields, qnx, qny, qnz, chunk_size) {
                                if is_host_rock(qmat) {
                                    let quartz_params = VeinGrowthParams {
                                        ore: Material::Quartz,
                                        min_size: 1,
                                        max_size: 3,
                                        bias: VeinBias::Planar(rng.gen_range(0..3)),
                                        exclude_aureole: true,
                                    };
                                    let quartz_vein = grow_vein(density_fields, (qnx, qny, qnz), &quartz_params, chunk_size, rng);
                                    for &qvpos in &quartz_vein {
                                        if global_deposited.contains(&qvpos) { continue; }
                                        let (qck, qlx, qly, qlz) = world_to_chunk_local(qvpos.0, qvpos.1, qvpos.2, chunk_size);
                                        if let Some(qdf) = density_fields.get(&qck) {
                                            let qsample = qdf.get(qlx, qly, qlz);
                                            vein_candidates.push(VeinCandidate {
                                                chunk_key: qck, lx: qlx, ly: qly, lz: qlz,
                                                old_material: qsample.material,
                                                old_density: qsample.density,
                                                new_material: Material::Quartz,
                                            });
                                            global_deposited.insert(qvpos);
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }

                let _ = selected; // consumed above
            }
        }
    }

    // Apply vein candidates
    let mut conversions: std::collections::BTreeMap<(u8, u8), u32> = std::collections::BTreeMap::new();
    for c in &vein_candidates {
        *conversions.entry((c.old_material as u8, c.new_material as u8)).or_insert(0) += 1;
        set_voxel_synced(density_fields, c.chunk_key, c.lx, c.ly, c.lz, c.new_material, None, chunk_size);

        result.manifest.record_voxel_change(
            c.chunk_key, c.lx, c.ly, c.lz,
            c.old_material, c.old_density,
            c.new_material, c.old_density,
        );

        if result.glimpse_pos.is_none() {
            result.glimpse_chunk = Some(c.chunk_key);
            // Exact world voxel position of this deposit
            let (cx, cy, cz) = c.chunk_key;
            result.glimpse_pos = Some((
                cx * chunk_size as i32 + c.lx as i32,
                cy * chunk_size as i32 + c.ly as i32,
                cz * chunk_size as i32 + c.lz as i32,
            ));
        }

        result.veins_deposited += 1;
    }

    // --- Cave Formation Growth ---
    struct GrowthCandidate {
        chunk_key: (i32, i32, i32),
        lx: usize, ly: usize, lz: usize,
        new_material: Material,
        growth_type: u8, // 0=crystal, 1=calcite, 2=flowstone
    }

    let mut growth_candidates: Vec<GrowthCandidate> = Vec::new();
    let _chunk_set: HashSet<(i32, i32, i32)> = chunks.iter().copied().collect();

    for &chunk_key in chunks {
        let (cx, cy, cz) = chunk_key;
        let df = match density_fields.get(&chunk_key) {
            Some(df) => df,
            None => continue,
        };

        let mut crystal_count = 0u32;
        let mut calcite_count = 0u32;

        for lz in 0..field_size {
            for ly in 0..field_size {
                for lx in 0..field_size {
                    let sample = df.get(lx, ly, lz);
                    if sample.material != Material::Air {
                        continue;
                    }

                    let wx = cx * (chunk_size as i32) + lx as i32;
                    let wy = cy * (chunk_size as i32) + ly as i32;
                    let wz = cz * (chunk_size as i32) + lz as i32;

                    // Crystal growth: air with 2+ Crystal/Amethyst neighbors
                    if config.crystal_growth_enabled && crystal_count < config.crystal_growth_max_per_chunk {
                        let crystal_neighbors = count_neighbors(
                            density_fields, wx, wy, wz, chunk_size,
                            |m| m == Material::Crystal || m == Material::Amethyst,
                        );
                        if crystal_neighbors >= 2 && rng.gen::<f32>() < config.crystal_growth_prob {
                            growth_candidates.push(GrowthCandidate {
                                chunk_key, lx, ly, lz,
                                new_material: Material::Crystal,
                                growth_type: 0,
                            });
                            crystal_count += 1;
                            continue;
                        }
                    }

                    // Calcite infill: air with 3+ Limestone faces
                    if config.calcite_infill_enabled && calcite_count < config.calcite_infill_max_per_chunk {
                        let ls_neighbors = count_neighbors(
                            density_fields, wx, wy, wz, chunk_size,
                            |m| m == Material::Limestone,
                        );
                        if ls_neighbors >= 3 && rng.gen::<f32>() < config.calcite_infill_prob {
                            growth_candidates.push(GrowthCandidate {
                                chunk_key, lx, ly, lz,
                                new_material: Material::Limestone,
                                growth_type: 1,
                            });
                            calcite_count += 1;
                            continue;
                        }
                    }
                }
            }
        }
    }

    // Flowstone: air along water paths (adjacent to water flow cells)
    if config.flowstone_enabled && !fluid_snapshot.chunks.is_empty() {
        let cs = fluid_snapshot.chunk_size;
        let mut flowstone_per_chunk: HashMap<(i32, i32, i32), u32> = HashMap::new();

        for (&fchunk, cells) in &fluid_snapshot.chunks {
            let (cx, cy, cz) = fchunk;
            for z in 0..cs {
                for y in 0..cs {
                    for x in 0..cs {
                        let idx = z * cs * cs + y * cs + x;
                        let cell = &cells[idx];
                        if cell.level <= 0.001 || !cell.fluid_type.is_water() {
                            continue;
                        }

                        let wx = cx * (cs as i32) + x as i32;
                        let wy = cy * (cs as i32) + y as i32;
                        let wz = cz * (cs as i32) + z as i32;

                        // Check air neighbors for flowstone deposition
                        // Flowstone = calcite precipitation -> requires carbonate host rock nearby
                        for &(dx, dy, dz) in &FACE_OFFSETS {
                            let nx = wx + dx;
                            let ny = wy + dy;
                            let nz = wz + dz;

                            if let Some(mat) = sample_material(density_fields, nx, ny, nz, chunk_size) {
                                if mat == Material::Air && rng.gen::<f32>() < config.flowstone_prob {
                                    // Verify at least one solid neighbor of target air is carbonate
                                    let has_carbonate = FACE_OFFSETS.iter().any(|&(dx2, dy2, dz2)| {
                                        if let Some(nm) = sample_material(density_fields, nx + dx2, ny + dy2, nz + dz2, chunk_size) {
                                            matches!(nm, Material::Limestone | Material::Sandstone | Material::Marble)
                                        } else {
                                            false
                                        }
                                    });
                                    if !has_carbonate { continue; }
                                    let (ck, _, _, _) = world_to_chunk_local(nx, ny, nz, chunk_size);
                                    let count = flowstone_per_chunk.entry(ck).or_insert(0);
                                    if *count < config.flowstone_max_per_chunk {
                                        let (ck2, flx, fly, flz) = world_to_chunk_local(nx, ny, nz, chunk_size);
                                        growth_candidates.push(GrowthCandidate {
                                            chunk_key: ck2, lx: flx, ly: fly, lz: flz,
                                            new_material: Material::Limestone,
                                            growth_type: 2,
                                        });
                                        *count += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // --- Ambient Groundwater Flowstone ---
    // Only limestone/sandstone ceilings produce calcite flowstone.
    // Flowstone = calcite precipitation from carbonate-saturated water.
    let mut ambient_flowstone_count = 0u32;
    if config.flowstone_enabled && groundwater.enabled {
        let mut flowstone_per_chunk: HashMap<(i32, i32, i32), u32> = HashMap::new();
        for &chunk_key in chunks {
            let (cx, cy, cz) = chunk_key;
            let df = match density_fields.get(&chunk_key) {
                Some(df) => df,
                None => continue,
            };
            for lz in 0..field_size {
                for ly in 0..field_size {
                    for lx in 0..field_size {
                        let sample = df.get(lx, ly, lz);
                        if sample.material != Material::Air {
                            continue;
                        }
                        let wx = cx * (chunk_size as i32) + lx as i32;
                        let wy = cy * (chunk_size as i32) + ly as i32;
                        let wz = cz * (chunk_size as i32) + lz as i32;

                        // Only limestone/sandstone ceilings (calcite source)
                        let ceiling_mat = sample_material(density_fields, wx, wy + 1, wz, chunk_size);
                        let ceiling = match ceiling_mat {
                            Some(m) if matches!(m, Material::Limestone | Material::Sandstone) => m,
                            _ => continue,
                        };

                        let moisture = ambient_moisture(groundwater, wy + 1, ceiling, true);
                        if moisture > 0.0 && rng.gen::<f32>() < config.flowstone_prob * moisture * groundwater.flowstone_power * groundwater.soft_rock_mult {
                            let count = flowstone_per_chunk.entry(chunk_key).or_insert(0);
                            if *count < config.flowstone_max_per_chunk {
                                growth_candidates.push(GrowthCandidate {
                                    chunk_key, lx, ly, lz,
                                    new_material: Material::Limestone, // calcite flowstone
                                    growth_type: 2,
                                });
                                *count += 1;
                                ambient_flowstone_count += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    // Apply growth candidates
    let mut formation_total = 0u32;
    let mut crystal_total = 0u32;
    let mut calcite_total = 0u32;
    let mut flowstone_total = 0u32;

    for g in &growth_candidates {
        *conversions.entry((Material::Air as u8, g.new_material as u8)).or_insert(0) += 1;
        let new_density = rng.gen_range(config.growth_density_min..=config.growth_density_max);
        if let Some(df) = density_fields.get_mut(&g.chunk_key) {
            let sample = df.get_mut(g.lx, g.ly, g.lz);
            let old_material = sample.material;
            let old_density = sample.density;
            sample.material = g.new_material;
            sample.density = new_density;

            result.manifest.record_voxel_change(
                g.chunk_key, g.lx, g.ly, g.lz,
                old_material, old_density,
                g.new_material, new_density,
            );
        }

        formation_total += 1;
        match g.growth_type {
            0 => crystal_total += 1,
            1 => calcite_total += 1,
            2 => flowstone_total += 1,
            _ => {}
        }

        if result.glimpse_chunk.is_none() {
            result.glimpse_chunk = Some(g.chunk_key);
        }
    }

    result.formations_grown = formation_total;

    // Build transform log
    if result.veins_deposited > 0 {
        result.transform_log.push(TransformEntry {
            description: format!(
                "The Veins \u{2014} 500,000 years: {} ore voxels deposited from {} convergence zones",
                result.veins_deposited, convergence_count
            ),
            count: result.veins_deposited,
        });
    }
    if formation_total > 0 {
        let mut parts = Vec::new();
        if crystal_total > 0 { parts.push(format!("{} crystal", crystal_total)); }
        if calcite_total > 0 { parts.push(format!("{} calcite", calcite_total)); }
        if flowstone_total > 0 { parts.push(format!("{} flowstone", flowstone_total)); }
        result.transform_log.push(TransformEntry {
            description: format!(
                "The Veins \u{2014} 500,000 years: {} formations grown ({})",
                formation_total, parts.join(", ")
            ),
            count: formation_total,
        });
    }
    if ambient_flowstone_count > 0 {
        result.transform_log.push(TransformEntry {
            description: format!(
                "The Veins \u{2014} 500,000 years: {} flowstone deposited by ambient groundwater",
                ambient_flowstone_count
            ),
            count: ambient_flowstone_count,
        });
    }

    // --- Coordinate Debug ---
    if !diag_water_positions.is_empty() || !diag_wall_sites.is_empty() {
        let mut dbg = format!(
            "[VEIN_COORD_DEBUG] chunk_size={} fluid_cs={} activated_count={} total_wall_sites={} total_veins={}\n",
            diag_chunk_size, diag_fluid_cs, convergence_count, diag_total_wall_sites, result.veins_deposited
        );
        for (i, ((wpos, hpos), raw)) in diag_water_positions.iter()
            .zip(diag_heat_positions.iter())
            .zip(diag_water_raw.iter())
            .enumerate()
        {
            dbg.push_str(&format!(
                "  water[{}]: fluid_chunk=({},{},{}) local=({},{},{}) -> world=({},{},{})\n",
                i, raw.0.0, raw.0.1, raw.0.2, raw.1, raw.2, raw.3, wpos.0, wpos.1, wpos.2
            ));
            dbg.push_str(&format!(
                "  heat[{}]: world=({},{},{})\n",
                i, hpos.0, hpos.1, hpos.2
            ));
        }
        // Scan box info
        if let Some(first_water) = diag_water_positions.first() {
            let spread = config.horizontal_spread as i32;
            dbg.push_str(&format!(
                "  scan_box: X=[{},{}] Z=[{},{}] (spread={})\n",
                first_water.0 - spread, first_water.0 + spread,
                first_water.2 - spread, first_water.2 + spread, spread
            ));
        }
        for (i, ws) in diag_wall_sites.iter().enumerate() {
            dbg.push_str(&format!(
                "  wall_site[{}]: world=({},{},{}) normal=({},{},{}) host={:?} score={:.2}\n",
                i, ws.0.0, ws.0.1, ws.0.2, ws.1.0, ws.1.1, ws.1.2, ws.2, ws.3
            ));
        }
        for (i, vpos) in diag_vein_positions.iter().enumerate() {
            let (ck, lx, ly, lz) = world_to_chunk_local(vpos.0, vpos.1, vpos.2, chunk_size);
            dbg.push_str(&format!(
                "  vein[{}]: world=({},{},{}) -> density_chunk=({},{},{}) local=({},{},{})\n",
                i, vpos.0, vpos.1, vpos.2, ck.0, ck.1, ck.2, lx, ly, lz
            ));
        }
        // Density field keys present
        let mut dk: Vec<(i32,i32,i32)> = density_fields.keys().copied().collect();
        dk.sort();
        dbg.push_str(&format!("  density_keys: {:?}\n", &dk[..dk.len().min(20)]));
        // Heat map summary
        dbg.push_str(&format!("  heat_map: {} sources", heat_map.len()));
        if let Some(first) = heat_map.first() {
            dbg.push_str(&format!(", first=({},{},{})", first.pos.0, first.pos.1, first.pos.2));
        }
        dbg.push_str("\n");
        result.transform_log.push(TransformEntry {
            description: dbg,
            count: 1,
        });
    }

    // --- Diagnostics ---
    let actual_output = result.veins_deposited + formation_total;
    result.diagnostics = PhaseDiagnostics {
        conversions,
        theoretical_max,
        actual_output,
        bottlenecks: compute_veins_bottlenecks(census, heat_map),
    };

    result
}

fn compute_veins_bottlenecks(census: &ResourceCensus, _heat_map: &HeatMap) -> Vec<Bottleneck> {
    let mut bottlenecks = Vec::new();

    let total_heat = census.heat_source_lava + census.heat_source_kimberlite;
    if total_heat == 0 {
        bottlenecks.push(Bottleneck {
            severity: 1.0,
            description: "No heat sources \u{2014} hydrothermal veins require lava or kimberlite".into(),
        });
    }

    if census.water.cell_count == 0 {
        bottlenecks.push(Bottleneck {
            severity: 1.0,
            description: "No water \u{2014} hydrothermal veins require water near heat sources. Direct water toward lava/kimberlite.".into(),
        });
    }

    if census.fissure_count > census.open_wall_count * 3 && census.fissure_count > 20 {
        bottlenecks.push(Bottleneck {
            severity: 0.4,
            description: format!(
                "{} fissures vs {} open walls \u{2014} mostly tight cracks, wider tunnels deposit more ore",
                census.fissure_count, census.open_wall_count
            ),
        });
    }

    // Check host rock availability near air
    let host_exposed: u32 = [
        Material::Sandstone as u8, Material::Limestone as u8,
        Material::Granite as u8, Material::Basalt as u8,
        Material::Slate as u8, Material::Marble as u8,
    ].iter().map(|m| census.exposed_surfaces_by_material.get(m).copied().unwrap_or(0)).sum();

    if host_exposed == 0 {
        bottlenecks.push(Bottleneck {
            severity: 0.8,
            description: "No exposed host rock adjacent to air \u{2014} mine into rock to create deposition surfaces".into(),
        });
    }

    bottlenecks.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap_or(std::cmp::Ordering::Equal));
    bottlenecks.truncate(3);
    bottlenecks
}
