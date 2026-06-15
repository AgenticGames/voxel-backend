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
use crate::util::{FACE_OFFSETS, sample_material, set_voxel_synced, grow_vein, VeinGrowthParams, VeinBias, default_vein_bias, ChunkSampleCache};
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
    pub amphibolite_placed: u32,
    pub veins_placed: u32,
    pub manifest: ChangeManifest,
    pub glimpse_chunk: Option<(i32, i32, i32)>,
    /// Exact world voxel position of the most intense aureole zone centroid
    pub glimpse_pos: Option<(i32, i32, i32)>,
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
    Basalt,
}

/// Determine the dominant host rock around the zone by sampling face-neighbors
/// of lava cells directly (avoids centroid-in-air problem). Picks Limestone vs.
/// Basalt vs. Slate (catch-all) based on which material's voxels touch lava most.
fn determine_aureole_type(
    zone: &LavaZone,
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
) -> AureoleType {
    let mut limestone_count = 0u32;
    let mut basalt_count = 0u32;
    let mut other_count = 0u32;

    // Sample up to 200 cells for perf
    let limit = zone.cells.len().min(200);
    for &(cx, cy, cz) in &zone.cells[..limit] {
        for &(dx, dy, dz) in &FACE_OFFSETS {
            if let Some(mat) = sample_material(density_fields, cx + dx, cy + dy, cz + dz, chunk_size) {
                match mat {
                    Material::Limestone => limestone_count += 1,
                    Material::Basalt => basalt_count += 1,
                    m if m.is_host_rock() => other_count += 1,
                    _ => {}
                }
            }
        }
    }

    // Limestone wins if it dominates, else basalt if it dominates, else slate (catch-all).
    if limestone_count > basalt_count && limestone_count > other_count {
        AureoleType::Limestone
    } else if basalt_count > limestone_count && basalt_count > other_count {
        AureoleType::Basalt
    } else {
        AureoleType::Slate
    }
}

// ──────────────────────────────────────────────────────────────
// Water Boost
// ──────────────────────────────────────────────────────────────

/// Single-slot chunk-pointer cache for `FluidSnapshot` cell lookups,
/// mirroring `ChunkSampleCache` for density fields. Inside the Phase-1 /
/// Phase-2 BFS in `compute_water_boost` the same FluidSnapshot chunk is
/// probed dozens of times in a row — 6 face neighbors per frontier cell,
/// plus consecutive frontier cells that share a chunk. This collapses every
/// same-chunk probe after the first into a `(i32,i32,i32)` equality +
/// pointer deref instead of a SipHash + HashMap probe.
///
/// Negative-result caching (None) is intentional: an unloaded chunk probed
/// repeatedly (BFS hitting the world edge) should also short-circuit.
struct FluidChunkCache<'a> {
    last_key: Option<(i32, i32, i32)>,
    last_cells: Option<&'a Vec<voxel_fluid::cell::FluidCell>>,
}

impl<'a> FluidChunkCache<'a> {
    #[inline]
    fn new() -> Self { Self { last_key: None, last_cells: None } }

    #[inline]
    fn cell(
        &mut self,
        snapshot: &'a FluidSnapshot,
        wx: i32,
        wy: i32,
        wz: i32,
    ) -> Option<&'a voxel_fluid::cell::FluidCell> {
        let cs_i = snapshot.chunk_size as i32;
        let key = (wx.div_euclid(cs_i), wy.div_euclid(cs_i), wz.div_euclid(cs_i));
        let cells = if self.last_key == Some(key) {
            self.last_cells
        } else {
            let r = snapshot.chunks.get(&key);
            self.last_key = Some(key);
            self.last_cells = r;
            r
        };
        let cells = cells?;
        let cs = snapshot.chunk_size;
        let lx = wx.rem_euclid(cs_i) as usize;
        let ly = wy.rem_euclid(cs_i) as usize;
        let lz = wz.rem_euclid(cs_i) as usize;
        let idx = lz * cs * cs + ly * cs + lx;
        cells.get(idx)
    }
}

/// Detailed water-boost computation for a single lava zone using the
/// **hydrothermal v2** model:
///
/// - **Phase 1** (thermally-active zone): BFS outward from every lava cell
///   up to `aureole_water_search_radius` hops. Water cells encountered are
///   weighted by `aureole_water_phase1_weight` (1.0 default).
/// - **Phase 2** (supply network): from each Phase-1 water cell, BFS
///   through face-adjacent water cells up to `aureole_water_network_max_hops`
///   away. These are "potential supply" cells (a river / aquifer behind
///   the direct heat zone) weighted at `aureole_water_phase2_weight`
///   (0.25 default).
/// - **Cap**: scales with lava zone size — `lava_cells * water_to_lava_ratio`,
///   floored at `water_phase1_max_floor`. So a tiny lava puddle saturates
///   at puddle-scale water, a giant chamber at ocean-scale.
///
/// Returns `(legacy_shell_boost, deposit_mult, count_mult, phase1_cells,
/// phase2_cells, weighted_total, water_cap, water_frac)`.
fn compute_water_boost(
    zone: &LavaZone,
    fluid_snapshot: &FluidSnapshot,
    config: &AureoleConfig,
) -> (f32, f32, f32, u32, u32, f32, f32, f32) {
    let search_r = config.aureole_water_search_radius.max(1) as i32;
    let net_max_hops = config.aureole_water_network_max_hops.max(0) as i32;
    let p1_weight = config.aureole_water_phase1_weight.max(0.0);
    let p2_weight = config.aureole_water_phase2_weight.max(0.0);

    let lava_set: HashSet<(i32, i32, i32)> = zone.cells.iter().copied().collect();

    // ── Phase 1: BFS-outward-from-lava ─────────────────────────────────
    // Visits each cell once via a face-neighbor BFS up to `search_r` deep.
    // Water cells found enter `phase1_water` (which seeds Phase 2).
    // Cost: O(|union of r-shell|) instead of the legacy O(N_lava * (2r+1)^3),
    // so r=15 is ~15ms instead of multiple seconds.
    let mut visited: HashSet<(i32, i32, i32)> = lava_set.clone();
    let mut frontier: Vec<(i32, i32, i32)> = zone.cells.clone();
    let mut phase1_water: Vec<(i32, i32, i32)> = Vec::new();
    // Single cache hoisted across every BFS shell — face neighbors of a
    // frontier cell, and consecutive frontier cells inside one shell, mostly
    // share a chunk. Collapses ~6N HashMap probes into ~N + chunk-transitions.
    let mut fluid_cache = FluidChunkCache::new();
    for _depth in 0..search_r {
        let mut next: Vec<(i32, i32, i32)> = Vec::new();
        for pos in frontier.drain(..) {
            for &(dx, dy, dz) in &FACE_OFFSETS {
                let n = (pos.0 + dx, pos.1 + dy, pos.2 + dz);
                if !visited.insert(n) { continue; }
                if lava_set.contains(&n) { continue; }
                if let Some(cell) = fluid_cache.cell(fluid_snapshot, n.0, n.1, n.2) {
                    if cell.level > 0.001 && cell.fluid_type.is_water() {
                        phase1_water.push(n);
                    }
                }
                next.push(n);
            }
        }
        frontier = next;
        if frontier.is_empty() { break; }
    }

    // ── Phase 2: BFS through connected water network ───────────────────
    // From each Phase-1 water cell, walk face-adjacent water cells up to
    // `net_max_hops` away. Each step extends the reachable supply network.
    // Cells are deduped globally so a single river isn't double-counted
    // even if multiple Phase-1 cells touch it.
    let mut phase2_visited: HashSet<(i32, i32, i32)> = phase1_water.iter().copied().collect();
    let mut phase2_count: u32 = 0;
    let mut p2_frontier: Vec<(i32, i32, i32)> = phase1_water.clone();
    // Reuse the cache across Phase 2 BFS — water network cells stay clustered
    // along rivers/aquifers so same-chunk runs are even longer here.
    let mut fluid_cache_p2 = FluidChunkCache::new();
    for _hops in 0..net_max_hops {
        let mut next: Vec<(i32, i32, i32)> = Vec::new();
        for pos in p2_frontier.drain(..) {
            for &(dx, dy, dz) in &FACE_OFFSETS {
                let n = (pos.0 + dx, pos.1 + dy, pos.2 + dz);
                if !phase2_visited.insert(n) { continue; }
                if let Some(cell) = fluid_cache_p2.cell(fluid_snapshot, n.0, n.1, n.2) {
                    if cell.level > 0.001 && cell.fluid_type.is_water() {
                        phase2_count += 1;
                        next.push(n);
                    }
                }
            }
        }
        p2_frontier = next;
        if p2_frontier.is_empty() { break; }
    }

    // ── Weighted total + ratio-scaled cap ──────────────────────────────
    let phase1_count = phase1_water.len() as u32;
    let weighted = (phase1_count as f32) * p1_weight + (phase2_count as f32) * p2_weight;
    let ratio = config.aureole_water_to_lava_ratio.max(0.0);
    let floor = config.aureole_water_phase1_max_floor.max(1) as f32;
    let cap = ((zone.cells.len() as f32) * ratio).max(floor);
    let weighted_clamped = weighted.min(cap);
    let water_frac = if cap > 0.0 { weighted_clamped / cap } else { 0.0 };

    let legacy_boost = 1.0 + water_frac * config.water_boost_max;
    let deposit_mult = 1.0 + water_frac * config.aureole_water_deposit_mult;
    let count_mult = 1.0 + water_frac * config.aureole_water_count_mult;

    (legacy_boost, deposit_mult, count_mult, phase1_count, phase2_count, weighted, cap, water_frac)
}

// ──────────────────────────────────────────────────────────────
// Per-zone experiment instrumentation
// ──────────────────────────────────────────────────────────────

/// Captures every input + output number for ONE lava zone so the user can
/// diff runs (no water / small water / more water / etc.) and see exactly
/// what changed. Written to `aureole_experiment.csv` after each zone.
#[derive(Debug, Default)]
pub(super) struct ZoneStats {
    pub aureole_type: &'static str, // "Slate" | "Limestone" | "Basalt"
    pub lava_cells: u32,
    /// Water cells in Phase 1 (within search radius of any lava cell).
    pub water_cells: u32,
    /// Water cells in Phase 2 (connected supply network behind Phase 1).
    pub water_phase2_cells: u32,
    /// `phase1*p1_weight + phase2*p2_weight` before cap.
    pub weighted_water: f32,
    /// Saturation cap for THIS zone: `lava_cells * water_to_lava_ratio`,
    /// floored at `water_phase1_max_floor`.
    pub water_cap_for_zone: f32,
    pub water_max_cells_cap: u32, // legacy flat cap (kept for backward CSV-column compat)
    pub water_frac: f32,
    pub water_search_radius: u32,
    pub legacy_boost: f32,
    pub water_deposit_mult: f32,
    /// NEW: water → vein/pocket COUNT multiplier at this saturation.
    pub water_count_mult: f32,
    pub lava_deposit_mult: f32,
    pub lava_count_mult: f32,
    pub combined_deposit_mult: f32,
    pub combined_count_mult: f32,
    pub final_depth: i32,
    pub hornfels: u32,
    pub skarn: u32,
    pub amphibolite: u32,
    pub converted: u32,
    pub total_vein_voxels: u32,
    // Per-material vein voxel counts (any placement — outer veins + pockets)
    pub veins_copper: u32,
    pub veins_iron: u32,
    pub veins_tin: u32,
    pub veins_gold: u32,
    pub veins_sulfide: u32,
    pub veins_pyrite: u32,
    pub veins_garnet: u32,
    pub veins_diopside: u32,
    // Per-material POCKET count (how many compact-pocket seeds were placed)
    pub pockets_pyrite: u32,
    pub pockets_garnet: u32,
    pub pockets_diopside: u32,
    // Outer-vein seed counts (how many wall-climbing seeds were placed)
    pub outer_seeds: u32,
}

pub(super) fn add_vein_count(stats: &mut ZoneStats, ore: Material, count: u32) {
    match ore {
        Material::Copper => stats.veins_copper += count,
        Material::Iron => stats.veins_iron += count,
        Material::Tin => stats.veins_tin += count,
        Material::Gold => stats.veins_gold += count,
        Material::Sulfide => stats.veins_sulfide += count,
        Material::Pyrite => stats.veins_pyrite += count,
        Material::Garnet => stats.veins_garnet += count,
        Material::Diopside => stats.veins_diopside += count,
        _ => {}
    }
}

fn write_experiment_row(stats: &ZoneStats, run_ts: f64, zone_idx: usize) {
    use std::io::Write;
    const PATH: &str = "D:/Unreal Projects/Mithril2026/Saved/aureole_experiment.csv";
    let exists = std::path::Path::new(PATH).exists();
    let needs_header = !exists || std::fs::metadata(PATH).map(|m| m.len() == 0).unwrap_or(true);
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(PATH) {
        if needs_header {
            let _ = writeln!(f, "run_ts,zone_idx,aureole_type,lava_cells,water_phase1,water_phase2,weighted_water,water_cap_zone,water_frac,water_search_r,legacy_boost,water_dep_mult,water_count_mult,lava_dep_mult,lava_count_mult,combined_dep_mult,combined_count_mult,final_depth,hornfels,skarn,amphibolite,converted,total_vein_voxels,outer_seeds,veins_copper,veins_iron,veins_tin,veins_gold,veins_sulfide,veins_pyrite,veins_garnet,veins_diopside,pockets_pyrite,pockets_garnet,pockets_diopside");
        }
        let _ = writeln!(f,
            "{:.3},{},{},{},{},{},{:.2},{:.2},{:.4},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            run_ts, zone_idx, stats.aureole_type,
            stats.lava_cells, stats.water_cells, stats.water_phase2_cells,
            stats.weighted_water, stats.water_cap_for_zone,
            stats.water_frac, stats.water_search_radius,
            stats.legacy_boost, stats.water_deposit_mult, stats.water_count_mult,
            stats.lava_deposit_mult, stats.lava_count_mult,
            stats.combined_deposit_mult, stats.combined_count_mult,
            stats.final_depth,
            stats.hornfels, stats.skarn, stats.amphibolite, stats.converted,
            stats.total_vein_voxels, stats.outer_seeds,
            stats.veins_copper, stats.veins_iron, stats.veins_tin, stats.veins_gold,
            stats.veins_sulfide, stats.veins_pyrite, stats.veins_garnet, stats.veins_diopside,
            stats.pockets_pyrite, stats.pockets_garnet, stats.pockets_diopside,
        );
        let _ = f.flush();
    }
}

// ──────────────────────────────────────────────────────────────
// Metamorphic Shell Placement (multi-source BFS from lava cells)
// ──────────────────────────────────────────────────────────────

/// Place metamorphic shell via BFS from every lava cell outward into solid rock.
/// Limestone → Skarn, Basalt → Amphibolite, other host rock → Hornfels.
/// Air gaps block propagation.
/// Returns (hornfels_count, skarn_count, amphibolite_count, set of converted world positions).
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
) -> (u32, u32, u32, HashSet<(i32, i32, i32)>) {
    let mut hornfels_count = 0u32;
    let mut skarn_count = 0u32;
    let mut amphibolite_count = 0u32;
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

            if mat == Material::Hornfels || mat == Material::Skarn || mat == Material::Amphibolite {
                // Already metamorphosed — continue BFS through but don't re-convert
                queue.push_back((n, next_dist));
                continue;
            }

            let new_mat = if mat == Material::Limestone {
                Material::Skarn
            } else if mat == Material::Basalt {
                Material::Amphibolite
            } else if mat.is_host_rock() {
                Material::Hornfels
            } else {
                // Non-host-rock solid (ore, etc.) — block BFS
                continue;
            };

            // Convert with boundary sync
            let spread = if max_depth > 0 { next_dist as f32 / max_depth as f32 } else { 0.0 };
            set_voxel_synced(density_fields, key, lx, ly, lz, new_mat, None, chunk_size);
            manifest.record_voxel_change_with_spread(key, lx, ly, lz, mat, density, new_mat, density, spread);
            converted.insert(n);

            match new_mat {
                Material::Skarn => skarn_count += 1,
                Material::Amphibolite => amphibolite_count += 1,
                _ => hornfels_count += 1,
            }

            queue.push_back((n, next_dist));
        }
    }

    (hornfels_count, skarn_count, amphibolite_count, converted)
}

// ──────────────────────────────────────────────────────────────
// Aureole Boundary Seed Finding
// ──────────────────────────────────────────────────────────────

/// Find vein seed positions at the aureole boundary: converted voxels that have
/// at least one air face-neighbor (visible) AND at least one unconverted host-rock
/// face-neighbor (vein can grow into). Seeds are placed where players will see them.
fn find_aureole_boundary_seeds(
    converted: &HashSet<(i32, i32, i32)>,
    lava_set: &HashSet<(i32, i32, i32)>,
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
    count: usize,
    rng: &mut ChaCha8Rng,
    spread: f32,
    inner_boost: f32,
) -> Vec<(i32, i32, i32)> {
    // Collect boundary candidates with air-neighbor count for weighting.
    // Inner-facing seeds (where a face-neighbor IS a position that was lava
    // when sleep began) get a heavy weight bonus — these are the surfaces
    // visible from inside the original lava chamber. Without this bias, as
    // the aureole shell grows with water saturation, ore would preferentially
    // spawn at the OUTER perimeter (deep in surrounding rock, hidden from
    // the player's view inside the pit). The lava-facing boost keeps the
    // visible reward proportional to the boost.
    let mut candidates: Vec<((i32, i32, i32), f32)> = Vec::new();

    for &pos in converted {
        let mut air_count = 0u32;
        let mut lava_facing_count = 0u32;
        let mut has_host = false;
        for &(dx, dy, dz) in &FACE_OFFSETS {
            let n = (pos.0 + dx, pos.1 + dy, pos.2 + dz);
            // If this face touches a position that was lava at sleep start,
            // it's "inner-facing" — the cavity surface the player sees from
            // inside the original chamber.
            if lava_set.contains(&n) {
                lava_facing_count += 1;
                air_count += 1;
                continue;
            }
            if let Some(mat) = sample_material(density_fields, n.0, n.1, n.2, chunk_size) {
                if !mat.is_solid() {
                    air_count += 1;
                } else if mat.is_host_rock()
                    && mat != Material::Hornfels
                    && mat != Material::Skarn
                    && mat != Material::Amphibolite
                    && !converted.contains(&n)
                {
                    has_host = true;
                }
            }
        }
        // Two ways to qualify as a seed:
        //   * Inner-face seed: any face touches the original lava cavity.
        //     The vein will grow outward through the metamorphic shell, so
        //     it outcrops on the chamber wall the player is looking at.
        //   * Outer-boundary seed: legacy — face into air AND face into
        //     unaltered host rock. Vein grows into surrounding rock.
        // Inner-face seeds get a heavy weight bonus so they dominate when
        // the shell expands with water saturation. Outer seeds still appear
        // for variety / "mine deeper to find more ore" depth.
        let is_inner = lava_facing_count > 0;
        let is_outer = air_count >= 1 && has_host;
        if is_inner || is_outer {
            let w = air_count as f32 + (lava_facing_count as f32) * inner_boost;
            candidates.push((pos, w));
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
        // Compute weights: base (air-count + lava-facing bonus) × spread repulsion
        let weights: Vec<f32> = remaining.iter().map(|&(pos, base_w)| {
            let mut w = base_w;
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
        // Veins appear mid-sequence (spread 0.5) — after aureole shell, before outermost
        manifest.record_voxel_change_with_spread(key, lx, ly, lz, old_mat, old_density, material, old_density, 0.5);
        count += 1;
    }
    count
}

/// Place ore veins for a Slate-hosted aureole zone.
fn place_slate_veins(
    converted: &HashSet<(i32, i32, i32)>,
    lava_set: &HashSet<(i32, i32, i32)>,
    config: &AureoleConfig,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
    manifest: &mut ChangeManifest,
    rng: &mut ChaCha8Rng,
    skarn_count: u32,
    deposit_mult: f32,
    count_mult: f32,
    zone_cells: u32,
    stats: &mut ZoneStats,
) -> u32 {
    // Base count + linear scaling from zone size
    let n = config.aureole_cells_per_extra.max(1) as f32;
    // Cap lava cells that contribute to extras at the same threshold as the
    // lava_count_mult / lava_deposit_mult saturation. Without this, a giant
    // magma chamber would spawn thousands of extra seeds/pockets linearly
    // with size; capped, "more lava beyond N cells stops mattering" — same
    // as the multipliers. Single knob (aureole_lava_volume_max_cells)
    // controls both saturation behaviours.
    let extras_cap = config.aureole_lava_volume_max_cells.max(1);
    let effective_zone_cells = zone_cells.min(extras_cap) as f32;
    let extra_veins = if config.aureole_veins_per_n_cells > 0.0 {
        (effective_zone_cells / n * config.aureole_veins_per_n_cells).floor() as usize
    } else { 0 };
    let vein_count = (((config.aureole_vein_count as f32 * count_mult).round() as usize) + extra_veins).max(1);
    let seeds = find_aureole_boundary_seeds(converted, lava_set, density_fields, chunk_size, vein_count, rng, config.aureole_vein_spread, 5.0);
    if seeds.is_empty() {
        return 0;
    }

    // Slate-hosted pluton-adjacent: 50% Copper / 30% Iron / 20% Tin (signature).
    // Plus a separate sprinkling of small Pyrite veins (the marker accessory
    // for slate aureoles in real geology — added as additional seeds, NOT
    // pulled from the main allotment, so all 4 ores are always visible.
    let vein_min = ((config.aureole_vein_min as f32 * deposit_mult).round() as u32).max(2);
    let vein_max = ((config.aureole_vein_max as f32 * deposit_mult).round() as u32).max(vein_min + 1);
    let small_min = ((config.small_vein_base_size as f32 * deposit_mult).round() as u32).max(2);
    let small_max = ((small_min as f32 * 1.5).round() as u32).max(small_min + 1);

    let mut total_placed = 0u32;
    stats.outer_seeds = seeds.len() as u32;
    let n_seeds = seeds.len() as f32;

    // Main veins: Cu 50% / Fe 30% / Tin 20%
    for (i, &seed) in seeds.iter().enumerate() {
        let frac = (i as f32 + 0.5) / n_seeds.max(1.0);
        let ore = if frac < 0.5 { Material::Copper }
                  else if frac < 0.8 { Material::Iron }
                  else { Material::Tin };
        let s_min = ((vein_min as f32 * deposit_mult).round() as u32).max(2);
        let s_max = ((vein_max as f32 * deposit_mult).round() as u32).max(s_min + 1);
        let (actual_min, actual_max, bias) = if config.aureole_wall_climbing {
            let wall_normal = FACE_OFFSETS.iter()
                .find(|&&(dx, dy, dz)| {
                    sample_material(density_fields, seed.0 + dx, seed.1 + dy, seed.2 + dz, chunk_size)
                        .map_or(false, |m| !m.is_solid())
                })
                .copied()
                .unwrap_or((0, 1, 0));
            (s_min, s_max, VeinBias::WallClimbing {
                wall_normal,
                weight_up: config.aureole_weight_up,
                weight_depth: config.aureole_weight_depth,
                weight_lateral: config.aureole_weight_lateral,
                weight_down: config.aureole_weight_down,
                surface_ratio: config.aureole_surface_ratio,
            })
        } else {
            (s_min, s_max, default_vein_bias(ore, rng))
        };
        let params = VeinGrowthParams { ore, min_size: actual_min, max_size: actual_max, bias, exclude_aureole: false, min_connectivity: config.aureole_min_connectivity };
        let positions = grow_vein(density_fields, seed, &params, chunk_size, rng);
        let placed = apply_vein_to_world(&positions, ore, density_fields, chunk_size, manifest);
        add_vein_count(stats, ore, placed);
        total_placed += placed;
    }

    // Extra small Pyrite veins (~25% of main count) — separate seed pass.
    // Always present so slate zones reliably show the FeS2 accessory.
    let pyrite_extra = (seeds.len() / 4).max(1);
    let pyrite_seeds = find_aureole_boundary_seeds(
        converted, lava_set, density_fields, chunk_size, pyrite_extra, rng, config.aureole_vein_spread, 5.0,
    );
    for &seed in &pyrite_seeds {
        let (actual_min, actual_max, bias) = if config.aureole_wall_climbing {
            let wall_normal = FACE_OFFSETS.iter()
                .find(|&&(dx, dy, dz)| {
                    sample_material(density_fields, seed.0 + dx, seed.1 + dy, seed.2 + dz, chunk_size)
                        .map_or(false, |m| !m.is_solid())
                })
                .copied()
                .unwrap_or((0, 1, 0));
            (small_min, small_max, VeinBias::WallClimbing {
                wall_normal,
                weight_up: config.aureole_weight_up,
                weight_depth: config.aureole_weight_depth,
                weight_lateral: config.aureole_weight_lateral,
                weight_down: config.aureole_weight_down,
                surface_ratio: config.aureole_surface_ratio,
            })
        } else {
            (small_min, small_max, default_vein_bias(Material::Pyrite, rng))
        };
        let params = VeinGrowthParams { ore: Material::Pyrite, min_size: actual_min, max_size: actual_max, bias, exclude_aureole: false, min_connectivity: config.aureole_min_connectivity };
        let positions = grow_vein(density_fields, seed, &params, chunk_size, rng);
        let placed = apply_vein_to_world(&positions, Material::Pyrite, density_fields, chunk_size, manifest);
        add_vein_count(stats, Material::Pyrite, placed);
        total_placed += placed;
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
            let extra_garnet = if config.aureole_garnet_per_n_cells > 0.0 {
                (effective_zone_cells / n * config.aureole_garnet_per_n_cells).floor() as u32
            } else { 0 };
            // Pocket COUNT now scales with count_mult so water gives more
            // pockets (not just bigger pockets). If the config explicitly
            // disables pockets (base=0), stay at 0 — only nonzero base scales.
            let garnet_base = config.garnet_pocket_count + extra_garnet;
            let garnet_total = if garnet_base == 0 { 0 } else {
                ((garnet_base as f32 * count_mult).round() as u32).max(1)
            };
            for _ in 0..garnet_total {
                let garnet_seed = skarn_seeds[rng.gen_range(0..skarn_seeds.len())];
                let params = VeinGrowthParams {
                    ore: Material::Garnet,
                    min_size: (g_size * 8) / 10,
                    max_size: g_size,
                    bias: VeinBias::Compact,
                    exclude_aureole: false,
                    min_connectivity: 1,
                };
                let positions = grow_vein(density_fields, garnet_seed, &params, chunk_size, rng);
                let placed = apply_vein_to_world(&positions, Material::Garnet, density_fields, chunk_size, manifest);
                add_vein_count(stats, Material::Garnet, placed);
                stats.pockets_garnet += 1;
                total_placed += placed;
            }

            let d_size = ((config.diopside_compact_size as f32 * deposit_mult).round() as u32).max(3);
            let extra_diopside = if config.aureole_diopside_per_n_cells > 0.0 {
                (effective_zone_cells / n * config.aureole_diopside_per_n_cells).floor() as u32
            } else { 0 };
            let diopside_base = config.diopside_pocket_count + extra_diopside;
            let diopside_total = if diopside_base == 0 { 0 } else {
                ((diopside_base as f32 * count_mult).round() as u32).max(1)
            };
            for _ in 0..diopside_total {
                let diopside_seed = skarn_seeds[rng.gen_range(0..skarn_seeds.len())];
                let params = VeinGrowthParams {
                    ore: Material::Diopside,
                    min_size: (d_size * 8) / 10,
                    max_size: d_size,
                    bias: VeinBias::Compact,
                    exclude_aureole: false,
                    min_connectivity: 1,
                };
                let positions = grow_vein(density_fields, diopside_seed, &params, chunk_size, rng);
                let placed = apply_vein_to_world(&positions, Material::Diopside, density_fields, chunk_size, manifest);
                add_vein_count(stats, Material::Diopside, placed);
                stats.pockets_diopside += 1;
                total_placed += placed;
            }
        }
    }

    total_placed
}

/// Place ore veins for a Limestone-hosted (Skarn) aureole zone.
fn place_limestone_veins(
    converted: &HashSet<(i32, i32, i32)>,
    lava_set: &HashSet<(i32, i32, i32)>,
    config: &AureoleConfig,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
    manifest: &mut ChangeManifest,
    rng: &mut ChaCha8Rng,
    deposit_mult: f32,
    count_mult: f32,
    zone_cells: u32,
    stats: &mut ZoneStats,
) -> u32 {
    // Ore veins at boundary seeds — base + linear scaling from zone size
    let n = config.aureole_cells_per_extra.max(1) as f32;
    // Cap lava cells that contribute to extras at the same threshold as the
    // lava_count_mult / lava_deposit_mult saturation. Without this, a giant
    // magma chamber would spawn thousands of extra seeds/pockets linearly
    // with size; capped, "more lava beyond N cells stops mattering" — same
    // as the multipliers. Single knob (aureole_lava_volume_max_cells)
    // controls both saturation behaviours.
    let extras_cap = config.aureole_lava_volume_max_cells.max(1);
    let effective_zone_cells = zone_cells.min(extras_cap) as f32;
    let extra_veins = if config.aureole_veins_per_n_cells > 0.0 {
        (effective_zone_cells / n * config.aureole_veins_per_n_cells).floor() as usize
    } else { 0 };
    let ore_count = (((config.aureole_vein_count as f32 * count_mult * 0.6).round() as usize) + extra_veins).max(1);
    let seeds = find_aureole_boundary_seeds(converted, lava_set, density_fields, chunk_size, ore_count, rng, config.aureole_vein_spread, 5.0);
    if seeds.is_empty() {
        return 0;
    }

    // Limestone-hosted skarn: 50% Copper / 30% Iron / 20% Gold (signature).
    // Gold is the rare accent that screams "this is a skarn aureole."
    let vein_min = ((config.aureole_vein_min as f32 * deposit_mult).round() as u32).max(2);
    let vein_max = ((config.aureole_vein_max as f32 * deposit_mult).round() as u32).max(vein_min + 1);

    let mut total_placed = 0u32;
    stats.outer_seeds = seeds.len() as u32;
    let n_seeds = seeds.len() as f32;

    for (i, &seed) in seeds.iter().enumerate() {
        let frac = (i as f32 + 0.5) / n_seeds.max(1.0);
        let ore = if frac < 0.5 { Material::Copper }
                  else if frac < 0.8 { Material::Iron }
                  else { Material::Gold };
        let (actual_min, actual_max, bias) = if config.aureole_wall_climbing {
            let wall_normal = FACE_OFFSETS.iter()
                .find(|&&(dx, dy, dz)| {
                    sample_material(density_fields, seed.0 + dx, seed.1 + dy, seed.2 + dz, chunk_size)
                        .map_or(false, |m| !m.is_solid())
                })
                .copied()
                .unwrap_or((0, 1, 0));
            (vein_min, vein_max, VeinBias::WallClimbing {
                wall_normal,
                weight_up: config.aureole_weight_up,
                weight_depth: config.aureole_weight_depth,
                weight_lateral: config.aureole_weight_lateral,
                weight_down: config.aureole_weight_down,
                surface_ratio: config.aureole_surface_ratio,
            })
        } else {
            (vein_min, vein_max, default_vein_bias(ore, rng))
        };
        let params = VeinGrowthParams { ore, min_size: actual_min, max_size: actual_max, bias, exclude_aureole: false, min_connectivity: config.aureole_min_connectivity };
        let positions = grow_vein(density_fields, seed, &params, chunk_size, rng);
        let placed = apply_vein_to_world(&positions, ore, density_fields, chunk_size, manifest);
        add_vein_count(stats, ore, placed);
        total_placed += placed;
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
        let extra_garnet = if config.aureole_garnet_per_n_cells > 0.0 {
            (effective_zone_cells / n * config.aureole_garnet_per_n_cells).floor() as u32
        } else { 0 };
        let garnet_base = config.garnet_pocket_count + extra_garnet;
        let garnet_total = if garnet_base == 0 { 0 } else {
            ((garnet_base as f32 * count_mult).round() as u32).max(1)
        };
        for _ in 0..garnet_total {
            let seed = skarn_seeds[rng.gen_range(0..skarn_seeds.len())];
            let params = VeinGrowthParams {
                ore: Material::Garnet,
                min_size: (g_size * 8) / 10,
                max_size: g_size,
                bias: VeinBias::Compact,
                exclude_aureole: false,
                min_connectivity: 1,
            };
            let positions = grow_vein(density_fields, seed, &params, chunk_size, rng);
            let placed = apply_vein_to_world(&positions, Material::Garnet, density_fields, chunk_size, manifest);
            add_vein_count(stats, Material::Garnet, placed);
            stats.pockets_garnet += 1;
            total_placed += placed;
        }

        let d_size = ((config.diopside_compact_size as f32 * deposit_mult).round() as u32).max(3);
        let extra_diopside = if config.aureole_diopside_per_n_cells > 0.0 {
            (effective_zone_cells / n * config.aureole_diopside_per_n_cells).floor() as u32
        } else { 0 };
        let diopside_base = config.diopside_pocket_count + extra_diopside;
        let diopside_total = if diopside_base == 0 { 0 } else {
            ((diopside_base as f32 * count_mult).round() as u32).max(1)
        };
        for _ in 0..diopside_total {
            let seed = skarn_seeds[rng.gen_range(0..skarn_seeds.len())];
            let params = VeinGrowthParams {
                ore: Material::Diopside,
                min_size: (d_size * 8) / 10,
                max_size: d_size,
                bias: VeinBias::Compact,
                exclude_aureole: false,
                min_connectivity: 1,
            };
            let positions = grow_vein(density_fields, seed, &params, chunk_size, rng);
            let placed = apply_vein_to_world(&positions, Material::Diopside, density_fields, chunk_size, manifest);
            add_vein_count(stats, Material::Diopside, placed);
            stats.pockets_diopside += 1;
            total_placed += placed;
        }
    }

    total_placed
}

/// Place ore veins for a Basalt-hosted (Amphibolite) aureole zone.
///
/// Geological model: basalt-hosted contact metamorphism + hydrothermal alteration
/// produces volcanogenic massive sulfide style deposits. Outer veins favour
/// Cu/Fe/Sulfide; inside the amphibolite shell, compact pockets of Pyrite + Garnet
/// form (signature high-grade indicators of basalt-hosted hydrothermal systems).
fn place_basalt_veins(
    converted: &HashSet<(i32, i32, i32)>,
    lava_set: &HashSet<(i32, i32, i32)>,
    config: &AureoleConfig,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
    manifest: &mut ChangeManifest,
    rng: &mut ChaCha8Rng,
    amphibolite_count: u32,
    deposit_mult: f32,
    count_mult: f32,
    zone_cells: u32,
    stats: &mut ZoneStats,
) -> u32 {
    let n = config.aureole_cells_per_extra.max(1) as f32;
    // Cap lava cells that contribute to extras at the same threshold as the
    // lava_count_mult / lava_deposit_mult saturation. Without this, a giant
    // magma chamber would spawn thousands of extra seeds/pockets linearly
    // with size; capped, "more lava beyond N cells stops mattering" — same
    // as the multipliers. Single knob (aureole_lava_volume_max_cells)
    // controls both saturation behaviours.
    let extras_cap = config.aureole_lava_volume_max_cells.max(1);
    let effective_zone_cells = zone_cells.min(extras_cap) as f32;
    let extra_veins = if config.aureole_veins_per_n_cells > 0.0 {
        (effective_zone_cells / n * config.aureole_veins_per_n_cells).floor() as usize
    } else { 0 };
    let vein_count = (((config.aureole_vein_count as f32 * count_mult).round() as usize) + extra_veins).max(1);
    let seeds = find_aureole_boundary_seeds(converted, lava_set, density_fields, chunk_size, vein_count, rng, config.aureole_vein_spread, 5.0);
    if seeds.is_empty() {
        return 0;
    }

    // Basalt-hosted VMS: 50% Copper / 30% Iron / 20% Sulfide (signature).
    // Demo-friendly: every zone shows the full ore palette, with Sulfide
    // as the rare accent that identifies "this is a basalt aureole."
    let vein_min = ((config.aureole_vein_min as f32 * deposit_mult).round() as u32).max(2);
    let vein_max = ((config.aureole_vein_max as f32 * deposit_mult).round() as u32).max(vein_min + 1);

    let mut total_placed = 0u32;
    stats.outer_seeds = seeds.len() as u32;
    let n_seeds = seeds.len() as f32;

    for (i, &seed) in seeds.iter().enumerate() {
        let frac = (i as f32 + 0.5) / n_seeds.max(1.0);
        let ore = if frac < 0.5 { Material::Copper }
                  else if frac < 0.8 { Material::Iron }
                  else { Material::Sulfide };
        let (actual_min, actual_max, bias) = if config.aureole_wall_climbing {
            let wall_normal = FACE_OFFSETS.iter()
                .find(|&&(dx, dy, dz)| {
                    sample_material(density_fields, seed.0 + dx, seed.1 + dy, seed.2 + dz, chunk_size)
                        .map_or(false, |m| !m.is_solid())
                })
                .copied()
                .unwrap_or((0, 1, 0));
            (vein_min, vein_max, VeinBias::WallClimbing {
                wall_normal,
                weight_up: config.aureole_weight_up,
                weight_depth: config.aureole_weight_depth,
                weight_lateral: config.aureole_weight_lateral,
                weight_down: config.aureole_weight_down,
                surface_ratio: config.aureole_surface_ratio,
            })
        } else {
            (vein_min, vein_max, default_vein_bias(ore, rng))
        };
        let params = VeinGrowthParams { ore, min_size: actual_min, max_size: actual_max, bias, exclude_aureole: false, min_connectivity: config.aureole_min_connectivity };
        let positions = grow_vein(density_fields, seed, &params, chunk_size, rng);
        let placed = apply_vein_to_world(&positions, ore, density_fields, chunk_size, manifest);
        add_vein_count(stats, ore, placed);
        total_placed += placed;
    }

    // Compact Pyrite + Garnet pockets seeded inside amphibolite voxels
    if amphibolite_count > 0 {
        let mut amph_seeds: Vec<(i32, i32, i32)> = converted.iter()
            .filter(|&&pos| {
                sample_material(density_fields, pos.0, pos.1, pos.2, chunk_size)
                    .map_or(false, |m| m == Material::Amphibolite)
            })
            .copied()
            .collect();
        amph_seeds.sort();
        if !amph_seeds.is_empty() {
            // Pyrite pockets — primary signature mineral (basalt FeS2 hydrothermal)
            let p_size = ((config.amphibolite_pyrite_compact_size as f32 * deposit_mult).round() as u32).max(3);
            let extra_pyrite = if config.aureole_amphibolite_pyrite_per_n_cells > 0.0 {
                (effective_zone_cells / n * config.aureole_amphibolite_pyrite_per_n_cells).floor() as u32
            } else { 0 };
            let pyrite_base = config.amphibolite_pyrite_pocket_count + extra_pyrite;
            let pyrite_total = if pyrite_base == 0 { 0 } else {
                ((pyrite_base as f32 * count_mult).round() as u32).max(1)
            };
            for _ in 0..pyrite_total {
                let seed = amph_seeds[rng.gen_range(0..amph_seeds.len())];
                let params = VeinGrowthParams {
                    ore: Material::Pyrite,
                    min_size: (p_size * 8) / 10,
                    max_size: p_size,
                    bias: VeinBias::Compact,
                    exclude_aureole: false,
                    min_connectivity: 1,
                };
                let positions = grow_vein(density_fields, seed, &params, chunk_size, rng);
                let placed = apply_vein_to_world(&positions, Material::Pyrite, density_fields, chunk_size, manifest);
                add_vein_count(stats, Material::Pyrite, placed);
                stats.pockets_pyrite += 1;
                total_placed += placed;
            }

            // Garnet pockets — secondary high-grade indicator (garnet-amphibolite facies)
            let g_size = ((config.garnet_compact_size as f32 * deposit_mult).round() as u32).max(3);
            let extra_garnet = if config.aureole_amphibolite_garnet_per_n_cells > 0.0 {
                (effective_zone_cells / n * config.aureole_amphibolite_garnet_per_n_cells).floor() as u32
            } else { 0 };
            let garnet_base = config.amphibolite_garnet_pocket_count + extra_garnet;
            let garnet_total = if garnet_base == 0 { 0 } else {
                ((garnet_base as f32 * count_mult).round() as u32).max(1)
            };
            for _ in 0..garnet_total {
                let seed = amph_seeds[rng.gen_range(0..amph_seeds.len())];
                let params = VeinGrowthParams {
                    ore: Material::Garnet,
                    min_size: (g_size * 8) / 10,
                    max_size: g_size,
                    bias: VeinBias::Compact,
                    exclude_aureole: false,
                    min_connectivity: 1,
                };
                let positions = grow_vein(density_fields, seed, &params, chunk_size, rng);
                let placed = apply_vein_to_world(&positions, Material::Garnet, density_fields, chunk_size, manifest);
                add_vein_count(stats, Material::Garnet, placed);
                stats.pockets_garnet += 1;
                total_placed += placed;
            }
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
        crate::trace(&format!("aureole: clustering {} heat sources", heat_map.len()));
        let zones = cluster_lava_zones(heat_map, config.min_lava_zone_size);
        crate::trace(&format!("aureole: {} zones found", zones.len()));
        result.lava_zones_found = zones.len() as u32;

        let mut best_glimpse_score: u32 = 0;

        let run_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        for (zone_idx, zone) in zones.iter().enumerate() {
            crate::trace(&format!("aureole: zone {} starting ({} cells)", zone_idx, zone.cells.len()));
            let mut stats = ZoneStats::default();
            stats.lava_cells = zone.cells.len() as u32;
            // Compute BFS depth from zone size using ln() for sensible shell thicknesses
            // 5 cells→2, 50→4, 200→5, 958→7 voxels
            let cell_count = zone.cells.len() as f32;
            let base_depth = (cell_count.ln().max(1.0) * config.radius_scale * config.heat_multiplier)
                .min(config.max_radius)
                .max(2.0);
            let (
                water_boost,
                water_deposit_mult,
                water_count_mult,
                phase1_cells,
                phase2_cells,
                weighted_water,
                water_cap_for_zone,
                water_frac,
            ) = compute_water_boost(zone, fluid_snapshot, config);
            let final_depth = (base_depth * water_boost).ceil() as i32;

            // Lava volume scaling: fraction of zone cells vs max_cells cap
            let lava_max = config.aureole_lava_volume_max_cells.max(1);
            let lava_frac = (cell_count.min(lava_max as f32)) / (lava_max as f32);
            let lava_deposit_mult = 1.0 + lava_frac * config.aureole_lava_deposit_mult;
            let lava_count_mult = 1.0 + lava_frac * config.aureole_lava_count_mult;
            let combined_deposit_mult = lava_deposit_mult * water_deposit_mult;
            // Water boost v2: now also multiplies into vein/pocket COUNT
            // (used to be lava-only — water gave bigger but not more deposits).
            let combined_count_mult = lava_count_mult * water_count_mult;

            // Phase 1 + Phase 2 cell counts replace the old single water_cells field.
            stats.water_cells = phase1_cells;
            stats.water_phase2_cells = phase2_cells;
            stats.weighted_water = weighted_water;
            stats.water_cap_for_zone = water_cap_for_zone;
            stats.water_max_cells_cap = config.aureole_water_max_cells.max(1);
            stats.water_frac = water_frac;
            stats.water_search_radius = config.aureole_water_search_radius;
            stats.legacy_boost = water_boost;
            stats.water_deposit_mult = water_deposit_mult;
            stats.water_count_mult = water_count_mult;
            stats.lava_deposit_mult = lava_deposit_mult;
            stats.lava_count_mult = lava_count_mult;
            stats.combined_deposit_mult = combined_deposit_mult;
            stats.combined_count_mult = combined_count_mult;
            stats.final_depth = final_depth;

            if final_depth < 1 {
                // Still write a row so the user sees the zone was considered
                write_experiment_row(&stats, run_ts, zone_idx);
                continue;
            }

            let aureole_type = determine_aureole_type(zone, density_fields, chunk_size);
            stats.aureole_type = match aureole_type {
                AureoleType::Slate => "Slate",
                AureoleType::Limestone => "Limestone",
                AureoleType::Basalt => "Basalt",
            };
            crate::trace(&format!("aureole: zone {} type={:?} depth={}", zone_idx, aureole_type, final_depth));

            // Pass 1: metamorphic shell via BFS from lava cells
            let (hornfels_n, skarn_n, amphibolite_n, converted) = place_metamorphic_shell(
                zone,
                final_depth,
                density_fields,
                chunk_size,
                &mut result.manifest,
            );
            result.hornfels_placed += hornfels_n;
            result.skarn_placed += skarn_n;
            result.amphibolite_placed += amphibolite_n;
            result.voxels_metamorphosed += hornfels_n + skarn_n + amphibolite_n;
            stats.hornfels = hornfels_n;
            stats.skarn = skarn_n;
            stats.amphibolite = amphibolite_n;
            stats.converted = converted.len() as u32;

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
                    "[ZONE_DIAG] zone_idx={} cells={} depth={} centroid=({},{},{}) lava_min=({},{},{}) lava_max=({},{},{}) hornfels={} skarn={} amphibolite={} placed_min=({},{},{}) placed_max=({},{},{})",
                    result.debug_zones.len() - 1, zone.cells.len(), final_depth,
                    zone.centroid.0, zone.centroid.1, zone.centroid.2,
                    lmin.0, lmin.1, lmin.2, lmax.0, lmax.1, lmax.2,
                    hornfels_n, skarn_n, amphibolite_n,
                    hmin.0, hmin.1, hmin.2, hmax.0, hmax.1, hmax.2,
                ));
                // Parseable bounding boxes for UE debug visualization
                if hornfels_n + skarn_n + amphibolite_n > 0 {
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

            // Pass 2: ore veins + pockets (grow into just-placed metamorphic rock).
            // Build the lava_set once so vein placement can bias seeds toward
            // surfaces facing the original chamber (visible from the pit).
            let zone_cell_count = zone.cells.len() as u32;
            let lava_set: HashSet<(i32, i32, i32)> = zone.cells.iter().copied().collect();
            crate::trace(&format!("aureole: zone {} shell done (h={} s={} a={}, converted={}), placing veins ({:?})", zone_idx, hornfels_n, skarn_n, amphibolite_n, converted.len(), aureole_type));
            let veins_placed = match aureole_type {
                AureoleType::Slate => place_slate_veins(
                    &converted, &lava_set, config,
                    density_fields, chunk_size, &mut result.manifest, rng, skarn_n,
                    combined_deposit_mult, combined_count_mult, zone_cell_count,
                    &mut stats,
                ),
                AureoleType::Limestone => place_limestone_veins(
                    &converted, &lava_set, config,
                    density_fields, chunk_size, &mut result.manifest, rng,
                    combined_deposit_mult, combined_count_mult, zone_cell_count,
                    &mut stats,
                ),
                AureoleType::Basalt => place_basalt_veins(
                    &converted, &lava_set, config,
                    density_fields, chunk_size, &mut result.manifest, rng, amphibolite_n,
                    combined_deposit_mult, combined_count_mult, zone_cell_count,
                    &mut stats,
                ),
            };
            crate::trace(&format!("aureole: zone {} done veins={}", zone_idx, veins_placed));
            stats.total_vein_voxels = veins_placed;
            write_experiment_row(&stats, run_ts, zone_idx);
            result.veins_placed += veins_placed;

            // Glimpse selection: pick the zone with the most total transformation
            // (metamorphic shell + ore veins) for the montage showcase
            let zone_total = hornfels_n + skarn_n + amphibolite_n + veins_placed;
            if zone_total > best_glimpse_score {
                best_glimpse_score = zone_total;
                result.glimpse_pos = Some(zone.centroid);
                let (key, _, _, _) = world_to_chunk_local(
                    zone.centroid.0, zone.centroid.1, zone.centroid.2, chunk_size,
                );
                result.glimpse_chunk = Some(key);
            }
        }

        // Add transform log entry for zone metamorphism
        if result.lava_zones_found > 0 {
            let total_meta = result.hornfels_placed + result.skarn_placed
                + result.amphibolite_placed + result.veins_placed;
            result.transform_log.push(TransformEntry {
                description: format!(
                    "The Aureole \u{2014} 100,000 years: {} lava zones, {} hornfels, {} skarn, {} amphibolite, {} ore vein voxels",
                    result.lava_zones_found, result.hornfels_placed, result.skarn_placed,
                    result.amphibolite_placed, result.veins_placed
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

        // Per-scan chunk-pointer cache shared across every water cell's 6 face
        // probes. Adjacent water cells almost always share chunks, so the cache
        // hit rate is high (~80–95%). density_fields is read-only during scan
        // (candidates are applied later) so this is safe to hoist.
        let mut nbr_cache = ChunkSampleCache::new();
        for &((wx, wy, wz), _idx, level, _) in &water_cells {
            // Scale erosion probability by water cell level (more water = stronger erosion)
            let level_factor = level.min(1.0);
            for &(dx, dy, dz) in &FACE_OFFSETS {
                let nx = wx + dx;
                let ny = wy + dy;
                let nz = wz + dz;
                if let Some(mat) = nbr_cache.material(density_fields, nx, ny, nz, chunk_size) {
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

            // Per-chunk neighbor-probe cache. Interior voxels (the vast majority
            // of a 17³ field) have all 6 face neighbors land in this same
            // chunk; only the 1-voxel-thick boundary shell crosses chunks at
            // most once per probe. Cache hit rate is ~95%+, replacing each
            // hash probe (~30-50 ns) with a pointer compare (~3 ns).
            let mut nbr_cache = ChunkSampleCache::new();

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
                            if let Some(neighbor) = nbr_cache.material(density_fields, wx + dx, wy + dy, wz + dz, chunk_size) {
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
    let actual_output = candidates.len() as u32 + result.hornfels_placed + result.skarn_placed
        + result.amphibolite_placed + result.veins_placed;
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
