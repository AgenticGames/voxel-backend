use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use glam::Vec3;
use rayon::prelude::*;
use voxel_core::dual_contouring::mesh_gen::generate_mesh;
use voxel_core::dual_contouring::solve::solve_dc_vertices;
use voxel_core::hermite::HermiteData;
use voxel_core::material::Material;
use voxel_core::mesh::Mesh;
use voxel_core::octree::node::VoxelSample;
use voxel_core::stress::{StressField, SupportField, SupportType};
use voxel_gen::config::{GenerationConfig, StressConfig};
use voxel_gen::density::DensityField;
use voxel_gen::hermite_extract::extract_hermite_data;
use voxel_gen::region_gen::{self, region_key, ChunkSeamData};
use voxel_gen::worm::path::WormSegment;

use crate::convert::convert_mesh_to_ue_scaled;
use crate::delta::{ChunkSnapshot, ModificationTracker, WorldSaveData};
use crate::stress::CollapseEvent;
use crate::types::ConvertedMesh;

/// Result from combined cavern location search.
pub struct CavernLocations {
    pub spring: Vec3,
    pub chrysalis: Vec3,
    pub spawn: Vec3,
}

/// Per-chunk cached data needed for mining and re-meshing.
pub struct ChunkStore {
    pub density_fields: HashMap<(i32, i32, i32), DensityField>,
    pub hermite_data: HashMap<(i32, i32, i32), HermiteData>,
    /// Tracks which regions have had their densities generated (with global worms).
    generated_regions: HashSet<(i32, i32, i32)>,
    /// Per-chunk seam data (DC vertices + boundary edges) for seam stitching.
    pub chunk_seam_data: HashMap<(i32, i32, i32), ChunkSeamData>,
    /// Cached base meshes (pre-seam) for fast seam pass reuse.
    pub base_meshes: HashMap<(i32, i32, i32), Mesh>,
    /// Per-chunk stress data for the collapse system.
    pub stress_fields: HashMap<(i32, i32, i32), StressField>,
    /// Per-chunk support structure data.
    pub support_fields: HashMap<(i32, i32, i32), SupportField>,
    /// Tracks which cells have been terraced for building placement.
    pub terraced_cells: HashSet<(i32, i32, i32)>,
    /// Maps (x, z) → floor_y for terraced columns (fast nearby-terrace lookup).
    pub terraced_columns: HashMap<(i32, i32), i32>,
    /// Worm paths per region key, for cross-region worm sharing.
    pub region_worm_paths: HashMap<(i32, i32, i32), Vec<Vec<WormSegment>>>,
    /// Per-chunk crystal placement data (computed during generation).
    pub crystal_placements: HashMap<(i32, i32, i32), Vec<voxel_gen::CrystalPlacement>>,
    /// Per-chunk mushroom placement data (computed once during generation,
    /// then re-emitted on every remesh so cosmetic mushrooms persist
    /// across mining/sleep updates).
    pub mushroom_placements: HashMap<(i32, i32, i32), Vec<voxel_gen::MushroomPlacement>>,
    /// Region size for computing region keys (needed by unload).
    pub region_size: i32,
    /// Localized stress recalculation events (queued by mining).
    pub stress_dirty_events: Vec<voxel_core::stress::StressDirtyEvent>,
    /// Timestamp of the last mine action that dirtied stress.
    pub stress_dirty_time: Option<std::time::Instant>,
    /// Tracks which chunks have been modified by mining/flatten/sleep.
    pub modification_tracker: ModificationTracker,
    /// Density snapshots preserved from unloaded modified chunks (for save).
    pub preserved_snapshots: BTreeMap<(i32, i32, i32), ChunkSnapshot>,
    /// Pending snapshots loaded from a save file — applied as chunks are generated.
    pub pending_snapshots: Option<WorldSaveData>,
    /// Round 7: content hash of the last ChunkMesh we sent over FFI per chunk.
    /// Seam passes often regenerate identical mesh content (6.5× avg per chunk
    /// during mining when the dirty neighborhood doesn't actually touch each
    /// neighbor's geometry). Hash-compare combined (base + seam) mesh before
    /// sending; skip the FFI round-trip when content is identical. UE's
    /// hash-skip catches these on its side, this prevents Rust from even
    /// doing the convert + bucket_by_material + FFI send.
    pub last_sent_mesh_hash: HashMap<(i32, i32, i32), u64>,
    /// Creative-mode brush undo stack — each stroke captures pre-state of all
    /// chunks in the brush AABB. `voxel_request_undo` pops the last stroke and
    /// restores those snapshots in-place. Bounded by `undo_max_depth`.
    pub undo_stack: std::collections::VecDeque<crate::brushes::UndoStroke>,
    pub undo_max_depth: usize,
    /// Editor-authored scripted collapse triggers (tunnel gates, boss arena
    /// pillars). Evaluated after every carve event; firing one synthesizes
    /// a `CollapseEventV2` and runs the existing cinematic pipeline.
    /// Persisted in the world save alongside chunk snapshots.
    pub triggers: Vec<crate::triggers::EditorCollapseTrigger>,
    /// Monotonic id allocator for triggers. Survives across save/load.
    pub next_trigger_id: u32,
    /// Trigger ids the player/editor has flagged for "fire now" preview.
    /// The stress queue eval skips `should_fire` for ids in this set —
    /// they fire unconditionally on the next tick. Cleared when consumed.
    /// Never persisted (transient editor convenience).
    pub force_fire_trigger_ids: Vec<u32>,
}

impl ChunkStore {
    pub fn new(region_size: i32) -> Self {
        Self {
            density_fields: HashMap::new(),
            hermite_data: HashMap::new(),
            generated_regions: HashSet::new(),
            chunk_seam_data: HashMap::new(),
            base_meshes: HashMap::new(),
            stress_fields: HashMap::new(),
            support_fields: HashMap::new(),
            terraced_cells: HashSet::new(),
            terraced_columns: HashMap::new(),
            region_worm_paths: HashMap::new(),
            crystal_placements: HashMap::new(),
            mushroom_placements: HashMap::new(),
            region_size,
            stress_dirty_events: Vec::new(),
            stress_dirty_time: None,
            modification_tracker: ModificationTracker::new(),
            preserved_snapshots: BTreeMap::new(),
            pending_snapshots: None,
            last_sent_mesh_hash: HashMap::new(),
            undo_stack: std::collections::VecDeque::new(),
            undo_max_depth: 64,
            triggers: Vec::new(),
            next_trigger_id: 1,
            force_fire_trigger_ids: Vec::new(),
        }
    }

    /// Allocate the next available trigger id (does not insert).
    pub fn alloc_trigger_id(&mut self) -> u32 {
        let id = self.next_trigger_id;
        self.next_trigger_id = self.next_trigger_id.saturating_add(1);
        id
    }

    /// Look up a trigger by id.
    pub fn find_trigger(&self, id: u32) -> Option<&crate::triggers::EditorCollapseTrigger> {
        self.triggers.iter().find(|t| t.id == id)
    }

    /// Mutable lookup of a trigger by id.
    pub fn find_trigger_mut(
        &mut self,
        id: u32,
    ) -> Option<&mut crate::triggers::EditorCollapseTrigger> {
        self.triggers.iter_mut().find(|t| t.id == id)
    }

    /// Remove a trigger by id. Returns the removed trigger if it existed.
    pub fn remove_trigger(&mut self, id: u32) -> Option<crate::triggers::EditorCollapseTrigger> {
        if let Some(pos) = self.triggers.iter().position(|t| t.id == id) {
            Some(self.triggers.remove(pos))
        } else {
            None
        }
    }

    pub fn has_density(&self, key: &(i32, i32, i32)) -> bool {
        self.density_fields.contains_key(key)
    }

    pub fn is_region_generated(&self, region_key: &(i32, i32, i32)) -> bool {
        self.generated_regions.contains(region_key)
    }

    pub fn mark_region_generated(&mut self, region_key: (i32, i32, i32)) {
        self.generated_regions.insert(region_key);
    }

    /// Store worm paths for a region key (used for cross-region sharing).
    pub fn store_region_worms(&mut self, region_key: (i32, i32, i32), paths: Vec<Vec<WormSegment>>) {
        self.region_worm_paths.insert(region_key, paths);
    }

    /// Get all stored region worm paths for forward sharing.
    pub fn get_all_region_worm_paths(&self) -> &HashMap<(i32, i32, i32), Vec<Vec<WormSegment>>> {
        &self.region_worm_paths
    }

    pub fn chunks_loaded(&self) -> usize {
        self.density_fields.len()
    }

    /// Sync boundary density planes between dirty chunks and their neighbors.
    /// Extends `dirty_bounds` with extra neighbor chunks that need remeshing.
    pub fn sync_boundaries(
        &mut self,
        dirty_bounds: &mut Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)>,
        chunk_size: usize,
    ) {
        let extra = sync_boundary_density(&mut self.density_fields, dirty_bounds, chunk_size);
        dirty_bounds.extend(extra);
    }

    pub fn insert(&mut self, key: (i32, i32, i32), density: DensityField, hermite: HermiteData) {
        let size = density.size;
        self.density_fields.insert(key, density);
        self.hermite_data.insert(key, hermite);
        // Initialize stress and support fields if not already present
        self.stress_fields.entry(key).or_insert_with(|| StressField::new(size));
        self.support_fields.entry(key).or_insert_with(|| SupportField::new(size));
    }

    pub fn unload(&mut self, key: (i32, i32, i32)) {
        // Preserve density+painted-stress snapshot if this chunk was modified
        // (mining/flatten/sleep/PaintStress brush). Capturing the painted overlay
        // here means it round-trips through unload→reload without leaking memory.
        if self.modification_tracker.dirty_chunks.contains(&key) {
            if let Some(df) = self.density_fields.get(&key) {
                let sf = self.stress_fields.get(&key);
                self.preserved_snapshots.insert(key, ChunkSnapshot::from_chunk(df, sf));
            }
            self.modification_tracker.remove(&key);
        }
        self.density_fields.remove(&key);
        self.hermite_data.remove(&key);
        self.chunk_seam_data.remove(&key);
        self.base_meshes.remove(&key);
        self.stress_fields.remove(&key);
        self.support_fields.remove(&key);
        self.crystal_placements.remove(&key);
        self.last_sent_mesh_hash.remove(&key);
        // Events referencing this chunk will still fire but the chunk won't be found
        // in density_fields during recalc — harmless skip.

        // Clear region flag immediately — region is no longer intact.
        // Next generate will re-run region gen; has_density() guard
        // prevents overwriting siblings that are still loaded.
        let rk = region_key(key.0, key.1, key.2, self.region_size);
        self.generated_regions.remove(&rk);

        // Clean up worm paths if no other chunks in this region remain loaded
        let region_base_x = rk.0 * self.region_size;
        let region_base_y = rk.1 * self.region_size;
        let region_base_z = rk.2 * self.region_size;
        let any_remaining = (0..self.region_size).any(|dz| {
            (0..self.region_size).any(|dy| {
                (0..self.region_size).any(|dx| {
                    self.density_fields.contains_key(&(
                        region_base_x + dx,
                        region_base_y + dy,
                        region_base_z + dz,
                    ))
                })
            })
        });
        if !any_remaining {
            self.region_worm_paths.remove(&rk);
        }
    }

    /// Queue a localized stress recalculation event (called after mining).
    /// center: mine point in world voxel coords, radius: effective stress radius in voxels.
    pub fn queue_stress_dirty(&mut self, center: (i32, i32, i32), radius: i32) {
        self.stress_dirty_events.push(voxel_core::stress::StressDirtyEvent { center, radius });
        self.stress_dirty_time = Some(std::time::Instant::now());
    }

    /// Legacy: queue dirty chunks (used by flatten/sleep paths that don't have a mine center).
    /// Converts each chunk center to a large-radius event covering the full chunk.
    pub fn queue_stress_dirty_chunks(&mut self, chunk_keys: &[(i32, i32, i32)], chunk_size: usize) {
        let half = chunk_size as i32 / 2;
        let radius = chunk_size as i32 + 22; // Full chunk + span search + air decay
        for &(cx, cy, cz) in chunk_keys {
            let center = (cx * chunk_size as i32 + half, cy * chunk_size as i32 + half, cz * chunk_size as i32 + half);
            self.stress_dirty_events.push(voxel_core::stress::StressDirtyEvent { center, radius });
        }
        self.stress_dirty_time = Some(std::time::Instant::now());
    }

    /// Drain the stress dirty queue if the deferred timer has elapsed.
    /// Returns None if timer hasn't elapsed or queue is empty.
    pub fn drain_stress_dirty(&mut self, defer_secs: f32) -> Option<Vec<voxel_core::stress::StressDirtyEvent>> {
        if let Some(t) = self.stress_dirty_time {
            if t.elapsed().as_secs_f32() >= defer_secs {
                self.stress_dirty_time = None;
                let q = std::mem::take(&mut self.stress_dirty_events);
                if !q.is_empty() {
                    return Some(q);
                }
            }
        }
        None
    }

    /// Return mutable references to density, stress, and support fields simultaneously.
    /// Needed by the sleep system which requires write access to all three at once.
    pub fn sleep_fields_mut(
        &mut self,
    ) -> (
        &mut HashMap<(i32, i32, i32), DensityField>,
        &mut HashMap<(i32, i32, i32), StressField>,
        &mut HashMap<(i32, i32, i32), SupportField>,
    ) {
        (&mut self.density_fields, &mut self.stress_fields, &mut self.support_fields)
    }

    /// Cache seam data for a chunk.
    pub fn add_seam_data(
        &mut self,
        chunk: (i32, i32, i32),
        seam_data: ChunkSeamData,
    ) {
        self.chunk_seam_data.insert(chunk, seam_data);
    }

    /// Re-mesh dirty chunks using full hermite re-extraction.
    /// Returns converted meshes in UE coordinate space.
    /// Also updates seam data so seam stitching reflects post-mining geometry.
    pub fn remesh_dirty(
        &mut self,
        dirty_chunks: &[((i32, i32, i32), usize, usize, usize, usize, usize, usize)],
        config: &GenerationConfig,
        world_scale: f32,
    ) -> Vec<((i32, i32, i32), ConvertedMesh)> {
        let chunk_size = config.chunk_size;

        // Phase 1: parallel compute. Each chunk's hermite extraction, DC solve,
        // mesh generation, smooth, normals, boundary edges, and UE conversion
        // are pure functions of an immutably-borrowed DensityField. Run them
        // across rayon's pool so multi-chunk flushes (building flatten typically
        // touches 1–8 chunks; sleep collapses can hit dozens) parallelize across
        // cores instead of stalling the store write-lock holder for the full
        // serial walk.
        let voxel_scale = config.voxel_scale();
        let smooth_iters = config.mesh_smooth_iterations;
        let smooth_strength = config.mesh_smooth_strength;
        let smooth_boundary = config.mesh_boundary_smooth;
        let recalc_normals = config.mesh_recalc_normals;

        let outputs: Vec<((i32, i32, i32), HermiteData, Mesh, Vec<Vec3>, Vec<_>, ConvertedMesh)> =
            dirty_chunks
                .par_iter()
                .filter_map(|&(key, _min_x, _min_y, _min_z, _max_x, _max_y, _max_z)| {
                    let density = self.density_fields.get(&key)?;

                    // Full re-extraction ensures no stale edges from smoothing
                    // boundary effects.
                    let hermite = extract_hermite_data(density);
                    let cell_size = density.size - 1;
                    let dc_vertices = solve_dc_vertices(&hermite, cell_size);
                    let mut mesh = generate_mesh(&hermite, &dc_vertices, cell_size);
                    mesh.smooth(smooth_iters, smooth_strength, smooth_boundary, Some(cell_size));
                    if recalc_normals > 0 { mesh.recalculate_normals(); }

                    let boundary_edges = region_gen::extract_boundary_edges(&hermite, chunk_size);

                    let mut converted = convert_mesh_to_ue_scaled(&mesh, voxel_scale, world_scale);
                    crate::convert::bucket_mesh_by_material(&mut converted);

                    Some((key, hermite, mesh, dc_vertices, boundary_edges, converted))
                })
                .collect();

        // Phase 2: serial write-back. The HashMaps aren't shared mutably across
        // threads, so apply the per-chunk results here in one pass.
        let mut results = Vec::with_capacity(outputs.len());
        for (key, hermite, mesh, dc_vertices, boundary_edges, converted) in outputs {
            self.hermite_data.insert(key, hermite);
            self.base_meshes.insert(key, mesh);
            self.chunk_seam_data.insert(
                key,
                ChunkSeamData {
                    dc_vertices,
                    world_origin: Vec3::ZERO,
                    boundary_edges,
                },
            );
            results.push((key, converted));
        }

        results
    }

    /// Place a support structure at a world position.
    /// All strut types can be placed in air or solid voxels.
    /// Returns (success, collapse_events, dirty_chunks_with_bounds).
    pub fn place_support(
        &mut self,
        world_pos: (i32, i32, i32),
        support_type: SupportType,
        _stress_config: &StressConfig,
        chunk_size: usize,
    ) -> (bool, Vec<CollapseEvent>, Vec<((i32, i32, i32), (usize, usize, usize, usize, usize, usize))>) {
        let cs = chunk_size as i32;
        let cx = world_pos.0.div_euclid(cs);
        let cy = world_pos.1.div_euclid(cs);
        let cz = world_pos.2.div_euclid(cs);
        let lx = world_pos.0.rem_euclid(cs) as usize;
        let ly = world_pos.1.rem_euclid(cs) as usize;
        let lz = world_pos.2.rem_euclid(cs) as usize;
        let key = (cx, cy, cz);

        // Only SupportType::None is invalid for placement
        if support_type == SupportType::None {
            return (false, Vec::new(), Vec::new());
        }

        // Place support
        if let Some(sf) = self.support_fields.get_mut(&key) {
            sf.set(lx, ly, lz, support_type);
        } else {
            let size = chunk_size + 1;
            let mut sf = SupportField::new(size);
            sf.set(lx, ly, lz, support_type);
            self.support_fields.insert(key, sf);
        }

        // Stress deferred to sleep-only — just remesh the affected chunk
        let dirty_with_bounds = vec![
            (key, (0usize, 0usize, 0usize, chunk_size, chunk_size, chunk_size))
        ];

        (true, Vec::new(), dirty_with_bounds)
    }

    /// Remove a support structure at a world position.
    /// Returns (removed_type, collapse_events, dirty_chunks_with_bounds).
    pub fn remove_support(
        &mut self,
        world_pos: (i32, i32, i32),
        _stress_config: &StressConfig,
        chunk_size: usize,
    ) -> (Option<SupportType>, Vec<CollapseEvent>, Vec<((i32, i32, i32), (usize, usize, usize, usize, usize, usize))>) {
        let cs = chunk_size as i32;
        let cx = world_pos.0.div_euclid(cs);
        let cy = world_pos.1.div_euclid(cs);
        let cz = world_pos.2.div_euclid(cs);
        let lx = world_pos.0.rem_euclid(cs) as usize;
        let ly = world_pos.1.rem_euclid(cs) as usize;
        let lz = world_pos.2.rem_euclid(cs) as usize;
        let key = (cx, cy, cz);

        // Get current support type
        let old_type = self.support_fields
            .get(&key)
            .map(|sf| sf.get(lx, ly, lz))
            .unwrap_or(SupportType::None);

        if old_type == SupportType::None {
            return (None, Vec::new(), Vec::new());
        }

        // Remove support
        if let Some(sf) = self.support_fields.get_mut(&key) {
            sf.set(lx, ly, lz, SupportType::None);
        }

        // Stress deferred to sleep-only — just remesh the affected chunk
        let dirty_with_bounds = vec![
            (key, (0usize, 0usize, 0usize, chunk_size, chunk_size, chunk_size))
        ];

        (Some(old_type), Vec::new(), dirty_with_bounds)
    }

    /// Find the best spring location (wall seep / ceiling drip) near the player.
    /// Scans all loaded density fields for open cavern cells adjacent to a wall.
    /// Returns the world-space (Rust coords) position of the best candidate.
    /// Parallelized with rayon; skips solid chunks and guards geode checks with metadata.
    pub fn find_spring_location(&self, player_pos: Vec3, chunk_size: usize, effective_bounds: f32) -> Option<Vec3> {
        let cs = effective_bounds;
        let cs_i = chunk_size as i32;

        let chunks: Vec<_> = self.density_fields.iter().collect();

        chunks.par_iter()
            .filter_map(|(&(cx, cy, cz), density)| {
                // Skip all-solid chunks (no air cells to search)
                if density.air_cell_count == 0 {
                    return None;
                }

                let origin = Vec3::new(cx as f32 * cs, cy as f32 * cs, cz as f32 * cs);
                let chunk_has_geode = density.has_geode_material;
                let mut best_score: f32 = -1.0;
                let mut best_pos: Option<Vec3> = None;

                for z in 1..(chunk_size - 1) {
                    for y in 1..(chunk_size - 1) {
                        for x in 1..(chunk_size - 1) {
                            let sample = density.get(x, y, z);
                            if sample.material.is_solid() {
                                continue;
                            }

                            let mut air_count: u32 = 0;
                            for dz in -1i32..=1 {
                                for dy in -1i32..=1 {
                                    for dx in -1i32..=1 {
                                        if dx == 0 && dy == 0 && dz == 0 {
                                            continue;
                                        }
                                        let nx = (x as i32 + dx) as usize;
                                        let ny = (y as i32 + dy) as usize;
                                        let nz = (z as i32 + dz) as usize;
                                        if !density.get(nx, ny, nz).material.is_solid() {
                                            air_count += 1;
                                        }
                                    }
                                }
                            }

                            if air_count < 15 {
                                continue;
                            }

                            // Guard geode check behind chunk-level metadata
                            if chunk_has_geode {
                                let wx = cx * cs_i + x as i32;
                                let wy = cy * cs_i + y as i32;
                                let wz = cz * cs_i + z as i32;
                                if self.is_geode_interior(wx, wy, wz, cs_i) {
                                    continue;
                                }
                            }

                            let solid_above = density.get(x, y + 1, z).material.is_solid();
                            let solid_below = density.get(x, y.wrapping_sub(1), z).material.is_solid();
                            let solid_xp = density.get(x + 1, y, z).material.is_solid();
                            let solid_xn = density.get(x.wrapping_sub(1), y, z).material.is_solid();
                            let solid_zp = density.get(x, y, z + 1).material.is_solid();
                            let solid_zn = density.get(x, y, z.wrapping_sub(1)).material.is_solid();

                            let has_solid_side = solid_xp || solid_xn || solid_zp || solid_zn;
                            let air_below = !solid_below;

                            if !solid_above && !has_solid_side {
                                continue;
                            }

                            let wall_bonus = if has_solid_side && !solid_above {
                                2.0_f32
                            } else if solid_above {
                                1.5
                            } else {
                                1.0
                            };

                            let air_below_bonus = if air_below { 1.5_f32 } else { 1.0 };

                            let mut air_above: u32 = 0;
                            for dy in 1..=15u32 {
                                let ny = y + dy as usize;
                                if ny >= density.size {
                                    break;
                                }
                                if !density.get(x, ny, z).material.is_solid() {
                                    air_above += 1;
                                } else {
                                    break;
                                }
                            }
                            let clearance_bonus = if air_above >= 10 {
                                2.0_f32
                            } else if air_above >= 5 {
                                1.5
                            } else {
                                0.5
                            };

                            let world_pos = origin + Vec3::new(x as f32, y as f32, z as f32);
                            let distance = (world_pos - player_pos).length();

                            let score = (air_count as f32) * wall_bonus * air_below_bonus * clearance_bonus
                                / (1.0 + distance);

                            if score > best_score {
                                best_score = score;
                                best_pos = Some(world_pos);
                            }
                        }
                    }
                }

                best_pos.map(|pos| (best_score, pos))
            })
            .reduce_with(|a, b| if a.0 > b.0 { a } else { b })
            .map(|(_, pos)| pos)
    }

    /// Find surface-facing ore voxels near the player, sorted by distance.
    ///
    /// All inputs/outputs are in Rust (voxel) space. The caller is responsible
    /// for converting between UE world space and voxel space.
    ///
    /// - `radius` is in voxel units.
    /// - `material_filter` of `0xFF` means "any ore" (`Material::is_ore()`).
    ///   Any other value is treated as a specific `Material as u8` to match.
    /// - "Surface" means at least one of the 6 face-neighbors is non-solid.
    /// - Returns up to `max_results` voxel centers sorted by squared distance.
    ///
    /// Parallelized with rayon; skips chunks via `has_ore_material` early reject
    /// and a chunk-center broad-phase distance check.
    pub fn find_ore_voxels(
        &self,
        player_pos: Vec3,
        radius: f32,
        material_filter: u8,
        max_results: usize,
        chunk_size: usize,
        effective_bounds: f32,
    ) -> Vec<(Vec3, u8)> {
        if max_results == 0 || radius <= 0.0 || chunk_size < 2 {
            return Vec::new();
        }

        let cs = effective_bounds;
        let voxel_scale = cs / chunk_size as f32;
        let radius_sq = radius * radius;

        // Broad-phase: skip chunks whose center is farther than radius + half-diagonal.
        let half_diag = cs * 0.5 * 3.0_f32.sqrt();
        let broad_dist = radius + half_diag;
        let broad_dist_sq = broad_dist * broad_dist;

        let chunks: Vec<_> = self.density_fields.iter().collect();

        let mut all: Vec<(Vec3, u8, f32)> = chunks
            .par_iter()
            .flat_map_iter(|(&(cx, cy, cz), density)| {
                let mut local: Vec<(Vec3, u8, f32)> = Vec::new();

                // Chunk-level early reject — no ore material anywhere in chunk.
                if !density.has_ore_material {
                    return local.into_iter();
                }

                let origin = Vec3::new(cx as f32 * cs, cy as f32 * cs, cz as f32 * cs);
                let chunk_center = origin + Vec3::splat(cs * 0.5);
                if (chunk_center - player_pos).length_squared() > broad_dist_sq {
                    return local.into_iter();
                }

                // Inner grid scan, leaving a 1-voxel border so 6-neighbor lookups
                // stay in-bounds. Ores on the absolute chunk boundary are missed;
                // acceptable for v1 — the player can move a step to surface them.
                let end = chunk_size; // exclusive — neighbor x+1 < density.size
                for z in 1..end {
                    for y in 1..end {
                        for x in 1..end {
                            let sample = density.get(x, y, z);
                            let mat = sample.material;
                            if !mat.is_solid() {
                                continue;
                            }

                            // Material filter.
                            if material_filter == 0xFF {
                                if !mat.is_ore() {
                                    continue;
                                }
                            } else if (mat as u8) != material_filter {
                                continue;
                            }

                            // Per-face air check — feeds both the surface filter
                            // AND the exposed-centroid calc below.
                            let n_xp = !density.get(x + 1, y, z).material.is_solid();
                            let n_xn = !density.get(x - 1, y, z).material.is_solid();
                            let n_yp = !density.get(x, y + 1, z).material.is_solid();
                            let n_yn = !density.get(x, y - 1, z).material.is_solid();
                            let n_zp = !density.get(x, y, z + 1).material.is_solid();
                            let n_zn = !density.get(x, y, z - 1).material.is_solid();

                            let mut sum_dir = Vec3::ZERO;
                            let mut n_exposed = 0u32;
                            if n_xp { sum_dir.x += 1.0; n_exposed += 1; }
                            if n_xn { sum_dir.x -= 1.0; n_exposed += 1; }
                            if n_yp { sum_dir.y += 1.0; n_exposed += 1; }
                            if n_yn { sum_dir.y -= 1.0; n_exposed += 1; }
                            if n_zp { sum_dir.z += 1.0; n_exposed += 1; }
                            if n_zn { sum_dir.z -= 1.0; n_exposed += 1; }
                            if n_exposed == 0 {
                                continue; // fully buried — not visible
                            }

                            // Geometric voxel center in chunk-local space.
                            let cx_local = (x as f32 + 0.5) * voxel_scale;
                            let cy_local = (y as f32 + 0.5) * voxel_scale;
                            let cz_local = (z as f32 + 0.5) * voxel_scale;

                            // Visible centroid: shift from the voxel's geometric
                            // center toward the average of its exposed-face normals
                            // by half a voxel. For one-face-exposed voxels the
                            // marker hugs that face; for corner voxels it lands on
                            // the visible corner; for fully-exposed voxels the
                            // shift is zero. Without this, a vein with most of its
                            // mass behind the wall plants the bracket inside the
                            // rock and reads as "buggy".
                            let half_scale = 0.5 * voxel_scale;
                            let avg = sum_dir / n_exposed as f32;
                            let world_pos = origin
                                + Vec3::new(cx_local, cy_local, cz_local)
                                + avg * half_scale;

                            let dist_sq = (world_pos - player_pos).length_squared();
                            if dist_sq > radius_sq {
                                continue;
                            }

                            local.push((world_pos, mat as u8, dist_sq));
                        }
                    }
                }

                local.into_iter()
            })
            .collect();

        all.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        all.truncate(max_results);
        all.into_iter().map(|(p, m, _)| (p, m)).collect()
    }

    /// Check whether a bounding box of air voxels exists at a world position,
    /// with a solid floor below. Handles cross-chunk boundaries via div_euclid/rem_euclid.
    /// Returns false if any required chunk is not loaded (conservative).
    pub fn check_clearance(
        &self,
        wx: i32,
        wy: i32,
        wz: i32,
        height: i32,
        radius: i32,
        chunk_size: i32,
    ) -> bool {
        // Floor check: voxel at (wx, wy-1, wz) must be solid
        {
            let cx = wx.div_euclid(chunk_size);
            let cy = (wy - 1).div_euclid(chunk_size);
            let cz = wz.div_euclid(chunk_size);
            let lx = wx.rem_euclid(chunk_size) as usize;
            let ly = (wy - 1).rem_euclid(chunk_size) as usize;
            let lz = wz.rem_euclid(chunk_size) as usize;
            match self.density_fields.get(&(cx, cy, cz)) {
                Some(df) => {
                    if !df.get(lx, ly, lz).material.is_solid() {
                        return false;
                    }
                }
                None => return false,
            }
        }

        // Air column: all voxels in [wy..wy+height] x [wx-radius..wx+radius] x [wz-radius..wz+radius]
        for dy in 0..height {
            for dx in -radius..=radius {
                for dz in -radius..=radius {
                    let vx = wx + dx;
                    let vy = wy + dy;
                    let vz = wz + dz;
                    let cx = vx.div_euclid(chunk_size);
                    let cy = vy.div_euclid(chunk_size);
                    let cz = vz.div_euclid(chunk_size);
                    let lx = vx.rem_euclid(chunk_size) as usize;
                    let ly = vy.rem_euclid(chunk_size) as usize;
                    let lz = vz.rem_euclid(chunk_size) as usize;
                    match self.density_fields.get(&(cx, cy, cz)) {
                        Some(df) => {
                            if df.get(lx, ly, lz).material.is_solid() {
                                return false;
                            }
                        }
                        None => return false,
                    }
                }
            }
        }

        true
    }

    /// Find a validated spawn location for the player capsule.
    /// Parallelized with rayon; skips solid chunks and guards geode checks with metadata.
    pub fn find_spawn_location(
        &self,
        target: Vec3,
        exclude_center: Vec3,
        exclude_radius: f32,
        chunk_size: usize,
        effective_bounds: f32,
        height: i32,
        radius: i32,
    ) -> Option<Vec3> {
        let cs = effective_bounds;
        let excl_r2 = exclude_radius * exclude_radius;
        let cs_i = chunk_size as i32;

        let chunks: Vec<_> = self.density_fields.iter().collect();

        chunks.par_iter()
            .filter_map(|(&(cx, cy, cz), density)| {
                if density.air_cell_count == 0 {
                    return None;
                }

                let origin = Vec3::new(cx as f32 * cs, cy as f32 * cs, cz as f32 * cs);
                let chunk_has_geode = density.has_geode_material;
                let mut best_dist = f32::MAX;
                let mut best_pos: Option<Vec3> = None;

                for z in 1..(chunk_size - 1) {
                    for y in 1..(chunk_size - 1) {
                        for x in 1..(chunk_size - 1) {
                            let sample = density.get(x, y, z);
                            if sample.material.is_solid() {
                                continue;
                            }

                            if chunk_has_geode {
                                let wx = cx * cs_i + x as i32;
                                let wy = cy * cs_i + y as i32;
                                let wz = cz * cs_i + z as i32;
                                if self.is_geode_interior(wx, wy, wz, cs_i) {
                                    continue;
                                }
                            }

                            let world_pos = origin + Vec3::new(x as f32, y as f32, z as f32);

                            if (world_pos - exclude_center).length_squared() < excl_r2 {
                                continue;
                            }

                            let dist = (world_pos - target).length_squared();
                            if dist >= best_dist {
                                continue;
                            }

                            let wx = cx * cs_i + x as i32;
                            let wy = cy * cs_i + y as i32;
                            let wz = cz * cs_i + z as i32;
                            if self.check_clearance(wx, wy, wz, height, radius, cs_i) {
                                best_dist = dist;
                                best_pos = Some(world_pos);
                            }
                        }
                    }
                }

                best_pos.map(|pos| (best_dist, pos))
            })
            .reduce_with(|a, b| if a.0 < b.0 { a } else { b })
            .map(|(_, pos)| pos)
    }

    /// Find a validated spawn location for the chrysalis (quest giver).
    /// Parallelized with rayon; skips solid chunks and guards geode checks with metadata.
    pub fn find_chrysalis_location(
        &self,
        target: Vec3,
        exclude_center: Vec3,
        exclude_radius: f32,
        chunk_size: usize,
        effective_bounds: f32,
        height: i32,
        radius: i32,
    ) -> Option<Vec3> {
        let cs = effective_bounds;
        let excl_r2 = exclude_radius * exclude_radius;
        let cs_i = chunk_size as i32;

        let chunks: Vec<_> = self.density_fields.iter().collect();

        chunks.par_iter()
            .filter_map(|(&(cx, cy, cz), density)| {
                if density.air_cell_count == 0 {
                    return None;
                }

                let origin = Vec3::new(cx as f32 * cs, cy as f32 * cs, cz as f32 * cs);
                let chunk_has_geode = density.has_geode_material;
                let mut best_dist = f32::MAX;
                let mut best_pos: Option<Vec3> = None;

                for z in 2..(chunk_size - 2) {
                    for y in 1..(chunk_size - 1) {
                        for x in 2..(chunk_size - 2) {
                            let sample = density.get(x, y, z);
                            if sample.material.is_solid() {
                                continue;
                            }

                            if chunk_has_geode {
                                let wx = cx * cs_i + x as i32;
                                let wy = cy * cs_i + y as i32;
                                let wz = cz * cs_i + z as i32;
                                if self.is_geode_interior(wx, wy, wz, cs_i) {
                                    continue;
                                }
                            }

                            let world_pos = origin + Vec3::new(x as f32, y as f32, z as f32);

                            if (world_pos - exclude_center).length_squared() < excl_r2 {
                                continue;
                            }

                            let adj_solid = density.get(x + 1, y, z).material.is_solid()
                                || density.get(x.wrapping_sub(1), y, z).material.is_solid()
                                || density.get(x, y, z + 1).material.is_solid()
                                || density.get(x, y, z.wrapping_sub(1)).material.is_solid();
                            if adj_solid {
                                continue;
                            }

                            let mut near_wall = false;
                            'outer: for ddx in -3i32..=3 {
                                for ddz in -3i32..=3 {
                                    if ddx.abs() <= 1 && ddz.abs() <= 1 {
                                        continue;
                                    }
                                    let nx = (x as i32 + ddx) as usize;
                                    let nz = (z as i32 + ddz) as usize;
                                    if nx < density.size && nz < density.size {
                                        if density.get(nx, y, nz).material.is_solid() {
                                            near_wall = true;
                                            break 'outer;
                                        }
                                    }
                                }
                            }
                            if !near_wall {
                                continue;
                            }

                            let dist = (world_pos - target).length_squared();
                            if dist >= best_dist {
                                continue;
                            }

                            let wx = cx * cs_i + x as i32;
                            let wy = cy * cs_i + y as i32;
                            let wz = cz * cs_i + z as i32;
                            if self.check_clearance(wx, wy, wz, height, radius, cs_i) {
                                best_dist = dist;
                                best_pos = Some(world_pos);
                            }
                        }
                    }
                }

                best_pos.map(|pos| (best_dist, pos))
            })
            .reduce_with(|a, b| if a.0 < b.0 { a } else { b })
            .map(|(_, pos)| pos)
    }

    /// Find a wall-adjacent air cell near `target`.
    /// Parallelized with rayon; skips solid chunks and guards geode checks with metadata.
    pub fn find_wall_location_near(
        &self,
        target: Vec3,
        exclude_center: Vec3,
        exclude_radius: f32,
        chunk_size: usize,
        effective_bounds: f32,
    ) -> Option<Vec3> {
        let cs = effective_bounds;
        let excl_r2 = exclude_radius * exclude_radius;
        let cs_i = chunk_size as i32;

        let chunks: Vec<_> = self.density_fields.iter().collect();

        chunks.par_iter()
            .filter_map(|(&(cx, cy, cz), density)| {
                if density.air_cell_count == 0 {
                    return None;
                }

                let origin = Vec3::new(cx as f32 * cs, cy as f32 * cs, cz as f32 * cs);
                let chunk_has_geode = density.has_geode_material;
                let mut best_dist = f32::MAX;
                let mut best_pos: Option<Vec3> = None;

                for z in 1..(chunk_size - 1) {
                    for y in 1..(chunk_size - 1) {
                        for x in 1..(chunk_size - 1) {
                            let sample = density.get(x, y, z);
                            if sample.material.is_solid() {
                                continue;
                            }

                            if chunk_has_geode {
                                let wx = cx * cs_i + x as i32;
                                let wy = cy * cs_i + y as i32;
                                let wz = cz * cs_i + z as i32;
                                if self.is_geode_interior(wx, wy, wz, cs_i) {
                                    continue;
                                }
                            }

                            let world_pos = origin + Vec3::new(x as f32, y as f32, z as f32);

                            if (world_pos - exclude_center).length_squared() < excl_r2 {
                                continue;
                            }

                            let has_wall = density.get(x + 1, y, z).material.is_solid()
                                || density.get(x.wrapping_sub(1), y, z).material.is_solid()
                                || density.get(x, y + 1, z).material.is_solid()
                                || density.get(x, y.wrapping_sub(1), z).material.is_solid()
                                || density.get(x, y, z + 1).material.is_solid()
                                || density.get(x, y, z.wrapping_sub(1)).material.is_solid();

                            if !has_wall {
                                continue;
                            }

                            let dist = (world_pos - target).length_squared();
                            if dist < best_dist {
                                best_dist = dist;
                                best_pos = Some(world_pos);
                            }
                        }
                    }
                }

                best_pos.map(|pos| (best_dist, pos))
            })
            .reduce_with(|a, b| if a.0 < b.0 { a } else { b })
            .map(|(_, pos)| pos)
    }

    /// Check if an air cell is inside a geode (crystal/amethyst shell nearby).
    /// Scans a 5x5x5 cube (radius 2) around the cell for geode shell materials.
    pub fn is_geode_interior(&self, wx: i32, wy: i32, wz: i32, chunk_size: i32) -> bool {
        for dz in -2..=2i32 {
            for dy in -2..=2i32 {
                for dx in -2..=2i32 {
                    let vx = wx + dx;
                    let vy = wy + dy;
                    let vz = wz + dz;
                    let cx = vx.div_euclid(chunk_size);
                    let cy = vy.div_euclid(chunk_size);
                    let cz = vz.div_euclid(chunk_size);
                    let lx = vx.rem_euclid(chunk_size) as usize;
                    let ly = vy.rem_euclid(chunk_size) as usize;
                    let lz = vz.rem_euclid(chunk_size) as usize;
                    if let Some(df) = self.density_fields.get(&(cx, cy, cz)) {
                        if df.get(lx, ly, lz).material.is_geode_shell() {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// BFS flood-fill from a seed cell to find all connected air cells in the same cavern.
    /// 6-connected, cross-chunk aware. Skips solid voxels, geode interiors, and unloaded chunks.
    /// Returns None if the fill exceeds max_cells (cavern too large for constraint).
    /// Uses per-chunk Vec<bool> bitset for O(1) visited checks instead of HashSet hashing.
    pub fn flood_fill_cavern(
        &self,
        seed_wx: i32,
        seed_wy: i32,
        seed_wz: i32,
        chunk_size: i32,
        max_cells: usize,
    ) -> Option<HashSet<(i32, i32, i32)>> {
        let cs = chunk_size as usize;
        let cells_per_chunk = cs * cs * cs;
        let mut visited_chunks: HashMap<(i32, i32, i32), Vec<bool>> = HashMap::new();
        let mut total_visited: usize = 0;
        let mut queue = VecDeque::new();

        // Helper: check and mark visited in bitset
        let is_visited = |chunks: &HashMap<(i32, i32, i32), Vec<bool>>, ck: (i32, i32, i32), lx: usize, ly: usize, lz: usize| -> bool {
            if let Some(bits) = chunks.get(&ck) {
                bits[lz * cs * cs + ly * cs + lx]
            } else {
                false
            }
        };

        let mark_visited = |chunks: &mut HashMap<(i32, i32, i32), Vec<bool>>, ck: (i32, i32, i32), lx: usize, ly: usize, lz: usize| {
            let bits = chunks.entry(ck).or_insert_with(|| vec![false; cells_per_chunk]);
            bits[lz * cs * cs + ly * cs + lx] = true;
        };

        // Seed
        {
            let cx = seed_wx.div_euclid(chunk_size);
            let cy = seed_wy.div_euclid(chunk_size);
            let cz = seed_wz.div_euclid(chunk_size);
            let lx = seed_wx.rem_euclid(chunk_size) as usize;
            let ly = seed_wy.rem_euclid(chunk_size) as usize;
            let lz = seed_wz.rem_euclid(chunk_size) as usize;
            mark_visited(&mut visited_chunks, (cx, cy, cz), lx, ly, lz);
            total_visited += 1;
        }
        queue.push_back((seed_wx, seed_wy, seed_wz));

        let directions: [(i32, i32, i32); 6] = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ];

        while let Some((wx, wy, wz)) = queue.pop_front() {
            if total_visited > max_cells {
                return None;
            }

            for &(dx, dy, dz) in &directions {
                let nx = wx + dx;
                let ny = wy + dy;
                let nz = wz + dz;

                let cx = nx.div_euclid(chunk_size);
                let cy = ny.div_euclid(chunk_size);
                let cz = nz.div_euclid(chunk_size);
                let lx = nx.rem_euclid(chunk_size) as usize;
                let ly = ny.rem_euclid(chunk_size) as usize;
                let lz = nz.rem_euclid(chunk_size) as usize;
                let ck = (cx, cy, cz);

                if is_visited(&visited_chunks, ck, lx, ly, lz) {
                    continue;
                }

                match self.density_fields.get(&ck) {
                    Some(df) => {
                        if df.get(lx, ly, lz).material.is_solid() {
                            continue;
                        }
                        // Guard geode check behind chunk-level metadata
                        if df.has_geode_material && self.is_geode_interior(nx, ny, nz, chunk_size) {
                            continue;
                        }
                    }
                    None => continue,
                }

                mark_visited(&mut visited_chunks, ck, lx, ly, lz);
                total_visited += 1;
                queue.push_back((nx, ny, nz));
            }
        }

        // Convert bitset back to HashSet for downstream compatibility
        let mut result = HashSet::with_capacity(total_visited);
        for (&(cx, cy, cz), bits) in &visited_chunks {
            for z in 0..cs {
                for y in 0..cs {
                    for x in 0..cs {
                        if bits[z * cs * cs + y * cs + x] {
                            let wx = cx * chunk_size + x as i32;
                            let wy = cy * chunk_size + y as i32;
                            let wz = cz * chunk_size + z as i32;
                            result.insert((wx, wy, wz));
                        }
                    }
                }
            }
        }

        Some(result)
    }

    /// Same as find_spawn_location but constrained to a pre-computed cavern volume.
    /// Parallelized with rayon; skips solid chunks.
    pub fn find_spawn_in_cavern(
        &self,
        cavern: &HashSet<(i32, i32, i32)>,
        target: Vec3,
        exclude_center: Vec3,
        exclude_radius: f32,
        chunk_size: usize,
        effective_bounds: f32,
        height: i32,
        radius: i32,
    ) -> Option<Vec3> {
        let cs = effective_bounds;
        let excl_r2 = exclude_radius * exclude_radius;
        let cs_i = chunk_size as i32;

        let chunks: Vec<_> = self.density_fields.iter().collect();

        chunks.par_iter()
            .filter_map(|(&(cx, cy, cz), density)| {
                if density.air_cell_count == 0 {
                    return None;
                }

                let origin = Vec3::new(cx as f32 * cs, cy as f32 * cs, cz as f32 * cs);
                let mut best_dist = f32::MAX;
                let mut best_pos: Option<Vec3> = None;

                for z in 1..(chunk_size - 1) {
                    for y in 1..(chunk_size - 1) {
                        for x in 1..(chunk_size - 1) {
                            let wx = cx * cs_i + x as i32;
                            let wy = cy * cs_i + y as i32;
                            let wz = cz * cs_i + z as i32;

                            if !cavern.contains(&(wx, wy, wz)) {
                                continue;
                            }

                            let sample = density.get(x, y, z);
                            if sample.material.is_solid() {
                                continue;
                            }

                            let world_pos = origin + Vec3::new(x as f32, y as f32, z as f32);

                            if (world_pos - exclude_center).length_squared() < excl_r2 {
                                continue;
                            }

                            let dist = (world_pos - target).length_squared();
                            if dist >= best_dist {
                                continue;
                            }

                            if self.check_clearance(wx, wy, wz, height, radius, cs_i) {
                                best_dist = dist;
                                best_pos = Some(world_pos);
                            }
                        }
                    }
                }

                best_pos.map(|pos| (best_dist, pos))
            })
            .reduce_with(|a, b| if a.0 < b.0 { a } else { b })
            .map(|(_, pos)| pos)
    }

    /// Same as find_chrysalis_location but constrained to a pre-computed cavern volume.
    /// Parallelized with rayon; skips solid chunks.
    pub fn find_chrysalis_in_cavern(
        &self,
        cavern: &HashSet<(i32, i32, i32)>,
        target: Vec3,
        exclude_center: Vec3,
        exclude_radius: f32,
        chunk_size: usize,
        effective_bounds: f32,
        height: i32,
        radius: i32,
    ) -> Option<Vec3> {
        let cs = effective_bounds;
        let excl_r2 = exclude_radius * exclude_radius;
        let cs_i = chunk_size as i32;

        let chunks: Vec<_> = self.density_fields.iter().collect();

        chunks.par_iter()
            .filter_map(|(&(cx, cy, cz), density)| {
                if density.air_cell_count == 0 {
                    return None;
                }

                let origin = Vec3::new(cx as f32 * cs, cy as f32 * cs, cz as f32 * cs);
                let mut best_dist = f32::MAX;
                let mut best_pos: Option<Vec3> = None;

                for z in 2..(chunk_size - 2) {
                    for y in 1..(chunk_size - 1) {
                        for x in 2..(chunk_size - 2) {
                            let wx = cx * cs_i + x as i32;
                            let wy = cy * cs_i + y as i32;
                            let wz = cz * cs_i + z as i32;

                            if !cavern.contains(&(wx, wy, wz)) {
                                continue;
                            }

                            let sample = density.get(x, y, z);
                            if sample.material.is_solid() {
                                continue;
                            }

                            let world_pos = origin + Vec3::new(x as f32, y as f32, z as f32);

                            if (world_pos - exclude_center).length_squared() < excl_r2 {
                                continue;
                            }

                            let adj_solid = density.get(x + 1, y, z).material.is_solid()
                                || density.get(x.wrapping_sub(1), y, z).material.is_solid()
                                || density.get(x, y, z + 1).material.is_solid()
                                || density.get(x, y, z.wrapping_sub(1)).material.is_solid();
                            if adj_solid {
                                continue;
                            }

                            let mut near_wall = false;
                            'outer: for ddx in -3i32..=3 {
                                for ddz in -3i32..=3 {
                                    if ddx.abs() <= 1 && ddz.abs() <= 1 {
                                        continue;
                                    }
                                    let nx = (x as i32 + ddx) as usize;
                                    let nz = (z as i32 + ddz) as usize;
                                    if nx < density.size && nz < density.size {
                                        if density.get(nx, y, nz).material.is_solid() {
                                            near_wall = true;
                                            break 'outer;
                                        }
                                    }
                                }
                            }
                            if !near_wall {
                                continue;
                            }

                            let dist = (world_pos - target).length_squared();
                            if dist >= best_dist {
                                continue;
                            }

                            if self.check_clearance(wx, wy, wz, height, radius, cs_i) {
                                best_dist = dist;
                                best_pos = Some(world_pos);
                            }
                        }
                    }
                }

                best_pos.map(|pos| (best_dist, pos))
            })
            .reduce_with(|a, b| if a.0 < b.0 { a } else { b })
            .map(|(_, pos)| pos)
    }

    /// Combined entry point: find spring, flood-fill cavern, then find chrysalis and spawn
    /// constrained to the same cavern volume.
    /// Falls back to independent (geode-filtered) searches if flood fill overflows.
    pub fn find_cavern_locations(
        &self,
        player_pos: Vec3,
        chunk_size: usize,
        effective_bounds: f32,
    ) -> Option<CavernLocations> {
        // Step 1: Find spring (already geode-filtered)
        let spring = self.find_spring_location(player_pos, chunk_size, effective_bounds)?;

        let cs_i = chunk_size as i32;
        let spring_wx = spring.x as i32;
        let spring_wy = spring.y as i32;
        let spring_wz = spring.z as i32;

        // Step 2: Flood-fill cavern from spring
        let cavern_opt = self.flood_fill_cavern(spring_wx, spring_wy, spring_wz, cs_i, 50_000);

        if let Some(ref cavern) = cavern_opt {
            // Step 3: Find chrysalis in same cavern
            let chrysalis = self.find_chrysalis_in_cavern(
                cavern, spring, spring, 30.0,
                chunk_size, effective_bounds, 4, 2,
            );

            if let Some(chr) = chrysalis {
                // Step 4: Find spawn in same cavern (excluding chrysalis)
                let spawn = self.find_spawn_in_cavern(
                    cavern, spring, chr, 20.0,
                    chunk_size, effective_bounds, 13, 3,
                );

                if let Some(sp) = spawn {
                    return Some(CavernLocations { spring, chrysalis: chr, spawn: sp });
                }
            }
        }

        // Fallback: independent searches (still geode-filtered)
        let chrysalis = self.find_chrysalis_location(
            spring, spring, 30.0,
            chunk_size, effective_bounds, 4, 2,
        )?;

        let spawn = self.find_spawn_location(
            spring, chrysalis, 20.0,
            chunk_size, effective_bounds, 13, 3,
        )?;

        Some(CavernLocations { spring, chrysalis, spawn })
    }

    /// Sync this chunk's boundaries with all 26 neighbors (faces + edges +
    /// corners). Used by the diagnostic Force Resync button — more thorough
    /// than `sync_cross_region_densities` (which only does 6 face neighbors)
    /// because seam mismatches at edge/corner cells are common after mid-
    /// session brush ops. Returns the set of chunks whose density was
    /// modified — caller re-extracts hermite + remeshes those.
    pub fn sync_chunk_full_boundaries(
        &mut self,
        chunk: (i32, i32, i32),
        chunk_size: usize,
    ) -> Vec<(i32, i32, i32)> {
        let gs = chunk_size;
        // 13 forward offsets covering faces (3) + edges (6) + corners (4).
        // Each pair of (chunk, neighbor) syncs once; both sides written.
        let offsets: [(i32, i32, i32); 13] = [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1),
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        ];
        // Also include the reverse 13 so we cover ALL 26 face/edge/corner
        // neighbors of `chunk` (not just the forward half-set).
        let all_offsets: Vec<(i32, i32, i32)> = offsets
            .iter()
            .copied()
            .chain(offsets.iter().map(|&(dx, dy, dz)| (-dx, -dy, -dz)))
            .collect();

        let mut updates: Vec<((i32, i32, i32), usize, usize, usize, f32, voxel_core::material::Material)> = Vec::new();
        let mut dirty: HashSet<(i32, i32, i32)> = HashSet::new();

        for (dx, dy, dz) in all_offsets {
            let neighbor = (chunk.0 + dx, chunk.1 + dy, chunk.2 + dz);
            if !self.density_fields.contains_key(&neighbor) {
                continue;
            }
            // Build per-axis pair iterators: 1=>(gs, 0), -1=>(0, gs), 0=>full range.
            let pair_for = |d: i32| -> Vec<(usize, usize)> {
                if d == 1 { vec![(gs, 0)] }
                else if d == -1 { vec![(0, gs)] }
                else { (0..=gs).map(|i| (i, i)).collect() }
            };
            let xs = pair_for(dx);
            let ys = pair_for(dy);
            let zs = pair_for(dz);
            for &(az, bz) in &zs {
                for &(ay, by) in &ys {
                    for &(ax, bx) in &xs {
                        let sa = self.density_fields[&chunk].get(ax, ay, az);
                        let sb = self.density_fields[&neighbor].get(bx, by, bz);
                        let (d, m) = average_boundary_voxel(sa, sb);
                        if sa.density != d || sa.material != m {
                            updates.push((chunk, ax, ay, az, d, m));
                            dirty.insert(chunk);
                        }
                        if sb.density != d || sb.material != m {
                            updates.push((neighbor, bx, by, bz, d, m));
                            dirty.insert(neighbor);
                        }
                    }
                }
            }
        }
        for (key, x, y, z, d, m) in updates {
            if let Some(field) = self.density_fields.get_mut(&key) {
                let s = field.get_mut(x, y, z);
                s.density = d;
                s.material = m;
            }
        }
        for &k in &dirty {
            if let Some(df) = self.density_fields.get_mut(&k) {
                df.compute_metadata();
            }
        }
        dirty.into_iter().collect()
    }

    /// Sync boundary densities between newly generated region chunks and their
    /// already-loaded cross-region face neighbors. Only marks chunks dirty when
    /// voxel values actually change. Returns ALL dirty keys (both region and
    /// non-region) — caller is responsible for hermite re-extraction and
    /// filtering to non-region keys for remeshing.
    pub fn sync_cross_region_densities(
        &mut self,
        region_coords: &[(i32, i32, i32)],
        chunk_size: usize,
    ) -> Vec<(i32, i32, i32)> {
        let gs = chunk_size;
        let region_set: HashSet<_> = region_coords.iter().copied().collect();
        let mut updates: Vec<((i32, i32, i32), usize, usize, usize, f32, Material)> = Vec::new();
        let mut dirty: HashSet<(i32, i32, i32)> = HashSet::new();

        for &(cx, cy, cz) in region_coords {
            let face_neighbors: [(i32, i32, i32); 6] = [
                (cx + 1, cy, cz), (cx - 1, cy, cz),
                (cx, cy + 1, cz), (cx, cy - 1, cz),
                (cx, cy, cz + 1), (cx, cy, cz - 1),
            ];
            for &neighbor in &face_neighbors {
                if region_set.contains(&neighbor) { continue; }
                if !self.density_fields.contains_key(&neighbor) { continue; }

                // Determine shared face axis and boundary coordinates
                let (axis, a_coord, b_coord) = if neighbor.0 != cx {
                    (0, if neighbor.0 > cx { gs } else { 0 }, if neighbor.0 > cx { 0 } else { gs })
                } else if neighbor.1 != cy {
                    (1, if neighbor.1 > cy { gs } else { 0 }, if neighbor.1 > cy { 0 } else { gs })
                } else {
                    (2, if neighbor.2 > cz { gs } else { 0 }, if neighbor.2 > cz { 0 } else { gs })
                };

                let mut face_a_changed = false;
                let mut face_b_changed = false;

                for u in 0..=gs {
                    for v in 0..=gs {
                        let (ax, ay, az) = match axis {
                            0 => (a_coord, u, v),
                            1 => (u, a_coord, v),
                            _ => (u, v, a_coord),
                        };
                        let (bx, by, bz) = match axis {
                            0 => (b_coord, u, v),
                            1 => (u, b_coord, v),
                            _ => (u, v, b_coord),
                        };

                        let sample_a = self.density_fields[&(cx, cy, cz)].get(ax, ay, az);
                        let sample_b = self.density_fields[&neighbor].get(bx, by, bz);
                        let (d, m) = average_boundary_voxel(sample_a, sample_b);
                        if sample_a.density != d || sample_a.material != m {
                            updates.push(((cx, cy, cz), ax, ay, az, d, m));
                            face_a_changed = true;
                        }
                        if sample_b.density != d || sample_b.material != m {
                            updates.push((neighbor, bx, by, bz, d, m));
                            face_b_changed = true;
                        }
                    }
                }
                if face_a_changed { dirty.insert((cx, cy, cz)); }
                if face_b_changed { dirty.insert(neighbor); }
            }
        }

        if updates.is_empty() {
            return Vec::new();
        }

        // Apply all density updates
        for (key, x, y, z, d, m) in updates {
            if let Some(field) = self.density_fields.get_mut(&key) {
                let sample = field.get_mut(x, y, z);
                sample.density = d;
                sample.material = m;
            }
        }

        // Recompute metadata for dirty chunks (requires &mut, is cheap)
        for &key in &dirty {
            if let Some(density) = self.density_fields.get_mut(&key) {
                density.compute_metadata();
            }
        }

        dirty.into_iter().collect()
    }

    // ── Save/Load ──────────────────────────────────────────────────────

    /// Collect all world modification data needed for saving.
    ///
    /// Gathers snapshots from THREE tiers (later ones override earlier ones):
    ///
    /// 1. **Carry-forward from `pending_snapshots`** — the previous save file's
    ///    chunks that haven't been visited yet this session. CRITICAL: without
    ///    this, the second save after loading a world drops every snapshot the
    ///    player didn't walk to. The chunk has no preserved entry (never
    ///    unloaded this session) and no dirty entry (never edited this
    ///    session), so it would silently disappear from the new save and
    ///    re-generate as fresh worldgen on next load — appearing as "the world
    ///    ate the chunks past where I explored". Bug spotted by user
    ///    2026-05-09: cave systems sealed off with uniform host-rock past the
    ///    chunks they personally re-visited.
    ///
    /// 2. **`preserved_snapshots`** — chunks the player modified earlier this
    ///    session and then walked away from (snapshot captured on unload).
    ///    Always at least as fresh as #1 for the same key.
    ///
    /// 3. **Currently loaded dirty chunks** (in `density_fields` +
    ///    `modification_tracker`) — captured fresh from live density. Always
    ///    at least as fresh as #1 and #2 for the same key.
    ///
    /// `BTreeMap::insert` overwrites in tier order, so freshest wins.
    pub fn collect_save_data(&self) -> WorldSaveData {
        // Tier 1: carry forward unvisited entries from the loaded save file.
        let mut snapshots: BTreeMap<(i32, i32, i32), ChunkSnapshot> = match &self.pending_snapshots
        {
            Some(data) => data.chunk_snapshots.clone(),
            None => BTreeMap::new(),
        };

        // Tier 2: chunks unloaded this session.
        for (k, snap) in &self.preserved_snapshots {
            snapshots.insert(*k, snap.clone());
        }

        // Tier 3: currently loaded dirty chunks (freshest). Capture the
        // painted-stress overlay too so PaintStress brush strokes persist.
        for key in &self.modification_tracker.dirty_chunks {
            if let Some(df) = self.density_fields.get(key) {
                let sf = self.stress_fields.get(key);
                snapshots.insert(*key, ChunkSnapshot::from_chunk(df, sf));
            }
        }

        let terraced_cells: Vec<(i32, i32, i32)> = {
            let mut v: Vec<_> = self.terraced_cells.iter().copied().collect();
            v.sort();
            v
        };

        let terraced_columns: BTreeMap<(i32, i32), i32> =
            self.terraced_columns.iter().map(|(&k, &v)| (k, v)).collect();

        WorldSaveData {
            chunk_snapshots: snapshots,
            terraced_cells,
            terraced_columns,
            triggers: self.triggers.clone(),
            next_trigger_id: self.next_trigger_id,
            // Engine::export_save_data fills this in — store doesn't own anchors.
            crystal_anchors_json: String::new(),
        }
    }

    /// Load saved world data for application during chunk generation.
    ///
    /// Stores the data internally; the worker thread checks `pending_snapshots`
    /// after generating each chunk and patches the density field if a snapshot exists.
    /// Also restores terrace data immediately.
    pub fn load_save_data(&mut self, data: WorldSaveData) {
        // Restore terrace data
        self.terraced_cells = data.terraced_cells.iter().copied().collect();
        self.terraced_columns = data.terraced_columns.iter().map(|(&k, &v)| (k, v)).collect();

        // Restore editor collapse triggers. Loaded eagerly (not deferred like
        // chunk snapshots) because their evaluation needs to be live from the
        // first mining event of the session.
        self.triggers = data.triggers.clone();
        if data.next_trigger_id > self.next_trigger_id {
            self.next_trigger_id = data.next_trigger_id;
        }

        // Store chunk snapshots for on-demand application during generation
        self.pending_snapshots = Some(data);
    }

    /// Check if a snapshot exists for a chunk and apply it if so.
    /// Two sources, preserved_snapshots wins because it's always at least as fresh:
    ///   1. preserved_snapshots — chunks the player modified earlier this session
    ///      then walked away from. Streaming re-loads them here.
    ///   2. pending_snapshots — save-file deltas loaded via Ctrl+L.
    /// Returns true if a snapshot was applied (density field was patched).
    ///
    /// Without checking preserved_snapshots, walking away and back to a modified
    /// chunk silently buries the player's hand-authored work under fresh worldgen.
    pub fn apply_pending_snapshot(&mut self, key: (i32, i32, i32)) -> bool {
        // Prefer the in-session preserved snapshot — it's the most recent state.
        // Clone, not remove, so a failed apply (e.g. wrong size, density not yet
        // inserted) doesn't lose the snapshot — we only drop it on success.
        let snap = self
            .preserved_snapshots
            .get(&key)
            .cloned()
            .or_else(|| {
                self.pending_snapshots
                    .as_ref()
                    .and_then(|d| d.chunk_snapshots.get(&key).cloned())
            });
        if let Some(snap) = snap {
            if let Some(df) = self.density_fields.get_mut(&key) {
                if snap.apply_to(df) {
                    // Also restore the painted-stress overlay, if any. The
                    // stress_field is created lazily in `insert()` so it
                    // exists by the time we get here.
                    if let Some(sf) = self.stress_fields.get_mut(&key) {
                        snap.apply_painted_stress_to(sf);
                    }
                    // Successfully restored — drop the preserved entry so memory
                    // doesn't grow unbounded across re-stream cycles. Save-file
                    // pending_snapshots stays put: a future re-stream might still
                    // legitimately need to re-apply (e.g. region regen from disk).
                    self.preserved_snapshots.remove(&key);
                    self.modification_tracker.mark_dirty(key);
                    return true;
                }
            }
        }
        false
    }

    /// Returns true if there are any world modifications to save.
    pub fn has_modifications(&self) -> bool {
        self.modification_tracker.has_modifications()
            || !self.preserved_snapshots.is_empty()
            || !self.terraced_cells.is_empty()
    }

}

/// Extract a solid mask bitfield from a density field.
///
/// One bit per voxel for the inner chunk_size^3 grid (not the +1 border).
/// Solid = density > 0.0 or material.is_solid().
pub fn extract_solid_mask(density: &DensityField, chunk_size: usize) -> Vec<u64> {
    let total = chunk_size * chunk_size * chunk_size;
    let num_words = (total + 63) / 64;
    let mut mask = vec![0u64; num_words];

    for z in 0..chunk_size {
        for y in 0..chunk_size {
            for x in 0..chunk_size {
                let sample = density.get(x, y, z);
                if sample.material.is_solid() {
                    let idx = z * chunk_size * chunk_size + y * chunk_size + x;
                    let word = idx / 64;
                    let bit = idx % 64;
                    mask[word] |= 1u64 << bit;
                }
            }
        }
    }

    mask
}

/// Average two voxel samples at the same world position for boundary sync.
/// Density uses min (carved side wins); material preserves solid when possible.
fn average_boundary_voxel(a: &VoxelSample, b: &VoxelSample) -> (f32, Material) {
    let avg_density = a.density.min(b.density);  // carved side wins, no 0.0 degenerate surface
    let material = if a.material.is_solid() && b.material.is_solid() {
        if a.density >= b.density { a.material } else { b.material }
    } else if a.material.is_solid() {
        a.material
    } else if b.material.is_solid() {
        b.material
    } else {
        Material::Air
    };
    // Enforce invariant: Air density must be non-positive
    if !material.is_solid() && avg_density > 0.0 {
        (0.0, material)
    } else {
        (avg_density, material)
    }
}

/// Post-smoothing boundary density sync: average overlap voxels between
/// dirty chunks and their face, edge, and corner neighbors so hermite edges
/// match at seams.
///
/// Returns extra neighbor chunks that need remeshing but weren't already dirty.
pub(crate) fn sync_boundary_density(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    dirty_chunks: &[((i32, i32, i32), usize, usize, usize, usize, usize, usize)],
    chunk_size: usize,
) -> Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> {
    let cs = chunk_size; // density grid is cs+1 in each dimension
    let dirty_keys: HashSet<(i32, i32, i32)> = dirty_chunks.iter().map(|d| d.0).collect();

    // Faces to check: (axis condition on dirty bounds, neighbor offset, local coord on A, local coord on B)
    // For each face we collect updates as: (chunk_key, x, y, z, density, material)
    let mut updates: Vec<((i32, i32, i32), usize, usize, usize, f32, Material)> = Vec::new();
    let mut extra_neighbors: HashSet<(i32, i32, i32)> = HashSet::new();

    for &(key, min_x, min_y, min_z, max_x, max_y, max_z) in dirty_chunks {
        let (cx, cy, cz) = key;

        // Per-axis dirty range, indexed [X=0, Y=1, Z=2]. Used to clamp face/
        // edge iteration to only the cells the caller actually modified —
        // critical for mining (~5×5 dirty patch in a 31×31 face), modest for
        // wide ops like flatten that pass full chunk bounds.
        let dirty_min: [usize; 3] = [min_x, min_y, min_z];
        let dirty_max: [usize; 3] = [max_x, max_y, max_z];

        let faces: [(bool, (i32, i32, i32), usize, usize); 6] = [
            // (dirty touches this face?, neighbor key, coord in A, coord in B)
            (max_x >= cs, (cx + 1, cy, cz), cs, 0),     // +X
            (min_x == 0, (cx - 1, cy, cz), 0, cs),      // -X
            (max_y >= cs, (cx, cy + 1, cz), cs, 0),      // +Y
            (min_y == 0, (cx, cy - 1, cz), 0, cs),       // -Y
            (max_z >= cs, (cx, cy, cz + 1), cs, 0),      // +Z
            (min_z == 0, (cx, cy, cz - 1), 0, cs),       // -Z
        ];

        for (face_idx, &(touches, neighbor, coord_a, coord_b)) in faces.iter().enumerate() {
            if !touches {
                continue;
            }
            // Skip if neighbor not loaded
            if !density_fields.contains_key(&neighbor) {
                continue;
            }

            let axis = face_idx / 2; // 0=X, 1=Y, 2=Z

            // The two free dimensions for this face axis: u and v.
            // axis=0 (X face) → u=Y, v=Z. axis=1 (Y face) → u=X, v=Z. axis=2 (Z face) → u=X, v=Y.
            let (u_axis, v_axis) = match axis {
                0 => (1usize, 2usize),
                1 => (0usize, 2usize),
                _ => (0usize, 1usize),
            };
            let u_lo = dirty_min[u_axis];
            let u_hi = dirty_max[u_axis].min(cs);
            let v_lo = dirty_min[v_axis];
            let v_hi = dirty_max[v_axis].min(cs);

            // Iterate over the face plane, clamped to the projected dirty bounds.
            for u in u_lo..=u_hi {
                for v in v_lo..=v_hi {
                    let (ax, ay, az) = match axis {
                        0 => (coord_a, u, v),
                        1 => (u, coord_a, v),
                        _ => (u, v, coord_a),
                    };
                    let (bx, by, bz) = match axis {
                        0 => (coord_b, u, v),
                        1 => (u, coord_b, v),
                        _ => (u, v, coord_b),
                    };

                    let sample_a = density_fields[&key].get(ax, ay, az);
                    let sample_b = density_fields[&neighbor].get(bx, by, bz);

                    let (avg_d, avg_m) = average_boundary_voxel(sample_a, sample_b);

                    updates.push((key, ax, ay, az, avg_d, avg_m));
                    updates.push((neighbor, bx, by, bz, avg_d, avg_m));

                    if !dirty_keys.contains(&neighbor) {
                        extra_neighbors.insert(neighbor);
                    }
                }
            }
        }

        // --- Edge sync (12 edges): sync the 1D line of voxels shared with diagonal neighbors ---
        // Each edge is the intersection of two face boundaries.
        // dir_i, dir_j are +1 or -1 for the two boundary axes; the free axis iterates 0..=cs.
        let edge_defs: [(bool, bool, i32, i32, usize, usize, usize, usize, u8, u8); 12] = [
            // (touches_i, touches_j, di, dj, coord_a_i, coord_b_i, coord_a_j, coord_b_j, axis_i, axis_j)
            // X-Y edges (free axis = Z)
            (max_x >= cs, max_y >= cs, 1, 1, cs, 0, cs, 0, 0, 1),
            (max_x >= cs, min_y == 0,  1,-1, cs, 0,  0,cs, 0, 1),
            (min_x == 0,  max_y >= cs,-1, 1,  0,cs, cs, 0, 0, 1),
            (min_x == 0,  min_y == 0, -1,-1,  0,cs,  0,cs, 0, 1),
            // X-Z edges (free axis = Y)
            (max_x >= cs, max_z >= cs, 1, 1, cs, 0, cs, 0, 0, 2),
            (max_x >= cs, min_z == 0,  1,-1, cs, 0,  0,cs, 0, 2),
            (min_x == 0,  max_z >= cs,-1, 1,  0,cs, cs, 0, 0, 2),
            (min_x == 0,  min_z == 0, -1,-1,  0,cs,  0,cs, 0, 2),
            // Y-Z edges (free axis = X)
            (max_y >= cs, max_z >= cs, 1, 1, cs, 0, cs, 0, 1, 2),
            (max_y >= cs, min_z == 0,  1,-1, cs, 0,  0,cs, 1, 2),
            (min_y == 0,  max_z >= cs,-1, 1,  0,cs, cs, 0, 1, 2),
            (min_y == 0,  min_z == 0, -1,-1,  0,cs,  0,cs, 1, 2),
        ];

        for &(touches_i, touches_j, di, dj, ca_i, cb_i, ca_j, cb_j, axis_i, axis_j) in &edge_defs {
            if !touches_i || !touches_j {
                continue;
            }
            let neighbor = match (axis_i, axis_j) {
                (0, 1) => (cx + di, cy + dj, cz),
                (0, 2) => (cx + di, cy, cz + dj),
                _      => (cx, cy + di, cz + dj), // (1, 2)
            };
            if !density_fields.contains_key(&neighbor) {
                continue;
            }
            // Free axis is the one that's neither axis_i nor axis_j: 0+1+2=3
            let free_axis = 3 - axis_i - axis_j;
            // Clamp the 1D edge sweep to the dirty range on the free axis —
            // same rationale as the face clamp above.
            let t_lo = dirty_min[free_axis as usize];
            let t_hi = dirty_max[free_axis as usize].min(cs);
            for t in t_lo..=t_hi {
                let (ax, ay, az) = {
                    let mut c = [0usize; 3];
                    c[axis_i as usize] = ca_i;
                    c[axis_j as usize] = ca_j;
                    c[free_axis as usize] = t;
                    (c[0], c[1], c[2])
                };
                let (bx, by, bz) = {
                    let mut c = [0usize; 3];
                    c[axis_i as usize] = cb_i;
                    c[axis_j as usize] = cb_j;
                    c[free_axis as usize] = t;
                    (c[0], c[1], c[2])
                };

                let sample_a = density_fields[&key].get(ax, ay, az);
                let sample_b = density_fields[&neighbor].get(bx, by, bz);
                let (avg_d, avg_m) = average_boundary_voxel(sample_a, sample_b);

                updates.push((key, ax, ay, az, avg_d, avg_m));
                updates.push((neighbor, bx, by, bz, avg_d, avg_m));

                if !dirty_keys.contains(&neighbor) {
                    extra_neighbors.insert(neighbor);
                }
            }
        }

        // --- Corner sync (8 corners): sync single voxel shared with diagonal corner neighbor ---
        let corner_defs: [(bool, bool, bool, i32, i32, i32, usize, usize, usize, usize, usize, usize); 8] = [
            // (touches_x, touches_y, touches_z, dx, dy, dz, ax, bx, ay, by, az, bz)
            (max_x >= cs, max_y >= cs, max_z >= cs,  1, 1, 1, cs, 0, cs, 0, cs, 0),
            (max_x >= cs, max_y >= cs, min_z == 0,   1, 1,-1, cs, 0, cs, 0,  0,cs),
            (max_x >= cs, min_y == 0,  max_z >= cs,  1,-1, 1, cs, 0,  0,cs, cs, 0),
            (max_x >= cs, min_y == 0,  min_z == 0,   1,-1,-1, cs, 0,  0,cs,  0,cs),
            (min_x == 0,  max_y >= cs, max_z >= cs, -1, 1, 1,  0,cs, cs, 0, cs, 0),
            (min_x == 0,  max_y >= cs, min_z == 0,  -1, 1,-1,  0,cs, cs, 0,  0,cs),
            (min_x == 0,  min_y == 0,  max_z >= cs, -1,-1, 1,  0,cs,  0,cs, cs, 0),
            (min_x == 0,  min_y == 0,  min_z == 0,  -1,-1,-1,  0,cs,  0,cs,  0,cs),
        ];

        for &(tx, ty, tz, dx, dy, dz, ax, bx, ay, by, az, bz) in &corner_defs {
            if !tx || !ty || !tz {
                continue;
            }
            let neighbor = (cx + dx, cy + dy, cz + dz);
            if !density_fields.contains_key(&neighbor) {
                continue;
            }

            let sample_a = density_fields[&key].get(ax, ay, az);
            let sample_b = density_fields[&neighbor].get(bx, by, bz);
            let (avg_d, avg_m) = average_boundary_voxel(sample_a, sample_b);

            updates.push((key, ax, ay, az, avg_d, avg_m));
            updates.push((neighbor, bx, by, bz, avg_d, avg_m));

            if !dirty_keys.contains(&neighbor) {
                extra_neighbors.insert(neighbor);
            }
        }
    }

    // Pass 2: apply all updates
    for (chunk_key, x, y, z, density, material) in updates {
        if let Some(field) = density_fields.get_mut(&chunk_key) {
            let sample = field.get_mut(x, y, z);
            sample.density = density;
            sample.material = material;
        }
    }

    // Build extra dirty entries for neighbors that weren't already dirty.
    let mut extra_dirty: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> =
        Vec::new();
    for neighbor in extra_neighbors {
        // Full chunk bounds (conservative — only boundary face was modified but remesh needs context)
        extra_dirty.push((neighbor, 0, 0, 0, cs, cs, cs));
    }

    extra_dirty
}


#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a solid density field of given grid size (chunk_size+1).
    fn make_solid_field(size: usize) -> DensityField {
        let mut field = DensityField::new(size);
        for s in &mut field.samples {
            s.density = 1.0;
            s.material = Material::Limestone;
        }
        field
    }

    /// Mine asymmetric patterns in two adjacent chunks and smooth independently,
    /// then verify sync_boundary_density makes the overlap match.
    #[test]
    fn test_boundary_density_sync_after_mine() {
        let chunk_size = 4usize;
        let size = chunk_size + 1; // grid size = 5

        // Two adjacent chunks along X: A=(0,0,0), B=(1,0,0)
        let mut fields: HashMap<(i32, i32, i32), DensityField> = HashMap::new();
        fields.insert((0, 0, 0), make_solid_field(size));
        fields.insert((1, 0, 0), make_solid_field(size));

        // Asymmetric mining: carve a wide tunnel in A but narrow in B.
        // This creates different neighbor patterns so smoothing diverges at overlap.
        // Chunk A: carve x=2..4, y=0..4, z=1..3  (wide, reaching overlap at x=4)
        for z in 1..=3 {
            for y in 0..size {
                for x in 2..size {
                    let s = fields.get_mut(&(0, 0, 0)).unwrap().get_mut(x, y, z);
                    s.density = -1.0;
                    s.material = Material::Air;
                }
            }
        }
        // Chunk B: carve only x=0, y=2..2, z=2..2  (narrow, overlap at x=0)
        {
            let s = fields.get_mut(&(1, 0, 0)).unwrap().get_mut(0, 2, 2);
            s.density = -1.0;
            s.material = Material::Air;
        }

        // Smooth each chunk independently (simulating post-mine smoothing)
        crate::mining::smooth_mine_boundary(
            fields.get_mut(&(0, 0, 0)).unwrap(),
            1, 0, 0, chunk_size, chunk_size, chunk_size,
            3, 0.5,
        );
        crate::mining::smooth_mine_boundary(
            fields.get_mut(&(1, 0, 0)).unwrap(),
            0, 0, 0, 1, chunk_size, chunk_size,
            3, 0.5,
        );

        // Before sync: overlap voxels should differ due to asymmetric carving
        let mut any_differ = false;
        for y in 0..size {
            for z in 0..size {
                let a = fields[&(0, 0, 0)].get(chunk_size, y, z).density;
                let b = fields[&(1, 0, 0)].get(0, y, z).density;
                if (a - b).abs() > 1e-6 {
                    any_differ = true;
                }
            }
        }
        assert!(any_differ, "smoothing should have desynchronized at least some overlap voxels");

        // Run boundary sync
        let dirty_chunks = vec![
            ((0, 0, 0), 1usize, 0usize, 0usize, chunk_size, chunk_size, chunk_size),
            ((1, 0, 0), 0usize, 0usize, 0usize, 1usize, chunk_size, chunk_size),
        ];
        let extra = sync_boundary_density(&mut fields, &dirty_chunks, chunk_size);

        // Both chunks were already dirty, so no extra neighbors expected
        assert!(extra.is_empty(), "both chunks already dirty, no extras expected");

        // After sync: overlap voxels must match exactly
        for y in 0..size {
            for z in 0..size {
                let a = fields[&(0, 0, 0)].get(chunk_size, y, z);
                let b = fields[&(1, 0, 0)].get(0, y, z);
                assert!(
                    (a.density - b.density).abs() < 1e-6,
                    "density mismatch at overlap y={y} z={z}: A={} B={}",
                    a.density, b.density
                );
                assert_eq!(
                    a.material, b.material,
                    "material mismatch at overlap y={y} z={z}"
                );
            }
        }
    }

    /// Mine one chunk with dirty_expand reaching the boundary; verify
    /// the neighbor gets added to extra_dirty and overlaps match after sync.
    #[test]
    fn test_boundary_sync_single_chunk_dirty_expand() {
        let chunk_size = 4usize;
        let size = chunk_size + 1;

        let mut fields: HashMap<(i32, i32, i32), DensityField> = HashMap::new();
        fields.insert((0, 0, 0), make_solid_field(size));
        fields.insert((1, 0, 0), make_solid_field(size));

        // Mine near the +X face of chunk A only (x=3,4)
        for y in 1..=3 {
            for z in 1..=3 {
                for x in 3..=4 {
                    let s = fields.get_mut(&(0, 0, 0)).unwrap().get_mut(x, y, z);
                    s.density = -1.0;
                    s.material = Material::Air;
                }
            }
        }

        // Smooth only chunk A
        crate::mining::smooth_mine_boundary(
            fields.get_mut(&(0, 0, 0)).unwrap(),
            2, 0, 0, 4, 4, 4,
            2, 0.5,
        );

        // Only chunk A is dirty, with max_x reaching chunk_size (the overlap face)
        let dirty_chunks = vec![
            ((0, 0, 0), 2usize, 0usize, 0usize, chunk_size, chunk_size, chunk_size),
        ];
        let extra = sync_boundary_density(&mut fields, &dirty_chunks, chunk_size);

        // Neighbor (1,0,0) should be added as extra dirty
        assert_eq!(extra.len(), 1, "neighbor should be added as extra dirty");
        assert_eq!(extra[0].0, (1, 0, 0));

        // After sync: overlap voxels must match
        for y in 0..size {
            for z in 0..size {
                let a = fields[&(0, 0, 0)].get(chunk_size, y, z);
                let b = fields[&(1, 0, 0)].get(0, y, z);
                assert!(
                    (a.density - b.density).abs() < 1e-6,
                    "density mismatch at overlap y={y} z={z}: A={} B={}",
                    a.density, b.density
                );
                assert_eq!(
                    a.material, b.material,
                    "material mismatch at overlap y={y} z={z}"
                );
            }
        }
    }

    #[test]
    fn test_boundary_voxel_solid_plus_air_preserves_solid() {
        let solid = VoxelSample { density: 0.8, material: Material::Granite };
        let air = VoxelSample { density: -0.5, material: Material::Air };

        // Solid + Air → preserves solid material
        let (d, m) = average_boundary_voxel(&solid, &air);
        assert_eq!(m, Material::Granite, "solid+air should preserve solid material");
        assert!((d - (-0.5)).abs() < 1e-6, "density should be min of the two");

        // Air + Solid → preserves solid material
        let (d2, m2) = average_boundary_voxel(&air, &solid);
        assert_eq!(m2, Material::Granite, "air+solid should preserve solid material");
        assert!((d2 - (-0.5)).abs() < 1e-6, "density should be min of the two");
    }

    #[test]
    fn test_boundary_voxel_solid_solid_picks_higher_density() {
        let a = VoxelSample { density: 0.9, material: Material::Granite };
        let b = VoxelSample { density: 0.5, material: Material::Iron };

        let (_, m) = average_boundary_voxel(&a, &b);
        assert_eq!(m, Material::Granite, "should pick material with higher density");

        // Swap: b has higher density
        let (_, m2) = average_boundary_voxel(&b, &a);
        assert_eq!(m2, Material::Granite, "should pick material with higher density (a)");
    }

    #[test]
    fn test_boundary_voxel_air_plus_air_stays_air() {
        let a = VoxelSample { density: -1.0, material: Material::Air };
        let b = VoxelSample { density: -0.3, material: Material::Air };

        let (_, m) = average_boundary_voxel(&a, &b);
        assert_eq!(m, Material::Air, "air+air should remain air");
    }

    /// Place one ore voxel at a known position with an air neighbor (surface).
    fn place_surface_ore(
        field: &mut DensityField,
        ore: Material,
        x: usize,
        y: usize,
        z: usize,
        air_side: (i32, i32, i32),
    ) {
        let s = field.get_mut(x, y, z);
        s.density = 1.0;
        s.material = ore;
        let nx = (x as i32 + air_side.0) as usize;
        let ny = (y as i32 + air_side.1) as usize;
        let nz = (z as i32 + air_side.2) as usize;
        let n = field.get_mut(nx, ny, nz);
        n.density = -1.0;
        n.material = Material::Air;
    }

    #[test]
    fn test_find_ore_voxels_filter_surface_sort_truncate() {
        let chunk_size = 4usize;
        let size = chunk_size + 1; // grid = 5
        let eb = chunk_size as f32; // voxel_scale = 1.0

        let mut store = ChunkStore::new(8);

        // Chunk A at (0,0,0):
        //   - Tin at (1,1,1) with air neighbor at (1,2,1)  -> SURFACE, world ~ (1.5,1.5,1.5)
        //   - Iron at (3,2,2) with air neighbor at (2,2,2) -> SURFACE, world ~ (3.5,2.5,2.5)
        //   - Tin BURIED at (2,1,1) — neighbors are all solid     -> SHOULD BE EXCLUDED
        let mut a = make_solid_field(size);
        place_surface_ore(&mut a, Material::Tin, 1, 1, 1, (0, 1, 0));
        place_surface_ore(&mut a, Material::Iron, 3, 2, 2, (-1, 0, 0));
        a.get_mut(2, 1, 1).material = Material::Tin; // buried, no air neighbor
        a.compute_metadata();
        assert!(a.has_ore_material, "chunk metadata must flag ores");
        store.density_fields.insert((0, 0, 0), a);

        // Chunk B at (1,0,0): one far Tin voxel.
        // World pos for (x=2,y=2,z=2) here = (1*4 + 2.5, 2.5, 2.5) = (6.5, 2.5, 2.5).
        let mut b = make_solid_field(size);
        place_surface_ore(&mut b, Material::Tin, 2, 2, 2, (0, -1, 0));
        b.compute_metadata();
        store.density_fields.insert((1, 0, 0), b);

        // Chunk C at (0,1,0): all-solid Limestone, NO ore. Must be skipped by metadata gate.
        let mut c = make_solid_field(size);
        c.compute_metadata();
        assert!(!c.has_ore_material);
        store.density_fields.insert((0, 1, 0), c);

        // Player near chunk A origin.
        let player = Vec3::new(0.5, 1.0, 1.0);

        // ── Test 1: filter on Tin specifically — should return 2 Tin voxels, near first.
        let tin_only = store.find_ore_voxels(
            player,
            100.0,
            Material::Tin as u8,
            16,
            chunk_size,
            eb,
        );
        assert_eq!(tin_only.len(), 2, "should find both Tin surface voxels and skip the buried one");
        for (_, m) in &tin_only {
            assert_eq!(*m, Material::Tin as u8);
        }
        // First result should be the closer Tin (chunk A).
        let d0 = (tin_only[0].0 - player).length();
        let d1 = (tin_only[1].0 - player).length();
        assert!(d0 <= d1, "results must be sorted by distance");
        assert!(d0 < 2.0, "near tin should be within ~2 units of player, got {d0}");

        // ── Test 2: any-ore filter (0xFF) — should return Tin x2 + Iron x1.
        let any_ore = store.find_ore_voxels(
            player,
            100.0,
            0xFF,
            16,
            chunk_size,
            eb,
        );
        assert_eq!(any_ore.len(), 3, "any-ore should return all 3 surface ore voxels");
        let mut materials: Vec<u8> = any_ore.iter().map(|(_, m)| *m).collect();
        materials.sort();
        assert_eq!(materials, vec![Material::Iron as u8, Material::Tin as u8, Material::Tin as u8]);

        // ── Test 3: max_results truncation.
        let capped = store.find_ore_voxels(player, 100.0, 0xFF, 1, chunk_size, eb);
        assert_eq!(capped.len(), 1);

        // ── Test 4: radius excludes far voxels.
        // The chunk-B Tin is at world (6.5,2.5,2.5), distance from player ~6.3.
        let near_only = store.find_ore_voxels(player, 3.0, Material::Tin as u8, 16, chunk_size, eb);
        assert_eq!(near_only.len(), 1, "radius=3 should exclude far Tin in chunk B");

        // ── Test 5: zero max_results returns empty quickly.
        let empty = store.find_ore_voxels(player, 100.0, 0xFF, 0, chunk_size, eb);
        assert!(empty.is_empty());
    }
}
