use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use glam::Vec3;
use rayon::prelude::*;
use voxel_core::dual_contouring::mesh_gen::{compute_cell_normals, generate_mesh};
use voxel_core::dual_contouring::solve::solve_dc_vertices;
use voxel_core::hermite::HermiteData;
use voxel_core::mesh::Mesh;
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

mod boundary;
mod search;
#[cfg(test)]
mod tests;

// Re-export so existing `crate::store::sync_boundary_density` paths resolve
// unchanged after the file->folder split. `ChunkStore`, `CavernLocations`,
// and `extract_solid_mask` are defined in this module; `sync_boundary_density`
// lives in `boundary`. (`average_boundary_voxel` stays pub(crate) in
// `boundary` and is reached directly by the tests via `super::boundary`.)
pub(crate) use boundary::sync_boundary_density;


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
    /// Chunks the sleep-montage cinematic is actively filming. `unload()` refuses
    /// to drop these so the camera planner's voxel queries (rock-vs-air ray
    /// clamp) always have density to read — UE pins the UE-side chunk, but
    /// nothing stopped Rust from evicting the density underneath it, which left
    /// QuerySurface reporting "unloaded" (=solid) and the planner blind. Set via
    /// `voxel_montage_set_protected_chunks`, cleared at montage end. Rust coords.
    pub montage_protected: HashSet<(i32, i32, i32)>,
    /// Per-chunk seam data (DC vertices + boundary edges) for seam stitching.
    /// Arc-wrapped so seam passes can snapshot the entries they need under a
    /// brief read lock and run quad generation WITHOUT holding the store lock
    /// (long read holds were serializing against generation write locks during
    /// the initial-load flood). Entries are immutable once inserted — updates
    /// replace the whole Arc.
    pub chunk_seam_data: HashMap<(i32, i32, i32), Arc<ChunkSeamData>>,
    /// Cached base meshes (pre-seam) for fast seam pass reuse. Arc-wrapped for
    /// the same reason: seam passes grab the Arc under the read lock and do the
    /// deep clone + seam append + normal recalc outside it.
    pub base_meshes: HashMap<(i32, i32, i32), Arc<Mesh>>,
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
    /// Mushroom placements preserved from unloaded chunks. Captured on
    /// `unload()` whenever the chunk had any mushrooms; restored by
    /// `apply_pending_snapshot()` when the chunk streams back in. Lives
    /// alongside `preserved_snapshots` rather than inside it so a chunk can
    /// be mushroom-edited without forcing a density snapshot.
    pub preserved_mushrooms: BTreeMap<(i32, i32, i32), Vec<voxel_gen::MushroomPlacement>>,
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
            montage_protected: HashSet::new(),
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
            preserved_mushrooms: BTreeMap::new(),
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

    pub fn insert(&mut self, key: (i32, i32, i32), density: DensityField, hermite: HermiteData) {
        let size = density.size;
        self.density_fields.insert(key, density);
        self.hermite_data.insert(key, hermite);
        // Initialize stress and support fields if not already present
        self.stress_fields.entry(key).or_insert_with(|| StressField::new(size));
        self.support_fields.entry(key).or_insert_with(|| SupportField::new(size));
    }

    pub fn unload(&mut self, key: (i32, i32, i32)) {
        // Montage guarantee: never evict a chunk the cinematic is filming. The
        // camera planner samples voxel density for these chunks (rock-vs-air ray
        // clamp); dropping it mid-montage made QuerySurface report "unloaded"
        // (=solid) and blinded the planner (the false-open / outside-the-walls
        // shots). UE pins the UE-side actor, but only this guard keeps the Rust
        // density alive. Cleared at montage end via voxel_montage_clear_protected.
        if self.montage_protected.contains(&key) {
            return;
        }
        // Preserve density+painted-stress+support snapshot if this chunk was
        // modified (mining/flatten/sleep/PaintStress brush/strut placement).
        // Capturing all three overlays here means everything round-trips
        // through unload→reload without leaking memory. Struts especially
        // MUST be captured here — pre-2026-05-26 they were silently dropped
        // on chunk eviction, leaving visible UE actors with zero Rust-side
        // stress contribution.
        if self.modification_tracker.dirty_chunks.contains(&key) {
            if let Some(df) = self.density_fields.get(&key) {
                let sf = self.stress_fields.get(&key);
                let supf = self.support_fields.get(&key);
                self.preserved_snapshots.insert(key, ChunkSnapshot::from_chunk(df, sf, supf));
            }
            self.modification_tracker.remove(&key);
        }
        // Also preserve struts even when the chunk isn't dirty-tracked — a
        // chunk that has only had struts placed in it (no density edits) is
        // not in `dirty_chunks` but its supports MUST survive unload. Mirror
        // the mushroom logic below.
        if let Some(supf) = self.support_fields.get(&key) {
            if !supf.is_empty() && !self.preserved_snapshots.contains_key(&key) {
                if let Some(df) = self.density_fields.get(&key) {
                    let sf = self.stress_fields.get(&key);
                    self.preserved_snapshots.insert(key, ChunkSnapshot::from_chunk(df, sf, Some(supf)));
                }
            }
        }
        // Preserve mushrooms whenever the chunk has any — independent of the
        // dirty-tracker because painting mushrooms doesn't mutate density. A
        // chunk can be "clean" w.r.t. density but still have painted mushrooms
        // that MUST survive unload→reload.
        if let Some(placements) = self.mushroom_placements.get(&key) {
            if !placements.is_empty() {
                self.preserved_mushrooms.insert(key, placements.clone());
            }
        }
        self.mushroom_placements.remove(&key);
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
        self.chunk_seam_data.insert(chunk, Arc::new(seam_data));
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

        let outputs: Vec<((i32, i32, i32), HermiteData, Mesh, Vec<Vec3>, Vec<Vec3>, Vec<_>, ConvertedMesh)> =
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
                    let dc_normals = compute_cell_normals(&hermite, cell_size);

                    let mut converted = convert_mesh_to_ue_scaled(&mesh, voxel_scale, world_scale);
                    crate::convert::bucket_mesh_by_material(&mut converted);

                    Some((key, hermite, mesh, dc_vertices, dc_normals, boundary_edges, converted))
                })
                .collect();

        // Phase 2: serial write-back. The HashMaps aren't shared mutably across
        // threads, so apply the per-chunk results here in one pass.
        let mut results = Vec::with_capacity(outputs.len());
        for (key, hermite, mesh, dc_vertices, dc_normals, boundary_edges, converted) in outputs {
            self.hermite_data.insert(key, hermite);
            self.base_meshes.insert(key, Arc::new(mesh));
            self.chunk_seam_data.insert(
                key,
                Arc::new(ChunkSeamData {
                    dc_vertices,
                    dc_normals,
                    world_origin: Vec3::ZERO,
                    boundary_edges,
                }),
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
        // painted-stress overlay + struts + HP too so all overlays persist.
        for key in &self.modification_tracker.dirty_chunks {
            if let Some(df) = self.density_fields.get(key) {
                let sf = self.stress_fields.get(key);
                let supf = self.support_fields.get(key);
                snapshots.insert(*key, ChunkSnapshot::from_chunk(df, sf, supf));
            }
        }
        // Tier 3b: chunks with struts but no density edits — still need
        // the support overlay persisted. The earlier `unload()` path
        // captures these on eviction; the same logic here picks up
        // currently-loaded ones at save time.
        for (key, supf) in &self.support_fields {
            if supf.is_empty() { continue; }
            if snapshots.contains_key(key) { continue; }
            if let Some(df) = self.density_fields.get(key) {
                let sf = self.stress_fields.get(key);
                snapshots.insert(*key, ChunkSnapshot::from_chunk(df, sf, Some(supf)));
            }
        }

        let terraced_cells: Vec<(i32, i32, i32)> = {
            let mut v: Vec<_> = self.terraced_cells.iter().copied().collect();
            v.sort();
            v
        };

        let terraced_columns: BTreeMap<(i32, i32), i32> =
            self.terraced_columns.iter().map(|(&k, &v)| (k, v)).collect();

        // Mushroom gather — same 3-tier carry-forward model as density.
        // Tier 1: previous save file's entries (carry forward for chunks
        // not visited this session).
        // Tier 2: preserved_mushrooms (chunks unloaded this session).
        // Tier 3: currently-loaded chunks that have placements — captured
        // straight from `mushroom_placements`. Painted-and-still-loaded
        // chunks land here. We don't gate on `dirty_chunks` because
        // mushroom edits don't touch the density dirty tracker.
        let mut mushroom_placements: BTreeMap<(i32, i32, i32), Vec<voxel_gen::MushroomPlacement>> =
            match &self.pending_snapshots {
                Some(data) => data.mushroom_placements.clone(),
                None => BTreeMap::new(),
            };
        for (k, m) in &self.preserved_mushrooms {
            mushroom_placements.insert(*k, m.clone());
        }
        // Tier 3: only capture from currently-loaded chunks that the player
        // touched this session. Iterating ALL of mushroom_placements would
        // include every worldgen-only chunk, bloating saves to MB. The
        // mushroom brushes call modification_tracker.mark_dirty(key) on
        // every affected chunk, so this set is the authoritative
        // "player-edited" list.
        for key in &self.modification_tracker.dirty_chunks {
            if let Some(m) = self.mushroom_placements.get(key) {
                // Empty Vec is meaningful here ("had mushrooms, erased them all")
                // — keep it so the worker's regen-skip gate fires on reload.
                mushroom_placements.insert(*key, m.clone());
            }
        }

        WorldSaveData {
            chunk_snapshots: snapshots,
            terraced_cells,
            terraced_columns,
            triggers: self.triggers.clone(),
            next_trigger_id: self.next_trigger_id,
            // Engine::export_save_data fills this in — store doesn't own anchors.
            crystal_anchors_json: String::new(),
            mushroom_placements,
            // Engine::export_save_data fills this in — store doesn't own WorldMemory.
            world_memory_blob: Vec::new(),
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

        // Mushrooms have their own preservation path — independent of the
        // density snapshot because painting mushrooms doesn't dirty density.
        // Restore from session-preserved first, then save-file pending.
        let mushrooms = self
            .preserved_mushrooms
            .get(&key)
            .cloned()
            .or_else(|| {
                self.pending_snapshots
                    .as_ref()
                    .and_then(|d| d.mushroom_placements.get(&key).cloned())
            });

        let mut applied = false;
        if let Some(snap) = snap {
            if let Some(df) = self.density_fields.get_mut(&key) {
                if snap.apply_to(df) {
                    if let Some(sf) = self.stress_fields.get_mut(&key) {
                        snap.apply_painted_stress_to(sf);
                    }
                    // Make sure the support field exists at the chunk size
                    // before applying. SupportField is created lazily on
                    // first place; absent here means "never had struts in
                    // this session", which is fine — we'll insert a fresh
                    // one and apply_supports_to fills it in.
                    let chunk_size = df.size;
                    let supf = self.support_fields.entry(key)
                        .or_insert_with(|| SupportField::new(chunk_size));
                    snap.apply_supports_to(supf);
                    self.preserved_snapshots.remove(&key);
                    self.modification_tracker.mark_dirty(key);
                    applied = true;
                }
            }
        }

        if let Some(placements) = mushrooms {
            // Restore mushrooms even when there's no density snapshot — a
            // chunk can have only mushroom edits. Insert wholesale so the
            // chunk-gen path's `compute_mushrooms` step sees the entry and
            // skips regeneration (see worker.rs gate).
            self.mushroom_placements.insert(key, placements);
            self.preserved_mushrooms.remove(&key);
            applied = true;
        }

        applied
    }

    /// Returns true if there are any world modifications to save.
    pub fn has_modifications(&self) -> bool {
        // Mushroom-only edits ride along: the brush marks chunks dirty in
        // modification_tracker so the dirty-chunks branch covers them.
        // preserved_mushrooms covers the "painted then walked away" case
        // even if density was never edited in those chunks.
        self.modification_tracker.has_modifications()
            || !self.preserved_snapshots.is_empty()
            || !self.preserved_mushrooms.is_empty()
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
