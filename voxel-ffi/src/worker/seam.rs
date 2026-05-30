//! Seam-pass + mesh-dedup helpers shared across worker request handlers.
//!
//! Pure code-movement out of the former monolithic `worker.rs`. Behavior is
//! unchanged; the only edits are visibility widening to `pub(crate)` so the
//! sibling handler modules can call these.

use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use std::collections::HashSet;

use crossbeam_channel::Sender;
use rayon::prelude::*;
use voxel_gen::config::GenerationConfig;
use voxel_gen::region_gen;
use voxel_fluid::FluidEvent;

use crate::convert::convert_mesh_to_ue_scaled;
use crate::store::ChunkStore;
use crate::types::{FfiCrystalPlacement, WorkerResult};

/// Split one collapse slab into 1-5 visual sub-slabs by spatial grid.
///
/// The pile is still computed from the WHOLE slab (one heap), but for the
/// FALLING animation we want big collapses to read as multiple chunks
/// breaking off the ceiling rather than one monolithic block. Each sub-slab
/// gets its own SlabFall emission, so UE spawns N falling-slab actors per
/// event with independent tumble axes, materials, and impact zones.
///
/// Grid choice mirrors `voxel-core::collapse_pile::fragment_slab` so the
/// visual fragments roughly correspond to the pile's peak distribution.
pub(crate) fn split_slab_for_visual(
    slab: &voxel_core::stress::CollapseSlab,
) -> Vec<voxel_core::stress::CollapseSlab> {
    use voxel_core::material::Material;
    use voxel_core::stress::{CollapseSlab, CollapsedVoxel};

    let volume = slab.voxels.len();
    if volume < 24 {
        // Tiny slabs read fine as one piece — splitting makes them too small.
        return vec![slab.clone()];
    }

    let dx = (slab.bb_max.0 - slab.bb_min.0 + 1).max(1);
    let dz = (slab.bb_max.2 - slab.bb_min.2 + 1).max(1);

    let (nx, nz) = if volume < 60 {
        if dx >= dz { (2, 1) } else { (1, 2) }
    } else if volume < 200 {
        if dx >= 2 * dz { (3, 1) }
        else if dz >= 2 * dx { (1, 3) }
        else { (2, 2) }
    } else {
        if dx >= dz { (3, 2) } else { (2, 3) }
    };

    if nx == 1 && nz == 1 {
        return vec![slab.clone()];
    }

    let cell_dx = dx as f32 / nx as f32;
    let cell_dz = dz as f32 / nz as f32;
    let mut buckets: Vec<Vec<CollapsedVoxel>> = vec![Vec::new(); nx * nz];
    for v in &slab.voxels {
        let fx = (((v.world_x - slab.bb_min.0) as f32 / cell_dx).floor() as i32)
            .clamp(0, nx as i32 - 1) as usize;
        let fz = (((v.world_z - slab.bb_min.2) as f32 / cell_dz).floor() as i32)
            .clamp(0, nz as i32 - 1) as usize;
        buckets[fz * nx + fx].push(v.clone());
    }

    let mut out: Vec<CollapseSlab> = Vec::new();
    for voxels in buckets.into_iter().filter(|b| !b.is_empty()) {
        // Drop fragments that ended up too small (< 3 voxels) — they'd
        // produce tiny meshes that read as visual noise.
        if voxels.len() < 3 {
            continue;
        }
        let mut min_x = i32::MAX; let mut max_x = i32::MIN;
        let mut min_y = i32::MAX; let mut max_y = i32::MIN;
        let mut min_z = i32::MAX; let mut max_z = i32::MIN;
        let mut sum_x = 0.0f32; let mut sum_y = 0.0f32; let mut sum_z = 0.0f32;
        let mut mat_counts: std::collections::HashMap<Material, u32> = std::collections::HashMap::new();
        for v in &voxels {
            min_x = min_x.min(v.world_x); max_x = max_x.max(v.world_x);
            min_y = min_y.min(v.world_y); max_y = max_y.max(v.world_y);
            min_z = min_z.min(v.world_z); max_z = max_z.max(v.world_z);
            sum_x += v.world_x as f32;
            sum_y += v.world_y as f32;
            sum_z += v.world_z as f32;
            *mat_counts.entry(v.material).or_insert(0) += 1;
        }
        let n = voxels.len() as f32;
        // Filter Air + non-renderable (>41) materials when picking dominant
        // for the sub-slab; fall back to parent's dominant.
        let dom = mat_counts.iter()
            .filter(|(m, _)| (**m as u8) > 0 && (**m as u8) <= 41)
            .max_by_key(|(_, c)| *c)
            .map(|(m, _)| *m)
            .unwrap_or(slab.dominant_material);
        out.push(CollapseSlab {
            voxels,
            bb_min: (min_x, min_y, min_z),
            bb_max: (max_x, max_y, max_z),
            center: (sum_x / n, sum_y / n, sum_z / n),
            landing_y: slab.landing_y,
            fall_distance: slab.fall_distance,
            dominant_material: dom,
        });
    }

    if out.is_empty() {
        // All buckets were too small — fall back to the original whole slab.
        vec![slab.clone()]
    } else {
        out
    }
}

/// Global rate limiter for cinematic-collapse chunk remeshes.
///
/// Multi-region collapses (6 simultaneous events) used to fire all their
/// chunk-remesh + seam-pass batches within a ~1.6 s window after impact,
/// dumping ~24 chunk-mesh updates onto the game thread back-to-back. UE
/// ProcMesh `CreateMeshSection`/`UpdateMeshSection` is roughly 30-80 ms per
/// chunk depending on triangle count, so the cumulative game-thread cost
/// during the burst was ~1-2 s of stutter — exactly what the user reported
/// as a "2 second freeze".
///
/// This atomic enforces a minimum gap between collapse remesh batches:
/// each deferred thread that's about to run `remesh_dirty + seam pass`
/// claims a slot at `now` (or later if claims are stacking up), advances
/// the cursor by `COLLAPSE_REMESH_GAP_MS`, and sleeps until its slot.
///
/// Net effect: 6 events spread across ~6 × 250 ms = 1.5 s minimum, plus
/// natural impact-time spacing → ~3 s total. Game thread stays responsive
/// because each remesh batch arrives in its own ~250 ms window with time
/// to drain.
const COLLAPSE_REMESH_GAP_MS: u64 = 250;
static NEXT_COLLAPSE_REMESH_MS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Reserve a remesh slot in the global rate-limiter and sleep until it's
/// our turn. Returns the number of ms we actually waited (for logging).
pub(crate) fn throttle_collapse_remesh() -> u64 {
    use std::sync::atomic::Ordering;
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    // Compare-exchange loop: claim the next slot at max(current, now) and
    // advance the cursor by COLLAPSE_REMESH_GAP_MS.
    loop {
        let current = NEXT_COLLAPSE_REMESH_MS.load(Ordering::SeqCst);
        let target = current.max(now_ms);
        let new_next = target + COLLAPSE_REMESH_GAP_MS;
        if NEXT_COLLAPSE_REMESH_MS
            .compare_exchange(current, new_next, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            let wait_ms = target.saturating_sub(now_ms);
            if wait_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(wait_ms));
            }
            return wait_ms;
        }
        // CAS failed → another thread won; retry.
    }
}

/// Map SpringType → FluidType u8 for debug-colored water rendering.
pub(crate) fn spring_type_to_fluid_u8(st: &voxel_gen::springs::SpringType) -> u8 {
    use voxel_gen::springs::SpringType;
    match st {
        SpringType::SpringLine => 3,  // WaterSpringLine (cyan)
        SpringType::VadoseDrip => 4,  // WaterDrip (purple)
        SpringType::AquiferBreach => 5, // WaterBreach (yellow-green)
        SpringType::RiverSource => 6, // WaterRiver (green)
        SpringType::Artesian => 7,    // WaterArtesian (silver)
    }
}

/// FNV-1a hash over mesh's explicit fields (NOT raw struct bytes — those have
/// undefined padding). Used to hash-compare combined (base + seam) meshes and
/// skip FFI round-trips when content is unchanged. ~150μs for a 2000-vertex chunk.
pub(crate) fn hash_mesh(m: &voxel_core::mesh::Mesh) -> u64 {
    let mut h = 14695981039346656037u64;
    let prime = 1099511628211u64;
    let mut mix = |x: u64| { h ^= x; h = h.wrapping_mul(prime); };
    mix(m.vertices.len() as u64);
    mix(m.triangles.len() as u64);
    for v in &m.vertices {
        mix(v.position.x.to_bits() as u64);
        mix(v.position.y.to_bits() as u64);
        mix(v.position.z.to_bits() as u64);
        mix(v.normal.x.to_bits() as u64);
        mix(v.normal.y.to_bits() as u64);
        mix(v.normal.z.to_bits() as u64);
        mix(v.material as u8 as u64);
    }
    for t in &m.triangles {
        mix(t.indices[0] as u64);
        mix(t.indices[1] as u64);
        mix(t.indices[2] as u64);
    }
    h
}

/// Retrieve existing crystal data from ChunkStore for a chunk, converted to UE coords.
/// Used by remesh/seam/mining paths that don't recompute crystals from density.
pub(crate) fn retrieve_crystal_data(
    store: &Arc<RwLock<ChunkStore>>,
    key: (i32, i32, i32),
    voxel_scale: f32,
    world_scale: f32,
) -> Vec<FfiCrystalPlacement> {
    let s = store.read().unwrap();
    match s.crystal_placements.get(&key) {
        Some(placements) if !placements.is_empty() => {
            crate::convert::convert_crystals_to_ue(placements, voxel_scale, world_scale)
        }
        _ => Vec::new(),
    }
}

pub(crate) fn retrieve_mushroom_data(
    store: &Arc<RwLock<ChunkStore>>,
    key: (i32, i32, i32),
    voxel_scale: f32,
    world_scale: f32,
) -> Vec<crate::types::FfiMushroomInstance> {
    let s = store.read().unwrap();
    match s.mushroom_placements.get(&key) {
        Some(placements) if !placements.is_empty() => {
            crate::convert::convert_mushrooms_to_ue(placements, voxel_scale, world_scale)
        }
        _ => Vec::new(),
    }
}

/// Timing breakdown from the seam pass.
pub(crate) struct SeamPassTimings {
    pub total: Duration,
    pub quad_gen: Duration,
    pub mesh_retrieve: Duration,
    pub convert: Duration,
    pub candidates_tried: u32,
    pub candidates_sent: u32,
}

/// After meshing chunk C, attempt seam generation for C and its full
/// 26-neighborhood (face, edge, and corner neighbors). Any chunk that produces
/// non-empty seam quads gets combined with the cached base mesh and re-sent.
///
/// generate_chunk_seam_quads gracefully handles missing neighbors — it simply
/// skips quads where neighbor data isn't available yet. So calling it repeatedly
/// as neighbors arrive is safe and produces progressively more complete seams.
pub(crate) fn incremental_seam_pass(
    chunk: (i32, i32, i32),
    cfg: &GenerationConfig,
    store: &Arc<RwLock<ChunkStore>>,
    result_tx: &Sender<WorkerResult>,
    world_scale: f32,
) -> SeamPassTimings {
    let pass_start = Instant::now();
    let mut t_quad_gen = Duration::ZERO;
    let mut t_mesh_retrieve = Duration::ZERO;
    let mut t_convert = Duration::ZERO;
    let mut candidates_tried: u32 = 0;
    let mut candidates_sent: u32 = 0;

    let mut candidates = Vec::with_capacity(27);
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                candidates.push((chunk.0 + dx, chunk.1 + dy, chunk.2 + dz));
            }
        }
    }

    // Batch: acquire ONE read lock, generate all seam quads + clone base meshes
    let mut to_send: Vec<((i32, i32, i32), voxel_core::mesh::Mesh)> = Vec::new();
    {
        let t0 = Instant::now();
        let s = store.read().unwrap();
        let lock_wait = t0.elapsed();
        t_mesh_retrieve += lock_wait; // attribute lock wait to mesh_retrieve

        for &target in &candidates {
            if !s.chunk_seam_data.contains_key(&target) {
                continue;
            }

            let tq = Instant::now();
            let seam_mesh = region_gen::generate_chunk_seam_quads(target, &s.chunk_seam_data, cfg.chunk_size);
            t_quad_gen += tq.elapsed();
            candidates_tried += 1;

            if seam_mesh.triangles.is_empty() {
                continue;
            }

            let tm = Instant::now();
            let base = match s.base_meshes.get(&target) {
                Some(m) => m.clone(),
                None => continue,
            };
            let mut mesh = base;
            mesh.append(seam_mesh);
            if cfg.mesh_recalc_normals > 0 { mesh.recalculate_normals(); }
            t_mesh_retrieve += tm.elapsed();

            to_send.push((target, mesh));
        }
    } // read lock released

    // Hash + filter: skip sends whose combined mesh matches last-sent.
    // Without this, every neighbor seam pass resends unchanged meshes on
    // every mine — batched_seam_pass had hash-skip; single-chunk path didn't.
    let hashed: Vec<((i32, i32, i32), voxel_core::mesh::Mesh, u64)> =
        to_send.into_iter().map(|(k, m)| { let h = hash_mesh(&m); (k, m, h) }).collect();

    // Fuse hash-filter + crystal-data fetch into ONE read lock (was 2 acquisitions).
    // Also takes crystal data by-value into the tuple, avoiding a later .cloned()
    // per target in the send loop.
    let mut kept: Vec<((i32, i32, i32), voxel_core::mesh::Mesh, u64, Vec<FfiCrystalPlacement>, Vec<crate::types::FfiMushroomInstance>)> =
        Vec::with_capacity(hashed.len());
    {
        let s = store.read().unwrap();
        for (target, mesh, new_hash) in hashed {
            // Never hash-skip the chunk that owns this seam pass: K's per-chunk
            // pipeline just sent its base mesh to UE, and that base send is the
            // LAST thing UE will see for K unless K's own incremental re-sends
            // combined. A concurrent worker's incremental (firing for target=K
            // while K is in its 27-neighborhood) can race ahead, send K's
            // combined first, and record h_combined into last_sent[K] BEFORE
            // K's own incremental reads it. K's own pass then hash-matches and
            // skips — and UE's channel order ends up being [W2 combined, W1
            // base], leaving UE on the seam-less base. Always sending K's own
            // combined as the final word guarantees UE ends on combined.
            if target != chunk {
                if let Some(&prev) = s.last_sent_mesh_hash.get(&target) {
                    if prev == new_hash { continue; }
                }
            }
            let crystal_data = match s.crystal_placements.get(&target) {
                Some(p) if !p.is_empty() => {
                    crate::convert::convert_crystals_to_ue(p, cfg.voxel_scale(), world_scale)
                }
                _ => Vec::new(),
            };
            let mushroom_data = match s.mushroom_placements.get(&target) {
                Some(p) if !p.is_empty() => {
                    crate::convert::convert_mushrooms_to_ue(p, cfg.voxel_scale(), world_scale)
                }
                _ => Vec::new(),
            };
            kept.push((target, mesh, new_hash, crystal_data, mushroom_data));
        }
    }

    // Convert and send outside the lock (non-blocking sends)
    let mut to_record: Vec<((i32, i32, i32), u64)> = Vec::with_capacity(kept.len());
    for (target, combined, new_hash, crystal_data, mushroom_data) in kept {
        let t2 = Instant::now();
        let mut converted = convert_mesh_to_ue_scaled(&combined, cfg.voxel_scale(), world_scale);
        crate::convert::bucket_mesh_by_material(&mut converted);
        t_convert += t2.elapsed();

        if converted.indices.is_empty() {
            continue;  // Don't overwrite base mesh with empty seam update
        }

        let _ = result_tx.send(WorkerResult::ChunkMesh {
            chunk: target,
            mesh: converted,
            generation: 0,
            crystal_data,
            mushroom_data,
            zone_descriptors: Vec::new(),
        });
        to_record.push((target, new_hash));
        candidates_sent += 1;
    }
    if !to_record.is_empty() {
        let mut s = store.write().unwrap();
        for (k, h) in to_record { s.last_sent_mesh_hash.insert(k, h); }
    }

    SeamPassTimings {
        total: pass_start.elapsed(),
        quad_gen: t_quad_gen,
        mesh_retrieve: t_mesh_retrieve,
        convert: t_convert,
        candidates_tried,
        candidates_sent,
    }
}

/// Deduplicated seam pass for multiple dirty chunks.
/// Computes the union of all 27-neighborhoods and runs each candidate exactly once,
/// avoiding the N× duplication when overlapping neighborhoods re-generate the same seams.
///
/// Recompute crystal placements for chunks whose density was just modified.
///
/// Crystals are spawned by `voxel_gen::compute_crystals` based on current
/// material+density state. When mining/flatten/collapse changes a chunk's
/// density, any crystals that were on now-air cells become "floating"
/// HISM instances in UE because UE's `ApplyMeshData` clears+reapplies the
/// HISM list from `crystal_placements`, but if `crystal_placements` still
/// holds the old list (computed against the old density), the HISMs come
/// back at the old positions.
///
/// This helper recomputes the list against the current density and writes
/// it back to the store so the next ChunkMesh send carries the fresh data.
/// Mining already has an inline equivalent — this exists for the non-mine
/// paths (single/batch flatten, collapse, levelling).
pub(crate) fn recompute_crystals_for_chunks(
    store: &Arc<RwLock<ChunkStore>>,
    cfg: &GenerationConfig,
    chunks: &[(i32, i32, i32)],
) {
    if chunks.is_empty() { return; }
    let new_placements: Vec<_> = {
        let s = store.read().unwrap();
        chunks.iter().filter_map(|&key| {
            s.density_fields.get(&key).map(|density| {
                let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                (key, voxel_gen::compute_crystals(coord, density, cfg))
            })
        }).collect()
    };
    let mut s = store.write().unwrap();
    for (key, placements) in new_placements {
        s.crystal_placements.insert(key, placements);
    }
}

/// Mushroom destruction hookup. For each chunk whose density changed, drop
/// any mushroom placement whose anchor voxel is no longer solid. This is
/// what makes mushrooms destructible — when the player mines the voxel a
/// mushroom is growing from, the instance disappears on the next remesh.
///
/// Unlike crystals, mushrooms are NOT re-detected against the new surfaces.
/// Once placed at worldgen, they only ever disappear (they're not a
/// "what's currently visible" overlay; they're physical objects that lived
/// at a specific anchor).
pub(crate) fn prune_destroyed_mushrooms_for_chunks(
    store: &Arc<RwLock<ChunkStore>>,
    chunks: &[(i32, i32, i32)],
) {
    if chunks.is_empty() { return; }
    let pruned: Vec<((i32, i32, i32), Vec<voxel_gen::MushroomPlacement>)> = {
        let s = store.read().unwrap();
        chunks.iter().filter_map(|&key| {
            let placements = s.mushroom_placements.get(&key)?;
            if placements.is_empty() {
                return None;
            }
            let density = s.density_fields.get(&key)?;
            let size = density.size;
            let kept: Vec<voxel_gen::MushroomPlacement> = placements.iter()
                .filter(|p| {
                    let lx = p.anchor_lx as usize;
                    let ly = p.anchor_ly as usize;
                    let lz = p.anchor_lz as usize;
                    if lx >= size || ly >= size || lz >= size {
                        return false;
                    }
                    density.get(lx, ly, lz).material.is_solid()
                })
                .cloned()
                .collect();
            if kept.len() == placements.len() {
                None  // Nothing changed — skip the write lock
            } else {
                Some((key, kept))
            }
        }).collect()
    };
    if pruned.is_empty() { return; }
    let mut s = store.write().unwrap();
    for (key, kept) in pruned {
        s.mushroom_placements.insert(key, kept);
    }
}

/// Dirty chunks are guaranteed to have their mesh sent even if they have no seam quads,
/// since callers rely on this function as the sole sender for mine/flatten results.
///
/// Always notifies the fluid thread of density changes for `dirty_keys` via
/// `FluidEvent::TerrainModified` before the seam pass. Mining had been the
/// only path doing this manually — every brush, flatten, slab and undo
/// route now shares the same plumbing here so creative-mode carving lets
/// adjacent lava actually flow into the new air cells.
pub(crate) fn batched_seam_pass(
    dirty_keys: &[(i32, i32, i32)],
    cfg: &GenerationConfig,
    store: &Arc<RwLock<ChunkStore>>,
    result_tx: &Sender<WorkerResult>,
    fluid_event_tx: &Sender<FluidEvent>,
    world_scale: f32,
) {
    batched_seam_pass_inner(dirty_keys, cfg, store, result_tx, fluid_event_tx, world_scale, false);
}

pub(crate) fn batched_seam_pass_mine(
    dirty_keys: &[(i32, i32, i32)],
    cfg: &GenerationConfig,
    store: &Arc<RwLock<ChunkStore>>,
    result_tx: &Sender<WorkerResult>,
    fluid_event_tx: &Sender<FluidEvent>,
    world_scale: f32,
) {
    batched_seam_pass_inner(dirty_keys, cfg, store, result_tx, fluid_event_tx, world_scale, true);
}

pub(crate) fn batched_seam_pass_inner(
    dirty_keys: &[(i32, i32, i32)],
    cfg: &GenerationConfig,
    store: &Arc<RwLock<ChunkStore>>,
    result_tx: &Sender<WorkerResult>,
    fluid_event_tx: &Sender<FluidEvent>,
    world_scale: f32,
    batch_as_mine: bool,
) {
    // Notify the fluid sim that these chunks' densities changed BEFORE
    // running the seam pass. This refreshes the fluid thread's density
    // cache + cell_capacity for each cell, which is what makes
    // newly-carved cells reachable for fluid flow and what triggers
    // squeeze-out for cells that just became solid. Idempotent: callers
    // that already sent TerrainModified explicitly (mining, flatten,
    // sleep) just write the same densities to the cache twice.
    if !dirty_keys.is_empty() {
        let s = store.read().unwrap();
        for &key in dirty_keys {
            if let Some(density) = s.density_fields.get(&key) {
                let densities: Vec<f32> = density.samples.iter().map(|s| s.density).collect();
                let _ = fluid_event_tx.send(FluidEvent::TerrainModified {
                    chunk: key,
                    densities,
                });
            }
        }
    }

    // Mushroom destruction. Every density-mutating path (mining, brushes,
    // flatten, etc.) funnels through this function, so pruning here means
    // we don't need per-call-site hooks. Idempotent — pruning a chunk
    // whose anchors are all still solid is a no-op.
    prune_destroyed_mushrooms_for_chunks(store, dirty_keys);

    let dirty_set: HashSet<(i32, i32, i32)> = dirty_keys.iter().copied().collect();

    let mut candidates: HashSet<(i32, i32, i32)> = HashSet::new();
    for &key in dirty_keys {
        for dx in -1..=1i32 {
            for dy in -1..=1i32 {
                for dz in -1..=1i32 {
                    candidates.insert((key.0 + dx, key.1 + dy, key.2 + dz));
                }
            }
        }
    }

    let mut to_send: Vec<((i32, i32, i32), voxel_core::mesh::Mesh)> = Vec::new();
    let mut sent_keys: HashSet<(i32, i32, i32)> = HashSet::new();
    {
        let s = store.read().unwrap();
        for &target in &candidates {
            if !s.chunk_seam_data.contains_key(&target) {
                continue;
            }
            let seam_mesh = region_gen::generate_chunk_seam_quads(target, &s.chunk_seam_data, cfg.chunk_size);
            let base = match s.base_meshes.get(&target) {
                Some(m) => m.clone(),
                None => continue,
            };
            if seam_mesh.triangles.is_empty() {
                // No seam quads — only send the base mesh if this is a dirty chunk
                // (must still receive its updated mesh after mine/flatten)
                if dirty_set.contains(&target) {
                    let mut mesh = base;
                    if cfg.mesh_recalc_normals > 0 {
                        mesh.recalculate_normals();
                    }
                    to_send.push((target, mesh));
                    sent_keys.insert(target);
                }
                continue;
            }
            let mut mesh = base;
            mesh.append(seam_mesh);
            if cfg.mesh_recalc_normals > 0 {
                mesh.recalculate_normals();
            }
            to_send.push((target, mesh));
            sent_keys.insert(target);
        }

        // Fallback: dirty chunks that have a base mesh but no seam data entry at all
        for &key in dirty_keys {
            if sent_keys.contains(&key) {
                continue;
            }
            if let Some(base) = s.base_meshes.get(&key) {
                let mut mesh = base.clone();
                if cfg.mesh_recalc_normals > 0 {
                    mesh.recalculate_normals();
                }
                to_send.push((key, mesh));
            }
        }
    }

    // Round 7: hash combined mesh content; skip chunks whose hash matches
    // the last sent. Saves Rust-side convert + bucket + FFI round-trip for
    // duplicates. UE's hash-skip catches these downstream; doing it here
    // prevents even doing convert + bucket + FFI send on the Rust side.

    // Hash all meshes FIRST (no lock held — hashing uses only owned data).
    let hashed: Vec<((i32, i32, i32), voxel_core::mesh::Mesh, u64)> =
        to_send.into_iter().map(|(k, m)| { let h = hash_mesh(&m); (k, m, h) }).collect();

    // Fuse hash-filter + crystal-data fetch into ONE read lock (was 2 read locks with
    // a write lock sandwiched between them — 3 acquisitions total). Hashes are now
    // recorded in a single write lock AFTER the read, and crystal data is carried
    // by-value so the send loop doesn't .cloned() it per target.
    // `was_previously_sent` lets the empty-mesh skip distinguish "first-time
    // empty chunk" (drop, no UE actor needed) from "chunk that just became
    // empty after a carve" (must send so UE clears the old mesh + collision —
    // otherwise a fully-carved chunk leaves a ghost actor visible until reload).
    let mut kept: Vec<((i32, i32, i32), voxel_core::mesh::Mesh, u64, Vec<FfiCrystalPlacement>, Vec<crate::types::FfiMushroomInstance>, bool)> =
        Vec::with_capacity(hashed.len());
    {
        let s = store.read().unwrap();
        for (target, mesh, new_hash) in hashed {
            let prev_entry = s.last_sent_mesh_hash.get(&target).copied();
            if let Some(prev) = prev_entry {
                if prev == new_hash {
                    continue;
                }
            }
            let was_previously_sent = prev_entry.is_some();
            let crystal_data = match s.crystal_placements.get(&target) {
                Some(p) if !p.is_empty() => {
                    crate::convert::convert_crystals_to_ue(p, cfg.voxel_scale(), world_scale)
                }
                _ => Vec::new(),
            };
            let mushroom_data = match s.mushroom_placements.get(&target) {
                Some(p) if !p.is_empty() => {
                    crate::convert::convert_mushrooms_to_ue(p, cfg.voxel_scale(), world_scale)
                }
                _ => Vec::new(),
            };
            kept.push((target, mesh, new_hash, crystal_data, mushroom_data, was_previously_sent));
        }
    }
    // Record new hashes (brief write lock)
    if !kept.is_empty() {
        let mut s = store.write().unwrap();
        for (target, _mesh, new_hash, _crystals, _mushrooms, _was_prev) in &kept {
            s.last_sent_mesh_hash.insert(*target, *new_hash);
        }
    }

    if batch_as_mine {
        // Send all mine mesh updates as one atomic result — no pop-in
        let mut batch = Vec::new();
        for (target, combined, _hash, crystal_data, mushroom_data, was_previously_sent) in kept {
            let mut converted = convert_mesh_to_ue_scaled(&combined, cfg.voxel_scale(), world_scale);
            crate::convert::bucket_mesh_by_material(&mut converted);
            // Only drop empties for chunks that were never sent — for chunks
            // UE already has, an empty mesh is a clear command, not a no-op.
            if converted.indices.is_empty() && !was_previously_sent { continue; }
            batch.push((target, converted, crystal_data, mushroom_data));
        }
        if !batch.is_empty() {
            let _ = result_tx.send(WorkerResult::MineBatchMesh { meshes: batch });
        }
    } else {
        for (target, combined, _hash, crystal_data, mushroom_data, was_previously_sent) in kept {
            let mut converted = convert_mesh_to_ue_scaled(&combined, cfg.voxel_scale(), world_scale);
            crate::convert::bucket_mesh_by_material(&mut converted);
            if converted.indices.is_empty() && !was_previously_sent { continue; }
            let _ = result_tx.send(WorkerResult::ChunkMesh {
                chunk: target, mesh: converted, generation: 0, crystal_data, mushroom_data,
                zone_descriptors: Vec::new(),
            });
        }
    }
}
