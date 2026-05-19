use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use std::collections::{HashMap, HashSet};

use crossbeam_channel::{Receiver, Sender};
use dashmap::DashMap;
use rayon::prelude::*;
use voxel_core::dual_contouring::mesh_gen::generate_mesh;
use voxel_core::dual_contouring::solve::solve_dc_vertices;
use voxel_core::stress::SupportType;
use voxel_fluid::FluidEvent;
use voxel_gen::config::{GenerationConfig, StressConfig};
use voxel_gen::hermite_extract::extract_hermite_data;
use voxel_gen::region_gen::{
    self, generate_region_densities, region_chunks, region_key, sync_region_boundary_densities,
    ChunkSeamData, RegionTimings,
};

use crate::convert::{convert_mesh_to_ue_scaled, from_ue_normal, from_ue_world_pos};
use crate::engine::terrace_size_for_scale;
use crate::profiler::{ChunkTimings, StreamingProfiler};
use crate::store::ChunkStore;
use crate::types::{FfiCollapseEvent, FfiCrystalPlacement, FfiSlabFallData, FfiZoneDescriptor, WorkerRequest, WorkerResult};

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
fn split_slab_for_visual(
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
fn throttle_collapse_remesh() -> u64 {
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
fn spring_type_to_fluid_u8(st: &voxel_gen::springs::SpringType) -> u8 {
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
fn hash_mesh(m: &voxel_core::mesh::Mesh) -> u64 {
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
fn retrieve_crystal_data(
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

fn retrieve_mushroom_data(
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

/// Worker thread main loop. Each worker pulls from shared channels.
pub fn worker_loop(
    shutdown: Arc<AtomicBool>,
    generate_rx: Receiver<WorkerRequest>,
    mine_rx: Receiver<WorkerRequest>,
    result_tx: Sender<WorkerResult>,
    store: Arc<RwLock<ChunkStore>>,
    config: Arc<RwLock<GenerationConfig>>,
    stress_config: Arc<RwLock<StressConfig>>,
    generation_counters: Arc<DashMap<(i32, i32, i32), AtomicU64>>,
    world_scale: f32,
    fluid_event_tx: Sender<FluidEvent>,
    profiler: Arc<StreamingProfiler>,
    worker_id: usize,
    morph_manifest: Arc<Mutex<Option<voxel_sleep::ChangeManifest>>>,
    regions_in_flight: Arc<DashMap<(i32, i32, i32), Arc<Mutex<()>>>>,
) {
    while !shutdown.load(Ordering::Relaxed) {
        // Priority 1: mine requests (non-blocking)
        if let Ok(req) = mine_rx.try_recv() {
            handle_request(
                req, &result_tx, &store, &config, &stress_config, &generation_counters,
                world_scale, &fluid_event_tx, &profiler, worker_id, &generate_rx, &mine_rx, &morph_manifest,
                &regions_in_flight,
            );
            continue;
        }

        // Priority 1.5: deferred stress recalculation (only worker 0 handles this)
        if worker_id == 0 {
            if try_process_stress_queue(&store, &stress_config, &config, &result_tx, &fluid_event_tx, world_scale) {
                continue;
            }
        }

        // Priority 2: generate requests (blocking with timeout)
        let idle_start = Instant::now();
        match generate_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(req) => {
                profiler.record_worker_idle(worker_id, idle_start.elapsed());
                handle_request(
                    req, &result_tx, &store, &config, &stress_config, &generation_counters,
                    world_scale, &fluid_event_tx, &profiler, worker_id, &generate_rx, &mine_rx, &morph_manifest,
                    &regions_in_flight,
                );
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                profiler.record_worker_idle(worker_id, idle_start.elapsed());
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }
}

/// Check mine queue and handle any pending mine request. Returns true if one was handled.
fn try_handle_mine(
    mine_rx: &Receiver<WorkerRequest>,
    result_tx: &Sender<WorkerResult>,
    store: &Arc<RwLock<ChunkStore>>,
    config: &Arc<RwLock<GenerationConfig>>,
    world_scale: f32,
    fluid_event_tx: &Sender<FluidEvent>,
) -> bool {
    if let Ok(req) = mine_rx.try_recv() {
        match req {
            WorkerRequest::Mine { request } => {
                // Inline mine handling (same as handle_request Mine branch)
                let cfg = config.read().unwrap().clone();
                let center = from_ue_world_pos(
                    request.world_x, request.world_y, request.world_z, world_scale,
                );
                let radius = request.radius / world_scale;
                let mut s = store.write().unwrap();
                let outcome = if request.mode == 0 {
                    crate::mining::mine_sphere(&mut s, center, radius, &cfg, world_scale)
                } else {
                    let normal = from_ue_normal(request.normal_x, request.normal_y, request.normal_z);
                    crate::mining::mine_peel(&mut s, center, normal, radius, &cfg, world_scale)
                };
                let dirty_keys: Vec<(i32, i32, i32)> = outcome.meshes.into_iter().map(|(k, _)| k).collect();
                let mined = outcome.mined;
                // Fix B: crystal recompute only for chunks where a material flip
                // actually occurred. Boundary-sync chunks (density tweaks only) keep
                // their existing crystal placements — no recompute needed.
                for &key in &outcome.flipped_chunks {
                    if let Some(density) = s.density_fields.get(&key) {
                        let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                        let placements = voxel_gen::compute_crystals(coord, density, &cfg);
                        s.crystal_placements.insert(key, placements);
                    }
                }
                // Queue position-based stress recalculation at mine point
                let stress_center = (center.x as i32, center.y as i32, center.z as i32);
                let stress_radius = radius as i32 + 4; // mine radius + air decay(2) + small margin
                s.queue_stress_dirty(stress_center, stress_radius);
                drop(s);
                let _ = result_tx.send(WorkerResult::MinedMaterials { mined });
                batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
                return true;
            }
            _ => {} // non-mine request, ignore
        }
    }
    false
}

/// Deferred stress recalculation: checks if the stress dirty queue timer has elapsed,
/// runs the v2 stress algorithm, emits warnings, and triggers collapse if needed.
/// Returns true if work was done (so worker loop can continue instead of sleeping).
fn try_process_stress_queue(
    store: &Arc<RwLock<ChunkStore>>,
    stress_config: &Arc<RwLock<StressConfig>>,
    config: &Arc<RwLock<GenerationConfig>>,
    result_tx: &Sender<WorkerResult>,
    fluid_event_tx: &Sender<FluidEvent>,
    world_scale: f32,
) -> bool {
    use voxel_core::stress::{
        recalc_stress_region_v2_filtered, detect_and_execute_collapses_v2,
    };
    use crate::types::FfiStressWarning;
    use std::io::Write;
    use std::collections::HashSet;

    let debug_path = "D:/Unreal Projects/Mithril2026/Saved/stress_debug.txt";
    let mut dbg = |msg: String| {
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(debug_path) {
            let _ = writeln!(f, "[{:.2}] {}", std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs_f64() % 10000.0, msg);
        }
    };

    // Check if stress dirty queue is ready (timer elapsed)
    let events = {
        let mut s = store.write().unwrap();
        s.drain_stress_dirty(0.4) // 400ms deferred timer
    };

    let events = match events {
        Some(e) if !e.is_empty() => e,
        _ => return false,
    };

    // Keep a parallel reference for editor-trigger evaluation later — the
    // local `events` gets shadowed by the v2 collapse events vec further
    // down, but trigger eval still needs to know which AABBs were mined.
    let mined_dirty_events: Vec<voxel_core::stress::StressDirtyEvent> = events.clone();

    let cfg = config.read().unwrap().clone();
    let stress_cfg = stress_config.read().unwrap().clone();
    let chunk_size = cfg.chunk_size;

    // Derive affected chunks from events (union of all event bounding boxes)
    let mut dirty_chunks: Vec<(i32, i32, i32)> = {
        let s = store.read().unwrap();
        let mut chunk_set: HashSet<(i32, i32, i32)> = HashSet::new();
        for event in &events {
            for key in event.affected_chunks(chunk_size) {
                if s.density_fields.contains_key(&key) {
                    chunk_set.insert(key);
                }
            }
        }
        chunk_set.into_iter().collect()
    };

    if dirty_chunks.is_empty() { return false; }

    dbg(format!("=== STRESS RECALC START === events={} derived_chunks={} chunk_size={}",
        events.len(), dirty_chunks.len(), chunk_size));
    for (i, e) in events.iter().enumerate() {
        dbg(format!("  event[{}]: center=({},{},{}) radius={}", i, e.center.0, e.center.1, e.center.2, e.radius));
    }
    dbg(format!("  config: span_w={:.3} min_safe_span={} min_collapse={} slab_cohesion={:.2} max_vol={} depth_scale={:.0}",
        stress_cfg.span_weight, stress_cfg.min_safe_span,
        stress_cfg.min_collapse_region, stress_cfg.slab_cohesion_threshold, stress_cfg.max_collapse_volume,
        stress_cfg.depth_pressure_scale));

    let recalc_start = std::time::Instant::now();

    // Run v2 stress recalculation — only voxels within event radii are recalculated
    let mut result = {
        let mut s = store.write().unwrap();
        let (density, stress, support) = s.sleep_fields_mut();
        recalc_stress_region_v2_filtered(
            density,
            stress,
            support,
            &stress_cfg,
            &dirty_chunks,
            &events,
            chunk_size,
        )
    };

    let recalc_ms = recalc_start.elapsed().as_secs_f64() * 1000.0;

    // Count stress distribution for DIRTY CHUNKS ONLY (what we just recalculated)
    {
        let s = store.read().unwrap();
        let mut air = 0u32;
        let mut stress_zero = 0u32;
        let mut stress_dust = 0u32;   // 0.01 .. 0.4
        let mut stress_creak = 0u32;  // 0.4 .. 0.6
        let mut stress_shake = 0u32;  // 0.6 .. 0.8
        let mut stress_danger = 0u32; // 0.8 .. 1.0
        let mut stress_over = 0u32;   // >= 1.0
        let grid_size = chunk_size + 1;
        for &key in &dirty_chunks {
            if let (Some(df), Some(ssf)) = (s.density_fields.get(&key), s.stress_fields.get(&key)) {
                for z in 0..grid_size {
                    for y in 0..grid_size {
                        for x in 0..grid_size {
                            if !df.get(x, y, z).material.is_solid() { air += 1; continue; }
                            let stress = ssf.get(x, y, z);
                            if stress <= 0.001 { stress_zero += 1; }
                            else if stress < 0.4 { stress_dust += 1; }
                            else if stress < 0.6 { stress_creak += 1; }
                            else if stress < 0.8 { stress_shake += 1; }
                            else if stress < 1.0 { stress_danger += 1; }
                            else { stress_over += 1; }
                        }
                    }
                }
            }
        }
        let solid = stress_zero + stress_dust + stress_creak + stress_shake + stress_danger + stress_over;
        dbg(format!("  recalc {:.1}ms — {} dirty chunks, {} solid: {} zero, {} dust(<0.4), {} creak(<0.6), {} shake(<0.8), {} danger(<1.0), {} OVER(1.0+)",
            recalc_ms, dirty_chunks.len(), solid, stress_zero, stress_dust, stress_creak, stress_shake, stress_danger, stress_over));
    }
    dbg(format!("  overstressed={} affected_chunks={}",
        result.overstressed.len(), result.affected_chunks.len()));

    // Log stress distribution
    if !result.overstressed.is_empty() {
        let max_stress = result.overstressed.iter().map(|v| v.stress).fold(0.0f32, f32::max);
        let min_stress = result.overstressed.iter().map(|v| v.stress).fold(f32::MAX, f32::min);
        let avg_stress: f32 = result.overstressed.iter().map(|v| v.stress).sum::<f32>() / result.overstressed.len() as f32;
        dbg(format!("  overstressed: min={:.2} avg={:.2} max={:.2}", min_stress, avg_stress, max_stress));
        // Log top 5 by stress
        let mut sorted: Vec<_> = result.overstressed.iter().collect();
        sorted.sort_by(|a, b| b.stress.partial_cmp(&a.stress).unwrap_or(std::cmp::Ordering::Equal));
        // Detailed stress breakdown for top 5 voxels
        let s = store.read().unwrap();
        for (i, ov) in sorted.iter().take(5).enumerate() {
            let (wx, wy, wz) = (ov.world_x, ov.world_y, ov.world_z);
            // Reconstruct components
            let (key, lx, ly, lz) = voxel_core::stress::world_to_chunk_local(wx, wy, wz, chunk_size);
            let mat = s.density_fields.get(&key)
                .map(|df| df.get(lx, ly, lz).material)
                .unwrap_or(voxel_core::material::Material::Air);
            let hardness = stress_cfg.material_hardness[mat as u8 as usize];
            let air_below = voxel_core::stress::count_air_below(&s.density_fields, wx, wy, wz, chunk_size);

            // Count air face-neighbors for cross-section
            let mut air_faces = 0u32;
            for &(dx, dy, dz) in &[(1i32,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)] {
                if let Some((_, m)) = voxel_core::stress::sample_world(&s.density_fields, wx+dx, wy+dy, wz+dz, chunk_size) {
                    if !m.is_solid() { air_faces += 1; }
                }
            }

            dbg(format!("  top[{}]: ({},{},{}) stress={:.3} | mat={:?} hard={:.2} air_below={} air_faces={} | raw≈grav({:.2})+oh({:.2})+xsec({:.2})/h({:.2})",
                i, wx, wy, wz, ov.stress, mat, hardness, air_below, air_faces,
                stress_cfg.gravity_weight, air_below as f32 * stress_cfg.overhang_weight,
                if air_faces >= 2 { (air_faces - 1) as f32 * 0.15 } else { 0.0 },
                hardness));
        }
        drop(s);
    }

    // ── Editor trigger pre-pass ──
    //
    // CRITICAL: evaluate scripted triggers BEFORE the warning scan below so
    // the seed cells get their stress written into the stress field. Without
    // this, the dust/creak/shake warnings (and the on-screen cracks, smoke,
    // wall-shake effects on UE side) appear only at the natural overstressed
    // cells around the mine point — which is NOT where the designer painted.
    // Writing the seed stress here makes the cinematic warning visuals show
    // up at the painted region instead of at the mining point.
    let (early_has_armed, early_has_force) = {
        let s = store.read().unwrap();
        (
            s.triggers.iter().any(|t| t.armed),
            !s.force_fire_trigger_ids.is_empty(),
        )
    };
    // Build the painted-seed overstressed list AND write seed stress into
    // the stress field in one pass. The result feeds both:
    //   (a) the warning scan below — so cracks/dust/shake appear at the
    //       painted region instead of only at the mine point.
    //   (b) the force_collapse detect call further down (post-pass).
    let mut trigger_seed_overstressed: Vec<voxel_core::stress::OverstressedVoxel> = Vec::new();
    let mut trigger_extra_dirty_chunks: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut trigger_eval_log_pending: Vec<String> = Vec::new();
    if early_has_armed || early_has_force {
        let mined_aabbs: Vec<crate::triggers::VoxelAabb> = mined_dirty_events
            .iter()
            .map(|ev| {
                let r = ev.radius.max(1);
                crate::triggers::VoxelAabb {
                    min: (ev.center.0 - r, ev.center.1 - r, ev.center.2 - r),
                    max: (ev.center.0 + r, ev.center.1 + r, ev.center.2 + r),
                }
            })
            .collect();

        let mut s = store.write().unwrap();
        let forced_ids: Vec<u32> = std::mem::take(&mut s.force_fire_trigger_ids);
        let mut fired_ids: Vec<u32> = forced_ids.clone();
        for trig in &s.triggers {
            if !trig.armed || forced_ids.contains(&trig.id) {
                continue;
            }
            let fires = mined_aabbs
                .iter()
                .any(|aabb| trig.should_fire(aabb, &s.density_fields, chunk_size));
            if fires {
                fired_ids.push(trig.id);
            }
        }
        trigger_eval_log_pending.push(format!(
            "  TRIGGER eval: fired_ids={:?} (from {} mined AABBs, force-fire={})",
            fired_ids, mined_aabbs.len(), forced_ids.len()
        ));

        for tid in &fired_ids {
            let trig_clone = match s.find_trigger(*tid) {
                Some(t) => t.clone(),
                None => continue,
            };
            let mut seeded_count: u32 = 0;
            let cs = chunk_size as i32;
            for &(wx, wy, wz) in &trig_clone.target_slab_voxels {
                if !crate::engine::cell_has_solid_center(&s, wx, wy, wz, cs) {
                    continue;
                }
                let cx = wx.div_euclid(cs);
                let cy = wy.div_euclid(cs);
                let cz = wz.div_euclid(cs);
                let lx = wx.rem_euclid(cs) as usize;
                let ly = wy.rem_euclid(cs) as usize;
                let lz = wz.rem_euclid(cs) as usize;
                // Write stress into the field so the warning scan picks it
                // up (cracks/dust/shake at the painted region, not just the
                // mine point).
                if let Some(sf) = s.stress_fields.get_mut(&(cx, cy, cz)) {
                    sf.set(lx, ly, lz, 1.5);
                }
                trigger_extra_dirty_chunks.insert((cx, cy, cz));
                // Push onto the forced overstressed list for the force
                // collapse pass downstream.
                trigger_seed_overstressed.push(voxel_core::stress::OverstressedVoxel {
                    world_x: wx,
                    world_y: wy,
                    world_z: wz,
                    stress: 1.5,
                });
                seeded_count += 1;
            }
            // Disarm now — this trigger has fired.
            if let Some(t) = s.find_trigger_mut(*tid) {
                t.armed = false;
            }
            trigger_eval_log_pending.push(format!(
                "  TRIGGER {} ('{}') fired: {}/{} solid seed cells (stress_field bumped + queued for force collapse)",
                tid, trig_clone.name, seeded_count, trig_clone.target_slab_voxels.len()
            ));
        }
    }
    for line in &trigger_eval_log_pending {
        dbg(line.clone());
    }

    // Merge seed chunks into dirty_chunks so the warning scan visits them.
    // Mining-driven dirty_chunks alone might not cover seed locations far
    // from the mine point.
    if !trigger_extra_dirty_chunks.is_empty() {
        let existing: HashSet<(i32, i32, i32)> = dirty_chunks.iter().copied().collect();
        for k in &trigger_extra_dirty_chunks {
            if !existing.contains(k) {
                dirty_chunks.push(*k);
            }
        }
    }

    // Emit stress warnings for UE — scan ALL voxels with stress above dust threshold.
    // This is push-based: Rust tells UE where the stress is, UE checks proximity to player.
    {
        let s = store.read().unwrap();
        let mut warnings = Vec::new();
        let mut dust_count = 0u32;
        let mut creak_count = 0u32;
        let mut shake_count = 0u32;
        let grid_size = chunk_size + 1;

        for &(cx, cy, cz) in &dirty_chunks {
            let sf = match s.stress_fields.get(&(cx, cy, cz)) {
                Some(f) => f,
                None => continue,
            };
            for z in 0..grid_size {
                for y in 0..grid_size {
                    for x in 0..grid_size {
                        let stress = sf.get(x, y, z);
                        if stress < stress_cfg.warn_dust_threshold {
                            continue;
                        }
                        let warning_type = if stress >= stress_cfg.warn_shake_threshold {
                            shake_count += 1; 3u8
                        } else if stress >= stress_cfg.warn_creak_threshold {
                            creak_count += 1; 2u8
                        } else {
                            dust_count += 1; 1u8
                        };
                        let wx = cx * chunk_size as i32 + x as i32;
                        let wy = cy * chunk_size as i32 + y as i32;
                        let wz = cz * chunk_size as i32 + z as i32;
                        warnings.push(FfiStressWarning {
                            world_x: wx as f32 * world_scale,
                            world_y: -(wz as f32) * world_scale,
                            world_z: wy as f32 * world_scale,
                            stress,
                            warning_type,
                        });
                    }
                }
            }
        }
        if dust_count + creak_count + shake_count > 0 {
            dbg(format!("  warnings: dust={} creak={} shake={} (scanned from stress fields)",
                dust_count, creak_count, shake_count));
        }
        // Send the highest-stress warnings (sorted, capped at 128)
        warnings.sort_by(|a, b| b.stress.partial_cmp(&a.stress).unwrap_or(std::cmp::Ordering::Equal));
        warnings.truncate(128);
        if !warnings.is_empty() {
            let _ = result_tx.send(WorkerResult::StressWarnings { warnings });
        }
    }

    // Trigger eval was done in the pre-pass above (so warning visuals
    // appear at the painted region). Here we just need to run the natural
    // + forced collapse detections using already-collected data.
    if !result.overstressed.is_empty() || !trigger_seed_overstressed.is_empty() {
        let mut s = store.write().unwrap();

        let collapse_start = std::time::Instant::now();
        let mut events: Vec<voxel_core::stress::CollapseEventV2> = Vec::new();
        // Natural collapse pass: only stress-recalc'd overstressed cells.
        // Filters (size / grounding / cohesion) apply normally so player
        // mining doesn't trigger spurious cave-ins on supported rock.
        if !result.overstressed.is_empty() {
            let (density, stress, support) = s.sleep_fields_mut();
            let natural = voxel_core::stress::detect_and_execute_collapses_v2_with_options(
                density, stress, support,
                &result.overstressed,
                &stress_cfg, chunk_size,
                true, // defer_pile
            );
            events.extend(natural);
        }
        // Scripted-trigger pass: force_collapse=true. Bypasses the
        // grounding filter; designer-painted regions on cave walls or
        // pillars fall even though they're physically supported.
        if !trigger_seed_overstressed.is_empty() {
            let (density, stress, support) = s.sleep_fields_mut();
            let forced = voxel_core::stress::detect_and_execute_collapses_v2_with_force(
                density, stress, support,
                &trigger_seed_overstressed,
                &stress_cfg, chunk_size,
                true,  // defer_pile
                true,  // force_collapse
            );
            events.extend(forced);
        }
        let collapse_ms = collapse_start.elapsed().as_secs_f64() * 1000.0;

        if !events.is_empty() {
            let total_voxels: u32 = events.iter().map(|e| e.total_volume).sum();
            let total_slabs: usize = events.iter().map(|e| e.slabs.len()).sum();
            dbg(format!("  === COLLAPSE === events={} total_slabs={} total_voxels={} in {:.1}ms",
                events.len(), total_slabs, total_voxels, collapse_ms));

            for (ei, event) in events.iter().enumerate() {
                dbg(format!("  event[{}]: vol={} slabs={} center=({:.1},{:.1},{:.1}) affected_chunks={}",
                    ei, event.total_volume, event.slabs.len(),
                    event.center.0, event.center.1, event.center.2,
                    event.affected_chunks.len()));
                for (si, slab) in event.slabs.iter().enumerate() {
                    dbg(format!("    slab[{}]: voxels={} fall={} bb=({},{},{})→({},{},{}) mat={:?}",
                        si, slab.voxels.len(), slab.fall_distance,
                        slab.bb_min.0, slab.bb_min.1, slab.bb_min.2,
                        slab.bb_max.0, slab.bb_max.1, slab.bb_max.2,
                        slab.dominant_material));
                }
            }

            // Collect all affected chunks for remeshing
            let mut all_dirty: Vec<((i32,i32,i32), usize, usize, usize, usize, usize, usize)> = Vec::new();
            for event in &events {
                for &key in &event.affected_chunks {
                    all_dirty.push((key, 0, 0, 0, chunk_size, chunk_size, chunk_size));
                }
            }
            all_dirty.sort_by_key(|&(k, ..)| k);
            all_dirty.dedup_by_key(|k| k.0);
            dbg(format!("  remeshing {} chunks", all_dirty.len()));

            // Mark collapse-modified chunks for save persistence
            let collapse_keys: Vec<_> = all_dirty.iter().map(|&(k, ..)| k).collect();
            s.modification_tracker.mark_dirty_many(&collapse_keys);

            let _base_meshes = s.remesh_dirty(&all_dirty, &cfg, world_scale);
            drop(s);

            // ── Cinematic emission ──
            //
            // For each collapse event:
            //   1. Emit a CollapseWarning result carrying AABB + ETA so UE can
            //      play Acts 1-2 (warning shake, dust, cracks) BEFORE the
            //      falling slab actor appears.
            //   2. For each slab in the event, extract a real DC mesh (works
            //      post-pile because the slab struct preserves voxel data),
            //      then emit a SlabFall result with the mesh + fall metadata.
            //      UE spawns one AVoxelCollapseSlabActor per slab fragment.
            //
            // The old CollapseResult emission is dropped — the slab+warning
            // pipeline replaces it. UE consumers should switch over.

            // Collapse region size distribution
            let mut vol_hist = [0u32; 5];
            for e in &events {
                let bucket = if e.total_volume <= 5 { 0 }
                    else if e.total_volume <= 15 { 1 }
                    else if e.total_volume <= 50 { 2 }
                    else if e.total_volume <= 150 { 3 }
                    else { 4 };
                vol_hist[bucket] += 1;
            }
            dbg(format!("  size distribution: tiny(1-5)={} small(6-15)={} medium(16-50)={} large(51-150)={} huge(150+)={}",
                vol_hist[0], vol_hist[1], vol_hist[2], vol_hist[3], vol_hist[4]));

            // Snapshot density field reference for slab extraction.
            let s_ref = store.read().unwrap();
            let voxel_scale = cfg.voxel_scale();

            // Open the cinematic diagnostics log. UE writes phase transitions
            // to the same file (different prefix) so the timeline can be
            // reconstructed end-to-end.
            use std::io::Write;
            let log_path = "D:/Unreal Projects/Mithril2026/Saved/collapse_log.txt";
            let mut log_file = std::fs::OpenOptions::new()
                .create(true).append(true).open(log_path).ok();
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis()).unwrap_or(0);
            if let Some(f) = &mut log_file {
                let _ = writeln!(f, "");
                let _ = writeln!(f, "================================================================");
                let _ = writeln!(f, "[{}][R] === COLLAPSE BURST: {} events ===", now_ms, events.len());
                let _ = writeln!(f, "================================================================");
            }

            for (event_idx, event) in events.iter().enumerate() {
                // Compute event bounds in UE units for the warning event.
                let mut ev_min = (f32::MAX, f32::MAX, f32::MAX);
                let mut ev_max = (f32::MIN, f32::MIN, f32::MIN);
                for slab in &event.slabs {
                    let lo_ue = (slab.bb_min.0 as f32 * world_scale,
                                 -(slab.bb_min.2 as f32) * world_scale,
                                 slab.bb_min.1 as f32 * world_scale);
                    let hi_ue = (slab.bb_max.0 as f32 * world_scale,
                                 -(slab.bb_max.2 as f32) * world_scale,
                                 slab.bb_max.1 as f32 * world_scale);
                    ev_min.0 = ev_min.0.min(lo_ue.0.min(hi_ue.0));
                    ev_min.1 = ev_min.1.min(lo_ue.1.min(hi_ue.1));
                    ev_min.2 = ev_min.2.min(lo_ue.2.min(hi_ue.2));
                    ev_max.0 = ev_max.0.max(lo_ue.0.max(hi_ue.0));
                    ev_max.1 = ev_max.1.max(lo_ue.1.max(hi_ue.1));
                    ev_max.2 = ev_max.2.max(lo_ue.2.max(hi_ue.2));
                }
                let center_ue = (
                    event.center.0 * world_scale,
                    -event.center.2 * world_scale,
                    event.center.1 * world_scale,
                );
                let extent_ue = (
                    (ev_max.0 - ev_min.0) * 0.5,
                    (ev_max.1 - ev_min.1) * 0.5,
                    (ev_max.2 - ev_min.2) * 0.5,
                );

                // Volume-scaled warning ETA (1.5s for tiny → 3.5s for huge).
                let eta_ms = (1500 + (event.total_volume.min(200) as u32) * 8).min(3500);
                let severity = if event.total_volume >= 80 { 4 }
                    else if event.total_volume >= 30 { 3 }
                    else if event.total_volume >= 10 { 2 }
                    else { 1 };

                // ── Diagnostic log: event header ──
                if let Some(f) = &mut log_file {
                    let _ = writeln!(f, "[{}][R] --- EVENT[{}] ---", now_ms, event_idx);
                    let _ = writeln!(f, "[{}][R]   center_rust=({:.2},{:.2},{:.2}) center_ue=({:.1},{:.1},{:.1})",
                        now_ms, event.center.0, event.center.1, event.center.2,
                        center_ue.0, center_ue.1, center_ue.2);
                    let _ = writeln!(f, "[{}][R]   total_volume={} slab_count={} affected_chunks={}",
                        now_ms, event.total_volume, event.slabs.len(), event.affected_chunks.len());
                    let _ = writeln!(f, "[{}][R]   bounds_ue: min=({:.1},{:.1},{:.1}) max=({:.1},{:.1},{:.1}) extent=({:.1},{:.1},{:.1})",
                        now_ms,
                        ev_min.0, ev_min.1, ev_min.2,
                        ev_max.0, ev_max.1, ev_max.2,
                        extent_ue.0, extent_ue.1, extent_ue.2);
                    let _ = writeln!(f, "[{}][R]   warning: severity={} eta_ms={} (PreFallWarning duration)",
                        now_ms, severity, eta_ms);
                }

                let _ = result_tx.send(WorkerResult::CollapseWarning {
                    center_ue,
                    bounds_extent_ue: extent_ue,
                    severity,
                    eta_ms,
                    volume: event.total_volume,
                });

                // Per-slab falling visuals — split each event slab into 1-5
                // visual sub-fragments via grid partition so big collapses
                // read as multiple chunks breaking off rather than one
                // monolithic block. Each sub-slab gets its own falling-slab
                // actor in UE with independent tumble axis + impact zone.
                let visual_slabs: Vec<voxel_core::stress::CollapseSlab> = event.slabs.iter()
                    .flat_map(|s| split_slab_for_visual(s).into_iter())
                    .collect();
                if let Some(f) = &mut log_file {
                    let _ = writeln!(f, "[{}][R]   visual_split: {} event slab(s) → {} sub-slab(s)",
                        now_ms, event.slabs.len(), visual_slabs.len());
                }
                for (slab_idx, slab) in visual_slabs.iter().enumerate() {
                    // Material breakdown for the slab (top 4 materials).
                    let mut mat_counts: std::collections::HashMap<voxel_core::material::Material, u32> = std::collections::HashMap::new();
                    for v in &slab.voxels {
                        *mat_counts.entry(v.material).or_insert(0) += 1;
                    }
                    let mut mat_breakdown: Vec<(voxel_core::material::Material, u32)> =
                        mat_counts.into_iter().collect();
                    mat_breakdown.sort_by(|a, b| b.1.cmp(&a.1));
                    let breakdown_str = mat_breakdown.iter().take(4)
                        .map(|(m, c)| format!("{:?}={}", m, c))
                        .collect::<Vec<_>>().join(", ");

                    let mesh_opt = crate::slab::extract_slab_mesh(
                        slab, &s_ref.density_fields, chunk_size,
                        voxel_scale, world_scale,
                    );

                    if let Some(f) = &mut log_file {
                        let _ = writeln!(f, "[{}][R]   slab[{}]:", now_ms, slab_idx);
                        let _ = writeln!(f, "[{}][R]     voxels={} fall_dist={} dom_mat={:?}",
                            now_ms, slab.voxels.len(), slab.fall_distance,
                            slab.dominant_material);
                        let _ = writeln!(f, "[{}][R]     materials: {}", now_ms, breakdown_str);
                        let _ = writeln!(f, "[{}][R]     bb_rust: min=({},{},{}) max=({},{},{})",
                            now_ms, slab.bb_min.0, slab.bb_min.1, slab.bb_min.2,
                            slab.bb_max.0, slab.bb_max.1, slab.bb_max.2);
                        match &mesh_opt {
                            Some(m) => {
                                let _ = writeln!(f, "[{}][R]     mesh extracted: positions={} indices={} ({} tris)",
                                    now_ms, m.positions.len(), m.indices.len(), m.indices.len() / 3);
                            }
                            None => {
                                let _ = writeln!(f, "[{}][R]     ★ MESH EXTRACTION FAILED — slab actor will fall back to procedural box",
                                    now_ms);
                            }
                        }
                    }

                    let Some(mesh) = mesh_opt else { continue };

                    let center_rust = slab.center;
                    let spawn_ue = (
                        center_rust.0 * world_scale,
                        -center_rust.2 * world_scale,
                        center_rust.1 * world_scale,
                    );
                    let land_ue_z = (center_rust.1 - slab.fall_distance as f32) * world_scale;
                    let land_ue = (spawn_ue.0, spawn_ue.1, land_ue_z);
                    let slab_extent_ue = (
                        (slab.bb_max.0 - slab.bb_min.0).max(1) as f32 * world_scale * 0.5,
                        (slab.bb_max.2 - slab.bb_min.2).max(1) as f32 * world_scale * 0.5,
                        (slab.bb_max.1 - slab.bb_min.1).max(1) as f32 * world_scale * 0.5,
                    );

                    // ── Principal horizontal axis = leading-edge direction ──
                    // Compute the 2×2 XZ covariance of the slab voxels and
                    // pick the dominant eigenvector. Magnitude scaled by
                    // anisotropy so chunky cubes get ~0 (no preferred tilt)
                    // and long thin slabs get ~1 (strong domino direction).
                    let (lead_ue_x, lead_ue_y) = {
                        let cx_r = center_rust.0;
                        let cz_r = center_rust.2;
                        let mut cov_xx = 0.0f32;
                        let mut cov_xz = 0.0f32;
                        let mut cov_zz = 0.0f32;
                        for v in &slab.voxels {
                            let dx = v.world_x as f32 - cx_r;
                            let dz = v.world_z as f32 - cz_r;
                            cov_xx += dx * dx;
                            cov_xz += dx * dz;
                            cov_zz += dz * dz;
                        }
                        let n = (slab.voxels.len().max(1)) as f32;
                        cov_xx /= n; cov_xz /= n; cov_zz /= n;

                        let trace = cov_xx + cov_zz;
                        let det = cov_xx * cov_zz - cov_xz * cov_xz;
                        let disc = (trace * trace - 4.0 * det).max(0.0);
                        let s = disc.sqrt();
                        let lambda_max = (trace + s) * 0.5;
                        let lambda_min = (trace - s) * 0.5;

                        // Eigenvector for lambda_max: v = (cov_xz, lambda_max - cov_xx)
                        let ex = cov_xz;
                        let ez = lambda_max - cov_xx;
                        let mag = (ex * ex + ez * ez).sqrt();
                        let (lead_rx, lead_rz) = if mag > 1e-4 {
                            (ex / mag, ez / mag)
                        } else if cov_xx >= cov_zz {
                            (1.0, 0.0)
                        } else {
                            (0.0, 1.0)
                        };

                        // Anisotropy in [0..1]: 0 = round, 1 = elongated.
                        let aniso = if lambda_max > 1e-6 {
                            (1.0 - (lambda_min / lambda_max).max(0.0)).clamp(0.0, 1.0)
                        } else { 0.0 };

                        // Rust X → UE X; Rust Z → -UE Y. Vertical Y unused
                        // because tilting is horizontal (no fall component).
                        (lead_rx * aniso, -lead_rz * aniso)
                    };

                    let mut fall = FfiSlabFallData::default();
                    fall.spawn_x = spawn_ue.0;
                    fall.spawn_y = spawn_ue.1;
                    fall.spawn_z = spawn_ue.2;
                    fall.land_x = land_ue.0;
                    fall.land_y = land_ue.1;
                    fall.land_z = land_ue.2;
                    fall.fall_distance = slab.fall_distance as f32 * world_scale;
                    fall.bounds_extent_x = slab_extent_ue.0;
                    fall.bounds_extent_y = slab_extent_ue.1;
                    fall.bounds_extent_z = slab_extent_ue.2;
                    fall.volume = slab.voxels.len() as u32;
                    fall.dominant_material = slab.dominant_material as u8;
                    // ★ CRITICAL FIX: propagate the warning ETA to every slab
                    // so the slab actor's PreFallWarning + ImminentDetach phases
                    // actually run. Without this, every slab skipped straight
                    // to Falling, rendering the cinematic invisible.
                    fall.warning_severity = severity;
                    fall.warning_eta_ms = eta_ms;
                    fall.leading_edge_dir_x = lead_ue_x;
                    fall.leading_edge_dir_y = lead_ue_y;

                    if let Some(f) = &mut log_file {
                        let _ = writeln!(f, "[{}][R]     spawn_ue=({:.1},{:.1},{:.1}) land_ue=({:.1},{:.1},{:.1}) fall_dist_ue={:.1}",
                            now_ms, spawn_ue.0, spawn_ue.1, spawn_ue.2,
                            land_ue.0, land_ue.1, land_ue.2,
                            fall.fall_distance);
                        let _ = writeln!(f, "[{}][R]     extent_ue=({:.1},{:.1},{:.1}) → emitting SlabFall (eta={}ms severity={})",
                            now_ms, slab_extent_ue.0, slab_extent_ue.1, slab_extent_ue.2,
                            eta_ms, severity);
                    }

                    let _ = result_tx.send(WorkerResult::SlabFall {
                        mesh,
                        fall_data: fall,
                    });
                }
            }
            drop(s_ref);
            if let Some(f) = &mut log_file {
                let _ = writeln!(f, "[{}][R] === BURST COMPLETE ===", now_ms);
            }

            dbg(format!("  emitted {} CollapseWarning + per-slab SlabFall events", events.len()));

            let dirty_keys: Vec<(i32, i32, i32)> = all_dirty.iter().map(|&(k, ..)| k).collect();

            // Crystal recompute can fire NOW — UE will pick up fresh placements
            // when each chunk mesh arrives in its scheduled batch.
            recompute_crystals_for_chunks(store, &cfg, &dirty_keys);
            prune_destroyed_mushrooms_for_chunks(store, &dirty_keys);

            // ── Per-event cinematic-aligned chunk-mesh scheduling. ──
            //
            // Density was already mutated (slab cells cleared + pile placed)
            // inside detect_and_execute_collapses_v2. If we send chunk meshes
            // for ALL affected chunks now, UE shows the cave roof hole AND
            // the floor pile immediately, before the falling-slab actor has
            // visually impacted.
            //
            // Fix: per event, split affected chunks into:
            //   - SLAB chunks (cave roof hole) → send at fall-start time
            //     so the roof opens up just as the slab actor visually
            //     detaches.
            //   - PILE chunks (floor pile) → send at impact time so the
            //     pile appears under the slab right when it lands.
            //   - chunks in BOTH → send at impact (the later time wins,
            //     and the dust burst masks the transition).
            //
            // Each event has its own timing — short falls update fast,
            // long falls take longer. They DON'T wait for each other.
            const SLAB_GRAVITY: f32 = 327.0; // matches UE's slab actor

            for event in &events {
                let eta_ms = (1500u32 + (event.total_volume.min(200) as u32) * 8).min(3500);
                // Imminent-detach beat sits between warning and falling.
                // Bumped from 600ms → 1600ms so the fracture cracks have a
                // full second of unobstructed peak-intensity visibility
                // BEFORE the falling slab pulls the player's eye away.
                // MUST stay in sync with `AVoxelCollapseSlabActor::ImminentDuration`
                // on the UE side (currently 1.6s).
                let imminent_ms = 1600u32;
                let max_fall_uu = event.slabs.iter()
                    .map(|s| s.fall_distance)
                    .max().unwrap_or(0) as f32 * world_scale;
                let fall_dur_ms = if max_fall_uu > 0.0 {
                    ((2.0 * max_fall_uu / SLAB_GRAVITY).sqrt() * 1000.0).clamp(500.0, 6000.0) as u32
                } else { 1000 };
                let fall_start_ms = eta_ms + imminent_ms;
                let impact_ms = fall_start_ms + fall_dur_ms;

                // Split chunk sets:
                let pile_set: std::collections::HashSet<(i32, i32, i32)> =
                    event.pile_chunks.iter().copied().collect();
                let slab_only: Vec<(i32, i32, i32)> = event.slab_chunks.iter()
                    .copied()
                    .filter(|k| !pile_set.contains(k))
                    .collect();
                let pile_or_both: Vec<(i32, i32, i32)> = event.pile_chunks.clone();

                dbg(format!("  event center=({:.1},{:.1},{:.1}) vol={} slab_only_chunks={} pile_chunks={} fall_start={}ms impact={}ms",
                    event.center.0, event.center.1, event.center.2, event.total_volume,
                    slab_only.len(), pile_or_both.len(), fall_start_ms, impact_ms));

                // Schedule slab-chunk remesh at fall-start (cave roof opens).
                // At this point, density has slab cells cleared but NO pile
                // yet (deferred), so the remesh shows roof hole only.
                //
                // ★ CRITICAL: must call `remesh_dirty` BEFORE the seam pass
                // because `batched_seam_pass_mine` reads from the cached
                // `base_meshes`. The slab cells were cleared earlier
                // (inside `detect_and_execute_collapses_v2_with_options`)
                // but the cached mesh still shows the slab in place. Without
                // the explicit re-DC, the seam pass publishes stale meshes
                // and the cave roof stays visually intact until something
                // else (e.g., the pile thread's neighborhood seam at
                // impact + PILE_PREVIEW_MS) eventually re-DCs the area.
                // Symptom: "see-through wall" / cave roof opens 1.2s late.
                if !event.slab_chunks.is_empty() {
                    let store_c = std::sync::Arc::clone(store);
                    let cfg_c = cfg.clone();
                    let tx_c = result_tx.clone();
                    let fluid_tx_c = fluid_event_tx.clone();
                    let keys = event.slab_chunks.clone();
                    let ws = world_scale;
                    let event_center_for_log = event.center;
                    std::thread::spawn(move || {
                        let log_line = |msg: String| {
                            use std::io::Write;
                            let now_ms = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_millis()).unwrap_or(0);
                            if let Ok(mut f) = std::fs::OpenOptions::new()
                                .create(true).append(true)
                                .open("D:/Unreal Projects/Mithril2026/Saved/collapse_log.txt")
                            {
                                let _ = writeln!(f, "[{}][R-SLABFX] {}", now_ms, msg);
                            }
                        };

                        std::thread::sleep(std::time::Duration::from_millis(fall_start_ms as u64));
                        log_line(format!(
                            "fall_start fired for event center=({:.1},{:.1},{:.1}) slab_chunks={}",
                            event_center_for_log.0, event_center_for_log.1, event_center_for_log.2,
                            keys.len()));

                        // Stagger consecutive collapse remeshes via the
                        // global rate limiter so the game thread isn't
                        // dumped with 6 events × N chunks at once.
                        let rate_wait = throttle_collapse_remesh();
                        if rate_wait > 0 {
                            log_line(format!("rate-limited: extra wait {}ms", rate_wait));
                        }

                        // Refresh base_meshes cache so the seam pass
                        // publishes the post-clear mesh, not the pre-clear
                        // (slab-still-there) mesh.
                        let cs = cfg_c.chunk_size;
                        let dirty_bounds: Vec<_> = keys.iter()
                            .map(|&k| (k, 0usize, 0usize, 0usize, cs, cs, cs))
                            .collect();
                        {
                            let mut s = store_c.write().unwrap();
                            let _ = s.remesh_dirty(&dirty_bounds, &cfg_c, ws);
                        }
                        log_line(format!("remeshed {} slab chunks (cache refreshed)", keys.len()));

                        // HISMs anchored on now-cleared roof surfaces must
                        // recompute too — otherwise crystals stay floating
                        // where the slab used to be.
                        recompute_crystals_for_chunks(&store_c, &cfg_c, &keys);
                        prune_destroyed_mushrooms_for_chunks(&store_c, &keys);

                        // Force fresh send by clearing the last-sent-hash so
                        // the seam pass doesn't skip thinking nothing changed.
                        {
                            let mut s = store_c.write().unwrap();
                            for k in &keys {
                                s.last_sent_mesh_hash.remove(k);
                            }
                        }

                        batched_seam_pass_mine(&keys, &cfg_c, &store_c, &tx_c, &fluid_tx_c, ws);
                        log_line(format!("batched_seam_pass_mine called (cave roof now open)"));
                    });
                }

                // Schedule pile placement at impact, then a 4-tier preview
                // reveal over PILE_PREVIEW_MS, then the real chunk remesh.
                //
                // Sequence (deferred thread):
                //   t = impact_ms                   — apply pile to density;
                //                                     extract 4 tier preview
                //                                     meshes; emit them.
                //   t = impact_ms + 0..PREVIEW_MS   — UE animates debris
                //                                     actor reveal.
                //   t = impact_ms + PREVIEW_MS      — recompute crystals,
                //                                     remesh pile_chunks, run
                //                                     seam pass; the real
                //                                     chunk mesh now matches
                //                                     the pile and the debris
                //                                     actor despawns.
                //
                // Density is applied immediately so subsequent gameplay
                // queries (mining, flatten, stress) see the correct world;
                // only the chunk MESH lags by PREVIEW_MS, which is the
                // illusion window.
                if !event.pending_piles.is_empty() {
                    // 6 tiers × 200 ms/tier = 1200 ms total reveal. Each tier
                    // is one cumulative slice of the heap; UE swaps which one
                    // is visible at each slot boundary with a small dust puff
                    // per swap. After the full window the chunk remesh fires
                    // and the debris actor self-destructs.
                    const PILE_PREVIEW_MS: u32 = 1200;
                    let store_c = std::sync::Arc::clone(store);
                    let cfg_c = cfg.clone();
                    let stress_cfg_c = stress_cfg.clone();
                    let tx_c = result_tx.clone();
                    let fluid_tx_c = fluid_event_tx.clone();
                    let pending = event.pending_piles.clone();
                    let cs_c = chunk_size;
                    let ws = world_scale;
                    let pending_count = pending.len();
                    let event_center = event.center;
                    std::thread::spawn(move || {
                        // Helper to write to the cinematic diagnostics log
                        // (same file Rust+UE share, tagged [R-DEFERRED]).
                        let log_line = |msg: String| {
                            use std::io::Write;
                            let now_ms = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_millis()).unwrap_or(0);
                            if let Ok(mut f) = std::fs::OpenOptions::new()
                                .create(true).append(true)
                                .open("D:/Unreal Projects/Mithril2026/Saved/collapse_log.txt")
                            {
                                let _ = writeln!(f, "[{}][R-DEFERRED] {}", now_ms, msg);
                            }
                        };

                        log_line(format!(
                            "thread-spawn impact_ms={} pending_count={} center=({:.1},{:.1},{:.1})",
                            impact_ms, pending_count,
                            event_center.0, event_center.1, event_center.2));

                        std::thread::sleep(std::time::Duration::from_millis(impact_ms as u64));
                        log_line(format!("woke after sleep, applying {} piles", pending_count));

                        // ── Apply pile + collect written_cells for preview ──
                        let mut pile_chunks: Vec<(i32, i32, i32)> = Vec::new();
                        let mut all_written: Vec<voxel_core::density_ops::WrittenCell> = Vec::new();
                        {
                            let mut s = match store_c.write() {
                                Ok(s) => s,
                                Err(e) => {
                                    log_line(format!("FAILED to acquire store write lock: {:?}", e));
                                    return;
                                }
                            };
                            for (i, p) in pending.iter().enumerate() {
                                let pr = voxel_core::stress::apply_pending_pile_with_result(
                                    &mut s.density_fields, &stress_cfg_c,
                                    p, cs_c,
                                );
                                log_line(format!(
                                    "  pile[{}]: collapsed_voxels={} bb=({},{},{})..({},{},{}) → {} chunks affected, {} written cells",
                                    i, p.collapsed_voxels.len(),
                                    p.bb_min.0, p.bb_min.1, p.bb_min.2,
                                    p.bb_max.0, p.bb_max.1, p.bb_max.2,
                                    pr.affected_chunks.len(), pr.written_cells.len()));
                                for k in pr.affected_chunks {
                                    pile_chunks.push(k);
                                }
                                all_written.extend(pr.written_cells);
                            }
                        }
                        pile_chunks.sort();
                        pile_chunks.dedup();
                        log_line(format!("total pile_chunks (deduped)={}, total written_cells={}",
                            pile_chunks.len(), all_written.len()));

                        if pile_chunks.is_empty() {
                            log_line("⚠ pile_chunks is empty — nothing to remesh".to_string());
                            return;
                        }

                        // ── Extract N-tier preview meshes & emit ──
                        // Use the first pending pile's spawn coord as the
                        // anchor UE keys the debris actor by. (Each event
                        // has a single pending pile in practice; if there
                        // were multiple, we'd loop and emit per-pile.)
                        // Tier count comes from PILE_PREVIEW_TIER_COUNT
                        // (currently 8) so this code stays correct on bumps.
                        let voxel_scale = 1.0_f32;
                        let tier_meshes = crate::pile_preview::extract_pile_tier_meshes(
                            &all_written, cs_c, voxel_scale, ws,
                        );
                        // Anchor in UE coords: pile bbox center at landing Y.
                        let anchor_p = &pending[0];
                        let cx = ((anchor_p.bb_min.0 + anchor_p.bb_max.0) as f32 * 0.5
                            + anchor_p.landing_offset as f32 * 0.0) as f32;
                        let cy = ((anchor_p.bb_min.1 + anchor_p.bb_max.1) as f32 * 0.5
                            - anchor_p.landing_offset as f32) as f32;
                        let cz = ((anchor_p.bb_min.2 + anchor_p.bb_max.2) as f32 * 0.5) as f32;
                        // Convert Rust voxel coord → UE world (Y-up→Z-up).
                        let ue_x = cx * ws;
                        let ue_y = -cz * ws;
                        let ue_z = cy * ws;
                        let bbox_x = ((anchor_p.bb_max.0 - anchor_p.bb_min.0).max(1)) as f32 * ws;
                        let bbox_y = ((anchor_p.bb_max.1 - anchor_p.bb_min.1).max(1)) as f32 * ws;
                        let bbox_z = ((anchor_p.bb_max.2 - anchor_p.bb_min.2).max(1)) as f32 * ws;
                        let total_volume: u32 = pending.iter()
                            .map(|p| p.collapsed_voxels.len() as u32).sum();
                        let dom_mat = anchor_p.dominant_material as u8;

                        for (tier_idx, mesh) in tier_meshes.into_iter().enumerate() {
                            let mut fall = FfiSlabFallData::default();
                            fall.spawn_x = ue_x;
                            fall.spawn_y = ue_y;
                            fall.spawn_z = ue_z;
                            fall.land_x = ue_x;
                            fall.land_y = ue_y;
                            fall.land_z = ue_z;
                            fall.bounds_extent_x = bbox_x;
                            fall.bounds_extent_y = bbox_z; // Y-up→Z-up swap
                            fall.bounds_extent_z = bbox_y;
                            fall.volume = total_volume;
                            fall.dominant_material = dom_mat;
                            fall.pile_tier_index = tier_idx as u8;
                            fall.warning_eta_ms = PILE_PREVIEW_MS;
                            let _ = tx_c.send(WorkerResult::PilePreviewTier {
                                mesh,
                                fall_data: fall,
                            });
                        }
                        log_line(format!("emitted {} tier preview meshes anchor=({:.1},{:.1},{:.1})",
                            crate::pile_preview::PILE_PREVIEW_TIER_COUNT,
                            ue_x, ue_y, ue_z));

                        // ── Hold the preview window ──
                        std::thread::sleep(std::time::Duration::from_millis(PILE_PREVIEW_MS as u64));

                        // ── Throttle through the global rate limiter ──
                        // Multi-region collapses queue 6+ deferred threads
                        // that all wake within ~1.6 s after the first
                        // impact. Without throttling they'd all dump their
                        // chunk remesh + seam pass results onto the game
                        // thread back-to-back, producing the "2 second
                        // freeze" the user reported. Rate limiter spreads
                        // these batches by `COLLAPSE_REMESH_GAP_MS` each.
                        let rate_wait = throttle_collapse_remesh();
                        if rate_wait > 0 {
                            log_line(format!("rate-limited: extra wait {}ms before remesh", rate_wait));
                        }

                        // ── Real commit: crystals + remesh + seam ──
                        recompute_crystals_for_chunks(&store_c, &cfg_c, &pile_chunks);
                        prune_destroyed_mushrooms_for_chunks(&store_c, &pile_chunks);
                        log_line("crystals recomputed".to_string());

                        // ★ CRITICAL: re-DC the chunks so `base_meshes` cache
                        // reflects the just-placed pile. `batched_seam_pass_mine`
                        // reads from `base_meshes` — without this remesh, the
                        // seam pass would publish stale meshes.
                        let cs = cfg_c.chunk_size;
                        let dirty_bounds: Vec<_> = pile_chunks.iter()
                            .map(|&k| (k, 0usize, 0usize, 0usize, cs, cs, cs))
                            .collect();
                        {
                            let mut s = store_c.write().unwrap();
                            let _ = s.remesh_dirty(&dirty_bounds, &cfg_c, ws);
                        }
                        log_line(format!("remeshed {} pile chunks (base_meshes cache refreshed)",
                            pile_chunks.len()));

                        // Force fresh send by clearing the last-sent-hash.
                        {
                            let mut s = store_c.write().unwrap();
                            let mut cleared = 0u32;
                            for k in &pile_chunks {
                                if s.last_sent_mesh_hash.remove(k).is_some() {
                                    cleared += 1;
                                }
                            }
                            log_line(format!("cleared {} stale hashes from {} pile chunks",
                                cleared, pile_chunks.len()));
                        }

                        batched_seam_pass_mine(&pile_chunks, &cfg_c, &store_c, &tx_c, &fluid_tx_c, ws);
                        log_line(format!("batched_seam_pass_mine called for {} chunks", pile_chunks.len()));
                    });
                } else {
                    if let Some(f) = &mut log_file {
                        let _ = writeln!(f, "[{}][R] ⚠ event has no pending_piles — pile defer was a no-op (pile already placed inline?)", now_ms);
                    }
                }
            }
        } else {
            dbg(format!("  no collapses triggered (overstressed={} but regions too small or grounded) in {:.1}ms",
                result.overstressed.len(), collapse_ms));
        }
    } else {
        dbg(format!("  no overstressed voxels — stress is stable"));
    }

    dbg("=== STRESS RECALC END ===".to_string());
    true
}

fn handle_request(
    req: WorkerRequest,
    result_tx: &Sender<WorkerResult>,
    store: &Arc<RwLock<ChunkStore>>,
    config: &Arc<RwLock<GenerationConfig>>,
    stress_config: &Arc<RwLock<StressConfig>>,
    generation_counters: &Arc<DashMap<(i32, i32, i32), AtomicU64>>,
    world_scale: f32,
    fluid_event_tx: &Sender<FluidEvent>,
    profiler: &Arc<StreamingProfiler>,
    worker_id: usize,
    generate_rx: &Receiver<WorkerRequest>,
    mine_rx: &Receiver<WorkerRequest>,
    morph_manifest: &Arc<Mutex<Option<voxel_sleep::ChangeManifest>>>,
    regions_in_flight: &Arc<DashMap<(i32, i32, i32), Arc<Mutex<()>>>>,
) {
    match req {
        // ComputePath is handled exclusively by the dedicated path-worker
        // (see `path_worker_loop`). If it ever lands here it means routing
        // confusion — silently drop rather than panic.
        WorkerRequest::ComputePath { .. } => {}
        WorkerRequest::PriorityGenerate { chunk, generation } |
        WorkerRequest::Generate { chunk, generation } => {
            let chunk_start = Instant::now();
            let profiling = profiler.is_enabled();

            // Check if this generation is still current (stale detection)
            if let Some(counter) = generation_counters.get(&chunk) {
                if counter.load(Ordering::Relaxed) != generation {
                    profiler.record_stale_skip(worker_id);
                    return; // Stale request, skip
                }
            }

            let cfg = config.read().unwrap().clone();
            let rk = region_key(chunk.0, chunk.1, chunk.2, cfg.region_size);

            // Timing accumulators
            let mut t_region_density = Duration::ZERO;
            let mut t_hermite = Duration::ZERO;
            let mut t_store_read_wait = Duration::ZERO;
            let mut t_store_write_wait = Duration::ZERO;
            let mut t_dc_solve = Duration::ZERO;
            let mut t_mesh_gen = Duration::ZERO;
            let mut t_mesh_smooth = Duration::ZERO;
            let mut was_slow_path = false;
            let mut region_timings = RegionTimings::default();
            let mut t_worm_forward_sharing = Duration::ZERO;
            let mut t_worm_backward_carve = Duration::ZERO;
            let mut t_worm_backward_remesh = Duration::ZERO;
            let mut backward_dirty_count: u32 = 0;

            // Tracked across the slow-path block so the per-snapshot smart
            // re-seam below (after incremental_seam_pass) knows which chunks
            // had snapshots restored during this region regen. Required to
            // force-clear stale hashes for their backward face neighbors —
            // otherwise hash-skip suppresses the new combined meshes.
            let mut applied_snapshots: Vec<(i32, i32, i32)> = Vec::new();

            // Tracks chunks whose seam_data + base_mesh were freshly regenerated
            // by the post-sync DC re-solve (STALE-SEAM_DATA fix). Their old
            // dc_vertices baked against pre-sync hermite would have produced
            // degenerate seams when neighbors stitched against them; after
            // re-solving we need to ship the new combined meshes via a final
            // batched_seam_pass so UE actually receives the correct geometry.
            let mut sync_remeshed: Vec<(i32, i32, i32)> = Vec::new();

            // Fast path: region generated AND this chunk has data → mesh under one read lock
            let mesh_result = {
                let t0 = Instant::now();
                let s = store.read().unwrap();
                if profiling { t_store_read_wait += t0.elapsed(); }

                if s.is_region_generated(&rk) && s.has_density(&chunk) {
                    let density = s.density_fields.get(&chunk).unwrap();
                    let hermite = s.hermite_data.get(&chunk).unwrap();
                    let cell_size = density.size - 1;

                    let t1 = Instant::now();
                    let dc_verts = solve_dc_vertices(hermite, cell_size);
                    t_dc_solve += t1.elapsed();

                    let t2 = Instant::now();
                    let mut m = generate_mesh(hermite, &dc_verts, cell_size);
                    t_mesh_gen += t2.elapsed();

                    let t_s = Instant::now();
                    m.smooth(cfg.mesh_smooth_iterations, cfg.mesh_smooth_strength, cfg.mesh_boundary_smooth, Some(cell_size));
                    if cfg.mesh_recalc_normals > 0 { m.recalculate_normals(); }
                    t_mesh_smooth += t_s.elapsed();

                    let b_edges = region_gen::extract_boundary_edges(hermite, cfg.chunk_size);
                    Some((m, dc_verts, b_edges))
                } else {
                    None
                }
            };

            let mut pool_fluid_seeds: Vec<voxel_gen::pools::FluidSeed> = Vec::new();
            let mut region_river_springs: Vec<((i32, i32, i32), voxel_gen::springs::SpringDescriptor)> = Vec::new();
            let mut region_zone_descriptors: Vec<FfiZoneDescriptor> = Vec::new();

            let (mesh, dc_vertices, boundary_edges) = if let Some(result) = mesh_result {
                result
            } else {
                // Fix A: Per-region mutex — blocks if another worker is
                // generating this region, preventing redundant slow-path work.
                let region_mutex = regions_in_flight
                    .entry(rk)
                    .or_insert_with(|| Arc::new(Mutex::new(())))
                    .clone();
                // The mutex guards (), not data — there's no invariant a
                // panicked worker could have violated. Recover from
                // poisoning so a single panic in region-gen code (e.g. an
                // OOB in voxel-gen) does NOT cascade through every other
                // worker that touches the same region. Without this, every
                // peer hits PoisonError on lock().unwrap() and the whole
                // pool dies — which is exactly how PANIC #1 took out 4
                // peers via worker.rs:596 PoisonError.
                let _region_guard = region_mutex.lock().unwrap_or_else(|p| p.into_inner());

                // Re-check fast path under the guard — the owner may have
                // just finished generating this region while we were waiting.
                let retry_result = {
                    let t0 = Instant::now();
                    let s = store.read().unwrap();
                    if profiling { t_store_read_wait += t0.elapsed(); }

                    if s.is_region_generated(&rk) && s.has_density(&chunk) {
                        let density = s.density_fields.get(&chunk).unwrap();
                        let hermite = s.hermite_data.get(&chunk).unwrap();
                        let cell_size = density.size - 1;

                        let t1 = Instant::now();
                        let dc_verts = solve_dc_vertices(hermite, cell_size);
                        t_dc_solve += t1.elapsed();

                        let t2 = Instant::now();
                        let mut m = generate_mesh(hermite, &dc_verts, cell_size);
                        t_mesh_gen += t2.elapsed();

                        let t_s = Instant::now();
                        m.smooth(cfg.mesh_smooth_iterations, cfg.mesh_smooth_strength, cfg.mesh_boundary_smooth, Some(cell_size));
                        if cfg.mesh_recalc_normals > 0 { m.recalculate_normals(); }
                        t_mesh_smooth += t_s.elapsed();

                        let b_edges = region_gen::extract_boundary_edges(hermite, cfg.chunk_size);
                        Some((m, dc_verts, b_edges))
                    } else {
                        None
                    }
                };

                if let Some(result) = retry_result {
                    result
                } else {
                // Slow path: (re)generate region densities.
                // Region guard is held throughout — other workers for this
                // region block on the mutex until we finish.
                was_slow_path = true;

                let t0 = Instant::now();
                let coords = region_chunks(rk, cfg.region_size);
                let (mut densities, _pools, fluid_seeds, worm_paths, rt, river_springs, zones) = generate_region_densities(&coords, &cfg);
                pool_fluid_seeds = fluid_seeds;
                region_river_springs = river_springs;

                // Convert zone descriptors to FFI format with coordinate transform
                {
                    let voxel_scale = cfg.effective_bounds() / cfg.chunk_size as f32;
                    let scale = voxel_scale * world_scale;
                    region_zone_descriptors = zones.iter().map(|zd| {
                        FfiZoneDescriptor {
                            zone_type: zd.zone_type as u8,
                            // Rust Y-up → UE Z-up: swap Y↔Z, negate new Y
                            center_x: zd.center.x * scale,
                            center_y: -zd.center.z * scale,
                            center_z: zd.center.y * scale,
                            min_x: zd.world_min.x * scale,
                            min_y: -zd.world_max.z * scale,
                            min_z: zd.world_min.y * scale,
                            max_x: zd.world_max.x * scale,
                            max_y: -zd.world_min.z * scale,
                            max_z: zd.world_max.y * scale,
                        }
                    }).collect();
                }
                t_region_density += t0.elapsed();
                region_timings = rt;
                if profiling {
                }
                // Forward sharing: apply worm paths from already-generated regions
                // into our new density fields (before hermite extraction)
                let t_fwd = Instant::now();
                {
                    let s = store.read().unwrap();
                    let stored = s.get_all_region_worm_paths();
                    let mut external: Vec<&[voxel_gen::worm::path::WormSegment]> = Vec::new();
                    for (rk_other, paths) in stored {
                        if *rk_other == rk { continue; }
                        for path in paths {
                            external.push(path);
                        }
                    }
                    if !external.is_empty() {
                        let as_vecs: Vec<Vec<voxel_gen::worm::path::WormSegment>> =
                            external.into_iter().map(|s| s.to_vec()).collect();
                        region_gen::apply_external_worm_paths(&mut densities, &as_vecs, &cfg);
                        // Recompute metadata after carving external worms
                        for density in densities.values_mut() {
                            density.compute_metadata();
                        }
                        // Re-sync intra-region boundaries broken by external worm carving
                        sync_region_boundary_densities(&mut densities, cfg.chunk_size);
                    }
                }
                if profiling { t_worm_forward_sharing = t_fwd.elapsed(); }

                // Check for pending mine requests between phases
                try_handle_mine(mine_rx, result_tx, store, config, world_scale, fluid_event_tx);

                // Pre-extract hermite data BEFORE acquiring write lock (expensive part).
                // Round 6 Fix B experiment: tried serial here to avoid rayon contention
                // across 8 workers. Reverted — measured +58% regression on initial_load.
                // Parallelism wins even with contention.
                let t2 = Instant::now();
                let keyed_data: Vec<_> = densities
                    .into_par_iter()
                    .map(|(key, density)| {
                        let hermite = extract_hermite_data(&density);
                        (key, density, hermite)
                    })
                    .collect();
                t_hermite += t2.elapsed();

                // Write lock held only for fast inserts + worm path storage
                {
                    let t1 = Instant::now();
                    let mut s = store.write().unwrap();
                    if profiling { t_store_write_wait += t1.elapsed(); }

                    if !s.is_region_generated(&rk) || !s.has_density(&chunk) {
                        let mut newly_inserted: Vec<(i32, i32, i32)> = Vec::new();
                        for (key, density, hermite) in keyed_data {
                            if !s.has_density(&key) {
                                s.insert(key, density, hermite);
                                newly_inserted.push(key);
                                // Apply saved snapshot if loading a saved game
                                if s.apply_pending_snapshot(key) {
                                    applied_snapshots.push(key);
                                    // Density was patched — re-extract hermite from patched data.
                                    // (May be re-extracted again below if re-sync touches this chunk.)
                                    if let Some(df) = s.density_fields.get(&key) {
                                        let df_clone = df.clone();
                                        let new_hermite = extract_hermite_data(&df_clone);
                                        s.hermite_data.insert(key, new_hermite);
                                    }
                                }
                            }
                        }
                        s.mark_region_generated(rk);

                        // ── CRITICAL: re-sync ALL region chunks after insert ──
                        //
                        // Two bug classes covered here:
                        //
                        // 1) SNAPSHOT-OVERWRITE: apply_pending_snapshot rewrites
                        //    all 29 791 cells of restored chunks — including
                        //    boundary cells that sync_region_boundary_densities
                        //    just made consistent in the temp HashMap. Without
                        //    re-sync, restored chunk boundaries diverge from
                        //    neighbors → seam holes / mesh gaps.
                        //
                        // 2) STALE-PRE-EXISTING: the insert loop above uses
                        //    `if !s.has_density(&key)` so it SKIPS chunks
                        //    already in store. If region regen runs a second
                        //    time (e.g. region marker cleared by an unload but
                        //    some chunks stayed loaded, OR config changed
                        //    mid-session like BlankCanvas being toggled), the
                        //    fresh sync'd density gets discarded for those
                        //    pre-existing chunks. They keep their old values
                        //    and disagree with their freshly-streamed
                        //    neighbors → 961-cell face mismatches in the
                        //    diagnostic.
                        //
                        // Fix: re-sync EVERY chunk in `coords` against its 26
                        // neighbors using the store's current density. This
                        // cleans up both bug classes regardless of which
                        // chunks were freshly inserted vs already-existing,
                        // and regardless of whether snapshots were applied.
                        //
                        // sync_chunk_full_boundaries is idempotent (min(a,b) =
                        // min(min(a,b),b)) so duplicate work between adjacent
                        // chunks in the region is harmless. Hermite
                        // re-extraction only fires for chunks whose density
                        // actually changed, so the cost scales with the size
                        // of inconsistency — typically zero on a clean region.
                        {
                            use std::collections::HashSet;
                            let mut hermite_dirty: HashSet<(i32, i32, i32)> = HashSet::new();
                            for &chunk_key in coords.iter() {
                                if !s.density_fields.contains_key(&chunk_key) {
                                    continue;
                                }
                                let resync_dirty = s
                                    .sync_chunk_full_boundaries(chunk_key, cfg.chunk_size);
                                for k in resync_dirty {
                                    hermite_dirty.insert(k);
                                }
                            }
                            for key in hermite_dirty {
                                // Re-extract hermite from patched density.
                                let new_hermite_opt = s.density_fields.get(&key).map(|df| {
                                    let df_clone = df.clone();
                                    extract_hermite_data(&df_clone)
                                });
                                let Some(new_hermite) = new_hermite_opt else { continue; };

                                // STALE-SEAM_DATA fix: if this chunk was previously
                                // fully meshed (has seam_data + base_mesh from a prior
                                // load), the dc_vertices in its seam_data were solved
                                // against the OLD hermite. After sync touched its
                                // boundary, the dc_vertices are now stale — neighbors
                                // looking up cell positions from this chunk's seam_data
                                // get coordinates that don't match the current density,
                                // producing degenerate/missing seam quads at the shared
                                // face. Symptom: chunks render with their base mesh but
                                // have a black gap toward this neighbor that only
                                // Force Resync clears.
                                //
                                // Re-solve DC + regen base_mesh + boundary_edges
                                // against the fresh hermite so seam computations against
                                // this chunk produce correct geometry. Track the keys so
                                // a batched seam pass can re-emit downstream.
                                let prev_meshed = s.chunk_seam_data.contains_key(&key)
                                    && s.base_meshes.contains_key(&key);
                                if prev_meshed {
                                    let cell_size = cfg.chunk_size;
                                    let dc_verts = solve_dc_vertices(&new_hermite, cell_size);
                                    let mut mesh = generate_mesh(&new_hermite, &dc_verts, cell_size);
                                    mesh.smooth(
                                        cfg.mesh_smooth_iterations,
                                        cfg.mesh_smooth_strength,
                                        cfg.mesh_boundary_smooth,
                                        Some(cell_size),
                                    );
                                    if cfg.mesh_recalc_normals > 0 { mesh.recalculate_normals(); }
                                    let b_edges = region_gen::extract_boundary_edges(&new_hermite, cfg.chunk_size);
                                    s.hermite_data.insert(key, new_hermite);
                                    s.base_meshes.insert(key, mesh);
                                    s.add_seam_data(key, ChunkSeamData {
                                        dc_vertices: dc_verts,
                                        world_origin: glam::Vec3::ZERO,
                                        boundary_edges: b_edges,
                                    });
                                    sync_remeshed.push(key);
                                } else {
                                    s.hermite_data.insert(key, new_hermite);
                                }
                            }
                        }
                    }
                    s.store_region_worms(rk, worm_paths.clone());
                }
                // Backward sharing: carve new worms into already-loaded chunks
                // from other regions, then re-extract hermite and re-mesh
                if !worm_paths.is_empty() {
                    let eb = cfg.effective_bounds();
                    let mut backward_dirty: Vec<(i32, i32, i32)> = Vec::new();
                    let t_bwd_carve = Instant::now();
                    // Phase 1: Carve worms + compute_metadata (write lock)
                    {
                        let mut s = store.write().unwrap();
                        for path in &worm_paths {
                            if path.is_empty() { continue; }
                            let (path_min, path_max) = region_gen::worm_path_aabb(path);
                            let min_cx = (path_min.x / eb).floor() as i32;
                            let max_cx = (path_max.x / eb).floor() as i32;
                            let min_cy = (path_min.y / eb).floor() as i32;
                            let max_cy = (path_max.y / eb).floor() as i32;
                            let min_cz = (path_min.z / eb).floor() as i32;
                            let max_cz = (path_max.z / eb).floor() as i32;

                            for cz in min_cz..=max_cz {
                                for cy in min_cy..=max_cy {
                                    for cx in min_cx..=max_cx {
                                        let key = (cx, cy, cz);
                                        if coords.contains(&key) { continue; }
                                        // Critical: skip chunks the user has hand-edited.
                                        // Backward worm carving writes directly into live
                                        // density without going through the snapshot
                                        // preserve/restore mechanism — without this guard,
                                        // entering a fresh region next to a hand-authored
                                        // area destroys the player's custom geometry
                                        // (walls, structures, painted detail) by carving
                                        // worm tunnels through it.
                                        if s.modification_tracker.dirty_chunks.contains(&key) {
                                            continue;
                                        }
                                        if let Some(density) = s.density_fields.get_mut(&key) {
                                            let coord = voxel_core::chunk::ChunkCoord::new(cx, cy, cz);
                                            if !voxel_gen::worm::carve::worm_overlaps_chunk(
                                                path,
                                                coord.world_origin_bounds(eb),
                                                density.size,
                                            ) {
                                                continue;
                                            }
                                            voxel_gen::worm::carve::carve_worm_into_density(
                                                density,
                                                path,
                                                coord.world_origin_bounds(eb),
                                                cfg.worm.falloff_power,
                                            );
                                            density.compute_metadata();
                                            if !backward_dirty.contains(&key) {
                                                backward_dirty.push(key);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Sync backward-carved chunks with their loaded neighbors
                    if !backward_dirty.is_empty() {
                        let extra_dirty = {
                            let mut s = store.write().unwrap();
                            s.sync_cross_region_densities(&backward_dirty, cfg.chunk_size)
                        };
                        for key in extra_dirty {
                            if !backward_dirty.contains(&key) {
                                backward_dirty.push(key);
                            }
                        }
                    }
                    // Phase 2: Extract hermite (read lock — doesn't block other readers)
                    if !backward_dirty.is_empty() {
                        let hermite_updates: Vec<_> = {
                            let s = store.read().unwrap();
                            backward_dirty.iter().filter_map(|&key| {
                                s.density_fields.get(&key).map(|d| (key, extract_hermite_data(d)))
                            }).collect()
                        };
                        // Phase 3: Store hermite results (brief write lock)
                        {
                            let mut s = store.write().unwrap();
                            for (key, hermite) in hermite_updates {
                                s.hermite_data.insert(key, hermite);
                            }
                        }
                    }
                    if profiling { t_worm_backward_carve = t_bwd_carve.elapsed(); }
                    backward_dirty_count = backward_dirty.len() as u32;

                    // Check for pending mine requests between phases
                    try_handle_mine(mine_rx, result_tx, store, config, world_scale, fluid_event_tx);

                    let t_bwd_remesh = Instant::now();
                    for &key in &backward_dirty {
                        // 1. Solve DC + generate mesh from updated hermite (read lock)
                        let computed = {
                            let s = store.read().unwrap();
                            let density = s.density_fields.get(&key);
                            let hermite = s.hermite_data.get(&key);
                            if let (Some(density), Some(hermite)) = (density, hermite) {
                                let cell_size = density.size - 1;
                                let dc_verts = solve_dc_vertices(hermite, cell_size);
                                let mut m = generate_mesh(hermite, &dc_verts, cell_size);
                                m.smooth(cfg.mesh_smooth_iterations, cfg.mesh_smooth_strength,
                                         cfg.mesh_boundary_smooth, Some(cell_size));
                                if cfg.mesh_recalc_normals > 0 { m.recalculate_normals(); }
                                let b_edges = region_gen::extract_boundary_edges(hermite, cfg.chunk_size);
                                Some((dc_verts, m, b_edges))
                            } else { None }
                        };
                        if let Some((dc_verts, mesh, _b_edges)) = computed {
                            // Update seam data + base mesh (write lock).
                            // No base-only send — batched_seam_pass below is the sole
                            // sender. A base-only send here races with other workers'
                            // seam passes: if their combined-mesh send arrives first
                            // and records the hash, this base-only arrives later and
                            // wipes seams via ClearAllMeshSections on UE, while
                            // batched_seam_pass then hash-skips the redo.
                            let mut s = store.write().unwrap();
                            s.add_seam_data(key, ChunkSeamData {
                                dc_vertices: dc_verts,
                                world_origin: glam::Vec3::ZERO,
                                boundary_edges: _b_edges,
                            });
                            s.base_meshes.insert(key, mesh);
                        }
                    }
                    if profiling { t_worm_backward_remesh = t_bwd_remesh.elapsed(); }

                    // Seam pass for backward-carved chunks: the base-only sends above
                    // land on UE via ClearAllMeshSections, wiping any seams those chunks
                    // had before. Re-stitch seams now so the chunks don't show gaps.
                    if !backward_dirty.is_empty() {
                        batched_seam_pass(&backward_dirty, &cfg, store, result_tx, fluid_event_tx, world_scale);
                    }
                }

                // Cross-region boundary density sync: ensure region edge chunks
                // match their already-loaded neighbors from other regions
                {
                    // Phase 1: Sync densities (write lock)
                    let all_dirty_keys: Vec<(i32, i32, i32)>;
                    {
                        let mut s = store.write().unwrap();
                        all_dirty_keys = s.sync_cross_region_densities(&coords, cfg.chunk_size);
                    }

                    let non_region_dirty: Vec<(i32, i32, i32)> = {
                        let region_set: HashSet<_> = coords.iter().copied().collect();
                        all_dirty_keys.iter().copied().filter(|k| !region_set.contains(k)).collect()
                    };

                    if !all_dirty_keys.is_empty() {
                        // Phase 2: Extract hermite for all dirty chunks (read lock)
                        let hermite_updates: Vec<_> = {
                            let s = store.read().unwrap();
                            all_dirty_keys.iter().filter_map(|&key| {
                                s.density_fields.get(&key).map(|d| (key, extract_hermite_data(d)))
                            }).collect()
                        };

                        // Phase 3: Store hermite results (brief write lock)
                        {
                            let mut s = store.write().unwrap();
                            for (key, hermite) in hermite_updates {
                                s.hermite_data.insert(key, hermite);
                            }
                        }

                        // Phase 4: Remesh non-region dirty chunks
                        let region_set: HashSet<_> = coords.iter().copied().collect();
                        for &key in all_dirty_keys.iter().filter(|k| !region_set.contains(k)) {
                            let computed = {
                                let s = store.read().unwrap();
                                let density = s.density_fields.get(&key);
                                let hermite = s.hermite_data.get(&key);
                                if let (Some(density), Some(hermite)) = (density, hermite) {
                                    let cell_size = density.size - 1;
                                    let dc_verts = solve_dc_vertices(hermite, cell_size);
                                    let mut m = generate_mesh(hermite, &dc_verts, cell_size);
                                    m.smooth(cfg.mesh_smooth_iterations, cfg.mesh_smooth_strength,
                                             cfg.mesh_boundary_smooth, Some(cell_size));
                                    if cfg.mesh_recalc_normals > 0 { m.recalculate_normals(); }
                                    let b_edges = region_gen::extract_boundary_edges(hermite, cfg.chunk_size);
                                    Some((dc_verts, m, b_edges))
                                } else { None }
                            };
                            if let Some((dc_verts, mesh, b_edges)) = computed {
                                // No base-only send — see backward-carve comment above.
                                let mut s = store.write().unwrap();
                                s.add_seam_data(key, ChunkSeamData {
                                    dc_vertices: dc_verts,
                                    world_origin: glam::Vec3::ZERO,
                                    boundary_edges: b_edges,
                                });
                                s.base_meshes.insert(key, mesh);
                            }
                        }
                    }

                    // Seam pass for cross-region dirty chunks: sole sender.
                    if !non_region_dirty.is_empty() {
                        batched_seam_pass(&non_region_dirty, &cfg, store, result_tx, fluid_event_tx, world_scale);
                    }
                }

                // Mesh under fresh read lock
                let t3 = Instant::now();
                let s = store.read().unwrap();
                if profiling { t_store_read_wait += t3.elapsed(); }

                let density = match s.density_fields.get(&chunk) {
                    Some(d) => d,
                    None => {
                        profiler.record_error();
                        let _ = result_tx.send(WorkerResult::Error { chunk, generation });
                        return;
                    }
                };
                let hermite = match s.hermite_data.get(&chunk) {
                    Some(h) => h,
                    None => {
                        profiler.record_error();
                        let _ = result_tx.send(WorkerResult::Error { chunk, generation });
                        return;
                    }
                };
                let cell_size = density.size - 1;

                let t4 = Instant::now();
                let dc_verts = solve_dc_vertices(hermite, cell_size);
                if profiling { t_dc_solve += t4.elapsed(); }

                let t5 = Instant::now();
                let mut m = generate_mesh(hermite, &dc_verts, cell_size);
                if profiling { t_mesh_gen += t5.elapsed(); }

                let t_s = Instant::now();
                m.smooth(cfg.mesh_smooth_iterations, cfg.mesh_smooth_strength, cfg.mesh_boundary_smooth, Some(cell_size));
                if cfg.mesh_recalc_normals > 0 { m.recalculate_normals(); }
                if profiling { t_mesh_smooth += t_s.elapsed(); }

                let b_edges = region_gen::extract_boundary_edges(hermite, cfg.chunk_size);
                (m, dc_verts, b_edges)
                }
            };

            // Targeted dump: when worldgen finishes for the suspect chunk(s),
            // write the full mesh + density stats to a focused log so we can
            // see what produced the visible "slate cube wall" without sifting
            // through every chunk in the world. Targets are hardcoded; flip
            // the empty array off when not debugging.
            {
                const TARGETS: &[(i32, i32, i32)] = &[
                    (-2, -1, -1), // user-confirmed slate-cube chunk
                    (-1, -1, -1), (-3, -1, -1),
                    (-2, 0, -1), (-2, -2, -1),
                    (-2, -1, 0), (-2, -1, -2),
                ];
                if TARGETS.contains(&chunk) {
                    use std::io::Write;
                    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true)
                        .open("D:/Unreal Projects/Mithril2026/Saved/chunk_dump.txt")
                    {
                        let s = store.read().unwrap();
                        let df = s.density_fields.get(&chunk);
                        let _ = writeln!(f, "\n=== chunk {:?} ===", chunk);
                        // Sample density slices on each chunk boundary face — if the
                        // cells just INSIDE a boundary are uniformly solid (e.g. all 1.0)
                        // and the boundary cell got sync'd to air, DC produces a flat
                        // wall. If they have natural noise variation (0.3, 0.7, 0.5),
                        // the wall is just a normal rock face.
                        if let Some(df) = df {
                            let cs = cfg.chunk_size;
                            // For each face, sample density at lx=cs-1 (just inside)
                            // and lx=cs (boundary itself), in a 5x5 grid spread.
                            let _ = writeln!(f, "boundary slice +X (interior cell x=cs-1=29):");
                            for v in (0..=cs).step_by(cs/4) {
                                let mut row = String::new();
                                for u in (0..=cs).step_by(cs/4) {
                                    let s_in = df.get(cs-1, u, v);
                                    let s_b = df.get(cs, u, v);
                                    row += &format!(" [{:>5.2}|{:>5.2}]", s_in.density, s_b.density);
                                }
                                let _ = writeln!(f, "  y/z={}:{}", v, row);
                            }
                            let _ = writeln!(f, "boundary slice +Y (interior cell y=cs-1=29):");
                            for v in (0..=cs).step_by(cs/4) {
                                let mut row = String::new();
                                for u in (0..=cs).step_by(cs/4) {
                                    let s_in = df.get(u, cs-1, v);
                                    let s_b = df.get(u, cs, v);
                                    row += &format!(" [{:>5.2}|{:>5.2}]", s_in.density, s_b.density);
                                }
                                let _ = writeln!(f, "  x/z={}:{}", v, row);
                            }
                            let _ = writeln!(f, "boundary slice -Z (interior cell z=1, vs boundary z=0):");
                            for v in (0..=cs).step_by(cs/4) {
                                let mut row = String::new();
                                for u in (0..=cs).step_by(cs/4) {
                                    let s_in = df.get(u, v, 1);
                                    let s_b = df.get(u, v, 0);
                                    row += &format!(" [{:>5.2}|{:>5.2}]", s_in.density, s_b.density);
                                }
                                let _ = writeln!(f, "  x/y={}:{}", v, row);
                            }
                        }
                        if let Some(df) = df {
                            let total = df.samples.len();
                            let mut solid = 0u32;
                            let mut air = 0u32;
                            let mut min_d = f32::INFINITY;
                            let mut max_d = f32::NEG_INFINITY;
                            let mut mat_counts: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
                            for sm in &df.samples {
                                if sm.density > 0.0 { solid += 1; } else { air += 1; }
                                if sm.density < min_d { min_d = sm.density; }
                                if sm.density > max_d { max_d = sm.density; }
                                *mat_counts.entry(format!("{:?}", sm.material)).or_insert(0) += 1;
                            }
                            let _ = writeln!(f, "density: total={} solid={} air={} min={:.3} max={:.3}",
                                total, solid, air, min_d, max_d);
                            let _ = writeln!(f, "materials: {:?}", mat_counts);
                        } else {
                            let _ = writeln!(f, "(no density field in store)");
                        }

                        // Mesh stats
                        let _ = writeln!(f, "mesh: vertices={} triangles={}",
                            mesh.vertices.len(), mesh.triangles.len());
                        let mut vmin = glam::Vec3::splat(f32::INFINITY);
                        let mut vmax = glam::Vec3::splat(f32::NEG_INFINITY);
                        for v in &mesh.vertices {
                            vmin = vmin.min(v.position);
                            vmax = vmax.max(v.position);
                        }
                        let _ = writeln!(f, "mesh AABB: min=({:.2},{:.2},{:.2}) max=({:.2},{:.2},{:.2})",
                            vmin.x, vmin.y, vmin.z, vmax.x, vmax.y, vmax.z);

                        // Coplanar tri counts: bin all triangles by which axis-aligned
                        // plane they're on. A flat planar wall = many tris in one bin.
                        let eps = 0.05_f32;
                        let mut x_bins: std::collections::HashMap<i32, u32> = std::collections::HashMap::new();
                        let mut y_bins: std::collections::HashMap<i32, u32> = std::collections::HashMap::new();
                        let mut z_bins: std::collections::HashMap<i32, u32> = std::collections::HashMap::new();
                        for tri in &mesh.triangles {
                            let p0 = mesh.vertices[tri.indices[0] as usize].position;
                            let p1 = mesh.vertices[tri.indices[1] as usize].position;
                            let p2 = mesh.vertices[tri.indices[2] as usize].position;
                            if (p0.x - p1.x).abs() < eps && (p1.x - p2.x).abs() < eps {
                                *x_bins.entry((p0.x * 2.0).round() as i32).or_insert(0) += 1;
                            }
                            if (p0.y - p1.y).abs() < eps && (p1.y - p2.y).abs() < eps {
                                *y_bins.entry((p0.y * 2.0).round() as i32).or_insert(0) += 1;
                            }
                            if (p0.z - p1.z).abs() < eps && (p1.z - p2.z).abs() < eps {
                                *z_bins.entry((p0.z * 2.0).round() as i32).or_insert(0) += 1;
                            }
                        }
                        let mut x_top: Vec<_> = x_bins.iter().collect();
                        x_top.sort_by_key(|(_, &c)| std::cmp::Reverse(c));
                        let mut y_top: Vec<_> = y_bins.iter().collect();
                        y_top.sort_by_key(|(_, &c)| std::cmp::Reverse(c));
                        let mut z_top: Vec<_> = z_bins.iter().collect();
                        z_top.sort_by_key(|(_, &c)| std::cmp::Reverse(c));
                        let _ = writeln!(f, "coplanar tris top X-planes (x*2 → count): {:?}",
                            x_top.iter().take(5).collect::<Vec<_>>());
                        let _ = writeln!(f, "coplanar tris top Y-planes: {:?}",
                            y_top.iter().take(5).collect::<Vec<_>>());
                        let _ = writeln!(f, "coplanar tris top Z-planes: {:?}",
                            z_top.iter().take(5).collect::<Vec<_>>());
                        let _ = writeln!(f, "boundary edges: {}", boundary_edges.len());
                        let _ = writeln!(f, "dc_vertices: {} (NaN-count={})",
                            dc_vertices.len(),
                            dc_vertices.iter().filter(|v| v.x.is_nan()).count());
                    }
                }
            }

            // Cache seam data and base mesh for this chunk
            {
                let mut s = store.write().unwrap();
                s.add_seam_data(
                    chunk,
                    ChunkSeamData {
                        dc_vertices,
                        world_origin: glam::Vec3::ZERO,
                        boundary_edges,
                    },
                );
                s.base_meshes.insert(chunk, mesh.clone());
            }

            // Extract density values and send to fluid thread
            {
                let s = store.read().unwrap();
                if let Some(density) = s.density_fields.get(&chunk) {
                    let densities: Vec<f32> = density.samples.iter().map(|s| s.density).collect();
                    let _ = fluid_event_tx.send(FluidEvent::DensityUpdate {
                        chunk,
                        densities,
                    });
                    if cfg.fluid_sources_enabled {
                        let _ = fluid_event_tx.send(FluidEvent::PlaceSources { chunk });
                    }

                    if cfg.fluid_sources_enabled {
                        // Detect geological springs (spring lines + vadose drips)
                        let mut geo_springs: Vec<(u8, u8, u8, f32, u8)> = Vec::new();
                        {
                            let springs = voxel_gen::springs::detect_spring_lines(
                                density, chunk, cfg.chunk_size,
                                &cfg.water_table, &cfg.ore.host_rock, cfg.seed,
                            );
                            for s in &springs {
                                geo_springs.push((s.lx, s.ly, s.lz, s.level, spring_type_to_fluid_u8(&s.source_type)));
                            }
                            let drips = voxel_gen::springs::detect_vadose_drips(
                                density, chunk, cfg.chunk_size,
                                &cfg.water_table, cfg.seed,
                            );
                            for d in &drips {
                                geo_springs.push((d.lx, d.ly, d.lz, d.level, spring_type_to_fluid_u8(&d.source_type)));
                            }
                            // Hydrothermal springs near heat sources
                            let hydro = voxel_gen::springs::detect_hydrothermal_springs(
                                density, chunk, cfg.chunk_size,
                                &cfg.water_table, &cfg.hydrothermal, cfg.seed,
                            );
                            for h in &hydro {
                                // Hydrothermal springs use SpringLine type but render as amber
                                geo_springs.push((h.lx, h.ly, h.lz, h.level, 8)); // WaterHydrothermal
                            }
                            // Artesian springs from confined aquifer
                            let artesian = voxel_gen::springs::detect_artesian_springs(
                                density, chunk, cfg.chunk_size,
                                &cfg.artesian, cfg.seed,
                            );
                            for a in &artesian {
                                geo_springs.push((a.lx, a.ly, a.lz, a.level, spring_type_to_fluid_u8(&a.source_type)));
                            }
                            // River springs from carve_rivers (collected during region gen)
                            for (rs_chunk, rs_desc) in &region_river_springs {
                                if *rs_chunk == chunk {
                                    geo_springs.push((rs_desc.lx, rs_desc.ly, rs_desc.lz, rs_desc.level, spring_type_to_fluid_u8(&rs_desc.source_type)));
                                }
                            }
                        }
                        if !geo_springs.is_empty() {
                            let _ = fluid_event_tx.send(FluidEvent::PlaceGeologicalSprings {
                                chunk,
                                springs: geo_springs,
                            });
                        }

                        // Detect pipe lava sources near kimberlite
                        let pipe_lava = voxel_gen::springs::detect_pipe_lava(
                            density, chunk, cfg.chunk_size,
                            &cfg.pipe_lava, cfg.seed,
                        );
                        for lv in &pipe_lava {
                            let _ = fluid_event_tx.send(FluidEvent::AddFluid {
                                chunk,
                                x: lv.lx,
                                y: lv.ly,
                                z: lv.lz,
                                fluid_type: voxel_fluid::cell::FluidType::Lava,
                                level: lv.level,
                                is_source: true,
                                max_flow_dist: 0, // legacy procedural source — unbounded
                            });
                        }
                    }
                }
            }

            // Inject pool fluid seeds into the fluid simulation
            // When fluid_sources_enabled is off, only inject cauldron seeds (is_source == false)
            if was_slow_path {
                for seed in &pool_fluid_seeds {
                    if !cfg.fluid_sources_enabled && seed.is_source {
                        continue; // skip infinite pool sources when toggle is off
                    }
                    let _ = fluid_event_tx.send(FluidEvent::AddFluid {
                        chunk: seed.chunk,
                        x: seed.lx,
                        y: seed.ly,
                        z: seed.lz,
                        fluid_type: match seed.fluid_type {
                            voxel_gen::pools::PoolFluid::Water => voxel_fluid::cell::FluidType::WaterPool,
                            voxel_gen::pools::PoolFluid::Lava => voxel_fluid::cell::FluidType::Lava,
                        },
                        level: 1.0,
                        is_source: seed.is_source,
                        max_flow_dist: 0, // legacy procedural source — unbounded
                    });
                }
            }

            // Compute crystal placements and store them
            let crystal_data = {
                let placements_opt = {
                    let s = store.read().unwrap();
                    s.density_fields.get(&chunk).map(|density| {
                        let coord = voxel_core::chunk::ChunkCoord::new(chunk.0, chunk.1, chunk.2);
                        voxel_gen::compute_crystals(coord, density, &cfg)
                    })
                };
                if let Some(placements) = placements_opt {
                    if !placements.is_empty() {
                        eprintln!("[CRYSTAL] Chunk {:?}: {} placements (first ore_type={}, scale={:.2})",
                            chunk, placements.len(),
                            placements[0].ore_type, placements[0].scale);
                    }
                    let ue_crystals = crate::convert::convert_crystals_to_ue(&placements, cfg.voxel_scale(), world_scale);
                    let mut sw = store.write().unwrap();
                    sw.crystal_placements.insert(chunk, placements);
                    ue_crystals
                } else {
                    Vec::new()
                }
            };

            // Compute mushroom placements and store them (mirrors crystal flow).
            let mushroom_data = {
                let placements_opt = {
                    let s = store.read().unwrap();
                    s.density_fields.get(&chunk).map(|density| {
                        let coord = voxel_core::chunk::ChunkCoord::new(chunk.0, chunk.1, chunk.2);
                        voxel_gen::compute_mushrooms(coord, density, &cfg)
                    })
                };
                if let Some(placements) = placements_opt {
                    let ue_mushrooms = crate::convert::convert_mushrooms_to_ue(&placements, cfg.voxel_scale(), world_scale);
                    let mut sw = store.write().unwrap();
                    sw.mushroom_placements.insert(chunk, placements);
                    ue_mushrooms
                } else {
                    Vec::new()
                }
            };

            // Gate 1: replace mesh with hardcoded test cube
            #[cfg(feature = "diag-gate-1")]
            let mesh = crate::convert::diagnostic_test_cube();

            // Convert to UE coordinates and send initial result (no seams yet)
            let t_coord_start = Instant::now();
            let mut converted = convert_mesh_to_ue_scaled(&mesh, cfg.voxel_scale(), world_scale);
            crate::convert::bucket_mesh_by_material(&mut converted);
            let t_coord_transform = if profiling { t_coord_start.elapsed() } else { Duration::ZERO };

            if !converted.positions.is_empty() && converted.indices.is_empty() {
                eprintln!("[WARN] Chunk {:?}: {} vertices but 0 indices (all triangles filtered)",
                    chunk, converted.positions.len());
            }

            // Capture mesh complexity before sending
            let vertex_count = converted.positions.len() as u32;
            let triangle_count = (converted.indices.len() / 3) as u32;
            // Count unique material sections
            let section_count = {
                let mut mats: Vec<u8> = converted.material_ids.iter().copied().collect();
                mats.sort_unstable();
                mats.dedup();
                mats.len() as u32
            };
            let mesh_bytes = (converted.positions.len() * std::mem::size_of::<crate::types::FfiVec3>()
                + converted.normals.len() * std::mem::size_of::<crate::types::FfiVec3>()
                + converted.material_ids.len()
                + converted.indices.len() * std::mem::size_of::<u32>()) as u32;

            // Capture per-submesh details for profiler before sending (converted is moved)
            let submesh_details: Vec<(u8, u32, u32)> = if profiling {
                converted.submeshes.iter().map(|s| (s.material_id, s.vertex_count, s.index_count)).collect()
            } else {
                Vec::new()
            };

            // Record the hash of the base-only mesh we're about to send. Without
            // this, a cross-worker race can cause UE to end up with base-only:
            //   W2 sends combined_A first (from its own seam pass as neighbor),
            //   records hash_combined. W1 sends base-only A afterwards (this
            //   line). W1's incremental_seam_pass computes combined_A (same
            //   content as W2's) and hash-matches W2's recorded hash → skips.
            //   UE channel order: combined, base-only, (nothing). UE ends on
            //   base-only. Recording hash_base here means W1's subsequent
            //   incremental sees hash_base (different from combined) and sends.
            //
            // Hash is computed BEFORE acquiring the write lock — hashing a base
            // mesh iterates every triangle, and holding the write lock during
            // that iteration serializes all other workers waiting for store access.
            let base_hash = hash_mesh(&mesh);
            {
                let mut s = store.write().unwrap();
                s.last_sent_mesh_hash.insert(chunk, base_hash);
            }

            // Debug throttle: GLOBAL mutex so all 8 workers serialize through
            // a single send queue. Each chunk waits its turn → real one-at-a-time
            // streaming the user can observe. File-controlled delay (no rebuild).
            {
                use std::sync::Mutex;
                static THROTTLE: Mutex<()> = Mutex::new(());
                let delay_ms = std::fs::read_to_string(
                    "D:/Unreal Projects/Mithril2026/Saved/voxel_stream_delay_ms.txt"
                ).ok().and_then(|s| s.trim().parse::<u64>().ok()).unwrap_or(0);
                if delay_ms > 0 {
                    let _guard = THROTTLE.lock().unwrap_or_else(|p| p.into_inner());
                    use std::io::Write;
                    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true)
                        .open("D:/Unreal Projects/Mithril2026/Saved/voxel_stream_log.txt")
                    {
                        const WS: f32 = 40.0;
                        let ux = chunk.0 as f32 * cfg.chunk_size as f32 * WS;
                        let uy = -(chunk.2 as f32) * cfg.chunk_size as f32 * WS;
                        let uz = chunk.1 as f32 * cfg.chunk_size as f32 * WS;
                        let _ = writeln!(f,
                            "[STREAM] sending chunk_rust=({},{},{}) chunk_ue=({:.0},{:.0},{:.0}) delay_ms={}",
                            chunk.0, chunk.1, chunk.2, ux, uy, uz, delay_ms
                        );
                    }
                    std::thread::sleep(std::time::Duration::from_millis(delay_ms));
                }
            }

            let t_send_start = Instant::now();
            let _ = result_tx.send(WorkerResult::ChunkMesh {
                chunk,
                mesh: converted,
                generation,
                crystal_data,
                mushroom_data,
                zone_descriptors: std::mem::take(&mut region_zone_descriptors),
            });
            let t_send_block = if profiling { t_send_start.elapsed() } else { Duration::ZERO };

            // Try to generate seams for this chunk and its neighbors
            // Gate 3: skip seam pass entirely
            #[cfg(feature = "diag-gate-3")]
            let seam_timings = SeamPassTimings {
                total: Duration::ZERO,
                quad_gen: Duration::ZERO,
                mesh_retrieve: Duration::ZERO,
                convert: Duration::ZERO,
                candidates_tried: 0,
                candidates_sent: 0,
            };
            #[cfg(not(feature = "diag-gate-3"))]
            let seam_timings = incremental_seam_pass(chunk, &cfg, store, result_tx, world_scale);
            let t_seam_pass = if profiling { seam_timings.total } else { Duration::ZERO };

            // ── Per-snapshot smart re-seam ────────────────────────────────
            //
            // When chunks in this region had their pending snapshots applied
            // during region regen, every backward face neighbor of those
            // chunks may have a STALE last_sent_mesh_hash recorded from when
            // it was originally meshed without these chunks loaded. The
            // hash skip in incremental_seam_pass / batched_seam_pass then
            // suppresses the new combined meshes that include the seams to
            // the freshly-restored chunks — leading to chunks that show
            // their base mesh in UE but never receive the seam quads
            // pointing toward the restored neighbor.
            //
            // Force-clear the hash for every snapshot-applied chunk and
            // its 3 backward face neighbors (-X/-Y/-Z), then run a
            // batched_seam_pass over the union. batched_seam_pass expands
            // the dirty set into its 27-neighborhood, so this also catches
            // diagonal/edge neighbors. Hash-skip would have re-suppressed
            // any genuine no-op so we have to remove the hashes BEFORE
            // running the pass — relying on hash-skip after force-clear
            // gives us cheap dedup for chunks whose seams genuinely
            // didn't change.
            //
            // Cost: a batched_seam_pass over <region_size>³ chunks is
            // single-digit ms in the worker thread; the game thread sees
            // small mesh updates trickle through. See
            // [streaming-perf.md] notes — these are existing-actor updates
            // and bypass per-tick spawn budgets.
            if !applied_snapshots.is_empty() {
                use std::collections::HashSet;
                let bwd_offsets: [(i32, i32, i32); 3] = [(-1, 0, 0), (0, -1, 0), (0, 0, -1)];
                let mut targets: HashSet<(i32, i32, i32)> = HashSet::new();
                for &k in &applied_snapshots {
                    targets.insert(k);
                    for &(dx, dy, dz) in &bwd_offsets {
                        targets.insert((k.0 + dx, k.1 + dy, k.2 + dz));
                    }
                }
                {
                    let mut s = store.write().unwrap();
                    for &t in &targets {
                        s.last_sent_mesh_hash.remove(&t);
                    }
                }
                let target_vec: Vec<(i32, i32, i32)> = targets.into_iter().collect();
                batched_seam_pass(&target_vec, &cfg, store, result_tx, fluid_event_tx, world_scale);
            }

            // ── STALE-SEAM_DATA fix: ship the regenerated meshes ──
            //
            // The post-sync DC re-solve in the region-regen block freshened
            // base_mesh + seam_data for every sync-dirty chunk that already
            // had a mesh. Now we have to:
            //   (a) drop their last_sent_mesh_hash so the upcoming
            //       batched_seam_pass can't be hash-skipped, and
            //   (b) run batched_seam_pass over the union, which will fan
            //       out into their 27-neighborhood and emit every combined
            //       mesh whose dc_vertices are now valid.
            //
            // Without this, the regenerated seam_data sits in the store
            // unused — UE keeps showing the stale combined or base-only.
            if !sync_remeshed.is_empty() {
                {
                    let mut s = store.write().unwrap();
                    for &k in &sync_remeshed {
                        s.last_sent_mesh_hash.remove(&k);
                        // Also drop hashes of the 3 backward face neighbors —
                        // their previously-sent combined included this chunk's
                        // stale dc_vertices, so it's also out of date.
                        let bwd: [(i32,i32,i32); 3] = [(-1,0,0),(0,-1,0),(0,0,-1)];
                        for &(dx,dy,dz) in &bwd {
                            s.last_sent_mesh_hash.remove(&(k.0+dx, k.1+dy, k.2+dz));
                        }
                    }
                }
                batched_seam_pass(&sync_remeshed, &cfg, store, result_tx, fluid_event_tx, world_scale);
            }

            // Write pipeline timing report
            {
                let total = chunk_start.elapsed();
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true)
                    .open("D:/Unreal Projects/Mithril2026/Saved/gen_perf.txt")
                {
                    let rt = &region_timings;
                    let _ = writeln!(f, "chunk({},{},{}) total={:.1}ms slow={} | region_total={:.1} [base_density={:.1} caverns={:.1} worm_plan={:.1} worm_carve={:.1} zones={:.1} pools={:.1} formations={:.1} boundary={:.1} metadata={:.1} worms={}] hermite={:.1} dc={:.1} mesh={:.1} smooth={:.1} seam={:.1} worm_fwd={:.1} bwd_carve={:.1} bwd_remesh={:.1} bwd_dirty={}",
                        chunk.0, chunk.1, chunk.2,
                        total.as_secs_f64() * 1000.0,
                        was_slow_path,
                        t_region_density.as_secs_f64() * 1000.0,
                        rt.base_density.as_secs_f64() * 1000.0,
                        rt.cavern_centers.as_secs_f64() * 1000.0,
                        rt.worm_planning.as_secs_f64() * 1000.0,
                        rt.worm_carving.as_secs_f64() * 1000.0,
                        rt.zones.as_secs_f64() * 1000.0,
                        rt.pools.as_secs_f64() * 1000.0,
                        rt.formations.as_secs_f64() * 1000.0,
                        rt.boundary_sync.as_secs_f64() * 1000.0,
                        rt.metadata.as_secs_f64() * 1000.0,
                        rt.worm_count,
                        t_hermite.as_secs_f64() * 1000.0,
                        t_dc_solve.as_secs_f64() * 1000.0,
                        t_mesh_gen.as_secs_f64() * 1000.0,
                        t_mesh_smooth.as_secs_f64() * 1000.0,
                        t_seam_pass.as_secs_f64() * 1000.0,
                        t_worm_forward_sharing.as_secs_f64() * 1000.0,
                        t_worm_backward_carve.as_secs_f64() * 1000.0,
                        t_worm_backward_remesh.as_secs_f64() * 1000.0,
                        backward_dirty_count,
                    );
                }
            }

            // Record profiling data
            if profiling {
                let gen_queue_len = generate_rx.len() as u32;
                let result_queue_len = 0u32; // Sender doesn't expose queue length

                let timings = ChunkTimings {
                    region_density: t_region_density,
                    hermite: t_hermite,
                    dc_solve: t_dc_solve,
                    mesh_gen: t_mesh_gen,
                    mesh_smooth: t_mesh_smooth,
                    seam_pass: t_seam_pass,
                    coord_transform: t_coord_transform,
                    store_read_wait: t_store_read_wait,
                    store_write_wait: t_store_write_wait,
                    total: chunk_start.elapsed(),
                    was_slow_path,
                    vertex_count,
                    triangle_count,
                    section_count,
                    mesh_bytes,
                    seam_quad_gen: seam_timings.quad_gen,
                    seam_mesh_retrieve: seam_timings.mesh_retrieve,
                    seam_convert: seam_timings.convert,
                    seam_candidates_tried: seam_timings.candidates_tried,
                    seam_candidates_sent: seam_timings.candidates_sent,
                    send_block: t_send_block,
                    coarse_skip: vertex_count == 0 && triangle_count == 0,
                    worm_base_density: region_timings.base_density,
                    worm_cavern_centers: region_timings.cavern_centers,
                    worm_planning: region_timings.worm_planning,
                    worm_carving: region_timings.worm_carving,
                    worm_pools: region_timings.pools,
                    worm_formations: region_timings.formations,
                    worm_forward_sharing: t_worm_forward_sharing,
                    worm_backward_carve: t_worm_backward_carve,
                    worm_backward_remesh: t_worm_backward_remesh,
                    worm_count: region_timings.worm_count,
                    worm_segment_count: region_timings.worm_segment_count,
                    backward_dirty_count,
                    stress_update: Duration::ZERO,
                    collapse_detect: Duration::ZERO,
                    collapse_remesh: Duration::ZERO,
                };
                profiler.record_chunk_with_coord(
                    worker_id, chunk, timings, gen_queue_len, result_queue_len,
                );
                if !submesh_details.is_empty() {
                    profiler.attach_submesh_info(submesh_details);
                }
            }
        }
        WorkerRequest::Flatten { base_x, base_y, base_z, host_material } => {
            let cfg = config.read().unwrap().clone();
            let mat = voxel_core::material::Material::from_u8(host_material);
            let mut s = store.write().unwrap();
            let ts = terrace_size_for_scale(world_scale);
            // DK2-style zone flatten: integer Y is fine (no per-tile float).
            let meshes = crate::terrain_ops::flatten_terrace(&mut s, glam::IVec3::new(base_x, base_y, base_z), base_y as f32, mat, &cfg, world_scale, ts, 2);
            let dirty_keys: Vec<(i32, i32, i32)> = meshes.into_iter().map(|(k, _)| k).collect();
            drop(s);

            recompute_crystals_for_chunks(store, &cfg, &dirty_keys);
            prune_destroyed_mushrooms_for_chunks(store, &dirty_keys);
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
        }
        WorkerRequest::FlattenBatch { tiles } => {
            let cfg = config.read().unwrap().clone();
            let mut s = store.write().unwrap();
            let ts = terrace_size_for_scale(world_scale);
            let meshes = crate::terrain_ops::flatten_terrace_batch(&mut s, &tiles, &cfg, world_scale, ts);
            let dirty_keys: Vec<_> = meshes.into_iter().map(|(k, _)| k).collect();
            drop(s);
            recompute_crystals_for_chunks(store, &cfg, &dirty_keys);
            prune_destroyed_mushrooms_for_chunks(store, &dirty_keys);
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
        }
        WorkerRequest::BuildingFlattenBatch { buildings } => {
            // Cheap path: per-tile call to the legacy ramp flatten. Each tile
            // carries its own sub-voxel base_y_float so conveyors don't
            // float/sink (3A density tweak applied inside apply_ramp_column).
            let cfg = config.read().unwrap().clone();
            let mut s = store.write().unwrap();
            let mut all_dirty: Vec<(i32, i32, i32)> = Vec::new();
            for &(bx, by, bz, by_f, host_mat, footprint, clearance) in &buildings {
                let mat = voxel_core::material::Material::from_u8(host_mat);
                let bts = footprint.max(1);
                let meshes = crate::terrain_ops::flatten_terrace(
                    &mut s,
                    glam::IVec3::new(bx, by, bz),
                    by_f,
                    mat,
                    &cfg,
                    world_scale,
                    bts,
                    clearance.max(2),
                );
                all_dirty.extend(meshes.into_iter().map(|(k, _)| k));
            }
            // Deduplicate dirty keys
            all_dirty.sort();
            all_dirty.dedup();
            drop(s);
            // Single seam pass for all flattens combined
            recompute_crystals_for_chunks(store, &cfg, &all_dirty);
            prune_destroyed_mushrooms_for_chunks(store, &all_dirty);
            batched_seam_pass_mine(&all_dirty, &cfg, store, result_tx, fluid_event_tx, world_scale);
        }
        WorkerRequest::BuildingFlatten { base_x, base_y, base_z, base_y_float, host_material, footprint_voxels, clearance_voxels } => {
            let cfg = config.read().unwrap().clone();
            let mat = voxel_core::material::Material::from_u8(host_material);
            let mut s = store.write().unwrap();
            let bts = footprint_voxels.max(1);

            // Single placement: route to the all-in SDF flatten (1C+3C+2C).
            // base_y_float carries the exact sub-voxel Y so the iso surface
            // lands exactly where UE wants it (no float/sink). Cantilever
            // columns get convex-hull buttresses to nearby natural rock.
            let meshes = crate::flatten_sdf::flatten_terrace_sdf(
                &mut s,
                glam::IVec3::new(base_x, base_y, base_z),
                base_y_float,
                mat,
                &cfg,
                world_scale,
                bts,
                clearance_voxels,
            );
            let dirty_keys: Vec<(i32, i32, i32)> = meshes.into_iter().map(|(k, _)| k).collect();
            drop(s);

            recompute_crystals_for_chunks(store, &cfg, &dirty_keys);
            prune_destroyed_mushrooms_for_chunks(store, &dirty_keys);
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
        }
        WorkerRequest::Mine { request } => {
            let cfg = config.read().unwrap().clone();

            let center = from_ue_world_pos(
                request.world_x, request.world_y, request.world_z, world_scale,
            );
            let radius = request.radius / world_scale;

            let mut s = store.write().unwrap();
            let store_chunks_count = s.density_fields.len();
            let outcome = if request.mode == 0 {
                crate::mining::mine_sphere(&mut s, center, radius, &cfg, world_scale)
            } else {
                let normal = from_ue_normal(
                    request.normal_x, request.normal_y, request.normal_z,
                );
                crate::mining::mine_peel(&mut s, center, normal, radius, &cfg, world_scale)
            };
            let dirty_count = outcome.meshes.len();
            let mined = outcome.mined;
            let flipped_chunks = outcome.flipped_chunks;
            let meshes = outcome.meshes;
            drop(s);

            // Perf: all mine_debug.txt I/O is now after drop(s), consolidated into
            // ONE file open (was 3 opens, 2 of them under the write lock — blocking
            // every other worker for ~200-800μs on a contended NTFS handle cache).
            {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true)
                    .open("D:/Unreal Projects/Mithril2026/Saved/mine_debug.txt")
                {
                    let _ = writeln!(f, "[MINE] request: ({},{},{}) r={} mode={} | rust coords: ({:.1},{:.1},{:.1}) r={:.1}, store chunks={} | complete: {} dirty chunks",
                        request.world_x, request.world_y, request.world_z, request.radius, request.mode,
                        center.x, center.y, center.z, radius, store_chunks_count,
                        dirty_count);
                }
            }

            // Collect dirty chunk keys — don't send meshes yet (seam pass will send
            // them with seam quads included, avoiding a seamless→seamed flash)
            let dirty_keys: Vec<(i32, i32, i32)> = meshes.into_iter().map(|(k, _)| k).collect();

            // Recompute crystal placements for dirty chunks so batched_seam_pass
            // picks up the updated data via retrieve_crystal_data.
            // Single read lock for all computes, single write lock for all inserts —
            // replaces N× per-key read+write acquisition (~2-5ms saved per mine with
            // many dirty chunks under contended locking).
            // Fix B: iterate only flipped_chunks (material actually changed).
            // Boundary-sync extras in dirty_keys have density tweaks only, no new
            // material placement, so their crystal layout is unchanged.
            let new_placements: Vec<_> = {
                let s = store.read().unwrap();
                flipped_chunks.iter().filter_map(|&key| {
                    s.density_fields.get(&key).map(|density| {
                        let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                        (key, voxel_gen::compute_crystals(coord, density, &cfg))
                    })
                }).collect()
            };
            // Merge crystal insert + stress-dirty queue into a single write lock.
            // Previously these were two separate write lock acquisitions sandwiching
            // a channel send — every mine paid for two lock round-trips on the
            // contended store. Channel send moved after the write lock; ordering
            // is independent (stress + crystal writes don't depend on the send).
            {
                let mut s = store.write().unwrap();
                for (key, placements) in new_placements {
                    s.crystal_placements.insert(key, placements);
                }
                let stress_center = (center.x as i32, center.y as i32, center.z as i32);
                let stress_radius = radius as i32 + 22;
                s.queue_stress_dirty(stress_center, stress_radius);
            }

            // Mushroom destruction: drop any whose anchor voxel was mined.
            // Uses dirty_keys (all chunks whose density actually changed in this op).
            prune_destroyed_mushrooms_for_chunks(store, &dirty_keys);

            // Send mined material counts (outside the store lock)
            let _ = result_tx.send(WorkerResult::MinedMaterials { mined });

            // Send terrain modifications to fluid thread + detect aquifer breaches
            {
                // Approximate mined cells: cells near the mine center in each dirty chunk
                let mine_cx = (center.x / cfg.chunk_size as f32).floor() as i32;
                let mine_cy = (center.y / cfg.chunk_size as f32).floor() as i32;
                let mine_cz = (center.z / cfg.chunk_size as f32).floor() as i32;
                let mine_lx = ((center.x - mine_cx as f32 * cfg.chunk_size as f32) as usize).min(cfg.chunk_size - 1);
                let mine_ly = ((center.y - mine_cy as f32 * cfg.chunk_size as f32) as usize).min(cfg.chunk_size - 1);
                let mine_lz = ((center.z - mine_cz as f32 * cfg.chunk_size as f32) as usize).min(cfg.chunk_size - 1);
                let mined_cells = vec![(mine_lx, mine_ly, mine_lz)];

                let s = store.read().unwrap();
                for &key in &dirty_keys {
                    if let Some(density) = s.density_fields.get(&key) {
                        let densities: Vec<f32> = density.samples.iter().map(|s| s.density).collect();
                        let _ = fluid_event_tx.send(FluidEvent::TerrainModified {
                            chunk: key,
                            densities,
                        });

                        // Detect aquifer breaches near the mined area
                        if cfg.fluid_sources_enabled {
                            let breaches = voxel_gen::springs::detect_aquifer_breaches(
                                density, key, cfg.chunk_size,
                                &cfg.water_table, &cfg.ore.host_rock, cfg.seed,
                                &mined_cells,
                            );
                            for b in &breaches {
                                let _ = fluid_event_tx.send(FluidEvent::AddFluid {
                                    chunk: key,
                                    x: b.lx,
                                    y: b.ly,
                                    z: b.lz,
                                    fluid_type: voxel_fluid::cell::FluidType::WaterBreach,
                                    level: b.level,
                                    is_source: true,
                                    max_flow_dist: 0, // procedural breach — unbounded
                                });
                            }
                        }
                    }
                }
            }

            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
        }
        WorkerRequest::MineAndFillFluid { world_x, world_y, world_z, radius, fluid_type, world_scale: ws } => {
            let cfg = config.read().unwrap().clone();

            // Convert UE world position to Rust coordinates
            let center = from_ue_world_pos(world_x, world_y, world_z, ws);
            let rust_radius = radius / ws;

            // Step 1: Mine the sphere (same as normal pick)
            let mut s = store.write().unwrap();
            let outcome = crate::mining::mine_sphere(&mut s, center, rust_radius, &cfg, ws);
            drop(s);
            let meshes = outcome.meshes;
            let mined = outcome.mined;

            let dirty_keys: Vec<(i32, i32, i32)> = meshes.into_iter().map(|(k, _)| k).collect();

            // Step 2: Fill bottom half with non-source fluid
            {
                let s = store.read().unwrap();
                let eb = cfg.effective_bounds();
                let vs = cfg.voxel_scale();
                let r2 = rust_radius * rust_radius;
                let ft = voxel_fluid::cell::FluidType::from_u8(fluid_type);

                let min_cx = ((center.x - rust_radius) / eb).floor() as i32;
                let max_cx = ((center.x + rust_radius) / eb).floor() as i32;
                let min_cy = ((center.y - rust_radius) / eb).floor() as i32;
                let max_cy = ((center.y + rust_radius) / eb).floor() as i32;
                let min_cz = ((center.z - rust_radius) / eb).floor() as i32;
                let max_cz = ((center.z + rust_radius) / eb).floor() as i32;

                for cz in min_cz..=max_cz {
                    for cy in min_cy..=max_cy {
                        for cx in min_cx..=max_cx {
                            let key = (cx, cy, cz);
                            if let Some(density) = s.density_fields.get(&key) {
                                let origin = glam::Vec3::new(cx as f32 * eb, cy as f32 * eb, cz as f32 * eb);
                                let grid_center = (center - origin) / vs;
                                let grid_radius = rust_radius / vs;
                                let lo_x = ((grid_center.x - grid_radius).floor() as i32).max(0) as usize;
                                let hi_x = ((grid_center.x + grid_radius).ceil() as usize + 1).min(density.size);
                                let lo_y = ((grid_center.y - grid_radius).floor() as i32).max(0) as usize;
                                let hi_y = ((grid_center.y + grid_radius).ceil() as usize + 1).min(density.size);
                                let lo_z = ((grid_center.z - grid_radius).floor() as i32).max(0) as usize;
                                let hi_z = ((grid_center.z + grid_radius).ceil() as usize + 1).min(density.size);

                                for z in lo_z..hi_z {
                                    for y in lo_y..hi_y {
                                        for x in lo_x..hi_x {
                                            let world_pos = origin + glam::Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                            let dist2 = (world_pos - center).length_squared();
                                            // Bottom half of sphere (Rust Y-up: below center) and cell is air
                                            if dist2 <= r2 && world_pos.y < center.y && density.get(x, y, z).density <= 0.0 {
                                                eprintln!("[FLUID_FILL]   placing fluid at chunk({},{},{}) local({},{},{}) world({:.0},{:.0},{:.0})",
                                                    key.0, key.1, key.2, x, y, z, world_pos.x, world_pos.y, world_pos.z);
                                                let _ = fluid_event_tx.send(FluidEvent::AddFluid {
                                                    chunk: key,
                                                    x: x as u8,
                                                    y: y as u8,
                                                    z: z as u8,
                                                    fluid_type: ft,
                                                    level: 1.0,
                                                    is_source: false,
                                                    max_flow_dist: 0, // non-source — irrelevant
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Step 3: Store updated crystal data (seam pass will pick it up)
            for &key in &dirty_keys {
                let s = store.read().unwrap();
                if let Some(density) = s.density_fields.get(&key) {
                    let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                    let placements = voxel_gen::compute_crystals(coord, density, &cfg);
                    drop(s);
                    store.write().unwrap().crystal_placements.insert(key, placements);
                } else {
                    drop(s);
                }
            }

            // Step 4: Send mined material counts
            let _ = result_tx.send(WorkerResult::MinedMaterials { mined });

            // Step 5: Send terrain modifications to fluid thread + detect aquifer breaches
            {
                let mine_cx = (center.x / cfg.chunk_size as f32).floor() as i32;
                let mine_cy = (center.y / cfg.chunk_size as f32).floor() as i32;
                let mine_cz = (center.z / cfg.chunk_size as f32).floor() as i32;
                let mine_lx = ((center.x - mine_cx as f32 * cfg.chunk_size as f32) as usize).min(cfg.chunk_size - 1);
                let mine_ly = ((center.y - mine_cy as f32 * cfg.chunk_size as f32) as usize).min(cfg.chunk_size - 1);
                let mine_lz = ((center.z - mine_cz as f32 * cfg.chunk_size as f32) as usize).min(cfg.chunk_size - 1);
                let mined_cells = vec![(mine_lx, mine_ly, mine_lz)];

                let s = store.read().unwrap();
                for &key in &dirty_keys {
                    if let Some(density) = s.density_fields.get(&key) {
                        let densities: Vec<f32> = density.samples.iter().map(|s| s.density).collect();
                        let _ = fluid_event_tx.send(FluidEvent::TerrainModified {
                            chunk: key,
                            densities,
                        });

                        if cfg.fluid_sources_enabled {
                            let breaches = voxel_gen::springs::detect_aquifer_breaches(
                                density, key, cfg.chunk_size,
                                &cfg.water_table, &cfg.ore.host_rock, cfg.seed,
                                &mined_cells,
                            );
                            for b in &breaches {
                                let _ = fluid_event_tx.send(FluidEvent::AddFluid {
                                    chunk: key,
                                    x: b.lx,
                                    y: b.ly,
                                    z: b.lz,
                                    fluid_type: voxel_fluid::cell::FluidType::WaterBreach,
                                    level: b.level,
                                    is_source: true,
                                    max_flow_dist: 0, // procedural breach — unbounded
                                });
                            }
                        }
                    }
                }
            }

            // Step 6: Regenerate seams
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, ws);
        }
        WorkerRequest::BrushSphere { request } => {
            let cfg = config.read().unwrap().clone();
            let center = from_ue_world_pos(
                request.world_x, request.world_y, request.world_z, world_scale,
            );
            let radius = request.radius / world_scale;

            let mut s = store.write().unwrap();
            let outcome = match request.mode {
                0 => {
                    let mat = voxel_core::material::Material::from_u8(request.material);
                    crate::brushes::paint_material_sphere(
                        &mut s, center, radius, mat, &cfg, world_scale,
                    )
                }
                1 => crate::brushes::carve_sphere(&mut s, center, radius, &cfg, world_scale),
                2 => {
                    let mat = voxel_core::material::Material::from_u8(request.material);
                    crate::brushes::fill_sphere(
                        &mut s, center, radius, mat, &cfg, world_scale,
                    )
                }
                _ => crate::brushes::BrushOutcome { meshes: Vec::new(), flipped_chunks: Vec::new() },
            };
            drop(s);

            let dirty_keys: Vec<(i32, i32, i32)> =
                outcome.meshes.into_iter().map(|(k, _)| k).collect();
            // Recompute crystal placements on flipped chunks (mirrors mine path).
            let new_placements: Vec<_> = {
                let s = store.read().unwrap();
                outcome.flipped_chunks.iter().filter_map(|&key| {
                    s.density_fields.get(&key).map(|density| {
                        let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                        (key, voxel_gen::compute_crystals(coord, density, &cfg))
                    })
                }).collect()
            };
            {
                let mut s = store.write().unwrap();
                for (key, placements) in new_placements {
                    s.crystal_placements.insert(key, placements);
                }
            }
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
        }
        WorkerRequest::BrushTunnel { points, radius, material } => {
            let cfg = config.read().unwrap().clone();
            let mat = material.map(voxel_core::material::Material::from_u8);

            let mut s = store.write().unwrap();
            let outcome = crate::brushes::tunnel(
                &mut s, &points, radius, mat, &cfg, world_scale,
            );
            drop(s);

            let dirty_keys: Vec<(i32, i32, i32)> =
                outcome.meshes.into_iter().map(|(k, _)| k).collect();
            let new_placements: Vec<_> = {
                let s = store.read().unwrap();
                outcome.flipped_chunks.iter().filter_map(|&key| {
                    s.density_fields.get(&key).map(|density| {
                        let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                        (key, voxel_gen::compute_crystals(coord, density, &cfg))
                    })
                }).collect()
            };
            {
                let mut s = store.write().unwrap();
                for (key, placements) in new_placements {
                    s.crystal_placements.insert(key, placements);
                }
            }
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
        }
        WorkerRequest::BrushPlaceMushroom { center_rust, kind, search_radius, scale, yaw } => {
            let cfg = config.read().unwrap().clone();
            let mut s = store.write().unwrap();
            let placed_chunk = crate::brushes::place_mushroom_at_world(
                &mut s, center_rust, kind, search_radius, scale, yaw, &cfg,
            );
            drop(s);

            // No density change — but UE needs the new mushroom_data. Bypass
            // the hash-skip in `batched_seam_pass_inner` by clearing the
            // last-sent hash for that chunk, then trigger a seam pass on it.
            // The seam pass will re-emit the same combined mesh + the new
            // mushroom_data, and UE's ApplyMushroomData hash-skip lets the
            // crystal+mesh apply paths short-circuit.
            if let Some(chunk_key) = placed_chunk {
                {
                    let mut s = store.write().unwrap();
                    s.last_sent_mesh_hash.remove(&chunk_key);
                }
                let dirty = [chunk_key];
                batched_seam_pass_mine(&dirty, &cfg, store, result_tx, fluid_event_tx, world_scale);
            }
        }
        WorkerRequest::BrushPlaceMushroomSphere { center_rust, radius, density, kind, seed } => {
            let cfg = config.read().unwrap().clone();
            let mut s = store.write().unwrap();
            let affected = crate::brushes::place_mushrooms_brush_sphere(
                &mut s, center_rust, kind, radius, density, seed, &cfg,
            );
            drop(s);

            if !affected.is_empty() {
                {
                    let mut s = store.write().unwrap();
                    for key in &affected {
                        s.last_sent_mesh_hash.remove(key);
                    }
                }
                let dirty: Vec<(i32, i32, i32)> = affected.into_iter().collect();
                batched_seam_pass_mine(&dirty, &cfg, store, result_tx, fluid_event_tx, world_scale);
            }
        }
        WorkerRequest::BrushFormation { center_rust, formation_type, material, height, radius } => {
            let cfg = config.read().unwrap().clone();
            let mat = voxel_core::material::Material::from_u8(material);

            let mut s = store.write().unwrap();
            let outcome = crate::brushes::place_formation(
                &mut s, center_rust, formation_type, height, radius, mat, &cfg, world_scale,
            );
            drop(s);

            let dirty_keys: Vec<(i32, i32, i32)> =
                outcome.meshes.into_iter().map(|(k, _)| k).collect();
            let new_placements: Vec<_> = {
                let s = store.read().unwrap();
                outcome.flipped_chunks.iter().filter_map(|&key| {
                    s.density_fields.get(&key).map(|density| {
                        let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                        (key, voxel_gen::compute_crystals(coord, density, &cfg))
                    })
                }).collect()
            };
            {
                let mut s = store.write().unwrap();
                for (key, placements) in new_placements {
                    s.crystal_placements.insert(key, placements);
                }
            }
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
        }
        WorkerRequest::ForceChunkResync { chunk } => {
            let cfg = config.read().unwrap().clone();
            let neighbors_face: [(i32, i32, i32); 6] = [
                (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
            ];
            // Trace file so we can confirm the request reached the worker
            // and what the seam pass actually emitted. Lives next to other
            // worker logs the user already collects.
            let log_line = |line: String| {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true)
                    .open("D:/Unreal Projects/Mithril2026/Saved/voxel_force_resync.txt")
                {
                    let _ = writeln!(f, "[ForceResync] {}", line);
                }
            };
            log_line(format!("BEGIN chunk={:?}", chunk));

            // Phase 1: sync boundaries
            {
                let mut s = store.write().unwrap();
                let modified = s.sync_chunk_full_boundaries(chunk, cfg.chunk_size);
                log_line(format!("sync_chunk_full_boundaries → {} chunks modified", modified.len()));
            }

            // Phase 2: collect targets (chunk + 6 face neighbors that exist)
            let mut targets: Vec<(i32, i32, i32)> = vec![chunk];
            {
                let s = store.read().unwrap();
                for &(dx, dy, dz) in &neighbors_face {
                    let n = (chunk.0 + dx, chunk.1 + dy, chunk.2 + dz);
                    if s.density_fields.contains_key(&n) {
                        targets.push(n);
                    }
                }
            }
            log_line(format!("targets: {:?}", targets));

            // Phase 3: regenerate base mesh + hermite + seam_data for each
            // target.
            let chunk_size = cfg.chunk_size;
            for &target in &targets {
                let density_clone = {
                    let s = store.read().unwrap();
                    s.density_fields.get(&target).cloned()
                };
                let Some(density) = density_clone else { continue; };
                let hermite = extract_hermite_data(&density);
                let cell_size = density.size - 1;
                let dc_vertices = solve_dc_vertices(&hermite, cell_size);
                let mut mesh = generate_mesh(&hermite, &dc_vertices, cell_size);
                mesh.smooth(
                    cfg.mesh_smooth_iterations,
                    cfg.mesh_smooth_strength,
                    cfg.mesh_boundary_smooth,
                    Some(cell_size),
                );
                if cfg.mesh_recalc_normals > 0 { mesh.recalculate_normals(); }
                let boundary_edges = voxel_gen::region_gen::extract_boundary_edges(&hermite, chunk_size);
                let base_verts = mesh.vertices.len();
                {
                    let mut s = store.write().unwrap();
                    s.hermite_data.insert(target, hermite);
                    s.base_meshes.insert(target, mesh);
                    s.add_seam_data(target, ChunkSeamData {
                        dc_vertices,
                        world_origin: glam::Vec3::ZERO,
                        boundary_edges,
                    });
                }
                log_line(format!("regenerated target {:?}: base verts={}", target, base_verts));
            }

            // Phase 4: combine base + seam quads and send DIRECTLY as
            // ChunkMesh results. Bypasses the MineBatchMesh requeue dance
            // which seems to drop seam updates for Force Resync. Also
            // forcibly invalidates last_sent_mesh_hash so hash-skip can't
            // suppress the send (the user explicitly asked for a refresh).
            for &target in &targets {
                let combined: Option<voxel_core::mesh::Mesh> = {
                    let s = store.read().unwrap();
                    let base = match s.base_meshes.get(&target) { Some(m) => m.clone(), None => { continue; } };
                    let seam = voxel_gen::region_gen::generate_chunk_seam_quads(target, &s.chunk_seam_data, cfg.chunk_size);
                    let seam_tris = seam.triangles.len();
                    let mut combined = base;
                    combined.append(seam);
                    if cfg.mesh_recalc_normals > 0 { combined.recalculate_normals(); }
                    log_line(format!("target {:?}: base+seam → {} verts / {} tris (seam_tris={})",
                        target, combined.vertices.len(), combined.triangles.len(), seam_tris));
                    Some(combined)
                };
                let Some(combined) = combined else { continue; };

                // Force-invalidate hash so send isn't skipped.
                {
                    let mut s = store.write().unwrap();
                    s.last_sent_mesh_hash.remove(&target);
                }

                let mut converted = convert_mesh_to_ue_scaled(&combined, cfg.voxel_scale(), world_scale);
                crate::convert::bucket_mesh_by_material(&mut converted);
                let v = converted.positions.len();
                let i = converted.indices.len();
                log_line(format!("target {:?}: sending ChunkMesh — {} verts / {} indices", target, v, i));
                let _ = result_tx.send(WorkerResult::ChunkMesh {
                    chunk: target,
                    mesh: converted,
                    generation: 0,
                    crystal_data: Vec::new(),
                    mushroom_data: Vec::new(),
                    zone_descriptors: Vec::new(),
                });

                // Record new hash so subsequent passes don't redo it
                let h = hash_mesh(&combined);
                {
                    let mut s = store.write().unwrap();
                    s.last_sent_mesh_hash.insert(target, h);
                }
            }
            log_line(format!("END chunk={:?}", chunk));
        }
        WorkerRequest::BrushCavernStamp { chunk_origin, extent, decorate, fluids, seed } => {
            let cfg = config.read().unwrap().clone();

            let mut s = store.write().unwrap();
            let outcome = crate::brushes::cavern_stamp_brush(
                &mut s, chunk_origin, extent, decorate, fluids, seed as u64, &cfg, world_scale,
            );
            drop(s);

            let dirty_keys: Vec<(i32, i32, i32)> =
                outcome.meshes.into_iter().map(|(k, _)| k).collect();
            let new_placements: Vec<_> = {
                let s = store.read().unwrap();
                outcome.flipped_chunks.iter().filter_map(|&key| {
                    s.density_fields.get(&key).map(|density| {
                        let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                        (key, voxel_gen::compute_crystals(coord, density, &cfg))
                    })
                }).collect()
            };
            {
                let mut s = store.write().unwrap();
                for (key, placements) in new_placements {
                    s.crystal_placements.insert(key, placements);
                }
            }
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
        }
        WorkerRequest::BrushFormationStamp { center_rust, radius, seed } => {
            let cfg = config.read().unwrap().clone();

            let mut s = store.write().unwrap();
            let outcome = crate::brushes::random_formations_brush(
                &mut s, center_rust, radius, seed as u64, &cfg, world_scale,
            );
            drop(s);

            let dirty_keys: Vec<(i32, i32, i32)> =
                outcome.meshes.into_iter().map(|(k, _)| k).collect();
            let new_placements: Vec<_> = {
                let s = store.read().unwrap();
                outcome.flipped_chunks.iter().filter_map(|&key| {
                    s.density_fields.get(&key).map(|density| {
                        let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                        (key, voxel_gen::compute_crystals(coord, density, &cfg))
                    })
                }).collect()
            };
            {
                let mut s = store.write().unwrap();
                for (key, placements) in new_placements {
                    s.crystal_placements.insert(key, placements);
                }
            }
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
        }
        WorkerRequest::BrushBox { center_rust, half_ext_rust, yaw_rad, op, material } => {
            let cfg = config.read().unwrap().clone();
            let mat = voxel_core::material::Material::from_u8(material);

            let mut s = store.write().unwrap();
            let outcome = crate::brushes::box_brush(
                &mut s, center_rust, half_ext_rust, yaw_rad, op, mat, &cfg, world_scale,
            );
            drop(s);

            let dirty_keys: Vec<(i32, i32, i32)> =
                outcome.meshes.into_iter().map(|(k, _)| k).collect();
            let new_placements: Vec<_> = {
                let s = store.read().unwrap();
                outcome.flipped_chunks.iter().filter_map(|&key| {
                    s.density_fields.get(&key).map(|density| {
                        let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                        (key, voxel_gen::compute_crystals(coord, density, &cfg))
                    })
                }).collect()
            };
            {
                let mut s = store.write().unwrap();
                for (key, placements) in new_placements {
                    s.crystal_placements.insert(key, placements);
                }
            }
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
        }
        WorkerRequest::BrushCylinder { center_rust, radius, height, op, material } => {
            let cfg = config.read().unwrap().clone();
            let mat = voxel_core::material::Material::from_u8(material);

            let mut s = store.write().unwrap();
            let outcome = crate::brushes::cylinder_brush(
                &mut s, center_rust, radius, height, op, mat, &cfg, world_scale,
            );
            drop(s);

            let dirty_keys: Vec<(i32, i32, i32)> =
                outcome.meshes.into_iter().map(|(k, _)| k).collect();
            let new_placements: Vec<_> = {
                let s = store.read().unwrap();
                outcome.flipped_chunks.iter().filter_map(|&key| {
                    s.density_fields.get(&key).map(|density| {
                        let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                        (key, voxel_gen::compute_crystals(coord, density, &cfg))
                    })
                }).collect()
            };
            {
                let mut s = store.write().unwrap();
                for (key, placements) in new_placements {
                    s.crystal_placements.insert(key, placements);
                }
            }
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
        }
        WorkerRequest::BrushSmooth { center_rust, radius, iterations, strength } => {
            let cfg = config.read().unwrap().clone();
            let mut s = store.write().unwrap();
            let outcome = crate::brushes::smooth_brush(
                &mut s, center_rust, radius, iterations, strength, &cfg, world_scale,
            );
            drop(s);
            let dirty_keys: Vec<(i32, i32, i32)> =
                outcome.meshes.into_iter().map(|(k, _)| k).collect();
            // Smooth doesn't change material, so crystal recompute isn't strictly needed —
            // but keep consistent with other brush handlers for safety.
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
        }
        WorkerRequest::BrushOrePaint {
            center_rust,
            radius,
            cluster_size,
            min_spacing,
            channel_prob,
            channel_length,
            channel_radius,
            density,
            seed,
            weights,
        } => {
            let cfg = config.read().unwrap().clone();
            let mut s = store.write().unwrap();
            let outcome = crate::brushes::paint_ore_deposits(
                &mut s,
                center_rust,
                radius,
                weights,
                cluster_size,
                min_spacing,
                channel_prob,
                channel_length,
                channel_radius,
                density,
                seed,
                &cfg,
                world_scale,
            );
            drop(s);

            let dirty_keys: Vec<(i32, i32, i32)> =
                outcome.meshes.into_iter().map(|(k, _)| k).collect();
            // Material flipped on solid voxels — recompute crystal placements
            // exactly like BrushSphere does so quartz/amethyst/crystal-y ores
            // get their geode visual contribution refreshed.
            let new_placements: Vec<_> = {
                let s = store.read().unwrap();
                outcome.flipped_chunks.iter().filter_map(|&key| {
                    s.density_fields.get(&key).map(|density_field| {
                        let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                        (key, voxel_gen::compute_crystals(coord, density_field, &cfg))
                    })
                }).collect()
            };
            {
                let mut s = store.write().unwrap();
                for (key, placements) in new_placements {
                    s.crystal_placements.insert(key, placements);
                }
            }
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
        }
        WorkerRequest::BrushPaintStress { center_rust, radius, amount, cap, op, falloff } => {
            let cfg = config.read().unwrap().clone();
            let mut s = store.write().unwrap();
            let _outcome = crate::brushes::paint_stress_sphere(
                &mut s, center_rust, radius, amount, falloff, op, cap, &cfg, world_scale,
            );
            // PaintStress does not change density/material → no remesh or seam
            // pass is needed. The UE side picks up the updated painted overlay
            // on the next `voxel_query_stress` call (and the V-key overlay
            // recalc preview already drives that path).
            drop(s);
        }
        WorkerRequest::BrushUndo => {
            let cfg = config.read().unwrap().clone();
            let mut s = store.write().unwrap();
            let outcome = crate::brushes::apply_undo(&mut s, &cfg, world_scale);
            drop(s);
            if let Some(outcome) = outcome {
                let dirty_keys: Vec<(i32, i32, i32)> =
                    outcome.meshes.into_iter().map(|(k, _)| k).collect();
                let new_placements: Vec<_> = {
                    let s = store.read().unwrap();
                    outcome.flipped_chunks.iter().filter_map(|&key| {
                        s.density_fields.get(&key).map(|density| {
                            let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                            (key, voxel_gen::compute_crystals(coord, density, &cfg))
                        })
                    }).collect()
                };
                {
                    let mut s = store.write().unwrap();
                    for (key, placements) in new_placements {
                        s.crystal_placements.insert(key, placements);
                    }
                }
                batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
            }
        }
        WorkerRequest::BrushNoise { center_rust, radius, frequency, strength, seed } => {
            let cfg = config.read().unwrap().clone();
            let mut s = store.write().unwrap();
            let outcome = crate::brushes::noise_brush(
                &mut s, center_rust, radius, frequency, strength, seed, &cfg, world_scale,
            );
            drop(s);
            let dirty_keys: Vec<(i32, i32, i32)> =
                outcome.meshes.into_iter().map(|(k, _)| k).collect();
            let new_placements: Vec<_> = {
                let s = store.read().unwrap();
                outcome.flipped_chunks.iter().filter_map(|&key| {
                    s.density_fields.get(&key).map(|density| {
                        let coord = voxel_core::chunk::ChunkCoord::new(key.0, key.1, key.2);
                        (key, voxel_gen::compute_crystals(coord, density, &cfg))
                    })
                }).collect()
            };
            {
                let mut s = store.write().unwrap();
                for (key, placements) in new_placements {
                    s.crystal_placements.insert(key, placements);
                }
            }
            batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
        }
        WorkerRequest::BrushFluidSphere { center_rust, radius, fluid_type, is_source, op, max_flow_dist } => {
            // op semantics:
            //   0 = fill air (no carve)
            //   1 = clear (level=0)
            //   2 = pool-dig (carve solid + bottom-half non-source — designer's "make a basin")
            //   3 = carve + full fill (carve solid + fill entire sphere; respects is_source)
            let cfg = config.read().unwrap().clone();
            let ft = voxel_fluid::cell::FluidType::from_u8(fluid_type);
            let level = if op == 1 { 0.0 } else { 1.0 };
            let does_carve = op == 2 || op == 3;
            let bottom_half_only = op == 2;
            let force_non_source = op == 2; // pool-dig body always drains

            // Carve first (if applicable).
            let mut dig_outcome = None;
            if does_carve {
                let mut s = store.write().unwrap();
                dig_outcome = Some(crate::brushes::carve_sphere(&mut s, center_rust, radius, &cfg, world_scale));
                drop(s);
            }

            let cells = {
                let s = store.read().unwrap();
                crate::brushes::collect_fluid_cells_in_sphere(&s, center_rust, radius, bottom_half_only, &cfg)
            };

            // Refresh the fluid thread's density cache for every chunk this
            // brush touches. The fluid thread keeps its own density cache and
            // AddFluid checks `cell_capacity > MIN_LEVEL` against it. The
            // cache only refreshes when the worker sends DensityUpdate (at
            // gen time) or TerrainModified (after density changes).
            //
            // This needs to fire BOTH when carving (cells just changed
            // solid→air, fluid thread's cache still says solid) AND when
            // painting without carving (the cache may be stale from any
            // prior carve / config change / save-load cycle that didn't
            // notify the fluid thread). Without this, AddFluid silently
            // rejects every event because cell_capacity reads 0.
            //
            // Driving the chunk list from `cells` (rather than just the
            // carve outcome) covers paint-only mode too — the worker has
            // already filtered the cell list to "store thinks this is
            // air", so refreshing those chunks brings the fluid thread's
            // cache into agreement before AddFluid lands.
            {
                use std::collections::HashSet;
                let mut chunks_to_refresh: HashSet<(i32, i32, i32)> = HashSet::new();
                for c in &cells {
                    chunks_to_refresh.insert(c.chunk);
                }
                if let Some(ref outcome) = dig_outcome {
                    for (key, _) in &outcome.meshes {
                        chunks_to_refresh.insert(*key);
                    }
                }
                let s = store.read().unwrap();
                for key in chunks_to_refresh {
                    if let Some(density) = s.density_fields.get(&key) {
                        let densities: Vec<f32> = density.samples.iter().map(|s| s.density).collect();
                        let _ = fluid_event_tx.send(FluidEvent::TerrainModified {
                            chunk: key,
                            densities,
                        });
                    }
                }
            }

            for cell in cells {
                let cell_is_source = if force_non_source { false } else { is_source };
                let _ = fluid_event_tx.send(FluidEvent::AddFluid {
                    chunk: cell.chunk,
                    x: cell.x, y: cell.y, z: cell.z,
                    fluid_type: ft,
                    level,
                    is_source: cell_is_source,
                    // Bounded-flow only meaningful on sources.
                    max_flow_dist: if cell_is_source { max_flow_dist } else { 0 },
                });
            }

            if let Some(outcome) = dig_outcome {
                let dirty_keys: Vec<(i32, i32, i32)> =
                    outcome.meshes.into_iter().map(|(k, _)| k).collect();
                if !dirty_keys.is_empty() {
                    batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
                }
            }
        }
        WorkerRequest::BrushFluidBox { center_rust, half_ext_rust, fluid_type, is_source, op, max_flow_dist } => {
            let cfg = config.read().unwrap().clone();
            let ft = voxel_fluid::cell::FluidType::from_u8(fluid_type);
            let level = if op == 1 { 0.0 } else { 1.0 };

            // Op 2 (carve+fill): carve the box region first, then fill all air cells inside.
            // Fluid box brush is unrotated for now — add yaw_rad to FfiBrushFluidBoxRequest if needed.
            let mut dig_outcome = None;
            if op == 2 {
                let mut s = store.write().unwrap();
                dig_outcome = Some(crate::brushes::box_brush(
                    &mut s, center_rust, half_ext_rust, /*yaw_rad*/0.0, 1, // op=1 (carve)
                    voxel_core::material::Material::Air, &cfg, world_scale));
                drop(s);
            }

            let cells = {
                let s = store.read().unwrap();
                crate::brushes::collect_fluid_cells_in_box(&s, center_rust, half_ext_rust, &cfg)
            };

            // Refresh fluid thread density cache for every chunk touched —
            // covers carve mode AND paint-only mode. See BrushFluidSphere
            // for full rationale.
            {
                use std::collections::HashSet;
                let mut chunks_to_refresh: HashSet<(i32, i32, i32)> = HashSet::new();
                for c in &cells {
                    chunks_to_refresh.insert(c.chunk);
                }
                if let Some(ref outcome) = dig_outcome {
                    for (key, _) in &outcome.meshes {
                        chunks_to_refresh.insert(*key);
                    }
                }
                let s = store.read().unwrap();
                for key in chunks_to_refresh {
                    if let Some(density) = s.density_fields.get(&key) {
                        let densities: Vec<f32> = density.samples.iter().map(|s| s.density).collect();
                        let _ = fluid_event_tx.send(FluidEvent::TerrainModified {
                            chunk: key,
                            densities,
                        });
                    }
                }
            }

            for cell in cells {
                let cell_is_source = if op == 2 { false } else { is_source };
                let _ = fluid_event_tx.send(FluidEvent::AddFluid {
                    chunk: cell.chunk,
                    x: cell.x, y: cell.y, z: cell.z,
                    fluid_type: ft,
                    level,
                    // Carve-and-fill semantics: the body becomes non-source so it drains/spreads.
                    is_source: cell_is_source,
                    max_flow_dist: if cell_is_source { max_flow_dist } else { 0 },
                });
            }
            if let Some(outcome) = dig_outcome {
                let dirty_keys: Vec<(i32, i32, i32)> =
                    outcome.meshes.into_iter().map(|(k, _)| k).collect();
                if !dirty_keys.is_empty() {
                    batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
                }
            }
        }
        WorkerRequest::BrushFluidRiver { points, radius, fluid_type, is_source, op, max_flow_dist } => {
            let cfg = config.read().unwrap().clone();
            let ft = voxel_fluid::cell::FluidType::from_u8(fluid_type);

            // Op 2: carve a channel first using the existing tunnel brush, then fill it.
            let mut dig_outcome = None;
            if op == 2 {
                let mut s = store.write().unwrap();
                dig_outcome = Some(crate::brushes::tunnel(&mut s, &points, radius, None, &cfg, world_scale));
                drop(s);
            }

            let cells = {
                let s = store.read().unwrap();
                crate::brushes::collect_fluid_cells_in_capsule_chain(&s, &points, radius, &cfg)
            };

            // Refresh fluid thread density cache for every chunk touched —
            // covers carve mode AND paint-only mode. See BrushFluidSphere
            // for full rationale.
            {
                use std::collections::HashSet;
                let mut chunks_to_refresh: HashSet<(i32, i32, i32)> = HashSet::new();
                for c in &cells {
                    chunks_to_refresh.insert(c.chunk);
                }
                if let Some(ref outcome) = dig_outcome {
                    for (key, _) in &outcome.meshes {
                        chunks_to_refresh.insert(*key);
                    }
                }
                let s = store.read().unwrap();
                for key in chunks_to_refresh {
                    if let Some(density) = s.density_fields.get(&key) {
                        let densities: Vec<f32> = density.samples.iter().map(|s| s.density).collect();
                        let _ = fluid_event_tx.send(FluidEvent::TerrainModified {
                            chunk: key,
                            densities,
                        });
                    }
                }
            }
            for cell in cells {
                let _ = fluid_event_tx.send(FluidEvent::AddFluid {
                    chunk: cell.chunk,
                    x: cell.x, y: cell.y, z: cell.z,
                    fluid_type: ft,
                    level: 1.0,
                    is_source,
                    max_flow_dist: if is_source { max_flow_dist } else { 0 },
                });
            }
            if let Some(outcome) = dig_outcome {
                let dirty_keys: Vec<(i32, i32, i32)> =
                    outcome.meshes.into_iter().map(|(k, _)| k).collect();
                if !dirty_keys.is_empty() {
                    batched_seam_pass_mine(&dirty_keys, &cfg, store, result_tx, fluid_event_tx, world_scale);
                }
            }
        }
        // (removed BrushFluidStream handler — replaced by bounded sources via max_flow_dist on FluidCell)
        WorkerRequest::ApplyLavaQuench { obsidian, scoria, drained_water: _ } => {
            // Apply the live lava↔water quench plan emitted by the fluid sim.
            // - obsidian cells: Material::Obsidian (glassy quench skin, hardness 0.85)
            // - scoria cells:   Material::Scoria   (steam-altered halo, hardness 0.45)
            // The fluid grid already drained these cells locally; here we make
            // the underlying voxels solid + remesh + nudge the fluid sim's
            // density cache so it knows the cells are now solid.
            if obsidian.is_empty() && scoria.is_empty() {
                return;
            }
            let cfg = config.read().unwrap().clone();
            let cs = cfg.chunk_size as i32;

            let mut s = store.write().unwrap();
            let mut dirty_set: HashSet<(i32, i32, i32)> = HashSet::new();
            let mut written: Vec<voxel_core::density_ops::WrittenCell> = Vec::new();
            let mut changed: u32 = 0;

            // Helper: convert local (key, lx, ly, lz) → world coords + write
            // through density_ops::write_all_locations so chunk-boundary
            // mirrors stay in sync.
            for (key, lx, ly, lz) in &obsidian {
                let wx = key.0 * cs + *lx as i32;
                let wy = key.1 * cs + *ly as i32;
                let wz = key.2 * cs + *lz as i32;
                voxel_core::density_ops::write_all_locations(
                    &mut s.density_fields, cs, wx, wy, wz,
                    |_old_d, _old_m| Some((1.0, voxel_core::material::Material::Obsidian)),
                    &mut dirty_set, &mut written, &mut changed,
                );
            }
            for (key, lx, ly, lz) in &scoria {
                let wx = key.0 * cs + *lx as i32;
                let wy = key.1 * cs + *ly as i32;
                let wz = key.2 * cs + *lz as i32;
                voxel_core::density_ops::write_all_locations(
                    &mut s.density_fields, cs, wx, wy, wz,
                    |_old_d, old_m| {
                        // Don't overwrite obsidian we just placed at the same
                        // boundary position (rim wins over halo).
                        if old_m == voxel_core::material::Material::Obsidian {
                            None
                        } else {
                            Some((1.0, voxel_core::material::Material::Scoria))
                        }
                    },
                    &mut dirty_set, &mut written, &mut changed,
                );
            }

            // Persist these changes across save/load.
            let dirty_chunks: Vec<(i32, i32, i32)> = dirty_set.iter().copied().collect();
            s.modification_tracker.mark_dirty_many(&dirty_chunks);

            // Remesh affected chunks (full chunk bounds — we touched solid voxels).
            let dirty_bounds: Vec<_> = dirty_chunks.iter().map(|&k| {
                (k, 0usize, 0usize, 0usize, cfg.chunk_size, cfg.chunk_size, cfg.chunk_size)
            }).collect();
            let meshes = s.remesh_dirty(&dirty_bounds, &cfg, world_scale);

            // Capture density for the fluid TerrainModified events while we
            // still hold the lock, so the fluid sim's density cache is updated
            // and stops trying to flow into the new solid voxels.
            let mut terrain_updates: Vec<((i32,i32,i32), Vec<f32>)> = Vec::new();
            for &k in &dirty_chunks {
                if let Some(df) = s.density_fields.get(&k) {
                    let densities: Vec<f32> = df.samples.iter().map(|s| s.density).collect();
                    terrain_updates.push((k, densities));
                }
            }
            drop(s);

            // Ship remeshed chunks back to UE through the normal pipeline.
            for (chunk, mesh) in meshes {
                let _ = result_tx.send(WorkerResult::ChunkMesh {
                    chunk,
                    mesh,
                    generation: 0,
                    crystal_data: Vec::new(),
                    mushroom_data: Vec::new(),
                    zone_descriptors: Vec::new(),
                });
            }

            // Notify fluid sim of new solid voxels so its density cache
            // matches the world state.
            for (chunk, densities) in terrain_updates {
                let _ = fluid_event_tx.send(FluidEvent::TerrainModified {
                    chunk, densities,
                });
            }

            // Seam pass so chunk-boundary quads tween correctly between
            // the new solid (1.0) and adjacent air/density values.
            if !dirty_chunks.is_empty() {
                batched_seam_pass(&dirty_chunks, &cfg, store, result_tx, fluid_event_tx, world_scale);
            }
        }
        WorkerRequest::Unload { chunk } => {
            let mut s = store.write().unwrap();
            s.unload(chunk);
            generation_counters.remove(&chunk);
            let _ = fluid_event_tx.send(FluidEvent::ChunkUnloaded { chunk });
        }
        WorkerRequest::PlaceSupport { world_x, world_y, world_z, support_type } => {
            let cfg = config.read().unwrap().clone();
            let stress_cfg = stress_config.read().unwrap().clone();
            let st = SupportType::from_u8(support_type);

            let mut s = store.write().unwrap();
            let (success, _collapse_events, dirty_bounds) = s.place_support(
                (world_x, world_y, world_z), st, &stress_cfg, cfg.chunk_size,
            );

            // Remesh affected chunks
            let remesh_bounds: Vec<_> = dirty_bounds.iter().map(|&(key, (min_x, min_y, min_z, max_x, max_y, max_z))| {
                (key, min_x, min_y, min_z, max_x, max_y, max_z)
            }).collect();
            let meshes = s.remesh_dirty(&remesh_bounds, &cfg, world_scale);
            drop(s);

            // Send support result with remeshed chunks
            let mesh_pairs: Vec<_> = meshes.into_iter().collect();
            let _ = result_tx.send(WorkerResult::SupportResult {
                success,
                meshes: mesh_pairs,
            });
        }
        WorkerRequest::RemoveSupport { world_x, world_y, world_z } => {
            let cfg = config.read().unwrap().clone();
            let stress_cfg = stress_config.read().unwrap().clone();

            let mut s = store.write().unwrap();
            let (removed, _collapse_events, dirty_bounds) = s.remove_support(
                (world_x, world_y, world_z), &stress_cfg, cfg.chunk_size,
            );

            // Remesh affected chunks
            let remesh_bounds: Vec<_> = dirty_bounds.iter().map(|&(key, (min_x, min_y, min_z, max_x, max_y, max_z))| {
                (key, min_x, min_y, min_z, max_x, max_y, max_z)
            }).collect();
            let meshes = s.remesh_dirty(&remesh_bounds, &cfg, world_scale);
            drop(s);

            // Send support result
            let mesh_pairs: Vec<_> = meshes.into_iter().collect();
            let _ = result_tx.send(WorkerResult::SupportResult {
                success: removed.is_some(),
                meshes: mesh_pairs,
            });
        }
        WorkerRequest::Sleep { player_chunk, sleep_count, sleep_config: sc } => {
            let cfg = config.read().unwrap().clone();
            let sleep_config = sc;
            let t_worker_start = Instant::now();
            crate::panic_log::note(&format!("[SLEEP_TRACE] enter Sleep handler player_chunk=({},{},{})", player_chunk.0, player_chunk.1, player_chunk.2));

            // Request fluid snapshot for geological processes
            let (snap_tx, snap_rx) = crossbeam_channel::bounded(1);
            crate::panic_log::note("[SLEEP_TRACE] sending SnapshotRequest");
            let _ = fluid_event_tx.send(voxel_fluid::FluidEvent::SnapshotRequest { reply_tx: snap_tx });
            let mut fluid_snapshot = snap_rx.recv().unwrap_or_else(|_| voxel_fluid::FluidSnapshot::default());
            crate::panic_log::note(&format!("[SLEEP_TRACE] got fluid snapshot ({} chunks)", fluid_snapshot.chunks.len()));

            crate::panic_log::note("[SLEEP_TRACE] acquiring store write lock");
            let mut s = store.write().unwrap();
            crate::panic_log::note("[SLEEP_TRACE] store write lock acquired");

            // Use helper to get three simultaneous &mut borrows (borrow checker
            // cannot split borrows through method calls on the same struct).
            let (density_fields, stress_fields, support_fields) = s.sleep_fields_mut();
            crate::panic_log::note(&format!("[SLEEP_TRACE] entering execute_sleep ({} density chunks)", density_fields.len()));

            // Execute the sleep cycle on the mutable store fields
            let sleep_result = voxel_sleep::execute_sleep(
                &sleep_config,
                density_fields,
                stress_fields,
                support_fields,
                &mut fluid_snapshot,
                player_chunk,
                sleep_count,
                None, // No progress channel for now
            );
            crate::panic_log::note(&format!("[SLEEP_TRACE] execute_sleep returned: chunks_changed={} dirty_chunks={} metamorphosed={}", sleep_result.chunks_changed, sleep_result.dirty_chunks.len(), sleep_result.voxels_metamorphosed));

            // Drain solidified lava from the real fluid system
            if sleep_result.lava_solidified > 0 {
                let lava_chunks: Vec<(i32, i32, i32)> = fluid_snapshot.chunks.keys().copied().collect();
                let _ = fluid_event_tx.send(voxel_fluid::FluidEvent::DrainLavaChunks { chunks: lava_chunks });
            }

            // Remesh all dirty chunks (full chunk bounds)
            let t_remesh = Instant::now();
            let dirty_count = sleep_result.dirty_chunks.len();
            let mut dirty_bounds: Vec<_> = sleep_result.dirty_chunks.iter().map(|&key| {
                (key, 0usize, 0usize, 0usize, cfg.chunk_size, cfg.chunk_size, cfg.chunk_size)
            }).collect();

            // Mark sleep-modified chunks for save persistence.
            // (Re-locking `store.write()` here would deadlock — the outer scope
            // already holds the write guard `s` from before execute_sleep.)
            s.modification_tracker.mark_dirty_many(&sleep_result.dirty_chunks);

            // NOTE: Do NOT call sync_boundaries here. Sleep uses set_voxel_synced()
            // which already keeps boundary overlap voxels consistent. Running
            // sync_boundary_density on top of that causes material bleeding:
            // its average_boundary_voxel() picks material by density comparison,
            // which can propagate hornfels/skarn to distant chunk boundaries.

            // However, set_voxel_synced writes mirror copies into neighbor chunks'
            // density fields. Those neighbors need remeshing too. Add all 26
            // face/edge/corner neighbors of dirty chunks that are loaded.
            {
                let dirty_set: std::collections::HashSet<(i32,i32,i32)> =
                    sleep_result.dirty_chunks.iter().copied().collect();
                let mut extra: Vec<((i32,i32,i32), usize, usize, usize, usize, usize, usize)> = Vec::new();
                for &(cx, cy, cz) in &sleep_result.dirty_chunks {
                    for dx in -1i32..=1 {
                        for dy in -1i32..=1 {
                            for dz in -1i32..=1 {
                                if dx == 0 && dy == 0 && dz == 0 { continue; }
                                let nk = (cx + dx, cy + dy, cz + dz);
                                if !dirty_set.contains(&nk) && s.density_fields.contains_key(&nk) {
                                    extra.push((nk, 0, 0, 0, cfg.chunk_size, cfg.chunk_size, cfg.chunk_size));
                                }
                            }
                        }
                    }
                }
                // Deduplicate
                extra.sort_by_key(|e| e.0);
                extra.dedup_by_key(|e| e.0);
                dirty_bounds.extend(extra);
            }

            let meshes = s.remesh_dirty(&dirty_bounds, &cfg, world_scale);
            drop(s);
            let t_remesh_elapsed = t_remesh.elapsed();

            // Send each dirty chunk mesh through the normal ChunkMesh pipeline
            // so UE auto-remeshes existing chunk actors
            let t_mesh_send = Instant::now();
            eprintln!("[SLEEP_REMESH] Sending {} chunk meshes (from {} dirty + neighbors)", meshes.len(), dirty_count);
            for (chunk, mesh) in meshes {
                let ue = crate::convert::rust_chunk_to_ue(chunk.0, chunk.1, chunk.2);
                let vert_count = mesh.positions.len() / 3;
                eprintln!("[SLEEP_REMESH]   Rust({},{},{}) → UE({},{},{})  verts={}",
                    chunk.0, chunk.1, chunk.2, ue.0, ue.1, ue.2, vert_count);
                let crystal_data = retrieve_crystal_data(store, chunk, cfg.voxel_scale(), world_scale);
                let mushroom_data = retrieve_mushroom_data(store, chunk, cfg.voxel_scale(), world_scale);
                let _ = result_tx.send(WorkerResult::ChunkMesh {
                    chunk,
                    mesh,
                    generation: 0, // Sleep remesh
                    crystal_data,
                    mushroom_data,
                    zone_descriptors: Vec::new(),
                });
            }
            let t_mesh_send_elapsed = t_mesh_send.elapsed();

            // Send collapse events through the normal CollapseResult pipeline
            let t_collapse_send = Instant::now();
            if !sleep_result.collapse_events.is_empty() {
                let ffi_events: Vec<FfiCollapseEvent> = sleep_result.collapse_events.iter().map(|e| {
                    FfiCollapseEvent {
                        center_x: e.center.0 * world_scale,
                        center_y: -e.center.2 * world_scale,  // Rust Y-up -> UE Z-up
                        center_z: e.center.1 * world_scale,
                        volume: e.volume,
                    }
                }).collect();
                let _ = result_tx.send(WorkerResult::CollapseResult {
                    events: ffi_events,
                    meshes: Vec::new(), // Meshes already sent above
                });
            }
            let t_collapse_send_elapsed = t_collapse_send.elapsed();

            // Regenerate seams for dirty chunks
            let t_seam = Instant::now();
            let seam_count = sleep_result.dirty_chunks.len();
            batched_seam_pass(&sleep_result.dirty_chunks, &cfg, store, result_tx, fluid_event_tx, world_scale);
            let t_seam_elapsed = t_seam.elapsed();

            // Build combined profile report with worker timings appended
            let t_worker_total = t_worker_start.elapsed();
            let worker_post_total = t_remesh_elapsed + t_mesh_send_elapsed + t_collapse_send_elapsed + t_seam_elapsed;
            let dur_ms = |d: Duration| d.as_secs_f64() * 1000.0;
            let mut report = sleep_result.profile_report.clone();
            use std::fmt::Write as FmtWrite;
            let _ = writeln!(report);
            let _ = writeln!(report, "─── Worker Post-Processing ─────────────────────────");
            let _ = writeln!(report, "  Remesh ({} chunks):  {:.2} ms", dirty_count, dur_ms(t_remesh_elapsed));
            let _ = writeln!(report, "  Mesh send:           {:.2} ms", dur_ms(t_mesh_send_elapsed));
            let _ = writeln!(report, "  Collapse events:     {:.2} ms", dur_ms(t_collapse_send_elapsed));
            let _ = writeln!(report, "  Seam regen ({}):     {:.2} ms", seam_count, dur_ms(t_seam_elapsed));
            let _ = writeln!(report, "  Worker post total:   {:.2} ms", dur_ms(worker_post_total));
            let _ = writeln!(report);
            let _ = writeln!(report, "═══════════════════════════════════════════════════════");
            let _ = writeln!(report, "  GRAND TOTAL (worker): {:.2} ms", dur_ms(t_worker_total));
            let _ = writeln!(report, "═══════════════════════════════════════════════════════");

            // Compact & serialize manifest for morph system — filter to aureole block only
            // Compact manifest (merge multi-phase changes per voxel) but don't filter by block —
            // cinematic mode uses a player-aimed block that differs from Rust's showcase block.
            // Manifest is cached once via set_morph_manifest, so full size (~30MB) is acceptable.
            crate::panic_log::note("[SLEEP_TRACE] cloning + compacting manifest");
            let mut compact_manifest = sleep_result.manifest.clone();
            compact_manifest.compact();
            crate::panic_log::note("[SLEEP_TRACE] serializing manifest to JSON");
            let manifest_json = compact_manifest.to_json().unwrap_or_default();
            crate::panic_log::note(&format!("[SLEEP_TRACE] manifest JSON serialized ({} bytes), sending SleepComplete", manifest_json.len()));

            // Send sleep completion stats (intercepted by engine.poll_result)
            let _ = result_tx.send(WorkerResult::SleepComplete {
                chunks_changed: sleep_result.chunks_changed,
                voxels_metamorphosed: sleep_result.voxels_metamorphosed,
                minerals_grown: sleep_result.minerals_grown,
                supports_degraded: sleep_result.supports_degraded,
                collapses_triggered: sleep_result.collapses_triggered,
                acid_dissolved: sleep_result.acid_dissolved,
                veins_deposited: sleep_result.veins_deposited,
                voxels_enriched: sleep_result.voxels_enriched,
                formations_grown: sleep_result.formations_grown,
                sulfide_dissolved: sleep_result.sulfide_dissolved,
                coal_matured: sleep_result.coal_matured,
                diamonds_formed: sleep_result.diamonds_formed,
                voxels_silicified: sleep_result.voxels_silicified,
                nests_fossilized: sleep_result.nests_fossilized,
                channels_eroded: sleep_result.channels_eroded,
                corpses_fossilized: sleep_result.corpses_fossilized,
                lava_solidified: sleep_result.lava_solidified,
                profile_report: report,
                aureole_glimpse_pos: sleep_result.aureole_glimpse_pos,
                aureole_showcase_block: sleep_result.aureole_showcase_block,
                manifest_json,
                lava_cells: sleep_result.lava_cells,
            });
        }
        WorkerRequest::AureoleOnly { player_chunk, sleep_config: sc } => {
            let cfg = config.read().unwrap().clone();

            // Request fluid snapshot (same as Sleep)
            let (snap_tx, snap_rx) = crossbeam_channel::bounded(1);
            let _ = fluid_event_tx.send(voxel_fluid::FluidEvent::SnapshotRequest { reply_tx: snap_tx });
            let mut fluid_snapshot = snap_rx.recv().unwrap_or_else(|_| voxel_fluid::FluidSnapshot::default());

            let mut s = store.write().unwrap();

            // Run aureole-only (no stress/support fields needed)
            let sleep_result = voxel_sleep::execute_aureole_only(
                &sc,
                &mut s.density_fields,
                &mut fluid_snapshot,
                player_chunk,
            );

            // Drain solidified lava if any
            if sleep_result.lava_solidified > 0 {
                let lava_chunks: Vec<(i32, i32, i32)> = fluid_snapshot.chunks.keys().copied().collect();
                let _ = fluid_event_tx.send(voxel_fluid::FluidEvent::DrainLavaChunks { chunks: lava_chunks });
            }

            // Remesh dirty chunks (same pattern as Sleep)
            let dirty_count = sleep_result.dirty_chunks.len();
            let mut dirty_bounds: Vec<_> = sleep_result.dirty_chunks.iter().map(|&key| {
                (key, 0usize, 0usize, 0usize, cfg.chunk_size, cfg.chunk_size, cfg.chunk_size)
            }).collect();

            // Expand to 26-neighbors so boundary voxels remesh correctly
            {
                let dirty_set: std::collections::HashSet<(i32,i32,i32)> =
                    sleep_result.dirty_chunks.iter().copied().collect();
                let mut extra: Vec<((i32,i32,i32), usize, usize, usize, usize, usize, usize)> = Vec::new();
                for &(cx, cy, cz) in &sleep_result.dirty_chunks {
                    for dx in -1i32..=1 {
                        for dy in -1i32..=1 {
                            for dz in -1i32..=1 {
                                if dx == 0 && dy == 0 && dz == 0 { continue; }
                                let nk = (cx + dx, cy + dy, cz + dz);
                                if !dirty_set.contains(&nk) && s.density_fields.contains_key(&nk) {
                                    extra.push((nk, 0, 0, 0, cfg.chunk_size, cfg.chunk_size, cfg.chunk_size));
                                }
                            }
                        }
                    }
                }
                extra.sort_by_key(|e| e.0);
                extra.dedup_by_key(|e| e.0);
                dirty_bounds.extend(extra);
            }

            let meshes = s.remesh_dirty(&dirty_bounds, &cfg, world_scale);
            drop(s);

            eprintln!("[AUREOLE_REMESH] Sending {} chunk meshes (from {} dirty + neighbors)", meshes.len(), dirty_count);
            for (chunk, mesh) in meshes {
                let ue = crate::convert::rust_chunk_to_ue(chunk.0, chunk.1, chunk.2);
                let vert_count = mesh.positions.len() / 3;
                eprintln!("[AUREOLE_REMESH]   Rust({},{},{}) → UE({},{},{})  verts={}",
                    chunk.0, chunk.1, chunk.2, ue.0, ue.1, ue.2, vert_count);
                let crystal_data = retrieve_crystal_data(store, chunk, cfg.voxel_scale(), world_scale);
                let mushroom_data = retrieve_mushroom_data(store, chunk, cfg.voxel_scale(), world_scale);
                let _ = result_tx.send(WorkerResult::ChunkMesh {
                    chunk,
                    mesh,
                    generation: 0,
                    crystal_data,
                    mushroom_data,
                    zone_descriptors: Vec::new(),
                });
            }

            // Build report
            let mut report = sleep_result.profile_report.clone();
            use std::fmt::Write as FmtWrite;
            let _ = writeln!(report, "\nRemeshed {} dirty chunks (+neighbors)", dirty_count);

            // Deliver via SleepComplete so UE can parse debug lines and display results
            let _ = result_tx.send(WorkerResult::SleepComplete {
                chunks_changed: sleep_result.chunks_changed,
                voxels_metamorphosed: sleep_result.voxels_metamorphosed,
                minerals_grown: 0,
                supports_degraded: 0,
                collapses_triggered: 0,
                acid_dissolved: 0,
                veins_deposited: 0,
                voxels_enriched: 0,
                formations_grown: 0,
                sulfide_dissolved: 0,
                coal_matured: sleep_result.coal_matured,
                diamonds_formed: sleep_result.diamonds_formed,
                voxels_silicified: sleep_result.voxels_silicified,
                nests_fossilized: 0,
                channels_eroded: sleep_result.channels_eroded,
                corpses_fossilized: 0,
                lava_solidified: sleep_result.lava_solidified,
                profile_report: report,
                aureole_glimpse_pos: sleep_result.aureole_glimpse_pos,
                aureole_showcase_block: sleep_result.aureole_showcase_block,
                manifest_json: String::new(), // Aureole-only doesn't need morph
                lava_cells: sleep_result.lava_cells,
            });
        }
        WorkerRequest::MorphStep { chunks, step, total_steps } => {
            let cfg = config.read().unwrap().clone();

            // Borrow cached manifest (set once via set_morph_manifest, reused for all 30 steps)
            // Hold lock for duration of step — acceptable since only one morph runs at a time
            let manifest_guard = morph_manifest.lock().unwrap();
            let manifest = match manifest_guard.as_ref() {
                Some(m) => m,
                None => {
                    eprintln!("[MORPH] No cached manifest — call set_morph_manifest first");
                    drop(manifest_guard);
                    let _ = result_tx.send(WorkerResult::MorphMeshes {
                        step, total_steps, meshes: Vec::new(),
                    });
                    return;
                }
            };

            let t = if total_steps > 0 { step as f32 / total_steps as f32 } else { 1.0 };

            // Force ALL chunks active every step to prevent seam cracks between
            // active (morph-updated) and inactive (stale) neighbors.
            // Parallelized mesh gen (rayon) keeps this fast.
            let active: Vec<bool> = vec![true; chunks.len()];

            let s = store.read().unwrap();

            // Phase 1: Clone active density fields and apply manifest interpolation
            let mut density_fields: Vec<Option<voxel_core::density::DensityField>> = Vec::with_capacity(chunks.len());
            for (i, &key) in chunks.iter().enumerate() {
                if !active[i] {
                    density_fields.push(None); // Skip — existing mesh preserved
                    continue;
                }
                match s.density_fields.get(&key) {
                    Some(d) => {
                        let mut df = d.clone();
                        if let Some(delta) = manifest.chunk_deltas.get(&key) {
                            for change in &delta.voxel_changes {
                                let sample = df.get_mut(change.lx, change.ly, change.lz);
                                let old_d = change.old_density;
                                let new_d = change.new_density;
                                // Per-voxel spreading: voxels near heat source (spread=0)
                                // transform first, farthest (spread=1) start at t=0.6
                                let delay_factor = 0.6_f32;
                                let voxel_delay = change.spread_distance * delay_factor;
                                let voxel_t = ((t - voxel_delay) / (1.0 - voxel_delay)).clamp(0.0, 1.0);
                                sample.density = old_d + (new_d - old_d) * voxel_t;
                                let old_mat = voxel_core::material::Material::from_u8(change.old_material);
                                let new_mat = voxel_core::material::Material::from_u8(change.new_material);
                                sample.material = if voxel_t >= 0.5 { new_mat } else { old_mat };
                            }
                        }
                        density_fields.push(Some(df));
                    }
                    None => density_fields.push(None),
                }
            }
            drop(s);

            // Phase 2: Sync boundaries between adjacent chunks in the block.
            // Generalized boundary sync: for each pair of chunks that are adjacent
            // (differ by 1 in exactly one axis), sync the shared face.
            {
                let cs_val = density_fields.iter().flatten().next()
                    .map(|df| df.size - 1).unwrap_or(16);

                let sync_face = |dfs: &mut [Option<voxel_core::density::DensityField>], a: usize, b: usize, axis: usize, cs: usize| {
                    let boundary: Vec<(usize, usize, f32, voxel_core::material::Material)> = {
                        let src = match &dfs[a] { Some(d) => d, None => return };
                        let mut out = Vec::with_capacity((cs + 1) * (cs + 1));
                        for u in 0..=cs {
                            for v in 0..=cs {
                                let s = match axis {
                                    0 => src.get(cs, u, v),
                                    1 => src.get(u, cs, v),
                                    _ => src.get(u, v, cs),
                                };
                                out.push((u, v, s.density, s.material));
                            }
                        }
                        out
                    };
                    if let Some(dst) = &mut dfs[b] {
                        for &(u, v, density, material) in &boundary {
                            let s = match axis {
                                0 => dst.get_mut(0, u, v),
                                1 => dst.get_mut(u, 0, v),
                                _ => dst.get_mut(u, v, 0),
                            };
                            s.density = density;
                            s.material = material;
                        }
                    }
                };

                // Find adjacent pairs: chunks[a] and chunks[b] are adjacent if they
                // differ by exactly 1 in one axis and 0 in the others
                for a in 0..chunks.len() {
                    for b in (a + 1)..chunks.len() {
                        let (ax, ay, az) = chunks[a];
                        let (bx, by, bz) = chunks[b];
                        let dx = bx - ax;
                        let dy = by - ay;
                        let dz = bz - az;
                        if dx == 1 && dy == 0 && dz == 0 {
                            sync_face(&mut density_fields, a, b, 0, cs_val);
                        } else if dx == 0 && dy == 1 && dz == 0 {
                            sync_face(&mut density_fields, a, b, 1, cs_val);
                        } else if dx == 0 && dy == 0 && dz == 1 {
                            sync_face(&mut density_fields, a, b, 2, cs_val);
                        }
                    }
                }
            }

            // Phase 3: Mesh all density fields + build seam data (parallelized with rayon)
            let chunk_size = cfg.chunk_size;

            // Parallel mesh generation — each chunk independently computes hermite + DC + mesh + smooth
            type BEdges = Vec<(voxel_core::hermite::EdgeKey, voxel_core::hermite::EdgeIntersection)>;
            let mesh_results: Vec<Option<(voxel_core::mesh::Mesh, Vec<glam::Vec3>, BEdges)>> =
                density_fields.par_iter().map(|df_opt| {
                    match df_opt {
                        Some(df) => {
                            let h = extract_hermite_data(df);
                            let cell_size = df.size - 1;
                            let dc_verts = solve_dc_vertices(&h, cell_size);
                            let mut mesh = generate_mesh(&h, &dc_verts, cell_size);
                            mesh.smooth(cfg.mesh_smooth_iterations, cfg.mesh_smooth_strength, cfg.mesh_boundary_smooth, Some(cell_size));
                            let boundary_edges = region_gen::extract_boundary_edges(&h, chunk_size);
                            Some((mesh, dc_verts, boundary_edges))
                        }
                        None => None,
                    }
                }).collect();

            // Collect results back into sequential structures (seam_data_map is not thread-safe).
            // Pre-populate with the global store's seam data for out-of-block neighbors so
            // edge chunks in the morph block can seam against their stored neighbors.
            // Without this, `generate_chunk_seam_quads` skips every quad that references
            // a chunk outside the morph block (returns `valid = false` on missing lookup)
            // — leaving visible seam gaps along the morph-block perimeter, the exact
            // behaviour the recent brush-side seam fix addressed. Brushes already pull
            // from `store.chunk_seam_data`; we mirror that here for the morph path.
            // Mid-morph the out-of-block DC vertices are the post-sleep snapshot (t=1),
            // so the seam will be slightly inconsistent until t→1 — acceptable trade-off
            // vs. fully missing quads. At t=1 it matches the dirty-chunks state exactly.
            let chunks_set: std::collections::HashSet<(i32, i32, i32)> =
                chunks.iter().copied().collect();
            let mut seam_data_map: std::collections::HashMap<(i32, i32, i32), ChunkSeamData> = {
                let s = store.read().unwrap();
                let mut map = std::collections::HashMap::new();
                for &c in &chunks {
                    for dx in -1..=1i32 {
                        for dy in -1..=1i32 {
                            for dz in -1..=1i32 {
                                let n = (c.0 + dx, c.1 + dy, c.2 + dz);
                                if chunks_set.contains(&n) { continue; }
                                if let Some(data) = s.chunk_seam_data.get(&n) {
                                    map.insert(n, data.clone());
                                }
                            }
                        }
                    }
                }
                map
            };
            let mut base_meshes: Vec<Option<voxel_core::mesh::Mesh>> = Vec::with_capacity(chunks.len());

            for (i, result) in mesh_results.into_iter().enumerate() {
                match result {
                    Some((mesh, dc_verts, boundary_edges)) => {
                        seam_data_map.insert(chunks[i], ChunkSeamData {
                            dc_vertices: dc_verts,
                            world_origin: glam::Vec3::ZERO,
                            boundary_edges,
                        });
                        base_meshes.push(Some(mesh));
                    }
                    None => {
                        base_meshes.push(None);
                    }
                }
            }

            // Phase 4: Generate seam quads and append to base meshes, then convert
            let mut meshes = Vec::with_capacity(chunks.len());
            for (i, base_opt) in base_meshes.into_iter().enumerate() {
                match base_opt {
                    Some(mut mesh) => {
                        // Generate seam quads for this chunk using neighbors in the block
                        let seam_mesh = region_gen::generate_chunk_seam_quads(
                            chunks[i], &seam_data_map, chunk_size);
                        if !seam_mesh.triangles.is_empty() {
                            mesh.append(seam_mesh);
                        }
                        if cfg.mesh_recalc_normals > 0 { mesh.recalculate_normals(); }

                        let mut converted = convert_mesh_to_ue_scaled(&mesh, cfg.voxel_scale(), world_scale);
                        crate::convert::bucket_mesh_by_material(&mut converted);
                        meshes.push(converted);
                    }
                    None => {
                        meshes.push(crate::types::ConvertedMesh {
                            positions: Vec::new(),
                            normals: Vec::new(),
                            material_ids: Vec::new(),
                            indices: Vec::new(),
                            submeshes: Vec::new(),
                        });
                    }
                }
            }

            eprintln!("[MORPH] Step {}/{}: meshed {} chunks", step, total_steps, meshes.len());

            let _ = result_tx.send(WorkerResult::MorphMeshes {
                step, total_steps, meshes,
            });
        }
        WorkerRequest::WorldScan => {
            let cfg = config.read().unwrap().clone();
            let s = store.read().unwrap();

            // Convert store data to scan-compatible types
            let scan_seam_data: std::collections::HashMap<(i32,i32,i32), voxel_core::world_scan::ScanSeamData> =
                s.chunk_seam_data.iter().map(|(&k, v)| {
                    (k, voxel_core::world_scan::ScanSeamData {
                        boundary_edges: v.boundary_edges.clone(),
                    })
                }).collect();

            let scan_worm_paths: Vec<Vec<voxel_core::world_scan::ScanWormSegment>> =
                s.region_worm_paths.values().flat_map(|paths| {
                    paths.iter().map(|path| {
                        path.iter().map(|seg| voxel_core::world_scan::ScanWormSegment {
                            position: [seg.position.x, seg.position.y, seg.position.z],
                            radius: seg.radius,
                        }).collect::<Vec<_>>()
                    })
                }).collect();

            let result = voxel_core::world_scan::scan_world(
                &s.density_fields,
                &s.base_meshes,
                &scan_seam_data,
                &scan_worm_paths,
                cfg.chunk_size,
            );

            let json = result.to_json_string();
            drop(s);

            let _ = result_tx.send(WorkerResult::ScanComplete { json_report: json });
        }
        WorkerRequest::WorldScanWithConfig { config: scan_config } => {
            let cfg = config.read().unwrap().clone();
            let s = store.read().unwrap();

            let scan_seam_data: std::collections::HashMap<(i32,i32,i32), voxel_core::world_scan::ScanSeamData> =
                s.chunk_seam_data.iter().map(|(&k, v)| {
                    (k, voxel_core::world_scan::ScanSeamData {
                        boundary_edges: v.boundary_edges.clone(),
                    })
                }).collect();

            let scan_worm_paths: Vec<Vec<voxel_core::world_scan::ScanWormSegment>> =
                s.region_worm_paths.values().flat_map(|paths| {
                    paths.iter().map(|path| {
                        path.iter().map(|seg| voxel_core::world_scan::ScanWormSegment {
                            position: [seg.position.x, seg.position.y, seg.position.z],
                            radius: seg.radius,
                        }).collect::<Vec<_>>()
                    })
                }).collect();

            let result = voxel_core::world_scan::scan_world_with_config(
                &s.density_fields,
                &s.base_meshes,
                Some(&s.hermite_data),
                &scan_seam_data,
                &scan_worm_paths,
                cfg.chunk_size,
                &scan_config,
            );

            let json = result.to_json_string();
            drop(s);

            let _ = result_tx.send(WorkerResult::ScanComplete { json_report: json });
        }
        WorkerRequest::ForceSpawnPool { world_x, world_y, world_z, fluid_type } => {
            let cfg = config.read().unwrap().clone();

            // Convert UE world position to Rust coordinates
            let center = from_ue_world_pos(world_x, world_y, world_z, world_scale);
            let chunk_size = cfg.chunk_size;

            // Compute chunk coordinate and local position
            let cx = (center.x / chunk_size as f32).floor() as i32;
            let cy = (center.y / chunk_size as f32).floor() as i32;
            let cz = (center.z / chunk_size as f32).floor() as i32;
            let key = (cx, cy, cz);

            let lx = ((center.x - cx as f32 * chunk_size as f32) as usize).min(chunk_size);
            let ly = ((center.y - cy as f32 * chunk_size as f32) as usize).min(chunk_size);
            let lz = ((center.z - cz as f32 * chunk_size as f32) as usize).min(chunk_size);

            // Check if chunk is loaded
            let has_density = {
                let s = store.read().unwrap();
                s.density_fields.contains_key(&key)
            };

            if !has_density {
                let json = serde_json::json!({
                    "error": "chunk not loaded",
                    "chunk": [cx, cy, cz],
                }).to_string();
                let _ = result_tx.send(WorkerResult::ForceSpawnPoolComplete { json_report: json });
                return;
            }

            let pool_fluid = if fluid_type == 1 {
                voxel_gen::pools::PoolFluid::Lava
            } else {
                voxel_gen::pools::PoolFluid::Water
            };

            // Force spawn pool: carve basin + generate diagnostics
            let (diagnostics, fluid_seeds) = {
                let mut s = store.write().unwrap();
                let density = s.density_fields.get_mut(&key).unwrap();
                let world_origin = glam::Vec3::new(
                    cx as f32 * chunk_size as f32,
                    cy as f32 * chunk_size as f32,
                    cz as f32 * chunk_size as f32,
                );
                voxel_gen::pools::force_spawn_pool(
                    density,
                    &cfg.pools,
                    world_origin,
                    cfg.seed,
                    lx,
                    ly,
                    lz,
                    pool_fluid,
                    key,
                )
            };

            // Remesh the dirty chunk
            {
                let cs = chunk_size;
                let remesh_bounds = vec![(key, 0usize, 0usize, 0usize, cs, cs, cs)];
                let mut s = store.write().unwrap();
                let meshes = s.remesh_dirty(&remesh_bounds, &cfg, world_scale);
                drop(s);

                for (mkey, mesh) in meshes {
                    let crystal_data = retrieve_crystal_data(store, mkey, cfg.voxel_scale(), world_scale);
                    let mushroom_data = retrieve_mushroom_data(store, mkey, cfg.voxel_scale(), world_scale);
                    let _ = result_tx.send(WorkerResult::ChunkMesh {
                        chunk: mkey,
                        mesh,
                        generation: 0,
                        crystal_data,
                        mushroom_data,
                        zone_descriptors: Vec::new(),
                    });
                }
            }

            // Inject fluid seeds (skip when fluid sources disabled)
            for seed in &fluid_seeds {
                if !cfg.fluid_sources_enabled {
                    continue; // skip fluid when sources disabled
                }
                let ft = match seed.fluid_type {
                    voxel_gen::pools::PoolFluid::Water => voxel_fluid::cell::FluidType::WaterPool,
                    voxel_gen::pools::PoolFluid::Lava => voxel_fluid::cell::FluidType::Lava,
                };
                let _ = fluid_event_tx.send(FluidEvent::AddFluid {
                    chunk: seed.chunk,
                    x: seed.lx,
                    y: seed.ly,
                    z: seed.lz,
                    fluid_type: ft,
                    level: 1.0,
                    is_source: seed.is_source,
                    max_flow_dist: 0, // procedural pool seed — unbounded
                });
            }

            // Run incremental seam pass
            let _ = incremental_seam_pass(key, &cfg, store, result_tx, world_scale);

            // Send diagnostics as JSON
            let json = serde_json::to_string(&diagnostics).unwrap_or_else(|_| "{}".to_string());
            let _ = result_tx.send(WorkerResult::ForceSpawnPoolComplete { json_report: json });
        }
    }
}

/// Timing breakdown from the seam pass.
struct SeamPassTimings {
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
fn incremental_seam_pass(
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
fn recompute_crystals_for_chunks(
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
fn prune_destroyed_mushrooms_for_chunks(
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
fn batched_seam_pass(
    dirty_keys: &[(i32, i32, i32)],
    cfg: &GenerationConfig,
    store: &Arc<RwLock<ChunkStore>>,
    result_tx: &Sender<WorkerResult>,
    fluid_event_tx: &Sender<FluidEvent>,
    world_scale: f32,
) {
    batched_seam_pass_inner(dirty_keys, cfg, store, result_tx, fluid_event_tx, world_scale, false);
}

fn batched_seam_pass_mine(
    dirty_keys: &[(i32, i32, i32)],
    cfg: &GenerationConfig,
    store: &Arc<RwLock<ChunkStore>>,
    result_tx: &Sender<WorkerResult>,
    fluid_event_tx: &Sender<FluidEvent>,
    world_scale: f32,
) {
    batched_seam_pass_inner(dirty_keys, cfg, store, result_tx, fluid_event_tx, world_scale, true);
}

fn batched_seam_pass_inner(
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

// ─── Path-worker loop ────────────────────────────────────────────────
//
// Dedicated thread for AI path queries. Reads from `path_rx` (which is fed by
// the FFI `voxel_path_request` call → engine `path_tx`). Each request runs
// A* against the live ChunkStore and emits a `PathComputed` result through
// the shared `result_tx` — intercepted in engine.rs `poll_result` and stashed
// into `path_results` keyed by request_id.

/// Cell factor for the path planner — one pathing cell covers a 2×2×2 voxel
/// block. See `pathing.rs::ChunkStoreGrid`.
const PATH_CELL_FACTOR: i32 = 2;

pub fn path_worker_loop(
    shutdown: Arc<AtomicBool>,
    path_rx: Receiver<WorkerRequest>,
    result_tx: Sender<WorkerResult>,
    store: Arc<RwLock<ChunkStore>>,
    config: Arc<RwLock<GenerationConfig>>,
    world_scale: f32,
) {
    while !shutdown.load(Ordering::Relaxed) {
        // Block (with timeout) on the path channel — separate from the main
        // mine/generate workers so neither gets head-of-line blocked.
        match path_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(WorkerRequest::ComputePath { request }) => {
                handle_path_request(request, &result_tx, &store, &config, world_scale);
            }
            // path_rx should only ever carry ComputePath; ignore anything else.
            Ok(_other) => {}
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }
}

fn handle_path_request(
    request: crate::pathing::PathRequestInternal,
    result_tx: &Sender<WorkerResult>,
    store: &Arc<RwLock<ChunkStore>>,
    config: &Arc<RwLock<GenerationConfig>>,
    world_scale: f32,
) {
    let request_id = request.request_id;

    // Block on read lock — fine, we're on a dedicated worker thread.
    let chunk_size = {
        let cfg = match config.read() {
            Ok(c) => c,
            Err(_) => {
                let _ = result_tx.send(WorkerResult::PathComputed {
                    request_id,
                    status: voxel_path::PathStatus::NoPath as u8,
                    nodes_ue: Vec::new(),
                });
                return;
            }
        };
        cfg.chunk_size
    };

    let store_guard = match store.read() {
        Ok(s) => s,
        Err(_) => {
            let _ = result_tx.send(WorkerResult::PathComputed {
                request_id,
                status: voxel_path::PathStatus::NoPath as u8,
                nodes_ue: Vec::new(),
            });
            return;
        }
    };

    let grid = crate::pathing::ChunkStoreGrid {
        store: &store_guard,
        chunk_size,
        cell_factor: PATH_CELL_FACTOR,
    };

    let (path_req, _mode) = crate::pathing::to_path_request(&request, PATH_CELL_FACTOR);
    let outcome = voxel_path::compute_path(&grid, path_req);

    // Drop the store guard before doing the UE conversion — keeps the read
    // lock window as short as possible.
    drop(store_guard);

    let nodes_ue = crate::pathing::nodes_to_ue(&outcome.nodes, PATH_CELL_FACTOR, world_scale);

    let _ = result_tx.send(WorkerResult::PathComputed {
        request_id,
        status: outcome.status as u8,
        nodes_ue,
    });
}
