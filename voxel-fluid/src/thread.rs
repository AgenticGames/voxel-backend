use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};

use crate::cell::{ChunkDensityCache, ChunkFluidGrid};
use crate::mesh::{mesh_fluid, BoundaryLevels};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use std::collections::VecDeque;

use crate::sim::{detect_lava_water_quench_with_scratch, equalize_horizontal, regen_sources, squeeze_excess_fluid_collect, tick_fluid, try_grow_pillow_voxel, QuenchScratch};
use crate::sim::displacement::{queue_displacement, spill_displacements};

/// Collapse-into-water ledger (2026-09-06): appended to Saved/fluid_debug.txt
/// (same convention as stress_debug.txt / collapse_log.txt) whenever a
/// displacement or wave region is active, so "the water vanished" can be
/// answered from a file instead of a guess.
fn fluid_debug(msg: String) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true).append(true)
        .open("D:/Unreal Projects/Mithril2026/Saved/fluid_debug.txt")
    {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs_f64() % 10000.0;
        let _ = writeln!(f, "[{:.2}] {}", t, msg);
    }
}

/// Total water level over every loaded fluid grid (the ledger's "how much
/// water exists" number).
fn total_water(chunks: &HashMap<(i32, i32, i32), ChunkFluidGrid>) -> f32 {
    let mut sum = 0.0f32;
    for grid in chunks.values() {
        if !grid.has_fluid { continue; }
        for c in &grid.cells {
            if c.fluid_type.is_water() { sum += c.level; }
        }
    }
    sum
}

/// Bounds on the sim tick rate (Hz). The low end keeps the tick interval a
/// finite `Duration`; the high end stops a fat-fingered menu value from
/// spinning the fluid thread flat out.
pub const MIN_TICK_RATE: f32 = 0.1;
pub const MAX_TICK_RATE: f32 = 240.0;

/// "Pool pull" — when a water cell is drained at the heat interface, also
/// drain one extra connected water cell from the network behind it via BFS.
/// Without this, the main water blob barely shrinks during the steaming
/// phase: flow refills the contact cell as fast as we drain it, so the pool's
/// *total volume* stays roughly constant. The pull propagates the
/// consumption back into the blob so the player sees water visibly receding
/// from the contact area, not just the thin flow at the edge being eaten.
///
/// Source cells are never drained (they're infinite reservoirs by design)
/// but we walk *through* them in the BFS so a chain of sources separating
/// us from a drainable pool doesn't block the pull.
fn pool_pull_drain(
    chunks: &mut HashMap<(i32, i32, i32), ChunkFluidGrid>,
    chunk_size: usize,
    start_world: (i32, i32, i32),
) {
    const POOL_PULL_BFS_LIMIT: usize = 16;
    const FACE_OFFSETS: [(i32, i32, i32); 6] = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ];
    let cs_i = chunk_size as i32;
    let mut visited: HashSet<(i32, i32, i32)> = HashSet::new();
    visited.insert(start_world);
    let mut queue: VecDeque<(i32, i32, i32)> = VecDeque::new();
    queue.push_back(start_world);
    let mut explored: usize = 0;
    while let Some(pos) = queue.pop_front() {
        explored += 1;
        if explored > POOL_PULL_BFS_LIMIT { return; }
        for &(dx, dy, dz) in &FACE_OFFSETS {
            let nw = (pos.0 + dx, pos.1 + dy, pos.2 + dz);
            if !visited.insert(nw) { continue; }
            let key = (nw.0.div_euclid(cs_i), nw.1.div_euclid(cs_i), nw.2.div_euclid(cs_i));
            let lx = nw.0.rem_euclid(cs_i) as usize;
            let ly = nw.1.rem_euclid(cs_i) as usize;
            let lz = nw.2.rem_euclid(cs_i) as usize;
            if let Some(grid) = chunks.get_mut(&key) {
                let cell = grid.get_mut(lx, ly, lz);
                if cell.level > crate::cell::MIN_LEVEL && cell.fluid_type.is_water() {
                    if !cell.is_source {
                        cell.level = 0.0;
                        grid.dirty = true;
                        return; // drained one extra cell — done
                    }
                    // Source cells: walk through but don't drain
                    queue.push_back(nw);
                }
            }
        }
    }
}

/// Per-source pillow growth state. Created when a lava source first contacts
/// water; ticked forward periodically to grow one obsidian voxel at a time
/// outward from the source, mimicking pillow lava accretion.
#[derive(Debug, Clone, Copy)]
struct PillowState {
    growth_count: u32,
    last_growth_tick: u64,
}

/// Per-quenched-voxel heat & depth info. Depth = distance (in voxel hops) from
/// the original water-side rim:
///   * 0 = outer Obsidian (where water originally touched lava)
///   * 1 = initial Scoria halo (immediate cooling shell from the BFS)
///   * 2+ = inward-growth Scoria (forms slowly as the wall thickens
///     toward equilibrium)
/// Conversion probability halves every depth level so the wall converges
/// to a stable thickness instead of consuming the chamber forever.
#[derive(Debug, Clone, Copy)]
struct HotInfo {
    expires_tick: u64,
    depth: u8,
}
use crate::sources::place_sources;
use crate::{FluidConfig, FluidEvent, FluidImportStash, FluidResult, FluidSnapshot, PendingFluidCell};

/// Main fluid simulation loop running on its own thread.
///
/// Drains events from `event_rx`, ticks the simulation at the configured rate,
/// meshes dirty chunks, and sends results through `result_tx`. Save-restore
/// fluid arrives via `import_stash` (see `FluidImportStash` — the bounded
/// event channel drops sends under the load-time flood).
pub fn fluid_sim_loop(
    shutdown: Arc<AtomicBool>,
    event_rx: Receiver<FluidEvent>,
    result_tx: Sender<FluidResult>,
    config: FluidConfig,
    import_stash: FluidImportStash,
) {
    let mut config = config;
    let mut chunks: HashMap<(i32, i32, i32), ChunkFluidGrid> = HashMap::new();
    // Lightweight density-only storage for chunks without fluid
    let mut chunk_densities: HashMap<(i32, i32, i32), ChunkDensityCache> = HashMap::new();
    // Save-load fluid restoration buffer. Entries are queued by
    // `PendingFluidLoad` events at load time and applied to the grid the
    // moment the matching chunk's density is in cache (DensityUpdate or
    // TerrainModified). Without the wait, AddFluid would land on a default
    // grid (cell_cap=1.0 everywhere) and end up with fluid in cells that
    // turn out to be solid once density arrives.
    let mut pending_fluid: HashMap<(i32, i32, i32), Vec<PendingFluidCell>> = HashMap::new();
    // Once-per-chunk guard for procedural fluid features (kind 0 = noise
    // lava, 1 = geological springs) — the worker re-sends placement events
    // on every stream-in, which used to refill every pool and resurrect
    // self-extinguished sources whenever the streaming set churned (#216).
    let mut features_placed: HashSet<((i32, i32, i32), u8)> = HashSet::new();
    // Active pillow sources — lava vent positions (world voxel coords) that
    // are growing obsidian mounds. Cleared when growth_count hits the cap or
    // no water remains within reach. See sim::try_grow_pillow_voxel.
    let mut active_pillows: HashMap<(i32, i32, i32), PillowState> = HashMap::new();
    // **Heat zone** — world voxel positions of freshly-quenched rock that
    // continue to vaporize any non-source water touching them. Refreshed
    // each tick for positions still face-adjacent to active lava (chamber-
    // wall rim with magma behind it stays hot indefinitely; outer rim
    // whose lava has retreated via inward growth cools after the timer).
    let mut hot_positions: HashMap<(i32, i32, i32), HotInfo> = HashMap::new();
    // Deterministic RNG for inward growth probability rolls — seeded so that
    // the wall grows the same way across reproducible test runs.
    let mut quench_rng = ChaCha8Rng::seed_from_u64(0xC001_5C04_C001_BABEu64);
    // Long-lived scratch for the per-tick lava↔water quench scan; lets us
    // skip allocating the 4 working HashSets + per-contact-cell BFS sets/vecs
    // every tick. Cleared internally at the start of each detect call.
    let mut quench_scratch = QuenchScratch::default();
    // Tunables. Hardcoded for now; can be promoted to FluidConfig later.
    const PILLOW_GROWTH_INTERVAL_TICKS: u64 = 150; // ~5s @ 30Hz
    const PILLOW_MAX_VOXELS: u32 = 30;
    const PILLOW_SEARCH_RADIUS: i32 = 4;
    const HEAT_COOLING_TICKS: u64 = 150;            // ~5s — how long quenched rock vaporizes water
    const HEAT_FACE_OFFSETS: [(i32, i32, i32); 6] = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ];
    // Inward-growth tunables.
    // Base probability for converting a lava cell at depth-1 (immediately
    // behind the initial scoria layer). Halves every depth level beyond.
    // 1/30 ≈ ~3% per tick → at 30Hz, a depth-2 lava cell converts in ~2s of
    // continuous contact; depth-3 in ~8s; depth-4+ effectively never.
    const INWARD_GROWTH_BASE_PROB: f32 = 1.0 / 30.0;
    /// Maximum inward depth — at this depth conversion probability is
    /// already negligible; treat as hard stop so the wall stabilizes.
    /// Lowered from 4 → 3 for the demo: shallower wall, less lava consumed,
    /// outer rock cools (stops vaporizing nearby water) sooner once the
    /// lava behind the rim retreats past the cap.
    const INWARD_GROWTH_MAX_DEPTH: u8 = 3;
    let chunk_size = config.chunk_size;

    let mut last_tick = Instant::now();
    let mut last_wave_tick = Instant::now();
    let mut wave_ticks: u64 = 0;
    let wave_interval = Duration::from_secs_f32(1.0 / crate::sim::wave::WAVE_TICK_HZ);
    let mut tick_count: u64 = 0;
    // Track chunks that have active (non-empty) fluid meshes so we can send
    // an empty mesh when they transition to empty (e.g. after DrainLavaChunks).
    let mut active_fluid_meshes: HashSet<(i32, i32, i32)> = HashSet::new();

    while !shutdown.load(Ordering::Relaxed) {
        // Drain all pending events
        loop {
            match event_rx.try_recv() {
                Ok(event) => handle_event(event, &mut chunks, &mut chunk_densities, &mut pending_fluid, &mut features_placed, chunk_size, &mut config, &result_tx, &mut active_fluid_meshes),
                Err(_) => break,
            }
        }

        // Save-restore imports (guaranteed delivery — see FluidImportStash).
        drain_import_stash(&import_stash, &mut chunks, &mut chunk_densities, &mut pending_fluid, &mut features_placed, chunk_size, &mut config, &result_tx, &mut active_fluid_meshes);

        // Rate knobs are read fresh every iteration (not latched before the
        // loop) so an `UpdateFluidRates` event actually changes sim speed on
        // a live world. Clamped because these come straight from a JSON file
        // the codex menu writes — a 0 or negative tick_rate would make
        // `from_secs_f32` panic on an infinite duration.
        let tick_interval =
            Duration::from_secs_f32(1.0 / config.tick_rate.clamp(MIN_TICK_RATE, MAX_TICK_RATE));
        let lava_divisor = config.lava_tick_divisor.max(1) as u64;

        let now = Instant::now();

        // ── Collapse impact fast path (2026-09-06) ──
        // Displacement spill + wave regions run at WAVE_TICK_HZ on their own
        // clock and remesh only the chunks they touched, so a crest moves
        // and renders at 30 Hz even while the pool automaton ticks at 3.
        if (crate::sim::wave::region_count() > 0
            || crate::sim::displacement::pending_count() > 0
            || crate::sim::wave::foam_active())
            && now.duration_since(last_wave_tick) >= wave_interval
        {
            last_wave_tick = now;
            wave_ticks += 1;
            let mut wave_dirty = spill_displacements(&mut chunks, chunk_size);
            wave_dirty.extend(crate::sim::wave::decay_foam(&mut chunks));
            wave_dirty.extend(crate::sim::wave::step_waves(&mut chunks, chunk_size));
            if !wave_dirty.is_empty() {
                mesh_and_send(&mut chunks, &wave_dirty, chunk_size, &config, &result_tx, &mut active_fluid_meshes);
            }
            if wave_ticks % 15 == 0 {
                let (mut foam_cells, mut foam_max, mut foam_grids) = (0usize, 0.0f32, 0usize);
                for g in chunks.values() {
                    if g.foam.is_empty() { continue; }
                    foam_grids += 1;
                    for &f in &g.foam { if f > 0.1 { foam_cells += 1; } if f > foam_max { foam_max = f; } }
                }
                fluid_debug(format!(
                    "wave tick {}: water in grids {:.1} | regions {} | displacement pending {} dropped {:.2} | wave overflow-lost {:.2} deficit-created {:.2} | foam grids {} cells>0.1 {} max {:.2} | remeshed {}",
                    wave_ticks, total_water(&chunks),
                    crate::sim::wave::region_count(),
                    crate::sim::displacement::pending_count(),
                    crate::sim::displacement::dropped_total(),
                    crate::sim::wave::lost_overflow_total(),
                    crate::sim::wave::created_deficit_total(),
                    foam_grids, foam_cells, foam_max, wave_dirty.len()));
            }
        }

        // Check if it's time for a tick
        if now.duration_since(last_tick) < tick_interval {
            std::thread::sleep(Duration::from_millis(1));
            continue;
        }
        last_tick = now;
        tick_count += 1;

        if chunks.is_empty() {
            continue;
        }

        // Regenerate sources
        regen_sources(&mut chunks);

        // Tick water every tick with multiple substeps, lava every N ticks (single step)
        let is_lava_tick = tick_count % lava_divisor == 0;
        let substeps = config.water_substeps.max(1) as usize;
        let mut dirty_water = HashSet::new();
        // (Collapse displacement spill + impact waves run on the fast path
        // above at WAVE_TICK_HZ; equalize below skips wave-owned columns.)
        // Equalize first: set flat baseline, then slope flow gets the final word
        // to create gradients toward drains (prevents equalization from undoing drainage)
        let dirty_eq = equalize_horizontal(&mut chunks, chunk_size, false);
        dirty_water.extend(dirty_eq);
        for i in 0..substeps {
            let decrement_grace = i == substeps - 1; // only on last substep
            let dirty = tick_fluid(&mut chunks, &chunk_densities, chunk_size, false, &config, decrement_grace);
            dirty_water.extend(dirty);
        }

        let dirty_lava = if is_lava_tick {
            tick_fluid(&mut chunks, &chunk_densities, chunk_size, true, &config, true)
        } else {
            HashSet::new()
        };

        // ── Live lava↔water quench ───────────────────────────────────────
        // Detect contact zones and build a structured plan (obsidian rim +
        // volume-aware scoria halo + drained water + pillow source registry).
        let plan = detect_lava_water_quench_with_scratch(&chunks, &mut quench_scratch);

        // Locally drain the lava cells we're turning into solid voxels and
        // the water cells we're vaporizing — keeps the fluid grid consistent
        // with what the worker is about to write into density_fields.
        //
        // 2026-08-25 ("i made scoria and it was still ouching me"): quenched
        // cells also clear their RENDER state — the same frozen-ribbon class
        // the montage drain hit (render_level/flux_ema/stream_mark mesh at
        // STREAM_FLOOR with no raw-level gate and only decay inside
        // tick_chunk, which a chunk whose lava just all solidified may never
        // meaningfully enter again). Left alone, the quenched rim kept a
        // ghost lava mesh over the new scoria and the #248 contact burn read
        // it as live lava.
        for (key, x, y, z) in plan.obsidian.iter().chain(plan.scoria.iter()) {
            if let Some(grid) = chunks.get_mut(key) {
                let idx = grid.index(*x, *y, *z);
                grid.cells[idx].level = 0.0;
                grid.cells[idx].is_source = false;
                if idx < grid.render_level.len() { grid.render_level[idx] = 0.0; }
                if idx < grid.flux_ema.len() { grid.flux_ema[idx] = 0.0; }
                if idx < grid.stream_mark.len() { grid.stream_mark[idx] = false; }
                if idx < grid.mesh_sticky.len() { grid.mesh_sticky[idx] = false; }
                if idx < grid.influx_hold.len() { grid.influx_hold[idx] = 0; }
                if idx < grid.momentum.len() { grid.momentum[idx] = [0.0, 0.0]; }
                grid.dirty = true;
            }
        }
        for (key, x, y, z) in &plan.drained_water {
            if let Some(grid) = chunks.get_mut(key) {
                let idx = grid.index(*x, *y, *z);
                grid.cells[idx].level = 0.0;
                if idx < grid.render_level.len() { grid.render_level[idx] = 0.0; }
                if idx < grid.mesh_sticky.len() { grid.mesh_sticky[idx] = false; }
                grid.dirty = true;
            }
        }

        // Register any newly-contacted lava sources for pillow growth.
        for src in &plan.pillow_sources {
            active_pillows.entry(*src).or_insert(PillowState {
                growth_count: 0,
                last_growth_tick: tick_count,
            });
        }

        // Aggregated obsidian + scoria positions to send to the worker this
        // tick — starts with contact rim + halo, pillow growths get appended
        // below, and inward-growth scoria gets appended after that.
        let mut quench_obsidian = plan.obsidian.clone();
        let mut quench_scoria = plan.scoria.clone();
        let quench_drained = plan.drained_water.clone();

        // ── Pillow growth tick ────────────────────────────────────────────
        // Each active pillow grows by one obsidian voxel every PILLOW_GROWTH_INTERVAL_TICKS,
        // up to PILLOW_MAX_VOXELS per source. The growth point is the closest
        // water cell to the source — gives an outward, bulbous accretion shape.
        let mut pillows_to_drop: Vec<(i32, i32, i32)> = Vec::new();
        for (src_pos, state) in active_pillows.iter_mut() {
            if state.growth_count >= PILLOW_MAX_VOXELS {
                pillows_to_drop.push(*src_pos);
                continue;
            }
            if tick_count.saturating_sub(state.last_growth_tick) < PILLOW_GROWTH_INTERVAL_TICKS {
                continue;
            }
            if let Some(addr) = try_grow_pillow_voxel(*src_pos, &mut chunks, chunk_size, PILLOW_SEARCH_RADIUS) {
                quench_obsidian.push(addr);
                state.growth_count += 1;
                state.last_growth_tick = tick_count;
            }
            // If no water in reach this tick, the pillow stalls but stays
            // registered — water may flow back near the vent later.
        }
        for k in pillows_to_drop {
            active_pillows.remove(&k);
        }

        // ── Heat zone bookkeeping ────────────────────────────────────────
        // Add this tick's quench voxels to the hot zone (world coords) with
        // depth info. Obsidian (rim contact + pillow growth) is depth 0;
        // initial scoria halo is depth 1.
        let cs_i = chunk_size as i32;
        let cool_at = tick_count + HEAT_COOLING_TICKS;
        for (key, lx, ly, lz) in &quench_obsidian {
            let wx = key.0 * cs_i + *lx as i32;
            let wy = key.1 * cs_i + *ly as i32;
            let wz = key.2 * cs_i + *lz as i32;
            // Don't overwrite a deeper-depth entry with depth 0 if one exists
            // (shouldn't happen in practice but defensive).
            hot_positions.entry((wx, wy, wz))
                .and_modify(|info| {
                    info.expires_tick = cool_at;
                    info.depth = info.depth.min(0);
                })
                .or_insert(HotInfo { expires_tick: cool_at, depth: 0 });
        }
        for (key, lx, ly, lz) in &quench_scoria {
            let wx = key.0 * cs_i + *lx as i32;
            let wy = key.1 * cs_i + *ly as i32;
            let wz = key.2 * cs_i + *lz as i32;
            hot_positions.entry((wx, wy, wz))
                .and_modify(|info| {
                    info.expires_tick = cool_at;
                    info.depth = info.depth.min(1);
                })
                .or_insert(HotInfo { expires_tick: cool_at, depth: 1 });
        }

        // ── Inward-growth pass ───────────────────────────────────────────
        // Each tick, walk the hot rim; for every face-neighbor lava cell,
        // roll for conversion to new scoria at depth = parent_depth + 1.
        // Probability halves with each additional depth so the wall reaches
        // a stable thickness (~3-4 voxels) instead of consuming the chamber.
        //
        // We iterate hot positions (small bounded set around contacts) not
        // all lava cells, so cost is O(N_hot * 6) per tick — trivial.
        let mut inward_new_scoria: Vec<((i32, i32, i32), usize, usize, usize)> = Vec::new();
        let mut inward_new_depths: Vec<((i32, i32, i32), u8)> = Vec::new();
        let mut converted_this_tick: HashSet<(i32, i32, i32)> = HashSet::new();
        let hot_snapshot: Vec<((i32, i32, i32), HotInfo)> =
            hot_positions.iter().map(|(k, v)| (*k, *v)).collect();
        for ((wx, wy, wz), info) in &hot_snapshot {
            let new_depth = info.depth.saturating_add(1);
            if new_depth > INWARD_GROWTH_MAX_DEPTH {
                continue;
            }
            // Halves per depth step beyond the contact: depth 2 → BASE/2,
            // depth 3 → BASE/4, depth 4 → BASE/8 (~0.4% per tick).
            let shift = new_depth.saturating_sub(1) as u32;
            let denom = 1u32.checked_shl(shift).unwrap_or(u32::MAX) as f32;
            let prob = INWARD_GROWTH_BASE_PROB / denom.max(1.0);

            for &(dx, dy, dz) in &HEAT_FACE_OFFSETS {
                let nw = (wx + dx, wy + dy, wz + dz);
                if converted_this_tick.contains(&nw) { continue; }
                if hot_positions.contains_key(&nw) { continue; }  // already solid
                let key = (nw.0.div_euclid(cs_i), nw.1.div_euclid(cs_i), nw.2.div_euclid(cs_i));
                let lx = nw.0.rem_euclid(cs_i) as usize;
                let ly = nw.1.rem_euclid(cs_i) as usize;
                let lz = nw.2.rem_euclid(cs_i) as usize;
                let Some(grid) = chunks.get(&key) else { continue; };
                let cell = grid.get(lx, ly, lz);
                if cell.level < crate::cell::MIN_LEVEL || !cell.fluid_type.is_lava() {
                    continue;
                }
                if cell.is_source { continue; }  // sources never convert (pillow handles them)
                if quench_rng.gen::<f32>() < prob {
                    converted_this_tick.insert(nw);
                    inward_new_scoria.push((key, lx, ly, lz));
                    inward_new_depths.push((nw, new_depth));
                }
            }
        }
        // Apply: drain the converted lava cells locally + register hot info
        // so the next inward-growth tick uses the deeper depth.
        for (key, lx, ly, lz) in &inward_new_scoria {
            if let Some(grid) = chunks.get_mut(key) {
                grid.get_mut(*lx, *ly, *lz).level = 0.0;
                grid.dirty = true;
            }
        }
        for (pos, depth) in inward_new_depths {
            hot_positions.insert(pos, HotInfo { expires_tick: cool_at, depth });
        }
        // Send the newly-grown scoria to the worker too.
        quench_scoria.extend(inward_new_scoria);

        // Drain non-source water face-adjacent to any hot position +
        // propagate one cell of consumption back into the connected pool
        // (pool_pull_drain). Without the pull the main blob stays at the
        // same volume — flow refills the contact cell as fast as it drains,
        // so only the thin flow looks like it's being eaten. The pull pulls
        // a connected water cell out of the back of the blob so the player
        // sees the pool itself receding during the steaming phase.
        let hot_snapshot_for_drain: Vec<(i32, i32, i32)> =
            hot_positions.keys().copied().collect();
        for (wx, wy, wz) in hot_snapshot_for_drain {
            for &(dx, dy, dz) in &HEAT_FACE_OFFSETS {
                let (nx, ny, nz) = (wx + dx, wy + dy, wz + dz);
                let key = (nx.div_euclid(cs_i), ny.div_euclid(cs_i), nz.div_euclid(cs_i));
                let lx = nx.rem_euclid(cs_i) as usize;
                let ly = ny.rem_euclid(cs_i) as usize;
                let lz = nz.rem_euclid(cs_i) as usize;
                let mut drained_here = false;
                if let Some(grid) = chunks.get_mut(&key) {
                    let cell = grid.get_mut(lx, ly, lz);
                    if cell.level > crate::cell::MIN_LEVEL
                        && cell.fluid_type.is_water()
                        && !cell.is_source
                    {
                        cell.level = 0.0;
                        grid.dirty = true;
                        drained_here = true;
                    }
                }
                if drained_here {
                    pool_pull_drain(&mut chunks, chunk_size, (nx, ny, nz));
                }
            }
        }

        // Refresh hot positions still face-adjacent to active lava (anywhere)
        // — these are chamber-wall rims with magma behind them and should
        // stay hot indefinitely. Positions whose lava has been fully sealed
        // off or converted via inward growth will not get refreshed and will
        // eventually cool out, letting water flow over them.
        let now = tick_count;
        let positions_to_check: Vec<(i32, i32, i32)> = hot_positions.keys().copied().collect();
        for pos in positions_to_check {
            let (wx, wy, wz) = pos;
            for &(dx, dy, dz) in &HEAT_FACE_OFFSETS {
                let (nx, ny, nz) = (wx + dx, wy + dy, wz + dz);
                let key = (nx.div_euclid(cs_i), ny.div_euclid(cs_i), nz.div_euclid(cs_i));
                let lx = nx.rem_euclid(cs_i) as usize;
                let ly = ny.rem_euclid(cs_i) as usize;
                let lz = nz.rem_euclid(cs_i) as usize;
                if let Some(grid) = chunks.get(&key) {
                    let cell = grid.get(lx, ly, lz);
                    if cell.level > crate::cell::MIN_LEVEL && cell.fluid_type.is_lava() {
                        if let Some(info) = hot_positions.get_mut(&pos) {
                            info.expires_tick = cool_at;
                        }
                        break;
                    }
                }
            }
        }

        // Expire cooled-down hot positions.
        hot_positions.retain(|_, info| info.expires_tick > now);

        // Send the per-tick quench plan to the worker (engine forwards as
        // a WorkerRequest). Voxels are written into density_fields and chunks
        // are remeshed there.
        if !quench_obsidian.is_empty() || !quench_scoria.is_empty() || !quench_drained.is_empty() {
            let _ = result_tx.send(FluidResult::LavaQuench {
                obsidian: quench_obsidian,
                scoria: quench_scoria,
                drained_water: quench_drained,
            });
        }

        // Collect all dirty chunks
        let mut all_dirty: HashSet<(i32, i32, i32)> = HashSet::new();
        all_dirty.extend(&dirty_water);
        all_dirty.extend(&dirty_lava);
        // Also include chunks marked dirty by events
        for (&k, grid) in &mut chunks {
            if grid.dirty {
                all_dirty.insert(k);
            }
        }

        mesh_and_send(&mut chunks, &all_dirty, chunk_size, &config, &result_tx, &mut active_fluid_meshes);
    }
}

/// Mesh every chunk in `dirty` (plus wet-seam neighbours) and send the
/// results. Factored out of the tick loop (2026-09-06) so the collapse-wave
/// fast path can remesh its few chunks between pool ticks.
fn mesh_and_send(
    chunks: &mut HashMap<(i32, i32, i32), ChunkFluidGrid>,
    dirty: &HashSet<(i32, i32, i32)>,
    chunk_size: usize,
    config: &FluidConfig,
    result_tx: &Sender<FluidResult>,
    active_fluid_meshes: &mut HashSet<(i32, i32, i32)>,
) {
    // A dirty chunk's render field feeds its neighbors' boundary
    // sampling. When the shared face is wet on either side, the neighbor
    // must re-mesh THIS pass too, or the two sides of the seam show
    // different ticks (falls "skip a line" at chunk boundaries).
    let mut mesh_set = dirty.clone();
    for key in dirty {
        let dirs: [((i32, i32, i32), usize, bool); 6] = [
            ((1, 0, 0), 0, true),
            ((-1, 0, 0), 0, false),
            ((0, 1, 0), 1, true),
            ((0, -1, 0), 1, false),
            ((0, 0, 1), 2, true),
            ((0, 0, -1), 2, false),
        ];
        for (d, axis, hi) in dirs {
            let nkey = (key.0 + d.0, key.1 + d.1, key.2 + d.2);
            if mesh_set.contains(&nkey) {
                continue;
            }
            let Some(nbr) = chunks.get(&nkey) else { continue };
            let me = chunks.get(key).unwrap();
            if face_row_wet(me, axis, hi, chunk_size) || face_row_wet(nbr, axis, !hi, chunk_size) {
                mesh_set.insert(nkey);
            }
        }
    }

    // Phase 1: refresh render state (EMA field + ribbon/fringe flags —
    // or legacy hysteresis) for EVERY chunk being meshed, BEFORE any
    // boundary sampling. Updating mid-loop made seam continuity depend
    // on HashSet iteration order: one side sampled the other one tick
    // stale.
    for key in &mesh_set {
        if let Some(grid) = chunks.get_mut(key) {
            grid.update_render_field(
                config.mesh_sticky_release,
                config.mesh_flux_render,
                config.mesh_stream_ribbon,
            );
        }
    }

    // Phase 2: mesh and send.
    for key in &mesh_set {
        let boundary = build_boundary_levels(*key, chunks, chunk_size);
        if let Some(grid) = chunks.get_mut(key) {
            let mesh = mesh_fluid(grid, &boundary, config);
            grid.dirty = false;

            if !mesh.positions.is_empty() {
                active_fluid_meshes.insert(*key);
                let _ = result_tx.send(FluidResult::FluidMesh {
                    chunk: *key,
                    mesh,
                });
            } else if active_fluid_meshes.remove(key) {
                // Was previously non-empty — send empty mesh to clear visual
                let _ = result_tx.send(FluidResult::FluidMesh {
                    chunk: *key,
                    mesh,
                });
            }
        }
    }
}

fn handle_event(
    event: FluidEvent,
    chunks: &mut HashMap<(i32, i32, i32), ChunkFluidGrid>,
    chunk_densities: &mut HashMap<(i32, i32, i32), ChunkDensityCache>,
    pending_fluid: &mut HashMap<(i32, i32, i32), Vec<PendingFluidCell>>,
    features_placed: &mut HashSet<((i32, i32, i32), u8)>,
    chunk_size: usize,
    config: &mut FluidConfig,
    result_tx: &Sender<FluidResult>,
    active_fluid_meshes: &mut HashSet<(i32, i32, i32)>,
) {
    // features_placed kinds: 0 = noise-lava sources, 1 = geological springs.
    const FEATURE_SOURCES: u8 = 0;
    const FEATURE_SPRINGS: u8 = 1;
    match event {
        FluidEvent::DensityUpdate { chunk, densities } => {
            // Store density in lightweight cache only — do NOT create a full grid
            let cache = chunk_densities
                .entry(chunk)
                .or_insert_with(|| ChunkDensityCache::new(chunk_size));
            cache.update_density(&densities);

            // If a grid already exists (fluid was placed before density arrived), update it too
            if let Some(grid) = chunks.get_mut(&chunk) {
                grid.update_density(&densities);
                grid.dirty = true;
            }

            // Drain any save-load pending fluid for this chunk now that the
            // density (and therefore cell_capacity) is current.
            apply_pending_fluid(chunks, chunk_densities, pending_fluid, chunk, chunk_size, config);
        }
        FluidEvent::PlaceSources { chunk } => {
            // ONCE per chunk per session (2026-08-04, bug #216): the worker
            // re-sends PlaceSources on EVERY stream-in, and re-planting
            // refilled every noise-lava pool to full in one tick ("all 3
            // pools refilled at once" whenever the player's streaming set
            // churned — e.g. flying under the world), resurrected
            // self-extinguished sources, and the overfill transient burst
            // basins that were otherwise holding. Chunks restored from a
            // save are marked placed by PendingFluidLoad so loaded state is
            // never stomped either.
            if !features_placed.insert((chunk, FEATURE_SOURCES)) {
                return;
            }
            // Only create grid if density exists and sources are actually placed
            ensure_grid(chunks, chunk_densities, chunk, chunk_size);
            if let Some(grid) = chunks.get_mut(&chunk) {
                place_sources(grid, chunk, chunk_size, config);
            }
        }
        FluidEvent::TerrainModified { chunk, densities } => {
            // Update density cache
            let cache = chunk_densities
                .entry(chunk)
                .or_insert_with(|| ChunkDensityCache::new(chunk_size));
            cache.update_density(&densities);

            // If grid exists, update its density and squeeze excess. What the
            // squeeze cannot place locally is queued as a displacement and
            // re-injected around the impact over the next ticks (2026-09-06)
            // instead of evaporating - rock landing in a pool used to eat it.
            let before = if chunks.contains_key(&chunk) { Some(total_water(chunks)) } else { None };
            let mut lost_total = 0.0f32;
            if let Some(grid) = chunks.get_mut(&chunk) {
                grid.update_density(&densities);
                let lost = squeeze_excess_fluid_collect(grid);
                grid.dirty = true;
                lost_total = lost.iter().map(|r| r.lost).sum();
                queue_displacement(chunk, chunk_size, &lost);
            }
            if lost_total > 0.0 {
                let after = total_water(chunks);
                fluid_debug(format!(
                    "terrain {:?}: squeeze displaced {:.2} cells | water in grids {:.1} -> {:.1} (displacement pending {}, wave regions {})",
                    chunk, lost_total, before.unwrap_or(0.0), after,
                    crate::sim::displacement::pending_count(), crate::sim::wave::region_count()));
            }

            // Drain any save-load pending fluid for this chunk now that the
            // density is current.
            apply_pending_fluid(chunks, chunk_densities, pending_fluid, chunk, chunk_size, config);
        }
        FluidEvent::ChunkUnloaded { chunk } => {
            // Preserve worth-saving fluid into pending_fluid BEFORE dropping
            // the grid. When the chunk later re-streams, the DensityUpdate
            // event drains pending_fluid back into the new grid via
            // apply_pending_fluid — exactly the chunk_store
            // preserved_snapshots lifecycle, but for fluid.
            //
            // Without this, walking out of a region and back loses every
            // brush-placed lava pool / water source: ChunkUnloaded would
            // simply remove the grid and the next DensityUpdate would land
            // on an empty grid with no source to repopulate it.
            //
            // SnapshotRequest synthesizes cells for unloaded chunks from
            // this same map so a save taken while the chunk is unloaded
            // still captures the fluid.
            if let Some(grid) = chunks.get(&chunk) {
                let mut preserved: Vec<PendingFluidCell> = Vec::new();
                for (idx, cell) in grid.cells.iter().enumerate() {
                    if cell.is_source || cell.level > crate::cell::MIN_LEVEL {
                        preserved.push(PendingFluidCell {
                            idx: idx as u32,
                            fluid_type: cell.fluid_type,
                            level: cell.level,
                            is_source: cell.is_source,
                            max_flow_dist: cell.max_flow_dist,
                        });
                    }
                }
                if !preserved.is_empty() {
                    // Replace any stale pending entries (would only exist if
                    // a save-load PendingFluidLoad raced with this unload —
                    // the live grid is the freshest copy).
                    pending_fluid.insert(chunk, preserved);
                }
            }
            chunks.remove(&chunk);
            chunk_densities.remove(&chunk);
        }
        FluidEvent::SnapshotRequest { reply_tx } => {
            // Loaded chunks: clone the live grid cells.
            let mut snapshot_chunks: HashMap<(i32, i32, i32), Vec<crate::cell::FluidCell>> =
                chunks.iter().map(|(&k, g)| (k, g.cells.clone())).collect();
            // Unloaded chunks with preserved fluid: synthesize a sparse cell
            // array from pending_fluid so save export sees them too. This is
            // what makes "place fluid → walk away → save → reload" survive
            // even when the player is far from where the fluid was placed.
            let total = chunk_size * chunk_size * chunk_size;
            for (chunk_key, pending) in pending_fluid.iter() {
                if snapshot_chunks.contains_key(chunk_key) { continue; }
                let mut cells = vec![crate::cell::FluidCell::default(); total];
                for p in pending {
                    let idx = p.idx as usize;
                    if idx < total {
                        cells[idx].fluid_type = p.fluid_type;
                        cells[idx].level = p.level;
                        cells[idx].is_source = p.is_source;
                        cells[idx].max_flow_dist = p.max_flow_dist;
                    }
                }
                snapshot_chunks.insert(*chunk_key, cells);
            }
            let snapshot = FluidSnapshot {
                chunks: snapshot_chunks,
                chunk_size,
            };
            let _ = reply_tx.send(snapshot);
        }
        FluidEvent::DrainLavaChunks { chunks: drain_chunks } => {
            let mut live_chunks = 0u32;
            let mut purged_pending = 0u32;
            let mut cleared_unloaded = 0u32;
            for chunk_key in drain_chunks {
                if let Some(grid) = chunks.get_mut(&chunk_key) {
                    let total = grid.size * grid.size * grid.size;
                    // Dormancy evaporates ALL standing fluid — water included
                    // (user directive 2026-08-20: 1.25Ma of geological time
                    // leaves neither lava nor pools). No level guard: a
                    // fall-feeding source's post-tick level is 0.0 (free-fall
                    // gulps the whole refill), so gating on level left
                    // is_source alive and the stream re-pumped seconds after
                    // the "drained" montage (World_1991).
                    for idx in 0..total {
                        grid.cells[idx].level = 0.0;
                        grid.cells[idx].is_source = false;
                        grid.cells[idx].grace_ticks = 0;
                    }
                    // Render-state clear (round 2): ribbon/fringe stream
                    // marks mesh at STREAM_FLOOR with no raw-level gate, and
                    // they only decay inside tick_chunk — which a drained
                    // empty chunk never enters again. Left alone, pool rims
                    // and falls (the flux-carrying cells) freeze as burning
                    // ghost meshes.
                    grid.render_level.fill(0.0);
                    grid.flux_ema.fill(0.0);
                    grid.stream_mark.fill(false);
                    grid.mesh_sticky.fill(false);
                    grid.influx_hold.fill(0);
                    grid.momentum.fill([0.0, 0.0]);
                    grid.has_fluid = false;
                    grid.has_lava = false;
                    grid.has_sources = false;
                    grid.dirty = true;
                    live_chunks += 1;
                }
                // 2026-08-13 ("lava popped back and BURNED me"): unloaded
                // chunks keep their fluid in pending_fluid at FULL level with
                // LIVE is_source flags — the drain only touched live grids, so
                // any chunk fluid-unloaded at drain time (montage pin/unpin
                // churn guarantees some) resurrected its whole pool, sources
                // included, the moment it re-streamed. Evaporate the stash too.
                if let Some(pending) = pending_fluid.remove(&chunk_key) {
                    purged_pending += pending.len() as u32;
                }
                // 2026-08-20 residual of the above: with no live grid there is
                // nothing to re-mesh, so the dirty→mesh-pass→empty-send path
                // never fires and the UE actor keeps its last non-empty fluid
                // mesh — visible, and burning via #248 contact if lava — until
                // the actor recycles. Send the clearing empty mesh from here;
                // the tracker gate keeps never-meshed chunks from spamming
                // empties.
                if !chunks.contains_key(&chunk_key) && active_fluid_meshes.remove(&chunk_key) {
                    let _ = result_tx.send(FluidResult::FluidMesh {
                        chunk: chunk_key,
                        mesh: crate::mesh::FluidMeshData {
                            positions: Vec::new(),
                            normals: Vec::new(),
                            fluid_types: Vec::new(),
                            indices: Vec::new(),
                            uvs: Vec::new(),
                            flow_directions: Vec::new(),
                            foam: Vec::new(),
                        },
                    });
                    cleared_unloaded += 1;
                }
            }
            eprintln!(
                "[FLUID-DRAIN] evaporated {} live chunks (all fluid), purged {} stashed cells, cleared {} stale meshes on unloaded chunks",
                live_chunks, purged_pending, cleared_unloaded
            );
        }
        FluidEvent::RemeshAllFluid => {
            // Dirty every live grid — the mesh pass re-sends non-empty truth
            // and sends the clearing empty for tracked grids that emptied.
            let mut swept = 0u32;
            for grid in chunks.values_mut() {
                grid.dirty = true;
                swept += 1;
            }
            // Tracked chunks with NO live grid can never re-mesh on their own;
            // clear them explicitly so a wiped UE component doesn't stay blank
            // while the tracker still claims it has a mesh.
            let gridless: Vec<(i32, i32, i32)> = active_fluid_meshes
                .iter()
                .filter(|k| !chunks.contains_key(k))
                .copied()
                .collect();
            let cleared = gridless.len();
            for chunk_key in gridless {
                active_fluid_meshes.remove(&chunk_key);
                let _ = result_tx.send(FluidResult::FluidMesh {
                    chunk: chunk_key,
                    mesh: crate::mesh::FluidMeshData {
                        positions: Vec::new(),
                        normals: Vec::new(),
                        fluid_types: Vec::new(),
                        indices: Vec::new(),
                        uvs: Vec::new(),
                        flow_directions: Vec::new(),
                        foam: Vec::new(),
                    },
                });
            }
            eprintln!(
                "[FLUID-REMESH-ALL] dirty-swept {} grids, cleared {} tracked gridless meshes",
                swept, cleared
            );
        }
        FluidEvent::PlaceGeologicalSprings { chunk, springs } => {
            // Same once-per-chunk guard as PlaceSources (worker re-sends on
            // every stream-in), keyed separately so the two events don't
            // lock each other out.
            if !features_placed.insert((chunk, FEATURE_SPRINGS)) {
                return;
            }
            ensure_grid(chunks, chunk_densities, chunk, chunk_size);
            if let Some(grid) = chunks.get_mut(&chunk) {
                for (lx, ly, lz, level, fluid_type_u8) in springs {
                    let xu = lx as usize;
                    let yu = ly as usize;
                    let zu = lz as usize;
                    if xu < chunk_size && yu < chunk_size && zu < chunk_size
                        && grid.cell_capacity(xu, yu, zu) > crate::cell::MIN_LEVEL
                        && !grid.is_mostly_solid(xu, yu, zu, config.solid_corner_threshold)
                    {
                        let cap = grid.cell_capacity(xu, yu, zu);
                        let ft = crate::cell::FluidType::from_u8(fluid_type_u8);
                        let cell = grid.get_mut(xu, yu, zu);
                        cell.fluid_type = ft;
                        cell.level = level.min(crate::cell::MAX_LEVEL).min(cap);
                        cell.is_source = true; // geological springs are infinite sources
                        grid.dirty = true;
                        grid.has_fluid = true;
                        grid.has_sources = true;
                        if ft.is_lava() {
                            grid.has_lava = true;
                        }
                    }
                }
            }
        }
        FluidEvent::PlacePipeLava { chunk, cells } => {
            // Once-per-chunk like PlaceSources/springs (kind 2): the worker
            // re-sends on every stream-in; re-adding refilled vents to full
            // and resurrected self-extinguished ones (#216 refill class).
            const FEATURE_PIPE_LAVA: u8 = 2;
            if !features_placed.insert((chunk, FEATURE_PIPE_LAVA)) {
                return;
            }
            ensure_grid(chunks, chunk_densities, chunk, chunk_size);
            if let Some(grid) = chunks.get_mut(&chunk) {
                for (x, y, z, level) in cells {
                    let (xu, yu, zu) = (x as usize, y as usize, z as usize);
                    if xu < chunk_size && yu < chunk_size && zu < chunk_size
                        && grid.cell_capacity(xu, yu, zu) > crate::cell::MIN_LEVEL
                        && !grid.is_mostly_solid(xu, yu, zu, config.solid_corner_threshold)
                    {
                        let cap = grid.cell_capacity(xu, yu, zu);
                        let cell = grid.get_mut(xu, yu, zu);
                        cell.fluid_type = crate::cell::FluidType::Lava;
                        cell.level = level.min(crate::cell::MAX_LEVEL).min(cap);
                        cell.is_source = true;
                        cell.hops_from_source = 0;
                        cell.max_flow_dist = 12; // geological bound, like springs
                        grid.dirty = true;
                        grid.has_fluid = true;
                        grid.has_lava = true;
                        grid.has_sources = true;
                    }
                }
            }
        }
        FluidEvent::PlaceSeedFluids { chunk, cells } => {
            // Once-per-chunk (kind 3): region re-generation after store
            // eviction re-runs the slow path and re-sends gen-time seeds —
            // without the guard every pool/formation/zone basin snapped back
            // to gen-fresh full on return flights (#216).
            const FEATURE_SEEDS: u8 = 3;
            if !features_placed.insert((chunk, FEATURE_SEEDS)) {
                return;
            }
            ensure_grid(chunks, chunk_densities, chunk, chunk_size);
            if let Some(grid) = chunks.get_mut(&chunk) {
                for (x, y, z, ft_u8, is_source, max_flow_dist) in cells {
                    let (xu, yu, zu) = (x as usize, y as usize, z as usize);
                    if xu < chunk_size && yu < chunk_size && zu < chunk_size
                        && grid.cell_capacity(xu, yu, zu) > crate::cell::MIN_LEVEL
                        && !grid.is_mostly_solid(xu, yu, zu, config.solid_corner_threshold)
                    {
                        let cap = grid.cell_capacity(xu, yu, zu);
                        let ft = crate::cell::FluidType::from_u8(ft_u8);
                        let cell = grid.get_mut(xu, yu, zu);
                        cell.fluid_type = ft;
                        cell.level = crate::cell::MAX_LEVEL.min(cap);
                        cell.is_source = is_source;
                        if is_source {
                            cell.hops_from_source = 0;
                            cell.max_flow_dist = max_flow_dist;
                        }
                        grid.dirty = true;
                        grid.has_fluid = true;
                        grid.has_sources |= is_source;
                        if ft.is_lava() {
                            grid.has_lava = true;
                        }
                    }
                }
            }
        }
        FluidEvent::AddFluid { chunk, x, y, z, fluid_type, level, is_source, max_flow_dist } => {
            ensure_grid(chunks, chunk_densities, chunk, chunk_size);
            if let Some(grid) = chunks.get_mut(&chunk) {
                let xu = x as usize;
                let yu = y as usize;
                let zu = z as usize;
                if xu < chunk_size && yu < chunk_size && zu < chunk_size
                    && grid.cell_capacity(xu, yu, zu) > crate::cell::MIN_LEVEL
                    && !grid.is_mostly_solid(xu, yu, zu, config.solid_corner_threshold)
                {
                    let cap = grid.cell_capacity(xu, yu, zu);
                    let cell = grid.get_mut(xu, yu, zu);
                    cell.fluid_type = fluid_type;
                    // Cell capacity is now fractional (air_corners/8); clamp the
                    // requested level so a brush placement on a half-rock cell
                    // doesn't sit over capacity and trigger redistribution every
                    // tick.
                    cell.level = level.min(cap);
                    cell.is_source = is_source;
                    if is_source {
                        cell.level = crate::cell::MAX_LEVEL.min(cap);
                        cell.hops_from_source = 0;
                        cell.max_flow_dist = max_flow_dist;
                    }
                    // Grant grace period to non-source fluid with near-full level
                    if !is_source && level >= 0.99 {
                        cell.grace_ticks = config.source_grace_ticks;
                    }
                    grid.dirty = true;
                    grid.has_fluid = true;
                    if fluid_type.is_lava() {
                        grid.has_lava = true;
                    }
                    if is_source {
                        grid.has_sources = true;
                    }
                }
            }
        }
        FluidEvent::UpdateFluidConfig { source_grace_ticks } => {
            // Only affects newly-placed sources (read at PlaceSources / cell
            // creation, lines ~665 / ~741). cell_cap is derived purely from
            // corner densities, so nothing here invalidates per-chunk state —
            // skip the full-grid recompute + dirty sweep the old handler did.
            config.source_grace_ticks = source_grace_ticks;
        }
        FluidEvent::UpdateFluidRates {
            tick_rate,
            lava_tick_divisor,
            water_flow_rate,
            water_spread_rate,
            lava_flow_rate,
            lava_spread_rate,
        } => {
            // Pure rate swap — no cell state depends on these, so nothing to
            // invalidate. The main loop re-reads tick_rate / lava_tick_divisor
            // every iteration and tick_chunk reads the flow rates per tick,
            // so the change lands on the next tick.
            config.tick_rate = tick_rate.clamp(MIN_TICK_RATE, MAX_TICK_RATE);
            config.lava_tick_divisor = lava_tick_divisor.max(1);
            config.water_flow_rate = water_flow_rate.max(0.0);
            config.water_spread_rate = water_spread_rate.max(0.0);
            config.lava_flow_rate = lava_flow_rate.max(0.0);
            config.lava_spread_rate = lava_spread_rate.max(0.0);
        }
        FluidEvent::UpdateFluidMeshFlags {
            sticky_release, floor_clamp, buried_cull,
            flux_render, stream_ribbon, transit_retention,
            channel_bias, channel_focus, momentum,
        } => {
            let changed = config.mesh_sticky_release != sticky_release
                || config.mesh_floor_clamp != floor_clamp
                || config.mesh_buried_cull != buried_cull
                || config.mesh_flux_render != flux_render
                || config.mesh_stream_ribbon != stream_ribbon
                || config.lava_transit_retention != transit_retention
                || config.lava_channel_bias != channel_bias
                || config.lava_channel_focus != channel_focus
                || config.lava_momentum != momentum;
            config.mesh_sticky_release = sticky_release;
            config.mesh_floor_clamp = floor_clamp;
            config.mesh_buried_cull = buried_cull;
            config.mesh_flux_render = flux_render;
            config.mesh_stream_ribbon = stream_ribbon;
            config.lava_transit_retention = transit_retention;
            config.lava_channel_bias = channel_bias;
            config.lava_channel_focus = channel_focus;
            config.lava_momentum = momentum;
            // Dirty-sweep so settled (never-again-dirty) pools re-mesh with
            // the new flags immediately — this is what makes the A/B toggle
            // land on screen without touching the fluid.
            if changed {
                for grid in chunks.values_mut() {
                    if grid.has_fluid {
                        grid.dirty = true;
                    }
                }
            }
        }
        FluidEvent::PendingFluidLoad { chunk, cells } => {
            // Stash the cells; they'll be applied on the next DensityUpdate
            // or TerrainModified for this chunk. If the density is already
            // cached (chunk was streamed before fluid load was issued), drain
            // immediately so the player sees fluid without waiting.
            // Loaded chunks count as feature-placed: their saved fluid IS the
            // truth, and a later PlaceSources/PlaceGeologicalSprings from the
            // stream-in path must not stomp it back to gen-fresh full.
            features_placed.insert((chunk, FEATURE_SOURCES));
            features_placed.insert((chunk, FEATURE_SPRINGS));
            features_placed.insert((chunk, 2)); // pipe lava — same protection
            features_placed.insert((chunk, 3)); // gen-time seeds — same protection
            pending_fluid.entry(chunk).or_default().extend(cells);
            if chunk_densities.contains_key(&chunk) {
                apply_pending_fluid(chunks, chunk_densities, pending_fluid, chunk, chunk_size, config);
            }
        }
    }
}

/// Drain pending fluid for `chunk` and apply each cell to the live grid.
/// Same gating logic as the AddFluid event handler: requires the cell to
/// be reachable (cell_capacity > MIN_LEVEL and not mostly solid). Sources
/// are restored at MAX_LEVEL with hops_from_source=0.
fn apply_pending_fluid(
    chunks: &mut HashMap<(i32, i32, i32), ChunkFluidGrid>,
    chunk_densities: &HashMap<(i32, i32, i32), ChunkDensityCache>,
    pending_fluid: &mut HashMap<(i32, i32, i32), Vec<PendingFluidCell>>,
    chunk: (i32, i32, i32),
    chunk_size: usize,
    config: &FluidConfig,
) {
    let Some(cells) = pending_fluid.remove(&chunk) else { return; };
    if cells.is_empty() { return; }

    ensure_grid(chunks, chunk_densities, chunk, chunk_size);
    let Some(grid) = chunks.get_mut(&chunk) else { return; };
    let size = chunk_size;
    for cell in cells {
        let idx = cell.idx as usize;
        let total = size * size * size;
        if idx >= total { continue; }
        let xu = idx % size;
        let yu = (idx / size) % size;
        let zu = idx / (size * size);
        if grid.cell_capacity(xu, yu, zu) <= crate::cell::MIN_LEVEL { continue; }
        if grid.is_mostly_solid(xu, yu, zu, config.solid_corner_threshold) { continue; }

        let cap = grid.cell_capacity(xu, yu, zu);
        let dst = grid.get_mut(xu, yu, zu);
        dst.fluid_type = cell.fluid_type;
        dst.level = cell.level.min(cap);
        dst.is_source = cell.is_source;
        if cell.is_source {
            dst.level = crate::cell::MAX_LEVEL.min(cap);
            dst.hops_from_source = 0;
            dst.max_flow_dist = cell.max_flow_dist;
            grid.has_sources = true;
        } else if cell.level >= 0.99 {
            dst.grace_ticks = config.source_grace_ticks;
        }
        grid.dirty = true;
        grid.has_fluid = true;
        if cell.fluid_type.is_lava() {
            grid.has_lava = true;
        }
    }
}

/// Any mesh-relevant fluid on a chunk face row (raw level OR render field —
/// ribbon floors and EMA tails render without raw fluid)? Decides whether a
/// seam needs both sides re-meshed on the same pass.
fn face_row_wet(grid: &ChunkFluidGrid, axis: usize, hi: bool, size: usize) -> bool {
    let edge = if hi { size - 1 } else { 0 };
    for b in 0..size {
        for a in 0..size {
            let (x, y, z) = match axis {
                0 => (edge, a, b),
                1 => (a, edge, b),
                _ => (a, b, edge),
            };
            if grid.get(x, y, z).level > 0.001 || grid.mesh_level(x, y, z) > 0.001 {
                return true;
            }
        }
    }
    false
}

/// Build boundary levels from neighboring chunks for seamless fluid meshing.
// Boundary faces carry the neighbor's RENDER field (mesh_level: EMA +
// ribbon/fringe floors), not raw levels — the MC field is continuous across
// the seam only if both sides speak the same language. Raw sampling cut the
// surface at every chunk boundary (user repro: "broken at all seams").
pub(crate) fn build_boundary_levels(
    key: (i32, i32, i32),
    chunks: &HashMap<(i32, i32, i32), ChunkFluidGrid>,
    size: usize,
) -> BoundaryLevels {
    let mut boundary = BoundaryLevels::empty(size);

    // +X neighbor: extract x=0 face levels
    let px_key = (key.0 + 1, key.1, key.2);
    if let Some(nbr) = chunks.get(&px_key) {
        if nbr.has_fluid {
            let mut levels = vec![0.0f32; size * size];
            for z in 0..size {
                for y in 0..size {
                    levels[z * size + y] = nbr.mesh_level(0, y, z);
                }
            }
            boundary.pos_x = Some(levels);
        }
    }

    // +Y neighbor: extract y=0 face levels
    let py_key = (key.0, key.1 + 1, key.2);
    if let Some(nbr) = chunks.get(&py_key) {
        if nbr.has_fluid {
            let mut levels = vec![0.0f32; size * size];
            for z in 0..size {
                for x in 0..size {
                    levels[z * size + x] = nbr.mesh_level(x, 0, z);
                }
            }
            boundary.pos_y = Some(levels);
        }
    }

    // +Z neighbor: extract z=0 face levels
    let pz_key = (key.0, key.1, key.2 + 1);
    if let Some(nbr) = chunks.get(&pz_key) {
        if nbr.has_fluid {
            let mut levels = vec![0.0f32; size * size];
            for y in 0..size {
                for x in 0..size {
                    levels[y * size + x] = nbr.mesh_level(x, y, 0);
                }
            }
            boundary.pos_z = Some(levels);
        }
    }

    // -Y neighbor: openness of its TOP cells (density, not fluid — needed
    // even for a bone-dry chunk below so the y==0 floor extension knows an
    // open seam from a world floor and doesn't draw a lid across a fall).
    let ny_key = (key.0, key.1 - 1, key.2);
    if let Some(nbr) = chunks.get(&ny_key) {
        let mut open = vec![false; size * size];
        for z in 0..size {
            for x in 0..size {
                open[z * size + x] = nbr.cell_capacity(x, size - 1, z) > 0.001;
            }
        }
        boundary.neg_y_open = Some(open);
    }

    boundary
}

/// Drain the save-restore import stash into the sim, routed through the
/// same handler as `FluidEvent::PendingFluidLoad` so the feature guards and
/// the wait-for-density semantics stay identical to the event path.
fn drain_import_stash(
    stash: &FluidImportStash,
    chunks: &mut HashMap<(i32, i32, i32), ChunkFluidGrid>,
    chunk_densities: &mut HashMap<(i32, i32, i32), ChunkDensityCache>,
    pending_fluid: &mut HashMap<(i32, i32, i32), Vec<PendingFluidCell>>,
    features_placed: &mut HashSet<((i32, i32, i32), u8)>,
    chunk_size: usize,
    config: &mut FluidConfig,
    result_tx: &Sender<FluidResult>,
    active_fluid_meshes: &mut HashSet<(i32, i32, i32)>,
) {
    let imported: Vec<((i32, i32, i32), Vec<PendingFluidCell>)> = {
        let mut guard = match stash.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        std::mem::take(&mut *guard)
    };
    if imported.is_empty() {
        return;
    }
    let n_chunks = imported.len();
    let n_cells: usize = imported.iter().map(|(_, c)| c.len()).sum();
    for (chunk, cells) in imported {
        handle_event(
            FluidEvent::PendingFluidLoad { chunk, cells },
            chunks, chunk_densities, pending_fluid, features_placed,
            chunk_size, config, result_tx, active_fluid_meshes,
        );
    }
    eprintln!("[FLUID-IMPORT] drained {n_chunks} chunks / {n_cells} cells from the import stash");
}

/// Ensure a full fluid grid exists for a chunk, promoting from density cache if needed.
fn ensure_grid(
    chunks: &mut HashMap<(i32, i32, i32), ChunkFluidGrid>,
    chunk_densities: &HashMap<(i32, i32, i32), ChunkDensityCache>,
    chunk: (i32, i32, i32),
    chunk_size: usize,
) {
    if chunks.contains_key(&chunk) {
        return;
    }
    let grid = if let Some(cache) = chunk_densities.get(&chunk) {
        ChunkFluidGrid::from_density_cache(cache)
    } else {
        ChunkFluidGrid::new(chunk_size)
    };
    chunks.insert(chunk, grid);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive `handle_event` the way the sim loop does, with empty world state.
    fn apply(config: &mut FluidConfig, event: FluidEvent) {
        let mut chunks = HashMap::new();
        let mut densities = HashMap::new();
        let mut pending = HashMap::new();
        let mut placed = HashSet::new();
        let (result_tx, _result_rx) = crossbeam_channel::unbounded();
        let mut active_meshes = HashSet::new();
        handle_event(event, &mut chunks, &mut densities, &mut pending, &mut placed, config.chunk_size, config, &result_tx, &mut active_meshes);
    }

    /// 2026-08-13 regression ("lava popped back and burned me"): DrainLavaChunks
    /// only touched LIVE grids — a chunk fluid-unloaded at drain time kept its
    /// fluid in pending_fluid at FULL level with is_source=true, and
    /// re-streaming re-injected the whole pool, sources included. The drain
    /// must purge the stash too. (2026-08-20 round 3: water evaporates along
    /// with the lava — the whole entry goes, not just lava cells.)
    #[test]
    fn drain_lava_purges_pending_stash() {
        let mut config = FluidConfig::default();
        let mut chunks = HashMap::new();
        let mut densities = HashMap::new();
        let mut placed = HashSet::new();
        let mut pending: HashMap<(i32, i32, i32), Vec<PendingFluidCell>> = HashMap::new();
        pending.insert((3, -4, 0), vec![
            PendingFluidCell { idx: 5, fluid_type: crate::cell::FluidType::Lava, level: 1.0, is_source: true, max_flow_dist: 12 },
            PendingFluidCell { idx: 9, fluid_type: crate::cell::FluidType::Water, level: 0.5, is_source: false, max_flow_dist: 0 },
        ]);
        pending.insert((4, -4, 0), vec![
            PendingFluidCell { idx: 1, fluid_type: crate::cell::FluidType::Lava, level: 1.0, is_source: true, max_flow_dist: 12 },
        ]);
        let (result_tx, _result_rx) = crossbeam_channel::unbounded();
        let mut active_meshes = HashSet::new();
        handle_event(
            FluidEvent::DrainLavaChunks { chunks: vec![(3, -4, 0), (4, -4, 0)] },
            &mut chunks, &mut densities, &mut pending, &mut placed,
            config.chunk_size, &mut config, &result_tx, &mut active_meshes,
        );
        assert!(!pending.contains_key(&(3, -4, 0)), "mixed stash evaporates entirely (water included)");
        assert!(!pending.contains_key(&(4, -4, 0)), "lava-only stash must be removed");
    }

    /// 2026-08-20 residual of the 08-13 stash purge: a chunk fluid-UNLOADED at
    /// drain time has no live grid, so the dirty→re-mesh→empty-mesh path never
    /// fires for it — the UE actor keeps its last non-empty lava mesh (visible,
    /// and burning via #248 contact) until the actor recycles. The drain must
    /// send an explicit empty FluidMesh for every drained key that has no live
    /// grid but a previously-sent mesh (active_fluid_meshes gates the send so
    /// never-meshed chunks don't spam empties). Live grids stay on the normal
    /// dirty→re-mesh path.
    #[test]
    fn drain_lava_sends_empty_mesh_for_unloaded_chunks() {
        let mut config = FluidConfig::default();
        let size = config.chunk_size;
        let mut chunks = HashMap::new();
        let mut densities = HashMap::new();
        let mut placed = HashSet::new();
        let mut pending: HashMap<(i32, i32, i32), Vec<PendingFluidCell>> = HashMap::new();
        let (result_tx, result_rx) = crossbeam_channel::unbounded();
        let mut active_meshes: HashSet<(i32, i32, i32)> = HashSet::new();

        // (3,-4,0): fluid-unloaded — UE holds a lava mesh, the stash holds the lava.
        pending.insert((3, -4, 0), vec![
            PendingFluidCell { idx: 5, fluid_type: crate::cell::FluidType::Lava, level: 1.0, is_source: true, max_flow_dist: 12 },
        ]);
        active_meshes.insert((3, -4, 0));
        // (4,-4,0): stashed lava but no mesh was ever sent — no empty needed.
        pending.insert((4, -4, 0), vec![
            PendingFluidCell { idx: 1, fluid_type: crate::cell::FluidType::Lava, level: 1.0, is_source: true, max_flow_dist: 12 },
        ]);
        // (5,-4,0): live grid with a mesh — the dirty→re-mesh path owns it.
        let mut grid = ChunkFluidGrid::new(size);
        {
            let cell = grid.get_mut(1, 1, 1);
            cell.fluid_type = crate::cell::FluidType::Lava;
            cell.level = 1.0;
        }
        grid.has_lava = true;
        chunks.insert((5, -4, 0), grid);
        active_meshes.insert((5, -4, 0));

        handle_event(
            FluidEvent::DrainLavaChunks { chunks: vec![(3, -4, 0), (4, -4, 0), (5, -4, 0)] },
            &mut chunks, &mut densities, &mut pending, &mut placed,
            size, &mut config, &result_tx, &mut active_meshes,
        );

        let results: Vec<FluidResult> = result_rx.try_iter().collect();
        assert_eq!(
            results.len(), 1,
            "exactly one explicit empty mesh: the unloaded, previously-meshed chunk (got {})",
            results.len()
        );
        match &results[0] {
            FluidResult::FluidMesh { chunk, mesh } => {
                assert_eq!(*chunk, (3, -4, 0));
                assert!(mesh.positions.is_empty() && mesh.indices.is_empty(), "the clearing mesh must be empty");
            }
            _ => panic!("expected FluidResult::FluidMesh"),
        }
        assert!(!active_meshes.contains(&(3, -4, 0)), "cleared chunk must leave the tracker");
        assert!(active_meshes.contains(&(5, -4, 0)), "live-grid chunk stays tracked for the mesh pass");
        assert!(chunks[&(5, -4, 0)].dirty, "live-grid chunk must be dirty so the mesh pass re-sends truth");
    }

    /// 2026-08-20 live-run regression (World_1991, first montage after the
    /// 08-20 teardown fixes): the drain predicate `is_lava() && level > 0.001`
    /// skips lava SOURCE cells sitting at ~0 level. Since free-fall lifted the
    /// gravity rate cap, a source feeding an open shaft moves its ENTIRE level
    /// out every lava tick — its post-tick level IS 0.0 — so the drain left
    /// `is_source=true` and regen re-pumped the "drained" lavafall within
    /// seconds of the montage ending (and slowly re-filled the pool below).
    /// The drain must extinguish every lava-typed cell regardless of level.
    #[test]
    fn drain_lava_extinguishes_zero_level_sources() {
        let mut config = FluidConfig::default();
        let size = config.chunk_size;
        let mut chunks = HashMap::new();
        let mut densities = HashMap::new();
        let mut placed = HashSet::new();
        let mut pending = HashMap::new();
        let (result_tx, _result_rx) = crossbeam_channel::unbounded();
        let mut active_meshes = HashSet::new();

        let mut grid = ChunkFluidGrid::new(size);
        {
            // Fall-feeding source in post-tick state: regen refilled it, the
            // free-fall gulp emptied it — level 0, is_source still true.
            let cell = grid.get_mut(2, 2, 2);
            cell.fluid_type = crate::cell::FluidType::Lava;
            cell.level = 0.0;
            cell.is_source = true;
            cell.max_flow_dist = 12;
        }
        grid.has_lava = true;
        grid.has_sources = true;
        chunks.insert((1, 1, 1), grid);

        handle_event(
            FluidEvent::DrainLavaChunks { chunks: vec![(1, 1, 1)] },
            &mut chunks, &mut densities, &mut pending, &mut placed,
            size, &mut config, &result_tx, &mut active_meshes,
        );

        let cell = chunks[&(1, 1, 1)].get(2, 2, 2);
        assert!(
            !cell.is_source,
            "zero-level lava source must be extinguished by the drain (it re-pumps the stream otherwise)"
        );
    }

    /// 2026-08-20 round 3 (user directive): "the water is supposed to
    /// evaporate with the dormancy along with the lava" — 1.25Ma of
    /// geological time leaves no standing fluid. The dormancy drain must
    /// remove water (cells, sources, stash, render state) exactly like lava.
    #[test]
    fn dormancy_drain_evaporates_water_too() {
        let mut config = FluidConfig::default();
        let size = config.chunk_size;
        let total = size * size * size;
        let mut chunks = HashMap::new();
        let mut densities = HashMap::new();
        let mut placed = HashSet::new();
        let mut pending: HashMap<(i32, i32, i32), Vec<PendingFluidCell>> = HashMap::new();
        let (result_tx, _result_rx) = crossbeam_channel::unbounded();
        let mut active_meshes = HashSet::new();

        let mut grid = ChunkFluidGrid::new(size);
        {
            let cell = grid.get_mut(3, 3, 3);
            cell.fluid_type = crate::cell::FluidType::Water;
            cell.level = 0.8;
        }
        {
            // A spring: sources must not survive and re-pump the pool.
            let cell = grid.get_mut(5, 2, 5);
            cell.fluid_type = crate::cell::FluidType::Water;
            cell.level = 1.0;
            cell.is_source = true;
        }
        grid.has_fluid = true;
        grid.has_sources = true;
        let pool_idx = grid.index(3, 3, 3);
        grid.stream_mark = vec![false; total];
        grid.stream_mark[pool_idx] = true;
        chunks.insert((1, 0, 1), grid);
        // Fluid-unloaded chunk: stashed water must evaporate too.
        pending.insert((2, 0, 1), vec![
            PendingFluidCell { idx: 4, fluid_type: crate::cell::FluidType::Water, level: 0.9, is_source: true, max_flow_dist: 0 },
        ]);

        handle_event(
            FluidEvent::DrainLavaChunks { chunks: vec![(1, 0, 1), (2, 0, 1)] },
            &mut chunks, &mut densities, &mut pending, &mut placed,
            size, &mut config, &result_tx, &mut active_meshes,
        );

        let grid = &chunks[&(1, 0, 1)];
        assert_eq!(grid.get(3, 3, 3).level, 0.0, "standing water must evaporate in the dormancy drain");
        assert!(!grid.get(5, 2, 5).is_source, "water sources must not survive to re-fill the pool");
        assert_eq!(grid.get(5, 2, 5).level, 0.0);
        assert!(!grid.stream_mark[grid.index(3, 3, 3)], "water render state clears like lava's");
        assert!(!pending.contains_key(&(2, 0, 1)), "stashed water must evaporate too");
    }

    /// 2026-08-20 round 2 (user verify: "pool interiors drained, but the
    /// EDGES and the lavafall persist as burning lava mixed with water
    /// material"): the ribbon/fringe renderer meshes stream-marked cells at
    /// STREAM_FLOOR with NO raw-level gate, and stream_mark only exits when
    /// flux_ema decays below STREAM_FLUX_OFF — but flux only decays in
    /// tick_chunk, which a drained lava-only chunk never enters again. The
    /// pool rims and falls (exactly the flux-carrying cells) froze as ghost
    /// lava meshes that still burn via #248. The drain must zero the
    /// per-cell render state along with the fluid.
    #[test]
    fn drain_lava_clears_ribbon_and_render_state() {
        let mut config = FluidConfig::default();
        let size = config.chunk_size;
        let total = size * size * size;
        let mut chunks = HashMap::new();
        let mut densities = HashMap::new();
        let mut placed = HashSet::new();
        let mut pending = HashMap::new();
        let (result_tx, _result_rx) = crossbeam_channel::unbounded();
        let mut active_meshes = HashSet::new();

        let mut grid = ChunkFluidGrid::new(size);
        let rim = grid.index(4, 2, 4);
        {
            // Pool-rim spill cell as the live game leaves it: modest raw
            // level, ribbon-marked, high flux average, held in hysteresis.
            let cell = grid.get_mut(4, 2, 4);
            cell.fluid_type = crate::cell::FluidType::Lava;
            cell.level = 0.08;
        }
        grid.has_lava = true;
        grid.render_flux = true;
        grid.render_ribbon = true;
        grid.render_level = vec![0.0; total];
        grid.render_level[rim] = 0.9;
        grid.flux_ema = vec![0.0; total];
        grid.flux_ema[rim] = 0.3;
        grid.stream_mark = vec![false; total];
        grid.stream_mark[rim] = true;
        grid.mesh_sticky = vec![false; total];
        grid.mesh_sticky[rim] = true;
        chunks.insert((2, 0, 2), grid);

        handle_event(
            FluidEvent::DrainLavaChunks { chunks: vec![(2, 0, 2)] },
            &mut chunks, &mut densities, &mut pending, &mut placed,
            size, &mut config, &result_tx, &mut active_meshes,
        );

        let grid = &chunks[&(2, 0, 2)];
        assert!(
            grid.mesh_level(4, 2, 4) < crate::cell::MESH_STICKY_ON,
            "drained rim cell must not mesh (got mesh_level={}) — ribbon floor / EMA ghost",
            grid.mesh_level(4, 2, 4)
        );
        assert!(!grid.stream_mark[rim], "stream mark must clear with the lava");
        assert_eq!(grid.flux_ema[rim], 0.0, "flux average must clear with the lava");
        assert_eq!(grid.render_level[rim], 0.0, "render EMA must clear with the lava");
        assert!(!grid.mesh_sticky[rim], "hysteresis hold must clear with the lava");
    }

    /// 2026-08-20 companion to the montage teardown: RestoreMontageFluidState
    /// wipes every montage-touched fluid component and needs a RELIABLE truth
    /// re-send. The old "unchanged UpdateConfig push" swept nothing (the
    /// UpdateFluidMeshFlags handler only sweeps when a flag changes), so
    /// settled water/lava stayed invisible until the next organic dirty —
    /// the user's mine made it all "pop back in" at once. RemeshAllFluid must
    /// dirty every live grid and explicitly clear tracked-but-gridless meshes.
    #[test]
    fn remesh_all_fluid_sweeps_grids_and_clears_gridless_tracked() {
        let mut config = FluidConfig::default();
        let size = config.chunk_size;
        let mut chunks = HashMap::new();
        let mut densities = HashMap::new();
        let mut placed = HashSet::new();
        let mut pending = HashMap::new();
        let (result_tx, result_rx) = crossbeam_channel::unbounded();
        let mut active_meshes: HashSet<(i32, i32, i32)> = HashSet::new();

        // Settled water pool: live grid, not dirty, previously meshed.
        let mut grid = ChunkFluidGrid::new(size);
        {
            let cell = grid.get_mut(3, 3, 3);
            cell.fluid_type = crate::cell::FluidType::Water;
            cell.level = 0.8;
        }
        grid.has_fluid = true;
        grid.dirty = false;
        chunks.insert((0, 0, 0), grid);
        active_meshes.insert((0, 0, 0));
        // Tracked mesh whose grid is gone: must get the explicit empty.
        active_meshes.insert((7, -2, 3));

        handle_event(
            FluidEvent::RemeshAllFluid,
            &mut chunks, &mut densities, &mut pending, &mut placed,
            size, &mut config, &result_tx, &mut active_meshes,
        );

        assert!(chunks[&(0, 0, 0)].dirty, "live grid must be dirty so the mesh pass re-sends its truth");
        assert!(active_meshes.contains(&(0, 0, 0)), "live tracked chunk stays tracked (mesh pass owns it)");
        assert!(!active_meshes.contains(&(7, -2, 3)), "gridless tracked chunk must leave the tracker");
        let results: Vec<FluidResult> = result_rx.try_iter().collect();
        assert_eq!(results.len(), 1, "exactly one explicit empty for the gridless tracked chunk");
        match &results[0] {
            FluidResult::FluidMesh { chunk, mesh } => {
                assert_eq!(*chunk, (7, -2, 3));
                assert!(mesh.positions.is_empty() && mesh.indices.is_empty());
            }
            _ => panic!("expected FluidResult::FluidMesh"),
        }
    }

    /// 2026-08-19 regression ("fluid pick lava sometimes doesn't survive a
    /// load"): saved fluid used to arrive as try_send events on the bounded
    /// event channel, which the load-time streaming flood fills — whole
    /// chunks of saved fluid were silently dropped (same save restored
    /// all / none / a-sixth across three loads). Imports now ride the
    /// FluidImportStash, drained every sim iteration: delivery must be
    /// guaranteed and must behave exactly like a PendingFluidLoad event —
    /// parked until density arrives, feature guards marked so stream-in
    /// placement can't stomp the restored state.
    #[test]
    fn import_stash_delivers_like_pending_fluid_load() {
        let mut config = FluidConfig::default();
        let size = config.chunk_size;
        let stash: FluidImportStash = Default::default();
        stash.lock().unwrap().push(((2, 0, -1), vec![
            PendingFluidCell { idx: 7, fluid_type: crate::cell::FluidType::Lava, level: 0.8, is_source: false, max_flow_dist: 12 },
        ]));

        let mut chunks = HashMap::new();
        let mut densities = HashMap::new();
        let mut pending = HashMap::new();
        let mut placed = HashSet::new();
        let chunk_size = size;
        let (stash_tx, _stash_rx) = crossbeam_channel::unbounded();
        let mut stash_meshes = HashSet::new();
        drain_import_stash(&stash, &mut chunks, &mut densities, &mut pending, &mut placed, chunk_size, &mut config, &stash_tx, &mut stash_meshes);

        assert!(stash.lock().unwrap().is_empty(), "stash must drain");
        let parked = pending.get(&(2, 0, -1)).expect("no density yet: cells park in pending_fluid");
        assert_eq!(parked.len(), 1);
        assert!(placed.contains(&((2, 0, -1), 0)), "restored chunks must be feature-guarded");

        // Density arrives (all-open chunk) -> the parked fluid lands in the grid.
        let lattice = (size + 1) * (size + 1) * (size + 1);
        let (result_tx, _result_rx) = crossbeam_channel::unbounded();
        let mut active_meshes = HashSet::new();
        handle_event(
            FluidEvent::DensityUpdate { chunk: (2, 0, -1), densities: vec![-1.0; lattice] },
            &mut chunks, &mut densities, &mut pending, &mut placed,
            chunk_size, &mut config, &result_tx, &mut active_meshes,
        );
        let grid = chunks.get(&(2, 0, -1)).expect("grid exists after density");
        assert!(grid.cells[7].level > 0.5, "restored cell must hold its saved level");
        assert!(pending.get(&(2, 0, -1)).is_none(), "pending drains on density arrival");
    }

    /// Bug #216 regression: the worker re-sends PlaceSources on every chunk
    /// stream-in. Re-planting must be a no-op — it used to refill every
    /// noise-lava pool to full and resurrect self-extinguished sources
    /// whenever the player's streaming set churned.
    #[test]
    fn place_sources_fires_once_per_chunk() {
        let mut config = FluidConfig::default();
        config.lava_depth_min = -1.0e9;
        config.lava_depth_max = 1.0e9;
        config.lava_source_threshold = -1.0; // every candidate cell qualifies
        let chunk = (0, 0, 0);
        let size = config.chunk_size;

        let mut chunks = HashMap::new();
        let mut densities = HashMap::new();
        let mut pending = HashMap::new();
        let mut placed = HashSet::new();

        // Density must exist for ensure_grid to build the chunk.
        let stride = size + 1;
        let lattice = vec![-1.0f32; stride * stride * stride];
        let (result_tx, _result_rx) = crossbeam_channel::unbounded();
        let mut active_meshes = HashSet::new();
        handle_event(
            FluidEvent::DensityUpdate { chunk, densities: lattice },
            &mut chunks, &mut densities, &mut pending, &mut placed, size, &mut config,
            &result_tx, &mut active_meshes,
        );
        handle_event(
            FluidEvent::PlaceSources { chunk },
            &mut chunks, &mut densities, &mut pending, &mut placed, size, &mut config,
            &result_tx, &mut active_meshes,
        );

        // Find a planted source and simulate a self-extinguish + drain.
        let grid = chunks.get_mut(&chunk).expect("grid created");
        let idx = grid.cells.iter().position(|c| c.is_source).expect("sources planted");
        grid.cells[idx].is_source = false;
        grid.cells[idx].level = 0.1;

        // Stream-in re-send must not stomp the demoted state.
        handle_event(
            FluidEvent::PlaceSources { chunk },
            &mut chunks, &mut densities, &mut pending, &mut placed, size, &mut config,
            &result_tx, &mut active_meshes,
        );
        let cell = &chunks[&chunk].cells[idx];
        assert!(!cell.is_source, "re-sent PlaceSources resurrected a demoted source");
        assert!(
            (cell.level - 0.1).abs() < 1e-6,
            "re-sent PlaceSources refilled a drained cell: level={}",
            cell.level
        );
    }

    fn rates(tick_rate: f32, divisor: u8, lava_flow: f32, lava_spread: f32) -> FluidEvent {
        FluidEvent::UpdateFluidRates {
            tick_rate,
            lava_tick_divisor: divisor,
            water_flow_rate: 1.5,
            water_spread_rate: 0.75,
            lava_flow_rate: lava_flow,
            lava_spread_rate: lava_spread,
        }
    }

    #[test]
    fn update_fluid_rates_applies_live() {
        let mut config = FluidConfig::default();
        apply(&mut config, rates(30.0, 2, 0.4, 0.6));
        assert_eq!(config.tick_rate, 30.0);
        assert_eq!(config.lava_tick_divisor, 2);
        assert_eq!(config.lava_flow_rate, 0.4);
        assert_eq!(config.lava_spread_rate, 0.6);
        assert_eq!(config.water_flow_rate, 1.5);
        assert_eq!(config.water_spread_rate, 0.75);
    }

    #[test]
    fn update_fluid_rates_leaves_non_rate_fields_alone() {
        // The codex writes several config files; the rate event must not
        // clobber fields owned by the water-config / creation-time paths.
        let mut config = FluidConfig::default();
        config.source_grace_ticks = 123;
        config.solid_corner_threshold = 3;
        config.water_substeps = 4;
        config.seed = 999;
        apply(&mut config, rates(20.0, 8, 0.2, 0.3));
        assert_eq!(config.source_grace_ticks, 123);
        assert_eq!(config.solid_corner_threshold, 3);
        assert_eq!(config.water_substeps, 4);
        assert_eq!(config.seed, 999);
    }

    #[test]
    fn update_fluid_rates_clamps_hostile_values() {
        // These come from a hand-editable JSON file. A zero/negative tick rate
        // would panic `Duration::from_secs_f32` on an infinite interval, and a
        // zero divisor would panic on `tick_count % 0`.
        let mut config = FluidConfig::default();
        apply(&mut config, rates(0.0, 0, -1.0, -1.0));
        assert_eq!(config.tick_rate, MIN_TICK_RATE);
        assert_eq!(config.lava_tick_divisor, 1);
        assert_eq!(config.lava_flow_rate, 0.0);
        assert_eq!(config.lava_spread_rate, 0.0);
        // The interval the loop derives must stay finite.
        let interval = Duration::from_secs_f32(1.0 / config.tick_rate);
        assert!(interval.as_secs_f32().is_finite());

        apply(&mut config, rates(100_000.0, 8, 0.1, 0.1));
        assert_eq!(config.tick_rate, MAX_TICK_RATE);
    }
}
