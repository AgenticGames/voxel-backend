use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};

use crate::cell::{ChunkDensityCache, ChunkFluidGrid};
use crate::mesh::{mesh_fluid, BoundaryLevels};
use crate::sim::{detect_solidification, regen_sources, squeeze_excess_fluid, tick_fluid};
use crate::sources::place_sources;
use crate::{FluidConfig, FluidEvent, FluidResult, FluidSnapshot};

/// Main fluid simulation loop running on its own thread.
///
/// Drains events from `event_rx`, ticks the simulation at the configured rate,
/// meshes dirty chunks, and sends results through `result_tx`.
pub fn fluid_sim_loop(
    shutdown: Arc<AtomicBool>,
    event_rx: Receiver<FluidEvent>,
    result_tx: Sender<FluidResult>,
    config: FluidConfig,
) {
    let mut config = config;
    let mut chunks: HashMap<(i32, i32, i32), ChunkFluidGrid> = HashMap::new();
    // Lightweight density-only storage for chunks without fluid
    let mut chunk_densities: HashMap<(i32, i32, i32), ChunkDensityCache> = HashMap::new();
    let chunk_size = config.chunk_size;

    let tick_interval = Duration::from_secs_f32(1.0 / config.tick_rate);
    let mut last_tick = Instant::now();
    let mut tick_count: u64 = 0;
    let lava_divisor = config.lava_tick_divisor.max(1) as u64;

    while !shutdown.load(Ordering::Relaxed) {
        // Drain all pending events
        loop {
            match event_rx.try_recv() {
                Ok(event) => handle_event(event, &mut chunks, &mut chunk_densities, chunk_size, &mut config),
                Err(_) => break,
            }
        }

        // Check if it's time for a tick
        let now = Instant::now();
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

        // Tick water every tick, lava every N ticks
        let is_lava_tick = tick_count % lava_divisor == 0;
        let dirty_water = tick_fluid(&mut chunks, &chunk_densities, chunk_size, false, &config);
        let dirty_lava = if is_lava_tick {
            tick_fluid(&mut chunks, &chunk_densities, chunk_size, true, &config)
        } else {
            HashSet::new()
        };

        // Detect solidification (lava+water contact)
        let solidify = detect_solidification(&chunks);
        for (key, x, y, z) in &solidify {
            if let Some(grid) = chunks.get_mut(key) {
                let cell = grid.get_mut(*x, *y, *z);
                cell.level = 0.0;
                grid.dirty = true;
            }
        }

        // Send solidification requests to the main engine
        if !solidify.is_empty() {
            let positions: Vec<((i32, i32, i32), usize, usize, usize)> = solidify;
            let _ = result_tx.send(FluidResult::SolidifyRequest { positions });
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

        // Mesh dirty chunks and send results
        for key in &all_dirty {
            let boundary = build_boundary_levels(*key, &chunks, chunk_size);
            if let Some(grid) = chunks.get_mut(key) {
                let mesh = mesh_fluid(grid, &boundary, &config);
                grid.dirty = false;

                if !mesh.positions.is_empty() {
                    let _ = result_tx.send(FluidResult::FluidMesh {
                        chunk: *key,
                        mesh,
                    });
                }
            }
        }
    }
}

fn handle_event(
    event: FluidEvent,
    chunks: &mut HashMap<(i32, i32, i32), ChunkFluidGrid>,
    chunk_densities: &mut HashMap<(i32, i32, i32), ChunkDensityCache>,
    chunk_size: usize,
    config: &mut FluidConfig,
) {
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
        }
        FluidEvent::PlaceSources { chunk } => {
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

            // If grid exists, update its density and squeeze excess
            if let Some(grid) = chunks.get_mut(&chunk) {
                grid.update_density(&densities);
                squeeze_excess_fluid(grid);
                grid.dirty = true;
            }
        }
        FluidEvent::ChunkUnloaded { chunk } => {
            chunks.remove(&chunk);
            chunk_densities.remove(&chunk);
        }
        FluidEvent::SnapshotRequest { reply_tx } => {
            let snapshot = FluidSnapshot {
                chunks: chunks.iter().map(|(&k, g)| (k, g.cells.clone())).collect(),
                chunk_size,
            };
            let _ = reply_tx.send(snapshot);
        }
        FluidEvent::PlaceGeologicalSprings { chunk, springs } => {
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
                        let cell = grid.get_mut(xu, yu, zu);
                        cell.fluid_type = crate::cell::FluidType::from_u8(fluid_type_u8);
                        cell.level = level.min(crate::cell::MAX_LEVEL);
                        cell.is_source = true; // geological springs are infinite sources
                        grid.dirty = true;
                        grid.has_fluid = true;
                    }
                }
            }
        }
        FluidEvent::AddFluid { chunk, x, y, z, fluid_type, level, is_source } => {
            ensure_grid(chunks, chunk_densities, chunk, chunk_size);
            if let Some(grid) = chunks.get_mut(&chunk) {
                let xu = x as usize;
                let yu = y as usize;
                let zu = z as usize;
                if xu < chunk_size && yu < chunk_size && zu < chunk_size
                    && grid.cell_capacity(xu, yu, zu) > crate::cell::MIN_LEVEL
                    && !grid.is_mostly_solid(xu, yu, zu, config.solid_corner_threshold)
                {
                    let cell = grid.get_mut(xu, yu, zu);
                    cell.fluid_type = fluid_type;
                    cell.level = level;
                    cell.is_source = is_source;
                    if is_source {
                        cell.level = crate::cell::MAX_LEVEL;
                    }
                    // Grant grace period to non-source fluid with near-full level
                    if !is_source && level >= 0.99 {
                        cell.grace_ticks = config.source_grace_ticks;
                    }
                    grid.dirty = true;
                    grid.has_fluid = true;
                }
            }
        }
        FluidEvent::UpdateFluidConfig { source_grace_ticks } => {
            config.source_grace_ticks = source_grace_ticks;
            // Recompute capacity with binary classification for all loaded chunks
            let keys: Vec<_> = chunks.keys().copied().collect();
            for chunk_key in keys {
                if let Some(grid) = chunks.get_mut(&chunk_key) {
                    grid.recompute_capacity();
                    grid.dirty = true;
                }
            }
        }
    }
}

/// Build boundary levels from neighboring chunks for seamless fluid meshing.
fn build_boundary_levels(
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
                    levels[z * size + y] = nbr.get(0, y, z).level;
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
                    levels[z * size + x] = nbr.get(x, 0, z).level;
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
                    levels[y * size + x] = nbr.get(x, y, 0).level;
                }
            }
            boundary.pos_z = Some(levels);
        }
    }

    boundary
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
