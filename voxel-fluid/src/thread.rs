use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};

use crate::cell::ChunkFluidGrid;
use crate::mesh::mesh_fluid;
use crate::sim::{detect_solidification, regen_sources, tick_fluid};
use crate::sources::place_sources;
use crate::{FluidConfig, FluidEvent, FluidResult};

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
    let mut chunks: HashMap<(i32, i32, i32), ChunkFluidGrid> = HashMap::new();
    let chunk_size = config.chunk_size;

    let tick_interval = Duration::from_secs_f32(1.0 / config.tick_rate);
    let mut last_tick = Instant::now();
    let mut tick_count: u64 = 0;
    let lava_divisor = config.lava_tick_divisor.max(1) as u64;

    while !shutdown.load(Ordering::Relaxed) {
        // Drain all pending events
        loop {
            match event_rx.try_recv() {
                Ok(event) => handle_event(event, &mut chunks, chunk_size, &config),
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
        let dirty_water = tick_fluid(&mut chunks, chunk_size, false);
        let dirty_lava = if is_lava_tick {
            tick_fluid(&mut chunks, chunk_size, true)
        } else {
            Vec::new()
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
        let mut all_dirty: Vec<(i32, i32, i32)> = Vec::new();
        for k in dirty_water {
            if !all_dirty.contains(&k) {
                all_dirty.push(k);
            }
        }
        for k in dirty_lava {
            if !all_dirty.contains(&k) {
                all_dirty.push(k);
            }
        }
        // Also include chunks marked dirty by events
        for (&k, grid) in &mut chunks {
            if grid.dirty && !all_dirty.contains(&k) {
                all_dirty.push(k);
            }
        }

        // Mesh dirty chunks and send results
        for key in &all_dirty {
            if let Some(grid) = chunks.get_mut(key) {
                let mesh = mesh_fluid(grid);
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
    chunk_size: usize,
    config: &FluidConfig,
) {
    match event {
        FluidEvent::SolidMaskUpdate { chunk, mask } => {
            let grid = chunks
                .entry(chunk)
                .or_insert_with(|| ChunkFluidGrid::new(chunk_size));
            grid.update_solid_mask(&mask);
            grid.dirty = true;
        }
        FluidEvent::PlaceSources { chunk } => {
            let grid = chunks
                .entry(chunk)
                .or_insert_with(|| ChunkFluidGrid::new(chunk_size));
            place_sources(grid, chunk, chunk_size, config);
        }
        FluidEvent::TerrainModified { chunk, mask } => {
            if let Some(grid) = chunks.get_mut(&chunk) {
                grid.update_solid_mask(&mask);
                grid.dirty = true;
            }
        }
        FluidEvent::ChunkUnloaded { chunk } => {
            chunks.remove(&chunk);
        }
        FluidEvent::AddFluid { chunk, x, y, z, fluid_type, level, is_source } => {
            let grid = chunks
                .entry(chunk)
                .or_insert_with(|| ChunkFluidGrid::new(chunk_size));
            let xu = x as usize;
            let yu = y as usize;
            let zu = z as usize;
            if xu < chunk_size && yu < chunk_size && zu < chunk_size && !grid.is_solid(xu, yu, zu) {
                let cell = grid.get_mut(xu, yu, zu);
                cell.fluid_type = fluid_type;
                cell.level = level;
                if is_source {
                    // Sources stay at MAX_LEVEL via regen_sources
                    cell.level = crate::cell::MAX_LEVEL;
                }
                grid.dirty = true;
            }
        }
    }
}
