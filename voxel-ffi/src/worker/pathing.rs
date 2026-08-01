//! Dedicated path-worker thread loop + request handler.
//!
//! Pure code-movement out of the former monolithic `worker.rs`; behavior is
//! unchanged.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender};
use voxel_gen::config::GenerationConfig;

use crate::store::ChunkStore;
use crate::types::{WorkerRequest, WorkerResult};

// ─── Path-worker loop ────────────────────────────────────────────────
//
// Dedicated thread for AI path queries. Reads from `path_rx` (which is fed by
// the FFI `voxel_path_request` call → engine `path_tx`). Each request runs
// A* against the live ChunkStore and emits a `PathComputed` result through
// the shared `result_tx` — intercepted in engine.rs `poll_result` and stashed
// into `path_results` keyed by request_id.

/// Cell factor re-export — see `crate::pathing::PATH_CELL_FACTOR`.
use crate::pathing::PATH_CELL_FACTOR;

pub fn path_worker_loop(
    shutdown: Arc<AtomicBool>,
    path_rx: Receiver<WorkerRequest>,
    result_tx: Sender<WorkerResult>,
    store: Arc<RwLock<ChunkStore>>,
    config: Arc<RwLock<GenerationConfig>>,
    occupied_cells: Arc<RwLock<Arc<std::collections::HashSet<(i32, i32, i32)>>>>,
    world_scale: f32,
) {
    while !shutdown.load(Ordering::Relaxed) {
        // Block (with timeout) on the path channel — separate from the main
        // mine/generate workers so neither gets head-of-line blocked.
        match path_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(WorkerRequest::ComputePath { request }) => {
                handle_path_request(request, &result_tx, &store, &config, &occupied_cells, world_scale);
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
    occupied_cells: &Arc<RwLock<Arc<std::collections::HashSet<(i32, i32, i32)>>>>,
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

    // Snapshot the occupancy set for this request via Arc clone — the lock is
    // held only for the refcount bump, never across A*. Holding a read guard
    // across the solve (the previous design) blocked UE's 10Hz
    // `voxel_path_set_obstacle_cells` write on the game thread for the full
    // duration of any in-flight solve: 45-51ms GT hitches every couple of
    // seconds in the 2026-06-13 idle trace.
    let occupied_snapshot: Option<std::sync::Arc<std::collections::HashSet<(i32, i32, i32)>>> =
        occupied_cells.read().ok().map(|g| std::sync::Arc::clone(&*g));
    let occupied_ref = occupied_snapshot.as_deref();

    // Fine-cell requests (sense trail) plan on single voxels — exact
    // solidity, no 1-of-8 sampling — and skip the cross-species occupancy
    // layer entirely: its coords are quantized at the coarse factor, and a
    // player-guidance ribbon shouldn't route around wasps anyway.
    let cell_factor = if request.fine_cells { 1 } else { PATH_CELL_FACTOR };
    let occupied_for_grid = if request.fine_cells { None } else { occupied_ref };

    // Compute the requester's pathing cell so the grid can self-exclude.
    // Same math as `to_path_request` below but inline so we don't run it
    // twice.
    let cf = cell_factor as f32;
    let requester_cell = glam::IVec3::new(
        (request.from_voxel.x / cf).floor() as i32,
        (request.from_voxel.y / cf).floor() as i32,
        (request.from_voxel.z / cf).floor() as i32,
    );

    let grid = crate::pathing::ChunkStoreGrid {
        store: &store_guard,
        chunk_size,
        cell_factor,
        occupied_cells: occupied_for_grid,
        requester_cell: Some(requester_cell),
    };

    let (path_req, _mode) = crate::pathing::to_path_request(&request, cell_factor);
    let outcome = voxel_path::compute_path(&grid, path_req);

    // Drop the store guard before doing the UE conversion — keeps the read
    // lock window as short as possible.
    drop(store_guard);
    drop(occupied_snapshot);

    let nodes_ue = crate::pathing::nodes_to_ue(&outcome.nodes, cell_factor, world_scale);

    let _ = result_tx.send(WorkerResult::PathComputed {
        request_id,
        status: outcome.status as u8,
        nodes_ue,
    });
}
