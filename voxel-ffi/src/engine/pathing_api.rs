use std::collections::HashSet;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};

use crossbeam_channel::{bounded, Receiver, Sender};
use dashmap::DashMap;
use voxel_fluid::FluidConfig;
use voxel_fluid::FluidEvent;
use voxel_core::stress::StressField;
use voxel_core::world_scan::ScanConfig;
use voxel_gen::config::{
    BandedIronConfig, CrystalConfig, FormationConfig, GenerationConfig, GeodeConfig, HostRockConfig,
    KimberlitePipeConfig, MineConfig, NoiseConfig, OreConfig, OreCrystalConfig, OreVeinParams,
    PoolConfig, StressConfig, SulfideBlobConfig, WormConfig,
};

use crate::convert::ue_chunk_to_rust;
use crate::pathing::{
    build_request_from_ue, FfiPathNode, FfiPathRequest, FfiPathResult, PathResultStore,
    StashedPathResult,
};
use crate::profiler::StreamingProfiler;
use crate::store::ChunkStore;
use crate::types::*;
use crate::worker::{path_worker_loop, worker_loop};

use super::VoxelEngine;

impl VoxelEngine {
    // ─── Pathfinding public API ─────────────────────────────────

    /// Submit a path request. Returns the request id (>= 1) on success;
    /// returns 0 if the path channel is full or the request was malformed.
    /// The returned id is later passed to `poll_path` to fetch the result.
    pub fn request_path(&self, request: FfiPathRequest) -> u32 {
        let request_id = self.next_path_request_id.fetch_add(1, Ordering::Relaxed);
        let internal = build_request_from_ue(
            request_id,
            request.from_ue_x, request.from_ue_y, request.from_ue_z,
            request.to_ue_x, request.to_ue_y, request.to_ue_z,
            request.agent_radius_ue,
            request.movement_mode,
            request.max_nodes,
            self.world_scale,
        );
        match self.path_tx.try_send(WorkerRequest::ComputePath { request: internal }) {
            Ok(()) => request_id,
            Err(_) => 0,
        }
    }

    /// Poll for a completed path result.
    /// Returns 0 = pending (request id known, A* not finished),
    ///         1 = result populated (caller must voxel_path_free),
    ///         2 = unknown id (never submitted, or already polled, or expired).
    /// On status 1, `out` is filled with a heap-allocated FfiPathResult — caller
    /// owns the memory and MUST call `voxel_path_free` to release it.
    pub fn poll_path(&self, request_id: u32) -> Option<FfiPathResult> {
        let mut store = self.path_results.lock().ok()?;
        let result = store.take(request_id)?;

        // Heap-allocate the node array; UE takes ownership.
        let node_count = result.nodes_ue.len() as u32;
        let nodes_ptr: *mut FfiPathNode = if node_count == 0 {
            std::ptr::null_mut()
        } else {
            let mut boxed: Box<[FfiPathNode]> = result
                .nodes_ue
                .iter()
                .map(|n| FfiPathNode {
                    x: n.x, y: n.y, z: n.z,
                    nx: n.nx, ny: n.ny, nz: n.nz,
                })
                .collect::<Vec<_>>()
                .into_boxed_slice();
            let ptr = boxed.as_mut_ptr();
            std::mem::forget(boxed); // ownership transferred to caller
            ptr
        };

        Some(FfiPathResult {
            request_id: result.request_id,
            status: result.status as u8,
            _pad: [0; 3],
            nodes: nodes_ptr,
            node_count,
            _pad2: 0,
        })
    }

    /// Release the node array of an FfiPathResult previously returned by
    /// `poll_path`. Idempotent on null pointer.
    pub fn free_path_nodes(&self, nodes: *mut FfiPathNode, count: u32) {
        if nodes.is_null() || count == 0 {
            return;
        }
        unsafe {
            // Reconstitute the Box<[FfiPathNode]> we leaked in poll_path.
            let slice = std::slice::from_raw_parts_mut(nodes, count as usize);
            let _boxed: Box<[FfiPathNode]> = Box::from_raw(slice);
        }
    }
}
