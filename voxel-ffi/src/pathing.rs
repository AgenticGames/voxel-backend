//! Live-world `CellGrid` impl over `ChunkStore`, plus the path-result types
//! and conversions used by the FFI worker.
//!
//! Pathing cells are larger than voxels: `cell_factor = 2` means one pathing
//! cell covers a 2×2×2 voxel block. The trait's `is_solid` samples the lower-
//! corner voxel of the block — for typical tunnel-wide gameplay terrain that's
//! a good-enough approximation. If AI starts clipping through thin walls,
//! upgrade to conservative bool-OR over all 8 sub-voxels.
//!
//! Unloaded chunks return `is_solid = true` AND `is_loaded = false`. A* uses
//! the second signal to flag `PathStatus::PartiallyUnloaded` results.

use glam::{IVec3, Vec3};
use std::collections::HashMap;
use voxel_path::{CellGrid, MovementMode, PathNode, PathStatus};

use crate::convert::from_ue_world_pos;
use crate::store::ChunkStore;

/// CellGrid implementation backed by a live ChunkStore reference.
///
/// Cells are addressed in "pathing-cell" coordinates — multiply by `cell_factor`
/// to get voxel coordinates.
pub struct ChunkStoreGrid<'a> {
    pub store: &'a ChunkStore,
    pub chunk_size: usize,
    pub cell_factor: i32,
}

impl<'a> ChunkStoreGrid<'a> {
    /// Convert a pathing-cell coord to (chunk_coord, local_voxel_coord).
    #[inline]
    fn cell_to_chunk_local(&self, cell: IVec3) -> ((i32, i32, i32), (usize, usize, usize)) {
        let vx = cell.x * self.cell_factor;
        let vy = cell.y * self.cell_factor;
        let vz = cell.z * self.cell_factor;
        let cs = self.chunk_size as i32;
        let cx = vx.div_euclid(cs);
        let cy = vy.div_euclid(cs);
        let cz = vz.div_euclid(cs);
        let lx = vx.rem_euclid(cs) as usize;
        let ly = vy.rem_euclid(cs) as usize;
        let lz = vz.rem_euclid(cs) as usize;
        ((cx, cy, cz), (lx, ly, lz))
    }
}

impl<'a> CellGrid for ChunkStoreGrid<'a> {
    fn cell_size(&self) -> f32 {
        self.cell_factor as f32
    }

    fn is_solid(&self, cell: IVec3) -> bool {
        let (chunk_key, (lx, ly, lz)) = self.cell_to_chunk_local(cell);
        match self.store.density_fields.get(&chunk_key) {
            Some(field) => field.get(lx, ly, lz).material.is_solid(),
            None => true, // unloaded → impassable barrier
        }
    }

    fn is_loaded(&self, cell: IVec3) -> bool {
        let (chunk_key, _) = self.cell_to_chunk_local(cell);
        self.store.density_fields.contains_key(&chunk_key)
    }

    fn surface_normal_at(&self, cell: IVec3) -> Vec3 {
        let mut n = Vec3::ZERO;
        for (dx, dy, dz) in [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ] {
            let neighbor = IVec3::new(cell.x + dx, cell.y + dy, cell.z + dz);
            if self.is_solid(neighbor) {
                // Vector pointing AWAY from the solid (i.e. into the air cell).
                n -= Vec3::new(dx as f32, dy as f32, dz as f32);
            }
        }
        n.normalize_or_zero()
    }
}

// ─── Path types crossing the FFI boundary ─────────────────────────

/// Path-result struct stashed on the engine, keyed by `request_id`.
/// Stays in UE-space coords from worker → poll → FFI caller — only the search
/// itself operates in Rust voxel space.
#[derive(Debug, Clone)]
pub struct StashedPathResult {
    pub request_id: u32,
    pub status: PathStatus,
    pub nodes_ue: Vec<PathNodeUE>,
    pub stash_time: std::time::Instant,
}

/// A single path waypoint already transformed to UE world coordinates.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct PathNodeUE {
    pub x: f32, pub y: f32, pub z: f32,
    pub nx: f32, pub ny: f32, pub nz: f32,
}

/// Internal request data — pre-converted to Rust coordinates so the worker
/// thread doesn't redo the transform.
#[derive(Debug, Clone)]
pub struct PathRequestInternal {
    pub request_id: u32,
    pub from_voxel: glam::Vec3,
    pub to_voxel: glam::Vec3,
    pub agent_radius_voxels: f32,
    pub movement_mode_kind: u8, // 0=Flying, 1=Walking, 2=Surface
    pub max_nodes: u32,
}

/// Build the internal request from UE-space inputs. Caller supplies
/// `world_scale` (UE units per voxel — typically 40).
pub fn build_request_from_ue(
    request_id: u32,
    from_ue_x: f32, from_ue_y: f32, from_ue_z: f32,
    to_ue_x: f32, to_ue_y: f32, to_ue_z: f32,
    agent_radius_ue: f32,
    movement_mode: u8,
    max_nodes: u32,
    world_scale: f32,
) -> PathRequestInternal {
    PathRequestInternal {
        request_id,
        from_voxel: from_ue_world_pos(from_ue_x, from_ue_y, from_ue_z, world_scale),
        to_voxel: from_ue_world_pos(to_ue_x, to_ue_y, to_ue_z, world_scale),
        agent_radius_voxels: agent_radius_ue / world_scale,
        movement_mode_kind: movement_mode,
        max_nodes,
    }
}

/// Convert PathRequestInternal → `voxel_path::PathRequest` ready for A*.
/// Cells are voxel coordinates / cell_factor.
pub fn to_path_request(
    internal: &PathRequestInternal,
    cell_factor: i32,
) -> (voxel_path::PathRequest, MovementMode) {
    let cf = cell_factor as f32;
    let from_cell = IVec3::new(
        (internal.from_voxel.x / cf).floor() as i32,
        (internal.from_voxel.y / cf).floor() as i32,
        (internal.from_voxel.z / cf).floor() as i32,
    );
    let to_cell = IVec3::new(
        (internal.to_voxel.x / cf).floor() as i32,
        (internal.to_voxel.y / cf).floor() as i32,
        (internal.to_voxel.z / cf).floor() as i32,
    );
    let agent_radius_cells = internal.agent_radius_voxels / cf;
    let mode = match internal.movement_mode_kind {
        1 => MovementMode::Walking { agent_radius_cells },
        2 => MovementMode::Surface { agent_radius_cells },
        _ => MovementMode::Flying { agent_radius_cells },
    };
    let req = voxel_path::PathRequest {
        from: from_cell,
        to: to_cell,
        mode,
        max_nodes: internal.max_nodes,
        smooth: true,
    };
    (req, mode)
}

/// Convert path nodes (in voxel-cell coords) to UE-space waypoints.
///
/// `cell_factor` is multiplied in to get voxel coords; centered by +0.5 cells
/// so the waypoint sits in the middle of its pathing cell. Then the standard
/// Rust→UE coord swap is applied with `world_scale`.
pub fn nodes_to_ue(
    nodes: &[PathNode],
    cell_factor: i32,
    world_scale: f32,
) -> Vec<PathNodeUE> {
    let cf = cell_factor as f32;
    let half = cf * 0.5;
    nodes
        .iter()
        .map(|n| {
            let vx = n.cell.x as f32 * cf + half;
            let vy = n.cell.y as f32 * cf + half;
            let vz = n.cell.z as f32 * cf + half;
            // Rust (x, y, z) → UE (x * scale, -z * scale, y * scale)
            let ue_x = vx * world_scale;
            let ue_y = -vz * world_scale;
            let ue_z = vy * world_scale;
            // Normal transform: (nx, ny, nz) → (nx, -nz, ny). Already unit-length.
            PathNodeUE {
                x: ue_x, y: ue_y, z: ue_z,
                nx: n.surface_normal.x,
                ny: -n.surface_normal.z,
                nz: n.surface_normal.y,
            }
        })
        .collect()
}

/// Engine-side stash — request_id → (status, nodes). Pruned by TTL on every
/// poll to prevent leaks when UE never collects a result (agent died, etc).
#[derive(Default)]
pub struct PathResultStore {
    pub results: HashMap<u32, StashedPathResult>,
}

impl PathResultStore {
    /// Drop entries older than `ttl_secs`.
    pub fn prune(&mut self, ttl_secs: u64) {
        let now = std::time::Instant::now();
        self.results.retain(|_, r| {
            now.duration_since(r.stash_time).as_secs() < ttl_secs
        });
    }

    pub fn stash(&mut self, result: StashedPathResult) {
        self.results.insert(result.request_id, result);
    }

    pub fn take(&mut self, request_id: u32) -> Option<StashedPathResult> {
        self.results.remove(&request_id)
    }
}

// ─── FFI structs (C ABI) — mirrored exactly in VoxelFFI.h ──────────

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiPathRequest {
    pub from_ue_x: f32, pub from_ue_y: f32, pub from_ue_z: f32,
    pub to_ue_x: f32, pub to_ue_y: f32, pub to_ue_z: f32,
    pub agent_radius_ue: f32,
    pub movement_mode: u8,   // 0=Flying, 1=Walking, 2=Surface
    pub _pad: [u8; 3],       // explicit padding to match C layout
    pub max_nodes: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiPathNode {
    pub x: f32, pub y: f32, pub z: f32,
    pub nx: f32, pub ny: f32, pub nz: f32,
}

#[repr(C)]
#[derive(Debug)]
pub struct FfiPathResult {
    pub request_id: u32,
    pub status: u8,               // PathStatus discriminant
    pub _pad: [u8; 3],
    pub nodes: *mut FfiPathNode,  // heap-allocated; UE must call voxel_path_free
    pub node_count: u32,
    pub _pad2: u32,
}
