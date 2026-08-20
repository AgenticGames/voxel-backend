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
use std::collections::{HashMap, HashSet};
use voxel_path::{CellGrid, MovementMode, PathNode, PathStatus};

use crate::convert::from_ue_world_pos;
use crate::store::ChunkStore;

/// Cell factor for the live ChunkStore-backed path planner — one pathing cell
/// covers a 2×2×2 voxel block. Public so both the async path worker (in
/// `worker.rs`) and the synchronous `voxel_query_path_exists` FFI can share
/// the same scale without drifting.
pub const PATH_CELL_FACTOR: i32 = 2;

/// CellGrid implementation backed by a live ChunkStore reference.
///
/// Cells are addressed in "pathing-cell" coordinates — multiply by `cell_factor`
/// to get voxel coordinates.
///
/// `occupied_cells` is the optional cross-species avoidance layer: cells in
/// the set are treated as solid (impassable) by A* — used so spiders/wasps/
/// creatures route around each other instead of phasing through. The agent's
/// OWN cell (`requester_cell`) is excluded so it can start its search from
/// inside its current cell even when the snapshot still says that cell is
/// occupied (it's about to vacate it as it moves anyway).
pub struct ChunkStoreGrid<'a> {
    pub store: &'a ChunkStore,
    pub chunk_size: usize,
    pub cell_factor: i32,
    pub occupied_cells: Option<&'a HashSet<(i32, i32, i32)>>,
    pub requester_cell: Option<IVec3>,
    /// Treat cells in UNLOADED chunks as open instead of solid. Opt-in for
    /// guidance queries (the sense trail): the dormancy bank keeps only a
    /// small bubble of chunks resident around the player, so with the default
    /// "unloaded = solid" a long route dies against a wall of ignorance a few
    /// hundred UU out. Open-unknown lets the search assume cave where it has
    /// no data — A* taxes those cells (epistemic penalty) so KNOWN cave is
    /// preferred, and the result is flagged PartiallyUnloaded so callers can
    /// replan as reality streams in. AI requests never set this: an enemy
    /// must not chase through rock it merely hopes is open.
    pub unknown_open: bool,
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

    // ─── TODO: lava-as-solid for ground modes (audit S8, 2026-05-22) ──
    // Spiders currently path freely through lava pools because lava lives in
    // the voxel-fluid grid (separate from density), and the fluid sim owns
    // its state behind an event-based channel — no Arc<RwLock<>> is shared
    // with the path workers. To plumb this:
    //   1. Add a "lava cell mask" shared via Arc<RwLock<HashMap<chunk, BitSet>>>
    //      in `engine.rs`, populated by the fluid sim when lava cells flip.
    //   2. Pass the mask into the path-worker spawn alongside `store`.
    //   3. Extend ChunkStoreGrid with `lava_mask: Option<&'a LavaCellMask>` and
    //      `agent_can_fly: bool` (derived from MovementMode at grid
    //      construction in worker.rs::handle_path_request).
    //   4. Here: `if !self.agent_can_fly && mask.is_lava(cell) { return true; }`
    // Skipped this batch — non-trivial cross-crate plumbing.
    fn is_solid(&self, cell: IVec3) -> bool {
        let (chunk_key, (lx, ly, lz)) = self.cell_to_chunk_local(cell);
        let static_solid = match self.store.density_fields.get(&chunk_key) {
            Some(field) => field.get(lx, ly, lz).material.is_solid(),
            // Unloaded: impassable barrier by default; assumed-open for
            // opt-in guidance queries (see `unknown_open` above).
            None => !self.unknown_open,
        };
        if static_solid {
            return true;
        }

        // Cross-species avoidance: another AI agent's cell. Exclude the
        // requester's own cell so A* can search from inside it even when
        // the snapshot says it's occupied (we are the occupant). Adjacent
        // occupied cells become walls and A* routes around them naturally.
        if let Some(occupied) = self.occupied_cells {
            let is_requester = matches!(self.requester_cell, Some(rc) if rc == cell);
            if !is_requester && occupied.contains(&(cell.x, cell.y, cell.z)) {
                return true;
            }
        }
        false
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
    pub smooth: bool,
    pub fine_cells: bool,
    pub unknown_open: bool,
}

/// Build the internal request from UE-space inputs. Caller supplies
/// `world_scale` (UE units per voxel — typically 40).
pub fn build_request_from_ue(
    request_id: u32,
    from_ue_x: f32, from_ue_y: f32, from_ue_z: f32,
    to_ue_x: f32, to_ue_y: f32, to_ue_z: f32,
    agent_radius_ue: f32,
    movement_mode: u8,
    smooth_disable: u8,
    fine_cells: u8,
    unknown_open: u8,
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
        smooth: smooth_disable == 0,
        fine_cells: fine_cells != 0,
        unknown_open: unknown_open != 0,
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
        smooth: internal.smooth,
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
    /// 1 = skip theta* LOS smoothing and return the raw corridor-following
    /// cell path. Smoothing's zero-width cell ray collapses a route to a few
    /// long chords that can shave mesh the voxel grid calls open — fine for
    /// AI steering, fatal for the sense trail's swept-ribbon validation
    /// (#206). Carved out of the old `_pad`, so the C layout is unchanged
    /// and zero-initialized callers keep smoothing on.
    pub smooth_disable: u8,
    /// 1 = plan on SINGLE-VOXEL cells (cell_factor 1) instead of the default
    /// 2×2×2 blocks. The block grid samples one voxel of eight for solidity,
    /// so on slopes A* threads "open" cells that are mostly rock — 26 of 91
    /// sense-trail corridor points landed inside UE collision on the #206
    /// route. Fine cells are exact (and let one-voxel passages path at all);
    /// AI keeps the coarse grid, whose costs are tuned around it. Also skips
    /// the cross-species occupancy layer (quantized at the coarse factor,
    /// and a guidance ribbon shouldn't dodge wasps anyway).
    pub fine_cells: u8,
    /// 1 = treat unloaded chunks as OPEN (taxed) instead of solid — see
    /// `ChunkStoreGrid::unknown_open`. Sense-trail guidance only; carved
    /// from the last `_pad` byte, so the C layout is unchanged and
    /// zero-initialized callers (all AI) keep unloaded-as-solid.
    pub unknown_open: u8,
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
