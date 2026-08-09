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

use super::{terrace_size_for_scale, VoxelEngine};

impl VoxelEngine {
    /// Queue a support placement request.
    pub fn request_place_support(&self, world_x: i32, world_y: i32, world_z: i32, support_type: u8) -> u32 {
        match self.mine_tx.try_send(WorkerRequest::PlaceSupport {
            world_x, world_y, world_z, support_type,
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Queue a support removal request.
    pub fn request_remove_support(&self, world_x: i32, world_y: i32, world_z: i32) -> u32 {
        match self.mine_tx.try_send(WorkerRequest::RemoveSupport {
            world_x, world_y, world_z,
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Synchronously read a strut's HP/type at a Rust voxel position.
    /// Returns `(support_type, hp, max_hp, valid)` where valid=0 means the
    /// store lock was contended — UE should hold the previous bar value
    /// and retry next frame. Used by the UE strut-inspect HP bar widget.
    pub fn query_strut_hp(&self, world_x: i32, world_y: i32, world_z: i32) -> (u8, u16, u16, u8) {
        use voxel_core::stress::{world_to_chunk_local, STRUT_TUNING};
        let cfg = self.config.read().unwrap();
        let chunk_size = cfg.chunk_size;
        drop(cfg);
        let (key, lx, ly, lz) = world_to_chunk_local(world_x, world_y, world_z, chunk_size);
        // Retry 5x/2ms on contention — mirrors voxel_query_stress pattern.
        for _attempt in 0..5 {
            if let Ok(s) = self.store.try_read() {
                if let Some(supf) = s.support_fields.get(&key) {
                    let stype = supf.get(lx, ly, lz);
                    let hp = supf.get_hp(lx, ly, lz);
                    let max_hp = STRUT_TUNING[stype as u8 as usize].max_hp;
                    return (stype as u8, hp, max_hp, 1);
                }
                return (0, 0, 0, 1); // chunk loaded but no support field — no strut
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        (0, 0, 0, 0) // contended, signal UE to keep prior state
    }

    /// Request flattening a terrace at a UE world position.
    /// Snaps to a terrace_size-aligned grid on all axes and determines host rock from depth.
    /// Returns 1 on success, 0 if queue full.
    pub fn request_flatten(&self, ue_x: f32, ue_y: f32, ue_z: f32, scale: f32) -> u32 {
        let rust_x = ue_x / scale;
        let rust_y = ue_z / scale;
        let rust_z = -ue_y / scale;

        let ts = terrace_size_for_scale(scale);
        let base_x = (rust_x as i32).div_euclid(ts) * ts;
        let base_y = (rust_y as i32).div_euclid(ts) * ts;
        let base_z = (rust_z as i32).div_euclid(ts) * ts;

        let host_material = {
            let cfg = self.config.read().unwrap();
            voxel_gen::density::host_rock_for_depth(rust_y as f64, &cfg.ore.host_rock) as u8
        };

        match self.mine_tx.try_send(WorkerRequest::Flatten {
            base_x,
            base_y,
            base_z,
            host_material,
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Request flattening a batch of terrace tiles in a single lock + remesh pass.
    /// Each UE position is converted to Rust space and snapped to the terrace grid independently.
    /// Duplicate tile positions are deduplicated. Returns 1 on success, 0 if queue full.
    pub fn request_flatten_batch(&self, ue_positions: &[(f32, f32, f32)], scale: f32) -> u32 {
        if ue_positions.is_empty() {
            return 0;
        }

        let ts = terrace_size_for_scale(scale);
        let mut seen: std::collections::HashSet<glam::IVec3> = std::collections::HashSet::new();
        let mut tiles: Vec<(glam::IVec3, voxel_core::material::Material)> = Vec::new();

        let cfg = self.config.read().unwrap();
        for &(ue_x, ue_y, ue_z) in ue_positions {
            let rust_x = ue_x / scale;
            let rust_y = ue_z / scale;
            let rust_z = -ue_y / scale;

            let base_x = (rust_x as i32).div_euclid(ts) * ts;
            let base_y = (rust_y as i32).div_euclid(ts) * ts;
            let base_z = (rust_z as i32).div_euclid(ts) * ts;
            let key = glam::IVec3::new(base_x, base_y, base_z);

            if seen.insert(key) {
                let mat = voxel_gen::density::host_rock_for_depth(rust_y as f64, &cfg.ore.host_rock);
                tiles.push((key, mat));
            }
        }
        drop(cfg);

        match self.mine_tx.try_send(WorkerRequest::FlattenBatch { tiles }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Query whether a terrace exists at a UE world position.
    /// Returns Some(material_id) if terraced, None otherwise.
    pub fn query_terrace(&self, ue_x: f32, ue_y: f32, ue_z: f32, scale: f32) -> Option<u8> {
        let rust_x = ue_x / scale;
        let rust_y = ue_z / scale;
        let rust_z = -ue_y / scale;

        let ts = terrace_size_for_scale(scale);
        let base_x = (rust_x as i32).div_euclid(ts) * ts;
        let base_y = (rust_y as i32).div_euclid(ts) * ts;
        let base_z = (rust_z as i32).div_euclid(ts) * ts;

        let store = self.store.try_read().ok()?;
        crate::terrain_ops::query_terrace(&*store, glam::IVec3::new(base_x, base_y, base_z), ts)
            .map(|m| m as u8)
    }

    /// Query floor support for a flatten ghost preview.
    /// Returns (solid_count, clearance_count, snapped_ue_x, snapped_ue_y, snapped_ue_z).
    pub fn query_flatten_support(&self, ue_x: f32, ue_y: f32, ue_z: f32, scale: f32) -> (u8, u8, f32, f32, f32) {
        let rust_x = ue_x / scale;
        let rust_y = ue_z / scale;
        let rust_z = -ue_y / scale;

        let ts = terrace_size_for_scale(scale);
        let base_x = (rust_x as i32).div_euclid(ts) * ts;
        let base_y = (rust_y as i32).div_euclid(ts) * ts;
        let base_z = (rust_z as i32).div_euclid(ts) * ts;

        let cs = { self.config.read().unwrap().chunk_size as i32 };
        let store = match self.store.try_read() {
            Ok(s) => s,
            Err(_) => return (0, 0, 0.0, 0.0, 0.0),
        };
        let (count, clearance) = crate::terrain_ops::query_flatten_support(&*store, glam::IVec3::new(base_x, base_y, base_z), cs, ts);

        // Convert snapped position back to UE coords
        let snapped_ue_x = base_x as f32 * scale;
        let snapped_ue_y = -(base_z as f32) * scale;
        let snapped_ue_z = base_y as f32 * scale;

        (count, clearance, snapped_ue_x, snapped_ue_y, snapped_ue_z)
    }

    /// Query floor support for a building placement.
    /// footprint_voxels controls the NxN footprint (e.g. 4 = 4x4, 2 = 2x2).
    /// Returns (solid_count, total_columns, host_mat_u8, snapped_ue_x, snapped_ue_y, snapped_ue_z).
    /// The returned UE position is the authoritative floor surface center.
    pub fn query_building_support(&self, ue_x: f32, ue_y: f32, ue_z: f32, scale: f32, footprint_voxels: i32) -> (u8, u8, u8, f32, f32, f32) {
        let rust_x = ue_x / scale;
        let rust_y = ue_z / scale;
        let rust_z = -ue_y / scale;

        let bts = footprint_voxels.max(1);
        // Center the footprint on the building position (UE already snapped XY)
        let base_x = rust_x.round() as i32 - bts / 2;
        let base_z = rust_z.round() as i32 - bts / 2;

        let cs = { self.config.read().unwrap().chunk_size as i32 };
        let store = match self.store.try_read() {
            Ok(s) => s,
            Err(_) => return (0, 0, 0, ue_x, ue_y, ue_z),
        };

        // Find actual surface Y at footprint center by scanning density field
        let center_x = base_x + bts / 2;
        let center_z = base_z + bts / 2;
        let approx_y = rust_y.round() as i32;
        let mut surface_y = approx_y;

        let probe_cx = center_x.div_euclid(cs);
        let probe_cz = center_z.div_euclid(cs);
        let probe_lx = center_x.rem_euclid(cs) as usize;
        let probe_lz = center_z.rem_euclid(cs) as usize;

        let probe_cy = approx_y.div_euclid(cs);
        let probe_ly = approx_y.rem_euclid(cs) as usize;

        let is_solid_at_approx = store.density_fields
            .get(&(probe_cx, probe_cy, probe_cz))
            .map(|df| df.get(probe_lx, probe_ly, probe_lz).density > 0.0)
            .unwrap_or(false);

        if is_solid_at_approx {
            // Inside solid — scan UP to find first air
            for dy in 1..=8i32 {
                let sy = approx_y + dy;
                let scy = sy.div_euclid(cs);
                let sly = sy.rem_euclid(cs) as usize;
                if let Some(df) = store.density_fields.get(&(probe_cx, scy, probe_cz)) {
                    if df.get(probe_lx, sly, probe_lz).density <= 0.0 {
                        surface_y = sy - 1; // Last solid voxel before air
                        break;
                    }
                }
            }
        } else {
            // In air — scan DOWN to find first solid
            for dy in 1..=8i32 {
                let sy = approx_y - dy;
                let scy = sy.div_euclid(cs);
                let sly = sy.rem_euclid(cs) as usize;
                if let Some(df) = store.density_fields.get(&(probe_cx, scy, probe_cz)) {
                    if df.get(probe_lx, sly, probe_lz).density > 0.0 {
                        surface_y = sy; // Top solid voxel
                        break;
                    }
                }
            }
        }

        let base_y = surface_y;
        let (solid, total, mat) = crate::terrain_ops::query_building_support(&*store, glam::IVec3::new(base_x, base_y, base_z), cs, bts);

        // Convert footprint center + surface back to UE coords
        let center_vx = base_x as f32 + bts as f32 / 2.0;
        let center_vz = base_z as f32 + bts as f32 / 2.0;
        let snapped_ue_x = center_vx * scale;
        let snapped_ue_y = -(center_vz * scale);
        let snapped_ue_z = base_y as f32 * scale;

        (solid, total, mat as u8, snapped_ue_x, snapped_ue_y, snapped_ue_z)
    }

    // ─── Cell-rect building API ──────────────────────────────────────────
    //
    // The float entry points above take a UE world position and then snap and
    // re-centre it themselves. UE also snaps, on its own per-type grid. Two
    // independent snappers is exactly how the placement grid drifted: they
    // agreed for even footprints and disagreed for odd ones, and neither side
    // owned the answer.
    //
    // These take an ALREADY-SNAPPED cell rect. UE owns snapping; Rust owns
    // terrain truth. Nothing here rounds, re-centres or second-guesses the
    // caller's rect.
    //
    // COORD CONTRACT — per-function, do NOT generalise to other exports:
    //   Inputs are UE-space CELL indices, where cell `i` spans world
    //   [i*scale, (i+1)*scale). Mapping to Rust space:
    //       rust_x =  ue_x
    //       rust_y =  ue_z                      (UE Z is up; Rust Y is up)
    //       rust_z = -ue_y                      (negated, so a UE Y span
    //                [y0, y0+size) becomes a Rust Z span from -(y0+size))
    //   Cells in, cells out — no `scale` is needed or accepted, which is the
    //   point: a scale parameter is what invites a second rounding step.

    /// Support under an already-snapped footprint rect.
    ///
    /// Returns `(solid_columns, total_columns, host_material, surface_cell_ue_z)`
    /// where `surface_cell_ue_z` indexes the TOP SOLID cell — the caller's
    /// building body starts one cell above it.
    pub fn query_building_support_cells(
        &self,
        ue_min_x: i32,
        ue_min_y: i32,
        ue_approx_z: i32,
        size_x: i32,
        size_y: i32,
    ) -> (u8, u8, u8, i32) {
        let sx = size_x.max(1);
        let sy = size_y.max(1);
        let (rust_x0, rust_z0) = crate::terrain_ops::ue_cell_rect_to_rust_xz(ue_min_x, ue_min_y, sy);

        let cs = { self.config.read().unwrap().chunk_size as i32 };
        let store = match self.store.try_read() {
            Ok(s) => s,
            // Worker holds the write lock mid-carve: report "no support" rather
            // than blocking the game thread on a placement preview.
            Err(_) => return (0, 0, 0, ue_approx_z),
        };

        // Walk the centre column to the real surface. The caller's approx Z is
        // a ray hit, which lands anywhere within a cell of the truth.
        let centre_x = rust_x0 + sx / 2;
        let centre_z = rust_z0 + sy / 2;
        let probe_cx = centre_x.div_euclid(cs);
        let probe_cz = centre_z.div_euclid(cs);
        let probe_lx = centre_x.rem_euclid(cs) as usize;
        let probe_lz = centre_z.rem_euclid(cs) as usize;

        let solid_at = |y: i32| -> bool {
            store
                .density_fields
                .get(&(probe_cx, y.div_euclid(cs), probe_cz))
                .map(|df| df.get(probe_lx, y.rem_euclid(cs) as usize, probe_lz).density > 0.0)
                .unwrap_or(false)
        };

        let mut surface_y = ue_approx_z;
        if solid_at(ue_approx_z) {
            // Buried — climb to the last solid cell before air.
            for dy in 1..=8i32 {
                if !solid_at(ue_approx_z + dy) {
                    surface_y = ue_approx_z + dy - 1;
                    break;
                }
            }
        } else {
            // In air — drop to the first solid cell.
            for dy in 1..=8i32 {
                if solid_at(ue_approx_z - dy) {
                    surface_y = ue_approx_z - dy;
                    break;
                }
            }
        }

        let (solid, total, mat) = crate::terrain_ops::query_building_support_rect(
            &*store,
            glam::IVec3::new(rust_x0, surface_y, rust_z0),
            cs,
            sx,
            sy,
        );
        (solid, total, mat as u8, surface_y)
    }

    /// Flatten a pad under an already-snapped footprint rect.
    ///
    /// `ue_base_z` is the surface cell from `query_building_support_cells`.
    ///
    /// ⚠️ The SDF carve underneath (`flatten_terrace_sdf`) is square-only, so a
    /// non-square footprint gets a CENTRED SQUARE pad of `max(size_x, size_y)`.
    /// That is deliberate: an oversized flat pad under a building is harmless,
    /// an undersized one would leave it hanging over a hole. Every shipping
    /// footprint is square today, so this is exact; make the carve rectangular
    /// before shipping a non-square building.
    pub fn request_building_flatten_cells(
        &self,
        ue_min_x: i32,
        ue_min_y: i32,
        ue_base_z: i32,
        size_x: i32,
        size_y: i32,
        clearance_cells: i32,
    ) -> u32 {
        let sx = size_x.max(1);
        let sy = size_y.max(1);

        // Carve the rect as given. This USED to square it off —
        // `pad = sx.max(sy)` centred by `(pad - sx) / 2` — on the stated
        // grounds that centring was "a no-op while footprints are square".
        // Footprints are square in BUILDING CELLS, not in voxels: the 40 UU
        // cell lattice and the WorldScale voxel lattice are incommensurate,
        // so BuildingCellsToVoxelSpanXY rounds outward to different sizes per
        // axis. A furnace's 240 UU span is 8 or 9 voxels depending where it
        // lands, and X and Y resolve independently. With sx=8, sy=9 the pad
        // was 9 and the offset `1 / 2` truncated to 0, so the whole extra
        // voxel landed on one side — 30 UU more room there and none opposite.
        // Reported from play as one side of a furnace always being roomier.
        let (rx0, rz0) = crate::terrain_ops::ue_cell_rect_to_rust_xz(ue_min_x, ue_min_y, sy);
        let base_x = rx0;
        let base_z = rz0;

        let host_material = {
            let cfg = self.config.read().unwrap();
            voxel_gen::density::host_rock_for_depth(ue_base_z as f64, &cfg.ore.host_rock) as u8
        };

        match self.mine_tx.send_timeout(
            WorkerRequest::BuildingFlatten {
                base_x,
                base_y: ue_base_z,
                base_z,
                // Cells are exact integers here — there is no sub-voxel
                // remainder to carry, unlike the float entry point where UE's
                // world Z could land anywhere inside a cell.
                base_y_float: ue_base_z as f32,
                host_material,
                footprint_x: sx,
                footprint_z: sy,
                clearance_voxels: clearance_cells.max(2),
            },
            std::time::Duration::from_millis(100),
        ) {
            Ok(()) => 1,
            Err(e) => {
                eprintln!("[voxel] request_building_flatten_cells: send failed: {}", e);
                0
            }
        }
    }

    /// Batch form of `request_building_flatten_cells` — one worker job, one
    /// seam pass, for a whole belt drag chain.
    pub fn request_building_flatten_cells_batch(
        &self,
        rects: &[(i32, i32, i32)],
        size_x: i32,
        size_y: i32,
        clearance_cells: i32,
    ) -> u32 {
        if rects.is_empty() {
            return 0;
        }
        let sx = size_x.max(1);
        let sy = size_y.max(1);
        let clr = clearance_cells.max(2);
        let cfg = self.config.read().unwrap();
        // Same square-pad bias the single path had — see the note there.
        let buildings: Vec<(i32, i32, i32, f32, u8, i32, i32, i32)> = rects
            .iter()
            .map(|&(ue_min_x, ue_min_y, ue_base_z)| {
                let (base_x, base_z) =
                    crate::terrain_ops::ue_cell_rect_to_rust_xz(ue_min_x, ue_min_y, sy);
                let host_material =
                    voxel_gen::density::host_rock_for_depth(ue_base_z as f64, &cfg.ore.host_rock)
                        as u8;
                (base_x, ue_base_z, base_z, ue_base_z as f32, host_material, sx, sy, clr)
            })
            .collect();
        drop(cfg);

        match self.mine_tx.send_timeout(
            WorkerRequest::BuildingFlattenBatch { buildings },
            std::time::Duration::from_millis(500),
        ) {
            Ok(()) => 1,
            Err(e) => {
                eprintln!("[voxel] request_building_flatten_cells_batch: send failed: {}", e);
                0
            }
        }
    }

    /// Request auto-terrace for a building placement.
    /// footprint_voxels controls the NxN footprint (e.g. 4 = 4x4, 2 = 2x2).
    /// clearance_voxels controls how many air voxels to carve above the floor.
    /// Returns 1 on success, 0 if queue full.
    pub fn request_building_flatten(&self, ue_x: f32, ue_y: f32, ue_z: f32, scale: f32, footprint_voxels: i32, clearance_voxels: i32) -> u32 {
        let rust_x = ue_x / scale;
        let rust_y = ue_z / scale;
        let rust_z = -ue_y / scale;

        let bts = footprint_voxels.max(1);
        // Center the footprint on the building position (UE already snapped).
        let base_x = rust_x.round() as i32 - bts / 2;
        // KEEP the exact float Y for sub-voxel surface placement; the integer
        // base_y is just floor() for chunk indexing.
        let base_y_float = rust_y;
        let base_y = rust_y.floor() as i32;
        let base_z = rust_z.round() as i32 - bts / 2;

        let host_material = {
            let cfg = self.config.read().unwrap();
            voxel_gen::density::host_rock_for_depth(rust_y as f64, &cfg.ore.host_rock) as u8
        };

        match self.mine_tx.send_timeout(WorkerRequest::BuildingFlatten {
            base_x,
            base_y,
            base_z,
            base_y_float,
            host_material,
            footprint_x: bts,
            footprint_z: bts,
            clearance_voxels: clearance_voxels.max(2),
        }, std::time::Duration::from_millis(100)) {
            Ok(()) => 1,
            Err(e) => {
                eprintln!("[voxel] request_building_flatten: send failed: {}", e);
                0
            }
        }
    }

    /// Batch building flatten: multiple buildings in one worker job, one seam pass.
    pub fn request_building_flatten_batch(
        &self,
        ue_positions: &[(f32, f32, f32)],
        scale: f32,
        footprint_voxels: i32,
        clearance_voxels: i32,
    ) -> u32 {
        if ue_positions.is_empty() {
            return 0;
        }
        let bts = footprint_voxels.max(1);
        let clr = clearance_voxels.max(2);
        let cfg = self.config.read().unwrap();
        let buildings: Vec<(i32, i32, i32, f32, u8, i32, i32, i32)> = ue_positions
            .iter()
            .map(|&(ue_x, ue_y, ue_z)| {
                let rust_x = ue_x / scale;
                let rust_y = ue_z / scale;
                let rust_z = -ue_y / scale;
                let base_x = rust_x.round() as i32 - bts / 2;
                let base_y_float = rust_y;
                let base_y = rust_y.floor() as i32;
                let base_z = rust_z.round() as i32 - bts / 2;
                let host_material =
                    voxel_gen::density::host_rock_for_depth(rust_y as f64, &cfg.ore.host_rock)
                        as u8;
                (base_x, base_y, base_z, base_y_float, host_material, bts, bts, clr)
            })
            .collect();
        drop(cfg);

        match self.mine_tx.send_timeout(
            WorkerRequest::BuildingFlattenBatch { buildings },
            std::time::Duration::from_millis(500),
        ) {
            Ok(()) => 1,
            Err(e) => {
                eprintln!("[voxel] request_building_flatten_batch: send failed: {}", e);
                0
            }
        }
    }
}
