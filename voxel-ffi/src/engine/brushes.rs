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
    /// Creative-mode sphere brush (paint/carve/fill).
    /// Returns 1 on success, 0 if queue full.
    pub fn request_brush_sphere(&self, request: FfiBrushSphereRequest) -> u32 {
        match self.mine_tx.try_send(WorkerRequest::BrushSphere { request }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Creative "OrePaint" brush — drops wall-exposed ore deposits + optional
    /// deep channels inside a sphere. Returns 1 on success, 0 if queue full.
    pub fn request_brush_ore_paint(&self, request: crate::types::FfiBrushOrePaintRequest) -> u32 {
        let scale = self.world_scale;
        let center_rust = crate::convert::from_ue_world_pos(
            request.world_x, request.world_y, request.world_z, scale,
        );
        let radius = request.radius / scale;
        let w = request.weights;
        let weights = crate::brushes::OreWeights {
            iron: w[0], copper: w[1], malachite: w[2], tin: w[3], gold: w[4],
            diamond: w[5], kimberlite: w[6], sulfide: w[7], quartz: w[8],
            pyrite: w[9], amethyst: w[10], crystal: w[11], coal: w[12],
        };
        match self.mine_tx.try_send(WorkerRequest::BrushOrePaint {
            center_rust,
            radius,
            cluster_size: request.cluster_size,
            min_spacing: request.min_spacing,
            channel_prob: request.channel_prob,
            channel_length: request.channel_length,
            channel_radius: request.channel_radius,
            density: request.density,
            seed: request.seed,
            weights,
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Creative "PaintStress" brush — additively paints into the per-voxel
    /// painted-stress overlay inside a sphere.
    /// Returns 1 on success, 0 if queue full.
    pub fn request_brush_paint_stress(&self, request: crate::types::FfiBrushPaintStressRequest) -> u32 {
        let scale = self.world_scale;
        let center_rust = crate::convert::from_ue_world_pos(
            request.world_x, request.world_y, request.world_z, scale,
        );
        let radius = request.radius / scale;
        match self.mine_tx.try_send(WorkerRequest::BrushPaintStress {
            center_rust,
            radius,
            amount: request.amount,
            cap: request.cap,
            op: request.op,
            falloff: request.falloff,
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Wipe every loaded chunk's painted-stress overlay back to empty.
    /// Returns 1 on success, 0 if queue full.
    pub fn request_brush_clear_all_painted_stress(&self) -> u32 {
        match self.mine_tx.try_send(WorkerRequest::BrushClearAllPaintedStress) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Creative-mode tunnel-along-polyline brush.
    /// `points` are UE world coords; converted to Rust space here.
    /// `material == 255` means carve; otherwise fill with that material.
    /// Returns 1 on success, 0 if queue full or invalid input.
    pub fn request_brush_tunnel(
        &self,
        ue_points: &[(f32, f32, f32)],
        ue_radius: f32,
        material: u8,
    ) -> u32 {
        if ue_points.len() < 2 || ue_radius <= 0.0 {
            return 0;
        }
        let scale = self.world_scale;
        let points: Vec<glam::Vec3> = ue_points
            .iter()
            .map(|&(x, y, z)| crate::convert::from_ue_world_pos(x, y, z, scale))
            .collect();
        let radius = ue_radius / scale;
        let mat = if material == 255 { None } else { Some(material) };
        match self.mine_tx.try_send(WorkerRequest::BrushTunnel {
            points,
            radius,
            material: mat,
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Sphere fluid brush — fill / clear / pool-dig / carve+fill.
    pub fn request_brush_fluid_sphere(&self, request: FfiBrushFluidSphereRequest) -> u32 {
        let scale = self.world_scale;
        let center_rust = crate::convert::from_ue_world_pos(
            request.world_x, request.world_y, request.world_z, scale,
        );
        let radius = request.radius / scale;
        match self.mine_tx.try_send(WorkerRequest::BrushFluidSphere {
            center_rust,
            radius,
            fluid_type: request.fluid_type,
            is_source: request.is_source != 0,
            op: request.op,
            max_flow_dist: request.max_flow_dist,
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Box fluid brush — fill / clear / carve+fill within an axis-aligned box.
    pub fn request_brush_fluid_box(&self, request: FfiBrushFluidBoxRequest) -> u32 {
        let scale = self.world_scale;
        let center_rust = crate::convert::from_ue_world_pos(
            request.world_x, request.world_y, request.world_z, scale,
        );
        let half_ext_rust = glam::Vec3::new(
            request.half_x / scale,
            request.half_z / scale,
            request.half_y / scale,
        );
        match self.mine_tx.try_send(WorkerRequest::BrushFluidBox {
            center_rust,
            half_ext_rust,
            fluid_type: request.fluid_type,
            is_source: request.is_source != 0,
            op: request.op,
            max_flow_dist: request.max_flow_dist,
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// River (capsule chain) fluid brush. Points are UE world coords.
    pub fn request_brush_fluid_river(
        &self,
        ue_points: &[(f32, f32, f32)],
        ue_radius: f32,
        fluid_type: u8,
        is_source: bool,
        op: u8,
        max_flow_dist: u8,
    ) -> u32 {
        if ue_points.len() < 2 || ue_radius <= 0.0 {
            return 0;
        }
        let scale = self.world_scale;
        let points: Vec<glam::Vec3> = ue_points
            .iter()
            .map(|&(x, y, z)| crate::convert::from_ue_world_pos(x, y, z, scale))
            .collect();
        let radius = ue_radius / scale;
        match self.mine_tx.try_send(WorkerRequest::BrushFluidRiver {
            points,
            radius,
            fluid_type,
            is_source,
            op,
            max_flow_dist,
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Pop the most recent brush stroke and restore the captured chunk state.
    /// Returns 1 on success, 0 if queue full. (Returns 1 even if the undo stack
    /// is empty — the worker will simply no-op; query depth via `undo_depth`.)
    pub fn request_brush_undo(&self) -> u32 {
        match self.mine_tx.try_send(WorkerRequest::BrushUndo) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Current undo stack depth (number of strokes available to undo).
    pub fn undo_depth(&self) -> u32 {
        self.store.read().unwrap().undo_stack.len() as u32
    }

    /// Creative-mode axis-aligned-or-yawed box brush (paint/carve/fill).
    /// `yaw_deg` rotates around UE vertical (Z); since UE Z = Rust Y, the
    /// rotation is around Rust Y axis. UE positive yaw maps to Rust positive
    /// rotation (left-hand vs right-hand chirality cancels because both axes
    /// are vertical).
    pub fn request_brush_box(&self, request: FfiBrushBoxRequest) -> u32 {
        let scale = self.world_scale;
        let center_rust = crate::convert::from_ue_world_pos(
            request.world_x, request.world_y, request.world_z, scale,
        );
        // UE half-extents (X, Y, Z) → Rust (X, Z, Y) since UE Z is height.
        let half_ext_rust = glam::Vec3::new(
            request.half_x / scale,
            request.half_z / scale, // UE Z (height) → Rust Y
            request.half_y / scale, // UE Y → Rust Z
        );
        let yaw_rad = request.yaw_deg.to_radians();
        match self.mine_tx.try_send(WorkerRequest::BrushBox {
            center_rust,
            half_ext_rust,
            yaw_rad,
            op: request.op,
            material: request.material,
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Creative-mode Y-axis-aligned cylinder brush (paint/carve/fill).
    /// `height` is in UE units along the UE Z (which becomes Rust Y).
    pub fn request_brush_cylinder(&self, request: FfiBrushCylinderRequest) -> u32 {
        let scale = self.world_scale;
        let center_rust = crate::convert::from_ue_world_pos(
            request.world_x, request.world_y, request.world_z, scale,
        );
        let radius = request.radius / scale;
        let height = request.height / scale;
        match self.mine_tx.try_send(WorkerRequest::BrushCylinder {
            center_rust,
            radius,
            height,
            op: request.op,
            material: request.material,
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Creative-mode smooth brush.
    pub fn request_brush_smooth(&self, request: FfiBrushSmoothRequest) -> u32 {
        let scale = self.world_scale;
        let center_rust = crate::convert::from_ue_world_pos(
            request.world_x, request.world_y, request.world_z, scale,
        );
        let radius = request.radius / scale;
        match self.mine_tx.try_send(WorkerRequest::BrushSmooth {
            center_rust,
            radius,
            iterations: request.iterations,
            strength: request.strength,
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Creative-mode noise brush.
    /// `frequency` from UE is in 1/UE-units; convert by *scale (so a UE frequency of 0.01
    /// maps to a Rust-space frequency of 0.4 at scale=40 — small numbers in UE → bigger
    /// noise scale in Rust).
    pub fn request_brush_noise(&self, request: FfiBrushNoiseRequest) -> u32 {
        let scale = self.world_scale;
        let center_rust = crate::convert::from_ue_world_pos(
            request.world_x, request.world_y, request.world_z, scale,
        );
        let radius = request.radius / scale;
        let frequency_rust = request.frequency * scale;
        match self.mine_tx.try_send(WorkerRequest::BrushNoise {
            center_rust,
            radius,
            frequency: frequency_rust,
            strength: request.strength,
            seed: request.seed,
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Creative-mode formation placer (one stalactite/stalagmite/column/etc.).
    /// Returns 1 on success, 0 if queue full.
    pub fn request_brush_formation(&self, request: FfiBrushFormationRequest) -> u32 {
        let scale = self.world_scale;
        let center_rust = crate::convert::from_ue_world_pos(
            request.world_x, request.world_y, request.world_z, scale,
        );
        let height = request.height / scale;
        let radius = request.radius / scale;
        match self.mine_tx.try_send(WorkerRequest::BrushFormation {
            center_rust,
            formation_type: request.formation_type,
            material: request.material,
            height,
            radius,
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Creative-mode mushroom-placer brush. Places one mushroom instance at
    /// the cursor anchor; does not modify density.
    pub fn request_brush_place_mushroom(&self, request: FfiBrushPlaceMushroomRequest) -> u32 {
        let scale = self.world_scale;
        let center_rust = crate::convert::from_ue_world_pos(
            request.world_x, request.world_y, request.world_z, scale,
        );
        let search_radius = (request.search_radius / scale).max(0.5);
        match self.mine_tx.try_send(WorkerRequest::BrushPlaceMushroom {
            center_rust,
            kind: request.kind,
            search_radius,
            scale: request.scale,
            yaw: request.yaw_radians,
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Sphere-area mushroom brush — paints OR erases depending on `request.op`.
    /// `request.radius` is UE units; `request.density` is per-candidate accept
    /// probability; `request.clustering` shapes the local distribution.
    pub fn request_brush_place_mushroom_sphere(
        &self,
        request: crate::types::FfiBrushPlaceMushroomSphereRequest,
    ) -> u32 {
        let scale = self.world_scale;
        let center_rust = crate::convert::from_ue_world_pos(
            request.world_x, request.world_y, request.world_z, scale,
        );
        let radius_voxels = (request.radius / scale).max(0.5);
        let msg = if request.op == 1 {
            WorkerRequest::BrushEraseMushroomSphere {
                center_rust,
                radius: radius_voxels,
                kind_filter: request.kind,
            }
        } else {
            WorkerRequest::BrushPlaceMushroomSphere {
                center_rust,
                radius: radius_voxels,
                density: request.density.clamp(0.0, 1.0),
                clustering: request.clustering.clamp(0.0, 1.0),
                kind: request.kind,
                seed: request.seed,
            }
        };
        match self.mine_tx.try_send(msg) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Diagnostic: force a single chunk to re-sync its boundaries with all
    /// face-adjacent neighbors and remesh anything modified. Use when seam
    /// mismatches accumulate (visible as flat walls or mesh holes) and
    /// you want to repair them without a full quit+reload.
    /// Returns 1 if queued, 0 if queue full.
    pub fn request_force_chunk_resync(&self, cx: i32, cy: i32, cz: i32) -> u32 {
        match self.mine_tx.try_send(WorkerRequest::ForceChunkResync {
            chunk: (cx, cy, cz),
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Creative-mode cavern stamp brush — chunk-snapped cave generator.
    /// `chunk_x/y/z` is the lo-corner chunk in Rust chunk coords; the brush
    /// affects an `extent_x × extent_y × extent_z` chunk region.
    /// Returns 1 on success, 0 if queue full.
    pub fn request_brush_cavern_stamp(&self, request: FfiBrushCavernStampRequest) -> u32 {
        match self.mine_tx.try_send(WorkerRequest::BrushCavernStamp {
            chunk_origin: (request.chunk_x, request.chunk_y, request.chunk_z),
            extent: (
                request.extent_x.max(1),
                request.extent_y.max(1),
                request.extent_z.max(1),
            ),
            decorate: request.decorate != 0,
            fluids: request.fluids != 0,
            seed: request.seed,
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Creative-mode formation stamp brush — runs the worldgen formation
    /// pipeline within a sphere with a randomized seed. Returns 1 on success,
    /// 0 if queue full.
    pub fn request_brush_formation_stamp(&self, request: FfiBrushFormationStampRequest) -> u32 {
        let scale = self.world_scale;
        let center_rust = crate::convert::from_ue_world_pos(
            request.world_x, request.world_y, request.world_z, scale,
        );
        let radius = request.radius / scale;
        match self.mine_tx.try_send(WorkerRequest::BrushFormationStamp {
            center_rust,
            radius,
            seed: request.seed,
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Query nearby existing terrace to snap Z for extending terraces on the same plane.
    /// Returns Some((snapped_ue_x, snapped_ue_y, snapped_ue_z)) if found within range.
    pub fn query_nearby_terrace(
        &self,
        ue_x: f32,
        ue_y: f32,
        ue_z: f32,
        scale: f32,
    ) -> Option<(f32, f32, f32)> {
        let rust_x = ue_x / scale;
        let rust_y = ue_z / scale;
        let rust_z = -ue_y / scale;

        let ts = terrace_size_for_scale(scale);
        let base_x = (rust_x as i32).div_euclid(ts) * ts;
        let base_z = (rust_z as i32).div_euclid(ts) * ts;
        let approx_y = (rust_y as i32).div_euclid(ts) * ts;

        let store = self.store.try_read().ok()?;
        let search_radius = 10;
        let max_y_diff = 6; // 6 voxels = 240 UU at scale 40
        crate::terrain_ops::query_nearby_terrace_y(&*store, base_x, base_z, approx_y, search_radius, max_y_diff)
            .map(|found_y| {
                let snap_ue_x = base_x as f32 * scale;
                let snap_ue_y = -(base_z as f32) * scale;
                let snap_ue_z = found_y as f32 * scale;
                (snap_ue_x, snap_ue_y, snap_ue_z)
            })
    }

    /// Query the host rock material at a UE world position based on depth.
    /// Returns material id as u8.
    pub fn query_host_rock_at(&self, _ue_x: f32, _ue_y: f32, ue_z: f32, scale: f32) -> u8 {
        let rust_y = ue_z / scale;
        let cfg = self.config.read().unwrap();
        voxel_gen::density::host_rock_for_depth(rust_y as f64, &cfg.ore.host_rock) as u8
    }
}
