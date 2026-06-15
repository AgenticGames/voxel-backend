

use crate::types::*;

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
            footprint_voxels: bts,
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
        let buildings: Vec<(i32, i32, i32, f32, u8, i32, i32)> = ue_positions
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
                (base_x, base_y, base_z, base_y_float, host_material, bts, clr)
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
