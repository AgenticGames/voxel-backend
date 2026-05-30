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

use super::{VoxelEngine, COLLAPSE_IMMINENT_STRESS};

impl VoxelEngine {
    /// Find the best spring location near the player.
    /// Takes UE world coords, returns UE world coords via Option.
    pub fn find_spring(&self, ue_x: f32, ue_y: f32, ue_z: f32, world_scale: f32) -> Option<(f32, f32, f32)> {
        use crate::convert::from_ue_world_pos;

        let cfg = self.config.read().ok()?;
        let chunk_size = cfg.chunk_size;
        let eb = cfg.effective_bounds();
        drop(cfg);
        let rust_pos = from_ue_world_pos(ue_x, ue_y, ue_z, world_scale);

        let store = self.store.try_read().ok()?;
        let best = store.find_spring_location(rust_pos, chunk_size, eb)?;

        // Convert Rust pos back to UE: (x * scale, -z * scale, y * scale)
        Some((
            best.x * world_scale,
            -best.z * world_scale,
            best.y * world_scale,
        ))
    }

    /// Find surface-facing ore voxels near the player.
    /// All position inputs/outputs are UE world coords; `ue_radius` is in UE units.
    /// Returns up to `max_results` (UE-space) voxel centers sorted by distance,
    /// each with its raw `Material as u8`.
    /// `material_filter` of `0xFF` means "any ore" (`Material::is_ore()`).
    pub fn find_ore_voxels(
        &self,
        ue_x: f32,
        ue_y: f32,
        ue_z: f32,
        ue_radius: f32,
        material_filter: u8,
        max_results: usize,
        world_scale: f32,
    ) -> Vec<(f32, f32, f32, u8)> {
        use crate::convert::from_ue_world_pos;

        let cfg = match self.config.read() {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };
        let chunk_size = cfg.chunk_size;
        let eb = cfg.effective_bounds();
        drop(cfg);

        let rust_pos = from_ue_world_pos(ue_x, ue_y, ue_z, world_scale);
        // UE distance / world_scale → voxel-unit distance.
        let voxel_radius = ue_radius / world_scale;

        let store = match self.store.try_read() {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let hits = store.find_ore_voxels(
            rust_pos,
            voxel_radius,
            material_filter,
            max_results,
            chunk_size,
            eb,
        );

        hits.into_iter()
            .map(|(p, m)| (p.x * world_scale, -p.z * world_scale, p.y * world_scale, m))
            .collect()
    }

    /// Find a wall-adjacent air cell near a target, excluding a radius around an exclusion point.
    /// Takes UE world coords, returns UE world coords via Option.
    pub fn find_wall_near(
        &self,
        ue_x: f32, ue_y: f32, ue_z: f32,
        exclude_ue_x: f32, exclude_ue_y: f32, exclude_ue_z: f32,
        exclude_radius: f32,
        world_scale: f32,
    ) -> Option<(f32, f32, f32)> {
        use crate::convert::from_ue_world_pos;

        let cfg = self.config.read().ok()?;
        let chunk_size = cfg.chunk_size;
        let eb = cfg.effective_bounds();
        drop(cfg);
        let target = from_ue_world_pos(ue_x, ue_y, ue_z, world_scale);
        let exclude = from_ue_world_pos(exclude_ue_x, exclude_ue_y, exclude_ue_z, world_scale);
        // Convert UE-unit radius to voxel-unit radius
        let voxel_radius = exclude_radius / world_scale;

        let store = self.store.try_read().ok()?;
        let best = store.find_wall_location_near(target, exclude, voxel_radius, chunk_size, eb)?;

        // Convert Rust pos back to UE: (x * scale, -z * scale, y * scale)
        Some((
            best.x * world_scale,
            -best.z * world_scale,
            best.y * world_scale,
        ))
    }

    /// Find a validated spawn location for the player capsule.
    /// Takes UE world coords, returns UE world coords via Option.
    /// `height` and `radius` are in voxel units.
    pub fn find_spawn_location(
        &self,
        ue_x: f32, ue_y: f32, ue_z: f32,
        exclude_ue_x: f32, exclude_ue_y: f32, exclude_ue_z: f32,
        exclude_radius: f32,
        world_scale: f32,
        height: i32,
        radius: i32,
    ) -> Option<(f32, f32, f32)> {
        use crate::convert::from_ue_world_pos;

        let cfg = self.config.read().ok()?;
        let chunk_size = cfg.chunk_size;
        let eb = cfg.effective_bounds();
        drop(cfg);
        let target = from_ue_world_pos(ue_x, ue_y, ue_z, world_scale);
        let exclude = from_ue_world_pos(exclude_ue_x, exclude_ue_y, exclude_ue_z, world_scale);
        let voxel_radius = exclude_radius / world_scale;

        let store = self.store.try_read().ok()?;
        let best = store.find_spawn_location(target, exclude, voxel_radius, chunk_size, eb, height, radius)?;

        Some((
            best.x * world_scale,
            -best.z * world_scale,
            best.y * world_scale,
        ))
    }

    /// Find a validated spawn location for the chrysalis.
    /// Takes UE world coords, returns UE world coords via Option.
    pub fn find_chrysalis_location(
        &self,
        ue_x: f32, ue_y: f32, ue_z: f32,
        exclude_ue_x: f32, exclude_ue_y: f32, exclude_ue_z: f32,
        exclude_radius: f32,
        world_scale: f32,
        height: i32,
        radius: i32,
    ) -> Option<(f32, f32, f32)> {
        use crate::convert::from_ue_world_pos;

        let cfg = self.config.read().ok()?;
        let chunk_size = cfg.chunk_size;
        let eb = cfg.effective_bounds();
        drop(cfg);
        let target = from_ue_world_pos(ue_x, ue_y, ue_z, world_scale);
        let exclude = from_ue_world_pos(exclude_ue_x, exclude_ue_y, exclude_ue_z, world_scale);
        let voxel_radius = exclude_radius / world_scale;

        let store = self.store.try_read().ok()?;
        let best = store.find_chrysalis_location(target, exclude, voxel_radius, chunk_size, eb, height, radius)?;

        Some((
            best.x * world_scale,
            -best.z * world_scale,
            best.y * world_scale,
        ))
    }

    /// Find spring, chrysalis, and spawn locations all in the same cavern.
    /// Takes UE world coords for player position, returns UE world coords for all three.
    /// Returns None if any of the three couldn't be found.
    pub fn find_cavern_locations(
        &self,
        ue_x: f32, ue_y: f32, ue_z: f32,
        world_scale: f32,
    ) -> Option<((f32, f32, f32), (f32, f32, f32), (f32, f32, f32))> {
        use crate::convert::from_ue_world_pos;

        let cfg = self.config.read().ok()?;
        let chunk_size = cfg.chunk_size;
        let eb = cfg.effective_bounds();
        drop(cfg);
        let player_pos = from_ue_world_pos(ue_x, ue_y, ue_z, world_scale);

        let store = self.store.try_read().ok()?;
        let locations = store.find_cavern_locations(player_pos, chunk_size, eb)?;

        // Convert all three positions from Rust to UE coords: (x * scale, -z * scale, y * scale)
        let spring_ue = (
            locations.spring.x * world_scale,
            -locations.spring.z * world_scale,
            locations.spring.y * world_scale,
        );
        let chrysalis_ue = (
            locations.chrysalis.x * world_scale,
            -locations.chrysalis.z * world_scale,
            locations.chrysalis.y * world_scale,
        );
        let spawn_ue = (
            locations.spawn.x * world_scale,
            -locations.spawn.z * world_scale,
            locations.spawn.y * world_scale,
        );

        Some((spring_ue, chrysalis_ue, spawn_ue))
    }

    /// Queue a priority generate request for a chunk. Sent through the mine channel
    /// for immediate processing. Coords are UE space. Returns 1 on success, 0 if full.
    pub fn request_priority_generate(&self, cx: i32, cy: i32, cz: i32) -> u32 {
        let key = ue_chunk_to_rust(cx, cy, cz);
        let counter_ref = self
            .generation_counters
            .entry(key)
            .or_insert_with(|| AtomicU64::new(0));
        let generation = counter_ref.load(Ordering::Relaxed) + 1;

        match self.mine_tx.try_send(WorkerRequest::PriorityGenerate {
            chunk: key,
            generation,
        }) {
            Ok(()) => {
                counter_ref.store(generation, Ordering::Relaxed);
                1
            }
            Err(_) => 0,
        }
    }

    /// Query the stress field for a chunk. Returns a cloned StressField if loaded.
    pub fn query_stress(&self, chunk: (i32, i32, i32)) -> Option<StressField> {
        // Blocking read on purpose: the PaintStress overlay refresh path on
        // the UE side polls this immediately after dispatching a paint, and
        // a try_read race against the worker's write lock would silently
        // return valid=0 — making the V-overlay stop auto-updating after
        // any collapse cascade (the worker holds write for hundreds of ms
        // during cascade + post-cascade mesh updates). Blocking briefly
        // here is the right tradeoff vs. silently dropping refreshes.
        let store = self.store.read().ok()?;
        store.stress_fields.get(&chunk).cloned()
    }

    /// Synchronously recalculate stress on nearby chunks for V-key preview.
    /// Returns the UE chunk coords of chunks that were computed (for overlay display).
    pub fn recalc_stress_preview(&self, center: (i32, i32, i32)) -> Vec<(i32, i32, i32)> {
        use voxel_core::stress::recalc_stress_region_v2;
        use crate::convert::rust_chunk_to_ue;

        let stress_cfg = self.stress_config.read().unwrap().clone();
        let cfg = self.config.read().unwrap().clone();
        let chunk_size = cfg.chunk_size;

        // 3x3x3 cube around center in Rust coords
        let mut chunk_keys = Vec::with_capacity(27);
        for dy in -1..=1i32 {
            for dz in -1..=1i32 {
                for dx in -1..=1i32 {
                    chunk_keys.push((center.0 + dx, center.1 + dy, center.2 + dz));
                }
            }
        }

        let mut store = self.store.write().unwrap();
        let loaded_keys: Vec<(i32, i32, i32)> = chunk_keys
            .into_iter()
            .filter(|k| store.density_fields.contains_key(k))
            .collect();

        if !loaded_keys.is_empty() {
            let (density, stress, support) = store.sleep_fields_mut();
            recalc_stress_region_v2(density, stress, support, &stress_cfg, &loaded_keys, chunk_size);
        }

        // Convert computed Rust chunk keys to UE chunk keys for the overlay
        loaded_keys.iter().map(|&(rx, ry, rz)| rust_chunk_to_ue(rx, ry, rz)).collect()
    }

    /// Query stress at a single world voxel position.
    pub fn query_stress_at(&self, wx: i32, wy: i32, wz: i32, chunk_size: usize) -> f32 {
        let cs = chunk_size as i32;
        let cx = wx.div_euclid(cs);
        let cy = wy.div_euclid(cs);
        let cz = wz.div_euclid(cs);
        let lx = wx.rem_euclid(cs) as usize;
        let ly = wy.rem_euclid(cs) as usize;
        let lz = wz.rem_euclid(cs) as usize;

        let store = match self.store.try_read() {
            Ok(s) => s,
            Err(_) => return 0.0,
        };
        store.stress_fields
            .get(&(cx, cy, cz))
            .map(|sf| sf.get(lx, ly, lz))
            .unwrap_or(0.0)
    }

    /// Probe the density field at a UE world point. Used by spider-nest /
    /// wasp-hive placement validators — see [`crate::surface_probe`].
    ///
    /// Returns the probe result with normal + clearance translated back to
    /// UE world space. `normal_hint_ue` is the caller's surface-normal hint
    /// (used only as a fallback when the local gradient is flat); pass
    /// `(0, 0, 1)` (UE up) if no hint is available.
    pub fn query_surface(
        &self,
        ue_x: f32, ue_y: f32, ue_z: f32,
        hint_ue_x: f32, hint_ue_y: f32, hint_ue_z: f32,
    ) -> Option<crate::surface_probe::ProbeResult> {
        use crate::convert::{from_ue_world_pos, from_ue_normal};
        use crate::surface_probe::probe_surface;

        let chunk_size = self.chunk_size();
        let world_scale = self.get_world_scale();

        let rust_pos = from_ue_world_pos(ue_x, ue_y, ue_z, world_scale);
        let normal_hint_rust = from_ue_normal(hint_ue_x, hint_ue_y, hint_ue_z);

        // Bounded spin-retry (2026-05-30). A bare `try_read().ok()?` fails the
        // INSTANT a worker holds the write lock — and the sleep-montage camera
        // planner fires QuerySurface in tight bursts (the rock-vs-air ray clamp)
        // that race the generation workers' density-insert writes. Losing that
        // race returned None → the caller read "unloaded"(=solid) and the clamp
        // went blind even though the density was right there. Spin briefly so a
        // short writer doesn't blind us, but cap the wait (~1ms) so a pathological
        // long hold can never stall the game thread — on timeout fall back to the
        // original "treat as unavailable" behavior.
        let store = {
            let mut guard = None;
            for _ in 0..6 {
                if let Ok(s) = self.store.try_read() {
                    guard = Some(s);
                    break;
                }
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
            guard?
        };
        Some(probe_surface(&store, rust_pos, chunk_size, normal_hint_rust))
    }

    /// Cheap TRI-STATE solidity for the camera planner: 0=air, 1=loaded-solid,
    /// 2=unloaded. ~1000× cheaper than `query_surface`. Lock-busy → 2 (unknown ≈
    /// unloaded): the clamp then treats it as rock (safe), the exposure check as
    /// void — both conservative, and it's rare (the spin-retry usually wins).
    pub fn is_solid_at_ue(&self, ue_x: f32, ue_y: f32, ue_z: f32) -> u32 {
        let chunk_size = self.chunk_size();
        let world_scale = self.get_world_scale();
        let store = {
            let mut guard = None;
            for _ in 0..6 {
                if let Ok(s) = self.store.try_read() {
                    guard = Some(s);
                    break;
                }
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
            match guard {
                Some(g) => g,
                None => return 2,
            }
        };
        crate::surface_probe::solidity_at_ue(&store, chunk_size, world_scale, ue_x, ue_y, ue_z) as u32
    }

    /// Enumerate cells in a single chunk whose effective stress is high enough
    /// to be in the "imminent collapse" band — past the bare 1.0 threshold and
    /// well into the saturated-white range of the stress overlay. Returns one
    /// entry per surface-exposed over-stress cell with its UE-world center,
    /// surface normal, and stress value — ready for UE to drop a crack decal
    /// at and fire warning dust.
    ///
    /// **Why higher than the collapse pass's 1.0**: the V-overlay maps stress
    /// 1.0 → pure red, 1.5 → pure white, with a red→white lerp in between
    /// (see `StressToColor` in VoxelChunkActor.cpp). The collapse pass starts
    /// considering cells at 1.0, but cells in the 1.0-1.3 range are often
    /// edge-of-region — they sit forever without dropping because their slab
    /// isn't coherent enough or the region's median landing offset is <= 0.
    /// Players reasonably read "red but not white" as "stressed but stable"
    /// and find dust there confusing. Filtering at [`COLLAPSE_IMMINENT_STRESS`]
    /// (1.5) aligns warning FX with the visual "pure white" core where slabs
    /// actually form.
    ///
    /// Interior (fully-enclosed) cells are skipped: they have no visible
    /// surface to decorate.
    ///
    /// `chunk_rust` is the Rust chunk coord (caller converts from UE).
    ///
    /// Returns `(cells, valid)`. `valid=false` means the store lock was
    /// contended even after retry — the caller MUST NOT treat the empty Vec
    /// as "no cells here" since it might just be a transient race with the
    /// brush worker. UE uses this to preserve existing decals instead of
    /// wiping them on every paint click that happens to race with a worker.
    pub fn enumerate_overstressed_in_chunk(
        &self,
        chunk_rust: (i32, i32, i32),
    ) -> (Vec<crate::types::FfiOverstressedCell>, bool) {
        use crate::surface_probe::probe_surface;
        use glam::Vec3;
        use voxel_core::stress::{unpack_surface, SURFACE_INTERIOR};

        let chunk_size = self.chunk_size();
        let world_scale = self.get_world_scale();

        // Try-read with short retry. The brush worker holds the write lock
        // for ~10ms during a paint update. Without retry, UE refreshes
        // immediately after the FFI call land before the worker releases,
        // try_read fails, we return empty, and UE wipes all the decals
        // ("1 in 4 clicks the decals disappear" bug). 5 retries x 2ms
        // = up to 10ms blocking on the game thread, acceptable for an
        // event-driven refresh (not per-frame).
        let store = {
            let mut got = None;
            for _ in 0..5 {
                if let Ok(s) = self.store.try_read() { got = Some(s); break; }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            match got {
                Some(s) => s,
                None => return (Vec::new(), false),  // contended -> caller preserves
            }
        };
        let sf = match store.stress_fields.get(&chunk_rust) {
            Some(sf) => sf,
            None => return (Vec::new(), true),  // store OK, just no field for this chunk
        };

        let cs = chunk_size as i32;
        let (cx, cy, cz) = chunk_rust;
        let mut out = Vec::new();

        for lz in 0..chunk_size {
            for ly in 0..chunk_size {
                for lx in 0..chunk_size {
                    let eff = sf.effective(lx, ly, lz);
                    if eff < COLLAPSE_IMMINENT_STRESS {
                        continue;
                    }
                    let surface_kind = unpack_surface(sf.get_class(lx, ly, lz));
                    if surface_kind == SURFACE_INTERIOR {
                        continue;
                    }

                    let rust_x = cx * cs + lx as i32;
                    let rust_y = cy * cs + ly as i32;
                    let rust_z = cz * cs + lz as i32;
                    let rust_center = Vec3::new(
                        rust_x as f32 + 0.5,
                        rust_y as f32 + 0.5,
                        rust_z as f32 + 0.5,
                    );

                    let probe = probe_surface(&store, rust_center, chunk_size, Vec3::Y);

                    let ue_x = rust_center.x * world_scale;
                    let ue_y = -rust_center.z * world_scale;
                    let ue_z = rust_center.y * world_scale;

                    let nx_ue = probe.normal.x;
                    let ny_ue = -probe.normal.z;
                    let nz_ue = probe.normal.y;

                    out.push(crate::types::FfiOverstressedCell {
                        world_x: ue_x,
                        world_y: ue_y,
                        world_z: ue_z,
                        normal_x: nx_ue,
                        normal_y: ny_ue,
                        normal_z: nz_ue,
                        stress: eff,
                        surface_kind,
                        _padding: [0; 3],
                    });
                }
            }
        }
        (out, true)
    }

    /// Enumerate over-stress cells inside a UE world-space sphere. Used by
    /// the mining post-recalc handler to fire a "you're undermining a fragile
    /// area" dust burst at every primed cell within the mining impact zone.
    ///
    /// `center_ue` + `radius_ue` are in UE units (centimeters). The result
    /// includes every surface-exposed over-stress cell whose center is within
    /// `radius_ue` of the impact point.
    ///
    /// Returns `(cells, valid)` — `valid=false` signals store-lock contention;
    /// see `enumerate_overstressed_in_chunk` for the same semantics.
    pub fn enumerate_overstressed_in_sphere(
        &self,
        center_ue_x: f32,
        center_ue_y: f32,
        center_ue_z: f32,
        radius_ue: f32,
    ) -> (Vec<crate::types::FfiOverstressedCell>, bool) {
        use crate::convert::from_ue_world_pos;
        use crate::surface_probe::probe_surface;
        use glam::Vec3;
        use voxel_core::stress::{unpack_surface, SURFACE_INTERIOR};

        let chunk_size = self.chunk_size();
        let world_scale = self.get_world_scale();

        let rust_center = from_ue_world_pos(center_ue_x, center_ue_y, center_ue_z, world_scale);
        let radius_voxels = radius_ue / world_scale;
        let r2 = radius_voxels * radius_voxels;

        let cs = chunk_size as i32;
        let min_rx = (rust_center.x - radius_voxels).floor() as i32;
        let max_rx = (rust_center.x + radius_voxels).ceil() as i32;
        let min_ry = (rust_center.y - radius_voxels).floor() as i32;
        let max_ry = (rust_center.y + radius_voxels).ceil() as i32;
        let min_rz = (rust_center.z - radius_voxels).floor() as i32;
        let max_rz = (rust_center.z + radius_voxels).ceil() as i32;

        let cx_min = min_rx.div_euclid(cs);
        let cx_max = max_rx.div_euclid(cs);
        let cy_min = min_ry.div_euclid(cs);
        let cy_max = max_ry.div_euclid(cs);
        let cz_min = min_rz.div_euclid(cs);
        let cz_max = max_rz.div_euclid(cs);

        // Same retry/contention semantics as the chunk variant — see comment
        // there. Critical for the mining-burst path that fires while the
        // worker is still updating stress fields.
        let store = {
            let mut got = None;
            for _ in 0..5 {
                if let Ok(s) = self.store.try_read() { got = Some(s); break; }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            match got {
                Some(s) => s,
                None => return (Vec::new(), false),
            }
        };

        let mut out = Vec::new();
        for cz in cz_min..=cz_max {
            for cy in cy_min..=cy_max {
                for cx in cx_min..=cx_max {
                    let sf = match store.stress_fields.get(&(cx, cy, cz)) {
                        Some(sf) => sf,
                        None => continue,
                    };
                    for lz in 0..chunk_size {
                        for ly in 0..chunk_size {
                            for lx in 0..chunk_size {
                                let eff = sf.effective(lx, ly, lz);
                                if eff < COLLAPSE_IMMINENT_STRESS {
                                    continue;
                                }
                                let surface_kind = unpack_surface(sf.get_class(lx, ly, lz));
                                if surface_kind == SURFACE_INTERIOR {
                                    continue;
                                }

                                let rust_x = cx * cs + lx as i32;
                                let rust_y = cy * cs + ly as i32;
                                let rust_z = cz * cs + lz as i32;
                                let rust_center_cell = Vec3::new(
                                    rust_x as f32 + 0.5,
                                    rust_y as f32 + 0.5,
                                    rust_z as f32 + 0.5,
                                );

                                let dx = rust_center_cell.x - rust_center.x;
                                let dy = rust_center_cell.y - rust_center.y;
                                let dz = rust_center_cell.z - rust_center.z;
                                if dx * dx + dy * dy + dz * dz > r2 {
                                    continue;
                                }

                                let probe = probe_surface(&store, rust_center_cell, chunk_size, Vec3::Y);

                                let ue_x = rust_center_cell.x * world_scale;
                                let ue_y = -rust_center_cell.z * world_scale;
                                let ue_z = rust_center_cell.y * world_scale;

                                let nx_ue = probe.normal.x;
                                let ny_ue = -probe.normal.z;
                                let nz_ue = probe.normal.y;

                                out.push(crate::types::FfiOverstressedCell {
                                    world_x: ue_x,
                                    world_y: ue_y,
                                    world_z: ue_z,
                                    normal_x: nx_ue,
                                    normal_y: ny_ue,
                                    normal_z: nz_ue,
                                    stress: eff,
                                    surface_kind,
                                    _padding: [0; 3],
                                });
                            }
                        }
                    }
                }
            }
        }
        (out, true)
    }

    /// Synchronous "is a path possible from A to B" check. Runs A* under a
    /// read lock on the live ChunkStore and returns the path status without
    /// the async request/poll dance. Intended for placement validation
    /// (wasp hive flight-path check) where the caller is OK paying a brief
    /// blocking cost — placement is off the hot path (~1 query per cluster
    /// spawn). For AI agents, use the async `voxel_path_request` flow.
    ///
    /// Returns the `PathStatus` raw u8 (matches `voxel_path::PathStatus`):
    /// 0 = Success, 1 = NoPath, 2 = MaxNodesReached, 3 = PartiallyUnloaded,
    /// 4 = InvalidEndpoint. None when the store lock couldn't be acquired.
    pub fn query_path_exists(
        &self,
        from_ue_x: f32, from_ue_y: f32, from_ue_z: f32,
        to_ue_x: f32, to_ue_y: f32, to_ue_z: f32,
        agent_radius_ue: f32,
        movement_mode: u8,
        max_nodes: u32,
    ) -> Option<u8> {
        let chunk_size = self.chunk_size();
        let world_scale = self.get_world_scale();

        let internal = crate::pathing::build_request_from_ue(
            0, // request_id unused for sync queries
            from_ue_x, from_ue_y, from_ue_z,
            to_ue_x, to_ue_y, to_ue_z,
            agent_radius_ue,
            movement_mode,
            if max_nodes == 0 { 10_000 } else { max_nodes },
            world_scale,
        );

        let store = self.store.try_read().ok()?;
        // Sync query — skip the cross-species occupancy layer. This path is
        // only used by `voxel_query_path_exists` which is a boolean reachability
        // check, not actual AI routing; including dynamic obstacles would make
        // the answer flicker as agents move and bear no relation to whether
        // the static geometry permits a route.
        let grid = crate::pathing::ChunkStoreGrid {
            store: &store,
            chunk_size,
            cell_factor: crate::pathing::PATH_CELL_FACTOR,
            occupied_cells: None,
            requester_cell: None,
        };
        let (path_req, _mode) = crate::pathing::to_path_request(&internal, crate::pathing::PATH_CELL_FACTOR);
        // smooth = false — we only care whether a path exists; we don't
        // need the post-processed waypoint list.
        let path_req = voxel_path::PathRequest { smooth: false, ..path_req };
        let outcome = voxel_path::compute_path(&grid, path_req);
        Some(outcome.status as u8)
    }
}
