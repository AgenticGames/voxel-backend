use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};

use crossbeam_channel::{bounded, Receiver, Sender};
use dashmap::DashMap;
use voxel_fluid::FluidConfig;
use voxel_fluid::FluidEvent;
use voxel_core::stress::StressField;
use voxel_gen::config::{
    BandedIronConfig, FormationConfig, GenerationConfig, GeodeConfig, HostRockConfig,
    KimberlitePipeConfig, MineConfig, NoiseConfig, OreConfig, OreVeinParams, PoolConfig,
    StressConfig, SulfideBlobConfig, WormConfig,
};

use crate::convert::ue_chunk_to_rust;
use crate::store::ChunkStore;
use crate::types::*;
use crate::worker::worker_loop;

/// Data returned when a sleep cycle completes.
pub struct SleepCompleteData {
    pub chunks_changed: u32,
    pub voxels_metamorphosed: u32,
    pub minerals_grown: u32,
    pub supports_degraded: u32,
    pub collapses_triggered: u32,
}

pub struct VoxelEngine {
    // Channels
    generate_tx: Sender<WorkerRequest>,
    mine_tx: Sender<WorkerRequest>,
    result_rx: Receiver<WorkerResult>,

    // Fluid
    fluid_event_tx: Sender<FluidEvent>,
    fluid_thread: Option<JoinHandle<()>>,

    // Shared state
    store: Arc<RwLock<ChunkStore>>,
    config: Arc<RwLock<GenerationConfig>>,
    stress_config: Arc<RwLock<StressConfig>>,
    generation_counters: Arc<DashMap<(i32, i32, i32), AtomicU64>>,
    shutdown: Arc<AtomicBool>,

    // Sleep
    sleep_complete: Arc<Mutex<Option<SleepCompleteData>>>,

    // Worker threads
    workers: Vec<JoinHandle<()>>,
}

impl VoxelEngine {
    pub fn new(ffi_config: &FfiEngineConfig) -> Self {
        let config = ffi_config_to_generation(ffi_config);
        let fluid_config = ffi_config_to_fluid(ffi_config);
        let world_scale = ffi_config.world_scale;
        let num_workers = if ffi_config.worker_threads == 0 {
            num_cpus()
        } else {
            ffi_config.worker_threads as usize
        };

        let (generate_tx, generate_rx) = bounded::<WorkerRequest>(256);
        let (mine_tx, mine_rx) = bounded::<WorkerRequest>(16);
        let (result_tx, result_rx) = bounded::<WorkerResult>(256);

        // Fluid event channel
        let (fluid_event_tx, fluid_event_rx) = bounded::<FluidEvent>(512);

        let store = Arc::new(RwLock::new(ChunkStore::new()));
        let config = Arc::new(RwLock::new(config));
        let stress_config = Arc::new(RwLock::new(StressConfig::default()));
        let generation_counters: Arc<DashMap<(i32, i32, i32), AtomicU64>> =
            Arc::new(DashMap::new());
        let shutdown = Arc::new(AtomicBool::new(false));

        // Spawn fluid simulation thread
        let fluid_result_tx = result_tx.clone();
        let fluid_shutdown = Arc::clone(&shutdown);
        let fluid_world_scale = world_scale;
        let fluid_thread = thread::spawn(move || {
            fluid_sim_loop_wrapper(
                fluid_shutdown,
                fluid_event_rx,
                fluid_result_tx,
                fluid_config,
                fluid_world_scale,
            );
        });

        let mut workers = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let shutdown = Arc::clone(&shutdown);
            let generate_rx = generate_rx.clone();
            let mine_rx = mine_rx.clone();
            let result_tx = result_tx.clone();
            let store = Arc::clone(&store);
            let config = Arc::clone(&config);
            let stress_cfg = Arc::clone(&stress_config);
            let gen_counters = Arc::clone(&generation_counters);
            let fluid_tx = fluid_event_tx.clone();

            let handle = thread::spawn(move || {
                worker_loop(
                    shutdown,
                    generate_rx,
                    mine_rx,
                    result_tx,
                    store,
                    config,
                    stress_cfg,
                    gen_counters,
                    world_scale,
                    fluid_tx,
                );
            });
            workers.push(handle);
        }

        VoxelEngine {
            generate_tx,
            mine_tx,
            result_rx,
            fluid_event_tx,
            fluid_thread: Some(fluid_thread),
            store,
            config,
            stress_config,
            generation_counters,
            shutdown,
            sleep_complete: Arc::new(Mutex::new(None)),
            workers,
        }
    }

    /// Queue a single chunk for generation. Coords are UE space, converted internally.
    /// Returns 1 on success, 0 if queue full.
    pub fn request_generate(&self, cx: i32, cy: i32, cz: i32) -> u32 {
        let key = ue_chunk_to_rust(cx, cy, cz);
        let generation = self
            .generation_counters
            .entry(key)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed)
            + 1;
        // Store the new generation value
        if let Some(counter) = self.generation_counters.get(&key) {
            counter.store(generation, Ordering::Relaxed);
        }

        match self.generate_tx.try_send(WorkerRequest::Generate {
            chunk: key,
            generation,
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Queue multiple chunks for generation. Returns count successfully queued.
    pub fn request_generate_batch(&self, chunks: &[(i32, i32, i32)]) -> u32 {
        let mut count = 0;
        for &(cx, cy, cz) in chunks {
            count += self.request_generate(cx, cy, cz);
        }
        count
    }

    /// Queue a mine request. Returns 1 on success, 0 if queue full.
    pub fn request_mine(&self, request: FfiMineRequest) -> u32 {
        match self.mine_tx.try_send(WorkerRequest::Mine { request }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Request unloading a chunk's cached data. Coords are UE space.
    pub fn request_unload(&self, cx: i32, cy: i32, cz: i32) -> u32 {
        let key = ue_chunk_to_rust(cx, cy, cz);
        match self
            .generate_tx
            .try_send(WorkerRequest::Unload { chunk: key })
        {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Cancel any pending generation for a chunk by bumping its generation counter.
    /// Workers will see the stale generation and skip. Coords are UE space.
    pub fn cancel_chunk(&self, cx: i32, cy: i32, cz: i32) {
        let key = ue_chunk_to_rust(cx, cy, cz);
        self.generation_counters
            .entry(key)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Non-blocking poll for a completed result. Returns None if nothing ready.
    /// SleepComplete results are intercepted and stored internally; they are
    /// retrieved via `poll_sleep_complete()` instead.
    pub fn poll_result(&self) -> Option<WorkerResult> {
        match self.result_rx.try_recv() {
            Ok(WorkerResult::SleepComplete {
                chunks_changed,
                voxels_metamorphosed,
                minerals_grown,
                supports_degraded,
                collapses_triggered,
            }) => {
                if let Ok(mut sc) = self.sleep_complete.lock() {
                    *sc = Some(SleepCompleteData {
                        chunks_changed,
                        voxels_metamorphosed,
                        minerals_grown,
                        supports_degraded,
                        collapses_triggered,
                    });
                }
                // Don't expose to the FfiResult pipeline; UE polls via voxel_poll_sleep_result
                None
            }
            Ok(other) => Some(other),
            Err(_) => None,
        }
    }

    /// Start a deep sleep cycle. Sends request through the mine channel
    /// (which has exclusive write-lock priority).
    /// Returns 1 on success, 0 if queue full.
    pub fn start_sleep(&self, player_chunk: (i32, i32, i32), sleep_count: u32) -> u32 {
        match self.mine_tx.try_send(WorkerRequest::Sleep {
            player_chunk,
            sleep_count,
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Poll for a completed sleep result. Returns None if no sleep has completed yet.
    pub fn poll_sleep_complete(&self) -> Option<SleepCompleteData> {
        let mut sc = self.sleep_complete.lock().ok()?;
        sc.take()
    }

    /// Get current engine statistics.
    pub fn get_stats(&self) -> FfiEngineStats {
        let chunks_loaded = self.store.read().map(|s| s.chunks_loaded()).unwrap_or(0);
        FfiEngineStats {
            chunks_loaded: chunks_loaded as u32,
            pending_requests: self.generate_tx.len() as u32,
            completed_results: self.result_rx.len() as u32,
            worker_threads_active: self.workers.len() as u32,
        }
    }

    /// Inject fluid at a UE world position. Computes chunk + local cell automatically.
    /// fluid_type: 1=Water, 2=Lava. Returns 1 on success, 0 on failure.
    pub fn add_fluid(&self, world_x: f32, world_y: f32, world_z: f32,
                     fluid_type: u8, is_source: bool, world_scale: f32) -> u32 {
        use crate::convert::from_ue_world_pos;
        use voxel_fluid::cell::FluidType;

        let (chunk_size, eb) = self.config.read()
            .map(|c| (c.chunk_size, c.effective_bounds()))
            .unwrap_or((16, 16.0));
        let cs = eb;

        // Convert UE world pos -> Rust voxel pos
        let rust_pos = from_ue_world_pos(world_x, world_y, world_z, world_scale);

        // Compute chunk coord and local cell
        let cx = (rust_pos.x / cs).floor() as i32;
        let cy = (rust_pos.y / cs).floor() as i32;
        let cz = (rust_pos.z / cs).floor() as i32;

        let lx = ((rust_pos.x - cx as f32 * cs) as i32).clamp(0, chunk_size as i32 - 1) as u8;
        let ly = ((rust_pos.y - cy as f32 * cs) as i32).clamp(0, chunk_size as i32 - 1) as u8;
        let lz = ((rust_pos.z - cz as f32 * cs) as i32).clamp(0, chunk_size as i32 - 1) as u8;

        let ft = if fluid_type == 2 { FluidType::Lava } else { FluidType::Water };

        match self.fluid_event_tx.try_send(FluidEvent::AddFluid {
            chunk: (cx, cy, cz),
            x: lx,
            y: ly,
            z: lz,
            fluid_type: ft,
            level: voxel_fluid::cell::MAX_LEVEL,
            is_source,
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Find the best spring location near the player.
    /// Takes UE world coords, returns UE world coords via Option.
    pub fn find_spring(&self, ue_x: f32, ue_y: f32, ue_z: f32, world_scale: f32) -> Option<(f32, f32, f32)> {
        use crate::convert::from_ue_world_pos;

        let cfg = self.config.read().ok()?;
        let chunk_size = cfg.chunk_size;
        let eb = cfg.effective_bounds();
        drop(cfg);
        let rust_pos = from_ue_world_pos(ue_x, ue_y, ue_z, world_scale);

        let store = self.store.read().ok()?;
        let best = store.find_spring_location(rust_pos, chunk_size, eb)?;

        // Convert Rust pos back to UE: (x * scale, -z * scale, y * scale)
        Some((
            best.x * world_scale,
            -best.z * world_scale,
            best.y * world_scale,
        ))
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

        let store = self.store.read().ok()?;
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

        let store = self.store.read().ok()?;
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

        let store = self.store.read().ok()?;
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

        let store = self.store.read().ok()?;
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
        let generation = self
            .generation_counters
            .entry(key)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed)
            + 1;
        if let Some(counter) = self.generation_counters.get(&key) {
            counter.store(generation, Ordering::Relaxed);
        }

        match self.mine_tx.try_send(WorkerRequest::PriorityGenerate {
            chunk: key,
            generation,
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Query the stress field for a chunk. Returns a cloned StressField if loaded.
    pub fn query_stress(&self, chunk: (i32, i32, i32)) -> Option<StressField> {
        let store = self.store.read().ok()?;
        store.stress_fields.get(&chunk).cloned()
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

        let store = match self.store.read() {
            Ok(s) => s,
            Err(_) => return 0.0,
        };
        store.stress_fields
            .get(&(cx, cy, cz))
            .map(|sf| sf.get(lx, ly, lz))
            .unwrap_or(0.0)
    }

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

    /// Request flattening a 2x2 terrace at a UE world position.
    /// Snaps to 2x2 grid and determines host rock from depth.
    /// Returns 1 on success, 0 if queue full.
    pub fn request_flatten(&self, ue_x: f32, ue_y: f32, ue_z: f32, scale: f32) -> u32 {
        let rust_x = ue_x / scale;
        let rust_y = ue_z / scale;
        let rust_z = -ue_y / scale;

        let base_x = (rust_x as i32) & !1;
        let base_y = rust_y as i32;
        let base_z = (rust_z as i32) & !1;

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

    /// Query whether a 2x2 terrace exists at a UE world position.
    /// Returns Some(material_id) if terraced, None otherwise.
    pub fn query_terrace(&self, ue_x: f32, ue_y: f32, ue_z: f32, scale: f32) -> Option<u8> {
        let rust_x = ue_x / scale;
        let rust_y = ue_z / scale;
        let rust_z = -ue_y / scale;

        let base_x = (rust_x as i32) & !1;
        let base_y = rust_y as i32;
        let base_z = (rust_z as i32) & !1;

        let store = self.store.read().unwrap();
        store
            .query_terrace(glam::IVec3::new(base_x, base_y, base_z))
            .map(|m| m as u8)
    }

    /// Query floor support for a 2x2 flatten ghost preview.
    /// Returns (solid_count, snapped_ue_x, snapped_ue_y, snapped_ue_z).
    pub fn query_flatten_support(&self, ue_x: f32, ue_y: f32, ue_z: f32, scale: f32) -> (u8, f32, f32, f32) {
        let rust_x = ue_x / scale;
        let rust_y = ue_z / scale;
        let rust_z = -ue_y / scale;

        let base_x = (rust_x as i32) & !1;
        let base_y = rust_y as i32;
        let base_z = (rust_z as i32) & !1;

        let cs = { self.config.read().unwrap().chunk_size as i32 };
        let store = self.store.read().unwrap();
        let count = store.query_flatten_support(glam::IVec3::new(base_x, base_y, base_z), cs);

        // Convert snapped position back to UE coords
        let snapped_ue_x = base_x as f32 * scale;
        let snapped_ue_y = -(base_z as f32) * scale;
        let snapped_ue_z = base_y as f32 * scale;

        (count, snapped_ue_x, snapped_ue_y, snapped_ue_z)
    }

    /// Query the host rock material at a UE world position based on depth.
    /// Returns material id as u8.
    pub fn query_host_rock_at(&self, _ue_x: f32, _ue_y: f32, ue_z: f32, scale: f32) -> u8 {
        let rust_y = ue_z / scale;
        let cfg = self.config.read().unwrap();
        voxel_gen::density::host_rock_for_depth(rust_y as f64, &cfg.ore.host_rock) as u8
    }

    /// Update the stress configuration.
    pub fn update_stress_config(&self, new_config: StressConfig) {
        if let Ok(mut cfg) = self.stress_config.write() {
            *cfg = new_config;
        }
    }

    /// Hot-reload configuration (affects future generation requests).
    pub fn update_config(&self, ffi_config: &FfiEngineConfig) {
        let new_config = ffi_config_to_generation(ffi_config);
        if let Ok(mut cfg) = self.config.write() {
            *cfg = new_config;
        }
    }

    /// Gracefully shut down all workers and wait for them to finish.
    pub fn shutdown(mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        // Drop senders to unblock recv_timeout
        drop(self.generate_tx);
        drop(self.mine_tx);
        drop(self.fluid_event_tx);
        for handle in self.workers {
            let _ = handle.join();
        }
        if let Some(handle) = self.fluid_thread.take() {
            let _ = handle.join();
        }
    }
}

/// Convert FFI config struct to internal GenerationConfig.
fn ffi_config_to_generation(c: &FfiEngineConfig) -> GenerationConfig {
    GenerationConfig {
        seed: c.seed,
        chunk_size: c.chunk_size as usize,
        noise: NoiseConfig {
            cavern_frequency: c.cavern_frequency,
            cavern_threshold: c.cavern_threshold,
            detail_octaves: c.detail_octaves,
            detail_persistence: c.detail_persistence,
            warp_amplitude: c.warp_amplitude,
        },
        worm: WormConfig {
            worms_per_region: c.worms_per_region,
            radius_min: c.worm_radius_min,
            radius_max: c.worm_radius_max,
            step_length: c.worm_step_length,
            max_steps: c.worm_max_steps,
            falloff_power: c.worm_falloff_power,
        },
        ore: OreConfig {
            host_rock: HostRockConfig {
                sandstone_depth: c.host_sandstone_depth,
                granite_depth: c.host_granite_depth,
                basalt_depth: c.host_basalt_depth,
                slate_depth: c.host_slate_depth,
                boundary_noise_amplitude: c.host_boundary_noise_amp,
                boundary_noise_frequency: c.host_boundary_noise_freq,
                basalt_intrusion_frequency: c.host_basalt_intrusion_freq,
                basalt_intrusion_threshold: c.host_basalt_intrusion_thresh,
                basalt_intrusion_depth_max: c.host_basalt_intrusion_depth_max,
            },
            iron: BandedIronConfig {
                band_frequency: c.iron_band_frequency,
                noise_perturbation: c.iron_noise_perturbation,
                noise_frequency: c.iron_noise_frequency,
                threshold: c.iron_threshold,
                depth_min: c.iron_depth_min,
                depth_max: c.iron_depth_max,
            },
            copper: OreVeinParams {
                frequency: c.copper_frequency,
                threshold: c.copper_threshold,
                depth_min: c.copper_depth_min,
                depth_max: c.copper_depth_max,
            },
            malachite: OreVeinParams {
                frequency: c.malachite_frequency,
                threshold: c.malachite_threshold,
                depth_min: c.malachite_depth_min,
                depth_max: c.malachite_depth_max,
            },
            quartz: OreVeinParams {
                frequency: c.quartz_frequency,
                threshold: c.quartz_threshold,
                depth_min: c.quartz_depth_min,
                depth_max: c.quartz_depth_max,
            },
            gold: OreVeinParams {
                frequency: c.gold_frequency,
                threshold: c.gold_threshold,
                depth_min: c.gold_depth_min,
                depth_max: c.gold_depth_max,
            },
            pyrite: OreVeinParams {
                frequency: c.pyrite_frequency,
                threshold: c.pyrite_threshold,
                depth_min: c.pyrite_depth_min,
                depth_max: c.pyrite_depth_max,
            },
            kimberlite: KimberlitePipeConfig {
                pipe_frequency_2d: c.kimb_pipe_freq_2d,
                pipe_threshold: c.kimb_pipe_threshold,
                depth_min: c.kimb_depth_min,
                depth_max: c.kimb_depth_max,
                diamond_threshold: c.kimb_diamond_threshold,
                diamond_frequency: c.kimb_diamond_frequency,
            },
            sulfide: SulfideBlobConfig {
                frequency: c.sulfide_frequency,
                threshold: c.sulfide_threshold,
                tin_threshold: c.sulfide_tin_threshold,
                depth_min: c.sulfide_depth_min,
                depth_max: c.sulfide_depth_max,
            },
            geode: GeodeConfig {
                frequency: c.geode_frequency,
                center_threshold: c.geode_center_threshold,
                shell_thickness: c.geode_shell_thickness,
                hollow_factor: c.geode_hollow_factor,
                depth_min: c.geode_depth_min,
                depth_max: c.geode_depth_max,
            },
        },
        formations: FormationConfig::default(),
        pools: PoolConfig::default(),
        mine: MineConfig {
            smooth_iterations: if c.mine_smooth_iterations == 0 && c.mine_smooth_strength == 0.0 {
                2 // default
            } else {
                c.mine_smooth_iterations
            },
            smooth_strength: if c.mine_smooth_strength > 0.0 { c.mine_smooth_strength } else { 0.3 },
            min_triangle_area: if c.mine_min_triangle_area > 0.0 { c.mine_min_triangle_area } else { 0.01 },
            dirty_expand: if c.mine_dirty_expand > 0 { c.mine_dirty_expand } else { 2 },
        },
        octree_max_depth: 4,
        max_edge_length: c.max_edge_length,
        region_size: if c.region_size == 0 { 3 } else { c.region_size as i32 },
        bounds_size: c.bounds_size,
    }
}

/// Convert FFI config to FluidConfig.
fn ffi_config_to_fluid(c: &FfiEngineConfig) -> FluidConfig {
    FluidConfig {
        seed: c.seed,
        chunk_size: c.chunk_size as usize,
        tick_rate: if c.fluid_tick_rate > 0.0 { c.fluid_tick_rate } else { 15.0 },
        lava_tick_divisor: if c.fluid_lava_tick_divisor > 0 { c.fluid_lava_tick_divisor } else { 4 },
        water_spring_threshold: if c.fluid_water_spring_threshold > 0.0 { c.fluid_water_spring_threshold } else { 2.0 },
        lava_source_threshold: if c.fluid_lava_source_threshold > 0.0 { c.fluid_lava_source_threshold } else { 0.98 },
        lava_depth_max: if c.fluid_lava_depth_max != 0.0 { c.fluid_lava_depth_max } else { -50.0 },
        water_noise_frequency: if c.fluid_water_noise_frequency > 0.0 { c.fluid_water_noise_frequency } else { 0.05 },
        water_depth_min: if c.fluid_water_depth_min != 0.0 { c.fluid_water_depth_min } else { -9999.0 },
        water_depth_max: if c.fluid_water_depth_max != 0.0 { c.fluid_water_depth_max } else { 9999.0 },
        water_flow_rate: if c.fluid_water_flow_rate > 0.0 { c.fluid_water_flow_rate } else { 0.25 },
        water_spread_rate: if c.fluid_water_spread_rate > 0.0 { c.fluid_water_spread_rate } else { 0.125 },
        lava_noise_frequency: if c.fluid_lava_noise_frequency > 0.0 { c.fluid_lava_noise_frequency } else { 0.03 },
        lava_depth_min: if c.fluid_lava_depth_min != 0.0 { c.fluid_lava_depth_min } else { -9999.0 },
        lava_flow_rate: if c.fluid_lava_flow_rate > 0.0 { c.fluid_lava_flow_rate } else { 0.1 },
        lava_spread_rate: if c.fluid_lava_spread_rate > 0.0 { c.fluid_lava_spread_rate } else { 0.125 },
        cavern_source_bias: c.fluid_cavern_source_bias,
        tunnel_bend_threshold: c.fluid_tunnel_bend_threshold,
    }
}

/// Wrapper for the fluid simulation loop that converts FluidResults to WorkerResults
/// with coordinate transformation from Rust space to UE space.
fn fluid_sim_loop_wrapper(
    shutdown: Arc<AtomicBool>,
    event_rx: Receiver<FluidEvent>,
    result_tx: Sender<WorkerResult>,
    config: FluidConfig,
    world_scale: f32,
) {
    use voxel_fluid::FluidResult;
    use voxel_fluid::thread::fluid_sim_loop;

    let chunk_size = config.chunk_size;

    // Create internal channels for the fluid sim
    let (internal_tx, internal_rx) = bounded::<FluidResult>(128);

    let sim_shutdown = Arc::clone(&shutdown);
    let sim_config = config.clone();
    let sim_handle = thread::spawn(move || {
        fluid_sim_loop(sim_shutdown, event_rx, internal_tx, sim_config);
    });

    // Relay loop: convert FluidResult -> WorkerResult with coord transform
    while !shutdown.load(Ordering::Relaxed) {
        match internal_rx.recv_timeout(std::time::Duration::from_millis(50)) {
            Ok(fluid_result) => match fluid_result {
                FluidResult::FluidMesh { chunk, mesh } => {
                    let converted = convert_fluid_mesh_to_ue(&mesh, chunk, chunk_size, world_scale);
                    let _ = result_tx.send(WorkerResult::FluidMesh {
                        chunk,
                        mesh: converted,
                    });
                }
                FluidResult::SolidifyRequest { positions } => {
                    let _ = result_tx.send(WorkerResult::SolidifyRequest { positions });
                }
            },
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }

    let _ = sim_handle.join();
}

/// Convert a fluid mesh from Rust local chunk space to UE local chunk space.
/// Positions are local [0, chunk_size] — the chunk actor provides the world offset.
fn convert_fluid_mesh_to_ue(
    mesh: &voxel_fluid::mesh::FluidMeshData,
    _chunk: (i32, i32, i32),
    _chunk_size: usize,
    scale: f32,
) -> ConvertedFluidMesh {
    let mut positions = Vec::with_capacity(mesh.positions.len());
    let mut normals = Vec::with_capacity(mesh.normals.len());

    for p in &mesh.positions {
        // Rust Y-up -> UE Z-up: (x, -z, y) * scale
        // Positions are local to the chunk (no origin offset needed)
        positions.push(FfiVec3 {
            x: p[0] * scale,
            y: -p[2] * scale,
            z: p[1] * scale,
        });
    }

    for n in &mesh.normals {
        normals.push(FfiVec3 {
            x: n[0],
            y: -n[2],
            z: n[1],
        });
    }

    ConvertedFluidMesh {
        positions,
        normals,
        fluid_types: mesh.fluid_types.clone(),
        indices: mesh.indices.clone(),
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}
