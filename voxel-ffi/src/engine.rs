use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
use crate::profiler::StreamingProfiler;
use crate::store::ChunkStore;
use crate::types::*;
use crate::worker::worker_loop;

/// Compute terrace size in voxels from world scale, targeting ~80 UU snap steps (2 voxels).
pub(crate) fn terrace_size_for_scale(scale: f32) -> i32 {
    (80.0f32 / scale).round().max(1.0) as i32
}

/// Compute building terrace size in voxels from world scale, targeting ~160 UU (4 voxels).
pub(crate) fn building_terrace_size_for_scale(scale: f32) -> i32 {
    (160.0f32 / scale).round().max(1.0) as i32
}

/// Data returned when a sleep cycle completes.
pub struct SleepCompleteData {
    pub chunks_changed: u32,
    pub voxels_metamorphosed: u32,
    pub minerals_grown: u32,
    pub supports_degraded: u32,
    pub collapses_triggered: u32,
    pub acid_dissolved: u32,
    pub veins_deposited: u32,
    pub voxels_enriched: u32,
    pub formations_grown: u32,
    pub sulfide_dissolved: u32,
    pub coal_matured: u32,
    pub diamonds_formed: u32,
    pub voxels_silicified: u32,
    pub nests_fossilized: u32,
    pub profile_report: String,
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
    sleep_config: Arc<RwLock<voxel_sleep::SleepConfig>>,
    generation_counters: Arc<DashMap<(i32, i32, i32), AtomicU64>>,
    shutdown: Arc<AtomicBool>,

    // Sleep
    sleep_complete: Arc<Mutex<Option<SleepCompleteData>>>,

    // World Scan
    scan_complete: Arc<Mutex<Option<String>>>,

    // Force Spawn Pool
    force_spawn_complete: Arc<Mutex<Option<String>>>,

    // Profiler
    profiler: Arc<StreamingProfiler>,

    // Worker threads
    workers: Vec<JoinHandle<()>>,

    // Scale
    world_scale: f32,
}

impl VoxelEngine {
    pub fn new(ffi_config: &FfiEngineConfig) -> Self {
        debug_log_pool_config(ffi_config);
        let config = ffi_config_to_generation(ffi_config);
        let voxel_scale = config.voxel_scale();
        let fluid_config = ffi_config_to_fluid(ffi_config);
        let world_scale = ffi_config.world_scale;
        // Gate 4: force single worker thread to test for concurrency races
        #[cfg(feature = "diag-gate-4")]
        let num_workers = 1;
        #[cfg(not(feature = "diag-gate-4"))]
        let num_workers = {
            if ffi_config.worker_threads == 0 {
                num_cpus()
            } else {
                ffi_config.worker_threads as usize
            }
        };

        let (generate_tx, generate_rx) = bounded::<WorkerRequest>(256);
        let (mine_tx, mine_rx) = bounded::<WorkerRequest>(16);
        let (result_tx, result_rx) = bounded::<WorkerResult>(2048);

        // Fluid event channel
        let (fluid_event_tx, fluid_event_rx) = bounded::<FluidEvent>(512);

        let region_size = config.region_size;
        let store = Arc::new(RwLock::new(ChunkStore::new(region_size)));
        let config = Arc::new(RwLock::new(config));
        let stress_config = Arc::new(RwLock::new(StressConfig::default()));
        let sleep_config = Arc::new(RwLock::new(voxel_sleep::SleepConfig::default()));
        let generation_counters: Arc<DashMap<(i32, i32, i32), AtomicU64>> =
            Arc::new(DashMap::new());
        let shutdown = Arc::new(AtomicBool::new(false));

        // Spawn fluid simulation thread
        let fluid_result_tx = result_tx.clone();
        let fluid_shutdown = Arc::clone(&shutdown);
        let fluid_world_scale = voxel_scale * world_scale;
        let fluid_thread = thread::spawn(move || {
            fluid_sim_loop_wrapper(
                fluid_shutdown,
                fluid_event_rx,
                fluid_result_tx,
                fluid_config,
                fluid_world_scale,
            );
        });

        let profiler = Arc::new(StreamingProfiler::new(num_workers));

        let mut workers = Vec::with_capacity(num_workers);
        for worker_id in 0..num_workers {
            let shutdown = Arc::clone(&shutdown);
            let generate_rx = generate_rx.clone();
            let mine_rx = mine_rx.clone();
            let result_tx = result_tx.clone();
            let store = Arc::clone(&store);
            let config = Arc::clone(&config);
            let stress_cfg = Arc::clone(&stress_config);
            let gen_counters = Arc::clone(&generation_counters);
            let fluid_tx = fluid_event_tx.clone();
            let prof = Arc::clone(&profiler);

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
                    prof,
                    worker_id,
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
            sleep_config,
            generation_counters,
            shutdown,
            sleep_complete: Arc::new(Mutex::new(None)),
            scan_complete: Arc::new(Mutex::new(None)),
            force_spawn_complete: Arc::new(Mutex::new(None)),
            profiler,
            workers,
            world_scale,
        }
    }

    /// Queue a single chunk for generation. Coords are UE space, converted internally.
    /// Returns 1 on success, 0 if queue full.
    pub fn request_generate(&self, cx: i32, cy: i32, cz: i32) -> u32 {
        let key = ue_chunk_to_rust(cx, cy, cz);
        let counter_ref = self
            .generation_counters
            .entry(key)
            .or_insert_with(|| AtomicU64::new(0));
        let generation = counter_ref.load(Ordering::Relaxed) + 1;

        match self.generate_tx.try_send(WorkerRequest::Generate {
            chunk: key,
            generation,
        }) {
            Ok(()) => {
                counter_ref.store(generation, Ordering::Relaxed);
                self.profiler.record_request(key);
                1
            }
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
                acid_dissolved,
                veins_deposited,
                voxels_enriched,
                formations_grown,
                sulfide_dissolved,
                coal_matured,
                diamonds_formed,
                voxels_silicified,
                nests_fossilized,
                profile_report,
            }) => {
                if let Ok(mut sc) = self.sleep_complete.lock() {
                    *sc = Some(SleepCompleteData {
                        chunks_changed,
                        voxels_metamorphosed,
                        minerals_grown,
                        supports_degraded,
                        collapses_triggered,
                        acid_dissolved,
                        veins_deposited,
                        voxels_enriched,
                        formations_grown,
                        sulfide_dissolved,
                        coal_matured,
                        diamonds_formed,
                        voxels_silicified,
                        nests_fossilized,
                        profile_report,
                    });
                }
                // Don't expose to the FfiResult pipeline; UE polls via voxel_poll_sleep_result
                None
            }
            Ok(WorkerResult::ScanComplete { json_report }) => {
                if let Ok(mut sc) = self.scan_complete.lock() {
                    *sc = Some(json_report);
                }
                // Don't expose to the FfiResult pipeline; UE polls via voxel_poll_scan_result
                None
            }
            Ok(WorkerResult::ForceSpawnPoolComplete { json_report }) => {
                if let Ok(mut fc) = self.force_spawn_complete.lock() {
                    *fc = Some(json_report);
                }
                None
            }
            Ok(other) => Some(other),
            Err(_) => None,
        }
    }

    /// Set spider nest positions for fossilization during sleep.
    pub fn set_sleep_nests(&self, positions: Vec<(i32, i32, i32)>) {
        if let Ok(mut sc) = self.sleep_config.write() {
            sc.nest_positions = positions;
        }
    }

    /// Start a deep sleep cycle. Sends request through the mine channel
    /// (which has exclusive write-lock priority).
    /// Returns 1 on success, 0 if queue full.
    pub fn start_sleep(&self, player_chunk: (i32, i32, i32), sleep_count: u32) -> u32 {
        let sc = self.sleep_config.read().unwrap().clone();
        match self.mine_tx.try_send(WorkerRequest::Sleep {
            player_chunk,
            sleep_count,
            sleep_config: sc,
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

    /// Request a world scan. Sent through the mine channel for immediate processing.
    /// Returns 1 on success, 0 if queue full.
    pub fn request_world_scan(&self) -> u32 {
        match self.mine_tx.try_send(WorkerRequest::WorldScan) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Request a world scan with custom configuration. Sent through the mine
    /// channel for immediate processing. Returns 1 on success, 0 if queue full.
    pub fn request_world_scan_with_config(&self, config: ScanConfig) -> u32 {
        match self.mine_tx.try_send(WorkerRequest::WorldScanWithConfig { config }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Poll for a completed scan result. Returns the JSON report string if ready.
    pub fn poll_scan_complete(&self) -> Option<String> {
        let mut sc = self.scan_complete.lock().ok()?;
        sc.take()
    }

    /// Request force-spawning a pool at a UE world position. Sent through the mine channel.
    /// fluid_type: 0=water, 1=lava. Returns 1 on success, 0 if queue full.
    pub fn request_force_spawn_pool(&self, world_x: f32, world_y: f32, world_z: f32, fluid_type: u8) -> u32 {
        match self.mine_tx.try_send(WorkerRequest::ForceSpawnPool {
            world_x,
            world_y,
            world_z,
            fluid_type,
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Request mining a sphere and filling the bottom half with fluid.
    /// Sent through the mine channel. Returns 1 on success, 0 if queue full.
    pub fn mine_and_fill_fluid(&self, world_x: f32, world_y: f32, world_z: f32, radius: f32, fluid_type: u8, world_scale: f32) -> u32 {
        match self.mine_tx.try_send(WorkerRequest::MineAndFillFluid {
            world_x,
            world_y,
            world_z,
            radius,
            fluid_type,
            world_scale,
        }) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    /// Poll for a completed force-spawn pool result. Returns the JSON diagnostics string if ready.
    pub fn poll_force_spawn_complete(&self) -> Option<String> {
        let mut fc = self.force_spawn_complete.lock().ok()?;
        fc.take()
    }

    /// Get current engine statistics.
    pub fn get_stats(&self) -> FfiEngineStats {
        let chunks_loaded = self.store.try_read().map(|s| s.chunks_loaded()).unwrap_or(0);
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

        let ft = FluidType::from_u8(fluid_type);

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

        let store = self.store.try_read().ok()?;
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
        let store = self.store.try_read().ok()?;
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

        let store = match self.store.try_read() {
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
        store
            .query_terrace(glam::IVec3::new(base_x, base_y, base_z), ts)
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
        let (count, clearance) = store.query_flatten_support(glam::IVec3::new(base_x, base_y, base_z), cs, ts);

        // Convert snapped position back to UE coords
        let snapped_ue_x = base_x as f32 * scale;
        let snapped_ue_y = -(base_z as f32) * scale;
        let snapped_ue_z = base_y as f32 * scale;

        (count, clearance, snapped_ue_x, snapped_ue_y, snapped_ue_z)
    }

    /// Query floor support for a building placement (4x4 footprint).
    /// Returns (solid_count, total_columns, host_mat_u8).
    pub fn query_building_support(&self, ue_x: f32, ue_y: f32, ue_z: f32, scale: f32) -> (u8, u8, u8) {
        let rust_x = ue_x / scale;
        let rust_y = ue_z / scale;
        let rust_z = -ue_y / scale;

        let bts = building_terrace_size_for_scale(scale);
        let ts = terrace_size_for_scale(scale);
        let base_x = (rust_x as i32).div_euclid(bts) * bts;
        let base_y = (rust_y as i32).div_euclid(ts) * ts;
        let base_z = (rust_z as i32).div_euclid(bts) * bts;

        let cs = { self.config.read().unwrap().chunk_size as i32 };
        let store = match self.store.try_read() {
            Ok(s) => s,
            Err(_) => return (0, 0, 0),
        };
        let (solid, total, mat) = store.query_building_support(glam::IVec3::new(base_x, base_y, base_z), cs, bts);
        (solid, total, mat as u8)
    }

    /// Request auto-terrace for a building placement (4x4 footprint).
    /// Returns 1 on success, 0 if queue full.
    pub fn request_building_flatten(&self, ue_x: f32, ue_y: f32, ue_z: f32, scale: f32) -> u32 {
        let rust_x = ue_x / scale;
        let rust_y = ue_z / scale;
        let rust_z = -ue_y / scale;

        let bts = building_terrace_size_for_scale(scale);
        let ts = terrace_size_for_scale(scale);
        let base_x = (rust_x as i32).div_euclid(bts) * bts;
        let base_y = (rust_y as i32).div_euclid(ts) * ts;
        let base_z = (rust_z as i32).div_euclid(bts) * bts;

        let host_material = {
            let cfg = self.config.read().unwrap();
            voxel_gen::density::host_rock_for_depth(rust_y as f64, &cfg.ore.host_rock) as u8
        };

        match self.mine_tx.send_timeout(WorkerRequest::BuildingFlatten {
            base_x,
            base_y,
            base_z,
            host_material,
        }, std::time::Duration::from_millis(100)) {
            Ok(()) => 1,
            Err(e) => {
                eprintln!("[voxel] request_building_flatten: send failed: {}", e);
                0
            }
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
        let max_y_diff = 3;
        store
            .query_nearby_terrace_y(base_x, base_z, approx_y, search_radius, max_y_diff)
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

    /// Update the stress configuration.
    pub fn update_stress_config(&self, new_config: StressConfig) {
        if let Ok(mut cfg) = self.stress_config.write() {
            *cfg = new_config;
        }
    }

    /// Update the sleep configuration.
    pub fn update_sleep_config(&self, new_config: voxel_sleep::SleepConfig) {
        if let Ok(mut cfg) = self.sleep_config.write() {
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

    // ── Profiler API ──

    /// Enable or disable the streaming profiler.
    pub fn profiler_set_enabled(&self, enabled: bool) {
        self.profiler.set_enabled(enabled);
    }

    /// Check if profiler is enabled.
    pub fn profiler_is_enabled(&self) -> bool {
        self.profiler.is_enabled()
    }

    /// Begin a new profiling session. Resets metrics, captures config snapshot.
    pub fn profiler_begin_session(&self) -> u64 {
        let config_snapshot = if let Ok(cfg) = self.config.read() {
            format!(
                "seed={}\nchunk_size={}\nworkers={}\nworld_scale={:.1}\nregion_size={}\n\
                 cavern_freq={:.4}\ncavern_thresh={:.2}\ndetail_octaves={}\ndetail_persistence={:.2}\nwarp_amplitude={:.1}\n\
                 worms_per_region={:.1}\nworm_radius_min={:.1}\nworm_radius_max={:.1}\nworm_step_length={:.1}\nworm_max_steps={}\nworm_falloff_power={:.1}\n\
                 ore_domain_warp_strength={:.2}\nore_warp_frequency={:.4}\nore_edge_falloff={:.4}\nore_detail_weight={:.2}\n\
                 mesh_smooth_iterations={}\nmesh_smooth_strength={:.2}\nmesh_boundary_smooth={:.2}\nmesh_recalc_normals={}",
                cfg.seed, cfg.chunk_size,
                self.workers.len(), self.world_scale, cfg.region_size,
                cfg.noise.cavern_frequency, cfg.noise.cavern_threshold,
                cfg.noise.detail_octaves, cfg.noise.detail_persistence, cfg.noise.warp_amplitude,
                cfg.worm.worms_per_region, cfg.worm.radius_min, cfg.worm.radius_max,
                cfg.worm.step_length, cfg.worm.max_steps, cfg.worm.falloff_power,
                cfg.ore.ore_domain_warp_strength, cfg.ore.ore_warp_frequency,
                cfg.ore.ore_edge_falloff, cfg.ore.ore_detail_weight,
                cfg.mesh_smooth_iterations, cfg.mesh_smooth_strength,
                cfg.mesh_boundary_smooth, cfg.mesh_recalc_normals,
            )
        } else {
            "(config unavailable)".to_string()
        };
        self.profiler.begin_session(config_snapshot)
    }

    /// End the current profiling session.
    pub fn profiler_end_session(&self) {
        self.profiler.end_session();
    }

    /// Generate a plain-text profiling report.
    pub fn profiler_get_report(&self) -> String {
        self.profiler.generate_report()
    }

    /// Generate report as C string for FFI. Caller must free with voxel_profiler_free_report.
    pub fn profiler_get_report_cstr(&self) -> *mut std::ffi::c_char {
        self.profiler.generate_report_cstr()
    }

    /// Get a reference to the profiler (for FFI poll instrumentation).
    pub fn profiler(&self) -> &Arc<StreamingProfiler> {
        &self.profiler
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
            coal: OreVeinParams {
                frequency: if c.ore_coal_frequency > 0.0 { c.ore_coal_frequency } else { 0.03 },
                threshold: if c.ore_coal_threshold > 0.0 { c.ore_coal_threshold } else { 0.62 },
                depth_min: c.ore_coal_depth_min,
                depth_max: if c.ore_coal_depth_max > 0.0 { c.ore_coal_depth_max } else { 80.0 },
            },
            ore_domain_warp_strength: c.ore_domain_warp_strength,
            ore_warp_frequency: if c.ore_warp_frequency > 0.0 { c.ore_warp_frequency } else { 0.02 },
            ore_edge_falloff: c.ore_edge_falloff,
            ore_detail_weight: c.ore_detail_weight,
            iron_sedimentary_only: c.ore_iron_sedimentary_only != 0,
            iron_depth_fade: c.ore_iron_depth_fade != 0,
            copper_supergene: c.ore_copper_supergene != 0,
            copper_granite_contact: c.ore_copper_granite_contact != 0,
            malachite_depth_bias: c.ore_malachite_depth_bias != 0,
            kimberlite_carrot_taper: c.ore_kimberlite_carrot_taper != 0,
            diamond_depth_grade: c.ore_diamond_depth_grade != 0,
            sulfide_gossan_cap: c.ore_sulfide_gossan_cap != 0,
            sulfide_disseminated: c.ore_sulfide_disseminated != 0,
            pyrite_ore_halo: c.ore_pyrite_ore_halo != 0,
            quartz_planar_veins: c.ore_quartz_planar_veins != 0,
            gold_bonanza: c.ore_gold_bonanza != 0,
            geode_volcanic_host: c.ore_geode_volcanic_host != 0,
            geode_depth_scaling: c.ore_geode_depth_scaling != 0,
            coal_sedimentary_host: c.ore_coal_sedimentary_host != 0,
            coal_shallow_ceiling: c.ore_coal_shallow_ceiling != 0,
            coal_depth_enrichment: c.ore_coal_depth_enrichment != 0,
        },
        formations: FormationConfig {
            enabled: c.formation_enabled != 0,
            placement_frequency: if c.formation_placement_frequency > 0.0 { c.formation_placement_frequency as f64 } else { FormationConfig::default().placement_frequency },
            placement_threshold: if c.formation_placement_threshold > 0.0 { c.formation_placement_threshold as f64 } else { FormationConfig::default().placement_threshold },
            stalactite_chance: if c.formation_stalactite_chance > 0.0 { c.formation_stalactite_chance } else { FormationConfig::default().stalactite_chance },
            stalagmite_chance: if c.formation_stalagmite_chance > 0.0 { c.formation_stalagmite_chance } else { FormationConfig::default().stalagmite_chance },
            flowstone_chance: if c.formation_flowstone_chance > 0.0 { c.formation_flowstone_chance } else { FormationConfig::default().flowstone_chance },
            column_chance: if c.formation_column_chance > 0.0 { c.formation_column_chance } else { FormationConfig::default().column_chance },
            column_max_gap: if c.formation_column_max_gap > 0 { c.formation_column_max_gap as usize } else { FormationConfig::default().column_max_gap },
            length_min: if c.formation_length_min > 0.0 { c.formation_length_min } else { FormationConfig::default().length_min },
            length_max: if c.formation_length_max > 0.0 { c.formation_length_max } else { FormationConfig::default().length_max },
            radius_min: if c.formation_radius_min > 0.0 { c.formation_radius_min } else { FormationConfig::default().radius_min },
            radius_max: if c.formation_radius_max > 0.0 { c.formation_radius_max } else { FormationConfig::default().radius_max },
            max_radius: if c.formation_max_radius > 0.0 { c.formation_max_radius } else { FormationConfig::default().max_radius },
            column_radius_min: if c.formation_column_radius_min > 0.0 { c.formation_column_radius_min } else { FormationConfig::default().column_radius_min },
            column_radius_max: if c.formation_column_radius_max > 0.0 { c.formation_column_radius_max } else { FormationConfig::default().column_radius_max },
            flowstone_length_min: if c.formation_flowstone_length_min > 0.0 { c.formation_flowstone_length_min } else { FormationConfig::default().flowstone_length_min },
            flowstone_length_max: if c.formation_flowstone_length_max > 0.0 { c.formation_flowstone_length_max } else { FormationConfig::default().flowstone_length_max },
            flowstone_thickness: if c.formation_flowstone_thickness > 0.0 { c.formation_flowstone_thickness } else { FormationConfig::default().flowstone_thickness },
            min_air_gap: if c.formation_min_air_gap > 0 { c.formation_min_air_gap as usize } else { FormationConfig::default().min_air_gap },
            min_clearance: if c.formation_min_clearance > 0 { c.formation_min_clearance as usize } else { FormationConfig::default().min_clearance },
            smoothness: if c.formation_smoothness > 0.0 { c.formation_smoothness } else { FormationConfig::default().smoothness },
            // Mega-Column
            mega_column_chance: if c.formation_mega_column_chance > 0.0 { c.formation_mega_column_chance } else { FormationConfig::default().mega_column_chance },
            mega_column_min_gap: if c.formation_mega_column_min_gap > 0 { c.formation_mega_column_min_gap as usize } else { FormationConfig::default().mega_column_min_gap },
            mega_column_radius_min: if c.formation_mega_column_radius_min > 0.0 { c.formation_mega_column_radius_min } else { FormationConfig::default().mega_column_radius_min },
            mega_column_radius_max: if c.formation_mega_column_radius_max > 0.0 { c.formation_mega_column_radius_max } else { FormationConfig::default().mega_column_radius_max },
            mega_column_noise_strength: if c.formation_mega_column_noise_strength > 0.0 { c.formation_mega_column_noise_strength } else { FormationConfig::default().mega_column_noise_strength },
            mega_column_ring_frequency: if c.formation_mega_column_ring_frequency > 0.0 { c.formation_mega_column_ring_frequency } else { FormationConfig::default().mega_column_ring_frequency },
            // Drapery
            drapery_chance: if c.formation_drapery_chance > 0.0 { c.formation_drapery_chance } else { FormationConfig::default().drapery_chance },
            drapery_length_min: if c.formation_drapery_length_min > 0.0 { c.formation_drapery_length_min } else { FormationConfig::default().drapery_length_min },
            drapery_length_max: if c.formation_drapery_length_max > 0.0 { c.formation_drapery_length_max } else { FormationConfig::default().drapery_length_max },
            drapery_wave_frequency: if c.formation_drapery_wave_frequency > 0.0 { c.formation_drapery_wave_frequency } else { FormationConfig::default().drapery_wave_frequency },
            drapery_wave_amplitude: if c.formation_drapery_wave_amplitude > 0.0 { c.formation_drapery_wave_amplitude } else { FormationConfig::default().drapery_wave_amplitude },
            // Rimstone Dam
            rimstone_chance: if c.formation_rimstone_chance > 0.0 { c.formation_rimstone_chance } else { FormationConfig::default().rimstone_chance },
            rimstone_dam_height_min: if c.formation_rimstone_dam_height_min > 0.0 { c.formation_rimstone_dam_height_min } else { FormationConfig::default().rimstone_dam_height_min },
            rimstone_dam_height_max: if c.formation_rimstone_dam_height_max > 0.0 { c.formation_rimstone_dam_height_max } else { FormationConfig::default().rimstone_dam_height_max },
            rimstone_pool_depth: if c.formation_rimstone_pool_depth > 0.0 { c.formation_rimstone_pool_depth } else { FormationConfig::default().rimstone_pool_depth },
            rimstone_min_slope: if c.formation_rimstone_min_slope > 0.0 { c.formation_rimstone_min_slope } else { FormationConfig::default().rimstone_min_slope },
            // Cave Shield
            shield_chance: if c.formation_shield_chance > 0.0 { c.formation_shield_chance } else { FormationConfig::default().shield_chance },
            shield_radius_min: if c.formation_shield_radius_min > 0.0 { c.formation_shield_radius_min } else { FormationConfig::default().shield_radius_min },
            shield_radius_max: if c.formation_shield_radius_max > 0.0 { c.formation_shield_radius_max } else { FormationConfig::default().shield_radius_max },
            shield_max_tilt: if c.formation_shield_max_tilt > 0.0 { c.formation_shield_max_tilt } else { FormationConfig::default().shield_max_tilt },
            shield_stalactite_chance: if c.formation_shield_stalactite_chance > 0.0 { c.formation_shield_stalactite_chance } else { FormationConfig::default().shield_stalactite_chance },
            // Cauldron
            cauldron_chance: if c.formation_cauldron_chance > 0.0 { c.formation_cauldron_chance } else { FormationConfig::default().cauldron_chance },
            cauldron_radius_min: if c.formation_cauldron_radius_min > 0.0 { c.formation_cauldron_radius_min } else { FormationConfig::default().cauldron_radius_min },
            cauldron_radius_max: if c.formation_cauldron_radius_max > 0.0 { c.formation_cauldron_radius_max } else { FormationConfig::default().cauldron_radius_max },
            cauldron_depth: if c.formation_cauldron_depth > 0.0 { c.formation_cauldron_depth } else { FormationConfig::default().cauldron_depth },
            cauldron_lip_height: if c.formation_cauldron_lip_height > 0.0 { c.formation_cauldron_lip_height } else { FormationConfig::default().cauldron_lip_height },
            cauldron_rim_stalagmite_count_min: if c.formation_cauldron_rim_stalagmite_count_min > 0 { c.formation_cauldron_rim_stalagmite_count_min } else { FormationConfig::default().cauldron_rim_stalagmite_count_min },
            cauldron_rim_stalagmite_count_max: if c.formation_cauldron_rim_stalagmite_count_max > 0 { c.formation_cauldron_rim_stalagmite_count_max } else { FormationConfig::default().cauldron_rim_stalagmite_count_max },
            cauldron_rim_stalagmite_scale: if c.formation_cauldron_rim_stalagmite_scale > 0.0 { c.formation_cauldron_rim_stalagmite_scale } else { FormationConfig::default().cauldron_rim_stalagmite_scale },
            cauldron_floor_noise: if c.formation_cauldron_floor_noise >= 0.0 { c.formation_cauldron_floor_noise } else { FormationConfig::default().cauldron_floor_noise },
            cauldron_water_chance: if c.formation_cauldron_water_chance >= 0.0 { c.formation_cauldron_water_chance } else { FormationConfig::default().cauldron_water_chance },
            cauldron_lava_chance: if c.formation_cauldron_lava_chance >= 0.0 { c.formation_cauldron_lava_chance } else { FormationConfig::default().cauldron_lava_chance },
            cauldron_wall_inset: if c.formation_cauldron_wall_inset > 0.0 { c.formation_cauldron_wall_inset } else { FormationConfig::default().cauldron_wall_inset },
            cauldron_floor_inset: if c.formation_cauldron_floor_inset > 0 { c.formation_cauldron_floor_inset } else { FormationConfig::default().cauldron_floor_inset },
        },
        pools: PoolConfig {
            enabled: c.pool_enabled != 0,
            // Pool fields pass through directly — the C++ struct already has correct
            // default initializers, and placement_threshold legitimately uses negative
            // values (e.g. -1.0 = accept all noise values).
            placement_frequency: if c.pool_placement_freq != 0.0 { c.pool_placement_freq } else { 0.08 },
            placement_threshold: c.pool_placement_thresh, // can be negative
            pool_chance: if c.pool_chance > 0.0 { c.pool_chance } else { 0.3 },
            min_area: if c.pool_min_area > 0 { c.pool_min_area as usize } else { 4 },
            max_radius: if c.pool_max_radius > 0 { c.pool_max_radius as usize } else { 4 },
            basin_depth: if c.pool_basin_depth > 0 { c.pool_basin_depth as usize } else { 2 },
            rim_height: if c.pool_rim_height > 0 { c.pool_rim_height as usize } else { 1 },
            water_pct: if c.pool_water_pct > 0.0 || c.pool_lava_pct > 0.0 || c.pool_empty_pct > 0.0 {
                c.pool_water_pct
            } else { 0.75 },
            lava_pct: if c.pool_water_pct > 0.0 || c.pool_lava_pct > 0.0 || c.pool_empty_pct > 0.0 {
                c.pool_lava_pct
            } else { 0.25 },
            empty_pct: c.pool_empty_pct,
            min_air_above: if c.pool_min_air_above > 0 { c.pool_min_air_above as usize } else { 3 },
            max_cave_height: if c.pool_max_cave_height > 0 { c.pool_max_cave_height as usize } else { 20 },
            min_floor_thickness: if c.pool_min_floor_thickness > 0 { c.pool_min_floor_thickness as usize } else { 2 },
            min_ground_depth: if c.pool_min_ground_depth > 0 { c.pool_min_ground_depth as usize } else { 2 },
            max_y_step: if c.pool_max_y_step > 0 { c.pool_max_y_step as usize } else { 2 },
            footprint_y_tolerance: if c.pool_footprint_y_tolerance > 0 { c.pool_footprint_y_tolerance as usize } else { 2 },
        },
        mine: MineConfig {
            smooth_iterations: if c.mine_smooth_iterations == 0 && c.mine_smooth_strength == 0.0 {
                2 // default
            } else {
                c.mine_smooth_iterations
            },
            smooth_strength: if c.mine_smooth_strength > 0.0 { c.mine_smooth_strength } else { 0.3 },
            dirty_expand: if c.mine_dirty_expand > 0 { c.mine_dirty_expand } else { 2 },
        },
        crystals: CrystalConfig {
            enabled: c.crystal_enabled != 0,
            iron: OreCrystalConfig {
                enabled: c.crystal_iron_enabled != 0,
                chance: if c.crystal_iron_chance > 0.0 { c.crystal_iron_chance } else { OreCrystalConfig::default().chance },
                density_threshold: c.crystal_iron_density_threshold,
                scale_min: if c.crystal_iron_scale_min > 0.0 { c.crystal_iron_scale_min } else { OreCrystalConfig::default().scale_min },
                scale_max: if c.crystal_iron_scale_max > 0.0 { c.crystal_iron_scale_max } else { OreCrystalConfig::default().scale_max },
                small_weight: if c.crystal_iron_small_weight > 0.0 { c.crystal_iron_small_weight } else { OreCrystalConfig::default().small_weight },
                medium_weight: if c.crystal_iron_medium_weight > 0.0 { c.crystal_iron_medium_weight } else { OreCrystalConfig::default().medium_weight },
                large_weight: if c.crystal_iron_large_weight > 0.0 { c.crystal_iron_large_weight } else { OreCrystalConfig::default().large_weight },
                normal_alignment: c.crystal_iron_normal_alignment,
                cluster_size: if c.crystal_iron_cluster_size > 0 { c.crystal_iron_cluster_size } else { OreCrystalConfig::default().cluster_size },
                cluster_radius: if c.crystal_iron_cluster_radius > 0.0 { c.crystal_iron_cluster_radius } else { OreCrystalConfig::default().cluster_radius },
                surface_offset: c.crystal_iron_surface_offset,
                vein_enabled: c.crystal_iron_vein_enabled != 0,
                vein_frequency: if c.crystal_iron_vein_frequency > 0.0 { c.crystal_iron_vein_frequency } else { OreCrystalConfig::default().vein_frequency },
                vein_thickness: if c.crystal_iron_vein_thickness > 0.0 { c.crystal_iron_vein_thickness } else { OreCrystalConfig::default().vein_thickness },
                vein_octaves: if c.crystal_iron_vein_octaves > 0 { c.crystal_iron_vein_octaves } else { OreCrystalConfig::default().vein_octaves },
                vein_lacunarity: if c.crystal_iron_vein_lacunarity > 0.0 { c.crystal_iron_vein_lacunarity } else { OreCrystalConfig::default().vein_lacunarity },
                vein_warp_strength: c.crystal_iron_vein_warp_strength,
                vein_density: if c.crystal_iron_vein_density > 0.0 { c.crystal_iron_vein_density } else { OreCrystalConfig::default().vein_density },
            },
            copper: OreCrystalConfig {
                enabled: c.crystal_copper_enabled != 0,
                chance: if c.crystal_copper_chance > 0.0 { c.crystal_copper_chance } else { OreCrystalConfig::default().chance },
                density_threshold: c.crystal_copper_density_threshold,
                scale_min: if c.crystal_copper_scale_min > 0.0 { c.crystal_copper_scale_min } else { OreCrystalConfig::default().scale_min },
                scale_max: if c.crystal_copper_scale_max > 0.0 { c.crystal_copper_scale_max } else { OreCrystalConfig::default().scale_max },
                small_weight: if c.crystal_copper_small_weight > 0.0 { c.crystal_copper_small_weight } else { OreCrystalConfig::default().small_weight },
                medium_weight: if c.crystal_copper_medium_weight > 0.0 { c.crystal_copper_medium_weight } else { OreCrystalConfig::default().medium_weight },
                large_weight: if c.crystal_copper_large_weight > 0.0 { c.crystal_copper_large_weight } else { OreCrystalConfig::default().large_weight },
                normal_alignment: c.crystal_copper_normal_alignment,
                cluster_size: if c.crystal_copper_cluster_size > 0 { c.crystal_copper_cluster_size } else { OreCrystalConfig::default().cluster_size },
                cluster_radius: if c.crystal_copper_cluster_radius > 0.0 { c.crystal_copper_cluster_radius } else { OreCrystalConfig::default().cluster_radius },
                surface_offset: c.crystal_copper_surface_offset,
                vein_enabled: c.crystal_copper_vein_enabled != 0,
                vein_frequency: if c.crystal_copper_vein_frequency > 0.0 { c.crystal_copper_vein_frequency } else { OreCrystalConfig::default().vein_frequency },
                vein_thickness: if c.crystal_copper_vein_thickness > 0.0 { c.crystal_copper_vein_thickness } else { OreCrystalConfig::default().vein_thickness },
                vein_octaves: if c.crystal_copper_vein_octaves > 0 { c.crystal_copper_vein_octaves } else { OreCrystalConfig::default().vein_octaves },
                vein_lacunarity: if c.crystal_copper_vein_lacunarity > 0.0 { c.crystal_copper_vein_lacunarity } else { OreCrystalConfig::default().vein_lacunarity },
                vein_warp_strength: c.crystal_copper_vein_warp_strength,
                vein_density: if c.crystal_copper_vein_density > 0.0 { c.crystal_copper_vein_density } else { OreCrystalConfig::default().vein_density },
            },
            malachite: OreCrystalConfig {
                enabled: c.crystal_malachite_enabled != 0,
                chance: if c.crystal_malachite_chance > 0.0 { c.crystal_malachite_chance } else { OreCrystalConfig::default().chance },
                density_threshold: c.crystal_malachite_density_threshold,
                scale_min: if c.crystal_malachite_scale_min > 0.0 { c.crystal_malachite_scale_min } else { OreCrystalConfig::default().scale_min },
                scale_max: if c.crystal_malachite_scale_max > 0.0 { c.crystal_malachite_scale_max } else { OreCrystalConfig::default().scale_max },
                small_weight: if c.crystal_malachite_small_weight > 0.0 { c.crystal_malachite_small_weight } else { OreCrystalConfig::default().small_weight },
                medium_weight: if c.crystal_malachite_medium_weight > 0.0 { c.crystal_malachite_medium_weight } else { OreCrystalConfig::default().medium_weight },
                large_weight: if c.crystal_malachite_large_weight > 0.0 { c.crystal_malachite_large_weight } else { OreCrystalConfig::default().large_weight },
                normal_alignment: c.crystal_malachite_normal_alignment,
                cluster_size: if c.crystal_malachite_cluster_size > 0 { c.crystal_malachite_cluster_size } else { OreCrystalConfig::default().cluster_size },
                cluster_radius: if c.crystal_malachite_cluster_radius > 0.0 { c.crystal_malachite_cluster_radius } else { OreCrystalConfig::default().cluster_radius },
                surface_offset: c.crystal_malachite_surface_offset,
                vein_enabled: c.crystal_malachite_vein_enabled != 0,
                vein_frequency: if c.crystal_malachite_vein_frequency > 0.0 { c.crystal_malachite_vein_frequency } else { OreCrystalConfig::default().vein_frequency },
                vein_thickness: if c.crystal_malachite_vein_thickness > 0.0 { c.crystal_malachite_vein_thickness } else { OreCrystalConfig::default().vein_thickness },
                vein_octaves: if c.crystal_malachite_vein_octaves > 0 { c.crystal_malachite_vein_octaves } else { OreCrystalConfig::default().vein_octaves },
                vein_lacunarity: if c.crystal_malachite_vein_lacunarity > 0.0 { c.crystal_malachite_vein_lacunarity } else { OreCrystalConfig::default().vein_lacunarity },
                vein_warp_strength: c.crystal_malachite_vein_warp_strength,
                vein_density: if c.crystal_malachite_vein_density > 0.0 { c.crystal_malachite_vein_density } else { OreCrystalConfig::default().vein_density },
            },
            tin: OreCrystalConfig {
                enabled: c.crystal_tin_enabled != 0,
                chance: if c.crystal_tin_chance > 0.0 { c.crystal_tin_chance } else { OreCrystalConfig::default().chance },
                density_threshold: c.crystal_tin_density_threshold,
                scale_min: if c.crystal_tin_scale_min > 0.0 { c.crystal_tin_scale_min } else { OreCrystalConfig::default().scale_min },
                scale_max: if c.crystal_tin_scale_max > 0.0 { c.crystal_tin_scale_max } else { OreCrystalConfig::default().scale_max },
                small_weight: if c.crystal_tin_small_weight > 0.0 { c.crystal_tin_small_weight } else { OreCrystalConfig::default().small_weight },
                medium_weight: if c.crystal_tin_medium_weight > 0.0 { c.crystal_tin_medium_weight } else { OreCrystalConfig::default().medium_weight },
                large_weight: if c.crystal_tin_large_weight > 0.0 { c.crystal_tin_large_weight } else { OreCrystalConfig::default().large_weight },
                normal_alignment: c.crystal_tin_normal_alignment,
                cluster_size: if c.crystal_tin_cluster_size > 0 { c.crystal_tin_cluster_size } else { OreCrystalConfig::default().cluster_size },
                cluster_radius: if c.crystal_tin_cluster_radius > 0.0 { c.crystal_tin_cluster_radius } else { OreCrystalConfig::default().cluster_radius },
                surface_offset: c.crystal_tin_surface_offset,
                vein_enabled: c.crystal_tin_vein_enabled != 0,
                vein_frequency: if c.crystal_tin_vein_frequency > 0.0 { c.crystal_tin_vein_frequency } else { OreCrystalConfig::default().vein_frequency },
                vein_thickness: if c.crystal_tin_vein_thickness > 0.0 { c.crystal_tin_vein_thickness } else { OreCrystalConfig::default().vein_thickness },
                vein_octaves: if c.crystal_tin_vein_octaves > 0 { c.crystal_tin_vein_octaves } else { OreCrystalConfig::default().vein_octaves },
                vein_lacunarity: if c.crystal_tin_vein_lacunarity > 0.0 { c.crystal_tin_vein_lacunarity } else { OreCrystalConfig::default().vein_lacunarity },
                vein_warp_strength: c.crystal_tin_vein_warp_strength,
                vein_density: if c.crystal_tin_vein_density > 0.0 { c.crystal_tin_vein_density } else { OreCrystalConfig::default().vein_density },
            },
            gold: OreCrystalConfig {
                enabled: c.crystal_gold_enabled != 0,
                chance: if c.crystal_gold_chance > 0.0 { c.crystal_gold_chance } else { OreCrystalConfig::default().chance },
                density_threshold: c.crystal_gold_density_threshold,
                scale_min: if c.crystal_gold_scale_min > 0.0 { c.crystal_gold_scale_min } else { OreCrystalConfig::default().scale_min },
                scale_max: if c.crystal_gold_scale_max > 0.0 { c.crystal_gold_scale_max } else { OreCrystalConfig::default().scale_max },
                small_weight: if c.crystal_gold_small_weight > 0.0 { c.crystal_gold_small_weight } else { OreCrystalConfig::default().small_weight },
                medium_weight: if c.crystal_gold_medium_weight > 0.0 { c.crystal_gold_medium_weight } else { OreCrystalConfig::default().medium_weight },
                large_weight: if c.crystal_gold_large_weight > 0.0 { c.crystal_gold_large_weight } else { OreCrystalConfig::default().large_weight },
                normal_alignment: c.crystal_gold_normal_alignment,
                cluster_size: if c.crystal_gold_cluster_size > 0 { c.crystal_gold_cluster_size } else { OreCrystalConfig::default().cluster_size },
                cluster_radius: if c.crystal_gold_cluster_radius > 0.0 { c.crystal_gold_cluster_radius } else { OreCrystalConfig::default().cluster_radius },
                surface_offset: c.crystal_gold_surface_offset,
                vein_enabled: c.crystal_gold_vein_enabled != 0,
                vein_frequency: if c.crystal_gold_vein_frequency > 0.0 { c.crystal_gold_vein_frequency } else { OreCrystalConfig::default().vein_frequency },
                vein_thickness: if c.crystal_gold_vein_thickness > 0.0 { c.crystal_gold_vein_thickness } else { OreCrystalConfig::default().vein_thickness },
                vein_octaves: if c.crystal_gold_vein_octaves > 0 { c.crystal_gold_vein_octaves } else { OreCrystalConfig::default().vein_octaves },
                vein_lacunarity: if c.crystal_gold_vein_lacunarity > 0.0 { c.crystal_gold_vein_lacunarity } else { OreCrystalConfig::default().vein_lacunarity },
                vein_warp_strength: c.crystal_gold_vein_warp_strength,
                vein_density: if c.crystal_gold_vein_density > 0.0 { c.crystal_gold_vein_density } else { OreCrystalConfig::default().vein_density },
            },
            diamond: OreCrystalConfig {
                enabled: c.crystal_diamond_enabled != 0,
                chance: if c.crystal_diamond_chance > 0.0 { c.crystal_diamond_chance } else { OreCrystalConfig::default().chance },
                density_threshold: c.crystal_diamond_density_threshold,
                scale_min: if c.crystal_diamond_scale_min > 0.0 { c.crystal_diamond_scale_min } else { OreCrystalConfig::default().scale_min },
                scale_max: if c.crystal_diamond_scale_max > 0.0 { c.crystal_diamond_scale_max } else { OreCrystalConfig::default().scale_max },
                small_weight: if c.crystal_diamond_small_weight > 0.0 { c.crystal_diamond_small_weight } else { OreCrystalConfig::default().small_weight },
                medium_weight: if c.crystal_diamond_medium_weight > 0.0 { c.crystal_diamond_medium_weight } else { OreCrystalConfig::default().medium_weight },
                large_weight: if c.crystal_diamond_large_weight > 0.0 { c.crystal_diamond_large_weight } else { OreCrystalConfig::default().large_weight },
                normal_alignment: c.crystal_diamond_normal_alignment,
                cluster_size: if c.crystal_diamond_cluster_size > 0 { c.crystal_diamond_cluster_size } else { OreCrystalConfig::default().cluster_size },
                cluster_radius: if c.crystal_diamond_cluster_radius > 0.0 { c.crystal_diamond_cluster_radius } else { OreCrystalConfig::default().cluster_radius },
                surface_offset: c.crystal_diamond_surface_offset,
                vein_enabled: c.crystal_diamond_vein_enabled != 0,
                vein_frequency: if c.crystal_diamond_vein_frequency > 0.0 { c.crystal_diamond_vein_frequency } else { OreCrystalConfig::default().vein_frequency },
                vein_thickness: if c.crystal_diamond_vein_thickness > 0.0 { c.crystal_diamond_vein_thickness } else { OreCrystalConfig::default().vein_thickness },
                vein_octaves: if c.crystal_diamond_vein_octaves > 0 { c.crystal_diamond_vein_octaves } else { OreCrystalConfig::default().vein_octaves },
                vein_lacunarity: if c.crystal_diamond_vein_lacunarity > 0.0 { c.crystal_diamond_vein_lacunarity } else { OreCrystalConfig::default().vein_lacunarity },
                vein_warp_strength: c.crystal_diamond_vein_warp_strength,
                vein_density: if c.crystal_diamond_vein_density > 0.0 { c.crystal_diamond_vein_density } else { OreCrystalConfig::default().vein_density },
            },
            kimberlite: OreCrystalConfig {
                enabled: c.crystal_kimberlite_enabled != 0,
                chance: if c.crystal_kimberlite_chance > 0.0 { c.crystal_kimberlite_chance } else { OreCrystalConfig::default().chance },
                density_threshold: c.crystal_kimberlite_density_threshold,
                scale_min: if c.crystal_kimberlite_scale_min > 0.0 { c.crystal_kimberlite_scale_min } else { OreCrystalConfig::default().scale_min },
                scale_max: if c.crystal_kimberlite_scale_max > 0.0 { c.crystal_kimberlite_scale_max } else { OreCrystalConfig::default().scale_max },
                small_weight: if c.crystal_kimberlite_small_weight > 0.0 { c.crystal_kimberlite_small_weight } else { OreCrystalConfig::default().small_weight },
                medium_weight: if c.crystal_kimberlite_medium_weight > 0.0 { c.crystal_kimberlite_medium_weight } else { OreCrystalConfig::default().medium_weight },
                large_weight: if c.crystal_kimberlite_large_weight > 0.0 { c.crystal_kimberlite_large_weight } else { OreCrystalConfig::default().large_weight },
                normal_alignment: c.crystal_kimberlite_normal_alignment,
                cluster_size: if c.crystal_kimberlite_cluster_size > 0 { c.crystal_kimberlite_cluster_size } else { OreCrystalConfig::default().cluster_size },
                cluster_radius: if c.crystal_kimberlite_cluster_radius > 0.0 { c.crystal_kimberlite_cluster_radius } else { OreCrystalConfig::default().cluster_radius },
                surface_offset: c.crystal_kimberlite_surface_offset,
                vein_enabled: c.crystal_kimberlite_vein_enabled != 0,
                vein_frequency: if c.crystal_kimberlite_vein_frequency > 0.0 { c.crystal_kimberlite_vein_frequency } else { OreCrystalConfig::default().vein_frequency },
                vein_thickness: if c.crystal_kimberlite_vein_thickness > 0.0 { c.crystal_kimberlite_vein_thickness } else { OreCrystalConfig::default().vein_thickness },
                vein_octaves: if c.crystal_kimberlite_vein_octaves > 0 { c.crystal_kimberlite_vein_octaves } else { OreCrystalConfig::default().vein_octaves },
                vein_lacunarity: if c.crystal_kimberlite_vein_lacunarity > 0.0 { c.crystal_kimberlite_vein_lacunarity } else { OreCrystalConfig::default().vein_lacunarity },
                vein_warp_strength: c.crystal_kimberlite_vein_warp_strength,
                vein_density: if c.crystal_kimberlite_vein_density > 0.0 { c.crystal_kimberlite_vein_density } else { OreCrystalConfig::default().vein_density },
            },
            sulfide: OreCrystalConfig {
                enabled: c.crystal_sulfide_enabled != 0,
                chance: if c.crystal_sulfide_chance > 0.0 { c.crystal_sulfide_chance } else { OreCrystalConfig::default().chance },
                density_threshold: c.crystal_sulfide_density_threshold,
                scale_min: if c.crystal_sulfide_scale_min > 0.0 { c.crystal_sulfide_scale_min } else { OreCrystalConfig::default().scale_min },
                scale_max: if c.crystal_sulfide_scale_max > 0.0 { c.crystal_sulfide_scale_max } else { OreCrystalConfig::default().scale_max },
                small_weight: if c.crystal_sulfide_small_weight > 0.0 { c.crystal_sulfide_small_weight } else { OreCrystalConfig::default().small_weight },
                medium_weight: if c.crystal_sulfide_medium_weight > 0.0 { c.crystal_sulfide_medium_weight } else { OreCrystalConfig::default().medium_weight },
                large_weight: if c.crystal_sulfide_large_weight > 0.0 { c.crystal_sulfide_large_weight } else { OreCrystalConfig::default().large_weight },
                normal_alignment: c.crystal_sulfide_normal_alignment,
                cluster_size: if c.crystal_sulfide_cluster_size > 0 { c.crystal_sulfide_cluster_size } else { OreCrystalConfig::default().cluster_size },
                cluster_radius: if c.crystal_sulfide_cluster_radius > 0.0 { c.crystal_sulfide_cluster_radius } else { OreCrystalConfig::default().cluster_radius },
                surface_offset: c.crystal_sulfide_surface_offset,
                vein_enabled: c.crystal_sulfide_vein_enabled != 0,
                vein_frequency: if c.crystal_sulfide_vein_frequency > 0.0 { c.crystal_sulfide_vein_frequency } else { OreCrystalConfig::default().vein_frequency },
                vein_thickness: if c.crystal_sulfide_vein_thickness > 0.0 { c.crystal_sulfide_vein_thickness } else { OreCrystalConfig::default().vein_thickness },
                vein_octaves: if c.crystal_sulfide_vein_octaves > 0 { c.crystal_sulfide_vein_octaves } else { OreCrystalConfig::default().vein_octaves },
                vein_lacunarity: if c.crystal_sulfide_vein_lacunarity > 0.0 { c.crystal_sulfide_vein_lacunarity } else { OreCrystalConfig::default().vein_lacunarity },
                vein_warp_strength: c.crystal_sulfide_vein_warp_strength,
                vein_density: if c.crystal_sulfide_vein_density > 0.0 { c.crystal_sulfide_vein_density } else { OreCrystalConfig::default().vein_density },
            },
            quartz: OreCrystalConfig {
                enabled: c.crystal_quartz_enabled != 0,
                chance: if c.crystal_quartz_chance > 0.0 { c.crystal_quartz_chance } else { OreCrystalConfig::default().chance },
                density_threshold: c.crystal_quartz_density_threshold,
                scale_min: if c.crystal_quartz_scale_min > 0.0 { c.crystal_quartz_scale_min } else { OreCrystalConfig::default().scale_min },
                scale_max: if c.crystal_quartz_scale_max > 0.0 { c.crystal_quartz_scale_max } else { OreCrystalConfig::default().scale_max },
                small_weight: if c.crystal_quartz_small_weight > 0.0 { c.crystal_quartz_small_weight } else { OreCrystalConfig::default().small_weight },
                medium_weight: if c.crystal_quartz_medium_weight > 0.0 { c.crystal_quartz_medium_weight } else { OreCrystalConfig::default().medium_weight },
                large_weight: if c.crystal_quartz_large_weight > 0.0 { c.crystal_quartz_large_weight } else { OreCrystalConfig::default().large_weight },
                normal_alignment: c.crystal_quartz_normal_alignment,
                cluster_size: if c.crystal_quartz_cluster_size > 0 { c.crystal_quartz_cluster_size } else { OreCrystalConfig::default().cluster_size },
                cluster_radius: if c.crystal_quartz_cluster_radius > 0.0 { c.crystal_quartz_cluster_radius } else { OreCrystalConfig::default().cluster_radius },
                surface_offset: c.crystal_quartz_surface_offset,
                vein_enabled: c.crystal_quartz_vein_enabled != 0,
                vein_frequency: if c.crystal_quartz_vein_frequency > 0.0 { c.crystal_quartz_vein_frequency } else { OreCrystalConfig::default().vein_frequency },
                vein_thickness: if c.crystal_quartz_vein_thickness > 0.0 { c.crystal_quartz_vein_thickness } else { OreCrystalConfig::default().vein_thickness },
                vein_octaves: if c.crystal_quartz_vein_octaves > 0 { c.crystal_quartz_vein_octaves } else { OreCrystalConfig::default().vein_octaves },
                vein_lacunarity: if c.crystal_quartz_vein_lacunarity > 0.0 { c.crystal_quartz_vein_lacunarity } else { OreCrystalConfig::default().vein_lacunarity },
                vein_warp_strength: c.crystal_quartz_vein_warp_strength,
                vein_density: if c.crystal_quartz_vein_density > 0.0 { c.crystal_quartz_vein_density } else { OreCrystalConfig::default().vein_density },
            },
            pyrite: OreCrystalConfig {
                enabled: c.crystal_pyrite_enabled != 0,
                chance: if c.crystal_pyrite_chance > 0.0 { c.crystal_pyrite_chance } else { OreCrystalConfig::default().chance },
                density_threshold: c.crystal_pyrite_density_threshold,
                scale_min: if c.crystal_pyrite_scale_min > 0.0 { c.crystal_pyrite_scale_min } else { OreCrystalConfig::default().scale_min },
                scale_max: if c.crystal_pyrite_scale_max > 0.0 { c.crystal_pyrite_scale_max } else { OreCrystalConfig::default().scale_max },
                small_weight: if c.crystal_pyrite_small_weight > 0.0 { c.crystal_pyrite_small_weight } else { OreCrystalConfig::default().small_weight },
                medium_weight: if c.crystal_pyrite_medium_weight > 0.0 { c.crystal_pyrite_medium_weight } else { OreCrystalConfig::default().medium_weight },
                large_weight: if c.crystal_pyrite_large_weight > 0.0 { c.crystal_pyrite_large_weight } else { OreCrystalConfig::default().large_weight },
                normal_alignment: c.crystal_pyrite_normal_alignment,
                cluster_size: if c.crystal_pyrite_cluster_size > 0 { c.crystal_pyrite_cluster_size } else { OreCrystalConfig::default().cluster_size },
                cluster_radius: if c.crystal_pyrite_cluster_radius > 0.0 { c.crystal_pyrite_cluster_radius } else { OreCrystalConfig::default().cluster_radius },
                surface_offset: c.crystal_pyrite_surface_offset,
                vein_enabled: c.crystal_pyrite_vein_enabled != 0,
                vein_frequency: if c.crystal_pyrite_vein_frequency > 0.0 { c.crystal_pyrite_vein_frequency } else { OreCrystalConfig::default().vein_frequency },
                vein_thickness: if c.crystal_pyrite_vein_thickness > 0.0 { c.crystal_pyrite_vein_thickness } else { OreCrystalConfig::default().vein_thickness },
                vein_octaves: if c.crystal_pyrite_vein_octaves > 0 { c.crystal_pyrite_vein_octaves } else { OreCrystalConfig::default().vein_octaves },
                vein_lacunarity: if c.crystal_pyrite_vein_lacunarity > 0.0 { c.crystal_pyrite_vein_lacunarity } else { OreCrystalConfig::default().vein_lacunarity },
                vein_warp_strength: c.crystal_pyrite_vein_warp_strength,
                vein_density: if c.crystal_pyrite_vein_density > 0.0 { c.crystal_pyrite_vein_density } else { OreCrystalConfig::default().vein_density },
            },
            amethyst: OreCrystalConfig {
                enabled: c.crystal_amethyst_enabled != 0,
                chance: if c.crystal_amethyst_chance > 0.0 { c.crystal_amethyst_chance } else { OreCrystalConfig::default().chance },
                density_threshold: c.crystal_amethyst_density_threshold,
                scale_min: if c.crystal_amethyst_scale_min > 0.0 { c.crystal_amethyst_scale_min } else { OreCrystalConfig::default().scale_min },
                scale_max: if c.crystal_amethyst_scale_max > 0.0 { c.crystal_amethyst_scale_max } else { OreCrystalConfig::default().scale_max },
                small_weight: if c.crystal_amethyst_small_weight > 0.0 { c.crystal_amethyst_small_weight } else { OreCrystalConfig::default().small_weight },
                medium_weight: if c.crystal_amethyst_medium_weight > 0.0 { c.crystal_amethyst_medium_weight } else { OreCrystalConfig::default().medium_weight },
                large_weight: if c.crystal_amethyst_large_weight > 0.0 { c.crystal_amethyst_large_weight } else { OreCrystalConfig::default().large_weight },
                normal_alignment: c.crystal_amethyst_normal_alignment,
                cluster_size: if c.crystal_amethyst_cluster_size > 0 { c.crystal_amethyst_cluster_size } else { OreCrystalConfig::default().cluster_size },
                cluster_radius: if c.crystal_amethyst_cluster_radius > 0.0 { c.crystal_amethyst_cluster_radius } else { OreCrystalConfig::default().cluster_radius },
                surface_offset: c.crystal_amethyst_surface_offset,
                vein_enabled: c.crystal_amethyst_vein_enabled != 0,
                vein_frequency: if c.crystal_amethyst_vein_frequency > 0.0 { c.crystal_amethyst_vein_frequency } else { OreCrystalConfig::default().vein_frequency },
                vein_thickness: if c.crystal_amethyst_vein_thickness > 0.0 { c.crystal_amethyst_vein_thickness } else { OreCrystalConfig::default().vein_thickness },
                vein_octaves: if c.crystal_amethyst_vein_octaves > 0 { c.crystal_amethyst_vein_octaves } else { OreCrystalConfig::default().vein_octaves },
                vein_lacunarity: if c.crystal_amethyst_vein_lacunarity > 0.0 { c.crystal_amethyst_vein_lacunarity } else { OreCrystalConfig::default().vein_lacunarity },
                vein_warp_strength: c.crystal_amethyst_vein_warp_strength,
                vein_density: if c.crystal_amethyst_vein_density > 0.0 { c.crystal_amethyst_vein_density } else { OreCrystalConfig::default().vein_density },
            },
            coal: OreCrystalConfig {
                enabled: c.crystal_coal_enabled != 0,
                chance: if c.crystal_coal_chance > 0.0 { c.crystal_coal_chance } else { OreCrystalConfig::default().chance },
                density_threshold: c.crystal_coal_density_threshold,
                scale_min: if c.crystal_coal_scale_min > 0.0 { c.crystal_coal_scale_min } else { OreCrystalConfig::default().scale_min },
                scale_max: if c.crystal_coal_scale_max > 0.0 { c.crystal_coal_scale_max } else { OreCrystalConfig::default().scale_max },
                small_weight: if c.crystal_coal_small_weight > 0.0 { c.crystal_coal_small_weight } else { OreCrystalConfig::default().small_weight },
                medium_weight: if c.crystal_coal_medium_weight > 0.0 { c.crystal_coal_medium_weight } else { OreCrystalConfig::default().medium_weight },
                large_weight: if c.crystal_coal_large_weight > 0.0 { c.crystal_coal_large_weight } else { OreCrystalConfig::default().large_weight },
                normal_alignment: c.crystal_coal_normal_alignment,
                cluster_size: if c.crystal_coal_cluster_size > 0 { c.crystal_coal_cluster_size } else { OreCrystalConfig::default().cluster_size },
                cluster_radius: if c.crystal_coal_cluster_radius > 0.0 { c.crystal_coal_cluster_radius } else { OreCrystalConfig::default().cluster_radius },
                surface_offset: c.crystal_coal_surface_offset,
                vein_enabled: c.crystal_coal_vein_enabled != 0,
                vein_frequency: if c.crystal_coal_vein_frequency > 0.0 { c.crystal_coal_vein_frequency } else { OreCrystalConfig::default().vein_frequency },
                vein_thickness: if c.crystal_coal_vein_thickness > 0.0 { c.crystal_coal_vein_thickness } else { OreCrystalConfig::default().vein_thickness },
                vein_octaves: if c.crystal_coal_vein_octaves > 0 { c.crystal_coal_vein_octaves } else { OreCrystalConfig::default().vein_octaves },
                vein_lacunarity: if c.crystal_coal_vein_lacunarity > 0.0 { c.crystal_coal_vein_lacunarity } else { OreCrystalConfig::default().vein_lacunarity },
                vein_warp_strength: c.crystal_coal_vein_warp_strength,
                vein_density: if c.crystal_coal_vein_density > 0.0 { c.crystal_coal_vein_density } else { OreCrystalConfig::default().vein_density },
            },
        },
        water_table: voxel_gen::config::WaterTableConfig {
            enabled: c.water_table_enabled != 0,
            base_y: if c.water_table_base_y != 0.0 { c.water_table_base_y } else { 170.0 },
            noise_amplitude: if c.water_table_noise_amplitude != 0.0 { c.water_table_noise_amplitude } else { 15.0 },
            noise_frequency: if c.water_table_noise_frequency > 0.0 { c.water_table_noise_frequency } else { 0.008 },
            spring_flow_rate: if c.water_table_spring_flow_rate > 0.0 { c.water_table_spring_flow_rate } else { 0.8 },
            min_porosity_for_spring: if c.water_table_min_porosity > 0.0 { c.water_table_min_porosity } else { 0.5 },
            drip_noise_frequency: if c.water_table_drip_noise_frequency > 0.0 { c.water_table_drip_noise_frequency } else { 0.15 },
            drip_noise_threshold: if c.water_table_drip_noise_threshold > 0.0 { c.water_table_drip_noise_threshold } else { 0.7 },
            drip_level: if c.water_table_drip_level > 0.0 { c.water_table_drip_level } else { 0.4 },
            max_springs_per_chunk: if c.water_table_max_springs > 0 { c.water_table_max_springs } else { 8 },
            max_drips_per_chunk: if c.water_table_max_drips > 0 { c.water_table_max_drips } else { 12 },
        },
        pipe_lava: voxel_gen::config::PipeLavaConfig {
            enabled: c.pipe_lava_enabled != 0,
            activation_depth: if c.pipe_lava_activation_depth != 0.0 { c.pipe_lava_activation_depth } else { -80.0 },
            max_lava_per_chunk: if c.pipe_lava_max_per_chunk > 0 { c.pipe_lava_max_per_chunk } else { 6 },
            depth_scaling: if c.pipe_lava_depth_scaling > 0.0 { c.pipe_lava_depth_scaling } else { 0.5 },
        },
        lava_tubes: voxel_gen::config::LavaTubeConfig {
            enabled: c.lava_tube_enabled != 0,
            tubes_per_region: if c.lava_tube_tubes_per_region > 0.0 { c.lava_tube_tubes_per_region } else { 2.0 },
            depth_min: if c.lava_tube_depth_min != 0.0 { c.lava_tube_depth_min } else { -250.0 },
            depth_max: if c.lava_tube_depth_max != 0.0 { c.lava_tube_depth_max } else { -50.0 },
            radius_min: if c.lava_tube_radius_min > 0.0 { c.lava_tube_radius_min } else { 2.0 },
            radius_max: if c.lava_tube_radius_max > 0.0 { c.lava_tube_radius_max } else { 4.0 },
            max_steps: if c.lava_tube_max_steps > 0 { c.lava_tube_max_steps } else { 150 },
            step_length: if c.lava_tube_step_length > 0.0 { c.lava_tube_step_length } else { 1.5 },
            active_depth: if c.lava_tube_active_depth != 0.0 { c.lava_tube_active_depth } else { -120.0 },
            pipe_connection_radius: if c.lava_tube_pipe_connection_radius > 0.0 { c.lava_tube_pipe_connection_radius } else { 20.0 },
        },
        hydrothermal: voxel_gen::config::HydrothermalConfig {
            enabled: c.hydrothermal_enabled != 0,
            radius: if c.hydrothermal_radius > 0 { c.hydrothermal_radius } else { 8 },
            max_per_chunk: if c.hydrothermal_max_per_chunk > 0 { c.hydrothermal_max_per_chunk } else { 4 },
        },
        rivers: voxel_gen::config::RiverConfig {
            enabled: c.river_enabled != 0,
            rivers_per_region: if c.river_rivers_per_region > 0.0 { c.river_rivers_per_region } else { 1.0 },
            width_min: if c.river_width_min > 0.0 { c.river_width_min } else { 3.0 },
            width_max: if c.river_width_max > 0.0 { c.river_width_max } else { 6.0 },
            height: if c.river_height > 0.0 { c.river_height } else { 2.5 },
            max_steps: if c.river_max_steps > 0 { c.river_max_steps } else { 300 },
            step_length: if c.river_step_length > 0.0 { c.river_step_length } else { 1.5 },
            layer_restriction: c.river_layer_restriction != 0,
            downslope_bias: if c.river_downslope_bias > 0.0 { c.river_downslope_bias } else { 0.02 },
        },
        artesian: voxel_gen::config::ArtesianConfig {
            enabled: c.artesian_enabled != 0,
            aquifer_y_center: if c.artesian_aquifer_y_center != 0.0 { c.artesian_aquifer_y_center } else { -15.0 },
            aquifer_thickness: if c.artesian_aquifer_thickness > 0.0 { c.artesian_aquifer_thickness } else { 3.0 },
            aquifer_noise_freq: if c.artesian_aquifer_noise_freq > 0.0 { c.artesian_aquifer_noise_freq } else { 0.01 },
            aquifer_noise_threshold: if c.artesian_aquifer_noise_threshold > 0.0 { c.artesian_aquifer_noise_threshold } else { 0.3 },
            pressure_noise_freq: if c.artesian_pressure_noise_freq > 0.0 { c.artesian_pressure_noise_freq } else { 0.02 },
            max_per_chunk: if c.artesian_max_per_chunk > 0 { c.artesian_max_per_chunk } else { 3 },
        },
        octree_max_depth: 4,
        region_size: if c.region_size == 0 { 3 } else { c.region_size as i32 },
        bounds_size: c.bounds_size,
        mesh_smooth_iterations: c.mesh_smooth_iterations,
        mesh_smooth_strength: if c.mesh_smooth_strength > 0.0 { c.mesh_smooth_strength } else { 0.3 },
        mesh_boundary_smooth: if c.mesh_boundary_smooth > 0.0 { c.mesh_boundary_smooth } else { 0.3 },
        mesh_recalc_normals: c.mesh_recalc_normals,
        ore_detail_multiplier: if c.ore_detail_multiplier > 0 { c.ore_detail_multiplier.min(4) } else { 1 },
        ore_protrusion: c.ore_protrusion.max(0.0).min(0.5),
        fluid_sources_enabled: c.fluid_sources_enabled != 0,
    }
}

/// Debug: log pool config as received from FFI (temporary diagnostic).
fn debug_log_pool_config(c: &FfiEngineConfig) {
    eprintln!("[FFI-POOL] enabled={} freq={} thresh={} chance={} min_area={} max_radius={} \
              basin_depth={} rim_height={} water={} lava={} empty={} air_above={} \
              max_cave_height={} min_floor_thickness={} min_ground_depth={}",
        c.pool_enabled, c.pool_placement_freq, c.pool_placement_thresh,
        c.pool_chance, c.pool_min_area, c.pool_max_radius,
        c.pool_basin_depth, c.pool_rim_height,
        c.pool_water_pct, c.pool_lava_pct, c.pool_empty_pct, c.pool_min_air_above,
        c.pool_max_cave_height, c.pool_min_floor_thickness, c.pool_min_ground_depth);
}

/// Convert FFI config to FluidConfig.
fn ffi_config_to_fluid(c: &FfiEngineConfig) -> FluidConfig {
    FluidConfig {
        seed: c.seed,
        chunk_size: c.chunk_size as usize,
        tick_rate: if c.fluid_tick_rate > 0.0 { c.fluid_tick_rate } else { 10.0 },
        lava_tick_divisor: if c.fluid_lava_tick_divisor > 0 { c.fluid_lava_tick_divisor } else { 4 },
        water_spring_threshold: if c.fluid_water_spring_threshold > 0.0 { c.fluid_water_spring_threshold } else { 2.0 },
        lava_source_threshold: if c.fluid_lava_source_threshold > 0.0 { c.fluid_lava_source_threshold } else { 0.98 },
        lava_depth_max: if c.fluid_lava_depth_max != 0.0 { c.fluid_lava_depth_max } else { -50.0 },
        water_noise_frequency: if c.fluid_water_noise_frequency > 0.0 { c.fluid_water_noise_frequency } else { 0.05 },
        water_depth_min: if c.fluid_water_depth_min != 0.0 { c.fluid_water_depth_min } else { -9999.0 },
        water_depth_max: if c.fluid_water_depth_max != 0.0 { c.fluid_water_depth_max } else { 9999.0 },
        water_flow_rate: if c.fluid_water_flow_rate > 0.0 { c.fluid_water_flow_rate } else { 1.0 },
        water_spread_rate: if c.fluid_water_spread_rate > 0.0 { c.fluid_water_spread_rate } else { 0.3 },
        lava_noise_frequency: if c.fluid_lava_noise_frequency > 0.0 { c.fluid_lava_noise_frequency } else { 0.03 },
        lava_depth_min: if c.fluid_lava_depth_min != 0.0 { c.fluid_lava_depth_min } else { -9999.0 },
        lava_flow_rate: if c.fluid_lava_flow_rate > 0.0 { c.fluid_lava_flow_rate } else { 0.1 },
        lava_spread_rate: if c.fluid_lava_spread_rate > 0.0 { c.fluid_lava_spread_rate } else { 0.125 },
        cavern_source_bias: c.fluid_cavern_source_bias,
        tunnel_bend_threshold: c.fluid_tunnel_bend_threshold,
        flow_anim_speed: 1.0,
        solid_threshold: 0.0,
        solid_corner_threshold: if c.fluid_solid_corner_threshold > 0 { c.fluid_solid_corner_threshold } else { 6 },
        mesh_smooth_iterations: 2,
        mesh_smooth_strength: 0.3,
        mesh_qef_refinement: true,
        mesh_recalc_normals: true,
    }
}

/// Convert FFI scan config to internal ScanConfig.
pub fn ffi_scan_config_to_scan_config(c: &FfiScanConfig) -> ScanConfig {
    ScanConfig {
        enable_density_seam: c.enable_density_seam != 0,
        enable_mesh_topology: c.enable_mesh_topology != 0,
        enable_seam_completeness: c.enable_seam_completeness != 0,
        enable_navigability: c.enable_navigability != 0,
        enable_worm_truncation: c.enable_worm_truncation != 0,
        enable_thin_walls: c.enable_thin_walls != 0,
        enable_winding_consistency: c.enable_winding_consistency != 0,
        enable_degenerate_triangles: c.enable_degenerate_triangles != 0,
        enable_worm_carve_verify: c.enable_worm_carve_verify != 0,
        enable_self_intersection: c.enable_self_intersection != 0,
        enable_seam_mesh_quality: c.enable_seam_mesh_quality != 0,
        density_subsample_count: c.density_subsample_count,
        raymarch_rays_per_chunk: c.raymarch_rays_per_chunk,
        raymarch_step_size: c.raymarch_step_size,
        max_vertex_zero_crossing_dist: c.max_vertex_zero_crossing_dist,
        min_passage_width: c.min_passage_width,
        min_triangle_area: c.min_triangle_area,
        max_edge_length: c.max_edge_length,
        thin_wall_max_thickness: c.thin_wall_max_thickness,
        self_intersection_tri_limit: c.self_intersection_tri_limit,
    }
}

/// Convert FFI config to SleepConfig.
pub fn ffi_config_to_sleep(c: &FfiEngineConfig) -> voxel_sleep::SleepConfig {
    use voxel_sleep::config::{CollapseConfig, DeepTimeConfig, GroundwaterConfig, MetamorphismConfig, MineralConfig, ReactionConfig, AureoleConfig, VeinConfig};
    // Build collapse config from new FFI fields (fall back to legacy fields if new are zero)
    let new_collapse = CollapseConfig {
        strut_survival: if c.sleep_strut_survival[1..].iter().any(|&v| v > 0.0) {
            c.sleep_strut_survival
        } else {
            CollapseConfig::default().strut_survival
        },
        stress_multiplier: if c.sleep_new_stress_multiplier > 0.0 { c.sleep_new_stress_multiplier }
            else if c.sleep_stress_multiplier > 0.0 { c.sleep_stress_multiplier } else { 1.5 },
        max_cascade_iterations: 8, // not exposed in new UI
        rubble_fill_ratio: if c.sleep_new_rubble_fill_ratio > 0.0 { c.sleep_new_rubble_fill_ratio }
            else if c.sleep_rubble_fill_ratio > 0.0 { c.sleep_rubble_fill_ratio } else { 0.40 },
        min_stress_for_cascade: if c.sleep_new_min_stress_cascade > 0.0 { c.sleep_new_min_stress_cascade }
            else if c.sleep_min_stress_for_cascade > 0.0 { c.sleep_min_stress_for_cascade } else { 0.7 },
        rubble_material_match: true,
        support_stress_penalty: if c.sleep_support_stress_penalty > 0.0 { c.sleep_support_stress_penalty } else { 1.0 },
        collapse_enabled: c.sleep_new_collapse_enabled != 0,
    };
    // Also build legacy collapse for backward compat
    let legacy_collapse = CollapseConfig {
        strut_survival: if c.sleep_strut_survival[1..].iter().any(|&v| v > 0.0) {
            c.sleep_strut_survival
        } else {
            CollapseConfig::default().strut_survival
        },
        stress_multiplier: if c.sleep_stress_multiplier > 0.0 { c.sleep_stress_multiplier } else { 1.5 },
        max_cascade_iterations: if c.sleep_max_cascade_iterations > 0 { c.sleep_max_cascade_iterations } else { 8 },
        rubble_fill_ratio: if c.sleep_rubble_fill_ratio > 0.0 { c.sleep_rubble_fill_ratio } else { 0.40 },
        min_stress_for_cascade: if c.sleep_min_stress_for_cascade > 0.0 { c.sleep_min_stress_for_cascade } else { 0.7 },
        rubble_material_match: c.sleep_rubble_material_match != 0,
        support_stress_penalty: if c.sleep_support_stress_penalty > 0.0 { c.sleep_support_stress_penalty } else { 1.0 },
        collapse_enabled: c.sleep_collapse_sub_enabled != 0,
    };
    voxel_sleep::SleepConfig {
        time_budget_ms: if c.sleep_time_budget_ms > 0 { c.sleep_time_budget_ms } else { 8000 },
        chunk_radius: c.sleep_chunk_radius.min(10),
        sleep_count: if c.sleep_count > 0 { c.sleep_count } else { 1 },
        nest_positions: Vec::new(),
        // New 4-phase system — now mapped from FFI fields
        phase1_enabled: c.sleep_phase1_enabled != 0,
        phase2_enabled: c.sleep_phase2_enabled != 0,
        phase3_enabled: c.sleep_phase3_enabled != 0,
        phase4_enabled: c.sleep_phase4_enabled != 0,
        groundwater: GroundwaterConfig {
            enabled: c.sleep_groundwater_enabled != 0,
            strength: if c.sleep_groundwater_strength > 0.0 { c.sleep_groundwater_strength } else { 0.3 },
            depth_baseline: 0.0,
            depth_scale: if c.sleep_groundwater_depth_scale > 0.0 { c.sleep_groundwater_depth_scale } else { 0.02 },
            drip_zone_multiplier: if c.sleep_groundwater_drip_multiplier > 0.0 { c.sleep_groundwater_drip_multiplier } else { 2.0 },
            erosion_power: if c.sleep_gw_erosion_power > 0.0 { c.sleep_gw_erosion_power } else { 1.0 },
            flowstone_power: if c.sleep_gw_flowstone_power > 0.0 { c.sleep_gw_flowstone_power } else { 1.0 },
            enrichment_power: if c.sleep_gw_enrichment_power > 0.0 { c.sleep_gw_enrichment_power } else { 1.0 },
            soft_rock_mult: if c.sleep_gw_soft_rock_mult > 0.0 { c.sleep_gw_soft_rock_mult } else { 1.0 },
            hard_rock_mult: if c.sleep_gw_hard_rock_mult > 0.0 { c.sleep_gw_hard_rock_mult } else { 0.15 },
            ..Default::default()
        },
        reaction: ReactionConfig {
            acid_dissolution_prob: if c.sleep_acid_dissolution_prob > 0.0 { c.sleep_acid_dissolution_prob } else { 0.60 },
            copper_oxidation_prob: if c.sleep_copper_oxidation_prob > 0.0 { c.sleep_copper_oxidation_prob } else { 0.50 },
            basalt_crust_prob: if c.sleep_basalt_crust_prob > 0.0 { c.sleep_basalt_crust_prob } else { 0.70 },
            ..Default::default()
        },
        aureole: AureoleConfig {
            aureole_radius: if c.sleep_aureole_radius > 0 { c.sleep_aureole_radius } else { 8 },
            contact_limestone_to_marble_prob: if c.sleep_contact_marble_prob > 0.0 { c.sleep_contact_marble_prob } else { 0.80 },
            water_erosion_prob: if c.sleep_water_erosion_prob > 0.0 { c.sleep_water_erosion_prob } else { 0.05 },
            water_erosion_enabled: c.sleep_water_erosion_enabled != 0,
            ..Default::default()
        },
        veins: VeinConfig {
            vein_deposition_prob: if c.sleep_vein_deposition_prob > 0.0 { c.sleep_vein_deposition_prob } else { 0.25 },
            vein_max_distance: if c.sleep_vein_max_distance > 0 { c.sleep_vein_max_distance } else { 16 },
            max_vein_voxels_per_source: if c.sleep_vein_max_per_source > 0 { c.sleep_vein_max_per_source } else { 12 },
            flowstone_prob: if c.sleep_flowstone_prob > 0.0 { c.sleep_flowstone_prob } else { 0.10 },
            ..Default::default()
        },
        deeptime: DeepTimeConfig {
            enrichment_prob: if c.sleep_enrichment_prob > 0.0 { c.sleep_enrichment_prob } else { 0.15 },
            vein_thickening_prob: if c.sleep_vein_thickening_prob > 0.0 { c.sleep_vein_thickening_prob } else { 0.10 },
            stalactite_growth_prob: if c.sleep_stalactite_growth_prob > 0.0 { c.sleep_stalactite_growth_prob } else { 0.10 },
            collapse: new_collapse,
            ..Default::default()
        },
        // Legacy fields (kept for backward compat, old FFI fields still map here)
        metamorphism_enabled: c.sleep_metamorphism_enabled != 0,
        minerals_enabled: c.sleep_minerals_enabled != 0,
        collapse_enabled: c.sleep_collapse_enabled != 0,
        metamorphism: MetamorphismConfig {
            limestone_to_marble_prob: if c.sleep_limestone_to_marble_prob > 0.0 { c.sleep_limestone_to_marble_prob } else { 0.40 },
            limestone_to_marble_depth: if c.sleep_limestone_to_marble_depth != 0.0 { c.sleep_limestone_to_marble_depth } else { -50.0 },
            limestone_to_marble_enabled: c.sleep_limestone_to_marble_enabled != 0,
            sandstone_to_granite_prob: if c.sleep_sandstone_to_granite_prob > 0.0 { c.sleep_sandstone_to_granite_prob } else { 0.25 },
            sandstone_to_granite_depth: if c.sleep_sandstone_to_granite_depth != 0.0 { c.sleep_sandstone_to_granite_depth } else { -100.0 },
            sandstone_to_granite_min_neighbors: if c.sleep_sandstone_to_granite_min_neighbors > 0 { c.sleep_sandstone_to_granite_min_neighbors } else { 4 },
            sandstone_to_granite_enabled: c.sleep_sandstone_to_granite_enabled != 0,
            slate_to_marble_prob: if c.sleep_slate_to_marble_prob > 0.0 { c.sleep_slate_to_marble_prob } else { 0.60 },
            slate_to_marble_enabled: c.sleep_slate_to_marble_enabled != 0,
            granite_to_basalt_prob: if c.sleep_granite_to_basalt_prob > 0.0 { c.sleep_granite_to_basalt_prob } else { 0.15 },
            granite_to_basalt_min_air: if c.sleep_granite_to_basalt_min_air > 0 { c.sleep_granite_to_basalt_min_air } else { 2 },
            granite_to_basalt_enabled: c.sleep_granite_to_basalt_enabled != 0,
            iron_to_pyrite_prob: if c.sleep_iron_to_pyrite_prob > 0.0 { c.sleep_iron_to_pyrite_prob } else { 0.35 },
            iron_to_pyrite_search_radius: if c.sleep_iron_to_pyrite_search_radius > 0 { c.sleep_iron_to_pyrite_search_radius } else { 2 },
            iron_to_pyrite_enabled: c.sleep_iron_to_pyrite_enabled != 0,
            copper_to_malachite_prob: if c.sleep_copper_to_malachite_prob > 0.0 { c.sleep_copper_to_malachite_prob } else { 0.50 },
            copper_to_malachite_enabled: c.sleep_copper_to_malachite_enabled != 0,
        },
        minerals: MineralConfig {
            crystal_growth_max: if c.sleep_crystal_growth_max > 0 { c.sleep_crystal_growth_max } else { 2 },
            crystal_growth_enabled: c.sleep_crystal_growth_enabled != 0,
            crystal_growth_prob: if c.sleep_crystal_growth_prob > 0.0 { c.sleep_crystal_growth_prob } else { 0.3 },
            malachite_stalactite_max: if c.sleep_malachite_stalactite_max > 0 { c.sleep_malachite_stalactite_max } else { 1 },
            malachite_stalactite_enabled: c.sleep_malachite_stalactite_enabled != 0,
            malachite_stalactite_prob: if c.sleep_malachite_stalactite_prob > 0.0 { c.sleep_malachite_stalactite_prob } else { 0.2 },
            quartz_extension_prob: if c.sleep_quartz_extension_prob > 0.0 { c.sleep_quartz_extension_prob } else { 0.10 },
            quartz_extension_max: if c.sleep_quartz_extension_max > 0 { c.sleep_quartz_extension_max } else { 1 },
            quartz_extension_enabled: c.sleep_quartz_extension_enabled != 0,
            calcite_infill_max: if c.sleep_calcite_infill_max > 0 { c.sleep_calcite_infill_max } else { 1 },
            calcite_infill_depth: if c.sleep_calcite_infill_depth != 0.0 { c.sleep_calcite_infill_depth } else { -30.0 },
            calcite_infill_min_faces: if c.sleep_calcite_infill_min_faces > 0 { c.sleep_calcite_infill_min_faces } else { 3 },
            calcite_infill_enabled: c.sleep_calcite_infill_enabled != 0,
            calcite_infill_prob: if c.sleep_calcite_infill_prob > 0.0 { c.sleep_calcite_infill_prob } else { 0.15 },
            pyrite_crust_max: if c.sleep_pyrite_crust_max > 0 { c.sleep_pyrite_crust_max } else { 1 },
            pyrite_crust_min_solid: if c.sleep_pyrite_crust_min_solid > 0 { c.sleep_pyrite_crust_min_solid } else { 2 },
            pyrite_crust_enabled: c.sleep_pyrite_crust_enabled != 0,
            pyrite_crust_prob: if c.sleep_pyrite_crust_prob > 0.0 { c.sleep_pyrite_crust_prob } else { 0.1 },
            growth_density_min: if c.sleep_growth_density_min > 0.0 { c.sleep_growth_density_min } else { 0.3 },
            growth_density_max: if c.sleep_growth_density_max > 0.0 { c.sleep_growth_density_max } else { 0.6 },
        },
        collapse: legacy_collapse,
        stress: voxel_core::stress::StressConfig::default(),
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
    let mut flow_directions = Vec::with_capacity(mesh.flow_directions.len());

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

    // Flow directions: (dx, dz, magnitude) — transform horizontal components
    // Rust (dx, dz) → UE (dx, -dz) to match the Y→Z axis swap
    for f in &mesh.flow_directions {
        flow_directions.push(FfiVec3 {
            x: f[0],       // dx unchanged
            y: -f[1],      // dz negated (Rust Z → UE -Y)
            z: f[2],       // magnitude
        });
    }

    ConvertedFluidMesh {
        positions,
        normals,
        fluid_types: mesh.fluid_types.clone(),
        indices: mesh.indices.clone(),
        uvs: mesh.uvs.clone(),
        flow_directions,
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}
