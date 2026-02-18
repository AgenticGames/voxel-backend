use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::thread::{self, JoinHandle};

use crossbeam_channel::{bounded, Receiver, Sender};
use dashmap::DashMap;
use voxel_fluid::FluidConfig;
use voxel_fluid::FluidEvent;
use voxel_gen::config::{
    BandedIronConfig, GenerationConfig, GeodeConfig, HostRockConfig, KimberlitePipeConfig,
    NoiseConfig, OreConfig, OreVeinParams, SulfideBlobConfig, WormConfig,
};

use crate::convert::ue_chunk_to_rust;
use crate::store::ChunkStore;
use crate::types::*;
use crate::worker::worker_loop;

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
    generation_counters: Arc<DashMap<(i32, i32, i32), AtomicU64>>,
    shutdown: Arc<AtomicBool>,

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
            generation_counters,
            shutdown,
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
    pub fn poll_result(&self) -> Option<WorkerResult> {
        self.result_rx.try_recv().ok()
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

        let chunk_size = self.config.read().map(|c| c.chunk_size).unwrap_or(16);
        let cs = chunk_size as f32;

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

        let chunk_size = self.config.read().map(|c| c.chunk_size).unwrap_or(16);
        let rust_pos = from_ue_world_pos(ue_x, ue_y, ue_z, world_scale);

        let store = self.store.read().ok()?;
        let best = store.find_spring_location(rust_pos, chunk_size)?;

        // Convert Rust pos back to UE: (x * scale, -z * scale, y * scale)
        Some((
            best.x * world_scale,
            -best.z * world_scale,
            best.y * world_scale,
        ))
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
        octree_max_depth: 4,
        max_edge_length: c.max_edge_length,
        region_size: if c.region_size == 0 { 3 } else { c.region_size as i32 },
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
