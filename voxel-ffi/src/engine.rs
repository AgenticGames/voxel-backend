use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::thread::{self, JoinHandle};

use crossbeam_channel::{bounded, Receiver, Sender};
use dashmap::DashMap;
use voxel_gen::config::{GenerationConfig, NoiseConfig, OreConfig, WormConfig};

use crate::convert::ue_chunk_to_rust;
use crate::store::ChunkStore;
use crate::types::*;
use crate::worker::worker_loop;

pub struct VoxelEngine {
    // Channels
    generate_tx: Sender<WorkerRequest>,
    mine_tx: Sender<WorkerRequest>,
    result_rx: Receiver<WorkerResult>,

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
        let world_scale = ffi_config.world_scale;
        let num_workers = if ffi_config.worker_threads == 0 {
            num_cpus()
        } else {
            ffi_config.worker_threads as usize
        };

        let (generate_tx, generate_rx) = bounded::<WorkerRequest>(256);
        let (mine_tx, mine_rx) = bounded::<WorkerRequest>(16);
        let (result_tx, result_rx) = bounded::<WorkerResult>(64);

        let store = Arc::new(RwLock::new(ChunkStore::new()));
        let config = Arc::new(RwLock::new(config));
        let generation_counters: Arc<DashMap<(i32, i32, i32), AtomicU64>> =
            Arc::new(DashMap::new());
        let shutdown = Arc::new(AtomicBool::new(false));

        let mut workers = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let shutdown = Arc::clone(&shutdown);
            let generate_rx = generate_rx.clone();
            let mine_rx = mine_rx.clone();
            let result_tx = result_tx.clone();
            let store = Arc::clone(&store);
            let config = Arc::clone(&config);
            let gen_counters = Arc::clone(&generation_counters);

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
                );
            });
            workers.push(handle);
        }

        VoxelEngine {
            generate_tx,
            mine_tx,
            result_rx,
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

    /// Hot-reload configuration (affects future generation requests).
    pub fn update_config(&self, ffi_config: &FfiEngineConfig) {
        let new_config = ffi_config_to_generation(ffi_config);
        if let Ok(mut cfg) = self.config.write() {
            *cfg = new_config;
        }
    }

    /// Gracefully shut down all workers and wait for them to finish.
    pub fn shutdown(self) {
        self.shutdown.store(true, Ordering::Relaxed);
        // Drop senders to unblock recv_timeout
        drop(self.generate_tx);
        drop(self.mine_tx);
        for handle in self.workers {
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
        ore: OreConfig::default(),
        octree_max_depth: 4,
        max_edge_length: c.max_edge_length,
        region_size: if c.region_size == 0 { 3 } else { c.region_size as i32 },
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}
