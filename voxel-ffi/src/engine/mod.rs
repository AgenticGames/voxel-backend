use std::collections::HashSet;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};

use crossbeam_channel::{bounded, Receiver, Sender};
use dashmap::DashMap;
use voxel_fluid::FluidEvent;
use voxel_gen::config::{
    GenerationConfig, StressConfig,
};

use crate::pathing::PathResultStore;
use crate::profiler::StreamingProfiler;
use crate::store::ChunkStore;
use crate::types::*;
use crate::worker::{path_worker_loop, worker_loop};

mod requests;
mod pathing_api;
mod sim_jobs;
mod queries;
mod terrain_ops;
mod brushes;
mod triggers_api;
mod persistence;
mod config_convert;
mod helpers;

pub use config_convert::{
    ffi_config_to_generation, ffi_config_to_sleep, ffi_scan_config_to_scan_config,
};
pub(crate) use config_convert::{debug_log_pool_config, ffi_config_to_fluid};
pub(crate) use helpers::cell_has_solid_center;
pub(crate) use helpers::{aabb_center, aabb_to_ffi, fluid_sim_loop_wrapper, num_cpus, zero_aabb};

/// Stress threshold above which a cell counts as "imminently collapsing" for
/// the warning-FX system (cracks + ambient dust + mining-impact dust burst).
///
/// **Not the same as the collapse-pass threshold.** The collapse pass at
/// `voxel-core::stress::detect_and_execute_collapses_v2` starts at `eff >= 1.0`
/// — the bare overstress threshold. But many cells in the 1.0-1.3 band sit
/// there forever without dropping (edge-of-slab, insufficient region coherence,
/// median landing offset <= 0).
///
/// The V-overlay's color map (`StressToColor` in VoxelChunkActor.cpp) goes
/// red at 1.0 and pure white at 1.5. Players read "white" as "this is going
/// down" and "red" as "stressed but stable." So warning FX fire at the higher
/// "white-bright" threshold, not the bare collapse-pass threshold — keeps the
/// promise that anywhere with visible cracks/dust is genuinely a collapse risk.
pub const COLLAPSE_IMMINENT_STRESS: f32 = 1.5;

/// TTL (seconds) for stashed path results UE never collected. Prunes once per
/// poll to bound memory growth from dead agents.
const PATH_RESULT_TTL_SECS: u64 = 10;

/// Compute terrace size in voxels from world scale, targeting ~80 UU snap steps (2 voxels).
pub(crate) fn terrace_size_for_scale(scale: f32) -> i32 {
    (80.0f32 / scale).round().max(1.0) as i32
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
    pub channels_eroded: u32,
    pub corpses_fossilized: u32,
    pub lava_solidified: u32,
    pub profile_report: String,
    pub aureole_glimpse_pos: Option<(i32, i32, i32)>,
    pub aureole_showcase_block: Option<Vec<(i32, i32, i32)>>,
    pub manifest_json: String,
    pub lava_cells: Vec<(i32, i32, i32)>,
    pub surface_changed_cells: Vec<(i32, i32, i32)>,
    pub surface_step_activity: [u16; voxel_sleep::SURFACE_ACTIVITY_BUCKETS],
}

/// Internal morph step result (Rust side, before FFI conversion).
pub struct MorphStepResult {
    pub step: u32,
    pub total_steps: u32,
    pub meshes: Vec<crate::types::ConvertedMesh>,
}

pub struct VoxelEngine {
    // Channels
    generate_tx: Sender<WorkerRequest>,
    mine_tx: Sender<WorkerRequest>,
    result_rx: Receiver<WorkerResult>,
    /// Priority queue for re-queued results (mine batch expansion)
    priority_results: std::sync::Mutex<std::collections::VecDeque<WorkerResult>>,

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
    // Sleep-montage reveal pause: when true, workers stop pulling chunk-generation
    // requests (mine/morph/sleep still run) so the on-screen morph's rayon mesh-gen
    // gets the full core count instead of fighting the POI gen "storm" for rayon.
    generation_paused: Arc<AtomicBool>,

    // Sleep
    sleep_complete: Arc<Mutex<Option<SleepCompleteData>>>,

    // Morph
    morph_results: Arc<Mutex<std::collections::VecDeque<MorphStepResult>>>,
    morph_manifest: Arc<Mutex<Option<voxel_sleep::ChangeManifest>>>,
    morph_snapshot: Arc<Mutex<crate::worker::MorphSnapshot>>,

    // World Scan
    scan_complete: Arc<Mutex<Option<String>>>,

    // Force Spawn Pool
    force_spawn_complete: Arc<Mutex<Option<String>>>,

    // Profiler
    profiler: Arc<StreamingProfiler>,

    // Worker threads
    workers: Vec<JoinHandle<()>>,

    // ─── Pathfinding ─────────────────────────────────────────
    /// Dedicated channel for path requests so heavy `BrushCavernStamp` /
    /// `Sleep` / etc. on `mine_tx` don't head-of-line block AI path queries.
    path_tx: Sender<WorkerRequest>,
    /// Stash of completed path results, keyed by request_id. Drained by
    /// `voxel_path_poll` and TTL-pruned in `poll_result`.
    path_results: Arc<Mutex<PathResultStore>>,
    /// Stash of broken-strut events. Worker pushes via `poll_result`'s
    /// intercept; UE drains all at once via `voxel_take_struts_broken`
    /// at end-of-frame.
    strut_broken_stash: Arc<Mutex<Vec<crate::types::FfiStrutBroken>>>,
    /// Server-side monotonic request id allocator so callers don't have to.
    next_path_request_id: Arc<AtomicU32>,

    /// Cross-species avoidance layer. Pathing-cell coordinates currently
    /// occupied by AI agents (spiders, wasps, creatures). Path workers read
    /// this when constructing ChunkStoreGrid; cells in the set are treated
    /// as obstacles by A* — so spiders/wasps route AROUND each other instead
    /// of phasing through. UE pushes a fresh snapshot ~10Hz from
    /// `UVoxelWorldSubsystem::Tick` via `voxel_path_set_obstacle_cells`.
    /// The requester's own cell is excluded at grid-construction time so the
    /// agent can start its search from inside the (temporarily marked-occupied)
    /// cell it currently stands in.
    /// Lock holds are O(1) on both sides: the writer swaps in a freshly built
    /// `Arc` (see `voxel_path_set_obstacle_cells`), path workers clone the
    /// `Arc` and solve lock-free — so the UE game thread never waits on an
    /// in-flight A* solve.
    pub occupied_cells: Arc<RwLock<Arc<HashSet<(i32, i32, i32)>>>>,

    // ─── Crystal Anchors (Crystal Growth Bridge feature) ─────
    /// Pending and grown crystal-anchor pairs. Mutex is fine: anchor ops
    /// are rare (player input + once-per-sleep) and never contended.
    pub crystal_anchors: Arc<Mutex<crate::crystal_anchors::CrystalAnchorManager>>,

    // ─── POI Tracker (background thread, continuous scoring) ─────
    /// Long-running scored chunk map — updated by the dedicated tracker
    /// thread every ~2s. UE reads top-K from here at sleep-montage time.
    /// Survives chunk unload (TTL'd at 30 minutes).
    pub poi_tracker: crate::poi_tracker::SharedPoiTracker,

    // ─── World Memory (Block 1: semantically-clustered POIs) ─────
    /// Persistent Scene store + event queue. Replaces the per-chunk
    /// `poi_tracker` over time. Block 1 leaves both running; UE migration
    /// in Block 2 retires the old tracker.
    pub world_memory: Arc<voxel_world_memory::WorldMemory>,

    // ─── Predictive Sleep (Block 1: removes "Time passes…" wait) ─────
    /// Latest cached prediction from the `voxel-sleep-predictor` thread.
    /// `None` before the first prediction lands; overwritten by real
    /// `execute_sleep` results when they arrive.
    pub predict_cache: Arc<RwLock<Option<voxel_sleep::predict::PredictedManifest>>>,
    /// Wake-up signal for the predictor thread. UE pokes this via
    /// `voxel_request_predict_now` when the player approaches a bedroll.
    pub predict_wake_tx: Sender<()>,

    // Scale
    world_scale: f32,
}

impl VoxelEngine {
    /// Read chunk_size and world_scale from the engine.
    pub fn chunk_size(&self) -> usize {
        self.config.read().map(|c| c.chunk_size).unwrap_or(16)
    }

    /// Return a snapshot of the current generation config.
    pub fn config_snapshot(&self) -> GenerationConfig {
        self.config.read().unwrap().clone()
    }

    pub fn get_world_scale(&self) -> f32 {
        self.world_scale
    }

    /// Borrow a clone of the shared `ChunkStore` handle. Used by helpers
    /// outside this module (e.g. cinema_bridge) that need to take a brief
    /// read lock. Cheap (Arc clone).
    pub fn store_arc(&self) -> Arc<RwLock<ChunkStore>> {
        Arc::clone(&self.store)
    }

    pub fn new(ffi_config: &FfiEngineConfig) -> Self {
        // Install panic hook + log file as early as possible. Idempotent
        // across engine creation; first install wins.
        crate::panic_log::install("D:/Unreal Projects/Mithril2026/Saved/voxel_panic.log");

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
        let (path_tx, path_rx) = bounded::<WorkerRequest>(256);
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
        let generation_paused = Arc::new(AtomicBool::new(false));

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
        let morph_manifest: Arc<Mutex<Option<voxel_sleep::ChangeManifest>>> = Arc::new(Mutex::new(None));
        let morph_snapshot: Arc<Mutex<crate::worker::MorphSnapshot>> = Arc::new(Mutex::new(crate::worker::MorphSnapshot::default()));

        // Per-region generation-in-flight mutexes. Prevents 2+ workers from
        // redundantly generating the same region's base_density (wasted CPU).
        // A worker claims a region via the per-region Mutex before slow-path;
        // other workers for the same region block on the mutex, then re-check
        // the fast path once the owner finishes.
        let regions_in_flight: Arc<DashMap<(i32, i32, i32), Arc<Mutex<()>>>> =
            Arc::new(DashMap::new());

        // Crystal Growth Bridge — shared manager; sleep handler in worker_loop
        // grows pending pairs during the geological-time pass.
        let crystal_anchors: Arc<Mutex<crate::crystal_anchors::CrystalAnchorManager>> =
            Arc::new(Mutex::new(crate::crystal_anchors::CrystalAnchorManager::default()));

        // POI tracker — long-running background scorer.
        let poi_tracker = crate::poi_tracker::new_tracker();

        // World Memory — Block 1 Scene store. Wrapped in Arc so the drift
        // thread + all FFI readers share the same handle.
        let world_memory: Arc<voxel_world_memory::WorldMemory> =
            Arc::new(voxel_world_memory::WorldMemory::new());

        // Predictive sleep — cache + wake channel. Cache populated by the
        // dedicated predictor thread; overwritten authoritatively by the
        // real `execute_sleep` in the worker loop when it lands.
        let predict_cache: Arc<RwLock<Option<voxel_sleep::predict::PredictedManifest>>> =
            Arc::new(RwLock::new(None));
        let (predict_wake_tx, predict_wake_rx) =
            crossbeam_channel::bounded::<()>(4);

        // Cross-species avoidance — shared occupancy set the path workers
        // read at grid-construction time. UE pushes fresh snapshots ~10Hz.
        let occupied_cells: Arc<RwLock<Arc<HashSet<(i32, i32, i32)>>>> =
            Arc::new(RwLock::new(Arc::new(HashSet::new())));

        // Per-worker activity heartbeats — one slot per worker. Workers stamp
        // what they're handling; the stall monitor (spawned below) reads them to
        // pinpoint a wedged worker when a priority sleep request never dequeues.
        let heartbeats: Arc<Vec<crate::worker::heartbeat::WorkerHeartbeat>> = Arc::new(
            (0..num_workers)
                .map(|_| crate::worker::heartbeat::WorkerHeartbeat::new())
                .collect(),
        );

        let mut workers = Vec::with_capacity(num_workers);
        // Shared by all workers: freshly generated regions whose VFX stress
        // pre-population is pending. Pushed by handle_generate's slow path,
        // drained when the generate queue idles (see worker_loop).
        let deferred_region_stress = Arc::new(crate::worker::DeferredRegionStress::new());

        for worker_id in 0..num_workers {
            let shutdown = Arc::clone(&shutdown);
            let generation_paused = Arc::clone(&generation_paused);
            let generate_rx = generate_rx.clone();
            let mine_rx = mine_rx.clone();
            let result_tx = result_tx.clone();
            let store = Arc::clone(&store);
            let config = Arc::clone(&config);
            let stress_cfg = Arc::clone(&stress_config);
            let gen_counters = Arc::clone(&generation_counters);
            let fluid_tx = fluid_event_tx.clone();
            let prof = Arc::clone(&profiler);
            let morph_man = Arc::clone(&morph_manifest);
            let morph_snap = Arc::clone(&morph_snapshot);
            let rif = Arc::clone(&regions_in_flight);
            let anchors = Arc::clone(&crystal_anchors);
            let heartbeats = Arc::clone(&heartbeats);
            let deferred_stress = Arc::clone(&deferred_region_stress);

            let builder = thread::Builder::new().name(format!("voxel-worker-{}", worker_id));
            let handle = builder
                .spawn(move || {
                    // Each worker runs `worker_loop` inside `catch_unwind` so a
                    // single .unwrap() panic does not silently kill the thread
                    // (which previously left every queued chunk stuck forever).
                    // We respawn up to MAX_RESPAWNS times — beyond that the
                    // panic is almost certainly a poisoned-lock cascade and
                    // looping faster only spams the log.
                    const MAX_RESPAWNS: u32 = 16;
                    let mut respawn = 0u32;
                    crate::panic_log::worker_started();
                    loop {
                        let outcome = {
                            let shutdown = Arc::clone(&shutdown);
                            let generation_paused = Arc::clone(&generation_paused);
                            let generate_rx = generate_rx.clone();
                            let mine_rx = mine_rx.clone();
                            let result_tx = result_tx.clone();
                            let store = Arc::clone(&store);
                            let config = Arc::clone(&config);
                            let stress_cfg = Arc::clone(&stress_cfg);
                            let gen_counters = Arc::clone(&gen_counters);
                            let fluid_tx = fluid_tx.clone();
                            let prof = Arc::clone(&prof);
                            let morph_man = Arc::clone(&morph_man);
                            let morph_snap = Arc::clone(&morph_snap);
                            let rif = Arc::clone(&rif);
                            let anchors = Arc::clone(&anchors);
                            let heartbeats = Arc::clone(&heartbeats);
                            let deferred_stress = Arc::clone(&deferred_stress);
                            std::panic::catch_unwind(AssertUnwindSafe(move || {
                                worker_loop(
                                    shutdown,
                                    generation_paused,
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
                                    morph_man,
                                    morph_snap,
                                    rif,
                                    anchors,
                                    heartbeats,
                                    deferred_stress,
                                );
                            }))
                        };

                        if shutdown.load(Ordering::Relaxed) {
                            break;
                        }
                        match outcome {
                            Ok(()) => break, // normal exit (shutdown / channel closed)
                            Err(payload) => {
                                respawn += 1;
                                let msg = crate::panic_log::payload_string(&*payload);
                                crate::panic_log::note(&format!(
                                    "worker {} caught panic (respawn {}/{}): {}",
                                    worker_id, respawn, MAX_RESPAWNS, msg
                                ));
                                if respawn >= MAX_RESPAWNS {
                                    crate::panic_log::note(&format!(
                                        "worker {} GIVING UP after {} respawns — pool degraded",
                                        worker_id, respawn
                                    ));
                                    break;
                                }
                                std::thread::sleep(std::time::Duration::from_millis(100));
                            }
                        }
                    }
                    crate::panic_log::worker_exited();
                })
                .expect("failed to spawn voxel worker thread");
            workers.push(handle);
        }

        // Stall monitor — watches the heartbeats + mine/generate queue depths and
        // dumps a `[WORKER_STALL]` snapshot to voxel_panic.log when a priority
        // (sleep/mine) request starves or a worker wedges. `len()`-only receiver
        // clones, so it never steals work. Silent until something actually hangs.
        crate::worker::heartbeat::spawn_stall_monitor(
            Arc::clone(&shutdown),
            Arc::clone(&heartbeats),
            mine_rx.clone(),
            generate_rx.clone(),
        );

        // ─── Path workers — multiple dedicated threads share `path_rx` (crossbeam
        // receivers are clonable; each spawned thread holds its own handle and
        // races to receive each request). A* is pure (no shared mutable state),
        // and the path workers take only short read locks on the ChunkStore so
        // contention stays light. With NUM_PATH_WORKERS=3 we comfortably handle
        // demo-density bursts (~50 enemies all requesting at wave start) where
        // a single-threaded worker showed visible tail-latency at ~1-3s for the
        // last few requests. Each worker uses the same catch_unwind respawn
        // pattern as the main pool.
        const NUM_PATH_WORKERS: u32 = 3;
        for path_worker_id in 0..NUM_PATH_WORKERS {
            let shutdown = Arc::clone(&shutdown);
            let path_rx = path_rx.clone();
            let result_tx = result_tx.clone();
            let store = Arc::clone(&store);
            let config_arc = Arc::clone(&config);
            let occupied_arc = Arc::clone(&occupied_cells);
            let world_scale_path = world_scale;
            let builder = thread::Builder::new()
                .name(format!("voxel-path-worker-{}", path_worker_id));
            let handle = builder
                .spawn(move || {
                    const MAX_RESPAWNS: u32 = 16;
                    let mut respawn = 0u32;
                    crate::panic_log::worker_started();
                    loop {
                        let outcome = {
                            let shutdown = Arc::clone(&shutdown);
                            let path_rx = path_rx.clone();
                            let result_tx = result_tx.clone();
                            let store = Arc::clone(&store);
                            let config_arc = Arc::clone(&config_arc);
                            let occupied_arc = Arc::clone(&occupied_arc);
                            std::panic::catch_unwind(AssertUnwindSafe(move || {
                                path_worker_loop(
                                    shutdown,
                                    path_rx,
                                    result_tx,
                                    store,
                                    config_arc,
                                    occupied_arc,
                                    world_scale_path,
                                );
                            }))
                        };
                        if shutdown.load(Ordering::Relaxed) {
                            break;
                        }
                        match outcome {
                            Ok(()) => break,
                            Err(payload) => {
                                respawn += 1;
                                let msg = crate::panic_log::payload_string(&*payload);
                                crate::panic_log::note(&format!(
                                    "path-worker-{} caught panic (respawn {}/{}): {}",
                                    path_worker_id, respawn, MAX_RESPAWNS, msg
                                ));
                                if respawn >= MAX_RESPAWNS {
                                    crate::panic_log::note(&format!(
                                        "path-worker-{} GIVING UP after respawn limit",
                                        path_worker_id
                                    ));
                                    break;
                                }
                                std::thread::sleep(std::time::Duration::from_millis(100));
                            }
                        }
                    }
                    crate::panic_log::worker_exited();
                })
                .expect("failed to spawn voxel path worker thread");
            workers.push(handle);
        }

        // ─── POI Tracker — background thread, scores chunks for the
        // sleep-montage POI rotation. Same catch_unwind respawn pattern.
        {
            let shutdown_t = Arc::clone(&shutdown);
            let store_t = Arc::clone(&store);
            let fluid_tx_t = fluid_event_tx.clone();
            let config_t = Arc::clone(&config);
            let tracker_t = Arc::clone(&poi_tracker);
            let builder = thread::Builder::new().name("voxel-poi-tracker".to_string());
            let handle = builder
                .spawn(move || {
                    const MAX_RESPAWNS: u32 = 8;
                    let mut respawn = 0u32;
                    loop {
                        let outcome = {
                            let shutdown_t = Arc::clone(&shutdown_t);
                            let store_t = Arc::clone(&store_t);
                            let fluid_tx_t = fluid_tx_t.clone();
                            let config_t = Arc::clone(&config_t);
                            let tracker_t = Arc::clone(&tracker_t);
                            std::panic::catch_unwind(AssertUnwindSafe(move || {
                                crate::poi_tracker::poi_tracker_loop(
                                    shutdown_t,
                                    store_t,
                                    fluid_tx_t,
                                    config_t,
                                    tracker_t,
                                );
                            }))
                        };
                        if shutdown_t.load(Ordering::Relaxed) {
                            break;
                        }
                        match outcome {
                            Ok(()) => break,
                            Err(payload) => {
                                respawn += 1;
                                let msg = crate::panic_log::payload_string(&*payload);
                                crate::panic_log::note(&format!(
                                    "poi-tracker caught panic (respawn {}/{}): {}",
                                    respawn, MAX_RESPAWNS, msg
                                ));
                                if respawn >= MAX_RESPAWNS {
                                    crate::panic_log::note(
                                        "poi-tracker GIVING UP — degraded mode (no continuous POI scoring)",
                                    );
                                    break;
                                }
                                std::thread::sleep(std::time::Duration::from_millis(500));
                            }
                        }
                    }
                })
                .expect("failed to spawn voxel POI tracker thread");
            workers.push(handle);
        }

        // ─── World Memory Drift — Block 1 background scanner that
        // produces per-cell-weighted Scene scores. Mirrors POI tracker's
        // catch_unwind respawn pattern.
        {
            let shutdown_t = Arc::clone(&shutdown);
            let store_t = Arc::clone(&store);
            let fluid_tx_t = fluid_event_tx.clone();
            let config_t = Arc::clone(&config);
            let wm_t = Arc::clone(&world_memory);
            let builder = thread::Builder::new().name("voxel-world-memory-drift".to_string());
            let handle = builder
                .spawn(move || {
                    const MAX_RESPAWNS: u32 = 8;
                    let mut respawn = 0u32;
                    loop {
                        let outcome = {
                            let shutdown_t = Arc::clone(&shutdown_t);
                            let store_t = Arc::clone(&store_t);
                            let fluid_tx_t = fluid_tx_t.clone();
                            let config_t = Arc::clone(&config_t);
                            let wm_t = Arc::clone(&wm_t);
                            std::panic::catch_unwind(AssertUnwindSafe(move || {
                                crate::world_memory_drift::world_memory_drift_loop(
                                    shutdown_t,
                                    store_t,
                                    fluid_tx_t,
                                    config_t,
                                    wm_t,
                                );
                            }))
                        };
                        if shutdown_t.load(Ordering::Relaxed) {
                            break;
                        }
                        match outcome {
                            Ok(()) => break,
                            Err(payload) => {
                                respawn += 1;
                                let msg = crate::panic_log::payload_string(&*payload);
                                crate::panic_log::note(&format!(
                                    "world-memory-drift caught panic (respawn {}/{}): {}",
                                    respawn, MAX_RESPAWNS, msg
                                ));
                                if respawn >= MAX_RESPAWNS {
                                    crate::panic_log::note(
                                        "world-memory-drift GIVING UP — Scene store frozen",
                                    );
                                    break;
                                }
                                std::thread::sleep(std::time::Duration::from_millis(500));
                            }
                        }
                    }
                })
                .expect("failed to spawn voxel-world-memory-drift thread");
            workers.push(handle);
        }

        // ─── Sleep Predictor — Block 1 background thread that runs a
        // cheap forward-pass to pre-warm the next sleep's outcome.
        {
            let shutdown_t = Arc::clone(&shutdown);
            let store_t = Arc::clone(&store);
            let fluid_tx_t = fluid_event_tx.clone();
            let config_t = Arc::clone(&config);
            let sleep_cfg_t = Arc::clone(&sleep_config);
            let cache_t = Arc::clone(&predict_cache);
            let wake_rx_t = predict_wake_rx.clone();
            let builder = thread::Builder::new().name("voxel-sleep-predictor".to_string());
            let handle = builder
                .spawn(move || {
                    const MAX_RESPAWNS: u32 = 8;
                    let mut respawn = 0u32;
                    loop {
                        let outcome = {
                            let shutdown_t = Arc::clone(&shutdown_t);
                            let store_t = Arc::clone(&store_t);
                            let fluid_tx_t = fluid_tx_t.clone();
                            let config_t = Arc::clone(&config_t);
                            let sleep_cfg_t = Arc::clone(&sleep_cfg_t);
                            let cache_t = Arc::clone(&cache_t);
                            let wake_rx_t = wake_rx_t.clone();
                            std::panic::catch_unwind(AssertUnwindSafe(move || {
                                crate::predictor_thread::predictor_thread_loop(
                                    shutdown_t,
                                    store_t,
                                    fluid_tx_t,
                                    config_t,
                                    sleep_cfg_t,
                                    cache_t,
                                    wake_rx_t,
                                );
                            }))
                        };
                        if shutdown_t.load(Ordering::Relaxed) {
                            break;
                        }
                        match outcome {
                            Ok(()) => break,
                            Err(payload) => {
                                respawn += 1;
                                let msg = crate::panic_log::payload_string(&*payload);
                                crate::panic_log::note(&format!(
                                    "sleep-predictor caught panic (respawn {}/{}): {}",
                                    respawn, MAX_RESPAWNS, msg
                                ));
                                if respawn >= MAX_RESPAWNS {
                                    crate::panic_log::note(
                                        "sleep-predictor GIVING UP — no prediction cache",
                                    );
                                    break;
                                }
                                std::thread::sleep(std::time::Duration::from_millis(500));
                            }
                        }
                    }
                })
                .expect("failed to spawn voxel-sleep-predictor thread");
            workers.push(handle);
        }

        VoxelEngine {
            generate_tx,
            mine_tx,
            result_rx,
            priority_results: std::sync::Mutex::new(std::collections::VecDeque::new()),
            fluid_event_tx,
            fluid_thread: Some(fluid_thread),
            store,
            config,
            stress_config,
            sleep_config,
            generation_counters,
            shutdown,
            generation_paused,
            sleep_complete: Arc::new(Mutex::new(None)),
            morph_results: Arc::new(Mutex::new(std::collections::VecDeque::new())),
            morph_manifest,
            morph_snapshot,
            scan_complete: Arc::new(Mutex::new(None)),
            force_spawn_complete: Arc::new(Mutex::new(None)),
            profiler,
            workers,
            path_tx,
            path_results: Arc::new(Mutex::new(PathResultStore::default())),
            strut_broken_stash: Arc::new(Mutex::new(Vec::new())),
            next_path_request_id: Arc::new(AtomicU32::new(1)),
            occupied_cells: Arc::clone(&occupied_cells),
            crystal_anchors,
            poi_tracker,
            world_memory,
            predict_cache,
            predict_wake_tx,
            world_scale,
        }
    }

    /// Sleep-montage reveal pause. While set, worker threads stop pulling
    /// chunk-generation requests so the on-screen morph's parallel mesh-gen
    /// gets the full rayon core count (no "storm" contention). Mine/morph/sleep
    /// requests still run. Cleared at the end of each reveal.
    pub fn set_generation_paused(&self, paused: bool) {
        self.generation_paused.store(paused, Ordering::Relaxed);
    }

    pub fn shutdown(mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        // Drop senders to unblock recv_timeout / select! waits. Each closed
        // channel kicks the corresponding thread's blocking recv to return
        // Err(Disconnected) immediately, so we don't sit through any
        // configured wait interval. Most critical for predict_wake_tx —
        // the predictor's select! has a 60-second default(interval), so
        // without this drop the join can block ~46-60 seconds (real
        // user-visible PIE-exit hang we shipped on 2026-05-28).
        drop(self.generate_tx);
        drop(self.mine_tx);
        drop(self.fluid_event_tx);
        drop(self.predict_wake_tx);
        for handle in self.workers {
            let _ = handle.join();
        }
        if let Some(handle) = self.fluid_thread.take() {
            let _ = handle.join();
        }
    }
}
