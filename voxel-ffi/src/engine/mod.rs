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
///
/// ⚠️ 2026-08-24 (user): 1.5 -> 0.85, i.e. BELOW the 1.0 collapse threshold.
/// Sitting above it made cracks structurally incapable of warning anyone —
/// anything that actually collapsed did so without ever showing one, and the
/// only cells that DID crack were those too incoherent to drop. Cracks are a
/// telegraph now: they appear in the band before the fall.
///
/// The confusion that originally pushed this to 1.5 ("red but stable" cells
/// wearing cracks forever) is now handled where it belongs — by COHERENCE,
/// not by raising the bar. `enumerate_overstressed_in_chunk` only emits cells
/// whose connected overstressed cluster is within one cell of
/// `min_collapse_region`, so a lone stressed voxel that can never form a slab
/// stays clean while a patch big enough to fall starts cracking first.
pub const COLLAPSE_IMMINENT_STRESS: f32 = 0.85;

/// Crack-visibility threshold for COAL only.
///
/// 2026-08-30 (user): "I need the stress threshold for Cracks showing up to be
/// reduced so coal on a wall is likely to be cracked out." Coal came out of the
/// 2026-08-24 pass in a dead band: its hardness was raised 0.30 -> 0.40
/// precisely so flat seams would stop caving, and that same change put the
/// ordinary exposed-seam voxel at ~0.75 effective (see the note on Coal in
/// `DEFAULT_MATERIAL_HARDNESS`) — just under the 0.85 crack bar. So coal
/// reliably sat one notch below cracking while the rock around it cracked.
///
/// **0.25 is measured, not guessed.** Same save, same fixed pose, same single
/// `mine` at (-2791,1167,595), only this constant varied — coal cells entering
/// the crack list in the coal-bearing chunk (-2,-1,0):
///
/// | this const | qualifying cells | of which coal |
/// |------------|------------------|---------------|
/// | 0.85 (old) | 22               | 0             |
/// | 0.45       | 35               | 13            |
/// | 0.25       | 50               | 28            |
/// | 0.05       | 69               | 47 (all of it)|
///
/// So the stressed-coal population in that chunk is 47 and it is bottom-heavy:
/// 0.25 takes ~60% of it while still leaving the least-loaded coal clean, and
/// the coherence gate below still refuses clusters too small to form a slab.
/// Push it lower only if coal should read as cracked essentially everywhere.
///
/// ⚠️ This is a LOOK, not a stability change. It is read only by
/// `enumerate_overstressed_in_chunk`, which is the crack-decal source and
/// nothing else. The collapse pass, the warn-dust/shake/creak tiers and
/// `enumerate_overstressed_in_sphere` (mining dust bursts + the strut-braced
/// quest probe) all still use the shared threshold, so cracked coal is not
/// newly unstable and no audio or quest trigger moved.
pub const COAL_CRACK_STRESS: f32 = 0.25;

/// Lowest crack threshold ANY material can have — the cheap pre-filter that
/// lets the per-cell scan skip its material lookup for cells that could not
/// crack whatever they are made of. Keep it equal to the smallest value
/// `crack_stress_threshold` can return (debug-asserted at the call site).
pub const MIN_CRACK_STRESS: f32 = COAL_CRACK_STRESS;

/// Effective stress at which a cell of `mat` starts wearing crack decals.
/// Per-material since 2026-08-30; everything except coal keeps the shared
/// [`COLLAPSE_IMMINENT_STRESS`] telegraph band.
#[inline]
pub fn crack_stress_threshold(mat: voxel_core::material::Material) -> f32 {
    match mat {
        voxel_core::material::Material::Coal => COAL_CRACK_STRESS,
        _ => COLLAPSE_IMMINENT_STRESS,
    }
}

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
    /// Worker requests the game thread could not place because `mine_tx` was
    /// full. The game thread must NEVER block on a channel send: 2026-08-13 a
    /// sleep's ~800-mesh result flood parked every worker on the bounded
    /// result queue, nobody pulled `mine_rx`, it filled to capacity, and the
    /// old blocking ApplyLavaQuench re-dispatch froze the game thread MID-
    /// DRAIN — permanent engine-wide gridlock (game thread was the only
    /// consumer that could have freed the result queue). Deferred requests
    /// are flushed with try_send at the top of every `poll_result`.
    pending_mine_redispatch: std::sync::Mutex<std::collections::VecDeque<WorkerRequest>>,

    // Fluid
    fluid_event_tx: Sender<FluidEvent>,
    /// Save-restore fluid import stash — the guaranteed-delivery lane the
    /// sim drains every iteration. Never send `PendingFluidLoad` through
    /// `fluid_event_tx`: the bounded channel drops under the load flood.
    pub(crate) fluid_import_stash: voxel_fluid::FluidImportStash,
    fluid_thread: Option<JoinHandle<()>>,

    // Shared state
    store: Arc<RwLock<ChunkStore>>,
    /// Last successfully sampled `chunks_loaded()` value. `get_stats` polls
    /// the store with `try_read` and must not block the game thread; when a
    /// writer holds the lock it reports this cached sample instead of 0
    /// (which made external pollers see the count flap 0↔real at idle).
    last_chunks_loaded: AtomicU32,
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
    /// Shared per-worker heartbeats (same Arc the stall monitor reads).
    /// Held here so `shutdown()`'s slow-join watchdog can dump each
    /// worker's live activity when a teardown join wedges.
    worker_heartbeats: Arc<Vec<crate::worker::heartbeat::WorkerHeartbeat>>,

    // ─── Pathfinding ─────────────────────────────────────────
    /// Dedicated channel for path requests so heavy `BrushCavernStamp` /
    /// `Sleep` / etc. on `mine_tx` don't head-of-line block AI path queries.
    path_tx: Sender<WorkerRequest>,
    /// Dedicated lane for sleep-montage morph steps — never queues behind
    /// gen/stress/quench work on `mine_tx` (2026-08-13 reveal-stall fix).
    morph_tx: Sender<WorkerRequest>,
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
        // Global rayon pool must be claimed before any rayon use so its
        // workers run below-normal (see thread_priority.rs).
        crate::thread_priority::init_rayon_below_normal();
        // Install panic hook + log file as early as possible. Idempotent
        // across engine creation; first install wins.
        // 2026-09-07: exe-relative. The hard-coded dev path meant no other machine
        // ever wrote this file - PC2 had "no voxel_panic.log" through three stalled
        // loads while the stall monitor was reporting into the void. The deployed
        // demo runs from <install>/Mithril2026/Binaries/Win64/, so on PC2 the file
        // lands under the MithrilDeploy share where the dev box can read it.
        // Order: the dev box's project Saved (exists only here) -> the per-user
        // %LOCALAPPDATA%\Mithril2026\Saved that the Shipping demo already uses for
        // demo_session.log (an install folder may not be writable for a standard
        // user - PC2 produced no file next to the exe) -> exe-relative.
        let dev_saved = std::path::PathBuf::from("D:/Unreal Projects/Mithril2026/Saved");
        let panic_path = if dev_saved.is_dir() {
            dev_saved.join("voxel_panic.log")
        } else if let Some(la) = std::env::var_os("LOCALAPPDATA") {
            std::path::PathBuf::from(la).join("Mithril2026").join("Saved").join("voxel_panic.log")
        } else {
            std::env::current_exe().ok()
                .and_then(|e| e.parent().map(|d| d.join("Saved").join("voxel_panic.log")))
                .unwrap_or_else(|| std::path::PathBuf::from("voxel_panic.log"))
        };
        if let Some(dir) = panic_path.parent() { let _ = std::fs::create_dir_all(dir); }
        crate::panic_log::install(panic_path);

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
        // Dedicated morph-step lane (2026-08-13, see worker::morph_worker_loop).
        let (morph_tx, morph_rx) = bounded::<WorkerRequest>(16);
        let (result_tx, result_rx) = bounded::<WorkerResult>(2048);

        // Fluid event channel
        // 2026-09-07: UNBOUNDED. This was bounded(512) with blocking sends from
        // every worker: once the fluid sim stalled on ITS full result channel,
        // the event queue filled and every worker blocked on its next send -
        // chunk generation froze at a fixed count with no panic anywhere
        // (PC2, session 619D51C7: 205 chunks for 120 s). Memory only grows if
        // the sim is wedged, and the sim can no longer wedge (see thread.rs).
        let (fluid_event_tx, fluid_event_rx) = crossbeam_channel::unbounded::<FluidEvent>();

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
        // Save-restore fluid bypasses the bounded event channel (which the
        // load-time flood fills — dropped imports) via this shared stash.
        let fluid_import_stash: voxel_fluid::FluidImportStash = Default::default();
        let fluid_import_stash_for_thread = Arc::clone(&fluid_import_stash);
        let fluid_thread = thread::spawn(move || {
            crate::thread_priority::set_current_below_normal();
            fluid_sim_loop_wrapper(
                fluid_shutdown,
                fluid_event_rx,
                fluid_result_tx,
                fluid_config,
                fluid_world_scale,
                fluid_import_stash_for_thread,
            );
        });

        let profiler = Arc::new(StreamingProfiler::new(num_workers));
        let morph_manifest: Arc<Mutex<Option<voxel_sleep::ChangeManifest>>> = Arc::new(Mutex::new(None));
        let morph_snapshot: Arc<Mutex<crate::worker::MorphSnapshot>> = Arc::new(Mutex::new(crate::worker::MorphSnapshot::default()));
        // Created pre-spawn so workers can push finished morph steps STRAIGHT
        // into the queue voxel_poll_morph_result pops (bypasses result_tx —
        // see HandlerCtx::morph_results).
        let morph_results: Arc<Mutex<std::collections::VecDeque<MorphStepResult>>> =
            Arc::new(Mutex::new(std::collections::VecDeque::new()));

        // Per-region generation-in-flight claims. Prevents 2+ workers from
        // redundantly generating the same region's base_density (wasted CPU).
        // A worker claims a region via the entry's gate before slow-path;
        // other workers for the same region PARK their request on the entry
        // (region-convoy fix — blocking here idled workers 3.5-5.2s) and are
        // re-dispatched via ParkedGenerates when the owner commits.
        let regions_in_flight: Arc<DashMap<(i32, i32, i32), Arc<crate::worker::RegionInFlight>>> =
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
        // Shared by all workers: chunks whose seam pass was deferred during a
        // generate-queue flood (bulk-load seam mode). Drained in batches when
        // the generate queue idles — see worker_loop's Timeout branch.
        let pending_seams = Arc::new(crate::worker::PendingSeams::new());
        // Shared by all workers: generate requests parked on an in-flight
        // region, re-dispatched with priority once the region's densities
        // commit. See worker::ParkedGenerates (region-convoy fix).
        let parked_generates = Arc::new(crate::worker::ParkedGenerates::new());
        // Bound concurrent region slow paths to half the pool (min 2) — see
        // worker::SlowPathPermits for the latency-vs-throughput rationale.
        let slow_path_permits = Arc::new(crate::worker::SlowPathPermits::new(
            (num_workers / 2).max(2),
        ));

        for worker_id in 0..num_workers {
            let shutdown = Arc::clone(&shutdown);
            let generation_paused = Arc::clone(&generation_paused);
            let generate_rx = generate_rx.clone();
            let mine_rx = mine_rx.clone();
            let mine_tx_w = mine_tx.clone();
            let result_tx = result_tx.clone();
            let store = Arc::clone(&store);
            let config = Arc::clone(&config);
            let stress_cfg = Arc::clone(&stress_config);
            let gen_counters = Arc::clone(&generation_counters);
            let fluid_tx = fluid_event_tx.clone();
            let prof = Arc::clone(&profiler);
            let morph_man = Arc::clone(&morph_manifest);
            let morph_snap = Arc::clone(&morph_snapshot);
            let morph_resq = Arc::clone(&morph_results);
            let rif = Arc::clone(&regions_in_flight);
            let anchors = Arc::clone(&crystal_anchors);
            let heartbeats = Arc::clone(&heartbeats);
            let deferred_stress = Arc::clone(&deferred_region_stress);
            let pending_seams = Arc::clone(&pending_seams);
            let parked_generates = Arc::clone(&parked_generates);
            let slow_path_permits = Arc::clone(&slow_path_permits);

            let builder = thread::Builder::new().name(format!("voxel-worker-{}", worker_id));
            let handle = builder
                .spawn(move || {
                    crate::thread_priority::set_current_below_normal();
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
                            let mine_tx_w = mine_tx_w.clone();
                            let result_tx = result_tx.clone();
                            let store = Arc::clone(&store);
                            let config = Arc::clone(&config);
                            let stress_cfg = Arc::clone(&stress_cfg);
                            let gen_counters = Arc::clone(&gen_counters);
                            let fluid_tx = fluid_tx.clone();
                            let prof = Arc::clone(&prof);
                            let morph_man = Arc::clone(&morph_man);
                            let morph_snap = Arc::clone(&morph_snap);
                            let morph_resq = Arc::clone(&morph_resq);
                            let rif = Arc::clone(&rif);
                            let anchors = Arc::clone(&anchors);
                            let heartbeats = Arc::clone(&heartbeats);
                            let deferred_stress = Arc::clone(&deferred_stress);
                            let pending_seams = Arc::clone(&pending_seams);
                            let parked_generates = Arc::clone(&parked_generates);
                            let slow_path_permits = Arc::clone(&slow_path_permits);
                            std::panic::catch_unwind(AssertUnwindSafe(move || {
                                worker_loop(
                                    shutdown,
                                    generation_paused,
                                    generate_rx,
                                    mine_rx,
                                    mine_tx_w,
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
                                    morph_resq,
                                    rif,
                                    anchors,
                                    heartbeats,
                                    deferred_stress,
                                    pending_seams,
                                    parked_generates,
                                    slow_path_permits,
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
                    crate::thread_priority::set_current_below_normal();
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

        // ─── Dedicated morph-step worker (2026-08-13) — one thread owns
        // morph_rx so sleep-montage reveal steps never queue behind gen /
        // stress / quench bursts on the shared pool (see
        // worker::morph_worker_loop for the stall forensics). Same
        // catch_unwind respawn pattern as the path workers.
        {
            let shutdown_m = Arc::clone(&shutdown);
            let morph_rx_m = morph_rx.clone();
            let generate_rx_m = generate_rx.clone();
            let mine_rx_m = mine_rx.clone();
            let mine_tx_m = mine_tx.clone();
            let result_tx_m = result_tx.clone();
            let store_m = Arc::clone(&store);
            let config_m = Arc::clone(&config);
            let stress_cfg_m = Arc::clone(&stress_config);
            let gen_counters_m = Arc::clone(&generation_counters);
            let fluid_tx_m = fluid_event_tx.clone();
            let prof_m = Arc::clone(&profiler);
            let morph_man_m = Arc::clone(&morph_manifest);
            let morph_snap_m = Arc::clone(&morph_snapshot);
            let morph_resq_m = Arc::clone(&morph_results);
            let rif_m = Arc::clone(&regions_in_flight);
            let anchors_m = Arc::clone(&crystal_anchors);
            let deferred_stress_m = Arc::clone(&deferred_region_stress);
            let pending_seams_m = Arc::clone(&pending_seams);
            let parked_generates_m = Arc::clone(&parked_generates);
            let slow_path_permits_m = Arc::clone(&slow_path_permits);
            let world_scale_m = world_scale;
            let builder = thread::Builder::new().name("voxel-morph-worker".to_string());
            let handle = builder
                .spawn(move || {
                    crate::thread_priority::set_current_below_normal();
                    const MAX_RESPAWNS: u32 = 16;
                    let mut respawn = 0u32;
                    crate::panic_log::worker_started();
                    loop {
                        let outcome = {
                            let shutdown = Arc::clone(&shutdown_m);
                            let morph_rx = morph_rx_m.clone();
                            let generate_rx = generate_rx_m.clone();
                            let mine_rx = mine_rx_m.clone();
                            let mine_tx = mine_tx_m.clone();
                            let result_tx = result_tx_m.clone();
                            let store = Arc::clone(&store_m);
                            let config = Arc::clone(&config_m);
                            let stress_cfg = Arc::clone(&stress_cfg_m);
                            let gen_counters = Arc::clone(&gen_counters_m);
                            let fluid_tx = fluid_tx_m.clone();
                            let prof = Arc::clone(&prof_m);
                            let morph_man = Arc::clone(&morph_man_m);
                            let morph_snap = Arc::clone(&morph_snap_m);
                            let morph_resq = Arc::clone(&morph_resq_m);
                            let rif = Arc::clone(&rif_m);
                            let anchors = Arc::clone(&anchors_m);
                            let deferred_stress = Arc::clone(&deferred_stress_m);
                            let pending_seams = Arc::clone(&pending_seams_m);
                            let parked_generates = Arc::clone(&parked_generates_m);
                            let slow_path_permits = Arc::clone(&slow_path_permits_m);
                            std::panic::catch_unwind(AssertUnwindSafe(move || {
                                crate::worker::morph_worker_loop(
                                    shutdown, morph_rx, generate_rx, mine_rx, mine_tx, result_tx,
                                    store, config, stress_cfg, gen_counters, world_scale_m,
                                    fluid_tx, prof, morph_man, morph_snap, morph_resq,
                                    rif, anchors, deferred_stress, pending_seams,
                                    parked_generates, slow_path_permits,
                                );
                            }))
                        };
                        if shutdown_m.load(Ordering::Relaxed) {
                            break;
                        }
                        match outcome {
                            Ok(()) => break,
                            Err(payload) => {
                                respawn += 1;
                                let msg = crate::panic_log::payload_string(&*payload);
                                crate::panic_log::note(&format!(
                                    "morph-worker PANIC ({}/{}): {}", respawn, MAX_RESPAWNS, msg));
                                if respawn >= MAX_RESPAWNS {
                                    crate::panic_log::note("morph-worker GIVING UP after respawn limit");
                                    break;
                                }
                                std::thread::sleep(std::time::Duration::from_millis(100));
                            }
                        }
                    }
                    crate::panic_log::worker_exited();
                })
                .expect("failed to spawn voxel morph worker thread");
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
                    crate::thread_priority::set_current_below_normal();
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
                    crate::thread_priority::set_current_below_normal();
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
                    crate::thread_priority::set_current_below_normal();
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
            pending_mine_redispatch: std::sync::Mutex::new(std::collections::VecDeque::new()),
            fluid_event_tx,
            fluid_import_stash,
            fluid_thread: Some(fluid_thread),
            store,
            config,
            stress_config,
            sleep_config,
            generation_counters,
            shutdown,
            generation_paused,
            sleep_complete: Arc::new(Mutex::new(None)),
            morph_results,
            morph_manifest,
            morph_snapshot,
            scan_complete: Arc::new(Mutex::new(None)),
            force_spawn_complete: Arc::new(Mutex::new(None)),
            profiler,
            workers,
            worker_heartbeats: heartbeats,
            path_tx,
            morph_tx,
            path_results: Arc::new(Mutex::new(PathResultStore::default())),
            strut_broken_stash: Arc::new(Mutex::new(Vec::new())),
            next_path_request_id: Arc::new(AtomicU32::new(1)),
            last_chunks_loaded: AtomicU32::new(0),
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
        // Drop the RESULT receiver too, for the send side of the same
        // deadlock family: UE stops polling results the instant PIE tears
        // down, so the bounded result queue (2048) fills and any thread
        // mid-`result_tx.send()` blocks forever — it never re-checks the
        // shutdown flag, and the join below hangs the game thread inside
        // DestroyEngine ("Rust thread joins" was the last SHUTDOWN-TRACE
        // breadcrumb of the 2026-07-17 infinite PIE-exit hang). The fluid
        // thread sends FluidMesh results through the same channel, so a
        // full result queue also stops it draining fluid events, which in
        // turn wedges workers on fluid_event_tx — dropping the receiver
        // fail-fasts the entire cascade (send returns SendError, loops see
        // `shutdown` and exit).
        drop(self.result_rx);

        // Slow-join watchdog: if teardown wedges, dump every worker's live
        // heartbeat to voxel_panic.log every 10s so the hang names its own
        // culprit (activity + chunk + how long it's been stuck). Exits as
        // soon as the joins complete — silent in the healthy case.
        let joins_done = Arc::new(AtomicBool::new(false));
        {
            let done = Arc::clone(&joins_done);
            let hb = Arc::clone(&self.worker_heartbeats);
            let _ = thread::Builder::new()
                .name("voxel-shutdown-watchdog".to_string())
                .spawn(move || {
                    use crate::worker::heartbeat::{activity, now_ms};
                    for round in 1u32..=18 {
                        for _ in 0..100 {
                            if done.load(Ordering::Relaxed) { return; }
                            std::thread::sleep(std::time::Duration::from_millis(100));
                        }
                        let now = now_ms();
                        let mut line = format!(
                            "[FFI_SHUTDOWN] joins still pending after ~{}s |", round * 10
                        );
                        for (i, h) in hb.iter().enumerate() {
                            let (act, since, coord, seq) = h.snapshot();
                            line.push_str(&format!(
                                " w{}={}{:?} {:.1}s seq{}",
                                i, activity::name(act), coord,
                                now.saturating_sub(since) as f64 / 1000.0, seq,
                            ));
                        }
                        crate::panic_log::note(&line);
                    }
                });
        }

        for (i, handle) in self.workers.into_iter().enumerate() {
            let _ = handle.join();
            crate::panic_log::note(&format!("[FFI_SHUTDOWN] worker {} joined", i));
        }
        if let Some(handle) = self.fluid_thread.take() {
            let _ = handle.join();
            crate::panic_log::note("[FFI_SHUTDOWN] fluid thread joined");
        }
        joins_done.store(true, Ordering::Relaxed);
        crate::panic_log::note("[FFI_SHUTDOWN] all joins complete");

        // Deallocate the ChunkStore on a detached reaper thread. Dropping a
        // large store on the game thread was ~100% of the remaining PIE-exit
        // hang once joins were fixed: a 16.8k-chunk store (runaway-fall
        // session, 2026-07-18) took ~200s of pure free() inside
        // DestroyEngine; even normal multi-hour sessions accumulate seconds
        // of drop time. All threads are joined at this point, so this Arc is
        // the last strong ref and the reaper does the real deallocation.
        // The DLL outlives PIE sessions, so the thread can't outlive its
        // code; at editor exit the heap dies with the process either way.
        let store = Arc::clone(&self.store);
        drop(self.store);
        let _ = thread::Builder::new()
            .name("voxel-store-reaper".to_string())
            .spawn(move || {
                let n = store.read().map(|s| s.chunks_loaded()).unwrap_or(0);
                let t = std::time::Instant::now();
                drop(store);
                crate::panic_log::note(&format!(
                    "[FFI_SHUTDOWN] store reaper: {} chunks freed in {:.1}s (off-thread)",
                    n,
                    t.elapsed().as_secs_f64()
                ));
            });
    }
}
