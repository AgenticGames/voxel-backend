//! Per-worker activity heartbeats + a stall monitor.
//!
//! Diagnoses the intermittent "deep sleep never ran" hang: the sleep request is
//! accepted onto the priority `mine` channel (UE logs `Deep sleep started`) but
//! no worker ever dequeues it — `voxel_panic.log` shows no `[SLEEP_TRACE] enter
//! Sleep handler`. The worker loop drains the mine channel FIRST, non-blocking,
//! at the top of every iteration, so the only way a mine request sits for 30s is
//! if every worker is simultaneously wedged mid-request (lock contention /
//! deadlock) and never loops back to check it.
//!
//! Each worker stamps what it's doing (activity tag + start time + a
//! representative coord) before handling a request and clears it after. A
//! dedicated monitor thread samples those heartbeats once a second together with
//! the mine/generate queue depths; when a priority request has been waiting too
//! long OR a worker has been stuck in one request past a threshold, it dumps a
//! `[WORKER_STALL]` snapshot to the panic log naming every worker's state. It is
//! SILENT in the common case (mine queue empty, all requests short), so it costs
//! nothing until something actually wedges — and the next stall pins the exact
//! worker + lock + chunk responsible instead of leaving us guessing.

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crossbeam_channel::Receiver;

use crate::types::WorkerRequest;

/// Activity tags (stored in an AtomicU8 per worker).
pub mod activity {
    pub const IDLE: u8 = 0;
    pub const GENERATE: u8 = 1;
    pub const SLEEP: u8 = 2;
    pub const AUREOLE: u8 = 3;
    pub const MINE: u8 = 4;
    pub const RESYNC: u8 = 5;
    pub const STRESS: u8 = 6;
    pub const BRUSH: u8 = 7;
    pub const OTHER: u8 = 8;
    pub const SEAM: u8 = 9;

    pub fn name(a: u8) -> &'static str {
        match a {
            IDLE => "idle",
            GENERATE => "Generate",
            SLEEP => "Sleep",
            AUREOLE => "AureoleOnly",
            MINE => "Mine",
            RESYNC => "Resync",
            STRESS => "Stress",
            BRUSH => "Brush",
            SEAM => "SeamFlush",
            _ => "Other",
        }
    }
}

/// Milliseconds since the Unix epoch (same clock family as `panic_log`'s
/// timestamps, so `[WORKER_STALL]` and `[SLEEP_TRACE]` lines cross-reference).
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// One worker's current activity. All fields atomic so the monitor thread can
/// read a consistent-enough snapshot without locking (a torn read at worst
/// mislabels one sample, which self-corrects on the next tick).
pub struct WorkerHeartbeat {
    activity: AtomicU8,
    since_ms: AtomicU64,
    cx: AtomicI32,
    cy: AtomicI32,
    cz: AtomicI32,
    /// Monotonic per-worker request counter — proves liveness across samples
    /// (a wedged worker's seq is frozen while a merely-busy one keeps climbing).
    seq: AtomicU64,
}

impl WorkerHeartbeat {
    pub fn new() -> Self {
        Self {
            activity: AtomicU8::new(activity::IDLE),
            since_ms: AtomicU64::new(now_ms()),
            cx: AtomicI32::new(0),
            cy: AtomicI32::new(0),
            cz: AtomicI32::new(0),
            seq: AtomicU64::new(0),
        }
    }

    /// Stamp the worker as entering a request. Call right before `handle_request`.
    pub fn enter(&self, act: u8, coord: (i32, i32, i32)) {
        self.cx.store(coord.0, Ordering::Relaxed);
        self.cy.store(coord.1, Ordering::Relaxed);
        self.cz.store(coord.2, Ordering::Relaxed);
        self.since_ms.store(now_ms(), Ordering::Relaxed);
        self.seq.fetch_add(1, Ordering::Relaxed);
        self.activity.store(act, Ordering::Release);
    }

    /// Clear back to idle. Call right after `handle_request` returns.
    pub fn idle(&self) {
        self.activity.store(activity::IDLE, Ordering::Release);
        self.since_ms.store(now_ms(), Ordering::Relaxed);
    }

    pub(crate) fn snapshot(&self) -> (u8, u64, (i32, i32, i32), u64) {
        let act = self.activity.load(Ordering::Acquire);
        (
            act,
            self.since_ms.load(Ordering::Relaxed),
            (
                self.cx.load(Ordering::Relaxed),
                self.cy.load(Ordering::Relaxed),
                self.cz.load(Ordering::Relaxed),
            ),
            self.seq.load(Ordering::Relaxed),
        )
    }
}

impl Default for WorkerHeartbeat {
    fn default() -> Self {
        Self::new()
    }
}

/// Classify a request into an activity tag + a representative coordinate, for the
/// stall snapshot. Borrows the request (called before it's moved into the
/// handler). Only the variants worth distinguishing are matched; the rest fold
/// into `BRUSH`/`OTHER` (their coords aren't needed to diagnose the sleep hang).
pub fn classify(req: &WorkerRequest) -> (u8, (i32, i32, i32)) {
    match req {
        WorkerRequest::Generate { chunk, .. } | WorkerRequest::PriorityGenerate { chunk, .. } => {
            (activity::GENERATE, *chunk)
        }
        WorkerRequest::Sleep { player_chunk, .. } => (activity::SLEEP, *player_chunk),
        WorkerRequest::AureoleOnly { player_chunk, .. } => (activity::AUREOLE, *player_chunk),
        WorkerRequest::Mine { .. } | WorkerRequest::MineAndFillFluid { .. } => {
            (activity::MINE, (0, 0, 0))
        }
        WorkerRequest::ForceChunkResync { chunk } => (activity::RESYNC, *chunk),
        WorkerRequest::ForceChunkResyncBatch { chunks } => {
            (activity::RESYNC, chunks.first().copied().unwrap_or((0, 0, 0)))
        }
        WorkerRequest::ComputePath { .. } => (activity::OTHER, (0, 0, 0)),
        _ => (activity::BRUSH, (0, 0, 0)),
    }
}

/// Spawn the background stall monitor. Holds `len()`-only clones of the mine and
/// generate receivers (never calls `recv`, so it can't steal work) and reads the
/// shared heartbeats. Exits when `shutdown` is set.
pub fn spawn_stall_monitor(
    shutdown: Arc<AtomicBool>,
    heartbeats: Arc<Vec<WorkerHeartbeat>>,
    mine_rx: Receiver<WorkerRequest>,
    generate_rx: Receiver<WorkerRequest>,
) {
    // A priority (mine) request waiting longer than this is suspicious — the
    // worker loop should drain it within one request-duration.
    const MINE_WAIT_MS: u64 = 2_000;
    // A single request running longer than this is suspicious (generation/morph
    // steps are sub-second; sleep's own execute_sleep is ~2.5s but it traces its
    // own progress, so this won't false-positive on a healthy sleep for long).
    const STUCK_MS: u64 = 8_000;
    // Throttle: re-emit a snapshot at most this often while a stall persists.
    const REEMIT_MS: u64 = 3_000;

    let builder = std::thread::Builder::new().name("voxel-stall-monitor".to_string());
    let spawn_res = builder.spawn(move || {
        // Prove the monitor is alive (its absence vs a real stall was ambiguous —
        // no [WORKER_STALL] could mean "no stall" OR "monitor never ran").
        crate::panic_log::note("[WORKER_HB] stall monitor started");
        let mut mine_nonempty_since: Option<u64> = None;
        let mut last_emit: u64 = 0;
        let mut alerting = false;
        let mut tick: u64 = 0;

        while !shutdown.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(1_000));
            tick += 1;
            let now = now_ms();
            let mine_q = mine_rx.len();
            let gen_q = generate_rx.len();

            // How long has the priority queue been backed up?
            if mine_q > 0 {
                mine_nonempty_since.get_or_insert(now);
            } else {
                mine_nonempty_since = None;
            }
            let mine_wait = mine_nonempty_since
                .map(|t| now.saturating_sub(t))
                .unwrap_or(0);

            // Is any worker stuck in one request past the threshold?
            let mut any_stuck = false;
            for hb in heartbeats.iter() {
                let (act, since, _coord, _seq) = hb.snapshot();
                if act != activity::IDLE && now.saturating_sub(since) > STUCK_MS {
                    any_stuck = true;
                    break;
                }
            }

            // Build a per-worker state string (shared by the heartbeat + alert).
            let worker_states = || -> String {
                let mut parts = Vec::with_capacity(heartbeats.len());
                for (i, hb) in heartbeats.iter().enumerate() {
                    let (act, since, coord, seq) = hb.snapshot();
                    if act == activity::IDLE {
                        parts.push(format!("w{}=idle", i));
                    } else {
                        parts.push(format!(
                            "w{}={}({},{},{}) {:.1}s seq{}",
                            i, activity::name(act), coord.0, coord.1, coord.2,
                            now.saturating_sub(since) as f64 / 1000.0, seq,
                        ));
                    }
                }
                parts.join(", ")
            };

            // Unconditional heartbeat every 5s — proves the monitor is alive and
            // shows exactly what it sees (mine/gen queue depths, worker states)
            // during a stall, even if the alert thresholds somehow don't trip.
            if tick % 5 == 0 {
                crate::panic_log::note(&format!(
                    "[WORKER_HB] alive={} mine_q={} gen_q={} mine_wait={:.1}s | {}",
                    crate::panic_log::workers_alive(),
                    mine_q, gen_q, mine_wait as f64 / 1000.0, worker_states(),
                ));
            }

            let alert = mine_wait > MINE_WAIT_MS || any_stuck;
            if alert {
                if now.saturating_sub(last_emit) >= REEMIT_MS || !alerting {
                    last_emit = now;
                    alerting = true;
                    crate::panic_log::note(&format!(
                        "[WORKER_STALL] alive={} mine_q={} (waiting {:.1}s) gen_q={} | {}",
                        crate::panic_log::workers_alive(),
                        mine_q,
                        mine_wait as f64 / 1000.0,
                        gen_q,
                        worker_states(),
                    ));
                }
            } else if alerting {
                alerting = false;
                crate::panic_log::note(&format!(
                    "[WORKER_STALL] cleared (mine_q={} gen_q={})",
                    mine_q, gen_q
                ));
            }
        }
    });
    if spawn_res.is_err() {
        crate::panic_log::note("[WORKER_HB] FAILED to spawn stall monitor thread");
    }
}
