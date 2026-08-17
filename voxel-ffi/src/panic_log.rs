//! Panic logging + accounting for the FFI engine.
//!
//! Workers are spawned with bare `thread::spawn` and can panic at any
//! `.unwrap()`. A panic on a worker is otherwise silent: stderr in a packaged
//! UE build goes nowhere visible, and the static `workers: Vec<JoinHandle>`
//! count keeps reporting the spawn-time worker count even if every thread is
//! dead. This module adds:
//!
//!   - a process-wide panic hook that appends each panic (location, message,
//!     backtrace) to a known log file, and
//!   - atomic counters surfaced through `voxel_get_stats` so the in-game
//!     monitor can show panics observed and currently-alive workers.

use std::any::Any;
use std::backtrace::Backtrace;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

static PANIC_COUNT: AtomicU64 = AtomicU64::new(0);
static WORKERS_ALIVE: AtomicUsize = AtomicUsize::new(0);
static LOG_PATH: OnceLock<PathBuf> = OnceLock::new();
static LOG_MUTEX: Mutex<()> = Mutex::new(());
static HOOK_INSTALLED: OnceLock<()> = OnceLock::new();

/// Install the process-wide panic hook (idempotent) and remember where to
/// write panic records. Safe to call from `VoxelEngine::new` on every engine
/// construction; only the first call wins.
pub fn install(path: impl Into<PathBuf>) {
    let _ = LOG_PATH.set(path.into());

    HOOK_INSTALLED.get_or_init(|| {
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            PANIC_COUNT.fetch_add(1, Ordering::Relaxed);
            log_panic_info(info);
            prev_hook(info);
        }));
    });
}

pub fn panic_count() -> u64 {
    PANIC_COUNT.load(Ordering::Relaxed)
}

pub fn workers_alive() -> usize {
    WORKERS_ALIVE.load(Ordering::Relaxed)
}

pub fn worker_started() {
    WORKERS_ALIVE.fetch_add(1, Ordering::Relaxed);
}

pub fn worker_exited() {
    WORKERS_ALIVE.fetch_sub(1, Ordering::Relaxed);
}

/// Append an extra line for context — used by the worker respawn loop to
/// note that a thread was caught and is restarting. The actual panic info
/// (location, backtrace) was already captured by the panic hook.
pub fn note(msg: &str) {
    let Some(path) = LOG_PATH.get() else { return; };
    let Ok(_g) = LOG_MUTEX.lock() else { return; };
    // PERSISTENT HANDLE (2026-08-18): open+close per note was an AV-scanned
    // syscall pair on every call — [MORPH-REQ]/[MORPH-STEP] alone note twice
    // per morph step, taxing the reveal prebuffer ~20-40ms per step. Per-line
    // write+flush is kept (a crash still pins the last line); only the
    // open/close churn is gone. Guarded by LOG_MUTEX like before.
    static NOTE_FILE: std::sync::Mutex<Option<std::fs::File>> = std::sync::Mutex::new(None);
    let Ok(mut guard) = NOTE_FILE.lock() else { return; };
    if guard.is_none() {
        *guard = OpenOptions::new().create(true).append(true).open(path).ok();
    }
    if let Some(f) = guard.as_mut() {
        let ts = unix_secs();
        let _ = writeln!(f, "[{:.3}] {}", ts, msg);
        let _ = f.flush();
    }
}

/// Best-effort downcast of a panic payload to a printable string.
pub fn payload_string(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        return (*s).to_string();
    }
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    "<non-string panic payload>".to_string()
}

fn log_panic_info(info: &std::panic::PanicHookInfo<'_>) {
    let Some(path) = LOG_PATH.get() else { return; };
    let Ok(_g) = LOG_MUTEX.lock() else { return; };
    let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) else { return; };

    let ts = unix_secs();
    let thread = std::thread::current();
    let thread_name = thread.name().unwrap_or("<unnamed>");

    let loc = info
        .location()
        .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
        .unwrap_or_else(|| "<unknown>".to_string());

    let msg = payload_string(info.payload());
    let backtrace = Backtrace::force_capture();
    let panic_n = PANIC_COUNT.load(Ordering::Relaxed);

    let _ = writeln!(f, "===== PANIC #{} @ ts={:.3} =====", panic_n, ts);
    let _ = writeln!(f, "thread:    {}", thread_name);
    let _ = writeln!(f, "location:  {}", loc);
    let _ = writeln!(f, "message:   {}", msg);
    let _ = writeln!(f, "backtrace:\n{}", backtrace);
    let _ = writeln!(f, "===== END PANIC #{} =====\n", panic_n);
    let _ = f.flush();
}

fn unix_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}
