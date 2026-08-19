//! Below-normal priority for backend compute threads (2026-08-19).
//!
//! During generation storms this DLL runs 8+ generate workers, fluid, morph,
//! path and stress threads plus rayon's global pool — 20+ runnable threads at
//! NORMAL priority against UE's game thread on a 16-logical-CPU box. Windows
//! round-robins equal-priority runnable threads, so the game thread waits
//! multi-quantum stretches: measured 50-190ms stalls INSIDE trivial FFI calls
//! (poll_result sections all fast; convert_mesh_to_ffi_result "took" 142ms
//! doing nothing but a pointer repack) with a bimodal P50=1us / max=188ms
//! profile. Player-facing symptom: "camera turns at ~25% for ~0.75s while
//! chunks stream in" — input was measured lossless end-to-end; the frames
//! themselves were being starved.
//!
//! Below-normal compute threads always yield the CPU to the game/render
//! threads under contention; throughput on idle cores is unchanged.
//!
//! Deliberately NOT lowered: the sleep-sim thread (its wall time is the
//! montage wait the 08-18 overhaul tuned — lowering it would regress
//! wait-to-curtain under load), the heartbeat/stall monitor (its timestamps
//! must stay honest during exactly these storms), and shutdown-time utility
//! threads.

#[cfg(windows)]
pub(crate) fn set_current_below_normal() {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThread() -> isize;
        fn SetThreadPriority(handle: isize, priority: i32) -> i32;
    }
    const THREAD_PRIORITY_BELOW_NORMAL: i32 = -1;
    unsafe {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    }
}

#[cfg(not(windows))]
pub(crate) fn set_current_below_normal() {}

/// Initialize rayon's GLOBAL pool with below-normal, named worker threads.
/// Must run before the first rayon use in the process — engine construction
/// is the earliest DLL entry point. A second engine in the same process gets
/// Err(GlobalPoolAlreadyInitialized), which is fine: the first pool stands.
pub(crate) fn init_rayon_below_normal() {
    let _ = rayon::ThreadPoolBuilder::new()
        .thread_name(|i| format!("voxel-rayon-{}", i))
        .start_handler(|_| set_current_below_normal())
        .build_global();
}
