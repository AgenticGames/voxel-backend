# Perf review of `eef7c97` (streaming optimization) — 2026-04-18

Scheduled review of the most recent commit: "Streaming optimization: per-region mutex dedup + seam-pass hash skip."

That commit added hash-skip to `batched_seam_pass_inner` (mine/flatten path) but missed several nearby wins. This report documents four improvements, three of which are implemented in `voxel-ffi/src/worker.rs` in the working tree (not committed — review and commit when ready).

---

## 1. `incremental_seam_pass` had no hash-skip (IMPLEMENTED)

**File:** `voxel-ffi/src/worker.rs`, `incremental_seam_pass` (called after every chunk generate)

**Problem:** The Round 7 hash-skip was only wired into `batched_seam_pass_inner` (mine/flatten). The single-chunk path called from `handle_request` after a chunk generates still sent every neighbor seam mesh unconditionally. Worse: because that path never *records* hashes in `last_sent_mesh_hash`, the **first** `batched_seam_pass` that fires for those chunks after a nearby mine also misses the cache and re-sends — the cache only kicks in on the *second* pass.

**Fix:** Added identical hash-compute + filter + record to `incremental_seam_pass`. Now every send path populates `last_sent_mesh_hash`, so any subsequent batched seam pass can skip unchanged neighbors.

**Estimated savings:** 10–25% reduction in FFI mesh-send count during steady-state mining (estimated from the fact that a mine produces ~8 dirty chunks × 27-neighborhood candidates = ~60-80 seam candidates, of which typically only the chunks adjacent to the mine point actually differ — the rest were being re-sent every single mine). Should further reduce the already-improved max-frame and steady-state streaming frametimes.

---

## 2. Crystal recompute lock thrashing in `WorkerRequest::Mine` (IMPLEMENTED)

**File:** `voxel-ffi/src/worker.rs`, `handle_request` → `WorkerRequest::Mine` branch

**Problem:** After mining, the code recomputes crystal placements one chunk at a time with a **read lock + write lock per key**:

```rust
for &key in &dirty_keys {
    let s = store.read().unwrap();   // read lock
    ... compute placements ...
    drop(s);
    store.write().unwrap().crystal_placements.insert(...);  // write lock
}
```

With ~8-15 dirty chunks per mine and 8 contending workers, this causes avoidable lock-handoff stalls. Also, each iteration re-acquires the RwLock even though all data is available under a single read.

**Fix:** Batch: one read lock to compute all placements (parallel-safe), one write lock to insert all. Saves ~N lock acquisitions per mine.

**Estimated savings:** ~1-4 ms per mine operation under contention (and eliminates periodic millisecond stalls visible as frame hitches during fast mining bursts). Frequency: every mine click.

---

## 3. Hash computation held the store read lock unnecessarily (IMPLEMENTED)

**File:** `voxel-ffi/src/worker.rs`, `batched_seam_pass_inner`

**Problem:** The hash-filter loop held `store.read()` while computing `hash_mesh` for every mesh:

```rust
let s = store.read().unwrap();
for (target, mesh) in to_send {
    let new_hash = hash_mesh(&mesh);   // ~150μs, doesn't touch store
    if let Some(&prev) = s.last_sent_mesh_hash.get(&target) { ... }
    ...
}
```

`hash_mesh` only touches owned `mesh` data — doesn't need the lock. But hashing 27 candidates = ~4 ms with the lock held, blocking any worker that wants a write lock (mine, generate completion, etc.).

**Fix:** Compute all hashes first (no lock), then take a brief read lock purely for the `last_sent_mesh_hash.get(&target)` lookups. Also extracted `hash_mesh` from the closure so `incremental_seam_pass` can reuse it.

**Estimated savings:** ~1-4 ms of lock-contention reduction per batched seam pass. Improves tail latency, not throughput. ~2% reduction in worker-stall metrics during mining bursts.

---

## 4. `regions_in_flight` DashMap grows unbounded (NOT IMPLEMENTED — flagged)

**File:** `voxel-ffi/src/engine.rs:175`, `voxel-ffi/src/worker.rs:71`

**Problem:** Round 1 Fix A adds an `Arc<Mutex<()>>` to `regions_in_flight` for every region ever entered. Entries are never removed. Each entry is small (~40 bytes), but over a multi-hour session with teleports to hundreds of regions, this grows to ~40-80 KB of permanently-held Arc<Mutex>. Not urgent, but worth cleaning up.

**Fix (suggested, not applied):** After the owning worker calls `s.mark_region_generated(rk)`, it can `regions_in_flight.remove(&rk)` just before dropping `_region_guard`. Any in-flight clones of the Arc are safe (they keep the Mutex alive). New workers after removal will hit `is_region_generated` → fast path, never touching `regions_in_flight`.

**Estimated savings:** ~40-80 KB leaked memory avoided over long sessions. Zero CPU impact. Left unimplemented to keep this PR focused on hot-path wins.

---

## Other observations (not fixed)

- **`dbg()` closure in `try_process_stress_queue` opens the debug file on every line** (lines 177-182). Dozens of file opens per stress recalc. Keep a single `BufWriter` open for the duration of the function. ~5-15 ms savings per stress event.
- **Per-mine debug logs in `WorkerRequest::Mine`** (lines 1310-1350) open `mine_debug.txt` three times per mine for diagnostic prints. If this debug logging is kept long-term, consolidate to a single file open.
- **`gen_perf.txt` append-open on every chunk generate** (line 1163). One file open per chunk × 8 workers = contention on NTFS handle cache. Consider a per-worker BufWriter.

These are secondary to the hot-path fixes above and all fall under the same pattern (repeated file open on hot path).

---

## Total estimated impact of the implemented changes

Aggregate per-mine steady-state: **~3-8% reduction** in average mine-to-visible-mesh latency, plus **~10-25% reduction in redundant FFI sends** during mining due to fix #1. The initial-load 7.59s → ~7.2-7.4s (small incremental improvement on top of the 58% win in the reviewed commit).

## Verification

- `cargo build -p voxel-ffi` — compiles clean (pre-existing warnings only)
- `cargo test -p voxel-ffi --lib` — 30/30 tests pass

Not committed. Review and `git commit` when ready.
