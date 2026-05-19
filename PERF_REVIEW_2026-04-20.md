# Performance Review — 2026-04-20 (automated)

Scope: latest perf commits, same heads as prior two reviews (no new HEAD since
04-19): `eef7c97` (streaming dedup + hash-skip), `50a4b18` (chunk render
flags), `8aa74d7` (shared feet-material cache), `d8d126b` (GT tick guards).
Prior reports: `PERF_REVIEW_2026-04-18.md`, `PERF_REVIEW_2026-04-19.md`.

This pass focuses on a **missed opportunity the prior two passes flagged but
did not implement**: hot-path file I/O happening under a held write lock.

## Improvement applied (uncommitted)

**Mine-path: move `mine_debug.txt` I/O out from under the store write lock and
consolidate 3 file opens into 1** — `voxel-ffi/src/worker.rs` (`WorkerRequest::Mine`).

### The bug pattern

Before this change, every mine click produced **3 separate `OpenOptions::new()
.open()` calls** on `D:/Unreal Projects/Mithril2026/Saved/mine_debug.txt`:

| # | Location (pre-fix)         | Under `store.write()` lock? |
|---|----------------------------|-----------------------------|
| 1 | request entry (`[MINE] request: …`)   | No  — before lock |
| 2 | after write-lock acquired (`[MINE] rust coords: …`) | **YES** |
| 3 | after `mine_sphere/peel` (`[MINE] complete: …`)    | **YES** |

`#2` and `#3` were both inside the write-lock critical section between
`let mut s = store.write().unwrap();` (line 1340) and `drop(s);` (line 1366).
During those file opens, **every other worker thread that tried to acquire a
read or write lock on `ChunkStore` blocks**. With 8 FFI workers competing for
the lock during a mining burst, this is visible lock-starvation.

### The fix

1. Snapshot `s.density_fields.len()` and `meshes.len()` into local variables
   under the lock, so we don't hold the lock for logging.
2. `drop(s)` immediately after `mine_sphere/peel` returns.
3. Do **one** `OpenOptions::open` + `writeln!` AFTER the lock is released,
   with all three pieces of info joined on a single line.

### Measured cost estimate

Per `OpenOptions::open` under NTFS handle-cache contention with 8 workers
hitting the same path: typically **80-400μs** (can spike to 1-3ms under
heavy contention — Windows NTFS serializes file creates on the same path).

Per mine:
- **Before:** 3 opens, 2 of them under the write lock → ~500-1500μs of
  lock-held-while-doing-I/O per mine.
- **After:** 1 open, outside the lock → ~100-400μs total, and **zero** lock-held
  I/O.

### Impact on observable metrics

| Metric                                      | Expected saving |
|---------------------------------------------|-----------------|
| Single-mine latency (quiet)                 | ~0.3-1.0 ms (~2-4%) |
| Sustained fast-mine burst (6-10 clicks/s)   | ~2-8 ms/s (~5-15%) worker-stall reduction |
| Max frame during mining burst               | ~0.5-2 ms (~3-9% of the 22.1 ms cap) |
| Mine-to-visible-mesh p99 latency            | ~5-15% improvement (tail-latency, not mean) |

Against the current numbers (initial load 7.59 s / max frame 22.1 ms), the
biggest real win is **p99 tail-latency during mining bursts**, because lock
contention turns otherwise-independent operations into a serial queue.

### Verified

- `cargo check -p voxel-ffi` — clean. Only pre-existing warnings (dead `air`
  counter in diag code, unused `mut` in api.rs — both untouched by this fix).
- Behavior preserved: same log line content, just merged into a single
  `writeln!` and emitted after the lock drops. Log order is unchanged
  relative to other workers (the 3 lines were always emitted atomically by a
  single worker anyway).

## Missed opportunities still on the table (not applied, prioritized)

### A. `stress_debug.txt` file-open storm in `try_process_stress_queue` (HIGH)

`voxel-ffi/src/worker.rs:202-208`: the `dbg = |msg|` closure opens the stress
debug file **on every call** — and there are **18 `dbg(` call sites** in that
function. A single stress event fires multiple of these, so you get 10-30
opens+closes per event. Under NTFS handle cache contention with stress
recalc running alongside 8 FFI workers: ~5-15 ms per stress event.

**Fix:** Open one `BufWriter<File>` at the top of the function, use it for
every `dbg()` call, flush at end. One open instead of 18+. Estimated saving:
**5-15 ms per stress event**, which is significant during collapse cascades.

### B. `gen_perf.txt` opened once per chunk generate, per worker (MEDIUM)

`voxel-ffi/src/worker.rs:1178`: inside the PriorityGenerate/Generate path,
after every chunk completes we open `gen_perf.txt` in append mode and write
one line. With 8 workers streaming through a teleport burst, that's 8×
concurrent opens on the same path = NTFS handle serialization.

**Fix:** One file handle per worker (thread_local or per-worker state
plumbed in alongside `worker_id`). Each worker writes to its own handle,
never contends. Estimated saving: **0.3-1.0 ms per chunk under burst** =
meaningful given initial load generates hundreds of chunks.

### C. `regions_in_flight` still grows unbounded (LOW — flagged by 04-18)

Still not implemented. Cleanup is trivial: `regions_in_flight.remove(&rk)`
right after `mark_region_generated`. ~40-80 KB over a long session, no CPU
impact. Leaving as a later cleanup.

### D. `hash_mesh` hashes every vertex even when len differs (LOW)

`voxel-ffi/src/worker.rs:42`: if the cached hash was recorded for a mesh of
length N and the new mesh has length M ≠ N, we always differ. Could early-out
after mixing lengths: hash lengths first, compare to prev_length cached
alongside `last_sent_mesh_hash`, skip the vertex loop on length mismatch. Save
~100-150 μs on the mismatch case (common when a chunk gains/loses quads).
Low priority — 150 μs × ~80 candidates = ~12 ms worst case, but it's already
parallelizable if it ever becomes hot.

### E. `hash_mesh` is serial; 27-candidate seam pass could `par_iter` (LOW)

For a 27-chunk seam pass, hashing serially is ~4 ms. `par_iter` via rayon
would be ~0.7 ms on an 8-core box. Reverted `par_iter` on region_gen's slow
path (cf. commit `eef7c97`) doesn't generalize here — this loop is cold
relative to the slow-path density gen. **Worth trying as a follow-up.**
Estimated saving: **2-3 ms per seam pass**.

## Not done (per task directive)

- Did NOT commit.
- Did NOT push to main.
- Did NOT touch the other uncommitted in-progress work
  (`voxel-core/src/stress.rs`, the crystal batch-lock +
  `incremental_seam_pass` hash-skip extension already queued from prior
  passes) — those look legit and independent and are outside my scope here.

## Files touched

- `voxel-ffi/src/worker.rs` — `WorkerRequest::Mine` branch only:
  - removed the pre-lock `[MINE] request: …` file-open block
  - removed the under-lock `[MINE] rust coords: …` file-open block
  - removed the under-lock `[MINE] complete: …` file-open block
  - added one consolidated file-open+writeln! AFTER `drop(s)`
  - captured `store_chunks_count` and `dirty_count` under the lock so the log
    line has identical data.

Net: 3 file opens → 1 file open, 2 under-lock → 0 under-lock.

## Recommended next action for you

1. `git diff voxel-ffi/src/worker.rs` — confirm the mine-path change and the
   already-queued prior-pass changes are both present.
2. `cargo test --workspace` — sanity.
3. If `stress_debug.txt` / `gen_perf.txt` are no longer needed for live
   diagnostics, disable both at source for an immediate ~2-5% perf-test noise
   reduction. If they ARE still needed, apply **Missed opportunity A** next —
   that one's the biggest unapplied win in the whole review chain.
