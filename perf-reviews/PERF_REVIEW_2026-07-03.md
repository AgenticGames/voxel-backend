# PERF REVIEW 2026-07-03 — Region slow-path write-lock hold trim

Scheduled "review latest commits" pass. Takes the **#2 confirmed lever** from
PERF_REVIEW_2026-06-12's in-game verification: *"region slow-path holds
`store.write()` 150–380 ms per insert — the other half of the serialization;
would need the same snapshot treatment on the write side."* The 06-12 seam
read-lock trim (`f15ae6f`) proved readers-blocking-writers was the small share;
`store_read_wait` stayed high because **writers block everyone** — and the
biggest single writer hold is the region slow-path commit in
`voxel-ffi/src/worker/generate.rs`.

## The defect(s)

The critical section (insert region chunks → apply save snapshots → re-sync all
boundaries → recompute hermite/meshes for changed chunks) held the store write
lock while doing all of this **serially**, with three separate inefficiencies:

1. **Double hermite extraction for snapshot-restored chunks.** Every chunk with
   a save/preserved snapshot got `extract_hermite_data` run eagerly at apply
   time — and then the boundary re-sync (which near-always dirties restored
   chunks, since the snapshot rewrites boundary cells) ran a **second** full
   extraction of the same chunk. The old code even admitted it: *"May be
   re-extracted again below if re-sync touches this chunk."* On a save-load
   region that's one wasted full-grid extraction per restored chunk, all inside
   the hold.

2. **~150 KB density clone per recompute**, twice per code path — a
   `df.clone()` existed purely to appease the borrow checker before each
   extraction.

3. **Serial recompute.** Each dirty chunk's work (hermite extract + for
   previously-meshed chunks a full DC re-solve + mesh gen + smooth + normal
   recalc + boundary-edge extract) is a **pure function of that chunk's own
   post-sync density** — embarrassingly parallel, but ran one chunk at a time
   while every other worker queued on the lock.

## The change

Extracted the critical section into `insert_region_chunks_and_resync()`
(called under the same write guard — atomicity vs other workers is
**unchanged**, only the hold gets shorter):

- Snapshot chunks no longer extract eagerly; they **seed the recompute set**.
  Chunks the re-sync also dirties are extracted exactly once (from the final
  post-sync density — same value the old second extraction produced). Chunks
  the re-sync did NOT dirty still get their one extraction (the old eager one),
  and — matching old behavior exactly — never take the prev_meshed re-solve
  path (that stays gated on sync-dirty).
- The recompute pass runs via `rayon par_iter` **while the lock is held**.
  Densities are frozen for the whole pass (nothing in it mutates
  `density_fields`), closures take no locks (pool threads can never block on
  the store → no deadlock; same pattern as the `store/search.rs` par scans that
  already run under read guards).
- Clones gone — extraction reads `&DensityField` in place.
- Recompute keys are sorted, so downstream send order via `sync_remeshed` is
  now deterministic (the old `HashSet` iteration order never was). The dead
  `newly_inserted` vec (write-only) was dropped.

## Proof

`voxel-ffi/src/worker/generate.rs::region_insert_tests` —

1. **Bit-identity (runs in CI):** a faithful replica of the pre-restructure
   critical section runs against an identically built store and the result is
   compared field-by-field via `f32::to_bits` across `density_fields`,
   `hermite_data`, `base_meshes`, `chunk_seam_data` (incl. boundary-edge
   ORDER — sound because `EdgeMap` uses an identity hasher, so iteration is
   deterministic regardless of thread), plus dirty-chunk tracking and both
   outcome key sets. Scenario exercises every path: fresh inserts,
   boundary-modifying snapshot (sync-dirty snapshot chunk), interior-only
   snapshots (the dedup path — asserted extract-only), a stale pre-existing
   in-region chunk, and biased pre-existing meshed neighbors (prev_meshed DC
   re-solve — asserted present). `sync_remeshed` compared as a set (old order
   was nondeterministic).

2. **Write-hold A/B** (`#[ignore]`d):
   `cargo test --release -p voxel-ffi --lib -- --ignored bench_region_insert --nocapture`
   — 5×1×5-chunk region sheet at cs=30 (real DC-meshed sinusoidal terrain), 17
   snapshot-restored chunks, 10 pre-existing meshed cross-region neighbors with
   biased densities (all sync-dirty + re-solved). 8C/16T machine:

   | round | baseline hold | optimized hold | delta |
   |---|---:|---:|---:|
   | 0 | 24.04 ms | 10.44 ms | **−56.5%** |
   | 1 | 23.87 ms | 10.70 ms | **−55.2%** |
   | 2 | 25.37 ms | 12.36 ms | **−51.3%** |

   The residual ~10 ms is the boundary re-sync itself (inherently serial — it
   mutates neighbor densities pairwise) plus insert/apply. The recompute
   portion (~14 ms serial) drops to ~2 ms parallel + dedup.

`cargo test --workspace` green (25 test binaries, 0 failures; voxel-ffi 131
incl. the new identity test).

## Expected end-to-end impact (estimate)

The live holds are 150–380 ms per slow region insert (bigger regions, cave-
heavy meshes, more restored chunks than the bench scenario — and the recompute
share grows with snapshot count, which is exactly the save-load case the
06-12 measurements profiled). Applying the measured ratio conservatively:
**~50% off each region slow-path write hold** (est. 150–380 ms → ~70–190 ms).
Since `store_read_wait` during initial load (4–10 s total across workers) is
dominated by these write holds, and gen-bound frames are 82–89% of the load
window, honest estimate: **~5–10% faster initial world load on save-load
(Continue) runs**, compounding with the earlier seam read-lock trim. The
worker-count A/B is worth redoing again after this — both halves of the
store-lock serialization flagged on 06-12 are now trimmed.

## Not taken (still open)

- **Bulk-load seam mode** (~70% fewer mesh sends during flood) — MEDIUM risk,
  unchanged status, still the biggest remaining Rust-side load lever.
- The boundary re-sync itself is still serial under the hold (~10 ms in the
  bench). Pairwise-disjoint chunk pairs could sync in parallel, but the
  both-sides-written aliasing makes that a genuinely risky restructure — not
  attempted in an autonomous pass.
- Crystal/mushroom `convert_*_to_ue` under the hash-filter read lock (small).
