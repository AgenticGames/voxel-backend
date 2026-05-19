# Perf review 2026-04-21 — follow-ons on top of 404e1ac

## TL;DR

Reviewed the last three commits (`eef7c97`, `f0760a9`, `404e1ac` — all streaming-perf).
Found four residual inefficiencies that the previous round didn't eliminate.
Implemented all four as **uncommitted** changes in `voxel-ffi/src/worker.rs`. Build clean, 30/30 voxel-ffi tests pass, 99/100 workspace (same pre-existing `blueprint_has_expected_structure` failure as last review — unrelated).

**Not pushed. Not committed.** Review the diff, validate the numbers with a pixel-server run, then commit + push if you're happy.

## The four fixes

### 1. Fuse hash-filter + crystal-fetch into one read lock (`incremental_seam_pass`)

**Before (404e1ac):** two separate read locks — one to filter `hashed → kept` on `last_sent_mesh_hash`, then a second to build a `crystal_map: HashMap<_, Vec<FfiCrystalPlacement>>`. Send loop then did `crystal_map.get(&target).cloned().unwrap_or_default()` — one Vec clone per send.

**After:** single read lock that both filters AND fetches crystal data, building `kept: Vec<(key, mesh, hash, crystal_data)>`. Crystal data moves by-value into the send tuple, so the send loop has zero clones.

**Why it matters:** the previous commit measured `store_read_wait max 150 ms → 2 ms` by going from N per-chunk acquires to 1 batched acquire. Going from 2 batched acquires to 1 is smaller but same mechanic — fewer round-trips under 8-worker contention. The `.cloned()` removal is a bonus: `FfiCrystalPlacement` is ~44 bytes, chunks can hold dozens of placements.

**Estimated saving:** **−2-4 % on `seam_pass` total**, **−5-15 % on `store_read_wait` avg**. Diminishing returns on top of 404e1ac's gains — this is the second-order cleanup.

### 2. Same fusion in `batched_seam_pass_inner`

**Before:** 3 lock acquisitions — read (filter), write (record hashes), read (crystal map). The interleaving is the worst case: the write lock sits between two reads, so any worker waiting on the store gets two chances to block.

**After:** 2 acquisitions — read (filter + crystals), write (record hashes). Same by-value crystal-data path as in (1).

**Estimated saving:** **−3-6 % on `seam_pass` total** (batched path is hotter than incremental — it's the sole sender for mine/flatten/worm-carve/cross-region, so every mine and every region boundary goes through it). Under sustained mining with 8 workers: one fewer round-trip per mine × contended RwLock → measurable store-wait reduction.

### 3. Move `hash_mesh` outside write lock in main gen path

**Before (404e1ac line 1162):**
```rust
{
    let base_hash = hash_mesh(&mesh);   // hash computed inside write lock
    let mut s = store.write().unwrap();
    s.last_sent_mesh_hash.insert(chunk, base_hash);
}
```
Actually inspecting the diff again — `hash_mesh` was called *before* the write lock, so this one is fine by luck of the original ordering. **No change needed.** (Note for future: I still split the braces to make the boundary explicit + added a comment so it doesn't regress.)

**Kept as a clarity fix, not a perf win.** Zero measurable impact. Leaving the comment/split in so a future hand doesn't innocently move the hash inside the lock.

### 4. Merge two write locks in mine path

**Before (404e1ac lines 1402-1418):**
```rust
if !new_placements.is_empty() {
    let mut s = store.write().unwrap();              // write lock #1
    for (key, placements) in new_placements {
        s.crystal_placements.insert(key, placements);
    }
}
let _ = result_tx.send(WorkerResult::MinedMaterials { mined });   // channel
{
    let mut s = store.write().unwrap();              // write lock #2
    s.queue_stress_dirty(stress_center, stress_radius);
}
```
Two write-lock acquisitions per mine, sandwiching a channel send.

**After:** single write lock covering both, channel send moved after. Ordering is independent (stress & crystal writes don't depend on the UE-bound `MinedMaterials` send).

**Estimated saving:** **−1 write lock round-trip per mine**. On a 4 Hz sustained hold-mine with 8 workers contending, this is ~200-600 μs/mine depending on load — roughly **−2-5 % on total mine latency** under heavy contention. Minor on an idle mine; noticeable when streaming is also active.

## Expected aggregate impact

Conservative, additive-on-top-of-404e1ac estimate:

| Metric | Expected additional Δ |
|---|---:|
| `seam_pass` total (cold load) | **−5 to −10 %** |
| `store_read_wait` avg/max | **−10 to −20 %** |
| Per-mine wall time under concurrent streaming | **−2 to −5 %** |
| Cold-load total wall time | **−1 to −3 %** |

These stack on top of the 404e1ac gains (−68 % seam_pass, −99 % store_read_wait max). The absolute ms saved here is smaller because the easy wins are already booked — what remains is second-order cleanup of lock-acquisition count.

## Verification done

- `cargo build -p voxel-ffi --release` — clean, 8 pre-existing warnings.
- `cargo test -p voxel-ffi --lib --release` — **30/30 pass**.
- `cargo test --workspace` — **99/100 pass**. Single failure `voxel-gen zones::mega_blueprint::blueprint_has_expected_structure` is the same pre-existing failure flagged in the 2026-04-18 and 2026-04-20 reviews. Unrelated.

**Not measured on pixel-server.** No cold-load or mining profile captures taken in this session — the estimates above are from reading the code + previous measurement baselines in the 404e1ac commit message. If you want to validate before committing, run two matched pixel-server cold loads (before/after these uncommitted changes) and diff the profiler reports.

## Files touched (uncommitted)

- `voxel-ffi/src/worker.rs`
  - `incremental_seam_pass` — fused filter + crystal fetch into single read lock, eliminated per-send `.cloned()`.
  - `batched_seam_pass_inner` — same fusion, write lock moved after the combined read, per-send `.cloned()` gone.
  - Main gen path (line ~1162) — added clarifying comment that `hash_mesh` runs before the write lock (was already correct; preserving against regression).
  - Mine path (line ~1402) — merged two write locks into one; channel send moved out of the lock sandwich.

## What to check before pushing

1. **Visual correctness**: cold load + a few mines, make sure seams are still present (these changes preserve the 404e1ac seam-gap fix logic; if seams regress, something is wrong).
2. **Measured numbers**: if you want to book the % gains above, run matched pixel-server profiles. Numbers here are engineering estimates, not measured.
3. **Lock held durations in the new merged write lock**: `queue_stress_dirty` is cheap (just inserts into a HashSet). Combining with `crystal_placements.insert` loop shouldn't blow out write-lock duration. If the stress queue ever grows to do real work, re-evaluate.

## Not done — other opportunities spotted, left for later

- **Full-density Vec materialization per dirty chunk on mine** (`worker.rs` line ~1434): `let densities: Vec<f32> = density.samples.iter().map(|s| s.density).collect()` runs per dirty chunk and allocates ~120 KB (30k floats × 4 bytes) to send over the fluid channel. Semantics may require the full field; skipped without deeper investigation. If the fluid thread only needs a small window around the mined voxels, this is a sizable allocation-churn fix.
- **Streaming `base_density` per-chunk** — the big structural fix flagged in 2026-04-20_NIGHT. Still deferred for the same reason (phase 4 worm carving needs region-wide context). Worth one dedicated session when worms come back on.
- **Parallelise backward-carve remesh loop and cross-region sync remesh loop**: currently serial `for` loops in `handle_request`. Small gain at today's `worms_per_region = 0`, trivial to do with rayon.
