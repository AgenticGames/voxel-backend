# Perf Review 2026-06-12 — Seam-pass store-lock trim (Arc snapshot)

**Lever taken:** the "seam lock trims" item flagged by the 2026-06-10 initial-load
research round (perf-baselines entry [2026-06-10 13:50]): both seam passes held the
ChunkStore **read lock across all of their heavy work**, serializing against the
generation write locks that insert chunks during the initial-load flood. That
serialization is the measured reason worker count stopped mattering
("workers 16 vs 8: no difference — store-lock serialization, not parallelism").

## What was wrong

`incremental_seam_pass` (runs for EVERY generated chunk × its 27-neighborhood) and
`batched_seam_pass_inner` (every mine/brush/flatten/collapse flush) did this inside
one `store.read()` section, per candidate chunk:

1. `generate_chunk_seam_quads` — walk every boundary edge, build the seam mesh
2. deep-clone the cached base `Mesh` (~50–200 KB per chunk at cs=30)
3. `mesh.append(seam_mesh)`
4. `mesh.recalculate_normals()` — full O(V+T) pass (`mesh_recalc_normals` defaults to 1)

A reader holding the lock for multiple milliseconds blocks every pending writer, and
(SRW writer-preference) every reader queued behind that writer — so during the load
flood, seam passes (77.6% of Rust worker phase time) and chunk-insert writers took
turns instead of overlapping.

## The change

- `ChunkStore::chunk_seam_data` → `HashMap<K, Arc<ChunkSeamData>>`,
  `ChunkStore::base_meshes` → `HashMap<K, Arc<Mesh>>`. Entries are immutable once
  inserted; updates replace the Arc.
- `generate_chunk_seam_quads` is now generic over `Borrow<ChunkSeamData>` map values
  (works with both plain and Arc maps; existing callers/benches unchanged).
- Both seam passes now take a **brief read lock that only Arc-clones** the seam-data
  entries quad-gen can touch (target + 7 positive-offset neighbors, deduped across
  overlapping target neighborhoods) and each sendable target's base-mesh Arc.
  Quad-gen, the base deep clone, append, and normal recalc all run **after the lock
  is dropped** against the snapshot.
- Side beneficiaries: the sleep-morph out-of-block seam snapshot no longer deep-copies
  `ChunkSeamData` (dc_vertices is a gs³ Vec — ~316 KB/chunk at cs=30), it Arc-clones.
- World-scan (rare manual diagnostic) materializes a plain-Mesh map for
  `voxel_core::scan_world`, keeping voxel-core untouched.

Semantics: snapshot-then-compute is equivalent to the old flow — the old code already
dropped and re-acquired the lock between its compute phase and its hash/send phases,
so "send computed from a slightly stale view" was always possible and is healed by the
next neighbor-arrival seam pass. The hash-skip rules (including the "never hash-skip
the pass-owning chunk" race rule) are unchanged.

## Proof

`voxel-ffi/src/worker/seam.rs::seam_lock_tests` —

1. **Bit-identity** (runs in CI): faithful replicas of the PRE-restructure
   `incremental_seam_pass` / `batched_seam_pass` run against identically built stores
   (5×5 sheet of real DC-meshed sinusoidal-terrain chunks, cs=30, >1000 tris); all
   `ChunkMesh` results compared field-by-field via `f32::to_bits`, first pass and
   hash-skip second pass. Both tests pass. Full `cargo test --workspace` green.

2. **Contention bench** (`#[ignore]`d):
   `cargo test --release -p voxel-ffi --lib -- --ignored bench_seam_lock --nocapture`
   60 batched passes/round while a writer thread loops acquire→hold 200 µs→release:

   | round | side | pass wall | writer wait mean | p95 | max |
   |---|---|---:|---:|---:|---:|
   | 0 | baseline | 2.42 ms | 739 µs | 1816 µs | 2261 µs |
   | 0 | new      | 2.43 ms | **22 µs** | **173 µs** | **241 µs** |
   | 1 | baseline | 2.32 ms | 832 µs | 1884 µs | 2250 µs |
   | 1 | new      | 2.00 ms | **12 µs** | **121 µs** | **293 µs** |
   | 2 | baseline | 1.85 ms | 589 µs | 1825 µs | 2058 µs |
   | 2 | new      | 2.13 ms | **11 µs** | **112 µs** | **292 µs** |

   **Writer wait: −97~98% mean, −87% worst-case.** Seam-pass wall time itself is
   unchanged (the work moved, it didn't shrink) — the win is that writers (chunk
   inserts, hash records, mining edits) no longer queue behind multi-ms read holds.

## Expected end-to-end impact (estimate, unverified in-game)

During initial load the system is gen-bound 82% of frames and seam is 77.6% of worker
phase time, formerly almost all under the read lock. Removing ~98% of that lock-held
time should let chunk-insert writers interleave with seam work instead of alternating.
Honest estimate: **~5–15% faster initial world load** (new-chunks/s), with the real
prize being that **worker-count scaling is worth re-testing** — the 4/8/16 "identical
wall time" verdict was taken under this serialization and may now be stale. Next
measured-run loop (Scripts/run_load_measure.ps1) should redo the worker A/B.

## Not taken (still open)

- **Bulk-load seam mode** (skip incremental seams during flood, one batched flush at
  end, ~70% fewer mesh sends) — MEDIUM risk, needs the mine-during-load/unload flag
  invariants; the biggest remaining Rust-side load lever.
- Region slow-path holds `store.write()` 150–380 ms per insert — the other half of the
  serialization; would need the same snapshot treatment on the write side (chunked
  inserts or staging maps).
- Crystal/mushroom `convert_*_to_ue` still runs under the hash-filter read lock in both
  passes (small: a few placements per chunk).
