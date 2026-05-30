# Perf review — 2026-05-31 (scheduled "review latest commits" task)

**Reviewer:** Claude (autonomous scheduled run). **Scope:** latest commits on `main`, headed by
`89bd270` *"Sleep montage backend: tri-state solidity, montage-protected density, worker heartbeat."*

## Finding (fixed in this commit): debug instrumentation left on a hot path

Commit `89bd270` shipped four blocks of code explicitly marked **`TEMP VFX diagnostic`** straight
to `main`. They were added today to chase a stress-crack-VFX bug (cells reading `field=MISSING` /
`interior_skip`), but they sit on performance-sensitive paths:

### 1. Game-thread file I/O per VFX query — the important one
`enumerate_overstressed_in_chunk` ([voxel-ffi/src/engine/queries.rs](voxel-ffi/src/engine/queries.rs))
is reached from the FFI export `voxel_list_overstressed_in_chunk`, which **UE calls per chunk on the
game thread** to drive stress-crack decals (per [crack-overlay-system] — VFX recalc runs at chunk
gen, so this fires across the streaming front).

Each call did:
- a `field=MISSING` early-return path that **opened a file + wrote a line and did nothing else**, and
- an end-of-call path that **opened `Saved/stress_vfx_qry.txt` (append) + wrote a line on every call.**

On Windows an append-open → write → close is a full `CreateFile` syscall round-trip (commonly
tens-to-hundreds of µs, occasionally ms-scale under AV/Defender scanning), executed **on the game
thread, once per chunk queried.** For the common case — a chunk with no imminent-collapse surface
cells — that file I/O was essentially the *entire* cost of the call (the enumeration loop finds
nothing). It's a textbook per-chunk game-thread filesystem stall / micro-hitch source.

### 2. Redundant ~30k-iteration debug loop per generated chunk
`worker/generate.rs` recomputed a full `gs³` pass (chunk_size+1 ≈ 31³ ≈ **29,791 voxel reads**) per
generated chunk purely to tally `dbg_solid/ge1.0/ge1.5/max`, then did a file write per chunk. This
is on the worker thread (off the game thread), but it duplicates iteration `recalc_stress_region_v2`
already performed and runs during the initial-load / stream-in storm when many chunks generate at once.

## What I changed
Removed all four `TEMP VFX diagnostic` blocks (the two file-I/O sites in `queries.rs`, plus the
`gs³` debug loop and its file write in `generate.rs`). Pure deletion of dead instrumentation — **no
behavioral change** to mesh, stress, or VFX results. Net **−52 lines**. `cargo test -p voxel-ffi`:
**125 passed, 0 failed.** Release `voxel_ffi.dll` rebuilt and synced to both UE locations
(Binaries/Win64 + Plugins/VoxelBridge/ThirdParty). **Editor restart required to load the new DLL.**

## Estimated savings (honest ranges — depend on cell counts + disk)
- **Game-thread VFX query (`#1`):** removes one file-open syscall + write per chunk query on the
  game thread. For chunks with **no** over-stress cells (the majority), the diagnostic was ~**80–95%**
  of that FFI call's wall-time → those calls get roughly an order of magnitude cheaper. System-wide
  this is best read as **eliminating a per-chunk game-thread I/O hitch** during streaming/paint
  refresh rather than a steady frame-time %. Biggest qualitative win of the two.
- **Worker chunk-gen (`#2`):** removes a redundant ~30k-iteration pass + file write per generated
  chunk → est. **~5–15% off the VFX-stress-precompute step** per chunk during the load storm
  (worker-thread CPU, not frame time).

## Caveat for your review
These were *active* debug instruments added hours earlier for an open VFX bug (cells classified
`interior`/`field=MISSING`). If that investigation is still live, restore them with
`git revert <this-commit>` — git has the exact lines. I judged committed-`TEMP`-code-on-a-hot-path
to be the clearest "missed opportunity" the task asked me to find; flagging rather than silently
keeping it.

## Other commits reviewed — no action needed
`3f4dcaa` (FluidChunkCache hoist), `2d5f42a` (folded water/lava scan), `aee6a14` (ChunkSampleCache
hoist), `49e12e8` (OrePaint spatial hash) are already solid, well-targeted optimizations. The
`split(...)` refactor commits are behavior-preserving and carry no perf cost.
