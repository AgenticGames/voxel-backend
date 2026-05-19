# Perf review 2026-04-20 (night session) — seam gap fix + mining lock contention

## TL;DR for your morning check

Three things shipped in commit `404e1ac`:

1. **Seam gaps on initial load — fixed.** Three separate races caused UE to end up showing base-only meshes with no seams at chunk/region boundaries. Gone now. Visually verify on first boot.
2. **Mining lock contention — fixed (fix A).** Crystal data pre-fetched in one read lock instead of N per-chunk acquires. Measured `store_read_wait max: 150 ms → 2 ms` (-99 %) on cold load.
3. **Mining crystal recompute — narrower (fix B).** Only recomputes crystals for chunks that had an actual air↔solid material flip, not boundary-sync neighbours.

Current committed state is **1 commit ahead of origin/main** — not pushed. Let me know if you want it pushed.

## What went down tonight

### Seam gaps

Started the session on a perf plan (fix A + B). You reported seam gaps on initial load. Put perf aside, chased the gaps.

Three bugs, all introduced by the streaming work in `eef7c97` and `f0760a9`:

1. **Cross-region density sync + worm backward-carve** (`worker.rs:850`, `worker.rs:919`): these paths remeshed boundary chunks and sent base-only meshes that wiped seams via UE's `ClearAllMeshSections`, with no seam-pass follow-up. **Fix**: dropped the base-only sends; `batched_seam_pass` (newly added after each loop) is the sole sender.
2. **Main gen path base-only send** (`worker.rs:1150`): same story but cross-worker. Worker 2's incremental seam pass for chunk A fires while worker 1 is still mid-gen; W2 sends combined_A with `hash_combined`; W1's base-only lands second, wipes seams on UE; W1's subsequent incremental computes combined_A with matching hash → skipped. **Fix**: record `hash(base)` into `last_sent_mesh_hash` before the send so W1's incremental sees a different stored hash and actually sends.
3. **Stale hash across unload/reload** (`store.rs:unload()`): `last_sent_mesh_hash` entries persisted after chunks unloaded, so a re-streamed chunk's seam pass falsely hash-matched and skipped. **Fix**: one-line `remove()` in `unload()`.

Net effect on load perf: small win, not a regression. Cold-load wall time stayed in its noise band (~7.0 - 7.3 s).

### Fix A — mining store RwLock contention

The 2026-04-18 research agent flagged `store_read_wait max = 568 ms` on mining as the biggest mining bottleneck. Root cause: `retrieve_crystal_data` in the seam-pass send loop acquires a fresh `store.read()` per target, 8 workers × N chunks = serious contention.

**Fix**: in both `incremental_seam_pass` and `batched_seam_pass_inner`, pre-fetch all crystal placements into a `HashMap` under **one** read lock before the send loop. Loop then reads from the map.

**Measured on initial_load profile** (not mining — I struggled to get the profiler session to stay alive during quiet mining periods, see below — but same code paths fire during cold load):

| Metric | Before | After A+B | Δ |
|---|---:|---:|---:|
| Wall time | 7.65 s | 7.34 s | −4 % |
| `store_read_wait` avg | 1.74 ms | 0.22 ms | **−87 %** |
| `store_read_wait` max | 150.75 ms | 2.02 ms | **−99 %** |
| `store_write_wait` avg | 3.51 ms | 0.99 ms | −72 % |
| `store_write_wait` max | 221.52 ms | 48.69 ms | −78 % |
| `seam_pass` total | 2202 ms | 708 ms | **−68 %** |
| `seam_convert` total | 157 ms | 73 ms | −54 % |
| UE `ProcessResults` | 630 ms | 427 ms | **−32 %** |
| Worker utilization | 73–76 % | 78.7–79.2 % | +5 pp |

The lock-wait collapse is the real prize. Max wait went 150 ms → 2 ms — workers no longer stall waiting for crystal-data read locks.

### Fix B — skip redundant crystal recompute

Mining re-computes crystal placements for every dirty chunk after a mine. But the boundary-sync extras (added to dirty set so neighbours remesh correctly) have density tweaks without any air↔solid flip — their crystal layout is unchanged, recompute is wasted.

**Fix**: `mine_sphere`/`mine_peel` now return a `MineOutcome` struct that separates `flipped_chunks` (real material change) from the full dirty set. Crystal recompute iterates only `flipped_chunks`.

Mechanically correct. **Not independently measured** — see next section for why the mining benchmark didn't capture cleanly.

### Mining benchmark

Built a scripted pixel-server harness: sprint → teleport → look_down → hold LMB. Added two UE-side bits to make this work:

- `hold_mine` JSON command (drives `UVoxelMiningComponent::HoldMine()` once per tick while timer > 0).
- `IsClaudeHoldMineActive()` flag the character's LMB polling checks, so the cancel-mine-when-LMB-not-down path doesn't torpedo the simulated mine.

Mining *works* (player consistently accumulated 40+ ore per run), but the profiler session auto-ends after 3 quiet frames once streaming work drops to zero, and mining doesn't push `UE::PendingRequests` (only chunk-gen does). So the session dies right after mining's first mesh update and emits a near-empty report. Auto-burst sessions do fire during surrounding sprint/teleport, but not reliably during sustained mining alone.

Workaround used: compare the `initial_load` profiles (same seam-pass code path, same lock contention dynamics, high sample count). Improvements above are from that comparison. If you want proper mining numbers, we need either (a) a profiler mode that ignores quiet-frame auto-end while a session is explicitly tagged, or (b) read Rust-side mining debug logs directly (`mine_debug.txt`, `gen_perf.txt`).

### Fix C — streaming base_density per-chunk — deferred

This is the big structural one: make phase 1 (`base_density`) emit per-chunk results so fast-path chunks flow earlier, instead of workers blocking the full 1.2 s for the whole region to finish.

Looked hard. Key issue: phase 2 (cavern detection) and phase 4 (worm carving) need region-wide context. Phase 3 (worm planning) runs across all cavern centres. Phase 4 carves density across multiple chunks in the region based on those plans. A chunk can't be "finalised" until the worms that touch it have carved. Per-chunk streaming BEFORE worm carving produces meshes missing cavities.

Today's config has `worms_per_region = 0`, so phase 4 is a no-op and the risk is theoretical — but the fix has to work generally. Proper implementation needs either:
- Per-chunk readiness signal + modified fast-path check (allow `has_density` fast-path even when region isn't fully "done"), plus re-mesh passes for chunks that later get worm-carved.
- OR a two-pass design: emit stub meshes fast, regenerate after full region is ready.

Either is several hours of careful work. Didn't want to rush it overnight right after fighting seam-gap races for an hour. Parking it.

**Lesser wins in the same area, if you want a follow-on:**
- Parallelise the backward-carve remesh loop and cross-region sync remesh loop (both currently serial `for` loops). Small gain in current config (worms=0 means backward-dirty is empty), but easy.
- Queue-priority fast-path: when a fast-path-eligible chunk is waiting behind a slow-path worker holding the region mutex, let another worker pick it up. Moderate effort, moderate gain.

## Files touched

- `voxel-ffi/src/worker.rs` — seam-gap fixes, fix A crystal pre-fetch, fix B call-site adaptation
- `voxel-ffi/src/mining.rs` — `MineOutcome` struct, expose `flipped_chunks`
- `voxel-ffi/src/store.rs` — `unload()` clears `last_sent_mesh_hash`
- UE: `VoxelClaudeAutomation.cpp`, `VoxelWorldSubsystem.h`, `Mithril2026Character.cpp` — `hold_mine` command + LMB-override plumbing (UE-side, won't affect shipping builds)

## Verification done

- `cargo build -p voxel-ffi --release` — clean.
- `cargo test -p voxel-ffi --lib --release` — **30/30 pass**.
- `cargo test --workspace` — 99/100 pass. Single failure `voxel-gen zones::mega_blueprint::blueprint_has_expected_structure` is pre-existing on HEAD, unrelated to these changes (same test was flagged in 2026-04-18 perf review).
- Two matched pixel-server cold-load runs, same seed, same config — numbers in the table above.

## Commits

- `404e1ac` — the lot (seam gaps + fix A + fix B). **Not pushed.**
- `f0760a9` — the earlier streaming perf commit that introduced the seam-gap races (pushed earlier today).

## What to check visually

1. **Initial load seams.** Play the game for a few minutes, walk around. You said most seams triggered with my earlier half-fix; should now trigger 100 %.
2. **Mining feels the same or better.** No functional change — just faster under contention.
3. **No new weirdness.** The `record hash(base)` line at `worker.rs:1150` is the only "subtle" change in steady-state behaviour. If UE starts seeing stale meshes in some edge case, that's the first place to look.
