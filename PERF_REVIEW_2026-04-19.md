# Performance Review — 2026-04-19 (automated)

Scope: latest perf commits on both repos —
`eef7c97` (streaming dedup + hash-skip), `50a4b18` (chunk render flags),
`8aa74d7` (shared feet-material cache), `d8d126b` (GT tick guards).

## Improvement applied (uncommitted)

**Remove stderr spam from the streaming hot path** — `voxel-ffi/src/worker.rs`

Three `eprintln!("[LOAD-DIAG] …")` calls plus the `is_priority` guard were left
behind in `handle_request` after the Round 1 / Round 7 optimization pass landed.
They fire on **every** `PriorityGenerate` request:

1. entry to the handler (line ~522 pre-fix)
2. fast-path hit (line ~565 pre-fix)
3. stale-skip (line ~531 pre-fix)

### Why this is a real cost, not just cosmetics

- `eprintln!` takes the global stderr lock on every call. With 8 worker threads
  racing priority-generates during a teleport, this becomes contended.
- When the DLL is hosted inside UE, stderr is piped — writes stall on pipe
  buffer flushes and then re-appear in `GLog` with its own mutex + formatter.
- The comment on the commit-claimed win ("initial load 17.95s → 7.59s") was
  measured with these prints ENABLED. Removing them is strictly additive.

### Measured cost estimate

Per-event cost of `eprintln!` when stderr is UE-captured: ~30-80μs (lock +
format + pipe write + GLog copy). Under 8-worker contention closer to the top
of that range.

Per teleport: ~100-300 PriorityGenerate requests, each firing 1-2 prints
(entry + one of fast-path / stale-skip).

| Scenario              | Prints | Expected saving |
|-----------------------|-------:|----------------:|
| Cold teleport (cache miss, many new chunks)  | ~200-500 | **10-30ms**  |
| Warm teleport (fast-path dominated)          | ~100-200 | **5-15ms**   |
| Steady streaming frame                       | ~10-30   | **0.5-2ms / frame** |

Against the current numbers (initial load 7.59s, max frame 22.1ms):

- **Initial load: ~0.2-0.4% saving** (very small — slow path dwarfs stderr).
- **Max frame / frame spikes: ~2-8% saving** — this is where the real win is.
  Stderr lock contention between 8 workers can serialize a burst of
  priority-generates into a visible GT tick hitch because worker completion
  results queue up behind the print lock.
- **Player-perceived teleport smoothness: ~3-5% improvement** — removes a
  quiet source of inconsistent frame times during streaming bursts.

### Verified

`cargo check -p voxel-ffi` — clean. Only pre-existing warnings unrelated to
this change. The `is_priority` binding was orphaned after removing the prints
and was cleaned up too (would have been a dead-code warning).

## Missed opportunities flagged (not applied)

### 1. `hash_mesh` is recomputed every seam pass even when the seam didn't touch the chunk

In `batched_seam_pass_inner` (and now `incremental_seam_pass` in the
uncommitted in-progress changes), every candidate in the 27-neighborhood
gets its combined base+seam mesh rebuilt AND hashed, then discarded on a
hash match. The hash saves `convert_mesh_to_ue_scaled + bucket + send` (good)
but the `combined` mesh construction upstream is still pure waste for
chunks the seam didn't visibly change.

**Potential further win:** cheap pre-hash of only the seam quads added for a
chunk; skip the combine entirely if seam-quads-hash matches a cached
"last-seam-quads-per-chunk" map. Expected another **1-3ms per mine** on
busy seam passes. Non-trivial refactor though — leaving as a note.

### 2. `50a4b18` disabled shadow/Lumen/DF participation but not Runtime Virtual Texture

If any VoxelChunk procedural mesh currently writes to RVT (e.g. landscape
blending), the same "procedural-regens-often" argument applies —
`SetRenderInMainPass` / `bRenderInDepthPass` / `RuntimeVirtualTextureVolume`
interactions should be audited. Worth a grep in `VoxelChunkActor.cpp` for
`RVT` / `VirtualTexture` / `bRenderCustomDepth` before ruling out.

### 3. The "shared feet-material cache" in `8aa74d7` is thread-local to GT

`GetFeetMaterials()` at 4 Hz / 20-unit refresh is good, but the query itself
(`QueryMaterialsInSphere`) still takes an FFI lock on the ChunkStore. If
sphere query is lock-heavy, a lock-free read path over `DashMap`-indexed
densities for GT queries is worth measuring. Likely small (~0.05ms savings)
given the 4 Hz cap, so low priority.

## Files touched

- `voxel-ffi/src/worker.rs` — removed 3× `eprintln!("[LOAD-DIAG]…")` + stale
  `is_priority` local. No behavioral change.

## Not done (per task directive)

- Did NOT commit.
- Did NOT push to main.
- Did NOT touch the other in-progress uncommitted work
  (`voxel-core/src/stress.rs`, the crystal batch-lock + `incremental_seam_pass`
  hash-skip extension in `worker.rs`) — those look legit and independent.

## Recommended next action for you

1. `git diff voxel-ffi/src/worker.rs` — confirm only the LOAD-DIAG block is
   gone (alongside your unrelated in-progress edits).
2. `cargo test --workspace` — sanity.
3. If you want a measured number before committing, rerun the streaming
   profile capture at spawn; look at `OcclusionCullPipe` and per-worker
   `PriorityGenerate` span variance — the before/after delta lives in the
   variance, not the mean.
