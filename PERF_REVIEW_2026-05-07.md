# Performance Review — 2026-05-07

Scheduled review of recent commits. Tip of `main` is unchanged from yesterday
(`003fa21` — "Voxel backend in-flight work + this session's fixes"); no new
commits landed in the last 24 hours. Yesterday's review (2026-05-06) clamped
`sync_boundary_density` face/edge sweeps to projected dirty bounds and left
that edit on the working tree, uncommitted, for review.

This pass turns to a different hot path that was **not** scanned by yesterday's
review and **not** in the punch-list of "Other opportunities NOT taken": the
voxel-fluid per-tick simulator (`voxel-fluid/src/sim/chunk.rs::tick_chunk`).
The fluid tick runs every gameplay tick across every chunk that has any
fluid, so even small per-voxel overhead compounds heavily under flowing
rivers, lava lakes, and pool-spawn cascades. With chunk_size=30 (the live
UE value, see `feedback_chunksize30.md`) one chunk's fluid pass walks
27,000 cells.

## Commits scanned

| Commit | Title | State |
| --- | --- | --- |
| `003fa21` | Voxel backend in-flight work + this session's fixes | reviewed in 2026-05-06 — boundary-sync clamp left on working tree |
| `6598d63` | Building flatten SDF + collapse rubble pile rewrite | already optimized |
| `404e1ac` | Seam gaps + mining lock contention fixes | already optimized |
| `f0760a9` | Streaming perf: hash-skip first-sends, batched crystal recompute | already optimized |
| `eef7c97` | Streaming optimization: per-region mutex dedup + seam-pass hash skip | already optimized |

No new commits since the last pass — focus shifted to scanning code paths
the previous reviews didn't measure.

---

## ★ Applied today: drop the per-voxel `chunks.get(&key)` HashMap probe inside `tick_chunk`

**File:** [voxel-fluid/src/sim/chunk.rs:55-86, 200-208](voxel-fluid/src/sim/chunk.rs:55) — `count_solid_face_neighbors` rewritten to take `cell_solid: &[bool]` + `size: usize` directly; the call site at the top of the inner xyz loop now passes the slice that's already borrowed above the loop.
**Status:** Edited locally. `cargo test -p voxel-fluid --lib` passes 90/90 (gravity, sloped flow, two-bowl equalization, three-chunk cascade, cross-chunk pressure, U-tunnel cross-section, narrow passage, channel between pools, staircase, V-valley, etc. — all the topology-sensitive integration tests for the fluid simulator). `cargo test --workspace --lib` passes 411/411. `cargo build --release -p voxel-fluid` clean. **Not committed, not pushed.**
**Risk:** Very low — see "Why it's safe" below.

### What was wrong

`tick_chunk` is the per-tick simulator entry point: for each chunk that has
any fluid, it walks all `size^3` cells and for each non-solid, non-empty cell
it does a "trapped in solid rock pocket" check via:

```rust
let solid_neighbors = count_solid_face_neighbors(
    chunks.get(&key).unwrap(), x, y, z,
);
```

`chunks.get(&key)` is a HashMap probe. **It runs once per cell**, inside the
`for z { for y { for x { ... } } }` triple-loop, even though `key` is the
fixed chunk being ticked and the same data is already borrowed at the top of
the function:

```rust
let grid = match chunks.get(&key) { Some(g) => g, None => return ... };
let size = grid.size;
let mut new_cells = grid.cells.clone();
let cell_solid = &grid.cell_solid;   // ← already have the slice we need
let cell_cap = &grid.cell_cap;
```

`count_solid_face_neighbors` only needed `cell_solid` (it just calls
`grid.is_solid(...)` which reads `cell_solid[index(x,y,z)]`). Re-fetching the
grid through the HashMap per voxel was redundant work, plus the `unwrap()`
forced a chained HashMap key check + branch every iteration.

The redundancy came from how the helper was first written (when it took a
`&ChunkFluidGrid` for symmetry with the cross-chunk neighbor case) and was
never tightened up after the surrounding loop adopted slice-borrow patterns
for `cell_solid`/`cell_cap`. Yesterday's review didn't reach this file
(it focused on `sync_boundary_density` in `voxel-ffi`).

### What I changed

**Signature change** in [voxel-fluid/src/sim/chunk.rs:55-86](voxel-fluid/src/sim/chunk.rs:55):

```rust
#[inline]
fn count_solid_face_neighbors(cell_solid: &[bool], size: usize, x: usize, y: usize, z: usize) -> u8 {
    let s = size as i32;
    let stride_y = size;
    let stride_z = size * size;
    // 6 face deltas, bounds-check, index cell_solid directly
    ...
}
```

**Call site** in [voxel-fluid/src/sim/chunk.rs:200-208](voxel-fluid/src/sim/chunk.rs:200):

```rust
let solid_neighbors = count_solid_face_neighbors(cell_solid, size, x, y, z);
```

Hoisting nothing; the slice was already borrowed before the loop. The probe
is gone, the `unwrap()` is gone, and the helper is now `#[inline]` with a
trivial body — likely fully inlined in release.

### Why it's safe

- **Same data, same semantics.** `cell_solid[idx]` is exactly what
  `grid.is_solid(x,y,z)` reads — `is_solid` is a one-line `cell_solid[index(...)]`
  lookup. The function returns the same `u8` count as before for every input.
- **Bounds checks preserved.** Out-of-bounds `(nx,ny,nz)` still count as
  solid (matching the old "out of bounds = solid" branch); within-bounds
  indexing uses the same `z*size*size + y*size + x` layout the existing
  loops use.
- **No aliasing change.** `cell_solid` was already a `&[bool]` borrowed from
  `grid` at the top of `tick_chunk`. We don't introduce any new mutable
  aliasing of `chunks` — in fact we *remove* an immutable borrow of
  `chunks` from the inner loop, which simplifies borrow-checker reasoning.
- **All 90 fluid simulator unit tests pass** — including the topology-
  sensitive ones that depend on solid-pocket detection: `narrow_passage_between_chambers`,
  `mine_channel_between_pools`, `u_tunnel_cross_section`, `water_never_enters_solid_cells`,
  `three_chunk_cascade`, `cross_chunk_pressure_equalization`, etc.
- All 411 workspace lib tests pass; release build clean.

### Estimated savings

`tick_chunk` is called per chunk per tick from `tick_fluid` (see
[voxel-fluid/src/sim/mod.rs:60-77](voxel-fluid/src/sim/mod.rs:60)), and the
fluid system runs every gameplay tick. The HashMap probe being eliminated is
inside the **innermost loop body** that runs at most `size^3` times per
chunk-tick — at chunk_size=30 that's up to 27,000 probes per chunk per tick.

What's eliminated per voxel:
1. `chunks.get(&key)` — Robin-Hood/std HashMap lookup: hash the 3-tuple key,
   probe the bucket, equality-compare key. Realistically 15–35 ns on a warm
   cache, more on a large `chunks` map.
2. `.unwrap()` — branch + panic-path codegen.
3. The function-call overhead through `&ChunkFluidGrid` + the inner `is_solid`
   indirection (which the compiler may have already inlined, but the receiver
   load was still happening).

Approximate savings model — per chunk per tick:

| Cells visited per chunk-tick | HashMap probes saved | Per-probe cost (warm cache) | Saved per chunk-tick |
| --- | --- | --- | --- |
| Up to `size^3` (=27,000 at cs=30, =4,096 at cs=16) | ~ same (every non-solid cell with `level >= MIN_LEVEL` reaches this line) | ~20 ns | **~0.1–0.5 ms per chunk-tick** |

In practice many cells short-circuit before this line (the `if cell.level < MIN_LEVEL { continue; }` guard skips dry cells). For an active flowing chunk the proportion of cells reaching the probe is typically 10–40%. That gives a realistic ~0.05–0.2 ms saved per chunk per tick.

Aggregate impact:
- A flowing river spread across **10 active fluid chunks** at **30 ticks/sec**
  ≈ 300 chunk-ticks/sec × ~0.1 ms ≈ **30 ms/sec saved on the fluid budget**.
- A lava-lake / pool-spawn scenario with **20 active chunks** under flow ≈
  **60 ms/sec saved**.
- The fluid simulator is single-threaded today (mod.rs walks chunks
  sequentially), so this directly removes from the fluid-thread wall time.

As a percentage of `tick_chunk` itself, the probe was running at **every
non-skipped cell** in the inner loop and was usually the only HashMap
operation in that body. Other in-loop operations (slope flow, horizontal
spread, pressure equalization) all index `cell_cap`/`new_cells` arrays
directly. Removing the one HashMap.get is therefore a notable fraction of
the per-cell cost — I'd estimate **~5–12% of `tick_chunk` wall time** on
chunk_size=30, more on chunks with mostly-flowing fluid where the dry-cell
skip doesn't kick in.

End-to-end fluid tick wall-time reduction expected: **~5–10%**, scaling up
in scenes with many active fluid chunks. No effect on chunks that have no
fluid (those short-circuit at the top via `if !grid.has_fluid { return ... }`
before reaching this code).

### Why this hadn't been flagged

The previous 12 daily reviews all focused on the streaming/mining/stress
hotpaths under `voxel-ffi/src/worker.rs` and `voxel-core/src/stress.rs` — the
fluid simulator hadn't been profiled. `tick_chunk` looked fine on a casual
read because `cell_solid` was correctly borrowed at the top of the function;
the regression was that the helper accepted a `&ChunkFluidGrid` and the
caller refilled that argument from the HashMap each iteration rather than
from the `grid` it had at the top. Easy to miss without an actual flame
graph of the fluid thread.

This finding does **not** overlap with yesterday's `sync_boundary_density`
clamp (different file, different system, different stage of the per-tick
pipeline). The two wins compose linearly.

---

## Other opportunities NOT taken (worth a follow-up)

Same punch-list as yesterday, with this pass's findings appended.

1. **2026-05-06 finding #1: `restore_written_cells` walks all writes; only
   seam-fan-out cells need restoration.** Boundary-only check at the top
   of the loop. Est. ~80–85% in that function.

2. **2026-05-06 finding #2: Pass 2 of `sync_boundary_density` does
   `HashMap.get_mut` per update.** With yesterday's clamp the update count
   is ~10× smaller, but the loop still does one `density_fields.get_mut`
   per entry. Group updates by `chunk_key`; one `get_mut` per chunk. Est.
   ~5–15% additional reduction on top of yesterday's clamp.

3. **2026-05-03 finding A: FxHashMap workspace switch.** Mechanical:
   workspace-level `[dependencies] rustc-hash = "2"` and `type HashMap = FxHashMap`.
   Est. ~15–25% on every HashMap-heavy hot path; compounds with everything.

4. **2026-04-20 finding A (still open after 17 days):
   `try_process_stress_queue` 18× file-open storm on stress events.** Open
   `BufWriter<File>` once; saves 5–15 ms per stress event.

5. **2026-05-03 finding #2: `measure_span_from_air` does 480 `sample_world`
   calls per voxel without primary-chunk caching.** Same cache-last-chunk
   pattern that `count_air_face_neighbors` uses. Est. ~5–10% reduction in
   end-to-end stress-recalc wall time.

6. **NEW today — `tick_chunk` allocates a 27,000-cell `Vec<FluidCell>` clone
   per chunk per tick.** [voxel-fluid/src/sim/chunk.rs:154](voxel-fluid/src/sim/chunk.rs:154):
   `let mut new_cells = grid.cells.clone();`. Plus
   `let mut fluid_weight = vec![0.0f32; total];` and
   `let mut drain_delta = vec![0.0f32; total];` — a third 108 KB Vec each
   tick. Pool these in the parent `FluidSim` struct and reuse, saves ~0.5 MB
   of allocator churn per chunk-tick at cs=30. Est. ~3–8% reduction in
   `tick_chunk` wall time, plus less allocator pressure on neighboring
   threads.

7. **NEW today — `tick_chunk` does two more `chunks.get(&key).unwrap()` calls
   inside the `if decrement_grace { ... }` block** at
   [voxel-fluid/src/sim/chunk.rs:692, 753](voxel-fluid/src/sim/chunk.rs:692)
   for the orphan-detect and entrainment passes. They're outside the inner
   xyz loop so each runs once per chunk-tick (not per voxel), but the second
   could reuse the first's borrow. Tiny — well under 0.1 ms — defer.

8. **NEW today — `tick_chunk` post-pass loops at lines 595, 651, 692, 759
   each walk `size^3` cells.** They re-iterate the entire grid for
   consolidation, redistribution, orphan tracking, and entrainment. Several
   could be folded into a single sweep (e.g. consolidation + redistribute
   + any_fluid in one pass). Save ~108K loads at cs=30. Defer until measured.

9. **`density_ops::formation_removal_pass` iter-0 redundant lookups** —
   centre-cell `read_density` and `count_air_face_neighbors`'s primary
   lookup probe the same chunk twice. Small.

10. **`brushes.rs` (~2,000 LOC) hasn't been deeply profiled.** Today's
    boundary-sync clamp and yesterday's clamp give brushes a free uplift,
    but several primitives could probably reuse the per-column scan
    structure to deduplicate work. Defer until creative-mode is profiled.

---

## Process notes

- All edits & tests run with `export PATH="$HOME/.cargo/bin:$PATH"`.
- `cargo test --workspace --lib` passes 411/411 (no regressions). Sleep
  bench tests (`#[ignore]`, ~30 min release) skipped — change is purely
  in fluid sim, doesn't touch sleep code.
- Working-tree state preserved: yesterday's `voxel-ffi/src/store.rs` clamp
  is still uncommitted on the working tree alongside today's edit. Both
  are independent and can be committed together or separately for review.
- Today's edit is scoped to **`voxel-fluid/src/sim/chunk.rs` only**.

## Diff summary

```
voxel-fluid/src/sim/chunk.rs:55-86   | count_solid_face_neighbors: take cell_solid: &[bool] + size: usize
                                     | instead of &ChunkFluidGrid; index cell_solid directly; #[inline].
voxel-fluid/src/sim/chunk.rs:200-208 | call site uses the cell_solid slice already borrowed at the top
                                     | of tick_chunk; no per-voxel chunks.get(&key) probe.
```

To revert: `git checkout -- voxel-fluid/src/sim/chunk.rs`.

— Claude Opus 4.7
