# Perf review — 2026-05-18 (implementation pass)

No new commits on `origin/main` since 2026-05-12 (`267196b`,
OrePaint creative brush). This is the 6th consecutive review with no
new committed work to analyze — the last 3 sessions have drained the
deferred-items queue on their own perf branches.

Today's session implemented the **last "smallish but free" item**
from yesterday's review:

> **`tick_fluid` re-collects keys twice** (lines `sim/mod.rs:34` and
> `:60`) — once before pre-promotion and once after. If pre-promotion
> adds 0 new chunks (the common case), the second collect duplicates
> work. Could collapse with a "did we add any?" flag from the promote
> loop. Smallish — ~one HashMap-key walk per tick — but free.

After looking at the actual semantics, the fix is **better than the
deferred description suggested**: the post-promotion re-collect was
*always* wasted work for newly promoted chunks (they land with
`has_fluid=false, dirty=false` and were always skipped by the
in-loop filter anyway). So we can collapse all three walks into one.

Code is on branch `perf/2026-05-18-tick-fluid-dedup` in worktree
`../voxel-backend-perf-2026-05-18` (commit `4806e98`). Branch is
stacked on yesterday's `perf/2026-05-17-fluid-sparse-index` so the
three fluid-perf branches (2026-05-16, 17, 18) compose cleanly in
order. Not pushed to `main` — left for review per scheduled-task
instructions.

Diff is **+33 / −23 in one file** (`voxel-fluid/src/sim/mod.rs`),
~10 minute review.

## What was implemented

### `tick_fluid` — three key collects → one
**File:** [voxel-fluid/src/sim/mod.rs](voxel-fluid/src/sim/mod.rs:30-90)

**Was:**

```rust
// (1) Full HashMap walk #1
let keys: Vec<_> = chunks.keys().copied().collect();

// (2) Per-key chunks.get(k) probe + filter into a new Vec
let fluid_keys: Vec<_> = keys.iter()
    .filter(|k| chunks.get(k).map_or(false, |g| g.has_fluid))
    .copied()
    .collect();

// (pre-promotion sweep that may insert new chunks)
for fk in &fluid_keys { ... }

// (3) Full HashMap walk #2
let keys: Vec<_> = chunks.keys().copied().collect();

for key in keys {
    let grid = chunks.get(&key)?;
    if !grid.has_fluid && !grid.dirty { continue; }   // skip filter
    let (changed, transfers) = tick_chunk(chunks, key, ...);
    ...
}
```

**Now:**

```rust
// One pass — builds both sets at the same time.
let mut fluid_keys = Vec::new();
let mut tick_keys  = Vec::new();
for (k, g) in chunks.iter() {
    if g.has_fluid { fluid_keys.push(*k); tick_keys.push(*k); }
    else if g.dirty { tick_keys.push(*k); }
}

// (pre-promotion sweep — promoted chunks have has_fluid=false,
//  dirty=false; the prior in-loop skip filter always discarded them
//  so they're correctly absent from tick_keys.)
for fk in &fluid_keys { ... }

for key in tick_keys {
    if !chunks.contains_key(&key) { continue; }       // defensive
    let (changed, transfers) = tick_chunk(chunks, key, ...);
    ...
}
```

**Why it matters.** `tick_fluid` runs every water tick and (when
scheduled) every lava tick. At `chunk_size=30` (the live UE override)
a loaded world holds 1–4k chunks. The previous code was paying:

- **Walk #1** — full `chunks.keys().copied().collect()`, dropped
  before tick_chunk runs.
- **Walk #2 (the filter)** — `keys.iter().filter(|k| chunks.get(k)...)`.
  Same length, but each step does an *extra* HashMap probe to fetch
  the grid back. Effectively a second `chunks.keys()` walk plus one
  hash per element.
- **Walk #3** — full `chunks.keys().copied().collect()` again,
  which then drives the iteration that has the in-loop skip filter.

The new code makes a **single `chunks.iter()` pass** (no second
HashMap probe needed — we already hold the `&grid` ref) and builds
the exact `tick_keys` set the inner loop wants, removing the runtime
skip filter from the hot path.

**Correctness invariant.** Newly promoted chunks intentionally do
*not* land in `tick_keys`. This is **equivalent** to the old code's
behavior because:
- `ChunkFluidGrid::from_density_cache` initializes
  `dirty=false, has_fluid=false`
  (see [voxel-fluid/src/cell.rs:285-303](voxel-fluid/src/cell.rs:285-303)).
- The old in-loop check
  `if !grid.has_fluid && !grid.dirty { continue; }`
  always skipped them.

Promoted chunks still get filled this tick — but via the cross-chunk
transfer apply pass (lines 82-122), which is what was always happening
anyway. They become tickable next tick once the transfer flips their
`dirty`/`has_fluid` flags.

**Defensive `contains_key`** — kept just in case a future code path
removes a chunk between the pre-pass and tick. No current callers do
this. Cost is one HashMap probe per tick_key, same as before.

## Estimated savings

This is a small absolute win, but **strictly free** — no new struct
fields, no incremental-maintenance invariants, no new allocations
beyond a `Vec` that replaces a `Vec`.

| Metric                                            | Was         | Now      | Saving |
|---------------------------------------------------|-------------|----------|--------|
| `chunks.keys()` walks per tick                    | 2 full      | 0        | 100% of that work |
| `chunks.iter()` walks per tick                    | 0           | 1        | (added) |
| Per-element HashMap probes for the filter         | N (chunks)  | 0        | 100% |
| Per-element skip-filter branch in the tick loop   | N           | 0        | 100% |
| Vec allocations per tick                          | 3           | 2        | 1 less |

In wall-time terms, on a ~2k-chunk loaded world this collapses 3
`O(n)` HashMap-bucket walks into 1 single-pass HashMap iter. The
HashMap walk is the dominant per-tick overhead in `tick_fluid`'s
*driver* (i.e., outside the per-chunk `tick_chunk` body which does the
actual work). I'd estimate this is **0.5–1.5% off whole `tick_fluid`
wall-time** on a steady-state world, possibly more on a sparse-fluid
world where `tick_chunk` exits quickly via `has_fluid` short-circuits.

Modest, but it's the **third consecutive day of stacking single-pass
wins on top of `equalize_horizontal` and the per-tick scratch reuse**.
Cumulative fluid-thread wall-time across the 3 unmerged branches is
plausibly **30–50% off** on pool-heavy / quench-active worlds; this
branch is the smallest of the three.

## Verification

- `cargo build -p voxel-fluid` — green, only pre-existing unrelated
  dead-code warnings from `voxel-core`.
- `cargo test -p voxel-fluid` — **90/90 pass**. Covers cross-chunk
  pressure, three-chunk cascade, narrow passages, V-valley, staircase,
  L-shaped containers, source regen, squeeze-excess, lava-water
  solidification — all the scenarios that exercise both pre-promotion
  (chunks growing fluid into neighbors with density-cache-only state)
  and the dirty/has_fluid combinations.
- `cargo test --workspace` — only the same 3 pre-existing
  `voxel-ffi::delta` failures every prior review documented
  (`binary_roundtrip_with_data`, `snapshot_roundtrip`,
  `realistic_chunk_size_roundtrip`, all `density at 101: left 1.01
  right 1.0`). Unchanged by this branch.

## Survey of uncommitted in-flight work on `main`

The user's working tree has substantial uncommitted work (1,289
insertions across 10 files). This is **not** in any commit yet, so
I left it untouched — but flagging perf-relevant observations for
when it lands.

### 1. ⚠️ Correctness bug — `has_ore_material` never invalidated by brushes

**The find_ore_voxels broad-phase will miss ore that OrePaint adds.**

The in-flight work adds `DensityField::has_ore_material` (computed
once by `compute_metadata`) as a chunk-level early reject in
`ChunkStore::find_ore_voxels` (see [voxel-ffi/src/store.rs:621](voxel-ffi/src/store.rs:621)).
This is a great idea — a `has_ore_material=false` chunk skips the
entire 30³ inner scan + 6-neighbor air checks.

But `compute_metadata` is **only called** from:
- generation (worker.rs handle_request, region-gen worm carve)
- save-load (delta.rs, store apply-snapshot paths)
- chunk-boundary sync (store.rs:1453, :1543)

**It is NOT called by `finalize_brush` / `remesh_dirty`** (the path
every brush mutation runs through — see
[voxel-ffi/src/brushes.rs:104](voxel-ffi/src/brushes.rs:104) and
[voxel-ffi/src/store.rs:306](voxel-ffi/src/store.rs:306)).

Concrete failure case:
1. Player loads a chunk that generated with no ore (gem chamber,
   barren marble shelf, etc.) → `has_ore_material = false`.
2. Player uses **OrePaint brush** to paint a vein into that chunk.
3. `finalize_brush` runs, mesh updates, ore is written into voxels.
4. `has_ore_material` is still `false`.
5. Ore tracker queries `find_ore_voxels` → broad-phase rejects the
   chunk → user never sees the painted vein on the tracker.

(Symmetric direction is fine — mining out all ore leaves the flag
true, costing only a wasted full scan but no false negatives.)

**Fix is a one-liner:** call `compute_metadata()` inside `remesh_dirty`'s
Phase 2 serial write-back, or in `finalize_brush` right before the
remesh call. The cost is one full chunk-grid walk per dirty chunk per
brush stroke (cheap — happens once per click, not per voxel).

This isn't a perf issue per se but it interacts with the perf
optimization, so flagging it here.

### 2. `place_mushroom_at_world` — AABB scan with redundant HashMap lookups

[voxel-ffi/src/brushes.rs:~2500](voxel-ffi/src/brushes.rs) — single-mushroom
placer. For `search_radius=3` it scans a 7³=343-voxel AABB around the
click, doing `store.density_fields.get(&key)` for **every** voxel.
With voxels at the boundary of a chunk, the key can change every
voxel along an axis, but typically the inner ~60% of the AABB hits
one or two chunks. Could be improved by:

- Pre-resolving chunk keys for the AABB (1–8 chunks max) and looking
  up the density field once per chunk;
- Doing the inner per-voxel sweep against the held `&DensityField`.

Same pattern as `paint_ore_sphere_voxels`. Saving: ~80–95% of the
HashMap lookups in this function, but it's a per-click brush so
absolute is microseconds. Low priority.

### 3. `find_ore_voxels` — already well-shaped

Reviewed and **no perf concerns**:
- Rayon parallelism via `par_iter().flat_map_iter()`.
- Two-level early reject: chunk-metadata flag (point #1 above) +
  chunk-center broad-phase distance check.
- Per-voxel 6-neighbor air check folded into the same loop that
  computes the visible-centroid offset (single pass, no second walk).
- Sort + truncate at the end — correct, since per-chunk results are
  collected, then a global sort gives nearest-N regardless of chunk
  iteration order.

The 1-voxel border skip (`for z in 1..end` etc.) is documented as a
known omission ("Ores on the absolute chunk boundary are missed;
acceptable for v1"). Fine.

### 4. `prune_destroyed_mushrooms_for_chunks` — well-designed

Called from 8 sites (every mining/flatten/slab/sleep dirty path).
- Empty-input short-circuit.
- Read lock first → only takes write lock if any chunk had a
  destroyed placement.
- Skips writeback when `kept.len() == placements.len()` (nothing
  changed). Good.

## Items still deferred

Carrying forward from prior reviews:

- **`scratch.obsidian_set` / `scoria_set` / `drained_set` should be
  `HashSet<u64>` keyed by a packed `(key, x, y, z)` u64** rather than
  `HashSet<CellAddr>`. ~10–15% off the quench contact-pass + final-
  collect cost. Deferring because `CellAddr` is part of the public
  `QuenchPlan` API and changing it ripples into downstream FFI.

- **Skip per-substep `cells.copy_from_slice` in `tick_chunk`** —
  requires the "rolling stale scratch" invariant be enforced project-
  wide; one bad writer outside `tick_chunk` would corrupt state.
  Worth its own focused session with a dedicated correctness review.

- **`apply_pending_fluid` loop in `thread.rs:735+`** doesn't dedup
  cells, so repeated `AddFluid + PendingFluidLoad` cycles can grow
  the per-chunk vec unboundedly. Low priority — only hit on save-
  load + brush-place.

New item that surfaced from in-flight survey:

- **`has_ore_material` invalidation** — the correctness bug above.
  Must be fixed before the ore-tracker feature ships, otherwise the
  broad-phase silently drops painted ore. One-line fix in
  `remesh_dirty` Phase 2.

## How to review

```bash
cd /c/Users/Shazbot/voxel-backend-perf-2026-05-18
git log --oneline main..HEAD       # 3 commits: 16-fluid-bfs, 17-sparse, 18-dedup
git diff main..HEAD -- voxel-fluid/src/sim/mod.rs
cargo test -p voxel-fluid
```

Or from the main worktree:
```bash
git diff main..perf/2026-05-18-tick-fluid-dedup
```

To merge **all 3 fluid-perf branches in order** (recommended):
```bash
git merge perf/2026-05-16-equalize-bfs           # drain-BFS + quench-BFS
git merge perf/2026-05-17-fluid-sparse-index     # sparse fluid_indices
git merge perf/2026-05-18-tick-fluid-dedup       # tick_fluid single-pass
```

To discard this one:
```bash
git worktree remove ../voxel-backend-perf-2026-05-18
git branch -D perf/2026-05-18-tick-fluid-dedup
```

Note: `perf/2026-05-15-easy-wins` (commit `22e8324`, stress-system +
brushes hot-path tweaks) is also still unmerged. That one is rooted on
`main`, independent of the fluid stack — merge in any order.
