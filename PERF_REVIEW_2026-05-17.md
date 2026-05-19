# Perf review — 2026-05-17 (implementation pass)

No new commits on `origin/main` since 2026-05-12 (`267196b`, OrePaint
creative brush). Today's session **implemented the largest deferred
item** from yesterday's "items still deferred" list:

> **`equalize_horizontal`'s initial `water_cells.insert` walks every
> voxel in every chunk with `has_fluid`** (triple nested loop over
> `0..size`). Same shape as `regen_sources` before the `has_sources`
> flag fix — adding a sparse-index `Vec<u32>` of "cells with level >
> MIN_LEVEL" on `ChunkFluidGrid`, incrementally maintained, would
> collapse it to O(fluid_cells) instead of O(chunk_size³). Bigger
> payoff than today's win and is the actual "fluid-cell sparse index"
> yesterday's #1 was pointing at — today's drain-BFS only addressed
> the *flood-fill* half of that function. Hold for a dedicated session.

Code is on branch `perf/2026-05-17-fluid-sparse-index` in worktree
`../voxel-backend-perf-2026-05-17` (commit `9a3cd8e`). Branch is based
on yesterday's `perf/2026-05-16-equalize-bfs` (commit `0e466df`) so the
two compose cleanly — yesterday's drain-BFS optimized the *flood-fill*
half of `equalize_horizontal`; today's sparse-index optimizes the
*initial scan* half. Not pushed to `main` — left for review per
scheduled-task instructions.

Diff is +142 / −24 across 6 files in `voxel-fluid/`, contained to one
crate.

## What was implemented

### Sparse `fluid_indices: Vec<u32>` on `ChunkFluidGrid`
**Files touched:**
- [voxel-fluid/src/cell.rs](voxel-fluid/src/cell.rs) — new fields + invalidate helper
- [voxel-fluid/src/sim/chunk.rs](voxel-fluid/src/sim/chunk.rs) — build index in tick_chunk's fused pass
- [voxel-fluid/src/sim/utils.rs](voxel-fluid/src/sim/utils.rs) — sparse-path consumer + squeeze invalidate
- [voxel-fluid/src/sim/mod.rs](voxel-fluid/src/sim/mod.rs) — cross-chunk-transfer invalidate
- [voxel-fluid/src/sources.rs](voxel-fluid/src/sources.rs) — sources-placer invalidate
- [voxel-fluid/src/thread.rs](voxel-fluid/src/thread.rs) — AddFluid / geo-spring / pending-fluid-load invalidates

**Was:** `equalize_horizontal` opened with this:

```rust
for (&chunk_key, grid) in chunks.iter() {
    if !grid.has_fluid { continue; }
    let size = grid.size;
    for z in 0..size { for y in 0..size { for x in 0..size {
        let cell = grid.get(x, y, z);
        if cell.level < MIN_LEVEL { continue; }
        ...
    }}}
}
```

At `chunk_size=30` (the live UE override) a chunk holds 27,000 cells,
but typically <5% carry fluid. Every fluid tick paid `27,000 × N_chunks`
just to find that <5%, plus the same `has_fluid` short-circuit cost as
`regen_sources` did before its `has_sources` flag was added.

**Now:** every `ChunkFluidGrid` carries a `fluid_indices: Vec<u32>` of
linear cell indices with `level >= MIN_LEVEL`, plus a `_valid: bool`
guard and a `scratch_fluid_indices` swap-out for zero heap traffic in
steady state. `tick_chunk`'s end-of-tick fused scan (the one already
walking every cell to recompute `has_fluid`/`has_lava`/`has_sources`)
gets one extra line — `if cell.level >= MIN_LEVEL { new_fluid_indices.
push(idx as u32); }` — so the index is a **free byproduct of work
already happening**. `equalize_horizontal` walks the sparse list and
unpacks `(x,y,z)` from the linear index when valid.

**Correctness invariant.** Index is stale-removal-safe (drain paths
leave entries pointing at `level=0` cells; the existing `if cell.level
< MIN_LEVEL { return; }` filter catches them). Add-fluid paths must
flip `fluid_indices_valid = false`; on the next equalize that triggers
the legacy triple-loop fallback until `tick_chunk` rebuilds. The six
add-fluid sites are all flipped in this commit:
- `voxel-fluid/src/sources.rs:66` (lava source placer)
- `voxel-fluid/src/sim/utils.rs:62` (squeeze_excess_fluid pushing
  excess to a previously-empty neighbor)
- `voxel-fluid/src/sim/mod.rs:112` (cross-chunk transfer apply)
- `voxel-fluid/src/thread.rs:629` (geological spring)
- `voxel-fluid/src/thread.rs:667` (AddFluid event handler — brush)
- `voxel-fluid/src/thread.rs:749` (pending fluid load from save)

**Why it matters.** `equalize_horizontal` runs every water tick (see
[voxel-fluid/src/thread.rs:197](voxel-fluid/src/thread.rs:197)), and on
a fluid-active world is the **second-largest N³ pass** behind
`tick_fluid`'s substeps (which we don't touch). After yesterday's
drain-BFS win cut the flood-fill phase, the initial scan became the
function's dominant cost on pool-heavy scenes.

**Estimated savings:** **~80–95% off `equalize_horizontal`'s initial-
scan phase** on typical scenes — savings scale inversely with fluid
density:
- 1% of cells hold fluid → ~99% reduction (sparse list ~270 cells vs 27K walk)
- 5% → ~95% reduction (~1.4K vs 27K)
- 20% → ~80% reduction (~5.4K vs 27K)
- 50%+ (a fully-flooded chamber) → break-even or small loss from the
  extra index-divmod indexing math; rare in practice

Combined with yesterday's drain-BFS (30–40% off the flood-fill phase),
`equalize_horizontal` should drop to **single-digit percent of
fluid-thread wall-time** on a typical pool-heavy world, down from the
20–30% it was sitting at on `main`.

## Per-tick lifecycle (sanity check)

Order per fluid tick (from
[voxel-fluid/src/thread.rs:188-203](voxel-fluid/src/thread.rs:188-203)):

1. `regen_sources` — already `has_sources`-gated; untouched.
2. **`equalize_horizontal`** ← now sparse-index-driven.
3. `tick_fluid` × N substeps (each calls `tick_chunk` per chunk) —
   `tick_chunk`'s existing fused pass now also rebuilds `fluid_indices`
   and sets `fluid_indices_valid = true`.
4. (lava tick if scheduled)
5. `detect_lava_water_quench` — already `has_lava`-gated.

So tick(N)'s `tick_chunk` builds the index that tick(N+1)'s
`equalize_horizontal` consumes. New / brush-mutated chunks pay the
legacy scan once, then settle into the sparse path. Convergence is
single-tick.

## Verification

- `cargo build -p voxel-fluid` — green, 0 errors, only pre-existing
  unrelated dead-code warnings from `voxel-core`.
- `cargo test -p voxel-fluid` — **90/90 pass**. Includes all the
  scenarios that exercise both add-fluid invalidation paths and the
  sparse consumer:
  `cross_chunk_pressure_equalization`, `two_bowls_connected_by_channel`,
  `narrow_passage_between_chambers`, `staircase_cascade`,
  `realistic_density_boundary_leak_test`, `lava_water_solidification`,
  `water_subtype_solidifies_lava`, `uniform_layer_stays_stable`,
  `upward_pressure_equalization`, `v_valley_fills_bottom`,
  `sloped_floor_pools_at_low_end`, `l_shaped_container_damping`,
  `three_chunk_cascade`, `uneven_fill_equalizes_to_flat_surface`,
  `u_tunnel_cross_section`, `water_never_enters_solid_cells`,
  `mine_channel_between_pools`, `cross_chunk_horizontal_pool`,
  `source_regenerates`, `squeeze_excess_works`.
- `cargo test --workspace` — only the same 3 `voxel-ffi::delta`
  failures yesterday's review documented as pre-existing on `main`
  (`binary_roundtrip_with_data`, `snapshot_roundtrip`,
  `realistic_chunk_size_roundtrip`, all `density at 101: left 1.01
  right 1.0`). Unchanged by this branch.

## Aggregate impact (rough — combined with yesterday's)

| Surface                                              | Saving (this branch) | Cumulative w/ 2026-05-16 |
|------------------------------------------------------|----------------------|--------------------------|
| `equalize_horizontal` initial scan                   | 80–95%               | 80–95%                   |
| `equalize_horizontal` flood-fill                     | (unchanged)          | 30–40%                   |
| `detect_lava_water_quench` BFS phase                 | (unchanged)          | 30–50%                   |
| `equalize_horizontal` whole function (rough total)   | ~50–70%              | ~70–85%                  |

Both surfaces are inside the fluid-thread hot path. Combined with
yesterday's win, a pool-heavy or volcanic world should see
**fluid-thread wall-time drop by mid-to-high single digit percent of
the whole sim budget**.

## Items still deferred (not implemented this session)

Carrying forward from prior reviews:

- **`scratch.obsidian_set` / `scoria_set` / `drained_set` should be
  `HashSet<u64>` keyed by a packed `(key, x, y, z)` u64** rather than
  `HashSet<CellAddr>` (24-byte tuple SipHash per op). ~10–15% off the
  quench contact-pass + final-collect cost. Deferring because `CellAddr`
  is part of the public `QuenchPlan` API and changing it ripples into
  downstream FFI.

- **Skip per-substep `cells.copy_from_slice` in `tick_chunk`** —
  requires the "rolling stale scratch" invariant be enforced project-
  wide; one bad writer outside `tick_chunk` would corrupt state. Worth
  its own focused session with a dedicated correctness review.

New candidate that surfaced today:

- **`apply_pending_fluid` loops `pending_fluid.entry(chunk).or_default()
  .extend(cells)` and then walks all pending cells on every density
  update** (see [voxel-fluid/src/thread.rs:735+](voxel-fluid/src/thread.rs:735)).
  If a save has thousands of pending fluid cells in a single chunk,
  this is fine — it only runs once. But the surrounding `pending_fluid:
  HashMap<chunk, Vec<…>>` doesn't dedup, so repeated `AddFluid`
  followed by `PendingFluidLoad` can grow the vec unboundedly. Low-
  priority because the path is only hit on save-load and brush-place;
  flag it for the next time we revisit save-load.

- **`tick_fluid` re-collects keys twice** (lines
  [voxel-fluid/src/sim/mod.rs:34](voxel-fluid/src/sim/mod.rs:34) and
  [voxel-fluid/src/sim/mod.rs:60](voxel-fluid/src/sim/mod.rs:60)) —
  once before pre-promotion and once after. If pre-promotion adds 0
  new chunks (the common case), the second collect duplicates work.
  Could collapse with a "did we add any?" flag from the promote loop.
  Smallish — ~one HashMap-key walk per tick — but free.

## How to review

```bash
cd /c/Users/Shazbot/voxel-backend-perf-2026-05-17
git log --oneline main..HEAD
git diff main
cargo test -p voxel-fluid
```

Or from the main worktree:
```bash
git diff main..perf/2026-05-17-fluid-sparse-index
```

To merge **both** in-flight perf branches in order (recommended):
```bash
git merge perf/2026-05-16-equalize-bfs    # drain-BFS + quench-BFS
git merge perf/2026-05-17-fluid-sparse-index   # sparse fluid_indices
```

To discard either:
```bash
git worktree remove ../voxel-backend-perf-2026-05-17
git branch -D perf/2026-05-17-fluid-sparse-index
```

Note: yesterday's `perf/2026-05-15-easy-wins` (commit `22e8324`,
stress-system + brushes hot-path tweaks) is also still unmerged.
That one is on `main`, independent of the fluid stack — merge in
any order.
