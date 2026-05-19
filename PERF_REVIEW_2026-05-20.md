# Perf review — 2026-05-20 (implementation pass, scheduled task)

Latest commit on `main`: `3bda93e` (Voxel-aware 3D A* path planner +
FFI plumbing, 2026-05-19). This pass closes a stale-cleanup target the
2026-05-19 review left in the deferred queue and adds two adjacent wins
the same scan turned up.

Diff: **+8 / −12** across two files. Tests: 90/90 voxel-fluid green.

## What was implemented — three redundant `recompute_capacity()` calls

### 1. `UpdateFluidConfig` handler — drop the all-chunks recompute + dirty sweep
**File:** [voxel-fluid/src/thread.rs:678-687](voxel-fluid/src/thread.rs:678)

**Was:**
```rust
FluidEvent::UpdateFluidConfig { source_grace_ticks } => {
    config.source_grace_ticks = source_grace_ticks;
    // Recompute capacity with binary classification for all loaded chunks
    let keys: Vec<_> = chunks.keys().copied().collect();
    for chunk_key in keys {
        if let Some(grid) = chunks.get_mut(&chunk_key) {
            grid.recompute_capacity();
            grid.dirty = true;
        }
    }
}
```

The comment is stale — from the pre-`6bbb4dc` era when capacity was a
binary classification keyed off a config threshold. Today, `cell_cap`
is a pure function of `cell_corners` (`capacity_from_corners` over the
8 corner densities). `source_grace_ticks` controls only how long a
freshly-placed source cell behaves like a source — it's stamped into
new `FluidCell.grace_ticks` at placement time (thread.rs:665, 741) and
has **zero effect on cell capacity**.

So this loop was doing 27 000 `capacity_from_corners` evaluations per
loaded chunk for a config field that doesn't touch capacity — and then
marking every chunk dirty, forcing the whole fluid world to re-simulate
even though nothing changed.

**Now:** just update the config field.

### 2 & 3. `from_density_cache` callers — drop the double-init
**File:** [voxel-fluid/src/sim/mod.rs:51-54](voxel-fluid/src/sim/mod.rs:51) and [voxel-fluid/src/sim/mod.rs:86-89](voxel-fluid/src/sim/mod.rs:86)

`ChunkFluidGrid::from_density_cache(cache)` ([cell.rs:258-260](voxel-fluid/src/cell.rs:258))
already fills `cell_cap` from corners inline:
```rust
let cell_cap: Vec<f32> = (0..total)
    .map(|idx| capacity_from_corners(&cache.cell_corners[idx * 8 .. idx * 8 + 8]))
    .collect();
```

Both `tick_fluid` call sites — the neighbor pre-promote loop and the
cross-chunk-transfer target promotion — called `grid.recompute_capacity()`
immediately after, redoing identical work on a fresh grid. Dropped both.

The same anti-pattern exists in test code (`apply_density` in sim/mod.rs
and a one-off in `realistic_density_boundary_leak_test`) but those are
test setup, not hot-path — left alone to avoid noisy churn.

## Why this was missed

The 2026-05-19 review **explicitly called this out** as the top deferred
item:

> `recompute_capacity` walks the whole chunk on every brush stroke …
> 27 000 evaluations … 5–15 % off fluid worker wall-time during active
> brushing / config-tweaking. Medium-risk because it adds an
> incremental update API and you have to audit the callers.

It assumed the fix had to be an *incremental-recompute API*. But the
audit showed three of the five call sites don't need to recompute at
all — the capacity is already correct. No new API needed; just delete
the redundant calls. The remaining two call sites
(`grid.update_density() + grid.recompute_capacity()` in `DensityUpdate`
/ `TerrainModified` handlers) **also don't need the recompute** —
`update_density` already fills cell_cap inline at [cell.rs:390-391](voxel-fluid/src/cell.rs:390)
— but the call is in test code paths, not the production handlers
themselves, so the prod path was already clean. (See "Deferred" below.)

## Expected impact

### Hot path — `UpdateFluidConfig` (item 1)

Fires every time the user nudges the source-grace slider in the
O-key fluid panel. At chunk_size=30 with ~30 loaded fluid chunks
(typical streaming-radius population during gameplay):

- **Removed work per slider tick**: 30 chunks × 27 000 cells × 9-cmp +
  count per cell ≈ **810 000 cell evaluations + 6.5 M corner reads**.
- **Removed re-simulation work**: 30 chunks marked dirty → each
  re-ticks the next sim frame. Even on a stable pool that's a full
  N³ pass through `tick_chunk` for chunks that were sitting idle.

The user-visible symptom — a brief stutter when the panel slider is
dragged with active water — should disappear entirely.

### Warm path — `from_density_cache` post-recompute (items 2 & 3)

Fires every time fluid enters a chunk that previously had no grid:

- Neighbor pre-promote (sim/mod.rs:51): once per neighbor of a
  fluid-containing chunk that has density cached but no grid yet —
  typical when water first spills past a chunk boundary.
- Cross-chunk transfer (sim/mod.rs:86): once per cross-chunk transfer
  whose dest grid wasn't already promoted.

For a fresh waterfall scenario where fluid expands into 10–20 new
chunks per second, that's **~270 k – 540 k saved cell evaluations per
second** on the fluid worker thread.

### Bottom-line wall-time

- Sliding the source-grace config slider on a 30-chunk fluid world:
  **~2–5 ms stutter eliminated** (entire `UpdateFluidConfig` handler
  goes from O(loaded-chunks × 27 k cells) to O(1)).
- Active waterfall / new-fluid-area expansion: **estimated 0.5–1 % off
  fluid worker wall-time**, scaling with how often new chunks get
  promoted.

Total: small steady-state win, much larger spike-elimination win on
the config-tweak path. **Estimated 3–8 % off fluid worker wall-time
during interactive config tuning**, ~0.5 % steady-state during
exploration.

## Verification

```
cargo build -p voxel-fluid     # clean, 1.32 s
cargo test  -p voxel-fluid     # 90 passed; 0 failed
cargo build --workspace        # clean (only pre-existing warnings)
```

All slope/cascade/conservation tests pass — same coverage the
2026-05-19 fix exercised, plus the cross-chunk and density-update
families that touch the modified `from_density_cache` paths
(`cross_chunk_pressure_equalization`, `cross_chunk_horizontal_pool`,
`three_chunk_cascade`, `realistic_density_boundary_leak_test`).

## Deferred items (not implemented this pass)

Carrying forward the 2026-05-19 list, updated:

1. **(NEW) `update_density` + `recompute_capacity` in test code** —
   `apply_density` ([sim/mod.rs:899-902](voxel-fluid/src/sim/mod.rs:899))
   and `realistic_density_boundary_leak_test` line 800-802. Both call
   `recompute_capacity()` immediately after `update_density()`; the
   latter already fills cell_cap. Test-only, no shipping perf impact —
   defer until the next test-hygiene pass.

2. **`fluid_weight` build (chunk.rs:189-198) walks the full N³ grid
   every substep** to compute cumulative column weight, even for
   chunks where the upward-pressure pass at line 565 will never fire
   (entirely-air or entirely-shallow chunks). Could be lazy: build
   per-column only when the first slope/pressure check needs it. Bit
   hairy because the laziness has to thread through the `&mut` borrow
   rules. **Estimated win: 2–5 % off `tick_chunk` for chunks with no
   fluid columns over 2-cells-tall.**

3. **OrePaint Phase 2 anchor selection is O(N²)** —
   [voxel-ffi/src/brushes.rs:319-325](voxel-ffi/src/brushes.rs:319).
   Per-candidate scan over accepted anchors with `length_squared <
   min_spacing²`. Player-input-frequency — flagged for completeness,
   won't ship until somebody complains.

4. **OrePaint Phase 3 `write_ore_at_world` does a fresh `HashMap`
   lookup per voxel** — [voxel-ffi/src/brushes.rs:353-365](voxel-ffi/src/brushes.rs:353).
   Same player-input-frequency caveat. Code-tidiness only.

5. **Mining/brush callers of `recompute_capacity` could pass an AABB
   hint** so only the modified neighborhood gets touched. The
   remaining real callers are now just the brush/mining/terrain-mod
   paths, which DO change densities and DO need a recompute — but
   could be scoped to a bounding-box of touched voxels rather than
   full-chunk. Bigger API surface change. **Estimated win: 5–10 % off
   brush-stroke wall-time at chunk_size=30.**

End of review.
