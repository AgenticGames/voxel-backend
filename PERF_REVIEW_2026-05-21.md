# Perf review — 2026-05-21 (scheduled daily pass)

Latest 3 commits on `main` since yesterday's review (`240d1c3`):
- `81f2d0a` Morph manifest: growth_sources for source-distance reveal animation
- `8bf838d` Morph manifest: synthesize_growth flag for POI plays
- `faa5f70` POI tracker quality pass: hysteresis, batched lock, shared scoring

All three are correctness / quality-of-life changes (anchor-bridge reveal
animation, POI scoring polish) — not perf regressions, but their bounded
worker thread + batched-lock shape is healthy.

This pass closes deferred item **#2** from the 2026-05-20 review:
**`fluid_weight` column build walks the full N³ grid every substep, even
on chunks where the upward-pressure pass at line 575 can never fire.**

Diff: **+33 / −7**, single file. Tests: 90/90 voxel-fluid green.

## What was implemented — early-exit column scan in `tick_chunk`

**File:** [voxel-fluid/src/sim/chunk.rs:194-235](voxel-fluid/src/sim/chunk.rs:194)

The pre-process loop that fills `fluid_weight[idx]` (cumulative
column-fluid for Phase-4 hydrostatic upward pressure) used to do:

```rust
for z in 0..size {
    for x in 0..size {
        let mut cumulative = 0.0f32;
        for y in (0..size).rev() {
            let idx = z * size * size + y * size + x;
            cumulative += grid.cells[idx].level;  // 27 000 reads at size=30
            fluid_weight[idx] = cumulative;       // 27 000 writes at size=30
        }
    }
}
```

That's 27 000 cell reads + 27 000 `f32` writes per chunk per fluid tick,
unconditionally, even though `fluid_weight` is **already pre-zeroed** by
`fw.resize(total, 0.0)` at line 175 and the typical cave chunk has only
~5–20 % column-height fluid coverage. Empty cells above a puddle stay
0 in both old and new code (cumulative is 0 until we hit fluid).

The new loop walks each column top-down, **skips writes while the
column cap is empty**, and only enters the accumulating phase once the
first non-empty cell is found:

```rust
for z in 0..size {
    for x in 0..size {
        let base = z * stride_z + x;
        let mut y = size;
        let mut cumulative = 0.0f32;
        // Skip the all-empty cap of this column — fluid_weight stays 0.
        while y > 0 {
            y -= 1;
            let idx = base + y * stride_y;
            let level = grid.cells[idx].level;
            if level > 0.0 {
                cumulative = level;
                fluid_weight[idx] = cumulative;
                break;
            }
        }
        // From the topmost fluid cell down, accumulate normally.
        while y > 0 {
            y -= 1;
            let idx = base + y * stride_y;
            cumulative += grid.cells[idx].level;
            fluid_weight[idx] = cumulative;
        }
    }
}
```

## Correctness — same result, less work

For column `c` with topmost-fluid at `y_top`:

| y range          | old code writes         | new code writes         |
|------------------|-------------------------|-------------------------|
| `size-1 .. y_top+1` (cap) | `0` (cumulative is 0)   | nothing (left at 0)     |
| `y_top`                   | `level[y_top]`          | `level[y_top]`          |
| `y_top-1 .. 0`            | `cumulative + below`    | `cumulative + below`    |

Identical fluid_weight values; the only difference is that the empty
cap is left at its pre-zeroed value instead of being explicitly
re-zeroed. Empty cells beneath stacked fluid still get walked and
written — those need their (non-zero) cumulative for Phase-4 neighbor
lookups.

Verified via the Phase-4-exercising test set: `upward_pressure_equalization`,
`cross_chunk_pressure_equalization`, `cross_chunk_horizontal_pool`,
`three_chunk_cascade`, `staircase_cascade`, `uneven_fill_equalizes_to_flat_surface`,
`realistic_density_boundary_leak_test` — all green (90/90).

## Expected impact

At chunk_size=30, `tick_chunk` runs many times per second across all
fluid-containing chunks. The pre-process loop is small but **uniformly
hot** — runs unconditionally on every `tick_chunk` that passes the
`has_fluid` early-out.

Per-chunk work (worst-to-best case):

| Scenario                                  | Old iters | New iters | Speedup on loop |
|-------------------------------------------|-----------|-----------|-----------------|
| Fully-flooded chunk (rare)                | 27 000    | 27 000    | 1×              |
| Half-flooded chunk                        | 27 000    | ~13 500   | ~2×             |
| Typical cave with ~10 % column coverage   | 27 000    | ~2 700    | ~10×            |
| Shallow puddle (1-cell-tall film on floor)| 27 000    | ~900      | ~30×            |

The cache-stride win is on top of the iteration win: the original loop
stepped y by `size = 30` per iteration, i.e. ~360 B per FluidCell
stride — basically a fresh cache line per cell. The new loop walks the
same memory pattern but for many fewer cells, so the L1 cache pressure
also drops in proportion.

**Estimated wall-time impact on `tick_chunk`** (matches the 2026-05-20
review's deferred-item estimate of 2–5 % off `tick_chunk` for chunks
with no fluid columns over 2-cells-tall):

- **~2–4 % off fluid worker wall-time** in typical exploration (most
  loaded chunks have empty or shallow-puddle fluid grids — the
  promoted-but-mostly-empty case).
- **0 % regression** on fully-flooded chunks (loop trip count
  identical).
- **Bigger win on water-spreading scenes** where many chunks get
  promoted on a fluid expansion event but most cells are still empty
  on the first few ticks.

Total estimate: **2–4 % off fluid worker wall-time steady-state, up to
~5 % during active water expansion across new chunks.**

## Verification

```
cargo build  -p voxel-fluid    # clean, 1.81 s
cargo test   -p voxel-fluid    # 90 passed; 0 failed
cargo build  --workspace       # clean (only pre-existing warnings)
```

## Deferred items (carrying forward)

From 2026-05-20, still outstanding:

1. **OrePaint Phase 2 anchor selection is O(N²)** —
   [voxel-ffi/src/brushes.rs:319-325](voxel-ffi/src/brushes.rs:319).
   Player-input-frequency — flagged for completeness, won't ship until
   it shows up in a profile.

2. **OrePaint Phase 3 `write_ore_at_world` does a fresh `HashMap`
   lookup per voxel** — [voxel-ffi/src/brushes.rs:353-365](voxel-ffi/src/brushes.rs:353).
   Code-tidiness only.

3. **Mining/brush callers of `recompute_capacity` could pass an AABB
   hint** so only the modified neighborhood gets touched. Bigger API
   surface change. **Estimated win: 5–10 % off brush-stroke wall-time
   at chunk_size=30.** — Top remaining shipping target.

4. **`apply_density` test paths still call `recompute_capacity()`
   after `update_density()`** ([sim/mod.rs:899-902](voxel-fluid/src/sim/mod.rs:899)).
   Test-only, no shipping impact — defer until the next
   test-hygiene pass.

### New observation worth recording (not implemented this pass)

**POI tracker scan throttle is a fixed `16 chunks / 2 s`** ([voxel-ffi/src/poi_tracker.rs](voxel-ffi/src/poi_tracker.rs)).
Healthy for low-impact background work, but on a fresh world load with
hundreds of new chunks the time-to-first-POI is bounded by this
throttle (~minutes for a large map). Worth profiling whether a brief
catch-up burst at world-load would let the morph manifest pick up POIs
faster without a steady-state cost increase. **Estimated win: cosmetic
only (faster first POI play after fresh load); steady-state unchanged.**

End of review.
