# Perf review — 2026-05-19 (implementation pass)

Still no new commits on `origin/main` since 2026-05-12 (`267196b`,
OrePaint creative brush) — 7th consecutive review with no fresh
committed work. The deferred-items queue is largely drained by prior
sessions, but a close re-read of `voxel-fluid/src/sim/chunk.rs`
turned up one **missed alloc** that the recent scratch-reuse work
(commits `6bbb4dc`, `a0ebed5`, `6539312`) walked right past.

Implemented as a single-file in-place edit (left in the working tree;
**not committed, not pushed** per scheduled-task instructions). The
change is co-located with the other in-flight uncommitted work the
user already has staged in `voxel-ffi/*`, `voxel-gen/*`, etc. — only
`voxel-fluid/src/sim/chunk.rs` is mine.

Diff: **+10 / −2** in one file. ~5-minute review.

## What was implemented

### `tick_chunk` — hoist per-cell `slope_candidates` Vec

**File:** [voxel-fluid/src/sim/chunk.rs](voxel-fluid/src/sim/chunk.rs:184-191)

**Was** — inside the slope-flow branch (taken once per "fluid cell
with a solid cell directly below"), allocated fresh every iteration:

```rust
// chunk.rs:334 (old, per-cell)
let mut candidates: Vec<(f32, f32, usize, bool, (i32, i32, i32),
                         usize, usize, usize)> = Vec::new();
// … up to 4 pushes …
candidates.sort_by(...);
for (_score, dst_space, ni, is_cross, dest_key, dest_x, dest_y, dest_z)
    in candidates { … }
```

`Vec::new()` itself is alloc-free, but the first `.push()` heap-allocates
~32–64 bytes (4 entries × 48 bytes/tuple = 192 bytes, rounded up by the
allocator), and the Vec is dropped at end-of-iteration.

**Now** — one `Vec::with_capacity(4)` hoisted to the top of `tick_chunk`,
re-used across all cells via `.clear()`:

```rust
// chunk.rs:184 (new, hoisted to tick_chunk scope)
let mut slope_candidates: Vec<…> = Vec::with_capacity(4);

// chunk.rs:340 (per-cell)
slope_candidates.clear();
let candidates = &mut slope_candidates;
// … same pushes/sort, then borrow via .iter() …
```

Same shape as the `scratch_cells` / `scratch_weights` / `scratch_drain`
reuse already on `ChunkFluidGrid` (commit `6bbb4dc`) and the
`region`/`bfs_queue` hoist in `equalize_horizontal` (commit `b2f522a`).

## Why this was missed

Each of the recent fluid perf commits targeted one specific function:

- `6bbb4dc` — `tick_chunk`'s **`cells.clone()`** + per-substep `Vec`s.
  ≈540 KB / 108 KB / 108 KB chunk-scale buffers — the obviously huge ones.
- `b2f522a` — `equalize_horizontal`'s **per-Y region Vec** + BFS deque.
- `a0ebed5` — `detect_lava_water_quench`'s **per-tick `HashSet` allocs**.
- `6539312` — `regen_sources` chunk-skip flag.

All four were "this function does N³ work or per-substep allocs."
`slope_candidates` is _per-cell_, capped at 4 entries — small enough
to escape that scan, but called inside the same hot triple-`for` loop
as everything else. It's the exact alloc that the `scratch_*`
treatment was invented for, just one nesting level deeper.

## Expected impact

The slope-flow branch fires for cells where the cell directly below
is solid (i.e. fluid sitting on a downward rock slope or floor). In
the bowl / staircase / sloped-floor test scenarios, that's a
substantial fraction of all active fluid cells — anywhere from
**10 % to 40 %** depending on terrain.

For a representative chunk_size=30 scene with ~2 000 active fluid
cells and ~30 % slope-active, that's **~600 saved allocs per
chunk-tick**. Across the typical streaming-radius fluid load (say
20–40 chunks with any fluid in them, water tick at ~10 Hz), the
old code path was paying **~120 k – 240 k tiny `Vec::push()`
allocs per second** on the fluid worker thread. The new path: zero.

Concrete savings, conservative:

- `tick_chunk` wall-time on slope-heavy scenes: **−1 % to −3 %**.
- Per-tick allocator pressure on the fluid worker thread:
  measurably down — these were small allocs and the system allocator
  bucketizes them, but they were still on the hot path.

It's the smallest of the five recent fluid-perf wins, and that's
fine — it was the last alloc on the per-cell loop and now matches
the convention. Bigger ticket: the **`recompute_capacity` N³ walk**
(see "Deferred" below).

## Verification

```
cargo build -p voxel-fluid          # clean, 1.61 s incremental
cargo test  -p voxel-fluid          # 90 passed; 0 failed
```

All 90 voxel-fluid tests pass, including the slope-flow exercising
families:

- `sloped_floor_pools_at_low_end`
- `staircase_cascade`
- `v_valley_fills_bottom`
- `three_chunk_cascade`
- `bowl_fractional_boundary_conservation` (the renamed test from `6bbb4dc`)
- `l_shaped_container_damping`
- `u_tunnel_cross_section`
- `narrow_passage_between_chambers`
- `cross_chunk_pressure_equalization`
- `cross_chunk_horizontal_pool`
- `realistic_density_boundary_leak_test`
- `water_never_enters_solid_cells`

Note: `cargo test --workspace` currently fails in `voxel-gen` (E0063
on a struct literal in `voxel-gen/src/zones/...` and an unused
variable). Confirmed by stash + retry: those errors come from
**preexisting uncommitted work in `voxel-gen/src/config.rs` and
`voxel-gen/src/lib.rs`** (your in-flight changes), not from my
edit. `voxel-fluid` is the only crate this review touched.

## Deferred items (not implemented this pass)

For next session — ranked by impact:

1. **`recompute_capacity` walks the whole chunk on every brush stroke**
   — [voxel-fluid/src/cell.rs:321](voxel-fluid/src/cell.rs:321). At
   chunk_size=30 that's 27 000 `capacity_from_corners` evaluations
   (216 000 corner reads) for what might be a 10-voxel paint stroke.
   The fractional-capacity change in `6bbb4dc` made this hotter
   because it's no longer a trivial scalar test — it's now a 9-cmp +
   counter per cell. Callsites — `thread.rs:684`, `sim/mod.rs:52`,
   `sim/mod.rs:87`, `sim/mod.rs:802`, `sim/mod.rs:901` — would each
   need to be re-examined to see how many of them genuinely need a
   full-chunk recompute (`UpdateFluidConfig` doesn't — densities
   didn't change). **Estimated win: 5–15 % off fluid worker
   wall-time during active brushing / config-tweaking.** Medium-risk
   change because it adds an incremental update API and you have to
   audit the callers.

2. **OrePaint Phase 2 anchor selection is O(N²)** —
   [voxel-ffi/src/brushes.rs:319-325](voxel-ffi/src/brushes.rs:319-325).
   Per-candidate scan over accepted anchors with `length_squared <
   min_spacing²`. For ~hundreds of candidates and ~tens of accepts
   it's fine; for a maxed-out density slider on a big brush it
   degenerates. Drop a 3-D spatial-hash grid keyed by
   `(world_pos / min_spacing).floor()` and only test 27 neighbor
   cells. **Brush is player-input frequency, so the perceptual
   savings are zero** — flagged for completeness only. Won't ship
   until somebody complains.

3. **OrePaint Phase 3 `write_ore_at_world` does a fresh `HashMap`
   lookup per voxel** —
   [voxel-ffi/src/brushes.rs:353-365](voxel-ffi/src/brushes.rs:353-365).
   For each cluster/channel sphere, every painted voxel re-does
   `div_euclid` + `density_fields.get_mut(&key)`. Could group by
   chunk-key and look up once per chunk per anchor. Same
   player-input-frequency caveat — invisible to the user, just code
   tidiness. Defer.

4. **`fluid_weight` build (chunk.rs:189-198) walks the full N³
   grid every substep** to compute cumulative column weight, even
   for chunks where the upward-pressure pass at line 565 will never
   fire (entirely-air or entirely-shallow chunks). Could be lazy:
   build per-column only when the first slope/pressure check needs
   it. Bit hairy because the laziness has to thread through the
   `&mut` borrow rules. **Estimated win: 2–5 % off `tick_chunk` for
   chunks with no fluid columns over 2-cells-tall.**

5. **Mining/brush callers of `recompute_capacity` could pass an
   AABB hint** so only the modified neighborhood gets touched.
   Stronger version of item 1. Big-API-surface change.

End of review.
