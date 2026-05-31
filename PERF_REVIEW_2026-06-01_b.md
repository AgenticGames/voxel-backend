# Perf review — 2026-06-01 (run b) — scheduled "review latest commits" task

**Reviewer:** Claude (autonomous scheduled `daily-code-improvement` run).
**Scope:** latest commits on `main`. Run *a* earlier today (`9f03c5e`) took the last
provably-safe win inside the stress system and concluded `recalc_stress_region_v2` /
`ground_connectivity_pass` are **exhausted** for autonomous work (the only remaining
lever there is a *behavioral* flood early-out, explicitly reserved for human review).
So this run moved to a **different, untouched hot path**: the per-chunk fluid tick.

## Finding: `tick_chunk` re-probed the chunk HashMap per *boundary fluid voxel* for cross-chunk reads

`voxel-fluid/src/sim/chunk.rs` :: `tick_chunk` is the per-chunk fluid simulation step
(gravity → slope flow → horizontal spread → upward pressure), run every fluid tick for
every chunk that holds fluid. It had already been tuned for **allocation** reuse (scratch
cells/weights/drain taken off the grid once per tick) and the `cell_solid` bitfield
replaced a former per-voxel `chunks.get` in `count_solid_face_neighbors`. But the
**cross-chunk neighbour reads** inside the per-voxel loop were still doing a fresh
`chunks.get(&neighbour_key)` — a std-HashMap **SipHash** probe on a 12-byte
`(i32,i32,i32)` key — for *every boundary fluid voxel*, at five sites:

| site | what it reads | key |
|------|---------------|-----|
| gravity, `y==0`            | chunk below            | `(cx, cy-1, cz)` |
| slope `below_solid` check, `y==0` | chunk below     | `(cx, cy-1, cz)` |
| slope gather, X/Z edge     | lateral neighbour      | one of ±X/±Z |
| slope gather, `ny<0`       | chunk below            | `(cx, cy-1, cz)` |
| horizontal spread, X/Z edge| lateral neighbour      | one of ±X/±Z |

Every one of those keys is **invariant for the whole chunk-tick** — they only depend on
`key` and a fixed offset, never on the voxel — yet they were re-hashed per voxel. A single
`y==0` fluid voxel could probe the same below-chunk key up to ~5 times in one iteration
(gravity + slope check + each in-bounds slope offset).

Crucially, `tick_chunk` **never mutates a neighbour chunk** inside the loop: every
cross-chunk effect is deferred into the returned `cross_transfers`, and every in-chunk
write goes to the owned `new_cells` scratch (moved off the grid via `mem::take`). So the
neighbour grids can be borrowed **immutably for the whole loop** with no aliasing — the
same kind of loop-invariant HashMap-probe hoist the recent stress passes (`ef56a78`,
`cfe9b43`, `9f03c5e`) applied per-cell.

## What I changed (1 file, voxel-fluid internal — no FFI/ABI surface)

Hoisted the five neighbour-chunk lookups out of the per-voxel loop, once per `tick_chunk`:

```rust
let key_below = (key.0, key.1 - 1, key.2);
let key_xp = (key.0 + 1, key.1, key.2);   // …xn / zp / zn
let nbr_below = chunks.get(&key_below);
let nbr_xp = chunks.get(&key_xp);          // …xn / zp / zn  (Option<&ChunkFluidGrid>, Copy)
```

- The three below-chunk probes now read the cached `nbr_below`.
- The two lateral sites pick the right cached ref with a handful of **tuple comparisons**
  (`dest_key == key_xp …`) instead of a SipHash probe. `resolve_neighbor` only ever
  returns single-axis crossings, so at those sites the destination is provably one of the
  four laterals (a `debug_assert_eq!` guards the final `else`).

**Behavior-preserving — provable:**
- Same keys, same `Option<&ChunkFluidGrid>` values: hoisting a lookup whose key never
  changes returns exactly what the per-voxel lookup returned (`Some(grid)` for a present
  neighbour, `None` for an absent one — both paths preserved, including the
  "no chunk below ⇒ treat as solid" branch).
- The lateral tuple-compare select resolves to the identical grid the old
  `chunks.get(&dest_key)` returned, because `dest_key` *is* one of `key_xp/xn/zp/zn` there.
- No new mutation, no aliasing: the loop only ever *reads* neighbours; all writes were
  already to owned scratch / deferred `cross_transfers`.

`cargo test --workspace`: **all green** — voxel-core 101, voxel-ffi 125, voxel-fluid 90,
voxel-sleep 106, 0 failures. The fluid suite's cross-chunk / slope / horizontal-spread
tests (downward boundary flow, diagonal slope, spread equalization, source containment)
all pass unchanged.

## Estimated savings (MEASURED, A/B microbench)

`git stash` A/B of just this diff, **release** build, total `step_fluid` wall-time over a
closed **3×3×3 block of adjacent water-saturated chunks** at `cs=30` (the live override),
with a perpetual source column per chunk so flow never settles (inner faces exercise the
real cross-chunk `Some` path, outer faces the `None` path). 40 ticks × 4 rounds/side, two
independent A/B rounds:

| side | round 1 best | round 2 best |
|------|-------------:|-------------:|
| baseline (per-voxel `chunks.get`) | 7.308 ms/tick | 7.426 ms/tick |
| optimized (hoisted neighbour refs) | 6.790 ms/tick | 6.822 ms/tick |

**≈ 7–8 % reduction in total fluid-tick wall-time** on this scene (−7.1 % and −8.1 %;
ranges across all rounds are non-overlapping, so the win is real, not noise). This is a
% of the **whole tick**, not an isolated sub-phase — every µs is the per-voxel
gravity/slope/horizontal/pressure math still running underneath.

**Honest caveat on the headline number:** this scene is deliberately *saturated* — 27
adjacent chunks, every cell full of water — to maximise the number of boundary fluid
voxels hitting the cross-chunk branches. The win scales with **boundary** fluid-voxel
count, so:
- **Active cross-chunk flow** (waterfalls/lava cascading between chunks, streams crossing
  boundaries, pools spreading into neighbours) → close to the measured 7–8 %.
- **Interior-only pools** (fluid not touching chunk faces) never enter the cross-chunk
  branches at all → near-zero change (already fast).
- **Sparse fluid** → between the two, proportional to how much fluid sits on chunk faces.

Where it lands: the fluid sim ticks every chunk with fluid each step; during the big
flow events (worldgen water settling, lava galleries, save-load fluid restore, in-game
spills) many boundary voxels probe neighbours every tick — that's exactly where this
trims ~a third-to-half of the per-voxel HashMap overhead down to 5 probes per tick.

Zero-risk, zero-ABI, clean `git revert`: a loop-invariant HashMap-probe hoist mirroring
the stress passes, validated by the full test suite + a two-round A/B.

## Other commits reviewed — no action taken
- Stress system (`recalc_stress_region_v2` / `ground_connectivity_pass`): exhausted for
  safe autonomous wins; remaining lever is behavioral (see `PERF_REVIEW_2026-06-01.md`).
- `tick_chunk`'s scratch-buffer reuse, `cell_solid` bitfield, and hoisted
  `slope_candidates` Vec are all already in place and correct.
