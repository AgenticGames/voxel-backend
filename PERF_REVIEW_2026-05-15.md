# Perf review — 2026-05-15 (implementation pass)

No new commits have landed on `origin/main` since yesterday's review
(`PERF_REVIEW_2026-05-14.md` covered `267196b`, `6539312`, `a0ebed5`,
`b2f522a`, `6bbb4dc`, `ddd4ac3`). Rather than re-analyze the same diffs,
this session **implemented** the three lowest-risk, highest-ROI items
from yesterday's priority list. Code is on branch
`perf/2026-05-15-easy-wins` in worktree `../voxel-backend-perf-2026-05-15`
(commit `22e8324`). Not pushed to `main` — left for review.

## What was implemented

### 1. `recalc_stress_region*` — single `get_mut` per voxel
**Files:** `voxel-core/src/stress.rs:1443-1465`, `:1499-1522`
**Was:** `stress_fields.get(&key).map(|sf| sf.painted(...))` followed by a
separate `stress_fields.get_mut(&key)` for the write. Two HashMap hashes
of the *same* chunk key per voxel.
**Now:** One `get_mut` grabs the field, reads `painted()` into a local
(it's an `&self` method, so it works through the live `&mut`), then
writes `set` / `set_class`.

**Why it matters:** `recalc_stress_region_v2_filtered` runs on every
mining action and during sleep stress passes. Touches tens of thousands
of voxels in a multi-chunk recalc. Each voxel was paying two HashMap
hashes when one suffices.

**Estimated savings:** **~5–15%** off `recalc_stress_region_v2_filtered`
wall-time on multi-chunk recalcs. Same fix applied to the legacy
`recalc_stress_region` (line 1486+) since it had the identical pattern.

---

### 2. OrePaint Phase 2 — spatial-grid Poisson, not O(n²)
**File:** `voxel-ffi/src/brushes.rs:599-635`
**Was:** Per-candidate `accepted.iter().any(|a| (a.world_pos - cand.world_pos).length_squared() < min_spacing2)`.
With `density=1.0` and a brush touching ~500 wall candidates, that's up
to **~75k length-squared comparisons** per brush call.
**Now:** Hash-bucket accepted anchors by `floor(world_pos / min_spacing)`.
For each candidate, only probe the 3×3×3 = 27 neighboring buckets. O(n)
overall.

**Why it matters:** OrePaint is a click-driven brush, so per-call latency
matters for responsiveness, and the rejection scan is its quadratic
inner loop.

**Estimated savings:** **~50–70%** off Phase 2 wall-time at high
density; ~3–7% off the whole brush call (Phase 2 is one of three phases).
All 6 OrePaint unit tests still pass (determinism, anti-clumping,
weights, wall-exposure, zero-weight no-op, density-untouched).

---

### 3. `paint_stress_sphere` — hoist `ensure_painted_alloc`, direct-write hot loop
**Files:** `voxel-core/src/stress.rs:181-186` (made `ensure_painted_alloc` `pub`),
`voxel-ffi/src/brushes.rs:262-329`
**Was:** Every per-voxel `sf.add_painted(...)` did `ensure_painted_alloc()`
(empty-check branch) + `index` + clamp + write. The branch is well-predicted
after first allocation but still costs per call.
**Now:** Pre-allocate the painted layer once per chunk (only for `op=add` /
`op=subtract`), then inline `index` + clamp + direct write into
`sf.painted_stress[i]` in the inner loop. `clear_painted` (op=2) keeps its
lazy path since it has its own empty-fast-out.

**Why it matters:** PaintStress is the only brush with a stress-overlay
write loop, and on big radii (8+) it can touch thousands of voxels per
click.

**Estimated savings:** **~5–10%** off paint loop wall-time on large
radii. All 4 PaintStress tests pass (accumulate, clear-op, undo, drives
overstressed threshold).

---

## Aggregate impact (rough)

| Surface | Saving | Frequency |
|---|---|---|
| `recalc_stress_region_v2_filtered` | 5–15% | every mining action |
| OrePaint brush call (high density) | 3–7% | per ore-paint click |
| `paint_stress_sphere` brush call | 5–10% | per paint-stress click |

Combined, this is ~5–10% off the average mining-action stress recalc
plus modest savings on two creative-brush call paths. None of these are
top-N tick-time consumers, but all three are pure wins with no correctness
or staleness risk, no FFI changes, no new flags, and no test regressions.

## Verification

- `cargo build --workspace` — green
- `cargo test -p voxel-ffi --lib brushes::` — **31/31 pass** (covers all
  brush tests including OrePaint determinism, anti-clumping, weights,
  and PaintStress overlay/undo/threshold)
- `cargo test -p voxel-core` — **97/97 pass** (covers stress v2 tests)
- `cargo test --workspace` — 3 failures in `voxel-ffi::delta::tests`
  (`binary_roundtrip_with_data`, `snapshot_roundtrip`,
  `realistic_chunk_size_roundtrip`) — **verified pre-existing on
  `origin/main`** by stashing changes and re-running tests. All three
  panic with `density at 101: left 1.01 right 1.0` — a density
  compression-roundtrip mismatch in the delta serializer that has nothing
  to do with stress or brushes. Likely from a recent FfiBuildingFlatten
  density-write edit; worth flagging separately but not this branch's
  concern.

## Items deliberately not implemented this session

From yesterday's priority list, left for a future session because they
need more careful design or have non-trivial risk:

- **Fluid-cell sparse index** (priority #1 yesterday — biggest single
  win, ~25–50% off `equalize_horizontal`). Needs an incrementally
  maintained `Vec<u32>` of cells-with-fluid on `ChunkFluidGrid`, with
  write hooks at every place that mutates `cells[i].level`. Cross-cutting
  change with staleness risk; deserves its own focused session.
- **Per-chunk BitVec for quench BFS visited set** (priority #3). Easy
  but lives inside `detect_lava_water_quench` which I didn't want to
  touch in a multi-fix commit.
- **Skip per-substep memcpy in `tick_chunk`** (priority #5). Needs the
  "rolling stale scratch" invariant to be enforced project-wide; one
  bad writer outside `tick_chunk` would corrupt state.

## How to review

```bash
cd /c/Users/Shazbot/voxel-backend-perf-2026-05-15
git log -1
git diff origin/main
cargo test -p voxel-ffi --lib brushes::
cargo test -p voxel-core
```

Or from the main worktree:
```bash
git diff main..perf/2026-05-15-easy-wins
```

To merge: `git merge perf/2026-05-15-easy-wins` (fast-forward). To
discard: `git worktree remove ../voxel-backend-perf-2026-05-15 &&
git branch -D perf/2026-05-15-easy-wins`.
