# Perf review — 2026-05-16 (implementation pass)

No new commits have landed on `origin/main` since yesterday's review.
HEAD is still `267196b` ("OrePaint creative brush"), and yesterday's
own perf branch `perf/2026-05-15-easy-wins` (commit `22e8324`) remains
unmerged. Rather than re-analyze the same diffs a third day, today's
session **implemented the two deferred items from yesterday's "left
for a future session" list** that were both bounded in scope:

> - **Fluid-cell sparse index** (priority #1 — biggest single win,
>   ~25–50% off `equalize_horizontal`).
> - **Per-chunk BitVec for quench BFS visited set** (priority #3).

Code is on branch `perf/2026-05-16-equalize-bfs` in worktree
`../voxel-backend-perf-2026-05-16` (commit `0e466df`). Not pushed to
`main` — left for review per scheduled-task instructions.

Both changes live in a single file (`voxel-fluid/src/sim/utils.rs`),
+63 / −18, and are localized enough to be reviewed in one sitting.

## What was implemented

### 1. `equalize_horizontal` — drain-based BFS, no separate visited set
**File:** `voxel-fluid/src/sim/utils.rs:120-195`
**Was:** Three hash probes per neighbor: `visited.contains(&neighbor)`,
`water_cells.contains_key(&neighbor)`, then later `water_cells[&pos]`
in the average / apply phase. Plus a whole second `HashSet` allocation
for `visited` that has to grow as the largest connected region does.
**Now:** A single `water_cells.remove(&neighbor)` acts as the combined
"is this a water cell?" + "have we visited it yet?" check, and the
value tuple `(chunk_key, lx, ly, lz, level, cap)` is carried into
`region: Vec<((world_pos), (cell_val))>` so the apply phase reads from
the local vec instead of probing the map again.

**Why it matters:** `equalize_horizontal` is called every fluid tick
for both water and lava, and the BFS phase scales with the number of
fluid cells (pools, cascades, fluid brushes — anywhere a connected
body of fluid touches). Per-neighbor hash work was paying for two
SipHash-of-12-bytes lookups plus an `insert` into `visited`; this drops
it to one `remove` (probe + maybe-take in a single op).

**Estimated savings:** **~30–40% off `equalize_horizontal` wall-time**
on pool-heavy scenes. Yesterday's review estimated ~25–50% for the
full sparse-index version; this captures most of that win with a
fraction of the risk (no new struct fields, no incremental-maintenance
invariants to enforce). Connectivity is identical because a cell is
"in the region" iff its key existed in `water_cells` when the BFS
started — exactly what the previous `visited`-and-`contains_key`
combo encoded.

---

### 2. `detect_lava_water_quench` scoria BFS — `Vec<bool>` visited + dirty-list
**File:** `voxel-fluid/src/sim/utils.rs:239-260` (scratch struct),
`:363-453` (BFS body)
**Was:** `bfs_visited: HashSet<(usize, usize, usize)>` cleared and
rebuilt per contact cell. Each `bfs_visited.insert(pos)` SipHashed a
24-byte tuple key.
**Now:** Two new `QuenchScratch` fields — `bfs_visited_marks:
Vec<bool>` (size³, allocated once on first use and reused) and
`bfs_visited_dirty: Vec<usize>` (linear indices we marked this BFS).
Visit check is `if marks[pi] { continue; }`, then `marks[pi] = true;
dirty.push(pi);`. Between contact cells we walk `dirty` and reset just
those slots, so we never wholesale-zero the 27 KB mark grid.

**Why it matters:** Quench-detection runs every lava tick and the BFS
fires once per lava-cell-touching-water (`contact_cells`). On a busy
quench scene (waterfall hitting lava, lava-water mixing brush) there
can be dozens of contact cells per tick, each kicking off a BFS that
visits up to ~150 cells at scoria_depth=3. The HashSet path was
~24-byte SipHash on every step; the array path is a single bounds-
checked byte read.

**Estimated savings:** **~30–50% off `detect_lava_water_quench_with_
scratch` wall-time** during active quench scenes. Quiet scenes (no
contact cells) are unchanged — they short-circuit out of the contact
pass before this code path is even reached.

---

## Aggregate impact (rough)

| Surface                                     | Saving   | Frequency             |
|---------------------------------------------|----------|-----------------------|
| `equalize_horizontal` (per fluid type/tick) | 30–40%   | every fluid tick      |
| `detect_lava_water_quench` (BFS phase)      | 30–50%   | every lava tick w/ contacts |

Both surfaces are inside the fluid-thread hot path, called from
`tick_fluid` on every fluid step. On a pool-heavy or
volcanic-zone-active world this should knock single-digit
percentage points off the *whole* fluid-thread budget, which has
historically been the second-largest sim-side cost behind chunk
streaming (see [perf-baselines.md](perf-baselines.md)).

## Verification

- `cargo build -p voxel-fluid` — green, 0 errors, only pre-existing
  unrelated dead-code warnings from `voxel-core`.
- `cargo test -p voxel-fluid` — **90/90 pass**. Covers all the
  integration scenes that exercise both modified functions:
  `cross_chunk_pressure_equalization`, `two_bowls_connected_by_channel`,
  `narrow_passage_between_chambers`, `staircase_cascade`,
  `realistic_density_boundary_leak_test`, `lava_water_solidification`,
  `water_subtype_solidifies_lava`, `uniform_layer_stays_stable`,
  `upward_pressure_equalization`, `v_valley_fills_bottom`,
  `sloped_floor_pools_at_low_end`, `l_shaped_container_damping`,
  `three_chunk_cascade`, `uneven_fill_equalizes_to_flat_surface`,
  `u_tunnel_cross_section`, `water_never_enters_solid_cells`.
- `cargo test --workspace` — only the same 3 `voxel-ffi::delta`
  failures yesterday's review documented as pre-existing on `main`
  (`binary_roundtrip_with_data`, `snapshot_roundtrip`,
  `realistic_chunk_size_roundtrip`, all `density at 101: left 1.01
  right 1.0`). Independently confirmed unrelated to this branch.

## Items still deferred (not implemented this session)

From yesterday's "left for a future session" list, the third item:

- **Skip per-substep memcpy in `tick_chunk`** — requires the "rolling
  stale scratch" invariant be enforced project-wide; one bad writer
  outside `tick_chunk` would corrupt state. Still worth doing in a
  focused session with its own correctness review.

Plus one new candidate that surfaced while reading the quench code:

- **`scratch.obsidian_set` / `scoria_set` / `drained_set` could be
  `HashSet<u64>` keyed by a packed `(key, x, y, z)` u64** — the current
  `HashSet<CellAddr>` SipHashes a 24-byte tuple per insert/contains.
  Quick win, ~10–15% off the contact-pass + final-collect cost.
  Deferring because the public `QuenchPlan` exposes `CellAddr` and
  changing it touches downstream FFI consumers.

- **`equalize_horizontal`'s initial `water_cells.insert` walks every
  voxel in every chunk with `has_fluid`** (triple nested loop over
  `0..size`). Same shape as `regen_sources` before the `has_sources`
  flag fix — adding a sparse-index `Vec<u32>` of "cells with level >
  MIN_LEVEL" on `ChunkFluidGrid`, incrementally maintained, would
  collapse it to O(fluid_cells) instead of O(chunk_size³). Bigger
  payoff than today's win and is the actual "fluid-cell sparse index"
  yesterday's #1 was pointing at — today's drain-BFS only addressed
  the *flood-fill* half of that function. Hold for a dedicated session.

## How to review

```bash
cd /c/Users/Shazbot/voxel-backend-perf-2026-05-16
git log -1
git diff main
cargo test -p voxel-fluid
```

Or from the main worktree:
```bash
git diff main..perf/2026-05-16-equalize-bfs
```

To merge: `git merge perf/2026-05-16-equalize-bfs` (fast-forward). To
discard: `git worktree remove ../voxel-backend-perf-2026-05-16 &&
git branch -D perf/2026-05-16-equalize-bfs`.

Note: yesterday's branch `perf/2026-05-15-easy-wins` (commit `22e8324`)
is also still unmerged. Today's branch is based on `main`, NOT on
yesterday's perf branch, so the two can be reviewed and merged
independently in either order.
