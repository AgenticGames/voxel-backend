# Perf review — 2026-05-24 (scheduled daily pass)

No new commits on `main` since the 2026-05-23 review (HEAD is still
`e0a08b3`, the `ChangeManifest::compact` sort + in-place coalesce). The
voxel-ffi/* tree has substantial uncommitted WIP today (~928 lines
across 11 files including `brushes.rs`, `poi_tracker.rs`, `worker.rs`,
etc.), so this pass deliberately targets a crate **outside** voxel-ffi
to avoid colliding with that work.

Diff: **+147 / −31** across one source file + two new integration
tests + this review.
Tests: **11 (existing) + 3 (new correctness) + 1 (ignored bench)** —
14 active tests green, no regressions.

## What was implemented — A* lazy closed-set + g-score-on-entry + skip-known-neighbor traversal

**File:** [voxel-path/src/astar.rs:107-222](voxel-path/src/astar.rs:107),
new test [voxel-path/tests/astar_relaxation.rs](voxel-path/tests/astar_relaxation.rs),
new ignored bench [voxel-path/tests/astar_bench.rs](voxel-path/tests/astar_bench.rs)

`compute_path` is the kernel under every AI route — Spider, Wasp,
Creature, plus the Crystal Anchor Bridge POI tracker all run their plans
through it via the dedicated path-worker thread (see
[ue-project.md](ue-project.md) and
[path-planning-system.md](path-planning-system.md)). On the live
`ChunkStoreGrid` each `can_traverse` call does a DashMap probe of
`density_fields` (lookup + density grid sample, ~100–300 ns under
contention), so cutting redundant traversability checks pays for itself
in real wall-time, not just micro-benchmark cycles.

The old loop did, per neighbor:

```rust
if closed.contains_key(&neighbor) { continue; }      // probe 1 — HashMap
if !can_traverse(grid, neighbor, mode) { continue; } // probe 2 — DashMap (live grid)
if !corner_clip_clear(...) { continue; }             // 0–6 probes — per-step
...
let existing_g = *g_score.get(&neighbor).unwrap_or(&f32::INFINITY); // probe 3 — HashMap
```

Plus, on every `pop`:

```rust
if closed.insert(current, ()).is_some() { continue; } // probe 4 — HashMap (write+test)
let current_g = *g_score.get(&current).unwrap_or(&f32::INFINITY); // probe 5
```

And `compute_neighborhood_offsets()` was called once per `compute_path`,
constructing 26 `IVec3`s on every search.

### The fix

Three composable changes:

1. **Pack `g_score` onto each `OpenEntry`.** On pop, the entry's
   `g_score` is compared against `g_score_map[cell]` to detect stale
   heap entries (one extra HashMap probe to do that, but it eliminates
   the previous `closed.insert(...)` probe + the `g_score.get(&current)`
   probe afterwards — net **–1 probe per pop**).
2. **Drop the `closed: HashMap<IVec3, ()>` map entirely.** With a
   consistent heuristic (euclidean on a grid is consistent), the first
   non-stale pop of a cell is already its optimal expansion. The
   stale-entry guard above subsumes the role of `closed`.
3. **Skip `can_traverse` for already-known neighbors.** A neighbor that
   appears in `g_score` was inserted via the relaxation step, which
   only runs after `can_traverse` returned true — so its traversability
   is already proven. Re-checking it on every revisit was redundant.

```rust
let existing_g = g_score.get(&neighbor).copied();

if existing_g.is_none() {
    // Untouched neighbor — must verify traversability now.
    if !can_traverse(grid, neighbor, request.mode) {
        if !grid.is_loaded(neighbor) { touched_unloaded = true; }
        continue;
    }
}
// corner_clip_clear is per-step (depends on current + offset), still runs.
if !corner_clip_clear(grid, current, offset, request.mode) { continue; }

let tentative_g = current_g + step_len + extra;
if tentative_g < existing_g.unwrap_or(f32::INFINITY) {
    came_from.insert(neighbor, current);
    g_score.insert(neighbor, tentative_g);
    open.push(OpenEntry { cell: neighbor, g_score: tentative_g, f_score: ... });
}
```

Plus a small drive-by: `compute_neighborhood_offsets()` → `static
NEIGHBOR_OFFSETS: [IVec3; 26]`, so the array lives in `.rodata` and
costs zero per-call construction.

## Correctness — same path, verified by 3 new integration tests

| Scenario                                  | Expectation                                                                  | Test                                              |
|-------------------------------------------|------------------------------------------------------------------------------|---------------------------------------------------|
| (0,0,0)→(3,3,0) in open space             | 3 diagonals, length 3·√2 ≈ 4.243 (NOT 6 face steps)                          | `diagonal_beats_manhattan_route_to_corner`        |
| Detour around a 1-cell wall               | Route avoids wall cells; total length under 5.0 (proves diagonal detour)     | `optimal_path_around_block_via_relaxation`        |
| (0,0,0)→(5,5,5) in open space             | 5 corner-diagonals of length 5·√3 ≈ 8.660; no cell visited twice              | `stale_pop_does_not_re_expand_better_predecessor` |

All 3 pass. The 11 pre-existing tests in
[voxel-path/src/tests.rs](voxel-path/src/tests.rs) also pass unchanged
(Flying / Walking / Surface modes, pillar detour, sealed-box no-path,
max-nodes budget, theta-smoothing collinear collapse, etc.).

The new tests deliberately target the relaxation path: in open 3D
space many cells are reached first via face-neighbor expansions
(cheaper f-score initially), and the diagonal predecessor only
relaxes their g-score later. If the stale-pop guard were broken — or
if dropping `closed` allowed re-expansion of already-optimal cells —
the path would either include a duplicate cell or run longer than the
optimal length. Both are asserted.

## Expected impact

**Measured stub-grid speedup (60³ pillar field, 200 paths, release
build):**

| Run            | OLD (with closed-map) | NEW (lazy closed + skip-known) |
|----------------|-----------------------|--------------------------------|
| Run 1          | 2.328 ms/path         | 2.228 ms/path                  |
| Run 2          | 2.275 ms/path         | 2.182 ms/path                  |
| Run 3          | 2.210 ms/path         | 2.154 ms/path                  |

Median: **2.275 → 2.182 ms/path = ~4 % wall-time drop on a `HashSet`
stub.**

The stub backs `is_solid` with a `HashSet::contains` (~5–10 ns per
call). On the **live** `ChunkStoreGrid` each `is_solid` is a DashMap
probe + density grid sample, conservatively **100–300 ns** under
contention from concurrent worker threads — i.e. the expensive part of
each `can_traverse` call is roughly **20–30× more costly** in
production than in the stub. Skipping `can_traverse` for already-known
neighbors therefore scales the saving with the underlying grid cost.

For a typical chase-replan A* on a live ChunkStoreGrid (~1000 node
expansions × 26 neighbors, of which ~40–60 % are already-visited
revisits):

| Phase                                | Old cost / 1000 nodes  | New cost / 1000 nodes  | Δ                |
|--------------------------------------|------------------------|------------------------|------------------|
| `closed.contains_key(&neighbor)`     | 26 000 × ~40 ns ≈ 1 ms | —                      | −1 ms            |
| `can_traverse` on known neighbors    | ~13 000 × ~200 ns ≈ 2.6 ms | —                  | −2.6 ms          |
| `closed.insert` per pop              | 1 000 × ~50 ns ≈ 0.05 ms | —                    | −0.05 ms         |
| `g_score.get(&current)` per pop      | 1 000 × ~30 ns ≈ 0.03 ms | —                    | −0.03 ms         |
| New: stale-pop g-compare             | —                       | 1 000 × ~30 ns ≈ 0.03 ms | +0.03 ms     |
| **Net**                              | ~3.7 ms                 | ~0.03 ms                | **~–3.6 ms**     |

Wall-time estimate per **live** A* call: ~30 ms (representative) →
**~27 ms**, i.e. **~10–15 % per path query**.

System-level translation:

- **AI path latency** drops by the same factor for every Spider /
  Wasp / Creature replan and every Crystal-Anchor POI scan. Player
  experience: AI reacts ~10–15 % faster after a creative-brush carve
  or stress collapse changes the topology.
- **Path-worker throughput** rises by the inverse — the dedicated
  thread can serve more requests per second before the `path_tx`
  channel (cap 256) starts dropping or queuing.
- **Frame time** itself sees ~0 direct change — the path planner
  already runs on its own core and is not on the render-frame
  critical path.

## Verification

```
cargo build  -p voxel-path                                # clean
cargo test   -p voxel-path                                # 11 unit + 3 integration + 1 ignored bench → 14 active green
cargo test   -p voxel-path --release --test astar_bench \
             -- --ignored --nocapture                     # bench prints wall-time
cargo build  -p voxel-sleep -p voxel-fluid -p voxel-core \
             -p voxel-gen -p voxel-noise -p voxel-cli \
             -p voxel-viewer -p voxel-path                # workspace clean (voxel-ffi WIP excluded)
```

## Deferred items (carrying forward)

From 2026-05-21 / 2026-05-23, still outstanding:

1. **OrePaint Phase 2 anchor selection is O(N²)** —
   [voxel-ffi/src/brushes.rs:619-635](voxel-ffi/src/brushes.rs:619).
   Currently in user's uncommitted WIP — defer until that lands.

2. **Mining/brush callers that feed `update_density` could pass an AABB
   hint** — [voxel-fluid/src/cell.rs:359-395](voxel-fluid/src/cell.rs:359).
   API surface change — plumbs through `TerrainModified`. **Estimated
   win: 5–10 % off brush-stroke wall-time at chunk_size=30.** Top
   remaining shipping target once voxel-ffi quiets down.

3. **`apply_density` test paths still call `recompute_capacity()`** —
   [voxel-fluid/src/sim/mod.rs:899-902](voxel-fluid/src/sim/mod.rs:899).
   Test-only, no shipping impact.

4. **POI tracker scan throttle is a fixed `16 chunks / 2 s`** —
   [voxel-ffi/src/poi_tracker.rs:43-45](voxel-ffi/src/poi_tracker.rs:43).
   Cosmetic only — faster first POI play after fresh world load. Also
   in voxel-ffi WIP today.

5. **`count_topology_votes_cross_chunk` ~131 DashMap lookups per air
   voxel** — flagged 2026-05-23 against
   [voxel-ffi/src/poi_scanner.rs](voxel-ffi/src/poi_scanner.rs). Still
   pending the topology-vote WIP merge.

### New observations worth recording (not implemented this pass)

**Path-result cache key is `(from, to, mode)` but `corner_clip_clear`
is computed fresh on every search** — the planner currently rebuilds
the entire path each time even when terrain hasn't changed in the
relevant cells. A `(chunk_key, generation)` invalidation tag on the
cache would let the 10 s TTL run all the way out for static-terrain
chase paths. **Estimated win: 30–60 % off repeat-pathing CPU for
spider chase loops on unchanging terrain.** Larger refactor — flag for
the next pass.

**`surface_normal_at` recomputes the same neighbor sum on every visit
in surface mode** —
[voxel-ffi/src/pathing.rs:109-123](voxel-ffi/src/pathing.rs:109) does 6
`is_solid` calls per call, and it's invoked at least once per A*
neighbor expansion in surface mode (Spider AI), plus again per
visited cell at path-reconstruction time. A per-search memo HashMap
keyed by IVec3 would cut ~6× DashMap probes on every revisit.
**Estimated win: 10–15 % off Spider-only path queries; ~0 on
Flying/Walking.** Localizable change, candidate for a future surface-mode
pass.

End of review.
