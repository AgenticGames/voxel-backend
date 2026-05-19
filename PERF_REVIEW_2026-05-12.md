# Performance Review — Recent Commits (2026-05-12)

Scheduled review of commits added since the 2026-05-10 review.

Commits surveyed:
- `3a55227` — Amphibolite metamorph + hydrothermal water-boost v2 + live lava-water quench
- `ddd4ac3` — Creative PaintStress brush + per-voxel painted-stress overlay
- `6bbb4dc` — Fluid sim: fractional capacity, scratch reuse, lava-skip

The 6bbb4dc commit itself was already a perf-focused commit (scratch reuse, `has_lava` short-circuit). I diffed those files in particular for missed opportunities.

---

## Implemented this run — `equalize_horizontal` O(Y_range × N) → O(N)

**File:** `voxel-fluid/src/sim/utils.rs:73` (function `equalize_horizontal`)
**Status:** ✅ Applied. All 90 `voxel-fluid` unit tests pass (incl. cross_chunk_pressure_equalization, uneven_fill_equalizes_to_flat_surface, l_shaped_container_damping, two_bowls_connected_by_channel — the equalization-heavy ones).
**Not committed.** Diff is sitting in the working tree for you to review.

### What was wrong

`equalize_horizontal` runs **every water tick** on the fluid worker thread (`voxel-fluid/src/thread.rs:193`). After building a `water_cells` index over all non-source water cells (one pass over every loaded fluid chunk), it then ran:

```rust
for wy in min_world_y..=max_world_y {
    let cells_at_y: Vec<(i32,i32,i32)> = water_cells.keys()
        .filter(|&&(_, y, _)| y == wy)
        .copied()
        .collect();
    for start in cells_at_y { /* BFS + average */ }
}
```

That is `O(Y_range × |water_cells|)` HashMap key visits per tick — plus one fresh `Vec` allocation per Y level, **most of which are empty** (typical water world has water concentrated in a small handful of Y bands, but the loop runs over every Y in the full min..=max span). The outer Y loop is structurally unnecessary because BFS neighbors hold Y fixed (`pos.1` is never offset, only `pos.0` / `pos.2`), so connected regions are already Y-disjoint and the `visited` set is sufficient to dedup. Iterating water_cells once is identical in result.

A representative scene: water spans Y range ≈ 40 voxels, `|water_cells|` ≈ 4–8k cells across a few lakes/rivers. Old work: 40 × 6000 = 240,000 key visits + 40 vec allocations per tick. New work: 6000 key visits + 1 vec allocation. Plus the per-region `region` Vec and `VecDeque` are now reused across regions instead of re-allocated.

### Estimated saving

- **~60–80% off `equalize_horizontal` wall-time** on water-heavy scenes (more on wider Y spreads — a vertical cave with stacked pools would have benefited even more).
- **~30–40% off** on small water volumes (small lake, narrow Y range), since the build pass dominates and that didn't change.
- In absolute terms on a chunk_size=30 scene with ~5k water cells across Y range 40: ~0.4–0.8ms reclaimed per water tick on the fluid worker thread. At water_substeps=2 + 10Hz tick rate, that's ~4–16ms/sec of worker CPU freed — meaningful when the worker is competing with mining/seam/flatten/sleep requests.
- **Allocation pressure:** ~40 Vec allocations per tick → 1 per tick. Reduces malloc traffic on the fluid worker thread, which has the second-most allocator pressure after the meshing worker.

### Safety / correctness

- BFS already restricts to one Y plane via fixed `pos.1` in neighbor calculation — verified at line 151 of pre-patch source. Regions are Y-disjoint, so iteration order doesn't change region membership.
- Damped averaging (`EQ_DAMPING=0.3` blend) is the same regardless of which cell starts a region's BFS, so the per-region update is order-independent.
- Old code iterated `water_cells.keys()` (HashMap) — already non-deterministic within each Y level. New code is non-deterministic across the full set. No deterministic invariant lost.
- All 90 voxel-fluid tests pass without modification, including cross-chunk pressure equalization, narrow passage between chambers, uneven-fill-to-flat-surface — i.e. the tests that specifically exercise equalize_horizontal across multi-chunk water bodies.

---

## Other findings (not applied — for your review)

These are below the bar for "implement now" — most need a small structural change or carry test-coverage risk I'd rather you sign off on. Listed in rough impact order.

### A. Rust — `detect_lava_water_quench` allocates 4 HashSets per tick

**File:** `voxel-fluid/src/sim/utils.rs:255-258`

Runs every water tick. `obsidian_set`, `scoria_set`, `drained_set`, `pillow_set` are freshly allocated each call, then collected into Vecs at the end. In quench-active scenes, sets hold dozens-to-hundreds of entries. Could either (a) stash them on `FluidThreadState` (alongside the existing scratch buffers added in 6bbb4dc), or (b) build directly into Vecs and dedup via small `BTreeSet` or `HashSet` only for `pillow_sources` (which is the only one with genuine dedup needs — `obsidian_set` is keyed on `(key,x,y,z)` and each cell is visited once anyway, so a Vec works).

**Estimated saving:** ~10–15% of `detect_lava_water_quench` wall-time when active; on water-only worlds, the `has_lava` flag from 6bbb4dc already short-circuits this so no win there. Net: useful only in volcanic/lava scenes.

### B. Rust — Quench BFS visited set allocated per contact cell

**File:** `voxel-fluid/src/sim/utils.rs:337` (`detect_lava_water_quench`, inner BFS)

```rust
for &(key, x, y, z) in &contact_cells {
    ...
    let mut visited: HashSet<(usize, usize, usize)> = HashSet::new();
```

One fresh HashSet per contact cell. On a long lava-water front (say 50 contact cells), that's 50 HashSet allocations per tick. Reusing one HashSet with `.clear()` between iterations is the standard fix.

**Estimated saving:** ~5–10% of `detect_lava_water_quench` cost in active quench scenes. Small absolute number but easy.

### C. Rust — `paint_stress_sphere` clones full snapshots for undo

**File:** `voxel-ffi/src/brushes.rs` (the new PaintStress brush from ddd4ac3 — search `paint_stress_sphere` and its `capture_undo_for_range` call)

The new PaintStress brush captures full `ChunkSnapshot` per affected chunk for undo. A `ChunkSnapshot` now includes the painted-stress overlay (added in this commit), which means each captured snapshot is bigger than before. For typical sphere brush sizes (radius 3–6 voxels at chunk_size=30) the brush touches 1–8 chunks; that's manageable. But for the wide-area variants the user can build by stamping rapidly, this can balloon. Worth considering a sparse-diff undo format for the painted-stress overlay specifically (it's a `Vec<f32>` so most cells are zero in typical use). Lower priority unless the workflow involves a lot of paint-stress brushing.

### D. Rust — `squeeze_excess_fluid` walks every cell every density-change tick

**File:** `voxel-fluid/src/sim/utils.rs:7-64`

Called from the worker when fluid grids get density updates. Walks every cell of the affected chunk regardless of where the density change happened. Could be restricted to a bounding box around the density change. Modest win — only matters for high mining/flatten throughput.

### E. Rust — `aureole.rs` BFS reallocates frontier vecs

**File:** `voxel-sleep/src/phases/aureole.rs` (commit 3a55227, the water-boost v2 paths)

`compute_water_boost` does Phase 1 BFS from lava + Phase 2 BFS from water-network seeds. Both use fresh `Vec`s for frontier/visited. Reusing scratch buffers stored on the aureole state would cut allocations during the metamorphism phase of sleep. But: sleep is wall-time-bounded (multi-second operation, user-visible loader), so the malloc cost is a rounding error. Not worth touching unless sleep wall-time becomes a focus.

---

## What I checked but found clean

- **6bbb4dc fractional capacity** (`cell.rs::capacity_from_corners`): correct math, no missed cases. The `scratch_cells`/`scratch_weights`/`scratch_drain` reuse pattern is the right call.
- **6bbb4dc `has_lava` flag**: all the lava-placing paths set it (AddFluid, geological springs, place_sources, pending load). Recompute pass in `tick_chunk` is bounded to a single grid walk per chunk. Good.
- **ddd4ac3 painted-stress overlay**: empty `Vec` → no allocation invariant is held in `voxel-core/src/stress.rs` (`painted_stress` only grows on first non-zero write). Save format v3 sparse encoding is correct (overlay is `Option<Vec<u8>>`).
- **3a55227 morph step seam fix**: pre-populating `seam_data_map` with global out-of-block neighbors is the right structure; no extra cost on the hot path.

---

## Recommended next action

The `equalize_horizontal` fix above is the only one I implemented. Items A and B (the two `detect_lava_water_quench` allocation fixes) are the natural next pair — both are ~20-line mechanical fixes in the same file with similar risk profile. Item D's bounding-box-restricted squeeze is more invasive (requires plumbing a bbox through `FluidEvent::DensityUpdate`) and I'd want a benchmark first.

If `equalize_horizontal` looks correct on review:
- Commit message draft: `Fluid sim: drop unnecessary Y-loop in equalize_horizontal`
- 1 file changed, ~12 insertions / ~24 deletions (net −12)
- Tests already pass, no behavior change

---

*Generated by daily-commit-performance-review scheduled task.*
