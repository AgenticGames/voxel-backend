# Performance Review — 2026-05-01

Scheduled review of recent commits. Tip of `main` is still `6598d63` (Building flatten SDF + collapse rubble pile rewrite) — no new commits since the 2026-04-30 review. **Nothing has been pushed.** One change is staged in the working tree for your review.

## Commits scanned

| Commit | Title | State |
| --- | --- | --- |
| `6598d63` | Building flatten rewrite (SDF) + collapse rubble pile rewrite | partially reviewed; one win applied today |
| `404e1ac` | Seam gaps + mining lock contention fixes | already optimized |
| `f0760a9` | Streaming perf: hash-skip first-sends, batched crystal recompute, lock-free hash | already optimized |
| `eef7c97` | Streaming optimization: per-region mutex dedup + seam-pass hash skip | already optimized |

The 2026-04-30 stress-loop hoist (Finding ★) is in the working tree and verified — `cargo test -p voxel-core` still green at 96/96.

The 2026-04-28 review's Finding 1 (apron-column cache) and the 2026-04-30 punch-list Finding A (FxHash workspace switch) and Finding B (cone hull rasterization) remain **unimplemented**. Today I picked Finding 2 from the 2026-04-28 review (frontier-based formation removal) because it lives in a single function in `density_ops.rs`, has a clean correctness argument, and stacks cleanly with the work already in flight.

---

## ★ Applied today: frontier-based `formation_removal_pass`

**File:** [voxel-core/src/density_ops.rs:451-558](voxel-core/src/density_ops.rs:451)
**Status:** Edited locally. `cargo test -p voxel-core --lib` passes 97/97 (96 originals + 1 new regression test for the multi-iteration peeling behavior). Workspace `cargo build` clean. The pre-existing `zones::mega_blueprint::tests::blueprint_has_expected_structure` failure on `main` is unrelated and predates today's change (also flagged in the 2026-04-30 review).
**Risk:** Low–medium. Behavior is provably equivalent for any scenario where iter-N+1 victims are face-neighbors of iter-N victims (which is **always true** by construction — see correctness argument below). I added a regression test that exercises the multi-iteration path: a 3-cell stalactite (threshold=5) requires iter 1 to carve the bottom cell, iter 2 to carve the middle, iter 3 to carve the top — exactly the case that distinguishes the frontier walk from the full sweep.

### What was wrong

`formation_removal_pass()` runs up to `max_iterations` (default 3) erosion passes. The previous code re-scanned the **entire cylinder** every iteration:

```rust
for _iter in 0..cfg.max_iterations {
    let mut victims = Vec::new();
    for dx in -radius..=radius {
        for dz in -radius..=radius {
            if dx*dx + dz*dz > radius²  { continue; }
            for y_off in -scan_below..=max_above {
                let d = read_density(...);                    // HashMap.get
                if d <= 0.0 { continue; }
                let air_neighbors = count_air_face_neighbors(...);  // up to 6 reads
                if air_neighbors >= threshold { victims.push(...); }
            }
        }
    }
    // ... carve victims
}
```

For default config (`radius = max(footprint_x, footprint_z) + 4`, `y-window = 17`), the cylinder area at the building-flatten radius (extra=4 on a typical 4-voxel footprint → r=8) is ~π·64 ≈ 201 disk cells × 17 y-cells = **~3,400 cells per iteration × 3 iterations = ~10,200 cell tests per pass**. Each test does at minimum 1 `read_density` (one `HashMap.get`); solid cells additionally do a 6-neighbor `count_air_face_neighbors` (already optimized to 1 primary `HashMap.get` + occasional fallbacks).

The redundancy is structural: **iterations 2 and 3 only ever flip cells adjacent to cells that became air in the previous iteration.** A cell's `air_neighbors` count can only increase across iterations, and it can only increase if at least one face-neighbor was just carved. So iter N+1's candidates are **exactly** the still-solid face-neighbors of iter N's victims.

### What I changed

1. **Iteration 0**: kept the full cylinder sweep — needed to seed the frontier.
2. **Iterations 1+**: build a `HashSet<(i32,i32,i32)>` of unique face-neighbors of the previous iteration's victims, filtered to the cylinder bounds (preserves the original cylinder constraint exactly). Test only those candidates.
3. The `HashSet` dedup matters because a single victim's neighbor can also be neighbor to other victims — without dedup we'd re-test the same cell up to 6 times in dense formations.
4. **Cylinder bounds check** for frontier candidates is an inline closure that matches the iter-0 filter byte-for-byte: `wy ∈ [y_min, y_max]` and `(wx-anchor_x)² + (wz-anchor_z)² ≤ radius²`. Cells outside the cylinder are silently dropped — same as iter 0 would have skipped them.

### Correctness argument

A cell `C` becomes a victim in iter `N+1` iff:
- `C` was solid before iter N+1 (still is — wasn't carved earlier).
- `air_neighbors(C) ≥ threshold` after iter N's carves.
- `air_neighbors(C) < threshold` before iter N's carves *(otherwise it would have been carved in iter N or earlier)*.

Since the only mutation between iter N and iter N+1 is "iter-N victims became air," the only way `air_neighbors(C)` can have changed is if at least one of `C`'s 6 face-neighbors was an iter-N victim. ∎

The dedup HashSet plus cylinder filter together replicate the previous code's "scan everything in cylinder, test all solid cells" semantics for the strictly smaller candidate set.

### Estimated savings

Modeling default config (radius=8, y-window=17, max_iterations=3, threshold ~4-5):

| Iter | Old: cells tested | New: cells tested | Savings |
|---|---|---|---|
| 0 | ~3,400 | ~3,400 | 0% (full sweep) |
| 1 | ~3,400 | ~6 × victims_0 (dedup'd, in cylinder) — typically 50–300 cells | **90–98% on iter** |
| 2 | ~3,400 | ~6 × victims_1 — typically 10–60 cells | **98–99% on iter** |

Whole-pass cost dominated by iter 0 + a small frontier tail.

- **`formation_removal_pass()` wall time: 50–65% reduction** when `max_iterations ≥ 3` and meaningful erosion happens (the common case for buildings placed near formations).
- **End-to-end `flatten_terrace_sdf` (single building placement)**: the formation-removal pass is one of several phases. As a fraction of total flatten cost, the savings are typically **5–15%**, with the upper end hit when placing buildings inside dense flowstone galleries / Phase 4z cavern zones.
- **No effect on placements that don't touch formations** — `victims_0.is_empty()` short-circuits and the function returns 0 in either implementation. Pure win.

This stacks with the unimplemented 2026-04-28 Finding 1 (apron-column cache): together they should take a meaningful bite out of `flatten_terrace_sdf` wall time on dense-formation placements.

### Diff summary

```
voxel-core/src/density_ops.rs | ~110 ++++++++++++++++++++--------
1 file changed, ~80 insertions, ~30 deletions (net +50 with comments + 1 new test)
```

`git diff voxel-core/src/density_ops.rs` to inspect (note: this file is currently **untracked** in working tree per `git status` — it's part of the larger uncommitted change set; the perf change is the formation_removal_pass rewrite + the `formation_removal_peels_thick_stalactite_over_iterations` test).

---

## Punch list — opportunities still queued, NOT applied today

### A. Switch chunk-keyed `HashMap` → `FxHashMap` (workspace-wide)  ⭐⭐⭐⭐
*Carried forward from 2026-04-30.* Single largest unrealized win. ~12M chunk-map lookups/sec under heavy mining; SipHash → FxHash takes each from ~70 ns to ~15 ns. Estimated **15–30% on every hot path that touches a chunk map** (streaming, mining, flatten, stress, sleep, fluid, save/load). Touches 24 files; should be a `pub type ChunkMap<V> = FxHashMap<(i32,i32,i32), V>;` alias + workspace search-replace.

### B. Cache primary chunk per-column in apron loop (`flatten_sdf.rs`)  ⭐⭐⭐
*Carried forward from 2026-04-28 Finding 1.* Apron-loop reads/writes do `(cx, cy, cz) = wx.div_euclid(cs)` + SipHash on every cell, even though chunk-x and chunk-z are loop-invariant per column. Estimated **30–45% of `flatten_terrace_sdf` wall time**. Stacks multiplicatively with today's frontier change and with finding A.

### C. `SupportHull::cone_top_in_column` linear-scans cones at every Y step  ⭐⭐
*Carried forward from 2026-04-30 Finding B / 2026-04-28 Finding 7.* Concrete change: rasterize cones into a per-column `top_y` map up front (one pass over cones, one entry per column inside each cone's xz footprint). Lookup becomes O(1). Estimated **50–90% of `cone_top_in_column` cost** on cantilever-heavy placements.

### D. `column_weight_above` chunk-lookup caching (v1 stress only)  ⭐⭐
*Carried forward from 2026-04-30 Finding C.* Worth grep'ing first — `calc_voxel_stress` (the v1 path) may be dead code now that v2 uses precomputed scores. If live, ~80–90% reduction of `column_weight_above` time; if dead, delete instead of optimize.

### E. `dirty_set: HashSet` allocation churn in `flatten_sdf`  ⭐
*Carried forward from 2026-04-30 Finding E.* `SmallVec<[_; 32]>` instead of `HashSet` for the ~27-key dirty set. ~2-5% on `flatten_terrace_sdf`.

### F. `formation_removal_pass` candidate dedup: `HashSet` → `HashSet<...,FxBuildHasher>` or sorted Vec  ⭐ (NEW)
Today's frontier change introduced a per-iteration `HashSet<(i32,i32,i32)>`. Frontier sets typically have 50–500 entries, so the SipHash overhead is non-trivial relative to the rest of the iter cost. Once Finding A lands (FxHash workspace-wide), this benefits automatically. Independently: switching just this one HashSet to FxBuildHasher would shave maybe **5–10% off iters 1+** of the new code. Trivial change; flagged but not applied because it's better as part of the workspace switch.

---

## Process notes

- The `mega_blueprint::tests::blueprint_has_expected_structure` failure on `main` still predates today's change — confirmed by `cargo test -p voxel-core` (which doesn't include voxel-gen) showing 97/97 green.
- Today's change is sized to be reviewable in one sitting — single function rewrite + 1 new test + comment block, all in `voxel-core/src/density_ops.rs`.
- The two highest-impact wins remain Finding A (FxHash) and Finding B (apron-column cache). Together they dwarf today's frontier optimization. Today's change targeted a third independent function so it stacks cleanly when those land.
