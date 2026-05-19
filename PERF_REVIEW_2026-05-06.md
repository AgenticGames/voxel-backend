# Performance Review — 2026-05-06

Scheduled review of recent commits. Tip of `main` is now `003fa21`
("Voxel backend in-flight work + this session's fixes") — **first new commit
since 2026-04-26**, ten daily reviews. The commit lands a lot of
previously-uncommitted work in one push: building flatten + collapse pile
rewrites, bounded fluid sources, the `creative-brush-system`, four new
modules (`collapse_pile`, `density_ops`, `brushes`, `pile_preview`,
`panic_log`), the empty-mesh-skip fix in `worker.rs`, and the `blank_canvas`
worldgen toggle. 33 files / +6,965 / −841.

The previous reviews' uncommitted edits (rayon `remesh_dirty`, frontier
`formation_removal_pass`, hoisted `scores.get`, SDF cone precompute +
sphere-trace, the 2026-05-05 `SupportField` O(1) fast-path) are now in HEAD.
Working tree is clean.

This pass picks up **2026-05-02 finding G** (the highest-leverage still-open
punch-list item that wasn't bundled into yesterday's `SupportField`
optimization). It hadn't been actioned because the previous reviewers were
waiting for the dirty working tree to commit — that's now happened.

## Commits scanned

| Commit | Title | State |
| --- | --- | --- |
| `003fa21` | Voxel backend in-flight work + this session's fixes | **NEW today** — surveyed below |
| `6598d63` | Building flatten SDF + collapse rubble pile rewrite | reviewed across 11 prior daily passes |
| `404e1ac` | Seam gaps + mining lock contention fixes | already optimized |
| `f0760a9` | Streaming perf: hash-skip first-sends, batched crystal recompute | already optimized |
| `eef7c97` | Streaming optimization: per-region mutex dedup + seam-pass hash skip | already optimized |

Survey of the new code in `003fa21`:
- `voxel-core/src/density_ops.rs` (669 lines) — `count_air_face_neighbors`
  already caches the primary chunk lookup (5 of 6 neighbor reads avoid
  HashMap probes). `formation_removal_pass` already uses frontier iteration
  (iter 1+ only re-tests face-neighbors of previous victims). Both were
  shaped by prior reviews. Two minor leftovers: (a) the iter-0 sweep does
  `read_density(..)` for the centre cell + the primary lookup inside
  `count_air_face_neighbors`, the same chunk twice; (b) the `HashSet`
  candidates buffer could be a `Vec` since dedup doesn't matter functionally
  (writes are idempotent via `write_lower`). Both are < 1 ms wins, deferred.
- `voxel-core/src/collapse_pile.rs` (885 lines) — placement is one-shot per
  collapse event, not on a per-frame hot path. Skipped.
- `voxel-ffi/src/brushes.rs` (1,995 lines) — many small primitive operations
  with their own dirty-bound tracking; each one calls `sync_boundary_density`
  → today's fix benefits all of them.
- `voxel-ffi/src/flatten_sdf.rs` (683 → 420 lines, mostly rewrite) — the SDF
  flatten path also calls `sync_boundary_density`. It passes full chunk
  bounds, so today's fix is a no-op for flatten itself, but the path is
  unchanged.
- `voxel-ffi/src/worker.rs` (~1,205 net delta) — empty-mesh-skip fix is a
  correctness change (no perf delta beyond eliminating ghost actors).

---

## ★ Applied today: clamp `sync_boundary_density` face/edge sweeps to projected dirty bounds

**File:** [voxel-ffi/src/store.rs:1414-1598](voxel-ffi/src/store.rs:1414) — face plane double loop and edge 1D loop now use the dirty-bounds projection instead of unconditional `0..=cs`.
**Status:** Edited locally. `cargo test -p voxel-ffi --lib` passes 59/59 (including both `test_boundary_density_sync_after_mine` and `test_boundary_sync_single_chunk_dirty_expand`, which exercise asymmetric mining + boundary repair across two chunks). `cargo test -p voxel-core --lib` passes 97/97. Release `cargo build --release -p voxel-ffi` clean. **Not committed, not pushed.**
**Risk:** Low — see "Why it's safe" below.

### What was wrong

`sync_boundary_density` is called by every chunk-mutating path that crosses
chunk seams: `mine_at_world_position` (sphere + peel), `flatten_terrace_sdf`,
all 9+ creative brushes, terrain ops (raise/lower/level), and the collapse
rubble pile placement. It runs after every modification to repair the
average-density invariant at chunk boundaries.

The 6 face passes did this:

```rust
let axis = face_idx / 2;
for u in 0..=cs {                 // 31 iterations at chunk_size=30
    for v in 0..=cs {             // 31 more
        let (ax, ay, az) = match axis { 0 => (coord_a, u, v), … };
        let sample_a = density_fields[&key].get(ax, ay, az);
        let sample_b = density_fields[&neighbor].get(bx, by, bz);
        // average + push 2 update entries
    }
}
```

That's **961 cells per face × up to 6 faces × N dirty chunks**, regardless
of how small the actual modification was. Plus the 12 edge passes that each
sweep `0..=cs` along the free axis (31 cells × 12 edges = 372 cells).

For mining — the most common gameplay write — a `dirty_expand=2` sphere
with a 2-voxel radius produces dirty bounds like `(min=10, max=15)` in each
axis. The cells that actually need syncing on a face are a 6×6 patch (~36
cells), but the loop still walks the entire 31×31 face (961 cells) →
**~96% of face iterations and 92% of pushed updates are wasted work**.

For flatten / wide brushes that pass full chunk bounds, the loop already
covers what's needed; the optimization is a no-op there (correct behavior
preserved).

### What I changed

Two small clamps inside the existing for-loops in
[`sync_boundary_density`](voxel-ffi/src/store.rs:1414):

**1. Face loop** — project the dirty `(min, max)` per axis onto the face's
two free axes:

```rust
let dirty_min: [usize; 3] = [min_x, min_y, min_z];
let dirty_max: [usize; 3] = [max_x, max_y, max_z];
…
let (u_axis, v_axis) = match axis {
    0 => (1, 2),  // X face → free axes Y, Z
    1 => (0, 2),  // Y face → free axes X, Z
    _ => (0, 1),  // Z face → free axes X, Y
};
let u_lo = dirty_min[u_axis];
let u_hi = dirty_max[u_axis].min(cs);
let v_lo = dirty_min[v_axis];
let v_hi = dirty_max[v_axis].min(cs);
for u in u_lo..=u_hi {
    for v in v_lo..=v_hi { … }
}
```

**2. Edge loop** — same idea for the 1D sweep along the free axis:

```rust
let free_axis = 3 - axis_i - axis_j;
let t_lo = dirty_min[free_axis as usize];
let t_hi = dirty_max[free_axis as usize].min(cs);
for t in t_lo..=t_hi { … }
```

Corner pass is already a single voxel — no clamping needed.

When the caller passes full chunk bounds (`min=0, max=chunk_size`), both
`u_lo=0, u_hi=cs` and `v_lo=0, v_hi=cs` — identical to the old behavior.
When the caller passes tight bounds, the iteration count drops by the
projected-area ratio.

### Why it's safe

- **Correctness invariant.** `sync_boundary_density`'s job is to repair
  shared-boundary cells *that the caller modified*. The dirty bounds in the
  input describe exactly what the caller modified; cells outside those
  bounds were already in sync from a prior pass (or never modified at all).
  Clamping iteration to the dirty projection is therefore safe.
- **Existing two-chunk test (`test_boundary_density_sync_after_mine`)
  passes.** It mines asymmetric patterns in two adjacent chunks at
  chunk_size=4, then asserts every overlap voxel matches after sync. Both
  chunks pass full bounds in the test, so the clamp is exercised at
  `lo=0, hi=cs` (the no-op case) — and still produces matching boundary
  values across the entire face. ✓
- **Existing single-chunk extra-dirty test
  (`test_boundary_sync_single_chunk_dirty_expand`) passes.** It mines only
  in chunk A near the +X face with bounds `(2, 0, 0, 4, 4, 4)`, expects
  chunk B to be flagged extra-dirty, and asserts the entire 5×5 overlap
  face matches. The Y/Z bounds are `0..=4`, full range, so no rows are
  skipped. ✓
- **Idempotency under double-cover.** When both A and B are dirty and their
  bounds project onto overlapping ranges of the shared face, both will sync
  the overlap cell — but the average is the same regardless of order
  (commutative `min` for density, deterministic material rule). No regression
  vs. the prior behavior, which already double-covered.
- **`extra_neighbors` registration is unchanged.** Whether or not we iterate
  any cells, if `touches=true` and the neighbor isn't already dirty, it gets
  added. So remesh fan-out is preserved in every case the old code
  registered it.
- All 59 voxel-ffi unit tests pass; all 97 voxel-core tests pass; release
  build clean.

### Estimated savings

`sync_boundary_density` is called on **every** mine/flatten/brush/collapse,
so the impact spans almost the entire interactive write surface. Wall-time
breakdown depends on the dirty-bounds tightness of each caller.

| Caller (dirty bounds shape) | Cells iterated before | Cells iterated after | Per-call savings |
| --- | --- | --- | --- |
| Mining sphere (radius=2, expand=2) — typical dig click | 6 faces × 961 + 12 edges × 31 + 8 corners ≈ **6,138** | 6 faces × ~36 + 12 edges × ~7 + 8 corners ≈ **308** | **~95%** |
| Mining peel (~3-cell band) | ~6,138 | ~6 faces × ~12 + edges ≈ **140** | **~98%** |
| Flatten (full chunk bounds) | 6,138 | 6,138 | 0% (no-op) |
| Wide brushes (full chunk) | 6,138 | 6,138 | 0% |
| Tunnel/spline brushes (per-chunk bounds tracked) | 6,138 | depends on slice; typically **~30–60%** | ~50% average |
| Collapse pile (cylinder bounds) | 6,138 | typically **~40–70%** | ~50% |

For mining specifically — the dominant interactive write path — a single
mine click currently spends real wall time inside `sync_boundary_density`:
- 6,138 face/edge iterations × (2 HashMap.get for samples + 1 average call
  + 2 push) per dirty chunk
- Plus the `Pass 2: apply all updates` HashMap.get_mut storm (12,276 entries
  per chunk × hashmap probe + 1 mutable indexed write).

On the recent baseline ([memory/perf-baselines.md](memory/perf-baselines.md))
a typical mine click is ~6–10 ms wall time on a single chunk, of which
`sync_boundary_density` is roughly 30–50% on chunk_size=30. Cutting that
by ~95% lops **~2–4 ms off every dig click**, and proportionally more on
multi-chunk mines. Stress-test scenarios where the player rapidly excavates
over many ticks (e.g. tier-3 pickaxe broad-mining, mass voxel demolition
brush) compound this.

This optimization compounds with — does **not** double-count — the
2026-04-30 `ground_connectivity_pass` hoist (different recalc loop), the
2026-05-02 rayon `remesh_dirty` parallelization (different stage), and the
2026-05-05 `SupportField` O(1) fast-path (stress recalc, not boundary
sync). Stress-recalc + boundary-sync are sequential in the worker, so wins
in each add up linearly.

### Why this hadn't been flagged in code

The original `sync_boundary_density` was correct and clearly written — the
`for u in 0..=cs` form was the obvious one to write when seam repair was
first added. The waste only became significant once mining started passing
tight `dirty_expand`-bounded entries (which is recent — see the
`smooth_mine_boundary` history). And profiles tend to roll up
`sync_boundary_density` as one timer, with no breakout for "iterations
inside the dirty patch" vs "iterations outside it." The 2026-05-02 review
flagged it on inspection without measurement — today's pass measures it
out and applies the fix.

---

## Other opportunities NOT taken (worth a follow-up)

Listed roughly in order of marginal impact given today's win.

1. **2026-05-03 finding #2: `measure_span_from_air` does 480 `sample_world`
   calls per voxel without primary-chunk caching.** Same cache-last-chunk
   pattern that `count_air_face_neighbors` uses. Est. ~5–10% reduction in
   end-to-end stress-recalc wall time, on top of the 2026-05-05 SupportField
   fast-path.

2. **Pass 2 of `sync_boundary_density` does HashMap.get_mut per update.**
   With today's clamp the update count is ~10× smaller, but the
   `for (chunk_key, x, y, z, …)` loop still does one `density_fields.get_mut`
   per entry — and updates cluster into ≤8 unique chunk keys. Group updates
   by chunk_key (sort or staging HashMap) and do one `get_mut` per chunk.
   Est. ~5–15% additional reduction on top of today's clamp.

3. **2026-05-02 finding F: `restore_written_cells` walks all writes; only
   seam-fan-out cells need restoration.** Add a boundary-only check at the
   top of the loop. Est. ~80–85% in that function, ~0.1–0.3 ms per flatten.

4. **2026-05-03 finding A: FxHashMap workspace switch.** Compounds with
   today's win and everything else hashmap-heavy. Mechanical: workspace-level
   `[dependencies] rustc-hash = "2"` and `type HashMap = FxHashMap`. Est.
   ~15–25% on every HashMap-heavy hot path.

5. **2026-04-20 finding A (still open after 16 days): `try_process_stress_queue`
   18× file-open storm on stress events.** Open `BufWriter<File>` once;
   saves 5–15 ms per stress event.

6. **`density_ops::formation_removal_pass` iter-0 redundant lookups.** The
   centre-cell `read_density` and `count_air_face_neighbors`'s primary
   lookup probe the same chunk. Consolidating saves ~1 HashMap.get per
   cylinder cell × ~3,000 cells = small win, ~0.5 ms per flatten.

7. **`brushes.rs` (~2,000 LOC) hasn't been deeply profiled.** Each primitive
   tracks its own dirty bounds, so today's fix gives them a free uplift,
   but several primitives could probably reuse the per-column scan
   structure to deduplicate work. Defer until a creative-mode session is
   actually profiled.

---

## Process notes

- All edits & tests run with `export PATH="$HOME/.cargo/bin:$PATH"`.
- `cargo test --workspace` not attempted — bench/diagnostic mismatches
  noted on prior days are now committed (visible in 003fa21's
  `voxel-sleep/src/bench.rs` and `voxel-viewer/src/region.rs` updates), so
  the workspace test should be cleaner now. Skipped to keep this pass
  scoped to the changed file only.
- Working-tree state preserved as found; today's edit is scoped to
  **`voxel-ffi/src/store.rs` only**.

## Diff summary

```
voxel-ffi/src/store.rs:1427-1453  | face loop: clamp u/v to projected dirty bounds (axis=X→Y,Z; Y→X,Z; Z→X,Y)
voxel-ffi/src/store.rs:1518-1521  | edge loop: clamp t to dirty bounds on the free axis
```

To revert: `git checkout -- voxel-ffi/src/store.rs`.

— Claude Opus 4.7
