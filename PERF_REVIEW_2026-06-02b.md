# Perf review — 2026-06-02 (run *b*) — scheduled "review latest commits" task

**Reviewer:** Claude (autonomous scheduled run). **Scope:** latest commits on `main`, headed by
`e1679be` (run *a*, earlier today) — the flat-sentinel vertex-map win in DC `generate_mesh`. That
commit's closing section explicitly flagged **the next lever**:

> `solve_dc_vertices`'s `qefs` map has the same dense-key pattern… Left for the next pass.

This run takes that lever — but **not** the way the note assumed, because the obvious version is a
**performance trap**. The honest A/B below shows the naive swap *losing* by 63%, and the version that
actually ships winning by ~41%.

## The target: `solve_dc_vertices`'s per-cell QEF accumulator

[`dual_contouring/solve.rs`](voxel-core/src/dual_contouring/solve.rs) `solve_dc_vertices` runs once per
meshed chunk, immediately upstream of `generate_mesh` (run *a*'s target) — same chunk-arrival critical
path (worker generate ×7, brush edits, seam re-stitch, store rebuild, slab, sleep-morph, pile preview).

For every sign-changing edge it accumulates the intersection into up to 4 adjacent cells' QEF
accumulators, keyed by a **dense** linear cell index in `0..grid_size³`:

```rust
let mut qefs: FastHashMap<usize, QefData> =
    FastHashMap::with_capacity_and_hasher(hermite.edges.len() / 2, Default::default());
...
qefs.entry(adj.cells[i]).or_default().add(intersection_pos, intersection.normal);   // hash probe per corner
...
for (&idx, qef) in &qefs { vertices[idx] = qef.solve_clamped(...); }                 // hash-ordered drain
```

This is the *same dense-key-in-a-HashMap* anti-pattern run *a* killed in `generate_mesh` — `~4 ×
edge_count` `entry()` probes per call through hashbrown's machinery.

### …but this map is NOT the same case as run *a*'s, and that matters

Run *a*'s map grew from empty and **rehashed ~10× per call** — eliminating that churn dominated its
−69%. This map is **pre-sized** (`with_capacity_and_hasher(edges.len()/2)`), so there is **no rehash
churn to remove**. The only cost on the table is the per-probe hashing itself — a much smaller prize.
And `QefData` is large (**~56 B**: `[f32;6]` ATA + two `Vec3` + `f32` + `u32`). Those two facts are why
the naive fix backfires.

## What I tried first — and why it LOST (recorded honestly)

The literal reading of run *a*'s note: replace the map with a dense `vec![QefData::default(); total]`
indexed by cell, plus a `touched: Vec<usize>` to keep the solve pass sparse.

**Measured (clean A/B, release, bench below): 550 µs/call vs 337 µs baseline → +63% SLOWER.**

Cause: for the live cs=30 grid, `total = 27 000` cells × ~56 B = a **~1.5 MB zeroed allocation every
call**. The memset of 1.5 MB per `solve_dc_vertices` swamps the hash-probe saving, and since the baseline
map was pre-sized there was no alloc churn to win back. A dense `Vec<QefData>` is the wrong structure
when the surface is sparse (~2 600 / 27 000 ≈ 10 % of cells touched) and the element is fat.

## What actually shipped — u32 indirection map → compact pool

Keep raw array indexing (no hashing) **without** paying a `QefData`-sized memset over the whole grid:

```rust
let mut cell_to_slot = vec![u32::MAX; total];                       // ~108 KB memset (u32, not QefData)
let mut qef_pool:     Vec<QefData> = Vec::with_capacity(edges.len());// only the touched cells live here
let mut touched_cells: Vec<usize> = Vec::with_capacity(edges.len());
...
let mut slot = cell_to_slot[cell];
if slot == u32::MAX {                       // first touch of this cell
    slot = qef_pool.len() as u32;
    cell_to_slot[cell] = slot;
    qef_pool.push(QefData::default());
    touched_cells.push(cell);
}
qef_pool[slot as usize].add(intersection_pos, intersection.normal);  // raw index, no hash
...
for (slot, &idx) in touched_cells.iter().enumerate() {               // sparse, pool-ordered drain
    vertices[idx] = qef_pool[slot].solve_clamped(min_bound, max_bound);
}
```

- The per-grid memset shrinks from **~1.5 MB (`QefData`) to ~108 KB (`u32`)** — ~14× less zeroing.
- The fat `QefData` accumulators live in a **contiguous pool sized to the surface** (`Vec::push`,
  pre-reserved to `edges.len()`), not scattered through a 27 000-slot array — better cache behaviour in
  the solve drain too.
- **Zero hashing**: `entry()` → one `u32` array read + a branch.

### Behavior-preserving — provable, output bit-identical

- `hermite.edges` iteration order and the `adjacent_cells_*` order are **untouched**, so each cell's
  `QefData` accumulates the *same* intersections in the *same* order → byte-identical accumulator to the
  old map's value for that cell.
- `vertices[idx]` is written **exactly once per touched cell** (each cell owns one pool slot), and
  `solve_clamped` is a pure deterministic function of the accumulator + cell bounds. Writes are
  order-independent (disjoint indices), so the hash-ordered drain and the pool-ordered drain produce the
  **same** `vertices` array.
- `u32::MAX` is an impossible real slot: a slot is `qef_pool.len()` at push time ≤ touched-cell count ≪
  4 e9, so the sentinel can never collide.
- New regression test `shared_cell_accumulates_into_one_slot`: two perpendicular edges (X- and Y-axis)
  bordering one shared cell must merge into a **single** slot and yield a vertex pulled in *both* x and y
  (fails if a second edge grabbed a fresh slot and overwrote the first).
- `cargo test --workspace`: **voxel-core 103** (102 + the new test), **voxel-ffi 126, voxel-fluid 90,
  voxel-sleep 106 — 0 failures.** Identical sink (non-NaN vertex count) on both A/B sides:
  31 248 000 = 2 604 surface cells × 2 000 calls × 6 rounds.

## Estimated savings (MEASURED, A/B microbench)

`git stash` A/B of just this diff, **release** build. Bench
([`voxel-core/examples/bench_dc_solve.rs`](voxel-core/examples/bench_dc_solve.rs)): the same realistic
hermite as run *a*'s bench — a sinusoidal terrain sheet through a cs=30 grid (the live UE override),
2 722 sign-changing edges, 2 604 surface cells of 27 000 — `solve_dc_vertices` timed in isolation over
2 000 calls × 6 rounds, best-of (round 0 cold excluded):

| side | best µs/call |
|------|-------------:|
| baseline (pre-sized `FastHashMap<usize, QefData>`) | **337.12** |
| **dense `vec![QefData; total]` + touched (naive — REJECTED)** | **550** *(+63 %, a loss)* |
| **u32 indirection map + compact pool (SHIPPED)** | **199.28** |
| **delta (shipped vs baseline)** | **−40.9 % (≈1.69× faster)** |

Ranges are wholly non-overlapping (baseline rounds 337–341, shipped rounds 199–205), so the win is real,
not noise.

**Honest scoping of the number:**
- The **−41 %** is of `solve_dc_vertices` **in isolation**, one stage of chunk meshing (density sample →
  hermite extract → **solve** → `generate_mesh` → `convert_mesh_to_ue_scaled`). Run *a* trimmed the
  `generate_mesh` stage by ~69 %; this trims the **solve** stage by ~41 %. They stack — both are on the
  same per-chunk path.
- The remaining ~199 µs is dominated by the **QEF solve itself** (Jacobi-SVD, up to 50 sweeps × 2 604
  surface cells) — that's arithmetic, not structure overhead, and is left alone. So this is close to the
  last *structural* win in `solve_dc_vertices`; further gains would mean touching the SVD math (a
  numerically-sensitive, human-reviewed change, not autonomous).
- Where it lands: **lower worker-thread CPU during the chunk-arrival storm** (startup, zone-in,
  save-load restore) and on **every brush/mining edit, seam re-stitch, slab, and sleep-morph re-mesh** —
  same path as run *a*. It does **not** change steady-state frame time when no chunks are being
  (re)meshed.

Zero-risk, zero-ABI (voxel-core internal, no FFI surface), clean `git revert`.

## The lesson for the next pass

The flagged "next lever" was real but the obvious fix was a **trap** — a dense `Vec` of a fat element
over a sparse domain loses to a pre-sized map. The winning shape is **small dense index map → compact
pool**. Whenever a future review sees "dense-key HashMap → flat array", check **(a)** was the map already
pre-sized (no rehash churn to win back), and **(b)** is the value fat and the domain sparse? If both,
indirect through a `u32` slot map; don't memset the value type over the whole grid.

## Other commits reviewed — no action needed
- Run *a*'s `generate_mesh` flat-vertex-map win confirmed still in place.
- `solve_clamped` / `jacobi_svd_3x3` are pure arithmetic — the dominant cost now, but a numerically
  sensitive math change reserved for a human pass.
- The `AdjacentCells` fixed-array (no per-edge heap alloc) is already lean and unchanged.
