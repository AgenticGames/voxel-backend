# Perf review — 2026-06-04 — scheduled "review latest commits" task

**Reviewer:** Claude (autonomous scheduled run). **Scope:** latest commits on `main`. The
2026-06-01→03 sweep finished off the **terrain per-chunk meshing path** in dual contouring and
region-gen:

- `89e67e6` / `e1679be` / `e1175fd` — DC convert / mesh / solve (intra-chunk)
- `0b1374e` — `generate_chunk_seam_quads` neighbor probe (cross-chunk seam)
- `b262b0e` / `832ceb1` — `sync_region_boundary_densities` key re-hash + per-field bucketing

Those stages are now structurally clean. This run moves to the **fluid meshing path**, which last saw
a pass on 2026-05-?? (`8e61222`, fluid *tick*) — its *mesh* build has never been touched — and takes the
single clearest structural defect there.

## Trap checked first: which smoothing/adjacency code is actually live?

`voxel-core/src/mesh.rs::Mesh::smooth` (terrain) and `voxel-fluid/src/mesh.rs::smooth_fluid_mesh` both
build a `Vec<Vec<u32>>` adjacency with a linear-`contains` dedup — an obvious-looking target. **Left
untouched on purpose:** the only available win there (reordering neighbor accumulation, or deduping in a
different order) changes the *summation order* of `avg += old_positions[ni]`, and f32 addition is
non-associative — so it would shift interior vertex positions by ~1 ULP and break the bit-identical bar
this review chain holds itself to. (Chunk-edge verts are pinned, so seams would still match, but the
no-visible-change / provably-identical standard would not survive an A/B `to_bits()` test.) Not worth it.

## The real target: `weld_vertices`' 27-cell neighbor scan

[`voxel-fluid/src/mesh.rs`](voxel-fluid/src/mesh.rs) `weld_vertices` runs inside `mesh_fluid`, which the
fluid worker thread ([`thread.rs:454`](voxel-fluid/src/thread.rs)) calls for **every dirty fluid chunk,
every tick** — a genuinely live, per-frame-class path during any flowing/settling water or lava. Marching
Cubes emits 3 fresh, unshared vertices per triangle, so weld is mandatory to collapse the coincident ones.

For each raw vertex it searched its home spatial-hash cell **plus all 26 neighbors** for a match within
`epsilon`:

```rust
'search: for dz in -1..=1 {
    for dy in -1..=1 {
        for dx in -1..=1 {
            let key = (gx + dx, gy + dy, gz + dz);
            if let Some(bucket) = spatial.get(&key) {   // SipHash over a 12-byte key, 27x per vertex
```

`spatial` is a `HashMap<(i32,i32,i32), Vec<(u32,[f32;3])>>` (SipHash). That is **27 SipHash probes per raw
vertex**; on a representative 6 042-vertex fluid mesh, ~163 k probes per weld — and 26 of every 27 are
near-certain misses.

### The key insight: the grid cell is 1000× epsilon, so 26 of the 27 probes are dead

`cell_size = 0.01`, `epsilon = 1e-5`. A vertex is only ever inserted into its **home** cell, so a neighbor
cell across a given face can hold a within-epsilon match **only if the query vertex is within `epsilon` of
that face**. Since the cell is 1000× epsilon wide, an interior vertex (>1e-3 from every face — the
overwhelming majority) provably has **no** match in any of the 26 neighbors. The old code re-hashed all 26
anyway.

## What shipped — gate each neighbor probe on the face distance

A small `axis_deltas(dist_lower, cell_size, margin)` helper returns, per axis, the offsets to visit:
always `0`; `-1` only when within `margin` of the lower face; `+1` only when within `margin` of the upper
face — in ascending order. The search iterates the cartesian product, preserving the exact `dz→dy→dx`
visit order of the old triple loop:

```rust
let (xds, xn) = axis_deltas(pos[0] - gx as f32 * cell_size, cell_size, margin);
let (yds, yn) = axis_deltas(pos[1] - gy as f32 * cell_size, cell_size, margin);
let (zds, zn) = axis_deltas(pos[2] - gz as f32 * cell_size, cell_size, margin);
'search: for &dz in &zds[..zn] {
    for &dy in &yds[..yn] {
        for &dx in &xds[..xn] {
```

`margin = 1e-3` (100× epsilon). The typical interior vertex now does **1** probe instead of 27; a vertex
near one face does 2; only the (astronomically rare) near-a-corner-on-all-axes case approaches the old 27.

### Behavior-preserving — provable, bit-identical

- We only ever **skip** a neighbor cell, and only one whose nearest possible point is farther than `margin`
  (≥ `margin` > `epsilon`) from the query — such a cell provably contains no point within `epsilon`, so the
  match the old scan would have found is unaffected. Visit order of the cells we *do* probe is unchanged,
  so the first-match-wins result (and `found` index) is identical.
- `margin` (1e-3) sits **100× above epsilon and ~100× above the worst-case f32 rounding error (~1e-5)** in
  computing `pos - g·cell_size` at the largest grid-local coords (~30), so a *live* cell is never wrongly
  skipped. Over-*including* a cell (e.g. a slightly-negative `dist_lower` adding a harmless `-1`) cannot
  change the result — the exact per-vertex `d² < ε²` test inside the loop is untouched. Correctness is
  therefore independent of `margin`'s exact value; it only trades probe count.
- New regression test `test_weld_bounded_search_is_bit_identical` welds a real 6k-vertex MC mesh with both
  the shipped fn and an inlined replica of the original 27-cell scan and asserts **byte-for-byte** equal
  positions (`to_bits()`), normals, indices, and fluid_types. Green.
- `cargo test --workspace`: **voxel-fluid 91, voxel-core 103, voxel-ffi 126, voxel-gen 107, voxel-sleep
  57, voxel-world-memory 53 — 0 failures.**

## Estimated savings (MEASURED, A/B microbench)

`bench_weld_ab` (`#[ignore]`d, in-binary A/B so both variants run on the *identical* real MC mesh; release;
clone+push overhead measured separately and subtracted; 2000 iters × 6 rounds, best-of; two runs):

| side | NET µs/weld |
|------|------------:|
| reference (original 27-cell scan) | **2449 / 2467** |
| **epsilon-bounded scan (SHIPPED)** | **1445 / 1463** |
| **delta** | **−41.0 % / −40.7 % (≈1.70× faster)** |

Both runs agree to within 0.3 pts; ref and new ranges are wholly non-overlapping.

**Honest scoping of the number:**
- The **−41 %** is of `weld_vertices` in isolation, one of five stages in `mesh_fluid`
  (MC → **weld** → QEF refine → smooth → normals). It is not 41% of the whole fluid mesh build.
- It is **not** the ~96% you'd guess from "27 probes → ~1": the 26 skipped probes were mostly *empty-bucket*
  `get`s (one SipHash each, returning `None`), while the home-cell bucket scan, the `entry().or_default()`
  insertions, and the parallel-array compaction are all unchanged. 41% is the realistic structural ceiling
  for this function without changing the hasher or the welding algorithm.
- Where it lands: **lower fluid-worker CPU per dirty chunk per tick** — i.e. during flowing water/lava,
  pools settling, and the mesh churn right after a mine/dig opens a new fluid path. No effect when no fluid
  is moving.
- Scales linearly with raw MC vertex count, so a busier water surface saves proportionally more.

Zero-risk, zero-ABI (voxel-fluid internal), clean `git revert`.

## The lesson for the next pass

The **"spatial-key HashMap probed over a small bounded neighborhood"** family (flagged 2026-06-03) has a
second form: not just "few distinct keys" (→ hoist into a fixed array, as the seam pass did) but **"most
keys in the neighborhood are provably empty for this query"** (→ gate the probe on a cheap geometric test).
When the search radius (`epsilon`) is far smaller than the bucket cell, the neighbor ring is dead weight.

## Next lever flagged (not taken)

Same file, `mesh.rs` lines ~334–352 of the **older** duplicate weld at the bottom of the test/legacy block
(`weld_vertices` calls at 969/993 are tests; confirm whether the second public-ish weld path is live before
touching it). Also: the home-cell bucket itself is a `Vec<(u32,[f32;3])>` scanned linearly — for the rare
hot cell with many coincident verts this is O(k²) over the cell; a first-fit on exact-equal `to_bits()`
position would shortcut the common exact-duplicate case (MC shared-edge verts are *bit-equal*, not merely
within epsilon) before the float-distance scan. Small win, needs care to stay bit-identical.

## Other commits reviewed — no action needed
- The 2026-06-01→03 terrain-meshing sweep confirmed intact.
- `Mesh::smooth` / `smooth_fluid_mesh` adjacency: left alone (float-summation-order, see trap above).
- `recalculate_fluid_normals` / `recalculate_normals`: single-pass, already minimal.
- `qef_refine_vertices`: per-vertex QEF solve, arithmetic-bound, no structural lookup defect.
