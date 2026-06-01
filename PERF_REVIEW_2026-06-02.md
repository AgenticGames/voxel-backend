# Perf review — 2026-06-02 — scheduled "review latest commits" task

**Reviewer:** Claude (autonomous scheduled run). **Scope:** latest commits on `main`, headed by the
2026-06-01 perf trio that each took the last provably-safe win in its hot path and declared it
exhausted for autonomous work:
- `89e67e6` (run *c*) — flat epoch-tagged remap in `bucket_mesh_by_material` (~76% off that call),
- `8e61222` (run *b*) — hoisted `tick_chunk`'s cross-chunk HashMap probes (~7–8% off fluid tick),
- `9f03c5e` (run *a*) — hoisted the stress flood's per-solid-cell `scores.get_mut` (~35% off
  `ground_connectivity_pass`).

Runs *a*/*b* flagged their remaining levers as **behavioral** (algorithmic early-outs reserved for a
human pass). Run *c* moved to an untouched subsystem — the FFI mesh **conversion** path
(`convert.rs`) — and noted DC solve / hermite extract / bucketing as the stages of chunk meshing. **This
run moves one stage upstream of *c*, into the dual-contouring core itself** — `voxel-core/src/dual_contouring/`,
which (like `convert.rs` was before *c*) appears in **zero** prior perf commits.

## Finding: `generate_mesh`'s per-corner vertex dedup used a HashMap over a DENSE cell key

[`dual_contouring/mesh_gen.rs`](voxel-core/src/dual_contouring/mesh_gen.rs) `generate_mesh` runs for
**every meshed chunk** — worker generate (7 call sites in `worker/generate.rs`), brush edits, sleep-morph,
store rebuilds, slabs, pile previews — squarely on the chunk-arrival critical path, the same path *c*
optimized one stage later.

For each sign-changing edge it visits up to 4 quad-corner cells and dedups their DC vertices through:

```rust
let mut vertex_map: FastHashMap<usize, u32> = FastHashMap::default();   // fresh map PER CALL
...
let vi = *vertex_map.entry(cell_idx).or_insert_with(|| { /* push vertex */ });   // probe PER corner
```

`cell_idx` is a **dense** linearized cell index in `0..dc_vertices.len()` — and `dc_vertices.len()` is
exactly `grid_size³` (`solve_dc_vertices` returns a `vec![sentinel; grid_size³]`). So the key space is a
contiguous `0..N` range, and a hash map is the wrong structure — the same observation run *c* made about
`bucket_mesh_by_material`'s per-bucket dedup map. Two costs were paid per `generate_mesh` call:

1. **~4 × edge_count hash probes** through hashbrown's `entry()` machinery (control-byte SIMD scan +
   `or_insert_with` closure), even though `FastHashMap` already uses the cheap `IdentityHasher`.
2. **A fresh map allocated per call that grows from empty** to ~surface-cell entries, **rehashing ~10×**
   on the way (0→4→8→…→4096 capacity) — pure alloc/rehash churn repeated for every chunk.

## What I changed (1 file, voxel-core internal — no FFI/ABI surface)

Replaced the map with a single flat sentinel array indexed by `cell_idx`:

```rust
// u32::MAX = "no vertex assigned yet for this cell"; a real index can never reach u32::MAX.
let mut vertex_map = vec![u32::MAX; dc_vertices.len()];
...
let mut vi = vertex_map[cell_idx];
if vi == u32::MAX {
    vi = mesh.vertices.len() as u32;
    mesh.vertices.push(Vertex { position: pos, normal: intersection.normal, material: intersection.material });
    vertex_map[cell_idx] = vi;
}
```

- **One allocation** (`vec![u32::MAX; grid_size³]`, ~108 KB for cs=30 = one memset) instead of a map
  that reallocates and rehashes as it grows — **every probe becomes a raw array index, no hashing**.
- The bounds guard `if cell_idx >= dc_vertices.len() { continue; }` already sits above the lookup, so
  `cell_idx` is always a valid index into the flat array.

**Behavior-preserving — provable, output bit-identical:**
- Edges are iterated in the **same** `hermite.edges` order (untouched), so the **first** edge that
  references a given cell is the same in both versions — and a new index is still assigned as
  `mesh.vertices.len()` at that first encounter, pushing the vertex with **that** edge's
  `normal`/`material`. Identical vertices, pushed in identical order, with identical indices.
- Subsequent references to the same cell return the stored index (`vi != u32::MAX`) — the identical
  dedup the map's `entry()` did.
- `u32::MAX` is an impossible real value: a stored index is `mesh.vertices.len()` (≤ surface-cell count
  ≪ 4e9), so the sentinel can never collide with an assigned index.
- `vertex_map[cell_idx].is_some()`-equivalent is now `!= u32::MAX`; nothing else reads/writes the array.

New unit test `dedup_shares_vertices_across_edges`: two overlapping Z-edge quads whose 8 corner
references span 6 distinct cells must dedup to exactly 6 vertices (would fail if the sentinel logic
double-pushed or mis-keyed). All existing `mesh_gen` / `dual_contouring` tests unchanged.

`cargo test --workspace`: **voxel-core 102** (101 + the new test), **voxel-ffi 126, voxel-fluid 90,
voxel-sleep 106 — 0 failures.**

## Estimated savings (MEASURED, A/B microbench)

`git stash` A/B of just this diff, **release** build. Bench (`voxel-core/examples/bench_mesh_gen.rs`): a
realistic chunk-like hermite — a **sinusoidal terrain sheet through a cs=30 grid** (the live UE override),
2,722 sign-changing edges, 2,604 surface cells, multi-material — solved **once**, then `generate_mesh`
timed in isolation over 2,000 calls × 6 rounds, best-of (warm rounds; round 0 cold excluded):

| side | round means | best µs/call |
|------|-------------|-------------:|
| baseline (per-corner `FastHashMap`) | 313–318 | **310.36** |
| optimized (flat sentinel `Vec<u32>`) | 96–98 | **95.81** |
| **delta** | | **−69.1 % (≈3.2× faster)** |

Ranges are wholly non-overlapping (baseline 310–318, optimized 95–98), so the win is real, not noise.

**Honest scoping of the number:**
- The **−69 %** is of `generate_mesh` **in isolation**, not a whole chunk build (which also runs the
  density sample, hermite extract, `solve_dc_vertices`, and `convert_mesh_to_ue_scaled`). This trims
  the **mesh-emit stage** by roughly two-thirds.
- The win is larger than a pure per-probe saving because the old map was **reallocated and regrown every
  call**; eliminating that alloc/rehash churn dominates. So the saving scales with **call count** (chunks
  meshed), and is realized on **every** chunk — not just dense ones.
- Where it lands: **lower worker-thread CPU during the chunk-arrival storm** (startup, zone-in, save-load
  restore) and on **every brush/mining edit, seam re-stitch, slab, and sleep-morph re-mesh**. It does not
  change steady-state frame time when no chunks are being (re)meshed.

Zero-risk, zero-ABI, clean `git revert`: a dense-key-HashMap → flat-array swap mirroring run *c*'s
`bucket_mesh_by_material` win, validated by the full suite + a two-side A/B.

## The next lever (flagged, NOT taken — same dense-key pattern, one function over)

[`dual_contouring/solve.rs`](voxel-core/src/dual_contouring/solve.rs) `solve_dc_vertices` has the
**identical** anti-pattern: `qefs: FastHashMap<usize, QefData>` keyed by the same dense `0..grid_size³`
cell index, accumulated per edge then drained. The same flat-array swap applies — but `QefData` is an
accumulator (~10 floats), so a flat `Vec<QefData>` is a larger allocation and the drain loop would either
scan all `grid_size³` cells or need a side `Vec` of touched indices to keep the surface-sparse iteration.
That's a slightly bigger, less trivially-bit-identical change (iteration becomes index-ordered rather than
hash-ordered — the *output* `vertices[idx]` is order-independent so it's still provably identical, but it
wants its own focused A/B). Left for the next pass to keep this diff minimal and the win cleanly isolated.

## Other commits reviewed — no action needed
- Runs *a*/*b*/*c* targets (stress flood, fluid `tick_chunk`, `bucket_mesh_by_material`) confirmed still
  in place; their remaining levers are behavioral and reserved for a human-reviewed pass.
- `solve_dc_vertices`'s `AdjacentCells` fixed-array (no heap alloc per edge) is already lean; only its
  `qefs` map is the open lever (above).
- The `is_degenerate_tri` cross-product and winding checks are pure arithmetic per triangle — left alone.
