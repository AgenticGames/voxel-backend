# Perf review — 2026-06-03 — scheduled "review latest commits" task

**Reviewer:** Claude (autonomous scheduled run). **Scope:** latest commits on `main`, headed by the
2026-06-01/02 run of dense-key-HashMap kills in the dual-contouring core:

- `89e67e6` — flat epoch remap in `bucket_mesh_by_material` (FFI convert path)
- `e1679be` — flat sentinel vertex-map in DC `generate_mesh`
- `e1175fd` — u32-indirection QEF pool in `solve_dc_vertices`

Those three swept the **intra-chunk** DC stages (solve → mesh → convert). The one stage on the
chunk-arrival path they did **not** touch is the **cross-chunk seam pass**. This run takes it — after
first avoiding a trap that would have wasted the whole pass on dead code.

## Trap avoided: the obvious seam target is test-only

The natural first hit when grepping the seam code is
[`dual_contouring/seam.rs`](voxel-core/src/dual_contouring/seam.rs) `stitch_seam`, which carries a textbook
offender: `let mut vertex_map: HashMap<(bool, usize), u32> = HashMap::new();` (a fresh SipHash map keyed
by a dense cell index, probed per quad cell). It looks perfect.

**It is dead on the live path.** Every caller of `stitch_seam` is inside its own `#[cfg(test)]` module —
the running engine stitches seams through `voxel_gen::region_gen::generate_chunk_seam_quads` instead
(called from [`worker/seam.rs`](voxel-ffi/src/worker/seam.rs), [`worker/brush.rs`](voxel-ffi/src/worker/brush.rs),
and [`worker/sleep_morph.rs`](voxel-ffi/src/worker/sleep_morph.rs)). Optimizing `stitch_seam` would have
measured a clean win on a function the game never calls. Verifying the caller graph *before* committing is
the whole reason this pass landed on the right function.

## The real target: `generate_chunk_seam_quads`'s per-cell neighbor probe

[`region_gen.rs`](voxel-gen/src/region_gen.rs) `generate_chunk_seam_quads` runs once per meshed chunk's
seam pass — the same chunk-arrival critical path as the three commits above (worker generate, every
brush/mining edit, store rebuild, sleep-morph re-mesh). `region_gen.rs`'s seam path appears in **zero**
prior perf commits.

For each boundary edge it forms a 4-cell quad and resolves each cell's DC vertex from the chunk that
actually owns it:

```rust
for (i, &(cell_x, cell_y, cell_z)) in cells.iter().enumerate() {
    let chunk_dx = if cell_x >= gs_i { 1 } else { 0 };
    let chunk_dy = if cell_y >= gs_i { 1 } else { 0 };
    let chunk_dz = if cell_z >= gs_i { 1 } else { 0 };
    let neighbor_key = (chunk_key.0 + chunk_dx, chunk_key.1 + chunk_dy, chunk_key.2 + chunk_dz);
    ...
    if let Some(neighbor) = all_seam_data.get(&neighbor_key) {   // hash probe, 4× per edge
```

`all_seam_data` is a **`std::collections::HashMap<(i32,i32,i32), ChunkSeamData>`** — SipHash over a 12-byte
key — and `.get(&neighbor_key)` fires **once per quad cell = 4× per boundary edge**. For the realistic
cs=30 grid that is **4 × 117 ≈ 468 SipHash probes per call**.

### The key insight: those 468 probes resolve to at most 8 distinct chunks

`neighbor_key = chunk_key + (dx, dy, dz)`, and each delta is **provably 0 or 1**. Boundary-edge coords run
`0..=gs`, the 4 quad cells are `coord` or `coord-1`, so a cell coord maxes at `gs` and `cell >= gs` flips a
delta to **at most 1**. The whole edge loop therefore references only the **8 chunks of this 2³ block** —
yet the old code re-hashed the key for every single cell.

## What shipped — hoist the 8 neighbor refs out of the loop

```rust
// Resolve the <=8 neighbor chunks ONCE; index by dx | dy<<1 | dz<<2 inside the loop.
let neighbors: [Option<&ChunkSeamData>; 8] = std::array::from_fn(|i| {
    let dx = (i & 1) as i32;
    let dy = ((i >> 1) & 1) as i32;
    let dz = ((i >> 2) & 1) as i32;
    all_seam_data.get(&(chunk_key.0 + dx, chunk_key.1 + dy, chunk_key.2 + dz))
});
...
let neighbor_slot = (chunk_dx | (chunk_dy << 1) | (chunk_dz << 2)) as usize;
if let Some(neighbor) = neighbors[neighbor_slot] {   // array read, no hashing
```

`4 × edge_count` SipHash probes → **8 probes total** + one array index per cell.

### Behavior-preserving — provable

- The 8 prefetched slots are exactly the keys the old code could form (`dx,dy,dz ∈ {0,1}`), so every cell
  resolves to the **same** `Option<&ChunkSeamData>` it did before — a missing chunk still yields `None`,
  which still drops the quad via the `valid = false; break` path. Identical control flow.
- Prefetching all 8 (vs. lazily on first use) only changes *when* the lookups happen, not their results —
  the references are read-only and the map is not mutated during the loop.
- Output is **bit-identical**: the bench prints `460 verts / 230 tris` and an identical `sink`
  (vertex+triangle count summed over all calls) on both A/B sides.
- `cargo test --workspace`: **voxel-gen 106, voxel-core 103, voxel-ffi 126, voxel-sleep 57,
  voxel-world-memory 53 — 0 failures.** The existing `region_gen` seam tests
  (`stitch_produces_seam_geometry`, `boundary_edges_detected_for_neg_face`, …) pass unchanged.

## Estimated savings (MEASURED, A/B microbench)

`git stash` A/B of just this diff, **release** build. Bench
([`voxel-gen/examples/bench_seam_quads.rs`](voxel-gen/examples/bench_seam_quads.rs)): a 2³ block of
adjacent chunks built from a sinusoidal terrain sheet at cs=30 (the live UE override) so the origin
chunk's +face/+edge/+corner neighbors are all present — 117 boundary edges — `generate_chunk_seam_quads`
timed in isolation over 5 000 calls × 6 rounds, best-of:

| side | best µs/call |
|------|-------------:|
| baseline (`all_seam_data.get(&neighbor_key)` per cell) | **15.897** |
| **hoisted 8-slot neighbor array (SHIPPED)** | **6.537** |
| **delta** | **−58.9 % (≈2.43× faster)** |

Rounds are wholly non-overlapping (baseline 15.9–16.7, shipped 6.5–6.7), so the win is real, not noise.

**Honest scoping of the number:**
- The **−59 %** is of `generate_chunk_seam_quads` **in isolation**, one stage of chunk finalization
  (density → hermite → solve → `generate_mesh` → **seam** → convert). It stacks with the 2026-06-01/02
  trio, which trimmed the other stages — all on the same per-chunk path.
- The win scales with **boundary-edge count**: a flatter/emptier chunk face has fewer seam edges and a
  smaller absolute saving; a busy face (terrain crossing the boundary at many cells) has more and saves
  more. 117 edges is a representative mid-load face.
- The remaining ~6.5 µs is the genuine geometry work (NaN/fallback checks, 4 vertex pushes per quad with
  `Vec` growth, 2 cross-product degeneracy tests) — arithmetic, not structure overhead. This is close to
  the last *structural* win in this function.
- Where it lands: **lower worker-thread CPU during the chunk-arrival storm** (startup, zone-in, save-load
  restore) and on **every brush/mining edit, store rebuild, and sleep-morph re-mesh** that re-runs the
  seam pass. It does **not** change steady-state frame time when no chunks are being (re)meshed.

Zero-risk, zero-ABI (voxel-gen internal, no FFI surface), clean `git revert`.

## The lesson for the next pass

**Verify the caller graph before optimizing.** The most obvious dense-key-HashMap offender in the seam
code (`stitch_seam`) is test-only; the live path is a *different* function in a *different* crate. A
microbench will happily report a beautiful speedup on dead code. Grep for non-test callers first.

Second: the dense-key-HashMap family has a sibling worth watching — the **spatial-key HashMap probed in a
loop whose key varies over a tiny bounded neighborhood** (`(i32,i32,i32)` chunk coords with `{0,1}`
deltas). When the distinct-key count is a small constant, hoist the lookups into a fixed-size array and
index by the packed deltas.

## Next lever flagged (not taken)

Same file, `sync_region_boundary_densities` / `generate_seam_mesh` and the `density_fields.contains_key`
+ double-`density_fields[&key]` index pattern around lines 786–877 of `region_gen.rs`: a `contains_key`
immediately followed by two more `[&key]` indexes re-hashes the same key 3×. That is a region-build (not
per-frame) path, so lower priority, but it is the same shape and a clean follow-up.

## Other commits reviewed — no action needed
- The 2026-06-01/02 DC trio (`89e67e6`/`e1679be`/`e1175fd`) confirmed still in place and intact.
- `is_degenerate` is pure arithmetic on 3 vertices — already minimal.
- `ChunkSeamData.boundary_edges` is already a `Vec` (not a map), so its iteration was never the issue —
  only the *neighbor* lookup was.
