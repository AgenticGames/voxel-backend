# Perf review — 2026-05-31 (c) — scheduled "review latest commits" task

**Reviewer:** Claude (autonomous scheduled run, third pass of the day). **Scope:** latest commits on
`main`, headed by today's two perf passes — `ef56a78` *"hoist loop-invariant HashMap lookups out of
stress recalc inner loop"* (the 05:35 run, pass **b**) and `c50532e` *"drop TEMP VFX diagnostics from
hot paths"* (the 03:46 run, pass **a**) — both of which sit on top of the feature commit `89bd270`
*"Sleep montage backend."*

Pass **b** hoisted the **per-chunk** map lookups (`support_scores`/`stress_fields` keyed by the
loop-invariant `(cx,cy,cz)`) out of the `gs³` Pass-2 loop, and explicitly left the **per-voxel
neighbor probes** as the next thing dominating that loop. This pass attacks exactly those.

## Finding: redundant cross-chunk `sample_world` probes for *same-chunk* neighbors

`recalc_stress_region_v2`'s Pass 2 ([voxel-core/src/stress/calc_v2.rs](voxel-core/src/stress/calc_v2.rs))
classifies every solid voxel. For the **grounded-interior** voxels (`support >= ground_threshold` —
the bulk of any mostly-solid chunk during the initial-load / zone-stream / save-load storm that
`89bd270` newly funnels through here) it ran a small classification block that calls
`sample_world` **7 times per voxel**:

- `sample_world(wx, wy-1, wz)` — the floor/ceiling test, plus
- `sample_world(wx±1/​wy±1/​wz±1)` — 6 face neighbors for the air-neighbor count.

Every one of those `sample_world` calls does `world_to_chunk_local` (a `div_euclid`/`rem_euclid`
decompose) **and a fresh `density_fields.get(&key)` HashMap probe** (default **SipHash** over a 12-byte
tuple). But for any voxel that isn't sitting on the chunk's outer shell, **all 7 neighbors live in the
very same chunk** — the `DensityField` the loop *already holds* in `df`. So those were ~7 redundant
chunk-key re-hashes + HashMap probes per grounded voxel, purely to re-find a field already in hand.

The heavier per-voxel path, `calc_voxel_stress_v2` (the non-grounded/surface voxels), had a smaller
sibling waste: it computed `below_solid = sample_world(wx, wy-1, wz)` **twice** — once for the
floor-protection early-return and again, identically, for the final surface classification.

## What I changed
1. Added `neighbor_solid_same_chunk(df, density_fields, cs, nlx,nly,nlz, wnx,wny,wnz)`: when the
   neighbor's local coords all fall **strictly inside `[0, cs-1]`**, it reads `df` directly (which
   `world_to_chunk_local` proves is **bit-identical** to what `sample_world` would return for that
   point); otherwise — genuine cross-chunk neighbors on the chunk face, including the shared `cs`
   overlap row — it **falls through to `sample_world` unchanged**. Returns `Option<bool>` with the
   same `None`-for-unloaded contract, so callers' `unwrap_or(true)` (below = solid) and the
   air-neighbor `Some(false)` test keep their exact semantics.
2. Routed both interior-skip blocks (`recalc_stress_region_v2_filtered` **and**
   `..._with_load_decay`) through it — eliding up to **7 HashMap probes per grounded voxel** away
   from the chunk shell.
3. Computed `below_solid` **once** in `calc_voxel_stress_v2` and reused it for the surface
   classification — one fewer `sample_world` per surface voxel that reaches the span path.

**Behavior-preserving:** the fast path returns the same value `sample_world` did (provable: when all
three neighbor local coords are in `[0, cs-1]`, `world_to_chunk_local` maps them back to `(cx,cy,cz)`
+ those exact indices → the same `df`); boundary neighbors still go through `sample_world`. The
dedup is a literal CSE of two identical pure calls.

`cargo test --workspace`: **voxel-core 101, voxel-ffi 125, voxel-fluid 90, voxel-sleep 106, … all
green, 0 failures** (the 19 active `stress::tests` — slab coherence, ground-connectivity, strut
reduction, tunnel stability — all pass unchanged).

## Estimated savings (MEASURED, A/B microbench)
Timed `recalc_stress_region_v2` over a 3×3×3 region of `cs=30` chunks (the live override), release
build, `git stash` A/B of just this diff, 5 runs/side:

| scene | baseline mean | optimized mean | delta |
|-------|--------------:|---------------:|------:|
| mixed tunnel-carved | 381.9 ms/iter | 376.9 ms/iter | **−1.3%** (best run −1.6%) |

**Honest caveat (same one pass b flagged):** this microbench's wall-time is **dominated by
`ground_connectivity_pass`** — the full-grid top-down flood + relaxation, which I do **not** touch.
A control run on a near-solid "dense" chunk (3-voxel cave, where grounded-interior voxels are ~all of
them) came out **383 ms ≈ the mixed 377 ms** — i.e. the flood cost is essentially scene-independent
and swamps Pass-2 in the total. So the **~1.3–1.6% is heavily diluted**; the saving *within Pass-2's
classification loop itself* (the only thing I changed) is a much larger fraction of that loop, it's
just a small slice of the whole call.

Net effect, where it lands:
- **Per-generated-chunk VFX precompute** (`89bd270`'s worker-thread path): fewer HashMap probes per
  grounded voxel during the load/stream/save storm, multiplied across the hundreds of chunks that
  come in at startup/zone-in. Reduces worker CPU during the storm, not steady frame-time.
- **Live mining + deep-sleep passes** (`_with_load_decay`): same per-voxel trim every recalc, plus
  one fewer `sample_world` per surface voxel in `calc_voxel_stress_v2`.

Modest but free and zero-risk: a same-chunk fast path + a common-subexpression dedup, both provably
behavior-preserving, on the loop the last two passes left as the remaining per-voxel hot spot.

## The real lever (flagged, NOT taken this pass)
Three consecutive passes now confirm the same thing by measurement: **`ground_connectivity_pass`
dominates `recalc_stress_region_v2`'s wall-time**, and it's scene-independent (it floods the whole
`gs³` grid + runs `support_propagation_iterations` relaxation sweeps regardless of how much is solid).
The per-voxel-probe hoists (passes b + c) are near the point of diminishing returns. **The next
worthwhile win is the flood itself** — e.g. skipping the relaxation sweep for chunks whose column
flood already converged (no air gaps → every solid cell is fully grounded at score 1.0, so
relaxation can't change anything), or early-outing columns with no air. That's a behavioral change to
a load-bearing function, so I'm leaving it for a human-reviewed pass rather than doing it
autonomously.

## Other commits reviewed — no action needed
- `89bd270`'s spin-retry `query_surface`/`is_solid_at_ue` and the `solidity_at_ue` tri-state helper:
  already lean (one chunk lookup + one voxel read), as pass **b** noted.
- Worker `heartbeat.rs`: silent in the common case, cheap clock reads only around request handlers.
- The VFX block's per-chunk `stress_config.clone()` is tiny POD — left alone.

## Caveat for your review
voxel-core-internal only — **no FFI/ABI surface touched**, no struct layout change, clean
`git revert` if you disagree. The one thing to eyeball is the boundary condition in
`neighbor_solid_same_chunk`: the `< cs` (not `<= cs`) bound is deliberate — local index `cs` is the
shared overlap row that `world_to_chunk_local` resolves into the *next* chunk, so it correctly falls
through to `sample_world` rather than reading `df`'s overlap cell.
