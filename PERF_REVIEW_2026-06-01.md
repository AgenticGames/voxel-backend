# Perf review — 2026-06-01 — scheduled "review latest commits" task

**Reviewer:** Claude (autonomous scheduled run). **Scope:** latest commits on `main`, headed by the
2026-05-31 perf trio that squeezed `recalc_stress_region_v2` —
`cfe9b43` *"same-chunk fast path for stress Pass-2 neighbor probes"* (pass c),
`ef56a78` *"hoist loop-invariant HashMap lookups out of stress recalc inner loop"* (pass b),
`c50532e` *"drop TEMP VFX diagnostics from hot paths"* (pass a) — all sitting on the feature
commit `89bd270` *"Sleep montage backend"*, which newly funnels a VFX-only stress precompute through
`recalc_stress_region_v2` for **every** chunk during the initial-load / zone-stream / save-load storm.

All three prior passes converged on the same conclusion and **explicitly flagged it for a human pass**:
`ground_connectivity_pass` ([voxel-core/src/stress/calc.rs:428](voxel-core/src/stress/calc.rs))
dominates `recalc_stress_region_v2`'s wall-time, and the next real lever is *inside* it — but they
left it alone because the obvious win (skipping the relaxation flood) is a **behavioral** change to a
load-bearing function. **This pass found a different, provably behavior-preserving win inside the same
function** and took it.

## Finding: the flood's per-solid-cell `scores.get_mut` was never hoisted (and the comment said it couldn't be)

`ground_connectivity_pass` has two phases: (1) a **global top-down column flood** that seeds support
scores, and (2) `support_propagation_iterations` (default **2**) relaxation sweeps. With only 2 sweeps,
the **flood is the bulk** of the function.

Pass **b** hoisted the relaxation loop's `scores.get(&key)` into a per-chunk `current_sf` cache. But
the **flood's write** was left as a per-solid-cell call:

```rust
if cached_in_expanded {
    if let Some(sf) = scores.get_mut(&cached_key) {   // <-- runs per SOLID cell in the column walk
        sf.set(lx, ly, lz, current_score);
    }
}
```

`cached_key` only changes when `cy` changes — i.e. it is constant for the ~`cs` consecutive cells of a
column inside one chunk — yet `scores.get_mut(&cached_key)` re-hashed that 12-byte key (std HashMap =
**SipHash**) on **every solid cell**. The flood already caches `density_fields.get` and the
`expanded_keys` membership across the `cy`-crossing; the `scores` write target changes on exactly the
same boundary, so it was the one per-cell HashMap probe left on the flood's hot path.

The existing code comment claimed this **couldn't** be hoisted:

> *"keeping it lifted would force unsafe (mutable + immutable map borrow simultaneously)"*

**That's incorrect.** `scores` and `density_fields` are **distinct** HashMaps, so holding a
`&mut SupportScoreField` borrowed from `scores` at the same time as a `&DensityField` borrowed from
`density_fields` is sound — no aliasing, no `unsafe`. The borrow checker confirms it compiles cleanly.

## What I changed (1 file, +15/−9, voxel-core internal — no FFI/ABI surface)

Cached the score-field write target alongside the existing `cached_df`, refreshed on the same
`cy`-change boundary:

```rust
let mut cached_sf: Option<&mut SupportScoreField> = None;
...
if cached_cy != Some(cy) {
    cached_cy = Some(cy);
    cached_key = (cx, cy, cz);
    cached_df = density_fields.get(&cached_key);
    cached_sf = scores.get_mut(&cached_key);   // one probe per cy-crossing, not per solid cell
}
...
if let Some(sf) = cached_sf.as_deref_mut() {
    sf.set(lx, ly, lz, current_score);
}
```

**Behavior-preserving — provable:**
- `scores` is initialized to contain **exactly** `expanded_keys` (the `for &key in &expanded_keys {
  scores.insert(key, ...) }` right above), so `scores.get_mut(&cached_key).is_some()` is **identical**
  to the old `cached_in_expanded = expanded_keys.contains(&cached_key)` guard. The dropped
  `cached_in_expanded` flag is fully subsumed by the `Option` being `Some`/`None`.
- The cache key is `cached_key`, refreshed on exactly the same `cy != cached_cy` boundary the write
  target changes on, so every write lands in the same cell it did before, with the same value.
- The `&mut` borrow lives only within one column's inner loop (the `cached_*` vars are declared per
  column), and `scores` is read/written **nowhere else** in the flood — so there is no aliasing with
  any other `scores` access.

`cargo test --workspace`: **voxel-core 101, voxel-ffi 125, voxel-fluid 90, voxel-sleep 106 — all green,
0 failures** (the 19 active `stress::tests` — slab coherence, ground-connectivity grounded/ceiling,
strut reduction, tunnel stability — all unchanged).

## Estimated savings (MEASURED, A/B microbench)

`git stash` A/B of just this diff, **release** build, `ground_connectivity_pass` timed over a 3×3×3
dirty region of `cs=30` chunks (the live override) with a horizontal tunnel band carved through every
chunk (mixed scene), 5 runs/side, two independent A/B rounds:

| side | round 1 mean | round 2 mean | best |
|------|-------------:|-------------:|-----:|
| baseline (per-cell `get_mut`) | 42.60 ms | 42.26 ms | 41.21 ms |
| optimized (hoisted `cached_sf`) | 27.58 ms | 27.42 ms | 26.10 ms |
| **delta** | **−35.3%** | **−35.1%** | **−36.7%** |

**≈ 35% off `ground_connectivity_pass`** in this scene, consistent across rounds. This is far larger
than passes b/c (≈1.3–1.6% off the whole recalc) because it removes a **SipHash probe per solid cell**
from the phase that those passes measured as dominant — and because the flood's per-cell body is
otherwise cheap (a `div_euclid`/`rem_euclid` decompose + an array index), the eliminated probe was a
large fraction of it.

**Honest caveat on the headline number:** the **35% is measured on `ground_connectivity_pass` in
isolation**, not on the full `recalc_stress_region_v2`. My isolated flood (~42 ms for this carved
scene) is a *smaller* absolute number than the ~377 ms the prior pass reported for the whole recalc
over a 3×3×3 region, so the flood's exact share of the full call depends on scene solidity (more solid
→ more solid cells → bigger flood-probe win, but also a heavier Pass-2). What's rock-solid is the
**apples-to-apples ~35% reduction of the function the last three passes named as the bottleneck**.

Where it lands:
- **Per-generated-chunk VFX precompute** (`89bd270`'s worker path): every chunk's stress precompute
  calls `ground_connectivity_pass`; this trims ~a third off that call across the hundreds of chunks
  arriving at startup / zone-in / save-load — i.e. lower worker CPU during the storm.
- **Live mining + deep-sleep recalcs** that re-flood support: same per-call trim.

Zero-risk, zero-ABI, clean `git revert` if you disagree: a `&mut`-cache hoist mirroring the one pass b
already did on the relaxation side, on the one per-cell HashMap probe it left behind.

## The real lever (still flagged, still NOT taken — same as pass c)

This pass took the *safe* win inside `ground_connectivity_pass`. The **behavioral** lever the prior
passes flagged is still open and still left for you: the relaxation sweep (and the flood) re-scan the
full `gs³` grid regardless of how much is air, and skipping converged columns / early-outing air-only
columns would cut the **algorithm's** work, not just its per-cell overhead. That changes results-vs-
tolerance on a load-bearing function, so it wants a human-reviewed pass — not an autonomous one.

## Other commits reviewed — no action needed
- The relaxation loop's `current_sf` cache (pass b), the `neighbor_solid_same_chunk` fast path
  (pass c), and the `below_solid` CSE (pass c) are all still in place and correct.
- `89bd270`'s spin-retry `query_surface`/`is_solid_at_ue` and worker `heartbeat.rs`: already lean
  (confirmed by passes b/c); nothing new.
