# Perf review — 2026-05-29 (daily scheduled run)

Scheduled daily perf-improvement pass. Picked the
**OrePaint Phase 2 anchor selection** O(N²) item flagged in
[PERF_REVIEW_2026-05-26_b.md](PERF_REVIEW_2026-05-26_b.md) under
"Outstanding from earlier passes":

> OrePaint Phase 2 anchor selection is O(N²) —
> `voxel-ffi/src/brushes.rs:619`.

## What was slow

[voxel-ffi/src/brushes.rs:637-651](voxel-ffi/src/brushes.rs:637)
(the `paint_ore_deposits` Poisson-disk-ish anchor selector) walked
every candidate and rejected it via an exhaustive `accepted.iter().any()`
scan:

```rust
let too_close = accepted
    .iter()
    .any(|a| (a.world_pos - cand.world_pos).length_squared() < min_spacing2);
```

For N candidates and K accepted anchors that grows as O(N · K). On a
big brush (radius 50 voxels, tight `min_spacing=2`, density slider 1.0)
K hits ~2 500 and the inner scan dominates Phase 2 wall-time.

## Fix — bounded-window spatial hash with a flat Vec backing store

[voxel-ffi/src/brushes.rs:637-700](voxel-ffi/src/brushes.rs:637).
Bucket already-accepted positions into a 3D cell grid with cell side
= `min_spacing`. Any pair closer than that lives in the same or a
26-neighbor cell, so each candidate only ever scans the 3×3×3 cell
window around its own bucket (at most a handful of points each).

First cut used `std::collections::HashMap<(i32,i32,i32), Vec<Vec3>>` —
the SipHash probe cost actually **slowed the function by ~3 %** on the
stress brush (HashMap probes are ~30–50 ns each; the saved distance
checks are ~3–5 ns each). Replaced with a flat `Vec<Vec<Vec3>>` of
size `dim_x · dim_y · dim_z` (brush AABB rounded out to cell coords —
~20 k cells for a max-size brush, ~470 KB peak). Lookup is now a
single bounds check + linear index, no hashing.

### Determinism

Same iteration order over `candidates` + exhaustive distance check
within the bounded window → identical acceptance set as the prior
O(N · K) scan. Verified by `ore_paint_seed_determinism` (bit-identical
material map after the same seed).

## Measured impact

New bench `bench_ore_paint_large_brush` (added under `#[ignore]`,
`voxel-ffi/src/brushes.rs:3861`): 3×3×3 grid of solid 64-vox chunks
with a 40-vox cavity carved in the middle, then paints a radius-50
OreDeposit at the center. 5 release runs:

| Configuration                                  | Baseline   | Patched    | Δ        |
|------------------------------------------------|-----------:|-----------:|---------:|
| Stress: `cluster=1, min_spacing=2, density=1.0`| **102.3 ms** | **74.1 ms** | **−27.6 %** |
| Default: `cluster=1.5, min_spacing=4, density=0.05` (UE-side default) | 38.6 ms | 36.4 ms | −5.7 % |

* Stress case is what shows up if a player cranks density to maximum
  and shrinks spacing on a large brush — Phase 2 went from O(K²)
  rejection scans to constant ~27 cell probes per candidate. ~28 ms
  per stroke saved.
* Default case (UE-side `OreDensity=0.05`, `OreMinSpacing=4.0`,
  `OreClusterSize=1.5`) is the realistic hit: K stays small enough
  that the old linear scan was already fine, so the win is modest
  (~2 ms per stroke). Cleanup is still worth it: cost is constant
  with K instead of quadratic, so no future surprise when density
  defaults shift or larger brushes ship.

Cluster-write Phase 3 still dominates total wall-time in both rows
(it's the big number under the bench's `paint_ore_sphere_voxels`
loop). This change touches Phase 2 only.

## Verification

* `cargo build -p voxel-ffi --release` — clean (only the existing
  20 warnings, none new).
* `cargo test --release -p voxel-ffi --lib ore_paint` — 6/6 passed
  including `ore_paint_seed_determinism` (bit-identical layout).
* `cargo test -p voxel-ffi --release --lib` — 121 passed, **4
  pre-existing failures unchanged**
  (`api::tests::stats_reports_correctly`,
  `delta::tests::binary_roundtrip_with_data`,
  `delta::tests::realistic_chunk_size_roundtrip`,
  `delta::tests::snapshot_roundtrip` — same baseline noted in the
  strut-overhaul memory and prior perf reviews).

## Estimated player-facing savings

For a max-density OrePaint brush stroke at radius 50 voxels:
**~28 % faster Phase 2** (102 → 74 ms total brush time). For the
default Atelier slider preset: ~6 %.

End-of-pass.
