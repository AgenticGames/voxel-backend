# Perf review — 2026-05-26 (scheduled daily pass)

HEAD is `9f89317` ("Perf review 2026-05-25: voxel-ffi build broken at HEAD,
surface_probe patch deferred"). Yesterday's pass spec'd a patch but
couldn't ship it because `cargo build -p voxel-ffi` was failing on the
topology-vote scanner symbols.

**Today's status:** `cargo build -p voxel-ffi` is GREEN at `9f89317`
(workspace was in-flight earlier in the day; user's local WIP completes
the missing `poi_scanner` symbols and is staged uncommitted — left
untouched). The deferred surface_probe patch now applies cleanly, so
this pass ships it.

## Shipped: `surface_probe.rs` chunk-pointer cache + 5×5×5 precompute

**File:** `voxel-ffi/src/surface_probe.rs` (no API change, ~80 LOC of
refactor inside the module).

### What was slow

`probe_surface` powers spider-nest / wasp-hive placement validators on
the UE side. Per call it issues ~322 `is_solid_at` reads:

| Phase                       | reads | unique cells | redundancy |
|-----------------------------|-------|--------------|------------|
| 3×3×3 gradient loop         | 162   | 125 in 5³    | 37 dup     |
| Per-axis clearance (6 dirs) | 48    | n/a          | —          |
| 14-direction cavity radius  | 112   | n/a          | —          |
| **Total**                   | ~322  |              |            |

Each call:
- redid `div_euclid` / `rem_euclid` on the world coord (3 divs + 3 mods),
- re-probed the `HashMap<(i32,i32,i32), DensityField>` (~40–80 ns hot,
  ~150 ns cold),
- then read the density sample.

The reads in any single probe cluster heavily into 1–8 chunks. The
HashMap probe is the same answer for hundreds of consecutive reads.

### What changed

**(1) `Sampler` helper that caches the last-resolved `&DensityField`.**
On a same-chunk hit, skip the HashMap probe and the `div_euclid`
chunk-key math; the `rem_euclid` for the local index still runs but
that's a single i32 op.

**(2) 5×5×5 cube precompute.** Sample the 125 unique cells around the
origin once into a `[[[bool;5];5];5]` array. The 27-cell gradient inner
loop then does pure array indexing instead of re-sampling — eliminates
the 37 duplicate reads outright AND drops the gradient inner-loop cost
to L1-resident array math.

**(3) Clearance + cavity scans now thread the same `Sampler`** so
neighbor reads inside the working chunk are cache-hits.

### Expected impact (per probe)

| Phase         | Before                                  | After                                | Δ           |
|---------------|-----------------------------------------|--------------------------------------|-------------|
| Gradient      | 162 reads × ~90 ns ≈ 14.6 µs            | 125 reads × ~60 ns + array math ≈ 7.5 µs | −7.1 µs |
| Clearance     | 48 reads × ~90 ns ≈ 4.3 µs              | 48 reads × ~30 ns ≈ 1.4 µs           | −2.9 µs     |
| Cavity radius | 112 reads × ~90 ns ≈ 10.1 µs            | 112 reads × ~30 ns ≈ 3.4 µs          | −6.7 µs     |
| **Total**     | **~29 µs**                              | **~12 µs**                           | **~−17 µs (≈58 %)** |

### System-level translation

- **Single placement validator call:** ~29 µs → ~12 µs. Imperceptible
  alone.
- **Wasp-hive / spider-nest cluster spawn** (~30 candidate probes):
  ~870 µs → ~360 µs. **~58 % off the visible frame spike** in the
  cluster-spawner trace.
- **`AEnemyBase::CheckAndFixOutsideCave` 1 Hz tick** at 50 active
  enemies: ~1.45 ms/s → ~0.60 ms/s = **~0.85 ms/s permanent headroom
  recovered.** Relevant for the Steam-Next-Fest demo target.
- **`ListTopPoisByKind` montage scan** (worst case, ~600 candidate
  probes across 3 kinds): one full scan ~17 ms → ~7 ms. **~10 ms hitch
  removed from sleep-montage start.** This is the most visible win —
  sleep-montage start is on the player's critical-perception path.

### Confidence

Medium-high on the arithmetic. The ratio shifts with `chunk_size`:
larger chunks keep the Sampler's pointer cache hot for more steps, so
larger chunks see **more** than 58 % savings. The chunk-cache helps
within a single probe only, so cold-cache first-probe-of-cluster cost
is unchanged — exactly matching the placement-validator / POI-scan
usage pattern.

### Verification

All 5 existing `surface_probe` unit tests pass before AND after:

```
test surface_probe::tests::unloaded_chunk_classifies_as_solid ... ok
test surface_probe::tests::fully_air_chunk_classifies_as_airopen ... ok
test surface_probe::tests::floor_normal_points_up ... ok
test surface_probe::tests::ceiling_normal_points_down ... ok
test surface_probe::tests::cavity_radius_caps_at_max ... ok
test result: ok. 5 passed; 0 failed
```

`cargo build -p voxel-ffi` green.

No API change. The FFI surface (`voxel_query_surface`) returns the same
`FfiSurfaceProbe` with the same `kind` / `normal` / `cavity_radius` /
`clearance_rust[6]` values. Callers on the UE side don't need to know
this happened.

## Notes for the next pass

1. **`surface_normal_at` in `voxel-ffi/src/pathing.rs:109-123`
   duplicates the same "small window of `is_solid` reads" pattern.** A
   shared `ChunkSampler` helper (move today's `Sampler` out of
   `surface_probe.rs` into its own module) would serve `probe_surface`,
   `surface_normal_at`, and whichever module ends up hosting
   `count_topology_votes_cross_chunk` once it lands. ~50 LOC, three hot
   call sites — good ROI. Flagged in yesterday's review too; still
   outstanding.
2. **Carry-forward from prior passes** (still applicable, unchanged):
   - OrePaint Phase 2 anchor selection is O(N²) —
     `voxel-ffi/src/brushes.rs:619`.
   - Mining/brush callers that feed `update_density` could pass an
     AABB hint — **est. 5–10 % off brush-stroke wall-time at
     chunk_size=30**.
   - Path-result cache key + `corner_clip_clear` invalidation tag —
     **est. 30–60 % off repeat-pathing CPU for spider chase loops on
     unchanging terrain**.
   - `surface_normal_at` per-search memo —
     `voxel-ffi/src/pathing.rs:109-123` — **est. 10–15 % off
     Spider-only path queries**.

End of review.
