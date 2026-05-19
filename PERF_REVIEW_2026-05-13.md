# Performance Review — Recent Commits (2026-05-13)

Scheduled review of commits added since the 2026-05-12 review.

Commits surveyed:
- `267196b` — OrePaint creative brush — wall-exposed deposits + deep channels
- `6539312` — Fluid sim: `has_sources` flag to skip `regen_sources` on source-less chunks
- `a0ebed5` — Fluid sim: reuse quench scratch sets across ticks
- `b2f522a` — Fluid sim: drop unnecessary Y-loop in `equalize_horizontal` (this is the fix I landed yesterday — confirms it shipped clean)

The four most recent fluid-sim commits (`6bbb4dc`, `b2f522a`, `a0ebed5`, `6539312`) are themselves a perf-optimization series that already cleared the obvious wins in `voxel-fluid`. So the audit this run focused on `267196b` (OrePaint, ~720 net-new lines in `voxel-ffi/src/brushes.rs`) which is the only commit that adds new work to a hot path.

---

## Implemented this run — `paint_ore_deposits` Phase 3: per-voxel → per-chunk `get_mut`

**File:** `voxel-ffi/src/brushes.rs` (function `paint_ore_deposits`, Phase 3 cluster + channel writes).
**Status:** ✅ Applied. All 31 `voxel-ffi` brush tests pass (incl. all 6 OrePaint tests — wall-exposed placement, weights honored, seed determinism, zero-weight no-op, density field unchanged, min-spacing anti-clumping). Pre-existing `delta::tests::*` failures (3 tests) are unrelated to this change — they fail on clean HEAD too (density round-trip mismatch `1.01 vs 1.0`).
**Not committed.** Diff is sitting in the working tree for you to review.

### What was wrong

The OrePaint brush's Phase 3 (cluster + channel sphere paints) used a per-voxel write helper:

```rust
fn write_ore_at_world(store: &mut ChunkStore, wx, wy, wz, target, ...) {
    let key = (wx.div_euclid(chunk_size_i), ...);
    let Some(df) = store.density_fields.get_mut(&key) else { return false };
    let lx = wx.rem_euclid(chunk_size_i) as usize;
    // ... single voxel write
}
```

Every single voxel of every cluster sphere AND every channel tube-segment sphere did:
1. Three `div_euclid` + three `rem_euclid` integer ops.
2. One `HashMap<(i32,i32,i32), DensityField>::get_mut` lookup.

This is the slow pattern. Every sibling brush in the same file (`paint_material_sphere` at line 129, `carve_sphere` at line 800, `fill_sphere` at line 873) instead iterates *chunks* in the brush AABB and does **one `get_mut` per chunk**, then iterates voxels locally. Since a typical OrePaint cluster fits inside one chunk (cluster_size is usually 1.0–3.0 voxels) and a channel tube-segment likewise (channel_radius typically 1.0–2.0), this means each cluster/segment was paying N HashMap lookups when it only needed 1.

### What changed

Replaced `write_ore_at_world` with `paint_ore_sphere_voxels(store, cwx, cwy, cwz, r_int, r2, target, ...)`. It iterates the chunks the sphere AABB overlaps (typically 1, up to 8 at a chunk corner), does one `get_mut` per chunk, then walks the local voxel slab clamped to both the chunk and the sphere AABB. Dirty-rect tracking is merged into the per-chunk dirty map at the end, same shape as before.

Same `is_solid && != target` write predicate, same dirty-rect semantics, same materials written — pure structural rewrite.

### Estimated saving

Concrete: for typical settings (target_count=50 anchors, cluster_size=2.0, channel_prob=0.5, channel_length=6, channel_radius=1.5):

| Pass               | Per-anchor voxel writes attempted | Old HashMap lookups | New HashMap lookups |
|--------------------|----------------------------------:|--------------------:|--------------------:|
| Cluster (r=2.0)    | ~33                               | ~33                 | 1 (≤8 at corner)    |
| Channel × 6 steps  | ~14 × 6 = ~84                     | ~84                 | ~6 (1 per step)     |
| **Per anchor total** | **~117**                        | **~117**            | **~7**              |
| **× 50 anchors**     | —                               | **~5,850**          | **~350**            |

That's **~94% reduction in HashMap lookups in Phase 3**. At ~60–100 ns per lookup (rustc-stdlib HashMap with FxHash via default state), this is ~0.35–0.55 ms reclaimed per brush click in Phase 3. Plus the constant integer-arithmetic cost of repeated `div_euclid`/`rem_euclid` per voxel collapses to once per chunk.

Whole-brush wall-time saving, ballpark: **~15–25% per brush click** for typical settings. Higher for clusters with bigger radii (cluster_size=4 → 257 voxels in cluster, ~30% saving). Lower for tiny clusters (cluster_size=0.5 single-voxel "freckles" — barely matters).

**Allocation pressure:** unchanged (the per-anchor work was on the stack already). The optimization is pure inline-loop-fusion + lookup elimination.

### Latency context

OrePaint is a one-shot creative brush — it runs once per LMB click on the worker thread. For Next Fest demo authoring (the system this exists for) the wall-time the user feels is the per-click stutter. Going from ~3 ms → ~2.3 ms is not perceptible; ~30 ms → ~22 ms is. The full brush cost is dominated by Phase 1 (wall-candidate enumeration over the entire brush AABB) and Phase 4 (`finalize_brush` = seam sync + remesh of dirty chunks). Phase 3 is a meaningful but not dominant slice; this is best understood as "bring Phase 3 in line with the established per-chunk pattern the rest of the file uses", and the % above applies to that slice.

### Safety / correctness

- The per-chunk loop visits exactly the same world voxels as the old per-voxel path. AABB derivation `lo_wx = cwx - r_int; hi_wx = cwx + r_int` followed by chunk-clamped `lo_lx..=hi_lx` covers identical integer voxel positions. Inside the loop, the same `d2 = dx*dx + dy*dy + dz*dz; if d2 > r2 { continue; }` filter is applied. Same write predicate (`is_solid && != target`).
- Cross-chunk spans (channel tubes that walk through a chunk boundary) work via the AABB chunk-iteration (`cklo..=ckhi`), which is how `carve_sphere` already handles cross-chunk spheres.
- Dirty rect accumulation: identical behavior — the old code wrote one voxel at a time and grew the per-chunk dirty rect cell-by-cell; the new code grows it per-chunk in a single update at the end of each chunk's slab. Same final rect.
- Determinism: the iteration order is `z → y → x` chunk-major, then `z → y → x` voxel-local within each chunk. Old code iterated `dz → dy → dx` around each anchor. **Iteration order changed.** But OrePaint's RNG is seeded outside the write loop (anchor selection and ore-type pick happen in Phase 2), so write order does not affect output for any seed value. The `ore_paint_seed_determinism` test (which asserts same-seed → identical material map) passes.
- The `ore_paint_density_field_unchanged` test passes — confirming density values are still pure-readonly in Phase 3.
- All 6 OrePaint tests + all 31 brushes-module tests pass with the new implementation.

---

## Other findings (not applied — for your review)

These are below the bar for "implement now" — either small absolute wins, structural change, or test-coverage risk I'd rather you sign off on. Listed in rough impact order.

### A. Rust — OrePaint Phase 1 enumerates every solid voxel in the brush AABB

**File:** `voxel-ffi/src/brushes.rs:478-575` (the wall-candidate scan).

The brush AABB is typically 30³ = 27k voxels at chunk_size=30. Phase 1 iterates every solid voxel and probes its 6 face-neighbors to classify as "wall-exposed". For deep-interior rock, every probe finds a solid neighbor and the candidate is discarded. That's wasted work — the deep interior of a chunk is the bulk of the voxels but contributes zero candidates.

Two possible cheap-cull strategies:

1. **`s.density > 1.0` early-skip.** The DC iso-surface is at `density = 0`; air voxels have `density < 0`. If the SDF is approximately Lipschitz-1 (which DC + the smoothing/clamping passes in this codebase enforce within ±1 voxel), then `density > 1.0` is a sufficient condition for "no face-neighbor is air" and we can skip the 6-probe entirely. Worst case is a missed candidate near a worn seam; acceptable degradation given the brush already random-subsamples via the density slider.

2. **Scan air voxels instead.** Air voxels in the brush AABB are typically far fewer than solid voxels (cavern voxels << host rock voxels in a typical underground scene). For each air voxel, push its 6 solid neighbors as candidates; dedup at the end. This inverts the problem — O(air_voxels × 6) instead of O(solid_voxels × 6).

Strategy 1 is 3 lines of code; strategy 2 is a structural rewrite. **Estimated saving with strategy 1**: ~30–50% off Phase 1 wall-time on caverns where solid voxels outnumber wall voxels ~10:1. End-to-end brush click: another **~10–20%** on top of the Phase 3 fix above.

Risk: I'd want to read the DC code in `voxel-core` to confirm the Lipschitz-1 property formally before committing. Marking this as your-call.

### B. Rust — Channel tube re-recomputes basis vectors per anchor

**File:** `voxel-ffi/src/brushes.rs:712-714` (inside channel branch).

```rust
let basis = if anchor.inward.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
let perp_a = anchor.inward.cross(basis).normalize_or_zero();
let perp_b = anchor.inward.cross(perp_a).normalize_or_zero();
```

Per-anchor two `cross + normalize` ops. Trivial absolute cost (~tens of ns per anchor × 50 anchors = ~µs total). Calling out only for symmetry — not worth changing.

### C. Rust — `aureole.rs` BFS still reallocates frontier vecs

**File:** `voxel-sleep/src/phases/aureole.rs` (commit 3a55227).

Carried over from yesterday's review. Sleep is a multi-second user-facing loader, so malloc cost is a rounding error here. Still not worth touching unless sleep wall-time becomes a focus.

### D. Rust — `squeeze_excess_fluid` walks entire chunk on density change

**File:** `voxel-fluid/src/sim/utils.rs:7-64`.

Carried over from yesterday. Modest win, requires plumbing a bbox through `FluidEvent::DensityUpdate`. Still not pulled the trigger.

---

## What I checked but found clean

- **`6539312` `has_sources` flag.** All source-placing paths set it (`AddFluid` + `is_source` branch, geological springs, `place_sources`, `pending_fluid_load` + `is_source` branch). Recompute pass in `tick_chunk`'s final fold is bounded to a single grid walk per chunk (alongside `has_fluid` and `has_lava` recomputation). `regen_sources` short-circuit at `voxel-fluid/src/sim/utils.rs:492` is correct. Good.
- **`a0ebed5` `QuenchScratch` reuse.** The 4 `HashSet`s + 2 `Vec`s are correctly cleared on entry to `detect_lava_water_quench_with_scratch`. The deprecated standalone `detect_lava_water_quench(&chunks)` still allocates fresh — that's fine, it's not on the hot path. No leak risk.
- **`b2f522a` (yesterday's `equalize_horizontal` fix).** Shipped clean. Verified the in-tree code matches what I left yesterday — no regressions introduced by the OrePaint commit.
- **`267196b` Phase 1 cross-chunk neighbor lookup.** I checked whether the OOB neighbor branch (lines 518-548) caches the neighbor chunk pointer. It does not, but the OOB branch is only hit for voxels on a chunk edge (~6/N of all voxels for chunk_size=N), and even then only for the axis on which the voxel is at the edge. For chunk_size=30 that's <20% of candidates × ⅙ of probes — small absolute cost. Not worth caching.
- **`267196b` Phase 2 Fisher-Yates over all candidates.** I considered partial shuffle (shuffle only first `target_count * k` for some k) but the min-spacing rejection downstream means we may legitimately need to walk most of the list. Not safe to truncate.
- **`267196b` worker handler `crystal_placements` recompute.** Same pattern as `BrushSphere` — necessary for visual correctness (quartz/amethyst/crystal ores feed `compute_crystals`), can't be skipped. Correct.

---

## Recommended next action

The `paint_ore_deposits` Phase 3 fix above is the only one I implemented. Item A (Phase 1 deep-interior cull) is the natural follow-up — same file, same brush, larger potential win — but I'd want to read `voxel-core/src/dual_contouring/` first to confirm the SDF Lipschitz-1 assumption before committing. If you've got that assumption documented somewhere I missed, it becomes a 3-line change with a ~10–20% extra brush-click speedup.

If the Phase 3 fix looks correct on review:
- Commit message draft: `OrePaint brush: chunk-batched Phase 3 writes`
- 1 file changed, ~95 insertions / ~30 deletions (net +65, but eliminates `write_ore_at_world` per-voxel HashMap thrash)
- All 31 brushes tests pass, no behavior change for any seed

---

*Generated by daily-commit-performance-review scheduled task.*
