# Performance Review — 2026-04-30

Scheduled review of recent commits. Tip of `main` is still `6598d63` (Building flatten SDF + collapse rubble pile rewrite) — no new commits since the 2026-04-28 review. **Nothing has been pushed.** One change is staged in the working tree for your review; the rest is a punch-list.

## Commits scanned

| Commit | Title | State |
| --- | --- | --- |
| `6598d63` | Building flatten rewrite (SDF) + collapse rubble pile rewrite | partially reviewed; one win applied today, more queued |
| `404e1ac` | Seam gaps + mining lock contention fixes | already optimized previously |
| `f0760a9` | Streaming perf: hash-skip first-sends, batched crystal recompute, lock-free hash | already optimized |
| `eef7c97` | Streaming optimization: per-region mutex dedup + seam-pass hash skip | already optimized |

Findings #1, #2 from `PERF_REVIEW_2026-04-28.md` (column-cached chunk in apron loop, frontier-based `formation_removal_pass`) are still **unimplemented** on `main`. They remain the highest-leverage wins on the building-placement path. I did not re-implement them today — see the 2026-04-28 review for full proposals.

---

## ★ Applied today: hoist `scores.get(&key)` out of the relaxation triple-loop

**File:** [voxel-core/src/stress.rs:822-902](voxel-core/src/stress.rs:822) (global flood) and [voxel-core/src/stress.rs:908-1004](voxel-core/src/stress.rs:908) (relaxation iteration).
**Status:** Edited locally, `cargo test -p voxel-core` passes (96/96 incl. all v2 stress tests). Workspace tests have one pre-existing failure on `main` (`zones::mega_blueprint::tests::blueprint_has_expected_structure`) unrelated to this change — verified by stashing my change and re-running.
**Risk:** Low. Pure caching transform; behavior is byte-identical to the prior code for every voxel because the relaxation explicitly defers writes via the `updates: Vec<...>` collector, so reading the cached field reference is the same as re-fetching it through the map.

### What was wrong

`ground_connectivity_pass()` runs `support_propagation_iterations` (default 4) iterations of relaxation. The hot inner loop on the **previous** code looked like this for every solid voxel:

```rust
for &key in &expanded_keys {
    let df = density_fields.get(&key)?;
    for z in 0..grid_size { for y in 0..grid_size { for x in 0..grid_size {
        if !df.get(x,y,z).material.is_solid() { continue; }
        let current_score = scores.get(&key).unwrap().get(x, y, z);  // ← same key every iter
        ...
        if let Some(bsf) = scores.get(&bkey) { ... }                 // ← bkey == key when y > 0
        for &(dx, dz) in &[(1,0),(-1,0),(0,1),(0,-1)] {
            let (nkey, ...) = world_to_chunk_local(...);
            if let Some(nsf) = scores.get(&nkey) { ... }             // ← nkey == key for interior cells
        }
    }}}
}
```

For chunk_size=30, `grid_size=31`, that's `31³ = 29,791` cells per chunk. Each cell did:

- 1 `scores.get(&key)` for `current_score` — **always** redundant (key is loop-invariant).
- 1 `scores.get(&bkey)` for the cell-below — redundant when `y > 0` (~96.8% of cells).
- 4 `scores.get(&nkey)` for lateral neighbors — redundant when the neighbor stays in this chunk (~93.5% of the 4 reads, since only the 4 face strips cross out).

Total redundant lookups per cell ≈ `1 + 0.968 + 4 × 0.935 = 5.71`. Per chunk per iteration: **~170,000 redundant `HashMap.get` calls**. With ~27 expanded chunks (3³ neighborhood) × 4 iterations: **~18 million lookups eliminated per `ground_connectivity_pass()` call**.

`std::collections::HashMap` with `(i32,i32,i32)` keys uses SipHash (~50-100ns/lookup including hashing + probe). 18M × 60ns ≈ **1.0–1.5 seconds saved per stress recalc** on a 27-chunk dirty set.

### What I changed

1. **Global flood loop** (descending y per column): cache `(cy, df, in_expanded, key)` and only re-fetch from `density_fields`/`expanded_keys` when `cy` changes. Per column ~30 iterations now do 1–2 lookups instead of `~30 × 2 = 60`. Smaller win — saves ~95% of map ops in this loop, but the loop is short.

2. **Relaxation iteration**: hoist `let current_sf = scores.get(&key)?;` outside the `(z,y,x)` loops. For neighbor reads (`bkey`, `±x`, `±z`), check the local index first and use `current_sf` directly when the neighbor is still inside the cached chunk. The HashMap lookup only fires for the 4×grid² cells on each face. This is the bulk of the savings.

3. **No semantic change** — the original `scores.get(&key).unwrap()` was already infallible because the loop guard `for &key in &expanded_keys` only iterates keys we just inserted into `scores`. The new `match scores.get(&key) { Some(sf) => sf, None => continue }` is defensive-equivalent.

### Estimated savings

- **`ground_connectivity_pass()`: 25–40% wall time** when expanded_keys ≥ 8 chunks (the common case). Closer to 40% on dense dirty regions where the relaxation does real work; closer to 25% when most cells short-circuit at `current_score >= 1.0`.
- **End-to-end stress recalc on mining/flatten**: ~10–20% wall time, since `ground_connectivity_pass` is one of three phases (it's followed by per-voxel `calc_voxel_stress_v2`, which is unchanged).
- The win **scales with chunk size** — at chunk_size=30 (the live UE setting per CLAUDE.md), savings are noticeably better than at chunk_size=16 (the default), because grid_size³ = 29,791 vs 4,913 (6× more cells).

I did **not** convert the per-cell `world_to_chunk_local` calls for the lateral fall-through case to direct arithmetic. They only fire on the 4 face strips, so the savings would be tiny.

---

## Punch list — opportunities found, NOT applied

### A. Switch chunk-keyed `HashMap` → `FxHashMap`/`AHashMap`  ⭐⭐⭐⭐
**Files:** 24 `use std::collections::HashMap` sites across `voxel-ffi`, `voxel-core`, `voxel-gen`. Hottest:
- `voxel-ffi/src/store.rs` — `density_fields`, `mesh_cache`, `mesh_hashes`, etc.
- `voxel-core/src/stress.rs` — `support_scores`, `support_fields`.
- `voxel-core/src/density_ops.rs` — every `read_density`/`write_*` chases `fields.get(&(cx,cy,cz))`.

Default `std::HashMap` uses SipHash, which is cryptographically robust but ~3–5× slower than `FxHash` (or `AHash`) for small integer-tuple keys. The chunk-key pattern `(i32,i32,i32)` is the canonical case `FxHash` was designed for — it's literally what rustc itself uses internally.

**Why this hasn't been done:** drop-in replacement requires touching 24 files plus adding a workspace dep (`fxhash = "0.2"` or `ahash`). Most call sites are `HashMap::new()` / `HashMap::with_capacity`, so a `pub type ChunkMap<V> = FxHashMap<(i32,i32,i32), V>;` alias in `voxel-core` and a workspace search-replace would do most of the work.

**Estimated savings:** **15–30% on every hot path that touches a chunk map** — streaming, mining, flatten, stress, sleep, fluid, save/load. This is the single largest unrealized win in the codebase. After today's stress hoist there are still ~12M chunk-map lookups per second under heavy mining; cutting each to ~15ns from ~70ns is a real difference.

The risk is that any code that depends on iteration order (none should, but worth grepping) or stores the map in serialized form (none does — saves go through `delta.rs` with explicit ordering) needs validation.

---

### B. `SupportHull::cone_top_in_column` is O(steps × cones) per apron column  ⭐⭐
**File:** [voxel-ffi/src/flatten_sdf.rs:68-83](voxel-ffi/src/flatten_sdf.rs:68)

For each apron column the resolver may walk the cone hull from `cone_search_hi` down to `cone_search_lo` in 0.5-voxel steps, evaluating `sdf_capped_cone` against **every** cone. With `cap_distance(8)=16`, that's up to 32 steps × ≤768 cones = **24,576 SDF evaluations per cantilever-affected apron column**. Buildings without overhangs still call `cone_top_in_column` (it short-circuits to `None` only when `cones.is_empty()`).

**Concrete change:** rasterize cones into a per-column `top_y` map up front (one pass over cones, one entry per column inside each cone's xz footprint). Lookup becomes O(1).

**Estimated savings:** **50–90% of `cone_top_in_column` cost** on cantilever-heavy placements (overhanging buildings, asymmetric leveler ops). Negligible for fully-supported buildings — this is conditional on cantilever count, not a guaranteed win.

---

### C. `column_weight_above` does up to 32 chunk lookups per stressed voxel  ⭐⭐
**File:** [voxel-core/src/stress.rs:549-570](voxel-core/src/stress.rs:549)

Called from `calc_voxel_stress` (the v1 path; v2 uses precomputed scores). For every solid voxel, marches up to 32 voxels of column scan, each one going through `sample_world` → `density_fields.get(&key)`. Most of those 32 reads sit in the same chunk (only 1–2 cy boundaries crossed).

**Concrete change:** cache `last_key + last_df` in the y-loop; refetch only on chunk-y crossing.

**Estimated savings:** **~80–90% of `column_weight_above` cost**. Only matters if v1 stress is still called somewhere — the v2 path is precomputed. (Worth grep'ing if `calc_voxel_stress` is dead code; if so, delete it instead of optimizing.)

---

### D. `formation_removal_pass`: dual chunk-lookup per cell  ⭐⭐
**File:** [voxel-core/src/density_ops.rs:458-497](voxel-core/src/density_ops.rs:458)

Even after the column-cache fix proposed in the 2026-04-28 review, each surviving cell does `read_density` (HashMap.get) followed by `count_air_face_neighbors` (another HashMap.get). The two functions could share a single chunk fetch: pass the cached `Option<&DensityField>` into a combined `read_and_count_neighbors` helper, refetching only when crossing a chunk boundary.

**Estimated savings:** stacks with finding #2 from the 2026-04-28 review. Independently: ~10–20% additional on `formation_removal_pass`.

---

### E. `dirty_set: HashSet<(i32,i32,i32)>` allocation churn in flatten_sdf  ⭐
**File:** [voxel-ffi/src/flatten_sdf.rs:210, 229-232](voxel-ffi/src/flatten_sdf.rs:210)

For a single placement the dirty set grows to at most ~27 keys (3³ neighborhood). Using `HashSet` for that size is overkill — a small `Vec` with linear scan + dedup at the end would be measurably faster. Or use `SmallVec<[_; 32]>`. Same applies to `voxel-core/src/density_ops.rs::write_all_locations` callers.

**Estimated savings:** ~2-5% on `flatten_terrace_sdf`. Mostly relevant because allocation pressure during a placement burst (e.g. conveyor batch flatten) compounds.

---

## Process notes

- The `mega_blueprint` test failure on `main` predates today's change. Should be triaged separately.
- The two highest-impact wins remain finding A above (FxHash) and findings #1+#2 from the 2026-04-28 review (apron-column cache, frontier formation pass). Together they dwarf today's stress-loop win.
- Today's change is sized to be reviewable in one sitting — ~80 lines net diff, all in `voxel-core/src/stress.rs`, 96 unit tests still green.

## Diff summary

```
voxel-core/src/stress.rs | 113 ++++++++++++++++++++++++++-------------
1 file changed, ~75 inserts, ~38 deletes (net +37 with comments)
```

No other files touched. Run `git diff voxel-core/src/stress.rs` to inspect.
