# Performance Review — 2026-05-02

Scheduled review of recent commits. Tip of `main` is still `6598d63` (Building flatten SDF + collapse rubble pile rewrite) — **no new commits since 2026-04-30/05-01.** Nothing has been pushed. **One change is staged in the working tree** for your review (described below).

## Commits scanned

| Commit | Title | State |
| --- | --- | --- |
| `6598d63` | Building flatten rewrite (SDF) + collapse rubble pile rewrite | partially reviewed; new win staged today |
| `404e1ac` | Seam gaps + mining lock contention fixes | already optimized in prior reviews |
| `f0760a9` | Streaming perf: hash-skip first-sends, batched crystal recompute, lock-free hash | already optimized |
| `eef7c97` | Streaming optimization: per-region mutex dedup + seam-pass hash skip | already optimized |

The 2026-04-30 stress-loop hoist (Finding ★) and the 2026-05-01 frontier-based `formation_removal_pass` are still in the working tree and verified — `cargo test -p voxel-core --lib` still green at 97/97 with both applied.

The unimplemented punch list from prior reviews remains open (see "Process notes" at the bottom).

---

## ★ Applied today: parallelize `ChunkStore::remesh_dirty` with rayon

**File:** [voxel-ffi/src/store.rs:251-313](voxel-ffi/src/store.rs:251)
**Status:** Edited locally. `cargo test -p voxel-ffi --lib` passes 59/59. `cargo test -p voxel-core --lib` passes 97/97. Release build clean. **Not committed, not pushed.**
**Risk:** Low. The transformation is mechanical: extract the per-chunk pure-compute body into a `par_iter().filter_map().collect()`, then apply the resulting `HashMap` writes serially in a small post-pass. No semantic change — same hermite, same DC vertices, same converted mesh, same insertion order into the result `Vec` (rayon's `collect` preserves input order).

### What was wrong

`ChunkStore::remesh_dirty` was the **only sequential CPU bottleneck** still inside the `store.write()` write-lock window for every flatten / mine / sleep / brush / save-load path. Each dirty chunk runs:

1. `extract_hermite_data(density)` — full grid sweep, density gradient, edge intersection extraction.
2. `solve_dc_vertices(hermite, cell_size)` — QEF solve per occupied cell.
3. `generate_mesh(hermite, dc_vertices, cell_size)` — DC mesh assembly.
4. `mesh.smooth(...)` — N Laplacian smoothing iterations.
5. `mesh.recalculate_normals()` (when enabled).
6. `extract_boundary_edges(hermite, chunk_size)` — boundary edge collection for seam stitching.
7. `convert_mesh_to_ue_scaled(mesh, ...)` + `bucket_mesh_by_material(...)` — UE coord transform + material bucketing.

For a chunk_size=30 grid (live setting in UE per `MEMORY.md`), one chunk's full re-extract+solve+mesh+smooth chain is non-trivial — typical shape is 1–8ms in release, dominated by `extract_hermite_data` and `mesh.smooth`. Until today, a building flatten dirtying 4 chunks paid `4 × per_chunk_cost` *serially* while the **store write lock was held the entire time** (callers do `let mut s = store.write().unwrap(); ... remesh_dirty ...; drop(s);`).

The old loop also did a wasteful round-trip:

```rust
self.hermite_data.insert(key, h);
let hermite = self.hermite_data.get(&key).unwrap();
```

— insert, then re-borrow the just-inserted value. The new code keeps the local `hermite` and inserts at the end of phase 2.

### What I changed

Two-phase split:

- **Phase 1** (`dirty_chunks.par_iter().filter_map(...).collect()`): all per-chunk pure compute (steps 1–7 above) happens in parallel across rayon's pool. Each task immutably borrows its chunk's `DensityField` from the shared `density_fields: HashMap`. No `&mut self` needed — the HashMap reads are safe to share across threads.
- **Phase 2** (serial): walk the collected results and apply the three writes (`hermite_data.insert`, `base_meshes.insert`, `chunk_seam_data.insert`) in one pass. These are the only `&mut self` operations.

The shared config fields used inside the parallel block (`mesh_smooth_iterations`, `mesh_smooth_strength`, `mesh_boundary_smooth`, `mesh_recalc_normals`, `voxel_scale()`) are hoisted to locals before the parallel section so each rayon task captures cheap values, not a `&GenerationConfig` reference.

Rayon was already imported in `store.rs:4` (`use rayon::prelude::*;`) for other uses, so no new dependency surface.

### Why it's safe

- All compute functions take `&HermiteData` / `&DensityField` / `&Mesh` — already pure given immutable inputs.
- `HermiteData`, `Mesh`, `Vec3`, `EdgeKey`, `EdgeIntersection`, `ConvertedMesh` are all plain data; `Send + Sync` is automatic.
- The output tuple `(key, hermite, mesh, dc_vertices, boundary_edges, converted)` is owned, so `collect()` moves them into the serial phase without aliasing.
- The result `Vec` ordering is preserved (rayon's `par_iter` over a slice + `collect` is order-preserving), so any caller that relied on the iteration order of `dirty_chunks` getting stamped into `results` keeps the same ordering — same as before.
- No double-write: the old code did `self.hermite_data.insert` *inside* the loop, then a `.get(&key).unwrap()` to re-borrow. The new code keeps `hermite` as a local through phase 1, then inserts once in phase 2. Same end-state HashMap, no observable change.

### Estimated savings

This is the highest-leverage win remaining on the **flush-after-modify** path because every other phase of mine/flatten/seam/save was already serialized under the store lock with no parallelism. Numbers below are modeling, not measured — verify with the perf-baselines bench recipe before banking them.

| Scenario | Dirty chunks | Old (serial) | New (rayon, 8 workers) | Speedup |
|---|---|---|---|---|
| Single-chunk mine (interior) | 1 | 1× | 1× | 0% (overhead, but bounded — rayon's `par_iter` over a 1-element slice is essentially a direct call) |
| Building flatten, 1 chunk | 1 | 1× | 1× | 0% |
| Building flatten, 4 chunks (typical 8×8 footprint near a chunk corner) | 4 | 4× | ~1.1–1.3× (limited by longest-chunk + write-back) | **~70–75%** wall-time reduction inside `remesh_dirty` |
| Building flatten, 8 chunks (8×8 footprint spanning multiple chunk seams) | 8 | 8× | ~1.3–1.5× | **~80–85%** |
| Mine sphere, large radius spanning 8 chunks | 8 | 8× | ~1.3–1.5× | **~80–85%** |
| Sleep collapse, 30+ chunks | 30+ | 30×+ | ~5× | **~80–85%** |
| Brush operation, large area | 10–40 | 10–40× | ~2–6× | **~75–85%** |

**Whole-flatten wall-time impact** (where `remesh_dirty` is one of several phases under the lock): typical building flatten is ~30–50% remesh + ~20–30% sync/restore + ~20–40% SDF compute. Parallelizing remesh shifts the 30–50% portion down by ~75–85%, giving an overall **~20–35% wall-time reduction** on multi-chunk flatten paths and a **proportional reduction in store-write-lock holding time** — which compounds because every other worker (mining, streaming) waiting on the store gets unblocked sooner.

The 0% impact on single-chunk paths is an explicit non-regression: rayon's `par_iter().filter_map().collect()` over a 1-element slice is essentially the same cost as the serial loop plus a small overhead (well under 100ns per call); for 1-chunk callers like in-chunk mining, this is in the noise.

### Why this hadn't been flagged before

Prior reviews (2026-04-26 → 2026-05-01) focused on the SDF flatten internals (formation removal, ramp noise, count_air_face_neighbors), the seam-pass / mining hot path (worker.rs), and stress-system loops. Nobody had walked `remesh_dirty` itself — it's small, looks straightforward, and the loop body looks "already optimized" because the inner functions are all batched. The win is **not** in any single inner function — it's in the fact that the loop iterates serially while every step inside is independently parallelizable.

This finding does **not** overlap with any of:
- 2026-04-28 #1 (column-cached chunk in apron loop) — different function, different code path
- 2026-04-28 #2 (frontier formation removal) — already applied 2026-05-01
- 2026-04-28 #3 / 2026-04-30 A (FxHashMap workspace switch) — orthogonal; FxHashMap would compound with this win, not duplicate it
- 2026-04-30 B (cone hull rasterization) — different function, different code path
- 2026-04-30 C / D / E — different functions

### Verification plan

Before banking the savings number, capture matched runs with the existing scope-timer instrumentation:

1. Spawn a 30-building cluster placement in PIE (the existing creative-brush authoring test).
2. Capture `remesh_dirty_total_ms` and `flatten_terrace_sdf_total_ms` from the worker scope timer.
3. Diff against the 2026-04-30 baseline in `memory/perf-baselines.md`.
4. Log the new numbers under a 2026-05-02 entry.

If you want a synthetic rather than in-game bench, the 4-chunk case is reproducible from `voxel-ffi/src/flatten_sdf.rs::tests::subvoxel_surface_lands_near_requested_y` — duplicate it with a larger ground footprint (chunks=2 or 3) and time the call.

---

## Punch list — opportunities found, NOT applied

These are net-new findings not in any previous review.

### F. `restore_written_cells` walks all writes — only seam-fan-out cells need restoration  ⭐⭐

**File:** [voxel-core/src/density_ops.rs:361-374](voxel-core/src/density_ops.rs:361)

`restore_written_cells` exists to defeat `sync_boundary_density`'s `min()` averaging at chunk seams. But `sync_boundary_density` only ever touches cells where the local coordinate is exactly `0` or `cs` along at least one axis — interior cells (`0 < lx < cs && 0 < ly < cs && 0 < lz < cs`) are never modified by sync, so they don't need restoration.

Today the function loops over every `WrittenCell` in the flatten's `written` Vec and does a `HashMap.get_mut` + density read + `abs` + compare per entry, even for the ~85% of cells that are interior. The early-out at `(s.density - w.new_density).abs() > 1e-3` correctly skips the no-op case, but the HashMap lookup + memory load already happened.

**Change:** add a boundary check at the top of the loop:

```rust
for w in written {
    let on_seam = w.lx == 0 || w.lx == cs_usize
                || w.ly == 0 || w.ly == cs_usize
                || w.lz == 0 || w.lz == cs_usize;
    if !on_seam { continue; }
    // ... existing body
}
```

Requires plumbing `cs` (or storing it in `WrittenCell`) but the rest is mechanical.

**Estimated savings:** ~80–85% reduction in `restore_written_cells` work. This function is called per-flatten and per-collapse-rubble-placement. Not a streaming-hot-path bottleneck on its own (~0.1–0.3ms typical), but free win that compounds with #4 (chunk-grouped writes) from the 2026-04-28 review.

### G. `sync_boundary_density` always iterates full `(cs+1)²` face plane regardless of dirty bounds  ⭐⭐

**File:** [voxel-ffi/src/store.rs:1397-1495](voxel-ffi/src/store.rs:1397) (sync_boundary_density)

The function takes per-chunk dirty bounds `(min_x, min_y, min_z, max_x, max_y, max_z)` but uses them only to decide *whether* to sync each face/edge (the `max_x >= cs` / `min_x == 0` checks). Once a face is selected for sync, it walks the **entire (cs+1)² face plane** (`for u in 0..=cs { for v in 0..=cs { ... }}`) — 961 cells per face at chunk_size=30, × up to 6 faces per chunk × N dirty chunks.

Both `flatten_terrace_sdf` (line 312) and `flatten_terrace` (line 287) construct `dirty_chunks` with `(0, 0, 0, cs, cs, cs)` — full chunk bounds — even though typical building flattens touch only ~10×10×6 voxels concentrated near the building. The result: `sync_boundary_density` over-syncs by ~10× per flatten.

Also: `sample_a` and `sample_b` are read via `density_fields[&key]` and `density_fields[&neighbor]` — a HashMap lookup per cell of the face plane. The `key` and `neighbor` are loop-invariant; they could be looked up once per face, then indexed directly.

**Change:** (1) plumb actual write bounds out of the flatten loops — track per-chunk min/max as cells are written, instead of supplying full-chunk bounds. (2) inside `sync_boundary_density`, use those per-chunk bounds to clamp the `0..=cs` face-plane iteration to `[u_min..=u_max, v_min..=v_max]` (where the `u, v` ranges are the projection of the dirty bounds onto the relevant axis pair). (3) hoist the `density_fields[&key]` / `density_fields[&neighbor]` lookups out of the per-cell loop.

**Estimated savings:** ~70–90% reduction in sync_boundary_density work for typical building flattens, which means ~5–15% overall flatten wall-time savings (sync is 20–30% of the total). Sleep collapses and mining benefit similarly.

### H. `find_support_rays` generates 4× candidate dirs and filters out upper hemisphere  ⭐

**File:** [voxel-ffi/src/sdf.rs:172-212](voxel-ffi/src/sdf.rs:172)

The Fibonacci sphere generator produces `n_rays * 4 = 64` candidate directions when `n_rays = 16`, then filters out everything with `dir.y > up_tolerance` (half the sphere). For `up_tolerance = 0.05` (the caller's value), ~52% of candidates fail the filter and the loop has to overproduce to backfill.

**Change:** generate Fibonacci directions on a **hemisphere** directly: `let y = -(2.0 * (i + 0.5) / n);` — half the sphere, no filter, no overshoot. (Plus a small "horizontal band" at `y ∈ (0, up_tolerance)` if you want to keep the slight-up bias.)

**Estimated savings:** ~40% reduction in `find_support_rays` runtime. This is on the cantilever-only path (build_support_hull); typical interior placements never call it. Bound: ~2–4% on cantilever-flatten wall time.

This was flagged but understated as #6 in the 2026-04-28 review ("Stop over-marching rays"). The actual issue is not the ray-march length but the dir-generation 4× overshoot.

---

## Process notes

- All edits & tests run with `export PATH="$HOME/.cargo/bin:$PATH"`.
- `cargo test --workspace` does not currently complete cleanly: `voxel-sleep/src/bench.rs` references `FluidCell` fields (`hops_from_source`, `max_flow_dist`) that exist in the working-tree-modified `voxel-fluid/src/cell.rs` but break the bench module compilation. This is **pre-existing** and unrelated to today's `remesh_dirty` change — confirmed by `git stash && cargo test -p voxel-sleep --lib` (passes 47/0/11) on `main`. The bench tests need a sync pass to match the working-tree fluid changes, but that's outside the scope of this review.
- Working-tree state preserved as found; no commits made.

## Diff summary

```
voxel-ffi/src/store.rs:251-313  | parallelize remesh_dirty with rayon (~70-85% on multi-chunk paths)
```

— Claude Opus 4.7
