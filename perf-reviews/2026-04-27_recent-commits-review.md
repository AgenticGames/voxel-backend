# Performance Review — Recent Commits

**Date:** 2026-04-27 (scheduled review)
**Scope:** Last 4 commits on voxel-backend `main`
**Verdict:** Recent commits are perf-focused and well-measured. Several missed opportunities remain; none are regressions, all are leftover wins.
**Action requested:** Review and pick which to land. Nothing pushed.

---

## Commits reviewed

| SHA | Title | Date | Net diff |
|---|---|---|---|
| 6598d63 | Building flatten rewrite (SDF) + collapse rubble pile | 2026-04-26 | +1905/-351 |
| 404e1ac | Seam gaps + mining lock contention fixes | 2026-04-20 | +132/-55 |
| f0760a9 | Streaming perf: hash-skip first-sends, batched crystal recompute, lock-free hash | 2026-04-20 | +84/-71 |
| eef7c97 | Streaming optimization: per-region mutex dedup + seam-pass hash skip | 2026-04-17 | +144/-5 |

What landed (already measured):
- 404e1ac — store_read_wait max 150.75ms → 2.02ms (-99%); seam_pass total 2202ms → 708ms (-68%); UE ProcessResults 630ms → 427ms (-32%).
- f0760a9 — wall-time cold load 8.72s → 7.24s (-17%); seam-pass total 41ms → 3ms (-93%).
- eef7c97 — initial load 17.95s → 7.59s (-58%); max frame 74.7ms → 22.1ms (-70%).

The streaming-side cleanup is in solid shape. The remaining leftover wins are concentrated in:
1. `try_process_stress_queue` (worker.rs) — fires on every mine.
2. The new `flatten_sdf` module (latest commit, not yet profiled in anger).
3. `hash_mesh` and a 30K-f32 churn point in the slow-path Generate handler.

---

## Missed opportunities

### #1 — `try_process_stress_queue` runs heavy debug logging unconditionally (worker.rs:191–591)

**Estimated savings: 20–40% of stress recalc wall-time (~2–12ms per mine op).**

Three issues, all in code that runs on every mine sphere through the 400 ms deferred timer:

a) **`dbg` closure opens/closes the log file per call** (worker.rs:206-211):
   ```rust
   let mut dbg = |msg: String| {
       if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true)
           .open(debug_path) { ... }
   };
   ```
   ~10–25 `dbg()` calls per recalc → 10–25 file-handle open/close per mine. On Windows, each open is a few hundred μs. Use a single `BufWriter<File>` opened once at the top of the function (or a `OnceLock<Mutex<File>>` shared by all workers), or gate the whole logger behind `cfg(debug_assertions)` / a `voxel.StressDebug` runtime flag.

b) **Whole-chunk voxel scan purely for one stat-line log** (worker.rs:282–304):
   ```rust
   for &key in &dirty_chunks {
       for z in 0..grid_size { for y in 0..grid_size { for x in 0..grid_size {
           // bucket stress into 6 categories
       }}}
   }
   dbg(format!("  recalc {:.1}ms — {} dirty chunks, {} solid: ...
   ```
   With ~27 dirty chunks at 31³ grid that's ~800K iterations per mine, just to print a histogram bar. Gate this whole block behind a debug toggle.

c) **Top-5 overstressed detail does extra store reads + per-voxel work** (worker.rs:319–343) — same gating fix.

The actual user-facing warning emission (lines 349-400) is real work and should stay. Only the diagnostic stats need gating.

---

### #2 — `flatten_sdf::SupportHull::cone_top_in_column` is the per-column hot loop and not vectorized (flatten_sdf.rs:68–83)

**Estimated savings: 30–60% of buttress-resolve time on cantilever-heavy placements (small absolute number; ~0.5–2 ms per flatten).**

```rust
fn cone_top_in_column(&self, wx: f32, wz: f32, search_lo: f32, search_hi: f32) -> Option<f32> {
    let mut y = search_hi;
    let step = 0.5;
    while y >= search_lo {
        for c in &self.cones {
            let d = sdf_capped_cone(p, c.base, c.tip, c.r_base, c.r_tip);
            if d < min_d { min_d = d; }
        }
        if min_d < 0.0 { return Some(y); }
        y -= step;
    }
}
```
Stepping at 0.5 voxels over a `cap_distance(8)+1 = 9` range = 18 evaluations per column, each doing a full O(cones) loop. For a 4×4 cantilever building with 12 cones, that's 216 SDF evals per apron column. Fix:

- For each cone, compute analytically the top Y where the cone surface intersects column (wx, wz) — a closed form for capped-cone-vs-vertical-line. Take the max over cones. ~20× faster than ray-marching.
- Or, at minimum, hoist `apron_radius_for(8)` and `cap_distance_for(8)` (line 180) out — they're recomputed per column.

---

### #3 — `build_support_hull` is sequential per-column raycasting (flatten_sdf.rs:86–128)

**Estimated savings: 60–80% on hull build for cantilever-heavy placements (~0.5–2 ms per flatten).**

Each cantilever column does `find_support_rays` = 16 candidate rays × march up to `cap_dist` voxels each = ~128 density samples per column. Fully data-parallel:

```rust
let cones: Vec<SupportCone> = (0..terrace_size).into_par_iter().flat_map(|dx| {
    (0..terrace_size).into_par_iter().flat_map(move |dz| {
        // per-column work returning Vec<SupportCone>
    })
}).collect();
```
Density fields are read-only here (no `&mut` to store); rayon-friendly.

---

### #4 — `hash_mesh` byte-shuffles every field through FNV-1a (worker.rs:42–63)

**Estimated savings: ~85% of hash time. ~80 ms saved on cold load (≈1–2% of total cold-load wall).**

```rust
fn hash_mesh(m: &Mesh) -> u64 {
    // FNV-1a over each x/y/z/normal/material/index, separately
}
```
Comment says ~150 μs per 2K-vert chunk. 8 workers × ~600 cold chunks × 150 μs = ~90 ms total hashing on cold load. Two cheap fixes:

- Add `#[repr(C)]` + `bytemuck::Pod` to `Vertex` and `Triangle` (already `Copy`, so this is mechanical) and bulk-hash via `seahash::hash` or `xxhash-rust::xxh3`. Drops to ~15–20 μs per chunk.
- Or keep FNV-1a but feed it 8 bytes per call via `wrapping_mul` instead of one field at a time — gets you to ~50 μs.

The `#[repr(C)]` change is the right answer because it lights up the same trick anywhere mesh data crosses an FFI/wire boundary.

---

### #5 — `density.samples.iter().map(|s| s.density).collect::<Vec<f32>>()` allocates ~120 KB per chunk per gen (worker.rs:1089)

**Estimated savings: 1–3% of cold-load wall-time (~70 MB allocator churn eliminated).**

Inside the per-chunk Generate handler:
```rust
let densities: Vec<f32> = density.samples.iter().map(|s| s.density).collect();
let _ = fluid_event_tx.send(FluidEvent::DensityUpdate { chunk, densities });
```
For a 31³ density field that's 30K f32 = 120 KB allocated, copied, sent over a crossbeam channel, dropped on the receiving thread. On a 600-chunk cold load that's ~70 MB of churn through the Windows allocator just for fluid bookkeeping. Options (cheapest first):
- Send `Arc<DensityField>` instead. The fluid thread is read-only here.
- Have the fluid thread sample density on-demand from `ChunkStore` rather than receive a snapshot.
- If you must copy, `Vec::with_capacity(density.samples.len())` + `extend(density.samples.iter().map(...))` is identical perf but more honest.

---

### #6 — `backward_dirty.contains(&key)` inside nested loop is O(n²) (worker.rs:873, 889)

**Estimated savings: bounded but real on regions with many backward-carved chunks (~0.5–3 ms on slow-path region gen).**

```rust
for cz in min_cz..=max_cz { for cy in ... { for cx in ... {
    if !backward_dirty.contains(&key) {
        backward_dirty.push(key);
    }
}}}
```
`backward_dirty: Vec<(i32,i32,i32)>`. Linear scan per insert. For a worm path that hits 50 chunks, that's 50²/2 = 1250 comparisons. Switch to:
```rust
let mut backward_dirty_set: HashSet<(i32,i32,i32)> = HashSet::new();
// or FxHashSet — see #7
```
and materialize `Vec` only when iteration order matters.

---

### #7 — Replace `std::collections::HashMap`/`HashSet` with `FxHashMap`/`FxHashSet` for hot per-call sets

**Estimated savings: 2–5% of flatten and seam-pass time (insert-heavy, small int keys).**

`std::collections::HashMap` uses SipHash with random per-run seeds (DoS-protection that we don't need). For `(i32, i32, i32)` chunk-coord keys, FxHash is ~2× faster on insert and ~3× faster on lookup.

Targets:
- `flatten_sdf::flatten_terrace_sdf::dirty_set` (per flatten, ~500–2000 inserts)
- `density_ops::WrittenCell` lookups
- `worker::try_process_stress_queue::chunk_set` (per stress recalc)
- `worker::dirty_chunks` derivations throughout

`fxhash` and `rustc-hash` are zero-dep additions.

---

### #8 — `region_set` HashSet built twice in cross-region sync (worker.rs:972, 994)

**Estimated savings: ~10 μs per slow-path region. Trivial fix worth doing for cleanliness.**

```rust
let non_region_dirty: Vec<...> = {
    let region_set: HashSet<_> = coords.iter().copied().collect();   // build #1
    ...
};
...
let region_set: HashSet<_> = coords.iter().copied().collect();        // build #2 (line 994)
for &key in all_dirty_keys.iter().filter(|k| !region_set.contains(k)) { ... }
```
Hoist once.

---

### #9 — `flatten_sdf` apron loop is sequential despite being almost-pure reads (flatten_sdf.rs:237–278)

**Estimated savings: 40–70% of flatten apron time on multi-chunk buildings (~1–4 ms per flatten).**

The (dx, dz) apron loop does:
1. `resolve_target_y` — pure read of density_fields.
2. `density_ops::write_raise/force/lower` — mutates density_fields.

Two-pass fix:
```rust
// Pass 1 (parallel): resolve target_y for every column
let column_targets: Vec<(i32, i32, Option<f32>, bool)> = (...).into_par_iter()
    .map(|(dx, dz)| (wx, wz, resolve_target_y(...), in_interior))
    .collect();

// Pass 2 (serial): apply writes
for (wx, wz, target_y, in_interior) in column_targets { ... }
```
The serial write pass is fast (just stores). Most cost is in `resolve_target_y` which calls `natural_floor_y_iso` (column scan) and `cone_top_in_column` (#2 above).

---

## Summary table

| # | Location | Estimated savings | Confidence | Effort |
|---|---|---|---|---|
| 1 | `try_process_stress_queue` debug logging gating | 20–40% of stress recalc; ~2–12 ms per mine | High | Trivial (gate behind `cfg(debug_assertions)` or runtime flag) |
| 4 | `hash_mesh` bulk-hash via repr(C)+seahash | ~80 ms cold load (~1–2%) | High | Low (add `#[repr(C)]`, swap hasher) |
| 5 | Density-vec allocation per chunk → Arc/lazy | 1–3% cold load; ~70 MB churn | Medium (depends on fluid-thread refactor cost) | Medium |
| 9 | Parallel `flatten_sdf` apron resolve | 40–70% of flatten apron; ~1–4 ms per flatten | Medium (need to bench) | Low |
| 2 | `cone_top_in_column` analytic instead of march | 30–60% of buttress resolve; ~0.5–2 ms per flatten | High | Medium (closed-form derivation) |
| 3 | Parallel `build_support_hull` raycasts | 60–80% of hull build; ~0.5–2 ms per flatten | High | Low |
| 7 | FxHashMap/Set for hot per-call sets | 2–5% flatten/seam | Medium | Low (workspace-wide find/replace) |
| 6 | `backward_dirty` Vec → HashSet | 0.5–3 ms on slow-path | High | Trivial |
| 8 | Hoist duplicate `region_set` build | ~10 μs | High | Trivial |

**Highest-ROI quick wins**: #1, #4, #6, #8 — all small mechanical changes with measurable returns.
**Bigger refactors worth scheduling**: #5 (Arc density), #9 (parallel apron resolve), #2 (analytic cone top).

---

## Notes for the next benchmark run

If you land any of #1, #4, #5, #9 — bench against `perf-baselines.md` 2026-04-25 entries. The relevant scopes to watch:

- #1 → `VoxelPR_MineResult` (currently 0.468 ms avg) and stress-recalc Rust-side dbg in `streaming_profile_*.txt`. Expect mine-tail latency improvement.
- #4 → cold-load wall-time only. Compare against the 7.24 s number from f0760a9.
- #5 → cold-load + steady-state allocator pressure. UE side won't see this; check Rust-side `region_density` + heap-profile if you have one.
- #9 → `Voxel_LevellerFlatten` (0.003 ms) is too small to see; bench instead `Voxel_BuildingSpawn` (1.5 ms avg) and `Voxel_ConveyorBatchPlace` (23 ms avg).
