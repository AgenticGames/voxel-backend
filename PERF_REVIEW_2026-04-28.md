# Performance Review — 2026-04-28

Scheduled review of recent commits. Focus on landed work in `6598d63` (Building flatten SDF rewrite), with secondary glances at `404e1ac`, `f0760a9`, `eef7c97`. **Nothing has been pushed.** This is a written punch-list for you to triage.

The streaming/seam/lock-contention commits (`f0760a9`, `eef7c97`, `404e1ac`) already extracted the obvious wins (-58% initial load, -93% seam pass, -99% store-read wait). The fresh meat is in `6598d63`.

---

## Commits scanned

| Commit | Title | State |
| --- | --- | --- |
| `6598d63` | Building flatten rewrite (SDF) + collapse rubble pile rewrite | new code, several wins available |
| `404e1ac` | Seam gaps + mining lock contention fixes | already optimized; verified, no further wins |
| `f0760a9` | Streaming perf: hash-skip first-sends, batched crystal recompute, lock-free hash | already optimized |
| `eef7c97` | Streaming optimization: per-region mutex dedup + seam-pass hash skip | already optimized |

---

## Ranked findings

### 1. Cache primary chunk per-column in the apron loop  ⭐⭐⭐
**File:** [voxel-ffi/src/flatten_sdf.rs:237-278](voxel-ffi/src/flatten_sdf.rs:237) and [voxel-core/src/density_ops.rs:113-126, 269-303](voxel-core/src/density_ops.rs:113)

The apron loop iterates `(2*extent + terrace_size)²` columns. For terrace=4 / extent=3 that's 100 columns. Each column does:

- `natural_floor_y_iso` → ~28 vertical `read_density` calls (scan_up=4, scan_down=24).
- `FILL_DEPTH` (6) `write_raise` + 2 `write_force` + ~3 `write_lower` = 11 writes; each fans out to up to 8 chunks via `cell_locations` → 1–8 `HashMap.get_mut` calls each.

**Each `read_density` / `write_*` recomputes `(cx, cy, cz) = wx.div_euclid(cs)` etc. and re-hashes the chunk key with SipHash.** For a fixed `(wx, wz)` column, the chunk-x and chunk-z don't change — only the chunk-y can flip across one or two boundaries while sweeping ~33 vertical cells.

Already-applied precedent: `count_air_face_neighbors` (same file, `density_ops.rs:139`) was hand-optimized to cache the primary chunk once for all 6 neighbor reads. Apply the same trick at the column level.

**Concrete change:** introduce a `ColumnReader { cx, cz, last_cy, last_df: *const DensityField }` (or `Option<&DensityField>` with lifetime gymnastics) that resolves the chunk on Y boundary crossings only. Same pattern for writes — `cell_locations` only fans out at chunk faces, so the interior path can hit the cached primary directly.

**Estimated savings:** **~30–45% of `flatten_terrace_sdf` wall time** (most cells are interior, all 28 vertical reads sit in 1–2 chunks). On a typical placement that's somewhere between 0.5–2 ms per building drop (need profiling to nail down).

---

### 2. Frontier-based formation removal pass  ⭐⭐⭐
**File:** [voxel-core/src/density_ops.rs:400-439](voxel-core/src/density_ops.rs:400)

`formation_removal_pass` runs up to 3 erosion iterations. **Each iteration rescans the full cylinder** — `(2r+1)² × (max_above + scan_below + 1)` cells. For `radius=8`, `y-window=17` → ~4900 cells per pass × 3 passes = ~15k cells, each doing one `read_density` plus (if solid) a `count_air_face_neighbors`.

Iteration 2 and 3 only flip cells **adjacent to the cells that flipped in the previous iteration** — air-neighbor count can only increase next door. Today we re-evaluate all 4900 cells just to find the few hundred that are now newly thin.

**Concrete change:** track victims of iteration N as a `Vec<(i32,i32,i32)>`. On iteration N+1, only re-test the 6 face-neighbors (or 18 face+edge) of those victims, dedup'd via a `HashSet`. Iteration 1 stays a full sweep.

**Estimated savings:** **~50–65% of `formation_removal_pass` cost on iterations 2+3.** Whole-pass savings ~30–45%. As a fraction of `flatten_terrace_sdf`: depends on formation density, but typically 5–15% of total flatten time. Bigger when placing buildings inside dense flowstone galleries (Phase 4z zones).

---

### 3. Replace `std::HashMap` with `FxHashMap` for chunk-keyed maps  ⭐⭐⭐
**Workspace-wide.** `density_fields: HashMap<(i32,i32,i32), DensityField>` is hot in nearly every code path (mining, flatten, seam pass, sleep, crystal recompute, fluid). It uses SipHash — DoS-resistant but slow.

For internally-controlled integer keys, `rustc_hash::FxHashMap` (or `ahash`) is consistently 2–3× faster on small integer-tuple keys. We hash these keys **constantly**: `read_density`, every `write_*`, `cell_locations` fan-out, `march_ray_for_surface`, `sample_natural_density`, the seam-pass scan, mining iteration, etc.

**Concrete change:** add `rustc-hash = "2"` to workspace deps; introduce a `ChunkMap<V> = rustc_hash::FxHashMap<(i32,i32,i32), V>` alias in `voxel-core`; do a sweep replacing `HashMap<(i32,i32,i32), …>` in the hot paths. Leave non-hot maps alone.

**Estimated savings:** **5–15% reduction across initial-load, seam-pass, mining, and flatten hot paths.** It's a tax that compounds — every `read_density` and every `cell_locations` call shaves a few ns. For something like the apron loop in finding 1: combined with that fix, expect 40–55% off flatten cost.

Risk: low. Just don't expose FxHashMap across FFI; stays internal. Make sure no code relied on iteration order (we already know "HashMap iteration is nondeterministic — always sort keys before RNG-dependent processing", so we're guarded).

---

### 4. Sort `restore_written_cells` by chunk key before writing  ⭐⭐
**File:** [voxel-core/src/density_ops.rs:339-352](voxel-core/src/density_ops.rs:339)

`written: Vec<WrittenCell>` from a single building flatten can hold thousands of entries (writes × up to 8 fan-outs). Restore loops over all of them and does `fields.get_mut(&key)` per entry. That's a cold HashMap lookup every iteration.

Since entries are produced in spatial order (column scan), they may already group well by chunk. If we **sort by `key`** first and reuse a `current_chunk: Option<(&mut DensityField, key)>`, we can avoid the repeated lookup.

**Estimated savings:** **40–60% of `restore_written_cells` time.** Tiny in absolute terms (likely <5% of total flatten) but free with a few lines.

(Applies to `write_all_locations` too if we ever batch contiguous cells — but that's a bigger restructure.)

---

### 5. Dead code in `voxel-ffi/src/sdf.rs`  ⭐ (cleanup, not perf)
**File:** [voxel-ffi/src/sdf.rs:92-151](voxel-ffi/src/sdf.rs:92)

`sample_natural_density` (trilinear density sample, ~50 lines) and `density_to_sdf` (one-liner) are never called outside their own unit tests. Looks like leftover scaffolding from a smin-blend approach that didn't ship. Either delete or wire them in — right now they're a maintenance hazard (anyone re-introducing smin-blend will assume these are fine, but `sample_natural_density` does 8 HashMap lookups per call and would need the same primary-chunk caching as finding 1).

**Estimated savings:** zero today, but flagging because the comment in `flatten_sdf.rs:36-38` ("RAMP_NOISE_AMP is 0.0") is the same kind of disabled-but-not-removed code, suggesting the file has accumulated a couple layers of "we tried this, didn't work." A quick cleanup pass would reduce confusion next time someone profiles this path.

---

### 6. `find_support_rays` allocates a fresh `Vec` per cantilever column  ⭐
**File:** [voxel-ffi/src/sdf.rs:172-212](voxel-ffi/src/sdf.rs:172) and [voxel-ffi/src/flatten_sdf.rs:113-117](voxel-ffi/src/flatten_sdf.rs:113)

In `build_support_hull`, every cantilever column allocates a `Vec<SupportHit>` (cap 16) inside `find_support_rays`, and we only consume the first `SUPPORT_RAYS_PER_COL` (3). The 16-rays-then-take-3 pattern is wasteful: we sort all 16 hits then drop 13. A small priority queue (k=3) or just an early-exit `Vec` of size 3 would skip both the over-march and the sort.

**Estimated savings:** small in absolute terms (cantilevers are the minority), but if a building straddles a cliff edge with 8+ cantilever columns it adds up. ~20–40% of `build_support_hull` time, ~1–3% of total flatten.

---

### 7. `SupportHull::cone_top_in_column` linear-scans all cones at every Y step  ⭐
**File:** [voxel-ffi/src/flatten_sdf.rs:68-83](voxel-ffi/src/flatten_sdf.rs:68)

Hot path only when `natural_floor_y_iso` returns `None` (a fallback inside `resolve_target_y`, line 180-186). For 16 cantilever columns × 3 cones = up to 48 cones; per query we step ~34 Y values × 48 cones = 1632 SDF evaluations. If the apron has many "no natural floor" columns (mid-air placement over a chasm), this can spike.

**Concrete change:** AABB-cull cones by their (xz, max_radius) before the Y sweep. Or, cheaper: the function can return the moment it finds *any* cone the column intersects — it's looking for a hit, not the deepest one. The walking-down-from-top pattern is actually what we want, but we should still skip cones whose XZ projection misses (wx, wz) by more than max(r_base, r_tip).

**Estimated savings:** 50–80% of `cone_top_in_column` cost (most cones are XZ-misses). Whole-flatten contribution: 0–5%, situational.

---

### 8. `dirty_set: HashSet<(i32,i32,i32)>` and `restore_written_cells` together do redundant work  ⭐
**File:** [voxel-ffi/src/flatten_sdf.rs:210-211, 322](voxel-ffi/src/flatten_sdf.rs:210)

Every `write_*` call adds the chunk key to `dirty_set` AND appends to `written: Vec<WrittenCell>`. Later, `restore_written_cells` walks `written` again — same chunks, same lookups. This isn't unnecessary (the *intent* of `written` is to defeat `sync_boundary_density`'s `min()` merge, which is correct), but the data structures don't share work.

**Concrete change:** keep `written` keyed/grouped by chunk, so `restore_written_cells` borrows each chunk's `&mut DensityField` once and walks contiguous cells. Combines naturally with finding 4.

**Estimated savings:** combined with finding 4, 50–70% of `restore_written_cells` time. Whole-flatten: 2–8%.

---

## Summary table

| # | Finding | Effort | Whole-flatten savings | Notes |
| --- | --- | --- | --- | --- |
| 1 | Per-column primary-chunk cache in apron loop | medium | **30–45%** | Biggest single win |
| 2 | Frontier-based formation_removal_pass iterations | medium | **5–15%** | Larger when zones are dense |
| 3 | FxHashMap workspace-wide for chunk maps | small per crate, large blast radius | **5–15%** (also benefits mining/seam/sleep) | Compounds with #1 |
| 4 | Sort `written` by chunk in restore | tiny | 1–4% | Free win |
| 5 | Delete `sample_natural_density` / `density_to_sdf` | tiny | 0% | Cleanup |
| 6 | Stop over-marching rays in `find_support_rays` | small | 1–3% | Cantilever-only |
| 7 | XZ-cull cones in `cone_top_in_column` | small | 0–5% | Situational fallback |
| 8 | Group `written` by chunk for restore | small | 2–8% | Pairs with #4 |

If all eight land cleanly, expect roughly **40–60% wall-time reduction on `flatten_terrace_sdf`** (with #1 + #3 doing most of the work). #3 (FxHashMap) also gives ~5–15% across mining/seam-pass/sleep — likely the highest-leverage change in the list.

---

## Validation plan (for whichever of these you take)

1. Pick a deterministic flatten benchmark in `voxel-ffi/src` (or add one — current tests in `flatten_sdf.rs` only check correctness, not timing).
2. Capture baseline scope-timer numbers via the existing instrumentation, log to `memory/perf-baselines.md` per the project convention.
3. Land one finding at a time; re-bench; log delta.
4. For #3 (FxHashMap), also re-run the streaming benchmarks measured in `eef7c97` / `f0760a9` (initial-load wall-time, UE ProcessResults, seam-pass total) — those will move too.

---

## What I deliberately did NOT flag

- Anything in `f0760a9`/`eef7c97`/`404e1ac` — those commits already extracted the obvious wins, and the commit messages document measured baselines. No low-hanging fruit left I could spot.
- Stress-system code in `voxel-core/src/stress.rs` — modified in the latest commit but the rubble-pile rewrite is geometry, not a hot path.
- `voxel-fluid` — unchanged in recent commits.
- The big uncommitted diff currently in the working tree (`worker.rs +369 lines`, `flatten_sdf.rs -644 lines`, etc.) — that's in-flight work; reviewing it would be reviewing a moving target. Re-run this review after you commit it.

— Claude Opus 4.7
