# Performance Review — 2026-05-08 (scheduled)

Scope: latest two commits on the local branch.

| Commit | Date | Summary |
|---|---|---|
| `003fa21` | 2026-05-06 | In-flight work + session fixes (worker.rs empty-mesh, blank_canvas, Shield/Drapery SDF, bounded fluids cross-crate) |
| `6598d63` | 2026-04-26 | Building flatten rewrite (SDF) + collapse rubble pile rewrite |

**Not pushed.** Findings ranked by impact-per-effort. Estimated `%` savings are of the OP itself, not of total frame time.

---

## Finding 1 — Pile preview re-extracts the same mesh 8× — est. **60–85%** of `extract_pile_tier_meshes`

**File:** [voxel-ffi/src/pile_preview.rs:100-204](voxel-ffi/src/pile_preview.rs)

**What:** The 8-tier cinematic reveal calls the full pipeline (`extract_hermite_data` → `solve_dc_vertices` → `generate_mesh` → `smooth` → `recalculate_normals` → `convert_mesh_to_ue_scaled` → `bucket_mesh_by_material`) once per tier. Tiers are **cumulative**, so tier 7 is a superset of tier 6 ⊂ tier 5… The same triangles are recomputed up to 8 times. Each tier also reallocates `temp_df.samples` (`grid_size³`, up to 96³ ≈ 884K cells) and a fresh `tier_cells` vec.

**Why slow:** Runs after every collapse on the worker thread before *any* tier ships to UE — the cinematic reveal latency is gated by total worker time, not per-tier time.

**Fix:** Generate the full pile mesh once at the top tier, then for tiers 0..N-1 either (a) filter triangles whose centroid Y is below the cumulative cutoff, or (b) sort solid_cells by Y once and emit incremental deltas (tier_k = tier_{k-1} ∪ new_cells_in_band_k). Reuse one `DensityField` and only zero-fill the cells that were touched in the previous tier (track a small dirty list).

**Estimated savings:** 60–85% of `extract_pile_tier_meshes` runtime. This is the single biggest win.

---

## Finding 2 — AOR diffusion allocates a fresh `Vec` per iteration — est. **40–60%** of AOR phase

**File:** [voxel-core/src/collapse_pile.rs:483-508](voxel-core/src/collapse_pile.rs)

**What:** `AOR_ITERATIONS = 8`. Each iteration does `let mut changes: Vec<(usize, f32)> = Vec::new();` then push/drain. For a 30×30 zone with 4 neighbors that's ~7,200 push/drain pairs × 8 = ~58K transient heap touches per collapse.

**Fix:** Allocate `changes` once outside the loop with `Vec::with_capacity(nx*nz*8)` and `.clear()` per iteration. One-line change.

**Estimated savings:** 40–60% of AOR diffusion (which itself is ~30–50% of pile placement). Total ≈ 10–25% of `place_collapse_pile`.

---

## Finding 3 — `cell_locations` 8-slot fan-out runs for interior cells too — est. **5–15%** of every flatten/pile/brush write

**File:** [voxel-core/src/density_ops.rs:59-92, 264](voxel-core/src/density_ops.rs)

**What:** `write_all_locations` always calls `cell_locations`, which builds `[Option<(IVec3, IVec3)>; 8]` via 3 nested for-loops. ~95% of cells inside a flatten/pile/brush zone are interior (single chunk) — they pay the full 8-fan-out cost for no reason.

**Fix:** Fast-path on the boundary booleans: `if !mx && !my && !mz { write_single_chunk(...); return; }`. Only build the 8-slot array when the cell actually straddles a chunk boundary.

**Estimated savings:** 5–15% of every write-heavy op (flatten, pile, formation removal, all 9 brushes). Hot — runs 5K–50K times per event.

---

## Finding 4 — Iterative formation removal re-walks the cylinder 3× via HashMap reads — est. **~50%** of formation-removal pass

**File:** [voxel-core/src/density_ops.rs:489-506](voxel-core/src/density_ops.rs), called from [voxel-ffi/src/flatten_sdf.rs:228](voxel-ffi/src/flatten_sdf.rs)

**What:** 3 erosion iterations, each iteration calls `count_air_face_neighbors` per cell, which does a `fields.get(&(cx,cy,cz))` HashMap lookup at the entry. For a contiguous sweep the chunk key is constant for many consecutive cells.

**Fix:** Snapshot the cylinder into a small flat `Vec<bool>` ("is solid") once, erode 3× in that local buffer (each erosion is just a face-neighbor read of a Vec), then write only the changed cells back to `fields`. Avoids 3× the HashMap traffic.

**Estimated savings:** ~50% of `formation_removal_pass`. Runs on every building flatten and every collapse pile placement.

---

## Finding 5 — Fluid weight scratch buffer reallocated every tick — est. **3–8%** of fluid tick

**File:** [voxel-fluid/src/sim/chunk.rs:171-182](voxel-fluid/src/sim/chunk.rs)

**What:** `tick_chunk` does `vec![0.0f32; total]` (typically 30³ = 27K floats = 108 KB) and fully fills it before the sim loop. Fresh alloc per chunk per tick.

**Fix:** Cache the buffer as a field on `ChunkFluidGrid` and overwrite in place. Or skip the precompute entirely — column weight is only consumed by Phase 4 (upward pressure), so compute lazily there.

**Estimated savings:** 3–8% of fluid tick. ~100 KB allocation per chunk per tick eliminated — also helps allocator behavior under multi-chunk fluid scenes.

---

## Finding 6 — Pile-preview tier filter scans entire `solid_cells` HashMap 8× — est. **70–85%** of tier_cells construction

**File:** [voxel-ffi/src/pile_preview.rs:114-117](voxel-ffi/src/pile_preview.rs)

**What:** For each of 8 tiers, full HashMap scan + `.collect()` into a fresh Vec.

**Fix:** Sort `solid_cells` once into a `Vec` keyed by Y. Each tier takes a prefix slice up to its cutoff Y — zero allocation, O(1) range selection.

**Estimated savings:** 70–85% of tier_cells construction. Stacks with Finding 1.

---

## Finding 7 — Stress recalc logging formats strings unconditionally in release — est. **1–5%** of stress recalc

**File:** [voxel-ffi/src/worker.rs:355-361, 394-487](voxel-ffi/src/worker.rs)

**What:** `format!("{:?}", material)` builds a String every call, then opens a log file each event. 30–50 of these per stress recalc, in release builds.

**Fix:** Cache existence of the log file path once per process; bail before `format!` if it doesn't exist. Or gate behind `cfg(debug_assertions)` / a static `AtomicBool`.

**Estimated savings:** A few ms per stress event — small but free.

---

## Finding 8 — `fragment_slab` rebuilds `HashMap<Material, u32>` per fragment — est. **5–10%** of fragmentation

**File:** [voxel-core/src/collapse_pile.rs:259-273, 292-303](voxel-core/src/collapse_pile.rs)

**What:** `vec![Vec::new(); nx*nz]` per collapse + per-fragment `HashMap<Material, u32>`. Material enum is bounded (≤49) — a stack `[u32; 49]` is faster and alloc-free.

**Fix:** Replace HashMap with `[u32; MATERIAL_COUNT]` indexed by `m as usize`. Reuse fragment bucket vec via `.clear()`.

**Estimated savings:** 5–10% of fragmentation; small absolute cost (fragmentation is not the dominant phase).

---

## Finding 9 — `build_support_hull` recomputes chunk key for every (dx,dz,dy) — minor

**File:** [voxel-ffi/src/flatten_sdf.rs:101-127](voxel-ffi/src/flatten_sdf.rs)

**What:** `read_density` does `div_euclid` + HashMap lookup per call inside `terrace_size² × SUPPORT_CHECK_DEPTH ≈ 64–256` calls per flatten. Adjacent cells share the chunk 99% of the time.

**Fix:** Cache the active `&DensityField` across the inner Y loop.

**Estimated savings:** 10–20% of hull build. Hull build is not a dominant path — low priority.

---

## Finding 10 — `flatten_sdf` debug log builds 600-char `format!` even when file open fails — minor (dev only)

**File:** [voxel-ffi/src/flatten_sdf.rs:286-310](voxel-ffi/src/flatten_sdf.rs)

**What:** Inside `#[cfg(debug_assertions)]` — `format!` runs before the file open check, plus extra density samples that aren't used if the file open fails.

**Fix:** Move sampling and `format!` inside `if let Ok(mut f) = OpenOptions::...`. Or gate behind an atomic bool flag (default off).

**Estimated savings:** 5–15% of dev/PIE flatten cost. Zero impact in release.

---

## Verdict

The flatten-SDF and collapse-pile rewrites are mostly well-engineered — the authors already inlined chunk-cache fast paths in `count_air_face_neighbors`, used `Vec`-backed structures, and called out the right patterns in comments. The wins left on the table cluster in two places:

1. **Pile preview tier extraction** (Findings 1, 6) — generates ≈8× redundant DC mesh work because tiers are cumulative supersets. Single biggest win, fixable by computing the full mesh once and slicing/sorting per cutoff Y. Order-of-magnitude latency drop on collapse cinematic.
2. **Per-iteration scratch allocations** (Findings 2, 3, 5) — `Vec::new()` inside loops, fresh `vec![0.0; 27K]` per fluid tick, `cell_locations` 8-slot fan-out on interior cells. Each one is a few-line fix with measurable savings on hot paths.

Findings 4, 7, 8, 9, 10 are smaller polish.

**Recommended next session:** start with #1 (pile preview merge), then #2 + #3 (one-line scratch-buffer changes), then #4 (cylinder snapshot for formation removal). That's likely 2–3 hours of work for a substantial post-collapse latency improvement and a measurable per-flatten/per-brush speedup.
