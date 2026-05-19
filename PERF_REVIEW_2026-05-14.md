# Perf review — last 6 commits (2026-05-14)

Scope: `267196b`, `6539312`, `a0ebed5`, `b2f522a`, `6bbb4dc`, `ddd4ac3`.

All wall-time % numbers are upper-bound estimates from a static cost model
(loop trip counts × per-iter cost relative to surrounding work). They are
"how much of this function's time could plausibly be reclaimed", not
end-to-end frame %. Where I write `~Xms` it's a rough hand-calc based on
the figures the commit messages already cite for the fluid tick.

---

## 267196b — OrePaint creative brush (new ~717 LOC)

### What it does
New `paint_ore_deposits` in `voxel-ffi/src/brushes.rs:417`. Phase 1 walks
solid voxels in the brush sphere and tags wall-exposed ones (6-neighbor
test, cross-chunk via `div_euclid`). Phase 2: Fisher-Yates shuffle +
Poisson-disk anchor selection. Phase 3: writes clusters + optional inward
channels via the (good) `paint_ore_sphere_voxels` helper, which already
batches one `get_mut` per chunk.

### Validated wins
The committed code is **better** than the diff message suggests: it
introduces `paint_ore_sphere_voxels` (line 638) that batches the AABB
chunk-iteration, replacing the per-voxel `write_ore_at_world` from the
original diff. That alone is the ~95% reduction in HashMap lookups the
comment claims, and it's real — confirmed by reading the current file.

### Missed opportunities

1. **Phase 2 anchor rejection is O(n²)** — `voxel-ffi/src/brushes.rs:600-612`.
   `accepted.iter().any(...)` scans every prior accepted anchor for each
   candidate. With `target_count = density * candidates.len()` and `density=1.0`
   this is up to `target * candidates` distance checks. For a radius-7 brush
   touching ~500 wall candidates and target ~150, that's ~75k length-squared
   calls. A spatial grid bucketed on `min_spacing` (1 HashMap<IVec3, Vec<usize>>
   indexed by `floor(pos / min_spacing)`) makes rejection O(27) per candidate.
   Saving: ~50–70% of Phase 2 wall-time on heavy density, which is itself
   maybe 5–10% of brush wall-time, so ~3–7% off the whole brush call.
   Estimation: 75k ops × ~10ns ≈ 0.75ms; bucket version ~0.15ms. Modest in
   absolute terms but easy.

2. **Phase 1 per-voxel cross-chunk neighbor probe**
   (`brushes.rs:497-548`). For every solid voxel inside the sphere, the
   inner loop does 6 neighbor checks; when a neighbor is OOB it does a full
   `store.density_fields.get(&nkey)` HashMap lookup plus `rem_euclid`. For a
   chunk-edge voxel that's ~6 HashMap lookups, but for interior voxels the
   OOB branch never fires (already in-chunk). The problem: the `else`
   branch still does `density_field.get(nx, ny, nz).material.is_solid()`
   which is fine, but the **outer loop reads `density_field.get(x,y,z)`
   followed by 6 neighbor `get()`s — 7 indexes per voxel**. For a radius-6
   brush at vs=1, that's ~900 solid voxels × 7 ≈ 6300 `get` calls.
   Replacing the 6 face-neighbor checks with a precomputed
   "is_solid_grid: BitVec" for the chunk lets you do 6 bit-fetches plus
   one preload at chunk entry. Saving: ~30–50% of Phase 1 wall-time.
   Phase 1 is the dominant phase for a brush that doesn't paint much
   (sparse density), so this is **~10–20% off total brush wall-time**.
   Cost model: 6300 `Sample` reads × ~5ns ≈ 31μs replaced by bit fetches
   at ~1ns ≈ 6μs. Small absolute, but the brush is a per-click op so it
   matters mainly for big radii.

3. **`per_chunk_dirty` uses `std::collections::HashMap`** (`brushes.rs:628`).
   This map peaks at the number of distinct chunks the brush spans —
   typically 1–8 chunks. A `Vec<(key, bounds)>` with linear search would
   be cheaper for that count and skip the hash. Worth ~tens of μs total,
   call it noise-level.

4. **`pick_ore` linear scan over `weight_arr`** (`brushes.rs:743`). Done once
   per anchor (~150 calls in heavy density), 13-element linear scan, no
   issue at all. Mentioning only to dismiss it.

5. **Each `paint_ore_sphere_voxels` call re-walks the AABB chunk range
   for every cluster + every channel step**. A radius-2 cluster always
   fits in 1 chunk, but the function still computes `cklo..=ckhi` and
   loops. With `channel_length=8` you call this 9 times per anchor for a
   tube. Hoisting "single-chunk fast path" (detect `cklo == ckhi`) avoids
   the triple-nested chunk loop overhead. ~5–10% off Phase 3 wall-time on
   tube-heavy strokes. Easy.

### Correctness / perf risks
- Phase 1 treats unloaded neighbors as **solid** (line 540) so we don't
  hallucinate walls at the streaming edge. Reasonable. But it means
  walls at chunk boundaries can be partially missed when the neighbor
  chunk isn't loaded yet — visual but not a perf bug.
- `pick_ore` falls through with `r -= w` on the wrong path if `total_weight`
  overflows (13 × u8 = max 3315, safe for u32). No bug.
- Big `WallCandidate` struct (4 × i32/usize + 2 × Vec3 = ~56 bytes).
  500 candidates ≈ 28KB heap allocation. Fine.

---

## 6539312 — `has_sources` flag

### What it does
`regen_sources` in `voxel-fluid/src/sim/utils.rs:491` now bails on chunks
with `has_sources=false`. Flag recomputed in `tick_chunk`'s fused fold.

### Validated win
Real. Same shape as `has_lava` from 6bbb4dc. Streaming worlds with thousands
of loaded chunks and a few dozen sources go from N³ × N_chunks walk to
boolean-check × N_chunks. For chunk_size=30 (27k cells) × 1000 chunks ×
3000 sources actually existing, the old cost was ~27M cell visits, new is
~3M (only sourced chunks). **~90% reduction in regen_sources wall-time**,
which the commit message already implies.

### Missed opportunities

1. **`equalize_horizontal` still walks every cell of every fluid chunk to
   build `water_cells`** (`utils.rs:84-114`). `has_fluid` short-circuits at
   the chunk level (good), but inside a fluid chunk we visit all 27k cells
   even when only a few hundred have water. Adding a `fluid_cells: Vec<u32>`
   index on `ChunkFluidGrid` (sorted indices of cells with level≥MIN_LEVEL,
   maintained incrementally by the same fold pass that sets `has_fluid`)
   turns this into "iterate ~N_fluid_cells per chunk" instead of "iterate
   N³". For a representative water-heavy scene with 5k fluid cells across
   100 chunks (50 cells/chunk average), old cost = 100 × 27k = 2.7M reads,
   new = 5k reads. **~99% reduction in the water_cells build phase**, which
   is roughly 40–60% of equalize_horizontal's wall-time post-b2f522a
   (the BFS+inserts are the other 40–60%). Net ~25–50% off `equalize_horizontal`.
   Cost: maintaining the sorted Vec on every cell write inside `tick_chunk`
   adds a few % to tick_chunk. Net win is positive in steady state because
   `equalize_horizontal` runs every water tick on every world; tick_chunk
   writes are bounded by N_cells_with_fluid which is the same set.

2. **`thread.rs:597` lava-cleanup loop** does `for idx in 0..total` to
   zero lava cells, gated only by has-fluid (not has_lava). This is the
   same shape as the original `detect_lava_water_quench` problem. Add a
   `has_lava` short-circuit here too. ~10–20% of that loop's wall-time on
   water-only worlds (the loop is short, so absolute saving is small —
   call it sub-μs per chunk).

3. **`apply_pending_fluid` sets `has_sources=true`** unconditionally in
   `thread.rs:739` if `is_source` was set on the source-cell path, but
   `is_source` isn't checked at that line — it's inside the `is_source`
   branch in the diff. Looks correct. **However** the flag is set on
   write paths but never **cleared** when the last source is mined/cleared.
   If a player carves through every source in a chunk, the flag stays
   true and `regen_sources` keeps doing the full N³ walk on that chunk.
   The per-tick fold in `tick_chunk` does recompute `any_source` from
   `cell.is_source` — so the flag self-heals **on the next tick where
   tick_chunk runs**, which is gated by `has_fluid`. If the last source
   was also the last fluid in the chunk, `has_fluid` is false and
   `tick_chunk` early-returns before recomputing flags. **`has_sources`
   can stay stuck at `true` on a fluid-empty chunk forever**, but in
   that case `regen_sources` walks 27k cells finding no source and does
   nothing. Wasted ~25μs per stuck chunk per tick. Worth gating
   `regen_sources` on `has_fluid && has_sources`. **Same risk for
   `has_lava`**.

---

## a0ebed5 — `QuenchScratch` reuse

### What it does
Hoists 4 HashSets + 1 Vec out of `detect_lava_water_quench` into a
caller-owned `QuenchScratch` struct in `voxel-fluid/src/sim/utils.rs:243`.

### Validated win
Real but modest. Per-call savings on alloc are ~150 transient allocs ×
~50ns/alloc ≈ 7.5μs/tick during active quench. The commit claims 10–20%
of quench wall-time; that's plausible for short scans, conservative for
long ones (BFS dominates). The structural change is a clear strict
improvement.

### Missed opportunities

1. **`QuenchPlan` still allocates 4 Vecs per call** (line ~395).
   `obsidian_set.into_iter().collect()` etc. The commit acknowledges this
   but defers because the Vecs go down a channel. **Could ship a pool**:
   the channel receiver returns the spent `Vec`s to the sender after
   consumption (e.g. via a ring of `Option<Vec<CellAddr>>` slots).
   ~4 × ~N_contact_cells alloc per tick → 0. Probably not worth the
   complexity until profiling shows it; flag for later.

2. **`scoria_set: HashSet` is keyed on `CellAddr=((i32,i32,i32),usize,usize,usize)`** —
   that's 36 bytes per entry. A typical quench front of ~200 contact
   cells with depth-3 BFS hits ~5000 entries. HashSet on a 36-byte key
   is slow; if cells stay within one chunk (which the comment confirms),
   key it as `(chunk_idx, packed_xyz_u32)` (8 bytes). Or just use a per-chunk
   `BitVec` of size N³ for the BFS-visited mask. ~30–50% off BFS phase,
   which is most of `detect_lava_water_quench` wall-time. **Estimated 10–25%
   off `detect_lava_water_quench` overall**. Bigger and more impactful
   than the scratch-reuse this commit shipped.

3. **`bfs_visited` is also a `HashSet<(usize,usize,usize)>`** — same fix
   (per-chunk BitVec). On the BFS hot loop the HashSet lookup is the
   inner-loop hot path.

---

## b2f522a — drop Y-loop in `equalize_horizontal`

### What it does
Removes `for wy in min_world_y..=max_world_y` outer loop; iterates
`water_cells` once. Hoists `region` and `queue` out of the loop.

### Validated win
Real and well-described. 200k key visits → 5k key visits. The 0.4–0.8 ms/tick
claim matches my back-of-envelope (5k HashMap key-visits × ~40ns ≈ 0.2ms,
200k × 40ns ≈ 8ms — the commit's "60–80% off" is correct for the iteration
component; total function time includes BFS which doesn't change).

### Missed opportunities

1. **`water_cells: HashMap` is still the wrong container**. Keys are
   already enumerable; we never need true random access by world-pos
   except inside the BFS for neighbor existence checks. A `HashSet`
   for membership + a `Vec` of `(pos, data)` for iteration is one
   less HashMap. Or — better — index water_cells by a packed `i64`
   (XYZ → 21 bits each) to halve the hash cost. Saving: ~10–15% off
   the water_cells build + membership checks. Net ~5–10% off
   `equalize_horizontal`.

2. **BFS membership check is `water_cells.contains_key(&neighbor)`**
   on every neighbor of every region cell — that's 4 hash lookups per
   BFS step. With ~5000 water cells in connected regions, that's 20k
   hashes. A flat `HashSet<i64>` (packed coords) or per-region BitVec
   is ~3× faster. ~10–20% off BFS wall-time, ~5–10% off function.

3. **The bigger win — see commit 6539312 §1**: the whole `water_cells`
   build is N³ × N_chunks. Index of fluid cells on the grid kills it.

### Risk
None spotted. The Y-disjoint argument is correct: BFS only offsets X
and Z. The visited set dedups across the iteration order. Tests pass.

---

## 6bbb4dc — fractional capacity + scratch reuse + has_lava

### What it does
Three changes (1) `cell_cap = air_corners/8` instead of binary 0/1;
(2) `scratch_cells/weights/drain` on `ChunkFluidGrid`, taken via `mem::take`;
(3) `has_lava` flag.

### Validated wins
- **Scratch reuse** is the big one. At chunk_size=30, `cells.clone()` is
  ~540KB. Allocated 6× per water tick × ~50 active chunks/sec = ~160MB/sec
  of heap traffic. Now 0 in steady state. **Largest single tick-perf win
  in this batch.** Real.
- **has_lava** mirrors `has_sources` analysis above. Real.
- **Fractional capacity** is a correctness fix (wall-clipping); perf neutral
  in arithmetic terms but reduces "fluid bleeding through" and the resulting
  pointless redistribution every tick, so it's a positive on busy boundaries.

### Missed opportunities

1. **The take/swap-back pattern requires `get_mut` then `get` then
   `get_mut`** (`chunk.rs:164-178, 180, 845+`). 3 HashMap lookups per
   chunk per substep × 6 substeps × N chunks. With chunks counted in the
   hundreds, that's a few thousand redundant hashes per tick. Hold a
   single `&mut ChunkFluidGrid` for the whole function by splitting the
   grid's reads from the cross-chunk transfer writes (the only reason
   the borrow is dropped). Saving: ~3–5% off tick_chunk on streaming
   worlds. Worth it because tick_chunk dominates fluid-thread wall-time.

2. **`new_cells.copy_from_slice(&g.cells)`** at line 170 is a 540KB memcpy
   on every substep. That's still ~540KB × 6 substeps × 50 chunks =
   ~160MB/sec of memcpy traffic. The copy is required because the algorithm
   needs the pre-state. But you can **swap the roles**: read from
   `g.cells` and write into `scratch_cells`, then swap at end. That's
   what the diff does (line 855 `mem::swap`). However we **still copy
   first to seed scratch**. The smarter pattern: maintain `scratch_cells`
   pre-seeded across ticks (so it's always a valid-but-stale copy), then
   at top: `mem::swap(&mut g.cells, &mut scratch_cells)` (now scratch is
   live, g.cells is the read-snapshot), write into g.cells using scratch
   as the read source, no memcpy needed. Saves the 540KB×6×50 = 160MB/sec
   memcpy. **~5–10% off tick_chunk on chunk_size=30 worlds.** Trickier
   than the current scheme — risk of staleness if other writers touch
   cells outside tick_chunk — but doable.

3. **`fluid_weight` rebuilt from zero every tick** (line 175-176, 189-198).
   It's a derived quantity (cumulative column sum) that only changes when
   that column's fluid changes. For columns where nothing changes
   between ticks it's recomputed for nothing. Cost is ~27k float adds
   per chunk per tick. Hard to incrementalize cleanly; skip.

4. **`is_source` check inside the fold for `any_source`** (post-6539312):
   `cell.is_source` is a separate bool from `level >= MIN_LEVEL`, so the
   branch ordering is fine. Note the fold runs every tick — see "stuck
   flag" risk in 6539312 §3.

### Correctness risk
The `mem::take` of scratch on first call sees empty Vecs, hits the `else`
branch at line 171 (`nc.extend_from_slice`), which allocates. After that
they're sized and `copy_from_slice` runs. So **the first tick per chunk
still allocates** the scratch buffers. That's a one-time cost per loaded
chunk, fine.

---

## ddd4ac3 — PaintStress brush + painted-stress overlay

### What it does
Adds `paint_stress_sphere` (`voxel-ffi/src/brushes.rs:208`), per-chunk
`StressField.painted_stress: Vec<f32>` (lazy-alloc), `effective() = stress
+ painted`. Folded into `recalc_stress_region*` overstressed check.

### Validated wins
The lazy-alloc (empty Vec = no allocation) is the right shape — chunks
without painted stress cost zero RAM. Confirmed in `voxel-core/src/stress.rs:104`.

### Missed opportunities

1. **`paint_stress_sphere` recomputes `d.sqrt()` per voxel** (`brushes.rs:269-282`).
   `falloff=1` (linear) and `falloff=2` (smoothstep) both call `d2.sqrt()`.
   For falloff=0 the `match` is fine. For falloff=1/2 you can work in `d2/r2`
   space throughout. `linear: 1 - sqrt(d2/r2)` → keep the sqrt; **but for
   "is it inside the sphere"** check (line 264 `if d2 > r2`) we already
   have d2. Smoothstep can use a polynomial in `(1 - d2/r2)` — not identical
   shape, but visually equivalent. Saving: ~20–30% of paint loop wall-time
   on large radii (sqrt ≈ 10ns × N_voxels). For a radius-8 brush ≈ 2000
   voxels, ~20μs saved per click. Tiny absolute, but mentioning because
   the brush is the only place these sqrts run.

2. **`add_painted` is called per-voxel, does a HashMap-like lookup pattern
   via `ensure_painted_alloc`** (`stress.rs:189-201`). Each call: check
   empty, then index. The `ensure` runs every call but is no-op after first.
   Inside a hot inner loop that's still a branch + bounds check. Hoist
   `ensure_painted_alloc()` outside the loop in `paint_stress_sphere`
   (once per chunk, after `stress_fields.entry(...).or_insert_with(...)`),
   then write directly: `sf.painted_stress[idx] = ...`. Saves ~one branch
   per voxel. ~5–10% of paint-loop wall-time.

3. **`recalc_stress_region_v2_filtered` does
   `stress_fields.get(&(cx,cy,cz)).map(...)` **immediately followed by**
   `stress_fields.get_mut(&(cx,cy,cz))`** (`voxel-core/src/stress.rs:1457-1465`).
   Two HashMap lookups for the same key per voxel. For a recalc covering
   tens of thousands of voxels, that's tens of thousands of extra hashes.
   Restructure to a single `get_mut`:
   ```rust
   if let Some(sf) = stress_fields.get_mut(&(cx, cy, cz)) {
       let painted = sf.painted(x, y, z);
       sf.set(x, y, z, stress);
       sf.set_class(x, y, z, classification);
       affected_chunks.insert((cx, cy, cz));
       let eff = stress + painted;
       if eff >= 1.0 { overstressed.push(...); }
   }
   ```
   Saving: ~one HashMap lookup per voxel. At chunk_size=30 in a multi-chunk
   recalc that's 27k+ saved lookups. **Estimated 5–15% off
   `recalc_stress_region_v2_filtered` wall-time on heavy recalcs.** Likely
   the highest-impact item in this commit since recalc runs on every mining
   action plus during sleep stress passes.

4. **`painted_stress: Vec<f32>` is 4 bytes/voxel × 27k voxels = 108KB per
   painted chunk**. Most painted regions are small (sphere of radius ~5 ≈
   500 voxels). A sparse representation (`HashMap<u32, f32>` or `Vec<(u32,f32)>`
   sorted) would use ~5KB instead. Trade-off: per-voxel lookup becomes a
   hash or binary search. For the recalc hot path that's worse than direct
   indexing. For RAM-constrained streaming worlds with many small paints,
   the dense rep is fine. Skip.

5. **Save format v3** writes `painted_stress: Option<Vec<u8>>` — encodes
   the f32 grid as bytes. The `apply_painted_stress_to` path likely does
   per-voxel float decode. Worth checking; if it's already bulk
   `bytemuck::cast_slice`-ing the bytes back to f32, it's fine.

### Correctness / perf risks
- The `recalc_stress_region` legacy path (line 1486+) has the **same**
  double-lookup pattern. Same fix applies.
- `effective()` and `painted()` both branch on `painted_stress.is_empty()`.
  If the recalc loop iterates many voxels in a chunk **without** a painted
  layer, the branch is well-predicted but still per-call. Hoist the
  `has_painted_layer()` test once per chunk.

---

## Priority recommendations

Ranked by impact × ease. Cite the file/line so you can pull-the-thread.

| # | Item | File | Est. impact | Ease |
|---|---|---|---|---|
| 1 | **Fluid-cell sparse index** (`Vec<u32>` of cells-with-fluid, maintained in `tick_chunk` fold) eliminates the N³ inner scan in `equalize_horizontal` and similar passes. | `voxel-fluid/src/sim/utils.rs:84-114`, `cell.rs` (add field) | ~25–50% off `equalize_horizontal`, also helps every future "scan fluid cells" pass | Medium — maintenance hook in tick_chunk |
| 2 | **Single get_mut in `recalc_stress_region_v2_filtered`** (also legacy `recalc_stress_region`) — drop the `get(...)` before the `get_mut(...)`. | `voxel-core/src/stress.rs:1457-1465`, `:1510-1520` | ~5–15% off recalc on multi-chunk passes; recalc runs on every mining action | Trivial |
| 3 | **Per-chunk BitVec for quench BFS visited set** replacing `HashSet<(usize,usize,usize)>`. | `voxel-fluid/src/sim/utils.rs:361-407` | ~10–25% off `detect_lava_water_quench` | Easy |
| 4 | **OrePaint Phase 2 spatial grid for anchor rejection** — drop O(n²) Poisson-disk rejection. | `voxel-ffi/src/brushes.rs:600-612` | ~3–7% off brush wall-time at high density | Easy |
| 5 | **Skip the per-substep memcpy in `tick_chunk`** by maintaining `scratch_cells` as a rolling copy across ticks. | `voxel-fluid/src/sim/chunk.rs:164-178` | ~5–10% off `tick_chunk`, which is the #1 fluid-thread function | Medium — staleness risk |

Also worth chasing eventually: `has_sources`/`has_lava` stuck-flag
recovery for fluid-empty chunks (cheap correctness improvement, sub-μs
perf), and hoisting `ensure_painted_alloc()` out of the paint inner loop
(trivial).
