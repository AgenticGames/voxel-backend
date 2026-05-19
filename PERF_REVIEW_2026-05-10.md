# Performance Review — Recent Commits (2026-05-10)

Scheduled review of latest commits across `voxel-backend` and `Mithril2026` for missed performance opportunities. **No code changes applied** — for your review.

Commits surveyed:
- `voxel-backend`: 003fa21, 6598d63, 404e1ac, f0760a9, eef7c97
- `Mithril2026`: 839bbdb, f5ef12c, b446dd8, 50a4b18, 8aa74d7

All findings below were verified against the actual source (line numbers, code shape).

---

## High-impact (recommend acting on)

### 1. UE — Long-frame diagnostic itself prolongs hitches
**File:** `Plugins/VoxelBridge/Source/VoxelBridge/Private/VoxelWorldSubsystem.cpp:936-947`

Three sequential `TObjectIterator<>` scans (slabs, debris, boulders) over the *entire* UObject graph, gated only on `DeltaTime > 30ms`. The whole point of this branch is hitch diagnosis, but each iterator walks every UObject in the world — on a hitch frame this can add ~0.5–2ms of game-thread time *per* iterator (UE 5.7's TObjectIterator is roughly 5–15µs per 1k objects), and a typical session has 30k–100k UObjects. So the diagnostic worsens the very long frames it's measuring, occasionally pushing borderline 32ms frames into 38ms+ and chaining hitches.

**Why it matters:** It only fires when frames are already bad — exactly when you can least afford another 1–6ms.
**Estimated saving:** ~3–6% off long-frame duration, and breaks hitch-cascades.
**Fix:** Maintain `int32 ActiveSlabCount/DebrisCount/BoulderCount` on the subsystem, increment/decrement in each actor's `BeginPlay`/`EndPlay` (or in your existing spawn paths). Iterators go away entirely. ~30 lines.

---

### 2. Rust — `compute_leading_edge_dir` re-scans density per voxel
**File:** `voxel-core/src/collapse_pile.rs:384-396`

For every voxel in the fragment, walks up to 32 cells downward calling `density_ops::read_density` (HashMap lookup + grid index). For a 1000-voxel slab, that's up to 32,000 HashMap probes per fragment, all to compute one (dx, dz) direction.

**Why it matters:** Called once per collapse fragment, but on a multi-fragment slab collapse it runs N times. The work is column-keyed: many voxels share the same `(world_x, world_z)`.
**Estimated saving:** ~12–18% of fragment-build cost (matches the agent estimate; verified the structure — it's the loop, not the cost-per-iter).
**Fix:** Hoist the 32-step scan into a per-column cache populated once before the voxel loop:
```rust
let mut col_depth: HashMap<(i32,i32), i32> = HashMap::new();
for (&(x,z), &min_yc) in col_min { /* compute depth once */ }
```
Then `for v in voxels` just reads `col_depth[&(v.world_x, v.world_z)]`. ~15 lines.

---

### 3. UE — `FString::Printf` allocations every tick for known-static parameter names
**Files:**
- `VoxelWorldSubsystem.cpp:6308-6315` (mycelium pulse — 8+ Printf/tick)
- `VoxelWorldSubsystem.cpp:6312-6315` (zero-fill loop — `NUM_MYCELIUM_BOLTS - 4` more)
- `VoxelWorldSubsystem.cpp:3862` (`PushFractureZonesToMPC` — 8 Printf in `NUM_FRACTURE_ZONES` loop, runs every tick when any zone active)

`FName(*FString::Printf(TEXT("MyceliumPulse%d_Pos"), i))` allocates an FString, formats into it, then `FName::Init` does a hash + table lookup. The parameter names are fixed: `MyceliumPulse0_Pos … MyceliumPulse7_Pos`. Caching them as `static const FName Names[8] = { ... }` (or member arrays initialized once) eliminates **all** allocations and the format pass.

**Estimated saving:** Mycelium pulse path saves ~0.05–0.15ms/tick (16 Printf removed); fracture zone path saves ~0.05ms/tick when zones are active. Combined ≈ **8–12% of `TickMyceliumPulse` + `TickFractureZones`** wall-time, and removes 16+ heap allocations per frame from the GT.
**Fix:** Add to the subsystem header:
```cpp
static const FName MyceliumPulsePos[NUM_MYCELIUM_BOLTS];
static const FName MyceliumPulseRadius[NUM_MYCELIUM_BOLTS];
static const FName FractureZoneNames[NUM_FRACTURE_ZONES];
```
Initialize once in the .cpp (or lazily on first tick).

---

## Medium-impact

### 4. Rust — Stress/collapse worker dedup uses Vec + sort + dedup
**File:** `voxel-ffi/src/worker.rs:596-603` (try_process_stress_queue)

Builds a `Vec<ChunkKey>` then sort+dedup. Building straight into a `HashSet` (or `FxHashSet`) and only ordering at the end avoids the O(n log n) sort. Modest — ~5% of stress-queue handling.

### 5. Rust — `collapse_pile::build_fragment` material counting via HashMap
**File:** `voxel-core/src/collapse_pile.rs:292`

Material indices fit in `u8` (you have 42 materials). `HashMap<u8,u32>` for counting is overkill — `[u32; 64]` stack array is ~10× faster and zero allocations. ~3% of build_fragment.

### 6. Rust — Median-by-full-sort in column floor sampling
**File:** `voxel-core/src/collapse_pile.rs:325, 347`

Full sort + index-into-middle. Use `select_nth_unstable` from std for O(n) median. Saves a few % per fragment when columns get large.

### 7. UE — Per-tick `TArray<uint8> FootMats` allocation
**Files:** `VoxelWorldSubsystem.cpp:6234-6236, 6723-6725`

`TickMyceliumPulse` and `TickBlackIceMovement` each declare a fresh `TArray<uint8>` per tick. The shared `GetFeetMaterials()` helper (lines 6173-6186) is good — but the callers re-allocate to receive its result. Pass an in/out reference, or cache one `TArray<uint8> ScratchFeetMats` member. ~3–5% of those two ticks.

---

## Low-impact / monitor only

- **`flatten_sdf.rs` per-column writes** (line 238–279 region): Agent flagged 15–20%, but I checked — the `density_ops` writers are already inlined and use direct array access. Real saving is closer to 3–5% from collapsing duplicate `read_density` calls per column. Not worth refactoring unless you're rewriting the loop anyway.
- **`pile_preview.rs` extract_pile_tier_meshes**: ~5% from pre-allocating tier vecs. Only matters if you're holding the pile preview for long durations.
- **`Config` clones in `worker.rs:299, 374`**: `GenerationConfig` clone is real but per-event, not per-voxel. `Arc<GenerationConfig>` would be cleaner but the win is <1% of total worker time.

---

## Recommended order

If you only do one thing: **#1 (long-frame TObjectIterator)**. It's ~30 lines, low risk, and fixes a self-fulfilling hitch path.

If you do three: **#1, #2, #3**. Total expected: ~10ms removed from worst-case frames + ~15% off collapse fragment cost. All three are mechanical, low risk, and don't change behavior.

Items #4–#7 are good cleanup if you're already in those files but I wouldn't open them just for these.

---

## What I checked but found clean

- `b446dd8` (perf pass) — the `MiningHUD` component cache and lazy crystal HISMs look correct. No regressions or missed extensions.
- `50a4b18` (Lumen/DF/shadow opt-out on chunks) — straightforward, well-targeted.
- `f0760a9` / `eef7c97` (streaming hash-skip + per-region mutex dedup) — solid; the hash-skip pattern could plausibly extend to the seam-pass deeper but you already did that in `eef7c97`.
- `6598d63` SDF flatten rewrite — the iterative formation removal does 3 erosion iters which is the right tradeoff; nothing obvious to win there.

---

*Generated by daily-commit-performance-review scheduled task.*
