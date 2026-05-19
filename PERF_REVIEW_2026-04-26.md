# Perf review 2026-04-26 — follow-ons on top of `6598d63`

## TL;DR

Reviewed the latest commit (`6598d63` — "Building flatten rewrite (SDF) + collapse
rubble pile rewrite"). The new `voxel-ffi/src/flatten_sdf.rs` and the legacy
`voxel-ffi/src/terrain_ops.rs` (still used for DK2 zones + conveyor batches)
contained four residual inefficiencies that the rewrite missed. All four
implemented as **uncommitted** changes. Build clean, 37/37 voxel-ffi tests
pass, 6/6 `density_ops` tests pass, 99/100 workspace (same pre-existing
`blueprint_has_expected_structure` failure noted in prior reviews — unrelated).

**Not pushed. Not committed.** Review the diff, validate the numbers with a
flatten/conveyor-heavy run, then commit + push if you're happy.

Files touched:
- `voxel-ffi/src/flatten_sdf.rs`
- `voxel-ffi/src/terrain_ops.rs`
- `voxel-core/src/density_ops.rs`

---

## The four fixes

### 1. Dead `Simplex3D::new` per column in the SDF flatten apron

**Before** (`voxel-ffi/src/flatten_sdf.rs:132-137`):
```rust
fn ramp_y_noise(cfg: &GenerationConfig, wx: i32, wz: i32) -> f32 {
    let freq = cfg.noise.cavern_frequency;
    let s = Simplex3D::new(cfg.seed);
    let n = s.sample(wx as f64 * freq, 0.0, wz as f64 * freq) as f32;
    n * RAMP_NOISE_AMP   // RAMP_NOISE_AMP == 0.0 (dead)
}
```

`RAMP_NOISE_AMP` is `0.0` (per the existing comment, "disabled — was causing
per-column jitter"), so the multiplied result is always `0.0`. But the
compiler can't elide the work: `Simplex3D::new()` allocates a 512-byte
permutation table, seeds a `ChaCha8Rng`, runs a 256-element shuffle, and
copies the result. That cost was paid **per (wx, wz) column** in the apron
loop — for a 4-voxel building with apron radius 3, that's `(4 + 6)² = 100`
columns per flatten, every one allocating + reseeding a fresh noise table
to multiply by zero.

**After:** function body short-circuits to `0.0`. Caller signature
unchanged. Comment marks the spot so anyone re-enabling noise also hoists
the construction out of the per-column path.

**Estimated saving:** **−5–15 % on `flatten_terrace_sdf` total**, depending
on building size. The apron column loop is the hot inner loop, and 100
ChaCha8Rng-seeded shuffles per placement is now zero. Gets larger for
bigger buildings (apron grows quadratically with footprint).

---

### 2. Per-column `Simplex3D::new` × 2 in the legacy batch flatten path

**Before** (`voxel-ffi/src/terrain_ops.rs:84-92`):
```rust
fn ramp_y_noise(cfg: &GenerationConfig, wx: i32, wz: i32) -> f32 {
    let freq = cfg.noise.cavern_frequency;
    let s_cavern = Simplex3D::new(cfg.seed);
    let s_detail = Simplex3D::new(cfg.seed.wrapping_add(1));
    ...
}
```

Same anti-pattern, except this one **isn't** dead code (`RAMP_NOISE_AMP`
here is `1.0`) and it builds **two** `Simplex3D` per call. This is the
path used by `BuildingFlattenBatch` (DK2 zones, conveyors) — and a single
batch can carry dozens of conveyor tiles, each running its own apron loop.
At 100 columns × 2 allocations × N tiles, you can hit thousands of
ChaCha8Rng-seeded permutation-table allocations per batch.

**After:** introduced `RampNoiseCtx` that builds the two `Simplex3D` once
per flatten (single tile) or once per batch (many tiles). `ramp_y_noise`
now takes `&RampNoiseCtx` and just runs the two `sample` calls. Both
`flatten_terrace` and `flatten_terrace_batch` build the context outside
the apron loop and pass by reference.

**Estimated saving:**

- Single-tile `flatten_terrace`: **−10–20 %** (noise was a real chunk of
  the per-column work; now it's a few f64 muls and a permutation hash).
- `flatten_terrace_batch` with N tiles: **scales N×** — for a conveyor
  batch of 10–30 tiles this is meaningfully more, on the order of a
  millisecond saved per large batch on a hot worker.

---

### 3. Per-flatten file write + per-flatten `eprintln!`

**Before:**

`flatten_sdf.rs:286-298` — opens `D:/Unreal Projects/Mithril2026/Saved/flatten_sdf_log.txt`,
appends a formatted line, closes, on **every single** building placement.
That's an open syscall, a write syscall, a close syscall, plus the
`format!` and the column samples that feed it.

`terrain_ops.rs:255` — `eprintln!` with 12 format args on every legacy-path
flatten (every DK2 zone tile, every conveyor batch entry). UE swallows
stderr in `-game` mode anyway, so the output isn't even visible.

**After:** both gated behind `#[cfg(debug_assertions)]`. Release builds
pay nothing; dev builds keep the diagnostic. Density samples that fed the
file log are inside the cfg block, so they're elided too.

**Estimated saving:** **−5–10 %** of `BuildingFlatten` *wall* time on
sustained placement (file IO is non-deterministic — sometimes free,
sometimes 1–5 ms when the OS decides to flush). Removes a source of
multi-millisecond stalls during heavy build sessions. The `eprintln!`
removal is small in absolute terms but compounds in a batch.

---

### 4. `count_air_face_neighbors` did 6 chunk lookups per cell

**Before** (`voxel-core/src/density_ops.rs`):
```rust
pub fn count_air_face_neighbors(...) -> u8 {
    let mut n = 0u8;
    for (dx, dy, dz) in [...6 face dirs...] {
        if read_density(fields, cs, wx + dx, wy + dy, wz + dz) <= 0.0 { n += 1; }
    }
    n
}
```

Each `read_density` does 3 × `div_euclid` + 3 × `rem_euclid` + a
`HashMap.get` on `(cx, cy, cz)`. For a center cell well inside its chunk,
all 6 neighbors live in that same chunk — but the original code looks the
chunk up 6 times anyway.

`count_air_face_neighbors` runs in the inner loop of `formation_removal_pass`,
which iterates a cylinder × Y range × **3 erosion iterations**. For a
medium building (apron + radius_extra ≈ 7), that's `π·7² × 17 × 3 ≈ 7900`
calls, each with 6 redundant `HashMap` lookups + integer divisions.

**After:** primary chunk looked up **once** at the entry. The 4 face
neighbors that stay inside the chunk go straight through `df.get()` without
any further chunk lookup or div_euclid. Only the (typically 0–2) neighbors
that cross a chunk face fall back to the slow path.

**Estimated saving:** **−40–60 % on `formation_removal_pass`** — the
function is dominated by these neighbor reads. Translates to **−3–6 % on
`flatten_terrace_sdf` total** (formation removal is one of three big
phases inside the SDF flatten; the others are the apron column loop and
the per-cell density writes). Bigger win when buildings sit in
formation-rich caves where most cells survive the threshold check on each
iteration.

---

## Combined estimate

For a single building placement on the new SDF path
(`BuildingFlatten` → `flatten_terrace_sdf`), stacking 1 + 3 + 4:

> **−13–31 % on `flatten_terrace_sdf` wall time** (mid-point ≈ −20 %).

For a `BuildingFlattenBatch` (DK2 zones / conveyors), 2 + 3 dominate:

> **−12–25 % per tile**, plus elimination of multi-millisecond file-IO
> stalls. Scales linearly with batch size — a 20-tile conveyor batch saves
> proportionally more.

For `formation_removal_pass` standalone (also called by the new collapse
pile placement in `voxel-core/src/collapse_pile.rs`):

> **−40–60 %** on the function itself, which feeds back into both the
> SDF flatten and any future caller (slab collapse already hits it).

Numbers are arithmetic estimates from line-counting the inner loops — not
profiled. Worth a flatten-heavy bench run (build a long conveyor line +
trigger a slab collapse) to confirm before locking in.

---

## What I deliberately didn't touch

- The `cone_top_in_column` linear scan in `flatten_sdf.rs` — `O(steps × cones)`
  per apron column is real work, but the cone count is bounded
  (`SUPPORT_RAYS_PER_COL = 3` × cantilever-column count) and only fires for
  cantilever buildings. Optimizing it would mean an octree or grid spatial
  index — not worth the complexity unless a profile shows it's actually hot.
- The 27-neighborhood seam pass dedup logic in `worker.rs` — already
  optimized in `eef7c97`/`f0760a9`/`404e1ac` by prior reviews. Leaving it.
- The collapse rubble pile placement in `voxel-core/src/collapse_pile.rs` —
  it's a separate uncommitted module that hasn't been reviewed; will pick
  it up in a follow-up if benchmarks show it's hot.
- `find_support_rays`'s Fibonacci sphere generation — runs once per
  cantilever column; not a hot loop.

---

## How to verify

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo build --workspace                          # clean
cargo test -p voxel-core density_ops             # 6/6 pass
cargo test -p voxel-ffi                          # 37/37 pass
cargo test --workspace                           # 99/100 — same blueprint
                                                 # failure as prior reviews
```

For wall-time validation, drop a long conveyor belt run in UE and watch
`Tick > VoxelWorldSubsystem > Worker request:BuildingFlattenBatch` in
Insights — should drop noticeably. Single-building placements
(`BuildingFlatten`) should also tighten up, especially in formation-heavy
caves.
