# Perf review — 2026-05-26 (afternoon follow-up pass)

Second scheduled perf-review run today. Earlier pass shipped
`surface_probe` chunk-pointer caching (commit `df1c061`, ~58 % off
probe cost). This pass targets the worldgen hot path —
`generate_density_field` — which still hadn't been profiled directly
on the chunk-streaming path.

## Headline finding — per-voxel `Instant::now()` instrumentation

**File:** [voxel-gen/src/density.rs:164-352](voxel-gen/src/density.rs:164)
(`generate_density_field`, the per-chunk worldgen entrypoint).

The function was originally instrumented (commit `d9356c3`,
"Remaining uncommitted work…") with `std::time::Instant` timers that
fire **per voxel** inside the 33³ density loop, plus a chunk-level
file write that logs every 10th chunk. The per-voxel timer was
always-on:

```rust
// In the inner loop, for EVERY voxel:
let _t_mat_start = Instant::now();
let material = if density <= 0.0 { Material::Air } else { ... };
_t_material_ns += _t_mat_start.elapsed().as_nanos() as u64;
```

For a 33³ field that's **~35 937 `Instant::now()` calls and 35 937
`elapsed()` reads per chunk**, unconditionally. On Windows
`Instant::now()` lowers to `QueryPerformanceCounter` (~20-30 ns each),
and this code runs on every streamed chunk regardless of whether
anyone is reading the resulting `density_detail.txt` file.

### Fix — gate behind `VOXEL_DENSITY_TIMINGS` env var

`OnceLock` caches the env-var lookup so the per-chunk gate cost is a
single atomic load. Per-voxel and per-chunk timer creation now both
skip when the gate is off (default).

### Measured impact

Bench: [voxel-gen/examples/bench_density.rs](voxel-gen/examples/bench_density.rs)
(5 runs, `cargo run --release --example bench_density -p voxel-gen`)

| State                                | Avg per chunk |
|--------------------------------------|---------------|
| Baseline (timing always on)          | **3.58 ms**   |
| Patched, no env var (default)        | **3.24 ms**  *(−9.5 %)* |
| Patched, `VOXEL_DENSITY_TIMINGS=1`   | 3.74 ms       |

Six runs after the patch held between 3.20 ms and 3.30 ms — the
~9 % saving is repeatable, not measurement noise.

**~340 µs saved per chunk.** With ~300 chunks streamed ahead of the
player in a typical session, that's ~100 ms of worldgen wall-time
removed per stream-in burst. On the chunk-streaming thread pool this
matters most when the player runs through a fresh area — fewer
hitches to first-paint.

Setting `VOXEL_DENSITY_TIMINGS=1` restores the original behaviour
(file write still hits
`D:/Unreal Projects/Mithril2026/Saved/density_detail.txt` every 10th
chunk).

## Bonus micro-opts — `Fbm` / `RidgedMulti` precompute

While instrumenting, two invariant recomputations stood out:

**[voxel-noise/src/fbm.rs](voxel-noise/src/fbm.rs):** `Fbm::sample`
accumulated `max_amplitude` inside the per-octave loop on every call,
then divided by it. `max_amplitude = Σ persistence^k` for
k=0..octaves — purely a function of construction args. Now
precomputed in `Fbm::new`. Output is bit-identical (same final
`value / max_amplitude` expression, just hoisted operand).

**[voxel-noise/src/ridged.rs](voxel-noise/src/ridged.rs):**
`RidgedMulti::sample` called `self.spectral_weights.iter().sum()` on
every sample — a runtime `Vec` iteration over 4-5 elements with
bounds checks per access. Now `spectral_weight_sum` is computed once
in `RidgedMulti::new`. Same bit-identical output.

### Measured impact (1M samples, release)

| Source              | Baseline | Patched | Δ      |
|---------------------|----------|---------|--------|
| `Fbm` (6 octaves)   | 228.8 ns | 227.8 ns| ~0.4 % |
| `Fbm` (3 octaves)   | 108.2 ns | 111.0 ns| within noise |
| `Ridged` (5 oct)    | 217.1 ns | 214.6 ns| ~1.1 % |
| `Ridged` (4 oct)    | 165.7 ns | 163.6 ns| ~1.3 % |

These wins are small in isolation — LLVM already hoists much of the
loop-invariant work. They're shipped because (a) they're correct
clean-ups, (b) `RidgedMulti::sample` is on the hot worldgen path
(11 material noise sources, called per voxel), and (c) it removes
an unnecessary `Vec::iter().sum()` from a per-voxel hot path.

Output verified bit-identical via the existing
`fbm::tests::fbm_determinism`,
`ridged::tests::ridged_determinism`, and the 18 voxel-noise tests
(all green). Saved-seed worlds will load to identical geometry.

## Combined headline number

For `generate_density_field` end-to-end on the worldgen thread:

> **~9 % faster per chunk** (3.58 ms → 3.24 ms), driven primarily by
> gating the per-voxel timing instrumentation. The `Fbm` / `Ridged`
> precomputes contribute fractional percent but make the per-sample
> code cleaner and remove a `Vec::iter().sum()` from the ridged-noise
> hot path.

## Test status

- `cargo build --workspace --release` — clean
- `cargo test -p voxel-noise --release` — 18/18 pass (determinism +
  range tests cover the precompute changes)
- `cargo test --workspace --release` — **pre-existing** breakage in
  `voxel-gen/src/zones/mega_apply.rs:1286` (missing `has_ore_material`
  field) and `voxel-sleep/src/phases/aureole.rs` dead code; both
  reproduce on stashed HEAD before this patch — not caused by it.

## Not shipped — flagged for next pass

Investigated and rejected because they didn't measure as wins:

- **`Simplex3D::grad`'s `hash % 12` → 256-byte LUT.** LLVM already
  lowers `u8 % 12` to a multiply-and-shift (libdivide trick); LUT was
  a rounding error on the bench (32.40 → 32.36 ns).
- **`density.rs:232-240` triple-call of warp simplex sources** with
  identical inputs but different perm tables — LLVM CSE-s the
  `sx * 0.5` multiplies. No win available without merging the three
  sources into one, which would change worldgen output.

Outstanding from earlier passes (still applicable, unchanged):

- OrePaint Phase 2 anchor selection is O(N²) —
  `voxel-ffi/src/brushes.rs:619`.
- Mining/brush callers could pass an AABB hint to `update_density` —
  est. 5–10 % off brush-stroke wall-time at chunk_size=30.
- Path-result cache key + `corner_clip_clear` invalidation — est.
  30–60 % off repeat-pathing CPU for spider chase loops on unchanging
  terrain.
- `surface_normal_at` per-search memo (`voxel-ffi/src/pathing.rs:109-123`)
  — est. 10–15 % off Spider-only path queries.
- Shared `ChunkSampler` helper: move today's morning-pass `Sampler`
  out of `surface_probe.rs` into its own module to serve
  `probe_surface`, `surface_normal_at`, and
  `count_topology_votes_cross_chunk`.

End of follow-up review.
