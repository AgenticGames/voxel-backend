# Perf review — 2026-05-04 (autonomous, scheduled task)

Reviewed the latest commits in `voxel-backend` and the UE plugin and applied
one focused improvement to the SDF building-flatten hot path.

## Commits reviewed
- `voxel-backend@6598d63` — Building flatten rewrite (SDF) + collapse rubble
  pile rewrite. Adds `voxel-ffi/src/flatten_sdf.rs` (419 lines) and
  `voxel-ffi/src/sdf.rs` (300 lines). Single-placement flatten now goes
  through `flatten_terrace_sdf`.
- `Mithril2026@f5ef12c` — Fabricator dual-mode + bag + research-gated
  recipes. Pure UI/gameplay; no obvious perf hotspots beyond what's already
  there. Skipped.

## Improvement applied (NOT pushed — left for review)

### Files touched
- `voxel-ffi/src/sdf.rs`
- `voxel-ffi/src/flatten_sdf.rs`

### What changed

**1. Removed dead code in `sdf_capped_cone`** (free win, was pure waste)

The previous implementation computed `dx` and `dy` distance components of the
trapezoid silhouette and then immediately discarded them with
`let _ = dx; let _ = dy;`. Two subtractions, an abs/clamp/max chain, and a
max — every call paid for results nobody read.

**2. Added `CompiledCone` + `sdf_compiled_cone` to skip per-call sqrt+div**

Each cone in the support hull was being sampled by
`SupportHull::cone_top_in_column` at ~14 Y steps per apron column, across
~16-30 apron columns per placement. Every sample recomputed:
- `axis.length()` — 1 sqrt + 1 div (axis normalize)
- `len_side = sqrt(dx_side² + dy_side²)` — 1 more sqrt + 2 divs (`nx`, `ny`)

These are all functions of `(base, tip, r_base, r_tip)` and invariant per
cone. Now precomputed once in `compile_cone()`. The hot per-sample path
shrinks from ~3 sqrt + 3 div + several mults down to **1 sqrt + 0 div** in
the common (inside cone height range) branch.

The wrapping `sdf_capped_cone(p, base, tip, r_base, r_tip)` keeps its public
signature for any external callers — it now just calls `compile_cone` then
`sdf_compiled_cone`.

**3. Sphere-traced the column descent in `cone_top_in_column`**

The previous loop marched Y down by a fixed `step = 0.5` from
`search_hi` (~+1.0) down to `search_lo` (~-7.0) — 16 fixed steps minimum,
even when the column was clearly far from any cone. Replaced with proper
sphere tracing: when the SDF reports `min_d > 0` (outside every cone), step
down by `min_d` (with a `min_step = 0.25` floor to avoid stalls). For columns
that never enter a cone, this shrinks 16 steps × N_cones SDF samples to
typically 2-3 steps. For columns that do hit a cone, accuracy stays at
`±0.25` voxels (down from `±0.5` actually — small bonus precision win).

Also added an early-out inside the per-cone loop: as soon as `min_d <= 0.0`,
return immediately instead of continuing to scan the rest of the cones.

### Estimated savings

For a typical building placement on uneven terrain with cantilever columns
producing ~30-50 buttress cones:

- **`sdf_capped_cone` cost per sample**: ~40-50% cheaper (removed dead code +
  removed 2 sqrt + 3 div via precomputation).
- **`cone_top_in_column` total samples**: ~50-70% fewer SDF evaluations on
  apron columns far from any cone (sphere-trace early descent), and faster
  early-out on columns that hit a cone (no need to evaluate remaining
  cones).
- **End-to-end `build_support_hull` + apron-resolution time**: estimated
  **~50-65% reduction** for placements with cantilever buttresses. For
  fully-supported placements (no cones), only the dead-code removal applies
  (~10% on `sdf_capped_cone`, but it's not on the hot path then).

These are not measured — no microbench was added for this change because the
existing tests adequately verify correctness. Real-world impact will depend
on building size and how often `natural_floor_y_iso` returns `None` (which
is what triggers the cone-hull path). Expect the win to scale roughly with
`(terrace_size + 2*apron_radius)² × cantilever_fraction`.

### Risk

Low.
- `sdf_capped_cone` public signature unchanged.
- `sdf_compiled_cone` is a numerically-equivalent inlined version of the
  inside-height branch; same `nx`, `ny` formula, same below/above-cap
  formulas.
- All 7 existing `voxel-ffi` SDF + flatten tests pass, including
  `subvoxel_surface_lands_near_requested_y` which exercises the full
  flatten pipeline through `cone_top_in_column`.
- Sphere-trace floor of `0.25` voxels means worst-case overshoot is half
  what the old fixed-step had.

### Tests run
- `cargo test --lib -p voxel-ffi` → 59 passed, 0 failed.
- `cargo test -p voxel-ffi --lib flatten_sdf` → 3/3 pass.
- `cargo test -p voxel-ffi --lib sdf::` → 7/7 pass.

(Workspace-wide test had pre-existing `voxel-sleep` `FluidCell` errors from
other unstaged work in `voxel-fluid/src/cell.rs` — confirmed unrelated by
stashing the working tree, at which point voxel-sleep passes 47/47.)

## Other opportunities NOT taken (worth a follow-up)

Logged here so they aren't forgotten. Each is a real win but skipped to keep
this change small and reviewable.

1. **`find_support_rays` rebuilds Fibonacci directions per column.** For a
   4×4 terrace with 16 cantilever columns, that's 16 × 64 candidate
   directions generated from scratch (`max_candidates = n_rays * 4 = 64`).
   The directions are deterministic — precompute the filtered list once
   (lazy_static or `OnceCell<[Vec3; 16]>`) and reuse. Estimated ~5-10%
   savings on `build_support_hull` for buildings with many cantilever
   columns.

2. **`march_ray_for_surface` does a HashMap lookup per voxel step.** Each
   ray is up to 7 voxels long; with 16 cols × 16 candidate rays that's
   ~1800 hashmap lookups per placement. Cache the last
   `(cx, cy, cz) → &DensityField` and skip the `HashMap::get` when the next
   sample's chunk coord matches. Estimated ~30-50% of `march_ray_for_surface`
   time saved on long contiguous rays.

3. **`sample_natural_density` (sdf.rs:92) does 8 hashmap lookups per
   trilinear sample.** Same cache-last-chunk trick applies; in the common
   case where all 8 corners share a chunk, it goes from 8 lookups to 1.
   This is on a less-hot path right now (only `sample_natural_density` is
   currently called via `density_to_sdf` in helper code), but worth
   tracking if it ever moves into the per-column loop.

4. **Diagnostic file write in `flatten_terrace_sdf`** is already
   `#[cfg(debug_assertions)]`-gated — good, no action needed. Noted for
   future: if `cargo build --release` ever flips on debug assertions, this
   does an open/write/close on every placement.

5. **Worker `WorkerRequest::BuildingFlatten` calls `cfg.clone()`** under
   the read lock then takes the store write lock. The config clone is
   cheap-ish but it's been a source of contention before; fine for now,
   would be worth measuring if the building placement path ever shows up
   hot.

## Status
- Working tree dirty; **not committed, not pushed**, per scheduled-task
  instructions. Review at your convenience.
- To revert: `git checkout -- voxel-ffi/src/sdf.rs voxel-ffi/src/flatten_sdf.rs`.
- To commit (if you accept): the changes are scoped to those two files only;
  the surrounding pre-existing dirty state (cell.rs, terrain_ops.rs, etc.)
  is unrelated work already in your working tree.
