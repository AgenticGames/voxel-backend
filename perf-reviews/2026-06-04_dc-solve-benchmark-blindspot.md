# Performance Review — Recent Commits (DC solve)

**Date:** 2026-06-04 (scheduled review)
**Scope:** Recent `main` perf commits, focused on the dual-contouring meshing path
**Verdict:** The hot paths touched by recent commits are already tightly optimized — but I found a
**measurement blind spot** that has been hiding the single biggest CPU cost in chunk meshing.
**Action landed:** Added a representative benchmark that exposes the true cost (safe, no engine
behaviour change). **Action requested:** review the recommended real win below — it changes mesh
vertices at the f32-ulp level, so I did **not** auto-apply it.

---

## Commits reviewed

Recent `main` is a run of well-measured micro-optimizations:

| SHA | Title | Measured |
|---|---|---|
| 603ace2 | cache chunk SupportField across the strut load cube | ~50% off accumulate_strut_load_at_voxel |
| 11f27d8 | epsilon-bounded weld neighbor scan | ~41% off weld_vertices |
| 832ceb1 | bucket sync_region_boundary_densities apply passes | ~51% off the call |
| e1175fd | u32-indirection qefs accumulator in solve_dc_vertices | ~41% off the call |
| e1679be | flat sentinel vertex-map in DC generate_mesh | ~69% off the call |
| 89e61222| flat epoch-tagged remap in bucket_mesh_by_material | ~76% off the call |

These are solid. The big, obvious wins on the worldgen / mesh / stress / fluid hot paths have
largely been harvested; what remains in those functions is mostly irreducible arithmetic.

---

## 🔍 Finding: `bench_dc_solve` under-measures real meshing cost by ~3.4×

`solve_dc_vertices` (the QEF vertex solve, voxel-core) is the heaviest per-chunk meshing step. Its
microbenchmark, `bench_dc_solve`, builds its hermite test data with **purely axis-aligned normals**
(`Vec3::X / Y / Z`):

```rust
normal: Vec3::X * if a < b { 1.0 } else { -1.0 }
```

With axis-aligned normals, every cell's `ATA = Σ nᵢ⊗nᵢ` matrix is **diagonal**. The Jacobi
eigensolver inside `QefData::solve` therefore hits its convergence check on the very first
iteration and does **zero rotations** — the SVD is essentially free.

Real worldgen does **not** feed axis-aligned normals. Hermite extraction stores the **gradient** of
the smooth density field as each edge normal (see `voxel-gen` hermite extraction); those point in
arbitrary directions, producing dense `ATA` matrices that take several Jacobi sweeps to diagonalize.

I measured this directly by instrumenting the iteration counter (instrumentation reverted; not committed):

| Hermite normals | Jacobi iters / cell | `solve_dc_vertices` cost (cs=30, 2604 surface cells) |
|---|---|---|
| Axis-aligned (`bench_dc_solve`) | **0.00** | **~205 µs/call** |
| Gradient (realistic) | **5.99** | **~686 µs/call** |

**The real cost is ~3.4× what the existing bench reports**, and ~70% of it is the Jacobi
eigensolver that the existing bench never exercises. Any perf work (including these daily reviews)
guided only by `bench_dc_solve` is blind to the dominant cost.

### Landed (this commit, safe — no engine change)

`voxel-core/examples/bench_dc_solve_realistic.rs` — identical to `bench_dc_solve` but derives each
edge normal from the analytic gradient of the terrain SDF, reproducing the dense-`ATA`,
~6-Jacobi-iteration case that real chunks hit. This makes the true bottleneck visible and gives a
stable A/B harness for the recommendation below. Output (`sink`) matches `bench_dc_solve` so it
exercises the same surface; only the timing differs.

```
[axis-aligned ]  BEST: ~205 µs/call
[gradient norm]  BEST: ~686 µs/call
```

---

## Recommended real win (NOT applied — needs your review)

**Replace the iterative Jacobi eigensolver with a closed-form 3×3 symmetric eigensolver.**

- **Where:** `voxel-core/src/dual_contouring/qef.rs` — `jacobi_eigen_3x3` / `jacobi_svd_3x3`,
  called once per surface cell via `solve_clamped → solve`.
- **Why it's the right target:** ~70% of `solve_dc_vertices` is this eigensolver, and it runs
  ~6 sweeps × 2604 cells = ~15.6K rotations per chunk, each with 2 `sqrt` + 1 `div`. A closed-form
  symmetric eigensolver (characteristic-cubic eigenvalues + eigenvectors via cross products, e.g.
  Eberly's "robust eigensolver for 3×3 symmetric matrices") computes the same decomposition with no
  iteration loop.
- **Estimated savings:** **~30–50% off `solve_dc_vertices`** on realistic data (≈ **200–340 µs/chunk**
  at cs=30), i.e. a meaningful cut to cold chunk-load / re-mesh time. Per-frame mining re-mesh
  benefits too.
- **⚠️ Why I did not auto-apply it:** a closed-form solver produces a numerically *different*
  (though equally/more accurate) decomposition, so DC vertices shift at the f32-ulp level. That is
  almost certainly invisible (sub-micrometer in a ×40 world) and harmless (mesh is recomputed, not a
  persisted seed), but it is a behaviour change to rendered geometry and a hand-written eigensolver
  has degenerate-eigenvalue edge cases. That deserves a human eye + a vertex-delta check before it
  lands — not an unattended commit. The new realistic bench + the existing QEF unit tests give you
  the A/B and correctness harness to evaluate it quickly.

---

## Candidates tested and rejected (so they don't get re-investigated)

All three were implemented, benchmarked, confirmed **bit-identical** (sink/acc unchanged), and
**reverted** because the compiler already handles them — zero measurable gain:

1. **Defer the 4 `hash3` calls in `Simplex3D::sample` into their `t >= 0` guards** (skip the ~2 of 4
   corner hashes that don't contribute). No change: ~33.9 ns/sample before and after. LLVM already
   sinks the pure loads.
2. **Unroll the `for r in 0..3 { if r != p && r != q }` element-update loop in `jacobi_eigen_3x3`**
   to a direct `r = 3 - p - q`. No change on *either* bench (and on the axis-aligned bench Jacobi
   runs 0 iterations anyway). Compiler already unrolls it.
3. **Thread-local epoch-tagged scratch for `cell_to_slot`** in `solve_dc_vertices` (eliminate the
   per-call ~108 KB `vec![u32::MAX; total]` memset). No change: the memset is cache-resident and the
   allocator reuses the freed block; the cost is the per-cell solve arithmetic, not the allocation.

Lesson for future runs: the voxel-backend hot paths have been swept thoroughly. The remaining wins
are **algorithmic** (like the eigensolver above), not micro-optimizations — and the benchmarks must
use representative inputs or they'll report a fixed function as "already fast".
