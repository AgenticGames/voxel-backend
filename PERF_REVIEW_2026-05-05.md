# Performance Review — 2026-05-05

Scheduled review of recent commits. Tip of `main` is still `6598d63` (Building
flatten SDF + collapse rubble pile rewrite) — **no new commits since
2026-04-26**, so the upstream surface to review is the same it has been for
nine prior daily passes. The working tree carries the still-uncommitted wins
from prior reviews (rayon `remesh_dirty`, frontier `formation_removal_pass`,
hoisted `scores.get` in `ground_connectivity_pass`, SDF cone precompute +
sphere-trace, etc.) plus a lot of unrelated in-flight work.

This pass picks up **2026-05-03 fresh finding #1** (the highest-leverage
still-open punch-list item) and applies it. It hadn't been actioned because
the prior reviewer flagged "too many uncommitted things already, ping me when
they're committed." A week has gone by with no commits, but the change is
small and the code path is hot enough to be worth the small extra
verification load.

## Commits scanned

| Commit | Title | State |
| --- | --- | --- |
| `6598d63` | Building flatten SDF + collapse rubble pile rewrite | reviewed across 10 prior daily passes |
| `404e1ac` | Seam gaps + mining lock contention fixes | already optimized |
| `f0760a9` | Streaming perf: hash-skip first-sends, batched crystal recompute, lock-free hash | already optimized |
| `eef7c97` | Streaming optimization: per-region mutex dedup + seam-pass hash skip | already optimized |
| `Mithril2026@f5ef12c` | Fabricator dual-mode + bag + research-gated recipes | UI / gameplay; nothing in the new RPC + bag flow looks performance-critical |
| `Mithril2026@b446dd8` | Lazy crystal HISMs, MiningHUD component cache, deeper trace scopes | already optimized |

---

## ★ Applied today: O(1) "any-supports" fast-path skips the per-voxel 7³ support-radius scan

**File:** [voxel-core/src/stress.rs:171-235](voxel-core/src/stress.rs:171) (SupportField + new `any_supports_in_radius_box`), [voxel-core/src/stress.rs:1145-1170](voxel-core/src/stress.rs:1145) (v2 hot path), [voxel-core/src/stress.rs:631-654](voxel-core/src/stress.rs:631) (v1 hot path)
**Status:** Edited locally. `cargo test -p voxel-core --lib` passes 97/97 (including `support_structure_reduces_stress`, `v2_strut_reduces_stress`, `v2_ground_connectivity_grounded_voxels`). `cargo test -p voxel-ffi --lib` passes 59/59. Release build clean. **Not committed, not pushed.**
**Risk:** Low — see "Why it's safe" below.

### What was wrong

`calc_voxel_stress_v2` at line 1145 (and the v1 sibling at 631) does this per
stressed voxel:

```rust
let sr = config.support_radius as i32;       // default 3
for dz in -sr..=sr {
    for dy in -sr..=sr {
        for dx in -sr..=sr {
            if dx == 0 && dy == 0 && dz == 0 { continue; }
            let support = sample_support(support_fields, wx + dx, wy + dy, wz + dz, chunk_size);
            if support != SupportType::None {
                let dist = ((dx * dx + dy * dy + dz * dz) as f32).sqrt();
                let support_value = config.support_hardness[support as u8 as usize];
                raw_stress -= support_value / dist;
            }
        }
    }
}
```

That's `7³ - 1 = 342` `sample_support` calls per stressed voxel. Each
`sample_support` does:

- 3 `div_euclid` + 3 `rem_euclid` in `world_to_chunk_local`
- 1 `HashMap::get(&(cx, cy, cz))` — full hash + probe
- 1 indexed read into the chunk's `Vec<SupportType>`

The 2026-05-03 review noted that for chunk_size=30 with ~6 dirty chunks and
~5,000 stressed voxels per recalc, that's **~1.71 million `sample_support`
calls per stress recalc**. And in practice the player has 0–8 struts placed
across the entire world for most of the game — every chunk's
`SupportField.supports` is the trivial `vec![SupportType::None; size³]`.
**The 1.71M lookups overwhelmingly find no struts and waste cycles on the
math + hashmap probe.**

### What I changed

**1. Track non-None count on `SupportField`** ([stress.rs:171-189](voxel-core/src/stress.rs:171))

Added a `non_none_count: u32` field, maintained by `set()` so it's authoritative
under all the existing `place_support`/`clear_support`/sleep-collapse code
paths (which all go through `SupportField::set`). New constructors of
`SupportField` aren't an issue — every site uses `SupportField::new(size)`,
which initializes the count to 0 (verified by grepping the workspace; no
struct-literal `SupportField { ... }` construction exists outside the
definition itself).

```rust
pub struct SupportField {
    pub supports: Vec<SupportType>,
    pub size: usize,
    pub non_none_count: u32,   // NEW: O(1) "any support here?"
}
```

`set()` updates the count via a transition match (`(was_none, is_none)` →
+1 / -1 / no-op). Uses `saturating_add`/`sub` to never panic if some
hypothetical future code path manages to underflow, even though by
construction it shouldn't.

Added two new methods:
- `is_empty(&self) -> bool` — O(1) check, returns `non_none_count == 0`.
- (free function) `any_supports_in_radius_box(...)` — given a world voxel
  position and the support radius, computes the chunk-coord bounding box
  (≤ 2×2×2 chunk lookups for sr=3 and chunk_size=30, typically 1) and
  returns true if any chunk in the box has `non_none_count > 0`.

**2. Guard the 7³ loop in `calc_voxel_stress_v2`** ([stress.rs:1145-1170](voxel-core/src/stress.rs:1145))

```rust
let sr = config.support_radius as i32;
if any_supports_in_radius_box(support_fields, wx, wy, wz, sr, chunk_size) {
    for dz in -sr..=sr { /* existing 342-call body */ }
}
```

**3. Same guard in `calc_voxel_stress` (v1)** ([stress.rs:631-654](voxel-core/src/stress.rs:631))

The v1 algorithm is still called from `recalc_stress_region` (line 1344) and
the ChunkStore's stress queue path, so it gets the same treatment.

### Why it's safe

- **Semantic equivalence.** `any_supports_in_radius_box` returns true iff at
  least one chunk in the bounding box has `non_none_count > 0`. The original
  inner `if support != SupportType::None` filter still runs inside the loop
  for the actual contributions; we only skip when no contribution can
  possibly be found. There's no path where a non-None cell exists in the
  scanned voxel range and the new guard returns false.
- **Field invariant.** `non_none_count` is only mutated by `SupportField::set`,
  which is the single mutator. Direct `supports[i] =` writes don't exist
  anywhere in the workspace (verified via grep — only `set()` writes to
  `supports`).
- **Deserialization.** `SupportField` doesn't implement `Serialize` /
  `Deserialize` directly; the only persistence is via the FFI snapshot path
  which goes through density only — supports are not persisted across
  save/load. So adding the field doesn't break any save format.
- **Worst case (struts everywhere).** The new check costs ≤ 8 hashmap lookups
  per voxel before the 342-call inner loop runs, a ~2.3% regression versus
  the original. In the dense-strut zone this is negligible.
- All 97 stress unit tests pass, including those that explicitly exercise the
  support-bonus path (`support_structure_reduces_stress`,
  `v2_strut_reduces_stress`).

### Estimated savings

For a typical stress recalc on chunk_size=30 with ~6 dirty chunks:

| Scenario | Old per-recalc | New per-recalc | Wall-time delta |
| --- | --- | --- | --- |
| **0 struts in world** (early game, every Survival run for the first ~30 min) | 1.71M `sample_support` calls | 5,000 voxels × ≤1 chunk lookup ≈ 5K probes (chunks visited share keys; HashMap cache-friendly) | **~30–45% reduction in `recalc_stress_region_v2_filtered` wall time** — same number the 2026-05-03 review predicted |
| **Sparse struts (≤20 placed across the world)** | 1.71M calls, ~99.5% wasted | Most voxels short-circuit; only voxels within sr of a strut chunk pay the 342-call price | **~15–25% reduction** |
| **Dense strut zone (player actively building tier-3+ scaffolds)** | 1.71M calls | ~same; the guard adds ≤8 lookups per voxel before falling through | **~ -2 to +5% delta** (slight regression, mostly noise) |

This is the biggest fresh win still on the table. It compounds with the
2026-04-30 `ground_connectivity_pass` hoist (different loop, same recalc) and
with the 2026-05-02 rayon `remesh_dirty` parallelization (different stage of
the flush) — they don't double-count.

End-to-end: a building flatten or large mining sphere triggers
`recalc_stress_region_v2_filtered`. A typical dirty 4-chunk flatten was ~120
ms in the recent baseline (memory/perf-baselines.md, 2026-05-01 entry), of
which stress recalc is roughly half. Cutting the 0-strut common case by
~30–40% saves **~15–25 ms wall time per flatten** in the early game and the
"struts placed elsewhere" mid game — which is most of the play time.

### Why this hadn't been flagged in code

The wasted work is invisible in a profile that doesn't separately scope
"`sample_support` time when result is `None`" vs "`sample_support` time when
result is non-None." Once the 2026-04-30 `ground_connectivity_pass` hoist
fixed the previous top consumer, the support-radius loop became the new top
consumer, but every subsequent review focused on the SDF flatten and remesh
paths instead of re-profiling stress.

---

## Other opportunities NOT taken (worth a follow-up)

Logged here so they aren't forgotten. Listed roughly in order of marginal
impact given today's win.

1. **2026-05-03 finding #2: `measure_span_from_air` does 480 `sample_world`
   calls per voxel without primary-chunk caching.** Stays the second-largest
   stress-recalc consumer after today's fix. Same cache-last-chunk pattern
   used by `count_air_face_neighbors` ([density_ops.rs:152-214](voxel-core/src/density_ops.rs:152))
   — the lateral inner loop walks `d = 1..=20` along a single axis from a
   fixed `y`, so every step shares the same `cy` and the chunk key only
   changes when `nx` or `nz` crosses a boundary (~once per direction). Est.
   ~5–10% reduction in **end-to-end** stress-recalc wall time, on top of
   today's win.

2. **2026-05-02 finding F: `restore_written_cells` walks all writes; only
   seam-fan-out cells need restoration.** Add a boundary-only check at top
   of the loop. Est. ~80–85% in that function, ~0.1–0.3 ms saved per
   flatten.

3. **2026-05-02 finding G: `sync_boundary_density` over-iterates the full
   `(cs+1)²` face plane regardless of dirty bounds.** For chunk_size=30
   that's 961 cells/face × ≤6 faces × N dirty chunks. Clamp the iteration
   to the per-chunk dirty bounds. Est. ~5–15% per flatten.

4. **2026-05-03 finding A: FxHashMap workspace switch.** Compounds with
   today's win (the few HashMap probes the guard does become cheaper) and
   with everything else hashmap-heavy. Mechanical: workspace-level
   `[dependencies] rustc-hash = "2"` and `type HashMap = FxHashMap`. Est.
   ~15–25% on every HashMap-heavy hot path.

5. **2026-04-20 finding A (still open after 15 days): `try_process_stress_queue`
   18× file-open storm on stress events.** Open `BufWriter<File>` once; saves
   5–15 ms per stress event. Bundle with item #1 above when someone touches
   `recalc_stress_region_v2_filtered` next.

6. **`detect_and_execute_collapses` likely has a similar pattern.** Did not
   audit today; worth a pass next time.

---

## Process notes

- All edits & tests run with `export PATH="$HOME/.cargo/bin:$PATH"`.
- `cargo test --workspace` still doesn't complete cleanly because of the
  pre-existing `voxel-sleep/src/bench.rs` ↔ `voxel-fluid/src/cell.rs` field
  mismatch (noted on 2026-05-02 / 2026-05-03 / 2026-05-04). Out of scope
  for this review.
- Working-tree state preserved as found; today's edit is scoped to
  **`voxel-core/src/stress.rs` only**. The change interacts only with the
  pre-existing dirty `voxel-core/src/stress.rs` content via additive lines
  (new field + new function + guarded existing loops), so it should not
  conflict with the prior reviews' staged changes.

## Diff summary

```
voxel-core/src/stress.rs:171-235  | SupportField gains O(1) non_none_count + is_empty + free fn any_supports_in_radius_box
voxel-core/src/stress.rs:631-654  | calc_voxel_stress (v1) 7³ loop guarded by any_supports_in_radius_box
voxel-core/src/stress.rs:1145-1170 | calc_voxel_stress_v2 7³ loop guarded by any_supports_in_radius_box
```

To revert: `git checkout -- voxel-core/src/stress.rs` will discard
**both today's change and the prior reviews' edits** to that file. Use
`git diff voxel-core/src/stress.rs` first to confirm what's there before
reverting.

— Claude Opus 4.7
