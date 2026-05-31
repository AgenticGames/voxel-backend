# Perf review — 2026-06-01 (run c) — scheduled "review latest commits" task

**Reviewer:** Claude (autonomous scheduled `daily-commit-performance-review` run).
**Scope:** latest commits on `main`. Runs *a* (`9f03c5e`, stress `ground_connectivity_pass`)
and *b* (`8e61222`, fluid `tick_chunk`) earlier today each took the last provably-safe win
in their hot path and declared it exhausted for autonomous work. So this run moved to a
**different, untouched subsystem**: the FFI mesh-conversion path, which every meshed chunk
crosses on its way to UE. `voxel-ffi/src/convert.rs` appears in **zero** prior perf commits.

## Finding: `bucket_mesh_by_material` allocated a fresh SipHash `HashMap` per material bucket and probed it per triangle-corner

`voxel-ffi/src/convert.rs :: bucket_mesh_by_material` regroups a `ConvertedMesh` so its
vertices/indices are contiguous per material (one `FfiSubmesh` section each), so UE can draw
one material per section. It runs once for **every chunk that gets meshed and bucketed** —
the worker generate path (`worker/generate.rs:946`), brush edits (`worker/brush.rs:732`),
seam stitching (`worker/seam.rs` ×3), store rebuilds (`store/mod.rs:396`), sleep-morph
(`worker/sleep_morph.rs` ×2), slabs and pile previews. It is squarely on the chunk-arrival
critical path (startup storm, zone-in, mining, save-load restore).

For each material bucket it re-emits that bucket's triangles into the new vertex arrays,
deduping shared vertices through a **per-bucket** remap table:

```rust
for (mat_id, triangles) in &buckets {
    let mut remap: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for &tri_idx in triangles {
        for corner in 0..3 {
            let orig_idx = mesh.indices[tri_idx as usize * 3 + corner];
            let new_idx = if let Some(&mapped) = remap.get(&orig_idx) { mapped }
                          else { /* push vertex; remap.insert(orig_idx, idx) */ };
            new_indices.push(new_idx);
        }
    }
}
```

Three problems, all on the hottest mesh path:

1. **A fresh `std::collections::HashMap` is allocated per material bucket** (≈ one alloc per
   distinct material in the chunk).
2. **`std` HashMap uses SipHash** — a deliberately slow, DoS-resistant hash — on a `u32` key,
   `get` + sometimes `insert` for **every one of the chunk's `3 × triangle_count`
   corners**. A cs=30 chunk surface is easily tens of thousands of triangles, so this is
   ~10⁵ SipHash probes per chunk.
3. The map has **no capacity hint**, so it rehashes/grows repeatedly as the bucket fills.

The original indices are **dense** (`0..vert_count`, they index straight into
`mesh.positions`), so a hash map is the wrong structure entirely.

## What I changed (1 file, voxel-ffi internal — no FFI/ABI surface)

Replaced the per-bucket `HashMap` with a **single flat remap array reused across all
buckets**, using an `epoch` tag per slot so each bucket "resets" the table for free:

```rust
let vert_count = mesh.positions.len();
let mut remap_idx:   Vec<u32> = vec![0u32; vert_count]; // new index for this slot
let mut remap_epoch: Vec<u32> = vec![0u32; vert_count]; // which bucket last wrote it
let mut epoch: u32 = 0;

for (mat_id, triangles) in &buckets {
    epoch += 1;                                   // every slot now reads as "absent"
    for &tri_idx in triangles {
        for corner in 0..3 {
            let oi = mesh.indices[tri_idx as usize * 3 + corner] as usize;
            let new_idx = if remap_epoch[oi] == epoch { remap_idx[oi] }
                          else { /* push vertex; remap_epoch[oi]=epoch; remap_idx[oi]=idx */ };
            new_indices.push(new_idx);
        }
    }
}
```

- **Two allocations total** (the two `Vec`s, once) instead of one `HashMap` per bucket.
- **Every `get`/`insert` becomes a raw array index** — no hashing at all.
- **No clearing pass between buckets**: bumping `epoch` invalidates every slot in O(1). Tags
  start at 0 and `epoch` starts at 0 → first bucket (epoch 1) sees all slots absent. With
  ≤256 materials, `epoch` cannot wrap.

**Behavior-preserving — provable.** Buckets are still iterated in the same `BTreeMap` (sorted
material) order, triangles within a bucket in the same order, and a new index is still
assigned as `new_positions.len()` at the moment of first encounter. Within one bucket the
epoch table maps each `orig_idx` to exactly one `new_idx` (identical dedup to the old map);
across buckets the epoch bump gives each bucket an independent table, so a vertex shared
across a material boundary is remapped independently in each — exactly as the per-bucket
`HashMap` did. Output `positions`/`normals`/`material_ids`/`indices`/`submeshes` are
bit-identical.

A new unit test (`bucket_preserves_triangles_and_partitions`) builds a multi-material mesh
with vertices shared across material boundaries (the cross-bucket-remap case) and asserts the
bucketed output reproduces the original triangle set exactly, keeps each vertex's attributes
paired, and that the submeshes partition the index buffer contiguously with every index
inside its own submesh's vertex span.

## MEASURED A/B (release, `git stash` of just this diff)

Microbench: a synthetic chunk-surface `ConvertedMesh` (13,824 verts, 48,668 tris, 6
materials, heavy cross-material vertex sharing). N=200 pre-cloned copies per round (the
per-call clone is **outside** the timed region so only `bucket_mesh_by_material` is measured),
6 rounds, best-of reported.

| | best µs / call |
|---|---:|
| baseline (per-bucket SipHash `HashMap`) | **2582.4** |
| optimized (reused epoch-tagged flat array) | **610.5** |
| **delta** | **−76.4% (≈4.2× faster)** |

Ranges did not overlap across rounds (baseline 2582–2665, optimized 610–695).

**Honest scoping of the number:**
- The **−76%** is of `bucket_mesh_by_material` **in isolation**, not of a whole frame or a
  whole chunk build. Bucketing is one stage of chunk meshing (which also runs DC solve,
  hermite extract, `convert_mesh_to_ue_scaled`); this trims that one stage by ~3/4.
- The relative win is roughly **scale-independent** (it's per-corner SipHash → array index),
  so smaller real chunk meshes see a similar **percentage** even though the absolute µs is
  smaller. Very tiny meshes (few triangles, 1–2 materials) see slightly less because the flat
  array's `vert_count`-sized init is a larger share — but those calls are already cheap.
- Where it lands: **lower worker-thread CPU during the chunk-arrival storm** (startup,
  zone-in, save-load restore) and on **every brush/mining edit and seam re-stitch**. It does
  not change steady-state frame time when no chunks are being (re)meshed.

## Validation
- `cargo test --workspace` green: voxel-core 101, voxel-ffi **126** (was 125 + the new
  bucketing test), voxel-fluid 90, voxel-sleep 106, 0 failures.
- voxel-ffi-internal only — **no FFI symbol, struct layout, or ABI change**. Clean
  `git revert` if you disagree.

## Other commits reviewed — no action taken
- Runs *a*/*b* targets (stress flood, fluid `tick_chunk`) confirmed still in place; their
  remaining levers are **behavioral** (algorithmic early-outs) and remain reserved for a
  human-reviewed pass, not autonomous work.
- `convert_mesh_to_ue_scaled` itself: already lean (single pass, pre-sized `Vec`s, pure
  arithmetic per vertex/triangle) — left alone.
- The `buckets` `BTreeMap` build (one `Vec` per material + a push per triangle) is cheap
  relative to the remap and was left unchanged to keep the diff minimal and the win isolated.
