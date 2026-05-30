# Perf review — 2026-05-31 (b) — scheduled "review latest commits" task

**Reviewer:** Claude (autonomous scheduled run, second pass of the day). **Scope:** latest commits on
`main`, headed by `c50532e` *"drop TEMP VFX diagnostics from hot paths"* (the earlier 03:46 run) and
the feature commit beneath it, `89bd270` *"Sleep montage backend: tri-state solidity,
montage-protected density, worker heartbeat."*

The 03:46 pass already removed the committed `TEMP VFX diagnostic` file-I/O. This pass looked for a
**missed opportunity** in the *non-diagnostic* code those same commits added — and found one on the
exact path `89bd270` newly made hot.

## Finding: loop-invariant HashMap lookups in the stress recalc inner loop

Commit `89bd270` added an **isolated VFX-only stress compute per generated chunk**
([voxel-ffi/src/worker/generate.rs](voxel-ffi/src/worker/generate.rs)) so crack/dust overlays light
up the instant a chunk streams in (instead of only after the player mines nearby). That block calls
`recalc_stress_region_v2` for **every generated chunk** — so this function, previously only on the
mine/sleep path, is now also on the **initial-load / zone-stream / save-load storm**.

Inside that function ([voxel-core/src/stress/calc_v2.rs](voxel-core/src/stress/calc_v2.rs)), Pass 2
walks the full `gs³` voxel grid (chunk_size+1 ≈ **29,791 voxels** at the live 30³ override). For each
voxel it was doing **HashMap lookups keyed by the chunk coord `(cx,cy,cz)` — which is constant for
the entire triple-nested loop**:

- `support_scores.get(&(cx,cy,cz))` — once per solid voxel,
- `stress_fields.get_mut(&(cx,cy,cz))` — once per voxel (air, interior-skip, and surface paths all hit it),
- `stress_fields.get(&(cx,cy,cz))` — again, to read the painted overlay before the set.

The map uses `std::collections::HashMap` (default **SipHash**); hashing a 12-byte tuple key is tens of
ns, and it was being paid **~30k–90k times per chunk** purely to re-find the same two field entries.
The `_with_load_decay` sibling (the live mining/sleep hot path) had the identical pattern **plus** a
duplicate painted read (`painted_now` and `painted` computed the same value via two separate lookups).

## What I changed
Hoisted the two per-chunk lookups out of the `gs³` loop in **both** v2 entry points
(`recalc_stress_region_v2_filtered` and `recalc_stress_region_v2_with_load_decay`): fetch
`support_scores.get(...)` and `stress_fields.get_mut(...)` **once per chunk**, then reborrow inside
the loop via `as_deref()/as_deref_mut()`. Collapsed the duplicate `painted_now`/`painted` reads in
`_with_load_decay` into one. **Strictly less work, never more** — and behavior-preserving (the values
read/written are byte-identical; the chunk key was always constant across the loop).

`cargo test --workspace`: **voxel-core 101, voxel-ffi 125, voxel-fluid 90, voxel-sleep 106, … all
green, 0 failures.** Release `voxel_ffi.dll` rebuilt and synced to both UE locations (Binaries/Win64
+ Plugins/VoxelBridge/ThirdParty). **Editor restart required to load the new DLL.**

## Estimated savings (MEASURED, A/B microbench)
Timed `recalc_stress_region_v2` on a mixed tunnel-carved chunk (air + floor/ceiling surfaces +
grounded interior + true stress voxels), 4000 iters, release build, `git stash` A/B of just this
diff:

| run | old (per-voxel lookups) | new (hoisted) | delta |
|-----|------------------------:|--------------:|------:|
| 1   | 6099 µs/chunk           | 5828 µs/chunk | **−4.5%** |
| 2   | 6072 µs/chunk           | 5886 µs/chunk | **−3.1%** |

→ **~3–4.5% off a full `recalc_stress_region_v2` call.** Important caveat: that microbench's wall-time
is *dominated by `ground_connectivity_pass`* (the ground-flood, which my change does **not** touch),
so the **3–4.5% is diluted** — the saving on Pass 2 itself (the loop I edited) is a substantially
larger fraction of *that pass*. Net effect:

- **Per-generated-chunk VFX precompute** (`89bd270`'s new path, worker thread): a few % off each
  chunk's stress precompute during the load/stream storm — multiplied across the hundreds of chunks
  that generate at startup/zone-in. Reduces worker CPU during the storm, not steady frame-time.
- **Live mining + deep-sleep stress passes** (`_with_load_decay`): same few-% trim on every recalc,
  steady-state. Plus one fewer painted-overlay lookup per surface voxel there.

Modest but free: a pure loop-invariant hoist with zero behavioral risk, on a function the latest
commit just promoted onto the streaming hot path.

## Other commits reviewed — no action needed
- `89bd270`'s **spin-retry** `query_surface` / `is_solid_at_ue` (6×100µs cap) and the new
  `solidity_at_ue` tri-state helper are already lean (one chunk lookup + one voxel read).
- The **worker heartbeat** (`heartbeat.rs`) is silent in the common case and stamps only around
  request handlers (`now_ms()` is a cheap user-mode clock read on Win) — negligible.
- The VFX block's unconditional `stress_config.clone()` per chunk is genuinely tiny (small POD
  config) — left alone; not worth the read-lock reorder.
- Fluid sim, OrePaint spatial-hash, FluidChunkCache/ChunkSampleCache hoists from prior weeks remain
  well-targeted.

## Caveat for your review
Pure mechanical hoist — if you diff it, every read/write inside the loop resolves to the same field
entry it did before; I only stopped re-finding that entry 30k× per chunk. Revert is a clean
`git revert <this-commit>` with no FFI/ABI implications (voxel-core internal only).
