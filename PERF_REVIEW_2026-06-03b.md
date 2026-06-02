# Perf review — 2026-06-03 (run b) — scheduled "review latest commits" task

**Reviewer:** Claude (autonomous scheduled run). **Scope:** latest commits on `main`, headed by
this morning's run a (`0b1374e`, `perf(voxel-gen): hoist generate_chunk_seam_quads's per-cell
neighbor probe`). That commit's writeup flagged the **next lever** explicitly:

> Same file, `sync_region_boundary_densities` … the `density_fields.contains_key` +
> double-`density_fields[&key]` index pattern … re-hashes the same key 3×. Lower priority
> (region-build path), same shape, clean follow-up.

This run takes that lever — and on inspection it is a *bigger* win than "3× one key": the repeated
lookups sit **inside the two inner cell-scan loops**, so the same two loop-invariant keys are
re-hashed `O((gs+1)²)` times per neighbor, not 3 times total.

## The target: `sync_region_boundary_densities` (`voxel-gen/src/region_gen.rs`)

Runs **once per region build** on the worker generate path
([`voxel-ffi/src/worker/generate.rs:234`](voxel-ffi/src/worker/generate.rs)) plus the two
`region_gen` region builders. It syncs overlapping boundary voxels between adjacent chunks so
chunk-local writes (formations/pools/zones) don't leave seam gaps. Two passes, both with the same
defect:

**Pass 1 — min-rule sync.** For each chunk key, for each of 13 forward neighbor offsets, it
re-resolved `density_fields[&(cx,cy,cz)]` and `density_fields[&neighbor]` **inside the boundary-pair
triple loop** (line 808–809) — and gated entry with `contains_key(&neighbor)` immediately followed
by `[&neighbor]` (two hashes of the same key). For the 3 **face** offsets the boundary-pair loop is
`(gs+1)²` iterations, so each face neighbor cost `2·(gs+1)²` SipHash probes of 12-byte keys
(`≈ 2·961 = 1922` at the live cs=30) for lookups whose result never changes across the loop.

**Pass 2 — gradient-blend (slate-cube cliff fix).** For each key, for each of 3 face offsets, the
`for u in 0..=gs { for v in 0..=gs { … }}` scan re-borrowed `&density_fields[&(cx,cy,cz)]` and
`&density_fields[&neighbor]` **every (u,v) iteration** (lines 860–861/871–872/882–883) — another
`2·(gs+1)²` probes per face neighbor.

For an interior chunk with all neighbors present that is `≈ 2 · 3 · 961 · 2 ≈ 11.5 k` redundant
SipHash probes **per chunk** across the two passes, all resolving to at most two distinct entries.

## What shipped — hoist the loop-invariant lookups

Both passes: resolve `f_a` (the key's own field) **once per key** and `f_b` (the neighbor) **once
per neighbor** via a single `.get` (which also replaces the `contains_key` + `[&neighbor]`
double-hash), then index those borrows inside the cell loops.

```rust
for &(cx, cy, cz) in &keys {
    let f_a = &density_fields[&(cx, cy, cz)];          // once per key
    for &(dx, dy, dz) in &offsets {
        let Some(f_b) = density_fields.get(&neighbor) else { continue };  // one hash, was two
        … f_a.get(ax,ay,az) … f_b.get(bx,by,bz) …      // array math, no hashing
```

`2·(gs+1)²` hashes per neighbor → **2 hashes per neighbor** (Pass 1: 1 for `f_b` + amortized
`f_a`; Pass 2 the same). The cell loops keep doing the identical reads, now off a borrow instead of
a fresh probe.

### Behavior-preserving — proved bit-identical

- The hoist is textbook loop-invariant code motion of **read-only** lookups. The gather loops never
  mutate `density_fields` (all writes are deferred to `updates` / `grad_updates` and applied in a
  separate pass), so prefetching the two borrows changes *when* the lookups happen, not their values
  or order. `let Some(f_b) = …get(&neighbor) else { continue }` is exactly the old
  `if !contains_key { continue }` + `[&neighbor]`.
- **New regression test** `boundary_sync_hoist_is_bit_identical` (voxel-gen) proves it empirically:
  it runs the shipped (hoisted) function and an inlined replica of the **pre-hoist** body on two
  clones of the same region and asserts every sample's `density.to_bits()` and `material` match.
  `HashMap::clone` preserves iteration order within a process, so both clones visit `keys()` in the
  same order — making the function's (intrinsically order-dependent, at shared edge/corner cells)
  output directly comparable. Green.
- `cargo test --workspace`: **voxel-gen 107** (106 + this test)**, voxel-core 103, voxel-ffi 126,
  voxel-sleep 57, voxel-world-memory 53 — 0 failures.**

#### Note: a cross-process checksum can NOT prove this (and why)

The bench prints a post-sync checksum, but it differs run-to-run **even for the same binary** —
`sync_region_boundary_densities` iterates `density_fields.keys()` unsorted, and shared edge/corner
cells receive several updates whose apply order follows that per-process-random iteration order.
That nondeterminism is **pre-existing** in the function (the in-repo memory even flags "HashMap
iteration is nondeterministic"), independent of this change, and is *not* something this perf pass
touches (sorting keys would change output for tie cells = a behavior change, out of scope). Hence
the in-process clone test above, not a checksum A/B, is the bit-identity proof.

## Estimated savings (MEASURED, A/B microbench)

`git stash` A/B of just the `region_gen.rs` diff, **release** build. Bench
([`voxel-gen/examples/bench_boundary_sync.rs`](voxel-gen/examples/bench_boundary_sync.rs)): a 3³
block of adjacent chunks from a sinusoidal terrain sheet at cs=30 (the live UE override) so the
center chunk has all 26 neighbors present and boundary cells straddle the cliff thresholds. Each
timed iteration runs one `sync_region_boundary_densities` on a freshly-cloned pristine region;
clone cost is measured separately and subtracted. 200 calls × 6 rounds, best-of:

| side | best µs/call (incl clone) | clone µs/call | **NET µs/call** |
|------|--------------------------:|--------------:|----------------:|
| baseline (per-cell `density_fields[&key]`) | 9174.3 | 1133.3 | **8041.0** |
| **hoisted (SHIPPED)** | 5516.6 | 1044.2 | **4472.4** |
| **delta** | | | **−44.4 % (≈1.80× faster)** |

Rounds are wholly non-overlapping (baseline incl-clone 9174–9612, shipped 5516–5671), so the win is
real, not noise.

**Honest scoping of the number:**
- The **−44 %** is of `sync_region_boundary_densities` **in isolation**. This is a **per-region**
  (not per-frame) pass — it runs once when a region's chunks are first generated, not on steady-state
  frames and not on every brush/mining edit. So unlike run a's seam hoist (per-chunk re-mesh path),
  this lowers worker CPU specifically during the **region-generation storm** (world startup, zone-in,
  save-load restore) — the same storm, one stage earlier in the pipeline (density sync precedes the
  per-chunk hermite → solve → mesh → seam stages the 06-01/02/03a commits trimmed).
- The win scales with **present-neighbor count and grid size**: the `(gs+1)²` factor means cs=30
  (961 cells/face) saves far more in absolute terms than a small chunk; a region edge chunk with
  fewer present neighbors saves proportionally less. The 3³-interior measurement is the high end
  (all neighbors present); a real region averages lower per chunk but pays it across every chunk.
- The remaining ~4.5 µs is genuine work: `(gs+1)²` `avg_boundary` calls, the cliff comparisons, and
  the deferred `get_mut` apply passes (which still hash per update — but those are *necessary*
  mutable writes, not redundant reads, and far fewer than the gather probes). This is the last
  *structural* HashMap-probe win in this function.

Zero-risk, zero-ABI (voxel-gen internal, no FFI surface), clean `git revert`.

## The lesson for the next pass

The dense/spatial-key-HashMap family that the 06-01/02/03 run swept has a recurring sub-shape:
**a HashMap lookup whose key is invariant across an inner loop, left inside that loop.** Run a's
seam fix was the bounded-neighborhood variant (≤8 distinct keys → fixed array); this one is the
purest variant (the key literally does not change across `(gs+1)²` iterations → hoist to a `let`).
Grep for `map[&k]` / `map.get(&k)` / `map.contains_key(&k)` *inside* `for` bodies where `k` is built
from the loop's outer variables, not the inner ones.

## Next lever flagged (not taken)

The **apply** passes of this same function (lines ~819 and ~915) still call
`density_fields.get_mut(&key)` once per queued update — and for a shared edge/corner cell that key
repeats across several updates, re-hashing it each time. These are *writes* so they can't be hoisted
as cleanly as the reads, but grouping `updates`/`grad_updates` by key (sort or bucket) before the
apply would resolve each field once and write a run of cells through one `get_mut`. Smaller win than
the gather hoist (the apply count ≈ boundary-cell count, not `(gs+1)²·neighbors`), and it would also
*incidentally* make the output deterministic if combined with a stable key sort — but that crosses
into behavior change, so it needs its own scoped pass. Lower priority.

## Other commits reviewed — no action needed
- Run a's seam hoist (`0b1374e`) and the 06-01/02 DC trio (`89e67e6`/`e1679be`/`e1175fd`) confirmed
  still in place and intact.
- `axis_boundary_pairs` / `avg_boundary` are small pure helpers — already minimal.
- The deferred-update design (gather then apply) is correct and worth keeping; it is *why* the read
  hoist is provably safe (no aliasing of mutable + immutable borrows during the scan).
