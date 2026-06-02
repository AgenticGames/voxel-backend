# Perf review — 2026-06-03 (run c) — scheduled "review latest commits" task

**Reviewer:** Claude (autonomous scheduled run, Opus 4.8). **Scope:** latest commits on `main`,
headed by today's run b (`b262b0e`, `perf(voxel-gen): hoist sync_region_boundary_densities'
per-cell key re-hashes`). That commit's writeup flagged the **next lever** explicitly:

> Next lever flagged (not taken): the apply passes re-hash repeated keys in `get_mut`; grouping
> updates by key would resolve each field once (smaller win, writes not reads).

This run takes that lever. On inspection it is **not** a "smaller win" — it is roughly the same
magnitude as run b (~2× on the function), because the apply passes were the *other half* of the
cost and they re-hash far more aggressively than the gather loop ever did.

## The target: the two **apply** passes of `sync_region_boundary_densities`

`voxel-gen/src/region_gen.rs`. Runs **once per region build** on the worker generate path
([`voxel-ffi/src/worker/generate.rs`](voxel-ffi/src/worker/generate.rs)) plus the two `region_gen`
region builders (startup, zone-in, save-load restore). Runs **b/01/02/03a** swept the *gather* side
(hoisting `f_a`/`f_b` borrows out of the cell loops). They left the *apply* side untouched:

```rust
// Pass 1 (min-rule):                       // Pass 2 (gradient-blend):
for (key, x, y, z, d, m) in updates {       for (key, x, y, z, d) in grad_updates {
    if let Some(field) =                        if let Some(field) =
        density_fields.get_mut(&key) {             density_fields.get_mut(&key) {
        ...                                        ...
```

**The defect.** Each `updates` entry carries its own `(i32,i32,i32)` key, and the gather loop pushes
**two entries per boundary pair** (the `(cx,cy,cz)` side and the `neighbor` side). So a field that
participates in many pairs has its 12-byte key stored — and then re-hashed by `get_mut` at apply —
once *per sample write*, not once per field. At the live `cs=30` override a single all-neighbours
chunk receives on the order of `2 · (3·(gs+1)² faces + 6·(gs+1) edges + 4 corners) ≈ 6.1k` writes,
each costing a fresh SipHash of the same handful of distinct keys. Across a 3³ region that is
~10⁵ redundant `get_mut` probes resolving to just 27 distinct fields. Pass 2 has the identical
shape (one `get_mut` SipHash per gradient write). On top of the hashing, the flat `updates` vec
grows to multiple **megabytes** (~166k × 44-byte tuples), churning the allocator.

## The fix: stable per-field bucketing

Build a `key -> dense index` map once (one hash per field), then route every write to a per-field
`Vec` bucket **during gather** (the source key's index is the outer-loop counter; the neighbor's
index is one `key_index` lookup per `(key, offset)`, not per pair). The apply pass then walks the
buckets and calls `get_mut` **exactly once per field**:

```rust
let key_index: HashMap<_, usize> = keys.iter().enumerate().map(|(i,&k)|(k,i)).collect();
let mut buckets: Vec<Vec<(usize,usize,usize,f32,Material)>> = vec![Vec::new(); keys.len()];
// gather: buckets[ki].push(...); buckets[ni].push(...);   (ni = key_index[&neighbor])
for (ki, &key) in keys.iter().enumerate() {
    let field = density_fields.get_mut(&key).expect("key from keys()");   // ONE probe / field
    for (x,y,z,d,m) in buckets[ki].drain(..) { ... }
}
```

`get_mut` SipHash count per region: `~6.1k · N  →  N` (one per field), plus `~14·N` cheap
gather-side index lookups. The flat multi-MB tuple vec is gone, replaced by N small buckets.

### Why it is bit-identical (provable)

Bucketing is a **stable partition by target key**. Every push targeting a given field is appended
in the *same outer-loop traversal order* that previously filled the flat `updates` vec, so for any
single cell the sequence of writes — and therefore the last-writer-wins result — is byte-for-byte
unchanged. Cross-field apply order changes, but distinct fields never alias, so it is invisible.

This is exactly the invariant the **existing** regression test
`boundary_sync_hoist_is_bit_identical` already asserts: it runs the shipped function against an
inlined pre-hoist replica (`sync_region_boundary_densities_prehoist`) on two clones of one region
and checks every sample's `density.to_bits()` + material. That test still passes against the
bucketed version, so the new code is provably bit-identical to the *original pre-hoist* baseline
too — not just to run b. `cargo test --workspace` green (voxel-gen 107, voxel-core 103,
voxel-ffi 126, voxel-sleep 57, voxel-world-memory 53 — 0 failures).

## Measured A/B (release, git-stash of just this diff)

`bench_boundary_sync`, 3³ block of `cs=30` sinusoidal-terrain chunks (center has all 26 neighbours),
one sync per timed call on a fresh clone with clone cost subtracted, 200 calls × 6 rounds, best-of.
Two runs each side:

| | NET run 1 | NET run 2 | clone overhead |
|---|---|---|---|
| **baseline** (flat `updates`/`grad_updates` vecs) | 4392 µs/call | 5203 µs/call | ~1020 µs |
| **optimized** (per-field buckets) | 2130 µs/call | 2336 µs/call | ~258 µs |

**Best-of-best: 4392 → 2130 µs/call NET = −51.5% (≈ 2.06× faster).**

Note the clone-overhead column: it is ~4× lower for the optimized binary and reproduces across both
runs. That is a *real secondary effect*, not noise — the baseline's multi-megabyte flat-vec
allocation churns the process allocator, so even the (sync-free) clone loop pays for it afterward.
The −51% NET figure therefore *understates* the total saving; it counts only the function's marginal
cost, not the allocator pressure the bucketing also removes.

## Scope / impact

- This is a **per-region** pass (region-generation storm: startup, zone-in, save-load restore), not
  per-frame and not per-edit. It lowers worker CPU during those bursts, one stage before the
  per-chunk DC stages the 06-01/02/03a commits trimmed.
- Scales with present-neighbour count and `(gs+1)²`. The win grows with chunk size and with how
  fully a region's neighbourhood is populated.
- Combined with run b, `sync_region_boundary_densities` is now ~3.7× faster than the start-of-day
  `main` (8041 → 4472 → 2130 µs/call NET on the run-b/run-c bench).

## Next lever flagged (not taken)

The `buckets: Vec<Vec<_>>` allocates N independent growable vecs per call. For large regions a
single flat **CSR-style** layout (one `Vec<usize>` of per-field counts → prefix-sum offsets → one
contiguous payload vec, two passes) would cut N small allocations to 2 and improve apply locality.
Likely a small win on top of this one, and slightly more code; left for a follow-up run. The other
remaining apply-side cost is `DensityField::get_mut(x,y,z)`'s index recompute per sample — already
cheap (no hashing), low priority.
