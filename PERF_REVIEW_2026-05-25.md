# Perf review — 2026-05-25 (scheduled daily pass)

HEAD has advanced to `0a25bff` ("Voxel topology probe + POI kinds for
placement validators"). **Critical finding first**, then the perf
opportunity I would have shipped if the workspace built cleanly.

## ⚠️ Critical — `voxel-ffi` does NOT compile at HEAD

`cargo build -p voxel-ffi` fails on `main` with 5 errors. Commit
`0a25bff`'s message advertises "POI scanner extended with three topology
kinds (poi_scanner.rs + poi_tracker.rs)" but the diff only modified
`poi_tracker.rs`. The new references in
[voxel-ffi/src/poi_tracker.rs:33-36](voxel-ffi/src/poi_tracker.rs:33)
and [:263-266](voxel-ffi/src/poi_tracker.rs:263) have no backing
declarations in
[voxel-ffi/src/poi_scanner.rs](voxel-ffi/src/poi_scanner.rs):

```
error[E0432]: unresolved imports `crate::poi_scanner::count_topology_votes_cross_chunk`,
                                 `crate::poi_scanner::TopologyVotes`
   --> voxel-ffi\src\poi_tracker.rs:34:51

error[E0061]: this function takes 3 arguments but 6 arguments were supplied
   --> voxel-ffi\src\poi_tracker.rs:263:29   (score_from_votes called with 6 args)

error[E0599]: no variant or associated item named `CeilingDome` found for enum `PoiKind`
   --> voxel-ffi\src\poi_tracker.rs:364:26
error[E0599]: no variant or associated item named `Chokepoint` ...
error[E0599]: no variant or associated item named `WallNiche`  ...
```

What's missing from `poi_scanner.rs`:

| Symbol                                    | Used at                                                          | Suggested signature                                                              |
|-------------------------------------------|------------------------------------------------------------------|----------------------------------------------------------------------------------|
| `PoiKind::CeilingDome` (= 4)              | `poi_tracker.rs:364`                                             | Add to the existing `#[repr(u8)] enum PoiKind`.                                  |
| `PoiKind::Chokepoint` (= 5)               | `poi_tracker.rs:364`                                             | Same.                                                                            |
| `PoiKind::WallNiche` (= 6)                | `poi_tracker.rs:364`                                             | Same.                                                                            |
| `struct TopologyVotes { dome_count, choke_count, niche_count, dome_pos_sum, choke_pos_sum, niche_pos_sum, ... }` | `poi_tracker.rs:232, 247, 252, 262, 268` | Needs `Default`, `Copy`, and `centroid_for(kind, chunk_size) -> IVec3`. |
| `fn count_topology_votes_cross_chunk(&ChunkStore, (i32,i32,i32), usize) -> TopologyVotes` | `poi_tracker.rs:245` | Per the commit msg, scans the chunk + cross-chunk neighbor reads for CeilingDome / Chokepoint / WallNiche signatures. |
| `score_from_votes(lava, water, stress, dome, choke, niche) -> ChunkScoreBreakdown` | `poi_tracker.rs:263` | Extend the 3-arg version to also score the three topology kinds; add `dome / choke / niche: f32` to `ChunkScoreBreakdown` and pick them up in `best()`. |

**No new tests fire today, no new perf code lands** — because every
edit in `voxel-ffi` would propagate through the broken compile. Once
the topology-vote scanner lands, the perf patch below applies cleanly.

## Today's intended perf target — `surface_probe.rs` gradient + sample cache

**File:** [voxel-ffi/src/surface_probe.rs:113-197](voxel-ffi/src/surface_probe.rs:113)
(new module added in `0a25bff`).

`probe_surface` powers spider-nest / wasp-hive placement validators
(per the memory note and the file's own header). On the UE side it's
called by:

- `voxel_query_surface` FFI per placement candidate (~tens per cluster
  spawn event, currently ~1 / 15 s of ambient gameplay).
- `AEnemyBase::CheckAndFixOutsideCave` 1 Hz tick across every active
  enemy (10s–50s on a populated cave), which routes through the same
  probe to detect "outside cave" → teleport back to hive/nest.
- `ListTopPoisByKind` validators that walk many candidate chunks per
  POI play.

So even though one probe is "cheap enough," steady-state cost across
50+ enemies + frequent validator scans is non-trivial.

### What's slow

Per `probe_surface` call:

| Phase                          | `is_solid_at` calls | Notes                                                                |
|--------------------------------|---------------------|----------------------------------------------------------------------|
| 3×3×3 gradient loop (l.129-143)|         162         | 27 cells × 6 neighbor reads — but only 125 *unique* cells in the 5³ extent |
| Per-axis clearance (l.154-161) |         48          | 6 directions × up to 8 steps, capped early on solid hit              |
| 14-direction cavity radius     |         112         | 14 dirs × up to 8 steps, capped early                                |
| **Total per probe**            |       **~322**      | Each call: `div_euclid` ×3 + `rem_euclid` ×3 + `HashMap::get` + bool |

Each `is_solid_at` redoes `div_euclid` / `rem_euclid` on the world coord
and re-probes `store.density_fields` (HashMap, ~40–80 ns hot cache,
~150 ns cold). The gradient loop alone wastes 37 of its 162 reads (162
samples land in 125 unique 5³ cells).

### Proposed patch — chunk-pointer cache + 5×5×5 precompute

Two composable changes:

**(1) Reuse a chunk-pointer between consecutive samples.** Most reads
in any given probe land in 1–8 chunks; once we resolve `(cx,cy,cz)` to
a `&DensityField`, subsequent reads against the same chunk skip the
HashMap probe AND the div/rem math (only `rem_euclid` is needed, and
even that gets short-cut to local-bounds delta).

```rust
struct Sampler<'a> {
    store: &'a ChunkStore,
    chunk_size_i: i32,
    last_key: (i32, i32, i32),
    last_df: Option<&'a voxel_core::density::DensityField>,
    have_last: bool,
}
impl<'a> Sampler<'a> {
    fn new(store: &'a ChunkStore, chunk_size: usize) -> Self { ... }
    #[inline]
    fn is_solid(&mut self, wx: i32, wy: i32, wz: i32) -> bool {
        let cs = self.chunk_size_i;
        let cx = wx.div_euclid(cs);
        let cy = wy.div_euclid(cs);
        let cz = wz.div_euclid(cs);
        let key = (cx, cy, cz);
        if !self.have_last || key != self.last_key {
            self.last_df  = self.store.density_fields.get(&key);
            self.last_key = key;
            self.have_last = true;
        }
        match self.last_df {
            Some(df) => {
                let lx = wx.rem_euclid(cs) as usize;
                let ly = wy.rem_euclid(cs) as usize;
                let lz = wz.rem_euclid(cs) as usize;
                df.get(lx, ly, lz).material.is_solid()
            }
            None => true, // unloaded == solid (placement validators want this)
        }
    }
}
```

**(2) Precompute the 5×5×5 cube once, then walk the gradient through
array indexing.** Eliminates 37 duplicate sampler calls and converts
the 27-cell gradient loop into pure register math.

```rust
let mut cube = [[[false; 5]; 5]; 5];
for cz in 0..5 {
    for cy in 0..5 {
        for cx in 0..5 {
            cube[cx][cy][cz] = sampler.is_solid(
                ox + cx as i32 - 2, oy + cy as i32 - 2, oz + cz as i32 - 2,
            );
        }
    }
}

let (mut nx, mut ny, mut nz) = (0i32, 0i32, 0i32);
for cz in 1..=3 {
    for cy in 1..=3 {
        for cx in 1..=3 {
            nx += cube[cx-1][cy  ][cz  ] as i32 - cube[cx+1][cy  ][cz  ] as i32;
            ny += cube[cx  ][cy-1][cz  ] as i32 - cube[cx  ][cy+1][cz  ] as i32;
            nz += cube[cx  ][cy  ][cz-1] as i32 - cube[cx  ][cy  ][cz+1] as i32;
        }
    }
}
```

Clearance + cavity scans then walk through the same `Sampler` (the
cardinal first-2 steps live inside the precomputed cube, but reusing
those is a small additional win — the bigger gain is the chunk-pointer
cache benefiting every step of every direction).

### Expected impact

Per-call sample arithmetic (numbers conservative, assumes hot DashMap
cache lines for the working chunks):

| Phase             | Before                                     | After                                       | Δ        |
|-------------------|--------------------------------------------|---------------------------------------------|----------|
| Gradient          | 162 × (3 div + 3 rem + HashMap.get ≈ 90 ns) ≈ 14.6 µs | 125 × ~60 ns ≈ 7.5 µs                | −7.1 µs  |
| Clearance         | 48  × ~90 ns ≈ 4.3 µs                      | 48  × ~30 ns (mostly cache hits) ≈ 1.4 µs   | −2.9 µs  |
| Cavity radius     | 112 × ~90 ns ≈ 10.1 µs                     | 112 × ~30 ns ≈ 3.4 µs                       | −6.7 µs  |
| **Per probe**     | **~29 µs**                                 | **~12 µs**                                  | **~−17 µs (≈58 %)** |

System-level translation:

- **Single placement validator call**: ~29 µs → ~12 µs. Imperceptible
  on its own.
- **Cluster-spawn event** (~30 candidates per spawner during a wasp /
  spider cluster placement): ~870 µs → ~360 µs. Halves a frame spike
  that already shows up in the spider-spawner trace.
- **`AEnemyBase::CheckAndFixOutsideCave` 1 Hz tick** with 50 active
  enemies: 50 probes/s × 29 µs ≈ 1.45 ms/s before; 0.6 ms/s after.
  Recovers ~0.85 ms/s of headroom permanently — relevant on the
  Steam-Next-Fest demo target where every CPU ms counts under TWI.
- **`ListTopPoisByKind` montage scoring** (worst case, scans ~200
  candidates per kind across ~3 kinds during a montage POI play): one
  full scan drops from ~17 ms → ~7 ms, i.e. **a 10 ms hitch removed
  from the sleep-montage start frame**. This is the most visible win
  by far — sleep-montage start is on the player's critical-perception
  path.

**Confidence:** Medium-high on the arithmetic; the actual ratio will
shift with `chunk_size` (the larger the chunk, the longer the cache
stays hot per probe, so larger chunks see *more* than 58 % savings).
Cold-cache first-probe-of-cluster is unchanged (the cache only helps
within a single `probe_surface` call), so the win is amortized across
clusters of probes — exactly the usage pattern of placement validators
and POI scans.

**Effort:** ~80 lines, all inside `surface_probe.rs`. No API change,
no behavior change. The existing 5 unit tests in the module already
exercise unloaded-as-solid, all-air, floor / ceiling normals, and the
cavity-radius cap — they're sufficient regression cover for this
refactor.

## Why not shipped today

`voxel-ffi` doesn't build (see Critical above). I can't:
- run the existing 5 `surface_probe` tests to confirm a green baseline,
- run them again after the refactor to confirm no regression,
- build the workspace at all to confirm nothing downstream breaks.

Shipping a perf change against a broken main and pushing it would
either (a) get masked by the existing breakage if I push without
building, or (b) require me to ALSO ship a guess at the missing
topology-vote scanner, which is a non-trivial multi-file feature the
commit author is mid-flight on — exactly the kind of speculative work
the daily-pass workflow says to avoid (cf. 2026-05-24's "deliberately
targets a crate outside voxel-ffi to avoid colliding with that work").

**Recommended sequence:**

1. Author lands the missing `poi_scanner.rs` symbols (`CeilingDome`,
   `Chokepoint`, `WallNiche`, `TopologyVotes`, `count_topology_votes_cross_chunk`,
   6-arg `score_from_votes`).
2. `cargo build --workspace` returns green.
3. Apply the patch above (single-file refactor, ~80 LOC, no API change).
4. `cargo test -p voxel-ffi --lib surface_probe` to confirm 5/5 green.
5. Commit + push as a small standalone change.

## Carry-forward — still outstanding from prior passes

1. **OrePaint Phase 2 anchor selection is O(N²)** —
   [voxel-ffi/src/brushes.rs:619-635](voxel-ffi/src/brushes.rs:619).
   Still blocked on voxel-ffi compile.
2. **Mining/brush callers that feed `update_density` could pass an AABB
   hint** —
   [voxel-fluid/src/cell.rs:359-395](voxel-fluid/src/cell.rs:359). Top
   remaining shipping target once voxel-ffi quiets down. **Estimated
   win: 5–10 % off brush-stroke wall-time at chunk_size=30.**
3. **`apply_density` test paths still call `recompute_capacity()`** —
   [voxel-fluid/src/sim/mod.rs:899-902](voxel-fluid/src/sim/mod.rs:899).
   Test-only, no shipping impact.
4. **POI tracker scan throttle is a fixed `16 chunks / 2 s`** —
   [voxel-ffi/src/poi_tracker.rs:43-45](voxel-ffi/src/poi_tracker.rs:43).
   Cosmetic; blocked.
5. **`count_topology_votes_cross_chunk` itself is the perf opportunity
   flagged 2026-05-23.** Whoever lands the scanner should apply the
   same chunk-pointer caching pattern proposed above — its working set
   is dominated by neighbor reads that cluster heavily inside the
   target chunk + its 6 neighbors.
6. **Path-result cache key + `corner_clip_clear` invalidation tag** —
   2026-05-24 carry-forward. **Estimated win: 30–60 % off repeat-pathing
   CPU for spider chase loops on unchanging terrain.**
7. **`surface_normal_at` per-search memo** —
   [voxel-ffi/src/pathing.rs:109-123](voxel-ffi/src/pathing.rs:109).
   **Estimated win: 10–15 % off Spider-only path queries.** Note this
   shares structure with today's `surface_probe.rs` finding — a small
   shared `chunk-pointer-cached sampler` utility module could serve
   both (and the existing path-planner ChunkStoreGrid). Worth folding
   into a single follow-up pass.

### New observation worth recording

**`probe_surface` and `surface_normal_at` (pathing.rs) duplicate the
same "sample a small window of `is_solid` reads against `ChunkStore`"
pattern.** A shared `ChunkSampler` helper that holds the chunk-pointer
cache + the `(div_euclid, rem_euclid)` math in one place would serve
both call sites + the broken `count_topology_votes_cross_chunk` once
it lands. Three call sites × hot inner loops = good ROI for ~50 LOC of
new helper. Flag for the pass after the surface_probe patch ships.

End of review.
