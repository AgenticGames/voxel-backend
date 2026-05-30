//! Ore-paint brush: weighted, Poisson-spaced wall-exposed ore deposits with
//! optional inward "deep channel" tubes. Density is never modified.

use glam::Vec3;
use voxel_core::material::Material;
use voxel_gen::config::GenerationConfig;

use crate::store::ChunkStore;

use super::common::{
    capture_undo_for_range, chunk_range_for_sphere, finalize_brush, local_sphere_bounds,
    BrushOutcome,
};

/// Per-ore weighting passed to `paint_ore_deposits`. Each weight is a relative
/// frequency for the corresponding material; the brush normalizes the sum at
/// runtime, so values like `[1, 0, 0, 0, 0, ..]` mean "iron only" and
/// `[3, 1, 0, 0, 0, ..]` means "75% iron / 25% copper". Indices match
/// `ORE_MATERIALS` inside `paint_ore_deposits`.
#[derive(Debug, Clone, Copy)]
pub struct OreWeights {
    pub iron: u8,
    pub copper: u8,
    pub malachite: u8,
    pub tin: u8,
    pub gold: u8,
    pub diamond: u8,
    pub kimberlite: u8,
    pub sulfide: u8,
    pub quartz: u8,
    pub pyrite: u8,
    pub amethyst: u8,
    pub crystal: u8,
    pub coal: u8,
}

impl OreWeights {
    /// Sensible "balanced ore field" defaults — iron + copper common, gold and
    /// diamond rare, accents (quartz/pyrite/amethyst) scattered. Used as the
    /// brush's initial state and as the test/CLI fallback.
    pub fn balanced() -> Self {
        OreWeights {
            iron: 30, copper: 20, malachite: 6, tin: 6, gold: 4, diamond: 1,
            kimberlite: 2, sulfide: 8, quartz: 6, pyrite: 7,
            amethyst: 3, crystal: 2, coal: 12,
        }
    }
}

// Tiny xorshift32 — used by the OrePaint brush for deterministic seeded
// placement without pulling in a `rand` dependency. Free fn (not a closure)
// so callers can pass `&mut state` alongside other `&mut`s without tripping
// the borrow checker.
fn ore_xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// OrePaint brush — drops wall-exposed ore deposits inside the sphere with
/// even (Poisson-disk) spacing, weighted ore-type picks, and optional inward
/// "deep channel" tubes for each cluster.
///
/// Tuning knobs:
/// * `cluster_size`  — voxel radius of each ore knob (typical 1.0 – 3.0).
/// * `min_spacing`   — minimum voxel distance between cluster anchors
///                     (anti-clumping; typical 2× cluster_size or more).
/// * `channel_prob`  — chance per anchor to extend a deeper tube into rock.
/// * `channel_length`— voxels along the inward normal to march the tube.
/// * `channel_radius`— voxels — tube thickness.
/// * `density`       — 0..1 — caps the total anchors as a fraction of the
///                     wall-surface-candidate count. 0.0 = no anchors,
///                     1.0 = pack as many as `min_spacing` allows.
/// * `seed`          — re-stamping the same brush with a new seed gives a
///                     fresh layout.
///
/// Output: a `BrushOutcome` containing meshes for every chunk a deposit (or
/// tube) touched. Density is never modified — only `sample.material`.
/// Air voxels are skipped. Existing host rock under the brush is overwritten
/// where ore lands.
pub fn paint_ore_deposits(
    store: &mut ChunkStore,
    center: Vec3,
    radius: f32,
    weights: OreWeights,
    cluster_size: f32,
    min_spacing: f32,
    channel_prob: f32,
    channel_length: f32,
    channel_radius: f32,
    density: f32,
    seed: u32,
    config: &GenerationConfig,
    world_scale: f32,
) -> BrushOutcome {
    const ORE_MATERIALS: [Material; 13] = [
        Material::Iron,
        Material::Copper,
        Material::Malachite,
        Material::Tin,
        Material::Gold,
        Material::Diamond,
        Material::Kimberlite,
        Material::Sulfide,
        Material::Quartz,
        Material::Pyrite,
        Material::Amethyst,
        Material::Crystal,
        Material::Coal,
    ];

    if radius <= 0.0 || density <= 0.0 {
        return BrushOutcome { meshes: Vec::new(), flipped_chunks: Vec::new() };
    }

    let weight_arr: [u8; 13] = [
        weights.iron, weights.copper, weights.malachite, weights.tin,
        weights.gold, weights.diamond, weights.kimberlite, weights.sulfide,
        weights.quartz, weights.pyrite, weights.amethyst, weights.crystal,
        weights.coal,
    ];
    let total_weight: u32 = weight_arr.iter().map(|&w| w as u32).sum();
    if total_weight == 0 {
        return BrushOutcome { meshes: Vec::new(), flipped_chunks: Vec::new() };
    }

    let eb = config.effective_bounds();
    let vs = config.voxel_scale();
    let r2 = radius * radius;

    // Conservative AABB covering the brush + worst-case channel extension so
    // tubes that march outside the brush sphere still get cross-chunk writes.
    let pad = cluster_size.max(channel_length.max(channel_radius)) + 2.0;
    let aabb_radius = radius + pad;
    let (lo, hi) = chunk_range_for_sphere(center, aabb_radius, eb);

    capture_undo_for_range(store, lo, hi);

    // ── Phase 1: gather wall-exposed solid voxels (candidate anchors) ──
    //
    // A "wall-exposed" voxel is solid and has at least one face-neighbor that
    // is air. We also compute an inward-normal direction (the average of the
    // unit vectors pointing from each air neighbor back into the rock), so
    // deep-channel tubes know which way to march. Candidates outside the
    // brush sphere are skipped.
    struct WallCandidate {
        chunk: (i32, i32, i32),
        lx: usize,
        ly: usize,
        lz: usize,
        world_pos: Vec3,
        inward: Vec3,
    }
    let mut candidates: Vec<WallCandidate> = Vec::new();

    let (clo, chi) = chunk_range_for_sphere(center, radius, eb);
    for cz in clo.2..=chi.2 {
        for cy in clo.1..=chi.1 {
            for cx in clo.0..=chi.0 {
                let key = (cx, cy, cz);
                let Some(density_field) = store.density_fields.get(&key) else { continue };
                let origin = Vec3::new(cx as f32 * eb, cy as f32 * eb, cz as f32 * eb);
                let (_, _, lo_x, lo_y, lo_z, hi_x, hi_y, hi_z) =
                    local_sphere_bounds(center, radius, origin, vs, density_field.size);
                let sz = density_field.size;

                for z in lo_z..hi_z {
                    for y in lo_y..hi_y {
                        for x in lo_x..hi_x {
                            let s = density_field.get(x, y, z);
                            if !s.material.is_solid() {
                                continue;
                            }
                            let world_pos = origin
                                + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                            if (world_pos - center).length_squared() > r2 {
                                continue;
                            }
                            // Walk the 6 face neighbors. For each that's air,
                            // accumulate the unit vector pointing FROM that air
                            // cell TOWARD this voxel — that's "inward".
                            //
                            // Out-of-bounds neighbors look across the chunk
                            // boundary via the store (so candidates near a
                            // chunk edge still get classified correctly).
                            const OFFSETS: [(i32, i32, i32); 6] = [
                                (1, 0, 0), (-1, 0, 0),
                                (0, 1, 0), (0, -1, 0),
                                (0, 0, 1), (0, 0, -1),
                            ];
                            let mut inward = Vec3::ZERO;
                            let mut any_air = false;
                            for &(dx, dy, dz) in &OFFSETS {
                                let nx = x as i32 + dx;
                                let ny = y as i32 + dy;
                                let nz = z as i32 + dz;
                                let neighbor_solid = if nx < 0
                                    || ny < 0
                                    || nz < 0
                                    || (nx as usize) >= sz
                                    || (ny as usize) >= sz
                                    || (nz as usize) >= sz
                                {
                                    let wx = cx * config.chunk_size as i32 + nx;
                                    let wy = cy * config.chunk_size as i32 + ny;
                                    let wz = cz * config.chunk_size as i32 + nz;
                                    let nkey = (
                                        wx.div_euclid(config.chunk_size as i32),
                                        wy.div_euclid(config.chunk_size as i32),
                                        wz.div_euclid(config.chunk_size as i32),
                                    );
                                    let nlx = wx.rem_euclid(config.chunk_size as i32) as usize;
                                    let nly = wy.rem_euclid(config.chunk_size as i32) as usize;
                                    let nlz = wz.rem_euclid(config.chunk_size as i32) as usize;
                                    store
                                        .density_fields
                                        .get(&nkey)
                                        .map(|df| df.get(nlx, nly, nlz).material.is_solid())
                                        // Unloaded neighbor — treat as solid so we don't
                                        // hallucinate a wall at the streaming edge.
                                        .unwrap_or(true)
                                } else {
                                    density_field
                                        .get(nx as usize, ny as usize, nz as usize)
                                        .material
                                        .is_solid()
                                };
                                if !neighbor_solid {
                                    any_air = true;
                                    inward -= Vec3::new(dx as f32, dy as f32, dz as f32);
                                }
                            }
                            if !any_air {
                                continue;
                            }
                            let inward = if inward.length_squared() > 1e-6 {
                                inward.normalize()
                            } else {
                                Vec3::new(0.0, -1.0, 0.0)
                            };
                            candidates.push(WallCandidate {
                                chunk: key,
                                lx: x,
                                ly: y,
                                lz: z,
                                world_pos,
                                inward,
                            });
                        }
                    }
                }
            }
        }
    }
    if candidates.is_empty() {
        return BrushOutcome { meshes: Vec::new(), flipped_chunks: Vec::new() };
    }

    // ── Phase 2: Poisson-disk-ish anchor selection ──
    //
    // Shuffle candidates with a deterministic seed (xorshift32), then accept
    // in order with a minimum-spacing rejection. Simpler than Bridson and
    // plenty good for ~hundreds of anchors; the brush sphere caps total work.
    // `target_count` is derived from the density slider so a low density gives
    // a sparse field even when the wall is full of candidates.
    let mut rng_state: u32 = seed.wrapping_mul(2654435761).wrapping_add(0x9E3779B9);
    if rng_state == 0 {
        rng_state = 0xDEADBEEF;
    }
    // Fisher–Yates with our xorshift32.
    let n = candidates.len();
    for i in (1..n).rev() {
        let j = (ore_xorshift32(&mut rng_state) as usize) % (i + 1);
        candidates.swap(i, j);
    }

    let target_count = ((candidates.len() as f32) * density).ceil() as usize;
    let target_count = target_count.max(1);
    let min_spacing2 = min_spacing * min_spacing;
    // Spatial-hash grid for O(1)-avg spacing check. Cell side = min_spacing, so
    // any pair closer than that lives in the same or a 26-neighbor cell — we
    // only ever scan a 3×3×3 window instead of the full `accepted` list.
    // Determinism: identical iteration order over `candidates` + exhaustive
    // distance check within the bounded window → same acceptance set as the
    // prior O(N·K) scan. Verified by `ore_paint_seed_determinism`.
    // Backing store is a flat Vec<Vec<Vec3>> indexed by linear cell coord —
    // a HashMap probe measured slower than the linear scan for typical K,
    // but flat indexing into a brush-AABB-sized grid removes that overhead.
    let inv_cell = if min_spacing > 1e-6 { 1.0 / min_spacing } else { 1.0 };
    // Brush AABB in cell coords. `aabb_radius` is the gather radius (already
    // padded for cluster_size / channel reach) so every candidate falls inside.
    let cell_min_x = ((center.x - aabb_radius) * inv_cell).floor() as i32 - 1;
    let cell_min_y = ((center.y - aabb_radius) * inv_cell).floor() as i32 - 1;
    let cell_min_z = ((center.z - aabb_radius) * inv_cell).floor() as i32 - 1;
    let cell_max_x = ((center.x + aabb_radius) * inv_cell).floor() as i32 + 1;
    let cell_max_y = ((center.y + aabb_radius) * inv_cell).floor() as i32 + 1;
    let cell_max_z = ((center.z + aabb_radius) * inv_cell).floor() as i32 + 1;
    let dim_x = (cell_max_x - cell_min_x + 1).max(1) as usize;
    let dim_y = (cell_max_y - cell_min_y + 1).max(1) as usize;
    let dim_z = (cell_max_z - cell_min_z + 1).max(1) as usize;
    let mut grid: Vec<Vec<Vec3>> = vec![Vec::new(); dim_x * dim_y * dim_z];
    let cell_idx = |gx: i32, gy: i32, gz: i32| -> Option<usize> {
        if gx < cell_min_x || gy < cell_min_y || gz < cell_min_z
            || gx > cell_max_x || gy > cell_max_y || gz > cell_max_z
        {
            return None;
        }
        let lx = (gx - cell_min_x) as usize;
        let ly = (gy - cell_min_y) as usize;
        let lz = (gz - cell_min_z) as usize;
        Some((lz * dim_y + ly) * dim_x + lx)
    };
    let mut accepted: Vec<&WallCandidate> = Vec::new();
    for cand in &candidates {
        if accepted.len() >= target_count {
            break;
        }
        let gx = (cand.world_pos.x * inv_cell).floor() as i32;
        let gy = (cand.world_pos.y * inv_cell).floor() as i32;
        let gz = (cand.world_pos.z * inv_cell).floor() as i32;
        let mut too_close = false;
        'outer: for dz in -1..=1i32 {
            for dy in -1..=1i32 {
                for dx in -1..=1i32 {
                    if let Some(ix) = cell_idx(gx + dx, gy + dy, gz + dz) {
                        for p in &grid[ix] {
                            if (*p - cand.world_pos).length_squared() < min_spacing2 {
                                too_close = true;
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }
        if !too_close {
            if let Some(ix) = cell_idx(gx, gy, gz) {
                grid[ix].push(cand.world_pos);
            }
            accepted.push(cand);
        }
    }

    if accepted.is_empty() {
        return BrushOutcome { meshes: Vec::new(), flipped_chunks: Vec::new() };
    }

    // ── Phase 3: write ore voxels (clusters + optional channels) ──
    //
    // Collect dirty rects keyed by chunk and grow per-chunk bounds as we
    // write. `paint_ore_sphere_voxels` iterates chunks in the brush AABB
    // and does ONE `get_mut` per chunk (same pattern as `paint_material_sphere`
    // and `carve_sphere`), rather than one HashMap lookup per written voxel.
    // For typical settings (cluster_size ≈ 2, channel_radius ≈ 1) this cuts
    // HashMap lookups by ~95% in this phase, since a single small sphere
    // usually lives entirely within one chunk.
    let chunk_size_i = config.chunk_size as i32;
    let mut per_chunk_dirty: std::collections::HashMap<
        (i32, i32, i32),
        (usize, usize, usize, usize, usize, usize),
    > = std::collections::HashMap::new();

    // Stamp `target` on every solid voxel inside a sphere of integer-voxel
    // radius `r_int` around (`cwx`, `cwy`, `cwz`), where `r2` is `r_int^2`
    // (computed once by the caller). Iterates chunks in the AABB; one
    // `get_mut` per chunk; only writes voxels strictly inside the sphere
    // and only when the existing material is solid and not already `target`.
    fn paint_ore_sphere_voxels(
        store: &mut ChunkStore,
        cwx: i32, cwy: i32, cwz: i32,
        r_int: i32,
        r2: f32,
        target: Material,
        chunk_size_i: i32,
        per_chunk_dirty: &mut std::collections::HashMap<
            (i32, i32, i32),
            (usize, usize, usize, usize, usize, usize),
        >,
    ) {
        let lo_wx = cwx - r_int;
        let lo_wy = cwy - r_int;
        let lo_wz = cwz - r_int;
        let hi_wx = cwx + r_int;
        let hi_wy = cwy + r_int;
        let hi_wz = cwz + r_int;
        let cklo = (
            lo_wx.div_euclid(chunk_size_i),
            lo_wy.div_euclid(chunk_size_i),
            lo_wz.div_euclid(chunk_size_i),
        );
        let ckhi = (
            hi_wx.div_euclid(chunk_size_i),
            hi_wy.div_euclid(chunk_size_i),
            hi_wz.div_euclid(chunk_size_i),
        );
        for ckz in cklo.2..=ckhi.2 {
            for cky in cklo.1..=ckhi.1 {
                for ckx in cklo.0..=ckhi.0 {
                    let key = (ckx, cky, ckz);
                    let Some(df) = store.density_fields.get_mut(&key) else { continue };
                    let base_wx = ckx * chunk_size_i;
                    let base_wy = cky * chunk_size_i;
                    let base_wz = ckz * chunk_size_i;
                    let sz_i = df.size as i32;
                    let lo_lx = (lo_wx - base_wx).max(0);
                    let lo_ly = (lo_wy - base_wy).max(0);
                    let lo_lz = (lo_wz - base_wz).max(0);
                    let hi_lx = (hi_wx - base_wx).min(sz_i - 1);
                    let hi_ly = (hi_wy - base_wy).min(sz_i - 1);
                    let hi_lz = (hi_wz - base_wz).min(sz_i - 1);
                    if hi_lx < lo_lx || hi_ly < lo_ly || hi_lz < lo_lz {
                        continue;
                    }
                    let mut wrote_any = false;
                    let mut mn_x = usize::MAX;
                    let mut mn_y = usize::MAX;
                    let mut mn_z = usize::MAX;
                    let mut mx_x = 0usize;
                    let mut mx_y = 0usize;
                    let mut mx_z = 0usize;
                    for lz in lo_lz..=hi_lz {
                        let dz = (base_wz + lz) - cwz;
                        let dz2 = (dz * dz) as f32;
                        for ly in lo_ly..=hi_ly {
                            let dy = (base_wy + ly) - cwy;
                            let dyz2 = dz2 + (dy * dy) as f32;
                            for lx in lo_lx..=hi_lx {
                                let dx = (base_wx + lx) - cwx;
                                let d2 = dyz2 + (dx * dx) as f32;
                                if d2 > r2 {
                                    continue;
                                }
                                let s = df.get_mut(lx as usize, ly as usize, lz as usize);
                                if !s.material.is_solid() || s.material == target {
                                    continue;
                                }
                                s.material = target;
                                let ulx = lx as usize;
                                let uly = ly as usize;
                                let ulz = lz as usize;
                                if !wrote_any {
                                    mn_x = ulx; mn_y = uly; mn_z = ulz;
                                    mx_x = ulx; mx_y = uly; mx_z = ulz;
                                    wrote_any = true;
                                } else {
                                    if ulx < mn_x { mn_x = ulx; }
                                    if uly < mn_y { mn_y = uly; }
                                    if ulz < mn_z { mn_z = ulz; }
                                    if ulx > mx_x { mx_x = ulx; }
                                    if uly > mx_y { mx_y = uly; }
                                    if ulz > mx_z { mx_z = ulz; }
                                }
                            }
                        }
                    }
                    if wrote_any {
                        let e = per_chunk_dirty
                            .entry(key)
                            .or_insert((mn_x, mn_y, mn_z, mx_x, mx_y, mx_z));
                        if mn_x < e.0 { e.0 = mn_x; }
                        if mn_y < e.1 { e.1 = mn_y; }
                        if mn_z < e.2 { e.2 = mn_z; }
                        if mx_x > e.3 { e.3 = mx_x; }
                        if mx_y > e.4 { e.4 = mx_y; }
                        if mx_z > e.5 { e.5 = mx_z; }
                    }
                }
            }
        }
    }

    // Pick a weighted ore type using the same xorshift state.
    let pick_ore = |rng: &mut u32| -> Material {
        let mut r = ore_xorshift32(rng) % total_weight;
        for (i, &w) in weight_arr.iter().enumerate() {
            let w = w as u32;
            if r < w {
                return ORE_MATERIALS[i];
            }
            r -= w;
        }
        ORE_MATERIALS[0] // unreachable for total_weight > 0, defensive
    };

    let cluster_r2 = cluster_size * cluster_size;
    let channel_r2 = channel_radius * channel_radius;

    let cs_int = cluster_size.ceil() as i32;
    let chan_rr = channel_radius.ceil() as i32;
    for anchor in &accepted {
        let ore = pick_ore(&mut rng_state);

        // ── Cluster: sphere of radius `cluster_size` around the anchor ──
        let anchor_wx = anchor.chunk.0 * chunk_size_i + anchor.lx as i32;
        let anchor_wy = anchor.chunk.1 * chunk_size_i + anchor.ly as i32;
        let anchor_wz = anchor.chunk.2 * chunk_size_i + anchor.lz as i32;
        paint_ore_sphere_voxels(
            store,
            anchor_wx, anchor_wy, anchor_wz,
            cs_int, cluster_r2, ore,
            chunk_size_i, &mut per_chunk_dirty,
        );

        // ── Channel: optional inward tube ──
        //
        // Rolls per anchor. The tube is a sequence of small spheres of radius
        // `channel_radius` stepped 1 voxel along the inward normal. Each step
        // gets a tiny perpendicular jitter so multiple channels don't look
        // perfectly straight. Stops early if it walks into air voxels (player
        // would have nothing to mine through).
        let roll = (ore_xorshift32(&mut rng_state) % 10_000) as f32 / 10_000.0;
        if roll < channel_prob && channel_length > 0.5 {
            let basis = if anchor.inward.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
            let perp_a = anchor.inward.cross(basis).normalize_or_zero();
            let perp_b = anchor.inward.cross(perp_a).normalize_or_zero();
            let length = channel_length.round() as i32;
            // Start one voxel inward so the channel attaches to the cluster
            // and continues away from the wall.
            let mut head = Vec3::new(anchor_wx as f32, anchor_wy as f32, anchor_wz as f32)
                + anchor.inward;
            for _step in 0..length {
                let jx = ((ore_xorshift32(&mut rng_state) % 1000) as f32 / 1000.0 - 0.5) * 0.8;
                let jy = ((ore_xorshift32(&mut rng_state) % 1000) as f32 / 1000.0 - 0.5) * 0.8;
                let pos = head + perp_a * jx + perp_b * jy;
                let cx = pos.x.round() as i32;
                let cy = pos.y.round() as i32;
                let cz = pos.z.round() as i32;

                // Bail out if the tube head walked into air — the tube has
                // poked into another cavern and there's nothing left to paint.
                let key = (
                    cx.div_euclid(chunk_size_i),
                    cy.div_euclid(chunk_size_i),
                    cz.div_euclid(chunk_size_i),
                );
                let lx = cx.rem_euclid(chunk_size_i) as usize;
                let ly = cy.rem_euclid(chunk_size_i) as usize;
                let lz = cz.rem_euclid(chunk_size_i) as usize;
                let center_solid = store
                    .density_fields
                    .get(&key)
                    .map(|df| {
                        if lx < df.size && ly < df.size && lz < df.size {
                            df.get(lx, ly, lz).material.is_solid()
                        } else {
                            true
                        }
                    })
                    .unwrap_or(true);
                if !center_solid {
                    break;
                }

                paint_ore_sphere_voxels(
                    store,
                    cx, cy, cz,
                    chan_rr, channel_r2, ore,
                    chunk_size_i, &mut per_chunk_dirty,
                );

                head += anchor.inward;
            }
        }
    }

    // ── Phase 4: finalize — sync seams, mark dirty, remesh ──
    let mut dirty_chunks: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> =
        Vec::with_capacity(per_chunk_dirty.len());
    for (key, (mn_x, mn_y, mn_z, mx_x, mx_y, mx_z)) in per_chunk_dirty {
        let Some(df) = store.density_fields.get(&key) else { continue };
        let expand = config.mine.dirty_expand as usize;
        let lo_x = mn_x.saturating_sub(expand);
        let lo_y = mn_y.saturating_sub(expand);
        let lo_z = mn_z.saturating_sub(expand);
        let hi_x = (mx_x + expand).min(df.size - 1);
        let hi_y = (mx_y + expand).min(df.size - 1);
        let hi_z = (mx_z + expand).min(df.size - 1);
        dirty_chunks.push((key, lo_x, lo_y, lo_z, hi_x, hi_y, hi_z));
    }

    if dirty_chunks.is_empty() {
        return BrushOutcome { meshes: Vec::new(), flipped_chunks: Vec::new() };
    }

    finalize_brush(store, dirty_chunks, config, world_scale)
}
