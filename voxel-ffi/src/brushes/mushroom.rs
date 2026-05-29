//! Mushroom placement / erase brushes. These do NOT modify density — they only
//! mutate `store.mushroom_placements` and capture mushroom-only undo strokes.

use glam::Vec3;
use voxel_gen::config::GenerationConfig;

use crate::store::ChunkStore;

use super::common::capture_mushroom_undo;

/// Place a single mushroom instance at `center_rust` (Rust voxel space).
///
/// Picks the nearest solid voxel within `search_radius` voxels as the anchor,
/// infers floor/wall/ceiling face from its air-neighbor pattern, generates a
/// `MushroomPlacement` in the chunk that owns that anchor, and inserts it
/// into `store.mushroom_placements`. Does NOT modify density.
///
/// Returns the (chunk_key) of the chunk that got a new placement so the
/// caller can trigger a remesh-and-resend (which carries the new
/// `mushroom_data` to UE). Returns `None` if no anchor was found.
pub fn place_mushroom_at_world(
    store: &mut ChunkStore,
    center_rust: Vec3,
    kind: u8,
    search_radius: f32,
    scale_override: f32,
    yaw_override: f32,
    config: &GenerationConfig,
) -> Option<(i32, i32, i32)> {
    // Resolve the kind first — bail on invalid input rather than placing garbage.
    let mushroom_kind = voxel_gen::MushroomKind::from_u8(kind)?;

    let eb = config.effective_bounds();
    let radius_voxels = (search_radius.max(1.0) / 1.0).ceil() as i32;

    // Scan a small AABB around `center_rust` for solid voxels and pick the
    // closest one. This handles the case where the player clicks slightly
    // off-surface (small offset in any direction).
    let cx0 = (center_rust.x - radius_voxels as f32).floor() as i32;
    let cy0 = (center_rust.y - radius_voxels as f32).floor() as i32;
    let cz0 = (center_rust.z - radius_voxels as f32).floor() as i32;
    let cx1 = (center_rust.x + radius_voxels as f32).ceil() as i32;
    let cy1 = (center_rust.y + radius_voxels as f32).ceil() as i32;
    let cz1 = (center_rust.z + radius_voxels as f32).ceil() as i32;

    let chunk_size = config.chunk_size as i32;
    let to_chunk = |w: i32| -> (i32, i32) {
        (w.div_euclid(chunk_size), w.rem_euclid(chunk_size))
    };

    // Search for the closest solid-voxel anchor.
    let mut best: Option<((i32, i32, i32), (usize, usize, usize), f32)> = None;
    for wx in cx0..=cx1 {
        for wy in cy0..=cy1 {
            for wz in cz0..=cz1 {
                let (chunk_cx, lx) = to_chunk(wx);
                let (chunk_cy, ly) = to_chunk(wy);
                let (chunk_cz, lz) = to_chunk(wz);
                let key = (chunk_cx, chunk_cy, chunk_cz);
                let density = match store.density_fields.get(&key) {
                    Some(d) => d,
                    None => continue,
                };
                if lx as usize >= density.size || ly as usize >= density.size || lz as usize >= density.size {
                    continue;
                }
                let sample = density.get(lx as usize, ly as usize, lz as usize);
                if !sample.material.is_solid() {
                    continue;
                }
                let dx = wx as f32 + 0.5 - center_rust.x;
                let dy = wy as f32 + 0.5 - center_rust.y;
                let dz = wz as f32 + 0.5 - center_rust.z;
                let d2 = dx * dx + dy * dy + dz * dz;
                if best.as_ref().is_none_or(|(_, _, bd)| d2 < *bd) {
                    best = Some((key, (lx as usize, ly as usize, lz as usize), d2));
                }
            }
        }
    }

    let (key, (lx, ly, lz), _d2) = best?;
    let density = store.density_fields.get(&key)?;
    let size = density.size;

    // Infer surface face from air-neighbor pattern around the anchor.
    let air_above = ly + 1 < size && !density.get(lx, ly + 1, lz).material.is_solid();
    let air_below = ly > 0 && !density.get(lx, ly - 1, lz).material.is_solid();
    let air_xn = lx > 0 && !density.get(lx - 1, ly, lz).material.is_solid();
    let air_xp = lx + 1 < size && !density.get(lx + 1, ly, lz).material.is_solid();
    let air_zn = lz > 0 && !density.get(lx, ly, lz - 1).material.is_solid();
    let air_zp = lz + 1 < size && !density.get(lx, ly, lz + 1).material.is_solid();

    // Floor face wins over wall (matches `compute_mushroom_placements` priority).
    let (nx, ny, nz) = if air_above {
        (0.0f32, 1.0f32, 0.0f32)
    } else if air_below {
        (0.0, -1.0, 0.0)
    } else if air_xn || air_xp || air_zn || air_zp {
        let mut nx = 0.0f32;
        let mut nz = 0.0f32;
        if air_xn { nx -= 1.0; }
        if air_xp { nx += 1.0; }
        if air_zn { nz -= 1.0; }
        if air_zp { nz += 1.0; }
        let len = (nx * nx + nz * nz).sqrt();
        if len > 0.0 { (nx / len, 0.0, nz / len) } else { (1.0, 0.0, 0.0) }
    } else {
        // Anchor has no exposed face — buried voxel. Default to up.
        (0.0, 1.0, 0.0)
    };

    // Scale + yaw — use kind config if caller passed 0.0. Determinism via a
    // tiny xorshift seeded from anchor + chunk coords (no rand crate dep).
    let kind_cfg = config.mushrooms.kind(mushroom_kind);
    let mut seed: u64 = (key.0 as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ (key.1 as u64).wrapping_mul(0xBF58476D1CE4E5B9)
        ^ (key.2 as u64).wrapping_mul(0x94D049BB133111EB)
        ^ (((lx as u64) << 32) | ((ly as u64) << 16) | (lz as u64));
    let mut next_unit = || -> f32 {
        // xorshift64 step
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        // Convert top 24 bits to [0, 1)
        ((seed >> 40) as f32) / ((1u32 << 24) as f32)
    };
    let scale = if scale_override > 0.0 {
        scale_override
    } else {
        kind_cfg.scale_min + (kind_cfg.scale_max - kind_cfg.scale_min) * next_unit()
    };
    let yaw = if yaw_override > 0.0 { yaw_override } else { next_unit() * std::f32::consts::TAU };

    // Build the placement. Position is the anchor cell center pushed half a
    // voxel along the normal so the mesh base sits on the surface, same as
    // worldgen.
    let chunk_origin = Vec3::new(
        key.0 as f32 * eb,
        key.1 as f32 * eb,
        key.2 as f32 * eb,
    );
    let _ = chunk_origin; // chunk_origin not needed because positions are chunk-relative

    let placement = voxel_gen::MushroomPlacement {
        x: lx as f32 + 0.5 + nx * 0.5,
        y: ly as f32 + 0.5 + ny * 0.5,
        z: lz as f32 + 0.5 + nz * 0.5,
        normal_x: nx,
        normal_y: ny,
        normal_z: nz,
        scale,
        yaw,
        kind: mushroom_kind,
        anchor_lx: lx.min(u8::MAX as usize) as u8,
        anchor_ly: ly.min(u8::MAX as usize) as u8,
        anchor_lz: lz.min(u8::MAX as usize) as u8,
    };

    store.mushroom_placements.entry(key).or_default().push(placement);
    Some(key)
}

/// Sphere-area mushroom brush. Scans every solid voxel in the sphere whose
/// preferred-face neighbor is air (kind→face: TurkeyTail→wall, Foxfire→ceiling,
/// GreenPepe/GhostTower→floor), rolls a Bernoulli with `density` per candidate,
/// applies min-spacing, and inserts one `MushroomPlacement` per accepted point.
///
/// `radius_voxels` is in voxel units (UE callers convert from UE units before
/// passing). `density` is the per-candidate accept probability in [0, 1].
/// `seed` randomizes the placement so repeated brush clicks on the same spot
/// produce different patterns. Returns the set of chunk keys that received at
/// least one new placement so the caller can trigger seam-pass remeshes.
pub fn place_mushrooms_brush_sphere(
    store: &mut ChunkStore,
    center_rust: Vec3,
    kind: u8,
    radius_voxels: f32,
    density: f32,
    clustering: f32,
    seed: u64,
    config: &GenerationConfig,
) -> std::collections::HashSet<(i32, i32, i32)> {
    let mut affected = std::collections::HashSet::new();
    let mushroom_kind = match voxel_gen::MushroomKind::from_u8(kind) {
        Some(k) => k,
        None => return affected,
    };
    if density <= 0.0 || radius_voxels <= 0.0 {
        return affected;
    }

    // Pre-scan affected chunks for undo snapshot. Capture every chunk the
    // sphere overlaps that has loaded density data, so undo can restore
    // even when no mushrooms actually got placed in some of them.
    let undo_keys: Vec<(i32, i32, i32)> = {
        let chunk_size = config.chunk_size as i32;
        let cx0 = ((center_rust.x - radius_voxels).floor() as i32).div_euclid(chunk_size);
        let cy0 = ((center_rust.y - radius_voxels).floor() as i32).div_euclid(chunk_size);
        let cz0 = ((center_rust.z - radius_voxels).floor() as i32).div_euclid(chunk_size);
        let cx1 = ((center_rust.x + radius_voxels).ceil() as i32).div_euclid(chunk_size);
        let cy1 = ((center_rust.y + radius_voxels).ceil() as i32).div_euclid(chunk_size);
        let cz1 = ((center_rust.z + radius_voxels).ceil() as i32).div_euclid(chunk_size);
        let mut keys = Vec::new();
        for cz in cz0..=cz1 { for cy in cy0..=cy1 { for cx in cx0..=cx1 {
            if store.density_fields.contains_key(&(cx, cy, cz)) {
                keys.push((cx, cy, cz));
            }
        }}}
        keys
    };
    capture_mushroom_undo(store, &undo_keys);

    // Clustering: sample a local Simplex noise and gate Bernoulli by it.
    // clustering=0 → uniform (gate=1 everywhere). clustering=1 → tight pockets.
    // Frequency rises with clustering (smaller patches), threshold rises with
    // clustering (more rejection outside patches). Compensate density inside
    // accepted regions so total count stays roughly comparable to clustering=0.
    let clustering = clustering.clamp(0.0, 1.0);
    let noise_freq = 0.04 + clustering as f64 * 0.18;   // 0.04..0.22 per voxel
    let noise_thresh = -1.0 + clustering * 1.5;          // -1.0..0.5
    let scatter = voxel_noise::simplex::Simplex3D::new(seed);
    use voxel_noise::NoiseSource;
    // Approximate fraction of space passing the threshold for a uniform-ish
    // noise distribution. Used to amplify density so the total count
    // doesn't collapse as clustering rises.
    let pass_frac = ((1.0 - noise_thresh) * 0.5).clamp(0.05, 1.0);
    let density_eff = if clustering > 0.0 { (density / pass_frac).min(1.0) } else { density };

    let radius = radius_voxels.max(0.5);
    let radius2 = radius * radius;
    let cx0 = (center_rust.x - radius).floor() as i32;
    let cy0 = (center_rust.y - radius).floor() as i32;
    let cz0 = (center_rust.z - radius).floor() as i32;
    let cx1 = (center_rust.x + radius).ceil() as i32;
    let cy1 = (center_rust.y + radius).ceil() as i32;
    let cz1 = (center_rust.z + radius).ceil() as i32;

    let chunk_size = config.chunk_size as i32;
    let to_chunk = |w: i32| -> (i32, i32) {
        (w.div_euclid(chunk_size), w.rem_euclid(chunk_size))
    };

    // xorshift64 seeded from request seed + center for determinism per click.
    let mut rng_state: u64 = seed
        .wrapping_mul(0x9E3779B97F4A7C15)
        ^ ((center_rust.x * 1000.0) as i64 as u64)
        ^ (((center_rust.y * 1000.0) as i64 as u64).wrapping_mul(0xBF58476D1CE4E5B9))
        ^ (((center_rust.z * 1000.0) as i64 as u64).wrapping_mul(0x94D049BB133111EB));
    if rng_state == 0 { rng_state = 0xC0FFEE_DEAD_BEEF; }
    let mut next_unit = || -> f32 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        ((rng_state >> 40) as f32) / ((1u32 << 24) as f32)
    };

    let kind_cfg = config.mushrooms.kind(mushroom_kind);
    let min_spacing = config.mushrooms.min_spacing_voxels.max(0.5);
    let min_spacing2 = min_spacing * min_spacing;
    // World-space placed positions for cross-chunk min-spacing check.
    let mut placed_world: Vec<(f32, f32, f32)> = Vec::new();

    for wx in cx0..=cx1 {
        for wy in cy0..=cy1 {
            for wz in cz0..=cz1 {
                let cx = wx as f32 + 0.5 - center_rust.x;
                let cy = wy as f32 + 0.5 - center_rust.y;
                let cz = wz as f32 + 0.5 - center_rust.z;
                if cx * cx + cy * cy + cz * cz > radius2 { continue; }

                let (cck_x, lx_i) = to_chunk(wx);
                let (cck_y, ly_i) = to_chunk(wy);
                let (cck_z, lz_i) = to_chunk(wz);
                let key = (cck_x, cck_y, cck_z);

                let density_field = match store.density_fields.get(&key) {
                    Some(d) => d,
                    None => continue,
                };
                let size = density_field.size;
                let (lx, ly, lz) = (lx_i as usize, ly_i as usize, lz_i as usize);
                if lx >= size || ly >= size || lz >= size { continue; }
                if !density_field.get(lx, ly, lz).material.is_solid() { continue; }

                let air_above = ly + 1 < size && !density_field.get(lx, ly + 1, lz).material.is_solid();
                let air_below = ly > 0 && !density_field.get(lx, ly - 1, lz).material.is_solid();
                let air_xn = lx > 0 && !density_field.get(lx - 1, ly, lz).material.is_solid();
                let air_xp = lx + 1 < size && !density_field.get(lx + 1, ly, lz).material.is_solid();
                let air_zn = lz > 0 && !density_field.get(lx, ly, lz - 1).material.is_solid();
                let air_zp = lz + 1 < size && !density_field.get(lx, ly, lz + 1).material.is_solid();

                let (nx, ny, nz): (f32, f32, f32) = match mushroom_kind {
                    voxel_gen::MushroomKind::TurkeyTail => {
                        if !(air_xn || air_xp || air_zn || air_zp) { continue; }
                        let mut nnx = 0.0f32;
                        let mut nnz = 0.0f32;
                        if air_xn { nnx -= 1.0; }
                        if air_xp { nnx += 1.0; }
                        if air_zn { nnz -= 1.0; }
                        if air_zp { nnz += 1.0; }
                        let len = (nnx * nnx + nnz * nnz).sqrt();
                        if len <= 0.0 { continue; }
                        (nnx / len, 0.0, nnz / len)
                    }
                    voxel_gen::MushroomKind::Foxfire => {
                        if !air_below { continue; }
                        (0.0, -1.0, 0.0)
                    }
                    voxel_gen::MushroomKind::GreenPepe
                    | voxel_gen::MushroomKind::GhostTower => {
                        if !air_above { continue; }
                        (0.0, 1.0, 0.0)
                    }
                };

                // Clustering noise gate. clustering=0 → gate=1 (always pass).
                let gate = if clustering > 0.0 {
                    let n = scatter.sample(
                        (wx as f64 + 0.5) * noise_freq,
                        (wy as f64 + 0.5) * noise_freq,
                        (wz as f64 + 0.5) * noise_freq,
                    ) as f32;
                    if n < noise_thresh { continue; }
                    // Smooth ramp from threshold to 1.0 so cluster edges feather.
                    ((n - noise_thresh) / (1.0 - noise_thresh)).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                if next_unit() >= density_eff * gate { continue; }

                // Sub-voxel jitter on the surface tangent plane (max ±0.4 voxels)
                let ja = (next_unit() - 0.5) * 0.8;
                let jb = (next_unit() - 0.5) * 0.8;
                let (jx, jy, jz) = if ny.abs() > 0.5 {
                    (ja, 0.0, jb) // floor/ceiling — jitter in XZ
                } else if nx.abs() > nz.abs() {
                    (0.0, ja, jb) // wall facing X — jitter in YZ
                } else {
                    (jb, ja, 0.0) // wall facing Z — jitter in XY
                };

                let world_px = wx as f32 + 0.5 + jx + nx * 0.5;
                let world_py = wy as f32 + 0.5 + jy + ny * 0.5;
                let world_pz = wz as f32 + 0.5 + jz + nz * 0.5;

                let mut conflict = false;
                for &(qx, qy, qz) in &placed_world {
                    let ddx = qx - world_px;
                    let ddy = qy - world_py;
                    let ddz = qz - world_pz;
                    if ddx * ddx + ddy * ddy + ddz * ddz < min_spacing2 {
                        conflict = true;
                        break;
                    }
                }
                if conflict { continue; }
                placed_world.push((world_px, world_py, world_pz));

                let scale = kind_cfg.scale_min
                    + (kind_cfg.scale_max - kind_cfg.scale_min) * next_unit();
                let yaw = next_unit() * std::f32::consts::TAU;

                // Convert world voxel position back to chunk-local for storage
                let chunk_origin_x = key.0 * chunk_size;
                let chunk_origin_y = key.1 * chunk_size;
                let chunk_origin_z = key.2 * chunk_size;
                let local_x = world_px - chunk_origin_x as f32;
                let local_y = world_py - chunk_origin_y as f32;
                let local_z = world_pz - chunk_origin_z as f32;

                store.mushroom_placements.entry(key).or_default().push(
                    voxel_gen::MushroomPlacement {
                        x: local_x,
                        y: local_y,
                        z: local_z,
                        normal_x: nx,
                        normal_y: ny,
                        normal_z: nz,
                        scale,
                        yaw,
                        kind: mushroom_kind,
                        anchor_lx: lx.min(u8::MAX as usize) as u8,
                        anchor_ly: ly.min(u8::MAX as usize) as u8,
                        anchor_lz: lz.min(u8::MAX as usize) as u8,
                    },
                );
                affected.insert(key);
            }
        }
    }

    affected
}

/// Erase mushrooms within a sphere. If `kind == 255`, removes every kind
/// inside the sphere; otherwise filters to the specified kind only. Captures
/// an undo snapshot before mutation. Returns the set of chunks where at
/// least one placement was removed.
pub fn erase_mushrooms_brush_sphere(
    store: &mut ChunkStore,
    center_rust: Vec3,
    kind_filter: u8, // 255 = any kind
    radius_voxels: f32,
    config: &GenerationConfig,
) -> std::collections::HashSet<(i32, i32, i32)> {
    let mut affected = std::collections::HashSet::new();
    if radius_voxels <= 0.0 {
        return affected;
    }
    let radius2 = radius_voxels * radius_voxels;

    let chunk_size = config.chunk_size as i32;
    let cx0 = ((center_rust.x - radius_voxels).floor() as i32).div_euclid(chunk_size);
    let cy0 = ((center_rust.y - radius_voxels).floor() as i32).div_euclid(chunk_size);
    let cz0 = ((center_rust.z - radius_voxels).floor() as i32).div_euclid(chunk_size);
    let cx1 = ((center_rust.x + radius_voxels).ceil() as i32).div_euclid(chunk_size);
    let cy1 = ((center_rust.y + radius_voxels).ceil() as i32).div_euclid(chunk_size);
    let cz1 = ((center_rust.z + radius_voxels).ceil() as i32).div_euclid(chunk_size);

    // Snapshot every overlapping chunk that has loaded density (so undo
    // can put back even chunks where nothing got removed — keeps undo idempotent).
    let mut undo_keys: Vec<(i32, i32, i32)> = Vec::new();
    for cz in cz0..=cz1 { for cy in cy0..=cy1 { for cx in cx0..=cx1 {
        if store.density_fields.contains_key(&(cx, cy, cz)) {
            undo_keys.push((cx, cy, cz));
        }
    }}}
    capture_mushroom_undo(store, &undo_keys);

    for cz in cz0..=cz1 { for cy in cy0..=cy1 { for cx in cx0..=cx1 {
        let key = (cx, cy, cz);
        let Some(list) = store.mushroom_placements.get_mut(&key) else { continue };
        let chunk_origin_x = (cx * chunk_size) as f32;
        let chunk_origin_y = (cy * chunk_size) as f32;
        let chunk_origin_z = (cz * chunk_size) as f32;
        let before = list.len();
        list.retain(|p| {
            // Convert placement back to world voxel space for the sphere check.
            let wx = chunk_origin_x + p.x;
            let wy = chunk_origin_y + p.y;
            let wz = chunk_origin_z + p.z;
            let dx = wx - center_rust.x;
            let dy = wy - center_rust.y;
            let dz = wz - center_rust.z;
            let in_sphere = dx * dx + dy * dy + dz * dz <= radius2;
            let kind_match = kind_filter == 255 || (p.kind as u8) == kind_filter;
            !(in_sphere && kind_match) // keep if NOT erased
        });
        if list.len() != before {
            affected.insert(key);
            if list.is_empty() {
                store.mushroom_placements.remove(&key);
            }
        }
    }}}

    affected
}
