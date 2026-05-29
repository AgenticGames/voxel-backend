//! Sphere brushes: paint-material, paint-stress, carve, fill, and the
//! `pub(crate)` batched/`_pub` wrappers consumed by `crystal_anchors.rs`.

use glam::Vec3;
use voxel_core::material::Material;
use voxel_gen::config::GenerationConfig;

use crate::store::ChunkStore;

use super::common::{
    capture_undo_for_range, chunk_range_for_sphere, finalize_brush, local_sphere_bounds,
    BrushOutcome,
};

/// Paint material on solid voxels within a sphere. Air voxels are untouched.
/// Density is preserved (no shape change), only `sample.material` is rewritten.
/// Useful for hand-placing ore deposits and wall variation in caverns.
pub fn paint_material_sphere(
    store: &mut ChunkStore,
    center: Vec3,
    radius: f32,
    target: Material,
    config: &GenerationConfig,
    world_scale: f32,
) -> BrushOutcome {
    let eb = config.effective_bounds();
    let vs = config.voxel_scale();
    let r2 = radius * radius;
    let (lo, hi) = chunk_range_for_sphere(center, radius, eb);

    capture_undo_for_range(store, lo, hi);

    let mut dirty_chunks: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> =
        Vec::new();

    for cz in lo.2..=hi.2 {
        for cy in lo.1..=hi.1 {
            for cx in lo.0..=hi.0 {
                if let Some(density) = store.density_fields.get_mut(&(cx, cy, cz)) {
                    let origin = Vec3::new(cx as f32 * eb, cy as f32 * eb, cz as f32 * eb);
                    let (_, _, lo_x, lo_y, lo_z, hi_x, hi_y, hi_z) =
                        local_sphere_bounds(center, radius, origin, vs, density.size);

                    let mut changed = false;
                    for z in lo_z..hi_z {
                        for y in lo_y..hi_y {
                            for x in lo_x..hi_x {
                                let world_pos = origin
                                    + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                if (world_pos - center).length_squared() > r2 {
                                    continue;
                                }
                                let sample = density.get_mut(x, y, z);
                                if sample.material.is_solid() && sample.material != target {
                                    sample.material = target;
                                    changed = true;
                                }
                            }
                        }
                    }

                    if changed {
                        let expand = config.mine.dirty_expand as usize;
                        let d_min_x = lo_x.saturating_sub(expand);
                        let d_min_y = lo_y.saturating_sub(expand);
                        let d_min_z = lo_z.saturating_sub(expand);
                        let d_max_x = (hi_x + expand).min(density.size - 1);
                        let d_max_y = (hi_y + expand).min(density.size - 1);
                        let d_max_z = (hi_z + expand).min(density.size - 1);
                        dirty_chunks.push((
                            (cx, cy, cz),
                            d_min_x, d_min_y, d_min_z,
                            d_max_x, d_max_y, d_max_z,
                        ));
                    }
                }
            }
        }
    }

    finalize_brush(store, dirty_chunks, config, world_scale)
}

/// PaintStress brush — additively writes into each chunk's painted-stress overlay
/// (`StressField::painted_stress`) inside a sphere. The brush does NOT change
/// density or material, so no remesh is required. The painted layer is
/// preserved across `recalc_stress_region*` passes and is folded into the
/// effective stress that drives collapses during sleep.
///
/// * `amount` — peak per-stroke additive value at the sphere center (typical: 0.2–0.8)
/// * `falloff`
///     - 0 = constant (everything inside the sphere gets the full `amount`)
///     - 1 = linear   (peak at center, 0 at the rim)
///     - 2 = smooth   (cosine smoothstep — easier to layer without hard edges)
/// * `op`
///     - 0 = add (`amount` is added to existing painted value, clamped to `cap`)
///     - 1 = subtract (right-click "lighten" — `amount` is subtracted; clamps to 0)
///     - 2 = clear (zero the painted overlay inside the sphere; ignores `amount`)
/// * `cap` — per-cell ceiling for the painted accumulator (typical: 2.0).
///
/// Returns an empty `BrushOutcome` (no meshes emitted) — the caller still uses
/// it to keep the per-brush "did we make changes" return shape consistent.
pub fn paint_stress_sphere(
    store: &mut ChunkStore,
    center: Vec3,
    radius: f32,
    amount: f32,
    falloff: u8,
    op: u8,
    cap: f32,
    config: &GenerationConfig,
    _world_scale: f32,
) -> BrushOutcome {
    let eb = config.effective_bounds();
    let vs = config.voxel_scale();
    let r = radius.max(0.0);
    let r2 = r * r;
    if r <= 0.0 {
        return BrushOutcome { meshes: Vec::new(), flipped_chunks: Vec::new() };
    }
    let (lo, hi) = chunk_range_for_sphere(center, r, eb);

    capture_undo_for_range(store, lo, hi);

    let chunk_size = config.chunk_size;
    let grid_size = chunk_size + 1;
    let mut touched_chunks: Vec<(i32, i32, i32)> = Vec::new();

    // Field-level disjoint borrow so we can read density (Add/Sub gate) and
    // mutate the stress overlay at the same time. modification_tracker is a
    // third disjoint field, mutated after these locals drop.
    let densities = &store.density_fields;
    let stresses = &mut store.stress_fields;

    for cz in lo.2..=hi.2 {
        for cy in lo.1..=hi.1 {
            for cx in lo.0..=hi.0 {
                let key = (cx, cy, cz);

                // We only paint stress in chunks that have a density field —
                // painting into the void is pointless and the stress consumers
                // index the chunk by the same key.
                let Some(df) = densities.get(&key) else { continue };

                // Lazily initialize the stress field if the chunk has none yet.
                // ChunkStore::insert already does this on first generate, but
                // pre-existing saves or unusual streaming orders can leave it
                // missing — make the brush self-healing.
                let sf = stresses
                    .entry(key)
                    .or_insert_with(|| voxel_core::stress::StressField::new(grid_size));

                let origin = Vec3::new(cx as f32 * eb, cy as f32 * eb, cz as f32 * eb);
                let (_, _, lo_x, lo_y, lo_z, hi_x, hi_y, hi_z) =
                    local_sphere_bounds(center, r, origin, vs, sf.size);

                let mut changed = false;
                for z in lo_z..hi_z {
                    for y in lo_y..hi_y {
                        for x in lo_x..hi_x {
                            let world_pos = origin
                                + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                            let d2 = (world_pos - center).length_squared();
                            if d2 > r2 {
                                continue;
                            }

                            // Stress is a property of rock. Painting into air
                            // would leave dormant values that later "wake up"
                            // when debris settles into the cell, causing the
                            // recollapse loop. Gate Add/Sub on solid; let
                            // Clear pass through so it can erase any legacy
                            // air-cell paint from saves predating this gate.
                            let is_solid = df.get(x, y, z).material.is_solid();
                            if op != 2 && !is_solid {
                                continue;
                            }

                            // Weight by falloff.
                            let w = match falloff {
                                0 => 1.0,
                                1 => {
                                    // Linear: 1 at center, 0 at rim.
                                    let d = d2.sqrt();
                                    (1.0 - (d / r)).max(0.0)
                                }
                                _ => {
                                    // Smoothstep on (1 - d/r): a cosine-ish bell.
                                    let d = d2.sqrt();
                                    let t = (1.0 - (d / r)).clamp(0.0, 1.0);
                                    t * t * (3.0 - 2.0 * t)
                                }
                            };

                            match op {
                                // Add
                                0 => {
                                    let delta = amount * w;
                                    if delta != 0.0 {
                                        sf.add_painted(x, y, z, delta, cap);
                                        changed = true;
                                    }
                                }
                                // Subtract
                                1 => {
                                    let delta = -(amount * w);
                                    if delta != 0.0 {
                                        sf.add_painted(x, y, z, delta, cap);
                                        changed = true;
                                    }
                                }
                                // Clear
                                _ => {
                                    sf.clear_painted(x, y, z);
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                if changed {
                    touched_chunks.push(key);
                }
            }
        }
    }

    if !touched_chunks.is_empty() {
        store.modification_tracker.mark_dirty_many(&touched_chunks);
    }

    // No mesh updates emitted — painted_stress doesn't affect geometry. The UE
    // side can re-`voxel_query_stress` the affected chunks to refresh its
    // overlay (the V/C-key stress preview already drives the same path).
    BrushOutcome { meshes: Vec::new(), flipped_chunks: Vec::new() }
}


/// Carve a sphere — set solid voxels to Air. Same shape as `mining::mine_sphere` but
/// without mined-material accounting and without Laplacian boundary smoothing
/// (smoothing is for player mining; creative carving uses the raw SDF).
pub fn carve_sphere(
    store: &mut ChunkStore,
    center: Vec3,
    radius: f32,
    config: &GenerationConfig,
    world_scale: f32,
) -> BrushOutcome {
    let eb = config.effective_bounds();
    let vs = config.voxel_scale();
    let r2 = radius * radius;
    let (lo, hi) = chunk_range_for_sphere(center, radius, eb);

    capture_undo_for_range(store, lo, hi);

    let mut dirty_chunks: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> =
        Vec::new();

    for cz in lo.2..=hi.2 {
        for cy in lo.1..=hi.1 {
            for cx in lo.0..=hi.0 {
                if let Some(density) = store.density_fields.get_mut(&(cx, cy, cz)) {
                    let origin = Vec3::new(cx as f32 * eb, cy as f32 * eb, cz as f32 * eb);
                    let (_, _, lo_x, lo_y, lo_z, hi_x, hi_y, hi_z) =
                        local_sphere_bounds(center, radius, origin, vs, density.size);

                    let mut changed = false;
                    for z in lo_z..hi_z {
                        for y in lo_y..hi_y {
                            for x in lo_x..hi_x {
                                let world_pos = origin
                                    + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                let dist2 = (world_pos - center).length_squared();
                                if dist2 > r2 {
                                    continue;
                                }
                                let sample = density.get_mut(x, y, z);
                                if sample.material.is_solid() {
                                    let sdf = dist2.sqrt() - radius;
                                    sample.density = sdf.min(sample.density);
                                    if sample.density <= 0.0 {
                                        sample.material = Material::Air;
                                    }
                                    changed = true;
                                }
                            }
                        }
                    }

                    if changed {
                        let expand = config.mine.dirty_expand as usize;
                        let d_min_x = lo_x.saturating_sub(expand);
                        let d_min_y = lo_y.saturating_sub(expand);
                        let d_min_z = lo_z.saturating_sub(expand);
                        let d_max_x = (hi_x + expand).min(density.size - 1);
                        let d_max_y = (hi_y + expand).min(density.size - 1);
                        let d_max_z = (hi_z + expand).min(density.size - 1);
                        dirty_chunks.push((
                            (cx, cy, cz),
                            d_min_x, d_min_y, d_min_z,
                            d_max_x, d_max_y, d_max_z,
                        ));
                    }
                }
            }
        }
    }

    finalize_brush(store, dirty_chunks, config, world_scale)
}

/// Fill a sphere — set air voxels to solid `target` material with an inverse SDF.
/// Density becomes `radius - dist` (positive inside), material becomes `target`.
/// If the voxel is already solid with a different material, the material is overwritten.
pub fn fill_sphere(
    store: &mut ChunkStore,
    center: Vec3,
    radius: f32,
    target: Material,
    config: &GenerationConfig,
    world_scale: f32,
) -> BrushOutcome {
    if !target.is_solid() {
        return BrushOutcome { meshes: Vec::new(), flipped_chunks: Vec::new() };
    }
    let (lo, hi) = chunk_range_for_sphere(center, radius, config.effective_bounds());
    capture_undo_for_range(store, lo, hi);

    let mut dirty_chunks: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> =
        Vec::new();
    fill_sphere_into(store, center, radius, target, config, &mut dirty_chunks);
    finalize_brush(store, dirty_chunks, config, world_scale)
}

/// Batched variant of [`fill_sphere`] for callers that want to fold many sphere
/// fills into a single seam-sync + remesh + undo capture. Pushes per-chunk
/// dirty-range entries into `dirty_chunks` and DOES NOT call `finalize_brush`
/// or capture undo. The caller is responsible for both.
///
/// Crystal Growth Bridge uses this to write 100s of segments under a single
/// finalize at sleep time.
pub(crate) fn fill_sphere_into(
    store: &mut ChunkStore,
    center: Vec3,
    radius: f32,
    target: Material,
    config: &GenerationConfig,
    dirty_chunks: &mut Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)>,
) {
    if !target.is_solid() {
        return;
    }
    let eb = config.effective_bounds();
    let vs = config.voxel_scale();
    let r2 = radius * radius;
    let (lo, hi) = chunk_range_for_sphere(center, radius, eb);

    for cz in lo.2..=hi.2 {
        for cy in lo.1..=hi.1 {
            for cx in lo.0..=hi.0 {
                if let Some(density) = store.density_fields.get_mut(&(cx, cy, cz)) {
                    let origin = Vec3::new(cx as f32 * eb, cy as f32 * eb, cz as f32 * eb);
                    let (_, _, lo_x, lo_y, lo_z, hi_x, hi_y, hi_z) =
                        local_sphere_bounds(center, radius, origin, vs, density.size);

                    let mut changed = false;
                    for z in lo_z..hi_z {
                        for y in lo_y..hi_y {
                            for x in lo_x..hi_x {
                                let world_pos = origin
                                    + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                let dist2 = (world_pos - center).length_squared();
                                if dist2 > r2 {
                                    continue;
                                }
                                let sample = density.get_mut(x, y, z);
                                let inside = radius - dist2.sqrt();
                                if inside > sample.density {
                                    sample.density = inside;
                                    sample.material = target;
                                    changed = true;
                                } else if sample.material.is_solid() && sample.material != target {
                                    sample.material = target;
                                    changed = true;
                                }
                            }
                        }
                    }

                    if changed {
                        let expand = config.mine.dirty_expand as usize;
                        let d_min_x = lo_x.saturating_sub(expand);
                        let d_min_y = lo_y.saturating_sub(expand);
                        let d_min_z = lo_z.saturating_sub(expand);
                        let d_max_x = (hi_x + expand).min(density.size - 1);
                        let d_max_y = (hi_y + expand).min(density.size - 1);
                        let d_max_z = (hi_z + expand).min(density.size - 1);
                        dirty_chunks.push((
                            (cx, cy, cz),
                            d_min_x, d_min_y, d_min_z,
                            d_max_x, d_max_y, d_max_z,
                        ));
                    }
                }
            }
        }
    }
}

/// Apply `finalize_brush` to a batch of dirty entries. Public-to-crate so
/// crystal_anchors can call it from sleep-time bridge growth.
pub(crate) fn finalize_brush_batch(
    store: &mut ChunkStore,
    dirty_chunks: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)>,
    config: &GenerationConfig,
    world_scale: f32,
) -> BrushOutcome {
    finalize_brush(store, dirty_chunks, config, world_scale)
}

/// Capture an undo snapshot for a chunk AABB. Public-to-crate so crystal_anchors
/// can take a single snapshot covering the full bridge bounds.
pub(crate) fn capture_undo_for_range_pub(
    store: &mut ChunkStore,
    lo: (i32, i32, i32),
    hi: (i32, i32, i32),
) {
    capture_undo_for_range(store, lo, hi);
}

/// Helper for computing the chunk-coord range of a sphere — used by Crystal
/// Growth Bridge to bound the undo snapshot before any segment is written.
pub(crate) fn chunk_range_for_sphere_pub(
    center: Vec3,
    radius: f32,
    effective_bounds: f32,
) -> ((i32, i32, i32), (i32, i32, i32)) {
    chunk_range_for_sphere(center, radius, effective_bounds)
}
