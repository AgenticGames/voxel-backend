//! Creative-mode terrain authoring brushes: paint material, sphere carve, sphere fill,
//! tunnel spline carve/fill, and formation placement.
//!
//! All brushes mirror the mining mutation pattern in `mining.rs`:
//!  1. Iterate chunks overlapping the brush region
//!  2. Mutate `DensityField.samples` (density and/or material)
//!  3. Track per-chunk dirty bounds (with `dirty_expand`)
//!  4. Sync boundary density across seams
//!  5. `modification_tracker.mark_dirty_many()` for save persistence
//!  6. `store.remesh_dirty()` to produce updated meshes
//!
//! Brushes are intentionally simpler than mining — no mined-material counts,
//! no Laplacian smoothing, no SDF gradient blending (callers can mine first,
//! then paint, if they want a smoothed border).

use glam::Vec3;
use voxel_core::material::Material;
use voxel_gen::config::GenerationConfig;

use crate::delta::ChunkSnapshot;
use crate::store::{sync_boundary_density, ChunkStore};
use crate::types::ConvertedMesh;

/// One stroke of brush history. Stores pre-state of every chunk in the brush AABB
/// so the operation can be reversed exactly via `apply_to`.
pub struct UndoStroke {
    pub snapshots: Vec<((i32, i32, i32), ChunkSnapshot)>,
}

/// Capture pre-state snapshots for any density-loaded chunks in `[lo..=hi]`,
/// push as a single undo stroke. Bounded by `store.undo_max_depth` — oldest
/// strokes are dropped when full.
///
/// Captures BOTH density+material and (if present) the painted-stress overlay
/// so PaintStress-brush undo round-trips correctly. Chunks with no painted
/// layer still cost ~0 extra bytes (Option<Vec<u8>> stays `None`).
fn capture_undo_for_range(
    store: &mut ChunkStore,
    lo: (i32, i32, i32),
    hi: (i32, i32, i32),
) {
    let mut snapshots = Vec::new();
    for cz in lo.2..=hi.2 {
        for cy in lo.1..=hi.1 {
            for cx in lo.0..=hi.0 {
                let key = (cx, cy, cz);
                if let Some(density) = store.density_fields.get(&key) {
                    let sf = store.stress_fields.get(&key);
                    snapshots.push((key, ChunkSnapshot::from_chunk(density, sf)));
                }
            }
        }
    }
    if snapshots.is_empty() {
        return;
    }
    store.undo_stack.push_back(UndoStroke { snapshots });
    while store.undo_stack.len() > store.undo_max_depth {
        store.undo_stack.pop_front();
    }
}

pub struct BrushOutcome {
    pub meshes: Vec<((i32, i32, i32), ConvertedMesh)>,
    /// Chunks where material/solidity actually flipped (for crystal recompute).
    pub flipped_chunks: Vec<(i32, i32, i32)>,
}

/// Compute the inclusive chunk-coord range overlapping a sphere.
fn chunk_range_for_sphere(center: Vec3, radius: f32, eb: f32) -> ((i32, i32, i32), (i32, i32, i32)) {
    let lo = (
        ((center.x - radius) / eb).floor() as i32,
        ((center.y - radius) / eb).floor() as i32,
        ((center.z - radius) / eb).floor() as i32,
    );
    let hi = (
        ((center.x + radius) / eb).floor() as i32,
        ((center.y + radius) / eb).floor() as i32,
        ((center.z + radius) / eb).floor() as i32,
    );
    (lo, hi)
}

/// Standard "iterate one chunk's voxels overlapping a sphere" loop body.
/// Returns local grid bounds for the dirty rect plus a `changed` flag.
fn local_sphere_bounds(
    center: Vec3,
    radius: f32,
    origin: Vec3,
    vs: f32,
    grid_size: usize,
) -> (Vec3, f32, usize, usize, usize, usize, usize, usize) {
    let grid_center = (center - origin) / vs;
    let grid_radius = radius / vs;
    let lo_x = ((grid_center.x - grid_radius).floor() as i32).max(0) as usize;
    let hi_x = ((grid_center.x + grid_radius).ceil() as usize + 1).min(grid_size);
    let lo_y = ((grid_center.y - grid_radius).floor() as i32).max(0) as usize;
    let hi_y = ((grid_center.y + grid_radius).ceil() as usize + 1).min(grid_size);
    let lo_z = ((grid_center.z - grid_radius).floor() as i32).max(0) as usize;
    let hi_z = ((grid_center.z + grid_radius).ceil() as usize + 1).min(grid_size);
    (grid_center, grid_radius, lo_x, lo_y, lo_z, hi_x, hi_y, hi_z)
}

fn finalize_brush(
    store: &mut ChunkStore,
    mut dirty_chunks: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)>,
    config: &GenerationConfig,
    world_scale: f32,
) -> BrushOutcome {
    let flipped_chunks: Vec<(i32, i32, i32)> = dirty_chunks.iter().map(|&(k, ..)| k).collect();

    let extra_dirty = sync_boundary_density(
        &mut store.density_fields,
        &dirty_chunks,
        config.chunk_size,
    );
    dirty_chunks.extend(extra_dirty);

    let dirty_keys: Vec<_> = dirty_chunks.iter().map(|&(k, ..)| k).collect();
    store.modification_tracker.mark_dirty_many(&dirty_keys);

    let meshes = store.remesh_dirty(&dirty_chunks, config, world_scale);
    BrushOutcome { meshes, flipped_chunks }
}

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

    for cz in lo.2..=hi.2 {
        for cy in lo.1..=hi.1 {
            for cx in lo.0..=hi.0 {
                let key = (cx, cy, cz);

                // We only paint stress in chunks that have a density field —
                // painting into the void is pointless and the stress consumers
                // index the chunk by the same key.
                if !store.density_fields.contains_key(&key) {
                    continue;
                }

                // Lazily initialize the stress field if the chunk has none yet.
                // ChunkStore::insert already does this on first generate, but
                // pre-existing saves or unusual streaming orders can leave it
                // missing — make the brush self-healing.
                let sf = store
                    .stress_fields
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
                                let inside = radius - dist2.sqrt();
                                // Take the max of existing density and the fill SDF —
                                // existing solid stays solid; air becomes solid.
                                if inside > sample.density {
                                    sample.density = inside;
                                    sample.material = target;
                                    changed = true;
                                } else if sample.material.is_solid() && sample.material != target {
                                    // Overwrite the material on already-solid voxels in range
                                    // so a fill brush also paints in one operation.
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

/// Carve (or fill) a tunnel along a polyline of points. Each segment is treated
/// as a capsule of `radius`. If `material` is `None` the tunnel carves; otherwise
/// it fills with that material (useful for "tube of ore" deposits).
///
/// Points are in Rust world coords (already converted from UE).
pub fn tunnel(
    store: &mut ChunkStore,
    points: &[Vec3],
    radius: f32,
    material: Option<Material>,
    config: &GenerationConfig,
    world_scale: f32,
) -> BrushOutcome {
    if points.len() < 2 || radius <= 0.0 {
        return BrushOutcome { meshes: Vec::new(), flipped_chunks: Vec::new() };
    }
    let eb = config.effective_bounds();
    let vs = config.voxel_scale();
    let r2 = radius * radius;

    // Compute bounding box across all segments
    let mut min = points[0];
    let mut max = points[0];
    for p in points.iter().skip(1) {
        min = min.min(*p);
        max = max.max(*p);
    }
    min -= Vec3::splat(radius);
    max += Vec3::splat(radius);

    let lo = (
        (min.x / eb).floor() as i32,
        (min.y / eb).floor() as i32,
        (min.z / eb).floor() as i32,
    );
    let hi = (
        (max.x / eb).floor() as i32,
        (max.y / eb).floor() as i32,
        (max.z / eb).floor() as i32,
    );

    capture_undo_for_range(store, lo, hi);

    // Pre-build segment data (start, dir, length²)
    let segments: Vec<(Vec3, Vec3, f32)> = points
        .windows(2)
        .map(|w| {
            let dir = w[1] - w[0];
            let len_sq = dir.length_squared();
            (w[0], dir, len_sq)
        })
        .collect();

    let dist_to_polyline_sq = |p: Vec3| -> f32 {
        let mut best = f32::INFINITY;
        for &(start, dir, len_sq) in &segments {
            let to_p = p - start;
            let t = if len_sq > 1e-6 {
                (to_p.dot(dir) / len_sq).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let closest = start + dir * t;
            let d2 = (p - closest).length_squared();
            if d2 < best {
                best = d2;
            }
        }
        best
    };

    let mut dirty_chunks: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> =
        Vec::new();

    for cz in lo.2..=hi.2 {
        for cy in lo.1..=hi.1 {
            for cx in lo.0..=hi.0 {
                if let Some(density) = store.density_fields.get_mut(&(cx, cy, cz)) {
                    let origin = Vec3::new(cx as f32 * eb, cy as f32 * eb, cz as f32 * eb);
                    // Local AABB intersection of chunk × tunnel bbox
                    let chunk_min = origin;
                    let chunk_max = origin + Vec3::splat(eb);
                    let isect_min = chunk_min.max(min);
                    let isect_max = chunk_max.min(max);
                    if isect_min.cmpgt(isect_max).any() {
                        continue;
                    }
                    let lo_x = (((isect_min.x - origin.x) / vs).floor() as i32).max(0) as usize;
                    let hi_x = (((isect_max.x - origin.x) / vs).ceil() as usize + 1).min(density.size);
                    let lo_y = (((isect_min.y - origin.y) / vs).floor() as i32).max(0) as usize;
                    let hi_y = (((isect_max.y - origin.y) / vs).ceil() as usize + 1).min(density.size);
                    let lo_z = (((isect_min.z - origin.z) / vs).floor() as i32).max(0) as usize;
                    let hi_z = (((isect_max.z - origin.z) / vs).ceil() as usize + 1).min(density.size);

                    let mut changed = false;
                    for z in lo_z..hi_z {
                        for y in lo_y..hi_y {
                            for x in lo_x..hi_x {
                                let world_pos = origin
                                    + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                let d2 = dist_to_polyline_sq(world_pos);
                                if d2 > r2 {
                                    continue;
                                }
                                let sample = density.get_mut(x, y, z);
                                match material {
                                    None => {
                                        // Carve: only affect solid voxels
                                        if sample.material.is_solid() {
                                            let sdf = d2.sqrt() - radius;
                                            sample.density = sdf.min(sample.density);
                                            if sample.density <= 0.0 {
                                                sample.material = Material::Air;
                                            }
                                            changed = true;
                                        }
                                    }
                                    Some(target) => {
                                        let inside = radius - d2.sqrt();
                                        if inside > sample.density {
                                            sample.density = inside;
                                            sample.material = target;
                                            changed = true;
                                        }
                                    }
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

/// Place a single hand-authored formation at `center`. The formation type maps
/// to a primitive shape: stalactite (cone tip-down), stalagmite (cone tip-up),
/// column (capsule), drapery (vertical fin), flowstone (mound), shield (disc),
/// rimstone-dam (curved wall). All formations are baked as voxel writes so they
/// persist via the standard chunk-snapshot save path.
///
/// `formation_type`: 0=Stalactite, 1=Stalagmite, 2=Column, 3=Drapery, 4=Flowstone,
///                   5=Shield, 6=RimstoneDam
/// `height`/`radius` in Rust world units; orientation is implicit per type.
pub fn place_formation(
    store: &mut ChunkStore,
    center: Vec3,
    formation_type: u8,
    height: f32,
    radius: f32,
    material: Material,
    config: &GenerationConfig,
    world_scale: f32,
) -> BrushOutcome {
    if height <= 0.0 || radius <= 0.0 || !material.is_solid() {
        return BrushOutcome { meshes: Vec::new(), flipped_chunks: Vec::new() };
    }

    // Half-extents of the formation's AABB (in Rust world units).
    let (half, base_offset) = match formation_type {
        0 => (Vec3::new(radius, height * 0.5, radius), Vec3::new(0.0, -height * 0.5, 0.0)), // stalactite hangs below center
        1 => (Vec3::new(radius, height * 0.5, radius), Vec3::new(0.0,  height * 0.5, 0.0)), // stalagmite rises above center
        2 => (Vec3::new(radius, height * 0.5, radius), Vec3::ZERO),                          // column centered
        3 => (Vec3::new(radius, height * 0.5, radius * 0.50), Vec3::ZERO),                   // drapery wavy fin (Z extent = thin half ±wave amp)
        4 => (Vec3::new(radius, height, radius), Vec3::new(0.0, height * 0.5, 0.0)),         // flowstone mound on floor
        5 => (Vec3::new(radius, height * 0.5, radius), Vec3::ZERO),                          // shield disc
        6 => (Vec3::new(radius, height * 0.5, radius), Vec3::ZERO),                          // rimstone arc
        _ => return BrushOutcome { meshes: Vec::new(), flipped_chunks: Vec::new() },
    };

    let aabb_center = center + base_offset;
    let aabb_min = aabb_center - half;
    let aabb_max = aabb_center + half;

    let eb = config.effective_bounds();
    let vs = config.voxel_scale();
    let lo = (
        (aabb_min.x / eb).floor() as i32,
        (aabb_min.y / eb).floor() as i32,
        (aabb_min.z / eb).floor() as i32,
    );
    let hi = (
        (aabb_max.x / eb).floor() as i32,
        (aabb_max.y / eb).floor() as i32,
        (aabb_max.z / eb).floor() as i32,
    );

    capture_undo_for_range(store, lo, hi);

    // SDF for this formation type, evaluated at world position `p`.
    // Returns "inside-ness" (positive = solid). Mirrors the carve_sphere/fill_sphere
    // sign convention.
    let formation_sdf = |p: Vec3| -> f32 {
        let local = p - aabb_center;
        match formation_type {
            // Stalactite (tip down): cone with apex at bottom
            0 => {
                let h = height;
                let t = ((local.y + h * 0.5) / h).clamp(0.0, 1.0); // 0=tip, 1=base
                let r_at_y = radius * t;
                let dxz = (local.x * local.x + local.z * local.z).sqrt();
                let inside_radial = r_at_y - dxz;
                let inside_y = (h * 0.5) - local.y.abs();
                inside_radial.min(inside_y)
            }
            // Stalagmite (tip up): cone with apex at top
            1 => {
                let h = height;
                let t = (1.0 - (local.y + h * 0.5) / h).clamp(0.0, 1.0); // 0=tip, 1=base
                let r_at_y = radius * t;
                let dxz = (local.x * local.x + local.z * local.z).sqrt();
                let inside_radial = r_at_y - dxz;
                let inside_y = (h * 0.5) - local.y.abs();
                inside_radial.min(inside_y)
            }
            // Column: cylinder
            2 => {
                let dxz = (local.x * local.x + local.z * local.z).sqrt();
                let inside_radial = radius - dxz;
                let inside_y = (height * 0.5) - local.y.abs();
                inside_radial.min(inside_y)
            }
            // Drapery: wavy thin Z-fin (X = wide, Z = thin, Y = tall) — the fin
            // undulates side-to-side along its X axis so it actually reads as a
            // hanging cave curtain instead of a flat slab. Wave wavelength scales
            // with radius so the curtain has ~3 humps regardless of size.
            3 => {
                let wave_freq = std::f32::consts::TAU * 1.5 / radius.max(1.0); // ~3 humps across
                let wave_amp  = radius * 0.20;                                 // displacement strength
                let z_offset  = (local.x * wave_freq).sin() * wave_amp;
                let inside_x  = radius - local.x.abs();
                let inside_z  = (radius * 0.25) - (local.z - z_offset).abs();
                let inside_y  = (height * 0.5) - local.y.abs();
                inside_x.min(inside_z).min(inside_y)
            }
            // Flowstone: half-ellipsoid mound (rises from floor)
            4 => {
                let nx = local.x / radius;
                let ny = (local.y / height).max(0.0);
                let nz = local.z / radius;
                let r = (nx * nx + ny * ny + nz * nz).sqrt();
                if local.y < 0.0 { -1.0 } else { (1.0 - r) * radius }
            }
            // Shield: oblate disc (Y thin, XZ wide). Earlier version had
            // `inside_y * 4.0` which was meant to "exaggerate flatness" but
            // multiplying by >1 made the Y constraint LESS limiting — the
            // result was a cylinder, not a disc. Compress the Y half-extent
            // to ~10% of `height` so the SDF is actually disc-shaped.
            5 => {
                let dxz = (local.x * local.x + local.z * local.z).sqrt();
                let inside_radial = radius - dxz;
                let disc_half_h   = (height * 0.5) * 0.2; // 20% of half-height = thin disc
                let inside_y      = disc_half_h - local.y.abs();
                inside_radial.min(inside_y)
            }
            // Rimstone dam: torus-arc wall
            6 => {
                let dxz = (local.x * local.x + local.z * local.z).sqrt();
                let dist_from_ring = (dxz - radius).abs();
                let inside_thickness = (radius * 0.25) - dist_from_ring;
                let inside_y = (height * 0.5) - local.y.abs();
                inside_thickness.min(inside_y)
            }
            _ => -1.0,
        }
    };

    let mut dirty_chunks: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> =
        Vec::new();

    for cz in lo.2..=hi.2 {
        for cy in lo.1..=hi.1 {
            for cx in lo.0..=hi.0 {
                if let Some(density) = store.density_fields.get_mut(&(cx, cy, cz)) {
                    let origin = Vec3::new(cx as f32 * eb, cy as f32 * eb, cz as f32 * eb);
                    let chunk_min = origin;
                    let chunk_max = origin + Vec3::splat(eb);
                    let isect_min = chunk_min.max(aabb_min);
                    let isect_max = chunk_max.min(aabb_max);
                    if isect_min.cmpgt(isect_max).any() {
                        continue;
                    }
                    let lo_x = (((isect_min.x - origin.x) / vs).floor() as i32).max(0) as usize;
                    let hi_x = (((isect_max.x - origin.x) / vs).ceil() as usize + 1).min(density.size);
                    let lo_y = (((isect_min.y - origin.y) / vs).floor() as i32).max(0) as usize;
                    let hi_y = (((isect_max.y - origin.y) / vs).ceil() as usize + 1).min(density.size);
                    let lo_z = (((isect_min.z - origin.z) / vs).floor() as i32).max(0) as usize;
                    let hi_z = (((isect_max.z - origin.z) / vs).ceil() as usize + 1).min(density.size);

                    let mut changed = false;
                    for z in lo_z..hi_z {
                        for y in lo_y..hi_y {
                            for x in lo_x..hi_x {
                                let world_pos = origin
                                    + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                let inside = formation_sdf(world_pos);
                                if inside <= 0.0 {
                                    continue;
                                }
                                let sample = density.get_mut(x, y, z);
                                if inside > sample.density {
                                    sample.density = inside;
                                    sample.material = material;
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

/// Formation Stamp brush: places only stalactites, stalagmites, and
/// cave shields inside the brush sphere. Drapery, columns/mega-columns,
/// flowstone, rimstone dams, and cauldrons are excluded so the user can
/// paint pure decoration without changing the cave's macro silhouette.
///
/// Shield params are boosted relative to worldgen defaults so the rare
/// shield shape is actually visible from a single brush click and the
/// disk has clear tilt + a hanging stalactite for visual interest.
///
/// Materials are picked from each surface's natural host rock. Undo
/// captures the pre-state of every overlapping chunk so the user can
/// iterate vibes.
pub fn random_formations_brush(
    store: &mut ChunkStore,
    center: Vec3,
    radius: f32,
    seed: u64,
    config: &GenerationConfig,
    world_scale: f32,
) -> BrushOutcome {
    if radius <= 0.0 || !config.formations.enabled {
        return BrushOutcome { meshes: Vec::new(), flipped_chunks: Vec::new() };
    }

    let eb = config.effective_bounds();
    let (lo, hi) = chunk_range_for_sphere(center, radius, eb);

    capture_undo_for_range(store, lo, hi);

    // Margin around the sphere AABB to capture formation writes that extend
    // past the anchor (mega-column r_ceil up to ~base_radius+2 ≈ 10 cells,
    // stalactite cones up to ~length cells). Use a generous fixed margin.
    let dirty_margin: usize = 12;

    let mut dirty_chunks: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> =
        Vec::new();

    let aabb_min = center - Vec3::splat(radius);
    let aabb_max = center + Vec3::splat(radius);

    // Brush-specific FormationConfig override. Worldgen's defaults make
    // shields a 0.8%-per-wall-surface event — over a small brush region you'd
    // often get zero. Boosted chance + larger radii + steeper tilt range
    // turn each click into a visible shield cluster on whatever wall
    // surfaces the spatial filter picks up, with the hanging stalactite
    // always present so they read as "shields with drips" not flat coins.
    let mut brush_formations = config.formations.clone();
    brush_formations.shield_chance            = brush_formations.shield_chance.max(0.4);
    brush_formations.shield_radius_min        = brush_formations.shield_radius_min.max(2.5);
    brush_formations.shield_radius_max        = brush_formations.shield_radius_max.max(5.0);
    brush_formations.shield_max_tilt          = brush_formations.shield_max_tilt.max(60.0);
    brush_formations.shield_stalactite_chance = 1.0;

    let allowed = voxel_gen::formations::FORMATION_STALACTITE
        | voxel_gen::formations::FORMATION_STALAGMITE
        | voxel_gen::formations::FORMATION_SHIELD;

    for cz in lo.2..=hi.2 {
        for cy in lo.1..=hi.1 {
            for cx in lo.0..=hi.0 {
                let chunk_coord = (cx, cy, cz);
                let Some(density) = store.density_fields.get_mut(&chunk_coord) else {
                    continue;
                };

                let origin = Vec3::new(cx as f32 * eb, cy as f32 * eb, cz as f32 * eb);

                // Per-call randomized chunk_seed so re-stamping the same area
                // gives different placements. Mix the user-provided seed with
                // chunk coords for spatial coherence within a single click.
                let chunk_seed = seed
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add((cx as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
                    .wrapping_add((cy as u64).wrapping_mul(0x94D0_49BB_1331_11EB))
                    .wrapping_add((cz as u64).wrapping_mul(0x6C62_272E_07BB_0142));

                let _seeds = voxel_gen::formations::place_formations_filtered(
                    density,
                    &brush_formations,
                    origin,
                    config.seed,
                    chunk_seed,
                    chunk_coord,
                    Some((center, radius)),
                    allowed,
                );

                // Compute dirty rect = intersection(chunk AABB, brush sphere
                // AABB) expanded by `dirty_margin` cells to cover formation
                // writes that extend past the anchor.
                let chunk_min = origin;
                let chunk_max = origin + Vec3::splat(eb);
                let isect_min = chunk_min.max(aabb_min);
                let isect_max = chunk_max.min(aabb_max);
                if isect_min.cmpgt(isect_max).any() {
                    continue;
                }
                let vs = config.voxel_scale();
                let lo_x = (((isect_min.x - origin.x) / vs).floor() as i32 - dirty_margin as i32)
                    .max(0) as usize;
                let lo_y = (((isect_min.y - origin.y) / vs).floor() as i32 - dirty_margin as i32)
                    .max(0) as usize;
                let lo_z = (((isect_min.z - origin.z) / vs).floor() as i32 - dirty_margin as i32)
                    .max(0) as usize;
                let hi_x = (((isect_max.x - origin.x) / vs).ceil() as usize + dirty_margin + 1)
                    .min(density.size);
                let hi_y = (((isect_max.y - origin.y) / vs).ceil() as usize + dirty_margin + 1)
                    .min(density.size);
                let hi_z = (((isect_max.z - origin.z) / vs).ceil() as usize + dirty_margin + 1)
                    .min(density.size);

                dirty_chunks.push((chunk_coord, lo_x, lo_y, lo_z, hi_x, hi_y, hi_z));
            }
        }
    }

    finalize_brush(store, dirty_chunks, config, world_scale)
}

/// Cavern Stamp brush — chunk-snapped cave generator. Runs the worldgen
/// cave-carving phases (worms ± lava tubes/rivers) on a NxMxK chunk-aligned
/// region, optionally followed by pools + formations decoration. Each click
/// uses a fresh seed so re-stamping the same area gives a different cavern
/// layout. Worms carve additively (existing user edits in the chunks
/// survive — only solid → cave transitions happen).
///
/// `chunk_origin`: low corner chunk (x, y, z) of the brush region.
/// `extent`: number of chunks in each axis (NxMxK), each ≥ 1.
/// `decorate`: also run pools + formations after carving.
/// `fluids`: also run lava tubes + rivers.
/// `seed`: drives all randomness for this stamp; same seed + same input = same result.
pub fn cavern_stamp_brush(
    store: &mut ChunkStore,
    chunk_origin: (i32, i32, i32),
    extent: (u8, u8, u8),
    decorate: bool,
    fluids: bool,
    seed: u64,
    config: &GenerationConfig,
    world_scale: f32,
) -> BrushOutcome {
    if extent.0 == 0 || extent.1 == 0 || extent.2 == 0 {
        return BrushOutcome { meshes: Vec::new(), flipped_chunks: Vec::new() };
    }

    // 1. Build the coord list and capture undo for it
    let mut coords: Vec<(i32, i32, i32)> = Vec::with_capacity(
        extent.0 as usize * extent.1 as usize * extent.2 as usize,
    );
    for dz in 0..extent.2 as i32 {
        for dy in 0..extent.1 as i32 {
            for dx in 0..extent.0 as i32 {
                coords.push((
                    chunk_origin.0 + dx,
                    chunk_origin.1 + dy,
                    chunk_origin.2 + dz,
                ));
            }
        }
    }
    let lo = chunk_origin;
    let hi = (
        chunk_origin.0 + extent.0 as i32 - 1,
        chunk_origin.1 + extent.1 as i32 - 1,
        chunk_origin.2 + extent.2 as i32 - 1,
    );
    capture_undo_for_range(store, lo, hi);

    // 2. Run cavern carving on the brush coords (modifies store.density_fields
    //    in place but only entries whose key is in `coords`).
    voxel_gen::region_gen::carve_caverns_into_existing(
        &coords,
        &mut store.density_fields,
        config,
        seed,
        decorate,
        fluids,
    );

    // 3. Mark every brush chunk dirty in full. Cavern carving can touch any
    //    cell in any of these chunks, so re-extract the entire mesh per chunk.
    let mut dirty_chunks: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> =
        Vec::with_capacity(coords.len());
    for c in &coords {
        if let Some(density) = store.density_fields.get(c) {
            let s = density.size;
            dirty_chunks.push((*c, 0, 0, 0, s, s, s));
        }
    }

    finalize_brush(store, dirty_chunks, config, world_scale)
}

// =====================================================================
// Fluid brush helpers
// =====================================================================
//
// These compute lists of (chunk, local-x/y/z) cells that should receive a
// fluid placement event. They DON'T touch the fluid system directly — the
// worker handler iterates the returned list and sends `FluidEvent::AddFluid`
// events to the fluid simulation thread (one per cell).
//
// `bottom_half_only=true` mirrors the existing `MineAndFillFluid` pattern:
// only fill cells below `center.y` so a pool sits at the bottom of a carved
// basin instead of completely flooding it.

#[derive(Debug, Clone, Copy)]
pub struct FluidPlacement {
    pub chunk: (i32, i32, i32),
    pub x: u8,
    pub y: u8,
    pub z: u8,
}

/// Collect air cells inside a sphere region (Rust world coords).
pub fn collect_fluid_cells_in_sphere(
    store: &ChunkStore,
    center: Vec3,
    radius: f32,
    bottom_half_only: bool,
    config: &GenerationConfig,
) -> Vec<FluidPlacement> {
    let eb = config.effective_bounds();
    let vs = config.voxel_scale();
    let r2 = radius * radius;
    let (lo, hi) = chunk_range_for_sphere(center, radius, eb);
    let mut out = Vec::new();
    for cz in lo.2..=hi.2 {
        for cy in lo.1..=hi.1 {
            for cx in lo.0..=hi.0 {
                if let Some(density) = store.density_fields.get(&(cx, cy, cz)) {
                    let origin = Vec3::new(cx as f32 * eb, cy as f32 * eb, cz as f32 * eb);
                    let (_, _, lo_x, lo_y, lo_z, hi_x, hi_y, hi_z) =
                        local_sphere_bounds(center, radius, origin, vs, density.size);
                    for z in lo_z..hi_z {
                        for y in lo_y..hi_y {
                            for x in lo_x..hi_x {
                                let world_pos = origin
                                    + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                let d2 = (world_pos - center).length_squared();
                                if d2 > r2 { continue; }
                                if bottom_half_only && world_pos.y >= center.y { continue; }
                                let s = density.get(x, y, z);
                                // Air cell — eligible for fluid placement.
                                if !s.material.is_solid() && s.density <= 0.0 {
                                    out.push(FluidPlacement {
                                        chunk: (cx, cy, cz),
                                        x: x as u8, y: y as u8, z: z as u8,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Collect air cells inside an axis-aligned box.
pub fn collect_fluid_cells_in_box(
    store: &ChunkStore,
    center: Vec3,
    half_ext: Vec3,
    config: &GenerationConfig,
) -> Vec<FluidPlacement> {
    let eb = config.effective_bounds();
    let vs = config.voxel_scale();
    let aabb_min = center - half_ext;
    let aabb_max = center + half_ext;
    let lo = (
        (aabb_min.x / eb).floor() as i32,
        (aabb_min.y / eb).floor() as i32,
        (aabb_min.z / eb).floor() as i32,
    );
    let hi = (
        (aabb_max.x / eb).floor() as i32,
        (aabb_max.y / eb).floor() as i32,
        (aabb_max.z / eb).floor() as i32,
    );
    let mut out = Vec::new();
    for cz in lo.2..=hi.2 {
        for cy in lo.1..=hi.1 {
            for cx in lo.0..=hi.0 {
                if let Some(density) = store.density_fields.get(&(cx, cy, cz)) {
                    let origin = Vec3::new(cx as f32 * eb, cy as f32 * eb, cz as f32 * eb);
                    let chunk_min = origin;
                    let chunk_max = origin + Vec3::splat(eb);
                    let isect_min = chunk_min.max(aabb_min);
                    let isect_max = chunk_max.min(aabb_max);
                    if isect_min.cmpgt(isect_max).any() { continue; }
                    let lo_x = (((isect_min.x - origin.x) / vs).floor() as i32).max(0) as usize;
                    let hi_x = (((isect_max.x - origin.x) / vs).ceil() as usize + 1).min(density.size);
                    let lo_y = (((isect_min.y - origin.y) / vs).floor() as i32).max(0) as usize;
                    let hi_y = (((isect_max.y - origin.y) / vs).ceil() as usize + 1).min(density.size);
                    let lo_z = (((isect_min.z - origin.z) / vs).floor() as i32).max(0) as usize;
                    let hi_z = (((isect_max.z - origin.z) / vs).ceil() as usize + 1).min(density.size);
                    for z in lo_z..hi_z {
                        for y in lo_y..hi_y {
                            for x in lo_x..hi_x {
                                let world_pos = origin
                                    + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                let local = world_pos - center;
                                if local.x.abs() > half_ext.x || local.y.abs() > half_ext.y || local.z.abs() > half_ext.z {
                                    continue;
                                }
                                let s = density.get(x, y, z);
                                if !s.material.is_solid() && s.density <= 0.0 {
                                    out.push(FluidPlacement {
                                        chunk: (cx, cy, cz),
                                        x: x as u8, y: y as u8, z: z as u8,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Collect air cells inside a capsule-chain (river/spline) region.
pub fn collect_fluid_cells_in_capsule_chain(
    store: &ChunkStore,
    points: &[Vec3],
    radius: f32,
    config: &GenerationConfig,
) -> Vec<FluidPlacement> {
    if points.len() < 2 || radius <= 0.0 {
        return Vec::new();
    }
    let eb = config.effective_bounds();
    let vs = config.voxel_scale();
    let r2 = radius * radius;

    let mut min = points[0];
    let mut max = points[0];
    for p in points.iter().skip(1) {
        min = min.min(*p);
        max = max.max(*p);
    }
    min -= Vec3::splat(radius);
    max += Vec3::splat(radius);

    let lo = (
        (min.x / eb).floor() as i32,
        (min.y / eb).floor() as i32,
        (min.z / eb).floor() as i32,
    );
    let hi = (
        (max.x / eb).floor() as i32,
        (max.y / eb).floor() as i32,
        (max.z / eb).floor() as i32,
    );

    let segments: Vec<(Vec3, Vec3, f32)> = points
        .windows(2)
        .map(|w| {
            let dir = w[1] - w[0];
            let len_sq = dir.length_squared();
            (w[0], dir, len_sq)
        })
        .collect();

    let dist_to_polyline_sq = |p: Vec3| -> f32 {
        let mut best = f32::INFINITY;
        for &(start, dir, len_sq) in &segments {
            let to_p = p - start;
            let t = if len_sq > 1e-6 {
                (to_p.dot(dir) / len_sq).clamp(0.0, 1.0)
            } else { 0.0 };
            let closest = start + dir * t;
            let d2 = (p - closest).length_squared();
            if d2 < best { best = d2; }
        }
        best
    };

    let mut out = Vec::new();
    for cz in lo.2..=hi.2 {
        for cy in lo.1..=hi.1 {
            for cx in lo.0..=hi.0 {
                if let Some(density) = store.density_fields.get(&(cx, cy, cz)) {
                    let origin = Vec3::new(cx as f32 * eb, cy as f32 * eb, cz as f32 * eb);
                    let chunk_min = origin;
                    let chunk_max = origin + Vec3::splat(eb);
                    let isect_min = chunk_min.max(min);
                    let isect_max = chunk_max.min(max);
                    if isect_min.cmpgt(isect_max).any() { continue; }
                    let lo_x = (((isect_min.x - origin.x) / vs).floor() as i32).max(0) as usize;
                    let hi_x = (((isect_max.x - origin.x) / vs).ceil() as usize + 1).min(density.size);
                    let lo_y = (((isect_min.y - origin.y) / vs).floor() as i32).max(0) as usize;
                    let hi_y = (((isect_max.y - origin.y) / vs).ceil() as usize + 1).min(density.size);
                    let lo_z = (((isect_min.z - origin.z) / vs).floor() as i32).max(0) as usize;
                    let hi_z = (((isect_max.z - origin.z) / vs).ceil() as usize + 1).min(density.size);
                    for z in lo_z..hi_z {
                        for y in lo_y..hi_y {
                            for x in lo_x..hi_x {
                                let world_pos = origin
                                    + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                if dist_to_polyline_sq(world_pos) > r2 { continue; }
                                let s = density.get(x, y, z);
                                if !s.material.is_solid() && s.density <= 0.0 {
                                    out.push(FluidPlacement {
                                        chunk: (cx, cy, cz),
                                        x: x as u8, y: y as u8, z: z as u8,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

// (removed) The cosmetic-stream brush was replaced by a real fluid-sim feature:
// bounded sources via `max_flow_dist` on FluidCell. See voxel-fluid for impl.

#[cfg(any())] // disabled — kept only for archive
fn collect_bounded_stream_cells_archived() {
/// Bounded stream brush: walks a polyline and paints fluid cells with explicit
/// levels along the path.
///
/// `full_dist`: distance along path (Rust units) where every cell gets level=1.0.
/// `taper_dist`: distance past `full_dist` where level ramps linearly 1.0 → 0.0.
/// Cells past `full_dist + taper_dist` are skipped.
///
/// `head_source_dist`: cells whose along-spline distance is < this become
/// `is_source = true` so the head doesn't drain. Set to e.g. 1 voxel to anchor
/// the spring at the very start; set larger to keep more of the stream as a
/// permanent spring.
pub fn collect_bounded_stream_cells(
    store: &ChunkStore,
    points: &[Vec3],
    radius: f32,
    full_dist: f32,
    taper_dist: f32,
    head_source_dist: f32,
    config: &GenerationConfig,
) -> Vec<FluidStreamPlacement> {
    if points.len() < 2 || radius <= 0.0 || (full_dist + taper_dist) <= 0.0 {
        return Vec::new();
    }
    let eb = config.effective_bounds();
    let vs = config.voxel_scale();
    let r2 = radius * radius;
    let total_dist = full_dist + taper_dist;

    // Pre-compute segment data with cumulative distance from start.
    // segs[i] = (start, dir, len, len², cum_dist_at_start)
    let mut segments: Vec<(Vec3, Vec3, f32, f32, f32)> = Vec::with_capacity(points.len() - 1);
    let mut cum = 0.0f32;
    for w in points.windows(2) {
        let dir = w[1] - w[0];
        let len_sq = dir.length_squared();
        let len = len_sq.sqrt();
        segments.push((w[0], dir, len, len_sq, cum));
        cum += len;
    }
    let total_path_len = cum;

    // Bounding box of (path × radius), capped at the total stream reach
    // so we don't iterate cells well past the taper tail.
    let mut min = points[0];
    let mut max = points[0];
    for p in points.iter().skip(1) { min = min.min(*p); max = max.max(*p); }
    min -= Vec3::splat(radius);
    max += Vec3::splat(radius);

    let lo = (
        (min.x / eb).floor() as i32,
        (min.y / eb).floor() as i32,
        (min.z / eb).floor() as i32,
    );
    let hi = (
        (max.x / eb).floor() as i32,
        (max.y / eb).floor() as i32,
        (max.z / eb).floor() as i32,
    );

    // For a given world position, returns (closest_distance², along_spline_distance).
    let closest_along = |p: Vec3| -> (f32, f32) {
        let mut best_d2 = f32::INFINITY;
        let mut best_along = 0.0f32;
        for &(start, dir, len, len_sq, cum_start) in &segments {
            let to_p = p - start;
            let t = if len_sq > 1e-6 {
                (to_p.dot(dir) / len_sq).clamp(0.0, 1.0)
            } else { 0.0 };
            let closest = start + dir * t;
            let d2 = (p - closest).length_squared();
            if d2 < best_d2 {
                best_d2 = d2;
                best_along = cum_start + t * len;
            }
        }
        (best_d2, best_along)
    };

    let mut out = Vec::new();
    for cz in lo.2..=hi.2 {
        for cy in lo.1..=hi.1 {
            for cx in lo.0..=hi.0 {
                if let Some(density) = store.density_fields.get(&(cx, cy, cz)) {
                    let origin = Vec3::new(cx as f32 * eb, cy as f32 * eb, cz as f32 * eb);
                    let chunk_min = origin;
                    let chunk_max = origin + Vec3::splat(eb);
                    let isect_min = chunk_min.max(min);
                    let isect_max = chunk_max.min(max);
                    if isect_min.cmpgt(isect_max).any() { continue; }
                    let lo_x = (((isect_min.x - origin.x) / vs).floor() as i32).max(0) as usize;
                    let hi_x = (((isect_max.x - origin.x) / vs).ceil() as usize + 1).min(density.size);
                    let lo_y = (((isect_min.y - origin.y) / vs).floor() as i32).max(0) as usize;
                    let hi_y = (((isect_max.y - origin.y) / vs).ceil() as usize + 1).min(density.size);
                    let lo_z = (((isect_min.z - origin.z) / vs).floor() as i32).max(0) as usize;
                    let hi_z = (((isect_max.z - origin.z) / vs).ceil() as usize + 1).min(density.size);

                    for z in lo_z..hi_z {
                        for y in lo_y..hi_y {
                            for x in lo_x..hi_x {
                                let world_pos = origin
                                    + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                let (d2, along) = closest_along(world_pos);
                                if d2 > r2 { continue; }
                                if along > total_dist { continue; }
                                if along > total_path_len + radius { continue; }

                                // Level: 1.0 in full zone, linear ramp in taper zone.
                                let level = if along <= full_dist {
                                    1.0
                                } else {
                                    let t = (along - full_dist) / taper_dist.max(1e-6);
                                    (1.0 - t).clamp(0.0, 1.0)
                                };

                                // Only place into air cells.
                                let s = density.get(x, y, z);
                                if !s.material.is_solid() && s.density <= 0.0 {
                                    out.push(FluidStreamPlacement {
                                        chunk: (cx, cy, cz),
                                        x: x as u8, y: y as u8, z: z as u8,
                                        level,
                                        is_source: along <= head_source_dist,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}
} // end #[cfg(any())] archive block

/// Pop the most-recent undo stroke, restore each captured chunk's density
/// in-place, and return the dirty rect for each restored chunk so the caller
/// can route them through the standard remesh pipeline.
///
/// Returns `None` if the undo stack was empty.
pub fn apply_undo(
    store: &mut ChunkStore,
    config: &GenerationConfig,
    world_scale: f32,
) -> Option<BrushOutcome> {
    let stroke = store.undo_stack.pop_back()?;
    if stroke.snapshots.is_empty() {
        return None;
    }
    let mut dirty_chunks: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> =
        Vec::with_capacity(stroke.snapshots.len());

    for (key, snapshot) in &stroke.snapshots {
        if let Some(density) = store.density_fields.get_mut(key) {
            snapshot.apply_to(density);
            // Whole-chunk dirty rect — undo restores everything.
            let s_max = density.size - 1;
            dirty_chunks.push((*key, 0, 0, 0, s_max, s_max, s_max));
        }
        // Restore the painted-stress overlay (no-op for non-PaintStress strokes
        // — their snapshots have painted_stress: None and that just wipes the
        // overlay back to empty, which is the pre-state if it was empty before).
        if let Some(sf) = store.stress_fields.get_mut(key) {
            snapshot.apply_painted_stress_to(sf);
        }
    }

    if dirty_chunks.is_empty() {
        return None;
    }

    // Mark dirty for save persistence (the restored chunks are still "modified
    // relative to the procedural baseline").
    let dirty_keys: Vec<_> = dirty_chunks.iter().map(|&(k, ..)| k).collect();
    store.modification_tracker.mark_dirty_many(&dirty_keys);

    let flipped_chunks = dirty_keys.clone();
    let meshes = store.remesh_dirty(&dirty_chunks, config, world_scale);
    Some(BrushOutcome { meshes, flipped_chunks })
}

// =====================================================================
// New brushes: box, cylinder, smooth, noise
// =====================================================================

/// Axis-aligned-or-yawed box brush. `op`: 0=paint material, 1=carve, 2=fill.
/// `half_ext` is the half-extent in each axis (Rust world units).
/// `yaw_rad`: rotation around the Rust Y (vertical) axis in radians.
/// 0.0 = AABB (legacy behavior). Non-zero = OBB rotated horizontally.
pub fn box_brush(
    store: &mut ChunkStore,
    center: Vec3,
    half_ext: Vec3,
    yaw_rad: f32,
    op: u8,
    target: Material,
    config: &GenerationConfig,
    world_scale: f32,
) -> BrushOutcome {
    if half_ext.x <= 0.0 || half_ext.y <= 0.0 || half_ext.z <= 0.0 {
        return BrushOutcome { meshes: Vec::new(), flipped_chunks: Vec::new() };
    }
    if op == 2 && !target.is_solid() {
        return BrushOutcome { meshes: Vec::new(), flipped_chunks: Vec::new() };
    }

    let eb = config.effective_bounds();
    let vs = config.voxel_scale();

    // Yawed-OBB world-space bounding AABB. Yaw rotates the OBB's footprint
    // (XZ plane) so the AABB grows along both axes proportional to sin/cos.
    // Vertical extent (Y) is unchanged — yaw is around Y.
    let cos_y = yaw_rad.cos().abs();
    let sin_y = yaw_rad.sin().abs();
    let aabb_hx = half_ext.x * cos_y + half_ext.z * sin_y;
    let aabb_hz = half_ext.z * cos_y + half_ext.x * sin_y;
    let aabb_min = center - Vec3::new(aabb_hx, half_ext.y, aabb_hz);
    let aabb_max = center + Vec3::new(aabb_hx, half_ext.y, aabb_hz);
    let lo = (
        (aabb_min.x / eb).floor() as i32,
        (aabb_min.y / eb).floor() as i32,
        (aabb_min.z / eb).floor() as i32,
    );
    let hi = (
        (aabb_max.x / eb).floor() as i32,
        (aabb_max.y / eb).floor() as i32,
        (aabb_max.z / eb).floor() as i32,
    );

    // Pre-compute the inverse rotation (rotate world point into OBB-local frame).
    let inv_cos = yaw_rad.cos();
    let inv_sin = yaw_rad.sin();

    capture_undo_for_range(store, lo, hi);

    let mut dirty_chunks: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> =
        Vec::new();

    for cz in lo.2..=hi.2 {
        for cy in lo.1..=hi.1 {
            for cx in lo.0..=hi.0 {
                if let Some(density) = store.density_fields.get_mut(&(cx, cy, cz)) {
                    let origin = Vec3::new(cx as f32 * eb, cy as f32 * eb, cz as f32 * eb);
                    let chunk_min = origin;
                    let chunk_max = origin + Vec3::splat(eb);
                    let isect_min = chunk_min.max(aabb_min);
                    let isect_max = chunk_max.min(aabb_max);
                    if isect_min.cmpgt(isect_max).any() {
                        continue;
                    }
                    let lo_x = (((isect_min.x - origin.x) / vs).floor() as i32).max(0) as usize;
                    let hi_x = (((isect_max.x - origin.x) / vs).ceil() as usize + 1).min(density.size);
                    let lo_y = (((isect_min.y - origin.y) / vs).floor() as i32).max(0) as usize;
                    let hi_y = (((isect_max.y - origin.y) / vs).ceil() as usize + 1).min(density.size);
                    let lo_z = (((isect_min.z - origin.z) / vs).floor() as i32).max(0) as usize;
                    let hi_z = (((isect_max.z - origin.z) / vs).ceil() as usize + 1).min(density.size);

                    let mut changed = false;
                    for z in lo_z..hi_z {
                        for y in lo_y..hi_y {
                            for x in lo_x..hi_x {
                                let world_pos = origin
                                    + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                let world_local = world_pos - center;
                                // Rotate world_local by -yaw to bring it into the OBB's local frame
                                // (yaw is around Rust Y axis, so X and Z permute).
                                let local_x = world_local.x *  inv_cos + world_local.z * inv_sin;
                                let local_z = -world_local.x * inv_sin + world_local.z * inv_cos;
                                let local_y = world_local.y;
                                // Inside-ness in OBB-local frame: positive if inside, negative if outside.
                                let inside = (half_ext.x - local_x.abs())
                                    .min(half_ext.y - local_y.abs())
                                    .min(half_ext.z - local_z.abs());
                                if inside <= 0.0 {
                                    continue;
                                }

                                let sample = density.get_mut(x, y, z);
                                match op {
                                    0 => {
                                        // Paint
                                        if sample.material.is_solid() && sample.material != target {
                                            sample.material = target;
                                            changed = true;
                                        }
                                    }
                                    1 => {
                                        // Carve
                                        if sample.material.is_solid() {
                                            let new_density = (-inside).min(sample.density);
                                            sample.density = new_density;
                                            if sample.density <= 0.0 {
                                                sample.material = Material::Air;
                                            }
                                            changed = true;
                                        }
                                    }
                                    2 => {
                                        // Fill
                                        if inside > sample.density {
                                            sample.density = inside;
                                            sample.material = target;
                                            changed = true;
                                        } else if sample.material.is_solid() && sample.material != target {
                                            sample.material = target;
                                            changed = true;
                                        }
                                    }
                                    _ => {}
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

/// Y-axis aligned cylinder brush. `op`: 0=paint, 1=carve, 2=fill.
/// `radius` is the XZ-plane radius; `height` is the full cylinder height (Rust units).
pub fn cylinder_brush(
    store: &mut ChunkStore,
    center: Vec3,
    radius: f32,
    height: f32,
    op: u8,
    target: Material,
    config: &GenerationConfig,
    world_scale: f32,
) -> BrushOutcome {
    if radius <= 0.0 || height <= 0.0 {
        return BrushOutcome { meshes: Vec::new(), flipped_chunks: Vec::new() };
    }
    if op == 2 && !target.is_solid() {
        return BrushOutcome { meshes: Vec::new(), flipped_chunks: Vec::new() };
    }

    let eb = config.effective_bounds();
    let vs = config.voxel_scale();
    let half_h = height * 0.5;
    let aabb_min = center - Vec3::new(radius, half_h, radius);
    let aabb_max = center + Vec3::new(radius, half_h, radius);
    let lo = (
        (aabb_min.x / eb).floor() as i32,
        (aabb_min.y / eb).floor() as i32,
        (aabb_min.z / eb).floor() as i32,
    );
    let hi = (
        (aabb_max.x / eb).floor() as i32,
        (aabb_max.y / eb).floor() as i32,
        (aabb_max.z / eb).floor() as i32,
    );

    capture_undo_for_range(store, lo, hi);

    let mut dirty_chunks: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> =
        Vec::new();

    for cz in lo.2..=hi.2 {
        for cy in lo.1..=hi.1 {
            for cx in lo.0..=hi.0 {
                if let Some(density) = store.density_fields.get_mut(&(cx, cy, cz)) {
                    let origin = Vec3::new(cx as f32 * eb, cy as f32 * eb, cz as f32 * eb);
                    let chunk_min = origin;
                    let chunk_max = origin + Vec3::splat(eb);
                    let isect_min = chunk_min.max(aabb_min);
                    let isect_max = chunk_max.min(aabb_max);
                    if isect_min.cmpgt(isect_max).any() {
                        continue;
                    }
                    let lo_x = (((isect_min.x - origin.x) / vs).floor() as i32).max(0) as usize;
                    let hi_x = (((isect_max.x - origin.x) / vs).ceil() as usize + 1).min(density.size);
                    let lo_y = (((isect_min.y - origin.y) / vs).floor() as i32).max(0) as usize;
                    let hi_y = (((isect_max.y - origin.y) / vs).ceil() as usize + 1).min(density.size);
                    let lo_z = (((isect_min.z - origin.z) / vs).floor() as i32).max(0) as usize;
                    let hi_z = (((isect_max.z - origin.z) / vs).ceil() as usize + 1).min(density.size);

                    let mut changed = false;
                    for z in lo_z..hi_z {
                        for y in lo_y..hi_y {
                            for x in lo_x..hi_x {
                                let world_pos = origin
                                    + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                let local = world_pos - center;
                                let dxz = (local.x * local.x + local.z * local.z).sqrt();
                                let inside_radial = radius - dxz;
                                let inside_y = half_h - local.y.abs();
                                let inside = inside_radial.min(inside_y);
                                if inside <= 0.0 {
                                    continue;
                                }

                                let sample = density.get_mut(x, y, z);
                                match op {
                                    0 => {
                                        if sample.material.is_solid() && sample.material != target {
                                            sample.material = target;
                                            changed = true;
                                        }
                                    }
                                    1 => {
                                        if sample.material.is_solid() {
                                            let new_density = (-inside).min(sample.density);
                                            sample.density = new_density;
                                            if sample.density <= 0.0 {
                                                sample.material = Material::Air;
                                            }
                                            changed = true;
                                        }
                                    }
                                    2 => {
                                        if inside > sample.density {
                                            sample.density = inside;
                                            sample.material = target;
                                            changed = true;
                                        } else if sample.material.is_solid() && sample.material != target {
                                            sample.material = target;
                                            changed = true;
                                        }
                                    }
                                    _ => {}
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

/// Smooth brush: Laplacian average of densities in a sphere. Material is preserved.
/// `iterations` × `strength` controls how much smoothing is applied (mine smoothing
/// uses 1-2 iterations at 0.3-0.5 strength as a reference).
pub fn smooth_brush(
    store: &mut ChunkStore,
    center: Vec3,
    radius: f32,
    iterations: u32,
    strength: f32,
    config: &GenerationConfig,
    world_scale: f32,
) -> BrushOutcome {
    if radius <= 0.0 || iterations == 0 || strength <= 0.0 {
        return BrushOutcome { meshes: Vec::new(), flipped_chunks: Vec::new() };
    }
    let eb = config.effective_bounds();
    let vs = config.voxel_scale();
    let r2 = radius * radius;
    let (lo, hi) = chunk_range_for_sphere(center, radius, eb);

    capture_undo_for_range(store, lo, hi);

    // Collect (chunk, voxel_idx) targets first, then run iterations of double-buffered
    // averaging on each targeted voxel. Material stays untouched.
    let mut dirty_chunks: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> =
        Vec::new();

    for cz in lo.2..=hi.2 {
        for cy in lo.1..=hi.1 {
            for cx in lo.0..=hi.0 {
                if let Some(density) = store.density_fields.get_mut(&(cx, cy, cz)) {
                    let origin = Vec3::new(cx as f32 * eb, cy as f32 * eb, cz as f32 * eb);
                    let (_, _, lo_x, lo_y, lo_z, hi_x, hi_y, hi_z) =
                        local_sphere_bounds(center, radius, origin, vs, density.size);

                    // Collect target voxels (those inside sphere)
                    let mut targets: Vec<(usize, usize, usize)> = Vec::new();
                    for z in lo_z..hi_z {
                        for y in lo_y..hi_y {
                            for x in lo_x..hi_x {
                                let world_pos = origin
                                    + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                if (world_pos - center).length_squared() <= r2 {
                                    targets.push((x, y, z));
                                }
                            }
                        }
                    }
                    if targets.is_empty() {
                        continue;
                    }

                    // Iterate Laplacian smoothing
                    let s = density.size;
                    for _ in 0..iterations {
                        let mut updates: Vec<(usize, usize, usize, f32)> =
                            Vec::with_capacity(targets.len());
                        for &(x, y, z) in &targets {
                            // Average of 6 face neighbors (clamped to bounds)
                            let mut sum = 0.0f32;
                            let mut count = 0u32;
                            let neighbors: [(i32, i32, i32); 6] = [
                                (-1, 0, 0), (1, 0, 0),
                                (0, -1, 0), (0, 1, 0),
                                (0, 0, -1), (0, 0, 1),
                            ];
                            for (dx, dy, dz) in neighbors {
                                let nx = x as i32 + dx;
                                let ny = y as i32 + dy;
                                let nz = z as i32 + dz;
                                if nx >= 0 && nx < s as i32
                                    && ny >= 0 && ny < s as i32
                                    && nz >= 0 && nz < s as i32
                                {
                                    sum += density.get(nx as usize, ny as usize, nz as usize).density;
                                    count += 1;
                                }
                            }
                            if count > 0 {
                                let avg = sum / count as f32;
                                let old = density.get(x, y, z).density;
                                let new_val = (1.0 - strength) * old + strength * avg;
                                updates.push((x, y, z, new_val));
                            }
                        }
                        for (x, y, z, new_density) in updates {
                            let sample = density.get_mut(x, y, z);
                            sample.density = new_density;
                            // Enforce invariant: Air must have non-positive density.
                            if !sample.material.is_solid() && sample.density > 0.0 {
                                sample.density = 0.0;
                            }
                        }
                    }

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

    finalize_brush(store, dirty_chunks, config, world_scale)
}

/// Noise brush: perturb density values in a sphere by a 3D simplex noise field
/// (real gradient noise, not hash noise — produces continuous, organic-looking
/// roughness instead of high-frequency jitter).
/// Falloff is Hermite-smoothed from sphere edge to center so edits don't show
/// hard seams. Material is preserved (no air↔solid flip unless density crosses 0).
pub fn noise_brush(
    store: &mut ChunkStore,
    center: Vec3,
    radius: f32,
    frequency: f32,
    strength: f32,
    seed: u32,
    config: &GenerationConfig,
    world_scale: f32,
) -> BrushOutcome {
    if radius <= 0.0 || strength.abs() < 1e-6 {
        return BrushOutcome { meshes: Vec::new(), flipped_chunks: Vec::new() };
    }
    let eb = config.effective_bounds();
    let vs = config.voxel_scale();
    let r2 = radius * radius;
    let (lo, hi) = chunk_range_for_sphere(center, radius, eb);

    capture_undo_for_range(store, lo, hi);

    // Real simplex noise from voxel-noise crate. Domain-warp via 2 octaves of
    // simplex for richer detail (rough pebbly look rather than uniform noise).
    use voxel_noise::NoiseSource;
    let simplex = voxel_noise::simplex::Simplex3D::new(seed as u64);
    let simplex_warp = voxel_noise::simplex::Simplex3D::new(seed as u64 ^ 0xDEADBEEF);
    let noise_at = |p: Vec3, freq: f32| -> f32 {
        let f = freq as f64;
        let wx = simplex_warp.sample(p.x as f64 * f * 0.5, p.y as f64 * f * 0.5, p.z as f64 * f * 0.5);
        // Light domain warp (~0.5 voxel) breaks up axis-aligned simplex artifacts.
        let n = simplex.sample(
            (p.x as f64) * f + wx * 0.5,
            (p.y as f64) * f - wx * 0.5,
            (p.z as f64) * f + wx * 0.3,
        );
        n as f32 // simplex returns roughly [-1, 1]
    };

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
                                let t = (dist2 / r2).clamp(0.0, 1.0);
                                // Hermite falloff: 1 at center, 0 at edge.
                                let falloff = 1.0 - t * t * (3.0 - 2.0 * t);
                                let n = noise_at(world_pos, frequency);
                                let delta = n * strength * falloff;
                                if delta.abs() < 1e-5 {
                                    continue;
                                }
                                let sample = density.get_mut(x, y, z);
                                // Clamp to legal density range. Without this,
                                // repeated noise-brush strokes accumulate
                                // density outside [-1, 1] (we've seen
                                // density=-5.6 after ~14 strokes), which
                                // breaks DC's edge-intersection math at
                                // chunk seams and produces sometimes-broken
                                // seam quads + huge internal cliffs in the
                                // diagnostic dump.
                                let new_density = (sample.density + delta).clamp(-1.0, 1.0);
                                let was_solid = sample.material.is_solid();
                                let now_solid = new_density > 0.0;
                                sample.density = new_density;
                                if was_solid && !now_solid {
                                    sample.material = Material::Air;
                                }
                                // Don't auto-promote air → solid here; that's ambiguous (which material?).
                                // If the user wants noise to reveal solid surfaces under air, they can
                                // run a fill brush first.
                                changed = true;
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

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::material::Material;
    use voxel_gen::config::GenerationConfig;
    use voxel_gen::density::DensityField;

    fn make_store_with_solid_chunk(chunk_size: usize) -> (ChunkStore, GenerationConfig) {
        let mut config = GenerationConfig::default();
        config.chunk_size = chunk_size;
        let mut store = ChunkStore::new(8);
        let size = chunk_size + 1;
        let mut field = DensityField::new(size);
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let s = field.get_mut(x, y, z);
                    s.density = 1.0;
                    s.material = Material::Limestone;
                }
            }
        }
        store.density_fields.insert((0, 0, 0), field);
        (store, config)
    }

    #[test]
    fn paint_changes_material_keeps_density() {
        let (mut store, config) = make_store_with_solid_chunk(8);
        let center = Vec3::new(4.0, 4.0, 4.0);
        let _ = paint_material_sphere(&mut store, center, 2.0, Material::Granite, &config, 1.0);
        let f = store.density_fields.get(&(0, 0, 0)).unwrap();
        let s = f.get(4, 4, 4);
        assert_eq!(s.material, Material::Granite);
        assert!(s.density > 0.0, "density preserved");
    }

    #[test]
    fn paint_skips_air() {
        let (mut store, config) = make_store_with_solid_chunk(8);
        // Carve out the center first
        let _ = carve_sphere(&mut store, Vec3::new(4.0, 4.0, 4.0), 1.5, &config, 1.0);
        // Now paint over the same region
        let _ = paint_material_sphere(&mut store, Vec3::new(4.0, 4.0, 4.0), 1.5, Material::Granite, &config, 1.0);
        let f = store.density_fields.get(&(0, 0, 0)).unwrap();
        let s = f.get(4, 4, 4);
        assert_eq!(s.material, Material::Air, "air voxels should not be painted");
    }

    #[test]
    fn fill_sphere_creates_solid() {
        let (mut store, config) = make_store_with_solid_chunk(8);
        // Carve first to make air
        let _ = carve_sphere(&mut store, Vec3::new(4.0, 4.0, 4.0), 2.0, &config, 1.0);
        // Verify air
        {
            let f = store.density_fields.get(&(0, 0, 0)).unwrap();
            assert_eq!(f.get(4, 4, 4).material, Material::Air);
        }
        // Fill
        let _ = fill_sphere(&mut store, Vec3::new(4.0, 4.0, 4.0), 1.5, Material::Granite, &config, 1.0);
        let f = store.density_fields.get(&(0, 0, 0)).unwrap();
        let s = f.get(4, 4, 4);
        assert_eq!(s.material, Material::Granite);
        assert!(s.density > 0.0);
    }

    #[test]
    fn carve_sphere_creates_air() {
        let (mut store, config) = make_store_with_solid_chunk(8);
        let _ = carve_sphere(&mut store, Vec3::new(4.0, 4.0, 4.0), 1.5, &config, 1.0);
        let f = store.density_fields.get(&(0, 0, 0)).unwrap();
        let s = f.get(4, 4, 4);
        assert_eq!(s.material, Material::Air);
        assert!(s.density <= 0.0);
    }

    #[test]
    fn tunnel_carves_along_path() {
        let (mut store, config) = make_store_with_solid_chunk(8);
        let path = vec![Vec3::new(2.0, 4.0, 4.0), Vec3::new(6.0, 4.0, 4.0)];
        let _ = tunnel(&mut store, &path, 1.0, None, &config, 1.0);
        let f = store.density_fields.get(&(0, 0, 0)).unwrap();
        // Mid-path should be carved
        assert_eq!(f.get(4, 4, 4).material, Material::Air);
        // Off-path should be solid
        assert!(f.get(4, 7, 4).material.is_solid());
    }

    #[test]
    fn tunnel_fills_along_path_with_material() {
        let (mut store, config) = make_store_with_solid_chunk(8);
        // Carve a region first
        let _ = carve_sphere(&mut store, Vec3::new(4.0, 4.0, 4.0), 3.0, &config, 1.0);
        let path = vec![Vec3::new(2.0, 4.0, 4.0), Vec3::new(6.0, 4.0, 4.0)];
        let _ = tunnel(&mut store, &path, 0.8, Some(Material::Granite), &config, 1.0);
        let f = store.density_fields.get(&(0, 0, 0)).unwrap();
        assert_eq!(f.get(4, 4, 4).material, Material::Granite);
    }

    #[test]
    fn box_brush_carves_cuboid_air() {
        let (mut store, config) = make_store_with_solid_chunk(16);
        let _ = box_brush(
            &mut store,
            Vec3::new(8.0, 8.0, 8.0),
            Vec3::new(2.0, 1.0, 3.0),
            0.0, // no yaw
            1, // carve
            Material::Air,
            &config,
            1.0,
        );
        let f = store.density_fields.get(&(0, 0, 0)).unwrap();
        // Inside box should be air
        assert_eq!(f.get(8, 8, 8).material, Material::Air);
        assert_eq!(f.get(7, 8, 9).material, Material::Air);
        // Outside box should be solid
        assert!(f.get(8, 8, 14).material.is_solid());
        assert!(f.get(2, 8, 8).material.is_solid());
    }

    #[test]
    fn box_brush_yaw_90_swaps_x_z() {
        // A 90-degree yaw should swap the X/Z extents of the AABB.
        // Half-extents (3, 1, 1) at 90deg yaw → effectively (1, 1, 3) AABB.
        // So a voxel at offset (+2, 0, 0) should be OUTSIDE the rotated box,
        // and a voxel at offset (0, 0, +2) should be INSIDE.
        let (mut store, config) = make_store_with_solid_chunk(16);
        use std::f32::consts::FRAC_PI_2;
        let _ = box_brush(
            &mut store,
            Vec3::new(8.0, 8.0, 8.0),
            Vec3::new(3.0, 1.0, 1.0),
            FRAC_PI_2, // 90 deg
            1, // carve
            Material::Air,
            &config,
            1.0,
        );
        let f = store.density_fields.get(&(0, 0, 0)).unwrap();
        // Z+2 should be carved (was the X-axis pre-rotation)
        assert_eq!(f.get(8, 8, 10).material, Material::Air,
            "after 90deg yaw, Z+2 should be inside the rotated box");
        // X+2 should NOT be carved (the long axis rotated to Z)
        assert!(f.get(10, 8, 8).material.is_solid(),
            "after 90deg yaw, X+2 should be outside the rotated box");
    }

    #[test]
    fn box_brush_fills_cuboid_solid() {
        let (mut store, config) = make_store_with_solid_chunk(16);
        // Carve big air pocket first
        let _ = carve_sphere(&mut store, Vec3::new(8.0, 8.0, 8.0), 5.0, &config, 1.0);
        // Fill a smaller box
        let _ = box_brush(
            &mut store,
            Vec3::new(8.0, 8.0, 8.0),
            Vec3::new(1.5, 1.5, 1.5),
            0.0, // no yaw
            2, // fill
            Material::Granite,
            &config,
            1.0,
        );
        let f = store.density_fields.get(&(0, 0, 0)).unwrap();
        assert_eq!(f.get(8, 8, 8).material, Material::Granite);
    }

    #[test]
    fn cylinder_brush_carves_vertical_shaft() {
        let (mut store, config) = make_store_with_solid_chunk(16);
        let _ = cylinder_brush(
            &mut store,
            Vec3::new(8.0, 8.0, 8.0),
            1.5, // radius
            6.0, // height
            1, // carve
            Material::Air,
            &config,
            1.0,
        );
        let f = store.density_fields.get(&(0, 0, 0)).unwrap();
        // Center of shaft should be air
        assert_eq!(f.get(8, 8, 8).material, Material::Air);
        // Top of shaft (still in cylinder, since height=6 → half=3 → y range 5..11) should be air
        assert_eq!(f.get(8, 10, 8).material, Material::Air);
        // Bottom of shaft (y=5) should be air
        assert_eq!(f.get(8, 6, 8).material, Material::Air);
        // Outside cylinder radius should still be solid
        assert!(f.get(8, 8, 12).material.is_solid());
        // Far above cylinder should still be solid
        assert!(f.get(8, 13, 8).material.is_solid());
    }

    #[test]
    fn smooth_brush_preserves_material() {
        let (mut store, config) = make_store_with_solid_chunk(16);
        // Mark a voxel with a different material to see if smoothing preserves it
        {
            let f = store.density_fields.get_mut(&(0, 0, 0)).unwrap();
            f.get_mut(8, 8, 8).material = Material::Granite;
            f.get_mut(8, 8, 8).density = 5.0;
        }
        let _ = smooth_brush(
            &mut store,
            Vec3::new(8.0, 8.0, 8.0),
            2.0,
            2,
            0.5,
            &config,
            1.0,
        );
        let f = store.density_fields.get(&(0, 0, 0)).unwrap();
        // Material is preserved through smoothing (only density is averaged)
        assert_eq!(f.get(8, 8, 8).material, Material::Granite);
    }

    #[test]
    fn noise_brush_perturbs_density() {
        let (mut store, config) = make_store_with_solid_chunk(16);
        // Snapshot densities in a band around the brush sphere.
        let before: Vec<f32> = {
            let f = store.density_fields.get(&(0, 0, 0)).unwrap();
            (0..27)
                .map(|i| {
                    let dx = i % 3;
                    let dy = (i / 3) % 3;
                    let dz = i / 9;
                    f.get(7 + dx, 7 + dy, 7 + dz).density
                })
                .collect()
        };
        let _ = noise_brush(
            &mut store,
            Vec3::new(8.0, 8.0, 8.0),
            3.0,
            0.5,  // freq — simplex at integer lattice gives 0; use sub-1 freq so samples land off-lattice
            1.0,
            42,
            &config,
            1.0,
        );
        // At least one voxel in the affected region should have changed.
        let after_changed_count = {
            let f = store.density_fields.get(&(0, 0, 0)).unwrap();
            (0..27).filter(|&i| {
                let dx = i % 3;
                let dy = (i / 3) % 3;
                let dz = i / 9;
                let after = f.get(7 + dx, 7 + dy, 7 + dz).density;
                (after - before[i as usize]).abs() > 1e-3
            }).count()
        };
        assert!(after_changed_count > 0, "noise should perturb at least one voxel in the brush sphere");
    }

    #[test]
    fn undo_restores_pre_brush_state() {
        let (mut store, config) = make_store_with_solid_chunk(8);
        // Capture initial state at a probe voxel.
        let before = {
            let f = store.density_fields.get(&(0, 0, 0)).unwrap();
            (f.get(4, 4, 4).density, f.get(4, 4, 4).material)
        };
        // Carve a sphere — should change the probe.
        let _ = carve_sphere(&mut store, Vec3::new(4.0, 4.0, 4.0), 2.0, &config, 1.0);
        {
            let f = store.density_fields.get(&(0, 0, 0)).unwrap();
            assert_eq!(f.get(4, 4, 4).material, Material::Air, "carve flipped to air");
        }
        assert_eq!(store.undo_stack.len(), 1, "undo stroke pushed");

        // Apply undo.
        let outcome = apply_undo(&mut store, &config, 1.0);
        assert!(outcome.is_some(), "undo returned remesh data");
        assert_eq!(store.undo_stack.len(), 0, "undo stack popped");

        // Probe voxel should be restored exactly.
        let after = {
            let f = store.density_fields.get(&(0, 0, 0)).unwrap();
            (f.get(4, 4, 4).density, f.get(4, 4, 4).material)
        };
        assert_eq!(after, before, "undo restored exact pre-state");
    }

    #[test]
    fn paint_stress_adds_to_overlay() {
        let (mut store, config) = make_store_with_solid_chunk(8);
        // Before: no painted layer.
        assert!(!store
            .stress_fields
            .get(&(0, 0, 0))
            .map(|sf| sf.has_painted_layer())
            .unwrap_or(false));

        // Paint a sphere — additive, smoothstep falloff.
        let _ = paint_stress_sphere(
            &mut store,
            Vec3::new(4.0, 4.0, 4.0),
            2.5,
            /*amount*/ 0.5,
            /*falloff*/ 2,
            /*op*/ 0,
            /*cap*/ 2.0,
            &config,
            1.0,
        );

        let sf = store.stress_fields.get(&(0, 0, 0)).unwrap();
        assert!(sf.has_painted_layer(), "painted layer allocated");
        // Voxel at sphere center gets close to full amount.
        let v_center = sf.painted(4, 4, 4);
        assert!(v_center > 0.4, "center painted (got {v_center})");
        // Voxel far outside sphere stays 0.
        assert_eq!(sf.painted(0, 0, 0), 0.0);
        // Effective stress = base (0) + painted at center.
        assert!((sf.effective(4, 4, 4) - v_center).abs() < 1e-6);
    }

    #[test]
    fn paint_stress_accumulates_capped() {
        let (mut store, config) = make_store_with_solid_chunk(8);
        // Two strokes, each 0.6, cap 1.0 — total should clamp at 1.0.
        for _ in 0..2 {
            let _ = paint_stress_sphere(
                &mut store,
                Vec3::new(4.0, 4.0, 4.0),
                1.0,
                /*amount*/ 0.6,
                /*falloff*/ 0, // constant
                /*op*/ 0,
                /*cap*/ 1.0,
                &config,
                1.0,
            );
        }
        let sf = store.stress_fields.get(&(0, 0, 0)).unwrap();
        assert!((sf.painted(4, 4, 4) - 1.0).abs() < 1e-6, "capped at 1.0");
    }

    #[test]
    fn paint_stress_clear_op_resets_overlay() {
        let (mut store, config) = make_store_with_solid_chunk(8);
        // First, paint some stress.
        let _ = paint_stress_sphere(
            &mut store,
            Vec3::new(4.0, 4.0, 4.0),
            2.0,
            0.5,
            0,
            0,
            2.0,
            &config,
            1.0,
        );
        assert!(store.stress_fields.get(&(0, 0, 0)).unwrap().painted(4, 4, 4) > 0.0);

        // Then clear inside a sphere with op=2.
        let _ = paint_stress_sphere(
            &mut store,
            Vec3::new(4.0, 4.0, 4.0),
            2.0,
            0.0,
            0,
            /*op=clear*/ 2,
            2.0,
            &config,
            1.0,
        );
        assert_eq!(
            store.stress_fields.get(&(0, 0, 0)).unwrap().painted(4, 4, 4),
            0.0,
            "clear op zeroed the painted overlay"
        );
    }

    #[test]
    fn paint_stress_undo_restores_overlay() {
        let (mut store, config) = make_store_with_solid_chunk(8);

        // No painted layer yet.
        let _ = paint_stress_sphere(
            &mut store,
            Vec3::new(4.0, 4.0, 4.0),
            2.0,
            0.5,
            2,
            0,
            2.0,
            &config,
            1.0,
        );
        let painted_after_paint = store
            .stress_fields
            .get(&(0, 0, 0))
            .unwrap()
            .painted(4, 4, 4);
        assert!(painted_after_paint > 0.0, "PaintStress wrote a value");

        // Undo — overlay should be wiped back to empty (its pre-state).
        let outcome = apply_undo(&mut store, &config, 1.0);
        assert!(outcome.is_some(), "undo returned an outcome");
        let sf = store.stress_fields.get(&(0, 0, 0)).unwrap();
        assert!(
            !sf.has_painted_layer(),
            "undo wiped the painted overlay back to empty pre-state"
        );
    }

    #[test]
    fn paint_stress_drives_overstressed_threshold() {
        use voxel_core::stress::{recalc_stress_region_v2, StressConfig};
        use voxel_core::stress::SupportField;

        let (mut store, config) = make_store_with_solid_chunk(8);
        // Add a stress_field so painted_stress survives the recalc.
        let size = config.chunk_size + 1;
        store
            .stress_fields
            .insert((0, 0, 0), voxel_core::stress::StressField::new(size));
        store
            .support_fields
            .insert((0, 0, 0), SupportField::new(size));

        // Paint stress past 1.0 at the center.
        let _ = paint_stress_sphere(
            &mut store,
            Vec3::new(4.0, 4.0, 4.0),
            1.5,
            /*amount*/ 1.5,
            /*falloff*/ 0,
            /*op*/ 0,
            /*cap*/ 2.0,
            &config,
            1.0,
        );

        // Recalc — overstressed list should include the painted voxels even
        // though raw geological stress is 0 (the chunk is solid all around).
        // We don't assert exact counts (the recalc skips fully-grounded
        // voxels), just that the painted value rides through to effective().
        let stress_config = StressConfig::default();
        let chunks: Vec<_> = vec![(0, 0, 0)];
        let _ = recalc_stress_region_v2(
            &store.density_fields,
            &mut store.stress_fields,
            &store.support_fields,
            &stress_config,
            &chunks,
            config.chunk_size,
        );

        // The painted layer must survive the recalc (only `stress[]` is rewritten).
        let sf = store.stress_fields.get(&(0, 0, 0)).unwrap();
        assert!(
            sf.painted(4, 4, 4) > 0.0,
            "painted layer survives recalc_stress_region_v2"
        );
        assert!(
            sf.effective(4, 4, 4) >= sf.painted(4, 4, 4),
            "effective folds in painted layer"
        );
    }

    #[test]
    fn chunk_snapshot_painted_stress_roundtrip() {
        use crate::delta::ChunkSnapshot;
        use voxel_core::stress::StressField;

        let (store, _config) = make_store_with_solid_chunk(8);
        let df = store.density_fields.get(&(0, 0, 0)).unwrap();
        let mut sf = StressField::new(df.size);

        // Capture None when nothing has been painted.
        let snap_empty = ChunkSnapshot::from_chunk(df, Some(&sf));
        assert!(snap_empty.painted_stress.is_none(), "None when no overlay");

        // Paint a few cells and re-capture.
        sf.add_painted(4, 4, 4, 0.7, 2.0);
        sf.add_painted(5, 4, 4, 0.4, 2.0);
        let snap_with = ChunkSnapshot::from_chunk(df, Some(&sf));
        assert!(snap_with.painted_stress.is_some(), "Some after paint");

        // Restore onto a fresh field.
        let mut sf2 = StressField::new(df.size);
        snap_with.apply_painted_stress_to(&mut sf2);
        assert!((sf2.painted(4, 4, 4) - 0.7).abs() < 1e-6);
        assert!((sf2.painted(5, 4, 4) - 0.4).abs() < 1e-6);

        // Restoring `None` wipes the overlay back to empty.
        let mut sf3 = sf2.clone();
        snap_empty.apply_painted_stress_to(&mut sf3);
        assert!(
            !sf3.has_painted_layer(),
            "applying None-snapshot wipes the overlay"
        );
    }

    #[test]
    fn undo_stack_bounded_by_max_depth() {
        let (mut store, config) = make_store_with_solid_chunk(8);
        store.undo_max_depth = 3;
        for _ in 0..10 {
            let _ = carve_sphere(&mut store, Vec3::new(4.0, 4.0, 4.0), 1.0, &config, 1.0);
        }
        assert_eq!(store.undo_stack.len(), 3, "undo stack capped at max_depth");
    }

    #[test]
    fn fluid_sphere_collects_only_air_cells() {
        let (mut store, config) = make_store_with_solid_chunk(16);
        // Carve a half-buried air pocket. After this, ~half the brush sphere will be air.
        let _ = carve_sphere(&mut store, Vec3::new(8.0, 8.0, 8.0), 3.0, &config, 1.0);
        let cells = collect_fluid_cells_in_sphere(
            &store,
            Vec3::new(8.0, 8.0, 8.0),
            3.5,
            false, // not bottom-half-only
            &config,
        );
        assert!(!cells.is_empty(), "should find air cells inside sphere");
        // Every collected cell should be air in the density field.
        let f = store.density_fields.get(&(0, 0, 0)).unwrap();
        for cell in &cells {
            let s = f.get(cell.x as usize, cell.y as usize, cell.z as usize);
            assert!(!s.material.is_solid(), "fluid cell should be air");
        }
    }

    #[test]
    fn fluid_sphere_bottom_half_only() {
        let (mut store, config) = make_store_with_solid_chunk(16);
        let _ = carve_sphere(&mut store, Vec3::new(8.0, 8.0, 8.0), 3.0, &config, 1.0);
        let cells = collect_fluid_cells_in_sphere(
            &store,
            Vec3::new(8.0, 8.0, 8.0),
            3.5,
            true, // bottom half only
            &config,
        );
        // Every cell.y should be < center.y (8.0)
        for cell in &cells {
            assert!((cell.y as f32) < 8.0, "bottom-half cells should be below center y");
        }
    }

    #[test]
    fn fluid_box_collects_air_in_aabb() {
        let (mut store, config) = make_store_with_solid_chunk(16);
        let _ = box_brush(
            &mut store,
            Vec3::new(8.0, 8.0, 8.0),
            Vec3::new(2.0, 2.0, 2.0),
            0.0, // no yaw
            1, // carve
            Material::Air,
            &config,
            1.0,
        );
        let cells = collect_fluid_cells_in_box(
            &store,
            Vec3::new(8.0, 8.0, 8.0),
            Vec3::new(2.0, 2.0, 2.0),
            &config,
        );
        assert!(!cells.is_empty(), "should find air cells in carved box");
    }

    #[test]
    fn fluid_river_capsule_collects_air_along_path() {
        let (mut store, config) = make_store_with_solid_chunk(16);
        let path = vec![Vec3::new(3.0, 8.0, 8.0), Vec3::new(13.0, 8.0, 8.0)];
        let _ = tunnel(&mut store, &path, 1.5, None, &config, 1.0);
        let cells = collect_fluid_cells_in_capsule_chain(&store, &path, 1.5, &config);
        assert!(!cells.is_empty(), "should find air cells along carved tunnel");
    }

    #[test]
    fn place_formation_column_writes_solid() {
        let (mut store, config) = make_store_with_solid_chunk(16);
        // Carve an air pocket first
        let _ = carve_sphere(&mut store, Vec3::new(8.0, 8.0, 8.0), 4.0, &config, 1.0);
        // Place a column inside the pocket
        let _ = place_formation(
            &mut store,
            Vec3::new(8.0, 8.0, 8.0),
            2,           // column
            3.0,         // height
            0.8,         // radius
            Material::Limestone,
            &config,
            1.0,
        );
        let f = store.density_fields.get(&(0, 0, 0)).unwrap();
        // Column should fill the center
        assert_eq!(f.get(8, 8, 8).material, Material::Limestone);
    }
}
