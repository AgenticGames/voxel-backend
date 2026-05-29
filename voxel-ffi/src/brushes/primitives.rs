//! Tunnel, box, cylinder, smooth, and noise primitive brushes, plus `apply_undo`.

use glam::Vec3;
use voxel_core::material::Material;
use voxel_gen::config::GenerationConfig;

use crate::store::ChunkStore;

use super::common::{
    capture_undo_for_range, chunk_range_for_sphere, finalize_brush, local_sphere_bounds,
    BrushOutcome,
};

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
    if stroke.snapshots.is_empty() && stroke.mushroom_snapshots.is_empty() {
        return None;
    }
    let mut dirty_chunks: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> =
        Vec::with_capacity(stroke.snapshots.len() + stroke.mushroom_snapshots.len());

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

    // Restore mushroom placements (mushroom-paint/erase brush undo). The
    // density restore loop above handles dirty-rect entries for density
    // changes; mushroom-only strokes need their chunks added too so the
    // remesh path re-emits mushroom_data to UE.
    for (key, prior) in &stroke.mushroom_snapshots {
        if prior.is_empty() {
            store.mushroom_placements.remove(key);
        } else {
            store.mushroom_placements.insert(*key, prior.clone());
        }
        // Force a re-emit by clearing the last-sent mesh hash; the seam
        // pass will rebuild + ship new mushroom_data.
        store.last_sent_mesh_hash.remove(key);
        if !dirty_chunks.iter().any(|(k, ..)| k == key) {
            if let Some(density) = store.density_fields.get(key) {
                let s_max = density.size - 1;
                dirty_chunks.push((*key, 0, 0, 0, s_max, s_max, s_max));
            }
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
