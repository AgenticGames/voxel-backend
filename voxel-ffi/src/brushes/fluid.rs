//! Fluid-cell collection helpers: compute lists of (chunk, local-x/y/z) cells
//! that should receive a fluid placement event. They DON'T touch the fluid
//! system directly.

use glam::Vec3;
use voxel_gen::config::GenerationConfig;

use crate::store::ChunkStore;

use super::common::{chunk_range_for_sphere, local_sphere_bounds};

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
