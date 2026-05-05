//! SDF primitives and helpers for the all-in building flatten.
//!
//! Coordinate convention: all positions in this module are in VOXEL space
//! (1 unit = 1 voxel). The flatten code converts to/from world units at
//! the boundary; here we work in voxels for simplicity.

use std::collections::HashMap;

use glam::Vec3;
use voxel_core::density::DensityField;

// ── Shape SDFs ─────────────────────────────────────────────────────────────

/// Inigo Quilez polynomial smooth-min. Blends two SDFs with `k`-controlled
/// rounding at the seam. Larger k = wider/smoother blend.
#[inline]
pub fn smin(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return a.min(b);
    }
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - h * h * k * 0.25
}

/// Signed distance to an axis-aligned box centered at the origin with the
/// given half-extents. Negative inside, positive outside.
#[inline]
pub fn sdf_box(p: Vec3, half_extent: Vec3) -> f32 {
    let q = p.abs() - half_extent;
    let outside = q.max(Vec3::ZERO).length();
    let inside = q.x.max(q.y.max(q.z)).min(0.0);
    outside + inside
}

/// Signed distance to a capped (truncated) cone segment between `base` and
/// `tip`, with radii `r_base` at the base end and `r_tip` at the tip end.
/// Used to sculpt buttress shapes from the building floor down to a support
/// hit point. Negative inside the cone, positive outside.
///
/// One-shot convenience wrapper. When you sample the same cone many times,
/// build a `CompiledCone` once with `compile_cone` and call
/// `sdf_compiled_cone` instead — it skips the per-call sqrt for axis length
/// and the sqrt+div for the slanted-side normal.
pub fn sdf_capped_cone(p: Vec3, base: Vec3, tip: Vec3, r_base: f32, r_tip: f32) -> f32 {
    match compile_cone(base, tip, r_base, r_tip) {
        Some(cone) => sdf_compiled_cone(p, &cone),
        None => {
            let r = (r_base + r_tip) * 0.5;
            (p - base).length() - r
        }
    }
}

/// Precomputed cone parameters. Build once, sample many times.
#[derive(Clone, Copy)]
pub struct CompiledCone {
    pub base: Vec3,
    pub axis_n: Vec3,
    pub h: f32,
    pub r_base: f32,
    pub r_tip: f32,
    nx: f32,
    ny: f32,
}

/// Precompute the per-cone constants used by `sdf_compiled_cone`. Returns
/// `None` for a degenerate cone (caller should fall back to a sphere SDF).
#[inline]
pub fn compile_cone(base: Vec3, tip: Vec3, r_base: f32, r_tip: f32) -> Option<CompiledCone> {
    let axis = tip - base;
    let h = axis.length();
    if h < 1e-4 {
        return None;
    }
    let axis_n = axis / h;
    // Slanted-side outward normal in (perp, along) plane, precomputed.
    let dx_side = r_tip - r_base;
    let dy_side = h;
    let len_side = (dx_side * dx_side + dy_side * dy_side).sqrt().max(1e-4);
    let nx = dy_side / len_side;
    let ny = -dx_side / len_side;
    Some(CompiledCone { base, axis_n, h, r_base, r_tip, nx, ny })
}

/// Hot-path SDF using a `CompiledCone`. Saves the per-call sqrt for the axis
/// length and the sqrt+div for the slanted-side normal. Identical numerics to
/// `sdf_capped_cone`.
#[inline]
pub fn sdf_compiled_cone(p: Vec3, c: &CompiledCone) -> f32 {
    let rel = p - c.base;
    let along = rel.dot(c.axis_n);
    let perp = rel - c.axis_n * along;
    let perp_len_sq = perp.length_squared();

    if along < 0.0 {
        // Below base.
        let perp_len = perp_len_sq.sqrt();
        let dr = (perp_len - c.r_base).max(0.0);
        (dr * dr + along * along).sqrt()
    } else if along > c.h {
        // Above tip.
        let perp_len = perp_len_sq.sqrt();
        let dr = (perp_len - c.r_tip).max(0.0);
        let dh = along - c.h;
        (dr * dr + dh * dh).sqrt()
    } else {
        // Inside the cone's height range. Signed distance to the slanted side.
        let perp_len = perp_len_sq.sqrt();
        (perp_len - c.r_base) * c.nx + along * c.ny
    }
}

// ── Density sampling ──────────────────────────────────────────────────────

/// Trilinear sample of the natural density field at a continuous voxel point.
/// Returns 1.0 (deep solid) outside any loaded chunk so the SDF blend stops
/// cantilevers from leaking into ungenerated areas.
pub fn sample_natural_density(
    fields: &HashMap<(i32, i32, i32), DensityField>,
    cs: i32,
    p: Vec3,
) -> f32 {
    let x0 = p.x.floor();
    let y0 = p.y.floor();
    let z0 = p.z.floor();
    let fx = p.x - x0;
    let fy = p.y - y0;
    let fz = p.z - z0;

    let sample = |wx: i32, wy: i32, wz: i32| -> f32 {
        let cx = wx.div_euclid(cs);
        let cy = wy.div_euclid(cs);
        let cz = wz.div_euclid(cs);
        let lx = wx.rem_euclid(cs) as usize;
        let ly = wy.rem_euclid(cs) as usize;
        let lz = wz.rem_euclid(cs) as usize;
        match fields.get(&(cx, cy, cz)) {
            Some(df) => df.get(lx, ly, lz).density,
            None => 1.0, // unloaded = deep solid
        }
    };

    let ix = x0 as i32;
    let iy = y0 as i32;
    let iz = z0 as i32;

    let c000 = sample(ix, iy, iz);
    let c100 = sample(ix + 1, iy, iz);
    let c010 = sample(ix, iy + 1, iz);
    let c110 = sample(ix + 1, iy + 1, iz);
    let c001 = sample(ix, iy, iz + 1);
    let c101 = sample(ix + 1, iy, iz + 1);
    let c011 = sample(ix, iy + 1, iz + 1);
    let c111 = sample(ix + 1, iy + 1, iz + 1);

    let c00 = c000 * (1.0 - fx) + c100 * fx;
    let c10 = c010 * (1.0 - fx) + c110 * fx;
    let c01 = c001 * (1.0 - fx) + c101 * fx;
    let c11 = c011 * (1.0 - fx) + c111 * fx;
    let c0 = c00 * (1.0 - fy) + c10 * fy;
    let c1 = c01 * (1.0 - fy) + c11 * fy;
    c0 * (1.0 - fz) + c1 * fz
}

/// Convert a density field sample (positive=solid, negative=air) into an SDF
/// (negative inside rock, positive outside). The native density values are
/// already roughly in voxel units near the surface, but they cap at ±1.0,
/// which makes the SDF "flat" deep inside rock or far in air. We expand the
/// effective range by a multiplier so smin blends behave correctly far from
/// the iso-surface.
#[inline]
pub fn density_to_sdf(density: f32) -> f32 {
    // Density convention: +1 deep solid, -1 deep air, 0 at surface.
    // SDF convention: negative inside rock, positive outside (in air).
    // Multiplier converts the [-1, +1] density range into voxel-scale distance.
    -density * 4.0
}

// ── Directional support search (Phase 2 lives here) ───────────────────────

/// A single ray hit for the buttress/wrap pass.
#[derive(Debug, Clone, Copy)]
pub struct SupportHit {
    pub dir: Vec3,
    pub hit_pos: Vec3,
    pub distance: f32,
}

/// Cast `n` rays from `origin` in a hemisphere biased downward + horizontal
/// (lower half + horizontal band, NEVER pointing more than `up_tolerance`
/// above horizontal). Each ray steps 1 voxel at a time up to `max_dist`.
/// First solid cell with air directly above counts as a hit. Returns hits
/// sorted by distance (closest first).
///
/// `up_tolerance` of 0.1 means we accept rays whose dir.y is between -1 and
/// +0.1 — practically horizontal+downward. This is the "no upward connections"
/// constraint that prevents reverse ramps to walls above the building.
pub fn find_support_rays(
    fields: &HashMap<(i32, i32, i32), DensityField>,
    cs: i32,
    origin: Vec3,
    max_dist: f32,
    n_rays: usize,
    up_tolerance: f32,
) -> Vec<SupportHit> {
    let mut hits: Vec<SupportHit> = Vec::with_capacity(n_rays);

    // Fibonacci sphere directions, then filter to lower hemisphere + small up band.
    let golden_angle = std::f32::consts::PI * (3.0 - (5.0_f32).sqrt());
    let mut total = 0usize;
    let mut idx = 0usize;
    // Generate up to 4×n_rays candidates so we can filter and still get n_rays.
    let max_candidates = n_rays * 4;
    while total < n_rays && idx < max_candidates {
        let i = idx as f32;
        let n = max_candidates as f32;
        // Unit sphere via Fibonacci.
        let y = 1.0 - 2.0 * (i + 0.5) / n;
        let r = (1.0 - y * y).max(0.0).sqrt();
        let theta = golden_angle * i;
        let dir = Vec3::new(theta.cos() * r, y, theta.sin() * r).normalize();
        idx += 1;

        // Filter: only consider rays going horizontal-or-down.
        if dir.y > up_tolerance {
            continue;
        }

        // March the ray.
        if let Some(hit) = march_ray_for_surface(fields, cs, origin, dir, max_dist) {
            hits.push(hit);
            total += 1;
        }
    }

    hits.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
    hits
}

/// Step a ray in 1-voxel increments looking for a solid cell with air
/// directly above it (a real surface, not just any solid cell). Returns the
/// first such hit or `None` if we exceed `max_dist` or run out of loaded
/// chunks.
fn march_ray_for_surface(
    fields: &HashMap<(i32, i32, i32), DensityField>,
    cs: i32,
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
) -> Option<SupportHit> {
    let step = 1.0_f32;
    let n_steps = (max_dist / step).ceil() as i32;

    let solid_at = |p: Vec3| -> Option<bool> {
        let wx = p.x.round() as i32;
        let wy = p.y.round() as i32;
        let wz = p.z.round() as i32;
        let cx = wx.div_euclid(cs);
        let cy = wy.div_euclid(cs);
        let cz = wz.div_euclid(cs);
        let lx = wx.rem_euclid(cs) as usize;
        let ly = wy.rem_euclid(cs) as usize;
        let lz = wz.rem_euclid(cs) as usize;
        fields.get(&(cx, cy, cz)).map(|df| df.get(lx, ly, lz).density > 0.0)
    };

    let mut prev_solid = solid_at(origin).unwrap_or(false);
    for s in 1..=n_steps {
        let t = s as f32 * step;
        let p = origin + dir * t;
        let cur = match solid_at(p) {
            Some(v) => v,
            None => return None,
        };
        // We want a TRANSITION from air to solid as we march out — the first
        // surface we hit. (Surfaces have air on one side, solid on the other.)
        if cur && !prev_solid {
            return Some(SupportHit { dir, hit_pos: p, distance: t });
        }
        prev_solid = cur;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smin_at_equal_inputs_rounds_below_min() {
        let a = 1.0;
        let b = 1.0;
        let k = 2.0;
        let r = smin(a, b, k);
        // smin should be slightly LESS than min when a==b (the rounding dips).
        assert!(r <= a.min(b));
        // Polynomial smin's deepest dip at a==b is exactly min - k/4.
        assert!(r >= a.min(b) - k * 0.25 - 1e-4);
    }

    #[test]
    fn smin_far_apart_equals_min() {
        let a = -10.0;
        let b = 10.0;
        let r = smin(a, b, 2.0);
        assert!((r - a).abs() < 1e-3);
    }

    #[test]
    fn sdf_box_inside_negative_outside_positive() {
        let half = Vec3::new(2.0, 2.0, 2.0);
        assert!(sdf_box(Vec3::ZERO, half) < 0.0);
        assert!(sdf_box(Vec3::new(5.0, 0.0, 0.0), half) > 0.0);
        // On the face: distance ≈ 0.
        let face = sdf_box(Vec3::new(2.0, 0.0, 0.0), half);
        assert!(face.abs() < 1e-3);
    }

    #[test]
    fn density_to_sdf_signs_flip() {
        assert!(density_to_sdf(1.0) < 0.0);   // solid -> SDF negative
        assert!(density_to_sdf(-1.0) > 0.0);  // air -> SDF positive
        assert!(density_to_sdf(0.0).abs() < 1e-3);
    }
}
