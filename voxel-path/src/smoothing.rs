//! Theta*-style line-of-sight smoothing on a finished A* path.
//!
//! Repeatedly walk through the node list dropping intermediate nodes whose
//! pre/post neighbors have a clear 3D-Bresenham line between them (every cell
//! along that line is traversable). Surface mode is conservative: only drop a
//! node when its neighbors' surface normals are within 30° of each other so we
//! don't cut through a real corner.

use crate::astar::PathNode;
use crate::grid::CellGrid;
use crate::movement::{can_traverse, MovementMode};
use glam::{IVec3, Vec3};

/// Returns a smoothed copy of `path` (or the input unchanged if nothing
/// could be smoothed).
pub fn smooth_path<G: CellGrid>(grid: &G, mode: MovementMode, path: Vec<PathNode>) -> Vec<PathNode> {
    if path.len() < 3 {
        return path;
    }

    // Repeated passes — each pass drops at most every other intermediate
    // node. Two passes converges in practice for typical paths; we cap at 3.
    let mut current = path;
    for _ in 0..3 {
        let next = smooth_one_pass(grid, mode, &current);
        if next.len() == current.len() {
            return next;
        }
        current = next;
    }
    current
}

fn smooth_one_pass<G: CellGrid>(grid: &G, mode: MovementMode, path: &[PathNode]) -> Vec<PathNode> {
    let mut out: Vec<PathNode> = Vec::with_capacity(path.len());
    out.push(path[0].clone());
    let mut anchor_idx = 0;

    let mut i = 1;
    while i < path.len() - 1 {
        let anchor = &path[anchor_idx];
        let probe = &path[i + 1];

        let surface_safe = if mode.is_surface() {
            normal_within_threshold(anchor.surface_normal, probe.surface_normal, 30.0)
        } else {
            true
        };

        if surface_safe && line_clear(grid, mode, anchor.cell, probe.cell) {
            // We can skip path[i] entirely — leave anchor_idx fixed.
            i += 1;
        } else {
            out.push(path[i].clone());
            anchor_idx = i;
            i += 1;
        }
    }
    out.push(path.last().unwrap().clone());
    out
}

fn normal_within_threshold(a: Vec3, b: Vec3, degrees: f32) -> bool {
    if a.length_squared() < 1e-6 || b.length_squared() < 1e-6 {
        return true; // missing normal (e.g., walking/flying mode) → no constraint
    }
    let dot = a.normalize().dot(b.normalize()).clamp(-1.0, 1.0);
    let angle = dot.acos().to_degrees();
    angle <= degrees
}

/// 3D-Bresenham line cell traversal — visit every cell intersected by the
/// segment from `a` to `b` (inclusive of both endpoints). Returns true if
/// every cell along the way is traversable in the given mode.
fn line_clear<G: CellGrid>(grid: &G, mode: MovementMode, a: IVec3, b: IVec3) -> bool {
    let dx = (b.x - a.x).abs();
    let dy = (b.y - a.y).abs();
    let dz = (b.z - a.z).abs();
    let sx = if a.x < b.x { 1 } else { -1 };
    let sy = if a.y < b.y { 1 } else { -1 };
    let sz = if a.z < b.z { 1 } else { -1 };

    let mut x = a.x;
    let mut y = a.y;
    let mut z = a.z;

    // Driving axis = the one with the largest delta.
    let max_axis = dx.max(dy).max(dz);
    if max_axis == 0 {
        return true; // same cell
    }

    let mut err_xy: i32 = (dy << 1) - dx;
    let mut err_xz: i32 = (dz << 1) - dx;
    let mut err_yx: i32 = (dx << 1) - dy;
    let mut err_yz: i32 = (dz << 1) - dy;
    let mut err_zx: i32 = (dx << 1) - dz;
    let mut err_zy: i32 = (dy << 1) - dz;

    if dx >= dy && dx >= dz {
        for _ in 0..dx {
            if !can_traverse(grid, IVec3::new(x, y, z), mode) {
                return false;
            }
            if err_xy > 0 {
                y += sy;
                err_xy -= dx << 1;
            }
            if err_xz > 0 {
                z += sz;
                err_xz -= dx << 1;
            }
            err_xy += dy << 1;
            err_xz += dz << 1;
            x += sx;
        }
    } else if dy >= dx && dy >= dz {
        for _ in 0..dy {
            if !can_traverse(grid, IVec3::new(x, y, z), mode) {
                return false;
            }
            if err_yx > 0 {
                x += sx;
                err_yx -= dy << 1;
            }
            if err_yz > 0 {
                z += sz;
                err_yz -= dy << 1;
            }
            err_yx += dx << 1;
            err_yz += dz << 1;
            y += sy;
        }
    } else {
        for _ in 0..dz {
            if !can_traverse(grid, IVec3::new(x, y, z), mode) {
                return false;
            }
            if err_zx > 0 {
                x += sx;
                err_zx -= dz << 1;
            }
            if err_zy > 0 {
                y += sy;
                err_zy -= dz << 1;
            }
            err_zx += dx << 1;
            err_zy += dy << 1;
            z += sz;
        }
    }
    can_traverse(grid, IVec3::new(x, y, z), mode)
}
