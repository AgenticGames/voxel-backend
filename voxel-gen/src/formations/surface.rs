//! Surface detection helpers: scan the density field for ceiling/floor/wall
//! voxels adjacent to air, and measure air/solid extents for placement sizing.

use crate::density::DensityField;

/// Type of surface detected adjacent to air.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SurfaceKind {
    /// Solid with air below (stalactite anchor)
    Ceiling,
    /// Solid with air above (stalagmite anchor)
    Floor,
    /// Solid with air at an X/Z neighbor (flowstone anchor)
    Wall,
}

/// A detected surface point where a formation can be anchored.
#[derive(Debug, Clone)]
pub(super) struct SurfacePoint {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub kind: SurfaceKind,
    pub material: voxel_core::material::Material,
}

/// Scan voxels at indices 1..size-1 (avoid boundary) for solid voxels adjacent to air.
pub(super) fn detect_surfaces(density: &DensityField, size: usize) -> Vec<SurfacePoint> {
    let mut surfaces = Vec::new();
    let scan_max = size - 1;

    for z in 1..scan_max {
        for y in 1..scan_max {
            for x in 1..scan_max {
                let sample = density.get(x, y, z);
                if sample.density <= 0.0 {
                    continue; // air voxel, not a surface
                }
                let material = sample.material;

                // Check if air below → ceiling surface
                if y > 0 && density.get(x, y - 1, z).density <= 0.0 {
                    surfaces.push(SurfacePoint {
                        x,
                        y,
                        z,
                        kind: SurfaceKind::Ceiling,
                        material,
                    });
                }

                // Check if air above → floor surface
                if y + 1 < size && density.get(x, y + 1, z).density <= 0.0 {
                    surfaces.push(SurfacePoint {
                        x,
                        y,
                        z,
                        kind: SurfaceKind::Floor,
                        material,
                    });
                }

                // Check X/Z neighbors for wall surface
                let has_air_neighbor = (x > 0 && density.get(x - 1, y, z).density <= 0.0)
                    || (x + 1 < size && density.get(x + 1, y, z).density <= 0.0)
                    || (z > 0 && density.get(x, y, z - 1).density <= 0.0)
                    || (z + 1 < size && density.get(x, y, z + 1).density <= 0.0);

                if has_air_neighbor {
                    // Only add wall if this is NOT already a ceiling or floor
                    // (avoid double-counting corners)
                    let is_ceiling = y > 0 && density.get(x, y - 1, z).density <= 0.0;
                    let is_floor = y + 1 < size && density.get(x, y + 1, z).density <= 0.0;
                    if !is_ceiling && !is_floor {
                        surfaces.push(SurfacePoint {
                            x,
                            y,
                            z,
                            kind: SurfaceKind::Wall,
                            material,
                        });
                    }
                }
            }
        }
    }

    surfaces
}

/// Detect ceiling slope at a position by comparing ceiling Y at adjacent X/Z.
/// Returns slope direction (dx, dz) if slope delta >= 1.
pub(super) fn detect_ceiling_slope(
    density: &DensityField,
    x: usize,
    y: usize,
    z: usize,
    size: usize,
) -> Option<(f32, f32)> {
    if x < 1 || x >= size - 1 || z < 1 || z >= size - 1 {
        return None;
    }

    // Find ceiling Y at a given (px, pz) by scanning upward from y
    let find_ceil = |px: usize, pz: usize| -> Option<usize> {
        for py in y..size.min(y + 6) {
            if density.get(px, py, pz).density > 0.0 {
                return Some(py);
            }
        }
        None
    };

    let cy = find_ceil(x, z)?;
    let cx_pos = find_ceil(x + 1, z).unwrap_or(cy);
    let cx_neg = find_ceil(x.saturating_sub(1), z).unwrap_or(cy);
    let cz_pos = find_ceil(x, z + 1).unwrap_or(cy);
    let cz_neg = find_ceil(x, z.saturating_sub(1)).unwrap_or(cy);

    let dx = cx_pos as f32 - cx_neg as f32;
    let dz = cz_pos as f32 - cz_neg as f32;

    let mag = (dx * dx + dz * dz).sqrt();
    if mag >= 1.0 {
        Some((dx / mag, dz / mag))
    } else {
        None
    }
}

/// Detect floor slope using central differences with wide sampling.
/// Returns (grad_x, grad_z, magnitude).
///
/// Uses a sampling radius of 3 voxels (±3) to catch gentle slopes that
/// change only 1 voxel over several horizontal steps. Also searches both
/// up and down from the anchor Y to find floor surfaces at neighbor columns.
pub(super) fn detect_floor_slope(
    density: &DensityField,
    x: usize,
    y: usize,
    z: usize,
    size: usize,
) -> Option<(f32, f32, f32)> {
    let radius: usize = 3;
    if x < radius || x >= size - radius || z < radius || z >= size - radius {
        return None;
    }

    // Find nearest floor surface at (px, pz): solid with air above,
    // searching both up and down from anchor y.
    let find_floor = |px: usize, pz: usize| -> Option<usize> {
        let search_range = 4usize;
        for dy in 0..=search_range {
            // Check y - dy
            if dy <= y {
                let py = y - dy;
                if py + 1 < size
                    && density.get(px, py, pz).density > 0.0
                    && density.get(px, py + 1, pz).density <= 0.0
                {
                    return Some(py);
                }
            }
            // Check y + dy (skip dy=0 to avoid double-checking)
            if dy > 0 {
                let py = y + dy;
                if py + 1 < size
                    && density.get(px, py, pz).density > 0.0
                    && density.get(px, py + 1, pz).density <= 0.0
                {
                    return Some(py);
                }
            }
        }
        None
    };

    let fy = find_floor(x, z)?;
    let fx_pos = find_floor(x + radius, z).unwrap_or(fy);
    let fx_neg = find_floor(x - radius, z).unwrap_or(fy);
    let fz_pos = find_floor(x, z + radius).unwrap_or(fy);
    let fz_neg = find_floor(x, z - radius).unwrap_or(fy);

    let r = radius as f32;
    let grad_x = (fx_pos as f32 - fx_neg as f32) / (2.0 * r);
    let grad_z = (fz_pos as f32 - fz_neg as f32) / (2.0 * r);
    let magnitude = (grad_x * grad_x + grad_z * grad_z).sqrt();

    Some((grad_x, grad_z, magnitude))
}

/// Find 3D wall normal at a position using density gradient.
pub(super) fn find_wall_normal_3d(
    density: &DensityField,
    x: usize,
    y: usize,
    z: usize,
    size: usize,
) -> (f32, f32, f32) {
    let sample = |px: usize, py: usize, pz: usize| -> f32 {
        if px < size && py < size && pz < size {
            density.get(px, py, pz).density
        } else {
            0.0
        }
    };

    let gx = sample(x + 1, y, z) - sample(x.saturating_sub(1), y, z);
    let gy = sample(x, y + 1, z) - sample(x, y.saturating_sub(1), z);
    let gz = sample(x, y, z + 1) - sample(x, y, z.saturating_sub(1));

    let mag = (gx * gx + gy * gy + gz * gz).sqrt();
    if mag > 0.001 {
        (gx / mag, gy / mag, gz / mag)
    } else {
        (1.0, 0.0, 0.0)
    }
}

/// Find the dominant horizontal normal direction from a wall surface into air.
pub(super) fn find_wall_normal(
    density: &DensityField,
    x: usize,
    y: usize,
    z: usize,
    size: usize,
) -> (f32, f32) {
    let mut nx: f32 = 0.0;
    let mut nz: f32 = 0.0;

    if x > 0 && density.get(x - 1, y, z).density <= 0.0 {
        nx -= 1.0;
    }
    if x + 1 < size && density.get(x + 1, y, z).density <= 0.0 {
        nx += 1.0;
    }
    if z > 0 && density.get(x, y, z - 1).density <= 0.0 {
        nz -= 1.0;
    }
    if z + 1 < size && density.get(x, y, z + 1).density <= 0.0 {
        nz += 1.0;
    }

    // Normalize
    let len = (nx * nx + nz * nz).sqrt();
    if len > 0.0 {
        (nx / len, nz / len)
    } else {
        (0.0, 0.0)
    }
}

/// Measure contiguous air voxels downward from anchor at (x, anchor_y, z).
/// anchor_y is the solid voxel; air starts at anchor_y - 1.
pub(super) fn measure_air_down(density: &DensityField, x: usize, anchor_y: usize, z: usize, _size: usize) -> usize {
    if anchor_y == 0 {
        return 0;
    }
    let mut gap = 0;
    let mut cy = anchor_y - 1;
    loop {
        if density.get(x, cy, z).density > 0.0 {
            break;
        }
        gap += 1;
        if cy == 0 {
            break;
        }
        cy -= 1;
    }
    gap
}

/// Measure contiguous air voxels upward from anchor at (x, anchor_y, z).
/// anchor_y is the solid voxel; air starts at anchor_y + 1.
pub(super) fn measure_air_up(density: &DensityField, x: usize, anchor_y: usize, z: usize, size: usize) -> usize {
    let mut gap = 0;
    let mut cy = anchor_y + 1;
    while cy < size {
        if density.get(x, cy, z).density > 0.0 {
            break;
        }
        gap += 1;
        cy += 1;
    }
    gap
}

/// Measure contiguous solid voxels downward from anchor (inclusive).
/// anchor_y is a solid floor voxel; counts solid starting from anchor_y downward.
pub(super) fn measure_solid_down(density: &DensityField, x: usize, anchor_y: usize, z: usize) -> usize {
    let mut count = 0;
    let mut cy = anchor_y;
    loop {
        if density.get(x, cy, z).density <= 0.0 {
            break;
        }
        count += 1;
        if cy == 0 {
            break;
        }
        cy -= 1;
    }
    count
}

/// Measure available air space from a wall surface in the normal direction.
pub(super) fn measure_air_extent(
    density: &DensityField,
    x: usize,
    y: usize,
    z: usize,
    nx: f32,
    nz: f32,
    size: usize,
) -> usize {
    let mut extent = 0;
    for step in 1..size {
        let sx = (x as f32 + nx * step as f32).round() as i32;
        let sz = (z as f32 + nz * step as f32).round() as i32;
        if sx < 0 || sx >= size as i32 || sz < 0 || sz >= size as i32 {
            break;
        }
        if density.get(sx as usize, y, sz as usize).density > 0.0 {
            break;
        }
        extent += 1;
    }
    extent
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::material::Material;

    /// Helper: create a density field with a horizontal cave (air slab).
    /// Solid above y_ceil and below y_floor, air in between.
    fn make_cave_field(size: usize, y_floor: usize, y_ceil: usize) -> DensityField {
        let mut field = DensityField::new(size);
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let sample = field.get_mut(x, y, z);
                    if y > y_floor && y < y_ceil {
                        // Air
                        sample.density = -1.0;
                        sample.material = Material::Air;
                    } else {
                        // Solid
                        sample.density = 1.0;
                        sample.material = Material::Limestone;
                    }
                }
            }
        }
        field
    }

    #[test]
    fn test_detect_surfaces_finds_ceiling_and_floor() {
        let field = make_cave_field(17, 4, 12);
        let surfaces = detect_surfaces(&field, 17);

        let has_ceiling = surfaces.iter().any(|s| s.kind == SurfaceKind::Ceiling);
        let has_floor = surfaces.iter().any(|s| s.kind == SurfaceKind::Floor);

        assert!(has_ceiling, "Should detect ceiling surfaces");
        assert!(has_floor, "Should detect floor surfaces");
    }
}
