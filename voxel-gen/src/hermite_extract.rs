use crate::density::DensityField;
use voxel_core::hermite::{EdgeIntersection, EdgeKey, HermiteData};

/// Extract hermite data from density field by detecting sign changes along edges.
///
/// Walks all grid edges (X, Y, Z axes). For each edge where the sign changes
/// between endpoints, interpolates the intersection t parameter and estimates
/// the surface normal via central differences on the density field.
pub fn extract_hermite_data(density: &DensityField) -> HermiteData {
    let size = density.size;
    // Pre-allocate: typically ~15-25% of edges have sign changes
    let estimated_edges = size * size * size / 5;
    let mut data = HermiteData::with_capacity(estimated_edges);

    // Walk X-edges (axis = 0)
    for z in 0..size {
        for y in 0..size {
            for x in 0..size - 1 {
                let da = density.get(x, y, z).density;
                let db = density.get(x + 1, y, z).density;
                if (da > 0.0) != (db > 0.0) {
                    let t = da / (da - db);
                    let normal = estimate_normal_fast(density, x, y, z, t, 0);
                    let mat_a = density.get(x, y, z).material;
                    let mat_b = density.get(x + 1, y, z).material;
                    let material = if mat_a.is_solid() && mat_b.is_solid() {
                        if da > 0.0 { mat_a } else { mat_b }
                    } else if mat_a.is_solid() {
                        mat_a
                    } else if mat_b.is_solid() {
                        mat_b
                    } else {
                        mat_a
                    };
                    debug_assert!(material.is_solid(),
                        "Hermite edge at ({},{},{}) axis 0 has Air material: mat_a={:?} mat_b={:?} da={} db={}",
                        x, y, z, mat_a, mat_b, da, db);
                    let key = EdgeKey::new(x as u8, y as u8, z as u8, 0);
                    data.edges.insert(key, EdgeIntersection { t, normal, material });
                }
            }
        }
    }

    // Walk Y-edges (axis = 1)
    for z in 0..size {
        for y in 0..size - 1 {
            for x in 0..size {
                let da = density.get(x, y, z).density;
                let db = density.get(x, y + 1, z).density;
                if (da > 0.0) != (db > 0.0) {
                    let t = da / (da - db);
                    let normal = estimate_normal_fast(density, x, y, z, t, 1);
                    let mat_a = density.get(x, y, z).material;
                    let mat_b = density.get(x, y + 1, z).material;
                    let material = if mat_a.is_solid() && mat_b.is_solid() {
                        if da > 0.0 { mat_a } else { mat_b }
                    } else if mat_a.is_solid() {
                        mat_a
                    } else if mat_b.is_solid() {
                        mat_b
                    } else {
                        mat_a
                    };
                    debug_assert!(material.is_solid(),
                        "Hermite edge at ({},{},{}) axis 1 has Air material: mat_a={:?} mat_b={:?} da={} db={}",
                        x, y, z, mat_a, mat_b, da, db);
                    let key = EdgeKey::new(x as u8, y as u8, z as u8, 1);
                    data.edges.insert(key, EdgeIntersection { t, normal, material });
                }
            }
        }
    }

    // Walk Z-edges (axis = 2)
    for z in 0..size - 1 {
        for y in 0..size {
            for x in 0..size {
                let da = density.get(x, y, z).density;
                let db = density.get(x, y, z + 1).density;
                if (da > 0.0) != (db > 0.0) {
                    let t = da / (da - db);
                    let normal = estimate_normal_fast(density, x, y, z, t, 2);
                    let mat_a = density.get(x, y, z).material;
                    let mat_b = density.get(x, y, z + 1).material;
                    let material = if mat_a.is_solid() && mat_b.is_solid() {
                        if da > 0.0 { mat_a } else { mat_b }
                    } else if mat_a.is_solid() {
                        mat_a
                    } else if mat_b.is_solid() {
                        mat_b
                    } else {
                        mat_a
                    };
                    debug_assert!(material.is_solid(),
                        "Hermite edge at ({},{},{}) axis 2 has Air material: mat_a={:?} mat_b={:?} da={} db={}",
                        x, y, z, mat_a, mat_b, da, db);
                    let key = EdgeKey::new(x as u8, y as u8, z as u8, 2);
                    data.edges.insert(key, EdgeIntersection { t, normal, material });
                }
            }
        }
    }

    data
}

/// Incrementally update hermite data: remove old edges in the dirty region,
/// re-extract edges only within [min..max] bounds (inclusive), then insert
/// into the existing hermite data. Much faster than full extraction when
/// only a small region changed (e.g., after mining).
pub fn patch_hermite_data(
    hermite: &mut HermiteData,
    density: &DensityField,
    min_x: usize, min_y: usize, min_z: usize,
    max_x: usize, max_y: usize, max_z: usize,
) {
    let size = density.size;
    // Clamp to valid range
    let max_x = max_x.min(size - 1);
    let max_y = max_y.min(size - 1);
    let max_z = max_z.min(size - 1);

    // Remove all existing edges in the dirty region
    hermite.edges.remove_in_range(min_x, max_x, min_y, max_y, min_z, max_z);

    // Re-extract X-edges in range
    for z in min_z..=max_z {
        for y in min_y..=max_y {
            for x in min_x..max_x.min(size - 2) + 1 {
                let da = density.get(x, y, z).density;
                let db = density.get(x + 1, y, z).density;
                if (da > 0.0) != (db > 0.0) {
                    let t = da / (da - db);
                    let normal = estimate_normal_fast(density, x, y, z, t, 0);
                    let mat_a = density.get(x, y, z).material;
                    let mat_b = density.get(x + 1, y, z).material;
                    let material = if mat_a.is_solid() && mat_b.is_solid() {
                        if da > 0.0 { mat_a } else { mat_b }
                    } else if mat_a.is_solid() {
                        mat_a
                    } else if mat_b.is_solid() {
                        mat_b
                    } else {
                        mat_a
                    };
                    debug_assert!(material.is_solid(),
                        "Hermite patch edge at ({},{},{}) axis 0 has Air material: mat_a={:?} mat_b={:?} da={} db={}",
                        x, y, z, mat_a, mat_b, da, db);
                    hermite.edges.insert(
                        EdgeKey::new(x as u8, y as u8, z as u8, 0),
                        EdgeIntersection { t, normal, material },
                    );
                }
            }
        }
    }

    // Re-extract Y-edges in range
    for z in min_z..=max_z {
        for y in min_y..max_y.min(size - 2) + 1 {
            for x in min_x..=max_x {
                let da = density.get(x, y, z).density;
                let db = density.get(x, y + 1, z).density;
                if (da > 0.0) != (db > 0.0) {
                    let t = da / (da - db);
                    let normal = estimate_normal_fast(density, x, y, z, t, 1);
                    let mat_a = density.get(x, y, z).material;
                    let mat_b = density.get(x, y + 1, z).material;
                    let material = if mat_a.is_solid() && mat_b.is_solid() {
                        if da > 0.0 { mat_a } else { mat_b }
                    } else if mat_a.is_solid() {
                        mat_a
                    } else if mat_b.is_solid() {
                        mat_b
                    } else {
                        mat_a
                    };
                    debug_assert!(material.is_solid(),
                        "Hermite patch edge at ({},{},{}) axis 1 has Air material: mat_a={:?} mat_b={:?} da={} db={}",
                        x, y, z, mat_a, mat_b, da, db);
                    hermite.edges.insert(
                        EdgeKey::new(x as u8, y as u8, z as u8, 1),
                        EdgeIntersection { t, normal, material },
                    );
                }
            }
        }
    }

    // Re-extract Z-edges in range
    for z in min_z..max_z.min(size - 2) + 1 {
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let da = density.get(x, y, z).density;
                let db = density.get(x, y, z + 1).density;
                if (da > 0.0) != (db > 0.0) {
                    let t = da / (da - db);
                    let normal = estimate_normal_fast(density, x, y, z, t, 2);
                    let mat_a = density.get(x, y, z).material;
                    let mat_b = density.get(x, y, z + 1).material;
                    let material = if mat_a.is_solid() && mat_b.is_solid() {
                        if da > 0.0 { mat_a } else { mat_b }
                    } else if mat_a.is_solid() {
                        mat_a
                    } else if mat_b.is_solid() {
                        mat_b
                    } else {
                        mat_a
                    };
                    debug_assert!(material.is_solid(),
                        "Hermite patch edge at ({},{},{}) axis 2 has Air material: mat_a={:?} mat_b={:?} da={} db={}",
                        x, y, z, mat_a, mat_b, da, db);
                    hermite.edges.insert(
                        EdgeKey::new(x as u8, y as u8, z as u8, 2),
                        EdgeIntersection { t, normal, material },
                    );
                }
            }
        }
    }
}

/// Fast normal estimation using grid-aligned central differences.
/// Uses the grid point nearest to the intersection instead of trilinear interpolation.
/// 6 density lookups instead of 48 (8x faster than trilinear approach).
#[inline]
fn estimate_normal_fast(
    density: &DensityField,
    x: usize, y: usize, z: usize,
    t: f32, axis: u8,
) -> glam::Vec3 {
    let s = density.size;

    // Pick the grid point nearest to the intersection
    let (gx, gy, gz) = match axis {
        0 => (if t < 0.5 { x } else { x + 1 }, y, z),
        1 => (x, if t < 0.5 { y } else { y + 1 }, z),
        _ => (x, y, if t < 0.5 { z } else { z + 1 }),
    };

    // Central differences at the grid point (clamped to bounds)
    let dx = if gx > 0 && gx + 1 < s {
        density.get(gx + 1, gy, gz).density - density.get(gx - 1, gy, gz).density
    } else if gx + 1 < s {
        density.get(gx + 1, gy, gz).density - density.get(gx, gy, gz).density
    } else if gx > 0 {
        density.get(gx, gy, gz).density - density.get(gx - 1, gy, gz).density
    } else {
        0.0
    };

    let dy = if gy > 0 && gy + 1 < s {
        density.get(gx, gy + 1, gz).density - density.get(gx, gy - 1, gz).density
    } else if gy + 1 < s {
        density.get(gx, gy + 1, gz).density - density.get(gx, gy, gz).density
    } else if gy > 0 {
        density.get(gx, gy, gz).density - density.get(gx, gy - 1, gz).density
    } else {
        0.0
    };

    let dz = if gz > 0 && gz + 1 < s {
        density.get(gx, gy, gz + 1).density - density.get(gx, gy, gz - 1).density
    } else if gz + 1 < s {
        density.get(gx, gy, gz + 1).density - density.get(gx, gy, gz).density
    } else if gz > 0 {
        density.get(gx, gy, gz).density - density.get(gx, gy, gz - 1).density
    } else {
        0.0
    };

    let normal = glam::Vec3::new(dx, dy, dz);
    let len = normal.length();
    if len > 1e-8 {
        normal / len
    } else {
        glam::Vec3::Y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::material::Material;

    #[test]
    fn test_extract_empty_field_all_solid() {
        // All positive density -> no sign changes -> empty hermite data
        let field = DensityField::new(4);
        let data = extract_hermite_data(&field);
        assert!(data.edges.is_empty());
    }

    #[test]
    fn test_extract_with_sign_change() {
        let mut field = DensityField::new(4);
        // Set one voxel to air (negative density)
        let sample = field.get_mut(1, 1, 1);
        sample.density = -0.5;
        sample.material = Material::Air;

        let data = extract_hermite_data(&field);
        // Should have sign changes on edges adjacent to (1,1,1)
        assert!(!data.edges.is_empty());
    }

    #[test]
    fn test_extract_from_generated_field() {
        let config = crate::config::GenerationConfig::default();
        let origin = glam::Vec3::ZERO;
        let field = crate::density::generate_density_field(&config, origin);
        let data = extract_hermite_data(&field);
        // Should produce some hermite data (depends on noise, but should be consistent)
        // At minimum, verify it runs without panic
        let _ = data.edges.len();
    }

    #[test]
    fn test_material_prefers_solid_over_air() {
        // Regression: if one endpoint has positive density but Air material
        // (corrupted boundary state), the hermite edge should get the solid
        // material from the other endpoint instead of Air.
        let mut field = DensityField::new(4);

        // (0,0,0): positive density, Air material (corrupted)
        let a = field.get_mut(0, 0, 0);
        a.density = 0.5;
        a.material = Material::Air;

        // (1,0,0): negative density, Granite material (solid, air side)
        let b = field.get_mut(1, 0, 0);
        b.density = -0.3;
        b.material = Material::Granite;

        let data = extract_hermite_data(&field);
        let key = EdgeKey::new(0, 0, 0, 0); // X-edge at (0,0,0)
        let edge = data.edges.get(&key).expect("Expected sign change at X-edge (0,0,0)");

        // Old code would pick Air (da > 0 → endpoint A). New code should pick Granite.
        assert_eq!(edge.material, Material::Granite,
            "Should prefer solid material over Air at boundary");
    }

    #[test]
    fn test_material_both_solid_uses_density_sign() {
        // When both endpoints are solid, material should follow density sign
        // (positive density = solid side = that material).
        let mut field = DensityField::new(4);

        // (0,0,0): positive density, Granite
        let a = field.get_mut(0, 0, 0);
        a.density = 0.5;
        a.material = Material::Granite;

        // (1,0,0): negative density, Iron
        let b = field.get_mut(1, 0, 0);
        b.density = -0.3;
        b.material = Material::Iron;

        let data = extract_hermite_data(&field);
        let key = EdgeKey::new(0, 0, 0, 0);
        let edge = data.edges.get(&key).expect("Expected sign change at X-edge (0,0,0)");

        // da > 0, both solid → pick mat_a (Granite)
        assert_eq!(edge.material, Material::Granite);
    }

    #[test]
    fn test_interpolation_t_range() {
        let mut field = DensityField::new(4);
        // Create a single sign change along x-edge at (0,0,0)
        field.get_mut(0, 0, 0).density = 0.3;
        field.get_mut(1, 0, 0).density = -0.7;

        let data = extract_hermite_data(&field);
        let key = EdgeKey::new(0, 0, 0, 0);
        if let Some(edge) = data.edges.get(&key) {
            assert!(edge.t >= 0.0 && edge.t <= 1.0, "t should be in [0,1], got {}", edge.t);
            // t = 0.3 / (0.3 - (-0.7)) = 0.3 / 1.0 = 0.3
            assert!((edge.t - 0.3).abs() < 0.01);
        } else {
            panic!("Expected sign change at edge (0,0,0,0)");
        }
    }
}
