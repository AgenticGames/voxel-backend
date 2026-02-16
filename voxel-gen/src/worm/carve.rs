use super::path::WormSegment;
use crate::density::DensityField;
use voxel_core::material::Material;

/// Carve worm tunnel into density field using smooth sphere falloff.
///
/// For each WormSegment, finds affected grid cells within the bounding box
/// of the segment's radius, then applies a smooth falloff:
/// `density *= smoothstep(distance / radius) ^ falloff_power`
///
/// This ensures smooth transitions at tunnel edges.
pub fn carve_worm_into_density(
    density: &mut DensityField,
    segments: &[WormSegment],
    world_origin: glam::Vec3,
    falloff_power: f32,
) {
    let size = density.size;

    for segment in segments {
        let local_pos = segment.position - world_origin;
        let r = segment.radius;
        let r2 = r * r;

        // Compute bounding box in grid space
        let min_x = ((local_pos.x - r).floor() as i32).max(0) as usize;
        let min_y = ((local_pos.y - r).floor() as i32).max(0) as usize;
        let min_z = ((local_pos.z - r).floor() as i32).max(0) as usize;
        let max_x = ((local_pos.x + r).ceil() as usize + 1).min(size);
        let max_y = ((local_pos.y + r).ceil() as usize + 1).min(size);
        let max_z = ((local_pos.z + r).ceil() as usize + 1).min(size);

        for z in min_z..max_z {
            for y in min_y..max_y {
                for x in min_x..max_x {
                    let grid_pos = glam::Vec3::new(x as f32, y as f32, z as f32);
                    let dist2 = grid_pos.distance_squared(local_pos);

                    if dist2 < r2 {
                        let t = (dist2 / r2).sqrt();
                        // smoothstep falloff: stronger carving near center
                        let falloff = smoothstep(t).powf(falloff_power);

                        let sample = density.get_mut(x, y, z);
                        // Reduce density toward negative (air)
                        sample.density = sample.density * falloff - (1.0 - falloff);

                        if sample.density <= 0.0 {
                            sample.material = Material::Air;
                        }
                    }
                }
            }
        }
    }
}

/// Smoothstep interpolation for falloff
#[inline]
fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::density::DensityField;
    use glam::Vec3;

    #[test]
    fn test_carve_single_segment() {
        let mut field = DensityField::new(10);
        // All density defaults to 1.0 (solid)

        let segment = WormSegment {
            position: Vec3::new(5.0, 5.0, 5.0),
            radius: 3.0,
        };

        carve_worm_into_density(&mut field, &[segment], Vec3::ZERO, 2.0);

        // Center should be carved (negative density)
        assert!(
            field.get(5, 5, 5).density < 0.0,
            "Center should be air, got {}",
            field.get(5, 5, 5).density
        );

        // Far corner should be untouched
        assert_eq!(field.get(0, 0, 0).density, 1.0);
    }

    #[test]
    fn test_carve_preserves_outside() {
        let mut field = DensityField::new(10);
        let segment = WormSegment {
            position: Vec3::new(5.0, 5.0, 5.0),
            radius: 2.0,
        };

        carve_worm_into_density(&mut field, &[segment], Vec3::ZERO, 2.0);

        // Corners far from the segment should be untouched
        assert_eq!(field.get(0, 0, 0).density, 1.0);
        assert_eq!(field.get(9, 9, 9).density, 1.0);
    }

    #[test]
    fn test_carve_with_world_origin() {
        let mut field = DensityField::new(10);
        let origin = Vec3::new(10.0, 10.0, 10.0);
        let segment = WormSegment {
            position: Vec3::new(15.0, 15.0, 15.0), // center of field in world coords
            radius: 3.0,
        };

        carve_worm_into_density(&mut field, &[segment], origin, 2.0);

        // Grid position (5,5,5) corresponds to world (15,15,15), should be carved
        assert!(field.get(5, 5, 5).density < 0.0);
    }

    #[test]
    fn test_carve_empty_segments() {
        let mut field = DensityField::new(5);
        carve_worm_into_density(&mut field, &[], Vec3::ZERO, 2.0);
        // No changes
        for sample in &field.samples {
            assert_eq!(sample.density, 1.0);
        }
    }
}
