use super::path::WormSegment;
use crate::density::DensityField;
use voxel_core::material::Material;

/// Carve worm tunnel into density field using smooth sphere falloff.
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
                        let falloff = smoothstep(t).powf(falloff_power);
                        let sample = density.get_mut(x, y, z);
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

/// Carve a junction sphere at a worm connection point to guarantee an opening.
///
/// Used at worm start/end positions to ensure tunnels connect to caverns
/// even if the worm path drifts away from the target. Out-of-chunk centers
/// are harmless no-ops due to AABB clamping.
pub fn carve_junction_sphere(
    density: &mut DensityField,
    center: glam::Vec3,
    radius: f32,
    world_origin: glam::Vec3,
    falloff_power: f32,
) {
    let size = density.size;
    let local_pos = center - world_origin;
    let r = radius;
    let r2 = r * r;
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
                    let falloff = smoothstep(t).powf(falloff_power);
                    let sample = density.get_mut(x, y, z);
                    sample.density = sample.density * falloff - (1.0 - falloff);
                    if sample.density <= 0.0 {
                        sample.material = Material::Air;
                    }
                }
            }
        }
    }
}

/// Check if any worm segment actually overlaps a chunk's density grid.
pub fn worm_overlaps_chunk(
    segments: &[WormSegment],
    world_origin: glam::Vec3,
    grid_size: usize,
) -> bool {
    let size_f = grid_size as f32;
    for segment in segments {
        let local = segment.position - world_origin;
        let r = segment.radius;
        let nx = local.x.clamp(0.0, size_f);
        let ny = local.y.clamp(0.0, size_f);
        let nz = local.z.clamp(0.0, size_f);
        let dx = local.x - nx;
        let dy = local.y - ny;
        let dz = local.z - nz;
        if dx * dx + dy * dy + dz * dz <= r * r {
            return true;
        }
    }
    false
}

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
        let segment = WormSegment {
            position: Vec3::new(5.0, 5.0, 5.0),
            radius: 3.0,
        };
        carve_worm_into_density(&mut field, &[segment], Vec3::ZERO, 2.0);
        assert!(
            field.get(5, 5, 5).density < 0.0,
            "Center should be air, got {}",
            field.get(5, 5, 5).density
        );
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
        assert_eq!(field.get(0, 0, 0).density, 1.0);
        assert_eq!(field.get(9, 9, 9).density, 1.0);
    }

    #[test]
    fn test_carve_with_world_origin() {
        let mut field = DensityField::new(10);
        let origin = Vec3::new(10.0, 10.0, 10.0);
        let segment = WormSegment {
            position: Vec3::new(15.0, 15.0, 15.0),
            radius: 3.0,
        };
        carve_worm_into_density(&mut field, &[segment], origin, 2.0);
        assert!(field.get(5, 5, 5).density < 0.0);
    }

    #[test]
    fn test_carve_empty_segments() {
        let mut field = DensityField::new(5);
        carve_worm_into_density(&mut field, &[], Vec3::ZERO, 2.0);
        for sample in &field.samples {
            assert_eq!(sample.density, 1.0);
        }
    }

    #[test]
    fn test_worm_overlaps_chunk_inside() {
        let segments = vec![WormSegment {
            position: glam::Vec3::new(8.0, 8.0, 8.0),
            radius: 3.0,
        }];
        assert!(worm_overlaps_chunk(&segments, glam::Vec3::ZERO, 17));
    }

    #[test]
    fn test_worm_overlaps_chunk_far_outside() {
        let segments = vec![WormSegment {
            position: glam::Vec3::new(100.0, 100.0, 100.0),
            radius: 3.0,
        }];
        assert!(!worm_overlaps_chunk(&segments, glam::Vec3::ZERO, 17));
    }

    #[test]
    fn test_worm_overlaps_chunk_edge_touch() {
        let segments = vec![WormSegment {
            position: glam::Vec3::new(-2.0, 8.0, 8.0),
            radius: 3.0,
        }];
        assert!(worm_overlaps_chunk(&segments, glam::Vec3::ZERO, 17));
    }

    #[test]
    fn test_worm_overlaps_chunk_just_outside() {
        let segments = vec![WormSegment {
            position: glam::Vec3::new(-4.0, 8.0, 8.0),
            radius: 3.0,
        }];
        assert!(!worm_overlaps_chunk(&segments, glam::Vec3::ZERO, 17));
    }

    #[test]
    fn test_carve_junction_sphere_center_carved() {
        let mut field = DensityField::new(10);
        carve_junction_sphere(
            &mut field,
            glam::Vec3::new(5.0, 5.0, 5.0),
            3.0,
            glam::Vec3::ZERO,
            2.0,
        );
        assert!(
            field.get(5, 5, 5).density < 0.0,
            "Junction sphere center should be carved, got {}",
            field.get(5, 5, 5).density
        );
        assert_eq!(field.get(0, 0, 0).density, 1.0);
    }

    #[test]
    fn test_carve_junction_sphere_out_of_chunk_is_noop() {
        let mut field = DensityField::new(10);
        carve_junction_sphere(
            &mut field,
            glam::Vec3::new(200.0, 200.0, 200.0),
            3.0,
            glam::Vec3::ZERO,
            2.0,
        );
        for sample in &field.samples {
            assert_eq!(sample.density, 1.0, "Out-of-chunk sphere should not modify field");
        }
    }
}
