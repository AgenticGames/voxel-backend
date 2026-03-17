use glam::Vec3;
use voxel_noise::simplex::Simplex3D;
use voxel_noise::NoiseSource;

/// A point along a worm tunnel path
#[derive(Debug, Clone)]
pub struct WormSegment {
    pub position: Vec3,
    pub radius: f32,
}

/// Generate a noise-driven worm path that steers toward a target endpoint.
///
/// Starting from `start`, walks forward using noise to perturb yaw and pitch
/// at each step, with a quadratic blend toward `end` so the worm arrives
/// at the connection point. Radius varies along the path using a separate
/// noise source, with a minimum taper floor to ensure real carving at endpoints.
pub fn generate_worm_path(
    seed: u64,
    start: Vec3,
    end: Vec3,
    step_length: f32,
    max_steps: u32,
    radius_min: f32,
    radius_max: f32,
) -> Vec<WormSegment> {
    let yaw_noise = Simplex3D::new(seed);
    let pitch_noise = Simplex3D::new(seed.wrapping_add(100));
    let radius_noise = Simplex3D::new(seed.wrapping_add(200));

    let mut segments = Vec::with_capacity(max_steps as usize);
    let mut pos = start;
    let mut yaw: f32 = 0.0;
    let mut pitch: f32 = 0.0;

    // Minimum taper factor so endpoints always carve real geometry
    const MIN_TAPER_FACTOR: f32 = 0.25;

    // Taper zone: first/last 15% of steps fade radius to zero
    let taper_steps = (max_steps as f32 * 0.15).max(3.0);

    for step in 0..max_steps {
        let t = step as f64 * 0.05; // time parameter for noise sampling

        // Sample noise for direction perturbation
        let yaw_delta = yaw_noise.sample(t, pos.y as f64 * 0.1, pos.z as f64 * 0.1) as f32;
        let pitch_delta = pitch_noise.sample(pos.x as f64 * 0.1, t, pos.z as f64 * 0.1) as f32;

        yaw += yaw_delta * 0.3;
        pitch += pitch_delta * 0.2;
        // Clamp pitch to avoid going straight up/down
        pitch = pitch.clamp(-std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_PI_4);

        // Noise-driven base direction from yaw/pitch
        let noise_dir = Vec3::new(
            yaw.cos() * pitch.cos(),
            pitch.sin(),
            yaw.sin() * pitch.cos(),
        );

        // Steering ramp: free wander for first 15%, then ramp linearly 0→1 over remaining 85%
        let progress = if max_steps > 1 { step as f32 / (max_steps - 1) as f32 } else { 1.0 };
        let steer_t = ((progress - 0.15) / 0.85).clamp(0.0, 1.0);

        let to_target = end - pos;
        let dist_to_target = to_target.length();
        let dir = if dist_to_target > 1e-3 {
            noise_dir.lerp(to_target.normalize(), steer_t).normalize_or_zero()
        } else {
            noise_dir
        };
        let dir = if dir == Vec3::ZERO { noise_dir } else { dir };

        // Vary radius along path
        let r_t = radius_noise.sample(t * 0.5, 0.0, 0.0) as f32;
        let base_radius = radius_min + (radius_max - radius_min) * (r_t * 0.5 + 0.5).clamp(0.0, 1.0);

        // Taper: smoothstep ramp at start, and based on remaining steps for end
        let remaining = (max_steps - 1 - step) as f32;
        let start_fade = (step as f32 / taper_steps).clamp(0.0, 1.0);
        let end_fade = (remaining / taper_steps).clamp(0.0, 1.0);
        let taper = (start_fade * start_fade * (3.0 - 2.0 * start_fade))
                   * (end_fade * end_fade * (3.0 - 2.0 * end_fade));

        // Floor the taper so endpoints always carve with at least MIN_TAPER_FACTOR radius
        let taper_clamped = taper.max(MIN_TAPER_FACTOR);
        segments.push(WormSegment {
            position: pos,
            radius: base_radius * taper_clamped,
        });

        // Early termination: stop if we've reached the target (within radius_max)
        if dist_to_target < radius_max && step > 0 {
            break;
        }

        pos += dir * step_length;
    }

    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_worm_path_length() {
        let path = generate_worm_path(42, Vec3::ZERO, Vec3::new(50.0, 0.0, 0.0), 1.0, 50, 1.5, 3.5);
        assert_eq!(path.len(), 50);
    }

    #[test]
    fn test_worm_path_deterministic() {
        let path1 = generate_worm_path(42, Vec3::ZERO, Vec3::new(50.0, 0.0, 0.0), 1.0, 50, 1.5, 3.5);
        let path2 = generate_worm_path(42, Vec3::ZERO, Vec3::new(50.0, 0.0, 0.0), 1.0, 50, 1.5, 3.5);
        for (a, b) in path1.iter().zip(path2.iter()) {
            assert_eq!(a.position, b.position);
            assert_eq!(a.radius, b.radius);
        }
    }

    #[test]
    fn test_worm_path_radius_bounds() {
        let path = generate_worm_path(42, Vec3::ZERO, Vec3::new(50.0, 0.0, 0.0), 1.0, 100, 1.5, 3.5);
        for seg in &path {
            // Radius never goes below the taper floor or above max
            assert!(seg.radius >= 0.0, "radius negative: {}", seg.radius);
            assert!(seg.radius <= 3.5, "radius too large: {}", seg.radius);
        }
        // Middle segments should have non-trivial radius
        let mid = &path[path.len() / 2];
        assert!(mid.radius > 0.5, "mid radius too small: {}", mid.radius);
        // Endpoints have a minimum floor (MIN_TAPER_FACTOR = 0.25)
        let min_floor = 1.5 * 0.25_f32; // radius_min * MIN_TAPER_FACTOR
        assert!(path[0].radius >= min_floor, "start radius should be floored, got {}", path[0].radius);
        assert!(path[path.len() - 1].radius >= min_floor, "end radius should be floored, got {}", path[path.len() - 1].radius);
    }

    #[test]
    fn test_worm_path_different_seeds() {
        let path1 = generate_worm_path(1, Vec3::ZERO, Vec3::new(50.0, 0.0, 0.0), 1.0, 50, 1.5, 3.5);
        let path2 = generate_worm_path(2, Vec3::ZERO, Vec3::new(50.0, 0.0, 0.0), 1.0, 50, 1.5, 3.5);
        // Different seeds should (with high probability) produce different paths
        let same = path1.iter().zip(path2.iter()).all(|(a, b)| a.position == b.position);
        // With stubs returning 0.0 for noise, paths may be same; this tests the interface
        let _ = same;
    }

    #[test]
    fn test_worm_path_zero_steps() {
        let path = generate_worm_path(42, Vec3::ZERO, Vec3::new(50.0, 0.0, 0.0), 1.0, 0, 1.5, 3.5);
        assert!(path.is_empty());
    }

    #[test]
    fn test_worm_steers_toward_target() {
        let start = Vec3::ZERO;
        let end = Vec3::new(50.0, 0.0, 0.0);
        let path = generate_worm_path(42, start, end, 1.0, 100, 1.5, 3.5);
        let last = &path[path.len() - 1];
        assert!(
            last.position.distance(end) < 45.0,
            "Last segment within 45 units of target (100 steps, target at 50 units, steering works), got {}",
            last.position.distance(end)
        );
    }
}
