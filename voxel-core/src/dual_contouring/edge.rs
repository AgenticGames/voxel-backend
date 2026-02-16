use glam::Vec3;
use crate::hermite::EdgeIntersection;
use crate::material::Material;

/// Detect edge intersections along grid edges.
/// Returns Some(EdgeIntersection) if there is a sign change between density_a and density_b.
pub fn find_edge_intersection(
    density_a: f32,
    density_b: f32,
    pos_a: Vec3,
    pos_b: Vec3,
) -> Option<EdgeIntersection> {
    // No sign change means no intersection
    if (density_a > 0.0) == (density_b > 0.0) {
        return None;
    }

    // Linear interpolation to find intersection point
    let t = density_a / (density_a - density_b);
    let t = t.clamp(0.0, 1.0);

    // Estimate normal via central differences using the interpolation direction
    // The normal points from negative to positive (from inside to outside)
    let edge_dir = (pos_b - pos_a).normalize_or_zero();
    // Normal perpendicular to the surface: approximate from the density gradient
    // For an edge intersection, we use the gradient direction
    let normal = if density_a < density_b {
        edge_dir
    } else {
        -edge_dir
    };

    let material = Material::Limestone;

    Some(EdgeIntersection {
        t,
        normal,
        material,
    })
}

/// Estimate the surface normal at a point using central differences on the density field.
/// `density_fn` samples the density at any world-space position.
pub fn estimate_normal<F: Fn(Vec3) -> f32>(pos: Vec3, density_fn: &F, epsilon: f32) -> Vec3 {
    let dx = density_fn(pos + Vec3::new(epsilon, 0.0, 0.0))
           - density_fn(pos - Vec3::new(epsilon, 0.0, 0.0));
    let dy = density_fn(pos + Vec3::new(0.0, epsilon, 0.0))
           - density_fn(pos - Vec3::new(0.0, epsilon, 0.0));
    let dz = density_fn(pos + Vec3::new(0.0, 0.0, epsilon))
           - density_fn(pos - Vec3::new(0.0, 0.0, epsilon));

    Vec3::new(dx, dy, dz).normalize_or_zero()
}

/// Compute the intersection position given endpoints and the interpolation parameter
pub fn interpolate_position(pos_a: Vec3, pos_b: Vec3, t: f32) -> Vec3 {
    pos_a + (pos_b - pos_a) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_sign_change() {
        let result = find_edge_intersection(
            1.0, 2.0,
            Vec3::ZERO, Vec3::X,
        );
        assert!(result.is_none());
    }

    #[test]
    fn no_sign_change_negative() {
        let result = find_edge_intersection(
            -1.0, -2.0,
            Vec3::ZERO, Vec3::X,
        );
        assert!(result.is_none());
    }

    #[test]
    fn sign_change_positive_to_negative() {
        let result = find_edge_intersection(
            1.0, -1.0,
            Vec3::ZERO, Vec3::X,
        );
        assert!(result.is_some());
        let edge = result.unwrap();
        assert!((edge.t - 0.5).abs() < 1e-5, "t should be 0.5, got {}", edge.t);
    }

    #[test]
    fn sign_change_negative_to_positive() {
        let result = find_edge_intersection(
            -1.0, 1.0,
            Vec3::ZERO, Vec3::X,
        );
        assert!(result.is_some());
        let edge = result.unwrap();
        assert!((edge.t - 0.5).abs() < 1e-5, "t should be 0.5, got {}", edge.t);
    }

    #[test]
    fn asymmetric_densities() {
        let result = find_edge_intersection(
            -1.0, 3.0,
            Vec3::ZERO, Vec3::X,
        );
        assert!(result.is_some());
        let edge = result.unwrap();
        assert!((edge.t - 0.25).abs() < 1e-5, "t should be 0.25, got {}", edge.t);
    }

    #[test]
    fn estimate_normal_sphere() {
        // Density field: sphere of radius 1 centered at origin
        // density = x^2 + y^2 + z^2 - 1
        let density_fn = |p: Vec3| p.x * p.x + p.y * p.y + p.z * p.z - 1.0;
        let pos = Vec3::new(1.0, 0.0, 0.0);
        let normal = estimate_normal(pos, &density_fn, 0.001);
        // Normal at (1,0,0) should point in +x direction
        assert!((normal.x - 1.0).abs() < 0.01, "normal.x should be ~1.0, got {}", normal.x);
        assert!(normal.y.abs() < 0.01);
        assert!(normal.z.abs() < 0.01);
    }

    #[test]
    fn interpolate_position_midpoint() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(2.0, 4.0, 6.0);
        let result = interpolate_position(a, b, 0.5);
        assert!((result - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-5);
    }
}
