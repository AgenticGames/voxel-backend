use glam::{Mat3, Vec3};

/// QEF (Quadratic Error Function) accumulator.
/// Uses raw f32 fields for the symmetric 3x3 ATA matrix to avoid
/// Mat3 conversion overhead in the hot `add()` path.
#[derive(Debug, Clone)]
pub struct QefData {
    // Upper-triangular of symmetric ATA: [a00, a01, a02, a11, a12, a22]
    pub ata: [f32; 6],
    pub atb: Vec3,
    pub btb: f32,
    pub mass_point_sum: Vec3,
    pub count: u32,
}

impl Default for QefData {
    fn default() -> Self {
        Self {
            ata: [0.0; 6],
            atb: Vec3::ZERO,
            btb: 0.0,
            mass_point_sum: Vec3::ZERO,
            count: 0,
        }
    }
}

impl QefData {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an intersection point + normal to the QEF.
    /// Each intersection contributes the constraint: dot(normal, x - intersection) = 0.
    #[inline]
    pub fn add(&mut self, intersection: Vec3, normal: Vec3) {
        let nx = normal.x;
        let ny = normal.y;
        let nz = normal.z;
        let d = nx * intersection.x + ny * intersection.y + nz * intersection.z;

        // ATA += outer_product(n, n) — symmetric, only 6 unique elements
        self.ata[0] += nx * nx;
        self.ata[1] += nx * ny;
        self.ata[2] += nx * nz;
        self.ata[3] += ny * ny;
        self.ata[4] += ny * nz;
        self.ata[5] += nz * nz;

        self.atb += normal * d;
        self.btb += d * d;
        self.mass_point_sum += intersection;
        self.count += 1;
    }

    /// Merge another QEF into this one
    pub fn merge(&mut self, other: &QefData) {
        for i in 0..6 {
            self.ata[i] += other.ata[i];
        }
        self.atb += other.atb;
        self.btb += other.btb;
        self.mass_point_sum += other.mass_point_sum;
        self.count += other.count;
    }

    /// Reconstruct ATA as Mat3 for solve operations.
    #[inline]
    fn ata_mat3(&self) -> Mat3 {
        Mat3::from_cols(
            Vec3::new(self.ata[0], self.ata[1], self.ata[2]),
            Vec3::new(self.ata[1], self.ata[3], self.ata[4]),
            Vec3::new(self.ata[2], self.ata[4], self.ata[5]),
        )
    }

    /// Solve the QEF using 3x3 Jacobi SVD.
    /// Returns the optimal vertex position that minimizes quadratic error.
    pub fn solve(&self) -> Vec3 {
        if self.count == 0 {
            return Vec3::ZERO;
        }

        let mass_point = self.mass_point_sum / self.count as f32;
        let ata = self.ata_mat3();

        // Shift the problem to be relative to mass_point for numerical stability
        let shifted_atb = self.atb - ata * mass_point;

        // Solve ATA * x_shifted = shifted_atb using Jacobi SVD
        let (u, sigma, v) = jacobi_svd_3x3(ata);

        // Pseudo-inverse: V * Sigma^-1 * U^T
        let threshold = 0.1;
        let mut sigma_inv = [0.0f32; 3];
        for i in 0..3 {
            if sigma[i] > threshold {
                sigma_inv[i] = 1.0 / sigma[i];
            }
        }

        let ut_b = Vec3::new(
            u.col(0).dot(shifted_atb),
            u.col(1).dot(shifted_atb),
            u.col(2).dot(shifted_atb),
        );

        let scaled = Vec3::new(
            ut_b.x * sigma_inv[0],
            ut_b.y * sigma_inv[1],
            ut_b.z * sigma_inv[2],
        );

        mass_point + v * scaled
    }

    /// Solve the QEF with clamping to a bounding box defined by min_bound and max_bound
    pub fn solve_clamped(&self, min_bound: Vec3, max_bound: Vec3) -> Vec3 {
        self.solve().clamp(min_bound, max_bound)
    }

    /// Compute the error for a given position
    pub fn error(&self, pos: Vec3) -> f32 {
        let atax = self.ata_mat3() * pos;
        pos.dot(atax) - 2.0 * self.atb.dot(pos) + self.btb
    }
}

/// 3x3 Jacobi eigendecomposition for symmetric matrices
/// Returns eigenvalues and eigenvectors
#[allow(clippy::needless_range_loop)]
fn jacobi_eigen_3x3(mat: Mat3) -> (Vec3, Mat3) {
    let mut a = [
        [mat.col(0).x, mat.col(1).x, mat.col(2).x],
        [mat.col(0).y, mat.col(1).y, mat.col(2).y],
        [mat.col(0).z, mat.col(1).z, mat.col(2).z],
    ];

    let mut v = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    let max_iters = 50;
    for _ in 0..max_iters {
        // Find largest off-diagonal element
        let mut p = 0;
        let mut q = 1;
        let mut max_val = a[0][1].abs();

        if a[0][2].abs() > max_val {
            p = 0;
            q = 2;
            max_val = a[0][2].abs();
        }
        if a[1][2].abs() > max_val {
            p = 1;
            q = 2;
            max_val = a[1][2].abs();
        }

        if max_val < 1e-10 {
            break;
        }

        // Compute rotation
        let diff = a[q][q] - a[p][p];
        let t = if diff.abs() < 1e-10 {
            1.0f32
        } else {
            let tau = diff / (2.0 * a[p][q]);
            let sign = if tau >= 0.0 { 1.0 } else { -1.0 };
            sign / (tau.abs() + (1.0 + tau * tau).sqrt())
        };

        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        // Apply Jacobi rotation
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];

        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        // Update other elements
        for r in 0..3 {
            if r != p && r != q {
                let arp = a[r][p];
                let arq = a[r][q];
                a[r][p] = c * arp - s * arq;
                a[p][r] = a[r][p];
                a[r][q] = s * arp + c * arq;
                a[q][r] = a[r][q];
            }
        }

        // Update eigenvectors
        for r in 0..3 {
            let vrp = v[r][p];
            let vrq = v[r][q];
            v[r][p] = c * vrp - s * vrq;
            v[r][q] = s * vrp + c * vrq;
        }
    }

    let eigenvalues = Vec3::new(a[0][0], a[1][1], a[2][2]);
    let eigenvectors = Mat3::from_cols(
        Vec3::new(v[0][0], v[1][0], v[2][0]),
        Vec3::new(v[0][1], v[1][1], v[2][1]),
        Vec3::new(v[0][2], v[1][2], v[2][2]),
    );

    (eigenvalues, eigenvectors)
}

/// SVD of a 3x3 matrix via Jacobi eigendecomposition of ATA
/// Returns (U, singular_values, V) where A = U * diag(S) * V^T
fn jacobi_svd_3x3(mat: Mat3) -> (Mat3, [f32; 3], Mat3) {
    // For symmetric positive semi-definite matrices (like ATA),
    // the eigendecomposition IS the SVD: ATA = V * S^2 * V^T
    // Since ATA is symmetric, U = V for our purposes.
    let (eigenvalues, v) = jacobi_eigen_3x3(mat);

    let sigma = [
        eigenvalues.x.abs().sqrt(),
        eigenvalues.y.abs().sqrt(),
        eigenvalues.z.abs().sqrt(),
    ];

    // For symmetric matrices, U = V
    (v, sigma, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qef_single_plane() {
        // A horizontal plane at y=0.5
        // Multiple intersections along edges with normal pointing up
        let mut qef = QefData::new();
        let normal = Vec3::new(0.0, 1.0, 0.0);

        qef.add(Vec3::new(0.0, 0.5, 0.0), normal);
        qef.add(Vec3::new(1.0, 0.5, 0.0), normal);
        qef.add(Vec3::new(0.0, 0.5, 1.0), normal);
        qef.add(Vec3::new(1.0, 0.5, 1.0), normal);

        let result = qef.solve();
        // The vertex should be at y=0.5, x and z at mass_point (0.5, 0.5)
        assert!((result.y - 0.5).abs() < 0.01,
            "y should be 0.5, got {}", result.y);
    }

    #[test]
    fn qef_corner_two_planes() {
        // Two perpendicular planes meeting at a corner
        let mut qef = QefData::new();

        // Plane at x=1: normal (1,0,0), intersection at (1,0,0)
        qef.add(Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        qef.add(Vec3::new(1.0, 1.0, 0.0), Vec3::new(1.0, 0.0, 0.0));

        // Plane at y=1: normal (0,1,0), intersection at (0,1,0)
        qef.add(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        qef.add(Vec3::new(1.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0));

        let result = qef.solve();
        // The vertex should be at approximately (1, 1, z)
        assert!((result.x - 1.0).abs() < 0.2,
            "x should be near 1.0, got {}", result.x);
        assert!((result.y - 1.0).abs() < 0.2,
            "y should be near 1.0, got {}", result.y);
    }

    #[test]
    fn qef_three_planes_corner() {
        // Three perpendicular planes meeting at (1, 1, 1)
        let mut qef = QefData::new();

        qef.add(Vec3::new(1.0, 0.5, 0.5), Vec3::new(1.0, 0.0, 0.0));
        qef.add(Vec3::new(0.5, 1.0, 0.5), Vec3::new(0.0, 1.0, 0.0));
        qef.add(Vec3::new(0.5, 0.5, 1.0), Vec3::new(0.0, 0.0, 1.0));

        let result = qef.solve();
        assert!((result.x - 1.0).abs() < 0.1,
            "x should be near 1.0, got {}", result.x);
        assert!((result.y - 1.0).abs() < 0.1,
            "y should be near 1.0, got {}", result.y);
        assert!((result.z - 1.0).abs() < 0.1,
            "z should be near 1.0, got {}", result.z);
    }

    #[test]
    fn qef_merge() {
        let mut qef1 = QefData::new();
        qef1.add(Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        qef1.add(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0));

        let mut qef2 = QefData::new();
        qef2.add(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 1.0));

        qef1.merge(&qef2);

        assert_eq!(qef1.count, 3);
        let expected_mass = Vec3::new(1.0, 1.0, 1.0);
        assert!((qef1.mass_point_sum - expected_mass).length() < 1e-5);
    }

    #[test]
    fn qef_solve_clamped() {
        let mut qef = QefData::new();
        // Create a QEF that would solve outside [0,1]^3
        qef.add(Vec3::new(2.0, 0.5, 0.5), Vec3::new(1.0, 0.0, 0.0));
        qef.add(Vec3::new(0.5, 2.0, 0.5), Vec3::new(0.0, 1.0, 0.0));
        qef.add(Vec3::new(0.5, 0.5, 2.0), Vec3::new(0.0, 0.0, 1.0));

        let clamped = qef.solve_clamped(Vec3::ZERO, Vec3::ONE);
        assert!(clamped.x <= 1.0 && clamped.y <= 1.0 && clamped.z <= 1.0);
        assert!(clamped.x >= 0.0 && clamped.y >= 0.0 && clamped.z >= 0.0);
    }

    #[test]
    fn qef_empty() {
        let qef = QefData::new();
        let result = qef.solve();
        assert_eq!(result, Vec3::ZERO);
    }

    #[test]
    fn jacobi_eigen_identity() {
        let (eigenvalues, _) = jacobi_eigen_3x3(Mat3::IDENTITY);
        assert!((eigenvalues.x - 1.0).abs() < 1e-5);
        assert!((eigenvalues.y - 1.0).abs() < 1e-5);
        assert!((eigenvalues.z - 1.0).abs() < 1e-5);
    }

    #[test]
    fn jacobi_eigen_diagonal() {
        let m = Mat3::from_cols(
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
            Vec3::new(0.0, 0.0, 5.0),
        );
        let (eigenvalues, _) = jacobi_eigen_3x3(m);
        let mut ev = [eigenvalues.x, eigenvalues.y, eigenvalues.z];
        ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((ev[0] - 2.0).abs() < 1e-4);
        assert!((ev[1] - 3.0).abs() < 1e-4);
        assert!((ev[2] - 5.0).abs() < 1e-4);
    }
}
