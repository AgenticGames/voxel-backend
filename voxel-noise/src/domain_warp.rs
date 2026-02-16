use crate::NoiseSource;

pub struct DomainWarp<N: NoiseSource, W: NoiseSource> {
    pub source: N,
    pub warp_x: W,
    pub warp_y: W,
    pub warp_z: W,
    pub amplitude: f64,
}

impl<N: NoiseSource, W: NoiseSource> DomainWarp<N, W> {
    pub fn new(source: N, warp_x: W, warp_y: W, warp_z: W, amplitude: f64) -> Self {
        Self {
            source,
            warp_x,
            warp_y,
            warp_z,
            amplitude,
        }
    }
}

impl<N: NoiseSource, W: NoiseSource> NoiseSource for DomainWarp<N, W> {
    fn sample(&self, x: f64, y: f64, z: f64) -> f64 {
        let wx = x + self.warp_x.sample(x, y, z) * self.amplitude;
        let wy = y + self.warp_y.sample(x, y, z) * self.amplitude;
        let wz = z + self.warp_z.sample(x, y, z) * self.amplitude;
        self.source.sample(wx, wy, wz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simplex::Simplex3D;

    #[test]
    fn domain_warp_determinism() {
        let dw = DomainWarp::new(
            Simplex3D::new(1),
            Simplex3D::new(2),
            Simplex3D::new(3),
            Simplex3D::new(4),
            0.5,
        );
        let a = dw.sample(1.0, 2.0, 3.0);
        let b = dw.sample(1.0, 2.0, 3.0);
        assert_eq!(a, b);
    }

    #[test]
    fn zero_amplitude_passthrough() {
        let source = Simplex3D::new(42);
        let dw = DomainWarp::new(
            Simplex3D::new(42),
            Simplex3D::new(100),
            Simplex3D::new(200),
            Simplex3D::new(300),
            0.0,
        );
        for i in 0..100 {
            let x = i as f64 * 0.3;
            let y = i as f64 * 0.5;
            let z = i as f64 * 0.7;
            assert_eq!(
                source.sample(x, y, z),
                dw.sample(x, y, z),
                "Zero amplitude warp should pass through"
            );
        }
    }

    #[test]
    fn nonzero_amplitude_changes_output() {
        let base = Simplex3D::new(42);
        let dw = DomainWarp::new(
            Simplex3D::new(42),
            Simplex3D::new(100),
            Simplex3D::new(200),
            Simplex3D::new(300),
            2.0,
        );
        let mut diff_count = 0;
        for i in 0..100 {
            let x = i as f64 * 0.5;
            if (base.sample(x, 0.0, 0.0) - dw.sample(x, 0.0, 0.0)).abs() > 1e-10 {
                diff_count += 1;
            }
        }
        assert!(diff_count > 50, "Warp should change most outputs");
    }
}
