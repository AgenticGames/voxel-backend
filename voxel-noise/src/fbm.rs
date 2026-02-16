use crate::NoiseSource;

pub struct Fbm<N: NoiseSource> {
    pub source: N,
    pub octaves: u32,
    pub lacunarity: f64,
    pub persistence: f64,
}

impl<N: NoiseSource> Fbm<N> {
    pub fn new(source: N, octaves: u32, lacunarity: f64, persistence: f64) -> Self {
        Self {
            source,
            octaves,
            lacunarity,
            persistence,
        }
    }
}

impl<N: NoiseSource> NoiseSource for Fbm<N> {
    fn sample(&self, x: f64, y: f64, z: f64) -> f64 {
        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_amplitude = 0.0;

        for _ in 0..self.octaves {
            value += self.source.sample(
                x * frequency,
                y * frequency,
                z * frequency,
            ) * amplitude;
            max_amplitude += amplitude;
            amplitude *= self.persistence;
            frequency *= self.lacunarity;
        }

        value / max_amplitude
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simplex::Simplex3D;

    #[test]
    fn fbm_determinism() {
        let fbm = Fbm::new(Simplex3D::new(42), 6, 2.0, 0.5);
        let a = fbm.sample(1.0, 2.0, 3.0);
        let b = fbm.sample(1.0, 2.0, 3.0);
        assert_eq!(a, b);
    }

    #[test]
    fn fbm_range() {
        let fbm = Fbm::new(Simplex3D::new(42), 6, 2.0, 0.5);
        for i in 0..5000 {
            let x = (i as f64 * 0.17) - 400.0;
            let y = (i as f64 * 0.23) - 575.0;
            let z = (i as f64 * 0.31) - 775.0;
            let v = fbm.sample(x, y, z);
            assert!(v >= -1.0 && v <= 1.0, "FBM value {} out of range at ({}, {}, {})", v, x, y, z);
        }
    }

    #[test]
    fn octaves_affect_output() {
        let fbm1 = Fbm::new(Simplex3D::new(42), 1, 2.0, 0.5);
        let fbm6 = Fbm::new(Simplex3D::new(42), 6, 2.0, 0.5);
        // More octaves should produce different output from single octave
        let mut total_diff = 0.0;
        for i in 0..100 {
            let x = i as f64 * 0.5;
            let y = i as f64 * 0.3;
            let z = i as f64 * 0.7;
            total_diff += (fbm1.sample(x, y, z) - fbm6.sample(x, y, z)).abs();
        }
        assert!(total_diff > 1.0, "Different octave counts should produce meaningfully different results, total_diff={}", total_diff);
    }
}
