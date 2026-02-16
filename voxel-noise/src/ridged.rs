use crate::NoiseSource;

pub struct RidgedMulti<N: NoiseSource> {
    pub source: N,
    pub octaves: u32,
    pub lacunarity: f64,
    pub gain: f64,
    spectral_weights: Vec<f64>,
}

impl<N: NoiseSource> RidgedMulti<N> {
    pub fn new(source: N, octaves: u32, lacunarity: f64, gain: f64) -> Self {
        // Precompute spectral weights based on frequency
        let mut spectral_weights = Vec::with_capacity(octaves as usize);
        let mut freq: f64 = 1.0;
        for _ in 0..octaves {
            spectral_weights.push(freq.powf(-1.0));
            freq *= lacunarity;
        }
        Self {
            source,
            octaves,
            lacunarity,
            gain,
            spectral_weights,
        }
    }
}

impl<N: NoiseSource> NoiseSource for RidgedMulti<N> {
    fn sample(&self, x: f64, y: f64, z: f64) -> f64 {
        let mut value = 0.0;
        let mut weight = 1.0;
        let mut frequency = 1.0;

        for i in 0..self.octaves as usize {
            let signal = self.source.sample(
                x * frequency,
                y * frequency,
                z * frequency,
            );
            // Ridge: invert absolute value
            let ridge = 1.0 - signal.abs();
            // Square for sharper ridges
            let ridge = ridge * ridge;
            // Weight by previous octave
            let ridge = ridge * weight;
            // Update weight for next octave
            weight = (ridge * self.gain).clamp(0.0, 1.0);

            value += ridge * self.spectral_weights[i];
            frequency *= self.lacunarity;
        }

        // Normalize to roughly [-1, 1]
        // The sum of spectral_weights gives the max possible value
        let max_val: f64 = self.spectral_weights.iter().sum();
        (value / max_val) * 2.0 - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simplex::Simplex3D;

    #[test]
    fn ridged_determinism() {
        let ridged = RidgedMulti::new(Simplex3D::new(42), 6, 2.0, 2.0);
        let a = ridged.sample(1.0, 2.0, 3.0);
        let b = ridged.sample(1.0, 2.0, 3.0);
        assert_eq!(a, b);
    }

    #[test]
    fn ridged_range() {
        let ridged = RidgedMulti::new(Simplex3D::new(42), 6, 2.0, 2.0);
        for i in 0..5000 {
            let x = (i as f64 * 0.17) - 400.0;
            let y = (i as f64 * 0.23) - 575.0;
            let z = (i as f64 * 0.31) - 775.0;
            let v = ridged.sample(x, y, z);
            assert!(v >= -1.1 && v <= 1.1,
                "Ridged value {} out of range at ({}, {}, {})", v, x, y, z);
        }
    }

    #[test]
    fn ridged_different_from_fbm() {
        use crate::fbm::Fbm;
        let ridged = RidgedMulti::new(Simplex3D::new(42), 4, 2.0, 2.0);
        let fbm = Fbm::new(Simplex3D::new(42), 4, 2.0, 0.5);
        let mut diff = 0.0;
        for i in 0..100 {
            let x = i as f64 * 0.5;
            diff += (ridged.sample(x, 0.0, 0.0) - fbm.sample(x, 0.0, 0.0)).abs();
        }
        assert!(diff > 1.0, "Ridged and FBM should produce different outputs");
    }
}
