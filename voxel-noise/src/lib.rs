pub mod permutation;
pub mod simplex;
pub mod fbm;
pub mod ridged;
pub mod domain_warp;
pub mod utils;

/// Trait for all noise sources
pub trait NoiseSource: Send + Sync {
    fn sample(&self, x: f64, y: f64, z: f64) -> f64;
}
