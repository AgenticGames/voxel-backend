/// Linear interpolation
#[inline]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// Smoothstep (Hermite interpolation)
#[inline]
pub fn smoothstep(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

/// Quintic interpolation (Ken Perlin's improved curve)
#[inline]
pub fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}
