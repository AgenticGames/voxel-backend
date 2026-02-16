/// Smooth blend between worm carving and cavern noise
#[inline]
pub fn smooth_blend(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    a * (1.0 - h) + b * h - k * h * (1.0 - h)
}
