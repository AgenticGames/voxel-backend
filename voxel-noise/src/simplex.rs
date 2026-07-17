use crate::permutation::PermutationTable;
use crate::NoiseSource;

/// 3D gradient vectors (midpoints of edges of a cube)
const GRAD3: [[f64; 3]; 12] = [
    [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0], [0.0, -1.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, -1.0],
];

// Skewing/unskewing factors for 3D
const F3: f64 = 1.0 / 3.0;
const G3: f64 = 1.0 / 6.0;

pub struct Simplex3D {
    perm: PermutationTable,
}

impl Simplex3D {
    pub fn new(seed: u64) -> Self {
        Self {
            perm: PermutationTable::new(seed),
        }
    }

    #[inline]
    fn grad(&self, hash: u8, x: f64, y: f64, z: f64) -> f64 {
        let g = &GRAD3[(hash % 12) as usize];
        g[0] * x + g[1] * y + g[2] * z
    }
}

impl NoiseSource for Simplex3D {
    fn sample(&self, x: f64, y: f64, z: f64) -> f64 {
        // Skew the input space to determine which simplex cell we're in
        let s = (x + y + z) * F3;
        let i = (x + s).floor() as i32;
        let j = (y + s).floor() as i32;
        let k = (z + s).floor() as i32;

        let t = (i.wrapping_add(j).wrapping_add(k)) as f64 * G3;
        // Unskew the cell origin back to (x, y, z) space
        let x0 = x - (i as f64 - t);
        let y0 = y - (j as f64 - t);
        let z0 = z - (k as f64 - t);

        // Determine which simplex we are in (of the 6 tetrahedra that tile a cube)
        let (i1, j1, k1, i2, j2, k2) = if x0 >= y0 {
            if y0 >= z0 {
                (1, 0, 0, 1, 1, 0)
            } else if x0 >= z0 {
                (1, 0, 0, 1, 0, 1)
            } else {
                (0, 0, 1, 1, 0, 1)
            }
        } else {
            // x0 < y0
            if y0 < z0 {
                (0, 0, 1, 0, 1, 1)
            } else if x0 < z0 {
                (0, 1, 0, 0, 1, 1)
            } else {
                (0, 1, 0, 1, 1, 0)
            }
        };

        // Offsets for second corner in (x, y, z) coords
        let x1 = x0 - i1 as f64 + G3;
        let y1 = y0 - j1 as f64 + G3;
        let z1 = z0 - k1 as f64 + G3;
        // Offsets for third corner
        let x2 = x0 - i2 as f64 + 2.0 * G3;
        let y2 = y0 - j2 as f64 + 2.0 * G3;
        let z2 = z0 - k2 as f64 + 2.0 * G3;
        // Offsets for last corner
        let x3 = x0 - 1.0 + 3.0 * G3;
        let y3 = y0 - 1.0 + 3.0 * G3;
        let z3 = z0 - 1.0 + 3.0 * G3;

        // Hash coordinates of the four simplex corners
        let gi0 = self.perm.hash3(i, j, k);
        let gi1 = self.perm.hash3(i + i1, j + j1, k + k1);
        let gi2 = self.perm.hash3(i + i2, j + j2, k + k2);
        let gi3 = self.perm.hash3(i + 1, j + 1, k + 1);

        // Calculate contribution from four corners
        let mut n = 0.0;

        let t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
        if t0 >= 0.0 {
            let t0 = t0 * t0;
            n += t0 * t0 * self.grad(gi0, x0, y0, z0);
        }

        let t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
        if t1 >= 0.0 {
            let t1 = t1 * t1;
            n += t1 * t1 * self.grad(gi1, x1, y1, z1);
        }

        let t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
        if t2 >= 0.0 {
            let t2 = t2 * t2;
            n += t2 * t2 * self.grad(gi2, x2, y2, z2);
        }

        let t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
        if t3 >= 0.0 {
            let t3 = t3 * t3;
            n += t3 * t3 * self.grad(gi3, x3, y3, z3);
        }

        // Scale to [-1, 1]
        32.0 * n
    }
}

/// Sample THREE independently-seeded `Simplex3D` sources at the SAME point,
/// computing the seed-independent simplex geometry (skew, cell pick,
/// tetrahedron branch, corner offsets, falloff weights) ONCE and doing only
/// the per-corner hash + gradient dot per channel.
///
/// BIT-IDENTICAL to three separate `sample()` calls by construction: each
/// channel's contribution is accumulated with the same expression tree
/// (`n += (t²·t²) · grad`) in the same corner order, and every shared
/// quantity is computed with the exact operations `sample()` uses. The
/// worldgen hot loops call 3-seed domain warps per voxel (cavern warp, ore
/// warp) — this saves ~2/3 of the geometry work on those.
pub fn sample3_fused(
    a: &Simplex3D,
    b: &Simplex3D,
    c: &Simplex3D,
    x: f64,
    y: f64,
    z: f64,
) -> (f64, f64, f64) {
    let s = (x + y + z) * F3;
    let i = (x + s).floor() as i32;
    let j = (y + s).floor() as i32;
    let k = (z + s).floor() as i32;

    let t = (i.wrapping_add(j).wrapping_add(k)) as f64 * G3;
    let x0 = x - (i as f64 - t);
    let y0 = y - (j as f64 - t);
    let z0 = z - (k as f64 - t);

    let (i1, j1, k1, i2, j2, k2) = if x0 >= y0 {
        if y0 >= z0 {
            (1, 0, 0, 1, 1, 0)
        } else if x0 >= z0 {
            (1, 0, 0, 1, 0, 1)
        } else {
            (0, 0, 1, 1, 0, 1)
        }
    } else {
        if y0 < z0 {
            (0, 0, 1, 0, 1, 1)
        } else if x0 < z0 {
            (0, 1, 0, 0, 1, 1)
        } else {
            (0, 1, 0, 1, 1, 0)
        }
    };

    let x1 = x0 - i1 as f64 + G3;
    let y1 = y0 - j1 as f64 + G3;
    let z1 = z0 - k1 as f64 + G3;
    let x2 = x0 - i2 as f64 + 2.0 * G3;
    let y2 = y0 - j2 as f64 + 2.0 * G3;
    let z2 = z0 - k2 as f64 + 2.0 * G3;
    let x3 = x0 - 1.0 + 3.0 * G3;
    let y3 = y0 - 1.0 + 3.0 * G3;
    let z3 = z0 - 1.0 + 3.0 * G3;

    // Per-corner falloff weights, shared across channels. Keep the exact
    // `let t = t * t; t * t` shape so `w · grad` matches `sample()`'s
    // `(t*t) * (t*t) * grad` grouping bit-for-bit... it does not: sample()
    // computes `t0*t0` then `n += t0 * t0 * grad` which parses as
    // `((t0sq * t0sq) * grad)`. `w = t0sq * t0sq; n += w * grad` is the
    // same tree. Corners outside the falloff contribute exactly 0.0 (skip),
    // matching sample()'s `if t >= 0.0` guard.
    let corners = [
        (x0, y0, z0, i, j, k),
        (x1, y1, z1, i + i1, j + j1, k + k1),
        (x2, y2, z2, i + i2, j + j2, k + k2),
        (x3, y3, z3, i + 1, j + 1, k + 1),
    ];

    let mut na = 0.0;
    let mut nb = 0.0;
    let mut nc = 0.0;
    for &(cx, cy, cz, ci, cj, ck) in &corners {
        let t = 0.6 - cx * cx - cy * cy - cz * cz;
        if t >= 0.0 {
            let tsq = t * t;
            let w = tsq * tsq;
            na += w * a.grad(a.perm.hash3(ci, cj, ck), cx, cy, cz);
            nb += w * b.grad(b.perm.hash3(ci, cj, ck), cx, cy, cz);
            nc += w * c.grad(c.perm.hash3(ci, cj, ck), cx, cy, cz);
        }
    }

    (32.0 * na, 32.0 * nb, 32.0 * nc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn determinism() {
        let noise = Simplex3D::new(42);
        let a = noise.sample(1.5, 2.3, 0.7);
        let b = noise.sample(1.5, 2.3, 0.7);
        assert_eq!(a, b);
    }

    #[test]
    fn same_seed_same_output() {
        let n1 = Simplex3D::new(42);
        let n2 = Simplex3D::new(42);
        for i in 0..100 {
            let x = i as f64 * 0.37;
            let y = i as f64 * 0.53;
            let z = i as f64 * 0.71;
            assert_eq!(n1.sample(x, y, z), n2.sample(x, y, z));
        }
    }

    #[test]
    fn output_range() {
        let noise = Simplex3D::new(12345);
        let mut min = f64::MAX;
        let mut max = f64::MIN;
        for i in 0..10000 {
            let x = (i as f64 * 0.1) - 500.0;
            let y = (i as f64 * 0.073) - 365.0;
            let z = (i as f64 * 0.051) - 255.0;
            let v = noise.sample(x, y, z);
            min = min.min(v);
            max = max.max(v);
        }
        assert!(min >= -1.0, "min was {}", min);
        assert!(max <= 1.0, "max was {}", max);
        // Should use a reasonable portion of the range
        assert!(min < -0.3, "min should be below -0.3, was {}", min);
        assert!(max > 0.3, "max should be above 0.3, was {}", max);
    }

    #[test]
    fn mean_near_zero() {
        let noise = Simplex3D::new(999);
        let n = 10000;
        let mut sum = 0.0;
        for i in 0..n {
            let x = (i as f64 * 0.37) - 1850.0;
            let y = (i as f64 * 0.53) - 2650.0;
            let z = (i as f64 * 0.71) - 3550.0;
            sum += noise.sample(x, y, z);
        }
        let mean = sum / n as f64;
        assert!(
            mean.abs() < 0.1,
            "Mean should be near 0, was {}",
            mean
        );
    }

    /// The fused 3-channel sampler must be BIT-identical to three separate
    /// sample() calls — worldgen substitutes it into deterministic seeded
    /// generation, so even 1-ulp drift would change worlds.
    #[test]
    fn sample3_fused_bit_identical_to_three_samples() {
        let seeds = [(42u64, 43u64, 44u64), (7, 1000003, 999), (0, 1, u64::MAX)];
        for &(sa, sb, sc) in &seeds {
            let a = Simplex3D::new(sa);
            let b = Simplex3D::new(sb);
            let c = Simplex3D::new(sc);
            let mut checked = 0u32;
            for i in 0..30000 {
                // Mix of coordinate scales incl. negatives and the halved
                // warp-frequency domain the hot path actually uses.
                let f = i as f64;
                let x = (f * 0.371 - 5000.0) * 0.013;
                let y = (f * 0.533 - 3000.0) * 0.017;
                let z = (f * 0.719 - 7000.0) * 0.011;
                let (fa, fb, fc) = sample3_fused(&a, &b, &c, x, y, z);
                assert_eq!(fa.to_bits(), a.sample(x, y, z).to_bits(), "ch A at {x},{y},{z}");
                assert_eq!(fb.to_bits(), b.sample(x, y, z).to_bits(), "ch B at {x},{y},{z}");
                assert_eq!(fc.to_bits(), c.sample(x, y, z).to_bits(), "ch C at {x},{y},{z}");
                checked += 3;
            }
            assert_eq!(checked, 90000);
        }
    }

    #[test]
    fn different_seeds_differ() {
        let n1 = Simplex3D::new(1);
        let n2 = Simplex3D::new(2);
        let mut diff_count = 0;
        for i in 0..100 {
            let x = i as f64 * 0.5;
            if (n1.sample(x, 0.0, 0.0) - n2.sample(x, 0.0, 0.0)).abs() > 1e-10 {
                diff_count += 1;
            }
        }
        assert!(diff_count > 50, "Most samples should differ between seeds");
    }
}
