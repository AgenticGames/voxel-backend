use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Seed-deterministic permutation table for noise generation
pub struct PermutationTable {
    pub perm: [u8; 512],
}

impl PermutationTable {
    pub fn new(seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut values: Vec<u8> = (0..=255).collect();
        values.shuffle(&mut rng);

        let mut perm = [0u8; 512];
        perm[..256].copy_from_slice(&values[..256]);
        perm[256..512].copy_from_slice(&values[..256]);
        Self { perm }
    }

    #[inline]
    pub fn hash(&self, x: i32) -> u8 {
        self.perm[(x & 255) as usize]
    }

    #[inline]
    pub fn hash2(&self, x: i32, y: i32) -> u8 {
        self.perm[(self.perm[(x & 255) as usize] as i32 + y) as usize & 511]
    }

    #[inline]
    pub fn hash3(&self, x: i32, y: i32, z: i32) -> u8 {
        let h1 = self.perm[(x & 255) as usize] as usize;
        let h2 = self.perm[(h1 + (y & 255) as usize) & 511] as usize;
        self.perm[(h2 + (z & 255) as usize) & 511]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_same_seed() {
        let a = PermutationTable::new(42);
        let b = PermutationTable::new(42);
        assert_eq!(a.perm, b.perm);
    }

    #[test]
    fn different_seeds_differ() {
        let a = PermutationTable::new(42);
        let b = PermutationTable::new(99);
        assert_ne!(a.perm[..256], b.perm[..256]);
    }

    #[test]
    fn duplicated_halves() {
        let t = PermutationTable::new(12345);
        for i in 0..256 {
            assert_eq!(t.perm[i], t.perm[i + 256]);
        }
    }

    #[test]
    fn all_values_present() {
        let t = PermutationTable::new(7);
        let mut counts = [0u32; 256];
        for i in 0..256 {
            counts[t.perm[i] as usize] += 1;
        }
        for c in &counts {
            assert_eq!(*c, 1, "Each value 0..255 should appear exactly once");
        }
    }
}
