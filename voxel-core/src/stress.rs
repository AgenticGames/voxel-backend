/// Per-voxel stress values for a chunk. Same layout as DensityField (17^3 for chunk_size=16).
#[derive(Debug, Clone)]
pub struct StressField {
    pub stress: Vec<f32>,
    pub size: usize,
}

impl StressField {
    pub fn new(size: usize) -> Self {
        Self {
            stress: vec![0.0; size * size * size],
            size,
        }
    }

    #[inline]
    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        z * self.size * self.size + y * self.size + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> f32 {
        self.stress[self.index(x, y, z)]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, val: f32) {
        let idx = self.index(x, y, z);
        self.stress[idx] = val;
    }
}

/// Support type enum (NOT a Material variant).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportType {
    None = 0,
    WoodBeam = 1,
    MetalBeam = 2,
    Reinforcement = 3,
}

impl SupportType {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => SupportType::WoodBeam,
            2 => SupportType::MetalBeam,
            3 => SupportType::Reinforcement,
            _ => SupportType::None,
        }
    }
}

/// Per-voxel support data for a chunk.
#[derive(Debug, Clone)]
pub struct SupportField {
    pub supports: Vec<SupportType>,
    pub size: usize,
}

impl SupportField {
    pub fn new(size: usize) -> Self {
        Self {
            supports: vec![SupportType::None; size * size * size],
            size,
        }
    }

    #[inline]
    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        z * self.size * self.size + y * self.size + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> SupportType {
        self.supports[self.index(x, y, z)]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, support_type: SupportType) {
        let idx = self.index(x, y, z);
        self.supports[idx] = support_type;
    }

    #[inline]
    pub fn has_support(&self, x: usize, y: usize, z: usize) -> bool {
        self.get(x, y, z) != SupportType::None
    }
}

/// Default hardness per Material (index by Material as u8).
/// Air = 0.0 (no resistance). Higher = harder to collapse.
pub const DEFAULT_MATERIAL_HARDNESS: [f32; 19] = [
    0.0,   // Air
    0.45,  // Sandstone (soft)
    0.55,  // Limestone
    0.80,  // Granite (hard)
    0.75,  // Basalt
    0.60,  // Slate
    0.65,  // Marble
    0.50,  // Iron
    0.45,  // Copper
    0.40,  // Malachite
    0.40,  // Tin
    0.55,  // Gold
    0.90,  // Diamond
    0.70,  // Kimberlite
    0.50,  // Sulfide
    0.65,  // Quartz
    0.55,  // Pyrite
    0.60,  // Amethyst
    0.70,  // Crystal
];

/// Support hardness values (how much stress each support type absorbs).
pub const SUPPORT_HARDNESS: [f32; 4] = [
    0.0,   // None
    0.95,  // WoodBeam
    1.50,  // MetalBeam
    1.20,  // Reinforcement
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stress_field_basic() {
        let mut sf = StressField::new(17);
        assert_eq!(sf.stress.len(), 17 * 17 * 17);
        assert_eq!(sf.get(0, 0, 0), 0.0);
        sf.set(5, 5, 5, 0.75);
        assert!((sf.get(5, 5, 5) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn support_field_basic() {
        let mut sf = SupportField::new(17);
        assert_eq!(sf.supports.len(), 17 * 17 * 17);
        assert!(!sf.has_support(0, 0, 0));
        sf.set(3, 3, 3, SupportType::WoodBeam);
        assert!(sf.has_support(3, 3, 3));
        assert_eq!(sf.get(3, 3, 3), SupportType::WoodBeam);
    }

    #[test]
    fn support_type_from_u8() {
        assert_eq!(SupportType::from_u8(0), SupportType::None);
        assert_eq!(SupportType::from_u8(1), SupportType::WoodBeam);
        assert_eq!(SupportType::from_u8(2), SupportType::MetalBeam);
        assert_eq!(SupportType::from_u8(3), SupportType::Reinforcement);
        assert_eq!(SupportType::from_u8(255), SupportType::None);
    }

    #[test]
    fn hardness_tables_correct_length() {
        assert_eq!(DEFAULT_MATERIAL_HARDNESS.len(), 19);
        assert_eq!(SUPPORT_HARDNESS.len(), 4);
    }
}
