/// Fluid type: Water or Lava.
/// Values match UE rendering expectations: 1=Water, 2=Lava.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FluidType {
    Water = 1,
    Lava = 2,
}

/// A single fluid cell within a chunk.
#[derive(Debug, Clone, Copy)]
pub struct FluidCell {
    pub level: f32,
    pub fluid_type: FluidType,
}

impl Default for FluidCell {
    fn default() -> Self {
        Self {
            level: 0.0,
            fluid_type: FluidType::Water,
        }
    }
}

impl FluidCell {
    pub fn is_empty(&self) -> bool {
        self.level < MIN_LEVEL
    }

    pub fn is_source(&self) -> bool {
        self.level >= SOURCE_LEVEL
    }
}

/// Minimum fluid level to consider non-empty.
pub const MIN_LEVEL: f32 = 0.001;
/// Level at which a cell is considered a full source block.
pub const SOURCE_LEVEL: f32 = 1.0;
/// Maximum fluid level.
pub const MAX_LEVEL: f32 = 1.0;

/// Per-chunk fluid grid: 16^3 cells + 64-bit solid mask.
///
/// The solid_mask uses a packed bitfield: one bit per voxel in the 16^3 grid.
/// 16*16*16 = 4096 bits = 64 u64 values.
pub struct ChunkFluidGrid {
    pub cells: Vec<FluidCell>,
    pub solid_mask: Vec<u64>,
    pub size: usize,
    pub dirty: bool,
}

impl ChunkFluidGrid {
    pub fn new(size: usize) -> Self {
        let total = size * size * size;
        let mask_words = (total + 63) / 64;
        Self {
            cells: vec![FluidCell::default(); total],
            solid_mask: vec![0u64; mask_words],
            size,
            dirty: false,
        }
    }

    #[inline]
    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        z * self.size * self.size + y * self.size + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> &FluidCell {
        &self.cells[self.index(x, y, z)]
    }

    #[inline]
    pub fn get_mut(&mut self, x: usize, y: usize, z: usize) -> &mut FluidCell {
        let idx = self.index(x, y, z);
        &mut self.cells[idx]
    }

    #[inline]
    pub fn is_solid(&self, x: usize, y: usize, z: usize) -> bool {
        let idx = self.index(x, y, z);
        let word = idx / 64;
        let bit = idx % 64;
        (self.solid_mask[word] >> bit) & 1 == 1
    }

    #[inline]
    pub fn set_solid(&mut self, x: usize, y: usize, z: usize, solid: bool) {
        let idx = self.index(x, y, z);
        let word = idx / 64;
        let bit = idx % 64;
        if solid {
            self.solid_mask[word] |= 1u64 << bit;
        } else {
            self.solid_mask[word] &= !(1u64 << bit);
        }
    }

    /// Update solid mask from a flat bitfield (provided by voxel-ffi after density gen).
    pub fn update_solid_mask(&mut self, mask: &[u64]) {
        let len = self.solid_mask.len().min(mask.len());
        self.solid_mask[..len].copy_from_slice(&mask[..len]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_indexing() {
        let grid = ChunkFluidGrid::new(16);
        assert_eq!(grid.index(0, 0, 0), 0);
        assert_eq!(grid.index(15, 0, 0), 15);
        assert_eq!(grid.index(0, 1, 0), 16);
        assert_eq!(grid.index(0, 0, 1), 256);
    }

    #[test]
    fn solid_mask_set_get() {
        let mut grid = ChunkFluidGrid::new(16);
        assert!(!grid.is_solid(5, 5, 5));
        grid.set_solid(5, 5, 5, true);
        assert!(grid.is_solid(5, 5, 5));
        grid.set_solid(5, 5, 5, false);
        assert!(!grid.is_solid(5, 5, 5));
    }

    #[test]
    fn cell_default_empty() {
        let cell = FluidCell::default();
        assert!(cell.is_empty());
        assert!(!cell.is_source());
    }
}
