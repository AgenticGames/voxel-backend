/// Fluid type: Water (6 debug-colored subtypes) or Lava.
/// Values match UE rendering expectations: 1=Water, 2=Lava, 3-9=water subtypes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FluidType {
    Water = 1,
    Lava = 2,
    WaterSpringLine = 3,
    WaterDrip = 4,
    WaterBreach = 5,
    WaterRiver = 6,
    WaterArtesian = 7,
    WaterHydrothermal = 8,
    WaterPool = 9,
}

impl FluidType {
    /// Returns true for all water-family types (Water + 6 subtypes).
    #[inline]
    pub fn is_water(self) -> bool {
        self != FluidType::Lava
    }

    /// Returns true only for Lava.
    #[inline]
    pub fn is_lava(self) -> bool {
        self == FluidType::Lava
    }

    /// Convert from raw u8, defaulting unknown values to Water.
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => FluidType::Water,
            2 => FluidType::Lava,
            3 => FluidType::WaterSpringLine,
            4 => FluidType::WaterDrip,
            5 => FluidType::WaterBreach,
            6 => FluidType::WaterRiver,
            7 => FluidType::WaterArtesian,
            8 => FluidType::WaterHydrothermal,
            9 => FluidType::WaterPool,
            _ => FluidType::Water,
        }
    }
}

/// A single fluid cell within a chunk.
#[derive(Debug, Clone, Copy)]
pub struct FluidCell {
    pub level: f32,
    pub fluid_type: FluidType,
    /// True for infinite sources (geological springs, lava vents).
    /// Non-source cells (e.g. cauldron water) drain when flowing.
    pub is_source: bool,
    /// Grace period ticks remaining. While > 0, cell behaves like a source
    /// (level is not deducted during flow). Decremented each tick.
    pub grace_ticks: u16,
    /// Ticks of stagnation for orphan puddle detection. Incremented when
    /// level < ORPHAN_THRESHOLD and no flow occurred. Reset on movement.
    pub stagnant_ticks: u8,
    /// Source self-extinguish (2026-08-04): consecutive ticks this SOURCE
    /// cell ended below SOURCE_DRAIN_LEVEL despite regen refilling it to
    /// full at tick start — meaning its fluid is vanishing into a sink
    /// (void pinhole, out-of-world, thin-film loss) and it can never reach
    /// steady state. At SOURCE_DRAIN_DEMOTE_TICKS the source demotes to a
    /// one-shot fill (is_source=false) and the pump dies. Sealed pools and
    /// spring-fed rivers equalize (post-tick level stays high), so they
    /// never accumulate a streak. Only meaningful while is_source.
    pub drain_ticks: u8,
    /// **Bounded-flow tracking** — hops from the originating source cell.
    /// Source cells reset to 0 each tick; flow propagation increments by 1
    /// on each transfer. Cells whose hop count meets/exceeds the source's
    /// `max_flow_dist` will not propagate flow further (Minecraft-style
    /// hard length limit). 255 = "no source recorded" sentinel.
    pub hops_from_source: u8,
    /// Maximum hops this *source* permits its children to spread.
    /// `0` = unlimited (legacy / geological default — keeps existing behavior).
    /// `>0` = bounded: flow stops once a child cell's `hops_from_source` reaches this value.
    /// Only meaningful on cells where `is_source = true`; children inherit the
    /// effective limit from the source they propagated from (carried via the
    /// source-side check at transfer time).
    pub max_flow_dist: u8,
}

impl Default for FluidCell {
    fn default() -> Self {
        Self {
            level: 0.0,
            fluid_type: FluidType::Water,
            is_source: false,
            grace_ticks: 0,
            stagnant_ticks: 0,
            drain_ticks: 0,
            hops_from_source: 255, // sentinel: no source recorded
            max_flow_dist: 0,      // 0 = unlimited (legacy)
        }
    }
}

impl FluidCell {
    pub fn is_empty(&self) -> bool {
        self.level < MIN_LEVEL
    }

    pub fn is_source(&self) -> bool {
        self.is_source
    }
}

/// Minimum fluid level to consider non-empty.
pub const MIN_LEVEL: f32 = 0.001;
/// Orphan puddle threshold: cells below this level get boosted slope flow.
pub const ORPHAN_THRESHOLD: f32 = 0.15;
/// Ticks of stagnation before orphan puddles start evaporating.
pub const ORPHAN_EVAP_TICKS: u8 = 35;
/// Source self-extinguish: if none of a source's passable neighbors holds
/// at least this level, nothing it emits is accumulating (it's all
/// vanishing into a sink).
pub const SOURCE_DRAIN_LEVEL: f32 = 0.5;
/// Consecutive draining ticks before a source demotes to a one-shot fill.
/// ~40 lava ticks ≈ a few seconds at default rates — matches the user's
/// "infinite pool turns off after ~4 seconds" design instinct.
pub const SOURCE_DRAIN_DEMOTE_TICKS: u8 = 40;
/// Level at which a cell is considered a full source block.
pub const SOURCE_LEVEL: f32 = 1.0;
/// Maximum fluid level.
pub const MAX_LEVEL: f32 = 1.0;

/// Corner offsets for the 8 corners of a cell, matching MC convention:
///   0=(0,0,0) 1=(1,0,0) 2=(1,1,0) 3=(0,1,0)
///   4=(0,0,1) 5=(1,0,1) 6=(1,1,1) 7=(0,1,1)
const CELL_CORNER_OFFSETS: [[usize; 3]; 8] = [
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
];

/// Convert 8 corner densities into a [0.0, 1.0] fluid capacity for the cell.
///
/// Counts air corners (density <= 0). cap = air_corners / 8.
/// Returns 1.0 for fully-air cells, 0.0 for fully-solid, and a fraction in
/// between for boundary cells. This makes the simulator's notion of free
/// space match the marching-cubes mesh surface — without it, a cell with
/// 5 of 8 corners inside rock could still hold full water and the fluid
/// would visibly clip through the wall (the mesher carves the geometry,
/// but the simulator kept pouring more fluid in).
#[inline]
pub fn capacity_from_corners(corners: &[f32]) -> f32 {
    debug_assert_eq!(corners.len(), 8);
    let mut air = 0u8;
    for &d in corners {
        if d <= 0.0 { air += 1; }
    }
    air as f32 * 0.125 // /8
}

/// Lightweight density-only cache for chunks that have no fluid yet.
/// Avoids allocating 4096 FluidCells until fluid actually enters the chunk.
pub struct ChunkDensityCache {
    pub cell_density: Vec<f32>,   // 4096
    pub cell_corners: Vec<f32>,   // 32768
    pub size: usize,
}

impl ChunkDensityCache {
    pub fn new(size: usize) -> Self {
        let total = size * size * size;
        Self {
            cell_density: vec![-1.0; total],
            cell_corners: vec![-1.0; total * 8],
            size,
        }
    }

    /// Update density data from a raw 17^3 DensityField (same logic as ChunkFluidGrid::update_density).
    pub fn update_density(&mut self, densities: &[f32]) {
        let size = self.size;
        let stride = size + 1;
        if densities.len() < stride * stride * stride {
            return;
        }
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let cell_idx = z * size * size + y * size + x;
                    let mut sum = 0.0f32;
                    for (c, offsets) in CELL_CORNER_OFFSETS.iter().enumerate() {
                        let gx = x + offsets[0];
                        let gy = y + offsets[1];
                        let gz = z + offsets[2];
                        let grid_idx = gz * stride * stride + gy * stride + gx;
                        let d = densities[grid_idx];
                        self.cell_corners[cell_idx * 8 + c] = d;
                        sum += d;
                    }
                    self.cell_density[cell_idx] = sum / 8.0;
                }
            }
        }
    }
}


/// Per-chunk fluid grid: 16^3 cells with continuous density data.
///
/// Replaces the old binary solid_mask with density values from the terrain's
/// DensityField. Uses binary cell classification: center density > 0 = solid
/// (cap 0.0), center density <= 0 = air (cap 1.0). No fractional capacity.
pub struct ChunkFluidGrid {
    pub cells: Vec<FluidCell>,
    /// Density at each cell center (average of 8 corners). 16^3 = 4096 values.
    /// Positive = solid, negative = air.
    pub cell_density: Vec<f32>,
    /// 8 corner densities per cell for meshing shoreline clipping.
    /// Layout: cell_corners[cell_idx * 8 + corner] where corner is MC ordering.
    /// 16^3 * 8 = 32768 values.
    pub cell_corners: Vec<f32>,
    /// Precomputed: true if ALL 8 corner densities > 0 (fully solid cell).
    pub cell_solid: Vec<bool>,
    /// Precomputed fractional capacity per cell (0.0 = fully solid, 1.0 = fully open).
    /// Computed as `air_corner_count / 8` so the sim respects partial-volume
    /// rock just like the mesher does — fluid no longer clips through walls.
    pub cell_cap: Vec<f32>,
    pub size: usize,
    pub dirty: bool,
    /// True if any cell has level > MIN_LEVEL. Used to skip empty chunks in sim.
    pub has_fluid: bool,
    /// True if any cell has fluid AND is lava. Used to skip whole chunks in
    /// the per-tick lava↔water quench scan, which is otherwise an N³ cost
    /// paid even for water-only worlds.
    pub has_lava: bool,
    /// True if any cell is a source (`is_source = true` with level above
    /// MIN_LEVEL). Used to skip whole chunks in the per-tick `regen_sources`
    /// pass — otherwise it's an N³ walk per chunk per water tick paying for
    /// the common case where most chunks have no sources at all.
    pub has_sources: bool,
    /// Reusable scratch buffer for `tick_chunk`'s double-buffered cell write.
    /// Owning this on the grid lets the simulator `mem::take` it instead of
    /// `cells.clone()`-ing every substep — eliminates the dominant per-tick
    /// allocation (≈540KB at chunk_size=30, ×6 substeps × N chunks/sec).
    pub scratch_cells: Vec<FluidCell>,
    /// Reusable per-tick column-weight buffer for upward-pressure equalization.
    pub scratch_weights: Vec<f32>,
    /// Reusable per-tick drain-delta buffer for entrainment.
    pub scratch_drain: Vec<f32>,
}

impl ChunkFluidGrid {
    pub fn new(size: usize) -> Self {
        let total = size * size * size;
        Self {
            cells: vec![FluidCell::default(); total],
            cell_density: vec![-1.0; total], // default to air (negative density)
            cell_corners: vec![-1.0; total * 8],
            cell_solid: vec![false; total],
            cell_cap: vec![1.0; total],
            size,
            dirty: false,
            has_fluid: false,
            has_lava: false,
            has_sources: false,
            scratch_cells: Vec::new(),
            scratch_weights: Vec::new(),
            scratch_drain: Vec::new(),
        }
    }

    /// Create a grid from a density cache, promoting it when fluid first enters.
    pub fn from_density_cache(cache: &ChunkDensityCache) -> Self {
        let size = cache.size;
        let total = size * size * size;
        let cell_solid: Vec<bool> = (0..total)
            .map(|idx| (0..8).all(|c| cache.cell_corners[idx * 8 + c] > 0.0))
            .collect();
        // Fractional capacity from corner counts — matches the MC mesh surface
        // so fluid can't pour into mostly-solid cells and visibly clip walls.
        let cell_cap: Vec<f32> = (0..total)
            .map(|idx| capacity_from_corners(&cache.cell_corners[idx * 8 .. idx * 8 + 8]))
            .collect();
        Self {
            cells: vec![FluidCell::default(); total],
            cell_density: cache.cell_density.clone(),
            cell_corners: cache.cell_corners.clone(),
            cell_solid,
            cell_cap,
            size,
            dirty: false,
            has_fluid: false,
            has_lava: false,
            has_sources: false,
            scratch_cells: Vec::new(),
            scratch_weights: Vec::new(),
            scratch_drain: Vec::new(),
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

    /// Returns true if the cell is fully solid (ALL 8 corner densities > 0).
    #[inline]
    pub fn is_solid(&self, x: usize, y: usize, z: usize) -> bool {
        self.cell_solid[self.index(x, y, z)]
    }

    /// Returns the fluid capacity of a cell: how much fluid it can hold.
    /// Uses precomputed fractional capacity from cell_cap.
    #[inline]
    pub fn cell_capacity(&self, x: usize, y: usize, z: usize) -> f32 {
        self.cell_cap[self.index(x, y, z)]
    }

    /// Returns true if the cell is mostly solid (threshold+ of 8 corners have positive density).
    /// Used as a safety guard to prevent placing fluid in boundary cells that straddle
    /// the solid/air interface (e.g. cauldron walls/floors).
    #[inline]
    pub fn is_mostly_solid(&self, x: usize, y: usize, z: usize, threshold: u8) -> bool {
        let idx = self.index(x, y, z);
        let base = idx * 8;
        let count = (0..8).filter(|&c| self.cell_corners[base + c] > 0.0).count();
        count >= threshold as usize
    }

    /// Recompute cell capacity from corner densities (fractional: air_corners/8).
    /// Replaces the previous binary classification; cells straddling the
    /// solid/air boundary now hold partial capacity matching the mesh surface.
    pub fn recompute_capacity(&mut self) {
        let total = self.size * self.size * self.size;
        for idx in 0..total {
            self.cell_cap[idx] =
                capacity_from_corners(&self.cell_corners[idx * 8 .. idx * 8 + 8]);
        }
    }

    /// Set density for a single cell (used in tests and terrain modification).
    /// Positive = solid, negative = air.
    #[inline]
    pub fn set_density(&mut self, x: usize, y: usize, z: usize, density: f32) {
        let idx = self.index(x, y, z);
        self.cell_density[idx] = density;
        // Also set all 8 corners to the same value for consistency in tests
        for c in 0..8 {
            self.cell_corners[idx * 8 + c] = density;
        }
        self.cell_solid[idx] = density > 0.0; // all corners set to same value
        // All 8 corners share the same sign here, so fractional cap collapses
        // to the binary 0/1 it always was for set_density callers.
        self.cell_cap[idx] = if density > 0.0 { 0.0 } else { 1.0 };
    }

    /// Get the 8 corner densities for a cell (for meshing/shoreline clipping).
    #[inline]
    pub fn get_corners(&self, x: usize, y: usize, z: usize) -> [f32; 8] {
        let idx = self.index(x, y, z);
        let base = idx * 8;
        let mut corners = [0.0f32; 8];
        corners.copy_from_slice(&self.cell_corners[base..base + 8]);
        corners
    }

    /// Update density data from a raw 17^3 DensityField.
    ///
    /// Extracts center densities (average of 8 corners) and per-cell corner
    /// densities from the full (chunk_size+1)^3 density grid.
    pub fn update_density(&mut self, densities: &[f32]) {
        let size = self.size;
        let stride = size + 1; // 17 for chunk_size=16

        // Validate input size
        if densities.len() < stride * stride * stride {
            return;
        }

        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let cell_idx = z * size * size + y * size + x;

                    // Extract 8 corner densities from the 17^3 grid
                    let mut sum = 0.0f32;
                    for (c, offsets) in CELL_CORNER_OFFSETS.iter().enumerate() {
                        let gx = x + offsets[0];
                        let gy = y + offsets[1];
                        let gz = z + offsets[2];
                        let grid_idx = gz * stride * stride + gy * stride + gx;
                        let d = densities[grid_idx];
                        self.cell_corners[cell_idx * 8 + c] = d;
                        sum += d;
                    }

                    // Center density = average of 8 corners
                    self.cell_density[cell_idx] = sum / 8.0;
                    // Fully solid only if ALL 8 corners are positive
                    self.cell_solid[cell_idx] = (0..8).all(|c| self.cell_corners[cell_idx * 8 + c] > 0.0);
                    // Fractional capacity from corner counts (matches MC mesh surface).
                    self.cell_cap[cell_idx] =
                        capacity_from_corners(&self.cell_corners[cell_idx * 8 .. cell_idx * 8 + 8]);
                }
            }
        }
    }

    /// Raw terrain density at grid point (gx, gy, gz) in the 17^3 density grid.
    /// Used by the mesher to align fluid boundaries with DC mesh surfaces.
    #[inline]
    pub fn grid_point_density(&self, gx: usize, gy: usize, gz: usize) -> f32 {
        let size = self.size;
        let cx = gx.min(size - 1);
        let cy = gy.min(size - 1);
        let cz = gz.min(size - 1);
        // CELL_CORNER_OFFSETS: 0=(0,0,0) 1=(1,0,0) 2=(1,1,0) 3=(0,1,0)
        //                      4=(0,0,1) 5=(1,0,1) 6=(1,1,1) 7=(0,1,1)
        let corner = match (gx - cx, gy - cy, gz - cz) {
            (0, 0, 0) => 0, (1, 0, 0) => 1, (1, 1, 0) => 2, (0, 1, 0) => 3,
            (0, 0, 1) => 4, (1, 0, 1) => 5, (1, 1, 1) => 6, (0, 1, 1) => 7,
            _ => 0,
        };
        self.cell_corners[self.index(cx, cy, cz) * 8 + corner]
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
    fn density_solid_check() {
        let mut grid = ChunkFluidGrid::new(16);
        // Default is air (density -1.0)
        assert!(!grid.is_solid(5, 5, 5));
        assert!((grid.cell_capacity(5, 5, 5) - 1.0).abs() < 0.01);

        // Set to solid
        grid.set_density(5, 5, 5, 1.0);
        assert!(grid.is_solid(5, 5, 5));
        assert!(grid.cell_capacity(5, 5, 5) < 0.01);

        // Set back to air (any negative density → capacity 1.0)
        grid.set_density(5, 5, 5, -0.5);
        assert!(!grid.is_solid(5, 5, 5));
        assert!((grid.cell_capacity(5, 5, 5) - 1.0).abs() < 0.01);
    }

    #[test]
    fn cell_capacity_clamped() {
        let mut grid = ChunkFluidGrid::new(16);
        // Very negative density → capacity capped at 1.0
        grid.set_density(0, 0, 0, -10.0);
        assert!((grid.cell_capacity(0, 0, 0) - 1.0).abs() < 0.001);

        // Negative density → binary capacity 1.0
        grid.set_density(0, 0, 0, -0.3);
        assert!((grid.cell_capacity(0, 0, 0) - 1.0).abs() < 0.001);

        // Solid → zero capacity
        grid.set_density(0, 0, 0, 0.5);
        assert!(grid.cell_capacity(0, 0, 0) < 0.001);
    }

    #[test]
    fn update_density_from_grid() {
        let size = 4; // small for testing
        let stride = size + 1;
        let mut grid = ChunkFluidGrid::new(size);

        // Create a 5^3 density field: all air (-1.0)
        let mut densities = vec![-1.0f32; stride * stride * stride];

        // Make cell (1,1,1) solid by setting all its corners to positive
        for offsets in &CELL_CORNER_OFFSETS {
            let gx = 1 + offsets[0];
            let gy = 1 + offsets[1];
            let gz = 1 + offsets[2];
            densities[gz * stride * stride + gy * stride + gx] = 1.0;
        }

        grid.update_density(&densities);

        // Cell (1,1,1) should be solid
        assert!(grid.is_solid(1, 1, 1));
        assert!(grid.cell_capacity(1, 1, 1) < 0.001);

        // Cell (0,0,0) should be air
        assert!(!grid.is_solid(0, 0, 0));
        assert!(grid.cell_capacity(0, 0, 0) > 0.5);

        // Corner densities should be accessible
        let corners = grid.get_corners(1, 1, 1);
        for c in &corners {
            assert!(*c > 0.0);
        }
    }

    #[test]
    fn cell_default_empty() {
        let cell = FluidCell::default();
        assert!(cell.is_empty());
        assert!(!cell.is_source());
    }

    #[test]
    fn is_water_and_is_lava() {
        assert!(FluidType::Water.is_water());
        assert!(FluidType::WaterSpringLine.is_water());
        assert!(FluidType::WaterDrip.is_water());
        assert!(FluidType::WaterBreach.is_water());
        assert!(FluidType::WaterRiver.is_water());
        assert!(FluidType::WaterArtesian.is_water());
        assert!(FluidType::WaterHydrothermal.is_water());
        assert!(FluidType::WaterPool.is_water());
        assert!(!FluidType::Lava.is_water());

        assert!(FluidType::Lava.is_lava());
        assert!(!FluidType::Water.is_lava());
        assert!(!FluidType::WaterSpringLine.is_lava());
        assert!(!FluidType::WaterDrip.is_lava());
        assert!(!FluidType::WaterBreach.is_lava());
        assert!(!FluidType::WaterRiver.is_lava());
        assert!(!FluidType::WaterArtesian.is_lava());
        assert!(!FluidType::WaterHydrothermal.is_lava());
        assert!(!FluidType::WaterPool.is_lava());
    }

    #[test]
    fn from_u8_roundtrip() {
        assert_eq!(FluidType::from_u8(1), FluidType::Water);
        assert_eq!(FluidType::from_u8(2), FluidType::Lava);
        assert_eq!(FluidType::from_u8(3), FluidType::WaterSpringLine);
        assert_eq!(FluidType::from_u8(4), FluidType::WaterDrip);
        assert_eq!(FluidType::from_u8(5), FluidType::WaterBreach);
        assert_eq!(FluidType::from_u8(6), FluidType::WaterRiver);
        assert_eq!(FluidType::from_u8(7), FluidType::WaterArtesian);
        assert_eq!(FluidType::from_u8(8), FluidType::WaterHydrothermal);
        assert_eq!(FluidType::from_u8(9), FluidType::WaterPool);
        // Unknown values default to Water
        assert_eq!(FluidType::from_u8(0), FluidType::Water);
        assert_eq!(FluidType::from_u8(10), FluidType::Water);
        assert_eq!(FluidType::from_u8(255), FluidType::Water);
    }

    #[test]
    fn is_mostly_solid_all_positive() {
        let mut grid = ChunkFluidGrid::new(16);
        grid.set_density(3, 3, 3, 1.0); // all 8 corners positive
        assert!(grid.is_mostly_solid(3, 3, 3, 6));
    }

    #[test]
    fn is_mostly_solid_all_negative() {
        let grid = ChunkFluidGrid::new(16);
        // Default is air (-1.0), all corners negative
        assert!(!grid.is_mostly_solid(3, 3, 3, 6));
    }

    #[test]
    fn is_mostly_solid_mixed_boundary() {
        let size = 4;
        let stride = size + 1;
        let mut grid = ChunkFluidGrid::new(size);

        // Create a density field where cell (1,1,1) has 6/8 corners solid
        let mut densities = vec![-1.0f32; stride * stride * stride];
        for (i, offsets) in CELL_CORNER_OFFSETS.iter().enumerate() {
            let gx = 1 + offsets[0];
            let gy = 1 + offsets[1];
            let gz = 1 + offsets[2];
            if i < 6 {
                densities[gz * stride * stride + gy * stride + gx] = 1.0;
            } else {
                densities[gz * stride * stride + gy * stride + gx] = -1.0;
            }
        }
        grid.update_density(&densities);

        // 6/8 corners solid → is_mostly_solid(threshold=6) = true
        assert!(grid.is_mostly_solid(1, 1, 1, 6));
    }

    #[test]
    fn is_mostly_solid_threshold_8() {
        let size = 4;
        let stride = size + 1;
        let mut grid = ChunkFluidGrid::new(size);

        // Cell (1,1,1) has 6/8 corners solid
        let mut densities = vec![-1.0f32; stride * stride * stride];
        for (i, offsets) in CELL_CORNER_OFFSETS.iter().enumerate() {
            let gx = 1 + offsets[0];
            let gy = 1 + offsets[1];
            let gz = 1 + offsets[2];
            if i < 6 {
                densities[gz * stride * stride + gy * stride + gx] = 1.0;
            } else {
                densities[gz * stride * stride + gy * stride + gx] = -1.0;
            }
        }
        grid.update_density(&densities);

        // With threshold=8, 6/8 is NOT enough
        assert!(!grid.is_mostly_solid(1, 1, 1, 8));
        // With threshold=6, 6/8 IS enough
        assert!(grid.is_mostly_solid(1, 1, 1, 6));
    }
}
