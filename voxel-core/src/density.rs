use crate::material::Material;
use crate::octree::node::VoxelSample;

/// Flat density field: (chunk_size+1)^3 samples in Z-Y-X order
#[derive(Debug, Clone)]
pub struct DensityField {
    pub samples: Vec<VoxelSample>,
    pub size: usize, // chunk_size + 1
    /// True if ANY cell contains Crystal or Amethyst (geode shell materials).
    /// Computed by `compute_metadata()` after generation completes.
    pub has_geode_material: bool,
    /// Count of non-solid (air) cells in the inner chunk grid (0..chunk_size)^3.
    /// Computed by `compute_metadata()` after generation completes.
    pub air_cell_count: u32,
}

impl DensityField {
    pub fn new(size: usize) -> Self {
        Self {
            samples: vec![VoxelSample::default(); size * size * size],
            size,
            has_geode_material: false,
            air_cell_count: 0,
        }
    }

    #[inline]
    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        z * self.size * self.size + y * self.size + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> &VoxelSample {
        &self.samples[self.index(x, y, z)]
    }

    #[inline]
    pub fn get_mut(&mut self, x: usize, y: usize, z: usize) -> &mut VoxelSample {
        let idx = self.index(x, y, z);
        &mut self.samples[idx]
    }

    /// Get flat density array (for octree builder compatibility)
    pub fn densities(&self) -> Vec<f32> {
        self.samples.iter().map(|s| s.density).collect()
    }

    /// Compute cached metadata (has_geode_material, air_cell_count).
    /// Must be called after all density modifications (noise, worms, pools, formations).
    /// Scans the inner chunk grid (0..chunk_size)^3 in a single pass.
    pub fn compute_metadata(&mut self) {
        let chunk_size = self.size - 1; // inner grid = size - 1
        let mut has_geode = false;
        let mut air_count = 0u32;
        for z in 0..chunk_size {
            for y in 0..chunk_size {
                for x in 0..chunk_size {
                    let sample = self.get(x, y, z);
                    if sample.material.is_geode_shell() {
                        has_geode = true;
                    }
                    if !sample.material.is_solid() {
                        air_count += 1;
                    }
                }
            }
        }
        self.has_geode_material = has_geode;
        self.air_cell_count = air_count;
    }

    /// Downsample this density field by taking every `factor`-th sample.
    /// Produces a field with `size = (self.size - 1) / factor + 1`.
    /// Used to extract base-resolution seam data from high-res ore chunks.
    pub fn downsample(&self, factor: usize) -> DensityField {
        let new_size = (self.size - 1) / factor + 1;
        let mut out = DensityField::new(new_size);
        for z in 0..new_size {
            for y in 0..new_size {
                for x in 0..new_size {
                    let src = self.get(x * factor, y * factor, z * factor);
                    let dst = out.get_mut(x, y, z);
                    *dst = *src;
                }
            }
        }
        out
    }

    /// Force boundary faces to solid, sealing the chunk on specified faces.
    ///
    /// Used with closed-boundary generation to make outer region faces watertight.
    pub fn clamp_boundary_faces(
        &mut self,
        neg_x: bool,
        pos_x: bool,
        neg_y: bool,
        pos_y: bool,
        neg_z: bool,
        pos_z: bool,
    ) {
        let s = self.size;
        for z in 0..s {
            for y in 0..s {
                for x in 0..s {
                    let on_face = (neg_x && x == 0)
                        || (pos_x && x == s - 1)
                        || (neg_y && y == 0)
                        || (pos_y && y == s - 1)
                        || (neg_z && z == 0)
                        || (pos_z && z == s - 1);
                    if on_face {
                        let sample = self.get_mut(x, y, z);
                        sample.density = 1.0;
                        sample.material = Material::Limestone;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_density_field_indexing() {
        let field = DensityField::new(5);
        assert_eq!(field.index(0, 0, 0), 0);
        assert_eq!(field.index(4, 0, 0), 4);
        assert_eq!(field.index(0, 1, 0), 5);
        assert_eq!(field.index(0, 0, 1), 25);
    }

    #[test]
    fn test_clamp_boundary_single_face() {
        let mut field = DensityField::new(5);
        for s in &mut field.samples {
            s.density = -1.0;
            s.material = Material::Air;
        }
        field.clamp_boundary_faces(false, true, false, false, false, false);
        for z in 0..5 {
            for y in 0..5 {
                assert_eq!(field.get(4, y, z).density, 1.0);
                assert!(matches!(field.get(4, y, z).material, Material::Limestone));
                assert_eq!(field.get(2, y, z).density, -1.0);
            }
        }
    }

    #[test]
    fn test_compute_metadata() {
        // size=5 means chunk_size=4, inner grid = 4^3 = 64 cells
        // Default VoxelSample is solid (density=1.0, material=Limestone)
        let mut field = DensityField::new(5);
        field.compute_metadata();
        assert_eq!(field.air_cell_count, 0); // all solid
        assert!(!field.has_geode_material);

        // Set one cell to Crystal (geode shell)
        field.get_mut(1, 1, 1).material = Material::Crystal;
        // Set one cell to air
        field.get_mut(2, 2, 2).density = -1.0;
        field.get_mut(2, 2, 2).material = Material::Air;

        field.compute_metadata();
        assert_eq!(field.air_cell_count, 1);
        assert!(field.has_geode_material);

        // Make all inner cells air
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    let sample = field.get_mut(x, y, z);
                    sample.density = -1.0;
                    sample.material = Material::Air;
                }
            }
        }
        field.compute_metadata();
        assert_eq!(field.air_cell_count, 64);
        assert!(!field.has_geode_material);
    }

    #[test]
    fn test_clamp_boundary_all_faces() {
        let mut field = DensityField::new(5);
        for s in &mut field.samples {
            s.density = -1.0;
            s.material = Material::Air;
        }
        field.clamp_boundary_faces(true, true, true, true, true, true);
        for z in 0..5 {
            for y in 0..5 {
                for x in 0..5 {
                    let on_boundary =
                        x == 0 || x == 4 || y == 0 || y == 4 || z == 0 || z == 4;
                    let sample = field.get(x, y, z);
                    if on_boundary {
                        assert_eq!(sample.density, 1.0, "({x},{y},{z}) should be solid");
                    } else {
                        assert_eq!(
                            sample.density, -1.0,
                            "({x},{y},{z}) should be air"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_downsample() {
        // 33-size field (32-cell grid, i.e. 2x supersampled from 16)
        let mut field = DensityField::new(33);
        // Set known values at factor-2 positions
        for z in 0..17 {
            for y in 0..17 {
                for x in 0..17 {
                    let s = field.get_mut(x * 2, y * 2, z * 2);
                    s.density = (x + y * 17 + z * 17 * 17) as f32;
                    s.material = Material::Iron;
                }
            }
        }
        let down = field.downsample(2);
        assert_eq!(down.size, 17);
        for z in 0..17 {
            for y in 0..17 {
                for x in 0..17 {
                    let s = down.get(x, y, z);
                    assert_eq!(s.density, (x + y * 17 + z * 17 * 17) as f32);
                    assert_eq!(s.material, Material::Iron);
                }
            }
        }
    }
}
