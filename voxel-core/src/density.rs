use crate::material::Material;
use crate::octree::node::VoxelSample;

/// Flat density field: (chunk_size+1)^3 samples in Z-Y-X order
#[derive(Debug, Clone)]
pub struct DensityField {
    pub samples: Vec<VoxelSample>,
    pub size: usize, // chunk_size + 1
}

impl DensityField {
    pub fn new(size: usize) -> Self {
        Self {
            samples: vec![VoxelSample::default(); size * size * size],
            size,
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
}
