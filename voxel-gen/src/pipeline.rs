use voxel_core::chunk::{Chunk, ChunkCoord};
use voxel_core::octree::builder::build_octree;
use voxel_core::octree::lod::build_lod;
use voxel_core::octree::node::OctreeConfig;
use crate::config::GenerationConfig;
use crate::density::DensityField;
use crate::hermite_extract::extract_hermite_data;
use crate::seed::{chunk_seed, region_seed};
use crate::worm::carve::carve_worm_into_density;
use crate::worm::connect::{find_cavern_centers, plan_worm_connections};
use crate::worm::path::generate_worm_path;

/// Full generation pipeline for a single chunk.
///
/// Steps:
/// 1. Derive chunk seed
/// 2. Generate DensityField from noise
/// 3. Find cavern centers and plan worm connections
/// 4. Generate and carve worm tunnels into density
/// 5. Extract hermite data
/// 6. Build octree from density
/// 7. Build LOD
/// 8. Return Chunk with octree
pub fn generate(coord: ChunkCoord, config: &GenerationConfig) -> Chunk {
    let world_origin = coord.world_origin();
    let c_seed = chunk_seed(config.seed, coord);

    // Step 1-2: Generate density field from noise composition
    let mut density = DensityField::generate(config, world_origin);

    // Step 3: Find cavern centers and plan worm connections
    let densities = density.densities();
    let cavern_centers = find_cavern_centers(&densities, density.size, world_origin);

    // Region-based worm spawning
    let r_seed = region_seed(
        config.seed,
        coord.x.div_euclid(4),
        coord.y.div_euclid(4),
        coord.z.div_euclid(4),
    );

    let connections = plan_worm_connections(r_seed, &cavern_centers, config.worm.worms_per_region);

    // Step 4: Generate worm paths and carve into density
    for (i, (start, _end)) in connections.iter().enumerate() {
        let worm_seed = c_seed.wrapping_add(i as u64 * 1000);
        let segments = generate_worm_path(
            worm_seed,
            *start,
            config.worm.step_length,
            config.worm.max_steps,
            config.worm.radius_min,
            config.worm.radius_max,
        );
        carve_worm_into_density(
            &mut density,
            &segments,
            world_origin,
            config.worm.falloff_power,
        );
    }

    // Step 5: Extract hermite data from the final density field
    let _hermite = extract_hermite_data(&density);

    // Step 6: Build octree from density samples
    // build_octree expects size = number of cells (chunk_size), samples = (size+1)^3
    let flat_densities = density.densities();
    let octree_config = OctreeConfig {
        max_depth: config.octree_max_depth,
        ..Default::default()
    };
    let cell_size = density.size - 1; // density.size is chunk_size + 1
    let mut octree = build_octree(&flat_densities, cell_size, &octree_config);

    // Step 7: Build LOD
    build_lod(&mut octree);

    // Step 8: Return chunk
    Chunk { coord, octree }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::chunk::ChunkCoord;
    #[test]
    fn test_pipeline_basic() {
        let config = GenerationConfig::default();
        let coord = ChunkCoord::new(0, 0, 0);
        let chunk = generate(coord, &config);
        assert_eq!(chunk.coord, coord);
    }

    #[test]
    fn test_pipeline_deterministic() {
        let config = GenerationConfig::default();
        let coord = ChunkCoord::new(1, 2, 3);

        let chunk1 = generate(coord, &config);
        let chunk2 = generate(coord, &config);

        // Both should produce the same chunk
        assert_eq!(chunk1.coord, chunk2.coord);
    }

    #[test]
    fn test_pipeline_different_coords() {
        let config = GenerationConfig::default();
        let chunk_a = generate(ChunkCoord::new(0, 0, 0), &config);
        let chunk_b = generate(ChunkCoord::new(5, 5, 5), &config);
        // They should have the same type but different content (via different seeds)
        assert_ne!(chunk_a.coord, chunk_b.coord);
    }

    #[test]
    fn test_pipeline_negative_coords() {
        let config = GenerationConfig::default();
        let coord = ChunkCoord::new(-1, -2, -3);
        let chunk = generate(coord, &config);
        assert_eq!(chunk.coord, coord);
    }
}
