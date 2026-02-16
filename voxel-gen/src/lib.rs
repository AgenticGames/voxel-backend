pub mod config;
pub mod density;
pub mod worm;
pub mod blend;
pub mod hermite_extract;
pub mod pipeline;
pub mod chunk_manager;
pub mod seed;

use voxel_core::chunk::{Chunk, ChunkCoord};
use config::GenerationConfig;
use density::DensityField;

/// Top-level function to generate a single chunk
pub fn generate_chunk(coord: ChunkCoord, config: &GenerationConfig) -> Chunk {
    pipeline::generate(coord, config)
}

/// Generate the full density field for a chunk, including noise + worm carving.
/// This is the shared density pipeline used by both the full pipeline and CLI commands.
pub fn generate_density(coord: ChunkCoord, config: &GenerationConfig) -> DensityField {
    let world_origin = coord.world_origin();
    let c_seed = seed::chunk_seed(config.seed, coord);

    // Step 1: Generate base density from noise
    let mut density = DensityField::generate(config, world_origin);

    // Step 2: Find cavern centers and plan worm connections
    let densities = density.densities();
    let cavern_centers = worm::connect::find_cavern_centers(&densities, density.size, world_origin);

    let r_seed = seed::region_seed(
        config.seed,
        coord.x.div_euclid(4),
        coord.y.div_euclid(4),
        coord.z.div_euclid(4),
    );

    let connections = worm::connect::plan_worm_connections(
        r_seed,
        &cavern_centers,
        config.worm.worms_per_region,
    );

    // Step 3: Generate worm paths and carve into density
    for (i, (start, _end)) in connections.iter().enumerate() {
        let worm_seed = c_seed.wrapping_add(i as u64 * 1000);
        let segments = worm::path::generate_worm_path(
            worm_seed,
            *start,
            config.worm.step_length,
            config.worm.max_steps,
            config.worm.radius_min,
            config.worm.radius_max,
        );
        worm::carve::carve_worm_into_density(
            &mut density,
            &segments,
            world_origin,
            config.worm.falloff_power,
        );
    }

    density
}
