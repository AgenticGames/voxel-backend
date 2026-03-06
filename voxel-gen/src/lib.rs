pub mod config;
pub mod density;
pub mod worm;
pub mod blend;
pub mod formations;
pub mod pools;
pub mod springs;
pub mod lava_tubes;
pub mod rivers;
pub mod crystal_placements;
pub mod hermite_extract;
pub mod pipeline;
pub mod chunk_manager;
pub mod seed;
pub mod region_gen;

use voxel_core::chunk::{Chunk, ChunkCoord};
use config::GenerationConfig;
use density::DensityField;
pub use pools::{FluidSeed, PoolDescriptor};
pub use springs::{SpringDescriptor, SpringType, LavaDescriptor};
pub use crystal_placements::CrystalPlacement;

/// Top-level function to generate a single chunk
pub fn generate_chunk(coord: ChunkCoord, config: &GenerationConfig) -> Chunk {
    pipeline::generate(coord, config)
}

/// Generate the full density field for a chunk, including noise + worm carving + pools.
/// This is the shared density pipeline used by both the full pipeline and CLI commands.
/// Returns the density field, pool descriptors, and fluid seeds placed in this chunk.
pub fn generate_density(coord: ChunkCoord, config: &GenerationConfig) -> (DensityField, Vec<PoolDescriptor>, Vec<FluidSeed>) {
    let world_origin = coord.world_origin_sized(config.chunk_size);
    let c_seed = seed::chunk_seed(config.seed, coord);

    // Step 1: Generate base density from noise
    let mut density = density::generate_density_field(config, world_origin);

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
        config.worm.worms_per_region.ceil() as u32,
    );

    // Step 3: Generate worm paths and carve into density
    let junction_radius = config.worm.radius_max * 1.5;
    for (i, (start, end)) in connections.iter().enumerate() {
        let worm_seed = c_seed.wrapping_add(i as u64 * 1000);
        let segments = worm::path::generate_worm_path(
            worm_seed,
            *start,
            *end,
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
        worm::carve::carve_junction_sphere(&mut density, *start, junction_radius, world_origin, config.worm.falloff_power);
        worm::carve::carve_junction_sphere(&mut density, *end, junction_radius, world_origin, config.worm.falloff_power);
    }

    // Step 3a: Carve lava tubes (basalt-lined tunnels at depth)
    let _tube_lava = lava_tubes::carve_lava_tubes(
        &mut density,
        &config.lava_tubes,
        (world_origin.x as f64, world_origin.y as f64, world_origin.z as f64),
        config.seed,
        c_seed,
    );

    // Step 3a2: Carve underground rivers (wide flat passages in limestone)
    let _river_springs = rivers::carve_rivers(
        &mut density,
        &config.rivers,
        &config.water_table,
        (world_origin.x as f64, world_origin.y as f64, world_origin.z as f64),
        config.seed,
        c_seed,
    );

    // Step 3b: Place cave pools (water/lava lakes on cave floors)
    let (pool_descriptors, fluid_seeds) = pools::place_pools(
        &mut density,
        &config.pools,
        world_origin,
        config.seed,
        c_seed,
    );

    // Step 4: Place cave formations (stalactites, stalagmites, columns, flowstone)
    if config.formations.enabled {
        formations::place_formations(&mut density, &config.formations, world_origin, config.seed, c_seed);
    }

    // Step 4b: Apply ore protrusion (push ore surfaces outward with smooth falloff)
    if config.ore_protrusion > 0.0 {
        density::apply_ore_protrusion(&mut density, config.ore_protrusion);
    }

    // Step 5: Compute cached metadata (geode flag, air count) for search optimization
    density.compute_metadata();

    (density, pool_descriptors, fluid_seeds)
}

/// Compute crystal placements for a chunk after density generation.
/// Pure read-only scan — does NOT modify the density field.
pub fn compute_crystals(
    coord: ChunkCoord,
    density: &voxel_core::density::DensityField,
    config: &GenerationConfig,
) -> Vec<CrystalPlacement> {
    let world_origin = coord.world_origin_sized(config.chunk_size);
    let c_seed = seed::chunk_seed(config.seed, coord);
    crystal_placements::compute_crystal_placements(
        density,
        &config.crystals,
        world_origin,
        config.seed,
        c_seed,
    )
}
