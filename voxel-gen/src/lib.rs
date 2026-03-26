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
pub mod zones;

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
/// Returns the density field, pool descriptors, fluid seeds, and river spring descriptors.
pub fn generate_density(coord: ChunkCoord, config: &GenerationConfig) -> (DensityField, Vec<PoolDescriptor>, Vec<FluidSeed>, Vec<SpringDescriptor>) {
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
    // Only carve worms whose path endpoint lands within 50 voxels (2000 UU) of the target cavern
    let junction_radius = config.worm.radius_max * 1.5;
    let max_endpoint_drift = 50.0; // voxels (= 2000 UU at scale 40)
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
        // Validate: path must actually reach near the target cavern
        if let Some(last) = segments.last() {
            if last.position.distance(*end) > max_endpoint_drift {
                continue; // Worm drifted too far — discard
            }
        }
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
    let river_springs = rivers::carve_rivers(
        &mut density,
        &config.rivers,
        &config.water_table,
        (world_origin.x as f64, world_origin.y as f64, world_origin.z as f64),
        config.seed,
        c_seed,
    );

    // Step 3b: Place cave pools (water/lava lakes on cave floors)
    let (pool_descriptors, mut fluid_seeds) = pools::place_pools(
        &mut density,
        &config.pools,
        world_origin,
        config.seed,
        c_seed,
        (coord.x, coord.y, coord.z),
    );

    // Step 4: Place cave formations (stalactites, stalagmites, columns, flowstone)
    if config.formations.enabled {
        let formation_seeds = formations::place_formations(
            &mut density, &config.formations, world_origin, config.seed, c_seed,
            (coord.x, coord.y, coord.z),
        );
        fluid_seeds.extend(formation_seeds);
    }

    // Step 4b: Apply ore protrusion (push ore surfaces outward with smooth falloff)
    if config.ore_protrusion > 0.0 {
        density::apply_ore_protrusion(&mut density, config.ore_protrusion);
    }

    // Step 5: Compute cached metadata (geode flag, air count) for search optimization
    density.compute_metadata();

    (density, pool_descriptors, fluid_seeds, river_springs)
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

/// Lightweight zone scan: generates density fields (noise only, no worms/ores)
/// for chunks in a radius, then runs zone detection. Returns zone descriptors.
pub fn scan_zones_only(
    center: (i32, i32, i32),
    chunk_radius: i32,
    config: &GenerationConfig,
) -> Vec<zones::ZoneDescriptor> {
    use std::collections::HashMap;

    let mut density_fields: HashMap<(i32, i32, i32), voxel_core::density::DensityField> = HashMap::new();

    // Generate base density (noise only) for all chunks in radius
    for cx in (center.0 - chunk_radius)..=(center.0 + chunk_radius) {
        for cy in (center.1 - chunk_radius)..=(center.1 + chunk_radius) {
            for cz in (center.2 - chunk_radius)..=(center.2 + chunk_radius) {
                let key = (cx, cy, cz);
                let world_origin = glam::Vec3::new(
                    cx as f32 * config.chunk_size as f32,
                    cy as f32 * config.chunk_size as f32,
                    cz as f32 * config.chunk_size as f32,
                );
                let df = density::generate_density_field(
                    config,
                    world_origin,
                );
                density_fields.insert(key, df);
            }
        }
    }

    // Run zone detection on the density fields
    let eb = config.effective_bounds();
    let (descriptors, _bounds, _fluids) = zones::place_zones(
        &mut density_fields,
        &config.zones,
        config.seed,
        eb,
    );

    descriptors
}
