use voxel_core::chunk::ChunkCoord;

/// Derive a deterministic seed for a specific chunk
pub fn chunk_seed(world_seed: u64, coord: ChunkCoord) -> u64 {
    let mut hash = world_seed;
    hash ^= (coord.x as u64).wrapping_mul(0x9E3779B97F4A7C15);
    hash ^= (coord.y as u64).wrapping_mul(0x517CC1B727220A95);
    hash ^= (coord.z as u64).wrapping_mul(0x6C62272E07BB0142);
    // Finalize with a bit mixer
    hash ^= hash >> 30;
    hash = hash.wrapping_mul(0xBF58476D1CE4E5B9);
    hash ^= hash >> 27;
    hash = hash.wrapping_mul(0x94D049BB133111EB);
    hash ^= hash >> 31;
    hash
}

/// Derive a seed for a region (4x4x4 chunks)
pub fn region_seed(world_seed: u64, region_x: i32, region_y: i32, region_z: i32) -> u64 {
    chunk_seed(world_seed, ChunkCoord::new(region_x, region_y, region_z))
}
