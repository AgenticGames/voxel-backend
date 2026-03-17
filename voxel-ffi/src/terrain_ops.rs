use std::collections::HashSet;

use voxel_core::material::Material;
use voxel_gen::config::GenerationConfig;

use crate::store::{sync_boundary_density, ChunkStore};
use crate::types::ConvertedMesh;

/// Flatten a terrace footprint for building placement.
/// Sets the floor layer to solid host_material, clears `clearance_voxels` layers above,
/// and fills down up to 2 voxels below to connect the platform to the cave floor.
/// Returns the re-meshed dirty chunks (in UE coords).
pub fn flatten_terrace(
    store: &mut ChunkStore,
    base: glam::IVec3,
    host_material: Material,
    config: &GenerationConfig,
    world_scale: f32,
    terrace_size: i32,
    clearance_voxels: i32,
) -> Vec<((i32, i32, i32), ConvertedMesh)> {
    let cs = config.chunk_size as i32;
    let clear = clearance_voxels.max(2); // at least 2 voxels of air

    let mut dirty_set: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut changed_count = 0u32;

    for dx in 0..terrace_size {
        for dz in 0..terrace_size {
            let wx = base.x + dx;
            let wy = base.y;
            let wz = base.z + dz;

            // Process vertical layers: floor (y+0), clearance (y+1 .. y+clear)
            for dy in 0..=clear {
                let vy = wy + dy;
                let cx = wx.div_euclid(cs);
                let cy = vy.div_euclid(cs);
                let cz = wz.div_euclid(cs);
                let lx = wx.rem_euclid(cs) as usize;
                let ly = vy.rem_euclid(cs) as usize;
                let lz = wz.rem_euclid(cs) as usize;
                let key = (cx, cy, cz);

                if let Some(density) = store.density_fields.get_mut(&key) {
                    let sample = density.get_mut(lx, ly, lz);
                    if dy == 0 {
                        // Floor: make solid with host material
                        let (new_d, new_m) = (1.0, host_material);
                        if (sample.density - new_d).abs() > 0.01 || sample.material != new_m {
                            changed_count += 1;
                        }
                        sample.density = new_d;
                        sample.material = new_m;
                    } else {
                        // Clearance: make air
                        let (new_d, new_m) = (-1.0, Material::Air);
                        if (sample.density - new_d).abs() > 0.01 || sample.material != new_m {
                            changed_count += 1;
                        }
                        sample.density = new_d;
                        sample.material = new_m;
                    }
                    dirty_set.insert(key);
                }
            }

            // Fill-down: scan up to 2 voxels below the floor, filling air with solid
            for dy in 1..=2i32 {
                let vy = wy - dy;
                let cx = wx.div_euclid(cs);
                let cy = vy.div_euclid(cs);
                let cz = wz.div_euclid(cs);
                let lx = wx.rem_euclid(cs) as usize;
                let ly = vy.rem_euclid(cs) as usize;
                let lz = wz.rem_euclid(cs) as usize;
                let key = (cx, cy, cz);

                if let Some(density) = store.density_fields.get_mut(&key) {
                    let sample = density.get_mut(lx, ly, lz);
                    if sample.density < 0.0 {
                        // Air gap: fill with host material
                        changed_count += 1;
                        sample.density = 1.0;
                        sample.material = host_material;
                        dirty_set.insert(key);
                    } else {
                        // Already solid (hit cave floor): stop filling this column
                        break;
                    }
                }
            }

            // Track this cell as terraced
            store.terraced_cells.insert((wx, wy, wz));
            store.terraced_columns.insert((wx, wz), wy);
        }
    }

    eprintln!("[voxel] flatten_terrace: base=({},{},{}), size={}, clearance={}, changed={} voxels, dirty={} chunks",
        base.x, base.y, base.z, terrace_size, clear, changed_count, dirty_set.len());

    // Build dirty chunks with full-chunk bounds for remeshing
    let chunk_size = config.chunk_size;
    let mut dirty_chunks: Vec<_> = dirty_set
        .into_iter()
        .map(|key| (key, 0usize, 0usize, 0usize, chunk_size, chunk_size, chunk_size))
        .collect();

    // Sync boundary density between dirty chunks and face neighbors (fixes seams)
    let extra_dirty = sync_boundary_density(
        &mut store.density_fields, &dirty_chunks, config.chunk_size,
    );
    dirty_chunks.extend(extra_dirty);

    store.remesh_dirty(&dirty_chunks, config, world_scale)
}

/// Flatten multiple terrace tiles in a single write lock + one remesh pass.
/// Identical per-cell logic to `flatten_terrace` but all tiles share one dirty_set,
/// one `sync_boundary_density` call, and one `remesh_dirty` call.
pub fn flatten_terrace_batch(
    store: &mut ChunkStore,
    tiles: &[(glam::IVec3, Material)],
    config: &GenerationConfig,
    world_scale: f32,
    terrace_size: i32,
) -> Vec<((i32, i32, i32), ConvertedMesh)> {
    if tiles.is_empty() {
        return Vec::new();
    }

    let cs = config.chunk_size as i32;

    let mut dirty_set: HashSet<(i32, i32, i32)> = HashSet::new();

    for (base, host_material) in tiles {
        for dx in 0..terrace_size {
            for dz in 0..terrace_size {
                let wx = base.x + dx;
                let wy = base.y;
                let wz = base.z + dz;

                for dy in 0..3i32 {
                    let vy = wy + dy;
                    let cx = wx.div_euclid(cs);
                    let cy = vy.div_euclid(cs);
                    let cz = wz.div_euclid(cs);
                    let lx = wx.rem_euclid(cs) as usize;
                    let ly = vy.rem_euclid(cs) as usize;
                    let lz = wz.rem_euclid(cs) as usize;
                    let key = (cx, cy, cz);

                    if let Some(density) = store.density_fields.get_mut(&key) {
                        let sample = density.get_mut(lx, ly, lz);
                        if dy == 0 {
                            sample.density = 1.0;
                            sample.material = *host_material;
                        } else {
                            sample.density = -1.0;
                            sample.material = Material::Air;
                        }
                        dirty_set.insert(key);
                    }
                }

                // Fill-down: scan up to 2 voxels below the floor, filling air with solid
                for dy in 1..=2i32 {
                    let vy = wy - dy;
                    let cx = wx.div_euclid(cs);
                    let cy = vy.div_euclid(cs);
                    let cz = wz.div_euclid(cs);
                    let lx = wx.rem_euclid(cs) as usize;
                    let ly = vy.rem_euclid(cs) as usize;
                    let lz = wz.rem_euclid(cs) as usize;
                    let key = (cx, cy, cz);

                    if let Some(density) = store.density_fields.get_mut(&key) {
                        let sample = density.get_mut(lx, ly, lz);
                        if sample.density < 0.0 {
                            sample.density = 1.0;
                            sample.material = *host_material;
                            dirty_set.insert(key);
                        } else {
                            break;
                        }
                    }
                }

                store.terraced_cells.insert((wx, wy, wz));
                store.terraced_columns.insert((wx, wz), wy);
            }
        }
    }

    let chunk_size = config.chunk_size;
    let mut dirty_chunks: Vec<_> = dirty_set
        .into_iter()
        .map(|key| (key, 0usize, 0usize, 0usize, chunk_size, chunk_size, chunk_size))
        .collect();

    let extra_dirty = sync_boundary_density(
        &mut store.density_fields, &dirty_chunks, config.chunk_size,
    );
    dirty_chunks.extend(extra_dirty);

    store.remesh_dirty(&dirty_chunks, config, world_scale)
}

/// Query floor support for a flatten preview.
/// Checks cells in the 2-voxel column below the terrace floor (base.y-1, base.y-2).
/// A column counts as supported if any voxel in that range is solid.
/// Returns (solid_count, clearance_count) — supported columns and solid cells at dy=1,2 above base.
pub fn query_flatten_support(store: &ChunkStore, base: glam::IVec3, chunk_size: i32, terrace_size: i32) -> (u8, u8) {
    let mut solid_count = 0u8;
    let mut clearance_count = 0u8;
    for dx in 0..terrace_size {
        for dz in 0..terrace_size {
            let wx = base.x + dx;
            let wz = base.z + dz;

            // Check 2-voxel column below: any solid = supported
            let mut column_supported = false;
            for dy in 1..=2i32 {
                let check_y = base.y - dy;
                let cx = wx.div_euclid(chunk_size);
                let cy = check_y.div_euclid(chunk_size);
                let cz = wz.div_euclid(chunk_size);
                let lx = wx.rem_euclid(chunk_size) as usize;
                let ly = check_y.rem_euclid(chunk_size) as usize;
                let lz = wz.rem_euclid(chunk_size) as usize;
                if let Some(df) = store.density_fields.get(&(cx, cy, cz)) {
                    if df.get(lx, ly, lz).density > 0.0 {
                        column_supported = true;
                        break;
                    }
                }
            }
            if column_supported {
                solid_count += 1;
            }

            // Clearance (dy=1 and dy=2)
            for dy in 1i32..=2 {
                let vy = base.y + dy;
                let (cx2, cy2, cz2) = (wx.div_euclid(chunk_size), vy.div_euclid(chunk_size), wz.div_euclid(chunk_size));
                let (lx2, ly2, lz2) = (wx.rem_euclid(chunk_size) as usize, vy.rem_euclid(chunk_size) as usize, wz.rem_euclid(chunk_size) as usize);
                if let Some(df) = store.density_fields.get(&(cx2, cy2, cz2)) {
                    if df.get(lx2, ly2, lz2).density > 0.0 {
                        clearance_count = clearance_count.saturating_add(1);
                    }
                }
            }
        }
    }
    (solid_count, clearance_count)
}

/// Query floor support for a building placement footprint.
/// Returns (solid_count, total_columns, first_floor_material).
pub fn query_building_support(store: &ChunkStore, base: glam::IVec3, chunk_size: i32, terrace_size: i32) -> (u8, u8, Material) {
    let total_columns = (terrace_size * terrace_size) as u8;
    let mut solid_count = 0u8;
    let mut first_mat = Material::Air;
    for dx in 0..terrace_size {
        for dz in 0..terrace_size {
            let wx = base.x + dx;
            let wz = base.z + dz;

            // Check 2-voxel column below: any solid = supported
            for dy in 1..=2i32 {
                let check_y = base.y - dy;
                let cx = wx.div_euclid(chunk_size);
                let cy = check_y.div_euclid(chunk_size);
                let cz = wz.div_euclid(chunk_size);
                let lx = wx.rem_euclid(chunk_size) as usize;
                let ly = check_y.rem_euclid(chunk_size) as usize;
                let lz = wz.rem_euclid(chunk_size) as usize;
                if let Some(df) = store.density_fields.get(&(cx, cy, cz)) {
                    let sample = df.get(lx, ly, lz);
                    if sample.density > 0.0 {
                        solid_count += 1;
                        if first_mat == Material::Air {
                            first_mat = sample.material;
                        }
                        break;
                    }
                }
            }
        }
    }
    (solid_count, total_columns, first_mat)
}

/// Query whether a terrace exists at the given base position.
/// Returns Some(material) of the floor if all cells are terraced, None otherwise.
/// Checks both `base.y` and `base.y - 1` because the mesh surface sits ~0.5
/// voxels above the floor, so building traces can snap to either Y or Y+1.
pub fn query_terrace(store: &ChunkStore, base: glam::IVec3, terrace_size: i32) -> Option<Material> {
    for y_offset in [0, -1] {
        let check_y = base.y + y_offset;
        let all_present = (0..terrace_size).all(|dx| {
            (0..terrace_size).all(|dz| {
                store.terraced_cells.contains(&(base.x + dx, check_y, base.z + dz))
            })
        });
        if all_present {
            let cs = 16i32;
            let cx = base.x.div_euclid(cs);
            let cy = check_y.div_euclid(cs);
            let cz = base.z.div_euclid(cs);
            let lx = base.x.rem_euclid(cs) as usize;
            let ly = check_y.rem_euclid(cs) as usize;
            let lz = base.z.rem_euclid(cs) as usize;
            return store.density_fields
                .get(&(cx, cy, cz))
                .map(|df| df.get(lx, ly, lz).material);
        }
    }
    None
}

/// Find the nearest terraced column within `search_radius` XY voxels and
/// `max_y_diff` vertical voxels of `approx_y`. Returns the floor Y if found.
pub fn query_nearby_terrace_y(
    store: &ChunkStore,
    base_x: i32,
    base_z: i32,
    approx_y: i32,
    search_radius: i32,
    max_y_diff: i32,
) -> Option<i32> {
    let mut best_dist_sq = i32::MAX;
    let mut best_y = None;
    for dx in -search_radius..=search_radius {
        for dz in -search_radius..=search_radius {
            if let Some(&y) = store.terraced_columns.get(&(base_x + dx, base_z + dz)) {
                if (y - approx_y).abs() <= max_y_diff {
                    let dist_sq = dx * dx + dz * dz;
                    if dist_sq < best_dist_sq {
                        best_dist_sq = dist_sq;
                        best_y = Some(y);
                    }
                }
            }
        }
    }
    best_y
}
