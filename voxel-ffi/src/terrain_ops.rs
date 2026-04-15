use std::collections::HashSet;

use voxel_core::material::Material;
use voxel_gen::config::GenerationConfig;

use crate::store::{sync_boundary_density, ChunkStore};
use crate::types::ConvertedMesh;

/// Flatten a terrace footprint for building placement.
/// The flatten zone is oversized by `MARGIN` voxels beyond the building footprint
/// so that DC edge artifacts are pushed away from the building mesh and hidden.
/// Fills down up to 4 voxels below to bridge air gaps over cliffs.
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
    let clear = clearance_voxels.max(2);
    const MARGIN: i32 = 2; // extra voxels beyond footprint in each direction
    const FILL_DOWN: i32 = 4; // voxels below floor to fill

    let mut dirty_set: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut changed_count = 0u32;

    // Oversized flatten zone with tapered edges.
    // Interior cells get full density (1.0) for a flat floor.
    // Margin cells get decreasing density based on distance from interior,
    // which makes DC place the surface progressively lower — creating a ramp.
    for dx in -MARGIN..(terrace_size + MARGIN) {
        for dz in -MARGIN..(terrace_size + MARGIN) {
            let wx = base.x + dx;
            let wy = base.y;
            let wz = base.z + dz;

            // Chebyshev distance from nearest interior cell
            let dist_x = 0.max(-dx).max(dx - (terrace_size - 1));
            let dist_z = 0.max(-dz).max(dz - (terrace_size - 1));
            let dist = dist_x.max(dist_z); // 0 = interior, 1..MARGIN = margin

            // Tapered floor density: 1.0 at interior, slopes down in margin
            let floor_density = if dist == 0 {
                1.0f32
            } else {
                (1.0 - dist as f32 / (MARGIN as f32 + 1.0)).max(0.05)
            };

            // Floor + clearance (clearance only for interior cells — margin cells
            // only get floor taper, not air carving, to avoid voiding cliff walls)
            let max_dy = if dist == 0 { clear } else { 0 };
            for dy in 0..=max_dy {
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
                        // Floor: use tapered density, but only raise — never lower
                        if floor_density > sample.density {
                            changed_count += 1;
                            sample.density = floor_density;
                            sample.material = host_material;
                        }
                    } else {
                        // Clearance: force to air (interior only)
                        if sample.density != -1.0 || sample.material != Material::Air {
                            changed_count += 1;
                            sample.density = -1.0;
                            sample.material = Material::Air;
                        }
                    }
                    dirty_set.insert(key);
                }
            }

            // Fill-down: solid support under the ramp.
            // Margin cells get tapered fill (less depth, lower density) so cliff
            // walls slope inward instead of being sheer vertical.
            let fill_depth = if dist == 0 {
                FILL_DOWN
            } else {
                (FILL_DOWN - dist).max(1)
            };
            let fill_density = floor_density; // match the taper

            for dy in 1..=fill_depth {
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
                    if sample.density < fill_density {
                        changed_count += 1;
                        sample.density = fill_density;
                        sample.material = host_material;
                        dirty_set.insert(key);
                    } else {
                        break;
                    }
                }
            }

            // Track interior cells as terraced (not the margin)
            if dx >= 0 && dx < terrace_size && dz >= 0 && dz < terrace_size {
                store.terraced_cells.insert((wx, wy, wz));
                store.terraced_columns.insert((wx, wz), wy);
            }
        }
    }

    // Collect all floor cells and their intended densities for post-sync restoration.
    // sync_boundary_density uses min() which can pull down floor densities at chunk seams.
    let mut floor_cells: Vec<((i32, i32, i32), usize, usize, usize, f32)> = Vec::new();
    for dx in -MARGIN..(terrace_size + MARGIN) {
        for dz in -MARGIN..(terrace_size + MARGIN) {
            let wx = base.x + dx;
            let wy = base.y;
            let wz = base.z + dz;
            let dist_x = 0.max(-dx).max(dx - (terrace_size - 1));
            let dist_z = 0.max(-dz).max(dz - (terrace_size - 1));
            let dist = dist_x.max(dist_z);
            let floor_density = if dist == 0 {
                1.0f32
            } else {
                (1.0 - dist as f32 / (MARGIN as f32 + 1.0)).max(0.05)
            };
            let cx = wx.div_euclid(cs);
            let cy = wy.div_euclid(cs);
            let cz = wz.div_euclid(cs);
            let lx = wx.rem_euclid(cs) as usize;
            let ly = wy.rem_euclid(cs) as usize;
            let lz = wz.rem_euclid(cs) as usize;
            floor_cells.push(((cx, cy, cz), lx, ly, lz, floor_density));
        }
    }

    eprintln!("[voxel] flatten_terrace: base=({},{},{}), size={} (+{}margin), clearance={}, fill_down={}, changed={} voxels, dirty={} chunks",
        base.x, base.y, base.z, terrace_size, MARGIN, clear, FILL_DOWN, changed_count, dirty_set.len());

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

    // Post-sync fixup: restore any floor densities that sync_boundary_density pulled down.
    // sync uses min() which is correct for mining but wrong for flatten floors.
    for &(key, lx, ly, lz, intended) in &floor_cells {
        if let Some(density) = store.density_fields.get_mut(&key) {
            let sample = density.get_mut(lx, ly, lz);
            if sample.density < intended {
                sample.density = intended;
            }
        }
    }

    // Mark modified chunks for save persistence
    let dirty_keys: Vec<_> = dirty_chunks.iter().map(|&(k, ..)| k).collect();
    store.modification_tracker.mark_dirty_many(&dirty_keys);

    store.remesh_dirty(&dirty_chunks, config, world_scale)
}

/// Flatten multiple terrace tiles in a single write lock + one remesh pass.
/// Same oversized-margin approach as `flatten_terrace`.
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
    const MARGIN: i32 = 2;
    const FILL_DOWN: i32 = 4;

    let mut dirty_set: HashSet<(i32, i32, i32)> = HashSet::new();

    for (base, host_material) in tiles {
        for dx in -MARGIN..(terrace_size + MARGIN) {
            for dz in -MARGIN..(terrace_size + MARGIN) {
                let wx = base.x + dx;
                let wy = base.y;
                let wz = base.z + dz;

                // Chebyshev distance from interior
                let dist_x = 0.max(-dx).max(dx - (terrace_size - 1));
                let dist_z = 0.max(-dz).max(dz - (terrace_size - 1));
                let dist = dist_x.max(dist_z);
                let floor_density = if dist == 0 {
                    1.0f32
                } else {
                    (1.0 - dist as f32 / (MARGIN as f32 + 1.0)).max(0.05)
                };

                // Clearance only for interior cells — margin only gets floor taper
                let max_dy = if dist == 0 { 2 } else { 0 };
                for dy in 0..=max_dy {
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
                            if floor_density > sample.density {
                                sample.density = floor_density;
                                sample.material = *host_material;
                            }
                        } else {
                            sample.density = -1.0;
                            sample.material = Material::Air;
                        }
                        dirty_set.insert(key);
                    }
                }

                let fill_depth = if dist == 0 { FILL_DOWN } else { (FILL_DOWN - dist).max(1) };
                let fill_density = floor_density;

                for dy in 1..=fill_depth {
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
                        if sample.density < fill_density {
                            sample.density = fill_density;
                            sample.material = *host_material;
                            dirty_set.insert(key);
                        } else {
                            break;
                        }
                    }
                }

                if dx >= 0 && dx < terrace_size && dz >= 0 && dz < terrace_size {
                    store.terraced_cells.insert((wx, wy, wz));
                    store.terraced_columns.insert((wx, wz), wy);
                }
            }
        }
    }

    // Collect floor cells for post-sync restoration
    let mut floor_cells: Vec<((i32, i32, i32), usize, usize, usize, f32)> = Vec::new();
    for (base, _host_material) in tiles {
        for dx in -MARGIN..(terrace_size + MARGIN) {
            for dz in -MARGIN..(terrace_size + MARGIN) {
                let wx = base.x + dx;
                let wy = base.y;
                let wz = base.z + dz;
                let dist_x = 0.max(-dx).max(dx - (terrace_size - 1));
                let dist_z = 0.max(-dz).max(dz - (terrace_size - 1));
                let dist = dist_x.max(dist_z);
                let floor_density = if dist == 0 {
                    1.0f32
                } else {
                    (1.0 - dist as f32 / (MARGIN as f32 + 1.0)).max(0.05)
                };
                let cx = wx.div_euclid(cs);
                let cy = wy.div_euclid(cs);
                let cz = wz.div_euclid(cs);
                let lx = wx.rem_euclid(cs) as usize;
                let ly = wy.rem_euclid(cs) as usize;
                let lz = wz.rem_euclid(cs) as usize;
                floor_cells.push(((cx, cy, cz), lx, ly, lz, floor_density));
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

    // Post-sync fixup: restore floor densities that sync pulled down
    for &(key, lx, ly, lz, intended) in &floor_cells {
        if let Some(density) = store.density_fields.get_mut(&key) {
            let sample = density.get_mut(lx, ly, lz);
            if sample.density < intended {
                sample.density = intended;
            }
        }
    }

    // Mark modified chunks for save persistence
    let dirty_keys: Vec<_> = dirty_chunks.iter().map(|&(k, ..)| k).collect();
    store.modification_tracker.mark_dirty_many(&dirty_keys);

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

            // Check 4-voxel column below: any solid = supported
            let mut column_supported = false;
            for dy in 1..=4i32 {
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
