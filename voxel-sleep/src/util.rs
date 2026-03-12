//! Shared utility functions for sleep phase modules.

use std::collections::{HashMap, HashSet, VecDeque};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use voxel_core::density::DensityField;
use voxel_core::material::Material;
use voxel_core::stress::world_to_chunk_local;

/// 6-connected face-neighbor offsets.
pub const FACE_OFFSETS: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

/// Write a material (and optionally density) to a voxel AND all overlapping boundary
/// copies in adjacent chunks. Density fields use a 17³ grid for 16-voxel chunks, so
/// boundary voxels (local coord 0 or chunk_size) exist in multiple chunks. This
/// function ensures all copies stay in sync, preventing `sync_boundary_density` from
/// incorrectly "fixing" sleep-modified voxels.
///
/// `new_density`: `None` = material-only change (preserve existing density),
/// `Some(d)` = set both material and density.
pub fn set_voxel_synced(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    chunk_key: (i32, i32, i32),
    lx: usize, ly: usize, lz: usize,
    new_material: Material,
    new_density: Option<f32>,
    chunk_size: usize,
) {
    // Write primary copy
    if let Some(df) = density_fields.get_mut(&chunk_key) {
        let sample = df.get_mut(lx, ly, lz);
        sample.material = new_material;
        if let Some(d) = new_density {
            sample.density = d;
        }
    }

    // Compute mirror positions for each axis where local coord is on a boundary.
    // Local 0 in chunk C is also local chunk_size in chunk C-1.
    // Local chunk_size in chunk C is also local 0 in chunk C+1.
    let mirrors: [(usize, i32, usize); 3] = [
        (lx, chunk_key.0, 0), // x-axis
        (ly, chunk_key.1, 1), // y-axis
        (lz, chunk_key.2, 2), // z-axis
    ];

    // For each axis, determine if there's a mirror neighbor and what the mirror coord is.
    // None = not on boundary, Some((neighbor_offset, mirror_local_coord))
    let mut axis_mirror: [Option<(i32, usize)>; 3] = [None; 3];
    for i in 0..3 {
        let (local, _chunk_coord, _) = mirrors[i];
        if local == 0 {
            axis_mirror[i] = Some((-1, chunk_size));
        } else if local == chunk_size {
            axis_mirror[i] = Some((1, 0));
        }
    }

    // Iterate all 2^3 = 8 combinations of (primary, mirror) per axis.
    // Skip the (primary, primary, primary) case (already written above).
    for x_use_mirror in 0..2u8 {
        for y_use_mirror in 0..2u8 {
            for z_use_mirror in 0..2u8 {
                if x_use_mirror == 0 && y_use_mirror == 0 && z_use_mirror == 0 {
                    continue; // primary already written
                }

                // Each axis must have a mirror available if we want to use it
                let (nx_offset, nx_local) = if x_use_mirror == 1 {
                    match axis_mirror[0] {
                        Some(v) => v,
                        None => continue, // x not on boundary, skip
                    }
                } else {
                    (0, lx)
                };

                let (ny_offset, ny_local) = if y_use_mirror == 1 {
                    match axis_mirror[1] {
                        Some(v) => v,
                        None => continue,
                    }
                } else {
                    (0, ly)
                };

                let (nz_offset, nz_local) = if z_use_mirror == 1 {
                    match axis_mirror[2] {
                        Some(v) => v,
                        None => continue,
                    }
                } else {
                    (0, lz)
                };

                let neighbor_key = (
                    chunk_key.0 + nx_offset,
                    chunk_key.1 + ny_offset,
                    chunk_key.2 + nz_offset,
                );

                if let Some(df) = density_fields.get_mut(&neighbor_key) {
                    let sample = df.get_mut(nx_local, ny_local, nz_local);
                    sample.material = new_material;
                    if let Some(d) = new_density {
                        sample.density = d;
                    }
                }
            }
        }
    }
}

/// Look up material at a world coordinate, returning None if the chunk is not loaded.
pub fn sample_material(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    wx: i32,
    wy: i32,
    wz: i32,
    chunk_size: usize,
) -> Option<Material> {
    let (chunk_key, lx, ly, lz) = world_to_chunk_local(wx, wy, wz, chunk_size);
    density_fields
        .get(&chunk_key)
        .map(|df| df.get(lx, ly, lz).material)
}

/// Count 6-connected neighbors matching a predicate.
pub fn count_neighbors(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    wx: i32,
    wy: i32,
    wz: i32,
    chunk_size: usize,
    predicate: impl Fn(Material) -> bool,
) -> u32 {
    let mut count = 0u32;
    for &(dx, dy, dz) in &FACE_OFFSETS {
        if let Some(mat) = sample_material(density_fields, wx + dx, wy + dy, wz + dz, chunk_size) {
            if predicate(mat) {
                count += 1;
            }
        }
    }
    count
}

#[allow(dead_code)]
/// Vein deposition probability multiplier based on tunnel aperture (air neighbor count).
/// Wider tunnels = more fluid flow = richer veins; narrow cracks = less deposition.
pub fn aperture_multiplier(air_neighbors: u32) -> f32 {
    match air_neighbors {
        0 => 0.0,
        1 => 1.40,
        2 => 1.15,
        3 => 1.00,
        4 => 0.65,
        5 => 0.40,
        _ => 0.20,
    }
}

/// Check if any voxel within Manhattan distance <= radius has the given material.
pub fn has_material_within_radius(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    wx: i32,
    wy: i32,
    wz: i32,
    chunk_size: usize,
    radius: i32,
    target: Material,
) -> bool {
    for dx in -radius..=radius {
        for dy in -radius..=radius {
            for dz in -radius..=radius {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                if dx.abs() + dy.abs() + dz.abs() > radius {
                    continue;
                }
                if let Some(mat) =
                    sample_material(density_fields, wx + dx, wy + dy, wz + dz, chunk_size)
                {
                    if mat == target {
                        return true;
                    }
                }
            }
        }
    }
    false
}

// ──────────────────────────────────────────────────────────────
// Vein Growth System
// ──────────────────────────────────────────────────────────────

/// Directional bias for vein growth morphology.
#[derive(Debug, Clone)]
pub enum VeinBias {
    /// Iron: favor XZ spread (banded layers)
    Horizontal,
    /// Copper: pick 2 directions when possible (dendritic)
    Branching,
    /// Quartz: compress one axis (0=X, 1=Y, 2=Z) for sheet veins
    Planar(u8),
    /// Gold/Sulfide: tight spherical clusters
    Compact,
    /// Veins grow upward on wall faces, thickening into rock behind
    WallClimbing { wall_normal: (i32, i32, i32) },
}

/// Parameters for growing a connected vein deposit.
#[derive(Debug, Clone)]
pub struct VeinGrowthParams {
    pub ore: Material,
    pub min_size: u32,
    pub max_size: u32,
    pub bias: VeinBias,
    /// Skip Hornfels/Skarn during growth (Phase 3 veins should not invade aureole).
    pub exclude_aureole: bool,
}

/// Grow a connected ore vein FROM a seed point INTO surrounding host rock.
///
/// Returns world-coordinate positions to convert to ore. The seed itself
/// is included in the result.
pub fn grow_vein(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    seed: (i32, i32, i32),
    params: &VeinGrowthParams,
    chunk_size: usize,
    rng: &mut ChaCha8Rng,
) -> Vec<(i32, i32, i32)> {
    let target_size = rng.gen_range(params.min_size..=params.max_size) as usize;
    let mut result = vec![seed];
    let mut visited: HashSet<(i32, i32, i32)> = HashSet::new();
    visited.insert(seed);
    let mut frontier: Vec<(i32, i32, i32)> = vec![seed];

    while result.len() < target_size && !frontier.is_empty() {
        // Pick frontier voxel
        let fi = match params.bias {
            VeinBias::Branching => rng.gen_range(0..frontier.len()),
            _ => frontier.len() - 1, // stack-like (depth-first)
        };
        let (fx, fy, fz) = frontier[fi];

        // Collect valid neighbors: solid host rock, not visited, not ore, not air
        let mut candidates: Vec<((i32, i32, i32), f32)> = Vec::new();
        for &(dx, dy, dz) in &FACE_OFFSETS {
            let n = (fx + dx, fy + dy, fz + dz);
            if visited.contains(&n) {
                continue;
            }
            if let Some(mat) = sample_material(density_fields, n.0, n.1, n.2, chunk_size) {
                if !mat.is_host_rock() {
                    continue;
                }
                if params.exclude_aureole && (mat == Material::Hornfels || mat == Material::Skarn) {
                    continue;
                }
                // Directional weight
                let weight = match &params.bias {
                    VeinBias::Horizontal => {
                        if dy != 0 { 0.3 } else { 1.0 }
                    }
                    VeinBias::Planar(axis) => {
                        let on_compressed = match axis {
                            0 => dx != 0,
                            1 => dy != 0,
                            _ => dz != 0,
                        };
                        if on_compressed { 0.3 } else { 1.0 }
                    }
                    VeinBias::Compact => 1.0,
                    VeinBias::Branching => 1.0,
                    VeinBias::WallClimbing { wall_normal } => {
                        // Y+ (up): primary climbing direction
                        if dy > 0 { 3.0 }
                        // Into wall (opposite of wall_normal): depth into rock
                        else if dx == -wall_normal.0 && dy == -wall_normal.1 && dz == -wall_normal.2 { 2.0 }
                        // Lateral (horizontal, perpendicular to wall_normal): width on wall face
                        else if dy == 0 && (dx != wall_normal.0 || dz != wall_normal.2) { 1.5 }
                        // Y- (down): rarely descend
                        else if dy < 0 { 0.3 }
                        // Toward air (same as wall_normal): avoid growing outward
                        else if dx == wall_normal.0 && dy == wall_normal.1 && dz == wall_normal.2 { 0.1 }
                        else { 1.0 }
                    },
                };
                candidates.push((n, weight));
            }
        }

        if candidates.is_empty() {
            frontier.swap_remove(fi);
            continue;
        }

        // Pick how many to add
        let pick_count = match params.bias {
            VeinBias::Branching => 2.min(candidates.len()),
            _ => 1,
        };

        for _ in 0..pick_count {
            if candidates.is_empty() || result.len() >= target_size {
                break;
            }
            // Weighted random selection
            let total_weight: f32 = candidates.iter().map(|(_, w)| w).sum();
            if total_weight <= 0.0 {
                break;
            }
            let mut roll = rng.gen::<f32>() * total_weight;
            let mut chosen_idx = 0;
            for (i, &(_, w)) in candidates.iter().enumerate() {
                roll -= w;
                if roll <= 0.0 {
                    chosen_idx = i;
                    break;
                }
            }
            let (pos, _) = candidates.remove(chosen_idx);
            visited.insert(pos);
            result.push(pos);
            frontier.push(pos);
        }
    }

    result
}

/// Default vein size range for a given ore type.
pub fn default_vein_size(ore: Material) -> (u32, u32) {
    match ore {
        Material::Iron | Material::Copper => (10, 25),
        Material::Tin | Material::Quartz | Material::Pyrite => (6, 15),
        Material::Gold | Material::Sulfide => (3, 8),
        Material::Malachite => (4, 10),
        _ => (4, 12),
    }
}

/// Vein sizes for sleep hydrothermal deposits.
/// Sized to be noticeable discoveries while still smaller than world-gen ore bodies.
pub fn sleep_vein_size(ore: Material) -> (u32, u32) {
    match ore {
        Material::Iron => (18, 40),
        Material::Copper => (14, 30),
        Material::Tin => (20, 50),
        Material::Quartz => (3, 8),
        Material::Gold => (2, 5),
        Material::Sulfide => (12, 25),
        Material::Malachite => (2, 5),
        Material::Pyrite => (5, 12),
        _ => (2, 4),
    }
}

/// Default vein bias for a given ore type.
pub fn default_vein_bias(ore: Material, rng: &mut ChaCha8Rng) -> VeinBias {
    match ore {
        Material::Iron => VeinBias::Horizontal,
        Material::Copper => VeinBias::Branching,
        Material::Quartz => VeinBias::Planar(rng.gen_range(0..3)),
        _ => VeinBias::Compact,
    }
}

// ──────────────────────────────────────────────────────────────
// Water Proximity Helpers (for vein thickening)
// ──────────────────────────────────────────────────────────────

/// Build a set of chunk keys that contain at least one water cell.
pub fn water_chunk_keys(fluid_snapshot: &voxel_fluid::FluidSnapshot) -> HashSet<(i32, i32, i32)> {
    let mut result = HashSet::new();
    for (&chunk_key, cells) in &fluid_snapshot.chunks {
        for cell in cells {
            if cell.fluid_type.is_water() && cell.level > 0.001 {
                result.insert(chunk_key);
                break;
            }
        }
    }
    result
}

/// Chunk-level water proximity classification.
/// Returns: 1 = definitely near water, -1 = definitely far, 0 = uncertain (needs per-voxel check).
pub fn chunk_water_classification(
    chunk_key: (i32, i32, i32),
    water_chunks: &HashSet<(i32, i32, i32)>,
    radius: f32,
    chunk_size: usize,
) -> i8 {
    if water_chunks.is_empty() {
        return -1;
    }
    // Chunk center in world coords
    let half = chunk_size as f32 * 0.5;
    let cx = chunk_key.0 as f32 * chunk_size as f32 + half;
    let cy = chunk_key.1 as f32 * chunk_size as f32 + half;
    let cz = chunk_key.2 as f32 * chunk_size as f32 + half;
    // Chunk diagonal half-length (~13.86 for chunk_size=16)
    let diag = (3.0f32 * (chunk_size as f32 * chunk_size as f32)).sqrt() * 0.5;

    let mut min_dist = f32::MAX;
    for &wk in water_chunks {
        let whalf = chunk_size as f32 * 0.5;
        let wx = wk.0 as f32 * chunk_size as f32 + whalf;
        let wy = wk.1 as f32 * chunk_size as f32 + whalf;
        let wz = wk.2 as f32 * chunk_size as f32 + whalf;
        let dx = cx - wx;
        let dy = cy - wy;
        let dz = cz - wz;
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist < min_dist {
            min_dist = dist;
        }
    }

    if min_dist + diag <= radius {
        1  // definitely near — even farthest corner is within radius
    } else if min_dist - diag > radius {
        -1 // definitely far — even closest corner is beyond radius
    } else {
        0  // uncertain — need per-voxel check
    }
}

/// Per-voxel water proximity check. Scans fluid snapshot cells for water within
/// Euclidean distance <= radius. Only call for "uncertain" chunks.
pub fn is_near_water(
    fluid_snapshot: &voxel_fluid::FluidSnapshot,
    water_chunks: &HashSet<(i32, i32, i32)>,
    wx: i32, wy: i32, wz: i32,
    radius: f32,
    chunk_size: usize,
) -> bool {
    let radius_sq = radius * radius;
    let r_chunks = (radius / chunk_size as f32).ceil() as i32 + 1;
    let voxel_chunk = (
        wx.div_euclid(chunk_size as i32),
        wy.div_euclid(chunk_size as i32),
        wz.div_euclid(chunk_size as i32),
    );

    for dck_x in -r_chunks..=r_chunks {
        for dck_y in -r_chunks..=r_chunks {
            for dck_z in -r_chunks..=r_chunks {
                let ck = (voxel_chunk.0 + dck_x, voxel_chunk.1 + dck_y, voxel_chunk.2 + dck_z);
                if !water_chunks.contains(&ck) {
                    continue;
                }
                if let Some(cells) = fluid_snapshot.chunks.get(&ck) {
                    let cs = fluid_snapshot.chunk_size;
                    for (idx, cell) in cells.iter().enumerate() {
                        if !cell.fluid_type.is_water() || cell.level <= 0.001 {
                            continue;
                        }
                        // Reconstruct world position from flat index
                        let lx = idx % cs;
                        let ly = (idx / cs) % cs;
                        let lz = idx / (cs * cs);
                        let fwx = ck.0 * cs as i32 + lx as i32;
                        let fwy = ck.1 * cs as i32 + ly as i32;
                        let fwz = ck.2 * cs as i32 + lz as i32;
                        let dx = (fwx - wx) as f32;
                        let dy = (fwy - wy) as f32;
                        let dz = (fwz - wz) as f32;
                        if dx * dx + dy * dy + dz * dz <= radius_sq {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

#[allow(dead_code)]
/// BFS through solid rock from a solid starting position to find nearby air voxels.
///
/// Returns `(air_position, bfs_distance)` pairs sorted by distance (closest first).
/// Used by Phase 3 to find air pathways from submerged heat sources (lava in solid bowls).
pub fn find_air_from_solid(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    start: (i32, i32, i32),
    max_radius: u32,
    chunk_size: usize,
) -> Vec<((i32, i32, i32), u32)> {
    let mut queue: VecDeque<((i32, i32, i32), u32)> = VecDeque::new();
    let mut visited: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut air_results: Vec<((i32, i32, i32), u32)> = Vec::new();

    visited.insert(start);
    queue.push_back((start, 0));

    while let Some(((sx, sy, sz), dist)) = queue.pop_front() {
        if dist >= max_radius {
            continue;
        }
        for &(dx, dy, dz) in &FACE_OFFSETS {
            let n = (sx + dx, sy + dy, sz + dz);
            if !visited.insert(n) {
                continue;
            }
            if let Some(mat) = sample_material(density_fields, n.0, n.1, n.2, chunk_size) {
                let next_dist = dist + 1;
                if !mat.is_solid() {
                    // Found air — record it
                    air_results.push((n, next_dist));
                } else {
                    // Solid — continue BFS through rock
                    queue.push_back((n, next_dist));
                }
            }
        }
    }

    air_results.sort_by_key(|&(_, d)| d);
    air_results
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::density::DensityField;

    const CHUNK_SIZE: usize = 16;
    const FIELD_SIZE: usize = CHUNK_SIZE + 1;

    fn make_filled_field(material: Material, density: f32) -> DensityField {
        let mut df = DensityField::new(FIELD_SIZE);
        for s in df.samples.iter_mut() {
            s.material = material;
            s.density = density;
        }
        df
    }

    #[test]
    fn test_set_voxel_synced_x_boundary() {
        // 2×1×1 chunk grid. Write at X=0 in chunk (1,0,0) which mirrors to
        // X=chunk_size in chunk (0,0,0).
        let mut fields: HashMap<(i32, i32, i32), DensityField> = HashMap::new();
        fields.insert((0, 0, 0), make_filled_field(Material::Granite, 1.0));
        fields.insert((1, 0, 0), make_filled_field(Material::Granite, 1.0));

        let target_key = (1, 0, 0);
        let lx = 0;
        let ly = 5;
        let lz = 5;

        set_voxel_synced(&mut fields, target_key, lx, ly, lz, Material::Hornfels, None, CHUNK_SIZE);

        // Primary copy
        assert_eq!(fields[&(1, 0, 0)].get(0, 5, 5).material, Material::Hornfels);
        // Mirror copy in neighbor
        assert_eq!(fields[&(0, 0, 0)].get(CHUNK_SIZE, 5, 5).material, Material::Hornfels);
    }

    #[test]
    fn test_set_voxel_synced_corner() {
        // Write at corner (0,0,0) in chunk (0,0,0) — should mirror to 7 neighbors.
        let mut fields: HashMap<(i32, i32, i32), DensityField> = HashMap::new();
        for cx in -1..=0 {
            for cy in -1..=0 {
                for cz in -1..=0 {
                    fields.insert((cx, cy, cz), make_filled_field(Material::Granite, 1.0));
                }
            }
        }

        set_voxel_synced(&mut fields, (0, 0, 0), 0, 0, 0, Material::Skarn, Some(0.5), CHUNK_SIZE);

        // Check all 8 copies (primary + 7 mirrors)
        let cs = CHUNK_SIZE;
        let expected = [
            ((0, 0, 0), 0, 0, 0),
            ((-1, 0, 0), cs, 0, 0),
            ((0, -1, 0), 0, cs, 0),
            ((0, 0, -1), 0, 0, cs),
            ((-1, -1, 0), cs, cs, 0),
            ((-1, 0, -1), cs, 0, cs),
            ((0, -1, -1), 0, cs, cs),
            ((-1, -1, -1), cs, cs, cs),
        ];
        for &(key, x, y, z) in &expected {
            let s = fields[&key].get(x, y, z);
            assert_eq!(s.material, Material::Skarn, "Failed at {:?} ({},{},{})", key, x, y, z);
            assert_eq!(s.density, 0.5, "Density mismatch at {:?} ({},{},{})", key, x, y, z);
        }
    }

    #[test]
    fn test_set_voxel_synced_missing_neighbor() {
        // Write at boundary with no neighbor chunk — should not panic.
        let mut fields: HashMap<(i32, i32, i32), DensityField> = HashMap::new();
        fields.insert((0, 0, 0), make_filled_field(Material::Granite, 1.0));

        // lx=0 mirrors to chunk (-1,0,0) which doesn't exist — should silently skip
        set_voxel_synced(&mut fields, (0, 0, 0), 0, 8, 8, Material::Hornfels, None, CHUNK_SIZE);

        // Primary copy should still be written
        assert_eq!(fields[&(0, 0, 0)].get(0, 8, 8).material, Material::Hornfels);
    }

    #[test]
    fn test_set_voxel_synced_interior_no_mirror() {
        // Write at interior position (not on any boundary) — no mirrors should be written.
        let mut fields: HashMap<(i32, i32, i32), DensityField> = HashMap::new();
        fields.insert((0, 0, 0), make_filled_field(Material::Granite, 1.0));
        fields.insert((1, 0, 0), make_filled_field(Material::Granite, 1.0));

        set_voxel_synced(&mut fields, (0, 0, 0), 8, 8, 8, Material::Iron, None, CHUNK_SIZE);

        assert_eq!(fields[&(0, 0, 0)].get(8, 8, 8).material, Material::Iron);
        // Neighbor should be untouched
        assert_eq!(fields[&(1, 0, 0)].get(8, 8, 8).material, Material::Granite);
    }
}
