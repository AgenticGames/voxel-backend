//! Spatial search methods for `ChunkStore`: spring/ore/spawn/chrysalis
//! location finders, geode-interior tests, and cavern flood-fill.
//!
//! Split out of the original `store.rs` god file (behavior-preserving).

use std::collections::{HashMap, HashSet, VecDeque};

use glam::Vec3;
use rayon::prelude::*;

use super::{CavernLocations, ChunkStore};

impl ChunkStore {
    /// Find the best spring location (wall seep / ceiling drip) near the player.
    /// Scans all loaded density fields for open cavern cells adjacent to a wall.
    /// Returns the world-space (Rust coords) position of the best candidate.
    /// Parallelized with rayon; skips solid chunks and guards geode checks with metadata.
    pub fn find_spring_location(&self, player_pos: Vec3, chunk_size: usize, effective_bounds: f32) -> Option<Vec3> {
        let cs = effective_bounds;
        let cs_i = chunk_size as i32;

        let chunks: Vec<_> = self.density_fields.iter().collect();

        chunks.par_iter()
            .filter_map(|(&(cx, cy, cz), density)| {
                // Skip all-solid chunks (no air cells to search)
                if density.air_cell_count == 0 {
                    return None;
                }

                let origin = Vec3::new(cx as f32 * cs, cy as f32 * cs, cz as f32 * cs);
                let chunk_has_geode = density.has_geode_material;
                let mut best_score: f32 = -1.0;
                let mut best_pos: Option<Vec3> = None;

                for z in 1..(chunk_size - 1) {
                    for y in 1..(chunk_size - 1) {
                        for x in 1..(chunk_size - 1) {
                            let sample = density.get(x, y, z);
                            if sample.material.is_solid() {
                                continue;
                            }

                            let mut air_count: u32 = 0;
                            for dz in -1i32..=1 {
                                for dy in -1i32..=1 {
                                    for dx in -1i32..=1 {
                                        if dx == 0 && dy == 0 && dz == 0 {
                                            continue;
                                        }
                                        let nx = (x as i32 + dx) as usize;
                                        let ny = (y as i32 + dy) as usize;
                                        let nz = (z as i32 + dz) as usize;
                                        if !density.get(nx, ny, nz).material.is_solid() {
                                            air_count += 1;
                                        }
                                    }
                                }
                            }

                            if air_count < 15 {
                                continue;
                            }

                            // Guard geode check behind chunk-level metadata
                            if chunk_has_geode {
                                let wx = cx * cs_i + x as i32;
                                let wy = cy * cs_i + y as i32;
                                let wz = cz * cs_i + z as i32;
                                if self.is_geode_interior(wx, wy, wz, cs_i) {
                                    continue;
                                }
                            }

                            let solid_above = density.get(x, y + 1, z).material.is_solid();
                            let solid_below = density.get(x, y.wrapping_sub(1), z).material.is_solid();
                            let solid_xp = density.get(x + 1, y, z).material.is_solid();
                            let solid_xn = density.get(x.wrapping_sub(1), y, z).material.is_solid();
                            let solid_zp = density.get(x, y, z + 1).material.is_solid();
                            let solid_zn = density.get(x, y, z.wrapping_sub(1)).material.is_solid();

                            let has_solid_side = solid_xp || solid_xn || solid_zp || solid_zn;
                            let air_below = !solid_below;

                            if !solid_above && !has_solid_side {
                                continue;
                            }

                            let wall_bonus = if has_solid_side && !solid_above {
                                2.0_f32
                            } else if solid_above {
                                1.5
                            } else {
                                1.0
                            };

                            let air_below_bonus = if air_below { 1.5_f32 } else { 1.0 };

                            let mut air_above: u32 = 0;
                            for dy in 1..=15u32 {
                                let ny = y + dy as usize;
                                if ny >= density.size {
                                    break;
                                }
                                if !density.get(x, ny, z).material.is_solid() {
                                    air_above += 1;
                                } else {
                                    break;
                                }
                            }
                            let clearance_bonus = if air_above >= 10 {
                                2.0_f32
                            } else if air_above >= 5 {
                                1.5
                            } else {
                                0.5
                            };

                            let world_pos = origin + Vec3::new(x as f32, y as f32, z as f32);
                            let distance = (world_pos - player_pos).length();

                            let score = (air_count as f32) * wall_bonus * air_below_bonus * clearance_bonus
                                / (1.0 + distance);

                            if score > best_score {
                                best_score = score;
                                best_pos = Some(world_pos);
                            }
                        }
                    }
                }

                best_pos.map(|pos| (best_score, pos))
            })
            .reduce_with(|a, b| if a.0 > b.0 { a } else { b })
            .map(|(_, pos)| pos)
    }

    /// Find surface-facing ore voxels near the player, sorted by distance.
    ///
    /// All inputs/outputs are in Rust (voxel) space. The caller is responsible
    /// for converting between UE world space and voxel space.
    ///
    /// - `radius` is in voxel units.
    /// - `material_filter` of `0xFF` means "any ore" (`Material::is_ore()`).
    ///   Any other value is treated as a specific `Material as u8` to match.
    /// - "Surface" means at least one of the 6 face-neighbors is non-solid.
    /// - Returns up to `max_results` voxel centers sorted by squared distance.
    ///
    /// Parallelized with rayon; skips chunks via `has_ore_material` early reject
    /// and a chunk-center broad-phase distance check.
    pub fn find_ore_voxels(
        &self,
        player_pos: Vec3,
        radius: f32,
        material_filter: u8,
        max_results: usize,
        chunk_size: usize,
        effective_bounds: f32,
    ) -> Vec<(Vec3, u8)> {
        if max_results == 0 || radius <= 0.0 || chunk_size < 2 {
            return Vec::new();
        }

        let cs = effective_bounds;
        let voxel_scale = cs / chunk_size as f32;
        let radius_sq = radius * radius;

        // Broad-phase: skip chunks whose center is farther than radius + half-diagonal.
        let half_diag = cs * 0.5 * 3.0_f32.sqrt();
        let broad_dist = radius + half_diag;
        let broad_dist_sq = broad_dist * broad_dist;

        let chunks: Vec<_> = self.density_fields.iter().collect();

        let mut all: Vec<(Vec3, u8, f32)> = chunks
            .par_iter()
            .flat_map_iter(|(&(cx, cy, cz), density)| {
                let mut local: Vec<(Vec3, u8, f32)> = Vec::new();

                // Chunk-level early reject — no ore material anywhere in chunk.
                if !density.has_ore_material {
                    return local.into_iter();
                }

                let origin = Vec3::new(cx as f32 * cs, cy as f32 * cs, cz as f32 * cs);
                let chunk_center = origin + Vec3::splat(cs * 0.5);
                if (chunk_center - player_pos).length_squared() > broad_dist_sq {
                    return local.into_iter();
                }

                // Inner grid scan, leaving a 1-voxel border so 6-neighbor lookups
                // stay in-bounds. Ores on the absolute chunk boundary are missed;
                // acceptable for v1 — the player can move a step to surface them.
                let end = chunk_size; // exclusive — neighbor x+1 < density.size
                for z in 1..end {
                    for y in 1..end {
                        for x in 1..end {
                            let sample = density.get(x, y, z);
                            let mat = sample.material;
                            if !mat.is_solid() {
                                continue;
                            }

                            // Material filter.
                            if material_filter == 0xFF {
                                if !mat.is_ore() {
                                    continue;
                                }
                            } else if (mat as u8) != material_filter {
                                continue;
                            }

                            // Per-face air check — feeds both the surface filter
                            // AND the exposed-centroid calc below.
                            let n_xp = !density.get(x + 1, y, z).material.is_solid();
                            let n_xn = !density.get(x - 1, y, z).material.is_solid();
                            let n_yp = !density.get(x, y + 1, z).material.is_solid();
                            let n_yn = !density.get(x, y - 1, z).material.is_solid();
                            let n_zp = !density.get(x, y, z + 1).material.is_solid();
                            let n_zn = !density.get(x, y, z - 1).material.is_solid();

                            let mut sum_dir = Vec3::ZERO;
                            let mut n_exposed = 0u32;
                            if n_xp { sum_dir.x += 1.0; n_exposed += 1; }
                            if n_xn { sum_dir.x -= 1.0; n_exposed += 1; }
                            if n_yp { sum_dir.y += 1.0; n_exposed += 1; }
                            if n_yn { sum_dir.y -= 1.0; n_exposed += 1; }
                            if n_zp { sum_dir.z += 1.0; n_exposed += 1; }
                            if n_zn { sum_dir.z -= 1.0; n_exposed += 1; }
                            if n_exposed == 0 {
                                continue; // fully buried — not visible
                            }

                            // Geometric voxel center in chunk-local space.
                            let cx_local = (x as f32 + 0.5) * voxel_scale;
                            let cy_local = (y as f32 + 0.5) * voxel_scale;
                            let cz_local = (z as f32 + 0.5) * voxel_scale;

                            // Visible centroid: shift from the voxel's geometric
                            // center toward the average of its exposed-face normals
                            // by half a voxel. For one-face-exposed voxels the
                            // marker hugs that face; for corner voxels it lands on
                            // the visible corner; for fully-exposed voxels the
                            // shift is zero. Without this, a vein with most of its
                            // mass behind the wall plants the bracket inside the
                            // rock and reads as "buggy".
                            let half_scale = 0.5 * voxel_scale;
                            let avg = sum_dir / n_exposed as f32;
                            let world_pos = origin
                                + Vec3::new(cx_local, cy_local, cz_local)
                                + avg * half_scale;

                            let dist_sq = (world_pos - player_pos).length_squared();
                            if dist_sq > radius_sq {
                                continue;
                            }

                            local.push((world_pos, mat as u8, dist_sq));
                        }
                    }
                }

                local.into_iter()
            })
            .collect();

        all.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        all.truncate(max_results);
        all.into_iter().map(|(p, m, _)| (p, m)).collect()
    }

    /// Check whether a bounding box of air voxels exists at a world position,
    /// with a solid floor below. Handles cross-chunk boundaries via div_euclid/rem_euclid.
    /// Returns false if any required chunk is not loaded (conservative).
    pub fn check_clearance(
        &self,
        wx: i32,
        wy: i32,
        wz: i32,
        height: i32,
        radius: i32,
        chunk_size: i32,
    ) -> bool {
        // Floor check: voxel at (wx, wy-1, wz) must be solid
        {
            let cx = wx.div_euclid(chunk_size);
            let cy = (wy - 1).div_euclid(chunk_size);
            let cz = wz.div_euclid(chunk_size);
            let lx = wx.rem_euclid(chunk_size) as usize;
            let ly = (wy - 1).rem_euclid(chunk_size) as usize;
            let lz = wz.rem_euclid(chunk_size) as usize;
            match self.density_fields.get(&(cx, cy, cz)) {
                Some(df) => {
                    if !df.get(lx, ly, lz).material.is_solid() {
                        return false;
                    }
                }
                None => return false,
            }
        }

        // Air column: all voxels in [wy..wy+height] x [wx-radius..wx+radius] x [wz-radius..wz+radius]
        for dy in 0..height {
            for dx in -radius..=radius {
                for dz in -radius..=radius {
                    let vx = wx + dx;
                    let vy = wy + dy;
                    let vz = wz + dz;
                    let cx = vx.div_euclid(chunk_size);
                    let cy = vy.div_euclid(chunk_size);
                    let cz = vz.div_euclid(chunk_size);
                    let lx = vx.rem_euclid(chunk_size) as usize;
                    let ly = vy.rem_euclid(chunk_size) as usize;
                    let lz = vz.rem_euclid(chunk_size) as usize;
                    match self.density_fields.get(&(cx, cy, cz)) {
                        Some(df) => {
                            if df.get(lx, ly, lz).material.is_solid() {
                                return false;
                            }
                        }
                        None => return false,
                    }
                }
            }
        }

        true
    }

    /// Find a validated spawn location for the player capsule.
    /// Parallelized with rayon; skips solid chunks and guards geode checks with metadata.
    pub fn find_spawn_location(
        &self,
        target: Vec3,
        exclude_center: Vec3,
        exclude_radius: f32,
        chunk_size: usize,
        effective_bounds: f32,
        height: i32,
        radius: i32,
    ) -> Option<Vec3> {
        let cs = effective_bounds;
        let excl_r2 = exclude_radius * exclude_radius;
        let cs_i = chunk_size as i32;

        let chunks: Vec<_> = self.density_fields.iter().collect();

        chunks.par_iter()
            .filter_map(|(&(cx, cy, cz), density)| {
                if density.air_cell_count == 0 {
                    return None;
                }

                let origin = Vec3::new(cx as f32 * cs, cy as f32 * cs, cz as f32 * cs);
                let chunk_has_geode = density.has_geode_material;
                let mut best_dist = f32::MAX;
                let mut best_pos: Option<Vec3> = None;

                for z in 1..(chunk_size - 1) {
                    for y in 1..(chunk_size - 1) {
                        for x in 1..(chunk_size - 1) {
                            let sample = density.get(x, y, z);
                            if sample.material.is_solid() {
                                continue;
                            }

                            if chunk_has_geode {
                                let wx = cx * cs_i + x as i32;
                                let wy = cy * cs_i + y as i32;
                                let wz = cz * cs_i + z as i32;
                                if self.is_geode_interior(wx, wy, wz, cs_i) {
                                    continue;
                                }
                            }

                            let world_pos = origin + Vec3::new(x as f32, y as f32, z as f32);

                            if (world_pos - exclude_center).length_squared() < excl_r2 {
                                continue;
                            }

                            let dist = (world_pos - target).length_squared();
                            if dist >= best_dist {
                                continue;
                            }

                            let wx = cx * cs_i + x as i32;
                            let wy = cy * cs_i + y as i32;
                            let wz = cz * cs_i + z as i32;
                            if self.check_clearance(wx, wy, wz, height, radius, cs_i) {
                                best_dist = dist;
                                best_pos = Some(world_pos);
                            }
                        }
                    }
                }

                best_pos.map(|pos| (best_dist, pos))
            })
            .reduce_with(|a, b| if a.0 < b.0 { a } else { b })
            .map(|(_, pos)| pos)
    }

    /// Find a validated spawn location for the chrysalis (quest giver).
    /// Parallelized with rayon; skips solid chunks and guards geode checks with metadata.
    pub fn find_chrysalis_location(
        &self,
        target: Vec3,
        exclude_center: Vec3,
        exclude_radius: f32,
        chunk_size: usize,
        effective_bounds: f32,
        height: i32,
        radius: i32,
    ) -> Option<Vec3> {
        let cs = effective_bounds;
        let excl_r2 = exclude_radius * exclude_radius;
        let cs_i = chunk_size as i32;

        let chunks: Vec<_> = self.density_fields.iter().collect();

        chunks.par_iter()
            .filter_map(|(&(cx, cy, cz), density)| {
                if density.air_cell_count == 0 {
                    return None;
                }

                let origin = Vec3::new(cx as f32 * cs, cy as f32 * cs, cz as f32 * cs);
                let chunk_has_geode = density.has_geode_material;
                let mut best_dist = f32::MAX;
                let mut best_pos: Option<Vec3> = None;

                for z in 2..(chunk_size - 2) {
                    for y in 1..(chunk_size - 1) {
                        for x in 2..(chunk_size - 2) {
                            let sample = density.get(x, y, z);
                            if sample.material.is_solid() {
                                continue;
                            }

                            if chunk_has_geode {
                                let wx = cx * cs_i + x as i32;
                                let wy = cy * cs_i + y as i32;
                                let wz = cz * cs_i + z as i32;
                                if self.is_geode_interior(wx, wy, wz, cs_i) {
                                    continue;
                                }
                            }

                            let world_pos = origin + Vec3::new(x as f32, y as f32, z as f32);

                            if (world_pos - exclude_center).length_squared() < excl_r2 {
                                continue;
                            }

                            let adj_solid = density.get(x + 1, y, z).material.is_solid()
                                || density.get(x.wrapping_sub(1), y, z).material.is_solid()
                                || density.get(x, y, z + 1).material.is_solid()
                                || density.get(x, y, z.wrapping_sub(1)).material.is_solid();
                            if adj_solid {
                                continue;
                            }

                            let mut near_wall = false;
                            'outer: for ddx in -3i32..=3 {
                                for ddz in -3i32..=3 {
                                    if ddx.abs() <= 1 && ddz.abs() <= 1 {
                                        continue;
                                    }
                                    let nx = (x as i32 + ddx) as usize;
                                    let nz = (z as i32 + ddz) as usize;
                                    if nx < density.size && nz < density.size {
                                        if density.get(nx, y, nz).material.is_solid() {
                                            near_wall = true;
                                            break 'outer;
                                        }
                                    }
                                }
                            }
                            if !near_wall {
                                continue;
                            }

                            let dist = (world_pos - target).length_squared();
                            if dist >= best_dist {
                                continue;
                            }

                            let wx = cx * cs_i + x as i32;
                            let wy = cy * cs_i + y as i32;
                            let wz = cz * cs_i + z as i32;
                            if self.check_clearance(wx, wy, wz, height, radius, cs_i) {
                                best_dist = dist;
                                best_pos = Some(world_pos);
                            }
                        }
                    }
                }

                best_pos.map(|pos| (best_dist, pos))
            })
            .reduce_with(|a, b| if a.0 < b.0 { a } else { b })
            .map(|(_, pos)| pos)
    }

    /// Find a wall-adjacent air cell near `target`.
    /// Parallelized with rayon; skips solid chunks and guards geode checks with metadata.
    pub fn find_wall_location_near(
        &self,
        target: Vec3,
        exclude_center: Vec3,
        exclude_radius: f32,
        chunk_size: usize,
        effective_bounds: f32,
    ) -> Option<Vec3> {
        let cs = effective_bounds;
        let excl_r2 = exclude_radius * exclude_radius;
        let cs_i = chunk_size as i32;

        let chunks: Vec<_> = self.density_fields.iter().collect();

        chunks.par_iter()
            .filter_map(|(&(cx, cy, cz), density)| {
                if density.air_cell_count == 0 {
                    return None;
                }

                let origin = Vec3::new(cx as f32 * cs, cy as f32 * cs, cz as f32 * cs);
                let chunk_has_geode = density.has_geode_material;
                let mut best_dist = f32::MAX;
                let mut best_pos: Option<Vec3> = None;

                for z in 1..(chunk_size - 1) {
                    for y in 1..(chunk_size - 1) {
                        for x in 1..(chunk_size - 1) {
                            let sample = density.get(x, y, z);
                            if sample.material.is_solid() {
                                continue;
                            }

                            if chunk_has_geode {
                                let wx = cx * cs_i + x as i32;
                                let wy = cy * cs_i + y as i32;
                                let wz = cz * cs_i + z as i32;
                                if self.is_geode_interior(wx, wy, wz, cs_i) {
                                    continue;
                                }
                            }

                            let world_pos = origin + Vec3::new(x as f32, y as f32, z as f32);

                            if (world_pos - exclude_center).length_squared() < excl_r2 {
                                continue;
                            }

                            let has_wall = density.get(x + 1, y, z).material.is_solid()
                                || density.get(x.wrapping_sub(1), y, z).material.is_solid()
                                || density.get(x, y + 1, z).material.is_solid()
                                || density.get(x, y.wrapping_sub(1), z).material.is_solid()
                                || density.get(x, y, z + 1).material.is_solid()
                                || density.get(x, y, z.wrapping_sub(1)).material.is_solid();

                            if !has_wall {
                                continue;
                            }

                            let dist = (world_pos - target).length_squared();
                            if dist < best_dist {
                                best_dist = dist;
                                best_pos = Some(world_pos);
                            }
                        }
                    }
                }

                best_pos.map(|pos| (best_dist, pos))
            })
            .reduce_with(|a, b| if a.0 < b.0 { a } else { b })
            .map(|(_, pos)| pos)
    }

    /// Check if an air cell is inside a geode (crystal/amethyst shell nearby).
    /// Scans a 5x5x5 cube (radius 2) around the cell for geode shell materials.
    pub fn is_geode_interior(&self, wx: i32, wy: i32, wz: i32, chunk_size: i32) -> bool {
        for dz in -2..=2i32 {
            for dy in -2..=2i32 {
                for dx in -2..=2i32 {
                    let vx = wx + dx;
                    let vy = wy + dy;
                    let vz = wz + dz;
                    let cx = vx.div_euclid(chunk_size);
                    let cy = vy.div_euclid(chunk_size);
                    let cz = vz.div_euclid(chunk_size);
                    let lx = vx.rem_euclid(chunk_size) as usize;
                    let ly = vy.rem_euclid(chunk_size) as usize;
                    let lz = vz.rem_euclid(chunk_size) as usize;
                    if let Some(df) = self.density_fields.get(&(cx, cy, cz)) {
                        if df.get(lx, ly, lz).material.is_geode_shell() {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// BFS flood-fill from a seed cell to find all connected air cells in the same cavern.
    /// 6-connected, cross-chunk aware. Skips solid voxels, geode interiors, and unloaded chunks.
    /// Returns None if the fill exceeds max_cells (cavern too large for constraint).
    /// Uses per-chunk Vec<bool> bitset for O(1) visited checks instead of HashSet hashing.
    pub fn flood_fill_cavern(
        &self,
        seed_wx: i32,
        seed_wy: i32,
        seed_wz: i32,
        chunk_size: i32,
        max_cells: usize,
    ) -> Option<HashSet<(i32, i32, i32)>> {
        let cs = chunk_size as usize;
        let cells_per_chunk = cs * cs * cs;
        let mut visited_chunks: HashMap<(i32, i32, i32), Vec<bool>> = HashMap::new();
        let mut total_visited: usize = 0;
        let mut queue = VecDeque::new();

        // Helper: check and mark visited in bitset
        let is_visited = |chunks: &HashMap<(i32, i32, i32), Vec<bool>>, ck: (i32, i32, i32), lx: usize, ly: usize, lz: usize| -> bool {
            if let Some(bits) = chunks.get(&ck) {
                bits[lz * cs * cs + ly * cs + lx]
            } else {
                false
            }
        };

        let mark_visited = |chunks: &mut HashMap<(i32, i32, i32), Vec<bool>>, ck: (i32, i32, i32), lx: usize, ly: usize, lz: usize| {
            let bits = chunks.entry(ck).or_insert_with(|| vec![false; cells_per_chunk]);
            bits[lz * cs * cs + ly * cs + lx] = true;
        };

        // Seed
        {
            let cx = seed_wx.div_euclid(chunk_size);
            let cy = seed_wy.div_euclid(chunk_size);
            let cz = seed_wz.div_euclid(chunk_size);
            let lx = seed_wx.rem_euclid(chunk_size) as usize;
            let ly = seed_wy.rem_euclid(chunk_size) as usize;
            let lz = seed_wz.rem_euclid(chunk_size) as usize;
            mark_visited(&mut visited_chunks, (cx, cy, cz), lx, ly, lz);
            total_visited += 1;
        }
        queue.push_back((seed_wx, seed_wy, seed_wz));

        let directions: [(i32, i32, i32); 6] = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ];

        while let Some((wx, wy, wz)) = queue.pop_front() {
            if total_visited > max_cells {
                return None;
            }

            for &(dx, dy, dz) in &directions {
                let nx = wx + dx;
                let ny = wy + dy;
                let nz = wz + dz;

                let cx = nx.div_euclid(chunk_size);
                let cy = ny.div_euclid(chunk_size);
                let cz = nz.div_euclid(chunk_size);
                let lx = nx.rem_euclid(chunk_size) as usize;
                let ly = ny.rem_euclid(chunk_size) as usize;
                let lz = nz.rem_euclid(chunk_size) as usize;
                let ck = (cx, cy, cz);

                if is_visited(&visited_chunks, ck, lx, ly, lz) {
                    continue;
                }

                match self.density_fields.get(&ck) {
                    Some(df) => {
                        if df.get(lx, ly, lz).material.is_solid() {
                            continue;
                        }
                        // Guard geode check behind chunk-level metadata
                        if df.has_geode_material && self.is_geode_interior(nx, ny, nz, chunk_size) {
                            continue;
                        }
                    }
                    None => continue,
                }

                mark_visited(&mut visited_chunks, ck, lx, ly, lz);
                total_visited += 1;
                queue.push_back((nx, ny, nz));
            }
        }

        // Convert bitset back to HashSet for downstream compatibility
        let mut result = HashSet::with_capacity(total_visited);
        for (&(cx, cy, cz), bits) in &visited_chunks {
            for z in 0..cs {
                for y in 0..cs {
                    for x in 0..cs {
                        if bits[z * cs * cs + y * cs + x] {
                            let wx = cx * chunk_size + x as i32;
                            let wy = cy * chunk_size + y as i32;
                            let wz = cz * chunk_size + z as i32;
                            result.insert((wx, wy, wz));
                        }
                    }
                }
            }
        }

        Some(result)
    }

    /// Same as find_spawn_location but constrained to a pre-computed cavern volume.
    /// Parallelized with rayon; skips solid chunks.
    pub fn find_spawn_in_cavern(
        &self,
        cavern: &HashSet<(i32, i32, i32)>,
        target: Vec3,
        exclude_center: Vec3,
        exclude_radius: f32,
        chunk_size: usize,
        effective_bounds: f32,
        height: i32,
        radius: i32,
    ) -> Option<Vec3> {
        let cs = effective_bounds;
        let excl_r2 = exclude_radius * exclude_radius;
        let cs_i = chunk_size as i32;

        let chunks: Vec<_> = self.density_fields.iter().collect();

        chunks.par_iter()
            .filter_map(|(&(cx, cy, cz), density)| {
                if density.air_cell_count == 0 {
                    return None;
                }

                let origin = Vec3::new(cx as f32 * cs, cy as f32 * cs, cz as f32 * cs);
                let mut best_dist = f32::MAX;
                let mut best_pos: Option<Vec3> = None;

                for z in 1..(chunk_size - 1) {
                    for y in 1..(chunk_size - 1) {
                        for x in 1..(chunk_size - 1) {
                            let wx = cx * cs_i + x as i32;
                            let wy = cy * cs_i + y as i32;
                            let wz = cz * cs_i + z as i32;

                            if !cavern.contains(&(wx, wy, wz)) {
                                continue;
                            }

                            let sample = density.get(x, y, z);
                            if sample.material.is_solid() {
                                continue;
                            }

                            let world_pos = origin + Vec3::new(x as f32, y as f32, z as f32);

                            if (world_pos - exclude_center).length_squared() < excl_r2 {
                                continue;
                            }

                            let dist = (world_pos - target).length_squared();
                            if dist >= best_dist {
                                continue;
                            }

                            if self.check_clearance(wx, wy, wz, height, radius, cs_i) {
                                best_dist = dist;
                                best_pos = Some(world_pos);
                            }
                        }
                    }
                }

                best_pos.map(|pos| (best_dist, pos))
            })
            .reduce_with(|a, b| if a.0 < b.0 { a } else { b })
            .map(|(_, pos)| pos)
    }

    /// Same as find_chrysalis_location but constrained to a pre-computed cavern volume.
    /// Parallelized with rayon; skips solid chunks.
    pub fn find_chrysalis_in_cavern(
        &self,
        cavern: &HashSet<(i32, i32, i32)>,
        target: Vec3,
        exclude_center: Vec3,
        exclude_radius: f32,
        chunk_size: usize,
        effective_bounds: f32,
        height: i32,
        radius: i32,
    ) -> Option<Vec3> {
        let cs = effective_bounds;
        let excl_r2 = exclude_radius * exclude_radius;
        let cs_i = chunk_size as i32;

        let chunks: Vec<_> = self.density_fields.iter().collect();

        chunks.par_iter()
            .filter_map(|(&(cx, cy, cz), density)| {
                if density.air_cell_count == 0 {
                    return None;
                }

                let origin = Vec3::new(cx as f32 * cs, cy as f32 * cs, cz as f32 * cs);
                let mut best_dist = f32::MAX;
                let mut best_pos: Option<Vec3> = None;

                for z in 2..(chunk_size - 2) {
                    for y in 1..(chunk_size - 1) {
                        for x in 2..(chunk_size - 2) {
                            let wx = cx * cs_i + x as i32;
                            let wy = cy * cs_i + y as i32;
                            let wz = cz * cs_i + z as i32;

                            if !cavern.contains(&(wx, wy, wz)) {
                                continue;
                            }

                            let sample = density.get(x, y, z);
                            if sample.material.is_solid() {
                                continue;
                            }

                            let world_pos = origin + Vec3::new(x as f32, y as f32, z as f32);

                            if (world_pos - exclude_center).length_squared() < excl_r2 {
                                continue;
                            }

                            let adj_solid = density.get(x + 1, y, z).material.is_solid()
                                || density.get(x.wrapping_sub(1), y, z).material.is_solid()
                                || density.get(x, y, z + 1).material.is_solid()
                                || density.get(x, y, z.wrapping_sub(1)).material.is_solid();
                            if adj_solid {
                                continue;
                            }

                            let mut near_wall = false;
                            'outer: for ddx in -3i32..=3 {
                                for ddz in -3i32..=3 {
                                    if ddx.abs() <= 1 && ddz.abs() <= 1 {
                                        continue;
                                    }
                                    let nx = (x as i32 + ddx) as usize;
                                    let nz = (z as i32 + ddz) as usize;
                                    if nx < density.size && nz < density.size {
                                        if density.get(nx, y, nz).material.is_solid() {
                                            near_wall = true;
                                            break 'outer;
                                        }
                                    }
                                }
                            }
                            if !near_wall {
                                continue;
                            }

                            let dist = (world_pos - target).length_squared();
                            if dist >= best_dist {
                                continue;
                            }

                            if self.check_clearance(wx, wy, wz, height, radius, cs_i) {
                                best_dist = dist;
                                best_pos = Some(world_pos);
                            }
                        }
                    }
                }

                best_pos.map(|pos| (best_dist, pos))
            })
            .reduce_with(|a, b| if a.0 < b.0 { a } else { b })
            .map(|(_, pos)| pos)
    }

    /// Combined entry point: find spring, flood-fill cavern, then find chrysalis and spawn
    /// constrained to the same cavern volume.
    /// Falls back to independent (geode-filtered) searches if flood fill overflows.
    pub fn find_cavern_locations(
        &self,
        player_pos: Vec3,
        chunk_size: usize,
        effective_bounds: f32,
    ) -> Option<CavernLocations> {
        // Step 1: Find spring (already geode-filtered)
        let spring = self.find_spring_location(player_pos, chunk_size, effective_bounds)?;

        let cs_i = chunk_size as i32;
        let spring_wx = spring.x as i32;
        let spring_wy = spring.y as i32;
        let spring_wz = spring.z as i32;

        // Step 2: Flood-fill cavern from spring
        let cavern_opt = self.flood_fill_cavern(spring_wx, spring_wy, spring_wz, cs_i, 50_000);

        if let Some(ref cavern) = cavern_opt {
            // Step 3: Find chrysalis in same cavern
            let chrysalis = self.find_chrysalis_in_cavern(
                cavern, spring, spring, 30.0,
                chunk_size, effective_bounds, 4, 2,
            );

            if let Some(chr) = chrysalis {
                // Step 4: Find spawn in same cavern (excluding chrysalis)
                let spawn = self.find_spawn_in_cavern(
                    cavern, spring, chr, 20.0,
                    chunk_size, effective_bounds, 13, 3,
                );

                if let Some(sp) = spawn {
                    return Some(CavernLocations { spring, chrysalis: chr, spawn: sp });
                }
            }
        }

        // Fallback: independent searches (still geode-filtered)
        let chrysalis = self.find_chrysalis_location(
            spring, spring, 30.0,
            chunk_size, effective_bounds, 4, 2,
        )?;

        let spawn = self.find_spawn_location(
            spring, chrysalis, 20.0,
            chunk_size, effective_bounds, 13, 3,
        )?;

        Some(CavernLocations { spring, chrysalis, spawn })
    }
}
