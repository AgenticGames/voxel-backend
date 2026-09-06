use std::collections::{HashMap, HashSet};

use crate::cell::{ChunkFluidGrid, FluidType, MIN_LEVEL, SOURCE_LEVEL};

/// What a squeeze could NOT place in the six neighbours of a shrunk cell
/// (2026-09-06). Chunk-local cell + the volume that would otherwise have
/// evaporated. `sim::displacement` re-injects it as an expanding ring.
#[derive(Debug, Clone, Copy)]
pub struct SqueezedRemainder {
    pub lx: usize,
    pub ly: usize,
    pub lz: usize,
    pub lost: f32,
    pub fluid_type: FluidType,
}

/// After a density update, squeeze excess fluid from cells whose capacity decreased.
/// Excess is pushed to non-solid neighbors; any remainder is evaporated.
/// (Legacy entry: callers that do not track displacement.)
pub fn squeeze_excess_fluid(grid: &mut ChunkFluidGrid) {
    let _ = squeeze_excess_fluid_collect(grid);
}

/// Same squeeze, but RETURNS the remainders instead of evaporating them so
/// the caller can conserve the volume (collapse displacement, 2026-09-06).
pub fn squeeze_excess_fluid_collect(grid: &mut ChunkFluidGrid) -> Vec<SqueezedRemainder> {
    let size = grid.size;
    let mut any_change = false;
    let mut remainders: Vec<SqueezedRemainder> = Vec::new();

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = z * size * size + y * size + x;
                let capacity = grid.cell_cap[idx];
                let level = grid.cells[idx].level;

                if level <= capacity {
                    continue;
                }

                let excess = level - capacity;
                grid.cells[idx].level = capacity;
                any_change = true;

                // Try to push excess to neighbors
                let mut remaining = excess;
                let fluid_type = grid.cells[idx].fluid_type;
                let deltas: [(i32, i32, i32); 6] = [
                    (0, 1, 0), (0, -1, 0), // up/down first
                    (1, 0, 0), (-1, 0, 0),
                    (0, 0, 1), (0, 0, -1),
                ];
                for (dx, dy, dz) in deltas {
                    if remaining < MIN_LEVEL {
                        break;
                    }
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;
                    if nx < 0 || nx >= size as i32 || ny < 0 || ny >= size as i32
                        || nz < 0 || nz >= size as i32
                    {
                        continue;
                    }
                    let ni = nz as usize * size * size + ny as usize * size + nx as usize;
                    let n_capacity = grid.cell_cap[ni];
                    let n_space = (n_capacity - grid.cells[ni].level).max(0.0);
                    if n_space > MIN_LEVEL {
                        let push = remaining.min(n_space);
                        grid.cells[ni].level += push;
                        grid.cells[ni].fluid_type = fluid_type;
                        remaining -= push;
                    }
                }
                // Whatever is left could not be placed locally - hand it to
                // the caller instead of evaporating it.
                if remaining >= MIN_LEVEL {
                    remainders.push(SqueezedRemainder { lx: x, ly: y, lz: z, lost: remaining, fluid_type });
                }
            }
        }
    }

    if any_change {
        grid.dirty = true;
    }
    remainders
}

/// Equalize water levels across connected horizontal regions at each Y level.
///
/// For each Y layer, flood-fills connected water cells across chunk boundaries
/// and averages their levels. This provides instant long-range communication —
/// the "suction" effect that pulls water toward lower elevation openings.
///
/// Returns the set of dirty chunk keys.
pub fn equalize_horizontal(
    chunks: &mut HashMap<(i32, i32, i32), ChunkFluidGrid>,
    chunk_size: usize,
    is_lava: bool,
) -> HashSet<(i32, i32, i32)> {
    let mut dirty = HashSet::new();

    // Build global index: collect all water cells with level > MIN_LEVEL
    // Key: (world_x, world_y, world_z) → (chunk_key, local_x, local_y, local_z, level, capacity)
    let mut water_cells: HashMap<(i32, i32, i32), ((i32, i32, i32), usize, usize, usize, f32, f32, u8)> = HashMap::new();

    for (&chunk_key, grid) in chunks.iter() {
        if !grid.has_fluid {
            continue;
        }
        let size = grid.size;
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let cell = grid.get(x, y, z);
                    if cell.level < MIN_LEVEL {
                        continue;
                    }
                    if cell.fluid_type.is_lava() != is_lava {
                        continue;
                    }
                    // Skip source cells — they maintain their own level
                    if cell.is_source() {
                        continue;
                    }
                    let cap = grid.cell_capacity(x, y, z);
                    if cap < MIN_LEVEL {
                        continue;
                    }
                    let wx = chunk_key.0 * chunk_size as i32 + x as i32;
                    let wy = chunk_key.1 * chunk_size as i32 + y as i32;
                    let wz = chunk_key.2 * chunk_size as i32 + z as i32;
                    // Face gating (2026-08-04, bug #215): record which lateral
                    // faces are open so the BFS can't equalize THROUGH a
                    // rendered surface (thin walls were permeable membranes).
                    let idx = z * size * size + y * size + x;
                    let mut open_mask = 0u8;
                    if !crate::cell::face_blocked(&grid.cell_corners, idx, 1, 0, 0) { open_mask |= 1; }
                    if !crate::cell::face_blocked(&grid.cell_corners, idx, -1, 0, 0) { open_mask |= 2; }
                    if !crate::cell::face_blocked(&grid.cell_corners, idx, 0, 0, 1) { open_mask |= 4; }
                    if !crate::cell::face_blocked(&grid.cell_corners, idx, 0, 0, -1) { open_mask |= 8; }
                    water_cells.insert((wx, wy, wz), (chunk_key, x, y, z, cell.level, cap, open_mask));
                }
            }
        }
    }

    if water_cells.is_empty() {
        return dirty;
    }

    // Flood-fill connected regions on the XZ plane (BFS neighbors hold Y fixed,
    // so regions are naturally Y-disjoint). Iterating water_cells once and
    // letting `visited` dedup is equivalent to — and substantially cheaper
    // than — looping over every Y in [min..=max] and re-filtering by Y, which
    // is O(Y_range * |water_cells|). Collect starts to a Vec so the chunks
    // mutation inside the loop doesn't conflict with iterating water_cells.
    let starts: Vec<(i32, i32, i32)> = water_cells.keys().copied().collect();
    let mut visited: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut region: Vec<(i32, i32, i32)> = Vec::new();
    let mut queue: std::collections::VecDeque<(i32, i32, i32)> = std::collections::VecDeque::new();

    for start in starts {
        if visited.contains(&start) {
            continue;
        }

        // BFS flood-fill on XZ plane at this Y level. Reuse `region` and
        // `queue` across regions to avoid per-region allocations.
        region.clear();
        queue.clear();
        queue.push_back(start);
        visited.insert(start);

        while let Some(pos) = queue.pop_front() {
            region.push(pos);
            // 4-connected on XZ plane, gated on open faces: two water cells
            // separated by a rendered surface are NOT connected.
            let src_mask = water_cells[&pos].6;
            for (bit, &(dx, dz)) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)].iter().enumerate() {
                if src_mask & (1u8 << bit) == 0 {
                    continue;
                }
                let neighbor = (pos.0 + dx, pos.1, pos.2 + dz);
                if !visited.contains(&neighbor) && water_cells.contains_key(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }

        if region.len() < 2 {
            continue; // single cell, nothing to equalize
        }

        // Compute total water and cell count, then average
        let mut total_water = 0.0f32;
        let mut total_cap = 0.0f32;
        for &pos in &region {
            let (_, _, _, _, level, cap, _) = water_cells[&pos];
            total_water += level;
            total_cap += cap;
        }

        // Don't equalize if total capacity is near zero
        if total_cap < MIN_LEVEL {
            continue;
        }

        // Damped equalization: blend toward the average rather than snapping.
        // This preserves flow gradients toward drains while still leveling pools.
        let avg_fill = total_water / total_cap;
        const EQ_DAMPING: f32 = 0.3; // blend 30% toward average each tick

        for &pos in &region {
            let (chunk_key, lx, ly, lz, old_level, cap, _) = water_cells[&pos];
            let target = (avg_fill * cap).min(cap);
            let new_level = old_level + EQ_DAMPING * (target - old_level);
            if (new_level - old_level).abs() > MIN_LEVEL {
                if let Some(grid) = chunks.get_mut(&chunk_key) {
                    let cell = grid.get_mut(lx, ly, lz);
                    cell.level = new_level;
                    grid.dirty = true;
                }
                dirty.insert(chunk_key);
            }
        }
    }

    dirty
}

/// (chunk_key, local x, y, z) — chunk-local cell address used by quench plans.
pub type CellAddr = ((i32, i32, i32), usize, usize, usize);

/// Result of `detect_lava_water_quench` — structured plan for one tick's worth
/// of contact-driven solidification. Applied by the worker thread (which owns
/// the density fields) on the next FluidResult poll.
#[derive(Debug, Default, Clone)]
pub struct QuenchPlan {
    /// Lava cells in direct contact with water — become **Obsidian** voxels.
    /// (Glassy quench skin: real-world lava entering water freezes into a
    /// thin obsidian rind in seconds.)
    pub obsidian: Vec<CellAddr>,
    /// Lava cells one or more rings inward from the obsidian rim — become
    /// **Scoria** voxels. Depth is volume-aware: bigger lava chambers produce
    /// thicker scoria zones (1 voxel for an isolated drip, up to 3 for a
    /// chamber interior).
    pub scoria: Vec<CellAddr>,
    /// Water cells touching the obsidian rim — drained (vaporized as steam).
    pub drained_water: Vec<CellAddr>,
    /// World voxel positions of lava SOURCE cells currently touching water.
    /// Sources are never solidified (real-world pillow lava: the vent keeps
    /// producing magma indefinitely); instead they're registered for ongoing
    /// pillow-mound growth handled by the fluid sim's pillow state machine.
    pub pillow_sources: Vec<(i32, i32, i32)>,
}

const QUENCH_FACE_OFFSETS: [(i32, i32, i32); 6] = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
];

#[inline]
fn in_bounds(p: (i32, i32, i32), size: i32) -> bool {
    p.0 >= 0 && p.0 < size && p.1 >= 0 && p.1 < size && p.2 >= 0 && p.2 < size
}

/// Reusable working sets for `detect_lava_water_quench`. Held by the fluid
/// thread across ticks so the hot quench scan doesn't allocate four HashSets
/// + one Vec + a per-contact-cell BFS HashSet/Vec/Vec on every tick.
///
/// Cleared at the start of each detection. The final `QuenchPlan` still
/// allocates its own Vecs since they're consumed downstream over a channel.
#[derive(Default)]
pub struct QuenchScratch {
    obsidian_set: HashSet<CellAddr>,
    scoria_set: HashSet<CellAddr>,
    drained_set: HashSet<CellAddr>,
    pillow_set: HashSet<(i32, i32, i32)>,
    contact_cells: Vec<CellAddr>,
    bfs_visited: HashSet<(usize, usize, usize)>,
    bfs_frontier: Vec<(usize, usize, usize)>,
    bfs_next: Vec<(usize, usize, usize)>,
}

impl QuenchScratch {
    #[inline]
    fn reset(&mut self) {
        self.obsidian_set.clear();
        self.scoria_set.clear();
        self.drained_set.clear();
        self.pillow_set.clear();
        self.contact_cells.clear();
        // bfs_* are cleared per contact-cell inside the BFS phase
    }
}

/// Detect lava-water contacts and build a structured solidification plan.
///
/// Replaces the older `detect_solidification` which only knew about "one
/// material" rims. This version produces a layered Obsidian + Scoria wall
/// with thickness scaled to the local lava cluster size:
/// - 0..=2 lava face-neighbors of the contact cell → scoria depth 1
/// - 3..=4 → scoria depth 2
/// - 5..=6 → scoria depth 3 (the contact cell sits at a chamber face)
///
/// Sources are never marked for solidification themselves; they are returned
/// in `pillow_sources` so the caller can grow a pillow mound around them
/// over many ticks. The BFS for scoria stays inside the contact cell's
/// chunk — cross-chunk continuation gets a slightly thinner scoria layer
/// at the boundary, which is acceptable for visual purposes.
///
/// Convenience wrapper that allocates a fresh scratch per call. Hot-path
/// callers (the fluid sim thread) should use
/// `detect_lava_water_quench_with_scratch` with a long-lived `QuenchScratch`.
pub fn detect_lava_water_quench(
    chunks: &HashMap<(i32, i32, i32), ChunkFluidGrid>,
) -> QuenchPlan {
    let mut scratch = QuenchScratch::default();
    detect_lava_water_quench_with_scratch(chunks, &mut scratch)
}

/// Same as `detect_lava_water_quench` but reuses caller-owned scratch sets
/// across ticks. Eliminates ~4 HashSet + N_contact_cells × (HashSet + 2 Vecs)
/// allocations per tick during active quench scenes.
pub fn detect_lava_water_quench_with_scratch(
    chunks: &HashMap<(i32, i32, i32), ChunkFluidGrid>,
    scratch: &mut QuenchScratch,
) -> QuenchPlan {
    scratch.reset();

    // First pass: identify contact lava cells (non-source) + drain candidates.
    // Each entry: (chunk_key, x, y, z) of a lava cell touching water.
    for (&key, grid) in chunks {
        if !grid.has_fluid {
            continue;
        }
        // Skip chunks with no lava — quench is a lava-side scan and the flag
        // is recomputed each tick by tick_chunk, so this is a tight short-
        // circuit for the common water-only / empty-chunk case.
        if !grid.has_lava {
            continue;
        }
        let size = grid.size;
        let sz = size as i32;
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let cell = grid.get(x, y, z);
                    if cell.level < MIN_LEVEL || !cell.fluid_type.is_lava() {
                        continue;
                    }
                    let mut touches_water = false;
                    for &(dx, dy, dz) in &QUENCH_FACE_OFFSETS {
                        let np = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
                        if !in_bounds(np, sz) {
                            continue;
                        }
                        let n = grid.get(np.0 as usize, np.1 as usize, np.2 as usize);
                        if n.level < MIN_LEVEL || !n.fluid_type.is_water() {
                            continue;
                        }
                        touches_water = true;
                        if !n.is_source {
                            scratch.drained_set.insert((key, np.0 as usize, np.1 as usize, np.2 as usize));
                        }
                    }
                    if !touches_water {
                        continue;
                    }
                    if cell.is_source {
                        // Source vent: register for pillow growth, don't quench the source itself.
                        let wx = key.0 * sz + x as i32;
                        let wy = key.1 * sz + y as i32;
                        let wz = key.2 * sz + z as i32;
                        scratch.pillow_set.insert((wx, wy, wz));
                    } else {
                        scratch.obsidian_set.insert((key, x, y, z));
                        scratch.contact_cells.push((key, x, y, z));
                    }
                }
            }
        }
    }

    // Second pass: BFS inward from each obsidian cell through lava to build
    // the scoria halo. Depth is volume-aware (count of lava face-neighbors
    // at the contact point). Iterating `contact_cells` directly conflicts
    // with mutating `scratch.bfs_*` inside, so we destructure to split the
    // borrows.
    let QuenchScratch {
        ref contact_cells,
        ref obsidian_set,
        ref mut scoria_set,
        ref mut bfs_visited,
        ref mut bfs_frontier,
        ref mut bfs_next,
        ..
    } = *scratch;

    for &(key, x, y, z) in contact_cells {
        let Some(grid) = chunks.get(&key) else { continue; };
        let size = grid.size;
        let sz = size as i32;

        // Volume sense: how surrounded by lava is the contact cell?
        let mut lava_n: u8 = 0;
        for &(dx, dy, dz) in &QUENCH_FACE_OFFSETS {
            let np = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
            if !in_bounds(np, sz) { continue; }
            let n = grid.get(np.0 as usize, np.1 as usize, np.2 as usize);
            if n.level >= MIN_LEVEL && n.fluid_type.is_lava() {
                lava_n += 1;
            }
        }
        let scoria_depth: u8 = if lava_n >= 5 { 3 } else if lava_n >= 3 { 2 } else { 1 };

        // Frontier BFS within this chunk — reuses scratch sets/vecs.
        bfs_visited.clear();
        bfs_visited.insert((x, y, z));
        bfs_frontier.clear();
        bfs_frontier.push((x, y, z));
        for _ in 0..scoria_depth {
            bfs_next.clear();
            for (px, py, pz) in bfs_frontier.drain(..) {
                for &(dx, dy, dz) in &QUENCH_FACE_OFFSETS {
                    let np = (px as i32 + dx, py as i32 + dy, pz as i32 + dz);
                    if !in_bounds(np, sz) { continue; }
                    let pos = (np.0 as usize, np.1 as usize, np.2 as usize);
                    if !bfs_visited.insert(pos) { continue; }
                    let n = grid.get(pos.0, pos.1, pos.2);
                    if n.level < MIN_LEVEL || !n.fluid_type.is_lava() { continue; }
                    if n.is_source { continue; }
                    let addr = (key, pos.0, pos.1, pos.2);
                    if obsidian_set.contains(&addr) { continue; }
                    scoria_set.insert(addr);
                    bfs_next.push(pos);
                }
            }
            std::mem::swap(bfs_frontier, bfs_next);
            if bfs_frontier.is_empty() { break; }
        }
    }

    QuenchPlan {
        obsidian: scratch.obsidian_set.iter().copied().collect(),
        scoria: scratch.scoria_set.iter().copied().collect(),
        drained_water: scratch.drained_set.iter().copied().collect(),
        pillow_sources: scratch.pillow_set.iter().copied().collect(),
    }
}

/// Try to grow one new obsidian voxel around an active pillow source.
/// Picks the closest water cell (Chebyshev distance) within `max_radius` of
/// the source and converts it to obsidian, draining it locally. Returns the
/// cell address to add to the next QuenchPlan, or None if no suitable water
/// is within reach (pillow stalls until water returns / cap is hit).
///
/// The closest-first scan gives bulbous outward growth: the pillow expands
/// uniformly around the source instead of streaking in one direction.
pub fn try_grow_pillow_voxel(
    source_pos_world: (i32, i32, i32),
    chunks: &mut HashMap<(i32, i32, i32), ChunkFluidGrid>,
    chunk_size: usize,
    max_radius: i32,
) -> Option<CellAddr> {
    let (sx, sy, sz) = source_pos_world;
    let cs = chunk_size as i32;
    let r = max_radius.max(1);
    // Scan rings of increasing Chebyshev distance so we pick the closest water.
    for d in 1..=r {
        for dz in -d..=d {
            for dy in -d..=d {
                for dx in -d..=d {
                    // Only consider the ring at exactly Chebyshev distance `d`
                    if dx.abs().max(dy.abs()).max(dz.abs()) != d {
                        continue;
                    }
                    let (wx, wy, wz) = (sx + dx, sy + dy, sz + dz);
                    let key = (wx.div_euclid(cs), wy.div_euclid(cs), wz.div_euclid(cs));
                    let lx = wx.rem_euclid(cs) as usize;
                    let ly = wy.rem_euclid(cs) as usize;
                    let lz = wz.rem_euclid(cs) as usize;
                    let Some(grid) = chunks.get_mut(&key) else { continue; };
                    let cell = grid.get_mut(lx, ly, lz);
                    if cell.level < MIN_LEVEL { continue; }
                    if !cell.fluid_type.is_water() { continue; }
                    if cell.is_source { continue; }
                    // Drain the water cell locally; the worker will turn its
                    // voxel into Obsidian on the next FluidResult poll.
                    cell.level = 0.0;
                    grid.dirty = true;
                    return Some((key, lx, ly, lz));
                }
            }
        }
    }
    None
}

// Kept for ABI compatibility — callers should migrate to detect_lava_water_quench.
/// Detect water-lava contact and return cells to solidify (legacy single-list path).
/// **DEPRECATED**: use `detect_lava_water_quench` for the layered obsidian + scoria
/// + pillow source plan.
#[allow(dead_code)]
pub fn detect_solidification(
    chunks: &HashMap<(i32, i32, i32), ChunkFluidGrid>,
) -> Vec<((i32, i32, i32), usize, usize, usize)> {
    detect_lava_water_quench(chunks).obsidian
}

/// Regenerate source blocks: source cells always maintain SOURCE_LEVEL,
/// clamped to the cell's actual capacity. Capacity is now fractional
/// (air_corners/8) so a source landing in a partial-rock boundary cell
/// would otherwise over-pour every tick, leaking unbounded fluid through
/// the redistribution pass.
///
/// Also resets `hops_from_source` to 0 on each source so children re-propagate
/// from a fresh hop count each tick (essential for bounded-flow correctness).
///
/// Skips chunks with `has_sources == false`. That flag is set on every
/// source-placing path (AddFluid, geological springs, place_sources,
/// pending_fluid_load) and recomputed each tick by tick_chunk, so the
/// common "chunk has no sources" case pays nothing here.
pub fn regen_sources(chunks: &mut HashMap<(i32, i32, i32), ChunkFluidGrid>) {
    for grid in chunks.values_mut() {
        if !grid.has_sources {
            continue;
        }
        let total = grid.size * grid.size * grid.size;
        for idx in 0..total {
            if grid.cells[idx].is_source() {
                let cap = grid.cell_cap[idx];
                grid.cells[idx].level = SOURCE_LEVEL.min(cap);
                grid.cells[idx].hops_from_source = 0;
                // max_flow_dist persists (it was set when the source was placed).
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use crate::cell::{ChunkDensityCache, ChunkFluidGrid, FluidType, SOURCE_LEVEL};

    fn make_chunk(size: usize) -> ChunkFluidGrid {
        ChunkFluidGrid::new(size)
    }

    fn empty_density_cache() -> HashMap<(i32, i32, i32), ChunkDensityCache> {
        HashMap::new()
    }

    #[test]
    fn squeeze_excess_works() {
        let mut grid = make_chunk(16);
        grid.get_mut(8, 8, 8).level = 0.8;
        grid.get_mut(8, 8, 8).fluid_type = FluidType::Water;
        grid.set_density(8, 8, 8, 0.5);
        squeeze_excess_fluid(&mut grid);

        assert!(grid.get(8, 8, 8).level <= 0.001, "Level should be squeezed to capacity, got {}", grid.get(8, 8, 8).level);
        let deltas: [(i32, i32, i32); 6] = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)];
        let mut total_neighbors = 0.0f32;
        for (dx, dy, dz) in deltas {
            let nx = (8i32 + dx) as usize;
            let ny = (8i32 + dy) as usize;
            let nz = (8i32 + dz) as usize;
            total_neighbors += grid.get(nx, ny, nz).level;
        }
        assert!(total_neighbors > 0.7, "Excess fluid should have been pushed to neighbors, got {}", total_neighbors);
    }

    #[test]
    fn lava_water_solidification() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);
        grid.get_mut(8, 8, 8).level = 0.5;
        grid.get_mut(8, 8, 8).fluid_type = FluidType::Lava;
        grid.get_mut(9, 8, 8).level = 0.5;
        grid.get_mut(9, 8, 8).fluid_type = FluidType::Water;
        grid.has_fluid = true;
        grid.has_lava = true;
        chunks.insert(key, grid);

        let solidify = detect_solidification(&chunks);
        assert!(!solidify.is_empty(), "Should detect solidification");
    }

    #[test]
    fn water_subtype_solidifies_lava() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);
        grid.get_mut(8, 8, 8).level = 0.5;
        grid.get_mut(8, 8, 8).fluid_type = FluidType::Lava;
        grid.get_mut(9, 8, 8).level = 0.5;
        grid.get_mut(9, 8, 8).fluid_type = FluidType::WaterSpringLine;
        grid.has_fluid = true;
        grid.has_lava = true;
        chunks.insert(key, grid);

        let solidify = detect_solidification(&chunks);
        assert!(!solidify.is_empty(), "Water subtype should solidify lava");
    }

    #[test]
    fn source_regenerates() {
        let mut chunks = HashMap::new();
        let key = (0, 0, 0);
        let mut grid = make_chunk(16);
        grid.get_mut(8, 8, 8).level = SOURCE_LEVEL;
        grid.get_mut(8, 8, 8).fluid_type = FluidType::Water;
        grid.get_mut(8, 8, 8).is_source = true;
        grid.has_fluid = true;
        chunks.insert(key, grid);

        let config = crate::FluidConfig::default();
        let density_cache = empty_density_cache();
        super::super::tick_fluid(&mut chunks, &density_cache, 16, false, &config, true);

        regen_sources(&mut chunks);
        let grid = &chunks[&key];
        assert_eq!(grid.get(8, 8, 8).level, SOURCE_LEVEL);
    }
}
