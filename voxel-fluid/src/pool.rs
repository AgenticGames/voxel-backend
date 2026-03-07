use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use crate::cell::{ChunkDensityCache, ChunkFluidGrid, FluidType, MIN_LEVEL, SOURCE_LEVEL};

/// Threshold for considering a cell "full" (for vertical pool stacking).
const FULL_THRESHOLD: f32 = 0.95;

/// Compact reference to a single cell in the world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CellRef {
    chunk: (i32, i32, i32),
    x: u8,
    y: u8,
    z: u8,
}

impl CellRef {
    /// World-space Y coordinate for layer grouping.
    fn world_y(&self, chunk_size: usize) -> i32 {
        self.chunk.1 * chunk_size as i32 + self.y as i32
    }
}

/// A connected pool of fluid cells.
struct Pool {
    cells: Vec<CellRef>,
    fluid_type: FluidType,
}

/// Resolve a potentially out-of-bounds cell coordinate to (chunk_key, local_x, local_y, local_z).
/// Returns None if the coordinate crosses multiple axes (diagonal) which we don't handle.
fn resolve_cell(
    chunk: (i32, i32, i32),
    x: i32,
    y: i32,
    z: i32,
    size: usize,
) -> Option<((i32, i32, i32), u8, u8, u8)> {
    let s = size as i32;
    let mut ck = chunk;
    let mut lx = x;
    let mut ly = y;
    let mut lz = z;
    let mut crosses = 0u8;

    if lx < 0 {
        ck.0 -= 1;
        lx = s - 1;
        crosses += 1;
    } else if lx >= s {
        ck.0 += 1;
        lx = 0;
        crosses += 1;
    }

    if ly < 0 {
        ck.1 -= 1;
        ly = s - 1;
        crosses += 1;
    } else if ly >= s {
        ck.1 += 1;
        ly = 0;
        crosses += 1;
    }

    if lz < 0 {
        ck.2 -= 1;
        lz = s - 1;
        crosses += 1;
    } else if lz >= s {
        ck.2 += 1;
        lz = 0;
        crosses += 1;
    }

    // Allow same-chunk (0 crosses) and single-axis crossing (1 cross)
    if crosses > 1 {
        return None;
    }

    Some((ck, lx as u8, ly as u8, lz as u8))
}

/// Check if a cell is solid in the given chunk grids.
fn is_solid(
    chunks: &HashMap<(i32, i32, i32), ChunkFluidGrid>,
    chunk_densities: &HashMap<(i32, i32, i32), ChunkDensityCache>,
    chunk: (i32, i32, i32),
    x: usize,
    y: usize,
    z: usize,
    chunk_size: usize,
) -> bool {
    if let Some(grid) = chunks.get(&chunk) {
        return grid.is_solid(x, y, z);
    }
    // Fall back to density cache
    if let Some(cache) = chunk_densities.get(&chunk) {
        let idx = z * chunk_size * chunk_size + y * chunk_size + x;
        // All corners positive = solid
        return (0..8).all(|c| cache.cell_corners[idx * 8 + c] > 0.0);
    }
    // Unknown chunk = treat as solid (world edge)
    true
}

/// Get a cell's fluid level, returning 0.0 if the chunk doesn't exist or cell is empty.
fn get_level(
    chunks: &HashMap<(i32, i32, i32), ChunkFluidGrid>,
    chunk: (i32, i32, i32),
    x: usize,
    y: usize,
    z: usize,
) -> f32 {
    if let Some(grid) = chunks.get(&chunk) {
        grid.get(x, y, z).level
    } else {
        0.0
    }
}

/// Get a cell's fluid type.
fn get_fluid_type(
    chunks: &HashMap<(i32, i32, i32), ChunkFluidGrid>,
    chunk: (i32, i32, i32),
    x: usize,
    y: usize,
    z: usize,
) -> FluidType {
    if let Some(grid) = chunks.get(&chunk) {
        grid.get(x, y, z).fluid_type
    } else {
        FluidType::Water
    }
}

/// Detect connected pools of resting water and equalize their levels.
///
/// Three phases:
/// 1. Scan all fluid cells bottom-to-top to find pool candidates (blocked below).
/// 2. Flood-fill BFS to group connected candidates into pools.
/// 3. Equalize each pool: fill layers bottom-up for a flat surface.
///
/// Returns the set of chunks that were modified.
pub fn detect_and_equalize_pools(
    chunks: &mut HashMap<(i32, i32, i32), ChunkFluidGrid>,
    chunk_densities: &HashMap<(i32, i32, i32), ChunkDensityCache>,
    chunk_size: usize,
    is_lava_tick: bool,
) -> HashSet<(i32, i32, i32)> {
    let target_lava = is_lava_tick;

    // Phase 1: Find pool candidates
    // A cell is a candidate if it has fluid, is not solid, and is "blocked below"
    // (cell below is solid, or cell below is full water that is itself blocked).
    // We scan bottom-to-top per column so the "blocked below" check is O(1).

    let mut candidates: HashSet<CellRef> = HashSet::new();

    // We need to read chunk data immutably for candidate detection
    let keys: Vec<(i32, i32, i32)> = chunks.keys().copied().collect();

    for &key in &keys {
        let grid = match chunks.get(&key) {
            Some(g) => g,
            None => continue,
        };
        if !grid.has_fluid {
            continue;
        }

        // Track which cells in this chunk are "blocked below" for stacking.
        // blocked_below[z * size * size + x] for each column stores whether
        // the cell at (x, current_y, z) is a valid pool candidate.
        // We scan y from 0 upward.
        let mut col_blocked: Vec<bool> = vec![false; chunk_size * chunk_size];

        for y in 0..chunk_size {
            for z in 0..chunk_size {
                for x in 0..chunk_size {
                    let cell = grid.get(x, y, z);
                    let level = cell.level;

                    // Skip empty or wrong fluid type
                    if level < MIN_LEVEL {
                        let col_idx = z * chunk_size + x;
                        col_blocked[col_idx] = false;
                        continue;
                    }
                    if cell.fluid_type.is_lava() != target_lava {
                        let col_idx = z * chunk_size + x;
                        col_blocked[col_idx] = false;
                        continue;
                    }
                    if grid.is_solid(x, y, z) {
                        let col_idx = z * chunk_size + x;
                        col_blocked[col_idx] = false;
                        continue;
                    }

                    let col_idx = z * chunk_size + x;

                    // Check if blocked below
                    let blocked = if y == 0 {
                        // Check the cell below in the neighboring chunk
                        if let Some((below_chunk, bx, by, bz)) =
                            resolve_cell(key, x as i32, -1, z as i32, chunk_size)
                        {
                            if is_solid(chunks, chunk_densities, below_chunk, bx as usize, by as usize, bz as usize, chunk_size) {
                                true
                            } else {
                                // Check if below cell is full water and a candidate
                                let below_level = get_level(chunks, below_chunk, bx as usize, by as usize, bz as usize);
                                let below_ref = CellRef { chunk: below_chunk, x: bx, y: by, z: bz };
                                below_level >= FULL_THRESHOLD && candidates.contains(&below_ref)
                            }
                        } else {
                            true // world edge
                        }
                    } else {
                        // Same chunk, check y-1
                        if grid.is_solid(x, y - 1, z) {
                            true
                        } else {
                            // Previous y in same column: was it a full candidate?
                            let below_level = grid.get(x, y - 1, z).level;
                            below_level >= FULL_THRESHOLD && col_blocked[col_idx]
                        }
                    };

                    col_blocked[col_idx] = blocked;

                    if blocked {
                        candidates.insert(CellRef {
                            chunk: key,
                            x: x as u8,
                            y: y as u8,
                            z: z as u8,
                        });
                    }
                }
            }
        }
    }

    if candidates.is_empty() {
        return HashSet::new();
    }

    // Phase 2: Flood-fill BFS to group connected candidates into pools
    let mut visited: HashSet<CellRef> = HashSet::with_capacity(candidates.len());
    let mut pools: Vec<Pool> = Vec::new();

    for &start in &candidates {
        if visited.contains(&start) {
            continue;
        }

        let fluid_type = get_fluid_type(
            chunks,
            start.chunk,
            start.x as usize,
            start.y as usize,
            start.z as usize,
        );

        let mut pool_cells: Vec<CellRef> = Vec::new();
        let mut queue: VecDeque<CellRef> = VecDeque::new();
        queue.push_back(start);
        visited.insert(start);

        while let Some(cell) = queue.pop_front() {
            pool_cells.push(cell);

            // Check 6 face neighbors
            let neighbors: [(i32, i32, i32); 6] = [
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            ];

            for (dx, dy, dz) in neighbors {
                let nx = cell.x as i32 + dx;
                let ny = cell.y as i32 + dy;
                let nz = cell.z as i32 + dz;

                let resolved = resolve_cell(cell.chunk, nx, ny, nz, chunk_size);
                let (nchunk, nlx, nly, nlz) = match resolved {
                    Some(r) => r,
                    None => continue,
                };

                let neighbor_ref = CellRef {
                    chunk: nchunk,
                    x: nlx,
                    y: nly,
                    z: nlz,
                };

                if visited.contains(&neighbor_ref) || !candidates.contains(&neighbor_ref) {
                    continue;
                }

                // Vertical connectivity: for +Y/-Y connections, the lower cell must be full
                if dy != 0 {
                    let (lower_chunk, lower_x, lower_y, lower_z) = if dy > 0 {
                        // Moving up: current cell is the lower one
                        (cell.chunk, cell.x as usize, cell.y as usize, cell.z as usize)
                    } else {
                        // Moving down: neighbor is the lower one
                        (nchunk, nlx as usize, nly as usize, nlz as usize)
                    };
                    let lower_level = get_level(chunks, lower_chunk, lower_x, lower_y, lower_z);
                    if lower_level < FULL_THRESHOLD {
                        continue;
                    }
                }

                visited.insert(neighbor_ref);
                queue.push_back(neighbor_ref);
            }
        }

        if pool_cells.len() >= 2 {
            pools.push(Pool {
                cells: pool_cells,
                fluid_type,
            });
        }
    }

    // Phase 3: Equalize each pool
    let mut dirty: HashSet<(i32, i32, i32)> = HashSet::new();

    for pool in &pools {
        // Separate source and non-source cells
        let mut non_source_cells: Vec<CellRef> = Vec::new();
        let mut total_non_source_volume: f32 = 0.0;

        for &cell in &pool.cells {
            let level = get_level(chunks, cell.chunk, cell.x as usize, cell.y as usize, cell.z as usize);
            if level >= SOURCE_LEVEL {
                // Sources stay at their level, not redistributed
            } else {
                non_source_cells.push(cell);
                total_non_source_volume += level;
            }
        }

        if non_source_cells.is_empty() {
            continue; // All sources, nothing to equalize
        }

        // Total volume available for redistribution (sources contribute but keep their level)
        let redistribute_volume = total_non_source_volume;

        // Group non-source cells by world-Y layer
        let mut layers: BTreeMap<i32, Vec<CellRef>> = BTreeMap::new();
        for &cell in &non_source_cells {
            let wy = cell.world_y(chunk_size);
            layers.entry(wy).or_default().push(cell);
        }

        // Fill bottom-up
        let mut remaining = redistribute_volume;
        let mut cell_levels: HashMap<CellRef, f32> = HashMap::new();

        for (_wy, layer_cells) in &layers {
            let layer_count = layer_cells.len() as f32;

            if remaining <= 0.0 {
                // No fluid left — these cells are empty
                for &cell in layer_cells {
                    cell_levels.insert(cell, 0.0);
                }
                continue;
            }

            let needed_to_fill = layer_count; // each cell holds up to 1.0
            if remaining >= needed_to_fill {
                // Full layer
                for &cell in layer_cells {
                    cell_levels.insert(cell, 1.0);
                }
                remaining -= needed_to_fill;
            } else {
                // Surface layer — distribute evenly
                let surface_level = remaining / layer_count;
                for &cell in layer_cells {
                    cell_levels.insert(cell, surface_level);
                }
                remaining = 0.0;
            }
        }

        // Write equalized levels back to chunk grids
        for (&cell, &new_level) in &cell_levels {
            if let Some(grid) = chunks.get_mut(&cell.chunk) {
                let old_level = grid.get(cell.x as usize, cell.y as usize, cell.z as usize).level;
                let diff = (new_level - old_level).abs();
                if diff > MIN_LEVEL {
                    let fluid_cell = grid.get_mut(cell.x as usize, cell.y as usize, cell.z as usize);
                    fluid_cell.level = new_level;
                    fluid_cell.fluid_type = pool.fluid_type;
                    grid.dirty = true;
                    if new_level >= MIN_LEVEL {
                        grid.has_fluid = true;
                    }
                    dirty.insert(cell.chunk);
                }
            }
        }
    }

    // Phase 4: Edge Expansion
    // For each equalized pool, seed dry neighbor cells at the pool surface
    // so pools can grow past basin edges one cell per tick.
    for pool in &pools {
        // Find the surface layer (highest world-Y with fluid)
        let mut max_wy = i32::MIN;
        for &cell in &pool.cells {
            let wy = cell.world_y(chunk_size);
            if wy > max_wy {
                max_wy = wy;
            }
        }

        // Collect surface-layer pool cells
        let surface_cells: Vec<CellRef> = pool.cells.iter()
            .filter(|c| c.world_y(chunk_size) == max_wy)
            .copied()
            .collect();

        for &cell in &surface_cells {
            let cell_level = get_level(chunks, cell.chunk, cell.x as usize, cell.y as usize, cell.z as usize);
            if cell_level < 0.1 || cell_level >= SOURCE_LEVEL {
                continue; // not enough to donate, or source (never drain sources)
            }

            // Check 4 horizontal neighbors
            let neighbors: [(i32, i32, i32); 4] = [
                (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1),
            ];
            for (dx, dy, dz) in neighbors {
                let nx = cell.x as i32 + dx;
                let ny = cell.y as i32 + dy;
                let nz = cell.z as i32 + dz;
                let resolved = resolve_cell(cell.chunk, nx, ny, nz, chunk_size);
                let (nchunk, nlx, nly, nlz) = match resolved {
                    Some(r) => r,
                    None => continue,
                };

                let neighbor_ref = CellRef { chunk: nchunk, x: nlx, y: nly, z: nlz };

                // Skip if already in pool
                if candidates.contains(&neighbor_ref) {
                    continue;
                }
                // Skip if solid
                if is_solid(chunks, chunk_densities, nchunk, nlx as usize, nly as usize, nlz as usize, chunk_size) {
                    continue;
                }
                // Must have solid below (sitting on a floor)
                let below = resolve_cell(nchunk, nlx as i32, nly as i32 - 1, nlz as i32, chunk_size);
                let has_floor = match below {
                    Some((bck, bx, by, bz)) => {
                        is_solid(chunks, chunk_densities, bck, bx as usize, by as usize, bz as usize, chunk_size)
                    }
                    None => true, // world edge = floor
                };
                if !has_floor {
                    continue;
                }
                // Must have room for water
                let nbr_level = get_level(chunks, nchunk, nlx as usize, nly as usize, nlz as usize);
                if nbr_level >= cell_level {
                    continue;
                }
                // Seed a small transfer
                let transfer = (cell_level * 0.1).min(0.05);
                // Re-check donor level (may have donated to previous neighbor)
                let donor_level = get_level(chunks, cell.chunk, cell.x as usize, cell.y as usize, cell.z as usize);
                if donor_level < 0.1 {
                    break; // exhausted
                }
                let transfer = transfer.min(donor_level - 0.05);
                if transfer < MIN_LEVEL {
                    continue;
                }
                // Transfer from pool cell to neighbor
                if let Some(grid) = chunks.get_mut(&cell.chunk) {
                    let fc = grid.get_mut(cell.x as usize, cell.y as usize, cell.z as usize);
                    fc.level -= transfer;
                    grid.dirty = true;
                    dirty.insert(cell.chunk);
                }
                if let Some(grid) = chunks.get_mut(&nchunk) {
                    let fc = grid.get_mut(nlx as usize, nly as usize, nlz as usize);
                    fc.level += transfer;
                    fc.fluid_type = pool.fluid_type;
                    grid.dirty = true;
                    grid.has_fluid = true;
                    dirty.insert(nchunk);
                }
            }
        }
    }

    dirty
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::ChunkFluidGrid;

    fn make_air_chunk(size: usize) -> ChunkFluidGrid {
        ChunkFluidGrid::new(size) // default is all air
    }

    fn set_solid_floor(grid: &mut ChunkFluidGrid, y: usize, size: usize) {
        for z in 0..size {
            for x in 0..size {
                grid.set_density(x, y, z, 1.0);
            }
        }
    }

    #[test]
    fn flat_floor_equalization() {
        // 5 cells on a flat floor at y=1 (floor at y=0), with varying levels.
        // After equalization, all should have the same level.
        // Enclosed by walls so edge expansion doesn't drain volume.
        let size = 16;
        let chunk_key = (0, 0, 0);
        let mut grid = make_air_chunk(size);

        // Solid floor at y=0
        set_solid_floor(&mut grid, 0, size);

        // Walls at y=1 to contain the pool (prevent edge expansion)
        grid.set_density(5, 1, 0, 1.0); // wall to the right
        for x in 0..6 {
            grid.set_density(x, 1, 1, 1.0); // wall behind (z=1)
        }

        // Place water at y=1, x=0..5, z=0 with varying levels
        let levels = [0.8, 0.6, 0.4, 0.2, 0.5];
        let total: f32 = levels.iter().sum();
        for (i, &level) in levels.iter().enumerate() {
            let cell = grid.get_mut(i, 1, 0);
            cell.level = level;
            cell.fluid_type = FluidType::Water;
        }
        grid.has_fluid = true;

        let mut chunks = HashMap::new();
        chunks.insert(chunk_key, grid);
        let densities = HashMap::new();

        let dirty = detect_and_equalize_pools(&mut chunks, &densities, size, false);
        assert!(!dirty.is_empty());

        let grid = chunks.get(&chunk_key).unwrap();
        let expected = total / levels.len() as f32;
        for i in 0..levels.len() {
            let level = grid.get(i, 1, 0).level;
            assert!(
                (level - expected).abs() < 0.01,
                "Cell {} has level {} but expected {}",
                i,
                level,
                expected
            );
        }
    }

    #[test]
    fn uneven_floor_fills_low_spots_first() {
        // Floor at y=0 everywhere, except x=0 has floor at y=1 (raised).
        // Place water at y=1 for x=1..3 and y=2 for x=0..3.
        // Water should fill the lower spots (y=1) first.
        let size = 16;
        let chunk_key = (0, 0, 0);
        let mut grid = make_air_chunk(size);

        // Solid floor at y=0 for all
        set_solid_floor(&mut grid, 0, size);
        // Raised floor at y=1 for x=0 only (making x=0 only available at y=2)
        grid.set_density(0, 1, 0, 1.0);

        // Water at y=1 for x=1,2 (sitting on y=0 floor).
        // Use 0.99 — above FULL_THRESHOLD (0.95) so stacking works,
        // but below SOURCE_LEVEL (1.0) so they're not treated as sources.
        for x in 1..3 {
            let cell = grid.get_mut(x, 1, 0);
            cell.level = 0.99;
            cell.fluid_type = FluidType::Water;
        }
        // Water at y=2 for x=0,1,2 (x=0 sits on y=1 raised floor, x=1,2 stacked on full y=1)
        let y2_levels = [0.6, 0.2, 0.4];
        for (i, x) in (0..3).enumerate() {
            let cell = grid.get_mut(x, 2, 0);
            cell.level = y2_levels[i];
            cell.fluid_type = FluidType::Water;
        }
        grid.has_fluid = true;

        let mut chunks = HashMap::new();
        chunks.insert(chunk_key, grid);
        let densities = HashMap::new();

        let dirty = detect_and_equalize_pools(&mut chunks, &densities, size, false);
        assert!(!dirty.is_empty());

        let grid = chunks.get(&chunk_key).unwrap();
        // Total non-source volume: 2*0.99 (y=1) + 0.6+0.2+0.4 (y=2) = 3.18
        // Layer y=1 has 2 cells, layer y=2 has 3 cells
        // Fill y=1 first: 2 cells * 1.0 = 2.0, remaining = 1.18
        // Fill y=2: 1.18 / 3 = ~0.3933
        assert!((grid.get(1, 1, 0).level - 1.0).abs() < 0.01);
        assert!((grid.get(2, 1, 0).level - 1.0).abs() < 0.01);
        let surface_level = grid.get(0, 2, 0).level;
        let expected_surface = 1.18 / 3.0;
        assert!((surface_level - expected_surface).abs() < 0.01);
        assert!((grid.get(1, 2, 0).level - surface_level).abs() < 0.01);
        assert!((grid.get(2, 2, 0).level - surface_level).abs() < 0.01);
    }

    #[test]
    fn source_stays_at_full() {
        // Source cell in pool should stay at 1.0, not be reduced by equalization.
        // Enclosed by walls so edge expansion doesn't drain non-source cells.
        let size = 16;
        let chunk_key = (0, 0, 0);
        let mut grid = make_air_chunk(size);

        set_solid_floor(&mut grid, 0, size);

        // Walls at y=1 to contain the pool
        grid.set_density(3, 1, 0, 1.0); // wall to the right
        for x in 0..4 {
            grid.set_density(x, 1, 1, 1.0); // wall behind
        }

        // Source at x=0
        let cell = grid.get_mut(0, 1, 0);
        cell.level = SOURCE_LEVEL;
        cell.fluid_type = FluidType::Water;

        // Non-source neighbors at x=1,2 with less water
        for x in 1..3 {
            let cell = grid.get_mut(x, 1, 0);
            cell.level = 0.3;
            cell.fluid_type = FluidType::Water;
        }
        grid.has_fluid = true;

        let mut chunks = HashMap::new();
        chunks.insert(chunk_key, grid);
        let densities = HashMap::new();

        detect_and_equalize_pools(&mut chunks, &densities, size, false);

        let grid = chunks.get(&chunk_key).unwrap();
        // Source stays at SOURCE_LEVEL (edge expansion skips sources)
        assert_eq!(grid.get(0, 1, 0).level, SOURCE_LEVEL);
        // Non-source cells equalize: total non-source = 0.6, cells = 2 → 0.3 each
        let expected = 0.3;
        assert!((grid.get(1, 1, 0).level - expected).abs() < 0.01);
        assert!((grid.get(2, 1, 0).level - expected).abs() < 0.01);
    }

    #[test]
    fn cross_chunk_pool() {
        // Pool spanning two chunks: chunk (0,0,0) x=15 and chunk (1,0,0) x=0.
        // Walled off with solid at y=1 to prevent edge expansion from draining volume.
        let size = 16;
        let key_a = (0, 0, 0);
        let key_b = (1, 0, 0);

        let mut grid_a = make_air_chunk(size);
        let mut grid_b = make_air_chunk(size);

        // Solid floor at y=0 in both
        set_solid_floor(&mut grid_a, 0, size);
        set_solid_floor(&mut grid_b, 0, size);

        // Walls at y=1 to contain pool cells
        grid_a.set_density(14, 1, 0, 1.0); // wall left of A's water cell
        grid_a.set_density(15, 1, 1, 1.0); // wall behind A's water cell
        grid_b.set_density(1, 1, 0, 1.0);  // wall right of B's water cell
        grid_b.set_density(0, 1, 1, 1.0);  // wall behind B's water cell

        // Water at boundary: chunk A x=15, chunk B x=0, both at y=1, z=0
        grid_a.get_mut(15, 1, 0).level = 0.8;
        grid_a.get_mut(15, 1, 0).fluid_type = FluidType::Water;
        grid_a.has_fluid = true;

        grid_b.get_mut(0, 1, 0).level = 0.4;
        grid_b.get_mut(0, 1, 0).fluid_type = FluidType::Water;
        grid_b.has_fluid = true;

        let mut chunks = HashMap::new();
        chunks.insert(key_a, grid_a);
        chunks.insert(key_b, grid_b);
        let densities = HashMap::new();

        let dirty = detect_and_equalize_pools(&mut chunks, &densities, size, false);
        assert!(!dirty.is_empty());

        // Both should equalize to (0.8 + 0.4) / 2 = 0.6
        let expected = 0.6;
        assert!(
            (chunks.get(&key_a).unwrap().get(15, 1, 0).level - expected).abs() < 0.01,
            "Chunk A level: {}",
            chunks.get(&key_a).unwrap().get(15, 1, 0).level
        );
        assert!(
            (chunks.get(&key_b).unwrap().get(0, 1, 0).level - expected).abs() < 0.01,
            "Chunk B level: {}",
            chunks.get(&key_b).unwrap().get(0, 1, 0).level
        );
    }

    #[test]
    fn single_cell_not_a_pool() {
        // A single isolated water cell should NOT be equalized (needs 2+ cells).
        let size = 16;
        let chunk_key = (0, 0, 0);
        let mut grid = make_air_chunk(size);

        set_solid_floor(&mut grid, 0, size);

        let cell = grid.get_mut(5, 1, 5);
        cell.level = 0.5;
        cell.fluid_type = FluidType::Water;
        grid.has_fluid = true;

        let mut chunks = HashMap::new();
        chunks.insert(chunk_key, grid);
        let densities = HashMap::new();

        let dirty = detect_and_equalize_pools(&mut chunks, &densities, size, false);
        assert!(dirty.is_empty());

        // Level unchanged
        assert!((chunks.get(&chunk_key).unwrap().get(5, 1, 5).level - 0.5).abs() < 0.01);
    }

    #[test]
    fn lava_pool_on_lava_tick() {
        // Lava pool should only equalize during lava ticks.
        // Enclosed by walls to prevent edge expansion.
        let size = 16;
        let chunk_key = (0, 0, 0);
        let mut grid = make_air_chunk(size);

        set_solid_floor(&mut grid, 0, size);

        // Walls at y=1 to contain the pool
        grid.set_density(3, 1, 0, 1.0); // wall to the right
        for x in 0..4 {
            grid.set_density(x, 1, 1, 1.0); // wall behind
        }

        for x in 0..3 {
            let cell = grid.get_mut(x, 1, 0);
            cell.level = if x == 0 { 0.9 } else { 0.3 };
            cell.fluid_type = FluidType::Lava;
        }
        grid.has_fluid = true;

        let mut chunks = HashMap::new();
        chunks.insert(chunk_key, grid);
        let densities = HashMap::new();

        // Water tick: should not touch lava
        let dirty = detect_and_equalize_pools(&mut chunks, &densities, size, false);
        assert!(dirty.is_empty());

        // Lava tick: should equalize
        let dirty = detect_and_equalize_pools(&mut chunks, &densities, size, true);
        assert!(!dirty.is_empty());

        let expected = (0.9 + 0.3 + 0.3) / 3.0;
        let grid = chunks.get(&chunk_key).unwrap();
        for x in 0..3 {
            assert!(
                (grid.get(x, 1, 0).level - expected).abs() < 0.01,
                "Lava cell {} has level {} but expected {}",
                x,
                grid.get(x, 1, 0).level,
                expected
            );
        }
    }

    #[test]
    fn pool_expands_to_basin_edge() {
        // Pool on flat floor with a dry neighbor cell at basin edge.
        // After equalization + edge expansion, water should seed the edge cell.
        let size = 16;
        let chunk_key = (0, 0, 0);
        let mut grid = make_air_chunk(size);

        // Solid floor at y=0
        set_solid_floor(&mut grid, 0, size);

        // Pool: 3 cells at y=1, x=0..3, z=0 with water
        for x in 0..3 {
            let cell = grid.get_mut(x, 1, 0);
            cell.level = 0.5;
            cell.fluid_type = FluidType::Water;
        }
        grid.has_fluid = true;

        // x=3, y=1, z=0 is dry but has solid floor below — expansion candidate

        let mut chunks = HashMap::new();
        chunks.insert(chunk_key, grid);
        let densities = HashMap::new();

        let dirty = detect_and_equalize_pools(&mut chunks, &densities, size, false);
        assert!(!dirty.is_empty());

        // The dry neighbor at x=3 should have been seeded with a small amount of water
        let grid = chunks.get(&chunk_key).unwrap();
        let edge_level = grid.get(3, 1, 0).level;
        assert!(
            edge_level > MIN_LEVEL,
            "Pool should expand to basin edge cell (3,1,0), got {}",
            edge_level
        );
        // The seeded amount should be small (≤ 0.05)
        assert!(
            edge_level <= 0.06,
            "Edge expansion seed should be small, got {}",
            edge_level
        );
    }
}
