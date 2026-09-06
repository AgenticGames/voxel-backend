use std::collections::HashMap;
use glam::Vec3;
use voxel_core::dual_contouring::qef::QefData;
use crate::cell::{ChunkFluidGrid, FluidType, MIN_LEVEL};
use crate::tables::{CORNER_OFFSETS, EDGE_TABLE, EDGE_VERTICES, TRI_TABLE};
use crate::FluidConfig;

/// Isosurface threshold for fluid meshing.
const ISO_LEVEL: f32 = 0.15;
/// How far past the terrain's density zero-crossing (in cell units) the fluid
/// rim is pushed INTO rock on mixed solid/non-solid edges. The terrain mesh
/// then occludes the seam — without this the rim lands at a fixed t=0.15 from
/// the air point and can stop up to 0.85 cells short of the DC wall (visible
/// gap slits around pool rims). 0.1 cells = 4 UE units at world scale 40.
const ROCK_RECESS_T: f32 = 0.1;
/// Recess used instead of ROCK_RECESS_T on floor-contact rim edges when the
/// mesh_floor_clamp flag is on: just enough burial to avoid z-fighting with
/// the terrain surface, shallow enough that the sheet hugs the floor instead
/// of diving under it (0.02 cells ≈ 0.6-0.8 UE units at world scale 30-40).
const FLOOR_RECESS_T: f32 = 0.02;

/// The rim/skirt fix flags, extracted from FluidConfig so the inner MC
/// functions don't need the whole config. See FluidConfig for what each does.
#[derive(Clone, Copy)]
pub struct MeshFlags {
    /// Mesher half of mesh_sticky_release: plain (non-special-rule) cells
    /// must hold mesh_level >= ISO_LEVEL to qualify a cube, not merely
    /// MIN_LEVEL. Sub-iso remnants adjacent to rock otherwise form crossings
    /// against the solid corners' constant 1.0 sample and paint rock-hugging
    /// phantom geometry (the raised rim ring) no matter what the hysteresis
    /// does. Hysteresis-held cells report mesh_level at iso+0.01, so the
    /// anti-strobe hold is unaffected.
    pub sticky_release: bool,
    pub floor_clamp: bool,
    pub buried_cull: bool,
}

impl MeshFlags {
    /// Pre-bundle behavior (all fixes off) — used by tests that assert the
    /// legacy geometry.
    pub fn legacy() -> Self {
        Self { sticky_release: false, floor_clamp: false, buried_cull: false }
    }

    pub fn from_config(config: &FluidConfig) -> Self {
        Self {
            sticky_release: config.mesh_sticky_release,
            floor_clamp: config.mesh_floor_clamp,
            buried_cull: config.mesh_buried_cull,
        }
    }
}
/// Tiny SDF value for out-of-bounds samples — places boundary faces near chunk edge.
const BOUNDARY_SDF: f32 = 0.001;
/// Field value for out-of-bounds samples — just below ISO_LEVEL so MC places face at edge.
const BOUNDARY_FIELD: f32 = ISO_LEVEL - BOUNDARY_SDF;
/// Fluid levels from neighboring chunks at the 3 positive boundary faces.
/// Used to create seamless mesh at chunk edges instead of sealing the isosurface.
pub struct BoundaryLevels {
    /// Neighbor's x=0 face fluid levels, indexed [z * size + y]. size*size values.
    pub pos_x: Option<Vec<f32>>,
    /// Neighbor's y=0 face fluid levels, indexed [z * size + x]. size*size values.
    pub pos_y: Option<Vec<f32>>,
    /// Neighbor's z=0 face fluid levels, indexed [y * size + x]. size*size values.
    pub pos_z: Option<Vec<f32>>,
    /// Whether the BELOW neighbor's top cell can hold fluid (capacity > 0),
    /// indexed [z * size + x]. Falls keep falling through an open seam — the
    /// y==0 floor-extension boost must not draw a lid there. None = no below
    /// neighbor known: treat the chunk bottom as a world floor (legacy).
    pub neg_y_open: Option<Vec<bool>>,
    pub size: usize,
}

impl BoundaryLevels {
    /// Create empty boundary levels (no neighbor data — mesh seals at edges).
    pub fn empty(size: usize) -> Self {
        Self {
            pos_x: None,
            pos_y: None,
            pos_z: None,
            neg_y_open: None,
            size,
        }
    }

    /// Is the cell directly below chunk-local (x, -1, z) open fluid space?
    /// False when unknown — unknown must behave like the legacy world floor.
    #[inline]
    pub fn below_open(&self, x: usize, z: usize) -> bool {
        self.neg_y_open
            .as_ref()
            .map_or(false, |v| v[z * self.size + x])
    }

    /// Get the fluid level at an out-of-bounds coordinate from neighbor data.
    /// Returns None for multi-axis overflow or if no neighbor data exists.
    pub fn get_level(&self, x: usize, y: usize, z: usize) -> Option<f32> {
        let size = self.size;
        let x_over = x >= size;
        let y_over = y >= size;
        let z_over = z >= size;

        // Multi-axis overflow or no overflow — no data
        if (x_over as u8 + y_over as u8 + z_over as u8) != 1 {
            return None;
        }

        if x_over {
            // x == size: look up pos_x face at [z * size + y]
            self.pos_x.as_ref().map(|v| v[z * size + y])
        } else if y_over {
            // y == size: look up pos_y face at [z * size + x]
            self.pos_y.as_ref().map(|v| v[z * size + x])
        } else {
            // z == size: look up pos_z face at [y * size + x]
            self.pos_z.as_ref().map(|v| v[y * size + x])
        }
    }
}

/// A fluid mesh produced by Marching Cubes isosurface extraction.
pub struct FluidMeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub fluid_types: Vec<u8>,
    pub indices: Vec<u32>,
    pub uvs: Vec<[f32; 2]>,
    pub flow_directions: Vec<[f32; 3]>, // (dx, dz, magnitude) for UV scroll
    /// Per-vertex whitewash 0..1 (collapse waves) -> UE vertex colour alpha.
    pub foam: Vec<f32>,
}

/// Build fluid isosurface mesh via Marching Cubes.
///
/// Extracts the isosurface at ISO_LEVEL and passes through the actual geological
/// FluidType from each cell so UE can render distinct colors per water source.
pub fn mesh_fluid(grid: &ChunkFluidGrid, boundary: &BoundaryLevels, config: &FluidConfig) -> FluidMeshData {
    let size = grid.size;
    if size < 2 {
        return FluidMeshData {
            positions: Vec::new(),
            normals: Vec::new(),
            fluid_types: Vec::new(),
            indices: Vec::new(),
            uvs: Vec::new(),
            flow_directions: Vec::new(),
            foam: Vec::new(),
        };
    }

    let flags = MeshFlags::from_config(config);
    let mut mesh = mesh_fluid_mc(grid, boundary, flags);
    weld_vertices(&mut mesh);
    // Vertices at/inside the terrain surface are rim-contact vertices placed by
    // the rock-crossing override in mesh_fluid_mc. QEF and smoothing must not
    // move them or the seam re-opens.
    let mut pinned = compute_rock_pins(&mesh, grid);
    if flags.buried_cull {
        cull_buried_triangles(&mut mesh, &mut pinned);
    }
    if config.mesh_qef_refinement {
        qef_refine_vertices(&mut mesh, size, &pinned);
    }
    if config.mesh_smooth_iterations > 0 {
        smooth_fluid_mesh(&mut mesh, config.mesh_smooth_iterations, config.mesh_smooth_strength, size, &pinned);
    }
    if config.mesh_recalc_normals {
        recalculate_fluid_normals(&mut mesh);
    }
    mesh
}

/// Check if the cell above (x,y+1,z) is non-solid and has real fluid.
/// Used to detect floor-adjacent cells that should be extended.
#[inline]
fn has_fluid_above(grid: &ChunkFluidGrid, x: usize, y: usize, z: usize) -> bool {
    let size = grid.size;
    if y + 1 >= size {
        return false;
    }
    grid.grid_point_density(x, y + 1, z) <= 0.0 && grid.mesh_level(x, y + 1, z) >= ISO_LEVEL
}

/// Sample the scalar field for MC meshing.
/// Returns fluid level for air cells. Solid cells return 1.0 (treated as "inside")
/// so no isosurface forms at rock/fluid boundaries — only at fluid/air boundaries.
/// Floor extension: non-solid cells with low fluid sitting on solid rock (or at chunk
/// bottom boundary) with fluid above get boosted to 1.0 to close the visual gap.
/// Out-of-bounds coordinates return BOUNDARY_FIELD to close mesh at chunk edges.
#[inline]
pub(crate) fn sample_field(grid: &ChunkFluidGrid, x: usize, y: usize, z: usize, boundary: &BoundaryLevels) -> f32 {
    let size = grid.size;
    if x >= size || y >= size || z >= size {
        // Density at boundary coords is valid via grid_point_density (handles up to size)
        if grid.grid_point_density(x, y, z) > 0.0 {
            return 1.0;
        }
        // Try neighbor fluid level
        if let Some(level) = boundary.get_level(x, y, z) {
            return level;
        }
        return BOUNDARY_FIELD;
    }
    if grid.grid_point_density(x, y, z) > 0.0 {
        1.0 // inside — prevents surface at rock/fluid boundary
    } else {
        let level = grid.mesh_level(x, y, z);
        // Floor extension: low-fluid cell on solid rock (or a genuinely
        // closed chunk bottom) with fluid above. At y==0 the "floor" is only
        // real when the chunk below is solid/unknown — an open seam must NOT
        // get a lid (falls read a flat cap at every vertical chunk boundary).
        let on_floor = if y == 0 {
            !boundary.below_open(x, z)
        } else {
            grid.grid_point_density(x, y - 1, z) > 0.0
        };
        if level < ISO_LEVEL && on_floor && has_fluid_above(grid, x, y, z) {
            1.0 // boost to close floor gap
        } else {
            level
        }
    }
}

/// Returns true if any corner of the unit cube at (x,y,z) has real fluid:
/// non-solid corners with level >= MIN_LEVEL, floor-extension cells (non-solid,
/// low fluid, on solid rock with fluid above), or SOLID corners whose cell holds
/// level >= ISO_LEVEL (water lapping over barely-submerged rock). Used to skip
/// cubes in pure rock/air regions — without this, treating solid as "inside"
/// would generate phantom water surfaces on every cave wall.
/// Out-of-bounds corners use neighbor boundary levels when available.
#[inline]
fn cube_has_fluid(grid: &ChunkFluidGrid, x: usize, y: usize, z: usize, boundary: &BoundaryLevels, flags: MeshFlags) -> bool {
    let size = grid.size;
    // With sticky_release: a plain cell only qualifies at iso — sub-iso
    // dribbles can't force cubes whose only >=iso corners are solid rock
    // (constant 1.0), which paint phantom rock-hugging geometry. All the
    // special rules below (solid-corner lapping, floor extension) keep their
    // own iso-based thresholds.
    let min_qual = if flags.sticky_release { ISO_LEVEL } else { MIN_LEVEL };
    for dz in 0..=1usize {
        for dy in 0..=1usize {
            for dx in 0..=1usize {
                let cx = x + dx;
                let cy = y + dy;
                let cz = z + dz;
                if cx >= size || cy >= size || cz >= size {
                    // Multi-axis overflow: skip
                    let overflows = (cx >= size) as u8 + (cy >= size) as u8 + (cz >= size) as u8;
                    if overflows > 1 {
                        continue;
                    }
                    // Single-axis: check density + boundary level
                    if grid.grid_point_density(cx, cy, cz) > 0.0 {
                        // Solid corner whose neighbor cell still holds real
                        // fluid (lapping/rim) — see in-bounds case below.
                        if let Some(level) = boundary.get_level(cx, cy, cz) {
                            if level >= ISO_LEVEL {
                                return true;
                            }
                        }
                    } else if let Some(level) = boundary.get_level(cx, cy, cz) {
                        if level >= min_qual {
                            return true;
                        }
                    }
                    continue;
                }
                if grid.grid_point_density(cx, cy, cz) > 0.0 {
                    // Solid lattice point whose CELL still holds real water:
                    // water lapping over barely-submerged rock (shorelines, pool
                    // rim cells straddling the basin wall). Without this the
                    // cube is skipped, the sheet is cut a cell early, and its
                    // open edge floats in mid-air over the terrain.
                    if grid.mesh_level(cx, cy, cz) >= ISO_LEVEL {
                        return true;
                    }
                    continue;
                }
                {
                    let level = grid.mesh_level(cx, cy, cz);
                    if level >= min_qual {
                        return true;
                    }
                    // Floor extension — below-aware at cy==0, same rule as
                    // sample_field (open seam ≠ floor).
                    let on_floor = if cy == 0 {
                        !boundary.below_open(cx, cz)
                    } else {
                        grid.grid_point_density(cx, cy - 1, cz) > 0.0
                    };
                    if level < ISO_LEVEL && on_floor && has_fluid_above(grid, cx, cy, cz) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Fluid level of the cell at lattice coords, falling back to neighbor-chunk
/// boundary data for single-axis out-of-bounds coords. 0.0 when unknown.
#[inline]
fn cell_level_at(grid: &ChunkFluidGrid, boundary: &BoundaryLevels, cx: usize, cy: usize, cz: usize) -> f32 {
    let size = grid.size;
    if cx < size && cy < size && cz < size {
        grid.mesh_level(cx, cy, cz)
    } else {
        boundary.get_level(cx, cy, cz).unwrap_or(0.0)
    }
}

/// Compute the interpolation parameter t for an MC edge crossing.
///
/// Three cases:
/// 1. Both endpoints on the same side of the rock surface — plain fluid-field
///    interpolation (top surface, floor extension, chunk-edge sealing).
/// 2. Mixed rock/non-rock edge (the pool rim seam): place the crossing at the
///    terrain's own density zero-crossing — the same linear estimate dual
///    contouring uses for the terrain mesh — recessed ROCK_RECESS_T into the
///    rock so the terrain mesh occludes the seam. The fluid field can't do
///    this itself: solid points sample a constant 1.0, which lands the rim at
///    a fixed t=0.15 from the air point regardless of where the wall actually
///    is (up to 0.85 cells of visible gap).
/// 3. Lapping edge — a vertical mixed edge whose solid end is BELOW and whose
///    cell still holds real fluid: water sitting on top of barely-submerged
///    rock (shorelines). The visible surface there is the water level, not the
///    rock face, so interpolate from the cell's level instead of snapping down
///    to the rock and denting the surface.
#[inline]
#[allow(clippy::too_many_arguments)]
fn edge_crossing_t(
    grid: &ChunkFluidGrid,
    boundary: &BoundaryLevels,
    flags: MeshFlags,
    x: usize,
    y: usize,
    z: usize,
    c0: usize,
    c1: usize,
    v0: f32,
    v1: f32,
    d0: f32,
    d1: f32,
) -> f32 {
    let solid0 = d0 > 0.0;
    let solid1 = d1 > 0.0;

    if solid0 != solid1 {
        let (sc, ac) = if solid0 { (c0, c1) } else { (c1, c0) };
        let off_s = CORNER_OFFSETS[sc];
        let off_a = CORNER_OFFSETS[ac];
        let vertical = off_s[0] == off_a[0] && off_s[2] == off_a[2];
        let floor_contact = vertical && off_s[1] < off_a[1];

        if floor_contact {
            let level = cell_level_at(grid, boundary, x + off_s[0], y + off_s[1], z + off_s[2]);
            if level >= ISO_LEVEL {
                // Lapping: solid-below, dry-above, but the solid point's cell
                // holds water — surface at the cell's fluid level.
                let (lv0, lv1) = if solid0 { (level, v1) } else { (v0, level) };
                let mut t = if (lv1 - lv0).abs() > 1e-6 {
                    ((ISO_LEVEL - lv0) / (lv1 - lv0)).clamp(0.0, 1.0)
                } else {
                    0.5
                };
                // Floor clamp: a barely-at-iso cell puts the "water level"
                // crossing essentially AT the solid lattice point, which on
                // a sloped floor sits up to a full cell inside rock — the
                // deepest under-floor verts in the skirt come from here, not
                // from the recess. Keep the lapping surface at/above the
                // terrain line (whisker below at most). Genuine lapping
                // (water level above the rock line) is untouched.
                if flags.floor_clamp {
                    let denom = d1 - d0;
                    if denom.abs() > 1e-6 {
                        let t_rock = (-d0 / denom).clamp(0.0, 1.0);
                        if solid0 {
                            t = t.max(t_rock - FLOOR_RECESS_T);
                        } else {
                            t = t.min(t_rock + FLOOR_RECESS_T);
                        }
                    }
                }
                return t;
            }
        }

        // Rim seam: terrain density zero-crossing, recessed into the rock.
        // Floor-contact edges (solid-below) get only a whisker of recess when
        // mesh_floor_clamp is on — the full recess is for WALLS, where the
        // terrain occludes the burial; on a floor it puts the sheet under the
        // ground the player is standing on.
        let denom = d1 - d0;
        if denom.abs() > 1e-6 {
            let t_rock = -d0 / denom;
            let mag = if flags.floor_clamp && floor_contact { FLOOR_RECESS_T } else { ROCK_RECESS_T };
            let recess = if solid1 { mag } else { -mag };
            return (t_rock + recess).clamp(0.0, 1.0);
        }
        return 0.5;
    }

    if (v1 - v0).abs() > 1e-6 {
        ((ISO_LEVEL - v0) / (v1 - v0)).clamp(0.0, 1.0)
    } else {
        0.5
    }
}

/// Trilinear terrain density at an arbitrary position in chunk-local lattice
/// coordinates (the space mesh vertices live in, [0, size]^3).
fn terrain_density_at(grid: &ChunkFluidGrid, p: [f32; 3]) -> f32 {
    let max_base = (grid.size - 1) as f32;
    let bx = p[0].floor().clamp(0.0, max_base);
    let by = p[1].floor().clamp(0.0, max_base);
    let bz = p[2].floor().clamp(0.0, max_base);
    let fx = (p[0] - bx).clamp(0.0, 1.0);
    let fy = (p[1] - by).clamp(0.0, 1.0);
    let fz = (p[2] - bz).clamp(0.0, 1.0);
    let (bx, by, bz) = (bx as usize, by as usize, bz as usize);

    let mut result = 0.0f32;
    for dz in 0..=1usize {
        for dy in 0..=1usize {
            for dx in 0..=1usize {
                let w = (if dx == 1 { fx } else { 1.0 - fx })
                    * (if dy == 1 { fy } else { 1.0 - fy })
                    * (if dz == 1 { fz } else { 1.0 - fz });
                if w > 0.0 {
                    result += w * grid.grid_point_density(bx + dx, by + dy, bz + dz);
                }
            }
        }
    }
    result
}

/// Flag vertices that sit at or inside the terrain surface — the rim-contact
/// vertices placed by the rock-crossing override (plus anything else already
/// buried in rock). QEF refinement and Laplacian smoothing must leave these in
/// place: smoothing in particular shrinks open boundaries and would drag the
/// rim back out of the wall, re-opening the seam.
fn compute_rock_pins(mesh: &FluidMeshData, grid: &ChunkFluidGrid) -> Vec<bool> {
    mesh.positions
        .iter()
        .map(|p| terrain_density_at(grid, *p) >= 0.0)
        .collect()
}

/// Marching Cubes fluid mesher.
/// Produces triangulated isosurface at ISO_LEVEL, passing through the actual
/// geological FluidType from each cell via dominant_fluid_type().
fn mesh_fluid_mc(grid: &ChunkFluidGrid, boundary: &BoundaryLevels, flags: MeshFlags) -> FluidMeshData {
    let size = grid.size;
    let mut mesh = FluidMeshData {
        positions: Vec::new(),
        normals: Vec::new(),
        fluid_types: Vec::new(),
        indices: Vec::new(),
        uvs: Vec::new(),
        flow_directions: Vec::new(),
        foam: Vec::new(),
    };

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                // Skip cubes with no actual fluid — prevents phantom surfaces on cave walls
                if !cube_has_fluid(grid, x, y, z, boundary, flags) {
                    continue;
                }

                // Sample 8 corners using Paul Bourke ordering (CORNER_OFFSETS)
                let mut corner_vals = [0.0f32; 8];
                let mut corner_density = [0.0f32; 8];
                for (i, off) in CORNER_OFFSETS.iter().enumerate() {
                    corner_vals[i] = sample_field(grid, x + off[0], y + off[1], z + off[2], boundary);
                    corner_density[i] = grid.grid_point_density(x + off[0], y + off[1], z + off[2]);
                }

                // Build cube index: bit i set if corner i >= ISO_LEVEL
                let mut cube_index: usize = 0;
                for i in 0..8 {
                    if corner_vals[i] >= ISO_LEVEL {
                        cube_index |= 1 << i;
                    }
                }

                let edge_mask = EDGE_TABLE[cube_index];
                if edge_mask == 0 {
                    continue;
                }

                // Interpolate edge vertices at ISO_LEVEL crossings
                let mut edge_verts = [[0.0f32; 3]; 12];
                for e in 0..12 {
                    if edge_mask & (1 << e) != 0 {
                        let [c0, c1] = EDGE_VERTICES[e];
                        let v0 = corner_vals[c0];
                        let v1 = corner_vals[c1];
                        let t = edge_crossing_t(
                            grid, boundary, flags, x, y, z,
                            c0, c1, v0, v1,
                            corner_density[c0], corner_density[c1],
                        );
                        let p0 = CORNER_OFFSETS[c0];
                        let p1 = CORNER_OFFSETS[c1];
                        edge_verts[e] = [
                            x as f32 + p0[0] as f32 + (p1[0] as f32 - p0[0] as f32) * t,
                            y as f32 + p0[1] as f32 + (p1[1] as f32 - p0[1] as f32) * t,
                            z as f32 + p0[2] as f32 + (p1[2] as f32 - p0[2] as f32) * t,
                        ];
                    }
                }

                // Emit triangles from TRI_TABLE
                let tri_row = &TRI_TABLE[cube_index];
                let flow = compute_flow_direction(grid, x, y, z);
                let foam = grid.foam_at(x, y, z);
                let ft = dominant_fluid_type(grid, x, y, z) as u8;
                let mut i = 0;
                while i < 15 && tri_row[i] >= 0 {
                    let v0 = edge_verts[tri_row[i] as usize];
                    let v1 = edge_verts[tri_row[i + 1] as usize];
                    let v2 = edge_verts[tri_row[i + 2] as usize];

                    // Cross-product normal, skip degenerate triangles
                    let ax = v1[0] - v0[0];
                    let ay = v1[1] - v0[1];
                    let az = v1[2] - v0[2];
                    let bx = v2[0] - v0[0];
                    let by = v2[1] - v0[1];
                    let bz = v2[2] - v0[2];
                    let nx = ay * bz - az * by;
                    let ny = az * bx - ax * bz;
                    let nz = ax * by - ay * bx;
                    let len = (nx * nx + ny * ny + nz * nz).sqrt();

                    if len >= 1e-4 {
                        let normal = [nx / len, ny / len, nz / len];
                        let base = mesh.positions.len() as u32;

                        for &v in &[v0, v1, v2] {
                            mesh.positions.push(v);
                            mesh.normals.push(normal);
                            mesh.fluid_types.push(ft);
                            mesh.uvs.push([v[0], v[2]]);
                            mesh.flow_directions.push(flow);
                            mesh.foam.push(foam);
                        }

                        mesh.indices.push(base);
                        mesh.indices.push(base + 1);
                        mesh.indices.push(base + 2);
                    }

                    i += 3;
                }
            }
        }
    }

    mesh
}

/// Drop triangles whose three vertices are all at/inside the terrain surface
/// (the `pinned` predicate). These form the buried closure skirt — the sheet
/// the rim-sealing system tucks under floors and into walls so the terrain
/// occludes the seam. It is only ever visible from inside the ground or
/// through terrain cracks, and under a pool floor it reads as fluid "below
/// the world". Triangles with at least one above-terrain vertex are kept, so
/// the visible waterline still terminates in a pinned rock contact.
/// Compacts the vertex arrays (and the pinned flags) after filtering.
fn cull_buried_triangles(mesh: &mut FluidMeshData, pinned: &mut Vec<bool>) {
    if mesh.indices.is_empty() {
        return;
    }

    let mut new_indices: Vec<u32> = Vec::with_capacity(mesh.indices.len());
    for t in 0..mesh.indices.len() / 3 {
        let i0 = mesh.indices[t * 3] as usize;
        let i1 = mesh.indices[t * 3 + 1] as usize;
        let i2 = mesh.indices[t * 3 + 2] as usize;
        if pinned[i0] && pinned[i1] && pinned[i2] {
            continue;
        }
        new_indices.push(i0 as u32);
        new_indices.push(i1 as u32);
        new_indices.push(i2 as u32);
    }
    if new_indices.len() == mesh.indices.len() {
        return;
    }

    // Compact: drop vertices no longer referenced by any surviving triangle.
    let mut remap: Vec<u32> = vec![u32::MAX; mesh.positions.len()];
    let mut kept = 0u32;
    for &idx in &new_indices {
        if remap[idx as usize] == u32::MAX {
            remap[idx as usize] = kept;
            kept += 1;
        }
    }
    let kept = kept as usize;
    let mut positions = Vec::with_capacity(kept);
    let mut normals = Vec::with_capacity(kept);
    let mut fluid_types = Vec::with_capacity(kept);
    let mut uvs = Vec::with_capacity(kept);
    let mut flow_directions = Vec::with_capacity(kept);
    let mut foam = Vec::with_capacity(kept);
    let mut new_pinned = Vec::with_capacity(kept);
    // remap assigns new ids in first-reference order; rebuild arrays in that order
    let mut order: Vec<(u32, usize)> = remap
        .iter()
        .enumerate()
        .filter(|&(_, &m)| m != u32::MAX)
        .map(|(old, &m)| (m, old))
        .collect();
    order.sort_unstable_by_key(|&(m, _)| m);
    for &(_, old) in &order {
        positions.push(mesh.positions[old]);
        normals.push(mesh.normals[old]);
        fluid_types.push(mesh.fluid_types[old]);
        uvs.push(mesh.uvs[old]);
        flow_directions.push(mesh.flow_directions[old]);
        foam.push(mesh.foam[old]);
        new_pinned.push(pinned[old]);
    }
    for idx in &mut new_indices {
        *idx = remap[*idx as usize];
    }

    mesh.positions = positions;
    mesh.normals = normals;
    mesh.fluid_types = fluid_types;
    mesh.uvs = uvs;
    mesh.flow_directions = flow_directions;
    mesh.foam = foam;
    mesh.indices = new_indices;
    *pinned = new_pinned;
}

/// Neighbor-cell offsets to probe along one axis during vertex welding.
///
/// Always includes `0`; includes `-1` only when the query vertex lies within
/// `margin` of the cell's lower face and `+1` only when within `margin` of the
/// upper face, emitting them in ascending order (`-1, 0, +1`) so the welding
/// search visits cells in the same order as the old fixed `-1..=1` triple loop.
/// Returns the offsets plus the count of valid entries.
#[inline]
fn axis_deltas(dist_lower: f32, cell_size: f32, margin: f32) -> ([i32; 3], usize) {
    let mut deltas = [0i32; 3];
    let mut n = 0;
    if dist_lower < margin {
        deltas[n] = -1;
        n += 1;
    }
    deltas[n] = 0;
    n += 1;
    if cell_size - dist_lower < margin {
        deltas[n] = 1;
        n += 1;
    }
    (deltas, n)
}

/// Weld coincident vertices (MC emits 3 fresh vertices per triangle with no sharing).
/// Uses spatial hashing (grid cell 0.01) to find coincident positions within epsilon=1e-5,
/// remaps indices, and compacts all parallel arrays.
///
/// A vertex is only ever inserted into its own home cell, so a neighbor cell can
/// hold a within-epsilon match only when the query vertex is within `epsilon` of
/// the face shared with that cell. The per-axis neighbor probe is therefore gated
/// on the vertex's distance to each cell face (see `axis_deltas`): cells beyond
/// the gate provably contain no point within epsilon, so skipping them cannot
/// change which vertex is welded — the result is bit-identical to the full
/// 27-cell scan. The typical interior vertex (>1e-3 from every face) now does a
/// single home-cell probe instead of 27.
fn weld_vertices(mesh: &mut FluidMeshData) {
    if mesh.positions.is_empty() {
        return;
    }

    let cell_size: f32 = 0.01;
    let epsilon: f32 = 1e-5;
    let inv_cell = 1.0 / cell_size;
    // A neighbor cell is probed only when the vertex lies within `margin` of the
    // face shared with it. `margin` (1e-3 = 100x epsilon) sits far above both
    // epsilon and the worst-case f32 rounding error (~1e-5) in the face-distance
    // computation, so a live cell is never wrongly skipped; over-including a cell
    // is harmless because the exact per-vertex distance test below is unchanged.
    let margin: f32 = 1e-3;

    // Spatial hash: grid cell -> list of (new_index, position)
    let mut spatial: HashMap<(i32, i32, i32), Vec<(u32, [f32; 3])>> = HashMap::new();
    let mut remap: Vec<u32> = Vec::with_capacity(mesh.positions.len());
    let mut new_positions: Vec<[f32; 3]> = Vec::new();
    let mut new_normals: Vec<[f32; 3]> = Vec::new();
    let mut new_fluid_types: Vec<u8> = Vec::new();
    let mut new_uvs: Vec<[f32; 2]> = Vec::new();
    let mut new_flow_directions: Vec<[f32; 3]> = Vec::new();
    let mut new_foam: Vec<f32> = Vec::new();

    for i in 0..mesh.positions.len() {
        let pos = mesh.positions[i];
        let gx = (pos[0] * inv_cell).floor() as i32;
        let gy = (pos[1] * inv_cell).floor() as i32;
        let gz = (pos[2] * inv_cell).floor() as i32;

        // Probe the home cell plus only the neighbor cells whose shared face the
        // vertex is within `margin` of (the rest provably hold no match). Same
        // visit order as the old fixed -1..=1 triple loop -> bit-identical result.
        let (xds, xn) = axis_deltas(pos[0] - gx as f32 * cell_size, cell_size, margin);
        let (yds, yn) = axis_deltas(pos[1] - gy as f32 * cell_size, cell_size, margin);
        let (zds, zn) = axis_deltas(pos[2] - gz as f32 * cell_size, cell_size, margin);

        let mut found = None;
        'search: for &dz in &zds[..zn] {
            for &dy in &yds[..yn] {
                for &dx in &xds[..xn] {
                    let key = (gx + dx, gy + dy, gz + dz);
                    if let Some(bucket) = spatial.get(&key) {
                        for &(idx, ref p) in bucket {
                            let d0 = pos[0] - p[0];
                            let d1 = pos[1] - p[1];
                            let d2 = pos[2] - p[2];
                            if d0 * d0 + d1 * d1 + d2 * d2 < epsilon * epsilon {
                                found = Some(idx);
                                break 'search;
                            }
                        }
                    }
                }
            }
        }

        let new_idx = if let Some(idx) = found {
            idx
        } else {
            let idx = new_positions.len() as u32;
            new_positions.push(pos);
            new_normals.push(mesh.normals[i]);
            new_fluid_types.push(mesh.fluid_types[i]);
            new_uvs.push(mesh.uvs[i]);
            new_flow_directions.push(mesh.flow_directions[i]);
            new_foam.push(mesh.foam[i]);
            spatial.entry((gx, gy, gz)).or_default().push((idx, pos));
            idx
        };
        remap.push(new_idx);
    }

    // Remap indices
    for idx in &mut mesh.indices {
        *idx = remap[*idx as usize];
    }

    mesh.positions = new_positions;
    mesh.normals = new_normals;
    mesh.fluid_types = new_fluid_types;
    mesh.uvs = new_uvs;
    mesh.flow_directions = new_flow_directions;
    mesh.foam = new_foam;
}

/// QEF vertex refinement: for each welded vertex, collect adjacent triangle normals,
/// build a QEF, and solve for optimal position. Clamps displacement to 0.4 max.
/// Pins chunk-edge vertices (coords < 0.5 or > size-0.5) and rock-contact
/// vertices (`pinned`), which must stay buried in the terrain.
fn qef_refine_vertices(mesh: &mut FluidMeshData, grid_size: usize, pinned: &[bool]) {
    if mesh.positions.is_empty() || mesh.indices.is_empty() {
        return;
    }

    let size_f = grid_size as f32;
    let lo = 0.5_f32;
    let hi = size_f - 0.5;
    let max_disp = 0.4_f32;
    let num_verts = mesh.positions.len();

    // Collect per-vertex triangle normals (area-weighted)
    let mut vert_qefs: Vec<QefData> = (0..num_verts).map(|_| QefData::new()).collect();
    let num_tris = mesh.indices.len() / 3;

    for t in 0..num_tris {
        let i0 = mesh.indices[t * 3] as usize;
        let i1 = mesh.indices[t * 3 + 1] as usize;
        let i2 = mesh.indices[t * 3 + 2] as usize;

        let p0 = Vec3::from(mesh.positions[i0]);
        let p1 = Vec3::from(mesh.positions[i1]);
        let p2 = Vec3::from(mesh.positions[i2]);

        let edge1 = p1 - p0;
        let edge2 = p2 - p0;
        let cross = edge1.cross(edge2);
        let len = cross.length();
        if len < 1e-8 {
            continue;
        }
        let normal = cross / len;
        let centroid = (p0 + p1 + p2) / 3.0;

        for &vi in &[i0, i1, i2] {
            let pos = Vec3::from(mesh.positions[vi]);
            vert_qefs[vi].add(pos, normal);
            // Also add centroid constraint to pull toward surface
            vert_qefs[vi].add(centroid, normal);
        }
    }

    // Solve QEF per vertex and apply clamped displacement
    for vi in 0..num_verts {
        let pos = mesh.positions[vi];
        // Pin chunk-edge and rock-contact vertices
        if pinned[vi]
            || pos[0] < lo || pos[1] < lo || pos[2] < lo
            || pos[0] > hi || pos[1] > hi || pos[2] > hi
        {
            continue;
        }

        let qef = &vert_qefs[vi];
        if qef.count < 2 {
            continue;
        }

        let solved = qef.solve();
        let original = Vec3::from(pos);
        let mut displacement = solved - original;
        let dist = displacement.length();
        if dist > max_disp {
            displacement *= max_disp / dist;
        }

        let refined = original + displacement;
        mesh.positions[vi] = refined.into();
    }
}

/// Laplacian smoothing for fluid mesh. Builds adjacency from welded index buffer,
/// pins chunk-edge and rock-contact vertices, iteratively blends toward
/// neighbor average. Regenerates UVs from smoothed positions (xz planar projection).
fn smooth_fluid_mesh(mesh: &mut FluidMeshData, iterations: u32, strength: f32, grid_size: usize, pinned: &[bool]) {
    if iterations == 0 || mesh.positions.is_empty() || mesh.indices.is_empty() {
        return;
    }

    let num_verts = mesh.positions.len();
    let size_f = grid_size as f32;
    let lo = 0.5_f32;
    let hi = size_f - 0.5;

    // Build adjacency from index buffer
    let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); num_verts];
    let num_tris = mesh.indices.len() / 3;
    for t in 0..num_tris {
        let i0 = mesh.indices[t * 3] as usize;
        let i1 = mesh.indices[t * 3 + 1] as usize;
        let i2 = mesh.indices[t * 3 + 2] as usize;
        for &(a, b) in &[(i0, i1), (i1, i2), (i2, i0)] {
            if !adjacency[a].contains(&(b as u32)) {
                adjacency[a].push(b as u32);
            }
            if !adjacency[b].contains(&(a as u32)) {
                adjacency[b].push(a as u32);
            }
        }
    }

    // Identify chunk-edge vertices to pin (rock-contact pins come in via `pinned`)
    let is_edge: Vec<bool> = mesh.positions.iter().enumerate().map(|(vi, p)| {
        pinned[vi] || p[0] < lo || p[1] < lo || p[2] < lo || p[0] > hi || p[1] > hi || p[2] > hi
    }).collect();

    // Iterative smoothing
    for _ in 0..iterations {
        let old: Vec<[f32; 3]> = mesh.positions.clone();
        for vi in 0..num_verts {
            if is_edge[vi] || adjacency[vi].is_empty() {
                continue;
            }
            let mut avg = [0.0f32; 3];
            for &ni in &adjacency[vi] {
                let np = &old[ni as usize];
                avg[0] += np[0];
                avg[1] += np[1];
                avg[2] += np[2];
            }
            let n = adjacency[vi].len() as f32;
            avg[0] /= n;
            avg[1] /= n;
            avg[2] /= n;

            let p = &old[vi];
            mesh.positions[vi] = [
                p[0] + (avg[0] - p[0]) * strength,
                p[1] + (avg[1] - p[1]) * strength,
                p[2] + (avg[2] - p[2]) * strength,
            ];
        }
    }

    // Regenerate UVs from smoothed positions (xz planar projection)
    for vi in 0..num_verts {
        mesh.uvs[vi] = [mesh.positions[vi][0], mesh.positions[vi][2]];
    }
}

/// Recalculate area-weighted vertex normals from triangle geometry.
/// Replaces flat per-triangle normals with smooth averaged normals.
fn recalculate_fluid_normals(mesh: &mut FluidMeshData) {
    if mesh.positions.is_empty() || mesh.indices.is_empty() {
        return;
    }

    let num_verts = mesh.positions.len();

    // Zero all normals
    for n in &mut mesh.normals {
        *n = [0.0, 0.0, 0.0];
    }

    // Accumulate area-weighted normals per triangle
    let num_tris = mesh.indices.len() / 3;
    for t in 0..num_tris {
        let i0 = mesh.indices[t * 3] as usize;
        let i1 = mesh.indices[t * 3 + 1] as usize;
        let i2 = mesh.indices[t * 3 + 2] as usize;

        let p0 = mesh.positions[i0];
        let p1 = mesh.positions[i1];
        let p2 = mesh.positions[i2];

        // Cross product (un-normalized = area-weighted)
        let ax = p1[0] - p0[0];
        let ay = p1[1] - p0[1];
        let az = p1[2] - p0[2];
        let bx = p2[0] - p0[0];
        let by = p2[1] - p0[1];
        let bz = p2[2] - p0[2];
        let nx = ay * bz - az * by;
        let ny = az * bx - ax * bz;
        let nz = ax * by - ay * bx;

        for &vi in &[i0, i1, i2] {
            mesh.normals[vi][0] += nx;
            mesh.normals[vi][1] += ny;
            mesh.normals[vi][2] += nz;
        }
    }

    // Normalize
    for vi in 0..num_verts {
        let n = &mut mesh.normals[vi];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > 1e-10 {
            n[0] /= len;
            n[1] /= len;
            n[2] /= len;
        }
    }
}

/// Compute flow direction from fluid level gradient (for UV animation).
fn compute_flow_direction(grid: &ChunkFluidGrid, x: usize, y: usize, z: usize) -> [f32; 3] {
    let size = grid.size;
    let level = grid.get(x, y, z).level;

    let mut dx = 0.0f32;
    let mut dz = 0.0f32;

    // X gradient
    if x > 0 && grid.grid_point_density(x - 1, y, z) <= 0.0 {
        dx += level - grid.get(x - 1, y, z).level;
    }
    if x + 1 < size && grid.grid_point_density(x + 1, y, z) <= 0.0 {
        dx += grid.get(x + 1, y, z).level - level;
    }

    // Z gradient
    if z > 0 && grid.grid_point_density(x, y, z - 1) <= 0.0 {
        dz += level - grid.get(x, y, z - 1).level;
    }
    if z + 1 < size && grid.grid_point_density(x, y, z + 1) <= 0.0 {
        dz += grid.get(x, y, z + 1).level - level;
    }

    // Downward gradient (waterfall detection)
    let mut dy_mag = 0.0f32;
    if y > 0 && grid.grid_point_density(x, y - 1, z) <= 0.0 {
        let below_level = grid.get(x, y - 1, z).level;
        dy_mag = (level - below_level).max(0.0);
    }

    let horiz_mag = (dx * dx + dz * dz).sqrt();
    let total_mag = (horiz_mag * horiz_mag + dy_mag * dy_mag).sqrt().min(1.0);

    if total_mag < 0.001 {
        [0.0, 0.0, 0.0]
    } else {
        let inv = if horiz_mag > 0.001 {
            1.0 / horiz_mag
        } else {
            0.0
        };
        [dx * inv, dz * inv, total_mag]
    }
}

/// Determine the dominant fluid type among non-empty neighboring cells.
/// When water-family wins over lava, returns the most common water subtype.
pub fn dominant_fluid_type(grid: &ChunkFluidGrid, x: usize, y: usize, z: usize) -> FluidType {
    let mut lava_count = 0u32;
    // Index 0 unused; indices 1,3-8 are water-family types
    let mut water_counts = [0u32; 10];
    let size = grid.size;

    for cz in z..=(z + 1).min(size - 1) {
        for cy in y..=(y + 1).min(size - 1) {
            for cx in x..=(x + 1).min(size - 1) {
                let cell = grid.get(cx, cy, cz);
                // Flux-carrying cells count by their REMEMBERED type even
                // when momentarily dry (stream ribbon cells mid supply-gap
                // must not tint as the generic-Water fallback).
                if cell.level >= MIN_LEVEL || grid.flux_at(cx, cy, cz) > 0.0 {
                    if cell.fluid_type.is_lava() {
                        lava_count += 1;
                    } else {
                        let idx = cell.fluid_type as u8 as usize;
                        if idx < water_counts.len() {
                            water_counts[idx] += 1;
                        }
                    }
                }
            }
        }
    }

    let total_water: u32 = water_counts.iter().sum();
    if lava_count > total_water {
        FluidType::Lava
    } else {
        // Prefer proper subtypes (3-9) over generic Water (1)
        let mut best_idx = 0u8;
        let mut best_count = 0u32;
        for i in 3..water_counts.len() {
            if water_counts[i] > best_count {
                best_count = water_counts[i];
                best_idx = i as u8;
            }
        }
        // Fall back to generic Water only if no subtype found
        if best_count == 0 {
            best_idx = 1;
        }
        FluidType::from_u8(best_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> FluidConfig {
        FluidConfig::default()
    }

    fn make_fluid_grid(regions: &[(std::ops::Range<usize>, std::ops::Range<usize>, std::ops::Range<usize>, f32)]) -> ChunkFluidGrid {
        let mut grid = ChunkFluidGrid::new(16);
        for (xr, yr, zr, level) in regions {
            for z in zr.clone() {
                for y in yr.clone() {
                    for x in xr.clone() {
                        let cell = grid.get_mut(x, y, z);
                        cell.level = *level;
                        cell.fluid_type = FluidType::Water;
                    }
                }
            }
        }
        grid
    }

    fn no_boundary() -> BoundaryLevels {
        BoundaryLevels::empty(16)
    }

    #[test]
    fn empty_grid_produces_no_mesh() {
        let grid = ChunkFluidGrid::new(16);
        let mesh = mesh_fluid(&grid, &no_boundary(), &default_config());
        assert!(mesh.positions.is_empty());
        assert!(mesh.indices.is_empty());
    }

    #[test]
    fn single_source_produces_mesh() {
        let grid = make_fluid_grid(&[(4..8, 4..8, 4..8, 1.0)]);
        let mesh = mesh_fluid(&grid, &no_boundary(), &default_config());
        assert!(!mesh.positions.is_empty(), "Should produce mesh vertices");
        assert!(!mesh.indices.is_empty(), "Should produce mesh indices");
        assert_eq!(mesh.indices.len() % 3, 0, "Indices should be triangles");
        assert_eq!(mesh.uvs.len(), mesh.positions.len(), "UVs per vertex");
        assert_eq!(
            mesh.flow_directions.len(),
            mesh.positions.len(),
            "Flow per vertex"
        );
    }

    #[test]
    fn test_mc_produces_mesh() {
        let grid = make_fluid_grid(&[(6..10, 6..10, 6..10, 1.0)]);
        let mesh = mesh_fluid_mc(&grid, &no_boundary(), MeshFlags::legacy());
        assert!(!mesh.positions.is_empty(), "MC should produce vertices");
        assert!(!mesh.indices.is_empty(), "MC should produce indices");
        assert_eq!(mesh.indices.len() % 3, 0, "MC indices should be triangles");
        // make_fluid_grid sets FluidType::Water (=1), so dominant_fluid_type returns Water
        for &ft in &mesh.fluid_types {
            assert_eq!(ft, FluidType::Water as u8, "MC fluid type should passthrough Water (1)");
        }
    }

    #[test]
    fn test_mc_no_mesh_for_empty() {
        let grid = ChunkFluidGrid::new(16);
        let mesh = mesh_fluid_mc(&grid, &no_boundary(), MeshFlags::legacy());
        assert!(mesh.positions.is_empty(), "Empty grid should produce no MC geometry");
        assert!(mesh.indices.is_empty());
    }

    #[test]
    fn test_mc_passthrough_subtype() {
        // Grid with WaterRiver cells should produce fluid_type=6 in the mesh
        let mut grid = ChunkFluidGrid::new(16);
        for z in 6..10 {
            for y in 6..10 {
                for x in 6..10 {
                    let cell = grid.get_mut(x, y, z);
                    cell.level = 1.0;
                    cell.fluid_type = FluidType::WaterRiver;
                }
            }
        }
        let mesh = mesh_fluid(&grid, &no_boundary(), &default_config());
        assert!(!mesh.positions.is_empty(), "Should produce vertices");
        for &ft in &mesh.fluid_types {
            assert_eq!(ft, FluidType::WaterRiver as u8, "Should passthrough WaterRiver (6)");
        }
    }

    #[test]
    fn test_solid_no_phantom_surface() {
        // A grid with solid rock and air but NO fluid should produce no mesh at all.
        // This verifies cube_has_fluid() prevents phantom surfaces at cave walls.
        let mut grid = ChunkFluidGrid::new(16);
        // Make bottom half solid, top half air
        for z in 0..16 {
            for y in 0..8 {
                for x in 0..16 {
                    grid.set_density(x, y, z, 1.0); // solid
                }
            }
        }
        let mesh = mesh_fluid(&grid, &no_boundary(), &default_config());
        assert!(
            mesh.positions.is_empty(),
            "Solid/air boundary with no fluid should produce no mesh, got {} verts",
            mesh.positions.len()
        );
    }

    #[test]
    fn test_fluid_on_rock_floor_not_floating() {
        // Fluid sitting on a solid floor should produce mesh.
        // With full fluid levels at the boundary, floor extension doesn't trigger
        // (only triggers for low-fluid transition cells).
        let mut grid = ChunkFluidGrid::new(16);
        // Solid floor at y=0..4
        for z in 4..12 {
            for y in 0..4 {
                for x in 4..12 {
                    grid.set_density(x, y, z, 1.0);
                }
            }
        }
        // Fluid at y=4..6 (sitting on the solid floor, full level)
        for z in 4..12 {
            for y in 4..6 {
                for x in 4..12 {
                    let cell = grid.get_mut(x, y, z);
                    cell.level = 1.0;
                    cell.fluid_type = FluidType::Water;
                }
            }
        }
        let mesh = mesh_fluid(&grid, &no_boundary(), &default_config());
        assert!(!mesh.positions.is_empty(), "Should produce fluid mesh");

        // The minimum Y of vertices should be near the solid/fluid boundary (y≈3-4).
        let min_y = mesh.positions.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
        assert!(
            min_y >= 2.9,
            "Mesh should not extend deep into solid rock, min_y = {:.2}",
            min_y
        );
    }

    #[test]
    fn test_floor_extension_mc() {
        // Production scenario: solid floor, transition cell with near-zero fluid,
        // then real fluid above. MC mesh should extend down through the transition cell.
        let mut grid = ChunkFluidGrid::new(16);
        // Solid floor at y=0..4
        for z in 4..12 {
            for y in 0..4 {
                for x in 4..12 {
                    grid.set_density(x, y, z, 1.0);
                }
            }
        }
        // Transition cell at y=4: near-zero fluid (mimics averaged density boundary)
        for z in 4..12 {
            for x in 4..12 {
                let cell = grid.get_mut(x, 4, z);
                cell.level = 0.02; // below ISO_LEVEL
                cell.fluid_type = FluidType::Water;
            }
        }
        // Real fluid at y=5..8
        for z in 4..12 {
            for y in 5..8 {
                for x in 4..12 {
                    let cell = grid.get_mut(x, y, z);
                    cell.level = 1.0;
                    cell.fluid_type = FluidType::Water;
                }
            }
        }
        let mesh = mesh_fluid_mc(&grid, &no_boundary(), MeshFlags::legacy());
        assert!(!mesh.positions.is_empty(), "MC should produce vertices");
        let min_y = mesh.positions.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
        // Without floor extension, mesh bottom would be at y≈4.9 (above transition cell).
        // With extension, transition cell gets boosted, so mesh extends down to y≈3-4.
        assert!(
            min_y < 4.9,
            "MC mesh should extend below the transition cell gap, min_y = {:.2}",
            min_y
        );
    }

    #[test]
    fn test_no_phantom_walls() {
        // Solid cells beside (not below) fluid should NOT generate phantom mesh.
        // Wall of solid rock next to fluid, with no solid floor below the fluid.
        let mut grid = ChunkFluidGrid::new(16);
        // Solid wall at x=0..4, y=4..12, z=4..12
        for z in 4..12 {
            for y in 4..12 {
                for x in 0..4 {
                    grid.set_density(x, y, z, 1.0);
                }
            }
        }
        // Fluid at x=4..8, y=4..8, z=4..8 (floating in air, next to wall)
        for z in 4..8 {
            for y in 4..8 {
                for x in 4..8 {
                    let cell = grid.get_mut(x, y, z);
                    cell.level = 1.0;
                    cell.fluid_type = FluidType::Water;
                }
            }
        }
        let mesh = mesh_fluid(&grid, &no_boundary(), &default_config());
        assert!(!mesh.positions.is_empty(), "Should produce fluid mesh");
        // The mesh should not generate phantom water on the wall face.
        // Solid wall cells return 1.0 ("inside"), same as fluid cells,
        // so no isosurface forms at the wall/fluid boundary.
        // Vertices can approach x=4 (the boundary) but should not go into x<3.
        let min_x = mesh.positions.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
        assert!(
            min_x >= 3.0,
            "Mesh should not extend deep into solid wall, min_x = {:.2}",
            min_x
        );
    }

    #[test]
    fn test_boundary_levels_seamless() {
        // Two adjacent fluid grids: mesh with boundary levels should not seal at the shared edge.
        // Grid A has fluid at x=12..16, Grid B (the +X neighbor) has fluid at x=0..4.
        // Without boundary data, mesh seals at x=16. With it, the surface should be open.
        let grid_a = make_fluid_grid(&[(12..16, 4..8, 4..8, 1.0)]);

        // Build boundary levels from the neighbor's x=0 face
        let mut boundary = BoundaryLevels::empty(16);
        let mut pos_x_levels = vec![0.0f32; 16 * 16];
        for z in 4..8 {
            for y in 4..8 {
                pos_x_levels[z * 16 + y] = 1.0; // neighbor has fluid at x=0
            }
        }
        boundary.pos_x = Some(pos_x_levels);

        let mesh_with_boundary = mesh_fluid_mc(&grid_a, &boundary, MeshFlags::legacy());
        let mesh_without_boundary = mesh_fluid_mc(&grid_a, &no_boundary(), MeshFlags::legacy());

        // With boundary data, the mesh should have fewer vertices at the +X face
        // (the surface is open there instead of sealed). Alternatively, the vertex count
        // should differ, indicating the boundary data changed the mesh.
        assert!(
            mesh_with_boundary.positions.len() != mesh_without_boundary.positions.len(),
            "Boundary levels should change the mesh at the shared edge (with={}, without={})",
            mesh_with_boundary.positions.len(),
            mesh_without_boundary.positions.len()
        );
    }

    #[test]
    fn test_dominant_type_prefers_subtype_over_generic() {
        // Mix of generic Water(1) and WaterRiver(6) cells — dominant should prefer WaterRiver
        let mut grid = ChunkFluidGrid::new(16);
        // Place 4 generic Water cells
        for x in 6..8 {
            for y in 6..8 {
                let cell = grid.get_mut(x, y, 6);
                cell.level = 1.0;
                cell.fluid_type = FluidType::Water;
            }
        }
        // Place 2 WaterRiver cells in the same 2x2x2 neighborhood
        for x in 6..8 {
            let cell = grid.get_mut(x, 6, 7);
            cell.level = 1.0;
            cell.fluid_type = FluidType::WaterRiver;
        }
        // Query at (6,6,6) — 2x2x2 cube covers both types
        let result = dominant_fluid_type(&grid, 6, 6, 6);
        assert_eq!(
            result,
            FluidType::WaterRiver,
            "Should prefer WaterRiver subtype over generic Water even when Water has more cells"
        );
    }

    #[test]
    fn test_weld_reduces_vertex_count() {
        // MC emits 3 vertices per triangle — welding should reduce count significantly
        let grid = make_fluid_grid(&[(4..8, 4..8, 4..8, 1.0)]);
        let raw = mesh_fluid_mc(&grid, &no_boundary(), MeshFlags::legacy());
        let raw_count = raw.positions.len();
        assert!(raw_count > 0, "Should have raw vertices");
        // Every vertex in raw MC is unique (3 per tri)
        assert_eq!(raw_count, raw.indices.len(), "Raw MC: 3 unique verts per tri");

        let mut welded = raw;
        weld_vertices(&mut welded);
        assert!(
            welded.positions.len() < raw_count,
            "Welding should reduce vertex count: {} -> {}",
            raw_count,
            welded.positions.len()
        );
        // All parallel arrays should match
        assert_eq!(welded.normals.len(), welded.positions.len());
        assert_eq!(welded.fluid_types.len(), welded.positions.len());
        assert_eq!(welded.uvs.len(), welded.positions.len());
        assert_eq!(welded.flow_directions.len(), welded.positions.len());
        // Indices should still be valid
        for &idx in &welded.indices {
            assert!((idx as usize) < welded.positions.len(), "Index out of range after weld");
        }
    }

    /// Reference implementation: the original fixed 27-cell (`-1..=1`³) weld scan,
    /// inlined here so the shipped epsilon-bounded `weld_vertices` can be proven
    /// byte-for-byte identical to the pre-optimization baseline.
    fn weld_vertices_ref(mesh: &mut FluidMeshData) {
        if mesh.positions.is_empty() {
            return;
        }
        let cell_size: f32 = 0.01;
        let epsilon: f32 = 1e-5;
        let inv_cell = 1.0 / cell_size;
        let mut spatial: HashMap<(i32, i32, i32), Vec<(u32, [f32; 3])>> = HashMap::new();
        let mut remap: Vec<u32> = Vec::with_capacity(mesh.positions.len());
        let mut new_positions: Vec<[f32; 3]> = Vec::new();
        let mut new_normals: Vec<[f32; 3]> = Vec::new();
        let mut new_fluid_types: Vec<u8> = Vec::new();
        let mut new_uvs: Vec<[f32; 2]> = Vec::new();
        let mut new_flow_directions: Vec<[f32; 3]> = Vec::new();
        let mut new_foam: Vec<f32> = Vec::new();
        for i in 0..mesh.positions.len() {
            let pos = mesh.positions[i];
            let gx = (pos[0] * inv_cell).floor() as i32;
            let gy = (pos[1] * inv_cell).floor() as i32;
            let gz = (pos[2] * inv_cell).floor() as i32;
            let mut found = None;
            'search: for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let key = (gx + dx, gy + dy, gz + dz);
                        if let Some(bucket) = spatial.get(&key) {
                            for &(idx, ref p) in bucket {
                                let d0 = pos[0] - p[0];
                                let d1 = pos[1] - p[1];
                                let d2 = pos[2] - p[2];
                                if d0 * d0 + d1 * d1 + d2 * d2 < epsilon * epsilon {
                                    found = Some(idx);
                                    break 'search;
                                }
                            }
                        }
                    }
                }
            }
            let new_idx = if let Some(idx) = found {
                idx
            } else {
                let idx = new_positions.len() as u32;
                new_positions.push(pos);
                new_normals.push(mesh.normals[i]);
                new_fluid_types.push(mesh.fluid_types[i]);
                new_uvs.push(mesh.uvs[i]);
                new_flow_directions.push(mesh.flow_directions[i]);
            new_foam.push(mesh.foam[i]);
                spatial.entry((gx, gy, gz)).or_default().push((idx, pos));
                idx
            };
            remap.push(new_idx);
        }
        for idx in &mut mesh.indices {
            *idx = remap[*idx as usize];
        }
        mesh.positions = new_positions;
        mesh.normals = new_normals;
        mesh.fluid_types = new_fluid_types;
        mesh.uvs = new_uvs;
        mesh.flow_directions = new_flow_directions;
    mesh.foam = new_foam;
    }

    #[test]
    fn test_weld_bounded_search_is_bit_identical() {
        // Build a real, sizeable MC mesh so the weld has many coincident verts and
        // near-face cases to resolve, then assert the shipped epsilon-bounded weld
        // produces byte-for-byte the same result as the full 27-cell scan.
        let grid = make_fluid_grid(&[(2..14, 2..14, 2..14, 1.0)]);
        let raw = mesh_fluid_mc(&grid, &no_boundary(), MeshFlags::legacy());
        assert!(raw.positions.len() > 500, "want a non-trivial mesh to exercise the search");

        let mut a = clone_mesh(&raw);
        let mut b = clone_mesh(&raw);
        weld_vertices(&mut a); // shipped (bounded)
        weld_vertices_ref(&mut b); // reference (full 27-cell)

        assert_eq!(a.positions.len(), b.positions.len(), "vertex count differs");
        assert_eq!(a.indices, b.indices, "remapped indices differ");
        for (va, vb) in a.positions.iter().zip(&b.positions) {
            for k in 0..3 {
                assert_eq!(va[k].to_bits(), vb[k].to_bits(), "position bits differ");
            }
        }
        for (na, nb) in a.normals.iter().zip(&b.normals) {
            for k in 0..3 {
                assert_eq!(na[k].to_bits(), nb[k].to_bits(), "normal bits differ");
            }
        }
        assert_eq!(a.fluid_types, b.fluid_types, "fluid_types differ");
    }

    /// A/B microbench (release): `cargo test -p voxel-fluid --release weld_ab -- --ignored --nocapture`.
    /// Times the shipped epsilon-bounded weld against the original 27-cell scan on
    /// an identical real MC mesh, in the same binary so the comparison is fair.
    #[test]
    #[ignore]
    fn bench_weld_ab() {
        use std::time::Instant;
        let grid = make_fluid_grid(&[(1..15, 1..15, 1..15, 1.0)]);
        let raw = mesh_fluid_mc(&grid, &no_boundary(), MeshFlags::legacy());
        println!("raw MC verts: {}", raw.positions.len());

        let rounds = 6;
        let iters = 2000;
        let mut best_new = f64::MAX;
        let mut best_ref = f64::MAX;
        let mut sink = 0u64;
        for _ in 0..rounds {
            let t = Instant::now();
            for _ in 0..iters {
                let mut m = clone_mesh(&raw);
                weld_vertices(&mut m);
                sink = sink.wrapping_add(m.positions.len() as u64);
            }
            best_new = best_new.min(t.elapsed().as_secs_f64() / iters as f64 * 1e6);

            let t = Instant::now();
            for _ in 0..iters {
                let mut m = clone_mesh(&raw);
                weld_vertices_ref(&mut m);
                sink = sink.wrapping_add(m.positions.len() as u64);
            }
            best_ref = best_ref.min(t.elapsed().as_secs_f64() / iters as f64 * 1e6);
        }
        // Subtract clone+push overhead (run the loop body with no search work).
        let mut best_clone = f64::MAX;
        for _ in 0..rounds {
            let t = Instant::now();
            for _ in 0..iters {
                let m = clone_mesh(&raw);
                sink = sink.wrapping_add(m.positions.len() as u64);
            }
            best_clone = best_clone.min(t.elapsed().as_secs_f64() / iters as f64 * 1e6);
        }
        let net_ref = best_ref - best_clone;
        let net_new = best_new - best_clone;
        println!(
            "weld A/B: ref(27-cell)={:.3}us new(bounded)={:.3}us clone={:.3}us | NET ref={:.3} new={:.3} -> {:.1}% ({:.2}x) sink={}",
            best_ref, best_new, best_clone, net_ref, net_new,
            (1.0 - net_new / net_ref) * 100.0, net_ref / net_new, sink
        );
    }

    fn clone_mesh(m: &FluidMeshData) -> FluidMeshData {
        FluidMeshData {
            positions: m.positions.clone(),
            normals: m.normals.clone(),
            fluid_types: m.fluid_types.clone(),
            indices: m.indices.clone(),
            uvs: m.uvs.clone(),
            flow_directions: m.flow_directions.clone(),
            foam: m.foam.clone(),
        }
    }

    #[test]
    fn test_smooth_normals_vary() {
        // After recalculating normals on a welded mesh, normals should vary
        // (not all identical flat per-triangle normals)
        let grid = make_fluid_grid(&[(4..8, 4..8, 4..8, 1.0)]);
        let mut mesh = mesh_fluid_mc(&grid, &no_boundary(), MeshFlags::legacy());
        weld_vertices(&mut mesh);
        recalculate_fluid_normals(&mut mesh);

        assert!(!mesh.normals.is_empty(), "Should have normals");
        // Check that not all normals are identical
        let first = mesh.normals[0];
        let has_variation = mesh.normals.iter().any(|n| {
            (n[0] - first[0]).abs() > 0.01
                || (n[1] - first[1]).abs() > 0.01
                || (n[2] - first[2]).abs() > 0.01
        });
        assert!(has_variation, "Recalculated normals should vary across the mesh");
    }

    #[test]
    fn test_full_pipeline_valid_mesh() {
        // Full pipeline: MC -> weld -> QEF -> smooth -> normals
        let grid = make_fluid_grid(&[(4..10, 4..10, 4..10, 1.0)]);
        let mesh = mesh_fluid(&grid, &no_boundary(), &default_config());

        assert!(!mesh.positions.is_empty(), "Pipeline should produce vertices");
        assert!(!mesh.indices.is_empty(), "Pipeline should produce indices");
        assert_eq!(mesh.indices.len() % 3, 0, "Indices should be triangles");

        // All parallel arrays equal length
        let n = mesh.positions.len();
        assert_eq!(mesh.normals.len(), n, "Normals count mismatch");
        assert_eq!(mesh.fluid_types.len(), n, "Fluid types count mismatch");
        assert_eq!(mesh.uvs.len(), n, "UVs count mismatch");
        assert_eq!(mesh.flow_directions.len(), n, "Flow directions count mismatch");

        // All indices in range
        for &idx in &mesh.indices {
            assert!((idx as usize) < n, "Index {} out of range (n={})", idx, n);
        }

        // All normals should be unit length (or zero for degenerate)
        for normal in &mesh.normals {
            let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            assert!(
                (len - 1.0).abs() < 0.01 || len < 1e-6,
                "Normal should be unit length, got {:.4}",
                len
            );
        }
    }

    /// Build a (size+1)^3 density field from a function of lattice coords.
    fn density_field_from_fn(size: usize, f: impl Fn(f32, f32, f32) -> f32) -> Vec<f32> {
        let stride = size + 1;
        let mut d = vec![-1.0f32; stride * stride * stride];
        for gz in 0..stride {
            for gy in 0..stride {
                for gx in 0..stride {
                    d[gz * stride * stride + gy * stride + gx] =
                        f(gx as f32, gy as f32, gz as f32);
                }
            }
        }
        d
    }

    /// Indices of vertices on open mesh edges (exactly one adjacent triangle).
    /// Only meaningful on a welded mesh.
    fn open_boundary_vertices(mesh: &FluidMeshData) -> Vec<usize> {
        let mut counts: HashMap<(u32, u32), u32> = HashMap::new();
        for t in 0..mesh.indices.len() / 3 {
            let i0 = mesh.indices[t * 3];
            let i1 = mesh.indices[t * 3 + 1];
            let i2 = mesh.indices[t * 3 + 2];
            for &(a, b) in &[(i0, i1), (i1, i2), (i2, i0)] {
                *counts.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        let mut verts: Vec<usize> = counts
            .iter()
            .filter(|&(_, &c)| c == 1)
            .flat_map(|(&(a, b), _)| [a as usize, b as usize])
            .collect();
        verts.sort_unstable();
        verts.dedup();
        verts
    }

    /// Basin with a vertical rock wall whose DC surface sits at x=8.7 — deep
    /// inside the boundary cell, where the old fixed-t rim crossing (t=0.15
    /// from the air point, x=8.15) left a 0.55-cell visible gap.
    fn make_walled_basin() -> ChunkFluidGrid {
        let size = 16;
        let mut grid = ChunkFluidGrid::new(size);
        let d = density_field_from_fn(size, |x, y, _z| ((x - 8.7) * 0.5).max((2.3 - y) * 0.5));
        grid.update_density(&d);
        for z in 2..=13 {
            for y in 3..=5 {
                for x in 2..=8 {
                    let cell = grid.get_mut(x, y, z);
                    cell.level = 1.0;
                    cell.fluid_type = FluidType::Water;
                }
            }
        }
        grid
    }

    /// Gently rising rock floor (shoreline): floor surface at y = 3 + 0.35x.
    /// Water surface at y = 6.625 meets the floor near x = 10.4. Cells at x=9
    /// have their min-corner lattice point inside rock but still hold water
    /// (the lapping configuration).
    fn make_shoreline() -> ChunkFluidGrid {
        let size = 16;
        let mut grid = ChunkFluidGrid::new(size);
        let d = density_field_from_fn(size, |x, y, _z| ((3.0 + 0.35 * x) - y) * 0.4);
        grid.update_density(&d);
        for z in 3..=12 {
            for x in 2..=9 {
                for y in 3..=5 {
                    let cell = grid.get_mut(x, y, z);
                    cell.level = 1.0;
                    cell.fluid_type = FluidType::Water;
                }
                let cell = grid.get_mut(x, 6, z);
                cell.level = 0.4;
                cell.fluid_type = FluidType::Water;
            }
        }
        grid
    }

    #[test]
    fn test_rim_reaches_recessed_rock_wall() {
        // The water rim must reach the DC wall at x=8.7 and tuck slightly
        // inside it — not stop at the old fixed crossing x=8.15.
        let grid = make_walled_basin();
        let mesh = mesh_fluid_mc(&grid, &no_boundary(), MeshFlags::legacy());
        assert!(!mesh.positions.is_empty(), "basin should produce a mesh");
        let max_x = mesh.positions.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_x >= 8.69,
            "water rim stops {:.2} cells short of the rock wall at x=8.7 (max_x={:.2})",
            8.7 - max_x, max_x
        );
        assert!(max_x <= 9.01, "water rim blew through the rock wall (max_x={:.2})", max_x);
    }

    #[test]
    fn test_rim_open_boundary_buried_in_rock() {
        // Full pipeline (QEF + smoothing enabled by default): every open mesh
        // edge that is not on a chunk face must lie at/inside the terrain
        // surface. This is the literal "no visible gap slit" property, and it
        // also proves the rock pins survive the post passes.
        let grid = make_walled_basin();
        let mesh = mesh_fluid(&grid, &no_boundary(), &default_config());
        assert!(!mesh.positions.is_empty());
        let open = open_boundary_vertices(&mesh);
        assert!(!open.is_empty(), "expected an open contact ring at the wall");
        let size_f = 16.0f32;
        let mut checked = 0;
        for vi in open {
            let p = mesh.positions[vi];
            // Chunk-face vertices are sealed/pinned by the chunk-edge rules.
            if p[0] < 0.5 || p[1] < 0.5 || p[2] < 0.5
                || p[0] > size_f - 0.5 || p[1] > size_f - 0.5 || p[2] > size_f - 0.5
            {
                continue;
            }
            checked += 1;
            let d = terrain_density_at(&grid, p);
            assert!(
                d >= -1e-3,
                "open-boundary vertex [{:.2},{:.2},{:.2}] floats in air (density {:.3}) — visible gap",
                p[0], p[1], p[2], d
            );
        }
        assert!(checked > 0, "no interior open-boundary vertices were checked");
    }

    #[test]
    fn test_top_surface_height_unchanged_by_rim_fix() {
        // Away from walls the top surface must stay where the fluid field puts
        // it: full cells at y=3..=5, dry above -> crossing at y = 5 + 0.85.
        let grid = make_walled_basin();
        let mesh = mesh_fluid_mc(&grid, &no_boundary(), MeshFlags::legacy());
        let mut top = f32::NEG_INFINITY;
        for p in &mesh.positions {
            if p[0] > 3.0 && p[0] < 7.0 && p[2] > 5.0 && p[2] < 10.0 {
                top = top.max(p[1]);
            }
        }
        assert!(
            (top - 5.85).abs() < 0.02,
            "interior top surface moved: y={:.3}, expected 5.85",
            top
        );
    }

    #[test]
    fn test_lapping_water_keeps_level_over_submerged_rock() {
        // At x=9 the floor (y=6.15) pokes just above the lattice plane y=6, so
        // the lattice point is solid but the cell holds 0.4 water — the surface
        // there must stay at the water level (y = 6 + 0.625), not snap down to
        // the rock face and dent the sheet.
        let grid = make_shoreline();
        let mesh = mesh_fluid_mc(&grid, &no_boundary(), MeshFlags::legacy());
        assert!(!mesh.positions.is_empty());
        let at_level = mesh.positions.iter().any(|p| {
            p[0] > 8.6 && p[0] < 9.4 && p[1] > 6.5 && p[1] < 6.72
        });
        assert!(
            at_level,
            "no water-surface vertex near x=9 at the fluid level (~6.625) — lapping rule broken"
        );
        // And no divot down at the rock face under the lapping zone — checked
        // only in the watered z-interior; at the body's z-ends the sheet
        // correctly seals downward into the floor.
        let dented = mesh.positions.iter().any(|p| {
            p[0] > 8.9 && p[0] < 9.1 && p[1] > 5.95 && p[1] < 6.15 && p[2] > 5.0 && p[2] < 11.0
        });
        assert!(!dented, "water surface dented down to the rock face in the lapping zone");
    }

    #[test]
    fn test_shoreline_tucks_under_floor() {
        // Past the last watered column the sheet must dive under the rising
        // floor (buried contact ring), never float above it in open air.
        let grid = make_shoreline();
        let mesh = mesh_fluid(&grid, &no_boundary(), &default_config());
        assert!(!mesh.positions.is_empty());
        let mut shoreline_verts = 0;
        for p in &mesh.positions {
            if p[0] >= 9.5 && p[0] <= 12.0 {
                shoreline_verts += 1;
                let d = terrain_density_at(&grid, *p);
                assert!(
                    d >= -0.05 || p[1] <= 6.66,
                    "shoreline vertex [{:.2},{:.2},{:.2}] floats above the floor (density {:.3})",
                    p[0], p[1], p[2], d
                );
            }
        }
        assert!(
            shoreline_verts > 0,
            "sheet never reached the shoreline band (x >= 9.5) — gate still cutting it early"
        );
    }

    #[test]
    fn test_chunk_edge_pinning() {
        // Vertices near chunk boundary should not drift negative after smoothing/QEF
        let grid = make_fluid_grid(&[(0..6, 4..10, 4..10, 1.0)]);
        let mesh = mesh_fluid(&grid, &no_boundary(), &default_config());

        for pos in &mesh.positions {
            assert!(
                pos[0] >= -0.01 && pos[1] >= -0.01 && pos[2] >= -0.01,
                "Vertex drifted negative: [{:.3}, {:.3}, {:.3}]",
                pos[0], pos[1], pos[2]
            );
            assert!(
                pos[0] <= 16.01 && pos[1] <= 16.01 && pos[2] <= 16.01,
                "Vertex exceeded chunk bounds: [{:.3}, {:.3}, {:.3}]",
                pos[0], pos[1], pos[2]
            );
        }
    }

    // ── Rim/skirt bundle (2026-08-04): settled saucer pool ────────────────
    // User repro: a fully-settled pool renders its RIM higher than the flat
    // middle with bare floor between them, and the fluid mesh continues
    // under the floor as a juttery skirt. Fixture: a saucer-shaped rock
    // floor, a settled pool in the middle, and a ring of drained rim cells
    // (sub-iso levels, stagnant, still hysteresis-held) one step up the
    // saucer wall — exactly what the sim leaves behind after settling.

    /// Saucer floor: height 3 + 0.08·r² from chunk center. Pool filled to
    /// y=5 (surface ≈ 5.85) out to r≈4.2; drained-but-held remnant cells in
    /// the ring r∈(4.2, 6.0] at the waterline row.
    fn make_settled_saucer() -> ChunkFluidGrid {
        let size = 16;
        let mut grid = ChunkFluidGrid::new(size);
        let floor_h = |x: f32, z: f32| 3.0 + 0.08 * ((x - 8.0).powi(2) + (z - 8.0).powi(2));
        let d = density_field_from_fn(size, |x, y, z| (floor_h(x, z) - y) * 0.4);
        grid.update_density(&d);

        // Pool body: full cells wherever the cell row sits above the local
        // floor, out to the shoreline.
        for z in 2..14usize {
            for x in 2..14usize {
                let r2 = (x as f32 - 8.0).powi(2) + (z as f32 - 8.0).powi(2);
                if r2 <= 4.2f32.powi(2) {
                    for y in 3..=5usize {
                        let cell = grid.get_mut(x, y, z);
                        cell.level = 1.0;
                        cell.fluid_type = FluidType::Lava;
                    }
                }
            }
        }

        // Drained rim rings: cells that were wet during flooding (entered
        // the mesh), then drained back into the hysteresis band and went
        // stagnant. Two rows: shoreline remnants at the waterline row (y=5,
        // partially inside rock — the lapping/skirt generators), and the
        // receded high-water ring one row ABOVE the settled surface (y=6, up
        // the saucer wall — the raised phantom ring).
        let mut ring: Vec<(usize, usize, usize)> = Vec::new();
        for z in 2..14usize {
            for x in 2..14usize {
                let r2 = (x as f32 - 8.0).powi(2) + (z as f32 - 8.0).powi(2);
                if r2 > 4.2f32.powi(2) && r2 <= 6.0f32.powi(2) {
                    ring.push((x, 5, z));
                }
                let fh = floor_h(x as f32, z as f32);
                if fh > 5.5 && fh < 6.5 {
                    ring.push((x, 6, z));
                }
            }
        }
        for &(x, y, z) in &ring {
            let cell = grid.get_mut(x, y, z);
            cell.level = 0.2; // wet enough to enter the mesh
            cell.fluid_type = FluidType::Lava;
        }
        grid.update_mesh_hysteresis(false); // ring cells become sticky
        for &(x, y, z) in &ring {
            let cell = grid.get_mut(x, y, z);
            cell.level = 0.10; // drained into [STICKY_OFF, STICKY_ON)
            cell.stagnant_ticks = 20; // settled
        }
        grid
    }

    fn saucer_config(sticky_release: bool, floor_clamp: bool, buried_cull: bool) -> FluidConfig {
        FluidConfig {
            mesh_sticky_release: sticky_release,
            mesh_floor_clamp: floor_clamp,
            mesh_buried_cull: buried_cull,
            ..FluidConfig::default()
        }
    }

    /// Mesh the saucer the way the fluid thread does: hysteresis refresh
    /// (honoring the release flag), then the full pipeline.
    fn mesh_saucer(grid: &mut ChunkFluidGrid, config: &FluidConfig) -> FluidMeshData {
        grid.update_mesh_hysteresis(config.mesh_sticky_release);
        mesh_fluid(grid, &no_boundary(), config)
    }

    /// The phantom-ring band: above the real fluid surface (5.85 + slack),
    /// outside the pool body.
    fn ring_verts_above_surface(mesh: &FluidMeshData) -> usize {
        mesh.positions
            .iter()
            .filter(|p| {
                let r2 = (p[0] - 8.0).powi(2) + (p[2] - 8.0).powi(2);
                r2 > 4.0f32.powi(2) && p[1] > 6.02
            })
            .count()
    }

    /// Triangles whose three vertices all sit strictly inside rock — the
    /// under-floor skirt, invisible unless the camera is inside the ground.
    fn buried_triangles(mesh: &FluidMeshData, grid: &ChunkFluidGrid, margin: f32) -> usize {
        (0..mesh.indices.len() / 3)
            .filter(|&t| {
                (0..3).all(|k| {
                    let p = mesh.positions[mesh.indices[t * 3 + k] as usize];
                    terrain_density_at(grid, p) >= margin
                })
            })
            .count()
    }

    #[test]
    fn saucer_legacy_reproduces_ring_and_skirt() {
        // RED-FIRST DOCUMENTATION: with the bundle off, the fixture must
        // reproduce BOTH user-reported symptoms. If this test ever fails the
        // fixture no longer models the bug and the bundle tests prove nothing.
        let mut grid = make_settled_saucer();
        let mesh = mesh_saucer(&mut grid, &saucer_config(false, false, false));
        assert!(!mesh.positions.is_empty());

        let ring = ring_verts_above_surface(&mesh);
        assert!(
            ring > 0,
            "legacy mesher no longer paints the raised phantom ring — fixture drift?"
        );
        let buried = buried_triangles(&mesh, &grid, 0.02);
        assert!(
            buried > 0,
            "legacy mesher no longer emits an under-floor skirt — fixture drift?"
        );
    }

    #[test]
    fn saucer_sticky_release_removes_phantom_ring() {
        let mut grid = make_settled_saucer();
        let mesh = mesh_saucer(&mut grid, &saucer_config(true, false, false));
        assert!(!mesh.positions.is_empty(), "pool body must still mesh");
        assert_eq!(
            ring_verts_above_surface(&mesh),
            0,
            "drained stagnant rim cells must drop out of the mesh (no raised ring)"
        );
    }

    #[test]
    fn saucer_buried_cull_removes_underfloor_skirt() {
        let mut grid = make_settled_saucer();
        let mesh = mesh_saucer(&mut grid, &saucer_config(false, false, true));
        assert!(!mesh.positions.is_empty(), "pool body must still mesh");
        assert_eq!(
            buried_triangles(&mesh, &grid, 0.02),
            0,
            "no triangle may survive with all three vertices inside rock"
        );
        // The waterline must still terminate in rock contact — culling the
        // buried skirt must not reopen the rim gap slits.
        let open = open_boundary_vertices(&mesh);
        let size_f = 16.0f32;
        for vi in open {
            let p = mesh.positions[vi];
            if p[0] < 0.5 || p[1] < 0.5 || p[2] < 0.5
                || p[0] > size_f - 0.5 || p[1] > size_f - 0.5 || p[2] > size_f - 0.5
            {
                continue;
            }
            let d = terrain_density_at(&grid, p);
            assert!(
                d >= -1e-3,
                "open-boundary vertex [{:.2},{:.2},{:.2}] floats in air (density {:.3}) — gap reopened",
                p[0], p[1], p[2], d
            );
        }
    }

    #[test]
    fn saucer_floor_clamp_shrinks_burial_depth() {
        // Flag 2 alone: rim crossings on floor-contact edges recess by a
        // whisker instead of 0.1 cells, so the deepest burial (in density
        // terms) must strictly shrink versus legacy.
        let mut grid = make_settled_saucer();
        let legacy = mesh_saucer(&mut grid, &saucer_config(false, false, false));
        let clamped = mesh_saucer(&mut grid, &saucer_config(false, true, false));
        let max_d = |mesh: &FluidMeshData| {
            mesh.positions
                .iter()
                .map(|p| terrain_density_at(&grid, *p))
                .fold(f32::NEG_INFINITY, f32::max)
        };
        let d_legacy = max_d(&legacy);
        let d_clamped = max_d(&clamped);
        assert!(
            d_clamped < d_legacy - 1e-3,
            "floor clamp did not shrink burial: legacy {:.4}, clamped {:.4}",
            d_legacy, d_clamped
        );
    }

    #[test]
    fn saucer_full_bundle_holds_all_invariants() {
        let mut grid = make_settled_saucer();
        let mesh = mesh_saucer(&mut grid, &saucer_config(true, true, true));
        assert!(!mesh.positions.is_empty(), "pool body must still mesh");

        // (a) nothing under the floor
        assert_eq!(buried_triangles(&mesh, &grid, 0.02), 0, "under-floor skirt survived");
        // (b) no raised rim above the fluid surface
        assert_eq!(ring_verts_above_surface(&mesh), 0, "raised phantom ring survived");
        // (c) interior surface height untouched by the bundle
        let mut top = f32::NEG_INFINITY;
        for p in &mesh.positions {
            let r2 = (p[0] - 8.0).powi(2) + (p[2] - 8.0).powi(2);
            if r2 < 2.0f32.powi(2) {
                top = top.max(p[1]);
            }
        }
        assert!(
            (top - 5.85).abs() < 0.05,
            "interior top surface moved: y={:.3}, expected ≈5.85",
            top
        );
        // (d) no bare-floor gap: every interior open edge lands in rock contact
        let open = open_boundary_vertices(&mesh);
        let size_f = 16.0f32;
        let mut checked = 0;
        for vi in open {
            let p = mesh.positions[vi];
            if p[0] < 0.5 || p[1] < 0.5 || p[2] < 0.5
                || p[0] > size_f - 0.5 || p[1] > size_f - 0.5 || p[2] > size_f - 0.5
            {
                continue;
            }
            checked += 1;
            let d = terrain_density_at(&grid, p);
            assert!(
                d >= -1e-3,
                "open-boundary vertex [{:.2},{:.2},{:.2}] floats in air (density {:.3})",
                p[0], p[1], p[2], d
            );
        }
        assert!(checked > 0, "no interior open boundary found — shoreline vanished?");
    }
}
