//! Cross-chunk boundary density syncing for `ChunkStore`.
//!
//! Holds the boundary-sync impl methods plus the shared free functions
//! `average_boundary_voxel` and `sync_boundary_density` (kept together
//! because the impl methods and the free fn all call `average_boundary_voxel`).
//!
//! Split out of the original `store.rs` god file (behavior-preserving).

use std::collections::{HashMap, HashSet};

use voxel_core::material::Material;
use voxel_core::octree::node::VoxelSample;
use voxel_gen::density::DensityField;

use super::ChunkStore;

impl ChunkStore {
    /// Sync boundary density planes between dirty chunks and their neighbors.
    /// Extends `dirty_bounds` with extra neighbor chunks that need remeshing.
    pub fn sync_boundaries(
        &mut self,
        dirty_bounds: &mut Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)>,
        chunk_size: usize,
    ) {
        let extra = sync_boundary_density(&mut self.density_fields, dirty_bounds, chunk_size);
        dirty_bounds.extend(extra);
    }

    /// Sync this chunk's boundaries with all 26 neighbors (faces + edges +
    /// corners). Used by the diagnostic Force Resync button — more thorough
    /// than `sync_cross_region_densities` (which only does 6 face neighbors)
    /// because seam mismatches at edge/corner cells are common after mid-
    /// session brush ops. Returns the set of chunks whose density was
    /// modified — caller re-extracts hermite + remeshes those.
    pub fn sync_chunk_full_boundaries(
        &mut self,
        chunk: (i32, i32, i32),
        chunk_size: usize,
    ) -> Vec<(i32, i32, i32)> {
        let gs = chunk_size;
        // 13 forward offsets covering faces (3) + edges (6) + corners (4).
        // Each pair of (chunk, neighbor) syncs once; both sides written.
        let offsets: [(i32, i32, i32); 13] = [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1),
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        ];
        // Also include the reverse 13 so we cover ALL 26 face/edge/corner
        // neighbors of `chunk` (not just the forward half-set).
        let all_offsets: Vec<(i32, i32, i32)> = offsets
            .iter()
            .copied()
            .chain(offsets.iter().map(|&(dx, dy, dz)| (-dx, -dy, -dz)))
            .collect();

        let mut updates: Vec<((i32, i32, i32), usize, usize, usize, f32, voxel_core::material::Material)> = Vec::new();
        let mut dirty: HashSet<(i32, i32, i32)> = HashSet::new();

        for (dx, dy, dz) in all_offsets {
            let neighbor = (chunk.0 + dx, chunk.1 + dy, chunk.2 + dz);
            if !self.density_fields.contains_key(&neighbor) {
                continue;
            }
            // Build per-axis pair iterators: 1=>(gs, 0), -1=>(0, gs), 0=>full range.
            let pair_for = |d: i32| -> Vec<(usize, usize)> {
                if d == 1 { vec![(gs, 0)] }
                else if d == -1 { vec![(0, gs)] }
                else { (0..=gs).map(|i| (i, i)).collect() }
            };
            let xs = pair_for(dx);
            let ys = pair_for(dy);
            let zs = pair_for(dz);
            for &(az, bz) in &zs {
                for &(ay, by) in &ys {
                    for &(ax, bx) in &xs {
                        let sa = self.density_fields[&chunk].get(ax, ay, az);
                        let sb = self.density_fields[&neighbor].get(bx, by, bz);
                        let (d, m) = average_boundary_voxel(sa, sb);
                        if sa.density != d || sa.material != m {
                            updates.push((chunk, ax, ay, az, d, m));
                            dirty.insert(chunk);
                        }
                        if sb.density != d || sb.material != m {
                            updates.push((neighbor, bx, by, bz, d, m));
                            dirty.insert(neighbor);
                        }
                    }
                }
            }
        }
        for (key, x, y, z, d, m) in updates {
            if let Some(field) = self.density_fields.get_mut(&key) {
                let s = field.get_mut(x, y, z);
                s.density = d;
                s.material = m;
            }
        }
        for &k in &dirty {
            if let Some(df) = self.density_fields.get_mut(&k) {
                df.compute_metadata();
            }
        }
        dirty.into_iter().collect()
    }

    /// Sync boundary densities between newly generated region chunks and their
    /// already-loaded cross-region face neighbors. Only marks chunks dirty when
    /// voxel values actually change. Returns ALL dirty keys (both region and
    /// non-region) — caller is responsible for hermite re-extraction and
    /// filtering to non-region keys for remeshing.
    pub fn sync_cross_region_densities(
        &mut self,
        region_coords: &[(i32, i32, i32)],
        chunk_size: usize,
    ) -> Vec<(i32, i32, i32)> {
        let gs = chunk_size;
        let region_set: HashSet<_> = region_coords.iter().copied().collect();
        let mut updates: Vec<((i32, i32, i32), usize, usize, usize, f32, Material)> = Vec::new();
        let mut dirty: HashSet<(i32, i32, i32)> = HashSet::new();

        for &(cx, cy, cz) in region_coords {
            let face_neighbors: [(i32, i32, i32); 6] = [
                (cx + 1, cy, cz), (cx - 1, cy, cz),
                (cx, cy + 1, cz), (cx, cy - 1, cz),
                (cx, cy, cz + 1), (cx, cy, cz - 1),
            ];
            for &neighbor in &face_neighbors {
                if region_set.contains(&neighbor) { continue; }
                if !self.density_fields.contains_key(&neighbor) { continue; }

                // Determine shared face axis and boundary coordinates
                let (axis, a_coord, b_coord) = if neighbor.0 != cx {
                    (0, if neighbor.0 > cx { gs } else { 0 }, if neighbor.0 > cx { 0 } else { gs })
                } else if neighbor.1 != cy {
                    (1, if neighbor.1 > cy { gs } else { 0 }, if neighbor.1 > cy { 0 } else { gs })
                } else {
                    (2, if neighbor.2 > cz { gs } else { 0 }, if neighbor.2 > cz { 0 } else { gs })
                };

                let mut face_a_changed = false;
                let mut face_b_changed = false;

                for u in 0..=gs {
                    for v in 0..=gs {
                        let (ax, ay, az) = match axis {
                            0 => (a_coord, u, v),
                            1 => (u, a_coord, v),
                            _ => (u, v, a_coord),
                        };
                        let (bx, by, bz) = match axis {
                            0 => (b_coord, u, v),
                            1 => (u, b_coord, v),
                            _ => (u, v, b_coord),
                        };

                        let sample_a = self.density_fields[&(cx, cy, cz)].get(ax, ay, az);
                        let sample_b = self.density_fields[&neighbor].get(bx, by, bz);
                        let (d, m) = average_boundary_voxel(sample_a, sample_b);
                        if sample_a.density != d || sample_a.material != m {
                            updates.push(((cx, cy, cz), ax, ay, az, d, m));
                            face_a_changed = true;
                        }
                        if sample_b.density != d || sample_b.material != m {
                            updates.push((neighbor, bx, by, bz, d, m));
                            face_b_changed = true;
                        }
                    }
                }
                if face_a_changed { dirty.insert((cx, cy, cz)); }
                if face_b_changed { dirty.insert(neighbor); }
            }
        }

        if updates.is_empty() {
            return Vec::new();
        }

        // Apply all density updates
        for (key, x, y, z, d, m) in updates {
            if let Some(field) = self.density_fields.get_mut(&key) {
                let sample = field.get_mut(x, y, z);
                sample.density = d;
                sample.material = m;
            }
        }

        // Recompute metadata for dirty chunks (requires &mut, is cheap)
        for &key in &dirty {
            if let Some(density) = self.density_fields.get_mut(&key) {
                density.compute_metadata();
            }
        }

        dirty.into_iter().collect()
    }
}

/// Average two voxel samples at the same world position for boundary sync.
/// Density uses min (carved side wins); material preserves solid when possible.
pub(crate) fn average_boundary_voxel(a: &VoxelSample, b: &VoxelSample) -> (f32, Material) {
    let avg_density = a.density.min(b.density);  // carved side wins, no 0.0 degenerate surface
    let material = if a.material.is_solid() && b.material.is_solid() {
        if a.density >= b.density { a.material } else { b.material }
    } else if a.material.is_solid() {
        a.material
    } else if b.material.is_solid() {
        b.material
    } else {
        Material::Air
    };
    // Enforce invariant: Air density must be non-positive
    if !material.is_solid() && avg_density > 0.0 {
        (0.0, material)
    } else {
        (avg_density, material)
    }
}

/// Post-smoothing boundary density sync: average overlap voxels between
/// dirty chunks and their face, edge, and corner neighbors so hermite edges
/// match at seams.
///
/// Returns extra neighbor chunks that need remeshing but weren't already dirty.
pub(crate) fn sync_boundary_density(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    dirty_chunks: &[((i32, i32, i32), usize, usize, usize, usize, usize, usize)],
    chunk_size: usize,
) -> Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> {
    let cs = chunk_size; // density grid is cs+1 in each dimension
    let dirty_keys: HashSet<(i32, i32, i32)> = dirty_chunks.iter().map(|d| d.0).collect();

    // Faces to check: (axis condition on dirty bounds, neighbor offset, local coord on A, local coord on B)
    // For each face we collect updates as: (chunk_key, x, y, z, density, material)
    let mut updates: Vec<((i32, i32, i32), usize, usize, usize, f32, Material)> = Vec::new();
    let mut extra_neighbors: HashSet<(i32, i32, i32)> = HashSet::new();

    for &(key, min_x, min_y, min_z, max_x, max_y, max_z) in dirty_chunks {
        let (cx, cy, cz) = key;

        // Per-axis dirty range, indexed [X=0, Y=1, Z=2]. Used to clamp face/
        // edge iteration to only the cells the caller actually modified —
        // critical for mining (~5×5 dirty patch in a 31×31 face), modest for
        // wide ops like flatten that pass full chunk bounds.
        let dirty_min: [usize; 3] = [min_x, min_y, min_z];
        let dirty_max: [usize; 3] = [max_x, max_y, max_z];

        let faces: [(bool, (i32, i32, i32), usize, usize); 6] = [
            // (dirty touches this face?, neighbor key, coord in A, coord in B)
            (max_x >= cs, (cx + 1, cy, cz), cs, 0),     // +X
            (min_x == 0, (cx - 1, cy, cz), 0, cs),      // -X
            (max_y >= cs, (cx, cy + 1, cz), cs, 0),      // +Y
            (min_y == 0, (cx, cy - 1, cz), 0, cs),       // -Y
            (max_z >= cs, (cx, cy, cz + 1), cs, 0),      // +Z
            (min_z == 0, (cx, cy, cz - 1), 0, cs),       // -Z
        ];

        for (face_idx, &(touches, neighbor, coord_a, coord_b)) in faces.iter().enumerate() {
            if !touches {
                continue;
            }
            // Skip if neighbor not loaded
            if !density_fields.contains_key(&neighbor) {
                continue;
            }

            let axis = face_idx / 2; // 0=X, 1=Y, 2=Z

            // The two free dimensions for this face axis: u and v.
            // axis=0 (X face) → u=Y, v=Z. axis=1 (Y face) → u=X, v=Z. axis=2 (Z face) → u=X, v=Y.
            let (u_axis, v_axis) = match axis {
                0 => (1usize, 2usize),
                1 => (0usize, 2usize),
                _ => (0usize, 1usize),
            };
            let u_lo = dirty_min[u_axis];
            let u_hi = dirty_max[u_axis].min(cs);
            let v_lo = dirty_min[v_axis];
            let v_hi = dirty_max[v_axis].min(cs);

            // Iterate over the face plane, clamped to the projected dirty bounds.
            for u in u_lo..=u_hi {
                for v in v_lo..=v_hi {
                    let (ax, ay, az) = match axis {
                        0 => (coord_a, u, v),
                        1 => (u, coord_a, v),
                        _ => (u, v, coord_a),
                    };
                    let (bx, by, bz) = match axis {
                        0 => (coord_b, u, v),
                        1 => (u, coord_b, v),
                        _ => (u, v, coord_b),
                    };

                    let sample_a = density_fields[&key].get(ax, ay, az);
                    let sample_b = density_fields[&neighbor].get(bx, by, bz);

                    let (avg_d, avg_m) = average_boundary_voxel(sample_a, sample_b);

                    updates.push((key, ax, ay, az, avg_d, avg_m));
                    updates.push((neighbor, bx, by, bz, avg_d, avg_m));

                    if !dirty_keys.contains(&neighbor) {
                        extra_neighbors.insert(neighbor);
                    }
                }
            }
        }

        // --- Edge sync (12 edges): sync the 1D line of voxels shared with diagonal neighbors ---
        // Each edge is the intersection of two face boundaries.
        // dir_i, dir_j are +1 or -1 for the two boundary axes; the free axis iterates 0..=cs.
        let edge_defs: [(bool, bool, i32, i32, usize, usize, usize, usize, u8, u8); 12] = [
            // (touches_i, touches_j, di, dj, coord_a_i, coord_b_i, coord_a_j, coord_b_j, axis_i, axis_j)
            // X-Y edges (free axis = Z)
            (max_x >= cs, max_y >= cs, 1, 1, cs, 0, cs, 0, 0, 1),
            (max_x >= cs, min_y == 0,  1,-1, cs, 0,  0,cs, 0, 1),
            (min_x == 0,  max_y >= cs,-1, 1,  0,cs, cs, 0, 0, 1),
            (min_x == 0,  min_y == 0, -1,-1,  0,cs,  0,cs, 0, 1),
            // X-Z edges (free axis = Y)
            (max_x >= cs, max_z >= cs, 1, 1, cs, 0, cs, 0, 0, 2),
            (max_x >= cs, min_z == 0,  1,-1, cs, 0,  0,cs, 0, 2),
            (min_x == 0,  max_z >= cs,-1, 1,  0,cs, cs, 0, 0, 2),
            (min_x == 0,  min_z == 0, -1,-1,  0,cs,  0,cs, 0, 2),
            // Y-Z edges (free axis = X)
            (max_y >= cs, max_z >= cs, 1, 1, cs, 0, cs, 0, 1, 2),
            (max_y >= cs, min_z == 0,  1,-1, cs, 0,  0,cs, 1, 2),
            (min_y == 0,  max_z >= cs,-1, 1,  0,cs, cs, 0, 1, 2),
            (min_y == 0,  min_z == 0, -1,-1,  0,cs,  0,cs, 1, 2),
        ];

        for &(touches_i, touches_j, di, dj, ca_i, cb_i, ca_j, cb_j, axis_i, axis_j) in &edge_defs {
            if !touches_i || !touches_j {
                continue;
            }
            let neighbor = match (axis_i, axis_j) {
                (0, 1) => (cx + di, cy + dj, cz),
                (0, 2) => (cx + di, cy, cz + dj),
                _      => (cx, cy + di, cz + dj), // (1, 2)
            };
            if !density_fields.contains_key(&neighbor) {
                continue;
            }
            // Free axis is the one that's neither axis_i nor axis_j: 0+1+2=3
            let free_axis = 3 - axis_i - axis_j;
            // Clamp the 1D edge sweep to the dirty range on the free axis —
            // same rationale as the face clamp above.
            let t_lo = dirty_min[free_axis as usize];
            let t_hi = dirty_max[free_axis as usize].min(cs);
            for t in t_lo..=t_hi {
                let (ax, ay, az) = {
                    let mut c = [0usize; 3];
                    c[axis_i as usize] = ca_i;
                    c[axis_j as usize] = ca_j;
                    c[free_axis as usize] = t;
                    (c[0], c[1], c[2])
                };
                let (bx, by, bz) = {
                    let mut c = [0usize; 3];
                    c[axis_i as usize] = cb_i;
                    c[axis_j as usize] = cb_j;
                    c[free_axis as usize] = t;
                    (c[0], c[1], c[2])
                };

                let sample_a = density_fields[&key].get(ax, ay, az);
                let sample_b = density_fields[&neighbor].get(bx, by, bz);
                let (avg_d, avg_m) = average_boundary_voxel(sample_a, sample_b);

                updates.push((key, ax, ay, az, avg_d, avg_m));
                updates.push((neighbor, bx, by, bz, avg_d, avg_m));

                if !dirty_keys.contains(&neighbor) {
                    extra_neighbors.insert(neighbor);
                }
            }
        }

        // --- Corner sync (8 corners): sync single voxel shared with diagonal corner neighbor ---
        let corner_defs: [(bool, bool, bool, i32, i32, i32, usize, usize, usize, usize, usize, usize); 8] = [
            // (touches_x, touches_y, touches_z, dx, dy, dz, ax, bx, ay, by, az, bz)
            (max_x >= cs, max_y >= cs, max_z >= cs,  1, 1, 1, cs, 0, cs, 0, cs, 0),
            (max_x >= cs, max_y >= cs, min_z == 0,   1, 1,-1, cs, 0, cs, 0,  0,cs),
            (max_x >= cs, min_y == 0,  max_z >= cs,  1,-1, 1, cs, 0,  0,cs, cs, 0),
            (max_x >= cs, min_y == 0,  min_z == 0,   1,-1,-1, cs, 0,  0,cs,  0,cs),
            (min_x == 0,  max_y >= cs, max_z >= cs, -1, 1, 1,  0,cs, cs, 0, cs, 0),
            (min_x == 0,  max_y >= cs, min_z == 0,  -1, 1,-1,  0,cs, cs, 0,  0,cs),
            (min_x == 0,  min_y == 0,  max_z >= cs, -1,-1, 1,  0,cs,  0,cs, cs, 0),
            (min_x == 0,  min_y == 0,  min_z == 0,  -1,-1,-1,  0,cs,  0,cs,  0,cs),
        ];

        for &(tx, ty, tz, dx, dy, dz, ax, bx, ay, by, az, bz) in &corner_defs {
            if !tx || !ty || !tz {
                continue;
            }
            let neighbor = (cx + dx, cy + dy, cz + dz);
            if !density_fields.contains_key(&neighbor) {
                continue;
            }

            let sample_a = density_fields[&key].get(ax, ay, az);
            let sample_b = density_fields[&neighbor].get(bx, by, bz);
            let (avg_d, avg_m) = average_boundary_voxel(sample_a, sample_b);

            updates.push((key, ax, ay, az, avg_d, avg_m));
            updates.push((neighbor, bx, by, bz, avg_d, avg_m));

            if !dirty_keys.contains(&neighbor) {
                extra_neighbors.insert(neighbor);
            }
        }
    }

    // Pass 2: apply all updates
    for (chunk_key, x, y, z, density, material) in updates {
        if let Some(field) = density_fields.get_mut(&chunk_key) {
            let sample = field.get_mut(x, y, z);
            sample.density = density;
            sample.material = material;
        }
    }

    // Build extra dirty entries for neighbors that weren't already dirty.
    let mut extra_dirty: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> =
        Vec::new();
    for neighbor in extra_neighbors {
        // Full chunk bounds (conservative — only boundary face was modified but remesh needs context)
        extra_dirty.push((neighbor, 0, 0, 0, cs, cs, cs));
    }

    extra_dirty
}
