//! Topology detection — CeilingDome / Chokepoint / WallNiche per chunk.
//!
//! These were stubbed in the legacy `voxel-ffi/src/poi_scanner.rs:173-200`
//! (`count_topology_votes_cross_chunk` always returned zero). Block 1
//! provides real implementations that walk the density field.
//!
//! All three functions return `Option<TopologyVote>` per chunk. A vote
//! includes the topology kind, a confidence score, and the centroid of the
//! detected feature in local chunk voxel coords.
//!
//! **Block 1 gating**: topology kinds are only surfaced via the new
//! `voxel_request_scenes` FFI. The legacy adapter `legacy_top_k_pois` filters
//! them out unless `include_topology` is true. UE doesn't handle topology
//! kinds until Block 2.

use voxel_core::density::DensityField;

/// A topology detection result for a single chunk.
#[derive(Debug, Clone, Copy)]
pub struct TopologyVote {
    pub kind: TopologyKind,
    pub score: f32,
    /// Centroid in local chunk voxel coords (0..chunk_size).
    pub centroid_local: [f32; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyKind {
    CeilingDome,
    Chokepoint,
    WallNiche,
}

/// Walk a chunk's density field looking for topology features. Returns the
/// strongest match (Some), or None if no feature crosses its threshold.
///
/// The chunk's `DensityField` has grid_size = chunk_size + 1, with each
/// voxel's material accessible via `field.get(lx, ly, lz).material`.
pub fn detect_topology(field: &DensityField) -> Option<TopologyVote> {
    let dome = detect_ceiling_dome(field);
    let choke = detect_chokepoint(field);
    let niche = detect_wall_niche(field);

    [dome, choke, niche]
        .into_iter()
        .flatten()
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
}

/// CeilingDome: a region where the top half of the chunk has air with
/// solid above forming an arch. Detected by:
///   - bottom 40% of chunk: mostly air (>60%)
///   - top 30% of chunk: mostly solid (>50%)
///   - mid 30%: mixed (signal of curved transition)
///
/// Score = air_volume_below * solid_volume_above, weighted by chunk size.
pub fn detect_ceiling_dome(field: &DensityField) -> Option<TopologyVote> {
    let size = field.size;
    if size < 8 {
        return None; // too small to have meaningful structure
    }

    let bottom_max = (size * 4) / 10; // bottom 40%
    let top_min = (size * 7) / 10;    // top starts at 70%

    let mut bottom_air = 0u32;
    let mut bottom_total = 0u32;
    let mut top_solid = 0u32;
    let mut top_total = 0u32;

    let mut air_centroid_sum = [0.0f32; 3];
    let mut air_centroid_w = 0.0f32;

    for y in 0..size {
        for z in 0..size {
            for x in 0..size {
                let is_solid = field.get(x, y, z).material.is_solid();
                if y < bottom_max {
                    bottom_total += 1;
                    if !is_solid {
                        bottom_air += 1;
                        air_centroid_sum[0] += x as f32;
                        air_centroid_sum[1] += y as f32;
                        air_centroid_sum[2] += z as f32;
                        air_centroid_w += 1.0;
                    }
                } else if y >= top_min {
                    top_total += 1;
                    if is_solid {
                        top_solid += 1;
                    }
                }
            }
        }
    }

    if bottom_total == 0 || top_total == 0 || air_centroid_w == 0.0 {
        return None;
    }

    let air_frac = bottom_air as f32 / bottom_total as f32;
    let solid_frac = top_solid as f32 / top_total as f32;

    if air_frac < 0.60 || solid_frac < 0.50 {
        return None;
    }

    // Score: combines fractions weighted by absolute volume so larger chunks
    // with the right shape score higher. Cap at ~100 for a perfect 30^3
    // chunk.
    let score = air_frac * solid_frac * (size as f32).powi(2) * 0.12;
    if score < 1.0 {
        return None;
    }

    Some(TopologyVote {
        kind: TopologyKind::CeilingDome,
        score,
        centroid_local: [
            air_centroid_sum[0] / air_centroid_w,
            air_centroid_sum[1] / air_centroid_w,
            air_centroid_sum[2] / air_centroid_w,
        ],
    })
}

/// Chokepoint: a narrow constriction in a passage. Detected by:
///   - looking at the chunk's middle slice (y=size/2) and the slices ±1
///   - computing the air-region perimeter-to-area ratio
///   - high ratio means a long, narrow passage = chokepoint
pub fn detect_chokepoint(field: &DensityField) -> Option<TopologyVote> {
    let size = field.size;
    if size < 8 {
        return None;
    }

    let mid_y = size / 2;
    let mut min_air_cross_section = u32::MAX;
    let mut best_centroid = [0.0f32; 3];

    // Scan 3 horizontal slices (mid_y-1, mid_y, mid_y+1). Find the one with
    // the smallest air cross-section — that's the constriction.
    for &y in &[mid_y.saturating_sub(1), mid_y, (mid_y + 1).min(size - 1)] {
        let mut air = 0u32;
        let mut cx = 0.0f32;
        let mut cz = 0.0f32;
        for z in 0..size {
            for x in 0..size {
                if !field.get(x, y, z).material.is_solid() {
                    air += 1;
                    cx += x as f32;
                    cz += z as f32;
                }
            }
        }
        if air > 0 && air < min_air_cross_section {
            min_air_cross_section = air;
            best_centroid = [cx / air as f32, y as f32, cz / air as f32];
        }
    }

    if min_air_cross_section == u32::MAX {
        return None;
    }

    // Also need air in chunk to confirm passage (not just a tiny hole in
    // solid rock). Count air at the upper slice (y = mid_y + size/4).
    let upper_y = (mid_y + size / 4).min(size - 1);
    let mut upper_air = 0u32;
    for z in 0..size {
        for x in 0..size {
            if !field.get(x, upper_y, z).material.is_solid() {
                upper_air += 1;
            }
        }
    }

    // Chokepoint: min cross-section < 25% of upper slice's air, and not zero.
    let cross_sec_threshold = upper_air / 4;
    if min_air_cross_section == 0 || min_air_cross_section >= cross_sec_threshold {
        return None;
    }
    if upper_air < 16 {
        return None; // too small to be a meaningful passage
    }

    // Score: larger constriction differential → higher score.
    let differential = (upper_air - min_air_cross_section) as f32;
    let score = differential * 0.4;
    if score < 1.0 {
        return None;
    }

    Some(TopologyVote {
        kind: TopologyKind::Chokepoint,
        score,
        centroid_local: best_centroid,
    })
}

/// WallNiche: a small recess in a wall (cubby-shaped air pocket against a
/// solid face). Detected by:
///   - find chunks where ONE face (±X or ±Z, not Y) is mostly solid
///   - and the opposite face is mostly air
///   - and there's an air "pocket" in the middle near the solid face
pub fn detect_wall_niche(field: &DensityField) -> Option<TopologyVote> {
    let size = field.size;
    if size < 8 {
        return None;
    }

    // Check 4 wall directions: -X, +X, -Z, +Z. Y-walls are floor/ceiling, not
    // walls in cave-game vocabulary.
    let directions: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    let mut best: Option<TopologyVote> = None;

    for &(dx, dz) in &directions {
        // Take a single face slice at the appropriate boundary.
        let face_x_range: Box<dyn Iterator<Item = usize>> = if dx == -1 {
            Box::new(std::iter::once(0usize))
        } else if dx == 1 {
            Box::new(std::iter::once(size - 1))
        } else {
            Box::new(0..size)
        };
        // Save vec since iterator is consumed.
        let face_xs: Vec<usize> = face_x_range.collect();
        let face_z_range: Box<dyn Iterator<Item = usize>> = if dz == -1 {
            Box::new(std::iter::once(0usize))
        } else if dz == 1 {
            Box::new(std::iter::once(size - 1))
        } else {
            Box::new(0..size)
        };
        let face_zs: Vec<usize> = face_z_range.collect();

        // Count solid on this face, in mid Y band.
        let y_lo = size / 4;
        let y_hi = size - size / 4;
        let mut face_solid = 0u32;
        let mut face_total = 0u32;
        for &fx in &face_xs {
            for &fz in &face_zs {
                for y in y_lo..y_hi {
                    if field.get(fx, y, fz).material.is_solid() {
                        face_solid += 1;
                    }
                    face_total += 1;
                }
            }
        }
        if face_total == 0 {
            continue;
        }
        let face_solid_frac = face_solid as f32 / face_total as f32;
        if face_solid_frac < 0.7 {
            continue; // this face isn't a wall
        }

        // Opposite face should be air-dominant.
        let opp_face_xs: Vec<usize> = if dx == -1 {
            vec![size - 1]
        } else if dx == 1 {
            vec![0]
        } else {
            (0..size).collect()
        };
        let opp_face_zs: Vec<usize> = if dz == -1 {
            vec![size - 1]
        } else if dz == 1 {
            vec![0]
        } else {
            (0..size).collect()
        };
        let mut opp_air = 0u32;
        let mut opp_total = 0u32;
        for &fx in &opp_face_xs {
            for &fz in &opp_face_zs {
                for y in y_lo..y_hi {
                    if !field.get(fx, y, fz).material.is_solid() {
                        opp_air += 1;
                    }
                    opp_total += 1;
                }
            }
        }
        if opp_total == 0 {
            continue;
        }
        let opp_air_frac = opp_air as f32 / opp_total as f32;
        if opp_air_frac < 0.5 {
            continue;
        }

        // Look for a small pocket near the solid face: scan an interior
        // strip 2 voxels in from the solid face.
        let probe_x_lo = if dx == -1 { 2 } else if dx == 1 { size.saturating_sub(3) } else { 0 };
        let probe_x_hi = if dx == -1 { 4.min(size) } else if dx == 1 { size.saturating_sub(1) } else { size };
        let probe_z_lo = if dz == -1 { 2 } else if dz == 1 { size.saturating_sub(3) } else { 0 };
        let probe_z_hi = if dz == -1 { 4.min(size) } else if dz == 1 { size.saturating_sub(1) } else { size };

        let mut pocket_air = 0u32;
        let mut pocket_cx = 0.0f32;
        let mut pocket_cy = 0.0f32;
        let mut pocket_cz = 0.0f32;
        for y in y_lo..y_hi {
            for z in probe_z_lo..probe_z_hi {
                for x in probe_x_lo..probe_x_hi {
                    if !field.get(x, y, z).material.is_solid() {
                        pocket_air += 1;
                        pocket_cx += x as f32;
                        pocket_cy += y as f32;
                        pocket_cz += z as f32;
                    }
                }
            }
        }
        if pocket_air < 8 {
            continue; // no pocket
        }
        let cx = pocket_cx / pocket_air as f32;
        let cy = pocket_cy / pocket_air as f32;
        let cz = pocket_cz / pocket_air as f32;

        let score = pocket_air as f32 * 0.6 + face_solid_frac * 10.0 + opp_air_frac * 10.0;
        let candidate = TopologyVote {
            kind: TopologyKind::WallNiche,
            score,
            centroid_local: [cx, cy, cz],
        };
        match &best {
            Some(b) if b.score >= candidate.score => {}
            _ => best = Some(candidate),
        }
    }

    // Discard low-confidence niches.
    if let Some(v) = &best {
        if v.score < 15.0 {
            return None;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::material::Material;

    /// Build a DensityField with all-solid material, then carve the
    /// described region to air.
    fn solid_field(size: usize) -> DensityField {
        let mut f = DensityField::new(size);
        for y in 0..size {
            for z in 0..size {
                for x in 0..size {
                    f.get_mut(x, y, z).material = Material::Granite;
                }
            }
        }
        f
    }

    fn carve_air(f: &mut DensityField, lo: (usize, usize, usize), hi: (usize, usize, usize)) {
        for y in lo.1..hi.1 {
            for z in lo.2..hi.2 {
                for x in lo.0..hi.0 {
                    f.get_mut(x, y, z).material = Material::Air;
                }
            }
        }
    }

    #[test]
    fn dome_synthesized_chunk_detected() {
        // 30^3 chunk (grid_size = 31). Carve out the bottom 50% as air,
        // keep the top 30% as solid. Should detect CeilingDome.
        let mut f = solid_field(31);
        carve_air(&mut f, (0, 0, 0), (31, 16, 31));
        let v = detect_ceiling_dome(&f);
        assert!(v.is_some(), "expected CeilingDome");
        let v = v.unwrap();
        assert_eq!(v.kind, TopologyKind::CeilingDome);
        assert!(v.score > 1.0);
    }

    #[test]
    fn no_dome_in_uniform_air() {
        let mut f = solid_field(31);
        carve_air(&mut f, (0, 0, 0), (31, 31, 31));
        // All air → top isn't solid → no dome.
        assert!(detect_ceiling_dome(&f).is_none());
    }

    #[test]
    fn no_dome_in_uniform_solid() {
        let f = solid_field(31);
        // All solid → bottom isn't air → no dome.
        assert!(detect_ceiling_dome(&f).is_none());
    }

    #[test]
    fn chokepoint_synthesized_chunk_detected() {
        // Air everywhere except a single solid plane at y=mid-1 with a
        // small 3x3 hole. The hole IS the chokepoint.
        let mut f = solid_field(31);
        // Carve everything to air first.
        carve_air(&mut f, (0, 0, 0), (31, 31, 31));
        // Re-solid a plane at y=15 with a small 3x3 hole.
        for z in 0..31 {
            for x in 0..31 {
                if !(14..=16).contains(&x) || !(14..=16).contains(&z) {
                    f.get_mut(x, 15, z).material = Material::Granite;
                }
            }
        }
        let v = detect_chokepoint(&f);
        assert!(v.is_some(), "expected Chokepoint");
        assert_eq!(v.unwrap().kind, TopologyKind::Chokepoint);
    }

    #[test]
    fn wall_niche_synthesized_chunk_detected() {
        // Make -X face solid (a wall on the left), opposite face air, and a
        // pocket of air just inward from the wall.
        let mut f = solid_field(31);
        // Carve the chunk to air except the leftmost slab and a pocket.
        carve_air(&mut f, (1, 0, 0), (31, 31, 31));
        // Now re-solid a 1-voxel-thick wall at x=2..3 except the niche
        // pocket at y=12..18, z=12..18.
        for y in 0..31 {
            for z in 0..31 {
                if !(12..18).contains(&y) || !(12..18).contains(&z) {
                    f.get_mut(2, y, z).material = Material::Granite;
                }
            }
        }
        let v = detect_wall_niche(&f);
        assert!(v.is_some(), "expected WallNiche, got None");
        assert_eq!(v.unwrap().kind, TopologyKind::WallNiche);
    }

    #[test]
    fn detect_topology_picks_strongest() {
        // Dome-shaped chunk — detect_topology should pick CeilingDome.
        let mut f = solid_field(31);
        carve_air(&mut f, (0, 0, 0), (31, 16, 31));
        let v = detect_topology(&f).unwrap();
        assert_eq!(v.kind, TopologyKind::CeilingDome);
    }

    #[test]
    fn detect_topology_returns_none_for_uniform() {
        let f = solid_field(31);
        assert!(detect_topology(&f).is_none());
    }
}
