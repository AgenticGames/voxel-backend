//! Pass 5: Worm truncation — worm path exits loaded chunk set.
//! Pass N3: Worm carve verify — worm center/offsets still solid.

use std::collections::{HashMap, HashSet};

use crate::density::DensityField;

use super::{IssueType, IssueSeverity, ScanWormSegment, WorldIssue};

/// Each worm segment: check if its chunk is in the loaded set.
/// If not → WormTruncation.
pub(super) fn pass_worm_truncation(
    worm_paths: &[Vec<ScanWormSegment>],
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
    issues: &mut Vec<WorldIssue>,
) {
    let loaded: HashSet<(i32, i32, i32)> = density_fields.keys().copied().collect();
    let cs = chunk_size as f32;

    for (worm_idx, path) in worm_paths.iter().enumerate() {
        for (seg_idx, seg) in path.iter().enumerate() {
            let chunk = (
                (seg.position[0] / cs).floor() as i32,
                (seg.position[1] / cs).floor() as i32,
                (seg.position[2] / cs).floor() as i32,
            );

            if !loaded.contains(&chunk) {
                // Determine which face it exits through
                let face = if seg_idx > 0 {
                    let prev = &path[seg_idx - 1];
                    let dx = seg.position[0] - prev.position[0];
                    let dy = seg.position[1] - prev.position[1];
                    let dz = seg.position[2] - prev.position[2];
                    let abs_dx = dx.abs();
                    let abs_dy = dy.abs();
                    let abs_dz = dz.abs();
                    if abs_dx >= abs_dy && abs_dx >= abs_dz {
                        if dx > 0.0 { "PosX" } else { "NegX" }
                    } else if abs_dy >= abs_dz {
                        if dy > 0.0 { "PosY" } else { "NegY" }
                    } else {
                        if dz > 0.0 { "PosZ" } else { "NegZ" }
                    }
                } else {
                    "Unknown"
                };

                issues.push(WorldIssue {
                    issue_type: IssueType::WormTruncation,
                    severity: IssueSeverity::Warning,
                    position: seg.position,
                    chunk: [chunk.0, chunk.1, chunk.2],
                    neighbor: None,
                    face: Some(face.to_string()),
                    detail: format!(
                        "Worm {} segment {} exits loaded chunk set at {} boundary",
                        worm_idx, seg_idx, face
                    ),
                    detail_a: worm_idx as u32,
                    detail_b: seg_idx as u32,
                });
                break; // One issue per worm
            }
        }
    }
}

/// For each worm segment in loaded chunks: sample density at center + 6 axis
/// offsets at 80% radius. If solid → WormCarveFailure.
pub(super) fn pass_worm_carve_verify(
    worm_paths: &[Vec<ScanWormSegment>],
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
    issues: &mut Vec<WorldIssue>,
) {
    let loaded: HashSet<(i32, i32, i32)> = density_fields.keys().copied().collect();
    let cs = chunk_size as f32;

    for (worm_idx, path) in worm_paths.iter().enumerate() {
        for (seg_idx, seg) in path.iter().enumerate() {
            let chunk = (
                (seg.position[0] / cs).floor() as i32,
                (seg.position[1] / cs).floor() as i32,
                (seg.position[2] / cs).floor() as i32,
            );

            if !loaded.contains(&chunk) {
                continue;
            }

            let density = &density_fields[&chunk];
            let local_x = seg.position[0] - chunk.0 as f32 * cs;
            let local_y = seg.position[1] - chunk.1 as f32 * cs;
            let local_z = seg.position[2] - chunk.2 as f32 * cs;

            // Check center
            let ix = (local_x as usize).min(chunk_size);
            let iy = (local_y as usize).min(chunk_size);
            let iz = (local_z as usize).min(chunk_size);

            let center_solid = density.get(ix, iy, iz).material.is_solid();

            // Check 6 axis offsets at 80% radius
            let r = seg.radius * 0.8;
            let offsets: [[f32; 3]; 6] = [
                [r, 0.0, 0.0], [-r, 0.0, 0.0],
                [0.0, r, 0.0], [0.0, -r, 0.0],
                [0.0, 0.0, r], [0.0, 0.0, -r],
            ];

            let mut solid_samples = if center_solid { 1u32 } else { 0 };
            for off in &offsets {
                let sx = local_x + off[0];
                let sy = local_y + off[1];
                let sz = local_z + off[2];

                if sx < 0.0 || sy < 0.0 || sz < 0.0
                    || sx >= cs || sy >= cs || sz >= cs
                {
                    continue;
                }

                let six = sx as usize;
                let siy = sy as usize;
                let siz = sz as usize;

                if density.get(six.min(chunk_size), siy.min(chunk_size), siz.min(chunk_size)).material.is_solid() {
                    solid_samples += 1;
                }
            }

            if solid_samples > 0 {
                issues.push(WorldIssue {
                    issue_type: IssueType::WormCarveFailure,
                    severity: IssueSeverity::Warning,
                    position: seg.position,
                    chunk: [chunk.0, chunk.1, chunk.2],
                    neighbor: None,
                    face: None,
                    detail: format!(
                        "Worm {} segment {} has {} solid samples at center/offsets (radius {:.1})",
                        worm_idx, seg_idx, solid_samples, seg.radius
                    ),
                    detail_a: worm_idx as u32,
                    detail_b: seg_idx as u32,
                });
                break; // One issue per worm
            }
        }
    }
}
