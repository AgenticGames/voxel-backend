//! Pass 1: Density seam check — cross-chunk density sign mismatch detection.

use std::collections::{HashMap, HashSet};

use crate::density::DensityField;

use super::{IssueType, IssueSeverity, ScanConfig, WorldIssue};

/// For each loaded chunk pair sharing a face, compare density signs at the
/// shared sample layer. Mismatched signs → DensityDiscontinuity.
pub(super) fn pass_density_seam(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
    config: &ScanConfig,
    issues: &mut Vec<WorldIssue>,
) {
    let loaded: HashSet<(i32, i32, i32)> = density_fields.keys().copied().collect();

    // 3 positive-direction faces to check (avoids double-counting)
    let faces: [(i32, i32, i32, &str); 3] = [
        (1, 0, 0, "PosX"),
        (0, 1, 0, "PosY"),
        (0, 0, 1, "PosZ"),
    ];

    // Use sorted iteration for determinism
    let mut sorted_chunks: Vec<_> = loaded.iter().copied().collect();
    sorted_chunks.sort();

    for &(cx, cy, cz) in &sorted_chunks {
        let density_a = &density_fields[&(cx, cy, cz)];

        for &(dx, dy, dz, face_name) in &faces {
            let neighbor = (cx + dx, cy + dy, cz + dz);
            if !loaded.contains(&neighbor) {
                continue;
            }
            let density_b = &density_fields[&neighbor];

            let mut mismatch_count = 0u32;
            let gs = chunk_size; // grid_size = chunk_size, sample grid is chunk_size+1

            // Compare the shared sample layer
            if dx == 1 {
                // A's x=chunk_size vs B's x=0
                for z in 0..=gs {
                    for y in 0..=gs {
                        let sign_a = density_a.get(gs, y, z).density <= 0.0; // air
                        let sign_b = density_b.get(0, y, z).density <= 0.0;
                        if sign_a != sign_b {
                            mismatch_count += 1;
                        }
                    }
                }
            } else if dy == 1 {
                for z in 0..=gs {
                    for x in 0..=gs {
                        let sign_a = density_a.get(x, gs, z).density <= 0.0;
                        let sign_b = density_b.get(x, 0, z).density <= 0.0;
                        if sign_a != sign_b {
                            mismatch_count += 1;
                        }
                    }
                }
            } else {
                // dz == 1
                for y in 0..=gs {
                    for x in 0..=gs {
                        let sign_a = density_a.get(x, y, gs).density <= 0.0;
                        let sign_b = density_b.get(x, y, 0).density <= 0.0;
                        if sign_a != sign_b {
                            mismatch_count += 1;
                        }
                    }
                }
            }

            if mismatch_count > 0 {
                let pos = [
                    (cx as f32 + 0.5) * chunk_size as f32,
                    (cy as f32 + 0.5) * chunk_size as f32,
                    (cz as f32 + 0.5) * chunk_size as f32,
                ];
                issues.push(WorldIssue {
                    issue_type: IssueType::DensityDiscontinuity,
                    severity: IssueSeverity::Error,
                    position: pos,
                    chunk: [cx, cy, cz],
                    neighbor: Some([neighbor.0, neighbor.1, neighbor.2]),
                    face: Some(face_name.to_string()),
                    detail: format!(
                        "{} density sign mismatches at {} boundary",
                        mismatch_count, face_name
                    ),
                    detail_a: mismatch_count,
                    detail_b: 0,
                });
            }

            // E1: Sub-voxel interpolation check
            if config.density_subsample_count > 0 {
                let sub_count = config.density_subsample_count;
                let mut sub_mismatch = 0u32;
                // Interpolate between adjacent grid points at the boundary
                if dx == 1 {
                    for z in 0..gs {
                        for y in 0..gs {
                            for s in 1..=sub_count {
                                let t = s as f32 / (sub_count + 1) as f32;
                                let da0 = density_a.get(gs, y, z).density;
                                let da1 = density_a.get(gs, y + 1, z).density;
                                let db0 = density_b.get(0, y, z).density;
                                let db1 = density_b.get(0, y + 1, z).density;
                                let interp_a = da0 * (1.0 - t) + da1 * t;
                                let interp_b = db0 * (1.0 - t) + db1 * t;
                                if (interp_a <= 0.0) != (interp_b <= 0.0) {
                                    sub_mismatch += 1;
                                }
                            }
                        }
                    }
                } else if dy == 1 {
                    for z in 0..gs {
                        for x in 0..gs {
                            for s in 1..=sub_count {
                                let t = s as f32 / (sub_count + 1) as f32;
                                let da0 = density_a.get(x, gs, z).density;
                                let da1 = density_a.get(x + 1, gs, z).density;
                                let db0 = density_b.get(x, 0, z).density;
                                let db1 = density_b.get(x + 1, 0, z).density;
                                let interp_a = da0 * (1.0 - t) + da1 * t;
                                let interp_b = db0 * (1.0 - t) + db1 * t;
                                if (interp_a <= 0.0) != (interp_b <= 0.0) {
                                    sub_mismatch += 1;
                                }
                            }
                        }
                    }
                } else {
                    for y in 0..gs {
                        for x in 0..gs {
                            for s in 1..=sub_count {
                                let t = s as f32 / (sub_count + 1) as f32;
                                let da0 = density_a.get(x, y, gs).density;
                                let da1 = density_a.get(x + 1, y, gs).density;
                                let db0 = density_b.get(x, y, 0).density;
                                let db1 = density_b.get(x + 1, y, 0).density;
                                let interp_a = da0 * (1.0 - t) + da1 * t;
                                let interp_b = db0 * (1.0 - t) + db1 * t;
                                if (interp_a <= 0.0) != (interp_b <= 0.0) {
                                    sub_mismatch += 1;
                                }
                            }
                        }
                    }
                }
                if sub_mismatch > 0 {
                    let pos = [
                        (cx as f32 + 0.5) * chunk_size as f32,
                        (cy as f32 + 0.5) * chunk_size as f32,
                        (cz as f32 + 0.5) * chunk_size as f32,
                    ];
                    issues.push(WorldIssue {
                        issue_type: IssueType::DensityDiscontinuity,
                        severity: IssueSeverity::Warning,
                        position: pos,
                        chunk: [cx, cy, cz],
                        neighbor: Some([neighbor.0, neighbor.1, neighbor.2]),
                        face: Some(face_name.to_string()),
                        detail: format!(
                            "{} sub-voxel density sign mismatches at {} boundary",
                            sub_mismatch, face_name
                        ),
                        detail_a: sub_mismatch,
                        detail_b: config.density_subsample_count,
                    });
                }
            }
        }
    }
}
