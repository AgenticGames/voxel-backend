//! Pass 3: Seam completeness — missing seam quads between adjacent chunks.
//! Pass N5: Seam mesh quality — boundary edge count mismatch.

use std::collections::{HashMap, HashSet};

use crate::density::DensityField;
use crate::mesh::Mesh;

use super::{IssueType, IssueSeverity, ScanSeamData, WorldIssue};

/// For adjacent loaded chunk pairs: count boundary edges that need seam quads.
/// If boundary edges > 0 but no seam mesh covers them → SeamGap.
pub(super) fn pass_seam_completeness(
    seam_data: &HashMap<(i32, i32, i32), ScanSeamData>,
    base_meshes: &HashMap<(i32, i32, i32), Mesh>,
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
    issues: &mut Vec<WorldIssue>,
) {
    let loaded: HashSet<(i32, i32, i32)> = density_fields.keys().copied().collect();

    let faces: [(i32, i32, i32, &str, u8); 3] = [
        (1, 0, 0, "PosX", 0), // axis X
        (0, 1, 0, "PosY", 1), // axis Y
        (0, 0, 1, "PosZ", 2), // axis Z
    ];

    let mut sorted_chunks: Vec<_> = loaded.iter().copied().collect();
    sorted_chunks.sort();

    for &(cx, cy, cz) in &sorted_chunks {
        let seam = match seam_data.get(&(cx, cy, cz)) {
            Some(s) => s,
            None => continue,
        };

        for &(dx, dy, dz, face_name, _axis) in &faces {
            let neighbor = (cx + dx, cy + dy, cz + dz);
            if !loaded.contains(&neighbor) {
                continue;
            }

            // Count boundary edges on this face
            let gs = chunk_size;
            let boundary_edge_count: u32 = seam.boundary_edges.iter().filter(|(ek, _)| {
                if dx == 1 { ek.x() as usize == gs }
                else if dy == 1 { ek.y() as usize == gs }
                else { ek.z() as usize == gs }
            }).count() as u32;

            if boundary_edge_count == 0 {
                continue;
            }

            // Check if the combined mesh (base + seam) has triangles bridging the boundary.
            // If base_meshes exist for both chunks, we expect seam quads were generated.
            let has_base_a = base_meshes.contains_key(&(cx, cy, cz));
            let has_base_b = base_meshes.contains_key(&neighbor);

            if has_base_a && has_base_b {
                // Check if neighbor also has seam data (indicating seam quads could be generated)
                let neighbor_has_seam = seam_data.contains_key(&neighbor);
                if !neighbor_has_seam {
                    let cs = chunk_size as f32;
                    let pos = [
                        cx as f32 * cs + if dx == 1 { cs } else { cs * 0.5 },
                        cy as f32 * cs + if dy == 1 { cs } else { cs * 0.5 },
                        cz as f32 * cs + if dz == 1 { cs } else { cs * 0.5 },
                    ];
                    issues.push(WorldIssue {
                        issue_type: IssueType::SeamGap,
                        severity: IssueSeverity::Error,
                        position: pos,
                        chunk: [cx, cy, cz],
                        neighbor: Some([neighbor.0, neighbor.1, neighbor.2]),
                        face: Some(face_name.to_string()),
                        detail: format!(
                            "{} boundary edges with no neighbor seam data at {} boundary",
                            boundary_edge_count, face_name
                        ),
                        detail_a: boundary_edge_count,
                        detail_b: 0,
                    });
                }
            }
        }
    }
}

/// For adjacent chunk pairs: count boundary edges from each side.
/// If counts differ by >50% → SeamQualityIssue.
pub(super) fn pass_seam_mesh_quality(
    seam_data: &HashMap<(i32, i32, i32), ScanSeamData>,
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
    issues: &mut Vec<WorldIssue>,
) {
    let loaded: HashSet<(i32, i32, i32)> = density_fields.keys().copied().collect();

    let faces: [(i32, i32, i32, &str); 3] = [
        (1, 0, 0, "PosX"),
        (0, 1, 0, "PosY"),
        (0, 0, 1, "PosZ"),
    ];

    let mut sorted_chunks: Vec<_> = loaded.iter().copied().collect();
    sorted_chunks.sort();

    for &(cx, cy, cz) in &sorted_chunks {
        let seam_a = match seam_data.get(&(cx, cy, cz)) {
            Some(s) => s,
            None => continue,
        };

        for &(dx, dy, dz, face_name) in &faces {
            let neighbor = (cx + dx, cy + dy, cz + dz);
            if !loaded.contains(&neighbor) {
                continue;
            }

            let seam_b = match seam_data.get(&neighbor) {
                Some(s) => s,
                None => continue,
            };

            let gs = chunk_size;

            // Count boundary edges on the shared face from side A
            let count_a: u32 = seam_a.boundary_edges.iter().filter(|(ek, _)| {
                if dx == 1 { ek.x() as usize == gs }
                else if dy == 1 { ek.y() as usize == gs }
                else { ek.z() as usize == gs }
            }).count() as u32;

            // Count boundary edges on the shared face from side B (at coord 0)
            let count_b: u32 = seam_b.boundary_edges.iter().filter(|(ek, _)| {
                if dx == 1 { ek.x() == 0 }
                else if dy == 1 { ek.y() == 0 }
                else { ek.z() == 0 }
            }).count() as u32;

            if count_a == 0 && count_b == 0 {
                continue;
            }

            let max_count = count_a.max(count_b) as f32;
            let min_count = count_a.min(count_b) as f32;

            if max_count > 0.0 && (max_count - min_count) / max_count > 0.5 {
                let cs_f = chunk_size as f32;
                let pos = [
                    cx as f32 * cs_f + if dx == 1 { cs_f } else { cs_f * 0.5 },
                    cy as f32 * cs_f + if dy == 1 { cs_f } else { cs_f * 0.5 },
                    cz as f32 * cs_f + if dz == 1 { cs_f } else { cs_f * 0.5 },
                ];
                issues.push(WorldIssue {
                    issue_type: IssueType::SeamQualityIssue,
                    severity: IssueSeverity::Info,
                    position: pos,
                    chunk: [cx, cy, cz],
                    neighbor: Some([neighbor.0, neighbor.1, neighbor.2]),
                    face: Some(face_name.to_string()),
                    detail: format!(
                        "Boundary edge count mismatch at {} boundary: {} vs {}",
                        face_name, count_a, count_b
                    ),
                    detail_a: count_a,
                    detail_b: count_b,
                });
            }
        }
    }
}
