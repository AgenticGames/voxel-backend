//! Pass N1: Winding consistency — inward-facing triangles.
//! Pass N2: Triangle quality — degenerate and stretched triangles.
//! Pass N4: Self intersection — overlapping triangle AABBs.

use std::collections::HashMap;

use crate::mesh::Mesh;

use super::{IssueType, IssueSeverity, ScanConfig, WorldIssue};

/// Compute solid centroid. For each triangle: cross product normal, dot with
/// (face_center - centroid). If >30% inward-facing → WindingInconsistency.
pub(super) fn pass_winding_consistency(
    base_meshes: &HashMap<(i32, i32, i32), Mesh>,
    chunk_size: usize,
    issues: &mut Vec<WorldIssue>,
) {
    let cs = chunk_size as f32;

    let mut sorted_chunks: Vec<_> = base_meshes.keys().copied().collect();
    sorted_chunks.sort();

    for &(cx, cy, cz) in &sorted_chunks {
        let mesh = &base_meshes[&(cx, cy, cz)];
        if mesh.triangles.len() < 4 {
            continue;
        }

        // Compute mesh centroid
        let mut centroid = glam::Vec3::ZERO;
        for v in &mesh.vertices {
            centroid += v.position;
        }
        centroid /= mesh.vertices.len() as f32;

        let mut inward_count = 0u32;
        let total = mesh.triangles.len() as u32;

        for tri in &mesh.triangles {
            let v0 = mesh.vertices[tri.indices[0] as usize].position;
            let v1 = mesh.vertices[tri.indices[1] as usize].position;
            let v2 = mesh.vertices[tri.indices[2] as usize].position;

            let face_center = (v0 + v1 + v2) / 3.0;
            let normal = (v1 - v0).cross(v2 - v0);
            let to_centroid = centroid - face_center;

            if normal.dot(to_centroid) > 0.0 {
                inward_count += 1;
            }
        }

        let inward_ratio = inward_count as f32 / total as f32;
        if inward_ratio > 0.3 {
            let world_pos = [
                centroid.x + cx as f32 * cs,
                centroid.y + cy as f32 * cs,
                centroid.z + cz as f32 * cs,
            ];
            issues.push(WorldIssue {
                issue_type: IssueType::WindingInconsistency,
                severity: IssueSeverity::Warning,
                position: world_pos,
                chunk: [cx, cy, cz],
                neighbor: None,
                face: None,
                detail: format!(
                    "{}/{} triangles ({:.0}%) face inward (winding inconsistency)",
                    inward_count, total, inward_ratio * 100.0
                ),
                detail_a: inward_count,
                detail_b: total,
            });
        }
    }
}

/// Area via cross product. Small area → DegenerateTriangle. Long edges →
/// StretchedTriangle.
pub(super) fn pass_triangle_quality(
    base_meshes: &HashMap<(i32, i32, i32), Mesh>,
    chunk_size: usize,
    config: &ScanConfig,
    issues: &mut Vec<WorldIssue>,
) {
    let cs = chunk_size as f32;

    let mut sorted_chunks: Vec<_> = base_meshes.keys().copied().collect();
    sorted_chunks.sort();

    for &(cx, cy, cz) in &sorted_chunks {
        let mesh = &base_meshes[&(cx, cy, cz)];
        if mesh.triangles.is_empty() {
            continue;
        }

        let mut degen_count = 0u32;
        let mut stretched_count = 0u32;
        let mut degen_pos_sum = [0.0f32; 3];
        let mut stretch_pos_sum = [0.0f32; 3];

        for tri in &mesh.triangles {
            let v0 = mesh.vertices[tri.indices[0] as usize].position;
            let v1 = mesh.vertices[tri.indices[1] as usize].position;
            let v2 = mesh.vertices[tri.indices[2] as usize].position;

            let cross = (v1 - v0).cross(v2 - v0);
            let area = cross.length() * 0.5;

            if area < config.min_triangle_area {
                degen_count += 1;
                let center = (v0 + v1 + v2) / 3.0;
                degen_pos_sum[0] += center.x;
                degen_pos_sum[1] += center.y;
                degen_pos_sum[2] += center.z;
            }

            // Check edge lengths
            let e0 = (v1 - v0).length();
            let e1 = (v2 - v1).length();
            let e2 = (v0 - v2).length();
            let max_len = e0.max(e1).max(e2);

            if max_len > config.max_edge_length {
                stretched_count += 1;
                let center = (v0 + v1 + v2) / 3.0;
                stretch_pos_sum[0] += center.x;
                stretch_pos_sum[1] += center.y;
                stretch_pos_sum[2] += center.z;
            }
        }

        if degen_count > 0 {
            let pos = [
                degen_pos_sum[0] / degen_count as f32 + cx as f32 * cs,
                degen_pos_sum[1] / degen_count as f32 + cy as f32 * cs,
                degen_pos_sum[2] / degen_count as f32 + cz as f32 * cs,
            ];
            issues.push(WorldIssue {
                issue_type: IssueType::DegenerateTriangle,
                severity: IssueSeverity::Info,
                position: pos,
                chunk: [cx, cy, cz],
                neighbor: None,
                face: None,
                detail: format!(
                    "{} triangles with area < {}",
                    degen_count, config.min_triangle_area
                ),
                detail_a: degen_count,
                detail_b: mesh.triangles.len() as u32,
            });
        }

        if stretched_count > 0 {
            let pos = [
                stretch_pos_sum[0] / stretched_count as f32 + cx as f32 * cs,
                stretch_pos_sum[1] / stretched_count as f32 + cy as f32 * cs,
                stretch_pos_sum[2] / stretched_count as f32 + cz as f32 * cs,
            ];
            issues.push(WorldIssue {
                issue_type: IssueType::StretchedTriangle,
                severity: IssueSeverity::Info,
                position: pos,
                chunk: [cx, cy, cz],
                neighbor: None,
                face: None,
                detail: format!(
                    "{} triangles with edge length > {:.1}",
                    stretched_count, config.max_edge_length
                ),
                detail_a: stretched_count,
                detail_b: mesh.triangles.len() as u32,
            });
        }
    }
}

/// Build per-triangle AABBs. For non-adjacent pairs with overlapping AABBs →
/// SelfIntersection. Skip chunks over tri_limit.
pub(super) fn pass_self_intersection(
    base_meshes: &HashMap<(i32, i32, i32), Mesh>,
    chunk_size: usize,
    config: &ScanConfig,
    issues: &mut Vec<WorldIssue>,
) {
    let cs = chunk_size as f32;

    let mut sorted_chunks: Vec<_> = base_meshes.keys().copied().collect();
    sorted_chunks.sort();

    for &(cx, cy, cz) in &sorted_chunks {
        let mesh = &base_meshes[&(cx, cy, cz)];
        if mesh.triangles.len() > config.self_intersection_tri_limit as usize {
            continue;
        }
        if mesh.triangles.len() < 2 {
            continue;
        }

        // Build per-triangle AABBs
        struct TriAabb {
            min: [f32; 3],
            max: [f32; 3],
            indices: [u32; 3],
        }

        let aabbs: Vec<TriAabb> = mesh.triangles.iter().map(|tri| {
            let v0 = mesh.vertices[tri.indices[0] as usize].position;
            let v1 = mesh.vertices[tri.indices[1] as usize].position;
            let v2 = mesh.vertices[tri.indices[2] as usize].position;
            TriAabb {
                min: [
                    v0.x.min(v1.x).min(v2.x),
                    v0.y.min(v1.y).min(v2.y),
                    v0.z.min(v1.z).min(v2.z),
                ],
                max: [
                    v0.x.max(v1.x).max(v2.x),
                    v0.y.max(v1.y).max(v2.y),
                    v0.z.max(v1.z).max(v2.z),
                ],
                indices: tri.indices,
            }
        }).collect();

        let mut intersect_count = 0u32;
        let mut intersect_pos_sum = [0.0f32; 3];

        for i in 0..aabbs.len() {
            for j in (i + 1)..aabbs.len() {
                let a = &aabbs[i];
                let b = &aabbs[j];

                // Skip adjacent triangles (shared vertex)
                let shared = a.indices.iter().any(|ai| b.indices.contains(ai));
                if shared {
                    continue;
                }

                // AABB overlap test
                let overlaps = a.min[0] <= b.max[0] && a.max[0] >= b.min[0]
                    && a.min[1] <= b.max[1] && a.max[1] >= b.min[1]
                    && a.min[2] <= b.max[2] && a.max[2] >= b.min[2];

                if overlaps {
                    intersect_count += 1;
                    let mid = [
                        (a.min[0] + a.max[0] + b.min[0] + b.max[0]) * 0.25,
                        (a.min[1] + a.max[1] + b.min[1] + b.max[1]) * 0.25,
                        (a.min[2] + a.max[2] + b.min[2] + b.max[2]) * 0.25,
                    ];
                    intersect_pos_sum[0] += mid[0];
                    intersect_pos_sum[1] += mid[1];
                    intersect_pos_sum[2] += mid[2];
                }
            }
        }

        if intersect_count > 0 {
            let world_pos = [
                intersect_pos_sum[0] / intersect_count as f32 + cx as f32 * cs,
                intersect_pos_sum[1] / intersect_count as f32 + cy as f32 * cs,
                intersect_pos_sum[2] / intersect_count as f32 + cz as f32 * cs,
            ];
            issues.push(WorldIssue {
                issue_type: IssueType::SelfIntersection,
                severity: IssueSeverity::Warning,
                position: world_pos,
                chunk: [cx, cy, cz],
                neighbor: None,
                face: None,
                detail: format!(
                    "{} non-adjacent triangle pairs with overlapping AABBs",
                    intersect_count
                ),
                detail_a: intersect_count,
                detail_b: mesh.triangles.len() as u32,
            });
        }
    }
}
