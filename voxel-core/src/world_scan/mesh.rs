//! Pass 2: Mesh topology (holes, non-manifold edges).
//! Pass E2: Raymarch holes.
//! Pass E3: Mesh-density alignment via hermite data.

use std::collections::HashMap;

use crate::density::DensityField;
use crate::mesh::Mesh;

use super::{IssueType, IssueSeverity, ScanConfig, WorldIssue};

/// Per-mesh edge-count analysis. Interior boundary edges (count=1, not on
/// chunk face) → MeshHole. Count>=3 → NonManifoldEdge.
pub(super) fn pass_mesh_topology(
    base_meshes: &HashMap<(i32, i32, i32), Mesh>,
    chunk_size: usize,
    issues: &mut Vec<WorldIssue>,
) {
    let mut sorted_chunks: Vec<_> = base_meshes.keys().copied().collect();
    sorted_chunks.sort();

    for &(cx, cy, cz) in &sorted_chunks {
        let mesh = &base_meshes[&(cx, cy, cz)];
        if mesh.triangles.is_empty() {
            continue;
        }

        // Count edge usage: edge key = sorted pair of vertex indices
        let mut edge_counts: HashMap<(u32, u32), u32> = HashMap::new();
        for tri in &mesh.triangles {
            let i = tri.indices;
            for &(a, b) in &[(i[0], i[1]), (i[1], i[2]), (i[2], i[0])] {
                let key = if a < b { (a, b) } else { (b, a) };
                *edge_counts.entry(key).or_insert(0) += 1;
            }
        }

        let cs = chunk_size as f32;
        let boundary_margin = 0.1;

        let mut hole_edges = 0u32;
        let mut non_manifold_edges = 0u32;
        let mut hole_pos_sum = [0.0f32; 3];
        let mut nm_pos_sum = [0.0f32; 3];

        for (&(va, vb), &count) in &edge_counts {
            if count == 1 {
                // Check if this edge is on a chunk boundary face
                let pa = mesh.vertices[va as usize].position;
                let pb = mesh.vertices[vb as usize].position;
                let mid_x = (pa.x + pb.x) * 0.5;
                let mid_y = (pa.y + pb.y) * 0.5;
                let mid_z = (pa.z + pb.z) * 0.5;

                let on_boundary = mid_x < boundary_margin
                    || mid_x > cs - boundary_margin
                    || mid_y < boundary_margin
                    || mid_y > cs - boundary_margin
                    || mid_z < boundary_margin
                    || mid_z > cs - boundary_margin;

                if !on_boundary {
                    hole_edges += 1;
                    hole_pos_sum[0] += mid_x;
                    hole_pos_sum[1] += mid_y;
                    hole_pos_sum[2] += mid_z;
                }
            } else if count >= 3 {
                let pa = mesh.vertices[va as usize].position;
                let pb = mesh.vertices[vb as usize].position;
                non_manifold_edges += 1;
                let mid_x = (pa.x + pb.x) * 0.5;
                let mid_y = (pa.y + pb.y) * 0.5;
                let mid_z = (pa.z + pb.z) * 0.5;
                nm_pos_sum[0] += mid_x;
                nm_pos_sum[1] += mid_y;
                nm_pos_sum[2] += mid_z;
            }
        }

        if hole_edges > 0 {
            let avg = [
                hole_pos_sum[0] / hole_edges as f32 + cx as f32 * cs,
                hole_pos_sum[1] / hole_edges as f32 + cy as f32 * cs,
                hole_pos_sum[2] / hole_edges as f32 + cz as f32 * cs,
            ];
            issues.push(WorldIssue {
                issue_type: IssueType::MeshHole,
                severity: IssueSeverity::Error,
                position: avg,
                chunk: [cx, cy, cz],
                neighbor: None,
                face: None,
                detail: format!(
                    "{} interior boundary edges in chunk mesh",
                    hole_edges
                ),
                detail_a: hole_edges,
                detail_b: 0,
            });
        }

        if non_manifold_edges > 0 {
            let avg = [
                nm_pos_sum[0] / non_manifold_edges as f32 + cx as f32 * cs,
                nm_pos_sum[1] / non_manifold_edges as f32 + cy as f32 * cs,
                nm_pos_sum[2] / non_manifold_edges as f32 + cz as f32 * cs,
            ];
            issues.push(WorldIssue {
                issue_type: IssueType::NonManifoldEdge,
                severity: IssueSeverity::Error,
                position: avg,
                chunk: [cx, cy, cz],
                neighbor: None,
                face: None,
                detail: format!(
                    "{} non-manifold edges (shared by 3+ triangles)",
                    non_manifold_edges
                ),
                detail_a: non_manifold_edges,
                detail_b: 0,
            });
        }
    }
}

/// Pick random air voxel origins, cast rays, check for mesh coverage at
/// air-to-solid transitions. No nearby triangle → RaymarchHole.
pub(super) fn pass_raymarch_holes(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    base_meshes: &HashMap<(i32, i32, i32), Mesh>,
    chunk_size: usize,
    config: &ScanConfig,
    issues: &mut Vec<WorldIssue>,
) {
    let cs = chunk_size as i32;
    let step = config.raymarch_step_size;

    let mut sorted_chunks: Vec<_> = density_fields.keys().copied().collect();
    sorted_chunks.sort();

    for &(cx, cy, cz) in &sorted_chunks {
        let density = &density_fields[&(cx, cy, cz)];
        let mesh = match base_meshes.get(&(cx, cy, cz)) {
            Some(m) if !m.triangles.is_empty() => m,
            _ => continue,
        };

        // Collect air voxels for deterministic ray origins
        let mut air_positions: Vec<[f32; 3]> = Vec::new();
        for lz in 1..cs - 1 {
            for ly in 1..cs - 1 {
                for lx in 1..cs - 1 {
                    if !density.get(lx as usize, ly as usize, lz as usize).material.is_solid() {
                        air_positions.push([lx as f32 + 0.5, ly as f32 + 0.5, lz as f32 + 0.5]);
                    }
                }
            }
        }

        if air_positions.is_empty() {
            continue;
        }

        // Deterministic selection using stride
        let ray_count = (config.raymarch_rays_per_chunk as usize).min(air_positions.len());
        let stride = if ray_count > 0 { air_positions.len() / ray_count } else { 1 };
        let directions: [[f32; 3]; 6] = [
            [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
        ];

        let mut hole_count = 0u32;
        let mut hole_pos_sum = [0.0f32; 3];

        for ray_idx in 0..ray_count {
            let origin = air_positions[ray_idx * stride.max(1)];
            for dir in &directions {
                // March along ray until we hit solid or leave chunk
                let max_steps = (chunk_size as f32 / step) as u32;
                for s in 1..=max_steps {
                    let t = s as f32 * step;
                    let px = origin[0] + dir[0] * t;
                    let py = origin[1] + dir[1] * t;
                    let pz = origin[2] + dir[2] * t;

                    // Out of chunk bounds
                    if px < 0.0 || py < 0.0 || pz < 0.0
                        || px >= chunk_size as f32 || py >= chunk_size as f32 || pz >= chunk_size as f32
                    {
                        break;
                    }

                    let ix = px as usize;
                    let iy = py as usize;
                    let iz = pz as usize;

                    if density.get(ix, iy, iz).material.is_solid() {
                        // Air-to-solid transition: check if any mesh triangle is nearby
                        let search_radius = 1.5;
                        let has_nearby_tri = mesh.triangles.iter().any(|tri| {
                            let v0 = mesh.vertices[tri.indices[0] as usize].position;
                            let v1 = mesh.vertices[tri.indices[1] as usize].position;
                            let v2 = mesh.vertices[tri.indices[2] as usize].position;
                            let center = (v0 + v1 + v2) / 3.0;
                            let dx = center.x - px;
                            let dy = center.y - py;
                            let dz = center.z - pz;
                            dx * dx + dy * dy + dz * dz < search_radius * search_radius
                        });
                        if !has_nearby_tri {
                            hole_count += 1;
                            hole_pos_sum[0] += px;
                            hole_pos_sum[1] += py;
                            hole_pos_sum[2] += pz;
                        }
                        break;
                    }
                }
            }
        }

        if hole_count > 0 {
            let world_pos = [
                hole_pos_sum[0] / hole_count as f32 + cx as f32 * cs as f32,
                hole_pos_sum[1] / hole_count as f32 + cy as f32 * cs as f32,
                hole_pos_sum[2] / hole_count as f32 + cz as f32 * cs as f32,
            ];
            issues.push(WorldIssue {
                issue_type: IssueType::RaymarchHole,
                severity: IssueSeverity::Warning,
                position: world_pos,
                chunk: [cx, cy, cz],
                neighbor: None,
                face: None,
                detail: format!("{} raymarch transitions with no nearby mesh triangle", hole_count),
                detail_a: hole_count,
                detail_b: 0,
            });
        }
    }
}

/// For each mesh vertex, find closest edge intersection in HermiteData.
/// If distance > threshold → MeshDensityMisalignment.
pub(super) fn pass_mesh_density_alignment(
    base_meshes: &HashMap<(i32, i32, i32), Mesh>,
    hermite_data: &HashMap<(i32, i32, i32), crate::hermite::HermiteData>,
    chunk_size: usize,
    config: &ScanConfig,
    issues: &mut Vec<WorldIssue>,
) {
    let cs = chunk_size as f32;

    let mut sorted_chunks: Vec<_> = base_meshes.keys().copied().collect();
    sorted_chunks.sort();

    for &(cx, cy, cz) in &sorted_chunks {
        let mesh = &base_meshes[&(cx, cy, cz)];
        let hermite = match hermite_data.get(&(cx, cy, cz)) {
            Some(h) => h,
            None => continue,
        };

        if mesh.vertices.is_empty() || hermite.edge_count() == 0 {
            continue;
        }

        // Collect all edge intersection positions
        let edge_positions: Vec<[f32; 3]> = hermite.iter().map(|(ek, ei)| {
            let x = ek.x() as f32;
            let y = ek.y() as f32;
            let z = ek.z() as f32;
            let axis = ek.axis();
            match axis {
                0 => [x + ei.t, y, z],
                1 => [x, y + ei.t, z],
                _ => [x, y, z + ei.t],
            }
        }).collect();

        let mut misalign_count = 0u32;
        let mut misalign_pos_sum = [0.0f32; 3];
        let threshold = config.max_vertex_zero_crossing_dist;

        for vert in &mesh.vertices {
            let vp = [vert.position.x, vert.position.y, vert.position.z];
            let min_dist_sq = edge_positions.iter().map(|ep| {
                let dx = vp[0] - ep[0];
                let dy = vp[1] - ep[1];
                let dz = vp[2] - ep[2];
                dx * dx + dy * dy + dz * dz
            }).fold(f32::MAX, f32::min);

            if min_dist_sq > threshold * threshold {
                misalign_count += 1;
                misalign_pos_sum[0] += vp[0];
                misalign_pos_sum[1] += vp[1];
                misalign_pos_sum[2] += vp[2];
            }
        }

        if misalign_count > 0 {
            let world_pos = [
                misalign_pos_sum[0] / misalign_count as f32 + cx as f32 * cs,
                misalign_pos_sum[1] / misalign_count as f32 + cy as f32 * cs,
                misalign_pos_sum[2] / misalign_count as f32 + cz as f32 * cs,
            ];
            issues.push(WorldIssue {
                issue_type: IssueType::MeshDensityMisalignment,
                severity: IssueSeverity::Warning,
                position: world_pos,
                chunk: [cx, cy, cz],
                neighbor: None,
                face: None,
                detail: format!(
                    "{} vertices farther than {:.2} from nearest hermite edge intersection",
                    misalign_count, threshold
                ),
                detail_a: misalign_count,
                detail_b: mesh.vertices.len() as u32,
            });
        }
    }
}
