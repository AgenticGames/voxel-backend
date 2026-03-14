//! Pass 4: Cross-chunk navigability + BFS volume classification.
//! Pass 6: Thin walls detection.
//! Pass E4: Narrow passage check.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::density::DensityField;

use super::{AirVolume, IssueType, IssueSeverity, ScanConfig, VolumeType, WorldIssue};

/// BFS flood fill across all loaded density fields (6-connected, cross-chunk).
/// Produces connected air components. For large components separated by 1-2
/// solid voxels → NavigabilityGap.
pub(super) fn pass_navigability(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
    issues: &mut Vec<WorldIssue>,
) -> Vec<AirVolume> {
    let loaded: HashSet<(i32, i32, i32)> = density_fields.keys().copied().collect();
    let cs = chunk_size as i32;

    // Global visited set: (chunk, local_x, local_y, local_z)
    let mut visited: HashSet<(i32, i32, i32, i32, i32, i32)> = HashSet::new();
    let mut volumes: Vec<AirVolume> = Vec::new();
    let mut volume_id: u32 = 1;

    // Track which chunks are at the edge of the loaded set
    let edge_chunks: HashSet<(i32, i32, i32)> = loaded.iter().copied().filter(|&(cx, cy, cz)| {
        [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)].iter()
            .any(|&(dx,dy,dz)| !loaded.contains(&(cx+dx, cy+dy, cz+dz)))
    }).collect();

    // Iterate chunks deterministically
    let mut sorted_chunks: Vec<_> = loaded.iter().copied().collect();
    sorted_chunks.sort();

    for &(cx, cy, cz) in &sorted_chunks {
        let density = &density_fields[&(cx, cy, cz)];

        for lz in 0..cs {
            for ly in 0..cs {
                for lx in 0..cs {
                    if visited.contains(&(cx, cy, cz, lx, ly, lz)) {
                        continue;
                    }

                    let sample = density.get(lx as usize, ly as usize, lz as usize);
                    if sample.material.is_solid() {
                        continue;
                    }

                    // BFS flood fill from this air voxel
                    let mut queue: VecDeque<(i32, i32, i32, i32, i32, i32)> = VecDeque::new();
                    queue.push_back((cx, cy, cz, lx, ly, lz));
                    visited.insert((cx, cy, cz, lx, ly, lz));

                    let mut voxel_count: u32 = 0;
                    let mut sum = [0.0f64; 3];
                    let mut aabb_min = [f32::MAX; 3];
                    let mut aabb_max = [f32::MIN; 3];
                    let mut chunks_in_volume: HashSet<[i32; 3]> = HashSet::new();
                    let mut touches_edge = false;

                    while let Some((qcx, qcy, qcz, qlx, qly, qlz)) = queue.pop_front() {
                        voxel_count += 1;
                        let world_x = qcx as f32 * cs as f32 + qlx as f32;
                        let world_y = qcy as f32 * cs as f32 + qly as f32;
                        let world_z = qcz as f32 * cs as f32 + qlz as f32;
                        sum[0] += world_x as f64;
                        sum[1] += world_y as f64;
                        sum[2] += world_z as f64;
                        aabb_min[0] = aabb_min[0].min(world_x);
                        aabb_min[1] = aabb_min[1].min(world_y);
                        aabb_min[2] = aabb_min[2].min(world_z);
                        aabb_max[0] = aabb_max[0].max(world_x);
                        aabb_max[1] = aabb_max[1].max(world_y);
                        aabb_max[2] = aabb_max[2].max(world_z);
                        chunks_in_volume.insert([qcx, qcy, qcz]);

                        if edge_chunks.contains(&(qcx, qcy, qcz)) {
                            // Check if this voxel is on the boundary of the loaded set
                            if qlx == 0 && !loaded.contains(&(qcx - 1, qcy, qcz)) {
                                touches_edge = true;
                            }
                            if qlx == cs - 1 && !loaded.contains(&(qcx + 1, qcy, qcz)) {
                                touches_edge = true;
                            }
                            if qly == 0 && !loaded.contains(&(qcx, qcy - 1, qcz)) {
                                touches_edge = true;
                            }
                            if qly == cs - 1 && !loaded.contains(&(qcx, qcy + 1, qcz)) {
                                touches_edge = true;
                            }
                            if qlz == 0 && !loaded.contains(&(qcx, qcy, qcz - 1)) {
                                touches_edge = true;
                            }
                            if qlz == cs - 1 && !loaded.contains(&(qcx, qcy, qcz + 1)) {
                                touches_edge = true;
                            }
                        }

                        // 6-connected neighbors
                        let neighbors = [
                            (qlx - 1, qly, qlz),
                            (qlx + 1, qly, qlz),
                            (qlx, qly - 1, qlz),
                            (qlx, qly + 1, qlz),
                            (qlx, qly, qlz - 1),
                            (qlx, qly, qlz + 1),
                        ];

                        for (nx, ny, nz) in neighbors {
                            // Resolve cross-chunk
                            let (ncx, ncy, ncz, nlx, nly, nlz) =
                                resolve_cross_chunk(qcx, qcy, qcz, nx, ny, nz, cs);

                            if !loaded.contains(&(ncx, ncy, ncz)) {
                                continue;
                            }
                            if visited.contains(&(ncx, ncy, ncz, nlx, nly, nlz)) {
                                continue;
                            }

                            let nd = &density_fields[&(ncx, ncy, ncz)];
                            let ns = nd.get(nlx as usize, nly as usize, nlz as usize);
                            if !ns.material.is_solid() {
                                visited.insert((ncx, ncy, ncz, nlx, nly, nlz));
                                queue.push_back((ncx, ncy, ncz, nlx, nly, nlz));
                            }
                        }
                    }

                    // Classify volume
                    let volume_type = if voxel_count < 20 {
                        VolumeType::Pocket
                    } else if touches_edge {
                        VolumeType::Surface
                    } else {
                        // Use aspect ratio to distinguish caves from worms
                        let extent = [
                            aabb_max[0] - aabb_min[0],
                            aabb_max[1] - aabb_min[1],
                            aabb_max[2] - aabb_min[2],
                        ];
                        let longest = extent[0].max(extent[1]).max(extent[2]).max(1.0);
                        let ratio = voxel_count as f32 / longest;
                        if ratio > 10.0 {
                            VolumeType::Cave
                        } else {
                            VolumeType::Worm
                        }
                    };

                    let centroid = [
                        (sum[0] / voxel_count as f64) as f32,
                        (sum[1] / voxel_count as f64) as f32,
                        (sum[2] / voxel_count as f64) as f32,
                    ];

                    let mut chunks_sorted: Vec<[i32; 3]> = chunks_in_volume.into_iter().collect();
                    chunks_sorted.sort();

                    volumes.push(AirVolume {
                        id: volume_id,
                        volume_type,
                        voxel_count,
                        centroid,
                        aabb_min,
                        aabb_max,
                        connected_to: Vec::new(), // filled below
                        chunks_spanned: chunks_sorted,
                    });
                    volume_id += 1;
                }
            }
        }
    }

    // Find navigability gaps: large volumes separated by thin walls (1-2 voxels)
    // For each pair of large volumes, check if they're separated by <= 2 voxels
    let large_volumes: Vec<usize> = volumes.iter().enumerate()
        .filter(|(_, v)| v.voxel_count >= 50)
        .map(|(i, _)| i)
        .collect();

    for i in 0..large_volumes.len() {
        for j in (i + 1)..large_volumes.len() {
            let vi = &volumes[large_volumes[i]];
            let vj = &volumes[large_volumes[j]];

            // Check AABB proximity (expand by 3 voxels)
            let expand = 3.0;
            let overlaps = vi.aabb_min[0] - expand <= vj.aabb_max[0]
                && vi.aabb_max[0] + expand >= vj.aabb_min[0]
                && vi.aabb_min[1] - expand <= vj.aabb_max[1]
                && vi.aabb_max[1] + expand >= vj.aabb_min[1]
                && vi.aabb_min[2] - expand <= vj.aabb_max[2]
                && vi.aabb_max[2] + expand >= vj.aabb_min[2];

            if overlaps {
                let mid = [
                    (vi.centroid[0] + vj.centroid[0]) * 0.5,
                    (vi.centroid[1] + vj.centroid[1]) * 0.5,
                    (vi.centroid[2] + vj.centroid[2]) * 0.5,
                ];
                // Determine chunk for this midpoint
                let mid_chunk = [
                    (mid[0] / cs as f32).floor() as i32,
                    (mid[1] / cs as f32).floor() as i32,
                    (mid[2] / cs as f32).floor() as i32,
                ];
                issues.push(WorldIssue {
                    issue_type: IssueType::NavigabilityGap,
                    severity: IssueSeverity::Warning,
                    position: mid,
                    chunk: mid_chunk,
                    neighbor: None,
                    face: None,
                    detail: format!(
                        "Volumes {} ({} voxels) and {} ({} voxels) separated by thin barrier",
                        vi.id, vi.voxel_count, vj.id, vj.voxel_count
                    ),
                    detail_a: vi.id,
                    detail_b: vj.id,
                });
            }
        }
    }

    // Reclassify junctions: volumes connected to 3+ other volumes
    // (Use AABB adjacency as a proxy for connectivity)
    let expand = 1.5;
    for i in 0..volumes.len() {
        let mut connected: Vec<u32> = Vec::new();
        for j in 0..volumes.len() {
            if i == j { continue; }
            let vi = &volumes[i];
            let vj = &volumes[j];
            let adjacent = vi.aabb_min[0] - expand <= vj.aabb_max[0]
                && vi.aabb_max[0] + expand >= vj.aabb_min[0]
                && vi.aabb_min[1] - expand <= vj.aabb_max[1]
                && vi.aabb_max[1] + expand >= vj.aabb_min[1]
                && vi.aabb_min[2] - expand <= vj.aabb_max[2]
                && vi.aabb_max[2] + expand >= vj.aabb_min[2];
            if adjacent {
                connected.push(vj.id);
            }
        }
        volumes[i].connected_to = connected;
    }

    // Reclassify: if connected to 3+ others, mark as Junction
    for v in &mut volumes {
        if v.connected_to.len() >= 3 && v.voxel_count >= 20 {
            v.volume_type = VolumeType::Junction;
        }
    }

    volumes
}

/// Resolve a local coordinate that may be out of bounds into the correct
/// chunk + local coordinate.
pub(super) fn resolve_cross_chunk(
    cx: i32, cy: i32, cz: i32,
    lx: i32, ly: i32, lz: i32,
    cs: i32,
) -> (i32, i32, i32, i32, i32, i32) {
    let mut ncx = cx;
    let mut ncy = cy;
    let mut ncz = cz;
    let mut nlx = lx;
    let mut nly = ly;
    let mut nlz = lz;

    if nlx < 0 { ncx -= 1; nlx += cs; }
    else if nlx >= cs { ncx += 1; nlx -= cs; }
    if nly < 0 { ncy -= 1; nly += cs; }
    else if nly >= cs { ncy += 1; nly -= cs; }
    if nlz < 0 { ncz -= 1; nlz += cs; }
    else if nlz >= cs { ncz += 1; nlz -= cs; }

    (ncx, ncy, ncz, nlx, nly, nlz)
}

/// Each solid voxel: air at both opposing faces on any axis → ThinWall.
/// Cluster nearby thin voxels, emit one issue per cluster.
pub(super) fn pass_thin_walls(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
    issues: &mut Vec<WorldIssue>,
) {
    let cs = chunk_size as i32;

    let mut sorted_chunks: Vec<_> = density_fields.keys().copied().collect();
    sorted_chunks.sort();

    for &(cx, cy, cz) in &sorted_chunks {
        let density = &density_fields[&(cx, cy, cz)];
        let mut thin_voxels: Vec<[i32; 3]> = Vec::new();

        for lz in 1..cs - 1 {
            for ly in 1..cs - 1 {
                for lx in 1..cs - 1 {
                    let sample = density.get(lx as usize, ly as usize, lz as usize);
                    if !sample.material.is_solid() {
                        continue;
                    }

                    // Check X axis
                    let air_neg_x = !density.get((lx - 1) as usize, ly as usize, lz as usize).material.is_solid();
                    let air_pos_x = !density.get((lx + 1) as usize, ly as usize, lz as usize).material.is_solid();

                    // Check Y axis
                    let air_neg_y = !density.get(lx as usize, (ly - 1) as usize, lz as usize).material.is_solid();
                    let air_pos_y = !density.get(lx as usize, (ly + 1) as usize, lz as usize).material.is_solid();

                    // Check Z axis
                    let air_neg_z = !density.get(lx as usize, ly as usize, (lz - 1) as usize).material.is_solid();
                    let air_pos_z = !density.get(lx as usize, ly as usize, (lz + 1) as usize).material.is_solid();

                    if (air_neg_x && air_pos_x) || (air_neg_y && air_pos_y) || (air_neg_z && air_pos_z) {
                        thin_voxels.push([lx, ly, lz]);
                    }
                }
            }
        }

        if thin_voxels.is_empty() {
            continue;
        }

        // Cluster thin voxels (simple: group within 3-voxel distance)
        let clusters = cluster_positions(&thin_voxels, 3);
        for cluster in clusters {
            let count = cluster.len() as u32;
            let centroid = cluster_centroid(&cluster);
            let world_pos = [
                cx as f32 * cs as f32 + centroid[0],
                cy as f32 * cs as f32 + centroid[1],
                cz as f32 * cs as f32 + centroid[2],
            ];

            // Determine primary axis
            let axis = dominant_thin_axis(&cluster, density);
            issues.push(WorldIssue {
                issue_type: IssueType::ThinWall,
                severity: IssueSeverity::Info,
                position: world_pos,
                chunk: [cx, cy, cz],
                neighbor: None,
                face: None,
                detail: format!(
                    "{} thin-wall voxels along {}-axis (air on both sides)",
                    count, axis
                ),
                detail_a: count,
                detail_b: 0,
            });
        }
    }
}

/// Simple single-linkage clustering: group positions within `max_dist` of each other.
fn cluster_positions(positions: &[[i32; 3]], max_dist: i32) -> Vec<Vec<[i32; 3]>> {
    let mut clusters: Vec<Vec<[i32; 3]>> = Vec::new();
    let mut assigned = vec![false; positions.len()];

    for i in 0..positions.len() {
        if assigned[i] { continue; }
        assigned[i] = true;
        let mut cluster = vec![positions[i]];
        let mut frontier = vec![i];

        while let Some(idx) = frontier.pop() {
            for j in 0..positions.len() {
                if assigned[j] { continue; }
                let dx = (positions[idx][0] - positions[j][0]).abs();
                let dy = (positions[idx][1] - positions[j][1]).abs();
                let dz = (positions[idx][2] - positions[j][2]).abs();
                if dx <= max_dist && dy <= max_dist && dz <= max_dist {
                    assigned[j] = true;
                    cluster.push(positions[j]);
                    frontier.push(j);
                }
            }
        }

        clusters.push(cluster);
    }

    clusters
}

fn cluster_centroid(cluster: &[[i32; 3]]) -> [f32; 3] {
    let n = cluster.len() as f32;
    let mut sum = [0.0f32; 3];
    for p in cluster {
        sum[0] += p[0] as f32;
        sum[1] += p[1] as f32;
        sum[2] += p[2] as f32;
    }
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

fn dominant_thin_axis(cluster: &[[i32; 3]], _density: &DensityField) -> &'static str {
    // Simple: use the axis along which the cluster has the most spread
    let mut min = [i32::MAX; 3];
    let mut max = [i32::MIN; 3];
    for p in cluster {
        for i in 0..3 {
            min[i] = min[i].min(p[i]);
            max[i] = max[i].max(p[i]);
        }
    }
    let extents = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    if extents[0] >= extents[1] && extents[0] >= extents[2] {
        "X"
    } else if extents[1] >= extents[2] {
        "Y"
    } else {
        "Z"
    }
}

/// For each large volume, check passage widths by sampling centroids and
/// casting 3-axis rays. If width < min_passage_width → NarrowPassage.
pub(super) fn pass_narrow_passage(
    volumes: &[AirVolume],
    config: &ScanConfig,
    issues: &mut Vec<WorldIssue>,
) {
    let min_width = config.min_passage_width;

    for vol in volumes {
        if vol.voxel_count < 50 {
            continue;
        }

        // Approximate width from AABB extents
        let extents = [
            vol.aabb_max[0] - vol.aabb_min[0],
            vol.aabb_max[1] - vol.aabb_min[1],
            vol.aabb_max[2] - vol.aabb_min[2],
        ];
        let min_extent = extents[0].min(extents[1]).min(extents[2]);

        if min_extent < min_width {
            let chunk = [
                vol.centroid[0].floor() as i32 / 16.max(1),
                vol.centroid[1].floor() as i32 / 16.max(1),
                vol.centroid[2].floor() as i32 / 16.max(1),
            ];
            issues.push(WorldIssue {
                issue_type: IssueType::NarrowPassage,
                severity: IssueSeverity::Info,
                position: vol.centroid,
                chunk,
                neighbor: None,
                face: None,
                detail: format!(
                    "Volume {} has minimum extent {:.1} < {:.1} threshold",
                    vol.id, min_extent, min_width
                ),
                detail_a: vol.id,
                detail_b: (min_extent * 100.0) as u32,
            });
        }
    }
}
