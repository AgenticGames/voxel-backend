//! World scan: machine-readable hole detection & topology diagnostics.
//!
//! Produces a structured JSON report for iterative fix loops.
//! Used by both CLI (`voxel-cli scan`) and FFI (`voxel_request_world_scan`).

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

use serde::Serialize;

use crate::density::DensityField;
use crate::hermite::EdgeKey;
use crate::mesh::Mesh;

// ── Types ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, PartialEq, Eq, Hash)]
pub enum IssueType {
    MeshHole,
    SeamGap,
    DensityDiscontinuity,
    WormTruncation,
    NavigabilityGap,
    ThinWall,
    NonManifoldEdge,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub enum IssueSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Clone, Debug, Serialize)]
pub enum VolumeType {
    Cave,
    Worm,
    Junction,
    Surface,
    Pocket,
}

#[derive(Clone, Debug, Serialize)]
pub struct WorldIssue {
    #[serde(rename = "type")]
    pub issue_type: IssueType,
    pub severity: IssueSeverity,
    pub position: [f32; 3],
    pub chunk: [i32; 3],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neighbor: Option<[i32; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub face: Option<String>,
    pub detail: String,
    pub detail_a: u32,
    pub detail_b: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct AirVolume {
    pub id: u32,
    #[serde(rename = "type")]
    pub volume_type: VolumeType,
    pub voxel_count: u32,
    pub centroid: [f32; 3],
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub connected_to: Vec<u32>,
    pub chunks_spanned: Vec<[i32; 3]>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ScanSummary {
    #[serde(rename = "MeshHole")]
    pub mesh_hole: u32,
    #[serde(rename = "SeamGap")]
    pub seam_gap: u32,
    #[serde(rename = "DensityDiscontinuity")]
    pub density_discontinuity: u32,
    #[serde(rename = "WormTruncation")]
    pub worm_truncation: u32,
    #[serde(rename = "NavigabilityGap")]
    pub navigability_gap: u32,
    #[serde(rename = "ThinWall")]
    pub thin_wall: u32,
    #[serde(rename = "NonManifoldEdge")]
    pub non_manifold_edge: u32,
    pub total_errors: u32,
    pub total_warnings: u32,
    pub total_info: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct WorldScanResult {
    pub timestamp: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    pub chunks_scanned: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_range: Option<ChunkRange>,
    pub scan_duration_ms: f64,
    pub total_air_voxels: u64,
    pub total_solid_voxels: u64,
    pub summary: ScanSummary,
    pub issues: Vec<WorldIssue>,
    pub volumes: Vec<AirVolume>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChunkRange {
    pub min: [i32; 3],
    pub max: [i32; 3],
}

/// Minimal seam data needed for scan (matches region_gen::ChunkSeamData layout).
pub struct ScanSeamData {
    pub boundary_edges: Vec<(EdgeKey, crate::hermite::EdgeIntersection)>,
}

/// Worm segment for scan (matches worm::path::WormSegment).
pub struct ScanWormSegment {
    pub position: [f32; 3],
    pub radius: f32,
}

impl WorldScanResult {
    pub fn to_json_string(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|e| {
            format!("{{\"error\": \"serialization failed: {}\"}}", e)
        })
    }
}

// ── Core Scan Function ─────────────────────────────────────────────

/// Run all 6 analysis passes over a generated region.
///
/// Both CLI and FFI paths call this with their respective data sources.
pub fn scan_world(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    base_meshes: &HashMap<(i32, i32, i32), Mesh>,
    seam_data: &HashMap<(i32, i32, i32), ScanSeamData>,
    worm_paths: &[Vec<ScanWormSegment>],
    chunk_size: usize,
) -> WorldScanResult {
    let start = Instant::now();

    let mut issues = Vec::new();
    let chunks: Vec<(i32, i32, i32)> = density_fields.keys().copied().collect();

    // Compute voxel counts
    let mut total_air: u64 = 0;
    let mut total_solid: u64 = 0;
    for density in density_fields.values() {
        total_air += density.air_cell_count as u64;
        let inner_count = (chunk_size * chunk_size * chunk_size) as u64;
        total_solid += inner_count - density.air_cell_count as u64;
    }

    // Chunk range
    let chunk_range = if !chunks.is_empty() {
        let mut min = [i32::MAX; 3];
        let mut max = [i32::MIN; 3];
        for &(cx, cy, cz) in &chunks {
            min[0] = min[0].min(cx);
            min[1] = min[1].min(cy);
            min[2] = min[2].min(cz);
            max[0] = max[0].max(cx);
            max[1] = max[1].max(cy);
            max[2] = max[2].max(cz);
        }
        Some(ChunkRange { min, max })
    } else {
        None
    };

    // Pass 1: Density seam check
    pass_density_seam(density_fields, chunk_size, &mut issues);

    // Pass 2: Mesh topology
    pass_mesh_topology(base_meshes, chunk_size, &mut issues);

    // Pass 3: Seam completeness
    pass_seam_completeness(seam_data, base_meshes, density_fields, chunk_size, &mut issues);

    // Pass 4: Cross-chunk navigability + volume classification
    let volumes = pass_navigability(density_fields, chunk_size, &mut issues);

    // Pass 5: Worm truncation
    pass_worm_truncation(worm_paths, density_fields, chunk_size, &mut issues);

    // Pass 6: Thin walls
    pass_thin_walls(density_fields, chunk_size, &mut issues);

    // Build summary
    let summary = build_summary(&issues);

    let elapsed = start.elapsed();

    WorldScanResult {
        timestamp: chrono_timestamp(),
        seed: None,
        chunks_scanned: chunks.len() as u32,
        chunk_range,
        scan_duration_ms: elapsed.as_secs_f64() * 1000.0,
        total_air_voxels: total_air,
        total_solid_voxels: total_solid,
        summary,
        issues,
        volumes,
    }
}

// ── Pass 1: Density Seam Check ─────────────────────────────────────

/// For each loaded chunk pair sharing a face, compare density signs at the
/// shared sample layer. Mismatched signs → DensityDiscontinuity.
fn pass_density_seam(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
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
        }
    }
}

// ── Pass 2: Mesh Topology ──────────────────────────────────────────

/// Per-mesh edge-count analysis. Interior boundary edges (count=1, not on
/// chunk face) → MeshHole. Count>=3 → NonManifoldEdge.
fn pass_mesh_topology(
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

// ── Pass 3: Seam Completeness ──────────────────────────────────────

/// For adjacent loaded chunk pairs: count boundary edges that need seam quads.
/// If boundary edges > 0 but no seam mesh covers them → SeamGap.
fn pass_seam_completeness(
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

// ── Pass 4: Cross-Chunk Navigability ───────────────────────────────

/// BFS flood fill across all loaded density fields (6-connected, cross-chunk).
/// Produces connected air components. For large components separated by 1-2
/// solid voxels → NavigabilityGap.
fn pass_navigability(
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
fn resolve_cross_chunk(
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

// ── Pass 5: Worm Truncation ────────────────────────────────────────

/// Each worm segment: check if its chunk is in the loaded set.
/// If not → WormTruncation.
fn pass_worm_truncation(
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

// ── Pass 6: Thin Walls ─────────────────────────────────────────────

/// Each solid voxel: air at both opposing faces on any axis → ThinWall.
/// Cluster nearby thin voxels, emit one issue per cluster.
fn pass_thin_walls(
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

// ── Summary Builder ────────────────────────────────────────────────

fn build_summary(issues: &[WorldIssue]) -> ScanSummary {
    let mut summary = ScanSummary::default();

    for issue in issues {
        match issue.issue_type {
            IssueType::MeshHole => summary.mesh_hole += 1,
            IssueType::SeamGap => summary.seam_gap += 1,
            IssueType::DensityDiscontinuity => summary.density_discontinuity += 1,
            IssueType::WormTruncation => summary.worm_truncation += 1,
            IssueType::NavigabilityGap => summary.navigability_gap += 1,
            IssueType::ThinWall => summary.thin_wall += 1,
            IssueType::NonManifoldEdge => summary.non_manifold_edge += 1,
        }
        match issue.severity {
            IssueSeverity::Error => summary.total_errors += 1,
            IssueSeverity::Warning => summary.total_warnings += 1,
            IssueSeverity::Info => summary.total_info += 1,
        }
    }

    summary
}

// ── Helpers ────────────────────────────────────────────────────────

fn chrono_timestamp() -> String {
    // Simple UTC timestamp without chrono dependency
    use std::time::SystemTime;
    match SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
        Ok(d) => {
            let secs = d.as_secs();
            // Simple epoch-to-ISO conversion
            let days = secs / 86400;
            let time_of_day = secs % 86400;
            let hours = time_of_day / 3600;
            let minutes = (time_of_day % 3600) / 60;
            let seconds = time_of_day % 60;

            // Calculate year/month/day from days since epoch (1970-01-01)
            let (year, month, day) = days_to_ymd(days);
            format!(
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
                year, month, day, hours, minutes, seconds
            )
        }
        Err(_) => "1970-01-01T00:00:00Z".to_string(),
    }
}

fn days_to_ymd(days_since_epoch: u64) -> (u64, u64, u64) {
    // Civil calendar from days since 1970-01-01
    let z = days_since_epoch + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year, m, d)
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Material;

    /// Create a density field filled with a given material and density value.
    fn make_density(chunk_size: usize, solid: bool) -> DensityField {
        let size = chunk_size + 1;
        let mut df = DensityField::new(size);
        for s in &mut df.samples {
            if solid {
                s.density = 1.0;
                s.material = Material::Limestone;
            } else {
                s.density = -1.0;
                s.material = Material::Air;
            }
        }
        df.compute_metadata();
        df
    }

    /// Create a density field with a cave (hollow center).
    fn make_cave_density(chunk_size: usize) -> DensityField {
        let size = chunk_size + 1;
        let mut df = DensityField::new(size);
        // Solid everywhere
        for s in &mut df.samples {
            s.density = 1.0;
            s.material = Material::Limestone;
        }
        // Hollow center
        let margin = 3;
        let cs = chunk_size as i32;
        for z in margin..cs - margin {
            for y in margin..cs - margin {
                for x in margin..cs - margin {
                    let s = df.get_mut(x as usize, y as usize, z as usize);
                    s.density = -1.0;
                    s.material = Material::Air;
                }
            }
        }
        df.compute_metadata();
        df
    }

    #[test]
    fn test_density_discontinuity() {
        let cs = 4;
        let mut density_a = make_density(cs, true);
        let density_b = make_density(cs, true);

        // Make boundary of A have air, but B has solid — mismatch
        for z in 0..=cs {
            for y in 0..=cs {
                let s = density_a.get_mut(cs, y, z);
                s.density = -1.0;
                s.material = Material::Air;
            }
        }

        let mut fields = HashMap::new();
        fields.insert((0, 0, 0), density_a);
        fields.insert((1, 0, 0), density_b);

        let meshes = HashMap::new();
        let seam = HashMap::new();
        let worms = Vec::new();

        let result = scan_world(&fields, &meshes, &seam, &worms, cs);
        let disc_issues: Vec<_> = result.issues.iter()
            .filter(|i| i.issue_type == IssueType::DensityDiscontinuity)
            .collect();
        assert!(!disc_issues.is_empty(), "Should detect density discontinuity");
        assert!(disc_issues[0].detail_a > 0);
    }

    #[test]
    fn test_mesh_hole() {
        use crate::mesh::{Mesh, Vertex, Triangle};
        use glam::Vec3;

        let cs = 8;
        // Create a mesh with an interior boundary edge (single triangle, not on boundary)
        let mesh = Mesh {
            vertices: vec![
                Vertex { position: Vec3::new(4.0, 4.0, 4.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(5.0, 4.0, 4.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(4.0, 5.0, 4.0), normal: Vec3::Y, material: Material::Limestone },
            ],
            triangles: vec![Triangle { indices: [0, 1, 2] }],
        };

        let mut meshes = HashMap::new();
        meshes.insert((0, 0, 0), mesh);

        let fields = HashMap::new();
        let seam = HashMap::new();
        let worms = Vec::new();

        let result = scan_world(&fields, &meshes, &seam, &worms, cs);
        let holes: Vec<_> = result.issues.iter()
            .filter(|i| i.issue_type == IssueType::MeshHole)
            .collect();
        assert!(!holes.is_empty(), "Should detect mesh hole from single triangle");
    }

    #[test]
    fn test_thin_wall() {
        let cs = 8;
        let mut density = make_density(cs, true);

        // Create air-solid-air pattern along X axis at y=4, z=4
        // x=3 is air, x=4 is solid (thin), x=5 is air
        let s = density.get_mut(3, 4, 4);
        s.density = -1.0;
        s.material = Material::Air;
        // x=4 stays solid (thin wall)
        let s = density.get_mut(5, 4, 4);
        s.density = -1.0;
        s.material = Material::Air;

        density.compute_metadata();

        let mut fields = HashMap::new();
        fields.insert((0, 0, 0), density);

        let meshes = HashMap::new();
        let seam = HashMap::new();
        let worms = Vec::new();

        let result = scan_world(&fields, &meshes, &seam, &worms, cs);
        let thin: Vec<_> = result.issues.iter()
            .filter(|i| i.issue_type == IssueType::ThinWall)
            .collect();
        assert!(!thin.is_empty(), "Should detect thin wall");
    }

    #[test]
    fn test_navigability_volumes() {
        let cs = 16;
        let density = make_cave_density(cs);

        let mut fields = HashMap::new();
        fields.insert((0, 0, 0), density);

        let meshes = HashMap::new();
        let seam = HashMap::new();
        let worms = Vec::new();

        let result = scan_world(&fields, &meshes, &seam, &worms, cs);
        assert!(!result.volumes.is_empty(), "Should detect air volumes");
        // Should have at least one cave volume
        let caves: Vec<_> = result.volumes.iter()
            .filter(|v| matches!(v.volume_type, VolumeType::Cave))
            .collect();
        assert!(!caves.is_empty(), "Should classify large air volume as Cave");
    }

    #[test]
    fn test_worm_truncation() {
        let cs = 16;
        let density = make_density(cs, true);

        let mut fields = HashMap::new();
        fields.insert((0, 0, 0), density);

        // Worm that exits the loaded chunk set
        let worm = vec![
            ScanWormSegment { position: [8.0, 8.0, 8.0], radius: 2.0 },
            ScanWormSegment { position: [12.0, 8.0, 8.0], radius: 2.0 },
            ScanWormSegment { position: [20.0, 8.0, 8.0], radius: 2.0 }, // Outside chunk (0,0,0)
        ];

        let meshes = HashMap::new();
        let seam = HashMap::new();
        let worms = vec![worm];

        let result = scan_world(&fields, &meshes, &seam, &worms, cs);
        let truncations: Vec<_> = result.issues.iter()
            .filter(|i| i.issue_type == IssueType::WormTruncation)
            .collect();
        assert!(!truncations.is_empty(), "Should detect worm truncation");
    }

    #[test]
    fn test_clean_region() {
        let cs = 4;
        // All solid, no air — should have no error-level issues
        let mut fields = HashMap::new();
        for cx in -1..=1 {
            for cy in -1..=1 {
                for cz in -1..=1 {
                    fields.insert((cx, cy, cz), make_density(cs, true));
                }
            }
        }

        let meshes = HashMap::new();
        let seam = HashMap::new();
        let worms = Vec::new();

        let result = scan_world(&fields, &meshes, &seam, &worms, cs);
        assert_eq!(result.summary.total_errors, 0, "All-solid region should have 0 errors");
    }

    #[test]
    fn test_volume_classification() {
        let cs = 16;

        // Large open cave
        let mut cave_density = make_density(cs, true);
        for z in 2..14 {
            for y in 2..14 {
                for x in 2..14 {
                    let s = cave_density.get_mut(x, y, z);
                    s.density = -1.0;
                    s.material = Material::Air;
                }
            }
        }
        cave_density.compute_metadata();

        let mut fields = HashMap::new();
        fields.insert((0, 0, 0), cave_density);

        let meshes = HashMap::new();
        let seam = HashMap::new();
        let worms = Vec::new();

        let result = scan_world(&fields, &meshes, &seam, &worms, cs);
        assert!(!result.volumes.is_empty());
        // Large cubic cavity should be a Cave
        let big_vol = result.volumes.iter()
            .max_by_key(|v| v.voxel_count)
            .unwrap();
        assert!(matches!(big_vol.volume_type, VolumeType::Cave),
            "Large cubic void should be Cave, got {:?}", big_vol.volume_type);
    }

    #[test]
    fn test_json_serialization() {
        let cs = 4;
        let mut fields = HashMap::new();
        fields.insert((0, 0, 0), make_density(cs, true));

        let meshes = HashMap::new();
        let seam = HashMap::new();
        let worms = Vec::new();

        let result = scan_world(&fields, &meshes, &seam, &worms, cs);
        let json = result.to_json_string();

        // Verify it's valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&json)
            .expect("Scan result should serialize to valid JSON");

        assert!(parsed.get("timestamp").is_some());
        assert!(parsed.get("chunks_scanned").is_some());
        assert!(parsed.get("summary").is_some());
        assert!(parsed.get("issues").is_some());
        assert!(parsed.get("volumes").is_some());
    }

    #[test]
    fn test_navigability_gap() {
        let cs = 16;

        // Two air pockets separated by 1 solid voxel
        let mut density = make_density(cs, true);

        // Pocket 1: x=2..6, y=2..6, z=2..6
        for z in 2..6 {
            for y in 2..6 {
                for x in 2..6 {
                    let s = density.get_mut(x, y, z);
                    s.density = -1.0;
                    s.material = Material::Air;
                }
            }
        }
        // Pocket 2: x=7..11, y=2..6, z=2..6 (separated by x=6 solid wall)
        for z in 2..6 {
            for y in 2..6 {
                for x in 7..11 {
                    let s = density.get_mut(x, y, z);
                    s.density = -1.0;
                    s.material = Material::Air;
                }
            }
        }
        density.compute_metadata();

        let mut fields = HashMap::new();
        fields.insert((0, 0, 0), density);

        let meshes = HashMap::new();
        let seam = HashMap::new();
        let worms = Vec::new();

        let result = scan_world(&fields, &meshes, &seam, &worms, cs);

        // Should have 2 separate volumes
        let large_vols: Vec<_> = result.volumes.iter()
            .filter(|v| v.voxel_count >= 50)
            .collect();
        assert!(large_vols.len() >= 2, "Should have at least 2 large volumes, got {}", large_vols.len());

        // Should detect navigability gap
        let gaps: Vec<_> = result.issues.iter()
            .filter(|i| i.issue_type == IssueType::NavigabilityGap)
            .collect();
        assert!(!gaps.is_empty(), "Should detect navigability gap between adjacent volumes");
    }
}
