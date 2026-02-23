//! World scan: machine-readable hole detection & topology diagnostics.
//!
//! Produces a structured JSON report for iterative fix loops.
//! Used by both CLI (`voxel-cli scan`) and FFI (`voxel_request_world_scan`).

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

use serde::{Serialize, Deserialize};

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
    RaymarchHole,
    MeshDensityMisalignment,
    NarrowPassage,
    WindingInconsistency,
    DegenerateTriangle,
    StretchedTriangle,
    WormCarveFailure,
    SelfIntersection,
    SeamQualityIssue,
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
    #[serde(rename = "RaymarchHole")]
    pub raymarch_hole: u32,
    #[serde(rename = "MeshDensityMisalignment")]
    pub mesh_density_misalignment: u32,
    #[serde(rename = "NarrowPassage")]
    pub narrow_passage: u32,
    #[serde(rename = "WindingInconsistency")]
    pub winding_inconsistency: u32,
    #[serde(rename = "DegenerateTriangle")]
    pub degenerate_triangle: u32,
    #[serde(rename = "StretchedTriangle")]
    pub stretched_triangle: u32,
    #[serde(rename = "WormCarveFailure")]
    pub worm_carve_failure: u32,
    #[serde(rename = "SelfIntersection")]
    pub self_intersection: u32,
    #[serde(rename = "SeamQualityIssue")]
    pub seam_quality_issue: u32,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scan_config: Option<ScanConfig>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChunkRange {
    pub min: [i32; 3],
    pub max: [i32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanConfig {
    pub enable_density_seam: bool,
    pub enable_mesh_topology: bool,
    pub enable_seam_completeness: bool,
    pub enable_navigability: bool,
    pub enable_worm_truncation: bool,
    pub enable_thin_walls: bool,
    pub enable_winding_consistency: bool,
    pub enable_degenerate_triangles: bool,
    pub enable_worm_carve_verify: bool,
    pub enable_self_intersection: bool,
    pub enable_seam_mesh_quality: bool,
    pub density_subsample_count: u32,
    pub raymarch_rays_per_chunk: u32,
    pub raymarch_step_size: f32,
    pub max_vertex_zero_crossing_dist: f32,
    pub min_passage_width: f32,
    pub min_triangle_area: f32,
    pub max_edge_length: f32,
    pub thin_wall_max_thickness: u32,
    pub self_intersection_tri_limit: u32,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            enable_density_seam: true,
            enable_mesh_topology: true,
            enable_seam_completeness: true,
            enable_navigability: true,
            enable_worm_truncation: true,
            enable_thin_walls: true,
            enable_winding_consistency: true,
            enable_degenerate_triangles: true,
            enable_worm_carve_verify: true,
            enable_self_intersection: false,
            enable_seam_mesh_quality: true,
            density_subsample_count: 0,
            raymarch_rays_per_chunk: 0,
            raymarch_step_size: 0.25,
            max_vertex_zero_crossing_dist: 0.5,
            min_passage_width: 2.0,
            min_triangle_area: 1e-6,
            max_edge_length: 4.0,
            thin_wall_max_thickness: 1,
            self_intersection_tri_limit: 10000,
        }
    }
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

/// Backward-compatible entry point (no hermite data, default config).
pub fn scan_world(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    base_meshes: &HashMap<(i32, i32, i32), Mesh>,
    seam_data: &HashMap<(i32, i32, i32), ScanSeamData>,
    worm_paths: &[Vec<ScanWormSegment>],
    chunk_size: usize,
) -> WorldScanResult {
    scan_world_with_config(density_fields, base_meshes, None, seam_data, worm_paths, chunk_size, &ScanConfig::default())
}

/// Run all analysis passes over a generated region.
///
/// Both CLI and FFI paths call this with their respective data sources.
pub fn scan_world_with_config(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    base_meshes: &HashMap<(i32, i32, i32), Mesh>,
    hermite_data: Option<&HashMap<(i32, i32, i32), crate::hermite::HermiteData>>,
    seam_data: &HashMap<(i32, i32, i32), ScanSeamData>,
    worm_paths: &[Vec<ScanWormSegment>],
    chunk_size: usize,
    config: &ScanConfig,
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
    if config.enable_density_seam {
        pass_density_seam(density_fields, chunk_size, config, &mut issues);
    }

    // Pass 2: Mesh topology
    if config.enable_mesh_topology {
        pass_mesh_topology(base_meshes, chunk_size, &mut issues);
    }

    // Pass 3: Seam completeness
    if config.enable_seam_completeness {
        pass_seam_completeness(seam_data, base_meshes, density_fields, chunk_size, &mut issues);
    }

    // Pass 4: Cross-chunk navigability + volume classification
    let volumes = if config.enable_navigability {
        pass_navigability(density_fields, chunk_size, &mut issues)
    } else {
        Vec::new()
    };

    // Pass 5: Worm truncation
    if config.enable_worm_truncation {
        pass_worm_truncation(worm_paths, density_fields, chunk_size, &mut issues);
    }

    // Pass 6: Thin walls
    if config.enable_thin_walls {
        pass_thin_walls(density_fields, chunk_size, &mut issues);
    }

    // Pass E2: Raymarch holes
    if config.raymarch_rays_per_chunk > 0 {
        pass_raymarch_holes(density_fields, base_meshes, chunk_size, config, &mut issues);
    }

    // Pass E3: Mesh-density alignment
    if hermite_data.is_some() {
        pass_mesh_density_alignment(base_meshes, hermite_data.unwrap(), chunk_size, config, &mut issues);
    }

    // Pass E4: Narrow passage
    if config.enable_navigability {
        pass_narrow_passage(&volumes, config, &mut issues);
    }

    // Pass N1: Winding consistency
    if config.enable_winding_consistency {
        pass_winding_consistency(base_meshes, chunk_size, &mut issues);
    }

    // Pass N2: Triangle quality
    if config.enable_degenerate_triangles {
        pass_triangle_quality(base_meshes, chunk_size, config, &mut issues);
    }

    // Pass N3: Worm carve verify
    if config.enable_worm_carve_verify {
        pass_worm_carve_verify(worm_paths, density_fields, chunk_size, &mut issues);
    }

    // Pass N4: Self intersection
    if config.enable_self_intersection {
        pass_self_intersection(base_meshes, chunk_size, config, &mut issues);
    }

    // Pass N5: Seam mesh quality
    if config.enable_seam_mesh_quality {
        pass_seam_mesh_quality(seam_data, density_fields, chunk_size, &mut issues);
    }

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
        scan_config: Some(config.clone()),
    }
}

// ── Pass 1: Density Seam Check ─────────────────────────────────────

/// For each loaded chunk pair sharing a face, compare density signs at the
/// shared sample layer. Mismatched signs → DensityDiscontinuity.
fn pass_density_seam(
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

// ── Pass E2: Raymarch Holes ─────────────────────────────────────────

/// Pick random air voxel origins, cast rays, check for mesh coverage at
/// air-to-solid transitions. No nearby triangle → RaymarchHole.
fn pass_raymarch_holes(
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

// ── Pass E3: Mesh-Density Alignment ────────────────────────────────

/// For each mesh vertex, find closest edge intersection in HermiteData.
/// If distance > threshold → MeshDensityMisalignment.
fn pass_mesh_density_alignment(
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

// ── Pass E4: Narrow Passage ────────────────────────────────────────

/// For each large volume, check passage widths by sampling centroids and
/// casting 3-axis rays. If width < min_passage_width → NarrowPassage.
fn pass_narrow_passage(
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

// ── Pass N1: Winding Consistency ───────────────────────────────────

/// Compute solid centroid. For each triangle: cross product normal, dot with
/// (face_center - centroid). If >30% inward-facing → WindingInconsistency.
fn pass_winding_consistency(
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

// ── Pass N2: Triangle Quality ──────────────────────────────────────

/// Area via cross product. Small area → DegenerateTriangle. Long edges →
/// StretchedTriangle.
fn pass_triangle_quality(
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

// ── Pass N3: Worm Carve Verify ─────────────────────────────────────

/// For each worm segment in loaded chunks: sample density at center + 6 axis
/// offsets at 80% radius. If solid → WormCarveFailure.
fn pass_worm_carve_verify(
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

// ── Pass N4: Self Intersection ─────────────────────────────────────

/// Build per-triangle AABBs. For non-adjacent pairs with overlapping AABBs →
/// SelfIntersection. Skip chunks over tri_limit.
fn pass_self_intersection(
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

// ── Pass N5: Seam Mesh Quality ─────────────────────────────────────

/// For adjacent chunk pairs: count boundary edges from each side.
/// If counts differ by >50% → SeamQualityIssue.
fn pass_seam_mesh_quality(
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
            IssueType::RaymarchHole => summary.raymarch_hole += 1,
            IssueType::MeshDensityMisalignment => summary.mesh_density_misalignment += 1,
            IssueType::NarrowPassage => summary.narrow_passage += 1,
            IssueType::WindingInconsistency => summary.winding_inconsistency += 1,
            IssueType::DegenerateTriangle => summary.degenerate_triangle += 1,
            IssueType::StretchedTriangle => summary.stretched_triangle += 1,
            IssueType::WormCarveFailure => summary.worm_carve_failure += 1,
            IssueType::SelfIntersection => summary.self_intersection += 1,
            IssueType::SeamQualityIssue => summary.seam_quality_issue += 1,
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

    #[test]
    fn test_winding_consistency() {
        use crate::mesh::{Mesh, Vertex, Triangle};
        use glam::Vec3;

        let cs = 16;
        // Build 6 quads (12 triangles) for a cube centered at (4,4,4), sized 2..6.
        // All triangles have deliberately INWARD-facing cross-product normals.
        // For face at z=2 (front), outward normal = -Z. We want inward normal = +Z.
        // (v1-v0) x (v2-v0) should give +Z. E.g. v0=(2,2,2), v1=(6,2,2), v2=(2,6,2):
        //   (4,0,0) x (0,4,0) = (0,0,16) -> +Z (inward). Good.
        let mesh = Mesh {
            vertices: vec![
                Vertex { position: Vec3::new(2.0, 2.0, 2.0), normal: Vec3::Y, material: Material::Limestone }, // 0
                Vertex { position: Vec3::new(6.0, 2.0, 2.0), normal: Vec3::Y, material: Material::Limestone }, // 1
                Vertex { position: Vec3::new(6.0, 6.0, 2.0), normal: Vec3::Y, material: Material::Limestone }, // 2
                Vertex { position: Vec3::new(2.0, 6.0, 2.0), normal: Vec3::Y, material: Material::Limestone }, // 3
                Vertex { position: Vec3::new(2.0, 2.0, 6.0), normal: Vec3::Y, material: Material::Limestone }, // 4
                Vertex { position: Vec3::new(6.0, 2.0, 6.0), normal: Vec3::Y, material: Material::Limestone }, // 5
                Vertex { position: Vec3::new(6.0, 6.0, 6.0), normal: Vec3::Y, material: Material::Limestone }, // 6
                Vertex { position: Vec3::new(2.0, 6.0, 6.0), normal: Vec3::Y, material: Material::Limestone }, // 7
            ],
            triangles: vec![
                // Front (z=2): want +Z normal. (1-0)x(3-0) = (4,0,0)x(0,4,0) = (0,0,16)
                Triangle { indices: [0, 1, 3] }, Triangle { indices: [1, 2, 3] },
                // Back (z=6): want -Z normal. (7-4)x(5-4) = (0,4,0)x(4,0,0) = (0,0,-16)
                Triangle { indices: [4, 7, 5] }, Triangle { indices: [7, 6, 5] },
                // Bottom (y=2): want +Y normal. (4-0)x(1-0) = (0,0,4)x(4,0,0) = (0,16,0)
                Triangle { indices: [0, 4, 1] }, Triangle { indices: [4, 5, 1] },
                // Top (y=6): want -Y normal. (2-3)x(7-3) = (4,0,0)x(0,0,4) = (0,-16,0)
                Triangle { indices: [3, 2, 7] }, Triangle { indices: [2, 6, 7] },
                // Left (x=2): want +X normal. (3-0)x(4-0) = (0,4,0)x(0,0,4) = (16,0,0)
                Triangle { indices: [0, 3, 4] }, Triangle { indices: [3, 7, 4] },
                // Right (x=6): want -X normal. (5-1)x(2-1) = (0,0,4)x(0,4,0) = (-16,0,0)
                Triangle { indices: [1, 5, 2] }, Triangle { indices: [5, 6, 2] },
            ],
        };

        let mut meshes = HashMap::new();
        meshes.insert((0, 0, 0), mesh);
        let fields = HashMap::new();
        let seam = HashMap::new();
        let worms = Vec::new();

        let result = scan_world_with_config(&fields, &meshes, None, &seam, &worms, cs, &ScanConfig::default());
        let winding: Vec<_> = result.issues.iter()
            .filter(|i| i.issue_type == IssueType::WindingInconsistency)
            .collect();
        assert!(!winding.is_empty(), "Should detect winding inconsistency with inverted normals");
    }

    #[test]
    fn test_degenerate_triangle() {
        use crate::mesh::{Mesh, Vertex, Triangle};
        use glam::Vec3;

        let cs = 8;
        // Zero-area triangle (all vertices collinear)
        let mesh = Mesh {
            vertices: vec![
                Vertex { position: Vec3::new(4.0, 4.0, 4.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(4.0, 4.0, 4.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(4.0, 4.0, 4.0), normal: Vec3::Y, material: Material::Limestone },
            ],
            triangles: vec![Triangle { indices: [0, 1, 2] }],
        };

        let mut meshes = HashMap::new();
        meshes.insert((0, 0, 0), mesh);
        let fields = HashMap::new();
        let seam = HashMap::new();
        let worms = Vec::new();

        let result = scan_world_with_config(&fields, &meshes, None, &seam, &worms, cs, &ScanConfig::default());
        let degen: Vec<_> = result.issues.iter()
            .filter(|i| i.issue_type == IssueType::DegenerateTriangle)
            .collect();
        assert!(!degen.is_empty(), "Should detect degenerate (zero-area) triangle");
    }

    #[test]
    fn test_stretched_triangle() {
        use crate::mesh::{Mesh, Vertex, Triangle};
        use glam::Vec3;

        let cs = 16;
        // Triangle with edge length > 4.0 (default max_edge_length)
        let mesh = Mesh {
            vertices: vec![
                Vertex { position: Vec3::new(1.0, 1.0, 1.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(10.0, 1.0, 1.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(1.0, 2.0, 1.0), normal: Vec3::Y, material: Material::Limestone },
            ],
            triangles: vec![Triangle { indices: [0, 1, 2] }],
        };

        let mut meshes = HashMap::new();
        meshes.insert((0, 0, 0), mesh);
        let fields = HashMap::new();
        let seam = HashMap::new();
        let worms = Vec::new();

        let result = scan_world_with_config(&fields, &meshes, None, &seam, &worms, cs, &ScanConfig::default());
        let stretched: Vec<_> = result.issues.iter()
            .filter(|i| i.issue_type == IssueType::StretchedTriangle)
            .collect();
        assert!(!stretched.is_empty(), "Should detect stretched triangle with edge > 4.0");
    }

    #[test]
    fn test_worm_carve_verify() {
        let cs = 16;
        // Solid chunk with a worm path through it — worm not actually carved
        let density = make_density(cs, true);

        let mut fields = HashMap::new();
        fields.insert((0, 0, 0), density);

        let meshes = HashMap::new();
        let seam = HashMap::new();

        // Worm segment in the middle of the solid chunk
        let worm = vec![
            ScanWormSegment { position: [8.0, 8.0, 8.0], radius: 3.0 },
        ];
        let worms = vec![worm];

        let result = scan_world_with_config(&fields, &meshes, None, &seam, &worms, cs, &ScanConfig::default());
        let carve_fail: Vec<_> = result.issues.iter()
            .filter(|i| i.issue_type == IssueType::WormCarveFailure)
            .collect();
        assert!(!carve_fail.is_empty(), "Should detect solid density at worm segment center");
    }

    #[test]
    fn test_self_intersection() {
        use crate::mesh::{Mesh, Vertex, Triangle};
        use glam::Vec3;

        let cs = 16;
        // Two non-adjacent triangles with overlapping AABBs
        let mesh = Mesh {
            vertices: vec![
                // Triangle 0
                Vertex { position: Vec3::new(4.0, 4.0, 4.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(6.0, 4.0, 4.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(5.0, 6.0, 4.0), normal: Vec3::Y, material: Material::Limestone },
                // Triangle 1 - overlapping AABB but no shared vertices
                Vertex { position: Vec3::new(4.5, 4.5, 4.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(5.5, 4.5, 4.0), normal: Vec3::Y, material: Material::Limestone },
                Vertex { position: Vec3::new(5.0, 5.5, 4.0), normal: Vec3::Y, material: Material::Limestone },
            ],
            triangles: vec![
                Triangle { indices: [0, 1, 2] },
                Triangle { indices: [3, 4, 5] },
            ],
        };

        let mut meshes = HashMap::new();
        meshes.insert((0, 0, 0), mesh);
        let fields = HashMap::new();
        let seam = HashMap::new();
        let worms = Vec::new();

        let mut config = ScanConfig::default();
        config.enable_self_intersection = true;

        let result = scan_world_with_config(&fields, &meshes, None, &seam, &worms, cs, &config);
        let si: Vec<_> = result.issues.iter()
            .filter(|i| i.issue_type == IssueType::SelfIntersection)
            .collect();
        assert!(!si.is_empty(), "Should detect self-intersection from overlapping non-adjacent triangles");
    }

    #[test]
    fn test_scan_config_disable_all() {
        let cs = 16;
        let density = make_cave_density(cs);

        let mut fields = HashMap::new();
        fields.insert((0, 0, 0), density);

        let meshes = HashMap::new();
        let seam = HashMap::new();
        let worms = Vec::new();

        let config = ScanConfig {
            enable_density_seam: false,
            enable_mesh_topology: false,
            enable_seam_completeness: false,
            enable_navigability: false,
            enable_worm_truncation: false,
            enable_thin_walls: false,
            enable_winding_consistency: false,
            enable_degenerate_triangles: false,
            enable_worm_carve_verify: false,
            enable_self_intersection: false,
            enable_seam_mesh_quality: false,
            ..ScanConfig::default()
        };

        let result = scan_world_with_config(&fields, &meshes, None, &seam, &worms, cs, &config);
        assert_eq!(result.issues.len(), 0, "All passes disabled should produce 0 issues, got {}", result.issues.len());
    }

    #[test]
    fn test_scan_config_in_json() {
        let cs = 4;
        let mut fields = HashMap::new();
        fields.insert((0, 0, 0), make_density(cs, true));

        let meshes = HashMap::new();
        let seam = HashMap::new();
        let worms = Vec::new();

        let result = scan_world_with_config(&fields, &meshes, None, &seam, &worms, cs, &ScanConfig::default());
        let json = result.to_json_string();

        let parsed: serde_json::Value = serde_json::from_str(&json)
            .expect("Should serialize to valid JSON");

        assert!(parsed.get("scan_config").is_some(), "scan_config should appear in JSON output");
        let sc = parsed.get("scan_config").unwrap();
        assert!(sc.get("enable_density_seam").is_some());
        assert!(sc.get("enable_self_intersection").is_some());
        assert_eq!(sc.get("enable_self_intersection").unwrap().as_bool(), Some(false));
        assert_eq!(sc.get("enable_density_seam").unwrap().as_bool(), Some(true));
    }
}
