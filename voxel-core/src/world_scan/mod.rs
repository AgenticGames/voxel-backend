//! World scan: machine-readable hole detection & topology diagnostics.
//!
//! Produces a structured JSON report for iterative fix loops.
//! Used by both CLI (`voxel-cli scan`) and FFI (`voxel_request_world_scan`).

mod density;
mod mesh;
mod seam;
mod volume;
mod worm;
mod winding;

use std::collections::HashMap;
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
        density::pass_density_seam(density_fields, chunk_size, config, &mut issues);
    }

    // Pass 2: Mesh topology
    if config.enable_mesh_topology {
        mesh::pass_mesh_topology(base_meshes, chunk_size, &mut issues);
    }

    // Pass 3: Seam completeness
    if config.enable_seam_completeness {
        seam::pass_seam_completeness(seam_data, base_meshes, density_fields, chunk_size, &mut issues);
    }

    // Pass 4: Cross-chunk navigability + volume classification
    let volumes = if config.enable_navigability {
        volume::pass_navigability(density_fields, chunk_size, &mut issues)
    } else {
        Vec::new()
    };

    // Pass 5: Worm truncation
    if config.enable_worm_truncation {
        worm::pass_worm_truncation(worm_paths, density_fields, chunk_size, &mut issues);
    }

    // Pass 6: Thin walls
    if config.enable_thin_walls {
        volume::pass_thin_walls(density_fields, chunk_size, &mut issues);
    }

    // Pass E2: Raymarch holes
    if config.raymarch_rays_per_chunk > 0 {
        mesh::pass_raymarch_holes(density_fields, base_meshes, chunk_size, config, &mut issues);
    }

    // Pass E3: Mesh-density alignment
    if hermite_data.is_some() {
        mesh::pass_mesh_density_alignment(base_meshes, hermite_data.unwrap(), chunk_size, config, &mut issues);
    }

    // Pass E4: Narrow passage
    if config.enable_navigability {
        volume::pass_narrow_passage(&volumes, config, &mut issues);
    }

    // Pass N1: Winding consistency
    if config.enable_winding_consistency {
        winding::pass_winding_consistency(base_meshes, chunk_size, &mut issues);
    }

    // Pass N2: Triangle quality
    if config.enable_degenerate_triangles {
        winding::pass_triangle_quality(base_meshes, chunk_size, config, &mut issues);
    }

    // Pass N3: Worm carve verify
    if config.enable_worm_carve_verify {
        worm::pass_worm_carve_verify(worm_paths, density_fields, chunk_size, &mut issues);
    }

    // Pass N4: Self intersection
    if config.enable_self_intersection {
        winding::pass_self_intersection(base_meshes, chunk_size, config, &mut issues);
    }

    // Pass N5: Seam mesh quality
    if config.enable_seam_mesh_quality {
        seam::pass_seam_mesh_quality(seam_data, density_fields, chunk_size, &mut issues);
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
