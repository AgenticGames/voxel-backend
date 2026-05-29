//! World-scan result + config FFI types.

use super::*;

/// FFI result for world scan. JSON report is passed as a heap-allocated string.
#[repr(C)]
pub struct FfiWorldScanResult {
    pub success: u32,
    pub json_report: *mut std::ffi::c_char,
    pub json_length: u32,
    pub chunks_scanned: u32,
    pub total_issues: u32,
    pub total_errors: u32,
    pub total_warnings: u32,
}

/// FFI-safe scan configuration. Uses u32 for booleans (C ABI compatibility).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiScanConfig {
    // Enable flags (u32: 0=false, nonzero=true)
    pub enable_density_seam: u32,
    pub enable_mesh_topology: u32,
    pub enable_seam_completeness: u32,
    pub enable_navigability: u32,
    pub enable_worm_truncation: u32,
    pub enable_thin_walls: u32,
    pub enable_winding_consistency: u32,
    pub enable_degenerate_triangles: u32,
    pub enable_worm_carve_verify: u32,
    pub enable_self_intersection: u32,
    pub enable_seam_mesh_quality: u32,
    // Accuracy params
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
