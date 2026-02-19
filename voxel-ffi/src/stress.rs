// Re-export stress types and functions from voxel-core.
// All logic now lives in voxel-core::stress for reuse by voxel-sleep and other crates.

pub use voxel_core::stress::{
    OverstressedVoxel,
    StressResult,
    CollapsedVoxel,
    RubbleVoxel,
    CollapseEvent,
    world_to_chunk_local,
    calc_voxel_stress,
    recalc_stress_region,
    detect_and_execute_collapses,
    post_change_stress_update,
    post_change_stress_update_with_iterations,
};
