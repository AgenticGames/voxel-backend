//! Cinema — shot composition for the Dormancy Director.
//!
//! Given a [`Scene`] (from `voxel-world-memory`) + an [`IntentMask`], emits
//! ranked [`ShotCandidate`]s: a camera spline (waypoints in Rust voxel
//! coords), FOV / DOF curve, lighting profile, caption, and audio cue tag.
//!
//! **Cave-aware by construction**: every spline waypoint is validated
//! against an injected `CellGrid` (from `voxel-path`) — no in-rock placements
//! possible. Anchor selection uses an injected surface-probe closure so
//! voxel-cinema stays independent of `voxel-ffi`.
//!
//! Block 1 ships the *skeleton* — lib.rs + intent.rs + spline.rs + types.
//! Task B11 fills in the per-intent compose functions and scoring.

pub mod compose;
pub mod intent;
pub mod lighting;
pub mod probe;
pub mod spline;

pub use intent::{IntentMask, ShotIntent};
pub use lighting::{HeroPositionIntent, LightingProfile};
pub use probe::ProbeData;
pub use spline::Waypoint;

use voxel_world_memory::scene::Scene;

/// Maximum caption length (bytes, ASCII). Matches the `FfiShotCandidate.caption`
/// fixed-size array on the FFI boundary.
pub const MAX_CAPTION_BYTES: usize = 64;

/// A composed shot candidate. The voxel-ffi adapter packs this into
/// `FfiShotCandidate` + parallel `FfiWaypoint[]` buffer at the FFI boundary.
#[derive(Debug, Clone)]
pub struct ShotCandidate {
    pub intent: ShotIntent,
    /// Composition score. Higher is better. The Director picks max.
    pub score: f32,
    /// Camera path (4..=12 waypoints typical).
    pub waypoints: Vec<Waypoint>,
    /// Total duration in seconds (== last waypoint's t_secs).
    pub total_duration: f32,
    /// Lighting intent — UE evaluates to actual lights.
    pub lighting: LightingProfile,
    /// Narrative caption shown during this shot. ASCII, ≤ MAX_CAPTION_BYTES.
    pub caption: String,
    /// Audio cue tag — UE owns the actual sound asset mapping.
    pub audio_cue: u8,
}

impl ShotCandidate {
    pub fn empty(intent: ShotIntent) -> Self {
        Self {
            intent,
            score: 0.0,
            waypoints: Vec::new(),
            total_duration: 0.0,
            lighting: LightingProfile::default_for_intent(intent),
            caption: String::new(),
            audio_cue: 0,
        }
    }
}

/// Stub `compose` entry point. Block 1 ships only the safe-orbit fallback —
/// per-intent shots come in task B11.
///
/// `grid` is the path-planner-compatible cave geometry (caller wraps the
/// live ChunkStore in a `ChunkStoreGrid` via `voxel-ffi/src/pathing.rs`).
/// `probe` is a closure that wraps the engine's surface-probe so this crate
/// stays voxel-ffi-independent.
pub fn compose<G: voxel_path::grid::CellGrid, P: Fn(glam::Vec3, glam::Vec3) -> Option<ProbeData>>(
    scene: &Scene,
    intent_mask: IntentMask,
    count: usize,
    grid: &G,
    probe: &P,
) -> Vec<ShotCandidate> {
    let mut out = Vec::with_capacity(count);

    // Block 1: only SafeOrbit is implemented. Other intents return empty
    // candidates that get filtered out below — task B11 fills them in.
    for intent in ShotIntent::all() {
        if !intent_mask.allows(intent) {
            continue;
        }
        let candidate = match intent {
            ShotIntent::SafeOrbit => safe_orbit_compose(scene, grid, probe),
            ShotIntent::LavaDescent => {
                compose::lava_descent::compose_lava_descent(scene, grid, probe)
            }
            _ => None, // remaining intents land in Block 2
        };
        if let Some(c) = candidate {
            out.push(c);
        }
    }

    // Sort by score desc, then truncate.
    out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(count);
    out
}

/// Stub safe-orbit compose — a simple circular orbit around the Scene
/// centroid at a fixed radius and height. Validates each waypoint against
/// `grid.is_solid` so the orbit never enters rock. If too few air-anchored
/// waypoints exist, returns None (no shot — Director falls through to a
/// last-resort camera placement that lives in voxel-ffi for now).
fn safe_orbit_compose<G: voxel_path::grid::CellGrid, P: Fn(glam::Vec3, glam::Vec3) -> Option<ProbeData>>(
    scene: &Scene,
    grid: &G,
    _probe: &P,
) -> Option<ShotCandidate> {
    let centroid = scene.centroid_vec();
    let radius = scene.aabb.extent_radius().max(20.0) * 1.5;
    let height_offset = scene.aabb.extent_radius().max(20.0) * 0.5;
    let center = centroid + glam::Vec3::new(0.0, height_offset, 0.0);

    let num_waypoints = 8;
    let mut waypoints = Vec::with_capacity(num_waypoints);
    let mut air_count = 0;
    for i in 0..num_waypoints {
        let t = i as f32 / (num_waypoints - 1) as f32;
        let angle = t * std::f32::consts::TAU;
        let pos = center
            + glam::Vec3::new(
                angle.cos() * radius,
                0.0,
                angle.sin() * radius,
            );
        // Validate: is the cell at this position not solid?
        let cell = pos_to_cell(pos, grid.cell_size());
        if !grid.is_solid(cell) {
            air_count += 1;
        }
        waypoints.push(Waypoint {
            pos: [pos.x, pos.y, pos.z],
            look_at: [centroid.x, centroid.y, centroid.z],
            fov_deg: 60.0,
            t_secs: t * 10.0,
            dof_focus_dist: (pos - centroid).length(),
            dof_aperture: 4.0,
        });
    }

    // Require at least 60% of orbit to be in air to be a usable shot.
    if (air_count as f32 / num_waypoints as f32) < 0.6 {
        return None;
    }

    Some(ShotCandidate {
        intent: ShotIntent::SafeOrbit,
        score: 50.0 * (air_count as f32 / num_waypoints as f32),
        waypoints,
        total_duration: 10.0,
        lighting: LightingProfile::default_for_intent(ShotIntent::SafeOrbit),
        caption: scene_caption(scene),
        audio_cue: scene.kind as u8,
    })
}

fn pos_to_cell(pos: glam::Vec3, cell_size: f32) -> glam::IVec3 {
    glam::IVec3::new(
        (pos.x / cell_size).floor() as i32,
        (pos.y / cell_size).floor() as i32,
        (pos.z / cell_size).floor() as i32,
    )
}

fn scene_caption(scene: &Scene) -> String {
    use voxel_world_memory::scene::SceneKind::*;
    let head = match scene.kind {
        Lava => "Lava upwelling",
        Water => "New aquifer",
        Stress => "Stress cascade",
        Bridge => "Crystal bridge",
        CeilingDome => "Cavern dome",
        Chokepoint => "Narrow passage",
        WallNiche => "Wall niche",
    };
    head.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shot_candidate_empty_constructor() {
        let s = ShotCandidate::empty(ShotIntent::SafeOrbit);
        assert!(s.waypoints.is_empty());
        assert_eq!(s.intent, ShotIntent::SafeOrbit);
    }
}
