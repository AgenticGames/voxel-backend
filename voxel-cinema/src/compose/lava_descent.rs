//! LavaDescent — camera descends from above the scene toward the heat
//! source, spiraling inward. Demonstrative implementation of a non-trivial
//! shot intent; remaining intents follow this pattern in Block 2.

use glam::Vec3;
use voxel_path::grid::CellGrid;
use voxel_world_memory::scene::{Scene, SceneKind};

use crate::intent::ShotIntent;
use crate::lighting::LightingProfile;
use crate::probe::ProbeData;
use crate::spline::Waypoint;
use crate::ShotCandidate;

/// Compose a LavaDescent shot for the given Scene. Returns None if the
/// scene isn't a Lava kind, or if too few waypoints land in air to be
/// playable.
pub fn compose_lava_descent<
    G: CellGrid,
    P: Fn(Vec3, Vec3) -> Option<ProbeData>,
>(
    scene: &Scene,
    grid: &G,
    _probe: &P,
) -> Option<ShotCandidate> {
    if scene.kind != SceneKind::Lava {
        return None;
    }

    let centroid = scene.centroid_vec();
    let extent = scene.aabb.extent_radius().max(20.0);

    // Start position: ~2.5× extent above centroid, offset to one side so
    // the descent traces a spiral instead of a vertical drop.
    let start_height = extent * 2.5;
    let start_radius = extent * 1.8;
    let end_height = extent * 0.4; // hover just above the lava
    let end_radius = extent * 0.4; // pulled in tight at the end

    let num_waypoints = 10;
    let total_duration = 9.0;

    let mut waypoints = Vec::with_capacity(num_waypoints);
    let mut air_count = 0;

    for i in 0..num_waypoints {
        let t = i as f32 / (num_waypoints - 1) as f32;
        // Spiral down: angle sweeps π over the shot. Radius shrinks from
        // start_radius to end_radius. Height drops from start_height to
        // end_height.
        let angle = t * std::f32::consts::PI;
        let radius = start_radius * (1.0 - t) + end_radius * t;
        let height = start_height * (1.0 - t) + end_height * t;

        let pos = centroid
            + Vec3::new(angle.cos() * radius, height, angle.sin() * radius);

        // Validate against the cave grid.
        let cell = pos_to_cell(pos, grid.cell_size());
        if !grid.is_solid(cell) {
            air_count += 1;
        }

        // FOV widens slightly as we descend (drama).
        let fov = 60.0 + t * 10.0;
        // DOF focus follows the centroid so the lava stays sharp.
        let focus_dist = (pos - centroid).length().max(1.0);

        waypoints.push(Waypoint {
            pos: [pos.x, pos.y, pos.z],
            look_at: [centroid.x, centroid.y, centroid.z],
            fov_deg: fov,
            t_secs: t * total_duration,
            dof_focus_dist: focus_dist,
            dof_aperture: 2.8, // shallow DOF, lava in focus
        });
    }

    // Require at least 60% of waypoints in air for the shot to play.
    let air_frac = air_count as f32 / num_waypoints as f32;
    if air_frac < 0.6 {
        return None;
    }

    // Composition score: extent (bigger lava = more dramatic) × air_frac
    // (cleaner trajectory = better shot). Capped at 200 to keep within
    // sane comparison range with SafeOrbit (which scores ~50).
    let score = (extent * 1.5 + air_frac * 60.0).min(200.0);

    Some(ShotCandidate {
        intent: ShotIntent::LavaDescent,
        score,
        waypoints,
        total_duration,
        lighting: LightingProfile::default_for_intent(ShotIntent::LavaDescent),
        caption: caption_for_lava_scene(scene),
        audio_cue: SceneKind::Lava as u8,
    })
}

fn pos_to_cell(pos: Vec3, cell_size: f32) -> glam::IVec3 {
    glam::IVec3::new(
        (pos.x / cell_size).floor() as i32,
        (pos.y / cell_size).floor() as i32,
        (pos.z / cell_size).floor() as i32,
    )
}

fn caption_for_lava_scene(scene: &Scene) -> String {
    if scene.tags.has(voxel_world_memory::scene::SceneTags::FRESH) {
        "A new lava chamber stirs.".to_string()
    } else {
        "The lava chamber breathes.".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_world_memory::scene::{Scene, SceneId};

    // Trivial CellGrid stub: everything is air.
    struct AirGrid;
    impl CellGrid for AirGrid {
        fn cell_size(&self) -> f32 {
            1.0
        }
        fn is_solid(&self, _cell: glam::IVec3) -> bool {
            false
        }
        fn surface_normal_at(&self, _cell: glam::IVec3) -> Vec3 {
            Vec3::ZERO
        }
    }

    struct SolidGrid;
    impl CellGrid for SolidGrid {
        fn cell_size(&self) -> f32 {
            1.0
        }
        fn is_solid(&self, _cell: glam::IVec3) -> bool {
            true
        }
        fn surface_normal_at(&self, _cell: glam::IVec3) -> Vec3 {
            Vec3::Y
        }
    }

    fn make_lava_scene() -> Scene {
        let mut s = Scene::new(SceneId(1), SceneKind::Lava, Vec3::new(100.0, 100.0, 100.0));
        s.score = 200.0;
        s.confidence = 0.9;
        s.aabb = voxel_world_memory::scene::Aabb {
            min: [80.0, 80.0, 80.0],
            max: [120.0, 120.0, 120.0],
        };
        s
    }

    #[test]
    fn lava_scene_in_air_produces_descent() {
        let scene = make_lava_scene();
        let grid = AirGrid;
        let probe = |_a: Vec3, _b: Vec3| None;
        let c = compose_lava_descent(&scene, &grid, &probe);
        assert!(c.is_some());
        let c = c.unwrap();
        assert_eq!(c.intent, ShotIntent::LavaDescent);
        assert!(c.waypoints.len() >= 8);
        // Height monotonically descends.
        let first_y = c.waypoints[0].pos[1];
        let last_y = c.waypoints.last().unwrap().pos[1];
        assert!(first_y > last_y, "descent should drop in Y");
        // Composition score should be > SafeOrbit baseline (~50).
        assert!(c.score > 50.0);
    }

    #[test]
    fn non_lava_scene_returns_none() {
        let mut s = make_lava_scene();
        s.kind = SceneKind::Water;
        let grid = AirGrid;
        let probe = |_a: Vec3, _b: Vec3| None;
        assert!(compose_lava_descent(&s, &grid, &probe).is_none());
    }

    #[test]
    fn fully_solid_grid_rejects_shot() {
        let scene = make_lava_scene();
        let grid = SolidGrid;
        let probe = |_a: Vec3, _b: Vec3| None;
        let c = compose_lava_descent(&scene, &grid, &probe);
        // All waypoints in solid → air_frac=0 → rejected.
        assert!(c.is_none());
    }

    #[test]
    fn fresh_scene_uses_different_caption() {
        let mut s = make_lava_scene();
        s.tags.set(voxel_world_memory::scene::SceneTags::FRESH);
        let grid = AirGrid;
        let probe = |_a: Vec3, _b: Vec3| None;
        let c = compose_lava_descent(&s, &grid, &probe).unwrap();
        assert!(c.caption.contains("new"));
    }
}
