//! Spline waypoints + Catmull-Rom interpolation.
//!
//! A `ShotCandidate` carries an ordered list of `Waypoint`s. UE-side
//! playback samples between waypoints using whatever curve it prefers
//! (Block 1 ships only the data shape; CR sample helpers in this module
//! exist primarily so voxel-ffi adapter + tests can validate / debug-render
//! the path).

use glam::Vec3;
use serde::{Deserialize, Serialize};

/// A single keyframe on the camera spline.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Waypoint {
    /// World position (Rust voxel coords).
    pub pos: [f32; 3],
    /// Look-at target (Rust voxel coords).
    pub look_at: [f32; 3],
    /// Field of view in degrees at this keyframe.
    pub fov_deg: f32,
    /// Timestamp in seconds from shot start.
    pub t_secs: f32,
    /// Depth-of-field focus distance.
    pub dof_focus_dist: f32,
    /// Depth-of-field aperture (f-stop-ish; UE realizes it).
    pub dof_aperture: f32,
}

impl Waypoint {
    pub fn pos_vec(&self) -> Vec3 {
        Vec3::new(self.pos[0], self.pos[1], self.pos[2])
    }

    pub fn look_at_vec(&self) -> Vec3 {
        Vec3::new(self.look_at[0], self.look_at[1], self.look_at[2])
    }
}

/// Sample the camera spline at time `t` (seconds from shot start). Returns
/// (position, look-at). Uses Catmull-Rom interpolation between the four
/// surrounding waypoints; falls back to linear at endpoints.
pub fn sample_at(waypoints: &[Waypoint], t: f32) -> Option<(Vec3, Vec3)> {
    if waypoints.is_empty() {
        return None;
    }
    if waypoints.len() == 1 {
        return Some((waypoints[0].pos_vec(), waypoints[0].look_at_vec()));
    }

    // Find the segment containing t.
    let last_idx = waypoints.len() - 1;
    let t_first = waypoints[0].t_secs;
    let t_last = waypoints[last_idx].t_secs;

    if t <= t_first {
        return Some((waypoints[0].pos_vec(), waypoints[0].look_at_vec()));
    }
    if t >= t_last {
        return Some((waypoints[last_idx].pos_vec(), waypoints[last_idx].look_at_vec()));
    }

    let mut seg = 0;
    for i in 0..last_idx {
        if t >= waypoints[i].t_secs && t < waypoints[i + 1].t_secs {
            seg = i;
            break;
        }
    }

    let p0 = if seg == 0 { waypoints[0] } else { waypoints[seg - 1] };
    let p1 = waypoints[seg];
    let p2 = waypoints[seg + 1];
    let p3 = if seg + 2 > last_idx { waypoints[last_idx] } else { waypoints[seg + 2] };

    let seg_t = (t - p1.t_secs) / (p2.t_secs - p1.t_secs).max(1e-6);
    let pos = catmull_rom(p0.pos_vec(), p1.pos_vec(), p2.pos_vec(), p3.pos_vec(), seg_t);
    let look = catmull_rom(
        p0.look_at_vec(),
        p1.look_at_vec(),
        p2.look_at_vec(),
        p3.look_at_vec(),
        seg_t,
    );
    Some((pos, look))
}

fn catmull_rom(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * (2.0 * p1
        + (p2 - p0) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wp(x: f32, y: f32, z: f32, t: f32) -> Waypoint {
        Waypoint {
            pos: [x, y, z],
            look_at: [0.0, 0.0, 0.0],
            fov_deg: 60.0,
            t_secs: t,
            dof_focus_dist: 100.0,
            dof_aperture: 4.0,
        }
    }

    #[test]
    fn empty_sample_returns_none() {
        assert!(sample_at(&[], 0.5).is_none());
    }

    #[test]
    fn single_waypoint_returns_constant() {
        let wps = [wp(10.0, 20.0, 30.0, 0.0)];
        let (p, _) = sample_at(&wps, 5.0).unwrap();
        assert_eq!(p, Vec3::new(10.0, 20.0, 30.0));
    }

    #[test]
    fn pre_first_clamps_to_first() {
        let wps = [wp(0.0, 0.0, 0.0, 5.0), wp(10.0, 0.0, 0.0, 10.0)];
        let (p, _) = sample_at(&wps, 0.0).unwrap();
        assert_eq!(p, Vec3::ZERO);
    }

    #[test]
    fn post_last_clamps_to_last() {
        let wps = [wp(0.0, 0.0, 0.0, 0.0), wp(10.0, 0.0, 0.0, 5.0)];
        let (p, _) = sample_at(&wps, 100.0).unwrap();
        assert_eq!(p, Vec3::new(10.0, 0.0, 0.0));
    }

    #[test]
    fn midpoint_interpolation_is_smooth() {
        let wps = [
            wp(0.0, 0.0, 0.0, 0.0),
            wp(5.0, 0.0, 0.0, 1.0),
            wp(10.0, 0.0, 0.0, 2.0),
            wp(15.0, 0.0, 0.0, 3.0),
        ];
        // Midpoint of segment 1→2 (t=1.5) should be roughly 7.5
        let (p, _) = sample_at(&wps, 1.5).unwrap();
        assert!((p.x - 7.5).abs() < 0.2);
    }
}
