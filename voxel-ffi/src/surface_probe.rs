//! Voxel-aware surface probe — answers "what's at this UE world point?" by
//! sampling the live density field through the `ChunkStore`. Returns surface
//! kind, averaged normal, largest empty-cavity radius, and per-axis clearance.
//!
//! Used by spider-nest / wasp-hive placement validators to confirm a
//! candidate is anchored to a real surface of the right kind (floor / wall
//! / ceiling) with enough cavity room around it. Cheap enough for tens of
//! queries per cluster-spawn event (~once per 15 s of ambient gameplay).
//!
//! All sampling is done in **Rust voxel coordinates**. The FFI layer in
//! [`crate::api::voxel_query_surface`] converts the result's normal and
//! clearance array back into UE space.
//!
//! Surface kind semantics (output is the [`SurfaceKind`] enum as `u8`):
//!   - `Solid`    — probe point is inside rock.
//!   - `AirOpen`  — probe point is air with no solid neighbor within 2 voxels.
//!   - `Floor`    — air on a near-horizontal floor (averaged normal.y > 0.85).
//!   - `Ceiling`  — air with rock above (averaged normal.y < -0.85).
//!   - `Wall`     — air with rock alongside, normal near-horizontal (|n.y| < 0.5).
//!   - `Overhang` — air with rock above-and-to-side, 0.5 ≤ |n.y| ≤ 0.85.

use glam::Vec3;

use crate::store::ChunkStore;

/// Surface-kind output of [`probe_surface`]. Matches the u8 encoding in
/// `FfiSurfaceProbe.kind`.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceKind {
    Solid = 0,
    AirOpen = 1,
    Floor = 2,
    Wall = 3,
    Ceiling = 4,
    Overhang = 5,
}

/// Cap on cavity-radius and per-axis clearance reads, in voxels. Placement
/// validators don't care beyond ~8 voxels — anything larger is "plenty of
/// room" and we save the sampling cost.
const MAX_PROBE_VOXELS: i32 = 8;

/// 14 step directions for the cavity-radius scan: 6 axis cardinals + 8
/// long-diagonal corners.
const PROBE_DIRECTIONS: [(i32, i32, i32); 14] = [
    ( 1, 0, 0), (-1, 0, 0),
    ( 0, 1, 0), ( 0,-1, 0),
    ( 0, 0, 1), ( 0, 0,-1),
    ( 1, 1, 1), (-1, 1, 1), ( 1,-1, 1), (-1,-1, 1),
    ( 1, 1,-1), (-1, 1,-1), ( 1,-1,-1), (-1,-1,-1),
];

/// Sample one world-voxel as solid or air. Treats unloaded chunks as solid
/// — placement validators should refuse to anchor against unloaded geometry
/// (the player can't see it either, and it may be carved away on stream-in).
fn is_solid_at(store: &ChunkStore, wx: i32, wy: i32, wz: i32, chunk_size: usize) -> bool {
    let cs_i = chunk_size as i32;
    let cx = wx.div_euclid(cs_i);
    let cy = wy.div_euclid(cs_i);
    let cz = wz.div_euclid(cs_i);
    let lx = wx.rem_euclid(cs_i) as usize;
    let ly = wy.rem_euclid(cs_i) as usize;
    let lz = wz.rem_euclid(cs_i) as usize;
    match store.density_fields.get(&(cx, cy, cz)) {
        Some(df) => df.get(lx, ly, lz).material.is_solid(),
        None => true,
    }
}

/// Step from `origin` in the given integer direction, returning the
/// Euclidean distance (in voxels) at which the first solid voxel was hit,
/// capped at `MAX_PROBE_VOXELS` worth of steps.
fn distance_to_solid(
    store: &ChunkStore,
    ox: i32, oy: i32, oz: i32,
    dx: i32, dy: i32, dz: i32,
    chunk_size: usize,
) -> f32 {
    let step_len = ((dx * dx + dy * dy + dz * dz) as f32).sqrt();
    for step in 1..=MAX_PROBE_VOXELS {
        let sx = ox + dx * step;
        let sy = oy + dy * step;
        let sz = oz + dz * step;
        if is_solid_at(store, sx, sy, sz, chunk_size) {
            return (step - 1) as f32 * step_len;
        }
    }
    MAX_PROBE_VOXELS as f32 * step_len
}

/// In-Rust-space probe result. The FFI layer converts `normal` and
/// `clearance_rust` into UE space before returning to the caller.
#[derive(Debug, Clone, Copy)]
pub struct ProbeResult {
    pub kind: SurfaceKind,
    /// Unit normal in Rust coords (Y-up), pointing from rock toward air.
    /// When the probe is in open air or fully solid the gradient vanishes;
    /// in that case the function falls back to the caller's `normal_hint`.
    pub normal: Vec3,
    /// Largest empty-sphere radius centered on the probe, in voxels. Capped
    /// at `MAX_PROBE_VOXELS * sqrt(3)`.
    pub cavity_radius: f32,
    /// Distance to nearest solid in Rust axis order: +X, -X, +Y, -Y, +Z, -Z.
    pub clearance_rust: [f32; 6],
}

/// Sample the density field at `rust_pos` and produce a [`ProbeResult`].
///
/// `normal_hint` is used only as a fallback when the local density gradient
/// is flat (open air or solid interior). It should be a unit vector in Rust
/// coords; callers that don't have one can pass `Vec3::Y`.
pub fn probe_surface(
    store: &ChunkStore,
    rust_pos: Vec3,
    chunk_size: usize,
    normal_hint: Vec3,
) -> ProbeResult {
    let ox = rust_pos.x.round() as i32;
    let oy = rust_pos.y.round() as i32;
    let oz = rust_pos.z.round() as i32;

    let origin_solid = is_solid_at(store, ox, oy, oz, chunk_size);

    // Averaged gradient over a 3x3x3 cell — points from rock toward air.
    let mut nx = 0.0_f32;
    let mut ny = 0.0_f32;
    let mut nz = 0.0_f32;
    for dz in -1..=1i32 {
        for dy in -1..=1i32 {
            for dx in -1..=1i32 {
                let s_minus_x = is_solid_at(store, ox + dx - 1, oy + dy, oz + dz, chunk_size) as i32;
                let s_plus_x  = is_solid_at(store, ox + dx + 1, oy + dy, oz + dz, chunk_size) as i32;
                let s_minus_y = is_solid_at(store, ox + dx, oy + dy - 1, oz + dz, chunk_size) as i32;
                let s_plus_y  = is_solid_at(store, ox + dx, oy + dy + 1, oz + dz, chunk_size) as i32;
                let s_minus_z = is_solid_at(store, ox + dx, oy + dy, oz + dz - 1, chunk_size) as i32;
                let s_plus_z  = is_solid_at(store, ox + dx, oy + dy, oz + dz + 1, chunk_size) as i32;
                nx += (s_minus_x - s_plus_x) as f32;
                ny += (s_minus_y - s_plus_y) as f32;
                nz += (s_minus_z - s_plus_z) as f32;
            }
        }
    }
    let normal = if nx * nx + ny * ny + nz * nz < 1e-6 {
        if normal_hint.length_squared() > 1e-6 {
            normal_hint.normalize()
        } else {
            Vec3::Y
        }
    } else {
        Vec3::new(nx, ny, nz).normalize()
    };

    let clearance = [
        distance_to_solid(store, ox, oy, oz,  1, 0, 0, chunk_size),
        distance_to_solid(store, ox, oy, oz, -1, 0, 0, chunk_size),
        distance_to_solid(store, ox, oy, oz,  0, 1, 0, chunk_size),
        distance_to_solid(store, ox, oy, oz,  0,-1, 0, chunk_size),
        distance_to_solid(store, ox, oy, oz,  0, 0, 1, chunk_size),
        distance_to_solid(store, ox, oy, oz,  0, 0,-1, chunk_size),
    ];

    let mut cavity_radius = (MAX_PROBE_VOXELS as f32) * (3.0_f32).sqrt();
    for &(dx, dy, dz) in &PROBE_DIRECTIONS {
        let d = distance_to_solid(store, ox, oy, oz, dx, dy, dz, chunk_size);
        if d < cavity_radius {
            cavity_radius = d;
        }
    }

    let kind = if origin_solid {
        SurfaceKind::Solid
    } else {
        // Need to distinguish open-air from on-a-surface: if no axis sees a
        // solid voxel within 2 cells, treat as open air. Otherwise classify
        // by the averaged-normal up-component.
        let any_solid_near = clearance.iter().any(|&d| d <= 2.0);
        if !any_solid_near {
            SurfaceKind::AirOpen
        } else if normal.y > 0.85 {
            SurfaceKind::Floor
        } else if normal.y < -0.85 {
            SurfaceKind::Ceiling
        } else if normal.y.abs() < 0.5 {
            SurfaceKind::Wall
        } else {
            SurfaceKind::Overhang
        }
    };

    ProbeResult {
        kind,
        normal,
        cavity_radius,
        clearance_rust: clearance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::density::DensityField;
    use voxel_core::material::Material;

    fn empty_store() -> ChunkStore {
        ChunkStore::new(8)
    }

    fn fill_chunk(store: &mut ChunkStore, key: (i32, i32, i32), size: usize, material: Material) {
        let mut df = DensityField::new(size);
        for s in df.samples.iter_mut() {
            s.material = material;
            s.density = -1.0;
        }
        df.compute_metadata();
        store.density_fields.insert(key, df);
    }

    fn fill_air(store: &mut ChunkStore, key: (i32, i32, i32), size: usize) {
        let mut df = DensityField::new(size);
        for s in df.samples.iter_mut() {
            s.material = Material::Air;
            s.density = 1.0;
        }
        df.compute_metadata();
        store.density_fields.insert(key, df);
    }

    #[test]
    fn unloaded_chunk_classifies_as_solid() {
        let store = empty_store();
        let p = probe_surface(&store, Vec3::new(4.0, 4.0, 4.0), 8, Vec3::Y);
        assert_eq!(p.kind, SurfaceKind::Solid);
    }

    #[test]
    fn fully_air_chunk_classifies_as_airopen() {
        let mut store = empty_store();
        // Build a 3x3x3 block of air chunks around (0,0,0) so the probe at
        // the center can't see any solid even with the 8-voxel clearance reach.
        for cz in -1..=1 {
            for cy in -1..=1 {
                for cx in -1..=1 {
                    fill_air(&mut store, (cx, cy, cz), 9);
                }
            }
        }
        let p = probe_surface(&store, Vec3::new(4.0, 4.0, 4.0), 8, Vec3::Y);
        assert_eq!(p.kind, SurfaceKind::AirOpen);
    }

    #[test]
    fn floor_normal_points_up() {
        let mut store = empty_store();
        // chunk (0,0,0): bottom half rock, top half air — probe at (4, 4, 4)
        // straddles the boundary; the cell above the boundary is air.
        let size = 9;
        let mut df = DensityField::new(size);
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let s = df.get_mut(x, y, z);
                    if y < 3 {
                        s.material = Material::Limestone;
                        s.density = -1.0;
                    } else {
                        s.material = Material::Air;
                        s.density = 1.0;
                    }
                }
            }
        }
        df.compute_metadata();
        store.density_fields.insert((0, 0, 0), df);

        // Probe at y=3 — first row of air just above the rock floor.
        let p = probe_surface(&store, Vec3::new(4.0, 3.0, 4.0), 8, Vec3::Y);
        assert_eq!(p.kind, SurfaceKind::Floor, "expected Floor, got {:?}", p.kind);
        assert!(p.normal.y > 0.85, "expected normal.y > 0.85, got {}", p.normal.y);
    }

    #[test]
    fn ceiling_normal_points_down() {
        let mut store = empty_store();
        let size = 9;
        let mut df = DensityField::new(size);
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let s = df.get_mut(x, y, z);
                    if y > 4 {
                        s.material = Material::Limestone;
                        s.density = -1.0;
                    } else {
                        s.material = Material::Air;
                        s.density = 1.0;
                    }
                }
            }
        }
        df.compute_metadata();
        store.density_fields.insert((0, 0, 0), df);

        // Probe at y=4 — last row of air just below the ceiling.
        let p = probe_surface(&store, Vec3::new(4.0, 4.0, 4.0), 8, Vec3::Y);
        assert_eq!(p.kind, SurfaceKind::Ceiling, "expected Ceiling, got {:?}", p.kind);
        assert!(p.normal.y < -0.85, "expected normal.y < -0.85, got {}", p.normal.y);
    }

    #[test]
    fn cavity_radius_caps_at_max() {
        let mut store = empty_store();
        for cz in -1..=1 {
            for cy in -1..=1 {
                for cx in -1..=1 {
                    fill_air(&mut store, (cx, cy, cz), 9);
                }
            }
        }
        let p = probe_surface(&store, Vec3::new(4.0, 4.0, 4.0), 8, Vec3::Y);
        let expected_max = MAX_PROBE_VOXELS as f32;
        // Cardinal directions cap at MAX_PROBE_VOXELS; the min over all 14
        // is the cap (diagonals are larger so they don't tighten it).
        assert!(
            (p.cavity_radius - expected_max).abs() < 1e-3,
            "expected cavity_radius ≈ {}, got {}",
            expected_max, p.cavity_radius
        );
    }
}
