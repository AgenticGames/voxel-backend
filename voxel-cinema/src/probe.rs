//! ProbeData — a minimal surface-probe data shape, neutral to voxel-ffi.
//!
//! voxel-cinema doesn't depend on voxel-ffi, so we can't import
//! `voxel_ffi::surface_probe::ProbeResult`. Instead, voxel-ffi adapters
//! convert their `ProbeResult` into this shape at the call site.
//!
//! Field semantics match voxel-ffi/src/surface_probe.rs:127 exactly.

use serde::{Deserialize, Serialize};

/// Surface kind. Numeric values stable with voxel-ffi's `SurfaceKind` and
/// the FFI's `FfiSurfaceProbe.kind`.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SurfaceKind {
    Solid = 0,
    AirOpen = 1,
    Floor = 2,
    Wall = 3,
    Ceiling = 4,
    Overhang = 5,
}

impl SurfaceKind {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Solid),
            1 => Some(Self::AirOpen),
            2 => Some(Self::Floor),
            3 => Some(Self::Wall),
            4 => Some(Self::Ceiling),
            5 => Some(Self::Overhang),
            _ => None,
        }
    }
}

/// Surface probe result, voxel-cinema-neutral form. Caller wraps voxel-ffi's
/// `ProbeResult` into this struct when calling `compose`.
#[derive(Debug, Clone, Copy)]
pub struct ProbeData {
    pub kind: SurfaceKind,
    /// Unit normal in Rust coords (Y-up), pointing rock → air.
    pub normal: [f32; 3],
    /// Largest empty-sphere radius centered on the probe, in voxels.
    pub cavity_radius: f32,
    /// Distance to nearest solid in Rust axis order: +X, -X, +Y, -Y, +Z, -Z.
    pub clearance_rust: [f32; 6],
}

impl ProbeData {
    pub fn is_in_air(self) -> bool {
        matches!(
            self.kind,
            SurfaceKind::AirOpen | SurfaceKind::Floor | SurfaceKind::Wall | SurfaceKind::Ceiling | SurfaceKind::Overhang
        )
    }

    pub fn is_solid(self) -> bool {
        self.kind == SurfaceKind::Solid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_kind_roundtrip() {
        for v in 0u8..=5 {
            assert_eq!(SurfaceKind::from_u8(v).map(|k| k as u8), Some(v));
        }
        assert!(SurfaceKind::from_u8(99).is_none());
    }

    #[test]
    fn is_in_air_correct() {
        let air = ProbeData {
            kind: SurfaceKind::AirOpen,
            normal: [0.0, 1.0, 0.0],
            cavity_radius: 5.0,
            clearance_rust: [3.0; 6],
        };
        assert!(air.is_in_air());
        assert!(!air.is_solid());

        let rock = ProbeData {
            kind: SurfaceKind::Solid,
            normal: [0.0, 1.0, 0.0],
            cavity_radius: 0.0,
            clearance_rust: [0.0; 6],
        };
        assert!(!rock.is_in_air());
        assert!(rock.is_solid());
    }
}
