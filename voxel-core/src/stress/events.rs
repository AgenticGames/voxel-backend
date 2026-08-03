//! Stress dirty-event regions, surface/source classification constants,
//! and classification pack/unpack helpers.
//!
//! Behavior-preserving split of the former `stress.rs` god file.

/// Per-voxel stress values for a chunk. Same layout as DensityField (17^3 for chunk_size=16).
/// Classification byte: top 4 bits = surface type, bottom 4 bits = dominant stress source.
/// Surface types: 0=interior, 1=floor, 2=ceiling, 3=wall, 4=thin_feature
/// Stress sources: 0=none, 1=gravity, 2=overhang, 3=span, 4=cross_section
pub const SURFACE_INTERIOR: u8 = 0;
pub const SURFACE_FLOOR: u8 = 1;
pub const SURFACE_CEILING: u8 = 2;
pub const SURFACE_WALL: u8 = 3;
pub const SURFACE_THIN: u8 = 4;

pub const SOURCE_NONE: u8 = 0;
pub const SOURCE_GRAVITY: u8 = 1;
pub const SOURCE_OVERHANG: u8 = 2;
pub const SOURCE_SPAN: u8 = 3;
pub const SOURCE_CROSS_SECTION: u8 = 4;

/// A localized stress recalculation event centered on a mine point.
/// Only voxels within `radius` of `center` (world voxel coords) get recalculated.
#[derive(Debug, Clone)]
pub struct StressDirtyEvent {
    pub center: (i32, i32, i32),
    pub radius: i32,
    /// Whether the recalc batch this event belongs to may EXECUTE collapses
    /// (#214, 2026-08-03). Mining/removal legitimately push stress UP and may
    /// collapse. Strut PLACEMENT only relieves — but its recalc region
    /// (radius+4) re-evaluates rock whose latent overstress predates the
    /// strut, and executing those turns "place a brace" into "trigger the
    /// cave-in you were preventing". Placement queues with false; the worker
    /// skips the collapse pass when NO drained event allows it (stress and
    /// crack decals still rewrite).
    pub allow_collapse: bool,
}

impl StressDirtyEvent {
    /// Returns the set of chunk keys whose bounding boxes overlap this event's sphere.
    pub fn affected_chunks(&self, chunk_size: usize) -> Vec<(i32, i32, i32)> {
        let cs = chunk_size as i32;
        let min_cx = (self.center.0 - self.radius).div_euclid(cs);
        let max_cx = (self.center.0 + self.radius).div_euclid(cs);
        let min_cy = (self.center.1 - self.radius).div_euclid(cs);
        let max_cy = (self.center.1 + self.radius).div_euclid(cs);
        let min_cz = (self.center.2 - self.radius).div_euclid(cs);
        let max_cz = (self.center.2 + self.radius).div_euclid(cs);
        let mut keys = Vec::new();
        for cz in min_cz..=max_cz {
            for cy in min_cy..=max_cy {
                for cx in min_cx..=max_cx {
                    keys.push((cx, cy, cz));
                }
            }
        }
        keys
    }

    /// Check if a world voxel position is within this event's radius.
    #[inline]
    pub fn contains(&self, wx: i32, wy: i32, wz: i32) -> bool {
        let dx = wx - self.center.0;
        let dy = wy - self.center.1;
        let dz = wz - self.center.2;
        dx * dx + dy * dy + dz * dz <= self.radius * self.radius
    }
}

/// Check if a world position is within ANY event's radius.
#[inline]
pub fn in_any_event(events: &[StressDirtyEvent], wx: i32, wy: i32, wz: i32) -> bool {
    events.iter().any(|e| e.contains(wx, wy, wz))
}

#[inline]
pub fn pack_classification(surface: u8, source: u8) -> u8 {
    (surface << 4) | (source & 0x0F)
}

#[inline]
pub fn unpack_surface(c: u8) -> u8 { c >> 4 }

#[inline]
pub fn unpack_source(c: u8) -> u8 { c & 0x0F }
