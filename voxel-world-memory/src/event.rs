//! WorldEvent — push-path event ingestion. Lock-free queue drained by the
//! drift loop within ~16 ms of the event firing.
//!
//! Events are *hints*, not authoritative state. The drift loop's periodic
//! scan re-derives Scene scores from the live density/stress/fluid data,
//! so a dropped event just means the Scene refreshes one drift cycle (~2 s)
//! later than ideal. The queue is bounded; overflow drops silently.

use serde::{Deserialize, Serialize};

use crate::scene::SceneKind;

/// World event push-payload. Variants are payload-typed enums.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum WorldEvent {
    /// A brush touched the world at this voxel position. Use kind hint to
    /// help the drift loop prioritize re-scoring (PaintStress hits the
    /// stress chunks, fluid brushes hit lava/water).
    BrushApplied {
        world_pos: [f32; 3],
        /// Optional hint about which Scene kind this likely affects (None ⇒
        /// drift loop figures it out from the live data).
        kind_hint: Option<SceneKind>,
    },
    /// A crystal-anchor was placed. Position is the anchor impact site in
    /// Rust voxel coords.
    AnchorPlaced { world_pos: [f32; 3], anchor_id: u64 },
    /// A collapse fired in a region. Pair (chunks_min, chunks_max) bound
    /// the affected chunk box.
    CollapseFired {
        center_world_pos: [f32; 3],
        affected_chunks: u32,
    },
    /// Deep sleep cycle completed. The predict cache is invalidated by the
    /// caller separately; this event nudges drift loop to re-score modified
    /// chunks.
    SleepCompleted {
        dirty_chunk_count: u32,
        manifest_size_bytes: u32,
    },
    /// Lava fluid extended into a new cell. Coalesced into LavaSpread Scenes.
    LavaSpread { world_pos: [f32; 3] },
    /// Water fluid extended into a new cell. Coalesced into Water Scenes.
    WaterChanged { world_pos: [f32; 3] },
}

impl WorldEvent {
    pub fn lava_spread_at(x: f32, y: f32, z: f32) -> Self {
        Self::LavaSpread {
            world_pos: [x, y, z],
        }
    }

    pub fn water_changed_at(x: f32, y: f32, z: f32) -> Self {
        Self::WaterChanged {
            world_pos: [x, y, z],
        }
    }

    pub fn brush_applied(world_pos: [f32; 3], kind_hint: Option<SceneKind>) -> Self {
        Self::BrushApplied {
            world_pos,
            kind_hint,
        }
    }

    pub fn anchor_placed(world_pos: [f32; 3], anchor_id: u64) -> Self {
        Self::AnchorPlaced {
            world_pos,
            anchor_id,
        }
    }

    pub fn collapse_fired(center_world_pos: [f32; 3], affected_chunks: u32) -> Self {
        Self::CollapseFired {
            center_world_pos,
            affected_chunks,
        }
    }

    pub fn sleep_completed(dirty_chunk_count: u32, manifest_size_bytes: u32) -> Self {
        Self::SleepCompleted {
            dirty_chunk_count,
            manifest_size_bytes,
        }
    }

    /// Best-effort kind hint — returns the Scene kind the drift loop should
    /// prioritize when ingesting this event.
    pub fn kind_hint(&self) -> Option<SceneKind> {
        match self {
            Self::BrushApplied { kind_hint, .. } => *kind_hint,
            Self::AnchorPlaced { .. } => Some(SceneKind::Bridge),
            Self::CollapseFired { .. } => Some(SceneKind::Stress),
            Self::SleepCompleted { .. } => None,
            Self::LavaSpread { .. } => Some(SceneKind::Lava),
            Self::WaterChanged { .. } => Some(SceneKind::Water),
        }
    }

    /// Best-effort position — returns the event's spatial anchor in Rust
    /// voxel coords, if any.
    pub fn world_pos(&self) -> Option<[f32; 3]> {
        match self {
            Self::BrushApplied { world_pos, .. } => Some(*world_pos),
            Self::AnchorPlaced { world_pos, .. } => Some(*world_pos),
            Self::CollapseFired {
                center_world_pos, ..
            } => Some(*center_world_pos),
            Self::SleepCompleted { .. } => None,
            Self::LavaSpread { world_pos } => Some(*world_pos),
            Self::WaterChanged { world_pos } => Some(*world_pos),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lava_event_kind_hint() {
        let e = WorldEvent::lava_spread_at(1.0, 2.0, 3.0);
        assert_eq!(e.kind_hint(), Some(SceneKind::Lava));
        assert_eq!(e.world_pos(), Some([1.0, 2.0, 3.0]));
    }

    #[test]
    fn anchor_event_pos_and_hint() {
        let e = WorldEvent::anchor_placed([10.0, 20.0, 30.0], 42);
        assert_eq!(e.kind_hint(), Some(SceneKind::Bridge));
        assert_eq!(e.world_pos(), Some([10.0, 20.0, 30.0]));
    }

    #[test]
    fn sleep_event_has_no_pos() {
        let e = WorldEvent::sleep_completed(10, 1024);
        assert_eq!(e.kind_hint(), None);
        assert_eq!(e.world_pos(), None);
    }

    #[test]
    fn collapse_event() {
        let e = WorldEvent::collapse_fired([5.0, 5.0, 5.0], 8);
        assert_eq!(e.kind_hint(), Some(SceneKind::Stress));
        assert_eq!(e.world_pos(), Some([5.0, 5.0, 5.0]));
    }

    #[test]
    fn brush_event_uses_explicit_hint() {
        let e = WorldEvent::brush_applied([0.0, 0.0, 0.0], Some(SceneKind::Water));
        assert_eq!(e.kind_hint(), Some(SceneKind::Water));
    }
}
