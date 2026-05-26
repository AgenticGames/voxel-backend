//! ShotIntent — the vocabulary of "kinds of camera shots" the cinematographer
//! knows how to compose. Per-intent compose functions live in `compose/*.rs`
//! (filled in by task B11). Block 1 ships only the SafeOrbit fallback.

use voxel_world_memory::scene::SceneKind;

/// Discriminant matches the `u8` carried over FFI in `FfiShotCandidate.intent`.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShotIntent {
    /// Fallback orbital camera around the scene centroid. Always safe; the
    /// Director uses this when no better intent produces a valid shot.
    SafeOrbit = 0,
    /// Bridge: travel along the bridge axis at low altitude (the camera
    /// "walks" the bridge).
    BridgeTraveling = 1,
    /// Bridge: aerial wide shot looking down at the full bridge length.
    BridgeAerial = 2,
    /// Lava: descend into the upwelling from above.
    LavaDescent = 3,
    /// Lava: top-down look straight down into the heat source.
    LavaTopdown = 4,
    /// Water: follow the flow direction (downstream).
    WaterFlowFollow = 5,
    /// Stress: sweeping shot of the stress cascade region with chunks
    /// foregrounded.
    StressCascade = 6,
    /// Topology: reveal-up shot of a ceiling dome (camera tilts from floor
    /// to dome).
    DomeRevealUp = 7,
    /// Topology: pull-back through a chokepoint, revealing both sides.
    ChokepointPull = 8,
    /// Topology: strafe parallel to the wall niche.
    WallNicheStrafe = 9,
}

impl ShotIntent {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::SafeOrbit),
            1 => Some(Self::BridgeTraveling),
            2 => Some(Self::BridgeAerial),
            3 => Some(Self::LavaDescent),
            4 => Some(Self::LavaTopdown),
            5 => Some(Self::WaterFlowFollow),
            6 => Some(Self::StressCascade),
            7 => Some(Self::DomeRevealUp),
            8 => Some(Self::ChokepointPull),
            9 => Some(Self::WallNicheStrafe),
            _ => None,
        }
    }

    /// All known intents — used by the compose loop in lib.rs.
    pub fn all() -> [ShotIntent; 10] {
        [
            ShotIntent::SafeOrbit,
            ShotIntent::BridgeTraveling,
            ShotIntent::BridgeAerial,
            ShotIntent::LavaDescent,
            ShotIntent::LavaTopdown,
            ShotIntent::WaterFlowFollow,
            ShotIntent::StressCascade,
            ShotIntent::DomeRevealUp,
            ShotIntent::ChokepointPull,
            ShotIntent::WallNicheStrafe,
        ]
    }

    /// Map a `SceneKind` to its preferred intent set. Used by the Director
    /// when not explicitly overriding.
    pub fn for_scene_kind(kind: SceneKind) -> IntentMask {
        match kind {
            SceneKind::Lava => IntentMask::from_intents(&[
                ShotIntent::LavaDescent,
                ShotIntent::LavaTopdown,
                ShotIntent::SafeOrbit,
            ]),
            SceneKind::Water => IntentMask::from_intents(&[
                ShotIntent::WaterFlowFollow,
                ShotIntent::SafeOrbit,
            ]),
            SceneKind::Stress => IntentMask::from_intents(&[
                ShotIntent::StressCascade,
                ShotIntent::SafeOrbit,
            ]),
            SceneKind::Bridge => IntentMask::from_intents(&[
                ShotIntent::BridgeTraveling,
                ShotIntent::BridgeAerial,
                ShotIntent::SafeOrbit,
            ]),
            SceneKind::CeilingDome => IntentMask::from_intents(&[
                ShotIntent::DomeRevealUp,
                ShotIntent::SafeOrbit,
            ]),
            SceneKind::Chokepoint => IntentMask::from_intents(&[
                ShotIntent::ChokepointPull,
                ShotIntent::SafeOrbit,
            ]),
            SceneKind::WallNiche => IntentMask::from_intents(&[
                ShotIntent::WallNicheStrafe,
                ShotIntent::SafeOrbit,
            ]),
        }
    }
}

/// Bitmask of allowed intents. The Director can compose against a subset
/// (e.g. "all Lava intents but exclude SafeOrbit"). Bit N set ⇒ intent
/// with discriminant N is allowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntentMask(pub u32);

impl IntentMask {
    /// All intents allowed.
    pub fn all() -> Self {
        Self(0xFFFFFFFF)
    }

    /// Just SafeOrbit.
    pub fn safe_only() -> Self {
        Self::from_intents(&[ShotIntent::SafeOrbit])
    }

    pub fn from_intents(intents: &[ShotIntent]) -> Self {
        let mut mask = 0u32;
        for &i in intents {
            mask |= 1u32 << (i as u8 as u32);
        }
        Self(mask)
    }

    pub fn allows(&self, intent: ShotIntent) -> bool {
        (self.0 & (1u32 << (intent as u8 as u32))) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intent_mask_basic() {
        let m = IntentMask::from_intents(&[ShotIntent::SafeOrbit, ShotIntent::LavaDescent]);
        assert!(m.allows(ShotIntent::SafeOrbit));
        assert!(m.allows(ShotIntent::LavaDescent));
        assert!(!m.allows(ShotIntent::BridgeTraveling));
    }

    #[test]
    fn intent_mask_all() {
        let m = IntentMask::all();
        for i in ShotIntent::all() {
            assert!(m.allows(i));
        }
    }

    #[test]
    fn intent_mask_safe_only() {
        let m = IntentMask::safe_only();
        assert!(m.allows(ShotIntent::SafeOrbit));
        assert!(!m.allows(ShotIntent::LavaDescent));
    }

    #[test]
    fn for_scene_kind_includes_safe_orbit() {
        for kind in [
            SceneKind::Lava,
            SceneKind::Water,
            SceneKind::Stress,
            SceneKind::Bridge,
            SceneKind::CeilingDome,
            SceneKind::Chokepoint,
            SceneKind::WallNiche,
        ] {
            let m = ShotIntent::for_scene_kind(kind);
            assert!(
                m.allows(ShotIntent::SafeOrbit),
                "kind {:?} must include SafeOrbit",
                kind
            );
        }
    }

    #[test]
    fn from_u8_roundtrip() {
        for i in ShotIntent::all() {
            assert_eq!(ShotIntent::from_u8(i as u8), Some(i));
        }
        assert_eq!(ShotIntent::from_u8(99), None);
    }
}
