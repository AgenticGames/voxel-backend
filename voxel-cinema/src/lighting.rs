//! LightingProfile — intent-driven lighting parameters that UE realizes
//! into actual point lights. Today's montage hard-codes 7 lights with magic
//! intensities (see VoxelSleepScanManager.cpp `PlaceShowcaseLights`); the
//! new model emits an *intent* and UE picks lights to match.

use serde::{Deserialize, Serialize};

use crate::intent::ShotIntent;

/// Where the "hero" key light should be positioned relative to the camera
/// + subject. UE realizes this into a specific point-light placement.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HeroPositionIntent {
    /// Hero light from above the subject, slightly behind.
    AboveBehind = 0,
    /// Hero light from below (uplighting; dramatic).
    Below = 1,
    /// Hero light camera-side (frontal flat).
    Frontal = 2,
    /// Hero light opposing the camera (rim/silhouette).
    BehindSubject = 3,
    /// No hero light — diffuse only.
    None = 4,
}

/// Lighting profile. UE evaluates these into a 5-7 point-light rig.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LightingProfile {
    /// Color temperature offset. -1.0 = very cool blue; +1.0 = very warm
    /// orange. 0.0 = neutral white. The existing montage uses ~0.5 (warm).
    pub warmth: f32,
    /// Contrast — 0.0 = flat fill; 1.0 = key:fill ratio of ~4:1.
    pub contrast: f32,
    /// Key-light absolute intensity (unitless; UE multiplies to lumens).
    /// Current default in VoxelSleepScanManager is `LightLumens = 4250`.
    pub key_intensity: f32,
    /// Fill ratio relative to key. 1.0 = same intensity (no contrast);
    /// 0.25 = key is 4× brighter than fill.
    pub fill_ratio: f32,
    pub hero_position_intent: HeroPositionIntent,
}

impl LightingProfile {
    /// Pick a sensible default lighting for a given intent. Block 1 just
    /// picks reasonable warmth/contrast; per-intent tuning lands in B11.
    pub fn default_for_intent(intent: ShotIntent) -> Self {
        match intent {
            ShotIntent::LavaDescent | ShotIntent::LavaTopdown => Self {
                warmth: 0.8, // hot orange
                contrast: 0.8,
                key_intensity: 5000.0,
                fill_ratio: 0.3,
                hero_position_intent: HeroPositionIntent::Below, // uplit drama
            },
            ShotIntent::WaterFlowFollow => Self {
                warmth: -0.3, // cool blue
                contrast: 0.4,
                key_intensity: 3500.0,
                fill_ratio: 0.6,
                hero_position_intent: HeroPositionIntent::Frontal,
            },
            ShotIntent::StressCascade => Self {
                warmth: 0.0,
                contrast: 0.9, // hard light, dramatic
                key_intensity: 4500.0,
                fill_ratio: 0.2,
                hero_position_intent: HeroPositionIntent::BehindSubject, // rim
            },
            ShotIntent::BridgeTraveling | ShotIntent::BridgeAerial => Self {
                warmth: 0.4,
                contrast: 0.6,
                key_intensity: 4250.0, // matches today's LightLumens
                fill_ratio: 0.5,
                hero_position_intent: HeroPositionIntent::AboveBehind,
            },
            ShotIntent::DomeRevealUp => Self {
                warmth: 0.2,
                contrast: 0.5,
                key_intensity: 4000.0,
                fill_ratio: 0.5,
                hero_position_intent: HeroPositionIntent::Below, // uplit dome
            },
            ShotIntent::ChokepointPull | ShotIntent::WallNicheStrafe => Self {
                warmth: 0.3,
                contrast: 0.7,
                key_intensity: 4000.0,
                fill_ratio: 0.4,
                hero_position_intent: HeroPositionIntent::Frontal,
            },
            ShotIntent::SafeOrbit => Self {
                warmth: 0.5, // warm, matches today's default
                contrast: 0.5,
                key_intensity: 4250.0,
                fill_ratio: 0.5,
                hero_position_intent: HeroPositionIntent::AboveBehind,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lava_lighting_is_warm_and_dramatic() {
        let p = LightingProfile::default_for_intent(ShotIntent::LavaDescent);
        assert!(p.warmth > 0.5);
        assert!(p.contrast > 0.6);
    }

    #[test]
    fn water_lighting_is_cool() {
        let p = LightingProfile::default_for_intent(ShotIntent::WaterFlowFollow);
        assert!(p.warmth < 0.0);
    }

    #[test]
    fn safe_orbit_uses_legacy_intensity() {
        let p = LightingProfile::default_for_intent(ShotIntent::SafeOrbit);
        assert!((p.key_intensity - 4250.0).abs() < 1.0);
    }
}
