//! Ambient groundwater model — depth-based moisture for passive geological effects.
//!
//! Pure-function moisture query: no precomputation, no state.
//! `moisture = (strength * depth_factor * porosity * drip_multiplier).clamp(0, 1)`

use voxel_core::material::Material;

use crate::config::GroundwaterConfig;

/// Return the porosity factor for a given material (0.0 = impermeable, 1.0 = highly porous).
pub fn porosity_of(config: &GroundwaterConfig, mat: Material) -> f32 {
    match mat {
        Material::Limestone => config.porosity_limestone,
        Material::Sandstone => config.porosity_sandstone,
        Material::Slate => config.porosity_slate,
        Material::Marble => config.porosity_marble,
        Material::Granite => config.porosity_granite,
        Material::Basalt => config.porosity_basalt,
        _ => 0.0,
    }
}

/// Fracture site: 1-2 air neighbors (narrow crack, not wide cave).
/// In hard rock, groundwater only flows through fractures.
pub fn is_fracture_site(air_count: u32) -> bool {
    air_count >= 1 && air_count <= 2
}

/// Compute ambient moisture at a world position.
///
/// - `wy`: world Y coordinate of the voxel
/// - `mat`: material at that position (determines porosity)
/// - `has_air_below`: true if the voxel below is air (drip zone)
///
/// Returns 0.0 if groundwater is disabled or the material has zero porosity.
pub fn ambient_moisture(
    config: &GroundwaterConfig,
    wy: i32,
    mat: Material,
    has_air_below: bool,
) -> f32 {
    if !config.enabled {
        return 0.0;
    }

    let depth_factor = ((config.depth_baseline - wy as f32) * config.depth_scale).clamp(0.0, 1.0);
    let porosity = porosity_of(config, mat);
    let drip = if has_air_below { config.drip_zone_multiplier } else { 1.0 };

    (config.strength * depth_factor * porosity * drip).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disabled_returns_zero() {
        let config = GroundwaterConfig {
            enabled: false,
            ..Default::default()
        };
        let m = ambient_moisture(&config, -50, Material::Limestone, true);
        assert_eq!(m, 0.0);
    }

    #[test]
    fn test_basic_limestone_at_depth() {
        let config = GroundwaterConfig::default();
        // At y=-50: depth_factor = (0.0 - (-50)) * 0.02 = 1.0 (clamped)
        // porosity_limestone = 1.0, no drip, strength = 0.3
        // moisture = 0.3 * 1.0 * 1.0 * 1.0 = 0.3
        let m = ambient_moisture(&config, -50, Material::Limestone, false);
        assert!((m - 0.3).abs() < 0.001, "Expected ~0.3, got {}", m);
    }

    #[test]
    fn test_drip_zone_boost() {
        let config = GroundwaterConfig::default();
        let no_drip = ambient_moisture(&config, -50, Material::Limestone, false);
        let with_drip = ambient_moisture(&config, -50, Material::Limestone, true);
        // drip multiplier is 2.0, so with_drip should be 2x no_drip (clamped to 1.0)
        assert!((with_drip - (no_drip * 2.0).min(1.0)).abs() < 0.001);
    }

    #[test]
    fn test_granite_low_porosity() {
        let config = GroundwaterConfig::default();
        let limestone = ambient_moisture(&config, -50, Material::Limestone, false);
        let granite = ambient_moisture(&config, -50, Material::Granite, false);
        // Granite porosity = 0.2 vs Limestone = 1.0
        assert!(granite < limestone, "Granite ({}) should be less than Limestone ({})", granite, limestone);
        assert!((granite / limestone - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_above_baseline_is_zero() {
        let config = GroundwaterConfig::default();
        // Above baseline (y=10, baseline=0) → depth_factor = (0 - 10) * 0.02 = -0.2, clamped to 0
        let m = ambient_moisture(&config, 10, Material::Limestone, false);
        assert_eq!(m, 0.0);
    }

    #[test]
    fn test_non_host_rock_is_zero() {
        let config = GroundwaterConfig::default();
        let m = ambient_moisture(&config, -50, Material::Iron, false);
        assert_eq!(m, 0.0);
    }
}
