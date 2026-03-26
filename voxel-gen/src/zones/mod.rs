//! Cavern Zone system — large-scale themed underground zones.
//!
//! Zones detect large air volumes after worm carving and reshape them into
//! themed areas (Cathedral Cavern, Subterranean Lake, Lava Tube Gallery, etc.).
//! They take priority over the smaller formation system, which is banned from
//! generating within zone bounds.

pub mod detect;
pub mod shapes;
pub mod cathedral;
pub mod lake;
pub mod canyon;
pub mod lava_gallery;
pub mod bioluminescent;
pub mod terraces;
pub mod frozen;

use std::collections::HashMap;

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::config::{ZoneConfig, ZoneType};
use crate::density::DensityField;
use crate::pools::{FluidSeed, PoolFluid};
use crate::worm::path::WormSegment;

/// Convert a world-space position to a FluidSeed (chunk-local coordinates).
pub fn world_to_fluid_seed(wx: f32, wy: f32, wz: f32, effective_bounds: f32, chunk_size: usize, is_lava: bool) -> FluidSeed {
    let cx = (wx / effective_bounds).floor() as i32;
    let cy = (wy / effective_bounds).floor() as i32;
    let cz = (wz / effective_bounds).floor() as i32;
    let vs = effective_bounds / chunk_size as f32;
    let lx = ((wx - cx as f32 * effective_bounds) / vs).floor().max(0.0).min((chunk_size - 1) as f32) as u8;
    let ly = ((wy - cy as f32 * effective_bounds) / vs).floor().max(0.0).min((chunk_size - 1) as f32) as u8;
    let lz = ((wz - cz as f32 * effective_bounds) / vs).floor().max(0.0).min((chunk_size - 1) as f32) as u8;
    FluidSeed {
        chunk: (cx, cy, cz),
        lx, ly, lz,
        fluid_type: if is_lava { PoolFluid::Lava } else { PoolFluid::Water },
        is_source: true,
    }
}

/// Bounding box of a placed zone, used to exclude formations/pools.
#[derive(Debug, Clone)]
pub struct ZoneBounds {
    pub world_min: Vec3,
    pub world_max: Vec3,
    pub zone_type: ZoneType,
}

impl ZoneBounds {
    /// Check if a world-space point falls inside this zone's AABB.
    pub fn contains(&self, pos: Vec3) -> bool {
        pos.x >= self.world_min.x && pos.x <= self.world_max.x
            && pos.y >= self.world_min.y && pos.y <= self.world_max.y
            && pos.z >= self.world_min.z && pos.z <= self.world_max.z
    }
}

/// Check if a world-space point is inside any zone (except Bioluminescent, which allows formations).
pub fn is_in_exclusion_zone(pos: Vec3, zone_bounds: &[ZoneBounds]) -> bool {
    zone_bounds.iter().any(|z| {
        z.zone_type != ZoneType::BioluminescentGrotto && z.contains(pos)
    })
}

/// Anchor point for UE rendering (bioluminescent lights, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoneAnchor {
    pub px: f32, pub py: f32, pub pz: f32,
    pub nx: f32, pub ny: f32, pub nz: f32,
}

/// Descriptor for a placed zone, returned to caller for UE/viewer consumption.
#[derive(Debug, Clone)]
pub struct ZoneDescriptor {
    pub zone_type: ZoneType,
    pub world_min: Vec3,
    pub world_max: Vec3,
    pub center: Vec3,
    pub anchors: Vec<ZoneAnchor>,
}

/// Main entry point: detect and place zones across a region's density fields.
///
/// Called after worm carving + lava tubes + rivers, before pools and formations.
/// Returns placed zone descriptors, zone exclusion bounds, and any fluid seeds.
pub fn place_zones(
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    config: &ZoneConfig,
    global_seed: u64,
    effective_bounds: f32,
    worm_paths: &[Vec<WormSegment>],
) -> (Vec<ZoneDescriptor>, Vec<ZoneBounds>, Vec<FluidSeed>) {
    if !config.enabled {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let mut descriptors = Vec::new();
    let mut bounds = Vec::new();
    let mut fluid_seeds = Vec::new();

    // Phase 0: Try to place a Frozen Mega-Vault (carves its own space from solid rock)
    if let Some((desc, zone_bounds)) = frozen::try_place_mega_vault(
        density_fields, config, global_seed, effective_bounds, worm_paths,
    ) {
        descriptors.push(desc);
        bounds.push(zone_bounds);
        // The carved air will be detected as a CavernVolume below and get ice-painted
    }

    // Step 1: Compute per-chunk air statistics
    let air_stats = detect::compute_air_stats(density_fields, effective_bounds);

    // Step 2: Cluster into CavernVolumes
    let volumes = detect::cluster_cavern_volumes(&air_stats, effective_bounds, 64);

    // Step 3: Select zone types and generate
    for volume in &volumes {
        let zone_type = detect::select_zone_type(volume, config, global_seed);
        if let Some(zt) = zone_type {
            let (desc, zone_bounds, seeds) = generate_zone(
                zt, volume, density_fields, config, global_seed, effective_bounds,
            );
            descriptors.push(desc);
            bounds.push(zone_bounds);
            fluid_seeds.extend(seeds);
        }
    }

    (descriptors, bounds, fluid_seeds)
}

/// Dispatch to zone-specific generation.
fn generate_zone(
    zone_type: ZoneType,
    volume: &detect::CavernVolume,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    config: &ZoneConfig,
    global_seed: u64,
    effective_bounds: f32,
) -> (ZoneDescriptor, ZoneBounds, Vec<FluidSeed>) {
    let zone_bounds = ZoneBounds {
        world_min: volume.world_bbox_min,
        world_max: volume.world_bbox_max,
        zone_type,
    };

    let (anchors, fluid_seeds) = match zone_type {
        ZoneType::Cathedral => cathedral::generate(volume, density_fields, config, global_seed, effective_bounds),
        ZoneType::SubterraneanLake => lake::generate(volume, density_fields, config, global_seed, effective_bounds),
        ZoneType::RiverCanyon => canyon::generate(volume, density_fields, config, global_seed, effective_bounds),
        ZoneType::LavaTubeGallery => lava_gallery::generate(volume, density_fields, config, global_seed, effective_bounds),
        ZoneType::BioluminescentGrotto => bioluminescent::generate(volume, density_fields, config, global_seed, effective_bounds),
        ZoneType::GeothermalTerraces => terraces::generate(volume, density_fields, config, global_seed, effective_bounds),
        ZoneType::FrozenGrotto => frozen::generate(volume, density_fields, config, global_seed, effective_bounds),
    };

    let descriptor = ZoneDescriptor {
        zone_type,
        world_min: volume.world_bbox_min,
        world_max: volume.world_bbox_max,
        center: volume.world_center,
        anchors,
    };

    (descriptor, zone_bounds, fluid_seeds)
}
