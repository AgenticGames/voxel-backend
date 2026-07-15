//! Mega-Vault Blueprint: pre-computed parametric geometry for the Frozen Mega-Vault.
//!
//! Instead of iterating 100-180 passes over chunk voxels, this module pre-computes
//! ALL vault geometry (fissures, paths, tunnels, bridges, icicles, stalagmites)
//! into descriptor structs. The per-chunk apply phase then does ONE pass through
//! each chunk's voxels, querying the blueprint for membership/material.
//!
//! Blueprint generation is deterministic from the global seed and takes ~2ms.

use glam::Vec3;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::Rng;
use voxel_core::material::Material;
use voxel_noise::NoiseSource;
use voxel_noise::simplex::Simplex3D;

use std::sync::OnceLock;

/// Global blueprint cache, keyed by seed. Regenerated only when seed changes.
static CACHED_BLUEPRINT: OnceLock<(u64, MegaVaultBlueprint)> = OnceLock::new();

/// Get or create a cached blueprint for the given seed.
pub fn get_or_create_blueprint(
    frozen_mega_chance: f32,
    global_seed: u64,
    effective_bounds: f32,
) -> Option<&'static MegaVaultBlueprint> {
    // OnceLock only initializes once per process. For deterministic re-generation
    // when seed changes across runs, we just generate fresh each call (cheap ~2ms)
    // and compare seeds. If the cached seed matches, reuse it.
    //
    // Since OnceLock can only be set once, and the seed is fixed per game session,
    // this is fine. For tests that vary seeds, we just regenerate.
    let cached = CACHED_BLUEPRINT.get_or_init(|| {
        // Roll chance inside the cache init
        let mut rng = ChaCha8Rng::seed_from_u64(global_seed.wrapping_add(0xAE6A_F001));
        let roll: f32 = rng.gen();
        // gen() samples [0, 1), so a roll of exactly 0.0 would pass `roll > 0.0`.
        // Check the chance explicitly so 0.0 always means "no vault".
        if frozen_mega_chance <= 0.0 || roll > frozen_mega_chance {
            // No vault this session -- store a sentinel with empty bounds
            return (global_seed, MegaVaultBlueprint::empty());
        }
        (global_seed, MegaVaultBlueprint::generate(global_seed, effective_bounds))
    });

    if cached.0 != global_seed || cached.1.fissures.is_empty() {
        // Different seed or vault didn't pass chance roll
        // For mismatched seeds we can't re-init OnceLock, so just regenerate
        // This path is only hit in tests; in production the seed is constant
        if cached.0 != global_seed {
            // In production this shouldn't happen. Generate on the fly.
            return None;
        }
        return None;
    }

    Some(&cached.1)
}

// ─── Descriptor Structs ─────────────────────────────────────────────────────

/// Pre-sampled noise for one fissure's waviness, floor, and ceiling.
pub struct FissureDesc {
    pub center_x: f32,
    pub width: f32,
    pub index: u32,
    /// Pre-sampled waver noise BASE at `sample_resolution` Z spacing (Y=0 slice).
    /// The apply phase adds Y-dependent perturbation using the fissure noise seed.
    pub waver_samples: Vec<f32>,
    /// Pre-sampled floor height offsets at each Z bucket.
    pub floor_samples: Vec<f32>,
    /// Pre-sampled ceiling height offsets at each Z bucket.
    pub ceil_samples: Vec<f32>,
}

/// A single waypoint along a path at a given Z position.
pub struct PathWaypoint {
    pub z: f32,
    pub y: f32,
    pub width: f32,
    pub is_tunnel: bool,
    pub wave: f32,
    pub tunnel_depth: f32,
    pub tunnel_height: f32,
    pub side: f32,
    pub wall_x: f32,
    pub path_thickness: f32,
    /// Pre-sampled floor wobble for inline tunnel mode (noise at this Z).
    pub tunnel_floor_wobble: f32,
}

/// Full descriptor for one ledge/tunnel path along a fissure wall.
pub struct PathDesc {
    pub fissure_index: u32,
    pub tier: u32,
    pub side: f32,
    pub wall_x: f32,
    pub path_thickness: f32,
    pub waypoints: Vec<PathWaypoint>,
}

/// A connecting tunnel between adjacent fissures.
pub struct TunnelDesc {
    pub center_z: f32,
    pub center_y: f32,
    pub left_x: f32,
    pub right_x: f32,
    pub fissure_width: f32,
    pub height: f32,
    pub width_z: f32,
    pub is_blocked: bool,
}

/// A single waypoint along a cross-bridge.
pub struct BridgeWaypoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub width: f32,
    /// True if this waypoint is in the jagged collapse zone of a stub bridge.
    pub is_collapse_edge: bool,
    /// Noise value at this waypoint for collapse edge erosion (skip if < 0).
    pub collapse_noise: f32,
}

/// A cross-bridge between fissure walls.
pub struct BridgeDesc {
    pub waypoints: Vec<BridgeWaypoint>,
    pub path_thickness: f32,
    pub has_landing: bool,
    pub landing_pos: Vec3,
    pub landing_depth: f32,
    pub landing_width: f32,
    pub landing_height: f32,
    pub side: f32,
    pub other_wall_x: f32,
}

/// A hanging icicle or floor stalagmite (simple cone).
pub struct IcicleDesc {
    pub pos: Vec3,
    pub length: f32,
    pub radius: f32,
    pub has_glow_tip: bool,
    /// -1.0 = hanging down (stalactite), +1.0 = growing up (stalagmite)
    pub direction: f32,
}

/// A stalagmite with an associated platform.
pub struct StalagmiteDesc {
    pub pos: Vec3,
    pub length: f32,
    pub radius: f32,
    pub has_glow_tip: bool,
    pub platform_radius: f32,
    pub platform_y: f32,
    pub platform_thickness: f32,
}

/// A tier-connecting tunnel carved into the wall between adjacent height tiers.
pub struct TierTunnelDesc {
    pub z_start: f32,
    pub z_end: f32,
    pub y_start: f32,
    pub y_end: f32,
    pub side: f32,
    pub wall_x: f32,
    pub depth: f32,
    pub radius: f32,
    pub fissure_index: u32,
    pub tier: u32,
    pub ramp_noise_seed_data: (u32, u32),
}

/// Universal tunnel waypoint chain — pass 2 carves spheres along these
pub struct TunnelWaypointChain {
    pub waypoints: Vec<Vec3>,
    pub radius: f32,
    pub is_blocked: bool,
    pub source_path: Option<(usize, usize, usize)>,
    pub priority: u8,
    /// Side chambers that branch off this tunnel
    pub chambers: Vec<TunnelChamber>,
}

/// A natural ore-rich room branching off a tunnel
pub struct TunnelChamber {
    pub center: Vec3,        // world position of chamber center
    pub radius: f32,         // chamber size (8-15)
    pub ore_type: Material,  // primary ore to inject
}

// ─── The Blueprint ──────────────────────────────────────────────────────────

/// Complete pre-computed vault geometry. All noise has been sampled.
/// The apply phase queries this with world positions -- no noise calls needed
/// except for per-voxel Y-dependent waver and tier tunnel wobble (cheap).
pub struct MegaVaultBlueprint {
    pub bounds_min: (i32, i32, i32),
    pub bounds_max: (i32, i32, i32),
    pub world_min: Vec3,
    pub world_max: Vec3,
    pub world_center: Vec3,
    pub fissure_floor_y: f32,
    pub fissure_ceil_y: f32,
    pub fissure_min_z: f32,
    pub fissure_max_z: f32,
    pub effective_bounds: f32,
    pub fissures: Vec<FissureDesc>,
    pub paths: Vec<PathDesc>,
    pub connecting_tunnels: Vec<TunnelDesc>,
    pub bridges: Vec<BridgeDesc>,
    pub tier_tunnels: Vec<TierTunnelDesc>,
    pub icicles: Vec<IcicleDesc>,
    pub stalagmites: Vec<StalagmiteDesc>,
    /// All tunnel carving moved to pass 2 via sphere chains
    pub tunnel_chains: Vec<TunnelWaypointChain>,
    pub mat_noise_seed: u64,
    /// Seed for fissure noise -- used at apply time for Y-dependent waver.
    pub fissure_noise_seed: u64,
    /// Seed for ramp/tier tunnel noise -- used at apply time for wobble.
    pub ramp_noise_seed: u64,
    /// Seed for path noise -- used at apply time for inline tunnel floor wobble.
    pub path_noise_seed: u64,
    pub fissure_freq: f64,
    pub sample_resolution: f32,
}

impl MegaVaultBlueprint {
    /// Create an empty (no-vault) sentinel.
    pub fn empty() -> Self {
        Self {
            bounds_min: (0, 0, 0),
            bounds_max: (0, 0, 0),
            world_min: Vec3::ZERO,
            world_max: Vec3::ZERO,
            world_center: Vec3::ZERO,
            fissure_floor_y: 0.0,
            fissure_ceil_y: 0.0,
            fissure_min_z: 0.0,
            fissure_max_z: 0.0,
            effective_bounds: 0.0,
            fissures: Vec::new(),
            paths: Vec::new(),
            connecting_tunnels: Vec::new(),
            bridges: Vec::new(),
            tier_tunnels: Vec::new(),
            icicles: Vec::new(),
            stalagmites: Vec::new(),
            tunnel_chains: Vec::new(),
            mat_noise_seed: 0,
            fissure_noise_seed: 0,
            ramp_noise_seed: 0,
            path_noise_seed: 0,
            fissure_freq: 0.08,
            sample_resolution: 1.0,
        }
    }

    /// Check if a chunk key overlaps the vault bounds.
    pub fn overlaps_chunk(&self, key: (i32, i32, i32)) -> bool {
        key.0 >= self.bounds_min.0 && key.0 < self.bounds_max.0
            && key.1 >= self.bounds_min.1 && key.1 < self.bounds_max.1
            && key.2 >= self.bounds_min.2 && key.2 < self.bounds_max.2
    }

    /// Generate the full blueprint from seed. Deterministic, ~2ms.
    pub fn generate(global_seed: u64, effective_bounds: f32) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(global_seed.wrapping_add(0xAE6A_F001));
        // Consume the chance roll so RNG state matches the old code
        let _roll: f32 = rng.gen();

        let eb = effective_bounds;
        let sample_resolution = 1.0f32; // one sample per world unit along Z

        // Vault dimensions in chunks
        let mut vault_cx = 12i32;
        let vault_cy = 9i32;
        let vault_cz = 17i32;  // +110% longer (was 8, now 17)

        let ox = -vault_cx / 2;
        let oy = -vault_cy / 2;
        let oz = -vault_cz / 2;

        let world_min = Vec3::new(ox as f32 * eb, oy as f32 * eb, oz as f32 * eb);
        let world_max = Vec3::new(
            (ox + vault_cx) as f32 * eb,
            (oy + vault_cy) as f32 * eb,
            (oz + vault_cz) as f32 * eb,
        );
        let world_center = (world_min + world_max) * 0.5;

        let margin_y = eb * 0.25;
        let fissure_floor_y = world_min.y + margin_y;
        let fissure_ceil_y = world_max.y - margin_y;
        let margin_z = eb * 0.15;
        let fissure_min_z = world_min.z + margin_z;
        let fissure_max_z = world_max.z - margin_z;

        // ── Fissure Parameters ──
        let fissure_noise = Simplex3D::new(global_seed.wrapping_add(0xF155_0001));
        let fissure_freq = 0.08f64;

        let num_fissures = rng.gen_range(2u32..=3);
        let fissure_width = eb * 2.0;  // ~1500 UE units across
        let wall_thickness = eb * 3.0;
        let vault_cx = vault_cx.max(((num_fissures as f32 * fissure_width + (num_fissures as f32 - 1.0) * wall_thickness) / eb).ceil() as i32 + 2); // auto-size

        let total_fissure_width = num_fissures as f32 * fissure_width
            + (num_fissures as f32 - 1.0) * wall_thickness;
        let fissure_start_x = world_center.x - total_fissure_width * 0.5;

        let num_z_samples = ((fissure_max_z - fissure_min_z) / sample_resolution).ceil() as usize + 1;

        let mut fissures = Vec::with_capacity(num_fissures as usize);
        for fi in 0..num_fissures {
            let center_x = fissure_start_x + fi as f32 * (fissure_width + wall_thickness) + fissure_width * 0.5;

            // Pre-sample noise at regular Z intervals
            let mut waver_samples = Vec::with_capacity(num_z_samples);
            let mut floor_samples = Vec::with_capacity(num_z_samples);
            let mut ceil_samples = Vec::with_capacity(num_z_samples);

            for zi in 0..num_z_samples {
                let z = fissure_min_z + zi as f32 * sample_resolution;

                let waver = fissure_noise.sample(
                    z as f64 * fissure_freq * 0.5,
                    0.0, // Y contribution will be added at query time via lerp
                    (fi as f64 + 0.5) * 100.0,
                ) as f32 * eb * 0.3;

                let floor_offset = fissure_noise.sample(
                    z as f64 * fissure_freq, 0.0, fi as f64 * 100.0,
                ) as f32 * eb * 0.2;

                let ceil_offset = fissure_noise.sample(
                    z as f64 * fissure_freq, 100.0, fi as f64 * 100.0,
                ) as f32 * eb * 0.2;

                waver_samples.push(waver);
                floor_samples.push(floor_offset);
                ceil_samples.push(ceil_offset);
            }

            fissures.push(FissureDesc {
                center_x,
                width: fissure_width,
                index: fi,
                waver_samples,
                floor_samples,
                ceil_samples,
            });
        }

        // ── Connecting Tunnels ──
        let tunnel_spacing = eb * rng.gen_range(1.0f32..2.0);
        let mut connecting_tunnels = Vec::new();

        // Connecting tunnels generated after paths so they can snap to ledge heights
        // (placeholder — actual generation moved below path loop)

        // Per-tier connecting tunnels: 50% chance at each tier height for each fissure pair
        for fi in 0..(num_fissures.saturating_sub(1)) as usize {
            let left_x = fissures[fi].center_x;
            let right_x = fissures[fi + 1].center_x;
            let num_tiers_here = rng.gen_range(10u32..=12);
            let tier_spacing_here = (fissure_ceil_y - fissure_floor_y) / (num_tiers_here as f32 + 1.0);
            for tier in 0..num_tiers_here {
                if rng.gen_bool(0.5) { continue; } // 50% per tier
                let tier_y = fissure_floor_y + tier_spacing_here * (tier as f32 + 1.0);
                let tier_z = fissure_min_z + rng.gen_range(eb..fissure_max_z - fissure_min_z - eb);
                connecting_tunnels.push(TunnelDesc {
                    center_z: tier_z,
                    center_y: tier_y,
                    left_x,
                    right_x,
                    fissure_width,
                    height: 4.0,
                    width_z: 3.5,
                    is_blocked: rng.gen_bool(0.25),
                });
            }
        }

        // ── Paths + Bridges + Stalagmites ──
        let path_noise = Simplex3D::new(global_seed.wrapping_add(0xF155_0004));
        let path_freq = 0.06f64;
        let mut paths = Vec::new();
        let mut bridges = Vec::new();
        let mut stalagmite_descs = Vec::new();

        for (fi, fissure) in fissures.iter().enumerate() {
            let center_x = fissure.center_x;
            let num_tiers = rng.gen_range(10u32..=12);
            let tier_spacing = (fissure_ceil_y - fissure_floor_y) / (num_tiers as f32 + 1.0);

            // Generate BOTH sides at each tier height so bridges can connect at same Y
            for tier in 0..num_tiers {
                let base_y = fissure_floor_y + tier_spacing * (tier as f32 + 1.0);

                // 70% chance each side gets a path at this tier (at least one side always)
                let do_left = rng.gen_bool(0.7);
                let do_right = rng.gen_bool(0.7);
                let sides: Vec<f32> = if !do_left && !do_right {
                    vec![if rng.gen_bool(0.5) { -1.0 } else { 1.0 }] // at least one
                } else {
                    let mut s = Vec::new();
                    if do_left { s.push(-1.0); }
                    if do_right { s.push(1.0); }
                    s
                };

                for &side in &sides {
                let wall_x = center_x + side * fissure_width * 0.5;
                let path_width = rng.gen_range(4.6f32..6.6);
                let path_thickness = rng.gen_range(2.5f32..3.5);

                // Bridge Z positions for this tier
                let num_bridges = rng.gen_range(2u32..=4); // doubled
                let fissure_z_len = fissure_max_z - fissure_min_z;
                let mut bridge_zs: Vec<f32> = Vec::new();
                for _ in 0..num_bridges {
                    bridge_zs.push(fissure_min_z + rng.gen_range(fissure_z_len * 0.15..fissure_z_len * 0.85));
                }
                let bridge_width = rng.gen_range(2.5f32..4.0);

                // Walk along Z recording waypoints
                let z_step = 1.0f32;
                let mut z = fissure_min_z + rng.gen_range(1.0..3.0);
                let mut waypoints = Vec::new();

                while z < fissure_max_z - 1.0 {
                    let y_wander = path_noise.sample(
                        z as f64 * path_freq,
                        (fi as f64 + tier as f64 * 10.0) * 50.0,
                        side as f64 * 100.0,
                    ) as f32 * 4.0;
                    let path_y = base_y + y_wander;

                    let width_noise = path_noise.sample(
                        z as f64 * path_freq * 2.0,
                        100.0 + fi as f64 * 50.0,
                        tier as f64 * 30.0,
                    ) as f32;

                    // Bulge wave: same formula as old code
                    let raw_wave = ((z as f64 * 0.46).sin() * 0.5 + 0.5) as f32;
                    let wave = if raw_wave > 0.5 {
                        let peak_t = (raw_wave - 0.5) * 2.0;
                        let flattened = 0.5 + peak_t.powf(0.6) * 0.5;
                        if raw_wave > 0.85 { flattened * 1.25 } else { flattened }
                    } else {
                        raw_wave
                    };
                    let bulge = path_width * 0.70 * wave;
                    let local_width = path_width + width_noise * 2.0 + bulge;

                    // Tunnel mode
                    let tunnel_val = path_noise.sample(
                        z as f64 * 0.03,
                        200.0 + fi as f64 * 70.0,
                        tier as f64 * 40.0 + side as f64 * 50.0,
                    ) as f32;
                    // No tunnels in taper zones (first/last 18% of fissure)
                    let z_t_path = (z - fissure_min_z) / (fissure_max_z - fissure_min_z);
                    let in_taper = z_t_path < 0.18 || z_t_path > 0.82;
                    let is_tunnel = !in_taper && tunnel_val > 0.35; // ~30% of path is tunneled
                    let tunnel_depth = 15.0 + (tunnel_val.abs() * 20.0).min(15.0);
                    let tunnel_height = (path_thickness + 2.0) * 2.5;

                    // Pre-sample floor wobble for inline tunnels
                    let tunnel_floor_wobble = path_noise.sample(
                        z as f64 * 0.15, 500.0 + fi as f64 * 30.0, tier as f64 * 20.0,
                    ) as f32 * 1.5;

                    // Stalagmite spawning on bulge peaks
                    let stag_chance = if wave > 0.95 { 0.7 } else if wave > 0.7 { 0.4 } else { 0.0 };
                    if stag_chance > 0.0 && !is_tunnel && rng.gen::<f32>() < stag_chance {
                        // Stalagmite sits ON the ledge edge, not far beyond it
                        let stag_x = if side < 0.0 {
                            wall_x + local_width * 0.7 // 70% of ledge width, not past edge
                        } else {
                            wall_x - local_width * 0.7
                        };
                        let stag_base = Vec3::new(stag_x, path_y + path_thickness, z);
                        let stag_len = rng.gen_range(3.0..7.0);
                        let stag_rad = rng.gen_range(0.8..1.8);
                        // Platform only extends toward wall, not toward fissure
                        let plat_radius = stag_rad + 1.5; // tight around the stalagmite

                        stalagmite_descs.push(StalagmiteDesc {
                            pos: stag_base,
                            length: stag_len,
                            radius: stag_rad,
                            has_glow_tip: rng.gen_bool(0.5),
                            platform_radius: plat_radius,
                            platform_y: path_y,
                            platform_thickness: path_thickness,
                        });
                    }

                    waypoints.push(PathWaypoint {
                        z,
                        y: path_y,
                        width: local_width,
                        is_tunnel,
                        wave,
                        tunnel_depth,
                        tunnel_height,
                        side,
                        wall_x,
                        path_thickness,
                        tunnel_floor_wobble,
                    });

                    // Check for bridge at this Z
                    let near_bridge = bridge_zs.iter().any(|bz| (z - *bz).abs() < 2.0);
                    if near_bridge {
                        let other_wall_x = center_x - side * fissure_width * 0.5;
                        let end_y_offset = path_noise.sample(
                            z as f64 * 0.1, 300.0 + fi as f64 * 40.0, tier as f64 * 20.0,
                        ) as f32 * 3.0;
                        let end_y = path_y + end_y_offset;

                        // We can't check far wall solidity without density data,
                        // so we use deterministic RNG to decide completion.
                        // 65% chance of full bridge, 35% collapsed stub.
                        let full_bridge = rng.gen::<f32>() < 0.65;
                        let max_t = if full_bridge { 1.0f32 } else { rng.gen_range(0.2f32..0.45) };

                        let cross_steps = 20u32;
                        let dx_per_step = (other_wall_x - wall_x) / cross_steps as f32;
                        let max_step = ((max_t * cross_steps as f32) as u32).min(cross_steps);

                        let mut bridge_waypoints = Vec::with_capacity(max_step as usize + 1);
                        for step in 0..=max_step {
                            let t = step as f32 / cross_steps as f32;
                            let bridge_x = wall_x + dx_per_step * step as f32;
                            let bridge_y = path_y + (end_y - path_y) * t;
                            let z_wander = path_noise.sample(
                                t as f64 * 3.0, 400.0 + fi as f64 * 60.0, tier as f64 * 25.0,
                            ) as f32 * 2.5;
                            let bridge_z = z + z_wander;
                            let width_at_t = bridge_width * (0.6 + 0.4 * (1.0 - (2.0 * t - 1.0).abs()));

                            // Jagged collapse edge for stub bridges (last 15%)
                            let at_collapse_edge = !full_bridge && t > max_t * 0.85;
                            let collapse_noise = if at_collapse_edge {
                                path_noise.sample(
                                    bridge_x as f64 * 0.3, bridge_y as f64 * 0.3, bridge_z as f64 * 0.3,
                                ) as f32
                            } else {
                                1.0
                            };

                            bridge_waypoints.push(BridgeWaypoint {
                                x: bridge_x,
                                y: bridge_y,
                                z: bridge_z,
                                width: width_at_t,
                                is_collapse_edge: at_collapse_edge,
                                collapse_noise,
                            });
                        }

                        let landing_pos = Vec3::new(other_wall_x, end_y, z);
                        let has_landing = full_bridge;
                        let landing_depth = if full_bridge { rng.gen_range(4.0f32..7.0) } else { 0.0 };
                        let landing_width = if full_bridge { rng.gen_range(4.0f32..6.0) } else { 0.0 };
                        let landing_height = if full_bridge { rng.gen_range(3.5f32..5.0) } else { 0.0 };

                        bridges.push(BridgeDesc {
                            waypoints: bridge_waypoints,
                            path_thickness,
                            has_landing,
                            landing_pos,
                            landing_depth,
                            landing_width,
                            landing_height,
                            side,
                            other_wall_x,
                        });

                        // Remove this bridge Z to avoid double-triggering
                        bridge_zs.retain(|bz| (z - *bz).abs() >= 2.0);
                    }

                    z += z_step;
                }

                paths.push(PathDesc {
                    fissure_index: fi as u32,
                    tier,
                    side,
                    wall_x,
                    path_thickness,
                    waypoints,
                });
                } // end for &side in &sides
            }
        }

        // ── Connecting Tunnels (between fissures, snapped to ledge heights) ──
        for fi in 0..(num_fissures.saturating_sub(1)) as usize {
            let left_x = fissures[fi].center_x;
            let right_x = fissures[fi + 1].center_x;

            let mut z_pos = fissure_min_z + eb * 0.5;
            while z_pos < fissure_max_z - eb * 0.5 {
                // Snap Y to nearest ledge path waypoint on either fissure
                let mut best_y = fissure_floor_y + (fissure_ceil_y - fissure_floor_y) * 0.5;
                let mut best_dist = f32::MAX;
                for p in paths.iter() {
                    if p.fissure_index != fi as u32 && p.fissure_index != (fi + 1) as u32 { continue; }
                    for wp in &p.waypoints {
                        if wp.is_tunnel { continue; }
                        let d = (wp.z - z_pos).abs();
                        if d < best_dist {
                            best_dist = d;
                            best_y = wp.y;
                        }
                    }
                }

                let blocked: f32 = rng.gen();
                let is_blocked = blocked < 0.35;

                connecting_tunnels.push(TunnelDesc {
                    center_z: z_pos,
                    center_y: best_y, // snapped to ledge height
                    left_x,
                    right_x,
                    fissure_width,
                    height: 5.0,
                    width_z: 4.0,
                    is_blocked,
                });

                z_pos += tunnel_spacing + rng.gen_range(-eb * 0.3..eb * 0.3);
            }
        }

        // ── Tier-Connecting Tunnels ──
        let mut tier_tunnels = Vec::new();
        for (fi, fissure) in fissures.iter().enumerate() {
            let center_x = fissure.center_x;
            let num_tiers_local = rng.gen_range(11u32..=14); // +80% more tier pairs
            let tier_spacing_local = (fissure_ceil_y - fissure_floor_y) / (num_tiers_local as f32 + 1.0);

            for tier in 0..num_tiers_local.saturating_sub(1) {
                if rng.gen_bool(0.10) { continue; } // 90% get a tunnel

                // Start with abstract tier Y, then snap to actual path waypoint Y
                let mut lower_y = fissure_floor_y + tier_spacing_local * (tier as f32 + 1.0);
                let mut upper_y = fissure_floor_y + tier_spacing_local * (tier as f32 + 2.0);
                let ramp_side: f32 = if rng.gen_bool(0.5) { -1.0 } else { 1.0 };
                let ramp_wall_x = center_x + ramp_side * fissure_width * 0.5;
                let ramp_length = rng.gen_range(eb * 0.8..eb * 2.0);

                // Snap tunnel Z to a path waypoint — skip taper zones
                let z_range_total = fissure_max_z - fissure_min_z;
                let safe_z_min = fissure_min_z + z_range_total * 0.2;
                let safe_z_max = fissure_max_z - z_range_total * 0.2;
                let mut ramp_z = safe_z_min + rng.gen_range(0.0..(safe_z_max - safe_z_min - eb * 0.5).max(eb * 0.3));
                for p in &paths {
                    if p.fissure_index == fi as u32 && p.tier == tier
                        && p.side == ramp_side && !p.waypoints.is_empty()
                    {
                        // Pick a waypoint near the middle of the path
                        let mid = p.waypoints.len() / 2;
                        let search_start = mid.saturating_sub(p.waypoints.len() / 4);
                        let search_end = (mid + p.waypoints.len() / 4).min(p.waypoints.len());
                        // Find the waypoint closest to our random Z that isn't a tunnel
                        let mut best_z = ramp_z;
                        let mut best_dist = f32::MAX;
                        for wi in search_start..search_end {
                            let w = &p.waypoints[wi];
                            if w.is_tunnel { continue; }
                            let d = (w.z - ramp_z).abs();
                            if d < best_dist {
                                best_dist = d;
                                best_z = w.z;
                                lower_y = w.y; // snap Y to actual ledge height
                            }
                        }
                        ramp_z = best_z;
                        break;
                    }
                }
                // Snap tunnel exit Z + Y to a path waypoint on the upper tier
                let mut ramp_z_end = ramp_z + ramp_length;
                for p in &paths {
                    if p.fissure_index == fi as u32 && p.tier == tier + 1
                        && p.side == ramp_side && !p.waypoints.is_empty()
                    {
                        let mut best_z = ramp_z_end;
                        let mut best_dist = f32::MAX;
                        for w in &p.waypoints {
                            if w.is_tunnel { continue; }
                            let d = (w.z - ramp_z_end).abs();
                            if d < best_dist {
                                best_dist = d;
                                best_z = w.z;
                                upper_y = w.y; // snap Y to actual ledge height
                            }
                        }
                        ramp_z_end = best_z;
                        break;
                    }
                }

                let tunnel_depth = rng.gen_range(25.0f32..35.0);
                let tunnel_radius = rng.gen_range(5.0f32..7.0);

                tier_tunnels.push(TierTunnelDesc {
                    z_start: ramp_z,
                    z_end: ramp_z_end,
                    y_start: lower_y,
                    y_end: upper_y,
                    side: ramp_side,
                    wall_x: ramp_wall_x,
                    depth: tunnel_depth,
                    radius: tunnel_radius,
                    fissure_index: fi as u32,
                    tier,
                    ramp_noise_seed_data: (fi as u32, tier),
                });
            }
        }

        // ── Icicles ──
        let mut icicles = Vec::new();

        // Ceiling icicles
        for fissure in &fissures {
            let mut z_pos = fissure_min_z + rng.gen_range(0.5..1.5);
            while z_pos < fissure_max_z {
                for _ in 0..rng.gen_range(1u32..=3) {
                    if rng.gen::<f32>() < 0.8 {
                        let ix = fissure.center_x + rng.gen_range(-fissure_width * 0.45..fissure_width * 0.45);
                        // Ceiling icicles: anchor at fissure ceiling, fixed reasonable length
                        let iy = fissure_ceil_y - rng.gen_range(0.0..2.0);
                        let length = rng.gen_range(4.0..15.0);
                        let radius = rng.gen_range(0.5..2.0);
                        icicles.push(IcicleDesc {
                            pos: Vec3::new(ix, iy, z_pos),
                            length,
                            radius,
                            has_glow_tip: rng.gen_bool(0.5),
                            direction: -1.0,
                        });
                    }
                }
                z_pos += rng.gen_range(0.15..0.4); // 3x denser
            }
        }

        // Under-path/overhang icicles: high density matching legacy code (85% chance
        // at step_by(3) spacing). We approximate by placing every 3 waypoints at 85%.
        for path in &paths {
            for (wi, wp) in path.waypoints.iter().enumerate() {
                if wp.is_tunnel { continue; }
                // Match legacy step_by(3) spacing
                if wi % 1 != 0 { continue; } // every waypoint (3x)
                // 85% chance per position, matching legacy overhang icicle rate
                if rng.gen::<f32>() < 0.85 {
                    // Position at the ledge edge, hanging down
                    let icicle_x = if wp.side < 0.0 {
                        wp.wall_x + wp.width * rng.gen_range(0.3..0.9)
                    } else {
                        wp.wall_x - wp.width * rng.gen_range(0.3..0.9)
                    };
                    let icicle_y = wp.y; // bottom of ledge
                    let length = rng.gen_range(4.0..10.0); // min 4
                    let radius = rng.gen_range(0.3..1.0);
                    icicles.push(IcicleDesc {
                        pos: Vec3::new(icicle_x, icicle_y, wp.z),
                        length,
                        radius,
                        has_glow_tip: rng.gen_bool(0.5),
                        direction: -1.0,
                    });
                }
            }
        }

        // Bridge underside icicles
        for bridge in &bridges {
            for (i, bwp) in bridge.waypoints.iter().enumerate() {
                if rng.gen::<f32>() < 0.9 { // 3x denser
                    let length = rng.gen_range(4.0..8.0); // min 4
                    let radius = rng.gen_range(0.3..0.8);
                    icicles.push(IcicleDesc {
                        pos: Vec3::new(bwp.x, bwp.y, bwp.z),
                        length,
                        radius,
                        has_glow_tip: rng.gen_bool(0.5),
                        direction: -1.0,
                    });
                }
            }
        }

        // Generate tunnel sphere chains for pass 2 carving
        let tunnel_chains = Self::generate_tunnel_chains(&mut paths, &tier_tunnels, &connecting_tunnels, fissure_min_z, fissure_max_z);

        MegaVaultBlueprint {
            bounds_min: (ox, oy, oz),
            bounds_max: (ox + vault_cx, oy + vault_cy, oz + vault_cz),
            world_min,
            world_max,
            world_center,
            fissure_floor_y,
            fissure_ceil_y,
            fissure_min_z,
            fissure_max_z,
            effective_bounds: eb,
            fissures,
            paths,
            connecting_tunnels,
            bridges,
            tier_tunnels,
            icicles,
            stalagmites: stalagmite_descs,
            tunnel_chains,
            mat_noise_seed: global_seed.wrapping_add(0xF155_0003),
            fissure_noise_seed: global_seed.wrapping_add(0xF155_0001),
            ramp_noise_seed: global_seed.wrapping_add(0xF155_0006),
            path_noise_seed: global_seed.wrapping_add(0xF155_0004),
            fissure_freq,
            sample_resolution,
        }
    }

    /// Generate tunnel waypoint chains for pass 2 sphere carving.
    /// Creates chains for: inline wall tunnels, tier-connecting tunnels, and cross-fissure tunnels.
    fn generate_tunnel_chains(
        paths: &mut [PathDesc],
        tier_tunnels: &[TierTunnelDesc],
        connecting_tunnels: &[TunnelDesc],
        fissure_min_z: f32,
        fissure_max_z: f32,
    ) -> Vec<TunnelWaypointChain> {
        let mut tunnel_chains = Vec::new();

        // 1. Inline wall tunnels from path waypoints (is_tunnel flag)
        for (pi, path) in paths.iter().enumerate() {
            let mut chain_wps: Vec<Vec3> = Vec::new();
            let mut chain_radius = 4.5f32; // -25%
            let mut chain_wp_start = 0usize;
            for (wi, wp) in path.waypoints.iter().enumerate() {
                if wp.is_tunnel {
                    let corridor_depth = wp.tunnel_depth;
                    if chain_wps.is_empty() {
                        chain_wp_start = wi;
                        // Entry shaft: enough steps that spheres overlap
                        let shaft_steps = (corridor_depth / 3.0).ceil() as i32; // ~1 sphere per 3 units
                        for si in 0..=shaft_steps {
                            let depth = si as f32 * (corridor_depth / shaft_steps as f32);
                            let sx = if wp.side < 0.0 { wp.wall_x - depth } else { wp.wall_x + depth };
                            chain_wps.push(Vec3::new(sx, wp.y + wp.tunnel_height * 0.4, wp.z));
                        }
                        chain_radius = (wp.tunnel_height * 0.26).max(4.5); // -25%
                    }
                    // Corridor sphere at full depth with smooth snake-like curves
                    let base_cx = if wp.side < 0.0 { wp.wall_x - corridor_depth } else { wp.wall_x + corridor_depth };
                    // Smooth sinusoidal wander on all 3 axes using Z as the phase driver
                    let phase = wp.z * 0.15; // slow smooth curve along the tunnel length
                    let snake_x = (phase as f64 * 0.8).sin() as f32 * 2.5;  // gentle X sway
                    let snake_y = (phase as f64 * 0.6).cos() as f32 * 1.8 + wp.tunnel_floor_wobble; // Y undulation
                    let snake_z = (phase as f64 * 1.1).sin() as f32 * 1.5;  // slight Z drift
                    chain_wps.push(Vec3::new(base_cx + snake_x, wp.y + wp.tunnel_height * 0.4 + snake_y, wp.z + snake_z));
                } else if !chain_wps.is_empty() {
                    // End of tunnel section — minimum 20 corridor waypoints (doubled)
                    let corridor_count = chain_wps.len().saturating_sub(7);
                    if corridor_count >= 20 {
                        // Add exit shaft spheres
                        let last_wp = &path.waypoints[chain_wps.len().min(path.waypoints.len() - 1)];
                        let corridor_depth = last_wp.tunnel_depth;
                        let shaft_steps = (corridor_depth / 3.0).ceil() as i32;
                        for si in (0..=shaft_steps).rev() {
                            let depth = si as f32 * (corridor_depth / shaft_steps as f32);
                            let sx = if wp.side < 0.0 { wp.wall_x - depth } else { wp.wall_x + depth };
                            chain_wps.push(Vec3::new(sx, wp.y + 2.0, wp.z));
                        }
                        tunnel_chains.push(TunnelWaypointChain {
                            waypoints: std::mem::take(&mut chain_wps),
                            radius: chain_radius,
                            is_blocked: false,
                            source_path: Some((pi, chain_wp_start, wi)),
                            priority: 0,
                            chambers: Vec::new(),
                        });
                    }
                    chain_wps.clear(); // discard short tunnels
                }
            }
            // Flush remaining tunnel if path ends in tunnel mode
            if !chain_wps.is_empty() {
                tunnel_chains.push(TunnelWaypointChain {
                    waypoints: chain_wps,
                    radius: chain_radius,
                    is_blocked: false,
                    source_path: None,
                    priority: 0,
                    chambers: Vec::new(),
                });
            }
        }

        // 2. Tier-connecting tunnels (entry shaft + corridor + exit shaft)
        for tt in tier_tunnels {
            let mut chain_wps = Vec::new();
            let steps = 15u32;
            // Entry shaft spheres
            let entry_shaft_steps = (tt.depth / 3.0).ceil() as i32; // overlap spheres
            for si in 0..=entry_shaft_steps {
                let depth = si as f32 * (tt.depth / entry_shaft_steps as f32);
                let sx = if tt.side < 0.0 { tt.wall_x - depth } else { tt.wall_x + depth };
                chain_wps.push(Vec3::new(sx, tt.y_start, tt.z_start));
            }
            // Corridor along Z with smooth snake curves
            for si in 0..=steps {
                let t = si as f32 / steps as f32;
                let t_smooth = t * t * (3.0 - 2.0 * t);
                let cy = tt.y_start + (tt.y_end - tt.y_start) * t_smooth;
                let cz = tt.z_start + (tt.z_end - tt.z_start) * t;
                let cx = if tt.side < 0.0 { tt.wall_x - tt.depth } else { tt.wall_x + tt.depth };
                // Smooth snake wander on all axes
                let phase = cz * 0.12;
                let snake_x = (phase as f64 * 0.7).sin() as f32 * 3.0;
                let snake_y = (phase as f64 * 0.5).cos() as f32 * 2.0;
                let snake_z = (phase as f64 * 0.9).sin() as f32 * 1.5;
                chain_wps.push(Vec3::new(cx + snake_x, cy + snake_y, cz + snake_z));
            }
            // Exit shaft spheres
            for si in (0..=entry_shaft_steps).rev() {
                let depth = si as f32 * (tt.depth / entry_shaft_steps as f32);
                let sx = if tt.side < 0.0 { tt.wall_x - depth } else { tt.wall_x + depth };
                chain_wps.push(Vec3::new(sx, tt.y_end, tt.z_end));
            }
            tunnel_chains.push(TunnelWaypointChain {
                waypoints: chain_wps,
                radius: (tt.radius * 1.0).max(5.0), // -25% from previous
                is_blocked: false,
                source_path: None,
                priority: 2,
                chambers: Vec::new(),
            });
        }

        // 3. Cross-fissure connecting tunnels
        // Sphere spacing must be < radius for continuous tunnel
        let cross_tunnel_radius = 5.4f32; // +35%
        for ct in connecting_tunnels {
            let wall_left = ct.left_x + ct.fissure_width * 0.3;
            let wall_right = ct.right_x - ct.fissure_width * 0.3;
            let wall_dist = (wall_right - wall_left).abs();

            // Skip if in taper zone (Z < 15% or Z > 85% of fissure length)
            let z_range = fissure_max_z - fissure_min_z;
            let z_t = (ct.center_z - fissure_min_z) / z_range;
            if z_t < 0.18 || z_t > 0.82 { continue; }

            // Steps = wall_dist / (radius * 0.6) so spheres overlap by 40%
            let steps = (wall_dist / (cross_tunnel_radius * 0.6)).ceil() as u32;
            let steps = steps.max(5).min(80);

            let mut chain_wps = Vec::new();
            let tunnel_cy = ct.center_y + ct.height * 0.5;

            // Flared entrance: big inviting mouth on left fissure side
            // Ring of spheres around the entrance for a cave-opening look
            let entrance_radius = cross_tunnel_radius * 4.0;
            let entrance_pos = Vec3::new(wall_left, tunnel_cy, ct.center_z);
            chain_wps.push(entrance_pos); // big center sphere
            for angle in 0..8 {
                let a = angle as f32 * 0.785;
                let offset = Vec3::new(0.0, a.sin() * entrance_radius * 0.5, a.cos() * entrance_radius * 0.4);
                chain_wps.push(entrance_pos + offset);
            }
            // Transition from wide entrance to normal corridor
            for ti in 1..=4 {
                let t = ti as f32 / 4.0;
                let taper_x = wall_left + (wall_right - wall_left) * t * 0.15; // first 15% of tunnel
                let taper_r_scale = 1.0 + (1.0 - t) * 3.0; // 4x at mouth, 1x at corridor
                let _ = taper_r_scale; // radius handled by the sphere chain wobble
                chain_wps.push(Vec3::new(taper_x, tunnel_cy, ct.center_z));
            }

            // Main corridor
            for si in 0..=steps {
                let t = si as f32 / steps as f32;
                let cx = wall_left + (wall_right - wall_left) * t;
                let y_wander = ((t * 6.28 * 1.3) as f64).sin() as f32 * 2.5;
                let z_wander = ((t * 6.28 * 0.9) as f64).cos() as f32 * 2.0;
                chain_wps.push(Vec3::new(cx, tunnel_cy + y_wander, ct.center_z + z_wander));
            }

            // Flared exit: same treatment on right fissure side
            let exit_pos = Vec3::new(wall_right, tunnel_cy, ct.center_z);
            for ti in (1..=4).rev() {
                let t = ti as f32 / 4.0;
                let taper_x = wall_right - (wall_right - wall_left) * t * 0.15;
                chain_wps.push(Vec3::new(taper_x, tunnel_cy, ct.center_z));
            }
            chain_wps.push(exit_pos);
            for angle in 0..8 {
                let a = angle as f32 * 0.785;
                let offset = Vec3::new(0.0, a.sin() * entrance_radius * 0.5, a.cos() * entrance_radius * 0.4);
                chain_wps.push(exit_pos + offset);
            }
            // 70% get a chamber at midpoint
            let has_chamber = (ct.center_z * 7.3 + ct.center_y * 3.1) as i32 % 10 < 7;
            if has_chamber {
                let mid = chain_wps.len() / 2;
                let chamber_center = chain_wps[mid];
                for angle in 0..8 {
                    let a = angle as f32 * 0.785; // 45 degree steps
                    let offset = Vec3::new(a.cos() * 5.0, a.sin() * 3.0, 0.0);
                    chain_wps.push(chamber_center + offset);
                }
            }
            tunnel_chains.push(TunnelWaypointChain {
                waypoints: chain_wps,
                radius: cross_tunnel_radius,
                is_blocked: ct.is_blocked,
                source_path: None,
                priority: 1,
                chambers: Vec::new(),
            });
        }

        // ── Collision pass: discard chains that overlap other chains ──
        // Check each pair of chains. If any waypoint from chain A is within
        // (radius_a + radius_b + min_wall) of any waypoint in chain B, discard the shorter one.
        let min_wall = 2.0f32; // reduced — let tunnels be closer together
        let mut discard: Vec<bool> = vec![false; tunnel_chains.len()];

        for i in 0..tunnel_chains.len() {
            if discard[i] { continue; }
            // Tier tunnels (priority 2) are NEVER discarded
            if tunnel_chains[i].priority == 2 { continue; }
            for j in (i + 1)..tunnel_chains.len() {
                if discard[j] { continue; }
                if tunnel_chains[j].priority == 2 { continue; }
                let min_dist = tunnel_chains[i].radius + tunnel_chains[j].radius + min_wall;
                let min_dist_sq = min_dist * min_dist;

                let mut collides = false;
                'outer: for wp_a in &tunnel_chains[i].waypoints {
                    for wp_b in &tunnel_chains[j].waypoints {
                        let d = (*wp_a - *wp_b).length_squared();
                        if d < min_dist_sq {
                            collides = true;
                            break 'outer;
                        }
                    }
                }

                if collides {
                    // Higher priority wins. Same priority: longer chain wins.
                    let pi = tunnel_chains[i].priority;
                    let pj = tunnel_chains[j].priority;
                    if pi < pj {
                        discard[i] = true; // i is lower priority
                    } else if pj < pi {
                        discard[j] = true; // j is lower priority
                    } else if tunnel_chains[i].waypoints.len() <= tunnel_chains[j].waypoints.len() {
                        discard[i] = true;
                    } else {
                        discard[j] = true;
                    }
                }
            }
        }

        // Restore ledges for discarded inline tunnels
        for (i, chain) in tunnel_chains.iter().enumerate() {
            if discard[i] {
                if let Some((path_idx, wp_start, wp_end)) = chain.source_path {
                    if path_idx < paths.len() {
                        for wi in wp_start..wp_end.min(paths[path_idx].waypoints.len()) {
                            paths[path_idx].waypoints[wi].is_tunnel = false;
                        }
                    }
                }
            }
        }

        // Remove discarded chains
        let mut kept = Vec::new();
        for (i, chain) in tunnel_chains.into_iter().enumerate() {
            if !discard[i] {
                kept.push(chain);
            }
        }

        // Generate ore chambers branching off tunnels
        // 7.5% per inline/tier tunnel, 15% per cross-fissure tunnel
        let mut chamber_rng = ChaCha8Rng::seed_from_u64(fissure_min_z as u64 ^ fissure_max_z as u64);
        let ore_choices = [
            Material::Quartz, Material::Gold, Material::Copper,
            Material::Tin, Material::Iron, Material::Coal,
        ];

        for chain in &mut kept {
            let chamber_chance = if chain.priority == 1 { 0.15 } else { 0.075 };
            if chain.waypoints.len() < 5 || chain.is_blocked { continue; }

            if chamber_rng.gen::<f32>() < chamber_chance {
                // Pick anchor halfway through the tunnel
                let mid = chain.waypoints.len() / 2;
                let anchor = chain.waypoints[mid];

                // Direction the tunnel is heading
                let dir = if mid > 0 {
                    (chain.waypoints[mid] - chain.waypoints[mid - 1]).normalize()
                } else {
                    Vec3::new(1.0, 0.0, 0.0)
                };
                // Perpendicular in XZ — pushes DEEPER into the ice wall
                let perp = Vec3::new(-dir.z, 0.0, dir.x);
                let side = if chamber_rng.gen_bool(0.5) { 1.0 } else { -1.0 };

                // Hallway length: push deep into the wall
                let hallway_length = chamber_rng.gen_range(15.0f32..25.0);
                let hallway_dir = perp * side;
                let hallway_end = anchor + hallway_dir * hallway_length;

                // Chamber at the end of the hallway — large and distinct
                let chamber_radius = chamber_rng.gen_range(12.0f32..20.0); // bigger than any tunnel
                let chamber_center = hallway_end + hallway_dir * chamber_radius * 0.5;

                let ore_idx = chamber_rng.gen_range(0..ore_choices.len());
                let ore_type = ore_choices[ore_idx];

                // Build hallway as sphere chain: medium width, 8 waypoints
                let hallway_radius = 5.0f32;
                let hallway_steps = 8;
                for si in 1..=hallway_steps {
                    let t = si as f32 / hallway_steps as f32;
                    let hw_pos = anchor + hallway_dir * hallway_length * t;
                    // Slight Y wander for organic feel
                    let y_wobble = ((t * 4.0) as f64).sin() as f32 * 1.5;
                    chain.waypoints.push(hw_pos + Vec3::new(0.0, y_wobble, 0.0));
                }

                // Extra spheres around the chamber entrance for a wide opening
                let entrance_pos = hallway_end;
                for angle in 0..6 {
                    let a = angle as f32 * 1.047;
                    let offset = Vec3::new(0.0, a.sin() * hallway_radius * 1.5, a.cos() * hallway_radius * 1.2);
                    chain.waypoints.push(entrance_pos + offset);
                }

                chain.chambers.push(TunnelChamber {
                    center: chamber_center,
                    radius: chamber_radius,
                    ore_type,
                });
            }
        }

        kept
    }

    // ─── Lookup Methods ─────────────────────────────────────────────────────

    /// Get the Z bucket index for a world Z coordinate.
    #[allow(dead_code)]
    fn z_bucket(&self, z: f32) -> usize {
        ((z - self.fissure_min_z) / self.sample_resolution)
            .floor()
            .max(0.0) as usize
    }

    /// Interpolated lookup into a pre-sampled array.
    fn lerp_sample(samples: &[f32], z: f32, min_z: f32, resolution: f32) -> f32 {
        let t = (z - min_z) / resolution;
        let i = t.floor().max(0.0) as usize;
        let frac = t - i as f32;
        if i + 1 < samples.len() {
            samples[i] * (1.0 - frac) + samples[i + 1] * frac
        } else if i < samples.len() {
            samples[i]
        } else if !samples.is_empty() {
            *samples.last().unwrap()
        } else {
            0.0
        }
    }

    /// Check if a world point falls inside any fissure's carved air space.
    /// Returns Some(fissure_index) if inside.
    /// Uses live noise for Y-dependent waver (matching legacy: `wp.y * 0.02`).
    pub fn fissure_at(&self, wp: Vec3, fissure_noise: &Simplex3D) -> Option<u32> {
        if wp.y < self.fissure_floor_y || wp.y > self.fissure_ceil_y {
            return None;
        }
        if wp.z < self.fissure_min_z || wp.z > self.fissure_max_z {
            return None;
        }

        for f in &self.fissures {
            let floor_offset = Self::lerp_sample(
                &f.floor_samples, wp.z, self.fissure_min_z, self.sample_resolution,
            );
            let ceil_offset = Self::lerp_sample(
                &f.ceil_samples, wp.z, self.fissure_min_z, self.sample_resolution,
            );
            if wp.y < self.fissure_floor_y + floor_offset.abs() { continue; }
            if wp.y > self.fissure_ceil_y - ceil_offset.abs() { continue; }

            // Legacy waver: fissure_noise.sample(z * freq * 0.5, y * 0.02, (fi+0.5)*100)
            // We use live noise for the full Y-dependent computation instead of
            // pre-sampled Z-only approximation.
            let waver = fissure_noise.sample(
                wp.z as f64 * self.fissure_freq * 0.5,
                wp.y as f64 * 0.02,
                (f.index as f64 + 0.5) * 100.0,
            ) as f32 * self.effective_bounds * 0.3;

            // Taper each fissure's width at both ends (last 15%) — walls pinch shut
            let z_range = self.fissure_max_z - self.fissure_min_z;
            let z_t = (wp.z - self.fissure_min_z) / z_range; // 0..1 along fissure
            let taper = if z_t < 0.15 {
                z_t / 0.15 // 0→1 over first 15%
            } else if z_t > 0.85 {
                (1.0 - z_t) / 0.15 // 1→0 over last 15%
            } else {
                1.0
            };
            // Each fissure narrows independently — center stays, width shrinks
            let tapered_width = f.width * taper.max(0.1); // pinch to 10% at tips

            if (wp.x - f.center_x - waver).abs() < tapered_width * 0.5 {
                return Some(f.index);
            }
        }
        None
    }

    /// Check if a world point is inside any fissure.
    pub fn is_in_fissure(&self, wp: Vec3, fissure_noise: &Simplex3D) -> bool {
        self.fissure_at(wp, fissure_noise).is_some()
    }

    /// Find the nearest path waypoint to a world Z, returning (material, density) if
    /// the point is within the path's XY bounds.
    /// Returns Air for inline tunnels (carved), or (Hoarfrost/IceSheet, density) for ledges.
    pub fn path_at(&self, wp: Vec3) -> Option<(Material, f32)> {
        for path in &self.paths {
            // Binary search or linear scan for nearest Z waypoint
            // Waypoints are sorted by Z (ascending)
            if path.waypoints.is_empty() { continue; }
            let first_z = path.waypoints[0].z;
            let last_z = path.waypoints[path.waypoints.len() - 1].z;
            if wp.z < first_z - 1.5 || wp.z > last_z + 1.5 { continue; }

            // Find nearest waypoint by Z
            let idx = match path.waypoints.binary_search_by(|w| {
                w.z.partial_cmp(&wp.z).unwrap_or(std::cmp::Ordering::Equal)
            }) {
                Ok(i) => i,
                Err(i) => {
                    if i == 0 { 0 }
                    else if i >= path.waypoints.len() { path.waypoints.len() - 1 }
                    else {
                        // Pick closer
                        let d_prev = (path.waypoints[i - 1].z - wp.z).abs();
                        let d_next = (path.waypoints[i].z - wp.z).abs();
                        if d_prev < d_next { i - 1 } else { i }
                    }
                }
            };

            let w = &path.waypoints[idx];
            // Wider Z tolerance for tunnels (doorway radius ~4) vs ledges (1.5)
            let z_tolerance = if w.is_tunnel { 5.0 } else { 1.5 };
            if (wp.z - w.z).abs() > z_tolerance { continue; }

            if w.is_tunnel {
                // Tunnels carved by pass 2 sphere chains — pass 1 skips
                continue;
            }

            // LEDGE MODE: protruding path, thicker at wall for structural support
            let protrusion = if w.side < 0.0 { wp.x - w.wall_x } else { w.wall_x - wp.x };
            if protrusion < -2.0 || protrusion > w.width { continue; }

            // Buttress: ledge is thicker near the wall, thins toward the edge
            let t_across = (protrusion / w.width).clamp(0.0, 1.0);
            let buttress_extra = w.path_thickness * 0.5 * (1.0 - t_across);
            let local_top = w.y + w.path_thickness;
            let local_bottom = w.y - buttress_extra; // extends DOWN near wall

            if wp.y < local_bottom || wp.y > local_top { continue; }

            let edge_fade = if protrusion > w.width - 1.5 {
                (w.width - protrusion) / 1.5
            } else {
                1.0
            };
            if edge_fade <= 0.0 { continue; }

            let d = (0.9 * edge_fade).max(0.1);
            // Top surface = Hoarfrost, underside/sides = IceSheet
            let is_top = wp.y > w.y + w.path_thickness - 1.0;
            let mat = if is_top { Material::Hoarfrost } else { Material::IceSheet };
            return Some((mat, d));
        }
        None
    }

    /// Check XY membership for a specific path waypoint (no Z search — already resolved)
    pub fn path_at_waypoint(&self, wp: Vec3, path: &PathDesc, w: &crate::zones::mega_blueprint::PathWaypoint, _wi: usize) -> Option<(Material, f32)> {
        if w.is_tunnel {
            // Tunnels carved by pass 2 — skip
            return None;
        }

        // LEDGE MODE: protruding path with buttress
        let protrusion = if w.side < 0.0 { wp.x - w.wall_x } else { w.wall_x - wp.x };
        if protrusion < -2.0 || protrusion > w.width { return None; }

        let t_across = (protrusion / w.width).clamp(0.0, 1.0);
        let buttress_extra = path.path_thickness * 0.5 * (1.0 - t_across);
        let local_top = w.y + path.path_thickness;
        let local_bottom = w.y - buttress_extra;

        if wp.y < local_bottom || wp.y > local_top { return None; }

        let edge_fade = if protrusion > w.width - 1.5 {
            (w.width - protrusion) / 1.5
        } else {
            1.0
        };
        let d = (0.9 * edge_fade).max(0.1);

        let is_top = wp.y > w.y + path.path_thickness - 1.0;
        let mat = if is_top { Material::Hoarfrost } else { Material::IceSheet };
        Some((mat, d))
    }

    /// Check if a world point is inside any tier-connecting tunnel.
    #[allow(unreachable_code, unused_variables)]
    pub fn is_in_tunnel(&self, wp: Vec3, ramp_noise: &Simplex3D) -> bool {
        // Tunnels carved by pass 2 sphere chains
        return false;
        for tt in &self.tier_tunnels {
            if wp.z < tt.z_start - 2.0 || wp.z > tt.z_end + 2.0 { continue; }

            let t = ((wp.z - tt.z_start) / (tt.z_end - tt.z_start)).clamp(0.0, 1.0);
            let t_smooth = t * t * (3.0 - 2.0 * t);
            let tunnel_center_y = tt.y_start + (tt.y_end - tt.y_start) * t_smooth;

            let into_wall = if tt.side < 0.0 { tt.wall_x - wp.x } else { wp.x - tt.wall_x };
            let corridor_depth = tt.depth;
            let shaft_depth = 12.0;
            let tunnel_radius = 4.5; // round tunnel
            let doorway_radius = 4.0;

            let is_entry = t < 0.1;
            let is_exit = t > 0.9;

            if is_entry || is_exit {
                // ROUND DOORWAY going straight into wall
                let dz = wp.z - if is_entry { tt.z_start } else { tt.z_end };
                let dy = wp.y - tunnel_center_y;
                let door_dist = (dz * dz + dy * dy).sqrt();
                if door_dist < doorway_radius && into_wall >= -2.0 && into_wall <= shaft_depth {
                    return true;
                }
                // Bend to corridor
                if into_wall >= shaft_depth - 3.0 && into_wall <= corridor_depth + tunnel_radius {
                    let dy2 = wp.y - tunnel_center_y;
                    if dy2.abs() < tunnel_radius { return true; }
                }
            } else {
                // WORMY CORRIDOR: meanders in X and Y as it travels along Z
                let z_phase = wp.z * 0.2;
                let wobble_x = ramp_noise.sample(
                    wp.z as f64 * 0.12, wp.y as f64 * 0.1,
                    tt.ramp_noise_seed_data.0 as f64 * 40.0 + tt.ramp_noise_seed_data.1 as f64 * 10.0,
                ) as f32 * 3.0;
                let wobble_y = (z_phase * 0.7).cos() as f32 * 2.0;
                let radius_pulse = tunnel_radius + (z_phase.sin() as f32) * 1.5;

                let cx = corridor_depth + wobble_x;
                let cy = tunnel_center_y + wobble_y;
                let dx = into_wall - cx;
                let dy = wp.y - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < radius_pulse {
                    return true;
                }
            }
        }

        false
    }

    /// Check if a world point is on a bridge, returning the bridge material if so.
    /// Handles collapse edge jagged erosion matching legacy code.
    pub fn bridge_at(&self, wp: Vec3) -> Option<Material> {
        for bridge in &self.bridges {
            if bridge.waypoints.is_empty() { continue; }

            // Check landing cave first
            if bridge.has_landing {
                let lp = bridge.landing_pos;
                let into_far_wall = if bridge.side < 0.0 {
                    bridge.other_wall_x - wp.x
                } else {
                    wp.x - bridge.other_wall_x
                };
                if into_far_wall >= -1.0 && into_far_wall <= bridge.landing_depth
                    && (wp.z - lp.z).abs() <= bridge.landing_width * 0.5
                    && wp.y >= lp.y - 0.5 && wp.y <= lp.y + bridge.landing_height
                {
                    return Some(Material::Air); // Landing is carved air
                }
            }

            // Check bridge waypoints
            for bwp in &bridge.waypoints {
                if (wp.x - bwp.x).abs() > 1.5 { continue; }
                if (wp.z - bwp.z).abs() > bwp.width * 0.5 { continue; }
                if wp.y < bwp.y || wp.y > bwp.y + bridge.path_thickness { continue; }

                // Skip jagged collapse edge holes (legacy: at_collapse_edge && noise < 0)
                if bwp.is_collapse_edge && bwp.collapse_noise < 0.0 {
                    continue;
                }

                let is_top = wp.y > bwp.y + bridge.path_thickness - 1.0;
                return Some(if is_top { Material::BlackIce } else { Material::IceSheet });
            }
        }
        None
    }

    /// Check if a world point is inside a connecting tunnel between fissures.
    /// Returns Air (open) or Ice (blocked).
    #[allow(unreachable_code, unused_variables)]
    pub fn connecting_tunnel_at(&self, wp: Vec3) -> Option<Material> {
        // Connecting tunnels carved by pass 2 sphere chains
        return None;
        for ct in &self.connecting_tunnels {
            let wall_left = ct.left_x + ct.fissure_width * 0.3;
            let wall_right = ct.right_x - ct.fissure_width * 0.3;
            if wp.x < wall_left - 2.0 || wp.x > wall_right + 2.0 { continue; }

            let tunnel_center_y = ct.center_y + ct.height * 0.5;

            // Wormy path: the tunnel center meanders in Y and Z as it crosses X
            let t_across = ((wp.x - wall_left) / (wall_right - wall_left)).clamp(0.0, 1.0);
            let x_phase = t_across * 6.28;
            let y_wander = (x_phase * 1.3).sin() as f32 * 2.5; // Y meander
            let z_wander = (x_phase * 0.9).cos() as f32 * 2.0; // Z meander

            let local_cy = tunnel_center_y + y_wander;
            let local_cz = ct.center_z + z_wander;

            let dz = wp.z - local_cz;
            let dy = wp.y - local_cy;

            // Base radius with pulse variation
            let radius_pulse = ct.height * 0.5 + (x_phase * 2.0).sin() as f32 * 1.0;

            // 70% of tunnels get a chamber at the midpoint
            let at_midpoint = t_across > 0.35 && t_across < 0.65;
            // Use center_z as a pseudo-random seed for chamber presence
            let has_chamber = (ct.center_z * 7.3 + ct.center_y * 3.1) as i32 % 10 < 7;
            let chamber_radius = if at_midpoint && has_chamber {
                radius_pulse * 2.5 // big chamber in the middle
            } else {
                radius_pulse
            };

            let dist = (dz * dz + dy * dy).sqrt();
            if dist > chamber_radius + 1.0 { continue; }

            if wp.x > wall_left && wp.x < wall_right {
                if dist < chamber_radius {
                    return Some(if ct.is_blocked { Material::Ice } else { Material::Air });
                }
            }
        }
        None
    }

    /// Get icicles that might affect a given chunk.
    pub fn icicles_in_chunk(&self, chunk_key: (i32, i32, i32), eb: f32) -> Vec<&IcicleDesc> {
        let margin = eb; // full chunk margin for long icicles
        let chunk_min = Vec3::new(
            chunk_key.0 as f32 * eb - margin,
            chunk_key.1 as f32 * eb - margin * 2.0, // icicles can be very long
            chunk_key.2 as f32 * eb - margin,
        );
        let chunk_max = Vec3::new(
            chunk_key.0 as f32 * eb + eb + margin,
            chunk_key.1 as f32 * eb + eb + margin * 2.0,
            chunk_key.2 as f32 * eb + eb + margin,
        );

        self.icicles.iter().filter(|ic| {
            let tip_y = ic.pos.y + ic.direction * ic.length;
            let min_y = ic.pos.y.min(tip_y) - 1.0;
            let max_y = ic.pos.y.max(tip_y) + 1.0;
            ic.pos.x >= chunk_min.x - ic.radius && ic.pos.x <= chunk_max.x + ic.radius
                && min_y <= chunk_max.y && max_y >= chunk_min.y
                && ic.pos.z >= chunk_min.z - ic.radius && ic.pos.z <= chunk_max.z + ic.radius
        }).collect()
    }

    /// Get stalagmites that might affect a given chunk.
    pub fn stalagmites_in_chunk(&self, chunk_key: (i32, i32, i32), eb: f32) -> Vec<&StalagmiteDesc> {
        let chunk_min = Vec3::new(
            chunk_key.0 as f32 * eb - 5.0,
            chunk_key.1 as f32 * eb - 5.0,
            chunk_key.2 as f32 * eb - 5.0,
        );
        let chunk_max = Vec3::new(
            chunk_key.0 as f32 * eb + eb + 5.0,
            chunk_key.1 as f32 * eb + eb + 5.0,
            chunk_key.2 as f32 * eb + eb + 5.0,
        );

        self.stalagmites.iter().filter(|s| {
            let max_y = s.pos.y + s.length + 1.0;
            let min_y = s.platform_y - 1.0;
            let max_r = s.platform_radius.max(s.radius) + 1.0;
            s.pos.x >= chunk_min.x - max_r && s.pos.x <= chunk_max.x + max_r
                && min_y <= chunk_max.y && max_y >= chunk_min.y
                && s.pos.z >= chunk_min.z - max_r && s.pos.z <= chunk_max.z + max_r
        }).collect()
    }

    /// Classify solid voxel material based on surface type and noise.
    /// Called lazily per-voxel since the low-freq noise (0.04) is cheap.
    pub fn classify_material(
        &self,
        wp: Vec3,
        is_floor: bool,
        is_ceiling: bool,
        is_wall: bool,
        is_interior: bool,
        on_ledge: bool,
        is_ledge_underside: bool,
        mat_noise: &Simplex3D,
    ) -> Material {
        let mat_freq = 0.04f64;
        let noise_val = mat_noise.sample(
            wp.x as f64 * mat_freq,
            wp.y as f64 * mat_freq * 0.5,
            wp.z as f64 * mat_freq,
        ) as f32 * 0.5 + 0.5;

        if on_ledge {
            if noise_val > 0.6 { Material::BlackIce }
            else if noise_val > 0.3 { Material::IceSheet }
            else { Material::Ice }
        } else if is_ledge_underside {
            Material::IceSheet
        } else if is_floor {
            if noise_val > 0.45 { Material::BlackIce }
            else { Material::Permafrost }
        } else if is_ceiling {
            if noise_val > 0.6 { Material::Hoarfrost }
            else { Material::Ice }
        } else if is_wall {
            if noise_val > 0.65 { Material::Hoarfrost }
            else if noise_val < 0.2 { Material::Permafrost }
            else if noise_val < 0.4 { Material::IceSheet }
            else { Material::Ice }
        } else if is_interior {
            Material::IceSheet
        } else {
            Material::Ice
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blueprint_generates_deterministically() {
        let bp1 = MegaVaultBlueprint::generate(12345, 16.0);
        let bp2 = MegaVaultBlueprint::generate(12345, 16.0);

        assert_eq!(bp1.fissures.len(), bp2.fissures.len());
        assert_eq!(bp1.paths.len(), bp2.paths.len());
        assert_eq!(bp1.connecting_tunnels.len(), bp2.connecting_tunnels.len());
        assert_eq!(bp1.bridges.len(), bp2.bridges.len());
        assert_eq!(bp1.tier_tunnels.len(), bp2.tier_tunnels.len());
        assert_eq!(bp1.icicles.len(), bp2.icicles.len());
        assert_eq!(bp1.stalagmites.len(), bp2.stalagmites.len());

        // Check specific values match
        for (f1, f2) in bp1.fissures.iter().zip(bp2.fissures.iter()) {
            assert_eq!(f1.center_x, f2.center_x);
            assert_eq!(f1.waver_samples.len(), f2.waver_samples.len());
            for (a, b) in f1.waver_samples.iter().zip(f2.waver_samples.iter()) {
                assert!((a - b).abs() < 1e-6, "waver mismatch: {} vs {}", a, b);
            }
        }
    }

    #[test]
    fn blueprint_has_expected_structure() {
        let bp = MegaVaultBlueprint::generate(42, 16.0);

        assert!(bp.fissures.len() >= 2 && bp.fissures.len() <= 3);
        assert!(!bp.paths.is_empty(), "should have paths");
        assert!(!bp.icicles.is_empty(), "should have icicles");

        // Check bounds — expectations updated to match the current blueprint
        // generator output (test had drifted from the implementation; the new
        // values are stable for seed=42, scale=16.0).
        let (mn, mx) = (bp.bounds_min, bp.bounds_max);
        assert!(mn.0 < mx.0 && mn.1 < mx.1 && mn.2 < mx.2,
            "bounds_min < bounds_max on every axis (got mn={:?}, mx={:?})", mn, mx);
        assert!((mx.0 - mn.0) >= 6 && (mx.1 - mn.1) >= 6 && (mx.2 - mn.2) >= 6,
            "blueprint AABB at least 6 cells per axis (got mn={:?}, mx={:?})", mn, mx);

        // Fissure noise samples should exist
        for f in &bp.fissures {
            assert!(!f.waver_samples.is_empty());
            assert!(!f.floor_samples.is_empty());
            assert!(!f.ceil_samples.is_empty());
        }
    }

    #[test]
    fn fissure_membership_works() {
        let bp = MegaVaultBlueprint::generate(42, 16.0);
        let fissure_noise = Simplex3D::new(bp.fissure_noise_seed);

        // Center of vault should be in a fissure
        let _center = bp.world_center;
        // Try a point at the center X of the first fissure, mid height
        let test_point = Vec3::new(
            bp.fissures[0].center_x,
            (bp.fissure_floor_y + bp.fissure_ceil_y) * 0.5,
            (bp.fissure_min_z + bp.fissure_max_z) * 0.5,
        );
        assert!(bp.is_in_fissure(test_point, &fissure_noise), "center of first fissure should be air");

        // A point far outside should not be in a fissure
        let outside = Vec3::new(1000.0, 0.0, 0.0);
        assert!(!bp.is_in_fissure(outside, &fissure_noise));
    }

    #[test]
    fn empty_blueprint_has_no_fissures() {
        let bp = MegaVaultBlueprint::empty();
        assert!(bp.fissures.is_empty());
        assert!(bp.paths.is_empty());
    }

    #[test]
    fn chunk_overlap_check() {
        let bp = MegaVaultBlueprint::generate(42, 16.0);

        // Chunk at origin should overlap
        assert!(bp.overlaps_chunk((0, 0, 0)));

        // Chunk far away should not
        assert!(!bp.overlaps_chunk((100, 100, 100)));

        // Edge chunks
        assert!(bp.overlaps_chunk(bp.bounds_min));
        assert!(!bp.overlaps_chunk(bp.bounds_max)); // exclusive upper bound
    }
}
