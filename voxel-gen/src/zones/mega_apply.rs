//! Mega-Vault per-chunk application: single pass through 17^3 voxels.
//!
//! Reads the pre-computed [`MegaVaultBlueprint`] and applies all vault geometry
//! to one chunk's density field. Replaces the old 100-180 pass approach with
//! a single iteration + a small post-pass for additive cone shapes.

use glam::Vec3;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::Rng;
use voxel_core::material::Material;
use voxel_noise::NoiseSource;
use voxel_noise::simplex::Simplex3D;

use crate::density::DensityField;

use super::mega_blueprint::MegaVaultBlueprint;

/// Apply the vault blueprint to a single chunk's density field.
///
/// Priority order within the single pass:
/// 1. Tier-connecting tunnels (carved air)
/// 2. Inline tunnels from path waypoints (carved air via path_at returning Air)
/// 3. Bridge geometry (solid with material)
/// 4. Path/ledge geometry (solid with material)
/// 5. Connecting tunnel between fissures (air or ice)
/// 6. Fissure carving (air)
/// 7. Organic end-wall bumps (boundary noise)
/// 8. Worm sealing (any existing air -> solid ice)
/// 9. Material classification for remaining solid
pub fn apply_vault_to_chunk(
    density: &mut DensityField,
    chunk_key: (i32, i32, i32),
    blueprint: &MegaVaultBlueprint,
    eb: f32,
) {
    use std::time::Instant;
    let chunk_start = Instant::now();
    let mut t_pass1 = std::time::Duration::ZERO;
    let mut t_2pre = std::time::Duration::ZERO;
    let mut t_2a = std::time::Duration::ZERO;
    let mut t_2b = std::time::Duration::ZERO;
    let mut t_2c = std::time::Duration::ZERO;
    let mut t_2d = std::time::Duration::ZERO;
    let mut t_2e = std::time::Duration::ZERO;
    let mut t_cones = std::time::Duration::ZERO;
    let size = density.size;
    let vs = eb / (size - 1) as f32;
    let origin = Vec3::new(
        chunk_key.0 as f32 * eb,
        chunk_key.1 as f32 * eb,
        chunk_key.2 as f32 * eb,
    );

    // Material classification noise (cheap at 0.04 freq, sampled lazily per-voxel)
    let mat_noise = Simplex3D::new(blueprint.mat_noise_seed);

    // End-wall noise for organic boundaries (legacy seed: global_seed + 0xF155_0005)
    let end_noise = Simplex3D::new(blueprint.mat_noise_seed.wrapping_add(0x0000_0002));

    // Fissure noise for Y-dependent waver (legacy: global_seed + 0xF155_0001)
    let fissure_noise = Simplex3D::new(blueprint.fissure_noise_seed);

    // Ramp noise for tier tunnel wobble (legacy: global_seed + 0xF155_0006)
    let ramp_noise = Simplex3D::new(blueprint.ramp_noise_seed);

    // ── Pre-filter: find which blueprint features overlap this chunk ──
    let chunk_world_min = origin - Vec3::splat(2.0);
    let chunk_world_max = origin + Vec3::splat(eb + 2.0);

    // Pre-filter fissures: which fissures could possibly affect this chunk?
    let relevant_fissure_indices: Vec<u32> = blueprint.fissures.iter()
        .filter(|f| {
            // Fissure runs along Z at center_x ± width/2 + waver margin
            let x_margin = f.width * 0.5 + eb * 0.4; // width + waver
            (f.center_x - x_margin) < chunk_world_max.x && (f.center_x + x_margin) > chunk_world_min.x
        })
        .map(|f| f.index)
        .collect();

    // Pre-filter paths: which paths have waypoints in this chunk's Z range?
    let relevant_paths: Vec<usize> = blueprint.paths.iter().enumerate()
        .filter(|(_, p)| {
            if p.waypoints.is_empty() { return false; }
            let first_z = p.waypoints[0].z;
            let last_z = p.waypoints[p.waypoints.len() - 1].z;
            // Check Z overlap
            if last_z < chunk_world_min.z - 5.0 || first_z > chunk_world_max.z + 5.0 { return false; }
            // Check X overlap (wall_x ± width)
            let x_min = p.wall_x - 15.0; // generous margin
            let x_max = p.wall_x + 15.0;
            x_max > chunk_world_min.x && x_min < chunk_world_max.x
        })
        .map(|(i, _)| i)
        .collect();

    // Pre-filter bridges: which bridges have waypoints near this chunk?
    let relevant_bridges: Vec<usize> = blueprint.bridges.iter().enumerate()
        .filter(|(_, b)| {
            b.waypoints.iter().any(|bwp| {
                bwp.x >= chunk_world_min.x - 5.0 && bwp.x <= chunk_world_max.x + 5.0
                && bwp.y >= chunk_world_min.y - 5.0 && bwp.y <= chunk_world_max.y + 5.0
                && bwp.z >= chunk_world_min.z - 5.0 && bwp.z <= chunk_world_max.z + 5.0
            })
        })
        .map(|(i, _)| i)
        .collect();

    // Pre-filter connecting tunnels
    let relevant_conn_tunnels: Vec<usize> = blueprint.connecting_tunnels.iter().enumerate()
        .filter(|(_, ct)| {
            ct.center_z >= chunk_world_min.z - ct.width_z && ct.center_z <= chunk_world_max.z + ct.width_z
            && ct.center_y >= chunk_world_min.y - ct.height && ct.center_y <= chunk_world_max.y + ct.height
            && ct.left_x <= chunk_world_max.x && ct.right_x >= chunk_world_min.x
        })
        .map(|(i, _)| i)
        .collect();

    // Check if this chunk is even inside the vault at all
    let in_vault = chunk_world_max.x > blueprint.world_min.x - eb
        && chunk_world_min.x < blueprint.world_max.x + eb
        && chunk_world_max.y > blueprint.world_min.y - eb
        && chunk_world_min.y < blueprint.world_max.y + eb
        && chunk_world_max.z > blueprint.world_min.z - eb
        && chunk_world_min.z < blueprint.world_max.z + eb;

    if !in_vault {
        // Quick timing and exit
        t_pass1 = chunk_start.elapsed();
        // Write timing report
        {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true)
                .open("D:/Unreal Projects/Mithril2026/Saved/vault_perf.txt")
            {
                let _ = writeln!(f, "chunk({},{},{}) SKIPPED (not in vault)",
                    chunk_key.0, chunk_key.1, chunk_key.2);
            }
        }
        return;
    }

    // Pre-compute nearest path waypoint index for each Z grid position
    // This replaces the per-voxel binary search with a O(1) table lookup
    let path_z_lookup: Vec<Vec<(usize, usize)>> = {
        let mut lookup: Vec<Vec<(usize, usize)>> = vec![Vec::new(); size]; // per grid-Z: (path_idx, waypoint_idx)
        for &pi in &relevant_paths {
            let path = &blueprint.paths[pi];
            for gz in 0..size {
                let world_z = origin.z + gz as f32 * vs;
                // Binary search once per grid Z, store result
                if let Ok(wi) = path.waypoints.binary_search_by(|w| {
                    w.z.partial_cmp(&world_z).unwrap_or(std::cmp::Ordering::Equal)
                }) {
                    lookup[gz].push((pi, wi));
                } else if let Err(wi) = path.waypoints.binary_search_by(|w| {
                    w.z.partial_cmp(&world_z).unwrap_or(std::cmp::Ordering::Equal)
                }) {
                    let actual_wi = if wi == 0 { 0 }
                        else if wi >= path.waypoints.len() { path.waypoints.len() - 1 }
                        else {
                            let d_prev = (path.waypoints[wi - 1].z - world_z).abs();
                            let d_next = (path.waypoints[wi].z - world_z).abs();
                            if d_prev < d_next { wi - 1 } else { wi }
                        };
                    let w = &path.waypoints[actual_wi];
                    let z_tolerance = if w.is_tunnel { 5.0 } else { 1.5 };
                    if (w.z - world_z).abs() <= z_tolerance {
                        lookup[gz].push((pi, actual_wi));
                    }
                }
            }
        }
        lookup
    };

    // ── Main pass: ONE iteration through all voxels ──
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = z * size * size + y * size + x;
                let wp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);

                // Quick bounds check -- skip voxels outside the vault's world extent
                // (small margin for boundary effects)
                if wp.x < blueprint.world_min.x - 2.0 || wp.x > blueprint.world_max.x + 2.0
                    || wp.y < blueprint.world_min.y - 2.0 || wp.y > blueprint.world_max.y + 2.0
                    || wp.z < blueprint.world_min.z - 2.0 || wp.z > blueprint.world_max.z + 2.0
                {
                    continue;
                }

                // Priority 1: Tier-connecting tunnels (with noise wobble)
                if blueprint.is_in_tunnel(wp, &ramp_noise) {
                    if density.samples[idx].density > 0.0 {
                        density.samples[idx].density = -1.0;
                        density.samples[idx].material = Material::Air;
                    }
                    continue;
                }

                // Priority 2+3: Path check using pre-computed Z lookup (no binary search)
                let path_result = {
                    let mut result: Option<(Material, f32)> = None;
                    for &(pi, wi) in &path_z_lookup[z] {
                        let path = &blueprint.paths[pi];
                        let w = &path.waypoints[wi];
                        // Inline path_at logic but skipping the search
                        if let Some(r) = blueprint.path_at_waypoint(wp, path, w, wi) {
                            result = Some(r);
                            break;
                        }
                    }
                    result
                };
                if let Some((path_mat, path_density)) = path_result {
                    if path_mat == Material::Air {
                        // Inline tunnel carving
                        if density.samples[idx].density > 0.0 {
                            density.samples[idx].density = -1.0;
                            density.samples[idx].material = Material::Air;
                        }
                    } else {
                        // Ledge writing — always force material (prevents Slate bleed-through)
                        density.samples[idx].density = density.samples[idx].density.max(path_density);
                        density.samples[idx].material = path_mat;
                    }
                    continue;
                }

                // Priority 4: Bridge geometry (skip if no bridges near this chunk)
                if !relevant_bridges.is_empty() {
                if let Some(bridge_mat) = blueprint.bridge_at(wp) {
                    if bridge_mat == Material::Air {
                        // Landing cave carving
                        if density.samples[idx].density > 0.0 {
                            density.samples[idx].density = -1.0;
                            density.samples[idx].material = Material::Air;
                        }
                    } else {
                        // Always force material on bridges
                        density.samples[idx].density = density.samples[idx].density.max(0.85);
                        density.samples[idx].material = bridge_mat;
                    }
                    continue;
                }
                } // end if !relevant_bridges.is_empty()

                // Priority 5: Connecting tunnels between fissures (skip if none near)
                if !relevant_conn_tunnels.is_empty() {
                if let Some(tunnel_mat) = blueprint.connecting_tunnel_at(wp) {
                    if tunnel_mat == Material::Air {
                        if density.samples[idx].density > 0.0 {
                            density.samples[idx].density = -1.0;
                            density.samples[idx].material = Material::Air;
                        }
                    } else {
                        density.samples[idx].density = 1.0;
                        density.samples[idx].material = tunnel_mat;
                    }
                    continue;
                }
                } // end if !relevant_conn_tunnels.is_empty()

                // Priority 6: Fissure carving (skip if no fissures overlap this chunk X)
                if !relevant_fissure_indices.is_empty() && blueprint.is_in_fissure(wp, &fissure_noise) {
                    // This is fissure air
                    density.samples[idx].density = -1.0;
                    density.samples[idx].material = Material::Air;
                    continue;
                }

                // Priority 7: Organic end-wall bumps near Z/Y boundaries
                let near_z_min = wp.z - blueprint.world_min.z;
                let near_z_max = blueprint.world_max.z - wp.z;
                let near_y_min = wp.y - blueprint.world_min.y;
                if near_z_min < eb * 0.5 || near_z_max < eb * 0.5 || near_y_min < eb * 0.3 {
                    let n = end_noise.sample(
                        wp.x as f64 * 0.08, wp.y as f64 * 0.08, wp.z as f64 * 0.08,
                    ) as f32;
                    let boundary_dist = near_z_min.min(near_z_max).min(near_y_min);
                    if density.samples[idx].density <= 0.0 && n > 0.2 && boundary_dist < eb * 0.3 {
                        density.samples[idx].density = 0.6;
                        density.samples[idx].material = Material::Ice;
                        continue;
                    }
                }

                // Priority 8: Seal worm holes -- any existing air inside vault becomes solid ice
                // BUT preserve air adjacent to ore voxels so ores stay exposed
                if density.samples[idx].density <= 0.0 {
                    let mut near_ore = false;
                    for &(dx, dy, dz) in &[(1i32,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)] {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        let nz = z as i32 + dz;
                        if nx >= 0 && nx < size as i32 && ny >= 0 && ny < size as i32 && nz >= 0 && nz < size as i32 {
                            let ni = nz as usize * size * size + ny as usize * size + nx as usize;
                            if density.samples[ni].density > 0.0 && density.samples[ni].material.is_ore() {
                                near_ore = true;
                                break;
                            }
                        }
                    }
                    if !near_ore {
                        density.samples[idx].density = 1.0;
                        density.samples[idx].material = Material::Ice;
                    }
                }

                // Priority 9: Material classification for solid voxels
                if density.samples[idx].density > 0.0 && !density.samples[idx].material.is_ore() {
                    // Surface classification
                    let above = if y + 1 < size { z * size * size + (y + 1) * size + x } else { idx };
                    let below = if y > 0 { z * size * size + (y - 1) * size + x } else { idx };
                    let is_floor = y + 1 < size && density.samples[above].density <= 0.0;
                    let is_ceiling = y > 0 && density.samples[below].density <= 0.0;
                    let is_wall = !is_floor && !is_ceiling && [
                        (x + 1, y, z), (x.wrapping_sub(1), y, z),
                        (x, y, z + 1), (x, y, z.wrapping_sub(1)),
                    ].iter().any(|&(nx, ny, nz)| {
                        nx < size && nz < size && {
                            let ni = nz * size * size + ny * size + nx;
                            density.samples[ni].density <= 0.0
                        }
                    });
                    let is_interior = !is_floor && !is_ceiling && !is_wall;

                    // Ledge detection
                    let below2 = if y > 1 { z * size * size + (y - 2) * size + x } else { idx };
                    let on_ledge = is_floor && y > 1 && density.samples[below2].density <= 0.0;
                    let is_ledge_underside = is_ceiling && y + 1 < size && {
                        let above2 = z * size * size + (y + 1) * size + x;
                        density.samples[above2].density > 0.0
                    };

                    // Only classify surface/near-surface voxels
                    if is_floor || is_ceiling || is_wall || is_interior {
                        density.samples[idx].material = blueprint.classify_material(
                            wp, is_floor, is_ceiling, is_wall, is_interior,
                            on_ledge, is_ledge_underside, &mat_noise,
                        );
                    }
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // PASS 2: Comprehensive geometry refinement
    // Sees actual carved result, fixes connectivity, adds details
    // ═══════════════════════════════════════════════════════════════════

    t_pass1 = chunk_start.elapsed();
    let t2_start = Instant::now();

    // ── 2-PRE: Natural boundaries + organic floor ──
    // 1. Where caves intersect vault bounds: noise-driven taper instead of flat wall
    // 2. Flat floor/ceiling/walls at vault bounds: add noise bumps for organic look
    {
        let boundary_depth = eb * 0.6;
        let floor_noise = Simplex3D::new(blueprint.mat_noise_seed.wrapping_add(0x0000_0099));

        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let wp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                    let idx = z * size * size + y * size + x;

                    let near_z_min = wp.z - blueprint.world_min.z;
                    let near_z_max = blueprint.world_max.z - wp.z;
                    let near_x_min = wp.x - blueprint.world_min.x;
                    let near_x_max = blueprint.world_max.x - wp.x;
                    let near_y_min = wp.y - blueprint.world_min.y;
                    let near_y_max = blueprint.world_max.y - wp.y;

                    // Floor organic treatment: noise-driven bumps on the bottom
                    if near_y_min < eb * 0.5 && near_y_min >= 0.0 {
                        let floor_n = floor_noise.sample(
                            wp.x as f64 * 0.07, wp.y as f64 * 0.1, wp.z as f64 * 0.07,
                        ) as f32;
                        // Add solid bumps rising from the floor
                        if density.samples[idx].density <= 0.0 && floor_n > 0.1 {
                            let bump_height = (floor_n - 0.1) * eb * 0.4;
                            if near_y_min < bump_height {
                                density.samples[idx].density = 0.6;
                                density.samples[idx].material = Material::Permafrost;
                            }
                        }
                        // Erode flat floor edges with noise
                        if density.samples[idx].density > 0.0 && floor_n < -0.2 && near_y_min < eb * 0.15 {
                            density.samples[idx].density = -1.0;
                            density.samples[idx].material = Material::Air;
                        }
                    }

                    // Ceiling organic treatment
                    if near_y_max < eb * 0.5 && near_y_max >= 0.0 {
                        let ceil_n = floor_noise.sample(
                            wp.x as f64 * 0.06, 100.0 + wp.y as f64 * 0.1, wp.z as f64 * 0.06,
                        ) as f32;
                        if density.samples[idx].density <= 0.0 && ceil_n > 0.15 {
                            let bump_depth = (ceil_n - 0.15) * eb * 0.35;
                            if near_y_max < bump_depth {
                                density.samples[idx].density = 0.5;
                                density.samples[idx].material = Material::Ice;
                            }
                        }
                    }

                    // Side/end wall transitions: noise taper where caves meet vault
                    let near_boundary = near_z_min.min(near_z_max).min(near_x_min).min(near_x_max);
                    if near_boundary < boundary_depth && near_boundary >= 0.0 {
                        if density.samples[idx].density > 0.0 {
                            // Check if any neighbor is external air
                            for &(dx, dy, dz) in &[(1i32,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)] {
                                let nx = x as i32 + dx;
                                let ny = y as i32 + dy;
                                let nz = z as i32 + dz;
                                if nx < 0 || nx >= size as i32 || ny < 0 || ny >= size as i32 || nz < 0 || nz >= size as i32 { continue; }
                                let ni = nz as usize * size * size + ny as usize * size + nx as usize;
                                if density.samples[ni].density <= 0.0 {
                                    let neighbor_wp = origin + Vec3::new(nx as f32 * vs, ny as f32 * vs, nz as f32 * vs);
                                    let in_vault_interior = neighbor_wp.x > blueprint.world_min.x + eb
                                        && neighbor_wp.x < blueprint.world_max.x - eb
                                        && neighbor_wp.z > blueprint.world_min.z + eb
                                        && neighbor_wp.z < blueprint.world_max.z - eb;
                                    if !in_vault_interior {
                                        // Noise-driven taper (not flat gradient)
                                        let taper_noise = floor_noise.sample(
                                            wp.x as f64 * 0.08, wp.y as f64 * 0.08, wp.z as f64 * 0.08,
                                        ) as f32;
                                        let taper_t = (near_boundary / boundary_depth).clamp(0.0, 1.0);
                                        let threshold = taper_t * 0.6 + taper_noise * 0.3;
                                        if threshold < 0.35 {
                                            // Carve — creates organic holes in the wall
                                            density.samples[idx].density = -1.0;
                                            density.samples[idx].material = Material::Air;
                                        } else if threshold < 0.5 {
                                            // Permafrost transition zone
                                            density.samples[idx].material = Material::Permafrost;
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let mut rng_pass2 = ChaCha8Rng::seed_from_u64(
        blueprint.mat_noise_seed.wrapping_add(chunk_key.0 as u64 * 7 + chunk_key.1 as u64 * 31 + chunk_key.2 as u64 * 97)
    );
    let mut overhang_icicles: Vec<(Vec3, f32, f32, bool)> = Vec::new();

    t_2pre = t2_start.elapsed();
    let t2a_start = Instant::now();

    // ── 2A: Tunnel sphere-chain carving ──
    // ALL tunnels (inline, tier, cross-fissure) are carved here via sphere chains.
    // Blueprint stores waypoint chains, pass 2 carves overlapping spheres along them.
    let chunk_min = origin - Vec3::splat(20.0); // margin for sphere overlap
    let chunk_max = origin + Vec3::splat(eb + 20.0);

    for chain in &blueprint.tunnel_chains {
        // Quick AABB check: does any waypoint fall near this chunk?
        let mut any_near = false;
        for wp in &chain.waypoints {
            if wp.x >= chunk_min.x && wp.x <= chunk_max.x
                && wp.y >= chunk_min.y && wp.y <= chunk_max.y
                && wp.z >= chunk_min.z && wp.z <= chunk_max.z
            {
                any_near = true;
                break;
            }
        }
        if !any_near { continue; }

        // Carve spheres at each waypoint — bounded iteration (only voxels near sphere)
        for (wi, wp) in chain.waypoints.iter().enumerate() {
            let wobble = ((wi as f32 * 0.7).sin() * 0.3 + 1.0) * chain.radius;

            // Compute grid-space bounding box of this sphere
            let local = *wp - origin;
            let gc = Vec3::new(local.x / vs, local.y / vs, local.z / vs);
            let gr = (wobble / vs).ceil() as i32 + 1;
            let lo_x = (gc.x as i32 - gr).max(0) as usize;
            let hi_x = ((gc.x as i32 + gr) as usize).min(size - 1);
            let lo_y = (gc.y as i32 - gr).max(0) as usize;
            let hi_y = ((gc.y as i32 + gr) as usize).min(size - 1);
            let lo_z = (gc.z as i32 - gr).max(0) as usize;
            let hi_z = ((gc.z as i32 + gr) as usize).min(size - 1);

            for z in lo_z..=hi_z { for y in lo_y..=hi_y { for x in lo_x..=hi_x {
                let vwp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                let dist = (vwp - *wp).length();
                if dist < wobble {
                    let idx = z * size * size + y * size + x;
                    if chain.is_blocked {
                        density.samples[idx].density = 1.0;
                        density.samples[idx].material = Material::Ice;
                    } else if density.samples[idx].density > 0.0 {
                        density.samples[idx].density = -1.0;
                        density.samples[idx].material = Material::Air;
                    }
                }
            }}}
        }
    }

    t_2a = t2a_start.elapsed();
    let t2b_start = Instant::now();

    // ── 2A-CHAMBERS: Carve ore rooms — distinct from tunnels ──
    for chain in &blueprint.tunnel_chains {
        for chamber in &chain.chambers {
            let local = chamber.center - origin;
            let gc = Vec3::new(local.x / vs, local.y / vs, local.z / vs);
            let gr = (chamber.radius / vs).ceil() as i32 + 3;

            let lo_x = (gc.x as i32 - gr).max(0) as usize;
            let hi_x = ((gc.x as i32 + gr) as usize).min(size - 1);
            let lo_y = (gc.y as i32 - gr).max(0) as usize;
            let hi_y = ((gc.y as i32 + gr) as usize).min(size - 1);
            let lo_z = (gc.z as i32 - gr).max(0) as usize;
            let hi_z = ((gc.z as i32 + gr) as usize).min(size - 1);

            if lo_x >= size || lo_y >= size || lo_z >= size { continue; }

            for z in lo_z..=hi_z { for y in lo_y..=hi_y { for x in lo_x..=hi_x {
                let vwp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                let dist = (vwp - chamber.center).length();
                let idx = z * size * size + y * size + x;

                // Slight noise for organic shape (not a perfect sphere)
                let shape_noise = ((vwp.x * 2.3 + vwp.y * 1.7 + vwp.z * 3.1) as f64).sin() as f32 * chamber.radius * 0.15;
                let effective_dist = dist + shape_noise;

                if effective_dist < chamber.radius * 0.75 {
                    // Inner chamber: carve to air
                    if density.samples[idx].density > 0.0 {
                        density.samples[idx].density = -1.0;
                        density.samples[idx].material = Material::Air;
                    }
                } else if effective_dist < chamber.radius * 0.85 {
                    // Transition: BlackIce floor, thin ore veins on walls
                    if density.samples[idx].density > 0.0 {
                        let above_idx = if y + 1 < size { z * size * size + (y + 1) * size + x } else { idx };
                        if y + 1 < size && density.samples[above_idx].density <= 0.0 {
                            density.samples[idx].material = Material::BlackIce;
                        } else {
                            let ore_noise = ((vwp.x * 3.7 + vwp.z * 5.3 + vwp.y * 2.1) as f64).sin() as f32 * 0.5 + 0.5;
                            if ore_noise > 0.55 { // halved: ~35% coverage (was ~75%)
                                density.samples[idx].material = chamber.ore_type;
                            } else {
                                density.samples[idx].material = Material::BlackIce;
                            }
                        }
                    }
                } else if effective_dist < chamber.radius {
                    // Outer shell: mostly BlackIce with sparse ore
                    if density.samples[idx].density > 0.0 && !density.samples[idx].material.is_ore() {
                        let ore_noise = ((vwp.x * 3.7 + vwp.z * 5.3 + vwp.y * 2.1) as f64).sin() as f32 * 0.5 + 0.5;
                        if ore_noise > 0.7 { // sparse: ~20% (was ~75%)
                            density.samples[idx].material = chamber.ore_type;
                        } else {
                            density.samples[idx].material = Material::BlackIce;
                        }
                    }
                }
            }}}
        }
    }

    // ── 2B: Widen cramped tunnel spots + smooth jagged walls ──
    // First collect changes, then apply (avoid read-during-write)
    let mut carve_queue: Vec<usize> = Vec::new();

    for z in 1..size.saturating_sub(1) {
        for y in 1..size.saturating_sub(1) {
            for x in 1..size.saturating_sub(1) {
                let idx = z * size * size + y * size + x;
                if density.samples[idx].density <= 0.0 { continue; } // only check solid

                // Count air neighbors (6-connected)
                let mut air_count = 0u32;
                for &(dx, dy, dz) in &[(1i32,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)] {
                    let nx = (x as i32 + dx) as usize;
                    let ny = (y as i32 + dy) as usize;
                    let nz = (z as i32 + dz) as usize;
                    if nx < size && ny < size && nz < size {
                        let ni = nz * size * size + ny * size + nx;
                        if density.samples[ni].density <= 0.0 { air_count += 1; }
                    }
                }

                // Sharp protrusion: solid with 4+ air neighbors = jagged, smooth it
                if air_count >= 4 {
                    carve_queue.push(idx);
                }

                // Thin wall detection: solid with air on opposite sides (X or Z axis)
                let air_left = if x > 0 { density.samples[z * size * size + y * size + (x-1)].density <= 0.0 } else { false };
                let air_right = if x + 1 < size { density.samples[z * size * size + y * size + (x+1)].density <= 0.0 } else { false };
                let air_front = if z > 0 { density.samples[(z-1) * size * size + y * size + x].density <= 0.0 } else { false };
                let air_back = if z + 1 < size { density.samples[(z+1) * size * size + y * size + x].density <= 0.0 } else { false };

                // Paper-thin wall between two air spaces = merge
                if (air_left && air_right) || (air_front && air_back) {
                    carve_queue.push(idx);
                }

                // Right-angle corner detection: solid with 2 perpendicular air faces
                // These create sharp 90-degree edges that look unnatural
                let air_above_local = if y + 1 < size { density.samples[z * size * size + (y+1) * size + x].density <= 0.0 } else { false };
                let air_below_local = if y > 0 { density.samples[z * size * size + (y-1) * size + x].density <= 0.0 } else { false };
                let perpendicular_pairs = [
                    (air_left, air_above_local), (air_left, air_below_local),
                    (air_right, air_above_local), (air_right, air_below_local),
                    (air_front, air_above_local), (air_front, air_below_local),
                    (air_back, air_above_local), (air_back, air_below_local),
                    (air_left, air_front), (air_left, air_back),
                    (air_right, air_front), (air_right, air_back),
                ];
                let corner_count = perpendicular_pairs.iter().filter(|&&(a, b)| a && b).count();
                // 3+ perpendicular air pairs = very exposed corner, smooth it
                if corner_count >= 3 && air_count >= 3 {
                    carve_queue.push(idx);
                }
            }
        }
    }

    // Apply carve queue
    for &idx in &carve_queue {
        density.samples[idx].density = -1.0;
        density.samples[idx].material = Material::Air;
    }

    t_2b = t2b_start.elapsed();
    let t2c_start = Instant::now();

    // ── 2C: Ledge-to-tunnel connectivity ──
    // For each tunnel doorway, check if there's a ledge nearby. If not, build a platform
    // connecting the doorway to the nearest ledge path waypoint.
    for tt in &blueprint.tier_tunnels {
        for &(door_z, door_y) in &[(tt.z_start, tt.y_start), (tt.z_end, tt.y_end)] {
            let door_world = Vec3::new(tt.wall_x, door_y, door_z);
            let local = door_world - origin;
            let gx = (local.x / vs).round() as i32;
            let gy = (local.y / vs).round() as i32;
            let gz = (local.z / vs).round() as i32;
            if gx < 0 || gx >= size as i32 || gy < 1 || gy >= size as i32 || gz < 0 || gz >= size as i32 { continue; }

            // Check for solid floor below doorway
            let floor_y = (gy - 1).max(0) as usize;
            let has_floor = {
                let fi = gz as usize * size * size + floor_y * size + gx as usize;
                gz >= 0 && (gz as usize) < size && (gx as usize) < size && density.samples[fi].density > 0.0
            };

            if !has_floor {
                // Build bigger platform: ±6 voxels, 40% chance BlackIce on top
                let plat = 6i32;
                for dz in -plat..=plat {
                    for dx in -plat..=plat {
                        for dy in 0..2i32 {
                            let px = gx + dx;
                            let py = floor_y as i32 - dy;
                            let pz = gz + dz;
                            if px >= 0 && px < size as i32 && py >= 0 && py < size as i32 && pz >= 0 && pz < size as i32 {
                                // Round platform shape
                                let dist = ((dx * dx + dz * dz) as f32).sqrt();
                                if dist < plat as f32 {
                                    let pi = pz as usize * size * size + py as usize * size + px as usize;
                                    if density.samples[pi].density <= 0.0 {
                                        density.samples[pi].density = 0.85;
                                        // 40% BlackIce on top layer, IceSheet underneath
                                        density.samples[pi].material = if dy == 0 && rng_pass2.gen::<f32>() < 0.4 {
                                            Material::BlackIce
                                        } else {
                                            Material::IceSheet
                                        };
                                    }
                                }
                            }
                        }
                    }
                }

                // Also try to bridge toward the fissure (build ledge connector)
                // Extend the platform toward the fissure opening along X
                let fissure_dir = if tt.side < 0.0 { 1i32 } else { -1i32 };
                for step in 0..12i32 { // wider connector
                    let bx = gx + fissure_dir * step;
                    if bx < 0 || bx >= size as i32 { break; }
                    for bz in (gz - 3)..=(gz + 3) { // wider Z spread
                        if bz < 0 || bz >= size as i32 { continue; }
                        for by in (floor_y as i32)..=(floor_y as i32 + 1) {
                            if by < 0 || by >= size as i32 { continue; }
                            let bi = bz as usize * size * size + by as usize * size + bx as usize;
                            if density.samples[bi].density <= 0.0 {
                                density.samples[bi].density = 0.85;
                                density.samples[bi].material = Material::IceSheet;
                            }
                        }
                    }
                }
            }
        }
    }

    t_2c = t2c_start.elapsed();
    let t2d_start = Instant::now();

    // ── 2D: Full geometry scan — icicles, materials, details ──
    for z in 0..size {
        for y in 1..size.saturating_sub(1) {
            for x in 0..size {
                let idx = z * size * size + y * size + x;
                let below_idx = z * size * size + (y - 1) * size + x;
                let above_idx = z * size * size + (y + 1) * size + x;

                let is_solid = density.samples[idx].density > 0.0;
                let air_below = density.samples[below_idx].density <= 0.0;
                let air_above = density.samples[above_idx].density <= 0.0;
                let mat = density.samples[idx].material;

                // Overhang icicles: solid with air below
                if is_solid && air_below {
                    if mat == Material::Ice || mat == Material::IceSheet || mat == Material::Hoarfrost {
                        if x % 2 == 0 && z % 2 == 0 && rng_pass2.gen::<f32>() < 0.30 {
                            let wp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                            let len = rng_pass2.gen_range(4.0..10.0);
                            let rad = rng_pass2.gen_range(0.3..1.0);
                            let glow = rng_pass2.gen_bool(0.5);
                            overhang_icicles.push((wp, len, rad, glow));
                        }
                    }

                    // Hoarfrost on ceilings
                    if !density.samples[idx].material.is_ore() && mat == Material::Ice && rng_pass2.gen::<f32>() < 0.3 {
                        density.samples[idx].material = Material::Hoarfrost;
                    }
                }

                // Ledge surface treatment: BlackIce patches + remove Hoarfrost
                if is_solid && air_above && !density.samples[idx].material.is_ore() {
                    // Remove Hoarfrost from ledge tops
                    if mat == Material::Hoarfrost {
                        density.samples[idx].material = Material::Ice;
                    }

                    // BlackIce in connected blobs using low-freq noise (not random scatter)
                    let wp_ledge = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                    let ice_noise = mat_noise.sample(
                        wp_ledge.x as f64 * 0.06,
                        wp_ledge.y as f64 * 0.03,
                        wp_ledge.z as f64 * 0.06,
                    ) as f32 * 0.5 + 0.5;
                    if ice_noise > 0.55 && density.samples[idx].material != Material::BlackIce {
                        // ~25% coverage in large connected patches
                        density.samples[idx].material = Material::BlackIce;
                    }

                    // Extra BlackIce on edges near drops
                    let mut air_sides = 0u32;
                    for &(dx, dz) in &[(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                        let nx = x as i32 + dx;
                        let nz = z as i32 + dz;
                        if nx >= 0 && nx < size as i32 && nz >= 0 && nz < size as i32 {
                            if y > 0 {
                                let ni = nz as usize * size * size + (y - 1) * size + nx as usize;
                                if density.samples[ni].density <= 0.0 { air_sides += 1; }
                            }
                        }
                    }
                    if air_sides >= 2 {
                        density.samples[idx].material = Material::BlackIce;
                    }
                }

                // Small floor stalagmites
                if is_solid && air_above && x % 4 == 0 && z % 4 == 0 {
                    let has_headroom = y + 3 < size && {
                        let h1 = z * size * size + (y + 1) * size + x;
                        let h2 = z * size * size + (y + 2) * size + x;
                        let h3 = z * size * size + (y + 3) * size + x;
                        density.samples[h1].density <= 0.0
                            && density.samples[h2].density <= 0.0
                            && density.samples[h3].density <= 0.0
                    };
                    if has_headroom && rng_pass2.gen::<f32>() < 0.08 {
                        let wp = origin + Vec3::new(x as f32 * vs, (y + 1) as f32 * vs, z as f32 * vs);
                        write_cone_inline(density, origin, vs, size, wp,
                            rng_pass2.gen_range(2.0..5.0), rng_pass2.gen_range(0.3..0.8),
                            1.0, Material::IceSheet, 2.0);
                    }
                }

                // Tunnel wall material refinement: solid adjacent to tunnel air
                if is_solid && !density.samples[idx].material.is_ore() && (air_below || air_above) {
                    // Check if this is a tunnel wall (multiple air neighbors in non-vertical directions)
                    let mut h_air = 0u32;
                    for &(dx, dz) in &[(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                        let nx = x as i32 + dx;
                        let nz = z as i32 + dz;
                        if nx >= 0 && nx < size as i32 && nz >= 0 && nz < size as i32 {
                            let ni = nz as usize * size * size + y * size + nx as usize;
                            if density.samples[ni].density <= 0.0 { h_air += 1; }
                        }
                    }
                    // Tunnel wall (enclosed, not fissure surface): use IceSheet
                    if h_air >= 2 && air_below {
                        density.samples[idx].material = Material::Hoarfrost; // tunnel ceiling
                    } else if h_air >= 2 && air_above {
                        density.samples[idx].material = Material::Permafrost; // tunnel floor
                    } else if h_air >= 1 {
                        density.samples[idx].material = Material::IceSheet; // tunnel wall
                    }
                }

                // Ore expansion: fatten deposits, connect isolates, spread along surfaces
                // Skip outer 5% of vault on all faces to avoid flat-zone blanket spread
                if is_solid && density.samples[idx].material.is_ore() {
                    let wp_ore = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                    let vault_sx = blueprint.world_max.x - blueprint.world_min.x;
                    let vault_sy = blueprint.world_max.y - blueprint.world_min.y;
                    let vault_sz = blueprint.world_max.z - blueprint.world_min.z;
                    let margin_x = vault_sx * 0.05;
                    let margin_y = vault_sy * 0.05;
                    let margin_z = vault_sz * 0.05;
                    let in_ore_zone = wp_ore.x > blueprint.world_min.x + margin_x
                        && wp_ore.x < blueprint.world_max.x - margin_x
                        && wp_ore.y > blueprint.world_min.y + margin_y
                        && wp_ore.y < blueprint.world_max.y - margin_y
                        && wp_ore.z > blueprint.world_min.z + margin_z
                        && wp_ore.z < blueprint.world_max.z - margin_z;
                    if !in_ore_zone { /* skip expansion near vault edges */ }
                    else if is_solid && density.samples[idx].material.is_ore() {
                    let ore_mat = density.samples[idx].material;

                    // Check if this ore is on a surface (has air neighbor)
                    let is_surface_ore = air_below || air_above || {
                        let mut has_h_air = false;
                        for &(dx, dz) in &[(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                            let nx = x as i32 + dx;
                            let nz = z as i32 + dz;
                            if nx >= 0 && nx < size as i32 && nz >= 0 && nz < size as i32 {
                                let ni = nz as usize * size * size + y * size + nx as usize;
                                if density.samples[ni].density <= 0.0 { has_h_air = true; break; }
                            }
                        }
                        has_h_air
                    };

                    {

                    let expand_chance = match ore_mat {
                        Material::Copper => 0.49,
                        Material::Gold => 0.56,
                        Material::Quartz => 0.56,
                        Material::Coal => 0.28,  // same as iron/rest
                        _ => 0.28,
                    };

                    let dirs: [(i32,i32,i32); 6] = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)];

                    if is_surface_ore {
                        // Surface ores: 100% spread to ALL ice neighbors (make veins visible and fat)
                        for &(dx, dy, dz) in &dirs {
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;
                            let nz = z as i32 + dz;
                            if nx >= 0 && nx < size as i32 && ny >= 0 && ny < size as i32 && nz >= 0 && nz < size as i32 {
                                let ni = nz as usize * size * size + ny as usize * size + nx as usize;
                                if density.samples[ni].density > 0.0 && !density.samples[ni].material.is_ore() {
                                    density.samples[ni].material = ore_mat;
                                }
                            }
                        }
                        // Also spread 1 voxel further along surface (extend reach)
                        for &(dx, dy, dz) in &dirs {
                            let nx = x as i32 + dx * 2;
                            let ny = y as i32 + dy * 2;
                            let nz = z as i32 + dz * 2;
                            if nx >= 0 && nx < size as i32 && ny >= 0 && ny < size as i32 && nz >= 0 && nz < size as i32 {
                                let ni = nz as usize * size * size + ny as usize * size + nx as usize;
                                if density.samples[ni].density > 0.0 && !density.samples[ni].material.is_ore() {
                                    if rng_pass2.gen::<f32>() < 0.5 { // 50% for 2nd ring
                                        density.samples[ni].material = ore_mat;
                                    }
                                }
                            }
                        }
                    } else {
                        // Interior ores: normal chance to spread to one random neighbor
                        if rng_pass2.gen::<f32>() < expand_chance {
                            let dir_idx = rng_pass2.gen_range(0..6usize);
                            let (dx, dy, dz) = dirs[dir_idx];
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;
                            let nz = z as i32 + dz;
                            if nx >= 0 && nx < size as i32 && ny >= 0 && ny < size as i32 && nz >= 0 && nz < size as i32 {
                                let ni = nz as usize * size * size + ny as usize * size + nx as usize;
                                if density.samples[ni].density > 0.0 && !density.samples[ni].material.is_ore() {
                                    density.samples[ni].material = ore_mat;
                                }
                            }
                        }
                    }

                    // Connect isolated ores: if this ore has same-type neighbor, try to bridge gaps
                    for &(dx, dy, dz) in &dirs {
                        let nx = x as i32 + dx * 2; // check 2 voxels away
                        let ny = y as i32 + dy * 2;
                        let nz = z as i32 + dz * 2;
                        let mx = x as i32 + dx; // the gap voxel between
                        let my = y as i32 + dy;
                        let mz = z as i32 + dz;
                        if nx >= 0 && nx < size as i32 && ny >= 0 && ny < size as i32 && nz >= 0 && nz < size as i32
                            && mx >= 0 && mx < size as i32 && my >= 0 && my < size as i32 && mz >= 0 && mz < size as i32
                        {
                            let ni = nz as usize * size * size + ny as usize * size + nx as usize;
                            let mi = mz as usize * size * size + my as usize * size + mx as usize;
                            // If there's same ore 2 away and the gap is ice, fill the gap
                            if density.samples[ni].material == ore_mat
                                && density.samples[mi].density > 0.0
                                && !density.samples[mi].material.is_ore()
                            {
                                density.samples[mi].material = ore_mat;
                            }
                        }
                    }
                    } // end else (coal skip)
                    } // end else (in_ore_zone)
                }

                // Permafrost at vault edges
                if is_solid && !density.samples[idx].material.is_ore() {
                    let wp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                    let dist_to_edge_x = (wp.x - blueprint.world_min.x).min(blueprint.world_max.x - wp.x);
                    let dist_to_edge_z = (wp.z - blueprint.world_min.z).min(blueprint.world_max.z - wp.z);
                    let dist_to_edge = dist_to_edge_x.min(dist_to_edge_z);
                    if dist_to_edge < eb * 0.3 && (air_below || air_above) {
                        density.samples[idx].material = Material::Permafrost;
                    }
                }
            }
        }
    }

    t_2d = t2d_start.elapsed();
    let t2e_start = Instant::now();

    // ── 2E: Ledge smoothing, buttress reinforcement, platform blending ──
    // Scan for ledge surfaces and fix jagged edges, strengthen wall connections,
    // and blend stalagmite platforms into the main ledge path.
    let mut ledge_fills: Vec<(usize, Material)> = Vec::new();

    for z in 1..size.saturating_sub(1) {
        for y in 1..size.saturating_sub(1) {
            for x in 1..size.saturating_sub(1) {
                let idx = z * size * size + y * size + x;
                let is_solid = density.samples[idx].density > 0.0;
                let above = z * size * size + (y + 1) * size + x;
                let below = z * size * size + (y - 1) * size + x;
                let air_above = y + 1 < size && density.samples[above].density <= 0.0;
                let air_below = y > 0 && density.samples[below].density <= 0.0;

                // 1. Ledge surface smoothing: if a ledge voxel has 2+ air side-neighbors
                //    at the same Y, fill the air gaps with solid to smooth the edge
                if is_solid && air_above {
                    let mut missing_neighbors = Vec::new();
                    for &(dx, dz) in &[(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                        let nx = x as i32 + dx;
                        let nz = z as i32 + dz;
                        if nx >= 0 && nx < size as i32 && nz >= 0 && nz < size as i32 {
                            let ni = nz as usize * size * size + y * size + nx as usize;
                            if density.samples[ni].density <= 0.0 {
                                // Check if the voxel BEYOND this gap is solid (gap is 1 wide = jagged edge)
                                let nx2 = nx + dx;
                                let nz2 = nz + dz;
                                if nx2 >= 0 && nx2 < size as i32 && nz2 >= 0 && nz2 < size as i32 {
                                    let ni2 = nz2 as usize * size * size + y * size + nx2 as usize;
                                    if density.samples[ni2].density > 0.0 {
                                        // 1-voxel gap between two solid ledge sections — fill it
                                        missing_neighbors.push(ni);
                                    }
                                }
                            }
                        }
                    }
                    for ni in missing_neighbors {
                        ledge_fills.push((ni, density.samples[idx].material));
                    }
                }

                // 2. Buttress reinforcement: if solid ledge has air below AND air on the
                //    wall-side (gap between ledge and cliff), fill with IceSheet to connect
                if is_solid && air_above && air_below {
                    // Check all 4 horizontal directions for the cliff wall
                    for &(dx, dz) in &[(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                        let nx = x as i32 + dx;
                        let nz = z as i32 + dz;
                        if nx < 0 || nx >= size as i32 || nz < 0 || nz >= size as i32 { continue; }
                        let ni = nz as usize * size * size + y * size + nx as usize;
                        if density.samples[ni].density <= 0.0 {
                            // Is there solid wall within 3 voxels in this direction?
                            let mut found_wall = false;
                            for step in 2..=3i32 {
                                let wx = x as i32 + dx * step;
                                let wz = z as i32 + dz * step;
                                if wx >= 0 && wx < size as i32 && wz >= 0 && wz < size as i32 {
                                    let wi = wz as usize * size * size + y * size + wx as usize;
                                    // Check if this is a tall wall (solid for 3+ voxels vertically)
                                    let wall_above = if y + 2 < size { density.samples[wz as usize * size * size + (y+2) * size + wx as usize].density > 0.0 } else { false };
                                    if density.samples[wi].density > 0.0 && wall_above {
                                        found_wall = true;
                                        break;
                                    }
                                }
                            }
                            if found_wall {
                                // Fill the gap between ledge and wall with IceSheet buttress
                                ledge_fills.push((ni, Material::IceSheet));
                                // Also fill below for thickness
                                if y > 0 {
                                    let below_ni = nz as usize * size * size + (y - 1) * size + nx as usize;
                                    ledge_fills.push((below_ni, Material::IceSheet));
                                }
                            }
                        }
                    }
                }

                // 3. Stalagmite platform blending: if a solid voxel at ledge height
                //    is IceSheet (platform) and has a neighbor that's Hoarfrost/Ice (ledge),
                //    smooth the material transition
                if is_solid && air_above && density.samples[idx].material == Material::IceSheet {
                    for &(dx, dz) in &[(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                        let nx = x as i32 + dx;
                        let nz = z as i32 + dz;
                        if nx >= 0 && nx < size as i32 && nz >= 0 && nz < size as i32 {
                            let ni = nz as usize * size * size + y * size + nx as usize;
                            let neighbor_mat = density.samples[ni].material;
                            if density.samples[ni].density > 0.0
                                && (neighbor_mat == Material::Ice || neighbor_mat == Material::BlackIce)
                            {
                                // Blend: make the platform edge match the ledge material
                                ledge_fills.push((idx, neighbor_mat));
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // Apply ledge fixes
    for (idx, mat) in &ledge_fills {
        if density.samples[*idx].density <= 0.0 {
            density.samples[*idx].density = 0.85;
        }
        if !density.samples[*idx].material.is_ore() {
            density.samples[*idx].material = *mat;
        }
    }

    t_2e = t2e_start.elapsed();
    let tcones_start = Instant::now();

    // Write overhang icicles discovered in pass 2
    for (pos, len, rad, glow) in &overhang_icicles {
        write_cone_inline(density, origin, vs, size,
            *pos, *len, *rad, -1.0, Material::IceSheet, 2.0);
        if *glow {
            let tip_pos = *pos + Vec3::new(0.0, -(*len - 1.5), 0.0);
            let tip_len = 1.5f32.min(*len * 0.3);
            write_cone_inline(density, origin, vs, size,
                tip_pos, tip_len, rad * 0.4, -1.0, Material::FrozenGlow, 2.5);
        }
    }

    // ── Post-pass: additive cone shapes from blueprint icicles + stalagmites ──

    let relevant_icicles = blueprint.icicles_in_chunk(chunk_key, eb);
    let relevant_stalagmites = blueprint.stalagmites_in_chunk(chunk_key, eb);

    for icicle in &relevant_icicles {
        write_cone_inline(density, origin, vs, size,
            icicle.pos, icicle.length, icicle.radius, icicle.direction,
            Material::IceSheet, 2.0);
        if icicle.has_glow_tip {
            let tip_offset = icicle.direction * (icicle.length - 1.5);
            let tip_pos = icicle.pos + Vec3::new(0.0, tip_offset, 0.0);
            let tip_len = 1.5f32.min(icicle.length * 0.3);
            write_cone_inline(density, origin, vs, size,
                tip_pos, tip_len, icicle.radius * 0.4, icicle.direction,
                Material::FrozenGlow, 2.5);
        }
    }

    for stag in &relevant_stalagmites {
        write_cone_inline(density, origin, vs, size,
            stag.pos, stag.length, stag.radius, 1.0,
            Material::IceSheet, 1.8);
        if stag.has_glow_tip {
            let tip_pos = stag.pos + Vec3::new(0.0, stag.length - 1.0, 0.0);
            let tip_len = 1.0f32.min(stag.length * 0.25);
            write_cone_inline(density, origin, vs, size,
                tip_pos, tip_len, stag.radius * 0.3, 1.0,
                Material::FrozenGlow, 2.5);
        }

        // Platform disc — bounded, clones ledge material, only extends toward wall
        {
            let local = stag.pos - origin;
            let gc = Vec3::new(local.x / vs, stag.platform_y / vs - origin.y / vs, local.z / vs);
            let gr = (stag.platform_radius / vs).ceil() as i32 + 1;
            let lo_x = (gc.x as i32 - gr).max(0) as usize;
            let hi_x = ((gc.x as i32 + gr) as usize).min(size - 1);
            let lo_y = ((stag.platform_y - origin.y) / vs).floor().max(0.0) as usize;
            let hi_y = (((stag.platform_y + stag.platform_thickness - origin.y) / vs).ceil() as usize).min(size - 1);
            let lo_z = (gc.z as i32 - gr).max(0) as usize;
            let hi_z = ((gc.z as i32 + gr) as usize).min(size - 1);

            // Find the dominant ledge material nearby to clone
            let mut ledge_mat = Material::Ice;
            'find_mat: for dz in -2i32..=2 {
                for dx in -2i32..=2 {
                    let sx = (gc.x as i32 + dx).max(0) as usize;
                    let sy = lo_y;
                    let sz = (gc.z as i32 + dz).max(0) as usize;
                    if sx < size && sy < size && sz < size {
                        let si = sz * size * size + sy * size + sx;
                        let m = density.samples[si].material;
                        if density.samples[si].density > 0.0 && (m == Material::Ice || m == Material::BlackIce || m == Material::IceSheet) {
                            ledge_mat = m;
                            break 'find_mat;
                        }
                    }
                }
            }

            for vz in lo_z..=hi_z {
                for vy in lo_y..=hi_y {
                    for vx in lo_x..=hi_x {
                        let vwp = origin + Vec3::new(vx as f32 * vs, vy as f32 * vs, vz as f32 * vs);
                        let dx = vwp.x - stag.pos.x;
                        let dz = vwp.z - stag.pos.z;
                        let dist_h = (dx * dx + dz * dz).sqrt();
                        if dist_h > stag.platform_radius { continue; }
                        let vidx = vz * size * size + vy * size + vx;
                        if 0.85 > density.samples[vidx].density {
                            density.samples[vidx].density = 0.85;
                            density.samples[vidx].material = ledge_mat;
                        }
                    }
                }
            }
        }
    }

    t_cones = tcones_start.elapsed();
    let total = chunk_start.elapsed();

    // Write timing report
    {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true)
            .open("D:/Unreal Projects/Mithril2026/Saved/vault_perf.txt")
        {
            let _ = writeln!(f, "chunk({},{},{}) total={:.2}ms | P1={:.2} 2PRE={:.2} 2A={:.2} 2B={:.2} 2C={:.2} 2D={:.2} 2E={:.2} cones={:.2}",
                chunk_key.0, chunk_key.1, chunk_key.2,
                total.as_secs_f64() * 1000.0,
                t_pass1.as_secs_f64() * 1000.0,
                t_2pre.as_secs_f64() * 1000.0,
                t_2a.as_secs_f64() * 1000.0,
                t_2b.as_secs_f64() * 1000.0,
                t_2c.as_secs_f64() * 1000.0,
                t_2d.as_secs_f64() * 1000.0,
                t_2e.as_secs_f64() * 1000.0,
                t_cones.as_secs_f64() * 1000.0,
            );
        }
    }
}

/// Write a cone shape directly into a single chunk's density field.
/// Same algorithm as `shapes::write_cone` but without HashMap indirection.
fn write_cone_inline(
    density: &mut DensityField,
    origin: Vec3,
    vs: f32,
    size: usize,
    anchor: Vec3,
    length: f32,
    base_radius: f32,
    direction: f32,
    material: Material,
    smoothness: f32,
) {
    // Bounded iteration: only check voxels in the cone's bounding box
    let tip = anchor + Vec3::new(0.0, direction * length, 0.0);
    let min_y = anchor.y.min(tip.y) - 1.0;
    let max_y = anchor.y.max(tip.y) + 1.0;
    let local_anchor = anchor - origin;
    let lo_x = ((local_anchor.x - base_radius - 1.0) / vs).floor().max(0.0) as usize;
    let hi_x = (((local_anchor.x + base_radius + 1.0) / vs).ceil() as usize).min(size - 1);
    let lo_y = (((min_y - origin.y) / vs).floor().max(0.0)) as usize;
    let hi_y = ((((max_y - origin.y) / vs).ceil()) as usize).min(size - 1);
    let lo_z = ((local_anchor.z - base_radius - 1.0) / vs).floor().max(0.0) as usize;
    let hi_z = (((local_anchor.z + base_radius + 1.0) / vs).ceil() as usize).min(size - 1);

    for z in lo_z..=hi_z {
        for y in lo_y..=hi_y {
            for x in lo_x..=hi_x {
                let world_pos = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                // Distance along cone axis (0 at anchor, 1 at tip)
                let t = (world_pos.y - anchor.y) / (direction * length);
                if t < 0.0 || t > 1.0 { continue; }
                // Radius tapers linearly from base_radius to 0
                let max_r = base_radius * (1.0 - t);
                let dist_h = ((world_pos.x - anchor.x).powi(2) + (world_pos.z - anchor.z).powi(2)).sqrt();
                if dist_h > max_r + 1.0 { continue; }
                let falloff = ((max_r - dist_h) * smoothness).min(1.0).max(0.0);
                if falloff > 0.0 {
                    let idx = z * size * size + y * size + x;
                    if falloff > density.samples[idx].density {
                        density.samples[idx].density = falloff;
                        density.samples[idx].material = material;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::mega_blueprint::MegaVaultBlueprint;
    use voxel_core::octree::node::VoxelSample;

    /// Create a test density field filled with solid rock.
    fn make_solid_chunk(size: usize) -> DensityField {
        DensityField {
            samples: vec![VoxelSample { density: 1.0, material: Material::Granite }; size * size * size],
            size,
            has_geode_material: false,
            air_cell_count: 0,
        }
    }

    #[test]
    fn apply_carves_fissure_air() {
        let bp = MegaVaultBlueprint::generate(42, 16.0);
        let eb = 16.0f32;
        let size = 17;

        // Pick a chunk that's inside the vault and likely intersects a fissure
        let chunk_key = (0, 0, 0);
        assert!(bp.overlaps_chunk(chunk_key), "origin chunk should overlap vault");

        let mut density = make_solid_chunk(size);
        apply_vault_to_chunk(&mut density, chunk_key, &bp, eb);

        // Count air voxels -- there should be some carved space
        let air_count = density.samples.iter().filter(|s| s.density <= 0.0).count();
        assert!(air_count > 0, "vault should have carved some air in chunk (0,0,0), got 0 air voxels");
    }

    #[test]
    fn apply_seals_existing_air() {
        let bp = MegaVaultBlueprint::generate(42, 16.0);
        let eb = 16.0f32;
        let size = 17;

        // Create a chunk that's mostly air (simulating worm holes)
        let chunk_key = (0, 2, 0); // upper part of vault
        let mut density = DensityField {
            samples: vec![VoxelSample { density: -1.0, material: Material::Air }; size * size * size],
            size,
            has_geode_material: false,
            air_cell_count: (size * size * size) as u32,
        };

        if bp.overlaps_chunk(chunk_key) {
            apply_vault_to_chunk(&mut density, chunk_key, &bp, eb);

            // Some previously-air voxels should now be solid (worm sealing)
            let solid_count = density.samples.iter().filter(|s| s.density > 0.0).count();
            assert!(solid_count > 0, "vault should seal some worm air into solid");
        }
    }

    #[test]
    fn apply_preserves_ores() {
        let bp = MegaVaultBlueprint::generate(42, 16.0);
        let eb = 16.0f32;
        let size = 17;
        let chunk_key = (0, 0, 0);

        if !bp.overlaps_chunk(chunk_key) { return; }

        // Create chunk with some ore
        let mut density = make_solid_chunk(size);
        // Place iron ore at a few positions
        for i in 0..10 {
            density.samples[i].material = Material::Iron;
        }

        apply_vault_to_chunk(&mut density, chunk_key, &bp, eb);

        // Ores that remain solid should still be ore
        let ore_count = density.samples.iter()
            .filter(|s| s.density > 0.0 && s.material == Material::Iron)
            .count();
        // Some ores might get carved (fissure air), but those that are still solid
        // should retain their material
        // We just check the classification didn't overwrite ALL of them
        assert!(ore_count > 0 || density.samples.iter().take(10).all(|s| s.density <= 0.0),
            "ores should be preserved unless carved to air");
    }
}
