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

                // Priority 2+3: Path check (handles both inline tunnels and ledges)
                if let Some((path_mat, path_density)) = blueprint.path_at(wp) {
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

                // Priority 4: Bridge geometry
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

                // Priority 5: Connecting tunnels between fissures
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

                // Priority 6: Fissure carving (main air space, with Y-dependent waver)
                if blueprint.is_in_fissure(wp, &fissure_noise) {
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
                if density.samples[idx].density <= 0.0 {
                    density.samples[idx].density = 1.0;
                    density.samples[idx].material = Material::Ice;
                    // Falls through to material classification below
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

    // ── Pass 2: Geometry refinement — reads actual carved result ──
    // Scans solid/air boundaries to:
    // 1. Spawn overhang icicles (solid above air = icicle candidate)
    // 2. Verify tunnel entrances are open (carve blocked doorways)
    // 3. Smooth jagged edges on ledge surfaces
    // 4. Add BlackIce patches on ledge edges near drops
    // 5. Place small ice formations on horizontal surfaces

    // ── Pass 2a: Tunnel doorway connectivity ──
    // For each tier tunnel, check entry/exit positions in this chunk.
    // If the doorway has no solid floor beneath it, write a small IceSheet platform.
    for tt in &blueprint.tier_tunnels {
        let entry_z = tt.z_start;
        let exit_z = tt.z_end;
        let tunnel_center_y_entry = tt.y_start;
        let tunnel_center_y_exit = tt.y_end;

        for &(door_z, door_y) in &[(entry_z, tunnel_center_y_entry), (exit_z, tunnel_center_y_exit)] {
            // Check if this doorway is in this chunk
            let door_world = Vec3::new(tt.wall_x, door_y, door_z);
            let local = door_world - origin;
            let gx = (local.x / vs).round() as i32;
            let gy = (local.y / vs).round() as i32;
            let gz = (local.z / vs).round() as i32;

            if gx < 0 || gx >= size as i32 || gy < 1 || gy >= size as i32 || gz < 0 || gz >= size as i32 {
                continue;
            }

            // Check if there's solid floor below the doorway
            let floor_y = gy - 1;
            if floor_y < 0 { continue; }

            let platform_half = 3i32; // platform extends ±3 voxels around doorway
            let mut has_floor = false;

            for dz in -platform_half..=platform_half {
                for dx in -platform_half..=platform_half {
                    let fx = gx + dx;
                    let fz = gz + dz;
                    if fx >= 0 && fx < size as i32 && fz >= 0 && fz < size as i32 {
                        let fi = fz as usize * size * size + floor_y as usize * size + fx as usize;
                        if density.samples[fi].density > 0.0 {
                            has_floor = true;
                            break;
                        }
                    }
                }
                if has_floor { break; }
            }

            // No floor found — write a small IceSheet platform
            if !has_floor {
                for dz in -platform_half..=platform_half {
                    for dx in -platform_half..=platform_half {
                        let fx = gx + dx;
                        let fz = gz + dz;
                        let fy = floor_y;
                        if fx >= 0 && fx < size as i32 && fz >= 0 && fz < size as i32 && fy >= 0 {
                            let fi = fz as usize * size * size + fy as usize * size + fx as usize;
                            if density.samples[fi].density <= 0.0 {
                                density.samples[fi].density = 0.85;
                                density.samples[fi].material = Material::IceSheet;
                            }
                            // Also write 1 voxel below for thickness
                            if fy > 0 {
                                let fi2 = fz as usize * size * size + (fy - 1) as usize * size + fx as usize;
                                if density.samples[fi2].density <= 0.0 {
                                    density.samples[fi2].density = 0.85;
                                    density.samples[fi2].material = Material::IceSheet;
                                }
                            }
                        }
                    }
                }
            }

            // Also verify the doorway itself is open (carve if blocked)
            for dy in 0..4i32 { // 4 voxels of headroom
                let door_gy = gy + dy;
                if door_gy >= size as i32 { break; }
                for dz in -1i32..=1 {
                    let door_gz = gz + dz;
                    if door_gz < 0 || door_gz >= size as i32 { continue; }
                    let di = door_gz as usize * size * size + door_gy as usize * size + gx as usize;
                    if density.samples[di].density > 0.0 {
                        density.samples[di].density = -1.0;
                        density.samples[di].material = Material::Air;
                    }
                }
            }
        }
    }

    // Also check inline tunnel entry/exit points from paths
    for path in &blueprint.paths {
        for (wi, wp) in path.waypoints.iter().enumerate() {
            if !wp.is_tunnel { continue; }
            // Check if this is entry or exit
            let prev_tunnel = if wi > 0 { path.waypoints[wi - 1].is_tunnel } else { false };
            let next_tunnel = if wi + 1 < path.waypoints.len() { path.waypoints[wi + 1].is_tunnel } else { false };
            if prev_tunnel && next_tunnel { continue; } // interior, skip

            let door_world = Vec3::new(wp.wall_x, wp.y, wp.z);
            let local = door_world - origin;
            let gx = (local.x / vs).round() as i32;
            let gy = (local.y / vs).round() as i32;
            let gz = (local.z / vs).round() as i32;
            if gx < 0 || gx >= size as i32 || gy < 0 || gy >= size as i32 || gz < 0 || gz >= size as i32 {
                continue;
            }

            // Ensure doorway headroom is clear
            for dy in 0..4i32 {
                let door_gy = gy + dy;
                if door_gy >= size as i32 { break; }
                let di = gz as usize * size * size + door_gy as usize * size + gx as usize;
                if density.samples[di].density > 0.0 {
                    density.samples[di].density = -1.0;
                    density.samples[di].material = Material::Air;
                }
            }
        }
    }

    // ── Pass 2b: Geometry scan ──
    let mut overhang_icicles: Vec<(Vec3, f32, f32, bool)> = Vec::new();
    let mut rng_pass2 = ChaCha8Rng::seed_from_u64(
        blueprint.mat_noise_seed.wrapping_add(chunk_key.0 as u64 * 7 + chunk_key.1 as u64 * 31 + chunk_key.2 as u64 * 97)
    );

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

                // ── 1. Overhang icicles: solid with air below ──
                if is_solid && air_below {
                    if mat == Material::Ice || mat == Material::IceSheet || mat == Material::Hoarfrost {
                        // 30% chance per overhang voxel (sampled every 2 to avoid clustering)
                        if x % 2 == 0 && z % 2 == 0 && rng_pass2.gen::<f32>() < 0.30 {
                            let wp = origin + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                            let len = rng_pass2.gen_range(4.0..10.0);
                            let rad = rng_pass2.gen_range(0.3..1.0);
                            let glow = rng_pass2.gen_bool(0.5);
                            overhang_icicles.push((wp, len, rad, glow));
                        }
                    }
                }

                // ── 2. BlackIce on ledge edges near drops ──
                // Solid floor with air on 2+ sides below = precarious edge
                if is_solid && air_above {
                    let mut air_sides = 0u32;
                    for &(dx, dz) in &[(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                        let nx = x as i32 + dx;
                        let nz = z as i32 + dz;
                        if nx >= 0 && nx < size as i32 && nz >= 0 && nz < size as i32 {
                            let ny = y as i32 - 1; // check below the neighbor
                            if ny >= 0 {
                                let ni = nz as usize * size * size + ny as usize * size + nx as usize;
                                if density.samples[ni].density <= 0.0 {
                                    air_sides += 1;
                                }
                            }
                        }
                    }
                    // Edge of a ledge: 2+ sides have air below = slippery danger zone
                    if air_sides >= 2 && !density.samples[idx].material.is_ore() {
                        density.samples[idx].material = Material::BlackIce;
                    }
                }

                // ── 3. Small stalagmites on wide floor surfaces ──
                // Solid floor with lots of headroom = place small upward spike
                if is_solid && air_above && x % 4 == 0 && z % 4 == 0 {
                    // Check 3 voxels of headroom
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
                        let len = rng_pass2.gen_range(2.0..5.0);
                        let rad = rng_pass2.gen_range(0.3..0.8);
                        // Write small stalagmite inline
                        write_cone_inline(density, origin, vs, size,
                            wp, len, rad, 1.0, Material::IceSheet, 2.0);
                    }
                }

                // ── 4. Hoarfrost accumulation on ceiling surfaces ──
                // Solid with air below = ceiling/overhang — add hoarfrost for frosty look
                if is_solid && air_below && !density.samples[idx].material.is_ore() {
                    if mat == Material::Ice && rng_pass2.gen::<f32>() < 0.3 {
                        density.samples[idx].material = Material::Hoarfrost;
                    }
                }

                // ── 5. Permafrost on deep interior surfaces near vault boundary ──
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

    let mut icicle_writes = 0u32;
    for icicle in &relevant_icicles {
        let before = density.samples.iter().filter(|s| s.material == Material::IceSheet).count();
        write_cone_inline(density, origin, vs, size,
            icicle.pos, icicle.length, icicle.radius, icicle.direction,
            Material::IceSheet, 2.0);
        let after = density.samples.iter().filter(|s| s.material == Material::IceSheet).count();
        if after > before { icicle_writes += (after - before) as u32; }
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

        // Platform disc under stalagmite
        for vz in 0..size {
            for vy in 0..size {
                for vx in 0..size {
                    let vwp = origin + Vec3::new(vx as f32 * vs, vy as f32 * vs, vz as f32 * vs);
                    let dx = vwp.x - stag.pos.x;
                    let dz = vwp.z - stag.pos.z;
                    let dist_h = (dx * dx + dz * dz).sqrt();
                    if dist_h > stag.platform_radius { continue; }
                    if vwp.y < stag.platform_y || vwp.y > stag.platform_y + stag.platform_thickness {
                        continue;
                    }
                    let vidx = vz * size * size + vy * size + vx;
                    if 0.85 > density.samples[vidx].density {
                        density.samples[vidx].density = 0.85;
                        density.samples[vidx].material = Material::IceSheet;
                    }
                }
            }
        }
    }

    // Diagnostic
    {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true)
            .open("D:/Unreal Projects/Mithril2026/Saved/icicle_debug.txt")
        {
            let _ = writeln!(f, "chunk({},{},{}) icicles={} stalags={} bp_total={} voxels_written={}",
                chunk_key.0, chunk_key.1, chunk_key.2,
                relevant_icicles.len(), relevant_stalagmites.len(),
                blueprint.icicles.len(), icicle_writes);
            if !relevant_icicles.is_empty() {
                let ic = &relevant_icicles[0];
                let _ = writeln!(f, "  first_icicle: pos=({:.1},{:.1},{:.1}) len={:.1} rad={:.1} dir={:.1}",
                    ic.pos.x, ic.pos.y, ic.pos.z, ic.length, ic.radius, ic.direction);
                let _ = writeln!(f, "  chunk_origin=({:.1},{:.1},{:.1}) vs={:.3} size={}",
                    origin.x, origin.y, origin.z, vs, size);
            }
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
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
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
