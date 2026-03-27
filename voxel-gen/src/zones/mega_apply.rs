//! Mega-Vault per-chunk application: single pass through 17^3 voxels.
//!
//! Reads the pre-computed [`MegaVaultBlueprint`] and applies all vault geometry
//! to one chunk's density field. Replaces the old 100-180 pass approach with
//! a single iteration + a small post-pass for additive cone shapes.

use glam::Vec3;
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

    // End-wall noise for organic boundaries
    let end_noise = Simplex3D::new(blueprint.mat_noise_seed.wrapping_add(0x0000_0002));

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

                // Priority 1: Tier-connecting tunnels
                if blueprint.is_in_tunnel(wp) {
                    if density.samples[idx].density > 0.0 {
                        density.samples[idx].density = -1.0;
                        density.samples[idx].material = Material::Air;
                    }
                    continue;
                }

                // Priority 2+3: Path check (handles both inline tunnels and ledges)
                if let Some(path_mat) = blueprint.path_at(wp) {
                    if path_mat == Material::Air {
                        // Inline tunnel carving
                        if density.samples[idx].density > 0.0 {
                            density.samples[idx].density = -1.0;
                            density.samples[idx].material = Material::Air;
                        }
                    } else {
                        // Ledge writing -- only write if denser than current
                        // (preserves existing solid that's already correct)
                        let edge_d = 0.9; // ledge density (matches old code)
                        if edge_d > density.samples[idx].density {
                            density.samples[idx].density = edge_d;
                            density.samples[idx].material = path_mat;
                        }
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
                        if 0.85 > density.samples[idx].density {
                            density.samples[idx].density = 0.85;
                            density.samples[idx].material = bridge_mat;
                        }
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

                // Priority 6: Fissure carving (main air space)
                if blueprint.is_in_fissure(wp) {
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

    // ── Post-pass: additive cone shapes for icicles + stalagmites ──
    // Inline cone writing directly into the density field (avoids HashMap overhead).

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
