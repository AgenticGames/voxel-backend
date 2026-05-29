//! Formation placement brushes: single hand-authored formations, random
//! scatter, and the procedural cavern-stamp brush.

use glam::Vec3;
use voxel_core::material::Material;
use voxel_gen::config::GenerationConfig;

use crate::store::ChunkStore;

use super::common::{capture_undo_for_range, chunk_range_for_sphere, finalize_brush, BrushOutcome};

/// Place a single hand-authored formation at `center`. The formation type maps
/// to a primitive shape: stalactite (cone tip-down), stalagmite (cone tip-up),
/// column (capsule), drapery (vertical fin), flowstone (mound), shield (disc),
/// rimstone-dam (curved wall). All formations are baked as voxel writes so they
/// persist via the standard chunk-snapshot save path.
///
/// `formation_type`: 0=Stalactite, 1=Stalagmite, 2=Column, 3=Drapery, 4=Flowstone,
///                   5=Shield, 6=RimstoneDam
/// `height`/`radius` in Rust world units; orientation is implicit per type.
pub fn place_formation(
    store: &mut ChunkStore,
    center: Vec3,
    formation_type: u8,
    height: f32,
    radius: f32,
    material: Material,
    config: &GenerationConfig,
    world_scale: f32,
) -> BrushOutcome {
    if height <= 0.0 || radius <= 0.0 || !material.is_solid() {
        return BrushOutcome { meshes: Vec::new(), flipped_chunks: Vec::new() };
    }

    // Half-extents of the formation's AABB (in Rust world units).
    let (half, base_offset) = match formation_type {
        0 => (Vec3::new(radius, height * 0.5, radius), Vec3::new(0.0, -height * 0.5, 0.0)), // stalactite hangs below center
        1 => (Vec3::new(radius, height * 0.5, radius), Vec3::new(0.0,  height * 0.5, 0.0)), // stalagmite rises above center
        2 => (Vec3::new(radius, height * 0.5, radius), Vec3::ZERO),                          // column centered
        3 => (Vec3::new(radius, height * 0.5, radius * 0.50), Vec3::ZERO),                   // drapery wavy fin (Z extent = thin half ±wave amp)
        4 => (Vec3::new(radius, height, radius), Vec3::new(0.0, height * 0.5, 0.0)),         // flowstone mound on floor
        5 => (Vec3::new(radius, height * 0.5, radius), Vec3::ZERO),                          // shield disc
        6 => (Vec3::new(radius, height * 0.5, radius), Vec3::ZERO),                          // rimstone arc
        _ => return BrushOutcome { meshes: Vec::new(), flipped_chunks: Vec::new() },
    };

    let aabb_center = center + base_offset;
    let aabb_min = aabb_center - half;
    let aabb_max = aabb_center + half;

    let eb = config.effective_bounds();
    let vs = config.voxel_scale();
    let lo = (
        (aabb_min.x / eb).floor() as i32,
        (aabb_min.y / eb).floor() as i32,
        (aabb_min.z / eb).floor() as i32,
    );
    let hi = (
        (aabb_max.x / eb).floor() as i32,
        (aabb_max.y / eb).floor() as i32,
        (aabb_max.z / eb).floor() as i32,
    );

    capture_undo_for_range(store, lo, hi);

    // SDF for this formation type, evaluated at world position `p`.
    // Returns "inside-ness" (positive = solid). Mirrors the carve_sphere/fill_sphere
    // sign convention.
    let formation_sdf = |p: Vec3| -> f32 {
        let local = p - aabb_center;
        match formation_type {
            // Stalactite (tip down): cone with apex at bottom
            0 => {
                let h = height;
                let t = ((local.y + h * 0.5) / h).clamp(0.0, 1.0); // 0=tip, 1=base
                let r_at_y = radius * t;
                let dxz = (local.x * local.x + local.z * local.z).sqrt();
                let inside_radial = r_at_y - dxz;
                let inside_y = (h * 0.5) - local.y.abs();
                inside_radial.min(inside_y)
            }
            // Stalagmite (tip up): cone with apex at top
            1 => {
                let h = height;
                let t = (1.0 - (local.y + h * 0.5) / h).clamp(0.0, 1.0); // 0=tip, 1=base
                let r_at_y = radius * t;
                let dxz = (local.x * local.x + local.z * local.z).sqrt();
                let inside_radial = r_at_y - dxz;
                let inside_y = (h * 0.5) - local.y.abs();
                inside_radial.min(inside_y)
            }
            // Column: cylinder
            2 => {
                let dxz = (local.x * local.x + local.z * local.z).sqrt();
                let inside_radial = radius - dxz;
                let inside_y = (height * 0.5) - local.y.abs();
                inside_radial.min(inside_y)
            }
            // Drapery: wavy thin Z-fin (X = wide, Z = thin, Y = tall) — the fin
            // undulates side-to-side along its X axis so it actually reads as a
            // hanging cave curtain instead of a flat slab. Wave wavelength scales
            // with radius so the curtain has ~3 humps regardless of size.
            3 => {
                let wave_freq = std::f32::consts::TAU * 1.5 / radius.max(1.0); // ~3 humps across
                let wave_amp  = radius * 0.20;                                 // displacement strength
                let z_offset  = (local.x * wave_freq).sin() * wave_amp;
                let inside_x  = radius - local.x.abs();
                let inside_z  = (radius * 0.25) - (local.z - z_offset).abs();
                let inside_y  = (height * 0.5) - local.y.abs();
                inside_x.min(inside_z).min(inside_y)
            }
            // Flowstone: half-ellipsoid mound (rises from floor)
            4 => {
                let nx = local.x / radius;
                let ny = (local.y / height).max(0.0);
                let nz = local.z / radius;
                let r = (nx * nx + ny * ny + nz * nz).sqrt();
                if local.y < 0.0 { -1.0 } else { (1.0 - r) * radius }
            }
            // Shield: oblate disc (Y thin, XZ wide). Earlier version had
            // `inside_y * 4.0` which was meant to "exaggerate flatness" but
            // multiplying by >1 made the Y constraint LESS limiting — the
            // result was a cylinder, not a disc. Compress the Y half-extent
            // to ~10% of `height` so the SDF is actually disc-shaped.
            5 => {
                let dxz = (local.x * local.x + local.z * local.z).sqrt();
                let inside_radial = radius - dxz;
                let disc_half_h   = (height * 0.5) * 0.2; // 20% of half-height = thin disc
                let inside_y      = disc_half_h - local.y.abs();
                inside_radial.min(inside_y)
            }
            // Rimstone dam: torus-arc wall
            6 => {
                let dxz = (local.x * local.x + local.z * local.z).sqrt();
                let dist_from_ring = (dxz - radius).abs();
                let inside_thickness = (radius * 0.25) - dist_from_ring;
                let inside_y = (height * 0.5) - local.y.abs();
                inside_thickness.min(inside_y)
            }
            _ => -1.0,
        }
    };

    let mut dirty_chunks: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> =
        Vec::new();

    for cz in lo.2..=hi.2 {
        for cy in lo.1..=hi.1 {
            for cx in lo.0..=hi.0 {
                if let Some(density) = store.density_fields.get_mut(&(cx, cy, cz)) {
                    let origin = Vec3::new(cx as f32 * eb, cy as f32 * eb, cz as f32 * eb);
                    let chunk_min = origin;
                    let chunk_max = origin + Vec3::splat(eb);
                    let isect_min = chunk_min.max(aabb_min);
                    let isect_max = chunk_max.min(aabb_max);
                    if isect_min.cmpgt(isect_max).any() {
                        continue;
                    }
                    let lo_x = (((isect_min.x - origin.x) / vs).floor() as i32).max(0) as usize;
                    let hi_x = (((isect_max.x - origin.x) / vs).ceil() as usize + 1).min(density.size);
                    let lo_y = (((isect_min.y - origin.y) / vs).floor() as i32).max(0) as usize;
                    let hi_y = (((isect_max.y - origin.y) / vs).ceil() as usize + 1).min(density.size);
                    let lo_z = (((isect_min.z - origin.z) / vs).floor() as i32).max(0) as usize;
                    let hi_z = (((isect_max.z - origin.z) / vs).ceil() as usize + 1).min(density.size);

                    let mut changed = false;
                    for z in lo_z..hi_z {
                        for y in lo_y..hi_y {
                            for x in lo_x..hi_x {
                                let world_pos = origin
                                    + Vec3::new(x as f32 * vs, y as f32 * vs, z as f32 * vs);
                                let inside = formation_sdf(world_pos);
                                if inside <= 0.0 {
                                    continue;
                                }
                                let sample = density.get_mut(x, y, z);
                                if inside > sample.density {
                                    sample.density = inside;
                                    sample.material = material;
                                    changed = true;
                                }
                            }
                        }
                    }

                    if changed {
                        let expand = config.mine.dirty_expand as usize;
                        let d_min_x = lo_x.saturating_sub(expand);
                        let d_min_y = lo_y.saturating_sub(expand);
                        let d_min_z = lo_z.saturating_sub(expand);
                        let d_max_x = (hi_x + expand).min(density.size - 1);
                        let d_max_y = (hi_y + expand).min(density.size - 1);
                        let d_max_z = (hi_z + expand).min(density.size - 1);
                        dirty_chunks.push((
                            (cx, cy, cz),
                            d_min_x, d_min_y, d_min_z,
                            d_max_x, d_max_y, d_max_z,
                        ));
                    }
                }
            }
        }
    }

    finalize_brush(store, dirty_chunks, config, world_scale)
}

/// Formation Stamp brush: places only stalactites, stalagmites, and
/// cave shields inside the brush sphere. Drapery, columns/mega-columns,
/// flowstone, rimstone dams, and cauldrons are excluded so the user can
/// paint pure decoration without changing the cave's macro silhouette.
///
/// Shield params are boosted relative to worldgen defaults so the rare
/// shield shape is actually visible from a single brush click and the
/// disk has clear tilt + a hanging stalactite for visual interest.
///
/// Materials are picked from each surface's natural host rock. Undo
/// captures the pre-state of every overlapping chunk so the user can
/// iterate vibes.
pub fn random_formations_brush(
    store: &mut ChunkStore,
    center: Vec3,
    radius: f32,
    seed: u64,
    config: &GenerationConfig,
    world_scale: f32,
) -> BrushOutcome {
    if radius <= 0.0 || !config.formations.enabled {
        return BrushOutcome { meshes: Vec::new(), flipped_chunks: Vec::new() };
    }

    let eb = config.effective_bounds();
    let (lo, hi) = chunk_range_for_sphere(center, radius, eb);

    capture_undo_for_range(store, lo, hi);

    // Margin around the sphere AABB to capture formation writes that extend
    // past the anchor (mega-column r_ceil up to ~base_radius+2 ≈ 10 cells,
    // stalactite cones up to ~length cells). Use a generous fixed margin.
    let dirty_margin: usize = 12;

    let mut dirty_chunks: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> =
        Vec::new();

    let aabb_min = center - Vec3::splat(radius);
    let aabb_max = center + Vec3::splat(radius);

    // Brush-specific FormationConfig override. Worldgen's defaults make
    // shields a 0.8%-per-wall-surface event — over a small brush region you'd
    // often get zero. Boosted chance + larger radii + steeper tilt range
    // turn each click into a visible shield cluster on whatever wall
    // surfaces the spatial filter picks up, with the hanging stalactite
    // always present so they read as "shields with drips" not flat coins.
    let mut brush_formations = config.formations.clone();
    brush_formations.shield_chance            = brush_formations.shield_chance.max(0.4);
    brush_formations.shield_radius_min        = brush_formations.shield_radius_min.max(2.5);
    brush_formations.shield_radius_max        = brush_formations.shield_radius_max.max(5.0);
    brush_formations.shield_max_tilt          = brush_formations.shield_max_tilt.max(60.0);
    brush_formations.shield_stalactite_chance = 1.0;

    let allowed = voxel_gen::formations::FORMATION_STALACTITE
        | voxel_gen::formations::FORMATION_STALAGMITE
        | voxel_gen::formations::FORMATION_SHIELD;

    for cz in lo.2..=hi.2 {
        for cy in lo.1..=hi.1 {
            for cx in lo.0..=hi.0 {
                let chunk_coord = (cx, cy, cz);
                let Some(density) = store.density_fields.get_mut(&chunk_coord) else {
                    continue;
                };

                let origin = Vec3::new(cx as f32 * eb, cy as f32 * eb, cz as f32 * eb);

                // Per-call randomized chunk_seed so re-stamping the same area
                // gives different placements. Mix the user-provided seed with
                // chunk coords for spatial coherence within a single click.
                let chunk_seed = seed
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add((cx as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
                    .wrapping_add((cy as u64).wrapping_mul(0x94D0_49BB_1331_11EB))
                    .wrapping_add((cz as u64).wrapping_mul(0x6C62_272E_07BB_0142));

                let _seeds = voxel_gen::formations::place_formations_filtered(
                    density,
                    &brush_formations,
                    origin,
                    config.seed,
                    chunk_seed,
                    chunk_coord,
                    Some((center, radius)),
                    allowed,
                );

                // Compute dirty rect = intersection(chunk AABB, brush sphere
                // AABB) expanded by `dirty_margin` cells to cover formation
                // writes that extend past the anchor.
                let chunk_min = origin;
                let chunk_max = origin + Vec3::splat(eb);
                let isect_min = chunk_min.max(aabb_min);
                let isect_max = chunk_max.min(aabb_max);
                if isect_min.cmpgt(isect_max).any() {
                    continue;
                }
                let vs = config.voxel_scale();
                let lo_x = (((isect_min.x - origin.x) / vs).floor() as i32 - dirty_margin as i32)
                    .max(0) as usize;
                let lo_y = (((isect_min.y - origin.y) / vs).floor() as i32 - dirty_margin as i32)
                    .max(0) as usize;
                let lo_z = (((isect_min.z - origin.z) / vs).floor() as i32 - dirty_margin as i32)
                    .max(0) as usize;
                let hi_x = (((isect_max.x - origin.x) / vs).ceil() as usize + dirty_margin + 1)
                    .min(density.size);
                let hi_y = (((isect_max.y - origin.y) / vs).ceil() as usize + dirty_margin + 1)
                    .min(density.size);
                let hi_z = (((isect_max.z - origin.z) / vs).ceil() as usize + dirty_margin + 1)
                    .min(density.size);

                dirty_chunks.push((chunk_coord, lo_x, lo_y, lo_z, hi_x, hi_y, hi_z));
            }
        }
    }

    finalize_brush(store, dirty_chunks, config, world_scale)
}

/// Cavern Stamp brush — chunk-snapped cave generator. Runs the worldgen
/// cave-carving phases (worms ± lava tubes/rivers) on a NxMxK chunk-aligned
/// region, optionally followed by pools + formations decoration. Each click
/// uses a fresh seed so re-stamping the same area gives a different cavern
/// layout. Worms carve additively (existing user edits in the chunks
/// survive — only solid → cave transitions happen).
///
/// `chunk_origin`: low corner chunk (x, y, z) of the brush region.
/// `extent`: number of chunks in each axis (NxMxK), each ≥ 1.
/// `decorate`: also run pools + formations after carving.
/// `fluids`: also run lava tubes + rivers.
/// `seed`: drives all randomness for this stamp; same seed + same input = same result.
pub fn cavern_stamp_brush(
    store: &mut ChunkStore,
    chunk_origin: (i32, i32, i32),
    extent: (u8, u8, u8),
    decorate: bool,
    fluids: bool,
    seed: u64,
    config: &GenerationConfig,
    world_scale: f32,
) -> BrushOutcome {
    if extent.0 == 0 || extent.1 == 0 || extent.2 == 0 {
        return BrushOutcome { meshes: Vec::new(), flipped_chunks: Vec::new() };
    }

    // 1. Build the coord list and capture undo for it
    let mut coords: Vec<(i32, i32, i32)> = Vec::with_capacity(
        extent.0 as usize * extent.1 as usize * extent.2 as usize,
    );
    for dz in 0..extent.2 as i32 {
        for dy in 0..extent.1 as i32 {
            for dx in 0..extent.0 as i32 {
                coords.push((
                    chunk_origin.0 + dx,
                    chunk_origin.1 + dy,
                    chunk_origin.2 + dz,
                ));
            }
        }
    }
    let lo = chunk_origin;
    let hi = (
        chunk_origin.0 + extent.0 as i32 - 1,
        chunk_origin.1 + extent.1 as i32 - 1,
        chunk_origin.2 + extent.2 as i32 - 1,
    );
    capture_undo_for_range(store, lo, hi);

    // 2. Run cavern carving on the brush coords (modifies store.density_fields
    //    in place but only entries whose key is in `coords`).
    voxel_gen::region_gen::carve_caverns_into_existing(
        &coords,
        &mut store.density_fields,
        config,
        seed,
        decorate,
        fluids,
    );

    // 3. Mark every brush chunk dirty in full. Cavern carving can touch any
    //    cell in any of these chunks, so re-extract the entire mesh per chunk.
    let mut dirty_chunks: Vec<((i32, i32, i32), usize, usize, usize, usize, usize, usize)> =
        Vec::with_capacity(coords.len());
    for c in &coords {
        if let Some(density) = store.density_fields.get(c) {
            let s = density.size;
            dirty_chunks.push((*c, 0, 0, 0, s, s, s));
        }
    }

    finalize_brush(store, dirty_chunks, config, world_scale)
}
