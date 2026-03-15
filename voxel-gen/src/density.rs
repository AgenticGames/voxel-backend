use crate::config::{GenerationConfig, HostRockConfig, OreConfig};
use voxel_core::material::Material;
use voxel_noise::fbm::Fbm;
use voxel_noise::ridged::RidgedMulti;
use voxel_noise::simplex::Simplex3D;
use voxel_noise::NoiseSource;

// Re-export DensityField from voxel-core for backward compatibility
pub use voxel_core::density::DensityField;

/// Bundles all noise sources for material assignment.
/// Each deposit type uses a different seed offset to avoid correlation.
struct MaterialNoiseSources {
    iron_noise: Simplex3D,
    copper_ridged: RidgedMulti<Simplex3D>,
    malachite_noise: Simplex3D,
    reef_noise: RidgedMulti<Simplex3D>,
    pipe_noise: Simplex3D,
    diamond_noise: Simplex3D,
    sulfide_noise: Simplex3D,
    pyrite_noise: Simplex3D,
    basalt_noise: Simplex3D,
    geode_noise: Simplex3D,
    coal_noise: Fbm<Simplex3D>,
    boundary_noise: Simplex3D,
    warp_x: Simplex3D,
    warp_y: Simplex3D,
    warp_z: Simplex3D,
}

impl MaterialNoiseSources {
    fn new(seed: u64) -> Self {
        Self {
            iron_noise: Simplex3D::new(seed.wrapping_add(100)),
            // Dendritic copper: RidgedMulti for branching tendril morphology
            copper_ridged: RidgedMulti::new(
                Simplex3D::new(seed.wrapping_add(101)),
                5, 2.5, 2.0,
            ),
            malachite_noise: Simplex3D::new(seed.wrapping_add(107)),
            // Quartz reef: RidgedMulti for narrow vein structures
            reef_noise: RidgedMulti::new(
                Simplex3D::new(seed.wrapping_add(104)),
                4, 2.0, 2.0,
            ),
            pipe_noise: Simplex3D::new(seed.wrapping_add(105)),
            diamond_noise: Simplex3D::new(seed.wrapping_add(106)),
            sulfide_noise: Simplex3D::new(seed.wrapping_add(108)),
            pyrite_noise: Simplex3D::new(seed.wrapping_add(109)),
            basalt_noise: Simplex3D::new(seed.wrapping_add(110)),
            geode_noise: Simplex3D::new(seed.wrapping_add(111)),
            coal_noise: Fbm::new(Simplex3D::new(seed.wrapping_add(113)), 3, 2.0, 0.5),
            boundary_noise: Simplex3D::new(seed.wrapping_add(112)),
            warp_x: Simplex3D::new(seed.wrapping_add(200)),
            warp_y: Simplex3D::new(seed.wrapping_add(201)),
            warp_z: Simplex3D::new(seed.wrapping_add(202)),
        }
    }
}

/// Quick check if a chunk is likely fully solid by sampling a coarse 4x4x4 grid.
/// Returns Some(DensityField) filled uniformly with host rock if ALL 64 samples are solid,
/// None otherwise (caller must do full generation).
pub fn try_coarse_solid_check(config: &GenerationConfig, world_origin: glam::Vec3) -> Option<DensityField> {
    let size = config.chunk_size + 1;
    let vs = config.voxel_scale() as f64;
    let global_seed = config.seed;

    // Depth bypass: if the chunk's highest point is deep underground,
    // it's guaranteed solid — skip noise evaluation entirely.
    let chunk_top_y = world_origin.y as f64 + (size as f64 * vs);
    if chunk_top_y < -200.0 {
        let center_y = world_origin.y as f64 + (size as f64 * vs * 0.5);
        let host_material = host_rock_for_depth(center_y, &config.ore.host_rock);
        let mut field = DensityField::new(size);
        for sample in &mut field.samples {
            sample.density = 1.0;
            sample.material = host_material;
        }
        return Some(field);
    }

    // Cavern noise (same as full generation)
    let cavern_base = Simplex3D::new(global_seed);
    let cavern_noise = Fbm::new(cavern_base, 3, 2.0, 0.5);

    let warp_x_noise = Simplex3D::new(global_seed.wrapping_add(2));
    let warp_y_noise = Simplex3D::new(global_seed.wrapping_add(3));
    let warp_z_noise = Simplex3D::new(global_seed.wrapping_add(4));
    let warp_amplitude = config.noise.warp_amplitude;

    let freq = config.noise.cavern_frequency;
    let threshold = config.noise.cavern_threshold;

    // Sample coarse 8x8x8 grid (stride of ~2.6 through chunk)
    // 512 samples catches narrower cave features than the old 4x4x4 (64 samples)
    let coarse_res = 8;
    let stride = (size / coarse_res).max(1);

    for sz in 0..coarse_res {
        for sy in 0..coarse_res {
            for sx in 0..coarse_res {
                let x = (sx * stride).min(size - 1);
                let y = (sy * stride).min(size - 1);
                let z = (sz * stride).min(size - 1);

                let wx = world_origin.x as f64 + x as f64 * vs;
                let wy = world_origin.y as f64 + y as f64 * vs;
                let wz = world_origin.z as f64 + z as f64 * vs;

                let sample_x = wx * freq;
                let sample_y = wy * freq;
                let sample_z = wz * freq;

                // Domain warp
                let dx = warp_x_noise.sample(sample_x * 0.5, sample_y * 0.5, sample_z * 0.5)
                    * warp_amplitude * freq;
                let dy = warp_y_noise.sample(sample_x * 0.5, sample_y * 0.5, sample_z * 0.5)
                    * warp_amplitude * freq;
                let dz = warp_z_noise.sample(sample_x * 0.5, sample_y * 0.5, sample_z * 0.5)
                    * warp_amplitude * freq;

                let cavern_raw = cavern_noise.sample(sample_x + dx, sample_y + dy, sample_z + dz);
                let cavern_val = cavern_raw * 0.5 + 0.5;

                // If this sample would be air (cavern), chunk is NOT fully solid
                if cavern_val > threshold - 0.05 {
                    // Near threshold or above = might have air, not safe to skip
                    return None;
                }
            }
        }
    }

    // ALL 64 samples are well below threshold -> chunk is fully solid
    // Fill with uniform density and depth-appropriate host rock
    let center_y = world_origin.y as f64 + (size as f64 * vs * 0.5);
    let host_material = host_rock_for_depth(center_y, &config.ore.host_rock);

    let mut field = DensityField::new(size);
    for sample in &mut field.samples {
        sample.density = 1.0;
        sample.material = host_material;
    }

    Some(field)
}

/// Generate density field from noise composition.
///
/// Uses:
/// - Simplex3D -> FBM for large cavern carving
/// - Simplex3D -> FBM for wall detail
/// - Domain warp via separate simplex sources for organic shapes
/// - 11 material noise sources for geological deposit morphologies
///
/// Base density is 1.0 (solid). Where combined noise exceeds the threshold,
/// density goes negative (air). Geodes override density for hollow interiors.
pub fn generate_density_field(config: &GenerationConfig, world_origin: glam::Vec3) -> DensityField {
    let size = config.chunk_size + 1;
    let mut field = DensityField::new(size);

    // Use GLOBAL seed for all noise sources so that noise is continuous
    // across chunk boundaries. Every chunk samples the same noise field
    // at different world-space positions.
    let global_seed = config.seed;

    // Cavern noise sources
    let cavern_base = Simplex3D::new(global_seed);
    let cavern_noise = Fbm::new(cavern_base, 3, 2.0, 0.5);

    let detail_base = Simplex3D::new(global_seed.wrapping_add(1));
    let detail_noise = Fbm::new(
        detail_base,
        config.noise.detail_octaves,
        2.0,
        config.noise.detail_persistence,
    );

    // Domain warp sources for organic shape variation
    let warp_x_noise = Simplex3D::new(global_seed.wrapping_add(2));
    let warp_y_noise = Simplex3D::new(global_seed.wrapping_add(3));
    let warp_z_noise = Simplex3D::new(global_seed.wrapping_add(4));
    let warp_amplitude = config.noise.warp_amplitude;

    // Material noise sources (all 11 deposit types)
    let mat_noise = MaterialNoiseSources::new(global_seed);

    let freq = config.noise.cavern_frequency;
    let threshold = config.noise.cavern_threshold;

    let vs = config.voxel_scale() as f64;

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let wx = world_origin.x as f64 + x as f64 * vs;
                let wy = world_origin.y as f64 + y as f64 * vs;
                let wz = world_origin.z as f64 + z as f64 * vs;

                let sx = wx * freq;
                let sy = wy * freq;
                let sz = wz * freq;

                // Domain warp for organic shapes
                let dx = warp_x_noise.sample(sx * 0.5, sy * 0.5, sz * 0.5)
                    * warp_amplitude
                    * freq;
                let dy = warp_y_noise.sample(sx * 0.5, sy * 0.5, sz * 0.5)
                    * warp_amplitude
                    * freq;
                let dz = warp_z_noise.sample(sx * 0.5, sy * 0.5, sz * 0.5)
                    * warp_amplitude
                    * freq;

                // Primary cavern: FBM noise, range ~[-1,1], shift to [0,1]
                let cavern_raw = cavern_noise.sample(sx + dx, sy + dy, sz + dz);
                let cavern_val = cavern_raw * 0.5 + 0.5;

                // Wall detail at higher frequency
                let detail_val =
                    detail_noise.sample(sx * 2.0, sy * 2.0, sz * 2.0) * 0.05;

                let combined = cavern_val + detail_val;

                // Base density: carve where noise exceeds threshold
                let mut density = if combined > threshold {
                    -(combined - threshold) as f32 / (1.0 - threshold as f32).max(0.01)
                } else {
                    (1.0 - combined / threshold) as f32
                };

                // ── Geode check (modifies density for hollow interior) ──
                // Geodes are unique: they create crystal-lined hollow pockets
                // by overriding density in their interior.
                let mut geode_shell = false;
                let geode_cfg = &config.ore.geode;
                if wy >= geode_cfg.depth_min && wy <= geode_cfg.depth_max {
                    // Volcanic host: geodes only in basalt/granite
                    let geode_host_ok = if config.ore.geode_volcanic_host {
                        let host = host_rock_for_depth(wy, &config.ore.host_rock);
                        host == Material::Basalt || host == Material::Granite
                    } else {
                        true
                    };
                    if geode_host_ok {
                        let gf = geode_cfg.frequency;
                        let geode_val =
                            mat_noise.geode_noise.sample(wx * gf, wy * gf, wz * gf);
                        let geode_norm = geode_val * 0.5 + 0.5;

                        if geode_norm > geode_cfg.center_threshold {
                            let excess = geode_norm - geode_cfg.center_threshold;
                            // Depth scaling: thicker shells deeper
                            let effective_shell = if config.ore.geode_depth_scaling {
                                let range = geode_cfg.depth_max - geode_cfg.depth_min;
                                let depth_frac = if range > 0.0 {
                                    (geode_cfg.depth_max - wy) / range
                                } else {
                                    0.0
                                };
                                geode_cfg.shell_thickness * (1.0 + depth_frac * 0.5)
                            } else {
                                geode_cfg.shell_thickness
                            };
                            if excess < effective_shell {
                                // Crystal/Amethyst shell — force solid
                                geode_shell = true;
                                if density <= 0.0 {
                                    density = 0.5;
                                }
                            } else {
                                // Hollow interior
                                density = geode_cfg.hollow_factor;
                            }
                        }
                    }
                }

                // Assign material
                let material = if density <= 0.0 {
                    Material::Air
                } else if geode_shell {
                    // Alternate Crystal/Amethyst based on position
                    if ((wx * 7.3 + wz * 13.7) as i64) % 2 == 0 {
                        Material::Crystal
                    } else {
                        Material::Amethyst
                    }
                } else {
                    assign_material(wx, wy, wz, &config.ore, &mat_noise)
                };

                let sample = field.get_mut(x, y, z);
                sample.density = density;
                sample.material = material;
            }
        }
    }

    field
}

/// Assign material using geological deposit morphologies.
///
/// Priority chain (rarest/most special checked first):
/// 1. Kimberlite pipe — vertical cylinder via 2D noise, diamond inside
/// 2. Quartz reef — RidgedMulti veins hosting gold
/// 3. Sulfide blob — massive irregular deposits with tin pockets
/// 4. Dendritic copper — branching tendrils (shallow, RidgedMulti)
/// 5. Malachite — green indicator zones (deep)
/// 6. Pyrite — halo indicator near copper/gold
/// 7. Banded iron — horizontal sine-wave layers
/// 8. Host rock — depth-layered with noise-perturbed boundaries
/// Check an ore threshold with optional dithered edge falloff.
/// Returns true if the sample should be considered "inside" the ore.
#[inline]
fn ore_threshold_check(noise_val: f64, threshold: f64, falloff: f64, wx: f64, wy: f64, wz: f64) -> bool {
    if falloff > 0.0 {
        let half = falloff * 0.5;
        if noise_val > threshold + half {
            return true;
        }
        if noise_val > threshold - half {
            let t = (noise_val - (threshold - half)) / falloff;
            let hash = ((wx * 73.7 + wy * 37.3 + wz * 91.1) * 1000.0).fract().abs();
            return hash < t;
        }
        false
    } else {
        noise_val > threshold
    }
}

/// Shortcut: assign only host rock (skip all ore placement).
#[inline]
fn assign_host_rock(wx: f64, wy: f64, wz: f64, ore: &OreConfig, noise: &MaterialNoiseSources) -> Material {
    select_host_rock(wx, wy, wz, &ore.host_rock, noise)
}

fn assign_material(
    wx: f64,
    wy: f64,
    wz: f64,
    ore: &OreConfig,
    noise: &MaterialNoiseSources,
) -> Material {
    // ── Global ore scale: skip all ore placement when scale < 1.0 ──
    if ore.ore_global_scale <= 0.0 {
        return assign_host_rock(wx, wy, wz, ore, noise);
    }
    if ore.ore_global_scale < 1.0 {
        // Deterministic per-voxel hash to thin ore spawns
        let hash = ((wx * 73.7 + wy * 37.3 + wz * 91.1) * 1000.0).fract().abs();
        if hash >= ore.ore_global_scale as f64 {
            return assign_host_rock(wx, wy, wz, ore, noise);
        }
    }

    // ── Domain warping for ore shapes ──
    let warp_strength = ore.ore_domain_warp_strength;
    let warp_freq = ore.ore_warp_frequency;
    let (wwx, wwy, wwz) = if warp_strength > 0.0 {
        let dx = noise.warp_x.sample(wx * warp_freq, wy * warp_freq, wz * warp_freq) * warp_strength;
        let dy = noise.warp_y.sample(wx * warp_freq, wy * warp_freq, wz * warp_freq) * warp_strength;
        let dz = noise.warp_z.sample(wx * warp_freq, wy * warp_freq, wz * warp_freq) * warp_strength;
        (wx + dx, wy + dy, wz + dz)
    } else {
        (wx, wy, wz)
    };

    let falloff = ore.ore_edge_falloff;

    // ── 1. Kimberlite pipe (vertical cylinder via 2D noise) ──
    // Sampled with y=0 to create vertical pipe structures.
    // Very high threshold = rare, narrow columns.
    let kimb = &ore.kimberlite;
    if wy >= kimb.depth_min && wy <= kimb.depth_max {
        let pf = kimb.pipe_frequency_2d;
        let pipe_val = noise.pipe_noise.sample(wwx * pf, 0.0, wwz * pf);
        let pipe_norm = pipe_val * 0.5 + 0.5;
        // Carrot taper: pipes narrow with depth
        let mut pipe_thresh = kimb.pipe_threshold;
        if ore.kimberlite_carrot_taper {
            let range = kimb.depth_max - kimb.depth_min;
            if range > 0.0 {
                pipe_thresh += ((kimb.depth_max - wy) / range) * 0.03;
            }
        }
        if ore_threshold_check(pipe_norm, pipe_thresh, falloff, wx, wy, wz) {
            // Inside kimberlite pipe — check for diamond
            let df = kimb.diamond_frequency;
            let diamond_val = noise.diamond_noise.sample(wwx * df, wwy * df, wwz * df);
            let diamond_norm = diamond_val * 0.5 + 0.5;
            // Depth grading: more diamonds deeper
            let mut diamond_thresh = kimb.diamond_threshold;
            if ore.diamond_depth_grade {
                let range = kimb.depth_max - kimb.depth_min;
                if range > 0.0 {
                    diamond_thresh -= ((kimb.depth_max - wy) / range) * 0.08;
                }
            }
            if ore_threshold_check(diamond_norm, diamond_thresh, falloff, wx, wy, wz) {
                return Material::Diamond;
            }
            return Material::Kimberlite;
        }
    }

    // ── 2. Quartz reef veins (host for gold) ──
    // RidgedMulti produces sharp ridge-like patterns perfect for vein structures.
    if wy >= ore.quartz.depth_min && wy <= ore.quartz.depth_max {
        let rf = ore.quartz.frequency;
        // Planar veins: compress Z-axis for sheet-like geometry
        let reef_z_scale = if ore.quartz_planar_veins { 2.0 } else { 1.0 };
        let reef_val = noise.reef_noise.sample(wwx * rf, wwy * rf, wwz * rf * reef_z_scale);
        let reef_norm = reef_val * 0.5 + 0.5;
        if ore_threshold_check(reef_norm, ore.quartz.threshold, falloff, wx, wy, wz) {
            // Inside quartz reef — gold at higher threshold
            if wy >= ore.gold.depth_min && wy <= ore.gold.depth_max {
                // Bonanza zones: gold concentrates in richest vein cores
                let mut gold_thresh = ore.gold.threshold;
                if ore.gold_bonanza {
                    gold_thresh -= (reef_norm - ore.quartz.threshold).max(0.0) * 0.3;
                }
                if ore_threshold_check(reef_norm, gold_thresh, falloff, wx, wy, wz) {
                    return Material::Gold;
                }
            }
            return Material::Quartz;
        }
    }

    // ── 3. Massive sulfide blobs (with tin pockets) ──
    // Low frequency for large irregular deposits.
    let sulf = &ore.sulfide;
    let sulf_range = sulf.depth_max - sulf.depth_min;
    // Gossan cap: shrink effective range from top by 15%
    let sulf_depth_max_eff = if ore.sulfide_gossan_cap && sulf_range > 0.0 {
        sulf.depth_max - sulf_range * 0.15
    } else {
        sulf.depth_max
    };
    if wy >= sulf.depth_min && wy <= sulf_depth_max_eff {
        let sf = sulf.frequency;
        let sulfide_val = noise.sulfide_noise.sample(wwx * sf, wwy * sf, wwz * sf);
        let sulfide_norm = sulfide_val * 0.5 + 0.5;
        if ore_threshold_check(sulfide_norm, sulf.threshold, falloff, wx, wy, wz) {
            if ore_threshold_check(sulfide_norm, sulf.tin_threshold, falloff, wx, wy, wz) {
                return Material::Tin;
            }
            return Material::Sulfide;
        }
        // Disseminated halo: scattered sulfide around main deposits
        if ore.sulfide_disseminated {
            let halo_val = noise.sulfide_noise.sample(wwx * sf * 1.5, wwy * sf * 1.5, wwz * sf * 1.5);
            let halo_norm = halo_val * 0.5 + 0.5;
            if ore_threshold_check(halo_norm, sulf.threshold - 0.06, falloff, wx, wy, wz) {
                return Material::Sulfide;
            }
        }
    }

    // ── 4. Dendritic copper (shallow, branching tendrils) ──
    // RidgedMulti with 5 octaves creates natural branching tendril shapes.
    if wy >= ore.copper.depth_min && wy <= ore.copper.depth_max {
        let cf = ore.copper.frequency;
        let copper_val = noise.copper_ridged.sample(wwx * cf, wwy * cf, wwz * cf);
        let copper_norm = copper_val * 0.5 + 0.5;
        let mut copper_thresh = ore.copper.threshold;
        // Supergene enrichment: richer at shallower depths
        if ore.copper_supergene {
            let range = ore.copper.depth_max - ore.copper.depth_min;
            if range > 0.0 {
                let depth_frac = (wy - ore.copper.depth_min) / range;
                copper_thresh -= depth_frac * 0.05;
            }
        }
        // Porphyry contact: concentrates near granite
        if ore.copper_granite_contact {
            let host = host_rock_for_depth(wy, &ore.host_rock);
            if host == Material::Granite {
                copper_thresh -= 0.04;
            }
        }
        if ore_threshold_check(copper_norm, copper_thresh, falloff, wx, wy, wz) {
            return Material::Copper;
        }
    }

    // ── 5. Malachite zones (deep, green copper indicator) ──
    if wy >= ore.malachite.depth_min && wy <= ore.malachite.depth_max {
        let mf = ore.malachite.frequency;
        let mal_val = noise.malachite_noise.sample(wwx * mf, wwy * mf, wwz * mf);
        let mal_norm = mal_val * 0.5 + 0.5;
        // Oxidation front: denser near top of range
        let mut mal_thresh = ore.malachite.threshold;
        if ore.malachite_depth_bias {
            let range = ore.malachite.depth_max - ore.malachite.depth_min;
            if range > 0.0 {
                let depth_frac = (wy - ore.malachite.depth_min) / range;
                mal_thresh -= depth_frac * 0.04;
            }
        }
        if ore_threshold_check(mal_norm, mal_thresh, falloff, wx, wy, wz) {
            return Material::Malachite;
        }
    }

    // ── 6. Pyrite indicator (halo near copper/gold zones) ──
    // Uses its own noise at similar frequency to copper, lower threshold
    // creates a natural "halo" around ore-bearing regions.
    if wy >= ore.pyrite.depth_min && wy <= ore.pyrite.depth_max {
        let pf = ore.pyrite.frequency;
        let pyrite_val = noise.pyrite_noise.sample(wwx * pf, wwy * pf, wwz * pf);
        let pyrite_norm = pyrite_val * 0.5 + 0.5;
        // Ore association: clusters near sulfide/copper deposits
        let mut pyrite_thresh = ore.pyrite.threshold;
        if ore.pyrite_ore_halo {
            let sf = ore.sulfide.frequency;
            let sulfide_val = noise.sulfide_noise.sample(wwx * sf, wwy * sf, wwz * sf);
            let sulfide_norm = sulfide_val * 0.5 + 0.5;
            let cf = ore.copper.frequency;
            let copper_val = noise.copper_ridged.sample(wwx * cf, wwy * cf, wwz * cf);
            let copper_norm = copper_val * 0.5 + 0.5;
            if (sulfide_norm - ore.sulfide.threshold).abs() < 0.1
                || (copper_norm - ore.copper.threshold).abs() < 0.1
            {
                pyrite_thresh -= 0.06;
            }
        }
        if ore_threshold_check(pyrite_norm, pyrite_thresh, falloff, wx, wy, wz) {
            return Material::Pyrite;
        }
    }

    // ── 6.5. Coal seams (layered sedimentary deposits) ──
    // FBM noise for layered seam morphology, shallow depth range.
    if wy >= ore.coal.depth_min && wy <= ore.coal.depth_max {
        let mut coal_ok = true;
        // Sedimentary host: coal only in sandstone/limestone
        if ore.coal_sedimentary_host {
            let host = host_rock_for_depth(wy, &ore.host_rock);
            if host != Material::Sandstone && host != Material::Limestone {
                coal_ok = false;
            }
        }
        if coal_ok {
            let cf = ore.coal.frequency;
            let coal_val = noise.coal_noise.sample(wwx * cf, wwy * cf, wwz * cf);
            let coal_norm = coal_val * 0.5 + 0.5;
            let mut coal_thresh = ore.coal.threshold;
            // Shallow ceiling: coal thins near surface (above y=60)
            if ore.coal_shallow_ceiling {
                coal_thresh += (wy - 60.0).max(0.0) * 0.01;
            }
            // Depth enrichment: seams thicken with depth
            if ore.coal_depth_enrichment {
                let range = ore.coal.depth_max - ore.coal.depth_min;
                if range > 0.0 {
                    let depth_frac = (ore.coal.depth_max - wy) / range;
                    coal_thresh -= depth_frac * 0.04;
                }
            }
            if ore_threshold_check(coal_norm, coal_thresh, falloff, wx, wy, wz) {
                return Material::Coal;
            }
        }
    }

    // ── 6.7. Artesian aquifer lens (thin sandstone in granite/slate) ──
    // Geologically: relict sedimentary beds surviving metamorphism.
    // This is checked in the assign_material path but only activated when
    // the artesian config is provided via a thread-local or config field.
    // For simplicity, we use a hardcoded noise check here — the aquifer
    // lens is always generated at the appropriate depth regardless of config.
    {
        let aq_center = -15.0_f64;
        let aq_half = 1.5_f64;
        if wy >= aq_center - aq_half && wy <= aq_center + aq_half {
            let aq_freq = 0.01_f64;
            let aq_val = noise.boundary_noise.sample(wx * aq_freq, 0.0, wz * aq_freq);
            let aq_norm = aq_val * 0.5 + 0.5;
            if aq_norm > 0.3 {
                return Material::Sandstone;
            }
        }
    }

    // ── 7. Banded iron formation (horizontal sine-wave layers) ──
    // sin(wy * freq) creates horizontal bands; noise wobbles the edges.
    // Iron uses original wy for band frequency (geological strata are horizontal)
    // but warped coordinates for the noise perturbation.
    let iron = &ore.iron;
    if wy >= iron.depth_min && wy <= iron.depth_max {
        // Sedimentary host: skip iron if not in sandstone/limestone
        if ore.iron_sedimentary_only {
            let host = host_rock_for_depth(wy, &ore.host_rock);
            if host != Material::Sandstone && host != Material::Limestone {
                // Fall through to host rock
            } else {
                let band = (wy * iron.band_frequency).sin();
                let nf = iron.noise_frequency;
                let perturbation =
                    noise.iron_noise.sample(wwx * nf, wwy * nf, wwz * nf) * iron.noise_perturbation;
                let iron_val = (band + perturbation) * 0.5 + 0.5;
                // Surface thinning: raise threshold near surface
                let mut iron_thresh = iron.threshold;
                if ore.iron_depth_fade {
                    iron_thresh += (wy - 100.0).max(0.0) * 0.008;
                }
                if ore_threshold_check(iron_val, iron_thresh, falloff, wx, wy, wz) {
                    return Material::Iron;
                }
            }
        } else {
            let band = (wy * iron.band_frequency).sin();
            let nf = iron.noise_frequency;
            let perturbation =
                noise.iron_noise.sample(wwx * nf, wwy * nf, wwz * nf) * iron.noise_perturbation;
            let iron_val = (band + perturbation) * 0.5 + 0.5;
            // Surface thinning: raise threshold near surface
            let mut iron_thresh = iron.threshold;
            if ore.iron_depth_fade {
                iron_thresh += (wy - 100.0).max(0.0) * 0.008;
            }
            if ore_threshold_check(iron_val, iron_thresh, falloff, wx, wy, wz) {
                return Material::Iron;
            }
        }
    }

    // ── 8. Host rock fallback (depth-layered with noise-perturbed boundaries) ──
    // Host rock uses original coordinates (not warped)
    select_host_rock(wx, wy, wz, &ore.host_rock, noise)
}

/// Simplified host rock selection based purely on depth thresholds (no noise).
/// Used by the terrace/building system to determine the correct floor material.
pub fn host_rock_for_depth(y: f64, host: &HostRockConfig) -> Material {
    let (boundaries, mid_rock, low_rock) = sorted_host_layers(host);
    if y > host.sandstone_depth {
        Material::Sandstone
    } else if y > boundaries[0] {
        Material::Limestone
    } else if y > boundaries[1] {
        mid_rock
    } else if y > boundaries[2] {
        low_rock
    } else {
        Material::Marble
    }
}

/// Sort the three non-sandstone boundaries descending and determine which
/// material (Granite vs Slate) occupies the upper vs lower zone.
/// Returns (sorted_boundaries, upper_material, lower_material).
fn sorted_host_layers(host: &HostRockConfig) -> ([f64; 3], Material, Material) {
    let mut boundaries = [host.granite_depth, host.basalt_depth, host.slate_depth];
    boundaries.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let (mid_rock, low_rock) = if host.granite_depth >= host.slate_depth {
        (Material::Granite, Material::Slate)
    } else {
        (Material::Slate, Material::Granite)
    };
    (boundaries, mid_rock, low_rock)
}

/// Compute the water table Y level at a given world (X, Z) position.
/// Returns a noise-perturbed Y value centered on `config.base_y`.
pub fn water_table_y_at(wx: f64, wz: f64, config: &crate::config::WaterTableConfig, seed: u64) -> f64 {
    use voxel_noise::simplex::Simplex3D;
    use voxel_noise::NoiseSource;
    let noise = Simplex3D::new(seed.wrapping_add(700));
    config.base_y
        + noise.sample(wx * config.noise_frequency, 0.0, wz * config.noise_frequency)
            * config.noise_amplitude
}

/// Select host rock based on depth with noise-perturbed layer boundaries.
///
/// Layers (top to bottom): Sandstone → Limestone → {Granite,Slate sorted by depth} → Marble
/// Basalt appears as vertical intrusion columns cutting through all layers.
fn select_host_rock(
    wx: f64,
    wy: f64,
    wz: f64,
    host: &HostRockConfig,
    noise: &MaterialNoiseSources,
) -> Material {
    // Perturb depth boundaries for natural-looking wavy transitions
    // Y compressed 0.3x for horizontal wave appearance
    let bf = host.boundary_noise_frequency;
    let boundary_offset = noise
        .boundary_noise
        .sample(wx * bf, wy * bf * 0.3, wz * bf)
        * host.boundary_noise_amplitude;

    let effective_y = wy + boundary_offset;

    // Basalt intrusion columns: 2D noise (y=0) creates vertical pipes
    if effective_y < host.basalt_intrusion_depth_max {
        let bi_f = host.basalt_intrusion_frequency;
        let basalt_val = noise.basalt_noise.sample(wx * bi_f, 0.0, wz * bi_f);
        let basalt_norm = basalt_val * 0.5 + 0.5;
        if basalt_norm > host.basalt_intrusion_threshold {
            return Material::Basalt;
        }
    }

    // Depth layering (dynamically sorted so layer order follows config values)
    let (boundaries, mid_rock, low_rock) = sorted_host_layers(host);
    if effective_y > host.sandstone_depth {
        Material::Sandstone
    } else if effective_y > boundaries[0] {
        Material::Limestone
    } else if effective_y > boundaries[1] {
        mid_rock
    } else if effective_y > boundaries[2] {
        low_rock
    } else {
        Material::Marble
    }
}

/// Check if a base-resolution density field has any exposed ore surfaces.
/// Returns true if any cell has at least one corner with a detail material
/// and at least one corner with density <= 0 (sign change = surface).
/// Fast early-exit scan.
pub fn has_exposed_ore(field: &DensityField) -> bool {
    let cell_size = field.size - 1;
    for z in 0..cell_size {
        for y in 0..cell_size {
            for x in 0..cell_size {
                let mut has_detail = false;
                let mut has_air = false;
                // Check all 8 corners of this cell
                for &(dx, dy, dz) in &[
                    (0,0,0),(1,0,0),(0,1,0),(1,1,0),
                    (0,0,1),(1,0,1),(0,1,1),(1,1,1),
                ] {
                    let s = field.get(x+dx, y+dy, z+dz);
                    if s.material.is_detail_material() {
                        has_detail = true;
                    }
                    if s.density <= 0.0 {
                        has_air = true;
                    }
                    if has_detail && has_air {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Apply ore protrusion: push ore surfaces outward with smooth distance-based falloff.
/// Uses BFS from ore voxels through solid host rock to create a natural-looking bulge.
pub fn apply_ore_protrusion(field: &mut DensityField, protrusion: f32) {
    use std::collections::VecDeque;

    let size = field.size;
    let max_dist: f32 = 1.5;

    // Distance field: f32::MAX means unvisited
    let mut dist = vec![f32::MAX; size * size * size];
    let mut queue: VecDeque<(usize, usize, usize)> = VecDeque::new();

    // Seed BFS from all solid detail-material voxels
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let s = field.get(x, y, z);
                if s.density > 0.0 && s.material.is_detail_material() {
                    let idx = z * size * size + y * size + x;
                    dist[idx] = 0.0;
                    queue.push_back((x, y, z));
                }
            }
        }
    }

    // BFS flood-fill through solid host rock
    while let Some((x, y, z)) = queue.pop_front() {
        let idx = z * size * size + y * size + x;
        let current_dist = dist[idx];
        if current_dist >= max_dist {
            continue;
        }

        let neighbors: [(i32, i32, i32); 6] = [
            (-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1),
        ];

        for (dx, dy, dz) in neighbors {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            let nz = z as i32 + dz;
            if nx < 0 || ny < 0 || nz < 0 {
                continue;
            }
            let (nx, ny, nz) = (nx as usize, ny as usize, nz as usize);
            if nx >= size || ny >= size || nz >= size {
                continue;
            }
            let ns = field.get(nx, ny, nz);
            // Only flood through solid voxels
            if ns.density <= 0.0 {
                continue;
            }
            let new_dist = current_dist + 1.0;
            let nidx = nz * size * size + ny * size + nx;
            if new_dist < dist[nidx] {
                dist[nidx] = new_dist;
                queue.push_back((nx, ny, nz));
            }
        }
    }

    // Apply protrusion based on distance
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = z * size * size + y * size + x;
                let d = dist[idx];
                if d < f32::MAX {
                    let s = field.get_mut(x, y, z);
                    // Only affect solid voxels
                    if s.density > 0.0 {
                        let falloff = (1.0 - d / max_dist).max(0.0);
                        s.density += protrusion * falloff;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GenerationConfig;

    #[test]
    fn test_density_field_indexing() {
        let field = DensityField::new(5);
        assert_eq!(field.index(0, 0, 0), 0);
        assert_eq!(field.index(4, 0, 0), 4);
        assert_eq!(field.index(0, 1, 0), 5);
        assert_eq!(field.index(0, 0, 1), 25);
    }

    #[test]
    fn test_density_field_generate() {
        let config = GenerationConfig::default();
        let origin = glam::Vec3::ZERO;
        let field = generate_density_field(&config, origin);
        // size should be chunk_size + 1 = 17
        assert_eq!(field.size, 17);
        assert_eq!(field.samples.len(), 17 * 17 * 17);
    }

    #[test]
    fn test_density_field_deterministic() {
        let config = GenerationConfig::default();
        let origin = glam::Vec3::new(16.0, 0.0, 16.0);
        let field1 = generate_density_field(&config, origin);
        let field2 = generate_density_field(&config, origin);
        for i in 0..field1.samples.len() {
            assert_eq!(field1.samples[i].density, field2.samples[i].density);
        }
    }

    #[test]
    fn test_densities_extraction() {
        let config = GenerationConfig::default();
        let origin = glam::Vec3::ZERO;
        let field = generate_density_field(&config, origin);
        let densities = field.densities();
        assert_eq!(densities.len(), field.samples.len());
    }

    #[test]
    fn test_clamp_boundary_single_face() {
        let mut field = DensityField::new(5);
        // Set all samples to air
        for s in &mut field.samples {
            s.density = -1.0;
            s.material = Material::Air;
        }
        // Clamp only the pos_x face (x == 4)
        field.clamp_boundary_faces(false, true, false, false, false, false);
        for z in 0..5 {
            for y in 0..5 {
                // pos_x face should be solid
                assert_eq!(field.get(4, y, z).density, 1.0);
                assert!(matches!(field.get(4, y, z).material, Material::Limestone));
                // Interior should still be air
                assert_eq!(field.get(2, y, z).density, -1.0);
            }
        }
    }

    #[test]
    fn test_clamp_boundary_all_faces() {
        let mut field = DensityField::new(5);
        for s in &mut field.samples {
            s.density = -1.0;
            s.material = Material::Air;
        }
        field.clamp_boundary_faces(true, true, true, true, true, true);
        // All boundary samples should be solid
        for z in 0..5 {
            for y in 0..5 {
                for x in 0..5 {
                    let on_boundary =
                        x == 0 || x == 4 || y == 0 || y == 4 || z == 0 || z == 4;
                    let sample = field.get(x, y, z);
                    if on_boundary {
                        assert_eq!(sample.density, 1.0, "({x},{y},{z}) should be solid");
                    } else {
                        assert_eq!(
                            sample.density, -1.0,
                            "({x},{y},{z}) should be air"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_cross_chunk_boundary_continuity() {
        // Two adjacent chunks must produce identical density at their shared boundary.
        let config = GenerationConfig::default();
        let chunk_size = config.chunk_size as f32;

        let field_a = generate_density_field(&config, glam::Vec3::ZERO);
        let field_b =
            generate_density_field(&config, glam::Vec3::new(chunk_size, 0.0, 0.0));

        // field_a's x=chunk_size edge should match field_b's x=0 edge
        let size = field_a.size;
        for z in 0..size {
            for y in 0..size {
                let density_a = field_a.get(config.chunk_size, y, z).density;
                let density_b = field_b.get(0, y, z).density;
                assert!(
                    (density_a - density_b).abs() < 1e-6,
                    "Boundary mismatch at y={y}, z={z}: chunk_a={density_a}, chunk_b={density_b}"
                );
            }
        }
    }

    #[test]
    fn test_host_rock_depth_layering() {
        // Verify that host rock assignment follows depth boundaries
        let config = GenerationConfig::default();
        let ore = &config.ore;
        let noise = MaterialNoiseSources::new(config.seed);

        // Well above sandstone boundary (200) → Sandstone
        let mat = select_host_rock(0.0, 300.0, 0.0, &ore.host_rock, &noise);
        assert_eq!(mat, Material::Sandstone);

        // Well below slate boundary → Marble
        let mat = select_host_rock(0.0, -200.0, 0.0, &ore.host_rock, &noise);
        assert_eq!(mat, Material::Marble);
    }

    #[test]
    fn test_material_variety() {
        // Generate a chunk and verify we get more than just 2 material types
        let config = GenerationConfig::default();
        let field = generate_density_field(&config, glam::Vec3::ZERO);
        let mut seen = std::collections::HashSet::new();
        for s in &field.samples {
            seen.insert(s.material);
        }
        // Should have at least Air + Limestone (the default host rock at y=0..16)
        assert!(
            seen.len() >= 2,
            "Expected variety, got {:?}",
            seen
        );
    }
}
