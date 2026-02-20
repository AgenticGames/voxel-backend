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
    boundary_noise: Simplex3D,
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
            boundary_noise: Simplex3D::new(seed.wrapping_add(112)),
        }
    }
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
                    let gf = geode_cfg.frequency;
                    let geode_val =
                        mat_noise.geode_noise.sample(wx * gf, wy * gf, wz * gf);
                    let geode_norm = geode_val * 0.5 + 0.5;

                    if geode_norm > geode_cfg.center_threshold {
                        let excess = geode_norm - geode_cfg.center_threshold;
                        if excess < geode_cfg.shell_thickness {
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
fn assign_material(
    wx: f64,
    wy: f64,
    wz: f64,
    ore: &OreConfig,
    noise: &MaterialNoiseSources,
) -> Material {
    // ── 1. Kimberlite pipe (vertical cylinder via 2D noise) ──
    // Sampled with y=0 to create vertical pipe structures.
    // Very high threshold = rare, narrow columns.
    let kimb = &ore.kimberlite;
    if wy >= kimb.depth_min && wy <= kimb.depth_max {
        let pf = kimb.pipe_frequency_2d;
        let pipe_val = noise.pipe_noise.sample(wx * pf, 0.0, wz * pf);
        let pipe_norm = pipe_val * 0.5 + 0.5;
        if pipe_norm > kimb.pipe_threshold {
            // Inside kimberlite pipe — check for diamond
            let df = kimb.diamond_frequency;
            let diamond_val = noise.diamond_noise.sample(wx * df, wy * df, wz * df);
            let diamond_norm = diamond_val * 0.5 + 0.5;
            if diamond_norm > kimb.diamond_threshold {
                return Material::Diamond;
            }
            return Material::Kimberlite;
        }
    }

    // ── 2. Quartz reef veins (host for gold) ──
    // RidgedMulti produces sharp ridge-like patterns perfect for vein structures.
    if wy >= ore.quartz.depth_min && wy <= ore.quartz.depth_max {
        let rf = ore.quartz.frequency;
        let reef_val = noise.reef_noise.sample(wx * rf, wy * rf, wz * rf);
        let reef_norm = reef_val * 0.5 + 0.5;
        if reef_norm > ore.quartz.threshold {
            // Inside quartz reef — gold at higher threshold
            if wy >= ore.gold.depth_min
                && wy <= ore.gold.depth_max
                && reef_norm > ore.gold.threshold
            {
                return Material::Gold;
            }
            return Material::Quartz;
        }
    }

    // ── 3. Massive sulfide blobs (with tin pockets) ──
    // Low frequency for large irregular deposits.
    let sulf = &ore.sulfide;
    if wy >= sulf.depth_min && wy <= sulf.depth_max {
        let sf = sulf.frequency;
        let sulfide_val = noise.sulfide_noise.sample(wx * sf, wy * sf, wz * sf);
        let sulfide_norm = sulfide_val * 0.5 + 0.5;
        if sulfide_norm > sulf.threshold {
            if sulfide_norm > sulf.tin_threshold {
                return Material::Tin;
            }
            return Material::Sulfide;
        }
    }

    // ── 4. Dendritic copper (shallow, branching tendrils) ──
    // RidgedMulti with 5 octaves creates natural branching tendril shapes.
    if wy >= ore.copper.depth_min && wy <= ore.copper.depth_max {
        let cf = ore.copper.frequency;
        let copper_val = noise.copper_ridged.sample(wx * cf, wy * cf, wz * cf);
        let copper_norm = copper_val * 0.5 + 0.5;
        if copper_norm > ore.copper.threshold {
            return Material::Copper;
        }
    }

    // ── 5. Malachite zones (deep, green copper indicator) ──
    if wy >= ore.malachite.depth_min && wy <= ore.malachite.depth_max {
        let mf = ore.malachite.frequency;
        let mal_val = noise.malachite_noise.sample(wx * mf, wy * mf, wz * mf);
        let mal_norm = mal_val * 0.5 + 0.5;
        if mal_norm > ore.malachite.threshold {
            return Material::Malachite;
        }
    }

    // ── 6. Pyrite indicator (halo near copper/gold zones) ──
    // Uses its own noise at similar frequency to copper, lower threshold
    // creates a natural "halo" around ore-bearing regions.
    if wy >= ore.pyrite.depth_min && wy <= ore.pyrite.depth_max {
        let pf = ore.pyrite.frequency;
        let pyrite_val = noise.pyrite_noise.sample(wx * pf, wy * pf, wz * pf);
        let pyrite_norm = pyrite_val * 0.5 + 0.5;
        if pyrite_norm > ore.pyrite.threshold {
            return Material::Pyrite;
        }
    }

    // ── 7. Banded iron formation (horizontal sine-wave layers) ──
    // sin(wy * freq) creates horizontal bands; noise wobbles the edges.
    let iron = &ore.iron;
    if wy >= iron.depth_min && wy <= iron.depth_max {
        let band = (wy * iron.band_frequency).sin();
        let nf = iron.noise_frequency;
        let perturbation =
            noise.iron_noise.sample(wx * nf, wy * nf, wz * nf) * iron.noise_perturbation;
        let iron_val = (band + perturbation) * 0.5 + 0.5;
        if iron_val > iron.threshold {
            return Material::Iron;
        }
    }

    // ── 8. Host rock fallback (depth-layered with noise-perturbed boundaries) ──
    select_host_rock(wx, wy, wz, &ore.host_rock, noise)
}

/// Simplified host rock selection based purely on depth thresholds (no noise).
/// Used by the terrace/building system to determine the correct floor material.
pub fn host_rock_for_depth(y: f64, host: &HostRockConfig) -> Material {
    if y > host.sandstone_depth {
        Material::Sandstone
    } else if y > host.granite_depth {
        Material::Limestone
    } else if y > host.basalt_depth {
        Material::Granite
    } else if y > host.slate_depth {
        Material::Slate
    } else {
        Material::Marble
    }
}

/// Select host rock based on depth with noise-perturbed layer boundaries.
///
/// Layers (top to bottom): Sandstone → Limestone → Granite → Slate → Marble
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

    // Depth layering
    if effective_y > host.sandstone_depth {
        Material::Sandstone
    } else if effective_y > host.granite_depth {
        Material::Limestone
    } else if effective_y > host.basalt_depth {
        Material::Granite
    } else if effective_y > host.slate_depth {
        Material::Slate
    } else {
        Material::Marble
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
