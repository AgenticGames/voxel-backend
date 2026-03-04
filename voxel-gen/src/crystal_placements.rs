//! Crystal placement computation for ore surface decorations.
//!
//! Scans density fields for solid ore voxels adjacent to air and computes
//! deterministic crystal instance positions, normals, and scales.

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;

use voxel_core::density::DensityField;
use voxel_core::material::Material;
use voxel_noise::domain_warp::DomainWarp;
use voxel_noise::ridged::RidgedMulti;
use voxel_noise::simplex::Simplex3D;
use voxel_noise::NoiseSource;

use crate::config::CrystalConfig;

/// A single crystal placement in chunk-relative coordinates (Rust Y-up).
#[derive(Debug, Clone)]
pub struct CrystalPlacement {
    /// Chunk-relative position X
    pub x: f32,
    /// Chunk-relative position Y
    pub y: f32,
    /// Chunk-relative position Z
    pub z: f32,
    /// Surface normal X
    pub normal_x: f32,
    /// Surface normal Y
    pub normal_y: f32,
    /// Surface normal Z
    pub normal_z: f32,
    /// Material enum value (7-19, excluding Crystal=18)
    pub ore_type: u8,
    /// Size class: 0=small, 1=medium, 2=large
    pub size_class: u8,
    /// Instance scale factor
    pub scale: f32,
}

/// Ore surface point detected during scanning.
struct OreSurface {
    x: usize,
    y: usize,
    z: usize,
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
    material: Material,
}

/// Compute crystal placements for a chunk's density field.
///
/// This is a pure function — does NOT modify the density field.
/// Returns an empty Vec if crystals are disabled or no ore surfaces exist.
pub fn compute_crystal_placements(
    density: &DensityField,
    config: &CrystalConfig,
    world_origin: glam::Vec3,
    world_seed: u64,
    chunk_seed: u64,
) -> Vec<CrystalPlacement> {
    if !config.enabled {
        eprintln!("[CRYSTAL] Crystals disabled in config");
        return Vec::new();
    }

    let size = density.size;

    // Step 1: Detect ore surfaces (solid ore voxels adjacent to air)
    let mut surfaces = detect_ore_surfaces(density, size);

    eprintln!("[CRYSTAL-DBG] origin=({:.0},{:.0},{:.0}) surfaces={} seed={} config.pyrite: enabled={} chance={} threshold={}",
        world_origin.x, world_origin.y, world_origin.z,
        surfaces.len(), world_seed,
        config.pyrite.enabled, config.pyrite.chance, config.pyrite.density_threshold);

    // Sort for deterministic RNG processing (HashMap iteration order is nondeterministic)
    surfaces.sort_by(|a, b| {
        a.z.cmp(&b.z)
            .then(a.y.cmp(&b.y))
            .then(a.x.cmp(&b.x))
    });

    // Step 2: Create scatter noise and RNG
    let scatter_noise = Simplex3D::new(world_seed.wrapping_add(300));
    let mut rng = ChaCha8Rng::seed_from_u64(chunk_seed.wrapping_add(0xCE75A100));

    let mut placements = Vec::new();

    // Step 3: Process each surface point
    for surface in &surfaces {
        let ore_config = config.ore_config(surface.material);

        // Skip if this ore type is disabled
        if !ore_config.enabled {
            continue;
        }

        // World position for noise sampling
        let wx = world_origin.x + surface.x as f32;
        let wy = world_origin.y + surface.y as f32;
        let wz = world_origin.z + surface.z as f32;

        let pass_gate = if ore_config.vein_enabled {
            // Vein mode: domain-warped ridged multifractal
            let mat_idx = surface.material as u64;
            let vein_seed = world_seed.wrapping_add(0xBE10_0000 + mat_idx * 100);
            let ridged = RidgedMulti::new(
                Simplex3D::new(vein_seed),
                ore_config.vein_octaves,
                ore_config.vein_lacunarity as f64,
                2.0, // gain
            );
            let vein_noise = DomainWarp::new(
                ridged,
                Simplex3D::new(vein_seed.wrapping_add(1)),
                Simplex3D::new(vein_seed.wrapping_add(2)),
                Simplex3D::new(vein_seed.wrapping_add(3)),
                ore_config.vein_warp_strength as f64,
            );
            let freq = ore_config.vein_frequency as f64;
            let v = vein_noise.sample(wx as f64 * freq, wy as f64 * freq, wz as f64 * freq);
            let v_norm = (v + 1.0) * 0.5; // Map [-1,1] to [0,1]
            if v_norm < (1.0 - ore_config.vein_thickness as f64) {
                false
            } else {
                rng.gen::<f32>() <= ore_config.vein_density
            }
        } else {
            // Scatter mode: original noise gate + chance gate
            let noise_val = scatter_noise.sample(wx as f64 * 0.1, wy as f64 * 0.1, wz as f64 * 0.1);
            let noise_normalized = (noise_val + 1.0) * 0.5;
            if noise_normalized < ore_config.density_threshold as f64 {
                false
            } else {
                rng.gen::<f32>() <= ore_config.chance
            }
        };

        if !pass_gate {
            continue;
        }

        // This is a seed point — generate a cluster
        generate_cluster(surface, ore_config, &mut rng, &mut placements);
    }

    placements
}

/// Scan density field for solid ore voxels adjacent to air.
fn detect_ore_surfaces(density: &DensityField, size: usize) -> Vec<OreSurface> {
    let mut surfaces = Vec::new();
    let scan_max = size - 1;

    for z in 1..scan_max {
        for y in 1..scan_max {
            for x in 1..scan_max {
                let sample = density.get(x, y, z);
                if sample.density <= 0.0 {
                    continue;
                }
                let mat = sample.material;
                if !is_crystal_eligible(mat) {
                    continue;
                }

                let mut nx = 0.0f32;
                let mut ny = 0.0f32;
                let mut nz = 0.0f32;
                let mut has_air = false;

                if x > 0 && density.get(x - 1, y, z).density <= 0.0 { nx -= 1.0; has_air = true; }
                if x + 1 < size && density.get(x + 1, y, z).density <= 0.0 { nx += 1.0; has_air = true; }
                if y > 0 && density.get(x, y - 1, z).density <= 0.0 { ny -= 1.0; has_air = true; }
                if y + 1 < size && density.get(x, y + 1, z).density <= 0.0 { ny += 1.0; has_air = true; }
                if z > 0 && density.get(x, y, z - 1).density <= 0.0 { nz -= 1.0; has_air = true; }
                if z + 1 < size && density.get(x, y, z + 1).density <= 0.0 { nz += 1.0; has_air = true; }

                if !has_air { continue; }

                let len = (nx * nx + ny * ny + nz * nz).sqrt().max(0.001);
                surfaces.push(OreSurface {
                    x, y, z,
                    normal_x: nx / len, normal_y: ny / len, normal_z: nz / len,
                    material: mat,
                });
            }
        }
    }
    surfaces
}

/// Generate a cluster of crystal placements from a seed surface point.
fn generate_cluster(
    surface: &OreSurface,
    ore_config: &crate::config::OreCrystalConfig,
    rng: &mut ChaCha8Rng,
    placements: &mut Vec<CrystalPlacement>,
) {
    let cluster_count = rng.gen_range(1..=ore_config.cluster_size.max(1));

    for c in 0..cluster_count {
        // Cluster offset (first crystal at exact position)
        let (ox, oy, oz) = if c == 0 {
            (0.0f32, 0.0f32, 0.0f32)
        } else {
            let r = ore_config.cluster_radius;
            (
                rng.gen_range(-r..=r),
                rng.gen_range(-r..=r),
                rng.gen_range(-r..=r),
            )
        };

        // Size class selection (weighted random)
        let size_class = select_size_class(
            rng,
            ore_config.small_weight,
            ore_config.medium_weight,
            ore_config.large_weight,
        );

        // Scale
        let scale = rng.gen_range(ore_config.scale_min..=ore_config.scale_max);

        // Normal alignment: lerp between random direction and surface normal
        let random_dir = random_unit_vec(rng);
        let alignment = ore_config.normal_alignment;
        let nx = lerp(random_dir.0, surface.normal_x, alignment);
        let ny = lerp(random_dir.1, surface.normal_y, alignment);
        let nz = lerp(random_dir.2, surface.normal_z, alignment);
        // Normalize
        let len = (nx * nx + ny * ny + nz * nz).sqrt().max(0.001);
        let nx = nx / len;
        let ny = ny / len;
        let nz = nz / len;

        // Position: at voxel center + configurable offset along blended normal
        let px = surface.x as f32 + 0.5 + ox + nx * ore_config.surface_offset;
        let py = surface.y as f32 + 0.5 + oy + ny * ore_config.surface_offset;
        let pz = surface.z as f32 + 0.5 + oz + nz * ore_config.surface_offset;

        placements.push(CrystalPlacement {
            x: px,
            y: py,
            z: pz,
            normal_x: nx,
            normal_y: ny,
            normal_z: nz,
            ore_type: surface.material as u8,
            size_class,
            scale,
        });
    }
}

fn is_crystal_eligible(mat: Material) -> bool {
    match mat {
        Material::Iron | Material::Copper | Material::Malachite | Material::Tin
        | Material::Gold | Material::Diamond | Material::Kimberlite | Material::Sulfide
        | Material::Quartz | Material::Pyrite | Material::Amethyst | Material::Coal => true,
        _ => false,
    }
}

fn select_size_class(rng: &mut ChaCha8Rng, small: f32, medium: f32, large: f32) -> u8 {
    let total = small + medium + large;
    if total <= 0.0 { return 0; }
    let roll = rng.gen::<f32>() * total;
    if roll < small { 0 } else if roll < small + medium { 1 } else { 2 }
}

fn random_unit_vec(rng: &mut ChaCha8Rng) -> (f32, f32, f32) {
    loop {
        let x = rng.gen_range(-1.0f32..=1.0);
        let y = rng.gen_range(-1.0f32..=1.0);
        let z = rng.gen_range(-1.0f32..=1.0);
        let len_sq = x * x + y * y + z * z;
        if len_sq > 0.01 && len_sq <= 1.0 {
            let len = len_sq.sqrt();
            return (x / len, y / len, z / len);
        }
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 { a + (b - a) * t }

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::density::DensityField;
    use voxel_core::material::Material;
    use crate::config::CrystalConfig;

    /// Helper: create a density field with a single ore voxel at (8,8,8)
    /// surrounded by air on all sides.
    fn make_test_field(mat: Material) -> DensityField {
        let size = 17; // chunk_size + 1
        let mut density = DensityField::new(size);
        // Default is solid (density=1.0, Limestone) so carve air around center
        // Set everything to air first
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let s = density.get_mut(x, y, z);
                    s.density = -1.0;
                    s.material = Material::Air;
                }
            }
        }
        // Place ore at center
        let s = density.get_mut(8, 8, 8);
        s.density = 1.0;
        s.material = mat;
        density
    }

    #[test]
    fn test_disabled_returns_empty() {
        let density = make_test_field(Material::Iron);
        let mut config = CrystalConfig::default();
        config.enabled = false;
        let placements = compute_crystal_placements(
            &density,
            &config,
            glam::Vec3::ZERO,
            42,
            100,
        );
        assert!(placements.is_empty());
    }

    #[test]
    fn test_no_ore_returns_empty() {
        let size = 17;
        let mut density = DensityField::new(size);
        // All solid limestone — no ore surfaces
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let s = density.get_mut(x, y, z);
                    s.density = 1.0;
                    s.material = Material::Limestone;
                }
            }
        }
        let config = CrystalConfig::default();
        let placements = compute_crystal_placements(
            &density,
            &config,
            glam::Vec3::ZERO,
            42,
            100,
        );
        assert!(placements.is_empty());
    }

    #[test]
    fn test_single_ore_produces_placements() {
        let density = make_test_field(Material::Gold);
        let mut config = CrystalConfig::default();
        // Force high chance so we always get a placement
        config.gold.chance = 1.0;
        config.gold.density_threshold = 0.0;
        config.gold.cluster_size = 1;
        let placements = compute_crystal_placements(
            &density,
            &config,
            glam::Vec3::ZERO,
            42,
            100,
        );
        assert!(!placements.is_empty(), "Should produce at least one placement");
        for p in &placements {
            assert_eq!(p.ore_type, Material::Gold as u8);
            assert!(p.scale >= config.gold.scale_min);
            assert!(p.scale <= config.gold.scale_max);
            assert!(p.size_class <= 2);
        }
    }

    #[test]
    fn test_deterministic() {
        let density = make_test_field(Material::Iron);
        let config = CrystalConfig::default();
        let a = compute_crystal_placements(&density, &config, glam::Vec3::ZERO, 42, 100);
        let b = compute_crystal_placements(&density, &config, glam::Vec3::ZERO, 42, 100);
        assert_eq!(a.len(), b.len());
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert_eq!(pa.x, pb.x);
            assert_eq!(pa.y, pb.y);
            assert_eq!(pa.z, pb.z);
            assert_eq!(pa.ore_type, pb.ore_type);
            assert_eq!(pa.scale, pb.scale);
        }
    }

    #[test]
    fn test_crystal_eligible() {
        assert!(is_crystal_eligible(Material::Iron));
        assert!(is_crystal_eligible(Material::Gold));
        assert!(is_crystal_eligible(Material::Coal));
        assert!(is_crystal_eligible(Material::Amethyst));
        assert!(!is_crystal_eligible(Material::Air));
        assert!(!is_crystal_eligible(Material::Limestone));
        assert!(!is_crystal_eligible(Material::Crystal));
    }

    #[test]
    fn test_ore_disabled_skips() {
        let density = make_test_field(Material::Iron);
        let mut config = CrystalConfig::default();
        config.iron.enabled = false;
        let placements = compute_crystal_placements(
            &density,
            &config,
            glam::Vec3::ZERO,
            42,
            100,
        );
        assert!(placements.is_empty());
    }

    #[test]
    fn test_normal_is_unit_length() {
        let density = make_test_field(Material::Copper);
        let mut config = CrystalConfig::default();
        config.copper.chance = 1.0;
        config.copper.density_threshold = 0.0;
        let placements = compute_crystal_placements(
            &density,
            &config,
            glam::Vec3::ZERO,
            42,
            100,
        );
        for p in &placements {
            let len = (p.normal_x * p.normal_x + p.normal_y * p.normal_y + p.normal_z * p.normal_z).sqrt();
            assert!((len - 1.0).abs() < 0.01, "Normal should be unit length, got {}", len);
        }
    }

    #[test]
    fn test_size_class_distribution() {
        // With equal weights, all three classes should appear over many rolls
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut counts = [0u32; 3];
        for _ in 0..1000 {
            let sc = select_size_class(&mut rng, 1.0, 1.0, 1.0);
            counts[sc as usize] += 1;
        }
        // Each should be roughly 333 +/- 50
        for (i, &c) in counts.iter().enumerate() {
            assert!(c > 250, "Size class {} count {} too low", i, c);
            assert!(c < 420, "Size class {} count {} too high", i, c);
        }
    }

    #[test]
    fn test_detect_ore_surfaces_interior_no_air() {
        let size = 5;
        let mut density = DensityField::new(size);
        // All solid iron — no air neighbors within scan range
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let s = density.get_mut(x, y, z);
                    s.density = 1.0;
                    s.material = Material::Iron;
                }
            }
        }
        let surfaces = detect_ore_surfaces(&density, size);
        assert!(surfaces.is_empty(), "Interior ore with no air should yield no surfaces");
    }
}
