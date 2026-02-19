use std::collections::HashMap;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use voxel_core::density::DensityField;
use voxel_core::material::Material;
use voxel_core::stress::world_to_chunk_local;
use crate::config::MineralConfig;
use crate::manifest::ChangeManifest;
use crate::TransformEntry;

/// Result of mineral growth pass.
#[derive(Debug, Default)]
pub struct MineralResult {
    pub minerals_grown: u32,
    pub manifest: ChangeManifest,
    pub glimpse_chunk: Option<(i32, i32, i32)>,
    /// Log of growth formations grouped by type
    pub transform_log: Vec<TransformEntry>,
}

/// 6-connected face-neighbor offsets.
const FACE_OFFSETS: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

/// A candidate mineral growth (collected in pass 1, applied in pass 2).
struct GrowthCandidate {
    chunk_key: (i32, i32, i32),
    lx: usize,
    ly: usize,
    lz: usize,
    old_material: Material,
    old_density: f32,
    new_material: Material,
    growth_type: GrowthType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum GrowthType {
    CrystalGrowth,
    MalachiteStalactite,
    QuartzExtension,
    CalciteInfill,
    PyriteCrust,
}

impl GrowthType {
    fn description(self) -> &'static str {
        match self {
            GrowthType::CrystalGrowth => "Crystal growth",
            GrowthType::MalachiteStalactite => "Malachite stalactite",
            GrowthType::QuartzExtension => "Quartz extension",
            GrowthType::CalciteInfill => "Calcite infill",
            GrowthType::PyriteCrust => "Pyrite crust",
        }
    }

    fn max_from_config(self, config: &MineralConfig) -> u32 {
        match self {
            GrowthType::CrystalGrowth => config.crystal_growth_max,
            GrowthType::MalachiteStalactite => config.malachite_stalactite_max,
            GrowthType::QuartzExtension => config.quartz_extension_max,
            GrowthType::CalciteInfill => config.calcite_infill_max,
            GrowthType::PyriteCrust => config.pyrite_crust_max,
        }
    }
}

/// Look up material at a world coordinate, returning None if the chunk is not loaded.
fn sample_material(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    wx: i32,
    wy: i32,
    wz: i32,
    chunk_size: usize,
) -> Option<Material> {
    let (chunk_key, lx, ly, lz) = world_to_chunk_local(wx, wy, wz, chunk_size);
    density_fields
        .get(&chunk_key)
        .map(|df| df.get(lx, ly, lz).material)
}

/// Count 6-connected neighbors matching a predicate.
fn count_neighbors_matching(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    wx: i32,
    wy: i32,
    wz: i32,
    chunk_size: usize,
    predicate: impl Fn(Material) -> bool,
) -> u32 {
    let mut count = 0u32;
    for &(dx, dy, dz) in &FACE_OFFSETS {
        if let Some(mat) = sample_material(density_fields, wx + dx, wy + dy, wz + dz, chunk_size) {
            if predicate(mat) {
                count += 1;
            }
        }
    }
    count
}

/// Check if any voxel within Manhattan distance <= radius has the given material.
fn has_material_within_radius(
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    wx: i32,
    wy: i32,
    wz: i32,
    chunk_size: usize,
    radius: i32,
    target: Material,
) -> bool {
    for dx in -radius..=radius {
        for dy in -radius..=radius {
            for dz in -radius..=radius {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                if dx.abs() + dy.abs() + dz.abs() > radius {
                    continue;
                }
                if let Some(mat) =
                    sample_material(density_fields, wx + dx, wy + dy, wz + dz, chunk_size)
                {
                    if mat == target {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Grow secondary minerals from solid into adjacent air.
pub fn apply_mineral_growth(
    config: &MineralConfig,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    chunks: &[(i32, i32, i32)],
    chunk_size: usize,
    rng: &mut ChaCha8Rng,
) -> MineralResult {
    let field_size = chunk_size + 1;

    // --- Pass 1: Scan air voxels for growth candidates ---
    let mut candidates: Vec<GrowthCandidate> = Vec::new();

    for &chunk_key in chunks {
        let (cx, cy, cz) = chunk_key;

        let df = match density_fields.get(&chunk_key) {
            Some(df) => df,
            None => continue,
        };

        for lz in 0..field_size {
            for ly in 0..field_size {
                for lx in 0..field_size {
                    let sample = df.get(lx, ly, lz);

                    // Mineral growth targets AIR voxels
                    if sample.material != Material::Air {
                        continue;
                    }

                    let old_density = sample.density;

                    // World coordinates
                    let wx = cx * (chunk_size as i32) + lx as i32;
                    let wy = cy * (chunk_size as i32) + ly as i32;
                    let wz = cz * (chunk_size as i32) + lz as i32;

                    // --- Crystal growth ---
                    // Air voxel with 2+ Crystal/Amethyst neighbors
                    {
                        let crystal_neighbors = count_neighbors_matching(
                            density_fields,
                            wx, wy, wz,
                            chunk_size,
                            |m| m == Material::Crystal || m == Material::Amethyst,
                        );
                        if crystal_neighbors >= 2 {
                            candidates.push(GrowthCandidate {
                                chunk_key,
                                lx, ly, lz,
                                old_material: Material::Air,
                                old_density,
                                new_material: Material::Crystal,
                                growth_type: GrowthType::CrystalGrowth,
                            });
                        }
                    }

                    // --- Malachite stalactite ---
                    // Air voxel where (x, y+1, z) is Copper AND Limestone within 2 Manhattan distance
                    {
                        let above_mat =
                            sample_material(density_fields, wx, wy + 1, wz, chunk_size);
                        if above_mat == Some(Material::Copper) {
                            let has_limestone = has_material_within_radius(
                                density_fields,
                                wx, wy, wz,
                                chunk_size,
                                2,
                                Material::Limestone,
                            );
                            if has_limestone {
                                candidates.push(GrowthCandidate {
                                    chunk_key,
                                    lx, ly, lz,
                                    old_material: Material::Air,
                                    old_density,
                                    new_material: Material::Malachite,
                                    growth_type: GrowthType::MalachiteStalactite,
                                });
                            }
                        }
                    }

                    // --- Quartz extension ---
                    // Air voxel at quartz vein tip: exactly 1 Quartz neighbor, 5 non-Quartz
                    {
                        let quartz_neighbors = count_neighbors_matching(
                            density_fields,
                            wx, wy, wz,
                            chunk_size,
                            |m| m == Material::Quartz,
                        );
                        if quartz_neighbors == 1 {
                            // Check probability
                            if rng.gen::<f32>() < config.quartz_extension_prob {
                                candidates.push(GrowthCandidate {
                                    chunk_key,
                                    lx, ly, lz,
                                    old_material: Material::Air,
                                    old_density,
                                    new_material: Material::Quartz,
                                    growth_type: GrowthType::QuartzExtension,
                                });
                            }
                        }
                    }

                    // --- Calcite infill ---
                    // Air with 3+ (calcite_infill_min_faces) Limestone 6-connected neighbors
                    // AND world Y < calcite_infill_depth
                    {
                        if (wy as f32) < config.calcite_infill_depth {
                            let limestone_neighbors = count_neighbors_matching(
                                density_fields,
                                wx, wy, wz,
                                chunk_size,
                                |m| m == Material::Limestone,
                            );
                            if limestone_neighbors >= config.calcite_infill_min_faces {
                                candidates.push(GrowthCandidate {
                                    chunk_key,
                                    lx, ly, lz,
                                    old_material: Material::Air,
                                    old_density,
                                    new_material: Material::Limestone,
                                    growth_type: GrowthType::CalciteInfill,
                                });
                            }
                        }
                    }

                    // --- Pyrite crust ---
                    // Air adjacent to Pyrite, where the Pyrite source has pyrite_crust_min_solid solid neighbors
                    {
                        // Check each face neighbor for Pyrite
                        let mut found_pyrite_source = false;
                        for &(dx, dy, dz) in &FACE_OFFSETS {
                            let nx = wx + dx;
                            let ny = wy + dy;
                            let nz = wz + dz;
                            if let Some(mat) =
                                sample_material(density_fields, nx, ny, nz, chunk_size)
                            {
                                if mat == Material::Pyrite {
                                    // Check that this Pyrite source has enough solid neighbors
                                    let solid_count = count_neighbors_matching(
                                        density_fields,
                                        nx, ny, nz,
                                        chunk_size,
                                        |m| m.is_solid(),
                                    );
                                    if solid_count >= config.pyrite_crust_min_solid {
                                        found_pyrite_source = true;
                                        break;
                                    }
                                }
                            }
                        }
                        if found_pyrite_source {
                            candidates.push(GrowthCandidate {
                                chunk_key,
                                lx, ly, lz,
                                old_material: Material::Air,
                                old_density,
                                new_material: Material::Pyrite,
                                growth_type: GrowthType::PyriteCrust,
                            });
                        }
                    }
                }
            }
        }
    }

    // --- Pass 2: Apply candidates, respecting per-type max caps ---
    let mut result = MineralResult::default();
    let mut type_counts: HashMap<GrowthType, u32> = HashMap::new();

    for candidate in &candidates {
        let max = candidate.growth_type.max_from_config(config);
        let current = *type_counts.get(&candidate.growth_type).unwrap_or(&0);
        if current >= max {
            continue;
        }

        // Generate growth density
        let new_density = rng.gen_range(config.growth_density_min..=config.growth_density_max);

        // Apply the growth
        if let Some(df) = density_fields.get_mut(&candidate.chunk_key) {
            let sample = df.get_mut(candidate.lx, candidate.ly, candidate.lz);
            sample.material = candidate.new_material;
            sample.density = new_density;
        }

        // Record in manifest
        result.manifest.record_voxel_change(
            candidate.chunk_key,
            candidate.lx,
            candidate.ly,
            candidate.lz,
            candidate.old_material,
            candidate.old_density,
            candidate.new_material,
            new_density,
        );

        // Set glimpse_chunk to the first growth
        if result.glimpse_chunk.is_none() {
            result.glimpse_chunk = Some(candidate.chunk_key);
        }

        *type_counts.entry(candidate.growth_type).or_insert(0) += 1;
        result.minerals_grown += 1;
    }

    // Build transform log
    for (growth_type, count) in &type_counts {
        result.transform_log.push(TransformEntry {
            description: growth_type.description().to_string(),
            count: *count,
        });
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha8Rng;
    use rand::SeedableRng;
    use voxel_core::density::DensityField;

    const CHUNK_SIZE: usize = 16;
    const FIELD_SIZE: usize = CHUNK_SIZE + 1; // 17

    /// Create a density field filled entirely with the given material and density.
    fn make_filled_field(material: Material, density: f32) -> DensityField {
        let mut df = DensityField::new(FIELD_SIZE);
        for s in df.samples.iter_mut() {
            s.material = material;
            s.density = density;
        }
        df
    }

    #[test]
    fn test_crystal_growth() {
        let chunk_key = (0, 0, 0);
        let mut density_fields: HashMap<(i32, i32, i32), DensityField> = HashMap::new();

        // Fill with solid Crystal, then carve a single air voxel at (8, 8, 8)
        // which will have 6 Crystal neighbors
        let mut df = make_filled_field(Material::Crystal, 1.0);
        {
            let s = df.get_mut(8, 8, 8);
            s.material = Material::Air;
            s.density = -1.0;
        }
        density_fields.insert(chunk_key, df);

        let chunks = vec![chunk_key];
        let config = MineralConfig {
            crystal_growth_max: 10,
            ..MineralConfig::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result =
            apply_mineral_growth(&config, &mut density_fields, &chunks, CHUNK_SIZE, &mut rng);

        assert!(
            result.minerals_grown > 0,
            "Expected air voxel with Crystal neighbors to grow Crystal"
        );

        let df = density_fields.get(&chunk_key).unwrap();
        assert_eq!(
            df.get(8, 8, 8).material,
            Material::Crystal,
            "Air voxel surrounded by Crystal should become Crystal"
        );
    }

    #[test]
    fn test_calcite_infill() {
        // Chunk at cy = -3 (world Y from -48 to -32), some voxels below -30
        let chunk_key = (0, -3, 0);
        let mut density_fields: HashMap<(i32, i32, i32), DensityField> = HashMap::new();

        // Fill with Limestone, carve one air voxel deep inside at (8, 2, 8)
        // World Y = -3 * 16 + 2 = -46, which is < -30
        // It has 6 Limestone neighbors -> 6 >= 3 min faces
        let mut df = make_filled_field(Material::Limestone, 1.0);
        {
            let s = df.get_mut(8, 2, 8);
            s.material = Material::Air;
            s.density = -1.0;
        }
        density_fields.insert(chunk_key, df);

        let chunks = vec![chunk_key];
        let config = MineralConfig {
            calcite_infill_max: 10,
            ..MineralConfig::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result =
            apply_mineral_growth(&config, &mut density_fields, &chunks, CHUNK_SIZE, &mut rng);

        assert!(
            result.minerals_grown > 0,
            "Expected calcite infill to grow"
        );

        let df = density_fields.get(&chunk_key).unwrap();
        assert_eq!(
            df.get(8, 2, 8).material,
            Material::Limestone,
            "Air surrounded by Limestone at depth should become Limestone (calcite infill)"
        );
    }

    #[test]
    fn test_no_growth_without_conditions() {
        // All solid, no air -- nothing can grow
        let chunk_key = (0, 0, 0);
        let mut density_fields: HashMap<(i32, i32, i32), DensityField> = HashMap::new();
        density_fields.insert(chunk_key, make_filled_field(Material::Granite, 1.0));

        let chunks = vec![chunk_key];
        let config = MineralConfig::default();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result =
            apply_mineral_growth(&config, &mut density_fields, &chunks, CHUNK_SIZE, &mut rng);

        assert_eq!(
            result.minerals_grown, 0,
            "No minerals should grow when there is no air"
        );
    }

    #[test]
    fn test_growth_density_range() {
        let chunk_key = (0, 0, 0);
        let mut density_fields: HashMap<(i32, i32, i32), DensityField> = HashMap::new();

        // Create a field with Crystal and a few air pockets to trigger crystal growth
        let mut df = make_filled_field(Material::Crystal, 1.0);
        // Make several air voxels interior (each surrounded by Crystal)
        let air_positions = vec![(4, 4, 4), (6, 6, 6), (8, 8, 8), (10, 10, 10)];
        for &(ax, ay, az) in &air_positions {
            let s = df.get_mut(ax, ay, az);
            s.material = Material::Air;
            s.density = -1.0;
        }
        density_fields.insert(chunk_key, df);

        let chunks = vec![chunk_key];
        let config = MineralConfig {
            crystal_growth_max: 100,
            growth_density_min: 0.3,
            growth_density_max: 0.6,
            ..MineralConfig::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result =
            apply_mineral_growth(&config, &mut density_fields, &chunks, CHUNK_SIZE, &mut rng);

        assert!(
            result.minerals_grown > 0,
            "Expected some crystal growths"
        );

        // Check that all grown voxels have density in the expected range
        let df = density_fields.get(&chunk_key).unwrap();
        for &(ax, ay, az) in &air_positions {
            let s = df.get(ax, ay, az);
            if s.material == Material::Crystal {
                assert!(
                    s.density >= 0.3 && s.density <= 0.6,
                    "Growth density {} should be in [0.3, 0.6]",
                    s.density
                );
            }
        }
    }
}
