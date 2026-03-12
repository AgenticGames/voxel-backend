use std::collections::HashMap;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use voxel_core::density::DensityField;
use voxel_core::material::Material;
use crate::config::MetamorphismConfig;
use crate::manifest::ChangeManifest;
use crate::util::{set_voxel_synced, count_neighbors, has_material_within_radius};
use crate::TransformEntry;

/// Result of metamorphism pass on a set of chunks.
#[derive(Debug, Default)]
pub struct MetamorphismResult {
    pub voxels_transformed: u32,
    pub manifest: ChangeManifest,
    /// Chunk where an interesting transform happened (for montage)
    pub glimpse_chunk: Option<(i32, i32, i32)>,
    /// Log of transformations grouped by type
    pub transform_log: Vec<TransformEntry>,
}

/// A candidate metamorphic transformation (collected in pass 1, applied in pass 2).
struct Candidate {
    chunk_key: (i32, i32, i32),
    lx: usize,
    ly: usize,
    lz: usize,
    old_material: Material,
    new_material: Material,
    density: f32,
    description: &'static str,
}

/// Run metamorphism transforms on the given chunks.
/// Two-pass: scan candidates first, then apply (prevents cascade within single sleep).
pub fn apply_metamorphism(
    config: &MetamorphismConfig,
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    chunks: &[(i32, i32, i32)],
    chunk_size: usize,
    rng: &mut ChaCha8Rng,
) -> MetamorphismResult {
    let field_size = chunk_size + 1;

    // --- Pass 1: Scan candidates ---
    let mut candidates: Vec<Candidate> = Vec::new();

    for &chunk_key in chunks {
        let (cx, cy, cz) = chunk_key;

        // We need immutable access to density_fields here
        let df = match density_fields.get(&chunk_key) {
            Some(df) => df,
            None => continue,
        };

        for lz in 0..field_size {
            for ly in 0..field_size {
                for lx in 0..field_size {
                    let sample = df.get(lx, ly, lz);
                    let material = sample.material;
                    let density = sample.density;

                    // Only transform solid voxels
                    if !material.is_solid() {
                        continue;
                    }

                    // World coordinates for this voxel
                    let wx = cx * (chunk_size as i32) + lx as i32;
                    let wy = cy * (chunk_size as i32) + ly as i32;
                    let wz = cz * (chunk_size as i32) + lz as i32;

                    match material {
                        Material::Limestone if config.limestone_to_marble_enabled => {
                            // Limestone -> Marble: deep OR adjacent to Basalt/Kimberlite
                            let deep = (wy as f32) < config.limestone_to_marble_depth;
                            let has_hot_neighbor = if !deep {
                                count_neighbors(
                                    density_fields,
                                    wx, wy, wz,
                                    chunk_size,
                                    |m| m == Material::Basalt || m == Material::Kimberlite,
                                ) > 0
                            } else {
                                true // already qualifies by depth
                            };

                            if (deep || has_hot_neighbor)
                                && rng.gen::<f32>() < config.limestone_to_marble_prob
                            {
                                candidates.push(Candidate {
                                    chunk_key,
                                    lx, ly, lz,
                                    old_material: material,
                                    new_material: Material::Marble,
                                    density,
                                    description: "Limestone \u{2192} Marble",
                                });
                            }
                        }
                        Material::Sandstone if config.sandstone_to_granite_enabled => {
                            // Sandstone -> Granite: deep AND 4+ solid neighbors
                            let deep = (wy as f32) < config.sandstone_to_granite_depth;
                            if deep {
                                let solid_count = count_neighbors(
                                    density_fields,
                                    wx, wy, wz,
                                    chunk_size,
                                    |m| m.is_solid(),
                                );
                                if solid_count >= config.sandstone_to_granite_min_neighbors
                                    && rng.gen::<f32>() < config.sandstone_to_granite_prob
                                {
                                    candidates.push(Candidate {
                                        chunk_key,
                                        lx, ly, lz,
                                        old_material: material,
                                        new_material: Material::Granite,
                                        density,
                                        description: "Sandstone \u{2192} Granite",
                                    });
                                }
                            }
                        }
                        Material::Slate if config.slate_to_marble_enabled => {
                            // Slate -> Marble: has adjacent Kimberlite neighbor
                            let has_kimberlite = count_neighbors(
                                density_fields,
                                wx, wy, wz,
                                chunk_size,
                                |m| m == Material::Kimberlite,
                            ) > 0;
                            if has_kimberlite
                                && rng.gen::<f32>() < config.slate_to_marble_prob
                            {
                                candidates.push(Candidate {
                                    chunk_key,
                                    lx, ly, lz,
                                    old_material: material,
                                    new_material: Material::Marble,
                                    density,
                                    description: "Slate \u{2192} Marble",
                                });
                            }
                        }
                        Material::Granite if config.granite_to_basalt_enabled => {
                            // Granite -> Basalt: has 2+ adjacent air voxels
                            let air_count = count_neighbors(
                                density_fields,
                                wx, wy, wz,
                                chunk_size,
                                |m| !m.is_solid(),
                            );
                            if air_count >= config.granite_to_basalt_min_air
                                && rng.gen::<f32>() < config.granite_to_basalt_prob
                            {
                                candidates.push(Candidate {
                                    chunk_key,
                                    lx, ly, lz,
                                    old_material: material,
                                    new_material: Material::Basalt,
                                    density,
                                    description: "Granite \u{2192} Basalt",
                                });
                            }
                        }
                        Material::Iron if config.iron_to_pyrite_enabled => {
                            // Iron -> Pyrite: has Sulfide within search radius
                            let radius = config.iron_to_pyrite_search_radius as i32;
                            let has_sulfide = has_material_within_radius(
                                density_fields,
                                wx, wy, wz,
                                chunk_size,
                                radius,
                                Material::Sulfide,
                            );
                            if has_sulfide
                                && rng.gen::<f32>() < config.iron_to_pyrite_prob
                            {
                                candidates.push(Candidate {
                                    chunk_key,
                                    lx, ly, lz,
                                    old_material: material,
                                    new_material: Material::Pyrite,
                                    density,
                                    description: "Iron \u{2192} Pyrite",
                                });
                            }
                        }
                        Material::Copper if config.copper_to_malachite_enabled => {
                            // Copper -> Malachite: has 1+ adjacent air voxel
                            let air_count = count_neighbors(
                                density_fields,
                                wx, wy, wz,
                                chunk_size,
                                |m| !m.is_solid(),
                            );
                            if air_count >= 1
                                && rng.gen::<f32>() < config.copper_to_malachite_prob
                            {
                                candidates.push(Candidate {
                                    chunk_key,
                                    lx, ly, lz,
                                    old_material: material,
                                    new_material: Material::Malachite,
                                    density,
                                    description: "Copper \u{2192} Malachite",
                                });
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // --- Pass 2: Apply candidates ---
    let mut result = MetamorphismResult::default();
    let mut counts: HashMap<&str, u32> = HashMap::new();

    for candidate in &candidates {
        // Apply the material change (synced to boundary neighbors)
        set_voxel_synced(density_fields, candidate.chunk_key, candidate.lx, candidate.ly, candidate.lz, candidate.new_material, None, chunk_size);

        // Record in manifest
        result.manifest.record_voxel_change(
            candidate.chunk_key,
            candidate.lx,
            candidate.ly,
            candidate.lz,
            candidate.old_material,
            candidate.density,
            candidate.new_material,
            candidate.density,
        );

        // Set glimpse_chunk to the first transform
        if result.glimpse_chunk.is_none() {
            result.glimpse_chunk = Some(candidate.chunk_key);
        }

        *counts.entry(candidate.description).or_insert(0) += 1;
        result.voxels_transformed += 1;
    }

    // Build transform log
    for (desc, count) in &counts {
        result.transform_log.push(TransformEntry {
            description: desc.to_string(),
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
    fn test_limestone_to_marble_deep() {
        // Place a chunk at cy = -4 (world Y from -64 to -48), well below -50
        let chunk_key = (0, -4, 0);
        let mut density_fields: HashMap<(i32, i32, i32), DensityField> = HashMap::new();
        density_fields.insert(chunk_key, make_filled_field(Material::Limestone, 1.0));

        let chunks = vec![chunk_key];
        let config = MetamorphismConfig {
            limestone_to_marble_prob: 1.0, // guarantee transforms for testing
            ..MetamorphismConfig::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = apply_metamorphism(&config, &mut density_fields, &chunks, CHUNK_SIZE, &mut rng);

        assert!(
            result.voxels_transformed > 0,
            "Expected some Limestone to transform to Marble at deep Y"
        );

        // Check that at least some voxels became Marble
        let df = density_fields.get(&chunk_key).unwrap();
        let mut marble_count = 0u32;
        for s in df.samples.iter() {
            if s.material == Material::Marble {
                marble_count += 1;
            }
        }
        assert!(
            marble_count > 0,
            "Expected some Marble in the field after metamorphism"
        );
    }

    #[test]
    fn test_copper_to_malachite_with_air() {
        let chunk_key = (0, 0, 0);
        let mut density_fields: HashMap<(i32, i32, i32), DensityField> = HashMap::new();

        // Fill with solid Copper, but leave a strip of Air along one edge
        let mut df = make_filled_field(Material::Copper, 1.0);
        // Set x=0 face to Air to give adjacent Copper voxels an air neighbor
        for z in 0..FIELD_SIZE {
            for y in 0..FIELD_SIZE {
                let s = df.get_mut(0, y, z);
                s.material = Material::Air;
                s.density = -1.0;
            }
        }
        density_fields.insert(chunk_key, df);

        let chunks = vec![chunk_key];
        let config = MetamorphismConfig {
            copper_to_malachite_prob: 1.0, // guarantee
            ..MetamorphismConfig::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = apply_metamorphism(&config, &mut density_fields, &chunks, CHUNK_SIZE, &mut rng);

        // The copper voxels at x=1 are adjacent to air at x=0
        assert!(
            result.voxels_transformed > 0,
            "Expected some Copper to transform to Malachite"
        );

        let df = density_fields.get(&chunk_key).unwrap();
        // x=1 copper should now be malachite (adjacent to air at x=0)
        let mat = df.get(1, 1, 1).material;
        assert_eq!(
            mat,
            Material::Malachite,
            "Copper adjacent to air should become Malachite"
        );
    }

    #[test]
    fn test_no_transform_without_conditions() {
        // All solid Sandstone at Y=0 (shallow) -- should NOT become Granite
        let chunk_key = (0, 0, 0);
        let mut density_fields: HashMap<(i32, i32, i32), DensityField> = HashMap::new();
        density_fields.insert(chunk_key, make_filled_field(Material::Sandstone, 1.0));

        let chunks = vec![chunk_key];
        let config = MetamorphismConfig::default(); // depth threshold is -100
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = apply_metamorphism(&config, &mut density_fields, &chunks, CHUNK_SIZE, &mut rng);

        assert_eq!(
            result.voxels_transformed, 0,
            "No transforms should occur for shallow Sandstone"
        );
    }

    #[test]
    fn test_deterministic() {
        let chunk_key = (0, -4, 0);
        let config = MetamorphismConfig::default();
        let chunks = vec![chunk_key];

        // Run 1
        let mut density_fields_1: HashMap<(i32, i32, i32), DensityField> = HashMap::new();
        density_fields_1.insert(chunk_key, make_filled_field(Material::Limestone, 1.0));
        let mut rng_1 = ChaCha8Rng::seed_from_u64(42);
        let result_1 =
            apply_metamorphism(&config, &mut density_fields_1, &chunks, CHUNK_SIZE, &mut rng_1);

        // Run 2 (same seed)
        let mut density_fields_2: HashMap<(i32, i32, i32), DensityField> = HashMap::new();
        density_fields_2.insert(chunk_key, make_filled_field(Material::Limestone, 1.0));
        let mut rng_2 = ChaCha8Rng::seed_from_u64(42);
        let result_2 =
            apply_metamorphism(&config, &mut density_fields_2, &chunks, CHUNK_SIZE, &mut rng_2);

        assert_eq!(
            result_1.voxels_transformed, result_2.voxels_transformed,
            "Same seed should produce same number of transforms"
        );

        // Check voxel-by-voxel equality
        let df1 = density_fields_1.get(&chunk_key).unwrap();
        let df2 = density_fields_2.get(&chunk_key).unwrap();
        for i in 0..df1.samples.len() {
            assert_eq!(
                df1.samples[i].material, df2.samples[i].material,
                "Same seed should produce identical results at sample {}", i
            );
        }
    }
}
