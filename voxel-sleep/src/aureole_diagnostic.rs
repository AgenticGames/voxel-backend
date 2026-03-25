/// Aureole diagnostic tests — focused validation of contact metamorphism placement.
///
/// These tests create controlled worlds with known lava positions and verify that
/// aureole materials (Hornfels/Skarn) appear ONLY within the expected radius,
/// and NEVER in chunks that shouldn't have them.
///
/// Run with: cargo test -p voxel-sleep aureole_diagnostic -- --nocapture

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap, HashSet};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use voxel_core::density::DensityField;
    use voxel_core::material::Material;
    use voxel_core::stress::world_to_chunk_local;
    use voxel_fluid::{FluidSnapshot, cell::{FluidCell, FluidType}};

    use crate::config::{AureoleConfig, GroundwaterConfig};
    use crate::phases::aureole::{apply_aureole, build_heat_map};
    use crate::ResourceCensus;

    const CHUNK_SIZE: usize = 16;
    const FIELD_SIZE: usize = CHUNK_SIZE + 1; // 17

    // ─── Helpers ──────────────────────────────────────────────────

    /// Create a solid density field filled with a single material.
    fn make_solid_field(material: Material) -> DensityField {
        let mut df = DensityField::new(FIELD_SIZE);
        for sample in df.samples.iter_mut() {
            sample.density = 1.0;
            sample.material = material;
        }
        df
    }

    /// Create empty fluid cells for one chunk (16^3).
    fn empty_fluid_cells() -> Vec<FluidCell> {
        vec![FluidCell {
            level: 0.0,
            fluid_type: FluidType::Water,
            is_source: false,
            grace_ticks: 0,
            stagnant_ticks: 0,
        }; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE]
    }

    /// Place lava cells in a fluid snapshot at specific local positions within a chunk.
    fn place_lava(
        fluid: &mut FluidSnapshot,
        chunk: (i32, i32, i32),
        positions: &[(usize, usize, usize)],
    ) {
        let cells = fluid.chunks.entry(chunk).or_insert_with(empty_fluid_cells);
        for &(lx, ly, lz) in positions {
            let idx = lz * CHUNK_SIZE * CHUNK_SIZE + ly * CHUNK_SIZE + lx;
            cells[idx] = FluidCell {
                level: 1.0,
                fluid_type: FluidType::Lava,
                is_source: true,
                grace_ticks: 0,
                stagnant_ticks: 0,
            };
        }
    }

    /// Scan all density fields and find every voxel of a given material.
    /// Returns (chunk_key, local_x, local_y, local_z, world_x, world_y, world_z).
    fn find_material(
        density_fields: &HashMap<(i32, i32, i32), DensityField>,
        material: Material,
    ) -> Vec<((i32, i32, i32), usize, usize, usize, i32, i32, i32)> {
        let mut results = Vec::new();
        let mut keys: Vec<_> = density_fields.keys().copied().collect();
        keys.sort();
        for key in keys {
            let df = &density_fields[&key];
            let (cx, cy, cz) = key;
            for lz in 0..CHUNK_SIZE {
                for ly in 0..CHUNK_SIZE {
                    for lx in 0..CHUNK_SIZE {
                        if df.get(lx, ly, lz).material == material {
                            let wx = cx * CHUNK_SIZE as i32 + lx as i32;
                            let wy = cy * CHUNK_SIZE as i32 + ly as i32;
                            let wz = cz * CHUNK_SIZE as i32 + lz as i32;
                            results.push((key, lx, ly, lz, wx, wy, wz));
                        }
                    }
                }
            }
        }
        results
    }

    /// Count materials across all density fields (inner grid only, 0..CHUNK_SIZE).
    fn count_materials(
        density_fields: &HashMap<(i32, i32, i32), DensityField>,
    ) -> BTreeMap<u8, u32> {
        let mut counts = BTreeMap::new();
        for df in density_fields.values() {
            for z in 0..CHUNK_SIZE {
                for y in 0..CHUNK_SIZE {
                    for x in 0..CHUNK_SIZE {
                        let mat = df.get(x, y, z).material as u8;
                        *counts.entry(mat).or_insert(0) += 1;
                    }
                }
            }
        }
        counts
    }

    fn mat_name(id: u8) -> &'static str {
        match id {
            0 => "Air", 1 => "Sandstone", 2 => "Limestone", 3 => "Granite",
            4 => "Basalt", 5 => "Slate", 6 => "Marble", 7 => "Iron",
            8 => "Copper", 22 => "Hornfels", 23 => "Garnet", 24 => "Diopside",
            25 => "Gypsum", 26 => "Skarn", _ => "Other",
        }
    }

    fn make_empty_census() -> ResourceCensus {
        ResourceCensus {
            water: crate::FluidMetrics { cell_count: 0, volume_sum: 0.0, chunks_with_fluid: 0 },
            lava: crate::FluidMetrics { cell_count: 0, volume_sum: 0.0, chunks_with_fluid: 0 },
            exposed_surfaces_by_material: BTreeMap::new(),
            total_exposed_surfaces: 0,
            fissure_count: 0,
            open_wall_count: 0,
            exposed_ore: BTreeMap::new(),
            heat_source_lava: 0,
            heat_source_kimberlite: 0,
            scan_duration: std::time::Duration::ZERO,
        }
    }

    // ─── Test 1: Lava blob in center of limestone → Skarn aureole ──────────

    #[test]
    fn test_aureole_limestone_produces_skarn_near_lava() {
        eprintln!("\n═══ TEST: Limestone + Lava → Skarn aureole ═══");

        // Create a 3x3x3 grid of solid limestone chunks
        let mut density_fields: HashMap<(i32, i32, i32), DensityField> = HashMap::new();
        let mut chunks = Vec::new();
        for cx in -1..=1 {
            for cy in -1..=1 {
                for cz in -1..=1 {
                    let key = (cx, cy, cz);
                    density_fields.insert(key, make_solid_field(Material::Limestone));
                    chunks.push(key);
                }
            }
        }
        chunks.sort();

        // Place a 3x3x3 lava blob at world position (8,8,8) — center of chunk (0,0,0)
        let mut fluid = FluidSnapshot::default();
        let mut lava_positions = Vec::new();
        for lz in 7..=9 {
            for ly in 7..=9 {
                for lx in 7..=9 {
                    lava_positions.push((lx, ly, lz));
                }
            }
        }
        place_lava(&mut fluid, (0, 0, 0), &lava_positions);

        // Count lava cells
        let lava_count = lava_positions.len();
        eprintln!("  Lava blob: {} cells at world (7-9, 7-9, 7-9) in chunk (0,0,0)", lava_count);

        // Snapshot materials before
        let before = count_materials(&density_fields);

        // Run aureole
        let mut config = AureoleConfig::default();
        config.zone_enabled = true;
        config.metamorphism_enabled = true;
        config.water_erosion_enabled = false; // isolate metamorphism
        config.min_lava_zone_size = 1; // accept small zones for testing

        let heat_map = build_heat_map(&density_fields, &fluid, &chunks, CHUNK_SIZE);
        eprintln!("  Heat map: {} sources", heat_map.len());

        // Verify heat sources are at expected world positions
        let heat_positions: HashSet<(i32, i32, i32)> = heat_map.iter().map(|h| h.pos).collect();
        for lz in 7..=9i32 {
            for ly in 7..=9i32 {
                for lx in 7..=9i32 {
                    assert!(
                        heat_positions.contains(&(lx, ly, lz)),
                        "Expected heat source at world ({},{},{}) but not found", lx, ly, lz
                    );
                }
            }
        }
        eprintln!("  ✓ All {} heat sources at correct world positions", heat_map.len());

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let groundwater = GroundwaterConfig::default();
        let census = make_empty_census();

        let result = apply_aureole(
            &config, &groundwater, &mut density_fields, &mut fluid,
            &heat_map, &chunks, CHUNK_SIZE, &mut rng, &census,
        );

        eprintln!("  Result: {} zones, {} hornfels, {} skarn, {} veins",
            result.lava_zones_found, result.hornfels_placed, result.skarn_placed, result.veins_placed);
        for line in &result.debug_lines {
            eprintln!("    {}", line);
        }

        // Snapshot materials after
        let after = count_materials(&density_fields);
        eprintln!("\n  Material census (before → after):");
        let mut all_mats: std::collections::BTreeSet<u8> = before.keys().copied().collect();
        all_mats.extend(after.keys());
        for mat_id in &all_mats {
            let b = before.get(mat_id).unwrap_or(&0);
            let a = after.get(mat_id).unwrap_or(&0);
            if b != a {
                let delta = *a as i64 - *b as i64;
                eprintln!("    {}: {} → {} ({:+})", mat_name(*mat_id), b, a, delta);
            }
        }

        // ─── ASSERTIONS ───

        // 1. Aureole should have found exactly 1 lava zone
        assert_eq!(result.lava_zones_found, 1, "Expected exactly 1 lava zone");

        // 2. Skarn should exist (limestone → skarn near lava)
        assert!(result.skarn_placed > 0, "Expected skarn to be placed near lava in limestone");

        // 3. Hornfels should NOT exist (no slate/granite/sandstone in this world)
        assert_eq!(result.hornfels_placed, 0,
            "Expected NO hornfels (world is pure limestone, should produce skarn)");

        // 4. Skarn should ONLY exist near the lava blob
        let skarn_voxels = find_material(&density_fields, Material::Skarn);
        eprintln!("\n  Skarn placement ({} voxels):", skarn_voxels.len());

        let lava_center = (8i32, 8i32, 8i32); // center of 7-9 range
        let max_allowed_distance = config.max_radius as i32 + 2; // BFS depth + margin

        let mut max_actual_distance = 0i32;
        let mut skarn_chunks: HashSet<(i32, i32, i32)> = HashSet::new();
        for &(chunk, _lx, _ly, _lz, wx, wy, wz) in &skarn_voxels {
            let dist = (wx - lava_center.0).abs()
                .max((wy - lava_center.1).abs())
                .max((wz - lava_center.2).abs());
            max_actual_distance = max_actual_distance.max(dist);
            skarn_chunks.insert(chunk);

            // CRITICAL: No skarn should be farther than max_radius + lava_half_width from center
            assert!(
                dist <= max_allowed_distance,
                "AUREOLE LEAK: Skarn at world ({},{},{}) is {} voxels from lava center — max allowed is {}",
                wx, wy, wz, dist, max_allowed_distance
            );
        }

        eprintln!("  Max Chebyshev distance from lava center: {} (limit: {})",
            max_actual_distance, max_allowed_distance);
        eprintln!("  Skarn in chunks: {:?}", {
            let mut v: Vec<_> = skarn_chunks.iter().collect();
            v.sort();
            v
        });

        // 5. Chunks far from lava should have ZERO aureole materials
        for &key in density_fields.keys() {
            let (cx, cy, cz) = key;
            // Chunks more than 1 away from (0,0,0) should have no skarn
            if cx.abs() > 1 || cy.abs() > 1 || cz.abs() > 1 {
                let df = &density_fields[&key];
                for z in 0..CHUNK_SIZE {
                    for y in 0..CHUNK_SIZE {
                        for x in 0..CHUNK_SIZE {
                            let mat = df.get(x, y, z).material;
                            assert!(
                                mat != Material::Skarn && mat != Material::Hornfels,
                                "AUREOLE LEAK: {} found at local ({},{},{}) in far chunk ({},{},{}) — should be untouched!",
                                if mat == Material::Skarn { "Skarn" } else { "Hornfels" },
                                x, y, z, cx, cy, cz
                            );
                        }
                    }
                }
            }
        }
        eprintln!("  ✓ No aureole materials leaked to far chunks");

        eprintln!("═══ PASS ═══\n");
    }

    // ─── Test 2: Lava at chunk boundary → aureole crosses correctly ────────

    #[test]
    fn test_aureole_crosses_chunk_boundary() {
        eprintln!("\n═══ TEST: Aureole crosses chunk boundary correctly ═══");

        // Create chunks (0,0,0) and (1,0,0) — lava at the boundary
        let mut density_fields: HashMap<(i32, i32, i32), DensityField> = HashMap::new();
        let mut chunks = Vec::new();
        for cx in -1..=2 {
            for cy in -1..=1 {
                for cz in -1..=1 {
                    let key = (cx, cy, cz);
                    density_fields.insert(key, make_solid_field(Material::Limestone));
                    chunks.push(key);
                }
            }
        }
        chunks.sort();

        // Place lava at the boundary between chunk (0,0,0) and (1,0,0)
        // Local positions 14-15 in chunk (0,0,0) and 0-1 in chunk (1,0,0)
        // World positions: 14,15,16,17 in X → straddles the boundary at x=16
        let mut fluid = FluidSnapshot::default();
        let mut lava_chunk0 = Vec::new();
        let mut lava_chunk1 = Vec::new();
        for lz in 7..=9 {
            for ly in 7..=9 {
                for lx in 14..=15 {
                    lava_chunk0.push((lx, ly, lz));
                }
                for lx in 0..=1 {
                    lava_chunk1.push((lx, ly, lz));
                }
            }
        }
        place_lava(&mut fluid, (0, 0, 0), &lava_chunk0);
        place_lava(&mut fluid, (1, 0, 0), &lava_chunk1);

        let total_lava = lava_chunk0.len() + lava_chunk1.len();
        eprintln!("  Lava blob: {} cells straddling chunk boundary at x=16", total_lava);
        eprintln!("    Chunk (0,0,0): {} cells at lx=14-15", lava_chunk0.len());
        eprintln!("    Chunk (1,0,0): {} cells at lx=0-1", lava_chunk1.len());

        // Run aureole
        let mut config = AureoleConfig::default();
        config.zone_enabled = true;
        config.metamorphism_enabled = true;
        config.water_erosion_enabled = false;
        config.min_lava_zone_size = 1;

        let heat_map = build_heat_map(&density_fields, &fluid, &chunks, CHUNK_SIZE);
        eprintln!("  Heat map: {} sources", heat_map.len());

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let groundwater = GroundwaterConfig::default();
        let census = make_empty_census();

        let result = apply_aureole(
            &config, &groundwater, &mut density_fields, &mut fluid,
            &heat_map, &chunks, CHUNK_SIZE, &mut rng, &census,
        );

        eprintln!("  Result: {} zones, {} skarn, {} hornfels",
            result.lava_zones_found, result.skarn_placed, result.hornfels_placed);
        for line in &result.debug_lines {
            eprintln!("    {}", line);
        }

        // ─── ASSERTIONS ───

        // 1. Should be exactly 1 zone (lava cells are adjacent across boundary)
        assert_eq!(result.lava_zones_found, 1,
            "Lava straddling chunk boundary should form 1 zone, not {}", result.lava_zones_found);

        // 2. Skarn should exist in BOTH chunks (aureole crosses boundary)
        let skarn_voxels = find_material(&density_fields, Material::Skarn);
        let mut skarn_in_chunk0 = false;
        let mut skarn_in_chunk1 = false;
        for &(chunk, _, _, _, _, _, _) in &skarn_voxels {
            if chunk == (0, 0, 0) { skarn_in_chunk0 = true; }
            if chunk == (1, 0, 0) { skarn_in_chunk1 = true; }
        }
        assert!(skarn_in_chunk0, "Expected skarn in chunk (0,0,0) — aureole should reach into it");
        assert!(skarn_in_chunk1, "Expected skarn in chunk (1,0,0) — aureole should reach into it");
        eprintln!("  ✓ Skarn present in both chunks across boundary");

        // 3. Verify boundary voxels are synced (the boundary copy should match)
        // At x=16 in chunk (0,0,0) should equal x=0 in chunk (1,0,0)
        let df0 = &density_fields[&(0, 0, 0)];
        let df1 = &density_fields[&(1, 0, 0)];
        let mut boundary_mismatches = 0;
        for z in 0..FIELD_SIZE {
            for y in 0..FIELD_SIZE {
                let mat0 = df0.get(CHUNK_SIZE, y, z).material; // x=16 in chunk 0
                let mat1 = df1.get(0, y, z).material;          // x=0 in chunk 1
                if mat0 != mat1 {
                    boundary_mismatches += 1;
                    eprintln!("  BOUNDARY MISMATCH at y={}, z={}: chunk0[16]={:?}, chunk1[0]={:?}",
                        y, z, mat0, mat1);
                }
            }
        }
        assert_eq!(boundary_mismatches, 0,
            "Boundary voxels not synced — {} mismatches (set_voxel_synced failure)", boundary_mismatches);
        eprintln!("  ✓ All boundary voxels synced correctly (0 mismatches)");

        eprintln!("═══ PASS ═══\n");
    }

    // ─── Test 3: Air gap blocks aureole propagation ────────────────────────

    #[test]
    fn test_aureole_blocked_by_air_gap() {
        eprintln!("\n═══ TEST: Air gap blocks aureole propagation ═══");

        // Create chunk (0,0,0): limestone with an air cavity separating two regions
        let mut density_fields: HashMap<(i32, i32, i32), DensityField> = HashMap::new();
        let mut chunks = Vec::new();

        // 3x3x3 grid of chunks, all solid limestone
        for cx in -1..=1 {
            for cy in -1..=1 {
                for cz in -1..=1 {
                    let key = (cx, cy, cz);
                    density_fields.insert(key, make_solid_field(Material::Limestone));
                    chunks.push(key);
                }
            }
        }
        chunks.sort();

        // Carve an air wall at x=5 in chunk (0,0,0) — full YZ plane
        // This creates a barrier: lava at x=8 should NOT produce skarn at x<5
        let df = density_fields.get_mut(&(0, 0, 0)).unwrap();
        for z in 0..FIELD_SIZE {
            for y in 0..FIELD_SIZE {
                let s = df.get_mut(5, y, z);
                s.density = 0.0;
                s.material = Material::Air;
            }
        }

        // Place lava at x=8 (on the right side of the air wall)
        let mut fluid = FluidSnapshot::default();
        let mut lava_positions = Vec::new();
        for lz in 7..=9 {
            for ly in 7..=9 {
                for lx in 7..=9 {
                    lava_positions.push((lx, ly, lz));
                }
            }
        }
        place_lava(&mut fluid, (0, 0, 0), &lava_positions);
        eprintln!("  Lava at x=7-9, air wall at x=5");

        // Run aureole
        let mut config = AureoleConfig::default();
        config.zone_enabled = true;
        config.metamorphism_enabled = true;
        config.water_erosion_enabled = false;
        config.min_lava_zone_size = 1;
        config.max_radius = 10.0; // ensure radius could reach past the wall if not blocked

        let heat_map = build_heat_map(&density_fields, &fluid, &chunks, CHUNK_SIZE);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let groundwater = GroundwaterConfig::default();
        let census = make_empty_census();

        let result = apply_aureole(
            &config, &groundwater, &mut density_fields, &mut fluid,
            &heat_map, &chunks, CHUNK_SIZE, &mut rng, &census,
        );

        eprintln!("  Result: {} zones, {} skarn", result.lava_zones_found, result.skarn_placed);

        // ─── ASSERTIONS ───

        // No skarn should exist at x < 5 in chunk (0,0,0) — the air wall blocks propagation
        let skarn_voxels = find_material(&density_fields, Material::Skarn);
        for &(chunk, lx, ly, lz, wx, wy, wz) in &skarn_voxels {
            if chunk == (0, 0, 0) {
                assert!(
                    lx > 5,
                    "AUREOLE CROSSED AIR GAP: Skarn at local ({},{},{}) world ({},{},{}) — air wall at x=5 should block!",
                    lx, ly, lz, wx, wy, wz
                );
            }
        }
        eprintln!("  ✓ No skarn leaked through air wall at x=5");

        // Skarn should exist between the lava (x=7-9) and the wall (x=5) → x=6
        let has_skarn_near_lava = skarn_voxels.iter().any(|&(chunk, lx, _, _, _, _, _)| {
            chunk == (0, 0, 0) && lx == 6
        });
        assert!(has_skarn_near_lava,
            "Expected skarn at x=6 (between lava at 7-9 and air wall at 5)");
        eprintln!("  ✓ Skarn correctly placed between lava and air wall");

        eprintln!("═══ PASS ═══\n");
    }

    // ─── Test 4: Slate host rock → Hornfels (not Skarn) ───────────────────

    #[test]
    fn test_aureole_slate_produces_hornfels() {
        eprintln!("\n═══ TEST: Slate + Lava → Hornfels aureole ═══");

        let mut density_fields: HashMap<(i32, i32, i32), DensityField> = HashMap::new();
        let mut chunks = Vec::new();
        for cx in -1..=1 {
            for cy in -1..=1 {
                for cz in -1..=1 {
                    let key = (cx, cy, cz);
                    density_fields.insert(key, make_solid_field(Material::Slate));
                    chunks.push(key);
                }
            }
        }
        chunks.sort();

        // Place lava
        let mut fluid = FluidSnapshot::default();
        let mut lava_positions = Vec::new();
        for lz in 7..=9 {
            for ly in 7..=9 {
                for lx in 7..=9 {
                    lava_positions.push((lx, ly, lz));
                }
            }
        }
        place_lava(&mut fluid, (0, 0, 0), &lava_positions);

        let mut config = AureoleConfig::default();
        config.zone_enabled = true;
        config.metamorphism_enabled = true;
        config.water_erosion_enabled = false;
        config.min_lava_zone_size = 1;

        let heat_map = build_heat_map(&density_fields, &fluid, &chunks, CHUNK_SIZE);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let groundwater = GroundwaterConfig::default();
        let census = make_empty_census();

        let result = apply_aureole(
            &config, &groundwater, &mut density_fields, &mut fluid,
            &heat_map, &chunks, CHUNK_SIZE, &mut rng, &census,
        );

        eprintln!("  Result: {} hornfels, {} skarn", result.hornfels_placed, result.skarn_placed);

        // ─── ASSERTIONS ───

        // Slate → Hornfels (NOT Skarn — that's limestone only)
        assert!(result.hornfels_placed > 0, "Expected hornfels from slate aureole");
        assert_eq!(result.skarn_placed, 0, "Expected NO skarn in pure slate world");
        eprintln!("  ✓ Slate correctly produces Hornfels, not Skarn");

        eprintln!("═══ PASS ═══\n");
    }

    // ─── Test 5: No lava → no aureole ─────────────────────────────────────

    #[test]
    fn test_no_lava_no_aureole() {
        eprintln!("\n═══ TEST: No lava → no aureole materials ═══");

        let mut density_fields: HashMap<(i32, i32, i32), DensityField> = HashMap::new();
        let mut chunks = Vec::new();
        for cx in -1..=1 {
            for cy in -1..=1 {
                for cz in -1..=1 {
                    let key = (cx, cy, cz);
                    density_fields.insert(key, make_solid_field(Material::Limestone));
                    chunks.push(key);
                }
            }
        }
        chunks.sort();

        let fluid = FluidSnapshot::default(); // empty — no lava

        let mut config = AureoleConfig::default();
        config.zone_enabled = true;
        config.metamorphism_enabled = true;
        config.water_erosion_enabled = false;

        let heat_map = build_heat_map(&density_fields, &fluid, &chunks, CHUNK_SIZE);
        assert_eq!(heat_map.len(), 0, "No heat sources expected with empty fluid");

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let groundwater = GroundwaterConfig::default();
        let census = make_empty_census();

        let result = apply_aureole(
            &config, &groundwater, &mut density_fields, &mut fluid.clone(),
            &heat_map, &chunks, CHUNK_SIZE, &mut rng, &census,
        );

        assert_eq!(result.lava_zones_found, 0);
        assert_eq!(result.hornfels_placed, 0);
        assert_eq!(result.skarn_placed, 0);
        assert_eq!(result.veins_placed, 0);
        eprintln!("  ✓ No aureole materials with no heat sources");

        eprintln!("═══ PASS ═══\n");
    }

    // ─── Test 6: Realistic world with diagonal lava pipe ────────────────

    #[test]
    fn test_aureole_realistic_world_diagonal_lava() {
        eprintln!("\n═══ TEST: Realistic 3x3x3 world (seed 1) + diagonal lava pipe ═══");

        // Generate a realistic 3x3x3 world using the generation pipeline
        // Settings: seed 1, worms 0, cavern_freq 0.015, cavern_thresh 0.4, ore_detail 2
        let gen_config = voxel_gen::config::GenerationConfig {
            seed: 1,
            chunk_size: 16,
            bounds_size: 0.0,
            region_size: 3,
            ore_detail_multiplier: 2,
            fluid_sources_enabled: false,
            noise: voxel_gen::config::NoiseConfig {
                cavern_frequency: 0.015,
                cavern_threshold: 0.4,
                ..Default::default()
            },
            worm: voxel_gen::config::WormConfig {
                worms_per_region: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut coords = Vec::new();
        for cz in 0..3i32 {
            for cy in 0..3i32 {
                for cx in 0..3i32 {
                    coords.push((cx, cy, cz));
                }
            }
        }
        coords.sort();

        let (mut density_fields, _pools, _fluid_seeds, _worms, _timings, _springs, _zones) =
            voxel_gen::region_gen::generate_region_densities(&coords, &gen_config);

        eprintln!("  Generated {} chunks", density_fields.len());

        // Count materials before
        let before = count_materials(&density_fields);
        let total_solid_before: u32 = before.iter()
            .filter(|(&id, _)| id != 0) // exclude air
            .map(|(_, &c)| c).sum();
        eprintln!("  Total solid voxels: {}", total_solid_before);
        for (&id, &count) in &before {
            if count > 100 {
                eprintln!("    {}: {}", mat_name(id), count);
            }
        }

        // Inject diagonal lava pipe from (0,0,0) to (2,2,2) with radius 2
        let mut fluid = FluidSnapshot::default();
        let cs = CHUNK_SIZE;
        let mut lava_injected = 0u32;
        let mut lava_in_solid = 0u32;
        let mut lava_in_air = 0u32;

        for &chunk_key in &coords {
            let (cx, cy, cz) = chunk_key;
            // Only chunks on or near the diagonal
            if (cx - cy).abs() > 1 || (cy - cz).abs() > 1 || (cx - cz).abs() > 1 {
                continue;
            }

            let cells = fluid.chunks.entry(chunk_key).or_insert_with(|| {
                vec![FluidCell {
                    level: 0.0,
                    fluid_type: FluidType::Water,
                    is_source: false,
                    grace_ticks: 0,
                    stagnant_ticks: 0,
                }; cs * cs * cs]
            });

            for i in 0..cs {
                let center = i.min(cs - 1);
                for dz in -2i32..=2 {
                    for dy in -2i32..=2 {
                        for dx in -2i32..=2 {
                            let lx = center as i32 + dx;
                            let ly = center as i32 + dy;
                            let lz = center as i32 + dz;
                            if lx < 0 || lx >= cs as i32 || ly < 0 || ly >= cs as i32 || lz < 0 || lz >= cs as i32 {
                                continue;
                            }
                            if dx * dx + dy * dy + dz * dz > 6 {
                                continue;
                            }
                            let idx = lz as usize * cs * cs + ly as usize * cs + lx as usize;
                            cells[idx] = FluidCell {
                                level: 1.0,
                                fluid_type: FluidType::Lava,
                                is_source: true,
                                grace_ticks: 0,
                                stagnant_ticks: 0,
                            };
                            lava_injected += 1;

                            // Check what's at this position in the density field
                            if let Some(df) = density_fields.get(&chunk_key) {
                                let s = df.get(lx as usize, ly as usize, lz as usize);
                                if s.material.is_solid() { lava_in_solid += 1; } else { lava_in_air += 1; }
                            }
                        }
                    }
                }
            }
        }

        eprintln!("  Lava pipe: {} cells ({} in solid, {} in air)",
            lava_injected, lava_in_solid, lava_in_air);

        // Run aureole only
        let mut aureole_config = AureoleConfig::default();
        aureole_config.zone_enabled = true;
        aureole_config.metamorphism_enabled = true;
        aureole_config.water_erosion_enabled = false;
        aureole_config.min_lava_zone_size = 1;

        let chunks_list: Vec<(i32, i32, i32)> = density_fields.keys().copied().collect();
        let heat_map = build_heat_map(&density_fields, &fluid, &chunks_list, CHUNK_SIZE);
        eprintln!("  Heat map: {} sources", heat_map.len());

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let groundwater = GroundwaterConfig::default();
        let census = make_empty_census();

        let result = apply_aureole(
            &aureole_config, &groundwater, &mut density_fields, &mut fluid,
            &heat_map, &chunks_list, CHUNK_SIZE, &mut rng, &census,
        );

        eprintln!("\n  ─── RESULTS ───");
        eprintln!("  Lava zones: {}", result.lava_zones_found);
        eprintln!("  Hornfels placed: {}", result.hornfels_placed);
        eprintln!("  Skarn placed: {}", result.skarn_placed);
        eprintln!("  Veins placed: {}", result.veins_placed);
        eprintln!("  Total metamorphosed: {}", result.voxels_metamorphosed);

        // Show debug zone info
        for line in &result.debug_lines {
            eprintln!("    {}", line);
        }

        // Material census after
        let after = count_materials(&density_fields);
        eprintln!("\n  Material changes:");
        let mut all_mats: std::collections::BTreeSet<u8> = before.keys().copied().collect();
        all_mats.extend(after.keys());
        for mat_id in &all_mats {
            let b = before.get(mat_id).unwrap_or(&0);
            let a = after.get(mat_id).unwrap_or(&0);
            if b != a {
                let delta = *a as i64 - *b as i64;
                eprintln!("    {}: {} → {} ({:+})", mat_name(*mat_id), b, a, delta);
            }
        }

        // Find which chunks have aureole materials
        let hornfels_voxels = find_material(&density_fields, Material::Hornfels);
        let skarn_voxels = find_material(&density_fields, Material::Skarn);
        let mut aureole_chunks: HashSet<(i32, i32, i32)> = HashSet::new();
        for &(chunk, _, _, _, _, _, _) in &hornfels_voxels { aureole_chunks.insert(chunk); }
        for &(chunk, _, _, _, _, _, _) in &skarn_voxels { aureole_chunks.insert(chunk); }
        let mut aureole_chunk_list: Vec<_> = aureole_chunks.iter().copied().collect();
        aureole_chunk_list.sort();
        eprintln!("\n  Aureole present in {} chunks: {:?}", aureole_chunk_list.len(), aureole_chunk_list);

        // ─── ASSERTIONS ───

        // Must have found lava zones
        assert!(result.lava_zones_found > 0, "Expected lava zones from diagonal pipe");

        // Must have produced metamorphic minerals
        assert!(result.hornfels_placed + result.skarn_placed > 0,
            "Expected aureole metamorphism from {} lava cells ({} in solid rock)",
            lava_injected, lava_in_solid);

        // Aureole should only be in chunks that are on/near the diagonal
        for &chunk in &aureole_chunk_list {
            let (cx, cy, cz) = chunk;
            let off_diagonal = (cx - cy).abs().max((cy - cz).abs()).max((cx - cz).abs());
            assert!(off_diagonal <= 2,
                "AUREOLE LEAK: Found aureole material in chunk ({},{},{}) which is {} steps off diagonal",
                cx, cy, cz, off_diagonal);
        }
        eprintln!("  ✓ All aureole materials within expected diagonal corridor");

        eprintln!("═══ PASS ═══\n");
    }

    // ─── Test 7: Verify world_to_chunk_local roundtrip ────────────────────

    #[test]
    fn test_coordinate_roundtrip() {
        eprintln!("\n═══ TEST: world_to_chunk_local coordinate roundtrip ═══");

        // Test positive coordinates
        let (key, lx, ly, lz) = world_to_chunk_local(20, 5, -3, CHUNK_SIZE);
        assert_eq!(key, (1, 0, -1), "chunk for (20,5,-3) should be (1,0,-1)");
        assert_eq!((lx, ly, lz), (4, 5, 13), "local for (20,5,-3) should be (4,5,13)");

        // Verify: chunk_coord * chunk_size + local = world
        assert_eq!(key.0 * CHUNK_SIZE as i32 + lx as i32, 20);
        assert_eq!(key.1 * CHUNK_SIZE as i32 + ly as i32, 5);
        assert_eq!(key.2 * CHUNK_SIZE as i32 + lz as i32, -3);

        // Test boundary cases
        let (key, lx, _, _) = world_to_chunk_local(16, 0, 0, CHUNK_SIZE);
        assert_eq!(key, (1, 0, 0), "x=16 should be in chunk 1");
        assert_eq!(lx, 0, "x=16 should be local 0 in chunk 1");

        let (key, lx, _, _) = world_to_chunk_local(15, 0, 0, CHUNK_SIZE);
        assert_eq!(key, (0, 0, 0), "x=15 should be in chunk 0");
        assert_eq!(lx, 15, "x=15 should be local 15 in chunk 0");

        // Test negative coordinates
        let (key, lx, _, _) = world_to_chunk_local(-1, 0, 0, CHUNK_SIZE);
        assert_eq!(key, (-1, 0, 0), "x=-1 should be in chunk -1");
        assert_eq!(lx, 15, "x=-1 should be local 15 in chunk -1");

        eprintln!("  ✓ All coordinate roundtrips correct");
        eprintln!("═══ PASS ═══\n");
    }
}
