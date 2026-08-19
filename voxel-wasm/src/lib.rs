//! Browser/WASM cave demo: the voxel-viewer HTTP endpoints re-exposed as
//! direct function calls for a Web Worker.
//!
//! Request/response bodies are byte-for-byte the same JSON the native
//! server produced, so the frontend only swaps its transport layer.
//! Errors come back as `{"error": "..."}` and the JS shim turns them
//! into rejected promises, mirroring a non-2xx fetch.

use std::collections::HashMap;

use wasm_bindgen::prelude::*;

use voxel_region::params::parse_generate_form;
use voxel_region::GeneratedRegion;
use voxel_sleep::config::SleepConfig;

#[wasm_bindgen]
pub struct CaveDemo {
    region: Option<GeneratedRegion>,
    sleep_config: SleepConfig,
}

fn err_json(message: &str) -> String {
    serde_json::json!({ "error": message }).to_string()
}

#[wasm_bindgen]
impl CaveDemo {
    #[wasm_bindgen(constructor)]
    pub fn new() -> CaveDemo {
        console_error_panic_hook::set_once();
        CaveDemo {
            region: None,
            sleep_config: SleepConfig::default(),
        }
    }

    /// POST /api/generate — body is the urlencoded form string.
    pub fn generate(&mut self, body: &str) -> String {
        let req = parse_generate_form(body);
        let (chunks_x, chunks_y, chunks_z) = req.chunks;
        let chunks_x = chunks_x.min(8);
        let chunks_y = chunks_y.min(8);
        let chunks_z = chunks_z.min(8);

        let region = GeneratedRegion::generate(
            req.config,
            (0, 0, 0),
            (chunks_x as i32, chunks_y as i32, chunks_z as i32),
            req.closed,
        );

        let mesh_json = region.to_json_mesh();
        let verts = mesh_json.positions.len() / 3;
        let tris = mesh_json.indices.len() / 3;

        let pool_descriptors = region.pool_descriptors.clone();
        let zone_descriptors: Vec<serde_json::Value> = region
            .zone_descriptors
            .iter()
            .map(|z| {
                serde_json::json!({
                    "zone_type": z.zone_type as u8,
                    "zone_name": format!("{:?}", z.zone_type),
                    "world_min": [z.world_min.x, z.world_min.y, z.world_min.z],
                    "world_max": [z.world_max.x, z.world_max.y, z.world_max.z],
                    "center": [z.center.x, z.center.y, z.center.z],
                    "anchor_count": z.anchors.len(),
                })
            })
            .collect();

        let output_msg = format!(
            "Generated {}x{}x{} in-browser: {} verts, {} tris, {} pools, {} zones",
            chunks_x, chunks_y, chunks_z, verts, tris,
            pool_descriptors.len(), zone_descriptors.len()
        );

        self.region = Some(region);
        self.sleep_config = req.sleep_config;

        serde_json::json!({
            "ok": true,
            "mesh": mesh_json,
            "output": output_msg,
            "pools": pool_descriptors,
            "zones": zone_descriptors,
        })
        .to_string()
    }

    /// POST /api/mine — body is the JSON `{x,y,z,mode,radius,nx,ny,nz}`.
    pub fn mine(&mut self, body: &str) -> String {
        let params: serde_json::Value = match serde_json::from_str(body) {
            Ok(v) => v,
            Err(e) => return err_json(&format!("Invalid JSON: {e}")),
        };

        let x = params["x"].as_f64().unwrap_or(0.0) as f32;
        let y = params["y"].as_f64().unwrap_or(0.0) as f32;
        let z = params["z"].as_f64().unwrap_or(0.0) as f32;
        let mode = params["mode"].as_str().unwrap_or("sphere");
        let radius = params["radius"].as_f64().unwrap_or(5.0) as f32;
        let nx = params["nx"].as_f64().unwrap_or(0.0) as f32;
        let ny = params["ny"].as_f64().unwrap_or(1.0) as f32;
        let nz = params["nz"].as_f64().unwrap_or(0.0) as f32;

        let center = glam::Vec3::new(x, y, z);
        let normal = glam::Vec3::new(nx, ny, nz).normalize_or_zero();

        let region = match self.region.as_mut() {
            Some(r) => r,
            None => return err_json("No region generated yet. Generate first."),
        };

        let is_lava_carve = mode == "lava-carve";
        let mine_result = match mode {
            "peel" => region.mine_peel(center, normal, radius),
            _ => region.mine_sphere(center, radius),
        };

        // Lava Carve: fill all air voxels in the carved sphere with lava seeds
        if is_lava_carve {
            let cs = region.config.chunk_size;
            let r = radius as i32;
            let cx = center.x as i32;
            let cy = center.y as i32;
            let cz = center.z as i32;
            for wz in (cz - r)..=(cz + r) {
                for wy in (cy - r)..=(cy + r) {
                    for wx in (cx - r)..=(cx + r) {
                        let dx = wx - cx;
                        let dy = wy - cy;
                        let dz = wz - cz;
                        if dx * dx + dy * dy + dz * dz > r * r {
                            continue;
                        }
                        let chunk_key = (
                            wx.div_euclid(cs as i32),
                            wy.div_euclid(cs as i32),
                            wz.div_euclid(cs as i32),
                        );
                        let lx = wx.rem_euclid(cs as i32) as usize;
                        let ly = wy.rem_euclid(cs as i32) as usize;
                        let lz = wz.rem_euclid(cs as i32) as usize;
                        // Only place lava in air voxels
                        if let Some(df) = region.density_fields.get(&chunk_key) {
                            if !df.get(lx, ly, lz).material.is_solid() {
                                region.fluid_seeds.push(voxel_gen::pools::FluidSeed {
                                    chunk: chunk_key,
                                    lx: lx as u8,
                                    ly: ly as u8,
                                    lz: lz as u8,
                                    fluid_type: voxel_gen::pools::PoolFluid::Lava,
                                    is_source: true,
                                });
                            }
                        }
                    }
                }
            }
        }

        // Pool containment check deliberately skipped, matching the server:
        // is_pool_contained is too strict for natural cave geometry.
        let surviving_pools = region.pool_descriptors.clone();

        let mesh_json = region.to_json_mesh();

        let mined: Vec<serde_json::Value> = mine_result
            .mined_materials
            .iter()
            .map(|(mat, count)| {
                serde_json::json!({
                    "material": mat.display_name(),
                    "count": count,
                })
            })
            .collect();

        serde_json::json!({
            "mesh": mesh_json,
            "mined": mined,
            "pools": surviving_pools,
        })
        .to_string()
    }

    /// POST /api/place-water — body is the JSON `{x,y,z,radius}`.
    pub fn place_water(&mut self, body: &str) -> String {
        let params: serde_json::Value = match serde_json::from_str(body) {
            Ok(v) => v,
            Err(e) => return err_json(&format!("Invalid JSON: {e}")),
        };

        let x = params["x"].as_f64().unwrap_or(0.0) as f32;
        let y = params["y"].as_f64().unwrap_or(0.0) as f32;
        let z = params["z"].as_f64().unwrap_or(0.0) as f32;
        let radius = params["radius"].as_f64().unwrap_or(3.0) as f32;

        let region = match self.region.as_mut() {
            Some(r) => r,
            None => return err_json("No region generated yet. Generate first."),
        };

        let count = region.place_water(x, y, z, radius);

        serde_json::json!({
            "placed": count,
            "position": [x, y, z],
            "radius": radius,
        })
        .to_string()
    }

    /// POST /api/sleep — no body.
    pub fn sleep(&mut self) -> String {
        let sleep_config = self.sleep_config.clone();
        let region = match self.region.as_mut() {
            Some(r) => r,
            None => return err_json("No region generated yet. Generate first."),
        };

        let (sleep_result, mesh_json) = region.apply_sleep(&sleep_config);

        let transform_log: Vec<serde_json::Value> = sleep_result
            .transform_log
            .iter()
            .map(|entry| {
                serde_json::json!({
                    "description": entry.description,
                    "count": entry.count,
                })
            })
            .collect();

        let mut material_counts: HashMap<String, i64> = HashMap::new();
        for df in region.density_fields.values() {
            for z in 0..region.config.chunk_size {
                for y in 0..region.config.chunk_size {
                    for x in 0..region.config.chunk_size {
                        let mat = df.get(x, y, z).material;
                        if mat.is_solid() {
                            *material_counts.entry(mat.display_name().to_string()).or_insert(0) += 1;
                        }
                    }
                }
            }
        }

        let material_diff: Vec<serde_json::Value> = material_counts
            .iter()
            .map(|(name, count)| {
                serde_json::json!({
                    "material": name,
                    "count": count,
                })
            })
            .collect();

        serde_json::json!({
            "ok": true,
            "mesh": mesh_json,
            "stats": {
                "chunks_changed": sleep_result.chunks_changed,
                "acid_dissolved": sleep_result.acid_dissolved,
                "voxels_oxidized": sleep_result.voxels_oxidized,
                "voxels_metamorphosed": sleep_result.voxels_metamorphosed,
                "veins_deposited": sleep_result.veins_deposited,
                "formations_grown": sleep_result.formations_grown,
                "voxels_enriched": sleep_result.voxels_enriched,
                "supports_degraded": sleep_result.supports_degraded,
                "collapses_triggered": sleep_result.collapses_triggered,
                "minerals_grown": sleep_result.minerals_grown,
                "sulfide_dissolved": sleep_result.sulfide_dissolved,
                "coal_matured": sleep_result.coal_matured,
                "diamonds_formed": sleep_result.diamonds_formed,
                "voxels_silicified": sleep_result.voxels_silicified,
                "nests_fossilized": sleep_result.nests_fossilized,
            },
            "transform_log": transform_log,
            "material_diff": material_diff,
        })
        .to_string()
    }
}
