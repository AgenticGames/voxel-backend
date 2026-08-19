
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use voxel_gen::config::GenerationConfig;
use voxel_sleep::config::SleepConfig;

use voxel_region::GeneratedRegion;

// Embed static files into the binary
const INDEX_HTML: &str = include_str!("static/index.html");
const APP_JS: &str = include_str!("static/app.js");
const STYLE_CSS: &str = include_str!("static/style.css");
const LOGO_PNG: &[u8] = include_bytes!("static/logo.png");

/// Shared state across requests
struct AppState {
    region: Option<GeneratedRegion>,
    sleep_config: SleepConfig,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut output_dir = PathBuf::from("./test-output");
    let mut port: u16 = 8080;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--dir" => {
                i += 1;
                if i < args.len() {
                    output_dir = PathBuf::from(&args[i]);
                }
            }
            "--port" => {
                i += 1;
                if i < args.len() {
                    port = args[i].parse().expect("Invalid port number");
                }
            }
            "--help" | "-h" => {
                println!("Usage: voxel-viewer [--dir <path>] [--port <port>]");
                println!("  --dir   Output directory with report.json and OBJ files (default: ./test-output)");
                println!("  --port  HTTP port (default: 8080)");
                return;
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let addr = format!("0.0.0.0:{port}");
    let server = tiny_http::Server::http(&addr).unwrap_or_else(|e| {
        eprintln!("Failed to bind to {addr}: {e}");
        std::process::exit(1);
    });

    let state: Arc<Mutex<AppState>> = Arc::new(Mutex::new(AppState {
        region: None,
        sleep_config: SleepConfig::default(),
    }));

    println!("Serving at http://localhost:{port}");
    println!("Output directory: {}", output_dir.display());

    for request in server.incoming_requests() {
        let url = request.url().to_string();
        let method = request.method().to_string();
        let remote = request.remote_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|| "unknown".to_string());

        // Log non-static requests (page loads + API calls). Unix seconds
        // instead of chrono: the viewer is a local dev tool now (the public
        // demo runs in-browser), and chrono 0.4.45+ can't build on this
        // machine's windows-gnu toolchain (raw-dylib needs a dlltool/as pair
        // the self-contained toolchain doesn't ship).
        if url == "/" || url.starts_with("/api/") {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            println!("[{now}] {remote} {method} {url}");
        }

        let result = match (method.as_str(), url.as_str()) {
            ("GET", "/") => serve_static(request, INDEX_HTML, "text/html"),
            ("GET", "/static/app.js") => serve_static(request, APP_JS, "application/javascript"),
            ("GET", "/static/style.css") => serve_static(request, STYLE_CSS, "text/css"),
            ("GET", "/static/logo.png") => serve_binary(request, LOGO_PNG, "image/png"),
            ("GET", "/api/report") => serve_report(request, &output_dir),
            ("GET", "/api/obj-files") => serve_obj_file_list(request, &output_dir),
            ("GET", path) if path.starts_with("/api/obj-file/") => serve_obj_by_name(request, path, &output_dir),
            ("DELETE", path) if path.starts_with("/api/obj-file/") => delete_obj_file(request, path, &output_dir),
            ("GET", path) if path.starts_with("/api/obj/") => serve_obj(request, path, &output_dir),
            ("POST", "/api/generate") => serve_generate(request, &state),
            ("POST", "/api/mine") => serve_mine(request, &state),
            ("POST", "/api/place-water") => serve_place_water(request, &state),
            ("POST", "/api/sleep") => serve_sleep(request, &state),
            ("POST", "/api/run-batch") => serve_run_batch(request, &output_dir),
            _ => serve_not_found(request),
        };

        if let Err(e) = result {
            eprintln!("Error handling request {url}: {e}");
        }
    }
}

fn serve_static(
    request: tiny_http::Request,
    body: &str,
    content_type: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let header = tiny_http::Header::from_bytes(b"Content-Type", content_type.as_bytes())
        .map_err(|_| "invalid header")?;
    let response = tiny_http::Response::from_string(body).with_header(header);
    request.respond(response)?;
    Ok(())
}

fn serve_binary(
    request: tiny_http::Request,
    body: &[u8],
    content_type: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let header = tiny_http::Header::from_bytes(b"Content-Type", content_type.as_bytes())
        .map_err(|_| "invalid header")?;
    let response = tiny_http::Response::from_data(body.to_vec()).with_header(header);
    request.respond(response)?;
    Ok(())
}

fn serve_report(
    request: tiny_http::Request,
    output_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let report_path = output_dir.join("report.json");
    match std::fs::read_to_string(&report_path) {
        Ok(json) => {
            if serde_json::from_str::<serde_json::Value>(&json).is_err() {
                return serve_error(request, 500, "Invalid JSON in report.json");
            }
            let header =
                tiny_http::Header::from_bytes(b"Content-Type", b"application/json")
                    .map_err(|_| "invalid header")?;
            let response = tiny_http::Response::from_string(json).with_header(header);
            request.respond(response)?;
            Ok(())
        }
        Err(e) => serve_error(
            request,
            404,
            &format!("Could not read {}: {e}", report_path.display()),
        ),
    }
}

fn serve_obj(
    request: tiny_http::Request,
    path: &str,
    output_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let seed_str = path.strip_prefix("/api/obj/").unwrap_or("");
    let seed: u64 = match seed_str.parse() {
        Ok(s) => s,
        Err(_) => return serve_error(request, 400, "Invalid seed number"),
    };

    let obj_path = output_dir.join(format!("seed_{seed}.obj"));
    let obj_path = if obj_path.exists() {
        obj_path
    } else {
        output_dir.join(format!("failure_{seed}.obj"))
    };
    match std::fs::read_to_string(&obj_path) {
        Ok(content) => {
            let header = tiny_http::Header::from_bytes(b"Content-Type", b"text/plain")
                .map_err(|_| "invalid header")?;
            let response = tiny_http::Response::from_string(content).with_header(header);
            request.respond(response)?;
            Ok(())
        }
        Err(_) => serve_error(
            request,
            404,
            &format!("OBJ file not found: {}", obj_path.display()),
        ),
    }
}

fn serve_run_batch(
    request: tiny_http::Request,
    output_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir_arg = output_dir.to_string_lossy().to_string();

    let result = std::process::Command::new("cargo")
        .args(["run", "-p", "voxel-cli", "--", "batch-test", "--output-dir", &dir_arg])
        .output();

    match result {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let body = if output.status.success() {
                format!("Batch test completed successfully.\n\n{stdout}")
            } else {
                format!("Batch test failed (exit code: {:?}).\n\nstdout:\n{stdout}\n\nstderr:\n{stderr}", output.status.code())
            };
            let header =
                tiny_http::Header::from_bytes(b"Content-Type", b"text/plain")
                    .map_err(|_| "invalid header")?;
            let response = tiny_http::Response::from_string(body).with_header(header);
            request.respond(response)?;
            Ok(())
        }
        Err(e) => serve_error(request, 500, &format!("Failed to run batch: {e}")),
    }
}

fn serve_obj_file_list(
    request: tiny_http::Request,
    output_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut files: Vec<String> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(output_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".obj") && !name.starts_with("seed_") && !name.starts_with("failure_") {
                files.push(name);
            }
        }
    }
    files.sort();
    let json = serde_json::to_string(&files).unwrap_or_else(|_| "[]".to_string());
    let header = tiny_http::Header::from_bytes(b"Content-Type", b"application/json")
        .map_err(|_| "invalid header")?;
    let response = tiny_http::Response::from_string(json).with_header(header);
    request.respond(response)?;
    Ok(())
}

fn serve_obj_by_name(
    request: tiny_http::Request,
    path: &str,
    output_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let filename = path.strip_prefix("/api/obj-file/").unwrap_or("");
    if filename.contains("..") || filename.contains('/') || filename.contains('\\') {
        return serve_error(request, 400, "Invalid filename");
    }
    if !filename.ends_with(".obj") {
        return serve_error(request, 400, "Only .obj files can be served");
    }
    let obj_path = output_dir.join(filename);
    match std::fs::read_to_string(&obj_path) {
        Ok(content) => {
            let header = tiny_http::Header::from_bytes(b"Content-Type", b"text/plain")
                .map_err(|_| "invalid header")?;
            let response = tiny_http::Response::from_string(content).with_header(header);
            request.respond(response)?;
            Ok(())
        }
        Err(_) => serve_error(request, 404, &format!("File not found: {filename}")),
    }
}

fn delete_obj_file(
    request: tiny_http::Request,
    path: &str,
    output_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let filename = path.strip_prefix("/api/obj-file/").unwrap_or("");
    if filename.contains("..") || filename.contains('/') || filename.contains('\\') {
        return serve_error(request, 400, "Invalid filename");
    }
    if !filename.ends_with(".obj") {
        return serve_error(request, 400, "Only .obj files can be deleted");
    }
    let file_path = output_dir.join(filename);
    match std::fs::remove_file(&file_path) {
        Ok(()) => {
            let header = tiny_http::Header::from_bytes(b"Content-Type", b"application/json")
                .map_err(|_| "invalid header")?;
            let response = tiny_http::Response::from_string("{\"ok\":true}").with_header(header);
            request.respond(response)?;
            Ok(())
        }
        Err(e) => serve_error(request, 404, &format!("Could not delete: {e}")),
    }
}

/// Generate multi-chunk mesh in-process, store region for mining, return JSON mesh.
fn serve_generate(
    mut request: tiny_http::Request,
    state: &Arc<Mutex<AppState>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut body = String::new();
    request.as_reader().read_to_string(&mut body)?;

    // Shared parser (voxel-region): identical parameter handling for the
    // native server and the browser/WASM demo.
    let req = voxel_region::params::parse_generate_form(&body);
    let config = req.config;
    let sleep_cfg = req.sleep_config;
    let (chunks_x, chunks_y, chunks_z) = req.chunks;
    let closed = req.closed;
    let seed = req.seed;

    println!("Generating {}x{}x{} region in-process (seed {})...", chunks_x, chunks_y, chunks_z, seed);
    let start = std::time::Instant::now();

    let region = GeneratedRegion::generate(
        config,
        (0, 0, 0),
        (chunks_x as i32, chunks_y as i32, chunks_z as i32),
        closed,
    );

    let elapsed = start.elapsed();
    let mesh_json = region.to_json_mesh();
    let verts = mesh_json.positions.len() / 3;
    let tris = mesh_json.indices.len() / 3;
    println!("  Generated in {:.2?}: {} vertices, {} triangles", elapsed, verts, tris);

    // Store region + sleep config for mining/sleep
    {
        let mut app = state.lock().unwrap();
        app.region = Some(region);
        app.sleep_config = sleep_cfg;
    }

    // Collect pool and zone descriptors from the region
    let (pool_descriptors, zone_descriptors) = {
        let app = state.lock().unwrap();
        let pools = app.region.as_ref().map(|r| r.pool_descriptors.clone()).unwrap_or_default();
        let zones = app.region.as_ref().map(|r| {
            r.zone_descriptors.iter().map(|z| {
                serde_json::json!({
                    "zone_type": z.zone_type as u8,
                    "zone_name": format!("{:?}", z.zone_type),
                    "world_min": [z.world_min.x, z.world_min.y, z.world_min.z],
                    "world_max": [z.world_max.x, z.world_max.y, z.world_max.z],
                    "center": [z.center.x, z.center.y, z.center.z],
                    "anchor_count": z.anchors.len(),
                })
            }).collect::<Vec<_>>()
        }).unwrap_or_default();
        (pools, zones)
    };

    // Serialize and respond
    let output_msg = format!("Generated {}x{}x{} in {:.2?}: {} verts, {} tris, {} pools, {} zones",
        chunks_x, chunks_y, chunks_z, elapsed, verts, tris, pool_descriptors.len(), zone_descriptors.len());

    let response_json = serde_json::json!({
        "ok": true,
        "mesh": mesh_json,
        "output": output_msg,
        "pools": pool_descriptors,
        "zones": zone_descriptors,
    });

    let json_str = serde_json::to_string(&response_json)?;
    let header = tiny_http::Header::from_bytes(b"Content-Type", b"application/json")
        .map_err(|_| "invalid header")?;
    let response = tiny_http::Response::from_string(json_str).with_header(header);
    request.respond(response)?;
    Ok(())
}

/// Mine endpoint: carve into the stored region and return updated mesh + mined materials.
fn serve_mine(
    mut request: tiny_http::Request,
    state: &Arc<Mutex<AppState>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut body = String::new();
    request.as_reader().read_to_string(&mut body)?;

    // Parse JSON body
    let params: serde_json::Value = serde_json::from_str(&body)
        .map_err(|e| format!("Invalid JSON: {e}"))?;

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

    let mut app = state.lock().unwrap();
    let region = match app.region.as_mut() {
        Some(r) => r,
        None => {
            drop(app);
            return serve_error(request, 400, "No region generated yet. Generate first.");
        }
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
        let mut lava_placed = 0u32;
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
                            lava_placed += 1;
                        }
                    }
                }
            }
        }
        eprintln!("Lava Carve: placed {} lava seeds in sphere at ({},{},{})", lava_placed, cx, cy, cz);
    }

    // Skip pool containment check — is_pool_contained is too strict for the
    // natural cave geometry (rim voxels at surface_y are often cave air even
    // before any mining).  Pool surfaces are cosmetic in the viewer.
    let surviving_pools = region.pool_descriptors.clone();

    let mesh_json = region.to_json_mesh();

    // Build mined materials array
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

    let response_json = serde_json::json!({
        "mesh": mesh_json,
        "mined": mined,
        "pools": surviving_pools,
    });

    let json_str = serde_json::to_string(&response_json)?;
    let header = tiny_http::Header::from_bytes(b"Content-Type", b"application/json")
        .map_err(|_| "invalid header")?;
    let response = tiny_http::Response::from_string(json_str).with_header(header);
    request.respond(response)?;
    Ok(())
}

/// Place water endpoint: inject water cells at a world position for hydrothermal testing.
fn serve_place_water(
    mut request: tiny_http::Request,
    state: &Arc<Mutex<AppState>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut body = String::new();
    request.as_reader().read_to_string(&mut body)?;

    let params: serde_json::Value = serde_json::from_str(&body)
        .map_err(|e| format!("Invalid JSON: {e}"))?;

    let x = params["x"].as_f64().unwrap_or(0.0) as f32;
    let y = params["y"].as_f64().unwrap_or(0.0) as f32;
    let z = params["z"].as_f64().unwrap_or(0.0) as f32;
    let radius = params["radius"].as_f64().unwrap_or(3.0) as f32;

    let mut app = state.lock().unwrap();
    let region = match app.region.as_mut() {
        Some(r) => r,
        None => {
            drop(app);
            return serve_error(request, 400, "No region generated yet. Generate first.");
        }
    };

    let count = region.place_water(x, y, z, radius);

    let response_json = serde_json::json!({
        "placed": count,
        "position": [x, y, z],
        "radius": radius,
    });
    let json_str = serde_json::to_string(&response_json)?;
    let header = tiny_http::Header::from_bytes(b"Content-Type", b"application/json")
        .map_err(|_| "invalid header")?;
    let response = tiny_http::Response::from_string(json_str).with_header(header);
    request.respond(response)?;
    Ok(())
}

/// Sleep endpoint: run deep sleep on the stored region, return updated mesh + transform log.
fn serve_sleep(
    request: tiny_http::Request,
    state: &Arc<Mutex<AppState>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut app = state.lock().unwrap();
    let sleep_config = app.sleep_config.clone();
    let region = match app.region.as_mut() {
        Some(r) => r,
        None => {
            drop(app);
            return serve_error(request, 400, "No region generated yet. Generate first.");
        }
    };

    println!("Running deep sleep cycle...");
    let start = std::time::Instant::now();

    let (sleep_result, mesh_json) = region.apply_sleep(&sleep_config);

    let elapsed = start.elapsed();
    println!("  Sleep completed in {:.2?}: {} chunks changed, {} acid dissolved, {} metamorphosed, {} veins deposited, {} enriched, {} supports degraded, {} collapses",
        elapsed,
        sleep_result.chunks_changed,
        sleep_result.acid_dissolved,
        sleep_result.voxels_metamorphosed,
        sleep_result.veins_deposited,
        sleep_result.voxels_enriched,
        sleep_result.supports_degraded,
        sleep_result.collapses_triggered,
    );

    // Build transform log for UI
    let transform_log: Vec<serde_json::Value> = sleep_result.transform_log.iter()
        .map(|entry| serde_json::json!({
            "description": entry.description,
            "count": entry.count,
        }))
        .collect();

    // Build material count diff (before/after comparison)
    // Count current materials across all density fields
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

    let material_diff: Vec<serde_json::Value> = material_counts.iter()
        .map(|(name, count)| serde_json::json!({
            "material": name,
            "count": count,
        }))
        .collect();

    let response_json = serde_json::json!({
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
    });

    let json_str = serde_json::to_string(&response_json)?;
    let header = tiny_http::Header::from_bytes(b"Content-Type", b"application/json")
        .map_err(|_| "invalid header")?;
    let response = tiny_http::Response::from_string(json_str).with_header(header);
    request.respond(response)?;
    Ok(())
}

fn serve_not_found(request: tiny_http::Request) -> Result<(), Box<dyn std::error::Error>> {
    serve_error(request, 404, "Not found")
}

fn serve_error(
    request: tiny_http::Request,
    code: u16,
    message: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let header = tiny_http::Header::from_bytes(b"Content-Type", b"text/plain")
        .map_err(|_| "invalid header")?;
    let response = tiny_http::Response::from_string(message)
        .with_status_code(code)
        .with_header(header);
    request.respond(response)?;
    Ok(())
}
