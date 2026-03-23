mod region;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use voxel_gen::config::GenerationConfig;
use voxel_sleep::config::SleepConfig;

use crate::region::GeneratedRegion;

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

        // Log non-static requests (page loads + API calls)
        if url == "/" || url.starts_with("/api/") {
            let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
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

    let mut seed = 42u64;
    let mut chunks_x = 3u32;
    let mut chunks_y = 3u32;
    let mut chunks_z = 1u32;
    let mut closed = false;

    let mut cavern_freq: Option<f64> = None;
    let mut cavern_threshold: Option<f64> = None;
    let mut detail_octaves: Option<u32> = None;
    let mut detail_persistence: Option<f64> = None;
    let mut warp_amplitude: Option<f64> = None;
    let mut worms_per_region: Option<f32> = None;
    let mut worm_radius_min: Option<f32> = None;
    let mut worm_radius_max: Option<f32> = None;
    let mut worm_step_length: Option<f32> = None;
    let mut worm_max_steps: Option<u32> = None;
    let mut worm_falloff_power: Option<f32> = None;
    let mut chunk_size: Option<usize> = None;
    let mut sandstone_depth: Option<f64> = None;
    let mut granite_depth: Option<f64> = None;
    let mut basalt_depth: Option<f64> = None;
    let mut slate_depth: Option<f64> = None;
    let mut iron_band_freq: Option<f64> = None;
    let mut iron_noise_freq: Option<f64> = None;
    let mut iron_perturbation: Option<f64> = None;
    let mut iron_threshold: Option<f64> = None;
    let mut copper_freq: Option<f64> = None;
    let mut copper_threshold: Option<f64> = None;
    let mut malachite_freq: Option<f64> = None;
    let mut malachite_threshold: Option<f64> = None;
    let mut kimberlite_pipe_freq: Option<f64> = None;
    let mut kimberlite_pipe_threshold: Option<f64> = None;
    let mut diamond_freq: Option<f64> = None;
    let mut diamond_threshold: Option<f64> = None;
    let mut sulfide_freq: Option<f64> = None;
    let mut sulfide_threshold: Option<f64> = None;
    let mut tin_threshold: Option<f64> = None;
    let mut pyrite_freq: Option<f64> = None;
    let mut pyrite_threshold: Option<f64> = None;
    let mut quartz_freq: Option<f64> = None;
    let mut quartz_threshold: Option<f64> = None;
    let mut gold_threshold: Option<f64> = None;
    let mut geode_freq: Option<f64> = None;
    let mut geode_center_threshold: Option<f64> = None;
    let mut geode_shell_thickness: Option<f64> = None;
    let mut geode_hollow_factor: Option<f32> = None;
    // Pool settings
    let mut pools_enabled: Option<bool> = None;
    let mut pool_placement_freq: Option<f64> = None;
    let mut pool_placement_threshold: Option<f64> = None;
    let mut pool_chance: Option<f32> = None;
    let mut pool_min_area: Option<usize> = None;
    let mut pool_max_radius: Option<usize> = None;
    let mut pool_basin_depth: Option<usize> = None;
    let mut pool_rim_height: Option<usize> = None;
    let mut pool_water_pct: Option<f32> = None;
    let mut pool_lava_pct: Option<f32> = None;
    let mut pool_empty_pct: Option<f32> = None;
    let mut pool_min_air_above: Option<usize> = None;
    let mut pool_max_cave_height: Option<usize> = None;
    let mut pool_min_floor_thickness: Option<usize> = None;
    let mut pool_min_ground_depth: Option<usize> = None;
    let mut pool_max_y_step: Option<usize> = None;
    let mut pool_footprint_y_tolerance: Option<usize> = None;
    // Formation settings
    let mut formations_enabled: Option<bool> = None;
    let mut form_placement_frequency: Option<f64> = None;
    let mut form_placement_threshold: Option<f64> = None;
    let mut form_stalactite_chance: Option<f32> = None;
    let mut form_stalagmite_chance: Option<f32> = None;
    let mut form_flowstone_chance: Option<f32> = None;
    let mut form_column_chance: Option<f32> = None;
    let mut form_column_max_gap: Option<usize> = None;
    let mut form_length_min: Option<f32> = None;
    let mut form_length_max: Option<f32> = None;
    let mut form_radius_min: Option<f32> = None;
    let mut form_radius_max: Option<f32> = None;
    let mut form_max_radius: Option<f32> = None;
    let mut form_column_radius_min: Option<f32> = None;
    let mut form_column_radius_max: Option<f32> = None;
    let mut form_flowstone_length_min: Option<f32> = None;
    let mut form_flowstone_length_max: Option<f32> = None;
    let mut form_flowstone_thickness: Option<f32> = None;
    let mut form_min_air_gap: Option<usize> = None;
    let mut form_min_clearance: Option<usize> = None;
    let mut form_smoothness: Option<f32> = None;
    // Mega-Column settings
    let mut form_mega_column_chance: Option<f32> = None;
    let mut form_mega_column_min_gap: Option<usize> = None;
    let mut form_mega_column_radius_min: Option<f32> = None;
    let mut form_mega_column_radius_max: Option<f32> = None;
    let mut form_mega_column_noise_strength: Option<f32> = None;
    let mut form_mega_column_ring_frequency: Option<f32> = None;
    // Drapery settings
    let mut form_drapery_chance: Option<f32> = None;
    let mut form_drapery_length_min: Option<f32> = None;
    let mut form_drapery_length_max: Option<f32> = None;
    let mut form_drapery_wave_frequency: Option<f32> = None;
    let mut form_drapery_wave_amplitude: Option<f32> = None;
    // Rimstone Dam settings
    let mut form_rimstone_chance: Option<f32> = None;
    let mut form_rimstone_dam_height_min: Option<f32> = None;
    let mut form_rimstone_dam_height_max: Option<f32> = None;
    let mut form_rimstone_pool_depth: Option<f32> = None;
    let mut form_rimstone_min_slope: Option<f32> = None;
    // Cave Shield settings
    let mut form_shield_chance: Option<f32> = None;
    let mut form_shield_radius_min: Option<f32> = None;
    let mut form_shield_radius_max: Option<f32> = None;
    let mut form_shield_max_tilt: Option<f32> = None;
    let mut form_shield_stalactite_chance: Option<f32> = None;
    // Cauldron settings
    let mut form_cauldron_chance: Option<f32> = None;
    let mut form_cauldron_radius_min: Option<f32> = None;
    let mut form_cauldron_radius_max: Option<f32> = None;
    let mut form_cauldron_depth: Option<f32> = None;
    let mut form_cauldron_lip_height: Option<f32> = None;
    let mut form_cauldron_rim_stal_min: Option<u32> = None;
    let mut form_cauldron_rim_stal_max: Option<u32> = None;
    let mut form_cauldron_rim_stal_scale: Option<f32> = None;
    let mut form_cauldron_floor_noise: Option<f32> = None;
    let mut form_cauldron_water_chance: Option<f32> = None;
    let mut form_cauldron_lava_chance: Option<f32> = None;
    // Stress settings
    let mut stress_gravity: Option<f32> = None;
    let mut stress_lateral: Option<f32> = None;
    let mut stress_vertical: Option<f32> = None;
    let mut stress_prop_radius: Option<u32> = None;
    let mut stress_max_collapse: Option<u32> = None;
    // Sleep collapse settings (strut survival per type)
    let mut collapse_slate: Option<f32> = None;
    let mut collapse_granite: Option<f32> = None;
    let mut collapse_limestone: Option<f32> = None;
    let mut collapse_copper: Option<f32> = None;
    let mut collapse_iron: Option<f32> = None;
    let mut collapse_steel: Option<f32> = None;
    let mut collapse_crystal: Option<f32> = None;
    let mut collapse_stress_mult: Option<f32> = None;
    let mut collapse_max_cascade: Option<u32> = None;
    let mut collapse_rubble: Option<f32> = None;
    // Geological realism toggles
    let mut iron_sedimentary_only: Option<bool> = None;
    let mut iron_depth_fade: Option<bool> = None;
    let mut copper_supergene: Option<bool> = None;
    let mut copper_granite_contact: Option<bool> = None;
    let mut malachite_depth_bias: Option<bool> = None;
    let mut kimberlite_carrot_taper: Option<bool> = None;
    let mut diamond_depth_grade: Option<bool> = None;
    let mut sulfide_gossan_cap: Option<bool> = None;
    let mut sulfide_disseminated: Option<bool> = None;
    let mut pyrite_ore_halo: Option<bool> = None;
    let mut quartz_planar_veins: Option<bool> = None;
    let mut gold_bonanza: Option<bool> = None;
    let mut geode_volcanic_host: Option<bool> = None;
    let mut geode_depth_scaling: Option<bool> = None;
    // Coal
    let mut coal_freq: Option<f64> = None;
    let mut coal_threshold: Option<f64> = None;
    let mut coal_depth_min: Option<f64> = None;
    let mut coal_depth_max: Option<f64> = None;
    let mut coal_sedimentary_host: Option<bool> = None;
    let mut coal_shallow_ceiling: Option<bool> = None;
    let mut coal_depth_enrichment: Option<bool> = None;
    let mut ore_detail_multiplier: Option<u32> = None;
    let mut ore_protrusion: Option<f32> = None;

    for pair in body.split('&') {
        let mut kv = pair.splitn(2, '=');
        let key = kv.next().unwrap_or("");
        let val = kv.next().unwrap_or("");
        match key {
            "seed" => { seed = val.parse().unwrap_or(42); }
            "chunks_x" => { chunks_x = val.parse().unwrap_or(3); }
            "chunks_y" => { chunks_y = val.parse().unwrap_or(3); }
            "chunks_z" => { chunks_z = val.parse().unwrap_or(1); }
            "closed" => { closed = val == "1" || val == "true"; }
            "cavern_freq" => { cavern_freq = val.parse().ok(); }
            "cavern_threshold" => { cavern_threshold = val.parse().ok(); }
            "detail_octaves" => { detail_octaves = val.parse().ok(); }
            "detail_persistence" => { detail_persistence = val.parse().ok(); }
            "warp_amplitude" => { warp_amplitude = val.parse().ok(); }
            "worms_per_region" => { worms_per_region = val.parse().ok(); }
            "worm_radius_min" => { worm_radius_min = val.parse().ok(); }
            "worm_radius_max" => { worm_radius_max = val.parse().ok(); }
            "worm_step_length" => { worm_step_length = val.parse().ok(); }
            "worm_max_steps" => { worm_max_steps = val.parse().ok(); }
            "worm_falloff_power" => { worm_falloff_power = val.parse().ok(); }
            "chunk_size" => { chunk_size = val.parse().ok(); }
            "sandstone_depth" => { sandstone_depth = val.parse().ok(); }
            "granite_depth" => { granite_depth = val.parse().ok(); }
            "basalt_depth" => { basalt_depth = val.parse().ok(); }
            "slate_depth" => { slate_depth = val.parse().ok(); }
            "iron_band_freq" => { iron_band_freq = val.parse().ok(); }
            "iron_noise_freq" => { iron_noise_freq = val.parse().ok(); }
            "iron_perturbation" => { iron_perturbation = val.parse().ok(); }
            "iron_threshold" => { iron_threshold = val.parse().ok(); }
            "copper_freq" => { copper_freq = val.parse().ok(); }
            "copper_threshold" => { copper_threshold = val.parse().ok(); }
            "malachite_freq" => { malachite_freq = val.parse().ok(); }
            "malachite_threshold" => { malachite_threshold = val.parse().ok(); }
            "kimberlite_pipe_freq" => { kimberlite_pipe_freq = val.parse().ok(); }
            "kimberlite_pipe_threshold" => { kimberlite_pipe_threshold = val.parse().ok(); }
            "diamond_freq" => { diamond_freq = val.parse().ok(); }
            "diamond_threshold" => { diamond_threshold = val.parse().ok(); }
            "sulfide_freq" => { sulfide_freq = val.parse().ok(); }
            "sulfide_threshold" => { sulfide_threshold = val.parse().ok(); }
            "tin_threshold" => { tin_threshold = val.parse().ok(); }
            "pyrite_freq" => { pyrite_freq = val.parse().ok(); }
            "pyrite_threshold" => { pyrite_threshold = val.parse().ok(); }
            "quartz_freq" => { quartz_freq = val.parse().ok(); }
            "quartz_threshold" => { quartz_threshold = val.parse().ok(); }
            "gold_threshold" => { gold_threshold = val.parse().ok(); }
            "geode_freq" => { geode_freq = val.parse().ok(); }
            "geode_center_threshold" => { geode_center_threshold = val.parse().ok(); }
            "geode_shell_thickness" => { geode_shell_thickness = val.parse().ok(); }
            "geode_hollow_factor" => { geode_hollow_factor = val.parse().ok(); }
            // Pool settings
            "pools_enabled" => { pools_enabled = Some(val == "1" || val == "true"); }
            "pool_placement_freq" => { pool_placement_freq = val.parse().ok(); }
            "pool_placement_threshold" => { pool_placement_threshold = val.parse().ok(); }
            "pool_chance" => { pool_chance = val.parse().ok(); }
            "pool_min_area" => { pool_min_area = val.parse().ok(); }
            "pool_max_radius" => { pool_max_radius = val.parse().ok(); }
            "pool_basin_depth" => { pool_basin_depth = val.parse().ok(); }
            "pool_rim_height" => { pool_rim_height = val.parse().ok(); }
            "pool_water_pct" => { pool_water_pct = val.parse().ok(); }
            "pool_lava_pct" => { pool_lava_pct = val.parse().ok(); }
            "pool_empty_pct" => { pool_empty_pct = val.parse().ok(); }
            "pool_min_air_above" => { pool_min_air_above = val.parse().ok(); }
            "pool_max_cave_height" => { pool_max_cave_height = val.parse().ok(); }
            "pool_min_floor_thickness" => { pool_min_floor_thickness = val.parse().ok(); }
            "pool_min_ground_depth" => { pool_min_ground_depth = val.parse().ok(); }
            "pool_max_y_step" => { pool_max_y_step = val.parse().ok(); }
            "pool_footprint_y_tolerance" => { pool_footprint_y_tolerance = val.parse().ok(); }
            // Formation settings
            "formations_enabled" => { formations_enabled = Some(val == "1" || val == "true"); }
            "form_placement_frequency" => { form_placement_frequency = val.parse().ok(); }
            "form_placement_threshold" => { form_placement_threshold = val.parse().ok(); }
            "form_stalactite_chance" => { form_stalactite_chance = val.parse().ok(); }
            "form_stalagmite_chance" => { form_stalagmite_chance = val.parse().ok(); }
            "form_flowstone_chance" => { form_flowstone_chance = val.parse().ok(); }
            "form_column_chance" => { form_column_chance = val.parse().ok(); }
            "form_column_max_gap" => { form_column_max_gap = val.parse().ok(); }
            "form_length_min" => { form_length_min = val.parse().ok(); }
            "form_length_max" => { form_length_max = val.parse().ok(); }
            "form_radius_min" => { form_radius_min = val.parse().ok(); }
            "form_radius_max" => { form_radius_max = val.parse().ok(); }
            "form_max_radius" => { form_max_radius = val.parse().ok(); }
            "form_column_radius_min" => { form_column_radius_min = val.parse().ok(); }
            "form_column_radius_max" => { form_column_radius_max = val.parse().ok(); }
            "form_flowstone_length_min" => { form_flowstone_length_min = val.parse().ok(); }
            "form_flowstone_length_max" => { form_flowstone_length_max = val.parse().ok(); }
            "form_flowstone_thickness" => { form_flowstone_thickness = val.parse().ok(); }
            "form_min_air_gap" => { form_min_air_gap = val.parse().ok(); }
            "form_min_clearance" => { form_min_clearance = val.parse().ok(); }
            "form_smoothness" => { form_smoothness = val.parse().ok(); }
            // Mega-Column settings
            "form_mega_column_chance" => { form_mega_column_chance = val.parse().ok(); }
            "form_mega_column_min_gap" => { form_mega_column_min_gap = val.parse().ok(); }
            "form_mega_column_radius_min" => { form_mega_column_radius_min = val.parse().ok(); }
            "form_mega_column_radius_max" => { form_mega_column_radius_max = val.parse().ok(); }
            "form_mega_column_noise_strength" => { form_mega_column_noise_strength = val.parse().ok(); }
            "form_mega_column_ring_frequency" => { form_mega_column_ring_frequency = val.parse().ok(); }
            // Drapery settings
            "form_drapery_chance" => { form_drapery_chance = val.parse().ok(); }
            "form_drapery_length_min" => { form_drapery_length_min = val.parse().ok(); }
            "form_drapery_length_max" => { form_drapery_length_max = val.parse().ok(); }
            "form_drapery_wave_frequency" => { form_drapery_wave_frequency = val.parse().ok(); }
            "form_drapery_wave_amplitude" => { form_drapery_wave_amplitude = val.parse().ok(); }
            // Rimstone Dam settings
            "form_rimstone_chance" => { form_rimstone_chance = val.parse().ok(); }
            "form_rimstone_dam_height_min" => { form_rimstone_dam_height_min = val.parse().ok(); }
            "form_rimstone_dam_height_max" => { form_rimstone_dam_height_max = val.parse().ok(); }
            "form_rimstone_pool_depth" => { form_rimstone_pool_depth = val.parse().ok(); }
            "form_rimstone_min_slope" => { form_rimstone_min_slope = val.parse().ok(); }
            // Cave Shield settings
            "form_shield_chance" => { form_shield_chance = val.parse().ok(); }
            "form_shield_radius_min" => { form_shield_radius_min = val.parse().ok(); }
            "form_shield_radius_max" => { form_shield_radius_max = val.parse().ok(); }
            "form_shield_max_tilt" => { form_shield_max_tilt = val.parse().ok(); }
            "form_shield_stalactite_chance" => { form_shield_stalactite_chance = val.parse().ok(); }
            // Cauldron settings
            "form_cauldron_chance" => { form_cauldron_chance = val.parse().ok(); }
            "form_cauldron_radius_min" => { form_cauldron_radius_min = val.parse().ok(); }
            "form_cauldron_radius_max" => { form_cauldron_radius_max = val.parse().ok(); }
            "form_cauldron_depth" => { form_cauldron_depth = val.parse().ok(); }
            "form_cauldron_lip_height" => { form_cauldron_lip_height = val.parse().ok(); }
            "form_cauldron_rim_stal_min" => { form_cauldron_rim_stal_min = val.parse().ok(); }
            "form_cauldron_rim_stal_max" => { form_cauldron_rim_stal_max = val.parse().ok(); }
            "form_cauldron_rim_stal_scale" => { form_cauldron_rim_stal_scale = val.parse().ok(); }
            "form_cauldron_floor_noise" => { form_cauldron_floor_noise = val.parse().ok(); }
            "form_cauldron_water_chance" => { form_cauldron_water_chance = val.parse().ok(); }
            "form_cauldron_lava_chance" => { form_cauldron_lava_chance = val.parse().ok(); }
            // Stress settings
            "stress_gravity" => { stress_gravity = val.parse().ok(); }
            "stress_lateral" => { stress_lateral = val.parse().ok(); }
            "stress_vertical" => { stress_vertical = val.parse().ok(); }
            "stress_prop_radius" => { stress_prop_radius = val.parse().ok(); }
            "stress_max_collapse" => { stress_max_collapse = val.parse().ok(); }
            // Sleep collapse settings
            "collapse_slate" => { collapse_slate = val.parse().ok(); }
            "collapse_granite" => { collapse_granite = val.parse().ok(); }
            "collapse_limestone" => { collapse_limestone = val.parse().ok(); }
            "collapse_copper" => { collapse_copper = val.parse().ok(); }
            "collapse_iron" => { collapse_iron = val.parse().ok(); }
            "collapse_steel" => { collapse_steel = val.parse().ok(); }
            "collapse_crystal" => { collapse_crystal = val.parse().ok(); }
            "collapse_stress_mult" => { collapse_stress_mult = val.parse().ok(); }
            "collapse_max_cascade" => { collapse_max_cascade = val.parse().ok(); }
            "collapse_rubble" => { collapse_rubble = val.parse().ok(); }
            // Geological realism toggles
            "iron_sedimentary_only" => { iron_sedimentary_only = Some(val == "1" || val == "true"); }
            "iron_depth_fade" => { iron_depth_fade = Some(val == "1" || val == "true"); }
            "copper_supergene" => { copper_supergene = Some(val == "1" || val == "true"); }
            "copper_granite_contact" => { copper_granite_contact = Some(val == "1" || val == "true"); }
            "malachite_depth_bias" => { malachite_depth_bias = Some(val == "1" || val == "true"); }
            "kimberlite_carrot_taper" => { kimberlite_carrot_taper = Some(val == "1" || val == "true"); }
            "diamond_depth_grade" => { diamond_depth_grade = Some(val == "1" || val == "true"); }
            "sulfide_gossan_cap" => { sulfide_gossan_cap = Some(val == "1" || val == "true"); }
            "sulfide_disseminated" => { sulfide_disseminated = Some(val == "1" || val == "true"); }
            "pyrite_ore_halo" => { pyrite_ore_halo = Some(val == "1" || val == "true"); }
            "quartz_planar_veins" => { quartz_planar_veins = Some(val == "1" || val == "true"); }
            "gold_bonanza" => { gold_bonanza = Some(val == "1" || val == "true"); }
            "geode_volcanic_host" => { geode_volcanic_host = Some(val == "1" || val == "true"); }
            "geode_depth_scaling" => { geode_depth_scaling = Some(val == "1" || val == "true"); }
            // Coal
            "coal_freq" => { coal_freq = val.parse().ok(); }
            "coal_threshold" => { coal_threshold = val.parse().ok(); }
            "coal_depth_min" => { coal_depth_min = val.parse().ok(); }
            "coal_depth_max" => { coal_depth_max = val.parse().ok(); }
            "coal_sedimentary_host" => { coal_sedimentary_host = Some(val == "1" || val == "true"); }
            "coal_shallow_ceiling" => { coal_shallow_ceiling = Some(val == "1" || val == "true"); }
            "coal_depth_enrichment" => { coal_depth_enrichment = Some(val == "1" || val == "true"); }
            "ore_detail_multiplier" => { ore_detail_multiplier = val.parse().ok(); }
            "ore_protrusion" => { ore_protrusion = val.parse().ok(); }
            _ => {}
        }
    }

    let chunks_x = chunks_x.min(8);
    let chunks_y = chunks_y.min(8);
    let chunks_z = chunks_z.min(8);

    let mut config = GenerationConfig {
        seed,
        ..Default::default()
    };
    if let Some(v) = chunk_size { config.chunk_size = v.clamp(4, 64); }
    if let Some(v) = cavern_freq { config.noise.cavern_frequency = v; }
    if let Some(v) = cavern_threshold { config.noise.cavern_threshold = v; }
    if let Some(v) = detail_octaves { config.noise.detail_octaves = v; }
    if let Some(v) = detail_persistence { config.noise.detail_persistence = v; }
    if let Some(v) = warp_amplitude { config.noise.warp_amplitude = v; }
    if let Some(v) = worms_per_region { config.worm.worms_per_region = v; }
    if let Some(v) = worm_radius_min { config.worm.radius_min = v; }
    if let Some(v) = worm_radius_max { config.worm.radius_max = v; }
    if let Some(v) = worm_step_length { config.worm.step_length = v; }
    if let Some(v) = worm_max_steps { config.worm.max_steps = v; }
    if let Some(v) = worm_falloff_power { config.worm.falloff_power = v; }
    // Host rock
    if let Some(v) = sandstone_depth { config.ore.host_rock.sandstone_depth = v; }
    if let Some(v) = granite_depth { config.ore.host_rock.granite_depth = v; }
    if let Some(v) = basalt_depth { config.ore.host_rock.basalt_depth = v; }
    if let Some(v) = slate_depth { config.ore.host_rock.slate_depth = v; }
    // Banded iron
    if let Some(v) = iron_band_freq { config.ore.iron.band_frequency = v; }
    if let Some(v) = iron_noise_freq { config.ore.iron.noise_frequency = v; }
    if let Some(v) = iron_perturbation { config.ore.iron.noise_perturbation = v; }
    if let Some(v) = iron_threshold { config.ore.iron.threshold = v; }
    // Copper
    if let Some(v) = copper_freq { config.ore.copper.frequency = v; }
    if let Some(v) = copper_threshold { config.ore.copper.threshold = v; }
    // Malachite
    if let Some(v) = malachite_freq { config.ore.malachite.frequency = v; }
    if let Some(v) = malachite_threshold { config.ore.malachite.threshold = v; }
    // Kimberlite
    if let Some(v) = kimberlite_pipe_freq { config.ore.kimberlite.pipe_frequency_2d = v; }
    if let Some(v) = kimberlite_pipe_threshold { config.ore.kimberlite.pipe_threshold = v; }
    if let Some(v) = diamond_freq { config.ore.kimberlite.diamond_frequency = v; }
    if let Some(v) = diamond_threshold { config.ore.kimberlite.diamond_threshold = v; }
    // Sulfide
    if let Some(v) = sulfide_freq { config.ore.sulfide.frequency = v; }
    if let Some(v) = sulfide_threshold { config.ore.sulfide.threshold = v; }
    if let Some(v) = tin_threshold { config.ore.sulfide.tin_threshold = v; }
    // Pyrite
    if let Some(v) = pyrite_freq { config.ore.pyrite.frequency = v; }
    if let Some(v) = pyrite_threshold { config.ore.pyrite.threshold = v; }
    // Quartz
    if let Some(v) = quartz_freq { config.ore.quartz.frequency = v; config.ore.gold.frequency = v; }
    if let Some(v) = quartz_threshold { config.ore.quartz.threshold = v; }
    // Gold
    if let Some(v) = gold_threshold { config.ore.gold.threshold = v; }
    // Geode
    if let Some(v) = geode_freq { config.ore.geode.frequency = v; }
    if let Some(v) = geode_center_threshold { config.ore.geode.center_threshold = v; }
    if let Some(v) = geode_shell_thickness { config.ore.geode.shell_thickness = v; }
    if let Some(v) = geode_hollow_factor { config.ore.geode.hollow_factor = v; }
    // Geological realism toggles
    if let Some(v) = iron_sedimentary_only { config.ore.iron_sedimentary_only = v; }
    if let Some(v) = iron_depth_fade { config.ore.iron_depth_fade = v; }
    if let Some(v) = copper_supergene { config.ore.copper_supergene = v; }
    if let Some(v) = copper_granite_contact { config.ore.copper_granite_contact = v; }
    if let Some(v) = malachite_depth_bias { config.ore.malachite_depth_bias = v; }
    if let Some(v) = kimberlite_carrot_taper { config.ore.kimberlite_carrot_taper = v; }
    if let Some(v) = diamond_depth_grade { config.ore.diamond_depth_grade = v; }
    if let Some(v) = sulfide_gossan_cap { config.ore.sulfide_gossan_cap = v; }
    if let Some(v) = sulfide_disseminated { config.ore.sulfide_disseminated = v; }
    if let Some(v) = pyrite_ore_halo { config.ore.pyrite_ore_halo = v; }
    if let Some(v) = quartz_planar_veins { config.ore.quartz_planar_veins = v; }
    if let Some(v) = gold_bonanza { config.ore.gold_bonanza = v; }
    if let Some(v) = geode_volcanic_host { config.ore.geode_volcanic_host = v; }
    if let Some(v) = geode_depth_scaling { config.ore.geode_depth_scaling = v; }
    // Coal
    if let Some(v) = coal_freq { config.ore.coal.frequency = v; }
    if let Some(v) = coal_threshold { config.ore.coal.threshold = v; }
    if let Some(v) = coal_depth_min { config.ore.coal.depth_min = v; }
    if let Some(v) = coal_depth_max { config.ore.coal.depth_max = v; }
    if let Some(v) = coal_sedimentary_host { config.ore.coal_sedimentary_host = v; }
    if let Some(v) = coal_shallow_ceiling { config.ore.coal_shallow_ceiling = v; }
    if let Some(v) = coal_depth_enrichment { config.ore.coal_depth_enrichment = v; }
    if let Some(v) = ore_detail_multiplier { config.ore_detail_multiplier = v.max(1).min(4); }
    if let Some(v) = ore_protrusion { config.ore_protrusion = v.max(0.0).min(0.5); }
    // Pool settings
    if let Some(v) = pools_enabled { config.pools.enabled = v; }
    if let Some(v) = pool_placement_freq { config.pools.placement_frequency = v; }
    if let Some(v) = pool_placement_threshold { config.pools.placement_threshold = v; }
    if let Some(v) = pool_chance { config.pools.pool_chance = v; }
    if let Some(v) = pool_min_area { config.pools.min_area = v; }
    if let Some(v) = pool_max_radius { config.pools.max_radius = v; }
    if let Some(v) = pool_basin_depth { config.pools.basin_depth = v; }
    if let Some(v) = pool_rim_height { config.pools.rim_height = v; }
    if let Some(v) = pool_water_pct { config.pools.water_pct = v; }
    if let Some(v) = pool_lava_pct { config.pools.lava_pct = v; }
    if let Some(v) = pool_empty_pct { config.pools.empty_pct = v; }
    if let Some(v) = pool_min_air_above { config.pools.min_air_above = v; }
    if let Some(v) = pool_max_cave_height { config.pools.max_cave_height = v; }
    if let Some(v) = pool_min_floor_thickness { config.pools.min_floor_thickness = v; }
    if let Some(v) = pool_min_ground_depth { config.pools.min_ground_depth = v; }
    if let Some(v) = pool_max_y_step { config.pools.max_y_step = v; }
    if let Some(v) = pool_footprint_y_tolerance { config.pools.footprint_y_tolerance = v; }
    // Formation settings
    if let Some(v) = formations_enabled { config.formations.enabled = v; }
    if let Some(v) = form_placement_frequency { config.formations.placement_frequency = v; }
    if let Some(v) = form_placement_threshold { config.formations.placement_threshold = v; }
    if let Some(v) = form_stalactite_chance { config.formations.stalactite_chance = v; }
    if let Some(v) = form_stalagmite_chance { config.formations.stalagmite_chance = v; }
    if let Some(v) = form_flowstone_chance { config.formations.flowstone_chance = v; }
    if let Some(v) = form_column_chance { config.formations.column_chance = v; }
    if let Some(v) = form_column_max_gap { config.formations.column_max_gap = v; }
    if let Some(v) = form_length_min { config.formations.length_min = v; }
    if let Some(v) = form_length_max { config.formations.length_max = v; }
    if let Some(v) = form_radius_min { config.formations.radius_min = v; }
    if let Some(v) = form_radius_max { config.formations.radius_max = v; }
    if let Some(v) = form_max_radius { config.formations.max_radius = v; }
    if let Some(v) = form_column_radius_min { config.formations.column_radius_min = v; }
    if let Some(v) = form_column_radius_max { config.formations.column_radius_max = v; }
    if let Some(v) = form_flowstone_length_min { config.formations.flowstone_length_min = v; }
    if let Some(v) = form_flowstone_length_max { config.formations.flowstone_length_max = v; }
    if let Some(v) = form_flowstone_thickness { config.formations.flowstone_thickness = v; }
    if let Some(v) = form_min_air_gap { config.formations.min_air_gap = v; }
    if let Some(v) = form_min_clearance { config.formations.min_clearance = v; }
    if let Some(v) = form_smoothness { config.formations.smoothness = v; }
    // Mega-Column settings
    if let Some(v) = form_mega_column_chance { config.formations.mega_column_chance = v; }
    if let Some(v) = form_mega_column_min_gap { config.formations.mega_column_min_gap = v; }
    if let Some(v) = form_mega_column_radius_min { config.formations.mega_column_radius_min = v; }
    if let Some(v) = form_mega_column_radius_max { config.formations.mega_column_radius_max = v; }
    if let Some(v) = form_mega_column_noise_strength { config.formations.mega_column_noise_strength = v; }
    if let Some(v) = form_mega_column_ring_frequency { config.formations.mega_column_ring_frequency = v; }
    // Drapery settings
    if let Some(v) = form_drapery_chance { config.formations.drapery_chance = v; }
    if let Some(v) = form_drapery_length_min { config.formations.drapery_length_min = v; }
    if let Some(v) = form_drapery_length_max { config.formations.drapery_length_max = v; }
    if let Some(v) = form_drapery_wave_frequency { config.formations.drapery_wave_frequency = v; }
    if let Some(v) = form_drapery_wave_amplitude { config.formations.drapery_wave_amplitude = v; }
    // Rimstone Dam settings
    if let Some(v) = form_rimstone_chance { config.formations.rimstone_chance = v; }
    if let Some(v) = form_rimstone_dam_height_min { config.formations.rimstone_dam_height_min = v; }
    if let Some(v) = form_rimstone_dam_height_max { config.formations.rimstone_dam_height_max = v; }
    if let Some(v) = form_rimstone_pool_depth { config.formations.rimstone_pool_depth = v; }
    if let Some(v) = form_rimstone_min_slope { config.formations.rimstone_min_slope = v; }
    // Cave Shield settings
    if let Some(v) = form_shield_chance { config.formations.shield_chance = v; }
    if let Some(v) = form_shield_radius_min { config.formations.shield_radius_min = v; }
    if let Some(v) = form_shield_radius_max { config.formations.shield_radius_max = v; }
    if let Some(v) = form_shield_max_tilt { config.formations.shield_max_tilt = v; }
    if let Some(v) = form_shield_stalactite_chance { config.formations.shield_stalactite_chance = v; }
    // Cauldron settings
    if let Some(v) = form_cauldron_chance { config.formations.cauldron_chance = v; }
    if let Some(v) = form_cauldron_radius_min { config.formations.cauldron_radius_min = v; }
    if let Some(v) = form_cauldron_radius_max { config.formations.cauldron_radius_max = v; }
    if let Some(v) = form_cauldron_depth { config.formations.cauldron_depth = v; }
    if let Some(v) = form_cauldron_lip_height { config.formations.cauldron_lip_height = v; }
    if let Some(v) = form_cauldron_rim_stal_min { config.formations.cauldron_rim_stalagmite_count_min = v; }
    if let Some(v) = form_cauldron_rim_stal_max { config.formations.cauldron_rim_stalagmite_count_max = v; }
    if let Some(v) = form_cauldron_rim_stal_scale { config.formations.cauldron_rim_stalagmite_scale = v; }
    if let Some(v) = form_cauldron_floor_noise { config.formations.cauldron_floor_noise = v; }
    if let Some(v) = form_cauldron_water_chance { config.formations.cauldron_water_chance = v; }
    if let Some(v) = form_cauldron_lava_chance { config.formations.cauldron_lava_chance = v; }

    // Build sleep config from UI overrides (stress settings embedded in sleep config)
    let mut sleep_cfg = SleepConfig::default();
    // Enable aureole + veins for testing, disable reaction/deeptime/accumulation
    sleep_cfg.phase1_enabled = false;  // reaction OFF
    sleep_cfg.phase2_enabled = true;   // aureole ON
    sleep_cfg.phase3_enabled = true;   // veins ON
    sleep_cfg.phase4_enabled = false;  // deeptime OFF
    sleep_cfg.accumulation_enabled = false; // accumulation OFF
    // Stress tuning
    if let Some(v) = stress_gravity { sleep_cfg.stress.gravity_weight = v; }
    if let Some(v) = stress_lateral { sleep_cfg.stress.lateral_support_factor = v; }
    if let Some(v) = stress_vertical { sleep_cfg.stress.vertical_support_factor = v; }
    if let Some(v) = stress_prop_radius { sleep_cfg.stress.propagation_radius = v; }
    if let Some(v) = stress_max_collapse { sleep_cfg.stress.max_collapse_volume = v; }
    // Sleep collapse
    if let Some(v) = collapse_slate { sleep_cfg.collapse.strut_survival[1] = v; }
    if let Some(v) = collapse_granite { sleep_cfg.collapse.strut_survival[2] = v; }
    if let Some(v) = collapse_limestone { sleep_cfg.collapse.strut_survival[3] = v; }
    if let Some(v) = collapse_copper { sleep_cfg.collapse.strut_survival[4] = v; }
    if let Some(v) = collapse_iron { sleep_cfg.collapse.strut_survival[5] = v; }
    if let Some(v) = collapse_steel { sleep_cfg.collapse.strut_survival[6] = v; }
    if let Some(v) = collapse_crystal { sleep_cfg.collapse.strut_survival[7] = v; }
    if let Some(v) = collapse_stress_mult { sleep_cfg.collapse.stress_multiplier = v; }
    if let Some(v) = collapse_max_cascade { sleep_cfg.collapse.max_cascade_iterations = v; }
    if let Some(v) = collapse_rubble { sleep_cfg.collapse.rubble_fill_ratio = v; }

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

    // Collect pool descriptors from the region
    let pool_descriptors = {
        let app = state.lock().unwrap();
        app.region.as_ref().map(|r| r.pool_descriptors.clone()).unwrap_or_default()
    };

    // Serialize and respond
    let output_msg = format!("Generated {}x{}x{} in {:.2?}: {} verts, {} tris, {} pools",
        chunks_x, chunks_y, chunks_z, elapsed, verts, tris, pool_descriptors.len());

    let response_json = serde_json::json!({
        "ok": true,
        "mesh": mesh_json,
        "output": output_msg,
        "pools": pool_descriptors,
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

    // Check pool containment after mining (drainable pools)
    region.check_pool_containment();
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
