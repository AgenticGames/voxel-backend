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

        let result = match (method.as_str(), url.as_str()) {
            ("GET", "/") => serve_static(request, INDEX_HTML, "text/html"),
            ("GET", "/static/app.js") => serve_static(request, APP_JS, "application/javascript"),
            ("GET", "/static/style.css") => serve_static(request, STYLE_CSS, "text/css"),
            ("GET", "/api/report") => serve_report(request, &output_dir),
            ("GET", "/api/obj-files") => serve_obj_file_list(request, &output_dir),
            ("GET", path) if path.starts_with("/api/obj-file/") => serve_obj_by_name(request, path, &output_dir),
            ("DELETE", path) if path.starts_with("/api/obj-file/") => delete_obj_file(request, path, &output_dir),
            ("GET", path) if path.starts_with("/api/obj/") => serve_obj(request, path, &output_dir),
            ("POST", "/api/generate") => serve_generate(request, &state),
            ("POST", "/api/mine") => serve_mine(request, &state),
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
    let mut worms_per_region: Option<u32> = None;
    let mut worm_radius_min: Option<f32> = None;
    let mut worm_radius_max: Option<f32> = None;
    let mut worm_step_length: Option<f32> = None;
    let mut worm_max_steps: Option<u32> = None;
    let mut worm_falloff_power: Option<f32> = None;
    let mut chunk_size: Option<usize> = None;
    let mut max_edge_length: Option<f32> = None;
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
    let mut pool_lava_fraction: Option<f32> = None;
    let mut pool_lava_depth_max: Option<f64> = None;
    let mut pool_min_air_above: Option<usize> = None;
    // Formation settings
    let mut formations_enabled: Option<bool> = None;
    let mut form_placement_threshold: Option<f64> = None;
    let mut form_stalactite_chance: Option<f32> = None;
    let mut form_stalagmite_chance: Option<f32> = None;
    let mut form_flowstone_chance: Option<f32> = None;
    let mut form_column_chance: Option<f32> = None;
    let mut form_length_min: Option<f32> = None;
    let mut form_length_max: Option<f32> = None;
    let mut form_max_radius: Option<f32> = None;
    let mut form_min_air_gap: Option<usize> = None;
    let mut form_min_clearance: Option<usize> = None;
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
            "max_edge_length" => { max_edge_length = val.parse().ok(); }
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
            "pool_lava_fraction" => { pool_lava_fraction = val.parse().ok(); }
            "pool_lava_depth_max" => { pool_lava_depth_max = val.parse().ok(); }
            "pool_min_air_above" => { pool_min_air_above = val.parse().ok(); }
            // Formation settings
            "formations_enabled" => { formations_enabled = Some(val == "1" || val == "true"); }
            "form_placement_threshold" => { form_placement_threshold = val.parse().ok(); }
            "form_stalactite_chance" => { form_stalactite_chance = val.parse().ok(); }
            "form_stalagmite_chance" => { form_stalagmite_chance = val.parse().ok(); }
            "form_flowstone_chance" => { form_flowstone_chance = val.parse().ok(); }
            "form_column_chance" => { form_column_chance = val.parse().ok(); }
            "form_length_min" => { form_length_min = val.parse().ok(); }
            "form_length_max" => { form_length_max = val.parse().ok(); }
            "form_max_radius" => { form_max_radius = val.parse().ok(); }
            "form_min_air_gap" => { form_min_air_gap = val.parse().ok(); }
            "form_min_clearance" => { form_min_clearance = val.parse().ok(); }
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
    if let Some(v) = max_edge_length { config.max_edge_length = v.max(0.5); }
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
    // Pool settings
    if let Some(v) = pools_enabled { config.pools.enabled = v; }
    if let Some(v) = pool_placement_freq { config.pools.placement_frequency = v; }
    if let Some(v) = pool_placement_threshold { config.pools.placement_threshold = v; }
    if let Some(v) = pool_chance { config.pools.pool_chance = v; }
    if let Some(v) = pool_min_area { config.pools.min_area = v; }
    if let Some(v) = pool_max_radius { config.pools.max_radius = v; }
    if let Some(v) = pool_basin_depth { config.pools.basin_depth = v; }
    if let Some(v) = pool_rim_height { config.pools.rim_height = v; }
    if let Some(v) = pool_lava_fraction { config.pools.lava_fraction = v; }
    if let Some(v) = pool_lava_depth_max { config.pools.lava_depth_max = v; }
    if let Some(v) = pool_min_air_above { config.pools.min_air_above = v; }
    // Formation settings
    if let Some(v) = formations_enabled { config.formations.enabled = v; }
    if let Some(v) = form_placement_threshold { config.formations.placement_threshold = v; }
    if let Some(v) = form_stalactite_chance { config.formations.stalactite_chance = v; }
    if let Some(v) = form_stalagmite_chance { config.formations.stalagmite_chance = v; }
    if let Some(v) = form_flowstone_chance { config.formations.flowstone_chance = v; }
    if let Some(v) = form_column_chance { config.formations.column_chance = v; }
    if let Some(v) = form_length_min { config.formations.length_min = v; }
    if let Some(v) = form_length_max { config.formations.length_max = v; }
    if let Some(v) = form_max_radius { config.formations.max_radius = v; }
    if let Some(v) = form_min_air_gap { config.formations.min_air_gap = v; }
    if let Some(v) = form_min_clearance { config.formations.min_clearance = v; }

    // Build sleep config from UI overrides (stress settings embedded in sleep config)
    let mut sleep_cfg = SleepConfig::default();
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

    let mine_result = match mode {
        "peel" => region.mine_peel(center, normal, radius),
        _ => region.mine_sphere(center, radius),
    };

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
    println!("  Sleep completed in {:.2?}: {} chunks changed, {} metamorphosed, {} minerals grown, {} supports degraded, {} collapses",
        elapsed,
        sleep_result.chunks_changed,
        sleep_result.voxels_metamorphosed,
        sleep_result.minerals_grown,
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
            "voxels_metamorphosed": sleep_result.voxels_metamorphosed,
            "minerals_grown": sleep_result.minerals_grown,
            "supports_degraded": sleep_result.supports_degraded,
            "collapses_triggered": sleep_result.collapses_triggered,
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
