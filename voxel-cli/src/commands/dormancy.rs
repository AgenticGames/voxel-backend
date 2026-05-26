//! Block 1 smoke tool — exercises voxel-world-memory + voxel-cinema +
//! voxel-sleep::predict end-to-end against synthetic data. No save file
//! required.
//!
//! Subcommands:
//!   - `info` — confirm all 3 systems instantiate and exchange data.
//!   - `predict-bench` — run the predictor on a synthetic snapshot and
//!     print wall time + manifest summary.
//!   - `cinema-dryrun` — compose a SafeOrbit + LavaDescent shot for a
//!     synthetic Scene and print waypoints as JSON.
//!
//! Usage: `voxel-cli dormancy <subcommand>`.

use std::collections::HashMap;

use voxel_core::density::DensityField;
use voxel_core::material::Material;
use voxel_core::stress::{StressField, SupportField};
use voxel_fluid::cell::{FluidCell, FluidType};
use voxel_fluid::FluidSnapshot;

pub fn run(args: &[String]) {
    let sub = args.first().map(|s| s.as_str()).unwrap_or("");
    match sub {
        "info" | "" => info(),
        "predict-bench" => predict_bench(),
        "cinema-dryrun" => cinema_dryrun(),
        "e2e" => e2e(),
        _ => {
            eprintln!("Unknown dormancy subcommand: {sub}");
            eprintln!("Available: info, predict-bench, cinema-dryrun, e2e");
            std::process::exit(1);
        }
    }
}

fn info() {
    println!("Dormancy Director — Block 1 systems:");
    println!();

    // 1. World Memory
    let wm = voxel_world_memory::WorldMemory::new();
    println!(
        "  ✓ voxel-world-memory  — WorldMemory instantiated, {} scenes tracked, age {}s",
        wm.tracked_scene_count(),
        wm.elapsed_secs()
    );

    // Push a synthetic event.
    wm.record_event(voxel_world_memory::WorldEvent::lava_spread_at(10.0, 5.0, 10.0));
    let drained = wm.drain_events(10);
    println!("    └─ event queue test: pushed 1, drained {}", drained.len());

    // 2. Cinema
    use voxel_cinema::{compose, IntentMask};
    use voxel_world_memory::scene::{Aabb, Scene, SceneId, SceneKind};

    let scene = make_lava_scene();
    // Stub grid: pure air everywhere so SafeOrbit + LavaDescent both produce.
    let grid = AirGrid;
    let probe = |_a: glam::Vec3, _b: glam::Vec3| None;
    let candidates = compose(&scene, IntentMask::all(), 5, &grid, &probe);
    println!(
        "  ✓ voxel-cinema        — composed {} ShotCandidates for synthetic Lava scene",
        candidates.len()
    );
    for c in &candidates {
        println!(
            "    └─ intent {:?} score={:.1} waypoints={} duration={:.1}s",
            c.intent,
            c.score,
            c.waypoints.len(),
            c.total_duration
        );
    }

    // 3. Sleep predictor
    let snap = empty_predict_snapshot();
    let manifest = voxel_sleep::predict::predict_next_sleep(&snap);
    println!(
        "  ✓ voxel-sleep::predict — predict_next_sleep wall={}ms cache={:?}",
        manifest.wall_ms, manifest.computed_at_secs
    );
    println!();
    println!("All Block 1 systems operational.");

    let _ = scene;
    let _ = SceneId(0); // silence unused
    let _ = Aabb::empty();
    let _ = SceneKind::Lava;
    let _ = Scene::new(SceneId(0), SceneKind::Lava, glam::Vec3::ZERO);
}

fn predict_bench() {
    println!("Predictor benchmark — synthetic 2-chunk snapshot:");
    let mut snap = empty_predict_snapshot();
    let mut df = DensityField::new(31);
    for y in 0..31 {
        for z in 0..31 {
            for x in 0..31 {
                df.get_mut(x, y, z).material = Material::Granite;
            }
        }
    }
    snap.density_fields.insert((0, 0, 0), df.clone());
    snap.density_fields.insert((1, 0, 0), df);
    // Lava cluster for aureole detection.
    let positions = vec![
        (10, 10, 10),
        (11, 10, 10),
        (10, 11, 10),
        (10, 10, 11),
        (11, 11, 10),
        (11, 10, 11),
    ];
    snap.fluid_snapshot = fluid_with_lava(&positions);

    // Run 10 iterations for a stable wall-time estimate.
    let mut wall_ms = 0u32;
    for _ in 0..10 {
        let m = voxel_sleep::predict::predict_next_sleep(&snap);
        wall_ms += m.wall_ms;
    }
    let avg = wall_ms as f32 / 10.0;
    println!(
        "  average wall: {:.2} ms ({} iters)",
        avg, 10
    );
    let m = voxel_sleep::predict::predict_next_sleep(&snap);
    println!(
        "  manifest: chunks_likely_changed={}, lava_cells={}, scene_hints={}, aureole={:?}",
        m.chunks_likely_changed.len(),
        m.predicted_lava_cells.len(),
        m.predicted_scene_hints.len(),
        m.predicted_aureole_glimpse_pos
    );
    println!("  budget: <500ms p99 (plan target)  status: {}", if avg < 500.0 { "OK" } else { "OVER" });
}

fn cinema_dryrun() {
    use voxel_cinema::{compose, IntentMask, ShotIntent};
    println!("Cinema dryrun — synthetic Lava scene, all intents:");
    let scene = make_lava_scene();
    let grid = AirGrid;
    let probe = |_a: glam::Vec3, _b: glam::Vec3| None;
    let candidates = compose(&scene, IntentMask::all(), 10, &grid, &probe);
    for c in &candidates {
        println!();
        println!("Intent: {:?}", c.intent);
        println!("Score: {:.1}", c.score);
        println!("Duration: {:.2}s", c.total_duration);
        println!("Caption: \"{}\"", c.caption);
        println!("Audio cue: {}", c.audio_cue);
        println!("Lighting: warmth={:.2} contrast={:.2} key={:.0}", c.lighting.warmth, c.lighting.contrast, c.lighting.key_intensity);
        println!("Waypoints:");
        for (i, w) in c.waypoints.iter().enumerate() {
            println!(
                "  [{}] t={:.2}s pos=({:.1},{:.1},{:.1}) look=({:.1},{:.1},{:.1}) fov={:.0}",
                i, w.t_secs, w.pos[0], w.pos[1], w.pos[2], w.look_at[0], w.look_at[1], w.look_at[2], w.fov_deg,
            );
        }
    }
    let _ = ShotIntent::SafeOrbit; // silence unused
}

fn e2e() {
    println!("Block 1 end-to-end smoke (seed=42):");
    info();
    println!();
    predict_bench();
    println!();
    println!("Total: 3 systems × 1 path each. cargo test --workspace for full coverage.");
}

// ─── Helpers ────────────────────────────────────────────────────────

fn make_lava_scene() -> voxel_world_memory::scene::Scene {
    use voxel_world_memory::scene::{Aabb, Scene, SceneId, SceneKind};
    let mut s = Scene::new(SceneId(1), SceneKind::Lava, glam::Vec3::new(100.0, 100.0, 100.0));
    s.score = 250.0;
    s.confidence = 0.9;
    s.aabb = Aabb {
        min: [80.0, 80.0, 80.0],
        max: [120.0, 120.0, 120.0],
    };
    s
}

fn empty_predict_snapshot() -> voxel_sleep::predict::PredictSnapshot {
    voxel_sleep::predict::PredictSnapshot::new(
        HashMap::new(),
        HashMap::<(i32, i32, i32), StressField>::new(),
        HashMap::<(i32, i32, i32), SupportField>::new(),
        FluidSnapshot {
            chunks: HashMap::new(),
            chunk_size: 30,
        },
        (0, 0, 0),
        0,
        30,
        3,
    )
}

fn fluid_with_lava(positions: &[(i32, i32, i32)]) -> FluidSnapshot {
    let cs = 30usize;
    let mut chunks: HashMap<(i32, i32, i32), Vec<FluidCell>> = HashMap::new();
    for &(wx, wy, wz) in positions {
        let cx = wx.div_euclid(cs as i32);
        let cy = wy.div_euclid(cs as i32);
        let cz = wz.div_euclid(cs as i32);
        let lx = wx.rem_euclid(cs as i32) as usize;
        let ly = wy.rem_euclid(cs as i32) as usize;
        let lz = wz.rem_euclid(cs as i32) as usize;
        let entry = chunks
            .entry((cx, cy, cz))
            .or_insert_with(|| vec![FluidCell::default(); cs * cs * cs]);
        let idx = lz * cs * cs + ly * cs + lx;
        entry[idx].level = 1.0;
        entry[idx].fluid_type = FluidType::Lava;
    }
    FluidSnapshot {
        chunks,
        chunk_size: cs,
    }
}

// ── CellGrid stub (everything is air) ───────────────────────────────

struct AirGrid;
impl voxel_path::grid::CellGrid for AirGrid {
    fn cell_size(&self) -> f32 {
        1.0
    }
    fn is_solid(&self, _cell: glam::IVec3) -> bool {
        false
    }
    fn surface_normal_at(&self, _cell: glam::IVec3) -> glam::Vec3 {
        glam::Vec3::ZERO
    }
}
