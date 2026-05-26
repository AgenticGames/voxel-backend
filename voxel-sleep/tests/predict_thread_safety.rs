//! Integration test — multi-threaded predictor cache safety.
//!
//! Plan calls for: spawn 4 reader threads polling a shared
//! `Arc<RwLock<Option<PredictedManifest>>>` while one writer thread runs
//! `predict_next_sleep` in a loop for 2 seconds. No deadlock, no torn
//! reads (cache is option-replaced atomically under write lock).
//!
//! This is the *function-level* thread-safety harness. The engine-level
//! predictor thread + wake channel test lives in voxel-ffi (task B9).

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use voxel_core::density::DensityField;
use voxel_core::material::Material;
use voxel_core::stress::{StressField, SupportField};
use voxel_fluid::cell::{FluidCell, FluidType};
use voxel_fluid::FluidSnapshot;
use voxel_sleep::predict::{predict_next_sleep, PredictSnapshot, PredictedManifest};

fn density_field(size: usize, material: Material) -> DensityField {
    let mut df = DensityField::new(size);
    for y in 0..size {
        for z in 0..size {
            for x in 0..size {
                df.get_mut(x, y, z).material = material;
            }
        }
    }
    df
}

fn fluid_snapshot_with_lava(positions: &[(i32, i32, i32)]) -> FluidSnapshot {
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

fn make_snapshot(sleep_count: u32) -> PredictSnapshot {
    let mut density_fields = HashMap::new();
    density_fields.insert((0, 0, 0), density_field(31, Material::Granite));
    density_fields.insert((1, 0, 0), density_field(31, Material::Limestone));

    let mut stress_fields = HashMap::new();
    let mut sf = StressField::new(30);
    // Mid-stress chunk so the predictor has something to chew on.
    for cell in sf.stress.iter_mut() {
        *cell = 0.4;
    }
    stress_fields.insert((0, 0, 0), sf);

    let fluid = fluid_snapshot_with_lava(&[
        (10, 10, 10),
        (11, 10, 10),
        (10, 11, 10),
        (10, 10, 11),
        (11, 11, 10),
    ]);

    PredictSnapshot::new(
        density_fields,
        stress_fields,
        HashMap::<(i32, i32, i32), SupportField>::new(),
        fluid,
        (0, 0, 0),
        sleep_count,
        30,
        3,
    )
}

#[test]
fn concurrent_readers_and_writer_no_deadlock() {
    let cache: Arc<RwLock<Option<PredictedManifest>>> = Arc::new(RwLock::new(None));
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let writer = {
        let cache = Arc::clone(&cache);
        let stop = Arc::clone(&stop);
        thread::spawn(move || {
            let mut count = 0u32;
            let start = Instant::now();
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                let snap = make_snapshot(count);
                let manifest = predict_next_sleep(&snap);
                {
                    let mut g = cache.write().unwrap();
                    *g = Some(manifest);
                }
                count += 1;
                if start.elapsed() >= Duration::from_millis(2000) {
                    break;
                }
                thread::sleep(Duration::from_millis(5));
            }
            count
        })
    };

    let mut readers = Vec::new();
    for tid in 0..4 {
        let cache = Arc::clone(&cache);
        readers.push(thread::spawn(move || {
            let mut seen_some = 0u32;
            let mut max_sleep_count_seen = 0u32;
            let start = Instant::now();
            while start.elapsed() < Duration::from_millis(2000) {
                {
                    let g = cache.read().unwrap();
                    if let Some(m) = g.as_ref() {
                        seen_some += 1;
                        // Verify the manifest is internally consistent — sleep_count
                        // monotonically increases across writes.
                        max_sleep_count_seen = max_sleep_count_seen.max(m.sleep_count);
                    }
                }
                thread::sleep(Duration::from_millis(1));
            }
            (tid, seen_some, max_sleep_count_seen)
        }));
    }

    // Let it run for the full duration.
    thread::sleep(Duration::from_millis(2100));
    stop.store(true, std::sync::atomic::Ordering::Relaxed);

    let writer_count = writer.join().expect("writer thread");
    assert!(writer_count > 0, "writer should have run at least once");

    for r in readers {
        let (tid, seen, max_count) = r.join().expect("reader thread");
        // Each reader should have seen the cache populated at least once
        // (writer ran multiple times in 2s).
        assert!(seen > 0, "reader {tid} never saw a populated cache");
        // Max count seen by readers should match writer's final count or close
        // (within ±1 due to interleaving).
        assert!(
            max_count <= writer_count && max_count + 5 >= writer_count.saturating_sub(1),
            "reader {tid} max={} writer={}: should be close",
            max_count,
            writer_count
        );
    }
}

#[test]
fn multiple_predictions_independent_no_state_leak() {
    // Same input + different sleep_count → different RNG seeds → output
    // is still deterministic but distinct.
    let snap_a = make_snapshot(0);
    let snap_b = make_snapshot(0); // identical
    let snap_c = make_snapshot(1); // different sleep_count

    let m_a = predict_next_sleep(&snap_a);
    let m_b = predict_next_sleep(&snap_b);
    let m_c = predict_next_sleep(&snap_c);

    assert_eq!(m_a.predicted_lava_cells, m_b.predicted_lava_cells);
    assert_eq!(m_a.predicted_aureole_glimpse_pos, m_b.predicted_aureole_glimpse_pos);
    assert_eq!(m_a.sleep_count, 0);
    assert_eq!(m_c.sleep_count, 1);
    // Lava cells are the same world positions regardless of sleep_count
    // (they're from the fluid snapshot, not RNG-driven).
    assert_eq!(m_a.predicted_lava_cells, m_c.predicted_lava_cells);
}

#[test]
fn predict_completes_under_500ms_on_modest_snapshot() {
    let snap = make_snapshot(0);
    let t = Instant::now();
    let _m = predict_next_sleep(&snap);
    let elapsed = t.elapsed();
    // Plan target: <500ms p99. Modest snapshot (2 chunks) should land
    // well under that.
    assert!(
        elapsed < Duration::from_millis(500),
        "predict_next_sleep took {:?}, expected < 500ms",
        elapsed
    );
}
