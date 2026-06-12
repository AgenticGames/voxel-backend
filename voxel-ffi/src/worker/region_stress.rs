//! Deferred region-level VFX stress pre-population.
//!
//! `handle_generate`'s slow path PUSHES freshly generated regions here instead
//! of computing stress inline: the inline version occupied every worker for
//! ~5-6s right in the middle of the visible loading window (the loading
//! screen's chunk counter sat frozen, then "shot to done" — user report
//! 2026-06-12). Workers DRAIN this queue only when the generate queue has
//! gone idle (the `recv_timeout` Timeout branch of `worker_loop`), so the
//! compute never competes with chunk generation.
//!
//! Two more bounds keep it polite:
//! - `MAX_ACTIVE` workers compute concurrently — the rest of the pool stays
//!   free for mine requests that arrive mid-compute.
//! - The per-chunk voxel pass runs through rayon (`recalc_chunk_stress_voxels`
//!   is data-race-free per chunk), cutting a region from ~5.7s serial to well
//!   under a second of wall time, so even a mid-compute mine request is only
//!   briefly behind one busy worker.
//!
//! Everything else (snapshot-in/short-lock-commit pattern, VFX-only
//! no-collapse guarantee, painted-stress preservation, the
//! `VOXEL_STRESS_VFX_DIAG` diagnostic) carries over unchanged from the
//! inline version — see the doc block at the push site in `generate.rs`.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use rayon::prelude::*;
use voxel_gen::config::{GenerationConfig, StressConfig};

use crate::store::ChunkStore;

type ChunkKey = (i32, i32, i32);

/// Shared FIFO of freshly generated regions awaiting their VFX stress
/// pre-population, plus the concurrency gate.
pub struct DeferredRegionStress {
    queue: Mutex<VecDeque<(ChunkKey, Vec<ChunkKey>)>>,
    active: AtomicUsize,
}

impl DeferredRegionStress {
    /// Max workers computing region stress at once. Keeps most of the pool
    /// free for mine/generate requests that arrive mid-drain.
    const MAX_ACTIVE: usize = 2;

    pub fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            active: AtomicUsize::new(0),
        }
    }

    /// Queue a freshly generated region. Called from the slow path of
    /// `handle_generate` (still holding nothing — push is O(1)).
    pub fn push(&self, region_key: ChunkKey, coords: Vec<ChunkKey>) {
        self.queue.lock().unwrap().push_back((region_key, coords));
    }

    pub fn is_empty(&self) -> bool {
        self.queue.lock().unwrap().is_empty()
    }
}

impl Default for DeferredRegionStress {
    fn default() -> Self {
        Self::new()
    }
}

/// Pop and compute ONE deferred region, respecting the concurrency cap.
/// Returns true if a region was processed (caller `continue`s its loop).
pub(super) fn try_process_deferred_region_stress(
    deferred: &DeferredRegionStress,
    store: &Arc<RwLock<ChunkStore>>,
    stress_config: &Arc<RwLock<StressConfig>>,
    config: &Arc<RwLock<GenerationConfig>>,
) -> bool {
    if deferred.is_empty() {
        return false;
    }
    // Concurrency gate: claim a slot before popping so a burst of idle
    // workers can't all dive into multi-hundred-ms computes at once.
    let prev = deferred.active.fetch_add(1, Ordering::AcqRel);
    if prev >= DeferredRegionStress::MAX_ACTIVE {
        deferred.active.fetch_sub(1, Ordering::AcqRel);
        return false;
    }

    let item = deferred.queue.lock().unwrap().pop_front();
    let result = if let Some((rk, coords)) = item {
        compute_region_stress(rk, &coords, store, stress_config, config);
        true
    } else {
        false
    };

    deferred.active.fetch_sub(1, Ordering::AcqRel);
    result
}

/// The actual region compute. Lock pattern: SNAPSHOT inputs under a short
/// read hold (region + 1-chunk ring — span search reaches 20 voxels, air
/// distance 2, strut radius 5, all under one 30-voxel chunk), run the span
/// search LOCK-FREE with the per-chunk pass parallelized over rayon, then
/// commit with a brief write hold.
///
/// Commit race: a mine during the unlocked compute could write fresher
/// stress that we then overwrite with load-time values. Accepted — the
/// mining queue recalcs the area on the next strike and the UE-side delayed
/// crack refreshes self-heal; not worth a version stamp.
fn compute_region_stress(
    rk: ChunkKey,
    region_coords: &[ChunkKey],
    store: &Arc<RwLock<ChunkStore>>,
    stress_config: &Arc<RwLock<StressConfig>>,
    config: &Arc<RwLock<GenerationConfig>>,
) {
    use voxel_core::stress::{
        ground_connectivity_pass, recalc_chunk_stress_voxels, StressField,
    };

    if region_coords.is_empty() {
        return;
    }

    let t_stress = Instant::now();
    let stress_cfg = stress_config.read().unwrap().clone();
    let chunk_size = config.read().unwrap().chunk_size;

    // Ring = region bounding box expanded by 1 chunk.
    let (mut min_c, mut max_c) = (region_coords[0], region_coords[0]);
    for k in region_coords {
        min_c = (min_c.0.min(k.0), min_c.1.min(k.1), min_c.2.min(k.2));
        max_c = (max_c.0.max(k.0), max_c.1.max(k.1), max_c.2.max(k.2));
    }

    // Snapshot under a short read hold (clones only).
    let (density_snap, support_snap, local, present) = {
        let s = store.read().unwrap();

        let present: Vec<ChunkKey> = region_coords
            .iter()
            .copied()
            .filter(|k| s.density_fields.contains_key(k))
            .collect();
        if present.is_empty() {
            return;
        }

        let mut density_snap = HashMap::new();
        let mut support_snap = HashMap::new();
        for cz in (min_c.2 - 1)..=(max_c.2 + 1) {
            for cy in (min_c.1 - 1)..=(max_c.1 + 1) {
                for cx in (min_c.0 - 1)..=(max_c.0 + 1) {
                    let k = (cx, cy, cz);
                    if let Some(df) = s.density_fields.get(&k) {
                        density_snap.insert(k, df.clone());
                    }
                    if let Some(sf) = s.support_fields.get(&k) {
                        support_snap.insert(k, sf.clone());
                    }
                }
            }
        }
        let mut local: Vec<(ChunkKey, StressField)> = Vec::with_capacity(present.len());
        for k in &present {
            match s.stress_fields.get(k) {
                // Clone preserves save-restored painted-stress (the voxel
                // pass's set() never touches the painted layer).
                Some(existing) => local.push((*k, existing.clone())),
                // store.insert() creates blanks, but stay robust.
                None => local.push((*k, StressField::new(chunk_size + 1))),
            }
        }
        (density_snap, support_snap, local, present)
    };

    // Connectivity flood once over the whole region (cheap, serial), then
    // the expensive per-chunk span pass fans out over rayon. Each closure
    // writes only its own StressField; all map inputs are read-only.
    let scores = ground_connectivity_pass(&density_snap, &present, chunk_size, &stress_cfg);
    let computed: Vec<(ChunkKey, StressField)> = local
        .into_par_iter()
        .map(|(k, mut sf)| {
            let mut discarded_overstressed = Vec::new();
            recalc_chunk_stress_voxels(
                &density_snap,
                &support_snap,
                &scores,
                &stress_cfg,
                k,
                chunk_size,
                Some(&mut sf),
                &[],
                // VFX-only: the overstressed list is deliberately DISCARDED.
                // Collapse stays exclusive to the mining stress queue.
                &mut discarded_overstressed,
            );
            (k, sf)
        })
        .collect();

    // Env-gated diagnostic: one line per region with the effective-stress
    // distribution, so "why are there no cracks here at load" is answerable
    // from a file. Zero cost unless VOXEL_STRESS_VFX_DIAG is set to a path.
    if let Ok(diag_path) = std::env::var("VOXEL_STRESS_VFX_DIAG") {
        let gs = chunk_size + 1;
        let mut ge10 = 0u32;
        let mut ge15 = 0u32;
        let mut max_eff = 0.0f32;
        for (_, sf) in &computed {
            for z in 0..gs {
                for y in 0..gs {
                    for x in 0..gs {
                        let e = sf.effective(x, y, z);
                        if e > max_eff {
                            max_eff = e;
                        }
                        if e >= 1.0 {
                            ge10 += 1;
                        }
                        if e >= 1.5 {
                            ge15 += 1;
                        }
                    }
                }
            }
        }
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&diag_path)
        {
            let _ = writeln!(
                f,
                "[STRESS-VFX] region=({},{},{}) chunks={} ge1.0={} ge1.5={} max_eff={:.2} compute_ms={:.1} (deferred)",
                rk.0, rk.1, rk.2,
                computed.len(),
                ge10, ge15, max_eff,
                t_stress.elapsed().as_secs_f64() * 1000.0
            );
        }
    }

    // Brief write hold: commit the region's fields.
    let mut s = store.write().unwrap();
    for (k, sf) in computed {
        s.stress_fields.insert(k, sf);
    }
}
