//! Dormancy world collapse (2026-08-25): when dormancy is triggered, every
//! LOADED area sitting at >= 85% effective stress collapses straight to its
//! end state — no slab-fall cinematic, no camera shake. The player finds the
//! cave-ins when they re-explore.
//!
//! Timing contract: this runs in the sleep handler's FAR-work window, i.e.
//! strictly AFTER the reveal curtain rises (`SLEEP_FAR_GO`), on the
//! otherwise-idle worker pool. It adds ZERO time to the montage-critical
//! path (sim / SleepComplete / near-remesh / prebuffer all happen first).
//! The scan reads under a READ lock so morph/streaming reads interleave;
//! only the cascade itself holds the write lock, bounded by a deadline.
//!
//! The collapse-dirty chunks are returned to the sleep handler, which folds
//! them into the existing FAR remesh + seam pass — so meshing rides the
//! established pipeline instead of inventing a new one.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use voxel_core::stress::{
    detect_and_execute_collapses_v2_with_force_deadline, BrokenStrutEvent, OverstressedVoxel,
};

use crate::types::{FfiStrutBroken, WorkerResult};

/// Effective-stress threshold at which an area collapses during dormancy.
/// 0.75 (user call 2026-08-25, was 0.85): deliberately BELOW the 0.85
/// crack-decal / creak tier, so dormancy also drops marginal rock that never
/// showed cracks — geological time is dramatic. Also matches the default
/// `slab_cohesion_threshold` (0.75), so seed clusters and their cohesion
/// expansion agree.
const DORMANCY_COLLAPSE_THRESHOLD: f32 = 0.75;

/// Wall-clock budget for scan + cascade. The pass runs behind the reveal,
/// so this deadline is not a UX budget — it only bounds how long the mine
/// lane can stay pinned on a pathologically stressed world. Partial
/// processing is safe: unprocessed areas simply stay standing.
const DORMANCY_COLLAPSE_BUDGET: Duration = Duration::from_secs(6);

/// Chunks within this Chebyshev radius of the player are exempt. Dropping a
/// slab (or burying the floor in its rubble pile) at the spot the player
/// wakes up on is a no-counterplay death; everything farther out is fair
/// geology.
const PLAYER_EXCLUDE_CHEB: i32 = 2;

/// Scan every loaded chunk for cells at >= 85% effective stress and collapse
/// the resulting slabs to their end state. Returns the set of chunks whose
/// density changed (origin slabs + rubble piles) so the caller can remesh +
/// seam-stitch them through the normal sleep pipeline.
pub(crate) fn apply_dormancy_world_collapse(
    ctx: &super::HandlerCtx<'_>,
    player_chunk: (i32, i32, i32),
) -> Vec<(i32, i32, i32)> {
    let t0 = Instant::now();
    let deadline = t0 + DORMANCY_COLLAPSE_BUDGET;
    let stress_cfg = ctx.stress_config.read().unwrap().clone();
    let chunk_size = ctx.config.read().unwrap().chunk_size;

    // ── Scan (read lock — writers wait, readers don't) ──────────────────
    let (mut seeds, chunks_scanned, skipped_player, skipped_protected) = {
        let s = ctx.store.read().unwrap();
        collect_dormancy_seeds(
            &s.stress_fields,
            &s.density_fields,
            &s.montage_protected,
            player_chunk,
            chunk_size,
        )
    };
    let scan_ms = t0.elapsed().as_secs_f64() * 1000.0;

    if seeds.is_empty() {
        crate::panic_log::note(&format!(
            "[DORMANCY-COLLAPSE] no cells >= {:.2} eff in {} loaded chunks (skipped: {} player-adjacent, {} montage-protected) — scan {:.0}ms, nothing to collapse",
            DORMANCY_COLLAPSE_THRESHOLD, chunks_scanned, skipped_player, skipped_protected, scan_ms
        ));
        return Vec::new();
    }

    // Nearest-first: if the deadline bites, the areas the player will reach
    // soonest are the ones guaranteed to have collapsed.
    let pcw = (
        player_chunk.0 * chunk_size as i32,
        player_chunk.1 * chunk_size as i32,
        player_chunk.2 * chunk_size as i32,
    );
    seeds.sort_by_key(|v| {
        let dx = (v.world_x - pcw.0) as i64;
        let dy = (v.world_y - pcw.1) as i64;
        let dz = (v.world_z - pcw.2) as i64;
        dx * dx + dy * dy + dz * dz
    });

    // ── Cascade (write lock, deadline-bounded) ──────────────────────────
    // Existing v2 slab machinery, straight to end state: slab -> air, pile
    // lands below. halt_at_struts=true so braced areas stay up (struts pay
    // BFS-halt HP as usual); force_collapse=false so grounded rock stays put.
    let t_cascade = Instant::now();
    let mut broken: Vec<BrokenStrutEvent> = Vec::new();
    let (events, hit_deadline) = {
        let mut s = ctx.store.write().unwrap();
        let (density_fields, stress_fields, support_fields) = s.sleep_fields_mut();
        detect_and_execute_collapses_v2_with_force_deadline(
            density_fields,
            stress_fields,
            support_fields,
            &seeds,
            &stress_cfg,
            chunk_size,
            false, // defer_pile — place rubble immediately (end state)
            false, // force_collapse — grounded slabs don't fall
            true,  // halt_at_struts — braced areas survive dormancy
            &mut broken,
            Some(deadline),
        )
    };
    let cascade_ms = t_cascade.elapsed().as_secs_f64() * 1000.0;

    let mut dirty: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut voxels_collapsed = 0usize;
    for ev in &events {
        for k in &ev.affected_chunks {
            dirty.insert(*k);
        }
        for slab in &ev.slabs {
            voxels_collapsed += slab.voxels.len();
        }
    }

    if !dirty.is_empty() {
        let mut s = ctx.store.write().unwrap();
        let dirty_vec: Vec<(i32, i32, i32)> = dirty.iter().copied().collect();
        // Save persistence — without this the cave-ins evaporate on reload.
        s.modification_tracker.mark_dirty_many(&dirty_vec);
        // Collapsed cells carry stale stress (they are air now) and the new
        // cavity edges carry fresh spans: queue deferred position recalcs so
        // warnings + crack decals re-evaluate on the normal post-montage
        // path. no_collapse: this pass IS the collapse — anything the recalc
        // surfaces is latent state, not a new hazard.
        for ev in &events {
            let center = (
                ev.center.0.round() as i32,
                ev.center.1.round() as i32,
                ev.center.2.round() as i32,
            );
            let mut max_cheb = 0i32;
            for slab in &ev.slabs {
                for cv in &slab.voxels {
                    let d = (cv.world_x - center.0)
                        .abs()
                        .max((cv.world_y - center.1).abs())
                        .max((cv.world_z - center.2).abs());
                    max_cheb = max_cheb.max(d);
                    // Pile lands below the slab — cover the landing too.
                    max_cheb = max_cheb.max(d + slab.fall_distance.abs());
                }
            }
            let radius = (max_cheb + 6).min(chunk_size as i32 * 3);
            s.queue_stress_dirty_no_collapse(center, radius);
        }
    }

    // Struts that broke bracing slabs ride the existing StrutsBroken result
    // so UE's placed-strut visuals + crack overlays stay in sync (dedup by
    // world position UE-side, same as the mine path).
    if !broken.is_empty() {
        let cs = chunk_size as i32;
        let ffi_struts: Vec<FfiStrutBroken> = broken
            .iter()
            .map(|ev| FfiStrutBroken {
                world_x: ev.chunk.0 * cs + ev.lx as i32,
                world_y: ev.chunk.1 * cs + ev.ly as i32,
                world_z: ev.chunk.2 * cs + ev.lz as i32,
                support_type: ev.support_type as u8,
                source: 1, // BFS halt
                _pad: [0; 2],
            })
            .collect();
        let _ = ctx.result_tx.send(WorkerResult::StrutsBroken { struts: ffi_struts });
    }

    // Per-event locations (UE coords) — verification + forensics: BugItGo to
    // center_ue and the cave-in should be visible.
    let ws = ctx.world_scale;
    for (i, ev) in events.iter().enumerate() {
        let vox: usize = ev.slabs.iter().map(|sl| sl.voxels.len()).sum();
        let fall = ev.slabs.first().map(|sl| sl.fall_distance).unwrap_or(0);
        crate::panic_log::note(&format!(
            "[DORMANCY-COLLAPSE-EVENT] {}: center_rust=({:.0},{:.0},{:.0}) center_ue=({:.0},{:.0},{:.0}) voxels={} fall={} chunks={}",
            i, ev.center.0, ev.center.1, ev.center.2,
            ev.center.0 * ws, -ev.center.2 * ws, ev.center.1 * ws,
            vox, fall, ev.affected_chunks.len()
        ));
    }

    crate::panic_log::note(&format!(
        "[DORMANCY-COLLAPSE] {} seeds in {} chunks (skipped: {} player-adjacent, {} montage-protected) -> {} collapses, {} voxels, {} dirty chunks, {} struts broken — scan {:.0}ms cascade {:.0}ms{}",
        seeds.len(), chunks_scanned, skipped_player, skipped_protected,
        events.len(), voxels_collapsed, dirty.len(), broken.len(),
        scan_ms, cascade_ms,
        if hit_deadline { " (HIT DEADLINE — remaining areas left standing)" } else { "" }
    ));

    dirty.into_iter().collect()
}

/// Pure scan: every solid cell at >= `DORMANCY_COLLAPSE_THRESHOLD` effective
/// stress across the given fields, excluding player-adjacent and
/// montage-protected chunks. Returns (seeds, chunks_scanned, skipped_player,
/// skipped_protected). Split out of the store plumbing for testability.
fn collect_dormancy_seeds(
    stress_fields: &std::collections::HashMap<(i32, i32, i32), voxel_core::stress::StressField>,
    density_fields: &std::collections::HashMap<(i32, i32, i32), voxel_core::density::DensityField>,
    montage_protected: &HashSet<(i32, i32, i32)>,
    player_chunk: (i32, i32, i32),
    chunk_size: usize,
) -> (Vec<OverstressedVoxel>, usize, usize, usize) {
    let mut seeds = Vec::new();
    let mut chunks_scanned = 0usize;
    let mut skipped_player = 0usize;
    let mut skipped_protected = 0usize;

    for (&key, sf) in stress_fields.iter() {
        let cheb = (key.0 - player_chunk.0)
            .abs()
            .max((key.1 - player_chunk.1).abs())
            .max((key.2 - player_chunk.2).abs());
        if cheb <= PLAYER_EXCLUDE_CHEB {
            skipped_player += 1;
            continue;
        }
        if montage_protected.contains(&key) {
            skipped_protected += 1;
            continue;
        }
        let Some(df) = density_fields.get(&key) else {
            continue;
        };
        chunks_scanned += 1;
        // 0..chunk_size (not the +1 overlap row) — overlap cells are owned
        // by the neighbor chunk's scan.
        for z in 0..chunk_size {
            for y in 0..chunk_size {
                for x in 0..chunk_size {
                    let eff = sf.effective(x, y, z);
                    if eff < DORMANCY_COLLAPSE_THRESHOLD {
                        continue;
                    }
                    if !df.get(x, y, z).material.is_solid() {
                        continue;
                    }
                    seeds.push(OverstressedVoxel {
                        world_x: key.0 * chunk_size as i32 + x as i32,
                        world_y: key.1 * chunk_size as i32 + y as i32,
                        world_z: key.2 * chunk_size as i32 + z as i32,
                        stress: eff,
                    });
                }
            }
        }
    }
    (seeds, chunks_scanned, skipped_player, skipped_protected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use voxel_core::density::DensityField;
    use voxel_core::material::Material;
    use voxel_core::stress::StressField;

    const CHUNK_SIZE: usize = 16;
    const FIELD_SIZE: usize = 17;

    fn solid_density() -> DensityField {
        let mut df = DensityField::new(FIELD_SIZE);
        for sample in df.samples.iter_mut() {
            sample.density = 1.0;
            sample.material = Material::Granite;
        }
        df
    }

    fn world(
        keys: &[(i32, i32, i32)],
    ) -> (
        HashMap<(i32, i32, i32), StressField>,
        HashMap<(i32, i32, i32), DensityField>,
    ) {
        let mut sf = HashMap::new();
        let mut df = HashMap::new();
        for &k in keys {
            sf.insert(k, StressField::new(FIELD_SIZE));
            df.insert(k, solid_density());
        }
        (sf, df)
    }

    #[test]
    fn threshold_is_respected() {
        // Player far away at (10,10,10) so nothing is player-excluded.
        let (mut sf, df) = world(&[(0, 0, 0)]);
        {
            let f = sf.get_mut(&(0, 0, 0)).unwrap();
            f.set(3, 3, 3, 0.74); // just under — must NOT seed
            f.set(5, 5, 5, 0.76); // over — must seed
        }
        let (seeds, scanned, _, _) = collect_dormancy_seeds(
            &sf, &df, &HashSet::new(), (10, 10, 10), CHUNK_SIZE,
        );
        assert_eq!(scanned, 1);
        assert_eq!(seeds.len(), 1, "only the 0.76 cell should seed");
        assert_eq!(
            (seeds[0].world_x, seeds[0].world_y, seeds[0].world_z),
            (5, 5, 5)
        );
    }

    #[test]
    fn painted_stress_counts_toward_effective() {
        let (mut sf, df) = world(&[(0, 0, 0)]);
        {
            let f = sf.get_mut(&(0, 0, 0)).unwrap();
            f.set(4, 4, 4, 0.50);
            f.set_painted(4, 4, 4, 0.40); // eff 0.90 — must seed
        }
        let (seeds, _, _, _) = collect_dormancy_seeds(
            &sf, &df, &HashSet::new(), (10, 10, 10), CHUNK_SIZE,
        );
        assert_eq!(seeds.len(), 1, "base+painted crossing 0.85 should seed");
    }

    #[test]
    fn air_cells_do_not_seed() {
        let (mut sf, mut df) = world(&[(0, 0, 0)]);
        sf.get_mut(&(0, 0, 0)).unwrap().set(6, 6, 6, 0.95);
        {
            let d = df.get_mut(&(0, 0, 0)).unwrap();
            let s = d.get_mut(6, 6, 6);
            s.density = -1.0;
            s.material = Material::Air;
        }
        let (seeds, _, _, _) = collect_dormancy_seeds(
            &sf, &df, &HashSet::new(), (10, 10, 10), CHUNK_SIZE,
        );
        assert!(seeds.is_empty(), "stale stress on an air cell must not seed");
    }

    #[test]
    fn player_adjacent_chunks_are_exempt() {
        let (mut sf, df) = world(&[(0, 0, 0), (5, 0, 0)]);
        sf.get_mut(&(0, 0, 0)).unwrap().set(3, 3, 3, 0.95);
        sf.get_mut(&(5, 0, 0)).unwrap().set(3, 3, 3, 0.95);
        // Player at (1,0,0): chunk (0,0,0) is cheb 1 <= 2 -> exempt;
        // chunk (5,0,0) is cheb 4 -> eligible.
        let (seeds, scanned, skipped_player, _) = collect_dormancy_seeds(
            &sf, &df, &HashSet::new(), (1, 0, 0), CHUNK_SIZE,
        );
        assert_eq!(skipped_player, 1);
        assert_eq!(scanned, 1);
        assert_eq!(seeds.len(), 1);
        assert_eq!(seeds[0].world_x, 5 * CHUNK_SIZE as i32 + 3);
    }

    #[test]
    fn montage_protected_chunks_are_exempt() {
        let (mut sf, df) = world(&[(4, 0, 0), (5, 0, 0)]);
        sf.get_mut(&(4, 0, 0)).unwrap().set(3, 3, 3, 0.95);
        sf.get_mut(&(5, 0, 0)).unwrap().set(3, 3, 3, 0.95);
        let mut protected = HashSet::new();
        protected.insert((4, 0, 0));
        let (seeds, scanned, _, skipped_protected) = collect_dormancy_seeds(
            &sf, &df, &protected, (20, 20, 20), CHUNK_SIZE,
        );
        assert_eq!(skipped_protected, 1);
        assert_eq!(scanned, 1);
        assert_eq!(seeds.len(), 1);
        assert_eq!(seeds[0].world_x, 5 * CHUNK_SIZE as i32 + 3);
    }
}
