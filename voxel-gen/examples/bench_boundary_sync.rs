//! Microbench for `region_gen::sync_region_boundary_densities` — the intra-region
//! boundary-sync pass run once per region build on the worker generate path
//! (`voxel-ffi/src/worker/generate.rs`), plus the two `region_gen` region builders.
//!
//! The function scans, for every chunk key, its forward neighbors and re-hashes
//! the SAME two `(i32,i32,i32)` keys (`f_a` for the key, `f_b` for the neighbor)
//! INSIDE its inner cell loops:
//!   - Pass 1 (min-rule): `density_fields[&key]` + `density_fields[&neighbor]` per
//!     boundary pair — (gs+1)^2 pairs for each of the 3 face offsets.
//!   - Pass 2 (gradient-blend): `&density_fields[&key]` + `&density_fields[&neighbor]`
//!     per (u,v) — (gs+1)^2 per face offset.
//! Both keys are loop-invariant; the optimization hoists them out (Pass 1: `f_a`
//! once per key, `f_b` once per neighbor via a single `.get`; Pass 2: same).
//!
//! Builds a 3^3 block of adjacent chunks from a sinusoidal terrain sheet (cs=30,
//! the live UE override) so the center chunk has all 26 neighbors present and the
//! boundary cells carry varied solid/air densities (exercising the cliff branch).
//! git-stash the diff to get the baseline side of the A/B.
//!
//! Run: `cargo run --release -p voxel-gen --example bench_boundary_sync`

use std::collections::HashMap;
use std::time::Instant;
use voxel_core::density::DensityField;
use voxel_core::material::Material;

/// Sinusoidal terrain SDF in WORLD space; positive (solid) below the surface so a
/// fraction of boundary cells are strongly solid next to air — the cliff case the
/// gradient-blend pass targets.
fn terrain_density(x: f32, y: f32, z: f32) -> f32 {
    let h = 30.0
        + 6.0 * (x * 0.45).sin()
        + 4.0 * (z * 0.37).cos()
        + 3.0 * ((x + z) * 0.21).sin();
    // Below surface => positive (solid), above => negative (air); clamp to a
    // sharp-ish band so adjacent cells can straddle the cliff threshold.
    (h - y).clamp(-1.0, 1.0)
}

fn build_chunk(cx: i32, cy: i32, cz: i32, gs: usize) -> DensityField {
    let mut f = DensityField::new(gs + 1); // size = chunk_size + 1
    let ox = cx * gs as i32;
    let oy = cy * gs as i32;
    let oz = cz * gs as i32;
    for z in 0..=gs {
        for y in 0..=gs {
            for x in 0..=gs {
                let d = terrain_density(
                    (ox + x as i32) as f32,
                    (oy + y as i32) as f32,
                    (oz + z as i32) as f32,
                );
                let s = f.get_mut(x, y, z);
                s.density = d;
                s.material = if d > 0.0 { Material::Limestone } else { Material::Air };
            }
        }
    }
    f
}

fn build_region(n: i32, gs: usize) -> HashMap<(i32, i32, i32), DensityField> {
    let mut all = HashMap::new();
    for cz in 0..n {
        for cy in 0..n {
            for cx in 0..n {
                all.insert((cx, cy, cz), build_chunk(cx, cy, cz, gs));
            }
        }
    }
    all
}

/// Whole-region density+material checksum. NOTE: this is NOT stable across
/// processes — `sync_region_boundary_densities` iterates `density_fields.keys()`
/// unsorted, and shared edge/corner cells receive multiple updates whose apply
/// order follows that (per-process random) iteration order, so the post-sync
/// field is intrinsically order-dependent. The bit-identity of the hoist is
/// proved instead by the in-process `boundary_sync_hoist_is_bit_identical` unit
/// test (clones preserve iteration order); this checksum is just a sanity print.
fn checksum(all: &HashMap<(i32, i32, i32), DensityField>) -> u64 {
    let mut keys: Vec<_> = all.keys().copied().collect();
    keys.sort();
    let mut acc = 0u64;
    for k in keys {
        let f = &all[&k];
        for s in &f.samples {
            acc = acc
                .wrapping_mul(1099511628211)
                .wrapping_add(s.density.to_bits() as u64)
                .wrapping_add(s.material as u64);
        }
    }
    acc
}

fn main() {
    use voxel_gen::region_gen::sync_region_boundary_densities;
    let gs: usize = 30; // live cs=30 override
    let n: i32 = 3; // 3^3 block => center chunk has all 26 neighbors present

    // Correctness/identity check: sync a fresh region once, print its checksum.
    let mut probe = build_region(n, gs);
    sync_region_boundary_densities(&mut probe, gs);
    println!(
        "grid_size={gs}  region={n}^3 chunks  post-sync checksum={}",
        checksum(&probe)
    );

    // Timing: one sync per timed iteration on a freshly-cloned pristine region so
    // every call does identical full-scan work (clone cost is measured separately
    // and subtracted).
    let pristine = build_region(n, gs);
    let calls = 200usize;
    let rounds = 6usize;

    // Measure clone overhead alone so we can subtract it.
    let mut clone_best = f64::INFINITY;
    let mut clone_sink = 0u64;
    for _ in 0..rounds {
        let t = Instant::now();
        for _ in 0..calls {
            let m = pristine.clone();
            clone_sink = clone_sink.wrapping_add(m.len() as u64);
        }
        clone_best = clone_best.min(t.elapsed().as_secs_f64() * 1e6 / calls as f64);
    }

    let mut best = f64::INFINITY;
    let mut sink = 0u64;
    for r in 0..rounds {
        let t = Instant::now();
        for _ in 0..calls {
            let mut m = pristine.clone();
            sync_region_boundary_densities(&mut m, gs);
            sink = sink.wrapping_add(m.len() as u64);
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / calls as f64;
        println!("round {r}: {us:.3} us/call (incl. clone)");
        best = best.min(us);
    }
    let net = best - clone_best;
    println!(
        "BEST: {best:.3} us/call incl clone  -  clone {clone_best:.3} us/call  =  {net:.3} us/call NET  (sink={sink}, clone_sink={clone_sink})"
    );
}
