//! Per-chunk diagnostic dump — generates a multi-line plain-text report
//! describing a chunk's identity, state, neighbors, density profile, mesh
//! stats, coplanar-tri distribution, edit state, boundary slices, **seam
//! consistency with all 26 neighbors**, internal cliffs, hermite/stress
//! data, and region worm-path context. Used by the UE creative-mode
//! "Chunk Diagnostic" component so the user can copy-paste a problem
//! chunk's stats into chat without grepping log files.
//!
//! Format goals: easy to read, easy to copy-paste (no fancy Unicode that
//! breaks chat clients), with inline `⚠` hints when something looks
//! suspicious so the AI on the other side can spot the actual issue
//! instead of fishing through generic numbers.

use crate::store::ChunkStore;
use std::collections::HashMap;
use std::fmt::Write;

pub fn build_chunk_diagnostic(
    store: &ChunkStore,
    chunk: (i32, i32, i32),
    chunk_size: usize,
    voxel_scale: f32,
    world_scale: f32,
) -> String {
    // Backward-compatible wrapper: derive UE coord via the inverse transform.
    let ue_chunk = (chunk.0, chunk.2, -chunk.1);
    build_chunk_diagnostic_with_ue(store, chunk, ue_chunk, chunk_size, voxel_scale, world_scale)
}

/// Same as build_chunk_diagnostic but takes the UE chunk coord explicitly,
/// so the dump can label both Rust (HashMap key) and UE (caller-side)
/// coords accurately. Used by the FFI which has the UE coord directly.
pub fn build_chunk_diagnostic_with_ue(
    store: &ChunkStore,
    chunk: (i32, i32, i32),
    ue_chunk: (i32, i32, i32),
    chunk_size: usize,
    voxel_scale: f32,
    world_scale: f32,
) -> String {
    let mut out = String::new();
    let cs = chunk_size;
    let cs_f = cs as f32;

    // ── Identity ──
    // Both coord systems shown explicitly. Rust = HashMap key (after
    // ue_chunk_to_rust transform). UE = what the caller passed in =
    // floor(UE_world / ChunkWorldSize). Confirms which key actually got
    // looked up vs what the user clicked on the UE actor.
    let _ = writeln!(out, "═══════════ CHUNK DIAGNOSTIC ═══════════");
    let _ = writeln!(out, "Rust chunk (HashMap key): ({}, {}, {})", chunk.0, chunk.1, chunk.2);
    let _ = writeln!(out, "UE chunk (caller input):  ({}, {}, {})", ue_chunk.0, ue_chunk.1, ue_chunk.2);
    let _ = writeln!(out, "Transform: rust = ue_chunk_to_rust(ue) = (ue.x, ue.z, -ue.y)");
    // Rust voxel origin (where Rust noise samples)
    let rust_origin = (
        chunk.0 as f32 * cs_f,
        chunk.1 as f32 * cs_f,
        chunk.2 as f32 * cs_f,
    );
    let _ = writeln!(
        out,
        "Rust origin (vx): ({:.0}, {:.0}, {:.0})",
        rust_origin.0, rust_origin.1, rust_origin.2
    );
    // UE actor world position = ue_chunk * ChunkWorldSize (no transform —
    // UE side does NOT apply ue_chunk_to_rust to position actors). This
    // is the position you'd see in the World Outliner / viewport.
    let scale_uu = voxel_scale * world_scale;
    let chunk_uu = cs_f * scale_uu;
    let ue_actor_origin = (
        ue_chunk.0 as f32 * chunk_uu,
        ue_chunk.1 as f32 * chunk_uu,
        ue_chunk.2 as f32 * chunk_uu,
    );
    let _ = writeln!(
        out,
        "UE actor pos (UU):    ({:.0}, {:.0}, {:.0})  ← chunk's actor location",
        ue_actor_origin.0, ue_actor_origin.1, ue_actor_origin.2
    );
    let _ = writeln!(
        out,
        "UE actor AABB (UU):   X=[{:.0}..{:.0}]  Y=[{:.0}..{:.0}]  Z=[{:.0}..{:.0}]",
        ue_actor_origin.0, ue_actor_origin.0 + chunk_uu,
        ue_actor_origin.1, ue_actor_origin.1 + chunk_uu,
        ue_actor_origin.2, ue_actor_origin.2 + chunk_uu,
    );
    let region_size = store.region_size;
    let rk = voxel_gen::region_gen::region_key(chunk.0, chunk.1, chunk.2, region_size);
    let _ = writeln!(
        out,
        "Region key:    ({}, {}, {})  (region size {}³)",
        rk.0, rk.1, rk.2, region_size
    );
    let _ = writeln!(out, "Chunk size:    {} (grid {}³)", cs, cs + 1);
    let _ = writeln!(
        out,
        "World scale:   voxel={:.3}  ue_per_voxel={:.1}",
        voxel_scale, scale_uu
    );

    // ── State ──
    let _ = writeln!(out, "");
    let _ = writeln!(out, "──────────── STATE ────────────");
    let dirty = store.modification_tracker.dirty_chunks.contains(&chunk);
    let preserved = store.preserved_snapshots.contains_key(&chunk);
    let pending = store
        .pending_snapshots
        .as_ref()
        .map(|d| d.chunk_snapshots.contains_key(&chunk))
        .unwrap_or(false);
    let region_gen = store.is_region_generated(&rk);
    let has_density = store.density_fields.contains_key(&chunk);
    let has_hermite = store.hermite_data.contains_key(&chunk);
    let has_mesh = store.base_meshes.contains_key(&chunk);
    let has_seam_data = store.chunk_seam_data.contains_key(&chunk);
    let has_stress = store.stress_fields.contains_key(&chunk);
    let crystal_count = store
        .crystal_placements
        .get(&chunk)
        .map(|v| v.len())
        .unwrap_or(0);
    let _ = writeln!(
        out,
        "density field:      {}",
        if has_density { "loaded" } else { "NOT LOADED" }
    );
    let hermite_n = store.hermite_data.get(&chunk).map(|h| h.edges.len()).unwrap_or(0);
    let _ = writeln!(
        out,
        "hermite data:       {}{}",
        if has_hermite { "loaded" } else { "NOT LOADED" },
        if has_hermite { format!(" ({} edges)", hermite_n) } else { String::new() }
    );
    let mesh_summary = if let Some(m) = store.base_meshes.get(&chunk) {
        format!(" ({} verts, {} tris)", m.vertices.len(), m.triangles.len())
    } else {
        String::new()
    };
    let _ = writeln!(
        out,
        "base mesh:          {}{}",
        if has_mesh { "loaded" } else { "NOT LOADED" },
        mesh_summary
    );
    let _ = writeln!(out, "seam data:          {}", if has_seam_data { "loaded" } else { "absent" });
    let _ = writeln!(out, "stress field:       {}", if has_stress { "loaded" } else { "absent" });
    let _ = writeln!(out, "region generated:   {}", region_gen);
    let _ = writeln!(
        out,
        "dirty (user-edit):  {}{}",
        dirty,
        if dirty { "  ← will be preserved on unload" } else { "" }
    );
    let _ = writeln!(
        out,
        "preserved snapshot: {}{}",
        preserved,
        if preserved { "  ← waiting to be re-applied on next stream-in" } else { "" }
    );
    let _ = writeln!(
        out,
        "pending snapshot:   {}{}",
        pending,
        if pending { "  ← from save file, applies once" } else { "" }
    );
    let _ = writeln!(out, "crystal placements: {}", crystal_count);
    // Worm context for this region
    let region_worm_count = store
        .region_worm_paths
        .get(&rk)
        .map(|v| v.len())
        .unwrap_or(0);
    let total_worm_segs: usize = store
        .region_worm_paths
        .get(&rk)
        .map(|v| v.iter().map(|p| p.len()).sum())
        .unwrap_or(0);
    let _ = writeln!(
        out,
        "region worm paths:  {} paths / {} total segments",
        region_worm_count, total_worm_segs
    );

    // ── Neighbors (face-adjacent) ──
    let _ = writeln!(out, "");
    let _ = writeln!(out, "──────────── NEIGHBORS (face-adjacent) ────────────");
    let face_offsets: [(&str, (i32, i32, i32)); 6] = [
        ("+X", (1, 0, 0)),
        ("-X", (-1, 0, 0)),
        ("+Y", (0, 1, 0)),
        ("-Y", (0, -1, 0)),
        ("+Z", (0, 0, 1)),
        ("-Z", (0, 0, -1)),
    ];
    for (label, off) in face_offsets {
        let n_key = (chunk.0 + off.0, chunk.1 + off.1, chunk.2 + off.2);
        let summary = if let Some(df) = store.density_fields.get(&n_key) {
            let mut s = 0u32;
            let mut a = 0u32;
            for sm in &df.samples {
                if sm.density > 0.0 {
                    s += 1;
                } else {
                    a += 1;
                }
            }
            let total = (s + a).max(1);
            let solid_pct = 100.0 * s as f32 / total as f32;
            let kind = if solid_pct > 90.0 {
                "mostly solid"
            } else if solid_pct < 10.0 {
                "mostly air"
            } else {
                "mixed"
            };
            let edited = if store.modification_tracker.dirty_chunks.contains(&n_key) {
                " [edited]"
            } else {
                ""
            };
            format!("loaded ({}, {:.0}% solid){}", kind, solid_pct, edited)
        } else {
            "NOT LOADED  ← seam may show artifacts here".to_string()
        };
        let _ = writeln!(
            out,
            "  {} ({}, {}, {}): {}",
            label, n_key.0, n_key.1, n_key.2, summary
        );
    }

    let df = match store.density_fields.get(&chunk) {
        Some(df) => df,
        None => {
            let _ = writeln!(out, "");
            let _ = writeln!(out, "(no density field — nothing more to dump)");
            return out;
        }
    };

    // ── Density stats ──
    let _ = writeln!(out, "");
    let _ = writeln!(out, "──────────── DENSITY STATS ────────────");
    let total = df.samples.len();
    let mut solid = 0u32;
    let mut air = 0u32;
    let mut min_d = f32::INFINITY;
    let mut max_d = f32::NEG_INFINITY;
    let mut nan_count = 0u32;
    let mut sum_d: f64 = 0.0;
    let mut mat_counts: HashMap<String, u32> = HashMap::new();
    for sm in &df.samples {
        if sm.density.is_nan() {
            nan_count += 1;
            continue;
        }
        if sm.density > 0.0 {
            solid += 1;
        } else {
            air += 1;
        }
        if sm.density < min_d {
            min_d = sm.density;
        }
        if sm.density > max_d {
            max_d = sm.density;
        }
        sum_d += sm.density as f64;
        *mat_counts.entry(format!("{:?}", sm.material)).or_insert(0) += 1;
    }
    let mean_d = if total > 0 { sum_d / total as f64 } else { 0.0 };
    let _ = writeln!(out, "total={}  solid={}  air={}  nan={}", total, solid, air, nan_count);
    let _ = writeln!(out, "min={:.3}  max={:.3}  mean={:.3}", min_d, max_d, mean_d);
    if nan_count > 0 {
        let _ = writeln!(out, "  ⚠ {} NaN cells — density field corruption", nan_count);
    }
    if max_d > 1.001 {
        let _ = writeln!(out, "  ⚠ density >1.0 — formation/zone writer didn't clamp");
    }
    // Suspicious-wipe heuristic: a chunk that's 100% solid (no air) but has
    // a face-neighbor with significant air AND a low-magnitude min density
    // (e.g. 0.24) at the boundary is the fingerprint of "this chunk had
    // user authoring that got wiped". The boundary cell stayed at the
    // sync'd-from-edited-neighbor value, but the interior reverted to host
    // rock — so what was a real wall is now invisible because there's no
    // surface inside the chunk.
    if solid == total as u32 && nan_count == 0 && min_d > 0.0 && min_d < 0.5 {
        let mut neighbor_has_air = false;
        for &(dx, dy, dz) in &[(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)] {
            let n = (chunk.0 + dx, chunk.1 + dy, chunk.2 + dz);
            if let Some(ndf) = store.density_fields.get(&n) {
                let n_air = ndf.samples.iter().filter(|s| s.density <= 0.0).count();
                if n_air > 50 {
                    neighbor_has_air = true;
                    break;
                }
            }
        }
        if neighbor_has_air {
            let _ = writeln!(out, "  ⚠ ALL-SOLID with cliff-shaped boundary (min={:.2}) AND a face-neighbor has air", min_d);
            let _ = writeln!(out, "  ⚠ likely SIGN OF A WIPE — this chunk's authoring may have been lost.");
            let _ = writeln!(out, "  ⚠ Real surface should produce hermite edges; current chunk has 0.");
        }
    }

    // Spot samples — center + 8 corners. Helps spot baseline cell health.
    let center = df.get(cs / 2, cs / 2, cs / 2);
    let _ = writeln!(out, "center cell ({},{},{}): density={:.3} mat={:?}",
        cs / 2, cs / 2, cs / 2, center.density, center.material);
    let _ = writeln!(out, "8 corner cell densities (cell-corner of grid 31³):");
    for &cx in &[0usize, cs] {
        for &cy in &[0usize, cs] {
            for &cz in &[0usize, cs] {
                let s = df.get(cx, cy, cz);
                let _ = writeln!(out, "  ({:>2},{:>2},{:>2}): d={:>6.3}  mat={:?}",
                    cx, cy, cz, s.density, s.material);
            }
        }
    }

    // ── Materials ──
    let _ = writeln!(out, "");
    let _ = writeln!(out, "──────────── MATERIALS (sorted, top 12) ────────────");
    let mut mat_sorted: Vec<_> = mat_counts.iter().collect();
    mat_sorted.sort_by_key(|(_, &c)| std::cmp::Reverse(c));
    let total_mat = total as f32;
    for (name, count) in mat_sorted.iter().take(12) {
        let pct = 100.0 * **count as f32 / total_mat;
        let _ = writeln!(out, "  {:<14} {:>6}  ({:>4.1}%)", name, count, pct);
    }
    if mat_sorted.len() > 12 {
        let _ = writeln!(out, "  ... +{} more", mat_sorted.len() - 12);
    }

    // ── Mesh + coplanar analysis ──
    let _ = writeln!(out, "");
    let _ = writeln!(out, "──────────── MESH ────────────");
    if let Some(mesh) = store.base_meshes.get(&chunk) {
        let _ = writeln!(out, "vertices={}  triangles={}", mesh.vertices.len(), mesh.triangles.len());
        if !mesh.vertices.is_empty() {
            let mut vmin = glam::Vec3::splat(f32::INFINITY);
            let mut vmax = glam::Vec3::splat(f32::NEG_INFINITY);
            let mut nan_verts = 0u32;
            for v in &mesh.vertices {
                if v.position.x.is_nan() || v.position.y.is_nan() || v.position.z.is_nan() {
                    nan_verts += 1;
                    continue;
                }
                vmin = vmin.min(v.position);
                vmax = vmax.max(v.position);
            }
            let _ = writeln!(out, "AABB (voxel): min=({:.2},{:.2},{:.2})  max=({:.2},{:.2},{:.2})",
                vmin.x, vmin.y, vmin.z, vmax.x, vmax.y, vmax.z);
            if nan_verts > 0 {
                let _ = writeln!(out, "  ⚠ {} NaN vertices in mesh", nan_verts);
            }
        }
        // Coplanar tri histogram
        let eps = 0.05_f32;
        let mut x_bins: HashMap<i32, u32> = HashMap::new();
        let mut y_bins: HashMap<i32, u32> = HashMap::new();
        let mut z_bins: HashMap<i32, u32> = HashMap::new();
        for tri in &mesh.triangles {
            let p0 = mesh.vertices[tri.indices[0] as usize].position;
            let p1 = mesh.vertices[tri.indices[1] as usize].position;
            let p2 = mesh.vertices[tri.indices[2] as usize].position;
            if (p0.x - p1.x).abs() < eps && (p1.x - p2.x).abs() < eps {
                *x_bins.entry((p0.x * 2.0).round() as i32).or_insert(0) += 1;
            }
            if (p0.y - p1.y).abs() < eps && (p1.y - p2.y).abs() < eps {
                *y_bins.entry((p0.y * 2.0).round() as i32).or_insert(0) += 1;
            }
            if (p0.z - p1.z).abs() < eps && (p1.z - p2.z).abs() < eps {
                *z_bins.entry((p0.z * 2.0).round() as i32).or_insert(0) += 1;
            }
        }
        let mut x_top: Vec<_> = x_bins.iter().collect();
        x_top.sort_by_key(|(_, &c)| std::cmp::Reverse(c));
        let mut y_top: Vec<_> = y_bins.iter().collect();
        y_top.sort_by_key(|(_, &c)| std::cmp::Reverse(c));
        let mut z_top: Vec<_> = z_bins.iter().collect();
        z_top.sort_by_key(|(_, &c)| std::cmp::Reverse(c));
        let _ = writeln!(out, "");
        let _ = writeln!(out, "──────────── COPLANAR TRIS (flat-wall detection) ────────────");
        let _ = writeln!(out, "Bins are (coord*2 → count); high count = many tris on one plane = flat wall.");
        let _ = writeln!(out, "X-planes top 5: {:?}", x_top.iter().take(5).map(|(k, v)| (**k, **v)).collect::<Vec<_>>());
        let _ = writeln!(out, "Y-planes top 5: {:?}", y_top.iter().take(5).map(|(k, v)| (**k, **v)).collect::<Vec<_>>());
        let _ = writeln!(out, "Z-planes top 5: {:?}", z_top.iter().take(5).map(|(k, v)| (**k, **v)).collect::<Vec<_>>());
        let max_count = x_top
            .first()
            .map(|(_, &c)| c)
            .unwrap_or(0)
            .max(y_top.first().map(|(_, &c)| c).unwrap_or(0))
            .max(z_top.first().map(|(_, &c)| c).unwrap_or(0));
        if max_count > 100 {
            let _ = writeln!(
                out,
                "  ⚠ HIGH COPLANAR COUNT ({}) — likely flat-wall artifact at chunk seam",
                max_count
            );
        }
    } else {
        let _ = writeln!(out, "(no mesh in store)");
    }

    // ── Hermite by axis ──
    if let Some(h) = store.hermite_data.get(&chunk) {
        let mut x_edges = 0u32;
        let mut y_edges = 0u32;
        let mut z_edges = 0u32;
        // Per-boundary-plane counts. Seam quads at each face require edges
        // at that face's plane (e.g. seam at +X face uses Y/Z edges at x=cs).
        let mut x_at_pos_x = 0u32;  // X-edges at +X face (impossible — X-edges span x..x+1, last is x=cs-1)
        let mut x_at_pos_y = 0u32;  // X-edges at +Y face (y=cs)
        let mut x_at_pos_z = 0u32;  // X-edges at +Z face (z=cs)
        let mut y_at_pos_x = 0u32;
        let mut y_at_pos_y = 0u32;
        let mut y_at_pos_z = 0u32;
        let mut z_at_pos_x = 0u32;
        let mut z_at_pos_y = 0u32;
        let mut z_at_pos_z = 0u32;
        let cs_u8 = cs as u8;
        for (key, _) in h.edges.iter() {
            let kx = key.x();
            let ky = key.y();
            let kz = key.z();
            match key.axis() {
                0 => {
                    x_edges += 1;
                    if kx == cs_u8 { x_at_pos_x += 1; }
                    if ky == cs_u8 { x_at_pos_y += 1; }
                    if kz == cs_u8 { x_at_pos_z += 1; }
                }
                1 => {
                    y_edges += 1;
                    if kx == cs_u8 { y_at_pos_x += 1; }
                    if ky == cs_u8 { y_at_pos_y += 1; }
                    if kz == cs_u8 { y_at_pos_z += 1; }
                }
                _ => {
                    z_edges += 1;
                    if kx == cs_u8 { z_at_pos_x += 1; }
                    if ky == cs_u8 { z_at_pos_y += 1; }
                    if kz == cs_u8 { z_at_pos_z += 1; }
                }
            }
        }
        let _ = writeln!(out, "");
        let _ = writeln!(out, "──────────── HERMITE EDGES BY AXIS ────────────");
        let _ = writeln!(out, "X-axis edges: {}  Y-axis edges: {}  Z-axis edges: {}",
            x_edges, y_edges, z_edges);
        // Seam-relevant counts: edges AT each boundary plane. The +X seam
        // uses Y- and Z-axis edges at x=cs. The +Y seam uses X- and Z-axis
        // edges at y=cs. The +Z seam uses X- and Y-axis edges at z=cs.
        let pos_x_seam = y_at_pos_x + z_at_pos_x;
        let pos_y_seam = x_at_pos_y + z_at_pos_y;
        let pos_z_seam = x_at_pos_z + y_at_pos_z;
        let _ = writeln!(out, "edges at +X plane (seam quad source): {}  (Y={} Z={})",
            pos_x_seam, y_at_pos_x, z_at_pos_x);
        let _ = writeln!(out, "edges at +Y plane (seam quad source): {}  (X={} Z={})",
            pos_y_seam, x_at_pos_y, z_at_pos_y);
        let _ = writeln!(out, "edges at +Z plane (seam quad source): {}  (X={} Z={})",
            pos_z_seam, x_at_pos_z, y_at_pos_z);
        if pos_x_seam == 0 && pos_y_seam == 0 && pos_z_seam == 0 {
            let _ = writeln!(out, "  ⚠ NO boundary-plane edges — this chunk's seam pass will emit 0 quads");
        }
    }

    // ── Seam pass simulation: actually run generate_chunk_seam_quads and
    // report what it would emit. Shows seam mesh size, coplanar-tri
    // distribution per face, and which face neighbors had their DC verts
    // looked up. If a face shows 0 quads despite having boundary edges,
    // the neighbor lookup failed (missing seam_data) or all DC verts were
    // NaN AND the fallback collapsed to degenerate triangles.
    if store.chunk_seam_data.contains_key(&chunk) {
        let seam = voxel_gen::region_gen::generate_chunk_seam_quads(
            chunk, &store.chunk_seam_data, cs,
        );
        let _ = writeln!(out, "");
        let _ = writeln!(out, "──────────── SEAM PASS SIMULATION ────────────");
        let _ = writeln!(out, "seam quads emitted: {} verts / {} tris", seam.vertices.len(), seam.triangles.len());
        // Bin seam triangles by which boundary plane they sit on.
        let mut at_x = 0u32;
        let mut at_y = 0u32;
        let mut at_z = 0u32;
        let cs_f = cs as f32;
        let eps = 0.5_f32; // generous: seam tris span up to ~1 cell off the boundary plane
        for tri in &seam.triangles {
            let p0 = seam.vertices[tri.indices[0] as usize].position;
            let p1 = seam.vertices[tri.indices[1] as usize].position;
            let p2 = seam.vertices[tri.indices[2] as usize].position;
            let cx = (p0.x + p1.x + p2.x) / 3.0;
            let cy = (p0.y + p1.y + p2.y) / 3.0;
            let cz = (p0.z + p1.z + p2.z) / 3.0;
            if (cx - cs_f).abs() < eps { at_x += 1; }
            if (cy - cs_f).abs() < eps { at_y += 1; }
            if (cz - cs_f).abs() < eps { at_z += 1; }
        }
        let _ = writeln!(out, "  near +X plane: {} tris", at_x);
        let _ = writeln!(out, "  near +Y plane: {} tris", at_y);
        let _ = writeln!(out, "  near +Z plane: {} tris", at_z);
        // Check if face neighbors have seam_data (required for quad lookups)
        let face_offsets: [(&str, (i32, i32, i32)); 3] = [
            ("+X", (1, 0, 0)), ("+Y", (0, 1, 0)), ("+Z", (0, 0, 1)),
        ];
        for (label, off) in face_offsets {
            let n = (chunk.0 + off.0, chunk.1 + off.1, chunk.2 + off.2);
            let has_seam = store.chunk_seam_data.contains_key(&n);
            let _ = writeln!(out, "  {} face neighbor {:?}: seam_data {}",
                label, n, if has_seam { "loaded" } else { "MISSING ⚠ — quads can't form" });
        }
    } else {
        let _ = writeln!(out, "");
        let _ = writeln!(out, "──────────── SEAM PASS SIMULATION ────────────");
        let _ = writeln!(out, "  ⚠ this chunk has NO seam_data — seam pass would skip it entirely");
    }

    // ── Stress field summary ──
    if let Some(_sf) = store.stress_fields.get(&chunk) {
        let _ = writeln!(out, "");
        let _ = writeln!(out, "──────────── STRESS ────────────");
        let _ = writeln!(out, "(stress field present — span/value details require dedicated dump)");
    }

    // ── SEAM CONSISTENCY CHECK (the big one for the seam-not-generating bug) ──
    let _ = writeln!(out, "");
    let _ = writeln!(out, "──────────── SEAM CONSISTENCY (vs all 26 neighbors) ────────────");
    let _ = writeln!(out, "For every shared cell on every shared face/edge/corner, check that");
    let _ = writeln!(out, "this chunk and the neighbor agree. Mismatches → mesh holes / cliffs.");
    let _ = writeln!(out, "Threshold: density delta > 0.001 OR material mismatch.");
    let _ = writeln!(out, "");
    let mut total_mismatch = 0u32;
    let mut neighbors_checked = 0u32;
    let mut neighbors_missing = 0u32;
    let mut max_delta_per_neighbor: Vec<((i32, i32, i32), f32, u32)> = Vec::new();
    for dz in -1i32..=1 {
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                let n_key = (chunk.0 + dx, chunk.1 + dy, chunk.2 + dz);
                let n_df = match store.density_fields.get(&n_key) {
                    Some(d) => d,
                    None => {
                        neighbors_missing += 1;
                        continue;
                    }
                };
                neighbors_checked += 1;
                // Iterate the shared cells. For each axis: dx=+1 → my x=cs == neighbor x=0
                // dx=-1 → my x=0 == neighbor x=cs.
                // For dx=0 → all x in 0..=cs are shared on that axis.
                let x_iter: Box<dyn Iterator<Item = (usize, usize)>> = match dx {
                    1 => Box::new(std::iter::once((cs, 0))),
                    -1 => Box::new(std::iter::once((0, cs))),
                    _ => Box::new((0..=cs).map(|i| (i, i))),
                };
                let mut mismatches = 0u32;
                let mut max_d: f32 = 0.0;
                for (mx, nx) in x_iter {
                    let y_iter: Box<dyn Iterator<Item = (usize, usize)>> = match dy {
                        1 => Box::new(std::iter::once((cs, 0))),
                        -1 => Box::new(std::iter::once((0, cs))),
                        _ => Box::new((0..=cs).map(|i| (i, i))),
                    };
                    for (my, ny_) in y_iter {
                        let z_iter: Box<dyn Iterator<Item = (usize, usize)>> = match dz {
                            1 => Box::new(std::iter::once((cs, 0))),
                            -1 => Box::new(std::iter::once((0, cs))),
                            _ => Box::new((0..=cs).map(|i| (i, i))),
                        };
                        for (mz, nz) in z_iter {
                            let s_a = df.get(mx, my, mz);
                            let s_b = n_df.get(nx, ny_, nz);
                            let delta = (s_a.density - s_b.density).abs();
                            if delta > 0.001 || s_a.material != s_b.material {
                                mismatches += 1;
                                if delta > max_d {
                                    max_d = delta;
                                }
                            }
                        }
                    }
                }
                if mismatches > 0 {
                    total_mismatch += mismatches;
                    max_delta_per_neighbor.push((n_key, max_d, mismatches));
                }
            }
        }
    }
    let _ = writeln!(
        out,
        "checked {} neighbors  ({} not loaded)  total mismatched cells: {}",
        neighbors_checked, neighbors_missing, total_mismatch
    );
    if total_mismatch > 0 {
        max_delta_per_neighbor.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let _ = writeln!(out, "  ⚠ Seam mismatches detected — may explain mesh holes / cliffs");
        for (k, mdelta, count) in max_delta_per_neighbor.iter().take(8) {
            let dir = (k.0 - chunk.0, k.1 - chunk.1, k.2 - chunk.2);
            let kind = match (dir.0.abs() + dir.1.abs() + dir.2.abs()) {
                1 => "face",
                2 => "edge",
                3 => "corner",
                _ => "?",
            };
            let _ = writeln!(out, "    {} neighbor {:?}: {} cells, max Δdensity={:.3}",
                kind, k, count, mdelta);
        }
    } else {
        let _ = writeln!(out, "  ✓ All shared cells consistent across loaded neighbors");
    }

    // ── Internal cliff scan ──
    // Scan inside the chunk for adjacent-cell density jumps > 0.5. A cliff
    // INSIDE a chunk is unusual (noise is smooth) — usually means a writer
    // (formation, zone, brush) wrote a hard value next to natural noise.
    let _ = writeln!(out, "");
    let _ = writeln!(out, "──────────── INTERNAL CLIFFS (density Δ > 0.5 between adjacent cells) ────────────");
    let mut cliff_count = 0u32;
    let mut top_cliffs: Vec<(usize, usize, usize, char, f32)> = Vec::new();
    for z in 0..cs {
        for y in 0..cs {
            for x in 0..cs {
                let d = df.get(x, y, z).density;
                if d.is_nan() { continue; }
                let dx = (df.get(x + 1, y, z).density - d).abs();
                let dy = (df.get(x, y + 1, z).density - d).abs();
                let dz = (df.get(x, y, z + 1).density - d).abs();
                if dx > 0.5 { cliff_count += 1; top_cliffs.push((x, y, z, 'X', dx)); }
                if dy > 0.5 { cliff_count += 1; top_cliffs.push((x, y, z, 'Y', dy)); }
                if dz > 0.5 { cliff_count += 1; top_cliffs.push((x, y, z, 'Z', dz)); }
            }
        }
    }
    let _ = writeln!(out, "total internal cliffs: {}", cliff_count);
    if !top_cliffs.is_empty() {
        top_cliffs.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
        let _ = writeln!(out, "top 10 cliffs by magnitude (cell, axis, Δ):");
        for (x, y, z, axis, d) in top_cliffs.iter().take(10) {
            let _ = writeln!(out, "  ({:>2},{:>2},{:>2}) along {} → Δ={:.3}", x, y, z, axis, d);
        }
    }
    if cliff_count > 200 {
        let _ = writeln!(out, "  ⚠ many internal cliffs — a writer may be saturating density");
    }

    // ── Boundary slices for ALL 6 faces (visual check) ──
    let _ = writeln!(out, "");
    let _ = writeln!(out, "──────────── BOUNDARY SLICES (interior | sync'd boundary) ────────────");
    let _ = writeln!(out, "5×5 sample grid per face. A 1-cell step (e.g. 1.00 → -0.07) is a CLIFF →");
    let _ = writeln!(out, "DC will produce a flat wall. Healthy gradients change <0.3 between cells.");
    let _ = writeln!(out, "");
    let face_dump = |out: &mut String, title: &str, axis: usize, dir: i32| {
        let _ = writeln!(out, "[{}]", title);
        let step = (cs / 4).max(1);
        for v in (0..=cs).step_by(step).take(5) {
            let mut row = String::new();
            for u in (0..=cs).step_by(step).take(5) {
                let (i_in, b_in) = match axis {
                    0 => (
                        if dir > 0 { (cs - 1, u, v) } else { (1, u, v) },
                        if dir > 0 { (cs, u, v) } else { (0, u, v) },
                    ),
                    1 => (
                        if dir > 0 { (u, cs - 1, v) } else { (u, 1, v) },
                        if dir > 0 { (u, cs, v) } else { (u, 0, v) },
                    ),
                    _ => (
                        if dir > 0 { (u, v, cs - 1) } else { (u, v, 1) },
                        if dir > 0 { (u, v, cs) } else { (u, v, 0) },
                    ),
                };
                let s_in = df.get(i_in.0, i_in.1, i_in.2);
                let s_b = df.get(b_in.0, b_in.1, b_in.2);
                let _ = write!(row, " [{:>5.2}|{:>5.2}]", s_in.density, s_b.density);
            }
            let label = match axis { 0 => "y/z", 1 => "x/z", _ => "x/y" };
            let _ = writeln!(out, "  {}={}:{}", label, v, row);
        }
    };
    face_dump(&mut out, "+X face", 0, 1);
    face_dump(&mut out, "-X face", 0, -1);
    face_dump(&mut out, "+Y face", 1, 1);
    face_dump(&mut out, "-Y face", 1, -1);
    face_dump(&mut out, "+Z face", 2, 1);
    face_dump(&mut out, "-Z face", 2, -1);
    let _ = writeln!(out, "═══════════════════════════════════════");
    out
}
