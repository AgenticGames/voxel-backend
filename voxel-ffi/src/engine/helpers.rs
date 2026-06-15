use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self};

use crossbeam_channel::{bounded, Receiver, Sender};
use voxel_fluid::FluidConfig;
use voxel_fluid::FluidEvent;

use crate::types::*;


/// Wrapper for the fluid simulation loop that converts FluidResults to WorkerResults
/// with coordinate transformation from Rust space to UE space.
pub(crate) fn fluid_sim_loop_wrapper(
    shutdown: Arc<AtomicBool>,
    event_rx: Receiver<FluidEvent>,
    result_tx: Sender<WorkerResult>,
    config: FluidConfig,
    world_scale: f32,
) {
    use voxel_fluid::FluidResult;
    use voxel_fluid::thread::fluid_sim_loop;

    let chunk_size = config.chunk_size;

    // Create internal channels for the fluid sim
    let (internal_tx, internal_rx) = bounded::<FluidResult>(128);

    let sim_shutdown = Arc::clone(&shutdown);
    let sim_config = config.clone();
    let sim_handle = thread::spawn(move || {
        fluid_sim_loop(sim_shutdown, event_rx, internal_tx, sim_config);
    });

    // Relay loop: convert FluidResult -> WorkerResult with coord transform
    while !shutdown.load(Ordering::Relaxed) {
        match internal_rx.recv_timeout(std::time::Duration::from_millis(50)) {
            Ok(fluid_result) => match fluid_result {
                FluidResult::FluidMesh { chunk, mesh } => {
                    let converted = convert_fluid_mesh_to_ue(&mesh, chunk, chunk_size, world_scale);
                    let _ = result_tx.send(WorkerResult::FluidMesh {
                        chunk,
                        mesh: converted,
                    });
                }
                FluidResult::SolidifyRequest { positions } => {
                    // Legacy single-list path. Treat as Obsidian-only quench
                    // so any old code that emits SolidifyRequest still produces
                    // a visible wall instead of being dropped.
                    let _ = result_tx.send(WorkerResult::LavaQuench {
                        obsidian: positions,
                        scoria: Vec::new(),
                        drained_water: Vec::new(),
                    });
                }
                FluidResult::LavaQuench { obsidian, scoria, drained_water } => {
                    let _ = result_tx.send(WorkerResult::LavaQuench {
                        obsidian,
                        scoria,
                        drained_water,
                    });
                }
            },
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }

    let _ = sim_handle.join();
}

/// Convert a fluid mesh from Rust local chunk space to UE local chunk space.
/// Positions are local [0, chunk_size] — the chunk actor provides the world offset.
fn convert_fluid_mesh_to_ue(
    mesh: &voxel_fluid::mesh::FluidMeshData,
    _chunk: (i32, i32, i32),
    _chunk_size: usize,
    scale: f32,
) -> ConvertedFluidMesh {
    let mut positions = Vec::with_capacity(mesh.positions.len());
    let mut normals = Vec::with_capacity(mesh.normals.len());
    let mut flow_directions = Vec::with_capacity(mesh.flow_directions.len());

    for p in &mesh.positions {
        // Rust Y-up -> UE Z-up: (x, -z, y) * scale
        // Positions are local to the chunk (no origin offset needed)
        positions.push(FfiVec3 {
            x: p[0] * scale,
            y: -p[2] * scale,
            z: p[1] * scale,
        });
    }

    for n in &mesh.normals {
        normals.push(FfiVec3 {
            x: n[0],
            y: -n[2],
            z: n[1],
        });
    }

    // Flow directions: (dx, dz, magnitude) — transform horizontal components
    // Rust (dx, dz) → UE (dx, -dz) to match the Y→Z axis swap
    for f in &mesh.flow_directions {
        flow_directions.push(FfiVec3 {
            x: f[0],       // dx unchanged
            y: -f[1],      // dz negated (Rust Z → UE -Y)
            z: f[2],       // magnitude
        });
    }

    ConvertedFluidMesh {
        positions,
        normals,
        fluid_types: mesh.fluid_types.clone(),
        indices: mesh.indices.clone(),
        uvs: mesh.uvs.clone(),
        flow_directions,
    }
}

pub(crate) fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

// ── Trigger helpers ────────────────────────────────────────────────────

pub(crate) fn aabb_to_ffi(a: &crate::triggers::VoxelAabb) -> crate::types::FfiVoxelAabb {
    crate::types::FfiVoxelAabb {
        min: crate::types::FfiVoxelCoord {
            x: a.min.0,
            y: a.min.1,
            z: a.min.2,
        },
        max: crate::types::FfiVoxelCoord {
            x: a.max.0,
            y: a.max.1,
            z: a.max.2,
        },
    }
}

pub(crate) fn zero_aabb() -> crate::types::FfiVoxelAabb {
    crate::types::FfiVoxelAabb {
        min: crate::types::FfiVoxelCoord { x: 0, y: 0, z: 0 },
        max: crate::types::FfiVoxelCoord { x: 0, y: 0, z: 0 },
    }
}

pub(crate) fn aabb_center(a: &crate::triggers::VoxelAabb) -> (i32, i32, i32) {
    (
        (a.min.0 + a.max.0) / 2,
        (a.min.1 + a.max.1) / 2,
        (a.min.2 + a.max.2) / 2,
    )
}

/// Test whether a cell at world voxel `(wx,wy,wz)` has positive center
/// density — i.e. its interior is rock and the cinematic will treat it
/// as a falling solid. Samples all 8 corners (potentially across chunk
/// borders) and returns true if their average is > 0.
///
/// This is the cell-aware test used by both `query_solid_voxels_in_sphere`
/// (paint filter) and `synthesize_collapse_event` (synth filter). A
/// previous single-corner test silently skipped every cell at the
/// rock/air boundary — most importantly cave-ceiling cells, where the
/// bottom corner sits in cave air below.
pub(crate) fn cell_has_solid_center(
    store: &crate::store::ChunkStore,
    wx: i32,
    wy: i32,
    wz: i32,
    chunk_size: i32,
) -> bool {
    let mut sum: f32 = 0.0;
    let mut samples: u32 = 0;
    for c in 0..8 {
        let ox = (c & 1) as i32;
        let oy = ((c >> 1) & 1) as i32;
        let oz = ((c >> 2) & 1) as i32;
        let sx = wx + ox;
        let sy = wy + oy;
        let sz = wz + oz;
        let cx = sx.div_euclid(chunk_size);
        let cy = sy.div_euclid(chunk_size);
        let cz = sz.div_euclid(chunk_size);
        let lx = sx.rem_euclid(chunk_size) as usize;
        let ly = sy.rem_euclid(chunk_size) as usize;
        let lz = sz.rem_euclid(chunk_size) as usize;
        if let Some(df) = store.density_fields.get(&(cx, cy, cz)) {
            // df.size = chunk_size + 1, so the +1 corner indices on the
            // far side of a chunk are valid (they index the shared
            // boundary slice that overlaps with the next chunk).
            if lx < df.size && ly < df.size && lz < df.size {
                sum += df.get(lx, ly, lz).density;
                samples += 1;
            }
        }
    }
    if samples == 0 {
        return false;
    }
    (sum / samples as f32) > 0.0
}
