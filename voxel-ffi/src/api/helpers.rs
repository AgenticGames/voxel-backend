//! Auto-split from the former monolithic `api.rs`. See `mod.rs`.
#[allow(unused_imports)]
use std::ffi::{c_char, c_void, CString};
#[allow(unused_imports)]
use std::ptr;
#[allow(unused_imports)]
use crate::engine::VoxelEngine;
#[allow(unused_imports)]
use crate::types::*;
use crate::convert::rust_chunk_to_ue;

// ── Internal helpers ──

pub(crate) fn convert_mesh_to_ffi_result(
    chunk: (i32, i32, i32),
    mesh: ConvertedMesh,
    generation: u64,
    crystal_data: Vec<FfiCrystalPlacement>,
    mushroom_data: Vec<FfiMushroomInstance>,
    zone_descriptors: Vec<FfiZoneDescriptor>,
) -> FfiResult {
    // Convert Rust chunk coords back to UE space for the caller
    let ue = rust_chunk_to_ue(chunk.0, chunk.1, chunk.2);
    FfiResult {
        result_type: FfiResultType::ChunkMesh,
        chunk: FfiChunkCoord {
            x: ue.0,
            y: ue.1,
            z: ue.2,
        },
        mesh: converted_mesh_to_ffi(mesh),
        mined: FfiMinedMaterials { counts: [0; 64] },
        generation,
        fluid_mesh: empty_fluid_mesh_data(),
        crystal_data: convert_crystal_vec_to_ffi(crystal_data),
        zone_data: convert_zone_vec_to_ffi(zone_descriptors),
        mushroom_data: convert_mushroom_vec_to_ffi(mushroom_data),
        slab_fall: FfiSlabFallData::default(),
    }
}

/// FNV-1a-like hash over the crystal placement set. UE compares this to its
/// last-applied-hash per chunk and skips the expensive HISM rebuild when
/// the incoming hash matches — measured to drop `Foliage Create Proxy`
/// from ~11K calls to ~1K in a 30-event collapse stress test.
///
/// Hash 0 is reserved as "always apply" sentinel (never returned for
/// non-empty data — the FNV offset basis 0xcbf29ce484222325 is non-zero,
/// and we OR a non-zero count salt before returning).
pub(crate) fn compute_crystal_hash(placements: &[FfiCrystalPlacement]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let prime: u64 = 0x100000001b3;
    let mut mix = |x: u64, h: &mut u64| {
        *h ^= x;
        *h = h.wrapping_mul(prime);
    };
    mix(placements.len() as u64, &mut h);
    for p in placements {
        mix(p.x.to_bits() as u64, &mut h);
        mix(p.y.to_bits() as u64, &mut h);
        mix(p.z.to_bits() as u64, &mut h);
        mix(p.normal_x.to_bits() as u64, &mut h);
        mix(p.normal_y.to_bits() as u64, &mut h);
        mix(p.normal_z.to_bits() as u64, &mut h);
        mix(p.ore_type as u64, &mut h);
        mix(p.size_class as u64, &mut h);
        mix(p.scale.to_bits() as u64, &mut h);
    }
    // Guarantee non-zero so 0 stays reserved as the "no hash / always apply"
    // sentinel even if FNV produced a perfect collision to zero.
    if h == 0 { 1 } else { h }
}

pub(crate) fn convert_crystal_vec_to_ffi(data: Vec<FfiCrystalPlacement>) -> FfiCrystalData {
    if data.is_empty() {
        return FfiCrystalData {
            placements: std::ptr::null_mut(),
            count: 0,
            _padding: 0,
            hash: 0,
        };
    }
    let hash = compute_crystal_hash(&data);
    let count = data.len() as u32;
    let mut boxed = data.into_boxed_slice();
    let ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);
    FfiCrystalData { placements: ptr, count, _padding: 0, hash }
}

pub(crate) fn empty_crystal_data() -> FfiCrystalData {
    FfiCrystalData {
        placements: std::ptr::null_mut(),
        count: 0,
        _padding: 0,
        hash: 0,
    }
}

pub(crate) fn compute_mushroom_hash(instances: &[FfiMushroomInstance]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let prime: u64 = 0x100000001b3;
    let mut mix = |x: u64, h: &mut u64| {
        *h ^= x;
        *h = h.wrapping_mul(prime);
    };
    mix(instances.len() as u64, &mut h);
    for p in instances {
        mix(p.x.to_bits() as u64, &mut h);
        mix(p.y.to_bits() as u64, &mut h);
        mix(p.z.to_bits() as u64, &mut h);
        mix(p.normal_x.to_bits() as u64, &mut h);
        mix(p.normal_y.to_bits() as u64, &mut h);
        mix(p.normal_z.to_bits() as u64, &mut h);
        mix(p.scale.to_bits() as u64, &mut h);
        mix(p.yaw.to_bits() as u64, &mut h);
        mix(p.kind as u64, &mut h);
    }
    if h == 0 { 1 } else { h }
}

pub(crate) fn convert_mushroom_vec_to_ffi(data: Vec<FfiMushroomInstance>) -> FfiMushroomData {
    if data.is_empty() {
        return empty_mushroom_data();
    }
    let hash = compute_mushroom_hash(&data);
    let count = data.len() as u32;
    let mut boxed = data.into_boxed_slice();
    let ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);
    FfiMushroomData { instances: ptr, count, _padding: 0, hash }
}

pub(crate) fn empty_mushroom_data() -> FfiMushroomData {
    FfiMushroomData {
        instances: std::ptr::null_mut(),
        count: 0,
        _padding: 0,
        hash: 0,
    }
}

pub(crate) fn convert_zone_vec_to_ffi(data: Vec<FfiZoneDescriptor>) -> FfiZoneData {
    if data.is_empty() {
        return FfiZoneData { descriptors: std::ptr::null_mut(), count: 0 };
    }
    let count = data.len() as u32;
    let mut boxed = data.into_boxed_slice();
    let ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);
    FfiZoneData { descriptors: ptr, count }
}

pub(crate) fn empty_zone_data() -> FfiZoneData {
    FfiZoneData { descriptors: std::ptr::null_mut(), count: 0 }
}

pub(crate) fn converted_mesh_to_ffi(mesh: ConvertedMesh) -> FfiMeshData {
    let vertex_count = mesh.positions.len() as u32;
    let index_count = mesh.indices.len() as u32;
    let submesh_count = mesh.submeshes.len() as u32;

    let mut positions = mesh.positions.into_boxed_slice();
    let mut normals = mesh.normals.into_boxed_slice();
    let mut material_ids = mesh.material_ids.into_boxed_slice();
    let mut indices = mesh.indices.into_boxed_slice();
    let mut submeshes = mesh.submeshes.into_boxed_slice();

    let positions_ptr = positions.as_mut_ptr();
    let normals_ptr = normals.as_mut_ptr();
    let material_ids_ptr = material_ids.as_mut_ptr();
    let indices_ptr = indices.as_mut_ptr();
    let submeshes_ptr = submeshes.as_mut_ptr();

    std::mem::forget(positions);
    std::mem::forget(normals);
    std::mem::forget(material_ids);
    std::mem::forget(indices);
    std::mem::forget(submeshes);

    FfiMeshData {
        positions: positions_ptr,
        normals: normals_ptr,
        material_ids: material_ids_ptr,
        vertex_count,
        indices: indices_ptr,
        index_count,
        submeshes: submeshes_ptr,
        submesh_count,
    }
}

pub(crate) fn empty_mesh_data() -> FfiMeshData {
    FfiMeshData {
        positions: ptr::null_mut(),
        normals: ptr::null_mut(),
        material_ids: ptr::null_mut(),
        vertex_count: 0,
        indices: ptr::null_mut(),
        index_count: 0,
        submeshes: ptr::null_mut(),
        submesh_count: 0,
    }
}

pub(crate) fn empty_fluid_mesh_data() -> FfiFluidMeshData {
    FfiFluidMeshData {
        positions: ptr::null_mut(),
        normals: ptr::null_mut(),
        fluid_types: ptr::null_mut(),
        vertex_count: 0,
        indices: ptr::null_mut(),
        index_count: 0,
        uvs: ptr::null_mut(),
        flow_directions: ptr::null_mut(),
    }
}

pub(crate) fn converted_fluid_mesh_to_ffi(mesh: ConvertedFluidMesh) -> FfiFluidMeshData {
    let vertex_count = mesh.positions.len() as u32;
    let index_count = mesh.indices.len() as u32;

    let mut positions = mesh.positions.into_boxed_slice();
    let mut normals = mesh.normals.into_boxed_slice();
    let mut fluid_types = mesh.fluid_types.into_boxed_slice();
    let mut indices = mesh.indices.into_boxed_slice();
    let mut uvs = mesh.uvs.into_boxed_slice();
    let mut flow_directions = mesh.flow_directions.into_boxed_slice();

    let positions_ptr = positions.as_mut_ptr();
    let normals_ptr = normals.as_mut_ptr();
    let fluid_types_ptr = fluid_types.as_mut_ptr();
    let indices_ptr = indices.as_mut_ptr();
    let uvs_ptr = uvs.as_mut_ptr();
    let flow_directions_ptr = flow_directions.as_mut_ptr();

    std::mem::forget(positions);
    std::mem::forget(normals);
    std::mem::forget(fluid_types);
    std::mem::forget(indices);
    std::mem::forget(uvs);
    std::mem::forget(flow_directions);

    FfiFluidMeshData {
        positions: positions_ptr,
        normals: normals_ptr,
        fluid_types: fluid_types_ptr,
        vertex_count,
        indices: indices_ptr,
        index_count,
        uvs: uvs_ptr,
        flow_directions: flow_directions_ptr,
    }
}

/// Convert a Rust voxel-space position to a UE world-space `FfiVec3`.
pub(crate) fn rust_pos_to_ue(p: glam::Vec3, world_scale: f32) -> FfiVec3 {
    FfiVec3 {
        x: p.x * world_scale,
        y: -p.z * world_scale,
        z: p.y * world_scale,
    }
}
