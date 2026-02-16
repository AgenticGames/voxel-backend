//! FFI stubs for future UE5 integration.
//!
//! Gated behind `#[cfg(feature = "ffi")]`.
//! These functions expose chunk generation and mesh extraction
//! to C/C++ callers via raw pointers.

use std::ptr;
use crate::chunk::Chunk;
use crate::mesh::{Mesh, Vertex, Triangle};

/// Opaque chunk data returned to FFI callers.
pub struct ChunkData {
    pub chunk: Chunk,
    pub mesh: Mesh,
}

/// Buffer of mesh data suitable for passing to a rendering engine.
#[repr(C)]
pub struct MeshBuffer {
    pub vertices: *const Vertex,
    pub vertex_count: u32,
    pub triangles: *const Triangle,
    pub triangle_count: u32,
}

/// Generate a chunk at the given coordinates and return a heap-allocated ChunkData.
///
/// # Safety
/// The caller must eventually call `free_chunk_ffi` on the returned pointer.
/// Returns null if generation fails.
#[no_mangle]
pub unsafe extern "C" fn generate_chunk_ffi(
    _seed: u64,
    _cx: i32,
    _cy: i32,
    _cz: i32,
) -> *mut ChunkData {
    // Stub: actual implementation will wire into voxel_gen::pipeline::generate
    ptr::null_mut()
}

/// Get mesh data from a previously generated chunk.
///
/// # Safety
/// `chunk` must be a valid pointer returned from `generate_chunk_ffi`.
/// The returned MeshBuffer borrows from the ChunkData and is only valid
/// while the ChunkData has not been freed.
#[no_mangle]
pub unsafe extern "C" fn get_mesh_data_ffi(chunk: *const ChunkData) -> MeshBuffer {
    if chunk.is_null() {
        return MeshBuffer {
            vertices: ptr::null(),
            vertex_count: 0,
            triangles: ptr::null(),
            triangle_count: 0,
        };
    }
    let data = &*chunk;
    MeshBuffer {
        vertices: data.mesh.vertices.as_ptr(),
        vertex_count: data.mesh.vertices.len() as u32,
        triangles: data.mesh.triangles.as_ptr(),
        triangle_count: data.mesh.triangles.len() as u32,
    }
}

/// Free a ChunkData previously returned by `generate_chunk_ffi`.
///
/// # Safety
/// `chunk` must be a valid pointer returned from `generate_chunk_ffi`,
/// or null (which is a no-op).
#[no_mangle]
pub unsafe extern "C" fn free_chunk_ffi(chunk: *mut ChunkData) {
    if !chunk.is_null() {
        drop(Box::from_raw(chunk));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_stub_returns_null() {
        let ptr = unsafe { generate_chunk_ffi(42, 0, 0, 0) };
        assert!(ptr.is_null());
    }

    #[test]
    fn get_mesh_from_null_returns_empty() {
        let buf = unsafe { get_mesh_data_ffi(ptr::null()) };
        assert!(buf.vertices.is_null());
        assert_eq!(buf.vertex_count, 0);
        assert_eq!(buf.triangle_count, 0);
    }

    #[test]
    fn free_null_is_noop() {
        unsafe { free_chunk_ffi(ptr::null_mut()) };
    }
}
