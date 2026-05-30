//! FFI surface for the voxel backend.
//!
//! This module was historically a single ~5.2k-line `api.rs`. It has been split
//! into behaviour-grouped submodules. Every `#[no_mangle] pub extern "C"` export
//! is re-exported here via `pub use <submodule>::*`, so the exported symbols and
//! all `crate::api::voxel_*` paths resolve exactly as before — `#[no_mangle]`
//! makes the linker symbol independent of the defining module, so the FFI ABI is
//! byte-for-byte unchanged.

// Imports kept at the umbrella level so the `#[cfg(test)] mod tests` child can
// reach them through `use super::*` exactly as it did when everything lived in
// one file.
#[allow(unused_imports)]
use std::ffi::{c_char, c_void, CString};
#[allow(unused_imports)]
use std::ptr;
#[allow(unused_imports)]
use crate::convert::rust_chunk_to_ue;
#[allow(unused_imports)]
use crate::engine::{ffi_scan_config_to_scan_config, VoxelEngine};
#[allow(unused_imports)]
use crate::types::*;

mod core_engine;
mod fluid;
mod stress_supports;
mod sleep;
mod world_scan;
mod morph;
mod brushes;
mod building_terrace;
mod locations;
mod diagnostics;
mod helpers;
mod save_load;
mod triggers;
mod pathing;
mod crystal_anchors;
mod world_memory_scenes;

pub use core_engine::*;
pub use fluid::*;
pub use stress_supports::*;
pub use sleep::*;
pub use world_scan::*;
pub use morph::*;
pub use brushes::*;
pub use building_terrace::*;
pub use locations::*;
pub use diagnostics::*;
pub use save_load::*;
pub use triggers::*;
pub use pathing::*;
pub use crystal_anchors::*;
pub use world_memory_scenes::*;

#[cfg(test)]
mod tests;
