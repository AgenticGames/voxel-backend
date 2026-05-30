//! FFI `#[repr(C)]` type definitions for the voxel engine DLL interface.
//!
//! Split from a single `types.rs` into domain submodules. Every submodule is
//! glob re-exported here so `crate::types::*` and named imports
//! (`crate::types::FfiEngineConfig`, etc.) resolve unchanged across the workspace.
//! The split is a pure relocation: no field order, padding, repr, or impl changed.

mod core_geom;
mod decoration;
mod stress;
mod engine_config;
mod brushes;
mod sleep;
mod scene;
mod scan_path;
mod worker_msgs;

pub use core_geom::*;
pub use decoration::*;
pub use stress::*;
pub use engine_config::*;
pub use brushes::*;
pub use sleep::*;
pub use scene::*;
pub use scan_path::*;
pub use worker_msgs::*;
