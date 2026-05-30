//! Creative-mode terrain authoring brushes: paint material, sphere carve, sphere fill,
//! tunnel spline carve/fill, and formation placement.
//!
//! All brushes mirror the mining mutation pattern in `mining.rs`:
//!  1. Iterate chunks overlapping the brush region
//!  2. Mutate `DensityField.samples` (density and/or material)
//!  3. Track per-chunk dirty bounds (with `dirty_expand`)
//!  4. Sync boundary density across seams
//!  5. `modification_tracker.mark_dirty_many()` for save persistence
//!  6. `store.remesh_dirty()` to produce updated meshes
//!
//! Brushes are intentionally simpler than mining — no mined-material counts,
//! no Laplacian smoothing, no SDF gradient blending (callers can mine first,
//! then paint, if they want a smoothed border).
//!
//! This module was split from a single `brushes.rs` into a submodule folder.
//! The split is behavior-preserving: pure code movement plus visibility
//! widening and re-exports. Every `crate::brushes::X` path resolves unchanged.

mod common;
mod formations;
mod fluid;
mod mushroom;
mod ore;
mod primitives;
mod sphere;

#[cfg(test)]
mod tests;

pub use common::{BrushOutcome, UndoStroke};
pub use formations::*;
pub use fluid::*;
pub use mushroom::*;
pub use ore::*;
pub use primitives::*;
pub use sphere::*;
