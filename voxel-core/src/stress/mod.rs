//! Structural stress + collapse system.
//!
//! This module was split out of a single large `stress.rs` file into a
//! submodule folder for maintainability. The split is **purely mechanical /
//! behavior-preserving**: every item lives at the same public path as before
//! via the `pub use` re-exports below, so external paths like
//! `voxel_core::stress::StressConfig` / `voxel_core::stress::calc_voxel_stress`
//! resolve unchanged.
//!
//! Submodule layout:
//! - [`events`]  — dirty-event regions + surface/source classification consts
//! - [`types`]   — field types, support tuning tables, hardness consts, collapse/result structs
//! - [`config`]  — [`StressConfig`] (+ its serde array helper)
//! - [`calc`]    — v1 per-voxel stress, sampling helpers, ground-connectivity pass
//! - [`calc_v2`] — v2 two-pass stress calc + region-recalc entry points
//! - [`collapse`]— collapse detection + execution (v1 + v2 slab pipeline)
//! - [`update`]  — high-level post-change update orchestration

mod events;
mod types;
mod config;
mod calc;
mod calc_v2;
mod collapse;
mod update;

#[cfg(test)]
mod tests;

// ── Re-exports: preserve the flat `stress::X` public surface unchanged ──
pub use events::*;
pub use types::*;
pub use config::*;
pub use calc::*;
pub use calc_v2::*;
pub use collapse::*;
pub use update::*;
