//! Per-intent composers. Each function takes a Scene + cave-grid + probe
//! closure and returns an `Option<ShotCandidate>` for its specific intent.
//!
//! Block 1 ships `lava_descent` as a demonstrative implementation; the
//! remaining intents (bridge_traveling, water_flow_follow, etc.) are
//! filled in Block 2 with directorial polish (leading lines, depth
//! layering, etc.).

pub mod lava_descent;
