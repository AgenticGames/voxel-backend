//! Creative-brush request FFI structs.

use super::*;

/// Creative-mode brush request. Sphere ops (paint/carve/fill) use this struct.
/// World coords/radius are in UE space; the worker converts them.
///
/// `mode`: 0 = paint material on solid voxels (no shape change)
///         1 = carve sphere (set solid → air)
///         2 = fill sphere (set air → solid with `material`, also overwrites
///             material on already-solid voxels in range)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushSphereRequest {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub radius: f32,
    pub mode: u8,
    pub material: u8,   // ignored for carve mode
    pub _pad: [u8; 2],
}

/// Tunnel-along-polyline brush. `points` are UE world coords; the worker
/// converts each point. If `material == 255` the tunnel carves; otherwise
/// it fills with that material.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushTunnelRequest {
    pub points: *const FfiVec3,
    pub point_count: u32,
    pub radius: f32,
    pub material: u8,
    pub _pad: [u8; 3],
}

/// Place a single hand-authored formation at a UE world position.
/// `formation_type`: 0=Stalactite, 1=Stalagmite, 2=Column, 3=Drapery,
///                   4=Flowstone, 5=Shield, 6=RimstoneDam
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushFormationRequest {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub formation_type: u8,
    pub material: u8,
    pub _pad: [u8; 2],
    pub height: f32,    // UE units
    pub radius: f32,    // UE units
}

/// Place a single mushroom instance at a UE world position. Does NOT touch
/// density — the brush picks the nearest solid voxel within `search_radius`
/// (UE units) as the anchor, infers the surface face from its air-neighbor
/// pattern, and inserts a `MushroomPlacement` into the chunk's store.
/// `kind` is the `MushroomKind` enum value (0=TurkeyTail, 1=Foxfire,
/// 2=GreenPepe, 3=GhostTower). `scale` is the instance scale; pass 0.0 to
/// use the kind's configured `scale_min..scale_max` random range.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushPlaceMushroomRequest {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub kind: u8,
    pub _pad: [u8; 3],
    pub search_radius: f32,  // UE units — radius to scan for an anchor voxel
    pub scale: f32,          // 0.0 = randomize per kind config
    pub yaw_radians: f32,    // 0.0 = randomize
}

/// Sphere-area mushroom brush — scatters multiple mushrooms within a radius.
/// `radius` is in UE units; `density` is 0..1 Bernoulli per viable surface
/// voxel. `kind` constrains placement to that species' preferred face
/// (TurkeyTail→walls, Foxfire→ceilings, GreenPepe/GhostTower→floors). `seed`
/// randomizes the pattern.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushPlaceMushroomSphereRequest {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub radius: f32,
    pub density: f32,
    pub clustering: f32, // 0..1 — local noise gate strength
    pub kind: u8,
    pub op: u8,          // 0=place, 1=erase (kind acts as filter; 255=any)
    pub _pad: [u8; 2],
    pub seed: u64,
}

/// Formation Stamp brush — runs the full worldgen formation pipeline
/// (random mix of stalactites/columns/drapery/etc. picked per the live
/// FormationConfig) on chunks overlapping a sphere, anchored within it.
/// `seed` randomizes the pick so re-stamping gives a different vibe.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushFormationStampRequest {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub radius: f32,    // UE units
    pub seed: u32,
}

/// Cavern Stamp brush — chunk-snapped cave generator. Runs worldgen worm
/// carving (additively — existing edits in the chunks survive) on a NxMxK
/// chunk region, optionally with lava tubes/rivers + pools/formations.
/// `chunk_x/y/z` is the lo-corner chunk in Rust chunk coords.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushCavernStampRequest {
    pub chunk_x: i32,
    pub chunk_y: i32,
    pub chunk_z: i32,
    pub extent_x: u8,
    pub extent_y: u8,
    pub extent_z: u8,
    pub decorate: u8,  // 0/1 — also run pools + formations
    pub fluids: u8,    // 0/1 — also run lava tubes + rivers
    pub seed: u32,
}

/// Axis-aligned-or-yawed box brush. `op`: 0=paint, 1=carve, 2=fill.
/// Half-extents in UE units. `yaw_deg`: rotation around UE vertical (Z) axis,
/// in degrees. 0 = AABB (no rotation).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushBoxRequest {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub half_x: f32,
    pub half_y: f32,
    pub half_z: f32,
    pub op: u8,
    pub material: u8,
    pub _pad: [u8; 2],
    pub yaw_deg: f32,
}

/// Y-axis-aligned cylinder brush. `op`: 0=paint, 1=carve, 2=fill.
/// `radius` and `height` in UE units; `height` is the full cylinder height.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushCylinderRequest {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub radius: f32,
    pub height: f32,
    pub op: u8,
    pub material: u8,
    pub _pad: [u8; 2],
}

/// Creative "OrePaint" brush — drops wall-exposed ore deposits inside the sphere
/// with even (Poisson-disk) spacing, weighted ore-type picks, and optional
/// inward "deep channel" tubes for each cluster. World coords are UE space.
/// Per-ore weights match `OreWeights` in `brushes.rs`:
/// `[iron, copper, malachite, tin, gold, diamond, kimberlite, sulfide,
///   quartz, pyrite, amethyst, crystal, coal]`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushOrePaintRequest {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub radius: f32,         // UE units — overall brush sphere
    pub cluster_size: f32,   // voxels — radius of each ore knob
    pub min_spacing: f32,    // voxels — minimum distance between cluster anchors
    pub channel_prob: f32,   // 0..1 — per-anchor chance to extend a deep channel
    pub channel_length: f32, // voxels — typical tube length into rock
    pub channel_radius: f32, // voxels — tube radius
    pub density: f32,        // 0..1 — fraction of wall candidates to keep as anchors
    pub seed: u32,
    pub weights: [u8; 13],   // per-ore frequency weights (see OreWeights ordering)
    pub _pad: [u8; 3],
}

/// Creative-mode "PaintStress" brush — additively writes into the per-voxel
/// painted-stress overlay (`StressField::painted_stress`) inside a sphere.
/// Does not change density/material, so no remesh is emitted; the new stress
/// is folded into `effective()` reads and drives extra collapses during sleep.
///
/// `op`:    0 = add, 1 = subtract, 2 = clear (zero the painted overlay inside the sphere)
/// `falloff`: 0 = constant, 1 = linear, 2 = smoothstep
/// `amount`: peak per-stroke additive (typical 0.2–0.8)
/// `cap`:    per-cell accumulation ceiling (typical 2.0)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushPaintStressRequest {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub radius: f32,    // UE units
    pub amount: f32,
    pub cap: f32,
    pub op: u8,         // 0=add, 1=sub, 2=clear
    pub falloff: u8,    // 0=constant, 1=linear, 2=smoothstep
    pub _pad: [u8; 2],
}

/// Smooth brush — Laplacian average of density in a sphere. Material preserved.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushSmoothRequest {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub radius: f32,
    pub iterations: u32,
    pub strength: f32,  // 0..1
}

/// Noise brush — perturb density by hash-based 3D noise within a sphere.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushNoiseRequest {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub radius: f32,
    pub frequency: f32,
    pub strength: f32,
    pub seed: u32,
}

/// Sphere fluid brush — places (or clears) fluid within a sphere.
/// `op`: 0=fill (level=1.0), 1=clear (level=0.0), 2=pool-dig (carve solid + fill bottom half non-source), 3=carve+full fill
/// `fluid_type`: 1=Water, 2=Lava, 3-9=specialized water sub-types
/// `is_source`: nonzero = treat placed fluid as an infinite source (spring); 0 = drains naturally
/// `max_flow_dist`: bounded-flow limit when placed as a source. 0 = unlimited (legacy
/// behavior). >0 = source's children stop propagating beyond this hop count, with
/// linear taper across the last `chunk::TAPER_HOPS` cells.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushFluidSphereRequest {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub radius: f32,
    pub fluid_type: u8,
    pub is_source: u8,
    pub op: u8,
    pub max_flow_dist: u8,
}

/// Box fluid brush — fills (or clears) fluid within an axis-aligned box.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushFluidBoxRequest {
    pub world_x: f32,
    pub world_y: f32,
    pub world_z: f32,
    pub half_x: f32,
    pub half_y: f32,
    pub half_z: f32,
    pub fluid_type: u8,
    pub is_source: u8,
    pub op: u8,            // 0=fill, 1=clear, 2=carve+fill
    pub max_flow_dist: u8, // 0 = unlimited
}

/// Capsule-chain (river/spline) fluid brush. Points are UE world coords.
/// Fills air voxels along the path; if `op == 2`, also carves the channel first.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBrushFluidRiverRequest {
    pub points: *const FfiVec3,
    pub point_count: u32,
    pub radius: f32,
    pub fluid_type: u8,
    pub is_source: u8,
    pub op: u8,            // 0=fill (only), 2=carve channel + fill
    pub max_flow_dist: u8, // 0 = unlimited
}

// (removed FfiBrushFluidStreamRequest — replaced by bounded sources via max_flow_dist on FluidCell)
