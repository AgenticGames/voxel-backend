//! Internal (non-FFI) converted-mesh types + worker request/result enums.

use super::*;

// ── Internal (non-FFI) types ──

/// Converted mesh data in UE coordinate space, ready to be handed out via FFI.
pub struct ConvertedMesh {
    pub positions: Vec<FfiVec3>,
    pub normals: Vec<FfiVec3>,
    pub material_ids: Vec<u8>,
    pub indices: Vec<u32>,
    pub submeshes: Vec<FfiSubmesh>,
    /// Per-vertex morph reveal time in [0,1] (0 = revealed first, 1 = last).
    /// Empty for normal (non-morph) meshes. Baked by the sleep-montage morph
    /// path so the GPU material can dissolve the mesh in over MorphProgress
    /// instead of the CPU re-meshing every step. Reordered alongside positions
    /// in `bucket_mesh_by_material`.
    pub reveal_t: Vec<f32>,
}

/// Converted fluid mesh data in UE coordinate space.
pub struct ConvertedFluidMesh {
    pub positions: Vec<FfiVec3>,
    pub normals: Vec<FfiVec3>,
    pub fluid_types: Vec<u8>,
    pub indices: Vec<u32>,
    pub uvs: Vec<[f32; 2]>,
    pub flow_directions: Vec<FfiVec3>,
}

/// Messages sent to worker threads.
pub enum WorkerRequest {
    /// Apply a lava↔water quench plan from the fluid sim.
    /// Writes Obsidian voxels at the contact rim and Scoria voxels in the
    /// inward halo, then remeshes the affected chunks. Sources are NOT in
    /// this list — they stay alive and grow via the fluid sim's pillow state.
    ApplyLavaQuench {
        obsidian: Vec<((i32, i32, i32), usize, usize, usize)>,
        scoria: Vec<((i32, i32, i32), usize, usize, usize)>,
        drained_water: Vec<((i32, i32, i32), usize, usize, usize)>,
    },
    Generate {
        chunk: (i32, i32, i32),
        generation: u64,
    },
    PriorityGenerate {
        chunk: (i32, i32, i32),
        generation: u64,
    },
    Mine {
        request: FfiMineRequest,
    },
    Flatten {
        base_x: i32,
        base_y: i32,
        base_z: i32,
        host_material: u8,
    },
    FlattenBatch {
        tiles: Vec<(glam::IVec3, voxel_core::material::Material)>,
    },
    BuildingFlatten {
        base_x: i32,
        base_y: i32,
        base_z: i32,
        // Exact sub-voxel Y from UE (in voxel units). The integer base_y is
        // floor(base_y_float). Used by the SDF flatten to position the iso
        // surface within sub-voxel precision so buildings don't float/sink.
        base_y_float: f32,
        host_material: u8,
        footprint_voxels: i32,
        clearance_voxels: i32,
    },
    BuildingFlattenBatch {
        // (base_x, base_y, base_z, base_y_float, host_material, footprint, clearance)
        buildings: Vec<(i32, i32, i32, f32, u8, i32, i32)>,
    },
    Unload {
        chunk: (i32, i32, i32),
    },
    PlaceSupport {
        world_x: i32,
        world_y: i32,
        world_z: i32,
        support_type: u8,
    },
    RemoveSupport {
        world_x: i32,
        world_y: i32,
        world_z: i32,
    },
    Sleep {
        player_chunk: (i32, i32, i32),
        sleep_count: u32,
        sleep_config: voxel_sleep::SleepConfig,
    },
    AureoleOnly {
        player_chunk: (i32, i32, i32),
        sleep_config: voxel_sleep::SleepConfig,
    },
    WorldScan,
    WorldScanWithConfig {
        config: voxel_core::world_scan::ScanConfig,
    },
    ForceSpawnPool {
        world_x: f32,
        world_y: f32,
        world_z: f32,
        fluid_type: u8,
    },
    MineAndFillFluid {
        world_x: f32,
        world_y: f32,
        world_z: f32,
        radius: f32,
        fluid_type: u8,
        world_scale: f32,
    },
    MorphStep {
        chunks: Vec<(i32, i32, i32)>,
        step: u32,
        total_steps: u32,
    },
    /// Sphere brush (paint/carve/fill) at a UE world position.
    BrushSphere {
        request: FfiBrushSphereRequest,
    },
    /// Tunnel-along-polyline brush.
    BrushTunnel {
        /// Points in Rust world coords (already converted from UE).
        points: Vec<glam::Vec3>,
        radius: f32,           // Rust world units
        material: Option<u8>,  // None = carve, Some(u8) = fill with material
    },
    /// Place a single formation primitive at a UE world position.
    BrushFormation {
        center_rust: glam::Vec3,
        formation_type: u8,
        material: u8,
        height: f32,   // Rust world units
        radius: f32,   // Rust world units
    },
    /// Formation Stamp brush — runs the full worldgen formation pipeline,
    /// spatially clipped to a sphere, with per-call randomized seed.
    BrushFormationStamp {
        center_rust: glam::Vec3,
        radius: f32,   // Rust world units
        seed: u32,
    },
    /// Cavern Stamp brush — chunk-snapped cave generator over NxMxK chunks.
    BrushCavernStamp {
        chunk_origin: (i32, i32, i32),
        extent: (u8, u8, u8),
        decorate: bool,
        fluids: bool,
        seed: u32,
    },
    /// Diagnostic action: re-sync a single chunk's boundaries with all 6
    /// face neighbors and re-mesh anything modified. Used by the UE
    /// "Force Resync" button to repair seam mismatches without a full
    /// quit+reload — handy when the in-memory state has drifted from
    /// neighbors via mid-session brush/snapshot/regen paths.
    ForceChunkResync { chunk: (i32, i32, i32) },
    /// Axis-aligned-or-yawed box brush.
    BrushBox {
        center_rust: glam::Vec3,
        half_ext_rust: glam::Vec3,
        yaw_rad: f32,
        op: u8,
        material: u8,
    },
    /// Y-axis-aligned cylinder brush.
    BrushCylinder {
        center_rust: glam::Vec3,
        radius: f32,    // Rust world units
        height: f32,    // Rust world units
        op: u8,
        material: u8,
    },
    /// Smooth brush.
    BrushSmooth {
        center_rust: glam::Vec3,
        radius: f32,
        iterations: u32,
        strength: f32,
    },
    /// Noise brush.
    BrushNoise {
        center_rust: glam::Vec3,
        radius: f32,
        frequency: f32,
        strength: f32,
        seed: u32,
    },
    /// Creative "OrePaint" brush — wall-exposed ore deposits + optional channels.
    BrushOrePaint {
        center_rust: glam::Vec3,
        radius: f32,           // Rust world units
        cluster_size: f32,     // voxels
        min_spacing: f32,      // voxels
        channel_prob: f32,     // 0..1
        channel_length: f32,   // voxels
        channel_radius: f32,   // voxels
        density: f32,          // 0..1
        seed: u32,
        weights: crate::brushes::OreWeights,
    },
    /// Creative "PaintStress" brush — additive sphere over the painted-stress overlay.
    BrushPaintStress {
        center_rust: glam::Vec3,
        radius: f32,    // Rust world units
        amount: f32,
        cap: f32,
        op: u8,
        falloff: u8,
    },
    /// Creative "Clear all painted stress" — wipes the painted overlay in
    /// every loaded chunk's StressField. No spatial argument; this is the
    /// "nuke everything" eraser for when the per-sphere Clear op is too
    /// fiddly. Atelier exposes a single button.
    BrushClearAllPaintedStress,
    /// Undo the most recent brush stroke (any creative brush).
    BrushUndo,
    /// Sphere fluid brush.
    BrushFluidSphere {
        center_rust: glam::Vec3,
        radius: f32,
        fluid_type: u8,
        is_source: bool,
        op: u8,         // 0=fill, 1=clear, 2=pool-dig, 3=carve+full fill
        max_flow_dist: u8,
    },
    /// Box fluid brush.
    BrushFluidBox {
        center_rust: glam::Vec3,
        half_ext_rust: glam::Vec3,
        fluid_type: u8,
        is_source: bool,
        op: u8,         // 0=fill, 1=clear, 2=carve+fill
        max_flow_dist: u8,
    },
    /// Place a single mushroom at the cursor position (creative brush).
    BrushPlaceMushroom {
        center_rust: glam::Vec3,
        kind: u8,
        search_radius: f32, // Rust voxel units
        scale: f32,         // 0 = randomize
        yaw: f32,           // 0 = randomize
    },
    /// Sphere-area mushroom brush — scatters N mushrooms of one kind within
    /// a radius via Bernoulli sampling against surface voxels of the kind's
    /// preferred face. `clustering` (0..1) raises a local Simplex-noise gate
    /// so high values produce tight family pockets instead of flat scatter.
    BrushPlaceMushroomSphere {
        center_rust: glam::Vec3,
        radius: f32,    // Rust voxel units
        density: f32,   // 0..1
        clustering: f32, // 0..1
        kind: u8,
        seed: u64,
    },
    /// Sphere-area mushroom eraser. `kind_filter == 255` erases any kind;
    /// otherwise only matching kind is removed.
    BrushEraseMushroomSphere {
        center_rust: glam::Vec3,
        radius: f32,    // Rust voxel units
        kind_filter: u8,
    },
    /// River (capsule chain) fluid brush.
    BrushFluidRiver {
        points: Vec<glam::Vec3>,
        radius: f32,
        fluid_type: u8,
        is_source: bool,
        op: u8,         // 0=fill, 2=carve channel + fill
        max_flow_dist: u8,
    },
    // (removed BrushFluidStream — replaced by bounded sources via max_flow_dist on FluidCell)
    /// Pathfinding request — runs A* against the live ChunkStore on a path
    /// worker thread, sends back `WorkerResult::PathComputed`. The internal
    /// request is pre-converted to Rust voxel coordinates by the FFI entry.
    ComputePath {
        request: crate::pathing::PathRequestInternal,
    },
}

/// Results sent back from worker threads.
pub enum WorkerResult {
    ChunkMesh {
        chunk: (i32, i32, i32),
        mesh: ConvertedMesh,
        generation: u64,
        crystal_data: Vec<FfiCrystalPlacement>,
        mushroom_data: Vec<FfiMushroomInstance>,
        zone_descriptors: Vec<FfiZoneDescriptor>,
    },
    Error {
        chunk: (i32, i32, i32),
        generation: u64,
    },
    MinedMaterials {
        mined: FfiMinedMaterials,
    },
    /// All mine mesh updates in one atomic result — prevents pop-in
    MineBatchMesh {
        meshes: Vec<((i32, i32, i32), ConvertedMesh, Vec<FfiCrystalPlacement>, Vec<FfiMushroomInstance>)>,
    },
    FluidMesh {
        chunk: (i32, i32, i32),
        mesh: ConvertedFluidMesh,
    },
    SolidifyRequest {
        positions: Vec<((i32, i32, i32), usize, usize, usize)>,
    },
    /// Live lava↔water quench plan from the fluid sim — apply Obsidian +
    /// Scoria voxel writes, drain water cells, and remesh affected chunks.
    LavaQuench {
        obsidian: Vec<((i32, i32, i32), usize, usize, usize)>,
        scoria: Vec<((i32, i32, i32), usize, usize, usize)>,
        drained_water: Vec<((i32, i32, i32), usize, usize, usize)>,
    },
    /// Collapse EVENTS only. Remeshed geometry must never ride this variant:
    /// `voxel_poll_result` can return exactly one FfiResult per call, so
    /// piggybacked meshes were Box::into_raw-leaked without ever reaching UE
    /// (and remesh_dirty output is base-only — delivering it raw would wipe
    /// seams, the cd07682 quench lesson). Send meshes via ChunkMesh /
    /// batched_seam_pass like every other path.
    CollapseResult {
        events: Vec<FfiCollapseEvent>,
    },
    /// Support place/remove ack. Struts change the stress model, not the
    /// density field — there is no geometry to remesh (relief lands via
    /// queue_stress_dirty). See CollapseResult note for why no mesh payload.
    SupportResult {
        success: bool,
    },
    SleepComplete {
        chunks_changed: u32,
        voxels_metamorphosed: u32,
        minerals_grown: u32,
        supports_degraded: u32,
        collapses_triggered: u32,
        acid_dissolved: u32,
        veins_deposited: u32,
        voxels_enriched: u32,
        formations_grown: u32,
        sulfide_dissolved: u32,
        coal_matured: u32,
        diamonds_formed: u32,
        voxels_silicified: u32,
        nests_fossilized: u32,
        channels_eroded: u32,
        corpses_fossilized: u32,
        lava_solidified: u32,
        profile_report: String,
        aureole_glimpse_pos: Option<(i32, i32, i32)>,
        aureole_showcase_block: Option<Vec<(i32, i32, i32)>>,
        manifest_json: String,
        lava_cells: Vec<(i32, i32, i32)>,
        surface_changed_cells: Vec<(i32, i32, i32)>,
        surface_step_activity: [u16; voxel_sleep::SURFACE_ACTIVITY_BUCKETS],
    },
    MorphMeshes {
        step: u32,
        total_steps: u32,
        meshes: Vec<ConvertedMesh>,
    },
    ScanComplete {
        json_report: String,
    },
    ForceSpawnPoolComplete {
        json_report: String,
    },
    /// V2 stress warnings for live feedback (dust/creak/shake positions).
    StressWarnings {
        warnings: Vec<FfiStressWarning>,
    },
    /// V2 collapse result with coherent slab data for animated falling.
    CollapseSlabResult {
        events: Vec<FfiCollapseEventV2>,
        meshes: Vec<((i32, i32, i32), ConvertedMesh)>,
    },
    /// One falling-slab visual: real DC mesh + fall metadata. Cinematic Act 3.
    /// Each fragment of a multi-fragment collapse becomes one of these.
    SlabFall {
        mesh: ConvertedMesh,
        fall_data: FfiSlabFallData,
    },
    /// Localized pre-collapse warning. Cinematic Acts 1-2.
    CollapseWarning {
        center_ue: (f32, f32, f32),
        bounds_extent_ue: (f32, f32, f32),
        severity: u8, // 1=dust, 2=creak, 3=shake, 4=imminent
        eta_ms: u32,
        volume: u32,
    },
    /// One tier of the 4-tier pile-buildup preview. UE buffers by spawn loc,
    /// builds a debris actor when it has 4 tiers, animates reveal.
    PilePreviewTier {
        mesh: ConvertedMesh,
        fall_data: FfiSlabFallData,
    },
    /// A* result for a path request — stashed into the engine's
    /// `path_results` map keyed by `request_id` and never touched by mesh
    /// consumers (intercepted in `poll_result`).
    PathComputed {
        request_id: u32,
        status: u8,
        nodes_ue: Vec<crate::pathing::PathNodeUE>,
    },
    /// One or more struts hit 0 HP. UE should look up the corresponding
    /// `AVoxelSupportActor` via `PlacedSupports` (keyed by voxel position),
    /// play the breaking VFX, and refresh the crack overlay in a sphere
    /// around the strut so the newly-unsupported walls flash red.
    StrutsBroken {
        struts: Vec<FfiStrutBroken>,
    },
}
