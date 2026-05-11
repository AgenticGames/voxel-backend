//! World save/load: snapshot-based persistence for modified chunks.
//!
//! Modified chunks (mined, flattened, sleep-transformed) are saved as full
//! density snapshots. On load, snapshots replace generated densities before
//! hermite extraction, so the player sees their exact world state.

use std::collections::{BTreeMap, HashSet};
use std::io::{self, Read, Write};

use voxel_core::material::Material;
use voxel_core::octree::node::VoxelSample;
use voxel_core::stress::StressField;
use voxel_gen::density::DensityField;

/// Magic bytes for the save file header.
const MAGIC: [u8; 4] = *b"MXSV";
/// Current binary format version.
///
/// Version history:
///   1 — chunk snapshots + terraced cells/columns
///   2 — adds editor collapse triggers + next_trigger_id (see triggers.rs)
///   3 — adds per-chunk painted-stress overlay (creative PaintStress brush)
const VERSION: u32 = 3;

// ── Data structures ────────────────────────────────────────────────────

/// Packed density snapshot for a single chunk.
#[derive(Debug, Clone)]
pub struct ChunkSnapshot {
    /// Grid size (chunk_size + 1, e.g. 31 for chunk_size=30).
    pub size: u32,
    /// Packed samples: 5 bytes each (4 f32 density LE + 1 u8 material).
    /// Length = size^3 * 5.
    pub packed: Vec<u8>,
    /// Optional painted-stress overlay (creative PaintStress brush).
    /// `None` = snapshot was captured before any painted-stress layer existed.
    /// `Some(bytes)` = length size^3 * 4 (LE f32 per voxel). On apply, this
    /// overwrites the chunk's `StressField::painted_stress` in full.
    pub painted_stress: Option<Vec<u8>>,
}

/// All world modification data needed to restore a saved game.
#[derive(Debug, Clone, Default)]
pub struct WorldSaveData {
    /// Full density snapshots keyed by chunk coordinate.
    pub chunk_snapshots: BTreeMap<(i32, i32, i32), ChunkSnapshot>,
    /// Terraced (flattened) cell positions in world-voxel coords.
    pub terraced_cells: Vec<(i32, i32, i32)>,
    /// Terraced column floor heights: (world_x, world_z) → floor_y.
    pub terraced_columns: BTreeMap<(i32, i32), i32>,
    /// Editor-authored collapse triggers (introduced in v2). v1 saves
    /// deserialize this as an empty vec.
    pub triggers: Vec<crate::triggers::EditorCollapseTrigger>,
    /// Monotonic trigger id counter. Persists so reload doesn't recycle
    /// ids that already exist. v1 reads this as 1.
    pub next_trigger_id: u32,
}

/// Tracks which chunks have been modified at runtime so we know what to save.
#[derive(Debug, Clone, Default)]
pub struct ModificationTracker {
    /// Chunk keys that have been modified since load/creation.
    pub dirty_chunks: HashSet<(i32, i32, i32)>,
}

// ── Snapshot capture & apply ───────────────────────────────────────────

impl ChunkSnapshot {
    /// Capture a snapshot from a live DensityField (density+material only).
    /// `painted_stress` is `None` — see [`Self::from_chunk`] to also capture
    /// the painted-stress overlay.
    pub fn from_density(df: &DensityField) -> Self {
        let total = df.samples.len();
        let mut packed = Vec::with_capacity(total * 5);
        for sample in &df.samples {
            packed.extend_from_slice(&sample.density.to_le_bytes());
            packed.push(sample.material as u8);
        }
        ChunkSnapshot {
            size: df.size as u32,
            packed,
            painted_stress: None,
        }
    }

    /// Capture density+material AND the painted-stress overlay if `sf` has one.
    /// Used by every brush so undo can restore the full chunk state, including
    /// any PaintStress strokes that touched the chunk.
    pub fn from_chunk(df: &DensityField, sf: Option<&StressField>) -> Self {
        let mut snap = Self::from_density(df);
        if let Some(sf) = sf {
            if sf.has_painted_layer() {
                let mut bytes = Vec::with_capacity(sf.painted_stress.len() * 4);
                for &v in &sf.painted_stress {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                snap.painted_stress = Some(bytes);
            }
        }
        snap
    }

    /// Apply this snapshot onto a DensityField, overwriting all samples.
    /// The target field must already be the correct size (matching self.size).
    /// Returns false if sizes don't match.
    pub fn apply_to(&self, df: &mut DensityField) -> bool {
        if df.size != self.size as usize {
            return false;
        }
        let total = df.samples.len();
        if self.packed.len() != total * 5 {
            return false;
        }
        for (i, sample) in df.samples.iter_mut().enumerate() {
            let offset = i * 5;
            let density_bytes: [u8; 4] = [
                self.packed[offset],
                self.packed[offset + 1],
                self.packed[offset + 2],
                self.packed[offset + 3],
            ];
            let raw = f32::from_le_bytes(density_bytes);
            // Clamp on load: existing saves from buggy versions can contain
            // density values outside [-1, 1] (e.g. -5.6 from accumulated
            // noise-brush strokes). DC's edge-intersection math behaves
            // poorly with extreme values — clamping on snapshot apply heals
            // those saves transparently. NaN passes through unchanged so
            // the diagnostic can still surface them.
            sample.density = if raw.is_nan() { raw } else { raw.clamp(-1.0, 1.0) };
            sample.material = Material::from_u8(self.packed[offset + 4]);
        }
        df.compute_metadata();
        true
    }

    /// Number of samples in this snapshot.
    pub fn sample_count(&self) -> usize {
        let s = self.size as usize;
        s * s * s
    }

    /// Restore the painted-stress overlay on `sf` from this snapshot.
    ///
    /// * `painted_stress == None` → wipes `sf.painted_stress` back to empty
    ///   (so undo of "first PaintStress stroke" reverts the layer's existence).
    /// * `painted_stress == Some(bytes)` → unpacks into `sf.painted_stress`,
    ///   allocating it if needed. Mismatched byte count is a silent no-op.
    pub fn apply_painted_stress_to(&self, sf: &mut StressField) {
        let n = sf.size * sf.size * sf.size;
        match &self.painted_stress {
            None => {
                // No painted layer at snapshot time — drop the current one entirely.
                sf.painted_stress = Vec::new();
            }
            Some(bytes) => {
                if bytes.len() != n * 4 {
                    return;
                }
                if sf.painted_stress.len() != n {
                    sf.painted_stress = vec![0.0; n];
                }
                for i in 0..n {
                    let off = i * 4;
                    let v = f32::from_le_bytes([
                        bytes[off],
                        bytes[off + 1],
                        bytes[off + 2],
                        bytes[off + 3],
                    ]);
                    sf.painted_stress[i] = v;
                }
            }
        }
    }
}

impl ModificationTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark a chunk as modified (call after mine/flatten/sleep/slab).
    pub fn mark_dirty(&mut self, key: (i32, i32, i32)) {
        self.dirty_chunks.insert(key);
    }

    /// Mark multiple chunks as modified.
    pub fn mark_dirty_many(&mut self, keys: &[(i32, i32, i32)]) {
        for &k in keys {
            self.dirty_chunks.insert(k);
        }
    }

    /// Remove a chunk from tracking (e.g. when unloaded and snapshot preserved).
    pub fn remove(&mut self, key: &(i32, i32, i32)) {
        self.dirty_chunks.remove(key);
    }

    /// Check if any modifications exist.
    pub fn has_modifications(&self) -> bool {
        !self.dirty_chunks.is_empty()
    }
}

// ── Binary serialization ───────────────────────────────────────────────

impl WorldSaveData {
    /// Serialize to a compact binary format.
    ///
    /// Format:
    /// ```text
    /// [4] magic "MXSV"
    /// [4] version u32
    /// [4] chunk_count u32
    /// per chunk:
    ///   [4] cx i32
    ///   [4] cy i32
    ///   [4] cz i32
    ///   [4] grid_size u32
    ///   [grid_size^3 * 5] packed samples
    /// [4] terrace_cell_count u32
    /// per cell:
    ///   [4] wx i32, [4] wy i32, [4] wz i32
    /// [4] terrace_column_count u32
    /// per column:
    ///   [4] wx i32, [4] wz i32, [4] floor_y i32
    /// ```
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.write_to(&mut buf).expect("Vec write cannot fail");
        buf
    }

    /// Write binary data to a writer.
    pub fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_all(&MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;

        // Chunk snapshots
        w.write_all(&(self.chunk_snapshots.len() as u32).to_le_bytes())?;
        for (&(cx, cy, cz), snap) in &self.chunk_snapshots {
            w.write_all(&cx.to_le_bytes())?;
            w.write_all(&cy.to_le_bytes())?;
            w.write_all(&cz.to_le_bytes())?;
            w.write_all(&snap.size.to_le_bytes())?;
            w.write_all(&snap.packed)?;
        }

        // Terraced cells
        w.write_all(&(self.terraced_cells.len() as u32).to_le_bytes())?;
        for &(wx, wy, wz) in &self.terraced_cells {
            w.write_all(&wx.to_le_bytes())?;
            w.write_all(&wy.to_le_bytes())?;
            w.write_all(&wz.to_le_bytes())?;
        }

        // Terraced columns
        w.write_all(&(self.terraced_columns.len() as u32).to_le_bytes())?;
        for (&(wx, wz), &floor_y) in &self.terraced_columns {
            w.write_all(&wx.to_le_bytes())?;
            w.write_all(&wz.to_le_bytes())?;
            w.write_all(&floor_y.to_le_bytes())?;
        }

        // v2: editor collapse triggers + next_trigger_id
        w.write_all(&self.next_trigger_id.to_le_bytes())?;
        w.write_all(&(self.triggers.len() as u32).to_le_bytes())?;
        for trig in &self.triggers {
            write_trigger(w, trig)?;
        }

        // v3: painted-stress overlays — a sparse list of (chunk_coord, bytes)
        // entries. Only chunks whose snapshot has Some(painted_stress) appear,
        // so worlds that never used the PaintStress brush pay almost nothing.
        //
        // Format:
        //   [4] painted_count u32
        //   per entry:
        //     [4] cx i32, [4] cy i32, [4] cz i32
        //     [4] byte_count u32   (= size^3 * 4)
        //     [byte_count] bytes   (LE f32 per voxel)
        // The chunk_size for the overlay must match the chunk_snapshot above —
        // we don't re-emit `size` since it'd duplicate the density snapshot's.
        let painted_entries: Vec<(&(i32, i32, i32), &Vec<u8>)> = self
            .chunk_snapshots
            .iter()
            .filter_map(|(k, snap)| snap.painted_stress.as_ref().map(|b| (k, b)))
            .collect();
        w.write_all(&(painted_entries.len() as u32).to_le_bytes())?;
        for (&(cx, cy, cz), bytes) in painted_entries {
            w.write_all(&cx.to_le_bytes())?;
            w.write_all(&cy.to_le_bytes())?;
            w.write_all(&cz.to_le_bytes())?;
            w.write_all(&(bytes.len() as u32).to_le_bytes())?;
            w.write_all(bytes)?;
        }

        Ok(())
    }

    /// Deserialize from binary data.
    pub fn deserialize(data: &[u8]) -> Result<Self, DeltaError> {
        let mut cursor = io::Cursor::new(data);
        Self::read_from(&mut cursor)
    }

    /// Read binary data from a reader.
    pub fn read_from<R: Read>(r: &mut R) -> Result<Self, DeltaError> {
        // Magic
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic).map_err(|_| DeltaError::TruncatedData)?;
        if magic != MAGIC {
            return Err(DeltaError::BadMagic);
        }

        // Version. Accept v1 (legacy, no triggers), v2 (triggers), v3 (current — adds painted_stress).
        let version = read_u32(r)?;
        if version != 1 && version != 2 && version != VERSION {
            return Err(DeltaError::UnsupportedVersion(version));
        }

        // Chunk snapshots
        let chunk_count = read_u32(r)? as usize;
        if chunk_count > 100_000 {
            return Err(DeltaError::TooManyChunks(chunk_count));
        }
        let mut chunk_snapshots = BTreeMap::new();
        for _ in 0..chunk_count {
            let cx = read_i32(r)?;
            let cy = read_i32(r)?;
            let cz = read_i32(r)?;
            let size = read_u32(r)?;
            if size == 0 || size > 256 {
                return Err(DeltaError::InvalidChunkSize(size));
            }
            let total_samples = (size as usize).pow(3);
            let byte_count = total_samples * 5;
            let mut packed = vec![0u8; byte_count];
            r.read_exact(&mut packed).map_err(|_| DeltaError::TruncatedData)?;
            chunk_snapshots.insert(
                (cx, cy, cz),
                ChunkSnapshot { size, packed, painted_stress: None },
            );
        }

        // Terraced cells
        let cell_count = read_u32(r)? as usize;
        if cell_count > 10_000_000 {
            return Err(DeltaError::TooManyTerraces(cell_count));
        }
        let mut terraced_cells = Vec::with_capacity(cell_count);
        for _ in 0..cell_count {
            let wx = read_i32(r)?;
            let wy = read_i32(r)?;
            let wz = read_i32(r)?;
            terraced_cells.push((wx, wy, wz));
        }

        // Terraced columns
        let col_count = read_u32(r)? as usize;
        if col_count > 10_000_000 {
            return Err(DeltaError::TooManyTerraces(col_count));
        }
        let mut terraced_columns = BTreeMap::new();
        for _ in 0..col_count {
            let wx = read_i32(r)?;
            let wz = read_i32(r)?;
            let floor_y = read_i32(r)?;
            terraced_columns.insert((wx, wz), floor_y);
        }

        // v2: editor collapse triggers. v1 saves end here; default to empty.
        let (triggers, next_trigger_id) = if version >= 2 {
            let next_id = read_u32(r)?;
            let trig_count = read_u32(r)? as usize;
            if trig_count > 100_000 {
                return Err(DeltaError::TooManyChunks(trig_count));
            }
            let mut triggers = Vec::with_capacity(trig_count);
            for _ in 0..trig_count {
                triggers.push(read_trigger(r)?);
            }
            (triggers, next_id.max(1))
        } else {
            (Vec::new(), 1)
        };

        // v3: per-chunk painted-stress overlays. v1/v2 saves end here.
        if version >= 3 {
            let painted_count = read_u32(r)? as usize;
            if painted_count > 100_000 {
                return Err(DeltaError::TooManyChunks(painted_count));
            }
            for _ in 0..painted_count {
                let cx = read_i32(r)?;
                let cy = read_i32(r)?;
                let cz = read_i32(r)?;
                let byte_count = read_u32(r)? as usize;
                if byte_count > 256 * 256 * 256 * 4 {
                    return Err(DeltaError::TruncatedData);
                }
                let mut bytes = vec![0u8; byte_count];
                r.read_exact(&mut bytes).map_err(|_| DeltaError::TruncatedData)?;
                if let Some(snap) = chunk_snapshots.get_mut(&(cx, cy, cz)) {
                    snap.painted_stress = Some(bytes);
                }
                // Painted overlay for a chunk that has no density snapshot is
                // dropped — the overlay only matters once the density side exists.
            }
        }

        Ok(WorldSaveData {
            chunk_snapshots,
            terraced_cells,
            terraced_columns,
            triggers,
            next_trigger_id,
        })
    }

    /// Returns true if there is nothing to save.
    pub fn is_empty(&self) -> bool {
        self.chunk_snapshots.is_empty()
            && self.terraced_cells.is_empty()
            && self.terraced_columns.is_empty()
            && self.triggers.is_empty()
    }
}

// ── Errors ─────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum DeltaError {
    BadMagic,
    UnsupportedVersion(u32),
    TruncatedData,
    TooManyChunks(usize),
    TooManyTerraces(usize),
    InvalidChunkSize(u32),
    Io(io::Error),
}

impl std::fmt::Display for DeltaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeltaError::BadMagic => write!(f, "bad magic bytes (expected MXSV)"),
            DeltaError::UnsupportedVersion(v) => write!(f, "unsupported version {v}"),
            DeltaError::TruncatedData => write!(f, "unexpected end of data"),
            DeltaError::TooManyChunks(n) => write!(f, "too many chunks ({n})"),
            DeltaError::TooManyTerraces(n) => write!(f, "too many terraces ({n})"),
            DeltaError::InvalidChunkSize(s) => write!(f, "invalid chunk size ({s})"),
            DeltaError::Io(e) => write!(f, "IO error: {e}"),
        }
    }
}

impl From<io::Error> for DeltaError {
    fn from(e: io::Error) -> Self {
        DeltaError::Io(e)
    }
}

// ── Reader helpers ─────────────────────────────────────────────────────

fn read_u32<R: Read>(r: &mut R) -> Result<u32, DeltaError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|_| DeltaError::TruncatedData)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32<R: Read>(r: &mut R) -> Result<i32, DeltaError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|_| DeltaError::TruncatedData)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_f32<R: Read>(r: &mut R) -> Result<f32, DeltaError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|_| DeltaError::TruncatedData)?;
    Ok(f32::from_le_bytes(buf))
}

// ── Trigger (de)serialization ──────────────────────────────────────────
//
// Format per trigger (all little-endian):
//   [4] id u32
//   [4] name_len u32, [name_len] name utf8 bytes
//   [1] armed u8
//   [4] fall_distance_uu f32
//   [1] activation_tag u8  (0 = OnFirstMine, 1 = OnPillarLoss)
//     OnFirstMine: [24] trigger_volume (6×i32)
//     OnPillarLoss:
//       [4] pillar_count u32
//       per pillar: [24] volume (6×i32) + [4] baseline_solid u32
//       [1] condition_tag u8 (0=Any, 1=N, 2=All), [1] n_value u8
//       [4] loss_threshold f32
//   [4] slab_count u32, per voxel: [12] (3×i32)
//   [4] pile_chunk_count u32, per chunk: [12] (3×i32)

fn write_aabb<W: Write>(w: &mut W, aabb: &crate::triggers::VoxelAabb) -> io::Result<()> {
    w.write_all(&aabb.min.0.to_le_bytes())?;
    w.write_all(&aabb.min.1.to_le_bytes())?;
    w.write_all(&aabb.min.2.to_le_bytes())?;
    w.write_all(&aabb.max.0.to_le_bytes())?;
    w.write_all(&aabb.max.1.to_le_bytes())?;
    w.write_all(&aabb.max.2.to_le_bytes())?;
    Ok(())
}

fn read_aabb<R: Read>(r: &mut R) -> Result<crate::triggers::VoxelAabb, DeltaError> {
    let mn = (read_i32(r)?, read_i32(r)?, read_i32(r)?);
    let mx = (read_i32(r)?, read_i32(r)?, read_i32(r)?);
    Ok(crate::triggers::VoxelAabb { min: mn, max: mx })
}

fn write_trigger<W: Write>(
    w: &mut W,
    t: &crate::triggers::EditorCollapseTrigger,
) -> io::Result<()> {
    use crate::triggers::{LossCondition, TriggerActivation};

    w.write_all(&t.id.to_le_bytes())?;
    let name_bytes = t.name.as_bytes();
    w.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
    w.write_all(name_bytes)?;
    w.write_all(&[if t.armed { 1 } else { 0 }])?;
    w.write_all(&t.fall_distance_uu.to_le_bytes())?;

    match &t.activation {
        TriggerActivation::OnFirstMine { trigger_volume } => {
            w.write_all(&[0u8])?;
            write_aabb(w, trigger_volume)?;
        }
        TriggerActivation::OnPillarLoss {
            pillars,
            condition,
            loss_threshold,
        } => {
            w.write_all(&[1u8])?;
            w.write_all(&(pillars.len() as u32).to_le_bytes())?;
            for p in pillars {
                write_aabb(w, &p.volume)?;
                w.write_all(&p.baseline_solid.to_le_bytes())?;
            }
            match condition {
                LossCondition::AnyPillar => w.write_all(&[0u8, 0u8])?,
                LossCondition::NPillars(n) => w.write_all(&[1u8, *n])?,
                LossCondition::AllPillars => w.write_all(&[2u8, 0u8])?,
            }
            w.write_all(&loss_threshold.to_le_bytes())?;
        }
    }

    w.write_all(&(t.target_slab_voxels.len() as u32).to_le_bytes())?;
    for &(x, y, z) in &t.target_slab_voxels {
        w.write_all(&x.to_le_bytes())?;
        w.write_all(&y.to_le_bytes())?;
        w.write_all(&z.to_le_bytes())?;
    }

    w.write_all(&(t.pile_chunks.len() as u32).to_le_bytes())?;
    for &(x, y, z) in &t.pile_chunks {
        w.write_all(&x.to_le_bytes())?;
        w.write_all(&y.to_le_bytes())?;
        w.write_all(&z.to_le_bytes())?;
    }
    Ok(())
}

fn read_trigger<R: Read>(
    r: &mut R,
) -> Result<crate::triggers::EditorCollapseTrigger, DeltaError> {
    use crate::triggers::{
        EditorCollapseTrigger, LossCondition, PillarRef, TriggerActivation,
    };

    let id = read_u32(r)?;
    let name_len = read_u32(r)? as usize;
    if name_len > 1024 {
        return Err(DeltaError::TruncatedData);
    }
    let mut name_buf = vec![0u8; name_len];
    r.read_exact(&mut name_buf).map_err(|_| DeltaError::TruncatedData)?;
    let name = String::from_utf8(name_buf).unwrap_or_else(|_| String::new());
    let mut armed_byte = [0u8; 1];
    r.read_exact(&mut armed_byte).map_err(|_| DeltaError::TruncatedData)?;
    let armed = armed_byte[0] != 0;
    let fall_distance_uu = read_f32(r)?;

    let mut tag = [0u8; 1];
    r.read_exact(&mut tag).map_err(|_| DeltaError::TruncatedData)?;
    let activation = match tag[0] {
        0 => {
            let trigger_volume = read_aabb(r)?;
            TriggerActivation::OnFirstMine { trigger_volume }
        }
        1 => {
            let pillar_count = read_u32(r)? as usize;
            if pillar_count > 256 {
                return Err(DeltaError::TruncatedData);
            }
            let mut pillars = Vec::with_capacity(pillar_count);
            for _ in 0..pillar_count {
                let volume = read_aabb(r)?;
                let baseline_solid = read_u32(r)?;
                pillars.push(PillarRef { volume, baseline_solid });
            }
            let mut cond_buf = [0u8; 2];
            r.read_exact(&mut cond_buf).map_err(|_| DeltaError::TruncatedData)?;
            let condition = match cond_buf[0] {
                0 => LossCondition::AnyPillar,
                1 => LossCondition::NPillars(cond_buf[1]),
                _ => LossCondition::AllPillars,
            };
            let loss_threshold = read_f32(r)?;
            TriggerActivation::OnPillarLoss {
                pillars,
                condition,
                loss_threshold,
            }
        }
        _ => return Err(DeltaError::TruncatedData),
    };

    let slab_count = read_u32(r)? as usize;
    if slab_count > 10_000_000 {
        return Err(DeltaError::TooManyTerraces(slab_count));
    }
    let mut target_slab_voxels = Vec::with_capacity(slab_count);
    for _ in 0..slab_count {
        let x = read_i32(r)?;
        let y = read_i32(r)?;
        let z = read_i32(r)?;
        target_slab_voxels.push((x, y, z));
    }

    let pile_count = read_u32(r)? as usize;
    if pile_count > 1_000_000 {
        return Err(DeltaError::TooManyTerraces(pile_count));
    }
    let mut pile_chunks = Vec::with_capacity(pile_count);
    for _ in 0..pile_count {
        let x = read_i32(r)?;
        let y = read_i32(r)?;
        let z = read_i32(r)?;
        pile_chunks.push((x, y, z));
    }

    Ok(EditorCollapseTrigger {
        id,
        name,
        armed,
        activation,
        target_slab_voxels,
        pile_chunks,
        fall_distance_uu,
    })
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_density(size: usize) -> DensityField {
        let mut df = DensityField::new(size);
        for (i, sample) in df.samples.iter_mut().enumerate() {
            sample.density = i as f32 * 0.01;
            sample.material = if i % 3 == 0 {
                Material::Granite
            } else if i % 3 == 1 {
                Material::Air
            } else {
                Material::Iron
            };
        }
        df
    }

    #[test]
    fn snapshot_roundtrip() {
        let df = make_test_density(5);
        let snap = ChunkSnapshot::from_density(&df);

        let mut restored = DensityField::new(5);
        assert!(snap.apply_to(&mut restored));

        for (i, (orig, rest)) in df.samples.iter().zip(restored.samples.iter()).enumerate() {
            assert_eq!(orig.density, rest.density, "density mismatch at {i}");
            assert_eq!(orig.material, rest.material, "material mismatch at {i}");
        }
    }

    #[test]
    fn snapshot_size_mismatch() {
        let df = make_test_density(5);
        let snap = ChunkSnapshot::from_density(&df);
        let mut wrong_size = DensityField::new(7);
        assert!(!snap.apply_to(&mut wrong_size));
    }

    #[test]
    fn binary_roundtrip_empty() {
        let data = WorldSaveData::default();
        assert!(data.is_empty());
        let bytes = data.serialize();
        let restored = WorldSaveData::deserialize(&bytes).unwrap();
        assert!(restored.is_empty());
    }

    #[test]
    fn binary_roundtrip_with_data() {
        let df1 = make_test_density(5);
        let df2 = make_test_density(7);

        let mut save = WorldSaveData::default();
        save.chunk_snapshots.insert((0, 0, 0), ChunkSnapshot::from_density(&df1));
        save.chunk_snapshots.insert((-3, 5, 1), ChunkSnapshot::from_density(&df2));
        save.terraced_cells = vec![(10, 20, 30), (-5, 0, 100)];
        save.terraced_columns.insert((10, 30), 20);
        save.terraced_columns.insert((-5, 100), 0);

        let bytes = save.serialize();
        let restored = WorldSaveData::deserialize(&bytes).unwrap();

        assert_eq!(restored.chunk_snapshots.len(), 2);
        assert!(restored.chunk_snapshots.contains_key(&(0, 0, 0)));
        assert!(restored.chunk_snapshots.contains_key(&(-3, 5, 1)));
        assert_eq!(restored.terraced_cells.len(), 2);
        assert_eq!(restored.terraced_cells[0], (10, 20, 30));
        assert_eq!(restored.terraced_columns.len(), 2);
        assert_eq!(restored.terraced_columns[&(10, 30)], 20);

        // Verify snapshot content roundtrips
        let snap = &restored.chunk_snapshots[&(0, 0, 0)];
        let mut check = DensityField::new(5);
        assert!(snap.apply_to(&mut check));
        for (i, (orig, rest)) in df1.samples.iter().zip(check.samples.iter()).enumerate() {
            assert_eq!(orig.density, rest.density, "chunk0 density at {i}");
            assert_eq!(orig.material, rest.material, "chunk0 material at {i}");
        }
    }

    #[test]
    fn binary_bad_magic() {
        let mut bytes = WorldSaveData::default().serialize();
        bytes[0] = b'X';
        assert!(matches!(
            WorldSaveData::deserialize(&bytes),
            Err(DeltaError::BadMagic)
        ));
    }

    #[test]
    fn modification_tracker() {
        let mut tracker = ModificationTracker::new();
        assert!(!tracker.has_modifications());

        tracker.mark_dirty((0, 0, 0));
        assert!(tracker.has_modifications());

        tracker.mark_dirty_many(&[(1, 0, 0), (0, 1, 0)]);
        assert_eq!(tracker.dirty_chunks.len(), 3);

        tracker.remove(&(0, 0, 0));
        assert_eq!(tracker.dirty_chunks.len(), 2);
    }

    #[test]
    fn realistic_chunk_size_roundtrip() {
        // Use chunk_size=30 (live UE config), grid size = 31
        let df = make_test_density(31);
        let snap = ChunkSnapshot::from_density(&df);
        assert_eq!(snap.sample_count(), 31 * 31 * 31);
        // 29791 samples * 5 bytes = 148955 bytes (~145 KB)
        assert_eq!(snap.packed.len(), 29791 * 5);

        let mut save = WorldSaveData::default();
        save.chunk_snapshots.insert((0, 0, 0), snap);
        let bytes = save.serialize();
        // Header(12) + chunk_header(16) + packed(148955) + terraces(8) = ~149KB
        assert!(bytes.len() > 148_000);
        assert!(bytes.len() < 150_000);

        let restored = WorldSaveData::deserialize(&bytes).unwrap();
        let rsnap = &restored.chunk_snapshots[&(0, 0, 0)];
        let mut check = DensityField::new(31);
        assert!(rsnap.apply_to(&mut check));
        for (i, (orig, rest)) in df.samples.iter().zip(check.samples.iter()).enumerate() {
            assert_eq!(orig.density, rest.density, "density at {i}");
            assert_eq!(orig.material, rest.material, "material at {i}");
        }
    }
}
