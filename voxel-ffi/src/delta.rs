//! World save/load: snapshot-based persistence for modified chunks.
//!
//! Modified chunks (mined, flattened, sleep-transformed) are saved as full
//! density snapshots. On load, snapshots replace generated densities before
//! hermite extraction, so the player sees their exact world state.

use std::collections::{BTreeMap, HashSet};
use std::io::{self, Read, Write};

use voxel_core::material::Material;
use voxel_core::octree::node::VoxelSample;
use voxel_core::stress::{StressField, SupportField, SupportType, STRUT_TUNING};
use voxel_gen::density::DensityField;

/// Magic bytes for the save file header.
const MAGIC: [u8; 4] = *b"MXSV";
/// Current binary format version.
///
/// Version history:
///   1 — chunk snapshots + terraced cells/columns
///   2 — adds editor collapse triggers + next_trigger_id (see triggers.rs)
///   3 — adds per-chunk painted-stress overlay (creative PaintStress brush)
///   4 — adds Crystal Anchor pending/grown state as JSON blob
///   5 — adds per-chunk mushroom placements (worldgen + painted)
///   6 — adds per-chunk supports + support_hp arrays (strut overhaul). Pre-v6
///       saves never persisted SupportField at all, so chunks reload with no
///       struts — visible AVoxelSupportActor actors in UE remain (loaded via
///       VoxelSaveManager) but contribute zero stress reduction until the
///       player re-places them. Acceptable: there were no shipped builds with
///       struts placed pre-v6.
///   7 — appends an opaque WorldMemory blob trailer (Scenes + history).
///       Pre-v7 saves load with empty WorldMemory; the drift thread
///       repopulates from live state in ~2 s. No regression.
const VERSION: u32 = 7;

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
    /// Optional per-voxel SupportType bytes (1 byte/cell). `None` = no struts
    /// placed (the common case — most chunks have zero struts). `Some(bytes)`
    /// = length size^3.
    pub supports: Option<Vec<u8>>,
    /// Optional per-voxel u16 HP. Always paired with `supports` when present —
    /// loading one without the other is a no-op. Length size^3 * 2 (LE u16).
    pub support_hp: Option<Vec<u8>>,
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
    /// Crystal Anchor manager state as JSON (introduced in v4). v1-v3 saves
    /// load this as an empty string — no anchors restored.
    pub crystal_anchors_json: String,
    /// Hand-painted + worldgen mushroom placements per chunk (v5+). On load,
    /// these replace the chunk's worldgen mushroom output entirely so painted
    /// /erased state persists across save→quit→reload. v1-v4 saves load this
    /// as an empty map.
    pub mushroom_placements: BTreeMap<(i32, i32, i32), Vec<voxel_gen::MushroomPlacement>>,
    /// Opaque WorldMemory state blob (v7+). voxel-world-memory carries its
    /// own internal magic + version inside this byte stream, so future
    /// WorldMemory schema changes don't need to bump `delta.rs::VERSION`.
    /// Pre-v7 saves load this as an empty Vec — the drift thread
    /// repopulates from live state on the first tick.
    pub world_memory_blob: Vec<u8>,
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
    /// `painted_stress` / `supports` / `support_hp` are `None` — see
    /// [`Self::from_chunk`] to also capture overlays + struts.
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
            supports: None,
            support_hp: None,
        }
    }

    /// Capture density+material AND the painted-stress overlay AND the
    /// support field (struts + HP) if present. Used by every brush so undo
    /// can restore the full chunk state, including any PaintStress strokes
    /// that touched the chunk AND any struts placed/broken in the region.
    pub fn from_chunk(
        df: &DensityField,
        sf: Option<&StressField>,
        supf: Option<&SupportField>,
    ) -> Self {
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
        if let Some(supf) = supf {
            if !supf.is_empty() {
                let n = supf.supports.len();
                let mut sbytes = Vec::with_capacity(n);
                for s in &supf.supports {
                    sbytes.push(*s as u8);
                }
                snap.supports = Some(sbytes);
                // HP array is lazy-allocated on first non-None set, so it
                // should be present whenever non_none_count > 0. Defensive:
                // emit a zero-filled array if it isn't.
                let hp_src: std::borrow::Cow<Vec<u16>> = if supf.support_hp.len() == n {
                    std::borrow::Cow::Borrowed(&supf.support_hp)
                } else {
                    std::borrow::Cow::Owned(vec![0u16; n])
                };
                let mut hp_bytes = Vec::with_capacity(n * 2);
                for &h in hp_src.iter() {
                    hp_bytes.extend_from_slice(&h.to_le_bytes());
                }
                snap.support_hp = Some(hp_bytes);
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

    /// Restore the support field (strut type + HP) from this snapshot. Mirrors
    /// `apply_painted_stress_to`. When `supports` is `None`, wipes any existing
    /// support entries (so undo of "first strut placement" reverts cleanly).
    ///
    /// Save format v6+ always uses the post-2026-05-26 SupportType IDs
    /// (Copper=1 .. Mithril=5). Pre-v6 saves never persisted supports at all.
    /// So this path uses `SupportType::from_u8` (no migration). For UE-side
    /// legacy actor bytes see `AVoxelSupportActor::MigrateLegacyId`.
    pub fn apply_supports_to(&self, supf: &mut SupportField) {
        let n = supf.supports.len();
        match (&self.supports, &self.support_hp) {
            (None, _) => {
                for i in 0..n {
                    supf.supports[i] = SupportType::None;
                }
                supf.support_hp.clear();
                supf.non_none_count = 0;
            }
            (Some(sbytes), hp_opt) => {
                if sbytes.len() != n { return; }
                // Decide once: do we have a valid HP array?
                let hp_bytes_opt: Option<&Vec<u8>> = hp_opt
                    .as_ref()
                    .filter(|hp| hp.len() == n * 2);
                if hp_bytes_opt.is_some() && supf.support_hp.len() != n {
                    supf.support_hp = vec![0u16; n];
                }
                let mut count = 0u32;
                for i in 0..n {
                    let stype = SupportType::from_u8(sbytes[i]);
                    supf.supports[i] = stype;
                    if stype != SupportType::None { count += 1; }
                    if let Some(hp_bytes) = hp_bytes_opt {
                        let off = i * 2;
                        let hp = u16::from_le_bytes([hp_bytes[off], hp_bytes[off + 1]]);
                        // If the load saw legacy IDs that got remapped, the HP
                        // value is whatever the old save persisted — clamp to
                        // the new tier's max so we don't end up with a Copper
                        // strut at 800 HP.
                        let max_hp = STRUT_TUNING[stype as u8 as usize].max_hp;
                        supf.support_hp[i] = if stype == SupportType::None {
                            0
                        } else {
                            hp.min(max_hp)
                        };
                    } else if stype != SupportType::None {
                        // No HP in save (was set() before HP existed) — refill
                        // to tier max so the strut starts fresh.
                        if supf.support_hp.is_empty() {
                            supf.support_hp = vec![0u16; n];
                        }
                        supf.support_hp[i] = STRUT_TUNING[stype as u8 as usize].max_hp;
                    }
                }
                supf.non_none_count = count;
            }
        }
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

        // v4: Crystal Anchor manager state. UTF-8 JSON, length-prefixed.
        let anchor_bytes = self.crystal_anchors_json.as_bytes();
        w.write_all(&(anchor_bytes.len() as u32).to_le_bytes())?;
        w.write_all(anchor_bytes)?;

        // v5: Mushroom placements — sparse per-chunk list. Each placement is
        // 36 bytes (8 f32 + 4 u8). Only chunks with at least one placement
        // get a record; worlds that never touched mushrooms pay 4 bytes
        // (the count u32).
        //
        // Format:
        //   [4] chunk_count u32
        //   per chunk:
        //     [4] cx i32, [4] cy i32, [4] cz i32
        //     [4] placement_count u32
        //     per placement (36 bytes):
        //       [4] x f32, [4] y f32, [4] z f32
        //       [4] nx f32, [4] ny f32, [4] nz f32
        //       [4] scale f32, [4] yaw f32
        //       [1] kind u8, [1] anchor_lx u8, [1] anchor_ly u8, [1] anchor_lz u8
        w.write_all(&(self.mushroom_placements.len() as u32).to_le_bytes())?;
        for (&(cx, cy, cz), placements) in &self.mushroom_placements {
            w.write_all(&cx.to_le_bytes())?;
            w.write_all(&cy.to_le_bytes())?;
            w.write_all(&cz.to_le_bytes())?;
            w.write_all(&(placements.len() as u32).to_le_bytes())?;
            for p in placements {
                w.write_all(&p.x.to_le_bytes())?;
                w.write_all(&p.y.to_le_bytes())?;
                w.write_all(&p.z.to_le_bytes())?;
                w.write_all(&p.normal_x.to_le_bytes())?;
                w.write_all(&p.normal_y.to_le_bytes())?;
                w.write_all(&p.normal_z.to_le_bytes())?;
                w.write_all(&p.scale.to_le_bytes())?;
                w.write_all(&p.yaw.to_le_bytes())?;
                w.write_all(&[p.kind as u8])?;
                w.write_all(&[p.anchor_lx])?;
                w.write_all(&[p.anchor_ly])?;
                w.write_all(&[p.anchor_lz])?;
            }
        }

        // v6: per-chunk SupportField (struts + HP). Sparse — only chunks with
        // at least one strut are emitted. Each entry has supports bytes
        // (size^3 * 1) + support_hp bytes (size^3 * 2). Chunks without strut
        // data write 0 in the count slot and skip the body.
        //
        // Format:
        //   [4] support_chunk_count u32
        //   per entry:
        //     [4] cx i32, [4] cy i32, [4] cz i32
        //     [4] support_byte_count u32   (= size^3)
        //     [size^3] supports bytes      (SupportType per cell)
        //     [4] hp_byte_count u32        (= size^3 * 2)
        //     [size^3 * 2] hp bytes        (LE u16 per cell)
        let support_entries: Vec<(&(i32, i32, i32), &Vec<u8>, &Vec<u8>)> = self
            .chunk_snapshots
            .iter()
            .filter_map(|(k, snap)| {
                match (snap.supports.as_ref(), snap.support_hp.as_ref()) {
                    (Some(s), Some(h)) => Some((k, s, h)),
                    _ => None,
                }
            })
            .collect();
        w.write_all(&(support_entries.len() as u32).to_le_bytes())?;
        for (&(cx, cy, cz), sbytes, hbytes) in support_entries {
            w.write_all(&cx.to_le_bytes())?;
            w.write_all(&cy.to_le_bytes())?;
            w.write_all(&cz.to_le_bytes())?;
            w.write_all(&(sbytes.len() as u32).to_le_bytes())?;
            w.write_all(sbytes)?;
            w.write_all(&(hbytes.len() as u32).to_le_bytes())?;
            w.write_all(hbytes)?;
        }

        // v7: opaque WorldMemory blob (Scenes + history). Carries its own
        // internal magic + version inside. v1-v6 saves load this as empty.
        //
        // Format:
        //   [4] blob_len u32  (0 = no scenes captured)
        //   [blob_len] bytes  (voxel_world_memory::serialize_blob output)
        w.write_all(&(self.world_memory_blob.len() as u32).to_le_bytes())?;
        w.write_all(&self.world_memory_blob)?;

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

        // Version. Accept v1..=7 (current). Older versions still load
        // with empty trailers (mushrooms, supports, world_memory_blob).
        let version = read_u32(r)?;
        if version < 1 || version > VERSION {
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
                ChunkSnapshot {
                    size, packed,
                    painted_stress: None,
                    supports: None,
                    support_hp: None,
                },
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
        let mut crystal_anchors_json = String::new();
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

        // v4: Crystal Anchor JSON blob (added 2026-05-20). v1-v3 saves end here.
        if version >= 4 {
            let json_len = read_u32(r)? as usize;
            if json_len > 4 * 1024 * 1024 {
                return Err(DeltaError::TruncatedData);
            }
            let mut bytes = vec![0u8; json_len];
            r.read_exact(&mut bytes).map_err(|_| DeltaError::TruncatedData)?;
            crystal_anchors_json = String::from_utf8(bytes).unwrap_or_default();
        }

        // v5: Per-chunk mushroom placements. v1-v4 saves end here; mushrooms
        // default to empty so worldgen will regenerate them on load.
        let mut mushroom_placements: BTreeMap<(i32, i32, i32), Vec<voxel_gen::MushroomPlacement>> =
            BTreeMap::new();
        if version >= 5 {
            let chunk_count = read_u32(r)? as usize;
            if chunk_count > 100_000 {
                return Err(DeltaError::TooManyChunks(chunk_count));
            }
            for _ in 0..chunk_count {
                let cx = read_i32(r)?;
                let cy = read_i32(r)?;
                let cz = read_i32(r)?;
                let placement_count = read_u32(r)? as usize;
                if placement_count > 100_000 {
                    return Err(DeltaError::TruncatedData);
                }
                let mut placements = Vec::with_capacity(placement_count);
                for _ in 0..placement_count {
                    let mut buf = [0u8; 4];
                    let read_f32 = |r: &mut R| -> Result<f32, DeltaError> {
                        let mut b = [0u8; 4];
                        r.read_exact(&mut b).map_err(|_| DeltaError::TruncatedData)?;
                        Ok(f32::from_le_bytes(b))
                    };
                    let x = read_f32(r)?;
                    let y = read_f32(r)?;
                    let z = read_f32(r)?;
                    let nx = read_f32(r)?;
                    let ny = read_f32(r)?;
                    let nz = read_f32(r)?;
                    let scale = read_f32(r)?;
                    let yaw = read_f32(r)?;
                    let mut kbuf = [0u8; 1];
                    r.read_exact(&mut kbuf).map_err(|_| DeltaError::TruncatedData)?;
                    let kind = voxel_gen::MushroomKind::from_u8(kbuf[0])
                        .ok_or(DeltaError::TruncatedData)?;
                    r.read_exact(&mut buf[..3]).map_err(|_| DeltaError::TruncatedData)?;
                    placements.push(voxel_gen::MushroomPlacement {
                        x, y, z,
                        normal_x: nx, normal_y: ny, normal_z: nz,
                        scale, yaw, kind,
                        anchor_lx: buf[0], anchor_ly: buf[1], anchor_lz: buf[2],
                    });
                }
                if !placements.is_empty() {
                    mushroom_placements.insert((cx, cy, cz), placements);
                }
            }
        }

        // v6: per-chunk SupportField (struts + HP). v1-v5 saves end here;
        // chunks reload with no struts (visible UE actors persist via
        // VoxelSaveManager but Rust-side stress reduction is gone until the
        // player re-places).
        if version >= 6 {
            let support_chunk_count = read_u32(r)? as usize;
            if support_chunk_count > 100_000 {
                return Err(DeltaError::TooManyChunks(support_chunk_count));
            }
            for _ in 0..support_chunk_count {
                let cx = read_i32(r)?;
                let cy = read_i32(r)?;
                let cz = read_i32(r)?;
                let s_count = read_u32(r)? as usize;
                if s_count > 256 * 256 * 256 {
                    return Err(DeltaError::TruncatedData);
                }
                let mut sbytes = vec![0u8; s_count];
                r.read_exact(&mut sbytes).map_err(|_| DeltaError::TruncatedData)?;
                let h_count = read_u32(r)? as usize;
                if h_count != s_count * 2 {
                    return Err(DeltaError::TruncatedData);
                }
                let mut hbytes = vec![0u8; h_count];
                r.read_exact(&mut hbytes).map_err(|_| DeltaError::TruncatedData)?;
                if let Some(snap) = chunk_snapshots.get_mut(&(cx, cy, cz)) {
                    snap.supports = Some(sbytes);
                    snap.support_hp = Some(hbytes);
                }
                // Drop entries for chunks without a density snapshot — they
                // can't be re-applied anyway.
            }
        }

        // v7: opaque WorldMemory blob. Pre-v7 saves load with empty Vec;
        // drift thread repopulates from live state on first tick.
        let world_memory_blob = if version >= 7 {
            let len = read_u32(r)? as usize;
            if len > 16 * 1024 * 1024 {
                return Err(DeltaError::TooManyChunks(len));
            }
            let mut buf = vec![0u8; len];
            r.read_exact(&mut buf).map_err(|_| DeltaError::TruncatedData)?;
            buf
        } else {
            Vec::new()
        };

        Ok(WorldSaveData {
            chunk_snapshots,
            terraced_cells,
            terraced_columns,
            triggers,
            next_trigger_id,
            crystal_anchors_json,
            mushroom_placements,
            world_memory_blob,
        })
    }

    /// Returns true if there is nothing to save.
    pub fn is_empty(&self) -> bool {
        self.chunk_snapshots.is_empty()
            && self.terraced_cells.is_empty()
            && self.terraced_columns.is_empty()
            && self.triggers.is_empty()
            && self.mushroom_placements.is_empty()
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
            // Densities MUST stay within [-1.0, 1.0]: ChunkSnapshot::apply_to
            // clamps to that band (the valid voxel-density range). The old
            // `i as f32 * 0.01` reached >1.0 (e.g. 1.01 at i=101, up to 3.42
            // at size=31) and clamped to 1.0 on round-trip, failing the
            // exact-equality check. Cycle i through [-1.00, 1.00] instead so
            // the f32 round-trip is exact while still exercising varied values.
            sample.density = ((i % 201) as i32 - 100) as f32 * 0.01;
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
    fn binary_roundtrip_mushroom_placements() {
        let mut save = WorldSaveData::default();
        let p1 = voxel_gen::MushroomPlacement {
            x: 5.5, y: 12.25, z: -3.75,
            normal_x: 1.0, normal_y: 0.0, normal_z: 0.0,
            scale: 0.85,
            yaw: 1.5708,
            kind: voxel_gen::MushroomKind::TurkeyTail,
            anchor_lx: 5, anchor_ly: 12, anchor_lz: 28,
        };
        let p2 = voxel_gen::MushroomPlacement {
            x: 18.0, y: 4.0, z: 22.5,
            normal_x: 0.0, normal_y: 1.0, normal_z: 0.0,
            scale: 1.3,
            yaw: 3.14,
            kind: voxel_gen::MushroomKind::GhostTower,
            anchor_lx: 18, anchor_ly: 4, anchor_lz: 22,
        };
        save.mushroom_placements.insert((0, 0, 0), vec![p1.clone(), p2.clone()]);
        save.mushroom_placements.insert((-2, 3, 7), vec![p1.clone()]);
        assert!(!save.is_empty());

        let bytes = save.serialize();
        let restored = WorldSaveData::deserialize(&bytes).unwrap();

        assert_eq!(restored.mushroom_placements.len(), 2);
        let chunk0 = &restored.mushroom_placements[&(0, 0, 0)];
        assert_eq!(chunk0.len(), 2);
        assert_eq!(chunk0[0].x, p1.x);
        assert_eq!(chunk0[0].normal_x, p1.normal_x);
        assert_eq!(chunk0[0].kind as u8, p1.kind as u8);
        assert_eq!(chunk0[0].anchor_lx, p1.anchor_lx);
        assert_eq!(chunk0[1].kind as u8, p2.kind as u8);
        assert_eq!(chunk0[1].yaw, p2.yaw);

        let chunk1 = &restored.mushroom_placements[&(-2, 3, 7)];
        assert_eq!(chunk1.len(), 1);
        assert_eq!(chunk1[0].scale, p1.scale);
    }

    #[test]
    fn painted_stress_save_load_roundtrip() {
        // Simulates: paint stress while loaded → unload (snapshot captures it)
        // → save (collect_save_data tier 2 grabs preserved_snapshots) →
        // serialize → deserialize → reload chunk → apply_pending_snapshot
        // restores via apply_painted_stress_to.
        //
        // Verifies the durability claim for the "paint then walk 20 chunks
        // away and save / load and walk back" scenarios.
        const SIZE: usize = 9;
        let df = make_test_density(SIZE);

        // Build a stress field with non-zero painted values at known cells.
        let mut sf = voxel_core::stress::StressField::new(SIZE);
        let pokes = [
            (2_usize, 3_usize, 4_usize, 0.45_f32),
            (5, 5, 5, 0.90),
            (7, 1, 6, 1.75),
        ];
        for &(x, y, z, v) in &pokes {
            sf.set_painted(x, y, z, v);
        }

        // Capture into a snapshot (what `unload` does for a dirty chunk),
        // tuck into a save, serialize to bytes, then go through the
        // round-trip the save file would do.
        let snap = ChunkSnapshot::from_chunk(&df, Some(&sf), None);
        assert!(snap.painted_stress.is_some(), "snapshot captured painted layer");

        let mut save = WorldSaveData::default();
        save.chunk_snapshots.insert((0, 0, 0), snap);

        let bytes = save.serialize();
        let restored_save = WorldSaveData::deserialize(&bytes).unwrap();
        let restored_snap = &restored_save.chunk_snapshots[&(0, 0, 0)];

        // Apply onto a fresh stress field — this is the path
        // apply_pending_snapshot uses when the chunk streams back in.
        let mut restored_sf = voxel_core::stress::StressField::new(SIZE);
        restored_snap.apply_painted_stress_to(&mut restored_sf);

        for &(x, y, z, v) in &pokes {
            let got = restored_sf.painted(x, y, z);
            assert!(
                (got - v).abs() < 1e-6,
                "painted value at ({x},{y},{z}) survived round-trip — expected {v}, got {got}"
            );
        }

        // Untouched cells stay zero.
        assert_eq!(restored_sf.painted(0, 0, 0), 0.0);
        assert_eq!(restored_sf.painted(8, 8, 8), 0.0);
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

    // ─── Strut persistence + migration tests (added 2026-05-26 overhaul) ──

    #[test]
    fn supports_capture_in_chunk_snapshot() {
        // ChunkSnapshot::from_chunk should capture struts + HP when supf is Some
        // and non-empty. Empty SupportField → supports/support_hp stay None.
        use voxel_core::stress::{SupportField, SupportType, STRUT_TUNING};

        let df = make_test_density(5);

        let empty_supf = SupportField::new(df.size);
        let snap_empty = ChunkSnapshot::from_chunk(&df, None, Some(&empty_supf));
        assert!(snap_empty.supports.is_none());
        assert!(snap_empty.support_hp.is_none());

        let mut supf = SupportField::new(df.size);
        supf.set(2, 2, 2, SupportType::Crystal);
        supf.set(3, 2, 2, SupportType::Mithril);
        let snap = ChunkSnapshot::from_chunk(&df, None, Some(&supf));
        assert!(snap.supports.is_some());
        assert!(snap.support_hp.is_some());

        // Roundtrip onto a fresh field and verify HP came back correctly.
        let mut restored = SupportField::new(df.size);
        snap.apply_supports_to(&mut restored);
        assert_eq!(restored.get(2, 2, 2), SupportType::Crystal);
        assert_eq!(restored.get(3, 2, 2), SupportType::Mithril);
        assert_eq!(
            restored.get_hp(2, 2, 2),
            STRUT_TUNING[SupportType::Crystal as usize].max_hp
        );
        assert_eq!(
            restored.get_hp(3, 2, 2),
            STRUT_TUNING[SupportType::Mithril as usize].max_hp
        );
        assert_eq!(restored.non_none_count, 2);
    }

    #[test]
    fn supports_legacy_id_migration() {
        // Save format v6 stores current IDs (1=Copper..5=Mithril), so the
        // delta apply path does NOT migrate. Legacy migration happens at
        // the UE-side actor save boundary (see AVoxelSupportActor::MigrateLegacyId)
        // and also in `request_place_support` for in-flight DLL/editor pairs.
        // Verify the Rust legacy helper directly.
        use voxel_core::stress::SupportType;

        assert_eq!(SupportType::from_legacy_u8(1), SupportType::Copper); // Slate
        assert_eq!(SupportType::from_legacy_u8(2), SupportType::Copper); // Granite
        assert_eq!(SupportType::from_legacy_u8(3), SupportType::Copper); // Limestone
        assert_eq!(SupportType::from_legacy_u8(4), SupportType::Copper); // Old Copper@4
        assert_eq!(SupportType::from_legacy_u8(5), SupportType::Iron);   // Old Iron@5
        assert_eq!(SupportType::from_legacy_u8(6), SupportType::Steel);  // Old Steel@6
        assert_eq!(SupportType::from_legacy_u8(7), SupportType::Crystal);// Old Crystal@7
    }

    #[test]
    fn save_format_v6_strut_roundtrip() {
        // Full save/deserialize cycle with struts present in two chunks.
        use voxel_core::stress::{SupportField, SupportType};

        let df1 = make_test_density(5);
        let mut supf1 = SupportField::new(df1.size);
        supf1.set(1, 1, 1, SupportType::Iron);

        let df2 = make_test_density(5);
        let mut supf2 = SupportField::new(df2.size);
        supf2.set(2, 2, 2, SupportType::Mithril);

        let mut save = WorldSaveData::default();
        save.chunk_snapshots.insert(
            (0, 0, 0), ChunkSnapshot::from_chunk(&df1, None, Some(&supf1))
        );
        save.chunk_snapshots.insert(
            (-1, 0, 0), ChunkSnapshot::from_chunk(&df2, None, Some(&supf2))
        );

        let bytes = save.serialize();
        let restored = WorldSaveData::deserialize(&bytes).unwrap();
        assert_eq!(restored.chunk_snapshots.len(), 2);

        let s1 = &restored.chunk_snapshots[&(0, 0, 0)];
        let mut sf1 = SupportField::new(df1.size);
        s1.apply_supports_to(&mut sf1);
        assert_eq!(sf1.get(1, 1, 1), SupportType::Iron);

        let s2 = &restored.chunk_snapshots[&(-1, 0, 0)];
        let mut sf2 = SupportField::new(df2.size);
        s2.apply_supports_to(&mut sf2);
        assert_eq!(sf2.get(2, 2, 2), SupportType::Mithril);
    }

    // ─── Block 1: v7 WorldMemory blob migration tests ─────────────────

    #[test]
    fn v7_empty_blob_roundtrips() {
        // A fresh save with no WorldMemory state — blob_len=0 trailer.
        let data = WorldSaveData::default();
        let bytes = data.serialize();
        let restored = WorldSaveData::deserialize(&bytes).expect("deserialize");
        assert!(restored.world_memory_blob.is_empty());
    }

    #[test]
    fn v7_populated_blob_roundtrips() {
        // Write a WorldSaveData with a synthetic 32-byte WM blob, read back.
        let blob: Vec<u8> = (0..32).map(|i| (i as u8).wrapping_mul(7)).collect();
        let data = WorldSaveData {
            world_memory_blob: blob.clone(),
            ..Default::default()
        };
        let bytes = data.serialize();
        let restored = WorldSaveData::deserialize(&bytes).expect("deserialize");
        assert_eq!(restored.world_memory_blob, blob);
    }

    #[test]
    fn v7_load_real_worldmemory_blob() {
        // Generate a real WM blob via voxel_world_memory, write a v7 save
        // containing it, read back, verify the WM round-trips.
        use voxel_world_memory::scene::{Scene, SceneKind};
        use voxel_world_memory::WorldMemory;

        let wm = WorldMemory::new();
        let mut s = Scene::new(
            wm.alloc_scene_id(),
            SceneKind::Lava,
            glam::Vec3::new(10.0, 20.0, 30.0),
        );
        s.score = 250.0;
        s.confidence = 0.95;
        s.chunks = vec![(1, 0, 0), (2, 0, 0)];
        wm.scenes.insert(s.id, s.clone());
        let blob = voxel_world_memory::persist::serialize_blob(&wm);
        assert!(!blob.is_empty());

        let data = WorldSaveData {
            world_memory_blob: blob,
            ..Default::default()
        };
        let bytes = data.serialize();
        let restored = WorldSaveData::deserialize(&bytes).expect("deserialize");

        let wm2 = WorldMemory::new();
        voxel_world_memory::persist::load_blob(&wm2, &restored.world_memory_blob)
            .expect("load blob");
        assert_eq!(wm2.tracked_scene_count(), 1);
        let restored_scene = wm2.scenes.get(&s.id).expect("scene present").value().clone();
        assert_eq!(restored_scene.kind, SceneKind::Lava);
        assert!((restored_scene.score - 250.0).abs() < 1e-3);
        assert_eq!(restored_scene.chunks, vec![(1, 0, 0), (2, 0, 0)]);
    }

    #[test]
    fn v7_garbage_blob_handled_gracefully() {
        // A v7 save with a corrupt WM blob — the delta loader passes the
        // bytes through; voxel-world-memory rejects them but the rest of
        // the save still loads fine.
        let garbage: Vec<u8> = (0..256).map(|i| ((i * 23) & 0xff) as u8).collect();
        let data = WorldSaveData {
            world_memory_blob: garbage.clone(),
            ..Default::default()
        };
        let bytes = data.serialize();
        // Delta layer should NOT reject garbage — it just carries the blob.
        let restored = WorldSaveData::deserialize(&bytes).expect("delta should not reject");
        assert_eq!(restored.world_memory_blob, garbage);

        // When the engine tries to apply it, voxel-world-memory rejects it
        // and starts empty. Verify by attempting to load.
        let wm = voxel_world_memory::WorldMemory::new();
        let load_result = voxel_world_memory::persist::load_blob(&wm, &restored.world_memory_blob);
        assert!(load_result.is_err()); // BadMagic
        assert_eq!(wm.tracked_scene_count(), 0); // no partial load
    }

    /// Build a v6-shaped save byte stream by hand so we can verify the
    /// v7 reader still accepts older formats. We construct the v7 writer
    /// output and then patch the version byte back to 6 + truncate the
    /// trailing world_memory_blob to simulate a v6 save.
    fn make_v6_save_bytes(data: &WorldSaveData) -> Vec<u8> {
        let mut bytes = data.serialize();
        // The world_memory_blob trailer is a u32 length + bytes. For an
        // empty blob that's exactly 4 trailing bytes of zero. Remove them
        // to mimic a v6 save (no trailer).
        let trailer_len = 4 + data.world_memory_blob.len();
        bytes.truncate(bytes.len() - trailer_len);
        // Patch the VERSION field (at offset 4..8) back to 6.
        bytes[4..8].copy_from_slice(&6u32.to_le_bytes());
        bytes
    }

    #[test]
    fn v6_save_loads_via_v7_reader() {
        // Synthetic save with terraced cells + crystal anchor JSON +
        // mushroom placements, but no WorldMemory blob (because v6).
        let mut data = WorldSaveData::default();
        data.terraced_cells = vec![(1, 2, 3), (4, 5, 6)];
        data.crystal_anchors_json = "{\"anchors\":[]}".to_string();
        let v6_bytes = make_v6_save_bytes(&data);

        let restored = WorldSaveData::deserialize(&v6_bytes).expect("v6 reads via v7 reader");
        assert_eq!(restored.terraced_cells, vec![(1, 2, 3), (4, 5, 6)]);
        assert_eq!(restored.crystal_anchors_json, "{\"anchors\":[]}");
        // v6 has no WorldMemory blob — must default to empty.
        assert!(restored.world_memory_blob.is_empty());
    }
}
