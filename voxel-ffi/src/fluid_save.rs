//! Fluid state persistence — serialize/deserialize the fluid simulation's
//! cell grid alongside the world delta save so player-placed lava and
//! water survive save/load cycles.
//!
//! The fluid thread owns its own state (level / fluid_type / is_source /
//! max_flow_dist per cell). At save we ask it for a `FluidSnapshot`, drop
//! cells that carry no useful information (level < threshold AND not a
//! source), and pack the survivors into a compact binary blob. At load we
//! parse the blob back into `PendingFluidCell` records and ship them to
//! the fluid thread, which applies them to each chunk the moment that
//! chunk's density arrives.
//!
//! Format (all little-endian):
//! ```text
//! [4]  magic  "MXFL"
//! [4]  version  u32 (1)
//! [4]  chunk_size  u32  (informational; loader trusts current chunk_size)
//! [4]  chunk_count  u32
//! per chunk:
//!   [4] cx i32, [4] cy i32, [4] cz i32
//!   [4] cell_count u32
//!   per cell:
//!     [4] cell_idx  u32  (z*size² + y*size + x)
//!     [4] level     f32
//!     [1] fluid_type u8
//!     [1] is_source  u8 (0/1)
//!     [1] max_flow_dist u8
//!     [1] _reserved u8 (zero, padding to 4-byte boundary)
//! ```

use std::collections::HashMap;
use std::io::{self, Read, Write};

use voxel_fluid::cell::{FluidCell, FluidType, MIN_LEVEL};
use voxel_fluid::{FluidSnapshot, PendingFluidCell};

const MAGIC: [u8; 4] = *b"MXFL";
const VERSION: u32 = 1;

/// One cell in the saved blob. Mirrors `PendingFluidCell` plus the chunk
/// key it belongs to.
#[derive(Debug, Clone, Copy)]
pub struct SavedFluidCell {
    pub idx: u32,
    pub fluid_type: FluidType,
    pub level: f32,
    pub is_source: bool,
    pub max_flow_dist: u8,
}

/// Drop cells that won't contribute meaningful state on restore.
///
/// Sources are always kept regardless of level (they regenerate to
/// MAX_LEVEL each tick). Non-sources need at least MIN_LEVEL to be worth
/// preserving — anything lower would evaporate immediately.
fn cell_worth_saving(cell: &FluidCell) -> bool {
    cell.is_source || cell.level > MIN_LEVEL
}

/// Build a save blob from a fluid snapshot. Chunks with no surviving
/// cells are dropped entirely (no per-chunk header overhead).
pub fn serialize(snapshot: &FluidSnapshot) -> Vec<u8> {
    let mut buf = Vec::new();
    let _ = write_to(snapshot, &mut buf);
    buf
}

fn write_to<W: Write>(snapshot: &FluidSnapshot, w: &mut W) -> io::Result<()> {
    w.write_all(&MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&(snapshot.chunk_size as u32).to_le_bytes())?;

    // First pass: filter chunks with any saveable cells, in deterministic
    // order so save bytes hash stably for tests / git-friendly diffs.
    let mut ordered: Vec<(&(i32, i32, i32), &Vec<FluidCell>)> = snapshot.chunks.iter().collect();
    ordered.sort_by_key(|(k, _)| **k);

    let mut chunks_to_write: Vec<(&(i32, i32, i32), Vec<(usize, &FluidCell)>)> = Vec::new();
    for (key, cells) in &ordered {
        let kept: Vec<(usize, &FluidCell)> = cells
            .iter()
            .enumerate()
            .filter(|(_, c)| cell_worth_saving(c))
            .collect();
        if !kept.is_empty() {
            chunks_to_write.push((*key, kept));
        }
    }

    w.write_all(&(chunks_to_write.len() as u32).to_le_bytes())?;
    for (key, cells) in chunks_to_write {
        w.write_all(&key.0.to_le_bytes())?;
        w.write_all(&key.1.to_le_bytes())?;
        w.write_all(&key.2.to_le_bytes())?;
        w.write_all(&(cells.len() as u32).to_le_bytes())?;
        for (idx, cell) in cells {
            w.write_all(&(idx as u32).to_le_bytes())?;
            w.write_all(&cell.level.to_le_bytes())?;
            w.write_all(&[cell.fluid_type as u8])?;
            w.write_all(&[if cell.is_source { 1 } else { 0 }])?;
            w.write_all(&[cell.max_flow_dist])?;
            w.write_all(&[0u8])?; // reserved padding
        }
    }
    Ok(())
}

/// Errors returned when parsing a fluid save blob.
#[derive(Debug)]
pub enum FluidSaveError {
    BadMagic,
    UnsupportedVersion(u32),
    Truncated,
    TooManyChunks(u32),
    TooManyCells(u32),
}

/// Parse a save blob into a per-chunk pending-fluid map ready to ship to
/// the fluid thread as `PendingFluidLoad` events.
pub fn deserialize(
    bytes: &[u8],
) -> Result<HashMap<(i32, i32, i32), Vec<PendingFluidCell>>, FluidSaveError> {
    let mut cur = io::Cursor::new(bytes);
    let mut magic = [0u8; 4];
    cur.read_exact(&mut magic).map_err(|_| FluidSaveError::Truncated)?;
    if magic != MAGIC {
        return Err(FluidSaveError::BadMagic);
    }
    let version = read_u32(&mut cur)?;
    if version != VERSION {
        return Err(FluidSaveError::UnsupportedVersion(version));
    }
    let _chunk_size = read_u32(&mut cur)?; // informational, loader uses live config
    let chunk_count = read_u32(&mut cur)?;
    if chunk_count > 1_000_000 {
        return Err(FluidSaveError::TooManyChunks(chunk_count));
    }

    let mut out: HashMap<(i32, i32, i32), Vec<PendingFluidCell>> = HashMap::new();
    for _ in 0..chunk_count {
        let cx = read_i32(&mut cur)?;
        let cy = read_i32(&mut cur)?;
        let cz = read_i32(&mut cur)?;
        let cell_count = read_u32(&mut cur)?;
        if cell_count > 10_000_000 {
            return Err(FluidSaveError::TooManyCells(cell_count));
        }
        // Cap the eager reservation (count is already hard-limited above, but
        // 10M cells * 16 B is ~160 MB a tiny truncated file could force).
        let mut cells = Vec::with_capacity((cell_count as usize).min(65_536));
        for _ in 0..cell_count {
            let idx = read_u32(&mut cur)?;
            let level = read_f32(&mut cur)?;
            let mut byte_buf = [0u8; 4];
            cur.read_exact(&mut byte_buf).map_err(|_| FluidSaveError::Truncated)?;
            let fluid_type = FluidType::from_u8(byte_buf[0]);
            let is_source = byte_buf[1] != 0;
            let max_flow_dist = byte_buf[2];
            // byte_buf[3] reserved
            cells.push(PendingFluidCell {
                idx,
                fluid_type,
                level,
                is_source,
                max_flow_dist,
            });
        }
        out.insert((cx, cy, cz), cells);
    }
    Ok(out)
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, FluidSaveError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|_| FluidSaveError::Truncated)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32<R: Read>(r: &mut R) -> Result<i32, FluidSaveError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|_| FluidSaveError::Truncated)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_f32<R: Read>(r: &mut R) -> Result<f32, FluidSaveError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|_| FluidSaveError::Truncated)?;
    Ok(f32::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot(chunk_size: usize) -> FluidSnapshot {
        let mut chunks = HashMap::new();
        let mut cells = vec![FluidCell::default(); chunk_size * chunk_size * chunk_size];
        cells[0].level = 0.5;
        cells[0].fluid_type = FluidType::Lava;
        cells[0].is_source = false;
        cells[5].level = 1.0;
        cells[5].fluid_type = FluidType::Water;
        cells[5].is_source = true;
        cells[5].max_flow_dist = 12;
        chunks.insert((1, 2, 3), cells);
        FluidSnapshot { chunks, chunk_size }
    }

    #[test]
    fn roundtrip_preserves_lava_and_water() {
        let snap = make_snapshot(8);
        let bytes = serialize(&snap);
        let parsed = deserialize(&bytes).expect("deserialize");
        let cells = &parsed[&(1, 2, 3)];
        assert_eq!(cells.len(), 2);
        let lava = cells.iter().find(|c| c.idx == 0).unwrap();
        assert_eq!(lava.fluid_type, FluidType::Lava);
        assert!((lava.level - 0.5).abs() < 1e-6);
        assert!(!lava.is_source);
        let spring = cells.iter().find(|c| c.idx == 5).unwrap();
        assert_eq!(spring.fluid_type, FluidType::Water);
        assert!(spring.is_source);
        assert_eq!(spring.max_flow_dist, 12);
    }

    #[test]
    fn drops_empty_cells() {
        let mut cells = vec![FluidCell::default(); 8 * 8 * 8];
        // Below MIN_LEVEL non-source: drop.
        cells[0].level = MIN_LEVEL * 0.5;
        cells[0].fluid_type = FluidType::Water;
        // Source at zero level: keep.
        cells[1].is_source = true;
        cells[1].fluid_type = FluidType::Lava;
        let mut chunks = HashMap::new();
        chunks.insert((0, 0, 0), cells);
        let snap = FluidSnapshot { chunks, chunk_size: 8 };
        let bytes = serialize(&snap);
        let parsed = deserialize(&bytes).expect("deserialize");
        let saved = &parsed[&(0, 0, 0)];
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0].idx, 1);
    }

    #[test]
    fn rejects_bad_magic() {
        let bytes = vec![0u8; 16];
        assert!(matches!(deserialize(&bytes), Err(FluidSaveError::BadMagic)));
    }
}
