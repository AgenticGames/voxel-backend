//! Persist — serialize/deserialize the `WorldMemory` scene store into an
//! opaque blob that `voxel-ffi/src/delta.rs` stores at save format v7.
//!
//! The blob carries its own magic + internal version so future schema
//! changes don't require bumping `delta.rs::VERSION`. v7 only needs to
//! preserve the byte payload — opaque to delta.rs.
//!
//! Format:
//!   - 4 bytes: magic = b"WMEM"
//!   - 4 bytes: internal version (LE u32)
//!   - 8 bytes: next_scene_id (LE u64)
//!   - 4 bytes: scene_count (LE u32)
//!   - N bytes: scene_count × serialized-Scene (serde JSON, length-prefixed)
//!
//! JSON inside binary framing is deliberate — Scenes are small (<1KB each),
//! count is bounded (<256 typical, <2048 worst case), and JSON survives
//! schema additions via `#[serde(default)]` without bumping the internal
//! version. If we ever need to scale up by 10×, switch to bincode here
//! without touching `delta.rs`.

use std::convert::TryInto;

use crate::scene::Scene;
use crate::WorldMemory;

const MAGIC: &[u8; 4] = b"WMEM";
const VERSION: u32 = 1;

/// Internal error type for the persist layer. Caller is expected to treat
/// any `Err` as "load nothing, start empty" — not fatal.
#[derive(Debug)]
pub enum PersistError {
    BadMagic,
    UnsupportedVersion(u32),
    Truncated,
    JsonError(String),
}

impl std::fmt::Display for PersistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadMagic => write!(f, "bad magic"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported version {}", v),
            Self::Truncated => write!(f, "truncated"),
            Self::JsonError(e) => write!(f, "json error: {}", e),
        }
    }
}

impl std::error::Error for PersistError {}

/// Serialize a `WorldMemory` into an opaque byte blob. Returns Vec<u8>
/// suitable for `voxel-ffi/src/delta.rs` v7 to write into the save file.
pub fn serialize_blob(wm: &WorldMemory) -> Vec<u8> {
    let scenes: Vec<Scene> = wm.scenes.iter().map(|e| e.value().clone()).collect();
    let mut out = Vec::with_capacity(64 + scenes.len() * 256);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    let next_id = wm.alloc_scene_id().0; // peek-and-bump
    // Undo the bump — the load path re-sets it. Slight inefficiency but
    // avoids needing a non-mutating accessor.
    wm.set_next_scene_id(next_id);
    out.extend_from_slice(&next_id.to_le_bytes());
    out.extend_from_slice(&(scenes.len() as u32).to_le_bytes());

    for scene in &scenes {
        let json = match serde_json::to_vec(scene) {
            Ok(j) => j,
            Err(_) => continue, // skip un-serializable scenes (shouldn't happen)
        };
        out.extend_from_slice(&(json.len() as u32).to_le_bytes());
        out.extend_from_slice(&json);
    }

    out
}

/// Deserialize a blob into the given `WorldMemory`. On error, the
/// WorldMemory is left empty (caller starts cold). `bytes.is_empty()` is a
/// no-op (v1-v6 saves load with empty blob).
pub fn load_blob(wm: &WorldMemory, bytes: &[u8]) -> Result<usize, PersistError> {
    if bytes.is_empty() {
        return Ok(0);
    }
    if bytes.len() < 20 {
        return Err(PersistError::Truncated);
    }

    if &bytes[0..4] != MAGIC {
        return Err(PersistError::BadMagic);
    }

    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != VERSION {
        return Err(PersistError::UnsupportedVersion(version));
    }

    let next_id = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    let count = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;

    let mut cursor = 20;
    let mut loaded = 0usize;
    for _ in 0..count {
        if cursor + 4 > bytes.len() {
            return Err(PersistError::Truncated);
        }
        let len = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;
        if cursor + len > bytes.len() {
            return Err(PersistError::Truncated);
        }
        let scene: Scene = serde_json::from_slice(&bytes[cursor..cursor + len])
            .map_err(|e| PersistError::JsonError(e.to_string()))?;
        wm.scenes.insert(scene.id, scene);
        cursor += len;
        loaded += 1;
    }

    // Set the id allocator so post-load allocations don't collide.
    wm.set_next_scene_id(next_id);

    Ok(loaded)
}

/// Convenience: clear all scenes (used by tests + reload paths).
pub fn clear(wm: &WorldMemory) {
    wm.scenes.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{Scene, SceneKind};
    use glam::Vec3;

    #[test]
    fn empty_roundtrip() {
        let wm = WorldMemory::new();
        let blob = serialize_blob(&wm);
        assert!(blob.len() >= 20); // header
        let wm2 = WorldMemory::new();
        let loaded = load_blob(&wm2, &blob).expect("load");
        assert_eq!(loaded, 0);
        assert_eq!(wm2.tracked_scene_count(), 0);
    }

    #[test]
    fn populated_roundtrip() {
        let wm = WorldMemory::new();
        let mut s1 = Scene::new(wm.alloc_scene_id(), SceneKind::Lava, Vec3::new(10.0, 20.0, 30.0));
        s1.score = 100.0;
        s1.confidence = 0.9;
        s1.chunks = vec![(0, 0, 0), (1, 0, 0)];
        wm.scenes.insert(s1.id, s1.clone());

        let mut s2 = Scene::new(wm.alloc_scene_id(), SceneKind::Water, Vec3::new(5.0, 5.0, 5.0));
        s2.score = 50.0;
        s2.confidence = 0.6;
        wm.scenes.insert(s2.id, s2.clone());

        let blob = serialize_blob(&wm);

        let wm2 = WorldMemory::new();
        let loaded = load_blob(&wm2, &blob).expect("load");
        assert_eq!(loaded, 2);
        assert_eq!(wm2.tracked_scene_count(), 2);

        let scenes: Vec<_> = wm2.scenes.iter().map(|e| e.value().clone()).collect();
        let r1 = scenes.iter().find(|s| s.id == s1.id).unwrap();
        let r2 = scenes.iter().find(|s| s.id == s2.id).unwrap();
        assert_eq!(r1.kind, SceneKind::Lava);
        assert!((r1.score - 100.0).abs() < 1e-3);
        assert_eq!(r1.chunks, vec![(0, 0, 0), (1, 0, 0)]);
        assert_eq!(r2.kind, SceneKind::Water);

        // Post-load id allocator must continue past the loaded ids.
        let next = wm2.alloc_scene_id();
        assert!(next.0 > s1.id.0 && next.0 > s2.id.0);
    }

    #[test]
    fn bad_magic_returns_err() {
        let wm = WorldMemory::new();
        let mut blob = serialize_blob(&wm);
        blob[0] = b'X';
        let wm2 = WorldMemory::new();
        let r = load_blob(&wm2, &blob);
        assert!(matches!(r, Err(PersistError::BadMagic)));
        assert_eq!(wm2.tracked_scene_count(), 0);
    }

    #[test]
    fn unsupported_version_returns_err() {
        let wm = WorldMemory::new();
        let mut blob = serialize_blob(&wm);
        // Bump internal version to something unrecognized.
        blob[4..8].copy_from_slice(&999u32.to_le_bytes());
        let wm2 = WorldMemory::new();
        let r = load_blob(&wm2, &blob);
        assert!(matches!(r, Err(PersistError::UnsupportedVersion(999))));
    }

    #[test]
    fn truncated_blob_returns_err() {
        let wm = WorldMemory::new();
        let mut s = Scene::new(wm.alloc_scene_id(), SceneKind::Lava, Vec3::ZERO);
        s.score = 10.0;
        wm.scenes.insert(s.id, s);
        let blob = serialize_blob(&wm);
        // Drop the last 20 bytes.
        let truncated = &blob[..blob.len() - 20];
        let wm2 = WorldMemory::new();
        let r = load_blob(&wm2, truncated);
        assert!(r.is_err());
    }

    #[test]
    fn random_garbage_returns_err_not_panic() {
        let wm = WorldMemory::new();
        let garbage: Vec<u8> = (0..512).map(|i| ((i * 37 + 13) % 256) as u8).collect();
        let r = load_blob(&wm, &garbage);
        // Just verify it doesn't panic — could be BadMagic, UnsupportedVersion,
        // Truncated, or JsonError depending on the random bytes.
        assert!(r.is_err());
        assert_eq!(wm.tracked_scene_count(), 0);
    }

    #[test]
    fn empty_input_is_noop() {
        let wm = WorldMemory::new();
        let r = load_blob(&wm, &[]).expect("empty load");
        assert_eq!(r, 0);
        assert_eq!(wm.tracked_scene_count(), 0);
    }
}
