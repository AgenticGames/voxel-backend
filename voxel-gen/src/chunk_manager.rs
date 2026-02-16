use std::collections::HashMap;
use rayon::prelude::*;
use voxel_core::chunk::{Chunk, ChunkCoord};
use crate::config::GenerationConfig;

/// Manages chunk loading/unloading with parallel generation via rayon
pub struct ChunkManager {
    pub chunks: HashMap<ChunkCoord, Chunk>,
    pub config: GenerationConfig,
}

impl ChunkManager {
    pub fn new(config: GenerationConfig) -> Self {
        Self {
            chunks: HashMap::new(),
            config,
        }
    }

    pub fn load_chunk(&mut self, coord: ChunkCoord) {
        if !self.chunks.contains_key(&coord) {
            let chunk = crate::generate_chunk(coord, &self.config);
            self.chunks.insert(coord, chunk);
        }
    }

    pub fn unload_chunk(&mut self, coord: &ChunkCoord) {
        self.chunks.remove(coord);
    }

    /// Load multiple chunks in parallel using rayon.
    pub fn load_chunks_parallel(&mut self, coords: &[ChunkCoord]) {
        let missing: Vec<ChunkCoord> = coords
            .iter()
            .filter(|c| !self.chunks.contains_key(c))
            .copied()
            .collect();

        let config = &self.config;
        let new_chunks: Vec<(ChunkCoord, Chunk)> = missing
            .par_iter()
            .map(|&coord| {
                let chunk = crate::generate_chunk(coord, config);
                (coord, chunk)
            })
            .collect();

        for (coord, chunk) in new_chunks {
            self.chunks.insert(coord, chunk);
        }
    }

    /// Get a reference to a loaded chunk
    pub fn get_chunk(&self, coord: &ChunkCoord) -> Option<&Chunk> {
        self.chunks.get(coord)
    }

    /// Number of loaded chunks
    pub fn loaded_count(&self) -> usize {
        self.chunks.len()
    }

    /// Unload all chunks
    pub fn clear(&mut self) {
        self.chunks.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::chunk::ChunkCoord;

    #[test]
    fn test_chunk_manager_basic() {
        let config = GenerationConfig::default();
        let mut manager = ChunkManager::new(config);
        assert_eq!(manager.loaded_count(), 0);

        manager.load_chunk(ChunkCoord::new(0, 0, 0));
        assert_eq!(manager.loaded_count(), 1);
        assert!(manager.get_chunk(&ChunkCoord::new(0, 0, 0)).is_some());
    }

    #[test]
    fn test_chunk_manager_no_duplicate_load() {
        let config = GenerationConfig::default();
        let mut manager = ChunkManager::new(config);

        manager.load_chunk(ChunkCoord::new(0, 0, 0));
        manager.load_chunk(ChunkCoord::new(0, 0, 0));
        assert_eq!(manager.loaded_count(), 1);
    }

    #[test]
    fn test_chunk_manager_unload() {
        let config = GenerationConfig::default();
        let mut manager = ChunkManager::new(config);

        let coord = ChunkCoord::new(0, 0, 0);
        manager.load_chunk(coord);
        assert_eq!(manager.loaded_count(), 1);

        manager.unload_chunk(&coord);
        assert_eq!(manager.loaded_count(), 0);
        assert!(manager.get_chunk(&coord).is_none());
    }

    #[test]
    fn test_chunk_manager_parallel() {
        let config = GenerationConfig::default();
        let mut manager = ChunkManager::new(config);

        let coords: Vec<ChunkCoord> = (0..4)
            .map(|i| ChunkCoord::new(i, 0, 0))
            .collect();

        manager.load_chunks_parallel(&coords);
        assert_eq!(manager.loaded_count(), 4);

        for coord in &coords {
            assert!(manager.get_chunk(coord).is_some());
        }
    }

    #[test]
    fn test_chunk_manager_clear() {
        let config = GenerationConfig::default();
        let mut manager = ChunkManager::new(config);

        manager.load_chunk(ChunkCoord::new(0, 0, 0));
        manager.load_chunk(ChunkCoord::new(1, 0, 0));
        assert_eq!(manager.loaded_count(), 2);

        manager.clear();
        assert_eq!(manager.loaded_count(), 0);
    }
}
