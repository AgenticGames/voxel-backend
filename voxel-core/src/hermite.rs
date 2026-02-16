use glam::Vec3;
use crate::material::Material;

/// Packs (x, y, z, axis) into a u32. Each coordinate gets 7 bits (0-127),
/// axis gets 2 bits = 23 bits total. Supports chunk_size up to 64.
///
/// Layout: [x:7][y:7][z:7][axis:2] = bits 22..16, 15..9, 8..2, 1..0
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeKey(pub u32);

impl EdgeKey {
    pub fn new(x: u8, y: u8, z: u8, axis: u8) -> Self {
        debug_assert!(x < 128 && y < 128 && z < 128 && axis < 3);
        Self(((x as u32) << 16) | ((y as u32) << 9) | ((z as u32) << 2) | (axis as u32))
    }

    pub fn x(self) -> u8 { ((self.0 >> 16) & 0x7F) as u8 }
    pub fn y(self) -> u8 { ((self.0 >> 9) & 0x7F) as u8 }
    pub fn z(self) -> u8 { ((self.0 >> 2) & 0x7F) as u8 }
    pub fn axis(self) -> u8 { (self.0 & 0x3) as u8 }
}

#[derive(Debug, Clone)]
pub struct EdgeIntersection {
    pub t: f32,
    pub normal: Vec3,
    pub material: Material,
}

/// Identity hasher for u32/usize keys — avoids SipHash overhead.
/// Safe because EdgeKey packing already distributes bits well.
#[derive(Default)]
pub struct IdentityHasher(u64);

impl std::hash::Hasher for IdentityHasher {
    #[inline]
    fn finish(&self) -> u64 { self.0 }
    #[inline]
    fn write(&mut self, _bytes: &[u8]) {}
    #[inline]
    fn write_u32(&mut self, i: u32) { self.0 = i as u64; }
    #[inline]
    fn write_usize(&mut self, i: usize) { self.0 = i as u64; }
}

pub type BuildIdentityHasher = std::hash::BuildHasherDefault<IdentityHasher>;

/// Re-export for use in solve.rs and mesh_gen.rs
pub type FastHashMap<K, V> = std::collections::HashMap<K, V, BuildIdentityHasher>;

/// Fast map from EdgeKey -> EdgeIntersection using identity hasher.
#[derive(Debug, Clone)]
pub struct EdgeMap {
    map: FastHashMap<u32, EdgeIntersection>,
}

impl Default for EdgeMap {
    fn default() -> Self {
        Self::new()
    }
}

impl EdgeMap {
    pub fn new() -> Self {
        Self {
            map: FastHashMap::default(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            map: FastHashMap::with_capacity_and_hasher(cap, BuildIdentityHasher::default()),
        }
    }

    pub fn insert(&mut self, key: EdgeKey, value: EdgeIntersection) {
        self.map.insert(key.0, value);
    }

    pub fn get(&self, key: &EdgeKey) -> Option<&EdgeIntersection> {
        self.map.get(&key.0)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Remove all edges whose coordinates fall within the given bounding box.
    /// Used for incremental hermite updates during mining.
    pub fn remove_in_range(
        &mut self,
        min_x: usize, max_x: usize,
        min_y: usize, max_y: usize,
        min_z: usize, max_z: usize,
    ) {
        self.map.retain(|&raw_key, _| {
            let key = EdgeKey(raw_key);
            let x = key.x() as usize;
            let y = key.y() as usize;
            let z = key.z() as usize;
            !(x >= min_x && x <= max_x && y >= min_y && y <= max_y && z >= min_z && z <= max_z)
        });
    }

    /// Iterate over all (EdgeKey, &EdgeIntersection) pairs.
    pub fn iter(&self) -> EdgeMapIter<'_> {
        EdgeMapIter {
            inner: self.map.iter(),
        }
    }
}

pub struct EdgeMapIter<'a> {
    inner: std::collections::hash_map::Iter<'a, u32, EdgeIntersection>,
}

impl<'a> Iterator for EdgeMapIter<'a> {
    type Item = (EdgeKey, &'a EdgeIntersection);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(&k, v)| (EdgeKey(k), v))
    }
}

impl<'a> IntoIterator for &'a EdgeMap {
    type Item = (EdgeKey, &'a EdgeIntersection);
    type IntoIter = EdgeMapIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Sparse hermite data using fast identity-hashed map.
#[derive(Debug, Clone, Default)]
pub struct HermiteData {
    pub edges: EdgeMap,
}

impl HermiteData {
    /// Create with pre-allocated capacity for expected edge count.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            edges: EdgeMap::with_capacity(cap),
        }
    }

    /// Number of sign-changing edges stored.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Iterate over (EdgeKey, &EdgeIntersection) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (EdgeKey, &EdgeIntersection)> {
        self.edges.iter()
    }
}
