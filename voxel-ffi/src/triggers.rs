//! Editor-authored collapse triggers.
//!
//! Designers paint regions in creative mode that fire scripted collapse
//! cinematics when their conditions are met:
//!   * `OnFirstMine` — first pickaxe swing inside the volume triggers it;
//!     used for "tutorial gate" tunnels that reveal the mechanic.
//!   * `OnPillarLoss` — one to many tracked pillar volumes; fires when
//!     enough of them drop below a load-bearing threshold (used for boss
//!     arena dome).
//!
//! When a trigger fires, the worker synthesizes a `CollapseEventV2` from
//! the trigger's pre-painted `target_slab_voxels` + `pile_chunks` and
//! injects it into the existing cinematic pipeline — same warning →
//! falling slab → debris pile → real remesh choreography as natural
//! stress-driven collapses.

use std::collections::HashMap;

use voxel_core::material::Material;
use voxel_gen::density::DensityField;

/// Axis-aligned bounding box in **world voxel coordinates** (inclusive on
/// both ends). Use `effective_bounds()` to convert between world voxels
/// and chunk coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoxelAabb {
    pub min: (i32, i32, i32),
    pub max: (i32, i32, i32),
}

impl VoxelAabb {
    pub fn intersects(&self, other: &VoxelAabb) -> bool {
        self.min.0 <= other.max.0 && self.max.0 >= other.min.0
            && self.min.1 <= other.max.1 && self.max.1 >= other.min.1
            && self.min.2 <= other.max.2 && self.max.2 >= other.min.2
    }

    pub fn contains_voxel(&self, v: (i32, i32, i32)) -> bool {
        v.0 >= self.min.0 && v.0 <= self.max.0
            && v.1 >= self.min.1 && v.1 <= self.max.1
            && v.2 >= self.min.2 && v.2 <= self.max.2
    }

    pub fn volume_voxels(&self) -> u32 {
        let dx = (self.max.0 - self.min.0 + 1).max(0) as u32;
        let dy = (self.max.1 - self.min.1 + 1).max(0) as u32;
        let dz = (self.max.2 - self.min.2 + 1).max(0) as u32;
        dx.saturating_mul(dy).saturating_mul(dz)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossCondition {
    /// Fire as soon as a single pillar drops below threshold.
    AnyPillar,
    /// Fire once at least N pillars are below threshold.
    NPillars(u8),
    /// Fire only when every pillar is below threshold.
    AllPillars,
}

#[derive(Debug, Clone)]
pub struct PillarRef {
    pub volume: VoxelAabb,
    /// Solid voxel count captured the moment the trigger was armed.
    /// Re-eval compares the current count to this baseline.
    pub baseline_solid: u32,
}

#[derive(Debug, Clone)]
pub enum TriggerActivation {
    /// Any mining event whose AABB intersects `trigger_volume` fires it.
    OnFirstMine {
        trigger_volume: VoxelAabb,
    },
    /// Re-evaluates after every mining event that touches at least one
    /// pillar. Pillar i is "lost" when its current solid voxel count is
    /// below `loss_threshold * baseline_solid`. The trigger fires when
    /// the lost count satisfies `condition`.
    OnPillarLoss {
        pillars: Vec<PillarRef>,
        condition: LossCondition,
        loss_threshold: f32,
    },
}

#[derive(Debug, Clone)]
pub struct EditorCollapseTrigger {
    pub id: u32,
    pub name: String,
    /// One-shot guard. Set to false the moment the trigger fires; the
    /// editor "rearm" button is the only thing that flips it back to
    /// true mid-session.
    pub armed: bool,
    pub activation: TriggerActivation,
    /// Author-painted voxels that fall during the cinematic. World voxel
    /// coords. The worker reads each voxel's current material from the
    /// density field at fire time to build the slab mesh.
    pub target_slab_voxels: Vec<(i32, i32, i32)>,
    /// Chunks where debris settles after the cinematic finishes. Usually
    /// the chunks beneath the slab; can be the same chunks if the slab
    /// itself lands in place.
    pub pile_chunks: Vec<(i32, i32, i32)>,
    /// Author override for fall distance in UE units (0 = auto-compute
    /// from slab/pile geometry).
    pub fall_distance_uu: f32,
}

impl EditorCollapseTrigger {
    /// Check whether this trigger should fire given a recent mining event.
    pub fn should_fire(
        &self,
        mined_volume: &VoxelAabb,
        density_fields: &HashMap<(i32, i32, i32), DensityField>,
        chunk_size: usize,
    ) -> bool {
        if !self.armed {
            return false;
        }
        match &self.activation {
            TriggerActivation::OnFirstMine { trigger_volume } => {
                mined_volume.intersects(trigger_volume)
            }
            TriggerActivation::OnPillarLoss {
                pillars,
                condition,
                loss_threshold,
            } => {
                // Skip the recount unless the mining at least *touched*
                // a pillar — boss arena pillars are tiny relative to the
                // whole world so this short-circuit matters.
                if !pillars.iter().any(|p| mined_volume.intersects(&p.volume)) {
                    return false;
                }
                let lost: u8 = pillars
                    .iter()
                    .filter(|p| {
                        if p.baseline_solid == 0 {
                            return false;
                        }
                        let current = count_solid_voxels(&p.volume, density_fields, chunk_size);
                        (current as f32 / p.baseline_solid as f32) < *loss_threshold
                    })
                    .count() as u8;
                match condition {
                    LossCondition::AnyPillar => lost >= 1,
                    LossCondition::NPillars(n) => lost >= *n,
                    LossCondition::AllPillars => lost == pillars.len() as u8,
                }
            }
        }
    }

    /// Compute the dominant material of the painted slab against current
    /// density. Returns `Material::Slate` if no solid voxels found.
    pub fn dominant_slab_material(
        &self,
        density_fields: &HashMap<(i32, i32, i32), DensityField>,
        chunk_size: usize,
    ) -> Material {
        let mut counts: [u32; 256] = [0; 256];
        let mut total: u32 = 0;
        for &(wx, wy, wz) in &self.target_slab_voxels {
            if let Some(s) = sample_voxel(wx, wy, wz, density_fields, chunk_size) {
                if s.density > 0.0 {
                    counts[s.material as u8 as usize] += 1;
                    total += 1;
                }
            }
        }
        if total == 0 {
            return Material::Slate;
        }
        let (best, _) = counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, c)| (i, *c))
            .unwrap_or((Material::Slate as u8 as usize, 0));
        Material::from_u8(best as u8)
    }
}

/// Count solid CELLS inside `aabb` using the cell-center density test
/// (avg of 8 corners > 0). Voxels outside loaded chunks are skipped — for
/// pillar baselines this is a non-issue because pillars are always inside
/// the playable region.
pub fn count_solid_voxels(
    aabb: &VoxelAabb,
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
) -> u32 {
    let mut count: u32 = 0;
    for wz in aabb.min.2..=aabb.max.2 {
        for wy in aabb.min.1..=aabb.max.1 {
            for wx in aabb.min.0..=aabb.max.0 {
                if cell_has_solid_center(wx, wy, wz, density_fields, chunk_size) {
                    count += 1;
                }
            }
        }
    }
    count
}

/// Sample one voxel from the loaded density fields. Returns `None` if the
/// containing chunk isn't loaded.
fn sample_voxel(
    wx: i32,
    wy: i32,
    wz: i32,
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
) -> Option<voxel_core::octree::node::VoxelSample> {
    let cs = chunk_size as i32;
    let cx = wx.div_euclid(cs);
    let cy = wy.div_euclid(cs);
    let cz = wz.div_euclid(cs);
    let lx = wx.rem_euclid(cs) as usize;
    let ly = wy.rem_euclid(cs) as usize;
    let lz = wz.rem_euclid(cs) as usize;
    density_fields.get(&(cx, cy, cz)).map(|df| *df.get(lx, ly, lz))
}

/// Cell-center density test: average of the cell's 8 corner samples > 0.
/// This is the same definition DC uses to decide whether a cell has
/// interior rock. A single-corner test misses cells at the rock/air
/// boundary (cave ceilings, walls) because one corner often sits in the
/// adjacent air.
///
/// Pair with the matching `engine::cell_has_solid_center` helper that
/// works against a `ChunkStore` directly — same algorithm, different
/// data-source signature.
pub fn cell_has_solid_center(
    wx: i32,
    wy: i32,
    wz: i32,
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
) -> bool {
    let cs = chunk_size as i32;
    let mut sum: f32 = 0.0;
    let mut samples: u32 = 0;
    for c in 0..8 {
        let ox = (c & 1) as i32;
        let oy = ((c >> 1) & 1) as i32;
        let oz = ((c >> 2) & 1) as i32;
        let sx = wx + ox;
        let sy = wy + oy;
        let sz = wz + oz;
        let cx = sx.div_euclid(cs);
        let cy = sy.div_euclid(cs);
        let cz = sz.div_euclid(cs);
        let lx = sx.rem_euclid(cs) as usize;
        let ly = sy.rem_euclid(cs) as usize;
        let lz = sz.rem_euclid(cs) as usize;
        if let Some(df) = density_fields.get(&(cx, cy, cz)) {
            if lx < df.size && ly < df.size && lz < df.size {
                sum += df.get(lx, ly, lz).density;
                samples += 1;
            }
        }
    }
    if samples == 0 {
        return false;
    }
    (sum / samples as f32) > 0.0
}

/// Capture the baseline solid voxel count for every pillar in a trigger.
/// Called when a trigger is armed.
pub fn refresh_pillar_baselines(
    trigger: &mut EditorCollapseTrigger,
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
) {
    if let TriggerActivation::OnPillarLoss { pillars, .. } = &mut trigger.activation {
        for pillar in pillars.iter_mut() {
            pillar.baseline_solid = count_solid_voxels(&pillar.volume, density_fields, chunk_size);
        }
    }
}

/// Synthesize a `CollapseEventV2` from a trigger's painted voxel set. The
/// caller is responsible for actually CLEARING the slab voxels from
/// density (see `clear_slab_voxels`) — the cinematic pipeline expects
/// "defer_pile=true" semantics where slab cells are already air and only
/// the pile placement is deferred.
///
/// Returns `None` if no solid voxels remain in the painted slab (already
/// mined out, never solid, or chunk not loaded — all valid edge cases
/// where firing a collapse would be meaningless).
pub fn synthesize_collapse_event(
    trigger: &EditorCollapseTrigger,
    density_fields: &HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
) -> Option<voxel_core::stress::CollapseEventV2> {
    use voxel_core::stress::{
        CollapseEventV2, CollapseSlab, CollapsedVoxel, PendingPilePlacement,
    };
    let cs = chunk_size as i32;

    let mut slab_voxels: Vec<CollapsedVoxel> = Vec::with_capacity(trigger.target_slab_voxels.len());
    let mut bb_min = (i32::MAX, i32::MAX, i32::MAX);
    let mut bb_max = (i32::MIN, i32::MIN, i32::MIN);
    let mut mat_counts: [u32; 256] = [0; 256];

    for &(wx, wy, wz) in &trigger.target_slab_voxels {
        // Cell-center density test (avg of 8 corners). Matches the test
        // used by the paint-time filter so paint markers and synthesized
        // slab voxels stay 1:1. Single-corner test would drop every
        // cave-ceiling cell because the bottom corner sits in cave air
        // while the upper corners are rock.
        if !cell_has_solid_center(wx, wy, wz, density_fields, chunk_size) {
            continue;
        }
        // Read material from a corner that's positive — prefer the corner
        // whose density is closest to the average so the chosen material
        // matches what the cinematic visualizes for this cell. In practice
        // any corner with density > 0 carries the local host-rock
        // material; sampling the (lx,ly,lz) corner is fine when the cell
        // is interior-solid, and for boundary cells we walk to find the
        // first solid corner.
        let cx = wx.div_euclid(cs);
        let cy = wy.div_euclid(cs);
        let cz = wz.div_euclid(cs);
        let lx = wx.rem_euclid(cs) as usize;
        let ly = wy.rem_euclid(cs) as usize;
        let lz = wz.rem_euclid(cs) as usize;
        let mut chosen_material: Option<voxel_core::material::Material> = None;
        for c in 0..8 {
            let ox = (c & 1) as i32;
            let oy = ((c >> 1) & 1) as i32;
            let oz = ((c >> 2) & 1) as i32;
            let sx = wx + ox;
            let sy = wy + oy;
            let sz = wz + oz;
            let scx = sx.div_euclid(cs);
            let scy = sy.div_euclid(cs);
            let scz = sz.div_euclid(cs);
            let slx = sx.rem_euclid(cs) as usize;
            let sly = sy.rem_euclid(cs) as usize;
            let slz = sz.rem_euclid(cs) as usize;
            if let Some(df) = density_fields.get(&(scx, scy, scz)) {
                if slx < df.size && sly < df.size && slz < df.size {
                    let s = df.get(slx, sly, slz);
                    if s.density > 0.0 {
                        chosen_material = Some(s.material);
                        break;
                    }
                }
            }
        }
        let material = chosen_material.unwrap_or_else(|| {
            // Fallback: cell center says solid but somehow no corner had
            // density>0 (numerically rare). Use the base-corner sample's
            // material so we at least emit something sensible.
            density_fields
                .get(&(cx, cy, cz))
                .map(|df| df.get(lx, ly, lz).material)
                .unwrap_or(voxel_core::material::Material::Slate)
        });
        slab_voxels.push(CollapsedVoxel {
            world_x: wx,
            world_y: wy,
            world_z: wz,
            material,
        });
        bb_min.0 = bb_min.0.min(wx);
        bb_min.1 = bb_min.1.min(wy);
        bb_min.2 = bb_min.2.min(wz);
        bb_max.0 = bb_max.0.max(wx);
        bb_max.1 = bb_max.1.max(wy);
        bb_max.2 = bb_max.2.max(wz);
        mat_counts[material as u8 as usize] += 1;
    }

    if slab_voxels.is_empty() {
        return None;
    }

    let center = (
        (bb_min.0 as f32 + bb_max.0 as f32) * 0.5,
        (bb_min.1 as f32 + bb_max.1 as f32) * 0.5,
        (bb_min.2 as f32 + bb_max.2 as f32) * 0.5,
    );
    let dominant_material = {
        let idx = mat_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(voxel_core::material::Material::Slate as u8 as usize);
        voxel_core::material::Material::from_u8(idx as u8)
    };

    // Landing Y: prefer the floor of the lowest pile chunk; fall back to
    // bb_min.1 - 1 if no pile_chunks were painted (slab "drops one voxel
    // in place" — visible but not dramatic).
    let landing_y = if !trigger.pile_chunks.is_empty() {
        trigger
            .pile_chunks
            .iter()
            .map(|c| c.1 * cs)
            .min()
            .unwrap_or(bb_min.1 - 1)
    } else {
        bb_min.1 - 1
    };
    let fall_distance = (bb_min.1 - landing_y).max(0);

    // Slab chunks: every chunk a slab voxel is in.
    let mut slab_chunks: Vec<(i32, i32, i32)> = slab_voxels
        .iter()
        .map(|v| {
            (
                v.world_x.div_euclid(cs),
                v.world_y.div_euclid(cs),
                v.world_z.div_euclid(cs),
            )
        })
        .collect();
    slab_chunks.sort();
    slab_chunks.dedup();

    let mut pile_chunks: Vec<(i32, i32, i32)> = trigger.pile_chunks.clone();
    pile_chunks.sort();
    pile_chunks.dedup();

    let mut affected_chunks: Vec<(i32, i32, i32)> =
        slab_chunks.iter().chain(pile_chunks.iter()).copied().collect();
    affected_chunks.sort();
    affected_chunks.dedup();

    let pending_pile = PendingPilePlacement {
        collapsed_voxels: slab_voxels.clone(),
        bb_min,
        bb_max,
        dominant_material,
        landing_offset: fall_distance,
    };
    let slab = CollapseSlab {
        voxels: slab_voxels.clone(),
        bb_min,
        bb_max,
        center,
        landing_y,
        fall_distance,
        dominant_material,
    };
    let total_volume = slab_voxels.len() as u32;

    Some(CollapseEventV2 {
        slabs: vec![slab],
        affected_chunks,
        slab_chunks,
        pile_chunks,
        pending_piles: vec![pending_pile],
        total_volume,
        center,
    })
}

/// Clear the slab voxels from density fields (sets density to negative,
/// material to Air). The natural collapse pipeline does this implicitly in
/// `detect_and_execute_collapses_v2_with_options`; for scripted collapses
/// we have to do it ourselves so the cave roof actually opens when the
/// slab visually detaches.
pub fn clear_slab_voxels(
    voxels: &[(i32, i32, i32)],
    density_fields: &mut HashMap<(i32, i32, i32), DensityField>,
    chunk_size: usize,
) -> Vec<(i32, i32, i32)> {
    let cs = chunk_size as i32;
    let mut dirty: std::collections::HashSet<(i32, i32, i32)> =
        std::collections::HashSet::new();
    for &(wx, wy, wz) in voxels {
        let cx = wx.div_euclid(cs);
        let cy = wy.div_euclid(cs);
        let cz = wz.div_euclid(cs);
        let lx = wx.rem_euclid(cs) as usize;
        let ly = wy.rem_euclid(cs) as usize;
        let lz = wz.rem_euclid(cs) as usize;
        if let Some(df) = density_fields.get_mut(&(cx, cy, cz)) {
            let sample = df.get_mut(lx, ly, lz);
            sample.density = -1.0;
            sample.material = voxel_core::material::Material::Air;
            dirty.insert((cx, cy, cz));
        }
    }
    // Caller should call `compute_metadata` on each dirty chunk before
    // re-meshing.
    for &k in &dirty {
        if let Some(df) = density_fields.get_mut(&k) {
            df.compute_metadata();
        }
    }
    dirty.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aabb_intersects() {
        let a = VoxelAabb { min: (0, 0, 0), max: (10, 10, 10) };
        let b = VoxelAabb { min: (5, 5, 5), max: (15, 15, 15) };
        let c = VoxelAabb { min: (20, 20, 20), max: (30, 30, 30) };
        assert!(a.intersects(&b));
        assert!(b.intersects(&a));
        assert!(!a.intersects(&c));
        assert!(a.contains_voxel((5, 5, 5)));
        assert!(!a.contains_voxel((11, 5, 5)));
    }

    #[test]
    fn aabb_volume() {
        let a = VoxelAabb { min: (0, 0, 0), max: (0, 0, 0) };
        assert_eq!(a.volume_voxels(), 1);
        let b = VoxelAabb { min: (0, 0, 0), max: (9, 9, 9) };
        assert_eq!(b.volume_voxels(), 1000);
    }

    #[test]
    fn on_first_mine_fires_on_intersect_only() {
        let trig = EditorCollapseTrigger {
            id: 1,
            name: "test".into(),
            armed: true,
            activation: TriggerActivation::OnFirstMine {
                trigger_volume: VoxelAabb { min: (0, 0, 0), max: (10, 10, 10) },
            },
            target_slab_voxels: vec![],
            pile_chunks: vec![],
            fall_distance_uu: 0.0,
        };
        let densities = HashMap::new();
        let hit = VoxelAabb { min: (5, 5, 5), max: (7, 7, 7) };
        let miss = VoxelAabb { min: (20, 20, 20), max: (25, 25, 25) };
        assert!(trig.should_fire(&hit, &densities, 30));
        assert!(!trig.should_fire(&miss, &densities, 30));
    }

    #[test]
    fn disarmed_trigger_never_fires() {
        let mut trig = EditorCollapseTrigger {
            id: 1,
            name: "test".into(),
            armed: false,
            activation: TriggerActivation::OnFirstMine {
                trigger_volume: VoxelAabb { min: (0, 0, 0), max: (10, 10, 10) },
            },
            target_slab_voxels: vec![],
            pile_chunks: vec![],
            fall_distance_uu: 0.0,
        };
        let densities = HashMap::new();
        let hit = VoxelAabb { min: (5, 5, 5), max: (7, 7, 7) };
        assert!(!trig.should_fire(&hit, &densities, 30));
        trig.armed = true;
        assert!(trig.should_fire(&hit, &densities, 30));
    }
}
