use std::collections::HashSet;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};

use crossbeam_channel::{bounded, Receiver, Sender};
use dashmap::DashMap;
use voxel_fluid::FluidConfig;
use voxel_fluid::FluidEvent;
use voxel_core::stress::StressField;
use voxel_core::world_scan::ScanConfig;
use voxel_gen::config::{
    BandedIronConfig, CrystalConfig, FormationConfig, GenerationConfig, GeodeConfig, HostRockConfig,
    KimberlitePipeConfig, MineConfig, NoiseConfig, OreConfig, OreCrystalConfig, OreVeinParams,
    PoolConfig, StressConfig, SulfideBlobConfig, WormConfig,
};

use crate::convert::ue_chunk_to_rust;
use crate::pathing::{
    build_request_from_ue, FfiPathNode, FfiPathRequest, FfiPathResult, PathResultStore,
    StashedPathResult,
};
use crate::profiler::StreamingProfiler;
use crate::store::ChunkStore;
use crate::types::*;
use crate::worker::{path_worker_loop, worker_loop};

use super::{aabb_center, aabb_to_ffi, zero_aabb, VoxelEngine};

impl VoxelEngine {
    pub fn create_trigger(
        &self,
        activation_kind: u8,
        name: &str,
        volumes: &[crate::triggers::VoxelAabb],
        loss_condition: u8,
        loss_n: u8,
        loss_threshold: f32,
        target_slab: &[(i32, i32, i32)],
        pile_chunks: &[(i32, i32, i32)],
        fall_distance_uu: f32,
    ) -> u32 {
        use crate::triggers::{
            EditorCollapseTrigger, LossCondition, PillarRef, TriggerActivation,
        };

        if target_slab.is_empty() || volumes.is_empty() {
            eprintln!("[voxel-ffi] create_trigger rejected: needs >=1 volume and >=1 slab voxel");
            return 0;
        }

        let activation = match activation_kind {
            0 => TriggerActivation::OnFirstMine {
                trigger_volume: volumes[0],
            },
            1 => {
                let pillars: Vec<PillarRef> = volumes
                    .iter()
                    .map(|&v| PillarRef {
                        volume: v,
                        baseline_solid: 0, // captured after insert
                    })
                    .collect();
                let condition = match loss_condition {
                    0 => LossCondition::AnyPillar,
                    1 => LossCondition::NPillars(loss_n.max(1)),
                    _ => LossCondition::AllPillars,
                };
                TriggerActivation::OnPillarLoss {
                    pillars,
                    condition,
                    loss_threshold: loss_threshold.clamp(0.01, 1.0),
                }
            }
            _ => {
                eprintln!(
                    "[voxel-ffi] create_trigger rejected: unknown activation_kind {}",
                    activation_kind
                );
                return 0;
            }
        };

        let mut store = self.store.write().unwrap();
        let id = store.alloc_trigger_id();
        let mut trig = EditorCollapseTrigger {
            id,
            name: name.to_string(),
            armed: true,
            activation,
            target_slab_voxels: target_slab.to_vec(),
            pile_chunks: pile_chunks.to_vec(),
            fall_distance_uu,
        };
        crate::triggers::refresh_pillar_baselines(
            &mut trig,
            &store.density_fields,
            self.cached_chunk_size(),
        );
        store.triggers.push(trig);
        eprintln!(
            "[voxel-ffi] created trigger {} '{}' kind={} slab_voxels={} pile_chunks={}",
            id,
            name,
            activation_kind,
            target_slab.len(),
            pile_chunks.len()
        );
        id
    }

    /// List all current trigger ids (any state). Order is creation order.
    pub fn list_trigger_ids(&self) -> Vec<u32> {
        let store = self.store.read().unwrap();
        store.triggers.iter().map(|t| t.id).collect()
    }

    /// Fetch summary info for a single trigger. Returns `None` if no
    /// trigger has that id.
    pub fn get_trigger_info(&self, id: u32) -> Option<crate::types::FfiTriggerInfo> {
        use crate::triggers::TriggerActivation;
        let store = self.store.read().unwrap();
        let trig = store.find_trigger(id)?;

        let mut info = crate::types::FfiTriggerInfo {
            id: trig.id,
            armed: if trig.armed { 1 } else { 0 },
            activation_kind: 0,
            volume_count: 0,
            loss_condition: 0,
            loss_n: 0,
            _padding: [0; 3],
            loss_threshold: 0.0,
            fall_distance_uu: trig.fall_distance_uu,
            slab_voxel_count: trig.target_slab_voxels.len() as u32,
            pile_chunk_count: trig.pile_chunks.len() as u32,
            primary_volume: zero_aabb(),
            pillar_volumes: [zero_aabb(); 8],
            name: [0u8; 64],
        };

        // Pack name (UTF-8, NUL-terminated, truncate to 63).
        let bytes = trig.name.as_bytes();
        let take = bytes.len().min(63);
        info.name[..take].copy_from_slice(&bytes[..take]);

        match &trig.activation {
            TriggerActivation::OnFirstMine { trigger_volume } => {
                info.activation_kind = 0;
                info.volume_count = 1;
                info.primary_volume = aabb_to_ffi(trigger_volume);
            }
            TriggerActivation::OnPillarLoss {
                pillars,
                condition,
                loss_threshold,
            } => {
                info.activation_kind = 1;
                info.volume_count = pillars.len().min(255) as u8;
                info.loss_threshold = *loss_threshold;
                use crate::triggers::LossCondition::*;
                let (tag, n) = match condition {
                    AnyPillar => (0u8, 0u8),
                    NPillars(n) => (1u8, *n),
                    AllPillars => (2u8, 0u8),
                };
                info.loss_condition = tag;
                info.loss_n = n;
                if let Some(p) = pillars.first() {
                    info.primary_volume = aabb_to_ffi(&p.volume);
                }
                for (i, p) in pillars.iter().take(8).enumerate() {
                    info.pillar_volumes[i] = aabb_to_ffi(&p.volume);
                }
            }
        }
        Some(info)
    }

    /// Remove a trigger by id. Returns true if it existed.
    pub fn remove_trigger(&self, id: u32) -> bool {
        let mut store = self.store.write().unwrap();
        store.remove_trigger(id).is_some()
    }

    /// Arm or disarm a trigger. When transitioning to armed, pillar
    /// baselines are recaptured (a previously-fired boss arena can be
    /// re-armed for iteration testing).
    pub fn set_trigger_armed(&self, id: u32, armed: bool) -> bool {
        let chunk_size = self.cached_chunk_size();
        let mut store = self.store.write().unwrap();
        let Some(trig) = store.find_trigger_mut(id) else {
            return false;
        };
        let was_armed = trig.armed;
        trig.armed = armed;
        // Recapture pillar baselines if newly armed (transition false→true).
        if armed && !was_armed {
            // Clone to break mutable borrow of `store` so we can read density.
            let trig_id = trig.id;
            drop(trig);
            // Now re-borrow mutably with a fresh lookup.
            let densities = store.density_fields.clone();
            if let Some(t) = store.find_trigger_mut(trig_id) {
                crate::triggers::refresh_pillar_baselines(t, &densities, chunk_size);
            }
        }
        true
    }

    /// Queue a trigger for force-fire on the next stress process tick. The
    /// trigger fires regardless of its `should_fire` evaluation — useful
    /// for editor "preview" buttons. Also queues a tiny stress dirty event
    /// so the stress queue wakes up promptly.
    pub fn fire_trigger_now(&self, id: u32) -> bool {
        let chunk_size = self.cached_chunk_size();
        let mut store = self.store.write().unwrap();
        let Some(trig) = store.find_trigger(id) else {
            return false;
        };
        // Center of primary volume (trigger_volume / pillars[0]).
        let center = match &trig.activation {
            crate::triggers::TriggerActivation::OnFirstMine { trigger_volume } => {
                aabb_center(trigger_volume)
            }
            crate::triggers::TriggerActivation::OnPillarLoss { pillars, .. } => pillars
                .first()
                .map(|p| aabb_center(&p.volume))
                .unwrap_or((0, 0, 0)),
        };
        if !store.force_fire_trigger_ids.contains(&id) {
            store.force_fire_trigger_ids.push(id);
        }
        // Ensure the trigger is armed so the dispatch picks it up.
        if let Some(t) = store.find_trigger_mut(id) {
            t.armed = true;
        }
        // Wake the stress queue.
        store
            .stress_dirty_events
            .push(voxel_core::stress::StressDirtyEvent {
                center,
                radius: 1,
                allow_collapse: true,
            });
        store.stress_dirty_time = Some(std::time::Instant::now());
        let _ = chunk_size;
        true
    }
}
