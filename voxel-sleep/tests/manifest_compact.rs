//! Integration test for `ChangeManifest::compact` to verify behavior of
//! the sort+coalesce rewrite. Lives outside the lib so the broken
//! `bench` test module (unrelated pre-existing compile error) doesn't
//! block running this verification.

use voxel_core::material::Material;
use voxel_core::stress::SupportType;
use voxel_sleep::ChangeManifest;

#[test]
fn compact_coalesces_repeats_first_old_last_new() {
    let mut manifest = ChangeManifest::new();
    manifest.record_voxel_change(
        (0, 0, 0), 3, 3, 3,
        Material::Limestone, 1.0,
        Material::Granite, 1.0,
    );
    manifest.record_voxel_change(
        (0, 0, 0), 3, 3, 3,
        Material::Granite, 1.0,
        Material::Marble, 0.8,
    );

    manifest.compact();
    let delta = manifest.chunk_deltas.get(&(0, 0, 0)).unwrap();
    assert_eq!(delta.voxel_changes.len(), 1);
    assert_eq!(delta.voxel_changes[0].old_material, Material::Limestone as u8);
    assert_eq!(delta.voxel_changes[0].old_density, 1.0);
    assert_eq!(delta.voxel_changes[0].new_material, Material::Marble as u8);
    assert_eq!(delta.voxel_changes[0].new_density, 0.8);
}

#[test]
fn compact_preserves_spread_distance_from_first_change() {
    let mut manifest = ChangeManifest::new();
    manifest.record_voxel_change_with_spread(
        (0, 0, 0), 1, 2, 3,
        Material::Limestone, 1.0,
        Material::Granite, 1.0,
        0.4,
    );
    manifest.record_voxel_change_with_spread(
        (0, 0, 0), 1, 2, 3,
        Material::Granite, 1.0,
        Material::Marble, 0.8,
        0.9,
    );

    manifest.compact();
    let delta = manifest.chunk_deltas.get(&(0, 0, 0)).unwrap();
    assert_eq!(delta.voxel_changes.len(), 1);
    assert!((delta.voxel_changes[0].spread_distance - 0.4).abs() < 1e-6,
        "spread_distance should be from first change");
}

#[test]
fn compact_distinct_voxels_preserved() {
    let mut manifest = ChangeManifest::new();
    manifest.record_voxel_change(
        (0, 0, 0), 1, 1, 1,
        Material::Limestone, 1.0,
        Material::Granite, 1.0,
    );
    manifest.record_voxel_change(
        (0, 0, 0), 2, 2, 2,
        Material::Limestone, 1.0,
        Material::Marble, 0.9,
    );
    manifest.record_voxel_change(
        (0, 0, 0), 3, 3, 3,
        Material::Limestone, 1.0,
        Material::Quartz, 1.0,
    );

    manifest.compact();
    let delta = manifest.chunk_deltas.get(&(0, 0, 0)).unwrap();
    assert_eq!(delta.voxel_changes.len(), 3);
    // After sort by (lx,ly,lz), expect order (1,1,1), (2,2,2), (3,3,3)
    assert_eq!(delta.voxel_changes[0].lx, 1);
    assert_eq!(delta.voxel_changes[1].lx, 2);
    assert_eq!(delta.voxel_changes[2].lx, 3);
}

#[test]
fn compact_empty_delta_is_noop() {
    let mut manifest = ChangeManifest::new();
    // Force-create an empty delta entry by recording then clearing.
    manifest.record_voxel_change(
        (5, 5, 5), 0, 0, 0,
        Material::Limestone, 1.0,
        Material::Granite, 1.0,
    );
    if let Some(delta) = manifest.chunk_deltas.get_mut(&(5, 5, 5)) {
        delta.voxel_changes.clear();
    }
    manifest.compact();
    // No panic, no creation of stray entries.
    let delta = manifest.chunk_deltas.get(&(5, 5, 5)).unwrap();
    assert!(delta.voxel_changes.is_empty());
}

#[test]
fn compact_three_voxel_run_keeps_first_and_last() {
    let mut manifest = ChangeManifest::new();
    // Three changes at the same voxel — verify we coalesce across runs >2.
    manifest.record_voxel_change(
        (0, 0, 0), 4, 4, 4,
        Material::Limestone, 1.0,
        Material::Granite, 1.0,
    );
    manifest.record_voxel_change(
        (0, 0, 0), 4, 4, 4,
        Material::Granite, 1.0,
        Material::Quartz, 0.95,
    );
    manifest.record_voxel_change(
        (0, 0, 0), 4, 4, 4,
        Material::Quartz, 0.95,
        Material::Marble, 0.7,
    );

    manifest.compact();
    let delta = manifest.chunk_deltas.get(&(0, 0, 0)).unwrap();
    assert_eq!(delta.voxel_changes.len(), 1);
    assert_eq!(delta.voxel_changes[0].old_material, Material::Limestone as u8);
    assert_eq!(delta.voxel_changes[0].new_material, Material::Marble as u8);
    assert!((delta.voxel_changes[0].new_density - 0.7).abs() < 1e-6);
}

#[test]
fn compact_support_changes_coalesce() {
    let mut manifest = ChangeManifest::new();
    manifest.record_support_change(
        (0, 0, 0), 2, 2, 2,
        SupportType::None, SupportType::SlateStrut,
    );
    manifest.record_support_change(
        (0, 0, 0), 2, 2, 2,
        SupportType::SlateStrut, SupportType::CrystalStrut,
    );

    manifest.compact();
    let delta = manifest.chunk_deltas.get(&(0, 0, 0)).unwrap();
    assert_eq!(delta.support_changes.len(), 1);
    assert_eq!(delta.support_changes[0].old_support, SupportType::None as u8);
    assert_eq!(delta.support_changes[0].new_support, SupportType::CrystalStrut as u8);
}
