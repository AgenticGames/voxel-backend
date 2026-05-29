/// Sleep benchmark & tuning test suite.
///
/// All tests are `#[ignore]` — run with:
///   cargo test -p voxel-sleep bench_ -- --ignored --nocapture

use std::collections::BTreeMap;

// Use u8 keys for maps since Material doesn't implement Ord.
pub(crate) type MatMap<V> = BTreeMap<u8, V>;

mod helpers;
mod fixtures;
mod report;
mod tests_profile;
mod tests_rock;
mod tests_exploit;

pub(crate) use helpers::*;
pub(crate) use fixtures::*;
pub(crate) use report::*;
pub(crate) use tests_profile::*;
pub(crate) use tests_rock::*;
pub(crate) use tests_exploit::*;
