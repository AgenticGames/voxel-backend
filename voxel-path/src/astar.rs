//! Pure A* over a [`CellGrid`].
//!
//! Implementation notes:
//!  - Open set: `BinaryHeap` keyed by f-score (negated for min-heap behavior).
//!  - g/came_from: `HashMap<IVec3, ...>`. No separate closed set — the open
//!    heap carries each entry's g-score; on pop, a stale entry (one whose
//!    g_score exceeds the best g_score we've recorded for that cell) is
//!    skipped. Equivalent to a lazy closed set, costs one fewer HashMap probe
//!    per neighbor expansion than the eager-closed-set form.
//!  - 26-connected neighborhood (all face/edge/corner offsets except `(0,0,0)`).
//!  - Heuristic and step cost both euclidean — admissible, optimal paths.
//!  - Diagonal corner-clip guard: a move along a corner diagonal requires the
//!    two face-shared cells along that diagonal to also be air, otherwise the
//!    agent would clip through a solid edge.
//!  - Surface mode: cost adds a normal-discontinuity penalty so spider paths
//!    prefer smooth surfaces over sharp normal flips.

use crate::grid::CellGrid;
use crate::movement::{can_traverse, MovementMode};
use crate::smoothing::smooth_path;
use glam::{IVec3, Vec3};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// Inputs to a path query.
#[derive(Debug, Clone)]
pub struct PathRequest {
    /// Start cell in pathing-grid coordinates.
    pub from: IVec3,
    /// Goal cell in pathing-grid coordinates.
    pub to: IVec3,
    pub mode: MovementMode,
    /// Open-list expansion bound. Once the closed set hits this count the
    /// search terminates with [`PathStatus::MaxNodesReached`]. Recommended
    /// default ~10_000 (covers ~21^3 cells, plenty for typical chase ranges).
    pub max_nodes: u32,
    /// If true, run theta*-style line-of-sight smoothing on the path before
    /// returning. Default true; set false to inspect the raw A* trail.
    pub smooth: bool,
}

impl Default for PathRequest {
    fn default() -> Self {
        Self {
            from: IVec3::ZERO,
            to: IVec3::ZERO,
            mode: MovementMode::Flying { agent_radius_cells: 0.5 },
            max_nodes: 10_000,
            smooth: true,
        }
    }
}

/// Final outcome of a path query.
#[derive(Debug, Clone)]
pub struct PathOutcome {
    pub status: PathStatus,
    /// Sequence of waypoints from `from` to `to` (inclusive of both). Empty
    /// when no path exists.
    pub nodes: Vec<PathNode>,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathStatus {
    Success = 0,
    NoPath = 1,
    MaxNodesReached = 2,
    /// Path was found but some traversed area lay in unloaded chunks (treated
    /// as solid). AI should re-plan once those chunks stream in.
    PartiallyUnloaded = 3,
    /// Start or goal cell was invalid (solid, or otherwise can't be occupied
    /// per the movement mode).
    InvalidEndpoint = 4,
}

#[derive(Debug, Clone)]
pub struct PathNode {
    pub cell: IVec3,
    /// Outward surface normal at this cell — `Vec3::ZERO` for flying / walking
    /// modes or when not adjacent to a solid. Surface mode populates this.
    pub surface_normal: Vec3,
}

/// Compute a path. Read-only against the grid.
///
/// Returns a [`PathOutcome`] with status + nodes. The nodes list is empty on
/// failure but the status conveys *why*.
pub fn compute_path<G: CellGrid>(grid: &G, request: PathRequest) -> PathOutcome {
    // Validate endpoints — solid start/goal means we can't even begin.
    if !can_traverse(grid, request.from, request.mode)
        || !can_traverse(grid, request.to, request.mode)
    {
        return PathOutcome {
            status: PathStatus::InvalidEndpoint,
            nodes: Vec::new(),
        };
    }

    // Same cell — trivial path of one node.
    if request.from == request.to {
        return PathOutcome {
            status: PathStatus::Success,
            nodes: vec![PathNode {
                cell: request.from,
                surface_normal: maybe_normal(grid, request.from, request.mode),
            }],
        };
    }

    // ─── A* core ─────────────────────────────────────────────────
    let mut open: BinaryHeap<OpenEntry> = BinaryHeap::new();
    let mut g_score: HashMap<IVec3, f32> = HashMap::new();
    let mut came_from: HashMap<IVec3, IVec3> = HashMap::new();
    let mut touched_unloaded = false;
    let mut nodes_expanded: u32 = 0;

    // Clearance cost for wide agents (radius ≥ 1 cell, same gate as the
    // traversability shell): a soft penalty per solid SECOND-shell face
    // neighbor, so where a passage is wider than the legal minimum the route
    // prefers its centre instead of hugging an edge. Shortest-path edge-
    // hugging is why the same doorway pathed or failed depending on where
    // the player stood — the swept-ribbon validation downstream can bridge a
    // centred corridor but not one shaved along the door frame. Uniformly
    // tight corridors get a uniform penalty, so nothing becomes unpathable.
    // Cached per cell (the penalty is a property of the cell, not the edge).
    let wide = !request.mode.is_surface() && request.mode.agent_radius() >= 1.0;
    let mut clearance_cache: HashMap<IVec3, f32> = HashMap::new();

    g_score.insert(request.from, 0.0);
    open.push(OpenEntry {
        cell: request.from,
        g_score: 0.0,
        f_score: heuristic(request.from, request.to),
    });

    while let Some(top) = open.pop() {
        let current = top.cell;

        // Lazy closed-set: skip stale heap entries whose g_score is no longer
        // the best we've found for this cell. With a consistent (admissible +
        // monotonic) heuristic — euclidean on a grid IS consistent — the first
        // non-stale pop of any cell is its optimal-cost expansion. This pairs
        // with the `tentative_g < existing_g` neighbor update below.
        if let Some(&best_g) = g_score.get(&current) {
            if top.g_score > best_g {
                continue;
            }
        }

        if current == request.to {
            // Reconstruct path
            let raw_path = reconstruct(&came_from, current);
            let mut nodes: Vec<PathNode> = raw_path
                .into_iter()
                .map(|cell| PathNode {
                    cell,
                    surface_normal: maybe_normal(grid, cell, request.mode),
                })
                .collect();

            if request.smooth {
                nodes = smooth_path(grid, request.mode, nodes);
            }

            let status = if touched_unloaded {
                PathStatus::PartiallyUnloaded
            } else {
                PathStatus::Success
            };
            return PathOutcome { status, nodes };
        }

        nodes_expanded += 1;
        if nodes_expanded >= request.max_nodes {
            return PathOutcome {
                status: PathStatus::MaxNodesReached,
                nodes: Vec::new(),
            };
        }

        // g_score for `current` is `top.g_score` — guaranteed valid by the
        // stale-entry guard above. No HashMap lookup needed.
        let current_g = top.g_score;
        let current_normal = if request.mode.is_surface() {
            grid.surface_normal_at(current)
        } else {
            Vec3::ZERO
        };

        for &offset in NEIGHBOR_OFFSETS.iter() {
            let neighbor = IVec3::new(
                current.x + offset.x,
                current.y + offset.y,
                current.z + offset.z,
            );

            // Single HashMap probe handles both the "is this already in the
            // search frontier?" check AND fetches existing_g for the relaxation
            // test. When existing_g is Some(_), we also KNOW the cell is
            // traversable (it wouldn't have been inserted otherwise), so we
            // can skip the `can_traverse` grid probe — which on the live
            // ChunkStoreGrid is the most expensive op in this loop (DashMap
            // lookup + density grid sample per call).
            let existing_g = g_score.get(&neighbor).copied();

            if existing_g.is_none() {
                // Untouched neighbor — must verify traversability now.
                if !can_traverse(grid, neighbor, request.mode) {
                    if !grid.is_loaded(neighbor) {
                        touched_unloaded = true;
                    }
                    continue;
                }
            }
            // Diagonal corner-clip guard is per-step (depends on `current` and
            // `offset`, not just on `neighbor`), so it must run on every
            // expansion regardless of whether the neighbor was seen before.
            if !corner_clip_clear(grid, current, offset, request.mode) {
                continue;
            }

            let step_len_sq = (offset.x.pow(2) + offset.y.pow(2) + offset.z.pow(2)) as f32;
            let step_len = step_len_sq.sqrt(); // 1.0, √2, √3 for face/edge/corner moves

            // Surface-mode normal-discontinuity penalty: prefer smooth
            // transitions. k=0.5 chosen empirically — strong enough to bias
            // toward flat surfaces, weak enough to still allow corners when
            // necessary.
            let extra = if request.mode.is_surface() {
                let n_b = grid.surface_normal_at(neighbor);
                if current_normal.length_squared() > 0.0 && n_b.length_squared() > 0.0 {
                    let dot = current_normal.dot(n_b).clamp(-1.0, 1.0);
                    (1.0 - dot) * 0.5 * step_len
                } else {
                    0.0
                }
            } else if wide {
                let unit = if let Some(&p) = clearance_cache.get(&neighbor) {
                    p
                } else {
                    let p = clearance_pressure(grid, neighbor);
                    clearance_cache.insert(neighbor, p);
                    p
                };
                unit * step_len
            } else {
                0.0
            };

            let tentative_g = current_g + step_len + extra;
            if tentative_g < existing_g.unwrap_or(f32::INFINITY) {
                came_from.insert(neighbor, current);
                g_score.insert(neighbor, tentative_g);
                let f = tentative_g + heuristic(neighbor, request.to);
                open.push(OpenEntry {
                    cell: neighbor,
                    g_score: tentative_g,
                    f_score: f,
                });
            }
        }
    }

    PathOutcome {
        status: PathStatus::NoPath,
        nodes: Vec::new(),
    }
}

// ─── Internals ───────────────────────────────────────────────────

/// Min-heap entry — implements Ord with f_score reversed (BinaryHeap is max-heap).
/// Carries the g_score the entry was pushed with, so:
///  1. On pop, we can detect stale entries (`g_score > g_score_map[cell]`)
///     without a second HashMap probe.
///  2. The expansion step reads `current_g` directly off the entry instead of
///     looking it up in `g_score` again.
#[derive(Debug, Clone, Copy)]
struct OpenEntry {
    cell: IVec3,
    g_score: f32,
    f_score: f32,
}

impl Eq for OpenEntry {}
impl PartialEq for OpenEntry {
    fn eq(&self, other: &Self) -> bool {
        self.f_score == other.f_score
    }
}
impl Ord for OpenEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Lower f_score has higher priority — reverse the comparison.
        other.f_score.partial_cmp(&self.f_score).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for OpenEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn heuristic(a: IVec3, b: IVec3) -> f32 {
    let dx = (a.x - b.x) as f32;
    let dy = (a.y - b.y) as f32;
    let dz = (a.z - b.z) as f32;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn reconstruct(came_from: &HashMap<IVec3, IVec3>, mut current: IVec3) -> Vec<IVec3> {
    let mut out = vec![current];
    while let Some(&prev) = came_from.get(&current) {
        out.push(prev);
        current = prev;
    }
    out.reverse();
    out
}

fn maybe_normal<G: CellGrid>(grid: &G, cell: IVec3, mode: MovementMode) -> Vec3 {
    if mode.is_surface() {
        grid.surface_normal_at(cell)
    } else {
        Vec3::ZERO
    }
}

/// 26-connected neighborhood (all integer offsets in [-1,1]^3 except origin).
/// `const` so it lives in the binary's data section — no per-call construction.
static NEIGHBOR_OFFSETS: [IVec3; 26] = [
    IVec3::new(-1, -1, -1), IVec3::new(-1, -1, 0), IVec3::new(-1, -1, 1),
    IVec3::new(-1,  0, -1), IVec3::new(-1,  0, 0), IVec3::new(-1,  0, 1),
    IVec3::new(-1,  1, -1), IVec3::new(-1,  1, 0), IVec3::new(-1,  1, 1),
    IVec3::new( 0, -1, -1), IVec3::new( 0, -1, 0), IVec3::new( 0, -1, 1),
    IVec3::new( 0,  0, -1),                         IVec3::new( 0,  0, 1),
    IVec3::new( 0,  1, -1), IVec3::new( 0,  1, 0), IVec3::new( 0,  1, 1),
    IVec3::new( 1, -1, -1), IVec3::new( 1, -1, 0), IVec3::new( 1, -1, 1),
    IVec3::new( 1,  0, -1), IVec3::new( 1,  0, 0), IVec3::new( 1,  0, 1),
    IVec3::new( 1,  1, -1), IVec3::new( 1,  1, 0), IVec3::new( 1,  1, 1),
];

/// Diagonal corner-clip guard.
///
/// For an edge-diagonal step (two non-zero offsets), require both of the two
/// adjacent face-cells along the diagonal to be air. For a corner-diagonal
/// (three non-zero offsets), require all three face-cells along the path to
/// be air. Otherwise the agent would clip through a solid edge or corner.
/// "How boxed-in is this cell beyond the legal minimum" — drives the
/// wide-agent clearance cost. Face neighbors at distance 1 are guaranteed
/// open by the traversability shell, so pressure comes from solids two cells
/// out along faces AND the 12 edge-diagonals at distance 1 (which catch
/// one-voxel-thin walls/shelves a distance-2 probe skips straight past —
/// the first version of this missed exactly that in its own unit test).
fn clearance_pressure<G: CellGrid>(grid: &G, cell: IVec3) -> f32 {
    const FACES: [(i32, i32, i32); 6] = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ];
    const EDGES: [(i32, i32, i32); 12] = [
        (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
        (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
        (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
    ];
    let mut p = 0.0;
    for (dx, dy, dz) in FACES {
        if grid.is_solid(IVec3::new(cell.x + 2 * dx, cell.y + 2 * dy, cell.z + 2 * dz)) {
            p += 0.3;
        }
    }
    for (dx, dy, dz) in EDGES {
        if grid.is_solid(IVec3::new(cell.x + dx, cell.y + dy, cell.z + dz)) {
            p += 0.15;
        }
    }
    p
}

fn corner_clip_clear<G: CellGrid>(
    grid: &G,
    current: IVec3,
    offset: IVec3,
    mode: MovementMode,
) -> bool {
    let nonzero = (offset.x != 0) as i32 + (offset.y != 0) as i32 + (offset.z != 0) as i32;
    if nonzero <= 1 {
        return true; // pure face step — always allowed
    }

    let is_blocked = |c: IVec3| -> bool {
        // For Walking/Surface, an air cell whose floor is missing still counts
        // as "not currently standable" but it's not a clip. Clip only checks
        // is_solid — solid = clipped through.
        let _ = mode;
        grid.is_solid(c)
    };

    if nonzero == 2 {
        // Edge diagonal: check the two face-shared neighbors.
        let a = IVec3::new(
            current.x + (offset.x != 0).then_some(offset.x).unwrap_or(0),
            current.y + (offset.y != 0).then_some(0).unwrap_or(0),
            current.z + (offset.z != 0).then_some(0).unwrap_or(0),
        );
        let b = IVec3::new(
            current.x + (offset.x != 0).then_some(0).unwrap_or(0),
            current.y + (offset.y != 0).then_some(offset.y).unwrap_or(0),
            current.z + (offset.z != 0).then_some(0).unwrap_or(0),
        );
        let c = IVec3::new(
            current.x + (offset.x != 0).then_some(0).unwrap_or(0),
            current.y + (offset.y != 0).then_some(0).unwrap_or(0),
            current.z + (offset.z != 0).then_some(offset.z).unwrap_or(0),
        );
        // Two of (a, b, c) are the relevant face-shared neighbors — the third
        // is `current` itself (when one of the offset components is zero).
        // Filter to those != current.
        let mut blocked = 0;
        for n in [a, b, c] {
            if n != current && is_blocked(n) {
                blocked += 1;
            }
        }
        // If either neighbor is blocked we can't slip the diagonal.
        blocked == 0
    } else {
        // Corner diagonal (all three offsets nonzero): require all three
        // face-shared and all three edge-shared neighbors (along the
        // diagonal axis) to be air. Conservative — allows the move only when
        // the agent has full air clearance through the corner.
        let face_a = IVec3::new(current.x + offset.x, current.y, current.z);
        let face_b = IVec3::new(current.x, current.y + offset.y, current.z);
        let face_c = IVec3::new(current.x, current.y, current.z + offset.z);
        let edge_ab = IVec3::new(current.x + offset.x, current.y + offset.y, current.z);
        let edge_ac = IVec3::new(current.x + offset.x, current.y, current.z + offset.z);
        let edge_bc = IVec3::new(current.x, current.y + offset.y, current.z + offset.z);

        ![face_a, face_b, face_c, edge_ab, edge_ac, edge_bc]
            .iter()
            .any(|&c| is_blocked(c))
    }
}
