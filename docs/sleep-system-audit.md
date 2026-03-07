# Deep Sleep / Geological Time Skip — Complete System Audit

> **Date:** 2026-03-06
> **Scope:** Rust voxel-sleep crate + FFI bridge + UE5 integration + geological accuracy
> **Total Duration Simulated:** 1,250,000 years per sleep cycle (4 phases)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [How It Works In-Game](#2-how-it-works-in-game)
3. [Phase-by-Phase Geological Processes](#3-phase-by-phase-geological-processes)
4. [Player Strategy & Exploitation](#4-player-strategy--exploitation)
5. [Voxel Range, Conditions & Budgets](#5-voxel-range-conditions--budgets)
6. [FFI Bridge & Data Flow](#6-ffi-bridge--data-flow)
7. [Geological Accuracy Audit](#7-geological-accuracy-audit)
8. [Implementation Gaps & Dead Code](#8-implementation-gaps--dead-code)
9. [Complete Transformation Reference](#9-complete-transformation-reference)
10. [Configuration Parameter Reference](#10-configuration-parameter-reference)

---

## 1. System Overview

The Deep Sleep system simulates geological time advancement when the player (an ancient insectoid) enters hibernation. Over 12-15 real-time seconds, 1.25 million years of geological change are applied across loaded chunks near the player. The system is designed so that **player actions before sleeping** (mining tunnels, placing lava, routing water, exposing pyrite) directly drive geological outcomes.

### Architecture

```
Player triggers sleep (U-key panel or Chrysalis)
        |
        v
UE5 VoxelWorldSubsystem::RequestDeepSleep()
        |
        v
FFI: voxel_start_sleep() → Worker thread
        |
        v
Worker acquires FluidSnapshot from voxel-fluid
        |
        v
voxel_sleep::execute_sleep() — 4 phases
        |
        v
ChangeManifest + SleepResult
        |
        v
Worker remeshes dirty chunks
        |
        v
FFI: voxel_poll_sleep_result() → Stats to UE5
```

### Crate: `voxel-sleep`

| Module | Purpose |
|--------|---------|
| `lib.rs` | Orchestrator: `execute_sleep()`, SleepResult, SleepProgress |
| `config.rs` | SleepConfig + 5 phase configs + GroundwaterConfig |
| `reaction.rs` | Phase 1: acid dissolution, oxidation, basalt crust |
| `aureole.rs` | Phase 2: contact metamorphism, coal maturation, silicification, erosion |
| `veins.rs` | Phase 3: hydrothermal BFS vein injection, formation growth |
| `deeptime.rs` | Phase 4: supergene enrichment, vein thickening, formations, fossilization |
| `collapse.rs` | Support degradation + stress cascade (called by Phase 4) |
| `groundwater.rs` | Ambient moisture model (porosity, depth, drip zones) |
| `manifest.rs` | ChangeManifest for persistence (VoxelChange/SupportChange, JSON) |
| `priority.rs` | ChunkTier classification (Critical/Important/Cosmetic) |
| `util.rs` | Shared: FACE_OFFSETS, sample_material(), aperture_multiplier() |
| `metamorphism.rs` | LEGACY (superseded by aureole.rs) |
| `minerals.rs` | LEGACY (superseded by veins.rs) |

### Determinism

- RNG: ChaCha8Rng seeded with `sleep_count * 7919 + 42`
- Identical inputs produce bit-identical outputs
- All chunk iteration is sorted before processing (HashMap nondeterminism guard)

---

## 2. How It Works In-Game

### Triggering Sleep

**U Key** toggles the Time Skip Panel (right-side, 420x920px, amber/gold theme). The panel has 7 sections with ~30 configurable parameters and two buttons:
- **SAVE SETTINGS** — persists config to `Saved/TimeSkipConfig.json` without sleeping
- **RUN TIME SKIP** — saves config AND immediately starts the sleep cycle

**Alternative:** The Chrysalis actor has a "Sleep" button that calls `RequestDeepSleep()` directly.

### Prerequisites

- Cannot open while in Build Mode, Inventory, Crafting UIs, or Formation Panel
- `bDeepSleepActive` must be false (no concurrent sleeps)
- VoxelEngine instance must be loaded with valid FFI function pointers

### During Sleep

- Sleep runs asynchronously on a Rust worker thread
- The entire ChunkStore (density/stress/support fields) is write-locked during execution
- Worker requests a FluidSnapshot from the fluid simulation thread
- All 4 phases execute sequentially; each phase produces a local ChangeManifest
- After all phases, dirty chunks are bulk-remeshed

### Results

- 15 stat counters returned via `FfiSleepResult` (acid dissolved, veins deposited, etc.)
- Profile report string saved to `Saved/SleepProfile/sleep_profile_YYYY-MM-DD_HHMMSS.txt`
- Stats logged to UE Output Log with `[SleepProfile]` prefix
- Dirty chunk meshes flow through the normal `voxel_poll_result()` pipeline
- Collapse events flow through `WorkerResult::CollapseResult`

### No Montage System Yet

The design document describes a cinematic montage with camera movement, phase text overlays, and per-phase timing (~2.5s/3s/4s/3s). **This is not yet implemented.** Currently, sleep runs as a black-box computation with results displayed after completion.

---

## 3. Phase-by-Phase Geological Processes

### Phase 1: "The Reaction" — 10,000 Years

Fast chemistry: acid attack and surface oxidation.

#### 1A. Acid Dissolution (Pyrite → Sulfuric Acid → Limestone Voids)

- **Trigger:** Pyrite voxel with >= 1 air neighbor (player-exposed)
- **Algorithm:** BFS through connected Limestone, max depth 3
- **Probability:** 60% per BFS node
- **Result:** Limestone → Air (density -1.0) — creates new cave voids
- **Teaching:** "Your mine exposed pyrite to air. The acid it generated dissolved the rock around it."

#### 1B. Copper Oxidation

- **Trigger:** Copper voxel with >= 1 air neighbor
- **Probability:** 50%
- **Result:** Copper → Malachite (green patina)
- **Teaching:** Copper exposed to air and moisture oxidizes over millennia

#### 1C. Basalt Crust Formation

- **Trigger:** Solid voxel adjacent to Lava (from FluidSnapshot)
- **Probability:** 70%
- **Result:** Any solid (except Basalt/Kimberlite) → Basalt (1-2 deep cooling rind)
- **Teaching:** Lava margins cool to form a basaltic shell

#### 1D. Sulfide Acid Dissolution

- **Trigger:** Sulfide voxel with >= 1 air neighbor
- **Algorithm:** BFS through Limestone, base radius 2
- **Water amplification:** If adjacent to water, radius doubles (2 → 4)
- **Probability:** 45% per node
- **Result:** Limestone → Air

---

### Phase 2: "The Aureole" — 100,000 Years

Heat transforms rock: metamorphism spreads from lava and kimberlite.

#### Heat Map Construction

Heat sources are collected from two places:
1. **Lava cells** from FluidSnapshot (player-placed fluid)
2. **Kimberlite voxels** from density fields (natural deposits)

#### 2A. Contact Metamorphism

Distance-based aureole zones around each heat source (Chebyshev distance):

| Zone | Distance | Transformations |
|------|----------|----------------|
| Contact | 0-2 | Limestone → Marble (80%), Sandstone → Granite (50%) |
| Mid | 3-5 | Limestone → Marble (50%), Sandstone → Granite (25%) |
| Outer | 6-8 | Limestone → Marble (20%), Slate → Marble (30%) |

**Teaching:** "The lava you placed recrystallized surrounding limestone into marble. The ring shows exactly how far the heat reached."

#### 2B. Coal Maturation

| Condition | Result | Probability |
|-----------|--------|-------------|
| Coal within 1 voxel of Kimberlite | Diamond | 15% |
| Coal within 2 voxels of any heat | Graphite | 70% |
| Coal within 3-5 voxels of any heat | Graphite | 35% |

#### 2C. Silicification (requires water in FluidSnapshot)

- Mid aureole (3-5): Limestone → Quartz (30%), Sandstone → Quartz (15%)
- Outer aureole (6+): Half probability

#### 2D. Water Erosion (two mechanisms)

1. **Explicit water:** Water cells from FluidSnapshot erode adjacent Limestone/Sandstone → Air (5%)
2. **Ambient groundwater:** Depth-dependent moisture model erodes air-adjacent soft rock

---

### Phase 3: "The Veins" — 500,000 Years

Hydrothermal ore deposition: THE GAMEPLAY PAYOFF.

#### 3A. Hydrothermal Vein Injection

- **Algorithm:** BFS from heat sources through AIR voxels (player-mined tunnels = fracture pathways)
- **Deposition:** Ore placed on solid host-rock walls adjacent to air BFS nodes
- **Max distance:** 16 voxels from heat source
- **Budget:** 12 ore voxels per heat source
- **Base probability:** 25%, modified by aperture scaling

**Temperature Zonation (Lindgren classification):**

| Zone | BFS Distance | Ore Deposited |
|------|-------------|---------------|
| Hypothermal (high temp) | 0-4 | Tin (40%) / Quartz (60%) |
| Mesothermal (mid temp) | 4-10 | Copper (50%) / Iron (50%) |
| Epithermal (low temp) | 10-16 | Gold (40%) / Sulfide (60%) |

**Aperture Scaling** (wider tunnels = richer veins):

| Air Neighbors | Multiplier | Interpretation |
|--------------|------------|----------------|
| 0 | 0.00x | No flow |
| 1 | 1.40x | Wide tunnel, rich |
| 2 | 1.15x | Moderately wide |
| 3 | 1.00x | Balanced |
| 4 | 0.65x | Tight crack |
| 5 | 0.40x | Very narrow |
| 6+ | 0.20x | Fissure |

**Teaching:** "Hot mineral-rich water circulated from your lava through your tunnels. As it cooled, copper deposited here, tin closer to the heat."

#### 3B. Cave Formation Growth

| Formation | Condition | Prob | Budget |
|-----------|-----------|------|--------|
| Crystal growth | Air + 2+ Crystal/Amethyst neighbors | 30% | 4/chunk |
| Calcite infill | Air + 3+ Limestone neighbors | 15% | 4/chunk |
| Flowstone | Air adjacent to water paths OR ceiling drip zone | 10% | 3/chunk |

---

### Phase 4: "The Deep Time" — 1,250,000 Years

Enrichment, maturation, and structural reckoning.

#### 4A. Supergene Enrichment

Two mechanisms:
1. **Explicit water:** Water cells → check 1-3 voxels above → if host rock with Cu/Fe/Au within Manhattan dist 5 → concentrate ore (15%, max 8/chunk)
2. **Ambient groundwater:** Drip zones (ceiling voxels with air below) in soft rock, or fracture sites (1-2 air neighbors) in hard rock

**Geochemical fallback** (no nearby ore detected):

| Host Rock | Trace Ore |
|-----------|-----------|
| Granite / Basalt | Iron (60%) / Copper (40%) |
| Limestone | Iron (50%) / Malachite (50%) |
| Sandstone | Iron (100%) |
| Slate / Marble | Copper (50%) / Quartz (50%) |

**Teaching:** "Water you diverted carried dissolved copper downward and concentrated it above the cave ceiling."

#### 4B. Vein Thickening

- Air-adjacent ore expands into neighboring host rock (10%, max 4/chunk)

#### 4C. Mature Formations

- Stalactite growth: Air under limestone ceiling → Limestone (10%)
- Column formation: Air between limestone above and below → Limestone (5%)

#### 4D. Nest Fossilization

Spider nest positions (passed via `voxel_set_sleep_nests()`) are mineralized:

| Condition | Result | Probability |
|-----------|--------|-------------|
| Iron-rich host + water + buried | Pyrite | 60% |
| Silica-rich host + water + buried | Opal | 40% |
| Limestone + water + buried + !iron-rich | Opal | 25% |
| Near lava | No fossilization | — |

Fossilization radius: Manhattan distance 2 from nest center.

#### 4E. Structural Collapse

Three sub-steps:
1. **Support degradation:** Each strut type has a survival rate (Slate 25% → Crystal 95%), modulated by local stress
2. **Stress amplification:** All stress values multiplied by 1.5x
3. **Collapse cascade:** Up to 8 iterations of stress propagation + rubble placement (40% fill ratio)

---

## 4. Player Strategy & Exploitation

### The Preparation Loop (Before Each Sleep)

| Action | Target Outcome | Phase |
|--------|---------------|-------|
| Place lava near desired ore zones | Drives metamorphism + hydrothermal veins | 2, 3 |
| Mine passages connecting heat to target walls | Creates BFS pathways for ore deposition | 3 |
| Route water to areas for enrichment | Enables supergene concentration | 4 |
| Expose pyrite near limestone walls | Acid carves new passages | 1 |
| Place high-tier struts in important areas | Survives Phase 4 collapse | 4 |
| Avoid water near storage/bases | Dissolution will eat it | 2 |

### Interaction Matrix

| Player Action | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|---------------|---------|---------|---------|---------|
| Place lava | Basalt crust | Marble aureole | **ORE VEINS** | Veins thicken |
| Mine passages near heat | — | — | **Veins follow tunnels** | Second-gen veins |
| Route water | — | Channel erosion | — | **Supergene enrichment** |
| Expose pyrite | **Acid voids** | Passage growth | — | Continued dissolution |
| Expose copper | Malachite | — | — | — |
| Place crystal struts | — | — | — | Survives collapse |
| Lava + water combo | — | — | **MAXIMUM veins** | Maximum enrichment |

### Optimal Exploitation Strategy

1. **First sleep:** Place lava, mine long tunnels radiating outward (16 voxels max), route water above. Sleep generates zonated ore veins along your tunnels.
2. **Between sleeps:** Mine the new ore, open new passages connecting to fresh rock faces. Place lava at new depth.
3. **Second sleep:** New veins in fresh passages + supergene enrichment matures (richest concentrations). Old areas may collapse, exposing new rock faces.
4. **Diminishing returns:** Same location yields less each cycle. Player must push deeper, placing new heat sources and creating new pathways.

### Key Insight: Sleep Count Matters

The RNG seed changes with `sleep_count`, so each sleep produces different random outcomes. However, the geology of an area becomes increasingly "used up" — limestone dissolved, ore deposited, host rock transformed.

---

## 5. Voxel Range, Conditions & Budgets

### Chunk Radius Filter

The `chunk_radius` parameter (default configurable in U-key panel) determines how many chunks around the player are affected:

| chunk_radius | Chunks Affected | Pattern |
|-------------|----------------|---------|
| 0 | 1 | Player chunk only |
| 1 | 7 | Player + 6 face-adjacent |
| 2 | 27 | 3x3x3 cube |
| 3 | 125 | 5x5x5 cube |

### Priority Tiers (which phases run where)

| Tier | Definition | Phases |
|------|-----------|--------|
| Critical | Player chunk + 6 face-adjacent (Manhattan <= 1) | All 4 |
| Important | 2-ring neighbors (Chebyshev <= 2) + chunks with supports | 1-3 |
| Cosmetic | All other loaded chunks | 1-2 only |

### Per-Process Budgets

| Process | Budget | Scope |
|---------|--------|-------|
| Acid dissolution | Unlimited (radius-limited to 3) | Per pyrite source |
| Copper oxidation | Unlimited | All air-adjacent copper |
| Basalt crust | Unlimited | All lava-adjacent solids |
| Sulfide acid | Unlimited (radius 2, or 4 with water) | Per sulfide source |
| Contact metamorphism | Unlimited (radius 8) | Per heat source |
| Hydrothermal veins | **12 ore voxels per heat source** | BFS max depth 16 |
| Crystal growth | **4 per chunk** | Air with 2+ crystal neighbors |
| Calcite infill | **4 per chunk** | Air with 3+ limestone neighbors |
| Flowstone | **3 per chunk** | Water paths or drip zones |
| Supergene enrichment | **8 per chunk** | Drip zones or fracture sites |
| Vein thickening | **4 per chunk** | Air-adjacent ore |
| Stalactites/columns | Unlimited | Limestone ceiling/floor |
| Nest fossilization | Unlimited | Manhattan radius 2 from nest |
| Collapse | **8 cascade iterations** | Stress-based |

### Groundwater Model

Ambient moisture enables geological effects without player-placed water:

```
moisture = (strength * depth_factor * porosity * drip_mult).clamp(0, 1)

depth_factor = ((baseline - world_y) * scale).clamp(0, 1)
```

**Porosity by Rock Type:**

| Rock | Porosity | Classification |
|------|----------|---------------|
| Limestone | 1.0 | Soft (karst-forming) |
| Sandstone | 0.8 | Soft |
| Slate | 0.5 | Hard |
| Marble | 0.3 | Hard |
| Granite | 0.2 | Hard |
| Basalt | 0.1 | Hard |

Hard rock requires fracture sites (1-2 air neighbors) for groundwater effects; soft rock works at any drip zone.

---

## 6. FFI Bridge & Data Flow

### Config Crossing the Boundary

Sleep config is NOT a separate FFI struct. All ~96 sleep parameters are embedded in the monolithic `FfiEngineConfig` / `FVoxelEngineConfig`. The mapping function `ffi_config_to_sleep()` converts to internal `SleepConfig`.

### FFI Functions

| Function | Direction | Purpose |
|----------|-----------|---------|
| `voxel_set_sleep_config` | UE → Rust | Push full engine config before sleep |
| `voxel_set_sleep_nests` | UE → Rust | Set spider nest world positions (coord-transformed) |
| `voxel_start_sleep` | UE → Rust | Initiate async sleep on worker thread |
| `voxel_poll_sleep_result` | Rust → UE | Poll for completion + 15 stat counters |
| `voxel_free_sleep_result` | UE → Rust | Free heap-allocated profile report string |

### Result Flow (Split Across Two Channels)

1. **Summary stats:** `voxel_poll_sleep_result()` returns `FfiSleepResult`
2. **Dirty chunk meshes:** Flow through normal `voxel_poll_result()` as `ChunkMesh` results
3. **Collapse events:** Flow through normal `voxel_poll_result()` as `CollapseResult`
4. **Profile report:** Heap-allocated C string in FfiSleepResult, freed via `voxel_free_sleep_result()`

### Critical Note

- DirtyChunks and CollapseEvents fields in FfiSleepResult are **always nullptr** — they flow through the normal result pipeline
- FluidSnapshot defaults silently to empty if the fluid thread is slow/crashed
- Worker thread holds a write lock on the entire ChunkStore during sleep — no incremental streaming
- Nest positions must be set BEFORE starting sleep (no nil-safety check)

---

## 7. Geological Accuracy Audit

### Scorecard

| Process | Verdict | Notes |
|---------|---------|-------|
| Pyrite → acid → limestone dissolution | **ACCURATE** | Textbook acid mine drainage |
| Copper → Malachite | **ACCURATE** | Supergene oxidation zone chemistry |
| Lava → Basalt crust | **ACCURATE** | Chilled margin formation |
| Sulfide → acid (water-amplified) | **ACCURATE** | AMD with correct water dependency |
| Limestone → Marble (contact) | **ACCURATE** | Classic contact metamorphism |
| **Sandstone → Granite (contact)** | **WRONG** | Granite is igneous, not metamorphic. Should be Sandstone → Quartzite |
| **Slate → Marble (outer aureole)** | **WRONG** | Slate is silicate, marble is carbonate. Impossible transformation |
| Coal → Graphite (heat) | **ACCURATE** | Documented in contact aureoles |
| Coal → Diamond (kimberlite) | **CREATIVE LICENSE** | Kimberlite transports diamonds, doesn't create them. Acceptable for gameplay |
| Silicification (Limestone → Quartz) | **ACCURATE** | Chert/silica replacement is real |
| Water erosion (karst) | **ACCURATE** | Fundamental speleogenesis |
| Hydrothermal vein zonation | **ACCURATE** | Follows Lindgren classification correctly |
| Aperture scaling (wider = richer) | **ACCURATE** | Fluid dynamics principle |
| Supergene enrichment | **ACCURATE** | Creates world's richest copper deposits |
| Vein thickening | **ACCURATE** | Wall-rock replacement is standard |
| Nest → Pyrite fossilization | **DEFENSIBLE** | Real pyritization process, correct conditions |
| Nest → Opal fossilization | **DEFENSIBLE** | Real opalization, correct silica-rich requirement |
| Cave formation growth rates | **ACCURATE** | 0.01-0.3 mm/yr, consistent with 1.25M yr |
| Structural collapse | **ACCURATE** | Natural long-term cave breakdown |

### Detailed Error Analysis

#### ERROR 1: Sandstone → Granite (HIGH SEVERITY)

**The problem:** Granite is an *igneous* rock formed from slow cooling of silica-rich magma deep in Earth's crust. Contact metamorphism of sandstone produces **quartzite** — the quartz grains recrystallize and fuse together. These are fundamentally different rock formation processes.

**Real process:** Sandstone + heat → Quartzite (quartz grain recrystallization)

**Recommended fix:** Change to Sandstone → Quartz (closest existing material). If adding a new material, Quartzite would be material index 22.

**Location in code:** `voxel-sleep/src/aureole.rs` — contact and mid aureole zones

#### ERROR 2: Slate → Marble (HIGH SEVERITY)

**The problem:** Slate is a silicate/aluminosilicate rock (metamorphosed mudstone). Marble forms exclusively from carbonate rocks (limestone, dolostone). A silicate rock cannot transform into a carbonate rock — the chemistry is entirely wrong. Slate's metamorphic progression is: Slate → Phyllite → Schist → Gneiss.

**Real process:** Slate + extreme heat → Hornfels (not in palette) or partial melt → granitic composition

**Recommended fix options:**
1. Slate → Granite (at extreme metamorphic grade, partial melting of aluminous slate produces granitic melts — this is geologically defensible)
2. Remove the transformation entirely
3. Add Hornfels as a new material

**Location in code:** `voxel-sleep/src/aureole.rs` — outer aureole zone only

#### CREATIVE LICENSE: Coal → Diamond

**The reality:** Diamonds form at 150-700 km depth in Earth's mantle (45-60 kbar pressure, 900-1300 degrees C). Kimberlite pipes are the delivery mechanism, carrying pre-existing diamonds to the surface — they don't create them.

**Why it's acceptable:** Coal contains carbon, kimberlite is associated with diamonds, and the distance <= 1 restriction makes it rare. The thematic logic works for gameplay even if the physics are wrong.

### Timescale Assessment

| Phase | Simulated Time | Real-World Accuracy |
|-------|---------------|-------------------|
| Phase 1 | 10,000 yr | Reasonable. AMD and oxidation occur in decades-millennia. Basalt forms in days (conservative). |
| Phase 2 | 100,000 yr | Accurate. Contact metamorphism takes thousands-hundreds of thousands of years. |
| Phase 3 | 500,000 yr | Accurate. Major hydrothermal systems operate 100K-millions of years. |
| Phase 4 | 1,250,000 yr | Accurate. Supergene enrichment blankets develop over 100K-millions of years. |

**The relative ordering is correct:** fast chemistry → metamorphism → ore deposition → enrichment. This mirrors real geological sequencing.

### Missing Real-World Processes (Not Simulated)

| Process | What Happens | Difficulty to Add |
|---------|-------------|-------------------|
| Dolomitization | Limestone + Mg-fluids → Dolomite | Medium (needs new material) |
| Granite weathering | Granite + water → Clay/Kaolin | Medium (needs new material) |
| Iron gossan | Iron ore + air + water → Limonite crust | Low (surface indicator only) |
| Coal rank progression | Lignite → Bituminous → Anthracite → Graphite | Low (intermediate steps) |
| Secondary sulfide zone | Cu leaching → Chalcocite/Covellite blanket below malachite | Medium |
| Calcite reprecipitation | Dissolved CaCO3 → Travertine/Tufa where CO2 degasses | Low (partially covered by flowstone) |

---

## 8. Implementation Gaps & Dead Code

### Implemented But No Clear Gameplay Use

| Feature | Status | Gap |
|---------|--------|-----|
| Legacy metamorphism.rs | Code exists, configs populated | Superseded by aureole.rs but still in config. Dead code. |
| Legacy minerals.rs | Code exists, configs populated | Superseded by veins.rs but still in config. Dead code. |
| 53 legacy FFI fields | Struct space occupied | Mapped to configs but ignored by new phase code |
| `time_budget_ms` config | Exposed in U-key panel | **Not actually enforced** — sleep runs to completion regardless |
| Montage camera system | Design doc exists | **Not implemented** — no camera movement or phase overlays |
| Phase-aware transform log | Design doc describes it | Transform log exists but **no per-phase text overlays in UE** |
| Sleep duration slider | Design doc mentions it | **Not implemented** |
| Sound design per phase | Design doc mentions it | **Not implemented** |
| VFX per phase | Design doc mentions it | **Not implemented** |
| `voxel_set_sleep_nests` | Function exists in Rust | **No C++ typedef in VoxelFFI.h** — not callable from UE |

### Asymmetries in Phase Processing

| Phase | Runs On | Gap |
|-------|---------|-----|
| Phase 1 (Reaction) | all_chunks | Widest spread — acid can dissolve far from player |
| Phase 2 (Aureole) | all_chunks | Widest spread — metamorphism affects distant chunks |
| Phase 3 (Veins) | mineral_chunks only (Critical + Important) | Narrower — ore only near player |
| Phase 4 (DeepTime) | collapse_chunks only (Critical + Important) | Narrower — enrichment only near player |

This means acid dissolution and metamorphism spread much farther than ore deposition, which could feel inconsistent if the player expects ore everywhere marble formed.

### Config Parameters Not Exposed to UE Panel

Many sub-phase parameters are hardcoded in Rust with no FFI mapping:

- Sulfide acid radius and water amplification factor
- Individual aureole zone probabilities (only `contact_marble_prob` exposed)
- Aperture scaling enable/disable
- Temperature zonation boundaries (hypothermal_max, mesothermal_max)
- Crystal/calcite/flowstone individual enables and per-chunk caps
- Enrichment search radius and geochemical fallback table
- Nest fossilization sub-parameters (radius, buried_required, water requirements)
- All individual strut survival rates

### FluidSnapshot Dependency

If no fluid simulation is running (viewer, CLI, or fluid thread crash), FluidSnapshot defaults to empty. This means:
- **Phase 1:** No basalt crust (no lava detected), sulfide acid gets base radius only
- **Phase 2:** No heat from lava (only kimberlite provides heat), no explicit water erosion
- **Phase 3:** No lava-based vein injection (kimberlite only), no explicit flowstone
- **Phase 4:** No explicit water enrichment, only ambient groundwater

The ambient groundwater model compensates for some of this, but **a sleep without fluids produces dramatically less change**.

---

## 9. Complete Transformation Reference

### All Material Transformations

| Input | Output | Phase | Condition | Probability |
|-------|--------|-------|-----------|-------------|
| Limestone | Air | 1 | Adjacent to exposed pyrite, BFS depth <= 3 | 60% |
| Limestone | Air | 1 | Adjacent to exposed sulfide, BFS depth <= 2 (4 w/ water) | 45% |
| Copper | Malachite | 1 | >= 1 air neighbor | 50% |
| Any solid (not Basalt/Kimberlite) | Basalt | 1 | Adjacent to lava | 70% |
| Limestone | Marble | 2 | Within 2 of heat source | 80% |
| Limestone | Marble | 2 | 3-5 from heat source | 50% |
| Limestone | Marble | 2 | 6-8 from heat source | 20% |
| Sandstone | Granite | 2 | Within 2 of heat source | 50% |
| Sandstone | Granite | 2 | 3-5 from heat source | 25% |
| Slate | Marble | 2 | 6-8 from heat source | 30% |
| Coal | Diamond | 2 | Within 1 of Kimberlite | 15% |
| Coal | Graphite | 2 | Within 2 of heat | 70% |
| Coal | Graphite | 2 | 3-5 from heat | 35% |
| Limestone | Quartz | 2 | 3-5 from heat + water exists | 30% |
| Limestone | Quartz | 2 | 6+ from heat + water exists | 15% |
| Sandstone | Quartz | 2 | 3-5 from heat + water exists | 15% |
| Sandstone | Quartz | 2 | 6+ from heat + water exists | 7.5% |
| Limestone/Sandstone | Air | 2 | Adjacent to water cell | 5% |
| Limestone/Sandstone | Air | 2 | Ambient groundwater + air neighbor | moisture * 5% |
| Host rock | Tin/Quartz | 3 | BFS dist 0-4 from heat, on tunnel wall | 25% * aperture |
| Host rock | Copper/Iron | 3 | BFS dist 4-10 from heat, on tunnel wall | 25% * aperture |
| Host rock | Gold/Sulfide | 3 | BFS dist 10-16 from heat, on tunnel wall | 25% * aperture |
| Air | Crystal | 3 | 2+ Crystal/Amethyst neighbors | 30% (4/chunk) |
| Air | Limestone | 3 | 3+ Limestone neighbors (calcite infill) | 15% (4/chunk) |
| Air | Limestone | 3 | Adjacent to water path (flowstone) | 10% (3/chunk) |
| Host rock | Cu/Fe/Au (nearby ore) | 4 | Above water + ore within dist 5 | 15% (8/chunk) |
| Host rock | Geochemical ore | 4 | Drip zone + groundwater + no nearby ore | 7.5% * moisture |
| Host rock (adjacent to ore) | Same ore | 4 | Ore has air neighbor | 10% (4/chunk) |
| Air | Limestone | 4 | Limestone above (stalactite) | 10% |
| Air | Limestone | 4 | Limestone above AND below (column) | 5% |
| Solid (near nest) | Pyrite | 4 | Iron-rich host + water + buried | 60% |
| Solid (near nest) | Opal | 4 | Silica-rich host + water + buried | 40% |

---

## 10. Configuration Parameter Reference

### Phase 1: Reaction

| Parameter | Default | Description |
|-----------|---------|-------------|
| `acid_dissolution_prob` | 0.60 | Per-BFS-node dissolution chance |
| `acid_dissolution_radius` | 3 | Max BFS depth from pyrite |
| `copper_oxidation_prob` | 0.50 | Air-adjacent copper → malachite |
| `basalt_crust_prob` | 0.70 | Lava-adjacent → basalt |
| `sulfide_acid_prob` | 0.45 | Sulfide-driven dissolution |
| `sulfide_acid_radius` | 2 | Base BFS depth (x2 with water) |
| `sulfide_water_amplification` | 2.0 | Radius multiplier when water present |

### Phase 2: Aureole

| Parameter | Default | Description |
|-----------|---------|-------------|
| `aureole_radius` | 8 | Max distance from heat for metamorphism |
| `contact_limestone_to_marble_prob` | 0.80 | Contact zone (0-2) |
| `contact_sandstone_to_granite_prob` | 0.50 | Contact zone (0-2) |
| `mid_limestone_to_marble_prob` | 0.50 | Mid aureole (3-5) |
| `mid_sandstone_to_granite_prob` | 0.25 | Mid aureole (3-5) |
| `outer_limestone_to_marble_prob` | 0.20 | Outer aureole (6-8) |
| `outer_slate_to_marble_prob` | 0.30 | Outer aureole (6-8) |
| `coal_to_graphite_prob` | 0.70 | Contact zone coal maturation |
| `coal_to_graphite_mid_prob` | 0.35 | Mid aureole coal maturation |
| `graphite_to_diamond_prob` | 0.15 | Kimberlite-contact diamond genesis |
| `silicification_limestone_prob` | 0.30 | Limestone → Quartz (water required) |
| `silicification_sandstone_prob` | 0.15 | Sandstone → Quartz (water required) |
| `water_erosion_prob` | 0.05 | Erosion per water contact |
| `water_erosion_enabled` | true | Toggle erosion sub-system |
| `coal_maturation_enabled` | true | Toggle coal → graphite/diamond |
| `silicification_enabled` | true | Toggle silicification |

### Phase 3: Veins

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vein_deposition_prob` | 0.25 | Base ore placement chance |
| `max_vein_voxels_per_source` | 12 | Budget per heat source |
| `vein_max_distance` | 16 | Max BFS depth |
| `hypothermal_max` | 4 | High-temp zone boundary |
| `mesothermal_max` | 10 | Mid-temp zone boundary |
| `aperture_scaling_enabled` | true | Wider tunnels = richer veins |
| `crystal_growth_prob` | 0.30 | Crystal formation chance |
| `crystal_growth_max_per_chunk` | 4 | Crystal budget per chunk |
| `calcite_infill_prob` | 0.15 | Calcite fill chance |
| `calcite_infill_max_per_chunk` | 4 | Calcite budget per chunk |
| `flowstone_prob` | 0.10 | Flowstone formation chance |
| `flowstone_max_per_chunk` | 3 | Flowstone budget per chunk |
| `growth_density_min` | 0.3 | Min density for crystal growth |
| `growth_density_max` | 0.6 | Max density for crystal growth |

### Phase 4: Deep Time

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enrichment_prob` | 0.15 | Supergene enrichment chance |
| `max_enrichment_per_chunk` | 8 | Enrichment budget per chunk |
| `enrichment_search_radius` | 5 | Manhattan distance for ore search |
| `vein_thickening_prob` | 0.10 | Ore expansion chance |
| `vein_thickening_max_per_chunk` | 4 | Thickening budget per chunk |
| `stalactite_growth_prob` | 0.10 | Stalactite formation chance |
| `column_formation_prob` | 0.05 | Column formation chance |
| `mature_formations_enabled` | true | Toggle formations |
| `vein_thickening_enabled` | true | Toggle thickening |
| `enrichment_enabled` | true | Toggle enrichment |

### Nest Fossilization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nest_fossilization.enabled` | true | Toggle system |
| `nest_fossilization.nest_radius` | 2 | Manhattan distance from nest center |
| `nest_fossilization.pyrite_prob` | 0.60 | Iron-rich → pyrite chance |
| `nest_fossilization.opal_prob` | 0.40 | Silica-rich → opal chance |
| `nest_fossilization.buried_required` | true | Must have 0 air neighbors |
| `nest_fossilization.water_required_for_pyrite` | true | Needs adjacent water |
| `nest_fossilization.water_required_for_opal` | true | Needs adjacent water |

### Collapse

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stress_multiplier` | 1.5 | Geological stress amplification |
| `max_cascade_iterations` | 8 | Cascade simulation depth |
| `rubble_fill_ratio` | 0.40 | Fraction of void filled with rubble |
| `min_stress_for_cascade` | 0.70 | Threshold to trigger cascade |

### Strut Survival Rates

| Strut Type | Survival | Failure (base) |
|-----------|----------|----------------|
| Slate | 25% | 75% |
| Limestone | 25% | 75% |
| Granite | 30% | 70% |
| Copper | 55% | 45% |
| Iron | 70% | 30% |
| Steel | 85% | 15% |
| Crystal | 95% | 5% |

*Actual failure = base_failure * (1.0 + local_stress), capped at 1.0*

### Groundwater

| Parameter | Default | Description |
|-----------|---------|-------------|
| `groundwater.enabled` | true | Toggle ambient moisture |
| `groundwater.strength` | 0.3 | Overall passive moisture |
| `groundwater.depth_baseline` | 0.0 | Y-level where moisture starts |
| `groundwater.depth_scale` | 0.02 | Moisture gain per Y unit below baseline |
| `groundwater.drip_zone_multiplier` | 2.0 | Boost for ceiling voxels |
| `groundwater.erosion_power` | 1.0 | Erosion rate multiplier |
| `groundwater.flowstone_power` | 1.0 | Flowstone rate multiplier |
| `groundwater.enrichment_power` | 1.0 | Enrichment rate multiplier |
| `groundwater.soft_rock_mult` | 1.0 | Limestone/Sandstone rate multiplier |
| `groundwater.hard_rock_mult` | 0.15 | Granite/Basalt/Slate/Marble rate multiplier |

---

*Report generated 2026-03-06 by research agents analyzing voxel-sleep (Rust), voxel-ffi (FFI bridge), VoxelBridge (UE5 plugin), and cross-referencing with geological literature.*
