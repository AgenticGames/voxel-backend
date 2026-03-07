# /build — Structured Build Workflow

You are working on a **UE5 + Rust voxel game project**. The Rust backend lives in `~/voxel-backend/` (8 crates) and the UE5 project is at `D:/Unreal Projects/Mithril2026/` with the `VoxelBridge` plugin providing FFI integration. Consult MEMORY.md and its linked topic files for architecture details.

## Mandatory Workflow

You MUST follow this workflow exactly. Do not skip steps.

### Phase 1: Research (Agents)

Before writing any plan, you MUST use the **Task tool** to spawn research agents (subagent_type: `Explore` or `general-purpose`) to investigate:
- Which files and systems are involved in the requested feature
- Existing patterns, types, and interfaces that the feature must integrate with
- Potential conflicts or dependencies with other systems

Launch multiple research agents **in parallel** when investigating independent areas (e.g., one for Rust-side, one for UE-side). Do NOT start planning until research agents have reported back.

### Phase 2: Plan (EnterPlanMode)

After research is complete, you MUST use **EnterPlanMode** to create a detailed implementation plan. The plan MUST include:
- A summary of research findings
- Step-by-step implementation tasks broken down by file/module
- Which tasks can be parallelized using the **TeamCreate tool** and Task agents
- Explicit identification of shared interfaces / types that must be built first before parallel work begins
- A testing/validation strategy (cargo test, UE CLI build, viewer checks)

Wait for user approval of the plan before proceeding.

### Phase 3: Execute (Teams + Agents)

When executing the approved plan, you MUST use **TeamCreate** and spawn teammate agents via the **Task tool** to parallelize independent work streams. For example:
- One agent handles Rust-side implementation
- One agent handles UE C++ implementation
- One agent handles tests or viewer integration

Use **TaskCreate** to build the task list, assign owners, and track progress. Coordinate agents through the team messaging system. Ensure shared interfaces are committed before parallel agents begin dependent work.

### Phase 4: Validate & Backup

After implementation:
1. Run `cargo test --workspace` (export PATH to include `~/.cargo/bin`)
2. Run UE CLI build if C++ was changed
3. Commit and push to GitHub (per project conventions)
4. Update memory topic files if a major system was added

## Key Reminders
- Cargo: `~/.cargo/bin/cargo.exe` (not in default PATH)
- UE CLI build: see MEMORY.md for the UnrealBuildTool command
- DLL deploy path: `D:/cargo-target/release/voxel_ffi.dll` → ThirdParty dir
- Always check MEMORY.md topic files for existing system docs before designing new features
- HashMap iteration is nondeterministic — sort keys before RNG-dependent processing
- Adding GenerationConfig fields requires updating the explicit struct in `voxel-ffi/src/engine.rs`
