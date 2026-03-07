# /design — Collaborative Design Workflow

You are working on a **UE5 + Rust voxel game project**. The Rust backend lives in `~/voxel-backend/` (8 crates) and the UE5 project is at `D:/Unreal Projects/Mithril2026/` with the `VoxelBridge` plugin providing FFI integration. Consult MEMORY.md and its linked topic files for architecture details.

## Mandatory Workflow

You MUST follow this workflow exactly. Do not skip steps.

### Phase 1: Clarify Requirements (AskUserQuestion — HEAVY USE)

Before ANY research or planning, you MUST use **AskUserQuestion** to deeply understand what the user wants. Ask about:
- **Scope**: What exactly should this feature do? What should it NOT do?
- **UX/Gameplay**: How does the player interact with it? What keys/UI/feedback?
- **Integration**: Should this tie into existing systems (crafting, XP, talents, stress, hotbar, build mode)?
- **Priority**: What's the MVP vs. nice-to-have?
- **Constraints**: Any performance budgets, visual style requirements, or platform concerns?

Ask **multiple rounds** of questions. Do NOT move to research until you have a clear, detailed picture. Use follow-up questions to resolve any ambiguity. If the user's answer opens new questions, ASK THEM. Aim for at least 2-3 rounds of AskUserQuestion before moving on.

Throughout ALL subsequent phases, continue using **AskUserQuestion** whenever you encounter:
- A design decision with multiple valid approaches
- Uncertainty about how something should look, feel, or behave
- Trade-offs between complexity, performance, and features
- Anything where the user's preference matters

**Default to asking, not assuming.**

### Phase 2: Research (Agents)

After requirements are clear, use the **Task tool** to spawn research agents (subagent_type: `Explore` or `general-purpose`) to investigate:
- Which files and systems are involved
- Existing patterns, types, and interfaces that must be integrated
- Potential conflicts or dependencies

Launch multiple research agents **in parallel** for independent areas. Do NOT start planning until agents report back.

After research, use **AskUserQuestion** again to:
- Present what you found about existing systems
- Ask if the user wants to reuse existing patterns or try something different
- Confirm integration points and any surprising constraints discovered

### Phase 3: Plan (EnterPlanMode)

Use **EnterPlanMode** to create a detailed implementation plan. The plan MUST include:
- A summary of the user's requirements (from Phase 1)
- A summary of research findings (from Phase 2)
- Step-by-step implementation tasks broken down by file/module
- Which tasks can be parallelized using **TeamCreate** and Task agents
- Shared interfaces that must be built first
- A testing/validation strategy
- Open questions flagged with **[ASK USER]** markers for anything still uncertain

Present the plan for approval. If the user has feedback, revise and re-present.

### Phase 4: Execute (Teams + Agents)

Use **TeamCreate** and spawn teammate agents via the **Task tool** to parallelize independent work. For example:
- One agent handles Rust-side implementation
- One agent handles UE C++ implementation
- One agent handles tests or viewer integration

Use **TaskCreate** to build the task list, assign owners, and track progress.

During execution, use **AskUserQuestion** when hitting:
- Implementation choices with multiple valid approaches
- Naming decisions (functions, types, config fields)
- Visual/UX details (colors, sizes, positions, animations)
- Whether to add optional enhancements or keep it minimal

### Phase 5: Review & Validate

After implementation:
1. Use **AskUserQuestion** to ask if the user wants to review changes before testing
2. Run `cargo test --workspace`
3. Run UE CLI build if C++ was changed
4. Ask the user if they want to test in the web viewer or UE editor
5. Commit and push to GitHub
6. Update memory topic files if a major system was added

## Key Reminders
- Cargo: `~/.cargo/bin/cargo.exe` (not in default PATH)
- UE CLI build: see MEMORY.md for the UnrealBuildTool command
- DLL deploy path: `D:/cargo-target/release/voxel_ffi.dll` → ThirdParty dir
- Always check MEMORY.md topic files for existing system docs before designing new features
- HashMap iteration is nondeterministic — sort keys before RNG-dependent processing
- Adding GenerationConfig fields requires updating the explicit struct in `voxel-ffi/src/engine.rs`
- **When in doubt, ASK. Do not assume user preferences.**
