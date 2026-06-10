# Voxel Backend

## Build & Test
```
export PATH="$HOME/.cargo/bin:$PATH"
cargo build --workspace          # dev build
cargo build --release -p voxel-viewer  # release viewer
cargo test --workspace           # run all tests
```

## Git Backup
- **GitHub CLI** path: `/c/Program Files/GitHub CLI` (must export to PATH)
- After completing any significant feature or change, commit and push to origin/main
- Use descriptive commit messages summarizing the "why"
- Always run `cargo test --workspace` before committing

## Multi-agent protocol
- Multiple Claude sessions run in parallel: work in your **own worktree** (`git worktree add ../voxel-backend-<name> -b <branch>`), never in someone else's.
- Sharing `D:\cargo-target` across worktrees is fine — cargo queues on its target-dir lock ("Blocking: waiting for file lock" is normal). Different branches thrash incremental caches (extra rebuilds, never corruption).
- Anything that touches the UE editor or UBT builds (e.g. verifying a voxel_ffi.dll change in-game) requires the machine-wide UE lease — a hook enforces it. Full protocol in `D:\Unreal Projects\Mithril2026\CLAUDE.md` ("Multi-agent protocol"); acquire via `powershell -ExecutionPolicy Bypass -File "$env:USERPROFILE\.claude\scripts\ue-lease.ps1" acquire ...`.
- DLL sync goes to the **UE worktree you are verifying in** (both copies), not blindly to the main checkout.

## Architecture
- 9-crate workspace: voxel-noise, voxel-core, voxel-gen, voxel-cli, voxel-viewer, voxel-ffi, voxel-fluid, voxel-sleep, voxel-path (+ nav-debug helper)
- Viewer runs on localhost:8080, static files embedded at compile time
- Kill old viewer before rebuilding: `taskkill //F //IM voxel-viewer.exe`
