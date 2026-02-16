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

## Architecture
- 5-crate workspace: voxel-noise, voxel-core, voxel-gen, voxel-cli, voxel-viewer
- Viewer runs on localhost:8080, static files embedded at compile time
- Kill old viewer before rebuilding: `taskkill //F //IM voxel-viewer.exe`
