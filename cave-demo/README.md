# Mithril cave demo — browser/WASM edition

The public cave generator demo at https://cave.playmithril.com. The whole
generator runs in the visitor's browser (WASM in a Web Worker); Cloudflare
only serves static files. No server, no shared state: every visitor gets
their own cave, and mining/Dormancy never block anyone else.

## Layout

- `site/` — everything that gets deployed
  - `index.html`, `app.js`, `style.css`, `logo.png` — forked from
    `voxel-viewer/src/static/` (transport swapped from fetch to a Web Worker;
    see the "WASM transport" block near the top of `app.js`)
  - `worker.js` — Web Worker that owns the WASM generator
  - `pkg/` — wasm-pack output (committed so deploys don't need a Rust toolchain)
- `wrangler.jsonc` — Cloudflare Worker config (`mithril-cave`, assets-only,
  custom domain cave.playmithril.com)

## Rebuild the WASM (after Rust changes)

```bash
cd voxel-wasm
CARGO_HOME=/d/cargo RUSTUP_HOME=/d/rustup CARGO_TARGET_DIR=/d/cargo-target \
  wasm-pack build --release --target no-modules --out-dir ../cave-demo/site/pkg
rm ../cave-demo/site/pkg/.gitignore
```

(wasm-pack was installed via `npm install -g wasm-pack`.)

## Deploy

```bash
cd cave-demo
npx wrangler deploy
```

Uses the machine's wrangler OAuth login (agenticgames@gmail.com account).

## Keeping the frontend in sync

`voxel-viewer/src/static/` is the dev tool and stays fetch-based;
`site/` is the public demo and is worker-based. If the dev viewer's
frontend changes meaningfully, re-copy the files and re-apply the two
demo-side changes: the WASM transport block in `app.js` (plus the
`fetch(apiUrl(` → `apiFetch(apiUrl(` rename) and the relative asset
paths in `index.html`.
