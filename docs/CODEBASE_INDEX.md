# Ripped Shorts — existing-tool indexing pilot

This project uses [codebase-index](https://github.com/denfry/codebase-index), an MIT-licensed, local Python code indexing tool. No new mapping server or production dependency is required.

## Setup in a local or cloud Codex checkout

```sh
python -m pip install "codebase-index[watch,mcp]"
codebase-index init --target codex
codebase-index index
codebase-index doctor
codebase-index search "where does Audio Master handoff enter Ripped Shorts"
```

Review any `init` changes to `AGENTS.md` and retain the committed safety guidance. To keep the index fresh during local edits, run `codebase-index watch --debounce 500` in a separate terminal; after pulling code run `codebase-index update`.

For optional MCP clients, configure the documented `codebase-index mcp --root /absolute/path/to/repo` command. Codex's skill/AGENTS setup is separate from MCP client configuration.

The `.github/workflows/codebase-index.yml` workflow smoke-tests indexing on PRs and pushes to main. CI's index is ephemeral, not a shared service or automatic synchronization to developer workspaces.

## Cross-repository limits
- Audio Master has handoff/ingest runtime modules; inspect the corresponding receiver code here.
- Use the index in **both** checkouts when changing API payloads, endpoints, job state, or delivery semantics.
- The index does not automatically map HTTP endpoints across repositories, external providers, runtime monkey patches, or undocumented contracts.
- Verify actual source and tests before modifying or concluding there is no impact.

## Acceptance
- `index`, `doctor`, and search complete successfully.
- Real receiver symbols return file:line evidence and useful impact paths.
- `update` or `watch` reflects a local code change.
- No changes to production, deployment, publishing, approvals, or persistent data.
