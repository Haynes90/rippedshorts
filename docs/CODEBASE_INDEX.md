# Ripped Shorts — existing-tool indexing pilot

This project uses [codebase-index](https://github.com/denfry/codebase-index), an MIT-licensed, local Python code indexing tool. No new mapping server or production dependency is required.

## Local and cloud Codex setup

Use Python 3.11+ in a virtual environment inside the development workspace. Activate the environment and run these commands from the checkout. For cloud Codex, install and build the index inside that cloud workspace.

```sh
python -m pip install "codebase-index[watch,mcp]==2.1.3"
python -c "from tree_sitter_language_pack import cache_dir, get_parser; get_parser('python'); print(cache_dir())"
codebase-index index
codebase-index doctor
codebase-index search "Audio Master handoff"
codebase-index refs normalize_sermon
codebase-index impact normalize_sermon --direction up
```

The parser preflight loads Python through the indexer's Tree-sitter dependency. A cold cache needs access to the language pack's GitHub release assets. For offline use, provision the dependency's cache for the same OS, architecture, and package version and confirm the preflight succeeds in that workspace. Use the path printed by `cache_dir()`. On Linux, set `XDG_CACHE_HOME` consistently for preflight, indexing, and the watcher if you use a dedicated cache. A successful build can still fall back to text search when parsing fails. Check that the build reports zero parse failures and that `doctor` reports working symbol extraction.

Codex can invoke the local CLI in the workspace shell. Optional `codebase-index init --target codex` scaffolds CLI guidance, so review its changes and preserve this repository's committed safety rules. Native MCP requires a separate stdio registration in an authorized Codex client. For Codex CLI, use a distinct server name and the absolute executable and checkout paths.

```sh
codex mcp add rippedshorts-index -- /absolute/path/to/codebase-index mcp --root /absolute/path/to/checkout
codex mcp list
```

Open a new Codex session and verify a native `search_code` call. CLI output and a listed registration alone do not establish that Codex can call the server.

For automatic refresh while editing, keep `codebase-index watch --debounce 500` running in that checkout. Use `codebase-index update` after pulling or switching branches when a watcher is unavailable. The CI index is temporary and is built separately from local and cloud indexes.

## Cross-repository limits
- Audio Master has handoff/ingest runtime modules; inspect the corresponding receiver code here.
- Use the index in **both** checkouts when changing API payloads, endpoints, job state, or delivery semantics.
- The index does not automatically map HTTP endpoints across repositories, external providers, runtime monkey patches, or undocumented contracts.
- Verify actual source and tests before modifying or concluding there is no impact.

## Acceptance
- The build reports zero parse failures and `doctor` confirms symbol extraction and an ignored cache.
- Search locates the handoff module and the index confirms `process_handoff` calls `normalize_sermon`.
- Both `update` and `watch` detect temporary edits. CI restores the file and verifies that the probe symbols disappear.
- No changes to production, deployment, publishing, approvals, or persistent data.
