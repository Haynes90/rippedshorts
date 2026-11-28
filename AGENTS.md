# Ripped Shorts — Codex guidance

## Railway operational policy

For enchanting-fascination / production, start at `railway-workflow/AGENTS.md` in the Tech Wizardry workspace. Use application requirements RS-01, RS-02 and SM-01. Preserve artifact and approval identity across intake routes. Keep the customer pilot governed by its own scope.

## Navigate before changing code
- In an indexed checkout, run `codebase-index doctor` and `codebase-index update`; use `search`, `refs`, `impact`, and `diff-impact` to narrow code exploration.
- Verify graph results against source code, tests, and runtime contracts; missing edges do not prove a dependency is absent.
- If the indexer is unavailable, inspect source directly. Never invent index findings.
- For changes to the Audio Master handoff, inspect `Haynes90/audiomaster` independently. This tool does not provide a unified multi-repository graph.

## Change discipline
- For source/provider failures, distinguish acquisition from verification and
  inspect retained file/transcript checkpoints before any retry. Preserve exact
  timing and source revision; reuse the official public metadata and existing
  speech caches before considering new credentials or provider calls. Optional
  notifications must not undo successful machine work or replace human approval.
- Change only verified responsible code; avoid refactors and unrelated edits.
- Preserve retained jobs, existing approvals, publishing pauses, scheduling constraints, and retry/idempotency behavior unless explicitly authorized.
- Run targeted and contract tests. Do not deploy, replay, publish, or change production settings without explicit authorization.

## Freshness
- The CI smoke test builds a temporary index on push/PR; it is not shared with local or cloud Codex workspaces.
- On each local checkout/pull, run `codebase-index update`. For continuous local updates, run `codebase-index watch --debounce 500` with the `[watch]` extra.
- For local and cloud installation, parser preflight, and optional MCP setup, follow `docs/CODEBASE_INDEX.md`.
