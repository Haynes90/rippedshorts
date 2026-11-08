# Pass B: finish the existing media workflow

This review branch closes the first milestone: persist and enforce a source-bound
speech specification before the existing TCB/TDOG selected-asset renderer. It does
not deploy, restore selections, release holds, queue previews or schedule content.
The deployed Pass A baseline and its completed media remain protected.

## Existing behavior and exact gaps

| Requirement | Existing code to reuse | Gap / smallest change |
|---|---|---|
| Episode understanding, fallbacks, rubrics and learning | `hierarchical_intelligence.py`, `intelligence_workflow.py`, `production_intelligence.py`, selection specifications | Working foundation; no changes in this branch. Do not rerun an episode to test rendering. |
| Stable selected assets and independent retries | `render_recovery.ticket`, immutable records, request-scoped guards | Preserve all collections, identities and completed siblings. This branch adds evidence inside the guarded render, without authorizing a retry. |
| Sentence/thought assessment | `clip_completion.review_thought`, verified source timing | Previously lacked persisted source-bound evidence and surrounding setup/payoff context. Extend the existing reviewer, not the analysis hierarchy. |
| Acoustic boundary evidence before FFmpeg | Existing FFmpeg source/audio tools and timed transcript | Measure quiet near both exact cuts, label uncertainty and propose measured alternatives without applying them. Unresolved evidence holds only the asset. |
| Immutable specification | Existing SQLite `intelligence_records` and `source_analyses` | Add `boundary_specification.py`; bind request, asset, approved revisions, source SHA, transcript digest, original references, intended passage, semantic evidence, acoustic evidence and rendering configuration. No new storage service or schema. |
| Exact renderer consumption | `clip_completion.render_complete_clip`, existing two render wrappers | Validate saved fingerprints, live request/review revisions, source bytes and exact bounds before calling FFmpeg. Never silently clamp a saved cut. |
| Independent output verification | Existing duration, stream and aspect checks | For the specification path, compare exported opening/ending PCM against the exact source interval without lag searching, fully decode, and measure audio. This replaces a second ASR spelling vote only on this path. Legacy calls retain their checks. |
| Preview before creator selection | Existing render executor, Drive upload and website cards | Still missing: current rendering starts after selection. Next extend these components to track preview readiness separately from owner selection and final approval, reuse unchanged verified video, and notify through Telegram navigation. Do not create another review interface. |
| Retry-safe Drive reuse | `main.upload_clip_to_drive`, `list_existing_rendered_assets` | Upload currently creates a file; filename/newest matching is insufficient for revision-specific reuse. Bind existing receipts to specification/configuration/file identity and reconcile uncertain outcomes before uploading again. Rejection must not delete Drive previews. |
| Completion-based local expiry | Existing source protection and normal/emergency cleanup | Active/held source protection exists; mtime-based expiry is not a durable completion-eligibility clock. Extend existing records with eligibility time only after all linked work is complete/closed, clearing eligibility when new dependent work appears. |
| Final scheduling | `telegram_intake._handoff_shorts_to_schedule_master`, existing outbox, `batch_progress.summarize`, `workflow_test_status.project` | Existing gates/receipts/readback evidence are the foundation. Acceptance must compare each actual destination/date/asset revision against authoritative Schedule Master records. An API acknowledgment alone is insufficient. No Schedule Master changes proposed here. |

## First milestone implemented here

The existing explicit TCB/TDOG selected-render path calls `prepare_for_render` before
attachment/export. This is deliberately not a migration or a preview queue.
Previously successful media is reused by the existing guards, and unchanged held
assets remain held. The current six-clip batch is never modified by these tests.

The specification includes original segment/word indices and speaker evidence,
the intended excerpt, subject and rationale, contextual setup/payoff references,
semantic completeness, source-edge measurements, timing precision, immutable input
and review revisions, source/transcript fingerprints and render configuration.
Segment-timed units remain whole segments, never invented word timing. Punctuation
and ASR spelling alone are not acoustic proof. A necessary boundary change is a
proposal requiring explicit review; this branch does not apply it.

The existing semantic reviewer is called once per new immutable input; cached held
or failed evidence prevents identical repeated provider calls. Successful and held
assessments are durable. Essential early source/timing failures also persist through
the existing guarded per-asset failure records. The renderer rechecks current
request identity and approvals immediately before export. Caption/creator decisions
are neither rewritten nor interpreted as media-edit permission.

The acoustic rule is conservative: quiet must span the chosen cut with measurable
uncertainty allowance. Continuous/overlapping speech without a defensible cut may
need human review; this implementation does not claim phoneme precision. The
surrounding semantic context is bounded and may also produce a review hold rather
than guessing about distant setup. Source/export PCM correlation is evidence that
the chosen passage survived export, not an independent proof of its editorial merit.

## Completion order and dependencies

1. Review this boundary-contract change and isolated regression evidence. After
   separate authorization, perform source-only assessment of the five existing holds
   and obtain explicit approval for any necessary media revision. Preserve completed
   media; do not dispatch unchanged held inputs.
2. Extend the same renderer/records/Drive receipts for preview-first processing.
   Both formats may render within the existing resource bound. One failed preview
   cannot block healthy siblings. Keep preview readiness, selection, final media,
   copy, collaborators and scheduling approvals distinct. Use the existing website;
   Telegram provides request-specific status/navigation. Keep all previews in Drive
   until an explicitly approved retention/deletion policy permits otherwise.
3. Extend existing source cleanup with shared-use and completion-eligibility records.
   Preserve active/review/edit/hold sources in both cleanup modes. Start the existing
   expiry interval from durable eligibility, not download time. This small independent
   storage change can be tested alongside preview work; it is not a prerequisite for
   assessing an individual retained clip.
4. Exercise final website approval and the existing idempotent handoff, then read
   back actual Schedule Master rows/receipts and configured cadence. Preserve the
   12-hour audit, 300-second reconciliation, interleaving and separate highlight rules.
   No new scheduler or dates. Public publishing and Agape stay disabled throughout.

No new paid service, database, model stage or worker is needed. Each next change must
close one of these acceptance gaps, rather than reopen successful intelligence work.

## Tests and real acceptance

`tests/test_boundary_specification.py` exercises actual SQLite persistence and exact
specification consumption; stale source/review/transcript/config changes; explicit
render wrappers in both formats; Q&A/context and payoff holds; malformed timing;
overlap/speaker preservation; segment fallback; independent asset failures; cached
deterministic failures; and no arbitrary highlight maximum (minimum remains 180s).
Provider responses in behavioral fixtures are explicitly injected.

Real local FFmpeg/AAC tests measure source quiet, fully decode an export, compare
its audio and reject a shifted passage. Additional FFprobe-backed tests run the
persisted contract through actual vertical and >=180-second horizontal exports.
Synthetic signals establish media mechanics, not real TDOG speech or framing quality.
GitHub's existing focused workflow installs FFmpeg/FFprobe for these tests.

Local execution (2026-11-07):

- `python -m pytest -p validation_network tests/test_boundary_specification.py -q -ra --tb=short`: **33 passed, 0 failed, 2 skipped** (FFprobe absent).
- `python -m pytest -p validation_network -q -ra --tb=short`: **640 passed, 0 failed, 4 skipped, 2 subtests passed** in 117.78s. Two existing stored-render and two new contract/export cases require local FFprobe/FFmpeg tools on PATH. The new direct AAC/PCM tests did execute using the existing bundled FFmpeg binary.
- Both runs used isolated temporary SQLite/media directories; the validation network
  plugin blocked external operations. The full suite permitted localhost HTTP tests.
  Four existing dependency/lifespan deprecation warnings remain.

An initial wrapper test omitted its failure-notification mock; that test fixture
was corrected. An earlier sandboxed combined run stalled in localhost tests and was
stopped, then superseded by the completed isolated full run above. No production
credentials or APIs were used. GitHub results must be checked on the final commit;
these local results alone do not establish deployment or real-speech readiness.

Actual TDOG acceptance remains separate: inspect/listen to retained context at both
edges, verify necessary question/setup/payoff, review proposed revisions with the
owner, then authorize only specific eligible renders. Check real whole-tile/safe
blurred framing, audible speech, sync, decode, dimensions and authenticated website
playback. No old deleted failed export is claimed playable. The successful Pass A
Short must remain unchanged and visible. No production acceptance was run here.

Preview acceptance must additionally demonstrate retained Drive identities, website
playback before selection, independent failures, immediate resumable approvals,
unchanged-media reuse and no video rerender for copy/collaborator-only edits.
Storage tests must cover >48h and seven-day active sources, shared users, completion
clock reset, normal/emergency cleanup and unchanged Drive media. Scheduling acceptance
requires owner-approved finished revisions and actual destination/date readback,
duplicate/uncertain receipt reconciliation, and publishing still paused.

## Storage and rollout constraints

This milestone adds small evidence records and hashes existing source bytes. It does
not duplicate source media, upload files or generate production exports. PCM scans
are bounded and transient. Later preview storage grows with actual candidate count
and duration; measure retained Drive media and Railway cache/intermediates before
considering larger volumes. Existing protections remain in force until the durable
eligibility-clock change is reviewed. Never reclaim unresolved source or Drive media.

Deployment, six-clip changes, render retries, hold release and scheduling are not
authorized by this branch. Before any later rollout, verify tested SHA/CI, backups,
persistent storage, disabled auto-deployment, one worker/no overlap, protected-record
fingerprints and publishing/Agape gates. Production readiness requires observed media,
website and scheduler evidence; unit tests alone cannot establish it.
