# Preview-first prerequisite: retain verified revisions without duplicate uploads

This dependent change closes one Drive-retention gap from the Pass B plan. It does
not start preview jobs, expose a new review UI, migrate historical media or modify
production records. It depends on PR91's persisted boundary specification.

## Implemented

The existing uploader accepts verified clip evidence from both existing render
adapters. Legacy calls retain their existing behavior. A specification-bound output
must have exact request/asset identity, boundary fingerprint, bounds, completed
media checks and a full-decode receipt before this retention path can run.

Existing SQLite evidence records retain an immutable upload intent and success
receipt. The upload key binds specification, source, asset and destination folder.
Output SHA256/MD5/size are frozen; a different encoding for the same identity is
held rather than uploaded as a duplicate. The Drive object carries a hashed
appProperty for authoritative reconciliation. A successful receipt is read back
against Drive size, checksum, folder, MIME type and correlation tag before reuse.

A lost API response or database-commit failure can reconcile the already-created
object without another create. Multiple matches, changed/deleted media, absent
checksums and uncertain empty listings hold for review. Concurrent callers cannot
both claim creation. Read-only listing failures before creation begins are safe to
retry; uncertainty after the creation intent cannot silently replay. No automatic
deletion, public sharing, caption edit, owner approval, invitation or scheduling is
performed. Individual render/upload failures retain the existing sibling isolation.

This reuses an upload; it does not yet skip FFmpeg when a preview qualifies as final.
Historical successful media is not migrated or reuploaded. In particular, an already
completed protected asset continues through its existing success guard unchanged.

## Isolated tests

`tests/test_retained_preview_upload.py` uses actual SQLite and an explicitly mocked
Drive API. It checks exact-revision requirements, one upload/one receipt, successful
readback reuse, response loss, commit failure, concurrent creation, changed output,
multiple remote matches, remote checksum/folder/tag changes and safe pre-create read
retries. The existing uploader is exercised, not only the receipt helper. These are
not claims of live Drive uploads, real preview readiness or completed scheduling.

Executed locally: `python -m pytest -p validation_network -q -ra --tb=short`:
**662 passed, 0 failed, 4 skipped, 2 subtests passed**, 42.04s. The 22 new cases are
included. Four skips require local FFmpeg/FFprobe tools on PATH; GitHub's focused
workflow installs those tools. Tests use isolated SQLite/media and block external
operations; local HTTP integration tests run normally. Live media retention and
website preview-before-selection acceptance remain unperformed.

The first CI run exposed two previously observed optional-intelligence fixture
races: background highlight futures could outlive a test's provider and database
patches. The fixture now waits for its own futures at teardown, while retaining
concurrent execution and every existing safety assertion. No application analysis
code or production worker behavior was changed for this correction.

## Shortest remaining path

1. Resolve PR91's real-speech acceptance and its conservative quiet-interval rule
   using retained source audio and contextual timing. No automatic approved-cut edits.
2. Register verified preview receipts for unselected candidates through the current
   bounded render executor, after their immutable specification is validated. Keep
   preview readiness separate from candidate, finished-media, copy and collaborator
   approvals. This change provides retention receipts; it does not claim step2 done.
3. Extend the existing website cards/media route to these unselected verified previews.
   Reuse the exact receipt on approval before invoking FFmpeg. Caption/creator-only
   edits must not regenerate video. Keep Telegram request-specific navigation only.
4. Add the completion-eligibility timestamp to existing source usage records; normal
   and emergency cleanup must protect shared active/review/edit/held sources. Start
   expiry only after all linked work completes or is explicitly closed. No new storage
   service and no deletion of retained Drive originals/previews/finals.
5. After actual owner final approvals, exercise the existing idempotent Schedule
   Master handoff and independently read destination/date/revision records. Keep
   cadence, audits, reconciliation, public-publishing pause and Agape isolation.

Review branches only. Merge, deployment, production retries, hold release and scheduling
still require separate authorization. No paid infrastructure or analysis stage added.
