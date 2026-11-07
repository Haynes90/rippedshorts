# Issue 89: preserve selected assets during render recovery

## Confirmed defects

The missing-render action redirected by source-video identity, queried video-wide
Sheets approvals, and replaced the request's complete candidate and review lists
with unfinished historical Shorts. That removed a successful sibling from review,
imported approvals from another editorial pass, and omitted independent highlights.
The render workers also committed by list position, and notification failures after
a successful commit could downgrade the completed asset to failed.

## Correction

* `render_recovery.py` uses the existing SQLite evidence and durable-lease tables.
  It freezes request/asset/candidate/source/approved-input revisions before queueing,
  checks them at execution and commit, and writes immutable selection, attempt,
  hold and successful-media receipts. No new worker, database or external service.
* Retries inspect only the named request and owning chat. They preserve complete
  selection collections, completed siblings, copy, creator decisions and scheduling
  records. Both formats recover independently. Sheets is not a retry authority.
* Unchanged deterministic boundary/framing failures stay held. Only a relevant
  explicit approved correction can open a new validation revision. Unknown upload
  or commit outcomes and abandoned RUNNING leases require reconciliation, including
  after lease expiry. Only positively classified pre-render source-access failures
  are automatically eligible for an ordinary retry.
* The watchdog does not consume retry budgets or resubmit work when all selected
  assets are completed or held. Repeated Approve/Approve All preserves failed and
  completed review metadata.
* Restoration is a separate, operator-only operation, never invoked by retry,
  startup or watchdog. `legacy_selection_receipt` requires an explicitly ordered
  manifest of exact request-scoped fingerprinted approval/attempt records, matching
  PASSED candidate-cache records and approved bounds. `restore_from_receipt`
  additionally requires independent successful-media verification and refuses
  submitted assets, source mismatches and concurrent changes. It preserves current
  caption, creator, final-review and scheduler fields. Incomplete evidence holds.
* `clip_completion.py` labels ASR-only mismatches as unconfirmed alignment evidence
  and retains specific speech-proximity diagnostics. Acceptance thresholds, semantic
  checks, acoustic checks and failed-export exclusion are unchanged.

## Isolated acceptance evidence

The behavioral fixture contains four selected Shorts and two highlights, one
successful export, five validation holds, another request for the same source and
older unfinished approvals. Tests execute the real SQLite, queue, decorated render,
restoration and authenticated review HTTP paths with external services mocked.
They cover duplicate/concurrent execution, changed identities/revisions, lost-worker
uncertainty, unrelated framing approval, partial failure, late notification failure,
legacy reconstruction, retained media visibility, playback routing and final-review
controls without scheduling.

Executed from the repository snapshot with the existing virtual environment and a
validation-only external-network blocker:

```text
python -m pytest -p validation_network tests -q -ra --tb=short
598 passed, 0 failed, 2 skipped, 2 subtests passed (27.61 seconds)
```

The two local skips require FFmpeg/FFprobe on PATH. The GitHub stored-render job
installs these tools; its results must be checked at the PR head. Deprecation warnings
are existing FastAPI/Starlette notices. This is not a claim of production recovery.

Legacy text assertions requiring Sheets-based replacement and request redirection
were replaced by executable exact-request/six-selection preservation checks. Bounded
executor and idempotency assertions remain, plus new duplicate-execution tests.

## Actual media findings and limits

Read-only inspection independently recovered the successful export's receipt from
the existing request-scoped feedback ledger. Its retained Drive object exists,
matches its stored size and MD5, fully decodes, has the intended vertical dimensions
and non-silent stereo audio. It was not rerendered or registered back into production.
Its absent website tile is a selection-state defect, not a missing-media finding.

The five held assets include two ASR end-of-speech proximity failures and three
boundary-token alignment failures. Source-only acoustic measurements show quiet
intervals at some disputed edges; this does not establish final-export integrity or
justify changing approved boundaries. Fillers and proper-name recognition differences
alone do not prove an unsafe cut. Old rejected exports were deleted by the existing
fail-closed renderer, limiting retrospective source/export comparisons. Each affected
asset remains held for specific acoustic/editorial review. No shared FFmpeg or
source-access failure was established for the five original holds.

## Proposed bounded recovery — separate authorization required

1. Review this draft and passing head checks. Keep publishing paused and Agape
   disabled. No deployment or render recovery has been performed by this PR.
2. If deployment is authorized, back up the exact request, source fingerprint,
   approvals, feedback, receipts and schedules; stop the old worker before replacement.
3. Prepare and review the ordered original-six manifest from immutable request-scoped
   evidence and PASSED candidate caches. Do not include historical same-source assets.
   Stop on missing, ambiguous or contradictory approval evidence.
4. Independently verify retained successful media; apply the explicit restoration
   only if separately authorized. Verify the six identities/order and existing copy,
   collaborator, approval and schedule fingerprints. Reuse the successful export.
5. Review each held boundary against actual speech. Do not infer unsafe speech from
   ASR spelling differences. Any boundary/framing change requires the existing owner
   review and a new approved revision. Unchanged failures remain held.
6. Authorize only specifically eligible asset/revision tickets, at most one bounded
   attempt each. Reconcile uncertain upload/commit results before any replay. Preserve
   all successful siblings and record individual results.
7. Verify finished playback, audio and final owner approvals through the existing
   website. No automatic scheduling, invitations, publishing or release of other holds.
