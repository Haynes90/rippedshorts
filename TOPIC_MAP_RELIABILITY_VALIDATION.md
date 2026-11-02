# Topic-map reliability and watchdog hold — review handoff

## Scope and production boundary

Base: [9a9d3effe1213eb8be9edfed73c1f050b5fa8e72](https://github.com/Haynes90/rippedshorts/commit/9a9d3effe1213eb8be9edfed73c1f050b5fa8e72).
Development and isolated tests only. No production deployment, request resume,
provider request, Telegram notification, intake, queue edit or scheduling action
was performed during this correction. No publishing or Agape setting was changed.

Protected request: `8999063e-bbc7-4000-bba4-e2490ca27135`.
Its verified source and 1,189 timed segments remain the future recovery inputs.
The five older selections, approvals, learning records and receipts are untouched.
Production configuration was not re-audited in this development-only pass.

## Confirmed failure and correction

The previously retained diagnostic identifies topic position 12 (topic 13):
`first_segment=1161`, `last_segment=1120`, transcript bounds `0..1188`.
These independently generated endpoints cannot form a valid inclusive interval.
This pass does not reinterpret, swap or clamp that failed result.

The new `semantic-boundaries-v4` provider contract requests semantic START
positions in depth-first chapter/child order, not independent topic ends.
Every start must be an actual integer transcript index; siblings must increase
strictly, the first root must start at zero, and parents must precede children.
The application derives an inclusive end from the next sibling start minus one,
or the enclosing parent's end; the final root ends at the final verified segment.
Invalid sequences, closed-parent references, out-of-parent children, duplicate
keys and unsolicited provider ends fail locally. No sorting or repair occurs.

The model still supplies subjects, transitions, semantic titles/summaries,
hierarchy, related topics, subtopics, questions/responses and memorable moments.
Nested moment intervals remain explicit provider evidence and must fit the
derived parent. Their existing typed diagnostics and local validation remain.
Structural coverage does not prove semantic quality or acoustically safe clips;
the selection rubrics, speech checks and human reviews remain necessary.

Successful legacy analyses are reused. A changed generation recipe uses a
separate analysis key under the same stable source-map identity. Previous failed
rows and exhausted retry budgets are preserved, not reset. The request must
explicitly identify the reviewed new key during a later authorized recovery.
Both selection lanes retain their shared map and existing learning inputs.

Deterministic map failures now preserve sanitized diagnostics in the existing
failed-analysis record and produce an explicit review hold, not a generic
retryable error or misleading independent-format continuation. Cached failures
retain their validation classification. The watchdog skips held maps, including
legacy failed map records, before any worker submission or retry-counter change.
A held newest source also prevents revival of an older duplicate request.
An `analysis_recovery.mode=owner_bounded` marker disables watchdog resubmission
while a specifically authorized recovery is running or has failed. Only a
reviewed explicit state transition may clear it. Operational errors remain
distinct; this change does not add automatic model retries for network failures.

One regression run demonstrated a pre-existing learning timestamp tie: approval
and rejection had exactly the same stored timestamp, and the older approval won.
The query now uses insertion order as the tie-break; a dedicated equal-time
test retains the intended latest-decision behavior without changing learning
weights or selection standards.

## Executed validation

Final complete suite, isolated SQLite/storage, external networking blocked:

```powershell
$env:PYTHONUTF8='1'
$env:TEMP=(Resolve-Path ../ripped-intelligence/.test-tmp).Path
$env:TMP=$env:TEMP
$env:DATA_DIR="$env:TEMP/semantic-boundary-release-data"
$env:PYTHONPATH="$(Resolve-Path ../ripped-intelligence/.test-tmp/opencv4);$pwd;$(Resolve-Path ../pr-validation-handoff)"
& ../rippedshorts/.venv/Scripts/python.exe -m pytest -p validation_network tests -q -rs --basetemp=../ripped-intelligence/.test-tmp/semantic-boundary-release --junitxml=../ripped-intelligence/.test-tmp/semantic-boundary-release.xml
```

**480 passed, 0 failed, 2 skipped, 2 subtests passed, 4 framework deprecation
warnings; 16.14 seconds.** The two skips are FFmpeg-dependent stored-render tests
at lines 142 and 176; FFmpeg/ffprobe are not on this local test PATH. Existing
GitHub render CI installs FFmpeg and runs those media tests. No new real-media
render or real provider analysis was attempted in this development pass.

Earlier focused run: 94 passed. An initial harness run had four missing isolated
watchdog environment dependencies, corrected in the harness. The next complete
run exposed the genuine timestamp-tie failure above (478 passed, 1 failed,
2 skipped); it was fixed, not removed. The final full run is the result reported
above. All modified Python files also passed AST parsing.

Added tests execute the actual watchdog function with isolated records and a
mock executor, rather than merely asserting source text. Coverage includes:

- Valid unequal semantic chapter lengths, nested hierarchy and preserved moments.
- Invalid/reversed/duplicate/non-integer starts, missing beginning, closed parent,
  child beyond parent, provider endpoint rejection and nested reference failures.
- Full coverage of a synthetic 1,189-segment fixture, including starts 1120 and
  1161, without inventing transcript text or claiming real semantic analysis.
- Durable sanitized failures, cached rejection without another provider call,
  malformed JSON, missing generation contract and operational provider timeout.
- Repeated watchdog scans, bounded recovery in flight, one explicit retry followed
  by a renewed hold, no older duplicate revival, and retained operational limits.
- Successful legacy reuse and a separate new-recipe key retaining all old failure
  history; both candidate formats still receive map/learning provenance.
- Equal-time feedback events selecting the later recorded decision.

The existing GitHub topic-contract job is extended to execute the new tests plus
the existing intelligence, reference and retry-loop regressions. GitHub results
must be checked at the PR head before any future merge authorization.

## Proposed bounded recovery — NOT EXECUTED

1. Review this PR and its final checks. Obtain separate merge, paused deployment
   and single-request recovery authorization. Keep publishing and Agape disabled.
2. Back up the persistent database/request state and verify protected-record
   fingerprints. Stop the old worker before replacing it; preserve storage and
   credentials. Deploy Ripped Shorts only after authorization and verify SHA,
   health and storage before any recovery.
3. Read-only verify the existing source video ID, complete retained media,
   duration/streams/integrity and the existing 1,189 valid timed segments. Stop
   on missing media, authentication errors or changed source/transcript identity.
   Do not download or transcribe again.
4. For the SAME request only, record the explicit recovery authorization and
   `analysis_recovery={mode: owner_bounded, attempt: 1}`. Select the new key
   `<existing-map-id>:semantic-boundaries-v4` in `source_intelligence.analysis_id`;
   preserve its `map_id`. Clear only this request's reviewed validation hold.
   Do not call the old broad retry-budget reset or reopen historical failed rows.
   If the new key already failed or ran, stop and reconcile; do not replay it.
5. Dispatch the existing resume path exactly once. The model has one map-generation
   attempt. On any invalid sequence/reference, persist diagnostics and hold;
   watchdog scans must not dispatch another attempt. Do not regenerate a failed
   map repeatedly. Preserve successful media and transcript checkpoints.
6. If valid, allow the existing independent 9:16 and 16:9 selectors to consume
   the map and learning context. Verify their actual candidate counts, provenance
   and persistence; present Shorts before highlights and wait for owner selections.
   Keep the bounded watchdog guard until an explicit reviewed state transition.
   No automatic clip choices, scheduling, publication or collaborator invitations.

## Remaining acceptance

**PASS — isolated structural, failure-persistence, watchdog and regression tests.**
**NOT VERIFIED — actual provider semantic quality and real request candidate
generation with the new contract.** Those require the separately authorized
paused deployment and bounded recovery. No new candidates or finished clips are
claimed here. Review approval is the next required owner action.
