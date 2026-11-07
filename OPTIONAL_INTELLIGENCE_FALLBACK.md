# Optional topic metadata — completion-focused correction

Base/deployed code: `55b64b5c19dc87474e8c90106fd4a2e083a4eb06`.
Development and isolated tests only. No production browser/API calls, merge,
deployment, recovery, intake, acquisition, transcription, scheduling, invitations
or publishing. No production records, credentials or Agape settings modified.

## Confirmed defect and correction

The previous authorized TDOG run held on conversation3, topic_position5,
SELF_REFERENCE. Fifteen topic identities were available; the rejected link and
its own topic had the same safe digest. This supplementary relationship failure
was previously coupled to the chronological-topic checkpoint. The rejected
provider response was not retained as accepted spans: this correction does not
pretend those proposed topics can be recovered from an accepted checkpoint.

`hierarchical_intelligence.py` now saves accepted chronological spans FIRST.
The same provider response's proposed relationships are saved privately under
`unverified_relationships`, outside accepted topic fields. A local independently
cached `topic_links` validation then checks the complete verified key set. It
requires no additional model call. SELF_REFERENCE, UNKNOWN_TARGET, invalid type,
malformed list and duplicate links retain specific sanitized diagnostics and a
FAILED link record. Temporal spans remain PASSED even if links fail. No link is
renamed, substituted, fabricated or labelled valid after rejection.

Optional malformed classification/tags are explicitly rejected with safe field,
type/digest and position evidence in the persisted core/aggregate records. They
are omitted from the accepted metadata view, not replaced with invented values.
The rejected links remain durably distinguishable from the topics-only view.

## Bounded fallback paths in the existing pipeline

Preferred path is full-hierarchy. The existing metadata checkpoints/provider calls
remain bounded by their immutable input cache; the only extra checkpoint is local
link validation, justified by the demonstrated coupling defect. No new provider,
worker, database, service or competing analysis system is introduced.

| Available evidence | Candidate provenance path | Behavior |
| --- | --- | --- |
| Complete validated hierarchy/moments | full-hierarchy | Normal preferred selection |
| Valid topics, rejected/unavailable supplementary links/classification/moments | topics-only | Retain topics; exclude unverified metadata explicitly; selectors use full transcript |
| Verified conversations, one or more detailed topic failures | conversations-only | Retain successful sibling topics and complete conversation coverage; selectors discover details from actual transcript |
| Episode/conversation metadata unavailable or invalid | transcript-direct | No fabricated topic IDs/associations; existing selectors receive the full normalized transcript and approved rubrics |

The complete verified transcript is always supplied to BOTH existing selectors.
Discovery windows are input/context hints, never proposed clip intervals: one
whole-episode window guarantees coverage, conversation windows overlap their
neighbors, and actual question cues include surrounding response context. They
are original segment indices; punctuation is not acoustic evidence. No arbitrary
fixed-duration clipping or forced candidate count is used.

A deterministic optional failure is cached and recorded once, with the selected
fallback path in the compiled map and every candidate's provenance. Re-entry
reuses successful/degraded results and does not retry failed optional stages.
Known provider operational failures store type/status/category without raw error
bodies. Timeouts may degrade metadata; authentication401/403, storage faults,
invalid transcript identity/order and unexpected software failures still stop.
The watchdog and owner-bounded recovery protections are unchanged.

## Five-checkpoint reuse

Aggregate recipe: `hierarchical-source-v4`. New uncompleted topic cores use
`chronological-spans-v3`; local links use `topic-links-v1`. Episode, conversation
and conversation-link recipes and key inputs remain unchanged. Exact-input PASSED
legacy topic versions `hierarchical-source-v1` and `chronological-topics-v2` are
reused under their original IDs. Failed legacy rows/budgets are never reopened.

The isolated recovery test seeds exactly five passed checkpoints and an old
conversation3 failure. Corrected execution calls ONLY topics3, topics4 and moments;
all earlier rows (including failure diagnostics) remain byte-for-byte identical.
A bad new topic3 link becomes an optional failure while its accepted temporal
spans and topic4 survive. No new episode, conversation or model relationship pass.

## Existing selectors, review and safety

`intelligence_workflow.py` retains concurrent format analysis, actual approved
rubrics, scoped learning, semantic/sentence checks, candidate recording and cached
per-format results. Both format prompts contain the actual available map, path,
feedback and full transcript. Shorts explicitly farm complete explanations,
questions/answers and humorous exchanges across chapter boundaries. Highlights
retain coherent focused arcs and minimum180s; no new maximum is imposed.

A confirmed existing validator cap `valid[:20]` contradicted the all-qualified
TCB/TDOG discovery contract. The intelligence path now passes maximum=None through
both validation calls; unrelated legacy callers retain maximum20. A25-candidate
regression proves qualified results are not silently discarded. No boundary,
quality, approval or acoustic condition was loosened.

Malformed Shorts provider JSON is converted to a sanitized format-validation
failure. A valid cached highlight sibling can reach review without repeating the
failed Shorts call or selecting anything. Shared authentication failures are not
misclassified as editorial rejection. A failed highlight leaves successful Shorts
available. Per-format failures remain explicit and owner-retry gated.

The existing Telegram navigation/outbox, website session and creator controls are
reused. The website's existing recommendation evidence now includes subject,
excerpt, reason and provenance; no website redesign or new approval flow.
`_send_candidates` can deliver the valid sibling's link when Shorts are held.
Selection, finished media, copy/creators and final scheduling remain separate.
No Schedule Master code or routing changes. Rendering/framing/acoustic checks
are unchanged and still required before any finished-media acceptance.

## Isolated acceptance evidence (not actual TDOG results)

The new complete-episode fixture has four conversations, returning tomato/storage
subjects, practical explanations, question/answer exchanges, humor and an
incomplete final sentence. It executes actual OpenAI request builders/response
parsers with mocked HTTP transport, not mocked candidate counts. Actual Shorts
sentence/transcript validation, highlight minimum duration construction, learning
ranking, both selector orchestrators and SQLite candidate persistence execute.
External media preflight is isolated/mocked; no fixture is called real acoustic
or rendered-video acceptance.

Cases include self/unknown/malformed/missing relationships, invalid detailed
references, classification failures, missing/invalid conversation or episode
metadata, moments failure and provider timeout. Every eligible fallback reaches
both selectors; unsupported final speech and a120s highlight are rejected.
Full-source moments and Q&A topics are retained on the primary path. Scoped
feedback changes ranking, approved historical examples remain eligible, and
candidate IDs/provenance are stable with no duplicate rows or provider replay.
Direct candidates have no invented topic associations. Acoustic complete-thought
failure still stops rendering even with transcript-direct provenance.

A real local HTTP test serves the existing review website, using the actual
request session/outbox code and a mocked Telegram acknowledgment. It checks the
secure request-specific button, saved receipt, resumability, no duplicate send,
subject/excerpt/reason/path, unchanged request decisions and409 on premature
Finish & Schedule. This proves the isolated contract only: no real Telegram send,
live website session, Drive media, Google Sheets readback or actual TDOG clips.
The fixture feedback ledger marks synced revisions explicitly; that is not a
claim of a new production Sheets synchronization check.

## Proposed single production recovery — requires separate approval

1. Review the draft PR, exact head, complete regression and current GitHub checks.
   Obtain separate merge/paused Ripped Shorts-only deployment and one-recovery
   authorization. No production action is authorized by this development report.
2. Verify current DB/approval/queue/receipt backups, auto-deploy disabled, persistent
   /data, unchanged credentials, one worker with stopped old deployment before
   replacement, and disabled global/Agape publishing. Stop on any failed safeguard.
3. Verify existing request8999063e-bbc7-4000-bba4-e2490ca27135, source9mW0bhmUaw0,
   retained media checksum/streams/duration, and all1,189 timed segments. Verify
   exact immutable IDs/fingerprints for episode, conversations, conversation links,
   first-conversation legacy topics and second-conversation chronological topics.
   Stop if evidence/media is missing; no new intake/download/transcription.
4. After approved deployment and health/protected-record verification, record one
   distinct owner_bounded revision targeting hierarchical-source-v4. Preserve all
   earlier authorizations, failures and retry budgets. Dispatch the SAME request
   once. Five existing passes must be verified cache hits.
5. Topics3/4 use the new temporal core key. Save spans before local links. Record
   optional failures and proceed via A/B/C as appropriate; do not retry failed
   metadata within the run. Essential source/auth/storage/boundary failure holds
   affected work. Both selectors remain independent and cached.
6. Inspect actual topic quality, candidate counts, reasons, complete-thought
   boundaries, map/path/rubric/prompt/feedback evidence. Previously worthwhile
   moments may recur; no forced novelty or quota. Verify both real format results,
   then the existing Telegram website link and actual delivery receipt. Present
   Shorts before highlights when both succeed; identify a held format accurately.
7. Stop for owner selection. Rendering, acoustic/audio/framing verification,
   playable media, copy and collaborator approval remain required. Later approved
   scheduling uses existing idempotent receipts and independently verified dates/
   destinations. No public publishing or collaborator invitations are authorized.

The measurable next LIVE milestone is actual recommendations from the complete
TDOG recording and a working Telegram-delivered owner review. This development
pass does not claim that milestone has already occurred.

## Tests actually executed

- Initial focused pass after wiring fallback paths:66 passed.
- Expanded focused acceptance:72 passed before quota/authentication additions.
- An intermediate full run found one obsolete timeout assertion (546 passed,
  1 failed,2 skipped). It expected no failure result; replaced with assertions
  that sanitized operational evidence exists and is NOT semantic validation.
- Final full regression on the complete application correction:
  **549 passed,0 failed,2 skipped,2 passing subtests;27.89 seconds**.
- Four preexisting FastAPI/Starlette deprecation warnings. Local skips are
  tests/test_stored_transcript_render.py:142 and176, requiring FFmpeg/ffprobe.
  The existing GitHub render job installs those tools.
- Changed application/acceptance Python files parsed successfully with ast.parse.

Exact final command, from ripped-review-simplification:

```powershell
$env:PYTHONUTF8='1'
$env:TEMP=(Resolve-Path ../ripped-intelligence/.test-tmp).Path
$env:TMP=$env:TEMP
$env:DATA_DIR="$env:TEMP/optional-release-data"
$env:PYTHONPATH="$(Resolve-Path ../ripped-intelligence/.test-tmp/opencv4);$pwd;$(Resolve-Path ../pr-validation-handoff)"
& ../rippedshorts/.venv/Scripts/python.exe -m pytest -p validation_network tests -q -rs --tb=short --basetemp=../ripped-intelligence/.test-tmp/optional-release --junitxml=../ripped-intelligence/.test-tmp/optional-release.xml
```

The run used authorized local sockets for HTTP tests with external networking
blocked by validation_network. All storage/test records are isolated. Existing
strict validator/hold tests still execute; the old aggregate moments-failure test
now verifies retained failed evidence and explicit topics-only continuation,
reflecting the newly authorized fallback policy. Checkpoint/evidence counts include
the separate LOCAL link records; no provider-call count was increased for links.
No safety assertion was removed to get a passing count. The existing GitHub
workflow includes the new20-case file and httpx for the actual local HTTP test.
GitHub results are reported separately on the draft PR, not assumed from local tests.
