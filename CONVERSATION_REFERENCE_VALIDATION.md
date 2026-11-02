# Conversation references — focused correction and review handoff

Base/deployed code: `92779d2795e6facd3baa11915e6ea6257f10dd26`.
Development and isolated tests only. No production API/browser access, deployment,
resume, intake, media acquisition, transcription, scheduling or Telegram action.
No protected production record or publishing/Agape configuration was modified.

## Established cause and limits

The November 2 held request is `8999063e-bbc7-4000-bba4-e2490ca27135`.
Its episode stage passed; conversation position 1 failed with
`Invalid hierarchical related reference`.

Inspection of the deployed validator establishes its exact failure predicate:
`related_keys` was not a list, or an entry was not a string, was absent from the
complete conversation key set, or equaled its own conversation key. The validator
built that complete set before checking any links. Therefore a valid forward
reference was already allowed; validation order cannot alone explain this error.
Duplicate links alone did not trigger this predicate, because duplicates were
not checked there. The model was asked to generate keys and links simultaneously,
before receiving any independently verified identity set: that is a confirmed
contract weakness, not proof of which invalid entry it produced.

The original provider response/rejected literal was not retained. The evidence
cannot distinguish unknown target, self-reference or malformed collection/type.
No specific subtype or provider response has been invented. The tests below use
explicit isolated shapes; they are not claimed as a replay of that lost response.

## Contract and changes

`hierarchical_intelligence.py`:

- `stage_version`/`STAGE_VERSIONS`: independent recipes. Episode, topics and
  moments retain `hierarchical-source-v1`; conversations use
  `conversation-boundaries-v2`; links use `conversation-relationships-v1`.
  Compiled-map version is `hierarchical-source-v2`.
- `schema`, `instructions`, `provider`: the boundary response no longer requests
  or accepts `related_keys`. It still covers the entire transcript through
  ordered semantic starts and deterministic ends. It preserves titles, summaries,
  themes, hierarchy and actual global transcript indices.
- `analyze`: save validated episode and conversation checkpoints first. Then
  generate supplementary links from the complete verified conversation set.
  The link call receives those verified summaries/identities, not another copy
  of the transcript. Topics, full-source/cross-transition moments and the two
  existing format selectors continue after the link checkpoint passes.
- `conversation_id`: supplies deterministic opaque IDs scoped to the source and
  verified conversation key/position/bounds. This is identity assignment before
  link generation, never a replacement for an unknown returned reference.
- `validate_relationships`: accepts `{relationships: [{conversation_id,
  related_ids}]}`. Both fields use an enum of the already verified IDs in the
  provider schema. Local validation rejects unknown source/target, non-string
  IDs, malformed arrays/items, self-links, duplicate targets, duplicate source
  entries and unexpected fields. No links is valid; no relationship quota.
  Forward links are allowed. Links are directed metadata: explicitly supplied
  reverse links are allowed, but never automatically fabricated.
- `safe_reference`: valid opaque IDs are retained literally. Other reference and
  conversation-key values retain type plus a stable SHA256 identifier, not raw
  source text. The validator never renames, clamps, sorts away or drops bad data.

Typed failures use the existing `TopicValidationError` and durable failed-analysis
path. Diagnostics preserve stage/version, conversation position and safe key/ID,
rejected reference type/safe identifier, precise failure kind, item/reference
position, valid-reference count/set digest, schema contract details, prompt
version/hash, global parent bounds and immutable stage analysis ID. Raw provider
output, transcript text and credentials are excluded.

No database schema, watchdog, source acquisition, renderer, review UI, collaborator
control, learning algorithm or Schedule Master code was changed.

Other files:

- `tests/test_conversation_relationships.py`: 22 behavioral cases for valid and
  invalid links, forward references, safe diagnostic persistence, real watchdog
  function suppression, independent cache preservation and exact legacy episode
  reuse. Also checks that unsolicited boundary-stage links fail rather than being
  silently dropped and that verified links reach compiled topic IDs.
- `tests/test_hierarchical_intelligence.py`: supplies the separate link-stage
  fixture and asserts all six checkpoints while retaining full coverage, global
  indices, hierarchy, moments, failure persistence and both-format provenance.
- `tests/test_intelligence_orchestration.py`: supplies the link-stage fixture and
  checks the new compiled-map version reaches both real selector orchestration
  paths alongside the existing rubric and scoped feedback assertions.
- `.github/workflows/stored-transcript-render.yml`: includes the new regression
  file in the existing trigger and intelligence test job.
- This report: records evidence and the proposed recovery procedure.

## Checkpoint reuse

The episode cache-key formula, version, schema and prompt instructions remain
unchanged. An isolated test seeds the exact legacy key formula and verifies the
corrected pipeline starts with conversations, making **no episode provider call**.
It retains historical FAILED aggregate entries and does not reclassify them.

A link failure leaves PASSED episode/conversation rows untouched. Re-entering a
failed link returns its saved hold with zero additional provider calls. A test
explicitly reopens only that link key in isolated SQLite and confirms the next
calls are links → topics → topics → moments; neither episode nor conversation
boundaries repeat. Future relationship recipe changes can be versioned without
changing conversation or episode recipes. A failed outer compiled-map cache also
requires explicit reviewed reopening; no automatic retry or budget reset is added.

The existing watchdog hold and `owner_bounded` recovery marker are preserved.
Tests execute the actual production watchdog function against isolated held
records and verify repeated scans cause no save, counter increment or submission.

## Tests actually executed

Focused pre-final run: **36 passed, 0 failed**, two existing framework warnings.
Final full regression, with external networking denied and isolated storage:

```powershell
$env:PYTHONUTF8='1'
$env:TEMP=(Resolve-Path ../ripped-intelligence/.test-tmp).Path
$env:TMP=$env:TEMP
$env:DATA_DIR="$env:TEMP/conversation-link-full-data"
$env:PYTHONPATH="$(Resolve-Path ../ripped-intelligence/.test-tmp/opencv4);$pwd;$(Resolve-Path ../pr-validation-handoff)"
& ../rippedshorts/.venv/Scripts/python.exe -m pytest -p validation_network tests -q -rs --basetemp=../ripped-intelligence/.test-tmp/conversation-link-full --junitxml=../ripped-intelligence/.test-tmp/conversation-link-full.xml
```

**512 passed, 0 failed, 2 skipped, 2 subtests passed, 4 existing deprecation
warnings; 21.12 seconds.** The two stored-render skips require FFmpeg/ffprobe on
the local PATH (test lines 142 and 176). The existing GitHub render job installs
those tools. All modified Python files also parsed successfully with `ast.parse`.

The full suite retains complete-speech/acoustic checks, real selector orchestration,
independent formats, learning inputs/provenance, Telegram/website persistence,
creator approvals, safe blurred framing and idempotent scheduler handoff tests.
No assertion was removed to obtain a passing result.

GitHub's existing `topic-contract` and `render` jobs execute after draft publication;
their actual final-head conclusions/counts are recorded in the PR report. This
file's isolated results do not pre-claim CI or production semantic acceptance.

## Proposed single bounded recovery — not authorized or executed

1. Review the draft PR and final GitHub checks. Obtain separate merge, paused
   deployment and one-recovery authorization. Recheck backups, disabled auto-deploy,
   persistent storage, one worker/no old-new overlap, credentials and disabled
   global/Agape publishing. Deploy Ripped Shorts only after authorization.
2. Verify this same request/source, retained complete video, duration/streams/
   integrity, and all 1,189 timed segments. Verify the existing passed episode
   key and its immutable result fingerprint **before dispatch**. Stop if the
   recomputed key differs or the successful checkpoint/artifact is unavailable.
   Do not download, transcribe or start another episode analysis.
3. Preserve all previous recovery receipts and failures. Record a distinct new
   owner-authorized bounded recovery revision on the same request; keep
   `analysis_recovery.mode=owner_bounded`. Select the new aggregate key
   `<existing-map-id>:hierarchical-source-v2` and corrected conversation recipe.
   Clear only the specifically reviewed request hold. Do not reopen v1 failures
   or broadly reset budgets. Stop if the new recipe already ran unexpectedly.
4. Dispatch once through the existing authorized recovery workflow. The episode
   must be a cache hit. Old failed conversation content is unavailable and must
   not be reused as a valid map: the corrected boundary stage runs once, then the
   independent identity-constrained link stage runs once. Preserve each success.
5. Any deterministic failure holds with diagnostics and no repeated call. If
   links fail, keep valid boundaries; do not restart the episode or conversation
   analysis. Require another explicit review/authorization for any later recovery.
6. Only after valid hierarchy/moments may normal independent 9:16/16:9 analysis
   consume it and scoped historical feedback. Verify actual rubric/map/prompt
   provenance and candidate quality; present Shorts before highlights and deliver
   the secure request-specific website link with a real Telegram receipt.
   No automatic selections, scheduling, public posts or collaboration invitations.

## Remaining production limitations

This pass does not prove a real provider will return semantically correct
conversations/links or acceptable clips. It does not create candidates or verify
new live website media/scheduled records. Those remain acceptance steps after
separate authorization. The prior literal related-reference subtype remains
unknown permanently unless an independent retained response is discovered.

Production was not accessed or mutated in this correction pass. The existing
request, successful episode, media/transcript, previous selections/holds,
learning/Sheets history, schedules and Telegram receipts were untouched.
No publishing or Agape setting was enabled or changed.
