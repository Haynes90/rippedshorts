# Hierarchical intelligence — acceptance evidence

Extension of PR #83, based on `889d5fe76f9619639bdaf8cc88f2ff6deb5bc451`.

## Implementation

`hierarchical_intelligence.py` uses the existing `source_analyses` SQLite cache,
typed diagnostics and model endpoint. No new database, worker or external service.

1. Full verified transcript → concise episode understanding: themes, flow and
   beginning/middle/ending. The full transcript remains the input to later passes.
2. Full transcript plus episode context → overarching semantic conversations.
   Ordered starts derive inclusive ends; introductions and closing remain covered.
3. Each verified conversation → topics/subtopics and explanatory context, using
   original global indices within that parent's actual interval. Children are
   independently validated and checkpointed, with no renumbering or range repair.
4. Full transcript plus validated hierarchy → interesting moments, including
   exchanges across chapter transitions. These are not restricted to one topic.
5. Existing independent format selectors consume the same compiled map, episode
   moments and historical feedback, retaining separate approved rubrics and
   sequential owner presentation. No quota or forced novelty is introduced.

Stage keys include source identity, transcript revision, immutable parent context,
recipe version and configured model. Successful checkpoints are reused; failed
children do not erase siblings. Invalid stage output retains sanitized stage ID,
parent bounds, reference shapes and prompt provenance in the failed-analysis row.
Outer cached failures preserve the child validation classification. PR #83's
review hold, legacy-cache protection and bounded-recovery watchdog guard remain.

Candidate evidence now includes hierarchy stage IDs, input-map identity, scoped
feedback IDs/digest, format/show/brand scope and existing prompt/specification
provenance. Learning examples distinguish whether their existing Sheets
projection digest was verified. SQLite remains the processing authority; this is
not a new learning system or a claim of live Sheets readback in these tests.

Rendering, collaborator controls, website decisions and scheduler approval gates
are reused. They have not been redesigned or bypassed.

## Executed tests

Focused hierarchical/orchestration/watchdog run: **50 passed**, two existing
framework warnings. Provider/publishing operations mocked; isolated storage.

Complete regression command (from repository snapshot):

```powershell
$env:PYTHONUTF8='1'
$env:TEMP=(Resolve-Path ../ripped-intelligence/.test-tmp).Path
$env:TMP=$env:TEMP
$env:DATA_DIR="$env:TEMP/hierarchy-full-data"
$env:PYTHONPATH="$(Resolve-Path ../ripped-intelligence/.test-tmp/opencv4);$pwd;$(Resolve-Path ../pr-validation-handoff)"
& ../rippedshorts/.venv/Scripts/python.exe -m pytest -p validation_network tests -q -rs --basetemp=../ripped-intelligence/.test-tmp/hierarchy-full --junitxml=../ripped-intelligence/.test-tmp/hierarchy-full.xml
```

**490 passed, 0 failed, 2 skipped, 2 subtests passed, 4 deprecation warnings;
18.01 seconds.** The two skips require FFmpeg/ffprobe on the local PATH. Existing
GitHub render CI installs these tools. No assertion was removed for green counts.
The orchestration fixture now supplies the staged provider contract rather than
the superseded one-response generator. Its reuse assertion checks the specific
versioned analysis ID instead of assuming the root source key is the analysis key.

New behavioral tests verify global parent bounds, full-source coverage,
cross-conversation moments, independent durable stage reuse, deterministic failed
stage holds, explicit isolated child reopening, retained prompt evidence, and
both selector payloads receiving hierarchy, approved rubric and scoped historical
feedback with verified-projection evidence. Existing complete-speech, renderer,
Telegram, website, collaborator and handoff regressions run in the full suite.

These are isolated implementation results, not real semantic-quality acceptance.

## Live read-only reconciliation before release

Current Ripped Shorts deployment observed:
`f0d063bb-2f26-4517-b79a-cb44006ac86d`, commit
`9a9d3effe1213eb8be9edfed73c1f050b5fa8e72`; persistent `/data`, SQLite integrity
`ok`, one uvicorn process. Retained fresh-request source file exists.

Fresh request `8999063e-bbc7-4000-bba4-e2490ca27135`: still
`awaiting_topic_map_review`, no candidate/copy/scheduler results.

Previous request `94893455-5f48-46de-bdfb-642212ea19a5`: still
`awaiting_render_retry`. Three selected Shorts remain `render_failed` with
exported-speech/boundary mismatch diagnostics. Both selected highlights remain
`render_failed`; one has a speech/boundary mismatch and one ends too close to
speech. Two copy drafts are approved and a scheduling request timestamp exists,
but `schedule_master` is null. This proves website submission intent, not scheduler
acknowledgement or actual scheduled records. No retry or resubmission was made.
Independent Schedule Master queue reconciliation is still a release checkpoint.

Ripped Shorts does not expose global/Agape publishing gates in its own process
environment; their absence is not treated as proof of disabled publishing.
The authoritative Schedule Master runtime must be verified before merge/deploy.

## Release/recovery gates

Pending: final GitHub checks, authoritative publishing-gate confirmation, current
verified backups including queue/receipts, auto-deploy disabled, one replica with
no overlapping workers, protected-record fingerprints, then authorized merge and
Ripped-Shorts-only paused deployment. Stop on any failed safeguard.

After deployment, select the new immutable `hierarchical-source-v1` analysis ID for
the existing fresh request only, mark `analysis_recovery.mode=owner_bounded`, and
dispatch once using retained verified media and 1,189 segments. Do not reset old
failed analyses or retry budgets. An invalid stage must hold with evidence; a
successful run must be assessed for actual semantic quality and both candidate
counts, followed by a receipt-backed request-specific Telegram website link.

No automatic clip selection or public publishing. Final scheduling remains gated
on the owner's explicit final website approvals, verified finished media/copy/
collaborators, idempotent receipts and independently confirmed schedule records.

Actual conversation titles, candidate counts, finished-media acceptance, Telegram
receipt and scheduled dates must be filled only from real run evidence; none are
claimed by this implementation report.
