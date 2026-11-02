# Hierarchical intelligence — controlled release and recovery

## Code and tests

- PR #83: https://github.com/Haynes90/rippedshorts/pull/83
- Ordered-boundary/watchdog commit: `889d5fe76f9619639bdaf8cc88f2ff6deb5bc451`.
- Hierarchical integration commit: `be9708c0d200aa7c43f44e418570a565defed019`.
- Merge/deployed commit: `92779d2795e6facd3baa11915e6ea6257f10dd26`.
- Merge tree `d6fc7d2e2c2588648f16e97fcaf95f4f030a929a` exactly matches the tested PR tree.
- Complete isolated regression: **490 passed, 0 failed, 2 skipped, 2 subtests
  passed**, 4 existing deprecation warnings, 18.01s. Exact command and fixtures:
  [implementation report](https://github.com/Haynes90/rippedshorts/blob/92779d2795e6facd3baa11915e6ea6257f10dd26/HIERARCHICAL_ACCEPTANCE.md).
- GitHub topic-contract: **110 passed**; GitHub stored-render: **12 passed**,
  including FFmpeg-dependent tests unavailable on the local test PATH.
- Existing episode source, topic, candidate and feedback persistence is reused.
  New stage checkpoints live in the existing analysis cache. No new worker,
  database, paid service, creator UI or publishing architecture was introduced.

## Safeguards and release

- Current backup: `/data/release-backups/hierarchy-20261102T040624Z`.
  SQLite integrity `ok`; SHA256
  `e7ad05c11530157d743414d708d498688b88d60de47549c36fe1a29cfc8fcffb63`.
  All 15 database tables, approvals, receipts and all 25 workbook tabs backed up.
- Railway auto-deploy visibly disabled; one replica with attached persistent
  volume. Existing effective Schedule Master gates: global publishing `false`,
  Agape Meta `false`, Agape YouTube `false`; audit/reconcile remain 12h/300s.
- Automatic approval review initially rejected the deployment menu's Remove
  action as possibly destructive. Official Railway documentation established
  that it stops the running deployment and retains it in history, with the
  service/volume preserved. The action was then approved on that evidence.
- Old deployment `f0d063bb-2f26-4517-b79a-cb44006ac86d` stopped; Railway visibly
  showed no active deployment and Service offline before replacement started.
- Replacement deployment `c6b3e0e6-7aaf-4eac-8689-40e577914a62`, exact SHA above.
  `/ping` and readiness HTTP 200; readiness `ready=true`, all checks true;
  persistent storage integrity `ok`, one uvicorn worker.
- All 15 table hashes and credential fingerprints matched immediately after
  replacement. Protected workbook tabs unchanged. Only unrelated Church Job
  Tracker Updated At cells changed (58 rows before/after); independently checked
  rather than representing the entire workbook as byte-identical.
- Audio Master and Schedule Master were not redeployed. No credentials,
  destinations, publishing gates or posting intervals were changed.

## Prior TDOG review reconciliation

Existing request `94893455-5f48-46de-bdfb-642212ea19a5` remains held at
`awaiting_render_retry`; all five original selected identities and decisions
were preserved. Three selected Shorts failed exported-speech/boundary matching.
Two highlights remain failed: one exported-speech mismatch and one ending too
close to speech. Two copy drafts are approved and a Finish & Schedule intent
timestamp exists, but no `schedule_master` acknowledgement is saved.

Read-only verification from Schedule Master's own runtime used its real
`schedule_tab_for_brand('TCB')` routing: actual tab **TCB**, 279 existing rows,
**zero matching records for either TDOG request**. An initial lookup using the
unused generic fallback tab name returned a missing-range error; the routed
brand queue was then checked successfully. Website submission is not represented
as a scheduled delivery. No previous handoff was repeated or old render retried.

## Retained inputs and real-media validation

- Fresh request `8999063e-bbc7-4000-bba4-e2490ca27135` preserved.
- Source `9mW0bhmUaw0`; retained video SHA256
  `45314424b1529f9d169f8d08c4fce31ddab10b17b122ed107c98458376b80a67`
  verified before the isolated media check. No source download or transcription.
- Existing **1,189 nonblank ordered segments**, first start 2.36s, last end
  3678.92s. Existing source-validation identity, duration, stream/decode,
  authenticated Audio Master job and transcript provenance remain saved.
- An isolated real-source 3-second export through the unchanged safe full-frame
  blurred-background function produced 1080×1920 H.264 video, AAC audio,
  duration 3.000s, 1,112,084 bytes, max volume **−0.3 dB**. Sharp foreground uses
  the complete horizontal source over a moving blurred/darkened duplicate.
  This is a validation fragment, not a proposed or approved candidate; it does
  not establish clip-specific semantic boundaries, human visual acceptance or
  speaker-tracking quality across all layouts.

## One bounded recovery

- Explicit owner authorization retained in the existing request with
  `analysis_recovery.mode=owner_bounded`, attempt **1**. No broad retry reset.
- Immutable outer analysis key:
  `5cd48cbd49649163472c37ea1c5c410d:hierarchical-source-v1`.
- Dispatch ID `hierarchy-8999063e-bbc7-4000-bba4-e2490ca27135-1`.
- Authenticated existing intake callback dispatched once; HTTP 200 acknowledged
  the same request at **2026-11-02T04:17:30.300264+00:00**. This was an authorized
  administrative resume through the existing bot workflow, not a new Telegram
  user-origin submission or replacement intake.
- Durable operational receipt:
  `/data/release-backups/hierarchy-20261102T040624Z/hierarchical-bounded-recovery.json`.
- Runtime confirms retained Audio Master source/transcript status
  `reused_verified`. Old watchdog count remains at its historical **3**; the
  bounded marker prevents further automatic dispatch.
- Historical failed map results and diagnostics remain preserved. They are not
  treated as successful maps. No original selections were replaced.

## Actual recovery outcome — BATCH BLOCKED

The episode stage **PASSED** and remains durably reusable. Its saved title is
“TDOG S2E39 Seed Saving Strategies.” The saved summary covers seed storage and
germination testing; seed-specific handling and fermenting tomato seeds;
propagation/cloning; practical gardening questions; workshop/event promotion;
and informal humorous tangents/product mentions. This is meaningfully aligned
with the episode subject and includes peripheral exchanges. It is not a claim
that all later semantic or clip-quality acceptance has passed.

Actual episode provenance: `hierarchical-source-v1:episode`, provider OpenAI,
model `gpt-5-mini`, actual prompt SHA saved in the passed checkpoint. The complete
verified transcript was supplied, not just a summary. The next stage received
the complete transcript plus that saved episode context.

The **conversations stage FAILED** after its ordered-range construction, on
the application's semantic related-key validation for **topic_position 1**
(the second conversation):

```json
{
  "category": "TOPIC_MAP_INVALID",
  "reason": "Invalid hierarchical related reference",
  "prompt_version": "hierarchical-source-v1:conversations",
  "topic_position": 1,
  "stage": "conversations",
  "parent_first_segment": 0,
  "parent_last_segment": 1188
}
```

The actual prompt hash and stage analysis ID are retained in SQLite and in
`hierarchical-recovery-outcome.json` in the backup folder. Episode checkpoint:
`5cd48cbd49649163472c37ea1c5c410d:hierarchical-source-v1:episode:a32ebb7ddce19662bcc95c985d97ec2c`.
Failed conversation checkpoint:
`5cd48cbd49649163472c37ea1c5c410d:hierarchical-source-v1:conversations:981afa260f090385abe3b5fba26c8e7f`.

This is an application validation rejection of a returned semantic link, not
an HTTP authentication error or the previous reversed interval. The current
diagnostic identifies the precise stage, position, parent bounds and prompt,
but **does not retain the rejected literal related key**. It therefore cannot
distinguish an unknown target from a self-reference after the response has been
discarded. No missing value or specific subtype is invented here. Conversation
count/titles cannot be certified from an accepted map because that stage failed.
This diagnostic limitation and related-link generation contract need the next
focused correction; the successful episode stage must be preserved.

Topics, moments and both format selectors **did not run**. Actual new candidate
counts: **9:16 = 0; 16:9 = 0**. No new clips were selected, rendered, approved,
handed off or scheduled. There is no fresh finished-media review URL ready.
Actual selector consumption of hierarchical/learning inputs is therefore not
claimed; that integration passed isolated orchestration tests only.

The same request is held at `awaiting_topic_map_review`, `retryable=false`;
bounded attempt remains **1** and historical watchdog count remains **3**.
No additional recovery/provider retry was dispatched. The outer failed map and
failed conversation entries represent failure propagation through two cache
levels, not two separate conversation provider attempts. The episode success
and conversation failure are the two executed hierarchical provider stages.

A private stop-status notification through verified `@rippedshortsbot` was
acknowledged as **message 2699**, saved COMPLETE in the durable outbox and
`telegram-hold-notice.json`. It is explicitly a failure/status notification,
**not a review-link or finished-clip delivery receipt**.

Final read-only checks confirm **39 existing request records before and after**,
no new intake identities, and no changes to any other request's saved state.
The TCB queue, queue/support schedule and handoff-receipt tabs match the backup.
The five earlier selections and their holds remain unchanged. These checks are
saved in `final-protected-record-check.json`. Service readiness remains true.

## Remaining blocker and next action

The paused release is healthy, but end-to-end acceptance and scheduling readiness
are **BLOCKED** by the conversation semantic-link validation failure. Passing
regressions are not represented as successful production acceptance.

Next: a focused correction/review of semantic related-reference generation and
safe diagnostic evidence. Any later recovery must receive separate authorization,
reopen only the affected reviewed analysis identities, and reuse the passed
episode checkpoint plus retained source/transcript. No new intake, download,
transcription, old-request reset, broad budget reset or automatic retry.

Owner clip decisions cannot begin until valid candidates exist. After that,
finished rendering, website copy/framing/creator approvals and verified
exactly-once Schedule Master records are still required. Publishing and Agape
remain disabled; no public post or native collaboration invitation was sent.
