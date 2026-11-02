# PR #84 — paused release and bounded recovery

## Authorization and code

Owner authorized PR #84 merge, Ripped Shorts-only paused deployment and one
bounded recovery of existing request `8999063e-bbc7-4000-bba4-e2490ca27135`.
Owner separately approved restoring the retained Drive copy after preflight
found its local video cache absent.

- PR: https://github.com/Haynes90/rippedshorts/pull/84
- Tested head: `edc482c1d806132942dd78fcd8230f3c6d558a7d`.
- Merge: `461e8b7144376ac3d4496f20476a378646212fa1`.
- Merge/tested tree: `f7ee0ba50c2729ca54c4fe0f6ce96c07b0753fc5`, identical.
- Existing complete regression at unchanged head: 512 passed, 0 failed,
  2 local FFmpeg skips, 2 passing subtests. No redundant rerun after merge.
- Final GitHub checks: 132 intelligence tests and 12 render tests passed.

## Source restoration and preflight

The first preflight stopped before merge because the saved local video path was
absent. Transcript and source-validation artifacts remained. No recovery ran.
Drive metadata confirmed the retained object accessible/not trashed with matching
MD5 and SHA256. Owner explicitly approved restoring that same artifact.

Restoration reused the existing Ripped Shorts service identity and retained
Drive object, without YouTube acquisition, retranscription, new intake or episode
generation. The temporary download was checksum-verified before atomic placement
at the original request cache path. Restored size: 207,383,479 bytes. SHA256:
`45314424b1529f9d169f8d08c4fce31ddab10b17b122ed107c98458376b80a67`.
FFprobe verified audio/video streams and duration 3679.132154s, consistent with
the retained validation evidence. Restoration receipt:
`/data/release-backups/pr84-source-restoration.json`.

Current verified backup:
`/data/release-backups/pr84-20261102T151211Z`.
Contains the SQLite snapshot with integrity `ok`, all 15 table manifests,
credential/configuration fingerprints and all 25 workbook tabs, including queue,
approval, copy and receipt records. Backup-result JSON contains the database and
episode SHA256 values.

The passed episode checkpoint is unchanged and its canonical result fingerprint
matches the earlier saved recovery outcome:
`5cd48cbd49649163472c37ea1c5c410d:hierarchical-source-v1:episode:a32ebb7ddce19662bcc95c985d97ec2c`.
All 1,189 nonblank ordered timed segments remain saved.

Preflight verified current base SHA `92779d2`, healthy `/ping` and readiness,
one uvicorn worker, attached persistent volume and one configured replica.
Railway auto-deployment visibly disabled. Schedule Master's effective global,
Agape Meta and Agape YouTube publishing gates are all false; audit/reconciliation
remain 12h/300s. No credentials or publishing settings changed.

## Controlled deployment

Old deployment `c6b3e0e6-7aaf-4eac-8689-40e577914a62` was stopped through Railway's
deployment Remove action (documented to stop the deployment, retaining service
and volume). Railway explicitly showed Service offline/no active deployment
before replacement was requested.

Replacement deployment: `7a98fa01-fb39-43dd-a900-396fa1f6ae62`.
Only Ripped Shorts was requested. Audio Master and Schedule Master were untouched.

## Verified deployment and dispatch

Runtime verified exact SHA `461e8b7144376ac3d4496f20476a378646212fa1`, healthy
`/ping` HTTP 200, readiness true, SQLite integrity ok and one uvicorn worker.
All 15 predeployment table hashes and credential fingerprints match. Protected
workbook data is unchanged; only unrelated Church Job Tracker Updated At values
changed with the row count preserved. Restored source remains checksum-valid.

The corrected code recomputed the exact legacy episode identity from actual
request metadata and normalized transcript inputs. It matched the passed saved
checkpoint and its canonical fingerprint (SHA256 prefix `7320ef1c51d1`). This
was verified before the bounded dispatch, rather than assuming cache reuse.

One authenticated administrative dispatch through the existing intake callback
was acknowledged HTTP 200 at **2026-11-02T15:20:09.379036+00:00** for the SAME request.
Dispatch: `pr84-hierarchy-v2-8999063e-bbc7-4000-bba4-e2490ca27135-1`.
Recipe `hierarchical-source-v2`, recovery revision **2**, attempt **1**,
`analysis_recovery.mode=owner_bounded`. The earlier v1 authorization, failure hold
and source identity were appended to recovery history, not discarded. Failed v1
analysis records/budgets were not reopened or reset. The episode ID/fingerprint
are recorded in the new authorization. No replacement intake was created.

Runtime receipts and evidence are in the backup folder:
`post-deployment-verification.json`, `pr84-bounded-recovery.json`, and
`latest-recovery-progress.json`.

Actual hierarchy stages, candidate counts and owner-review receipts remain to be
verified from runtime. The API acknowledgement does not establish candidate,
render, review or scheduling acceptance.

## Actual recovery outcome — BLOCKED, stopped

The one authorized recovery stopped at `awaiting_topic_map_review`, retryable
false. Recovery revision 2/attempt 1 remains held; historical watchdog count 3
is unchanged. No additional recovery was dispatched.

Verified stages:

- Episode checkpoint reused unchanged (all 1,189 timed segments).
- Conversation boundaries passed; four ordered intervals cover all 1,189 indices.
- Conversation relationships passed: four sources, seven directed links.
- First conversation topics passed: eight nodes (one root and seven subtopics).
- Second conversation topics failed validation; later topics, moments and both
  clip selectors did not execute.

| Conversation | Inclusive original segment range |
| --- | --- |
| Opening sketch, casual check-ins and travel banter | 0–129 |
| Episode focus revealed; PlantCon & workshop promotions and giveaways | 130–350 |
| Seed-saving strategies, storage and germination techniques | 351–900 |
| Viewer Q&A, product mentions, practical takeaways and sign-off | 901–1188 |

These boundaries reflect irregular semantic transitions and retain introductions,
promotions and closing material. Full hierarchy semantic acceptance remains
incomplete: later topics and interesting moments have not been generated.

Actual final durable diagnostic: `TOPIC_MAP_INVALID`, stage `topics`,
`TopicValidationError`, reason **Semantic hierarchy must follow depth-first
parent order**, zero-based **topic_position 10** (eleventh proposed topic in the
second conversation). An earlier progress interpretation reported position 5;
the final persisted outcome establishes position 10. Parent bounds 130–350;
transcript bounds 0–1188. Topic-stage version/prompt remains
`hierarchical-source-v1` / `hierarchical-source-v1:topics`.
Prompt hash `bd0f7ef8595f8365f40794dbd6695d0c2366a4cf619cc0a3e3ad3dfe8d46bdbc`.

The validator rejects a supplied parent that is not a string, has not already
been seen, or is no longer on the active depth-first ancestry stack. The saved
diagnostic does not identify the offending parent key; the original rejected
provider topic response was not retained. Consequently, unknown/forward target,
closed-parent reference and invalid type cannot be distinguished from available
evidence. The diagnostic's null `reference` is an omitted diagnostic argument,
not proof that the provider supplied a null parent. No references were repaired,
clamped or fabricated. The child and enclosing analysis failure records reflect
one propagated failure, not two recovery attempts.

Final runtime evidence is saved as `pr84-recovery-outcome.json` in the verified
backup folder. It confirms:

- Episode fingerprint unchanged.
- Other request states unchanged, including previous TDOG selections/holds.
- Request count 39 before and after; no new intake identities.
- No changes in protected TCB queue, schedule or handoff-outbox records.
- Readiness true; no scheduling or publication triggered.

Fresh candidate counts: **9:16 = 0; 16:9 = 0**. No fresh finished renders, framing
acceptance, owner review session or scheduled deliveries can be claimed.
Hierarchy and historical learning consumption by the live selectors is **NOT
VERIFIED** because neither selector ran. No candidate-ready Telegram receipt or
new working review URL exists for this recovery; sending such a link would
misrepresent its state. Existing website/collaborator behavior was regression
covered but not reached by this production attempt.

The prior website submission remains preserved. The earlier read-only
reconciliation found no corresponding successful Schedule Master records; this
recovery did not resubmit it or create new scheduled work. No dates or captions
were changed. Publishing globally and Agape remain disabled.

## Next action

A focused topic-parent contract/diagnostic correction is required before another
production recovery. Preserve the passed episode, conversation, relationship and
first-conversation topic checkpoints; do not rerun them unnecessarily. Any new
code deployment and additional bounded recovery require separate owner approval.
This release is healthy and paused, but the end-to-end acceptance is **BLOCKED**
at topic hierarchy validation, not ready for owner clip selection or scheduling.
