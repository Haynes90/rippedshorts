# PR #86 — production preflight stopped before merge

## Outcome: BLOCKED on required retained source media

Owner authorized a conditional merge, paused Ripped Shorts-only deployment and
one bounded recovery of request8999063e-bbc7-4000-bba4-e2490ca27135. No merge,
deployment, intake/recovery, download, transcription, scheduling, invitation or
publication was executed. PR86 remains draft/unmerged.

Verified GitHub head: `3cf5af0c7747d65dfaf641db0fd34a8e9d381df5`, mergeable.
Final workflow run41540429637 passed:169 topic-contract tests and12 render tests.
Existing full regression evidence:549 passed,0 failed,2 local FFmpeg skips,
2 passing subtests (JUnit optional-release.xml). Code has not changed since.
Current production remains `55b64b5c19dc87474e8c90106fd4a2e083a4eb06`, Railway
deployment `0ac16e7f-29d5-469c-96c4-fbfaf74ee949`; replacement not requested.

## Passed preflight evidence

- SQLite integrity:ok, existing request still awaiting_topic_map_review.
- All1,189 saved timed transcript segments remain; five PASSED checkpoints remain.
- Fresh backup completed at `/data/release-backups/pr86-20261103T015533Z`:
  database snapshot, all15 table manifests,25 workbook tabs, environment
  fingerprints, five checkpoint fingerprints and backup-result receipt.
- The four previously fingerprinted checkpoints match the retained PR85 baseline;
  the fifth PASSED chronological conversation2 result was captured separately.
- Service health /ping200, readiness=true, one uvicorn worker and attached volume.
- Railway auto-deploy visibly disabled; one configured replica.
- Effective Schedule Master global publishing=false, Agape Meta=false,
  Agape YouTube=false; audit/reconciliation remain12h/300s. No settings changed.

The backup script initially referenced an unavailable older hierarchy comparison
report. That operator-script reference was changed to the retained PR85 baseline;
the current DB, latest backup and checkpoints passed. No application code changed.
The older PR85 environment-fingerprints.json is no longer available, so an
unchanged-credentials comparison against that older release could not complete.
The fresh current environment fingerprint snapshot exists for future postdeploy
comparison. No credential values were read into the report or changed.

## Exact essential failure

The fresh backup initially verified the retained source SHA256:
`45314424b1529f9d169f8d08c4fce31ddab10b17b122ed107c98458376b80a67`
(size207,383,479 bytes). A later required FFprobe check failed with
**No such file or directory**, exit1, at:
`/data/sources/telegram-8999063e-bbc7-4000-bba4-e2490ca27135/9mW0bhmUaw0-source.mp4`.

A final independent read-only check confirmed:
- saved video_path is that same path;
- saved_path_exists=false and expected_path_exists=false;
- segments=1189, passed_checkpoint_count=5;
- stage=awaiting_topic_map_review, recovery_revision=3 (not advanced).

This is an essential media-availability failure, not optional metadata. Fallback
must not bypass it. The file became unavailable between successful checksum
verification and later probe. The cause is NOT established; cleanup, external
intervention and storage behavior are hypotheses, not findings. No deletion,
move, media regeneration or source download command was issued by this task.
No stale media validation receipt was treated as proof of current availability.

## Production acceptance not reached

Five checkpoint REUSE during a new run is not claimed: recovery was never
submitted. Current checkpoint presence/fingerprints were verified only.
No new hierarchical-source-v4 analysis, fallback selection, 9:16/16:9 candidates,
real candidate provenance, Telegram review delivery or website session was produced.
No finished render, scheduled record or public post is claimed.
All existing selections, approvals, captions, collaborators, schedules, historical
analysis failures and receipts were left untouched by this task. Existing holds
were not released. Global publishing and Agape remain disabled.

## Exact next action

Restore the previously retained source artifact into the SAME request cache and
verify its checksum, FFprobe duration/streams and continued availability. Confirm
why it became unavailable before relying on that cache for deployment/recovery.
The owner's latest instruction prohibits another download and requires a stop on
failed safeguards, so restoring the retained Drive copy requires a specific
exception/approval; do not silently substitute a YouTube download or new intake.
After the retained media safeguard passes, recheck current backup/configuration,
then continue the already authorized conditional PR86 merge, Ripped Shorts-only
paused deployment and exactly one same-request owner-bounded v4 recovery. Preserve
all five passes, old failures and retry budgets; do not regenerate the episode.
