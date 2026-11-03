# PR86 source restoration and retention gate

## Status: BLOCKED before merge/deployment/recovery

The existing retained Drive artifact was restored atomically to the original TDOG request cache on 2026-11-03 at 04:10:34 UTC using the application's configured Drive identity. No YouTube acquisition, intake, transcription or analysis was performed.

Request: 8999063e-bbc7-4000-bba4-e2490ca27135

- SHA256: 45314424b1529f9d169f8d08c4fce31ddab10b17b122ed107c98458376b80a67
- Size: 207,383,479 bytes
- FFprobe duration: 3679.132154 seconds; audio and video streams present.
- Drive saved MD5 and SHA256 verified before atomic replacement.
- Application worker PID2 mount view and identity can read the file.
- Independently rechecked at 04:18:59 UTC: same checksum, size and worker readability, after the 04:13 cleanup cycle.
- Runtime receipt: /data/release-backups/pr86-source-restoration.json

## Confirmed defect and limits of evidence

/data is a mounted persistent volume; the source resides under /data/sources and is included in active-source discovery. Current retention is 12 hours. The normal cleanup ignores active source directories. Emergency cleanup's age condition also overrides protection. Cleanup traverses release backups. Discovery failures currently log and fail open. Isolated tests reproduce deletion of a held source and backups.

Production logs show periodic normal cleanup at 04:03 (6 files/435267 bytes), 04:08 (4 files/15048 bytes), and 04:13 (0 files) UTC. Watchdog logs retain the validation hold. No per-file unlink audit exists: the exact historical deletion event cannot be proved. There is no evidence of a missing persistent mount or a new source download. A freshly restored file surviving one cycle does not prove safe 12-hour retention.

## Narrow proposed correction

Protect active/held source directories during normal and emergency cleanup, protect release-backups, and fail closed when source ownership cannot be determined. Do not change retention intervals, watchdog recovery authorization, scheduling or publishing controls. This correction is separate from PR86's unchanged tested head 3cf5af0c7747d65dfaf641db0fd34a8e9d381df5.

## Executed tests

Five focused tests failed against existing cleanup and passed with the correction. Full combined regression: **554 passed, 0 failed, 2 skipped, 2 subtests passed in 30.18s**. Skips require FFmpeg tools locally. Command: python -m pytest -p validation_network tests -q -rs --tb=short --basetemp=../ripped-intelligence/.test-tmp/retention-full --junitxml=../ripped-intelligence/.test-tmp/retention-full.xml. External operations blocked by validation_network; HTTP integration uses localhost only.

## Production actions and next authorization

No merge, deployment, recovery, candidate generation, scheduling or public publishing occurred. Production remains 55b64b5c19dc87474e8c90106fd4a2e083a4eb06, deployment 0ac16e7f-29d5-469c-96c4-fbfaf74ee949. Global publishing and Agape were verified disabled. Source restoration does not alter database selections, checkpoints, captions, approvals or receipts.

Owner review/authorization is needed for this additional retention correction. Proposed order: merge PR86, reconcile this dependent fix to main, verify final checks, then deploy Ripped Shorts once paused. Reverify backups, worker/storage and source checksum across transition before one owner-bounded hierarchical-source-v4 recovery. Reuse five successful checkpoints and 1,189 transcript segments. Stop if any safeguard fails. Current new TDOG candidates remain 0 Shorts/0 highlights; no Telegram review-delivery success is claimed.
