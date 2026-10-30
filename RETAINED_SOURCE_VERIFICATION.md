# Trusted retained-source verification — 2026-10-30

Scope: Ripped Shorts source verification only. Publishing and Agape stay disabled.

The existing TDOG request 94893455-5f48-46de-bdfb-642212ea19a5 stopped because verify_youtube_source called yt-dlp extract_info(download=False) even though Audio job b7d6bbd5-b48b-476a-b5c0-a551e53f6490 had already produced a verified full source and 1189 valid timed segments.

## Correction
- Read the existing Audio job through its authenticated status GET, never a job-creating POST.
- Match job ID, approved YouTube video ID/URL, request/test-run correlation and brand.
- Require passed upstream media/transcript checkpoints and complete validated duration/stream evidence.
- Bind cached media to the exact Drive artifact named by Audio using authenticated Drive ID, MIME type, size and checksum metadata.
- Compute local SHA256/MD5, probe duration/streams, decode the complete video and audio, and reject changes during verification.
- Read the exact authenticated transcript artifact; validate source ID, intervals and advertised segment count. Preserve honest timing provenance.
- Save a durable source receipt and reuse the verified transcript before ordinary discovery. No source download, transcription, new intake or yt-dlp lookup occurs on the sufficient-evidence path.
- Insufficient evidence falls back to normal source verification; identity, checksum, decode and authentication failures stop processing.
- A retained receipt is not accepted as a substitute for fresh authenticated source binding.
- Ordinary source receipts now include a content digest, so size/mtime alone cannot certify modified media.
- Preserve source/channel distinctions: an unknown channel stays unknown; no creator identity is inferred.

For this recovery, restore only the named request's known Audio job lookup key after verifying its authenticated correlation. This repairs a previously lost upstream ID without changing request identity, selections or approvals. Normal future jobs already retain that lookup key.

## Tests actually executed
- Focused source tests: 29 passing (included in regression total).
- Ripped regression: 130 passed, 0 failed, 2 existing renderer prerequisite skips.
- Existing cross-repository approval/scheduling/mixed-success integration: 4 passed, 0 failed.
- Total: 134 passed, 0 failed, 2 skipped.
- External networking blocked, isolated records, mocked publishing and provider responses.
- Real preflight: existing cached TDOG video size and checksum match the authenticated Drive source artifact; no media download performed.

Commands:
```
python -m pytest -p validation_network tests/test_retained_source_verification.py tests/test_transcript_timing_recovery.py tests/test_clip_completion.py tests/test_batch_progress.py tests/test_creator_media_workflow.py tests/test_workflow_test_status.py tests/test_caption_review_web.py tests/test_render_and_ledger_repair.py tests/test_stored_transcript_render.py -q --tb=short --junitxml=retained-source-results.xml
python -m pytest -p validation_network tests/test_cross_repository_poc.py -q --tb=short --junitxml=retained-source-cross-results.xml
```

Coverage includes mismatched IDs/correlation, missing media, same-size corruption, incomplete files, bad authenticated/local durations, missing streams, decode failure, invalid transcript intervals/counts, missing evidence fallback, authorization failure, forged cache receipts and the real process path reaching candidate generation without acquisition or ASR.

## Authorized rollout
Check current PR checks and mergeability; back up existing persistent records and verify paused publishing/unchanged queue. Merge and deploy Ripped Shorts only, with the old worker fully offline before replacement. Verify deployed SHA, health, one worker and storage/record integrity.

Authenticate the existing Audio job and restore its known reference on the same request. Resume that request once through the normal Telegram retry path, with a durable additional-attempt marker. Never create a new intake, redownload the source, change credentials, release old holds or publish.

Stop at normal Telegram selection after candidate generation; no agent clip selection or candidate caps. Any further integrity/authentication failure stops recovery. The batch remains incomplete until human-selected clips are rendered, reviewed and durably scheduled.

