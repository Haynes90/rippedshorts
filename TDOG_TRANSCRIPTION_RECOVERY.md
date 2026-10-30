# TDOG transcription recovery — 2026-10-30

Status: CODE VALIDATED; MERGE / PAUSED FIX DEPLOYMENT / SAME-REQUEST RECOVERY AWAIT OWNER APPROVAL.
No production recovery was executed. No existing release was redeployed.

## Existing identities
- TDOG source: https://www.youtube.com/watch?v=9mW0bhmUaw0
- Ripped request: 94893455-5f48-46de-bdfb-642212ea19a5
- Test run: TW-94893455-5f48-46de-bdfb-642212ea19a5
- Audio job: b7d6bbd5-b48b-476a-b5c0-a551e53f6490
- Retained full video: Drive file 1-NMfHs3Nse-KMkR47-qZcLGY5qnleSdz, validated 3679.132154 seconds, video and audio.
- Retained Ripped video: /data/sources/telegram-94893455-5f48-46de-bdfb-642212ea19a5/9mW0bhmUaw0-source.mp4 (207383479 bytes).
- Retained Audio audio: /app/data/audio/9mW0bhmUaw0.mp3; all seven chunk files remain available.

## Confirmed diagnosis
1. Audio's saved job says seven completed chunks followed by Invalid provider word timeline. Its timed TXT was written after all chunks returned, but raw provider JSON was discarded before final word validation.
2. One bounded diagnostic used cached chunk 000, with no job restart, source download, queue write or credential change. OpenAI returned HTTP 200, 1610 word records and 213 segment records. A blank word at index 590 had start=end=228.1999969482422. Six blank segment records also had zero duration. Transcript text and credentials were not printed or committed.
3. Audio's validator rejects every empty word, including this non-speech zero-duration token, and thereby rejects the entire otherwise usable transcript. The deployed Ripped validator has the same rejection condition.
4. The reproduced token is in the interior of chunk 000 with offset zero. It is not a double-offset or chunk-boundary failure. Offset behavior is regression-tested. No evidence establishes a failed audio extraction.
5. Ripped's original fallback discarded individual response/validation reasons and ultimately reported only Word transcription failed after three attempts. Its original response was not retained; the exact original failing word/window is NOT VERIFIED. It is incorrect to claim a provider HTTP failure or prove that it was precisely the same blank token. The shared validator defect is reproduced, and the new path retains evidence and stops deterministic validation retries.
6. The retained full timed TXT contains 1197 records: eight blank zero-duration records and 1189 nonblank ordered intervals. All lines parse, source identity matches, and all 1189 intervals are finite, positive and within the complete source duration. It can be recovered as segment timing without another full-source transcription.
7. A later automatic attempt recorded HTTP 409 Workflow correlation is immutable. The known Audio job acknowledgment was compared as if it changed immutable workflow identity. The correction permits only the same known job ID; request/source/brand identity remains immutable.

## Corrections
- Shared pure timing validator in the two existing apps: retain actual valid word timing; omit only blank zero-duration records containing no lexical speech; preserve all nonblank content.
- Explicit provider_segment_timing fallback when word evidence is absent/invalid but segment intervals validate. No caption words receive estimated times. When only some chunks have words, provenance is mixed_provider_timing.
- Existing exported-speech checks remain required. Segment-timed export verification additionally requires measured acoustic quiet at both cuts. Punctuation alone cannot pass a boundary.
- Successful provider JSON is retained privately, tied to audio content and model, so deterministic bad timing is not repeatedly submitted. Sanitized structural reasons are retained; credentials and provider error bodies are not logged.
- Audio RIPPED-only recovery can reuse the complete saved timed TXT for this exact legacy failure/source instead of repeating audio acquisition or all transcription chunks.
- RIPPED deterministic timing failures stop same-audio retries; transient failures remain bounded/retryable. Ordinary podcast request settings remain unchanged.
- Audio status exposes the timing basis/word count. Ripped preserves the authenticated upstream MEDIA_READY checkpoint when transcription fails.
- Known same-job acknowledgments no longer invalidate the durable workflow identity.
- Updated obsolete render assertions reproduced as failures against deployed Ripped base code. Updated two Audio source assertions that referred to an already-replaced queue helper or rejected an explanatory comment rather than executable calls. No sermon or podcast processing changes were made to satisfy those assertions.

## Actual final tests
- Audio Master: 124 passed, 0 failed, 0 skipped.
- Ripped Shorts: 101 passed, 0 failed, 2 skipped.
- Existing cross-repository mixed-success / approval / scheduling / mocked adapter integration: 4 passed, 0 failed, 0 skipped.
- Total: 229 passed, 0 failed, 2 skipped. No full three-repository investigation or Schedule 252-test rerun.
- Two skips: existing real renderer prerequisites (FFmpeg and FFprobe together on local PATH). This is not proof of real rendered-clip completion.
- Earlier local test attempts hit Windows sandbox loopback/socketpair restrictions and temp-path setup issues. Final tests used authorized local loopback access with external networking blocked and isolated data. These were environment limitations, not claimed passing runs.

Commands (each app root; PYTHONPATH contains existing local test dependencies and validation_network):
```
python -m pytest -p validation_network [17 existing Audio workflow regression files listed below] tests/test_transcript_timing_recovery.py tests/test_ripped_shorts_ingest_runtime.py tests/test_ripped_shorts_source_video.py -q --tb=short --junitxml=timing-recovery-results.xml
python -m pytest -p validation_network tests/test_transcript_timing_recovery.py tests/test_clip_completion.py tests/test_batch_progress.py tests/test_creator_media_workflow.py tests/test_workflow_test_status.py tests/test_caption_review_web.py tests/test_render_and_ledger_repair.py tests/test_stored_transcript_render.py -q --tb=short --junitxml=timing-recovery-results.xml
python -m pytest -p validation_network tests/test_cross_repository_poc.py -q --tb=short --junitxml=timing-recovery-cross-results.xml
```
All publishing was mocked. Local JUnit XML/logs retain final exact counts.

Audio regression files: test_approval_learning_runtime_contract.py, test_clipmaster_unblock_runtime_contract.py, test_ffmpeg_signal11_log_contract.py, test_media_ingest_foundation.py, test_partial_sermon_api_stream_runtime.py, test_partial_sermon_format_runtime.py, test_partial_sermon_no_reencode_runtime.py, test_partial_sermon_timeline_runtime.py, test_publisher_youtube_runtime_contract.py, test_ripped_media_contract_behavior.py, test_same_video_confirmed_resume_runtime.py, test_show_config_routing_contract.py, test_show_specific_podhome_key_contract.py, test_sunday_audio_path_guard_runtime.py, test_system_ffmpeg_runtime.py, test_video_master_rerun_progress_contract.py, test_workflow_contract_diagnostics.py.

## Approval-dependent recovery procedure
1. Review the two recovery PRs and checks. Approve their merge and paused deployment explicitly. Schedule Master requires no code change or redeployment.
2. Snapshot the named failed Audio job and Ripped request before the fix deployment; preserve all volumes, credentials, holds and existing queue records. Recheck global publishing false / Agape disabled.
3. Deploy only the approved Audio fix, then Ripped fix, one old worker fully stopped before each replacement. Keep publishing false; no credential or unrelated configuration changes.
4. Invoke existing Audio POST /jobs/b7d6bbd5-b48b-476a-b5c0-a551e53f6490/transcribe once. The exact existing failure permits reuse of its complete timed TXT, validating source ID and intervals against retained audio duration. No source download/new intake. Verify the existing authenticated ingest status acknowledges the same job, usable transcript artifacts and honest segment timing.
5. Use existing Telegram command /retry 94893455-5f48-46de-bdfb-642212ea19a5 once, without creating another source request. Existing cached media remains local; approved transcript artifact IDs are reused. Stop on missing cache or integrity failures rather than downloading a replacement.
6. Continue normal both-format candidate review without artificial caps or agent clip selections. Notify the owner through the existing Telegram interface. Follow existing human selection, per-clip rendering/audio/playback, batch creator and copy approval gates, then actual scheduled delivery verification.
7. Stop before all public publishing or real Instagram collaborator invitations. Any live test requires separate explicit authorization.

## Current checkpoint / remaining limit
Production remains at failed transcription; no candidates selected, rendered, website-approved or newly scheduled. Selection completion remains NOT STARTED, percentage N/A; operational acceptance batch remains BLOCKED. Offline tests and artifact validation do not mean production recovery succeeded or the full selected batch completed.

The remaining owner action is approval of the two reviewed fixes' merge and paused deployment, followed by recovery of these same named IDs. No new source URL, key replacement or selection on the owner's behalf is needed.

Reference: official OpenAI transcription API supports word and segment granularities with verbose_json:
https://developers.openai.com/api/reference/python/resources/audio/subresources/transcriptions/methods/create

