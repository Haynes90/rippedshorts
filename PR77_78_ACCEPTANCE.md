# Ripped Shorts PR #77 + #78 — Coordinated Acceptance

Date: 2026-10-31. **BLOCKED for acceptance / NOT READY for merge or deployment. Both PRs remain draft.**

[PR #77](https://github.com/Haynes90/rippedshorts/pull/77) is the source-intelligence base; [PR #78](https://github.com/Haynes90/rippedshorts/pull/78) depends on it. Reviewed heads were `6115d0a34ce40d63699f7e3e69d6d01d653b9dd5` and `2860973f00b0914b954048c2c0ae51f3932e9ebb`. Review corrections are committed on these same branches. No merge, application deployment, queue mutation, publication, credential change or request restart was performed.

## Acceptance results

| Gate | Result | Evidence |
|---|---|---|
| Four previously deferred HTTP tests | PASS | All four executed in authorized isolated local environment; 4 passed. They cover one creator section/copy gate, authenticated read-only status, frozen submitted captions and pre-review service-authenticated status. |
| Real Google Sheets topic/learning synchronization | PASS, isolated workbook | Exact SHA-verified proposed module ran in a temporary independent process/SQLite file using deployed Ripped service identity. Source/topic/candidate/feedback writes: 4; repeated sync: 0. All schema checks and row readbacks passed. |
| Learning changes actual future ranking/discovery inputs | PASS in isolated tests | Approval/rejection changes later ranking; edits alter future prompt context; brand/format scopes and novel candidates preserved. No real-world quality uplift claimed. |
| Website copy/creator/scheduler contracts | PASS in mocked regression | Actual handlers/handoff tested with isolated records and mocked downstream providers. Existing protected production reviews were not exercised or changed. |
| Real source availability | PASS | Retained Drive MP4: 207,383,479 bytes, SHA256 `45314424b1529f9d169f8d08c4fce31ddab10b17b122ed107c98458376b80a67`; 640x360, 30fps, approximately 3679.07 seconds. Existing transcript artifact retained. No YouTube reacquisition. |
| All five TDOG full exports | BLOCKED | Source-window diagnostics explain failure shapes but do not establish safe complete exports. All five retain their original failure/selection state. |
| Real vertical layout / speaker switching | PASS limited safety sample; NOT VERIFIED full acceptance | Real changing group scene now uses whole-source stable fallback, explicit framing-review flag and no inferred speaker. Actual automatic switches and real two/four-person fixtures remain unverified. |
| Full regression | FAIL | Final combined run: **355 passed, 23 failed, 2 skipped**. All remaining failures are baseline IDs. They have not been deleted, skipped or declared automatically safe. |

## Protected production evidence

Request `94893455-5f48-46de-bdfb-642212ea19a5`; Audio job `b7d6bbd5-b48b-476a-b5c0-a551e53f6490`; source `9mW0bhmUaw0`.

Read-only production query observed 38 request records, request status `awaiting_render_retry`, 11 proposed Shorts and 2 highlights, with Shorts indices 8/9/10 and highlights indices 0/1 selected. All five remained `render_failed`. State fingerprint at inspection: `09ed5ff5065dc33befb7a68acf231f1b59868cb72a9f8820761564f4bb7201dd`.

No command issued a write to the production database, intake, review, scheduling or job API. Isolated diagnostic files were written only under temporary directories. A final fingerprint reread was attempted, but the browser console stalled; do not interpret that unsuccessful reread as a new preservation verification. Selection/approval/scheduling mutations were neither issued nor authorized.

## Each retained failure: actual media diagnostics

Ten bounded 18-second edge ASR checks used the retained local source and deployed transcription credential, in an independent diagnostic process; no job was resumed. The 1,189-segment transcript was not regenerated. These are source-window checks, not replays of the missing historical failed exports. Original exact failing edge transcripts were not saved by the old code, so historical causation has a limit.

| Selected asset | Bounds (seconds) | Stored failure / diagnostic result | Disposition |
|---|---|---|---|
| Short 9 / index 8 | 2567.82–2620.22 | Stored token mismatch. Replay omitted leading “uh”; tail words matched. Word-timed tail margin only 0.02s. | Hold; do not silently delete fillers or move approved bounds. |
| Short 10 / index 9 | 2651.76–2715.76 | Stored token mismatch. Replay included “well” before the approved opening. Tail matched but provider returned segment timing. | Hold; segment endpoint fix alone cannot certify actual acoustic edge. |
| Short 11 / index 10 | 2791.34–2861.64 | Stored token mismatch. Replay omitted opening “um” and tail disfluency “for”; measured tail margin 0.02s. | Hold; strict guard remains effective; semantic/acoustic review needed. |
| Highlight 1 / index 0 | 2551.66–2916.66 | Stored token mismatch. Replay omitted opening “you’re”; tail matched, margin 0.02s. | Hold; cannot classify missing meaningful opening solely as harmless ASR variation. |
| Highlight 2 / index 1 | 3171.76–3624.04 | Stored <0.08s final-word margin. Replay matched both ends, returned word timing and 0.10s tail margin. | Original failure not reproduced by source-window replay; complete encoded export still unverified. No release. |

The earlier proposed correction is valid for segment-timed samples: require independent acoustic guard instead of pretending a segment endpoint is a measured word endpoint. It does **not** resolve all word-timed margins or token mismatches. Expected/heard diagnostics and durable ASR caching are preserved. No fuzzy acceptance or unapproved transcript/boundary rewrite was introduced.

## Actual framing and audio evidence

Inspected source frames at 60, 1200, 2655, 2800 and 3200 seconds. Footage includes single-camera views, three side-by-side participants, viewer-comment overlays, tickers and a changing two/three-person arrangement. The automatic detector failed stable-region consensus on the 64-second selected Short interval. This was a real acceptance finding, not a provider permission failure.

Corrections support balanced three-column visual separators when participant evidence is sufficient and prevent uncertain/changing groups from falling through to the legacy two-face crop. The conservative path preserves the complete source, declares zero layout confidence, emits `requires_framing_review=true`, and makes no active-speaker claim.

Two isolated 8-second real-source outputs were encoded and decoded at 1080x1920: a manually inspected whole-tile composition primitive and the actual automatic stable-group fallback. Audio RMS was approximately 0.0901 and 0.1238 respectively. Images were visually inspected locally. These short samples do not certify full-duration speech boundaries, every participant transition or subjective loudness throughout each selected clip. The source edge ASR checks independently recovered intelligible speech.

No verified real four-person fixture was available in this acceptance set. Two-person footage occurs during source layout changes, but a dedicated complete two-person active-speaker-switch fixture has not passed actual-media acceptance. Synthetic two/three/four/mixed-layout tests are not counted as substitutes. Four-person stable grid remains tested synthetically only. Human framing review remains necessary for the uncertain real group view.

Local retained validation media is intentionally not committed or uploaded to production. The original source and all five selections remain unchanged.

## Real Sheets integration method

Created a separate owner-owned [isolated acceptance workbook](https://docs.google.com/spreadsheets/d/1sytM66z0Nz6BE-XzrV3x68XYXsZp_56gb8RfNg-1tKY/edit), shared only with the already intended Railway service account for testing. No production workbook/tab was modified. Service-account direct spreadsheet creation initially returned HTTP403; owner creation and sharing resolved that test-environment limitation without replacing credentials or changing production access.

The existing SDK authenticated inside Railway. Proposed `production_intelligence.py` was retrieved by pinned PR commit, its normalized SHA256 verified as `e7785fdffaa7b504d1aaeb94ce4332d2e53ae9c6b97ca3bb096cb6f0750e0963`, and executed independently with a temporary database binding, no application startup, worker, job or production DB import. Actual module schema creation, key upserts and readbacks passed for all four record kinds; repeated sync wrote zero. Numeric readbacks use formatted strings, which the module explicitly normalizes for comparison. Mocked uncertain-write retry/schema-conflict tests also pass. This does not prove performance or quota resilience under future sustained production volume.

## Narrow corrections in this pass

- `durable_jobs.py`: close SQLite connections explicitly; reject unexpired same-replica reentry. Original release/resume and completed-handoff behavior retained.
- `telegram_intake.py`: reject exact duplicate candidate intervals produced by sentence expansion; affects future proposal validation only, never existing owner selections.
- `requirements.txt`: constrain existing OpenCV dependency below 5. The available 5.0 build lacks `CascadeClassifier`; real validation used isolated 4.11.0.86. No extra production package/service added.
- `camera_layout.py`: recognize balanced three-column layouts when evidence permits; preserve full source under changing/uncertain group layouts instead of unsafe two-person fallback; accurately label evidence.
- `tests/test_durable_jobs.py`: same-worker duplicate regression and explicit test-owned connection closure.
- `tests/test_caption_checkpoints.py`: keep extraction focused on copy resolution rather than unrelated workflow journaling; saved approved wording assertion retained.
- `tests/test_telegram_intake.py`: load real sentence helper for the AST fixture; original completion assertions now expose and cover duplicate expansion correction.
- `tests/test_camera_tiles.py`: three-column geometry, false-line rejection and uncertain-group safety regressions. Two-pixel inset expected by existing composition is checked, not removed.

## Baseline failure classification

The original combined run had 29 failures, all shared with unchanged base. Four connection-lifetime failures and two broken fixture cases now pass; the restored candidate fixture exposed a material duplicate-interval defect, also fixed. The 23 remaining failures remain visible. Production-relevant source-string assertions are not proof that runtime behavior passes; newer behavior tests provide the explicitly limited evidence below. A clean or formally accepted regression gate is still required before ready-for-review/merge status.

| Baseline failing test | Classification | Evidence / disposition |
|---|---|---|
| `test_missing_assets_are_delegated_to_audio_master` | Safety relevant; obsolete call-string assertion | Retained-source runtime regressions exercise authenticated upstream binding and reuse; exact former call spelling is no longer authoritative. |
| `test_legacy_handoff_applies_sheet_copy_after_generation` | FIXED test isolation | Extracted snippet extended into workflow journaling without creators fixture. Restored the copy-only extraction boundary; saved-caption assertion retained and passes. |
| `test_review_carries_origin_and_callback_without_reintake` | Safety relevant; contradictory static assertion | The intake URL occurs in the forbidden-callback list. The adjacent callback rejection test requires that same URL. No reintake is demonstrated by this failure. |
| `test_already_rendered_youtube_id_skips_gpt_selection` | Safety relevant; obsolete branch-string assertion | Cached analysis/artifact reuse and concurrent-analysis tests pass; live repeat-intake behavior is not certified by a string test. |
| `test_under_ten_reviewed_shorts_triggers_additional_search` | Obsolete quota contract | Owner-approved quality-over-quota specification supersedes forced minimum-ten search. Assertion retained as identified test debt. |
| `test_telegram_uses_short_not_candidate_label` | Presentation; obsolete string | Exact f-string changed. Not evidence of a broken render, approval or handoff. |
| `test_short_selection_uses_sheet_approval_learning` | Brand relevance; obsolete prompt phrase | Scripture-only phrase is not the TCB/TDOG rubric. Scoped approval/rejection learning changes future rank in executed tests. |
| `test_plain_link_starts_shorts_and_horizontal_topic_lanes` | Production relevant; obsolete branch-string assertion | Actual selector orchestration test proves simultaneous lane starts, unchanged standards and no automatic human decision. |
| `test_horizontal_segments_cover_full_timeline_without_overlap` | Obsolete selection architecture assertion | Semantic highlights replace arbitrary 480-second slicing; three-minute minimum retained. |
| `test_shorts_expand_to_complete_sentence_boundaries` | Production relevant; obsolete wording | Existing executable boundary/completion and retained transcript tests pass; actual five held cuts still require review. |
| `test_topic_coherence_precedes_eight_minute_reference` | Production relevant; obsolete wording | Full-source semantic map and exact topic coverage tests pass. No eight-minute maximum introduced. |
| `test_completed_handoff_is_exactly_once` | FIXED material resource defect | SQLite connection context committed but did not close. Explicit closing now releases handles; completed handoff assertion passes. |
| `test_only_one_live_owner_can_claim` | FIXED material concurrency/resource defect | Same replica could reenter its own unexpired lease. Live lease now rejects any claimant; new same-worker regression and original different-owner test pass. |
| `test_released_processing_job_can_resume` | FIXED material resource defect | Explicit close releases connection; released-action resumption assertion still passes. |
| `test_schema_includes_outbox` | FIXED test resource lifetime | Test-owned schema connection also needed explicit close on Windows. Outbox assertion retained. |
| `test_failed_approved_renders_block_schedule_handoff` | Obsolete and unsafe all-or-nothing contract | Owner explicitly authorized mixed-success batches. Existing batch/creator/handoff tests preserve failures while eligible approved successes proceed; old blocking sentence not reinstated. |
| `test_source_winner_is_logged_to_console_and_jsonl` | Operational diagnostics; obsolete storage assertion | Source winner is logged and recorded through existing workflow reliability ledger. A particular former JSONL filename is not the current authority. |
| `test_renderer_does_not_use_removed_mediapipe_package_attribute` | Production relevant; static import expectation | Old MediaPipe import is no longer used. Real media exposed an independent OpenCV 5 removal of CascadeClassifier; constrain OpenCV below 5 to preserve existing detector. |
| `test_renderer_has_opencv_and_center_crop_fallbacks` | Production relevant; static return-string expectation | Old return 0.5 text is absent. Actual group-layout fallback was unsafe for changing three-person scenes; now preserves full source with explicit framing-review evidence. |
| `test_prompt_requires_distinct_non_overlapping_complete_thoughts` | Production relevant; prompt moved to approved Doc | TCB/TDOG workflow actually calls non-overlap selector after completion validation. Original Doc standard is preserved; old main.py literal not restored. |
| `test_16_9_waits_for_explicit_shorts_confirmation` | Production relevant; obsolete button text | Shorts-first human approval remains; format analyses are concurrent as authorized. Prepared-highlights orchestration passes; no existing candidate selection changed. |
| `test_complete_candidate_validation` | FIXED material duplicate-candidate defect | AST fixture omitted sentence helper. Once restored, test exposed duplicate intervals after expansion. Reject duplicate normalized intervals; original one-valid/one-rejected assertions now pass. |
| `test_copy_reply_is_bound_to_exact_prompt_and_preserves_route` | Production relevant; Telegram source-text contract | Exact former helper/transaction wording changed. Website frozen-copy and actual creator/copy handoff tests pass; exact Telegram reply-route behavior still needs a dedicated behavioral replacement before a clean regression gate. |
| `test_copy_reply_updates_existing_prompt_instead_of_sending_status_spam` | Presentation/operational; source-text contract | Former notification/log strings changed. No assertion removed; Telegram notification UX remains a review item. |
| `test_edited_caption_stays_visible_as_final_reviewed_copy` | Production relevant; source-text contract | Executed caption checkpoint and HTTP frozen-caption tests pass. Exact legacy Telegram card rendering is not established by these newer tests. |
| `test_schedule_handoff_prefers_durable_caption_checkpoint` | Production relevant; source-text contract | Checkpoint resolution and actual handoff preserve human copy in executed tests; old inline provenance marker moved. Assertion retained. |
| `test_short_caption_keeps_hashtags_separate_and_clean` | Production relevant; prompt-string contract | Current copy/structured-hashtag checks and approved-copy handoff run; missing former Hook phrase alone does not prove altered human copy. Full old assertion remains failing. |
| `test_webhook_monitor_distinguishes_stale_errors_and_group_privacy` | Operational; source-text contract | Old notification wording absent. No webhook or notification production changes made; operational string expectation remains unmodernized. |
| `test_short_learning_uses_weighted_wins_and_losses` | Production relevant; obsolete prompt wording | Executed scoped history tests prove approvals/rejections/edits affect future recommendations without suppressing new categories or changing approval standards. |

## Executed final tests

Counts overlap and must not be summed. Python: workspace venv. All external test networking blocked by `validation_network`; TEMP/TMP/DATA_DIR and basetemp were isolated. Four HTTP cases and final combined suite used authorized local execution outside the Windows sandbox.

| Run | Passed | Failed | Skipped |
|---|---:|---:|---:|
| Four deferred HTTP integration cases | 4 | 0 | 0 |
| Durable lease focused run | 6 | 0 | 0 |
| Final source-only affected suite | 136 | 0 | 0 |
| Final combined suite, OpenCV4 | 355 | 23 | 2 |

Exact final commands (environment variables and isolated paths described above):

```text
python -m pytest -p validation_network tests/test_creator_media_workflow.py::test_one_website_section_and_server_gate_preserve_copy tests/test_workflow_test_status.py::test_authenticated_status_get_is_read_only_and_does_not_claim_live_publication tests/test_workflow_test_status.py::test_submitted_caption_cannot_change_after_batch_handoff tests/test_workflow_test_status.py::test_service_authenticated_status_works_before_review_session_and_preserves_database -q --tb=short --basetemp=.test-tmp/http-acceptance --junitxml=http-acceptance-results.xml
python -m pytest -p validation_network tests/test_durable_jobs.py -q --tb=short --basetemp=.test-tmp/lease-acceptance --junitxml=lease-acceptance-results.xml
python -m pytest -p validation_network tests/test_production_intelligence.py tests/test_intelligence_orchestration.py tests/test_render_edge_evidence.py tests/test_retained_source_verification.py tests/test_transcript_timing_recovery.py tests/test_clip_completion.py tests/test_batch_progress.py tests/test_caption_review_web.py tests/test_selection_specs.py tests/test_durable_jobs.py tests/test_caption_checkpoints.py tests/test_telegram_intake.py::TelegramParsingTests::test_complete_candidate_validation -q --tb=short --basetemp=.test-tmp/source-final --junitxml=source-final-acceptance-results.xml
python -m pytest -p validation_network tests -q --tb=short --basetemp=.test-tmp/final-v3 --junitxml=acceptance-final-v3-results.xml
```

Two stored-render tests skipped because FFprobe is absent locally. Earlier branch CI passed those tests, but new-head CI must be checked separately; local skips are not claimed as new-head CI passes. Warnings concern FastAPI lifespan/deprecated httpx/anyio interfaces, not suppressed failures. Intermediate acceptance runs were 351/26/2 and 354/23/2 before the final corrections; final counts above supersede them.

## Required next action

Keep both PRs draft. Do not merge, deploy or resume the protected request. Review the five named cut boundaries and safe-group preview before authorizing any boundary/selection change or individual production retry. Provide or identify verified real two- and four-person/speaker-switch fixtures to close the missing layout evidence. Complete behavioral disposition of the remaining production-relevant baseline assertions before claiming a clean regression gate. No new general architecture work is required.

Any eventual merge order is #77 followed by #78 after acceptance and separate owner approval. Existing production publishing/Agape settings, account mappings, dates, copy, collaborator approvals and scheduler configuration were not changed. No real social-media request or invitation was sent.
