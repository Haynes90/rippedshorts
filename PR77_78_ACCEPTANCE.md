# PR #77 + #78 — Focused Acceptance Completion

Date: 2026-10-31 (owner timezone). **NEEDS ATTENTION — keep both PRs draft/unmerged. No production deployment authorized or performed.**

[PR #77](https://github.com/Haynes90/rippedshorts/pull/77) is the source-intelligence base; [PR #78](https://github.com/Haynes90/rippedshorts/pull/78) is stacked on it. Starting reviewed commits: `e59a29888b093446b17d0e27e80ea913ea054684` and `a8dda47c78aaf1f6ad1b799c6a70761dcde1f2e9`. This pass changes tests and acceptance evidence only; no new application architecture or production behavior.

## Current gates

| Area | Status | Evidence |
|---|---|---|
| Source-only regression | PASS | 360 passed, 0 failed, 2 local FFprobe-dependent skips. |
| Combined regression | PASS with local media-tool limitation | 378 passed, 0 failed, 2 local FFprobe-dependent skips; four HTTP tests included. |
| Remaining 23 legacy failures | RESOLVED with behavior tests | All 23 definitions replaced or explicitly renamed for current approved contracts; no blanket skip/xfail or discarded safety requirement. Mapping below. |
| Google Sheets topic/learning sync | PASS isolated, prior pass retained | Actual Railway identity: four record kinds synchronized/read back; repeat sync zero writes. No production workbook mutation. |
| Five selected cut comparisons | READY FOR OWNER REVIEW, not approved | Twenty full original/proposed/context videos rendered locally, decoded, audio checked and privately uploaded after explicit owner permission. Five original selections unchanged. |
| Two-person real framing | PARTIAL | Verified actual two-person interval; actual pipeline preserves whole source in stable-group fallback, 1080x1920, non-silent audio. High-confidence speaker switching remains unverified. |
| Three-person/changing group real framing | PARTIAL | Earlier real source/whole-tile primitive and automatic conservative fallback verified. No claim that low-confidence group fallback is an optimized active-speaker layout. |
| Four-person real framing | NOT VERIFIED | No suitable verified fixture obtained. Additional retained source exceeded connector's 256MiB materialization cap; native Drive player/download attempt did not yield a usable local file. Synthetic grid evidence is not substituted. |
| Full five-clip final export acceptance | WAITING FOR OWNER / NOT VERIFIED | Boundary proposals are diagnostic copies, not released successful renders or scheduled deliveries. Production job remains held and was not resumed. |
| Merge readiness | NEEDS ATTENTION | Regression gate now green, but owner boundary decisions and outstanding real framing/handoff acceptance remain. Keep draft. |

## Five original selections and proposed alternatives

Protected request `94893455-5f48-46de-bdfb-642212ea19a5`, source `9mW0bhmUaw0`, Audio job `b7d6bbd5-b48b-476a-b5c0-a551e53f6490`. Original five selections, all production requests, website approvals, captions/titles, creator revisions, scheduler records, holds and account routing were not written. No intake, resume, social publishing or native collaboration invitation was issued.

Each clip has four owner-review files: original full cut, proposed full cut, opening context and ending context, with retained source audio. A private Drive START HERE index supplies the links, nearby segment transcript and reasoning. Owner explicitly approved these private diagnostic uploads; folder/file metadata confirms owner-only access and no duplicate/missing comparison files. The production review website was not changed: attaching alternative assets to its protected request records would require production writes or deployment, neither authorized. The MP4s are privately downloadable and local decoding/audio playback is verified. In the automated browser, the Drive player remained loading; remote streaming playback is NOT VERIFIED and must not be described as completed website playback acceptance. The local index.html offers side-by-side playable comparisons after opening/downloading the review set.

| Existing selection | Original seconds | Proposed review alternative | Reason / required owner decision |
|---|---|---|---|
| Short 9 / index 8 | 2567.82–2620.22 | 2567.82–2641.04 (73.22s) | Original introduces the collard project immediately before ending. Alternative includes its harvesting method and payoff. Listen and retain original or approve extension; segment join still needs acoustic judgment. |
| Short 10 / index 9 | 2651.76–2715.76 | 2646.12–2716.16 (70.04s) | Includes complete preceding seed-buying question and 0.40s after stored sentence end, before next timed response. ASR adding “well” alone is not grounds to change it. |
| Short 11 / index 10 | 2791.34–2861.64 | 2848.14–2916.66 (68.52s) | Original ends during continuing stratification disagreement. Alternative centers pushback and variety-dependent qualifications. Owner must assess its opening context. Full original question through all qualifications exceeds 90s; no duration rule bypassed. |
| Highlight 1 / index 0 | 2551.66–2916.66 | 2544.64–2916.66 (372.02s) | Preceding transition contains the lead-in “Lynn, I'm glad” before “you're talking”. Added setup is an editorial alternative, not a claim that ASR omission proves a clipped word. |
| Highlight 2 / index 1 | 3171.76–3624.04 | 3171.76–3621.18 (449.42s) | Alternative ends after the final substantive reference, before closing housekeeping. Proposed endpoint lies inside a segment-timed gap; it is not fabricated word precision. Original remains an option if its audible ending is complete. |

**All five dispositions: OWNER REVIEW REQUIRED. No proposal is preselected, applied or scheduled.** The original token/margin failures are retained as diagnostics; ASR differences alone are not proof of unsafe cuts. Proposed files preserve the original source/selection identity in their manifest, but are separate diagnostic artifacts with no production asset/delivery identity.

Twenty comparison MP4s were independently decoded, had nonempty media and non-silent source audio checked from their opening samples. Full encoded durations are recorded in local `media-validation.json`; actual audible content is available for owner playback throughout every full cut. This is not a claim that an agent perceptually listened to every second or that all proposed acoustic/semantic edges have passed the production validator. Source-view video preserves horizontal camera context for editorial comparison, not final vertical composition approval.

## Real-media limits and next fixture

Retained source MP4 SHA256 remains `45314424b1529f9d169f8d08c4fce31ddab10b17b122ed107c98458376b80a67`. Source frames and actual local automatic renders were inspected. A stable two-person side-by-side segment exists at 2664–2674s; six inspected frames confirm the tile arrangement. The actual audio RMS analysis supplies 200 samples. The detector still chooses explicit `uncertain_group` / whole-source framing under imperfect face/separator evidence; it does not invent a named or active speaker. Output is 1080x1920 with non-silent audio. This preserves full camera regions, but is less legible than confidently separated panels and is flagged for human framing review.

Earlier 2660–2675s diagnostic spans a three-to-two layout change and must not be labelled a stable two-person fixture. Three-person real footage at approximately 2655/2800/3200s and actual safe-group output remain available locally. Actual high-confidence speaker handoff tracking is **not verified** by these samples; ambiguous behavior correctly retains a stable group view. Synthetic correlated-mouth handoff tests are not real-speaker acceptance.

Targeted TCB/TDOG Drive searches and retained-source listing did not yield a verified four-person test. `gKnDkM_b1CU-source.mp4` exists as retained media, but its 417,512,241 bytes exceed the connector's 268,435,456-byte cap; the fallback native Drive player remained loading and its download did not produce a usable local artifact. That is a validation access/tooling limitation, not proof that the production media is inaccessible. Owner can identify an accessible four-person source/excerpt (preferably below 256MiB) and a known speaker handoff; do not create a new intake or restart any old job to obtain it.

## Justified legacy replacements

The previous 23 failures were obsolete literals, contradictory callback assertions, stale quotas/full-timeline rules or unexercised behavior. New tests execute real application functions with isolated SQLite records and mocked external providers. Original test count is retained; obsolete names are explicitly mapped. The six other issues in the original 29 were fixed in the prior pass (resource lifetime, same-worker lease and duplicate-normalization/fixture issues). No production requirement is waived because it failed on base.

| Old failure | Replacement | Preserved requirement / justification |
|---|---|---|
| `test_audio_master_ingestion.py::test_missing_assets_are_delegated_to_audio_master` | `test_missing_assets_are_delegated_to_audio_master` | Real ingestion client waits for BOTH retained video and transcript; asserts one authenticated intake and polling rather than the old literal call spelling. |
| `test_clipmaster_review.py::test_review_carries_origin_and_callback_without_reintake` | `test_review_carries_origin_and_callback_without_reintake` | Executes completion callback twice and verifies origin identity, sections, exact completion URL and single provider request; forbidden-intake URL remains a separate safety test. |
| `test_concurrent_render_safety.py::test_already_rendered_youtube_id_skips_gpt_selection` | `test_saved_source_analysis_does_not_repeat_model_selection` | Executes immutable-source analysis twice and asserts preserved result/asset identity with one model call. Explicit rerip/reuse and retained-source tests remain intact. |
| `test_concurrent_render_safety.py::test_under_ten_reviewed_shorts_triggers_additional_search` | `test_quality_selector_does_not_pad_to_a_review_quota` | Owner quality-over-quota policy supersedes forced ten-count replenishment; actual prompt/selector must return one strong proposal or zero without padding. |
| `test_concurrent_render_safety.py::test_telegram_uses_short_not_candidate_label` | `test_telegram_uses_short_not_candidate_label` | Builds actual review payload; verifies original candidate number 9, request/index-bound approval callback and no automatic decisions. |
| `test_concurrent_render_safety.py::test_short_selection_uses_sheet_approval_learning` | `test_short_selection_uses_sheet_approval_learning` | Executes ledger prompt over mocked approval/rejection rows; both examples and complete-thought/payoff standards must survive. |
| `test_concurrent_render_safety.py::test_plain_link_starts_shorts_and_horizontal_topic_lanes` | `test_plain_link_starts_shorts_and_horizontal_topic_lanes` | Checks actual plain-link mode and simultaneous lane starts using barriers; separate existing full-orchestration tests remain. |
| `test_concurrent_render_safety.py::test_horizontal_segments_cover_full_timeline_without_overlap` | `test_semantic_highlights_leave_weak_gaps_and_keep_minimum` | Owner semantic selection supersedes arbitrary full-timeline slices; actual builder preserves selected substantive intervals, weak gaps and 180-second floor. |
| `test_concurrent_render_safety.py::test_shorts_expand_to_complete_sentence_boundaries` | `test_shorts_expand_to_complete_sentence_boundaries` | Executes sentence expansion/validation from mid-thought proposal and verifies exact complete setup/payoff text and bounds. |
| `test_concurrent_render_safety.py::test_topic_coherence_precedes_eight_minute_reference` | `test_topic_coherence_precedes_eight_minute_reference` | Executes 900-second coherent highlight without a forced maximum; rejects 100-second fallback. |
| `test_missing_render_recovery_and_google_identity.py::test_failed_approved_renders_block_schedule_handoff` | `test_individual_failure_does_not_block_successful_handoff` | Owner authorized mixed success. Executes actual completion gate with one success/one failure, asserts exactly one handoff submission, retained failed status and correct counts. |
| `test_rapidapi_source_race.py::test_source_winner_is_logged_to_console_and_jsonl` | `test_source_winner_is_logged_and_durably_recorded` | Executes source-winner recording, checks console event and durable ledger provider/profile/source bytes. JSONL filename was obsolete; diagnostics still required. |
| `test_render_fallback.py::test_renderer_does_not_use_removed_mediapipe_package_attribute` | `test_opencv_detector_loads_and_missing_model_falls_back` | Loads actual OpenCV4 cascade and tests absent model fallback; no obsolete MediaPipe import dependency. |
| `test_render_fallback.py::test_renderer_has_opencv_and_center_crop_fallbacks` | `test_renderer_has_opencv_and_center_crop_fallbacks` | Runs no-motion/no-frames camera fallback; verifies center track and capture release. No dependency on the stacked camera module in source-only PR. |
| `test_source_ingestion.py::test_prompt_requires_distinct_non_overlapping_complete_thoughts` | `test_prompt_requires_distinct_non_overlapping_complete_thoughts` | Runs current non-overlap selector against overlapping/distinct complete proposals; approved Doc remains the prompt authority. |
| `test_telegram_intake.py::test_16_9_waits_for_explicit_shorts_confirmation` | `test_16_9_waits_for_explicit_shorts_confirmation` | Builds real control payload before/after confirmation; confirms request-bound Continue control and no inferred human decision. |
| `test_telegram_intake.py::test_copy_reply_is_bound_to_exact_prompt_and_preserves_route` | `test_copy_reply_is_bound_to_exact_prompt_and_preserves_route` | Executes actual Telegram reply handler against two isolated pending requests. Only the exact prompt target changes; route preserved; checkpoint sees committed edited copy. |
| `test_telegram_intake.py::test_copy_reply_updates_existing_prompt_instead_of_sending_status_spam` | `test_copy_reply_updates_existing_prompt_instead_of_sending_status_spam` | Executes actual edit handler and review refresh; existing cards edited, no new sendMessage on success, exact original prompt reused. |
| `test_telegram_intake.py::test_edited_caption_stays_visible_as_final_reviewed_copy` | `test_edited_caption_stays_visible_as_final_reviewed_copy` | Executes edit and actual review payload; exact multiline human text remains visible at same index and marked EDITED FINAL COPY. |
| `test_telegram_intake.py::test_schedule_handoff_prefers_durable_caption_checkpoint` | `test_schedule_handoff_prefers_durable_caption_checkpoint` | Executes checkpoint reader with edited AI/owner copy and wrong-request row; exact human wording wins and request isolation holds. Actual handoff tests remain. |
| `test_telegram_intake.py::test_short_caption_keeps_hashtags_separate_and_clean` | `test_short_caption_keeps_hashtags_separate_and_clean` | Executes edit/preview and serialized hashtag normalization. Human copy unchanged, hashtags remain separate, no generated Hook label added. |
| `test_telegram_intake.py::test_webhook_monitor_distinguishes_stale_errors_and_group_privacy` | `test_webhook_monitor_distinguishes_stale_errors_and_group_privacy` | Executes webhook monitor using mocked Telegram responses: historical error informational, recent error warning, privacy limitation warning; no webhook changes when URL matches. |
| `test_weighted_learning_and_fresh_start_menu.py::test_short_learning_uses_weighted_wins_and_losses` | `test_short_learning_uses_weighted_wins_and_losses` | Executes weighted history prompt and asserts positive/negative category effects, exact same-source/recency weights and no copying prior timestamps. |

## Test execution

Final complete suites used workspace Python, isolated OpenCV 4.11.0.86, blocked external sockets via `validation_network`, isolated TEMP/TMP/DATA_DIR, and separate basetemp per branch. Authorized execution outside the Windows sandbox allowed the HTTP/TestClient cases. No application worker startup or real publishing API was permitted.

```text
# Combined branch
python -m pytest -p validation_network tests -q --tb=short --basetemp=.test-tmp/review-final-0 --junitxml=review-final-results.xml
# Source-only branch
python -m pytest -p validation_network tests -q --tb=short --basetemp=.test-tmp/review-final-1 --junitxml=review-final-results.xml
```

Combined: 378 passed, 0 failed, 2 skipped; source-only: 360 passed, 0 failed, 2 skipped. Two subtests additionally passed in each run. Four deprecation warnings concern FastAPI/httpx/anyio interfaces; warnings are not failures. The two skips require FFmpeg plus FFprobe on PATH; local installed FFmpeg encoded/decoded real media independently, but FFprobe is absent. New-head GitHub stored-render result is checked separately after committing; prior-head green checks are not relabelled as final-head results. Focused legacy affected suite also executed: 113 passed, 0 failed. Final counts overlap; do not sum them.

## Exact deployment safeguards — not executed

1. Complete the outstanding media/owner acceptance and obtain separate merge and paused-deployment authorization. Keep both PRs draft until then. Review the final stacked diff; merge #77 before #78 only with approved checks and no conflicts.
2. Before any authorized production change, confirm Railway automatic deployment is disabled. Record exact existing deployed SHA, replica count, persistent mount and redacted configuration. No credential or destination-map replacement.
3. Take consistent backups of SQLite state (including jobs, approvals, review history, held failures, outbox/receipts and source bindings), retained artifacts and existing queue/Sheet dates/copy. Verify the backup can be read independently. Do not reset or bulk-reschedule rows.
4. Confirm Schedule Master remains `PUBLISHING_ENABLED=false` and Agape remains disabled. Preserve all platform/brand gates, TCB/TDOG shared destinations, audit/reconciliation intervals and pending receipts. No Instagram invitation/publication may be sent.
5. Preserve exactly one Ripped worker/replica and its existing persistent storage. Stop the old worker and verify full termination before replacement starts. Deploy only the specifically approved Ripped merged SHA; Audio Master/Schedule Master redeployment is outside these PRs.
6. Confirm the protected TDOG request is still in its human-waiting `awaiting_render_retry` state. The existing watchdog skips awaiting/review states; **do not change it to `render_recovery`, `awaiting_render_completion` or machine-owned processing during deployment**, which can trigger automatic work. Preserve all selected indices and prior diagnostics. No resume/retry is authorized by these instructions.
7. Verify deployed SHA, health/readiness, persistent state, authenticated read-only status, review links and unchanged protected selected bounds/copy/creator revisions. Compare scheduled dates, queue/outbox counts and receipt IDs with the backup. Stop on unexplained mutations, overlap, missing storage or unexpected publishing.
8. Only after separate named boundary/retry approval, process the original request's individually approved failed clips without new intake or reselection. Validate complete exports, audio, intended framing and owner copy/creator approvals before scheduling. Preserve successful siblings and all failed clip diagnostics. Keep publication paused.
9. Rollback, if authorized/needed: stop the replacement fully, deploy the last known good SHA with the same paused gates and persistent volume, verify identities and read-only integrity. Restore state only after explicit data-restoration approval; do not blindly overwrite newer approvals/receipts from a backup.

## Next owner action

Review the private START HERE comparison and decide the five named boundaries; no new general implementation authorization is needed. Identify a suitable four-person fixture and known actual speaker handoff for the remaining media acceptance. PRs remain draft and unmerged; no deployment/public publishing, protected boundary change, new intake or TDOG restart occurred.
