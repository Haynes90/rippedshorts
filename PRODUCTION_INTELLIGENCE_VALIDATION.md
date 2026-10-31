# Ripped Shorts — TCB/TDOG production-intelligence evidence

Date: 2026-10-30. Status: **IMPLEMENTED FOR REVIEW; ACCEPTANCE INCOMPLETE; NOT READY FOR DEPLOYMENT.**

Base: [38a2413f2756154958233d511f9de8661c6dddd7](https://github.com/Haynes90/rippedshorts/commit/38a2413f2756154958233d511f9de8661c6dddd7).

Review groups:

1. `feat/source-intelligence-20261030`: source maps, scoped learning, independent format analysis, retained-source/render-validator corrections, review feedback and status.
2. `feat/camera-tile-composition-20261030`: stacked on group 1; conservative camera-region composition and framing evidence. See CAMERA_TILE_VALIDATION.md on that branch.

Neither branch is authorized for merge or deployment. Audio Master and Schedule Master code, destination mappings, credentials, publishing flags and Agape configuration were not changed. No production queue, approvals, source artifacts, dates, receipts, candidates or selections were written. The existing TDOG request was not resumed or restarted.

## 1. Existing implementation reused

Classification describes the inspected base, not a claim of completed live acceptance. Class 1 = already implemented and tested; 2 = implemented but defective/incomplete; 3 = specified but not implemented; 4 = missing additive capability. Existing tests may use simulated providers.

| Requested surface/capability | Base class | Result and evidence |
|---|---|---|
| Audio/source ingestion and source verification | 2 | Reused Audio job/status GET, retained Drive artifacts, checksum/stream/duration checks and transcript conversion. Corrected ordinary retry after expired local cache and loss of top-level measured words. Retained-source regressions cover the real process path. |
| Full-source semantic topic map | 4 | Added versioned maps in the existing SQLite DB. Full transcript coverage, exact segment references, hierarchy, related topics and moment references validated before selection. No arbitrary time slicing or approval requirement for a Short's parent highlight. |
| Google Sheets topic intelligence | 4 | Added retryable projections into four dedicated tabs; existing schemas untouched. SQLite remains authoritative. Schema conflict, uncertain write and readback tested with a simulated Sheets API. Live topic write/readback remains NOT VERIFIED. |
| Adaptive learning | 2 | Existing weighted Shorts ledger and quick edits retained. Added show/brand/format/topic-scoped evidence, explicit edits/favorites, conservative weighted reranking and prompt conditioning. Tests demonstrate a later recommendation actually changes. Confidence is a heuristic, not an empirically calibrated probability or owner instruction. |
| 9:16 selection/rubric | 2 | Reused full-transcript selector and complete-candidate validator. Removed forced quotas for the new workflow, normalized the approved Doc's `clips`/`score_total` output, kept explanations and scoring fields, and excluded automatic overlap. No existing candidate set regenerated. |
| 16:9 substantial explanations | 2 | Existing semantic highlight builder retained. Both normal/fallback passes enforce at least 180 seconds; no maximum. Correct editable Doc identified, with packaged fallback. |
| Concurrent discovery, ordered approval | 3 | Shared map precedes two independent leased/cached analyses. Existing Telegram Shorts-first confirmation presents cached highlights afterward. Format failure preserves its sibling. Tests use real orchestration with simulated model responses and simultaneous-start barriers. |
| Two-person mouth/audio tracking | 1 | Existing active_speaker.py implementation and fallback retained. Its mixed-audio evidence estimates spatial activity, not named identity. |
| Three/four-person and mixed camera layouts | 3 | Camera branch adds conservative visual-separator/face-region detection, stable spatial tracks and whole-tile graphs. Synthetic media passes; real program layout/speaker accuracy remains NOT VERIFIED. |
| Completion and acoustic boundaries | 2 | Existing semantic review, export ASR, acoustic edge guards, audio-energy checks and output stream/duration checks retained. Corrected segment endpoint versus measured word endpoint handling. Mismatches remain held. |
| Individual render failures | 1 | Existing per-clip failure isolation and individual retry retained. Failure evidence made more useful and export ASR cache retained outside temporary render folders. No selected clip silently discarded. |
| Website playback, copy/creator approvals | 1/2 | Existing request-scoped URL, preview, editable copy and single batch creator section retained. Added per-clip failures/evidence, explicit favorite/hold/feedback fields and scheduled-delivery display. Direct handler tests pass; four HTTP integration tests need rerun outside the restricted environment. |
| Batch accounting and scheduler handoff | 1 | Existing selected-clip denominator, receipt/idempotency protections, human approvals and brand gates retained. New owner holds block handoff and success counting. Topic/candidate/asset/delivery relations recorded without changing downstream scheduling. |
| Optional human framing corrections | 3 | No new tile-position editor implemented. Framing remains inspectable; ambiguous cases retain conservative views/fallbacks. This must not be described as a finished manual framing editor. |

## 2. Verified configuration and document inspection

Read-only inspection of the existing [Podcast workbook](https://docs.google.com/spreadsheets/d/14VruBxjaaE9DyPSdBidMeuPHew3nHUC5sRNGXtLCsis/edit) confirmed Show Config, Ripped Shorts, Caption Learning, Chapter Definitions, Workflow Jobs, source repositories and schedule/outbox tabs. Header/sentinel reads were bounded. Existing chapter definitions belong to podcast/service contracts and were not repurposed or changed.

The owner confirmed TCB and TDOG currently use the same configured TCB publishing destinations. This change retains that contract and TDOG's confirmed recurring-host defaults. It does not create a TDOG publishing account. Live platform credentials/account permissions were not revalidated in this assignment.

Both original guides were found and read:

- [9:16 Social Clip Selection Prompt](https://docs.google.com/document/d/1HLTSueOOClqAIVMdakiL9sT2-03xvKmp--2pVXiCszM/edit): quality over quota; truthful hook, complete payoff, no automatic overlapping cuts, richer scoring/output contract.
- [16:9 Highlight Prompt and Contract](https://docs.google.com/document/d/1QyDKGKJscc107U38QsWtLP6fhqv8VPZhdoSwIC-bQdM/edit): at least three minutes, no maximum, semantic coherence and no forced count. Its old instruction to delay analysis until Shorts confirmation is superseded by the owner's current explicit concurrent-analysis directive. Approval order remains unchanged. The Doc itself was not edited.

The editable 9:16 loader already exists. The correct 16:9 loader now uses the verified reference (optional `RIPPED_16X9_SELECTION_DOC` override) and a packaged copy if service access fails. Owner-connected Doc access succeeded; Railway service-identity access to both Docs is NOT VERIFIED in this pass. Earlier 9:16 service access failed and used its packaged approved fallback; this was not a transcription authentication failure.

## 3. Actual TDOG failure findings

Request: `94893455-5f48-46de-bdfb-642212ea19a5`.
Audio job: `b7d6bbd5-b48b-476a-b5c0-a551e53f6490`.
Source: `9mW0bhmUaw0`.

Read-only runtime inspection observed 11 proposed Shorts, two highlights, three selected Shorts and two selected highlights. All five selected clips were marked render_failed. The state was awaiting_render_retry. Three Shorts and one highlight reported `Actual exported speech does not match approved boundary words`; the other highlight reported `Export ends too close to speech; adjust the approved boundary`.

The retained source MP4 was present (207,383,479 bytes) and the request retained authenticated-source verification evidence. The successful prior source/transcript recovery and its 1,189 timed segments were preserved; they were not repeated or recounted as newly executed transcription tests.

Confirmed code defects:

1. The export validator checked a segment's final timestamp as if it were a measured final word timestamp, even after an independent acoustic edge guard. Corrected: segment timing requires the acoustic guard; only measured words use the word margin. Tests cover acceptance with measured quiet, rejection without quiet and unchanged measured-word safety.
2. Export ASR cache lived beside temporary output and disappeared when that folder was removed. It now persists under the existing data volume's speech-evidence directory.
3. Boundary mismatch exceptions omitted expected/heard evidence. Sanitized token/timing diagnostics now persist per failed clip. Exact speech mismatches still block; there is no fuzzy acceptance, automatic trimming or silent transcript replacement.
4. Ordinary source retries with an expired local file could fail before Drive recovery; measured word evidence could be dropped by the retained path. Both corrected and covered.
5. The fallback highlight pass allowed 90-second clips, contrary to the three-minute contract. Corrected for future discovery; no approved existing bounds were rewritten.
6. The approved Shorts Doc uses `clips` and `score_total`, while the old adapter assumed `segments` and `score`. Both forms now normalize explicitly; missing lists fail validation rather than masquerading as no qualifying content.

**Causation limit:** the old failed exports and their provider edge transcripts were not retained, so this pass cannot prove whether each TDOG mismatch was ASR variation or an actually unsafe cut. The validator defect reproduces the relevant failure shape; its responsibility for the fifth real clip is not proven without replay. No failed real clip has been declared fixed or released. The real batch has not passed acceptance or acquired newly verified scheduled deliveries in this assignment.

## 4. Durable contracts and safety

- Existing SQLite database only: additive `intelligence_records`, `source_analyses` and `analysis_failures` tables. No separate database or service.
- Source map IDs bind brand, source, transcript revision and analysis version. Topic and candidate IDs are stable for that revision. Original proposal rank/score, transcript references and provenance remain available.
- Atomic topic-to-candidate link updates retain both parallel formats. Review events connect candidate → human decision → asset → acknowledged delivery IDs/dates/destinations where supplied by Schedule Master.
- Full transcript bodies are not repeated in topic/candidate Sheet rows. Copy revisions and full runtime state stay in SQLite; projection rows carry references and compact evidence.
- Existing request, clip/asset IDs, captions, titles, collaborators, held retries and scheduling rules remain authoritative. Source learning never edits approved copy or approves creators.
- Analysis leases protect duplicate work; completed results are reused. Explicit bounded failed-analysis retry preserves attempt evidence and completed sibling results. A failed format is not reported as an empty successful analysis.
- Sheets synchronization checks exact dedicated headers, uses record-key upserts, RAW values and readbacks, and recovers an uncertain write by reading the key before another write. A single writer/replica remains required. Existing status executor/watchdog handles coalesced retries; no new recurring worker.
- Public posting and native collaborator invitations are not part of internal completion. Existing invitation-submission versus recipient-acceptance distinctions remain unchanged.

Automatic learning from a future public publication is not live-verified here. The ledger can preserve received scheduler/outcome evidence; no new live publication or publication-outcome callback was exercised.

## 5. Tests actually executed

Evidence inventory: PRODUCTION_INTELLIGENCE_EVIDENCE.json. Counts below are separate runs/subsets and must not be added together.

| Run | Passed | Failed | Skipped/deferred |
|---|---:|---:|---|
| Unchanged base regression | 297 | 30 | 2 skipped |
| Combined implementation regression | 341 | 29 | 2 skipped; 4 HTTP tests deselected |
| Source-only review branch focused suite | 120 | 0 | None |
| Camera branch focused suite | 15 | 0 | None |

All 29 failing combined test IDs also fail on the unchanged base. Four are Windows SQLite-file teardown failures; the rest are source-text assertions or isolated helper-fixture failures. One directly affected quota test was replaced with a prompt behavior check. No unrelated assertion deletion or production code change was made to turn the full suite green. Baseline failure is not proof that a behavior is safe; the full suite remains failing.

Two existing real-render tests require both FFmpeg and FFprobe on PATH and were skipped. The camera suite separately executed FFmpeg through the available imageio binary, decoded actual synthetic outputs, checked dimensions/panel pixels and measured audible nonzero audio energy. Those synthetic results are not real podcast acceptance.

Four final HTTP tests were deferred because the Windows sandbox blocks their event-loop/TestClient operations. Automatic approval review could not authorize the requested outside-sandbox run because its usage limit was reached; that final escalated action did not execute. Pure handler tests inside the sandbox additionally verified feedback/copy persistence. Do not claim final HTTP integration was rerun successfully.

Commands (Python is the workspace test virtual environment; PYTHONUTF8=1, TEMP/TMP/DATA_DIR use isolated workspace directories; `validation_network` blocks external connections):

```
python -m pytest -p validation_network tests -q --tb=no --junitxml=baseline-results.xml
python -m pytest -p validation_network tests -q --tb=no -k 'not test_one_website_section_and_server_gate_preserve_copy and not test_authenticated_status_get_is_read_only_and_does_not_claim_live_publication and not test_submitted_caption_cannot_change_after_batch_handoff and not test_service_authenticated_status_works_before_review_session_and_preserves_database' --junitxml=intelligence-results.xml
python -m pytest -p validation_network tests/test_production_intelligence.py tests/test_intelligence_orchestration.py tests/test_render_edge_evidence.py tests/test_retained_source_verification.py tests/test_transcript_timing_recovery.py tests/test_clip_completion.py tests/test_batch_progress.py tests/test_caption_review_web.py tests/test_selection_specs.py -q --junitxml=source-results.xml
python -m pytest -p validation_network tests/test_camera_tiles.py -q --junitxml=camera-results.xml
```

## 6. Acceptance gaps and deployment prerequisites

| Gate | Status | Remaining work |
|---|---|---|
| Source-map persistence and related IDs | PASS in isolated tests | Verify semantic quality on retained real sources. |
| Scoped learning affects later discovery/ranking | PASS in isolated tests | Owner evaluation on real later selections; no measured quality-improvement claim yet. |
| Parallel analysis / sequential approval | PASS in isolated orchestration | Exercise both real model analyses and Telegram stages after separately authorized deployment. |
| Topic Sheets schema/write/readback | PASS simulated; NOT VERIFIED live | Isolated live workbook roundtrip with the actual Railway identity; then review additive tab creation before production rollout. |
| Caption, collaborator and scheduling preservation | PASS focused simulated checks | Rerun the four HTTP tests and existing paused integration against final deployed code. |
| All selected TDOG renders | BLOCKED | Four edge mismatches and one margin failure remain real failures. Obtain retained-media access and a separately authorized bounded diagnostic/individual retry after deployment review; preserve all five selections and the original 13 candidates. |
| Real layout/framing/audio acceptance | NOT VERIFIED | Retained TCB/TDOG single/two/three/four-person and presentation samples are not available locally. Drive binary fetch transport closed and browser/CDP media inspection failed. Do not infer a production media-permission outage from these tooling failures. |
| Full regression | FAIL | 29 baseline failures remain; classify/resolve relevant failures on Linux CI before merge readiness. Final four HTTP checks also remain outstanding. |
| Live publishing readiness | NOT VERIFIED / NOT AUTHORIZED | No publication, permissions preflight or Collab invitation was sent. |

Before any rollout: review the two diffs in order, complete the relevant Linux/API and real-media gates, verify current publishing remains globally paused and Agape disabled, verify a single worker/replica and persistent volume, back up all existing records, and obtain separate merge/deployment authorization. Deploy Ripped only unless a subsequently demonstrated integration gap requires another service. Preserve all current source artifacts and approvals. Do not use generic source restart/reselection to validate this change against the protected TDOG batch.

After an approved paused deployment, use retained source artifacts and individual failed-clip review/retry, not a new intake. Verify actual complete exports and sound, the request-specific website, owner copy/creator approvals, and actual Schedule Master dates/destinations. Stop on authentication, shared integrity/storage problems, uncertain publishing receipts or unexpected protected-record changes. Public publishing requires separate named-delivery authorization.
