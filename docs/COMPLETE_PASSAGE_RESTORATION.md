# Complete-passage and seven-candidate restoration

Acceptance gap: request `e4b5a329-e2a6-46d2-8367-0b22a820f2a5` has two verified Shorts, three held Shorts and two held highlights. Production remains `4eb480eace4303f792c65d634f2382534be08a7b`. No production candidate, decision, boundary, hold or media was changed during this correction.

## Actual retained evidence

The owner authorized a read-only SQLite export using the established temporary one-off method. The script used SQLite `mode=ro`, imported no application modules and exported 36 request-scoped construction/assessment/hold records. The normal worker was restored on the same production commit. The existing 427-segment Drive transcript was read and checked against its stored size and MD5; no audio/video source acquisition or transcription occurred. The test fixture contains only the relevant retained transcript windows and assessment evidence. Floating-point source timings are preserved in the fixture; this table rounds to milliseconds.

| Candidate | Original proposal (seconds) | Last attempted constructed passage | Actual result / reason | Correction evidence |
|---|---|---|---|---|
| Short 1 | 168.280–247.340 | 241.180–247.340 | Rendered, 6.166667 seconds. Semantic selection replaced the molasses explanation/recipe with a generic nutrient-uptake sentence. | Original model response explicitly called the opening a mid-clause and chose context words 216–234. The question and recipe are retained in the transcript; a new complete recipe passage needs semantic and acoustic assessment, not forced reuse of 79 seconds. Existing file stays unchanged. |
| Short 2 | 520.080–599.440 | 520.080–599.440 | `NO_EVIDENCE_SUPPORTED_PROGRESS`; preceding assessment: complete thought/context, unresolved acoustic start. | Recorded quiet 519.272398–520.024734 supports an earlier entrance without discarding intended speech. |
| Short 3 | 1339.820–1384.980 | 1339.752–1384.980 | Same terminal hold; complete instructional thought/context, unresolved acoustic end. | Recorded quiet 1385.010971–1385.363420 supports preserving the word release after the timed segment endpoint. |
| Short 4 | 2127.680–2179.620 | 2127.608–2179.620 | Rendered, 52.010998 seconds; semantic/context/acoustic and export checks passed. | Preserve and reuse the completed file and receipt. |
| Short 5 | 2553.400–2580.060 | 2553.400–2580.168 | Same terminal hold; complete thought/context, unresolved acoustic start. | Recorded quiet 2553.161921–2553.271604 supports an earlier entrance. |
| Highlight 1 | 1178.940–1413.040 | 1178.739990–1413.039993 | First assessment accepted the full 234-second arc, held its acoustic start. A redundant second assessment then failed `SEMANTIC_EVIDENCE_REFERENCE_INVALID/setup_units`. That second response was not retained, so its exact malformed value cannot be asserted. | Recorded quiet 1177.942892–1178.514298 supports the start correction while preserving the original arc. No new topic selection is needed. |
| Highlight 2 | 2285.620–2475.160 | 2285.533895–2475.160004 | `NO_EVIDENCE_SUPPORTED_PROGRESS`; first assessment accepted the full 189-second troubleshooting arc and context, but held its acoustic end. | Recorded quiet 2475.212725–2475.771297 supports completing the final word and natural release. |

Only Shorts 1 and 4 have successful render/Drive receipts. Their existing Drive IDs are `1d1QVn8V8exwW5_byOiV9otWvyeSNIxTl` and `1MCktJ7dsscUIcdYRYxnTbky3kzQC5E1S`, respectively. The other five stopped before FFmpeg; no successful export/upload is claimed for them.

## Precise causes and minimal correction

PR #93 added `construct_preview_candidate()`. On a held assessment it unconditionally assigned `bounds = assessment['suggested_semantic_bounds']`. Before that construction path, a semantic disagreement was a hold rather than an automatic replacement passage. In the observed Short 1 step this assigned 241.17999267578125–247.33999633789062, removing the recipe. PR #102 prevents obsolete editorial scores from validating a materially changed passage; this correction moves that same existing editorial contract into the first construction assessment, so the original message guides boundary resolution before a contraction is adopted. It adds no provider stage.

`prepare_for_render()` also supplied only four surrounding units on each side. For word-timed material those were four words, not four transcript segments, leaving the actual earlier question out of context. Context now uses the existing four-segment budget, with all measured word references preserved and the context revision included in the assessment cache binding.

The acoustic resolver already measured the relevant quiet intervals, but constrained an opening to be after the preceding ASR unit's end and a closing to be before the following unit's start. Contiguous segment timestamps therefore provided zero room even when source audio demonstrated a real transition. Retrying the semantic endpoints produced the same draft and the `NO_EVIDENCE_SUPPORTED_PROGRESS` stop.

Construction now binds the intended timed-unit range separately from its measured acoustic entrance/release margin. It never removes any intended timed speech; the margin is chosen only from the existing bounded acoustic scan. The immutable specification records the range and rejects missing capture, changed ranges, source/transcript changes or editorial mismatch. Segment timing is never promoted to word precision. Silence is neither newly required nor treated as evidence of editorial completeness. Valid context citations outside the initially suggested range expand the passage instead of being discarded as invalid references.

The website keeps the existing playable gallery and approval denominator. Its existing Shorts/highlights sections now account for all discovered previews: ready entries link to their player; held/pending entries show truthful status, intended transcript and bounds without viewing/approval controls. Raw diagnostics remain in durable records; creator status is a plain-language projection. Held siblings do not block unrelated finished-video finalization.

## Verification and release boundary

Nine new focused tests replay the real five acoustic holds, reject the real Short 1 tail substitution before rendering, expand a valid setup citation, reject quiet intervals inside intended speech, and verify seven statuses/two selectable videos without changing records. Replay semantic responses and Drive are explicitly isolated doubles; these are not new live model decisions or public media acceptance.

Existing tests cover meaningful short passages without a new minimum, exact specification consumption, unchanged-media reuse, restart survival, owner decision/version preservation, source/export alignment and real FFmpeg in both formats through authenticated media and retained receipts. The CI production image supplies the approved OpenCV and FFprobe dependencies.

Local affected suites: 71 passed, 4 tool-dependent skips. Local full suite: 778 passed, 7 skips, 2 subtests; one unrelated environment failure because the existing Windows virtualenv has OpenCV 5.0 without `CascadeClassifier`, unlike the pinned production OpenCV 4.11.0.86. No dependencies were installed or changed. Exact-head GitHub CI is required before rollout.

No holds are cleared by this patch. The existing unchanged-hold, immutable-construction, approved-revision and media-reuse guards remain active. Deployment and explicit asset/revision-specific continuation require separate authorization. Fresh source-audio/final editorial validation and exported-media/browser checks are still required before any corrected production preview can be called playable. The existing six historical selections, two current exports, all owner decisions and Schedule Master receipts remain protected. Publishing and Agape stay disabled.
