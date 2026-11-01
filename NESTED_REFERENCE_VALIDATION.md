# Nested topic-reference recovery validation

Base: 1e967a71ecfeb8ca2a802727a72f3bcb085dd9fe. Focused change only in production_intelligence.py and its existing contract tests.

## Correction
NestedReferenceError extends the existing TopicReferenceError and therefore uses the same lease-owned failed-analysis persistence. All three fields (subtopics, questions_responses, memorable_moments) retain zero-based topic/item position, field name, actual reference types and safe literal values, parent first/last indices, transcript bounds, model/provider and prompt hash/version.
Malformed collections/items are explicitly rejected and diagnosed rather than treated as empty arrays. Arbitrary strings/objects are redacted; numeric strings retain their actual type and value. Titles, summaries, transcript bodies, tokens and raw provider errors are not copied into diagnostics.
Valid boundaries, hierarchy/coverage checks, existing successful cached maps, owner selections, approvals, framing, collaborators, receipts and scheduling controls are unchanged.

## Generation contract
Strict JSON continues to constrain integer indices to the full transcript. Nested object descriptions and prompt semantic-source-map-v3-nested-contract explicitly require GLOBAL indices and topic.first <= item.first <= item.last <= topic.last.
Parent bounds are themselves generated values; the current single-response schema does not enforce that cross-field relationship. Local validation remains mandatory. No clamping, shifting, inferred words, silent repair, omitted invalid moments or automatic provider retry.
The previous real invalid nested values were discarded by the old generic ValueError; they cannot be reconstructed retroactively. New regression cases reproduce that validation condition, not an invented claim about the lost raw response.

## Executed tests
Full isolated regression: **445 passed, 0 failed, 2 skipped, 2 subtests passed**, 4 existing framework deprecation warnings, 18.98 seconds.
Command: python -m pytest -p validation_network tests -q --basetemp=<isolated nested-reference> --junitxml=<isolated nested-reference.xml>.
External networking denied by validation plugin; providers mocked; loopback HTTP permitted. Two local skips require FFmpeg/ffprobe on PATH. Existing GitHub render and topic-contract checks must pass on the final commit.
32 focused cases added across all nested fields: outside-parent, reversed, wrong type, redaction, malformed collections/items, valid endpoints, persistence after reopening SQLite, and no repeated provider call. Existing full-source coverage and hierarchy regressions remain.

## Authorized production acceptance
After passing CI, verify auto-deploy off, fresh database/workbook/configuration backups, one persistent worker, credentials and publishing/Agape gates unchanged. Stop old deployment before replacement; Ripped Shorts only.
Exactly one authorized recovery of existing request 8999063e-bbc7-4000-bba4-e2490ca27135 using verified retained source and 1,189 timed segments. Preserve the old five selections. No new intake, source acquisition, transcription, backlog restart, scheduling or publication.
If another validation fails, retain actual diagnostics and hold. No second recovery is authorized. Candidate generation and real finished-media acceptance must be reported from actual runtime evidence, separately from these tests.

Reference: https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat
