# PR #85 — controlled paused release and TDOG recovery

## Verified preflight

- Tested head: `16218b15915bea44daff6b9e6460fc639a74575e`.
- Full isolated regression: 529 passed, 0 failed, 2 local FFmpeg skips, 2 passing subtests.
- Final GitHub run 41475233109: topic-contract 149 passed; render 12 passed.
- Current merge: `55b64b5c19dc87474e8c90106fd4a2e083a4eb06`.
- Backups: `/data/release-backups/pr85-20261102T155437Z`; SQLite integrity ok,
  all 15 tables, 25 workbook tabs, environment fingerprints and four PASSED
  checkpoint fingerprints. Persistent /data volume and one worker verified.
- Retained video SHA256 `45314424b1529f9d169f8d08c4fce31ddab10b17b122ed107c98458376b80a67`,
  duration 3679.132154s, audio/video streams. All 1,189 timed segments verified.
- Credential fingerprint differences from prior release: none.
- Railway auto-deploy disabled; replica count one with attached volume.
- Effective Schedule Master publishing_enabled=false, Agape Meta=false,
  Agape YouTube=false; audit/reconciliation unchanged at 12h/300s.

Deployment and bounded recovery evidence will follow. Merge alone does not
establish deployment, recovery, candidate review or scheduling success.

## Merge and replacement requested

Merge tree `1a15872b269b35e0b7ec9904a4e85b613437b26b` exactly matches the
validated final PR tree. No new application change or regression rerun needed.
Old deployment `7a98fa01-fb39-43dd-a900-396fa1f6ae62` was stopped; Railway
visibly confirmed Service offline and no active deployment before replacement.
New Ripped Shorts-only deployment requested:
`0ac16e7f-29d5-469c-96c4-fbfaf74ee949`.
No Audio Master or Schedule Master deployment requested.

## Verified deployment and one dispatch

Runtime SHA `55b64b5c19dc87474e8c90106fd4a2e083a4eb06`; deployment
`0ac16e7f-29d5-469c-96c4-fbfaf74ee949`; instance
`f3f5c142-d256-472b-a350-ae99fd1c68a8`. /ping HTTP200, readiness=true,
SQLite integrity ok, one uvicorn worker. All 15 table hashes, credentials and
protected workbook values unchanged (only unrelated Church Job Tracker Updated At
heartbeats changed). Source checksum remains verified.

New code recomputed exact immutable IDs for episode, conversations,
relationships and first-conversation legacy topics from source/transcript/parent
inputs. All four result fingerprints matched predeploy backups. Evidence:
`post-deployment-verification.json`, `recomputed-checkpoints.json`.

Single authenticated intake callback for the existing request acknowledged HTTP200
at authorization timestamp `2026-11-02T16:00:54.706383+00:00`.
Dispatch ID `pr85-hierarchy-v3-8999063e-bbc7-4000-bba4-e2490ca27135-1`.
Owner-bounded revision 3, attempt 1, recipe hierarchical-source-v3. Prior revision2
and its hold/errors were appended to history. Old failed rows/budgets unchanged.
No new intake, acquisition or transcription was requested.
Actual stage and candidate acceptance remain to be verified; HTTP200 is not proof.

## Actual recovery outcome — BLOCKED, no further dispatch

The single authorized run is held at `awaiting_topic_map_review`, retryable=false,
revision 3/attempt 1. Historical watchdog count remains 3. Its authorization receipt
is `pr85-bounded-recovery.json`; final read-only reconciliation is
`pr85-recovery-outcome.json`. No second recovery was sent.

| Stage | Actual result |
| --- | --- |
| Episode understanding | PASSED, original checkpoint reused unchanged |
| Four conversations / full-source boundaries | PASSED, original checkpoint reused unchanged |
| Conversation relationships | PASSED, original checkpoint reused unchanged: 4 sources, 7 directed links |
| First-conversation topics, segments 0–129 | PASSED, original checkpoint reused: 8 nodes (1 root, 7 subtopics) |
| Second-conversation topics, segments 130–350 | PASSED under chronological-topics-v2: 10 chronological topics |
| Third-conversation topics, segments 351–900 | FAILED on thematic SELF_REFERENCE at zero-based topic_position 5 |
| Fourth-conversation topics, segments 901–1188 | NOT STARTED |
| Whole-source moments | NOT STARTED |
| 9:16 sentence-level selector | NOT STARTED, candidate count 0 |
| 16:9 highlight selector | NOT STARTED, candidate count 0 |

The previously validated conversations remain:
1. Opening sketch, casual check-ins and travel banter (0–129).
2. Episode focus revealed; PlantCon & workshop promotions and giveaways (130–350).
3. Seed-saving strategies, storage and germination techniques (351–900).
4. Viewer Q&A, product mentions, practical takeaways and sign-off (901–1188).
Their boundaries cover all 1,189 original transcript indices. Complete hierarchical
analysis and semantic acceptance have NOT passed because later topics/moments
remain incomplete.

## Real topic quality and new precise diagnostic

The second conversation's ten chronological topics include opening/check-in,
weekend weather, propagation workshop, a carved velociraptor discussion,
wood-carving skills, project scope and social-media documentation. Transcript
spot checks at original indices 130,145,180,242,304,324 corroborated the generated
subject labels (including the surprising carving tangent). Returning-topic links
validated. These are genuine provider outputs, not fixture results. They are not
claimed as approved clip recommendations.

The main seed-saving conversation then failed with:

- category: `TOPIC_MAP_INVALID`.
- reason: `Invalid chronological topic reference`.
- failure_kind: **SELF_REFERENCE**.
- topic_position: **5** (sixth proposed topic).
- stage: `topics`; analysis_version: `chronological-topics-v2`.
- prompt_version: `chronological-topics-v2:topics`; actual request prompt hash saved.
- reference_type: `str`; topic key and rejected-reference SHA256 digests match.
- valid_reference_count: 15; verified parent bounds 351–900.
- schema: temporal parent is the verified conversation; thematic field related_keys;
  forward links permitted. The target failed because it referenced itself, not
  because it was forward, returning or outside a depth-first ancestry stack.
- Immutable failed stage ID and sanitized key/set digests retained in SQLite and
  the final outcome JSON; no full transcript/provider response or credential exposed.

Unlike the prior failure, the exact contract violation is now confirmed. The
provider supplied a thematic link to the same topic. The local validator held it
without silently deleting, renaming or substituting the reference. The earlier
second-conversation failure is fixed in this real run; complete acceptance is
still blocked by supplementary thematic metadata in the following conversation.
The failed topic provider body was not retained as a successful checkpoint; do
not claim its 15 proposed topics as accepted. The enclosing aggregate failure is
propagation of the same child failure, not an additional provider/recovery call.

## Integrity, review and scheduling

Final reconciliation confirmed:
- All four preexisting PASSED checkpoint fingerprints unchanged.
- Other request states unchanged, including previous five TDOG selections,
  approvals and held renders under request 94893455-5f48-46de-bdfb-642212ea19a5.
- Request count 39 before/after, no new intake IDs.
- Protected TCB queue/schedule/handoff-outbox values unchanged.
- Readiness=true; no scheduler work, new scheduled dates or deliveries.
- Historical failed analysis records and earlier authorization evidence retained.
- No new source acquisition, transcription, clip choices or render retries.
- No public publication, collaborator invitation or Agape configuration change.

Because selectors were not reached, actual hierarchy/rubric/scoped-feedback
influence on fresh recommendations, candidate identity/boundaries, real framing,
finished-media playback and owner-review website acceptance are **NOT VERIFIED**.
There is no candidate-ready Telegram delivery receipt or new review link for this
attempt. No placeholder link was sent as though useful recommendations existed.
Existing website/collaborator/scheduler behavior remains regression-tested but
was not exercised through fresh production assets in this recovery.

## Next action / explicit stop

Do not resume this request again under the current authorization. A narrow,
reviewable correction is needed for the topic thematic-link contract, including
preservation of already validated temporal topic spans independently of a bad
supplementary self-link; it must not silently accept or discard invalid metadata.
Do not add another broad architecture investigation or regenerate passed stages.
After a focused correction is reviewed, a separately authorized paused deployment
and bounded recovery should reuse all five PASSED checkpoints, including the new
second-conversation topics. Current status: healthy paused release; production
candidate acceptance **BLOCKED**. The next measurable milestone remains actual
9:16 and 16:9 recommendations delivered for owner review, not more unit tests alone.

