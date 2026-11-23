# Bounded review-to-scheduling reliability assessment

Scope: the existing November 20 request and its six owner-approved assets. No
transcription, discovery, rendering, publishing, credential changes or architecture changes.

## Observed live result

Final copy approval checkpoint: 2026-11-23 16:29:47 UTC. Schedule Master acknowledgment:
16:29:54 UTC. Six accepted outbox records correspond to 16 unique schedule rows
(TCB rows 288–303). The authoritative rows are SCHEDULE_PLANNED; Short 4 is absent.
All frozen approved copy/title snapshots and Drive identities match the owner
records. Creator snapshots match, including equivalent UTC timestamp serialization.

Receipt correlation: TW-e4b5a329-e2a6-46d2-8367-0b22a820f2a5.
Track: TCB:OD0uyyq13d8. Delivery IDs are the existing asset IDs suffixed by platform.

| Asset | Instagram (America/New_York) | Facebook | YouTube |
|---|---|---|---|
| Short 1 | Nov 24 09:00 | Nov 24 09:10 | Nov 24 09:00 |
| Short 2 | Nov 23 21:00 | Nov 23 21:10 | Nov 23 21:00 |
| Short 3 | Nov 24 21:00 | Nov 24 21:10 | Nov 24 21:00 |
| Short 5 | Nov 25 09:00 | Nov 25 09:10 | Nov 25 09:00 |
| Highlight 1 | N/A | Dec 19 19:15 | Dec 19 19:15 |
| Highlight 2 | N/A | Dec 30 16:00 | Dec 30 16:00 |

Dates are 2026; offsets are -05:00. The existing queue assigned them; this inspection
did not change them. Ten Meta deliveries passed the existing nonpublishing provider
preflight. Six YouTube deliveries have valid schedules but channel-read verification
is blocked (YOUTUBE_CHANNEL_READ_SCOPE_MISSING); the upload grant itself passes.
Global publishing and Agape stay disabled. No post was published. Canva is not an
active configured scheduling destination; the owner's intended meaning of “Canvas”
cannot be inferred from configuration alone.

## Classification of the incident-related paths

| Path/risk | Classification | Evidence/action |
|---|---|---|
| Complete-passage construction, material contraction/editorial score reuse | Already resolved and adequately tested | PRs 102/103; sentence-first and immutable-boundary regressions, seven retained real exports. Rough conversational openings/recaps remain editorial choices, not proof of an acoustic defect. |
| Source identity, pre-render audio, exact bounds, export verification | Already resolved and adequately tested | Source/revision bindings, full decode, source/export alignment, independent audio checks; existing boundary/export tests. No reprocessing needed. |
| Lost website votes, stale revisions, ambiguous save response | Already resolved and adequately tested | PRs 102/104, durable decisions and copy survive actual deployment/reload; existing conflict and readback tests. Six actual owner votes persisted. |
| Missing highlight copy/overwriting owner text | Already resolved and adequately tested | PR104 generation connection, immutable cache and concurrent-edit protection. Actual final descriptions reached Schedule Master. |
| Stale learning checkpoint overriding unified-editor wording during handoff | Missing regression protection | Existing transport_copy already restores authoritative exact post and removes duplicate hashtag transport. Added an actual handoff regression; no production-code change needed. |
| Telegram send fails AFTER accepted receipt/local commit | Confirmed failure-path defect, not observed for this accepted batch | Red test reproduced accepted state being changed to failed and outbox demoted to RETRY. Isolate that send exception; retain accepted result and receipt. |
| Uncertain provider response and interrupted outbox ACK projection | Missing regression protection | Added explicit tests preserving approval, retry payload identity and locally durable receipt without an immediate duplicate POST. Existing mechanisms reused. |
| Render interruption/unknown outcomes | Already resolved and adequately tested | Existing leases and immutable receipts keep uncertain renders held; completed siblings survive and unchanged media is reused. |
| General schema/architecture/style changes | Low-priority hypothetical issue | No demonstrated need; no changes. |

## Minimal correction and release gate

Only `_handoff_shorts_to_schedule_master` changes: notification exceptions cannot
demote committed acceptance; failed-handoff notification errors cannot obscure its
already-persisted diagnostic. No handoff payload, schedule, retry policy, publishing
permission, worker or storage design changes.

Targeted suites cover the correction plus existing sentence-first, boundary,
request-scoped recovery, caption, website finalization, creator and durable-job
contracts. This PR requires separate deployment authorization. The current batch
must not be resubmitted to deploy or test it. A new normal TDOG request can use the
existing processing/review path; unrestricted publishing readiness is not claimed
while global pause and the specific YouTube channel-read safeguard remain in effect.
