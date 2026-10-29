# Brand-aware media workflow — implementation blueprint v0.1

Status: **Design approved in conversation; implementation pending.** This document is the agreed floor plan, not proof of deployed functionality. Applies first to TCB and TDOG. Agape can use the common architecture later.

## Goals and non-goals

- Keep the existing **Telegram brand selection**, clip acceptance, and web caption approval; no new standalone app.
- On source submission, start **full-video acquisition and Audio Master transcription concurrently** wherever technically possible. Full source acquisition is the preferred clipping path; do not depend on unproven remote partial-video clipping.
- Retain the downloaded video and transcript separately as durable artifacts so each survives failure of the other. Do not download the same video repeatedly when a validated copy exists.
- Produce complete sentences and **speech boundaries**, not just punctuation or coarse transcript chunks; preserve setups, full thoughts, payoffs, and especially multi-minute explanations/discussions for 16:9.
- Create a shared creator directory with **brand-specific defaults**, suggestions and approvals. Creator association and Instagram collaborator invitations are different things.
- Never bypass human clip/copy/collaborator review, silently rewrite human-approved captions, publish before approvals, or enable Agape as a consequence of the TCB POC.
- Do not assume a native Instagram Collab invitation is supported by the current Meta adapter. Verify capabilities and record actual outcome; use an explicit manual-invitation state if unsupported.

## Existing integration surfaces (verify before coding)

- `rippedshorts/telegram_intake.py`: brand selection, Audio Master ingest invocation, Drive-cache reuse, fallback transcription, human clip selection, approved render queue, Schedule Master handoff.
- `rippedshorts/source_ingestion.py`: `ingest_with_audio_master`, `reuse_from_drive`, source video download fallback.
- `rippedshorts/clip_completion.py`: sentence/word-timing construction, clip-boundary validation, render-edge checks.
- `rippedshorts/caption_review_web.py`: existing tokenized mobile copy review, edits, completion.
- `rippedshorts/main.py`: AI candidate selection, FFmpeg 9:16 speaker crop and 16:9 render.
- `audiomaster/ripped_shorts_ingest_runtime.py` and `media_ingest/`: reusable transcript/source-video status. Confirm actual contract and update additively.
- `ScheduleMaster`: brand/platform destination gates, schedule intake, approved copy preservation, posting receipts.

## Proposed end-to-end flow

1. **Telegram intake:** select TCB or TDOG (existing brand-aware UX); submit YouTube URL; determine stable source video and channel ID; offer creators/guests step.
2. **Parallel work immediately after acceptance:**
   - Lane A: Audio Master acquires/transcribes source audio and creates durable timed transcript.
   - Lane B: acquire **entire source video** from existing permitted provider/route, check identity, duration, streams and integrity, cache and expose accessible file reference.
   - These lanes have separate statuses/attempts and independent failure recovery. Do not wait for transcription to finish before starting video acquisition. If transcription itself requires downloaded audio, acquire that audio as early as possible without blocking full-video acquisition.
3. **Reusable transcript:** sentence-level spans assembled across chunk seams; original words/timestamps (when available); word onset/offset, speaker turn where reliable, acoustic speech onset/offset, pauses and confidence. Separate **text sentence boundaries** from **actual speech boundaries**; never represent inferred boundaries as verified.
4. **Selection:** Ripped Shorts scans the full time-aligned transcript, proposes complete 9:16 thoughts within configured length, and 16:9 full explanations/stories/interviews (possibly several minutes and multiple speakers). No truncation at arbitrary transcription chunk boundary.
5. **Human clip acceptance in Telegram** (existing). Check candidate entry/exit against actual speech, adjoining sentence, word onset/offset and natural pause. Where no safe cut exists, request boundary adjustment/review; silence alone is insufficient evidence.
6. **Render after video is ready:** FFmpeg extracts approved ranges; preserve 9:16 speaker-aware crop and native 16:9 formatting. Independently validate opening/closing speech in exported file using targeted boundary checks rather than automatic full-video retranscription.
7. **Creator directory suggestions:** initialize based on selected brand, **verified** source-channel creator association, show hosts and episode guest metadata. These suggestions appear at the **top of the existing website**, where the user confirms the list once per batch. Unknown or unverified identity never silently becomes an approved collaborator.
8. **Existing web approval — batch-level collaborator confirmation (V0.1):** at the TOP of the review website show **Who's featured?** with suggested verified creators, saved Instagram usernames, TDOG recurring hosts or TCB YouTube-source suggestions. The user confirms/adds/removes creators **once for the entire accepted batch**, and chooses whether to invite them as collaborators or use caption mentions. The single confirmed choice applies to **all approved clips in that batch**; do not require repeated per-clip approvals. Continue the existing per-asset caption/title/description editing beneath this section. Future version: allow per-clip collaborator overrides, but do NOT build or require those now. Only hand off after batch creators and copy review are explicitly completed.
9. **Schedule Master intake:** receive approved render file, metadata, brand, source IDs, creator IDs and per-platform collaborator intent. Preserve the exact approved caption. Publish on established cadence after all safety gates. Record whether native collab invite was submitted/accepted or requires manual action; never report a plain @mention as a successful invite.

## Brand-specific collaborator rules

| Brand | Defaults | Creator prompt | Per-clip behavior |
| --- | --- | --- | --- |
| TDOG (The Dirt on Gardening) | Saved recurring hosts suggested every episode | Ask for additional episode guests; recognize returning guests such as Britt Parrish **only after identity/handle confirmation** | Show default hosts and episode guests together at top of web review. Confirm once for entire batch; per-clip exceptions deferred. |
| TCB (The Chocolate Botanist) | Do **not** assume source-channel owner appears in every clip | Ask **“Who’s featured in this video?”**; suggest known creator from YouTube channel ID (e.g. Black in the Garden after user supplies and verifies mapping) | One creator/guest confirmation at the top of the website applies to every approved asset in the current batch; per-asset exceptions are deferred. |
| Other brands | No assumed defaults | Keep configurable, off until specified | Preserve isolation and existing brand gates. |

Recurring TDOG hosts must have an editable saved list; do not hardcode names or invent Instagram handles. Source YouTube channel ID is a strong **source account** match, not proof that person appears. YouTube video ID can map to an episode and its guest roster; use verified episode metadata or explicit user confirmation for people actually featured. Persist confirmed mapping for future recommendations. Default selection is a **suggestion**, with explicit or previously approved standing-rule authorization for actual invites.

## Shared creator directory — proposed contract

Creator record:
`creator_id`, `display_name`, `aliases`, `youtube_channel_ids[]`, `instagram_username`, `instagram_account_id?`, `profile_links`, `verification_status`, `verified_by`, `verified_at`, `active`.

Brand relationship:
`brand_id`, `creator_id`, `relationship` (host/source_creator/frequent_guest), `default_suggest`, `standing_invite_authorized` (default false), `last_confirmed_at`.

Episode/source:
`brand_id`, `youtube_video_id`, `youtube_channel_id`, `source_creator_id?`, `episode_guest_creator_ids[]`, `confirmed_featured_creator_ids[]`, `provenance`.

Batch collaboration approval (V0.1): `request_id`, `brand_id`, `confirmed_creator_ids[]`, `collaboration_intent_by_creator`, `approved_by`, `approved_at`, `scope=approved_batch`, `approval_revision`. Snapshot this approved selection into each eligible delivery so downstream retries preserve it. Never apply it to rejected/unselected clips. Individual-clip overrides are explicitly out of scope for V0.1.

Per delivery (future-compatible schema):
`asset_id`, `creator_id`, `feature_status`, `collaboration_intent` (none/mention/invite), `approval_status`, `approval_at`, `invitation_status` (not_attempted/submitted/accepted/rejected/manual_required/unsupported), `platform_receipt?`.

Store durable creator and decision data in an existing supported persistence layer, not ephemeral local memory; avoid inventing handles, excessive scope, duplicate identity records, or logging credentials. Define upsert/deduplication keyed by stable creator ID and verified external IDs.

## Timing/data handoff

Proposed reusable transcript sentence object (additive):
```json
{
  "text": "Original sentence with complete thought.",
  "start": 112.41, "end": 118.94,
  "first_word_start": 112.41, "last_word_end": 118.94,
  "speech_start": 112.39, "speech_end": 118.96,
  "speaker_id": "speaker_1", "pause_after_ms": 320,
  "boundary_confidence": "verified", "timing_source": "audio_master",
  "words": []
}
```
Fields such as acoustic onset/offset and confidence are nullable unless measured. Clip selection may use sentence spans; final clipping uses original-media timestamps and explicit speech-edge verification. Do not assign fake precisions.

## State machine and recovery

Separate `transcription_status` and `source_video_status` (queued/running/ready/retryable_error/permanent_error), each with durable artifact IDs and checksums/size or equivalent validation. Candidate selection requires transcript ready. Rendering requires full-video ready **plus** accepted selection. Review/scheduling handoff requires approved final copy, collaborators, and valid rendered video. Restart should resume missing lane only; bounded retry for transient errors, no destructive restart of completed stages. If either lane stalls, show specific diagnostic, not a generic three-attempt failure.

Known failed test: request `92128853-2d02-41f6-a172-7a769775abae` / YouTube `gKnDkM_b1CU`, transcript failure `Word transcription failed after three attempts`. Determine whether Audio Master returned reusable transcript and video for this source, and why Ripped Shorts attempted its independent full-source transcription. Preserve the actual upstream error and do not falsely conclude Audio Master was broken.

## Platform integrity

Native Instagram collaboration invitations are NOT equivalent to caption @mentions. Before implementing an automated invite, verify the exact supported Meta API flow/permissions for the deployed account type and document tested behavior and limits. If unsupported, explicitly mark `manual_required` and present the exact creator account; never claim `submitted`. Respect platform collaborator limits, account approvals and per-post recipient confirmation. YouTube/Facebook creator tagging is separate platform-specific capability.

## Implementation phases (approval gates)

1. **Inventory/contract**: inspect concrete Audio Master transcription and retained-video artifact format; identify minimal additive fields and existing directory/storage options. Record findings without broad refactors.
2. **Parallel ingestion and durable reuse**: independent concurrent acquisition/transcription statuses, test failure isolation, full-video integrity, and no repeat downloads.
3. **Sentence + speech timing**: accurate chunk seam joining, speech onset/offset and preview validation with uncertain-boundary holds. Preserve accepted outputs.
4. **Creator directory and Telegram UX**: TDOG recurring hosts; TCB feature prompt and source-channel suggestion; guest alias reconciliation and confirmation.
5. **Web review + Schedule Master contract**: one top-of-page **batch collaborator confirmation** applied to all accepted clips, idempotent delivery snapshots, platform capability gate, human-copy precedence. Defer per-clip collaborator exceptions to a later version.
6. **End-to-end TCB then TDOG canary**: selected source → parallel acquisition → transcription → complete boundary → clip acceptance → actual render → caption/collaborator approval → safe handoff. No public publishing without separate authorization. Agape rollout deferred.

## Acceptance checks

- Video download starts without waiting for full transcript; transcription can proceed while full video downloads where supported.
- Full video is reusable, validated, with independently recoverable transcript; no automatic full-source retranscription when Audio Master output is usable.
- Word and sentence continuity across chunk boundaries; full spoken explanations and natural entry/exit in both 9:16 and 16:9, including multi-speaker 16:9.
- A speech edge with no silence does not falsely pass by silence-only logic; uncertain edge is held for review.
- TCB prompts for who is featured; suggests creator matching verified YouTube channel; no assumed tagging without confirmation.
- TDOG recurring hosts suggested each episode; returning guests reused after confirmation; one top-of-site selection applies to all accepted clips in the batch. Per-clip overrides are future scope.
- One explicit creator/collaborator approval at the top of the existing website applies to all approved clips in that batch, never rejected ones; individual caption edits remain available per asset.
- Exact human-edited captions and decisions survive handoff and retries.
- Native collaborator invitation success is verified or marked unsupported/manual; never confused with @mention.
- No duplicate source acquisition, no duplicate deliveries, no Agape activation. No changes to live publishing flags.

## Decisions before coding

- User approved workflow direction, **not yet a specific storage schema, live API permissions, changes to platform capabilities, merging/deployment or publishing**.
- Confirm saved recurring TDOG hosts' actual Instagram identities and creator directory backing store.
- Confirm whether feature detection comes from episode metadata, creator/channel match, or explicit user selection; never infer guest presence from channel ownership alone.

## Confirmed scope update — batch creator approval on web review (2026-10-29)

The user specifically chose **one confirmation at the top of the existing website for the whole batch**. Present creator identities and proposed Instagram collaborators in a single **Who's featured?** section; a single approval covers all approved clips in that batch (including 9:16 and 16:9 where selected). Keep separate caption/title editing for each clip. Save the creator decision and propagate unchanged to every accepted delivery. No repeated creator prompts for individual clips. **Do not build per-clip collaborator customization in this version**; later versions may add it. Account-to-creator associations still require identity verification, and native invite behavior must be capability-checked rather than equated to mentions.

## Scheduling requirement — source-aware interleaving (confirmed)

For approved clips from multiple original source videos, Schedule Master should **interleave distinct sources** instead of exhausting all clips from one video before moving to the next. Example: `A1 → B1 → C1 → A2 → B2 → A3 → C2` (subject to eligibility). Sources are grouped by stable source video ID / source group, **not** by display title. This is controlled rotation, not unpredictable random reshuffling.

- Preserve brand separation, configured daily posting cadence, time windows, minimum gaps, release/approval dates, human copy, locked slots, source spacing, duplicate suppression and 9:16 vs. 16:9 scheduling lanes.
- Favor another eligible source when multiple sources are ready; do not require strict round-robin if doing so would prevent filling today's slots or violate priority/spacing.
- Retain original clip order **within each source** by default, unless an explicitly approved clip priority overrides it.
- New approved source B or C should enter the **next available eligible** scheduling window, not be forced to wait behind all already-queued clips from A. Rebalance only future, unposted, unlocked queue slots; never move published posts or silently modify explicitly locked schedules.
- If only source A is eligible, continue A while honoring cadence and spacing instead of leaving avoidable daily gaps.
- Reconciliation should be stable: repeated audits with unchanged inputs do not reshuffle the entire calendar. Use deterministic tie-breaking and preserve idempotency/companion destinations (IG/YT/FB) for each underlying clip group.
- Verify with tests for A/B/C interleaving, late B arrival, only-A backlog, locked dates, companion rows, no duplicate/extra post and same-day queue filling.

This is a **cross-repository implementation requirement** for `Haynes90/ScheduleMaster`, not evidence the algorithm has been changed or deployed. Inspect existing `app/scheduler.py` source rotation before modifying; make only necessary targeted changes.
