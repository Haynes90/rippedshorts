# Fresh TDOG acceptance: topic-reference contract blocker

## Release completed
PR #79 merged as 78b4b34d5516da8032f78aa065ebff0c44d5b79d. PR #80 reconciled to main and merged as c5d50634acd9cc972102a27f169d27800f6b5ec2.
Ripped Shorts alone deployed that revision with publishing paused, persistent storage and one application worker. Original selections, approval records, credentials and scheduling safeguards were preserved.

## Confirmed real acceptance failure
Authenticated retained-source verification passed; the completed Audio Master job and 1,189 valid timed transcript segments were reused. No new source transcription or YouTube acquisition was needed.
The fresh source-map analysis failed before either format's candidate generation:
`Topic references are outside the verified transcript`.
This exception occurs only when a top-level first_segment/last_segment is not an integer or is out of the ordered inclusive transcript range.
The original provider map was not durably retained. Therefore the precise offending reference values and whether the defect was type, range or ordering are NOT VERIFIED. There is no basis to claim a one-based/off-by-one root cause.
The original provider request returned parseable JSON; this is a source-map contract/integrity failure, not the earlier transcription authentication failure.
The fresh request is held awaiting_topic_map_review, with no selections/renders/handoffs and no automatic retry. Failed-analysis identity and original request are preserved. An owner Telegram status notification was acknowledged.

## Narrow correction in this branch
- Require strict structured JSON with integer indices bounded to the actual transcript, including nested moment references.
- Explicitly state zero-based inclusive semantics; retain local full-source coverage, parent, moment and semantic-reference checks. Do not coerce, clamp, shift or invent indices.
- Preserve sanitized failing reference types/numeric values, allowed range, topic position and prompt hash/version in the existing source_analyses failure result. Do not retain transcript/provider free text in public errors.
- Existing successful maps and sibling analysis results remain reusable. A deterministic failure remains held until explicit bounded recovery.
No production changes from this correction have been deployed. No credentials or selection decisions changed.

## Tests
Final isolated full suite: **413 passed, 0 failed, 2 skipped, 2 subtests passed**, 4 existing framework deprecation warnings, 17.65 seconds.
Command:
`python -m pytest -p validation_network tests -q --basetemp=<isolated topic-contract-final directory> --junitxml=<isolated topic-contract-final.xml>`
Mocked provider calls; loopback HTTP allowed, other network denied. Two local skips require FFmpeg/ffprobe on PATH. GitHub's stored-render check supplies those tools.
Nine added cases cover malformed/out-of-range/reversed/oversized references, durable sanitized diagnostics, deterministic no-repeat behavior, actual 1,189-segment schema limits and preservation of the coverage gate.
This is not a reproduction of the lost raw response, and no genuine model retry has been performed to claim real-source recovery.

## Required next step
Review and separately authorize this correction's merge and paused Ripped Shorts-only deployment. Then explicitly reopen the existing fresh analysis once, reuse retained verified media/transcript, and continue its normal two-format Telegram review. Do not create another intake or reset the legacy request.
A valid full-source topic map and actual candidate review are still required. Finished clips, actual audio/framing/playback, human approvals and durable Schedule Master acknowledgement remain pending.

Contract reference: [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat). JSON mode guarantees JSON syntax; strict structured outputs constrain the response schema. Semantic coverage remains the application's responsibility.
