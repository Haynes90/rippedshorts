# Complete speech for both clip formats

Every new 16:9 highlight and 9:16 Short passes the same completion gate before
Drive upload. This also applies to retries and clips recovered from saved selections.
Previously uploaded assets are not automatically replaced.

1. Re-extract source audio and rebuild word timestamps, including 30 seconds of
   surrounding context on each side. Do not rely on cached segment timestamps.
2. Review sentence completion and the whole thought: setup, explanation, examples,
   questions and answers, and conclusion/payoff. The reviewer chooses real word
   indices; the validator rejects invented indices or removal of approved content.
3. Find an actual audio pause after the final word, before the next spoken word.
   Keep a small leading/trailing margin. Never cut a thought to meet a length target.
4. Render, check output duration, and independently transcribe both exported edges.
   The opening/final words must match and no next-sentence words may be appended.
5. If verification fails, rebuild timings and review with 90 seconds of surrounding
   context, then rerender and recheck. Stop after two attempts. Unverified output is
   removed locally and never uploaded. The existing failed-render review state
   displays the reason; the user can adjust the selection and retry approval.

Shorts retain the existing 90-second maximum, including speech margins. A complete
thought that cannot fit is held for boundary review. Horizontal highlights have no
eight-minute maximum. If completion lies beyond the available context or source,
hold for review instead of inventing an ending or truncating speech.

Missing source transcripts are rebuilt from overlapping audio windows (four seconds
on either side of each chunk). Words are assigned to one chunk and combined into
sentences across seams. Empty/invalid word responses retry with freshly extracted
audio, up to three attempts. Empty owned chunks retry and then fail explicitly;
an empty or silent chunk is not silently accepted as complete speech coverage.

Successful adjusted bounds, transcript, and `completion_check` receipt are saved
back into the job before scheduling uses the asset. Receipt version 1 records the
original bounds, verification attempt, semantic reason, audio pause, and export
edge checks. Already completed clips retain their existing receipts and links.

The gate uses the configured OpenAI key/model for semantic review and `whisper-1`
word timestamps for source and export verification. This adds transcription/model
work per approved clip. Missing credentials or unavailable analysis holds the clip;
there is no bypass that uploads an unverified render. FFmpeg and FFprobe are supplied
by the existing Docker image. Timings and speech recognition are estimates, not a
guarantee of editorial perfection: music, overlap, recognition variation, absent
punctuation, or a source ending mid-sentence can require human review.

Regression coverage: `tests/test_clip_completion.py`, including real FFmpeg pause
detection, chunk seams, incomplete thoughts, Shorts limits, omitted/extra export
words, bounded rebuild retries, and both upload call sites.

API contract: https://developers.openai.com/api/reference/resources/audio
