# Ripped Shorts render investigation

## Evidence and prior changes

- The supplied September 18 log shows request `18b822d0-93b5-43f2-8ec3-ddd50c70b30f` waiting 1,200 seconds for Audio Master (last status `transcribed`), then failing on an undefined `download_youtube_resilient`. Current main already imports that function. This does not prove which revision Railway currently runs.
- The later audio-source validation and candidate preflight changes reject media without audio. They do not solve rendering with an existing transcript and can prevent review entirely.
- Serializing transcription and bounding render concurrency limit contention, but leave repeated speech transcription in the render path.
- Current `render_complete_clip` previously transcribed the source again, called a speech boundary reviewer, required an audio pause, and transcribed the export edges. This directly explains the remaining audio dependency.
- Current render recovery improvements preserve approvals and recover sources, but recovery still rejected an existing video solely for missing audio.

## This patch

- Keep the existing downloader import; add a regression check for it.
- Download video directly when the stored timed transcript exists and video is missing, bypassing the Audio Master wait in that case.
- Remove the audio prerequisite from candidate preflight and source recovery.
- Render both aspect ratios using the selected transcript and timestamps, without re-transcription or audio pause verification. Existing FFmpeg renderers preserve audio when present.
- Check finite bounds, source duration, the Shorts duration limit, stored sentence punctuation, output existence, output duration, and a video stream before upload.
- Correct the preflight closing-quote handling.
- Record verification basis explicitly; do not claim word alignment or speech-edge verification. Stored transcript timing accuracy and complete-thought quality still depend on upstream selection and human review; punctuation alone does not prove either.

## Verification and limits

Ten isolated standard-library regression tests passed locally, covering both aspect ratios without speech-service calls, video-only source recovery, quoted sentence endings, bounds, duration, and failed output cleanup. Python compilation passed for changed production modules.

These tests inject media probes and a renderer; they are not a real FFmpeg render or a live Telegram/Drive delivery. No production job or transcription was triggered. Production deployment revision and successful approval-video delivery remain unverified. Do not describe the incident as resolved until a real render succeeds on the deployed revision.

