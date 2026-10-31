# Camera-tile composition — review evidence

Status: **DRAFT; NOT PRODUCTION ACCEPTED.** Depends on `feat/source-intelligence-20261030`.

This change extends active_speaker.py and main.py. It uses the existing OpenCV face detector, FFmpeg renderer and audio-activity path. It does not add a publishing destination, change credentials, replace the renderer, or require a paid service.

The new path requires sustained visual separators and consistent participant occupancy across sampled frames before declaring camera regions. A face midpoint alone is not a tile boundary. Whole camera rectangles are inset away from neighboring borders and fitted into panels without stretching. Unknown meaningful regions, including screen shares, remain visible in a full-source composition. A confidently empty decorative region can be excluded from a three-camera composition.

Spatial mouth-motion tracks are gated by mixed audio activity and correlation; audio spikes do not identify a person. Sustained distinctive evidence may feature a primary spatial track. Ambiguous evidence retains a stable group view. The existing single-person/face-aware/stacked fallbacks remain available. Four-person scenes use a stable grid rather than force unreadable secondary strips. No named creator identity is inferred from a camera track.

Framing plans are thread-local to the render and saved beside the output, then retained in the clip's review evidence. Plans reset between renders. A plan is NOT proof that the real output contains the intended named participant; actual visual review is still required.

## Executed tests

`python -m pytest -p validation_network tests/test_camera_tiles.py -q --junitxml=camera-results.xml`

**15 passed, 0 failed.** Includes actual FFmpeg encoding/decoding of synthetic two-person side-by-side, two-person stacked, three-camera, four-camera and camera/presentation layouts. Tests check output 1080×1920 dimensions, stream decoding, expected panel/context pixels, nonempty output and audible-tone energy. Other cases cover absent separators, unstable layouts, ambiguous mixed audio and conservative speaker handoffs.

The initial GitHub stored-render job failed two cases because its direct `python tests/test_stored_transcript_render.py -v` harness did not put the repository root on sys.path. The new production framing-evidence import therefore could not resolve active_speaker. The focused harness correction adds the existing repository root; it does not mock out the import, alter application behavior or relax assertions. The PR's latest check is authoritative for CI revalidation status.

Combined stack regression: **341 passed, 29 baseline failures, 2 skipped, 4 HTTP tests deferred**. See PRODUCTION_INTELLIGENCE_VALIDATION.md and PRODUCTION_INTELLIGENCE_EVIDENCE.json on the base review branch. Counts overlap and must not be summed.

## Unverified limitations

- No retained real TCB/TDOG program was rendered with this new path in this assignment. Binary Drive fetch/browser access failed; this does not establish a production media permission problem.
- Actual speaker accuracy, decorative-border detection, busy graphics, changing layouts and real three/four-person legibility need calibration against verified footage.
- Existing single-person fallback tests remain; synthetic fixtures do not substitute for the requested real single-person example.
- Conservative detection can decline a layout and use the established fallback. The system does not know a person's identity merely from their screen region.
- No manual tile editor was added. The website exposes evidence; owner visual review remains necessary.
- Production merges, deployment, publishing, retries of the existing TDOG job and real invitations were not performed.

Review/complete the real-media gates before any separately authorized paused rollout. Keep all current selections, approvals, schedules and holds intact.
