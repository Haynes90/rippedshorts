# Finished-clip review and source-preserving framing validation

Validated 2026-11-01 against main 3bddbf84a776c3f72d743a1c7c62bf9d95fa75bc. These are review branches; no merge or deployment performed.

## Evidence and results
- Independent review/provenance branch: 401 passed, zero failed, 2 skipped; 2 subtests passed.
- Combined review and framing stack: 404 passed, zero failed, 2 skipped; 2 subtests passed.
- Command: python -m pytest -p validation_network tests -q --basetemp=<isolated temporary directory> --junitxml=<isolated XML output>.
- External calls denied by the validation_network plugin; loopback HTTP allowed. Existing virtualenv, OpenCV 4 isolation and temporary DATA_DIR used.
- Two skips require ffprobe on PATH. GitHub stored-render workflow supplies FFmpeg/ffprobe; its result must be checked separately.
- Four existing framework deprecation warnings remain.

## Verified corrections
Finished media requires successful completion evidence and an actual export URL. Comparison previews cannot stand in for completed exports. Approvals bind to the exact rendered revision; subsequent media changes invalidate approval. Specific rejection/adjustment feedback is durable and influences future scoped recommendations without turning audio/framing defects into topic prohibitions. History and novel eligible candidates remain preserved.
Finish & Schedule checks eligible verified clips, copy and creator gates; saved intent is never described as a scheduler receipt. Actual receipt and asset-group identities remain required.
Optional future framing references do not force routine comparison decisions. Explicit owner decisions and requests for adjustment remain binding. Existing protected reviews are unchanged.
The existing collaborator component remains unchanged; regression coverage verifies its controls and durable approval are preserved. Clip editorial approval does not authorize invitations.
Exact analysis request hashes, effective live/fallback specification provenance, model, source/transcript/map versions and actual feedback references are recorded. Prepared analyses are not silently regenerated after review feedback.

## Production read-only findings
The legacy five-clip submission saved review choices and scheduling intent, but retained failed final renders. No matching scheduler acknowledgement or queue records were found. No production retry, rescheduling, deletion or replacement was performed. Historical provenance is not fabricated.
The owner retired those legacy clips from the new acceptance plan, while preserving all production/audit records.
Both configured selection Docs initially denied the deployed service identity and caused packaged fallback. Owner-authorized reader grants were applied only to those two Docs; both live content reads subsequently succeeded through Railway's existing service identity. No credentials or Doc contents changed.

## Framing scope and limitations
The dependent framing change preserves the whole source on moving blurred/darkened background for three/four participants, uncertain speaker selection and presentation context. Confident single whole tiles and existing stable confident two-person compositions remain supported. Persisted plans describe the effective graph.
Retained real TDOG source/diagnostic previews provide existing two/three-person and audible blurred-source evidence. They are not fresh acceptance exports. Real four-person footage and confident speaker-handoff examples remain unverified. No owner-approved boundaries or existing rendered assets were replaced.

## Acceptance and deployment safeguards
Live finished-card playback, fresh full-batch approval and actual scheduler delivery creation are NOT VERIFIED for this undeployed change.
Review both stacked PRs before separately authorizing merge/paused deployment. Back up persistent records; disable auto-deploy; ensure one worker with no overlap; preserve volume, credentials, selections, approvals, holds, receipts and schedules. Keep publishing disabled and Agape disabled.
After authorization, use a fresh normal full-batch workflow under the live selection Docs and topic-map revision. Do not impose clip quotas or choose clips for the owner. Require actual completed media, owner approval and durable scheduler receipts before claiming completion.

## GitHub renderer harness correction
The first dependent stored-render CI run exposed an AST-isolated test namespace missing json after framing evidence became unconditional. The production module already imports json. The harness now supplies that existing dependency; every audible-audio/render assertion is retained. Final CI result must be checked on the correction commit.
