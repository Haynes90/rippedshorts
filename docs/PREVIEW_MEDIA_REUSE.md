# Preview-to-website acceptance (PR92, depends on PR91)

## Implemented

New TCB/TDOG discovery can enter preview-first mode with RIPPED_PREVIEW_FIRST_ENABLED=true (default false). Existing selected/reviewed requests cannot be automatically migrated. No production setting was changed.

The existing bounded render executor processes Shorts and highlights independently. Each candidate receives PR91's persisted source/transcript/revision-bound specification before rendering. A successful matching preview receipt bypasses provider, FFmpeg and upload work. A claimed but uncertain render is held; a verified export is retained before upload so a recoverable upload operation can reuse it. Existing Drive reconciliation checks checksum, size, folder and correlation tag and never blindly repeats an uncertain create.

Both existing render adapters consume exact saved timestamps. Export verification remains mandatory. Source acoustics are measured; verified word timing can support a fully captured passage without compulsory silence. The 100ms word-timing allowance is a conservative planning check, not proof of phoneme precision. Segment-only uncertain cuts remain held. No legacy endpoint edits are applied.

The existing authenticated review website lists the entire preview pool before selection, with playable media, transcript, rationale, independent decisions and explicit playback/framing confirmation. Only explicit approval promotes that exact preview into selected media. Copy drafts remain pending and collaborator/copy/scheduling approvals remain separate. Rejecting, requesting an edit or leaving unused never deletes retained media. Caption-only changes do not change the media revision or render again. SQLite receipts survive a new process. No scheduling is triggered by preview generation or selection.

Existing cleanup now keeps a durable completion eligibility clock across requests sharing a source. Active jobs, unresolved reviews/edits/holds and unscheduled selected work protect local media. Verified completion or explicit owner closure starts the existing retention interval; reopening resets it. Emergency cleanup does not shorten this source interval. Drive media and protected database/backup files are unaffected.

## Verification and limits

Full isolated suite: python -m pytest -p validation_network -q -ra --tb=short. Most recent completed run: 675 passed, 0 failed, 6 local tool-dependent skips, 2 subtests passed. Final CI is authoritative for FFmpeg/FFprobe cases.

New tests exercise real SQLite, authenticated existing HTTP routes, candidate discovery dispatch, explicit approvals, sibling isolation, process restart, immutable media reuse, seven-day-old shared caches and normal/emergency completion expiry. Drive API operations and semantic judgments are explicitly mocked; these are not live provider or podcast listening results.

CI installs FFmpeg/FFprobe and renders both a 1080x1920 Short and a 1920x1080 highlight exceeding 180 seconds from generated synthetic audiovisual media, through the existing adapters and exported-media checks. Tests fetch authenticated media (including range requests), reject unauthorized access, approve the preview and verify no duplicate export/upload. Synthetic website evidence is retained as a three-day CI artifact. Synthetic tones are not evidence of natural speech or real creator playback acceptance.

## Controlled live acceptance still required

Keep both PRs draft and unmerged until reviewed. Dependency order: PR91 then PR92. Production remains untouched; the six protected TDOG assets, successful Short13, five holds and provisional boundary suggestions are unchanged.

After separate merge/deployment authorization: back up durable state and queue, verify one worker and persistent storage, deploy paused with publishing/Agape disabled, and enable the preview flag only for a newly authorized unprotected test request. Verify actual source speech, Drive service identity/uploads, both website previews and owner playback/framing/copy/collaborator decisions. Do not retry the protected five assets or rerender Short13. After separate scheduling authorization, use the existing idempotent handoff and read back authoritative Schedule Master asset/destination/date/receipt records. Existing scheduling and reconciliation code is not replaced by this PR. Live Drive, actual scheduling and paused-production end-to-end acceptance are NOT VERIFIED by isolated tests.
