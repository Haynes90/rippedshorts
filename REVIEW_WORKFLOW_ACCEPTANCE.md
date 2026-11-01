# Ripped Shorts #77/#78 — Editorial, Framing and Website Acceptance

Status: **NEEDS ATTENTION — draft, unmerged, not deployed.** Updated November 1, 2026 (America/New_York). The tests below executed October 31 local time. This document supersedes older test counts in the PR descriptions and preserves the previous acceptance report as history.

## Implemented changes

- Existing `caption_review_web.py` portal now displays original/proposed/context videos, editorial actions and separate framing actions. Decisions have immutable comparison revisions, optimistic versions, saved feedback and actor evidence in additive tables in the existing SQLite database. No replacement application, database or worker.
- Token-authenticated private MP4 streaming supports browser byte ranges, request isolation, expiry, retained-storage restrictions and artifact integrity checks. Partial Drive downloads cannot become cache hits. No public sharing.
- `clip_review_state.py` keeps original selections intact, freezes submitted assets, rejects superseded controls, binds framing previews to editorial bounds and prevents old renders satisfying new approval revisions. Only the existing explicit render action resolves selected bounds/framing; decision POSTs never render, hand off, publish or invite collaborators.
- `clip_review_ui.py` extends the same portal. Previous bad framing can be shown as a diagnostic reference, never as an approved smart option. Smart alternatives are offered only when the renderer has appropriate audiovisual evidence. Existing caption/title/description controls and batch creator review remain.
- `review_navigation.py` uses the existing bot, saved authorized chat and `durable_outbox`. **🌐 Open Clip Review** links target Shorts, highlights, editorial, framing or final stages. Acknowledged message IDs are retained; unchanged notifications deduplicate, expired links edit the known message, and uncertain sends hold for verification instead of resending blindly. Telegram hooks were added to the existing candidate/render review paths.
- `production_intelligence.py` consumes editorial/framing feedback separately so one does not erase the other's influence. Boundary changes and framing preferences reach future scoped prompts without being misrepresented as explicit universal owner instructions. Existing topic maps, concurrent format analyses, ordered Telegram selection and Sheets projection are preserved.
- #78 changes `main.py`, `active_speaker.py` and `camera_layout.py`: uncertainty uses a sharp full-frame foreground over a moving enlarged, blurred and moderately darkened duplicate. Unsafe legacy center/two-face fallback is no longer executed. Confident tile compositions and stable audiovisual switching remain. A separately selected blurred render cannot silently become a smart crop, or vice versa.
- The existing explicit render path can retain a blurred framing preview and notify the owner. Preview preparation failure does not erase the successful export; scheduling remains blocked for that asset. Failed siblings remain independent.
- Final-review JavaScript saves only actual copy cards, excluding workflow/comparison panels. Final and downstream gates require matching verified render revisions; editorial/framing approval alone is insufficient.

## Actual test commands and outcomes

Both final frozen snapshots were tested, with mocked external operations and `validation_network` denying external sockets (loopback permitted):

```powershell
# From ripped-intelligence (combined #77/#78 snapshot)
$env:PYTHONUTF8='1'
$env:TEMP=(Resolve-Path ../ripped-intelligence/.test-tmp).Path
$env:TMP=$env:TEMP
$env:DATA_DIR="$env:TEMP/combined-release-review-data"
$env:PYTHONPATH="$(Resolve-Path ../ripped-intelligence/.test-tmp/opencv4);$pwd;$(Resolve-Path ../pr-validation-handoff)"
& ../rippedshorts/.venv/Scripts/python.exe -m pytest -p validation_network tests -q --basetemp=../ripped-intelligence/.test-tmp/combined-release-review --junitxml=../ripped-intelligence/.test-tmp/combined-release-review.xml

# From ripped-intelligence-source (#77 alone); same setup, isolated DATA_DIR
# and basetemp/JUnit names source-release-review.
& ../rippedshorts/.venv/Scripts/python.exe -m pytest -p validation_network tests -q --basetemp=../ripped-intelligence/.test-tmp/source-release-review --junitxml=../ripped-intelligence/.test-tmp/source-release-review.xml
```

| Suite | Passed | Failed | Skipped | Additional evidence |
|---|---:|---:|---:|---|
| Combined final snapshot | 389 | 0 | 2 | Two subtests passed; 17.66 seconds |
| Source-only final snapshot | 371 | 0 | 2 | Two subtests passed; 14.09 seconds |

The two local skips require FFprobe on PATH. Real local FFmpeg/OpenCV acceptance below is separate; the skips are not called passes. Four deprecation warnings concern existing FastAPI/httpx/anyio APIs. Counts overlap and must not be summed.

Eleven new HTTP/state tests cover independent saved decisions, feedback, stale controls, submitted-asset freeze, token/request isolation, byte ranges, expiry/renewal, Telegram receipts and uncertainty, immutable original selection, render revision mismatch, scoped learning, preview tampering, superseded reviews, sibling isolation and unavailable smart framing. Updated crop tests retain pixel/audio/dimension assertions while replacing obsolete forced-crop assumptions with full-frame preservation. Synthetic audiovisual handoffs still prove graph switching; they do not prove real-world speaker identification.

## Real retained media and website demonstration

Request `94893455-5f48-46de-bdfb-642212ea19a5`; Audio Master job `b7d6bbd5-b48b-476a-b5c0-a551e53f6490`; source `9mW0bhmUaw0`. Retained source SHA256: `45314424b1529f9d169f8d08c4fce31ddab10b17b122ed107c98458376b80a67`.

All twenty existing comparison videos were reused. Three new full-length blurred previews use the **original** Short 9/10/11 bounds, not approved replacements. Each fully decoded, has 1080x1920 video and non-silent retained audio. Original bounds remain 2567.82–2620.22, 2651.76–2715.76 and 2791.34–2861.64 respectively.

| Asset | Decoded video duration | Audio RMS | Audio/video duration difference |
|---|---:|---:|---:|
| Short 9 | 52.367s | 0.15648 | 0.041s |
| Short 10 | 63.967s | 0.15055 | 0.051s |
| Short 11 | 70.267s | 0.15207 | 0.043s |

Inspected actual output frames preserve the entire three-person source and the native two-person interval at source 2664–2674s, including labels/graphics. Background frames move with the source; foreground is not stretched or cropped. These checks establish actual retained-media decoding, composition and sound energy, not owner approval of the speech cuts. No claim of a full human listening pass or acoustically approved final exports.

The **existing portal/router**, served on loopback with an isolated SQLite database and explicit ISOLATED ACCEPTANCE DEMO banner, loaded all twenty existing comparison media plus the three blurred files. Chrome reported usable metadata/readiness for all 23 initial video elements, including both full highlights. Native controls advanced the unmuted TDOG preview. A demonstration-only Keep Original decision persisted through reload. Mobile viewport 390px showed no horizontal overflow. Later framing-reference panels reuse the same original files side by side; they do not create new media or approve the old crop.

Replay tools: `tools/render_retained_blurred_acceptance.py` and `tools/serve_retained_review_demo.py`. They require the retained acceptance artifacts already on disk and never fetch YouTube, transcribe, create production intake or connect the demo to production. Run with the repository root on PYTHONPATH. The demo stores only under `.test-tmp/tdog-review-demo`, binds loopback and refuses final scheduling.

Local evidence: `acceptance-media/boundary-review/blurred-media-validation.json`, retained original/proposed/context manifest and decode/audio evidence. Media files and local bearer tokens are not committed to the public repository.

## Five production decisions remain pending

| Clip | Original seconds | Proposed seconds | Owner decision |
|---|---|---|---|
| Short 9 | 2567.82–2620.22 | 2567.82–2641.04 | Keep original or include harvesting method/payoff |
| Short 10 | 2651.76–2715.76 | 2646.12–2716.16 | Keep original or include full question/ending margin |
| Short 11 | 2791.34–2861.64 | 2848.14–2916.66 | Judge context of narrower complete explanation |
| Highlight 1 | 2551.66–2916.66 | 2544.64–2916.66 | Keep original or include prior transition |
| Highlight 2 | 3171.76–3624.04 | 3171.76–3621.18 | Judge substantive ending versus closing transition |

Segment timing is not fabricated word timing. ASR omissions/filler differences alone do not establish an unsafe cut. These are editorial alternatives requiring listening. Request Another Edit/Framing Adjustment saves feedback and leaves the review pending until a revised verified preview is registered; it does not pretend that a replacement has already rendered. If a boundary changes, framing previews for the old interval cannot approve the new interval.

## Outstanding gates and exact next action

| Gate | Result |
|---|---|
| Isolated website and decision persistence | PASS |
| Real safe fallback, retained two/three-person footage | PASS for full-source preservation; owner quality approval pending |
| Real confident speaker handoffs | NOT VERIFIED; synthetic handoff tests only |
| Real four-person footage | NOT VERIFIED; no suitable accessible verified fixture obtained |
| Sheets topic/learning roundtrip | Prior isolated Railway-identity pass retained; not rerun or relabelled as a new production test |
| New Telegram website link delivery | Mocked adapter/receipt checks PASS; actual new-feature production link NOT SENT |
| All five final corrected exports/scheduler records | NOT VERIFIED; original request not resumed and owner choices not made |
| Production settings/queue | No changes in this pass; not freshly audited as live configuration evidence |

**Owner action:** review these draft changes and authorize a paused rollout separately when acceptable. Current authorization expressly forbids merge/deploy/production approval writes. Therefore live Telegram-to-new-website acceptance cannot honestly be declared complete now. Do not send a production URL suggesting the undeployed features are live.

Before any later rollout: confirm auto-deploy disabled; back up persistent DB/approvals/queue/receipts and compare protected request/selection hashes; ensure one service replica and persistent DATA_DIR; stop the old worker before replacement; preserve credentials, schedules, audit intervals, holds, destination mappings and Agape settings; verify global publishing remains false. Review/merge #77 then reconciled #78 only after separate authorization and current checks. Deploy Ripped Shorts only, verify SHA/health/storage. Register the retained verified comparison descriptors into the new separate review tables, using exact existing request/asset IDs and existing media IDs; stage blurred files in retained private storage. Do not modify original selections or resume the job. Then invoke the receipt-backed notification for the editorial revision to the request's saved authorized chat. Verify Telegram acknowledgment and actual owner website playback. Owner editorial/framing choices must precede separately authorized bounded rendering; final copy/creator/playback approval precedes scheduling. No publication or real Collab invitation is authorized.
