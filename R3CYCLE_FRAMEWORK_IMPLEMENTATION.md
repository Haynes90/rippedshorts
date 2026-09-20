# Customer selection pipeline, version 2

Reference: https://docs.google.com/document/d/1jqzB-oXydEYGR4TSuJuxK0JFxbhtaxPh3fhxitrzf8I/edit

Customer branch only: `customer/r3cycle-pilot`. Personal entry point and ingestion remain unchanged.

1. Acquire YouTube through the original RapidAPI downloader contract (`RAPIDAPI_KEY`, optional `YOUTUBE_DL_HOST`, optional `YOUTUBE_DL_PATH_TEMPLATE`). Reuse validated local source first. Direct yt-dlp fallback is off unless `PILOT_YTDLP_FALLBACK=1`.
2. Trim selected ranges, cache word-aligned sentences, and analyze all selected sections together. Never join across an unselected gap.
3. Build dynamic source classification and a shared topic map.
4. Generate independent Shorts and highlight proposals using the map and same-account feedback.
5. Audit all proposals in one semantic review. Adjust sentence boundaries for context/core idea/payoff before presentation. Omit incomplete, duplicate, unsupported or over-duration candidates.
6. Save original proposals, semantic corrections, word boundaries and source fingerprint with each candidate. Present corrected candidates for human approval.
7. Render that approved semantic unit. Pad within neighboring word bounds, detect a safe ending pause, then verify exported boundary words. Do not run another semantic editor after approval.

Three cached text-model passes per selected source; no per-candidate semantic calls or automatic paid render retries. Failed or malformed stages fail closed. Topic, proposal and audit outputs remain in project state. Physical retention deletion is still manual.

Old selections can be rebuilt only before any clip is queued, rendering or rendered. Prior selections and decisions are archived in the project and still contribute same-account feedback. Cached section media/transcripts are reused. Automatic cross-account learning is not enabled.

Scope limits: this implements the document's source-understanding and boundary order. It does not yet implement favorites, manual boundary editing, publication feedback, weighted similarity retrieval, confidence scoring or permanent learning independent of job retention. Semantic quality is model-dependent and still requires human review. YouTube acquisition through a provider is not a guarantee against provider outages or quota limits.

Validation: 62 focused tests passed before deployment, covering the new pipeline, existing completion checks, review APIs and customer portal. Live render/download and live RapidAPI delivery must be recorded separately; unit tests do not prove them.
