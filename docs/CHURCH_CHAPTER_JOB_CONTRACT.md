# Church Chapter Job Contract

Clip Master accepts church chapter-analysis work from Audio Master at `POST /clip-jobs`.

Required lifecycle:

1. Audio Master matches the source through its Show Config sheet.
2. Audio Master transcribes the full service with timed segments.
3. Audio Master sends the timed transcript, prompt reference, source audio, show code, and configured Podhome destination to Clip Master.
4. Clip Master identifies sermon, sermonette, and other publishable chapters.
5. Clip Master stores the job as `waiting_for_approval`.
6. Clip Master sends a Telegram message listing the proposed chapters.
7. Podhome publishing remains blocked until human approval.

Operational endpoints:

- `GET /clip-jobs` — all church chapter jobs
- `GET /clip-jobs/{job_id}` — one job and its proposed chapters
- `GET /approvals` — jobs waiting for approval
- `GET /clip-master-info` — runtime and Telegram configuration status

Required Railway variables:

- `TELEGRAM_BOT_TOKEN` (or legacy `TELEGRAM_TOKEN`)
- `TELEGRAM_CHAT_ID` (or legacy `TELEGRAM_CHANNEL_ID`)

The startup log must contain `CLIP MASTER STARTUP: church-chapter-jobs-v1.0`.
