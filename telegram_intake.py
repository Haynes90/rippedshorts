"""Telegram-triggered intake for reusable Drive sources and arbitrary YouTube videos."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import requests
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request

from audio_master_handoff import DB_PATH, SOURCE_DIR, connect, download_drive, drive_metadata, get_job
from source_ingestion import (
    ingest_with_audio_master,
    restrict_to_boundary,
    reuse_from_drive,
    select_non_overlapping,
)

router = APIRouter()

YOUTUBE_RE = re.compile(r"https?://(?:www\.)?(?:youtube\.com/(?:watch\?[^\s]*v=|shorts/)|youtu\.be/)([A-Za-z0-9_-]{6,20})", re.I)
DRIVE_RE = re.compile(r"https?://drive\.google\.com/(?:file/d/|open\?id=|uc\?(?:[^\s]*&)?id=)([A-Za-z0-9_-]+)", re.I)
_LOCK = threading.RLock()
logger = logging.getLogger("ripped-shorts.telegram")
RENDER_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(1, int(os.getenv("RIPPED_SHORTS_RENDER_WORKERS", "3")))
)
RIPPED_LOG_SHEET_ID = (
    os.getenv("RIPPED_SHORTS_LOG_SHEET_ID")
    or os.getenv("PODCAST_SHEET_ID")
    or "14VruBxjaaE9DyPSdBidMeuPHew3nHUC5sRNGXtLCsis"
).strip()
RIPPED_LOG_SHEET_TAB = os.getenv("RIPPED_SHORTS_LOG_SHEET_TAB", "Ripped Shorts").strip()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _notify_render_queue_complete(request_id: str, chat_id: str) -> None:
    """Send one Telegram summary when the currently approved render queue drains."""
    summary = None
    with _LOCK, _telegram_db() as db:
        row = db.execute(
            "SELECT * FROM telegram_requests WHERE request_id=?", (request_id,)
        ).fetchone()
        if not row:
            return
        state = json.loads(row["state_json"])
        reviews = dict(state.get("candidate_reviews") or {})
        statuses = {
            str(index): str(review.get("status") or "")
            for index, review in reviews.items()
            if str(review.get("status") or "")
            in {"queued", "rendering", "rendered", "render_failed"}
        }
        active = sum(status in {"queued", "rendering"} for status in statuses.values())
        rendered = sum(status == "rendered" for status in statuses.values())
        failed = sum(status == "render_failed" for status in statuses.values())
        if active or not (rendered or failed):
            return
        signature = json.dumps(statuses, sort_keys=True)
        if state.get("render_queue_completion_signature") == signature:
            return
        state["render_queue_completion_signature"] = signature
        db.execute(
            "UPDATE telegram_requests SET state_json=?, updated_at=? WHERE request_id=?",
            (json.dumps(state), now(), request_id),
        )
        summary = (rendered, failed)

    if summary:
        rendered, failed = summary
        folder_id = (
            os.getenv("DRIVE_FOLDER_ID") or os.getenv("Drive_Folder_ID") or ""
        ).strip()
        folder_line = (
            f"\nDrive folder: https://drive.google.com/drive/folders/{folder_id}"
            if folder_id
            else ""
        )
        send(
            chat_id,
            "✅ Current Ripped Shorts render queue complete"
            f"\nRendered: {rendered}"
            f"\nFailed: {failed}"
            f"{folder_line}",
        )


def _csv_env(name: str) -> set[str]:
    return {item.strip() for item in os.getenv(name, "").split(",") if item.strip()}


def _telegram_db() -> sqlite3.Connection:
    db = connect()
    db.execute(
        "CREATE TABLE IF NOT EXISTS telegram_requests ("
        "request_id TEXT PRIMARY KEY, update_id TEXT UNIQUE, chat_id TEXT NOT NULL, "
        "user_id TEXT NOT NULL, status TEXT NOT NULL, mode TEXT NOT NULL, source_kind TEXT NOT NULL, "
        "source_value TEXT NOT NULL, state_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    return db


def parse_request(text: str) -> dict[str, Any]:
    clean = (text or "").strip()
    youtube = YOUTUBE_RE.search(clean)
    drive = DRIVE_RE.findall(clean)
    lowered = clean.lower()
    if "topic" in lowered and not any(word in lowered for word in ("short", "highlight")):
        mode: Literal["shorts", "topics", "both"] = "topics"
    elif any(word in lowered for word in ("short", "highlight")) and "topic" not in lowered:
        mode = "shorts"
    else:
        # The currently deployed selector/renderer is short-form. Topic mode is
        # parsed explicitly so it can move to the dual-lane selector once that
        # pipeline lands, without silently producing the wrong aspect ratio.
        mode = "shorts"
    if youtube:
        return {"mode": mode, "source_kind": "youtube", "source_value": youtube.group(0), "video_id": youtube.group(1)}
    if drive:
        return {"mode": mode, "source_kind": "drive", "source_value": drive[0], "drive_ids": drive}
    raise ValueError("Send a YouTube link or a Google Drive video link.")


def _authorized(chat_id: str, user_id: str) -> bool:
    chats, users = _csv_env("TELEGRAM_ALLOWED_CHAT_IDS"), _csv_env("TELEGRAM_ALLOWED_USER_IDS")
    existing_chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if existing_chat:
        chats.add(existing_chat)
    # Fail closed: at least one allow-list must be configured.
    if not chats and not users:
        return False
    return (bool(chats) and chat_id in chats) or (bool(users) and user_id in users)


def telegram(method: str, payload: dict[str, Any]) -> dict:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured")
    response = requests.post(f"https://api.telegram.org/bot{token}/{method}", json=payload, timeout=(10, 60))
    if response.status_code != 200:
        raise RuntimeError(f"Telegram {method} failed ({response.status_code}): {response.text[:1000]}")
    return response.json()


def send(chat_id: str, text: str) -> None:
    telegram("sendMessage", {"chat_id": chat_id, "text": text, "disable_web_page_preview": True})


def _save(request_id: str, status: str, state: dict[str, Any]) -> None:
    with _LOCK, _telegram_db() as db:
        db.execute(
            "UPDATE telegram_requests SET status=?, state_json=?, updated_at=? WHERE request_id=?",
            (status, json.dumps(state), now(), request_id),
        )


def _reusable_youtube_job(video_id: str) -> dict | None:
    with _LOCK, connect() as db:
        rows = db.execute("SELECT * FROM source_jobs ORDER BY updated_at DESC").fetchall()
    for row in rows:
        manifest = json.loads(row["manifest_json"])
        state = json.loads(row["state_json"])
        if manifest.get("source_youtube_video_id") == video_id:
            video = Path(state.get("video_path", ""))
            transcript = Path(state.get("normalized_transcript_path", ""))
            if video.is_file() and transcript.is_file():
                return {"source_job_id": row["source_job_id"], "video_path": video, "transcript_path": transcript}
    return None


def _segments(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("segments", data) if isinstance(data, dict) else data


def validate_complete_candidates(payload: dict, transcript_segments: list[dict], tolerance: float = 0.75) -> dict:
    """Reject invented text and clip boundaries that cut through transcript segments."""
    boundaries = []
    for item in transcript_segments:
        start = float(item.get("start", 0))
        end = float(item.get("end", start + float(item.get("duration", 0))))
        boundaries.append((start, end, str(item.get("text", "")).strip()))
    valid, rejected = [], []
    for candidate in payload.get("segments", []):
        start = float(candidate.get("start", -1))
        end = float(candidate.get("end", start + float(candidate.get("duration", 0))))
        begins_cleanly = any(abs(start - seg_start) <= tolerance for seg_start, _, _ in boundaries)
        ends_cleanly = any(abs(end - seg_end) <= tolerance for _, seg_end, _ in boundaries)
        included = [text for seg_start, seg_end, text in boundaries if seg_start >= start - tolerance and seg_end <= end + tolerance]
        expected = " ".join(included).casefold()
        quoted = str(candidate.get("transcript", "")).strip().casefold()
        if begins_cleanly and ends_cleanly and quoted and expected and (quoted in expected or expected in quoted) and end > start:
            candidate.update({"start": start, "end": end, "duration": round(end - start, 3)})
            valid.append(candidate)
        else:
            rejected.append({"start": start, "end": end, "reason": "Unsupported text or incomplete transcript boundary"})
    return {**payload, "segments": valid[:20], "validation_rejections": rejected}


def _timecode(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _log_candidate_decision(
    state: dict[str, Any],
    request_id: str,
    index: int,
    decision: str,
    user_id: str,
    *,
    render_status: str,
    clip_url: str = "",
    rendered_at: str = "",
) -> None:
    """Upsert one candidate decision into the Podcast/Ripped Shorts worksheet."""
    import main

    candidate = state["result"]["segments"][index]
    analysis = state["result"].get("analysis") or {}
    parsed = state.get("parsed") or {}
    reviewed = (state.get("candidate_reviews") or {}).get(str(index), {})
    row_values = [
        now(),
        request_id,
        parsed.get("video_id", ""),
        parsed.get("source_value", ""),
        analysis.get("content_type", ""),
        analysis.get("main_theme", ""),
        ", ".join(str(value) for value in analysis.get("key_ideas", [])),
        ", ".join(str(value) for value in analysis.get("keywords", [])),
        index + 1,
        float(candidate.get("start", 0)),
        float(candidate.get("end", 0)),
        float(candidate.get("duration", 0)),
        candidate.get("category", ""),
        candidate.get("transcript", ""),
        candidate.get("reason", ""),
        decision,
        reviewed.get("reviewed_at", ""),
        rendered_at,
        clip_url,
        render_status,
        user_id,
    ]
    _, _, sheets = main.get_google_services()
    response = sheets.spreadsheets().values().get(
        spreadsheetId=RIPPED_LOG_SHEET_ID,
        range=f"'{RIPPED_LOG_SHEET_TAB}'!A:U",
    ).execute()
    rows = response.get("values", [])
    target_row = None
    for row_number, values in enumerate(rows[1:], start=2):
        if (
            len(values) > 8
            and str(values[1]) == request_id
            and str(values[8]) == str(index + 1)
        ):
            target_row = row_number
            break
    if target_row is None:
        sheets.spreadsheets().values().append(
            spreadsheetId=RIPPED_LOG_SHEET_ID,
            range=f"'{RIPPED_LOG_SHEET_TAB}'!A:U",
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": [row_values]},
        ).execute()
    else:
        sheets.spreadsheets().values().update(
            spreadsheetId=RIPPED_LOG_SHEET_ID,
            range=f"'{RIPPED_LOG_SHEET_TAB}'!A{target_row}:U{target_row}",
            valueInputOption="RAW",
            body={"values": [row_values]},
        ).execute()


def _safe_log_candidate(*args, **kwargs) -> None:
    try:
        _log_candidate_decision(*args, **kwargs)
    except Exception as exc:
        logger.exception("Ripped Shorts decision log failed: %s", exc)


def _send_candidates(chat_id: str, request_id: str, result: dict) -> None:
    clips = result.get("segments", [])
    send(chat_id, f"✅ Analysis complete\nJob ID: {request_id}\nComplete-thought candidates: {len(clips)}\n\nNothing has been rendered yet. Approve only the clips you want created.")
    for index, clip in enumerate(clips, 1):
        transcript = str(clip.get("transcript", "")).strip()
        if len(transcript) > 2600:
            transcript = transcript[:2597] + "..."
        text = (
            f"Candidate {index} of {len(clips)}\n\n"
            f"Time: {_timecode(float(clip['start']))}–{_timecode(float(clip['end']))}\n"
            f"Duration: {round(float(clip['duration']))} seconds\n"
            f"Category: {clip.get('category', 'social clip')}\n"
            f"Score: {clip.get('score', '')}\n\n"
            f"Why selected:\n{clip.get('reason', '')}\n\n"
            f"Transcript:\n{transcript}"
        )
        telegram("sendMessage", {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": True,
            "reply_markup": {"inline_keyboard": [[
                {"text": "✅ Approve & Render", "callback_data": f"rs:approve:{request_id}:{index-1}"},
                {"text": "❌ Reject", "callback_data": f"rs:reject:{request_id}:{index-1}"},
            ]]},
        })


def _transcribe(video_path: Path) -> list[dict]:
    """Extract audio and ask OpenAI for timestamped segments when Drive has no transcript."""
    import subprocess

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to transcribe a Drive video without a transcript")
    audio = video_path.with_suffix(".mp3")
    completed = subprocess.run(
        [os.getenv("FFMPEG_BINARY", "ffmpeg"), "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", str(audio)],
        capture_output=True, text=True, timeout=3600,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {completed.stderr[-1000:]}")
    with audio.open("rb") as handle:
        response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            data={"model": os.getenv("TRANSCRIPTION_MODEL", "whisper-1"), "response_format": "verbose_json", "timestamp_granularities[]": "segment"},
            files={"file": (audio.name, handle, "audio/mpeg")}, timeout=(10, 1800),
        )
    if response.status_code != 200:
        raise RuntimeError(f"Transcription failed ({response.status_code}): {response.text[:1000]}")
    result = []
    for item in response.json().get("segments", []):
        start, end = float(item["start"]), float(item["end"])
        result.append({"start": start, "end": end, "duration": end - start, "text": str(item["text"]).strip()})
    if not result:
        raise RuntimeError("Transcription returned no timed segments")
    return result


def _process(request_id: str) -> None:
    with _LOCK, _telegram_db() as db:
        row = db.execute("SELECT * FROM telegram_requests WHERE request_id=?", (request_id,)).fetchone()
    if not row:
        return
    state, chat_id = json.loads(row["state_json"]), row["chat_id"]
    try:
        _save(request_id, "processing", {**state, "stage": "source_resolution"})
        send(chat_id, f"⬇️ Retrieving source\nJob ID: {request_id}")
        work = SOURCE_DIR / f"telegram-{request_id}"
        work.mkdir(parents=True, exist_ok=True)
        if row["source_kind"] == "drive":
            ids = state["parsed"]["drive_ids"]
            video_id = ids[0]
            meta = drive_metadata(video_id)
            video = download_drive(video_id, work / "source-video.mp4", meta)
            if len(ids) > 1:
                transcript_meta = drive_metadata(ids[1])
                transcript_path = download_drive(ids[1], work / "source-transcript", transcript_meta)
                from audio_master_handoff import parse_transcript
                segments = parse_transcript(transcript_path)
                for item in segments:
                    item["duration"] = item["end"] - item["start"]
            else:
                send(chat_id, "📝 No transcript link supplied; transcribing the Drive video.")
                segments = _transcribe(video)
            reused = True
        else:
            parsed = state["parsed"]
            video_id = parsed["video_id"]
            boundary = None
            cache = {"video_path": None, "segments": [], "sermon_boundary": None}
            try:
                send(chat_id, f"🔎 Checking Google Drive for existing files containing YouTube ID {video_id}.")
                cache = reuse_from_drive(video_id, work)
            except Exception as cache_error:
                state.setdefault("warnings", []).append(f"Drive cache lookup failed: {cache_error}")

            reusable = _reusable_youtube_job(video_id)
            video = cache.get("video_path")
            segments = cache.get("segments") or []
            boundary = cache.get("sermon_boundary")
            reused = bool(video or segments)
            if reusable:
                video = video or reusable["video_path"]
                segments = segments or _segments(reusable["transcript_path"])
                reused = True
            if not video or not segments:
                missing = []
                if not video:
                    missing.append("source video")
                if not segments:
                    missing.append("timed transcript")
                send(
                    chat_id,
                    "🎧 Drive cache is missing "
                    + " and ".join(missing)
                    + ". Preparing the reusable video and transcript now.",
                )
                audio_master_result = ingest_with_audio_master(video_id, parsed["source_value"])
                state["audio_master_ingest"] = {
                    "job_id": audio_master_result.get("job_id"),
                    "status": audio_master_result.get("status"),
                    "skip_metadata_podhome": audio_master_result.get("skip_metadata_podhome"),
                }
                cache = reuse_from_drive(video_id, work)
                video = cache.get("video_path")
                segments = cache.get("segments") or []
                boundary = cache.get("sermon_boundary") or boundary
                if not video or not segments:
                    raise RuntimeError(
                        "Audio Master completed but the reusable source video or timed transcript "
                        f"for YouTube ID {video_id} was not found in the configured Drive folder."
                    )
                reused = True

            if boundary:
                bounded = restrict_to_boundary(segments, boundary)
                if bounded:
                    segments = bounded
                    state["sermon_boundary"] = boundary
                    send(
                        chat_id,
                        f"✂️ Reusing approved sermon boundary: {_timecode(boundary['start'])}–{_timecode(boundary['end'])}.",
                    )
                else:
                    state.setdefault("warnings", []).append("Approved sermon boundary did not contain reusable transcript segments.")

        _save(request_id, "selecting", {**state, "stage": "selection", "video_path": str(video), "source_reused": reused})
        send(chat_id, "🧠 Reviewing the full eligible transcript and selecting up to 20 distinct, non-overlapping complete thoughts.")
        import main
        enriched = [{**item, "video_id": state["parsed"].get("video_id", "drive-source")} for item in segments]
        result = main.call_openai_for_clips(enriched, None)
        result = validate_complete_candidates(result, enriched)
        result["segments"] = select_non_overlapping(result.get("segments", []), limit=20)
        # Existing renderer is shorts-only. Topic requests are retained for the dual-lane selector.
        if row["mode"] == "topics":
            raise RuntimeError("Topic-only Telegram processing requires the dual-lane selector before rendering")
        clips = result.get("segments", [])
        _save(request_id, "awaiting_review", {**state, "stage": "awaiting_review", "video_path": str(video), "source_reused": reused, "result": result, "candidate_reviews": {}})
        _send_candidates(chat_id, request_id, result)
    except Exception as exc:
        _save(request_id, "error", {**state, "stage": "error", "error_type": type(exc).__name__, "error": str(exc), "retryable": True})
        send(chat_id, f"❌ Processing failed\nJob ID: {request_id}\n{str(exc)[:1500]}\n\nSend /retry {request_id} to try again.")


def _accept_update(update: dict, background_tasks: BackgroundTasks) -> dict:
    callback = update.get("callback_query") or {}
    message = callback.get("message") or update.get("message") or update.get("edited_message") or {}
    chat_id = str((message.get("chat") or {}).get("id", ""))
    user_id = str((callback.get("from") or message.get("from") or {}).get("id", ""))
    callback_data = str(callback.get("data") or "")
    text = str(message.get("text") or message.get("caption") or "").strip()
    if not chat_id or not user_id:
        return {"status": "ignored"}
    if not _authorized(chat_id, user_id):
        return {"status": "unauthorized"}
    action = re.fullmatch(r"rs:(approve|reject):([A-Za-z0-9-]+):(\d+)", callback_data)
    if action:
        verb, request_id, index_text = action.groups()
        index = int(index_text)
        with _LOCK, _telegram_db() as db:
            row = db.execute("SELECT * FROM telegram_requests WHERE request_id=?", (request_id,)).fetchone()
        if not row:
            return {"status": "not_found"}
        with _LOCK, _telegram_db() as db:
            current = db.execute(
                "SELECT * FROM telegram_requests WHERE request_id=?", (request_id,)
            ).fetchone()
            state = json.loads(current["state_json"])
            reviews = dict(state.get("candidate_reviews") or {})
            existing_status = (reviews.get(str(index)) or {}).get("status")
            if verb == "approve" and existing_status in {"queued", "rendering", "rendered"}:
                return {
                    "status": f"already_{existing_status}",
                    "request_id": request_id,
                    "candidate_index": index,
                }
            reviews[str(index)] = {
                "status": "queued" if verb == "approve" else "reject",
                "reviewed_at": now(),
                "user_id": user_id,
            }
            state["candidate_reviews"] = reviews
            if verb == "approve":
                state.pop("render_queue_completion_signature", None)
            db.execute(
                "UPDATE telegram_requests SET status=?, state_json=?, updated_at=? WHERE request_id=?",
                (
                    "awaiting_review",
                    json.dumps(state),
                    now(),
                    request_id,
                ),
            )
        if verb == "approve":
            _safe_log_candidate(
                state,
                request_id,
                index,
                "approved",
                user_id,
                render_status="queued",
            )
            RENDER_EXECUTOR.submit(_render_approved, request_id, index, chat_id)
            send(
                chat_id,
                f"Candidate {index + 1} approved and queued for rendering. "
                f"Up to {RENDER_EXECUTOR._max_workers} clips render at once; the rest wait.",
            )
        else:
            _safe_log_candidate(
                state,
                request_id,
                index,
                "rejected",
                user_id,
                render_status="not_rendered",
            )
            send(chat_id, f"Candidate {index + 1} rejected.")
        return {"status": verb, "request_id": request_id, "candidate_index": index}
    retry_match = re.fullmatch(r"/retry\s+([A-Za-z0-9-]+)", text, re.I)
    if retry_match:
        request_id = retry_match.group(1)
        background_tasks.add_task(_process, request_id)
        return {"status": "retry_accepted", "request_id": request_id}
    try:
        parsed = parse_request(text)
    except ValueError:
        return {"status": "ignored_non_ripped_shorts_message"}
    request_id, update_id, stamp = str(uuid.uuid4()), str(update.get("update_id", "")), now()
    state = {"stage": "accepted", "parsed": parsed, "message_id": message.get("message_id")}
    try:
        with _LOCK, _telegram_db() as db:
            db.execute("INSERT INTO telegram_requests VALUES (?,?,?,?,?,?,?,?,?,?,?)", (request_id, update_id, chat_id, user_id, "accepted", parsed["mode"], parsed["source_kind"], parsed["source_value"], json.dumps(state), stamp, stamp))
    except sqlite3.IntegrityError:
        return {"status": "duplicate_update"}
    send(chat_id, f"✅ Ripped Shorts request accepted\nSource: {parsed['source_kind'].title()}\nJob ID: {request_id}")
    background_tasks.add_task(_process, request_id)
    return {"status": "accepted", "request_id": request_id}


def _render_approved(request_id: str, index: int, chat_id: str) -> None:
    user_id = ""
    try:
        with _LOCK, _telegram_db() as db:
            row = db.execute(
                "SELECT * FROM telegram_requests WHERE request_id=?", (request_id,)
            ).fetchone()
        if not row:
            raise RuntimeError(f"Ripped Shorts request not found: {request_id}")
        state = json.loads(row["state_json"])
        reviews = dict(state.get("candidate_reviews") or {})
        reviews[str(index)] = {
            **reviews.get(str(index), {}),
            "status": "rendering",
            "render_started_at": now(),
        }
        state["candidate_reviews"] = reviews
        with _LOCK, _telegram_db() as db:
            db.execute(
                "UPDATE telegram_requests SET state_json=?, updated_at=? WHERE request_id=?",
                (json.dumps(state), now(), request_id),
            )
        send(chat_id, f"🎬 Candidate {index + 1} is now rendering.")
        candidate = state["result"]["segments"][index]
        user_id = str(
            (state.get("candidate_reviews") or {}).get(str(index), {}).get("user_id", "")
        )
        video = Path(state["video_path"])
        import main

        payload_candidate = dict(candidate)
        payload_candidate["candidate_number"] = index + 1
        payload = {"segments": [payload_candidate]}
        rendered = main.attach_clip_assets(
            payload,
            state["parsed"].get("video_id", request_id),
            None,
            video_path_override=video,
        )
        clip = rendered["segments"][0]
        rendered_at = now()
        with _LOCK, _telegram_db() as db:
            latest = db.execute(
                "SELECT * FROM telegram_requests WHERE request_id=?", (request_id,)
            ).fetchone()
            latest_state = json.loads(latest["state_json"])
            reviews = dict(latest_state.get("candidate_reviews") or {})
            reviews[str(index)] = {
                **reviews.get(str(index), {}),
                "status": "rendered",
                "clip_url": clip.get("clip_url"),
                "rendered_at": rendered_at,
            }
            latest_state["candidate_reviews"] = reviews
            db.execute(
                "UPDATE telegram_requests SET status=?, state_json=?, updated_at=? WHERE request_id=?",
                ("awaiting_review", json.dumps(latest_state), now(), request_id),
            )
        _safe_log_candidate(
            latest_state,
            request_id,
            index,
            "approved",
            user_id,
            render_status="rendered",
            clip_url=clip.get("clip_url", ""),
            rendered_at=rendered_at,
        )
        send(
            chat_id,
            f"✅ Candidate {index + 1} rendered and uploaded to DRIVE_FOLDER_ID:\n"
            f"{clip.get('clip_url', '')}",
        )
        _notify_render_queue_complete(request_id, chat_id)
    except Exception as exc:
        logger.exception(
            "Candidate render failed request_id=%s candidate=%s", request_id, index + 1
        )
        try:
            with _LOCK, _telegram_db() as db:
                latest = db.execute(
                    "SELECT * FROM telegram_requests WHERE request_id=?", (request_id,)
                ).fetchone()
                latest_state = json.loads(latest["state_json"])
                reviews = dict(latest_state.get("candidate_reviews") or {})
                reviews[str(index)] = {
                    **reviews.get(str(index), {}),
                    "status": "render_failed",
                    "render_error": str(exc),
                    "rendered_at": now(),
                }
                latest_state["candidate_reviews"] = reviews
                db.execute(
                    "UPDATE telegram_requests SET status=?, state_json=?, updated_at=? WHERE request_id=?",
                    ("awaiting_review", json.dumps(latest_state), now(), request_id),
                )
            _safe_log_candidate(
                latest_state,
                request_id,
                index,
                "approved",
                user_id,
                render_status="render_failed",
                rendered_at=now(),
            )
        except Exception:
            logger.exception("Could not persist render failure")
        send(chat_id, f"❌ Candidate {index + 1} render failed:\n{str(exc)[:1500]}")
        _notify_render_queue_complete(request_id, chat_id)




@router.get("/api/ripped-shorts/runtime-info")
def ripped_shorts_runtime_info() -> dict:
    audio_master_url = (os.getenv("AUDIO_MASTER_INTERNAL_URL") or "").strip()
    if audio_master_url and "://" not in audio_master_url:
        audio_master_url = f"https://{audio_master_url}"
    return {
        "service_role": os.getenv("SERVICE_ROLE", "").strip(),
        "audio_master_internal_url": audio_master_url.rstrip("/"),
        "audio_master_ingest_secret_set": bool(
            os.getenv("AUDIO_MASTER_INGEST_SECRET")
            or os.getenv("AUDIO_MASTER_WEBHOOK_SECRET")
        ),
        "drive_folder_id_set": bool(
            os.getenv("DRIVE_FOLDER_ID") or os.getenv("Drive_Folder_ID")
        ),
        "google_credentials_set": bool(
            os.getenv("GOOGLE_CREDENTIALS")
            or (os.getenv("GOOGLE_CLIENT_EMAIL") and os.getenv("GOOGLE_PRIVATE_KEY"))
        ),
        "openai_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "telegram_bot_token_set": bool(os.getenv("TELEGRAM_BOT_TOKEN")),
        "telegram_chat_id_set": bool(os.getenv("TELEGRAM_CHAT_ID")),
        "ingestion_mode": "drive_id_reuse_then_audio_master",
        "podhome_mode": "disabled_for_ripped_shorts_ingest",
    }


@router.post("/api/telegram/webhook")
async def telegram_gateway(request: Request, x_telegram_bot_api_secret_token: str | None = Header(None)):
    """Clip Master owns Telegram and forwards only Ripped Shorts messages."""
    if os.getenv("SERVICE_ROLE", "").strip().lower() != "clip_master":
        raise HTTPException(status_code=404, detail="Telegram webhook is owned by Clip Master")
    expected = os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()
    if not expected or x_telegram_bot_api_secret_token != expected:
        raise HTTPException(status_code=401, detail="Invalid Telegram webhook secret")
    update = await request.json()
    target = os.getenv("RIPPED_SHORTS_INTERNAL_URL", "").rstrip("/")
    secret = os.getenv("RIPPED_SHORTS_SHARED_SECRET", "").strip()
    if not target or not secret:
        raise HTTPException(status_code=503, detail="Ripped Shorts forwarding is not configured")
    response = requests.post(f"{target}/api/ripped-shorts/intake", json=update, headers={"x-ripped-shorts-secret": secret}, timeout=(10, 60))
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Ripped Shorts intake failed: {response.text[:1000]}")
    return response.json()


@router.post("/api/ripped-shorts/intake")
async def internal_intake(request: Request, background_tasks: BackgroundTasks, x_ripped_shorts_secret: str | None = Header(None)):
    expected = os.getenv("RIPPED_SHORTS_SHARED_SECRET", "").strip()
    if not expected or x_ripped_shorts_secret != expected:
        raise HTTPException(status_code=401, detail="Invalid Ripped Shorts service secret")
    return _accept_update(await request.json(), background_tasks)
