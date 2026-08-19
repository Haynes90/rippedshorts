"""Telegram-triggered intake for reusable Drive sources and arbitrary YouTube videos."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import requests
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request

from audio_master_handoff import DB_PATH, SOURCE_DIR, connect, download_drive, drive_metadata, get_job

router = APIRouter()

YOUTUBE_RE = re.compile(r"https?://(?:www\.)?(?:youtube\.com/(?:watch\?[^\s]*v=|shorts/)|youtu\.be/)([A-Za-z0-9_-]{6,20})", re.I)
DRIVE_RE = re.compile(r"https?://drive\.google\.com/(?:file/d/|open\?id=|uc\?(?:[^\s]*&)?id=)([A-Za-z0-9_-]+)", re.I)
_LOCK = threading.RLock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    # Reuse the existing Clip Master chat variable when the bot already has a
    # single approved review chat. No second Telegram bot is required.
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
    """Keep only candidates whose boundaries and quoted words exist in the transcript."""
    boundaries = []
    for item in transcript_segments:
        start = float(item.get("start", 0))
        end = float(item.get("end", start + float(item.get("duration", 0))))
        boundaries.append((start, end, str(item.get("text", "")).strip()))
    valid = []
    rejected = []
    for candidate in payload.get("segments", []):
        start = float(candidate.get("start", -1))
        end = float(candidate.get("end", start + float(candidate.get("duration", 0))))
        begins_cleanly = any(abs(start - seg_start) <= tolerance for seg_start, _, _ in boundaries)
        ends_cleanly = any(abs(end - seg_end) <= tolerance for _, seg_end, _ in boundaries)
        included = [text for seg_start, seg_end, text in boundaries if seg_start >= start - tolerance and seg_end <= end + tolerance]
        expected = " ".join(included).casefold()
        quoted = str(candidate.get("transcript", "")).strip().casefold()
        transcript_matches = bool(quoted and expected and (quoted in expected or expected in quoted))
        if begins_cleanly and ends_cleanly and transcript_matches and end > start:
            candidate["start"] = start
            candidate["end"] = end
            candidate["duration"] = round(end - start, 3)
            valid.append(candidate)
        else:
            rejected.append({"start": start, "end": end, "reason": "Candidate did not align to complete transcript boundaries or quoted unsupported text"})
    return {**payload, "segments": valid, "validation_rejections": rejected}


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
            reusable = _reusable_youtube_job(parsed["video_id"])
            if reusable:
                video, segments, reused = reusable["video_path"], _segments(reusable["transcript_path"]), True
            else:
                import main
                video = main.download_youtube_video(parsed["video_id"], parsed["source_value"], work)
                segments = main.get_transcript(parsed["video_id"])
                reused = False
        _save(request_id, "selecting", {**state, "stage": "selection", "video_path": str(video), "source_reused": reused})
        send(chat_id, "🧠 Selecting clip candidates.")
        import main
        enriched = [{**item, "video_id": state["parsed"].get("video_id", "drive-source")} for item in segments]
        result = main.call_openai_for_clips(enriched, None)
        result = validate_complete_candidates(result, enriched)
        # Existing renderer is shorts-only. Topic requests are retained for the dual-lane selector.
        if row["mode"] == "topics":
            raise RuntimeError("Topic-only Telegram processing requires the dual-lane selector before rendering")
        result = main.attach_clip_assets(result, state["parsed"].get("video_id", request_id), state["parsed"].get("source_value") if row["source_kind"] == "youtube" and not reused else None, video_path_override=video)
        clips = result.get("segments", [])
        _save(request_id, "completed", {**state, "stage": "completed", "source_reused": reused, "result": result})
        lines = [f"✅ Processing complete\nJob ID: {request_id}\nCandidates: {len(clips)}\nSource reused: {'Yes' if reused else 'No'}"]
        for index, clip in enumerate(clips[:20], 1):
            lines.append(f"{index}. {clip.get('category', 'clip')} — {clip.get('clip_url', '')}")
        send(chat_id, "\n".join(lines))
    except Exception as exc:
        _save(request_id, "error", {**state, "stage": "error", "error_type": type(exc).__name__, "error": str(exc), "retryable": True})
        send(chat_id, f"❌ Processing failed\nJob ID: {request_id}\n{str(exc)[:1500]}\n\nSend /retry {request_id} to try again.")


@router.post("/api/telegram/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks, x_telegram_bot_api_secret_token: str | None = Header(None)):
    expected = os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()
    if not expected or x_telegram_bot_api_secret_token != expected:
        raise HTTPException(status_code=401, detail="Invalid Telegram webhook secret")
    update = await request.json()
    message = update.get("message") or update.get("edited_message") or {}
    chat_id = str((message.get("chat") or {}).get("id", ""))
    user_id = str((message.get("from") or {}).get("id", ""))
    text = str(message.get("text") or message.get("caption") or "").strip()
    if not chat_id or not user_id:
        return {"status": "ignored"}
    if not _authorized(chat_id, user_id):
        send(chat_id, "This chat is not authorized to start Ripped Shorts jobs.")
        return {"status": "unauthorized"}
    retry_match = re.fullmatch(r"/retry\s+([A-Za-z0-9-]+)", text, re.I)
    if retry_match:
        request_id = retry_match.group(1)
        background_tasks.add_task(_process, request_id)
        return {"status": "retry_accepted", "request_id": request_id}
    try:
        parsed = parse_request(text)
    except ValueError as exc:
        send(chat_id, f"Send a YouTube or Google Drive video link.\n\nExamples:\nFind clips: https://youtu.be/...\nFind shorts only: https://drive.google.com/file/d/...\n\n{exc}")
        return {"status": "help"}
    request_id, update_id, stamp = str(uuid.uuid4()), str(update.get("update_id", "")), now()
    state = {"stage": "accepted", "parsed": parsed, "message_id": message.get("message_id")}
    try:
        with _LOCK, _telegram_db() as db:
            db.execute("INSERT INTO telegram_requests VALUES (?,?,?,?,?,?,?,?,?,?,?)", (request_id, update_id, chat_id, user_id, "accepted", parsed["mode"], parsed["source_kind"], parsed["source_value"], json.dumps(state), stamp, stamp))
    except sqlite3.IntegrityError:
        return {"status": "duplicate_update"}
    send(chat_id, f"✅ Processing request accepted\nSource: {parsed['source_kind'].title()}\nRequest: {parsed['mode'].title()}\nJob ID: {request_id}")
    background_tasks.add_task(_process, request_id)
    return {"status": "accepted", "request_id": request_id}
