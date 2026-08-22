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
RIPPED_SHORTS_RENDER_WORKERS = max(
    1, int(os.getenv("RIPPED_SHORTS_RENDER_WORKERS", "3"))
)
RENDER_EXECUTOR = ThreadPoolExecutor(max_workers=RIPPED_SHORTS_RENDER_WORKERS)
RIPPED_LOG_SHEET_ID = (
    os.getenv("RIPPED_SHORTS_LOG_SHEET_ID")
    or os.getenv("PODCAST_SHEET_ID")
    or "14VruBxjaaE9DyPSdBidMeuPHew3nHUC5sRNGXtLCsis"
).strip()
RIPPED_LOG_SHEET_TAB = os.getenv("RIPPED_SHORTS_LOG_SHEET_TAB", "Ripped Shorts").strip()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _render_progress_text(request_id: str) -> str:
    with _LOCK, _telegram_db() as db:
        row = db.execute(
            "SELECT state_json FROM telegram_requests WHERE request_id=?", (request_id,)
        ).fetchone()
    if not row:
        return "Render progress unavailable."
    state = json.loads(row["state_json"])
    reviews = dict(state.get("candidate_reviews") or {})
    statuses = [
        str(review.get("status") or "")
        for review in reviews.values()
        if str(review.get("status") or "")
        in {"queued", "rendering", "rendered", "render_failed"}
    ]
    total = len(statuses)
    rendered = sum(status == "rendered" for status in statuses)
    failed = sum(status == "render_failed" for status in statuses)
    rendering = sum(status == "rendering" for status in statuses)
    queued = sum(status == "queued" for status in statuses)
    processed = rendered + failed
    percent = round((processed / total) * 100) if total else 0
    return (
        f"Progress: {processed}/{total} processed ({percent}%)"
        f" | {rendering} rendering | {queued} queued"
        f" | {rendered} rendered | {failed} failed"
    )


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
    if any(word in lowered for word in ("topic", "segment", "16:9", "horizontal")) and not any(
        word in lowered for word in ("short", "both")
    ):
        mode: Literal["shorts", "topics", "both"] = "topics"
    elif "short" in lowered and not any(
        word in lowered for word in ("topic", "segment", "16:9", "horizontal", "both")
    ):
        mode = "shorts"
    else:
        # A plain link starts both editorial lanes.
        mode = "both"
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


def _sentence_complete_candidate(
    candidate: dict, transcript_segments: list[dict], tolerance: float = 0.75
) -> dict:
    """Expand a Short to transcript sentence boundaries when the 90s limit permits."""
    ordered = []
    for item in transcript_segments:
        seg_start = float(item.get("start", 0))
        seg_end = float(item.get("end", seg_start + float(item.get("duration", 0))))
        ordered.append((seg_start, seg_end, str(item.get("text", "")).strip()))
    if not ordered:
        return candidate
    start = float(candidate.get("start", -1))
    end = float(candidate.get("end", start + float(candidate.get("duration", 0))))
    start_index = min(range(len(ordered)), key=lambda i: abs(ordered[i][0] - start))
    end_index = min(range(len(ordered)), key=lambda i: abs(ordered[i][1] - end))
    sentence_end = r"[.!?][\"’']?$"

    # If the previous transcript line did not finish a sentence, this candidate
    # began mid-thought. Walk backward to the prior completed sentence.
    while (
        start_index > 0
        and not re.search(sentence_end, ordered[start_index - 1][2])
        and ordered[end_index][1] - ordered[start_index - 1][0] <= 90
    ):
        start_index -= 1

    # Finish the current sentence, but never grow beyond the Shorts maximum.
    while (
        end_index + 1 < len(ordered)
        and not re.search(sentence_end, ordered[end_index][2])
        and ordered[end_index + 1][1] - ordered[start_index][0] <= 90
    ):
        end_index += 1

    selected = ordered[start_index : end_index + 1]
    completed_start, completed_end = selected[0][0], selected[-1][1]
    return {
        **candidate,
        "start": completed_start,
        "end": completed_end,
        "duration": round(completed_end - completed_start, 3),
        "transcript": " ".join(text for _, _, text in selected),
    }


def validate_complete_candidates(
    payload: dict, transcript_segments: list[dict], tolerance: float = 0.75
) -> dict:
    """Reject invented text and enforce complete sentence/thought boundaries."""
    boundaries = []
    for item in transcript_segments:
        start = float(item.get("start", 0))
        end = float(item.get("end", start + float(item.get("duration", 0))))
        boundaries.append((start, end, str(item.get("text", "")).strip()))
    valid, rejected = [], []
    sentence_end = r"[.!?][\"’']?$"
    for original in payload.get("segments", []):
        candidate = _sentence_complete_candidate(
            dict(original), transcript_segments, tolerance
        )
        start = float(candidate.get("start", -1))
        end = float(candidate.get("end", start + float(candidate.get("duration", 0))))
        begins_cleanly = any(
            abs(start - seg_start) <= tolerance for seg_start, _, _ in boundaries
        )
        ends_cleanly = any(
            abs(end - seg_end) <= tolerance for _, seg_end, _ in boundaries
        )
        included = [
            text
            for seg_start, seg_end, text in boundaries
            if seg_start >= start - tolerance and seg_end <= end + tolerance
        ]
        expected = " ".join(included).casefold()
        quoted = str(candidate.get("transcript", "")).strip().casefold()
        complete_end = bool(
            re.search(sentence_end, str(candidate.get("transcript", "")).strip())
        )
        if (
            begins_cleanly
            and ends_cleanly
            and quoted
            and expected
            and (quoted in expected or expected in quoted)
            and end > start
            and end - start <= 90
            and complete_end
        ):
            candidate.update(
                {"start": start, "end": end, "duration": round(end - start, 3)}
            )
            valid.append(candidate)
        else:
            rejected.append(
                {
                    "start": start,
                    "end": end,
                    "reason": "Unsupported text or incomplete sentence/thought boundary",
                }
            )
    return {
        **payload,
        "segments": valid[:20],
        "validation_rejections": rejected,
    }


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
    log_request_id = candidate.get("_sheet_request_id") or request_id
    candidate_number = int(candidate.get("candidate_number") or index + 1)
    analysis = state["result"].get("analysis") or {}
    parsed = state.get("parsed") or {}
    reviewed = (state.get("candidate_reviews") or {}).get(str(index), {})
    row_values = [
        now(),
        log_request_id,
        parsed.get("video_id", ""),
        parsed.get("source_value", ""),
        analysis.get("content_type", ""),
        analysis.get("main_theme", ""),
        ", ".join(str(value) for value in analysis.get("key_ideas", [])),
        ", ".join(str(value) for value in analysis.get("keywords", [])),
        candidate_number,
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
            and str(values[1]) == str(log_request_id)
            and str(values[8]) == str(candidate_number)
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


def _approved_clip_history_from_sheet(video_id: str) -> dict[str, list[dict[str, Any]]]:
    """Return approved rendered and unfinished clips for an exact YouTube ID."""
    import main

    _, _, sheets = main.get_google_services()
    response = sheets.spreadsheets().values().get(
        spreadsheetId=RIPPED_LOG_SHEET_ID,
        range=f"'{RIPPED_LOG_SHEET_TAB}'!A:U",
    ).execute()
    rows = response.get("values", [])
    recovered: dict[tuple[float, float], dict[str, Any]] = {}
    for values in rows[1:]:
        padded = list(values) + [""] * (21 - len(values))
        if str(padded[2]).strip() != video_id:
            continue
        if str(padded[15]).strip().lower() != "approved":
            continue
        render_status = str(padded[19]).strip().lower()
        clip_url = str(padded[18]).strip()
        is_rendered = render_status == "rendered" and bool(clip_url)
        try:
            start = float(padded[9])
            end = float(padded[10])
            duration = float(padded[11]) if str(padded[11]).strip() else end - start
            candidate_number = int(float(padded[8]))
        except (TypeError, ValueError):
            logger.warning("Skipping malformed approved Ripped Shorts row for %s", video_id)
            continue
        recovered[(start, end)] = {
            "start": start,
            "end": end,
            "duration": duration,
            "category": padded[12],
            "transcript": padded[13],
            "reason": padded[14],
            "candidate_number": candidate_number,
            "_sheet_request_id": str(padded[1]).strip(),
            "_reviewed_at": str(padded[16]).strip(),
            "_reviewer_user_id": str(padded[20]).strip(),
            "_render_status": render_status,
            "_clip_url": clip_url,
            "_is_rendered": is_rendered,
        }
    ordered = sorted(recovered.values(), key=lambda item: item["candidate_number"])
    return {
        "rendered": [item for item in ordered if item["_is_rendered"]],
        "unfinished": [item for item in ordered if not item["_is_rendered"]],
    }


def _reviewed_short_history_from_sheet(video_id: str) -> list[dict[str, Any]]:
    """Return distinct reviewed shorts, including approvals and rejections."""
    import main

    _, _, sheets = main.get_google_services()
    response = sheets.spreadsheets().values().get(
        spreadsheetId=RIPPED_LOG_SHEET_ID,
        range=f"'{RIPPED_LOG_SHEET_TAB}'!A:U",
    ).execute()
    rows = response.get("values", [])
    found: dict[tuple[float, float], dict[str, Any]] = {}
    for values in rows[1:]:
        padded = list(values) + [""] * (21 - len(values))
        if str(padded[2]).strip() != video_id:
            continue
        try:
            start, end = float(padded[9]), float(padded[10])
            number = int(float(padded[8]))
        except (TypeError, ValueError):
            continue
        found[(start, end)] = {
            "start": start,
            "end": end,
            "transcript": str(padded[13]).strip(),
            "decision": str(padded[15]).strip().lower(),
            "candidate_number": number,
        }
    return sorted(found.values(), key=lambda item: item["candidate_number"])


def _short_learning_prompt(video_id: str) -> str:
    """Build balanced approval/rejection examples from the durable Shorts ledger."""
    import main

    _, _, sheets = main.get_google_services()
    response = sheets.spreadsheets().values().get(
        spreadsheetId=RIPPED_LOG_SHEET_ID,
        range=f"'{RIPPED_LOG_SHEET_TAB}'!A:U",
    ).execute()
    rows = response.get("values", [])[1:]
    same_video_types = {
        str((list(values) + [""] * 21)[4]).strip().lower()
        for values in rows
        if str((list(values) + [""] * 21)[2]).strip() == video_id
        and str((list(values) + [""] * 21)[4]).strip()
    }
    examples = []
    for recency, values in enumerate(rows):
        padded = list(values) + [""] * (21 - len(values))
        decision = str(padded[15]).strip().lower()
        if decision not in {"approved", "rejected"}:
            continue
        transcript = str(padded[13]).strip()
        if not transcript:
            continue
        content_type = str(padded[4]).strip().lower()
        priority = (
            2 if str(padded[2]).strip() == video_id else
            1 if content_type and content_type in same_video_types else
            0
        )
        examples.append({
            "decision": decision,
            "priority": priority,
            "recency": recency,
            "content_type": content_type or "unknown",
            "category": str(padded[12]).strip() or "unknown",
            "transcript": transcript[:700],
            "reason": str(padded[14]).strip()[:300],
        })

    examples.sort(
        key=lambda item: (item["priority"], item["recency"]), reverse=True
    )
    approved = [item for item in examples if item["decision"] == "approved"][:10]
    rejected = [item for item in examples if item["decision"] == "rejected"][:10]

    def format_examples(label: str, selected: list[dict[str, Any]]) -> str:
        if not selected:
            return f"{label}: none recorded yet"
        lines = [label + ":"]
        for item in selected:
            lines.append(
                f"- type={item['content_type']} category={item['category']} | "
                f"{item['transcript']}"
                + (f" | prior reason={item['reason']}" if item["reason"] else "")
            )
        return "\n".join(lines)

    return (
        "LEARNING FROM DEREK'S SHORT REVIEWS\n"
        "Treat approvals as positive structural examples and rejections as negative "
        "examples. Learn the qualities, not their exact wording or timestamps. "
        "Reject scripture-only readings without interpretation/application, generic "
        "blurbs without a payoff, incomplete setups, and context-dependent fragments. "
        "Require a standalone hook, developed point, and payoff.\n"
        + format_examples("APPROVED SHORTS", approved)
        + "\n"
        + format_examples("REJECTED SHORTS", rejected)
    )


def _safe_log_candidate(*args, **kwargs) -> None:
    try:
        _log_candidate_decision(*args, **kwargs)
    except Exception as exc:
        logger.exception("Ripped Shorts decision log failed: %s", exc)


def _send_candidates(
    chat_id: str, request_id: str, result: dict, *, start_index: int = 0
) -> None:
    clips = result.get("segments", [])
    review_count = max(0, len(clips) - start_index)
    send(
        chat_id,
        f"✅ Analysis complete\nJob ID: {request_id}\n"
        f"New shorts for review: {review_count}\n\n"
        "Approve only the shorts you want created.",
    )
    for zero_index in range(start_index, len(clips)):
        clip = clips[zero_index]
        short_number = int(clip.get("candidate_number") or zero_index + 1)
        transcript = str(clip.get("transcript", "")).strip()
        if len(transcript) > 2600:
            transcript = transcript[:2597] + "..."
        text = (
            f"Short {short_number}\n\n"
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
                {"text": "✅ Approve & Render", "callback_data": f"rs:approve:{request_id}:{zero_index}"},
                {"text": "❌ Reject", "callback_data": f"rs:reject:{request_id}:{zero_index}"},
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


def _topic_break_suggestions(transcript_segments: list[dict]) -> list[dict]:
    """Ask OpenAI for semantic chapter boundaries; the local planner enforces coverage."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return []
    lines = []
    for item in transcript_segments:
        start = float(item.get("start", 0))
        end = float(item.get("end", start + float(item.get("duration", 0))))
        text = str(item.get("text", "")).replace("\n", " ").strip()
        lines.append(f"[{start:.2f}-{end:.2f}] {text}")
    prompt = (
        "Divide this full transcript into coherent horizontal-video sections for "
        "YouTube and Facebook. Identify changes in point, subject, story, or tangent. "
        "Aim near 8 minutes per section. Every boundary must be between complete "
        "sentences or thoughts. Do not omit any part of the eligible timeline and do "
        "not overlap sections. Return strict JSON only as "
        '{"segments":[{"start":0,"end":480,"title":"...","summary":"..."}]}. '
        "Use only exact transcript timestamps.\n\n"
        + "\n".join(lines)
    )
    payload = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "messages": [
            {"role": "system", "content": "Return strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    timeout = max(180, int(os.getenv("OPENAI_TOPIC_TIMEOUT_SECONDS", "600")))
    last_error = None
    for attempt in range(1, 3):
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=(15, timeout),
            )
            if response.status_code == 200:
                data = response.json()["choices"][0]["message"]["content"].strip()
                if data.startswith("```"):
                    data = "\n".join(data.splitlines()[1:-1]).strip()
                parsed = json.loads(data)
                return parsed.get("segments", []) if isinstance(parsed, dict) else []
            last_error = f"HTTP {response.status_code}: {response.text[:500]}"
        except Exception as exc:
            last_error = str(exc)
        if attempt < 2:
            import time
            time.sleep(3)
    logger.warning("Semantic 16:9 boundary selection failed; using transcript fallback: %s", last_error)
    return []


def _build_contiguous_topic_segments(
    transcript_segments: list[dict], suggestions: list[dict]
) -> list[dict]:
    """Create contiguous, non-overlapping >3 minute sections targeting eight minutes."""
    ordered = sorted(transcript_segments, key=lambda item: float(item.get("start", 0)))
    if not ordered:
        return []
    timeline_start = float(ordered[0].get("start", 0))
    timeline_end = float(
        ordered[-1].get(
            "end",
            float(ordered[-1].get("start", 0))
            + float(ordered[-1].get("duration", 0)),
        )
    )
    total = timeline_end - timeline_start
    minimum = max(181.0, float(os.getenv("TOPIC_SEGMENT_MIN_SECONDS", "181")))
    target = max(minimum, float(os.getenv("TOPIC_SEGMENT_TARGET_SECONDS", "480")))
    maximum = max(target, float(os.getenv("TOPIC_SEGMENT_MAX_SECONDS", "720")))
    if total < minimum:
        return []

    segment_count = max(1, round(total / target))
    while segment_count > 1 and total / segment_count < minimum:
        segment_count -= 1
    while total / segment_count > maximum:
        segment_count += 1

    transcript_ends = [
        float(
            item.get(
                "end",
                float(item.get("start", 0)) + float(item.get("duration", 0)),
            )
        )
        for item in ordered[:-1]
    ]
    suggested_ends = []
    for item in suggestions:
        try:
            value = float(item.get("end"))
        except (TypeError, ValueError):
            continue
        if timeline_start < value < timeline_end:
            suggested_ends.append(value)

    boundaries = [timeline_start]
    for index in range(1, segment_count):
        desired = timeline_start + total * index / segment_count
        low = boundaries[-1] + minimum
        high = timeline_end - minimum * (segment_count - index)
        semantic = [value for value in suggested_ends if low <= value <= high]
        if semantic:
            boundary = min(semantic, key=lambda value: abs(value - desired))
        else:
            valid_ends = [value for value in transcript_ends if low <= value <= high]
            if not valid_ends:
                boundary = desired
            else:
                boundary = min(valid_ends, key=lambda value: abs(value - desired))
        boundaries.append(boundary)
    boundaries.append(timeline_end)

    results = []
    for index in range(len(boundaries) - 1):
        start, end = boundaries[index], boundaries[index + 1]
        included = []
        for item in ordered:
            seg_start = float(item.get("start", 0))
            seg_end = float(
                item.get("end", seg_start + float(item.get("duration", 0)))
            )
            if seg_start >= start - 0.75 and seg_end <= end + 0.75:
                included.append(str(item.get("text", "")).strip())
        midpoint = (start + end) / 2
        matching = []
        for item in suggestions:
            try:
                item_start = float(item.get("start", start))
                item_end = float(item.get("end", end))
            except (TypeError, ValueError):
                continue
            if item_start <= midpoint <= item_end:
                matching.append(item)
        metadata = matching[0] if matching else {}
        results.append(
            {
                "segment_number": index + 1,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "title": str(metadata.get("title") or f"Part {index + 1}").strip(),
                "summary": str(metadata.get("summary") or "").strip(),
                "transcript": " ".join(included),
                "aspect_ratio": "16:9",
            }
        )
    return results


def _send_topic_candidates(
    chat_id: str, request_id: str, topic_result: dict
) -> None:
    topics = topic_result.get("segments", [])
    send(
        chat_id,
        f"📺 16:9 section analysis complete\nJob ID: {request_id}\n"
        f"Sections covering the eligible video: {len(topics)}\n\n"
        "These sections do not overlap. Approve the horizontal videos you want rendered.",
    )
    for index, segment in enumerate(topics):
        text = (
            f"16:9 Segment {index + 1}: {segment.get('title', '')}\n\n"
            f"Time: {_timecode(float(segment['start']))}–"
            f"{_timecode(float(segment['end']))}\n"
            f"Duration: {round(float(segment['duration']) / 60, 1)} minutes"
            + (
                f"\n\nSection summary:\n{segment.get('summary', '')}"
                if segment.get("summary")
                else ""
            )
        )
        telegram(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": text,
                "disable_web_page_preview": True,
                "reply_markup": {
                    "inline_keyboard": [[
                        {
                            "text": "✅ Approve 16:9",
                            "callback_data": f"rs:topic_approve:{request_id}:{index}",
                        },
                        {
                            "text": "❌ Skip",
                            "callback_data": f"rs:topic_reject:{request_id}:{index}",
                        },
                    ]]
                },
            },
        )


def _process_topics(
    request_id: str,
    state: dict[str, Any],
    chat_id: str,
    video_id: str,
    video: Path,
    segments: list[dict],
    reused: bool,
) -> dict[str, Any]:
    send(
        chat_id,
        "📺 Mapping the full eligible video into non-overlapping 16:9 sections "
        "targeting about 8 minutes each.",
    )
    suggestions = _topic_break_suggestions(segments)
    topics = _build_contiguous_topic_segments(segments, suggestions)
    if not topics:
        send(
            chat_id,
            "ℹ️ The eligible video is not longer than three minutes, so no 16:9 "
            "topic segment was created.",
        )
    topic_result = {"segments": topics, "coverage": "full_eligible_timeline"}
    state.update(
        {
            "video_path": str(video),
            "source_reused": reused,
            "topic_result": topic_result,
            "topic_reviews": state.get("topic_reviews") or {},
            "topic_stage": "awaiting_review" if topics else "not_eligible",
        }
    )
    _save(request_id, "awaiting_review" if topics else "processing", state)
    if topics:
        _send_topic_candidates(chat_id, request_id, topic_result)
    return state


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

        if row["mode"] in {"topics", "both"}:
            state = _process_topics(
                request_id,
                state,
                chat_id,
                video_id,
                video,
                segments,
                reused,
            )
            if row["mode"] == "topics":
                return

        force_rerip = bool(state.get("force_rerip"))
        reuse_existing = bool(state.get("reuse_existing"))
        approved_clips: list[dict[str, Any]] = []
        rendered_clips: list[dict[str, Any]] = []
        prior_shorts: list[dict[str, Any]] = []
        if row["source_kind"] == "youtube":
            approval_history = _approved_clip_history_from_sheet(video_id)
            approved_clips = approval_history["unfinished"]
            rendered_clips = approval_history["rendered"]
            prior_shorts = _reviewed_short_history_from_sheet(video_id)
            if force_rerip:
                # A re-rip is a fresh editorial pass; old approvals are learning
                # examples, not automatic render instructions.
                approved_clips = []

            # Ten reviewed shorts is enough history to treat this ID as already
            # discovered. Below ten, search again for additional distinct shorts.
            if reuse_existing or (len(prior_shorts) >= 10 and not force_rerip):
                if approved_clips:
                    reviews = {
                        str(index): {
                            "status": "queued",
                            "reviewed_at": clip.get("_reviewed_at") or now(),
                            "user_id": clip.get("_reviewer_user_id", ""),
                            "recovered_from_sheet": True,
                        }
                        for index, clip in enumerate(approved_clips)
                    }
                    recovered_result = {
                        "analysis": {
                            "content_type": "",
                            "main_theme": "",
                            "key_ideas": [],
                            "keywords": [],
                        },
                        "segments": approved_clips,
                    }
                    recovered_state = {
                        **state,
                        "stage": "awaiting_review",
                        "video_path": str(video),
                        "source_reused": reused,
                        "result": recovered_result,
                        "candidate_reviews": reviews,
                        "recovered_approvals_from_sheet": True,
                    }
                    _save(request_id, "awaiting_review", recovered_state)
                    send(
                        chat_id,
                        f"♻️ Recovered {len(approved_clips)} approved unfinished short(s) "
                        f"for YouTube ID {video_id}. Rendering only those shorts now.",
                    )
                    for index in range(len(approved_clips)):
                        RENDER_EXECUTOR.submit(_render_approved, request_id, index, chat_id)
                    return
                if rendered_clips:
                    existing_links = [
                        str(clip.get("_clip_url") or "")
                        for clip in rendered_clips
                        if clip.get("_clip_url")
                    ]
                    _save(
                        request_id,
                        "already_rendered",
                        {
                            **state,
                            "stage": "already_rendered",
                            "video_path": str(video),
                            "source_reused": reused,
                            "existing_clip_links": existing_links,
                        },
                    )
                    links_text = "\n".join(existing_links[:20])
                    send(
                        chat_id,
                        f"✅ YouTube ID {video_id} already has "
                        f"{len(prior_shorts)} reviewed shorts in the sheet, so GPT "
                        "selection was not run again."
                        + (f"\n\n{links_text}" if links_text else ""),
                    )
                    return

        prior_count = len(prior_shorts)
        needed = 20 if force_rerip else max(1, 20 - prior_count)
        _save(
            request_id,
            "selecting",
            {
                **state,
                "stage": "selection",
                "video_path": str(video),
                "source_reused": reused,
            },
        )
        if force_rerip:
            send(
                chat_id,
                f"🧠 Re-ripping YouTube ID {video_id}. Selecting a fresh set of up "
                "to 20 Shorts using your approval and rejection history.",
            )
        elif prior_count:
            send(
                chat_id,
                f"🧠 Only {prior_count} previously reviewed short(s) were found for "
                f"{video_id}. Searching again for up to {needed} additional distinct shorts.",
            )
        else:
            send(
                chat_id,
                "🧠 Reviewing the full eligible transcript and targeting approximately "
                "20 distinct complete shorts.",
            )
        import main
        enriched = [
            {**item, "video_id": state["parsed"].get("video_id", "drive-source")}
            for item in segments
        ]
        learning_prompt = _short_learning_prompt(
            video_id if row["source_kind"] == "youtube" else ""
        )
        prior_summary = "\n".join(
            f"- {item['start']:.2f}-{item['end']:.2f}: {item.get('transcript', '')}"
            for item in prior_shorts
        )
        if force_rerip:
            supplemental = (
                "Perform a fresh re-rip and select up to 20 of the strongest Shorts. "
                "Previously approved Shorts may be selected again if they remain among "
                "the best. Never select a previously rejected complete Short again.\n"
                + learning_prompt
                + (f"\nSAME-VIDEO REVIEW HISTORY:\n{prior_summary[:12000]}" if prior_summary else "")
            )
        elif prior_shorts:
            supplemental = (
                f"Find up to {needed} ADDITIONAL distinct Shorts. Do not repeat any "
                "previous complete thought, lesson, payoff, or transcript. Time overlap "
                "is allowed only when the new Short is materially different.\n"
                + learning_prompt
                + f"\nPREVIOUSLY REVIEWED SHORTS:\n{prior_summary[:12000]}"
            )
        else:
            supplemental = learning_prompt
        result = main.call_openai_for_clips(enriched, supplemental)
        result = validate_complete_candidates(result, enriched)
        excluded_history = (
            [item for item in prior_shorts if item.get("decision") == "rejected"]
            if force_rerip
            else prior_shorts
        )
        prior_texts = {
            " ".join(str(item.get("transcript", "")).lower().split())
            for item in excluded_history
            if item.get("transcript")
        }
        new_segments = [
            clip
            for clip in result.get("segments", [])
            if " ".join(str(clip.get("transcript", "")).lower().split())
            not in prior_texts
        ]
        new_segments = select_non_overlapping(
            new_segments, limit=needed, allow_overlap=True
        )
        next_number = max(
            [int(item.get("candidate_number") or 0) for item in prior_shorts] or [0]
        ) + 1
        for offset, clip in enumerate(new_segments):
            clip["candidate_number"] = next_number + offset

        combined = approved_clips + new_segments
        result["segments"] = combined
        reviews = {
            str(index): {
                "status": "queued",
                "reviewed_at": clip.get("_reviewed_at") or now(),
                "user_id": clip.get("_reviewer_user_id", ""),
                "recovered_from_sheet": True,
            }
            for index, clip in enumerate(approved_clips)
        }
        final_state = {
            **state,
            "stage": "awaiting_review",
            "video_path": str(video),
            "source_reused": reused,
            "result": result,
            "candidate_reviews": reviews,
        }
        _save(request_id, "awaiting_review", final_state)
        if approved_clips:
            send(
                chat_id,
                f"♻️ Resuming {len(approved_clips)} previously approved unfinished "
                "short(s) while presenting additional shorts for review.",
            )
            for index in range(len(approved_clips)):
                RENDER_EXECUTOR.submit(_render_approved, request_id, index, chat_id)
        _send_candidates(
            chat_id, request_id, result, start_index=len(approved_clips)
        )
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
    rerip_choice = re.fullmatch(
        r"rs:(rerip|reuse):([A-Za-z0-9-]+)", callback_data
    )
    if rerip_choice:
        choice, request_id = rerip_choice.groups()
        with _LOCK, _telegram_db() as db:
            row = db.execute(
                "SELECT * FROM telegram_requests WHERE request_id=?", (request_id,)
            ).fetchone()
            if not row:
                return {"status": "not_found"}
            state = json.loads(row["state_json"])
            state["force_rerip"] = choice == "rerip"
            state["reuse_existing"] = choice == "reuse"
            state["stage"] = "accepted"
            db.execute(
                "UPDATE telegram_requests SET status=?, state_json=?, updated_at=? "
                "WHERE request_id=?",
                ("accepted", json.dumps(state), now(), request_id),
            )
        send(
            chat_id,
            (
                "♻️ Re-ripping this video's Shorts with your approval/rejection "
                "learning."
                if choice == "rerip"
                else "▶️ Reusing the existing reviewed Shorts and approved renders."
            ),
        )
        background_tasks.add_task(_process, request_id)
        return {"status": choice, "request_id": request_id}

    topic_action = re.fullmatch(
        r"rs:topic_(approve|reject):([A-Za-z0-9-]+):(\d+)", callback_data
    )
    if topic_action:
        verb, request_id, index_text = topic_action.groups()
        index = int(index_text)
        with _LOCK, _telegram_db() as db:
            row = db.execute(
                "SELECT * FROM telegram_requests WHERE request_id=?", (request_id,)
            ).fetchone()
            if not row:
                return {"status": "not_found"}
            state = json.loads(row["state_json"])
            topics = (state.get("topic_result") or {}).get("segments", [])
            if index >= len(topics):
                return {"status": "topic_not_found"}
            reviews = dict(state.get("topic_reviews") or {})
            existing = (reviews.get(str(index)) or {}).get("status")
            if verb == "approve" and existing in {"queued", "rendering", "rendered"}:
                return {
                    "status": f"already_{existing}",
                    "request_id": request_id,
                    "topic_index": index,
                }
            reviews[str(index)] = {
                "status": "queued" if verb == "approve" else "rejected",
                "reviewed_at": now(),
                "user_id": user_id,
            }
            state["topic_reviews"] = reviews
            db.execute(
                "UPDATE telegram_requests SET state_json=?, updated_at=? WHERE request_id=?",
                (json.dumps(state), now(), request_id),
            )
        if verb == "approve":
            RENDER_EXECUTOR.submit(_render_topic_approved, request_id, index, chat_id)
            send(
                chat_id,
                f"16:9 Segment {index + 1} approved and queued. "
                f"Up to {RIPPED_SHORTS_RENDER_WORKERS} total videos render at once.",
            )
        else:
            send(chat_id, f"16:9 Segment {index + 1} skipped.")
        return {
            "status": f"topic_{verb}",
            "request_id": request_id,
            "topic_index": index,
        }

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
                    "short_index": index,
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
                f"Short {index + 1} approved and queued for rendering. "
                f"Up to {RIPPED_SHORTS_RENDER_WORKERS} clips render at once; the rest wait.\n"
                f"{_render_progress_text(request_id)}",
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
            send(chat_id, f"Short {index + 1} rejected.")
        return {"status": verb, "request_id": request_id, "short_index": index}
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
    if parsed["source_kind"] == "youtube":
        try:
            prior_shorts = _reviewed_short_history_from_sheet(parsed["video_id"])
        except Exception:
            logger.exception("Could not check prior Shorts history before intake")
            prior_shorts = []
        if prior_shorts:
            state["stage"] = "awaiting_rerip_choice"
            state["prior_reviewed_short_count"] = len(prior_shorts)
            _save(request_id, "awaiting_rerip_choice", state)
            telegram(
                "sendMessage",
                {
                    "chat_id": chat_id,
                    "text": (
                        f"♻️ YouTube ID {parsed['video_id']} already has "
                        f"{len(prior_shorts)} reviewed Short(s).\n\n"
                        "Do you want to re-rip the Shorts using everything learned "
                        "from your approvals and rejections?"
                    ),
                    "reply_markup": {
                        "inline_keyboard": [[
                            {
                                "text": "♻️ Re-rip Shorts",
                                "callback_data": f"rs:rerip:{request_id}",
                            },
                            {
                                "text": "▶️ Use Existing",
                                "callback_data": f"rs:reuse:{request_id}",
                            },
                        ]]
                    },
                },
            )
            return {
                "status": "awaiting_rerip_choice",
                "request_id": request_id,
                "reviewed_shorts": len(prior_shorts),
            }
    send(
        chat_id,
        f"✅ Ripped Shorts request accepted\n"
        f"Source: {parsed['source_kind'].title()}\nJob ID: {request_id}",
    )
    background_tasks.add_task(_process, request_id)
    return {"status": "accepted", "request_id": request_id}


def _render_topic_approved(request_id: str, index: int, chat_id: str) -> None:
    try:
        with _LOCK, _telegram_db() as db:
            row = db.execute(
                "SELECT * FROM telegram_requests WHERE request_id=?", (request_id,)
            ).fetchone()
            if not row:
                raise RuntimeError(f"Ripped Shorts request not found: {request_id}")
            state = json.loads(row["state_json"])
            reviews = dict(state.get("topic_reviews") or {})
            reviews[str(index)] = {
                **reviews.get(str(index), {}),
                "status": "rendering",
                "render_started_at": now(),
            }
            state["topic_reviews"] = reviews
            db.execute(
                "UPDATE telegram_requests SET state_json=?, updated_at=? WHERE request_id=?",
                (json.dumps(state), now(), request_id),
            )
        send(chat_id, f"🎬 16:9 Segment {index + 1} is now rendering.")
        segment = state["topic_result"]["segments"][index]
        video = Path(state["video_path"])
        video_id = state["parsed"].get("video_id", request_id)
        import main

        rendered = main.attach_topic_segment_asset(
            dict(segment), video_id, video, index + 1
        )
        with _LOCK, _telegram_db() as db:
            latest = db.execute(
                "SELECT * FROM telegram_requests WHERE request_id=?", (request_id,)
            ).fetchone()
            latest_state = json.loads(latest["state_json"])
            reviews = dict(latest_state.get("topic_reviews") or {})
            reviews[str(index)] = {
                **reviews.get(str(index), {}),
                "status": "rendered",
                "segment_url": rendered.get("segment_url"),
                "rendered_at": now(),
            }
            latest_state["topic_reviews"] = reviews
            db.execute(
                "UPDATE telegram_requests SET state_json=?, updated_at=? WHERE request_id=?",
                (json.dumps(latest_state), now(), request_id),
            )
        send(
            chat_id,
            f"✅ 16:9 Segment {index + 1} rendered and uploaded to DRIVE_FOLDER_ID:\n"
            f"{rendered.get('segment_url', '')}",
        )
    except Exception as exc:
        logger.exception(
            "16:9 render failed request_id=%s segment=%s", request_id, index + 1
        )
        try:
            with _LOCK, _telegram_db() as db:
                latest = db.execute(
                    "SELECT * FROM telegram_requests WHERE request_id=?", (request_id,)
                ).fetchone()
                latest_state = json.loads(latest["state_json"])
                reviews = dict(latest_state.get("topic_reviews") or {})
                reviews[str(index)] = {
                    **reviews.get(str(index), {}),
                    "status": "render_failed",
                    "render_error": str(exc),
                    "rendered_at": now(),
                }
                latest_state["topic_reviews"] = reviews
                db.execute(
                    "UPDATE telegram_requests SET state_json=?, updated_at=? WHERE request_id=?",
                    (json.dumps(latest_state), now(), request_id),
                )
        except Exception:
            logger.exception("Could not persist 16:9 render failure")
        send(
            chat_id,
            f"❌ 16:9 Segment {index + 1} render failed:\n{str(exc)[:1500]}",
        )


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
        send(
            chat_id,
            f"🎬 Short {index + 1} is now rendering.\n"
            f"{_render_progress_text(request_id)}",
        )
        candidate = state["result"]["segments"][index]
        user_id = str(
            (state.get("candidate_reviews") or {}).get(str(index), {}).get("user_id", "")
        )
        video = Path(state["video_path"])
        import main

        payload_candidate = dict(candidate)
        payload_candidate["candidate_number"] = int(
            candidate.get("candidate_number") or index + 1
        )
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
            f"✅ Short {index + 1} rendered and uploaded to DRIVE_FOLDER_ID:\n"
            f"{clip.get('clip_url', '')}\n"
            f"{_render_progress_text(request_id)}",
        )
        _notify_render_queue_complete(request_id, chat_id)
    except Exception as exc:
        logger.exception(
            "Short render failed request_id=%s candidate=%s", request_id, index + 1
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
        send(
            chat_id,
            f"❌ Short {index + 1} render failed:\n{str(exc)[:1500]}\n"
            f"{_render_progress_text(request_id)}",
        )
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
, ordered[start_index - 1][2])
        and ordered[end_index][1] - ordered[start_index - 1][0] <= 90
    ):
        start_index -= 1

    # Finish the current sentence, but never grow beyond the Shorts maximum.
    while (
        end_index + 1 < len(ordered)
        and not re.search(r'[.!?]["’\']?

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
    log_request_id = candidate.get("_sheet_request_id") or request_id
    candidate_number = int(candidate.get("candidate_number") or index + 1)
    analysis = state["result"].get("analysis") or {}
    parsed = state.get("parsed") or {}
    reviewed = (state.get("candidate_reviews") or {}).get(str(index), {})
    row_values = [
        now(),
        log_request_id,
        parsed.get("video_id", ""),
        parsed.get("source_value", ""),
        analysis.get("content_type", ""),
        analysis.get("main_theme", ""),
        ", ".join(str(value) for value in analysis.get("key_ideas", [])),
        ", ".join(str(value) for value in analysis.get("keywords", [])),
        candidate_number,
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
            and str(values[1]) == str(log_request_id)
            and str(values[8]) == str(candidate_number)
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


def _approved_clip_history_from_sheet(video_id: str) -> dict[str, list[dict[str, Any]]]:
    """Return approved rendered and unfinished clips for an exact YouTube ID."""
    import main

    _, _, sheets = main.get_google_services()
    response = sheets.spreadsheets().values().get(
        spreadsheetId=RIPPED_LOG_SHEET_ID,
        range=f"'{RIPPED_LOG_SHEET_TAB}'!A:U",
    ).execute()
    rows = response.get("values", [])
    recovered: dict[tuple[float, float], dict[str, Any]] = {}
    for values in rows[1:]:
        padded = list(values) + [""] * (21 - len(values))
        if str(padded[2]).strip() != video_id:
            continue
        if str(padded[15]).strip().lower() != "approved":
            continue
        render_status = str(padded[19]).strip().lower()
        clip_url = str(padded[18]).strip()
        is_rendered = render_status == "rendered" and bool(clip_url)
        try:
            start = float(padded[9])
            end = float(padded[10])
            duration = float(padded[11]) if str(padded[11]).strip() else end - start
            candidate_number = int(float(padded[8]))
        except (TypeError, ValueError):
            logger.warning("Skipping malformed approved Ripped Shorts row for %s", video_id)
            continue
        recovered[(start, end)] = {
            "start": start,
            "end": end,
            "duration": duration,
            "category": padded[12],
            "transcript": padded[13],
            "reason": padded[14],
            "candidate_number": candidate_number,
            "_sheet_request_id": str(padded[1]).strip(),
            "_reviewed_at": str(padded[16]).strip(),
            "_reviewer_user_id": str(padded[20]).strip(),
            "_render_status": render_status,
            "_clip_url": clip_url,
            "_is_rendered": is_rendered,
        }
    ordered = sorted(recovered.values(), key=lambda item: item["candidate_number"])
    return {
        "rendered": [item for item in ordered if item["_is_rendered"]],
        "unfinished": [item for item in ordered if not item["_is_rendered"]],
    }


def _reviewed_short_history_from_sheet(video_id: str) -> list[dict[str, Any]]:
    """Return distinct reviewed shorts, including approvals and rejections."""
    import main

    _, _, sheets = main.get_google_services()
    response = sheets.spreadsheets().values().get(
        spreadsheetId=RIPPED_LOG_SHEET_ID,
        range=f"'{RIPPED_LOG_SHEET_TAB}'!A:U",
    ).execute()
    rows = response.get("values", [])
    found: dict[tuple[float, float], dict[str, Any]] = {}
    for values in rows[1:]:
        padded = list(values) + [""] * (21 - len(values))
        if str(padded[2]).strip() != video_id:
            continue
        try:
            start, end = float(padded[9]), float(padded[10])
            number = int(float(padded[8]))
        except (TypeError, ValueError):
            continue
        found[(start, end)] = {
            "start": start,
            "end": end,
            "transcript": str(padded[13]).strip(),
            "decision": str(padded[15]).strip().lower(),
            "candidate_number": number,
        }
    return sorted(found.values(), key=lambda item: item["candidate_number"])


def _short_learning_prompt(video_id: str) -> str:
    """Build balanced approval/rejection examples from the durable Shorts ledger."""
    import main

    _, _, sheets = main.get_google_services()
    response = sheets.spreadsheets().values().get(
        spreadsheetId=RIPPED_LOG_SHEET_ID,
        range=f"'{RIPPED_LOG_SHEET_TAB}'!A:U",
    ).execute()
    rows = response.get("values", [])[1:]
    same_video_types = {
        str((list(values) + [""] * 21)[4]).strip().lower()
        for values in rows
        if str((list(values) + [""] * 21)[2]).strip() == video_id
        and str((list(values) + [""] * 21)[4]).strip()
    }
    examples = []
    for recency, values in enumerate(rows):
        padded = list(values) + [""] * (21 - len(values))
        decision = str(padded[15]).strip().lower()
        if decision not in {"approved", "rejected"}:
            continue
        transcript = str(padded[13]).strip()
        if not transcript:
            continue
        content_type = str(padded[4]).strip().lower()
        priority = (
            2 if str(padded[2]).strip() == video_id else
            1 if content_type and content_type in same_video_types else
            0
        )
        examples.append({
            "decision": decision,
            "priority": priority,
            "recency": recency,
            "content_type": content_type or "unknown",
            "category": str(padded[12]).strip() or "unknown",
            "transcript": transcript[:700],
            "reason": str(padded[14]).strip()[:300],
        })

    examples.sort(
        key=lambda item: (item["priority"], item["recency"]), reverse=True
    )
    approved = [item for item in examples if item["decision"] == "approved"][:10]
    rejected = [item for item in examples if item["decision"] == "rejected"][:10]

    def format_examples(label: str, selected: list[dict[str, Any]]) -> str:
        if not selected:
            return f"{label}: none recorded yet"
        lines = [label + ":"]
        for item in selected:
            lines.append(
                f"- type={item['content_type']} category={item['category']} | "
                f"{item['transcript']}"
                + (f" | prior reason={item['reason']}" if item["reason"] else "")
            )
        return "\n".join(lines)

    return (
        "LEARNING FROM DEREK'S SHORT REVIEWS\n"
        "Treat approvals as positive structural examples and rejections as negative "
        "examples. Learn the qualities, not their exact wording or timestamps. "
        "Reject scripture-only readings without interpretation/application, generic "
        "blurbs without a payoff, incomplete setups, and context-dependent fragments. "
        "Require a standalone hook, developed point, and payoff.\n"
        + format_examples("APPROVED SHORTS", approved)
        + "\n"
        + format_examples("REJECTED SHORTS", rejected)
    )


def _safe_log_candidate(*args, **kwargs) -> None:
    try:
        _log_candidate_decision(*args, **kwargs)
    except Exception as exc:
        logger.exception("Ripped Shorts decision log failed: %s", exc)


def _send_candidates(
    chat_id: str, request_id: str, result: dict, *, start_index: int = 0
) -> None:
    clips = result.get("segments", [])
    review_count = max(0, len(clips) - start_index)
    send(
        chat_id,
        f"✅ Analysis complete\nJob ID: {request_id}\n"
        f"New shorts for review: {review_count}\n\n"
        "Approve only the shorts you want created.",
    )
    for zero_index in range(start_index, len(clips)):
        clip = clips[zero_index]
        short_number = int(clip.get("candidate_number") or zero_index + 1)
        transcript = str(clip.get("transcript", "")).strip()
        if len(transcript) > 2600:
            transcript = transcript[:2597] + "..."
        text = (
            f"Short {short_number}\n\n"
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
                {"text": "✅ Approve & Render", "callback_data": f"rs:approve:{request_id}:{zero_index}"},
                {"text": "❌ Reject", "callback_data": f"rs:reject:{request_id}:{zero_index}"},
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

        force_rerip = bool(state.get("force_rerip"))
        reuse_existing = bool(state.get("reuse_existing"))
        approved_clips: list[dict[str, Any]] = []
        rendered_clips: list[dict[str, Any]] = []
        prior_shorts: list[dict[str, Any]] = []
        if row["source_kind"] == "youtube":
            approval_history = _approved_clip_history_from_sheet(video_id)
            approved_clips = approval_history["unfinished"]
            rendered_clips = approval_history["rendered"]
            prior_shorts = _reviewed_short_history_from_sheet(video_id)
            if force_rerip:
                # A re-rip is a fresh editorial pass; old approvals are learning
                # examples, not automatic render instructions.
                approved_clips = []

            # Ten reviewed shorts is enough history to treat this ID as already
            # discovered. Below ten, search again for additional distinct shorts.
            if reuse_existing or (len(prior_shorts) >= 10 and not force_rerip):
                if approved_clips:
                    reviews = {
                        str(index): {
                            "status": "queued",
                            "reviewed_at": clip.get("_reviewed_at") or now(),
                            "user_id": clip.get("_reviewer_user_id", ""),
                            "recovered_from_sheet": True,
                        }
                        for index, clip in enumerate(approved_clips)
                    }
                    recovered_result = {
                        "analysis": {
                            "content_type": "",
                            "main_theme": "",
                            "key_ideas": [],
                            "keywords": [],
                        },
                        "segments": approved_clips,
                    }
                    recovered_state = {
                        **state,
                        "stage": "awaiting_review",
                        "video_path": str(video),
                        "source_reused": reused,
                        "result": recovered_result,
                        "candidate_reviews": reviews,
                        "recovered_approvals_from_sheet": True,
                    }
                    _save(request_id, "awaiting_review", recovered_state)
                    send(
                        chat_id,
                        f"♻️ Recovered {len(approved_clips)} approved unfinished short(s) "
                        f"for YouTube ID {video_id}. Rendering only those shorts now.",
                    )
                    for index in range(len(approved_clips)):
                        RENDER_EXECUTOR.submit(_render_approved, request_id, index, chat_id)
                    return
                if rendered_clips:
                    existing_links = [
                        str(clip.get("_clip_url") or "")
                        for clip in rendered_clips
                        if clip.get("_clip_url")
                    ]
                    _save(
                        request_id,
                        "already_rendered",
                        {
                            **state,
                            "stage": "already_rendered",
                            "video_path": str(video),
                            "source_reused": reused,
                            "existing_clip_links": existing_links,
                        },
                    )
                    links_text = "\n".join(existing_links[:20])
                    send(
                        chat_id,
                        f"✅ YouTube ID {video_id} already has "
                        f"{len(prior_shorts)} reviewed shorts in the sheet, so GPT "
                        "selection was not run again."
                        + (f"\n\n{links_text}" if links_text else ""),
                    )
                    return

        prior_count = len(prior_shorts)
        needed = 20 if force_rerip else max(1, 20 - prior_count)
        _save(
            request_id,
            "selecting",
            {
                **state,
                "stage": "selection",
                "video_path": str(video),
                "source_reused": reused,
            },
        )
        if force_rerip:
            send(
                chat_id,
                f"🧠 Re-ripping YouTube ID {video_id}. Selecting a fresh set of up "
                "to 20 Shorts using your approval and rejection history.",
            )
        elif prior_count:
            send(
                chat_id,
                f"🧠 Only {prior_count} previously reviewed short(s) were found for "
                f"{video_id}. Searching again for up to {needed} additional distinct shorts.",
            )
        else:
            send(
                chat_id,
                "🧠 Reviewing the full eligible transcript and targeting approximately "
                "20 distinct complete shorts.",
            )
        import main
        enriched = [
            {**item, "video_id": state["parsed"].get("video_id", "drive-source")}
            for item in segments
        ]
        learning_prompt = _short_learning_prompt(
            video_id if row["source_kind"] == "youtube" else ""
        )
        prior_summary = "\n".join(
            f"- {item['start']:.2f}-{item['end']:.2f}: {item.get('transcript', '')}"
            for item in prior_shorts
        )
        if force_rerip:
            supplemental = (
                "Perform a fresh re-rip and select up to 20 of the strongest Shorts. "
                "Previously approved Shorts may be selected again if they remain among "
                "the best. Never select a previously rejected complete Short again.\n"
                + learning_prompt
                + (f"\nSAME-VIDEO REVIEW HISTORY:\n{prior_summary[:12000]}" if prior_summary else "")
            )
        elif prior_shorts:
            supplemental = (
                f"Find up to {needed} ADDITIONAL distinct Shorts. Do not repeat any "
                "previous complete thought, lesson, payoff, or transcript. Time overlap "
                "is allowed only when the new Short is materially different.\n"
                + learning_prompt
                + f"\nPREVIOUSLY REVIEWED SHORTS:\n{prior_summary[:12000]}"
            )
        else:
            supplemental = learning_prompt
        result = main.call_openai_for_clips(enriched, supplemental)
        result = validate_complete_candidates(result, enriched)
        excluded_history = (
            [item for item in prior_shorts if item.get("decision") == "rejected"]
            if force_rerip
            else prior_shorts
        )
        prior_texts = {
            " ".join(str(item.get("transcript", "")).lower().split())
            for item in excluded_history
            if item.get("transcript")
        }
        new_segments = [
            clip
            for clip in result.get("segments", [])
            if " ".join(str(clip.get("transcript", "")).lower().split())
            not in prior_texts
        ]
        new_segments = select_non_overlapping(
            new_segments, limit=needed, allow_overlap=True
        )
        next_number = max(
            [int(item.get("candidate_number") or 0) for item in prior_shorts] or [0]
        ) + 1
        for offset, clip in enumerate(new_segments):
            clip["candidate_number"] = next_number + offset

        if row["mode"] == "topics":
            raise RuntimeError(
                "Topic-only Telegram processing requires the dual-lane selector before rendering"
            )

        combined = approved_clips + new_segments
        result["segments"] = combined
        reviews = {
            str(index): {
                "status": "queued",
                "reviewed_at": clip.get("_reviewed_at") or now(),
                "user_id": clip.get("_reviewer_user_id", ""),
                "recovered_from_sheet": True,
            }
            for index, clip in enumerate(approved_clips)
        }
        final_state = {
            **state,
            "stage": "awaiting_review",
            "video_path": str(video),
            "source_reused": reused,
            "result": result,
            "candidate_reviews": reviews,
        }
        _save(request_id, "awaiting_review", final_state)
        if approved_clips:
            send(
                chat_id,
                f"♻️ Resuming {len(approved_clips)} previously approved unfinished "
                "short(s) while presenting additional shorts for review.",
            )
            for index in range(len(approved_clips)):
                RENDER_EXECUTOR.submit(_render_approved, request_id, index, chat_id)
        _send_candidates(
            chat_id, request_id, result, start_index=len(approved_clips)
        )
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
    rerip_choice = re.fullmatch(
        r"rs:(rerip|reuse):([A-Za-z0-9-]+)", callback_data
    )
    if rerip_choice:
        choice, request_id = rerip_choice.groups()
        with _LOCK, _telegram_db() as db:
            row = db.execute(
                "SELECT * FROM telegram_requests WHERE request_id=?", (request_id,)
            ).fetchone()
            if not row:
                return {"status": "not_found"}
            state = json.loads(row["state_json"])
            state["force_rerip"] = choice == "rerip"
            state["reuse_existing"] = choice == "reuse"
            state["stage"] = "accepted"
            db.execute(
                "UPDATE telegram_requests SET status=?, state_json=?, updated_at=? "
                "WHERE request_id=?",
                ("accepted", json.dumps(state), now(), request_id),
            )
        send(
            chat_id,
            (
                "♻️ Re-ripping this video's Shorts with your approval/rejection "
                "learning."
                if choice == "rerip"
                else "▶️ Reusing the existing reviewed Shorts and approved renders."
            ),
        )
        background_tasks.add_task(_process, request_id)
        return {"status": choice, "request_id": request_id}

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
                    "short_index": index,
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
                f"Short {index + 1} approved and queued for rendering. "
                f"Up to {RIPPED_SHORTS_RENDER_WORKERS} clips render at once; the rest wait.\n"
                f"{_render_progress_text(request_id)}",
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
            send(chat_id, f"Short {index + 1} rejected.")
        return {"status": verb, "request_id": request_id, "short_index": index}
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
    if parsed["source_kind"] == "youtube":
        try:
            prior_shorts = _reviewed_short_history_from_sheet(parsed["video_id"])
        except Exception:
            logger.exception("Could not check prior Shorts history before intake")
            prior_shorts = []
        if prior_shorts:
            state["stage"] = "awaiting_rerip_choice"
            state["prior_reviewed_short_count"] = len(prior_shorts)
            _save(request_id, "awaiting_rerip_choice", state)
            telegram(
                "sendMessage",
                {
                    "chat_id": chat_id,
                    "text": (
                        f"♻️ YouTube ID {parsed['video_id']} already has "
                        f"{len(prior_shorts)} reviewed Short(s).\n\n"
                        "Do you want to re-rip the Shorts using everything learned "
                        "from your approvals and rejections?"
                    ),
                    "reply_markup": {
                        "inline_keyboard": [[
                            {
                                "text": "♻️ Re-rip Shorts",
                                "callback_data": f"rs:rerip:{request_id}",
                            },
                            {
                                "text": "▶️ Use Existing",
                                "callback_data": f"rs:reuse:{request_id}",
                            },
                        ]]
                    },
                },
            )
            return {
                "status": "awaiting_rerip_choice",
                "request_id": request_id,
                "reviewed_shorts": len(prior_shorts),
            }
    send(
        chat_id,
        f"✅ Ripped Shorts request accepted\n"
        f"Source: {parsed['source_kind'].title()}\nJob ID: {request_id}",
    )
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
        send(
            chat_id,
            f"🎬 Short {index + 1} is now rendering.\n"
            f"{_render_progress_text(request_id)}",
        )
        candidate = state["result"]["segments"][index]
        user_id = str(
            (state.get("candidate_reviews") or {}).get(str(index), {}).get("user_id", "")
        )
        video = Path(state["video_path"])
        import main

        payload_candidate = dict(candidate)
        payload_candidate["candidate_number"] = int(
            candidate.get("candidate_number") or index + 1
        )
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
            f"✅ Short {index + 1} rendered and uploaded to DRIVE_FOLDER_ID:\n"
            f"{clip.get('clip_url', '')}\n"
            f"{_render_progress_text(request_id)}",
        )
        _notify_render_queue_complete(request_id, chat_id)
    except Exception as exc:
        logger.exception(
            "Short render failed request_id=%s candidate=%s", request_id, index + 1
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
        send(
            chat_id,
            f"❌ Short {index + 1} render failed:\n{str(exc)[:1500]}\n"
            f"{_render_progress_text(request_id)}",
        )
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
, ordered[end_index][2])
        and ordered[end_index + 1][1] - ordered[start_index][0] <= 90
    ):
        end_index += 1

    selected = ordered[start_index : end_index + 1]
    completed_start, completed_end = selected[0][0], selected[-1][1]
    return {
        **candidate,
        "start": completed_start,
        "end": completed_end,
        "duration": round(completed_end - completed_start, 3),
        "transcript": " ".join(text for _, _, text in selected),
    }


def validate_complete_candidates(payload: dict, transcript_segments: list[dict], tolerance: float = 0.75) -> dict:
    """Reject invented text and enforce complete sentence/thought boundaries."""
    boundaries = []
    for item in transcript_segments:
        start = float(item.get("start", 0))
        end = float(item.get("end", start + float(item.get("duration", 0))))
        boundaries.append((start, end, str(item.get("text", "")).strip()))
    valid, rejected = [], []
    for original in payload.get("segments", []):
        candidate = _sentence_complete_candidate(dict(original), transcript_segments, tolerance)
        start = float(candidate.get("start", -1))
        end = float(candidate.get("end", start + float(candidate.get("duration", 0))))
        begins_cleanly = any(abs(start - seg_start) <= tolerance for seg_start, _, _ in boundaries)
        ends_cleanly = any(abs(end - seg_end) <= tolerance for _, seg_end, _ in boundaries)
        included = [text for seg_start, seg_end, text in boundaries if seg_start >= start - tolerance and seg_end <= end + tolerance]
        expected = " ".join(included).casefold()
        quoted = str(candidate.get("transcript", "")).strip().casefold()
        complete_end = bool(re.search(r'[.!?]["’\']?

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
    log_request_id = candidate.get("_sheet_request_id") or request_id
    candidate_number = int(candidate.get("candidate_number") or index + 1)
    analysis = state["result"].get("analysis") or {}
    parsed = state.get("parsed") or {}
    reviewed = (state.get("candidate_reviews") or {}).get(str(index), {})
    row_values = [
        now(),
        log_request_id,
        parsed.get("video_id", ""),
        parsed.get("source_value", ""),
        analysis.get("content_type", ""),
        analysis.get("main_theme", ""),
        ", ".join(str(value) for value in analysis.get("key_ideas", [])),
        ", ".join(str(value) for value in analysis.get("keywords", [])),
        candidate_number,
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
            and str(values[1]) == str(log_request_id)
            and str(values[8]) == str(candidate_number)
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


def _approved_clip_history_from_sheet(video_id: str) -> dict[str, list[dict[str, Any]]]:
    """Return approved rendered and unfinished clips for an exact YouTube ID."""
    import main

    _, _, sheets = main.get_google_services()
    response = sheets.spreadsheets().values().get(
        spreadsheetId=RIPPED_LOG_SHEET_ID,
        range=f"'{RIPPED_LOG_SHEET_TAB}'!A:U",
    ).execute()
    rows = response.get("values", [])
    recovered: dict[tuple[float, float], dict[str, Any]] = {}
    for values in rows[1:]:
        padded = list(values) + [""] * (21 - len(values))
        if str(padded[2]).strip() != video_id:
            continue
        if str(padded[15]).strip().lower() != "approved":
            continue
        render_status = str(padded[19]).strip().lower()
        clip_url = str(padded[18]).strip()
        is_rendered = render_status == "rendered" and bool(clip_url)
        try:
            start = float(padded[9])
            end = float(padded[10])
            duration = float(padded[11]) if str(padded[11]).strip() else end - start
            candidate_number = int(float(padded[8]))
        except (TypeError, ValueError):
            logger.warning("Skipping malformed approved Ripped Shorts row for %s", video_id)
            continue
        recovered[(start, end)] = {
            "start": start,
            "end": end,
            "duration": duration,
            "category": padded[12],
            "transcript": padded[13],
            "reason": padded[14],
            "candidate_number": candidate_number,
            "_sheet_request_id": str(padded[1]).strip(),
            "_reviewed_at": str(padded[16]).strip(),
            "_reviewer_user_id": str(padded[20]).strip(),
            "_render_status": render_status,
            "_clip_url": clip_url,
            "_is_rendered": is_rendered,
        }
    ordered = sorted(recovered.values(), key=lambda item: item["candidate_number"])
    return {
        "rendered": [item for item in ordered if item["_is_rendered"]],
        "unfinished": [item for item in ordered if not item["_is_rendered"]],
    }


def _reviewed_short_history_from_sheet(video_id: str) -> list[dict[str, Any]]:
    """Return distinct reviewed shorts, including approvals and rejections."""
    import main

    _, _, sheets = main.get_google_services()
    response = sheets.spreadsheets().values().get(
        spreadsheetId=RIPPED_LOG_SHEET_ID,
        range=f"'{RIPPED_LOG_SHEET_TAB}'!A:U",
    ).execute()
    rows = response.get("values", [])
    found: dict[tuple[float, float], dict[str, Any]] = {}
    for values in rows[1:]:
        padded = list(values) + [""] * (21 - len(values))
        if str(padded[2]).strip() != video_id:
            continue
        try:
            start, end = float(padded[9]), float(padded[10])
            number = int(float(padded[8]))
        except (TypeError, ValueError):
            continue
        found[(start, end)] = {
            "start": start,
            "end": end,
            "transcript": str(padded[13]).strip(),
            "decision": str(padded[15]).strip().lower(),
            "candidate_number": number,
        }
    return sorted(found.values(), key=lambda item: item["candidate_number"])


def _short_learning_prompt(video_id: str) -> str:
    """Build balanced approval/rejection examples from the durable Shorts ledger."""
    import main

    _, _, sheets = main.get_google_services()
    response = sheets.spreadsheets().values().get(
        spreadsheetId=RIPPED_LOG_SHEET_ID,
        range=f"'{RIPPED_LOG_SHEET_TAB}'!A:U",
    ).execute()
    rows = response.get("values", [])[1:]
    same_video_types = {
        str((list(values) + [""] * 21)[4]).strip().lower()
        for values in rows
        if str((list(values) + [""] * 21)[2]).strip() == video_id
        and str((list(values) + [""] * 21)[4]).strip()
    }
    examples = []
    for recency, values in enumerate(rows):
        padded = list(values) + [""] * (21 - len(values))
        decision = str(padded[15]).strip().lower()
        if decision not in {"approved", "rejected"}:
            continue
        transcript = str(padded[13]).strip()
        if not transcript:
            continue
        content_type = str(padded[4]).strip().lower()
        priority = (
            2 if str(padded[2]).strip() == video_id else
            1 if content_type and content_type in same_video_types else
            0
        )
        examples.append({
            "decision": decision,
            "priority": priority,
            "recency": recency,
            "content_type": content_type or "unknown",
            "category": str(padded[12]).strip() or "unknown",
            "transcript": transcript[:700],
            "reason": str(padded[14]).strip()[:300],
        })

    examples.sort(
        key=lambda item: (item["priority"], item["recency"]), reverse=True
    )
    approved = [item for item in examples if item["decision"] == "approved"][:10]
    rejected = [item for item in examples if item["decision"] == "rejected"][:10]

    def format_examples(label: str, selected: list[dict[str, Any]]) -> str:
        if not selected:
            return f"{label}: none recorded yet"
        lines = [label + ":"]
        for item in selected:
            lines.append(
                f"- type={item['content_type']} category={item['category']} | "
                f"{item['transcript']}"
                + (f" | prior reason={item['reason']}" if item["reason"] else "")
            )
        return "\n".join(lines)

    return (
        "LEARNING FROM DEREK'S SHORT REVIEWS\n"
        "Treat approvals as positive structural examples and rejections as negative "
        "examples. Learn the qualities, not their exact wording or timestamps. "
        "Reject scripture-only readings without interpretation/application, generic "
        "blurbs without a payoff, incomplete setups, and context-dependent fragments. "
        "Require a standalone hook, developed point, and payoff.\n"
        + format_examples("APPROVED SHORTS", approved)
        + "\n"
        + format_examples("REJECTED SHORTS", rejected)
    )


def _safe_log_candidate(*args, **kwargs) -> None:
    try:
        _log_candidate_decision(*args, **kwargs)
    except Exception as exc:
        logger.exception("Ripped Shorts decision log failed: %s", exc)


def _send_candidates(
    chat_id: str, request_id: str, result: dict, *, start_index: int = 0
) -> None:
    clips = result.get("segments", [])
    review_count = max(0, len(clips) - start_index)
    send(
        chat_id,
        f"✅ Analysis complete\nJob ID: {request_id}\n"
        f"New shorts for review: {review_count}\n\n"
        "Approve only the shorts you want created.",
    )
    for zero_index in range(start_index, len(clips)):
        clip = clips[zero_index]
        short_number = int(clip.get("candidate_number") or zero_index + 1)
        transcript = str(clip.get("transcript", "")).strip()
        if len(transcript) > 2600:
            transcript = transcript[:2597] + "..."
        text = (
            f"Short {short_number}\n\n"
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
                {"text": "✅ Approve & Render", "callback_data": f"rs:approve:{request_id}:{zero_index}"},
                {"text": "❌ Reject", "callback_data": f"rs:reject:{request_id}:{zero_index}"},
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

        force_rerip = bool(state.get("force_rerip"))
        reuse_existing = bool(state.get("reuse_existing"))
        approved_clips: list[dict[str, Any]] = []
        rendered_clips: list[dict[str, Any]] = []
        prior_shorts: list[dict[str, Any]] = []
        if row["source_kind"] == "youtube":
            approval_history = _approved_clip_history_from_sheet(video_id)
            approved_clips = approval_history["unfinished"]
            rendered_clips = approval_history["rendered"]
            prior_shorts = _reviewed_short_history_from_sheet(video_id)
            if force_rerip:
                # A re-rip is a fresh editorial pass; old approvals are learning
                # examples, not automatic render instructions.
                approved_clips = []

            # Ten reviewed shorts is enough history to treat this ID as already
            # discovered. Below ten, search again for additional distinct shorts.
            if reuse_existing or (len(prior_shorts) >= 10 and not force_rerip):
                if approved_clips:
                    reviews = {
                        str(index): {
                            "status": "queued",
                            "reviewed_at": clip.get("_reviewed_at") or now(),
                            "user_id": clip.get("_reviewer_user_id", ""),
                            "recovered_from_sheet": True,
                        }
                        for index, clip in enumerate(approved_clips)
                    }
                    recovered_result = {
                        "analysis": {
                            "content_type": "",
                            "main_theme": "",
                            "key_ideas": [],
                            "keywords": [],
                        },
                        "segments": approved_clips,
                    }
                    recovered_state = {
                        **state,
                        "stage": "awaiting_review",
                        "video_path": str(video),
                        "source_reused": reused,
                        "result": recovered_result,
                        "candidate_reviews": reviews,
                        "recovered_approvals_from_sheet": True,
                    }
                    _save(request_id, "awaiting_review", recovered_state)
                    send(
                        chat_id,
                        f"♻️ Recovered {len(approved_clips)} approved unfinished short(s) "
                        f"for YouTube ID {video_id}. Rendering only those shorts now.",
                    )
                    for index in range(len(approved_clips)):
                        RENDER_EXECUTOR.submit(_render_approved, request_id, index, chat_id)
                    return
                if rendered_clips:
                    existing_links = [
                        str(clip.get("_clip_url") or "")
                        for clip in rendered_clips
                        if clip.get("_clip_url")
                    ]
                    _save(
                        request_id,
                        "already_rendered",
                        {
                            **state,
                            "stage": "already_rendered",
                            "video_path": str(video),
                            "source_reused": reused,
                            "existing_clip_links": existing_links,
                        },
                    )
                    links_text = "\n".join(existing_links[:20])
                    send(
                        chat_id,
                        f"✅ YouTube ID {video_id} already has "
                        f"{len(prior_shorts)} reviewed shorts in the sheet, so GPT "
                        "selection was not run again."
                        + (f"\n\n{links_text}" if links_text else ""),
                    )
                    return

        prior_count = len(prior_shorts)
        needed = 20 if force_rerip else max(1, 20 - prior_count)
        _save(
            request_id,
            "selecting",
            {
                **state,
                "stage": "selection",
                "video_path": str(video),
                "source_reused": reused,
            },
        )
        if force_rerip:
            send(
                chat_id,
                f"🧠 Re-ripping YouTube ID {video_id}. Selecting a fresh set of up "
                "to 20 Shorts using your approval and rejection history.",
            )
        elif prior_count:
            send(
                chat_id,
                f"🧠 Only {prior_count} previously reviewed short(s) were found for "
                f"{video_id}. Searching again for up to {needed} additional distinct shorts.",
            )
        else:
            send(
                chat_id,
                "🧠 Reviewing the full eligible transcript and targeting approximately "
                "20 distinct complete shorts.",
            )
        import main
        enriched = [
            {**item, "video_id": state["parsed"].get("video_id", "drive-source")}
            for item in segments
        ]
        learning_prompt = _short_learning_prompt(
            video_id if row["source_kind"] == "youtube" else ""
        )
        prior_summary = "\n".join(
            f"- {item['start']:.2f}-{item['end']:.2f}: {item.get('transcript', '')}"
            for item in prior_shorts
        )
        if force_rerip:
            supplemental = (
                "Perform a fresh re-rip and select up to 20 of the strongest Shorts. "
                "Previously approved Shorts may be selected again if they remain among "
                "the best. Never select a previously rejected complete Short again.\n"
                + learning_prompt
                + (f"\nSAME-VIDEO REVIEW HISTORY:\n{prior_summary[:12000]}" if prior_summary else "")
            )
        elif prior_shorts:
            supplemental = (
                f"Find up to {needed} ADDITIONAL distinct Shorts. Do not repeat any "
                "previous complete thought, lesson, payoff, or transcript. Time overlap "
                "is allowed only when the new Short is materially different.\n"
                + learning_prompt
                + f"\nPREVIOUSLY REVIEWED SHORTS:\n{prior_summary[:12000]}"
            )
        else:
            supplemental = learning_prompt
        result = main.call_openai_for_clips(enriched, supplemental)
        result = validate_complete_candidates(result, enriched)
        excluded_history = (
            [item for item in prior_shorts if item.get("decision") == "rejected"]
            if force_rerip
            else prior_shorts
        )
        prior_texts = {
            " ".join(str(item.get("transcript", "")).lower().split())
            for item in excluded_history
            if item.get("transcript")
        }
        new_segments = [
            clip
            for clip in result.get("segments", [])
            if " ".join(str(clip.get("transcript", "")).lower().split())
            not in prior_texts
        ]
        new_segments = select_non_overlapping(
            new_segments, limit=needed, allow_overlap=True
        )
        next_number = max(
            [int(item.get("candidate_number") or 0) for item in prior_shorts] or [0]
        ) + 1
        for offset, clip in enumerate(new_segments):
            clip["candidate_number"] = next_number + offset

        if row["mode"] == "topics":
            raise RuntimeError(
                "Topic-only Telegram processing requires the dual-lane selector before rendering"
            )

        combined = approved_clips + new_segments
        result["segments"] = combined
        reviews = {
            str(index): {
                "status": "queued",
                "reviewed_at": clip.get("_reviewed_at") or now(),
                "user_id": clip.get("_reviewer_user_id", ""),
                "recovered_from_sheet": True,
            }
            for index, clip in enumerate(approved_clips)
        }
        final_state = {
            **state,
            "stage": "awaiting_review",
            "video_path": str(video),
            "source_reused": reused,
            "result": result,
            "candidate_reviews": reviews,
        }
        _save(request_id, "awaiting_review", final_state)
        if approved_clips:
            send(
                chat_id,
                f"♻️ Resuming {len(approved_clips)} previously approved unfinished "
                "short(s) while presenting additional shorts for review.",
            )
            for index in range(len(approved_clips)):
                RENDER_EXECUTOR.submit(_render_approved, request_id, index, chat_id)
        _send_candidates(
            chat_id, request_id, result, start_index=len(approved_clips)
        )
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
    rerip_choice = re.fullmatch(
        r"rs:(rerip|reuse):([A-Za-z0-9-]+)", callback_data
    )
    if rerip_choice:
        choice, request_id = rerip_choice.groups()
        with _LOCK, _telegram_db() as db:
            row = db.execute(
                "SELECT * FROM telegram_requests WHERE request_id=?", (request_id,)
            ).fetchone()
            if not row:
                return {"status": "not_found"}
            state = json.loads(row["state_json"])
            state["force_rerip"] = choice == "rerip"
            state["reuse_existing"] = choice == "reuse"
            state["stage"] = "accepted"
            db.execute(
                "UPDATE telegram_requests SET status=?, state_json=?, updated_at=? "
                "WHERE request_id=?",
                ("accepted", json.dumps(state), now(), request_id),
            )
        send(
            chat_id,
            (
                "♻️ Re-ripping this video's Shorts with your approval/rejection "
                "learning."
                if choice == "rerip"
                else "▶️ Reusing the existing reviewed Shorts and approved renders."
            ),
        )
        background_tasks.add_task(_process, request_id)
        return {"status": choice, "request_id": request_id}

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
                    "short_index": index,
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
                f"Short {index + 1} approved and queued for rendering. "
                f"Up to {RIPPED_SHORTS_RENDER_WORKERS} clips render at once; the rest wait.\n"
                f"{_render_progress_text(request_id)}",
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
            send(chat_id, f"Short {index + 1} rejected.")
        return {"status": verb, "request_id": request_id, "short_index": index}
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
    if parsed["source_kind"] == "youtube":
        try:
            prior_shorts = _reviewed_short_history_from_sheet(parsed["video_id"])
        except Exception:
            logger.exception("Could not check prior Shorts history before intake")
            prior_shorts = []
        if prior_shorts:
            state["stage"] = "awaiting_rerip_choice"
            state["prior_reviewed_short_count"] = len(prior_shorts)
            _save(request_id, "awaiting_rerip_choice", state)
            telegram(
                "sendMessage",
                {
                    "chat_id": chat_id,
                    "text": (
                        f"♻️ YouTube ID {parsed['video_id']} already has "
                        f"{len(prior_shorts)} reviewed Short(s).\n\n"
                        "Do you want to re-rip the Shorts using everything learned "
                        "from your approvals and rejections?"
                    ),
                    "reply_markup": {
                        "inline_keyboard": [[
                            {
                                "text": "♻️ Re-rip Shorts",
                                "callback_data": f"rs:rerip:{request_id}",
                            },
                            {
                                "text": "▶️ Use Existing",
                                "callback_data": f"rs:reuse:{request_id}",
                            },
                        ]]
                    },
                },
            )
            return {
                "status": "awaiting_rerip_choice",
                "request_id": request_id,
                "reviewed_shorts": len(prior_shorts),
            }
    send(
        chat_id,
        f"✅ Ripped Shorts request accepted\n"
        f"Source: {parsed['source_kind'].title()}\nJob ID: {request_id}",
    )
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
        send(
            chat_id,
            f"🎬 Short {index + 1} is now rendering.\n"
            f"{_render_progress_text(request_id)}",
        )
        candidate = state["result"]["segments"][index]
        user_id = str(
            (state.get("candidate_reviews") or {}).get(str(index), {}).get("user_id", "")
        )
        video = Path(state["video_path"])
        import main

        payload_candidate = dict(candidate)
        payload_candidate["candidate_number"] = int(
            candidate.get("candidate_number") or index + 1
        )
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
            f"✅ Short {index + 1} rendered and uploaded to DRIVE_FOLDER_ID:\n"
            f"{clip.get('clip_url', '')}\n"
            f"{_render_progress_text(request_id)}",
        )
        _notify_render_queue_complete(request_id, chat_id)
    except Exception as exc:
        logger.exception(
            "Short render failed request_id=%s candidate=%s", request_id, index + 1
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
        send(
            chat_id,
            f"❌ Short {index + 1} render failed:\n{str(exc)[:1500]}\n"
            f"{_render_progress_text(request_id)}",
        )
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
, str(candidate.get("transcript", "")).strip()))
        if begins_cleanly and ends_cleanly and quoted and expected and (quoted in expected or expected in quoted) and end > start and end - start <= 90 and complete_end:
            candidate.update({"start": start, "end": end, "duration": round(end - start, 3)})
            valid.append(candidate)
        else:
            rejected.append({"start": start, "end": end, "reason": "Unsupported text or incomplete sentence/thought boundary"})
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
    log_request_id = candidate.get("_sheet_request_id") or request_id
    candidate_number = int(candidate.get("candidate_number") or index + 1)
    analysis = state["result"].get("analysis") or {}
    parsed = state.get("parsed") or {}
    reviewed = (state.get("candidate_reviews") or {}).get(str(index), {})
    row_values = [
        now(),
        log_request_id,
        parsed.get("video_id", ""),
        parsed.get("source_value", ""),
        analysis.get("content_type", ""),
        analysis.get("main_theme", ""),
        ", ".join(str(value) for value in analysis.get("key_ideas", [])),
        ", ".join(str(value) for value in analysis.get("keywords", [])),
        candidate_number,
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
            and str(values[1]) == str(log_request_id)
            and str(values[8]) == str(candidate_number)
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


def _approved_clip_history_from_sheet(video_id: str) -> dict[str, list[dict[str, Any]]]:
    """Return approved rendered and unfinished clips for an exact YouTube ID."""
    import main

    _, _, sheets = main.get_google_services()
    response = sheets.spreadsheets().values().get(
        spreadsheetId=RIPPED_LOG_SHEET_ID,
        range=f"'{RIPPED_LOG_SHEET_TAB}'!A:U",
    ).execute()
    rows = response.get("values", [])
    recovered: dict[tuple[float, float], dict[str, Any]] = {}
    for values in rows[1:]:
        padded = list(values) + [""] * (21 - len(values))
        if str(padded[2]).strip() != video_id:
            continue
        if str(padded[15]).strip().lower() != "approved":
            continue
        render_status = str(padded[19]).strip().lower()
        clip_url = str(padded[18]).strip()
        is_rendered = render_status == "rendered" and bool(clip_url)
        try:
            start = float(padded[9])
            end = float(padded[10])
            duration = float(padded[11]) if str(padded[11]).strip() else end - start
            candidate_number = int(float(padded[8]))
        except (TypeError, ValueError):
            logger.warning("Skipping malformed approved Ripped Shorts row for %s", video_id)
            continue
        recovered[(start, end)] = {
            "start": start,
            "end": end,
            "duration": duration,
            "category": padded[12],
            "transcript": padded[13],
            "reason": padded[14],
            "candidate_number": candidate_number,
            "_sheet_request_id": str(padded[1]).strip(),
            "_reviewed_at": str(padded[16]).strip(),
            "_reviewer_user_id": str(padded[20]).strip(),
            "_render_status": render_status,
            "_clip_url": clip_url,
            "_is_rendered": is_rendered,
        }
    ordered = sorted(recovered.values(), key=lambda item: item["candidate_number"])
    return {
        "rendered": [item for item in ordered if item["_is_rendered"]],
        "unfinished": [item for item in ordered if not item["_is_rendered"]],
    }


def _reviewed_short_history_from_sheet(video_id: str) -> list[dict[str, Any]]:
    """Return distinct reviewed shorts, including approvals and rejections."""
    import main

    _, _, sheets = main.get_google_services()
    response = sheets.spreadsheets().values().get(
        spreadsheetId=RIPPED_LOG_SHEET_ID,
        range=f"'{RIPPED_LOG_SHEET_TAB}'!A:U",
    ).execute()
    rows = response.get("values", [])
    found: dict[tuple[float, float], dict[str, Any]] = {}
    for values in rows[1:]:
        padded = list(values) + [""] * (21 - len(values))
        if str(padded[2]).strip() != video_id:
            continue
        try:
            start, end = float(padded[9]), float(padded[10])
            number = int(float(padded[8]))
        except (TypeError, ValueError):
            continue
        found[(start, end)] = {
            "start": start,
            "end": end,
            "transcript": str(padded[13]).strip(),
            "decision": str(padded[15]).strip().lower(),
            "candidate_number": number,
        }
    return sorted(found.values(), key=lambda item: item["candidate_number"])


def _short_learning_prompt(video_id: str) -> str:
    """Build balanced approval/rejection examples from the durable Shorts ledger."""
    import main

    _, _, sheets = main.get_google_services()
    response = sheets.spreadsheets().values().get(
        spreadsheetId=RIPPED_LOG_SHEET_ID,
        range=f"'{RIPPED_LOG_SHEET_TAB}'!A:U",
    ).execute()
    rows = response.get("values", [])[1:]
    same_video_types = {
        str((list(values) + [""] * 21)[4]).strip().lower()
        for values in rows
        if str((list(values) + [""] * 21)[2]).strip() == video_id
        and str((list(values) + [""] * 21)[4]).strip()
    }
    examples = []
    for recency, values in enumerate(rows):
        padded = list(values) + [""] * (21 - len(values))
        decision = str(padded[15]).strip().lower()
        if decision not in {"approved", "rejected"}:
            continue
        transcript = str(padded[13]).strip()
        if not transcript:
            continue
        content_type = str(padded[4]).strip().lower()
        priority = (
            2 if str(padded[2]).strip() == video_id else
            1 if content_type and content_type in same_video_types else
            0
        )
        examples.append({
            "decision": decision,
            "priority": priority,
            "recency": recency,
            "content_type": content_type or "unknown",
            "category": str(padded[12]).strip() or "unknown",
            "transcript": transcript[:700],
            "reason": str(padded[14]).strip()[:300],
        })

    examples.sort(
        key=lambda item: (item["priority"], item["recency"]), reverse=True
    )
    approved = [item for item in examples if item["decision"] == "approved"][:10]
    rejected = [item for item in examples if item["decision"] == "rejected"][:10]

    def format_examples(label: str, selected: list[dict[str, Any]]) -> str:
        if not selected:
            return f"{label}: none recorded yet"
        lines = [label + ":"]
        for item in selected:
            lines.append(
                f"- type={item['content_type']} category={item['category']} | "
                f"{item['transcript']}"
                + (f" | prior reason={item['reason']}" if item["reason"] else "")
            )
        return "\n".join(lines)

    return (
        "LEARNING FROM DEREK'S SHORT REVIEWS\n"
        "Treat approvals as positive structural examples and rejections as negative "
        "examples. Learn the qualities, not their exact wording or timestamps. "
        "Reject scripture-only readings without interpretation/application, generic "
        "blurbs without a payoff, incomplete setups, and context-dependent fragments. "
        "Require a standalone hook, developed point, and payoff.\n"
        + format_examples("APPROVED SHORTS", approved)
        + "\n"
        + format_examples("REJECTED SHORTS", rejected)
    )


def _safe_log_candidate(*args, **kwargs) -> None:
    try:
        _log_candidate_decision(*args, **kwargs)
    except Exception as exc:
        logger.exception("Ripped Shorts decision log failed: %s", exc)


def _send_candidates(
    chat_id: str, request_id: str, result: dict, *, start_index: int = 0
) -> None:
    clips = result.get("segments", [])
    review_count = max(0, len(clips) - start_index)
    send(
        chat_id,
        f"✅ Analysis complete\nJob ID: {request_id}\n"
        f"New shorts for review: {review_count}\n\n"
        "Approve only the shorts you want created.",
    )
    for zero_index in range(start_index, len(clips)):
        clip = clips[zero_index]
        short_number = int(clip.get("candidate_number") or zero_index + 1)
        transcript = str(clip.get("transcript", "")).strip()
        if len(transcript) > 2600:
            transcript = transcript[:2597] + "..."
        text = (
            f"Short {short_number}\n\n"
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
                {"text": "✅ Approve & Render", "callback_data": f"rs:approve:{request_id}:{zero_index}"},
                {"text": "❌ Reject", "callback_data": f"rs:reject:{request_id}:{zero_index}"},
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

        force_rerip = bool(state.get("force_rerip"))
        reuse_existing = bool(state.get("reuse_existing"))
        approved_clips: list[dict[str, Any]] = []
        rendered_clips: list[dict[str, Any]] = []
        prior_shorts: list[dict[str, Any]] = []
        if row["source_kind"] == "youtube":
            approval_history = _approved_clip_history_from_sheet(video_id)
            approved_clips = approval_history["unfinished"]
            rendered_clips = approval_history["rendered"]
            prior_shorts = _reviewed_short_history_from_sheet(video_id)
            if force_rerip:
                # A re-rip is a fresh editorial pass; old approvals are learning
                # examples, not automatic render instructions.
                approved_clips = []

            # Ten reviewed shorts is enough history to treat this ID as already
            # discovered. Below ten, search again for additional distinct shorts.
            if reuse_existing or (len(prior_shorts) >= 10 and not force_rerip):
                if approved_clips:
                    reviews = {
                        str(index): {
                            "status": "queued",
                            "reviewed_at": clip.get("_reviewed_at") or now(),
                            "user_id": clip.get("_reviewer_user_id", ""),
                            "recovered_from_sheet": True,
                        }
                        for index, clip in enumerate(approved_clips)
                    }
                    recovered_result = {
                        "analysis": {
                            "content_type": "",
                            "main_theme": "",
                            "key_ideas": [],
                            "keywords": [],
                        },
                        "segments": approved_clips,
                    }
                    recovered_state = {
                        **state,
                        "stage": "awaiting_review",
                        "video_path": str(video),
                        "source_reused": reused,
                        "result": recovered_result,
                        "candidate_reviews": reviews,
                        "recovered_approvals_from_sheet": True,
                    }
                    _save(request_id, "awaiting_review", recovered_state)
                    send(
                        chat_id,
                        f"♻️ Recovered {len(approved_clips)} approved unfinished short(s) "
                        f"for YouTube ID {video_id}. Rendering only those shorts now.",
                    )
                    for index in range(len(approved_clips)):
                        RENDER_EXECUTOR.submit(_render_approved, request_id, index, chat_id)
                    return
                if rendered_clips:
                    existing_links = [
                        str(clip.get("_clip_url") or "")
                        for clip in rendered_clips
                        if clip.get("_clip_url")
                    ]
                    _save(
                        request_id,
                        "already_rendered",
                        {
                            **state,
                            "stage": "already_rendered",
                            "video_path": str(video),
                            "source_reused": reused,
                            "existing_clip_links": existing_links,
                        },
                    )
                    links_text = "\n".join(existing_links[:20])
                    send(
                        chat_id,
                        f"✅ YouTube ID {video_id} already has "
                        f"{len(prior_shorts)} reviewed shorts in the sheet, so GPT "
                        "selection was not run again."
                        + (f"\n\n{links_text}" if links_text else ""),
                    )
                    return

        prior_count = len(prior_shorts)
        needed = 20 if force_rerip else max(1, 20 - prior_count)
        _save(
            request_id,
            "selecting",
            {
                **state,
                "stage": "selection",
                "video_path": str(video),
                "source_reused": reused,
            },
        )
        if force_rerip:
            send(
                chat_id,
                f"🧠 Re-ripping YouTube ID {video_id}. Selecting a fresh set of up "
                "to 20 Shorts using your approval and rejection history.",
            )
        elif prior_count:
            send(
                chat_id,
                f"🧠 Only {prior_count} previously reviewed short(s) were found for "
                f"{video_id}. Searching again for up to {needed} additional distinct shorts.",
            )
        else:
            send(
                chat_id,
                "🧠 Reviewing the full eligible transcript and targeting approximately "
                "20 distinct complete shorts.",
            )
        import main
        enriched = [
            {**item, "video_id": state["parsed"].get("video_id", "drive-source")}
            for item in segments
        ]
        learning_prompt = _short_learning_prompt(
            video_id if row["source_kind"] == "youtube" else ""
        )
        prior_summary = "\n".join(
            f"- {item['start']:.2f}-{item['end']:.2f}: {item.get('transcript', '')}"
            for item in prior_shorts
        )
        if force_rerip:
            supplemental = (
                "Perform a fresh re-rip and select up to 20 of the strongest Shorts. "
                "Previously approved Shorts may be selected again if they remain among "
                "the best. Never select a previously rejected complete Short again.\n"
                + learning_prompt
                + (f"\nSAME-VIDEO REVIEW HISTORY:\n{prior_summary[:12000]}" if prior_summary else "")
            )
        elif prior_shorts:
            supplemental = (
                f"Find up to {needed} ADDITIONAL distinct Shorts. Do not repeat any "
                "previous complete thought, lesson, payoff, or transcript. Time overlap "
                "is allowed only when the new Short is materially different.\n"
                + learning_prompt
                + f"\nPREVIOUSLY REVIEWED SHORTS:\n{prior_summary[:12000]}"
            )
        else:
            supplemental = learning_prompt
        result = main.call_openai_for_clips(enriched, supplemental)
        result = validate_complete_candidates(result, enriched)
        excluded_history = (
            [item for item in prior_shorts if item.get("decision") == "rejected"]
            if force_rerip
            else prior_shorts
        )
        prior_texts = {
            " ".join(str(item.get("transcript", "")).lower().split())
            for item in excluded_history
            if item.get("transcript")
        }
        new_segments = [
            clip
            for clip in result.get("segments", [])
            if " ".join(str(clip.get("transcript", "")).lower().split())
            not in prior_texts
        ]
        new_segments = select_non_overlapping(
            new_segments, limit=needed, allow_overlap=True
        )
        next_number = max(
            [int(item.get("candidate_number") or 0) for item in prior_shorts] or [0]
        ) + 1
        for offset, clip in enumerate(new_segments):
            clip["candidate_number"] = next_number + offset

        if row["mode"] == "topics":
            raise RuntimeError(
                "Topic-only Telegram processing requires the dual-lane selector before rendering"
            )

        combined = approved_clips + new_segments
        result["segments"] = combined
        reviews = {
            str(index): {
                "status": "queued",
                "reviewed_at": clip.get("_reviewed_at") or now(),
                "user_id": clip.get("_reviewer_user_id", ""),
                "recovered_from_sheet": True,
            }
            for index, clip in enumerate(approved_clips)
        }
        final_state = {
            **state,
            "stage": "awaiting_review",
            "video_path": str(video),
            "source_reused": reused,
            "result": result,
            "candidate_reviews": reviews,
        }
        _save(request_id, "awaiting_review", final_state)
        if approved_clips:
            send(
                chat_id,
                f"♻️ Resuming {len(approved_clips)} previously approved unfinished "
                "short(s) while presenting additional shorts for review.",
            )
            for index in range(len(approved_clips)):
                RENDER_EXECUTOR.submit(_render_approved, request_id, index, chat_id)
        _send_candidates(
            chat_id, request_id, result, start_index=len(approved_clips)
        )
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
    rerip_choice = re.fullmatch(
        r"rs:(rerip|reuse):([A-Za-z0-9-]+)", callback_data
    )
    if rerip_choice:
        choice, request_id = rerip_choice.groups()
        with _LOCK, _telegram_db() as db:
            row = db.execute(
                "SELECT * FROM telegram_requests WHERE request_id=?", (request_id,)
            ).fetchone()
            if not row:
                return {"status": "not_found"}
            state = json.loads(row["state_json"])
            state["force_rerip"] = choice == "rerip"
            state["reuse_existing"] = choice == "reuse"
            state["stage"] = "accepted"
            db.execute(
                "UPDATE telegram_requests SET status=?, state_json=?, updated_at=? "
                "WHERE request_id=?",
                ("accepted", json.dumps(state), now(), request_id),
            )
        send(
            chat_id,
            (
                "♻️ Re-ripping this video's Shorts with your approval/rejection "
                "learning."
                if choice == "rerip"
                else "▶️ Reusing the existing reviewed Shorts and approved renders."
            ),
        )
        background_tasks.add_task(_process, request_id)
        return {"status": choice, "request_id": request_id}

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
                    "short_index": index,
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
                f"Short {index + 1} approved and queued for rendering. "
                f"Up to {RIPPED_SHORTS_RENDER_WORKERS} clips render at once; the rest wait.\n"
                f"{_render_progress_text(request_id)}",
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
            send(chat_id, f"Short {index + 1} rejected.")
        return {"status": verb, "request_id": request_id, "short_index": index}
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
    if parsed["source_kind"] == "youtube":
        try:
            prior_shorts = _reviewed_short_history_from_sheet(parsed["video_id"])
        except Exception:
            logger.exception("Could not check prior Shorts history before intake")
            prior_shorts = []
        if prior_shorts:
            state["stage"] = "awaiting_rerip_choice"
            state["prior_reviewed_short_count"] = len(prior_shorts)
            _save(request_id, "awaiting_rerip_choice", state)
            telegram(
                "sendMessage",
                {
                    "chat_id": chat_id,
                    "text": (
                        f"♻️ YouTube ID {parsed['video_id']} already has "
                        f"{len(prior_shorts)} reviewed Short(s).\n\n"
                        "Do you want to re-rip the Shorts using everything learned "
                        "from your approvals and rejections?"
                    ),
                    "reply_markup": {
                        "inline_keyboard": [[
                            {
                                "text": "♻️ Re-rip Shorts",
                                "callback_data": f"rs:rerip:{request_id}",
                            },
                            {
                                "text": "▶️ Use Existing",
                                "callback_data": f"rs:reuse:{request_id}",
                            },
                        ]]
                    },
                },
            )
            return {
                "status": "awaiting_rerip_choice",
                "request_id": request_id,
                "reviewed_shorts": len(prior_shorts),
            }
    send(
        chat_id,
        f"✅ Ripped Shorts request accepted\n"
        f"Source: {parsed['source_kind'].title()}\nJob ID: {request_id}",
    )
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
        send(
            chat_id,
            f"🎬 Short {index + 1} is now rendering.\n"
            f"{_render_progress_text(request_id)}",
        )
        candidate = state["result"]["segments"][index]
        user_id = str(
            (state.get("candidate_reviews") or {}).get(str(index), {}).get("user_id", "")
        )
        video = Path(state["video_path"])
        import main

        payload_candidate = dict(candidate)
        payload_candidate["candidate_number"] = int(
            candidate.get("candidate_number") or index + 1
        )
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
            f"✅ Short {index + 1} rendered and uploaded to DRIVE_FOLDER_ID:\n"
            f"{clip.get('clip_url', '')}\n"
            f"{_render_progress_text(request_id)}",
        )
        _notify_render_queue_complete(request_id, chat_id)
    except Exception as exc:
        logger.exception(
            "Short render failed request_id=%s candidate=%s", request_id, index + 1
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
        send(
            chat_id,
            f"❌ Short {index + 1} render failed:\n{str(exc)[:1500]}\n"
            f"{_render_progress_text(request_id)}",
        )
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
