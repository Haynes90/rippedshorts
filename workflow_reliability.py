"""Durable workflow ledger, readiness checks, and shared failure taxonomy."""
from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("ripped-shorts.reliability")

WORKFLOW_JOBS_TAB = os.getenv("WORKFLOW_JOBS_TAB", "Workflow Jobs").strip()
HEADERS = [
    "job_id", "source_url", "video_id", "show_code", "mode",
    "current_owner", "current_stage", "last_successful_stage", "next_action",
    "status", "attempt_count", "heartbeat_at", "error_class",
    "error_message", "chat_id", "user_id", "updated_at",
]
_LOCK = threading.RLock()
_READY_STAGES = {
    "source_resolution", "source_ready", "transcript_ready", "awaiting_review",
    "processing_16_9", "awaiting_render_completion", "copy_review",
    "schedule_handoff", "scheduled", "publishing", "published",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_error(error: Any) -> str:
    text = str(error or "").lower()
    if any(x in text for x in ("telegram", "too many requests", "retry_after", "webhook")):
        return "TELEGRAM_DELIVERY"
    if any(x in text for x in ("401", "403", "unauthorized", "forbidden", "token", "credential")):
        return "AUTHENTICATION"
    if any(x in text for x in ("youtube", "yt-dlp", "download", "source video")):
        return "SOURCE_ACQUISITION"
    if any(x in text for x in ("transcript", "whisper", "speech")):
        return "TRANSCRIPTION"
    if any(x in text for x in ("ffmpeg", "render", "moviepy")):
        return "RENDER"
    if any(x in text for x in ("drive", "upload")):
        return "DRIVE_UPLOAD"
    if any(x in text for x in ("schedule", "handoff", "intake")):
        return "HANDOFF"
    if any(x in text for x in ("meta", "instagram", "facebook", "publish", "youtube api")):
        return "PUBLISHING"
    if any(x in text for x in ("missing", "not configured", "environment")):
        return "CONFIGURATION"
    return "PROCESSING"


def next_action_for(stage: str, status: str) -> str:
    if status == "error":
        return "retry_or_user_action"
    return {
        "accepted": "resolve_route_and_source",
        "source_resolution": "prepare_source",
        "source_ready": "prepare_transcript",
        "transcript_ready": "select_candidates",
        "awaiting_review": "human_review",
        "processing_16_9": "select_16_9_highlights",
        "awaiting_render_completion": "finish_renders",
        "copy_review": "approve_copy",
        "schedule_handoff": "schedule_assets",
        "scheduled": "publish_when_due",
        "publishing": "record_receipts",
        "published": "complete",
    }.get(stage, "continue")


def _services():
    import main
    return main.get_google_services()


def _ensure_tab(sheets: Any, spreadsheet_id: str) -> None:
    metadata = sheets.spreadsheets().get(
        spreadsheetId=spreadsheet_id,
        fields="sheets.properties.title",
    ).execute()
    titles = {
        str(item.get("properties", {}).get("title", ""))
        for item in metadata.get("sheets", [])
    }
    if WORKFLOW_JOBS_TAB not in titles:
        sheets.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"requests": [{"addSheet": {"properties": {"title": WORKFLOW_JOBS_TAB}}}]},
        ).execute()
    values = sheets.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=f"'{WORKFLOW_JOBS_TAB}'!A1:Q1",
    ).execute().get("values", [])
    if not values:
        sheets.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=f"'{WORKFLOW_JOBS_TAB}'!A1:Q1",
            valueInputOption="RAW",
            body={"values": [HEADERS]},
        ).execute()


def upsert_job(spreadsheet_id: str, request_id: str, status: str, state: dict[str, Any],
               chat_id: str = "", user_id: str = "") -> None:
    parsed = state.get("parsed") or {}
    stage = str(state.get("stage") or status or "unknown")
    error = str(state.get("error") or state.get("notification_error") or "")
    row = [
        request_id,
        str(parsed.get("source_value") or ""),
        str(parsed.get("video_id") or ""),
        str(state.get("show_id") or state.get("show_code") or ""),
        str(parsed.get("mode") or state.get("mode") or ""),
        "RIPPED_SHORTS",
        stage,
        str(state.get("last_successful_stage") or (stage if stage in _READY_STAGES else "")),
        next_action_for(stage, status),
        status,
        int(state.get("attempt_count") or 0),
        utc_now(),
        classify_error(error) if error else "",
        error[:2000],
        chat_id,
        user_id,
        utc_now(),
    ]
    try:
        with _LOCK:
            _, _, sheets = _services()
            _ensure_tab(sheets, spreadsheet_id)
            result = sheets.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range=f"'{WORKFLOW_JOBS_TAB}'!A:Q",
            ).execute()
            rows = result.get("values", [])
            target = next(
                (number for number, values in enumerate(rows[1:], 2)
                 if values and str(values[0]) == request_id),
                None,
            )
            if target:
                sheets.spreadsheets().values().update(
                    spreadsheetId=spreadsheet_id,
                    range=f"'{WORKFLOW_JOBS_TAB}'!A{target}:Q{target}",
                    valueInputOption="RAW",
                    body={"values": [row]},
                ).execute()
            else:
                sheets.spreadsheets().values().append(
                    spreadsheetId=spreadsheet_id,
                    range=f"'{WORKFLOW_JOBS_TAB}'!A:Q",
                    valueInputOption="RAW",
                    insertDataOption="INSERT_ROWS",
                    body={"values": [row]},
                ).execute()
    except Exception:
        logger.exception("WORKFLOW_LEDGER_UPSERT_FAILED job_id=%s stage=%s", request_id, stage)


def latest_incomplete(spreadsheet_id: str, chat_id: str = "", user_id: str = "") -> dict[str, str] | None:
    _, _, sheets = _services()
    _ensure_tab(sheets, spreadsheet_id)
    rows = sheets.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=f"'{WORKFLOW_JOBS_TAB}'!A:Q",
    ).execute().get("values", [])
    records = []
    for values in rows[1:]:
        padded = list(values) + [""] * (17 - len(values))
        if padded[9].lower() in {"published", "complete", "cancelled"}:
            continue
        if chat_id and padded[14] and padded[14] != chat_id:
            continue
        if user_id and padded[15] and padded[15] != user_id:
            continue
        records.append(dict(zip(HEADERS, padded[:17])))
    return records[-1] if records else None


def readiness_snapshot(db_path: Path, public_url: str, token_set: bool,
                       group_id_set: bool, scheduler_url: str) -> dict[str, Any]:
    data_dir = Path(os.getenv("DATA_DIR", str(db_path.parent))).resolve()
    persistent = str(data_dir).startswith("/data") or bool(os.getenv("RAILWAY_VOLUME_MOUNT_PATH"))
    db_ok = False
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as db:
            db.execute("SELECT 1").fetchone()
        db_ok = True
    except Exception:
        logger.exception("READINESS_DATABASE_FAILED")
    checks = {
        "database": db_ok,
        "persistent_storage": persistent,
        "telegram_token": token_set,
        "telegram_group": group_id_set,
        "public_url": bool(public_url),
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "google_credentials": bool(
            os.getenv("GOOGLE_CREDENTIALS")
            or (os.getenv("GOOGLE_CLIENT_EMAIL") and os.getenv("GOOGLE_PRIVATE_KEY_KEY"))
            or (os.getenv("GOOGLE_CLIENT_EMAIL") and os.getenv("GOOGLE_PRIVATE_KEY"))
        ),
        "schedule_master": bool(scheduler_url),
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "checked_at": utc_now(),
    }
