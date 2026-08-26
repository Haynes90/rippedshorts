"""Universal Clip Master chapter review through the shared Telegram bot."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

import requests
from fastapi import APIRouter, Header, HTTPException, Request

from audio_master_handoff import connect
from telegram_quick_edits import OPTIONS_TEXT, apply_quick_command, is_quick_command, sections_text

router = APIRouter()
logger = logging.getLogger("clip-master.review")
_LOCK = threading.RLock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db() -> sqlite3.Connection:
    db = connect()
    db.execute(
        "CREATE TABLE IF NOT EXISTS clipmaster_reviews ("
        "review_id TEXT PRIMARY KEY, origin_job_key TEXT UNIQUE NOT NULL, "
        "chat_id TEXT NOT NULL, user_id TEXT NOT NULL, status TEXT NOT NULL, "
        "state_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    return db


def _secret() -> str:
    return (
        os.getenv("CLIP_MASTER_REVIEW_SECRET")
        or os.getenv("RIPPED_SHORTS_SHARED_SECRET")
        or ""
    ).strip()


def _authorized(secret: str | None) -> bool:
    expected = _secret()
    return bool(expected and secret == expected)


def _telegram(method: str, payload: dict[str, Any]) -> dict[str, Any]:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured")
    response = requests.post(
        f"https://api.telegram.org/bot{token}/{method}",
        json=payload,
        timeout=(10, 60),
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Telegram {method} failed ({response.status_code}): {response.text[:1000]}"
        )
    return response.json()


def _send(chat_id: str, text: str, reply_markup: dict | None = None) -> None:
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup
    _telegram("sendMessage", payload)


def _normalize_sections(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list) or not items:
        raise ValueError("sections must contain at least one section")
    sections = []
    for index, raw in enumerate(items, 1):
        if not isinstance(raw, dict):
            raise ValueError(f"section {index} must be an object")
        start = float(raw.get("start", raw.get("start_seconds", 0)) or 0)
        end_value = raw.get("end", raw.get("end_seconds"))
        if end_value is None:
            end_value = start + float(raw.get("duration", 0) or 0)
        end = float(end_value)
        if end <= start:
            raise ValueError(f"section {index} must end after it starts")
        sections.append(
            {
                **raw,
                "start": start,
                "end": end,
                "duration": end - start,
                "title": str(
                    raw.get("title")
                    or raw.get("display_title")
                    or raw.get("chapter_type")
                    or f"Section {index}"
                ),
                "review_status": str(raw.get("review_status") or "pending"),
            }
        )
    return sorted(sections, key=lambda item: float(item["start"]))


def _save(review_id: str, status: str, state: dict[str, Any]) -> None:
    with _LOCK, _db() as db:
        db.execute(
            "UPDATE clipmaster_reviews SET status=?, state_json=?, updated_at=? "
            "WHERE review_id=?",
            (status, json.dumps(state), now(), review_id),
        )


def _get(review_id: str) -> sqlite3.Row | None:
    with _LOCK, _db() as db:
        return db.execute(
            "SELECT * FROM clipmaster_reviews WHERE review_id=?", (review_id,)
        ).fetchone()


def _latest(chat_id: str, user_id: str) -> sqlite3.Row | None:
    with _LOCK, _db() as db:
        if user_id:
            row = db.execute(
                "SELECT * FROM clipmaster_reviews WHERE chat_id=? "
                "AND (user_id='' OR user_id=?) AND status='awaiting_review' "
                "ORDER BY updated_at DESC LIMIT 1",
                (chat_id, user_id),
            ).fetchone()
        else:
            row = db.execute(
                "SELECT * FROM clipmaster_reviews WHERE chat_id=? "
                "AND status='awaiting_review' ORDER BY updated_at DESC LIMIT 1",
                (chat_id,),
            ).fetchone()
    return row


def claims_update(update: dict[str, Any]) -> bool:
    callback = update.get("callback_query") or {}
    if str(callback.get("data") or "").startswith("cm:"):
        return True
    message = callback.get("message") or update.get("message") or {}
    chat_id = str((message.get("chat") or {}).get("id", ""))
    user_id = str((callback.get("from") or message.get("from") or {}).get("id", ""))
    text = str(message.get("text") or message.get("caption") or "").strip()
    return bool(chat_id and is_quick_command(text) and _latest(chat_id, user_id))


def _send_review(state: dict[str, Any]) -> None:
    chat_id = state["chat_id"]
    review_id = state["review_id"]
    _send(
        chat_id,
        "📋 Clip Master chapter review\n"
        f"Source: {state.get('source_service', 'unknown')}\n"
        f"Job: {state.get('origin_job_id', '')}\n\n"
        + sections_text(state),
        {
            "inline_keyboard": [
                [
                    {
                        "text": "✅ Approve All",
                        "callback_data": f"cm:approve_all:{review_id}",
                    },
                    {
                        "text": "✏️ Change / Add",
                        "callback_data": f"cm:options:{review_id}",
                    },
                ]
            ]
        },
    )
    for index, section in enumerate(state["service_segments"]):
        _send(
            chat_id,
            f"{index + 1}. {section.get('title', f'Section {index + 1}')}\n"
            f"{int(section['start'])}s–{int(section['end'])}s",
            {
                "inline_keyboard": [
                    [
                        {
                            "text": "✅ Keep",
                            "callback_data": f"cm:keep:{review_id}:{index}",
                        },
                        {
                            "text": "❌ Remove",
                            "callback_data": f"cm:remove:{review_id}:{index}",
                        },
                    ],
                    [
                        {
                            "text": "✏️ Options",
                            "callback_data": f"cm:options:{review_id}",
                        }
                    ],
                ]
            },
        )


def _log_learning(
    state: dict[str, Any],
    action: str,
    instruction: str,
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    user_id: str,
) -> None:
    try:
        import main

        first_before, last_before = (before[0] if before else {}), (
            before[-1] if before else {}
        )
        first_after, last_after = (after[0] if after else {}), (
            after[-1] if after else {}
        )
        stamp = now()
        ai_start = first_before.get("start", "")
        ai_end = last_before.get("end", "")
        final_start = first_after.get("start", "")
        final_end = last_after.get("end", "")
        values = [
            f"{state['review_id']}:{stamp}",
            state.get("origin_job_id", ""),
            state.get("video_id", ""),
            "service",
            ai_start,
            ai_end,
            final_start,
            final_end,
            (float(final_start) - float(ai_start))
            if ai_start != "" and final_start != ""
            else "",
            (float(final_end) - float(ai_end))
            if ai_end != "" and final_end != ""
            else "",
            "",
            "",
            instruction,
            stamp,
            user_id,
            "yes",
            "", "", "", "",
            action,
            instruction,
            json.dumps(before, separators=(",", ":"))[:20000],
            json.dumps(after, separators=(",", ":"))[:20000],
            stamp,
            user_id,
            "telegram",
            "yes",
            "yes",
            "v2",
        ]
        sheet_id = (
            os.getenv("PODCAST_SHEET_ID")
            or os.getenv("RIPPED_SHORTS_LOG_SHEET_ID")
            or "14VruBxjaaE9DyPSdBidMeuPHew3nHUC5sRNGXtLCsis"
        )
        _, _, sheets = main.get_google_services()
        sheets.spreadsheets().values().append(
            spreadsheetId=sheet_id,
            range="'Decision Log'!A:AD",
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": [values]},
        ).execute()
    except Exception:
        logger.exception("Could not log Clip Master review learning")


def _deliver(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("delivered_at"):
        return {"status": "already_delivered", "review_id": state["review_id"]}
    callback_url = str(state.get("callback_url") or "").strip()
    payload = {
        "event": "clipmaster_review_completed",
        "review_id": state["review_id"],
        "origin_job_id": state.get("origin_job_id"),
        "source_service": state.get("source_service"),
        "video_id": state.get("video_id"),
        "sections": state["service_segments"],
        "approved_at": now(),
    }
    if callback_url:
        headers = {"Content-Type": "application/json"}
        callback_secret = str(state.get("callback_secret") or "").strip()
        if callback_secret:
            headers["x-clip-master-callback-secret"] = callback_secret
        response = requests.post(
            callback_url, json=payload, headers=headers, timeout=(10, 60)
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Origin callback failed ({response.status_code}): {response.text[:1000]}"
            )
    state["delivered_at"] = now()
    state["delivery_payload"] = payload
    _save(state["review_id"], "approved", state)
    return {"status": "approved", "review_id": state["review_id"], "payload": payload}


def handle_update(update: dict[str, Any]) -> dict[str, Any]:
    callback = update.get("callback_query") or {}
    message = callback.get("message") or update.get("message") or {}
    chat_id = str((message.get("chat") or {}).get("id", ""))
    user_id = str((callback.get("from") or message.get("from") or {}).get("id", ""))
    callback_data = str(callback.get("data") or "")
    text = str(message.get("text") or message.get("caption") or "").strip()

    match = __import__("re").fullmatch(
        r"cm:(options|approve_all|keep|remove):([A-Za-z0-9-]+)(?::(\d+))?",
        callback_data,
    )
    row = _get(match.group(2)) if match else _latest(chat_id, user_id)
    if not row:
        return {"status": "no_pending_clipmaster_review"}
    state = json.loads(row["state_json"])
    if not state.get("user_id") and user_id:
        state["user_id"] = user_id

    if match:
        action, _, index_text = match.groups()
        if action == "options":
            _send(chat_id, OPTIONS_TEXT)
            return {"status": "options", "review_id": state["review_id"]}
        if action == "approve_all":
            before = json.loads(json.dumps(state["service_segments"]))
            for section in state["service_segments"]:
                section["review_status"] = "approved"
            _log_learning(
                state, "approve_all", "Approve all", before,
                state["service_segments"], user_id,
            )
            result = _deliver(state)
            _send(chat_id, "✅ Chapters approved and returned to the originating service.")
            return result
        index = int(index_text or -1)
        if index < 0 or index >= len(state["service_segments"]):
            return {"status": "section_not_found"}
        before = json.loads(json.dumps(state["service_segments"]))
        if action == "keep":
            state["service_segments"][index]["review_status"] = "approved"
            instruction = f"Keep {index + 1}"
        else:
            state["service_segments"].pop(index)
            instruction = f"Remove {index + 1}"
        state.setdefault("quick_edit_history", []).append(
            {"action": action, "segments": before}
        )
        state["quick_edit_history"] = state["quick_edit_history"][-20:]
        _save(state["review_id"], "awaiting_review", state)
        _log_learning(
            state, action, instruction, before, state["service_segments"], user_id
        )
        _send(chat_id, "✅ Updated.\n\n" + sections_text(state))
        return {"status": action, "review_id": state["review_id"]}

    try:
        result = apply_quick_command(state, text)
    except ValueError as exc:
        _send(chat_id, str(exc))
        return {"status": "invalid_command", "detail": str(exc)}
    if result.get("changed"):
        state = result["state"]
        state["user_id"] = user_id
        _save(state["review_id"], "awaiting_review", state)
        _log_learning(
            state,
            result.get("action", ""),
            text,
            result.get("before") or [],
            result.get("after") or [],
            user_id,
        )
        if result.get("action") == "approve_all":
            delivered = _deliver(state)
            _send(chat_id, "✅ Chapters approved and returned to the originating service.")
            return delivered
    _send(chat_id, result["message"])
    return {
        "status": "updated" if result.get("changed") else "help",
        "review_id": state["review_id"],
        "action": result.get("action"),
    }


@router.post("/api/clip-master/reviews", status_code=202)
async def create_review(
    request: Request,
    x_clip_master_review_secret: str | None = Header(None),
) -> dict[str, Any]:
    if not _authorized(x_clip_master_review_secret):
        raise HTTPException(status_code=401, detail="Invalid Clip Master review secret")
    payload = await request.json()
    origin_job_id = str(
        payload.get("origin_job_id") or payload.get("job_id") or ""
    ).strip()
    source_service = str(payload.get("source_service") or "unknown").strip()
    if not origin_job_id:
        raise HTTPException(status_code=422, detail="origin_job_id is required")
    try:
        sections = _normalize_sections(
            payload.get("sections") or payload.get("chapters")
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    origin_key = f"{source_service}:{origin_job_id}"
    with _LOCK, _db() as db:
        existing = db.execute(
            "SELECT * FROM clipmaster_reviews WHERE origin_job_key=?", (origin_key,)
        ).fetchone()
        if existing:
            return {
                "status": "duplicate",
                "review_id": existing["review_id"],
                "review_status": existing["status"],
            }
        review_id = str(uuid.uuid4())
        chat_id = str(
            payload.get("chat_id") or os.getenv("TELEGRAM_CHAT_ID") or ""
        ).strip()
        if not chat_id:
            raise HTTPException(
                status_code=503, detail="TELEGRAM_CHAT_ID is not configured"
            )
        stamp = now()
        state = {
            "review_id": review_id,
            "origin_job_id": origin_job_id,
            "source_service": source_service,
            "video_id": str(payload.get("video_id") or ""),
            "callback_url": str(payload.get("callback_url") or ""),
            "callback_secret": str(payload.get("callback_secret") or ""),
            "chat_id": chat_id,
            "user_id": str(payload.get("user_id") or ""),
            "service_segments": sections,
            "stage": "awaiting_review",
        }
        db.execute(
            "INSERT INTO clipmaster_reviews VALUES (?,?,?,?,?,?,?,?)",
            (
                review_id,
                origin_key,
                chat_id,
                state["user_id"],
                "awaiting_review",
                json.dumps(state),
                stamp,
                stamp,
            ),
        )
    _send_review(state)
    return {
        "status": "awaiting_review",
        "review_id": review_id,
        "origin_job_id": origin_job_id,
    }


@router.get("/api/clip-master/reviews/{review_id}")
def review_status(
    review_id: str,
    x_clip_master_review_secret: str | None = Header(None),
) -> dict[str, Any]:
    if not _authorized(x_clip_master_review_secret):
        raise HTTPException(status_code=401, detail="Invalid Clip Master review secret")
    row = _get(review_id)
    if not row:
        raise HTTPException(status_code=404, detail="Review not found")
    state = json.loads(row["state_json"])
    state.pop("callback_secret", None)
    return {"status": row["status"], **state}
