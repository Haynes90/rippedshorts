"""Resolve Ripped Shorts brand/show routes before processing arbitrary YouTube links.

Ripped Shorts owns the Telegram intake, so unknown YouTube channels pause here and ask
for a Show Config route. The confirmed show_id is persisted on the Ripped Shorts request
and can be handed downstream to Schedule Master.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from typing import Any

from yt_dlp import YoutubeDL

import telegram_intake as ti
from google_sheets import get_rows

_ROUTE_CALLBACK_RE = re.compile(r"rs:route:([A-Za-z0-9-]+):([A-Za-z0-9_\-]+)")
_INSTALLED = False
_ORIGINAL_ACCEPT_UPDATE = None


def _show_config_rows() -> list[dict[str, Any]]:
    sheet_id = (
        os.getenv("RIPPED_SHORTS_LOG_SHEET_ID")
        or os.getenv("PODCAST_SHEET_ID")
        or ti.RIPPED_LOG_SHEET_ID
    ).strip()
    rows = get_rows(sheet_id, "Show Config", "A1:AF1000")
    active = []
    for row in rows:
        if str(row.get("active", "TRUE")).strip().upper() == "FALSE":
            continue
        show_id = str(row.get("show_id", "")).strip()
        if not show_id or show_id.upper() == "RIPPED":
            continue
        active.append(row)
    return active


def _route_from_row(row: dict[str, Any]) -> dict[str, str]:
    return {
        "show_id": str(row.get("show_id", "")).strip(),
        "show_name": str(row.get("show_name", "")).strip(),
        "youtube_channel_id": str(row.get("youtube_channel_id", "")).strip(),
        "workflow_type": str(row.get("workflow_type", "")).strip(),
    }


def _route_by_show_id(show_id: str) -> dict[str, str] | None:
    wanted = str(show_id or "").strip().upper()
    for row in _show_config_rows():
        if str(row.get("show_id", "")).strip().upper() == wanted:
            return _route_from_row(row)
    return None


def _youtube_channel_id(url: str) -> str:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "socket_timeout": 30,
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return str((info or {}).get("channel_id") or "").strip()


def _route_by_channel(channel_id: str) -> dict[str, str] | None:
    channel_id = str(channel_id or "").strip()
    if not channel_id:
        return None
    for row in _show_config_rows():
        if str(row.get("youtube_channel_id", "")).strip() == channel_id:
            return _route_from_row(row)
    return None


def _route_keyboard(request_id: str) -> dict[str, Any]:
    rows = []
    for row in _show_config_rows():
        route = _route_from_row(row)
        label = route["show_name"] or route["show_id"]
        rows.append([
            {
                "text": f"{route['show_id']} — {label}"[:60],
                "callback_data": f"rs:route:{request_id}:{route['show_id']}",
            }
        ])
    return {"inline_keyboard": rows}


def _ask_for_route(chat_id: str, request_id: str, parsed: dict[str, Any], channel_id: str) -> None:
    channel_line = channel_id or "unknown / not matched"
    ti.telegram(
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": (
                "🧭 I need the route/brand for this Ripped Shorts job before processing.\n\n"
                f"YouTube ID: {parsed.get('video_id', '')}\n"
                f"Channel ID: {channel_line}\n\n"
                "Choose the Show Config route below. I’ll save that route on this job and continue."
            ),
            "reply_markup": _route_keyboard(request_id),
        },
    )


def _resume_after_route(request_id: str, chat_id: str, state: dict[str, Any], background_tasks) -> dict[str, Any]:
    parsed = state.get("parsed") or {}
    try:
        prior_shorts = ti._reviewed_short_history_from_sheet(parsed.get("video_id", ""))
    except Exception:
        ti.logger.exception("Could not check prior Shorts history after route resolution")
        prior_shorts = []

    if prior_shorts:
        state["stage"] = "awaiting_rerip_choice"
        state["prior_reviewed_short_count"] = len(prior_shorts)
        ti._save(request_id, "awaiting_rerip_choice", state)
        ti.telegram(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": (
                    f"♻️ YouTube ID {parsed.get('video_id', '')} already has "
                    f"{len(prior_shorts)} reviewed Short(s).\n\n"
                    "Do you want to re-rip the Shorts using everything learned "
                    "from your approvals and rejections?"
                ),
                "reply_markup": {
                    "inline_keyboard": [[
                        {"text": "♻️ Re-rip Shorts", "callback_data": f"rs:rerip:{request_id}"},
                        {"text": "▶️ Use Existing", "callback_data": f"rs:reuse:{request_id}"},
                    ]]
                },
            },
        )
        return {
            "status": "awaiting_rerip_choice",
            "request_id": request_id,
            "reviewed_shorts": len(prior_shorts),
            "show_id": state.get("show_id", ""),
        }

    state["stage"] = "accepted"
    ti._save(request_id, "accepted", state)
    ti.send(
        chat_id,
        "✅ Ripped Shorts request accepted\n"
        f"Route: {state.get('show_id', '')} — {state.get('show_name', '')}\n"
        f"Source: {str(parsed.get('source_kind', '')).title()}\n"
        f"Job ID: {request_id}",
    )
    background_tasks.add_task(ti._process, request_id)
    return {
        "status": "accepted",
        "request_id": request_id,
        "show_id": state.get("show_id", ""),
    }


def _handle_route_callback(
    update: dict[str, Any], background_tasks, *, trusted_source: bool = False
) -> dict[str, Any] | None:
    callback = update.get("callback_query") or {}
    match = _ROUTE_CALLBACK_RE.fullmatch(str(callback.get("data") or ""))
    if not match:
        return None

    message = callback.get("message") or {}
    chat_id = str((message.get("chat") or {}).get("id", ""))
    user_id = str((callback.get("from") or {}).get("id", ""))
    if not chat_id or not user_id or (
        not trusted_source and not ti._authorized(chat_id, user_id)
    ):
        return {"status": "unauthorized"}

    request_id, show_id = match.groups()
    route = _route_by_show_id(show_id)
    if not route:
        ti.send(chat_id, f"That route ({show_id}) is not active in Show Config.")
        return {"status": "invalid_route", "request_id": request_id}

    with ti._LOCK, ti._telegram_db() as db:
        row = db.execute(
            "SELECT * FROM telegram_requests WHERE request_id=?", (request_id,)
        ).fetchone()
        if not row:
            return {"status": "not_found"}
        state = json.loads(row["state_json"])
        if str(state.get("stage") or "") != "awaiting_route":
            return {"status": "route_not_pending", "request_id": request_id}
        state.update({
            "show_id": route["show_id"],
            "show_name": route["show_name"],
            "route_status": "confirmed",
            "route_source": "telegram",
            "route_confirmed_at": ti.now(),
        })
        db.execute(
            "UPDATE telegram_requests SET status=?, state_json=?, updated_at=? WHERE request_id=?",
            ("accepted", json.dumps(state), ti.now(), request_id),
        )

    callback_id = str(callback.get("id") or "")
    if callback_id:
        try:
            ti.telegram("answerCallbackQuery", {"callback_query_id": callback_id, "text": f"Route: {route['show_id']}"})
        except Exception:
            ti.logger.exception("Could not acknowledge route callback")
    return _resume_after_route(request_id, chat_id, state, background_tasks)


def _handle_new_youtube(
    update: dict[str, Any], background_tasks, *, trusted_source: bool = False
) -> dict[str, Any] | None:
    callback = update.get("callback_query") or {}
    if callback:
        return None
    message = update.get("message") or update.get("edited_message") or {}
    text = str(message.get("text") or message.get("caption") or "").strip()
    if not text:
        return None
    try:
        parsed = ti.parse_request(text)
    except ValueError:
        return None
    if parsed.get("source_kind") != "youtube":
        return None

    chat_id = str((message.get("chat") or {}).get("id", ""))
    user_id = str((message.get("from") or {}).get("id", ""))
    if not chat_id or not user_id:
        return {"status": "ignored"}
    if not trusted_source and not ti._authorized(chat_id, user_id):
        return {"status": "unauthorized"}

    request_id = str(uuid.uuid4())
    update_id = str(update.get("update_id", ""))
    stamp = ti.now()
    channel_id = ""
    route = None
    route_error = ""
    try:
        channel_id = _youtube_channel_id(str(parsed.get("source_value") or ""))
        route = _route_by_channel(channel_id)
    except Exception as exc:
        route_error = str(exc)
        ti.logger.exception("YouTube route lookup failed; asking Telegram for route")

    state = {
        "stage": "accepted" if route else "awaiting_route",
        "parsed": parsed,
        "message_id": message.get("message_id"),
        "youtube_channel_id": channel_id,
        "route_status": "resolved" if route else "pending",
        "route_source": "youtube_channel_id" if route else "telegram_required",
        "route_lookup_error": route_error,
    }
    if route:
        state.update({
            "show_id": route["show_id"],
            "show_name": route["show_name"],
        })

    try:
        with ti._LOCK, ti._telegram_db() as db:
            db.execute(
                "INSERT INTO telegram_requests VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    request_id,
                    update_id,
                    chat_id,
                    user_id,
                    "accepted" if route else "awaiting_route",
                    parsed["mode"],
                    parsed["source_kind"],
                    parsed["source_value"],
                    json.dumps(state),
                    stamp,
                    stamp,
                ),
            )
    except sqlite3.IntegrityError:
        return {"status": "duplicate_update"}

    if not route:
        _ask_for_route(chat_id, request_id, parsed, channel_id)
        return {
            "status": "awaiting_route",
            "request_id": request_id,
            "youtube_channel_id": channel_id,
        }

    return _resume_after_route(request_id, chat_id, state, background_tasks)


def gated_accept_update(
    update: dict[str, Any], background_tasks, *, trusted_source: bool = False
):
    routed = _handle_route_callback(
        update, background_tasks, trusted_source=trusted_source
    )
    if routed is not None:
        return routed
    youtube = _handle_new_youtube(
        update, background_tasks, trusted_source=trusted_source
    )
    if youtube is not None:
        return youtube
    return _ORIGINAL_ACCEPT_UPDATE(
        update, background_tasks, trusted_source=trusted_source
    )


def install_route_gate() -> None:
    global _INSTALLED, _ORIGINAL_ACCEPT_UPDATE
    if _INSTALLED:
        return
    _ORIGINAL_ACCEPT_UPDATE = ti._accept_update
    ti._accept_update = gated_accept_update
    _INSTALLED = True
