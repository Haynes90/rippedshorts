"""Telegram-triggered intake for reusable Drive sources and arbitrary YouTube videos."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import requests
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request, Response

from audio_master_handoff import DB_PATH, SOURCE_DIR, connect, download_drive, drive_metadata, get_job
from source_ingestion import (
    ingest_with_audio_master,
    restrict_to_boundary,
    reuse_from_drive,
    select_non_overlapping,
)
from google_drive import read_google_doc_text
from google_sheets import get_rows
from telegram_quick_edits import OPTIONS_TEXT, apply_quick_command, is_quick_command
from clipmaster_review import (
    claims_update as clipmaster_claims_update,
    handle_update as handle_clipmaster_update,
    router as clipmaster_review_router,
)
from workflow_reliability import (
    classify_error,
    latest_incomplete,
    readiness_snapshot,
    upsert_job,
)

router = APIRouter()
router.include_router(clipmaster_review_router)

YOUTUBE_RE = re.compile(
    r"https?://(?:(?:www\.|m\.)?youtube\.com/"
    r"(?:watch\?[^\s]*v=|shorts/|live/|embed/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{6,20})",
    re.I,
)
DRIVE_RE = re.compile(r"https?://drive\.google\.com/(?:file/d/|open\?id=|uc\?(?:[^\s]*&)?id=)([A-Za-z0-9_-]+)", re.I)
_LOCK = threading.RLock()
logger = logging.getLogger("ripped-shorts.telegram")
RIPPED_SHORTS_RENDER_WORKERS = max(
    1, int(os.getenv("RIPPED_SHORTS_RENDER_WORKERS", "3"))
)
RENDER_EXECUTOR = ThreadPoolExecutor(max_workers=RIPPED_SHORTS_RENDER_WORKERS)
STATUS_EXECUTOR = ThreadPoolExecutor(max_workers=1)
_STATUS_CARD_LAST: dict[str, float] = {}
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
    """Wait for every approved 9:16 and 16:9 render before scheduling."""
    summary = None
    with _LOCK, _telegram_db() as db:
        row = db.execute(
            "SELECT * FROM telegram_requests WHERE request_id=?", (request_id,)
        ).fetchone()
        if not row:
            return
        state = json.loads(row["state_json"])
        short_reviews = dict(state.get("candidate_reviews") or {})
        topic_reviews = dict(state.get("topic_reviews") or {})
        tracked = {"queued", "rendering", "rendered", "render_failed"}
        statuses = {
            f"short:{index}": str(review.get("status") or "")
            for index, review in short_reviews.items()
            if str(review.get("status") or "") in tracked
        }
        statuses.update(
            {
                f"highlight:{index}": str(review.get("status") or "")
                for index, review in topic_reviews.items()
                if str(review.get("status") or "") in tracked
            }
        )
        active = sum(status in {"queued", "rendering"} for status in statuses.values())
        rendered = sum(status == "rendered" for status in statuses.values())
        failed = sum(status == "render_failed" for status in statuses.values())
        # Schedule Now closes selection, but handoff is gated until every approved
        # 9:16 Short and 16:9 highlight has left the queued/rendering states.
        if active:
            return
        if not statuses and not state.get("schedule_requested_at"):
            return
        signature = json.dumps(statuses, sort_keys=True)
        if state.get("render_queue_completion_signature") == signature:
            if state.get("schedule_requested_at"):
                RENDER_EXECUTOR.submit(
                    _handoff_shorts_to_schedule_master, request_id, chat_id
                )
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
        with _LOCK, _telegram_db() as db:
            latest = db.execute(
                "SELECT state_json FROM telegram_requests WHERE request_id=?",
                (request_id,),
            ).fetchone()
        latest_state = json.loads(latest["state_json"]) if latest else {}
        if latest_state.get("schedule_requested_at"):
            RENDER_EXECUTOR.submit(
                _handoff_shorts_to_schedule_master, request_id, chat_id
            )


def _copy_review_assets(state: dict[str, Any], request_id: str) -> list[dict[str, Any]]:
    """Build metadata-only records for every approved render, including active ones."""
    assets = []
    reviews = dict(state.get("candidate_reviews") or {})
    clips = (state.get("result") or {}).get("segments", [])
    for index, clip in enumerate(clips):
        status = str((reviews.get(str(index)) or {}).get("status") or "")
        if status not in {"queued", "rendering", "rendered"}:
            continue
        number = int(clip.get("candidate_number") or index + 1)
        assets.append({
            "asset_id": f"{request_id}:short:{number}",
            "asset_type": "9:16_SHORT",
            "candidate_number": number,
            "transcript": str(clip.get("transcript") or ""),
            "title": "",
        })
    topic_reviews = dict(state.get("topic_reviews") or {})
    topics = (state.get("topic_result") or {}).get("segments", [])
    for index, segment in enumerate(topics):
        status = str((topic_reviews.get(str(index)) or {}).get("status") or "")
        if status not in {"queued", "rendering", "rendered"}:
            continue
        number = index + 1
        assets.append({
            "asset_id": f"{request_id}:highlight:{number}",
            "asset_type": "16:9_HIGHLIGHT",
            "candidate_number": number,
            "transcript": str(segment.get("transcript") or ""),
            "title": str(segment.get("title") or ""),
        })
    return assets


def _send_copy_review(chat_id: str, request_id: str, drafts: list[dict[str, Any]]) -> None:
    for index, draft in enumerate(drafts):
        if draft["asset_type"] == "9:16_SHORT":
            text = (
                f"✍️ 9:16 Caption Draft {draft.get('candidate_number', index + 1)}\n\n"
                f"{draft.get('social_caption', '')}\n\n{draft.get('hashtags', '')}"
            )
            buttons = [
                [
                    {
                        "text": "✅ Keep Caption",
                        "callback_data": f"rs:copy_keep:{request_id}:{index}",
                    },
                    {
                        "text": "✏️ Write My Caption",
                        "callback_data": f"rs:copy_edit_caption:{request_id}:{index}",
                    },
                ],
                [{
                    "text": "🔄 Generate Another",
                    "callback_data": f"rs:copy_regenerate:{request_id}:{index}",
                }],
            ]
        else:
            text = (
                f"📺 16:9 Metadata Draft {draft.get('candidate_number', index + 1)}\n\n"
                f"TITLE\n{draft.get('video_title', '')}\n\n"
                f"DESCRIPTION\n{draft.get('video_description', '')}\n\n"
                f"SEO TAGS\n{draft.get('hashtags', '')}"
            )
            buttons = [
                [{
                    "text": "✅ Keep Title & Description",
                    "callback_data": f"rs:copy_keep:{request_id}:{index}",
                }],
                [
                    {
                        "text": "✏️ Edit Title",
                        "callback_data": f"rs:copy_edit_title:{request_id}:{index}",
                    },
                    {
                        "text": "✏️ Edit Description",
                        "callback_data": f"rs:copy_edit_description:{request_id}:{index}",
                    },
                ],
                [{
                    "text": "🔄 Generate Another",
                    "callback_data": f"rs:copy_regenerate:{request_id}:{index}",
                }],
            ]
        telegram("sendMessage", {
            "chat_id": chat_id,
            "text": text[:4000],
            "disable_web_page_preview": True,
            "reply_markup": {"inline_keyboard": buttons},
        })
    telegram("sendMessage", {
        "chat_id": chat_id,
        "text": (
            "Review the drafts above. Edit anything you want; unchanged drafts are "
            "treated as approved. When finished, release the complete batch."
        ),
        "reply_markup": {"inline_keyboard": [[{
            "text": "✅ Approve Copy & Schedule",
            "callback_data": f"rs:copy_finish:{request_id}",
        }]]},
    })


def _regenerate_copy_draft(request_id: str, index: int, chat_id: str) -> None:
    """Regenerate one draft without blocking Telegram's webhook response."""
    try:
        with _LOCK, _telegram_db() as db:
            row = db.execute(
                "SELECT * FROM telegram_requests WHERE request_id=?", (request_id,)
            ).fetchone()
        if not row:
            return
        state = json.loads(row["state_json"])
        drafts = list(state.get("copy_drafts") or [])
        if index >= len(drafts):
            return
        fresh = _generate_schedule_copy(
            [drafts[index]],
            str(state.get("show_id") or ""),
            _state_vid_title(state),
            str((state.get("parsed") or {}).get("source_value") or ""),
        )[0]
        fresh["copy_status"] = "pending"
        fresh["regenerated_at"] = now()
        fresh["regeneration_count"] = int(drafts[index].get("regeneration_count") or 0) + 1
        drafts[index] = fresh
        state["copy_drafts"] = drafts
        with _LOCK, _telegram_db() as db:
            db.execute(
                "UPDATE telegram_requests SET state_json=?, updated_at=? WHERE request_id=?",
                (json.dumps(state), now(), request_id),
            )
        send(chat_id, f"✅ Draft {index + 1} was regenerated. Here is the refreshed review set:")
        _send_copy_review(chat_id, request_id, drafts)
    except Exception:
        logger.exception("Could not regenerate copy request_id=%s index=%s", request_id, index)
        send(chat_id, f"❌ I couldn't regenerate draft {index + 1}. Your previous draft is still saved.")


def _log_copy_learning(
    request_id: str, state: dict[str, Any], drafts: list[dict[str, Any]], user_id: str
) -> None:
    """Persist AI-versus-final wording so later generations learn the user's style."""
    try:
        import main
        _, _, sheets = main.get_google_services()
        source_title = _state_vid_title(state)
        brand_id = str(state.get("show_id") or "")
        values = []
        for draft in drafts:
            edited = bool(draft.get("user_edited"))
            values.append([
                now(),
                request_id,
                str(draft.get("asset_id") or ""),
                brand_id,
                str(draft.get("asset_type") or ""),
                source_title,
                str(draft.get("transcript") or ""),
                str(draft.get("ai_social_caption") or draft.get("social_caption") or ""),
                str(draft.get("social_caption") or ""),
                str(draft.get("ai_video_title") or draft.get("video_title") or ""),
                str(draft.get("video_title") or ""),
                str(draft.get("ai_video_description") or draft.get("video_description") or ""),
                str(draft.get("video_description") or ""),
                str(draft.get("hashtags") or ""),
                "EDITED" if edited else "ACCEPTED",
                user_id,
            ])
        if values:
            sheets.spreadsheets().values().append(
                spreadsheetId=RIPPED_LOG_SHEET_ID,
                range="'Caption Learning'!A:P",
                valueInputOption="RAW",
                insertDataOption="INSERT_ROWS",
                body={"values": values},
            ).execute()
    except Exception:
        logger.exception("Could not record Caption Learning rows request_id=%s", request_id)


def _generate_schedule_copy(
    assets: list[dict[str, Any]],
    show_id: str,
    source_title: str,
    source_url: str = "",
) -> list[dict[str, Any]]:
    """Create ready-to-schedule social copy with mandatory source attribution."""
    brand = str(show_id or "TCB").strip().upper()
    source_title = str(source_title or "").strip()
    source_url = str(source_url or "").strip()

    guide_doc = (
        os.getenv("SOCIAL_COPY_GUIDE_DOC_ID")
        or "1UrndyfvrbZbYP2s7tqhBE5jV72oymboMyoSugnbV-Ug"
    ).strip()
    guide_text = ""
    guide_rules: dict[str, str] = {}
    try:
        guide_text = read_google_doc_text(guide_doc).strip()
        for line in guide_text.splitlines():
            key, separator, value = line.partition(":")
            normalized_key = key.strip().upper()
            if separator and normalized_key in {
                "ATTRIBUTION_FORMAT",
                "SOURCE_LINK_LABEL",
                "SOURCE_CHANNEL_LABEL",
                "TCB_CTA",
                "TCB_ABOUT",
                "TCB_WEBSITE",
                "TCB_INSTAGRAM",
                "TCB_YOUTUBE",
                "TCB_FACEBOOK",
                "TCB_TIKTOK",
                "TCB_EMAIL",
                "TCB_HASHTAGS",
                "TDOG_CTA",
                "TDOG_ABOUT",
                "TDOG_HOST_ANDRE",
                "TDOG_HOST_GRAY",
                "TDOG_HOST_TCB",
                "TDOG_SPONSOR_RETAIN",
                "TDOG_SPONSOR_SONIAS",
                "TDOG_SPONSOR_ANDRES",
                "TDOG_TELEGRAM",
                "TDOG_EMAIL",
                "TDOG_HASHTAGS",
                "AGAPE_CTA",
                "AGAPE_LEADERSHIP",
                "AGAPE_ABOUT",
                "AGAPE_WEDNESDAY",
                "AGAPE_YOUTUBE_URL",
                "AGAPE_FACEBOOK_URL",
                "AGAPE_PASTOR_URL",
                "AGAPE_HASHTAGS",
            }:
                guide_rules[normalized_key] = value.strip()
    except Exception:
        logger.exception("Could not load Social Copy & CTA Guide; using built-in defaults")

    source_show = ""
    source_channel_url = ""
    if source_url and YOUTUBE_RE.search(source_url):
        try:
            oembed = requests.get(
                "https://www.youtube.com/oembed",
                params={"url": source_url, "format": "json"},
                timeout=(5, 20),
            )
            oembed.raise_for_status()
            oembed_data = oembed.json()
            source_show = str(oembed_data.get("author_name") or "").strip()
            source_channel_url = str(oembed_data.get("author_url") or "").strip()
        except Exception:
            logger.warning("Could not resolve source YouTube channel for copy attribution")
    source_show = source_show or "the source channel"

    if brand in {"TDOG", "THE_DOG"}:
        brand_rule = (
            "The Dirt on Gardening: identify the useful gardening idea, reference "
            "the hosts/guest when the transcript supports it, tag known handles only, "
            "and invite viewers to watch or follow The Dirt on Gardening."
        )
    elif brand.startswith("AGAPE"):
        brand_rule = (
            "Agape: Christianity, motivation, and inspiration. Use an inviting CTA "
            "such as See us Sunday or Join us live online when appropriate."
        )
    else:
        brand_rule = (
            "The Chocolate Botanist: lead with the clip's insight or personality and "
            "reference the source show, host, or channel when supported. Never invent tags."
        )

    if brand in {"TDOG", "THE_DOG"}:
        tdog_parts = [
            guide_rules.get(
                "TDOG_CTA",
                "Dig into more practical gardening conversations with The Dirt on Gardening. "
                "Subscribe and join us for the next episode.",
            ),
            guide_rules.get(
                "TDOG_ABOUT",
                "The Dirt on Gardening brings together practical experience, plant knowledge, "
                "honest conversation, and useful ideas for gardeners at every stage.",
            ),
        ]
        hosts = [
            guide_rules.get("TDOG_HOST_ANDRE", ""),
            guide_rules.get("TDOG_HOST_GRAY", ""),
            guide_rules.get("TDOG_HOST_TCB", ""),
        ]
        if any(hosts):
            tdog_parts.append("🎙️ Our Hosts\n" + "\n".join(item for item in hosts if item))
        sponsors = [
            guide_rules.get("TDOG_SPONSOR_RETAIN", ""),
            guide_rules.get("TDOG_SPONSOR_SONIAS", ""),
            guide_rules.get("TDOG_SPONSOR_ANDRES", ""),
        ]
        if any(sponsors):
            tdog_parts.append("🤝 Sponsors\n" + "\n".join(item for item in sponsors if item))
        telegram_url = guide_rules.get("TDOG_TELEGRAM", "")
        if telegram_url:
            tdog_parts.append(f"💬 Join our Telegram: {telegram_url}")
        email = guide_rules.get("TDOG_EMAIL", "")
        if email:
            tdog_parts.append(f"📧 Email us: {email}")
        tdog_hashtags = guide_rules.get("TDOG_HASHTAGS", "")
        if tdog_hashtags:
            tdog_parts.append(tdog_hashtags)
        brand_cta = "\n\n".join(part for part in tdog_parts if part)
    elif brand.startswith("AGAPE"):
        agape_parts = [
            guide_rules.get(
                "AGAPE_CTA",
                "Join us Sundays at 10 a.m. ET at 3601 Bastion Lane, Raleigh, NC, "
                "for our worship experience. We’re the loving place, and everyone is welcome.",
            ),
            guide_rules.get(
                "AGAPE_LEADERSHIP",
                "Agape International COGIC is under the spirited leadership of "
                "Superintendent Braxton D. Bowser Sr. and First Lady Dr. T’Shura Bowser.",
            ),
            guide_rules.get(
                "AGAPE_ABOUT",
                "Agape is an invigorating Christian ministry in Raleigh, North Carolina, "
                "committed to faith, community, love, and making a positive impact throughout "
                "the Triangle.",
            ),
            guide_rules.get(
                "AGAPE_WEDNESDAY",
                "Join Word on Wednesdays at 7 p.m. ET on Zoom.",
            ),
        ]
        youtube_url = guide_rules.get("AGAPE_YOUTUBE_URL", "https://bit.ly/3uLE6Vj")
        facebook_url = guide_rules.get("AGAPE_FACEBOOK_URL", "https://bit.ly/3oDvv5W")
        pastor_url = guide_rules.get("AGAPE_PASTOR_URL", "https://braxtonbowser.com")
        if youtube_url:
            agape_parts.append(f"Subscribe on YouTube: {youtube_url}")
        if facebook_url:
            agape_parts.append(f"Follow us on Facebook: {facebook_url}")
        if pastor_url:
            agape_parts.append(f"Learn more about Pastor Bowser: {pastor_url}")
        hashtags = guide_rules.get(
            "AGAPE_HASHTAGS",
            "#motivation #inspiration #nc2nd #cogic #churchofgodinchrist "
            "#COGICchurch #COGICstrong #agapestrong #teamcogic "
            "#agapeinternational #agapecogic #greatexploits #braxtonbowser",
        )
        if hashtags:
            agape_parts.append(hashtags)
        brand_cta = "\n\n".join(part for part in agape_parts if part)
    else:
        tcb_parts = [
            guide_rules.get(
                "TCB_CTA",
                "Follow and subscribe to The Chocolate Botanist for factual, funny plant "
                "science and gardening conversations. For business, speaking, and media "
                "bookings, email derek@thechocolatebotanist.com.",
            ),
            guide_rules.get(
                "TCB_ABOUT",
                "Derek Haynes—The Chocolate Botanist—is a Black botanist, scientific "
                "communicator, and garden communicator bringing together science, plants, "
                "facts, and humor while cultivating a passion for plants.",
            ),
        ]
        tcb_links = [
            ("Website", guide_rules.get("TCB_WEBSITE", "https://thechocolatebotanist.com")),
            ("Instagram", guide_rules.get("TCB_INSTAGRAM", "https://www.instagram.com/thechocolatebotanist")),
            ("YouTube", guide_rules.get("TCB_YOUTUBE", "https://www.youtube.com/TheChocolateBotanist")),
            ("Facebook", guide_rules.get("TCB_FACEBOOK", "https://www.facebook.com/TheChocolateBotanist")),
            ("TikTok", guide_rules.get("TCB_TIKTOK", "https://www.tiktok.com/@thechocolatebotanist")),
        ]
        tcb_parts.append(
            "Connect with The Chocolate Botanist\n"
            + "\n".join(f"{label}: {url}" for label, url in tcb_links if url)
        )
        tcb_email = guide_rules.get("TCB_EMAIL", "derek@thechocolatebotanist.com")
        if tcb_email:
            tcb_parts.append(f"Business and speaking inquiries: {tcb_email}")
        tcb_hashtags = guide_rules.get("TCB_HASHTAGS", "#TheChocolateBotanist")
        if tcb_hashtags:
            tcb_parts.append(tcb_hashtags)
        brand_cta = "\n\n".join(part for part in tcb_parts if part)

    attribution_format = guide_rules.get(
        "ATTRIBUTION_FORMAT",
        "🎬 Highlight from {source_show}'s video “{video_title}”",
    )
    source_link_label = guide_rules.get("SOURCE_LINK_LABEL", "Watch the original")
    source_channel_label = guide_rules.get(
        "SOURCE_CHANNEL_LABEL", "Explore more from {source_show}"
    )

    def complete_long_description(
        description: str, dynamic_hashtags: str = ""
    ) -> str:
        """Keep the unique hook first and guarantee source credit on every long video."""
        sections = [str(description or "").strip()]
        credit = (
            attribution_format
            .replace("{source_show}", source_show)
            .replace("{video_title}", source_title or "Untitled video")
        )
        if credit.lower() not in sections[0].lower():
            sections.append(credit)
        if source_url and source_url.lower() not in "\n".join(sections).lower():
            sections.append(f"{source_link_label}: {source_url}")
        if source_channel_url and source_channel_url.lower() not in "\n".join(sections).lower():
            channel_label = source_channel_label.replace("{source_show}", source_show)
            sections.append(f"{channel_label}: {source_channel_url}")
        if brand_cta.lower() not in "\n".join(sections).lower():
            sections.append(brand_cta)
        dynamic_hashtags = str(dynamic_hashtags or "").strip()
        if dynamic_hashtags and dynamic_hashtags.lower() not in "\n".join(sections).lower():
            sections.append(dynamic_hashtags)
        return "\n\n".join(section for section in sections if section).strip()[:5000]

    fallback = []
    for asset in assets:
        transcript = str(asset.get("transcript") or "").strip()
        title = str(asset.get("title") or "").strip() or source_title or "Video Highlight"
        excerpt = re.sub(r"\s+", " ", transcript)[:280].strip()
        caption = excerpt or f"A highlight from {source_title or 'this conversation'}."
        if brand.startswith("AGAPE"):
            caption = f"{caption}\n\nJoin us live online and see us Sunday."
        fallback.append(
            {
                **asset,
                "social_caption": caption,
                "video_title": title[:100],
                "video_description": complete_long_description(caption),
                "hashtags": "",
                "copy_source": "fallback",
            }
        )

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or not assets:
        return fallback
    compact_assets = [
        {
            "asset_id": asset["asset_id"],
            "asset_type": asset["asset_type"],
            "title": asset.get("title", ""),
            "transcript": str(asset.get("transcript") or "")[:12000],
        }
        for asset in assets
    ]
    learning_examples = ""
    try:
        rows = get_rows(RIPPED_LOG_SHEET_ID, "Caption Learning", "A1:P2000")
        matching = [
            row
            for row in rows
            if str(row.get("brand_id") or "").strip().upper() == brand
            and str(row.get("decision") or "").strip().upper()
            in {"ACCEPTED", "EDITED"}
        ][-25:]
        learning_examples = "\n".join(
            "- "
            + json.dumps(
                {
                    "asset_type": row.get("asset_type", ""),
                    "ai_caption": row.get("ai_caption", ""),
                    "final_caption": row.get("final_caption", ""),
                    "ai_title": row.get("ai_title", ""),
                    "final_title": row.get("final_title", ""),
                    "ai_description": row.get("ai_description", ""),
                    "final_description": row.get("final_description", ""),
                },
                ensure_ascii=False,
            )
            for row in matching
        )[:20000]
    except Exception:
        logger.exception("Could not load Caption Learning examples")

    prompt = (
        "Create publication-ready metadata for each supplied video asset. "
        "Return JSON with an assets array; every item must contain asset_id, "
        "social_caption, video_title, video_description, and hashtags. "
        "For 9:16, write an engaging natural social caption with a hook, useful "
        "context, attribution when known, a light CTA, and a few relevant hashtags. "
        "For 16:9, write a compelling YouTube/Facebook title and a fuller description. "
        "Put a unique two- or three-sentence hook and summary first. Do not repeat the title. "
        "The system will then append the mandatory source credit, original-video URL, and "
        "brand CTA, so do not invent or duplicate those lines. Keep hashtags out of the "
        "description because they are returned separately. Do not fabricate names, handles, "
        "guests, facts, or links. Avoid clickbait that the transcript does not earn.\n\n"
        f"BRAND RULE:\n{brand_rule}\n\n"
        f"NORMALIZED SOCIAL COPY GUIDE:\n{guide_text or 'Built-in defaults apply.'}\n\n"
        f"PAST AI-TO-FINAL EXAMPLES (imitate the final wording and SEO judgment, "
        f"not stale facts):\n{learning_examples or 'No examples yet.'}\n\n"
        f"SOURCE TITLE:\n{source_title}\n\n"
        f"SOURCE URL:\n{source_url or 'Unavailable'}\n\n"
        f"ASSETS:\n{json.dumps(compact_assets, ensure_ascii=False)}"
    )
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": os.getenv("OPENAI_METADATA_MODEL", "gpt-5-mini"),
                "messages": [
                    {"role": "system", "content": "Return strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
            },
            timeout=(10, 600),
        )
        response.raise_for_status()
        parsed = json.loads(response.json()["choices"][0]["message"]["content"])
        generated = {
            str(item.get("asset_id") or ""): item
            for item in parsed.get("assets", [])
            if item.get("asset_id")
        }
        enriched = []
        for item in fallback:
            copy = generated.get(str(item["asset_id"])) or {}
            generated_hashtags = str(copy.get("hashtags") or "").strip()
            enriched.append(
                {
                    **item,
                    "social_caption": str(copy.get("social_caption") or item["social_caption"]).strip(),
                    "video_title": str(copy.get("video_title") or item["video_title"]).strip()[:100],
                    "video_description": complete_long_description(
                        str(copy.get("video_description") or item["video_description"]).strip(),
                        generated_hashtags if item.get("asset_type") == "16:9_HIGHLIGHT" else "",
                    ),
                    "hashtags": generated_hashtags,
                    "copy_source": "openai" if copy else item["copy_source"],
                }
            )
        return enriched
    except Exception:
        logger.exception("Schedule copy generation failed; using transcript fallback")
        return fallback


SCHEDULE_OUTBOX_TAB = os.getenv("SCHEDULE_OUTBOX_TAB", "Schedule Handoff Outbox").strip()


def _persist_schedule_outbox(payload: dict[str, Any], status: str = "READY") -> None:
    """Upsert the complete approved handoff before network delivery."""
    import main
    from google_sheets import get_or_create_headers

    _, _, sheets = main.get_google_services()
    headers = [
        "recorded_at", "request_id", "asset_id", "brand_id",
        "youtube_video_id", "youtube_channel_id", "source_url", "source_title",
        "asset_type", "candidate_number", "drive_url", "transcript",
        "duration_seconds", "main_theme", "keywords", "social_caption",
        "video_title", "video_description", "hashtags", "copy_source",
        "handoff_status",
    ]
    headers = get_or_create_headers(
        RIPPED_LOG_SHEET_ID, SCHEDULE_OUTBOX_TAB, headers
    )
    result = sheets.spreadsheets().values().get(
        spreadsheetId=RIPPED_LOG_SHEET_ID,
        range=f"'{SCHEDULE_OUTBOX_TAB}'!A1:U5000",
    ).execute()
    values = result.get("values", [])
    existing = {}
    if values:
        normalized = [
            str(value or "").strip().lower().replace(" ", "_")
            for value in values[0]
        ]
        asset_column = normalized.index("asset_id")
        for row_number, row in enumerate(values[1:], start=2):
            asset_id = str(row[asset_column] if asset_column < len(row) else "").strip()
            if asset_id:
                existing[asset_id] = row_number

    rows_to_append = []
    updates = []
    common = {
        "recorded_at": now(),
        "request_id": str(payload.get("request_id") or ""),
        "brand_id": str(payload.get("show_id") or "").upper(),
        "youtube_video_id": str(payload.get("youtube_video_id") or ""),
        "youtube_channel_id": str(payload.get("youtube_channel_id") or ""),
        "source_url": str(payload.get("source_url") or ""),
        "source_title": str(payload.get("source_title") or ""),
        "handoff_status": status,
    }
    for asset in payload.get("assets") or []:
        row = {
            **common,
            "asset_id": str(asset.get("asset_id") or ""),
            "asset_type": str(asset.get("asset_type") or ""),
            "candidate_number": asset.get("candidate_number") or "",
            "drive_url": str(asset.get("drive_url") or ""),
            "transcript": str(asset.get("transcript") or ""),
            "duration_seconds": asset.get("duration_seconds") or "",
            "main_theme": str(asset.get("main_theme") or ""),
            "keywords": str(asset.get("keywords") or ""),
            "social_caption": str(asset.get("social_caption") or ""),
            "video_title": str(asset.get("video_title") or asset.get("title") or ""),
            "video_description": str(asset.get("video_description") or ""),
            "hashtags": str(asset.get("hashtags") or ""),
            "copy_source": str(asset.get("copy_source") or ""),
        }
        ordered = [row.get(header, "") for header in headers]
        asset_id = row["asset_id"]
        if asset_id in existing:
            row_number = existing[asset_id]
            updates.append({
                "range": f"'{SCHEDULE_OUTBOX_TAB}'!A{row_number}:U{row_number}",
                "values": [ordered],
            })
        else:
            rows_to_append.append(ordered)
    if updates:
        sheets.spreadsheets().values().batchUpdate(
            spreadsheetId=RIPPED_LOG_SHEET_ID,
            body={"valueInputOption": "RAW", "data": updates},
        ).execute()
    if rows_to_append:
        sheets.spreadsheets().values().append(
            spreadsheetId=RIPPED_LOG_SHEET_ID,
            range=f"'{SCHEDULE_OUTBOX_TAB}'!A1",
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": rows_to_append},
        ).execute()
    logger.warning(
        "Persisted Schedule Handoff Outbox request_id=%s assets=%s status=%s",
        payload.get("request_id"), len(payload.get("assets") or []), status,
    )


def _handoff_shorts_to_schedule_master(request_id: str, chat_id: str) -> None:
    """Send all final rendered 9:16 and 16:9 selections downstream once."""
    target = (
        os.getenv("SCHEDULE_MASTER_INTERNAL_URL")
        or os.getenv("SCHEDULE_MASTER_URL")
        or ""
    ).strip().rstrip("/")
    if not target:
        logger.warning(
            "Schedule Master handoff skipped request_id=%s reason=missing_url",
            request_id,
        )
        send(chat_id, "⚠️ Your Shorts are complete, but Schedule Master is not configured.")
        return
    with _LOCK, _telegram_db() as db:
        row = db.execute(
            "SELECT * FROM telegram_requests WHERE request_id=?", (request_id,)
        ).fetchone()
        if not row:
            return
        state = json.loads(row["state_json"])
        schedule = dict(state.get("schedule_master") or {})
        if schedule.get("shorts_status") in {"sending", "accepted"}:
            return
        reviews = dict(state.get("candidate_reviews") or {})
        clips = (state.get("result") or {}).get("segments", [])
        assets = []
        for index, clip in enumerate(clips):
            review = reviews.get(str(index)) or {}
            if review.get("status") != "rendered" or not review.get("clip_url"):
                continue
            number = int(clip.get("candidate_number") or index + 1)
            assets.append(
                {
                    "asset_id": f"{request_id}:short:{number}",
                    "candidate_number": number,
                    "asset_type": "9:16_SHORT",
                    "drive_url": review["clip_url"],
                    "transcript": str(clip.get("transcript") or ""),
                    "duration_seconds": float(clip.get("duration") or 0),
                    "main_theme": str(
                        (state.get("result") or {}).get("analysis", {}).get("main_theme")
                        or ""
                    ),
                    "keywords": ", ".join(
                        (state.get("result") or {}).get("analysis", {}).get("keywords")
                        or []
                    ),
                }
            )
        topic_reviews = dict(state.get("topic_reviews") or {})
        topics = (state.get("topic_result") or {}).get("segments", [])
        for index, segment in enumerate(topics):
            review = topic_reviews.get(str(index)) or {}
            if review.get("status") != "rendered" or not review.get("segment_url"):
                continue
            number = index + 1
            assets.append(
                {
                    "asset_id": f"{request_id}:highlight:{number}",
                    "candidate_number": number,
                    "asset_type": "16:9_HIGHLIGHT",
                    "drive_url": review["segment_url"],
                    "transcript": str(segment.get("transcript") or ""),
                    "duration_seconds": float(segment.get("duration") or 0),
                    "title": str(segment.get("title") or ""),
                    "summary": str(segment.get("summary") or ""),
                }
            )
        if not assets:
            send(chat_id, "No rendered videos were selected for Schedule Master.")
            return
        schedule["shorts_status"] = "sending"
        schedule["shorts_started_at"] = now()
        state["schedule_master"] = schedule
        db.execute(
            "UPDATE telegram_requests SET state_json=?, updated_at=? WHERE request_id=?",
            (json.dumps(state), now(), request_id),
        )
    parsed = state.get("parsed") or {}
    reviewed_copy = {
        str(item.get("asset_id") or ""): item
        for item in (state.get("copy_drafts") or [])
    }
    if reviewed_copy:
        assets = [
            {**asset, **reviewed_copy.get(str(asset.get("asset_id") or ""), {})}
            for asset in assets
        ]
    else:
        # Compatibility fallback for jobs created before the copy-review release.
        assets = _generate_schedule_copy(
            assets,
            str(state.get("show_id") or ""),
            _state_vid_title(state),
            str((state.get("parsed") or {}).get("source_value") or ""),
        )
    payload = {
        "request_id": request_id,
        "youtube_video_id": str(parsed.get("video_id") or ""),
        "youtube_channel_id": str(state.get("youtube_channel_id") or "") or None,
        "show_id": str(state.get("show_id") or "") or None,
        "source_url": str(parsed.get("source_value") or "") or None,
        "source_title": _state_vid_title(state),
        "assets": assets,
    }
    try:
        _persist_schedule_outbox(payload, "READY")
    except Exception:
        logger.exception(
            "Could not persist Schedule Handoff Outbox request_id=%s", request_id
        )
        send(
            chat_id,
            "❌ Scheduling paused because the durable handoff record could not be saved. "
            "Your rendered videos remain in Drive; use Schedule Now again after Sheets recovers.",
        )
        with _LOCK, _telegram_db() as db:
            row = db.execute(
                "SELECT state_json FROM telegram_requests WHERE request_id=?",
                (request_id,),
            ).fetchone()
            if row:
                latest_state = json.loads(row["state_json"])
                latest_schedule = dict(latest_state.get("schedule_master") or {})
                latest_schedule.update({
                    "shorts_status": "failed",
                    "shorts_error": "Durable Schedule Handoff Outbox write failed",
                    "shorts_failed_at": now(),
                })
                latest_state["schedule_master"] = latest_schedule
                db.execute(
                    "UPDATE telegram_requests SET state_json=?, updated_at=? WHERE request_id=?",
                    (json.dumps(latest_state), now(), request_id),
                )
        return
    headers = {}
    secret = os.getenv("SCHEDULE_MASTER_SHARED_SECRET", "").strip()
    if secret:
        headers["x-schedule-master-secret"] = secret
    try:
        response = requests.post(
            f"{target}/schedule/intake",
            json=payload,
            headers=headers,
            timeout=(10, 60),
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Schedule Master returned {response.status_code}: {response.text[:1000]}"
            )
        result = response.json()
        with _LOCK, _telegram_db() as db:
            row = db.execute(
                "SELECT state_json FROM telegram_requests WHERE request_id=?",
                (request_id,),
            ).fetchone()
            latest_state = json.loads(row["state_json"])
            latest_schedule = dict(latest_state.get("schedule_master") or {})
            latest_schedule.update(
                {
                    "shorts_status": "accepted",
                    "shorts_completed_at": now(),
                    "shorts_response": result,
                }
            )
            latest_state["schedule_master"] = latest_schedule
            db.execute(
                "UPDATE telegram_requests SET state_json=?, updated_at=? WHERE request_id=?",
                (json.dumps(latest_state), now(), request_id),
            )
        logger.info(
            "Schedule Master accepted Shorts request_id=%s assets=%s track_id=%s",
            request_id,
            len(assets),
            result.get("track_id", ""),
        )
        send(
            chat_id,
            f"✅ Selection complete. Schedule Master received {len(assets)} Short(s).",
        )
    except Exception as exc:
        logger.exception("Schedule Master Shorts handoff failed request_id=%s", request_id)
        with _LOCK, _telegram_db() as db:
            row = db.execute(
                "SELECT state_json FROM telegram_requests WHERE request_id=?",
                (request_id,),
            ).fetchone()
            if row:
                latest_state = json.loads(row["state_json"])
                latest_schedule = dict(latest_state.get("schedule_master") or {})
                latest_schedule.update(
                    {"shorts_status": "failed", "shorts_error": str(exc), "shorts_failed_at": now()}
                )
                latest_state["schedule_master"] = latest_schedule
                db.execute(
                    "UPDATE telegram_requests SET state_json=?, updated_at=? WHERE request_id=?",
                    (json.dumps(latest_state), now(), request_id),
                )
        send(chat_id, f"❌ Schedule Master handoff failed: {str(exc)[:1200]}")


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
    # Both editorial lanes are the normal workflow. Merely saying "Ripped Shorts",
    # "clip", "highlight", or "short" must not silently suppress 16:9 highlights.
    # A single-lane run requires an explicit "only" override.
    shorts_only = bool(
        re.search(
            r"\b(?:shorts?|9:16|vertical)(?:\s+(?:clips?|videos?))?\s+only\b"
            r"|\bonly\s+(?:shorts?|9:16|vertical)(?:\s+(?:clips?|videos?))?\b",
            lowered,
        )
    )
    topics_only = bool(
        re.search(
            r"\b(?:topics?|segments?|16:9|horizontal|highlights?)(?:\s+(?:clips?|videos?))?\s+only\b"
            r"|\bonly\s+(?:topics?|segments?|16:9|horizontal|highlights?)(?:\s+(?:clips?|videos?))?\b",
            lowered,
        )
    )
    if shorts_only and not topics_only:
        mode: Literal["shorts", "topics", "both"] = "shorts"
    elif topics_only and not shorts_only:
        mode = "topics"
    else:
        mode = "both"
    if youtube:
        return {"mode": mode, "source_kind": "youtube", "source_value": youtube.group(0), "video_id": youtube.group(1)}
    if drive:
        return {"mode": mode, "source_kind": "drive", "source_value": drive[0], "drive_ids": drive}
    raise ValueError("Send a YouTube link or a Google Drive video link.")


def _youtube_vid_title(youtube_url: str) -> str:
    """Resolve the source YouTube title used as the Drive grouping folder."""
    response = requests.get(
        "https://www.youtube.com/oembed",
        params={"url": youtube_url, "format": "json"},
        timeout=(10, 30),
    )
    response.raise_for_status()
    title = str(response.json().get("title") or "").strip()
    if not title:
        raise RuntimeError("YouTube returned no Vid Title for the Ripped Shorts folder")
    return title


def _drive_vid_title(metadata: dict[str, Any]) -> str:
    """Use the source Drive video's filename as Vid Title, without its video extension."""
    title = str(metadata.get("name") or "").strip()
    title = re.sub(r"\.(mp4|mov|m4v|webm|mkv|avi)$", "", title, flags=re.I).strip()
    if not title:
        raise RuntimeError("The source Drive video has no Vid Title for the Ripped Shorts folder")
    return title


def _state_vid_title(state: dict[str, Any]) -> str:
    """Resolve Vid Title for new and already-pending review state."""
    existing = str(state.get("vid_title") or "").strip()
    if existing:
        return existing
    parsed = state.get("parsed") or {}
    if parsed.get("source_kind") == "youtube":
        return _youtube_vid_title(str(parsed.get("source_value") or ""))
    drive_ids = parsed.get("drive_ids") or []
    if drive_ids:
        return _drive_vid_title(drive_metadata(str(drive_ids[0])))
    raise RuntimeError("Vid Title could not be resolved for the approved Ripped Shorts output")


def _authorized(chat_id: str, user_id: str) -> bool:
    chats, users = _csv_env("TELEGRAM_ALLOWED_CHAT_IDS"), _csv_env("TELEGRAM_ALLOWED_USER_IDS")
    for name in ("TELEGRAM_CHAT_ID", "TELEGRAM_GROUP_CHAT_ID", "Telegram_Group_Chat_ID"):
        existing_chat = os.getenv(name, "").strip()
        if existing_chat:
            chats.add(existing_chat)
    # Fail closed: at least one allow-list must be configured.
    if not chats and not users:
        return False
    return (bool(chats) and chat_id in chats) or (bool(users) and user_id in users)


def _ripped_bot_token() -> str:
    """Use the dedicated Ripped Shorts bot while preserving Railway's current names."""
    for name in (
        "RIPPED_SHORTS_TELEGRAM_BOT_TOKEN",
        "TELEGRAM_RIPPED_BOT_TOKEN",
        "Telegram_ripped_bot_token",
        "TELEGRAM_BOT_TOKEN",
    ):
        token = os.getenv(name, "").strip()
        if token:
            return token
    return ""


def telegram(method: str, payload: dict[str, Any]) -> dict:
    """Send through Telegram without allowing a temporary rate limit to corrupt a job."""
    token = _ripped_bot_token()
    if not token:
        raise RuntimeError("Telegram_ripped_bot_token is not configured")
    last_error = ""
    for attempt in range(1, 7):
        response = requests.post(
            f"https://api.telegram.org/bot{token}/{method}",
            json=payload,
            timeout=(10, 60),
        )
        try:
            data = response.json()
        except ValueError:
            data = {}
        if response.status_code == 429 or int(data.get("error_code") or 0) == 429:
            retry_after = max(
                1, int((data.get("parameters") or {}).get("retry_after") or 1)
            )
            last_error = str(data.get("description") or response.text[:1000])
            logger.warning(
                "Telegram %s rate limited attempt=%s retry_after=%ss",
                method,
                attempt,
                retry_after,
            )
            threading.Event().wait(min(retry_after + 1, 65))
            continue
        if response.status_code != 200 or not data.get("ok", False):
            raise RuntimeError(
                f"Telegram {method} failed ({response.status_code}): "
                f"{data.get('description') or response.text[:1000]}"
            )
        return data
    raise RuntimeError(
        f"Telegram {method} remained rate limited after retries: {last_error}"
    )


def send(chat_id: str, text: str) -> None:
    telegram("sendMessage", {"chat_id": chat_id, "text": text, "disable_web_page_preview": True})


def _status_card_text(request_id: str, status: str, state: dict[str, Any]) -> str:
    reviews = dict(state.get("candidate_reviews") or {})
    topics = dict(state.get("topic_reviews") or {})
    short_total = len((state.get("result") or {}).get("segments") or [])
    topic_total = len((state.get("topic_result") or {}).get("segments") or [])
    rendered = sum(
        str(item.get("status") or "") == "rendered"
        for item in [*reviews.values(), *topics.values()]
    )
    active = sum(
        str(item.get("status") or "") in {"queued", "rendering"}
        for item in [*reviews.values(), *topics.values()]
    )
    error = str(state.get("error") or state.get("notification_error") or "")
    return (
        f"📍 Ripped Shorts Job Status\n"
        f"Job: {request_id}\n"
        f"Brand: {state.get('show_id') or 'pending'}\n"
        f"Stage: {state.get('stage') or status}\n"
        f"Shorts: {short_total} candidates\n"
        f"16:9: {topic_total} candidates\n"
        f"Renders: {rendered} complete, {active} active\n"
        f"Next: {state.get('next_action') or 'continue'}"
        + (
            f"\nError: {classify_error(error)} — {error[:500]}"
            if error
            else ""
        )
    )


def _update_status_card(
    request_id: str, status: str, state: dict[str, Any], chat_id: str
) -> None:
    if not chat_id:
        return
    try:
        # Status updates are queued. Always refresh the latest row so an older
        # queued snapshot cannot overwrite newer workflow state or create a
        # second card before the first card's message ID is persisted.
        with _LOCK, _telegram_db() as db:
            row = db.execute(
                "SELECT status, state_json FROM telegram_requests WHERE request_id=?",
                (request_id,),
            ).fetchone()
        if row:
            status = str(row["status"] or status)
            state = json.loads(row["state_json"])
        text = _status_card_text(request_id, status, state)
        message_id = state.get("status_message_id")
        if message_id:
            try:
                telegram(
                    "editMessageText",
                    {
                        "chat_id": chat_id,
                        "message_id": message_id,
                        "text": text,
                        "disable_web_page_preview": True,
                    },
                )
                return
            except Exception as exc:
                if "message is not modified" in str(exc).lower():
                    return
                logger.warning(
                    "Status card edit failed; creating replacement job_id=%s",
                    request_id,
                )
        result = telegram(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": text,
                "disable_web_page_preview": True,
            },
        )
        new_message_id = (result.get("result") or {}).get("message_id")
        if new_message_id:
            # Merge only the Telegram message ID into the newest saved state.
            # Never replace the row with the stale state used to render a card.
            with _LOCK, _telegram_db() as db:
                latest = db.execute(
                    "SELECT state_json FROM telegram_requests WHERE request_id=?",
                    (request_id,),
                ).fetchone()
                if latest:
                    latest_state = json.loads(latest["state_json"])
                    latest_state["status_message_id"] = new_message_id
                    db.execute(
                        "UPDATE telegram_requests SET state_json=?, updated_at=? "
                        "WHERE request_id=?",
                        (json.dumps(latest_state), now(), request_id),
                    )
    except Exception:
        logger.exception("JOB_STATUS_CARD_FAILED job_id=%s", request_id)


def _save(request_id: str, status: str, state: dict[str, Any]) -> None:
    state["heartbeat_at"] = now()
    if status != "error":
        state["last_successful_stage"] = str(state.get("stage") or status)
    with _LOCK, _telegram_db() as db:
        db.execute(
            "UPDATE telegram_requests SET status=?, state_json=?, updated_at=? WHERE request_id=?",
            (status, json.dumps(state), now(), request_id),
        )
        row = db.execute(
            "SELECT chat_id, user_id FROM telegram_requests WHERE request_id=?",
            (request_id,),
        ).fetchone()
    chat_id = str(row["chat_id"]) if row else ""
    user_id = str(row["user_id"]) if row else ""
    upsert_job(
        RIPPED_LOG_SHEET_ID,
        request_id,
        status,
        state,
        chat_id=chat_id,
        user_id=user_id,
    )
    STATUS_EXECUTOR.submit(
        _update_status_card,
        request_id,
        status,
        dict(state),
        chat_id,
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


def _packaged_short_framework_prompt() -> str:
    """Return the last approved framework snapshot shipped with the service."""
    path = Path(__file__).with_name("ripped_shorts_9x16_framework.txt")
    try:
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    except Exception as exc:
        logger.error("Packaged 9:16 framework unavailable: %s", exc)
    return ""


def _configured_short_framework_prompt() -> str:
    """Prefer the editable Google Doc and fail over to its approved snapshot."""
    packaged = _packaged_short_framework_prompt()
    try:
        rows = get_rows(RIPPED_LOG_SHEET_ID, "Show Config", "A1:AF1000")
        ripped = next(
            (
                row
                for row in rows
                if str(row.get("show_id") or "").strip().upper() == "RIPPED"
            ),
            None,
        )
        reference = str(
            (ripped or {}).get("ai_prompt")
            or (ripped or {}).get("open_ai_prompt")
            or (ripped or {}).get("prompt")
            or ""
        ).strip()
        if not reference:
            logger.warning(
                "RIPPED Show Config has no prompt reference; using packaged approved framework"
            )
            return packaged
        configured = read_google_doc_text(reference).strip()
        if configured:
            logger.info("Loaded live 9:16 editorial framework from Google Docs")
            return configured
        logger.warning("Live 9:16 framework was blank; using packaged approved framework")
    except Exception as exc:
        logger.warning(
            "Live 9:16 editorial framework unavailable; using packaged approved "
            "framework instead: %s",
            exc,
        )
    if not packaged:
        raise RuntimeError(
            "Neither the live Google Doc nor the packaged approved 9:16 framework is available"
        )
    return packaged


def _brand_short_prompt(state: dict[str, Any]) -> str:
    show_id = str(state.get("show_id") or "").strip().upper()
    common = (
        "CONVERSATIONAL CLIP RULES\n"
        "Treat a compelling answer, response, reaction, rebuttal, or exchange as a "
        "first-class Short candidate. Include the question or the shortest necessary "
        "setup when the answer would otherwise be unclear. The final clip must still "
        "open cleanly, make sense to a new viewer, and reach the speaker's complete payoff."
    )
    if show_id == "TCB":
        return (
            common
            + "\nTCB SOURCE CONTEXT\nThis is a Chocolate Botanist appearance or "
            "livestream clip. Favor insightful, intriguing, useful, funny, or memorable "
            "moments that represent the appearance well. Preserve enough source-show "
            "context for later caption attribution."
        )
    if show_id == "TDOG":
        return (
            common
            + "\nTDOG SOURCE CONTEXT\nThis is The Dirt on Gardening. Favor complete "
            "gardening insights, host/guest exchanges, practical answers, stories, "
            "myth corrections, and memorable reactions."
        )
    if show_id == "AGAPE_CHURCH":
        return (
            common
            + "\nAGAPE SOURCE CONTEXT\nFavor complete Christian teaching, motivation, "
            "inspiration, testimony, prayer, and sermon application. Preserve theological "
            "context and never select an isolated line that changes the intended meaning."
        )
    return common


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
    """Send compact review pages instead of one Telegram message per candidate."""
    clips = result.get("segments", [])
    review_count = max(0, len(clips) - start_index)
    page_size = max(
        2, min(5, int(os.getenv("TELEGRAM_CANDIDATES_PER_PAGE", "4")))
    )
    selected = list(range(start_index, len(clips)))
    if not selected:
        send(
            chat_id,
            f"✅ Analysis complete\nJob ID: {request_id}\nNo new Shorts need review.",
        )
        _send_short_confirmation(chat_id, request_id)
        return

    total_pages = (len(selected) + page_size - 1) // page_size
    for page_number, offset in enumerate(range(0, len(selected), page_size), 1):
        indexes = selected[offset : offset + page_size]
        sections = [
            f"✅ Shorts review • Page {page_number}/{total_pages}\n"
            f"Job ID: {request_id}\n"
            f"{review_count} new candidate(s). Approve only what you want rendered."
        ]
        buttons = []
        for zero_index in indexes:
            clip = clips[zero_index]
            short_number = int(
                clip.get("candidate_number") or zero_index + 1
            )
            transcript = str(clip.get("transcript", "")).strip()
            if len(transcript) > 560:
                transcript = transcript[:557] + "..."
            sections.append(
                f"\nSHORT {short_number}\n"
                f"{_timecode(float(clip['start']))}–"
                f"{_timecode(float(clip['end']))} • "
                f"{round(float(clip['duration']))}s • "
                f"{clip.get('category', 'social clip')}\n"
                f"Why: {str(clip.get('reason', ''))[:260]}\n"
                f"{transcript}"
            )
            buttons.append(
                [
                    {
                        "text": f"✅ {short_number}",
                        "callback_data": (
                            f"rs:approve:{request_id}:{zero_index}"
                        ),
                    },
                    {
                        "text": f"❌ {short_number}",
                        "callback_data": (
                            f"rs:reject:{request_id}:{zero_index}"
                        ),
                    },
                ]
            )
        buttons.append(
            [{
                "text": "✏️ Change / Add / Options",
                "callback_data": f"rs:options:{request_id}",
            }]
        )
        telegram(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": "\n".join(sections)[:4000],
                "disable_web_page_preview": True,
                "reply_markup": {"inline_keyboard": buttons},
            },
        )
    _send_short_confirmation(chat_id, request_id)


def _send_short_confirmation(chat_id: str, request_id: str) -> None:
    with _LOCK, _telegram_db() as db:
        row = db.execute(
            "SELECT mode, state_json FROM telegram_requests WHERE request_id=?",
            (request_id,),
        ).fetchone()
    if not row:
        return
    state = json.loads(row["state_json"])
    if state.get("shorts_confirmed_at") or state.get("short_selection_completed_at"):
        return
    continue_text = (
        "Untouched Shorts will be skipped, your rendered choices will go to "
        "Schedule Master, and 16:9 analysis will begin."
        if row["mode"] == "both"
        else "Untouched Shorts will be skipped and your rendered choices will go "
        "to Schedule Master for captioning and scheduling."
    )
    telegram(
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": (
                "Approve the 9:16 Shorts you want. When you are finished choosing, "
                "tap the button below. " + continue_text
            ),
            "reply_markup": {
                "inline_keyboard": (
                    [
                        [{
                            "text": "✅ Continue to 16:9 Highlights",
                            "callback_data": f"rs:shorts_confirm:{request_id}",
                        }],
                        [{
                            "text": "📅 No More Videos — Schedule Now",
                            "callback_data": f"rs:schedule_now:{request_id}",
                        }],
                    ]
                    if row["mode"] == "both"
                    else [[{
                        "text": "📅 No More Shorts — Schedule Now",
                        "callback_data": f"rs:schedule_now:{request_id}",
                    }]]
                )
            },
        },
    )

def _transcribe(video_path: Path, progress=None) -> list[dict]:
    """Transcribe safely in bounded chunks and recombine absolute timestamps."""
    import subprocess

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to transcribe a video without a transcript")

    chunk_seconds = max(120, int(os.getenv("RIPPED_TRANSCRIPTION_CHUNK_SECONDS", "600")))
    workers = max(1, min(6, int(os.getenv("RIPPED_TRANSCRIPTION_WORKERS", "4"))))
    retries = max(1, int(os.getenv("RIPPED_TRANSCRIPTION_RETRIES", "3")))
    chunk_dir = video_path.parent / "transcription-chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    pattern = chunk_dir / "chunk-%03d.mp3"
    if progress:
        progress("📝 Preparing audio for safe, chunked transcription.")

    completed = subprocess.run(
        [
            os.getenv("FFMPEG_BINARY", "ffmpeg"), "-y", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k",
            "-f", "segment", "-segment_time", str(chunk_seconds),
            "-reset_timestamps", "1", str(pattern),
        ],
        capture_output=True, text=True, timeout=3600,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Audio chunking failed: {completed.stderr[-1000:]}")

    chunk_paths = sorted(
        path for path in chunk_dir.glob("chunk-*.mp3")
        if path.is_file() and path.stat().st_size > 0
    )
    if not chunk_paths:
        raise RuntimeError("Audio chunking produced no usable files")
    oversize = [path for path in chunk_paths if path.stat().st_size >= 24 * 1024 * 1024]
    if oversize:
        raise RuntimeError(
            "A transcription chunk exceeded the 24 MiB safety ceiling: "
            + ", ".join(f"{path.name}={path.stat().st_size}" for path in oversize)
        )
    if progress:
        progress(f"🎧 Audio prepared in {len(chunk_paths)} chunk(s); transcription started.")

    def transcribe_one(index_path):
        index, path = index_path
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                with path.open("rb") as handle:
                    response = requests.post(
                        "https://api.openai.com/v1/audio/transcriptions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        data={
                            "model": os.getenv("TRANSCRIPTION_MODEL", "whisper-1"),
                            "response_format": "verbose_json",
                            "timestamp_granularities[]": "segment",
                        },
                        files={"file": (path.name, handle, "audio/mpeg")},
                        timeout=(10, 1800),
                    )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"HTTP {response.status_code}: {response.text[:1000]}"
                    )
                offset = index * chunk_seconds
                adjusted = []
                for item in response.json().get("segments", []):
                    start = float(item["start"]) + offset
                    end = float(item["end"]) + offset
                    adjusted.append({
                        "start": start,
                        "end": end,
                        "duration": end - start,
                        "text": str(item["text"]).strip(),
                        "chunk_index": index + 1,
                    })
                return adjusted
            except Exception as exc:
                last_error = exc
                if attempt < retries:
                    import time
                    time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(
            f"Transcription chunk {index + 1}/{len(chunk_paths)} failed after "
            f"{retries} attempts: {last_error}"
        )

    result = []
    completed_count = 0
    last_bucket = -1
    with ThreadPoolExecutor(max_workers=min(workers, len(chunk_paths))) as executor:
        futures = {
            executor.submit(transcribe_one, item): item[0]
            for item in enumerate(chunk_paths)
        }
        for future in as_completed(futures):
            result.extend(future.result())
            completed_count += 1
            percent = int(completed_count * 100 / len(chunk_paths))
            bucket = percent // 25
            if progress and (completed_count == 1 or completed_count == len(chunk_paths) or bucket > last_bucket):
                progress(
                    f"📝 Transcript progress: {completed_count}/{len(chunk_paths)} "
                    f"chunk(s) complete ({percent}%)."
                )
                last_bucket = bucket

    result.sort(key=lambda item: (item["start"], item["end"]))
    if not result:
        raise RuntimeError("Transcription returned no timed segments")
    if progress:
        progress(f"✅ Timed transcript complete: {len(result)} segments.")
    return result

def _topic_break_suggestions(transcript_segments: list[dict]) -> list[dict]:
    """Select the strongest standalone 16:9 highlights from the full transcript."""
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
        "You are the 16:9 Highlight Editor for Ripped Shorts. Review the ENTIRE "
        "timestamped transcript before selecting anything. Select only the strongest "
        "standalone portions for YouTube and Facebook: a complete point or lesson, "
        "meaningful discussion, compelling story, useful explanation, strong argument "
        "or opinion, memorable exchange, focused tangent, or connected sequence of ideas.\n\n"
        "Do NOT divide the video into arbitrary blocks and do NOT try to cover the full "
        "timeline. Gaps are allowed. Each highlight must contain its central point plus "
        "the adjoining setup, supporting explanation, examples, questions and responses, "
        "story details, conclusion, lesson, or payoff needed to understand it. Do not add "
        "unrelated material merely to make it longer.\n\n"
        "Every highlight must be at least 180 seconds. Eight minutes is only a loose "
        "reference, never a target or maximum. Start at the natural beginning of the "
        "subject or setup, end after the point or payoff is complete, and use only exact "
        "transcript boundaries. Never cut a sentence, speaker, example, prayer, "
        "declaration, or conclusion. Highlights must be distinct, non-overlapping, "
        "chronological, understandable without the full source, and strong enough to "
        "publish separately. Select quality over quantity and return an empty segments "
        "array if nothing qualifies.\n\n"
        "Return strict JSON only as "
        '{"analysis":{"content_type":"other","main_theme":"","major_points":[]},'
        '"segments":[{"start":0,"end":180,"duration":180,"title":"...",'
        '"summary":"...","highlight_type":"point","reason":"..."}]}. '
        "Do not include Markdown or commentary.\n\nTRANSCRIPT:\n"
        + "\n".join(lines)
        + _boundary_learning_prompt()
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
    logger.warning("16:9 highlight selection failed: %s", last_error)
    return []


def _build_contiguous_topic_segments(
    transcript_segments: list[dict], suggestions: list[dict]
) -> list[dict]:
    """Validate selected standalone highlights without forcing full-timeline coverage."""
    ordered = sorted(transcript_segments, key=lambda item: float(item.get("start", 0)))
    if not ordered:
        return []
    minimum = max(180.0, float(os.getenv("TOPIC_SEGMENT_MIN_SECONDS", "180")))
    tolerance = 0.75
    transcript_starts = [float(item.get("start", 0)) for item in ordered]
    transcript_ends = [
        float(item.get("end", float(item.get("start", 0)) + float(item.get("duration", 0))))
        for item in ordered
    ]

    candidates = []
    for suggestion in suggestions:
        try:
            proposed_start = float(suggestion.get("start"))
            proposed_end = float(suggestion.get("end"))
        except (TypeError, ValueError):
            continue
        start = min(transcript_starts, key=lambda value: abs(value - proposed_start))
        end = min(transcript_ends, key=lambda value: abs(value - proposed_end))
        if end - start < minimum:
            continue
        included = []
        for item in ordered:
            seg_start = float(item.get("start", 0))
            seg_end = float(
                item.get("end", seg_start + float(item.get("duration", 0)))
            )
            if seg_start >= start - tolerance and seg_end <= end + tolerance:
                included.append(str(item.get("text", "")).strip())
        if not included:
            continue
        candidates.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "title": str(suggestion.get("title") or "16:9 Highlight").strip(),
                "summary": str(suggestion.get("summary") or "").strip(),
                "highlight_type": str(
                    suggestion.get("highlight_type") or "other"
                ).strip(),
                "reason": str(suggestion.get("reason") or "").strip(),
                "transcript": " ".join(included),
                "aspect_ratio": "16:9",
                "_selection_score": float(suggestion.get("score", 0) or 0),
            }
        )

    # Prefer the strongest non-overlapping candidates, then restore chronology.
    ranked = sorted(
        candidates,
        key=lambda item: (item["_selection_score"], item["duration"]),
        reverse=True,
    )
    selected = []
    for candidate in ranked:
        if any(
            candidate["start"] < existing["end"]
            and candidate["end"] > existing["start"]
            for existing in selected
        ):
            continue
        selected.append(candidate)
    selected.sort(key=lambda item: item["start"])
    for index, item in enumerate(selected, 1):
        item.pop("_selection_score", None)
        item["segment_number"] = index
    return selected


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
                    "inline_keyboard": [
                        [
                            {
                                "text": "✅ Approve 16:9",
                                "callback_data": f"rs:topic_approve:{request_id}:{index}",
                            },
                            {
                                "text": "❌ Skip",
                                "callback_data": f"rs:topic_reject:{request_id}:{index}",
                            },
                        ],
                        [
                            {
                                "text": "✏️ Change / Add / Options",
                                "callback_data": f"rs:options:{request_id}",
                            }
                        ],
                    ]
                },
            },
        )
    telegram(
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": (
                "When you have approved every 16:9 highlight you want, tap "
                "Schedule Now. Untouched highlights will be skipped."
            ),
            "reply_markup": {
                "inline_keyboard": [[{
                    "text": "📅 No More 16:9s — Schedule Now",
                    "callback_data": f"rs:schedule_now:{request_id}",
                }]]
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
        "📺 The 9:16 Shorts review is confirmed. Selecting the strongest standalone "
        "16:9 highlights now. Gaps are allowed; every highlight must be at least "
        "three minutes and contain a complete point, discussion, or story.",
    )
    suggestions = _topic_break_suggestions(segments)
    topics = _build_contiguous_topic_segments(segments, suggestions)
    if not topics:
        send(
            chat_id,
            "ℹ️ No standalone 16:9 highlight of at least three minutes met "
            "the quality and completeness requirements.",
        )
    topic_result = {"segments": topics, "selection": "best_standalone_highlights"}
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


def _start_16_9_after_confirmation(request_id: str, chat_id: str) -> None:
    try:
        with _LOCK, _telegram_db() as db:
            row = db.execute(
                "SELECT * FROM telegram_requests WHERE request_id=?", (request_id,)
            ).fetchone()
        if not row:
            raise RuntimeError(f"Ripped Shorts request not found: {request_id}")
        state = json.loads(row["state_json"])
        transcript_segments = state.get("topic_source_segments") or []
        video_path = Path(str(state.get("video_path") or ""))
        if not transcript_segments or not video_path.is_file():
            raise RuntimeError(
                "The saved source for 16:9 highlights is unavailable. Send /retry "
                f"{request_id} to restore the source."
            )
        parsed = state.get("parsed") or {}
        video_id = str(parsed.get("video_id") or parsed.get("drive_ids", [request_id])[0])
        _process_topics(
            request_id,
            state,
            chat_id,
            video_id,
            video_path,
            transcript_segments,
            bool(state.get("source_reused")),
        )
    except Exception as exc:
        logger.exception("16:9 highlight analysis failed after Shorts confirmation")
        with _LOCK, _telegram_db() as db:
            row = db.execute(
                "SELECT * FROM telegram_requests WHERE request_id=?", (request_id,)
            ).fetchone()
            if row:
                state = json.loads(row["state_json"])
                state["topic_stage"] = "failed"
                state["topic_error"] = str(exc)
                db.execute(
                    "UPDATE telegram_requests SET state_json=?, updated_at=? WHERE request_id=?",
                    (json.dumps(state), now(), request_id),
                )
        send(
            chat_id,
            f"❌ 16:9 highlight analysis failed\nJob ID: {request_id}\n"
            f"{str(exc)[:1200]}",
        )


def _process(request_id: str) -> None:
    logger.info("Ripped Shorts job starting request_id=%s", request_id)
    with _LOCK, _telegram_db() as db:
        row = db.execute("SELECT * FROM telegram_requests WHERE request_id=?", (request_id,)).fetchone()
    if not row:
        logger.error("Ripped Shorts job missing from database request_id=%s", request_id)
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
            state["vid_title"] = _drive_vid_title(meta)
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
                segments = _transcribe(video, progress=lambda message: send(chat_id, message))
            reused = True
        else:
            parsed = state["parsed"]
            video_id = parsed["video_id"]
            state["vid_title"] = _youtube_vid_title(parsed["source_value"])
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
                try:
                    audio_master_result = ingest_with_audio_master(
                        video_id,
                        parsed["source_value"],
                        progress=lambda message: send(chat_id, message),
                    )
                    state["audio_master_ingest"] = {
                        "job_id": audio_master_result.get("job_id"),
                        "status": audio_master_result.get("status"),
                        "skip_metadata_podhome": audio_master_result.get("skip_metadata_podhome"),
                    }
                except Exception as audio_master_error:
                    # Audio Master is a cache accelerator, not a single point of
                    # failure. Its retained-video path can fail independently
                    # while Ripped Shorts can still download/transcribe directly.
                    state["audio_master_ingest"] = {
                        "status": "fallback_to_ripped_shorts",
                        "error": str(audio_master_error),
                    }
                    state.setdefault("warnings", []).append(
                        f"Audio Master reusable-source path failed; using Ripped Shorts fallback: {audio_master_error}"
                    )
                    logger.warning(
                        "Audio Master source unavailable; continuing with direct Ripped Shorts ingestion "
                        "request_id=%s video_id=%s error=%s",
                        request_id,
                        video_id,
                        audio_master_error,
                    )
                    send(
                        chat_id,
                        "⚠️ Audio Master did not retain a usable source video. "
                        "Ripped Shorts is continuing with its direct backup path.",
                    )

                try:
                    cache = reuse_from_drive(video_id, work)
                except Exception as cache_error:
                    cache = {"video_path": None, "segments": [], "sermon_boundary": None}
                    state.setdefault("warnings", []).append(
                        f"Post-ingest Drive cache lookup failed: {cache_error}"
                    )
                video = cache.get("video_path") or video
                segments = cache.get("segments") or segments
                boundary = cache.get("sermon_boundary") or boundary

                if not video:
                    send(chat_id, "⬇️ Downloading the YouTube video with the Ripped Shorts backup path.")
                    video = download_youtube_resilient(video_id, parsed["source_value"], work)
                    reused = False
                if not segments:
                    send(chat_id, "📝 Creating the timed transcript with the Ripped Shorts backup path.")
                    segments = _transcribe(video, progress=lambda message: send(chat_id, message))
                    reused = False
                if not video or not segments:
                    raise RuntimeError(
                        "Neither Audio Master, Drive cache, nor the direct Ripped Shorts fallback "
                        f"produced a usable video and timed transcript for YouTube ID {video_id}."
                    )
                reused = bool(reused or cache.get("video_path") or cache.get("segments"))

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

        if row["mode"] == "topics":
            _process_topics(
                request_id,
                state,
                chat_id,
                video_id,
                video,
                segments,
                reused,
            )
            return
        if row["mode"] == "both":
            state.update(
                {
                    "video_path": str(video),
                    "source_reused": reused,
                    "topic_source_segments": segments,
                    "topic_stage": None,
                }
            )

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
                    _send_short_confirmation(chat_id, request_id)
                    return
                if rendered_clips or reuse_existing:
                    import main

                    durable = main.list_existing_rendered_assets(
                        video_id, _state_vid_title(state)
                    )
                    sheet_by_number = {
                        int(clip.get("candidate_number") or 0): clip
                        for clip in rendered_clips
                    }
                    recovered_shorts = []
                    short_reviews = {}
                    for index, asset in enumerate(durable.get("clips") or []):
                        number = int(asset["candidate_number"])
                        learned = sheet_by_number.get(number, {})
                        recovered_shorts.append(
                            {
                                **learned,
                                "candidate_number": number,
                                "clip_url": asset["clip_url"],
                            }
                        )
                        short_reviews[str(index)] = {
                            "status": "rendered",
                            "clip_url": asset["clip_url"],
                            "folder_id": asset.get("folder_id"),
                            "recovered_from_drive": True,
                        }

                    recovered_highlights = []
                    topic_reviews = {}
                    for index, asset in enumerate(durable.get("highlights") or []):
                        recovered_highlights.append(
                            {
                                "title": asset.get("title") or f"Highlight {index + 1}",
                                "summary": "",
                                "transcript": "",
                                "duration": 0,
                            }
                        )
                        topic_reviews[str(index)] = {
                            "status": "rendered",
                            "segment_url": asset["segment_url"],
                            "folder_id": asset.get("folder_id"),
                            "recovered_from_drive": True,
                        }

                    recovered_state = {
                        **state,
                        "stage": "already_rendered",
                        "video_path": str(video),
                        "source_reused": reused,
                        "result": {
                            "analysis": {
                                "content_type": "",
                                "main_theme": "",
                                "key_ideas": [],
                                "keywords": [],
                            },
                            "segments": recovered_shorts,
                        },
                        "candidate_reviews": short_reviews,
                        "topic_result": {
                            "segments": recovered_highlights,
                            "selection": "recovered_from_drive",
                        },
                        "topic_reviews": topic_reviews,
                        "topic_stage": (
                            "already_rendered" if recovered_highlights else "not_found"
                        ),
                        "recovered_renders_from_drive": True,
                    }
                    _save(request_id, "already_rendered", recovered_state)
                    send(
                        chat_id,
                        f"✅ Recovered {len(recovered_shorts)} rendered Short(s) and "
                        f"{len(recovered_highlights)} rendered 16:9 highlight(s) from "
                        "the existing Vid Title folder. GPT selection was not run again.",
                    )
                    _send_short_confirmation(chat_id, request_id)
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
        framework_prompt = _configured_short_framework_prompt()
        prompt_layers = [
            text
            for text in (
                framework_prompt,
                _brand_short_prompt(state),
                supplemental,
            )
            if str(text or "").strip()
        ]
        result = main.call_openai_for_clips(
            enriched,
            "\n\n".join(prompt_layers),
        )
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
        logger.exception("Ripped Shorts processing failed request_id=%s", request_id)
        # Never replace a durable candidate set with the older pre-result state
        # merely because Telegram could not deliver all review messages.
        with _LOCK, _telegram_db() as db:
            latest_row = db.execute(
                "SELECT state_json FROM telegram_requests WHERE request_id=?",
                (request_id,),
            ).fetchone()
        latest_state = json.loads(latest_row["state_json"]) if latest_row else state
        candidates_exist = bool(
            (latest_state.get("result") or {}).get("segments")
        )
        if candidates_exist:
            latest_state["notification_error"] = str(exc)
            latest_state["notification_retryable"] = True
            latest_state["stage"] = "awaiting_review"
            _save(request_id, "awaiting_review", latest_state)
            failure_text = (
                f"⚠️ Candidate review delivery was interrupted\nJob ID: {request_id}\n"
                f"{str(exc)[:1200]}\n\nYour candidates remain saved. "
                f"Send /retry {request_id} to resend the review."
            )
        else:
            latest_state.update(
                {
                    "stage": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "retryable": True,
                }
            )
            _save(request_id, "error", latest_state)
            failure_text = (
                f"❌ Processing failed\nJob ID: {request_id}\n{str(exc)[:1500]}"
                f"\n\nSend /retry {request_id} to try again."
            )
        try:
            send(chat_id, failure_text)
        except Exception:
            logger.exception(
                "Could not deliver Ripped Shorts processing failure notice request_id=%s",
                request_id,
            )




def _latest_editable_request(chat_id: str, user_id: str) -> sqlite3.Row | None:
    """Return the newest request that currently has reviewable sections."""
    with _LOCK, _telegram_db() as db:
        rows = db.execute(
            "SELECT * FROM telegram_requests WHERE chat_id=? AND user_id=? "
            "ORDER BY updated_at DESC LIMIT 25",
            (chat_id, user_id),
        ).fetchall()
    for row in rows:
        state = json.loads(row["state_json"])
        if (
            isinstance(state.get("service_segments"), list)
            or isinstance((state.get("topic_result") or {}).get("segments"), list)
            or isinstance((state.get("result") or {}).get("segments"), list)
        ):
            return row
    return None


def _log_quick_edit(
    row: sqlite3.Row,
    result: dict[str, Any],
    instruction: str,
    user_id: str,
) -> None:
    """Append an applied Telegram edit to Podcast / Decision Log."""
    try:
        import main

        state = result["state"]
        parsed = state.get("parsed") or {}
        before = result.get("before") or []
        after = result.get("after") or []
        first_before = before[0] if before else {}
        last_before = before[-1] if before else {}
        first_after = after[0] if after else {}
        last_after = after[-1] if after else {}
        decision_time = now()
        ai_start = first_before.get("start", first_before.get("start_seconds", ""))
        ai_end = last_before.get("end", last_before.get("end_seconds", ""))
        final_start = first_after.get("start", first_after.get("start_seconds", ""))
        final_end = last_after.get("end", last_after.get("end_seconds", ""))
        values = [
            f"{row['request_id']}:{decision_time}",
            row["request_id"],
            parsed.get("video_id", ""),
            result.get("lane", "service"),
            ai_start,
            ai_end,
            final_start,
            final_end,
            (float(final_start) - float(ai_start)) if ai_start != "" and final_start != "" else "",
            (float(final_end) - float(ai_end)) if ai_end != "" and final_end != "" else "",
            "",
            "",
            instruction,
            decision_time,
            user_id,
            "yes",
            "", "", "", "",
            result.get("action", ""),
            instruction,
            json.dumps(before, separators=(",", ":"))[:20000],
            json.dumps(after, separators=(",", ":"))[:20000],
            decision_time,
            user_id,
            "telegram",
            "yes",
            "yes",
            "v1",
        ]
        _, _, sheets = main.get_google_services()
        sheets.spreadsheets().values().append(
            spreadsheetId=RIPPED_LOG_SHEET_ID,
            range="'Decision Log'!A:AD",
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": [values]},
        ).execute()
    except Exception:
        logger.exception("Could not write Telegram edit learning to Podcast / Decision Log")


def _boundary_learning_prompt(limit: int = 50) -> str:
    """Summarize recent applied boundary edits for later section selection."""
    try:
        import main

        _, _, sheets = main.get_google_services()
        response = sheets.spreadsheets().values().get(
            spreadsheetId=RIPPED_LOG_SHEET_ID,
            range="'Decision Log'!A:AD",
        ).execute()
        rows = response.get("values", [])[1:]
        examples = []
        for values in reversed(rows):
            padded = list(values) + [""] * (30 - len(values))
            if str(padded[28]).strip().lower() != "yes":
                continue
            instruction = str(padded[21]).strip()
            action = str(padded[20]).strip()
            section_type = str(padded[3]).strip()
            if instruction or action:
                examples.append(
                    f"- {section_type}: {instruction or action} (action={action})"
                )
            if len(examples) >= limit:
                break
        if not examples:
            return ""
        return (
            "\n\nDEREK'S PRIOR BOUNDARY EDITS\n"
            "Use these as preferences, not absolute timestamp rules. Favor the kinds "
            "of complete openings, endings, additions, and removals Derek previously "
            "requested:\n" + "\n".join(examples)
        )
    except Exception:
        logger.exception("Could not read boundary learning from Podcast / Decision Log")
        return ""


def _telegram_message_text(message: dict[str, Any]) -> str:
    """Return visible text plus URLs Telegram stores only in message entities."""
    text = str(message.get("text") or message.get("caption") or "").strip()
    links: list[str] = []
    entity_sets = (
        (message.get("entities") or [], str(message.get("text") or "")),
        (message.get("caption_entities") or [], str(message.get("caption") or "")),
    )
    for entities, source in entity_sets:
        for entity in entities:
            entity_type = str(entity.get("type") or "")
            if entity_type == "text_link" and entity.get("url"):
                links.append(str(entity["url"]))
            elif entity_type == "url":
                try:
                    offset = int(entity.get("offset", 0))
                    length = int(entity.get("length", 0))
                    links.append(source[offset : offset + length])
                except (TypeError, ValueError):
                    continue
    return " ".join(part for part in [text, *links] if part).strip()


def _accept_update(
    update: dict,
    background_tasks: BackgroundTasks,
    *,
    trusted_source: bool = False,
) -> dict:
    callback = update.get("callback_query") or {}
    message = (
        callback.get("message")
        or update.get("message")
        or update.get("edited_message")
        or update.get("channel_post")
        or update.get("edited_channel_post")
        or {}
    )
    chat_id = str((message.get("chat") or {}).get("id", ""))
    user_id = str((callback.get("from") or message.get("from") or {}).get("id", ""))
    callback_data = str(callback.get("data") or "")
    text = _telegram_message_text(message)
    if not chat_id or not user_id:
        return {"status": "ignored"}
    if not trusted_source and not _authorized(chat_id, user_id):
        return {"status": "unauthorized"}
    if text and not callback_data:
        with _LOCK, _telegram_db() as db:
            rows = db.execute(
                "SELECT * FROM telegram_requests WHERE chat_id=? AND user_id=? "
                "ORDER BY updated_at DESC LIMIT 20",
                (chat_id, user_id),
            ).fetchall()
            edit_row = None
            edit_state = None
            for candidate_row in rows:
                candidate_state = json.loads(candidate_row["state_json"])
                waiting = candidate_state.get("awaiting_copy_input") or {}
                if str(waiting.get("user_id") or "") == user_id:
                    edit_row, edit_state = candidate_row, candidate_state
                    break
            if edit_row and edit_state:
                waiting = edit_state.pop("awaiting_copy_input")
                index = int(waiting["index"])
                field = str(waiting["field"])
                drafts = list(edit_state.get("copy_drafts") or [])
                if index >= len(drafts):
                    return {"status": "copy_draft_not_found"}
                key = {
                    "caption": "social_caption",
                    "title": "video_title",
                    "description": "video_description",
                }[field]
                drafts[index][key] = text
                drafts[index]["user_edited"] = True
                drafts[index]["copy_status"] = "edited"
                drafts[index]["edited_fields"] = sorted(
                    set(drafts[index].get("edited_fields") or []) | {field}
                )
                edit_state["copy_drafts"] = drafts
                db.execute(
                    "UPDATE telegram_requests SET state_json=?, updated_at=? "
                    "WHERE request_id=?",
                    (json.dumps(edit_state), now(), edit_row["request_id"]),
                )
                request_id = edit_row["request_id"]
                send(
                    chat_id,
                    f"✅ Your {field} replaced the AI draft and will be learned.\n\n{text[:3200]}",
                )
                telegram("sendMessage", {
                    "chat_id": chat_id,
                    "text": "Make another edit above or finish the copy review.",
                    "reply_markup": {"inline_keyboard": [[{
                        "text": "✅ Approve Copy & Schedule",
                        "callback_data": f"rs:copy_finish:{request_id}",
                    }]]},
                })
                return {
                    "status": "copy_updated",
                    "request_id": request_id,
                    "field": field,
                    "asset_index": index,
                }

    options_choice = re.fullmatch(
        r"rs:options:([A-Za-z0-9-]+)", callback_data
    )
    if options_choice:
        send(chat_id, OPTIONS_TEXT)
        return {
            "status": "quick_edit_help",
            "request_id": options_choice.group(1),
        }
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

    schedule_now = re.fullmatch(
        r"rs:schedule_now:([A-Za-z0-9-]+)", callback_data
    )
    if schedule_now:
        request_id = schedule_now.group(1)
        with _LOCK, _telegram_db() as db:
            row = db.execute(
                "SELECT * FROM telegram_requests WHERE request_id=?", (request_id,)
            ).fetchone()
            if not row:
                return {"status": "not_found"}
            state = json.loads(row["state_json"])
            if state.get("schedule_requested_at"):
                return {"status": "already_scheduling", "request_id": request_id}
            if state.get("copy_review_requested_at"):
                return {"status": "copy_review_pending", "request_id": request_id}

            clips = (state.get("result") or {}).get("segments", [])
            short_reviews = dict(state.get("candidate_reviews") or {})
            decided = {
                "queued", "rendering", "rendered", "render_failed", "reject", "rejected"
            }
            skipped_shorts = 0
            for index in range(len(clips)):
                if str((short_reviews.get(str(index)) or {}).get("status") or "") not in decided:
                    short_reviews[str(index)] = {
                        "status": "rejected",
                        "reviewed_at": now(),
                        "user_id": user_id,
                        "selection_complete_skip": True,
                    }
                    skipped_shorts += 1
            state["candidate_reviews"] = short_reviews
            state["shorts_confirmed_at"] = state.get("shorts_confirmed_at") or now()
            state["short_selection_completed_at"] = (
                state.get("short_selection_completed_at") or now()
            )

            topics = (state.get("topic_result") or {}).get("segments", [])
            topic_reviews = dict(state.get("topic_reviews") or {})
            skipped_topics = 0
            for index in range(len(topics)):
                if str((topic_reviews.get(str(index)) or {}).get("status") or "") not in decided:
                    topic_reviews[str(index)] = {
                        "status": "rejected",
                        "reviewed_at": now(),
                        "user_id": user_id,
                        "selection_complete_skip": True,
                    }
                    skipped_topics += 1
            state["topic_reviews"] = topic_reviews
            state["topic_selection_completed_at"] = now()
            state["topic_stage"] = "selection_complete"
            state["copy_review_requested_at"] = now()
            db.execute(
                "UPDATE telegram_requests SET status=?, state_json=?, updated_at=? "
                "WHERE request_id=?",
                ("awaiting_copy_review", json.dumps(state), now(), request_id),
            )

        draft_inputs = _copy_review_assets(state, request_id)
        drafts = _generate_schedule_copy(
            draft_inputs,
            str(state.get("show_id") or ""),
            _state_vid_title(state),
            str((state.get("parsed") or {}).get("source_value") or ""),
        )
        for draft in drafts:
            draft["ai_social_caption"] = draft.get("social_caption", "")
            draft["ai_video_title"] = draft.get("video_title", "")
            draft["ai_video_description"] = draft.get("video_description", "")
            draft["user_edited"] = False
        with _LOCK, _telegram_db() as db:
            row = db.execute(
                "SELECT state_json FROM telegram_requests WHERE request_id=?",
                (request_id,),
            ).fetchone()
            latest_state = json.loads(row["state_json"])
            latest_state["copy_drafts"] = drafts
            db.execute(
                "UPDATE telegram_requests SET state_json=?, updated_at=? WHERE request_id=?",
                (json.dumps(latest_state), now(), request_id),
            )
        send(
            chat_id,
            f"📅 Video selection closed. Skipped {skipped_shorts} untouched Short(s) "
            f"and {skipped_topics} untouched 16:9 highlight(s). Now review your "
            "captions, titles, and descriptions.",
        )
        _send_copy_review(chat_id, request_id, drafts)
        return {"status": "copy_review_requested", "request_id": request_id}

    copy_action = re.fullmatch(
        r"rs:copy_(keep|regenerate):([A-Za-z0-9-]+):(\d+)",
        callback_data,
    )
    if copy_action:
        action, request_id, index_text = copy_action.groups()
        index = int(index_text)
        with _LOCK, _telegram_db() as db:
            row = db.execute(
                "SELECT * FROM telegram_requests WHERE request_id=?", (request_id,)
            ).fetchone()
            if not row:
                return {"status": "not_found"}
            state = json.loads(row["state_json"])
            drafts = list(state.get("copy_drafts") or [])
            if index >= len(drafts):
                return {"status": "copy_draft_not_found"}
            if action == "keep":
                drafts[index]["copy_status"] = "approved"
                state["copy_drafts"] = drafts
                db.execute(
                    "UPDATE telegram_requests SET state_json=?, updated_at=? WHERE request_id=?",
                    (json.dumps(state), now(), request_id),
                )
                send(
                    chat_id,
                    f"✅ Draft {index + 1} approved. You can review another draft or finish scheduling.",
                )
                return {"status": "copy_approved", "request_id": request_id, "asset_index": index}
            drafts[index]["copy_status"] = "regenerating"
            state["copy_drafts"] = drafts
            db.execute(
                "UPDATE telegram_requests SET state_json=?, updated_at=? WHERE request_id=?",
                (json.dumps(state), now(), request_id),
            )
        send(chat_id, f"🔄 Creating a fresh version of draft {index + 1}…")
        background_tasks.add_task(
            _regenerate_copy_draft, request_id, index, chat_id
        )
        return {"status": "copy_regenerating", "request_id": request_id, "asset_index": index}

    copy_edit = re.fullmatch(
        r"rs:copy_edit_(caption|title|description):([A-Za-z0-9-]+):(\d+)",
        callback_data,
    )
    if copy_edit:
        field, request_id, index_text = copy_edit.groups()
        index = int(index_text)
        with _LOCK, _telegram_db() as db:
            row = db.execute(
                "SELECT * FROM telegram_requests WHERE request_id=?", (request_id,)
            ).fetchone()
            if not row:
                return {"status": "not_found"}
            state = json.loads(row["state_json"])
            drafts = list(state.get("copy_drafts") or [])
            if index >= len(drafts):
                return {"status": "copy_draft_not_found"}
            state["awaiting_copy_input"] = {
                "field": field,
                "index": index,
                "user_id": user_id,
            }
            db.execute(
                "UPDATE telegram_requests SET state_json=?, updated_at=? WHERE request_id=?",
                (json.dumps(state), now(), request_id),
            )
        telegram(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": (
                    f"Reply to this message with your replacement {field}. "
                    "It will replace this draft and become a learning example."
                ),
                "reply_markup": {
                    "force_reply": True,
                    "selective": True,
                    "input_field_placeholder": f"Write replacement {field}",
                },
            },
        )
        return {"status": "awaiting_copy_input", "request_id": request_id, "field": field}

    copy_finish = re.fullmatch(
        r"rs:copy_finish:([A-Za-z0-9-]+)", callback_data
    )
    if copy_finish:
        request_id = copy_finish.group(1)
        with _LOCK, _telegram_db() as db:
            row = db.execute(
                "SELECT * FROM telegram_requests WHERE request_id=?", (request_id,)
            ).fetchone()
            if not row:
                return {"status": "not_found"}
            state = json.loads(row["state_json"])
            if state.get("schedule_requested_at"):
                return {"status": "already_scheduling", "request_id": request_id}
            drafts = list(state.get("copy_drafts") or [])
            if not drafts:
                return {"status": "copy_drafts_missing", "request_id": request_id}
            state.pop("awaiting_copy_input", None)
            state["copy_review_completed_at"] = now()
            state["schedule_requested_at"] = now()
            db.execute(
                "UPDATE telegram_requests SET status=?, state_json=?, updated_at=? "
                "WHERE request_id=?",
                ("awaiting_render_completion", json.dumps(state), now(), request_id),
            )
        _log_copy_learning(request_id, state, drafts, user_id)
        send(
            chat_id,
            "✅ Copy approved. Schedule Master will receive the batch as soon as "
            "every approved render finishes.",
        )
        _notify_render_queue_complete(request_id, chat_id)
        return {"status": "schedule_requested", "request_id": request_id}

    shorts_confirm = re.fullmatch(
        r"rs:shorts_confirm:([A-Za-z0-9-]+)", callback_data
    )
    if shorts_confirm:
        request_id = shorts_confirm.group(1)
        with _LOCK, _telegram_db() as db:
            row = db.execute(
                "SELECT * FROM telegram_requests WHERE request_id=?", (request_id,)
            ).fetchone()
            if not row:
                return {"status": "not_found"}
            state = json.loads(row["state_json"])
            start_highlights = row["mode"] == "both"
            if state.get("shorts_confirmed_at"):
                return {"status": "already_confirmed", "request_id": request_id}
            clips = (state.get("result") or {}).get("segments", [])
            reviews = dict(state.get("candidate_reviews") or {})
            decided_statuses = {
                "queued", "rendering", "rendered", "render_failed", "reject", "rejected"
            }
            pending = [
                index + 1
                for index in range(len(clips))
                if str((reviews.get(str(index)) or {}).get("status") or "")
                not in decided_statuses
            ]
            for index in pending:
                reviews[str(index - 1)] = {
                    "status": "rejected",
                    "reviewed_at": now(),
                    "user_id": user_id,
                    "selection_complete_skip": True,
                }
            state["candidate_reviews"] = reviews
            state["shorts_confirmed_at"] = now()
            state["short_selection_completed_at"] = now()
            if start_highlights:
                state["topic_stage"] = "queued"
            else:
                state["schedule_requested_at"] = now()
            db.execute(
                "UPDATE telegram_requests SET status=?, state_json=?, updated_at=? "
                "WHERE request_id=?",
                (
                    "processing_16_9" if start_highlights else "awaiting_render_completion",
                    json.dumps(state),
                    now(),
                    request_id,
                ),
            )
        send(
            chat_id,
            (
                f"✅ You picked your Shorts. {len(pending)} untouched candidate(s) skipped. "
                + (
                    "Starting 16:9 highlight analysis and preparing the rendered "
                    "Shorts for Schedule Master."
                    if start_highlights
                    else "Preparing the rendered Shorts for Schedule Master."
                )
            ),
        )
        _notify_render_queue_complete(request_id, chat_id)
        if start_highlights:
            RENDER_EXECUTOR.submit(
                _start_16_9_after_confirmation, request_id, chat_id
            )
        return {
            "status": "shorts_confirmed",
            "request_id": request_id,
            "next_stage": (
                "16_9_highlights" if start_highlights else "schedule_master"
            ),
        }

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
    resume_match = re.fullmatch(
        r"/resume(?:@rippedshortsbot)?(?:\s+latest)?", text, re.I
    )
    if resume_match:
        try:
            durable = latest_incomplete(
                RIPPED_LOG_SHEET_ID,
                chat_id=chat_id,
                user_id=user_id,
            )
        except Exception as exc:
            logger.exception("Could not read durable Workflow Jobs ledger")
            send(chat_id, f"❌ Resume lookup failed: {str(exc)[:800]}")
            return {"status": "resume_lookup_failed"}
        if not durable:
            send(chat_id, "There is no unfinished Ripped Shorts job to resume.")
            return {"status": "no_unfinished_job"}
        request_id = str(durable.get("job_id") or "")
        with _LOCK, _telegram_db() as db:
            existing = db.execute(
                "SELECT request_id FROM telegram_requests WHERE request_id=?",
                (request_id,),
            ).fetchone()
        if not existing:
            source_url = str(durable.get("source_url") or "").strip()
            try:
                parsed = parse_request(source_url)
            except ValueError:
                send(
                    chat_id,
                    "⚠️ The latest durable job has no usable source URL. "
                    "Paste the original YouTube URL again.",
                )
                return {
                    "status": "resume_source_missing",
                    "request_id": request_id,
                }
            recorded_mode = str(durable.get("mode") or "").strip()
            if recorded_mode in {"shorts", "topics", "both"}:
                parsed["mode"] = recorded_mode
            stamp = now()
            recovered_state = {
                "stage": "accepted",
                "parsed": parsed,
                "show_id": str(durable.get("show_code") or ""),
                "attempt_count": int(durable.get("attempt_count") or 0) + 1,
                "resumed_from_workflow_jobs_at": stamp,
            }
            with _LOCK, _telegram_db() as db:
                db.execute(
                    "INSERT INTO telegram_requests VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        request_id,
                        str(update.get("update_id") or f"resume-{uuid.uuid4()}"),
                        chat_id,
                        user_id,
                        "accepted",
                        parsed["mode"],
                        parsed["source_kind"],
                        parsed["source_value"],
                        json.dumps(recovered_state),
                        stamp,
                        stamp,
                    ),
                )
            _save(request_id, "accepted", recovered_state)
        send(
            chat_id,
            f"▶️ Resuming the latest unfinished job\nJob ID: {request_id}",
        )
        background_tasks.add_task(_process, request_id)
        return {"status": "resume_accepted", "request_id": request_id}

    retry_match = re.fullmatch(
        r"/retry(?:@rippedshortsbot)?\s+([A-Za-z0-9-]+)", text, re.I
    )
    if retry_match:
        request_id = retry_match.group(1)
        with _LOCK, _telegram_db() as db:
            retry_row = db.execute(
                "SELECT request_id FROM telegram_requests WHERE request_id=?",
                (request_id,),
            ).fetchone()
        if not retry_row:
            send(
                chat_id,
                "⚠️ That job belonged to an earlier Railway container and its "
                "interactive record is no longer available. Paste the original "
                "YouTube URL again to create a recoverable new job. Ripped Shorts "
                "will check Drive for the existing video/transcript instead of "
                "starting the media work from scratch.",
            )
            return {
                "status": "retry_source_missing",
                "request_id": request_id,
                "next_action": "resend_original_youtube_url",
            }
        background_tasks.add_task(_process, request_id)
        return {"status": "retry_accepted", "request_id": request_id}
    if is_quick_command(text):
        row = _latest_editable_request(chat_id, user_id)
        if not row:
            send(chat_id, "There are no sections ready to edit yet.")
            return {"status": "no_editable_sections"}
        state = json.loads(row["state_json"])
        try:
            result = apply_quick_command(state, text)
        except ValueError as exc:
            send(chat_id, str(exc))
            return {"status": "quick_edit_invalid", "detail": str(exc)}
        if result.get("changed"):
            if result.get("action") == "approve_all":
                lane = result.get("lane")
                if lane == "short":
                    reviews = dict(result["state"].get("candidate_reviews") or {})
                    for index in range(len(result.get("after") or [])):
                        reviews[str(index)] = {
                            "status": "queued",
                            "reviewed_at": now(),
                            "user_id": user_id,
                        }
                    result["state"]["candidate_reviews"] = reviews
                elif lane == "topic":
                    reviews = dict(result["state"].get("topic_reviews") or {})
                    for index in range(len(result.get("after") or [])):
                        reviews[str(index)] = {
                            "status": "queued",
                            "reviewed_at": now(),
                            "user_id": user_id,
                        }
                    result["state"]["topic_reviews"] = reviews
            _save(row["request_id"], row["status"], result["state"])
            _log_quick_edit(row, result, text, user_id)
            if result.get("action") == "approve_all":
                if result.get("lane") == "short":
                    for index in range(len(result.get("after") or [])):
                        RENDER_EXECUTOR.submit(
                            _render_approved, row["request_id"], index, chat_id
                        )
                elif result.get("lane") == "topic":
                    for index in range(len(result.get("after") or [])):
                        RENDER_EXECUTOR.submit(
                            _render_topic_approved, row["request_id"], index, chat_id
                        )
        send(chat_id, result["message"])
        return {
            "status": "quick_edit_applied" if result.get("changed") else "quick_edit_help",
            "request_id": row["request_id"],
            "action": result.get("action"),
        }
    try:
        parsed = parse_request(text)
    except ValueError:
        command = text.split(None, 1)[0][:80] if text else ""
        logger.info(
            "Ripped Shorts ignored group message command=%r has_youtube_entity=%s "
            "message_type=%s",
            command,
            bool(YOUTUBE_RE.search(text)),
            "callback" if callback_data else "text",
        )
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
    logger.info(
        "Ripped Shorts job queued request_id=%s update_id=%s source_kind=%s mode=%s",
        request_id,
        update_id,
        parsed["source_kind"],
        parsed["mode"],
    )
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
        vid_title = _state_vid_title(state)
        import main

        rendered = main.attach_topic_segment_asset(
            dict(segment),
            video_id,
            video,
            index + 1,
            vid_title=vid_title,
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
                "folder_id": rendered.get("folder_id"),
                "rendered_at": now(),
            }
            latest_state["topic_reviews"] = reviews
            db.execute(
                "UPDATE telegram_requests SET state_json=?, updated_at=? WHERE request_id=?",
                (json.dumps(latest_state), now(), request_id),
            )
        send(
            chat_id,
            f"✅ 16:9 Segment {index + 1} rendered and uploaded to the Vid Title folder ({vid_title}):\n"
            f"{rendered.get('segment_url', '')}",
        )
        _notify_render_queue_complete(request_id, chat_id)
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
        _notify_render_queue_complete(request_id, chat_id)


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
        vid_title = _state_vid_title(state)
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
            vid_title=vid_title,
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
                "folder_id": clip.get("folder_id"),
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
            f"✅ Short {index + 1} rendered and uploaded to the Vid Title folder ({vid_title}):\n"
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


def _readiness() -> dict[str, Any]:
    return readiness_snapshot(
        DB_PATH,
        _ripped_webhook_url(),
        bool(_ripped_bot_token()),
        bool(
            os.getenv("TELEGRAM_CHAT_ID")
            or os.getenv("TELEGRAM_GROUP_CHAT_ID")
            or os.getenv("Telegram_Group_Chat_ID")
        ),
        str(
            os.getenv("SCHEDULE_MASTER_INTERNAL_URL")
            or os.getenv("SCHEDULE_MASTER_URL")
            or ""
        ).strip(),
    )


@router.get("/api/ripped-shorts/readiness")
def ripped_shorts_readiness() -> dict[str, Any]:
    return _readiness()


@router.get("/api/ripped-shorts/smoke-test")
def ripped_shorts_smoke_test() -> dict[str, Any]:
    snapshot = _readiness()
    telegram_check: dict[str, Any] = {
        "bot_identity": False,
        "webhook_owned": False,
    }
    try:
        token = _ripped_bot_token()
        me = dict(_telegram_api(token, "getMe").get("result") or {})
        info = dict(_telegram_api(token, "getWebhookInfo").get("result") or {})
        telegram_check = {
            "bot_identity": str(me.get("username") or "").lower()
            == "rippedshortsbot",
            "webhook_owned": str(info.get("url") or "")
            == _ripped_webhook_url(),
            "pending_updates": int(info.get("pending_update_count") or 0),
            "group_read_all": bool(me.get("can_read_all_group_messages")),
        }
    except Exception as exc:
        telegram_check["error_class"] = classify_error(exc)
        telegram_check["error"] = str(exc)[:800]
    snapshot["telegram"] = telegram_check
    snapshot["ready"] = bool(
        snapshot.get("ready")
        and telegram_check.get("bot_identity")
        and telegram_check.get("webhook_owned")
    )
    return snapshot


@router.post("/api/telegram/webhook")
async def telegram_gateway(request: Request, x_telegram_bot_api_secret_token: str | None = Header(None)):
    """Clip Master owns Telegram and forwards only Ripped Shorts messages."""
    if os.getenv("SERVICE_ROLE", "").strip().lower() != "clip_master":
        raise HTTPException(status_code=404, detail="Telegram webhook is owned by Clip Master")
    expected = os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()
    if not expected or x_telegram_bot_api_secret_token != expected:
        raise HTTPException(status_code=401, detail="Invalid Telegram webhook secret")
    update = await request.json()
    callback = update.get("callback_query") or {}
    callback_data = str(callback.get("data") or "")
    message = callback.get("message") or update.get("message") or update.get("edited_message") or {}
    chat_id = str((message.get("chat") or {}).get("id", ""))
    user_id = str((callback.get("from") or message.get("from") or {}).get("id", ""))
    text = str(message.get("text") or message.get("caption") or "").strip()

    # Explicit Clip Master buttons always remain local. For ordinary text, first
    # ask Ripped Shorts whether Write My Caption/Edit Metadata is waiting. This
    # prevents captions beginning with words such as "add" from being mistaken
    # for Clip Master boundary commands.
    if callback_data.startswith("cm:") and clipmaster_claims_update(update):
        return handle_clipmaster_update(update)
    target = os.getenv("RIPPED_SHORTS_INTERNAL_URL", "").rstrip("/")
    secret = os.getenv("RIPPED_SHORTS_SHARED_SECRET", "").strip()
    if text and not callback_data and target and secret:
        try:
            pending_response = requests.post(
                f"{target}/api/ripped-shorts/pending-copy-input",
                json={"chat_id": chat_id, "user_id": user_id},
                headers={"x-ripped-shorts-secret": secret},
                timeout=(5, 20),
            )
            pending_response.raise_for_status()
            if bool(pending_response.json().get("pending")):
                response = requests.post(
                    f"{target}/api/ripped-shorts/intake",
                    json=update,
                    headers={"x-ripped-shorts-secret": secret},
                    timeout=(10, 60),
                )
                if response.status_code >= 400:
                    raise HTTPException(
                        status_code=502,
                        detail=f"Ripped Shorts caption intake failed: {response.text[:1000]}",
                    )
                return response.json()
        except HTTPException:
            raise
        except Exception:
            logger.exception("Could not check pending Ripped Shorts copy input")

    if clipmaster_claims_update(update):
        return handle_clipmaster_update(update)
    if not chat_id or not user_id:
        raise HTTPException(status_code=422, detail="Telegram update is missing chat or user identity")
    if not _authorized(chat_id, user_id):
        raise HTTPException(status_code=403, detail="Telegram chat or user is not authorized")
    if not target or not secret:
        raise HTTPException(status_code=503, detail="Ripped Shorts forwarding is not configured")
    response = requests.post(f"{target}/api/ripped-shorts/intake", json=update, headers={"x-ripped-shorts-secret": secret}, timeout=(10, 60))
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Ripped Shorts intake failed: {response.text[:1000]}")
    return response.json()


@router.post("/api/ripped-shorts/pending-copy-input")
async def pending_copy_input(
    request: Request,
    x_ripped_shorts_secret: str | None = Header(None),
):
    """Tell the Telegram gateway whether the next plain message is copy input."""
    expected = os.getenv("RIPPED_SHORTS_SHARED_SECRET", "").strip()
    if not expected or x_ripped_shorts_secret != expected:
        raise HTTPException(status_code=401, detail="Invalid Ripped Shorts service secret")
    payload = await request.json()
    chat_id = str(payload.get("chat_id") or "")
    user_id = str(payload.get("user_id") or "")
    if not chat_id or not user_id:
        return {"pending": False}
    with _LOCK, _telegram_db() as db:
        rows = db.execute(
            "SELECT state_json FROM telegram_requests WHERE chat_id=? AND user_id=? "
            "ORDER BY updated_at DESC LIMIT 20",
            (chat_id, user_id),
        ).fetchall()
    for row in rows:
        state = json.loads(row["state_json"])
        waiting = state.get("awaiting_copy_input") or {}
        if str(waiting.get("user_id") or "") == user_id:
            return {
                "pending": True,
                "request_id": str(waiting.get("request_id") or ""),
                "field": str(waiting.get("field") or ""),
            }
    return {"pending": False}


@router.post("/api/ripped-shorts/intake")
async def internal_intake(
    request: Request,
    background_tasks: BackgroundTasks,
    response: Response,
    x_ripped_shorts_secret: str | None = Header(None),
):
    expected = os.getenv("RIPPED_SHORTS_SHARED_SECRET", "").strip()
    if not expected or x_ripped_shorts_secret != expected:
        raise HTTPException(status_code=401, detail="Invalid Ripped Shorts service secret")
    update = await request.json()
    result = _accept_update(update, background_tasks, trusted_source=True)
    status = str(result.get("status") or "unknown")
    logger.info(
        "Ripped Shorts intake result update_id=%s status=%s request_id=%s",
        update.get("update_id", ""),
        status,
        result.get("request_id", ""),
    )
    if status == "ignored":
        raise HTTPException(status_code=422, detail="Telegram update is missing chat or user identity")
    if status == "unauthorized":
        callback = update.get("callback_query") or {}
        message = (
            callback.get("message")
            or update.get("message")
            or update.get("edited_message")
            or {}
        )
        rejected_chat_id = str((message.get("chat") or {}).get("id", ""))
        rejected_user_id = str(
            (callback.get("from") or message.get("from") or {}).get("id", "")
        )
        logger.warning(
            "Ripped Telegram authorization rejected chat_id=%s user_id=%s; "
            "set Telegram_Group_Chat_ID to this chat_id or add the user_id to "
            "TELEGRAM_ALLOWED_USER_IDS",
            rejected_chat_id,
            rejected_user_id,
        )
        raise HTTPException(status_code=403, detail="Telegram chat or user is not authorized")
    if status in {"accepted", "retry_accepted"}:
        response.status_code = 202
    return result


def _ripped_webhook_secret() -> str:
    for name in (
        "RIPPED_SHORTS_TELEGRAM_WEBHOOK_SECRET",
        "TELEGRAM_RIPPED_WEBHOOK_SECRET",
        "TELEGRAM_WEBHOOK_SECRET",
    ):
        value = os.getenv(name, "").strip()
        if value and re.fullmatch(r"[A-Za-z0-9_-]{1,256}", value):
            return value
        if value:
            logger.warning("%s contains characters Telegram does not allow; webhook will be registered without a secret header", name)
    return ""


@router.get("/api/ripped-shorts/telegram/webhook")
def ripped_telegram_webhook_health() -> dict[str, Any]:
    """Allow Railway/browser health checks without pretending to be Telegram."""
    return {
        "status": "ok",
        "bot": "@rippedshortsbot",
        "delivery_method": "POST",
    }


class _ExecutorBackgroundTasks:
    """Submit follow-on work when an update is processed after webhook ACK."""

    @staticmethod
    def add_task(function: Any, *args: Any, **kwargs: Any) -> None:
        RENDER_EXECUTOR.submit(function, *args, **kwargs)


def _process_ripped_telegram_update_after_ack(update: dict[str, Any]) -> None:
    try:
        result = _accept_update(
            update,
            _ExecutorBackgroundTasks(),
            trusted_source=False,
        )
        logger.info(
            "Ripped Shorts Telegram update processed after ACK "
            "update_id=%s status=%s request_id=%s",
            update.get("update_id", ""),
            result.get("status", "unknown"),
            result.get("request_id", ""),
        )
    except Exception:
        logger.exception(
            "Ripped Shorts Telegram update failed after ACK update_id=%s",
            update.get("update_id", ""),
        )


@router.post("/api/ripped-shorts/telegram/webhook", status_code=202)
async def ripped_telegram_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_telegram_bot_api_secret_token: str | None = Header(None),
):
    """Acknowledge Telegram immediately; process conversation work afterward."""
    expected = _ripped_webhook_secret()
    if expected and x_telegram_bot_api_secret_token != expected:
        raise HTTPException(status_code=401, detail="Invalid Telegram webhook secret")
    update = await request.json()

    callback = update.get("callback_query") or {}
    message = (
        callback.get("message")
        or update.get("message")
        or update.get("edited_message")
        or update.get("channel_post")
        or update.get("edited_channel_post")
        or {}
    )
    chat_id = str((message.get("chat") or {}).get("id", ""))
    user_id = str((callback.get("from") or message.get("from") or {}).get("id", ""))
    if not chat_id or not user_id:
        raise HTTPException(
            status_code=422,
            detail="Telegram update is missing chat or user identity",
        )
    if not _authorized(chat_id, user_id):
        logger.warning(
            "Ripped Telegram authorization rejected before ACK "
            "chat_id=%s user_id=%s",
            chat_id,
            user_id,
        )
        raise HTTPException(
            status_code=403,
            detail="Telegram chat or user is not authorized",
        )

    background_tasks.add_task(_process_ripped_telegram_update_after_ack, update)
    logger.info(
        "Ripped Shorts Telegram webhook ACK update_id=%s",
        update.get("update_id", ""),
    )
    return {
        "status": "accepted_for_processing",
        "update_id": update.get("update_id", ""),
    }


_RIPPED_WEBHOOK_WATCHDOG_STARTED = False
_RIPPED_WEBHOOK_WATCHDOG_LOCK = threading.Lock()


def _ripped_bot_token() -> str:
    return next(
        (
            os.getenv(name, "").strip()
            for name in (
                "RIPPED_SHORTS_TELEGRAM_BOT_TOKEN",
                "TELEGRAM_RIPPED_BOT_TOKEN",
                "Telegram_ripped_bot_token",
            )
            if os.getenv(name, "").strip()
        ),
        "",
    )


def _ripped_webhook_url() -> str:
    domain = (
        os.getenv("RIPPED_SHORTS_PUBLIC_URL")
        or os.getenv("RAILWAY_PUBLIC_DOMAIN")
        or ""
    ).strip().rstrip("/")
    if domain and not domain.startswith(("http://", "https://")):
        domain = "https://" + domain
    return domain + "/api/ripped-shorts/telegram/webhook" if domain else ""


def _telegram_api(token: str, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call Telegram and reject HTTP-200 responses whose JSON says ok=false."""
    response = requests.post(
        f"https://api.telegram.org/bot{token}/{method}",
        json=payload or {},
        timeout=(10, 30),
    )
    response.raise_for_status()
    data = response.json()
    if not data.get("ok"):
        raise RuntimeError(
            f"Telegram {method} rejected the request: "
            f"{data.get('error_code', 'unknown')} {data.get('description', data)}"
        )
    return data


def _ensure_ripped_telegram_webhook() -> dict[str, Any]:
    """Verify bot identity and ensure Telegram points it at this service."""
    token = _ripped_bot_token()
    target_url = _ripped_webhook_url()
    if not token:
        raise RuntimeError("dedicated Ripped Shorts Telegram bot token is missing")
    if not target_url:
        raise RuntimeError("Ripped Shorts public URL / Railway public domain is missing")

    me = dict(_telegram_api(token, "getMe").get("result") or {})
    username = str(me.get("username") or "").strip()
    if username.lower() != "rippedshortsbot":
        raise RuntimeError(
            "The configured dedicated token belongs to "
            f"@{username or 'unknown'}, not @rippedshortsbot"
        )

    before = dict(_telegram_api(token, "getWebhookInfo").get("result") or {})
    current_url = str(before.get("url") or "")
    if current_url != target_url:
        payload: dict[str, Any] = {
            "url": target_url,
            "allowed_updates": [
                "message",
                "edited_message",
                "channel_post",
                "edited_channel_post",
                "callback_query",
            ],
            "drop_pending_updates": False,
        }
        secret = _ripped_webhook_secret()
        if secret:
            payload["secret_token"] = secret
        _telegram_api(token, "setWebhook", payload)

    info = dict(_telegram_api(token, "getWebhookInfo").get("result") or {})
    registered_url = str(info.get("url") or "")
    if registered_url != target_url:
        raise RuntimeError(
            f"Telegram reports webhook {registered_url!r}; expected {target_url!r}"
        )

    last_error = str(info.get("last_error_message") or "").strip()
    last_error_date = int(info.get("last_error_date") or 0)
    last_error_age = (
        max(0, int(datetime.now(timezone.utc).timestamp()) - last_error_date)
        if last_error_date
        else None
    )
    can_read_group_messages = bool(me.get("can_read_all_group_messages"))

    configured_chat_id = next(
        (
            os.getenv(name, "").strip()
            for name in (
                "TELEGRAM_CHAT_ID",
                "TELEGRAM_GROUP_CHAT_ID",
                "Telegram_Group_Chat_ID",
            )
            if os.getenv(name, "").strip()
        ),
        "",
    )
    membership_status = "unknown"
    if configured_chat_id:
        try:
            membership = dict(
                _telegram_api(
                    token,
                    "getChatMember",
                    {
                        "chat_id": configured_chat_id,
                        "user_id": me.get("id"),
                    },
                ).get("result")
                or {}
            )
            membership_status = str(membership.get("status") or "unknown")
        except Exception:
            logger.exception(
                "Could not verify @%s membership in configured Telegram group",
                username,
            )

    logger.info(
        "Ripped Shorts Telegram webhook verified bot=@%s url=%s "
        "pending_updates=%s group_read_all=%s group_status=%s "
        "last_error=%s last_error_age_seconds=%s",
        username,
        registered_url,
        info.get("pending_update_count", 0),
        can_read_group_messages,
        membership_status,
        last_error or "none",
        last_error_age if last_error_age is not None else "none",
    )
    if (
        not can_read_group_messages
        and membership_status != "administrator"
    ):
        logger.warning(
            "@%s cannot receive plain group messages. Disable privacy mode "
            "for this bot with BotFather or make it a group administrator; "
            "addressed commands and replies remain available.",
            username,
        )
    if last_error and last_error_age is not None and last_error_age <= 600:
        logger.warning(
            "Telegram reports a recent delivery error for @%s age=%ss: %s",
            username,
            last_error_age,
            last_error,
        )
    elif last_error:
        logger.info(
            "Telegram retains a historical delivery error for @%s age=%ss: %s",
            username,
            last_error_age,
            last_error,
        )
    return info


def _ripped_webhook_watchdog() -> None:
    interval = max(
        60, int(os.getenv("RIPPED_TELEGRAM_WEBHOOK_CHECK_SECONDS", "300"))
    )
    while True:
        threading.Event().wait(interval)
        try:
            _ensure_ripped_telegram_webhook()
        except Exception:
            logger.exception(
                "Ripped Shorts Telegram webhook watchdog could not verify or repair ownership"
            )


def configure_ripped_telegram_webhook() -> None:
    """Verify the dedicated bot webhook and keep its ownership from drifting."""
    global _RIPPED_WEBHOOK_WATCHDOG_STARTED
    try:
        _ensure_ripped_telegram_webhook()
    except Exception:
        logger.exception("Could not configure Ripped Shorts Telegram webhook")
        return

    readiness = _readiness()
    logger.info(
        "RIPPED_SHORTS_READINESS ready=%s checks=%s",
        readiness.get("ready"),
        json.dumps(readiness.get("checks") or {}, sort_keys=True),
    )
    if not readiness.get("ready"):
        logger.warning(
            "RIPPED_SHORTS_NOT_READY failed_checks=%s",
            [
                name
                for name, passed in (readiness.get("checks") or {}).items()
                if not passed
            ],
        )

    with _RIPPED_WEBHOOK_WATCHDOG_LOCK:
        if _RIPPED_WEBHOOK_WATCHDOG_STARTED:
            return
        thread = threading.Thread(
            target=_ripped_webhook_watchdog,
            name="ripped-telegram-webhook-watchdog",
            daemon=True,
        )
        thread.start()
        _RIPPED_WEBHOOK_WATCHDOG_STARTED = True
