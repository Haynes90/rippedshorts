"""Temporary mobile copy-review portal for Ripped Shorts."""
from __future__ import annotations

import html
import json
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response

from audio_master_handoff import connect

router = APIRouter()

_REVIEW_TTL_DAYS = max(1, int(os.getenv("COPY_REVIEW_TTL_DAYS", "14")))
_DRIVE_ID_RE = re.compile(r"/file/d/([A-Za-z0-9_-]+)")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _public_base_url() -> str:
    value = (
        os.getenv("RIPPED_SHORTS_PUBLIC_URL")
        or os.getenv("RAILWAY_PUBLIC_DOMAIN")
        or ""
    ).strip().rstrip("/")
    if value and not value.startswith(("http://", "https://")):
        value = "https://" + value
    return value


def _db():
    db = connect()
    db.execute(
        "CREATE TABLE IF NOT EXISTS copy_review_sessions ("
        "token TEXT PRIMARY KEY, request_id TEXT NOT NULL, created_at TEXT NOT NULL, "
        "expires_at TEXT NOT NULL, last_opened_at TEXT, completed_at TEXT)"
    )
    return db


def ensure_review_session(request_id: str) -> dict[str, str]:
    """Return a live 14-day review link for a Ripped Shorts request."""
    now_dt = _now()
    with _db() as db:
        rows = db.execute(
            "SELECT * FROM copy_review_sessions WHERE request_id=? "
            "ORDER BY created_at DESC",
            (request_id,),
        ).fetchall()
        for row in rows:
            try:
                expires = datetime.fromisoformat(str(row["expires_at"]))
            except ValueError:
                continue
            if expires > now_dt and not row["completed_at"]:
                token = str(row["token"])
                base = _public_base_url()
                return {
                    "token": token,
                    "url": f"{base}/review/{token}" if base else "",
                    "expires_at": str(row["expires_at"]),
                }

        token = secrets.token_urlsafe(24)
        expires = now_dt + timedelta(days=_REVIEW_TTL_DAYS)
        db.execute(
            "INSERT INTO copy_review_sessions "
            "(token, request_id, created_at, expires_at, last_opened_at, completed_at) "
            "VALUES (?,?,?,?,?,?)",
            (token, request_id, _iso(now_dt), _iso(expires), "", ""),
        )

    base = _public_base_url()
    return {
        "token": token,
        "url": f"{base}/review/{token}" if base else "",
        "expires_at": _iso(expires),
    }


def review_url_for_request(request_id: str) -> str:
    return ensure_review_session(request_id).get("url", "")


def _session(token: str):
    with _db() as db:
        row = db.execute(
            "SELECT * FROM copy_review_sessions WHERE token=?",
            (token,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Review session not found")
        try:
            expires = datetime.fromisoformat(str(row["expires_at"]))
        except ValueError:
            raise HTTPException(status_code=410, detail="Review session expired")
        if expires <= _now():
            raise HTTPException(status_code=410, detail="Review session expired")
        db.execute(
            "UPDATE copy_review_sessions SET last_opened_at=? WHERE token=?",
            (_iso(_now()), token),
        )
        return row


def _request_row(request_id: str):
    with connect() as db:
        row = db.execute(
            "SELECT * FROM telegram_requests WHERE request_id=?",
            (request_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Ripped Shorts request not found")
    return row


def _media_for_state(state: dict[str, Any], request_id: str) -> dict[str, str]:
    media: dict[str, str] = {}
    clips = (state.get("result") or {}).get("segments", [])
    reviews = dict(state.get("candidate_reviews") or {})
    for index, clip in enumerate(clips):
        number = int(clip.get("candidate_number") or index + 1)
        url = str((reviews.get(str(index)) or {}).get("clip_url") or "")
        if url:
            media[f"{request_id}:short:{number}"] = url

    topics = (state.get("topic_result") or {}).get("segments", [])
    topic_reviews = dict(state.get("topic_reviews") or {})
    for index, _segment in enumerate(topics):
        url = str((topic_reviews.get(str(index)) or {}).get("segment_url") or "")
        if url:
            media[f"{request_id}:highlight:{index + 1}"] = url
    return media


def _drive_preview(url: str) -> str:
    match = _DRIVE_ID_RE.search(url or "")
    if not match:
        return ""
    return f"https://drive.google.com/file/d/{match.group(1)}/preview"


def _payload_for_session(token: str) -> dict[str, Any]:
    session = _session(token)
    request_id = str(session["request_id"])
    row = _request_row(request_id)
    state = json.loads(row["state_json"])
    drafts = list(state.get("copy_drafts") or [])
    if not drafts:
        raise HTTPException(status_code=409, detail="Copy review is not ready yet")

    parsed = state.get("parsed") or {}
    video_id = str(parsed.get("video_id") or "")
    title = str(state.get("vid_title") or "").strip() or "Ripped Shorts Review"
    media = _media_for_state(state, request_id)

    items = []
    for index, draft in enumerate(drafts):
        drive_url = media.get(str(draft.get("asset_id") or ""), "")
        items.append(
            {
                "index": index,
                "asset_id": str(draft.get("asset_id") or ""),
                "asset_type": str(draft.get("asset_type") or ""),
                "candidate_number": int(draft.get("candidate_number") or index + 1),
                "social_caption": str(draft.get("social_caption") or ""),
                "video_title": str(draft.get("video_title") or ""),
                "video_description": str(draft.get("video_description") or ""),
                "hashtags": str(draft.get("hashtags") or ""),
                "copy_status": str(draft.get("copy_status") or "pending"),
                "user_edited": bool(draft.get("user_edited")),
                "regeneration_count": int(draft.get("regeneration_count") or 0),
                "drive_url": drive_url,
                "preview_url": _drive_preview(drive_url),
            }
        )

    completed = sum(
        item["copy_status"] in {"approved", "edited"} or item["user_edited"]
        for item in items
    )
    regenerated = sum(item["regeneration_count"] > 0 for item in items)
    return {
        "request_id": request_id,
        "title": title,
        "video_id": video_id,
        "source_url": str(parsed.get("source_value") or ""),
        "thumbnail_url": (
            f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg" if video_id else ""
        ),
        "items": items,
        "completed": completed,
        "regenerated": regenerated,
        "total": len(items),
        "expires_at": str(session["expires_at"]),
        "review_complete": bool(state.get("copy_review_completed_at")),
        "schedule_requested": bool(state.get("schedule_requested_at")),
    }


def _card_html(item: dict[str, Any], token: str) -> str:
    index = int(item["index"])
    kind = item["asset_type"]
    number = item["candidate_number"]
    status = "Edited" if item["user_edited"] else (
        "Kept" if item["copy_status"] == "approved" else "Pending"
    )
    preview = ""
    if item["preview_url"]:
        preview = (
            f'<iframe class="preview" src="{html.escape(item["preview_url"])}" '
            'allow="autoplay; encrypted-media" allowfullscreen></iframe>'
        )
    elif item["drive_url"]:
        preview = (
            f'<a class="watch" target="_blank" rel="noopener" '
            f'href="{html.escape(item["drive_url"])}">Open video in Drive</a>'
        )

    if kind == "9:16_SHORT":
        fields = f"""
          <label>Caption</label>
          <textarea id="caption-{index}" rows="10">{html.escape(item["social_caption"])}</textarea>
          <label>Hashtags</label>
          <textarea id="hashtags-{index}" rows="3">{html.escape(item["hashtags"])}</textarea>
        """
    else:
        fields = f"""
          <label>Title</label>
          <input id="title-{index}" value="{html.escape(item["video_title"])}">
          <label>Description</label>
          <textarea id="description-{index}" rows="10">{html.escape(item["video_description"])}</textarea>
          <label>Tags / hashtags</label>
          <textarea id="hashtags-{index}" rows="3">{html.escape(item["hashtags"])}</textarea>
        """

    return f"""
      <article class="asset" id="asset-{index}" data-index="{index}" data-type="{html.escape(kind)}">
        <div class="asset-head">
          <div>
            <span class="pill">{'9:16 Short' if kind == '9:16_SHORT' else '16:9 Highlight'}</span>
            <h2>{'Short' if kind == '9:16_SHORT' else 'Highlight'} {number}</h2>
          </div>
          <span class="status" id="status-{index}">{status}</span>
        </div>
        {preview}
        <div class="fields">{fields}</div>
        <div class="actions">
          <button class="secondary" onclick="keepAsset({index})">Keep</button>
          <button class="secondary" onclick="regenAsset({index})">Regenerate</button>
          <button class="primary" onclick="saveAsset({index})">Save</button>
        </div>
        <div class="saved" id="saved-{index}"></div>
      </article>
    """


def _page(payload: dict[str, Any], token: str) -> str:
    cards = "".join(_card_html(item, token) for item in payload["items"])
    expires = html.escape(payload["expires_at"][:10])
    thumb = (
        f'<img class="source-thumb" src="{html.escape(payload["thumbnail_url"])}" '
        'alt="YouTube thumbnail">'
        if payload["thumbnail_url"]
        else ""
    )
    state_json = json.dumps(payload).replace("</", "<\\/")
    token_json = json.dumps(token)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Copy Review</title>
<style>
:root {{ font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#eef4f8; background:#07131d; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:#07131d; }}
main {{ max-width:760px; margin:0 auto; padding:16px 14px 110px; }}
.source {{ background:#102431; border:1px solid #23404f; border-radius:18px; overflow:hidden; margin-bottom:18px; }}
.source-thumb {{ width:100%; display:block; aspect-ratio:16/9; object-fit:cover; }}
.source-copy {{ padding:16px; }}
h1 {{ font-size:1.25rem; margin:0 0 8px; }}
.meta {{ color:#a9c0cc; font-size:.92rem; }}
.progress {{ position:sticky; top:0; z-index:10; background:rgba(7,19,29,.96); backdrop-filter:blur(10px); padding:10px 0; }}
.progressbar {{ height:8px; background:#1c3442; border-radius:999px; overflow:hidden; }}
.progressbar > div {{ height:100%; background:#5ee0b7; width:0%; transition:width .2s; }}
.progress-text {{ margin-top:7px; font-size:.9rem; color:#c5d6df; }}
.asset {{ background:#102431; border:1px solid #23404f; border-radius:18px; padding:14px; margin:14px 0; }}
.asset-head {{ display:flex; align-items:flex-start; justify-content:space-between; gap:12px; }}
.asset h2 {{ margin:5px 0 10px; font-size:1.1rem; }}
.pill,.status {{ display:inline-block; border-radius:999px; padding:5px 9px; font-size:.78rem; background:#173543; color:#cfe6ef; }}
.status {{ background:#263b31; color:#b9ebcd; }}
.preview {{ width:100%; aspect-ratio:16/9; border:0; border-radius:12px; margin:8px 0 14px; background:#000; }}
.watch {{ display:block; padding:12px; margin:8px 0 14px; border-radius:12px; background:#173543; color:#d8f2ff; text-decoration:none; text-align:center; }}
label {{ display:block; margin:12px 0 6px; color:#bed0da; font-size:.86rem; font-weight:600; }}
textarea,input {{ width:100%; background:#081923; color:#fff; border:1px solid #315163; border-radius:12px; padding:12px; font:inherit; }}
textarea {{ resize:vertical; min-height:72px; }}
.actions {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; margin-top:12px; }}
button {{ border:0; border-radius:12px; padding:12px 8px; font-weight:700; font-size:.95rem; }}
.primary {{ background:#5ee0b7; color:#052016; }}
.secondary {{ background:#1a3442; color:#e5f5fb; }}
.saved {{ min-height:20px; padding-top:8px; color:#aee8cf; font-size:.85rem; }}
.finish {{ position:fixed; left:0; right:0; bottom:0; background:rgba(7,19,29,.97); border-top:1px solid #24414f; padding:10px 14px calc(10px + env(safe-area-inset-bottom)); }}
.finish-inner {{ max-width:760px; margin:0 auto; display:grid; grid-template-columns:1fr; }}
.finish button {{ width:100%; background:#5ee0b7; color:#052016; font-size:1rem; }}
.notice {{ color:#a9c0cc; font-size:.82rem; margin-top:8px; text-align:center; }}
@media (max-width:480px) {{ .actions {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<main>
  <section class="source">
    {thumb}
    <div class="source-copy">
      <div class="meta">Source video</div>
      <h1>{html.escape(payload["title"])}</h1>
      <div class="meta">Review link expires {expires}. Drive files and saved copy do not expire with this page.</div>
    </div>
  </section>

  <div class="progress">
    <div class="progressbar"><div id="bar"></div></div>
    <div class="progress-text" id="progressText"></div>
  </div>

  <section id="assets">{cards}</section>
</main>

<div class="finish">
  <div class="finish-inner">
    <button id="finishButton" onclick="finishReview()">Finish &amp; Schedule</button>
    <div class="notice" id="globalStatus">Edits save directly to the durable copy record.</div>
  </div>
</div>

<script>
const TOKEN = {token_json};
const INITIAL = {state_json};
let state = INITIAL;

function updateProgress(data=state) {{
  const total = data.total || data.items.length || 0;
  const completed = data.completed || 0;
  const regenerated = data.regenerated || 0;
  document.getElementById('progressText').textContent =
    `${{completed}}/${{total}} reviewed • ${{regenerated}} regenerated`;
  document.getElementById('bar').style.width =
    total ? `${{Math.round((completed/total)*100)}}%` : '0%';
}}
updateProgress();

function bodyFor(index, action='save') {{
  const card = document.getElementById(`asset-${{index}}`);
  const type = card.dataset.type;
  const body = {{action}};
  if (type === '9:16_SHORT') {{
    body.social_caption = document.getElementById(`caption-${{index}}`).value;
    body.hashtags = document.getElementById(`hashtags-${{index}}`).value;
  }} else {{
    body.video_title = document.getElementById(`title-${{index}}`).value;
    body.video_description = document.getElementById(`description-${{index}}`).value;
    body.hashtags = document.getElementById(`hashtags-${{index}}`).value;
  }}
  return body;
}}

async function saveAsset(index, action='save') {{
  const saved = document.getElementById(`saved-${{index}}`);
  saved.textContent = 'Saving…';
  try {{
    const response = await fetch(`/api/ripped-shorts/review/${{TOKEN}}/asset/${{index}}`, {{
      method:'POST',
      headers:{{'Content-Type':'application/json'}},
      body:JSON.stringify(bodyFor(index, action))
    }});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Save failed');
    saved.textContent = `Saved ✓ ${{new Date().toLocaleTimeString([],{{hour:'numeric',minute:'2-digit'}})}}`;
    document.getElementById(`status-${{index}}`).textContent =
      action === 'keep' ? 'Kept' : 'Edited';
    state.completed = data.completed;
    state.regenerated = data.regenerated;
    updateProgress();
    return data;
  }} catch (error) {{
    saved.textContent = `Not saved — ${{error.message}}`;
    throw error;
  }}
}}

async function keepAsset(index) {{
  await saveAsset(index, 'keep');
}}

async function regenAsset(index) {{
  const saved = document.getElementById(`saved-${{index}}`);
  saved.textContent = 'Regenerating…';
  try {{
    const response = await fetch(`/api/ripped-shorts/review/${{TOKEN}}/asset/${{index}}/regenerate`, {{method:'POST'}});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Regeneration failed');
    if (data.asset_type === '9:16_SHORT') {{
      document.getElementById(`caption-${{index}}`).value = data.social_caption || '';
      document.getElementById(`hashtags-${{index}}`).value = data.hashtags || '';
    }} else {{
      document.getElementById(`title-${{index}}`).value = data.video_title || '';
      document.getElementById(`description-${{index}}`).value = data.video_description || '';
      document.getElementById(`hashtags-${{index}}`).value = data.hashtags || '';
    }}
    document.getElementById(`status-${{index}}`).textContent = 'Pending';
    saved.textContent = 'New draft generated — review and Save or Keep.';
    state.regenerated = data.regenerated;
    updateProgress();
  }} catch(error) {{
    saved.textContent = `Regeneration failed — ${{error.message}}`;
  }}
}}

async function finishReview() {{
  const button = document.getElementById('finishButton');
  const globalStatus = document.getElementById('globalStatus');
  button.disabled = true;
  button.textContent = 'Finishing…';
  try {{
    const response = await fetch(`/api/ripped-shorts/review/${{TOKEN}}/finish`, {{method:'POST'}});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Finish failed');
    button.textContent = 'Sent to Schedule Master ✓';
    globalStatus.textContent = 'Your saved copy and Drive videos remain permanent. This review link can expire safely.';
  }} catch(error) {{
    button.disabled = false;
    button.textContent = 'Finish & Schedule';
    globalStatus.textContent = `Could not finish: ${{error.message}}`;
  }}
}}
</script>
</body>
</html>"""


@router.get("/review/{token}")
def copy_review_page(token: str):
    try:
        payload = _payload_for_session(token)
    except HTTPException as exc:
        if exc.status_code == 410:
            return Response(
                content=(
                    "<!doctype html><meta name='viewport' content='width=device-width'>"
                    "<body style='font-family:system-ui;padding:32px'>"
                    "<h1>This review link has expired.</h1>"
                    "<p>Your Drive videos and previously saved copy are still preserved.</p>"
                    "</body>"
                ),
                media_type="text/html",
                status_code=410,
            )
        raise
    return Response(content=_page(payload, token), media_type="text/html")


@router.get("/api/ripped-shorts/review/{token}")
def review_data(token: str):
    return _payload_for_session(token)


@router.post("/api/ripped-shorts/review/{token}/asset/{index}")
async def save_review_asset(token: str, index: int, request: Request):
    session = _session(token)
    request_id = str(session["request_id"])
    row = _request_row(request_id)
    state = json.loads(row["state_json"])
    drafts = list(state.get("copy_drafts") or [])
    if index < 0 or index >= len(drafts):
        raise HTTPException(status_code=404, detail="Review asset not found")

    body = await request.json()
    action = str(body.get("action") or "save").lower()
    draft = dict(drafts[index])
    asset_type = str(draft.get("asset_type") or "")
    changed = False

    if asset_type == "9:16_SHORT":
        for key in ("social_caption", "hashtags"):
            if key in body:
                value = str(body.get(key) or "")
                if value != str(draft.get(key) or ""):
                    changed = True
                draft[key] = value
    else:
        for key in ("video_title", "video_description", "hashtags"):
            if key in body:
                value = str(body.get(key) or "")
                if value != str(draft.get(key) or ""):
                    changed = True
                draft[key] = value

    if action == "keep":
        draft["copy_status"] = "approved"
    else:
        draft["copy_status"] = "edited" if changed or draft.get("user_edited") else "approved"
        if changed:
            draft["user_edited"] = True
            draft["edited_fields"] = sorted(
                set(draft.get("edited_fields") or [])
                | {
                    key
                    for key in ("social_caption", "video_title", "video_description", "hashtags")
                    if key in body
                }
            )
    draft["web_review_saved_at"] = _iso(_now())
    drafts[index] = draft
    state["copy_drafts"] = drafts
    state["copy_review_index"] = index

    with connect() as db:
        db.execute(
            "UPDATE telegram_requests SET state_json=?, updated_at=? WHERE request_id=?",
            (json.dumps(state), _iso(_now()), request_id),
        )

    import telegram_intake as tg
    tg._checkpoint_copy_edit(request_id, state, draft, str(row["user_id"] or "web_review"))
    tg._refresh_copy_review(str(row["chat_id"]), request_id, index=index)

    payload = _payload_for_session(token)
    return {
        "status": "saved",
        "asset_index": index,
        "completed": payload["completed"],
        "regenerated": payload["regenerated"],
    }


@router.post("/api/ripped-shorts/review/{token}/asset/{index}/regenerate")
def regenerate_review_asset(token: str, index: int):
    session = _session(token)
    request_id = str(session["request_id"])
    row = _request_row(request_id)
    state = json.loads(row["state_json"])
    drafts = list(state.get("copy_drafts") or [])
    if index < 0 or index >= len(drafts):
        raise HTTPException(status_code=404, detail="Review asset not found")

    import telegram_intake as tg

    old = drafts[index]
    fresh = tg._generate_schedule_copy(
        [old],
        str(state.get("show_id") or ""),
        tg._state_vid_title(state),
        str((state.get("parsed") or {}).get("source_value") or ""),
    )[0]
    fresh["ai_social_caption"] = fresh.get("social_caption", "")
    fresh["ai_video_title"] = fresh.get("video_title", "")
    fresh["ai_video_description"] = fresh.get("video_description", "")
    fresh["copy_status"] = "pending"
    fresh["user_edited"] = False
    fresh["regenerated_at"] = _iso(_now())
    fresh["regeneration_count"] = int(old.get("regeneration_count") or 0) + 1
    drafts[index] = fresh
    state["copy_drafts"] = drafts
    state["copy_review_index"] = index

    with connect() as db:
        db.execute(
            "UPDATE telegram_requests SET state_json=?, updated_at=? WHERE request_id=?",
            (json.dumps(state), _iso(_now()), request_id),
        )

    tg._refresh_copy_review(str(row["chat_id"]), request_id, index=index)
    payload = _payload_for_session(token)
    current = payload["items"][index]
    return {**current, "status": "regenerated", "regenerated": payload["regenerated"]}


@router.post("/api/ripped-shorts/review/{token}/finish")
def finish_review(token: str):
    session = _session(token)
    request_id = str(session["request_id"])
    row = _request_row(request_id)
    state = json.loads(row["state_json"])
    drafts = list(state.get("copy_drafts") or [])
    if not drafts:
        raise HTTPException(status_code=409, detail="No copy drafts are available")
    if state.get("schedule_requested_at"):
        return {"status": "already_scheduling", "request_id": request_id}

    state.pop("awaiting_copy_input", None)
    state["copy_review_completed_at"] = _iso(_now())
    state["schedule_requested_at"] = _iso(_now())
    state["copy_review_source"] = "temporary_web_review"

    with connect() as db:
        db.execute(
            "UPDATE telegram_requests SET status=?, state_json=?, updated_at=? WHERE request_id=?",
            ("awaiting_render_completion", json.dumps(state), _iso(_now()), request_id),
        )
    with _db() as db:
        db.execute(
            "UPDATE copy_review_sessions SET completed_at=? WHERE token=?",
            (_iso(_now()), token),
        )

    import telegram_intake as tg

    tg._log_copy_learning(request_id, state, drafts, str(row["user_id"] or "web_review"))
    tg.send(
        str(row["chat_id"]),
        "✅ Copy review finished from the mobile review page. "
        "Schedule Master will receive the batch as soon as every approved render finishes.",
    )
    tg._notify_render_queue_complete(request_id, str(row["chat_id"]))
    return {"status": "schedule_requested", "request_id": request_id}
