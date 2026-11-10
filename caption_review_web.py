"""Temporary mobile copy-review portal for Ripped Shorts."""
from __future__ import annotations

import html
import json
import os
import re
import secrets
import threading
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Header, HTTPException, Request, Response

from audio_master_handoff import connect

router = APIRouter()

_REVIEW_TTL_DAYS = max(1, int(os.getenv("COPY_REVIEW_TTL_DAYS", "14")))
_DRIVE_ID_RE = re.compile(r"/file/d/([A-Za-z0-9_-]+)")
_PREVIEW_LOCK = threading.Lock()


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


def review_url_for_request(request_id: str, stage: str = 'final') -> str:
    if stage not in {'shorts', 'highlights', 'editorial', 'framing', 'final'}:
        raise ValueError('Unknown review stage')
    url = ensure_review_session(request_id).get("url", "")
    return url + '?stage=' + stage + '#' + stage if url else ''


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


def _media_for_state(state: dict[str, Any], request_id: str, verified_clips=None) -> dict[str, str]:
    if state.get('preview_first_version') == 1:
        items = verified_clips if verified_clips is not None else __import__('finished_clip_review').clips(request_id, state)
        return {c['asset_id']:c['media_url'] for c in items if c['finished']}
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
    if not drafts and not state.get("result") and not state.get("topic_result"):
        raise HTTPException(status_code=409, detail="Copy review is not ready yet")

    parsed = state.get("parsed") or {}
    video_id = str(parsed.get("video_id") or "")
    title = str(state.get("vid_title") or "").strip() or "Ripped Shorts Review"
    verified_clips = __import__('finished_clip_review').clips(request_id, state, __import__('clip_review_state').reviews(request_id))
    media = _media_for_state(state, request_id, verified_clips)
    playable_assets = {c['asset_id'] for c in verified_clips if c['finished']}

    items = []
    for index, draft in enumerate(drafts):
        if state.get('preview_first_version') == 1 and str(draft.get('asset_id') or '') not in playable_assets:
            continue  # Keep original draft indexes and records; omit unavailable media from normal review.
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
                "favorite": bool(draft.get('favorite')),
                "editorial_hold": bool(draft.get('editorial_hold')),
                "selection_feedback": str(draft.get('selection_feedback') or ''),
                "drive_url": drive_url,
                "preview_url": ('/api/ripped-shorts/review/'+token+'/clip/'+quote(str(draft.get('asset_id') or ''),safe='')+'/media') if str(draft.get('asset_id') or '') in playable_assets else '',
            }
        )

    from batch_progress import summarize
    batch = summarize(request_id,state)
    candidate_evidence=[]
    for lane,key,review_key in [('9:16','result','candidate_reviews'),('16:9','topic_result','topic_reviews')]:
        for number,clip in enumerate((state.get(key) or {}).get('segments') or [],1):
            review=(state.get(review_key) or {}).get(str(number-1)) or {}
            candidate_evidence.append({'format':lane,'number':number,'candidate_id':clip.get('candidate_id'),
                'title':clip.get('subject_description') or clip.get('title') or next(iter(clip.get('topic_titles') or []),None),
                'transcript':clip.get('transcript'),'reason':clip.get('reason') or clip.get('why_it_will_work'),
                'selection_provenance':clip.get('selection_provenance') or {},
                'review_status':review.get('status','pending'),'error':review.get('render_error'),
                'diagnostics':review.get('render_diagnostics'), 'learning_influence':clip.get('learning_influence'),
                'framing_plan':clip.get('framing_plan')})
    from creator_approval import suggestions, approved_batch
    creator_brand = str(state.get("show_id") or "").upper()
    creator_required = creator_brand in {"TCB", "TDOG"}
    creator_suggestions = suggestions(creator_brand, str(state.get("youtube_channel_id") or "")) if creator_required else []
    creator_approval = approved_batch(request_id) if creator_required else None
    completed = sum(
        item["copy_status"] in {"approved", "edited"} or item["user_edited"]
        for item in items
    )
    regenerated = sum(item["regeneration_count"] > 0 for item in items)
    gallery = [c for c in verified_clips if c['finished']] if state.get('preview_first_version') == 1 else verified_clips
    schedule_blockers = []
    if state.get('preview_first_version') == 1:
        if state.get('acceptance_demo'):
            schedule_blockers.append('This acceptance review is not yet enabled for scheduling.')
        approved = [c for c in gallery if c['approval_current']]
        if not approved:
            schedule_blockers.append('Select at least one verified preview.')
        if creator_required and not creator_approval:
            schedule_blockers.append('Confirm the batch collaborators.')
        for clip in approved:
            draft = next((d for d in drafts if d.get('asset_id') == clip['asset_id']), None)
            if not draft or draft.get('copy_status') not in {'approved', 'edited'} or draft.get('editorial_hold'):
                schedule_blockers.append('Finalize the caption and metadata for ' + str(clip['title']) + '.')
        from clip_review_state import delivery_blockers
        approved_ids = {c['asset_id'] for c in approved}
        if any(b.split(': ',1)[0] in approved_ids for b in delivery_blockers(request_id,state)):
            schedule_blockers.append('Resolve the selected video edit or framing request.')
    return {
        "request_id": request_id,
        "preview_first": state.get('preview_first_version')==1,
        "acceptance_demo": bool(state.get('acceptance_demo')),
        "batch_summary": batch,
        "candidate_evidence": candidate_evidence,
        "comparisons": __import__('clip_review_state').reviews(request_id),
        "clips": gallery,
        "schedule_blockers": schedule_blockers,
        "preview_review_total": len(gallery),
        "preview_review_completed": sum(bool(c['decision'].get('choice')) and (c['decision'].get('choice') != 'approve' or c['approval_current']) for c in gallery),
        "creator_brand": creator_brand,
        "creator_required": creator_required,
        "creator_suggestions": creator_suggestions,
        "creator_approval": creator_approval,
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
    preview = (f'<video controls playsinline preload="metadata" style="width:100%;max-height:70vh" src="{html.escape(item["preview_url"],quote=True)}"></video>'
               if item.get('preview_url') else '<p role="status">Verified preview unavailable. Review its pending or held status above.</p>')

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
        <label><input type="checkbox" id="playback-{index}" style="width:auto" /> I played this clip and confirmed audible audio</label>
        <div class="fields">{fields}</div>
        <label><input type="checkbox" id="favorite-{index}" style="width:auto" {'checked' if item.get('favorite') else ''} /> Favorite — suggest more related quality material</label>
        <label><input type="checkbox" id="hold-{index}" style="width:auto" {'checked' if item.get('editorial_hold') else ''} /> Hold this clip from scheduling</label>
        <label>Selection feedback (optional)</label>
        <textarea id="feedback-{index}" rows="2">{html.escape(item.get('selection_feedback',''))}</textarea>
        <div class="actions">
          <button class="secondary" onclick="keepAsset({index})">Keep</button>
          <button class="secondary" onclick="regenAsset({index})">Regenerate</button>
          <button class="primary" onclick="saveAsset({index})">Save</button>
        </div>
        <div class="saved" id="saved-{index}"></div>
      </article>
    """


def _page(payload: dict[str, Any], token: str) -> str:
    from creator_review_ui import section, SCRIPT
    creator_section = section()
    creator_script = SCRIPT
    cards = "".join(_card_html(item, token) for item in payload["items"])
    failure_cards=''.join('<p>'+html.escape(str(c['format'])+' clip '+str(c['number'])+': '+str(c['error']))+'</p>'
        for c in payload.get('candidate_evidence',[]) if c.get('error'))
    learning_cards=''.join('<details><summary>'+html.escape(str(c['format'])+' clip '+str(c['number'])+' — recommendation/framing evidence')+'</summary><pre>'+html.escape(json.dumps(c,indent=2))+'</pre></details>'
        for c in payload.get('candidate_evidence',[]) if c.get('learning_influence') or c.get('framing_plan'))
    cards='<aside aria-label="Per-clip failures">'+failure_cards+'</aside>'+cards+learning_cards
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
  {'<p role="status">ISOLATED ACCEPTANCE DEMO — decisions here do not change the production TDOG request.</p>' if payload.get('acceptance_demo') else ''}
  <section class="source">
    {thumb}
    <div class="source-copy">
      <div class="meta">Source video</div>
      <h1>{html.escape(payload["title"])}</h1>
      <div class="meta">Review link expires {expires}. Drive files and saved copy do not expire with this page.</div>
    </div>
  </section>

  {creator_section}
  <section class="asset"><h2>Workflow test status</h2>
    <p>Internal checks do not publish content. Live testing requires separate owner approval.</p>
    <button type="button" id="refreshWorkflow">Refresh test status</button>
    <pre id="workflowStatus" style="white-space:pre-wrap">Status not loaded.</pre>
  </section>
  <script>
  document.getElementById('refreshWorkflow').onclick=async()=>{{
    const result=await fetch('/api/ripped-shorts/review/{token}/test-status');
    const data=await result.json();
    document.getElementById('workflowStatus').textContent=result.ok ?
      'Test run: '+data.test_run_id+'\n'+Object.entries(data.checkpoints).map(([name,v])=>name+': '+v.status).join('\n')+
      '\nOverall: '+data.overall_status+'\nPublishing: '+(data.publishing_globally_paused===true?'PAUSED':'NOT VERIFIED PAUSED')+
      '\nInstagram Collab: '+(data.instagram_collaborators_in_approved_payload?'VALIDATED APPROVAL, NOT PROOF OF SUBMISSION':'NOT VERIFIED')+
      '\nBatch: '+(data.batch_summary?.status||'NOT STARTED')+'\nCompleted clips: '+(data.batch_summary?.completed_clips||0)+'/'+(data.batch_summary?.selected_clips||0)+'\nScheduled deliveries:\n'+(data.batch_summary?.clips||[]).flatMap(c=>(c.deliveries||[]).map(d=>c.format+' • '+d.destination_id+' • '+d.scheduled_at)).join('\n')+'\nNext: '+data.next_required_action : 'Status unavailable. Refresh after the service recovers.';
  }};
  </script>
  <div class="progress">
    <div class="progressbar"><div id="bar"></div></div>
    <div class="progress-text" id="progressText"></div>
  </div>

  <section id="finished-clips"><h2>{'Watch previews, then choose clips' if payload.get('preview_first') else 'Review finished clips'}</h2>{__import__('clip_review_ui').finished_cards(payload.get('clips', []), token)}</section>
  <details id="editing-workflow" {'open' if payload.get('stage') in {'editorial','framing'} else ''}><summary>Edit boundaries or framing / compare alternatives</summary>
  {__import__('clip_review_ui').sections(payload.get('comparisons', []), token)}
  </details>
  <section id="shorts"><h2>9:16 candidate review</h2><p>{'Watch and choose Shorts above. Telegram provides notifications only.' if payload.get('preview_first') else 'Candidate selection remains in Telegram; saved selections are preserved.'}</p></section>
  <section id="highlights"><h2>16:9 candidate review</h2><p>{'Review the highlight previews after Shorts above; each decision is saved independently.' if payload.get('preview_first') else 'Highlight selection follows Shorts confirmation in Telegram.'}</p></section>
  <section id="final"><h2>Final rendered video and copy approval</h2><section id="assets">{cards}</section></section>
</main>

<div class="finish">
  <div class="finish-inner">
    <button id="finishButton" {'disabled' if payload.get('schedule_blockers') else ''} onclick="finishReview()">Finish &amp; Schedule</button>
    <div class="notice" id="globalStatus">{html.escape(' '.join(payload.get('schedule_blockers') or [])) or 'Edits save directly to the durable copy record.'}</div>
  </div>
</div>

<script>
const TOKEN = {token_json};
const INITIAL = {state_json};
let state = INITIAL;
{creator_script}

function updateProgress(data=state) {{
  if(data.preview_first){{
    const total=data.preview_review_total||0,completed=data.preview_review_completed||0;
    document.getElementById('progressText').textContent=`${{completed}}/${{total}} playable previews reviewed`;
    document.getElementById('bar').style.width=total?`${{Math.round(completed/total*100)}}%`:'0%';
    const blockers=data.schedule_blockers||[];
    document.getElementById('finishButton').disabled=blockers.length>0;
    document.getElementById('globalStatus').textContent=blockers.join(' ')||'Selected finished videos are ready for your final scheduling approval.';
    return;
  }}
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
  const body = {{action,favorite:document.getElementById('favorite-'+index)?.checked===true,
    editorial_hold:document.getElementById('hold-'+index)?.checked===true,
    selection_feedback:document.getElementById('feedback-'+index)?.value||''}};
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
      body:JSON.stringify({{...bodyFor(index, action),playback_verified:document.getElementById('playback-'+index)?.checked===true}})
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
  button.textContent = 'Saving edits…';
  try {{
    // Finish must use exactly what is currently written in the review fields,
    // even if the user did not tap Save on each individual card first.
    const cards = Array.from(document.querySelectorAll('.asset')).filter(card => card.hasAttribute('data-index'));
    for (const card of cards) {{
      const index = Number(card.dataset.index);
      await saveAsset(index, 'save');
    }}
    button.textContent = 'Finishing…';
    const response = await fetch(`/api/ripped-shorts/review/${{TOKEN}}/finish`, {{method:'POST'}});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Finish failed');
    button.textContent = data.status === 'scheduler_accepted' ? 'Schedule Master acknowledged ✓' : 'Scheduling requested';
    globalStatus.textContent = data.status === 'scheduler_accepted' ? 'Receipt confirmed. Check the actual delivery dates below.' : 'Your copy and scheduling request were saved. Scheduling is pending; no successful handoff is claimed.';
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
def copy_review_page(token: str, stage: str = 'final'):
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
    payload['stage'] = stage
    return Response(content=_page(payload, token), media_type="text/html", headers={'Cache-Control':'private, no-store'})


@router.get("/api/ripped-shorts/review/{token}")
def review_data(token: str):
    return _payload_for_session(token)


@router.post('/api/ripped-shorts/review/{token}/comparison/{revision}')
async def save_comparison(token: str, revision: str, request: Request):
    session = _session(token)
    body = await request.json()
    try:
        return __import__('clip_review_state').decide(str(session['request_id']), revision,
            body.get('choice'), body.get('feedback') or '', body.get('version', 0))
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get('/api/ripped-shorts/review/{token}/comparison/{revision}/media/{variant}')
def comparison_media(token: str, revision: str, variant: str):
    """Stream only a registered artifact belonging to this bearer review session."""
    session = _session(token)
    review = next((r for r in __import__('clip_review_state').reviews(str(session['request_id'])) if r['revision_id'] == revision), None)
    artifact = (review or {}).get('media', {}).get(variant)
    if not artifact:
        raise HTTPException(status_code=404, detail='Preview not found')
    return _preview_response(artifact)


def _preview_response(artifact):
    from pathlib import Path
    from fastapi.responses import FileResponse
    from audio_master_handoff import DATA_DIR
    root = Path(DATA_DIR).resolve()
    if artifact.get('local_path'):
        path = Path(artifact['local_path']).resolve()
        if not path.is_relative_to(root):
            raise HTTPException(status_code=403, detail='Preview outside retained storage')
    else:
        # Cache exact private artifact IDs, never arbitrary URLs. A failed download
        # leaves no usable partial file; successful verified files are reused.
        from audio_master_handoff import drive_metadata, download_drive
        file_id = artifact['drive_file_id']
        path = root / 'review-previews' / (file_id + '.mp4')
        with _PREVIEW_LOCK:
            if not path.is_file():
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_suffix('.partial')
                try:
                    metadata = drive_metadata(file_id)
                    if metadata.get('mimeType') != 'video/mp4' or metadata.get('trashed') or int(metadata.get('size') or 0) <= 0:
                        raise ValueError('Not an available MP4 review artifact')
                    download_drive(file_id, partial, metadata)
                    if partial.stat().st_size != int(metadata['size']):
                        raise ValueError('Incomplete private preview')
                    partial.replace(path)
                except Exception:
                    partial.unlink(missing_ok=True)
                    raise HTTPException(status_code=503, detail='Private preview unavailable; saved decisions are preserved')
    if not path.is_file() or path.stat().st_size <= 0:
        raise HTTPException(status_code=404, detail='Retained preview missing')
    if artifact.get('size') and path.stat().st_size != int(artifact['size']):
        raise HTTPException(status_code=409, detail='Preview integrity check failed')
    if artifact.get('sha256'):
        import hashlib
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != artifact['sha256']:
                raise HTTPException(status_code=409, detail='Preview fingerprint changed')
    return FileResponse(path, media_type='video/mp4', headers={'Cache-Control': 'private, no-store', 'Referrer-Policy': 'no-referrer'})


@router.post('/api/ripped-shorts/review/{token}/clip/{asset_id}')
async def save_clip_decision(token: str, asset_id: str, request: Request):
    session = _session(token)
    try:
        return __import__('finished_clip_review').save(str(session['request_id']), asset_id, await request.json())
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get('/api/ripped-shorts/review/{token}/clip/{asset_id}/media')
def finished_clip_media(token: str, asset_id: str):
    session = _session(token)
    row = _request_row(str(session['request_id']))
    items = __import__('finished_clip_review').clips(str(session['request_id']), json.loads(row['state_json']))
    item = next((c for c in items if c['asset_id'] == asset_id and c['finished']), None)
    match = _DRIVE_ID_RE.search(item['media_url']) if item else None
    if not match:
        raise HTTPException(status_code=409, detail='Verified finished media unavailable; diagnostic comparisons are not final exports')
    artifact={'drive_file_id':match.group(1)}
    if item.get('preview'):
        from production_intelligence import database
        from render_recovery import _get
        with database() as db:
            receipt=_get(db,item.get('media_receipt_id'),'retained_media')
        if not receipt or receipt['remote']['clip_id']!=match.group(1):
            raise HTTPException(status_code=409,detail='Exact preview retention receipt unavailable')
        artifact.update(size=receipt['expected']['bytes'],sha256=receipt['expected']['output_sha256'])
    return _preview_response(artifact)


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
    from batch_progress import copy_frozen
    if copy_frozen(state,str(draft.get('asset_id') or '')):
        if any(k in body and body[k]!=draft.get(k, False if k!='selection_feedback' else '') for k in ('favorite','editorial_hold','selection_feedback')):
            raise HTTPException(status_code=409,detail='Submitted approvals and holds are frozen')
        if all(str(body.get(k) or '')==str(draft.get(k) or '') for k in ('social_caption','video_title','video_description','hashtags') if k in body):
            if body.get('playback_verified') is True:
                media_url=_media_for_state(state,request_id).get(str(draft.get('asset_id') or ''))
                if not media_url:
                    raise HTTPException(status_code=409,detail='Rendered media is unavailable')
                with connect() as db:
                    db.execute('BEGIN IMMEDIATE')
                    latest=db.execute('SELECT state_json FROM telegram_requests WHERE request_id=?',(request_id,)).fetchone()
                    current=json.loads(latest['state_json'])
                    current.setdefault('website_playback_checks',{})[draft['asset_id']]={'verified':True,'media_url':media_url,'verified_by':str(row['user_id'] or 'web_review'),'verified_at':_iso(_now())}
                    db.execute('UPDATE telegram_requests SET state_json=?, updated_at=? WHERE request_id=?',(json.dumps(current),_iso(_now()),request_id))
            return {'status':'already_submitted','asset_index':index,
                'completed':sum(d.get('copy_status') in {'approved','edited'} for d in drafts),
                'regenerated':sum(int(d.get('regeneration_count') or 0)>0 for d in drafts)}
        raise HTTPException(status_code=409,detail='Submitted copy is frozen')
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
    for key in ('favorite','editorial_hold'):
        if key in body:
            if type(body[key]) is not bool:
                raise HTTPException(status_code=400,detail='Feedback flags must be boolean')
            draft[key]=body[key]
    if 'selection_feedback' in body:
        draft['selection_feedback']=str(body['selection_feedback'] or '')[:2000]
    draft["web_review_saved_at"] = _iso(_now())
    drafts[index] = draft
    state["copy_drafts"] = drafts
    state["copy_review_index"] = index
    if body.get('playback_verified') is True:
        media_url=_media_for_state(state,request_id).get(str(draft.get('asset_id') or ''))
        if not media_url:
            raise HTTPException(status_code=409,detail='This clip does not have rendered media yet')
        state.setdefault('website_playback_checks',{})[draft['asset_id']]={'verified':True,'media_url':media_url,'verified_by':str(row['user_id'] or 'web_review'),'verified_at':_iso(_now())}

    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        latest=db.execute('SELECT state_json FROM telegram_requests WHERE request_id=?',(request_id,)).fetchone()
        current=json.loads(latest['state_json'])
        latest_drafts=list(current.get('copy_drafts') or [])
        matching=next((i for i,d in enumerate(latest_drafts) if d.get('asset_id')==draft.get('asset_id')),None)
        if matching is None or copy_frozen(current,str(draft.get('asset_id') or '')):
            raise HTTPException(status_code=409,detail='Clip changed or was submitted while editing; refresh review')
        latest_drafts[matching]=draft
        current['copy_drafts']=latest_drafts
        current['copy_review_index']=matching
        if body.get('playback_verified') is True:
            current.setdefault('website_playback_checks',{})[draft['asset_id']]=state['website_playback_checks'][draft['asset_id']]
        state=current
        db.execute(
            "UPDATE telegram_requests SET state_json=?, updated_at=? WHERE request_id=?",
            (json.dumps(state), _iso(_now()), request_id),
        )

    import telegram_intake as tg
    from intelligence_workflow import observe
    try:
        observe(request_id,state)
    except Exception:
        __import__('logging').getLogger(__name__).exception('Review evidence projection pending')
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
    from batch_progress import copy_frozen
    if 0 <= index < len(drafts) and copy_frozen(state,str(drafts[index].get('asset_id') or '')):
        raise HTTPException(status_code=409,detail='Submitted copy is frozen')
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


@router.post("/api/ripped-shorts/review/{token}/creators")
async def confirm_review_creators(token: str, request: Request):
    session = _session(token)
    row = _request_row(str(session["request_id"]))
    state = json.loads(row["state_json"])
    if (state.get("schedule_master") or {}).get("shorts_status") in {"sending", "accepted"}:
        raise HTTPException(status_code=409, detail="This batch is already submitted; its approval is preserved")
    body = await request.json()
    from creator_approval import confirm_batch
    try:
        approved = confirm_batch(str(session["request_id"]), str(state.get("show_id") or ""),
            str((state.get("parsed") or {}).get("video_id") or ""),
            str(state.get("youtube_channel_id") or ""), body.get("entries"),
            str(row["user_id"] or "web_review"), body.get("confirmed") is True)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    from workflow_test_status import journal
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        latest=db.execute('SELECT state_json FROM telegram_requests WHERE request_id=?',(str(session['request_id']),)).fetchone()
        current=json.loads(latest['state_json'])
        current['creator_approval']=approved
        journal(str(session['request_id']),current,approved)
        db.execute('UPDATE telegram_requests SET state_json=?, updated_at=? WHERE request_id=?',
            (json.dumps(current),_iso(_now()),str(session['request_id'])))
        state=current
    if state.get("schedule_requested_at"):
        import telegram_intake as tg
        tg._notify_render_queue_complete(str(session["request_id"]), str(row["chat_id"]))
    return {"status":"approved", "creator_approval":approved}


@router.post("/api/ripped-shorts/review/{token}/finish")
def finish_review(token: str):
    session = _session(token)
    request_id = str(session["request_id"])
    row = _request_row(request_id)
    state = json.loads(row["state_json"])
    if state.get('acceptance_demo'):
        raise HTTPException(status_code=409, detail='Isolated demo cannot schedule or publish')
    from finished_clip_review import clips, blocked_assets
    accepted = (state.get('schedule_master') or {}).get('shorts_response') or {}
    if (state.get('schedule_master') or {}).get('shorts_status') == 'accepted' and accepted.get('asset_group_ids'):
        return {'status':'scheduler_accepted','request_id':request_id,'receipt':accepted}
    finished = [c for c in clips(request_id,state) if c['finished'] and c['approval_current'] and c['asset_id'] not in blocked_assets(request_id,state)]
    if not finished:
        raise HTTPException(status_code=409, detail='No approved verified final renders are available. Your saved review choices are preserved; nothing was sent to Schedule Master.')
    playable = set(_media_for_state(state, request_id))
    blockers = [b for b in __import__('clip_review_state').delivery_blockers(request_id, state) if b.split(': ', 1)[0] in playable]
    if blockers:
        raise HTTPException(status_code=409, detail='Editorial/framing review or verified rendering required: ' + '; '.join(blockers))
    drafts = list(state.get("copy_drafts") or [])
    if not drafts:
        raise HTTPException(status_code=409, detail="No copy drafts are available")
    if state.get("schedule_requested_at"):
        import telegram_intake as tg
        tg._notify_render_queue_complete(request_id,str(row['chat_id']))
        accepted = (state.get('schedule_master') or {}).get('shorts_response') or {}
        return {"status": "scheduler_accepted" if accepted.get('asset_group_ids') else "schedule_requested", "request_id": request_id, 'receipt': accepted}
    from creator_approval import approved_batch
    if str(state.get("show_id") or "").upper() in {"TCB", "TDOG"} and not approved_batch(request_id):
        raise HTTPException(status_code=409, detail="Confirm the featured creators once at the top of this page")

    eligible_ids = {c['asset_id'] for c in finished}
    if any(d.get('copy_status') not in {'approved','edited'} for d in drafts if str(d.get('asset_id') or '') in eligible_ids):
        raise HTTPException(status_code=409, detail='Approve every selected clip caption before finishing')
    state.pop("awaiting_copy_input", None)
    state["copy_review_completed_at"] = _iso(_now())
    state["schedule_requested_at"] = _iso(_now())
    state["copy_review_source"] = "temporary_web_review"
    __import__('workflow_test_status').journal(request_id,state,approved_batch(request_id))

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


@router.get('/api/ripped-shorts/review/{token}/test-status')
def workflow_status(token: str):
    # Existing bearer review token, checked without modifying last_opened_at or DB schema.
    import sqlite3
    import audio_master_handoff as storage
    from contextlib import closing
    from workflow_test_status import project
    with closing(sqlite3.connect(storage.DB_PATH.resolve().as_uri()+'?mode=ro',uri=True)) as db:
        db.row_factory=sqlite3.Row
        session=db.execute('SELECT * FROM copy_review_sessions WHERE token=?',(token,)).fetchone()
        if not session:
            raise HTTPException(status_code=404,detail='Review session not found')
        if datetime.fromisoformat(session['expires_at']) <= _now():
            raise HTTPException(status_code=410,detail='Review session expired')
        row=db.execute('SELECT state_json FROM telegram_requests WHERE request_id=?',(session['request_id'],)).fetchone()
        if not row:
            raise HTTPException(status_code=404,detail='Request not found')
        state=json.loads(row['state_json'])
    return _workflow_result(session['request_id'],state)


def _workflow_result(request_id, state):
    from workflow_test_status import project
    scheduler=None
    target=(os.getenv('SCHEDULE_MASTER_INTERNAL_URL') or os.getenv('SCHEDULE_MASTER_URL') or '').strip().rstrip('/')
    secret=os.getenv('SCHEDULE_MASTER_SHARED_SECRET','').strip()
    if (state.get('schedule_master') or {}).get('shorts_status')=='accepted' and target and secret:
        import requests
        from urllib.parse import quote
        try:
            response=requests.get(target+'/schedule/test-runs/'+quote(request_id,safe=''),
                headers={'x-schedule-master-secret':secret},timeout=(3,8))
            response.raise_for_status()
            scheduler=response.json()
        except Exception:
            # Do not show stale permission/preflight success while downstream is unavailable.
            scheduler={**((state.get('schedule_master') or {}).get('shorts_response') or {}),
                'publishing_globally_paused':None,'delivery_validation':{'status':'BLOCKED','reason':'SCHEDULER_STATUS_UNAVAILABLE'}}
    result=project(request_id,state,state.get('creator_approval'),scheduler)
    if state.get('source_intelligence'):
        from production_intelligence import analysis_status
        result['source_analyses']=analysis_status(state['source_intelligence']['map_id'])
    return result


@router.get('/api/ripped-shorts/test-runs/{request_id}')
def workflow_status_by_request(request_id: str, x_schedule_master_secret: str = Header(default='')):
    """Available before a review session exists, using the existing service secret."""
    import secrets
    import sqlite3
    import audio_master_handoff as storage
    from contextlib import closing
    expected=os.getenv('SCHEDULE_MASTER_SHARED_SECRET','').strip()
    if not expected or not secrets.compare_digest(expected,x_schedule_master_secret):
        raise HTTPException(status_code=401,detail='Invalid workflow service secret')
    with closing(sqlite3.connect(storage.DB_PATH.resolve().as_uri()+'?mode=ro',uri=True)) as db:
        row=db.execute('SELECT state_json FROM telegram_requests WHERE request_id=?',(request_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404,detail='Request not found')
        return _workflow_result(request_id,json.loads(row[0]))
