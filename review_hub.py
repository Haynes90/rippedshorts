"""Internal mobile review API. Existing Telegram jobs remain the source of truth.

No new paid infrastructure; deliberately not a public multi-tenant storefront.
The operator identity is configured on the server, never supplied by the browser.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import statistics
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/review", tags=["mobile review"])
STATIC = Path(__file__).parent / "review_static"
COOKIE = "rs_review"
LANES = {"shorts": ("result", "candidate_reviews"), "highlights": ("topic_result", "topic_reviews")}


def engine():
    import telegram_intake
    return telegram_intake


def config():
    key = os.getenv("REVIEW_HUB_SECRET", "")
    chat = os.getenv("REVIEW_HUB_CHAT_ID", "")
    user = os.getenv("REVIEW_HUB_USER_ID", "")
    if not key:
        raise HTTPException(503, "Studio setup: REVIEW_HUB_SECRET is missing from the running service. Add it in Railway and deploy.")
    if len(key) < 32:
        raise HTTPException(503, "Studio setup: REVIEW_HUB_SECRET must contain at least 32 characters. Update it in Railway and deploy.")
    if not chat or not user:
        raise HTTPException(503, "Studio setup: operator identity is missing. Set REVIEW_HUB_CHAT_ID=pilot and REVIEW_HUB_USER_ID=operator in Railway and deploy.")
    return key, chat, user


def identity(request: Request):
    key, chat, user = config()
    token = request.cookies.get(COOKIE, "")
    try:
        expires, nonce, signature = token.split(".")
        expected = hmac.new(key.encode(), f"{expires}.{nonce}".encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected) or int(expires) <= time.time():
            raise ValueError()
    except (ValueError, TypeError):
        raise HTTPException(401, "Sign in to the review hub")
    if request.method not in {"GET", "HEAD"}:
        # Cross-site forms cannot supply this header; no CORS is enabled.
        if request.headers.get("x-review-request") != "1":
            raise HTTPException(403, "Missing review request header")
        if request.headers.get("sec-fetch-site") == "cross-site":
            raise HTTPException(403, "Cross-site requests are not permitted")
    return chat, user


@router.get("")
@router.get("/")
def page():
    return FileResponse(STATIC / "basic.html", headers={"Cache-Control": "no-store"})


@router.get("/app.js")
def script():
    return FileResponse(STATIC / "basic.js", media_type="text/javascript")


@router.get("/style.css")
def style():
    return FileResponse(STATIC / "style.css", media_type="text/css")


class Login(BaseModel):
    secret: str = Field(min_length=1, max_length=256)


@router.post("/api/session")
def login(body: Login, request: Request, response: Response):
    key, _, _ = config()
    if not hmac.compare_digest(body.secret, key):
        raise HTTPException(401, "Incorrect access key")
    expires, nonce = str(int(time.time()) + 43200), secrets.token_hex(16)
    signature = hmac.new(key.encode(), f"{expires}.{nonce}".encode(), hashlib.sha256).hexdigest()
    response.set_cookie(COOKIE, f"{expires}.{nonce}.{signature}", max_age=43200,
                        httponly=True, secure=request.url.scheme == "https", samesite="strict", path="/review")
    response.headers["Cache-Control"] = "no-store"
    return {"status": "signed_in"}


@router.delete("/api/session")
def logout(response: Response, owner=Depends(identity)):
    response.delete_cookie(COOKIE, path="/review")
    return {"status": "signed_out"}


def owned(db, project_id, owner):
    row = db.execute("SELECT * FROM telegram_requests WHERE request_id=? AND chat_id=? AND user_id=?",
                     (project_id, *owner)).fetchone()
    if not row:
        raise HTTPException(404, "Project not found")
    return row


def dates(row):
    created = datetime.fromisoformat(row["created_at"])
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    state = json.loads(row["state_json"])
    return created, datetime.fromisoformat(state["expires_at"]) if state.get("expires_at") else created + timedelta(days=30)


def active(row):
    if dates(row)[1] <= datetime.now(timezone.utc):
        raise HTTPException(410, "The review window has expired")


def summary(row):
    state = json.loads(row["state_json"])
    created, expires = dates(row)
    counts = {}
    for lane, (results, reviews) in LANES.items():
        items = (state.get(results) or {}).get("segments", [])
        decisions = state.get(reviews) or {}
        counts[lane] = {"total": len(items), "ready": sum(
            (decisions.get(str(i)) or {}).get("status") == "rendered" for i in range(len(items)))}
    return {"project_id": row["request_id"], "account_id": state.get("account_id", f"telegram:{row['chat_id']}:{row['user_id']}"),
            "title": state.get("vid_title") or state.get("source_title") or "New source",
            "status": row["status"], "stage": state.get("stage") or row["status"],
            "highlight_stage": state.get("topic_stage"), "created_at": created.isoformat(),
            "expires_at": expires.isoformat(), "expired": expires <= datetime.now(timezone.utc),
            "retention": "review_access_only", "counts": counts,
            "customer_reference": state.get("customer_reference", ""),
            "selected_minutes": round(float(state.get("selected_seconds", 0)) / 60, 2),
            "error": state.get("error") or state.get("topic_error"),
            "eta_seconds": None, "eta_label": "Learning processing times"}


def estimate(result, row, history):
    state = json.loads(row["state_json"])
    processing = row["status"] in {"accepted", "processing", "selecting", "retrying"}
    if not processing:
        result["eta_label"] = "Ready for your review" if row["status"] == "awaiting_review" else result["stage"].replace("_", " ")
        return result
    samples = []
    for older in history:
        old = json.loads(older["state_json"])
        if older["mode"] != row["mode"] or not old.get("selection_ready_at"):
            continue
        seconds = (datetime.fromisoformat(old["selection_ready_at"]) - dates(older)[0]).total_seconds()
        if seconds > 0:
            samples.append(seconds)
    if len(samples) >= 3:
        elapsed = (datetime.now(timezone.utc) - dates(row)[0]).total_seconds()
        remaining = statistics.median(samples[:20]) - elapsed
        result["eta_seconds"] = max(0, round(remaining))
        result["eta_label"] = f"About {max(1, round(remaining / 60))} min to review · estimate" if remaining > 0 else "Taking longer than usual · still processing"
    return result


@router.get("/api/projects")
def projects(response: Response, owner=Depends(identity)):
    e = engine()
    with e._LOCK, e._telegram_db() as db:
        rows = db.execute("SELECT * FROM telegram_requests WHERE chat_id=? AND user_id=? ORDER BY created_at DESC LIMIT 100", owner).fetchall()
    response.headers["Cache-Control"] = "no-store"
    return {"projects": [estimate(summary(row), row, rows) for row in rows]}


class Source(BaseModel):
    url: str = Field(min_length=10, max_length=2048)
    mode: Literal["both", "shorts", "topics"] = "both"
    customer_reference: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    selected_ranges: list[dict] = Field(min_length=1, max_length=4)
    plan: Literal["starter", "trial"] = "starter"
    allowance_checked: bool = False


@router.post("/api/projects", status_code=202)
def submit(body: Source, tasks: BackgroundTasks, owner=Depends(identity)):
    return submit_source(body, tasks, owner)


def submit_source(body, tasks, owner, *, project_id_override=None, clip_limits=None):
    e = engine()
    from review_basic import normalize_ranges
    missing = [key for key in ("OPENAI_API_KEY", "DRIVE_FOLDER_ID", "RIPPED_SHORTS_LOG_SHEET_ID") if not os.getenv(key, "").strip()]
    if os.getenv("RIPPED_PILOT_MODE") == "1" and missing:
        raise HTTPException(503, "Missing Railway settings: " + ", ".join(missing) + ". Add non-empty values and deploy.")
    if not body.allowance_checked:
        raise HTTPException(422, "Verify payment/trial eligibility and remaining monthly allowance first")
    try:
        parsed = e.parse_request(body.url)
        selected_ranges = normalize_ranges(body.selected_ranges)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(422, str(exc))
    selected_seconds = sum(item["end"] - item["start"] for item in selected_ranges)
    if body.plan == "trial" and selected_seconds > 900:
        raise HTTPException(422, "Trial sections must total no more than 15 minutes")
    parsed["mode"] = body.mode
    project_id, stamp = project_id_override or str(uuid.uuid4()), e.now()
    state = {"stage": "accepted", "parsed": parsed, "project_id": project_id,
             "account_id": f"pilot:{body.customer_reference}", "review_hub": True, "basic_pilot": True,
             "customer_reference": body.customer_reference, "selected_ranges": selected_ranges,
             "selected_seconds": selected_seconds, "plan": body.plan,
             "clip_limits": clip_limits,
             "expires_at": (datetime.now(timezone.utc) + timedelta(days=7 if body.plan == "trial" else 30)).isoformat()}
    with e._LOCK, e._telegram_db() as db:
        # Bound expensive in-flight work per internal operator.
        in_flight = db.execute("SELECT count(*) FROM telegram_requests WHERE chat_id=? AND user_id=? AND status IN ('accepted','processing','selecting')", owner).fetchone()[0]
        if in_flight >= 2:
            raise HTTPException(429, "Two sources are already processing")
        db.execute("INSERT INTO telegram_requests VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   (project_id, f"web-{project_id}", *owner, "accepted", body.mode,
                    parsed["source_kind"], parsed["source_value"], json.dumps(state), stamp, stamp))
    tasks.add_task(e._process, project_id)
    return {"project_id": project_id, "status": "accepted"}


@router.get("/api/projects/{project_id}")
def project(project_id: str, response: Response, owner=Depends(identity)):
    e = engine()
    with e._LOCK, e._telegram_db() as db:
        row = owned(db, project_id, owner)
        history = db.execute("SELECT * FROM telegram_requests WHERE chat_id=? AND user_id=? ORDER BY created_at DESC LIMIT 100", owner).fetchall()
    state = json.loads(row["state_json"])
    result = estimate(summary(row), row, history)
    result["transcript_ready"] = bool(state.get("transcript_text"))
    result["lanes"] = {}
    for lane, (results, reviews) in LANES.items():
        result["lanes"][lane] = [{
            "index": i, "title": item.get("title") or item.get("topic") or f"Clip {i + 1}",
            "start": item.get("start", 0), "duration": item.get("duration", 0),
            "transcript": item.get("transcript", ""),
            "status": (state.get(reviews, {}).get(str(i)) or {}).get("status", "pending"),
            "error": (state.get(reviews, {}).get(str(i)) or {}).get("render_error"),
            "asset_url": None if result["expired"] else item.get("clip_url") or item.get("segment_url"),
            "source_offset": item.get("source_offset", 0),
        } for i, item in enumerate((state.get(results) or {}).get("segments", []))]
    response.headers["Cache-Control"] = "no-store"
    return result


def candidate(state, lane, index):
    if lane not in LANES:
        raise HTTPException(422, "Unknown video format")
    results, reviews = LANES[lane]
    items = (state.get(results) or {}).get("segments", [])
    if index < 0 or index >= len(items):
        raise HTTPException(404, "Clip not found")
    return items[index], reviews


class Decision(BaseModel):
    decision: Literal["approve", "reject"]


@router.post("/api/projects/{project_id}/{lane}/{index}/decision")
def decide(project_id: str, lane: str, index: int, body: Decision, owner=Depends(identity)):
    e = engine()
    with e._LOCK, e._telegram_db() as db:
        row = owned(db, project_id, owner)
        active(row)
        state = json.loads(row["state_json"])
        _, review_key = candidate(state, lane, index)
        reviews = state.setdefault(review_key, {})
        status = (reviews.get(str(index)) or {}).get("status", "pending")
        if status in {"queued", "rendering", "rendered"}:
            if body.decision == "approve":
                return {"status": status}
            raise HTTPException(409, "This clip is already approved; its render cannot be cancelled here")
        if state.get("basic_pilot") and body.decision == "approve":
            limit = (2 if lane == "shorts" else 1) if state.get("plan") == "trial" else (10 if lane == "shorts" else 2)
            if state.get("clip_limits") is not None:
                limit = min(limit, state["clip_limits"].get(lane, 0))
            used = sum(item.get("status") in {"queued", "rendering", "rendered"} for item in reviews.values())
            if used >= limit:
                raise HTTPException(409, "This project's render limit has been reached")
        status = "queued" if body.decision == "approve" else "rejected"
        reviews[str(index)] = {"status": status, "reviewed_at": e.now(), "user_id": owner[1]}
        state.pop("render_queue_completion_signature", None)
        db.execute("UPDATE telegram_requests SET state_json=?, updated_at=? WHERE request_id=?", (json.dumps(state), e.now(), project_id))
    if body.decision == "approve":
        worker = e._render_approved if lane == "shorts" else e._render_topic_approved
        e.RENDER_EXECUTOR.submit(worker, project_id, index, owner[0])
    if lane == "shorts":
        e._safe_log_candidate(state, project_id, index, "approved" if body.decision == "approve" else "rejected", owner[1], render_status=status)
    return {"status": status}


@router.post("/api/projects/{project_id}/highlights")
def start_highlights(project_id: str, owner=Depends(identity)):
    """Release the saved highlight lane without completing/rejecting Shorts."""
    e = engine()
    with e._LOCK, e._telegram_db() as db:
        row = owned(db, project_id, owner)
        active(row)
        state = json.loads(row["state_json"])
        if row["mode"] != "both" or not state.get("topic_source_segments"):
            raise HTTPException(409, "The shared source transcript is not ready for highlights")
        if state.get("topic_stage") not in {None, "failed"}:
            return {"status": state["topic_stage"]}
        state["topic_stage"] = "queued"
        db.execute("UPDATE telegram_requests SET state_json=?, updated_at=? WHERE request_id=?", (json.dumps(state), e.now(), project_id))
    e.RENDER_EXECUTOR.submit(e._start_16_9_after_confirmation, project_id, owner[0])
    return {"status": "queued"}


@router.get("/api/projects/{project_id}/source")
def source_video(project_id: str, lane: str = "shorts", index: int = 0, owner=Depends(identity)):
    e = engine()
    with e._LOCK, e._telegram_db() as db:
        row = owned(db, project_id, owner)
    active(row)
    state = json.loads(row["state_json"])
    item, _ = candidate(state, lane, index)
    path = Path(item.get("review_source_path") or state.get("video_path") or "").resolve()
    root = e.SOURCE_DIR.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(404, "Source preview is unavailable; use the finished Drive render")
    return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "private, no-store"})


@router.get("/api/projects/{project_id}/transcript.txt")
def transcript_download(project_id: str, owner=Depends(identity)):
    e = engine()
    with e._LOCK, e._telegram_db() as db:
        row = owned(db, project_id, owner)
    active(row)
    state = json.loads(row["state_json"])
    text = state.get("transcript_text")
    if not text:
        raise HTTPException(404, "The selected-section transcript is not ready")
    return Response(text, media_type="text/plain", headers={"Content-Disposition": 'attachment; filename="transcript.txt"', "Cache-Control": "no-store"})


@router.post("/api/projects/{project_id}/retry", status_code=202)
def retry(project_id: str, owner=Depends(identity)):
    e = engine()
    with e._LOCK, e._telegram_db() as db:
        row = owned(db, project_id, owner)
        active(row)
        if row["status"] != "error":
            raise HTTPException(409, "Only failed source jobs can be retried here")
        db.execute("UPDATE telegram_requests SET status=?, updated_at=? WHERE request_id=?", ("retrying", e.now(), project_id))
    e.RENDER_EXECUTOR.submit(e._process, project_id)
    return {"status": "retrying"}


@router.get("/api/projects/{project_id}/handoff.json")
def handoff_manifest(project_id: str, owner=Depends(identity)):
    """Reviewable hook for a later scheduler adapter; this route never publishes."""
    e = engine()
    with e._LOCK, e._telegram_db() as db:
        row = owned(db, project_id, owner)
    active(row)
    state = json.loads(row["state_json"])
    info = summary(row)
    assets = []
    for lane, (results, reviews) in LANES.items():
        for index, item in enumerate((state.get(results) or {}).get("segments", [])):
            if state.get(reviews, {}).get(str(index), {}).get("status") != "rendered":
                continue
            assets.append({"asset_id": f"{project_id}:{lane}:{index}", "project_id": project_id,
                "account_id": info["account_id"], "aspect_ratio": "9:16" if lane == "shorts" else "16:9",
                "drive_url": item.get("clip_url") or item.get("segment_url"),
                "expires_at": info["expires_at"]})
    payload = {"contract_version": 1, "project_id": project_id, "account_id": info["account_id"],
               "request_id": project_id, "assets": assets, "action": "review_before_scheduling"}
    return Response(json.dumps(payload, indent=2), media_type="application/json", headers={
        "Content-Disposition": 'attachment; filename="r3cycle-handoff.json"', "Cache-Control": "no-store"})
