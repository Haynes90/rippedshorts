import os
import time
from typing import Any

import requests
from fastapi import HTTPException
from pydantic import BaseModel, Field

import podcast_server
from main import JOBS, app, call_openai_for_clips, executor, logger

RUNTIME_VERSION = "church-chapter-jobs-v1.0"
TELEGRAM_BOT_TOKEN = str(os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or "").strip()
TELEGRAM_CHAT_ID = str(os.getenv("TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_CHANNEL_ID") or "").strip()


class TimedSegment(BaseModel):
    start_seconds: float = 0
    end_seconds: float = 0
    text: str


class ChurchClipJobRequest(BaseModel):
    job_id: str
    workflow: str = "church_chapter_review"
    source_video_url: str | None = None
    source_audio: str | None = None
    episode: dict[str, Any] = Field(default_factory=dict)
    prompt_reference: str | None = None
    podhome_destination: dict[str, Any] = Field(default_factory=dict)
    timed_transcript: list[TimedSegment]
    chapter_settings: dict[str, Any] = Field(default_factory=dict)


def _telegram(message: str) -> dict:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return {"sent": False, "reason": "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is not configured"}
    response = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "disable_web_page_preview": True},
        timeout=30,
    )
    response.raise_for_status()
    return {"sent": True, "response": response.json()}


def _chapter_prompt(req: ChurchClipJobRequest) -> str:
    return (
        "Analyze this church-service transcript and return strict JSON with a segments array. "
        "Identify complete publishable chapters, especially sermon and sermonette. "
        "Each segment must include title, category, start, duration, reason, and transcript. "
        "Do not invent timestamps. Human approval is required before publishing. "
        f"Prompt reference for audit: {req.prompt_reference or 'none'}"
    )


def _run_church_job(job_id: str, req: ChurchClipJobRequest) -> None:
    started = time.time()
    job = JOBS[job_id]
    job["status"] = "running"
    job["step"] = "chapter_analysis"
    try:
        transcript = []
        for segment in req.timed_transcript:
            start = float(segment.start_seconds)
            end = float(segment.end_seconds)
            transcript.append({
                "start": start,
                "duration": max(0.0, end - start),
                "text": segment.text,
                "video_id": req.episode.get("youtube_id") or "church",
            })
        result = call_openai_for_clips(transcript, _chapter_prompt(req))
        chapters = result.get("segments") or []
        job["result"] = {
            "chapters": chapters,
            "analysis": result.get("analysis") or {},
            "approval_required": True,
            "podhome_destination": req.podhome_destination,
        }
        job["status"] = "waiting_for_approval"
        job["step"] = "chapter_approval"
        job["elapsed_s"] = round(time.time() - started, 2)
        title = req.episode.get("title") or "Church Service"
        show = req.episode.get("show") or req.episode.get("show_code") or "Church"
        lines = [
            f"📖 Chapters ready for approval",
            f"Show: {show}",
            f"Service: {title}",
            f"Job: {job_id}",
            f"Chapters found: {len(chapters)}",
        ]
        for index, chapter in enumerate(chapters[:8], start=1):
            name = chapter.get("title") or chapter.get("category") or f"Chapter {index}"
            start = chapter.get("start", 0)
            duration = chapter.get("duration", 0)
            lines.append(f"{index}. {name} — start {start}s, duration {duration}s")
        lines.append("Open Clip Master /jobs to review before Podhome publishing.")
        job["telegram"] = _telegram("\n".join(lines))
        logger.info("[%s] church chapters ready count=%s telegram=%s", job_id, len(chapters), job["telegram"].get("sent"))
    except Exception as exc:
        job["status"] = "error"
        job["step"] = "failed"
        job["error"] = str(exc)
        job["elapsed_s"] = round(time.time() - started, 2)
        logger.exception("[%s] church chapter job failed", job_id)


@app.post("/clip-jobs", status_code=202)
def create_clip_job(req: ChurchClipJobRequest):
    if req.job_id in JOBS:
        existing = JOBS[req.job_id]
        raise HTTPException(status_code=409, detail={"job_id": req.job_id, "status": existing.get("status")})
    JOBS[req.job_id] = {
        "job_id": req.job_id,
        "type": "church_chapter_review",
        "workflow": req.workflow,
        "show_code": req.episode.get("show_code"),
        "show": req.episode.get("show"),
        "title": req.episode.get("title"),
        "source_video_url": req.source_video_url,
        "source_audio": req.source_audio,
        "prompt_reference": req.prompt_reference,
        "podhome_destination": req.podhome_destination,
        "status": "queued",
        "step": "queued",
        "created_at": time.time(),
    }
    executor.submit(_run_church_job, req.job_id, req)
    return {"job_id": req.job_id, "status": "queued", "status_url": f"/clip-jobs/{req.job_id}"}


@app.get("/clip-jobs")
def list_clip_jobs(status: str | None = None):
    jobs = [job for job in JOBS.values() if job.get("type") == "church_chapter_review"]
    if status:
        jobs = [job for job in jobs if job.get("status") == status]
    jobs.sort(key=lambda item: item.get("created_at", 0), reverse=True)
    return {"jobs": jobs, "count": len(jobs), "runtime_version": RUNTIME_VERSION}


@app.get("/clip-jobs/{job_id}")
def get_clip_job(job_id: str):
    job = JOBS.get(job_id)
    if not job or job.get("type") != "church_chapter_review":
        raise HTTPException(status_code=404, detail="clip job not found")
    return job


@app.get("/approvals")
def pending_approvals():
    jobs = [job for job in JOBS.values() if job.get("type") == "church_chapter_review" and job.get("status") == "waiting_for_approval"]
    jobs.sort(key=lambda item: item.get("created_at", 0), reverse=True)
    return {"jobs": jobs, "count": len(jobs)}


@app.get("/clip-master-info")
def clip_master_info():
    return {
        "runtime_version": RUNTIME_VERSION,
        "clip_jobs_endpoint": "/clip-jobs",
        "pending_approvals_endpoint": "/approvals",
        "telegram_configured": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
    }


print(f"CLIP MASTER STARTUP: {RUNTIME_VERSION}")
