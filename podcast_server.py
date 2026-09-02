import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field

from main import app, get_transcript, create_transcript_doc, resolve_video_id, JOBS, executor, logger
from podcast_pipeline import run_podcast_metadata_queue
from schedule_route_gate import install_route_gate

# Ripped Shorts owns Telegram intake. Install the route gate after main/telegram_intake
# are loaded so only YouTube route resolution is wrapped; all existing review/render
# commands continue to use the original handler.
install_route_gate()


class PodcastMetadataRequest(BaseModel):
    video_id: Optional[str] = Field(None, min_length=6)
    youtube_url: Optional[str] = None
    playlist_id: Optional[str] = None
    source_title: Optional[str] = None
    mp3_url: Optional[str] = None
    transcript_doc_url: Optional[str] = None
    transcript_json_url: Optional[str] = None
    wait_for_result: bool = False
    wait_timeout_s: int = Field(0, ge=0, le=900)


def run_podcast_metadata_job(job_id: str, req: PodcastMetadataRequest, resolved_video_id: str):
    started = time.time()
    JOBS[job_id]["status"] = "running"
    JOBS[job_id]["step"] = "podcast_transcript_fetch"
    try:
        transcript = get_transcript(resolved_video_id)
        JOBS[job_id]["step"] = "podcast_transcript_doc"
        doc_info = create_transcript_doc(resolved_video_id, transcript)
        transcript_doc_url = req.transcript_doc_url or doc_info.get("document_url", "")

        JOBS[job_id]["step"] = "podcast_metadata_queue"
        result = run_podcast_metadata_queue(
            video_id=resolved_video_id,
            playlist_id=req.playlist_id,
            source_title=req.source_title or "",
            transcript_segments=transcript,
            transcript_doc_url=transcript_doc_url,
            transcript_json_url=req.transcript_json_url or "",
            mp3_url=req.mp3_url or "",
            audio_path=None,
            job_id=job_id,
        )
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["step"] = "completed"
        JOBS[job_id]["result"] = result
        JOBS[job_id]["elapsed_s"] = round(time.time() - started, 2)
        logger.info("[%s] podcast metadata done elapsed_s=%s", job_id, JOBS[job_id]["elapsed_s"])
    except Exception as exc:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["step"] = "failed"
        JOBS[job_id]["error"] = str(exc)
        JOBS[job_id]["elapsed_s"] = round(time.time() - started, 2)
        logger.exception("[%s] podcast metadata failed: %s", job_id, exc)


@app.post("/podcast/metadata", status_code=202)
def podcast_metadata(req: PodcastMetadataRequest):
    try:
        resolved_video_id = resolve_video_id(req.video_id, req.youtube_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "job_id": job_id,
        "video_id": resolved_video_id,
        "youtube_url": req.youtube_url,
        "playlist_id": req.playlist_id,
        "source_title": req.source_title,
        "mp3_url": req.mp3_url,
        "status": "queued",
        "step": "queued",
        "created_at": time.time(),
        "type": "podcast_metadata",
    }
    executor.submit(run_podcast_metadata_job, job_id, req, resolved_video_id)

    if req.wait_for_result:
        timeout = req.wait_timeout_s if req.wait_timeout_s > 0 else 180
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = JOBS.get(job_id, {})
            status = job.get("status")
            if status in {"done", "error"}:
                return {
                    "status": status,
                    "job_id": job_id,
                    "video_id": resolved_video_id,
                    "step": job.get("step"),
                    "error": job.get("error"),
                    "elapsed_s": job.get("elapsed_s"),
                    "result": job.get("result"),
                }
            time.sleep(1)

    return {"status": "accepted", "job_id": job_id, "video_id": resolved_video_id, "step": "queued"}
