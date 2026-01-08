import os
import uuid
import time
import logging
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# -------------------------
# LOGGING (visible in Railway logs)
# -------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("ripped-shorts")

# -------------------------
# APP
# -------------------------
app = FastAPI(title="Ripped Shorts Backend")

# -------------------------
# EXECUTOR + IN-MEM JOB STORE
# (good enough for now; later move to Redis/DB)
# -------------------------
executor = ThreadPoolExecutor(max_workers=int(os.getenv("WORKERS", "2")))
JOBS: Dict[str, Dict[str, Any]] = {}

# -------------------------
# ENV
# -------------------------
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")
RAPIDAPI_HOST = "youtube-transcripts-transcribe-youtube-video-to-text.p.rapidapi.com"
RAPIDAPI_URL = "https://youtube-transcripts-transcribe-youtube-video-to-text.p.rapidapi.com/transcribe"

if not RAPIDAPI_KEY:
    logger.warning("RAPIDAPI_KEY not set (discover will fail until configured).")

# -------------------------
# MODELS
# -------------------------
class DiscoverRequest(BaseModel):
    video_id: str = Field(..., min_length=6)
    youtube_url: Optional[str] = None  # optional convenience

class DiscoverResponse(BaseModel):
    status: str
    job_id: str
    video_id: str

# -------------------------
# HEALTH
# -------------------------
@app.get("/ping")
def ping():
    return {"pong": True}

@app.get("/")
def health():
    return {"status": "ok"}

# -------------------------
# RapidAPI transcript fetch
# -------------------------
def get_transcript(video_id: str) -> List[dict]:
    if not RAPIDAPI_KEY:
        raise RuntimeError("RAPIDAPI_KEY not configured")

    payload = {"url": f"https://www.youtube.com/watch?v={video_id}"}
    headers = {
        "Content-Type": "application/json",
        "x-rapidapi-host": RAPIDAPI_HOST,
        "x-rapidapi-key": RAPIDAPI_KEY,
    }

    # Use sensible timeouts: (connect, read)
    resp = requests.post(RAPIDAPI_URL, json=payload, headers=headers, timeout=(10, 120))

    if resp.status_code != 200:
        raise RuntimeError(f"Transcript API failed ({resp.status_code}): {resp.text}")

    data = resp.json()
    transcript = data.get("transcript", [])
    if not transcript:
        raise RuntimeError("Transcript returned empty")

    segments = []
    for t in transcript:
        segments.append({
            "start": float(t.get("start", 0.0)),
            "duration": float(t.get("duration", 0.0)),
            "text": (t.get("text") or "").strip()
        })

    # Basic sanity
    segments = [s for s in segments if s["text"]]
    if not segments:
        raise RuntimeError("Transcript had no usable text segments")

    return segments

# -------------------------
# Chunking (3hr-safe)
# -------------------------
def chunk_transcript(segments: List[dict], chunk_seconds: int = 120) -> List[dict]:
    """
    Chunk by time window. 120s chunks is good for long videos.
    We do NOT return chunks to Zapier — internal use only.
    """
    chunks = []
    current = []
    current_start = segments[0]["start"]
    total = 0.0

    for s in segments:
        if total + s["duration"] > chunk_seconds and current:
            chunks.append({
                "start": float(current_start),
                "end": float(current_start + total),
                "segments": current
            })
            current = []
            current_start = s["start"]
            total = 0.0

        current.append(s)
        total += float(s["duration"])

    if current:
        chunks.append({
            "start": float(current_start),
            "end": float(current_start + total),
            "segments": current
        })

    return chunks

# -------------------------
# Background job (DISCOVERY)
# -------------------------
def run_discovery(job_id: str, video_id: str):
    started = time.time()
    JOBS[job_id]["status"] = "running"
    JOBS[job_id]["step"] = "transcript_fetch"

    logger.info(f"[{job_id}] discovery start video_id={video_id}")

    try:
        transcript = get_transcript(video_id)
        logger.info(f"[{job_id}] transcript segments={len(transcript)}")

        JOBS[job_id]["step"] = "chunking"
        chunks = chunk_transcript(transcript, chunk_seconds=120)
        logger.info(f"[{job_id}] chunks={len(chunks)}")

        # TODO (next):
        # - AI clip discovery per chunk
        # - global ranking / prune <= 30
        # - write proposed clips to Google Sheet
        #
        # For now, we mark transcript ready and chunked.
        # This keeps discover stable while you wire Sheets/AI cleanly.

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["step"] = "completed"
        JOBS[job_id]["result"] = {
            "video_id": video_id,
            "segments": len(transcript),
            "chunks": len(chunks),
        }
        JOBS[job_id]["elapsed_s"] = round(time.time() - started, 2)

        logger.info(f"[{job_id}] discovery done elapsed_s={JOBS[job_id]['elapsed_s']}")

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["step"] = "failed"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["elapsed_s"] = round(time.time() - started, 2)

        logger.exception(f"[{job_id}] discovery failed: {e}")

# -------------------------
# DISCOVER ENDPOINT (Zapier-safe)
# -------------------------
@app.post("/discover", response_model=DiscoverResponse, status_code=202)
def discover(req: DiscoverRequest):
    if not RAPIDAPI_KEY:
        # Return a clear error early rather than accepting jobs that cannot run.
        raise HTTPException(status_code=500, detail="RAPIDAPI_KEY not configured")

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "job_id": job_id,
        "video_id": req.video_id,
        "status": "queued",
        "step": "queued",
        "created_at": time.time(),
    }

    # Offload to background thread
    executor.submit(run_discovery, job_id, req.video_id)

    # IMPORTANT: return immediately, no transcript, no chunks
    return {"status": "accepted", "job_id": job_id, "video_id": req.video_id}

# -------------------------
# JOB STATUS (debug endpoint)
# -------------------------
@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job
