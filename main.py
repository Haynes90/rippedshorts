import os
import time
import uuid
import logging
from typing import List, Dict, Any
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
RAPIDAPI_HOST = "youtube-transcript3.p.rapidapi.com"
RAPIDAPI_URL = f"https://{RAPIDAPI_HOST}/api/transcript"

if not RAPIDAPI_KEY:
    logger.warning("RAPIDAPI_KEY not set (discover will fail until configured).")

# -------------------------
# MODELS
# -------------------------
class DiscoverRequest(BaseModel):
    video_id: str = Field(..., min_length=6)


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

    headers = {
        "x-rapidapi-host": RAPIDAPI_HOST,
        "x-rapidapi-key": RAPIDAPI_KEY,
    }

    resp = requests.get(
        RAPIDAPI_URL,
        headers=headers,
        params={"videoId": video_id},
        timeout=(10, 120),
    )

    try:
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(f"Transcript API non-JSON ({resp.status_code}): {resp.text}") from exc

    logger.info(
        "Transcript3 call=transcript status=%s success=%s",
        resp.status_code,
        data.get("success"),
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Transcript API HTTP error ({resp.status_code}): {data}")

    if not data.get("success"):
        raise RuntimeError(f"Transcript API reported failure: {data}")

    transcript = data.get("transcript")
    if not transcript or not isinstance(transcript, list):
        raise RuntimeError(f"Transcript empty or malformed: {data}")

    segments: List[dict] = []
    for entry in transcript:
        text = (entry.get("text") or "").strip()
        if not text:
            continue
        start = entry.get("start")
        if start is None:
            start = entry.get("offset", 0.0)
        segments.append({
            "start": float(start or 0.0),
            "duration": float(entry.get("duration", 0.0)),
            "text": text,
        })

    if not segments:
        raise RuntimeError("Transcript contained no usable segments")

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

    for segment in segments:
        if total + segment["duration"] > chunk_seconds and current:
            chunks.append({
                "start": float(current_start),
                "end": float(current_start + total),
                "segments": current,
            })
            current = []
            current_start = segment["start"]
            total = 0.0

        current.append(segment)
        total += float(segment["duration"])

    if current:
        chunks.append({
            "start": float(current_start),
            "end": float(current_start + total),
            "segments": current,
        })

    return chunks

# -------------------------
# Background job (DISCOVERY)
# -------------------------

def run_discovery(job_id: str, video_id: str):
    started = time.time()
    JOBS[job_id]["status"] = "running"
    JOBS[job_id]["step"] = "transcript_fetch"

    logger.info("[%s] discovery start video_id=%s", job_id, video_id)

    try:
        transcript = get_transcript(video_id)
        logger.info("[%s] transcript segments=%s", job_id, len(transcript))

        JOBS[job_id]["step"] = "chunking"
        chunks = chunk_transcript(transcript, chunk_seconds=120)
        logger.info("[%s] chunks=%s", job_id, len(chunks))

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["step"] = "completed"
        JOBS[job_id]["result"] = {
            "video_id": video_id,
            "segments": len(transcript),
            "chunks": len(chunks),
        }
        JOBS[job_id]["elapsed_s"] = round(time.time() - started, 2)

        logger.info("[%s] discovery done elapsed_s=%s", job_id, JOBS[job_id]["elapsed_s"])

    except Exception as exc:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["step"] = "failed"
        JOBS[job_id]["error"] = str(exc)
        JOBS[job_id]["elapsed_s"] = round(time.time() - started, 2)

        logger.exception("[%s] discovery failed: %s", job_id, exc)

# -------------------------
# DISCOVER ENDPOINT (Zapier-safe)
# -------------------------
@app.post("/discover", response_model=DiscoverResponse, status_code=202)
def discover(req: DiscoverRequest):
    if not RAPIDAPI_KEY:
        raise HTTPException(status_code=500, detail="RAPIDAPI_KEY not configured")

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "job_id": job_id,
        "video_id": req.video_id,
        "status": "queued",
        "step": "queued",
        "created_at": time.time(),
    }

    executor.submit(run_discovery, job_id, req.video_id)

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
