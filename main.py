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
# LOGGING (shows in Railway logs)
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
# (OK for now; later migrate to Redis/DB for persistence)
# -------------------------
executor = ThreadPoolExecutor(max_workers=int(os.getenv("WORKERS", "2")))
JOBS: Dict[str, Dict[str, Any]] = {}

# -------------------------
# ENV (RapidAPI)
# -------------------------
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

# Your correct RapidAPI product:
# curl --url https://video-transcript-scraper.p.rapidapi.com/transcript/youtube
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST", "video-transcript-scraper.p.rapidapi.com")
RAPIDAPI_URL = os.getenv(
    "RAPIDAPI_URL",
    "https://video-transcript-scraper.p.rapidapi.com/transcript/youtube",
)

if not RAPIDAPI_KEY:
    logger.warning("RAPIDAPI_KEY not set (discover will fail until configured).")

# -------------------------
# MODELS
# -------------------------
class DiscoverRequest(BaseModel):
    video_id: str = Field(..., min_length=6)
    youtube_url: Optional[str] = None  # optional; we derive from video_id if missing

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
# TRANSCRIPT (RapidAPI: video-transcript-scraper)
# -------------------------
def get_transcript(video_id: str) -> List[dict]:
    """
    Uses RapidAPI 'video-transcript-scraper' endpoint.
    Expected response usually includes a list under 'transcript'.
    We log keys to make debugging easy if format differs.
    """
    if not RAPIDAPI_KEY:
        raise RuntimeError("RAPIDAPI_KEY not configured")

    video_url = f"https://www.youtube.com/watch?v={video_id}"

    payload = {"video_url": video_url}

    headers = {
        "Content-Type": "application/json",
        "x-rapidapi-host": RAPIDAPI_HOST,
        "x-rapidapi-key": RAPIDAPI_KEY,
    }

    resp = requests.post(
        RAPIDAPI_URL,
        json=payload,
        headers=headers,
        timeout=(10, 180),  # connect, read (long videos can take time)
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Transcript API failed ({resp.status_code}): {resp.text}")

    data = resp.json()

    # Debug visibility: what did the API return?
    try:
        logger.info(f"RapidAPI response keys: {list(data.keys())}")
    except Exception:
        logger.info("RapidAPI response keys: <unable to list>")

    # Common key for this API
    transcript = data.get("transcript")

    # Some APIs return segments under other keys; defensively check a few:
    if not transcript:
        transcript = data.get("segments") or data.get("results") or data.get("data")

    if not transcript or not isinstance(transcript, list):
        # Show a truncated payload to help debug schema mismatches (safe length)
        logger.info(f"RapidAPI raw response (truncated): {str(data)[:1200]}")
        raise RuntimeError("Transcript empty or malformed from RapidAPI")

    segments: List[dict] = []
    for t in transcript:
        # Expected: {"text": "...", "start": 12.3, "duration": 4.1}
        text = (t.get("text") or "").strip()
        if not text:
            continue

        segments.append(
            {
                "start": float(t.get("start", 0.0)),
                "duration": float(t.get("duration", 0.0)),
                "text": text,
            }
        )

    if not segments:
        raise RuntimeError("Transcript contained no usable text segments")

    return segments

# -------------------------
# CHUNKING (3hr-safe)
# -------------------------
def chunk_transcript(segments: List[dict], chunk_seconds: int = 120) -> List[dict]:
    """
    Chunk transcript into time windows.
    We keep segments inside each chunk for downstream AI work,
    but we do NOT return them to Zapier.
    """
    chunks: List[dict] = []
    current: List[dict] = []
    current_start = segments[0]["start"]
    total = 0.0

    for s in segments:
        dur = float(s.get("duration", 0.0))
        if (total + dur) > chunk_seconds and current:
            chunks.append(
                {
                    "start": float(current_start),
                    "end": float(current_start + total),
                    "segments": current,
                }
            )
            current = []
            current_start = s["start"]
            total = 0.0

        current.append(s)
        total += dur

    if current:
        chunks.append(
            {
                "start": float(current_start),
                "end": float(current_start + total),
                "segments": current,
            }
        )

    return chunks

# -------------------------
# BACKGROUND JOB: DISCOVERY
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

        # NOTE: Next steps (not implemented yet):
        # - AI clip discovery per chunk
        # - Merge & rank globally; cap at 30
        # - Assign clip_index 1..30 and write rows to Google Sheets
        #
        # For now, we stop here to prove transcript+chunking is stable.

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
        raise HTTPException(status_code=500, detail="RAPIDAPI_KEY not configured")

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "job_id": job_id,
        "video_id": req.video_id,
        "status": "queued",
        "step": "queued",
        "created_at": time.time(),
    }

    # Offload to background thread so Zapier gets immediate response
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
