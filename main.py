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
# LOGGING
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
# EXECUTOR + IN-MEM JOBS
# -------------------------
executor = ThreadPoolExecutor(max_workers=int(os.getenv("WORKERS", "2")))
JOBS: Dict[str, Dict[str, Any]] = {}

# -------------------------
# ENV
# -------------------------
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
TRANSCRIPT_LANG = os.getenv("TRANSCRIPT_LANG", "auto")  # auto recommended
TRANSCRIPT_FLAT_TEXT = os.getenv("TRANSCRIPT_FLAT_TEXT", "false").lower() in ("1", "true", "yes")

YTT3_HOST = "youtube-transcript3.p.rapidapi.com"
YTT3_URL = "https://youtube-transcript3.p.rapidapi.com/api/transcript"

if not RAPIDAPI_KEY:
    logger.warning("RAPIDAPI_KEY not set (discover will fail until configured).")

# -------------------------
# MODELS
# -------------------------
class DiscoverRequest(BaseModel):
    video_id: str = Field(..., min_length=6)
    youtube_url: Optional[str] = None  # ignored by transcript3, kept for future use

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
# TRANSCRIPT (youtube-transcript3)
# -------------------------
def get_transcript(video_id: str) -> List[dict]:
    """
    Uses RapidAPI 'youtube-transcript3' only.
    GET /api/transcript?videoId=...&lang=auto&flat_text=false

    Success:
      { "success": true, "transcript": [ { "text":..., "duration":..., "offset":... }, ... ] }

    Failure (often still 200):
      { "success": false, "error": "..." }
    """
    if not RAPIDAPI_KEY:
        raise RuntimeError("RAPIDAPI_KEY not configured")

    headers = {
        "x-rapidapi-host": YTT3_HOST,
        "x-rapidapi-key": RAPIDAPI_KEY,
    }

    # IMPORTANT: flat_text must be false to preserve timestamps
    params = {
        "videoId": video_id,
        "lang": TRANSCRIPT_LANG,
        "flat_text": "true" if TRANSCRIPT_FLAT_TEXT else "false",
    }

    resp = requests.get(YTT3_URL, headers=headers, params=params, timeout=(10, 120))

    try:
        data = resp.json()
    except Exception:
        raise RuntimeError(f"Transcript API non-JSON response ({resp.status_code}): {resp.text}")

    if resp.status_code != 200:
        raise RuntimeError(f"Transcript API failed ({resp.status_code}): {data}")

    if not data.get("success"):
        # Example: {"success": false, "error": "YouTube video ID is required"}
        raise RuntimeError(f"Transcript API reported failure: {data}")

    transcript = data.get("transcript")

    # If flat_text=true, some versions may return transcript as a string
    if isinstance(transcript, str):
        raise RuntimeError("flat_text=true returns no timestamps; set TRANSCRIPT_FLAT_TEXT=false")

    if not transcript or not isinstance(transcript, list):
        raise RuntimeError(f"Transcript returned empty or malformed: {data}")

    segments: List[dict] = []
    for t in transcript:
        text = (t.get("text") or "").strip()
        if not text:
            continue
        segments.append({
            "start": float(t.get("offset", 0.0)),     # <-- offset is the start time
            "duration": float(t.get("duration", 0.0)),
            "text": text,
        })

    if not segments:
        raise RuntimeError("Transcript contained no usable segments")

    return segments

# -------------------------
# CHUNKING
# -------------------------
def chunk_transcript(segments: List[dict], chunk_seconds: int = 120) -> List[dict]:
    """
    Chunk transcript into time windows. We keep segments per chunk for AI discovery later.
    """
    chunks: List[dict] = []
    current: List[dict] = []
    current_start = segments[0]["start"]
    total = 0.0

    for s in segments:
        dur = float(s.get("duration", 0.0))

        if (total + dur) > chunk_seconds and current:
            chunks.append({
                "start": float(current_start),
                "end": float(current_start + total),
                "segments": current,
            })
            current = []
            current_start = s["start"]
            total = 0.0

        current.append(s)
        total += dur

    if current:
        chunks.append({
            "start": float(current_start),
            "end": float(current_start + total),
            "segments": current,
        })

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

        # Next steps to add AFTER transcript stability:
        # - AI clip discovery per chunk
        # - global merge/rank, cap <= 30
        # - write proposed clips to Google Sheets

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
        msg = str(e)

        # Treat transcript absence as BLOCKED (not a system error)
        # youtube-transcript3 failures are often returned with success:false and an error string.
        if "Transcript API reported failure" in msg and ("does" in msg or "exist" in msg or "not" in msg):
            JOBS[job_id]["status"] = "blocked"
            JOBS[job_id]["step"] = "transcript_unavailable"
            JOBS[job_id]["error"] = msg
            JOBS[job_id]["elapsed_s"] = round(time.time() - started, 2)
            logger.warning(f"[{job_id}] blocked: transcript unavailable ({msg})")
            return

        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["step"] = "failed"
        JOBS[job_id]["error"] = msg
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

    executor.submit(run_discovery, job_id, req.video_id)
    return {"status": "accepted", "job_id": job_id, "video_id": req.video_id}

# -------------------------
# JOB STATUS (debug)
# -------------------------
@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job
