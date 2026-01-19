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
    youtube-transcript3 ONLY, tries:
      1) /api/transcript?videoId=...
      2) /api/transcript-with-url?url=...
    Omits 'lang' when TRANSCRIPT_LANG=auto (provider behaves better that way).
    """
    if not RAPIDAPI_KEY:
        raise RuntimeError("RAPIDAPI_KEY not configured")

    headers = {
        "x-rapidapi-host": YTT3_HOST,
        "x-rapidapi-key": RAPIDAPI_KEY,
    }

    # Force timestamps
    flat_text = "false"

    lang_env = os.getenv("TRANSCRIPT_LANG", "auto").strip().lower()
    include_lang = (lang_env not in ("", "auto", "none"))

    video_url = f"https://www.youtube.com/watch?v={video_id}"

    def _call(endpoint: str, params: dict) -> dict:
        resp = requests.get(endpoint, headers=headers, params=params, timeout=(10, 120))
        try:
            data = resp.json()
        except Exception:
            raise RuntimeError(f"Transcript API non-JSON ({resp.status_code}): {resp.text}")

        logger.info(
            f"Transcript3 call={endpoint.split('/')[-1]} "
            f"status={resp.status_code} success={data.get('success')}"
        )

        if resp.status_code != 200:
            raise RuntimeError(f"Transcript API HTTP error ({resp.status_code}): {data}")

        if not data.get("success"):
            # Preserve exact provider error text for retry decision
            raise RuntimeError(f"TRANSCRIPT3_FAIL::{data}")

        return data

    # Attempt 1: by videoId
    params1 = {"videoId": video_id, "flat_text": flat_text}
    if include_lang:
        params1["lang"] = lang_env

    try:
        data = _call(YTT3_URL, params1)
    except Exception as e1:
        logger.warning(f"Transcript3 videoId attempt failed: {e1}")

        # Attempt 2: by URL (same provider)
        url_endpoint = "https://youtube-transcript3.p.rapidapi.com/api/transcript-with-url"
        params2 = {"url": video_url, "flat_text": flat_text}
        if include_lang:
            params2["lang"] = lang_env
        data = _call(url_endpoint, params2)

    transcript = data.get("transcript")

    if isinstance(transcript, str):
        raise RuntimeError("Provider returned flat text; timestamps unavailable (flat_text must be false).")

    if not transcript or not isinstance(transcript, list):
        raise RuntimeError(f"Transcript empty or malformed: {data}")

    segments: List[dict] = []
    for t in transcript:
        text = (t.get("text") or "").strip()
        if not text:
            continue
        segments.append({
            "start": float(t.get("offset", 0.0)),
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
    try:
        JOBS[job_id]["step"] = "transcript_fetch"

        # Retry strategy for transient provider failures
        backoff_seconds = [0, 15, 45, 120]  # total ~3 minutes max
        last_err = None

        for attempt, wait_s in enumerate(backoff_seconds, start=1):
            if wait_s:
                logger.info(f"[{job_id}] transcript retry in {wait_s}s (attempt {attempt}/{len(backoff_seconds)})")
                time.sleep(wait_s)

            try:
                transcript = get_transcript(video_id)
                logger.info(f"[{job_id}] transcript segments={len(transcript)} (attempt {attempt})")
                last_err = None
                break
            except Exception as e:
                last_err = str(e)
                logger.warning(f"[{job_id}] transcript attempt {attempt} failed: {last_err}")

                # Only keep retrying if provider says "not available at the moment"
                if "not available at the moment" not in last_err.lower():
                    break

        if last_err is not None:
            JOBS[job_id]["status"] = "blocked"
            JOBS[job_id]["step"] = "transcript_unavailable"
            JOBS[job_id]["error"] = last_err
            JOBS[job_id]["elapsed_s"] = round(time.time() - started, 2)
            logger.warning(f"[{job_id}] blocked: transcript unavailable after retries")
            return

        # continue with chunking...

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
