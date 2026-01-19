import os
import uuid
import timediff --git a/main.py b/main.py
index c9b54b2e6e16961bf40a386cb5597cf043a0a817..4ae9aca15d4ce812e243dc47d9c4f39b9a149853 100644
--- a/main.py
+++ b/main.py
@@ -1,13 +1,15 @@
 import os
+import re
 import uuid
 import time
 import logging
 from typing import List, Optional, Dict, Any
+from urllib.parse import urlparse, parse_qs
 from concurrent.futures import ThreadPoolExecutor
 
 import requests
 from fastapi import FastAPI, HTTPException
-from pydantic import BaseModel, Field
+from pydantic import BaseModel, Field, root_validator, validator
 
 # -------------------------
 # LOGGING (visible in Railway logs)
@@ -44,10 +46,52 @@ if not RAPIDAPI_KEY:
 # -------------------------
 # MODELS
 # -------------------------
+YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,20}$")
+
+
+def extract_video_id(youtube_url: str) -> Optional[str]:
+    parsed = urlparse(youtube_url)
+    host = parsed.netloc.lower()
+    if "youtube.com" in host:
+        query = parse_qs(parsed.query)
+        if "v" in query and query["v"]:
+            return query["v"][0]
+        if parsed.path.startswith("/shorts/"):
+            return parsed.path.split("/shorts/")[-1].split("/")[0]
+        if parsed.path.startswith("/embed/"):
+            return parsed.path.split("/embed/")[-1].split("/")[0]
+    if "youtu.be" in host:
+        return parsed.path.lstrip("/").split("/")[0]
+    return None
+
+
 class DiscoverRequest(BaseModel):
-    video_id: str = Field(..., min_length=6)
+    video_id: Optional[str] = Field(None, min_length=6)
     youtube_url: Optional[str] = None  # optional convenience
 
+    @root_validator(pre=True)
+    def require_id_or_url(cls, values):
+        video_id = values.get("video_id")
+        youtube_url = values.get("youtube_url")
+        if not video_id and not youtube_url:
+            raise ValueError("Either video_id or youtube_url is required")
+        return values
+
+    @validator("video_id")
+    def validate_video_id(cls, value):
+        if value and not YOUTUBE_ID_RE.match(value):
+            raise ValueError("video_id must be a valid YouTube id")
+        return value
+
+    @validator("youtube_url")
+    def validate_youtube_url(cls, value):
+        if value is None:
+            return value
+        parsed = urlparse(value)
+        if not parsed.scheme or not parsed.netloc:
+            raise ValueError("youtube_url must be a valid URL")
+        return value
+
 class DiscoverResponse(BaseModel):
     status: str
     job_id: str
@@ -107,30 +151,58 @@ def get_transcript(video_id: str) -> List[dict]:
 # -------------------------
 # Chunking (3hr-safe)
 # -------------------------
-def chunk_transcript(segments: List[dict], chunk_seconds: int = 120) -> List[dict]:
+SENTENCE_END_RE = re.compile(r"[.!?…]+$")  # simple heuristic for thought completion
+
+
+def chunk_transcript(
+    segments: List[dict],
+    chunk_seconds: int = 120,
+    min_chunk_seconds: int = 30,
+) -> List[dict]:
     """
-    Chunk by time window. 120s chunks is good for long videos.
+    Chunk by time window while trying to end on a sentence boundary.
     We do NOT return chunks to Zapier — internal use only.
     """
+    if not segments:
+        return []
     chunks = []
     current = []
     current_start = segments[0]["start"]
     total = 0.0
+    last_sentence_break_index = None
 
     for s in segments:
-        if total + s["duration"] > chunk_seconds and current:
-            chunks.append({
-                "start": float(current_start),
-                "end": float(current_start + total),
-                "segments": current
-            })
-            current = []
+        if not current:
             current_start = s["start"]
-            total = 0.0
-
         current.append(s)
         total += float(s["duration"])
 
+        if s["text"] and SENTENCE_END_RE.search(s["text"]):
+            last_sentence_break_index = len(current) - 1
+
+        if total > chunk_seconds and current:
+            if last_sentence_break_index is not None and total >= min_chunk_seconds:
+                split_at = last_sentence_break_index + 1
+                chunk_segments = current[:split_at]
+                chunks.append({
+                    "start": float(current_start),
+                    "end": float(chunk_segments[-1]["start"] + chunk_segments[-1]["duration"]),
+                    "segments": chunk_segments,
+                })
+                current = current[split_at:]
+                current_start = current[0]["start"] if current else current_start
+                total = sum(seg["duration"] for seg in current)
+                last_sentence_break_index = None
+            else:
+                chunks.append({
+                    "start": float(current_start),
+                    "end": float(current_start + total),
+                    "segments": current
+                })
+                current = []
+                total = 0.0
+                last_sentence_break_index = None
+
     if current:
         chunks.append({
             "start": float(current_start),
@@ -194,20 +266,32 @@ def discover(req: DiscoverRequest):
         # Return a clear error early rather than accepting jobs that cannot run.
         raise HTTPException(status_code=500, detail="RAPIDAPI_KEY not configured")
 
+    resolved_video_id = req.video_id
+    if req.youtube_url:
+        extracted = extract_video_id(req.youtube_url)
+        if not extracted:
+            raise HTTPException(status_code=400, detail="Unable to parse youtube_url")
+        if resolved_video_id and resolved_video_id != extracted:
+            raise HTTPException(status_code=400, detail="video_id does not match youtube_url")
+        resolved_video_id = extracted
+
+    if not resolved_video_id:
+        raise HTTPException(status_code=400, detail="video_id could not be resolved")
+
     job_id = str(uuid.uuid4())
     JOBS[job_id] = {
         "job_id": job_id,
-        "video_id": req.video_id,
+        "video_id": resolved_video_id,
         "status": "queued",
         "step": "queued",
         "created_at": time.time(),
     }
 
     # Offload to background thread
-    executor.submit(run_discovery, job_id, req.video_id)
+    executor.submit(run_discovery, job_id, resolved_video_id)
 
     # IMPORTANT: return immediately, no transcript, no chunks
-    return {"status": "accepted", "job_id": job_id, "video_id": req.video_id}
+    return {"status": "accepted", "job_id": job_id, "video_id": resolved_video_id}
 
 # -------------------------
 # JOB STATUS (debug endpoint)

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
    """
    youtube-transcript3 ONLY, but uses BOTH endpoints for reliability:
      1) /api/transcript?videoId=...
      2) /api/transcript-with-url?url=...
    """
    if not RAPIDAPI_KEY:
        raise RuntimeError("RAPIDAPI_KEY not configured")

    headers = {
        "x-rapidapi-host": YTT3_HOST,
        "x-rapidapi-key": RAPIDAPI_KEY,
    }

    # Always keep timestamps
    lang = os.getenv("TRANSCRIPT_LANG", "auto")
    flat_text = "false"  # force timestamps always

    video_url = f"https://www.youtube.com/watch?v={video_id}"

    def _call(endpoint: str, params: dict) -> dict:
        resp = requests.get(endpoint, headers=headers, params=params, timeout=(10, 120))
        try:
            data = resp.json()
        except Exception:
            raise RuntimeError(f"Transcript API non-JSON ({resp.status_code}): {resp.text}")

        # Log minimal debug context (safe)
        logger.info(f"Transcript3 call={endpoint.split('/')[-1]} status={resp.status_code} success={data.get('success')}")

        if resp.status_code != 200:
            raise RuntimeError(f"Transcript API HTTP error ({resp.status_code}): {data}")

        if not data.get("success"):
            raise RuntimeError(f"Transcript API reported failure: {data}")

        return data

    # Attempt 1: by videoId
    try:
        data = _call(
            YTT3_URL,
            {"videoId": video_id, "lang": lang, "flat_text": flat_text},
        )
    except Exception as e1:
        logger.warning(f"Transcript3 videoId attempt failed: {e1}")

        # Attempt 2: by URL (same provider)
        url_endpoint = "https://youtube-transcript3.p.rapidapi.com/api/transcript-with-url"
        data = _call(
            url_endpoint,
            {"url": video_url, "lang": lang, "flat_text": flat_text},
        )

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
