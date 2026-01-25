 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/main.py b/main.py
index f364ddc7ebfb844a73a909111b9732f66f957676..a76c6eed51cacbaf7d25d67c60614cc35718875e 100644
--- a/main.py
+++ b/main.py
@@ -1,161 +1,135 @@
 import os
 import uuid
 import time
 import logging
+import json
+import http.client
 from typing import List, Optional, Dict, Any
 from concurrent.futures import ThreadPoolExecutor
-
-import requests
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
-TRANSCRIPT_LANG = os.getenv("TRANSCRIPT_LANG", "auto")  # auto recommended
-TRANSCRIPT_FLAT_TEXT = os.getenv("TRANSCRIPT_FLAT_TEXT", "false").lower() in ("1", "true", "yes")
-
 YTT3_HOST = "youtube-transcript3.p.rapidapi.com"
-YTT3_URL = "https://youtube-transcript3.p.rapidapi.com/api/transcript"
 
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
-    youtube-transcript3 ONLY, tries:
-      1) /api/transcript?videoId=...
-      2) /api/transcript-with-url?url=...
-    Omits 'lang' when TRANSCRIPT_LANG=auto (provider behaves better that way).
+    Fetch transcript from youtube-transcript3 using the provided video ID.
     """
     if not RAPIDAPI_KEY:
         raise RuntimeError("RAPIDAPI_KEY not configured")
 
     headers = {
         "x-rapidapi-host": YTT3_HOST,
         "x-rapidapi-key": RAPIDAPI_KEY,
     }
 
-    # Force timestamps
-    flat_text = "false"
-
-    lang_env = os.getenv("TRANSCRIPT_LANG", "auto").strip().lower()
-    include_lang = (lang_env not in ("", "auto", "none"))
-
-    video_url = f"https://www.youtube.com/watch?v={video_id}"
-
-    def _call(endpoint: str, params: dict) -> dict:
-        resp = requests.get(endpoint, headers=headers, params=params, timeout=(10, 120))
-        try:
-            data = resp.json()
-        except Exception:
-            raise RuntimeError(f"Transcript API non-JSON ({resp.status_code}): {resp.text}")
-
-        logger.info(
-            f"Transcript3 call={endpoint.split('/')[-1]} "
-            f"status={resp.status_code} success={data.get('success')}"
-        )
-
-        if resp.status_code != 200:
-            raise RuntimeError(f"Transcript API HTTP error ({resp.status_code}): {data}")
-
-        if not data.get("success"):
-            # Preserve exact provider error text for retry decision
-            raise RuntimeError(f"TRANSCRIPT3_FAIL::{data}")
+    conn = http.client.HTTPSConnection(YTT3_HOST, timeout=30)
+    try:
+        path = f"/api/transcript?videoId={video_id}"
+        conn.request("GET", path, headers=headers)
+        res = conn.getresponse()
+        raw = res.read()
+    finally:
+        conn.close()
 
-        return data
+    try:
+        data = json.loads(raw.decode("utf-8"))
+    except json.JSONDecodeError as exc:
+        raise RuntimeError(f"Transcript API non-JSON ({res.status}): {raw.decode('utf-8', errors='replace')}") from exc
 
-    # Attempt 1: by videoId
-    params1 = {"videoId": video_id, "flat_text": flat_text}
-    if include_lang:
-        params1["lang"] = lang_env
+    logger.info(
+        "Transcript3 call=transcript status=%s success=%s",
+        res.status,
+        data.get("success"),
+    )
 
-    try:
-        data = _call(YTT3_URL, params1)
-    except Exception as e1:
-        logger.warning(f"Transcript3 videoId attempt failed: {e1}")
+    if res.status != 200:
+        raise RuntimeError(f"Transcript API HTTP error ({res.status}): {data}")
 
-        # Attempt 2: by URL (same provider)
-        url_endpoint = "https://youtube-transcript3.p.rapidapi.com/api/transcript-with-url"
-        params2 = {"url": video_url, "flat_text": flat_text}
-        if include_lang:
-            params2["lang"] = lang_env
-        data = _call(url_endpoint, params2)
+    if not data.get("success"):
+        # Preserve exact provider error text for retry decision
+        raise RuntimeError(f"TRANSCRIPT3_FAIL::{data}")
 
     transcript = data.get("transcript")
 
     if isinstance(transcript, str):
-        raise RuntimeError("Provider returned flat text; timestamps unavailable (flat_text must be false).")
+        raise RuntimeError("Provider returned flat text; timestamps unavailable.")
 
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
@@ -173,84 +147,117 @@ def chunk_transcript(segments: List[dict], chunk_seconds: int = 120) -> List[dic
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
-    try:
-        JOBS[job_id]["step"] = "transcript_fetch"
+def run_discovery(job_id: str, video_id: str):
+    started = time.time()
+    JOBS[job_id]["status"] = "running"
+    JOBS[job_id]["step"] = "transcript_fetch"
+
+    logger.info(f"[{job_id}] discovery start video_id={video_id}")
 
+    try:
         # Retry strategy for transient provider failures
         backoff_seconds = [0, 15, 45, 120]  # total ~3 minutes max
         last_err = None
+        transcript = None
 
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
 
-        # continue with chunking...
+        JOBS[job_id]["step"] = "chunking"
+        chunks = chunk_transcript(transcript, chunk_seconds=120)
+        logger.info(f"[{job_id}] chunks={len(chunks)}")
+
+        # Next steps to add AFTER transcript stability:
+        # - AI clip discovery per chunk
+        # - global merge/rank, cap <= 30
+        # - write proposed clips to Google Sheets
+
+        JOBS[job_id]["status"] = "done"
+        JOBS[job_id]["step"] = "completed"
+        JOBS[job_id]["result"] = {
+            "video_id": video_id,
+            "segments": len(transcript),
+            "chunks": len(chunks),
+        }
+        JOBS[job_id]["elapsed_s"] = round(time.time() - started, 2)
+
+        logger.info(f"[{job_id}] discovery done elapsed_s={JOBS[job_id]['elapsed_s']}")
+
+    except Exception as e:
+        msg = str(e)
+        JOBS[job_id]["status"] = "error"
+        JOBS[job_id]["step"] = "failed"
+        JOBS[job_id]["error"] = msg
+        JOBS[job_id]["elapsed_s"] = round(time.time() - started, 2)
+
+        logger.exception(f"[{job_id}] discovery failed: {e}")
 
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
 
EOF
)
