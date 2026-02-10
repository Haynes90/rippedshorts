 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/main.py b/main.py
index 2527ee85483fdaf75ec2ef579f0ba51f542df0ed..7cd33a79c88744898e82fdd2c6911a50eefb6a15 100644
--- a/main.py
+++ b/main.py
@@ -1,46 +1,44 @@
 import json
 import os
 import re
 import time
 import uuid
 import logging
 import math
 from typing import List, Dict, Any, Optional
 from concurrent.futures import ThreadPoolExecutor
 from pathlib import Path
 import subprocess
 from urllib.parse import urlparse, parse_qs
 
 import requests
 from fastapi import FastAPI, HTTPException
 from pydantic import BaseModel, Field
 from google.oauth2 import service_account
 from googleapiclient.discovery import build
 from googleapiclient.http import MediaFileUpload
-import cv2
-import mediapipe as mp
 
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
@@ -73,56 +71,61 @@ if not DRIVE_FOLDER_ID:
     logger.warning("Drive folder id not set (DRIVE_FOLDER_ID).")
 if not GOOGLE_CREDENTIALS and not GOOGLE_SERVICE_ACCOUNT_FILE and not (GOOGLE_CLIENT_EMAIL and GOOGLE_PRIVATE_KEY):
     logger.warning(
         "Google credentials not set (GOOGLE_CREDENTIALS, GOOGLE_SERVICE_ACCOUNT_FILE, "
         "or GOOGLE_CLIENT_EMAIL + GOOGLE_PRIVATE_KEY)."
     )
 if GOOGLE_API_KEY:
     logger.info("GOOGLE_API provided but not used for Docs/Drive write access.")
 if not OPENAI_API_KEY:
     logger.warning("OPENAI_API_KEY not set (clip discovery will fail until configured).")
 if not API_VIDEO_KEY:
     logger.warning("API_VIDEO_KEY not set (api.video clipping will be unavailable).")
 
 # -------------------------
 # MODELS
 # -------------------------
 YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,20}$")
 
 
 class DiscoverRequest(BaseModel):
     video_id: Optional[str] = Field(None, min_length=6)
     youtube_url: Optional[str] = None
     sheet_id: Optional[str] = None
     sheet_tab: Optional[str] = None
     prompt: Optional[str] = None
+    wait_for_result: bool = False
+    wait_timeout_s: int = Field(0, ge=0, le=600)
 
 
 class DiscoverResponse(BaseModel):
     status: str
     job_id: str
     video_id: str
+    step: Optional[str] = None
+    error: Optional[str] = None
+    elapsed_s: Optional[float] = None
 
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
 
 def resolve_video_id(video_id: Optional[str], youtube_url: Optional[str]) -> str:
     if not video_id and not youtube_url:
         raise ValueError("video_id or youtube_url is required")
     resolved = video_id
     if youtube_url:
         extracted = extract_video_id(youtube_url)
         if not extracted:
             raise ValueError("Unable to parse youtube_url")
@@ -358,60 +361,65 @@ def download_youtube_video(video_id: str, youtube_url: Optional[str], workdir: P
         timeout=(10, 60),
     )
     if resp.status_code != 200:
         raise RuntimeError(f"Youtube download API error ({resp.status_code}): {resp.text}")
     payload = resp.json()
     download_url = (
         payload.get("url")
         or payload.get("download")
         or payload.get("downloadUrl")
         or payload.get("videoUrl")
     )
     if not download_url and isinstance(payload.get("downloads"), list):
         for item in payload["downloads"]:
             if isinstance(item, dict):
                 download_url = item.get("url") or item.get("download") or item.get("downloadUrl")
                 if download_url:
                     break
     if not download_url:
         raise RuntimeError(f"Unable to extract download URL from API response: {payload}")
     output_path = workdir / f"{video_id}.mp4"
     download_video_asset(download_url, output_path)
     return output_path
 
 
 def _probe_video_dimensions(video_path: Path) -> tuple[int, int]:
+    import cv2
+
     cap = cv2.VideoCapture(str(video_path))
     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
     cap.release()
     if width <= 0 or height <= 0:
         raise RuntimeError("Unable to read video dimensions")
     return width, height
 
 
 def _estimate_speaker_center_x(video_path: Path, start: float, duration: float) -> float:
+    import cv2
+    import mediapipe as mp
+
     mp_face = mp.solutions.face_detection
     detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
     cap = cv2.VideoCapture(str(video_path))
     cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
     centers = []
     total_frames = int(min(duration, 30) * 2)
     step = max(1, int(cap.get(cv2.CAP_PROP_FPS) // 2 or 1))
     frame_idx = 0
     while len(centers) < total_frames:
         ok, frame = cap.read()
         if not ok:
             break
         if frame_idx % step == 0:
             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
             results = detector.process(rgb)
             if results.detections:
                 bbox = results.detections[0].location_data.relative_bounding_box
                 center_x = bbox.xmin + bbox.width / 2
                 centers.append(center_x)
         frame_idx += 1
     cap.release()
     detector.close()
     if centers:
         return sum(centers) / len(centers)
     return 0.5
@@ -920,37 +928,53 @@ def run_discovery(job_id: str, video_id: str):
 # -------------------------
 @app.post("/discover", response_model=DiscoverResponse, status_code=202)
 def discover(req: DiscoverRequest):
     if not RAPIDAPI_KEY:
         raise HTTPException(status_code=500, detail="RAPIDAPI_KEY not configured")
 
     try:
         resolved_video_id = resolve_video_id(req.video_id, req.youtube_url)
     except ValueError as exc:
         raise HTTPException(status_code=400, detail=str(exc)) from exc
 
     job_id = str(uuid.uuid4())
     JOBS[job_id] = {
         "job_id": job_id,
         "video_id": resolved_video_id,
         "youtube_url": req.youtube_url,
         "sheet_id": req.sheet_id,
         "sheet_tab": req.sheet_tab,
         "prompt": req.prompt,
         "status": "queued",
         "step": "queued",
         "created_at": time.time(),
     }
 
     executor.submit(run_discovery, job_id, resolved_video_id)
-
-    return {"status": "accepted", "job_id": job_id, "video_id": resolved_video_id}
+    if req.wait_for_result:
+        timeout = req.wait_timeout_s if req.wait_timeout_s > 0 else 120
+        deadline = time.time() + timeout
+        while time.time() < deadline:
+            job = JOBS.get(job_id, {})
+            status = job.get("status")
+            if status in {"done", "error"}:
+                return {
+                    "status": status,
+                    "job_id": job_id,
+                    "video_id": resolved_video_id,
+                    "step": job.get("step"),
+                    "error": job.get("error"),
+                    "elapsed_s": job.get("elapsed_s"),
+                }
+            time.sleep(1)
+
+    return {"status": "accepted", "job_id": job_id, "video_id": resolved_video_id, "step": "queued"}
 
 # -------------------------
 # JOB STATUS (debug endpoint)
 # -------------------------
 @app.get("/jobs/{job_id}")
 def job_status(job_id: str):
     job = JOBS.get(job_id)
     if not job:
         raise HTTPException(status_code=404, detail="job not found")
     return job
 
EOF
)
