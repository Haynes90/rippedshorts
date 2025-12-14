import os
import json
import uuid
import datetime
import subprocess
import io
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import yt_dlp

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

app = FastAPI()

# -----------------------------
# In-memory job store (debug/status)
# -----------------------------
JOBS: Dict[str, Dict[str, Any]] = {}

# Latest ingest (convenience fallback)
LATEST_INGEST: Dict[str, Optional[str]] = {
    "file_id": None,
    "title": None,
    "folder_id": None,
}

# -----------------------------
# Logging helpers
# -----------------------------
def _ts() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def log(job_id: str, message: str) -> None:
    if job_id not in JOBS:
        return
    JOBS[job_id].setdefault("logs", []).append(f"{_ts()} | {message}")

def print_raw(label: str, payload: Any) -> None:
    try:
        if isinstance(payload, (bytes, bytearray)):
            txt = payload.decode("utf-8", errors="ignore")
        else:
            txt = json.dumps(payload, ensure_ascii=False) if not isinstance(payload, str) else payload
        print(f"[RAW] {label}: {txt[:8000]}")
    except Exception:
        print(f"[RAW] {label}: <unprintable>")

# -----------------------------
# Environment + cookies
# -----------------------------
COOKIES_PATH = "/app/cookies.txt"
if os.environ.get("YOUTUBE_COOKIES"):
    try:
        with open(COOKIES_PATH, "w", encoding="utf-8") as f:
            f.write(os.environ["YOUTUBE_COOKIES"])
        print("[BOOT] Wrote cookies.txt from YOUTUBE_COOKIES")
    except Exception as e:
        print(f"[BOOT] Failed writing cookies.txt: {e}")

# -----------------------------
# Drive client
# -----------------------------
def get_drive_client():
    if "GOOGLE_CREDENTIALS" not in os.environ:
        raise Exception("Missing GOOGLE_CREDENTIALS env var")

    creds = service_account.Credentials.from_service_account_info(
        json.loads(os.environ["GOOGLE_CREDENTIALS"]),
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    return build("drive", "v3", credentials=creds)

def drive_upload(file_path: str, filename: str, folder_id: str) -> Dict[str, Any]:
    drive = get_drive_client()
    meta = {"name": filename, "parents": [folder_id]}
    media = MediaFileUpload(file_path, mimetype="video/mp4", resumable=True)

    req = drive.files().create(
        body=meta,
        media_body=media,
        fields="id,webViewLink",
        supportsAllDrives=True,
    )

    resp = None
    while resp is None:
        _, resp = req.next_chunk(num_retries=5)
    return resp

def drive_download(file_id: str) -> str:
    drive = get_drive_client()
    req = drive.files().get_media(fileId=file_id, supportsAllDrives=True)

    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, req)

    done = False
    while not done:
        _, done = downloader.next_chunk(num_retries=5)

    local_path = f"/tmp/{file_id}.mp4"
    with open(local_path, "wb") as f:
        f.write(fh.getvalue())
    return local_path

# -----------------------------
# Filename safety
# -----------------------------
def sanitize_filename(name: str) -> str:
    if not name:
        return "video"
    bad = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for ch in bad:
        name = name.replace(ch, "_")
    name = " ".join(name.split()).strip()
    return name or "video"

# -----------------------------
# yt-dlp download (full video)
# -----------------------------
def download_youtube(url: str, title: str, job_id: Optional[str] = None) -> str:
    safe = sanitize_filename(title)
    out_path = f"/tmp/{safe}.mp4"

    ydl_opts: Dict[str, Any] = {
        "outtmpl": out_path,
        "format": "mp4",
        "quiet": True,
        "noprogress": True,
        "retries": 5,
        "fragment_retries": 5,
        "concurrent_fragment_downloads": 1,
        # Helps with newer YT extraction changes
        "extractor_args": {"youtube": {"player_client": ["default"]}},
    }

    if os.path.exists(COOKIES_PATH):
        ydl_opts["cookies"] = COOKIES_PATH

    # Backoff to reduce 429 flakiness
    ydl_opts["sleep_interval"] = 1
    ydl_opts["max_sleep_interval"] = 5

    if job_id:
        log(job_id, f"yt-dlp downloading: {url}")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise Exception("Downloaded file is empty")

    if job_id:
        log(job_id, f"yt-dlp download OK -> {out_path}")

    return out_path

# -----------------------------
# ffmpeg clipper
# -----------------------------
def ffmpeg_clip(source_path: str, out_path: str, start: float, duration: float, job_id: Optional[str] = None, label: str = "") -> None:
    if job_id:
        log(job_id, f"ffmpeg clip {label} start={start} dur={duration}")

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", source_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "veryfast",
        out_path,
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        tail = (r.stderr or "")[-600:]
        raise Exception(f"ffmpeg failed: {tail}")

# -----------------------------
# Zapier-tolerant parsing
# -----------------------------
def _try_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None

def _extract_json_substring(s: str) -> Optional[str]:
    # Best-effort: grab first {...} or [...] block
    if not s:
        return None
    first_obj = s.find("{")
    first_arr = s.find("[")
    if first_obj == -1 and first_arr == -1:
        return None
    start = min([x for x in [first_obj, first_arr] if x != -1])
    end_obj = s.rfind("}")
    end_arr = s.rfind("]")
    end = max(end_obj, end_arr)
    if end <= start:
        return None
    return s[start:end + 1]

def normalize_segments(raw: Any) -> List[Dict[str, float]]:
    """
    Accepts:
      - already a list of {start,duration} OR {offset,duration}
      - dict wrapper containing "segments" or "segments_json" list
      - stringified JSON with {"segments":[...]} or wrapper dict
    Returns:
      list of {"start": float, "duration": float}
    """
    if raw is None:
        return []

    # If Zapier sends bytes-like (rare), decode
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="ignore")

    # If string, try parse
    if isinstance(raw, str):
        parsed = _try_json_loads(raw)
        if parsed is None:
            sub = _extract_json_substring(raw)
            if sub:
                parsed = _try_json_loads(sub)
        raw = parsed if parsed is not None else raw

    # If dict, unwrap common keys
    if isinstance(raw, dict):
        # Sometimes Zapier wraps outputs: {id, runtime_meta, segments_json:[...]}
        if "segments" in raw:
            raw = raw["segments"]
        elif "segments_json" in raw:
            raw = raw["segments_json"]
        elif "output_json" in raw and isinstance(raw["output_json"], dict) and "segments" in raw["output_json"]:
            raw = raw["output_json"]["segments"]

    # If still a string after all that, can't use
    if isinstance(raw, str):
        return []

    if not isinstance(raw, list):
        return []

    out: List[Dict[str, float]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        start = item.get("start")
        if start is None:
            start = item.get("offset")
        duration = item.get("duration")

        if start is None or duration is None:
            continue

        try:
            out.append({"start": float(start), "duration": float(duration)})
        except Exception:
            continue

    # Remove zeros / nonsense
    out = [s for s in out if s["duration"] > 0.05 and s["start"] >= 0.0]
    return out

def normalize_clips(raw: Any, video_title: str) -> List[Dict[str, Any]]:
    """
    Accepts user-style clips list or builds from segments.
    Clip item should become:
      {"index": int, "start": float, "duration": float, "name": str}
    """
    if raw is None:
        return []

    # If stringified JSON, parse
    if isinstance(raw, str):
        parsed = _try_json_loads(raw)
        if parsed is None:
            sub = _extract_json_substring(raw)
            if sub:
                parsed = _try_json_loads(sub)
        raw = parsed if parsed is not None else raw

    # dict wrapper
    if isinstance(raw, dict) and "clips" in raw:
        raw = raw["clips"]

    if not isinstance(raw, list):
        return []

    base = sanitize_filename(video_title)
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            continue
        start = item.get("start")
        if start is None:
            start = item.get("start_sec")
        if start is None:
            start = item.get("offset")
        duration = item.get("duration")
        if duration is None:
            duration = item.get("duration_sec")

        if start is None or duration is None:
            continue

        try:
            start_f = float(start)
            dur_f = float(duration)
        except Exception:
            continue

        idx = item.get("index")
        try:
            idx_i = int(idx) if idx is not None else i
        except Exception:
            idx_i = i

        name = item.get("name")
        if not name:
            name = f"{base}_{idx_i}"

        out.append({
            "index": idx_i,
            "start": start_f,
            "duration": dur_f,
            "name": sanitize_filename(name)
        })

    return out

# -----------------------------
# Background workers
# -----------------------------
def bg_ingest(job_id: str, url: str, title: str, folder_id: str, callback_url: Optional[str]) -> None:
    try:
        JOBS[job_id]["status"] = "downloading"
        log(job_id, "Stage: downloading from YouTube")

        video_path = download_youtube(url, title, job_id)

        JOBS[job_id]["status"] = "uploading"
        log(job_id, "Stage: uploading to Drive")

        uploaded = drive_upload(video_path, f"{sanitize_filename(title)}.mp4", folder_id)

        # update latest ingest
        LATEST_INGEST["file_id"] = uploaded.get("id")
        LATEST_INGEST["title"] = title
        LATEST_INGEST["folder_id"] = folder_id

        JOBS[job_id]["status"] = "success"
        JOBS[job_id]["file_id"] = uploaded.get("id")
        JOBS[job_id]["drive_link"] = uploaded.get("webViewLink")
        log(job_id, f"Ingest success file_id={uploaded.get('id')}")

        if callback_url:
            try:
                requests.post(callback_url, json={
                    "kind": "ingest",
                    "job_id": job_id,
                    "status": "success",
                    "file_id": uploaded.get("id"),
                    "video_title": title,
                }, timeout=10)
            except Exception as e:
                log(job_id, f"Callback failed: {e}")

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        log(job_id, f"Ingest failed: {e}")

        if callback_url:
            try:
                requests.post(callback_url, json={
                    "kind": "ingest",
                    "job_id": job_id,
                    "status": "failed",
                    "error": str(e),
                    "video_title": title,
                }, timeout=10)
            except Exception:
                pass

def bg_clip(job_id: str, drive_file_id: str, folder_id: str, video_title: str, clips: List[Dict[str, Any]], callback_url: Optional[str]) -> None:
    try:
        JOBS[job_id]["status"] = "downloading"
        log(job_id, f"Downloading source from Drive file_id={drive_file_id}")
        source_path = drive_download(drive_file_id)

        JOBS[job_id]["status"] = "clipping"
        JOBS[job_id]["clips"] = []
        base = sanitize_filename(video_title)

        for clip in clips:
            idx = int(clip["index"])
            start = float(clip["start"])
            duration = float(clip["duration"])
            name = sanitize_filename(clip.get("name") or f"{base}_{idx}")

            out_path = f"/tmp/{name}.mp4"
            ffmpeg_clip(source_path, out_path, start, duration, job_id, label=name)

            JOBS[job_id].setdefault("clips", [])
            up = drive_upload(out_path, f"{name}.mp4", folder_id)

            JOBS[job_id]["clips"].append({
                "index": idx,
                "name": name,
                "file_id": up.get("id"),
                "webViewLink": up.get("webViewLink"),
            })
            log(job_id, f"Uploaded clip {idx} -> {up.get('id')}")

        JOBS[job_id]["status"] = "success"
        log(job_id, f"Clip job success. clips={len(clips)}")

        if callback_url:
            try:
                requests.post(callback_url, json={
                    "kind": "clips",
                    "job_id": job_id,
                    "status": "success",
                    "clip_count": len(clips),
                }, timeout=15)
            except Exception as e:
                log(job_id, f"Callback failed: {e}")

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        log(job_id, f"Clip job failed: {e}")

        if callback_url:
            try:
                requests.post(callback_url, json={
                    "kind": "clips",
                    "job_id": job_id,
                    "status": "failed",
                    "error": str(e),
                }, timeout=15)
            except Exception:
                pass

# -----------------------------
# API: ingest
# -----------------------------
@app.post("/ingest")
async def ingest(request: Request, background_tasks: BackgroundTasks):
    raw = await request.body()
    print_raw("INGEST_BODY", raw)

    data = await request.json()
    url = data.get("url") or data.get("youtube_url")
    if not url:
        return JSONResponse({"error": "missing url"}, status_code=400)

    folder_id = data.get("folder_id") or os.environ.get("DRIVE_FOLDER_ID")
    if not folder_id:
        return JSONResponse({"error": "missing folder_id and DRIVE_FOLDER_ID env var"}, status_code=400)

    video_title = data.get("video_title") or data.get("title")
    if not video_title:
        # fallback to video id style
        video_title = url.split("v=")[-1].split("&")[0] if "v=" in url else "video"

    callback_url = data.get("callback_url")

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "kind": "ingest",
        "job_id": job_id,
        "status": "queued",
        "video_title": video_title,
        "url": url,
        "folder_id": folder_id,
        "logs": [],
    }

    background_tasks.add_task(bg_ingest, job_id, url, video_title, folder_id, callback_url)
    return {"status": "queued", "job_id": job_id}

# -----------------------------
# API: clips (direct timestamps)
# -----------------------------
@app.post("/clips")
async def create_clips(request: Request, background_tasks: BackgroundTasks):
    raw = await request.body()
    print_raw("CLIPS_BODY", raw)

    data = await request.json()

    callback_url = data.get("callback_url")
    video_title = data.get("video_title") or LATEST_INGEST.get("title") or "video"

    # Determine source drive file
    drive_file_id = data.get("drive_file_id") or LATEST_INGEST.get("file_id")
    if not drive_file_id:
        return JSONResponse({"error": "missing drive_file_id (and no latest ingest available)"}, status_code=400)

    # Determine folder for outputs
    folder_id = data.get("folder_id") or LATEST_INGEST.get("folder_id") or os.environ.get("DRIVE_FOLDER_ID")
    if not folder_id:
        return JSONResponse({"error": "missing folder_id (and no DRIVE_FOLDER_ID env var)"}, status_code=400)

    clips = normalize_clips(data.get("clips"), video_title)
    if not clips:
        return JSONResponse({"error": "missing/empty clips list"}, status_code=400)

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "kind": "clips",
        "job_id": job_id,
        "status": "queued",
        "drive_file_id": drive_file_id,
        "folder_id": folder_id,
        "video_title": video_title,
        "logs": [],
    }

    background_tasks.add_task(bg_clip, job_id, drive_file_id, folder_id, video_title, clips, callback_url)
    return {"status": "queued", "job_id": job_id, "clip_count": len(clips)}

# -----------------------------
# API: analyze-and-clip (Zapier segments_json -> clips)
# -----------------------------
@app.post("/analyze-and-clip")
async def analyze_and_clip(request: Request, background_tasks: BackgroundTasks):
    raw = await request.body()
    print_raw("ANALYZE_BODY", raw)

    data = await request.json()

    callback_url = data.get("callback_url")
    video_title = data.get("video_title") or LATEST_INGEST.get("title") or "video"

    # Determine source drive file (preferred: provided, else latest ingest)
    drive_file_id = data.get("drive_file_id") or LATEST_INGEST.get("file_id")
    if not drive_file_id:
        return JSONResponse({"error": "missing drive_file_id (and no latest ingest available)"}, status_code=400)

    # Determine folder for outputs
    folder_id = data.get("folder_id") or LATEST_INGEST.get("folder_id") or os.environ.get("DRIVE_FOLDER_ID")
    if not folder_id:
        return JSONResponse({"error": "missing folder_id (and no DRIVE_FOLDER_ID env var)"}, status_code=400)

    # Zapier sends segments_json in many shapes
    segments = normalize_segments(data.get("segments_json") or data.get("segements_json"))
    if not segments:
        # do not hard-fail; return OK so Zap continues if desired
        return {"status": "ok", "note": "no segments usable", "clips_found": 0}

    # Convert segments -> clips naming longformtitle_#
    base = sanitize_filename(video_title)
    clips: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments, 1):
        clips.append({
            "index": i,
            "start": seg["start"],
            "duration": seg["duration"],
            "name": f"{base}_{i}",
        })

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "kind": "analyze-and-clip",
        "job_id": job_id,
        "status": "queued",
        "drive_file_id": drive_file_id,
        "folder_id": folder_id,
        "video_title": video_title,
        "clips_found": len(clips),
        "logs": [],
    }

    background_tasks.add_task(bg_clip, job_id, drive_file_id, folder_id, video_title, clips, callback_url)
    return {"status": "queued", "job_id": job_id, "clips_found": len(clips)}

# -----------------------------
# API: status/health
# -----------------------------
@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/health")
async def health():
    return {"status": "ok"}
