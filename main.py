import os
import json
import uuid
import datetime
import subprocess
import io

import yt_dlp
import requests

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

# --------------------------------------------------
# APP
# --------------------------------------------------
app = FastAPI()

# --------------------------------------------------
# STATE
# --------------------------------------------------
JOBS = {}
LATEST_INGEST = {
    "file_id": None,
    "title": None,
    "folder_id": None
}

# --------------------------------------------------
# UTIL
# --------------------------------------------------
def log(job_id: str, msg: str):
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    JOBS.setdefault(job_id, {}).setdefault("logs", []).append(f"{ts} | {msg}")

def raw_log(label: str, payload):
    try:
        print(f"[RAW] {label}: {json.dumps(payload)[:8000]}")
    except Exception:
        print(f"[RAW] {label}: <unserializable>")

def sanitize_filename(name: str):
    for c in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        name = name.replace(c, "_")
    return name.strip() or "clip"

# --------------------------------------------------
# COOKIES
# --------------------------------------------------
COOKIES_PATH = "/app/cookies.txt"
if "YOUTUBE_COOKIES" in os.environ:
    with open(COOKIES_PATH, "w") as f:
        f.write(os.environ["YOUTUBE_COOKIES"])

# --------------------------------------------------
# GOOGLE DRIVE
# --------------------------------------------------
def get_drive_client():
    creds = service_account.Credentials.from_service_account_info(
        json.loads(os.environ["GOOGLE_CREDENTIALS"]),
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)

# --------------------------------------------------
# YOUTUBE INGEST (UNCHANGED, VERIFIED)
# --------------------------------------------------
def download_video(url: str, title: str, job_id=None):
    out = f"/tmp/{title}.mp4"

    opts = {
        "format": "bv*+ba/best",
        "merge_output_format": "mp4",
        "outtmpl": out,
        "cookies": COOKIES_PATH,
        "quiet": True,
        "retries": 5,
        "fragment_retries": 5,
        "extractor_args": {
            "youtube": {"player_client": ["android"]}
        },
    }

    log(job_id, "Starting YouTube ingest")

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    if not os.path.exists(out) or os.path.getsize(out) == 0:
        raise Exception("YouTube ingest produced empty file")

    log(job_id, f"Ingested video size: {os.path.getsize(out)} bytes")
    return out

# --------------------------------------------------
# DRIVE DOWNLOAD (FOR CLIPPING)
# --------------------------------------------------
def download_from_drive(file_id: str, job_id=None):
    drive = get_drive_client()
    req = drive.files().get_media(fileId=file_id, supportsAllDrives=True)

    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, req)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    path = f"/tmp/{file_id}.mp4"
    with open(path, "wb") as f:
        f.write(fh.getvalue())

    size = os.path.getsize(path)
    log(job_id, f"Downloaded source from Drive ({size} bytes)")

    if size == 0:
        raise Exception("Drive download produced empty file")

    return path

# --------------------------------------------------
# DRIVE UPLOAD
# --------------------------------------------------
def upload_to_drive(path: str, title: str, folder_id: str, job_id=None):
    drive = get_drive_client()

    media = MediaFileUpload(path, mimetype="video/mp4", resumable=True)
    meta = {"name": f"{title}.mp4", "parents": [folder_id]}

    req = drive.files().create(
        body=meta,
        media_body=media,
        fields="id,webViewLink",
        supportsAllDrives=True
    )

    resp = None
    while resp is None:
        _, resp = req.next_chunk()

    log(job_id, f"Uploaded file to Drive: {resp['id']}")
    return resp

# --------------------------------------------------
# FFMPEG CLIP
# --------------------------------------------------
def make_clip(src, out, start, dur, job_id=None):
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", src,
        "-t", str(dur),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "veryfast",
        out
    ]

    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise Exception(r.stderr[-400:])

    size = os.path.getsize(out)
    log(job_id, f"Created clip ({size} bytes)")

    if size == 0:
        raise Exception("ffmpeg produced empty clip")

# --------------------------------------------------
# BACKGROUND: INGEST
# --------------------------------------------------
def process_ingest(job_id, url, title, callback):
    try:
        path = download_video(url, title, job_id)
        res = upload_to_drive(path, title, os.environ["DRIVE_FOLDER_ID"], job_id)

        LATEST_INGEST.update({
            "file_id": res["id"],
            "title": title,
            "folder_id": os.environ["DRIVE_FOLDER_ID"]
        })

        JOBS[job_id]["status"] = "success"
        JOBS[job_id]["file_id"] = res["id"]

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)

    if callback:
        requests.post(callback, json=JOBS[job_id], timeout=15)

# --------------------------------------------------
# BACKGROUND: CLIPS
# --------------------------------------------------
def process_clips(job_id, payload):
    try:
        src = download_from_drive(payload["drive_file_id"], job_id)
        base = sanitize_filename(payload["video_title"])
        JOBS[job_id]["clips"] = []

        for clip in payload["clips"]:
            idx = int(clip["index"])
            out = f"/tmp/{base}_{idx}.mp4"

            make_clip(src, out, clip["start"], clip["duration"], job_id)
            up = upload_to_drive(out, f"{base}_{idx}", payload["folder_id"], job_id)

            JOBS[job_id]["clips"].append({
                "index": idx,
                "file_id": up["id"]
            })

        JOBS[job_id]["status"] = "success"

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)

    if payload.get("callback_url"):
        requests.post(payload["callback_url"], json=JOBS[job_id], timeout=20)

# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.post("/ingest")
async def ingest(req: Request, bg: BackgroundTasks):
    data = await req.json()
    raw_log("INGEST", data)

    url = data.get("url")
    if not url:
        return JSONResponse({"error": "missing url"}, 400)

    title = url.split("v=")[-1].split("&")[0]
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"job_id": job_id, "status": "queued"}

    bg.add_task(process_ingest, job_id, url, title, data.get("callback_url"))
    return {"status": "queued", "job_id": job_id}

@app.post("/analyze-and-clip")
async def analyze_and_clip(req: Request, bg: BackgroundTasks):
    data = await req.json()
    raw_log("ANALYZE", data)

    raw_segments = data.get("segments_json")
    segments = []

    try:
        if isinstance(raw_segments, str):
            raw_segments = json.loads(raw_segments)
        if isinstance(raw_segments, dict):
            raw_segments = raw_segments.get("segments", raw_segments.get("segments_json"))
        if isinstance(raw_segments, list):
            for s in raw_segments:
                if "start" in s and "duration" in s:
                    segments.append(s)
    except Exception:
        pass

    if not segments:
        return {"status": "ok", "note": "no segments", "clips_found": 0}

    if not LATEST_INGEST["file_id"]:
        return JSONResponse({"error": "no ingest available"}, 400)

    clips = []
    for i, seg in enumerate(segments, start=1):
        clips.append({
            "index": i,
            "start": float(seg["start"]),
            "duration": float(seg["duration"])
        })

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"job_id": job_id, "status": "queued"}

    bg.add_task(process_clips, job_id, {
        "drive_file_id": LATEST_INGEST["file_id"],
        "folder_id": LATEST_INGEST["folder_id"],
        "video_title": data.get("video_title", LATEST_INGEST["title"]),
        "clips": clips,
        "callback_url": data.get("callback_url")
    })

    return {"status": "queued", "job_id": job_id, "clips_requested": len(clips)}

@app.get("/status/{job_id}")
async def status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404)
    return JOBS[job_id]

@app.get("/health")
async def health():
    return {"status": "ok"}
