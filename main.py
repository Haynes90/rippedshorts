import os
import json
import uuid
import datetime
import subprocess
import io
import time

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
# YOUTUBE DOWNLOAD
# --------------------------------------------------
def download_video(url: str, title: str, job_id=None):
    out = f"/tmp/{title}.mp4"
    opts = {
        "outtmpl": out,
        "format": "mp4",
        "cookies": COOKIES_PATH,
        "quiet": True
    }
    if job_id:
        log(job_id, "Downloading YouTube video")
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    return out

# --------------------------------------------------
# DRIVE DOWNLOAD
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

    return path

# --------------------------------------------------
# DRIVE UPLOAD
# --------------------------------------------------
def upload_to_drive(path: str, title: str, folder_id: str, job_id=None):
    drive = get_drive_client()
    media = MediaFileUpload(path, mimetype="video/mp4", resumable=True)
    meta = {
        "name": f"{title}.mp4",
        "parents": [folder_id]
    }

    req = drive.files().create(
        body=meta,
        media_body=media,
        fields="id,webViewLink",
        supportsAllDrives=True
    )

    resp = None
    while resp is None:
        _, resp = req.next_chunk()

    return resp

# --------------------------------------------------
# FFMPEG
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

        JOBS[job_id].update({
            "status": "success",
            "file_id": res["id"]
        })

    except Exception as e:
        JOBS[job_id].update({
            "status": "failed",
            "error": str(e)
        })

    if callback:
        requests.post(callback, json=JOBS[job_id], timeout=15)

# --------------------------------------------------
# BACKGROUND: CLIPS
# --------------------------------------------------
def process_clips(job_id, payload):
    src = download_from_drive(payload["drive_file_id"], job_id)
    JOBS[job_id]["clips"] = []

    base = sanitize_filename(payload["video_title"])

    for clip in payload["clips"]:
        idx = int(clip["index"])
        out = f"/tmp/{base}_{idx}.mp4"

        make_clip(src, out, clip["start"], clip["duration"], job_id)
        up = upload_to_drive(out, f"{base}_{idx}", payload["folder_id"], job_id)

        JOBS[job_id]["clips"].append({
            "index": idx,
            "file_id": up["id"],
            "webViewLink": up["webViewLink"]
        })

    JOBS[job_id]["status"] = "success"

    if payload.get("callback_url"):
        requests.post(payload["callback_url"], json=JOBS[job_id], timeout=20)

# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.post("/ingest")
async def ingest(req: Request, bg: BackgroundTasks):
    data = await req.json()
    url = data.get("url")
    if not url:
        return JSONResponse({"error": "missing url"}, status_code=400)

    title = url.split("v=")[-1].split("&")[0]
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"job_id": job_id, "status": "queued"}

    bg.add_task(process_ingest, job_id, url, title, data.get("callback_url"))
    return {"status": "queued", "job_id": job_id}

@app.post("/analyze-and-clip")
async def analyze_and_clip(req: Request, bg: BackgroundTasks):
    data = await req.json()

    # --------------------------------------------------
    # RAW PAYLOAD LOGGING
    # --------------------------------------------------
    raw_job_id = str(uuid.uuid4())
    JOBS[raw_job_id] = {
        "job_id": raw_job_id,
        "status": "received",
        "raw_payload": data
    }

    # Also write to disk so you can inspect in Railway
    with open(f"/tmp/raw_payload_{raw_job_id}.json", "w") as f:
        json.dump(data, f, indent=2)

    # --------------------------------------------------
    # NESTED SEGMENTS NORMALIZATION
    # --------------------------------------------------
    raw = data.get("segments_json")
    segments = []

    if isinstance(raw, list):
        segments = raw

    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                segments = parsed
        except:
            segments = []

    elif isinstance(raw, dict):
        inner = raw.get("segments_json")
        if isinstance(inner, str):
            try:
                parsed = json.loads(inner)
                if isinstance(parsed, list):
                    segments = parsed
            except:
                segments = []
        elif "segments_array" in raw and isinstance(raw["segments_array"], list):
            segments = raw["segments_array"]

    # --------------------------------------------------
    # GRACEFUL FALLBACK
    # --------------------------------------------------
    if not segments:
        JOBS[raw_job_id]["status"] = "no_segments"
        return {
            "status": "ok",
            "note": "No usable segments; payload logged",
            "job_id": raw_job_id,
            "clips_found": 0
        }

    # --------------------------------------------------
    # OPENAI
    # --------------------------------------------------
    prompt = f"""
Choose up to {data['max_clip_count']} strong clips.
Each clip must be between {data['min_clip_len_sec']} and {data['max_clip_len_sec']} seconds.
Return JSON only.
"""

    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4o-mini",
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "Return ONLY JSON"},
                {"role": "user", "content": prompt},
                {"role": "user", "content": json.dumps(segments)}
            ]
        },
        timeout=45
    )

    result = json.loads(r.json()["choices"][0]["message"]["content"])
    clips = result.get("clips", [])

    formatted = []
    for i, c in enumerate(clips, start=1):
        formatted.append({
            "index": i,
            "start": c["start_sec"],
            "duration": c["duration_sec"],
            "name": f"clip_{i}"
        })

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"job_id": job_id, "status": "queued"}

    bg.add_task(process_clips, job_id, {
        "drive_file_id": data["drive_file_id"],
        "folder_id": data["folder_id"],
        "video_title": data.get("video_title", "video"),
        "clips": formatted,
        "callback_url": data.get("callback_url")
    })

    return {
        "status": "queued",
        "job_id": job_id,
        "clips_found": len(formatted),
        "raw_payload_job_id": raw_job_id
    }

@app.get("/status/{job_id}")
async def status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="not found")
    return JOBS[job_id]

@app.get("/health")
async def health():
    return {"status": "ok"}
