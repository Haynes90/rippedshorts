import os
import json
import uuid
import datetime
import subprocess
import io
import requests

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

app = FastAPI()

JOBS = {}

LATEST_INGEST = {
    "file_id": None,
    "title": None,
    "folder_id": None
}

def log(job_id, msg):
    ts = datetime.datetime.utcnow().isoformat()
    JOBS.setdefault(job_id, {}).setdefault("logs", []).append(f"{ts} | {msg}")

# ---------- GOOGLE DRIVE ----------
def get_drive():
    creds = service_account.Credentials.from_service_account_info(
        json.loads(os.environ["GOOGLE_CREDENTIALS"]),
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)

def download_from_drive(file_id):
    drive = get_drive()
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

def upload_to_drive(path, name, folder_id):
    drive = get_drive()
    media = MediaFileUpload(path, mimetype="video/mp4")
    meta = {"name": f"{name}.mp4", "parents": [folder_id]}
    return drive.files().create(
        body=meta,
        media_body=media,
        fields="id,webViewLink",
        supportsAllDrives=True
    ).execute()

# ---------- FFMPEG ----------
def clip_video(src, out, start, duration):
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", src,
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        out
    ]
    subprocess.run(cmd, check=True)

# ---------- INGEST ----------
@app.post("/ingest")
async def ingest(req: Request):
    data = await req.json()

    if not data.get("drive_file_id") or not data.get("folder_id"):
        return JSONResponse({"error": "missing drive_file_id or folder_id"}, 400)

    LATEST_INGEST["file_id"] = data["drive_file_id"]
    LATEST_INGEST["folder_id"] = data["folder_id"]
    LATEST_INGEST["title"] = data.get("video_title", "video")

    return {"status": "ok"}

# ---------- ANALYZE AND CLIP ----------
@app.post("/analyze-and-clip")
async def analyze_and_clip(req: Request, bg: BackgroundTasks):
    raw = await req.body()
    print("RAW PAYLOAD:", raw.decode(errors="ignore"))

    data = await req.json()

    # 🔧 ZAPIER NORMALIZATION FIX
    segments = None

    # Case 1: Zapier sent misspelled, stringified JSON
    if "segements_json" in data:
        try:
            parsed = json.loads(data["segements_json"])
            segments = parsed.get("segments")
        except Exception:
            pass

    # Case 2: Correct direct array
    if segments is None:
        segments = data.get("segments")

    if not isinstance(segments, list) or not segments:
        return JSONResponse({"error": "no segments usable"}, 400)

    if not LATEST_INGEST["file_id"]:
        return JSONResponse({"error": "no ingest available"}, 400)

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued"}

    bg.add_task(
        run_clips,
        job_id,
        segments,
        data.get("video_title"),
        data.get("callback_url")
    )

    return {
        "status": "ok",
        "clips_found": len(segments),
        "job_id": job_id
    }

def run_clips(job_id, segments, title, callback):
    src = download_from_drive(LATEST_INGEST["file_id"])
    folder = LATEST_INGEST["folder_id"]
    base = title or LATEST_INGEST["title"]

    results = []

    for i, seg in enumerate(segments, 1):
        out = f"/tmp/{base}_{i}.mp4"
        clip_video(src, out, seg["start"], seg["duration"])
        up = upload_to_drive(out, f"{base}_{i}", folder)
        results.append(up["id"])

    JOBS[job_id]["status"] = "success"
    JOBS[job_id]["clips"] = results

    if callback:
        requests.post(callback, json={
            "status": "done",
            "clips_created": len(results)
        })

# ---------- HEALTH ----------
@app.get("/health")
async def health():
    return {"status": "ok"}
