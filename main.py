import os
import json
import subprocess
import io
import requests

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse

import yt_dlp

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

app = FastAPI()

LATEST_INGEST = {
    "file_id": None,
    "folder_id": None,
    "title": None
}

# ---------- GOOGLE DRIVE ----------
def get_drive():
    creds = service_account.Credentials.from_service_account_info(
        json.loads(os.environ["GOOGLE_CREDENTIALS"]),
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)

def upload_to_drive(path, name, folder):
    drive = get_drive()
    media = MediaFileUpload(path, mimetype="video/mp4")
    meta = {"name": f"{name}.mp4", "parents": [folder]}
    return drive.files().create(
        body=meta,
        media_body=media,
        fields="id",
        supportsAllDrives=True
    ).execute()

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

# ---------- YOUTUBE DOWNLOAD ----------
def download_youtube(url, title):
    out = f"/tmp/{title}.mp4"
    opts = {
        "outtmpl": out,
        "format": "mp4",
        "quiet": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    return out

# ---------- FFMPEG ----------
def clip_video(src, out, start, duration):
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-i", src, "-t", str(duration),
         "-c:v", "libx264", "-c:a", "aac", out],
        check=True
    )

# ---------- INGEST ----------
@app.post("/ingest")
async def ingest(req: Request):
    data = await req.json()
    print("[INGEST RAW]", data)

    url = data.get("url")
    folder = data.get("folder_id")
    title = data.get("video_title")

    # 🔁 MODE B: keep last file, do nothing
    if not url or not folder:
        if LATEST_INGEST["file_id"]:
            return {
                "status": "ok",
                "note": "using previous ingest",
                "drive_file_id": LATEST_INGEST["file_id"]
            }
        return {
            "status": "ok",
            "note": "no ingest performed"
        }

    # 🔽 MODE A: full ingest
    try:
        local = download_youtube(url, title or "video")
        uploaded = upload_to_drive(local, title or "video", folder)

        LATEST_INGEST.update({
            "file_id": uploaded["id"],
            "folder_id": folder,
            "title": title or "video"
        })

        return {
            "status": "ok",
            "drive_file_id": uploaded["id"]
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)

# ---------- ANALYZE + CLIP ----------
@app.post("/analyze-and-clip")
async def analyze_and_clip(req: Request, bg: BackgroundTasks):
    raw = await req.body()
    print("[RAW]", raw.decode(errors="ignore"))

    data = await req.json()
    segments = []

    raw_segments = data.get("segments_json")

    if isinstance(raw_segments, str) and '"segments"' in raw_segments:
        try:
            parsed = json.loads(raw_segments)
            segments = parsed.get("segments", [])
        except Exception:
            return {"status": "ok", "clips_found": 0}

    if not segments or not LATEST_INGEST["file_id"]:
        return {"status": "ok", "clips_found": 0}

    bg.add_task(
        run_clips,
        segments,
        data.get("video_title"),
        data.get("callback_url")
    )

    return {"status": "ok", "clips_found": len(segments)}

def run_clips(segments, title, callback):
    src = download_from_drive(LATEST_INGEST["file_id"])
    folder = LATEST_INGEST["folder_id"]
    base = title or LATEST_INGEST["title"]

    count = 0
    for i, seg in enumerate(segments, 1):
        out = f"/tmp/{base}_{i}.mp4"
        clip_video(src, out, seg["start"], seg["duration"])
        upload_to_drive(out, f"{base}_{i}", folder)
        count += 1

    if callback:
        requests.post(callback, json={
            "status": "done",
            "clips_created": count
        })

# ---------- HEALTH ----------
@app.get("/health")
async def health():
    return {"ok": True}
