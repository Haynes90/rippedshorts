import os
import json
import uuid
import subprocess
import io
import yt_dlp
import requests

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

# --------------------------------------------------
# APP + STATE
# --------------------------------------------------
app = FastAPI()

LATEST_INGEST = {
    "file_id": None,
    "title": None,
    "folder_id": None
}

# --------------------------------------------------
# UTIL
# --------------------------------------------------
def raw_log(label, payload):
    try:
        print(f"[RAW] {label}: {json.dumps(payload)[:6000]}")
    except:
        print(f"[RAW] {label}: <unserializable>")

def sanitize(name):
    for c in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        name = name.replace(c, "_")
    return name.strip() or "clip"

# --------------------------------------------------
# GOOGLE DRIVE
# --------------------------------------------------
def drive_client():
    creds = service_account.Credentials.from_service_account_info(
        json.loads(os.environ["GOOGLE_CREDENTIALS"]),
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)

def upload_to_drive(path, name, folder_id):
    drive = drive_client()
    media = MediaFileUpload(path, mimetype="video/mp4", resumable=True)
    req = drive.files().create(
        body={"name": f"{name}.mp4", "parents": [folder_id]},
        media_body=media,
        fields="id",
        supportsAllDrives=True
    )
    res = None
    while res is None:
        _, res = req.next_chunk()
    return res

def download_from_drive(file_id):
    drive = drive_client()
    req = drive.files().get_media(fileId=file_id, supportsAllDrives=True)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, req)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    path = f"/tmp/{file_id}.mp4"
    with open(path, "wb") as f:
        f.write(fh.getvalue())
    if os.path.getsize(path) == 0:
        raise Exception("Downloaded Drive file is empty")
    return path

# --------------------------------------------------
# INGEST
# --------------------------------------------------
def download_youtube(url, title):
    out = f"/tmp/{sanitize(title)}.mp4"
    opts = {
        "format": "bv*+ba/best",
        "merge_output_format": "mp4",
        "outtmpl": out,
        "quiet": True,
        "extractor_args": {"youtube": {"player_client": ["android"]}}
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    if not os.path.exists(out) or os.path.getsize(out) == 0:
        raise Exception("YouTube download failed")
    return out

def process_ingest(url, title):
    path = download_youtube(url, title)
    up = upload_to_drive(path, title, os.environ["DRIVE_FOLDER_ID"])
    LATEST_INGEST.update({
        "file_id": up["id"],
        "title": title,
        "folder_id": os.environ["DRIVE_FOLDER_ID"]
    })

@app.post("/ingest")
async def ingest(req: Request, bg: BackgroundTasks):
    data = await req.json()
    raw_log("INGEST", data)

    if "url" not in data:
        return JSONResponse({"error": "missing url"}, 400)

    title = data["url"].split("v=")[-1].split("&")[0]
    bg.add_task(process_ingest, data["url"], title)
    return {"status": "queued"}

# --------------------------------------------------
# STEP 1 — BASE REEL
