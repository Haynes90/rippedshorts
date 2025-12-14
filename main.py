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
d
