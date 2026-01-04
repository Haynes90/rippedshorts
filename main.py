import os
import json
import uuid
import datetime
import subprocess
import io
import time
import yt_dlp
import requests

from typing import List, Dict
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

# Step-2 deps
import cv2
import numpy as np
import mediapipe as mp

# transcript
from youtube_transcript_api import YouTubeTranscriptApi


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
# LOGGING
# --------------------------------------------------
def ts():
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def debug(msg):
    print(f"[DEBUG] {ts()} | {msg}")

def raw_log(label, payload):
    try:
        print(f"[RAW] {label}: {json.dumps(payload)[:8000]}")
    except Exception:
        print(f"[RAW] {label}: <unserializable>")


# --------------------------------------------------
# UTIL
# --------------------------------------------------
def sanitize(name):
    for c in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        name = name.replace(c, "_")
    return name.strip() or "clip"

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def run_cmd(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise Exception(r.stderr.decode("utf-8", errors="ignore")[:8000])
    return r


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
        raise Exception("Drive video empty")
    return path


# --------------------------------------------------
# INGEST (FULL VIDEO — OPTIONAL)
# --------------------------------------------------
@app.post("/ingest")
async def ingest(req: Request):
    data = await req.json()
    raw_log("INGEST", data)

    if "url" not in data:
        return JSONResponse({"error": "missing url"}, 400)

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "running"}

    try:
        url = data["url"]
        title = url.split("v=")[-1].split("&")[0]
        out = f"/tmp/{sanitize(title)}.mp4"

        opts = {
            "format": "bv*+ba/best",
            "merge_output_format": "mp4",
            "outtmpl": out,
            "quiet": True,
            "retries": 5,
            "fragment_retries": 5,
            "cookies": COOKIES_PATH if os.path.exists(COOKIES_PATH) else None,
        }
        if opts["cookies"] is None:
            opts.pop("cookies", None)

        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

        up = upload_to_drive(out, title, os.environ["DRIVE_FOLDER_ID"])

        LATEST_INGEST.update({
            "file_id": up["id"],
            "title": title,
            "folder_id": os.environ["DRIVE_FOLDER_ID"]
        })

        JOBS[job_id]["status"] = "success"
        JOBS[job_id]["file_id"] = up["id"]

        if "callback_url" in data:
            requests.post(data["callback_url"], json=JOBS[job_id], timeout=10)

        return {"status": "success", "job_id": job_id, "file_id": up["id"]}

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        debug(f"INGEST ERROR: {e}")
        return JSONResponse({"error": str(e)}, 500)


# --------------------------------------------------
# DISCOVER (TRANSCRIPT-ONLY, CHUNKED)
# --------------------------------------------------
def fetch_transcript_chunks(video_id, chunk_seconds=900):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    chunks = []

    buf = []
    start = None
    dur = 0

    for seg in transcript:
        if start is None:
            start = seg["start"]

        buf.append(seg["text"])
        dur += seg["duration"]

        if dur >= chunk_seconds:
            chunks.append({
                "start": start,
                "duration": dur,
                "text": " ".join(buf)
            })
            buf, start, dur = [], None, 0

    if buf:
        chunks.append({
            "start": start,
            "duration": dur,
            "text": " ".join(buf)
        })

    return chunks


def discover_segments_from_chunk(chunk):
    text = chunk["text"].lower()
    segments = []

    if any(k in text for k in ["jesus", "lord", "faith", "worship"]):
        segments.append({
            "start": chunk["start"],
            "duration": min(60, chunk["duration"]),
            "score": 85,
            "category": "education",
            "reason": "Faith-centered teaching moment"
        })

    return segments


@app.post("/discover")
async def discover(req: Request):
    data = await req.json()
    raw_log("DISCOVER", data)

    video_id = data.get("video_id")
    callback_url = data.get("callback_url")

    if not video_id or not callback_url:
        return JSONResponse({"error": "missing video_id or callback_url"}, 400)

    try:
        all_segments = []
        chunks = fetch_transcript_chunks(video_id)

        for idx, chunk in enumerate(chunks):
            found = discover_segments_from_chunk(chunk)
            all_segments.extend(found)

            requests.post(
                callback_url,
                json={
                    "video_id": video_id,
                    "chunk_index": idx,
                    "segments": found,
                    "complete": False
                },
                timeout=10
            )

            time.sleep(0.3)

        requests.post(
            callback_url,
            json={
                "video_id": video_id,
                "segments": all_segments,
                "complete": True
            },
            timeout=10
        )

        return {"status": "ok", "segments_found": len(all_segments)}

    except Exception as e:
        debug(f"DISCOVER ERROR: {e}")
        return JSONResponse({"error": str(e)}, 500)


# --------------------------------------------------
# ANALYZE + CLIP (UNCHANGED FROM YOUR PIPELINE)
# --------------------------------------------------
# ⬇️ EVERYTHING BELOW HERE IS IDENTICAL TO WHAT YOU ALREADY HAVE
# ⬇️ normalize_segments, ffmpeg, ReelFramer, analyze-and-clip, health
# ⬇️ (LEFT OUT HERE FOR BREVITY IN THIS RESPONSE)
# --------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}
