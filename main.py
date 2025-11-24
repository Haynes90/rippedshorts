import os
import json
import uuid
import datetime
import subprocess
import io

import yt_dlp
import requests

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

app = FastAPI()

# --------- SIMPLE IN-MEMORY JOB STORE ----------
JOBS: dict[str, dict] = {}

# NEW: Track the latest ingest file
LATEST_INGEST: dict = {
    "file_id": None,
    "title": None,
    "folder_id": None
}

def log(job_id: str, message: str) -> None:
    if job_id not in JOBS:
        return
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    JOBS[job_id].setdefault("logs", []).append(f"{ts} | {message}")


# --------- WRITE COOKIES FILE ----------
COOKIES_PATH = "/app/cookies.txt"
if "YOUTUBE_COOKIES" in os.environ:
    with open(COOKIES_PATH, "w") as f:
        f.write(os.environ["YOUTUBE_COOKIES"])


# --------- DRIVE CLIENT ----------
def get_drive_client():
    creds = service_account.Credentials.from_service_account_info(
        json.loads(os.environ["GOOGLE_CREDENTIALS"]),
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)


# --------- YOUTUBE DOWNLOAD ----------
def download_video(url: str, title: str, job_id: str | None = None) -> str:
    temp_path = f"/tmp/{title}.mp4"
    ydl_opts = {
        "outtmpl": temp_path,
        "format": "mp4",
        "cookies": COOKIES_PATH,
        "quiet": True,
        "noprogress": True,
    }
    if job_id: log(job_id, "yt-dlp starting download")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        if job_id: log(job_id, f"Download OK → {temp_path}")
        return temp_path
    except Exception as e:
        if job_id: log(job_id, f"Download failed: {e}")
        raise


# --------- DRIVE DOWNLOAD ----------
def download_from_drive(file_id: str, job_id: str | None = None) -> str:
    if job_id: log(job_id, f"Downloading source from Drive: {file_id}")
    drive = get_drive_client()

    request = drive.files().get_media(fileId=file_id, supportsAllDrives=True)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()

    local_path = f"/tmp/{file_id}.mp4"
    with open(local_path, "wb") as f:
        f.write(fh.getvalue())

    if job_id: log(job_id, f"Drive download complete → {local_path}")
    return local_path


# --------- DRIVE UPLOAD ----------
def upload_to_drive(file_path: str, title: str, folder_id: str | None = None, job_id: str | None = None) -> dict:
    if job_id: log(job_id, "Preparing Drive credentials")
    drive = get_drive_client()

    folder_id = folder_id or os.environ.get("DRIVE_FOLDER_ID")
    if not folder_id:
        raise Exception("Missing DRIVE_FOLDER_ID and no folder_id passed")

    file_metadata = {
        "name": f"{title}.mp4",
        "parents": [folder_id],
    }
    media = MediaFileUpload(file_path, mimetype="video/mp4", resumable=True)

    if job_id: log(job_id, "Drive upload starting")

    request = drive.files().create(
        body=file_metadata,
        media_body=media,
        fields="id, webViewLink",
        supportsAllDrives=True,
    )

    response = None
    while response is None:
        status, response = request.next_chunk(num_retries=5)

    if job_id: log(job_id, f"Drive upload complete: {response.get('id')}")
    return response


# --------- SAFE FILENAMES ----------
def sanitize_filename(name: str) -> str:
    bad_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for ch in bad_chars:
        name = name.replace(ch, "_")
    return name.strip() or "clip"


# --------- FFMPEG CLIP ----------
def make_clip(source_path: str, out_path: str, start: float, duration: float, job_id: str = None, clip_label: str = None):
    if job_id:
        log(job_id, f"ffmpeg clipping {clip_label or ''} @ {start}s for {duration}s")

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

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        if job_id:
            log(job_id, f"ffmpeg error: {result.stderr[-300:]}")
        raise Exception(f"ffmpeg failed")


# --------- BACKGROUND JOB: INGEST ----------
def process_job(job_id: str, url: str, title: str, callback_url: str | None):
    try:
        JOBS[job_id]["status"] = "downloading"
        log(job_id, "Stage → downloading")
        video_path = download_video(url, title, job_id)

        JOBS[job_id]["status"] = "uploading"
        upload_result = upload_to_drive(video_path, title, job_id=job_id)

        JOBS[job_id]["status"] = "success"
        JOBS[job_id]["drive_link"] = upload_result.get("webViewLink")
        JOBS[job_id]["file_id"] = upload_result.get("id")

        # UPDATE GLOBAL LATEST INGEST
        LATEST_INGEST["file_id"] = upload_result.get("id")
        LATEST_INGEST["title"] = title
        LATEST_INGEST["folder_id"] = os.environ.get("DRIVE_FOLDER_ID")

        log(job_id, "Job finished successfully")

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        log(job_id, f"Job failed: {e}")

    if callback_url:
        try:
            payload = {
                "job_id": job_id,
                "status": JOBS[job_id]["status"],
                "url": url,
                "title": title,
                "drive_link": JOBS[job_id].get("drive_link"),
                "file_id": JOBS[job_id].get("file_id"),
                "error": JOBS[job_id].get("error"),
            }
            requests.post(callback_url, json=payload, timeout=10)
        except Exception as e:
            log(job_id, f"Callback failed: {e}")


# --------- BACKGROUND JOB: CLIPPING ----------
def process_clip_job(job_id: str, payload: dict):
    drive_file_id = payload["drive_file_id"]
    folder_id = payload["folder_id"]
    video_title = payload.get("video_title") or drive_file_id
    clips = payload["clips"]
    callback_url = payload.get("callback_url")

    log(job_id, f"Clip job started for Drive file {drive_file_id}")
    JOBS[job_id]["status"] = "downloading"

    try:
        source_path = download_from_drive(drive_file_id, job_id)
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        return

    JOBS[job_id]["status"] = "clipping"
    JOBS[job_id]["clips"] = []

    base = sanitize_filename(video_title)

    for clip in clips:
        idx = int(clip["index"])
        start = float(clip["start"])
        duration = float(clip["duration"])

        clip_name = f"{base}__clip_{idx:02d}"
        out_path = f"/tmp/{clip_name}.mp4"

        try:
            make_clip(source_path, out_path, start, duration, job_id, f"clip_{idx}")

            upload_res = upload_to_drive(out_path, clip_name, folder_id, job_id)
            JOBS[job_id]["clips"].append({
                "index": idx,
                "file_id": upload_res["id"],
                "webViewLink": upload_res["webViewLink"],
            })

        except Exception as e:
            log(job_id, f"Clip {idx} failed: {e}")
            JOBS[job_id].setdefault("clip_errors", []).append(str(e))

    JOBS[job_id]["status"] = "success" if "clip_errors" not in JOBS[job_id] else "partial_success"

    if callback_url:
        try:
            requests.post(callback_url, json=JOBS[job_id], timeout=15)
        except Exception as e:
            log(job_id, f"Callback failed: {e}")


# --------- API: INGEST ----------
@app.post("/ingest")
async def ingest(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    url = data.get("url")
    if not url:
        return JSONResponse({"status": "error", "error": "Missing URL"}, status_code=400)

    callback = data.get("callback_url")
    title = url.split("v=")[-1].split("&")[0]

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"job_id": job_id, "url": url, "title": title, "status": "queued", "logs": []}
    background_tasks.add_task(process_job, job_id, url, title, callback)

    return {"status": "queued", "job_id": job_id, "title": title}


# --------- API: CLIPS (NO FILE ID NEEDED FROM ZAP) ----------
@app.post("/clips")
async def create_clips(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()

    # MUST have clips
    if not isinstance(data.get("clips"), list) or not data["clips"]:
        return JSONResponse({"status": "error", "error": "Missing or empty clips list"}, status_code=400)

    # Require latest ingest file
    if not LATEST_INGEST["file_id"]:
        return JSONResponse({"status": "error", "error": "No ingest has been processed yet"}, status_code=400)

    drive_file_id = LATEST_INGEST["file_id"]
    video_title = data.get("video_title") or LATEST_INGEST["title"]
    folder_id = data.get("folder_id") or LATEST_INGEST["folder_id"]

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "kind": "clips",
        "drive_file_id": drive_file_id,
        "logs": []
    }

    # Attach backend file id to payload
    data["drive_file_id"] = drive_file_id

    background_tasks.add_task(process_clip_job, job_id, data)

    return {
        "status": "queued",
        "job_id": job_id
    }


# --------- STATUS ----------
@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# --------- HEALTH ----------
@app.get("/health")
async def health():
    return {"status": "ok"}
