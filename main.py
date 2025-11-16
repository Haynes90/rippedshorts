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
# This lives in memory on the container. Good enough for your use case.
JOBS: dict[str, dict] = {}

def log(job_id: str, message: str) -> None:
    """Append a timestamped message to a job's logs."""
    if job_id not in JOBS:
        return
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    JOBS[job_id].setdefault("logs", []).append(f"{ts} | {message}")


# --------- WRITE COOKIES FILE FROM ENV ----------
COOKIES_PATH = "/app/cookies.txt"
if "YOUTUBE_COOKIES" in os.environ:
    with open(COOKIES_PATH, "w") as f:
        f.write(os.environ["YOUTUBE_COOKIES"])


def get_drive_client():
    """Create an authorized Drive v3 client using the service account JSON in GOOGLE_CREDENTIALS."""
    creds = service_account.Credentials.from_service_account_info(
        json.loads(os.environ["GOOGLE_CREDENTIALS"]),
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)


# --------- DOWNLOAD VIDEO FROM YOUTUBE ----------
def download_video(url: str, title: str, job_id: str | None = None) -> str:
    temp_path = f"/tmp/{title}.mp4"

    ydl_opts = {
        "outtmpl": temp_path,
        "format": "mp4",
        "cookies": COOKIES_PATH,
        "quiet": True,
        "noprogress": True,
    }

    if job_id:
        log(job_id, "yt-dlp starting download")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if job_id:
            log(job_id, f"Download OK → {temp_path}")

        return temp_path

    except Exception as e:
        if job_id:
            log(job_id, f"Download failed: {e}")
        raise Exception(f"Download failed: {e}")


# --------- DOWNLOAD ORIGINAL FROM GOOGLE DRIVE ----------
def download_from_drive(file_id: str, job_id: str | None = None) -> str:
    """
    Download an existing video file from Drive (Shared Drive OK) to /tmp
    and return the local path.
    """
    if job_id:
        log(job_id, f"Downloading source from Drive: {file_id}")

    drive = get_drive_client()

    request = drive.files().get_media(fileId=file_id, supportsAllDrives=True)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()
        # Optionally log progress

    # Persist to disk
    local_path = f"/tmp/{file_id}.mp4"
    with open(local_path, "wb") as f:
        f.write(fh.getvalue())

    if job_id:
        log(job_id, f"Drive download complete → {local_path}")

    return local_path


# --------- UPLOAD TO GOOGLE DRIVE SHARED DRIVE ----------
def upload_to_drive(file_path: str, title: str, folder_id: str | None = None, job_id: str | None = None) -> dict:
    """
    Upload a local file to Drive.
    - title: base name without extension
    - folder_id: explicit Drive folder; if None, falls back to DRIVE_FOLDER_ID env
    """
    try:
        if job_id:
            log(job_id, "Preparing Drive credentials")

        drive = get_drive_client()

        if not folder_id:
            folder_id = os.environ.get("DRIVE_FOLDER_ID")

        if not folder_id:
            raise Exception("Missing DRIVE_FOLDER_ID env variable and no folder_id passed")

        file_metadata = {
            "name": f"{title}.mp4",
            "parents": [folder_id],
        }

        media = MediaFileUpload(file_path, mimetype="video/mp4", resumable=True)

        if job_id:
            log(job_id, "Drive resumable upload starting")

        request = drive.files().create(
            body=file_metadata,
            media_body=media,
            fields="id, webViewLink",
            supportsAllDrives=True,
        )

        response = None
        while response is None:
            status, response = request.next_chunk(num_retries=5)

        if job_id:
            log(job_id, f"Drive upload complete: file {response.get('id')}")

        return response

    except Exception as e:
        if job_id:
            log(job_id, f"Drive upload failed: {e}")
        raise Exception(f"Drive upload failed: {e}")


# --------- CLIP VIDEO LOCALLY WITH FFMPEG ----------
def sanitize_filename(name: str) -> str:
    # Remove/replace characters not allowed in filenames
    bad_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for ch in bad_chars:
        name = name.replace(ch, "_")
    return name.strip() or "clip"


def make_clip(
    source_path: str,
    out_path: str,
    start: float,
    duration: float,
    job_id: str | None = None,
    clip_label: str | None = None,
) -> None:
    """
    Use ffmpeg to cut a segment from source_path into out_path.
    Re-encodes for reliability on arbitrary MP4s.
    """
    if job_id:
        log(job_id, f"ffmpeg clipping {clip_label or ''} @ {start}s for {duration}s")

    cmd = [
        "ffmpeg",
        "-y",               # overwrite
        "-ss", str(start),  # seek start
        "-i", source_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "veryfast",
        out_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            if job_id:
                log(job_id, f"ffmpeg error: {result.stderr[-400:]}")
            raise Exception(f"ffmpeg failed with code {result.returncode}")
        if job_id:
            log(job_id, f"ffmpeg done → {out_path}")
    except Exception as e:
        if job_id:
            log(job_id, f"ffmpeg exception: {e}")
        raise


# --------- BACKGROUND WORKER: FULL VIDEO DOWNLOAD + UPLOAD ----------
def process_job(job_id: str, url: str, title: str, callback_url: str | None) -> None:
    try:
        JOBS[job_id]["status"] = "downloading"
        log(job_id, "Stage → downloading")

        video_path = download_video(url, title, job_id=job_id)

        JOBS[job_id]["status"] = "uploading"
        log(job_id, "Stage → uploading")

        upload_result = upload_to_drive(video_path, title, folder_id=None, job_id=job_id)

        JOBS[job_id]["status"] = "success"
        JOBS[job_id]["drive_link"] = upload_result.get("webViewLink")
        JOBS[job_id]["file_id"] = upload_result.get("id")
        log(job_id, "Job finished successfully")

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        log(job_id, f"Job failed: {e}")

    # Optional callback to Zapier (or anything else)
    if callback_url:
        try:
            payload = {
                "job_id": job_id,
                "status": JOBS[job_id]["status"],
                "url": JOBS[job_id]["url"],
                "title": JOBS[job_id]["title"],
                "drive_link": JOBS[job_id].get("drive_link"),
                "file_id": JOBS[job_id].get("file_id"),
                "error": JOBS[job_id].get("error"),
            }
            log(job_id, f"Posting callback to {callback_url}")
            requests.post(callback_url, json=payload, timeout=10)
        except Exception as e:
            log(job_id, f"Callback failed: {e}")
            JOBS[job_id]["callback_error"] = str(e)


# --------- BACKGROUND WORKER: CLIP EXISTING DRIVE VIDEO ----------
def process_clip_job(job_id: str, payload: dict) -> None:
    """
    payload expected:
    {
      "drive_file_id": "...",
      "video_title": "Full Video Title",
      "folder_id": "drive-folder-id",
      "clips": [
         {"index": 1, "start": 12.3, "duration": 25.0},
         ...
      ],
      "callback_url": "https://hooks.zapier.com/...",
    }
    """
    drive_file_id = payload["drive_file_id"]
    video_title = payload.get("video_title") or drive_file_id
    folder_id = payload.get("folder_id")
    clips = payload.get("clips", [])
    callback_url = payload.get("callback_url")

    JOBS[job_id]["status"] = "preparing"
    log(job_id, f"Clip job started for Drive file {drive_file_id}")

    # 1) Download original from Drive once
    try:
        source_path = download_from_drive(drive_file_id, job_id=job_id)
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = f"Drive download failed: {e}"
        log(job_id, JOBS[job_id]["error"])
        return

    JOBS[job_id]["status"] = "clipping"
    JOBS[job_id]["clips"] = []
    log(job_id, f"Clipping {len(clips)} segments")

    base_name = sanitize_filename(video_title)

    for clip in clips:
        try:
            idx = int(clip.get("index", 0))
        except Exception:
            idx = 0
        start = float(clip.get("start", 0))
        duration = float(clip.get("duration", 0))

        clip_label = f"clip_{idx}"
        clip_filename = f"{base_name}__clip_{idx:02d}"
        local_clip_path = f"/tmp/{clip_filename}.mp4"

        try:
            make_clip(
                source_path=source_path,
                out_path=local_clip_path,
                start=start,
                duration=duration,
                job_id=job_id,
                clip_label=clip_label,
            )

            upload_res = upload_to_drive(
                local_clip_path,
                title=clip_filename,
                folder_id=folder_id,
                job_id=job_id,
            )

            clip_record = {
                "index": idx,
                "start": start,
                "duration": duration,
                "file_id": upload_res.get("id"),
                "webViewLink": upload_res.get("webViewLink"),
            }
            JOBS[job_id]["clips"].append(clip_record)
            log(job_id, f"Clip {idx} uploaded as {upload_res.get('id')}")

        except Exception as e:
            # We keep going; record failure for this clip
            err_msg = f"Clip {idx} failed: {e}"
            log(job_id, err_msg)
            JOBS[job_id].setdefault("clip_errors", []).append(err_msg)

    # Final status
    if JOBS[job_id].get("clip_errors"):
        JOBS[job_id]["status"] = "partial_success"
    else:
        JOBS[job_id]["status"] = "success"

    log(job_id, f"Clip job finished with status {JOBS[job_id]['status']}")

    # Callback if present
    if callback_url:
        try:
            payload = {
                "job_id": job_id,
                "status": JOBS[job_id]["status"],
                "drive_file_id": drive_file_id,
                "video_title": video_title,
                "folder_id": folder_id,
                "clips": JOBS[job_id].get("clips", []),
                "clip_errors": JOBS[job_id].get("clip_errors"),
            }
            log(job_id, f"Posting clip callback to {callback_url}")
            requests.post(callback_url, json=payload, timeout=15)
        except Exception as e:
            log(job_id, f"Clip callback failed: {e}")
            JOBS[job_id]["callback_error"] = str(e)


# --------- API: INGEST (QUEUE FULL VIDEO JOB) ----------
@app.post("/ingest")
async def ingest(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()

    url = data.get("url")
    if not url:
        return JSONResponse({"status": "failed", "error": "Missing URL"}, status_code=400)

    # Optional callback URL Zapier will give us
    callback_url = data.get("callback_url")

    # Basic YouTube ID-style title
    title = url.split("v=")[-1].split("&")[0]

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "url": url,
        "title": title,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "logs": [],
    }
    log(job_id, "Job created")
    log(job_id, "Stage → precheck")

    # Kick work to background so the HTTP request returns fast
    background_tasks.add_task(process_job, job_id, url, title, callback_url)

    return {
        "status": "queued",
        "job_id": job_id,
        "title": title,
        "message": "Job queued; processing in background.",
    }


# --------- API: CLIP EXISTING DRIVE VIDEO ----------
@app.post("/clips")
async def create_clips(request: Request, background_tasks: BackgroundTasks):
    """
    Accepts JSON:
    {
      "drive_file_id": "1abc...",
      "video_title": "Original YT Title",
      "folder_id": "1KuNb6JZ...",   # TCB folder, or whatever
      "clips": [
        { "index": 1, "start": 12.6, "duration": 25.0 },
        { "index": 2, "start": 80.0, "duration": 30.0 }
      ],
      "callback_url": "https://hooks.zapier.com/..."
    }
    """
    data = await request.json()

    if not data.get("drive_file_id"):
        return JSONResponse(
            {"status": "failed", "error": "Missing drive_file_id"},
            status_code=400,
        )

    clips = data.get("clips") or []
    if not isinstance(clips, list) or not clips:
        return JSONResponse(
            {"status": "failed", "error": "Missing or empty clips array"},
            status_code=400,
        )

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "kind": "clips",
        "payload": data,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "logs": [],
    }
    log(job_id, "Clip job created")

    background_tasks.add_task(process_clip_job, job_id, data)

    return {
        "status": "queued",
        "job_id": job_id,
        "message": "Clip job queued; processing in background.",
    }


# --------- API: STATUS ----------
@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# --------- SIMPLE HEALTH CHECK ----------
@app.get("/health")
async def health():
    return {"status": "ok"}
