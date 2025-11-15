import os
import json
import uuid
import datetime
import yt_dlp
import requests

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

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


# --------- DOWNLOAD VIDEO ----------
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


# --------- UPLOAD TO GOOGLE DRIVE SHARED DRIVE ----------
def upload_to_drive(file_path: str, title: str, job_id: str | None = None) -> dict:
    try:
        if job_id:
            log(job_id, "Preparing Drive credentials")

        creds = service_account.Credentials.from_service_account_info(
            json.loads(os.environ["GOOGLE_CREDENTIALS"]),
            scopes=["https://www.googleapis.com/auth/drive"]
        )

        drive = build("drive", "v3", credentials=creds)

        folder_id = os.environ.get("DRIVE_FOLDER_ID")
        if not folder_id:
            raise Exception("Missing DRIVE_FOLDER_ID env variable")

        file_metadata = {
            "name": title + ".mp4",
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
            # We *could* log progress here, but that’s just noise.

        if job_id:
            log(job_id, f"Drive upload complete: file {response.get('id')}")

        return response

    except Exception as e:
        if job_id:
            log(job_id, f"Drive upload failed: {e}")
        raise Exception(f"Drive upload failed: {e}")


# --------- BACKGROUND WORKER ----------
def process_job(job_id: str, url: str, title: str, callback_url: str | None) -> None:
    try:
        JOBS[job_id]["status"] = "downloading"
        log(job_id, "Stage → downloading")

        video_path = download_video(url, title, job_id=job_id)

        JOBS[job_id]["status"] = "uploading"
        log(job_id, "Stage → uploading")

        upload_result = upload_to_drive(video_path, title, job_id=job_id)

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


# --------- API: INGEST (QUEUE JOB) ----------
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
