import os
import re
import uuid
import json
import tempfile
import asyncio
import subprocess
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

app = FastAPI()

# ----------------------------
# In-memory job tracking
# ----------------------------
JOBS = {}

# ----------------------------
# Models
# ----------------------------
class VideoRequest(BaseModel):
    video_url: str
    title: str

# ----------------------------
# Utilities
# ----------------------------
def sanitize_title(title: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", title)[:80]


def load_credentials():
    raw = os.getenv("GOOGLE_CREDENTIALS")
    if not raw:
        raise ValueError("GOOGLE_CREDENTIALS missing")

    try:
        data = json.loads(raw)
        if "private_key" not in data:
            raise ValueError("Invalid service account JSON — missing private_key")
        return data
    except Exception as e:
        raise ValueError(f"GOOGLE_CREDENTIALS JSON parse error: {str(e)}")


def get_drive_service():
    creds_data = load_credentials()
    creds = service_account.Credentials.from_service_account_info(
        creds_data,
        scopes=["https://www.googleapis.com/auth/drive.file"]
    )
    return build("drive", "v3", credentials=creds)


def log(job_id, message):
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    JOBS[job_id]["logs"].append(f"{timestamp} | {message}")


async def download_video(job_id, url, filename):
    log(job_id, "yt-dlp starting")
    output_path = f"/tmp/{filename}.mp4"

    try:
        subprocess.run([
            "yt-dlp",
            "-f", "bestvideo+bestaudio",
            "--merge-output-format", "mp4",
            "-o", output_path,
            url
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        log(job_id, f"yt-dlp error: {e.stderr.decode() if e.stderr else str(e)}")
        raise

    log(job_id, f"Download OK → {output_path}")
    return output_path


async def upload_to_drive(job_id, filepath, filename):
    folder_id = os.getenv("DRIVE_FOLDER_ID")
    if not folder_id:
        raise ValueError("DRIVE_FOLDER_ID missing")

    service = get_drive_service()
    log(job_id, "Drive resumable upload starting")

    media = MediaFileUpload(filepath, mimetype="video/mp4", resumable=True)
    request = service.files().create(
        body={"name": filename, "parents": [folder_id]},
        media_body=media,
        fields="id, webViewLink"
    )

    response = None
    while response is None:
        try:
            status, response = request.next_chunk()
        except Exception as e:
            log(job_id, f"Transient upload error: {str(e)}; retrying in 2s")
            await asyncio.sleep(2)

    return response["webViewLink"]


async def process_job(job_id):
    job = JOBS[job_id]
    url = job["url"]
    title = sanitize_title(job["title"])
    filename = f"{title}"

    try:
        log(job_id, "Stage → precheck")
        log(job_id, "Stage → downloading")
        path = await download_video(job_id, url, filename)

        log(job_id, "Stage → uploading")
        link = await upload_to_drive(job_id, path, f"{filename}.mp4")

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["drive_link"] = link

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        log(job_id, f"ERROR: {str(e)}")


# ----------------------------
# API Routes
# ----------------------------
@app.post("/ingest")
async def ingest(req: VideoRequest):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status": "queued",
        "title": req.title,
        "url": req.video_url,
        "logs": []
    }
    log(job_id, "Job created")

    asyncio.create_task(process_job(job_id))

    return {
        "job_id": job_id,
        "status_url": f"{os.getenv('PUBLIC_DOMAIN', '')}/status/{job_id}"
    }


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in JOBS:
        return {"detail": "Job not found"}
    return JOBS[job_id]


@app.get("/diagnostics")
async def diagnostics():
    diag = {
        "env": {
            "DRIVE_FOLDER_ID_present": bool(os.getenv("DRIVE_FOLDER_ID")),
            "GOOGLE_CREDENTIALS_present": bool(os.getenv("GOOGLE_CREDENTIALS")),
            "PUBLIC_DOMAIN": os.getenv("PUBLIC_DOMAIN")
        },
        "
