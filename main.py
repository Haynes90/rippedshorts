import os
import json
import uuid
import threading
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import yt_dlp

# -------------------------
# GLOBAL JOB STORAGE
# -------------------------

jobs = {}

# -------------------------
# FASTAPI APP
# -------------------------

app = FastAPI()


# -------------------------
# REQUEST MODEL
# -------------------------

class IngestRequest(BaseModel):
    url: str
    title: str | None = None


# -------------------------
# YOUTUBE DOWNLOAD
# -------------------------

def download_video(url, out_path):
    ydl_opts = {
        "outtmpl": out_path,
        "format": "mp4",
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


# -------------------------
# GOOGLE DRIVE UPLOAD
# -------------------------

def upload_to_drive(file_path, title):
    try:
        info = json.loads(os.environ["GOOGLE_CREDENTIALS"])
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/drive"]
        )

        drive = build("drive", "v3", credentials=creds)

        folder_id = os.environ["DRIVE_FOLDER_ID"]

        file_metadata = {
            "name": f"{title}.mp4",
            "parents": [folder_id]  # regular MyDrive folder — works fine
        }

        media = MediaFileUpload(file_path, mimetype="video/mp4", resumable=True)

        request = drive.files().create(
            body=file_metadata,
            media_body=media,
            fields="id, webViewLink",
            supportsAllDrives=False     # My Drive ONLY
        )

        response = None
        while response is None:
            status, response = request.next_chunk(num_retries=5)
            if status:
                print(f"Uploaded {int(status.progress() * 100)}%")

        return response

    except Exception as e:
        print("Upload failed:", e)
        raise


# -------------------------
# BACKGROUND WORKER
# -------------------------

def process_job(job_id, url, title):
    try:
        jobs[job_id]["status"] = "downloading"

        temp_path = f"/tmp/{job_id}.mp4"
        download_video(url, temp_path)

        jobs[job_id]["status"] = "uploading"

        result = upload_to_drive(temp_path, title)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = result

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


# -------------------------
# ROUTE: /ingest
# -------------------------

@app.post("/ingest")
def ingest(request: IngestRequest):
    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "status": "queued",
        "url": request.url,
        "title": request.title or "video",
        "result": None,
        "error": None,
    }

    thread = threading.Thread(
        target=process_job,
        args=(job_id, request.url, request.title or "video"),
        daemon=True
    )
    thread.start()

    return {"job_id": job_id}


# -------------------------
# ROUTE: /status/<job_id>
# -------------------------

@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs:
        return JSONResponse(
            {"error": "Job not found"},
            status_code=404
        )

    return jobs[job_id]
