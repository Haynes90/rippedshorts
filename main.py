import os
import re
import uvicorn
import tempfile
from fastapi import FastAPI
from pydantic import BaseModel
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import subprocess

app = FastAPI()

FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")

class VideoRequest(BaseModel):
    video_url: str
    title: str

def sanitize_title(title):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', title)[:80]

def download_video(url, title):
    safe_title = sanitize_title(title)
    output_path = os.path.join(tempfile.gettempdir(), f"{safe_title}.mp4")

    subprocess.run([
        "yt-dlp",
        "-f", "bestvideo+bestaudio",
        "--merge-output-format", "mp4",
        "-o", output_path,
        url
    ], check=True)

    return output_path

def upload_to_drive(filepath, filename):
    creds = service_account.Credentials.from_service_account_info(
        eval(os.getenv("GOOGLE_CREDENTIALS")),
        scopes=["https://www.googleapis.com/auth/drive.file"]
    )

    drive_service = build("drive", "v3", credentials=creds)
    file_metadata = {
        "name": filename,
        "parents": [FOLDER_ID],
    }
    media = MediaFileUpload(filepath, mimetype="video/mp4")

    file = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id, webViewLink"
    ).execute()

    return file["webViewLink"]

@app.post("/ingest")
async def ingest(data: VideoRequest):
    safe_title = sanitize_title(data.title)
    filename = f"{safe_title}.mp4"
    try:
        path = download_video(data.video_url, safe_title)
        link = upload_to_drive(path, filename)
        os.remove(path)
        return {"status": "uploaded", "drive_link": link}
    except subprocess.CalledProcessError as e:
        return {"status": "failed", "error": str(e)}
