import os
import json
import tempfile
import yt_dlp
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

app = FastAPI()

# --------- WRITE COOKIES FILE FROM ENV ----------
COOKIES_PATH = "/app/cookies.txt"
if "YOUTUBE_COOKIES" in os.environ:
    with open(COOKIES_PATH, "w") as f:
        f.write(os.environ["YOUTUBE_COOKIES"])


# --------- DOWNLOAD VIDEO ----------
def download_video(url, title):
    temp_path = f"/tmp/{title}.mp4"

    ydl_opts = {
        "outtmpl": temp_path,
        "format": "mp4",
        "cookies": COOKIES_PATH,
        "quiet": True,
        "noprogress": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return temp_path

    except Exception as e:
        raise Exception(f"Download failed: {e}")


# --------- UPLOAD TO GOOGLE DRIVE SHARED DRIVE ----------
def upload_to_drive(file_path, title):
    try:
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
            "parents": [folder_id]
        }

        media = MediaFileUpload(file_path, mimetype="video/mp4", resumable=True)

        request = drive.files().create(
            body=file_metadata,
            media_body=media,
            fields="id, webViewLink",
            supportsAllDrives=True
        )

        response = None
        while response is None:
            status, response = request.next_chunk(num_retries=5)
        
        return response

    except Exception as e:
        raise Exception(f"Drive upload failed: {e}")


# --------- API INGEST ENDPOINT ----------
@app.post("/ingest")
async def ingest(request: Request):
    data = await request.json()

    if "url" not in data:
        return JSONResponse({"status": "failed", "error": "Missing URL"}, status_code=400)

    url = data["url"]
    title = url.split("v=")[-1]

    try:
        video_path = download_video(url, title)
        upload_result = upload_to_drive(video_path, title)

        return {
            "status": "success",
            "title": title,
            "drive_link": upload_result.get("webViewLink"),
            "file_id": upload_result.get("id")
        }

    except Exception as e:
        return {"status": "failed", "error": str(e)}
