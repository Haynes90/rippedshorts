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

# Track last ingest for automatic clipping
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


# --------- COOKIES ----------
COOKIES_PATH = "/app/cookies.txt"
if "YOUTUBE_COOKIES" in os.environ:
    with open(COOKIES_PATH, "w") as f:
        f.write(os.environ["YOUTUBE_COOKIES"])


# --------- GOOGLE DRIVE ----------
def get_drive_client():
    creds = service_account.Credentials.from_service_account_info(
        json.loads(os.environ["GOOGLE_CREDENTIALS"]),
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)


# --------- YOUTUBE DOWNLOAD ----------
def download_video(url: str, title: str, job_
