import os
import json
import uuid
import shutil
import asyncio
import tempfile
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

import subprocess

app = FastAPI()

# ENV
FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")
GOOGLE_CREDS = os.getenv("GOOGLE_CREDENTIALS")
PUBLIC_DOMAIN = os.getenv("PUBLIC_DOMAIN")

if not FOLDER_ID or not GOOGLE_CREDS:
    raise RuntimeError("Missing environment variables DRIVE_FOLDER_ID or GOOGLE_CREDENTIALS.")

# ACTIVE JOBS MEMORY
jobs = {}

### ----------------------------------------------------------------------
### MODELS
### ----------------------------------------------------------------------

class IngestRequest(BaseModel):
    video_url: str
    title: str

### ----------------------------------------------------------------------
### HELPERS
### ----------------------------------------------------------------------

def now():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def append_log(job_id: str, message: str):
    jobs[job_id]["logs"].append(f"{now()} | {message}")


### ----------------------------------------------------------------------
### DRIVE UPLOAD (SHARED DRIVE FIXED)
### ----------------------------------------------------------------------

def upload_to_drive(filepath, filename):
    creds = service_account.Credentials.from_service_account_info(
        json.loads(GOOGLE_CREDS),
        scopes
