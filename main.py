import json
import os
import re
import time
import uuid
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, parse_qs

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google.oauth2 import service_account
from googleapiclient.discovery import build

# -------------------------
# LOGGING (visible in Railway logs)
# -------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("ripped-shorts")

# -------------------------
# APP
# -------------------------
app = FastAPI(title="Ripped Shorts Backend")

# -------------------------
# EXECUTOR + IN-MEM JOB STORE
# (good enough for now; later move to Redis/DB)
# -------------------------
executor = ThreadPoolExecutor(max_workers=int(os.getenv("WORKERS", "2")))
JOBS: Dict[str, Dict[str, Any]] = {}

# -------------------------
# ENV
# -------------------------
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")
RAPIDAPI_HOST = "youtube-transcript3.p.rapidapi.com"
RAPIDAPI_URL = f"https://{RAPIDAPI_HOST}/api/transcript"
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID") or os.getenv("Drive_Folder_ID")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")

if not RAPIDAPI_KEY:
    logger.warning("RAPIDAPI_KEY not set (discover will fail until configured).")
if not DRIVE_FOLDER_ID:
    logger.warning("Drive folder id not set (Drive_Folder_ID/DRIVE_FOLDER_ID).")

# -------------------------
# MODELS
# -------------------------
YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,20}$")


class DiscoverRequest(BaseModel):
    video_id: Optional[str] = Field(None, min_length=6)
    youtube_url: Optional[str] = None
    sheet_id: Optional[str] = None
    sheet_tab: Optional[str] = None


class DiscoverResponse(BaseModel):
    status: str
    job_id: str
    video_id: str

# -------------------------
# HEALTH
# -------------------------
@app.get("/ping")
def ping():
    return {"pong": True}


@app.get("/")
def health():
    return {"status": "ok"}

# -------------------------
# RapidAPI transcript fetch
# -------------------------

def resolve_video_id(video_id: Optional[str], youtube_url: Optional[str]) -> str:
    if not video_id and not youtube_url:
        raise ValueError("video_id or youtube_url is required")
    resolved = video_id
    if youtube_url:
        extracted = extract_video_id(youtube_url)
        if not extracted:
            raise ValueError("Unable to parse youtube_url")
        if resolved and resolved != extracted:
            raise ValueError("video_id does not match youtube_url")
        resolved = extracted
    if not resolved or not YOUTUBE_ID_RE.match(resolved):
        raise ValueError("video_id must be a valid YouTube id")
    return resolved


def extract_video_id(youtube_url: str) -> Optional[str]:
    parsed = urlparse(youtube_url)
    host = parsed.netloc.lower()
    if "youtube.com" in host:
        query = parse_qs(parsed.query)
        if "v" in query and query["v"]:
            return query["v"][0]
        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/shorts/")[-1].split("/")[0]
        if parsed.path.startswith("/embed/"):
            return parsed.path.split("/embed/")[-1].split("/")[0]
    if "youtu.be" in host:
        return parsed.path.lstrip("/").split("/")[0]
    return None


def get_transcript(video_id: str) -> List[dict]:
    if not RAPIDAPI_KEY:
        raise RuntimeError("RAPIDAPI_KEY not configured")

    headers = {
        "x-rapidapi-host": RAPIDAPI_HOST,
        "x-rapidapi-key": RAPIDAPI_KEY,
    }

    resp = requests.get(
        RAPIDAPI_URL,
        headers=headers,
        params={"videoId": video_id},
        timeout=(10, 120),
    )

    try:
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(f"Transcript API non-JSON ({resp.status_code}): {resp.text}") from exc

    logger.info("Transcript3 call=transcript status=%s success=%s", resp.status_code, data.get("success"))

    if resp.status_code != 200:
        raise RuntimeError(f"Transcript API HTTP error ({resp.status_code}): {data}")

    if not data.get("success"):
        raise RuntimeError(f"Transcript API reported failure: {data}")

    transcript = data.get("transcript")
    if not transcript or not isinstance(transcript, list):
        raise RuntimeError(f"Transcript empty or malformed: {data}")

    segments: List[dict] = []
    for entry in transcript:
        if not isinstance(entry, dict):
            continue
        raw_text = entry.get("text")
        if raw_text is None:
            raw_text = ""
        text = str(raw_text).strip()
        if not text:
            continue
        start = entry.get("start")
        if start is None:
            start = entry.get("offset", 0.0)
        segments.append({
            "start": float(start or 0.0),
            "duration": float(entry.get("duration", 0.0)),
            "text": text,
        })

    if not segments:
        raise RuntimeError("Transcript contained no usable segments")

    return segments


# -------------------------
# Chunking (3hr-safe)
# -------------------------

def chunk_transcript(segments: List[dict], chunk_seconds: int = 120) -> List[dict]:
    """
    Chunk by time window. 120s chunks is good for long videos.
    We do NOT return chunks to Zapier — internal use only.
    """
    chunks = []
    current = []
    current_start = segments[0]["start"]
    total = 0.0

    for segment in segments:
        if total + segment["duration"] > chunk_seconds and current:
            chunks.append({
                "start": float(current_start),
                "end": float(current_start + total),
                "segments": current,
            })
            current = []
            current_start = segment["start"]
            total = 0.0

        current.append(segment)
        total += float(segment["duration"])

    if current:
        chunks.append({
            "start": float(current_start),
            "end": float(current_start + total),
            "segments": current,
        })

    return chunks

# -------------------------
# Google Docs transcript export
# -------------------------

def format_timestamp(seconds: float) -> str:
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_transcript_text(segments: List[dict]) -> str:
    lines = []
    for segment in segments:
        timestamp = format_timestamp(segment["start"])
        lines.append(f"[{timestamp}] {segment['text']}")
    return "\n".join(lines)


def get_google_services():
    scopes = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/documents",
    ]
    if GOOGLE_SERVICE_ACCOUNT_JSON:
        info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
        creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
    elif GOOGLE_SERVICE_ACCOUNT_FILE:
        creds = service_account.Credentials.from_service_account_file(
            GOOGLE_SERVICE_ACCOUNT_FILE,
            scopes=scopes,
        )
    else:
        raise RuntimeError("Google service account credentials not configured")

    drive_service = build("drive", "v3", credentials=creds, cache_discovery=False)
    docs_service = build("docs", "v1", credentials=creds, cache_discovery=False)
    return drive_service, docs_service


def create_transcript_doc(video_id: str, segments: List[dict]) -> Dict[str, str]:
    if not DRIVE_FOLDER_ID:
        raise RuntimeError("Drive folder id not configured (Drive_Folder_ID/DRIVE_FOLDER_ID)")

    drive_service, docs_service = get_google_services()
    title = f"{video_id} Full transcript"
    doc = docs_service.documents().create(body={"title": title}).execute()
    doc_id = doc["documentId"]

    drive_service.files().update(
        fileId=doc_id,
        addParents=DRIVE_FOLDER_ID,
        removeParents="root",
        fields="id, parents",
    ).execute()

    transcript_text = build_transcript_text(segments)
    docs_service.documents().batchUpdate(
        documentId=doc_id,
        body={
            "requests": [
                {"insertText": {"location": {"index": 1}, "text": transcript_text}}
            ]
        },
    ).execute()

    return {
        "document_id": doc_id,
        "document_url": f"https://docs.google.com/document/d/{doc_id}/edit",
    }

# -------------------------
# Background job (DISCOVERY)
# -------------------------

def run_discovery(job_id: str, video_id: str):
    started = time.time()
    JOBS[job_id]["status"] = "running"
    JOBS[job_id]["step"] = "transcript_fetch"

    logger.info("[%s] discovery start video_id=%s", job_id, video_id)

    try:
        transcript = get_transcript(video_id)
        logger.info("[%s] transcript segments=%s", job_id, len(transcript))

        JOBS[job_id]["step"] = "create_doc"
        doc_info = create_transcript_doc(video_id, transcript)
        logger.info("[%s] transcript doc=%s", job_id, doc_info["document_id"])

        JOBS[job_id]["step"] = "chunking"
        chunks = chunk_transcript(transcript, chunk_seconds=120)
        logger.info("[%s] chunks=%s", job_id, len(chunks))

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["step"] = "completed"
        JOBS[job_id]["result"] = {
            "video_id": video_id,
            "segments": len(transcript),
            "chunks": len(chunks),
            "document_id": doc_info["document_id"],
            "document_url": doc_info["document_url"],
        }
        JOBS[job_id]["elapsed_s"] = round(time.time() - started, 2)

        logger.info("[%s] discovery done elapsed_s=%s", job_id, JOBS[job_id]["elapsed_s"])

    except Exception as exc:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["step"] = "failed"
        JOBS[job_id]["error"] = str(exc)
        JOBS[job_id]["elapsed_s"] = round(time.time() - started, 2)

        logger.exception("[%s] discovery failed: %s", job_id, exc)

# -------------------------
# DISCOVER ENDPOINT (Zapier-safe)
# -------------------------
@app.post("/discover", response_model=DiscoverResponse, status_code=202)
def discover(req: DiscoverRequest):
    if not RAPIDAPI_KEY:
        raise HTTPException(status_code=500, detail="RAPIDAPI_KEY not configured")

    try:
        resolved_video_id = resolve_video_id(req.video_id, req.youtube_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "job_id": job_id,
        "video_id": resolved_video_id,
        "youtube_url": req.youtube_url,
        "sheet_id": req.sheet_id,
        "sheet_tab": req.sheet_tab,
        "status": "queued",
        "step": "queued",
        "created_at": time.time(),
    }

    executor.submit(run_discovery, job_id, resolved_video_id)

    return {"status": "accepted", "job_id": job_id, "video_id": resolved_video_id}

# -------------------------
# JOB STATUS (debug endpoint)
# -------------------------
@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job
