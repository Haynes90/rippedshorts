import os
import re
import json
import uuid
import time
import shutil
import tempfile
import threading
from typing import Optional, Dict, Any, List, Tuple

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, HttpUrl
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from yt_dlp import YoutubeDL

# =========================
# App & Config
# =========================
app = FastAPI(title="YouTube → Drive Ingest", version="1.1.0")

DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")
GOOGLE_CREDENTIALS = os.getenv("GOOGLE_CREDENTIALS")
PUBLIC_DOMAIN = os.getenv("RAILWAY_PUBLIC_DOMAIN") or os.getenv("PUBLIC_BASE_URL") or ""  # optional

# Simple in-memory job store
JOBS: Dict[str, Dict[str, Any]] = {}  # {job_id: {"status": ..., "logs": [...], "result": {...}, "error": {...}}}

# =========================
# Models
# =========================
class IngestRequest(BaseModel):
    video_url: HttpUrl
    title: str = Field(min_length=1, max_length=200)
    callback_url: Optional[HttpUrl] = None  # Zapier hook to call on completion/failure (optional)

class IngestResponse(BaseModel):
    job_id: str
    status: str
    status_url: str

class ErrorReport(BaseModel):
    code: str
    message: str
    suggestion: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

# =========================
# Helper: Job logging
# =========================
def job_log(job_id: str, message: str):
    entry = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}"
    JOBS[job_id].setdefault("logs", []).append(entry)

def job_fail(job_id: str, code: str, message: str, suggestion: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
    JOBS[job_id]["status"] = "failed"
    JOBS[job_id]["error"] = ErrorReport(code=code, message=message, suggestion=suggestion, details=details).model_dump()
    job_log(job_id, f"ERROR [{code}]: {message}")

def job_stage(job_id: str, stage: str):
    JOBS[job_id]["status"] = stage
    job_log(job_id, f"Stage → {stage}")

# =========================
# Helper: Diagnostics
# =========================
def parse_creds() -> Tuple[Optional[Any], Optional[str]]:
    if not GOOGLE_CREDENTIALS:
        return None, "GOOGLE_CREDENTIALS not set"
    try:
        info = json.loads(GOOGLE_CREDENTIALS)
        creds = service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/drive.file"]
        )
        return creds, None
    except Exception as e:
        return None, f"Invalid GOOGLE_CREDENTIALS JSON: {e}"

def drive_service(creds):
    # cache_discovery=False to avoid warnings in serverless
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def check_drive_access(creds) -> Tuple[bool, Optional[str]]:
    try:
        svc = drive_service(creds)
        # Quick metadata call to validate auth; also try listing to confirm folder visibility
        if not DRIVE_FOLDER_ID:
            return False, "DRIVE_FOLDER_ID not set"
        q = f"'{DRIVE_FOLDER_ID}' in parents and trashed=false"
        svc.files().list(q=q, pageSize=1, fields="files(id,name)").execute()
        return True, None
    except Exception as e:
        return False, f"Drive access error: {e}"

def check_binaries() -> Dict[str, bool]:
    return {
        "ffmpeg_found": shutil.which("ffmpeg") is not None,
        "yt_dlp_found": shutil.which("yt-dlp") is not None or shutil.which("yt_dlp") is not None
    }

def free_disk_mb(path: str) -> int:
    usage = shutil.disk_usage(path)
    return int(usage.free / (1024 * 1024))

def diagnostics() -> Dict[str, Any]:
    diag: Dict[str, Any] = {}
    diag["env"] = {
        "DRIVE_FOLDER_ID_present": bool(DRIVE_FOLDER_ID),
        "GOOGLE_CREDENTIALS_present": bool(GOOGLE_CREDENTIALS),
        "PUBLIC_DOMAIN": PUBLIC_DOMAIN or "(not set)"
    }
    creds, cred_err = parse_creds()
    diag["credentials_ok"] = cred_err is None
    diag["credentials_error"] = cred_err

    if creds:
        ok, drive_err = check_drive_access(creds)
        diag["drive_access_ok"] = ok
        diag["drive_access_error"] = drive_err
    else:
        diag["drive_access_ok"] = False
        diag["drive_access_error"] = "Credentials invalid or missing"

    diag["binaries"] = check_binaries()
    diag["temp_free_disk_mb"] = free_disk_mb(tempfile.gettempdir())
    diag["suggestions"] = []

    if not diag["binaries"]["ffmpeg_found"]:
        diag["suggestions"].append("ffmpeg missing — ensure Docker installs it (apt-get install -y ffmpeg).")
    if not diag["binaries"]["yt_dlp_found"]:
        diag["suggestions"].append("yt-dlp missing — ensure requirements include yt-dlp and it installs successfully.")
    if not diag["env"]["DRIVE_FOLDER_ID_present"]:
        diag["suggestions"].append("Set DRIVE_FOLDER_ID to your target Google Drive folder ID.")
    if not diag["env"]["GOOGLE_CREDENTIALS_present"]:
        diag["suggestions"].append("Set GOOGLE_CREDENTIALS to a valid, compact (one-line) service account JSON.")
    if diag["credentials_ok"] and not diag["drive_access_ok"]:
        diag["suggestions"].append(
            "Share the target Drive folder with the service account email so it has write access."
        )
    if diag["temp_free_disk_mb"] < 2048:
        diag["suggestions"].append("Low temp disk space (<2GB). Increase ephemeral storage or reduce video size.")

    return diag

# =========================
# Helper: Titles / Paths
# =========================
def sanitize_title(title: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_\- ]', '_', title).strip()[:120]

def yt_download_to_temp(job_id: str, url: str, base_name: str) -> str:
    tmpdir = tempfile.mkdtemp(prefix="yt_ingest_")
    outtmpl = os.path.join(tmpdir, f"{base_name}.%(ext)s")

    # Make sure we have some free space
    free_mb = free_disk_mb(tmpdir)
    if free_mb < 4096:
        raise RuntimeError(f"Insufficient disk space for download/merge. Free: {free_mb} MB")

    ydl_opts = {
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "format": "bv*+ba/b",  # best video+audio, fallback best
        "restrictfilenames": True,
        "noplaylist": True,
        "postprocessors": [
            {"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}
        ],
        # Robustness for long videos:
        "retries": 10,
        "fragment_retries": 10,
        "concurrent_fragment_downloads": 5,
        "http_chunk_size": 10 * 1024 * 1024,  # 10MB
        "socket_timeout": 30,
        "nocheckcertificate": True,
        "quiet": True,
        "no_warnings": True,
    }

    job_log(job_id, "yt-dlp starting")
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        # Expected final path
        final_path = os.path.join(tmpdir, f"{base_name}.mp4")
        if os.path.exists(final_path):
            job_log(job_id, f"Download OK → {final_path}")
            return final_path

        # If not found, attempt to locate actual file
        if "requested_downloads" in info and info["requested_downloads"]:
            cand = info["requested_downloads"][0].get("filepath")
            if cand and os.path.exists(cand):
                job_log(job_id, f"Download OK (alt path) → {cand}")
                return cand

        for name in os.listdir(tmpdir):
            if name.lower().endswith(".mp4"):
                path = os.path.join(tmpdir, name)
                job_log(job_id, f"Download OK (scan) → {path}")
                return path

        raise RuntimeError("Download finished but mp4 file not found.")

def upload_to_drive_resumable(job_id: str, creds, filepath: str, filename: str) -> Dict[str, str]:
    svc = drive_service(creds)
    meta = {"name": filename, "parents": [DRIVE_FOLDER_ID]}
    media = MediaFileUpload(filepath, mimetype="video/mp4", resumable=True, chunksize=10 * 1024 * 1024)

    req = svc.files().create(body=meta, media_body=media, fields="id, webViewLink")
    response = None
    backoff = 1.0
    job_log(job_id, "Drive resumable upload starting")
    while response is None:
        try:
            status, response = req.next_chunk()
            if status:
                job_log(job_id, f"Upload progress: {int(status.progress() * 100)}%")
        except Exception as e:
            job_log(job_id, f"Transient upload error: {e}; retrying in {int(backoff)}s")
            if backoff > 60:
                raise
            time.sleep(backoff)
            backoff *= 2

    job_log(job_id, "Drive upload completed")
    return response

def notify_callback(callback_url: Optional[str], payload: Dict[str, Any]):
    if not callback_url:
        return
    try:
        requests.post(str(callback_url), json=payload, timeout=15)
    except Exception:
        # Don’t crash job on callback failures
        pass

def cleanup_temp_file(path: Optional[str]):
    if not path:
        return
    try:
        tmpdir = os.path.dirname(path)
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
        # remove rest of tmpdir
        try:
            for name in os.listdir(tmpdir):
                try:
                    os.remove(os.path.join(tmpdir, name))
                except Exception:
                    pass
            os.rmdir(tmpdir)
        except Exception:
            pass
    except Exception:
        pass

# =========================
# Background job
# =========================
def run_job(job_id: str, url: str, title: str, callback_url: Optional[str]):
    file_path = None
    result: Dict[str, Any] = {}

    # Pre-checks
    job_stage(job_id, "precheck")
    creds, cred_err = parse_creds()
    if cred_err:
        job_fail(
            job_id,
            code="CREDENTIALS_INVALID",
            message=cred_err,
            suggestion="Ensure GOOGLE_CREDENTIALS is a valid one-line JSON from Google Cloud (service account)."
        )
        notify_callback(callback_url, {"job_id": job_id, **JOBS[job_id]})
        return

    ok, drive_err = check_drive_access(creds)
    if not ok:
        job_fail(
            job_id,
            code="DRIVE_ACCESS",
            message=str(drive_err),
            suggestion=(
                "Share the target Drive folder with the service account email and confirm DRIVE_FOLDER_ID is correct."
            ),
            details={"folder_id": DRIVE_FOLDER_ID}
        )
        notify_callback(callback_url, {"job_id": job_id, **JOBS[job_id]})
        return

    bins = check_binaries()
    if not bins["ffmpeg_found"] or not bins["yt_dlp_found"]:
        job_fail(
            job_id,
            code="BINARIES_MISSING",
            message="ffmpeg and/or yt-dlp not found in container PATH.",
            suggestion="Verify Dockerfile installs ffmpeg and requirements include yt-dlp."
        )
        notify_callback(callback_url, {"job_id": job_id, **JOBS[job_id]})
        return

    # Download
    try:
        job_stage(job_id, "downloading")
        safe = sanitize_title(title)
        file_path = yt_download_to_temp(job_id, url, safe)
    except Exception as e:
        job_fail(
            job_id,
            code="DOWNLOAD_FAILED",
            message=f"yt-dlp failed: {e}",
            suggestion="Check the video URL accessibility, network, and that the video is available for download.",
        )
        notify_callback(callback_url, {"job_id": job_id, **JOBS[job_id]})
        cleanup_temp_file(file_path)
        return

    # Upload
    try:
        job_stage(job_id, "uploading")
        safe = sanitize_title(title)
        upload = upload_to_drive_resumable(job_id, creds, file_path, f"{safe}.mp4")
        result = {"drive_id": upload.get("id"), "drive_link": upload.get("webViewLink")}
    except Exception as e:
        job_fail(
            job_id,
            code="UPLOAD_FAILED",
            message=f"Drive upload failed: {e}",
            suggestion="Confirm the folder is shared with the service account and there is sufficient Drive quota.",
        )
        notify_callback(callback_url, {"job_id": job_id, **JOBS[job_id]})
        cleanup_temp_file(file_path)
        return
    finally:
        cleanup_temp_file(file_path)

    # Done
    JOBS[job_id]["status"] = "completed"
    JOBS[job_id]["result"] = result
    job_log(job_id, "Job completed successfully")
    notify_callback(callback_url, {"job_id": job_id, **JOBS[job_id]})

# =========================
# Routes
# =========================
@app.get("/health")
def health():
    return {"ok": True, "message": "App is running"}

@app.get("/diagnostics")
def diagnostics_endpoint():
    return diagnostics()

@app.get("/status/{job_id}")
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.post("/ingest", response_model=IngestResponse, status_code=202)
def ingest(payload: IngestRequest):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status": "queued",
        "title": payload.title,
        "url": str(payload.video_url),
        "logs": []  # per-job log buffer
    }
    job_log(job_id, "Job created")

    t = threading.Thread(
        target=run_job,
        args=(job_id, str(payload.video_url), payload.title, str(payload.callback_url) if payload.callback_url else None),
        daemon=True
    )
    t.start()

    status_url = f"/status/{job_id}"
    if PUBLIC_DOMAIN:
        status_url = f"https://{PUBLIC_DOMAIN.rstrip('/')}/status/{job_id}"

    return IngestResponse(job_id=job_id, status="queued", status_url=status_url)
