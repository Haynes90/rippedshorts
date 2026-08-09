"""Persistent Audio Master handoff and sermon-timeline normalization."""
from __future__ import annotations

import io
import json
import os
import re
import sqlite3
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from pydantic import BaseModel, Field
from googleapiclient.http import MediaIoBaseDownload

from google_drive import drive_service

router = APIRouter()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DB_PATH = DATA_DIR / "ripped_shorts.db"
SOURCE_DIR = DATA_DIR / "sources"
_DB_LOCK = threading.RLock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Handoff(BaseModel):
    source_job_id: str = Field(min_length=1)
    show_id: str = "AGAPE_CHURCH"
    source_type: Literal["approved_sermon"]
    source_video_drive_id: str = Field(min_length=1)
    transcript_drive_id: str = Field(min_length=1)
    transcript_timeline: Literal["original_service", "sermon_relative"]
    sermon_start_seconds: float = Field(ge=0)
    sermon_end_seconds: float = Field(gt=0)
    sermon_title: str = ""
    speaker: str = ""
    primary_scripture: str = ""
    service_date: str = ""
    source_youtube_video_id: str = ""
    source_youtube_url: str = ""
    destination_drive_folder_id: str = ""


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS source_jobs (source_job_id TEXT PRIMARY KEY, status TEXT NOT NULL, manifest_json TEXT NOT NULL, state_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
    return db


def get_job(source_job_id: str) -> dict | None:
    with _DB_LOCK, connect() as db:
        row = db.execute("SELECT * FROM source_jobs WHERE source_job_id=?", (source_job_id,)).fetchone()
    if not row:
        return None
    return {**dict(row), "manifest": json.loads(row["manifest_json"]), "state": json.loads(row["state_json"])}


def save_job(source_job_id: str, status: str, manifest: dict, state: dict) -> None:
    stamp = now()
    with _DB_LOCK, connect() as db:
        db.execute(
            "INSERT INTO source_jobs VALUES (?,?,?,?,?,?) ON CONFLICT(source_job_id) DO UPDATE SET status=excluded.status, manifest_json=excluded.manifest_json, state_json=excluded.state_json, updated_at=excluded.updated_at",
            (source_job_id, status, json.dumps(manifest), json.dumps(state), stamp, stamp),
        )


def drive_metadata(file_id: str) -> dict:
    return drive_service().files().get(fileId=file_id, fields="id,name,size,mimeType,trashed", supportsAllDrives=True).execute()


def download_drive(file_id: str, destination: Path, metadata: dict) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    service = drive_service()
    if str(metadata.get("mimeType", "")).startswith("application/vnd.google-apps"):
        request = service.files().export_media(fileId=file_id, mimeType="text/plain")
    else:
        request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with destination.open("wb") as handle:
        downloader = MediaIoBaseDownload(handle, request, chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    if not destination.exists() or destination.stat().st_size <= 0:
        raise RuntimeError(f"Drive file {file_id} downloaded empty")
    return destination


STAMP = re.compile(r"\[(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?\s*-\s*(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?\]\s*(.*)")


def _seconds(h: str, m: str, s: str, ms: str | None) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + (int((ms or "0").ljust(3, "0")) / 1000)


def parse_transcript(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        value = json.loads(raw)
        items = value.get("segments") if isinstance(value, dict) else value
        if isinstance(items, list):
            result = []
            for item in items:
                if not isinstance(item, dict) or not str(item.get("text", "")).strip():
                    continue
                start = float(item.get("start", item.get("start_seconds", 0)))
                end = float(item.get("end", item.get("end_seconds", start + float(item.get("duration", 0)))))
                result.append({"start": start, "end": end, "text": str(item["text"]).strip()})
            if result:
                return result
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    result = []
    for line in raw.splitlines():
        match = STAMP.match(line.strip())
        if match:
            result.append({"start": _seconds(*match.groups()[0:4]), "end": _seconds(*match.groups()[4:8]), "text": match.group(9).strip()})
    if not result:
        raise RuntimeError("Transcript contains no parseable timed segments")
    return result


def normalize_sermon(segments: list[dict], manifest: dict) -> list[dict]:
    start = float(manifest["sermon_start_seconds"])
    end = float(manifest["sermon_end_seconds"])
    if end <= start:
        raise RuntimeError("sermon_end_seconds must be greater than sermon_start_seconds")
    relative = manifest["transcript_timeline"] == "sermon_relative"
    selected = []
    for segment in segments:
        seg_start, seg_end = float(segment["start"]), float(segment["end"])
        if not relative and (seg_end < start or seg_start > end):
            continue
        local_start = seg_start if relative else seg_start - start
        local_end = seg_end if relative else seg_end - start
        local_start = max(0.0, local_start)
        local_end = min(end - start, local_end)
        if local_end > local_start:
            selected.append({"start": round(local_start, 3), "end": round(local_end, 3), "duration": round(local_end-local_start, 3), "text": segment["text"]})
    if not selected:
        raise RuntimeError("Existing transcript has no segments inside the approved sermon timeline")
    return selected


def video_duration(path: Path) -> float:
    completed = subprocess.run([os.getenv("FFPROBE_BINARY", "ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)], capture_output=True, text=True, timeout=120)
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {completed.stderr[-1000:]}")
    return float(completed.stdout.strip())


def process_handoff(source_job_id: str) -> None:
    job = get_job(source_job_id)
    if not job:
        return
    manifest, state = job["manifest"], job["state"]
    attempts = int(state.get("attempts", 0)) + 1
    state.update({"attempts": attempts, "stage": "drive_validation", "error": None, "started_at": now()})
    save_job(source_job_id, "processing", manifest, state)
    try:
        video_meta = drive_metadata(manifest["source_video_drive_id"])
        transcript_meta = drive_metadata(manifest["transcript_drive_id"])
        if video_meta.get("trashed") or transcript_meta.get("trashed"):
            raise RuntimeError("One or more handoff files are trashed in Google Drive")
        work = SOURCE_DIR / source_job_id
        video = download_drive(manifest["source_video_drive_id"], work / "approved-sermon.mp4", video_meta)
        transcript = download_drive(manifest["transcript_drive_id"], work / "source-transcript", transcript_meta)
        state["stage"] = "timeline_validation"
        segments = normalize_sermon(parse_transcript(transcript), manifest)
        duration = video_duration(video)
        tolerance = float(os.getenv("TIMELINE_TOLERANCE_SECONDS", "5"))
        if segments[-1]["end"] > duration + tolerance:
            raise RuntimeError(f"Transcript ends at {segments[-1]['end']:.3f}s but video is {duration:.3f}s")
        normalized_path = work / "sermon-relative-transcript.json"
        normalized_path.write_text(json.dumps({"timeline": "sermon_relative", "video_duration": duration, "segments": segments}, indent=2), encoding="utf-8")
        state.update({"stage": "ready_for_selection", "video_path": str(video), "normalized_transcript_path": str(normalized_path), "video_duration": duration, "segment_count": len(segments), "completed_at": now()})
        save_job(source_job_id, "ready_for_selection", manifest, state)
    except Exception as exc:
        state.update({"stage": "error", "error_type": type(exc).__name__, "error": str(exc), "retryable": True, "failed_at": now()})
        save_job(source_job_id, "error", manifest, state)


@router.post("/api/audio-master/handoff", status_code=202)
def handoff(payload: Handoff, background_tasks: BackgroundTasks, x_audio_master_secret: str | None = Header(None)):
    if not os.getenv("AUDIO_MASTER_HANDOFF_SECRET") or x_audio_master_secret != os.getenv("AUDIO_MASTER_HANDOFF_SECRET"):
        raise HTTPException(status_code=401, detail="Invalid handoff secret")
    manifest = payload.model_dump()
    existing = get_job(payload.source_job_id)
    if existing:
        return {"status": "duplicate", "source_job_id": payload.source_job_id, "processing_status": existing["status"]}
    save_job(payload.source_job_id, "accepted", manifest, {"stage": "accepted", "attempts": 0})
    background_tasks.add_task(process_handoff, payload.source_job_id)
    return {"status": "accepted", "source_job_id": payload.source_job_id}


@router.get("/api/source-jobs/{source_job_id}")
def status(source_job_id: str):
    job = get_job(source_job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Source job not found")
    return job


@router.post("/api/source-jobs/{source_job_id}/retry", status_code=202)
def retry(source_job_id: str, background_tasks: BackgroundTasks, x_audio_master_secret: str | None = Header(None)):
    if x_audio_master_secret != os.getenv("AUDIO_MASTER_HANDOFF_SECRET"):
        raise HTTPException(status_code=401, detail="Invalid handoff secret")
    job = get_job(source_job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Source job not found")
    background_tasks.add_task(process_handoff, source_job_id)
    return {"status": "retry_accepted", "source_job_id": source_job_id}

