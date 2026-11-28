"""Persistent Audio Master handoff and sermon-timeline normalization."""
from __future__ import annotations

import io
import hmac
import json
import math
import os
import re
import sqlite3
import subprocess
import threading
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from pydantic import AwareDatetime, BaseModel, Field
from googleapiclient.errors import HttpError
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
    source_job_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    show_id: Literal["AGAPE_CHURCH"] = "AGAPE_CHURCH"
    source_type: Literal["approved_sermon"]
    source_video_drive_id: str = Field(min_length=1)
    transcript_drive_id: str = Field(min_length=1)
    transcript_timeline: Literal["original_service", "sermon_relative"]
    sermon_start_seconds: float = Field(ge=0, allow_inf_nan=False)
    sermon_end_seconds: float = Field(gt=0, allow_inf_nan=False)
    sermon_title: str = ""
    speaker: str = ""
    primary_scripture: str = ""
    service_date: str = ""
    source_youtube_video_id: str = ""
    source_youtube_url: str = ""
    destination_drive_folder_id: str = ""
    approved_at: AwareDatetime


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS source_jobs (source_job_id TEXT PRIMARY KEY, status TEXT NOT NULL, manifest_json TEXT NOT NULL, state_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
    return db


def get_job(source_job_id: str) -> dict | None:
    with _DB_LOCK, closing(connect()) as db, db:
        row = db.execute("SELECT * FROM source_jobs WHERE source_job_id=?", (source_job_id,)).fetchone()
    if not row:
        return None
    return {**dict(row), "manifest": json.loads(row["manifest_json"]), "state": json.loads(row["state_json"])}


def save_job(source_job_id: str, status: str, manifest: dict, state: dict) -> None:
    stamp = now()
    with _DB_LOCK, closing(connect()) as db, db:
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
                segment = {**item, "start": start, "end": end, "text": str(item["text"]).strip()}
                if not segment.get('words') and isinstance(value, dict) and value.get('words'):
                    from transcript_timing import normalize_words
                    words, _ = normalize_words(value['words'])
                    segment['words'] = [w for w in words if w['start'] >= start and w['end'] <= end]
                result.append(segment)
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
            normalized = {**segment, "start": round(local_start, 3), "end": round(local_end, 3), "duration": round(local_end-local_start, 3)}
            offset = 0.0 if relative else -start
            for field, text_field in [('words', 'word'), ('sentences', 'text')]:
                if segment.get(field):
                    from transcript_timing import _rows
                    evidence, _ = _rows(segment[field], text_field)
                    normalized[field] = [{**item, 'start': item['start'] + offset, 'end': item['end'] + offset}
                                         for item in evidence if item['start'] + offset >= 0 and item['end'] + offset <= end-start]
                    if len(normalized[field]) != len(evidence):
                        normalized['needs_review'] = True
                        normalized['source_' + field] = evidence
            if local_start != seg_start + offset or local_end != seg_end + offset:
                normalized.update(needs_review=True, source_segment_bounds=[seg_start, seg_end])
            selected.append(normalized)
    if not selected:
        raise RuntimeError("Existing transcript has no segments inside the approved sermon timeline")
    return selected


def video_duration(path: Path) -> float:
    completed = subprocess.run([os.getenv("FFPROBE_BINARY", "ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)], capture_output=True, text=True, timeout=120)
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {completed.stderr[-1000:]}")
    return float(completed.stdout.strip())


def retained_file(file_id: str, path: Path, metadata: dict) -> Path:
    size = int(metadata.get("size") or 0)
    if path.is_file():
        actual = path.stat().st_size
        if actual > 0 and (not size or size == actual):
            return path
    partial = path.with_suffix(path.suffix + ".partial")
    result = download_drive(file_id, partial, metadata)
    if size and result.stat().st_size != size:
        raise RuntimeError("Retained Drive file size does not match metadata")
    result.replace(path)
    return path


def process_handoff(source_job_id: str) -> None:
    with _DB_LOCK, closing(connect()) as db, db:
        claimed = db.execute(
            "UPDATE source_jobs SET status='processing' WHERE source_job_id=? AND status='accepted'",
            (source_job_id,),
        ).rowcount
    if not claimed:
        return
    job = get_job(source_job_id)
    if not job:
        return
    manifest, state = job["manifest"], job["state"]
    attempts = int(state.get("attempts", 0)) + 1
    state.update({"attempts": attempts, "stage": "drive_validation", "error": None, "started_at": now()})
    save_job(source_job_id, "processing", manifest, state)
    try:
        work = SOURCE_DIR / source_job_id
        try:
            transcript_meta = drive_metadata(manifest["transcript_drive_id"])
            if transcript_meta.get("trashed") or str(transcript_meta.get("size")) == "0":
                raise ValueError("Retained transcript is unavailable")
            transcript = retained_file(manifest["transcript_drive_id"], work / "source-transcript", transcript_meta)
            original = parse_transcript(transcript)
            if any(not math.isfinite(item["start"]) or not math.isfinite(item["end"]) for item in original):
                raise ValueError("Transcript includes nonfinite timings")
            segments = normalize_sermon(original, manifest)
        except HttpError as exc:
            if exc.resp.status not in {404, 410}:
                raise
            save_job(source_job_id, "held_transcript", manifest, {**state, "stage": "held_transcript", "retryable": False})
            return
        except (ValueError, RuntimeError, UnicodeError):
            save_job(source_job_id, "held_transcript", manifest, {**state, "stage": "held_transcript", "retryable": False})
            return
        video_meta = drive_metadata(manifest["source_video_drive_id"])
        if video_meta.get("trashed"):
            raise RuntimeError("Retained video is trashed in Google Drive")
        video = retained_file(manifest["source_video_drive_id"], work / "approved-sermon.mp4", video_meta)
        duration = video_duration(video)
        tolerance = float(os.getenv("TIMELINE_TOLERANCE_SECONDS", "5"))
        if not math.isfinite(duration) or abs(duration - (manifest["sermon_end_seconds"] - manifest["sermon_start_seconds"])) > tolerance:
            raise RuntimeError("Retained video duration does not match the approved range")
        if max(segment["end"] for segment in segments) > duration + tolerance:
            save_job(source_job_id, "held_transcript", manifest, {**state, "stage": "held_transcript", "retryable": False})
            return
        normalized_path = work / "sermon-relative-transcript.json"
        partial = normalized_path.with_suffix(".partial")
        partial.write_text(json.dumps({"timeline": "sermon_relative", "video_duration": duration, "segments": segments}, indent=2), encoding="utf-8")
        partial.replace(normalized_path)
        state.update({"stage": "ready_for_selection", "video_path": str(video), "normalized_transcript_path": str(normalized_path), "video_duration": duration, "segment_count": len(segments), "completed_at": now()})
        save_job(source_job_id, "ready_for_selection", manifest, state)
    except Exception as exc:
        state.update({"stage": "error", "error_type": type(exc).__name__, "error": str(exc), "retryable": True, "failed_at": now()})
        save_job(source_job_id, "error", manifest, state)


def require_secret(value: str | None) -> None:
    expected = os.getenv("AUDIO_MASTER_HANDOFF_SECRET", "").strip()
    if not expected or not hmac.compare_digest((value or "").encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Invalid handoff secret")


@router.on_event("startup")
def resume_handoffs() -> None:
    if not os.getenv("AUDIO_MASTER_HANDOFF_SECRET", "").strip():
        return
    # This volume-backed deployment runs one worker. Resume only rows admitted
    # by the current contract, leaving legacy records for operator review.
    with _DB_LOCK, closing(connect()) as db, db:
        db.execute("UPDATE source_jobs SET status='accepted' WHERE status='processing' AND json_extract(state_json, '$.managed')=1")
        rows = db.execute("SELECT source_job_id FROM source_jobs WHERE status='accepted' AND json_extract(state_json, '$.managed')=1").fetchall()
    if rows:
        def resume():
            for row in rows:
                process_handoff(row["source_job_id"])
        threading.Thread(target=resume, daemon=True).start()


@router.post("/api/audio-master/handoff", status_code=202)
def handoff(payload: Handoff, background_tasks: BackgroundTasks, x_audio_master_secret: str | None = Header(None)):
    require_secret(x_audio_master_secret)
    if payload.sermon_end_seconds <= payload.sermon_start_seconds:
        raise HTTPException(status_code=422, detail="Invalid approved range")
    manifest = payload.model_dump(mode="json")
    stamp = now()
    with _DB_LOCK, closing(connect()) as db, db:
        added = db.execute(
            "INSERT INTO source_jobs VALUES (?,?,?,?,?,?) ON CONFLICT(source_job_id) DO NOTHING",
            (payload.source_job_id, "accepted", json.dumps(manifest),
             json.dumps({"stage": "accepted", "attempts": 0, "managed": True}), stamp, stamp),
        ).rowcount
    if not added:
        existing = get_job(payload.source_job_id)
        if existing["manifest"] != manifest:
            raise HTTPException(status_code=409, detail="Approved manifest differs for existing source job")
        if existing["status"] == "accepted":
            background_tasks.add_task(process_handoff, payload.source_job_id)
        return {"status": "duplicate", "source_job_id": payload.source_job_id, "processing_status": existing["status"]}
    save_job(payload.source_job_id, "accepted", manifest, {"stage": "accepted", "attempts": 0})
    background_tasks.add_task(process_handoff, payload.source_job_id)
    return {"status": "accepted", "source_job_id": payload.source_job_id}


@router.get("/api/source-jobs/{source_job_id}")
def status(source_job_id: str, x_audio_master_secret: str | None = Header(None)):
    require_secret(x_audio_master_secret)
    job = get_job(source_job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Source job not found")
    return {"source_job_id": source_job_id, "status": job["status"]}


@router.post("/api/source-jobs/{source_job_id}/retry", status_code=202)
def retry(source_job_id: str, background_tasks: BackgroundTasks, x_audio_master_secret: str | None = Header(None)):
    require_secret(x_audio_master_secret)
    job = get_job(source_job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Source job not found")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", source_job_id):
        raise HTTPException(status_code=409, detail="Legacy source identity requires review")
    with _DB_LOCK, closing(connect()) as db, db:
        changed = db.execute("UPDATE source_jobs SET status='accepted' WHERE source_job_id=? AND status='error' AND json_extract(state_json, '$.managed')=1", (source_job_id,)).rowcount
    if not changed:
        raise HTTPException(status_code=409, detail="Source job is not eligible for retry")
    background_tasks.add_task(process_handoff, source_job_id)
    return {"status": "retry_accepted", "source_job_id": source_job_id}

