"""Resilient YouTube ingestion and Google Drive reuse keyed by YouTube ID."""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from googleapiclient.http import MediaFileUpload
from yt_dlp import YoutubeDL
import requests

from google_drive import docs_service, download_drive_file, drive_service

TIMED_LINE = re.compile(
    r"^\[(?P<clock>[^\]|]+)(?:\s*\|\s*start=(?P<start>[0-9.]+)s\s*\|\s*dur=(?P<duration>[0-9.]+)s)?\]\s*(?P<text>.+)$"
)


def _clock_seconds(value: str) -> float:
    parts = [float(item) for item in value.strip().split(":")]
    total = 0.0
    for item in parts:
        total = total * 60 + item
    return total


def parse_timed_transcript(text: str) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for raw in text.splitlines():
        match = TIMED_LINE.match(raw.strip())
        if not match:
            continue
        start = float(match.group("start") or _clock_seconds(match.group("clock")))
        duration = float(match.group("duration") or 0.0)
        segments.append({"start": start, "duration": duration, "text": match.group("text").strip()})
    for index, item in enumerate(segments[:-1]):
        if item["duration"] <= 0:
            item["duration"] = max(0.01, segments[index + 1]["start"] - item["start"])
    if segments and segments[-1]["duration"] <= 0:
        segments[-1]["duration"] = 2.0
    return segments


def _folder_query(video_id: str) -> str:
    safe = video_id.replace("'", "\\'")
    query = f"name contains '{safe}' and trashed = false"
    folder = (os.getenv("DRIVE_FOLDER_ID") or os.getenv("Drive_Folder_ID") or "").strip()
    if folder:
        query += f" and '{folder}' in parents"
    return query


def find_drive_assets(video_id: str) -> list[dict[str, Any]]:
    """Search the configured folder, then all Drive files visible to the service account."""
    service = drive_service()
    fields = "files(id,name,mimeType,size,webViewLink,webContentLink,modifiedTime)"
    result = service.files().list(
        q=_folder_query(video_id),
        fields=fields,
        orderBy="modifiedTime desc",
        pageSize=100,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = list(result.get("files", []))

    # Audio Master and Ripped Shorts may be configured with different destination
    # folders while sharing the same service account. Do a global ID lookup too.
    folder = (os.getenv("DRIVE_FOLDER_ID") or os.getenv("Drive_Folder_ID") or "").strip()
    if folder:
        safe = video_id.replace("'", "\\'")
        global_result = service.files().list(
            q=f"name contains '{safe}' and trashed = false",
            fields=fields,
            orderBy="modifiedTime desc",
            pageSize=100,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        seen = {item.get("id") for item in files}
        files.extend(item for item in global_result.get("files", []) if item.get("id") not in seen)
    return files


def _read_doc(file_id: str) -> str:
    doc = docs_service().documents().get(documentId=file_id).execute()
    chunks: list[str] = []
    for item in doc.get("body", {}).get("content", []):
        for element in item.get("paragraph", {}).get("elements", []):
            run = element.get("textRun") or {}
            chunks.append(run.get("content", ""))
    return "".join(chunks)


def _read_json_asset(asset: dict[str, Any], workdir: Path) -> Any:
    target = workdir / asset["name"]
    download_drive_file(asset["id"], target)
    return json.loads(target.read_text(encoding="utf-8"))


def _segments_from_json(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("segments", "timed_transcript", "transcript"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        return []
    segments = []
    for item in payload:
        if not isinstance(item, dict) or not str(item.get("text") or "").strip():
            continue
        start = float(item.get("start", item.get("start_seconds", 0)) or 0)
        duration = item.get("duration")
        if duration is None and item.get("end") is not None:
            duration = float(item["end"]) - start
        if duration is None and item.get("end_seconds") is not None:
            duration = float(item["end_seconds"]) - start
        segments.append({"start": start, "duration": max(0.01, float(duration or 0.01)), "text": str(item["text"]).strip()})
    return segments


def _sermon_boundary(payload: Any) -> dict[str, float] | None:
    if not isinstance(payload, dict):
        return None
    candidates = payload.get("chapters")
    if not isinstance(candidates, list):
        analysis = payload.get("analysis") or payload.get("final_approved_analysis") or {}
        candidates = analysis.get("chapters") if isinstance(analysis, dict) else None
    if not isinstance(candidates, list):
        return None
    for chapter in candidates:
        if not isinstance(chapter, dict):
            continue
        label = " ".join(str(chapter.get(key) or "") for key in ("chapter_type", "display_title", "title", "name")).lower()
        if "sermon" not in label:
            continue
        start = float(chapter.get("start_seconds", chapter.get("start", 0)) or 0)
        end = chapter.get("end_seconds", chapter.get("end"))
        if end is not None and float(end) > start:
            return {"start": start, "end": float(end)}
    return None


def reuse_from_drive(video_id: str, workdir: Path) -> dict[str, Any]:
    """Find source video, timed transcript, and approved sermon bounds by ID."""
    assets = find_drive_assets(video_id)
    video_asset = next(
        (item for item in assets if str(item.get("mimeType", "")).startswith("video/")),
        None,
    )
    transcript_segments: list[dict[str, Any]] = []
    boundary = None

    json_assets = [
        item for item in assets
        if item.get("mimeType") == "application/json" or str(item.get("name", "")).lower().endswith(".json")
    ]
    for asset in json_assets:
        try:
            payload = _read_json_asset(asset, workdir)
            transcript_segments = transcript_segments or _segments_from_json(payload)
            boundary = boundary or _sermon_boundary(payload)
        except Exception:
            continue

    if not transcript_segments:
        text_asset = next(
            (
                item for item in assets
                if item.get("mimeType") == "application/vnd.google-apps.document"
                and "transcript" in str(item.get("name", "")).lower()
            ),
            None,
        )
        if text_asset:
            transcript_segments = parse_timed_transcript(_read_doc(text_asset["id"]))

    if not transcript_segments:
        plain_asset = next(
            (item for item in assets if "transcript" in str(item.get("name", "")).lower() and str(item.get("name", "")).lower().endswith(".txt")),
            None,
        )
        if plain_asset:
            target = download_drive_file(plain_asset["id"], workdir / plain_asset["name"])
            transcript_segments = parse_timed_transcript(target.read_text(encoding="utf-8"))

    video_path = None
    if video_asset:
        suffix = Path(video_asset["name"]).suffix or ".mp4"
        video_path = download_drive_file(video_asset["id"], workdir / f"{video_id}-source{suffix}")

    return {
        "assets": assets,
        "video_path": video_path,
        "segments": transcript_segments,
        "sermon_boundary": boundary,
        "reused_video": bool(video_path),
        "reused_transcript": bool(transcript_segments),
    }



def ingest_with_audio_master(video_id: str, youtube_url: str) -> dict[str, Any]:
    """Run Audio Master's cache/download/transcription path and wait for Drive assets."""
    base = (os.getenv("AUDIO_MASTER_INTERNAL_URL") or "").strip().rstrip("/")
    if base and "://" not in base:
        base = f"https://{base}"
    for suffix in ("/api/ripped-shorts/ingest", "/api/ripped-shorts"):
        if base.endswith(suffix):
            base = base[: -len(suffix)].rstrip("/")
            break
    secret = (
        os.getenv("AUDIO_MASTER_INGEST_SECRET")
        or os.getenv("AUDIO_MASTER_WEBHOOK_SECRET")
        or ""
    ).strip()
    if not base or not secret:
        raise RuntimeError(
            "Audio Master ingestion is not configured. Set AUDIO_MASTER_INTERNAL_URL "
            "and AUDIO_MASTER_INGEST_SECRET in Ripped Shorts."
        )
    headers = {"x-ripped-shorts-ingest-secret": secret}
    response = requests.post(
        f"{base}/api/ripped-shorts/ingest",
        json={"video_id": video_id, "youtube_url": youtube_url, "title": video_id},
        headers=headers,
        timeout=(10, 60),
    )
    if response.status_code not in {200, 202}:
        raise RuntimeError(f"Audio Master ingest failed ({response.status_code}): {response.text[:1000]}")
    accepted = response.json()
    job_id = str(accepted.get("job_id") or "")
    if not job_id:
        raise RuntimeError(f"Audio Master ingest returned no job_id: {accepted}")

    timeout_seconds = max(60, int(os.getenv("AUDIO_MASTER_INGEST_TIMEOUT_SECONDS", "5400")))
    poll_seconds = max(3, int(os.getenv("AUDIO_MASTER_INGEST_POLL_SECONDS", "15")))
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = accepted
    while time.monotonic() < deadline:
        status_response = requests.get(
            f"{base}/api/ripped-shorts/ingest/{job_id}",
            headers=headers,
            timeout=(10, 60),
        )
        if status_response.status_code != 200:
            raise RuntimeError(
                f"Audio Master status failed ({status_response.status_code}): {status_response.text[:1000]}"
            )
        last = status_response.json()
        status = str(last.get("status") or "").lower()
        if status in {"failed", "error", "download_failed", "transcription_failed"}:
            raise RuntimeError(f"Audio Master ingestion failed: {last.get('error') or last}")
        transcript = last.get("transcript") or {}
        source_video = last.get("source_video") or {}
        source_status = str(source_video.get("status") or "").lower()
        if source_status in {"failed", "awaiting_route_rerun"}:
            raise RuntimeError(
                "Audio Master's retained source-video path failed: "
                + str(last.get("error") or source_video)
            )
        transcript_ready = bool(transcript.get("drive_files"))
        source_ready = bool(source_video.get("drive_file_id")) or source_status == "ready"
        if transcript_ready and source_ready:
            return last
        time.sleep(poll_seconds)
    raise RuntimeError(
        f"Audio Master ingestion timed out after {timeout_seconds}s; "
        f"last status={last.get('status')}, job_id={job_id}"
    )

def download_youtube_resilient(video_id: str, youtube_url: str, workdir: Path) -> Path:
    """Use Audio Master's yt-dlp fallback pattern instead of a paid download API."""
    workdir.mkdir(parents=True, exist_ok=True)
    destination = workdir / f"{video_id}-source.mp4"
    cookie_file = (
        os.getenv("YTDLP_COOKIE_FILE")
        or os.getenv("YTDLP_COOKIES_FILE")
        or os.getenv("YOUTUBE_COOKIES_FILE")
        or os.getenv("YOUTUBE_COOKIE_FILE")
        or os.getenv("YT_DLP_COOKIE_FILE")
        or ""
    ).strip()
    generated_cookie_file = workdir / "youtube-cookies.txt"
    cookies_base64 = (os.getenv("YOUTUBE_COOKIES_BASE64") or "").strip()
    cookie_text = (os.getenv("YOUTUBE_COOKIES") or "").strip()

    # Railway variables cannot create a filesystem path by themselves. Prefer
    # the portable base64 secret whenever the configured path is unavailable.
    if cookies_base64 and (not cookie_file or not Path(cookie_file).is_file()):
        try:
            decoded = base64.b64decode(cookies_base64, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError) as exc:
            raise RuntimeError("YOUTUBE_COOKIES_BASE64 is not valid base64-encoded UTF-8 cookie text") from exc
        generated_cookie_file.write_text(decoded.rstrip() + "\n", encoding="utf-8")
        cookie_file = str(generated_cookie_file)
    elif cookie_text and (not cookie_file or not Path(cookie_file).is_file()):
        generated_cookie_file.write_text(cookie_text.replace("\\n", "\n").rstrip() + "\n", encoding="utf-8")
        cookie_file = str(generated_cookie_file)
    options: dict[str, Any] = {
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
        "outtmpl": str(destination),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "retries": 5,
        "fragment_retries": 5,
        "concurrent_fragment_downloads": 4,
        "socket_timeout": 30,
        "quiet": True,
        "no_warnings": True,
        "http_headers": {"User-Agent": "Mozilla/5.0"},
        "extractor_args": {"youtube": {"player_client": ["android_vr", "web_safari", "web"]}},
    }
    if cookie_file and Path(cookie_file).is_file():
        options["cookiefile"] = cookie_file
    with YoutubeDL(options) as ydl:
        ydl.extract_info(youtube_url, download=True)
    if not destination.is_file() or destination.stat().st_size <= 0:
        matches = sorted(workdir.glob(f"{video_id}-source.*"))
        if matches:
            destination = matches[0]
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise RuntimeError("yt-dlp completed without a usable source video")
    return destination


def upload_cache_file(path: Path, name: str, mime_type: str) -> dict[str, Any]:
    folder = (os.getenv("DRIVE_FOLDER_ID") or os.getenv("Drive_Folder_ID") or "").strip()
    metadata: dict[str, Any] = {"name": name}
    if folder:
        metadata["parents"] = [folder]
    result = drive_service().files().create(
        body=metadata,
        media_body=MediaFileUpload(str(path), mimetype=mime_type, resumable=True),
        fields="id,name,mimeType,size,webViewLink",
        supportsAllDrives=True,
    ).execute()
    return result


def persist_source_cache(video_id: str, video_path: Path, segments: list[dict[str, Any]], workdir: Path) -> dict[str, Any]:
    existing = find_drive_assets(video_id)
    names = {str(item.get("name") or "") for item in existing}
    uploaded: list[str] = []
    video_name = f"{video_id}-source.mp4"
    if video_name not in names:
        upload_cache_file(video_path, video_name, "video/mp4")
        uploaded.append(video_name)
    transcript_name = f"{video_id}_segments.json"
    if transcript_name not in names:
        payload_path = workdir / transcript_name
        payload_path.write_text(json.dumps({"video_id": video_id, "segments": segments}, ensure_ascii=False, indent=2), encoding="utf-8")
        upload_cache_file(payload_path, transcript_name, "application/json")
        uploaded.append(transcript_name)
    return {"uploaded": uploaded, "existing_count": len(existing)}


def restrict_to_boundary(segments: list[dict[str, Any]], boundary: dict[str, float] | None) -> list[dict[str, Any]]:
    if not boundary:
        return segments
    start, end = float(boundary["start"]), float(boundary["end"])
    return [
        item for item in segments
        if float(item["start"]) >= start and float(item["start"]) + float(item.get("duration", 0)) <= end
    ]


def select_non_overlapping(clips: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    ranked = sorted(clips, key=lambda item: float(item.get("score", 0)), reverse=True)
    chosen: list[dict[str, Any]] = []
    for clip in ranked:
        start = float(clip.get("start", 0))
        end = float(clip.get("end", start + float(clip.get("duration", 0))))
        if end <= start:
            continue
        if any(start < float(item["end"]) and end > float(item["start"]) for item in chosen):
            continue
        clip["start"], clip["end"], clip["duration"] = start, end, end - start
        chosen.append(clip)
        if len(chosen) >= limit:
            break
    return sorted(chosen, key=lambda item: float(item["start"]))
