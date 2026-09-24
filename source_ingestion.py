"""Resilient YouTube ingestion and Google Drive reuse keyed by YouTube ID."""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import subprocess
import time
import shutil
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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



def _has_audio_stream(path: Path) -> bool:
    """Reject cached/downloaded video-only files before transcription or rendering."""
    try:
        result = subprocess.run(
            [os.getenv("FFPROBE_BINARY", "ffprobe"), "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type", "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        return result.returncode == 0 and "audio" in result.stdout.lower()
    except (OSError, subprocess.SubprocessError):
        return False

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



def ingest_with_audio_master(video_id: str, youtube_url: str, progress=None) -> dict[str, Any]:
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

    timeout_seconds = max(1200, int(os.getenv("AUDIO_MASTER_INGEST_TIMEOUT_SECONDS", "5400")))
    poll_seconds = max(3, int(os.getenv("AUDIO_MASTER_INGEST_POLL_SECONDS", "15")))
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = accepted
    last_progress_signature = None
    if progress:
        progress("🎧 Audio Master accepted the source; preparing reusable media.")
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
        transcription_progress = last.get("transcription_progress") or {}
        progress_signature = (
            status,
            transcription_progress.get("stage"),
            transcription_progress.get("completed_chunks"),
            transcription_progress.get("total_chunks"),
            source_status,
        )
        if progress and progress_signature != last_progress_signature:
            completed = transcription_progress.get("completed_chunks")
            total = transcription_progress.get("total_chunks")
            percent = transcription_progress.get("percent")
            if completed is not None and total:
                progress(
                    f"📝 Audio Master transcription: {completed}/{total} chunk(s) "
                    f"complete ({percent or 0}%)."
                )
            elif status:
                progress(f"🎧 Audio Master stage: {status}.")
            last_progress_signature = progress_signature
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

def _run_rapidapi_profile(
    *,
    video_id: str,
    workdir: Path,
    stop_event: threading.Event,
) -> Path:
    """Request the configured RapidAPI downloader and poll its returned file URL."""
    api_key = (os.getenv("RAPIDAPI_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("RAPIDAPI_KEY not configured")

    host = (
        os.getenv("YOUTUBE_DL_HOST")
        or "youtube-video-fast-downloader-24-7.p.rapidapi.com"
    ).strip()
    path_template = (
        os.getenv("YOUTUBE_DL_PATH_TEMPLATE")
        or "/download_video/{video_id}"
    ).strip()
    quality = (os.getenv("YOUTUBE_DL_QUALITY") or "247").strip()
    request_url = f"https://{host}{path_template.format(video_id=video_id)}"
    headers = {
        "x-rapidapi-host": host,
        "x-rapidapi-key": api_key,
    }

    lane = workdir / ".youtube-lane-rapidapi_video_download"
    shutil.rmtree(lane, ignore_errors=True)
    lane.mkdir(parents=True, exist_ok=True)
    target = lane / f"{video_id}-source.mp4"

    print(
        f"RIPPED_SOURCE_PROFILE start video_id={video_id} "
        "profile=rapidapi_video_download provider=rapidapi",
        flush=True,
    )

    try:
        response = requests.get(
            request_url,
            headers=headers,
            params={"quality": quality},
            timeout=(10, 60),
            allow_redirects=True,
        )
        response.raise_for_status()

        content_type = (response.headers.get("content-type") or "").lower()
        if (
            "video/" in content_type
            or "application/octet-stream" in content_type
        ):
            target.write_bytes(response.content)
            if target.stat().st_size <= 0:
                raise RuntimeError("RapidAPI returned an empty video response")
            return target

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"RapidAPI downloader returned unexpected content-type={content_type!r}"
            ) from exc

        file_url = ""
        if isinstance(payload, dict):
            file_url = str(
                payload.get("file")
                or payload.get("download_url")
                or payload.get("video_url")
                or payload.get("url")
                or ""
            ).strip()
            if not file_url and isinstance(payload.get("data"), dict):
                data = payload["data"]
                file_url = str(
                    data.get("file")
                    or data.get("download_url")
                    or data.get("video_url")
                    or data.get("url")
                    or ""
                ).strip()
        if not file_url:
            raise RuntimeError(f"RapidAPI downloader returned no file URL: {payload}")

        deadline = time.monotonic() + max(
            30, int(os.getenv("YOUTUBE_DL_READY_TIMEOUT_SECONDS", "360"))
        )
        poll_seconds = max(
            2, int(os.getenv("YOUTUBE_DL_READY_POLL_SECONDS", "10"))
        )
        last_status = None
        while time.monotonic() < deadline:
            if stop_event.is_set():
                raise RuntimeError("cancelled because another source provider won")
            file_response = requests.get(
                file_url,
                timeout=(10, 120),
                stream=True,
                allow_redirects=True,
            )
            last_status = file_response.status_code
            if file_response.status_code == 404:
                file_response.close()
                time.sleep(poll_seconds)
                continue
            file_response.raise_for_status()
            download_type = (file_response.headers.get("content-type") or "").lower()
            if "text/html" in download_type or "application/json" in download_type:
                body = file_response.text[:500]
                file_response.close()
                raise RuntimeError(
                    f"RapidAPI file URL returned non-video content "
                    f"type={download_type!r}: {body}"
                )
            with target.open("wb") as handle:
                for chunk in file_response.iter_content(chunk_size=1024 * 1024):
                    if stop_event.is_set():
                        raise RuntimeError("cancelled because another source provider won")
                    if chunk:
                        handle.write(chunk)
            file_response.close()
            if target.is_file() and target.stat().st_size > 0:
                print(
                    f"RIPPED_SOURCE_PROFILE success video_id={video_id} "
                    f"profile=rapidapi_video_download provider=rapidapi "
                    f"bytes={target.stat().st_size}",
                    flush=True,
                )
                return target
            time.sleep(poll_seconds)

        raise RuntimeError(
            f"RapidAPI file was not ready before timeout; last_status={last_status}"
        )
    except Exception as exc:
        print(
            f"RIPPED_SOURCE_PROFILE failed video_id={video_id} "
            f"profile=rapidapi_video_download provider=rapidapi "
            f"error={type(exc).__name__}: {exc}",
            flush=True,
        )
        raise


def _run_source_profile(
    *,
    profile: dict[str, Any],
    video_id: str,
    youtube_url: str,
    workdir: Path,
    cookie_file: str,
    pot_home: str,
    stop_event: threading.Event,
) -> Path:
    if profile.get("provider") == "rapidapi":
        return _run_rapidapi_profile(
            video_id=video_id,
            workdir=workdir,
            stop_event=stop_event,
        )
    return _run_youtube_profile(
        profile=profile,
        video_id=video_id,
        youtube_url=youtube_url,
        workdir=workdir,
        cookie_file=cookie_file,
        pot_home=pot_home,
        stop_event=stop_event,
    )


def _log_source_winner(
    *,
    video_id: str,
    profile: dict[str, Any],
    path: Path,
    workdir: Path,
) -> None:
    provider = str(profile.get("provider") or "yt-dlp")
    record = {
        "event": "source_acquisition_winner",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "video_id": video_id,
        "provider": provider,
        "profile": profile.get("name"),
        "bytes": path.stat().st_size if path.exists() else 0,
    }
    print(
        "RIPPED_SOURCE_WINNER "
        + " ".join(f"{key}={value}" for key, value in record.items()),
        flush=True,
    )
    spreadsheet_id = (
        os.getenv("PODCAST_SHEET_ID")
        or "14VruBxjaaE9DyPSdBidMeuPHew3nHUC5sRNGXtLCsis"
    ).strip()
    if spreadsheet_id:
        try:
            import workflow_reliability
            workflow_reliability.record_source_winner(
                spreadsheet_id=spreadsheet_id,
                video_id=video_id,
                provider=provider,
                profile=str(profile.get("name") or ""),
                source_bytes=int(record["bytes"] or 0),
            )
        except Exception as exc:
            print(
                f"RIPPED_SOURCE_WINNER_SHEET failed video_id={video_id} error={exc}",
                flush=True,
            )


def _run_youtube_profile(
    *,
    profile: dict[str, Any],
    video_id: str,
    youtube_url: str,
    workdir: Path,
    cookie_file: str,
    pot_home: str,
    stop_event: threading.Event,
) -> Path:
    """Run one isolated yt-dlp profile so parallel attempts cannot collide."""
    lane = workdir / f".youtube-lane-{profile['name']}"
    shutil.rmtree(lane, ignore_errors=True)
    lane.mkdir(parents=True, exist_ok=True)
    lane_output = lane / f"{video_id}-source.%(ext)s"

    def _cancel_if_winner_exists(status: dict[str, Any]) -> None:
        if stop_event.is_set():
            raise RuntimeError("cancelled because another YouTube acquisition profile won")

    options: dict[str, Any] = {
        "outtmpl": str(lane_output),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "retries": 5,
        "fragment_retries": 5,
        "extractor_retries": 3,
        "concurrent_fragment_downloads": 4,
        "socket_timeout": 45,
        "quiet": True,
        "no_warnings": True,
        "http_headers": {"User-Agent": "Mozilla/5.0"},
        "progress_hooks": [_cancel_if_winner_exists],
    }
    if profile.get("format"):
        options["format"] = profile["format"]
    clients = profile.get("player_client")
    if clients:
        options["extractor_args"] = {"youtube": {"player_client": clients}}
    if profile.get("pot"):
        options.setdefault("extractor_args", {})["youtubepot-bgutilscript"] = {
            "server_home": [pot_home]
        }
    if profile.get("cookies"):
        options["cookiefile"] = cookie_file

    print(
        f"RIPPED_SOURCE_PROFILE start video_id={video_id} profile={profile['name']} "
        f"po_token={bool(profile.get('pot'))} cookies={bool(profile.get('cookies'))}",
        flush=True,
    )
    try:
        with YoutubeDL(options) as ydl:
            ydl.extract_info(youtube_url, download=True)
        if stop_event.is_set():
            raise RuntimeError("cancelled because another YouTube acquisition profile won")
        matches = sorted(lane.glob(f"{video_id}-source.*"))
        usable = next((item for item in matches if item.is_file() and item.stat().st_size > 0), None)
        if not usable:
            if usable:
                raise RuntimeError("profile produced a video-only source without an audio stream")
            raise RuntimeError("profile completed without a usable source video")
        print(
            f"RIPPED_SOURCE_PROFILE success video_id={video_id} profile={profile['name']} "
            f"bytes={usable.stat().st_size}",
            flush=True,
        )
        return usable
    except Exception as exc:
        print(
            f"RIPPED_SOURCE_PROFILE failed video_id={video_id} profile={profile['name']} "
            f"error={type(exc).__name__}: {exc}",
            flush=True,
        )
        raise


def download_youtube_resilient(video_id: str, youtube_url: str, workdir: Path) -> Path:
    """Acquire a full video with two concurrent fallback lanes; first valid download wins."""
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
    cookies_base64 = (
        os.getenv("YOUTUBE_COOKIES_BASE64")
        or os.getenv("YTDLP_COOKIES_BASE64")
        or ""
    ).strip()
    cookie_text = (os.getenv("YOUTUBE_COOKIES") or "").strip()

    if cookies_base64 and (not cookie_file or not Path(cookie_file).is_file()):
        try:
            decoded = base64.b64decode(cookies_base64, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError) as exc:
            raise RuntimeError("YouTube cookie base64 value is not valid UTF-8 cookie text") from exc
        generated_cookie_file.write_text(decoded.rstrip() + "\n", encoding="utf-8")
        cookie_file = str(generated_cookie_file)
    elif cookie_text and (not cookie_file or not Path(cookie_file).is_file()):
        generated_cookie_file.write_text(cookie_text.replace("\\n", "\n").rstrip() + "\n", encoding="utf-8")
        cookie_file = str(generated_cookie_file)

    pot_home = (
        os.getenv("YTDLP_POT_PROVIDER_HOME")
        or "/opt/bgutil-ytdlp-pot-provider/server"
    ).strip()
    has_pot = bool(pot_home and Path(pot_home).is_dir())
    print(
        f"PO_TOKEN_PROVIDER ready={str(has_pot).lower()} home={pot_home}",
        flush=True,
    )

    profiles: list[dict[str, Any]] = []
    rapidapi_ready = bool((os.getenv("RAPIDAPI_KEY") or "").strip())
    if rapidapi_ready:
        profiles.append(
            {
                "name": "rapidapi_video_download",
                "provider": "rapidapi",
            }
        )
    if has_pot:
        profiles.append(
            {
                "name": "automatic_po_token_mweb",
                "format": "bv*[height<=1080]+ba/b[height<=1080][acodec!=none]",
                "player_client": ["mweb"],
                "pot": True,
            }
        )
    profiles.extend(
        [
            {
                "name": "public_original_selector",
                "format": "bv*[height<=1080]+ba/b[height<=1080][acodec!=none]",
                "player_client": ["default", "tv_simply"],
            },
            {
                "name": "public_auto_format",
                "format": None,
                "player_client": ["default", "tv_simply"],
            },
            {
                "name": "public_android_vr",
                "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
                "player_client": ["android_vr"],
            },
            {
                "name": "public_web_safari",
                "format": None,
                "player_client": ["web_safari"],
            },
        ]
    )
    if has_pot:
        profiles.append(
            {
                "name": "automatic_po_token_web_safari",
                "format": None,
                "player_client": ["web_safari"],
                "pot": True,
            }
        )
    if cookie_file and Path(cookie_file).is_file():
        profiles.extend(
            [
                {
                    "name": "cookie_auto_format",
                    "format": None,
                    "player_client": None,
                    "cookies": True,
                },
                {
                    "name": "cookie_po_token_mweb",
                    "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
                    "player_client": ["mweb"],
                    "cookies": True,
                    "pot": has_pot,
                },
            ]
        )

    failures: list[str] = []
    stop_event = threading.Event()
    pending_profiles = iter(profiles)
    futures: dict[Any, dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="youtube-acquire") as executor:
        for _ in range(2):
            try:
                profile = next(pending_profiles)
            except StopIteration:
                break
            future = executor.submit(
                _run_source_profile,
                profile=profile,
                video_id=video_id,
                youtube_url=youtube_url,
                workdir=workdir,
                cookie_file=cookie_file,
                pot_home=pot_home,
                stop_event=stop_event,
            )
            futures[future] = profile

        winner: Path | None = None
        winner_profile: dict[str, Any] | None = None
        while futures and winner is None:
            done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
            for future in done:
                profile = futures.pop(future)
                try:
                    candidate = future.result()
                except Exception as exc:
                    failures.append(f"{profile['name']}: {type(exc).__name__}: {exc}")
                    try:
                        replacement_profile = next(pending_profiles)
                    except StopIteration:
                        continue
                    replacement_future = executor.submit(
                        _run_source_profile,
                        profile=replacement_profile,
                        video_id=video_id,
                        youtube_url=youtube_url,
                        workdir=workdir,
                        cookie_file=cookie_file,
                        pot_home=pot_home,
                        stop_event=stop_event,
                    )
                    futures[replacement_future] = replacement_profile
                    continue

                winner = candidate
                winner_profile = profile
                stop_event.set()
                break

        if winner is not None:
            for future in futures:
                future.cancel()
            if destination.exists():
                destination.unlink()
            shutil.move(str(winner), str(destination))
            if winner_profile is None:
                raise RuntimeError("source winner profile was not recorded")
            _log_source_winner(
                video_id=video_id,
                profile=winner_profile,
                path=destination,
                workdir=workdir,
            )
            print(
                f"RIPPED_SOURCE_RACE winner video_id={video_id} "
                f"profile={winner_profile['name']} "
                f"provider={winner_profile.get('provider', 'yt-dlp')}",
                flush=True,
            )
            return destination

    raise RuntimeError(
        "All Ripped Shorts full-video acquisition profiles failed for "
        f"{video_id}: " + "; ".join(failures[-12:])
    )


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


def select_non_overlapping(
    clips: list[dict[str, Any]],
    limit: int = 20,
    *,
    allow_overlap: bool = False,
) -> list[dict[str, Any]]:
    ranked = sorted(clips, key=lambda item: float(item.get("score", 0)), reverse=True)
    chosen: list[dict[str, Any]] = []
    for clip in ranked:
        start = float(clip.get("start", 0))
        end = float(clip.get("end", start + float(clip.get("duration", 0))))
        if end <= start:
            continue
        normalized_text = " ".join(
            str(clip.get("transcript", "")).lower().split()
        )
        if normalized_text and any(
            normalized_text
            == " ".join(str(item.get("transcript", "")).lower().split())
            for item in chosen
        ):
            continue
        if not allow_overlap and any(
            start < float(item["end"]) and end > float(item["start"])
            for item in chosen
        ):
            continue
        clip["start"], clip["end"], clip["duration"] = start, end, end - start
        chosen.append(clip)
        if len(chosen) >= limit:
            break
    return sorted(chosen, key=lambda item: float(item["start"]))
