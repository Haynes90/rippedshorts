import json
import os
import re
import time
import uuid
import logging
import math
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import shutil
from urllib.parse import urlparse, parse_qs

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

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
YOUTUBE_DL_HOST = os.environ.get("YOUTUBE_DL_HOST", "youtube-video-fast-downloader-24-7.p.rapidapi.com")
YOUTUBE_DL_PATH_TEMPLATE = os.environ.get("YOUTUBE_DL_PATH_TEMPLATE", "/download_video/{video_id}")
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID") or os.getenv("Drive_Folder_ID")
DEFAULT_SHEET_ID = os.getenv("DEFAULT_SHEET_ID", "1xfp-sjO9Mnvwe7-bM6htT-0RKiOig21HfP_otzO9xws")
DEFAULT_SHEET_TAB = os.getenv("DEFAULT_SHEET_TAB", "Sheet1")
GOOGLE_CREDENTIALS = os.getenv("GOOGLE_CREDENTIALS")
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
GOOGLE_CLIENT_EMAIL = os.getenv("GOOGLE_CLIENT_EMAIL")
GOOGLE_PRIVATE_KEY = os.getenv("GOOGLE_PRIVATE_KEY")
GOOGLE_PRIVATE_KEY_ID = os.getenv("GOOGLE_PRIVATE_KEY_ID")
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_CLIP_TIMEOUT_SECONDS = max(
    180, int(os.getenv("OPENAI_CLIP_TIMEOUT_SECONDS", "600"))
)
OPENAI_CLIP_ATTEMPTS = max(1, int(os.getenv("OPENAI_CLIP_ATTEMPTS", "3")))

if not RAPIDAPI_KEY:
    logger.warning("RAPIDAPI_KEY not set (discover will fail until configured).")
if not YOUTUBE_DL_HOST:
    logger.warning("YOUTUBE_DL_HOST not set (youtube downloader will fail until configured).")
if not DRIVE_FOLDER_ID:
    logger.warning("Drive folder id not set (DRIVE_FOLDER_ID).")
if not GOOGLE_CREDENTIALS and not GOOGLE_SERVICE_ACCOUNT_FILE and not (GOOGLE_CLIENT_EMAIL and GOOGLE_PRIVATE_KEY):
    logger.warning(
        "Google credentials not set (GOOGLE_CREDENTIALS, GOOGLE_SERVICE_ACCOUNT_FILE, "
        "or GOOGLE_CLIENT_EMAIL + GOOGLE_PRIVATE_KEY)."
    )
if GOOGLE_API_KEY:
    logger.info("GOOGLE_API provided but not used for Docs/Drive write access.")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set (clip discovery will fail until configured).")

# -------------------------
# MODELS
# -------------------------
YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,20}$")


class DiscoverRequest(BaseModel):
    video_id: Optional[str] = Field(None, min_length=6)
    youtube_url: Optional[str] = None
    sheet_id: Optional[str] = None
    sheet_tab: Optional[str] = None
    prompt: Optional[str] = None
    wait_for_result: bool = False
    wait_timeout_s: int = Field(0, ge=0, le=600)


class DiscoverResponse(BaseModel):
    status: str
    job_id: str
    video_id: str
    step: Optional[str] = None
    error: Optional[str] = None
    elapsed_s: Optional[float] = None

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
        start = float(segment["start"])
        duration = float(segment.get("duration", 0.0))
        lines.append(f"[{timestamp} | start={start:.2f}s | dur={duration:.2f}s] {segment['text']}")
    return "\n".join(lines)


def get_google_services():
    scopes = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/documents",
        "https://www.googleapis.com/auth/spreadsheets",
    ]
    if GOOGLE_CREDENTIALS:
        info = json.loads(GOOGLE_CREDENTIALS)
        creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
    elif GOOGLE_SERVICE_ACCOUNT_FILE:
        creds = service_account.Credentials.from_service_account_file(
            GOOGLE_SERVICE_ACCOUNT_FILE,
            scopes=scopes,
        )
    elif GOOGLE_CLIENT_EMAIL and GOOGLE_PRIVATE_KEY:
        private_key = GOOGLE_PRIVATE_KEY.replace("\\n", "\n")
        info = {
            "type": "service_account",
            "client_email": GOOGLE_CLIENT_EMAIL,
            "private_key": private_key,
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        if GOOGLE_PRIVATE_KEY_ID:
            info["private_key_id"] = GOOGLE_PRIVATE_KEY_ID
        if GOOGLE_PROJECT_ID:
            info["project_id"] = GOOGLE_PROJECT_ID
        if GOOGLE_CLIENT_ID:
            info["client_id"] = GOOGLE_CLIENT_ID
        creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
    else:
        raise RuntimeError(
            "Google credentials not configured (GOOGLE_CREDENTIALS, "
            "GOOGLE_SERVICE_ACCOUNT_FILE, or GOOGLE_CLIENT_EMAIL + GOOGLE_PRIVATE_KEY)"
        )

    drive_service = build("drive", "v3", credentials=creds, cache_discovery=False)
    docs_service = build("docs", "v1", credentials=creds, cache_discovery=False)
    sheets_service = build("sheets", "v4", credentials=creds, cache_discovery=False)
    return drive_service, docs_service, sheets_service


def create_transcript_doc(video_id: str, segments: List[dict]) -> Dict[str, str]:
    if not DRIVE_FOLDER_ID:
        raise RuntimeError("Drive folder id not configured (Drive_Folder_ID/DRIVE_FOLDER_ID)")

    drive_service, docs_service, _ = get_google_services()
    title = f"{video_id} Full transcript"
    file_metadata = {
        "name": title,
        "mimeType": "application/vnd.google-apps.document",
        "parents": [DRIVE_FOLDER_ID],
    }
    file_obj = drive_service.files().create(
        body=file_metadata,
        fields="id, parents",
        supportsAllDrives=True,
    ).execute()
    doc_id = file_obj["id"]

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


def download_youtube_video(video_id: str, youtube_url: Optional[str], workdir: Path) -> Path:
    """Use the same resilient yt-dlp source recovery pattern as Audio Master."""
    from source_ingestion import download_youtube_resilient

    source_url = youtube_url or f"https://www.youtube.com/watch?v={video_id}"
    return download_youtube_resilient(video_id, source_url, workdir)

def cleanup_old_temp_downloads(max_age_hours: int = 24) -> None:
    cutoff = time.time() - (max_age_hours * 3600)
    tmp_root = Path("/tmp")
    for path in tmp_root.glob("clips_*"):
        try:
            if not path.is_dir():
                continue
            if path.stat().st_mtime >= cutoff:
                continue
            shutil.rmtree(path, ignore_errors=True)
        except Exception as exc:
            logger.warning("Failed to cleanup temp dir %s: %s", path, exc)


def _probe_video_dimensions(video_path: Path) -> tuple[int, int]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError("Unable to read video dimensions")
    return width, height


def _load_face_cascade():
    import cv2

    candidates = [
        Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml",
        Path("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"),
        Path("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"),
    ]
    for path in candidates:
        if not path.is_file():
            continue
        cascade = cv2.CascadeClassifier(str(path))
        if not cascade.empty():
            logger.info("OpenCV speaker tracker face model=%s", path)
            return cascade
    logger.warning("OpenCV face model unavailable; motion/center tracking will be used")
    return None


def _motion_center(previous_gray, gray) -> Optional[float]:
    import cv2
    import numpy as np

    if previous_gray is None or previous_gray.shape != gray.shape:
        return None
    delta = cv2.absdiff(previous_gray, gray)
    delta = cv2.GaussianBlur(delta, (9, 9), 0)
    _, mask = cv2.threshold(delta, 24, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8)
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    meaningful = [contour for contour in contours if cv2.contourArea(contour) >= 400]
    if not meaningful:
        return None
    weighted_x = 0.0
    total_area = 0.0
    for contour in meaningful:
        area = float(cv2.contourArea(contour))
        x, _, width, _ = cv2.boundingRect(contour)
        weighted_x += (x + width / 2) * area
        total_area += area
    return (weighted_x / total_area) / gray.shape[1] if total_area else None


def _estimate_speaker_track(
    video_path: Path, start: float, duration: float
) -> list[tuple[float, float]]:
    """Track the active foreground face without chasing brief background detections."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    sample_seconds = max(0.5, float(os.getenv("SPEAKER_TRACK_SAMPLE_SECONDS", "1.0")))
    frame_step = max(1, int(round(fps * sample_seconds)))
    total_frames = max(1, int(round(duration * fps)))
    cascade = _load_face_cascade()
    samples: list[tuple[float, float]] = []
    previous_gray = None
    previous_center = 0.5
    previous_box = None
    smoothed_center = 0.5
    pending_switch_center: Optional[float] = None
    pending_switch_count = 0
    lost_samples = 0
    switch_distance = max(
        0.08, float(os.getenv("SPEAKER_SWITCH_DISTANCE", "0.18"))
    )
    switch_confirmations = max(
        2, int(os.getenv("SPEAKER_SWITCH_CONFIRMATIONS", "3"))
    )
    frame_number = 0

    while frame_number <= total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(start * fps)) + frame_number)
        ok, frame = cap.read()
        if not ok:
            break
        scale = min(1.0, 720.0 / max(frame.shape[:2]))
        analysis_frame = (
            cv2.resize(frame, None, fx=scale, fy=scale)
            if scale < 1.0
            else frame
        )
        gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
        faces = []
        if cascade is not None:
            faces = list(
                cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(40, 40),
                )
            )

        selected = None
        selected_score = -1.0
        frame_area = float(gray.shape[0] * gray.shape[1])
        for x, y, width, height in faces:
            center = (x + width / 2) / gray.shape[1]
            area_score = (width * height) / frame_area
            continuity = max(0.0, 1.0 - abs(center - previous_center) / 0.35)
            motion_score = 0.0
            if previous_gray is not None and previous_gray.shape == gray.shape:
                y2, x2 = min(gray.shape[0], y + height), min(gray.shape[1], x + width)
                if y2 > y and x2 > x:
                    motion_score = float(
                        np.mean(cv2.absdiff(previous_gray[y:y2, x:x2], gray[y:y2, x:x2]))
                    ) / 255.0
            foreground = min(1.0, area_score * 30.0)
            score = foreground * 2.5 + motion_score * 3.0 + continuity * 1.5
            if previous_box is not None and abs(center - previous_center) < 0.12:
                score += 0.75
            if score > selected_score:
                selected_score = score
                selected = (center, (x, y, width, height))

        if selected is None:
            lost_samples += 1
            pending_switch_center = None
            pending_switch_count = 0
            # A short detection loss must not make the crop chase motion or another face.
            if lost_samples <= 2:
                target_center = previous_center
            else:
                target_center = _motion_center(previous_gray, gray)
                if target_center is None:
                    target_center = previous_center
        else:
            lost_samples = 0
            detected_center, detected_box = selected
            if abs(detected_center - previous_center) >= switch_distance:
                if (
                    pending_switch_center is not None
                    and abs(detected_center - pending_switch_center) < 0.08
                ):
                    pending_switch_count += 1
                    pending_switch_center = (
                        pending_switch_center * (pending_switch_count - 1)
                        + detected_center
                    ) / pending_switch_count
                else:
                    pending_switch_center = detected_center
                    pending_switch_count = 1
                if pending_switch_count >= switch_confirmations:
                    target_center = pending_switch_center
                    previous_box = detected_box
                    pending_switch_center = None
                    pending_switch_count = 0
                else:
                    target_center = previous_center
            else:
                pending_switch_center = None
                pending_switch_count = 0
                target_center = detected_center
                previous_box = detected_box

        # This smooths noisy analysis samples only. The rendered crop is stationary
        # inside each framing section and never exposes this interpolation.
        if abs(target_center - smoothed_center) < 0.025:
            target_center = smoothed_center
        smoothed_center = 0.72 * smoothed_center + 0.28 * target_center
        smoothed_center = max(0.0, min(1.0, smoothed_center))
        previous_center = smoothed_center
        samples.append((frame_number / fps, smoothed_center))
        previous_gray = gray
        frame_number += frame_step

    cap.release()
    if not samples:
        return [(0.0, 0.5), (max(duration, 0.1), 0.5)]
    if samples[-1][0] < duration:
        samples.append((duration, samples[-1][1]))
    return samples


def _detect_audio_pause_boundaries(
    video_path: Path, start: float, duration: float
) -> list[float]:
    """Return clip-relative pause endpoints that can hide a deliberate reframe."""
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(video_path),
        "-af",
        "silencedetect=noise=-35dB:d=0.18",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=max(30, int(duration) + 15)
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("Pause detection unavailable: %s", exc)
        return []
    output = completed.stderr or ""
    return [
        value
        for value in (
            float(match)
            for match in re.findall(r"silence_end:\\s*([0-9.]+)", output)
        )
        if 0.0 < value < duration
    ]


def _plan_framing_sections(
    samples: list[tuple[float, float]],
    duration: float,
    preferred_boundaries: Optional[list[float]] = None,
) -> list[tuple[float, float]]:
    """Plan stationary 3–7 second crops, preferring pauses and tracking cuts."""
    from statistics import median

    if not samples:
        return [(0.0, 0.5)]
    target_seconds = min(
        7.0, max(3.0, float(os.getenv("SPEAKER_REFRAME_SECONDS", "4.0")))
    )
    minimum_seconds = min(
        target_seconds,
        max(3.0, float(os.getenv("SPEAKER_REFRAME_MIN_SECONDS", "3.0"))),
    )
    maximum_seconds = max(
        target_seconds,
        min(7.0, float(os.getenv("SPEAKER_REFRAME_MAX_SECONDS", "7.0"))),
    )
    movement_threshold = max(
        0.02, float(os.getenv("SPEAKER_REFRAME_THRESHOLD", "0.07"))
    )

    preferred = list(preferred_boundaries or [])
    # Abrupt, persistent position changes are useful proxies for a camera cut or
    # speaker change. Slow movement is intentionally not turned into crop motion.
    for index in range(1, len(samples)):
        if abs(samples[index][1] - samples[index - 1][1]) >= movement_threshold * 1.5:
            preferred.append(samples[index][0])
    preferred = sorted(set(round(value, 3) for value in preferred if 0 < value < duration))

    boundaries = [0.0]
    cursor = 0.0
    while duration - cursor > maximum_seconds:
        low = cursor + minimum_seconds
        # Leave at least the minimum hold for the final section too.
        high = min(cursor + maximum_seconds, duration - minimum_seconds)
        ideal = min(cursor + target_seconds, high)
        choices = [value for value in preferred if low <= value <= high]
        boundary = min(choices, key=lambda value: abs(value - ideal)) if choices else ideal
        boundaries.append(boundary)
        cursor = boundary
    if boundaries[-1] < duration:
        boundaries.append(duration)

    sections: list[tuple[float, float]] = []
    previous_center: Optional[float] = None
    for index in range(len(boundaries) - 1):
        section_start = boundaries[index]
        section_end = boundaries[index + 1]
        centers = [
            center
            for timestamp, center in samples
            if section_start <= timestamp < section_end
        ]
        proposed_center = median(centers) if centers else (
            previous_center if previous_center is not None else 0.5
        )
        proposed_center = max(0.0, min(1.0, float(proposed_center)))
        if (
            previous_center is not None
            and abs(proposed_center - previous_center) < movement_threshold
        ):
            proposed_center = previous_center
        if previous_center is None or proposed_center != previous_center:
            sections.append((section_start, proposed_center))
            previous_center = proposed_center

    return sections or [(0.0, 0.5)]


def _stepped_crop_expression(
    sections: list[tuple[float, float]], width: int, target_width: int
) -> str:
    """Build an FFmpeg crop expression with hard holds and intentional snaps."""
    positions = [
        (
            timestamp,
            max(
                0,
                min(
                    width - target_width,
                    int(center * width) - target_width // 2,
                ),
            ),
        )
        for timestamp, center in sections
    ]
    expression = str(positions[-1][1])
    for index in range(len(positions) - 2, -1, -1):
        next_timestamp = positions[index + 1][0]
        expression = (
            f"if(lt(t,{next_timestamp:.3f}),{positions[index][1]},{expression})"
        )
    return expression


def _build_crop_filter(video_path: Path, start: float, duration: float) -> str:
    width, height = _probe_video_dimensions(video_path)
    target_width = min(width, int(height * 9 / 16))
    target_width = max(2, target_width - (target_width % 2))
    samples = _estimate_speaker_track(video_path, start, duration)
    pause_boundaries = _detect_audio_pause_boundaries(video_path, start, duration)
    sections = _plan_framing_sections(samples, duration, pause_boundaries)
    crop_expression = _stepped_crop_expression(sections, width, target_width)
    logger.info(
        "Speaker reframing duration=%.2fs sections=%s pauses=%s",
        duration,
        len(sections),
        len(pause_boundaries),
    )
    return f"crop={target_width}:{height}:x='{crop_expression}':y=0"


def create_clip_file(video_path: Path, start: float, duration: float, output_path: Path) -> None:
    crop_filter = _build_crop_filter(video_path, start, duration)
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.2f}",
        "-t",
        f"{duration:.2f}",
        "-i",
        str(video_path),
        "-vf",
        f"{crop_filter},scale=1080:1920",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    if completed.returncode != 0:
        raise RuntimeError(
            f"FFmpeg render failed with code {completed.returncode}: "
            f"{(completed.stderr or completed.stdout)[-3000:]}"
        )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg render produced no usable clip: {output_path}")


def create_topic_segment_file(
    video_path: Path, start: float, duration: float, output_path: Path
) -> None:
    """Render a horizontal topic segment without applying the Shorts crop."""
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.2f}",
        "-t",
        f"{duration:.2f}",
        "-i",
        str(video_path),
        "-vf",
        "scale=1920:1080:force_original_aspect_ratio=decrease,"
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=7200)
    if completed.returncode != 0:
        raise RuntimeError(
            f"FFmpeg 16:9 render failed with code {completed.returncode}: "
            f"{(completed.stderr or completed.stdout)[-3000:]}"
        )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg 16:9 render produced no usable segment: {output_path}")


def attach_topic_segment_asset(
    segment: dict,
    video_id: str,
    video_path: Path,
    segment_number: int,
    vid_title: Optional[str] = None,
) -> dict:
    workdir = Path("/tmp") / f"topics_{video_id}"
    workdir.mkdir(parents=True, exist_ok=True)
    segment_name = f"{video_id}_Segment_{segment_number}.mp4"
    output_path = workdir / f".render-{uuid.uuid4().hex}-{segment_name}"
    start = float(segment.get("start", 0.0))
    duration = float(segment.get("duration", 0.0))
    if duration <= 0:
        raise RuntimeError("16:9 segment duration must be positive")
    create_topic_segment_file(video_path, start, duration, output_path)
    uploaded = upload_clip_to_drive(output_path, segment_name, vid_title=vid_title)
    return {
        **segment,
        "segment_number": segment_number,
        "segment_name": segment_name,
        "segment_url": uploaded["clip_url"],
        "folder_id": uploaded["folder_id"],
    }


def _drive_title_folder(vid_title: Optional[str]) -> str:
    """Return the reusable Vid Title folder inside the configured output folder."""
    if not DRIVE_FOLDER_ID:
        raise RuntimeError("Drive folder id not configured (Drive_Folder_ID/DRIVE_FOLDER_ID)")
    folder_name = str(vid_title or "").strip()
    if not folder_name:
        raise RuntimeError("Vid Title is required before approved Ripped Shorts can be uploaded")

    drive_service, _, _ = get_google_services()
    safe_name = folder_name.replace("\\", "\\\\").replace("'", "\\'")
    query = (
        "mimeType = 'application/vnd.google-apps.folder' "
        f"and name = '{safe_name}' and '{DRIVE_FOLDER_ID}' in parents and trashed = false"
    )
    existing = drive_service.files().list(
        q=query,
        fields="files(id,name)",
        pageSize=1,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute().get("files", [])
    if existing:
        return existing[0]["id"]

    created = drive_service.files().create(
        body={
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [DRIVE_FOLDER_ID],
        },
        fields="id,name",
        supportsAllDrives=True,
    ).execute()
    return created["id"]


def upload_clip_to_drive(
    clip_path: Path, clip_name: str, *, vid_title: Optional[str] = None
) -> dict:
    drive_service, _, _ = get_google_services()
    output_folder_id = _drive_title_folder(vid_title)
    file_metadata = {
        "name": clip_name,
        "parents": [output_folder_id],
    }
    media = MediaFileUpload(str(clip_path), mimetype="video/mp4", resumable=True)
    uploaded = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id, webViewLink",
        supportsAllDrives=True,
    ).execute()
    return {
        "clip_id": uploaded["id"],
        "clip_url": uploaded.get("webViewLink") or f"https://drive.google.com/file/d/{uploaded['id']}/view",
        "folder_id": output_folder_id,
    }


def download_video_asset(asset_url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(asset_url, stream=True, timeout=(10, 120)) as resp:
        resp.raise_for_status()
        with output_path.open("wb") as handle:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def _seconds_to_timecode(seconds: float) -> str:
    total_seconds = int(max(0, math.floor(seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def attach_clip_assets(
    clips_payload: dict,
    video_id: str,
    youtube_url: Optional[str],
    video_path_override: Optional[Path] = None,
    vid_title: Optional[str] = None,
) -> dict:
    cleanup_old_temp_downloads(max_age_hours=24)
    segments = clips_payload.get("segments", [])
    if not segments:
        return clips_payload

    workdir = Path("/tmp") / f"clips_{video_id}"
    workdir.mkdir(parents=True, exist_ok=True)
    video_path = video_path_override or download_youtube_video(video_id, youtube_url, workdir)
    for idx, segment in enumerate(segments, start=1):
        start = float(segment.get("start", 0.0))
        duration = float(segment.get("duration", 0.0))
        end = segment.get("end")
        if duration <= 0 and end is not None:
            duration = max(0.0, float(end) - start)
        if duration <= 0:
            continue
        candidate_number = int(segment.get("candidate_number") or idx)
        clip_name = f"{video_id}_Clip_{candidate_number}.mp4"
        # Keep the user-facing Drive filename predictable while ensuring
        # simultaneous FFmpeg processes never write to the same local path.
        output_path = workdir / f".render-{uuid.uuid4().hex}-{clip_name}"
        create_clip_file(video_path, start, duration, output_path)
        clip_info = upload_clip_to_drive(output_path, clip_name, vid_title=vid_title)
        segment["clip_name"] = clip_name
        segment["clip_url"] = clip_info["clip_url"]
        segment["folder_id"] = clip_info["folder_id"]
    return clips_payload


def openai_clip_prompt(transcript_segments: List[dict], prompt_override: Optional[str]) -> str:
    base_prompt = (
        "TASK\n"
        "You are a highlight editor for ANY type of content. Review the ENTIRE transcript in chronological order "
        "and select the best short-form clips.\n"
        "Target 18–20 final clips, each 10–90 seconds, prioritized by engagement and standalone clarity. "
        "When the transcript contains at least 20 qualified complete thoughts, return 20. "
        "Build up to 24 candidates internally so validation can retain close to 20.\n"
        "You MUST scan the full transcript before selecting any clips.\n\n"
        "TRANSCRIPT FORMAT (YOU MUST FOLLOW THIS)\n"
        "- Each transcript line is already time-aligned and looks like:\n"
        "  [MM:SS | start=###.##s | dur=##.##s] text...\n"
        "- The transcript is an ordered timeline. Do NOT reorder lines.\n"
        "- You may create a clip by selecting ONE line OR combining MULTIPLE ADJACENT lines only.\n"
        "- Never combine non-adjacent lines.\n"
        "- For a combined clip:\n"
        "  - start = start of the first included line\n"
        "  - end = (start of last included line) + (dur of last included line)\n"
        "  - duration = end - start\n"
        "  - transcript = exact concatenation of included texts, in order\n"
        "- Use timestamps EXACTLY as provided. Do not guess.\n\n"
        "HARD REQUIREMENTS\n"
        "- Each chosen clip MUST be 10–90 seconds.\n"
        "- Each clip MUST be a complete, standalone thought (no cut-off setup, mid-sentence starts, "
        "or missing payoff).\n"
        "- Every clip MUST represent a complete sentence or complete thought.\n"
        "- Finish the selected sentence/payoff where possible. End before the opening words of a new sentence, topic, or tangent.\n"
        "- Each clip MUST deliver at least one of: an impactful lesson, a strong insight, or a quotable line.\n"
        "- Do NOT paraphrase, rewrite, infer missing context, or fabricate.\n"
        "- Do NOT return duplicate or near-duplicate complete clips; each clip must be materially distinct.\n"
        "- Clips MAY overlap in time when overlap is needed to preserve two different complete thoughts.\n"
        "- Reusing a short setup is allowed, but do not return the same lesson, payoff, or complete clip twice.\n"
        "- Spread selections across the full eligible timeline instead of clustering around one section.\n"
        "- Prefer fewer excellent complete thoughts over padding the result to 20.\n"
        "- Avoid repeated lessons, examples, stories, claims, setups, and payoffs; maximize topical variety.\n\n"
        "PROCESS\n"
        "1) First pass: classify content_type, determine main_theme, 3–8 key ideas, and useful topic keywords.\n"
        "2) Second pass: build a larger candidate pool across the beginning, middle, and end.\n"
        "3) Remove clips that duplicate a complete thought, lack their setup/payoff, or cut a sentence.\n"
        "4) Rank the remaining distinct candidates and return as close to 20 as quality allows.\n"
        "5) Categorize each clip using the MASTER CATEGORY LIST.\n\n"
        "MASTER CATEGORY LIST (choose ONE per clip)\n"
        "- inspiration\n"
        "- education\n"
        "- humor\n"
        "- story\n"
        "- insight\n"
        "- call-to-action\n"
        "- controversy\n"
        "- behind-the-scenes\n"
        "- social-proof\n"
        "- empathy\n"
        "- mindset\n"
        "- leadership\n"
        "- business\n"
        "- science-tech\n"
        "- lifestyle\n"
        "- spirituality\n"
        "- community\n"
        "- quote\n\n"
        "SCORING (0–100)\n"
        "- Hook strength in first 2–3 seconds (0–30)\n"
        "- Standalone clarity / completeness (0–25)\n"
        "- Impact (emotion/usefulness/novelty) (0–25)\n"
        "- Shareability / quoteability (0–10)\n"
        "- Variety contribution vs other picks (0–10)\n\n"
        "OUTPUT FORMAT (STRICT JSON ONLY)\n"
        "{\n"
        "  \"analysis\": {\n"
        "    \"content_type\": \"church | podcast | livestream | interview | teaching | other\",\n"
        "    \"main_theme\": \"string\",\n"
        "    \"key_ideas\": [\"string\", \"string\"],\n"
        "    \"keywords\": [\"string\", \"string\"]\n"
        "  },\n"
        "  \"segments\": [\n"
        "    {\n"
        "      \"video_id\": \"string\",\n"
        "      \"start\": number,\n"
        "      \"end\": number,\n"
        "      \"duration\": number,\n"
        "      \"transcript\": \"string\",\n"
        "      \"score\": number,\n"
        "      \"category\": \"inspiration\" | \"education\" | \"humor\" | \"story\" | \"insight\" | "
        "\"call-to-action\" | \"controversy\" | \"behind-the-scenes\" | \"social-proof\" | \"empathy\" | "
        "\"mindset\" | \"leadership\" | \"business\" | \"science-tech\" | \"lifestyle\" | \"spirituality\" | "
        "\"community\" | \"quote\",\n"
        "      \"reason\": \"short justification\",\n"
        "      \"source_lines\": [\n"
        "        \"[MM:SS | start=###.##s | dur=##.##s] ...\",\n"
        "        \"[MM:SS | start=###.##s | dur=##.##s] ...\"\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "If nothing qualifies:\n"
        "{ \"analysis\": {\"content_type\": \"other\", \"main_theme\": \"\", "
        "\"key_ideas\": [], \"keywords\": []}, \"segments\": [] }\n"
    )
    if prompt_override:
        base_prompt += (
            "\n\nADDITIONAL DISCOVERY REQUIREMENTS\n"
            + str(prompt_override).strip()
            + "\n"
        )

    def _mmss(seconds: float) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    video_id = ""
    lines = []
    for seg in transcript_segments:
        start = float(seg["start"])
        dur = float(seg["duration"])
        text = str(seg["text"]).replace("\n", " ").strip()
        video_id = seg.get("video_id", video_id) or video_id
        lines.append(f"[{_mmss(start)} | start={start:.2f}s | dur={dur:.2f}s] {text}")

    transcript_block = "\n".join(lines)

    return (
        f"{base_prompt}\n\n"
        f"VIDEO_ID: {video_id or 'unknown'}\n"
        "TRANSCRIPT_TIMELINE (chronological, do not reorder):\n"
        f"{transcript_block}"
    )


def call_openai_for_clips(transcript_segments: List[dict], prompt_override: Optional[str]) -> dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not configured")

    prompt = openai_clip_prompt(transcript_segments, prompt_override)
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that returns strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    resp = None
    last_error = None
    for attempt in range(1, OPENAI_CLIP_ATTEMPTS + 1):
        try:
            logger.info(
                "OpenAI clip selection attempt=%s/%s model=%s transcript_segments=%s read_timeout=%ss",
                attempt,
                OPENAI_CLIP_ATTEMPTS,
                OPENAI_MODEL,
                len(transcript_segments),
                OPENAI_CLIP_TIMEOUT_SECONDS,
            )
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=(15, OPENAI_CLIP_TIMEOUT_SECONDS),
            )
            if resp.status_code == 200:
                break
            if resp.status_code not in {408, 409, 429} and resp.status_code < 500:
                raise RuntimeError(f"OpenAI API error ({resp.status_code}): {resp.text}")
            last_error = RuntimeError(
                f"OpenAI retryable API error ({resp.status_code}): {resp.text[:1000]}"
            )
        except (requests.ReadTimeout, requests.ConnectionError) as exc:
            last_error = exc
            logger.warning(
                "OpenAI clip selection attempt=%s/%s failed: %s",
                attempt,
                OPENAI_CLIP_ATTEMPTS,
                exc,
            )
        if attempt < OPENAI_CLIP_ATTEMPTS:
            time.sleep(min(10, 2 ** attempt))
    if resp is None or resp.status_code != 200:
        raise RuntimeError(
            "OpenAI clip selection failed after "
            f"{OPENAI_CLIP_ATTEMPTS} attempts with "
            f"{OPENAI_CLIP_TIMEOUT_SECONDS}s read timeout: {last_error}"
        )
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    raw_content = content.strip()
    if raw_content.startswith("```"):
        lines = raw_content.splitlines()
        if len(lines) >= 2:
            raw_content = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError:
        start = raw_content.find("{")
        end = raw_content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw_content[start : end + 1])
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"OpenAI response was not valid JSON: {raw_content}") from exc
        raise RuntimeError(f"OpenAI response was not valid JSON: {raw_content}")


def write_clips_to_sheet(
    sheet_id: str,
    sheet_tab: str,
    clips_payload: dict,
    video_id: str,
    transcript_segments: List[dict],
) -> dict:
    _, _, sheets_service = get_google_services()
    segments = clips_payload.get("segments", [])
    transcript_lookup = {seg["start"]: seg["text"] for seg in transcript_segments}
    values = []
    for segment in segments:
        start = segment.get("start", 0.0)
        duration = segment.get("duration", 0.0)
        text = segment.get("transcript") or transcript_lookup.get(start, "")
        values.append([
            video_id,
            start,
            duration,
            text,
            segment.get("score"),
            segment.get("category") or segment.get("primary_category"),
            segment.get("reason"),
            segment.get("clip_url", ""),
            segment.get("clip_name", ""),
        ])
    if not values:
        values = [[video_id, "", "", "", "", "", "No segments returned", "", ""]]

    existing = sheets_service.spreadsheets().values().get(
        spreadsheetId=sheet_id,
        range=f"{sheet_tab}!A:A",
    ).execute()
    existing_values = existing.get("values", [])
    if not existing_values:
        start_row = 2
    else:
        start_row = len(existing_values) + 1
    range_name = f"{sheet_tab}!A{start_row}"
    result = sheets_service.spreadsheets().values().update(
        spreadsheetId=sheet_id,
        range=range_name,
        valueInputOption="RAW",
        body={"values": values},
    ).execute()
    return {"updated_cells": result.get("updatedCells"), "range": result.get("updatedRange")}

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

        JOBS[job_id]["step"] = "clip_discovery"
        if not OPENAI_API_KEY:
            logger.warning("[%s] OPENAI_API_KEY missing, skipping clip discovery", job_id)
            clips_payload = {"segments": [], "error": "OPENAI_API_KEY not configured"}
        else:
            try:
                transcript_with_id = [dict(seg, video_id=video_id) for seg in transcript]
                clips_payload = call_openai_for_clips(transcript_with_id, JOBS[job_id].get("prompt"))
            except Exception as exc:
                logger.exception("[%s] clip discovery failed, continuing with empty results", job_id)
                clips_payload = {"segments": [], "error": str(exc)}
        clip_segments = clips_payload.get("segments", [])
        logger.info("[%s] clips=%s", job_id, len(clip_segments))

        JOBS[job_id]["step"] = "clip_render"
        if clip_segments:
            youtube_url = JOBS[job_id].get("youtube_url")
            clips_payload = attach_clip_assets(clips_payload, video_id, youtube_url)

        JOBS[job_id]["step"] = "sheet_write"
        sheet_id = (JOBS[job_id].get("sheet_id") or DEFAULT_SHEET_ID).strip()
        sheet_tab = (JOBS[job_id].get("sheet_tab") or DEFAULT_SHEET_TAB).strip()
        logger.info("[%s] writing clips to sheet_id=%s tab=%s", job_id, sheet_id, sheet_tab)
        sheet_info = write_clips_to_sheet(
            sheet_id,
            sheet_tab,
            clips_payload,
            video_id,
            transcript,
        )
        logger.info("[%s] sheet updated range=%s", job_id, sheet_info.get("range"))

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["step"] = "completed"
        JOBS[job_id]["result"] = {
            "video_id": video_id,
            "segments": len(transcript),
            "chunks": len(chunks),
            "document_id": doc_info["document_id"],
            "document_url": doc_info["document_url"],
            "clips": clip_segments,
            "sheet_id": sheet_id,
            "sheet_tab": sheet_tab,
            "sheet_range": sheet_info.get("range"),
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
        "prompt": req.prompt,
        "status": "queued",
        "step": "queued",
        "created_at": time.time(),
    }

    executor.submit(run_discovery, job_id, resolved_video_id)
    if req.wait_for_result:
        timeout = req.wait_timeout_s if req.wait_timeout_s > 0 else 120
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = JOBS.get(job_id, {})
            status = job.get("status")
            if status in {"done", "error"}:
                return {
                    "status": status,
                    "job_id": job_id,
                    "video_id": resolved_video_id,
                    "step": job.get("step"),
                    "error": job.get("error"),
                    "elapsed_s": job.get("elapsed_s"),
                }
            time.sleep(1)

    return {"status": "accepted", "job_id": job_id, "video_id": resolved_video_id, "step": "queued"}

# -------------------------
# JOB STATUS (debug endpoint)
# -------------------------
@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


# Audio Master downstream ingestion routes.
from audio_master_handoff import router as audio_master_handoff_router
app.include_router(audio_master_handoff_router)

from telegram_intake import router as telegram_intake_router
app.include_router(telegram_intake_router)
