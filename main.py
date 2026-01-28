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
from urllib.parse import urlparse, parse_qs

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import cv2
import mediapipe as mp

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
DEFAULT_SHEET_ID = os.getenv("DEFAULT_SHEET_ID", "1xfp-sjO9Mnvwe7-bM6htT-0RKiOig21HfP_otzO9xws")
DEFAULT_SHEET_TAB = os.getenv("DEFAULT_SHEET_TAB", "Sheet1")
API_VIDEO_KEY = os.getenv("API_VIDEO_KEY")
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

if not RAPIDAPI_KEY:
    logger.warning("RAPIDAPI_KEY not set (discover will fail until configured).")
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
if not API_VIDEO_KEY:
    logger.warning("API_VIDEO_KEY not set (api.video clipping will be unavailable).")

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
    workdir.mkdir(parents=True, exist_ok=True)
    url = youtube_url or f"https://www.youtube.com/watch?v={video_id}"
    output_template = str(workdir / f"{video_id}.%(ext)s")
    command = [
        "yt-dlp",
        "-f",
        "mp4",
        "-o",
        output_template,
        url,
    ]
    subprocess.run(command, check=True, capture_output=True)
    for candidate in workdir.glob(f"{video_id}.*"):
        if candidate.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}:
            return candidate
    raise RuntimeError("Unable to find downloaded video file")


def _probe_video_dimensions(video_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError("Unable to read video dimensions")
    return width, height


def _estimate_speaker_center_x(video_path: Path, start: float, duration: float) -> float:
    mp_face = mp.solutions.face_detection
    detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    centers = []
    total_frames = int(min(duration, 30) * 2)
    step = max(1, int(cap.get(cv2.CAP_PROP_FPS) // 2 or 1))
    frame_idx = 0
    while len(centers) < total_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)
            if results.detections:
                bbox = results.detections[0].location_data.relative_bounding_box
                center_x = bbox.xmin + bbox.width / 2
                centers.append(center_x)
        frame_idx += 1
    cap.release()
    detector.close()
    if centers:
        return sum(centers) / len(centers)
    return 0.5


def _build_crop_filter(video_path: Path, start: float, duration: float) -> str:
    width, height = _probe_video_dimensions(video_path)
    target_width = int(height * 9 / 16)
    center_ratio = _estimate_speaker_center_x(video_path, start, duration)
    center_x = int(center_ratio * width)
    crop_x = max(0, min(width - target_width, center_x - target_width // 2))
    return f"crop={target_width}:{height}:{crop_x}:0"


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
    subprocess.run(command, check=True, capture_output=True)


def upload_clip_to_drive(clip_path: Path, clip_name: str) -> dict:
    if not DRIVE_FOLDER_ID:
        raise RuntimeError("Drive folder id not configured (Drive_Folder_ID/DRIVE_FOLDER_ID)")
    drive_service, _, _ = get_google_services()
    file_metadata = {
        "name": clip_name,
        "parents": [DRIVE_FOLDER_ID],
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
    }


def _seconds_to_timecode(seconds: float) -> str:
    total_seconds = int(max(0, math.floor(seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def create_apivideo_clip(
    source_url: str,
    clip_name: str,
    start: float,
    duration: float,
) -> dict:
    if not API_VIDEO_KEY:
        raise RuntimeError("API_VIDEO_KEY not configured")
    end = start + duration
    payload = {
        "title": clip_name,
        "source": source_url,
        "clip": {
            "startTimecode": _seconds_to_timecode(start),
            "endTimecode": _seconds_to_timecode(end),
        },
    }
    resp = requests.post(
        "https://ws.api.video/videos",
        headers={
            "Authorization": f"Bearer {API_VIDEO_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=(10, 60),
    )
    if resp.status_code not in (200, 201, 202):
        raise RuntimeError(f"api.video create failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    assets = data.get("assets", {}) if isinstance(data, dict) else {}
    return {
        "clip_id": data.get("videoId") if isinstance(data, dict) else None,
        "clip_url": assets.get("player") or assets.get("hls") or assets.get("mp4"),
    }


def create_apivideo_clip_safe(
    source_url: str,
    clip_name: str,
    start: float,
    duration: float,
    video_id: str,
) -> Optional[dict]:
    try:
        return create_apivideo_clip(source_url, clip_name, start, duration)
    except Exception as exc:
        logger.exception("[%s] api.video clip failed for %s: %s", video_id, clip_name, exc)
        return None


def attach_clip_assets(
    clips_payload: dict,
    video_id: str,
    youtube_url: Optional[str],
) -> dict:
    segments = clips_payload.get("segments", [])
    if not segments:
        return clips_payload
    if API_VIDEO_KEY and youtube_url:
        for idx, segment in enumerate(segments, start=1):
            start = float(segment.get("start", 0.0))
            duration = float(segment.get("duration", 0.0))
            end = segment.get("end")
            if duration <= 0 and end is not None:
                duration = max(0.0, float(end) - start)
            if duration <= 0:
                continue
            clip_name = f"{video_id}_{idx:02d}"
            clip_info = create_apivideo_clip_safe(
                youtube_url,
                clip_name,
                start,
                duration,
                video_id,
            )
            if not clip_info:
                continue
            segment["clip_name"] = clip_name
            segment["clip_url"] = clip_info.get("clip_url")
        return clips_payload

    workdir = Path("/tmp") / f"clips_{video_id}"
    video_path = download_youtube_video(video_id, youtube_url, workdir)
    for idx, segment in enumerate(segments, start=1):
        start = float(segment.get("start", 0.0))
        duration = float(segment.get("duration", 0.0))
        end = segment.get("end")
        if duration <= 0 and end is not None:
            duration = max(0.0, float(end) - start)
        if duration <= 0:
            continue
        clip_name = f"{video_id}_{idx:02d}.mp4"
        output_path = workdir / clip_name
        create_clip_file(video_path, start, duration, output_path)
        clip_info = upload_clip_to_drive(output_path, clip_name)
        segment["clip_name"] = clip_name
        segment["clip_url"] = clip_info["clip_url"]
    return clips_payload


def openai_clip_prompt(transcript_segments: List[dict], prompt_override: Optional[str]) -> str:
    base_prompt = prompt_override or (
        "TASK\n"
        "You are a highlight editor for ANY type of content. Review the ENTIRE transcript in chronological order "
        "and select the best short-form clips.\n"
        "Return a MAX of 20 clips, each 10–90 seconds, prioritized by engagement and standalone clarity.\n\n"
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
        "- Do NOT paraphrase, rewrite, infer missing context, or fabricate.\n"
        "- Avoid duplicates/near-duplicates; maximize variety.\n\n"
        "PROCESS\n"
        "1) First pass: determine main_theme + 3–8 key ideas.\n"
        "2) Second pass: pick clips that best support those ideas AND will perform on social.\n"
        "3) Categorize each clip using the MASTER CATEGORY LIST.\n\n"
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
        "    \"main_theme\": \"string\",\n"
        "    \"key_ideas\": [\"string\", \"string\"]\n"
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
        "{ \"analysis\": {\"main_theme\": \"\", \"key_ideas\": []}, \"segments\": [] }\n"
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
        "temperature": 0.2,
    }
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=(10, 180),
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI API error ({resp.status_code}): {resp.text}")
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
