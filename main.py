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


def openai_clip_prompt(transcript_segments: List[dict], prompt_override: Optional[str]) -> str:
    base_prompt = prompt_override or """TASK
You are a highlight editor for ANY type of content (not just faith). Review the ENTIRE transcript and:
1) Identify the main theme and 3–8 key ideas.
2) Select up to 20 clip-worthy segments (10–90s) that best represent those ideas and will perform on social.
3) Categorize each selected clip using the MASTER CATEGORY LIST below.

HARD REQUIREMENTS
- Each chosen clip MUST be 10–90 seconds long.
- Each clip MUST be a complete, standalone thought (no cut-off setup, mid-sentence starts, or missing payoff).
- Use transcript offsets EXACTLY as provided.
- Do NOT paraphrase, rewrite, infer missing context, or fabricate.
- No duplicates or near-duplicates; maximize variety across picks.

HOW TO WORK (IMPORTANT)
A) FIRST PASS: Produce a short transcript understanding:
   - main_theme: 1 sentence
   - key_ideas: 3–8 bullets
B) SECOND PASS: Choose clips that strongly support the main_theme/key_ideas and have high engagement potential.
C) THIRD PASS: Categorize each clip using ONLY the master categories, and briefly justify why it will perform.

MASTER CATEGORY LIST (choose 1 primary, optional 0–2 secondary)
- inspiration: uplifting, motivating, empowering, hopeful, emotional resonance
- education: teaches a concept, explains how/why, tutorial-like, definitions, frameworks
- humor: funny moments, punchlines, witty observations (must still make sense standalone)
- story: narrative/testimony/anecdote with a clear arc or point
- insight: counterintuitive idea, strong opinion, reframing, “aha” moment
- call-to-action: direct challenge, next steps, invitation to act/try/reflect
- controversy: debate-worthy take, strong stance, surprising claim (not misinformation)
- behind-the-scenes: process, real-time problem solving, making-of, candid moments
- social-proof: results, wins, evidence, credibility, before/after, metrics
- empathy: vulnerability, struggle, validation, mental-health/relationship honesty
- mindset: habits, discipline, productivity, resilience, identity, self-talk
- leadership: management, teamwork, values, culture, decision-making
- business: strategy, marketing, sales, money, career, negotiation
- science-tech: data, experiments, tech concepts, engineering, AI, research
- lifestyle: wellness, fitness, food, travel, daily life, routines
- spirituality: faith, purpose, meaning, prayer/meditation, morals
- community: audience interaction, chants, call-and-response, group moments
- quote: a highly quotable one-liner/mantra (use when the clip is primarily a quote)

SELECTION GOAL (pick what will perform)
Select moments that drive engagement because they are:
- Hooky in the first 2–3 seconds
- Clear without extra context
- Emotionally resonant OR immediately useful OR very shareable
- Unique vs other picks (variety across categories and key ideas)

SCORING (0–100)
Score each segment based on:
- Hook strength in first 2–3 seconds (0–30)
- Standalone clarity / completeness (0–25)
- Impact (emotion/usefulness/novelty) (0–25)
- Shareability / quoteability (0–10)
- Variety contribution vs other picks (0–10)

SEGMENT CONSTRUCTION RULES
- You MAY select a single provided segment if it already fits 10–90 seconds.
- OR you MAY combine ADJACENT segments ONLY if the combined result stays 10–90 seconds and remains one coherent thought.
- Never combine non-adjacent segments.
- When combining, use the earliest start offset and total combined duration.

CONTENT BOUNDARIES
- Avoid filler, long pauses, housekeeping/logistics unless unusually compelling.
- Avoid misleading or medically/legal/financially dangerous advice; do not select clips that promote harm.

OUTPUT FORMAT (STRICT JSON ONLY)
{
  "analysis": {
    "main_theme": "string",
    "key_ideas": ["string", "string"],
    "dominant_categories": ["string", "string", "string"]
  },
  "segments": [
    {
      "start": number,
      "duration": number,
      "score": number,
      "primary_category": "inspiration" | "education" | "humor" | "story" | "insight" | "call-to-action" | "controversy" | "behind-the-scenes" | "social-proof" | "empathy" | "mindset" | "leadership" | "business" | "science-tech" | "lifestyle" | "spirituality" | "community" | "quote",
      "secondary_categories": ["inspiration" | "education" | "humor" | "story" | "insight" | "call-to-action" | "controversy" | "behind-the-scenes" | "social-proof" | "empathy" | "mindset" | "leadership" | "business" | "science-tech" | "lifestyle" | "spirituality" | "community" | "quote"],
      "hook": "the first 5–12 words of the segment",
      "reason": "1 short sentence why this will perform",
      "key_idea": "which key idea this clip supports (must match one of analysis.key_ideas)"
    }
  ]
}

If nothing qualifies:
{
  "analysis": {"main_theme": "", "key_ideas": [], "dominant_categories": []},
  "segments": []
}
"""
    transcript_payload = [
        {"offset": seg["start"], "duration": seg["duration"], "text": seg["text"]}
        for seg in transcript_segments
    ]
    return f"{base_prompt}\n\nTranscript Segments:\n{json.dumps(transcript_payload, ensure_ascii=False)}"


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
        text = transcript_lookup.get(start, "")
        values.append([
            video_id,
            start,
            duration,
            text,
            segment.get("score"),
            segment.get("primary_category") or segment.get("category"),
            segment.get("reason"),
        ])
    if not values:
        values = [[video_id, "", "", "", "", "", "No segments returned"]]

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
                clips_payload = call_openai_for_clips(transcript, JOBS[job_id].get("prompt"))
            except Exception as exc:
                logger.exception("[%s] clip discovery failed, continuing with empty results", job_id)
                clips_payload = {"segments": [], "error": str(exc)}
        clip_segments = clips_payload.get("segments", [])
        logger.info("[%s] clips=%s", job_id, len(clip_segments))

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
