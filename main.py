import os
import requests
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# -------------------------
# APP
# -------------------------
app = FastAPI()

# -------------------------
# HEALTH / PING
# -------------------------
@app.get("/ping")
def ping():
    return {"pong": True}

# -------------------------
# ENV
# -------------------------
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")
RAPIDAPI_HOST = "youtube-transcripts-transcribe-youtube-video-to-text.p.rapidapi.com"

if not RAPIDAPI_KEY:
    print("WARNING: RAPIDAPI_KEY not set")

# -------------------------
# MODELS
# -------------------------
class DiscoverRequest(BaseModel):
    video_id: str
    duration: Optional[float] = None
    callback_url: str

# -------------------------
# TRANSCRIPT (RapidAPI)
# -------------------------
def get_transcript(video_id: str) -> List[dict]:
    if not RAPIDAPI_KEY:
        raise RuntimeError("RAPIDAPI_KEY not configured")

    url = "https://youtube-transcripts-transcribe-youtube-video-to-text.p.rapidapi.com/transcribe"

    payload = {
        "url": f"https://www.youtube.com/watch?v={video_id}"
    }

    headers = {
        "Content-Type": "application/json",
        "x-rapidapi-host": RAPIDAPI_HOST,
        "x-rapidapi-key": RAPIDAPI_KEY,
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=30)

    if resp.status_code != 200:
        raise RuntimeError(f"Transcript API failed: {resp.text}")

    data = resp.json()

    transcript = data.get("transcript", [])
    if not transcript:
        raise RuntimeError("Transcript returned empty")

    segments = []
    for t in transcript:
        segments.append({
            "start": float(t["start"]),
            "duration": float(t["duration"]),
            "text": t["text"].strip()
        })

    return segments

# -------------------------
# CHUNKING (safe for 3hr vids)
# -------------------------
def chunk_transcript(segments, chunk_seconds=600):
    chunks = []
    current = []
    current_start = segments[0]["start"]

    total = 0
    for s in segments:
        if total + s["duration"] > chunk_seconds:
            chunks.append({
                "start": current_start,
                "segments": current
            })
            current = []
            current_start = s["start"]
            total = 0

        current.append(s)
        total += s["duration"]

    if current:
        chunks.append({
            "start": current_start,
            "segments": current
        })

    return chunks

# -------------------------
# CALLBACK
# -------------------------
def post_callback(url: str, payload: dict):
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print("Callback failed:", e)

# -------------------------
# DISCOVER ENDPOINT
# -------------------------
@app.post("/discover")
def discover(req: DiscoverRequest):
    if not RAPIDAPI_KEY:
        raise HTTPException(
            status_code=500,
            detail="RAPIDAPI_KEY not configured"
        )

    job_id = str(uuid.uuid4())

    try:
        transcript = get_transcript(req.video_id)
        chunks = chunk_transcript(transcript)

        post_callback(req.callback_url, {
            "job_id": job_id,
            "video_id": req.video_id,
            "status": "transcript_ready",
            "chunk_count": len(chunks),
            "chunks": chunks
        })

        return {
            "status": "accepted",
            "job_id": job_id,
            "chunks": len(chunks)
        }

   except Exception as e:
    print("DISCOVER FAILED:", repr(e))

    post_callback(req.callback_url, {
        "job_id": job_id,
        "video_id": req.video_id,
        "status": "error",
        "error": str(e)
    })

    raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# ROOT HEALTH
# -------------------------
@app.get("/")
def health():
    return {"status": "ok"}
