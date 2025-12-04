import os
import json
import uuid
import datetime
import subprocess
import io

import yt_dlp
import requests

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

app = FastAPI()

# --------- SIMPLE IN-MEMORY JOB STORE ----------
JOBS: dict[str, dict] = {}

# Track last ingest for automatic clipping
LATEST_INGEST: dict = {
    "file_id": None,
    "title": None,
    "folder_id": None
}


def log(job_id: str, message: str) -> None:
    if job_id not in JOBS:
        return
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    JOBS[job_id].setdefault("logs", []).append(f"{ts} | {message}")


# --------- COOKIES ----------
COOKIES_PATH = "/app/cookies.txt"
if "YOUTUBE_COOKIES" in os.environ:
    with open(COOKIES_PATH, "w") as f:
        f.write(os.environ["YOUTUBE_COOKIES"])


# --------- GOOGLE DRIVE ----------
def get_drive_client():
    creds = service_account.Credentials.from_service_account_info(
        json.loads(os.environ["GOOGLE_CREDENTIALS"]),
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)


# --------- YOUTUBE DOWNLOAD ----------
def download_video(url: str, title: str, job_id: str | None = None) -> str:
    temp_path = f"/tmp/{title}.mp4"
    opts = {
        "outtmpl": temp_path,
        "format": "mp4",
        "cookies": COOKIES_PATH,
        "quiet": True,
    }
    if job_id:
        log(job_id, "yt-dlp starting download")
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    if job_id:
        log(job_id, f"Download OK → {temp_path}")
    return temp_path


# --------- DRIVE DOWNLOAD ----------
def download_from_drive(file_id: str, job_id: str | None = None) -> str:
    if job_id:
        log(job_id, f"Downloading from Drive: {file_id}")

    drive = get_drive_client()
    req = drive.files().get_media(fileId=file_id, supportsAllDrives=True)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, req)

    done = False
    while not done:
        status, done = downloader.next_chunk()

    local_path = f"/tmp/{file_id}.mp4"
    with open(local_path, "wb") as f:
        f.write(fh.getvalue())

    return local_path


# --------- DRIVE UPLOAD ----------
def upload_to_drive(file_path: str, title: str, folder_id: str, job_id: str | None = None):
    drive = get_drive_client()

    metadata = {
        "name": f"{title}.mp4",
        "parents": [folder_id]
    }
    media = MediaFileUpload(file_path, mimetype="video/mp4", resumable=True)

    req = drive.files().create(
        body=metadata,
        media_body=media,
        fields="id, webViewLink",
        supportsAllDrives=True
    )

    response = None
    while response is None:
        status, response = req.next_chunk()

    if job_id:
        log(job_id, f"Uploaded → {response.get('id')}")

    return response


# --------- SAFE FILENAMES ----------
def sanitize_filename(name: str):
    bad = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for b in bad:
        name = name.replace(b, "_")
    return name.strip() or "clip"


# --------- FFMPEG CLIP ----------
def make_clip(source_path, out_path, start, duration, job_id=None, label=None):
    if job_id:
        log(job_id, f"Clipping {label} @ {start}s for {duration}s")

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", source_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "veryfast",
        out_path
    ]
    sub = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if sub.returncode != 0:
        raise Exception(f"FFmpeg failed: {sub.stderr[-300:]}")


# --------- BACKGROUND JOB: INGEST ----------
def process_job(job_id, url, title, callback_url):
    try:
        JOBS[job_id]["status"] = "downloading"
        video_path = download_video(url, title, job_id)

        JOBS[job_id]["status"] = "uploading"
        upload_res = upload_to_drive(video_path, title, os.environ["DRIVE_FOLDER_ID"], job_id)

        JOBS[job_id]["status"] = "success"
        JOBS[job_id]["file_id"] = upload_res["id"]

        LATEST_INGEST["file_id"] = upload_res["id"]
        LATEST_INGEST["title"] = title
        LATEST_INGEST["folder_id"] = os.environ["DRIVE_FOLDER_ID"]

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)

    if callback_url:
        try:
            requests.post(callback_url, json=JOBS[job_id], timeout=10)
        except:
            pass


# --------- BACKGROUND JOB: CLIPS ----------
def process_clip_job(job_id, payload):
    drive_file_id = payload["drive_file_id"]
    folder_id = payload["folder_id"]
    title = payload.get("video_title") or drive_file_id
    clips = payload["clips"]
    callback = payload.get("callback_url")

    source = download_from_drive(drive_file_id, job_id)

    JOBS[job_id]["clips"] = []

    base = sanitize_filename(title)

    for clip in clips:
        idx = int(clip["index"])
        start = float(clip["start"])
        duration = float(clip["duration"])

        clip_name = f"{base}__clip_{idx:02d}"
        out_path = f"/tmp/{clip_name}.mp4"

        try:
            make_clip(source, out_path, start, duration, job_id, f"clip_{idx}")
            up = upload_to_drive(out_path, clip_name, folder_id, job_id)

            JOBS[job_id]["clips"].append({
                "index": idx,
                "file_id": up["id"],
                "webViewLink": up["webViewLink"]
            })

        except Exception as e:
            JOBS[job_id].setdefault("clip_errors", []).append(str(e))

    JOBS[job_id]["status"] = (
        "success" if "clip_errors" not in JOBS[job_id] else "partial_success"
    )

    if callback:
        requests.post(callback, json=JOBS[job_id])


# --------- /INGEST ----------
@app.post("/ingest")
async def ingest(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    url = data.get("url")
    if not url:
        return {"error": "Missing URL"}

    title = url.split("v=")[-1].split("&")[0]
    cb = data.get("callback_url")

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"job_id": job_id, "status": "queued"}

    background_tasks.add_task(process_job, job_id, url, title, cb)

    return {"status": "queued", "job_id": job_id}


# --------- /CLIPS ----------
@app.post("/clips")
async def clips(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()

    if not data.get("clips"):
        return JSONResponse({"error": "Missing or empty clips list"}, status_code=400)

    if not LATEST_INGEST["file_id"]:
        return JSONResponse({"error": "No ingest has been processed yet"}, status_code=400)

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"job_id": job_id, "status": "queued"}

    data["drive_file_id"] = LATEST_INGEST["file_id"]
    data["folder_id"] = LATEST_INGEST["folder_id"]
    data["video_title"] = data.get("video_title") or LATEST_INGEST["title"]

    background_tasks.add_task(process_clip_job, job_id, data)

    return {"status": "queued", "job_id": job_id}


# --------- /STATUS ----------
@app.get("/status/{job_id}")
async def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    return job


# --------- /HEALTH ----------
@app.get("/health")
async def health():
    return {"status": "ok"}


# --------- /ANALYZE-AND-CLIP ----------
@app.post("/analyze-and-clip")
async def analyze_and_clip(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()

    # FIX: segments_json is now an ARRAY, not a string
    segments_json = data.get("segments_json")
    if not isinstance(segments_json, list):
        return JSONResponse({"error": "segments_json must be a JSON array"}, status_code=400)

    video_total_len_sec = data["video_total_len_sec"]
    min_clip_len_sec = data["min_clip_len_sec"]
    max_clip_len_sec = data["max_clip_len_sec"]
    max_clip_count = data["max_clip_count"]
    drive_file_id = data["drive_file_id"]
    folder_id = data["folder_id"]
    video_title = data.get("video_title") or "video"
    callback_url = data.get("callback_url")

    # ----- Build prompt -----
    prompt = f"""
You are an expert short-form editor.
Choose up to {max_clip_count} clips.

Rules:
- Use segments_json (provided separately below)
- Use only time ranges within {video_total_len_sec} seconds
- Each clip must be between {min_clip_len_sec} and {max_clip_len_sec} seconds
- Prefer hooks, educational beats, strong takeaways
- Avoid intros/outros
Return ONLY valid JSON with:
{{
  "clips": [
    {{
      "title": "string",
      "hook": "string",
      "summary": "string",
      "keywords_csv": "string",
      "start_sec": 0,
      "end_sec": 0,
      "duration_sec": 0,
      "confidence": 0.0,
      "reason": "string"
    }}
  ]
}}
"""

    # ----- OpenAI -----
    import time
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        return {"error": "Missing OPENAI_API_KEY"}

    retries = 4
    result = None
    last_err = None

    for attempt in range(retries):
        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": "Return ONLY valid JSON."},
                        {"role": "user", "content": prompt},
                        {"role": "user", "content": json.dumps(segments_json)}
                    ]
                },
                timeout=45
            )
            if r.status_code == 429:
                last_err = r.text
                time.sleep(2 + attempt)
                continue

            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            result = json.loads(content)
            break

        except Exception as e:
            last_err = str(e)
            time.sleep(2)

    if result is None:
        return {"error": f"OpenAI failed: {last_err}"}

    clips = result.get("clips", [])
    if not clips:
        return {"error": "OpenAI returned no clips"}

    formatted = []
    for i, clip in enumerate(clips, start=1):
        formatted.append({
            "index": i,
            "start": clip["start_sec"],
            "duration": clip["duration_sec"],
            "name": f"{video_title}_clip_{i}"
        })

    payload = {
        "drive_file_id": drive_file_id,
        "folder_id": folder_id,
        "video_title": video_title,
        "clips": formatted,
        "callback_url": callback_url
    }

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"job_id": job_id, "status": "queued"}

    background_tasks.add_task(process_clip_job, job_id, payload)

    return {"status": "queued", "job_id": job_id, "clips_found": len(formatted)}
