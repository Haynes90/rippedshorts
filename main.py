import os
import json
import uuid
import subprocess
import io
import yt_dlp
import requests

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

# --------------------------------------------------
# APP + STATE
# --------------------------------------------------
app = FastAPI()

JOBS = {}
LATEST_INGEST = {
    "file_id": None,
    "title": None,
    "folder_id": None
}

# --------------------------------------------------
# UTIL
# --------------------------------------------------
def raw_log(label, payload):
    try:
        print(f"[RAW] {label}: {json.dumps(payload)[:8000]}")
    except:
        print(f"[RAW] {label}: <unserializable>")

def sanitize(name):
    for c in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        name = name.replace(c, "_")
    return name.strip() or "clip"

# --------------------------------------------------
# SEGMENT EXTRACTION (FINAL, ROBUST)
# --------------------------------------------------
def extract_segments(obj, found=None):
    """
    Recursively walk ANY structure and extract
    all {start, duration} objects.
    """
    if found is None:
        found = []

    if isinstance(obj, dict):
        # Direct match
        if "start" in obj and "duration" in obj:
            try:
                found.append({
                    "start": float(obj["start"]),
                    "duration": float(obj["duration"])
                })
            except:
                pass

        # Recurse into values
        for v in obj.values():
            extract_segments(v, found)

    elif isinstance(obj, list):
        for item in obj:
            extract_segments(item, found)

    elif isinstance(obj, str):
        # Try parsing JSON strings
        try:
            parsed = json.loads(obj)
            extract_segments(parsed, found)
        except:
            pass

    return found

# --------------------------------------------------
# COOKIES
# --------------------------------------------------
COOKIES_PATH = "/app/cookies.txt"
if "YOUTUBE_COOKIES" in os.environ:
    with open(COOKIES_PATH, "w") as f:
        f.write(os.environ["YOUTUBE_COOKIES"])

# --------------------------------------------------
# GOOGLE DRIVE
# --------------------------------------------------
def drive_client():
    creds = service_account.Credentials.from_service_account_info(
        json.loads(os.environ["GOOGLE_CREDENTIALS"]),
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)

def upload_to_drive(path, name, folder_id):
    drive = drive_client()
    media = MediaFileUpload(path, mimetype="video/mp4", resumable=True)
    req = drive.files().create(
        body={"name": f"{name}.mp4", "parents": [folder_id]},
        media_body=media,
        fields="id",
        supportsAllDrives=True
    )
    res = None
    while res is None:
        _, res = req.next_chunk()
    return res

def download_from_drive(file_id):
    drive = drive_client()
    req = drive.files().get_media(fileId=file_id, supportsAllDrives=True)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, req)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    path = f"/tmp/{file_id}.mp4"
    with open(path, "wb") as f:
        f.write(fh.getvalue())
    if os.path.getsize(path) == 0:
        raise Exception("Drive video empty")
    return path

# --------------------------------------------------
# INGEST
# --------------------------------------------------
def download_youtube(url, title):
    out = f"/tmp/{sanitize(title)}.mp4"
    opts = {
        "format": "bv*+ba/best",
        "merge_output_format": "mp4",
        "outtmpl": out,
        "cookies": COOKIES_PATH,
        "quiet": True,
        "extractor_args": {"youtube": {"player_client": ["android"]}}
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    if not os.path.exists(out) or os.path.getsize(out) == 0:
        raise Exception("YouTube download failed")
    return out

def process_ingest(job_id, url, title, callback):
    try:
        path = download_youtube(url, title)
        up = upload_to_drive(path, title, os.environ["DRIVE_FOLDER_ID"])
        LATEST_INGEST.update({
            "file_id": up["id"],
            "title": title,
            "folder_id": os.environ["DRIVE_FOLDER_ID"]
        })
        JOBS[job_id]["status"] = "success"
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)

    if callback:
        requests.post(callback, json=JOBS[job_id], timeout=10)

@app.post("/ingest")
async def ingest(req: Request, bg: BackgroundTasks):
    data = await req.json()
    raw_log("INGEST", data)

    if "url" not in data:
        return JSONResponse({"error": "missing url"}, 400)

    title = data["url"].split("v=")[-1].split("&")[0]
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued"}

    bg.add_task(process_ingest, job_id, data["url"], title, data.get("callback_url"))
    return {"status": "queued", "job_id": job_id}

# --------------------------------------------------
# ANALYZE + CLIP
# --------------------------------------------------
@app.post("/analyze-and-clip")
async def analyze_and_clip(req: Request, bg: BackgroundTasks):
    data = await req.json()
    raw_log("ANALYZE", data)

    if not LATEST_INGEST["file_id"]:
        return JSONResponse({"error": "no ingest available"}, 400)

    segments = extract_segments(data)
    print(f"[DEBUG] Segments extracted: {len(segments)}")
    print(f"[DEBUG] First segment: {segments[0] if segments else None}")

    if not segments:
        return {"clips_found": 0}

    def run():
        try:
            print("[DEBUG] Background clip task started")
            src = download_from_drive(LATEST_INGEST["file_id"])
            base = sanitize(data.get("video_title") or LATEST_INGEST["title"])

            for i, s in enumerate(segments, 1):
                print(f"[DEBUG] Clipping #{i}: start={s['start']} dur={s['duration']}")
                out = f"/tmp/{base}_{i}.mp4"

                subprocess.run([
                    "ffmpeg", "-y",
                    "-ss", str(s["start"]),
                    "-i", src,
                    "-t", str(s["duration"]),
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    out
                ], check=True)

                up = upload_to_drive(out, f"{base}_{i}", LATEST_INGEST["folder_id"])
                print(f"[DEBUG] Uploaded clip #{i}: {up['id']}")

            if data.get("callback_url"):
                requests.post(data["callback_url"], json={
                    "status": "success",
                    "clips_created": len(segments),
                    "clips_uploaded": len(segments)
                }, timeout=10)

        except Exception as e:
            print(f"[ERROR] Background task failed: {e}")

    bg.add_task(run)
    return {"status": "queued", "clips": len(segments)}

@app.get("/health")
async def health():
    return {"status": "ok"}
