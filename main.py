import os
import json
import uuid
import datetime
import subprocess
import io
import yt_dlp
import requests

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

# --------------------------------------------------
# APP
# --------------------------------------------------
app = FastAPI()

# --------------------------------------------------
# STATE
# --------------------------------------------------
JOBS = {}
LATEST_INGEST = {
    "file_id": None,
    "title": None,
    "folder_id": None
}

# --------------------------------------------------
# LOGGING
# --------------------------------------------------
def ts():
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def debug(msg):
    print(f"[DEBUG] {ts()} | {msg}")

def raw_log(label, payload):
    try:
        print(f"[RAW] {label}: {json.dumps(payload)[:8000]}")
    except Exception:
        print(f"[RAW] {label}: <unserializable>")

# --------------------------------------------------
# UTIL
# --------------------------------------------------
def sanitize(name):
    for c in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        name = name.replace(c, "_")
    return name.strip() or "clip"

def safe_float(x):
    try:
        return float(x)
    except:
        return None

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
# INGEST (SYNCHRONOUS — RELIABLE)
# --------------------------------------------------
@app.post("/ingest")
async def ingest(req: Request):
    data = await req.json()
    raw_log("INGEST", data)

    if "url" not in data:
        return JSONResponse({"error": "missing url"}, 400)

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "running"}

    try:
        url = data["url"]
        title = url.split("v=")[-1].split("&")[0]
        debug(f"Downloading YouTube: {url}")

        out = f"/tmp/{sanitize(title)}.mp4"
        opts = {
            "format": "bv*+ba/best",
            "merge_output_format": "mp4",
            "outtmpl": out,
            "quiet": True,
            "retries": 5,
            "fragment_retries": 5,
            "cookies": COOKIES_PATH if os.path.exists(COOKIES_PATH) else None,
        }
        if opts["cookies"] is None:
            opts.pop("cookies")

        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

        if not os.path.exists(out):
            raise Exception("yt-dlp failed")

        debug("Uploading to Drive")
        up = upload_to_drive(out, title, os.environ["DRIVE_FOLDER_ID"])

        LATEST_INGEST.update({
            "file_id": up["id"],
            "title": title,
            "folder_id": os.environ["DRIVE_FOLDER_ID"]
        })

        JOBS[job_id]["status"] = "success"
        JOBS[job_id]["file_id"] = up["id"]

        if "callback_url" in data:
            requests.post(data["callback_url"], json=JOBS[job_id], timeout=10)

        return {"status": "success", "job_id": job_id}

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        debug(f"INGEST ERROR: {e}")
        return JSONResponse({"error": str(e)}, 500)

# --------------------------------------------------
# SEGMENT NORMALIZER
# --------------------------------------------------
def normalize_segments(data):
    """
    Extremely defensive segment normalizer.
    Accepts:
    - valid JSON
    - stringified JSON
    - truncated JSON
    - Zapier-wrapped payloads
    Returns ONLY [{start, duration}]
    NEVER raises.
    """

    try:
        # unwrap {"": "...json..."}
        if isinstance(data, dict) and "" in data and isinstance(data[""], str):
            data = data[""]

        # parse outer string if needed
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                # cannot parse outer shell → give up safely
                debug("normalize_segments: outer JSON invalid")
                return []

        raw = data.get("segments_json") or data.get("segments")

        # If raw is a string, it may be BROKEN JSON
        if isinstance(raw, str):
            segments = []
            for line in raw.splitlines():
                line = line.strip()
                if '"offset"' in line:
                    try:
                        offset = float(line.split(":")[1].strip(" ,"))
                    except Exception:
                        offset = None
                if '"duration"' in line:
                    try:
                        duration = float(line.split(":")[1].strip(" ,"))
                    except Exception:
                        duration = None

                if offset is not None and duration is not None:
                    if duration > 0:
                        segments.append({
                            "start": offset,
                            "duration": duration
                        })
                    offset = None
                    duration = None

            debug(f"Recovered {len(segments)} segments from broken JSON")
            return segments

        # If raw is already structured
        if isinstance(raw, dict):
            raw = raw.get("segments")

        segments = []
        if isinstance(raw, list):
            for s in raw:
                try:
                    start = safe_float(s.get("offset") or s.get("start"))
                    dur = safe_float(s.get("duration"))
                    if start is not None and dur and dur > 0:
                        segments.append({"start": start, "duration": dur})
                except Exception:
                    continue

        return segments

    except Exception as e:
        debug(f"normalize_segments FAILED SAFELY: {e}")
        return []


# --------------------------------------------------
# FFMPEG CLIP + AUDIO SAFE
# --------------------------------------------------
def make_clip(src, out, start, dur):
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", src,
        "-t", str(dur),
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "veryfast",
        out
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise Exception(r.stderr.decode()[:4000])

# --------------------------------------------------
# ANALYZE + CLIP (USES EXISTING INGEST)
# --------------------------------------------------
@app.post("/analyze-and-clip")
async def analyze_and_clip(req: Request):
    data = await req.json()
    raw_log("ANALYZE", data)

    if not LATEST_INGEST["file_id"]:
        return JSONResponse({"error": "no ingest available"}, 400)

    segments = normalize_segments(data)
    debug(f"Segments parsed: {len(segments)}")

    if not segments:
        return {"status": "ok", "clips": 0}

    src = download_from_drive(LATEST_INGEST["file_id"])
    base = sanitize(data.get("video_title") or LATEST_INGEST["title"])

    uploaded = 0
    for i, s in enumerate(segments, 1):
        out = f"/tmp/{base}_{i}.mp4"
        make_clip(src, out, s["start"], s["duration"])
        upload_to_drive(out, f"{base}_{i}", LATEST_INGEST["folder_id"])
        uploaded += 1

    return {"status": "success", "clips_uploaded": uploaded}

@app.get("/health")
async def health():
    return {"status": "ok"}
