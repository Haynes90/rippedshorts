import os
import json
import uuid
import datetime
import subprocess
import io
import yt_dlp
import requests

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

import cv2
import numpy as np
import mediapipe as mp


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


def job_log(job_id, msg):
    JOBS.setdefault(job_id, {}).setdefault("logs", []).append(f"{ts()} | {msg}")


# --------------------------------------------------
# UTIL
# --------------------------------------------------
def sanitize(name: str) -> str:
    for c in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        name = name.replace(c, "_")
    return (name.strip() or "clip")


def safe_float(x):
    try:
        return float(x)
    except Exception:
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
# INGEST (UNCHANGED)
# --------------------------------------------------
def download_youtube(url, title):
    out = f"/tmp/{sanitize(title)}.mp4"
    opts = {
        "format": "bv*+ba/best",
        "merge_output_format": "mp4",
        "outtmpl": out,
        "cookies": COOKIES_PATH if os.path.exists(COOKIES_PATH) else None,
        "quiet": True
    }
    if opts["cookies"] is None:
        opts.pop("cookies")

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    if not os.path.exists(out) or os.path.getsize(out) == 0:
        raise Exception("YouTube download failed")
    return out


def process_ingest(job_id, url, title, callback):
    JOBS[job_id] = {"status": "queued", "logs": []}
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


# --------------------------------------------------
# SEGMENT NORMALIZER
# --------------------------------------------------
def _extract_json_blob_from_weird_payload(data):
    if isinstance(data, dict) and "" in data and isinstance(data[""], str):
        try:
            return json.loads(data[""])
        except Exception:
            return None
    return data


def normalize_segments(data):
    data = _extract_json_blob_from_weird_payload(data) or data
    raw = data.get("segments_json") or data.get("segments")

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []

    if isinstance(raw, dict):
        raw = raw.get("segments")

    if not isinstance(raw, list):
        return []

    segments = []
    for s in raw:
        st = safe_float(s.get("start"))
        du = safe_float(s.get("duration"))
        if st is not None and du and du > 0:
            segments.append({"start": st, "duration": du})
    return segments


# --------------------------------------------------
# FFMPEG HELPERS
# --------------------------------------------------
def make_clip(src, out, start, dur):
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", src,
        "-t", str(dur),
        "-c:v", "libx264",
        "-c:a", "aac",
        out
    ]
    subprocess.check_call(cmd)


def mux_audio(video_path, audio_source_path, out_path):
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_source_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "160k",
        "-movflags", "+faststart",
        out_path
    ]
    subprocess.check_call(cmd)


# --------------------------------------------------
# REEL FRAMER (UNCHANGED)
# --------------------------------------------------
class ReelFramer:
    def __init__(self):
        self.face = mp.solutions.face_detection.FaceDetection(0, 0.6)
        self.out_w, self.out_h = 1080, 1920
        self.aspect = self.out_w / self.out_h

    def reframe_to_reel(self, in_path, out_path):
        cap = cv2.VideoCapture(in_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        out = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (self.out_w, self.out_h)
        )

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            crop_w = int(h * self.aspect)
            x1 = max(0, (w - crop_w) // 2)
            crop = frame[:, x1:x1 + crop_w]
            out.write(cv2.resize(crop, (self.out_w, self.out_h)))

        cap.release()
        out.release()


# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.post("/ingest")
async def ingest(req: Request, bg: BackgroundTasks):
    data = await req.json()
    job_id = str(uuid.uuid4())
    bg.add_task(process_ingest, job_id, data["url"], data["url"], data.get("callback_url"))
    return {"status": "queued", "job_id": job_id}


@app.post("/analyze-and-clip")
async def analyze_and_clip(req: Request, bg: BackgroundTasks):
    data = await req.json()
    segments = normalize_segments(data)

    if not segments:
        return {"status": "ok", "clips_created": 0}

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued", "clips": []}

    def run():
        src = download_from_drive(LATEST_INGEST["file_id"])
        framer = ReelFramer()
        base = sanitize(LATEST_INGEST["title"])

        for i, s in enumerate(segments, 1):
            raw_clip = f"/tmp/{base}_{i}.mp4"
            reel_tmp = f"/tmp/{base}_{i}_reel.mp4"
            final = f"/tmp/{base}_{i}_final.mp4"

            make_clip(src, raw_clip, s["start"], s["duration"])
            framer.reframe_to_reel(raw_clip, reel_tmp)
            mux_audio(reel_tmp, raw_clip, final)

            os.replace(final, raw_clip)
            up = upload_to_drive(raw_clip, f"{base}_{i}", LATEST_INGEST["folder_id"])
            JOBS[job_id]["clips"].append(up["id"])

        JOBS[job_id]["status"] = "success"

    bg.add_task(run)
    return {"status": "queued", "job_id": job_id}


@app.get("/health")
async def health():
    return {"status": "ok"}
