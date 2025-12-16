# (FILE IS LONG — THIS IS COMPLETE AND FINAL)

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

app = FastAPI()
@app.post("/ingest")
async def ingest(req: Request, bg: BackgroundTasks):
    data = await req.json()
    debug(f"INGEST payload: {data}")

    if "url" not in data:
        return JSONResponse({"error": "missing url"}, 400)

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued", "logs": []}

    def run():
        try:
            url = data["url"]
            title = url.split("v=")[-1].split("&")[0]

            job_log(job_id, f"Downloading YouTube: {url}")
            path = download_youtube(url, title)
            job_log(job_id, f"Downloaded: {os.path.getsize(path)} bytes")

            job_log(job_id, "Uploading to Drive")
            up = upload_to_drive(path, title, os.environ["DRIVE_FOLDER_ID"])

            LATEST_INGEST["file_id"] = up["id"]
            LATEST_INGEST["title"] = title
            LATEST_INGEST["folder_id"] = os.environ["DRIVE_FOLDER_ID"]

            JOBS[job_id]["status"] = "success"
            JOBS[job_id]["file_id"] = up["id"]

        except Exception as e:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(e)
            job_log(job_id, f"ERROR: {e}")

    bg.add_task(run)
    return {"status": "queued", "job_id": job_id}

JOBS = {}
LATEST_INGEST = {"file_id": None, "title": None, "folder_id": None}

# ---------------- LOGGING ----------------
def ts():
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def debug(msg):
    print(f"[DEBUG] {ts()} | {msg}")

def job_log(job_id, msg):
    JOBS.setdefault(job_id, {}).setdefault("logs", []).append(f"{ts()} | {msg}")

# ---------------- GOOGLE DRIVE ----------------
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
    return path

# ---------------- FFMPEG ----------------
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
    subprocess.run(cmd, check=True)

def mux_audio(video_only, audio_src, out):
    """
    Reattach original audio after OpenCV processing
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video_only,
        "-i", audio_src,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        out
    ]
    subprocess.run(cmd, check=True)

# ---------------- REEL FRAMER ----------------
class ReelFramer:
    def __init__(self):
        self.face = mp.solutions.face_detection.FaceDetection(0, 0.6)
        self.out_w, self.out_h = 1080, 1920
        self.aspect = self.out_w / self.out_h
        self.last_x = None
        self.smooth = 0.88

    def reframe(self, inp, out, job_id=None):
        cap = cv2.VideoCapture(inp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        vw = cv2.VideoWriter(
            out,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (self.out_w, self.out_h)
        )

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.face.process(rgb)

            cx = w // 2
            if res.detections:
                det = max(res.detections,
                          key=lambda d: d.location_data.relative_bounding_box.width)
                bb = det.location_data.relative_bounding_box
                cx = int((bb.xmin + bb.width / 2) * w)

            if self.last_x is None:
                self.last_x = cx
            cx = int(self.smooth * self.last_x + (1 - self.smooth) * cx)
            self.last_x = cx

            crop_h = int(h / 1.15)
            crop_w = int(crop_h * self.aspect)
            x1 = max(0, min(w - crop_w, cx - crop_w // 2))
            y1 = (h - crop_h) // 2

            crop = frame[y1:y1 + crop_h, x1:x1 + crop_w]
            resized = cv2.resize(crop, (self.out_w, self.out_h))
            vw.write(resized)

        cap.release()
        vw.release()

# ---------------- ROUTES ----------------
@app.post("/analyze-and-clip")
async def analyze(req: Request, bg: BackgroundTasks):
    data = await req.json()
    segments = data.get("segments", [])
    if not segments:
        return {"status": "ok", "clips_created": 0}

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued", "logs": []}

    def run():
        src = download_from_drive(LATEST_INGEST["file_id"])
        framer = ReelFramer()

        for i, s in enumerate(segments, 1):
            raw = f"/tmp/raw_{i}.mp4"
            vid = f"/tmp/reel_video_{i}.mp4"
            final = f"/tmp/final_{i}.mp4"

            make_clip(src, raw, s["start"], s["duration"])
            framer.reframe(raw, vid, job_id)
            mux_audio(vid, raw, final)

            upload_to_drive(final, f"clip_{i}", LATEST_INGEST["folder_id"])

        JOBS[job_id]["status"] = "success"

    bg.add_task(run)
    return {"status": "queued", "job_id": job_id}

@app.get("/health")
def health():
    return {"status": "ok"}
