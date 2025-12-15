import os
import json
import uuid
import subprocess
import io
import yt_dlp
import requests
import cv2
import mediapipe as mp
import numpy as np

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

# --------------------------------------------------
# APP + STATE
# --------------------------------------------------
app = FastAPI()

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
        print(f"[RAW] {label}: {json.dumps(payload)[:6000]}")
    except:
        print(f"[RAW] {label}: <unserializable>")

def sanitize(name):
    for c in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        name = name.replace(c, "_")
    return name.strip() or "clip"

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
        raise Exception("Downloaded Drive file empty")
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
        "quiet": True,
        "extractor_args": {"youtube": {"player_client": ["android"]}}
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    if not os.path.exists(out) or os.path.getsize(out) == 0:
        raise Exception("YouTube download failed")
    return out

def process_ingest(url, title):
    path = download_youtube(url, title)
    up = upload_to_drive(path, title, os.environ["DRIVE_FOLDER_ID"])
    LATEST_INGEST.update({
        "file_id": up["id"],
        "title": title,
        "folder_id": os.environ["DRIVE_FOLDER_ID"]
    })

@app.post("/ingest")
async def ingest(req: Request, bg: BackgroundTasks):
    data = await req.json()
    raw_log("INGEST", data)

    title = data["url"].split("v=")[-1].split("&")[0]
    bg.add_task(process_ingest, data["url"], title)
    return {"status": "queued"}

# --------------------------------------------------
# STEP 2 — FACE-AWARE REELIFY
# --------------------------------------------------
def reelify_face_follow(input_path: str, output_path: str):
    """
    Create a 9:16 reel that follows the dominant face.
    Falls back to center crop if no face is found.
    """

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    crop_h = height
    crop_w = int(crop_h * 9 / 16)

    face_centers = []

    mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )

    frame_idx = 0
    sample_every = int(fps // 3) if fps > 0 else 5  # ~3 samples/sec

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mp_face.process(rgb)

            if res.detections:
                # choose largest face
                best = max(
                    res.detections,
                    key=lambda d: d.location_data.relative_bounding_box.width
                )
                bbox = best.location_data.relative_bounding_box
                cx = int((bbox.xmin + bbox.width / 2) * width)
                face_centers.append(cx)

        frame_idx += 1

    cap.release()

    # Determine crop center
    if face_centers:
        center_x = int(np.mean(face_centers))
    else:
        center_x = width // 2  # fallback

    crop_x = max(0, min(center_x - crop_w // 2, width - crop_w))

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", f"crop={crop_w}:{crop_h}:{crop_x}:0,scale=1080:1920",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        output_path
    ]

    subprocess.run(cmd, check=True)

# --------------------------------------------------
# ANALYZE + CLIP
# --------------------------------------------------
@app.post("/analyze-and-clip")
async def analyze_and_clip(req: Request, bg: BackgroundTasks):
    data = await req.json()
    raw_log("ANALYZE", data)

    segments = data.get("segments", [])
    if not segments:
        return {"clips_found": 0}

    def run():
        src = download_from_drive(LATEST_INGEST["file_id"])
        base = sanitize(data.get("video_title") or LATEST_INGEST["title"])

        for i, s in enumerate(segments, 1):
            start = float(s["start"])
            dur = float(s["duration"])

            clip_path = f"/tmp/{base}_{i}.mp4"

            # clip segment
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", src,
                "-t", str(dur),
                "-c:v", "libx264",
                "-c:a", "aac",
                clip_path
            ], check=True)

            # face-aware reelify (replace)
            reelify_face_follow(clip_path, clip_path)

            upload_to_drive(
                clip_path,
                f"{base}_{i}",
                LATEST_INGEST["folder_id"]
            )

        if data.get("callback_url"):
            requests.post(data["callback_url"], json={
                "status": "success",
                "clips_uploaded": len(segments)
            }, timeout=10)

    bg.add_task(run)
    return {"status": "queued", "clips": len(segments)}

@app.get("/health")
async def health():
    return {"status": "ok"}
