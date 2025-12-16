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

# Step-2 deps (must be in requirements.txt)
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
    except Exception:
        return None

def run_cmd(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise Exception(r.stderr.decode("utf-8", errors="ignore")[:8000])
    return r

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
            "extractor_args": {"youtube": {"player_client": ["android"]}},
        }
        if opts["cookies"] is None:
            opts.pop("cookies", None)

        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

        if not os.path.exists(out) or os.path.getsize(out) == 0:
            raise Exception("yt-dlp failed (no output)")

        debug(f"Downloaded bytes: {os.path.getsize(out)}")
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
            try:
                requests.post(data["callback_url"], json=JOBS[job_id], timeout=10)
            except Exception as e:
                debug(f"Callback failed: {e}")

        return {"status": "success", "job_id": job_id, "file_id": up["id"]}

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        debug(f"INGEST ERROR: {e}")
        return JSONResponse({"error": str(e)}, 500)

# --------------------------------------------------
# SEGMENT NORMALIZER (DEFENSIVE AGAINST TRUNCATED JSON)
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
                debug("normalize_segments: outer JSON invalid")
                return []

        if not isinstance(data, dict):
            return []

        raw = data.get("segments_json") or data.get("segments")

        # If raw is a string, it may be BROKEN JSON; salvage offsets/durations line-by-line
        if isinstance(raw, str):
            segments = []
            offset = None
            duration = None
            for line in raw.splitlines():
                line = line.strip()
                if '"offset"' in line or '"start"' in line:
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
                        segments.append({"start": offset, "duration": duration})
                    offset = None
                    duration = None

            debug(f"Recovered {len(segments)} segments from broken JSON")
            return segments

        # If raw is dict, pull "segments"
        if isinstance(raw, dict):
            raw = raw.get("segments")

        segments = []
        if isinstance(raw, list):
            for s in raw:
                if not isinstance(s, dict):
                    continue
                start = safe_float(s.get("offset") or s.get("start"))
                dur = safe_float(s.get("duration"))
                if start is not None and dur is not None and dur > 0:
                    segments.append({"start": start, "duration": dur})

        return segments

    except Exception as e:
        debug(f"normalize_segments FAILED SAFELY: {e}")
        return []

# --------------------------------------------------
# FFMPEG CLIP (AUDIO SAFE)
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
        "-b:a", "128k",
        "-preset", "veryfast",
        "-movflags", "+faststart",
        out
    ]
    run_cmd(cmd)
    if not os.path.exists(out) or os.path.getsize(out) == 0:
        raise Exception("clip output empty")

# --------------------------------------------------
# AUDIO MUX (re-attach audio after OpenCV reframe)
# --------------------------------------------------
def mux_audio(video_no_audio_path, audio_source_path, out_path):
    """
    Take video stream from video_no_audio_path and audio from audio_source_path.
    Encode to IG-friendly mp4.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video_no_audio_path,
        "-i", audio_source_path,
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        "-movflags", "+faststart",
        out_path
    ]
    run_cmd(cmd)
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise Exception("mux output empty")

# --------------------------------------------------
# 9:16 SPEAKER-FOLLOW (VISION-BASED)
# - face detect + mouth motion proxy
# - dead-zone + smoothing
# - lock-on + agile switch
# - split-screen dampening
# --------------------------------------------------
class ReelFramer:
    def __init__(self):
        self.mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6
        )
        self.mp_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # output (IG reel)
        self.out_w = 1080
        self.out_h = 1920
        self.aspect = self.out_w / self.out_h

        # motion tuning
        self.base_zoom = 1.15
        self.max_zoom = 1.25
        self.split_cap_zoom = 1.15

        self.dead_zone_ratio = 0.20
        self.smooth_alpha = 0.88
        self.hold_seconds = 1.5
        self.switch_guard_seconds = 0.7

        # state
        self.last_target_x = None
        self.last_zoom = None
        self.last_good_time = None
        self.last_speaker_face = None
        self._last_switch_t = None
        self._current_face_id = None

    def _face_boxes(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mp_face.process(rgb)
        faces = []
        if res.detections:
            for det in res.detections:
                bb = det.location_data.relative_bounding_box
                x = max(0, int(bb.xmin * w))
                y = max(0, int(bb.ymin * h))
                bw = int(bb.width * w)
                bh = int(bb.height * h)
                bw = max(1, min(bw, w - x))
                bh = max(1, min(bh, h - y))
                faces.append((x, y, bw, bh))
        return faces

    def _iou(self, a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        a2x = ax + aw; a2y = ay + ah
        b2x = bx + bw; b2y = by + bh
        ix1 = max(ax, bx); iy1 = max(ay, by)
        ix2 = min(a2x, b2x); iy2 = min(a2y, b2y)
        iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
        inter = iw * ih
        union = (aw * ah) + (bw * bh) - inter
        return inter / union if union > 0 else 0.0

    def _mouth_motion_scores(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mp_mesh.process(rgb)
        out = []
        if not res.multi_face_landmarks:
            return out

        # upper lip 13, lower lip 14
        for lm in res.multi_face_landmarks:
            pts = lm.landmark
            xs = [p.x for p in pts]
            ys = [p.y for p in pts]
            x1 = int(min(xs) * w); x2 = int(max(xs) * w)
            y1 = int(min(ys) * h); y2 = int(max(ys) * h)
            x1 = max(0, min(w - 1, x1)); x2 = max(1, min(w, x2))
            y1 = max(0, min(h - 1, y1)); y2 = max(1, min(h, y2))

            up = pts[13]; lo = pts[14]
            mouth_open = abs(lo.y - up.y) * h

            rect = (x1, y1, x2 - x1, y2 - y1)
            out.append((mouth_open, rect))

        return out

    def _pick_speaker_face(self, frame_bgr, tsec):
        faces = self._face_boxes(frame_bgr)
        if not faces:
            return None, faces

        # split-screen heuristic: 2+ faces with far-apart X centers
        split_screen = False
        if len(faces) >= 2:
            centers = sorted([(x + bw / 2.0) for (x, y, bw, bh) in faces])
            if centers[-1] - centers[0] > (frame_bgr.shape[1] * 0.45):
                split_screen = True

        motion = self._mouth_motion_scores(frame_bgr)
        best = None

        if motion:
            scored = []
            for score, est_rect in motion:
                best_match = None
                best_iou = 0.0
                for f in faces:
                    i = self._iou(est_rect, f)
                    if i > best_iou:
                        best_iou = i
                        best_match = f
                if best_match is not None:
                    scored.append((score, best_match))

            if scored:
                scored.sort(key=lambda x: x[0], reverse=True)
                best = scored[0][1]

        # fallback: biggest face
        if best is None:
            best = max(faces, key=lambda f: f[2] * f[3])

        # switch guard (avoid ping-pong)
        bx, by, bw, bh = best
        face_id = int((bx + bw / 2) // 50)

        if self._current_face_id is None:
            self._current_face_id = face_id
            self._last_switch_t = tsec
        else:
            if face_id != self._current_face_id:
                if self._last_switch_t is None:
                    self._last_switch_t = tsec
                if (tsec - self._last_switch_t) >= self.switch_guard_seconds:
                    self._current_face_id = face_id
                    self._last_switch_t = tsec
                else:
                    # keep previous target if possible
                    if self.last_target_x is not None:
                        prev = min(
                            faces,
                            key=lambda f: abs((f[0] + f[2] / 2) - self.last_target_x)
                        )
                        best = prev
            else:
                self._last_switch_t = tsec

        return (best, split_screen), faces

    def _compute_camera(self, frame_bgr, speaker_rect, split_screen):
        h, w = frame_bgr.shape[:2]
        x, y, bw, bh = speaker_rect
        cx = x + bw / 2.0

        face_ratio = (bh / h)
        target_zoom = self.base_zoom + (0.10 * (0.22 - face_ratio))
        target_zoom = max(self.base_zoom, min(self.max_zoom, target_zoom))
        if split_screen:
            target_zoom = min(target_zoom, self.split_cap_zoom)

        # viewport size in source
        crop_h = int(h / target_zoom)
        crop_w = int(crop_h * self.aspect)
        if crop_w > w:
            crop_w = w
            crop_h = int(crop_w / self.aspect)

        # dead-zone + easing
        if self.last_target_x is None:
            cam_cx = cx
        else:
            cam_cx = self.last_target_x
            dz = w * self.dead_zone_ratio
            if abs(cx - cam_cx) > dz:
                cam_cx = cx

        if self.last_target_x is None:
            smoothed_cx = cam_cx
        else:
            smoothed_cx = (self.smooth_alpha * self.last_target_x) + ((1 - self.smooth_alpha) * cam_cx)

        if self.last_zoom is None:
            smoothed_zoom = target_zoom
        else:
            smoothed_zoom = (self.smooth_alpha * self.last_zoom) + ((1 - self.smooth_alpha) * target_zoom)

        self.last_target_x = smoothed_cx
        self.last_zoom = smoothed_zoom

        crop_h = int(h / smoothed_zoom)
        crop_w = int(crop_h * self.aspect)
        if crop_w > w:
            crop_w = w
            crop_h = int(crop_w / self.aspect)

        x1 = int(max(0, min(w - crop_w, smoothed_cx - crop_w / 2)))
        y1 = int(max(0, min(h - crop_h, (h - crop_h) / 2)))

        return x1, y1, crop_w, crop_h

    def reframe_to_reel_video_only(self, in_path, out_path, log_prefix=""):
        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            raise Exception("Could not open clip for reframing")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (self.out_w, self.out_h))
        if not out.isOpened():
            raise Exception("Could not open output writer for reel")

        # reset state per clip
        self.last_target_x = None
        self.last_zoom = None
        self._current_face_id = None
        self._last_switch_t = None
        self.last_good_time = None
        self.last_speaker_face = None

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            tsec = frame_idx / fps
            pick, _faces = self._pick_speaker_face(frame, tsec)
            speaker_rect = None
            split_screen = False

            if pick is not None:
                (speaker_rect, split_screen) = pick
                self.last_good_time = tsec
                self.last_speaker_face = speaker_rect
            else:
                # memory hold
                if self.last_good_time is not None and (tsec - self.last_good_time) <= self.hold_seconds:
                    speaker_rect = self.last_speaker_face

            if speaker_rect is not None:
                x1, y1, cw, ch = self._compute_camera(frame, speaker_rect, split_screen)
                crop = frame[y1:y1 + ch, x1:x1 + cw]
            else:
                # neutral center crop
                h, w = frame.shape[:2]
                zoom = self.base_zoom
                crop_h = int(h / zoom)
                crop_w = int(crop_h * self.aspect)
                if crop_w > w:
                    crop_w = w
                    crop_h = int(crop_w / self.aspect)
                x1 = (w - crop_w) // 2
                y1 = (h - crop_h) // 2
                crop = frame[y1:y1 + crop_h, x1:x1 + crop_w]

            resized = cv2.resize(crop, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)
            out.write(resized)

            frame_idx += 1
            if total and frame_idx % int(max(1, fps)) == 0:
                debug(f"{log_prefix}Reframe progress: {frame_idx}/{total} frames")

        cap.release()
        out.release()

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise Exception("Reframed video output empty")

# --------------------------------------------------
# ANALYZE + CLIP + REFRAME 9:16 + AUDIO PRESERVE
# --------------------------------------------------
@app.post("/analyze-and-clip")
async def analyze_and_clip(req: Request):
    data = await req.json()
    raw_log("ANALYZE", data)

    if not LATEST_INGEST["file_id"]:
        return JSONResponse({"error": "no ingest available"}, 400)

    # callback + title often come at top-level
    callback = None
    if isinstance(data, dict):
        callback = data.get("callback_url")

    base = sanitize((data.get("video_title") if isinstance(data, dict) else None) or LATEST_INGEST["title"])

    segments = normalize_segments(data)
    debug(f"Segments parsed: {len(segments)}")

    if not segments:
        # never fail Zapier when segments missing/broken
        if callback:
            try:
                requests.post(callback, json={"status": "ok", "clips_uploaded": 0, "error": None}, timeout=10)
            except Exception:
                pass
        return {"status": "ok", "clips_uploaded": 0}

    try:
        src = download_from_drive(LATEST_INGEST["file_id"])
        debug(f"Source downloaded bytes: {os.path.getsize(src)}")

        framer = ReelFramer()
        uploaded = 0

        for i, s in enumerate(segments, 1):
            start = s["start"]
            dur = s["duration"]

            raw_clip = f"/tmp/{base}_{i}.mp4"
            make_clip(src, raw_clip, start, dur)
            debug(f"Clip made: {raw_clip} bytes={os.path.getsize(raw_clip)}")

            # 1) reframe VIDEO ONLY
            reel_video_only = f"/tmp/{base}_{i}__reel_video.mp4"
            framer.reframe_to_reel_video_only(raw_clip, reel_video_only, log_prefix=f"[{base}_{i}] ")

            # 2) mux original audio back in
            final_clip = f"/tmp/{base}_{i}__final.mp4"
            mux_audio(reel_video_only, raw_clip, final_clip)

            # replace naming to keep it clean (no duplicates)
            os.replace(final_clip, raw_clip)

            # cleanup intermediates
            try:
                if os.path.exists(reel_video_only):
                    os.remove(reel_video_only)
            except Exception:
                pass

            # upload final 9:16+audio clip
            upload_to_drive(raw_clip, f"{base}_{i}", LATEST_INGEST["folder_id"])
            uploaded += 1
            debug(f"Uploaded clip {i}/{len(segments)}: {base}_{i}.mp4")

        if callback:
            try:
                requests.post(callback, json={"status": "success", "clips_uploaded": uploaded}, timeout=10)
            except Exception as e:
                debug(f"Callback failed: {e}")

        return {"status": "success", "clips_uploaded": uploaded}

    except Exception as e:
        debug(f"ANALYZE+CLIP ERROR: {e}")
        if callback:
            try:
                requests.post(callback, json={"status": "failed", "error": str(e), "clips_uploaded": 0}, timeout=10)
            except Exception:
                pass
        return JSONResponse({"error": str(e)}, 500)

# --------------------------------------------------
# HEALTH
# --------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}
