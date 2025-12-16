import os
import json
import uuid
import datetime
import subprocess
import io
import math
import yt_dlp
import requests

from fastapi import FastAPI, Request, BackgroundTasks
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
# INGEST (UNCHANGED, VERIFIED)
# --------------------------------------------------
def download_youtube(url, title):
    out = f"/tmp/{sanitize(title)}.mp4"
    opts = {
        "format": "bv*+ba/best",
        "merge_output_format": "mp4",
        "outtmpl": out,
        "cookies": COOKIES_PATH if os.path.exists(COOKIES_PATH) else None,
        "quiet": True,
        "retries": 5,
        "fragment_retries": 5,
        "extractor_args": {"youtube": {"player_client": ["android"]}}
    }
    # remove cookies key if None (yt-dlp can be picky)
    if opts["cookies"] is None:
        opts.pop("cookies", None)

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    if not os.path.exists(out) or os.path.getsize(out) == 0:
        raise Exception("YouTube download failed")
    return out

def process_ingest(job_id, url, title, callback):
    JOBS.setdefault(job_id, {"status": "queued"})
    try:
        job_log(job_id, f"Downloading YouTube: {url}")
        path = download_youtube(url, title)
        job_log(job_id, f"Downloaded size: {os.path.getsize(path)} bytes")

        job_log(job_id, "Uploading full video to Drive...")
        up = upload_to_drive(path, title, os.environ["DRIVE_FOLDER_ID"])
        job_log(job_id, f"Drive file_id: {up['id']}")

        LATEST_INGEST.update({
            "file_id": up["id"],
            "title": title,
            "folder_id": os.environ["DRIVE_FOLDER_ID"]
        })

        JOBS[job_id]["status"] = "success"
        JOBS[job_id]["file_id"] = up["id"]
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        job_log(job_id, f"ERROR: {e}")

    if callback:
        try:
            requests.post(callback, json=JOBS[job_id], timeout=10)
        except Exception as e:
            job_log(job_id, f"Callback failed: {e}")


# --------------------------------------------------
# SEGMENT NORMALIZER (ROBUST AGAINST ZAPIER)
# --------------------------------------------------
def _extract_json_blob_from_weird_payload(data):
    """
    Zapier sometimes posts:
      {"": "{...json...}"}
    or wraps everything in a string.
    """
    if isinstance(data, dict) and "" in data and isinstance(data[""], str):
        s = data[""]
        try:
            return json.loads(s)
        except Exception:
            return None
    return data

def normalize_segments(data):
    """
    Accept segments in many shapes:
    - {"segments_json": [{"start":..,"duration":..},...]}
    - {"segments_json": "{\"segments\":[...]}"}
    - {"segments_json": {"segments":[...]}}
    - {"segments":[...]}
    - raw stringified JSON with {"segments":[...]}
    Reject transcript-heavy stuff (offset/text/lang/etc).
    """
    data = _extract_json_blob_from_weird_payload(data) or data

    raw = None
    if isinstance(data, dict):
        raw = data.get("segments_json")
        if raw is None:
            raw = data.get("segments")

    # If raw is a string, try JSON parse
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            # sometimes zapier sends error json string
            try:
                err_obj = json.loads(raw)
                if isinstance(err_obj, dict) and "error" in err_obj:
                    return []
            except Exception:
                return []

    # If raw is dict, pull "segments"
    if isinstance(raw, dict):
        raw = raw.get("segments")

    # If raw is list, good. If raw is still None, try entire payload
    if raw is None:
        if isinstance(data, dict) and isinstance(data.get("segments"), list):
            raw = data["segments"]
        elif isinstance(data, list):
            raw = data
        else:
            raw = []

    if not isinstance(raw, list):
        return []

    segments = []
    for s in raw:
        if not isinstance(s, dict):
            continue
        # ignore transcript-like objects (offset/text etc)
        if "text" in s or "offset" in s or "lang" in s:
            continue
        st = safe_float(s.get("start"))
        du = safe_float(s.get("duration"))
        if st is None or du is None:
            continue
        if du <= 0:
            continue
        segments.append({"start": st, "duration": du})

    return segments


# --------------------------------------------------
# FFMPEG CLIP
# --------------------------------------------------
def make_clip(src, out, start, dur):
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", src,
        "-t", str(dur),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "veryfast",
        out
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise Exception(f"ffmpeg clip failed: {r.stderr.decode('utf-8', errors='ignore')[:8000]}")
    if not os.path.exists(out) or os.path.getsize(out) == 0:
        raise Exception("clip output empty")


# --------------------------------------------------
# 9:16 SPEAKER-FOLLOW (VISION-BASED)
#   - face detect + mouth-motion (approx active speaker)
#   - dead-zone + easing + memory
#   - face-close vibe (your choice B)
# --------------------------------------------------
class ReelFramer:
    def __init__(self):
        self.mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6
        )
        # Use FaceMesh to estimate mouth openness (lightweight)
        self.mp_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # camera state
        self.last_target_x = None
        self.last_zoom = None
        self.last_good_time = None  # seconds
        self.last_speaker_face = None

        # tuning (big-dog defaults)
        self.out_w = 1080
        self.out_h = 1920
        self.aspect = self.out_w / self.out_h

        self.base_zoom = 1.15   # face-close start
        self.max_zoom = 1.25
        self.split_cap_zoom = 1.15

        self.dead_zone_ratio = 0.20   # 20% of width
        self.smooth_alpha = 0.88      # easing
        self.hold_seconds = 1.5       # your YES
        self.switch_guard_seconds = 0.7  # must be stable before switching

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
                # clamp
                bw = max(1, min(bw, w - x))
                bh = max(1, min(bh, h - y))
                faces.append((x, y, bw, bh))
        return faces

    def _mouth_motion_scores(self, frame_bgr):
        """
        Returns list of (score, face_rect_estimate)
        Score approximates "talking" using mouth landmark motion proxy.
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mp_mesh.process(rgb)
        out = []

        if not res.multi_face_landmarks:
            return out

        # mouth landmarks indexes (rough)
        # upper lip (13) lower lip (14) in mediapipe canonical mesh
        for lm in res.multi_face_landmarks:
            pts = lm.landmark
            # approximate face bounds
            xs = [p.x for p in pts]
            ys = [p.y for p in pts]
            x1 = int(min(xs) * w); x2 = int(max(xs) * w)
            y1 = int(min(ys) * h); y2 = int(max(ys) * h)
            x1 = max(0, min(w - 1, x1)); x2 = max(1, min(w, x2))
            y1 = max(0, min(h - 1, y1)); y2 = max(1, min(h, y2))

            # mouth openness proxy
            up = pts[13]; lo = pts[14]
            mouth_open = abs(lo.y - up.y) * h

            rect = (x1, y1, x2 - x1, y2 - y1)
            out.append((mouth_open, rect))

        return out

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

    def _pick_speaker_face(self, frame_bgr, tsec):
        faces = self._face_boxes(frame_bgr)

        # If no faces, keep memory
        if not faces:
            return None, faces

        # Split-screen detection (simple): 2+ faces, far apart in X
        split_screen = False
        if len(faces) >= 2:
            centers = sorted([(x + bw / 2.0) for (x, y, bw, bh) in faces])
            if centers[-1] - centers[0] > (frame_bgr.shape[1] * 0.45):
                split_screen = True

        # mouth-motion picks the "talker"
        motion = self._mouth_motion_scores(frame_bgr)
        best = None

        if motion:
            # map motion face-rect to nearest detection rect using IoU
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

        # speaker switching guard (prevents ping-pong)
        # define "face id" by position bucket
        bx, by, bw, bh = best
        face_id = int((bx + bw / 2) // 50)  # coarse bucket

        if self._current_face_id is None:
            self._current_face_id = face_id
            self._last_switch_t = tsec
        else:
            if face_id != self._current_face_id:
                # switch only if stable long enough
                if self._last_switch_t is None:
                    self._last_switch_t = tsec
                if (tsec - self._last_switch_t) >= self.switch_guard_seconds:
                    self._current_face_id = face_id
                    self._last_switch_t = tsec
                else:
                    # not stable yet: keep previous if we can find it
                    # choose face closest to previous target x
                    if self.last_target_x is not None:
                        prev = min(
                            faces,
                            key=lambda f: abs((f[0] + f[2] / 2) - self.last_target_x)
                        )
                        best = prev
            else:
                self._last_switch_t = tsec

        # apply split-screen zoom cap (still locks to speaker, but less aggressive zoom)
        return (best, split_screen), faces

    def _compute_camera(self, frame_bgr, speaker_rect, split_screen):
        h, w = frame_bgr.shape[:2]
        x, y, bw, bh = speaker_rect
        cx = x + bw / 2.0

        # zoom based on face size (bigger face -> less zoom needed)
        face_ratio = (bh / h)
        # target zoom heuristics for face-close vibe
        # clamp to [base, max]
        target_zoom = self.base_zoom + (0.10 * (0.22 - face_ratio))
        target_zoom = max(self.base_zoom, min(self.max_zoom, target_zoom))

        if split_screen:
            target_zoom = min(target_zoom, self.split_cap_zoom)

        # viewport size inside source
        crop_h = int(h / target_zoom)
        crop_w = int(crop_h * self.aspect)
        if crop_w > w:
            crop_w = w
            crop_h = int(crop_w / self.aspect)

        # dead-zone: if speaker stays near current camera center, don't move
        if self.last_target_x is None:
            cam_cx = cx
        else:
            cam_cx = self.last_target_x
            dz = w * self.dead_zone_ratio
            if abs(cx - cam_cx) > dz:
                cam_cx = cx

        # easing
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
        y1 = int(max(0, min(h - crop_h, (h - crop_h) / 2)))  # keep vertical stable; framing is handled by resizing

        return x1, y1, crop_w, crop_h

    def reframe_to_reel(self, in_path, out_path, job_id=None):
        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            raise Exception("Could not open clip for reframing")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (self.out_w, self.out_h))
        if not out.isOpened():
            raise Exception("Could not open output writer for reel")

        frame_idx = 0
        self.last_target_x = None
        self.last_zoom = None
        self._current_face_id = None
        self._last_switch_t = None
        self.last_good_time = None
        self.last_speaker_face = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            tsec = frame_idx / fps
            pick, faces = self._pick_speaker_face(frame, tsec)
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
                else:
                    speaker_rect = None

            if speaker_rect is not None:
                x1, y1, cw, ch = self._compute_camera(frame, speaker_rect, split_screen)
                crop = frame[y1:y1 + ch, x1:x1 + cw]
            else:
                # neutral center crop at base zoom
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
            if job_id and total and frame_idx % int(max(1, fps)) == 0:
                # once per ~second
                job_log(job_id, f"Reframe progress: {frame_idx}/{total} frames")

        cap.release()
        out.release()

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise Exception("Reframed reel output empty")


# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.post("/ingest")
async def ingest(req: Request, bg: BackgroundTasks):
    data = await req.json()
    raw_log("INGEST", data)

    if "url" not in data:
        return JSONResponse({"error": "missing url"}, 400)

    title = data["url"].split("v=")[-1].split("&")[0]
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued", "logs": []}

    bg.add_task(process_ingest, job_id, data["url"], title, data.get("callback_url"))
    return {"status": "queued", "job_id": job_id}


@app.post("/analyze-and-clip")
async def analyze_and_clip(req: Request, bg: BackgroundTasks):
    data = await req.json()
    raw_log("ANALYZE", data)

    if not LATEST_INGEST["file_id"]:
        return JSONResponse({"error": "no ingest available"}, 400)

    segments = normalize_segments(data)
    debug(f"Segments parsed: {len(segments)}")
    debug(f"First segment: {segments[0] if segments else None}")

    # HARD RULE: If no valid segments, return OK with 0 (don’t fail Zap)
    if not segments:
        return {"status": "ok", "clips_found": 0, "clips_created": 0, "clips_uploaded": 0}

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued", "logs": [], "clips": []}

    callback = None
    payload_root = _extract_json_blob_from_weird_payload(data) or data
    if isinstance(payload_root, dict):
        callback = payload_root.get("callback_url")

    video_title = None
    if isinstance(payload_root, dict):
        video_title = payload_root.get("video_title")

    def run():
        JOBS.setdefault(job_id, {"status": "queued", "logs": [], "clips": []})
        try:
            job_log(job_id, "Downloading source from Drive...")
            src = download_from_drive(LATEST_INGEST["file_id"])
            job_log(job_id, f"Source downloaded: {os.path.getsize(src)} bytes")

            base = sanitize(video_title or LATEST_INGEST["title"])
            framer = ReelFramer()

            created = 0
            uploaded = 0

            for i, s in enumerate(segments, 1):
                start = s["start"]
                dur = s["duration"]

                raw_clip = f"/tmp/{base}_{i}.mp4"
                make_clip(src, raw_clip, start, dur)
                job_log(job_id, f"Clip made: {raw_clip} ({os.path.getsize(raw_clip)} bytes)")

                # Reframe into 9:16 speaker-follow and REPLACE original
                reel_tmp = f"/tmp/{base}_{i}__reel.mp4"
                framer.reframe_to_reel(raw_clip, reel_tmp, job_id=job_id)

                # Replace
                os.replace(reel_tmp, raw_clip)
                job_log(job_id, f"Clip reframed (replaced): {raw_clip} ({os.path.getsize(raw_clip)} bytes)")

                # Upload final (same naming convention)
                up = upload_to_drive(raw_clip, f"{base}_{i}", LATEST_INGEST["folder_id"])
                JOBS[job_id]["clips"].append(up["id"])
                uploaded += 1
                created += 1

                job_log(job_id, f"Uploaded to Drive: {base}_{i}.mp4 (id={up['id']})")

            JOBS[job_id]["status"] = "success"

            if callback:
                requests.post(callback, json={
                    "status": "success",
                    "clips_created": created,
                    "clips_uploaded": uploaded
                }, timeout=10)

        except Exception as e:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(e)
            job_log(job_id, f"ERROR: {e}")

            if callback:
                try:
                    requests.post(callback, json={
                        "status": "failed",
                        "error": str(e),
                        "clips_created": 0,
                        "clips_uploaded": 0
                    }, timeout=10)
                except Exception:
                    pass

    bg.add_task(run)
    return {"status": "queued", "job_id": job_id, "clips_found": len(segments)}


@app.get("/health")
async def health():
    return {"status": "ok"}
