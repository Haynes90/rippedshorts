"""Audiovisual active-speaker planning for vertical podcast/interview clips.

This module never owns the fallback. If analysis is uncertain it returns None and
the established face-aware/stacked renderer remains authoritative.
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from statistics import median
from typing import Optional

logger = logging.getLogger("ripped-shorts.active-speaker")


def _audio_rms(video_path: Path, start: float, duration: float, hz: int = 20):
    import numpy as np

    rate = 16000
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}", "-i", str(video_path), "-vn", "-ac", "1",
        "-ar", str(rate), "-f", "f32le", "-",
    ]
    completed = subprocess.run(command, capture_output=True, timeout=max(30, int(duration) + 20))
    if completed.returncode != 0 or not completed.stdout:
        return []
    samples = np.frombuffer(completed.stdout, dtype=np.float32)
    window = max(1, rate // hz)
    return [
        float(np.sqrt(np.mean(chunk * chunk)))
        for offset in range(0, len(samples), window)
        if (chunk := samples[offset:offset + window]).size
    ]


def _energy_at(energy: list[float], timestamp: float, hz: int = 20) -> float:
    if not energy:
        return 0.0
    index = max(0, min(len(energy) - 1, int(timestamp * hz)))
    return energy[index]


def _detect_tracks(video_path: Path, start: float, duration: float, load_cascade):
    import cv2
    import numpy as np

    cascade = load_cascade()
    if cascade is None:
        return []
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    interval = max(0.25, float(os.getenv("ACTIVE_SPEAKER_SAMPLE_SECONDS", "0.5")))
    step = max(1, int(round(fps * interval)))
    total = max(1, int(round(duration * fps)))
    previous_mouth = {"A": None, "B": None}
    samples = []
    frame_number = 0
    while frame_number <= total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(start * fps)) + frame_number)
        ok, frame = cap.read()
        if not ok:
            break
        scale = min(1.0, 720.0 / max(frame.shape[:2]))
        analysis = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
        gray = cv2.cvtColor(analysis, cv2.COLOR_BGR2GRAY)
        faces = list(cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        ))
        faces.sort(key=lambda box: box[2] * box[3], reverse=True)
        pair = None
        for i, first in enumerate(faces[:4]):
            for second in faces[i + 1:4]:
                c1 = (first[0] + first[2] / 2) / gray.shape[1]
                c2 = (second[0] + second[2] / 2) / gray.shape[1]
                if abs(c1 - c2) >= 0.22:
                    pair = sorted(((c1, first), (c2, second)), key=lambda item: item[0])
                    break
            if pair:
                break
        if pair:
            values = {}
            for label, (center, (x, y, width, height)) in zip(("A", "B"), pair):
                # Lower half of the detected face approximates the mouth/jaw region.
                top = max(0, y + height // 2)
                bottom = min(gray.shape[0], y + height)
                left, right = max(0, x), min(gray.shape[1], x + width)
                mouth = gray[top:bottom, left:right]
                prior = previous_mouth[label]
                motion = 0.0
                if mouth.size and prior is not None:
                    resized = cv2.resize(mouth, (64, 32))
                    motion = float(np.mean(cv2.absdiff(prior, resized))) / 255.0
                    previous_mouth[label] = resized
                elif mouth.size:
                    previous_mouth[label] = cv2.resize(mouth, (64, 32))
                values[label] = (float(center), motion)
            samples.append((frame_number / fps, values["A"], values["B"]))
        frame_number += step
    cap.release()
    return samples


def _stable_layout_sections(samples, energy, duration: float):
    if len(samples) < 4:
        return []
    motions = [max(a[1], b[1]) for _, a, b in samples]
    motion_floor = max(0.008, median(motions) * 0.65)
    audio_values = [value for value in energy if value > 0]
    audio_floor = median(audio_values) * 0.35 if audio_values else 0.0
    confidence_ratio = max(1.15, float(os.getenv("ACTIVE_SPEAKER_CONFIDENCE_RATIO", "1.35")))
    desired = []
    for timestamp, left, right in samples:
        speaking_audio = _energy_at(energy, timestamp) >= audio_floor if energy else True
        left_score = left[1] if speaking_audio else 0.0
        right_score = right[1] if speaking_audio else 0.0
        if max(left_score, right_score) < motion_floor:
            layout = "STACKED"
        elif left_score >= right_score * confidence_ratio:
            layout = "A"
        elif right_score >= left_score * confidence_ratio:
            layout = "B"
        else:
            layout = "STACKED"
        desired.append((timestamp, layout, left[0], right[0]))

    confirmations = max(2, int(os.getenv("ACTIVE_SPEAKER_CONFIRMATIONS", "3")))
    minimum_hold = max(1.5, float(os.getenv("ACTIVE_SPEAKER_MIN_HOLD_SECONDS", "2.5")))
    current = "STACKED"
    pending = None
    pending_count = 0
    last_change = 0.0
    sections = [(0.0, current, desired[0][2], desired[0][3])]
    for timestamp, layout, left, right in desired:
        if layout == current:
            pending, pending_count = None, 0
            continue
        if layout == pending:
            pending_count += 1
        else:
            pending, pending_count = layout, 1
        if pending_count >= confirmations and timestamp - last_change >= minimum_hold:
            current = layout
            sections.append((timestamp, current, left, right))
            last_change = timestamp
            pending, pending_count = None, 0
    # A useful active-speaker plan must contain at least one confident full frame.
    if not any(item[1] in {"A", "B"} for item in sections):
        return []
    return sections


def _crop_x(center: float, width: int, crop_width: int) -> int:
    return max(0, min(width - crop_width, int(center * width) - crop_width // 2))


def _section_filter(label: str, layout: str, left: float, right: float, width: int, height: int):
    if layout in {"A", "B"}:
        crop_width = min(width, int(height * 9 / 16))
        crop_width = max(2, crop_width - crop_width % 2)
        x = _crop_x(left if layout == "A" else right, width, crop_width)
        return f"[{label}]crop={crop_width}:{height}:{x}:0,scale=1080:1920[v{label[1:]}]"
    crop_width = min(width, int(height * 9 / 8))
    crop_width = max(2, crop_width - crop_width % 2)
    lx, rx = _crop_x(left, width, crop_width), _crop_x(right, width, crop_width)
    index = label[1:]
    return (
        f"[{label}]split=2[{label}a][{label}b];"
        f"[{label}a]crop={crop_width}:{height}:{lx}:0,scale=1080:960:"
        "force_original_aspect_ratio=increase,crop=1080:960"
        f"[top{index}];"
        f"[{label}b]crop={crop_width}:{height}:{rx}:0,scale=1080:960:"
        "force_original_aspect_ratio=increase,crop=1080:960"
        f"[bottom{index}];[top{index}][bottom{index}]vstack=2[v{index}]"
    )


def build_active_speaker_filter(
    video_path: Path,
    start: float,
    duration: float,
    width: int,
    height: int,
    load_cascade,
) -> Optional[str]:
    """Return an FFmpeg graph switching A/B/stacked, or None when uncertain."""
    try:
        samples = _detect_tracks(video_path, start, duration, load_cascade)
        energy = _audio_rms(video_path, start, duration)
        sections = _stable_layout_sections(samples, energy, duration)
        if not sections:
            return None
        boundaries = [item[0] for item in sections] + [duration]
        graphs = []
        outputs = []
        for index, (_, layout, left, right) in enumerate(sections):
            section_start, section_end = boundaries[index], boundaries[index + 1]
            label = f"s{index}"
            graphs.append(
                f"[0:v]trim=start={section_start:.3f}:end={section_end:.3f},"
                f"setpts=PTS-STARTPTS[{label}]"
            )
            graphs.append(_section_filter(label, layout, left, right, width, height))
            outputs.append(f"[v{index}]")
        graphs.append("".join(outputs) + f"concat=n={len(outputs)}:v=1:a=0[v]")
        logger.info(
            "Vertical active-speaker layouts=%s",
            ",".join(item[1] for item in sections),
        )
        return ";".join(graphs)
    except Exception:
        logger.exception("Active-speaker analysis failed; using deterministic fallback")
        return None
