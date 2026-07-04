import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from google_drive import read_google_doc_text
from google_sheets import append_queue_row, find_show_config
from openai_metadata import generate_episode_metadata
from podhome import publish_episode_placeholder
from scheduler import next_publish_datetime, queue_id

PODCAST_SHEET_ID = os.getenv("PODCAST_SHEET_ID", "14VruBxjaaE9DyPSdBidMeuPHew3nHUC5sRNGXtLCsis")
GENERATE_METADATA_AFTER_TRANSCRIBE = os.getenv("GENERATE_METADATA_AFTER_TRANSCRIBE", "false").lower() == "true"
POST_TO_PODHOME = os.getenv("POST_TO_PODHOME", "false").lower() == "true"


def transcript_text_from_segments(segments: List[dict]) -> str:
    lines = []
    for seg in segments:
        start = float(seg.get("start", 0.0))
        duration = float(seg.get("duration", 0.0))
        text = str(seg.get("text", "")).replace("\n", " ").strip()
        lines.append(f"[{start:.2f}s | dur={duration:.2f}s] {text}")
    return "\n".join(lines)


def get_prompt_reference(show_config: Dict[str, Any]) -> str:
    for key in ["prompt_doc", "prompt_doc_url", "prompt", "prompt_name", "ai_prompt", "metadata_prompt"]:
        value = str(show_config.get(key, "")).strip()
        if value:
            return value
    return "TDOG Prompt"


def get_show_id(show_config: Dict[str, Any]) -> str:
    return str(show_config.get("show_id") or show_config.get("podcast_id") or "TDOG").strip().upper()


def get_episode_number(show_config: Dict[str, Any]) -> str:
    for key in ["next_episode_number", "episode_number", "ep_number"]:
        value = str(show_config.get(key, "")).strip()
        if value:
            return value
    return ""


def run_podcast_metadata_queue(
    *,
    video_id: str,
    playlist_id: Optional[str],
    source_title: str,
    transcript_segments: List[dict],
    transcript_doc_url: str = "",
    transcript_json_url: str = "",
    mp3_url: str = "",
    audio_path: Optional[Path] = None,
    job_id: str = "",
) -> Dict[str, Any]:
    if not GENERATE_METADATA_AFTER_TRANSCRIBE:
        return {"enabled": False, "message": "GENERATE_METADATA_AFTER_TRANSCRIBE is false"}
    if not PODCAST_SHEET_ID:
        raise RuntimeError("PODCAST_SHEET_ID is not configured")

    show_config = find_show_config(PODCAST_SHEET_ID, playlist_id=playlist_id)
    show_id = get_show_id(show_config)
    prompt_reference = get_prompt_reference(show_config)
    prompt_text = read_google_doc_text(prompt_reference)
    transcript_text = transcript_text_from_segments(transcript_segments)

    metadata = generate_episode_metadata(
        prompt_text=prompt_text,
        transcript_text=transcript_text,
        show_config=show_config,
        source_title=source_title,
    )

    scheduled_date = next_publish_datetime(show_config)
    qid = queue_id(show_id, video_id, scheduled_date)

    podhome_result = {"status": "pending", "episode_id": ""}
    if POST_TO_PODHOME:
        podhome_result = publish_episode_placeholder(
            show_config=show_config,
            metadata=metadata,
            audio_path=audio_path,
            mp3_url=mp3_url,
            transcript_text=transcript_text,
            scheduled_date=scheduled_date,
        )

    podhome_status = podhome_result.get("status") or "pending"
    queue_row = {
        "Queue ID": qid,
        "Show ID": show_id,
        "Ep Number": get_episode_number(show_config),
        "MP3 URL": mp3_url,
        "Podhome Status": podhome_status,
        "Scheduled Date": scheduled_date,
        "Podhome Episode ID": podhome_result.get("episode_id", ""),
        "Episode Title": metadata.get("episode_title", ""),
        "Episode Description": metadata.get("episode_description", ""),
        "Transcript URL": transcript_doc_url,
        "Transcript JSON URL": transcript_json_url,
        "YouTube Video ID": video_id,
        "Job ID": job_id,
        "Error Message": podhome_result.get("error", ""),
        "Updated At": datetime.now(timezone.utc).isoformat(),
    }
    sheet_result = append_queue_row(PODCAST_SHEET_ID, queue_row, tab_name="Queue")

    metadata_path = Path("data") / "metadata" / f"{video_id}_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps({"metadata": metadata, "queue_row": queue_row, "show_config": show_config}, indent=2), encoding="utf-8")

    return {
        "enabled": True,
        "show_id": show_id,
        "queue_id": qid,
        "scheduled_date": scheduled_date,
        "metadata": metadata,
        "queue_row": queue_row,
        "sheet": sheet_result,
        "podhome": podhome_result,
        "metadata_path": str(metadata_path),
    }
