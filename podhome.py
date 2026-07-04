import os
from pathlib import Path
from typing import Any, Dict, Optional
import requests

PODHOME_API_KEY = os.getenv("PODHOME_API_KEY") or os.getenv("podhome")
PODHOME_API_BASE = os.getenv("PODHOME_API_BASE", "https://serve.podhome.fm")


def podhome_headers() -> Dict[str, str]:
    if not PODHOME_API_KEY:
        raise RuntimeError("Podhome API key not configured. Set Railway variable podhome or PODHOME_API_KEY.")
    return {"X-API-KEY": PODHOME_API_KEY}


def podhome_request(method: str, path: str, **kwargs) -> Dict[str, Any]:
    url = f"{PODHOME_API_BASE.rstrip('/')}/{path.lstrip('/')}"
    headers = kwargs.pop("headers", {}) or {}
    headers.update(podhome_headers())
    response = requests.request(method, url, headers=headers, timeout=(10, 240), **kwargs)
    if not response.ok:
        raise RuntimeError(f"Podhome API error {response.status_code} {path}: {response.text}")
    if not response.text:
        return {}
    try:
        return response.json()
    except ValueError:
        return {"text": response.text}


def publish_episode_placeholder(
    show_config: Dict[str, Any],
    metadata: Dict[str, Any],
    audio_path: Optional[Path],
    mp3_url: str,
    transcript_text: str,
    scheduled_date: str,
) -> Dict[str, Any]:
    """
    Phase 2 will replace this placeholder with the full Podhome upload workflow:
    begin_upload -> PUT mp3 -> finalize_upload -> add_transcript -> schedule_episode.
    For Phase 1 we only mark the Queue row pending, while keeping all data ready.
    """
    return {
        "status": "pending",
        "episode_id": "",
        "message": "Phase 1 complete. Podhome publishing not executed yet.",
    }
