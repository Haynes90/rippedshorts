"""Customer-only RapidAPI-first acquisition; personal ingestion is unchanged."""
import json
import os
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


def media_ok(path):
    if not path.is_file() or not path.stat().st_size:
        return False
    result = subprocess.run([os.getenv("FFPROBE_BINARY", "ffprobe"), "-v", "error",
        "-show_entries", "stream=codec_type:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=30)
    try:
        data = json.loads(result.stdout)
        return result.returncode == 0 and float(data['format']['duration']) > 0 and {'audio', 'video'} <= {s['codec_type'] for s in data['streams']}
    except (KeyError, ValueError, TypeError):
        return False


def download_url(payload):
    if not isinstance(payload, dict):
        return None
    for key in ('url', 'download', 'download_url', 'downloadUrl', 'videoUrl', 'mainDownloadUrl'):
        value = payload.get(key)
        if isinstance(value, str) and urlparse(value).scheme == 'https':
            return value
    for item in payload.get('formats', []):
        if isinstance(item, dict) and item.get('hasVideo') and item.get('hasAudio'):
            value = download_url(item)
            if value:
                return value
    return None


def download_youtube(video_id, youtube_url, workdir):
    # This is the provider contract previously used in main.py (commit 8fd6dcc).
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    output = workdir / f'{video_id}-source.mp4'
    if media_ok(output):
        return output
    key = os.getenv('RAPIDAPI_KEY', '').strip()
    if not key:
        raise RuntimeError('Add RAPIDAPI_KEY to the recycle-app Railway service to enable the primary YouTube downloader.')
    host = os.getenv('YOUTUBE_DL_HOST', 'youtube-video-fast-downloader-24-7.p.rapidapi.com').strip()
    if not re.fullmatch(r'[a-zA-Z0-9-]+\.p\.rapidapi\.com', host):
        raise RuntimeError('YOUTUBE_DL_HOST must be a RapidAPI host')
    template = os.getenv('YOUTUBE_DL_PATH_TEMPLATE', '/download_video/{video_id}')
    endpoint = f'https://{host}{template.format(video_id=video_id)}'
    partial = workdir / f'{video_id}-source.partial.mp4'
    failure = 'no download URL after bounded polling'
    try:
        for attempt in range(3):
            response = requests.get(endpoint, headers={'x-rapidapi-key': key, 'x-rapidapi-host': host},
                params={'quality': '247'}, timeout=(10, 60))
            response.raise_for_status()
            url = download_url(response.json())
            if url:
                # Provider credentials are never sent to the returned media URL.
                with requests.get(url, stream=True, timeout=(10, 120)) as media:
                    media.raise_for_status()
                    with partial.open('wb') as handle:
                        for chunk in media.iter_content(1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                if not media_ok(partial):
                    raise ValueError('Provider returned media without usable audio and video')
                partial.replace(output)
                return output
            if attempt < 2:
                time.sleep(420 if attempt == 0 else 120)
    except (requests.RequestException, ValueError, OSError) as exc:
        # Do not persist signed download URLs, API responses, or credential headers.
        status = getattr(getattr(exc, 'response', None), 'status_code', None)
        failure = f'HTTP {status}' if status is not None else type(exc).__name__
    finally:
        partial.unlink(missing_ok=True)
    if os.getenv('PILOT_YTDLP_FALLBACK', '0') == '1':
        from source_ingestion import download_youtube_resilient
        return download_youtube_resilient(video_id, youtube_url, workdir)
    raise RuntimeError(f'RapidAPI could not prepare a usable video ({failure}). Check the downloader subscription/quota, or submit the source through Google Drive. Direct YouTube fallback is disabled while bot blocking persists.')
