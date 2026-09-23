from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "source_ingestion.py").read_text(encoding="utf-8")


def test_existing_rapidapi_video_downloader_is_in_source_race():
    assert "youtube-video-fast-downloader-24-7.p.rapidapi.com" in SOURCE
    assert '"/download_video/{video_id}"' in SOURCE
    assert '"name": "rapidapi_video_download"' in SOURCE
    assert '"provider": "rapidapi"' in SOURCE
    assert "params={\"quality\": quality}" in SOURCE


def test_rapidapi_file_url_is_polled_until_ready():
    assert 'payload.get("file")' in SOURCE
    assert "file_response.status_code == 404" in SOURCE
    assert "YOUTUBE_DL_READY_TIMEOUT_SECONDS" in SOURCE


def test_race_dispatches_rapidapi_and_ytdlp():
    assert "def _run_source_profile(" in SOURCE
    assert 'profile.get("provider") == "rapidapi"' in SOURCE
    assert "ThreadPoolExecutor(max_workers=2" in SOURCE
    assert "_run_source_profile" in SOURCE


def test_source_winner_is_logged_to_console_and_jsonl():
    assert "RIPPED_SOURCE_WINNER" in SOURCE
    assert "source_acquisition.jsonl" in SOURCE
    assert '"provider": provider' in SOURCE
    assert '"profile": profile.get("name")' in SOURCE
