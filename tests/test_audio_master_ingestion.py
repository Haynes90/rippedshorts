from pathlib import Path


INGEST = (Path(__file__).resolve().parents[1] / "source_ingestion.py").read_text(encoding="utf-8")
TELEGRAM = (Path(__file__).resolve().parents[1] / "telegram_intake.py").read_text(encoding="utf-8")


def test_missing_assets_are_delegated_to_audio_master():
    assert "ingest_with_audio_master(video_id, parsed[\"source_value\"])" in TELEGRAM
    assert "download_youtube_resilient(video_id" not in TELEGRAM
    assert "Metadata and Podhome publishing are disabled" in TELEGRAM


def test_audio_master_client_uses_protected_dedicated_routes():
    assert 'f"{base}/api/ripped-shorts/ingest"' in INGEST
    assert 'f"{base}/api/ripped-shorts/ingest/{job_id}"' in INGEST
    assert '"x-ripped-shorts-ingest-secret": secret' in INGEST


def test_audio_master_client_requires_transcript_and_source_video():
    assert "transcript_ready and source_ready" in INGEST
    assert "Audio Master ingestion timed out" in INGEST


def test_audio_master_configuration_names_are_explicit():
    assert 'os.getenv("AUDIO_MASTER_INTERNAL_URL")' in INGEST
    assert 'os.getenv("AUDIO_MASTER_INGEST_SECRET")' in INGEST
    assert 'os.getenv("AUDIO_MASTER_WEBHOOK_SECRET")' in INGEST
