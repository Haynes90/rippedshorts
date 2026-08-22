from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[1] / "source_ingestion.py").read_text(encoding="utf-8")
TELEGRAM = (Path(__file__).resolve().parents[1] / "telegram_intake.py").read_text(encoding="utf-8")


def test_bare_audio_master_domain_gets_https_scheme():
    assert 'if base and "://" not in base' in SOURCE
    assert 'base = f"https://{base}"' in SOURCE


def test_accidental_endpoint_suffix_is_removed():
    assert '"/api/ripped-shorts/ingest"' in SOURCE
    assert "base = base[: -len(suffix)].rstrip" in SOURCE


def test_drive_reuse_searches_configured_folder_and_global_visibility():
    assert "_folder_query(video_id)" in SOURCE
    assert "global_result = service.files().list" in SOURCE
    assert "name contains" in SOURCE


def test_source_video_failure_is_reported_before_poll_timeout():
    assert '"awaiting_route_rerun"' in SOURCE
    assert "retained source-video path failed" in SOURCE


def test_safe_runtime_diagnostics_cover_required_dependencies():
    assert '@router.get("/api/ripped-shorts/runtime-info")' in TELEGRAM
    assert '"audio_master_ingest_secret_set"' in TELEGRAM
    assert '"google_credentials_set"' in TELEGRAM
    assert '"ingestion_mode": "drive_id_reuse_then_audio_master"' in TELEGRAM
