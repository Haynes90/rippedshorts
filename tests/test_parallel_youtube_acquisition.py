from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "source_ingestion.py").read_text(encoding="utf-8")
DOCKER = (ROOT / "Dockerfile").read_text(encoding="utf-8")


def test_po_token_runtime_is_installed_in_image():
    assert "bgutil-ytdlp-pot-provider.git" in DOCKER
    assert "/opt/bgutil-ytdlp-pot-provider" in DOCKER
    assert "deno install" in DOCKER


def test_po_token_readiness_is_logged():
    assert "PO_TOKEN_PROVIDER ready=" in SOURCE
    assert "automatic_po_token_mweb" in SOURCE


def test_youtube_acquisition_uses_two_parallel_lanes():
    assert "ThreadPoolExecutor(max_workers=2" in SOURCE
    assert "return_when=FIRST_COMPLETED" in SOURCE
    assert '".youtube-lane-' in SOURCE


def test_failed_lane_is_replaced_and_winner_cancels_others():
    assert "replacement_profile = next(pending_profiles)" in SOURCE
    assert "stop_event.set()" in SOURCE
    assert "future.cancel()" in SOURCE
    assert "RIPPED_SOURCE_RACE winner" in SOURCE
