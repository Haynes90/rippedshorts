from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[1] / "telegram_intake.py").read_text(encoding="utf-8")


def test_render_paths_rehydrate_missing_source_files():
    assert "def _ensure_render_source(" in SOURCE
    assert "reuse_from_drive(video_id, recovery_dir)" in SOURCE
    assert "download_youtube_resilient(video_id, youtube_url, recovery_dir)" in SOURCE
    assert "RENDER_SOURCE_REHYDRATED" in SOURCE


def test_short_and_topic_renders_use_source_recovery():
    assert SOURCE.count("video = _ensure_render_source(request_id, state)") >= 2
    assert 'video = Path(state["video_path"])' not in SOURCE


def test_recovered_path_is_persisted_back_to_job_state():
    assert 'state["video_path"] = str(path)' in SOURCE
    assert 'state["source_rehydrated_at"] = now()' in SOURCE
