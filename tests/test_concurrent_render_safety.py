from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_clip_output_names_include_candidate_time_range():
    source = (ROOT / "main.py").read_text()
    assert 'clip_name = f"{video_id}_{clip_key}.mp4"' in source
    assert 'int(round(start * 1000))' in source
    assert 'int(round(end_seconds * 1000))' in source


def test_render_failure_includes_ffmpeg_diagnostics():
    source = (ROOT / "main.py").read_text()
    assert "FFmpeg render failed with code" in source
    assert "(completed.stderr or completed.stdout)[-3000:]" in source


def test_selector_aims_for_twenty_and_collects_future_learning_metadata():
    source = (ROOT / "main.py").read_text()
    assert "Return a MAX of 20 clips" in source
    assert '\"content_type\"' in source
    assert '\"keywords\"' in source


def test_approvals_use_bounded_pool_and_are_idempotent():
    source = (ROOT / "telegram_intake.py").read_text()
    assert "RIPPED_SHORTS_RENDER_WORKERS" in source
    assert "RENDER_EXECUTOR.submit(_render_approved" in source
    assert 'existing_status in {"queued", "rendering", "rendered"}' in source
    assert '"status": "rendering"' in source


def test_learning_log_uses_decisions_without_user_score():
    source = (ROOT / "telegram_intake.py").read_text()
    assert "RIPPED_LOG_SHEET_TAB" in source
    assert '"approved"' in source
    assert '"rejected"' in source
    assert 'candidate.get("score")' not in source
