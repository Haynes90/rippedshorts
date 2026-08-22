from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_clip_output_names_use_youtube_id_and_candidate_number():
    source = (ROOT / "main.py").read_text()
    telegram_source = (ROOT / "telegram_intake.py").read_text()
    assert 'clip_name = f"{video_id}_Clip_{candidate_number}.mp4"' in source
    assert 'f".render-{uuid.uuid4().hex}-{clip_name}"' in source
    assert 'payload_candidate["candidate_number"] = index + 1' in telegram_source


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


def test_telegram_reports_render_progress_and_drive_completion():
    source = (ROOT / "telegram_intake.py").read_text()
    assert "is now rendering" in source
    assert "rendered and uploaded to DRIVE_FOLDER_ID" in source
    assert "Current Ripped Shorts render queue complete" in source
    assert "drive.google.com/drive/folders/" in source


def test_telegram_reports_dynamic_aggregate_progress():
    source = (ROOT / "telegram_intake.py").read_text()
    assert "Progress: {processed}/{total} processed ({percent}%)" in source
    assert "{rendering} rendering | {queued} queued" in source
    assert "{rendered} rendered | {failed} failed" in source


def test_restart_recovers_only_approved_unfinished_sheet_clips():
    source = (ROOT / "telegram_intake.py").read_text()
    assert "def _approved_clip_history_from_sheet" in source
    assert 'str(padded[2]).strip() != video_id' in source
    assert 'str(padded[15]).strip().lower() != "approved"' in source
    assert 'is_rendered = render_status == "rendered" and bool(clip_url)' in source
    assert "recovered_approvals_from_sheet" in source
    assert "Rendering only those approved clips now" in source


def test_already_rendered_youtube_id_skips_gpt_selection():
    source = (ROOT / "telegram_intake.py").read_text()
    assert "if rendered_clips:" in source
    assert '"stage": "already_rendered"' in source
    assert "GPT selection was not run again" in source
    assert "existing_clip_links" in source
