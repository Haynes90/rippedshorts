from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_clip_output_names_use_youtube_id_and_candidate_number():
    source = (ROOT / "main.py").read_text()
    telegram_source = (ROOT / "telegram_intake.py").read_text()
    assert 'clip_name = f"{video_id}_Clip_{candidate_number}.mp4"' in source
    assert 'f".render-{uuid.uuid4().hex}-{clip_name}"' in source
    assert 'payload_candidate["candidate_number"] = int(' in telegram_source


def test_render_failure_includes_ffmpeg_diagnostics():
    source = (ROOT / "main.py").read_text()
    assert "FFmpeg render failed with code" in source
    assert "(completed.stderr or completed.stdout)[-3000:]" in source


def test_selector_aims_for_twenty_and_collects_future_learning_metadata():
    source = (ROOT / "main.py").read_text()
    assert "Target 18–20 final clips" in source
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
    assert "Rendering only those shorts now" in source


def test_already_rendered_youtube_id_skips_gpt_selection():
    source = (ROOT / "telegram_intake.py").read_text()
    assert "if rendered_clips:" in source
    assert '"stage": "already_rendered"' in source
    assert "GPT selection was not run again" in source
    assert "existing_clip_links" in source


def test_reused_video_render_creates_output_directory():
    source = (ROOT / "main.py").read_text()
    assert 'workdir.mkdir(parents=True, exist_ok=True)' in source


def test_selector_targets_twenty_and_allows_distinct_overlap():
    prompt_source = (ROOT / "main.py").read_text()
    intake_source = (ROOT / "telegram_intake.py").read_text()
    selector_source = (ROOT / "source_ingestion.py").read_text()
    assert "Target 18–20 final clips" in prompt_source
    assert "return 20" in prompt_source
    assert "Clips MAY overlap in time" in prompt_source
    assert "allow_overlap=True" in intake_source
    assert "normalized_text" in selector_source


def test_under_ten_reviewed_shorts_triggers_additional_search():
    source = (ROOT / "telegram_intake.py").read_text()
    assert "def _reviewed_short_history_from_sheet" in source
    assert "if len(prior_shorts) >= 10:" in source
    assert "Searching again for up to {needed} additional distinct shorts" in source
    assert "PREVIOUSLY REVIEWED SHORTS" in source


def test_telegram_uses_short_not_candidate_label():
    source = (ROOT / "telegram_intake.py").read_text()
    assert 'f"Short {short_number}' in source
    assert "Approve only the shorts you want created" in source


def test_opencv_speaker_tracking_and_center_fallback():
    source = (ROOT / "main.py").read_text()
    docker = (ROOT / "Dockerfile").read_text()
    assert "opencv-data" in docker
    assert "def _estimate_speaker_track" in source
    assert "motion_score * 3.0" in source
    assert "Dead zone and exponential smoothing" in source
    assert "target_center = previous_center" in source
    assert "def _piecewise_crop_expression" in source
