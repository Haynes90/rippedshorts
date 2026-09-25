from pathlib import Path

INTAKE=(Path(__file__).resolve().parents[1]/"telegram_intake.py").read_text(encoding="utf-8")
MAIN=(Path(__file__).resolve().parents[1]/"main.py").read_text(encoding="utf-8")


def test_missing_approved_renders_can_be_requeued_from_sheet():
    assert "def _queue_missing_approved_renders(" in INTAKE
    assert "_approved_clip_history_from_sheet(video_id)" in INTAKE
    assert '"♻️ Retry Missing Renders"' in INTAKE
    assert 'r"/render-missing' in INTAKE


def test_missing_render_recovery_uses_exact_approved_sheet_segments():
    assert '"segments": missing' in INTAKE
    assert '"recovered_from_sheet": True' in INTAKE
    assert "RENDER_EXECUTOR.submit(_render_approved" in INTAKE


def test_google_service_account_identity_is_logged():
    assert "def google_credential_identity()" in MAIN
    assert "GOOGLE_SERVICE_ACCOUNT source=%s client_email=%s" in MAIN
    assert "GOOGLE_SERVICE_ACCOUNT_MISMATCH" in MAIN
    assert "because GOOGLE_CREDENTIALS takes precedence" in MAIN


def test_missing_render_recovery_redirects_old_cards_to_active_job():
    assert "def _active_request_for_video(" in INTAKE
    assert "RENDER_RECOVERY_REDIRECT" in INTAKE
    assert "That review card belongs to an older job" in INTAKE


def test_render_recovery_checkpoints_durable_state():
    assert '"stage": "render_recovery"' in INTAKE
    assert '_save(request_id, "render_recovery", state)' in INTAKE
    assert '_save(request_id, "awaiting_render_completion", state)' in INTAKE
    assert "awaiting_render_retry" in INTAKE


def test_watchdog_recovers_stuck_failed_approved_renders_once():
    assert "has_failed_approved_render" in INTAKE
    assert "RIPPED_JOB_WATCHDOG_RENDER_RECOVERY" in INTAKE
    assert "_queue_missing_approved_renders" in INTAKE


def test_failed_approved_renders_block_schedule_handoff():
    assert "Never hand off while an approved render is missing" in INTAKE
    assert "if should_handoff:" in INTAKE
