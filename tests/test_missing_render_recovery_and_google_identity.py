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
