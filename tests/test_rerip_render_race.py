from pathlib import Path

SOURCE=(Path(__file__).resolve().parents[1]/"telegram_intake.py").read_text(encoding="utf-8")


def test_duplicate_rerip_buttons_have_no_active_execution_path():
    assert 'RERIP_REUSE_IGNORED_ALREADY_PROCESSING' not in SOURCE
    assert 'rs:rerip:' not in SOURCE and 'rs:reuse:' not in SOURCE


def test_old_render_buttons_do_not_fail_when_result_is_missing():
    assert "RENDER_IGNORED_RESULT_NOT_READY" in SOURCE
    assert '"⏳ This job is still rebuilding its fresh clip list. "' in SOURCE
    assert 'result = state.get("result") or {}' in SOURCE
    assert 'candidate = segments[index]' in SOURCE


def test_fresh_candidate_state_clears_rebuild_flag():
    assert '"rebuild_in_progress": False' in SOURCE


def test_decision_logger_skips_missing_result_instead_of_throwing():
    assert "DECISION_LOG_SKIPPED_RESULT_NOT_READY" in SOURCE
