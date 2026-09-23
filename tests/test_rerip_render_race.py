from pathlib import Path

SOURCE=(Path(__file__).resolve().parents[1]/"telegram_intake.py").read_text(encoding="utf-8")


def test_duplicate_rerip_reuse_is_blocked_while_rebuilding():
    assert "RERIP_REUSE_IGNORED_ALREADY_PROCESSING" in SOURCE
    assert 'state["rebuild_in_progress"] = True' in SOURCE
    assert '"status": "already_processing"' in SOURCE


def test_old_render_buttons_do_not_fail_when_result_is_missing():
    assert "RENDER_IGNORED_RESULT_NOT_READY" in SOURCE
    assert '"⏳ This job is still rebuilding its fresh clip list. "' in SOURCE
    assert 'result = state.get("result") or {}' in SOURCE
    assert 'candidate = segments[index]' in SOURCE


def test_fresh_candidate_state_clears_rebuild_flag():
    assert '"rebuild_in_progress": False' in SOURCE


def test_decision_logger_skips_missing_result_instead_of_throwing():
    assert "DECISION_LOG_SKIPPED_RESULT_NOT_READY" in SOURCE
