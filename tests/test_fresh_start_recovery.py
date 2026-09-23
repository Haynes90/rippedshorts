from pathlib import Path

SOURCE=(Path(__file__).resolve().parents[1]/"telegram_intake.py").read_text(encoding="utf-8")


def test_recovered_approved_clips_clear_rebuild_before_render():
    assert '"rebuild_in_progress": False' in SOURCE
    assert '"recovered_approvals_from_sheet": True' in SOURCE


def test_fresh_start_creates_new_request_and_preserves_learning():
    assert "def _fresh_start_from_request(" in SOURCE
    assert '"approval_learning_preserved": True' in SOURCE
    assert '"supersedes_request_id": request_id' in SOURCE
    assert 'old_state["superseded_by_request_id"] = new_request_id' in SOURCE
    assert '"🧼 Fresh Start — Keep Learning"' in SOURCE
    assert 'r"/fresh-start' in SOURCE


def test_render_failures_keep_durable_history():
    assert '"render_failure_history"' in SOURCE
    assert '"last_render_failure"' in SOURCE
    assert '"error_type": type(exc).__name__' in SOURCE


def test_fresh_start_does_not_copy_old_result_or_reviews():
    helper=SOURCE.split("def _fresh_start_from_request(",1)[1].split("def _render_progress_text",1)[0]
    assert '"result"' not in helper
    assert '"candidate_reviews"' not in helper
    assert '"topic_reviews"' not in helper
