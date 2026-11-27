from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PORTAL = (ROOT / "caption_review_web.py").read_text(encoding="utf-8")
TELEGRAM = (ROOT / "telegram_intake.py").read_text(encoding="utf-8")
MAIN = (ROOT / "main.py").read_text(encoding="utf-8")


def test_mobile_review_portal_is_mounted():
    assert "caption_review_web_router" in MAIN
    assert 'router.get("/review/{token}")' in PORTAL
    assert 'router.post("/api/ripped-shorts/review/{token}/asset/{index}")' in PORTAL
    assert 'router.post("/api/ripped-shorts/review/{token}/finish")' in PORTAL


def test_review_session_is_temporary_but_saved_copy_is_durable():
    assert 'COPY_REVIEW_TTL_DAYS", "14"' in PORTAL
    assert "CREATE TABLE IF NOT EXISTS copy_review_sessions" in PORTAL
    assert "_checkpoint_copy_edit" in PORTAL
    assert "_log_copy_learning" in PORTAL
    assert "_notify_render_queue_complete" in PORTAL


def test_telegram_directs_review_to_website_without_competing_controls():
    assert 'def _website_navigation' in TELEGRAM
    assert 'from review_navigation import notify_safely' in TELEGRAM
    assert 'Finish & Schedule' not in TELEGRAM
    assert 'rs:copy_finish:' not in TELEGRAM


def test_mobile_page_groups_source_and_copy_assets():
    assert "Source video" in PORTAL
    assert "9:16 Short" in PORTAL
    assert "16:9 Highlight" in PORTAL
    assert "Finish &amp; Schedule" in PORTAL
    assert "Saved ✓" in PORTAL


def test_finish_saves_current_written_copy_before_scheduling():
    assert "Saving edits…" in PORTAL
    assert "document.querySelectorAll('.asset')" in PORTAL
    assert "await saveAsset(index, 'save')" in PORTAL
    assert "Scheduling is pending; no successful handoff is claimed" in PORTAL
    assert "data.status === 'scheduler_accepted'" in PORTAL
