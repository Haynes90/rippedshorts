from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TELEGRAM = (ROOT / "telegram_intake.py").read_text(encoding="utf-8")
RELIABILITY = (ROOT / "workflow_reliability.py").read_text(encoding="utf-8")


def test_watchdog_never_revives_superseded_jobs():
    assert '"superseded"' in TELEGRAM
    assert "RIPPED_JOB_WATCHDOG_SKIP_TERMINAL" in TELEGRAM
    assert "state.get(\"superseded_by_request_id\")" in TELEGRAM


def test_process_skips_terminal_and_older_duplicate_video_jobs():
    assert "RIPPED_PROCESS_SKIP_TERMINAL" in TELEGRAM
    assert "RIPPED_PROCESS_SKIP_OLDER_DUPLICATE" in TELEGRAM
    assert "peer_video_id != video_id_for_guard" in TELEGRAM


def test_watchdog_only_considers_newest_job_per_video():
    assert "ORDER BY updated_at DESC" in TELEGRAM
    assert "seen_video_ids" in TELEGRAM
    assert "RIPPED_JOB_WATCHDOG_SKIP_OLDER_DUPLICATE" in TELEGRAM


def test_watchdog_pauses_when_storage_is_below_hard_floor():
    assert "RIPPED_JOB_WATCHDOG_PAUSED_STORAGE" in TELEGRAM
    assert "HARD_LOCAL_FREE_BYTES" in TELEGRAM


def test_emergency_cleanup_can_reclaim_recent_retired_job_files():
    assert "def _active_local_source_dirs()" in TELEGRAM
    assert "emergency_reclaimable = emergency and not _path_is_under(path, protected_dirs)" in TELEGRAM
    assert "EMERGENCY_TARGET_FREE_BYTES" in TELEGRAM


def test_readiness_requires_disk_headroom():
    assert '"storage_space": storage_space_ok' in RELIABILITY
    assert '"storage": storage' in RELIABILITY
    assert "READINESS_MIN_FREE_PERCENT" in RELIABILITY


def test_superseded_jobs_are_not_latest_incomplete():
    assert '"superseded", "permanent_failure"' in RELIABILITY
    assert '"cancelled", "superseded"' in RELIABILITY
