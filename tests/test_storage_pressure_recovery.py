from pathlib import Path

SOURCE=(Path(__file__).resolve().parents[1]/"telegram_intake.py").read_text(encoding="utf-8")


def test_storage_snapshot_and_pressure_guard_exist():
    assert "def _storage_snapshot()" in SOURCE
    assert "def _ensure_storage_headroom()" in SOURCE
    assert "RIPPED_STORAGE_PRESSURE" in SOURCE
    assert "MIN_LOCAL_FREE_BYTES" in SOURCE
    assert "MIN_LOCAL_FREE_PERCENT" in SOURCE


def test_persistent_artifact_cleanup_runs_at_startup():
    assert "RIPPED_STORAGE_READY" in SOURCE
    assert "_purge_expired_local_artifacts()" in SOURCE


def test_sqlite_disk_full_triggers_emergency_cleanup_and_retry():
    assert "RIPPED_STORAGE_SQLITE_FULL" in SOURCE
    assert "_purge_expired_local_artifacts(emergency=True)" in SOURCE
    assert "RIPPED_STORAGE_RETRY_SUCCESS" in SOURCE


def test_default_local_retention_is_shorter_than_before():
    assert 'LOCAL_ARTIFACT_RETENTION_HOURS", "12"' in SOURCE
