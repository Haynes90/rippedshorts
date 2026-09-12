"""Atomic SQLite leases for restart-safe Ripped Shorts work.

Every machine step is claimed with BEGIN IMMEDIATE. A second worker cannot run the
same job/action until the lease expires. Completed actions remain idempotent.
"""
from __future__ import annotations

import functools
import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

logger = logging.getLogger("ripped-shorts.durable")
OWNER = os.getenv("RAILWAY_REPLICA_ID", "").strip() or str(uuid.uuid4())
LEASE_SECONDS = max(60, int(os.getenv("JOB_LEASE_SECONDS", "900")))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_schema(db: sqlite3.Connection) -> None:
    db.execute(
        """CREATE TABLE IF NOT EXISTS durable_job_leases (
        job_id TEXT NOT NULL,
        action TEXT NOT NULL,
        owner TEXT NOT NULL,
        lease_until TEXT NOT NULL,
        attempt_count INTEGER NOT NULL DEFAULT 1,
        state TEXT NOT NULL DEFAULT 'RUNNING',
        last_error TEXT NOT NULL DEFAULT '',
        updated_at TEXT NOT NULL,
        PRIMARY KEY(job_id, action))"""
    )
    db.execute(
        """CREATE TABLE IF NOT EXISTS durable_outbox (
        event_id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL,
        destination TEXT NOT NULL,
        payload_json TEXT NOT NULL DEFAULT '',
        state TEXT NOT NULL DEFAULT 'PENDING',
        attempt_count INTEGER NOT NULL DEFAULT 0,
        next_attempt_at TEXT NOT NULL DEFAULT '',
        receipt TEXT NOT NULL DEFAULT '',
        last_error TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL)"""
    )
    db.execute("CREATE INDEX IF NOT EXISTS idx_outbox_state ON durable_outbox(state, next_attempt_at)")


def _connect(db_path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(db_path, timeout=30, isolation_level=None)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=30000")
    ensure_schema(db)
    return db


def claim(db_path: Path, job_id: str, action: str, lease_seconds: int = LEASE_SECONDS) -> bool:
    now = _utc_now()
    until = (now + timedelta(seconds=lease_seconds)).isoformat()
    with _connect(db_path) as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT owner, lease_until, state, attempt_count FROM durable_job_leases "
            "WHERE job_id=? AND action=?", (job_id, action)
        ).fetchone()
        if row:
            if row["state"] == "COMPLETE":
                db.execute("COMMIT")
                return False
            try:
                expiry = datetime.fromisoformat(str(row["lease_until"]).replace("Z", "+00:00"))
            except ValueError:
                expiry = now - timedelta(seconds=1)
            if expiry > now and row["owner"] != OWNER:
                db.execute("COMMIT")
                return False
            db.execute(
                "UPDATE durable_job_leases SET owner=?, lease_until=?, "
                "attempt_count=attempt_count+1, state='RUNNING', last_error='', updated_at=? "
                "WHERE job_id=? AND action=?",
                (OWNER, until, now.isoformat(), job_id, action),
            )
        else:
            db.execute(
                "INSERT INTO durable_job_leases "
                "(job_id, action, owner, lease_until, attempt_count, state, updated_at) "
                "VALUES (?, ?, ?, ?, 1, 'RUNNING', ?)",
                (job_id, action, OWNER, until, now.isoformat()),
            )
        db.execute("COMMIT")
    return True


def finish(db_path: Path, job_id: str, action: str, error: str = "", *, complete: bool = False) -> None:
    now = _utc_now().isoformat()
    state = "RETRY_WAIT" if error else ("COMPLETE" if complete else "RELEASED")
    with _connect(db_path) as db:
        db.execute(
            "UPDATE durable_job_leases SET state=?, lease_until=?, last_error=?, updated_at=? "
            "WHERE job_id=? AND action=? AND owner=?",
            (state, now, error[:2000], now, job_id, action, OWNER),
        )


def durable_job(action: str, *, idempotent: bool = False):
    """Decorate a function whose first argument is its stable request/job ID."""
    def decorate(function):
        @functools.wraps(function)
        def wrapped(job_id, *args, **kwargs):
            # Lazy import avoids the telegram/audio module import cycle.
            from audio_master_handoff import DB_PATH
            path = Path(DB_PATH)
            if not claim(path, str(job_id), action):
                logger.info("JOB_LEASE_SKIPPED job_id=%s action=%s", job_id, action)
                return None
            try:
                result = function(job_id, *args, **kwargs)
            except Exception as exc:
                finish(path, str(job_id), action, f"{type(exc).__name__}: {exc}")
                raise
            finish(path, str(job_id), action, complete=bool(idempotent and result is True))
            return result
        return wrapped
    return decorate


def lease_snapshot(db_path: Path) -> dict:
    with _connect(db_path) as db:
        rows = db.execute(
            "SELECT state, COUNT(*) AS count FROM durable_job_leases GROUP BY state"
        ).fetchall()
    return {str(row["state"]): int(row["count"]) for row in rows}
