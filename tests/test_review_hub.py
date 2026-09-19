import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import review_hub as hub


@pytest.fixture
def studio(tmp_path, monkeypatch):
    def db():
        conn = sqlite3.connect(tmp_path / "test.db")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE IF NOT EXISTS telegram_requests (request_id TEXT PRIMARY KEY, update_id TEXT UNIQUE, chat_id TEXT, user_id TEXT, status TEXT, mode TEXT, source_kind TEXT, source_value TEXT, state_json TEXT, created_at TEXT, updated_at TEXT)")
        return conn
    clock = lambda: datetime.now(timezone.utc).isoformat()
    fake = SimpleNamespace(_LOCK=threading.RLock(), _telegram_db=db, now=clock,
        RENDER_EXECUTOR=SimpleNamespace(submit=Mock()), _render_approved=Mock(), _render_topic_approved=Mock(),
        _safe_log_candidate=Mock(), _process=Mock(), _start_16_9_after_confirmation=Mock(), SOURCE_DIR=tmp_path)
    monkeypatch.setattr(hub, "engine", lambda: fake)
    monkeypatch.setenv("REVIEW_HUB_SECRET", "a" * 40)
    monkeypatch.setenv("REVIEW_HUB_CHAT_ID", "chat")
    monkeypatch.setenv("REVIEW_HUB_USER_ID", "user")
    app = FastAPI()
    app.include_router(hub.router)
    client = TestClient(app)
    client.post("/review/api/session", json={"secret": "a" * 40})
    client.headers["X-Review-Request"] = "1"
    def seed(id="one", chat="chat", status="awaiting_review", days=0, state=None):
        now = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        base = {"vid_title": "Test video", "result": {"segments": [{"title": "A short", "start": 2, "duration": 10, "transcript": "A full thought."}]},
                "topic_result": {"segments": [{"title": "A highlight", "start": 0, "duration": 180}]}, "candidate_reviews": {}, "topic_reviews": {}}
        base.update(state or {})
        with db() as conn:
            conn.execute("INSERT INTO telegram_requests VALUES (?,?,?,?,?,?,?,?,?,?,?)", (id,id,chat,"user",status,"both","youtube","url",json.dumps(base),now,now))
    return client, fake, seed


def test_auth_fails_closed(studio, monkeypatch):
    client, _, _ = studio
    client.cookies.clear()
    assert client.get("/review/api/projects").status_code == 401
    monkeypatch.delenv("REVIEW_HUB_SECRET")
    assert client.get("/review/api/projects").status_code == 503


def test_session_tampering_and_csrf(studio):
    client, _, seed = studio
    seed()
    client.headers.pop("X-Review-Request")
    assert client.post("/review/api/projects/one/shorts/0/decision", json={"decision":"approve"}).status_code == 403
    client.cookies.set(hub.COOKIE, "9999999999.fake.fake")
    assert client.get("/review/api/projects").status_code == 401


def test_owner_isolation_for_reads_and_writes(studio):
    client, engine, seed = studio
    seed(); seed("other", chat="other")
    assert len(client.get("/review/api/projects").json()["projects"]) == 1
    assert client.get("/review/api/projects/other").status_code == 404
    assert client.post("/review/api/projects/other/shorts/0/decision", json={"decision":"approve"}).status_code == 404
    engine.RENDER_EXECUTOR.submit.assert_not_called()


@pytest.mark.parametrize("lane,worker", [("shorts","_render_approved"),("highlights","_render_topic_approved")])
def test_approval_idempotency_and_render_dispatch(studio, lane, worker):
    client, engine, seed = studio
    seed()
    path=f"/review/api/projects/one/{lane}/0/decision"
    assert client.post(path,json={"decision":"approve"}).json()["status"] == "queued"
    assert client.post(path,json={"decision":"approve"}).status_code == 200
    engine.RENDER_EXECUTOR.submit.assert_called_once_with(getattr(engine,worker),"one",0,"chat")
    assert client.post(path,json={"decision":"reject"}).status_code == 409


def test_invalid_clip_never_dispatches(studio):
    client, engine, seed = studio
    seed()
    for lane,index in [("shorts",-1),("shorts",5),("wrong",0)]:
        assert client.post(f"/review/api/projects/one/{lane}/{index}/decision",json={"decision":"approve"}).status_code in (404,422)
    engine.RENDER_EXECUTOR.submit.assert_not_called()


def test_expiry_blocks_mutation_and_source(studio):
    client, engine, seed = studio
    seed(days=31)
    assert client.post("/review/api/projects/one/shorts/0/decision",json={"decision":"approve"}).status_code == 410
    assert client.get("/review/api/projects/one/source").status_code == 410
    assert client.get("/review/api/projects/one").json()["expired"]






def test_source_path_cannot_escape_root(studio, tmp_path):
    client, _, seed = studio
    seed(state={"video_path":str(tmp_path.parent / "secret.mp4")})
    assert client.get("/review/api/projects/one/source").status_code == 404


def test_highlight_release_preserves_short_decisions(studio):
    client, engine, seed = studio
    seed(state={"topic_source_segments":[{"text":"Hello"}], "candidate_reviews":{"0":{"status":"rendered"}}})
    path="/review/api/projects/one/highlights"
    assert client.post(path).json()["status"] == "queued"
    client.post(path)
    engine.RENDER_EXECUTOR.submit.assert_called_once()
    assert client.get("/review/api/projects/one").json()["lanes"]["shorts"][0]["status"] == "rendered"








def test_no_estimate_until_enough_history(studio):
    client, _, seed = studio
    seed(status="processing")
    assert client.get("/review/api/projects/one").json()["eta_seconds"] is None
