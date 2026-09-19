"""Exercise shared-state merging and real FFmpeg subtitle rendering offline."""
import json
import shutil
import subprocess
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest

from test_review_hub import studio


def test_highlight_selection_merges_concurrent_short_review(studio, monkeypatch):
    _, fake, seed = studio
    import telegram_intake as engine
    seed(state={"parsed":{"mode":"both"}})
    monkeypatch.setattr(engine, "_LOCK", fake._LOCK)
    monkeypatch.setattr(engine, "_telegram_db", fake._telegram_db)
    monkeypatch.setattr(engine, "send", Mock())
    monkeypatch.setattr(engine, "_send_topic_candidates", Mock())
    def suggestions(_):
        with fake._telegram_db() as db:
            row=db.execute("SELECT state_json FROM telegram_requests WHERE request_id='one'").fetchone()
            state=json.loads(row["state_json"])
            state["candidate_reviews"]={"0":{"status":"queued"}}
            state["web_captions"]={"shorts:0":{"text":"Concurrent caption edit"}}
            db.execute("UPDATE telegram_requests SET state_json=?",(json.dumps(state),))
        return []
    monkeypatch.setattr(engine,"_topic_break_suggestions",suggestions)
    monkeypatch.setattr(engine,"_build_contiguous_topic_segments",lambda *_:[{"start":0,"duration":180}])
    result=engine._process_topics("one",{},"chat","video",Path("source.mp4"),[],True)
    assert result["candidate_reviews"]["0"]["status"]=="queued"
    assert result["web_captions"]["shorts:0"]["text"]=="Concurrent caption edit"


def test_web_jobs_launch_highlights_only_once(studio, monkeypatch):
    _, fake, seed=studio
    import telegram_intake as engine
    seed(state={"review_hub":True,"parsed":{"mode":"both"},"topic_source_segments":[{"text":"Hello"}]})
    for key in ("_LOCK","_telegram_db","RENDER_EXECUTOR"):
        monkeypatch.setattr(engine,key,getattr(fake,key))
    monkeypatch.setattr(engine,"upsert_job",Mock())
    monkeypatch.setattr(engine,"STATUS_EXECUTOR",Mock())
    stale={"review_hub":True,"topic_source_segments":[{"text":"Hello"}],"topic_stage":None}
    engine._save("one","awaiting_review",dict(stale))
    engine._save("one","awaiting_review",dict(stale))
    fake.RENDER_EXECUTOR.submit.assert_called_once()


