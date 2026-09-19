"""LOCAL FIXTURE ONLY. No external work is submitted. Run from the repository root."""
import json
import os
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import review_hub as hub
from fastapi import FastAPI

root = Path("review-test-output/demo")
root.mkdir(parents=True, exist_ok=True)
def db():
    conn = sqlite3.connect(root / "demo.db")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE IF NOT EXISTS telegram_requests (request_id TEXT PRIMARY KEY, update_id TEXT, chat_id TEXT, user_id TEXT, status TEXT, mode TEXT, source_kind TEXT, source_value TEXT, state_json TEXT, created_at TEXT, updated_at TEXT)")
    return conn
stamp = datetime.now(timezone.utc).isoformat()
state = {"vid_title":"Demo · Conversations that matter", "stage":"awaiting_review", "topic_stage":"awaiting_review",
    "result":{"segments":[{"title":"Start before you feel ready", "start":0,"duration":12,"transcript":"You do not have to see the whole path to take the first step."},
                           {"title":"Consistency beats intensity", "start":14,"duration":15,"transcript":"Small decisions repeated daily become the life you build."}]},
    "topic_result":{"segments":[{"title":"Building a practice that lasts", "start":30,"duration":182,"transcript":"A longer conversation about showing up, learning, and doing the work."}]},
    "candidate_reviews":{"0":{"status":"rendered"}}, "topic_reviews":{},
    "web_captions":{"shorts:0":{"text":"Take the first step.", "preset":"clean","placement":"lower", "revision":"demo", "status":"draft", "duration":12,
        "cues":[{"start":0,"end":2,"text":"You do not have to"},{"start":2,"end":4,"text":"see the whole path."}]}}}
with db() as conn:
    conn.execute("INSERT OR REPLACE INTO telegram_requests VALUES (?,?,?,?,?,?,?,?,?,?,?)",("demo","demo","demo-chat","demo-user","awaiting_review","both","youtube","demo",json.dumps(state),stamp,stamp))
fake = SimpleNamespace(_LOCK=threading.RLock(),_telegram_db=db,now=lambda:datetime.now(timezone.utc).isoformat(),
    RENDER_EXECUTOR=SimpleNamespace(submit=Mock()),_render_approved=Mock(),_render_topic_approved=Mock(),_safe_log_candidate=Mock(),SOURCE_DIR=root,
    _process=Mock(),_start_16_9_after_confirmation=Mock())
hub.engine=lambda:fake
os.environ["REVIEW_HUB_SECRET"]="local-demo-access-key-not-for-production"
os.environ["REVIEW_HUB_CHAT_ID"]="demo-chat"
os.environ["REVIEW_HUB_USER_ID"]="demo-user"
app=FastAPI()
app.include_router(hub.router)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8765)
