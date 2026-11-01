"""Isolated state/provider fixtures; no live job, Sheet or Telegram operations."""
import json
import sqlite3
import tempfile
from pathlib import Path
from contextlib import contextmanager, ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch


@contextmanager
def copy_review_fixture():
    import telegram_intake as tg
    import caption_review_web as web
    class Connection(sqlite3.Connection):
        def __exit__(self, *args):
            try:
                return super().__exit__(*args)
            finally:
                self.close()
    with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
        path = Path(folder) / 'isolated.db'
        def connect():
            db = sqlite3.connect(path, factory=Connection)
            db.row_factory = sqlite3.Row
            return db
        stack.enter_context(patch.object(tg, 'connect', connect))
        telegram = stack.enter_context(patch.object(tg, 'telegram', Mock(return_value={'result': {'message_id': 300}})))
        stack.enter_context(patch.object(tg, 'send', Mock()))
        stack.enter_context(patch.object(web, 'review_url_for_request', lambda _: 'https://isolated.test/review'))
        observed = []
        def read(request='first'):
            with tg._telegram_db() as db:
                return json.loads(db.execute('SELECT state_json FROM telegram_requests WHERE request_id=?', (request,)).fetchone()[0])
        def checkpoint(request, state, draft, user):
            observed.append((request, read(request), dict(draft)))
        stack.enter_context(patch.object(tg, '_checkpoint_copy_edit', checkpoint))
        for i, request in enumerate(('first', 'second')):
            state = {'show_id': 'TCB', 'route_source': 'owner', 'parsed': {'video_id': 'source'}, 'copy_review_message_id': 200+i,
                     'copy_drafts': [{'asset_id': request+':short:1', 'asset_type': '9:16_SHORT', 'candidate_number': 1,
                                      'social_caption': 'Original '+request, 'hashtags': '#Plants', 'copy_status': 'pending'}],
                     'awaiting_copy_input': {'field': 'caption', 'index': 0, 'user_id': 'owner', 'prompt_message_id': str(100+i)}}
            with tg._telegram_db() as db:
                db.execute('INSERT INTO telegram_requests VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                           (request, request, 'chat', 'owner', 'awaiting_copy_input', 'both', 'youtube', 'https://fixture.test', json.dumps(state), 'now', 'now'))
                db.execute('INSERT INTO telegram_copy_inputs VALUES (?,?,?,?,?,?,?,?)',
                           (str(100+i), request, 'chat', 'owner', 'caption', 0, 'now', 'now'))
        def reply(prompt='100', user='owner'):
            return tg._accept_update({'message': {'message_id': 999, 'chat': {'id': 'chat'}, 'from': {'id': user},
                                      'reply_to_message': {'message_id': int(prompt)}, 'text': 'My exact edited wording\n\nSecond paragraph.'}}, Mock(), trusted_source=True)
        yield SimpleNamespace(tg=tg, read=read, reply=reply, telegram=telegram, checkpoints=observed)


@contextmanager
def learning_rows():
    import main
    import telegram_intake as tg
    def row(video, decision, category, text):
        r = ['']*21
        r[2], r[4], r[12], r[13], r[15] = video, 'education', category, text, decision
        return r
    rows = [['header']*21, row('other', 'rejected', 'weak', 'Rejected weak setup.'),
            row('source', 'approved', 'useful', 'Approved complete useful explanation.')]
    service = Mock()
    service.spreadsheets.return_value.values.return_value.get.return_value.execute.return_value = {'values': rows}
    with patch.object(main, 'get_google_services', return_value=(None, None, service)):
        yield tg, service


@contextmanager
def intelligence_db():
    import audio_master_handoff as storage
    with tempfile.TemporaryDirectory() as folder, patch.object(storage, 'DB_PATH', Path(folder)/'isolated.db'):
        yield
