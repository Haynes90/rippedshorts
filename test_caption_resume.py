import ast
import json
import logging
import sqlite3
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock


SOURCE = Path(__file__).with_name('telegram_intake.py').read_text(encoding='utf-8')
TREE = ast.parse(SOURCE)


class CaptionResumeTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.db.row_factory = sqlite3.Row
        self.db.execute('CREATE TABLE telegram_requests (request_id, chat_id, user_id, status, state_json, updated_at)')
        self.state = {
            'stage': 'render_recovery', 'copy_review_requested_at': 'earlier',
            'copy_drafts': [{'social_caption': 'My edited caption'}, {'social_caption': 'Second'}],
            'copy_review_index': 1, 'copy_review_message_id': 583,
            'awaiting_copy_input': {'index': 1, 'field': 'social_caption'},
        }
        self.db.execute('INSERT INTO telegram_requests VALUES (?,?,?,?,?,?)',
                        ('job', 'chat', 'user', 'render_recovery', json.dumps(self.state), 'now'))
        self.db.commit()
        self.api = Mock(return_value={'result': {'message_id': 999}})
        self.ns = dict(json=json, _LOCK=threading.RLock(), _telegram_db=lambda: self.db,
                       _TERMINAL_JOB_STAGES={'scheduled', 'superseded', 'cancelled'},
                       telegram=self.api, logger=logging.getLogger('test'), Any=object,
                       durable_job=lambda *args, **kwargs: (lambda fn: fn),
                       _copy_review_payload=Mock(return_value=('saved draft', {})), now=lambda: 'later')
        names = {'_refresh_copy_review', '_resume_saved_copy_review', '_process'}
        exec(compile(ast.Module(body=[n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]), '<production>', 'exec'), self.ns)

    def tearDown(self):
        self.db.close()

    def row(self):
        return self.db.execute('SELECT * FROM telegram_requests').fetchone()

    def test_saved_cursor_and_edits_survive_resume(self):
        self.assertTrue(self.ns['_resume_saved_copy_review'](self.row()))
        self.ns['_copy_review_payload'].assert_called_once_with('job', self.state['copy_drafts'], 1)
        restored = json.loads(self.row()['state_json'])
        self.assertEqual(restored['copy_drafts'], self.state['copy_drafts'])
        self.assertEqual(restored['awaiting_copy_input'], self.state['awaiting_copy_input'])
        self.assertEqual(self.api.call_args.args[0], 'editMessageText')

    def test_deleted_card_is_replaced_and_persisted(self):
        self.api.side_effect = [RuntimeError('message to edit not found'), {'result': {'message_id': 999}}]
        self.ns['_resume_saved_copy_review'](self.row())
        self.assertEqual([c.args[0] for c in self.api.call_args_list], ['editMessageText', 'sendMessage'])
        self.assertEqual(json.loads(self.row()['state_json'])['copy_review_message_id'], 999)

    def test_finished_or_superseded_copy_is_not_reopened(self):
        for field in ['schedule_requested_at', 'copy_review_completed_at', 'superseded_by_request_id']:
            row = dict(self.row())
            row['state_json'] = json.dumps({**self.state, field: 'set'})
            self.assertFalse(self.ns['_resume_saved_copy_review'](row))
        self.api.assert_not_called()

    def test_process_resumes_copy_without_source_processing(self):
        self.ns.update(os=__import__('os'), _ensure_storage_headroom=lambda: {'free_bytes': 10**12})
        self.ns['_process']('job')
        self.api.assert_called_once()
        self.assertEqual(self.row()['status'], 'render_recovery')

    def test_resume_command_prefers_local_copy_to_stale_ledger(self):
        branch = next(n for n in ast.walk(TREE) if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == 'resume_match')
        wrapper = ast.parse('def resume():\n    pass').body[0]
        wrapper.body = branch.body
        self.ns.update(chat_id='chat', user_id='user', latest_incomplete=Mock(side_effect=AssertionError('must not read stale ledger')))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), '<resume-command>', 'exec'), self.ns)
        self.assertEqual(self.ns['resume'](), {'status': 'copy_review_resumed', 'request_id': 'job'})
        self.ns['latest_incomplete'].assert_not_called()


if __name__ == '__main__':
    unittest.main()

