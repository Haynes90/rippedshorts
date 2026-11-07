from pathlib import Path
from test_request_scoped_render_recovery import batch

INTAKE=(Path(__file__).resolve().parents[1]/"telegram_intake.py").read_text(encoding="utf-8")
MAIN=(Path(__file__).resolve().parents[1]/"main.py").read_text(encoding="utf-8")


def test_missing_render_action_uses_saved_request_selection():
    assert "def _queue_missing_approved_renders(" in INTAKE
    assert '"♻️ Retry Missing Renders"' in INTAKE
    assert 'r"/render-missing' in INTAKE


def test_missing_render_recovery_retains_exact_approved_request_segments(batch):
    before=batch['read']()
    batch['tg']._queue_missing_approved_renders('current','chat')
    assert batch['read']()==before
    batch['tg']._approved_clip_history_from_sheet.assert_not_called()


def test_google_service_account_identity_is_logged():
    assert "def google_credential_identity()" in MAIN
    assert "GOOGLE_SERVICE_ACCOUNT source=%s client_email=%s" in MAIN
    assert "GOOGLE_SERVICE_ACCOUNT_MISMATCH" in MAIN
    assert "because GOOGLE_CREDENTIALS takes precedence" in MAIN


def test_old_cards_cannot_redirect_retry_to_another_request(batch):
    other=batch['read']('older')
    result=batch['tg']._queue_missing_approved_renders('current','chat')
    assert result['request_id']=='current' and batch['read']('older')==other
    batch['tg']._active_request_for_video.assert_not_called()


def test_render_recovery_checkpoints_selection_without_replacing_request(batch):
    import render_recovery as recovery
    before=batch['read']()
    batch['tg']._queue_missing_approved_renders('current','chat')
    with recovery.evidence.database() as db:
        kinds={row['kind'] for row in db.execute('SELECT kind FROM intelligence_records')}
    assert {'selected_batch','render_success','render_hold'} <= kinds
    assert batch['read']()==before


def test_watchdog_recovers_stuck_failed_approved_renders_once():
    assert "has_failed_approved_render" in INTAKE
    assert "RIPPED_JOB_WATCHDOG_RENDER_RECOVERY" in INTAKE
    assert "_queue_missing_approved_renders" in INTAKE


def test_individual_failure_does_not_block_successful_handoff():
    # Owner explicitly authorized independent successful clips to progress, not all-or-nothing batches.
    from acceptance_fixtures import copy_review_fixture
    from unittest.mock import Mock,patch
    import json
    with copy_review_fixture() as f:
        state=f.read();state.update({'candidate_reviews':{'0':{'status':'rendered'},'1':{'status':'render_failed','render_error':'individual encoding error'}},'schedule_requested_at':'now'})
        with f.tg._telegram_db() as db:db.execute('UPDATE telegram_requests SET state_json=? WHERE request_id=?',(json.dumps(state),'first'))
        executor=Mock()
        with patch.object(f.tg,'RENDER_EXECUTOR',executor),patch.object(f.tg,'_save'),patch.object(f.tg,'_refresh_copy_review'):
            f.tg._notify_render_queue_complete('first','chat')
        assert executor.submit.call_count==1
        assert executor.submit.call_args.args==(f.tg._handoff_shorts_to_schedule_master,'first','chat')
        saved=f.read();assert saved['candidate_reviews']['1']['status']=='render_failed'
        assert saved['rendered_count']==1 and saved['render_failed_count']==1
