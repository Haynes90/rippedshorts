import ast
import json
import logging
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from test_creator_media_workflow import directory


@pytest.fixture
def handoff(directory,monkeypatch):
    import creator_approval as creators
    creators.confirm_batch('r','TCB','video','channel',[],'human',True)
    text='🌱 Exact owner words\n\n#Plants #Plants'
    state=dict(show_id='TCB',vid_title='Source title',parsed=dict(video_id='video'),
        result=dict(segments=[dict(transcript='Complete thought.',duration=30)]),
        candidate_reviews={'0':dict(status='rendered',clip_url='https://example.test/verified.mp4')},
        copy_drafts=[dict(asset_id='r:short:1',asset_type='9:16_SHORT',social_caption=text,
            post_text=text,post_editor_version=1,hashtags='#Plants',copy_status='edited',user_edited=True)])
    class DB:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def execute(self,sql,args):
            if sql.startswith('UPDATE'):state.clear();state.update(json.loads(args[0]))
            return self
        def fetchone(self):return {'state_json':json.dumps(state)}
    receipt={'request_id':'r','asset_group_ids':['r:short:1'],'delivery_ids':['r:short:1:FACEBOOK'],'track_id':'track'}
    provider=Mock(return_value=SimpleNamespace(status_code=200,json=lambda:receipt))
    persist=Mock();notify=Mock()
    env=dict(os=os,json=json,logger=logging.getLogger('isolated-handoff'),_LOCK=threading.RLock(),_telegram_db=DB,
        requests=SimpleNamespace(post=provider),send=notify,now=lambda:'2026-11-23T12:00:00Z',
        _state_vid_title=lambda s:'Source title',_persist_schedule_outbox=persist,
        _caption_learning_checkpoint_map=lambda _:{'r:short:1':{'social_caption':'Stale learning copy','hashtags':'#Stale'}})
    monkeypatch.setenv('SCHEDULE_MASTER_URL','https://isolated.test')
    monkeypatch.setattr(__import__('intelligence_workflow'),'observe',Mock())
    tree=ast.parse((Path(__file__).resolve().parents[1]/'telegram_intake.py').read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_handoff_shorts_to_schedule_master')
    fn.decorator_list=[];exec(compile(ast.Module(body=[fn],type_ignores=[]),'isolated-handoff','exec'),env)
    return SimpleNamespace(run=env[fn.name],state=state,provider=provider,persist=persist,notify=notify,receipt=receipt,text=text)


def test_post_ack_notification_failure_cannot_demote_acceptance(handoff):
    h=handoff;h.notify.side_effect=RuntimeError('Telegram unavailable')
    assert h.run('r','chat') is True
    assert h.state['schedule_master']['shorts_status']=='accepted'
    assert h.state['schedule_master']['shorts_response']==h.receipt
    assert [c.args[1] for c in h.persist.call_args_list]==['READY','ACCEPTED']
    h.notify.side_effect=None
    assert h.run('r','chat') is False
    assert h.provider.call_count==1


def test_saved_complete_post_overrides_stale_learning_at_actual_handoff(handoff):
    h=handoff;assert h.run('r','chat') is True
    asset=h.provider.call_args.kwargs['json']['assets'][0]
    assert asset['social_caption']==h.text and asset['hashtags']==''
    assert h.state['copy_drafts'][0]['post_text']==h.text
    assert h.provider.call_count==1


def test_outbox_ack_projection_failure_preserves_local_receipt(handoff):
    h=handoff
    def write(payload,status):
        if status=='ACCEPTED':raise RuntimeError('Sheets response interrupted')
    h.persist.side_effect=write
    assert h.run('r','chat') is True
    assert h.state['schedule_master']['shorts_status']=='accepted'
    assert h.state['schedule_master']['shorts_response']==h.receipt
    assert h.run('r','chat') is False
    assert h.provider.call_count==1


def test_uncertain_provider_result_preserves_approval_and_retry_identity(handoff):
    h=handoff;h.provider.side_effect=TimeoutError('No response')
    assert h.run('r','chat') is not True
    assert h.state['copy_drafts'][0]['post_text']==h.text
    assert h.state['schedule_master']['shorts_status']=='failed'
    assert not h.state['schedule_master'].get('shorts_response')
    payload=h.persist.call_args_list[0].args[0]
    assert h.persist.call_args_list[-1].args==(payload,'RETRY')
    assert h.provider.call_count==1  # no automatic re-POST inside this attempt
