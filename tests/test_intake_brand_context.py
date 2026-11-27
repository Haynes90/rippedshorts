import json
import pytest
from fastapi import BackgroundTasks
from test_editorial_framing_review import review


@pytest.fixture
def intake(review,monkeypatch):
    import schedule_route_gate as gate
    r=review;tg=r['tg'];r['gate']=gate
    monkeypatch.setattr(gate,'_show_config_rows',lambda:[{'show_id':'TCB','show_name':'The Chocolate Botanist','youtube_channel_id':'configured-channel'}])
    monkeypatch.setattr(gate,'_youtube_channel_id',lambda url:'configured-channel')
    monkeypatch.setattr(tg,'send',lambda *a,**k: None)
    def save(request_id,status,state):
        with tg._telegram_db() as db:
            db.execute('UPDATE telegram_requests SET status=?,state_json=? WHERE request_id=?',(status,json.dumps(state),request_id))
    monkeypatch.setattr(tg,'_save',save)
    return r


def update(**extra):
    return {'update_id':'fresh-tcb-source','message':{'chat':{'id':'saved-chat'},'from':{'id':'owner'},'text':'https://youtu.be/Fh8q__wOTDE'},**extra}


def saved(r,request_id):
    with r['tg']._telegram_db() as db:
        return json.loads(db.execute('SELECT state_json FROM telegram_requests WHERE request_id=?',(request_id,)).fetchone()[0])


@pytest.mark.parametrize('explicit',[False,True])
def test_new_routed_intake_reaches_existing_sentence_first_website_dispatch(intake,explicit):
    r=intake;tasks=BackgroundTasks()
    result=r['gate'].gated_accept_update(update(**({'show_id':'TCB'} if explicit else {})),tasks,trusted_source=True)
    assert result['status']=='accepted'
    state=saved(r,result['request_id'])
    assert state['show_id']=='TCB' and state['review_interface']=='website'
    assert state['parsed']['mode']=='both' and state['parsed']['video_id']=='Fh8q__wOTDE'
    from intelligence_workflow import sentence_first
    assert sentence_first(state)
    assert len(tasks.tasks)==1 and tasks.tasks[0].args==(result['request_id'],)
    assert not state.get('website_clip_reviews') and not state.get('schedule_requested_at')
    r['telegram'].assert_not_called()


def test_owner_named_brand_uses_config_without_spoofed_public_context(intake):
    r=intake;payload=update(show_id='AGAPE_CHURCH');payload['message']['text']='TCB: https://youtu.be/Fh8q__wOTDE'
    result=r['gate']._handle_new_youtube(payload,BackgroundTasks())
    state=saved(r,result['request_id'])
    assert state['show_id']=='TCB' and state['route_source']=='owner_message'


def test_invalid_authenticated_route_does_not_create_job_or_processing(intake):
    r=intake;tasks=BackgroundTasks()
    with r['tg']._telegram_db() as db:before=db.execute('SELECT count(*) FROM telegram_requests').fetchone()[0]
    result=r['gate']._handle_new_youtube(update(show_id='UNKNOWN'),tasks,trusted_source=True)
    assert result['status']=='invalid_route' and not tasks.tasks
    with r['tg']._telegram_db() as db:assert db.execute('SELECT count(*) FROM telegram_requests').fetchone()[0]==before


def test_history_does_not_reintroduce_retired_telegram_menus(intake,monkeypatch):
    r=intake;from unittest.mock import Mock
    history=Mock(side_effect=AssertionError('Existing processor owns media/history reuse'))
    monkeypatch.setattr(r['tg'],'_reviewed_short_history_from_sheet',history)
    result=r['gate']._handle_new_youtube(update(show_id='TCB'),BackgroundTasks(),trusted_source=True)
    assert result['status']=='accepted';history.assert_not_called();r['telegram'].assert_not_called()
