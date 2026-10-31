import asyncio
import json
import sqlite3
from threading import Event
from unittest.mock import Mock

import pytest

import intelligence_workflow as workflow
import production_intelligence as pi


@pytest.fixture
def integrated(tmp_path,monkeypatch):
    import main
    import telegram_intake as tg
    import audio_master_handoff as storage
    monkeypatch.setattr(storage,'DB_PATH',tmp_path/'workflow.db')
    monkeypatch.setattr(pi,'model_topic_map',lambda _: {'topics':[{'key':'one','title':'Complete discussion','first_segment':0,'last_segment':1,'tags':['gardening']}]})
    monkeypatch.setattr(tg,'validate_complete_candidates',lambda result,_:result)
    monkeypatch.setattr(tg,'_preflight_candidates',lambda _,clips:clips)
    monkeypatch.setattr(tg,'_build_contiguous_topic_segments',lambda _,clips,**kw:clips)
    timeline=[{'start':0.,'end':30.,'duration':30.,'text':'A whole thought.'},
              {'start':30.,'end':240.,'duration':210.,'text':'A substantial complete discussion.'}]
    state={'show_id':'TCB','parsed':{'video_id':'source'},'vid_title':'Test source'}
    return main,tg,state,timeline


def test_actual_selector_orchestration_is_parallel_but_does_not_select_for_owner(integrated,monkeypatch):
    main,tg,state,timeline=integrated
    short_started,long_started=Event(),Event()
    def short(*args):
        short_started.set();assert long_started.wait(2)
        return {'segments':[{'start':0.,'end':30.,'score':80,'category':'education','transcript':'A whole thought.'}]}
    def long(*args,**kwargs):
        long_started.set();assert short_started.wait(2)
        assert 'SHARED FULL-SOURCE TOPIC MAP' in kwargs['intelligence_context']
        return [{'start':0.,'end':240.,'score':85,'highlight_type':'discussion'}]
    short_mock=Mock(side_effect=short);long_mock=Mock(side_effect=long)
    monkeypatch.setattr(main,'call_openai_for_clips',short_mock)
    monkeypatch.setattr(tg,'_topic_break_suggestions',long_mock)
    future,mapping=workflow.start(state,timeline,'retained.mp4','existing rubric')
    short_result=future.result(timeout=4)
    prepared=workflow.prepared_highlights(state,timeline,'retained.mp4')
    assert len(short_result['segments'])==len(prepared['segments'])==1
    assert short_result['segments'][0]['map_id']==prepared['segments'][0]['map_id']==mapping['map_id']
    assert 'candidate_reviews' not in state and 'topic_reviews' not in state
    # New owner feedback after Shorts review must not repeat the completed long pass.
    pi.feedback_snapshot('prior',{'show_id':'TCB','parsed':{'video_id':'source'},'result':short_result,'candidate_reviews':{'0':{'status':'approved'}}})
    assert workflow.prepared_highlights(state,timeline,'retained.mp4')==prepared
    assert short_mock.call_count==long_mock.call_count==1


def test_short_failure_preserves_independently_completed_highlights(integrated,monkeypatch):
    main,tg,state,timeline=integrated
    monkeypatch.setattr(main,'call_openai_for_clips',Mock(side_effect=ValueError('short response rejected')))
    monkeypatch.setattr(tg,'_topic_break_suggestions',lambda *a,**kw:[{'start':0.,'end':240.,'score':80}])
    future,mapping=workflow.start(state,timeline,'retained.mp4','rubric')
    with pytest.raises(ValueError):future.result(timeout=4)
    assert workflow.prepared_highlights(state,timeline,'retained.mp4')['segments']
    status=pi.analysis_status(mapping['map_id'])
    assert any(r['status']=='FAILED' and ':9:16:' in r['analysis_id'] for r in status)
    assert any(r['status']=='PASSED' and ':16:9:' in r['analysis_id'] for r in status)


def test_failure_retry_is_explicit_bounded_and_keeps_completed_sibling(integrated):
    _,_,state,timeline=integrated
    mapping=workflow.prepare(state,timeline)
    key=mapping['map_id']+':16:9:test'
    for attempt in range(3):
        with pytest.raises(ValueError):pi.cached_analysis(key,lambda:(_ for _ in ()).throw(ValueError('bad response')))
        if attempt<2:pi.retry_failed(mapping['map_id'])
    with pytest.raises(ValueError,match='budget'):pi.retry_failed(mapping['map_id'])
    assert pi.cached_analysis(mapping['map_id'],Mock(side_effect=AssertionError('topic map must not repeat')))


def test_status_read_does_not_modify_database(integrated):
    import audio_master_handoff as storage
    _,_,state,timeline=integrated
    mapping=workflow.prepare(state,timeline)
    before=storage.DB_PATH.read_bytes()
    assert pi.analysis_status(mapping['map_id'])
    assert storage.DB_PATH.read_bytes()==before


def test_website_feedback_saves_hold_and_favorite_without_replacing_copy(tmp_path,monkeypatch):
    import caption_review_web as web
    import telegram_intake as tg
    import audio_master_handoff as storage
    db_path=tmp_path/'review.db'
    monkeypatch.setattr(storage,'DB_PATH',db_path)
    state={'show_id':'TCB','parsed':{'video_id':'video'},'copy_drafts':[{'asset_id':'request:short:1','asset_type':'9:16_SHORT','social_caption':'Owner exact wording','hashtags':'#Plants','copy_status':'pending'}]}
    def connect():
        db=sqlite3.connect(db_path);db.row_factory=sqlite3.Row;return db
    with connect() as db:
        db.execute('CREATE TABLE telegram_requests(request_id TEXT PRIMARY KEY,state_json TEXT,user_id TEXT,updated_at TEXT,chat_id TEXT)')
        db.execute('INSERT INTO telegram_requests VALUES(?,?,?,?,?)',('request',json.dumps(state),'owner','now','chat'))
    monkeypatch.setattr(web,'connect',connect)
    monkeypatch.setattr(web,'_session',lambda _: {'request_id':'request'})
    monkeypatch.setattr(web,'_request_row',lambda _: {'state_json':json.dumps(state),'user_id':'owner','chat_id':'chat'})
    monkeypatch.setattr(web,'_payload_for_session',lambda _: {'completed':1,'regenerated':0})
    monkeypatch.setattr(tg,'_checkpoint_copy_edit',Mock())
    monkeypatch.setattr(tg,'_refresh_copy_review',Mock())
    class Request:
        async def json(self):return {'social_caption':'Owner exact wording','favorite':True,'editorial_hold':True,'selection_feedback':'More useful seed-saving explanations.'}
    # Pure handler test: Request.json resolves immediately; no event loop/socket is needed.
    operation=web.save_review_asset('token',0,Request())
    with pytest.raises(StopIteration) as completed:
        operation.send(None)
    assert completed.value.value['status']=='saved'
    with connect() as db:
        saved=json.loads(db.execute('SELECT state_json FROM telegram_requests').fetchone()[0])
    draft=saved['copy_drafts'][0]
    assert draft['social_caption']=='Owner exact wording' and draft['editorial_hold'] and draft['favorite']
    assert draft['selection_feedback']=='More useful seed-saving explanations.'
